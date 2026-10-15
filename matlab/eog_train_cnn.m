function R = eog_train_cnn(DS, varargin)
%EOG_TRAIN_CNN  Train a 1-D CNN to classify EOG windows into six classes.
%
%   R = eog_train_cnn(DS) trains on the windows from eog_build_dataset and
%   returns the trained network alongside its scores.
%
%   Splitting is by subject, never at random. Windows from one saccade overlap
%   heavily and the forward and return halves of a trial are the same movement
%   reversed, so a random split would leave near-copies on both sides and
%   report an accuracy that says nothing about a new wearer.
%
%   Name-value options:
%     'TestSubject'   subject held out for the final score   (default last)
%     'ValSubject'    subject held out for early stopping    (default second last)
%     'MaxEpochs'     (default 60)
%     'MiniBatchSize' (default 64)
%     'LearnRate'     (default 1e-3)
%     'Balance'       oversample the training set to even the
%                     classes out, since blink has 5.5x the
%                     windows of up or down             (default true)
%     'Seed'          (default 0)
%     'Save'          .mat file for the trained network      (default none)

%% ------------------------------------------------------------------ options
subs = categories(DS.subject);
p = inputParser;
p.addParameter('TestSubject', subs{end}, @(s) ischar(s) || isstring(s));
p.addParameter('ValSubject', subs{max(1, end-1)}, @(s) ischar(s) || isstring(s));
p.addParameter('MaxEpochs', 60, @isnumeric);
p.addParameter('MiniBatchSize', 64, @isnumeric);
p.addParameter('LearnRate', 1e-3, @isnumeric);
p.addParameter('Balance', true, @islogical);
p.addParameter('Seed', 0, @isnumeric);
p.addParameter('Save', '', @(s) ischar(s) || isstring(s));
p.parse(varargin{:});
opt = p.Results;

rng(opt.Seed);

testSub = char(opt.TestSubject);
valSub  = char(opt.ValSubject);
if strcmp(testSub, valSub)
    error('eog_train_cnn:sameSubject', ...
          'TestSubject and ValSubject must differ; both are %s', testSub);
end

%% ------------------------------------------------------------------- split
isTest = DS.subject == testSub;
isVal  = DS.subject == valSub;
isTrain = ~isTest & ~isVal;

XTrain = DS.X(isTrain); YTrain = DS.Y(isTrain);
XVal   = DS.X(isVal);   YVal   = DS.Y(isVal);
XTest  = DS.X(isTest);  YTest  = DS.Y(isTest);

fprintf('Split by subject\n');
fprintf('  train : %-28s %5d windows\n', ...
        strjoin(setdiff(subs, {testSub, valSub})', ' '), numel(XTrain));
fprintf('  val   : %-28s %5d windows\n', valSub, numel(XVal));
fprintf('  test  : %-28s %5d windows\n\n', testSub, numel(XTest));

%% --------------------------------------------------- balance the training set
if opt.Balance
    [XTrain, YTrain] = oversample(XTrain, YTrain, DS.classes);
    fprintf('  balanced training set to %d windows\n\n', numel(XTrain));
end

%% ------------------------------------------------------------------ network
numChannels = size(DS.X{1}, 2);   % windows are time-by-channel
numClasses  = numel(DS.classes);

layers = [
    sequenceInputLayer(numChannels, MinLength=DS.window, Name="in")

    convolution1dLayer(7, 32, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, Stride=2)

    convolution1dLayer(5, 64, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, Stride=2)

    convolution1dLayer(3, 128, Padding="same")
    batchNormalizationLayer
    reluLayer

    globalAveragePooling1dLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(numClasses)
    softmaxLayer];

options = trainingOptions("adam", ...
    MaxEpochs=opt.MaxEpochs, ...
    MiniBatchSize=opt.MiniBatchSize, ...
    InitialLearnRate=opt.LearnRate, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=25, ...
    LearnRateDropFactor=0.2, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    ValidationFrequency=max(1, floor(numel(XTrain) / opt.MiniBatchSize)), ...
    ValidationPatience=12, ...
    OutputNetwork="best-validation", ...
    Verbose=true, ...
    VerboseFrequency=50, ...
    Plots="none", ...
    ExecutionEnvironment="auto");

fprintf('Training...\n');
net = trainnet(XTrain, YTrain, layers, "crossentropy", options);

%% ----------------------------------------------------------------- evaluate
R.net = net;
R.classes = DS.classes;
R.testSubject = testSub;
R.valSubject = valSub;

[R.valAccuracy,  R.valConfusion]  = scoreSet(net, XVal,  YVal,  DS.classes);
[R.testAccuracy, R.testConfusion] = scoreSet(net, XTest, YTest, DS.classes);

fprintf('\nValidation (%s): %.1f%%\n', valSub, 100 * R.valAccuracy);
fprintf('Test       (%s): %.1f%%\n\n', testSub, 100 * R.testAccuracy);

printConfusion(R.testConfusion, DS.classes, testSub);

if ~isempty(char(opt.Save))
    save(char(opt.Save), 'R', '-v7.3');
    fprintf('\n  saved %s\n', char(opt.Save));
end
end


%% ======================================================================
function [X, Y] = oversample(X, Y, classes)
%OVERSAMPLE  Draw with replacement from each class up to the largest one.
counts = countcats(removecats(Y));
target = max(counts);
cats = categories(removecats(Y));

keepX = {};
keepY = {};
for i = 1:numel(cats)
    idx = find(Y == cats{i});
    if isempty(idx)
        continue
    end
    pick = idx(randi(numel(idx), target, 1));
    keepX = [keepX, X(pick)];            %#ok<AGROW>
    keepY = [keepY; Y(pick)];            %#ok<AGROW>
end

order = randperm(numel(keepX));
X = keepX(order);
Y = categorical(keepY(order), classes);
end


%% ======================================================================
function [acc, C] = scoreSet(net, X, Y, classes)
%SCORESET  Accuracy and confusion counts over one split.
if isempty(X)
    acc = NaN;
    C = zeros(numel(classes));
    return
end
scores = minibatchpredict(net, X);
pred = scores2label(scores, classes);
pred = categorical(pred, classes);
acc = mean(pred == Y);
C = confusionmat(Y, pred, 'Order', categorical(classes, classes));
end


%% ======================================================================
function printConfusion(C, classes, name)
%PRINTCONFUSION  Rows are the true class, columns the prediction.
fprintf('Confusion on %s (rows true, columns predicted)\n', name);
fprintf('  %-7s', '');
fprintf('%7s', classes{:});
fprintf('%9s\n', 'recall');
for i = 1:numel(classes)
    fprintf('  %-7s', classes{i});
    fprintf('%7d', C(i, :));
    total = sum(C(i, :));
    if total > 0
        fprintf('%8.1f%%\n', 100 * C(i, i) / total);
    else
        fprintf('%9s\n', '-');
    end
end
end
