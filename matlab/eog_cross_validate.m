function CV = eog_cross_validate(DS, varargin)
%EOG_CROSS_VALIDATE  Leave-one-subject-out over the whole set.
%
%   With six recordings, a single held-out subject gives a score that depends
%   heavily on which one it was. This trains once per subject, each time
%   holding that subject out for test and the next one round for validation,
%   and reports the spread as well as the mean.
%
%   CV = eog_cross_validate(DS) runs all folds.
%   CV = eog_cross_validate(DS, 'Save', 'data/eog_cv.mat') keeps the result.
%
%   Accepts and forwards any eog_train_cnn option except TestSubject and
%   ValSubject, which it sets itself.

p = inputParser;
p.KeepUnmatched = true;
p.addParameter('Save', '', @(s) ischar(s) || isstring(s));
p.parse(varargin{:});
opt = p.Results;

forward = namedargs2cell(p.Unmatched);

subs = categories(DS.subject);
n = numel(subs);
classes = DS.classes;

acc = zeros(n, 1);
conf = zeros(numel(classes), numel(classes));
recall = nan(n, numel(classes));

for k = 1:n
    testSub = subs{k};
    valSub  = subs{mod(k, n) + 1};        % the next subject round the ring

    fprintf('\n================ fold %d of %d: test %s, val %s ================\n', ...
            k, n, testSub, valSub);

    R = eog_train_cnn(DS, 'TestSubject', testSub, 'ValSubject', valSub, ...
                      forward{:});

    acc(k) = R.testAccuracy;
    conf = conf + R.testConfusion;
    rowTotals = sum(R.testConfusion, 2);
    for c = 1:numel(classes)
        if rowTotals(c) > 0
            recall(k, c) = R.testConfusion(c, c) / rowTotals(c);
        end
    end
end

CV.subjects = subs;
CV.accuracy = acc;
CV.recall = recall;
CV.confusion = conf;
CV.classes = classes;

%% ------------------------------------------------------------------ report
fprintf('\n\n================ leave-one-subject-out ================\n\n');
fprintf('  %-8s %10s\n', 'held out', 'accuracy');
fprintf('  %s\n', repmat('-', 1, 20));
for k = 1:n
    fprintf('  %-8s %9.1f%%\n', subs{k}, 100 * acc(k));
end
fprintf('  %s\n', repmat('-', 1, 20));
fprintf('  %-8s %9.1f%%  (sd %.1f, worst %.1f)\n\n', 'mean', ...
        100 * mean(acc), 100 * std(acc), 100 * min(acc));

fprintf('  recall per class, across folds\n');
fprintf('    %-7s %9s %9s %9s\n', 'class', 'mean', 'worst', 'best');
fprintf('    %s\n', repmat('-', 1, 38));
for c = 1:numel(classes)
    r = recall(:, c);
    r = r(~isnan(r));
    fprintf('    %-7s %8.1f%% %8.1f%% %8.1f%%\n', classes{c}, ...
            100 * mean(r), 100 * min(r), 100 * max(r));
end

fprintf('\n  pooled confusion (rows true, columns predicted)\n');
fprintf('    %-7s', '');
fprintf('%7s', classes{:});
fprintf('%9s\n', 'recall');
for i = 1:numel(classes)
    fprintf('    %-7s', classes{i});
    fprintf('%7d', conf(i, :));
    t = sum(conf(i, :));
    if t > 0
        fprintf('%8.1f%%\n', 100 * conf(i, i) / t);
    else
        fprintf('%9s\n', '-');
    end
end

if ~isempty(char(opt.Save))
    save(char(opt.Save), 'CV', '-v7.3');
    fprintf('\n  saved %s\n', char(opt.Save));
end
end
