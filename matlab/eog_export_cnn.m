function eog_export_cnn(R, DS, varargin)
%EOG_EXPORT_CNN  Write a trained network's weights out as a C header.
%
%   eog_export_cnn(R, DS) writes src/eog_cnn_weights.h for src/eog_cnn.c,
%   and test vectors for checking the C inference against MATLAB's.
%
%   The C side is a fixed sequence, so this checks the network matches it and
%   refuses otherwise rather than emitting weights the C code would misread:
%
%     input -> [conv -> batchnorm -> relu -> maxpool] x 2
%           -> conv -> batchnorm -> relu
%           -> global average pool -> dropout -> fully connected -> softmax
%
%   Two transformations happen on the way out, neither of which changes the
%   result:
%
%     Batch normalization is folded into the convolution before it. At
%     inference it is a fixed per-channel scale and shift, so it merges into
%     the convolution's weights and bias, and the C code never sees it.
%
%     Dropout is dropped. It only acts during training.
%
%   Name-value options:
%     'Header'      weights header      (default src/eog_cnn_weights.h)
%     'Vectors'     test vectors header (default test/eog_cnn_vectors.h)
%     'PerClass'    windows per class embedded in the vectors   (default 3)
%     'CheckFile'   binary of every window with MATLAB's output, for checking
%                   the C side across the whole set  (default data/eog_cnn_check.bin)

p = inputParser;
p.addParameter('Header', fullfile('src', 'eog_cnn_weights.h'));
p.addParameter('Vectors', fullfile('test', 'eog_cnn_vectors.h'));
p.addParameter('PerClass', 3, @isnumeric);
p.addParameter('CheckFile', fullfile('data', 'eog_cnn_check.bin'));
p.parse(varargin{:});
opt = p.Results;

net = R.net;
L = net.Layers;

%% ------------------------------------------------------------ check shape
expected = {'SequenceInputLayer', ...
            'Convolution1DLayer', 'BatchNormalizationLayer', 'ReLULayer', 'MaxPooling1DLayer', ...
            'Convolution1DLayer', 'BatchNormalizationLayer', 'ReLULayer', 'MaxPooling1DLayer', ...
            'Convolution1DLayer', 'BatchNormalizationLayer', 'ReLULayer', ...
            'GlobalAveragePooling1DLayer', 'DropoutLayer', ...
            'FullyConnectedLayer', 'SoftmaxLayer'};
got = arrayfun(@(l) regexprep(class(l), '.*\.', ''), L, 'UniformOutput', false);
if numel(got) ~= numel(expected) || ~all(strcmp(got(:), expected(:)))
    error('eog_export_cnn:shape', ...
          ['Network does not match the layer sequence src/eog_cnn.c implements.\n' ...
           'Expected: %s\nGot:      %s'], ...
          strjoin(expected, ' '), strjoin(got, ' '));
end
if ~strcmp(L(1).Normalization, 'none')
    error('eog_export_cnn:norm', ...
          'Input normalization "%s" is not implemented on the C side.', ...
          L(1).Normalization);
end
for i = [5 9]
    if L(i).PoolSize ~= 2 || L(i).Stride ~= 2 || any(L(i).PaddingSize ~= 0)
        error('eog_export_cnn:pool', 'Pooling must be size 2, stride 2, unpadded.');
    end
end

convIdx = [2 6 10];
inLen = DS.window;
inCh = L(1).InputSize;

%% --------------------------------------------- fold batch norm into conv
conv = struct('K', {}, 'Cin', {}, 'Cout', {}, 'pad', {}, 'w', {}, 'b', {});
for j = 1:3
    c = L(convIdx(j));
    bn = L(convIdx(j) + 1);
    if c.Stride ~= 1 || c.DilationFactor ~= 1 || c.PaddingSize(1) ~= c.PaddingSize(2)
        error('eog_export_cnn:conv', ...
              'Conv %d needs stride 1, dilation 1 and symmetric padding.', j);
    end

    W = double(c.Weights);              % K x Cin x Cout
    b = double(c.Bias(:));              % Cout
    s = double(bn.Scale(:)) ./ sqrt(double(bn.TrainedVariance(:)) + bn.Epsilon);

    Wf = W .* reshape(s, 1, 1, []);
    bf = (b - double(bn.TrainedMean(:))) .* s + double(bn.Offset(:));

    % C reads w[co][k][ci], ci fastest. MATLAB stores K x Cin x Cout,
    % column-major, so swapping the first two dimensions lines that up.
    w = permute(Wf, [2 1 3]);

    conv(j).K = size(W, 1);
    conv(j).Cin = size(W, 2);
    conv(j).Cout = size(W, 3);
    conv(j).pad = c.PaddingSize(1);
    conv(j).w = single(w(:));
    conv(j).b = single(bf);
end

fc = L(15);
Wfc = double(fc.Weights);               % classes x features
fcW = Wfc.';                            % C reads w[o][i], i fastest
fcW = single(fcW(:));
fcB = single(double(fc.Bias(:)));
numClasses = size(Wfc, 1);
numFeatures = size(Wfc, 2);

%% ---------------------------------------------------------- weights header
if isfield(R, 'trainSubjects')
    provenance = strjoin(R.trainSubjects, ' ');
else
    provenance = 'unknown';
end
params = sum(cellfun(@numel, {conv.w})) + sum(cellfun(@numel, {conv.b})) + ...
         numel(fcW) + numel(fcB);

fid = fopen(opt.Header, 'w');
closer = onCleanup(@() fclose(fid));

fprintf(fid, '/**\n * @file eog_cnn_weights.h\n');
fprintf(fid, ' * @brief Trained weights for src/eog_cnn.c. Generated: do not edit.\n *\n');
fprintf(fid, ' * Regenerate from MATLAB with matlab/eog_export_cnn.m.\n *\n');
fprintf(fid, ' * Trained on: %s\n', provenance);
fprintf(fid, ' * Generated:  %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));
fprintf(fid, ' * Parameters: %d, batch normalization folded into the convolutions\n */\n\n', params);
fprintf(fid, '#ifndef EOG_CNN_WEIGHTS_H\n#define EOG_CNN_WEIGHTS_H\n\n');

fprintf(fid, '#define EOG_CNN_INPUT_LEN   %d\n', inLen);
fprintf(fid, '#define EOG_CNN_INPUT_CH    %d\n', inCh);
fprintf(fid, '#define EOG_CNN_NUM_CLASSES %d\n\n', numClasses);

for j = 1:3
    fprintf(fid, '#define EOG_CNN_C%d_K    %d\n', j, conv(j).K);
    fprintf(fid, '#define EOG_CNN_C%d_COUT %d\n', j, conv(j).Cout);
    fprintf(fid, '#define EOG_CNN_C%d_PAD  %d\n\n', j, conv(j).pad);
end
fprintf(fid, '#define EOG_CNN_FC_IN    %d\n\n', numFeatures);

fprintf(fid, ['/* The arrays are defined only where EOG_CNN_EMIT_WEIGHTS is set, which\n' ...
              ' * is src/eog_cnn.c. Anything else including this gets the sizes above\n' ...
              ' * without its own copy of the weights. */\n']);
fprintf(fid, '#ifdef EOG_CNN_EMIT_WEIGHTS\n\n');

fprintf(fid, 'static const char *const eog_cnn_class_names[EOG_CNN_NUM_CLASSES] = {\n    ');
fprintf(fid, '"%s", ', DS.classes{1:end-1});
fprintf(fid, '"%s"\n};\n\n', DS.classes{end});

for j = 1:3
    writeArray(fid, sprintf('eog_cnn_c%d_w', j), conv(j).w, ...
               sprintf('[cout][k][cin], %d x %d x %d', conv(j).Cout, conv(j).K, conv(j).Cin));
    writeArray(fid, sprintf('eog_cnn_c%d_b', j), conv(j).b, ...
               sprintf('[cout], %d', conv(j).Cout));
end
writeArray(fid, 'eog_cnn_fc_w', fcW, sprintf('[out][in], %d x %d', numClasses, numFeatures));
writeArray(fid, 'eog_cnn_fc_b', fcB, sprintf('[out], %d', numClasses));

fprintf(fid, '#endif /* EOG_CNN_EMIT_WEIGHTS */\n\n');
fprintf(fid, '#endif /* EOG_CNN_WEIGHTS_H */\n');
clear closer
fprintf('wrote %s (%d parameters)\n', opt.Header, params);

%% ---------------------------------------------------- MATLAB's own outputs
allP = minibatchpredict(net, DS.X);
if size(allP, 1) ~= numel(DS.X)
    allP = allP.';
end
allY = double(DS.Y);                    % 1-based class index

%% ------------------------------------------------------ embedded vectors
sel = [];
for k = 1:numClasses
    idx = find(allY == k);
    take = idx(round(linspace(1, numel(idx), min(opt.PerClass, numel(idx)))));
    sel = [sel; take(:)];               %#ok<AGROW>
end
n = numel(sel);

fid = fopen(opt.Vectors, 'w');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '/**\n * @file eog_cnn_vectors.h\n');
fprintf(fid, ' * @brief Inputs and MATLAB outputs for checking src/eog_cnn.c. Generated.\n *\n');
fprintf(fid, ' * %d windows, %d per class, with the class probabilities MATLAB''s\n', n, opt.PerClass);
fprintf(fid, ' * minibatchpredict gave for each. Regenerate with matlab/eog_export_cnn.m\n');
fprintf(fid, ' * whenever eog_cnn_weights.h is regenerated, since they must match.\n */\n\n');
fprintf(fid, '#ifndef EOG_CNN_VECTORS_H\n#define EOG_CNN_VECTORS_H\n\n');
fprintf(fid, '#define EOG_CNN_NUM_VECTORS %d\n\n', n);

inputs = zeros(EOGlen(inLen, inCh), n, 'single');
for i = 1:n
    xt = single(DS.X{sel(i)}).';        % time-major, channel fastest
    inputs(:, i) = xt(:);
end
writeArray(fid, 'eog_cnn_vec_input', inputs(:), ...
           sprintf('[vector][t][ch], %d x %d x %d', n, inLen, inCh));
probs = single(allP(sel, :)).';
writeArray(fid, 'eog_cnn_vec_prob', probs(:), ...
           sprintf('[vector][class], %d x %d', n, numClasses));
fprintf(fid, 'static const unsigned char eog_cnn_vec_label[%d] = {\n    ', n);
fprintf(fid, '%d, ', allY(sel(1:end-1)) - 1);
fprintf(fid, '%d\n};\n\n', allY(sel(end)) - 1);
fprintf(fid, '#endif /* EOG_CNN_VECTORS_H */\n');
clear closer
fprintf('wrote %s (%d vectors)\n', opt.Vectors, n);

%% ------------------------------------------------------ full check file
N = numel(DS.X);
fid = fopen(opt.CheckFile, 'wb');
closer = onCleanup(@() fclose(fid));
fwrite(fid, 'EOGCNNCK', 'char');
fwrite(fid, [N inLen inCh numClasses], 'uint32');
for i = 1:N
    xt = single(DS.X{i}).';
    fwrite(fid, xt(:), 'float32');
end
fwrite(fid, single(allP).', 'float32');
fwrite(fid, allY - 1, 'uint8');
clear closer
fprintf('wrote %s (%d windows)\n', opt.CheckFile, N);
end


%% ======================================================================
function n = EOGlen(len, ch)
n = len * ch;
end


%% ======================================================================
function writeArray(fid, name, v, shape)
%WRITEARRAY  Emit a const float array, eight to a line.
%
%   Nine significant digits is enough for any float32 to round-trip exactly,
%   so the C side gets the same weights MATLAB holds, bit for bit. The #
%   flag keeps the decimal point on whole numbers: without it 1 prints as
%   "1", and "1f" is not a C literal.
v = single(v(:));
if any(~isfinite(v))
    error('eog_export_cnn:nonfinite', '%s contains NaN or Inf', name);
end
fprintf(fid, '/* %s */\n', shape);
fprintf(fid, 'static const float %s[%d] = {\n', name, numel(v));
for i = 1:8:numel(v)
    chunk = v(i:min(i + 7, numel(v)));
    fprintf(fid, '    ');
    fprintf(fid, '%#.9gf, ', chunk);
    fprintf(fid, '\n');
end
fprintf(fid, '};\n\n');
end
