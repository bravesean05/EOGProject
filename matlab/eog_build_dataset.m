function DS = eog_build_dataset(varargin)
%EOG_BUILD_DATASET  Turn the filtered recordings into labelled CNN windows.
%
%   Six classes, all derived from the recordings rather than supplied with
%   them, because the dataset labels only forward saccade, return saccade and
%   blink. Where each comes from:
%
%     U D L R   A saccade's direction, read from TargetGA. The targets are
%               random points on the screen, not four cardinal directions, so
%               the class is the dominant axis of the target: left or right
%               when |h| >= |v|, up or down otherwise. A return saccade goes
%               back to centre, so it carries the opposite direction of the
%               forward saccade before it.
%
%               Targets within DiagonalTolerance degrees of the 45 degree
%               diagonal are dropped: there the dominant axis is decided by
%               noise, and roughly a fifth of all saccades sit there.
%
%     blink     The burst inside a blink-labelled stretch. The dataset labels
%               a full two seconds as blink, but the blink itself is short:
%               80% of the vertical energy falls in about 17% of the stretch.
%               The burst is found by thresholding a 50 ms envelope of the
%               vertical channel, which locates one in every stretch with a
%               median duration of 375 ms.
%
%     idle      The quiet remainder of those same stretches, kept clear of the
%               burst by IdleGuard. The recording has no idle label of its own:
%               every sample is marked forward, return or blink.
%
%   Name-value options:
%     'FiltDir'            folder holding S1.filt .. S6.filt   (default data)
%     'DatasetDir'         folder holding S1 .. S6 with TargetGA.mat
%     'Subjects'           cellstr of subject names            (default S1..S6)
%     'Window'             samples per example                 (default 192)
%     'DiagonalTolerance'  degrees either side of 45           (default 10)
%     'BlinkThreshold'     envelope level marking a blink      (default 1.0)
%     'IdleGuard'          seconds kept clear of a blink       (default 0.15)
%     'IdlePerStretch'     idle windows taken per stretch      (default 2)
%     'Save'               .mat file to write                  (default none)
%
%   Returns a struct with:
%     X         1 x N cell, each [channels x window]
%     Y         N x 1 categorical, one of U D L R idle blink
%     subject   N x 1 categorical, which recording the window came from
%     window, fs, classes, and the options used
%
%   The subject column matters: windows from one saccade overlap heavily, so
%   a random split would put near-copies on both sides of it and report an
%   accuracy the model has not earned. Split on subject instead.

%% ------------------------------------------------------------------ options
p = inputParser;
p.addParameter('FiltDir', fullfile(pwd, 'data'), @(s) ischar(s) || isstring(s));
p.addParameter('DatasetDir', '', @(s) ischar(s) || isstring(s));
p.addParameter('Subjects', {'S1','S2','S3','S4','S5','S6'}, @iscellstr);
p.addParameter('Window', 192, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('DiagonalTolerance', 10, @isnumeric);
p.addParameter('BlinkThreshold', 1.0, @isnumeric);
p.addParameter('IdleGuard', 0.15, @isnumeric);
p.addParameter('IdlePerStretch', 2, @isnumeric);
p.addParameter('Save', '', @(s) ischar(s) || isstring(s));
p.parse(varargin{:});
opt = p.Results;

if isempty(opt.DatasetDir)
    error('eog_build_dataset:noDataset', ...
          ['DatasetDir is required: the folder holding S1..S6 with ' ...
           'TargetGA.mat, which carries the saccade targets.']);
end

W = double(opt.Window);
classes = {'U', 'D', 'L', 'R', 'idle', 'blink'};

X = {};
Yc = {};
Sc = {};
report = zeros(numel(opt.Subjects), 7);   % per subject: U D L R idle blink dropped

%% ------------------------------------------------------------------ build
for si = 1:numel(opt.Subjects)
    subj = opt.Subjects{si};
    filtPath = fullfile(char(opt.FiltDir), [subj '.filt']);
    gaPath   = fullfile(char(opt.DatasetDir), subj, 'TargetGA.mat');

    D = eog_load_filtered(filtPath);
    if exist(gaPath, 'file') ~= 2
        error('eog_build_dataset:noTargets', 'Cannot find %s', gaPath);
    end
    ga = load(gaPath);
    TargetGA = ga.TargetGA;

    sig = [D.filt_h, D.filt_v]';          % channels x samples
    lab = D.label;
    fs  = D.fs;

    % ---- directional saccades ------------------------------------------
    fwd = findRuns(lab, 1);
    ret = findRuns(lab, 2);
    nPairs = min(size(fwd, 1), size(ret, 1));

    for k = 1:nPairs
        % TargetGA alternates: outward target, then centre for the return
        row = 2 * k - 1;
        if row > size(TargetGA, 1)
            break
        end
        h = TargetGA(row, 1);
        v = TargetGA(row, 2);

        [cls, ok] = directionOf(h, v, opt.DiagonalTolerance);
        if ~ok
            report(si, 7) = report(si, 7) + 2;   % the pair is dropped together
            continue
        end

        % forward goes to the target, return comes back the other way
        [xw, got] = takeWindow(sig, fwd(k, 1), W);
        if got
            X{end+1} = xw; Yc{end+1} = cls; Sc{end+1} = subj; %#ok<AGROW>
            report(si, classIndex(cls, classes)) = ...
                report(si, classIndex(cls, classes)) + 1;
        end

        back = oppositeOf(cls);
        [xw, got] = takeWindow(sig, ret(k, 1), W);
        if got
            X{end+1} = xw; Yc{end+1} = back; Sc{end+1} = subj; %#ok<AGROW>
            report(si, classIndex(back, classes)) = ...
                report(si, classIndex(back, classes)) + 1;
        end
    end

    % ---- blink bursts and the idle around them --------------------------
    env = movmean(abs(sig(2, :)), round(0.05 * fs));   % 50 ms, vertical channel
    guard = round(opt.IdleGuard * fs);
    stretches = findRuns(lab, 3);

    for k = 1:size(stretches, 1)
        a = stretches(k, 1);
        b = stretches(k, 2);
        inside = find(env(a:b) > opt.BlinkThreshold);
        if isempty(inside)
            continue
        end
        bs = a + inside(1) - 1;
        be = a + inside(end) - 1;

        % blink window, centred on the burst
        centre = round((bs + be) / 2);
        [xw, got] = takeWindow(sig, centre - floor(W / 2), W);
        if got
            X{end+1} = xw; Yc{end+1} = 'blink'; Sc{end+1} = subj; %#ok<AGROW>
            report(si, 6) = report(si, 6) + 1;
        end

        % idle windows from the quiet tail, clear of the burst
        quietFrom = be + guard;
        quietTo   = b - guard;
        taken = 0;
        pos = quietFrom;
        while taken < opt.IdlePerStretch && pos + W - 1 <= quietTo
            [xw, got] = takeWindow(sig, pos, W);
            if ~got
                break
            end
            X{end+1} = xw; Yc{end+1} = 'idle'; Sc{end+1} = subj; %#ok<AGROW>
            report(si, 5) = report(si, 5) + 1;
            taken = taken + 1;
            pos = pos + W;
        end
    end
end

%% ------------------------------------------------------------------ pack
DS.X       = X;
DS.Y       = categorical(Yc(:), classes);
DS.subject = categorical(Sc(:), opt.Subjects);
DS.window  = W;
DS.fs      = fs;
DS.classes = classes;
DS.options = opt;

%% ------------------------------------------------------------------ report
fprintf('\nWindows built: %d of %d samples each, %d channels\n', ...
        numel(X), W, size(X{1}, 2));
fprintf('  window spans %.0f ms at %g Hz\n\n', 1000 * W / fs, fs);

fprintf('  %-8s %6s %6s %6s %6s %7s %7s %9s\n', ...
        'subject', 'U', 'D', 'L', 'R', 'idle', 'blink', 'dropped');
fprintf('  %s\n', repmat('-', 1, 62));
for si = 1:numel(opt.Subjects)
    fprintf('  %-8s %6d %6d %6d %6d %7d %7d %9d\n', ...
            opt.Subjects{si}, report(si, 1), report(si, 2), report(si, 3), ...
            report(si, 4), report(si, 5), report(si, 6), report(si, 7));
end
fprintf('  %s\n', repmat('-', 1, 62));
fprintf('  %-8s %6d %6d %6d %6d %7d %7d %9d\n', 'total', ...
        sum(report(:, 1)), sum(report(:, 2)), sum(report(:, 3)), ...
        sum(report(:, 4)), sum(report(:, 5)), sum(report(:, 6)), ...
        sum(report(:, 7)));
fprintf('\n  dropped = saccades within %g deg of the diagonal\n', ...
        opt.DiagonalTolerance);

if ~isempty(char(opt.Save))
    save(char(opt.Save), 'DS', '-v7.3');
    fprintf('  saved %s\n', char(opt.Save));
end
end


%% ======================================================================
function runs = findRuns(label, want)
%FINDRUNS  Start and end indices of each contiguous run of `want`.
m = [false; label(:) == want; false];
d = diff(m);
starts = find(d == 1);
stops  = find(d == -1) - 1;
runs = [starts, stops];
end


%% ======================================================================
function [cls, ok] = directionOf(h, v, tolDeg)
%DIRECTIONOF  Dominant axis of a target, or ok=false when too close to 45 deg.
ok = false;
cls = '';
if abs(h) < 1e-6 && abs(v) < 1e-6
    return                                   % the centre is not a direction
end
ang = atan2d(abs(v), abs(h));
if abs(ang - 45) < tolDeg
    return                                   % the axis would be decided by noise
end
ok = true;
if abs(h) >= abs(v)
    if h > 0, cls = 'R'; else, cls = 'L'; end
else
    if v > 0, cls = 'U'; else, cls = 'D'; end
end
end


%% ======================================================================
function back = oppositeOf(cls)
switch cls
    case 'U', back = 'D';
    case 'D', back = 'U';
    case 'L', back = 'R';
    case 'R', back = 'L';
    otherwise, back = cls;
end
end


%% ======================================================================
function i = classIndex(cls, classes)
i = find(strcmp(classes, cls), 1);
end


%% ======================================================================
function [w, ok] = takeWindow(sig, from, W)
%TAKEWINDOW  W samples starting at `from`, or ok=false if it runs off the end.
%
%   Returned as time-by-channel, which is the orientation trainnet expects of
%   a cell of sequences, rather than the channel-by-time of older interfaces.
ok = false;
w = [];
if from < 1 || from + W - 1 > size(sig, 2)
    return
end
w = sig(:, from:(from + W - 1))';
ok = true;
end
