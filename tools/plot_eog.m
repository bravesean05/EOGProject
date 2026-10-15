function T = plot_eog(source, varargin)
%PLOT_EOG  Plot a raw EOG recording against the filtered output, side by side.
%
%   Reads the CSV written by filter_dataset, which carries the raw signal, the
%   standardized and smoothed output, the epoch statistics behind that output,
%   and the ground-truth label, all indexed by the same absolute sample.
%
%   The raw signal and the filtered output cannot share an axis: raw sits in
%   the hundreds while the output is a z-score of roughly +/-3, so one would
%   flatten the other. They go in stacked panels with a shared time axis
%   instead, linked so that zooming one zooms all.
%
%   plot_eog(csvPath)
%       Plots the first 20 seconds of both channels.
%
%   plot_eog(csvPath, 'TimeRange', [t0 t1])
%       Plots the window from t0 to t1 seconds.
%
%   plot_eog(csvPath, 'Channel', 'h')
%       Plots the horizontal channel only. Accepts 'h', 'v' or 'both'.
%
%   plot_eog(csvPath, 'ShowStats', false)
%       Omits the epoch-deviation panel.
%
%   T = plot_eog(...) returns the loaded table. A full recording is around
%   300,000 rows, and reading it takes far longer than drawing it, so pass the
%   table back in to look at another window without paying that cost again:
%
%       T = plot_eog('data/S1_filtered.csv', 'TimeRange', [0 20]);
%       plot_eog(T, 'TimeRange', [100 108]);
%       plot_eog(T, 'TimeRange', [540 548], 'Channel', 'h');
%
%   Examples:
%       plot_eog('data/S1_filtered.csv')
%       plot_eog('data/S1_filtered.csv', 'TimeRange', [100 108])
%       plot_eog('data/S1_filtered.csv', 'TimeRange', [0 12], 'Channel', 'h')
%
%   Panels, per channel:
%       1. Raw signal, with the epoch mean drawn over it. The gap between the
%          two is the baseline drift that standardization removes.
%       2. Filtered output: standardized, then Savitzky-Golay smoothed.
%       3. Epoch standard deviation. This is the scale the z-score divides out,
%          and is where saccades and blinks separate: saccades are mostly
%          horizontal in this protocol, blinks mostly vertical.
%
%   Background shading marks the ground-truth label of each sample.

%% ------------------------------------------------------------------ options
p = inputParser;
p.addRequired('source', @(x) ischar(x) || isstring(x) || istable(x));
p.addParameter('TimeRange', [], @(x) isnumeric(x) && numel(x) == 2);
p.addParameter('Channel', 'both', @(s) any(strcmpi(s, {'h', 'v', 'both'})));
p.addParameter('ShowStats', true, @islogical);
p.parse(source, varargin{:});
opt = p.Results;

channel = lower(opt.Channel);

%% --------------------------------------------------------------- load data
if istable(source)
    T = source;
else
    csvPath = char(source);
    if exist(csvPath, 'file') ~= 2
        error('plot_eog:fileNotFound', ...
              ['Cannot find %s\n\n' ...
               'Generate it first:\n' ...
               '  build/filter_dataset data/converted/S1.eog -o %s'], ...
              csvPath, csvPath);
    end
    fprintf('Reading %s ...\n', csvPath);
    T = readtable(csvPath);
end

required = {'time_s', 'raw_h', 'filt_h', 'mean_h', 'sd_h', ...
            'raw_v', 'filt_v', 'mean_v', 'sd_v', 'label'};
missing = required(~ismember(required, T.Properties.VariableNames));
if ~isempty(missing)
    error('plot_eog:badColumns', ...
          'CSV is missing column(s): %s\nRebuild it with the current filter_dataset.', ...
          strjoin(missing, ', '));
end

fprintf('  %d samples, %.1f to %.1f s\n', ...
        height(T), T.time_s(1), T.time_s(end));

%% ------------------------------------------------------------ select window
if isempty(opt.TimeRange)
    t0 = T.time_s(1);
    t1 = min(t0 + 20, T.time_s(end));
    fprintf('  no TimeRange given, showing %.1f-%.1f s\n', t0, t1);
else
    t0 = opt.TimeRange(1);
    t1 = opt.TimeRange(2);
end

keep = T.time_s >= t0 & T.time_s <= t1;
if ~any(keep)
    error('plot_eog:emptyRange', ...
          'No samples between %.3f and %.3f s. The file covers %.3f to %.3f s.', ...
          t0, t1, T.time_s(1), T.time_s(end));
end
W = T(keep, :);
fprintf('  plotting %d samples\n', height(W));

%% ------------------------------------------------------------------- layout
switch channel
    case 'both', channels = {'h', 'v'};
    otherwise,   channels = {channel};
end

nRows = 2 + double(opt.ShowStats);
nCols = numel(channels);

figure('Name', sprintf('EOG raw vs filtered  (%.1f-%.1f s)', t0, t1), ...
       'Color', 'w', 'Position', [80 60 640 * nCols 820]);

axesList = gobjects(nRows, nCols);

for c = 1:nCols
    ch = channels{c};
    if strcmp(ch, 'h')
        chName = 'Horizontal';
    else
        chName = 'Vertical';
    end

    raw  = W.(['raw_'  ch]);
    filt = W.(['filt_' ch]);
    mu   = W.(['mean_' ch]);
    sd   = W.(['sd_'   ch]);

    % -- panel 1: raw signal with the epoch mean over it -------------------
    ax = subplot(nRows, nCols, c);
    axesList(1, c) = ax;
    hold(ax, 'on');
    shadeLabels(ax, W.time_s, W.label);
    plot(ax, W.time_s, raw, 'Color', [0.15 0.15 0.15], 'LineWidth', 0.8);
    plot(ax, W.time_s, mu,  'Color', [0.85 0.20 0.20], 'LineWidth', 1.4);
    hold(ax, 'off');
    ylabel(ax, 'amplitude');
    title(ax, sprintf('%s - raw (black) and epoch mean (red)', chName));
    grid(ax, 'on');
    fitY(ax, [raw; mu]);

    % -- panel 2: filtered output ------------------------------------------
    ax = subplot(nRows, nCols, nCols + c);
    axesList(2, c) = ax;
    hold(ax, 'on');
    shadeLabels(ax, W.time_s, W.label);
    % Zero line drawn with plot rather than yline, which needs R2018b
    plot(ax, [W.time_s(1) W.time_s(end)], [0 0], ...
         'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    plot(ax, W.time_s, filt, 'Color', [0.10 0.35 0.75], 'LineWidth', 1.0);
    hold(ax, 'off');
    ylabel(ax, 'z-score');
    title(ax, sprintf('%s - standardized + Savitzky-Golay', chName));
    grid(ax, 'on');
    fitY(ax, filt);

    % -- panel 3: epoch deviation ------------------------------------------
    if opt.ShowStats
        ax = subplot(nRows, nCols, 2 * nCols + c);
        axesList(3, c) = ax;
        hold(ax, 'on');
        shadeLabels(ax, W.time_s, W.label);
        plot(ax, W.time_s, sd, 'Color', [0.10 0.55 0.25], 'LineWidth', 1.2);
        hold(ax, 'off');
        ylabel(ax, 'epoch sd');
        title(ax, sprintf('%s - epoch deviation (the scale divided out)', chName));
        grid(ax, 'on');
        fitY(ax, sd);
    end

    xlabel(axesList(end, c), 'time (s)');
end

% One shared time axis, so zooming or panning any panel moves them together
linkaxes(axesList(:), 'x');
xlim(axesList(1, 1), [t0 t1]);

addLabelLegend(axesList(1, 1));

end % plot_eog


%% ======================================================================
function fitY(ax, data)
%FITY  Set the y limits from the data, ignoring the shading patches.
%
%   The background patches deliberately span a far larger y range than the
%   data so they fill the panel whatever the limits turn out to be. Left on
%   automatic, the axis would grow to contain them and squash the traces into
%   an invisible line, so the limits are set from the data instead.

data = double(data(:));
data = data(isfinite(data));
if isempty(data)
    return
end

lo = min(data);
hi = max(data);
if hi <= lo
    pad = max(abs(hi), 1) * 0.05;   % flat series still needs a visible band
else
    pad = (hi - lo) * 0.08;
end
ylim(ax, [lo - pad, hi + pad]);
end


%% ======================================================================
function shadeLabels(ax, t, label)
%SHADELABELS  Shade the background by ground-truth label.
%
%   Draws one patch per contiguous run of the same label, so a 4 s trial
%   becomes three bands rather than thousands of slivers. Patches are drawn
%   before the data and pushed to the back, and are excluded from the legend.

if isempty(t)
    return
end

% Index by label value: row 1 is label 1, and so on. Label 0 (none) has no row
% and is left unshaded.
colours = [0.80 0.88 0.98;    % 1 forward saccade: blue
           0.82 0.94 0.82;    % 2 return saccade:  green
           0.99 0.91 0.78];   % 3 blink:           amber

label = double(label(:));     % readtable may hand back an integer type
t = double(t(:));

% Boundaries of each run of identical labels, so a 4 s trial becomes three
% bands rather than thousands of slivers
edges = [1; find(diff(label) ~= 0) + 1; numel(label) + 1];

yLo = -1e6;
yHi =  1e6;   % patches span far beyond the data; the axis limits clip them

for k = 1:numel(edges) - 1
    first = edges(k);
    last  = edges(k + 1) - 1;
    lab   = label(first);
    if lab < 1 || lab > size(colours, 1)
        continue
    end
    x0 = t(first);
    x1 = t(last);
    patch(ax, [x0 x1 x1 x0], [yLo yLo yHi yHi], colours(lab, :), ...
          'EdgeColor', 'none', 'FaceAlpha', 0.55, 'HandleVisibility', 'off');
end
% Callers shade before plotting, so these patches are created first and render
% behind the traces without any explicit restacking.
end


%% ======================================================================
function addLabelLegend(ax)
%ADDLABELLEGEND  Legend naming the shading colours.
%
%   The shading patches are hidden from the legend, so draw three invisible
%   dummy patches whose only job is to carry the names.

colours = [0.80 0.88 0.98; 0.82 0.94 0.82; 0.99 0.91 0.78];

hold(ax, 'on');
h = gobjects(1, 3);
for k = 1:3
    h(k) = patch(ax, 'XData', NaN, 'YData', NaN, ...
                 'FaceColor', colours(k, :), 'EdgeColor', 'none');
end
hold(ax, 'off');
legend(ax, h, {'forward saccade', 'return saccade', 'blink'}, ...
       'Location', 'northeast', 'Orientation', 'horizontal', ...
       'FontSize', 8, 'Box', 'off');
end
