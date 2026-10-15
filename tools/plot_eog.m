function T = plot_eog(source, varargin)
%PLOT_EOG  Plot a raw EOG recording against the filtered output.
%
%   Reads the CSV written by filter_dataset, which carries the raw signal, the
%   standardized and smoothed output, the epoch statistics behind that output,
%   and the ground-truth label, all indexed by the same absolute sample.
%
%   Raw and filtered cannot share one y axis. Raw sits in the hundreds while
%   the output is a z-score of roughly +/-3, so a single scale flattens one of
%   them into a line. Two ways out, both offered here:
%
%     'overlay' (default)  Both traces on one set of axes, raw against the
%                          left scale and the filtered output against the
%                          right, so their shapes can be compared directly.
%     'stacked'            Separate panels, one trace each.
%
%   plot_eog(csvPath)
%       Overlays the first 20 seconds of both channels.
%
%   plot_eog(csvPath, 'TimeRange', [t0 t1])
%       Plots the window from t0 to t1 seconds.
%
%   plot_eog(csvPath, 'Style', 'stacked')
%       One trace per panel instead of overlaid.
%
%   plot_eog(csvPath, 'Channel', 'h')
%       Plots the horizontal channel only. Accepts 'h', 'v' or 'both'.
%
%   plot_eog(csvPath, 'ShowMean', false)
%       Omits the epoch mean drawn over the raw trace.
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
%       plot_eog('data/S1_filtered.csv', 'Style', 'stacked')
%
%   Background shading marks the ground-truth label of each sample. The epoch
%   deviation panel shows the scale the z-score divides out, which is where
%   saccades and blinks separate: saccades are mostly horizontal in this
%   protocol, blinks mostly vertical.
%
%   Requires R2016a or later for yyaxis in the overlay style.

%% ------------------------------------------------------------------ colours
C.raw   = [0.15 0.15 0.15];
C.mean  = [0.85 0.20 0.20];
C.filt  = [0.10 0.35 0.75];
C.sd    = [0.10 0.55 0.25];
C.zero  = [0.60 0.60 0.60];
C.label = [0.80 0.88 0.98;    % 1 forward saccade: blue
           0.82 0.94 0.82;    % 2 return saccade:  green
           0.99 0.91 0.78];   % 3 blink:           amber

%% ------------------------------------------------------------------ options
p = inputParser;
p.addRequired('source', @(x) ischar(x) || isstring(x) || istable(x));
p.addParameter('TimeRange', [], @(x) isnumeric(x) && numel(x) == 2);
p.addParameter('Channel', 'both', @(s) any(strcmpi(s, {'h', 'v', 'both'})));
p.addParameter('Style', 'overlay', @(s) any(strcmpi(s, {'overlay', 'stacked'})));
p.addParameter('ShowStats', true, @islogical);
p.addParameter('ShowMean', true, @islogical);
p.parse(source, varargin{:});
opt = p.Results;

channel = lower(opt.Channel);
style   = lower(opt.Style);

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
fprintf('  plotting %d samples (%s)\n', height(W), style);

%% ------------------------------------------------------------------- layout
switch channel
    case 'both', channels = {'h', 'v'};
    otherwise,   channels = {channel};
end

if strcmp(style, 'overlay')
    nRows = 1 + double(opt.ShowStats);
else
    nRows = 2 + double(opt.ShowStats);
end
nCols = numel(channels);

figure('Name', sprintf('EOG raw vs filtered  (%.1f-%.1f s)', t0, t1), ...
       'Color', 'w', 'Position', [80 60 660 * nCols, 300 * nRows + 120]);

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

    if strcmp(style, 'overlay')
        ax = subplot(nRows, nCols, c);
        axesList(1, c) = ax;
        drawOverlay(ax, W.time_s, raw, mu, filt, W.label, C, ...
                    opt.ShowMean, chName);
        row = 1;
    else
        ax = subplot(nRows, nCols, c);
        axesList(1, c) = ax;
        drawRaw(ax, W.time_s, raw, mu, W.label, C, opt.ShowMean, chName);

        ax = subplot(nRows, nCols, nCols + c);
        axesList(2, c) = ax;
        drawFiltered(ax, W.time_s, filt, W.label, C, chName);
        row = 2;
    end

    if opt.ShowStats
        ax = subplot(nRows, nCols, row * nCols + c);
        axesList(row + 1, c) = ax;
        drawDeviation(ax, W.time_s, sd, W.label, C, chName);
    end

    xlabel(axesList(end, c), 'time (s)');
end

% One shared time axis, so zooming or panning any panel moves them together
linkaxes(axesList(:), 'x');
xlim(axesList(1, 1), [t0 t1]);

end % plot_eog


%% ======================================================================
function drawOverlay(ax, t, raw, mu, filt, label, C, showMean, chName)
%DRAWOVERLAY  Raw on the left scale, filtered output on the right.
%
%   Each side gets its own limits, set from its own data, so the two traces
%   fill the panel rather than one collapsing against the other's range. The
%   axis rulers are coloured to match their traces, since with two scales the
%   numbers alone do not say which belongs to which.

% ---- left: raw, in the recording's own units
yyaxis(ax, 'left');
hold(ax, 'on');
shadeLabels(ax, t, label, C.label);
h = gobjects(0);
names = {};
hRaw = plot(ax, t, raw, 'Color', C.raw, 'LineWidth', 0.8, 'LineStyle', '-');
h(end+1) = hRaw;  names{end+1} = 'raw';
if showMean
    hMean = plot(ax, t, mu, 'Color', C.mean, 'LineWidth', 1.4, 'LineStyle', '-');
    h(end+1) = hMean;  names{end+1} = 'epoch mean';
    fitY(ax, [raw; mu]);
else
    fitY(ax, raw);
end
ylabel(ax, 'raw amplitude');
ax.YAxis(1).Color = C.raw;

% ---- right: standardized and smoothed output
yyaxis(ax, 'right');
hold(ax, 'on');
plot(ax, [t(1) t(end)], [0 0], 'Color', C.zero, ...
     'LineStyle', '-', 'HandleVisibility', 'off');
hFilt = plot(ax, t, filt, 'Color', C.filt, 'LineWidth', 1.2, 'LineStyle', '-');
h(end+1) = hFilt;  names{end+1} = 'filtered (z)';
fitY(ax, filt);
ylabel(ax, 'filtered (z-score)');
ax.YAxis(2).Color = C.filt;

hold(ax, 'off');
title(ax, sprintf('%s - raw (left scale) over filtered (right scale)', chName));
grid(ax, 'on');

addLegend(ax, h, names, C.label);
end


%% ======================================================================
function drawRaw(ax, t, raw, mu, label, C, showMean, chName)
%DRAWRAW  Raw signal with the epoch mean over it, in its own panel.

hold(ax, 'on');
shadeLabels(ax, t, label, C.label);
h = plot(ax, t, raw, 'Color', C.raw, 'LineWidth', 0.8);
names = {'raw'};
if showMean
    h(end+1) = plot(ax, t, mu, 'Color', C.mean, 'LineWidth', 1.4);
    names{end+1} = 'epoch mean';
    fitY(ax, [raw; mu]);
else
    fitY(ax, raw);
end
hold(ax, 'off');
ylabel(ax, 'raw amplitude');
title(ax, sprintf('%s - raw and epoch mean', chName));
grid(ax, 'on');
addLegend(ax, h, names, C.label);
end


%% ======================================================================
function drawFiltered(ax, t, filt, label, C, chName)
%DRAWFILTERED  Standardized and smoothed output, in its own panel.

hold(ax, 'on');
shadeLabels(ax, t, label, C.label);
% Zero line drawn with plot rather than yline, which needs R2018b
plot(ax, [t(1) t(end)], [0 0], 'Color', C.zero, 'HandleVisibility', 'off');
plot(ax, t, filt, 'Color', C.filt, 'LineWidth', 1.0);
hold(ax, 'off');
fitY(ax, filt);
ylabel(ax, 'z-score');
title(ax, sprintf('%s - standardized + Savitzky-Golay', chName));
grid(ax, 'on');
end


%% ======================================================================
function drawDeviation(ax, t, sd, label, C, chName)
%DRAWDEVIATION  Epoch standard deviation, the scale the z-score divides out.

hold(ax, 'on');
shadeLabels(ax, t, label, C.label);
plot(ax, t, sd, 'Color', C.sd, 'LineWidth', 1.2);
hold(ax, 'off');
fitY(ax, sd);
ylabel(ax, 'epoch sd');
title(ax, sprintf('%s - epoch deviation (the scale divided out)', chName));
grid(ax, 'on');
end


%% ======================================================================
function fitY(ax, data)
%FITY  Set the y limits of the active side from the data alone.
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
function shadeLabels(ax, t, label, colours)
%SHADELABELS  Shade the background by ground-truth label.
%
%   Draws one patch per contiguous run of the same label, so a 4 s trial
%   becomes three bands rather than thousands of slivers. Row k of `colours`
%   is the colour for label k; label 0 (none) has no row and stays unshaded.

if isempty(t)
    return
end

label = double(label(:));     % readtable may hand back an integer type
t = double(t(:));

% Boundaries of each run of identical labels
edges = [1; find(diff(label) ~= 0) + 1; numel(label) + 1];

yLo = -1e6;
yHi =  1e6;   % patches span far beyond the data; fitY clips them back

for k = 1:numel(edges) - 1
    first = edges(k);
    last  = edges(k + 1) - 1;
    lab   = label(first);
    if lab < 1 || lab > size(colours, 1)
        continue
    end
    patch(ax, [t(first) t(last) t(last) t(first)], [yLo yLo yHi yHi], ...
          colours(lab, :), 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
          'HandleVisibility', 'off');
end
% Callers shade before plotting, so these patches are created first and render
% behind the traces without any explicit restacking.
end


%% ======================================================================
function addLegend(ax, traceHandles, traceNames, colours)
%ADDLEGEND  Name the traces and the label shading in one legend.
%
%   The shading patches are hidden from the legend, so three invisible dummy
%   patches carry the label names. Trace handles are passed in explicitly
%   because with two y axes a bare legend call sees only the active side.

labelNames = {'forward saccade', 'return saccade', 'blink'};

hold(ax, 'on');
dummies = gobjects(1, size(colours, 1));
for k = 1:size(colours, 1)
    dummies(k) = patch(ax, 'XData', NaN, 'YData', NaN, ...
                       'FaceColor', colours(k, :), 'EdgeColor', 'none');
end
hold(ax, 'off');

legend(ax, [traceHandles(:); dummies(:)], ...
       [traceNames(:); labelNames(:)], ...
       'Location', 'northoutside', 'Orientation', 'horizontal', ...
       'FontSize', 8, 'Box', 'off');
end
