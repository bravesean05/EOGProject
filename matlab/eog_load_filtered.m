function D = eog_load_filtered(filtPath)
%EOG_LOAD_FILTERED  Read a .filt binary written by filter_dataset -b.
%
%   D = eog_load_filtered('data/S1.filt') returns a struct with:
%
%     n            number of emitted samples
%     first        absolute index in the original recording of the first one
%     fs           sampling rate, Hz
%     filt_h       filtered horizontal channel, n x 1
%     filt_v       filtered vertical channel, n x 1
%     sd_h, sd_v   epoch deviation behind each sample
%     label        ground-truth control signal, 0 none, 1 forward saccade,
%                  2 return saccade, 3 blink
%     idx          absolute index of each sample, n x 1
%
%   The pipeline emits nothing until it has collected a full epoch, so `first`
%   is nonzero and `idx` is what lines these series up with the original
%   recording and with TargetGA.

if exist(filtPath, 'file') ~= 2
    error('eog_load_filtered:notFound', ...
          ['Cannot find %s\n\nGenerate it with:\n' ...
           '  build/filter_dataset data/converted/S1.eog -b %s'], ...
          filtPath, filtPath);
end

fid = fopen(filtPath, 'rb');
cleaner = onCleanup(@() fclose(fid));

magic = fread(fid, 8, '*char')';
if ~strcmp(magic, 'EOGFILT1')
    error('eog_load_filtered:badMagic', ...
          '%s does not start with EOGFILT1 (found "%s")', filtPath, magic);
end

n     = fread(fid, 1, 'uint32');
first = fread(fid, 1, 'uint32');
fs    = fread(fid, 1, 'float32');

D.n      = double(n);
D.first  = double(first);
D.fs     = double(fs);
D.filt_h = fread(fid, n, 'float32');
D.filt_v = fread(fid, n, 'float32');
D.sd_h   = fread(fid, n, 'float32');
D.sd_v   = fread(fid, n, 'float32');
D.label  = fread(fid, n, 'uint8');

if numel(D.label) ~= n
    error('eog_load_filtered:truncated', ...
          '%s ended early: expected %d samples, read %d', ...
          filtPath, n, numel(D.label));
end

D.idx = (D.first:(D.first + D.n - 1))';
end
