function [h, fs] = measure_rir(outFile, varargin)
%MEASURE_RIR Measure a room impulse response with averaged exponential sine sweeps.
%   [h, fs] = measure_rir('livingroom_pos1.wav')
%   [h, fs] = measure_rir(outFile, 'fs', 48000, 'T', 5, 'tail', 3, 'repeats', 3, 'level', 0.3)
%
%   Uses audioPlayerRecorder (Audio Toolbox) for sample-synchronous play/record.
%   Without the toolbox, falls back to audioplayer/audiorecorder; the I/O delay is
%   then arbitrary, which is fine because analysis aligns on the direct-path onset.
%   After measuring, run analyze_rir(outFile).

p = inputParser;
addParameter(p, 'fs', 48000);
addParameter(p, 'f1', 40);
addParameter(p, 'T', 5);
addParameter(p, 'tail', 3);
addParameter(p, 'repeats', 3);
addParameter(p, 'level', 0.3);
parse(p, varargin{:});
o = p.Results;
fs = o.fs;

[sweep, inv] = generate_ess(o.f1, 0.45*fs, o.T, fs);
play = [sweep * o.level; zeros(round(o.tail*fs), 1)];
irLen = round(o.tail * fs);
irs = zeros(irLen, o.repeats);

for r = 1:o.repeats
    fprintf('sweep %d/%d ...\n', r, o.repeats);
    rec = play_record(play, fs);
    if max(abs(rec)) > 0.98
        warning('Input clipped: lower the level or mic gain.');
    end
    full = fftfilt_full(rec / o.level, inv);
    irs(:, r) = full(length(sweep) : length(sweep) + irLen - 1);
end

% align repeats on the direct-path peak, then average (≈ +10log10(N) dB SNR)
[~, pk] = max(abs(irs));
shift = pk - min(pk);
L = irLen - max(shift);
aligned = zeros(L, o.repeats);
for r = 1:o.repeats
    aligned(:, r) = irs(1+shift(r) : shift(r)+L, r);
end
h = mean(aligned, 2);
h = 0.9 * h / max(abs(h));
audiowrite(outFile, h, fs, 'BitsPerSample', 32);
fprintf('saved %s\n', outFile);
end

function rec = play_record(x, fs)
if exist('audioPlayerRecorder', 'file') == 2 || exist('audioPlayerRecorder', 'class') == 8
    apr = audioPlayerRecorder('SampleRate', fs);
    frame = 1024;
    nFrames = ceil(length(x) / frame);
    x(end+1 : nFrames*frame) = 0;
    rec = zeros(size(x));
    for k = 1:nFrames
        idx = (k-1)*frame + (1:frame);
        rec(idx) = apr(x(idx));
    end
    release(apr);
else
    recObj = audiorecorder(fs, 24, 1);
    player = audioplayer(x, fs);
    record(recObj);
    pause(0.2);
    playblocking(player);
    pause(0.3);
    stop(recObj);
    rec = getaudiodata(recObj);
end
end

function y = fftfilt_full(a, b)
L = length(a) + length(b) - 1;
N = 2^nextpow2(L);
y = real(ifft(fft(a, N) .* fft(b, N)));
y = y(1:L);
end
