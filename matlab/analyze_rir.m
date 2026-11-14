function params = analyze_rir(rirFile, doPlot)
%ANALYZE_RIR ISO 3382-1 room parameters from an impulse response.
%   params = analyze_rir('livingroom_pos1.wav')
%   Returns struct with fields: bands, T20, T30, T60 (T30 with T20 fallback), EDT,
%   DRR (dB), C50 (dB). Uses the same definitions as roomest/acoustics.py:
%     - zero-phase 3rd-order Butterworth octave filters
%     - onset at -20 dB re. peak
%     - Schroeder backward integration with noise-floor handling (Lundeby-lite)
%     - DRR: +/-2.5 ms window around the direct-path peak; C50: 50 ms from direct sound
%   Validate against Python with: python -m roomest.measure analyze <file>

if nargin < 2, doPlot = true; end
[h, fs] = audioread(rirFile);
h = h(:, 1);
bands = [125 250 500 1000 2000 4000];
params = struct('bands', bands, 'T20', nan(1,6), 'T30', nan(1,6), 'T60', nan(1,6), 'EDT', nan(1,6));

if doPlot, figure('Name', 'Schroeder decay'); hold on; end
for b = 1:numel(bands)
    fc = bands(b);
    [B, A] = butter(3, [fc/sqrt(2), min(fc*sqrt(2), 0.95*fs/2)] / (fs/2), 'bandpass');
    hb = filtfilt(B, A, h);
    edc = schroeder_edc_db(hb, fs);
    params.T20(b) = decay_time(edc, fs, -5, -25);
    params.T30(b) = decay_time(edc, fs, -5, -35);
    params.EDT(b) = decay_time(edc, fs, 0, -10);
    params.T60(b) = params.T30(b);
    if isnan(params.T60(b)), params.T60(b) = params.T20(b); end
    if doPlot, plot((0:numel(edc)-1)/fs, edc, 'LineWidth', 1.4, 'DisplayName', sprintf('%d Hz', fc)); end
end
if doPlot
    ylim([-70 2]); grid on; xlabel('time (s)'); ylabel('energy decay (dB)');
    legend('show', 'Location', 'northeast'); title('Schroeder energy decay curves');
end

[~, pk] = max(abs(h));
w = round(2.5e-3 * fs);
a = max(1, pk - w); bEnd = pk + w;
params.DRR = 10*log10(sum(h(a:bEnd).^2) / (late_energy(h(bEnd+1:end), fs) + eps));
split = a + round(0.05 * fs);
params.C50 = 10*log10(sum(h(a:split-1).^2) / (late_energy(h(split:end), fs) + eps));

fprintf('\n%-8s', 'band'); fprintf('%8d', bands); fprintf('\n');
fprintf('%-8s', 'T30 (s)'); fprintf('%8.2f', params.T30); fprintf('\n');
fprintf('%-8s', 'T60 (s)'); fprintf('%8.2f', params.T60); fprintf('\n');
fprintf('%-8s', 'EDT (s)'); fprintf('%8.2f', params.EDT); fprintf('\n');
fprintf('DRR = %.1f dB   C50 = %.1f dB\n', params.DRR, params.C50);
end

% ------------------------------------------------------------------ helpers
function edc = schroeder_edc_db(h, fs)
e = h.^2;
on = find(e >= max(e) * 10^(-2), 1);           % -20 dB onset
e = e(on:end);
[cut, noise] = noise_truncation(e, fs);
e = max(e(1:cut) - noise, 0);
edc = flipud(cumsum(flipud(e)));
edc = 10*log10(edc / (edc(1) + eps) + eps);
end

function [cut, noise] = noise_truncation(e, fs)
n = numel(e);
tail = e(floor(0.9*n)+1:end);
win = max(1, round(0.01*fs));
half = floor(numel(tail)/2);
if half < win || abs(10*log10((mean(tail(1:half))+eps) / (mean(tail(half+1:end))+eps))) > 2
    cut = n; noise = 0; return;                 % tail still decaying: no noise floor
end
noise = mean(tail);
env = conv(e, ones(win,1)/win, 'same');
envDb = 10*log10(env + eps);
[~, pk] = max(envDb);
idx = find(envDb(pk:end) <= 10*log10(noise + eps) + 5, 1);
if isempty(idx), cut = n; else, cut = max(pk + idx - 1, 2*win); end
end

function E = late_energy(x, fs)
[cut, noise] = noise_truncation(x.^2, fs);
E = sum(max(x(1:cut).^2 - noise, 0));
end

function T = decay_time(edc, fs, startDb, endDb)
i0 = find(edc <= startDb, 1);
if isempty(i0), T = nan; return; end
i1 = find(edc(i0:end) <= endDb, 1);
if isempty(i1) || i1 < 5, T = nan; return; end
i1 = i0 + i1 - 1;
t = (i0:i1-1)' / fs;
p = polyfit(t, edc(i0:i1-1), 1);
if p(1) < 0, T = -60 / p(1); else, T = nan; end
end
