function [sweep, inv] = generate_ess(f1, f2, T, fs)
%GENERATE_ESS Exponential sine sweep and its inverse filter (Farina, 2000).
%   [sweep, inv] = generate_ess(f1, f2, T, fs)
%   f1, f2 : start / stop frequency (Hz);  T : sweep duration (s);  fs : sample rate
%   conv(sweep, inv) is a band-limited delta of unit peak at index length(sweep).
%   Mirrors roomest/measure.py:ess so MATLAB and Python measurements are identical.

n = round(T * fs);
t = (0:n-1)' / fs;
R = log(f2 / f1);
sweep = sin(2*pi*f1*T/R * (exp(t*R/T) - 1));

nf = round(0.05 * fs);                          % 50 ms raised-cosine fades
fade = 0.5 - 0.5*cos(pi*(0:nf-1)'/nf);
sweep(1:nf) = sweep(1:nf) .* fade;
sweep(end-nf+1:end) = sweep(end-nf+1:end) .* flipud(fade);

inv = flipud(sweep) .* exp(-t*R/T);             % -6 dB/oct amplitude compensation
inv = inv / max(abs(fftconv(sweep, inv)));
end

function y = fftconv(a, b)
L = length(a) + length(b) - 1;
N = 2^nextpow2(L);
y = real(ifft(fft(a, N) .* fft(b, N)));
y = y(1:L);
end
