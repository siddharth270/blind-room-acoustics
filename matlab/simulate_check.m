% SIMULATE_CHECK  Sanity-check analyze_rir on a synthetic RIR with known T60.
% Generates exponentially decaying noise (T60 = 0.6 s), writes it to disk and
% verifies the MATLAB analysis recovers T60 within 5 %. Mirrors the Python unit
% test tests/test_acoustics.py::test_t60_recovered.
fs = 16000;  T60 = 0.6;  n = 2*fs;
t = (0:n-1)'/fs;
rng(0);
h = randn(n,1) .* exp(-3*log(10)*t/T60);
h(1:round(0.003*fs)) = 0;
h(1) = sqrt(sum(h.^2));                         % DRR = 0 dB
f = [tempname '.wav'];
audiowrite(f, 0.9*h/max(abs(h)), fs, 'BitsPerSample', 32);
p = analyze_rir(f, false);
fprintf('mid-band T60 estimate %.3f s (true %.3f s)\n', mean(p.T60(3:4)), T60);
assert(abs(mean(p.T60(3:4)) - T60) / T60 < 0.05, 'T60 check failed');
assert(abs(p.DRR) < 1.0, 'DRR check failed');
disp('OK');
