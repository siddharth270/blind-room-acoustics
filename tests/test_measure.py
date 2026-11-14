import numpy as np
from scipy import signal

from roomest.acoustics import synthetic_rir, t60_from_edc, schroeder_edc_db
from roomest.measure import deconvolve, ess


def test_ess_deconvolution_recovers_rir():
    fs = 16000
    sweep, inv = ess(40, 7000, 2.0, fs)
    h = synthetic_rir(0.4, fs, 1.0)
    rec = signal.fftconvolve(sweep, h)
    rec += np.random.default_rng(0).standard_normal(len(rec)) * 1e-5
    est = deconvolve(rec, sweep, inv, len(h))
    assert np.argmax(np.abs(est)) == 0
    t_true = t60_from_edc(schroeder_edc_db(h, fs), fs)
    t_est = t60_from_edc(schroeder_edc_db(est, fs), fs)
    assert abs(t_est - t_true) / t_true < 0.05
