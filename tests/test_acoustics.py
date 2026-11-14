import numpy as np
import pytest

from roomest.acoustics import (band_t60, clarity, compute_labels, drr, edt, schroeder_edc_db,
                               synthetic_rir, t60_from_edc)

FS = 16000


@pytest.mark.parametrize("t60", [0.25, 0.6, 1.2])
def test_t60_recovered(t60):
    h = synthetic_rir(t60, FS, length_s=3 * t60 + 0.5)
    est = t60_from_edc(schroeder_edc_db(h, FS), FS)
    assert abs(est - t60) / t60 < 0.05


def test_band_t60_flat_decay():
    h = synthetic_rir(0.5, FS, 2.0)
    est = band_t60(h, FS)
    assert np.all(np.abs(est - 0.5) / 0.5 < 0.1)


@pytest.mark.parametrize("target", [-6.0, 0.0, 6.0])
def test_drr_recovered(target):
    h = synthetic_rir(0.5, FS, 2.0, drr_db=target)
    assert abs(drr(h, FS) - target) < 0.5


def test_c50_matches_analytic():
    # pure exponential decay: C50 = 10log10(exp(k*0.05) - 1), k = 6 ln10 / T60
    t60 = 0.5
    t = np.arange(2 * FS) / FS
    h = np.exp(-3 * np.log(10) * t / t60)  # deterministic envelope, energy ~ exp(-k t)
    k = 6 * np.log(10) / t60
    assert abs(clarity(h, FS) - 10 * np.log10(np.exp(k * 0.05) - 1)) < 0.3


def test_edt_equals_t60_for_single_slope():
    h = synthetic_rir(0.8, FS, 3.0)
    assert abs(edt(h, FS) - 0.8) / 0.8 < 0.1


def test_noise_compensation_handles_noise_floor():
    rng = np.random.default_rng(1)
    h = synthetic_rir(0.6, FS, 2.5)
    tail_level = np.std(h[50:500])  # reverberant tail level at onset (not the direct peak)
    noisy = h + rng.standard_normal(len(h)) * tail_level * 10 ** (-50 / 20)
    plain = t60_from_edc(schroeder_edc_db(noisy, FS, noise_comp=False), FS)
    comp = t60_from_edc(schroeder_edc_db(noisy, FS, noise_comp=True), FS)
    assert abs(comp - 0.6) < abs(plain - 0.6) + 1e-6
    assert abs(comp - 0.6) / 0.6 < 0.08


def test_noise_compensation_leaves_truncated_rirs_alone():
    # an RIR truncated while still decaying must not be "noise-compensated"
    h = synthetic_rir(0.6, FS, 0.5)
    a = schroeder_edc_db(h, FS, noise_comp=False)
    b = schroeder_edc_db(h, FS, noise_comp=True)
    assert np.allclose(a, b)


def test_label_vector_shape_and_volume():
    lab = compute_labels(synthetic_rir(0.5, FS, 2.0), FS, volume=100.0)
    assert lab.shape == (9,) and np.isclose(lab[-1], 2.0)
    assert np.isnan(compute_labels(synthetic_rir(0.5, FS, 2.0), FS)[-1])
