/*
 * Streaming log-mel front-end - C reference for embedded targets.
 *
 * Matches roomest/features.py:logmel_np (verified by tests/test_embedded.py).
 * - No dynamic allocation; state is a caller-owned struct (~6 KB).
 * - Constant work per hop: one 512-point FFT + sparse mel projection.
 * - Mean normalisation (one scalar over the analysis window) is applied by the
 *   caller over its ring buffer of frames right before inference, so per-frame
 *   cost stays constant.
 * A production port would swap lm_fft for CMSIS-DSP arm_rfft_fast_f32.
 */
#ifndef LOGMEL_H
#define LOGMEL_H
#include "mel_tables.h"

typedef struct {
    float re[LM_NFFT];
    float im[LM_NFFT];
    float power[LM_NFFT / 2 + 1];
} lm_state_t;

/* frame: LM_NFFT samples; out: LM_NMELS natural-log mel energies (un-normalised) */
void lm_frame(lm_state_t *st, const float *frame, float *out);

/* Convenience: full clip -> (n_mels x n_frames) row-major, mean-normalised.
 * Returns number of frames written (out must hold LM_NMELS * max_frames). */
int lm_clip(lm_state_t *st, const float *x, int n_samples, float *out, int max_frames);

#endif
