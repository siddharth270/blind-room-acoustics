#include "logmel.h"
#include <math.h>

#define LM_EPS 1e-8f
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* In-place iterative radix-2 complex FFT (N = LM_NFFT, power of two). */
static void lm_fft(float *re, float *im) {
    const int n = LM_NFFT;
    for (int i = 1, j = 0; i < n; i++) {           /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / len;
        const float wr = (float)cos(ang), wi = (float)sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                const int a = i + k, b = i + k + len / 2;
                const float tr = re[b] * cr - im[b] * ci;
                const float ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                const float nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
}

void lm_frame(lm_state_t *st, const float *frame, float *out) {
    for (int i = 0; i < LM_NFFT; i++) {
        st->re[i] = frame[i] * LM_WINDOW[i];
        st->im[i] = 0.0f;
    }
    lm_fft(st->re, st->im);
    for (int k = 0; k <= LM_NFFT / 2; k++)
        st->power[k] = st->re[k] * st->re[k] + st->im[k] * st->im[k];
    int w = 0;
    for (int m = 0; m < LM_NMELS; m++) {
        float acc = 0.0f;
        const int s = LM_MEL_START[m];
        for (int j = 0; j < LM_MEL_LEN[m]; j++) acc += LM_MEL_W[w + j] * st->power[s + j];
        w += LM_MEL_LEN[m];
        out[m] = logf(acc + LM_EPS);
    }
}

int lm_clip(lm_state_t *st, const float *x, int n_samples, float *out, int max_frames) {
    int n = 1 + (n_samples - LM_NFFT) / LM_HOP;
    if (n > max_frames) n = max_frames;
    float frame_out[LM_NMELS];
    double sum = 0.0;
    for (int t = 0; t < n; t++) {
        lm_frame(st, x + t * LM_HOP, frame_out);
        for (int m = 0; m < LM_NMELS; m++) {
            out[m * n + t] = frame_out[m];
            sum += frame_out[m];
        }
    }
    const float mean = (float)(sum / ((double)n * LM_NMELS));
    for (int i = 0; i < n * LM_NMELS; i++) out[i] -= mean;
    return n;
}
