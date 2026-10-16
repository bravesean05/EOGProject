/**
 * @file eog_cnn.c
 * @brief 1-D CNN inference, matching the MATLAB network layer for layer.
 *
 * The sequence, with batch normalization already folded into each convolution
 * by the exporter and dropout omitted as it only acts during training:
 *
 *   input  L x 2
 *   conv k7 -> relu -> maxpool 2     L   x 32  ->  L/2 x 32
 *   conv k5 -> relu -> maxpool 2     L/2 x 64  ->  L/4 x 64
 *   conv k3 -> relu                  L/4 x 128
 *   global average over time         128
 *   fully connected -> softmax       6
 *
 * Convolutions are cross-correlations with zero padding, which is what
 * MATLAB's convolution1dLayer computes; a flipped kernel would still train but
 * would not reproduce MATLAB's output, and the test vectors would catch it.
 *
 * Activations are stored time-major with channels fastest, the same layout as
 * the input, so every layer walks memory in order.
 */

/* The weights header defines its arrays only in the one file that asks, so
 * other files can include eog_cnn.h for the sizes without each carrying a
 * copy of every weight. */
#define EOG_CNN_EMIT_WEIGHTS
#include "eog_cnn.h"

#include <math.h>
#include <stddef.h>

/** Geometry after each stage, derived rather than restated. */
#define L0 EOG_CNN_INPUT_LEN
#define L1 (L0 / 2)
#define L2 (L1 / 2)

#if (L0 % 4) != 0
#error "EOG_CNN_INPUT_LEN must be divisible by 4 for the two 2x poolings"
#endif

/* Every activation reuses the two work buffers, which are sized for the first
 * conv's output. That holds while each block halves the length at least as
 * fast as it grows the channels; a regenerated network that breaks it must
 * fail here rather than overrun at run time. */
#if (L1 * EOG_CNN_C2_COUT) > EOG_CNN_WORK_FLOATS
#error "Second conv output does not fit the work buffers"
#endif
#if (L2 * EOG_CNN_C3_COUT) > EOG_CNN_WORK_FLOATS
#error "Third conv output does not fit the work buffers"
#endif
#if EOG_CNN_FC_IN != EOG_CNN_C3_COUT
#error "Fully connected input must match the third conv's channels"
#endif

/**
 * @brief Same-padded convolution with ReLU, one output time step at a time.
 *
 * @param in    len x cin, time-major
 * @param out   len x cout, time-major
 * @param w     cout x k x cin
 */
static void conv_relu(const float *in, int len, int cin,
                      const float *w, const float *bias,
                      int k, int cout, int pad, float *out)
{
    for (int t = 0; t < len; t++) {
        /* Taps that would fall outside the input read zero; skipping them is
         * the same as multiplying by a padded zero */
        int k_lo = pad - t;
        if (k_lo < 0) k_lo = 0;
        int k_hi = len + pad - t;
        if (k_hi > k) k_hi = k;

        float *o = out + (size_t)t * cout;
        for (int co = 0; co < cout; co++) {
            float acc = bias[co];
            const float *wc = w + (size_t)co * k * cin;
            for (int j = k_lo; j < k_hi; j++) {
                const float *x = in + (size_t)(t + j - pad) * cin;
                const float *wk = wc + (size_t)j * cin;
                for (int ci = 0; ci < cin; ci++) {
                    acc += wk[ci] * x[ci];
                }
            }
            o[co] = (acc > 0.0f) ? acc : 0.0f;
        }
    }
}

/** @brief Max over non-overlapping pairs in time: len x ch -> len/2 x ch. */
static void maxpool2(const float *in, int len, int ch, float *out)
{
    for (int t = 0; t < len / 2; t++) {
        const float *a = in + (size_t)(2 * t) * ch;
        const float *b = a + ch;
        float *o = out + (size_t)t * ch;
        for (int c = 0; c < ch; c++) {
            o[c] = (a[c] > b[c]) ? a[c] : b[c];
        }
    }
}

int eog_cnn_classify(const float *window, float *probs, EogCnnWork *work)
{
    if (window == NULL || work == NULL) {
        return -1;
    }

    float *a = work->a;
    float *b = work->b;

    /* block 1 */
    conv_relu(window, L0, EOG_CNN_INPUT_CH,
              eog_cnn_c1_w, eog_cnn_c1_b,
              EOG_CNN_C1_K, EOG_CNN_C1_COUT, EOG_CNN_C1_PAD, a);
    maxpool2(a, L0, EOG_CNN_C1_COUT, b);

    /* block 2 */
    conv_relu(b, L1, EOG_CNN_C1_COUT,
              eog_cnn_c2_w, eog_cnn_c2_b,
              EOG_CNN_C2_K, EOG_CNN_C2_COUT, EOG_CNN_C2_PAD, a);
    maxpool2(a, L1, EOG_CNN_C2_COUT, b);

    /* block 3, no pooling */
    conv_relu(b, L2, EOG_CNN_C2_COUT,
              eog_cnn_c3_w, eog_cnn_c3_b,
              EOG_CNN_C3_K, EOG_CNN_C3_COUT, EOG_CNN_C3_PAD, a);

    /* global average over time */
    float feat[EOG_CNN_C3_COUT];
    for (int c = 0; c < EOG_CNN_C3_COUT; c++) {
        feat[c] = 0.0f;
    }
    for (int t = 0; t < L2; t++) {
        const float *row = a + (size_t)t * EOG_CNN_C3_COUT;
        for (int c = 0; c < EOG_CNN_C3_COUT; c++) {
            feat[c] += row[c];
        }
    }
    for (int c = 0; c < EOG_CNN_C3_COUT; c++) {
        feat[c] /= (float)L2;
    }

    /* fully connected */
    float logit[EOG_CNN_NUM_CLASSES];
    int best = 0;
    for (int o = 0; o < EOG_CNN_NUM_CLASSES; o++) {
        float acc = eog_cnn_fc_b[o];
        const float *wo = eog_cnn_fc_w + (size_t)o * EOG_CNN_FC_IN;
        for (int i = 0; i < EOG_CNN_FC_IN; i++) {
            acc += wo[i] * feat[i];
        }
        logit[o] = acc;
        if (acc > logit[best]) {
            best = o;
        }
    }

    /* softmax, shifted by the largest logit so exp cannot overflow */
    if (probs != NULL) {
        float sum = 0.0f;
        for (int o = 0; o < EOG_CNN_NUM_CLASSES; o++) {
            probs[o] = expf(logit[o] - logit[best]);
            sum += probs[o];
        }
        for (int o = 0; o < EOG_CNN_NUM_CLASSES; o++) {
            probs[o] /= sum;
        }
    }

    return best;
}

const char *eog_cnn_class_name(int index)
{
    if (index < 0 || index >= EOG_CNN_NUM_CLASSES) {
        return "?";
    }
    return eog_cnn_class_names[index];
}
