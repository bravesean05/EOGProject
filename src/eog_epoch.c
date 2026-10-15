/**
 * @file eog_epoch.c
 * @brief Epoch-based standardization and Savitzky-Golay smoothing.
 *
 * Each evaluation walks the epoch a few times: once to compute the mean, once
 * for the variance, once to standardize, and once inside the filter. That is
 * O(epoch) per evaluation and O(epoch/stride) per sample — with a 100-sample
 * epoch and stride 10, ten operations per sample. Cheap enough that the
 * incremental sum bookkeeping needed for a stride of one buys nothing here, and
 * a direct two-pass mean and variance is both simpler and more accurate: it
 * never subtracts two large nearly equal quantities the way sum_sq/n - mean^2
 * does on a signal with a large DC baseline.
 */

#include "eog_epoch.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eog_normalize.h"
#include "savgolFilter.h"

#define EOG_EPOCH_DEFAULT_MIN_STDDEV 1e-6f

struct EogEpoch {
    /* Configuration */
    size_t epoch_samples;
    size_t stride;
    int    sg_half_window;
    float  min_stddev;
    bool   sweep_centre;

    /* Trailing sample buffer, circular */
    float *buffer;
    size_t write_pos;
    size_t received;        /**< Total samples pushed */
    size_t since_eval;      /**< Samples pushed since the last evaluation */

    /* Scratch: epoch in chronological order, and the filtered result */
    float *epoch;
    float *filtered;

    /* Savitzky-Golay stage */
    SavgolFilter *filter;

    /* Longer-window deviation supplying the dividing scale. NULL when each
     * epoch is scaled by its own deviation. eog_normalize keeps this in O(1)
     * per sample, so the longer window costs no more than the short one. */
    EogNorm *scale;

    /* Centre subtracted at the end of the previous block, for the sweep */
    float  prev_centre;
    bool   have_prev_centre;

    /* Statistics from the most recent evaluation */
    float last_mean;
    float last_stddev;
    float last_scale;
};

/*============================================================================
 * INTERNAL
 *============================================================================*/

/** @brief Copy the circular buffer into `epoch` oldest-first. */
static void eog_epoch_linearize(EogEpoch *ep)
{
    /* Once full, the oldest sample sits at write_pos */
    size_t oldest = ep->write_pos;
    size_t n = ep->epoch_samples;
    size_t first = n - oldest;

    memcpy(ep->epoch, ep->buffer + oldest, first * sizeof(float));
    if (oldest > 0) {
        memcpy(ep->epoch + first, ep->buffer, oldest * sizeof(float));
    }
}

/**
 * @brief Standardize `epoch` in place, returning the divisor that was used.
 *
 * The centre is the epoch's own mean. The scale is either that epoch's
 * deviation or, when a longer scale window is configured, the deviation over
 * that window: holding the scale steady is what keeps a large movement looking
 * larger than a quiet stretch instead of normalizing both to the same spread.
 *
 * Two passes rather than running sums: the mean is subtracted before squaring,
 * so no large cancellation occurs even when the baseline dwarfs the signal.
 */
static double eog_epoch_standardize(EogEpoch *ep)
{
    size_t n = ep->epoch_samples;

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (double)ep->epoch[i];
    }
    double mean = sum / (double)n;

    double ss = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)ep->epoch[i] - mean;
        ss += d * d;
    }
    double sd = sqrt(ss / (double)(n - 1));   /* Bessel corrected */

    ep->last_mean   = (float)mean;
    ep->last_stddev = (float)sd;

    double scale = (ep->scale != NULL)
                   ? (double)eog_norm_stddev(ep->scale)
                   : sd;

    /* Guard a flat window, where every sample equals the mean */
    if (scale < (double)ep->min_stddev) {
        scale = (double)ep->min_stddev;
    }
    ep->last_scale = (float)scale;

    for (size_t i = 0; i < n; i++) {
        ep->epoch[i] = (float)(((double)ep->epoch[i] - mean) / scale);
    }
    return scale;
}

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

EogEpoch *eog_epoch_create(const EogEpochConfig *config)
{
    EogEpochConfig cfg = (config != NULL) ? *config : EOG_EPOCH_DEFAULT_CONFIG;

    if (cfg.stride == 0 || cfg.sg_half_window < 1) {
        return NULL;
    }

    /* The emitted region must lie inside the filter's central region: it spans
     * [epoch - stride - half, epoch - half), and the central region is
     * [half, epoch - half). */
    size_t half = (size_t)cfg.sg_half_window;
    if (cfg.epoch_samples < cfg.stride + 2 * half) {
        return NULL;
    }

    /* The filter itself needs the polynomial to fit inside its kernel */
    if (cfg.sg_poly_order < 0 ||
        (size_t)cfg.sg_poly_order >= 2 * half + 1) {
        return NULL;
    }

    /* A scale window shorter than the epoch would be measuring less than the
     * block it scales, which defeats the point of holding the scale steady */
    if (cfg.scale_window_samples != 0 &&
        cfg.scale_window_samples < cfg.epoch_samples) {
        return NULL;
    }

    EogEpoch *ep = (EogEpoch *)calloc(1, sizeof(EogEpoch));
    if (ep == NULL) {
        return NULL;
    }

    ep->epoch_samples  = cfg.epoch_samples;
    ep->stride         = cfg.stride;
    ep->sg_half_window = cfg.sg_half_window;
    ep->sweep_centre   = cfg.sweep_centre;
    ep->min_stddev     = (cfg.min_stddev > 0.0f)
                         ? cfg.min_stddev : EOG_EPOCH_DEFAULT_MIN_STDDEV;

    ep->buffer   = (float *)calloc(cfg.epoch_samples, sizeof(float));
    ep->epoch    = (float *)calloc(cfg.epoch_samples, sizeof(float));
    ep->filtered = (float *)calloc(cfg.epoch_samples, sizeof(float));
    if (ep->buffer == NULL || ep->epoch == NULL || ep->filtered == NULL) {
        eog_epoch_destroy(ep);
        return NULL;
    }

    SavgolConfig sg = SAVGOL_SMOOTH(cfg.sg_half_window, cfg.sg_poly_order);
    ep->filter = savgol_create(&sg);
    if (ep->filter == NULL) {
        eog_epoch_destroy(ep);
        return NULL;
    }

    if (cfg.scale_window_samples != 0) {
        ep->scale = eog_norm_create_samples(cfg.scale_window_samples);
        if (ep->scale == NULL) {
            eog_epoch_destroy(ep);
            return NULL;
        }
    }

    return ep;
}

void eog_epoch_destroy(EogEpoch *ep)
{
    if (ep == NULL) {
        return;
    }
    eog_norm_destroy(ep->scale);
    savgol_destroy(ep->filter);
    free(ep->filtered);
    free(ep->epoch);
    free(ep->buffer);
    free(ep);
}

void eog_epoch_reset(EogEpoch *ep)
{
    if (ep == NULL) {
        return;
    }
    ep->write_pos        = 0;
    ep->received         = 0;
    ep->since_eval       = 0;
    ep->last_mean        = 0.0f;
    ep->last_stddev      = 0.0f;
    ep->last_scale       = 0.0f;
    ep->prev_centre      = 0.0f;
    ep->have_prev_centre = false;
    eog_norm_reset(ep->scale);
    memset(ep->buffer, 0, ep->epoch_samples * sizeof(float));
}

/*============================================================================
 * STREAMING OPERATION
 *============================================================================*/

int eog_epoch_push(EogEpoch *ep, float sample, float *out, int max_out)
{
    if (ep == NULL || out == NULL || max_out < 0) {
        return -1;
    }

    ep->buffer[ep->write_pos] = sample;
    ep->write_pos = (ep->write_pos + 1) % ep->epoch_samples;
    ep->received++;
    ep->since_eval++;

    /* The longer scale window tracks every sample, not just block boundaries */
    if (ep->scale != NULL) {
        eog_norm_push(ep->scale, sample, NULL);
    }

    /* Wait for a full epoch, then evaluate once per stride */
    if (ep->received < ep->epoch_samples || ep->since_eval < ep->stride) {
        return 0;
    }
    ep->since_eval = 0;

    if ((size_t)max_out < ep->stride) {
        return -1;
    }

    eog_epoch_linearize(ep);
    double scale = eog_epoch_standardize(ep);

    if (savgol_apply(ep->filter, ep->epoch, ep->filtered,
                     ep->epoch_samples) != 0) {
        return -1;
    }

    /* Step back by half the kernel so the emitted block sits in the central
     * region rather than on the trailing edge */
    size_t half  = (size_t)ep->sg_half_window;
    size_t begin = ep->epoch_samples - ep->stride - half;

    double mean = (double)ep->last_mean;

    if (!ep->sweep_centre || !ep->have_prev_centre) {
        memcpy(out, ep->filtered + begin, ep->stride * sizeof(float));
    } else {
        /* Re-centre the block onto a line running from the previous block's
         * centre to this one's.
         *
         * Subtracting a centre and the filter are both linear, and a smoothing
         * filter reproduces a constant exactly inside its central region,
         * where this block lies. So shifting the centre after filtering gives
         * the same result as having filtered against the shifted centre, and
         * costs one add per sample instead of a second pass.
         */
        double prev = (double)ep->prev_centre;
        for (size_t k = 0; k < ep->stride; k++) {
            double frac = (double)(k + 1) / (double)ep->stride;
            double centre = prev + (mean - prev) * frac;
            out[k] = (float)((double)ep->filtered[begin + k]
                             + (mean - centre) / scale);
        }
    }

    ep->prev_centre = (float)mean;
    ep->have_prev_centre = true;
    return (int)ep->stride;
}

/*============================================================================
 * STATE QUERIES
 *============================================================================*/

size_t eog_epoch_output_block(const EogEpoch *ep)
{
    return (ep != NULL) ? ep->stride : 0;
}

bool eog_epoch_ready(const EogEpoch *ep)
{
    return (ep != NULL) && (ep->received >= ep->epoch_samples);
}

size_t eog_epoch_latency_min(const EogEpoch *ep)
{
    return (ep != NULL) ? (size_t)ep->sg_half_window : 0;
}

size_t eog_epoch_latency_max(const EogEpoch *ep)
{
    if (ep == NULL) {
        return 0;
    }
    return (size_t)ep->sg_half_window + ep->stride - 1;
}

float eog_epoch_last_mean(const EogEpoch *ep)
{
    return (ep != NULL) ? ep->last_mean : 0.0f;
}

float eog_epoch_last_stddev(const EogEpoch *ep)
{
    return (ep != NULL) ? ep->last_stddev : 0.0f;
}

float eog_epoch_last_scale(const EogEpoch *ep)
{
    return (ep != NULL) ? ep->last_scale : 0.0f;
}
