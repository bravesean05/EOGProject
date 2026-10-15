/**
 * @file eog_normalize.c
 * @brief Rolling-window z-score standardization — implementation.
 *
 * Statistics are maintained incrementally so each sample costs O(1) rather than
 * O(window). Two numerical precautions matter more than the bookkeeping:
 *
 * 1. Offset compensation. Variance from raw running sums is computed as
 *    sum_sq/n - mean^2, which subtracts two large nearly equal quantities when
 *    the signal carries a large DC baseline and a small AC component — exactly
 *    the shape of an electrode recording. Accumulating (x - offset) instead,
 *    with offset fixed at the first sample of the current accumulation, keeps
 *    both terms small and the subtraction well conditioned. The offset cancels
 *    out of the variance entirely and is added back into the mean.
 *
 * 2. Periodic reaccumulation. Adding and subtracting in floating point leaves
 *    residue that grows without bound over a long recording. Every window
 *    length of samples the sums are rebuilt from the buffer, which costs O(n)
 *    once per n samples — O(1) amortized — and caps the error at whatever one
 *    window's worth of updates can accumulate.
 *
 * Sums are kept in double regardless of the float sample type, which costs
 * nothing on the target CPUs and removes the bulk of the drift on its own.
 */

#include "eog_normalize.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/** Default floor for the standard deviation divisor. */
#define EOG_NORM_DEFAULT_MIN_STDDEV 1e-6f

struct EogNorm {
    float  *buffer;         /**< Circular buffer of the trailing window */
    size_t  capacity;       /**< Window length in samples */
    size_t  count;          /**< Samples currently held (<= capacity) */
    size_t  write_pos;      /**< Index the next sample is written to */

    double  offset;         /**< Shift subtracted before accumulating */
    double  sum;            /**< Sum of (x - offset) over the window */
    double  sum_sq;         /**< Sum of (x - offset)^2 over the window */

    size_t  since_rebuild;  /**< Samples pushed since the last reaccumulation */

    float   min_stddev;     /**< Divisor floor */
    bool    strict;         /**< Suppress output until the window is full */
};

/*============================================================================
 * INTERNAL
 *============================================================================*/

/**
 * @brief Rebuild sum and sum_sq from the buffer, discarding accumulated error.
 *
 * Also re-centres the offset on the oldest sample currently held, so the
 * accumulated values stay small as the signal's baseline drifts.
 */
static void eog_norm_rebuild(EogNorm *norm)
{
    norm->since_rebuild = 0;

    if (norm->count == 0) {
        norm->offset = 0.0;
        norm->sum    = 0.0;
        norm->sum_sq = 0.0;
        return;
    }

    /* Oldest sample in the window: the slot about to be overwritten once the
     * buffer is full, or slot 0 while it is still filling. */
    size_t oldest = (norm->count == norm->capacity) ? norm->write_pos : 0;
    norm->offset = (double)norm->buffer[oldest];

    double sum = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < norm->count; i++) {
        size_t idx = (norm->count == norm->capacity)
                     ? (norm->write_pos + i) % norm->capacity
                     : i;
        double d = (double)norm->buffer[idx] - norm->offset;
        sum    += d;
        sum_sq += d * d;
    }
    norm->sum    = sum;
    norm->sum_sq = sum_sq;
}

/** @brief Population/sample variance over the window, clamped at zero. */
static double eog_norm_variance(const EogNorm *norm)
{
    if (norm->count < 2) {
        return 0.0;
    }
    double n = (double)norm->count;
    /* Sum of squared deviations, with the offset cancelling out */
    double ss = norm->sum_sq - (norm->sum * norm->sum) / n;
    if (ss < 0.0) {
        ss = 0.0;  /* rounding can push a near-zero result slightly negative */
    }
    return ss / (n - 1.0);  /* Bessel corrected */
}

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

EogNorm *eog_norm_create_samples(size_t window_samples)
{
    if (window_samples < 2) {
        return NULL;
    }

    EogNorm *norm = (EogNorm *)calloc(1, sizeof(EogNorm));
    if (norm == NULL) {
        return NULL;
    }

    norm->buffer = (float *)calloc(window_samples, sizeof(float));
    if (norm->buffer == NULL) {
        free(norm);
        return NULL;
    }

    norm->capacity   = window_samples;
    norm->min_stddev = EOG_NORM_DEFAULT_MIN_STDDEV;
    norm->strict     = false;
    return norm;
}

EogNorm *eog_norm_create(float sample_rate_hz, float window_seconds)
{
    if (!(sample_rate_hz > 0.0f) || !(window_seconds > 0.0f)) {
        return NULL;
    }

    double exact = (double)sample_rate_hz * (double)window_seconds;
    if (exact > (double)SIZE_MAX) {
        return NULL;
    }

    /* Nudge by a relative epsilon before rounding. A duration the caller writes
     * as a decimal rarely survives conversion to float exactly: 0.410f is
     * really 0.40999999..., so 250 Hz x 0.410 s evaluates to 102.4999991 and
     * would round down to 102 despite the decimal arithmetic giving 102.5. The
     * epsilon is far smaller than any meaningful difference in window length,
     * so it only decides cases already sitting on a rounding boundary. */
    size_t n = (size_t)(exact * (1.0 + 1e-6) + 0.5);
    return eog_norm_create_samples(n);
}

void eog_norm_destroy(EogNorm *norm)
{
    if (norm == NULL) {
        return;
    }
    free(norm->buffer);
    free(norm);
}

void eog_norm_reset(EogNorm *norm)
{
    if (norm == NULL) {
        return;
    }
    norm->count         = 0;
    norm->write_pos     = 0;
    norm->offset        = 0.0;
    norm->sum           = 0.0;
    norm->sum_sq        = 0.0;
    norm->since_rebuild = 0;
    memset(norm->buffer, 0, norm->capacity * sizeof(float));
}

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

void eog_norm_set_strict(EogNorm *norm, bool strict)
{
    if (norm != NULL) {
        norm->strict = strict;
    }
}

void eog_norm_set_min_stddev(EogNorm *norm, float min_stddev)
{
    if (norm != NULL && min_stddev > 0.0f) {
        norm->min_stddev = min_stddev;
    }
}

/*============================================================================
 * STREAMING OPERATION
 *============================================================================*/

float eog_norm_push(EogNorm *norm, float sample, bool *output_valid)
{
    if (output_valid != NULL) {
        *output_valid = false;
    }
    if (norm == NULL) {
        return 0.0f;
    }

    /* First sample of an empty window anchors the offset */
    if (norm->count == 0) {
        norm->offset = (double)sample;
        norm->sum    = 0.0;
        norm->sum_sq = 0.0;
    }

    if (norm->count == norm->capacity) {
        /* Window is full: the slot being overwritten leaves the window */
        double leaving = (double)norm->buffer[norm->write_pos] - norm->offset;
        norm->sum    -= leaving;
        norm->sum_sq -= leaving * leaving;
    } else {
        norm->count++;
    }

    double entering = (double)sample - norm->offset;
    norm->sum    += entering;
    norm->sum_sq += entering * entering;

    norm->buffer[norm->write_pos] = sample;
    norm->write_pos = (norm->write_pos + 1) % norm->capacity;

    /* Amortized O(1): one O(n) pass per n samples, bounding drift */
    if (++norm->since_rebuild >= norm->capacity) {
        eog_norm_rebuild(norm);
    }

    /* Not enough samples for a standard deviation, or caller wants a full
     * window before trusting the statistics */
    if (norm->count < 2 || (norm->strict && norm->count < norm->capacity)) {
        return 0.0f;
    }

    double mean = norm->offset + norm->sum / (double)norm->count;
    double sd   = sqrt(eog_norm_variance(norm));
    if (sd < (double)norm->min_stddev) {
        sd = (double)norm->min_stddev;
    }

    if (output_valid != NULL) {
        *output_valid = true;
    }
    return (float)(((double)sample - mean) / sd);
}

/*============================================================================
 * STATE QUERIES
 *============================================================================*/

size_t eog_norm_window_length(const EogNorm *norm)
{
    return (norm != NULL) ? norm->capacity : 0;
}

size_t eog_norm_count(const EogNorm *norm)
{
    return (norm != NULL) ? norm->count : 0;
}

bool eog_norm_ready(const EogNorm *norm)
{
    return (norm != NULL) && (norm->count == norm->capacity);
}

float eog_norm_mean(const EogNorm *norm)
{
    if (norm == NULL || norm->count == 0) {
        return 0.0f;
    }
    return (float)(norm->offset + norm->sum / (double)norm->count);
}

float eog_norm_stddev(const EogNorm *norm)
{
    if (norm == NULL) {
        return 0.0f;
    }
    return (float)sqrt(eog_norm_variance(norm));
}
