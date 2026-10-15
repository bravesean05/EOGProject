/**
 * @file eog_epoch.h
 * @brief Epoch-based preprocessing: rolling z-score followed by Savitzky-Golay.
 *
 * Collects samples into a trailing epoch of fixed length and reprocesses that
 * epoch every `stride` samples. Both stages run over the same epoch:
 *
 *   1. Standardization — z-score using the mean and standard deviation of the
 *      epoch itself, so a drifting electrode baseline is removed continuously.
 *   2. Savitzky-Golay — smoothing across the standardized epoch, using a kernel
 *      of 2*sg_half_window+1 samples that slides within the epoch.
 *
 * The epoch length and the Savitzky-Golay kernel are different things. The epoch
 * (say 100 samples) is the block handed to the filter; the kernel (say 11
 * samples) is the neighbourhood the filter consults around each point inside
 * that block. Setting the kernel to the epoch length would fit one polynomial
 * across the whole block, which flattens exactly the movements this is meant to
 * expose.
 *
 * Output region
 * -------------
 * Consecutive epochs overlap by epoch_samples - stride, so each sample falls in
 * several epochs. Only one epoch's result is kept per sample: the `stride`
 * samples ending sg_half_window short of the newest sample. Stepping back that
 * far keeps every emitted sample inside the filter's central region, where full
 * symmetric windows apply, instead of the trailing edge where the filter falls
 * back to asymmetric edge weights.
 *
 * Emitted samples therefore lag acquisition by sg_half_window to
 * sg_half_window + stride - 1 samples, and arrive in bursts of `stride` once per
 * stride samples. See eog_epoch_latency_min() and eog_epoch_latency_max().
 *
 * What this does to sustained deflections
 * ---------------------------------------
 * Standardizing against the epoch's own mean means a deflection held longer than
 * the epoch is absorbed into that mean and fades from the output. The pipeline
 * reports eye *movements*, not absolute eye *position*. That is the intent here;
 * it is the wrong tool if gaze angle has to survive.
 */

#ifndef EOG_EPOCH_H
#define EOG_EPOCH_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque pipeline state. */
typedef struct EogEpoch EogEpoch;

/** Pipeline configuration. */
typedef struct {
    size_t epoch_samples;   /**< Trailing epoch length. */
    size_t stride;          /**< Samples between successive epoch evaluations. */
    int    sg_half_window;  /**< Savitzky-Golay half window. Kernel is 2n+1. */
    int    sg_poly_order;   /**< Savitzky-Golay polynomial order. */
    float  min_stddev;      /**< Divisor floor for the z-score. 0 selects 1e-6. */

    /**
     * Trailing window the dividing scale is measured over, in samples.
     *
     * 0 scales each epoch by its own deviation. That normalizes every window
     * to the same spread, so a quiet stretch comes out as lively as a real
     * movement and the amplitude separating one movement from another is lost.
     *
     * A window several seconds long holds the scale steady while the baseline
     * still moves, which keeps that amplitude. Must be at least epoch_samples.
     */
    size_t scale_window_samples;

    /**
     * Sweep the subtracted centre across each emitted block.
     *
     * Subtracting one epoch's mean from a whole block leaves a step wherever
     * two blocks meet, recurring once per stride. Walking the centre from the
     * previous block's mean to this one's across the block removes that seam
     * without lagging the baseline the way a smoothed mean would.
     */
    bool sweep_centre;
} EogEpochConfig;

/**
 * @brief Default configuration: 100-sample epoch, stride 10, kernel 11, order 2,
 *        with a 2560-sample scale window and the centre swept.
 *
 * At 256 Hz that is a 391 ms epoch re-evaluated every 39 ms, a 43 ms smoothing
 * kernel, and a 10 s scale window.
 *
 * Measured on one subject of the Barbara et al. recordings, against scaling by
 * each epoch's own deviation: saccades stand out from blinks by 5.6x rather
 * than 1.0x, the seam between blocks disappears, and the output during blinks,
 * where the horizontal channel should be calm, settles by a factor of 3.7.
 */
#define EOG_EPOCH_DEFAULT_CONFIG \
    (EogEpochConfig){ .epoch_samples = 100, .stride = 10, \
                      .sg_half_window = 5, .sg_poly_order = 2, \
                      .min_stddev = 0.0f, \
                      .scale_window_samples = 2560, \
                      .sweep_centre = true }

/**
 * @brief Configuration scaling each epoch by its own deviation, with no sweep.
 *
 * The plain rolling z-score. Every window is normalized to the same spread,
 * which is the right thing when only shape matters and amplitude does not.
 */
#define EOG_EPOCH_PER_EPOCH_CONFIG \
    (EogEpochConfig){ .epoch_samples = 100, .stride = 10, \
                      .sg_half_window = 5, .sg_poly_order = 2, \
                      .min_stddev = 0.0f, \
                      .scale_window_samples = 0, \
                      .sweep_centre = false }

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

/**
 * @brief Create a pipeline.
 *
 * Requires epoch_samples >= stride + 2*sg_half_window, so the emitted region
 * lies wholly inside the filter's central region, and sg_poly_order <
 * 2*sg_half_window+1 as the filter itself demands.
 *
 * @param config  Configuration. NULL selects EOG_EPOCH_DEFAULT_CONFIG.
 * @return Handle, or NULL on invalid configuration or allocation failure.
 */
EogEpoch *eog_epoch_create(const EogEpochConfig *config);

/**
 * @brief Destroy a pipeline.
 * @param ep  Handle. NULL is safe (no-op).
 */
void eog_epoch_destroy(EogEpoch *ep);

/**
 * @brief Discard buffered samples and restart epoch accounting.
 * @param ep  Handle.
 */
void eog_epoch_reset(EogEpoch *ep);

/*============================================================================
 * STREAMING OPERATION
 *============================================================================*/

/**
 * @brief Push one raw sample.
 *
 * Returns 0 for most calls. Once the epoch has filled, every `stride`-th call
 * evaluates the epoch and writes `stride` processed samples to `out`, oldest
 * first.
 *
 * @param ep       Handle.
 * @param sample   Raw input sample.
 * @param out      Destination for processed samples. Needs room for `stride`.
 * @param max_out  Capacity of `out`.
 * @return Samples written (0 or stride), or -1 on error.
 */
int eog_epoch_push(EogEpoch *ep, float sample, float *out, int max_out);

/*============================================================================
 * STATE QUERIES
 *============================================================================*/

/** @brief Samples emitted per evaluation, equal to the configured stride. */
size_t eog_epoch_output_block(const EogEpoch *ep);

/** @brief True once a full epoch has been collected. */
bool eog_epoch_ready(const EogEpoch *ep);

/** @brief Smallest lag, in samples, between acquisition and emission. */
size_t eog_epoch_latency_min(const EogEpoch *ep);

/** @brief Largest lag, in samples, between acquisition and emission. */
size_t eog_epoch_latency_max(const EogEpoch *ep);

/** @brief Mean of the most recently evaluated epoch. */
float eog_epoch_last_mean(const EogEpoch *ep);

/**
 * @brief Deviation of the most recently evaluated epoch.
 *
 * Always the epoch's own deviation, whatever scale_window_samples is set to.
 * This is the movement-energy feature: on these recordings it averages 44
 * during saccades against 9 during blinks on the horizontal channel, and the
 * other way around on the vertical one.
 */
float eog_epoch_last_stddev(const EogEpoch *ep);

/**
 * @brief Scale the most recent block was divided by.
 *
 * The same value as eog_epoch_last_stddev() when scale_window_samples is 0,
 * and the deviation over the longer window otherwise.
 */
float eog_epoch_last_scale(const EogEpoch *ep);

#ifdef __cplusplus
}
#endif

#endif /* EOG_EPOCH_H */
