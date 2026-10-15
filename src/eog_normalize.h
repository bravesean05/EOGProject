/**
 * @file eog_normalize.h
 * @brief Rolling-window z-score standardization for real-time signals.
 *
 * Standardizes a sample stream as (x - mean) / stddev, where mean and stddev
 * come from a trailing window of the most recent N samples rather than from
 * cumulative statistics. The window slides with every sample, so the filter
 * follows a drifting baseline instead of being anchored to the start of the
 * recording.
 *
 * The window is trailing (causal), never centered: at sample t it spans
 * [t-N+1, t]. Standardization therefore adds no latency of its own, unlike the
 * Savitzky-Golay stage downstream.
 *
 * Intended use is as a preprocessing stage feeding savgol_stream_push():
 *
 *     float z = eog_norm_push(norm, raw_sample, &z_valid);
 *     if (z_valid) {
 *         float smoothed = savgol_stream_push(stream, z, &out_valid);
 *     }
 *
 * Cost is O(1) per sample: statistics are updated incrementally rather than
 * recomputed over the window.
 */

#ifndef EOG_NORMALIZE_H
#define EOG_NORMALIZE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque rolling-window standardizer state. */
typedef struct EogNorm EogNorm;

/*============================================================================
 * LIFECYCLE
 *============================================================================*/

/**
 * @brief Create a standardizer sized from a sampling rate and window duration.
 *
 * The window length in samples is round(sample_rate_hz * window_seconds), and
 * must come out to at least 2.
 *
 * @param sample_rate_hz  Sampling rate of the incoming signal, in Hz. > 0.
 * @param window_seconds  Trailing window duration, in seconds. > 0.
 * @return Handle, or NULL on invalid arguments or allocation failure.
 */
EogNorm *eog_norm_create(float sample_rate_hz, float window_seconds);

/**
 * @brief Create a standardizer with an explicit window length in samples.
 *
 * @param window_samples  Trailing window length. Must be >= 2.
 * @return Handle, or NULL on invalid arguments or allocation failure.
 */
EogNorm *eog_norm_create_samples(size_t window_samples);

/**
 * @brief Destroy a standardizer.
 * @param norm  Handle to destroy. NULL is safe (no-op).
 */
void eog_norm_destroy(EogNorm *norm);

/**
 * @brief Clear all buffered samples and statistics.
 *
 * Leaves the window length unchanged. Use between recording sessions.
 *
 * @param norm  Handle to reset.
 */
void eog_norm_reset(EogNorm *norm);

/*============================================================================
 * STREAMING OPERATION
 *============================================================================*/

/**
 * @brief Push one sample and get its standardized value.
 *
 * Until the window has filled, statistics are taken over the samples received
 * so far, so output starts at the second sample rather than waiting for the
 * full window. Pass require_full_window = true to eog_norm_set_strict() if you
 * would rather suppress output until the window is full.
 *
 * @param norm          Handle.
 * @param sample        Raw input sample.
 * @param output_valid  Set to true when the return value is meaningful. May be
 *                      NULL if the caller does not need it.
 * @return Standardized sample, valid only when *output_valid is true.
 */
float eog_norm_push(EogNorm *norm, float sample, bool *output_valid);

/**
 * @brief Require a full window before producing output.
 *
 * Off by default, in which case output begins once two samples have arrived and
 * a standard deviation is defined.
 *
 * @param norm    Handle.
 * @param strict  True to suppress output until the window is full.
 */
void eog_norm_set_strict(EogNorm *norm, bool strict);

/**
 * @brief Set the floor applied to the standard deviation before dividing.
 *
 * Guards against division by zero on a flat or near-flat signal. Samples whose
 * window standard deviation falls below this value are divided by the floor
 * instead, keeping the output finite. Default is 1e-6.
 *
 * @param norm      Handle.
 * @param min_stddev  Floor value. Must be > 0 to take effect.
 */
void eog_norm_set_min_stddev(EogNorm *norm, float min_stddev);

/*============================================================================
 * STATE QUERIES
 *============================================================================*/

/** @brief Window length in samples. */
size_t eog_norm_window_length(const EogNorm *norm);

/** @brief Samples currently held in the window (<= window length). */
size_t eog_norm_count(const EogNorm *norm);

/** @brief True once the window holds a full complement of samples. */
bool eog_norm_ready(const EogNorm *norm);

/** @brief Mean over the current window. Zero when no samples are held. */
float eog_norm_mean(const EogNorm *norm);

/** @brief Sample standard deviation over the current window (Bessel corrected). */
float eog_norm_stddev(const EogNorm *norm);

#ifdef __cplusplus
}
#endif

#endif /* EOG_NORMALIZE_H */
