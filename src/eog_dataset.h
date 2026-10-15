/**
 * @file eog_dataset.h
 * @brief Loader for the flat .eog files produced by tools/convert_dataset.py.
 *
 * The dataset ships as MATLAB v5 .mat files, which the converter rewrites into
 * a flat little-endian layout this reads directly:
 *
 *     offset  type         field
 *     0       char[8]      "EOGDATA1"
 *     8       uint32       n_samples
 *     12      float32      sample_rate_hz
 *     16      float32[n]   horizontal EOG
 *     ...     float32[n]   vertical EOG
 *     ...     uint8[n]     control signal
 *
 * The control signal labels every sample with the eye movement underway, which
 * lets a run be scored against ground truth without a second file.
 */

#ifndef EOG_DATASET_H
#define EOG_DATASET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Ground-truth label carried by each sample. */
typedef enum {
    EOG_LABEL_NONE            = 0,
    EOG_LABEL_FORWARD_SACCADE = 1,
    EOG_LABEL_RETURN_SACCADE  = 2,
    EOG_LABEL_BLINK           = 3
} EogLabel;

/** One subject's recording. */
typedef struct {
    size_t         n_samples;
    float          sample_rate_hz;
    float         *horizontal;   /**< EOGh, n_samples long */
    float         *vertical;     /**< EOGv, n_samples long */
    unsigned char *control;      /**< EogLabel per sample, n_samples long */
} EogDataset;

/**
 * @brief Load a .eog file.
 *
 * @param path  File to read.
 * @param out   Receives the recording on success; untouched on failure.
 * @return 0 on success, -1 on error, with a message on stderr.
 */
int eog_dataset_load(const char *path, EogDataset *out);

/**
 * @brief Release a loaded recording.
 * @param ds  Recording to free. NULL is safe, as is a zeroed struct.
 */
void eog_dataset_free(EogDataset *ds);

/** @brief Human-readable name for a label, for printing. */
const char *eog_label_name(unsigned char label);

#ifdef __cplusplus
}
#endif

#endif /* EOG_DATASET_H */
