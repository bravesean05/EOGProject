/**
 * @file eog_cnn.h
 * @brief 1-D CNN inference for classifying EOG windows.
 *
 * Runs the network trained by matlab/eog_train_cnn.m, using weights exported
 * by matlab/eog_export_cnn.m into eog_cnn_weights.h. Classifies one window of
 * the filtered two-channel signal into U, D, L, R, idle or blink.
 *
 * The input is a window of the same series filter_dataset emits: the output of
 * eog_epoch, horizontal and vertical, EOG_CNN_INPUT_LEN samples long. Feeding
 * it raw signal instead would put it on a scale the network never saw.
 *
 * No heap allocation. Intermediate activations live in an EogCnnWork the
 * caller provides, so the call is reentrant and the memory cost is visible:
 * sizeof(EogCnnWork) is the whole of it.
 */

#ifndef EOG_CNN_H
#define EOG_CNN_H

#include "eog_cnn_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Largest activation, time by channel, across the layers: the first conv's
 *  output at full length. Later layers halve the length as they double the
 *  channels, so they fit in the same space. */
#define EOG_CNN_WORK_FLOATS (EOG_CNN_INPUT_LEN * EOG_CNN_C1_COUT)

/** Scratch space for one inference. Reusable across calls. */
typedef struct {
    float a[EOG_CNN_WORK_FLOATS];
    float b[EOG_CNN_WORK_FLOATS];
} EogCnnWork;

/**
 * @brief Classify one window.
 *
 * @param window  EOG_CNN_INPUT_LEN x EOG_CNN_INPUT_CH samples, time-major with
 *                the channels interleaved: h0 v0 h1 v1 ...
 * @param probs   Receives EOG_CNN_NUM_CLASSES probabilities summing to 1.
 *                May be NULL when only the class is wanted.
 * @param work    Scratch space.
 * @return Index of the most probable class, or -1 on a NULL argument.
 */
int eog_cnn_classify(const float *window, float *probs, EogCnnWork *work);

/** @brief Name of a class index, such as "U" or "blink". "?" if out of range. */
const char *eog_cnn_class_name(int index);

#ifdef __cplusplus
}
#endif

#endif /* EOG_CNN_H */
