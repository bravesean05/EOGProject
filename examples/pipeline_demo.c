/**
 * @file pipeline_demo.c
 * @brief Epoch pipeline driven the way a live acquisition loop would drive it.
 *
 * Feeds a synthetic EOG-like signal one sample at a time and prints what comes
 * back, along with the timing the geometry implies.
 */

#include <math.h>
#include <stdio.h>

#include "eog_epoch.h"

#define SAMPLE_RATE_HZ 250.0

/**
 * @brief Stand-in for a device read: drifting baseline, noise, and eye movements.
 *
 * Movements are steps held for 300 samples (1.2 s), long enough to show that a
 * deflection outlasting the epoch fades from the standardized output.
 */
static float synth_sample(size_t i)
{
    double baseline = 2000.0 + 0.02 * (double)i;
    double noise    = 3.0 * sin((double)i * 0.7) + 1.5 * sin((double)i * 1.9);
    double movement = ((i / 300) % 2 == 1) ? 40.0 : 0.0;
    return (float)(baseline + noise + movement);
}

int main(void)
{
    EogEpoch *ep = eog_epoch_create(NULL);
    if (ep == NULL) {
        fprintf(stderr, "failed to create pipeline\n");
        return 1;
    }

    const size_t stride = eog_epoch_output_block(ep);
    const double lat_min = (double)eog_epoch_latency_min(ep) / SAMPLE_RATE_HZ * 1000.0;
    const double lat_max = (double)eog_epoch_latency_max(ep) / SAMPLE_RATE_HZ * 1000.0;

    printf("Pipeline geometry\n");
    printf("=================\n");
    printf("  Sample rate       : %.0f Hz\n", SAMPLE_RATE_HZ);
    printf("  Epoch             : 100 samples (400 ms)\n");
    printf("  Stride            : %zu samples (%.0f ms between evaluations)\n",
           stride, (double)stride / SAMPLE_RATE_HZ * 1000.0);
    printf("  Savitzky-Golay    : half window 5 -> 11-sample kernel (44 ms)\n");
    printf("  Emitted per block : %zu samples\n", stride);
    printf("  Latency           : %.0f-%.0f ms (kernel step-back, then stride)\n",
           lat_min, lat_max);

    /* --- acquisition loop ------------------------------------------------- */
    const size_t total = 1500;
    float out[64];
    size_t blocks = 0;
    size_t emitted = 0;

    printf("\nAbs. sample   raw        processed   note\n");
    printf("------------------------------------------------\n");

    size_t absolute = 100 - 10 - 5;   /* first emitted sample's absolute index */

    for (size_t i = 0; i < total; i++) {
        float raw = synth_sample(i);
        int n = eog_epoch_push(ep, raw, out, (int)(sizeof out / sizeof out[0]));
        if (n <= 0) {
            continue;
        }
        blocks++;
        emitted += (size_t)n;

        for (int k = 0; k < n; k++) {
            size_t abs_idx = absolute + (size_t)k;
            /* Print sparsely, plus a close look either side of a movement */
            bool near_move = (abs_idx > 295 && abs_idx < 306);
            if (abs_idx % 200 == 0 || near_move) {
                const char *note = "";
                if (abs_idx == 300) note = "<- movement starts";
                else if (abs_idx == 600) note = "<- still deflected, response faded";
                printf("  %6zu    %9.2f   %9.4f   %s\n",
                       abs_idx, (double)synth_sample(abs_idx),
                       (double)out[k], note);
            }
        }
        absolute += (size_t)n;
    }

    printf("\nThroughput\n");
    printf("  Samples in        : %zu\n", total);
    printf("  Blocks emitted    : %zu\n", blocks);
    printf("  Samples emitted   : %zu\n", emitted);
    printf("  Last epoch mean   : %.2f\n", (double)eog_epoch_last_mean(ep));
    printf("  Last epoch stddev : %.2f\n", (double)eog_epoch_last_stddev(ep));

    eog_epoch_destroy(ep);
    return 0;
}
