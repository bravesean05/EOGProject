/**
 * @file filter_dataset.c
 * @brief Run the epoch pipeline over a converted EOG recording.
 *
 * Reads a .eog file, pushes both channels through eog_epoch sample by sample
 * exactly as a live acquisition loop would, and writes the result as CSV
 * alongside the raw signal and the ground-truth label for each sample.
 *
 * Usage:
 *   filter_dataset <input.eog> [options]
 *
 *   -o <file>     Write CSV here. Without it, only the summary is printed.
 *   -e <n>        Epoch length in samples (default 100).
 *   -s <n>        Stride in samples (default 10).
 *   -w <n>        Savitzky-Golay half window (default 5).
 *   -p <n>        Savitzky-Golay polynomial order (default 2).
 *   -l <n>        Stop after n input samples, for a quick look.
 *
 * The CSV column `sample` is the absolute index in the input, so a row lines up
 * with the raw recording despite the pipeline's lag. Rows begin at the first
 * emitted sample rather than at zero, since nothing is emitted until an epoch
 * has been collected.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eog_dataset.h"
#include "eog_epoch.h"

#define MAX_BLOCK 4096

/** Number of distinct ground-truth labels, 0 through 3. */
#define N_LABELS 4

/** Per-channel pipeline state and accumulated statistics. */
typedef struct {
    EogEpoch *ep;
    float     block[MAX_BLOCK];
    double    sum;
    double    sum_sq;
    double    min;
    double    max;
    size_t    count;

    /* Per-label accumulators, for scoring the output against ground truth.
     * `out` tracks the magnitude of the standardized output, `sd` the epoch
     * deviation that produced it. The deviation is a block-level quantity, so
     * every sample in a block is credited with its block's value. */
    double    out_abs[N_LABELS];
    double    sd_sum[N_LABELS];
    size_t    label_count[N_LABELS];
} Channel;

static int channel_init(Channel *ch, const EogEpochConfig *cfg)
{
    memset(ch, 0, sizeof *ch);
    ch->ep = eog_epoch_create(cfg);
    ch->min = 1e300;
    ch->max = -1e300;
    return (ch->ep != NULL) ? 0 : -1;
}

/**
 * @brief Fold one emitted block into the running statistics.
 *
 * @param labels  Ground-truth label for each emitted sample.
 * @param sd      Epoch deviation that standardized this block.
 */
static void channel_accumulate(Channel *ch, const float *values, int n,
                               const unsigned char *labels, double sd)
{
    for (int i = 0; i < n; i++) {
        double v = (double)values[i];
        ch->sum += v;
        ch->sum_sq += v * v;
        if (v < ch->min) ch->min = v;
        if (v > ch->max) ch->max = v;
        ch->count++;

        unsigned char lab = labels[i];
        if (lab < N_LABELS) {
            ch->out_abs[lab] += (v < 0.0) ? -v : v;
            ch->sd_sum[lab] += sd;
            ch->label_count[lab]++;
        }
    }
}

/**
 * @brief Print output magnitude and epoch deviation per label.
 *
 * Standardizing each epoch against its own deviation puts every window on the
 * same scale, so the output magnitude column tends to flatten across labels
 * whatever the underlying movement was. The deviation column is the scale that
 * was divided out, and is where the separation between movement types survives.
 */
static void channel_label_report(const Channel *ch, const char *name)
{
    printf("\n  %s, by ground-truth label\n", name);
    printf("    %-16s %12s %14s %10s\n",
           "label", "mean |out|", "mean epoch sd", "samples");
    for (int lab = 0; lab < N_LABELS; lab++) {
        if (ch->label_count[lab] == 0) {
            continue;
        }
        double n = (double)ch->label_count[lab];
        printf("    %-16s %12.4f %14.2f %10zu\n",
               eog_label_name((unsigned char)lab),
               ch->out_abs[lab] / n, ch->sd_sum[lab] / n,
               ch->label_count[lab]);
    }
}

static void channel_report(const Channel *ch, const char *name)
{
    if (ch->count == 0) {
        printf("  %-12s no output\n", name);
        return;
    }
    double mean = ch->sum / (double)ch->count;
    double var = ch->sum_sq / (double)ch->count - mean * mean;
    if (var < 0.0) var = 0.0;   /* rounding can push a near-zero result negative */
    printf("  %-12s n=%zu  mean=%+.4f  sd=%.4f  min=%+.3f  max=%+.3f\n",
           name, ch->count, mean, sqrt(var), ch->min, ch->max);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <input.eog> [-o out.csv] [-e epoch] [-s stride]"
                " [-w half_window] [-p order] [-l limit]\n", argv[0]);
        return 2;
    }

    const char *in_path = argv[1];
    const char *out_path = NULL;
    size_t limit = 0;

    EogEpochConfig cfg = EOG_EPOCH_DEFAULT_CONFIG;

    for (int i = 2; i < argc; i++) {
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: missing value after %s\n", argv[0], argv[i]);
            return 2;
        }
        const char *val = argv[i + 1];
        if      (strcmp(argv[i], "-o") == 0) out_path = val;
        else if (strcmp(argv[i], "-e") == 0) cfg.epoch_samples = (size_t)atol(val);
        else if (strcmp(argv[i], "-s") == 0) cfg.stride = (size_t)atol(val);
        else if (strcmp(argv[i], "-w") == 0) cfg.sg_half_window = atoi(val);
        else if (strcmp(argv[i], "-p") == 0) cfg.sg_poly_order = atoi(val);
        else if (strcmp(argv[i], "-l") == 0) limit = (size_t)atol(val);
        else {
            fprintf(stderr, "%s: unknown option %s\n", argv[0], argv[i]);
            return 2;
        }
        i++;
    }

    if (cfg.stride > MAX_BLOCK) {
        fprintf(stderr, "%s: stride %zu exceeds the %d sample block buffer\n",
                argv[0], cfg.stride, MAX_BLOCK);
        return 2;
    }

    EogDataset ds;
    if (eog_dataset_load(in_path, &ds) != 0) {
        return 1;
    }

    size_t total = ds.n_samples;
    if (limit > 0 && limit < total) {
        total = limit;
    }

    Channel h, v;
    if (channel_init(&h, &cfg) != 0 || channel_init(&v, &cfg) != 0) {
        fprintf(stderr, "%s: epoch %zu, stride %zu, half window %d is not a "
                        "valid geometry (needs epoch >= stride + 2*half)\n",
                argv[0], cfg.epoch_samples, cfg.stride, cfg.sg_half_window);
        eog_dataset_free(&ds);
        return 1;
    }

    double rate = (double)ds.sample_rate_hz;
    printf("Input\n");
    printf("  file          : %s\n", in_path);
    printf("  samples       : %zu (%.1f s at %.0f Hz)\n",
           ds.n_samples, (double)ds.n_samples / rate, rate);
    if (total != ds.n_samples) {
        printf("  processing    : first %zu samples\n", total);
    }
    printf("\nPipeline\n");
    printf("  epoch         : %zu samples (%.1f ms)\n",
           cfg.epoch_samples, (double)cfg.epoch_samples / rate * 1000.0);
    printf("  stride        : %zu samples (%.1f ms)\n",
           cfg.stride, (double)cfg.stride / rate * 1000.0);
    printf("  SG kernel     : %d samples (%.1f ms), order %d\n",
           2 * cfg.sg_half_window + 1,
           (double)(2 * cfg.sg_half_window + 1) / rate * 1000.0,
           cfg.sg_poly_order);
    printf("  latency       : %.1f-%.1f ms\n",
           (double)eog_epoch_latency_min(h.ep) / rate * 1000.0,
           (double)eog_epoch_latency_max(h.ep) / rate * 1000.0);

    FILE *out = NULL;
    if (out_path != NULL) {
        out = fopen(out_path, "w");
        if (out == NULL) {
            fprintf(stderr, "%s: cannot write %s\n", argv[0], out_path);
            eog_epoch_destroy(h.ep);
            eog_epoch_destroy(v.ep);
            eog_dataset_free(&ds);
            return 1;
        }
        /* mean_* and sd_* are the epoch statistics that standardized the block
         * this row belongs to, repeated across the block's samples. sd_* is the
         * amplitude the z-score divided out, and carries the separation between
         * movement types that filt_* does not. */
        fprintf(out, "sample,time_s,"
                     "raw_h,filt_h,mean_h,sd_h,"
                     "raw_v,filt_v,mean_v,sd_v,label\n");
    }

    /* The first emitted sample sits this far into the first epoch */
    size_t absolute = cfg.epoch_samples - cfg.stride - (size_t)cfg.sg_half_window;
    size_t blocks = 0;

    for (size_t i = 0; i < total; i++) {
        int nh = eog_epoch_push(h.ep, ds.horizontal[i], h.block, MAX_BLOCK);
        int nv = eog_epoch_push(v.ep, ds.vertical[i], v.block, MAX_BLOCK);

        if (nh < 0 || nv < 0) {
            fprintf(stderr, "%s: pipeline error at sample %zu\n", argv[0], i);
            break;
        }
        if (nh == 0) {
            continue;
        }

        blocks++;

        /* Epoch statistics for the block just emitted. Read now, before the
         * next push replaces them. */
        double mean_h = (double)eog_epoch_last_mean(h.ep);
        double sd_h   = (double)eog_epoch_last_stddev(h.ep);
        double mean_v = (double)eog_epoch_last_mean(v.ep);
        double sd_v   = (double)eog_epoch_last_stddev(v.ep);

        channel_accumulate(&h, h.block, nh, ds.control + absolute, sd_h);
        channel_accumulate(&v, v.block, nv, ds.control + absolute, sd_v);

        if (out != NULL) {
            for (int k = 0; k < nh; k++) {
                size_t idx = absolute + (size_t)k;
                fprintf(out,
                        "%zu,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f,%u\n",
                        idx, (double)idx / rate,
                        (double)ds.horizontal[idx], (double)h.block[k],
                        mean_h, sd_h,
                        (double)ds.vertical[idx], (double)v.block[k],
                        mean_v, sd_v,
                        (unsigned)ds.control[idx]);
            }
        }
        absolute += (size_t)nh;
    }

    printf("\nOutput\n");
    printf("  blocks        : %zu\n", blocks);
    channel_report(&h, "horizontal");
    channel_report(&v, "vertical");
    if (out != NULL) {
        printf("  csv           : %s\n", out_path);
        fclose(out);
    }

    channel_label_report(&h, "horizontal");
    channel_label_report(&v, "vertical");

    eog_epoch_destroy(h.ep);
    eog_epoch_destroy(v.ep);
    eog_dataset_free(&ds);
    return 0;
}
