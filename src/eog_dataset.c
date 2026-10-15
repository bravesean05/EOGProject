/**
 * @file eog_dataset.c
 * @brief Loader for flat .eog recordings.
 *
 * The format is written by tools/convert_dataset.py on a little-endian machine
 * and read back here on one, so the fixed-width fields are read directly rather
 * than byte-swapped. The magic string is checked first, which catches both a
 * wrong file and one written by a future, incompatible converter.
 */

#include "eog_dataset.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EOG_MAGIC      "EOGDATA1"
#define EOG_MAGIC_LEN  8

/** Largest recording accepted, as a guard against a corrupt length field. */
#define EOG_MAX_SAMPLES ((size_t)100000000)

int eog_dataset_load(const char *path, EogDataset *out)
{
    if (path == NULL || out == NULL) {
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "eog_dataset: cannot open %s\n", path);
        return -1;
    }

    char magic[EOG_MAGIC_LEN];
    uint32_t n_samples = 0;
    float rate = 0.0f;

    if (fread(magic, 1, EOG_MAGIC_LEN, f) != EOG_MAGIC_LEN ||
        memcmp(magic, EOG_MAGIC, EOG_MAGIC_LEN) != 0) {
        fprintf(stderr, "eog_dataset: %s is not an .eog file "
                        "(run tools/convert_dataset.py first)\n", path);
        fclose(f);
        return -1;
    }

    if (fread(&n_samples, sizeof n_samples, 1, f) != 1 ||
        fread(&rate, sizeof rate, 1, f) != 1) {
        fprintf(stderr, "eog_dataset: %s has a truncated header\n", path);
        fclose(f);
        return -1;
    }

    if (n_samples == 0 || (size_t)n_samples > EOG_MAX_SAMPLES ||
        !(rate > 0.0f)) {
        fprintf(stderr, "eog_dataset: %s declares an implausible "
                        "%u samples at %g Hz\n", path,
                        (unsigned)n_samples, (double)rate);
        fclose(f);
        return -1;
    }

    size_t n = (size_t)n_samples;
    float *h = (float *)malloc(n * sizeof(float));
    float *v = (float *)malloc(n * sizeof(float));
    unsigned char *c = (unsigned char *)malloc(n);

    if (h == NULL || v == NULL || c == NULL) {
        fprintf(stderr, "eog_dataset: out of memory for %zu samples\n", n);
        free(h); free(v); free(c);
        fclose(f);
        return -1;
    }

    if (fread(h, sizeof(float), n, f) != n ||
        fread(v, sizeof(float), n, f) != n ||
        fread(c, 1, n, f) != n) {
        fprintf(stderr, "eog_dataset: %s ends early, expected %zu samples "
                        "per channel\n", path, n);
        free(h); free(v); free(c);
        fclose(f);
        return -1;
    }

    fclose(f);

    out->n_samples      = n;
    out->sample_rate_hz = rate;
    out->horizontal     = h;
    out->vertical       = v;
    out->control        = c;
    return 0;
}

void eog_dataset_free(EogDataset *ds)
{
    if (ds == NULL) {
        return;
    }
    free(ds->horizontal);
    free(ds->vertical);
    free(ds->control);
    ds->horizontal = NULL;
    ds->vertical   = NULL;
    ds->control    = NULL;
    ds->n_samples  = 0;
}

const char *eog_label_name(unsigned char label)
{
    switch (label) {
    case EOG_LABEL_FORWARD_SACCADE: return "forward";
    case EOG_LABEL_RETURN_SACCADE:  return "return";
    case EOG_LABEL_BLINK:           return "blink";
    case EOG_LABEL_NONE:            return "none";
    default:                        return "?";
    }
}
