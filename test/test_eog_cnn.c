/**
 * @file test_eog_cnn.c
 * @brief Check the C inference reproduces MATLAB's network.
 *
 * Accuracy is not the question here; the cross-validation answered that in
 * MATLAB. The question is whether the C code computes the same function the
 * MATLAB network does, so the measure is agreement with MATLAB's own outputs
 * on identical inputs: the class it picks, and how far the probabilities
 * differ.
 *
 * Without arguments, checks the windows embedded in eog_cnn_vectors.h, which
 * is what ctest runs. Given the path of the check file eog_export_cnn writes,
 * checks every window in the dataset as well.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "eog_cnn.h"
#include "eog_cnn_vectors.h"

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) printf("  %-55s ", name)
#define PASS() do { printf("[PASS]\n"); g_passed++; } while (0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_failed++; } while (0)

/** Largest probability gap tolerated against MATLAB. Float32 arithmetic in a
 *  different order, with batch normalization folded in, lands within 1e-5 in
 *  practice; this leaves headroom without hiding a real mismatch. */
#define PROB_TOL 1e-4

#define WINDOW_FLOATS (EOG_CNN_INPUT_LEN * EOG_CNN_INPUT_CH)

static EogCnnWork g_work;

static void test_embedded_vectors(void)
{
    double worst = 0.0;
    int agree = 0;
    int correct_c = 0;

    for (int i = 0; i < EOG_CNN_NUM_VECTORS; i++) {
        float probs[EOG_CNN_NUM_CLASSES];
        const float *x = eog_cnn_vec_input + (size_t)i * WINDOW_FLOATS;
        const float *ref = eog_cnn_vec_prob + (size_t)i * EOG_CNN_NUM_CLASSES;

        int cls = eog_cnn_classify(x, probs, &g_work);

        int ref_cls = 0;
        for (int k = 0; k < EOG_CNN_NUM_CLASSES; k++) {
            double d = fabs((double)probs[k] - (double)ref[k]);
            if (d > worst) worst = d;
            if (ref[k] > ref[ref_cls]) ref_cls = k;
        }
        if (cls == ref_cls) agree++;
        if (cls == eog_cnn_vec_label[i]) correct_c++;
    }

    TEST("Same class as MATLAB on every embedded window");
    if (agree == EOG_CNN_NUM_VECTORS) PASS();
    else {
        char m[64];
        snprintf(m, sizeof m, "%d of %d agree", agree, EOG_CNN_NUM_VECTORS);
        FAIL(m);
    }

    TEST("Probabilities within tolerance of MATLAB");
    if (worst < PROB_TOL) PASS();
    else {
        char m[64];
        snprintf(m, sizeof m, "max difference %.3e", worst);
        FAIL(m);
    }

    printf("      (%d windows, max probability difference %.2e, "
           "%d of %d match the label)\n",
           EOG_CNN_NUM_VECTORS, worst, correct_c, EOG_CNN_NUM_VECTORS);
}

static void test_probabilities_sum(void)
{
    float probs[EOG_CNN_NUM_CLASSES];
    eog_cnn_classify(eog_cnn_vec_input, probs, &g_work);
    double sum = 0.0;
    for (int k = 0; k < EOG_CNN_NUM_CLASSES; k++) sum += probs[k];

    TEST("Probabilities sum to one");
    if (fabs(sum - 1.0) < 1e-5) PASS();
    else FAIL("softmax does not normalize");
}

static void test_arguments(void)
{
    TEST("NULL window or work is rejected");
    float probs[EOG_CNN_NUM_CLASSES];
    if (eog_cnn_classify(NULL, probs, &g_work) == -1 &&
        eog_cnn_classify(eog_cnn_vec_input, probs, NULL) == -1) PASS();
    else FAIL("accepted a NULL argument");

    TEST("Probabilities are optional");
    int with = eog_cnn_classify(eog_cnn_vec_input, probs, &g_work);
    int without = eog_cnn_classify(eog_cnn_vec_input, NULL, &g_work);
    if (with == without) PASS();
    else FAIL("class depends on whether probabilities were requested");

    TEST("Class names cover every index");
    int ok = strcmp(eog_cnn_class_name(-1), "?") == 0 &&
             strcmp(eog_cnn_class_name(EOG_CNN_NUM_CLASSES), "?") == 0;
    for (int k = 0; k < EOG_CNN_NUM_CLASSES; k++) {
        if (strcmp(eog_cnn_class_name(k), "?") == 0) ok = 0;
    }
    if (ok) PASS();
    else FAIL("missing or out-of-range name");
}

/**
 * @brief Check every window in the export's check file.
 *
 * Reports agreement with MATLAB and, separately, each side's accuracy against
 * the labels. The two accuracies should be identical; if they are not, the
 * difference is the C code's doing, not the model's.
 */
static void test_full_set(const char *path)
{
    FILE *f = fopen(path, "rb");
    TEST("Check file opens");
    if (f == NULL) { FAIL(path); return; }
    PASS();

    char magic[8];
    uint32_t hdr[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "EOGCNNCK", 8) != 0 ||
        fread(hdr, sizeof(uint32_t), 4, f) != 4) {
        TEST("Check file header");
        FAIL("not an EOGCNNCK file");
        fclose(f);
        return;
    }
    uint32_t n = hdr[0];

    TEST("Check file geometry matches the compiled network");
    if (hdr[1] != EOG_CNN_INPUT_LEN || hdr[2] != EOG_CNN_INPUT_CH ||
        hdr[3] != EOG_CNN_NUM_CLASSES) {
        FAIL("exported from a different network");
        fclose(f);
        return;
    }
    PASS();

    float *x = (float *)malloc((size_t)n * WINDOW_FLOATS * sizeof(float));
    float *ref = (float *)malloc((size_t)n * EOG_CNN_NUM_CLASSES * sizeof(float));
    unsigned char *lab = (unsigned char *)malloc(n);
    if (x == NULL || ref == NULL || lab == NULL ||
        fread(x, sizeof(float), (size_t)n * WINDOW_FLOATS, f) != (size_t)n * WINDOW_FLOATS ||
        fread(ref, sizeof(float), (size_t)n * EOG_CNN_NUM_CLASSES, f) != (size_t)n * EOG_CNN_NUM_CLASSES ||
        fread(lab, 1, n, f) != n) {
        TEST("Check file body");
        FAIL("truncated or out of memory");
        free(x); free(ref); free(lab);
        fclose(f);
        return;
    }
    fclose(f);

    double worst = 0.0;
    uint32_t agree = 0, correct_c = 0, correct_m = 0;

    clock_t t0 = clock();
    for (uint32_t i = 0; i < n; i++) {
        float probs[EOG_CNN_NUM_CLASSES];
        const float *r = ref + (size_t)i * EOG_CNN_NUM_CLASSES;
        int cls = eog_cnn_classify(x + (size_t)i * WINDOW_FLOATS, probs, &g_work);

        int ref_cls = 0;
        for (int k = 0; k < EOG_CNN_NUM_CLASSES; k++) {
            double d = fabs((double)probs[k] - (double)r[k]);
            if (d > worst) worst = d;
            if (r[k] > r[ref_cls]) ref_cls = k;
        }
        if (cls == ref_cls) agree++;
        if (cls == lab[i]) correct_c++;
        if (ref_cls == lab[i]) correct_m++;
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

    TEST("Same class as MATLAB on every window in the set");
    if (agree == n) PASS();
    else {
        char m[64];
        snprintf(m, sizeof m, "%u of %u agree", agree, n);
        FAIL(m);
    }

    TEST("Probabilities within tolerance across the set");
    if (worst < PROB_TOL) PASS();
    else {
        char m[64];
        snprintf(m, sizeof m, "max difference %.3e", worst);
        FAIL(m);
    }

    printf("      (%u windows, max probability difference %.2e)\n", n, worst);
    printf("      (accuracy against labels: C %.2f%%, MATLAB %.2f%%)\n",
           100.0 * correct_c / n, 100.0 * correct_m / n);
    printf("      (%.1f us per window on this machine)\n", 1e6 * secs / n);

    free(x); free(ref); free(lab);
}

int main(int argc, char **argv)
{
    printf("\nCNN inference against MATLAB\n");
    printf("============================\n\n");
    printf("  work buffer: %zu bytes\n\n", sizeof(EogCnnWork));

    printf("Embedded vectors:\n");  test_embedded_vectors();
    printf("\nOutput:\n");          test_probabilities_sum();
    printf("\nArguments:\n");       test_arguments();

    if (argc > 1) {
        printf("\nFull set (%s):\n", argv[1]);
        test_full_set(argv[1]);
    }

    printf("\n=== Results: %d passed, %d failed ===\n\n", g_passed, g_failed);
    return (g_failed == 0) ? 0 : 1;
}
