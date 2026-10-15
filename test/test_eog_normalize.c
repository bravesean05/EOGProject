/**
 * @file test_eog_normalize.c
 * @brief Tests for the rolling-window standardizer.
 *
 * The central claim under test is that the O(1) incremental statistics agree
 * with a naive O(n) recomputation over the same trailing window, including
 * after long runs and on signals with a large DC baseline, where the incremental
 * form is most at risk of losing precision.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "eog_normalize.h"

/*============================================================================
 * TEST FRAMEWORK
 *============================================================================*/

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) printf("  %-55s ", name)
#define PASS() do { printf("[PASS]\n"); g_passed++; } while (0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_failed++; } while (0)

/*============================================================================
 * REFERENCE IMPLEMENTATION
 *============================================================================*/

/** Naive trailing-window mean over the last `window` samples ending at `end`. */
static double ref_mean(const float *data, size_t end, size_t window)
{
    size_t start = (end + 1 >= window) ? (end + 1 - window) : 0;
    size_t n = end - start + 1;
    double sum = 0.0;
    for (size_t i = start; i <= end; i++) {
        sum += (double)data[i];
    }
    return sum / (double)n;
}

/** Naive trailing-window sample standard deviation (Bessel corrected). */
static double ref_stddev(const float *data, size_t end, size_t window)
{
    size_t start = (end + 1 >= window) ? (end + 1 - window) : 0;
    size_t n = end - start + 1;
    if (n < 2) {
        return 0.0;
    }
    double mean = ref_mean(data, end, window);
    double ss = 0.0;
    for (size_t i = start; i <= end; i++) {
        double d = (double)data[i] - mean;
        ss += d * d;
    }
    return sqrt(ss / (double)(n - 1));
}

/** Deterministic pseudo-random generator, so failures reproduce. */
static unsigned long g_rng = 12345UL;
static double next_random(void)
{
    g_rng = g_rng * 1103515245UL + 12345UL;
    return (double)((g_rng >> 16) & 0x7FFF) / 32767.0;  /* [0, 1] */
}

/*============================================================================
 * TESTS
 *============================================================================*/

static void test_create_destroy(void)
{
    TEST("Create with explicit sample count");
    EogNorm *n = eog_norm_create_samples(103);
    if (n != NULL && eog_norm_window_length(n) == 103) PASS();
    else FAIL("wrong window length");
    eog_norm_destroy(n);

    TEST("Window length rounds from rate x duration");
    /* 250 Hz x 0.410 s = 102.5 -> 103, despite 0.410f being inexact */
    n = eog_norm_create(250.0f, 0.410f);
    if (n != NULL && eog_norm_window_length(n) == 103) PASS();
    else FAIL("expected 103 samples");
    eog_norm_destroy(n);

    TEST("Rounding boundary nudge does not distort ordinary cases");
    /* 1000 x 0.4104 = 410.4 -> 410, and 1000 x 0.4106 = 410.6 -> 411 */
    EogNorm *lo = eog_norm_create(1000.0f, 0.4104f);
    EogNorm *hi = eog_norm_create(1000.0f, 0.4106f);
    if (lo != NULL && hi != NULL &&
        eog_norm_window_length(lo) == 410 &&
        eog_norm_window_length(hi) == 411) PASS();
    else FAIL("rounding skewed away from nearest");
    eog_norm_destroy(lo);
    eog_norm_destroy(hi);

    TEST("Common EOG rates give the expected 410 ms window");
    /* 256 Hz -> 104.96 -> 105 ; 500 Hz -> 205 ; 1000 Hz -> 410 */
    EogNorm *a = eog_norm_create(256.0f, 0.410f);
    EogNorm *b = eog_norm_create(500.0f, 0.410f);
    EogNorm *c = eog_norm_create(1000.0f, 0.410f);
    if (eog_norm_window_length(a) == 105 &&
        eog_norm_window_length(b) == 205 &&
        eog_norm_window_length(c) == 410) PASS();
    else FAIL("unexpected window length");
    eog_norm_destroy(a);
    eog_norm_destroy(b);
    eog_norm_destroy(c);

    TEST("Reject window shorter than two samples");
    if (eog_norm_create_samples(1) == NULL &&
        eog_norm_create(10.0f, 0.01f) == NULL) PASS();
    else FAIL("accepted degenerate window");

    TEST("Reject non-positive arguments");
    if (eog_norm_create(0.0f, 1.0f) == NULL &&
        eog_norm_create(250.0f, 0.0f) == NULL &&
        eog_norm_create(-250.0f, 1.0f) == NULL) PASS();
    else FAIL("accepted invalid rate or duration");

    TEST("Destroy tolerates NULL");
    eog_norm_destroy(NULL);
    PASS();
}

static void test_output_gating(void)
{
    EogNorm *n = eog_norm_create_samples(10);

    TEST("First sample produces no output");
    bool valid;
    eog_norm_push(n, 1.0f, &valid);
    if (!valid) PASS();
    else FAIL("output before stddev is defined");

    TEST("Second sample produces output");
    eog_norm_push(n, 2.0f, &valid);
    if (valid) PASS();
    else FAIL("no output at two samples");

    eog_norm_destroy(n);

    TEST("Strict mode waits for a full window");
    n = eog_norm_create_samples(10);
    eog_norm_set_strict(n, true);
    bool any = false;
    for (int i = 0; i < 9; i++) {
        eog_norm_push(n, (float)i, &valid);
        if (valid) any = true;
    }
    if (!any) PASS();
    else FAIL("produced output before window filled");

    TEST("Strict mode outputs once window is full");
    eog_norm_push(n, 9.0f, &valid);
    if (valid && eog_norm_ready(n)) PASS();
    else FAIL("no output at full window");
    eog_norm_destroy(n);
}

static void test_matches_reference(void)
{
    const size_t window = 103;   /* 410 ms at 250 Hz */
    const size_t count  = 5000;

    float *data = (float *)malloc(count * sizeof(float));
    EogNorm *n = eog_norm_create_samples(window);

    /* Signal shaped like an electrode recording: large DC baseline, slow drift,
     * small AC component, occasional fast excursions */
    for (size_t i = 0; i < count; i++) {
        double baseline = 2000.0 + 0.05 * (double)i;
        double slow     = 12.0 * sin((double)i * 0.01);
        double noise    = (next_random() - 0.5) * 2.0;
        double burst    = ((i % 700) < 40) ? 25.0 : 0.0;
        data[i] = (float)(baseline + slow + noise + burst);
    }

    double worst_mean = 0.0;
    double worst_sd   = 0.0;
    double worst_z    = 0.0;

    for (size_t i = 0; i < count; i++) {
        bool valid;
        float z = eog_norm_push(n, data[i], &valid);

        double em = fabs((double)eog_norm_mean(n) - ref_mean(data, i, window));
        if (em > worst_mean) worst_mean = em;

        double es = fabs((double)eog_norm_stddev(n) - ref_stddev(data, i, window));
        if (es > worst_sd) worst_sd = es;

        if (valid) {
            double rm = ref_mean(data, i, window);
            double rs = ref_stddev(data, i, window);
            double expected = ((double)data[i] - rm) / rs;
            double ez = fabs((double)z - expected);
            if (ez > worst_z) worst_z = ez;
        }
    }

    TEST("Incremental mean matches recomputation");
    if (worst_mean < 1e-3) PASS();
    else { char m[96]; snprintf(m, sizeof m, "max error %.3e", worst_mean); FAIL(m); }

    TEST("Incremental stddev matches recomputation");
    if (worst_sd < 1e-3) PASS();
    else { char m[96]; snprintf(m, sizeof m, "max error %.3e", worst_sd); FAIL(m); }

    TEST("Standardized output matches recomputation");
    if (worst_z < 1e-3) PASS();
    else { char m[96]; snprintf(m, sizeof m, "max error %.3e", worst_z); FAIL(m); }

    printf("      (max errors: mean %.2e, stddev %.2e, z %.2e)\n",
           worst_mean, worst_sd, worst_z);

    eog_norm_destroy(n);
    free(data);
}

static void test_long_run_stability(void)
{
    /* The incremental sums are rebuilt once per window; this run covers many
     * rebuild cycles to confirm error does not creep upward over time. */
    const size_t window = 103;
    const size_t count  = 200000;

    EogNorm *n = eog_norm_create_samples(window);
    float *tail = (float *)malloc(window * sizeof(float));

    for (size_t i = 0; i < count; i++) {
        float s = (float)(5000.0 + 3.0 * sin((double)i * 0.02)
                          + (next_random() - 0.5));
        eog_norm_push(n, s, NULL);
        tail[i % window] = s;
    }

    /* Compare against a direct computation over the final window */
    double sum = 0.0;
    for (size_t i = 0; i < window; i++) sum += (double)tail[i];
    double mean = sum / (double)window;
    double ss = 0.0;
    for (size_t i = 0; i < window; i++) {
        double d = (double)tail[i] - mean;
        ss += d * d;
    }
    double sd = sqrt(ss / (double)(window - 1));

    TEST("Mean stable after 200k samples");
    if (fabs((double)eog_norm_mean(n) - mean) < 1e-3) PASS();
    else FAIL("drifted from recomputed mean");

    TEST("Stddev stable after 200k samples");
    if (fabs((double)eog_norm_stddev(n) - sd) < 1e-3) PASS();
    else FAIL("drifted from recomputed stddev");

    eog_norm_destroy(n);
    free(tail);
}

static void test_flat_signal(void)
{
    EogNorm *n = eog_norm_create_samples(20);

    TEST("Constant signal does not divide by zero");
    bool valid;
    float z = 0.0f;
    for (int i = 0; i < 30; i++) {
        z = eog_norm_push(n, 1234.5f, &valid);
    }
    if (valid && isfinite(z)) PASS();
    else FAIL("non-finite output on flat input");

    TEST("Constant signal standardizes to zero");
    if (fabs((double)z) < 1e-6) PASS();
    else FAIL("expected zero deviation");

    eog_norm_destroy(n);
}

static void test_known_values(void)
{
    /* Window of 5 over 1..5: mean 3, sample stddev sqrt(2.5) */
    EogNorm *n = eog_norm_create_samples(5);
    bool valid;
    float z = 0.0f;
    for (int i = 1; i <= 5; i++) {
        z = eog_norm_push(n, (float)i, &valid);
    }

    TEST("Known window: mean is 3");
    if (fabs((double)eog_norm_mean(n) - 3.0) < 1e-5) PASS();
    else FAIL("wrong mean");

    TEST("Known window: stddev is sqrt(2.5)");
    if (fabs((double)eog_norm_stddev(n) - sqrt(2.5)) < 1e-5) PASS();
    else FAIL("wrong stddev");

    TEST("Known window: last sample z is 2/sqrt(2.5)");
    if (fabs((double)z - 2.0 / sqrt(2.5)) < 1e-5) PASS();
    else FAIL("wrong standardized value");

    eog_norm_destroy(n);
}

static void test_window_slides(void)
{
    /* Once full, statistics must reflect only the trailing window, not history */
    EogNorm *n = eog_norm_create_samples(4);
    for (int i = 0; i < 4; i++) {
        eog_norm_push(n, 100.0f, NULL);
    }

    TEST("Old samples leave the window");
    for (int i = 0; i < 4; i++) {
        eog_norm_push(n, 7.0f, NULL);
    }
    /* Window now holds four 7s; the 100s must be gone */
    if (fabs((double)eog_norm_mean(n) - 7.0) < 1e-5) PASS();
    else FAIL("stale samples still counted");

    eog_norm_destroy(n);
}

static void test_reset(void)
{
    EogNorm *n = eog_norm_create_samples(10);
    for (int i = 0; i < 10; i++) {
        eog_norm_push(n, (float)(i * 100), NULL);
    }

    TEST("Reset clears buffered samples");
    eog_norm_reset(n);
    if (eog_norm_count(n) == 0 && !eog_norm_ready(n)) PASS();
    else FAIL("state survived reset");

    TEST("Window length survives reset");
    if (eog_norm_window_length(n) == 10) PASS();
    else FAIL("window length lost");

    eog_norm_destroy(n);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("\nRolling-window standardizer tests\n");
    printf("=================================\n\n");

    printf("Lifecycle:\n");            test_create_destroy();
    printf("\nOutput gating:\n");       test_output_gating();
    printf("\nKnown values:\n");        test_known_values();
    printf("\nWindow behaviour:\n");    test_window_slides();
    printf("\nAgainst reference:\n");   test_matches_reference();
    printf("\nLong-run stability:\n");  test_long_run_stability();
    printf("\nDegenerate input:\n");    test_flat_signal();
    printf("\nReset:\n");               test_reset();

    printf("\n=== Results: %d passed, %d failed ===\n\n", g_passed, g_failed);
    return (g_failed == 0) ? 0 : 1;
}
