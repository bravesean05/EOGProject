/**
 * @file test_eog_epoch.c
 * @brief Tests for the epoch pipeline.
 *
 * The claims that matter here are about alignment rather than arithmetic: that
 * emitted blocks tile the input without gaps or repeats, that each emitted
 * sample corresponds to the absolute input index the latency figures promise,
 * and that every emitted sample was produced by the filter's central region
 * rather than its edge weights.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eog_epoch.h"
#include "savgolFilter.h"

/*============================================================================
 * TEST FRAMEWORK
 *============================================================================*/

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) printf("  %-55s ", name)
#define PASS() do { printf("[PASS]\n"); g_passed++; } while (0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); g_failed++; } while (0)

#define EPOCH  100
#define STRIDE  10
#define HALF     5
#define ORDER    2

static EogEpochConfig default_cfg(void)
{
    EogEpochConfig c = EOG_EPOCH_DEFAULT_CONFIG;
    return c;
}

/*============================================================================
 * TESTS
 *============================================================================*/

static void test_config_validation(void)
{
    TEST("Default config is accepted");
    EogEpoch *ep = eog_epoch_create(NULL);
    if (ep != NULL) PASS();
    else FAIL("default rejected");
    eog_epoch_destroy(ep);

    TEST("Default geometry is 100/10/5/2");
    EogEpochConfig c = default_cfg();
    if (c.epoch_samples == EPOCH && c.stride == STRIDE &&
        c.sg_half_window == HALF && c.sg_poly_order == ORDER) PASS();
    else FAIL("unexpected defaults");

    TEST("Reject epoch too short for stride plus kernel");
    c = default_cfg();
    c.epoch_samples = STRIDE + 2 * HALF - 1;   /* one short of the minimum */
    if (eog_epoch_create(&c) == NULL) PASS();
    else FAIL("accepted an epoch that pushes output onto the edge");

    TEST("Accept epoch exactly at the minimum");
    c = default_cfg();
    c.epoch_samples = STRIDE + 2 * HALF;
    ep = eog_epoch_create(&c);
    if (ep != NULL) PASS();
    else FAIL("rejected the boundary case");
    eog_epoch_destroy(ep);

    TEST("Reject zero stride and zero half window");
    c = default_cfg(); c.stride = 0;
    EogEpochConfig c2 = default_cfg(); c2.sg_half_window = 0;
    if (eog_epoch_create(&c) == NULL && eog_epoch_create(&c2) == NULL) PASS();
    else FAIL("accepted degenerate geometry");

    TEST("Reject polynomial order the kernel cannot fit");
    c = default_cfg();
    c.sg_poly_order = 2 * HALF + 1;   /* needs to be < kernel width */
    if (eog_epoch_create(&c) == NULL) PASS();
    else FAIL("accepted order >= kernel width");

    TEST("Destroy tolerates NULL");
    eog_epoch_destroy(NULL);
    PASS();
}

static void test_emission_schedule(void)
{
    EogEpoch *ep = eog_epoch_create(NULL);
    float out[STRIDE];

    TEST("Silent until the epoch fills");
    int emissions = 0;
    for (int i = 0; i < EPOCH - 1; i++) {
        if (eog_epoch_push(ep, (float)i, out, STRIDE) > 0) emissions++;
    }
    if (emissions == 0 && !eog_epoch_ready(ep)) PASS();
    else FAIL("emitted before a full epoch");

    TEST("Emits a full block on the epoch-filling sample");
    int n = eog_epoch_push(ep, (float)(EPOCH - 1), out, STRIDE);
    if (n == STRIDE && eog_epoch_ready(ep)) PASS();
    else FAIL("no block at epoch completion");

    TEST("Then emits once per stride, not per sample");
    int blocks = 0;
    int pushes = 0;
    for (int i = 0; i < 10 * STRIDE; i++) {
        pushes++;
        if (eog_epoch_push(ep, (float)(EPOCH + i), out, STRIDE) > 0) blocks++;
    }
    if (blocks == 10 && pushes == 100) PASS();
    else FAIL("wrong emission cadence");

    TEST("Rejects an output buffer smaller than the stride");
    float small[STRIDE - 1];
    int rc = 0;
    for (int i = 0; i < STRIDE; i++) {
        rc = eog_epoch_push(ep, 0.0f, small, STRIDE - 1);
    }
    if (rc == -1) PASS();
    else FAIL("wrote past the caller's buffer");

    eog_epoch_destroy(ep);
}

/** @brief Standardize `src` over its own statistics, then filter, into `dst`. */
static void reference_epoch(const float *src, size_t n, float *dst,
                            SavgolFilter *f)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += (double)src[i];
    double mean = sum / (double)n;

    double ss = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)src[i] - mean;
        ss += d * d;
    }
    double sd = sqrt(ss / (double)(n - 1));

    float *z = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        z[i] = (float)(((double)src[i] - mean) / sd);
    }
    savgol_apply(f, z, dst, n);
    free(z);
}

static void test_block_alignment(void)
{
    /* Each emitted block must be the slice [EPOCH-STRIDE-HALF, EPOCH-HALF) of
     * its own epoch, and successive epochs must advance by exactly STRIDE.
     *
     * Run against the per-epoch configuration, whose reference is a plain
     * standardize-then-filter over one epoch. The default configuration scales
     * from a longer window and re-centres each block, both of which depend on
     * history beyond the epoch and so have no single-epoch reference to check
     * against; its own behaviour is pinned by the tests below. */
    EogEpochConfig epcfg = EOG_EPOCH_PER_EPOCH_CONFIG;
    EogEpoch *ep = eog_epoch_create(&epcfg);
    SavgolConfig cfg = SAVGOL_SMOOTH(HALF, ORDER);
    SavgolFilter *f = savgol_create(&cfg);

    const int total = 400;
    float *signal = (float *)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) {
        signal[i] = (float)(1500.0 + 0.03 * i + 12.0 * sin(i * 0.17)
                            + 4.0 * cos(i * 0.9));
    }

    float out[STRIDE];
    float ref[EPOCH];
    int blocks = 0;
    bool aligned = true;

    for (int i = 0; i < total; i++) {
        int n = eog_epoch_push(ep, signal[i], out, STRIDE);
        if (n != STRIDE) {
            continue;
        }
        /* This block covers epoch ending at absolute index i */
        int epoch_start = i - EPOCH + 1;
        reference_epoch(signal + epoch_start, EPOCH, ref, f);

        size_t begin = EPOCH - STRIDE - HALF;
        for (int k = 0; k < STRIDE; k++) {
            if (fabs((double)out[k] - (double)ref[begin + k]) > 1e-4) {
                aligned = false;
            }
        }
        blocks++;
    }

    TEST("Multiple blocks emitted over the run");
    if (blocks >= 20) PASS();
    else FAIL("too few blocks to be meaningful");

    TEST("Every block matches its own epoch's recomputation");
    if (aligned) PASS();
    else FAIL("a block did not correspond to its epoch slice");

    savgol_destroy(f);
    free(signal);
    eog_epoch_destroy(ep);
}

/**
 * @brief Largest boundary jump as a multiple of the ordinary sample spacing.
 *
 * 1.0 means blocks join as smoothly as samples within a block do, so the seam
 * is invisible. Anything well above 1 is a step recurring once per stride,
 * which in the output waveform is a periodic artifact.
 */
static double seam_ratio(const EogEpochConfig *cfg)
{
    EogEpoch *ep = eog_epoch_create(cfg);
    float out[STRIDE];

    double max_step = 0.0;
    double max_jump = 0.0;
    double prev_last = 0.0;
    bool have_prev = false;

    for (int i = 0; i < 3000; i++) {
        double v = 2000.0 + 10.0 * sin(i * 0.05);
        if (eog_epoch_push(ep, (float)v, out, STRIDE) != STRIDE) {
            continue;
        }
        double step = 0.0;
        for (int k = 1; k < STRIDE; k++) {
            step += fabs((double)out[k] - (double)out[k - 1]);
        }
        step /= (STRIDE - 1);
        if (step > max_step) max_step = step;

        if (have_prev) {
            double jump = fabs((double)out[0] - prev_last);
            if (jump > max_jump) max_jump = jump;
        }
        prev_last = (double)out[STRIDE - 1];
        have_prev = true;
    }

    eog_epoch_destroy(ep);
    return (max_step > 0.0) ? (max_jump / max_step) : 0.0;
}

static void test_boundary_continuity(void)
{
    EogEpochConfig plain = EOG_EPOCH_PER_EPOCH_CONFIG;
    EogEpochConfig swept = EOG_EPOCH_DEFAULT_CONFIG;

    double r_plain = seam_ratio(&plain);
    double r_swept = seam_ratio(&swept);

    TEST("Per-epoch scaling leaves a seam between blocks");
    if (r_plain > 1.2) PASS();
    else FAIL("expected a visible step without the sweep");

    TEST("Sweeping the centre removes the seam");
    if (r_swept < 1.2) PASS();
    else FAIL("blocks still step at the boundary");

    printf("      (seam: per-epoch %.1fx, default %.1fx)\n", r_plain, r_swept);
}

static void test_amplitude_preserved(void)
{
    /* The point of scaling from a longer window: a loud stretch should come out
     * louder than a quiet one. Scaling each epoch by its own deviation puts
     * both on the same spread, which is what loses the distinction.
     *
     * Feed alternating quiet and loud stretches, each several epochs long, and
     * compare the output magnitude between them. */
    const int RUN = 600;        /* samples per stretch */
    const int RUNS = 10;

    double ratio[2];
    EogEpochConfig cfgs[2];
    cfgs[0] = EOG_EPOCH_PER_EPOCH_CONFIG;
    cfgs[1] = EOG_EPOCH_DEFAULT_CONFIG;

    for (int which = 0; which < 2; which++) {
        EogEpoch *ep = eog_epoch_create(&cfgs[which]);
        float out[STRIDE];
        double quiet = 0.0, loud = 0.0;
        size_t nq = 0, nl = 0;

        for (int i = 0; i < RUN * RUNS; i++) {
            bool isLoud = ((i / RUN) % 2) == 1;
            double amp = isLoud ? 40.0 : 2.0;
            double v = 2000.0 + amp * sin(i * 0.25);
            int n = eog_epoch_push(ep, (float)v, out, STRIDE);
            if (n != STRIDE) {
                continue;
            }
            /* Attribute the block to the stretch its samples came from */
            for (int k = 0; k < n; k++) {
                if (isLoud) { loud += fabs((double)out[k]); nl++; }
                else        { quiet += fabs((double)out[k]); nq++; }
            }
        }
        eog_epoch_destroy(ep);
        ratio[which] = (nq && nl) ? ((loud / (double)nl) / (quiet / (double)nq))
                                  : 0.0;
    }

    TEST("Per-epoch scaling flattens loud against quiet");
    if (ratio[0] < 3.0) PASS();
    else FAIL("expected per-epoch scaling to normalize the difference away");

    TEST("A longer scale window keeps loud louder than quiet");
    if (ratio[1] > ratio[0] * 2.0) PASS();
    else FAIL("longer scale window did not preserve the amplitude difference");

    printf("      (loud/quiet: per-epoch %.2fx, default %.2fx)\n",
           ratio[0], ratio[1]);
}

static void test_scale_window_validation(void)
{
    TEST("Reject a scale window shorter than the epoch");
    EogEpochConfig cfg = EOG_EPOCH_DEFAULT_CONFIG;
    cfg.scale_window_samples = EPOCH - 1;
    if (eog_epoch_create(&cfg) == NULL) PASS();
    else FAIL("accepted a scale window smaller than the block it scales");

    TEST("Accept a scale window equal to the epoch");
    cfg.scale_window_samples = EPOCH;
    EogEpoch *ep = eog_epoch_create(&cfg);
    if (ep != NULL) PASS();
    else FAIL("rejected the boundary case");
    eog_epoch_destroy(ep);

    TEST("Scale reported separately from the epoch deviation");
    cfg = EOG_EPOCH_DEFAULT_CONFIG;
    ep = eog_epoch_create(&cfg);
    float out[STRIDE];
    for (int i = 0; i < 1200; i++) {
        double amp = (i > 600) ? 2.0 : 40.0;    /* loud, then quiet */
        eog_epoch_push(ep, (float)(2000.0 + amp * sin(i * 0.25)), out, STRIDE);
    }
    /* The epoch is quiet by now, but the longer window still remembers the
     * loud stretch, so the two must differ */
    if (eog_epoch_last_scale(ep) > eog_epoch_last_stddev(ep) * 1.5) PASS();
    else FAIL("scale tracked the epoch instead of the longer window");
    eog_epoch_destroy(ep);
}

static void test_latency_offset(void)
{
    /* An impulse at a known absolute index should surface in the output at the
     * offset the latency accessors advertise. */
    EogEpoch *ep = eog_epoch_create(NULL);
    float out[STRIDE];

    TEST("Latency accessors report kernel-derived bounds");
    if (eog_epoch_latency_min(ep) == HALF &&
        eog_epoch_latency_max(ep) == HALF + STRIDE - 1) PASS();
    else FAIL("unexpected latency bounds");

    const int impulse_at = 250;
    const int total = 500;
    /* The first emission covers epoch [0, EPOCH) and yields the slice starting
     * at EPOCH - STRIDE - HALF, so emitted samples begin at that absolute index
     * rather than at zero. */
    int absolute_out = EPOCH - STRIDE - HALF;
    int peak_index = -1;
    double peak_value = 0.0;

    for (int i = 0; i < total; i++) {
        float s = (i == impulse_at) ? 100.0f : 0.0f;
        int n = eog_epoch_push(ep, s, out, STRIDE);
        if (n == STRIDE) {
            for (int k = 0; k < STRIDE; k++) {
                if (fabs((double)out[k]) > fabs(peak_value)) {
                    peak_value = (double)out[k];
                    peak_index = absolute_out + k;
                }
            }
            absolute_out += STRIDE;
        }
    }

    TEST("Impulse appears at its own absolute index");
    if (peak_index == impulse_at) PASS();
    else {
        char m[96];
        snprintf(m, sizeof m, "peak at %d, expected %d", peak_index, impulse_at);
        FAIL(m);
    }

    eog_epoch_destroy(ep);
}

static void test_central_region_only(void)
{
    /* Every emitted sample must come from the filter's central region. Compare
     * the pipeline against applying the filter to the same epoch directly and
     * reading the same slice: they must agree exactly, and that slice must lie
     * within [half, epoch-half). */
    TEST("Emitted slice lies inside the central region");
    size_t begin = EPOCH - STRIDE - HALF;
    size_t end   = begin + STRIDE;
    if (begin >= HALF && end <= EPOCH - HALF) PASS();
    else FAIL("slice touches the edge region");

    EogEpoch *ep = eog_epoch_create(NULL);
    float out[STRIDE];

    /* Drive with an irregular signal so edge effects would be visible */
    float raw[EPOCH];
    for (int i = 0; i < EPOCH; i++) {
        raw[i] = (float)(1000.0 + 20.0 * sin(i * 0.3) + 5.0 * cos(i * 1.1));
    }

    int n = 0;
    for (int i = 0; i < EPOCH; i++) {
        n = eog_epoch_push(ep, raw[i], out, STRIDE);
    }

    /* Reference: standardize the same epoch, filter it, take the same slice */
    double sum = 0.0;
    for (int i = 0; i < EPOCH; i++) sum += raw[i];
    double mean = sum / EPOCH;
    double ss = 0.0;
    for (int i = 0; i < EPOCH; i++) { double d = raw[i] - mean; ss += d * d; }
    double sd = sqrt(ss / (EPOCH - 1));

    float z[EPOCH], ref[EPOCH];
    for (int i = 0; i < EPOCH; i++) z[i] = (float)((raw[i] - mean) / sd);

    SavgolConfig cfg = SAVGOL_SMOOTH(HALF, ORDER);
    SavgolFilter *f = savgol_create(&cfg);
    savgol_apply(f, z, ref, EPOCH);

    TEST("Pipeline output matches a direct standardize-then-filter");
    bool match = (n == STRIDE);
    for (int k = 0; k < STRIDE && match; k++) {
        if (fabs((double)out[k] - (double)ref[begin + k]) > 1e-5) match = false;
    }
    if (match) PASS();
    else FAIL("output diverges from the reference computation");

    savgol_destroy(f);
    eog_epoch_destroy(ep);
}

static void test_standardization_stats(void)
{
    EogEpoch *ep = eog_epoch_create(NULL);
    float out[STRIDE];

    /* Constant epoch: mean equals the value, stddev is zero */
    TEST("Flat epoch does not divide by zero");
    int n = 0;
    for (int i = 0; i < EPOCH; i++) {
        n = eog_epoch_push(ep, 1234.5f, out, STRIDE);
    }
    bool finite = (n == STRIDE);
    for (int k = 0; k < STRIDE && finite; k++) {
        if (!isfinite(out[k])) finite = false;
    }
    if (finite) PASS();
    else FAIL("non-finite output on flat input");

    TEST("Flat epoch reports its value as the mean");
    if (fabs((double)eog_epoch_last_mean(ep) - 1234.5) < 1e-2) PASS();
    else FAIL("wrong mean");

    TEST("Flat epoch reports zero deviation");
    if (fabs((double)eog_epoch_last_stddev(ep)) < 1e-3) PASS();
    else FAIL("wrong stddev");

    eog_epoch_destroy(ep);

    /* Known epoch: 0..99 has mean 49.5 */
    TEST("Ramp epoch reports the arithmetic mean");
    ep = eog_epoch_create(NULL);
    for (int i = 0; i < EPOCH; i++) {
        eog_epoch_push(ep, (float)i, out, STRIDE);
    }
    if (fabs((double)eog_epoch_last_mean(ep) - 49.5) < 1e-3) PASS();
    else FAIL("wrong mean on ramp");

    eog_epoch_destroy(ep);
}

static void test_baseline_rejection(void)
{
    /* A large DC baseline must not reach the output: standardization removes it.
     * Two runs of the same shape on different baselines must agree. */
    float out_a[STRIDE], out_b[STRIDE];
    EogEpoch *a = eog_epoch_create(NULL);
    EogEpoch *b = eog_epoch_create(NULL);

    int na = 0, nb = 0;
    for (int i = 0; i < EPOCH; i++) {
        double shape = 10.0 * sin(i * 0.2);
        na = eog_epoch_push(a, (float)(shape + 5.0),    out_a, STRIDE);
        nb = eog_epoch_push(b, (float)(shape + 5000.0), out_b, STRIDE);
    }

    TEST("Baseline offset does not reach the output");
    bool same = (na == STRIDE && nb == STRIDE);
    for (int k = 0; k < STRIDE && same; k++) {
        if (fabs((double)out_a[k] - (double)out_b[k]) > 1e-2) same = false;
    }
    if (same) PASS();
    else FAIL("output depends on the DC baseline");

    eog_epoch_destroy(a);
    eog_epoch_destroy(b);
}

static void test_reset(void)
{
    EogEpoch *ep = eog_epoch_create(NULL);
    float out[STRIDE];

    for (int i = 0; i < EPOCH; i++) {
        eog_epoch_push(ep, (float)i, out, STRIDE);
    }

    TEST("Reset clears readiness");
    eog_epoch_reset(ep);
    if (!eog_epoch_ready(ep)) PASS();
    else FAIL("still ready after reset");

    TEST("Pipeline refills cleanly after reset");
    int n = 0;
    for (int i = 0; i < EPOCH; i++) {
        n = eog_epoch_push(ep, (float)i, out, STRIDE);
    }
    if (n == STRIDE) PASS();
    else FAIL("no emission after refill");

    eog_epoch_destroy(ep);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("\nEpoch pipeline tests\n");
    printf("====================\n\n");

    printf("Configuration:\n");        test_config_validation();
    printf("\nEmission schedule:\n");   test_emission_schedule();
    printf("\nBlock alignment:\n");     test_block_alignment();
    printf("\nBoundary behaviour:\n");  test_boundary_continuity();
    printf("\nAmplitude:\n");           test_amplitude_preserved();
    printf("\nScale window:\n");        test_scale_window_validation();
    printf("\nLatency:\n");             test_latency_offset();
    printf("\nCentral region:\n");      test_central_region_only();
    printf("\nStandardization:\n");     test_standardization_stats();
    printf("\nBaseline rejection:\n");  test_baseline_rejection();
    printf("\nReset:\n");               test_reset();

    printf("\n=== Results: %d passed, %d failed ===\n\n", g_passed, g_failed);
    return (g_failed == 0) ? 0 : 1;
}
