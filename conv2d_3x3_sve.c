#include <arm_sme.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define N 256
#define M 256

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1e6;
}

// ---------------------------------------------------------------------
// Scalar reference implementation
// ---------------------------------------------------------------------
void conv2d_scalar(const float *restrict input,
                   const float *restrict kernel,
                   float       *restrict output,
                   int rows, int cols)
{
    for (int r = 1; r < rows - 1; r++)
        for (int c = 1; c < cols - 1; c++) {
            float sum = 0.0f;
            for (int kr = 0; kr < 3; kr++)
                for (int kc = 0; kc < 3; kc++)
                    sum += input[(r+kr-1)*cols+(c+kc-1)] * kernel[kr*3+kc];
            output[r*cols+c] = sum;
        }
}

// ---------------------------------------------------------------------
// Version 1: self-ext + predicate tail handling 
// ---------------------------------------------------------------------
__arm_locally_streaming
void conv2d_pred_tail(const float *restrict input,
                      const float *restrict kernel,
                      float       *restrict output,
                      int rows, int cols)
{
    const int vl   = (int)svcntsw();               // number of 32-bit elements per vector
    const int step = vl - 2;                        // safe step for svext (need 2 extra elements)

    svbool_t pg_all   = svptrue_b32();              // always-true predicate
    svbool_t pg_store = svwhilelt_b32(0u, (uint32_t)step); // predicate for full-vector stores

    // Broadcast kernel coefficients
    svfloat32_t vk00 = svdup_n_f32(kernel[0]);
    svfloat32_t vk01 = svdup_n_f32(kernel[1]);
    svfloat32_t vk02 = svdup_n_f32(kernel[2]);
    svfloat32_t vk10 = svdup_n_f32(kernel[3]);
    svfloat32_t vk11 = svdup_n_f32(kernel[4]);
    svfloat32_t vk12 = svdup_n_f32(kernel[5]);
    svfloat32_t vk20 = svdup_n_f32(kernel[6]);
    svfloat32_t vk21 = svdup_n_f32(kernel[7]);
    svfloat32_t vk22 = svdup_n_f32(kernel[8]);

    for (int r = 1; r < rows - 1; r++) {
        const float *p0 = input + (r-1)*cols;
        const float *p1 = input +  r   *cols;
        const float *p2 = input + (r+1)*cols;
        float       *po = output + r*cols;

        for (int c = 1; c < cols - 1; c += step) {
            int remain = (cols - 1) - c;            // number of output columns left

            // Load predicate: need 2 extra elements for shifting
            svbool_t pg_load = (remain >= step)
                ? pg_all
                : svwhilelt_b32(0u, (uint32_t)(remain + 2));
            // Store predicate: only the actual output columns
            svbool_t pg_store_iter = (remain >= step)
                ? pg_store
                : svwhilelt_b32(0u, (uint32_t)remain);

            // Load three rows starting from column c-1
            svfloat32_t r0 = svld1_f32(pg_load, p0 + c - 1);
            svfloat32_t r1 = svld1_f32(pg_load, p1 + c - 1);
            svfloat32_t r2 = svld1_f32(pg_load, p2 + c - 1);

            // Generate shifted versions
            svfloat32_t r0_s1 = svext_f32(r0, r0, 1);
            svfloat32_t r0_s2 = svext_f32(r0, r0, 2);
            svfloat32_t r1_s1 = svext_f32(r1, r1, 1);
            svfloat32_t r1_s2 = svext_f32(r1, r1, 2);
            svfloat32_t r2_s1 = svext_f32(r2, r2, 1);
            svfloat32_t r2_s2 = svext_f32(r2, r2, 2);

            // Compute convolution
            svfloat32_t acc = svdup_n_f32(0.0f);
            acc = svmla_f32_x(pg_all, acc, r0,    vk00);
            acc = svmla_f32_x(pg_all, acc, r0_s1, vk01);
            acc = svmla_f32_x(pg_all, acc, r0_s2, vk02);
            acc = svmla_f32_x(pg_all, acc, r1,    vk10);
            acc = svmla_f32_x(pg_all, acc, r1_s1, vk11);
            acc = svmla_f32_x(pg_all, acc, r1_s2, vk12);
            acc = svmla_f32_x(pg_all, acc, r2,    vk20);
            acc = svmla_f32_x(pg_all, acc, r2_s1, vk21);
            acc = svmla_f32_x(pg_all, acc, r2_s2, vk22);

            // Store results (predicated to handle tail)
            svst1_f32(pg_store_iter, po + c, acc);
        }
    }
}

// ---------------------------------------------------------------------
// Version 2: 9 explicit loads (Algorithm 2 style)
// ---------------------------------------------------------------------
__arm_locally_streaming
void conv2d_load9(const float *restrict input,
                  const float *restrict kernel,
                  float       *restrict output,
                  int rows, int cols)
{
    const int vl = (int)svcntsw();                 // vector length

    svbool_t pg_all = svptrue_b32();

    // Broadcast kernel coefficients
    svfloat32_t vk00 = svdup_n_f32(kernel[0]);
    svfloat32_t vk01 = svdup_n_f32(kernel[1]);
    svfloat32_t vk02 = svdup_n_f32(kernel[2]);
    svfloat32_t vk10 = svdup_n_f32(kernel[3]);
    svfloat32_t vk11 = svdup_n_f32(kernel[4]);
    svfloat32_t vk12 = svdup_n_f32(kernel[5]);
    svfloat32_t vk20 = svdup_n_f32(kernel[6]);
    svfloat32_t vk21 = svdup_n_f32(kernel[7]);
    svfloat32_t vk22 = svdup_n_f32(kernel[8]);

    for (int r = 1; r < rows - 1; r++) {
        const float *p0 = input + (r-1)*cols;
        const float *p1 = input +  r   *cols;
        const float *p2 = input + (r+1)*cols;
        float       *po = output + r*cols;

        for (int c = 1; c < cols - 1; c += vl) {
            int remain = (cols - 1) - c;
            svbool_t pg_iter = (remain >= vl)
                ? pg_all
                : svwhilelt_b32(0u, (uint32_t)remain);

            // Load 9 separate vectors for the 3x3 neighborhood
            svfloat32_t r00 = svld1_f32(pg_iter, p0 + c - 1);
            svfloat32_t r01 = svld1_f32(pg_iter, p0 + c    );
            svfloat32_t r02 = svld1_f32(pg_iter, p0 + c + 1);
            svfloat32_t r10 = svld1_f32(pg_iter, p1 + c - 1);
            svfloat32_t r11 = svld1_f32(pg_iter, p1 + c    );
            svfloat32_t r12 = svld1_f32(pg_iter, p1 + c + 1);
            svfloat32_t r20 = svld1_f32(pg_iter, p2 + c - 1);
            svfloat32_t r21 = svld1_f32(pg_iter, p2 + c    );
            svfloat32_t r22 = svld1_f32(pg_iter, p2 + c + 1);

            // Accumulate
            svfloat32_t acc = svdup_n_f32(0.0f);
            acc = svmla_f32_x(pg_all, acc, r00, vk00);
            acc = svmla_f32_x(pg_all, acc, r01, vk01);
            acc = svmla_f32_x(pg_all, acc, r02, vk02);
            acc = svmla_f32_x(pg_all, acc, r10, vk10);
            acc = svmla_f32_x(pg_all, acc, r11, vk11);
            acc = svmla_f32_x(pg_all, acc, r12, vk12);
            acc = svmla_f32_x(pg_all, acc, r20, vk20);
            acc = svmla_f32_x(pg_all, acc, r21, vk21);
            acc = svmla_f32_x(pg_all, acc, r22, vk22);

            svst1_f32(pg_iter, po + c, acc);
        }
    }
}

// ---------------------------------------------------------------------
// Version 3: cyclic kernel + predicate merge (Algorithm 1 style)
// ---------------------------------------------------------------------
__arm_locally_streaming
void conv2d_algo1(const float *restrict input,
                  const float *restrict kernel,
                  float       *restrict output,
                  int rows, int cols)
{
    const int vl   = (int)svcntsw();
    const int step = vl - 2;                        // same safe step as before

    svbool_t pg_all   = svptrue_b32();
    svbool_t pg_store = svwhilelt_b32(0u, (uint32_t)step);

    // Prepare cyclic kernel vectors: each vector contains kernel coefficients
    // rotated according to lane index modulo 3.
    svfloat32_t vk_X0_r0, vk_X0_r1, vk_X0_r2;
    svfloat32_t vk_X1_r0, vk_X1_r1, vk_X1_r2;
    svfloat32_t vk_X2_r0, vk_X2_r1, vk_X2_r2;
    {
        float buf[64];                               // temporary buffer, large enough for any VL

        // X0: vectors for columns (c00, c01, c02) — original order, no shift.
        for (int i = 0; i < vl; i++) buf[i] = kernel[0 * 3 + ((i + 0) % 3)];
        vk_X0_r0 = svld1_f32(pg_all, buf);
        for (int i = 0; i < vl; i++) buf[i] = kernel[1 * 3 + ((i + 0) % 3)];
        vk_X0_r1 = svld1_f32(pg_all, buf);
        for (int i = 0; i < vl; i++) buf[i] = kernel[2 * 3 + ((i + 0) % 3)];
        vk_X0_r2 = svld1_f32(pg_all, buf);

        // X1: vectors for columns (c02, c00, c01) — shifted right by 1.
        for (int i = 0; i < vl; i++) buf[i] = kernel[0 * 3 + ((i + 2) % 3)];
        vk_X1_r0 = svld1_f32(pg_all, buf);
        for (int i = 0; i < vl; i++) buf[i] = kernel[1 * 3 + ((i + 2) % 3)];
        vk_X1_r1 = svld1_f32(pg_all, buf);
        for (int i = 0; i < vl; i++) buf[i] = kernel[2 * 3 + ((i + 2) % 3)];
        vk_X1_r2 = svld1_f32(pg_all, buf);

        // X2: vectors for columns (c01, c02, c00) — shifted left by 1.
        for (int i = 0; i < vl; i++) buf[i] = kernel[0 * 3 + ((i + 1) % 3)];
        vk_X2_r0 = svld1_f32(pg_all, buf);
        for (int i = 0; i < vl; i++) buf[i] = kernel[1 * 3 + ((i + 1) % 3)];
        vk_X2_r1 = svld1_f32(pg_all, buf);
        for (int i = 0; i < vl; i++) buf[i] = kernel[2 * 3 + ((i + 1) % 3)];
        vk_X2_r2 = svld1_f32(pg_all, buf);
    }

    // Predicates for lane selection: which lane belongs to which modulo class
    svbool_t pred_x0, pred_x1;
    {
        svuint32_t vi = svindex_u32(0u, 1u);        // lane indices: 0,1,2,...
        svuint32_t d3 = svdiv_u32_x(pg_all, vi, svdup_n_u32(3u));
        svuint32_t m3 = svsub_u32_x(pg_all, vi,
                            svmul_u32_x(pg_all, d3, svdup_n_u32(3u))); // lane index % 3
        pred_x0 = svcmpeq_n_u32(pg_all, m3, 0u);    // lanes with remainder 0
        pred_x1 = svcmpeq_n_u32(pg_all, m3, 1u);    // lanes with remainder 1
    }

    svfloat32_t zero = svdup_n_f32(0.0f);

    for (int r = 1; r < rows - 1; r++) {
        const float *p0 = input + (r - 1) * cols;
        const float *p1 = input +  r       * cols;
        const float *p2 = input + (r + 1) * cols;
        float       *po = output + r * cols;

        for (int c = 1; c < cols - 1; c += step) {
            int remain = (cols - 1) - c;

            svbool_t pg_load = (remain >= step)
                ? pg_all
                : svwhilelt_b32(0u, (uint32_t)(remain + 2));
            svbool_t pg_st = (remain >= step)
                ? pg_store
                : svwhilelt_b32(0u, (uint32_t)remain);

            // Load three rows
            svfloat32_t d0 = svld1_f32(pg_load, p0 + c - 1);
            svfloat32_t d1 = svld1_f32(pg_load, p1 + c - 1);
            svfloat32_t d2 = svld1_f32(pg_load, p2 + c - 1);

            // Compute convolution for each modulo class separately,
            // then combine using svext to add contributions from adjacent lanes.
            svfloat32_t a0 = svmul_f32_x(pg_all, d0, vk_X0_r0);
            a0 = svmla_f32_x(pg_all, a0, d1, vk_X0_r1);
            a0 = svmla_f32_x(pg_all, a0, d2, vk_X0_r2);
            { svfloat32_t orig = a0;
              a0 = svadd_f32_x(pg_all, orig, svext_f32(orig, zero, 1));
              a0 = svadd_f32_x(pg_all, a0,   svext_f32(orig, zero, 2)); }

            svfloat32_t a1 = svmul_f32_x(pg_all, d0, vk_X1_r0);
            a1 = svmla_f32_x(pg_all, a1, d1, vk_X1_r1);
            a1 = svmla_f32_x(pg_all, a1, d2, vk_X1_r2);
            { svfloat32_t orig = a1;
              a1 = svadd_f32_x(pg_all, orig, svext_f32(orig, zero, 1));
              a1 = svadd_f32_x(pg_all, a1,   svext_f32(orig, zero, 2)); }

            svfloat32_t a2 = svmul_f32_x(pg_all, d0, vk_X2_r0);
            a2 = svmla_f32_x(pg_all, a2, d1, vk_X2_r1);
            a2 = svmla_f32_x(pg_all, a2, d2, vk_X2_r2);
            { svfloat32_t orig = a2;
              a2 = svadd_f32_x(pg_all, orig, svext_f32(orig, zero, 1));
              a2 = svadd_f32_x(pg_all, a2,   svext_f32(orig, zero, 2)); }

            // Select the correct result for each lane based on modulo class
            svfloat32_t merged = svsel_f32(pred_x0, a0,
                                    svsel_f32(pred_x1, a1, a2));
            svst1_f32(pg_st, po + c, merged);
        }
    }
}

// ---------------------------------------------------------------------
// Macro: single-row load9 convolution (used for tail rows in reg_block4)
// ---------------------------------------------------------------------
// Assumes pg_all, vk00..vk22, and vl are visible in the current scope.
#define CONV_SINGLE_ROW(p0_, p1_, p2_, po_)                             \
    do {                                                                 \
        for (int c = 1; c < cols - 1; c += vl) {                        \
            int _rem = (cols - 1) - c;                                   \
            svbool_t _pg = (_rem >= vl)                                  \
                ? pg_all : svwhilelt_b32(0u, (uint32_t)_rem);            \
            svfloat32_t _r00 = svld1_f32(_pg, (p0_) + c - 1);           \
            svfloat32_t _r01 = svld1_f32(_pg, (p0_) + c    );           \
            svfloat32_t _r02 = svld1_f32(_pg, (p0_) + c + 1);           \
            svfloat32_t _r10 = svld1_f32(_pg, (p1_) + c - 1);           \
            svfloat32_t _r11 = svld1_f32(_pg, (p1_) + c    );           \
            svfloat32_t _r12 = svld1_f32(_pg, (p1_) + c + 1);           \
            svfloat32_t _r20 = svld1_f32(_pg, (p2_) + c - 1);           \
            svfloat32_t _r21 = svld1_f32(_pg, (p2_) + c    );           \
            svfloat32_t _r22 = svld1_f32(_pg, (p2_) + c + 1);           \
            svfloat32_t _acc = svdup_n_f32(0.0f);                        \
            _acc = svmla_f32_x(pg_all, _acc, _r00, vk00);               \
            _acc = svmla_f32_x(pg_all, _acc, _r01, vk01);               \
            _acc = svmla_f32_x(pg_all, _acc, _r02, vk02);               \
            _acc = svmla_f32_x(pg_all, _acc, _r10, vk10);               \
            _acc = svmla_f32_x(pg_all, _acc, _r11, vk11);               \
            _acc = svmla_f32_x(pg_all, _acc, _r12, vk12);               \
            _acc = svmla_f32_x(pg_all, _acc, _r20, vk20);               \
            _acc = svmla_f32_x(pg_all, _acc, _r21, vk21);               \
            _acc = svmla_f32_x(pg_all, _acc, _r22, vk22);               \
            svst1_f32(_pg, (po_) + c, _acc);                             \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------
// Version 4: register blocking with 4 rows (load 18 vectors for 4*vl outputs)
// ---------------------------------------------------------------------
// Register usage analysis (SVE has 32 Z registers):
//   Input data: (4+2) rows × 3 offsets = 18 vectors
//   Accumulators: 4 vectors
//   Kernel coefficients: 9 vectors (broadcasted once, kept in registers)
//   Total: 18 + 4 + 9 = 31 → fits within 32, no spilling.
//
// Load savings compared to load9:
//   load9 per 4 rows: 4 × 9 = 36 loads for 4*vl outputs
//   reg_block4: 18 loads for 4*vl outputs → 50% reduction
//
// Data reuse pattern:
//   Row r   : uses R0,R1,R2
//   Row r+1 : uses R1,R2,R3  ← R1,R2 reused
//   Row r+2 : uses R2,R3,R4  ← R2,R3 reused
//   Row r+3 : uses R3,R4,R5  ← R3,R4 reused
//   R0 and R5 are used only once.
// ---------------------------------------------------------------------
__arm_locally_streaming
void conv2d_reg_block4(const float *restrict input,
                       const float *restrict kernel,
                       float       *restrict output,
                       int rows, int cols)
{
    const int vl    = (int)svcntsw();
    svbool_t pg_all = svptrue_b32();

    // Broadcast kernel coefficients (kept in registers throughout)
    svfloat32_t vk00 = svdup_n_f32(kernel[0]);
    svfloat32_t vk01 = svdup_n_f32(kernel[1]);
    svfloat32_t vk02 = svdup_n_f32(kernel[2]);
    svfloat32_t vk10 = svdup_n_f32(kernel[3]);
    svfloat32_t vk11 = svdup_n_f32(kernel[4]);
    svfloat32_t vk12 = svdup_n_f32(kernel[5]);
    svfloat32_t vk20 = svdup_n_f32(kernel[6]);
    svfloat32_t vk21 = svdup_n_f32(kernel[7]);
    svfloat32_t vk22 = svdup_n_f32(kernel[8]);

    // ---- Main loop: process 4 output rows per iteration ----
    int r;
    for (r = 1; r + 3 < rows - 1; r += 4) {
        // Input rows: R0 = row(r-1), R1 = row r, ..., R5 = row(r+4)
        const float *R0 = input + (r - 1) * cols;
        const float *R1 = input +  r       * cols;
        const float *R2 = input + (r + 1) * cols;
        const float *R3 = input + (r + 2) * cols;
        const float *R4 = input + (r + 3) * cols;
        const float *R5 = input + (r + 4) * cols;
        float       *O0 = output +  r       * cols;
        float       *O1 = output + (r + 1) * cols;
        float       *O2 = output + (r + 2) * cols;
        float       *O3 = output + (r + 3) * cols;

        for (int c = 1; c < cols - 1; c += vl) {
            int remain       = (cols - 1) - c;
            svbool_t pg_iter = (remain >= vl)
                ? pg_all
                : svwhilelt_b32(0u, (uint32_t)remain);

            // ---- Load 6 rows × 3 offsets = 18 vectors ----
            svfloat32_t d00 = svld1_f32(pg_iter, R0 + c - 1);
            svfloat32_t d01 = svld1_f32(pg_iter, R0 + c    );
            svfloat32_t d02 = svld1_f32(pg_iter, R0 + c + 1);

            svfloat32_t d10 = svld1_f32(pg_iter, R1 + c - 1);
            svfloat32_t d11 = svld1_f32(pg_iter, R1 + c    );
            svfloat32_t d12 = svld1_f32(pg_iter, R1 + c + 1);

            svfloat32_t d20 = svld1_f32(pg_iter, R2 + c - 1);
            svfloat32_t d21 = svld1_f32(pg_iter, R2 + c    );
            svfloat32_t d22 = svld1_f32(pg_iter, R2 + c + 1);

            svfloat32_t d30 = svld1_f32(pg_iter, R3 + c - 1);
            svfloat32_t d31 = svld1_f32(pg_iter, R3 + c    );
            svfloat32_t d32 = svld1_f32(pg_iter, R3 + c + 1);

            svfloat32_t d40 = svld1_f32(pg_iter, R4 + c - 1);
            svfloat32_t d41 = svld1_f32(pg_iter, R4 + c    );
            svfloat32_t d42 = svld1_f32(pg_iter, R4 + c + 1);

            svfloat32_t d50 = svld1_f32(pg_iter, R5 + c - 1);
            svfloat32_t d51 = svld1_f32(pg_iter, R5 + c    );
            svfloat32_t d52 = svld1_f32(pg_iter, R5 + c + 1);

            // ---- Output row r (uses R0,R1,R2) ----
            svfloat32_t acc0 = svdup_n_f32(0.0f);
            acc0 = svmla_f32_x(pg_all, acc0, d00, vk00);
            acc0 = svmla_f32_x(pg_all, acc0, d01, vk01);
            acc0 = svmla_f32_x(pg_all, acc0, d02, vk02);
            acc0 = svmla_f32_x(pg_all, acc0, d10, vk10);
            acc0 = svmla_f32_x(pg_all, acc0, d11, vk11);
            acc0 = svmla_f32_x(pg_all, acc0, d12, vk12);
            acc0 = svmla_f32_x(pg_all, acc0, d20, vk20);
            acc0 = svmla_f32_x(pg_all, acc0, d21, vk21);
            acc0 = svmla_f32_x(pg_all, acc0, d22, vk22);
            svst1_f32(pg_iter, O0 + c, acc0);

            // ---- Output row r+1 (uses R1,R2,R3) ----
            svfloat32_t acc1 = svdup_n_f32(0.0f);
            acc1 = svmla_f32_x(pg_all, acc1, d10, vk00);
            acc1 = svmla_f32_x(pg_all, acc1, d11, vk01);
            acc1 = svmla_f32_x(pg_all, acc1, d12, vk02);
            acc1 = svmla_f32_x(pg_all, acc1, d20, vk10);
            acc1 = svmla_f32_x(pg_all, acc1, d21, vk11);
            acc1 = svmla_f32_x(pg_all, acc1, d22, vk12);
            acc1 = svmla_f32_x(pg_all, acc1, d30, vk20);
            acc1 = svmla_f32_x(pg_all, acc1, d31, vk21);
            acc1 = svmla_f32_x(pg_all, acc1, d32, vk22);
            svst1_f32(pg_iter, O1 + c, acc1);

            // ---- Output row r+2 (uses R2,R3,R4) ----
            svfloat32_t acc2 = svdup_n_f32(0.0f);
            acc2 = svmla_f32_x(pg_all, acc2, d20, vk00);
            acc2 = svmla_f32_x(pg_all, acc2, d21, vk01);
            acc2 = svmla_f32_x(pg_all, acc2, d22, vk02);
            acc2 = svmla_f32_x(pg_all, acc2, d30, vk10);
            acc2 = svmla_f32_x(pg_all, acc2, d31, vk11);
            acc2 = svmla_f32_x(pg_all, acc2, d32, vk12);
            acc2 = svmla_f32_x(pg_all, acc2, d40, vk20);
            acc2 = svmla_f32_x(pg_all, acc2, d41, vk21);
            acc2 = svmla_f32_x(pg_all, acc2, d42, vk22);
            svst1_f32(pg_iter, O2 + c, acc2);

            // ---- Output row r+3 (uses R3,R4,R5) ----
            svfloat32_t acc3 = svdup_n_f32(0.0f);
            acc3 = svmla_f32_x(pg_all, acc3, d30, vk00);
            acc3 = svmla_f32_x(pg_all, acc3, d31, vk01);
            acc3 = svmla_f32_x(pg_all, acc3, d32, vk02);
            acc3 = svmla_f32_x(pg_all, acc3, d40, vk10);
            acc3 = svmla_f32_x(pg_all, acc3, d41, vk11);
            acc3 = svmla_f32_x(pg_all, acc3, d42, vk12);
            acc3 = svmla_f32_x(pg_all, acc3, d50, vk20);
            acc3 = svmla_f32_x(pg_all, acc3, d51, vk21);
            acc3 = svmla_f32_x(pg_all, acc3, d52, vk22);
            svst1_f32(pg_iter, O3 + c, acc3);
        }
    }

    // ---- Tail: process remaining 1-3 rows with single-row load9 ----
    for (; r < rows - 1; r++) {
        const float *p0 = input + (r - 1) * cols;
        const float *p1 = input +  r       * cols;
        const float *p2 = input + (r + 1) * cols;
        float       *po = output + r * cols;
        CONV_SINGLE_ROW(p0, p1, p2, po);
    }
}

// ---------------------------------------------------------------------
// Verification helper
// ---------------------------------------------------------------------
int verify(const float *ref, const float *out, int rows, int cols, float tol) {
    int errors = 0;
    for (int r = 1; r < rows - 1; r++)
        for (int c = 1; c < cols - 1; c++) {
            float diff = fabsf(ref[r*cols+c] - out[r*cols+c]);
            if (diff > tol) {
                if (errors < 5)
                    printf("  MISMATCH at (%d,%d): ref=%.6f out=%.6f\n",
                           r, c, ref[r*cols+c], out[r*cols+c]);
                errors++;
            }
        }
    return errors;
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(void)
{
    const int rows = N, cols = M;
    const int total = rows * cols;

    float *input   = (float *)aligned_alloc(64, total * sizeof(float));
    float *out_ref = (float *)aligned_alloc(64, total * sizeof(float));
    float *out_b   = (float *)aligned_alloc(64, total * sizeof(float)); // pred_tail
    float *out_c   = (float *)aligned_alloc(64, total * sizeof(float)); // load9
    float *out_d   = (float *)aligned_alloc(64, total * sizeof(float)); // algo1
    float *out_e   = (float *)aligned_alloc(64, total * sizeof(float)); // reg_block4

    float kernel[16] __attribute__((aligned(64))) = {
        1/16.f, 2/16.f, 1/16.f,
        2/16.f, 4/16.f, 2/16.f,
        1/16.f, 2/16.f, 1/16.f,
        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
    };

    srand(42);
    for (int i = 0; i < total; i++)
        input[i] = (float)(rand() % 256);

    memset(out_ref, 0, total * sizeof(float));
    memset(out_b,   0, total * sizeof(float));
    memset(out_c,   0, total * sizeof(float));
    memset(out_d,   0, total * sizeof(float));
    memset(out_e,   0, total * sizeof(float));

    /* ---- Functional verification ---- */
    conv2d_scalar      (input, kernel, out_ref, rows, cols);
    conv2d_pred_tail   (input, kernel, out_b,   rows, cols);
    conv2d_load9       (input, kernel, out_c,   rows, cols);
    conv2d_algo1       (input, kernel, out_d,   rows, cols);
    conv2d_reg_block4  (input, kernel, out_e,   rows, cols);

    int e2 = verify(out_ref, out_b, rows, cols, 1e-4f);
    int e3 = verify(out_ref, out_c, rows, cols, 1e-4f);
    int e4 = verify(out_ref, out_d, rows, cols, 1e-4f);
    int e5 = verify(out_ref, out_e, rows, cols, 1e-4f);
    printf("[Verify] pred_tail=%s  load9=%s  algo1=%s  reg_block4=%s\n\n",
           e2==0?"PASS":"FAIL",
           e3==0?"PASS":"FAIL",
           e4==0?"PASS":"FAIL",
           e5==0?"PASS":"FAIL");

    /* ---- Performance measurement ---- */
    const int RUNS = 2000;
    struct timespec t0, t1;

#define BENCH(fn, out_buf) \
    clock_gettime(CLOCK_MONOTONIC, &t0); \
    for (int i = 0; i < RUNS; i++) fn(input, kernel, out_buf, rows, cols); \
    clock_gettime(CLOCK_MONOTONIC, &t1);

    BENCH(conv2d_scalar,      out_ref); double ms_scalar = elapsed_ms(t0,t1)/RUNS;
    BENCH(conv2d_pred_tail,   out_b);   double ms_ptail  = elapsed_ms(t0,t1)/RUNS;
    BENCH(conv2d_load9,       out_c);   double ms_load9  = elapsed_ms(t0,t1)/RUNS;
    BENCH(conv2d_algo1,       out_d);   double ms_algo1  = elapsed_ms(t0,t1)/RUNS;
    BENCH(conv2d_reg_block4,  out_e);   double ms_rb4    = elapsed_ms(t0,t1)/RUNS;

#undef BENCH

    long inner = (long)(rows-2)*(cols-2);
    int vl   = (int)svcntsw();
    int step = vl - 2;

    printf("[Performance] image=%dx%d  vl=%d  step(svext/algo1)=%d  step(load9/rb4)=%d\n",
           rows, cols, vl, step, vl);
    printf("  %-48s : %7.3f ms   %8.1f Mpix/s\n",
           "Scalar (reference)", ms_scalar, inner/ms_scalar/1e3);
    printf("  %-48s : %7.3f ms   %8.1f Mpix/s   speedup=%.2fx\n",
           "SVE self-ext + predicate tail",           ms_ptail, inner/ms_ptail/1e3, ms_scalar/ms_ptail);
    printf("  %-48s : %7.3f ms   %8.1f Mpix/s   speedup=%.2fx\n",
           "SVE load9 (Algorithm 2 style)",           ms_load9, inner/ms_load9/1e3, ms_scalar/ms_load9);
    printf("  %-48s : %7.3f ms   %8.1f Mpix/s   speedup=%.2fx\n",
           "SVE cyclic-kernel + pred-merge (Algo1)",  ms_algo1, inner/ms_algo1/1e3, ms_scalar/ms_algo1);
    printf("  %-48s : %7.3f ms   %8.1f Mpix/s   speedup=%.2fx\n",
           "SVE load9 + register blocking (4 rows)",  ms_rb4,   inner/ms_rb4/1e3,   ms_scalar/ms_rb4);

    printf("\n[Load analysis per col-iteration (vl=%d outputs per row)]\n", vl);
    printf("  load9      : 9  loads / %d  outputs  =  %.3f loads/output\n",    vl,   9.0/vl);
    printf("  reg_block4 : 18 loads / %d outputs  =  %.3f loads/output"
           "  (%.1fx fewer than load9)\n", 4*vl, 18.0/(4*vl), (9.0/vl)/(18.0/(4*vl)));

    free(input); free(out_ref);
    free(out_b);  free(out_c);
    free(out_d);  free(out_e);
    return 0;
}