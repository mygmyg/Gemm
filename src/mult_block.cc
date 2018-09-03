#include "mult_block.h"
#include <mmintrin.h>
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3

typedef union {
  __m128d v;
  double d[2];
} v2df_t;

#define min(i, j) ((i) < (j) ? (i) : (j))

void Mult_Block::multAB(int type)
{

  double *a = ma->data;
  double *b = mb->data;
  double *c = mc->data;
  int lda = ma->ld;
  int ldb = mb->ld;
  int ldc = mc->ld;


  /* This time, we compute a mc x n block of C by a call to the InnerKernel */

  for (int p = 0; p < _k; p += _kc)
  {
    int pb = min(_k - p, _kc);
    for (int i = 0; i < _m; i += _mc)
    {
      int ib = min(_m - i, _mc);
      InnerKernel(ib, _n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
    }
  }
}

void Mult_Block::InnerKernel(int m, int n, int k, double *a, int lda,
                             double *b, int ldb,
                             double *c, int ldc)
{
  int i, j;

  for (j = 0; j < n; j += 4)
  { /* Loop over the columns of C, unrolled by 4 */
    for (i = 0; i < m; i += 4)
    { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      AddDot4x4_sse(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

void Mult_Block::AddDot4x4_sse(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
  int p;
  v2df_t
      c_00_c_10_vreg,
      c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,
      c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
      a_0p_a_1p_vreg,
      a_2p_a_3p_vreg,
      b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

  double
      /* Point to the current elements in the four columns of B */
      *b_p0_pntr,
      *b_p1_pntr, *b_p2_pntr, *b_p3_pntr;

  b_p0_pntr = &B(0, 0);
  b_p1_pntr = &B(0, 1);
  b_p2_pntr = &B(0, 2);
  b_p3_pntr = &B(0, 3);

  c_00_c_10_vreg.v = _mm_setzero_pd();
  c_01_c_11_vreg.v = _mm_setzero_pd();
  c_02_c_12_vreg.v = _mm_setzero_pd();
  c_03_c_13_vreg.v = _mm_setzero_pd();
  c_20_c_30_vreg.v = _mm_setzero_pd();
  c_21_c_31_vreg.v = _mm_setzero_pd();
  c_22_c_32_vreg.v = _mm_setzero_pd();
  c_23_c_33_vreg.v = _mm_setzero_pd();

  for (p = 0; p < k; p++)
  {
    a_0p_a_1p_vreg.v = _mm_load_pd((double *)&A(0, p));
    a_2p_a_3p_vreg.v = _mm_load_pd((double *)&A(2, p));

    b_p0_vreg.v = _mm_loaddup_pd((double *)b_p0_pntr++); /* load and duplicate */
    b_p1_vreg.v = _mm_loaddup_pd((double *)b_p1_pntr++); /* load and duplicate */
    b_p2_vreg.v = _mm_loaddup_pd((double *)b_p2_pntr++); /* load and duplicate */
    b_p3_vreg.v = _mm_loaddup_pd((double *)b_p3_pntr++); /* load and duplicate */

    /* First row and second rows */
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
    c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
    c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
    c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

    /* Third and fourth rows */
    c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
    c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
    c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
    c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
  }

  C(0, 0) += c_00_c_10_vreg.d[0];
  C(0, 1) += c_01_c_11_vreg.d[0];
  C(0, 2) += c_02_c_12_vreg.d[0];
  C(0, 3) += c_03_c_13_vreg.d[0];

  C(1, 0) += c_00_c_10_vreg.d[1];
  C(1, 1) += c_01_c_11_vreg.d[1];
  C(1, 2) += c_02_c_12_vreg.d[1];
  C(1, 3) += c_03_c_13_vreg.d[1];

  C(2, 0) += c_20_c_30_vreg.d[0];
  C(2, 1) += c_21_c_31_vreg.d[0];
  C(2, 2) += c_22_c_32_vreg.d[0];
  C(2, 3) += c_23_c_33_vreg.d[0];

  C(3, 0) += c_20_c_30_vreg.d[1];
  C(3, 1) += c_21_c_31_vreg.d[1];
  C(3, 2) += c_22_c_32_vreg.d[1];
  C(3, 3) += c_23_c_33_vreg.d[1];
}
