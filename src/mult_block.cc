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
