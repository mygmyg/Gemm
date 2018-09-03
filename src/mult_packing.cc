#include "mult_packing.h"
#include <mmintrin.h>
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3
#include<iostream>

typedef union {
  __m128d v;
  double d[2];
} v2df_t;

#define min(i, j) ((i) < (j) ? (i) : (j))

void Mult_Packing::multAB(int type)
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
      if(type==0){
        InnerKernel_A(ib, _n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
      }else if(type==1){
        InnerKernel_AB(ib, _n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, i == 0);
      }
 
    }
  }
}

void Mult_Packing::InnerKernel_A(int m, int n, int k, double *a, int lda,
                               double *b, int ldb,
                               double *c, int ldc)
{
  int i, j;
  double
      packedA[m * k];

  for (j = 0; j < n; j += 4)
  { /* Loop over the columns of C, unrolled by 4 */
    for (i = 0; i < m; i += 4)
    { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if (j == 0)
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
      AddDot4x4_sse(k, &packedA[i * k], 4, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

void Mult_Packing::InnerKernel_AB(int m, int n, int k, double *a, int lda,
                                  double *b, int ldb,
                                  double *c, int ldc, int first_time)
{
  int i, j;
  double
      packedA[m * k];
  double
      packedB[_kc * _nb]; /* Note: using a static buffer is not thread safe... */

  for (j = 0; j < n; j += 4)
  { /* Loop over the columns of C, unrolled by 4 */
    if (first_time)
      // std::cout << "before" << j << std::endl;
      PackMatrixB(k, &B(0, j), ldb, &packedB[k*j]);
      // std::cout<<"ok"<<j<<std::endl;
    for (i = 0; i < m; i += 4)
    { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */
      if (j == 0)
        PackMatrixA(k, &A(i, 0), lda, &packedA[i * k]);
      AddDot4x4_sse(k, &packedA[i * k], 4, &packedB[j * k], k, &C(i, j), ldc);
    }
  }
}

void Mult_Packing::PackMatrixA(int k, double *a, int lda, double *a_to)
{
  int j;

  for (j = 0; j < k; j++)
  { /* loop over columns of A */
    double
        *a_ij_pntr = &A(0, j);

    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr + 1);
    *a_to++ = *(a_ij_pntr + 2);
    *a_to++ = *(a_ij_pntr + 3);
  }
}

void Mult_Packing::PackMatrixB(int k, double *b, int ldb, double *b_to)
{
  int i;
  double
      *b_i0_pntr = &B(0, 0),
      *b_i1_pntr = &B(0, 1),
      *b_i2_pntr = &B(0, 2), *b_i3_pntr = &B(0, 3);

  for (i = 0; i < k; i++)
  { /* loop over rows of B */
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}

