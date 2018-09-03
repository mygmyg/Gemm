#ifndef MULT_BLOCK_H_
#define MULT_BLOCK_H_
#include "base_mult.h"

class Mult_Block : BaseMult
{
public:
  void multAB(int type); //type 0:1x4  1:register, 2:ptr,3:open
  Mult_Block(int m, int n, int k) : BaseMult(m, n, k){_mc=256;_kc=128;};

protected:
  int _mc;
  int _kc;
  void AddDot4x4_sse(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

  void InnerKernel(int m, int n, int k, double *a, int lda,
                   double *b, int ldb,
                   double *c, int ldc);
};

#endif