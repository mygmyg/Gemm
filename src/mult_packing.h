#ifndef MULT_PACKING_H_
#define MULT_PACKING_H_
#include "base_mult.h"
#include "mult4x4.h"

class Mult_Packing : public Mult_4x4
{
public:
  void multAB(int type); //type 0:1x4  1:register, 2:ptr,3:open
  Mult_Packing(int m, int n, int k) : Mult_4x4(m, n, k)
  {
    _mc = 256;
    _kc = 128;
    _nb = n;
  };

private:
int _kc;
int _mc;
int _nb;
void InnerKernel_A(int m, int n, int k, double *a, int lda,
                 double *b, int ldb,double *c, int ldc);
void InnerKernel_AB(int m, int n, int k, double *a, int lda,
                 double *b, int ldb,
                 double *c, int ldc, int first_time) ;
void PackMatrixA(int k, double *a, int lda, double *a_to);
void PackMatrixB(int k, double *b, int ldb, double *b_to);
};

#endif