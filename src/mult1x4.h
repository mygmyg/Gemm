#ifndef MULT1X4_H_
#define MULT1X4_H_
#include "base_mult.h"

class Mult_1x4:BaseMult{
public:
  void multAB(int type);//type 0:1x4  1:register, 2:ptr,3:open
  Mult_1x4(int m,int n,int k):BaseMult(m,n,k){};
private:
  void AddDot1x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
  void AddDot1x4_register( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
  void AddDot1x4_ptr( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );


};

#endif