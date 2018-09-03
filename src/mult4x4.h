#ifndef MULT4X4_H_
#define MULT4X4_H_
#include "base_mult.h"

class Mult_4x4:public BaseMult{
public:
  void multAB(int type);//type 0:1x4  1:register, 2:ptr,3:open
  Mult_4x4(int m,int n,int k):BaseMult(m,n,k){};
protected:

  void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
  void AddDot4x4_register( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
  void AddDot4x4_ptr( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
  void AddDot4x4_sse( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );


};

#endif