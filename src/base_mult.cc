#include "base_mult.h"

#include <iostream>





BaseMult::BaseMult(int m,int n,int k):
_m(m),_n(n),_k(k)
{
  ma=new MyMatrix(m,k);
  mb=new MyMatrix(k,n);
  mc=new MyMatrix(m,n);

  ma->randomMatrix();
  mb->randomMatrix();
  // c->randomMatrix();

}
BaseMult::~BaseMult(){
  if(ma){
    delete  ma;
    ma=NULL;
  }
  if(mb){
    delete  mb;
    mb=NULL;
  }
  if(mc){
    delete  mc;
    mc=NULL;
  }
}


void BaseMult::multAB(){
  int i, j, p;
  double *a=ma->data;
  double *b=mb->data;
  double *c=mc->data;
  int lda=ma->ld;
  int ldb=mb->ld;
  int ldc=mc->ld;


    
    for ( i=0; i<_m; i++ ){
        for ( j=0; j<_n; j++ ){
            for ( p=0; p<_k; p++ ){
              // c->data[j*c->n+i]=0;
                C( i,j ) = C( i,j ) +  A( i,p ) * B( p,j );
            }
        }
    }
}
