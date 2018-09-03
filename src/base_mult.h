#ifndef BASE_MULT_H_
#define BASE_MULT_H_
#include <assert.h>
#include <random>

using std::uniform_real_distribution;
using std::default_random_engine;

#define A( i, j ) a[ (j)*lda + (i) ]
#define B( i, j ) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

#if __cplusplus > 199711L
#define register // Deprecated in C++11.
#endif           // #if __cplusplus > 199711L

struct MyMatrix{
  
  int m;
  int n;
  int ld;
  int size;
  double *data;

  double operator () (int i,int j)const{ //GET value
          return data[(ld*j)+i];};
  double& operator () (int i,int j){  //SET value
          return data[(ld*j)+i];}
  
  MyMatrix(int row,int col){
    m=row;n=col;ld=row;
    size = (ld+1) * n;
    data=new double[size];
    
  }
  ~MyMatrix(){
    if(data){
      delete [] data;
      data=NULL;
    }
  }

  void randomMatrix(){
    double *ptr=data;
    default_random_engine e; 
    uniform_real_distribution<double> u(0, 100); //随机数分布对象 

    for(int i=0;i<size;i++){
      (*ptr++)=u(e);
    }
  }

};



class BaseMult{
public:
  BaseMult(int m,int n,int k);
  ~BaseMult();
  void multAB();
protected:

  int _m;
  int _n;
  int _k;

  MyMatrix* ma;
  MyMatrix* mb;
  MyMatrix* mc;
  
};

#endif