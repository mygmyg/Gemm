#include<iostream>

#include <sys/time.h>
#include <time.h>

#include "base_mult.h"
#include "mult1x4.h"
#include "mult4x4.h"

#include "mult_block.h"
#include "mult_packing.h"
static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock()
{
    double         the_time, norm_sec;
    struct timeval tv;
    
    gettimeofday( &tv, NULL );
    
    if ( gtod_ref_time_sec == 0.0 )
        gtod_ref_time_sec = ( double ) tv.tv_sec;
    
    norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;
    
    the_time = norm_sec + tv.tv_usec * 1.0e-6;
    
    return the_time;
}

void testBaseMult(int m,int n,int k){
  BaseMult* mult=new BaseMult(m,n,k);
  double start_time=dclock();

  mult->multAB();

  double duration=dclock()-start_time;
  std::cout<<"Base mult :"<<duration<<std::endl;

  delete mult;
}

void testMult1x4(int m,int n,int k){
  Mult_1x4* mult=new Mult_1x4(m,n,k);
  double start_time=dclock();

  mult->multAB(0);

  double duration=dclock()-start_time;
  std::cout<<"mult1x4 type:0 mult :"<<duration<<std::endl;


  start_time=dclock();

  mult->multAB(1);

  duration=dclock()-start_time;
  std::cout<<"mult1x4 type:register mult :"<<duration<<std::endl;
  start_time=dclock();

  mult->multAB(2);

  duration=dclock()-start_time;
  std::cout<<"mult1x4 type:ptr mult :"<<duration<<std::endl;



  delete mult;
}

void testMult4x4(int m,int n,int k){
  Mult_4x4* mult=new Mult_4x4(m,n,k);
  double start_time=dclock();

  mult->multAB(0);

  double duration=dclock()-start_time;
  std::cout<<"mult4x4 type:0 mult :"<<duration<<std::endl;


  start_time=dclock();

  mult->multAB(1);

  duration=dclock()-start_time;
  std::cout<<"mult4x4 type:register mult :"<<duration<<std::endl;

  start_time=dclock();
  mult->multAB(2);
  duration=dclock()-start_time;
  std::cout<<"mult4x4 type:ptr mult :"<<duration<<std::endl;

  start_time=dclock();
  mult->multAB(3);
  duration=dclock()-start_time;
  std::cout<<"mult4x4 type:sse mult :"<<duration<<std::endl;

  delete mult;
}

void testMult_block(int m, int n, int k)
{
  Mult_Block *mult = new Mult_Block(m, n, k);
  double start_time = dclock();

  mult->multAB(0);

  double duration = dclock() - start_time;
  std::cout << "mult_block type:0 mult :" << duration << std::endl;

  start_time = dclock();
  delete mult;
}

void testMult_packing(int m, int n, int k)
{
  Mult_Packing *mult = new Mult_Packing(m, n, k);

  double start_time = dclock();
  mult->multAB(0);
  double duration = dclock() - start_time;
  std::cout << "mult_Packing type:0 pack A mult :" << duration << std::endl;

  delete mult;
}

int main(int argc, char const *argv[])
{
  int m,n,k;
  if(argc==1)
  {
    m=n=k=400;
  }
  else if(argc==2){
    m = n = k = atof(argv[1]);
  }
  std::cout<<"m:"<<m<<" ,n:"<<n<<" ,k:"<<k<<std::endl;

  

  testBaseMult(m,n,k);

  testMult1x4(m,n,k);
  testMult4x4(m,n,k);
  testMult_block(m,n,k);

  testMult_packing(m, n, k);

  return 0;
}


