#include "mult4x4.h"
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

void Mult_4x4::multAB(int type){
  double *a=ma->data;
  double *b=mb->data;
  double *c=mc->data;
  int lda=ma->ld;
  int ldb=mb->ld;
  int ldc=mc->ld;
  int i, j;
    
  for ( j=0; j<_n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
      for ( i=0; i<_m; i+=4 ){        /* Loop over the rows of C */
          /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
            one routine (four inner products) */
          if(type==0){
            AddDot4x4( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
          }else if(type==1){
            AddDot4x4_register( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
          }else if(type==2){
            AddDot4x4_ptr( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
          }else if(type==3){
            AddDot4x4_sse( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
          }
        }
  }
}

void Mult_4x4::AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     In this version, we merge each set of four loops, computing four
     inner products simultaneously. */

  int p;

  for ( p=0; p<k; p++ ){
    /* First row */
    C( 0, 0 ) += A( 0, p ) * B( p, 0 );     
    C( 0, 1 ) += A( 0, p ) * B( p, 1 );     
    C( 0, 2 ) += A( 0, p ) * B( p, 2 );     
    C( 0, 3 ) += A( 0, p ) * B( p, 3 );     

    /* Second row */
    C( 1, 0 ) += A( 1, p ) * B( p, 0 );     
    C( 1, 1 ) += A( 1, p ) * B( p, 1 );     
    C( 1, 2 ) += A( 1, p ) * B( p, 2 );     
    C( 1, 3 ) += A( 1, p ) * B( p, 3 );     

    /* Third row */
    C( 2, 0 ) += A( 2, p ) * B( p, 0 );     
    C( 2, 1 ) += A( 2, p ) * B( p, 1 );     
    C( 2, 2 ) += A( 2, p ) * B( p, 2 );     
    C( 2, 3 ) += A( 2, p ) * B( p, 3 );     

    /* Fourth row */
    C( 3, 0 ) += A( 3, p ) * B( p, 0 );     
    C( 3, 1 ) += A( 3, p ) * B( p, 1 );     
    C( 3, 2 ) += A( 3, p ) * B( p, 2 );     
    C( 3, 3 ) += A( 3, p ) * B( p, 3 );     
  }
}


void Mult_4x4::AddDot4x4_register( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     In this version, we accumulate in registers and put A( 0, p ) in a register */

  int p;
  register double 
    /* hold contributions to
       C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) 
       C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ) 
       C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ) 
       C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )   */
       c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
       c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
       c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
       c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg,
    /* hold 
       A( 0, p ) 
       A( 1, p ) 
       A( 2, p ) 
       A( 3, p ) */
       a_0p_reg,
       a_1p_reg,
       a_2p_reg,
       a_3p_reg;

  c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
  c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
  c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
  c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

  for ( p=0; p<k; p++ ){
    a_0p_reg = A( 0, p );
    a_1p_reg = A( 1, p );
    a_2p_reg = A( 2, p );
    a_3p_reg = A( 3, p );

    /* First row */
    c_00_reg += a_0p_reg * B( p, 0 );     
    c_01_reg += a_0p_reg * B( p, 1 );     
    c_02_reg += a_0p_reg * B( p, 2 );     
    c_03_reg += a_0p_reg * B( p, 3 );     

    /* Second row */
    c_10_reg += a_1p_reg * B( p, 0 );     
    c_11_reg += a_1p_reg * B( p, 1 );     
    c_12_reg += a_1p_reg * B( p, 2 );     
    c_13_reg += a_1p_reg * B( p, 3 );     

    /* Third row */
    c_20_reg += a_2p_reg * B( p, 0 );     
    c_21_reg += a_2p_reg * B( p, 1 );     
    c_22_reg += a_2p_reg * B( p, 2 );     
    c_23_reg += a_2p_reg * B( p, 3 );     

    /* Four row */
    c_30_reg += a_3p_reg * B( p, 0 );     
    c_31_reg += a_3p_reg * B( p, 1 );     
    c_32_reg += a_3p_reg * B( p, 2 );     
    c_33_reg += a_3p_reg * B( p, 3 );     
  }

  C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
  C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
  C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
  C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
}

void Mult_4x4::AddDot4x4_ptr( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     In this version, we use pointer to track where in four columns of B we are */

  int p;
  register double 
    /* hold contributions to
       C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) 
       C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ) 
       C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ) 
       C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )   */
       c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,  
       c_10_reg,   c_11_reg,   c_12_reg,   c_13_reg,  
       c_20_reg,   c_21_reg,   c_22_reg,   c_23_reg,  
       c_30_reg,   c_31_reg,   c_32_reg,   c_33_reg,
    /* hold 
       A( 0, p ) 
       A( 1, p ) 
       A( 2, p ) 
       A( 3, p ) */
       a_0p_reg,
       a_1p_reg,
       a_2p_reg,
       a_3p_reg;
  double 
    /* Point to the current elements in the four columns of B */
    *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 

  c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
  c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
  c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
  c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

  for ( p=0; p<k; p++ ){
    a_0p_reg = A( 0, p );
    a_1p_reg = A( 1, p );
    a_2p_reg = A( 2, p );
    a_3p_reg = A( 3, p );
	  
    b_p0_pntr = &B( p, 0 );
    b_p1_pntr = &B( p, 1 );
    b_p2_pntr = &B( p, 2 );
    b_p3_pntr = &B( p, 3 );	  

    /* First row */
    c_00_reg += a_0p_reg * *b_p0_pntr;     
    c_01_reg += a_0p_reg * *b_p1_pntr;     
    c_02_reg += a_0p_reg * *b_p2_pntr;     
    c_03_reg += a_0p_reg * *b_p3_pntr;     

    /* Second row */
    c_10_reg += a_1p_reg * *b_p0_pntr;     
    c_11_reg += a_1p_reg * *b_p1_pntr;     
    c_12_reg += a_1p_reg * *b_p2_pntr;     
    c_13_reg += a_1p_reg * *b_p3_pntr;     

    /* Third row */
    c_20_reg += a_2p_reg * *b_p0_pntr;     
    c_21_reg += a_2p_reg * *b_p1_pntr;     
    c_22_reg += a_2p_reg * *b_p2_pntr;     
    c_23_reg += a_2p_reg * *b_p3_pntr;     

    /* Four row */
    c_30_reg += a_3p_reg * *b_p0_pntr++;     
    c_31_reg += a_3p_reg * *b_p1_pntr++;     
    c_32_reg += a_3p_reg * *b_p2_pntr++;     
    c_33_reg += a_3p_reg * *b_p3_pntr++;     
  }

  C( 0, 0 ) += c_00_reg;   C( 0, 1 ) += c_01_reg;   C( 0, 2 ) += c_02_reg;   C( 0, 3 ) += c_03_reg;
  C( 1, 0 ) += c_10_reg;   C( 1, 1 ) += c_11_reg;   C( 1, 2 ) += c_12_reg;   C( 1, 3 ) += c_13_reg;
  C( 2, 0 ) += c_20_reg;   C( 2, 1 ) += c_21_reg;   C( 2, 2 ) += c_22_reg;   C( 2, 3 ) += c_23_reg;
  C( 3, 0 ) += c_30_reg;   C( 3, 1 ) += c_31_reg;   C( 3, 2 ) += c_32_reg;   C( 3, 3 ) += c_33_reg;
}


typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

void Mult_4x4::AddDot4x4_sse( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */

  int p;

  v2df_t
    c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
    c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

  double 
    /* Point to the current elements in the four columns of B */
    *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 
    
  b_p0_pntr = &B( 0, 0 );
  b_p1_pntr = &B( 0, 1 );
  b_p2_pntr = &B( 0, 2 );
  b_p3_pntr = &B( 0, 3 );

  c_00_c_10_vreg.v = _mm_setzero_pd();   
  c_01_c_11_vreg.v = _mm_setzero_pd();
  c_02_c_12_vreg.v = _mm_setzero_pd(); 
  c_03_c_13_vreg.v = _mm_setzero_pd(); 
  c_20_c_30_vreg.v = _mm_setzero_pd();   
  c_21_c_31_vreg.v = _mm_setzero_pd();  
  c_22_c_32_vreg.v = _mm_setzero_pd();   
  c_23_c_33_vreg.v = _mm_setzero_pd(); 

  for ( p=0; p<k; p++ ){
    a_0p_a_1p_vreg.v = _mm_load_pd( (double *) &A( 0, p ) );
    a_2p_a_3p_vreg.v = _mm_load_pd( (double *) &A( 2, p ) );

    b_p0_vreg.v = _mm_loaddup_pd( (double *) b_p0_pntr++ );   /* load and duplicate */
    b_p1_vreg.v = _mm_loaddup_pd( (double *) b_p1_pntr++ );   /* load and duplicate */
    b_p2_vreg.v = _mm_loaddup_pd( (double *) b_p2_pntr++ );   /* load and duplicate */
    b_p3_vreg.v = _mm_loaddup_pd( (double *) b_p3_pntr++ );   /* load and duplicate */

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

  C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];  
  C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0]; 

  C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];  
  C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1]; 

  C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];  
  C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0]; 

  C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];  
  C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1]; 
}

