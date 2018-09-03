#include "mult1x4.h"

void Mult_1x4::multAB(int type){
  double *a=ma->data;
  double *b=mb->data;
  double *c=mc->data;
  int lda=ma->ld;
  int ldb=mb->ld;
  int ldc=mc->ld;

  int i, j;
    
  for ( j=0; j<_n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
      for ( i=0; i<_m; i+=1 ){        /* Loop over the rows of C */
          /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
            one routine (four inner products) */
          if(type==0){
            AddDot1x4( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
          }else if(type==1){
            AddDot1x4_register( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );  
          }else if(type==2){
            AddDot1x4_ptr( _k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );  
          
          }

      }
  }


}

void Mult_1x4::AddDot1x4( int k, double *a, int lda,  double *b, int ldb, 
                double *c, int ldc ){
  
  for (int p=0; p<k; p++ ){
        C( 0, 0 ) += A( 0, p ) * B( p, 0 );
        C( 0, 1 ) += A( 0, p ) * B( p, 1 );
        C( 0, 2 ) += A( 0, p ) * B( p, 2 );
        C( 0, 3 ) += A( 0, p ) * B( p, 3 );
    }
}

void Mult_1x4::AddDot1x4_register( int k, double *a, int lda,  double *b, int ldb, 
                          double *c, int ldc ){
  
   
    int p;
    register double
    /* hold contributions to
     C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) */
    c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,
    /* holds A( 0, p ) */
    a_0p_reg;
    
    c_00_reg = 0.0;
    c_01_reg = 0.0;
    c_02_reg = 0.0;
    c_03_reg = 0.0;
    
    for ( p=0; p<k; p++ ){
        a_0p_reg = A( 0, p );
        
        c_00_reg += a_0p_reg * B( p, 0 );
        c_01_reg += a_0p_reg * B( p, 1 );
        c_02_reg += a_0p_reg * B( p, 2 );
        c_03_reg += a_0p_reg * B( p, 3 );
    }
    
    C( 0, 0 ) += c_00_reg;
    C( 0, 1 ) += c_01_reg;
    C( 0, 2 ) += c_02_reg;
    C( 0, 3 ) += c_03_reg;
}


void Mult_1x4::AddDot1x4_ptr( int k, double *a, int lda,  double *b, int ldb, 
                double *c, int ldc ){

    int p;
    register double
    /* hold contributions to
     C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ) */
    c_00_reg,   c_01_reg,   c_02_reg,   c_03_reg,
    /* holds A( 0, p ) */
    a_0p_reg;
    double
    /* Point to the current elements in the four columns of B */
    *bp0_pntr, *bp1_pntr, *bp2_pntr, *bp3_pntr;
    
    bp0_pntr = &B( 0, 0 );
    bp1_pntr = &B( 0, 1 );
    bp2_pntr = &B( 0, 2 );
    bp3_pntr = &B( 0, 3 );
    
    c_00_reg = 0.0;
    c_01_reg = 0.0;
    c_02_reg = 0.0;
    c_03_reg = 0.0;
    
    for ( p=0; p<k; p++ ){
        a_0p_reg = A( 0, p );
        
        c_00_reg += a_0p_reg * *bp0_pntr++;
        c_01_reg += a_0p_reg * *bp1_pntr++;
        c_02_reg += a_0p_reg * *bp2_pntr++;
        c_03_reg += a_0p_reg * *bp3_pntr++;
    }
    
    C( 0, 0 ) += c_00_reg;
    C( 0, 1 ) += c_01_reg;
    C( 0, 2 ) += c_02_reg;
    C( 0, 3 ) += c_03_reg;

}