#ifndef CUDA_MATRIX3X3_H
#define CUDA_MATRIX3X3_H


#include "cudaVector3D.h"

#include <iosfwd>


/**
 * Defines a 3x3 matrix.
 * 3x3 matrices are extremely useful in computer graphics.
 */
class cudaMatrix3x3 {

  public:

  // The default constructor.
  __device__ cudaMatrix3x3(void) { }

  // The default constructor.
  __device__ __host__ cudaMatrix3x3(Matrix3x3 matrix) {entries[0] = matrix[0];  entries[1] = matrix[1]; entries[2] = matrix[2];}

  // Constructor for row major form data.
  // Transposes to the internal column major form.
  // REQUIRES: data should be of size 9 for a 3 by 3 matrix..
   __device__ cudaMatrix3x3(double * data)
  {
    for( int i = 0; i < 3; i++ ) {
      for( int j = 0; j < 3; j++ ) {
	        // Transpostion happens within the () query.
	        (*this)(i,j) = data[i*3 + j];
      } 
    }
  }

  /**
   * Sets all elements to val.
   */
  void zero(double val = 0.0 );

  /**
   * Returns the determinant of A.
   */
  double det( void ) const;

  /**
   * Returns the Frobenius norm of A.
   */
  double norm( void ) const;

  /**
   * Returns the 3x3 identity matrix.
   */
  static cudaMatrix3x3 identity( void );

  /**
   * Returns a matrix representing the (left) cross product with u.
   */
  static cudaMatrix3x3 crossProduct( const cudaVector3D& u );

  /**
   * Returns the ith column.
   */
        cudaVector3D& column( int i );
  const cudaVector3D& column( int i ) const;



  /**
   * Returns the inverse of A.
   */
  cudaMatrix3x3 inv( void ) const;

  // accesses element (i,j) of A using 0-based indexing
    __device__       double& operator()( int i, int j ){return entries[j][i];} 
    __device__  const double& operator()( int i, int j ) const{return entries[j][i];}

  // accesses the ith column of A
  __device__      cudaVector3D& operator[]( int i ){return entries[i];};
  __device__ const cudaVector3D& operator[]( int i ) const{return entries[i];};


  /**
   * Returns the transpose of A.
   */
  __device__ cudaMatrix3x3 T( void ) const
  {
    const cudaMatrix3x3& A( *this );
    cudaMatrix3x3 B;

    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ )
    {
       B(i,j) = A(j,i);
    }

    return B;
  }

  // increments by B
  void operator+=( const cudaMatrix3x3& B );

  // returns -A
  cudaMatrix3x3 operator-( void ) const;

  // returns A-B
  cudaMatrix3x3 operator-( const cudaMatrix3x3& B ) const;

  // returns c*A
  cudaMatrix3x3 operator*( double c ) const;

  // returns A*B
  cudaMatrix3x3 operator*( const cudaMatrix3x3& B ) const;

  // returns A*x
  __device__ cudaVector3D operator*( const cudaVector3D& x ) const{
    return x[0]*entries[0] +
           x[1]*entries[1] +
           x[2]*entries[2] ;
  };

  // divides each element by x
  void operator/=( double x );

  protected:

  // column vectors
  cudaVector3D entries[3];

}; // class cudaMatrix3x3

// returns the outer product of u and v
cudaMatrix3x3 outer( const cudaVector3D& u, const cudaVector3D& v );

// returns c*A
cudaMatrix3x3 operator*( double c, const cudaMatrix3x3& A );

// prints entries
std::ostream& operator<<( std::ostream& os, const cudaMatrix3x3& A );

#endif // CUDA_MATRIX3X3_H
