#ifndef CUDA_VECTOR3D_H
#define CUDA_VECTOR3D_H

#include "CMU462/CMU462.h"

#include <ostream>
#include <cmath>



/**
 * Defines 3D vectors.
 */
class cudaVector3D {
 public:

  // components
  double x, y, z;

  /**
   * Constructor.
   * Initializes tp vector (0,0,0).
   */
  __device__ cudaVector3D() : x( 0.0 ), y( 0.0 ), z( 0.0 ) { }

  /**
   * Constructor.
   * Initializes to vector (x,y,z).
   */
  __device__ cudaVector3D( double x, double y, double z) : x( x ), y( y ), z( z ) { }

  /**
   * Constructor.
   * Initializes to vector (c,c,c)
   */
  __device__ cudaVector3D( double c ) : x( c ), y( c ), z( c ) { }

  /**
   * Constructor.
   * Initializes from existing vector
   */
  __device__ cudaVector3D( const cudaVector3D& v ) : x( v.x ), y( v.y ), z( v.z ) { }

  // returns reference to the specified component (0-based indexing: x, y, z)
  __device__ inline double& operator[] ( const int& index ) {
    return ( &x )[ index ];
  }

  // returns const reference to the specified component (0-based indexing: x, y, z)
  __device__ inline const double& operator[] ( const int& index ) const {
    return ( &x )[ index ];
  }

  __device__ inline bool operator==( const cudaVector3D& v) const {
    return v.x == x && v.y == y && v.z == z;
  }

  // negation
  __device__ inline cudaVector3D operator-( void ) const {
    return cudaVector3D( -x, -y, -z );
  }

  // addition
  __device__ inline cudaVector3D operator+( const cudaVector3D& v ) const {
    return cudaVector3D( x + v.x, y + v.y, z + v.z );
  }

  // subtraction
  __device__ inline cudaVector3D operator-( const cudaVector3D& v ) const {
    return cudaVector3D( x - v.x, y - v.y, z - v.z );
  }

  // right scalar multiplication
  __device__ inline cudaVector3D operator*( const double& c ) const {
    return cudaVector3D( x * c, y * c, z * c );
  }

  // scalar division
  __device__ inline cudaVector3D operator/( const double& c ) const {
    const double rc = 1.0/c;
    return cudaVector3D( rc * x, rc * y, rc * z );
  }

  // addition / assignment
  __device__ inline void operator+=( const cudaVector3D& v ) {
    x += v.x; y += v.y; z += v.z;
  }

  // subtraction / assignment
  __device__ inline void operator-=( const cudaVector3D& v ) {
    x -= v.x; y -= v.y; z -= v.z;
  }

  // scalar multiplication / assignment
  __device__ inline void operator*=( const double& c ) {
    x *= c; y *= c; z *= c;
  }

  // scalar division / assignment
  __device__ inline void operator/=( const double& c ) {
    (*this) *= ( 1./c );
  }

  /**
   * Returns Euclidean length.
   */
  __device__ inline double norm( void ) const {
    return sqrt( x*x + y*y + z*z );
  }

  /**
   * Returns Euclidean length squared.
   */
  __device__ inline double norm2( void ) const {
    return x*x + y*y + z*z;
  }

  /**
   * Returns unit vector.
   */
  __device__ inline cudaVector3D unit( void ) const {
    double rNorm = 1. / sqrt( x*x + y*y + z*z );
    return cudaVector3D( rNorm*x, rNorm*y, rNorm*z );
  }

  /**
   * Divides by Euclidean length.
   */
  __device__ inline void normalize( void ) {
    (*this) /= norm();
  }

}; // class cudaVector3D

// left scalar multiplication
__device__ inline cudaVector3D operator* ( const double& c, const cudaVector3D& v ) {
  return cudaVector3D( c * v.x, c * v.y, c * v.z );
}

// dot product (a.k.a. inner or scalar product)
__device__ inline double dot( const cudaVector3D& u, const cudaVector3D& v ) {
  return u.x*v.x + u.y*v.y + u.z*v.z ;
}

// cross product
__device__ inline cudaVector3D cross( const cudaVector3D& u, const cudaVector3D& v ) {
  return cudaVector3D( u.y*v.z - u.z*v.y,
                   u.z*v.x - u.x*v.z,
                   u.x*v.y - u.y*v.x );
}

#endif // CUDA__VECTOR3D_H
