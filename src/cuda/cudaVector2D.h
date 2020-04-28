#ifndef CUDA_VECTOR2D_H
#define CUDA_VECTOR2D_H
#include "CMU462/CMU462.h"

#include <ostream>
#include <cmath>

/**
 * Defines 2D vectors.
 */
class cudaVector2D {
 public:

  // components
  double x, y;

  /**
   * Constructor.
   * Initializes to vector (0,0).
   */
  cudaVector2D() : x( 0.0 ), y( 0.0 ) { }

  /**
   * Constructor.
   * Initializes to vector (a,b).
   */
  cudaVector2D( double x, double y ) : x( x ), y( y ) { }

  /**
   * Constructor.
   * Copy constructor. Creates a copy of the given vector.
   */
  cudaVector2D( const cudaVector2D& v ) : x( v.x ), y( v.y ) { }

  // additive inverse
  inline cudaVector2D operator-( void ) const {
    return cudaVector2D( -x, -y );
  }

  // addition
  inline cudaVector2D operator+( const cudaVector2D& v ) const {
    cudaVector2D u = *this;
    u += v;
    return u;
  }

  // subtraction
  inline cudaVector2D operator-( const cudaVector2D& v ) const {
    cudaVector2D u = *this;
    u -= v;
    return u;
  }

  // right scalar multiplication
  inline cudaVector2D operator*( double r ) const {
    cudaVector2D vr = *this;
    vr *= r;
    return vr;
  }

  // scalar division
  inline cudaVector2D operator/( double r ) const {
    cudaVector2D vr = *this;
    vr /= r;
    return vr;
  }

  // add v
  inline void operator+=( const cudaVector2D& v ) {
    x += v.x;
    y += v.y;
  }

  // subtract v
  inline void operator-=( const cudaVector2D& v ) {
    x -= v.x;
    y -= v.y;
  }

  // scalar multiply by r
  inline void operator*=( double r ) {
    x *= r;
    y *= r;
  }

  // scalar divide by r
  inline void operator/=( double r ) {
    x /= r;
    y /= r;
  }

  /**
   * Returns norm.
   */
  inline double norm( void ) const {
    return sqrt( x*x + y*y );
  }

  /**
   * Returns norm squared.
   */
  inline double norm2( void ) const {
    return x*x + y*y;
  }

  /**
   * Returns unit vector parallel to this one.
   */
  inline cudaVector2D unit( void ) const {
    return *this / this->norm();
  }


}; // clasd cudaVector2D

// left scalar multiplication
inline cudaVector2D operator*( double r, const cudaVector2D& v ) {
   return v*r;
}

// inner product
inline double dot( const cudaVector2D& v1, const cudaVector2D& v2 ) {
  return v1.x*v2.x + v1.y*v2.y;
}

// cross product
inline double cross( const cudaVector2D& v1, const cudaVector2D& v2 ) {
  return v1.x*v2.y - v1.y*v2.x;
}


#endif // CMU462_VECTOR2D_H
