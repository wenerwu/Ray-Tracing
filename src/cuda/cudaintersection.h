#ifndef CUDA_INTERSECT_H
#define CUDA_INTERSECT_H

#include <vector>

#include "cudaVector3D.h"
#include "cudaSpectrum.h"
#include "CMU462/misc.h"

#include "cudabsdf.h"
#include "cudaPrimitive.h"

const double INF = std::numeric_limits<double>::infinity();

class cudaPrimitive;
 
/**
 * A record of an intersection point which includes the time of intersection
 * and other information needed for shading
 */
class cudaIntersection { 
  public:
  __device__ cudaIntersection() : t(INF), primitive(NULL), bsdf(NULL) {}

  double t;  ///< time of intersection

  const cudaPrimitive* primitive;  ///< the primitive intersected

  cudaVector3D n;  ///< normal at point of intersection

  cudaBSDF* bsdf;  ///< BSDF of the surface at point of intersection

};


#endif  // CUDA_INTERSECT_H
