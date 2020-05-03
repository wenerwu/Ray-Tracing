#ifndef CUDA_BBOX_H
#define CUDA_BBOX_H

#include <utility>
#include <algorithm>

#include "CMU462/CMU462.h"

#include "cudaRay.h"
#include "../bbox.h"


/**
  * Axis-aligned bounding box.
  * An AABB is given by two positions in space, the min and the max. An addition
  * component, the extent of the bounding box is stored as it is useful in a lot
  * of the operations on bounding boxes.
  */
struct cudaBBox {
  Vector3D max;     ///< min corner of the bounding box
  Vector3D min;     ///< max corner of the bounding box
  Vector3D extent;  ///< extent of the bounding box (min -> max)

  /**
    * Constructor.
    * The default constructor creates a new bounding box which contains no
    * points.
    */
  cudaBBox() {
    max = Vector3D(-INF_D, -INF_D, -INF_D);
    min = Vector3D(INF_D, INF_D, INF_D);
    extent = max - min;
  }

  cudaBBox(BBox box) {
    max = box.max;
    min = box.min;
    extent = box.extent;
  }

  /**
    * Constructor.
    * Creates a bounding box that includes a single point.
    */
  cudaBBox(const Vector3D &p) : min(p), max(p) { extent = max - min; }

  /**
    * Constructor.
    * Creates a bounding box with given bounds.
    * \param min the min corner
    * \param max the max corner
    */
  cudaBBox(const Vector3D &min, const Vector3D &max) : min(min), max(max) {
    extent = max - min;
  }

  /**
    * Constructor.
    * Creates a bounding box with given bounds (component wise).
    */
  cudaBBox(const double minX, const double minY, const double minZ,
       const double maxX, const double maxY, const double maxZ) {
    min = Vector3D(minX, minY, minZ);
    max = Vector3D(maxX, maxY, maxZ);
    extent = max - min;
  }

  /**
    * Expand the bounding box to include another (union).
    * If the given bounding box is contained within *this*, nothing happens.
    * Otherwise *this* is expanded to the minimum volume that contains the
    * given input.
    * \param bbox the bounding box to be included
    */
  void expand(const BBox &bbox) {
    min.x = std::min(min.x, bbox.min.x);
    min.y = std::min(min.y, bbox.min.y);
    min.z = std::min(min.z, bbox.min.z);
    max.x = std::max(max.x, bbox.max.x);
    max.y = std::max(max.y, bbox.max.y);
    max.z = std::max(max.z, bbox.max.z);
    extent = max - min;
  }

  /**
    * Expand the bounding box to include a new point in space.
    * If the given point is already inside *this*, nothing happens.
    * Otherwise *this* is expanded to a minimum volume that contains the given
    * point.
    * \param p the point to be included
    */
  void expand(const Vector3D &p) {
    min.x = std::min(min.x, p.x);
    min.y = std::min(min.y, p.y);
    min.z = std::min(min.z, p.z);
    max.x = std::max(max.x, p.x);
    max.y = std::max(max.y, p.y);
    max.z = std::max(max.z, p.z);
    extent = max - min;
  }

  Vector3D centroid() const { return (min + max) / 2; }

  /**
    * Compute the surface area of the bounding box.
    * \return surface area of the bounding box.
    */
  double surface_area() const {
    if (empty()) return 0.0;
    return 2 *
           (extent.x * extent.z + extent.x * extent.y + extent.y * extent.z);
  }

  /**
    * Check if bounding box is empty.
    * Bounding box that has no size is considered empty. Note that since
    * bounding box are used for objects with positive volumes, a bounding
    * box of zero size (empty, or contains a single vertex) are considered
    * empty.
    */
  bool empty() const { return min.x > max.x || min.y > max.y || min.z > max.z; }

  /**
    * Ray - bbox intersection.
    * Intersects ray with bounding box, does not store shading information.
    * \param r the ray to intersect with
    * \param t0 lower bound of intersection time
    * \param t1 upper bound of intersection time
    */

  __device__ bool intersect(const cudaRay &r, double &t0, double &t1)
  {
     // Implement ray - bounding box intersection test
    // If the ray intersected the bounding box within the range given by
    // t0, t1, update t0 and t1 with the new intersection times.
    double tminx;
    double tminy;
    double tminz;
    double tmaxx;
    double tmaxy;
    double tmaxz;

    if (r.sign[0] == 0)
    {
      
      tminx = (min.x - r.o.x) * r.inv_d.x;
      tmaxx = (max.x - r.o.x) * r.inv_d.x;
    }
    else
    {
      tminx = (max.x - r.o.x)*r.inv_d.x;
      tmaxx = (min.x - r.o.x)*r.inv_d.x;
    //	printf("TEST:%g, %g\n", max.x, min.x);
    }

    if (r.sign[1] == 0)
    {
      tminy = (min.y - r.o.y) * r.inv_d.y;
      tmaxy = (max.y - r.o.y) * r.inv_d.y;
    }
    else
    {
      tminy = (max.y - r.o.y) * r.inv_d.y;
      tmaxy = (min.y - r.o.y) * r.inv_d.y;
    }

    if (r.sign[2] == 0)
    {
      tminz = (min.z - r.o.z) * r.inv_d.z;
      tmaxz = (max.z - r.o.z) * r.inv_d.z;
    }
    else
    {
      tminz = (max.z - r.o.z) * r.inv_d.z;
      tmaxz = (min.z - r.o.z) * r.inv_d.z;
    }

    double tmin = tminx > tminy ? tminx : tminy;
    tmin = tmin > tminz ? tmin : tminz;
//     std::max(std::max(tminx, tminy), tminz);
  //  double tmax = std::min(std::min(tmaxx, tmaxy), tmaxz);
    double tmax = tmaxx < tmaxy ? tmaxx : tmaxy;
    tmax = tmax < tmaxz ? tmax : tmaxz;
    bool res = tmin > tmax;

    if (res)
      return false;

    t0 = tmin;
    t1 = tmax;
    
    return true;
  }

  /**
    * Draw box wireframe with OpenGL.
    * \param c color of the wireframe
    */
  void draw(Color c) const;
};


#endif  // CUDA_BBOX_H
