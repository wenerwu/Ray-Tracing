#ifndef CUDA_PRIMITIVE_H
#define CUDA_PRIMITIVE_H

#include "CMU462/CMU462.h"
#include "cudaintersection.h"
#include "../bbox.h"

class cudaIntersection;
/**
 * The abstract base class primitive is the bridge between geometry processing
 * and the shading subsystem. As such, its interface contains methods related
 * to both.
 */
class cudaPrimitive {
 public:
  /**
   * Get the world space bounding box of the primitive.
   * \return world space bounding box of the primitive
   */
  virtual BBox get_bbox() const = 0;

  /**
   * Ray - Primitive intersection.
   * Check if the given ray intersects with the primitive, no intersection
   * information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  virtual bool cudaintersect(const Ray& r) const = 0;

  /**
   * Ray - Primitive intersection 2.
   * Check if the given ray intersects with the primitive, if so, the input
   * intersection data is updated to contain intersection information for the
   * point of intersection.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the primitive,
             false otherwise
   */
  virtual bool cudaintersect(const Ray& r, cudaIntersection* i)= 0;

  /**
   * Get BSDF.
   * Return the BSDF of the surface material of the primitive.
   * Note that the BSDFs are not stored in each primitive but in the
   * SceneObject the primitive belongs to.
   */
 // virtual cudaBSDF* get_bsdf() const = 0;

  /**
   * Draw with OpenGL (for visualization)
   * \param c desired highlight color
   */
 // virtual void draw(const Color& c) const = 0;

  /**
   * Draw outline with OpenGL (for visualization)
   * \param c desired highlight color
   */
//  virtual void drawOutline(const Color& c) const = 0;
};

#endif  // CUDA_PRIMITIVE_H
