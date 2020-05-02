#ifndef CUDA_TRIANGLE_H
#define CUDA_TRIANGLE_H

#include "../static_scene/object.h"
#include "cudabsdf.h"
#include "cudaRay.h"
#include "cudaintersection.h"

using namespace CMU462;
using namespace StaticScene;

/**
 * A single triangle from a mesh.
 * To save space, it holds a pointer back to the data in the original mesh
 * rather than holding the data itself. This means that its lifetime is tied
 * to that of the original mesh. The primitive may refer back to the mesh
 * object for other information such as normal, texcoord, material.
 */
class cudaTriangle : public cudaPrimitive {
 public:
  /**
   * Constructor.
   * Construct a mesh triangle with the given indicies into the triangle mesh.
   * \param mesh pointer to the mesh the triangle is in
   * \param v1 index of triangle vertex in the mesh's attribute arrays
   * \param v2 index of triangle vertex in the mesh's attribute arrays
   * \param v3 index of triangle vertex in the mesh's attribute arrays
   */
  cudaTriangle(const Mesh* mesh, vector<size_t>& v);
  cudaTriangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3);

  /**
   * Get the world space bounding box of the triangle.
   * \return world space bounding box of the triangle
   */
  BBox get_bbox() const;

  /**
   * Ray - Triangle intersection.
   * Check if the given ray intersects with the triangle, no intersection
   * information is stored.
   * \param r ray to test intersection with
   * \return true if the given ray intersects with the triangle,
             false otherwise
   */
__device__  bool intersect(const Ray& r) const;

  /**
   * Ray - Triangle intersection 2.
   * Check if the given ray intersects with the triangle, if so, the input
   * intersection data is updated to contain intersection information for the
   * point of intersection.
   * \param r ray to test intersection with
   * \param i address to store intersection info
   * \return true if the given ray intersects with the triangle,
             false otherwise
   */
 __device__ bool intersect(const cudaRay& r, cudaIntersection* i);
  
  /**
   * Get BSDF.
   * In the case of a triangle, the surface material BSDF is stored in
   * the mesh it belongs to.
   */
 // cudaBSDF* get_bsdf() const { return mesh->get_bsdf(); }
    
  /**
   * Draw with OpenGL (for visualizer)
   */
 // void draw(const Color& c) const;

  /**
   * Draw outline with OpenGL (for visualizer)
   */
 // void drawOutline(const Color& c) const;

 public:
  const Mesh* mesh;  ///< pointer to the mesh the triangle is a part of

  size_t v1;  ///< index into the mesh attribute arrays
  size_t v2;  ///< index into the mesh attribute arrays
  size_t v3;  ///< index into the mesh attribute arrays

  cudaVector3D p0;
  cudaVector3D p1;
  cudaVector3D p2; 

  cudaVector3D n0;
  cudaVector3D n1;
  cudaVector3D n2;
  vector<size_t> v;

};  // class Triangle


#endif  // CMU462_STATICSCENE_TRIANGLE_H
