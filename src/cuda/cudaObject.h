#ifndef CUDA_OBJECT_H
#define CUDA_OBJECT_H

#include "../halfEdgeMesh.h"



/**
 * A triangle mesh object.
 */
class cudaMesh : public SceneObject {
 public:
  /**
   * Constructor.
   * Construct a static mesh for rendering from halfedge mesh used in editing.
   * Note that this converts the input halfedge mesh into a collection of
   * world-space triangle primitives.
   */
  cudaMesh(const HalfedgeMesh& mesh, BSDF* bsdf);

  /**
   * Get all the primitives (Triangle) in the mesh.
   * Note that Triangle reference the mesh for the actual data.
   * \return all the primitives in the mesh
   */
  vector<Primitive*> get_primitives() const;

  /**
   * Get the BSDF of the surface material of the mesh.
   * \return BSDF of the surface material of the mesh
   */
  BSDF* get_bsdf() const;

  cudaVector3D* positions;  ///< position array
  cudaVector3D* normals;    ///< normal array

 private:
  BSDF* bsdf;  ///< BSDF of surface material

  vector<size_t> indices;  ///< triangles defined by indices
};

/**
 * A sphere object.
 */
class cudaSphereObject : public SceneObject {
 public:
  /**
  * Constructor.
  * Construct a static sphere for rendering from given parameters
  */
  cudaSphereObject(const Vector3D& o, double r, BSDF* bsdf);

  /**
  * Get all the primitives (Sphere) in the sphere object.
  * Note that Sphere reference the sphere object for the actual data.
  * \return all the primitives in the sphere object
  */
  std::vector<Primitive*> get_primitives() const;

  /**
   * Get the BSDF of the surface material of the sphere.
   * \return BSDF of the surface material of the sphere
   */
  BSDF* get_bsdf() const;

  Vector3D o;  ///< origin
  double r;    ///< radius

 private:
  BSDF* bsdf;  ///< BSDF of the sphere objects' surface material

};  // class SphereObject



#endif  // CUDA_OBJECT_H
