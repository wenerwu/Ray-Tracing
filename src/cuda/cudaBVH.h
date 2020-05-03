#ifndef CUDA_BVH_H
#define CUDA_BVH_H

#include "../static_scene/scene.h"
#include "../bvh.h"

#include "cudaBBox.h"

class cudaBVHNode {
    public:
      cudaBVHNode()
  {
      start = 0;
      range = 0;
      
  } 

  cudaBVHNode(BBox bb, size_t start, size_t range)
      : bb(bb), start(start), range(range), l(NULL), r(NULL) {}



  __device__ inline bool isLeaf() const { return l == NULL && r == NULL; }

  cudaBBox bb;       ///< bounding box of the node
  size_t start;  ///< start index into the primitive list
  size_t range;  ///< range of index into the primitive list
  cudaBVHNode* l;    ///< left child node
  cudaBVHNode* r;    ///< right child node

};

#endif