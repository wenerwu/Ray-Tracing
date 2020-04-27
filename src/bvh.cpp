#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"
#include <omp.h>
#include <iostream>
#include <stack>
#include <memory.h>


#define BUCKETSIZE 16
using namespace std;
void testCudaPrintf();
namespace CMU462 {
namespace StaticScene {


BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {
  this->primitives = _primitives;

  // TODO (PathTracer):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bb;
  for (size_t i = 0; i < primitives.size(); ++i) {
    bb.expand(primitives[i]->get_bbox());
  }

  root = new BVHNode(bb, 0, primitives.size());
  splitNode(primitives, root, max_leaf_size);


}


BVHAccel::~BVHAccel() {
  // TODO (PathTracer):
  // Implement a proper destructor for your BVH accelerator aggregate

	destroyNode(root);
}

int BVHAccel::compute_Bucket(Primitive *primitive, int bucketSize, int axis, int centroid, int leftMost, int rightMost)
{
	double center;
	switch (axis)
	{
	case 0:
		center = primitive->get_bbox().centroid().x;
		break;
	case 1:
		center = primitive->get_bbox().centroid().y;
		break;
	case 2:
		center = primitive->get_bbox().centroid().z;
		break;
	default:
		center = primitive->get_bbox().centroid().x;
		break;
	}

	int left = 0;
	int right = bucketSize - 1;

	while (left != right)
	{
	//	printf("%d %d\n", center, centroid);
		int mid = (left + right) / 2;
		if (center == centroid)
			return mid;
		if (center < centroid)
		{
			centroid = (leftMost + centroid) / 2;
			right = mid;
		}
		else
		{
			if (left == mid)
				return left;
			centroid = (rightMost + centroid) / 2;
			left = mid;
		}

	}

	// in case any accident
	if (left < 0 || left > bucketSize)
		return 0;
	return left;
}

void BVHAccel::destroyNode(BVHNode* node)
{
	if (!node)
		return;

	if (node->isLeaf())
	{
		delete(node);
		return;
	}
	
	destroyNode(node->l);
	destroyNode(node->r);

	delete(node);
}

bool compareX(const Primitive* a, const Primitive* b)
{
	return a->get_bbox().centroid().x > b->get_bbox().centroid().x;
}

bool compareY(const Primitive* a, const Primitive* b)
{
	return a->get_bbox().centroid().y > b->get_bbox().centroid().y;
}
bool compareZ(const Primitive* a, const Primitive* b)
{
	return a->get_bbox().centroid().z > b->get_bbox().centroid().z;
}


void BVHAccel::splitNode(const std::vector<Primitive *> &_primitives, BVHNode* node, size_t max_leaf_size)
{
	if (node->range <= max_leaf_size)
		return;

	//	printf("RANGE:%d\n", node->range);
	BBox leftbb, rightbb;
	size_t i;
	int start = node->start;
	int range = node->range;


	double SN = node->bb.surface_area();

	int step = range < BUCKETSIZE ? 1 : range/BUCKETSIZE + 1; 
	//step = std::min(step, (int)max_leaf_size);
	vector<double> min_cost(3, INF_D);
	vector<int> min_index(3, step);

	for (int axis = 0; axis < 3; axis++)
	{
		
		switch (axis)
		{
		case 0:
			sort(primitives.begin() + start, primitives.begin() + start + range, compareX);
			break;

		case 1:
			sort(primitives.begin() + start, primitives.begin() + start + range, compareY);
			break;

		case 2:
			sort(primitives.begin() + start, primitives.begin() + start + range, compareZ);
			break;

		default:
			break;
		}
		//double min_cost;
		//int min_index;
		for (i = step; i < range; i += step)
		{
			int j = 0;
			BBox left;
			BBox right;
			int leftCount = 0;
			int rightCount = 0;

			for (j = 0; j < i; j++)
			{
			//	printf("left:%d %g\n", start + j, primitives[start + j]->get_bbox().max.x);
				left.expand(primitives[start + j]->get_bbox());
				leftCount++;
			}
			for (; j < range; j++)
			{
				right.expand(primitives[start + j]->get_bbox());
				rightCount++;
			}

			double SAH = 0;

			if (leftCount)
				SAH += 1.0 * leftCount * left.surface_area();
			if (rightCount)
				SAH += 1.0 * rightCount * right.surface_area();


	//			printf("%d %d %d || %g   %g %g %g   %g %g %g\n", i, leftCount, rightCount, SAH, left.extent.x, left.extent.y, left.extent.z, right.extent.x, right.extent.y, right.extent.z);
			if (SAH < min_cost[axis])
			{
				min_cost[axis] = SAH;
				min_index[axis] = i;
				//			printf("HERE! %d %d\n", min_index, i);
			}
			
	//		printf("%d\n", step);
		}
	//	printf("\n");
	}

	int min_axis = 0;
	int min_SAH = min_cost[0];
	for (int axis = 1; axis < 3; axis++)
	{
		if (min_cost[axis] < min_SAH)
		{
			min_SAH = min_cost[axis];
			min_axis = axis;
		}
	}
	int index = min_index[min_axis];

	if(min_axis == 0)
		sort(primitives.begin() + start, primitives.begin() + start + range, compareX);
	else if(min_axis == 1)
		sort(primitives.begin() + start, primitives.begin() + start + range, compareY);

	for (i = 0; i < index; ++i) {

		leftbb.expand(primitives[i + start]->get_bbox());
	}
	for (; i < range; ++i)
	{
		rightbb.expand(primitives[i + start]->get_bbox());
	}
	//	printf("TEST:%d %d %d\n", index, range, max_leaf_size);
	BVHNode* leftNode = new BVHNode(leftbb, start, index);
	BVHNode* rightNode = new BVHNode(rightbb, start + index, range - index);


	//TODO: split in a reasonable way
//	for (i = 0; i < range / 2; ++i) {
//		
//		leftbb.expand(primitives[i + start]->get_bbox());
//	}
//	for (; i < range; ++i)
//	{
//		rightbb.expand(primitives[i + start]->get_bbox());
//	}
////	printf("TEST:%g", leftbb.extent.x);
//	BVHNode* leftNode = new BVHNode(leftbb, start, range / 2);
//	BVHNode* rightNode = new BVHNode(rightbb, start + range / 2, range - range / 2);

	node->l = leftNode;
	node->r = rightNode;

	splitNode(_primitives, leftNode, max_leaf_size);
	splitNode(_primitives, rightNode, max_leaf_size);

}

BBox BVHAccel::get_bbox() const { return root->bb; }

bool BVHAccel::intersectWithNode(const Ray &ray, Intersection *isect)
{

//	testCudaPrintf();
	BVHNode* node = root;
	
//	q.push(root);

	// double t0, t1;
	// if(!root->bb.intersect(ray, t0, t1))
	// 	return false;

	bool res = false;
	double lt0, lt1, rt0, rt1;

	int pid = 0;


	BVHNode* near;
	BVHNode* far;
	bool hit = false;
	while(true)
	{
		// when it's leaf, intersect directly
		if(node->isLeaf())
		{	

			for (size_t p = 0; p < node->range; ++p) {
				if (primitives[node->start + p]->intersect(ray, isect))
				{
					hit = true;
				}
			}
			if(s.empty())
			{
				break;
			}
				
			
			node = s.top();
			if(pid == 0)
			{
				s.pop();
			}
					
			
		}
		else
		{
			/* Parallel read ?*/
			int hitleft = (bool)node->l->bb.intersect(ray, lt0, lt1);
			int hitright = (bool)node->r->bb.intersect(ray, rt0, rt1);

			/* Use parallel and barrier to init */
			// for(int i = 0; i <= 3; i++)
			// 	M[i] = 0;
			M[pid] = 0;
			sum = 0;
			// TODO: barrier here

			M[2*hitleft + hitright] = 1;
			// TODO: barrier here


			
			/* Visit both children */
			if(M[3] || (M[1] && M[2]))
			{

				/* Decide which to go in first */
				
				M[pid] = 2 * (hitright && (rt0 < lt0)) - 1;

				/* TODO: PARLLEL SUM OVER HERE */
			//	#pragma omp atomic
				sum += M[pid];

			//	#pragma omp barrier
				
				if(sum < 0)
				{
					near = node->l;
					far = node->r;
				}
				else
				{
					near = node->r;
					far = node->l;
				}

				if(pid == 0)
					s.push(far);
				node = near;
				
				
			}
			else if(M[2])
			{
			//	printf("HERELEFT\n");
				node = node->l;
			}
				
			else if(M[1])
			{
			//	printf("HERERIGHT\n");
				node = node->r;
			}
				
			else
			{
			//	#pragma omp barrier	//TODO??

				if(s.empty())
				{
				//	printf("HERE2 \n");
					break;
				}
						
				node = s.top();
			//	#pragma omp barrier	//TODO??
				if(pid == 0)
				s.pop();
			//	#pragma omp barrier	//TODO??
			}
			
			
		}
		
	}
	
	return hit;

}

bool BVHAccel::intersectWithNode(BVHNode* node, const Ray &ray, Intersection *isect)const
{
	if (!node)
		return false;


	double t0, t1;
	
	if (!node->bb.intersect(ray, t0, t1))
		return false;
	
	if (t0 > isect->t)
		return false;
	
	if (node->isLeaf())
	{
		bool hit = false;
		for (size_t p = 0; p < node->range; ++p) {
			if (primitives[node->start + p]->intersect(ray, isect))
			{
				hit = true;
			}
		}
		return hit;
	}

	bool res = false;
	double lt0, lt1, rt0, rt1;
	res = node->l->bb.intersect(ray, lt0, lt1);
	if(!res)
		return intersectWithNode(node->r, ray, isect);
	res = node->r->bb.intersect(ray, rt0, rt1);
	if(!res)
		return intersectWithNode(node->l, ray, isect);

	BVHNode*  first = lt0 < rt0 ? node->l : node->r;
	BVHNode*  second = lt0 < rt0 ? node->r : node->l;

	double smallt = lt0 < rt0 ? lt0 : rt0;
	double bigt = lt0 < rt0 ? rt0 : lt0;

	res = false;
	res |= intersectWithNode(first, ray, isect);
	if (!res || bigt < ray.max_t);
	res |= intersectWithNode(second, ray, isect);
	
	return res;
}

bool BVHAccel::intersect(const Ray &ray) const {
  // TODO (PathTracer):
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.
	Intersection isect;
	return intersectWithNode(root, ray, &isect);
  //bool hit = false;
  //for (size_t p = 0; p < primitives.size(); ++p) {
  //  if (primitives[p]->intersect(ray)) hit = true;
  //}

  //return hit;
}

bool BVHAccel::intersect(const Ray &ray, Intersection *isect)  {
  // TODO (PathTracer):
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.

//	return intersectWithNode(ray, isect);
	return intersectWithNode(root, ray, isect);
  //bool hit = false;
  //for (size_t p = 0; p < primitives.size(); ++p) {
  //  if (primitives[p]->intersect(ray, isect)) hit = true;
  //}

  //return hit;
}

}  // namespace StaticScene
}  // namespace CMU462
