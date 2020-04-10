#include "sphere.h"

#include <cmath>

#include "../bsdf.h"
#include "../misc/sphere_drawing.h"

namespace CMU462 {
namespace StaticScene {

bool Sphere::test(const Ray& r, double& t1, double& t2) const {
  // TODO (PathTracer):
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.
  
  // |d|^2 * t^2 + 2(o*d - s_o)t + |o-s_o|^2 - r^2 = 0		c_o means origin of sphere
	double a = r.d.norm2();
	double b = 2 * dot(r.o - o, r.d);
	double c = (r.o - o).norm2() - r2;

	double delta = b * b - 4 * a * c;
	if (delta < 0)
		return false;
	t1 = (-b - sqrt(delta)) / (2 * a);
	t2 = (-b + sqrt(delta)) / (2 * a);
//	printf("%f %f\n", t1, t2);
	//if (t1 < r.min_t || t2 > r.max_t)
	//	return false;

	//r.max_t = t1;
 // return true;
	if (t1 >= r.min_t && t1 <= r.max_t)
	{
		r.max_t = t1;
		return true;
	}
	else if (t2 >= r.min_t && t2 <= r.max_t)
	{
		r.max_t = t2;
		return true;
	}
	return false;
}

bool Sphere::intersect(const Ray& r) const {
  // TODO (PathTracer):
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.
	double t1;
	double t2;
	return test(r, t1, t2);
 // return false;
}

bool Sphere::intersect(const Ray& r, Intersection* isect) const {
  // TODO (PathTracer):
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.
	double t1;
	double t2;
	if (!test(r, t1, t2))
		return false;
//	r.max_t = t1;
	isect->t = r.max_t;
	isect->n = normal(r.o + r.max_t * r.d);
	if (dot(isect->n, r.d) > 0)
		isect->n *= -1;
	isect->primitive = this;
	isect->bsdf = get_bsdf();

	return true;
}

void Sphere::draw(const Color& c) const { Misc::draw_sphere_opengl(o, r, c); }

void Sphere::drawOutline(const Color& c) const {
  // Misc::draw_sphere_opengl(o, r, c);
}

}  // namespace StaticScene
}  // namespace CMU462
