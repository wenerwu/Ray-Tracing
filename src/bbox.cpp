#include "bbox.h"

#include "GL/glew.h"

#include <algorithm>
#include <iostream>

namespace CMU462 {

bool BBox::intersect(const Ray &r, double &t0, double &t1) const {
  // TODO (PathTracer):
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

	double tmin = std::max(std::max(tminx, tminy), tminz);
	double tmax = std::min(std::min(tmaxx, tmaxy), tmaxz);
	//printf("%g %g %g\n", tmin, tmax);
	bool res = tmin > tmax;// || tmin < r.min_t;// || tmax > r.max_t;
//	printf("%g %g %g %g\n", tmin, tmax, r.min_t, r.max_t);
	if (res)
		return false;

	t0 = tmin;
	t1 = tmax;
	
  return true;
}

void BBox::draw(Color c) const {
  glColor4f(c.r, c.g, c.b, c.a);

  // top
  glBegin(GL_LINE_STRIP);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(max.x, max.y, max.z);
  glEnd();

  // bottom
  glBegin(GL_LINE_STRIP);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, min.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glEnd();

  // side
  glBegin(GL_LINES);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(min.x, min.y, max.z);
  glEnd();
}

std::ostream &operator<<(std::ostream &os, const BBox &b) {
  return os << "BBOX(" << b.min << ", " << b.max << ")";
}

}  // namespace CMU462
