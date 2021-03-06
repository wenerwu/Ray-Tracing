#include "triangle.h"

#include "CMU462/CMU462.h"
#include "GL/glew.h"

namespace CMU462 {
namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, vector<size_t>& v) : mesh(mesh), v(v) {}
Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3)
    : mesh(mesh), v1(v1), v2(v2), v3(v3) {}

BBox Triangle::get_bbox() const {
  // TODO (PathTracer):
  // compute the bounding box of the triangle
	Vector3D p0 = mesh->positions[v1];
	Vector3D p1 = mesh->positions[v2];
	Vector3D p2 = mesh->positions[v3];

	double minx = min(min(p0.x, p1.x), p2.x);
	double miny = min(min(p0.y, p1.y), p2.y);
	double minz = min(min(p0.z, p1.z), p2.z);
	double maxx = max(max(p0.x, p1.x), p2.x);
	double maxy = max(max(p0.y, p1.y), p2.y);
	double maxz = max(max(p0.z, p1.z), p2.z);

	BBox res = BBox(minx, miny, minz, maxx, maxy, maxz);
  return res;
}

bool Triangle::intersect(const Ray& r) const {
//	printf("HERE!");
  // TODO (PathTracer): implement ray-triangle intersection
	Vector3D p0 = mesh->positions[v1];
	Vector3D p1 = mesh->positions[v2];
	Vector3D p2 = mesh->positions[v3];

	Vector3D o = r.o;
	Vector3D d = r.d;

	Vector3D e1 = p1 - p0;
	Vector3D e2 = p2 - p0;
	Vector3D s = o - p0;

	double denominator = dot(cross(e1, d), e2);
	if (denominator == 0)
		return false;
	
	Vector3D numerator = Vector3D(-dot(cross(s, e2), d), dot(cross(e1, d), s), -dot(cross(s, e2), e1));
	Vector3D ans = numerator / denominator;

	// in triangle
	if (ans.x < 0 || ans.x > 1 || ans.y < 0 || ans.y > 1 ||
		1 - ans.x - ans.y < 0 || 1 - ans.x - ans.y > 1 ||
		ans.z < r.min_t || ans.z > r.max_t)
		return false;

	r.max_t = ans.z;

  return true;
}

bool Triangle::intersect(const Ray& r, Intersection* isect) const {
//	printf("HERE!");
  // TODO (PathTracer):
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly
	Vector3D p0 = mesh->positions[v1];
	Vector3D p1 = mesh->positions[v2];
	Vector3D p2 = mesh->positions[v3];

	Vector3D o = r.o;
	Vector3D d = r.d;

	Vector3D e1 = p1 - p0;
	Vector3D e2 = p2 - p0;
	Vector3D s = o - p0;

	double denominator = dot(cross(e1, d), e2);
	if (denominator == 0)
		return false;

	Vector3D numerator = Vector3D(-dot(cross(s, e2), d), dot(cross(e1, d), s), -dot(cross(s, e2), e1));
	Vector3D ans = numerator / denominator;
//	return true;
	// in triangle
	if (ans.x < 0 || ans.x > 1 || ans.y < 0 || ans.y > 1 ||
		1 - ans.x - ans.y < 0 || 1 - ans.x - ans.y > 1 ||
		ans.z < r.min_t || ans.z > r.max_t)
		return false;

	double u = ans.x;
	double v = ans.y;
	double t = ans.z;
	r.max_t = t;


	isect->t = t;
	isect->n = (1 - u - v) * mesh->normals[v1] + u * mesh->normals[v2] + v * mesh->normals[v3];
	if (dot(isect->n, r.d) > 0)
		isect->n *= -1;
	isect->primitive = this;
	isect->bsdf = get_bsdf();		// ?

  return true;
}

void Triangle::draw(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_TRIANGLES);
  glVertex3d(mesh->positions[v1].x, mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x, mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x, mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

void Triangle::drawOutline(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_LOOP);
  glVertex3d(mesh->positions[v1].x, mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x, mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x, mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

}  // namespace StaticScene
}  // namespace CMU462
