/*
 * Implementations for Joint Based Skeletons.
 *
 * Started on October 29th, 2015 by Bryce Summers.
 */

#include "joint.h"
#include "skeleton.h"
#include "mesh.h"

#include "GL/glew.h"

#include <iostream>

namespace CMU462 {
namespace DynamicScene {

BBox Joint::get_bbox() {
  BBox bbox;
  Vector3D p1 = position;
  Vector3D p2 = p1 + axis;
  bbox.expand(p1);
  bbox.expand(p2);
  return bbox;
}

Info Joint::getInfo() {
  Info info;

  if (!scene || !scene->selected.element) {
    info.push_back("JOINT");
  } else {
    info = scene->selected.element->getInfo();
  }

  return info;
}

void Joint::drag(double x, double y, double dx, double dy,
                 const Matrix4x4& modelViewProj) {}

StaticScene::SceneObject* Joint::get_static_object() { return nullptr; }

// The real calculation.
void Joint::calculateAngleGradient(Joint* goalJoint, Vector3D q) {
  // TODO (Animation) task 2B
  Vector3D p_theta = goalJoint->getEndPosInWorld();
  Vector3D diff = p_theta - q;

  Joint* current = this;

  while(current)
  {
    vector<Vector3D> rs;
    current->getAxes(rs);
    Vector3D p = goalJoint->getEndPosInWorld() - current->getBasePosInWorld();
  //  goalJoint->getAxes
    Vector3D J_theta_x = cross(rs[0], p);
    Vector3D J_theta_y = cross(rs[1], p);
    Vector3D J_theta_z = cross(rs[2], p);

    current->ikAngleGradient += Vector3D(dot(J_theta_x, diff), dot(J_theta_y, diff), dot(J_theta_z, diff));
    current = current->parent;
  }

}

// The constructor sets the dynamic angle and velocity of
// the joint to zero (at a perfect vertical with no motion)
Joint::Joint(Skeleton* s) : skeleton(s), capsuleRadius(0.05), renderScale(1.0) {
  scale = Vector3D(1., 1., 1.);
  scales.setValue(0, scale);
}

Vector3D Joint::getAngle(double time) { return rotations(time); }

void Joint::setAngle(double time, Vector3D value) {
  rotations.setValue(time, value);
}

bool Joint::removeAngle(double time) { return rotations.removeKnot(time, .1); }

void Joint::keyframe(double t) {
  positions.setValue(t, position);
  rotations.setValue(t, rotation);
  scales.setValue(t, scale);
  for (Joint* j : kids) j->keyframe(t);
}

void Joint::unkeyframe(double t) {
  positions.removeKnot(t, 0.1);
  rotations.removeKnot(t, 0.1);
  scales.removeKnot(t, 0.1);
  for (Joint* j : kids) j->unkeyframe(t);
}

void Joint::removeJoint(Scene* scene) {
  if (this == skeleton->root) return;

  for (auto childJoint : kids) {
    if (childJoint != this) childJoint->removeJoint(scene);
  }

  scene->removeObject(this);

  auto& kids = parent->kids;
  kids.erase(std::remove(kids.begin(), kids.end(), this), kids.end());

  auto& joints = skeleton->joints;
  joints.erase(std::remove(joints.begin(), joints.end(), this), joints.end());

  delete this;
}

void Joint::getAxes(vector<Vector3D>& axes) {
  Matrix4x4 T = getRotation();
  for (Joint* j = parent; j != nullptr; j = j->parent) {
    T = j->getRotation() * T;
  }
  T = skeleton->mesh->getRotation() * T;
  axes.resize(3);
  axes[0] = T * Vector3D(1., 0., 0.);
  axes[1] = T * Vector3D(0., 1., 0.);
  axes[2] = T * Vector3D(0., 0., 1.);
}

Matrix4x4 Joint::getTransformation() {
  /* TODO (Animation) Task 2a
  Initialize a 4x4 identity transformation matrix. Traverse the hierarchy
  starting from the parent of this joint all the way up to the root (root has
  parent of nullptr) and accumulate their transformations on the left side of
  your transformation matrix. Don't forget to apply a translation which extends
  along the axis of those joints. Finally, apply the mesh's transformation at
  the end.
  */

  Matrix4x4 T = Matrix4x4::identity();
  Joint* p = parent;
  while(p)
  {
    T = p->SceneObject::getTransformation() * Matrix4x4::translation(p->axis) * T;
    p = p->parent;
  }

  T = skeleton->mesh->getTransformation() * T;
  return T;
}

Matrix4x4 Joint::getBindTransformation() {
  Matrix4x4 T = Matrix4x4::identity();
  for (Joint* j = parent; j != nullptr; j = j->parent) {
    T = Matrix4x4::translation(j->axis) * T;
  }

  // Allow skeleton translation by taking root's position into account
  T = Matrix4x4::translation(skeleton->root->position) * T;

  return T;
}

Vector3D Joint::getBasePosInWorld() {
  /* TODO (Animation) Task 2a
  This should be fairly simple once you implement Joint::getTransform(). You can
  utilize the transformation returned by Joint::getTransform() to compute the
  base position in world coordinate frame.
  */
  Vector3D base = Vector3D();
  return getTransformation() * base;
}

Vector3D Joint::getEndPosInWorld() {
  /* TODO (Animation) Task 2a
  In addition to what you did for getBasePosInWorld(), you need to apply this
  joint's transformation and translate along this joint's axis to get the end
  position in world coordinate frame.
  */
  Vector3D base = Vector3D();  
  return getTransformation() * SceneObject::getTransformation() * Matrix4x4::translation(axis) * base;
}
}  // namespace DynamicScene
}  // namespace CMU462
