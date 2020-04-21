#include "bsdf.h"

#include <algorithm>
#include <iostream>
#include <utility>


using std::min;
using std::max;
using std::swap;

namespace CMU462 {

void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {
  Vector3D z = Vector3D(n.x, n.y, n.z);
  Vector3D h = z;
  if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
    h.x = 1.0;
  else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  z.normalize();
  Vector3D y = cross(h, z);
  y.normalize();
  Vector3D x = cross(z, y);
  x.normalize();

  o2w[0] = x;
  o2w[1] = y;
  o2w[2] = z;
}

// Diffuse BSDF //

Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return albedo * (1.0 / PI);
}

Spectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO (PathTracer):
  // Implement DiffuseBSDF
  *wi = sampler.get_sample(pdf);
  return f(wo, *wi);
}

// Mirror BSDF //

Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
 // return Spectrum();
	Vector3D normal = Vector3D(0, 0, 1);
	return reflectance * (fabs(1.0/ dot(wo, normal)));// dot(wo, normal));
}

Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO (PathTracer):
  // Implement MirrorBSDF
	reflect(wo, wi);
	*pdf = 1;

	return f(wo, *wi);	
}

// Glossy BSDF //

/*
Spectrum GlossyBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}
Spectrum GlossyBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0f;
  return reflect(wo, wi, reflectance);
}
*/

// Refraction BSDF //

Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi,
                                  float* pdf) {
  // TODO (PathTracer):
  // Implement RefractionBSDF
  return Spectrum();
}

// Glass BSDF //

Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
return Spectrum();

}
	
Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // TODO (PathTracer):
  // Compute Fresnel coefficient and either reflect or refract based on it.
	Vector3D normal = Vector3D(0, 0, 1);

	if (!refract(wo, wi, ior))
	{
		reflect(wo, wi);
		*pdf = 1;
		
		return reflectance  * (1.f / fabs(dot(*wi, normal)));
	} 
	
	double cosi = dot(wo, normal);
	double cost = dot(*wi, normal);
	
	float ni = ior;
	float nt = 1.f;
	if (cost < 0)
	{
		ni = 1.f;
		nt = ior;
	}	
	double sint = sqrt(1 - cost * cost);
	double sini = nt / ni * sint;	// another direction of total internal!!!!

	double Fr;
	if(sini > 1)
	{
		Fr = 1;
	}
	else
	{
		cosi = fabs(cosi);
		cost = fabs(cost);

		double r1 = (nt * cosi - ni * cost) / (nt * cosi + ni * cost);
		double r2 = (ni * cosi - nt * cost) / (ni * cosi + nt * cost);

		Fr = 0.5 * (r1 * r1 + r2 * r2);
	}
	


	float randomFloat = ((float)std::rand() / RAND_MAX);
	if (randomFloat < Fr)
	{
		*pdf = Fr;
		reflect(wo, wi);

		return reflectance * Fr * (1.f / fabs(dot(*wi, normal)));

	}
	else 
	{
		*pdf = 1 - Fr;
		return transmittance * (1 - Fr) * (ni * ni / nt / nt ) * (1.f / fabs(dot(*wi, normal))) ;
	}
}

void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {
  // TODO (PathTracer):
  // Implement reflection of wo about normal (0,0,1) and store result in wi.

	*wi = Vector3D(-wo.x, -wo.y, wo.z);

}

bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior) {
  // TODO (PathTracer):
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.
	Vector3D normal = Vector3D(0, 0, 1);
	float ni = ior;	
	float nt = 1.f;
	//cos_theta(*wi);
	double coso = dot(wo, normal);
	if (coso < 0)
	{
		ni = 1.f;
		nt = ior;
	//	coso = abs(coso);
	}

	double sino = sqrt(1 - coso * coso);

	double sini = sino * nt / ni;
	if (sini >= 1)
		return false; 
	
	double cosi = sqrt(1 - sini * sini);

	Vector3D zfrac = Vector3D(0, 0, wo.z);
	zfrac.normalize();
	Vector3D xfrac = Vector3D(wo.x, wo.y, 0); 
	xfrac.normalize();

	*wi = -zfrac * cosi -  xfrac * sini;

	return true;
}

// Emission BSDF //

Spectrum EmissionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum EmissionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *wi = sampler.get_sample(pdf);
  return Spectrum();
}

}  // namespace CMU462