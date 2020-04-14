#include "environment_light.h"

namespace CMU462 {
namespace StaticScene {

EnvironmentLight::EnvironmentLight(const HDRImageBuffer* envMap)
    : envMap(envMap) {
  // TODO: (PathTracer) initialize things here as needed
}

Spectrum EnvironmentLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight, float* pdf) const {
  // TODO: (PathTracer) Implement
	*pdf = 1.f / (4.f * PI);

	double Xi1 = (double)(std::rand()) / RAND_MAX * 2 - 1;
	double Xi2 = (double)(std::rand()) / RAND_MAX;	

	double theta = acos(Xi1);
	double phi = 2.0 * PI * Xi2;

	double xs = sinf(theta) * cosf(phi);
	double ys = sinf(theta) * sinf(phi);
	double zs = cosf(theta);
	
//	printf("%f %f\n", Xi2, phi);
	*wi = Vector3D(xs, ys, zs);
	Ray r = Ray(p, *wi);	

	//distToLight!!!TODO!!
	//printf("%f %f %f\n", xs, ys, zs);
	//return Spectrum(1, 0, 0);
	return sample_dir(r, theta, phi);
}

Spectrum EnvironmentLight::sample_dir(const Ray& r, double theta, double phi) const {
  // TODO: (PathTracer) Implement
	size_t w = envMap->w;
	size_t h = envMap->h;

	Vector3D N = Vector3D(0, 0, 1);
	Vector3D N2 = Vector3D(0, 1, 0);

	//double theta = acos(dot(r.d, N));
	//double phi = acos(dot(Vector3D(r.d.x, r.d.y, 0), N2));
	
	//printf("acos: %f %f\n", theta, phi);
	int x = phi / (2 * PI) * w;
	int y = theta / PI * h;

	//TODO!!!! interpolate
	Spectrum map1 = envMap->data[y * h + x];
	//printf("%f %f %f\n", map1.r, map1.g, map1.b);
	return envMap->data[y * h + x];



  //return Spectrum(0, 0, 0);
}

}  // namespace StaticScene
}  // namespace CMU462
