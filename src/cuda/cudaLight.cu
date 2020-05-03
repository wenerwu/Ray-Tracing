#include "cudaLight.h"




// namespace cudaLightType {
//     enum TYPE{ NONE, DIRECTIONAL, INFINITEHEMISPHERE, POINT, SPOT, AREA, SPHERE, MESH };
//   }


// Directional Light //

cudaDirectionalLight::cudaDirectionalLight(const cudaSpectrum& rad,
    const cudaVector3D& lightDir)
: radiance(rad) {
dirToLight = -lightDir.unit();
}

cudaSpectrum cudaDirectionalLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
     float* distToLight, float* pdf) const {
*wi = dirToLight;
*distToLight = INF_D;
*pdf = 1.0;
return radiance;
}

// Infinite Hemisphere Light //

cudaInfiniteHemisphereLight::cudaInfiniteHemisphereLight(const cudaSpectrum& rad)
: radiance(rad) {
sampleToWorld[0] = cudaVector3D(1, 0, 0);
sampleToWorld[1] = cudaVector3D(0, 0, -1);
sampleToWorld[2] = cudaVector3D(0, 1, 0);
}

cudaSpectrum cudaInfiniteHemisphereLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
            float* distToLight,
            float* pdf) const {
cudaVector3D dir = sampler.get_sample();
*wi = sampleToWorld * dir;
*distToLight = INF_D;
*pdf = 1.0 / (2.0 * M_PI);
return radiance;
}

// Point Light //

cudaPointLight::cudaPointLight(const cudaSpectrum& rad, const cudaVector3D& pos)
: radiance(rad), position(pos) {}

cudaSpectrum cudaPointLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
float* distToLight, float* pdf) const {
cudaVector3D d = position - p;
*wi = d.unit();
*distToLight = d.norm();
*pdf = 1.0;
return radiance;
}

// Spot Light //

cudaSpotLight::cudaSpotLight(const cudaSpectrum& rad, const cudaVector3D& pos,
const cudaVector3D& dir, float angle) {}

cudaSpectrum cudaSpotLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
float* distToLight, float* pdf) const {
return cudaSpectrum();
}

// Area Light //

cudaAreaLight::cudaAreaLight(const cudaSpectrum& rad, const cudaVector3D& pos,
const cudaVector3D& dir, const cudaVector3D& dim_x,
const cudaVector3D& dim_y)
: radiance(rad),
position(pos),
direction(dir),
dim_x(dim_x),
dim_y(dim_y),
area(dim_x.norm() * dim_y.norm()) {}

cudaSpectrum cudaAreaLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
float* distToLight, float* pdf) const {
Vector2D sample = sampler.get_sample() - Vector2D(0.5f, 0.5f);
cudaVector3D d = position + sample.x * dim_x + sample.y * dim_y - p;
float cosTheta = dot(d, direction);
float sqDist = d.norm2();
float dist = sqrt(sqDist);
*wi = d / dist;
*distToLight = dist;
*pdf = sqDist / (area * fabs(cosTheta));
return cosTheta < 0 ? radiance : cudaSpectrum();
};

// Sphere Light //

cudaSphereLight::cudaSphereLight(const cudaSpectrum& rad, const SphereObject* sphere) {}

cudaSpectrum cudaSphereLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
float* distToLight, float* pdf) const {
return cudaSpectrum();
}

// Mesh Light

cudaMeshLight::cudaMeshLight(const cudaSpectrum& rad, const Mesh* mesh) {}

cudaSpectrum cudaMeshLight::sample_L(const cudaVector3D& p, cudaVector3D* wi,
float* distToLight, float* pdf) const {
return cudaSpectrum();
}