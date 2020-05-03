#ifndef CUDALIGHT_H
#define CUDALIGHT_H

#include "CMU462/CMU462.h"
#include "cudaSpectrum.h"
#include "cudaVector3D.h"
#include "cudaMatrix3x3.h"
#include "../sampler.h"  // UniformHemisphereSampler3D, UniformGridSampler2D
#include "../image.h"    // HDRImageBuffer

#include "cudaObject.h"  // Mesh, SphereObject


class cudaLight {
 public:
  virtual cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                            float* pdf) const = 0;
  virtual bool is_delta_light() const = 0;
};



class cudaComplexLight
{
  public:
  cudaComplexLight()
  {
     type = cudaLightType::TYPE::NONE;
    // radiance;
    // dirToLight;
    // sampleToWorld;
    // UniformHemisphereSampler3D sampler;
    // cudaVector3D position;
    // cudaVector3D direction;
    // float angle;
    // cudaVector3D dim_x;
    // cudaVector3D dim_y;
    // UniformGridSampler2D sampler;
    // float area;
    // const cudaSphereObject* sphere;
    // const Mesh* mesh;
  }
  
  bool is_delta_light() const { return true; }
  int get_type() const {return type; }
 
  public:
  int type;
  cudaSpectrum radiance;
  cudaVector3D dirToLight; 
  cudaMatrix3x3 sampleToWorld;
  UniformHemisphereSampler3D _3Dsampler; 
  cudaVector3D position;
  cudaVector3D direction;
  float angle;
  cudaVector3D dim_x;
  cudaVector3D dim_y;
  UniformGridSampler2D _2Dsampler;
  float area;
  cudaSphereObject* sphere;
  Mesh* mesh;
};

cudaComplexLight translateLight(SceneLight* light);

// Directional Light //

class cudaDirectionalLight : public cudaLight {
 public:
  cudaDirectionalLight(const cudaSpectrum& rad, const cudaVector3D& lightDir);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return true; }

 private:
  cudaSpectrum radiance;
  cudaVector3D dirToLight;

};  // class Directional Light

// Infinite Hemisphere Light //

class cudaInfiniteHemisphereLight : public cudaLight {
 public:
  cudaInfiniteHemisphereLight(const cudaSpectrum& rad);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return false; }

 private:
  cudaSpectrum radiance;
  Matrix3x3 sampleToWorld;
  UniformHemisphereSampler3D sampler;

};  // class InfiniteHemisphereLight

// Point Light //

class cudaPointLight : public cudaLight {
 public:
  cudaPointLight(const cudaSpectrum& rad, const cudaVector3D& pos);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return true; }

 private:
  cudaSpectrum radiance;
  cudaVector3D position;

};  // class PointLight

// Spot Light //

class cudaSpotLight : public cudaLight {
 public:
  cudaSpotLight(const cudaSpectrum& rad, const cudaVector3D& pos, const cudaVector3D& dir,
            float angle);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return true; }

 private:
  cudaSpectrum radiance;
  cudaVector3D position;
  cudaVector3D direction;
  float angle;

};  // class SpotLight

// Area Light //

class cudaAreaLight : public cudaLight {
 public:
  cudaAreaLight(const cudaSpectrum& rad, const cudaVector3D& pos, const cudaVector3D& dir,
            const cudaVector3D& dim_x, const cudaVector3D& dim_y);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return false; }

 private:
  cudaSpectrum radiance;
  cudaVector3D position;
  cudaVector3D direction;
  cudaVector3D dim_x;
  cudaVector3D dim_y;
  UniformGridSampler2D sampler;
  float area;

};  // class AreaLight

// Sphere Light //

class cudaSphereLight : public cudaLight {
 public:
  cudaSphereLight(const cudaSpectrum& rad, const cudaSphereObject* sphere);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return false; }

 private:
  const cudaSphereObject* sphere;
  cudaSpectrum radiance;
  UniformHemisphereSampler3D sampler;

};  // class SphereLight

// Mesh Light

class cudaMeshLight : public cudaLight {
 public:
  cudaMeshLight(const cudaSpectrum& rad, const Mesh* mesh);
  cudaSpectrum sample_L(const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
                    float* pdf) const;
  bool is_delta_light() const { return false; }

 private:
  const Mesh* mesh;
  cudaSpectrum radiance;

};  // class MeshLight


#endif