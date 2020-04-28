#ifndef CUDA_SPECTRUM_H
#define CUDA_SPECTRUM_H

#include "CMU462/CMU462.h"
#include "CMU462/color.h"

/**
 * Encodes radiance & irradiance values by the intensity of each visible
 * spectrum. Note that this is not strictly an actual spectrum with all
 * wavelengths, but it gives us enough information as we can only sense
 * a particular wavelengths.
 */
 class cudaSpectrum {
 public:
  float r;  ///< intensity of red spectrum
  float g;  ///< intensity of green spectrum
  float b;  ///< intensity of blue spectrum

  /**
   * Parameterized Constructor.
   * Initialize from component values.
   * \param r Intensity of the red spectrum
   * \param g Intensity of the green spectrum
   * \param b Intensity of the blue spectrum
   */
   __device__ cudaSpectrum(float r = 0, float g = 0, float b = 0) : r(r), g(g), b(b) {}

  /**
   * Constructor.
   * Initialize from an 8-bit RGB color represented as a uint8_t array.
   * \param arr Array containing component values.
   */
   __device__ cudaSpectrum(const uint8_t *arr);

  // operators //

  __device__ inline cudaSpectrum operator+(const cudaSpectrum &rhs) const {
    return cudaSpectrum(r + rhs.r, g + rhs.g, b + rhs.b);
  }

  __device__ inline cudaSpectrum &operator+=(const cudaSpectrum &rhs) {
    r += rhs.r;
    g += rhs.g;
    b += rhs.b;
    return *this;
  }

  __device__ inline cudaSpectrum operator*(const cudaSpectrum &rhs) const {
    return cudaSpectrum(r * rhs.r, g * rhs.g, b * rhs.b);
  }

  __device__ inline cudaSpectrum &operator*=(const cudaSpectrum &rhs) {
    r *= rhs.r;
    g *= rhs.g;
    b *= rhs.b;
    return *this;
  }

  __device__ inline cudaSpectrum operator*(float s) const {
    return cudaSpectrum(r * s, g * s, b * s);
  }

  __device__ inline cudaSpectrum &operator*=(float s) {
    r *= s;
    g *= s;
    b *= s;
    return *this;
  }

  __device__ inline bool operator==(const cudaSpectrum &rhs) const {
    return r == rhs.r && g == rhs.g && b == rhs.b;
  }

  __device__ inline bool operator!=(const cudaSpectrum &rhs) const {
    return !operator==(rhs);
  }

  __device__ inline Color toColor() const {
    return Color(r, g, b, 1); 
  }

  __device__ inline float illum() const { 
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }

  __device__ static cudaSpectrum fromColor(const Color &c) {
    return cudaSpectrum(c.a * c.r, c.a * c.g, c.a * c.b);
  }


};  // class Spectrum

// Commutable scalar multiplication
__device__  inline cudaSpectrum operator*(float s, const cudaSpectrum &c) { return c * s; }



#endif  // CMU462_SPECTRUM_H
