#include "environment_light.h"
#include <algorithm>
#include <vector>

namespace CMU462 {
namespace StaticScene {

EnvironmentLight::EnvironmentLight(const HDRImageBuffer* envMap)
    : envMap(envMap) {
  // TODO: (PathTracer) initialize things here as needed
	size_t w = envMap->w;
	size_t h = envMap->h;

	double total = 0;
	for(int y = 0; y < h; y++)
	{
		for(int x = 0; x < w; x++)
		{
			double probability = envMap->data[x + y * w].illum();
		//	printf("%f\n",probability)
			std::vector<double> tmp = {probability, (double)x, (double)y};
			probabilityStack.push_back(tmp);
			total += probability;
		}
	} 
 
	std::sort(probabilityStack.begin(), probabilityStack.end(), std::greater<std::vector<double>  >());
	double last = 0;
	for(int i = 0; i < probabilityStack.size(); i++)
	{
		probabilityStack[i][0] = last + probabilityStack[i][0] / total; 
		last = probabilityStack[i][0];
	//	printf("%g\n",probabilityStack[i][0]);
	}
}

Vector2D EnvironmentLight::randomFindPixel() const
{
	double x = (double)(std::rand()) / RAND_MAX;	
	Vector2D res = Vector2D(0, 0);
	// use binary search
	int left = 0;
	int right = probabilityStack.size() - 1;

	while(left < right)
	{
		int mid = (left + right) / 2;
		std::vector<double> tmp = probabilityStack[mid]; 
		if(tmp[0] == x)     
			return Vector2D(tmp[1], tmp[2]);
		else if(tmp[0] > x)
			right = mid;
		else
			left = mid;

		if(right - left == 1)
		{
		//	printf("left,right   %d %d        %g %g       %g\n", left, right, probabilityStack[left][0], probabilityStack[right][0], x);
			return Vector2D(tmp[1], tmp[2]);
		}
			
	}

	return res;
}

Spectrum EnvironmentLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight, float* pdf) const {
  // TODO: (PathTracer) Implement
 
	// double Xi1 = (double)(std::rand()) / RAND_MAX * 2 - 1;
	// double Xi2 = (double)(std::rand()) / RAND_MAX;	

	// double theta = acos(Xi1);
	// double phi = 2.0 * PI * Xi2;

	Vector2D pixel = randomFindPixel();

	double phi = 1.0 * pixel.x / (2 * M_PI);
	double theta = 1.0 * pixel.y / M_PI;
 

	double xs = sinf(theta) * cosf(phi);
	double ys = sinf(theta) * sinf(phi);
	double zs = cosf(theta);
	



	*wi = Vector3D(xs, ys, zs);
	*distToLight = INF_D;
	*pdf = 1.f / (4.f * M_PI);

	Ray r = Ray(p, *wi);	

	return sample_dir(r);
}

Spectrum EnvironmentLight::sample_dir(const Ray& r) const {
  // TODO: (PathTracer) Implement
	size_t w = envMap->w;
	size_t h = envMap->h;

	Vector3D N = Vector3D(0, 0, 1);
	Vector3D N2 = Vector3D(0, 1, 0);

	// double theta = acos(cos_theta(r.d));
	// double phi = acos(cos_phi(r.d));
	
	// if(r.y < 0)
	// 	phi = -phi;
	// phi += M_PI;

	// double theta = acos(r.d.y);
	// double phi = acos(cos_phi(r.d));//acos(clamp(r.d.x / r.d.z, -1.0, 1.0));

	// double theta = acos(dot(r.d, N));
	// double phi = acos(dot(Vector3D(r.d.x, r.d.y, 0), N2));

	double theta = acos(r.d.y);
	double phi = acos(r.d.z/sin(theta));
	// double phi = acos(r.d.y);
	// if(r.d.x < 0)
	// 	phi = 2 * M_PI - phi;
	// 	if(r.d.y < 0)
	// 	phi = -phi;
	// phi += M_PI;

	//printf("acos: %f %f\n", theta, phi);
	double x = 1.f*phi / (2*M_PI) * w;
	double y = 1.f*theta / M_PI * h;
	int x1 = floor(x);
	int y1 = floor(y);

	if(x1 >= w - 1)
	{
		x1 = w - 2;
		x -= floor(x) - x1;
	}
		
	if(y1 >= h - 1)
	{
		y1 = h - 2;
		y -= floor(y) - y1;
	}
		
 
	int x2 = x1 + 1;
	int y2 = y1 + 1;
	//TODO!!!! interpolate
	Spectrum map1 = envMap->data[x1 + y1 * w]; 
	Spectrum map2 = envMap->data[x1 + y2 * w]; 
	Spectrum map3 = envMap->data[x2 + y1 * w]; 
	Spectrum map4 = envMap->data[x2 + y2 * w]; 

	Spectrum map12 = (y - y1) * map1 + (y2 - y) * map2;
	Spectrum map34 = (y - y1) * map3 + (y2 - y) * map4;
//	printf("%f %f %f\n", map1.r, map1.g, map1.b);
	return (x - x1) * map12 + (x2 - x) * map34;



  //return Spectrum(0, 0, 0);
}

}  // namespace StaticScene
}  // namespace CMU462
