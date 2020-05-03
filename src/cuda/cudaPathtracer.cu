#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include "CMU462/CMU462.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"
#include "CMU462/lodepng.h"

#include "../static_scene/sphere.h"
#include "../static_scene/triangle.h"
#include "../static_scene/light.h" 
#include "../static_scene/scene.h"

#include "cudaPathtracer.h"


#include "cudaSpectrum.h"
#include "cudabsdf.h"
#include "cudaintersection.h"
#include "cudaPrimitive.h" 
#include "cudaTriangle.h"
#include "cudaCamera.h"
#include "cudaMatrix3x3.h"
#include "cudaRay.h"
#include "cudaLight.h"
#include "cudaBVH.h"


//using namespace std;
using namespace CMU462;
using namespace StaticScene;


using std::min;
using std::max;


#define BLOCKSIZE 256
#define MAXLIGHT 16
#define MAXSTACK 256


__constant__ int primitiveCount;
cudaTriangle* primitives;
__constant__ int lightCount;
cudaComplexLight* cudaLights;
__constant__ double sensorHeight; 
__constant__ double sensorWidth; 
__constant__ size_t width;
__constant__ size_t height;
__constant__ double INFD;

PathTracer* pathtracer;
cudaSpectrum* spectrum_buffer;
cudaPrimitive* cudaPrimitives;
cudaCamera* camera;  
cudaMatrix3x3 c2w;
cudaBVHNode* root;



cudaPathTracer::cudaPathTracer(PathTracer* _pathTracer) {
    pathtracer = _pathTracer;
 
}

cudaPathTracer::~cudaPathTracer() {
    cudaFree(spectrum_buffer);
    cudaFree(camera);
    // delete bvh;
    // delete gridSampler; 
    // delete hemisphereSampler;
}

cudaComplexLight translateLight(SceneLight* light)
{
    cudaComplexLight res = cudaComplexLight();
    res.type = light->get_type();
    // namespace cudaLightType {
    //   enum TYPE{ NONE, DIRECTIONAL, INFINITEHEMISPHERE, POINT, SPOT, AREA, SPHERE, MESH };
    // }

    DirectionalLight* dl;
    InfiniteHemisphereLight* il;
 
    switch(res.type)
    {
      case cudaLightType::DIRECTIONAL:
         dl = (DirectionalLight*)light;
        res.radiance = dl->radiance;
        res.dirToLight = dl->dirToLight;
        break;

      case cudaLightType::INFINITEHEMISPHERE:
         il = (InfiniteHemisphereLight*)light;
        res.radiance = il->radiance;
        res.sampleToWorld = il->sampleToWorld;
        res._3Dsampler = il->sampler;
        break;

    }

  //  printf("!!!!!!TYPE:%d\n", res.type);
  //  res.radiance = light->radiance;


    return res;
}

cudaBVHNode* loadBVHNode(BVHNode* node)
{
  if(!node)
    return NULL;

  cudaBVHNode cpuNode;
  cpuNode.bb = node->bb;   // todo CUDABBOX
  cpuNode.start = node->start;
  cpuNode.range = node->range;

  cpuNode.l = loadBVHNode(node->l);
  cpuNode.r = loadBVHNode(node->r);

  cudaBVHNode* newNode;
  cudaMalloc(&newNode, sizeof(cudaBVHNode));
  cudaMemcpy(newNode, &cpuNode, sizeof(cudaBVHNode), cudaMemcpyHostToDevice);
  
  return newNode;
}

void loadBVH()
{
  cudaError_t err;

  root = loadBVHNode(pathtracer->bvh->root);
 // cudaMemcpyToSymbol(root, tmpRoot,  sizeof(cudaBVHNode));

  err = cudaPeekAtLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to init load bvh (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

void loadLights()
{
  cudaError_t err;

  int lcount = pathtracer->scene->lights.size();
  cudaMemcpyToSymbol(lightCount, &lcount,  sizeof(int));

  err = cudaPeekAtLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to init light count (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaMalloc(&cudaLights, sizeof(cudaComplexLight) * lcount);
  err = cudaPeekAtLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to malloc light (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaComplexLight* cpuLight = (cudaComplexLight *)malloc(sizeof(cudaComplexLight) * lcount);
  for(int i = 0; i < lcount; i++)
  {
    cpuLight[i] = translateLight(pathtracer->scene->lights[i]);
  //  printf("LIGHT TYPE:%d\n", cpuLight[i].type);
  }

  cudaMemcpy(cudaLights, cpuLight, sizeof(cudaComplexLight) * lcount, cudaMemcpyHostToDevice);
  //  cudaMemcpyToSymbol(primitives, cpuTriangle,  sizeof(cudaTriangle)  * prim_num);
    free(cpuLight);
  
    err = cudaPeekAtLastError();
  
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to init light (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void loadPrimitives()
{
  cudaError_t err;
  int prim_num = pathtracer->bvh->primitives.size(); 
  cudaMemcpyToSymbol(primitiveCount, &prim_num,  sizeof(int));
  err = cudaPeekAtLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to init primitive count (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  cudaMalloc(&primitives, sizeof(cudaTriangle) * prim_num);
  err = cudaPeekAtLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to malloc primitive (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaTriangle* cpuTriangle = (cudaTriangle *)malloc(prim_num * sizeof(cudaTriangle));
  // TODO: NEED TO REALLY TRANSLATE IT
  for(int i = 0; i < prim_num; i++)
  {
    Triangle* prim = (Triangle*)pathtracer->bvh->primitives[i];
    cpuTriangle[i].mesh = prim->mesh;
    cpuTriangle[i].v1 = prim->v1;
    cpuTriangle[i].v2 = prim->v2;
    cpuTriangle[i].v3 = prim->v3;
    cpuTriangle[i].v = prim->v;

    
    cpuTriangle[i].p0 = prim->mesh->positions[prim->v1]; 
    cpuTriangle[i].p1 = prim->mesh->positions[prim->v2];
    cpuTriangle[i].p2 = prim->mesh->positions[prim->v3];

    cpuTriangle[i].n0 = prim->mesh->normals[prim->v1]; 
    cpuTriangle[i].n1 = prim->mesh->normals[prim->v2]; 
    cpuTriangle[i].n2 = prim->mesh->normals[prim->v3]; 

    DiffuseBSDF* bsdf = (DiffuseBSDF*)prim->mesh->bsdf;
    cpuTriangle[i].bsdf.albedo = bsdf->albedo;
    cpuTriangle[i].bsdf.sampler = bsdf->sampler;

  //  printf("%f \n",cpuTriangle[i].mesh->positions[cpuTriangle[i].v1].x);
  }
  cudaMemcpy(primitives, cpuTriangle, sizeof(cudaTriangle)  * prim_num, cudaMemcpyHostToDevice);
//  cudaMemcpyToSymbol(primitives, cpuTriangle,  sizeof(cudaTriangle)  * prim_num);
  free(cpuTriangle);

  err = cudaPeekAtLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to init primitive (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

void cudaPathTracer::set_scene(Scene *scene) {
  cudaError_t err;
  double sh = 2 * tan(radians(pathtracer->camera->vFov) / 2) * 1;	// distance is always 1
  double sw = sh * pathtracer->camera->ar;


  cudaMemcpyToSymbol(sensorHeight, &sh,  sizeof(double));
  cudaMemcpyToSymbol(sensorWidth, &sw,  sizeof(double));


  int num = pathtracer->sampleBuffer.w * pathtracer->sampleBuffer.h;;
  //  spectrum_buffer = (Spectrum*)malloc(sizeof(Spectrum) * num);

    cudaMalloc(&spectrum_buffer, sizeof(cudaSpectrum) * num);


    cudaMalloc(&camera, sizeof(cudaCamera));
    cudaMemcpy(camera, pathtracer->camera, sizeof(cudaCamera), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(width, &pathtracer->sampleBuffer.w,  sizeof(double));
    cudaMemcpyToSymbol(height, &pathtracer->sampleBuffer.h,  sizeof(double));
    
    err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to init scene (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    loadPrimitives();
    loadLights();
    loadBVH();

    double infine =  std::numeric_limits<double>::infinity();
    cudaMemcpyToSymbol(INFD, &infine,  sizeof(double));
 // cudaMalloc(&cudaPrimitives, sizeof(cudaPrimitive) * prim_num);
}

void cudaPathTracer::set_camera(Camera *camera) {
  //  pathtracer->set_camera(camera);
    // if (state != INIT) {
    // return;
    // }

    // this->camera = camera;
    // if (has_valid_configuration()) {
    // state = READY;
    // }
}

void cudaPathTracer::set_frame_size(size_t width, size_t height) {
    pathtracer->set_frame_size(width, height); 

}


void cudaPathTracer::update_screen() {
    pathtracer->update_screen(); 

  }

__device__ cudaVector3D getSample3D(curandState *state, size_t index)
{
  //TODO!!!!!!!!!!!!!!!!TRY FIND SOME SUBSTITUTE

  
  curandState local_state = state[index];
  double Xi1 = curand_uniform(&local_state);
  double Xi2 = curand_uniform(&local_state);
  state[index] = local_state;
  double theta = acos(Xi1);
  double phi = 2.0 * PI * Xi2;

  double xs = sinf(theta) * cosf(phi);
  double ys = sinf(theta) * sinf(phi);
  double zs = cosf(theta);

  return cudaVector3D(xs, ys, zs);
//  printf("index:%d %g %g\n", index, Xi1, Xi2);
//  return cudaVector3D(0.5, 0.5, 0.5);


}
__device__ cudaSpectrum sampleLight(cudaComplexLight* light, const cudaVector3D& p, cudaVector3D* wi, float* distToLight,
  float* pdf, curandState *state, size_t index)
{
  cudaSpectrum res;
  cudaVector3D dir;
  switch(light->type)
  {
    case cudaLightType::DIRECTIONAL:
      *wi = light->dirToLight;
      *distToLight = INFD;
      *pdf = 1.0;
      res = light->radiance;
    break;

    case cudaLightType::INFINITEHEMISPHERE:
      dir = getSample3D(state, index);
      *wi = light->sampleToWorld * dir;
      *distToLight = INFD;
      *pdf = 1.0 / (2.0 * M_PI);
      res = light->radiance;
    break;

    default:
    res = cudaSpectrum();
    break;
  }
  return res;
}


__device__ bool cudaintersectPrimitive(cudaTriangle* primitive, const cudaRay &r, cudaIntersection *isect)
{

    cudaVector3D p0 = primitive->p0; 
    cudaVector3D p1 = primitive->p1;
    cudaVector3D p2 = primitive->p2;
  
  
    cudaVector3D o = r.o;
    cudaVector3D d = r.d;
  
    cudaVector3D e1 = p1 - p0;
    cudaVector3D e2 = p2 - p0;
    cudaVector3D s = o - p0;
  
    double denominator = dot(cross(e1, d), e2);
    if (denominator == 0)
      return false;
  
    cudaVector3D numerator = cudaVector3D(-dot(cross(s, e2), d), dot(cross(e1, d), s), -dot(cross(s, e2), e1));
    cudaVector3D ans = numerator / denominator;
  //	return true;
    // in triangle
    if (ans.x < 0 || ans.x > 1 || ans.y < 0 || ans.y > 1 ||
      1 - ans.x - ans.y < 0 || 1 - ans.x - ans.y > 1 ||
      ans.z < r.min_t || ans.z > r.max_t)
      return false;
  
    double u = ans.x;
    double v = ans.y;
    double t = ans.z;
   
  //  cudaVector3D tt = cudaVector3D(t.x, t.y, t.z); 
    r.max_t = t;
  
    isect->t = t;

    cudaVector3D n0 = primitive->n0;
    cudaVector3D n1 = primitive->n1;
    cudaVector3D n2 = primitive->n2;
 
    cudaVector3D tmp = (1 - u - v) * n0 + u * n1 + v * n2;
     isect->n = cudaVector3D(tmp.x, tmp.y, tmp.z);
     if (dot(isect->n, r.d) > 0)
      isect->n *= -1;
    isect->primitive = primitive;
    isect->bsdf = &primitive->bsdf;	 
     
  return true; 
}

__device__ bool cudaintersectWithNode(const cudaRay &ray, cudaIntersection *isect, cudaTriangle* primitives, cudaBVHNode* root)
{
//	BVHNode* node = pathtracer->bvh->root;
  bool hit = false;

//  for (size_t p = 0; p < node->range; ++p) {
  for (size_t p = 0; p < primitiveCount; ++p) {
    if (cudaintersectPrimitive(&primitives[p], ray, isect))
//	if (pathtracer->bvh->primitives[node->start + p]->intersect(ray, isect))
  {
    hit = true;
  }
}
//   cudaBVHNode* s[MAXSTACK];
//   int stackSize = 0;

//   // TODO: FIRST NODE!
//   //stack<BVHNode*> s;
// 	double lt0, lt1, rt0, rt1;

// 	// TODO!!!
// //	int threadCount = 10;
// 	int pid = 0;
// 	int M[10];

// 	cudaBVHNode* near;
// 	cudaBVHNode* far;
  
//   cudaBVHNode* node = root;
// 	while(true)
// 	{
// 		// when it's leaf, intersect directly

// 		if(node->isLeaf())
// 		{	

// 			for (size_t p = 0; p < node->range; ++p) {
//         if (cudaintersectPrimitive(&primitives[node->start + p], ray, isect))
// 			//	if (pathtracer->bvh->primitives[node->start + p]->intersect(ray, isect))
// 				{
// 					hit = true;
// 				}
// 			}
// 			if(stackSize == 0)
// 				break;
// 			node = s[--stackSize];
//     }
    
// 		else
// 		{
// 			/* Parallel read ?*/
// 			int hitleft = (bool)node->l->bb.intersect(ray, lt0, lt1);
// 			int hitright = (bool)node->r->bb.intersect(ray, rt0, rt1);

// 			/* Use parallel and barrier to init */
// 			for(int i = 0; i <= 3; i++)
// 				M[i] = 0;

// 			// TODO: barrier here
// 			M[2*hitleft + hitright] = 1;
// 			// TODO: barrier here

// 			/* Visit both children */
// 			if(M[3] || (M[1] && M[2]))
// 			{
// 		//		printf("HERE!!\n");
// 				/* Decide which to go in first */
// 				M[pid] = 2 * (hitright && (rt0 < lt0)) - 1;

// 				/* TODO: PARLLEL SUM OVER HERE */
// 				if(M[pid] < 0)
// 				{
// 					near = node->l;
// 					far = node->r;
// 				}
// 				else
// 				{
// 					near = node->r;
// 					far = node->l;
//         }
//         s[stackSize++] = far;
// 				node = near;

// 			}
// 			else if(M[2])
// 			{
// 			//	printf("HERELEFT\n");
// 				node = node->l;
// 			}

// 			else if(M[1])
// 			{
// 			//	printf("HERERIGHT\n");
// 				node = node->r;
// 			}

// 			else
// 			{
// 				if(stackSize == 0)
// 					break;

//         node = s[--stackSize];
// 			}


// 		} // end else

// 	}

	return hit; 

}





__device__ cudaSpectrum trace_ray( const cudaRay &r, cudaTriangle* primitives, cudaComplexLight* lights, 
  curandState *state, size_t index, cudaBVHNode* root) 
  {
    cudaIntersection isect;  
   
   // if (!pathtracer->bvh->intersect(r, &isect)) {
    if (!cudaintersectWithNode(r, &isect, primitives, root)) {

        return cudaSpectrum(0, 0, 0);
    }

 //     return cudaSpectrum(1, 1, 1);
   // cudaSpectrum L_out =cudaSpectrum(1, 1, 1);
  //  cudaSpectrum L_out = (DiffuseBSDF)isect.bsdf->get_emission();  // Le
  cudaSpectrum L_out = cudaSpectrum();

    // TODO (PathTracer):
    // Instead of initializing this value to a constant color, use the direct,
    // indirect lighting components calculated in the code below. The starter
    // code overwrites L_out by (.5,.5,.5) so that you can test your geometry
    // queries before you implement path tracing.
  
    //L_out = Spectrum(5.f, 5.f, 5.f);
    //DirectionalLight dl = DirectionalLight(5, 100);
    
  
    cudaVector3D hit_p = r.o + r.d * isect.t;
    cudaVector3D hit_n = isect.n;
  
    // make a coordinate system for a hit point
    // with N aligned with the Z direction.
    cudaMatrix3x3 o2w;
    make_coord_space(o2w, isect.n);
    cudaMatrix3x3 w2o = o2w.T();
  
    // w_out points towards the source of the ray (e.g.,
    // toward the camera if this is a primary ray)
    cudaVector3D w_out = w2o * (r.o - hit_p);
    w_out.normalize();
  
  
  //  if (!isect.bsdf->is_delta()) {
      cudaVector3D dir_to_light;
      float dist_to_light;
      float pr;
  
      // ### Estimate direct lighting integral
      
    //  for (SceneLight* light : pathtracer->scene->lights) {
      for (int i = 0; i < lightCount; i++) {
        // no need to take multiple samples from a point/directional source
      //  int num_light_samples = light->is_delta_light() ? 1 : pathtracer->ns_area_light;
        cudaComplexLight* light = &lights[i];
        int num_light_samples = 1;
        // integrate light over the hemisphere about the normal
        for (int i = 0; i < num_light_samples; i++) {
  
          // returns a vector 'dir_to_light' that is a direction from
          // point hit_p to the point on the light source.  It also returns
          // the distance from point x to this point on the light source.
          // (pr is the probability of randomly selecting the random
          // sample point on the light source -- more on this in part 2)


          //  const cudaSpectrum& light_L = light->sample_L(hit_p, &dir_to_light, &dist_to_light, &pr);
          const cudaSpectrum& light_L = sampleLight(light, hit_p, &dir_to_light, &dist_to_light, &pr, state, index);

          // convert direction into coordinate space of the surface, where
          // the surface normal is [0 0 1]
          const cudaVector3D& w_in = w2o * dir_to_light;
          if (w_in.z < 0) continue;
  
            // note that computing dot(n,w_in) is simple
          // in surface coordinates since the normal is (0,0,1)
          double cos_theta = w_in.z;
            
          // evaluate surface bsdf
         // const cudaSpectrum& f = ((cudaDiffuseBSDF*)isect.bsdf)->f(w_out, w_in);
            const cudaSpectrum& f =  (isect.bsdf)->albedo * (1.0 / PI);
          // TODO (PathTracer):
          // (Task 4) Construct a shadow ray and compute whether the intersected surface is
          // in shadow. Only accumulate light if not in shadow.
  
          cudaVector3D o = hit_p + EPS_D * dir_to_light;
          float dist = dist_to_light - EPS_D;
  
          cudaRay shadow = cudaRay(o, dir_to_light, dist, 0);
          shadow.min_t = EPS_D;
  
         // if(!pathtracer->bvh->intersect(shadow))
         if(!cudaintersectWithNode(shadow, &isect, primitives, root)) 
           L_out += 1.0*(cos_theta / (num_light_samples * pr)) * f * light_L;
        }
      }
  //   }
  
  
    return L_out;
  
  }

  __device__ cudaRay generate_ray_cuda(cudaCamera* camera, double x, double y) {
    // TODO (PathTracer):
    // compute position of the input sensor sample coordinate on the
    // canonical sensor plane one unit away from the pinhole.
    x -= 0.5;
    y -= 0.5;
  //	printf("screen:%f %f %f\n", vFov, hFov, ar);

    cudaVector3D vec = cudaVector3D(x * sensorWidth, y * sensorHeight, -1);
    return cudaRay(camera->pos, camera->c2w * vec.unit());
  }

  __global__ void raytrace_pixel(cudaCamera* camera, cudaSpectrum* spectrum_buffer, cudaTriangle* primitives, 
    cudaComplexLight* lights, curandState *state, cudaBVHNode* root) 
  {
    // Sample the pixel with coordinate (x,y) and return the result spectrum.
    // The sample rate is given by the number of camera rays per pixel.

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x = index % width;
    size_t y = index / width;
//    printf("index:%d ", index);
    double px, py;

    px = (x + 0.5) / width;
    py = (y + 0.5) / height;
    
//    double color = (double)index / (width*height); 
//     printf("color%g\n",color);
    // if(x < width && y < height)
    // {
    //   spectrum_buffer[y*width+x].r = color;
    //   spectrum_buffer[y*width+x].g = color;
    //   spectrum_buffer[y*width+x].b = color;
    // }
   if(x < width && y < height)
      spectrum_buffer[y * width + x] = trace_ray(generate_ray_cuda(camera, px, py), primitives, lights, state, index, root);
    //   return trace_ray(pathtracer->camera->generate_ray(px, py));

  }

__global__ void setup_random_kernel(curandState *state,unsigned long seed)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  curand_init(seed, index, 0, &state[index]);
}
  
void cudaPathTracer::start_raytracing() {
 //   pathtracer->start_raytracing();
 cudaError_t err;
    pathtracer->rayLog.clear();
    pathtracer->workQueue.clear();
  
    pathtracer->state = pathtracer->RENDERING;
    pathtracer->continueRaytracing = true;
    pathtracer->workerDoneCount = 0;
   
    pathtracer->sampleBuffer.clear();
    pathtracer->frameBuffer.clear();
    pathtracer->num_tiles_w = pathtracer->sampleBuffer.w / pathtracer->imageTileSize + 1;
    pathtracer->num_tiles_h = pathtracer->sampleBuffer.h / pathtracer->imageTileSize + 1;
    pathtracer->tile_samples.resize(pathtracer->num_tiles_w * pathtracer->num_tiles_h);
    memset(&pathtracer->tile_samples[0], 0, pathtracer->num_tiles_w * pathtracer->num_tiles_h * sizeof(int));
  
    // launch threads
    fprintf(stdout, "[CudaPathTracer] Rendering... ");
    fflush(stdout);
        
    Timer timer;
    timer.start();
    

    size_t w = pathtracer->sampleBuffer.w;
    size_t h = pathtracer->sampleBuffer.h;
    // TODO: make cuda here
    size_t blockNum = (w * h + BLOCKSIZE - 1) / BLOCKSIZE;
    //printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%d\n", w*h);
    Spectrum* buffer = (Spectrum*)malloc(w*h * sizeof(cudaSpectrum));

    memset(buffer, 0,w*h * sizeof(cudaSpectrum));

    // TODO : D E B U G !!!!!!

    err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  UNKOWN (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    /* Random Number Generator */
    curandState *state;
    cudaMalloc((void **)&state, w*h*sizeof(curandState));
    setup_random_kernel<<<blockNum,BLOCKSIZE>>>(state, unsigned(time(NULL)));
    raytrace_pixel<<<blockNum, BLOCKSIZE>>>(camera, spectrum_buffer, primitives, cudaLights, state, root); 
    cudaDeviceSynchronize();

    err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel raytrace_pixel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else
      printf("launch success\n");

  //    err = cudaMemcpy(spectrum_buffer, buffer, w*h* sizeof(cudaSpectrum), cudaMemcpyHostToDevice);
    err = cudaMemcpy(buffer, spectrum_buffer, w*h* sizeof(cudaSpectrum), cudaMemcpyDeviceToHost);
    

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy spectrum (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (size_t y = 0; y < h; y ++) {
      for (size_t x = 0; x < w; x ++) {
         //   spectrum_buffer[y*w+x] = raytrace_pixel(x, y);
             pathtracer->sampleBuffer.update_pixel(buffer[y*w+x], x, y);  
            // Spectrum s = raytrace_pixel(x, y);
            // pathtracer->sampleBuffer.update_pixel(s, x, y);     
       }  
    }

    pathtracer->sampleBuffer.toColor(pathtracer->frameBuffer, 0, 0, pathtracer->sampleBuffer.w, pathtracer->sampleBuffer.h);
    timer.stop();
    fprintf(stdout, "Done! (%.4fs)\n", timer.duration());
    pathtracer->state = pathtracer->DONE;

    free(buffer);

  }

  void cudaPathTracer::save_image(string fname) {
      pathtracer->save_image(fname);
    // if (state != DONE) return;
  
    // uint32_t *frame = &frameBuffer.data[0];
    // size_t w = frameBuffer.w;
    // size_t h = frameBuffer.h;
    // uint32_t *frame_out = new uint32_t[w * h];
    // for (size_t i = 0; i < h; ++i) {
    //   memcpy(frame_out + i * w, frame + (h - i - 1) * w, 4 * w);
    // }
  
    // fprintf(stderr, "[PathTracer] Saving to file: %s... ", fname.c_str());
    // lodepng::encode(fname, (unsigned char *)frame_out, w, h);
    // fprintf(stderr, "Done!\n");
  }

  bool cudaPathTracer::is_done() {
    pathtracer->update_screen();
    return pathtracer->state == pathtracer->DONE;
}
 
  bool cudaPathTracer::is_done_headless() {
        return pathtracer->state == pathtracer->DONE;

  }
