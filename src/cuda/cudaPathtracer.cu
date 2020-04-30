
#include "CMU462/CMU462.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"
#include "CMU462/lodepng.h"

#include "../static_scene/sphere.h"
#include "../static_scene/triangle.h"
#include "../static_scene/light.h" 

#include "cudaPathtracer.h"


#include "cudaSpectrum.h"
#include "cudabsdf.h"
#include "cudaintersection.h"
#include "cudaPrimitive.h" 
#include "cudaCamera.h"
#include "cudaMatrix3x3.h"
#include "cudaRay.h"



using namespace CMU462;
using namespace StaticScene;

using std::min;
using std::max;

#define BLOCKSIZE 256
 cudaPrimitive* primitives;
double* sensorHeight; 
double* sensorWidth;
 
PathTracer* pathtracer;
cudaSpectrum* spectrum_buffer;
cudaPrimitive* cudaPrimitives;
cudaCamera* camera;  
size_t w;
size_t h;
cudaMatrix3x3 c2w;

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

void init_cudaPathTracer()
{
  cudaError_t err;
  // int prim_num = pathtracer->bvh->primitives.size(); 

  // cudaMalloc(&primitives, sizeof(cudaPrimitive) * prim_num);
   
  // cudaMemcpyToSymbol(primitives, &pathtracer->bvh->primitives,  sizeof(cudaPrimitive) * prim_num);
  
  // err = cudaPeekAtLastError();
  // if (err != cudaSuccess)
  // {
  //     fprintf(stderr, "Failed to Init primitive (error code %s)!\n", cudaGetErrorString(err));
  //     exit(EXIT_FAILURE);
  // }


  double sh = 2 * tan(radians(pathtracer->camera->vFov) / 2) * 1;	// distance is always 1
  double sw = sh * pathtracer->camera->ar;
  double test = 2.f;

  cudaMalloc((void**)&sensorHeight, sizeof(double));
  cudaMalloc((void**)&sensorWidth, sizeof(double));
  cudaMemcpy(sensorHeight, &sh, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(sensorWidth, &sw, sizeof(double), cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(sensorHeight, &test, sizeof(double));
  // cudaMemcpyToSymbol(sensorWidth, &sw, sizeof(double));
  
   err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to Init sensor Height, Width (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

      
  int num = pathtracer->frameBuffer.w * pathtracer->frameBuffer.h;
  //  spectrum_buffer = (Spectrum*)malloc(sizeof(Spectrum) * num);
  printf("BUFFER: %d\n", num);

    cudaMalloc(&spectrum_buffer, sizeof(Spectrum) * num);
    err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to Init spectrum_buffer (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    }

    cudaMalloc(&camera, sizeof(cudaCamera));
    cudaMemcpy(camera, pathtracer->camera, sizeof(cudaCamera), cudaMemcpyHostToDevice);

    err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to Init camera (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    }
}

void cudaPathTracer::set_scene(Scene *scene) {

  init_cudaPathTracer(); 



 // cudaMalloc(&cudaPrimitives, sizeof(cudaPrimitive) * prim_num);





  //pathtracer->set_scene(scene);
    // if (state != INIT) {
    // return;
    // }

    // if (this->scene != nullptr) {
    // delete scene;
    // delete bvh;
    // selectionHistory.pop();
    // }

    // if (this->envLight != nullptr) {
    // scene->lights.push_back(this->envLight);
    // }

    // this->scene = scene;
    // build_accel();

    // if (has_valid_configuration()) {
    // state = READY;
    // }
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
    //pathtracer->set_frame_size(width, height); 
    // if (state != INIT && state != READY) {
    // stop();
    // }
    // sampleBuffer.resize(width, height);
    // frameBuffer.resize(width, height);
    // if (has_valid_configuration()) {
    // state = READY;
    // }
}


void cudaPathTracer::update_screen() {
    pathtracer->update_screen(); 
    // switch (state) {
    //   case INIT:
    //   case READY:
    //     break;
    //   case VISUALIZE:
    //     visualize_accel();
    //     break;
    //   case RENDERING:
    //     glDrawPixels(frameBuffer.w, frameBuffer.h, GL_RGBA, GL_UNSIGNED_BYTE,
    //                  &frameBuffer.data[0]);
    //     break;
    //   case DONE:
    //     // sampleBuffer.tonemap(frameBuffer, tm_gamma, tm_level, tm_key, tm_wht);
    //     glDrawPixels(frameBuffer.w, frameBuffer.h, GL_RGBA, GL_UNSIGNED_BYTE,
    //                  &frameBuffer.data[0]);
    //     break;
    // }
  }
  

__device__ bool cudaintersectPrimitive(cudaPrimitive* primitives, const cudaRay &ray, cudaIntersection *isect)
{
  return false;
}

__device__ bool cudaintersectWithNode(PathTracer* pathtracer, const cudaRay &ray, cudaIntersection *isect, cudaPrimitive* primitives)
{
	BVHNode* node = pathtracer->bvh->root;
  bool hit = false;

  for (size_t p = 0; p < node->range; ++p) {
    if (cudaintersectPrimitive(primitives, ray, isect))
//	if (pathtracer->bvh->primitives[node->start + p]->intersect(ray, isect))
  {
    hit = true;
  }
}

// stack<BVHNode*> s;
// 	double lt0, lt1, rt0, rt1;

// 	// TODO!!!
// //	int threadCount = 10;
// 	int pid = 0;
// 	int M[10];

// 	BVHNode* near;
// 	BVHNode* far;
	
// 	while(true)
// 	{
// 		// when it's leaf, intersect directly

// 		if(node->isLeaf())
// 		{	

// 			for (size_t p = 0; p < node->range; ++p) {
//         	if (cudaintersectPrimitive(pathtracer->bvh->primitives[node->start + p], ray, isect))
// 			//	if (pathtracer->bvh->primitives[node->start + p]->intersect(ray, isect))
// 				{
// 					hit = true;
// 				}
// 			}
// 			if(s.empty())
// 				break;
// 			node = s.top();
// 			s.pop();	
// 		}
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
// 				}
// 				s.push(far);
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
// 				if(s.empty())
// 					break;

// 				node = s.top();
// 				s.pop();
// 			}


// 		}

// 	}

	return hit;

}





__device__ cudaSpectrum trace_ray(PathTracer* pathtracer, const cudaRay &r, cudaPrimitive* primitives) {
    cudaIntersection isect;  
   
   // if (!pathtracer->bvh->intersect(r, &isect)) {
    if (!cudaintersectWithNode(pathtracer, r, &isect, primitives)) {
      // if(pathtracer->envLight)
      // {
      //   Spectrum light_L = pathtracer->envLight->sample_dir(r);
      //   return light_L;
      // }
      // else
        return cudaSpectrum(0, 0, 0);
    }
    return cudaSpectrum(1, 1, 1);

    // Spectrum L_out = isect.bsdf->get_emission();  // Le
  
    // // TODO (PathTracer):
    // // Instead of initializing this value to a constant color, use the direct,
    // // indirect lighting components calculated in the code below. The starter
    // // code overwrites L_out by (.5,.5,.5) so that you can test your geometry
    // // queries before you implement path tracing.
  
    // //L_out = Spectrum(5.f, 5.f, 5.f);
    // //DirectionalLight dl = DirectionalLight(5, 100);
    
  
    // Vector3D hit_p = r.o + r.d * isect.t;
    // Vector3D hit_n = isect.n;
  
    // // make a coordinate system for a hit point
    // // with N aligned with the Z direction.
    // Matrix3x3 o2w;
    // make_coord_space(o2w, isect.n);
    // Matrix3x3 w2o = o2w.T();
  
    // // w_out points towards the source of the ray (e.g.,
    // // toward the camera if this is a primary ray)
    // Vector3D w_out = w2o * (r.o - hit_p);
    // w_out.normalize();
  
  
    // if (!isect.bsdf->is_delta()) {
    //   Vector3D dir_to_light;
    //   float dist_to_light;
    //   float pr;
  
    //   // ### Estimate direct lighting integral
      
    //   for (SceneLight* light : pathtracer->scene->lights) {
  
    //     // no need to take multiple samples from a point/directional source
    //     int num_light_samples = light->is_delta_light() ? 1 : pathtracer->ns_area_light;
      
    //     // integrate light over the hemisphere about the normal
    //     for (int i = 0; i < num_light_samples; i++) {
  
    //       // returns a vector 'dir_to_light' that is a direction from
    //       // point hit_p to the point on the light source.  It also returns
    //       // the distance from point x to this point on the light source.
    //       // (pr is the probability of randomly selecting the random
    //       // sample point on the light source -- more on this in part 2)
    //       const Spectrum& light_L = light->sample_L(hit_p, &dir_to_light, &dist_to_light, &pr);
  
    //       // convert direction into coordinate space of the surface, where
    //       // the surface normal is [0 0 1]
    //       const Vector3D& w_in = w2o * dir_to_light;
    //       if (w_in.z < 0) continue;
  
    //         // note that computing dot(n,w_in) is simple
    //       // in surface coordinates since the normal is (0,0,1)
    //       double cos_theta = w_in.z;
            
    //       // evaluate surface bsdf
    //       const Spectrum& f = isect.bsdf->f(w_out, w_in);
  
    //       // TODO (PathTracer):
    //       // (Task 4) Construct a shadow ray and compute whether the intersected surface is
    //       // in shadow. Only accumulate light if not in shadow.
  
    //       Vector3D o = hit_p + EPS_D * dir_to_light;
    //       float dist = dist_to_light - EPS_D;
  
    //       Ray shadow = Ray(o, dir_to_light, dist, 0);
    //       shadow.min_t = EPS_D;
  
    //       if(!pathtracer->bvh->intersect(shadow))
    //         L_out += 1.0*(cos_theta / (num_light_samples * pr)) * f * light_L;
    //     }
    //   }
    // }
  
  
    // return L_out;
  
  }

  __device__ cudaRay generate_ray_cuda(cudaCamera* camera, double x, double y, double* sensorHeight, double* sensorWidth) {
    // TODO (PathTracer):
    // compute position of the input sensor sample coordinate on the
    // canonical sensor plane one unit away from the pinhole.
    x -= 0.5;
    y -= 0.5;
  //	printf("screen:%f %f %f\n", vFov, hFov, ar);

    cudaVector3D vec = cudaVector3D(x * (*sensorWidth), y * (*sensorHeight), -1);
    return cudaRay(camera->pos, camera->c2w * vec.unit());
  }

  __global__ void raytrace_pixel(size_t w, size_t h, cudaCamera* camera, cudaSpectrum* spectrum_buffer, double* sensorHeight, double* sensorWidth) {
    // Sample the pixel with coordinate (x,y) and return the result spectrum.
    // The sample rate is given by the number of camera rays per pixel.
    

 //   spectrum_buffer[0].r = 1;

  //  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t x = index % w;
    // size_t y = index / w;

    // return;
    // double px, py;

    // px = (x + 0.5) / w;
    // py = (y + 0.5) / h;
    
    // if(x < w && y < h)
    //   spectrum_buffer[y * w + x] = trace_ray(pathtracer, generate_ray_cuda(camera, px, py), primitives);



    //  //   return trace_ray(pathtracer->camera->generate_ray(px, py));

  }
  
void cudaPathTracer::start_raytracing() {
    // pathtracer->start_raytracing();
    // return;
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
  

    const size_t w = pathtracer->sampleBuffer.w;
    const size_t h = pathtracer->sampleBuffer.h;

    // TODO: make cuda here
    size_t blockNum = (w * h + BLOCKSIZE - 1) / BLOCKSIZE;
    //printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%d\n", w*h);
    Spectrum* buffer = (Spectrum*)malloc(w*h * sizeof(Spectrum));

    for(int i = 0; i < w*h; i++)
    {
      buffer[i].r = 0.5;
      buffer[i].g = 0.5;
      buffer[i].b = 0.5;
    }
 //   memset(buffer, 0,w*h * sizeof(Spectrum));
 err = cudaPeekAtLastError();

 if (err != cudaSuccess)
 {
     fprintf(stderr, "Failed to UNKNOWN (error code %s)!\n", cudaGetErrorString(err));
     exit(EXIT_FAILURE);
 }

    // TODO: D E B U G !!!!!!
  //  printf("BLOCK INFO: %d %d\n", blockNum, BLOCKSIZE);
    raytrace_pixel<<<blockNum, BLOCKSIZE>>>( w, h,camera, spectrum_buffer, sensorHeight, sensorWidth ); 
    cudaDeviceSynchronize(); 


    // err = cudaPeekAtLastError();

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to launch Raytrace kernel (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

 //   cudaMemcpy(&buffer, &spectrum_buffer, w * h * sizeof(cudaSpectrum), cudaMemcpyDeviceToHost);

    // err = cudaPeekAtLastError();

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to launch memcpy (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    for (size_t y = 0; y < h; y ++) {
      for (size_t x = 0; x < w; x ++) {
         //   spectrum_buffer[y*w+x] = raytrace_pixel(x, y);
        //  if(buffer[y*w+x].r != 0.5)
        //  printf("%f %f %f\n",buffer[y*w+x].r, buffer[y*w+x].g, buffer[y*w+x].b );
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
