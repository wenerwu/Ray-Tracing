#include "pathtracer.h"
#include "bsdf.h"
#include "ray.h"

#include <stack>
#include <random>
#include <algorithm>

#include "CMU462/CMU462.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"
#include "CMU462/lodepng.h"

#include "GL/glew.h"

#include "static_scene/sphere.h"
#include "static_scene/triangle.h"
#include "static_scene/light.h"

using namespace CMU462::StaticScene;

using std::min;
using std::max;

namespace CMU462 {

//#define ENABLE_RAY_LOGGING 1

#if MPI
PathTracer::PathTracer(size_t ns_aa, size_t max_ray_depth, size_t ns_area_light,
                       size_t ns_diff, size_t ns_glsy, size_t ns_refr,
                       size_t num_threads, HDRImageBuffer *envmap,
                       size_t mpi_pcount, size_t mpi_id) {
  this->mpi_pcount = mpi_pcount;
  this->mpi_id = mpi_id;
#else
PathTracer::PathTracer(size_t ns_aa, size_t max_ray_depth, size_t ns_area_light,
                       size_t ns_diff, size_t ns_glsy, size_t ns_refr,
                       size_t num_threads, HDRImageBuffer *envmap) {
#endif
  state = INIT, this->ns_aa = ns_aa;
  this->max_ray_depth = max_ray_depth;
  this->ns_area_light = ns_area_light;
  this->ns_diff = ns_diff;
  this->ns_glsy = ns_diff;
  this->ns_refr = ns_refr;


  if (envmap) {
    this->envLight = new EnvironmentLight(envmap);
  } else {
    this->envLight = NULL;
  }

  bvh = NULL;
  scene = NULL;
  camera = NULL;

  gridSampler = new UniformGridSampler2D();
  hemisphereSampler = new UniformHemisphereSampler3D();

  show_rays = true;

  imageTileSize = 32;
  numWorkerThreads = num_threads;
  workerThreads.resize(numWorkerThreads);

  tm_gamma = 2.2f;
  tm_level = 1.0f;
  tm_key = 0.18;
  tm_wht = 5.0f;
}

PathTracer::~PathTracer() {
  delete bvh;
  delete gridSampler;
  delete hemisphereSampler;
}

void PathTracer::set_scene(Scene *scene) {
  if (state != INIT) {
    return;
  }

  if (this->scene != nullptr) {
    delete scene;
    delete bvh;
    selectionHistory.pop();
  }

  if (this->envLight != nullptr) {
    scene->lights.push_back(this->envLight);
  }

  this->scene = scene;
  build_accel();

  if (has_valid_configuration()) {
    state = READY;
  }
}

void PathTracer::set_camera(Camera *camera) {
  if (state != INIT) {
    return;
  }

  this->camera = camera;
  if (has_valid_configuration()) {
    state = READY;
  }
}

void PathTracer::set_frame_size(size_t width, size_t height) {
  if (state != INIT && state != READY) {
    stop();
  }
  sampleBuffer.resize(width, height);
  frameBuffer.resize(width, height);
  if (has_valid_configuration()) {
    state = READY;
  }
}

bool PathTracer::has_valid_configuration() {
  return scene && camera && gridSampler && hemisphereSampler &&
         (!sampleBuffer.is_empty());
}

void PathTracer::update_screen() {
  switch (state) {
    case INIT:
    case READY:
      break;
    case VISUALIZE:
      visualize_accel();
      break;
    case RENDERING:
      glDrawPixels(frameBuffer.w, frameBuffer.h, GL_RGBA, GL_UNSIGNED_BYTE,
                   &frameBuffer.data[0]);
      break;
    case DONE:
      // sampleBuffer.tonemap(frameBuffer, tm_gamma, tm_level, tm_key, tm_wht);
      glDrawPixels(frameBuffer.w, frameBuffer.h, GL_RGBA, GL_UNSIGNED_BYTE,
                   &frameBuffer.data[0]);
      break;
  }
}

void PathTracer::stop() {
  switch (state) {
    case INIT:
    case READY:
      break;
    case VISUALIZE:
      while (selectionHistory.size() > 1) {
        selectionHistory.pop();
      }
      state = READY;
      break;
    case RENDERING:
      continueRaytracing = false;
    case DONE:
#if !MPI
      for (int i = 0; i < numWorkerThreads; i++) {
        workerThreads[i]->join();
        delete workerThreads[i];
      }
#endif
      state = READY;
      break;
  }
}

void PathTracer::clear() {
  if (state != READY) return;
  delete bvh;
  bvh = NULL;
  scene = NULL;
  camera = NULL;
  selectionHistory.pop();
  sampleBuffer.resize(0, 0);
  frameBuffer.resize(0, 0);
  state = INIT;
}

void PathTracer::start_visualizing() {
  if (state != READY) {
    return;
  }
  state = VISUALIZE;
}

void PathTracer::start_raytracing() {
  if (state != READY) return;

  rayLog.clear();
  workQueue.clear();

  state = RENDERING;
  continueRaytracing = true;
  workerDoneCount = 0;

  sampleBuffer.clear();
  frameBuffer.clear();
  num_tiles_w = sampleBuffer.w / imageTileSize + 1;
  num_tiles_h = sampleBuffer.h / imageTileSize + 1;
  tile_samples.resize(num_tiles_w * num_tiles_h);
  memset(&tile_samples[0], 0, num_tiles_w * num_tiles_h * sizeof(int));

  // populate the tile work queue
  for (size_t y = 0; y < sampleBuffer.h; y += imageTileSize) {
    for (size_t x = 0; x < sampleBuffer.w; x += imageTileSize) {
      workQueue.put_work(WorkItem(x, y, imageTileSize, imageTileSize));
    }
  }

  // launch threads
#if MPI
  if (mpi_id == 0) {
  fprintf(stdout, "[PathTracer] Rendering... ");
  fflush(stdout);
  }
#else
  fprintf(stdout, "[PathTracer] Rendering... ");
  fflush(stdout);
#endif
#if MPI
  worker_thread();
  collect_frame_buffer();
#else
  for (int i = 0; i < numWorkerThreads; i++) {
    workerThreads[i] = new std::thread(&PathTracer::worker_thread, this);
  }
#endif
}

void PathTracer::build_accel() {
  // collect primitives //
#if MPI
  if (mpi_id == 0) {
#endif
  fprintf(stdout, "[PathTracer] Collecting primitives... ");
  fflush(stdout);
#if MPI
  }
#endif
  timer.start();
  vector<Primitive *> primitives;
  for (SceneObject *obj : scene->objects) {
    const vector<Primitive *> &obj_prims = obj->get_primitives();
	//printf("MYTEST:%g\n", obj_prims[0]->get_bbox().max.x);
    primitives.reserve(primitives.size() + obj_prims.size());
    primitives.insert(primitives.end(), obj_prims.begin(), obj_prims.end());
  }
  timer.stop();
#if MPI
  if (mpi_id == 0)
#endif
  fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());
  
  // build BVH //
#if MPI
  if (mpi_id == 0) {
#endif
  fprintf(stdout, "[PathTracer] Building BVH... ");
  fflush(stdout);
#if MPI
  }
#endif
  timer.start();
  bvh = new BVHAccel(primitives);
  timer.stop();
#if MPI
  if (mpi_id == 0)
#endif
  fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());

  // initial visualization //
  selectionHistory.push(bvh->get_root());
}

void PathTracer::log_ray_miss(const Ray &r) {
  rayLog.push_back(LoggedRay(r, -1.0));
}

void PathTracer::log_ray_hit(const Ray &r, double hit_t) {
  rayLog.push_back(LoggedRay(r, hit_t));
}

void PathTracer::visualize_accel() const {
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_LIGHTING);
  glBlendFunc(GL_ONE, GL_ZERO);
  glLineWidth(.001);
  glEnable(GL_DEPTH_TEST);

  // hardcoded color settings
  Color cnode = Color(.5, .5, .5, .25);
  Color cnode_hl = Color(1., .25, .0, .6);
  Color cnode_hl_child = Color(1., 1., 1., .6);

  Color cprim_hl_left = Color(.6, .6, 1., 1);
  Color cprim_hl_right = Color(.8, .8, 1., 1);
  Color cprim_hl_edges = Color(0., 0., 0., 0.5);

  BVHNode *selected = selectionHistory.top();

  // render solid geometry (with depth offset)
  glPolygonOffset(1.0, 1.0);
  glEnable(GL_POLYGON_OFFSET_FILL);

  if (selected->isLeaf()) {
    for (size_t i = 0; i < selected->range; ++i) {
      bvh->primitives[selected->start + i]->draw(cprim_hl_left);
    }
  } else {
    if (selected->l) {
      BVHNode *child = selected->l;
      for (size_t i = 0; i < child->range; ++i) {
        bvh->primitives[child->start + i]->draw(cprim_hl_left);
      }
    }
    if (selected->r) {
      BVHNode *child = selected->r;
      for (size_t i = 0; i < child->range; ++i) {
        bvh->primitives[child->start + i]->draw(cprim_hl_right);
      }
    }
  }

  glDisable(GL_POLYGON_OFFSET_FILL);

  // draw geometry outline
  for (size_t i = 0; i < selected->range; ++i) {
    bvh->primitives[selected->start + i]->drawOutline(cprim_hl_edges);
  }

  // keep depth buffer check enabled so that mesh occluded bboxes, but
  // disable depth write so that bboxes don't occlude each other.
  glDepthMask(GL_FALSE);

  // create traversal stack
  stack<BVHNode *> tstack;

  // push initial traversal data
  tstack.push(bvh->get_root());

  // draw all BVH bboxes with non-highlighted color
  while (!tstack.empty()) {
    BVHNode *current = tstack.top();
    tstack.pop();

    current->bb.draw(cnode);
    if (current->l) tstack.push(current->l);
    if (current->r) tstack.push(current->r);
  }

  // draw selected node bbox and primitives
  if (selected->l) selected->l->bb.draw(cnode_hl_child);
  if (selected->r) selected->r->bb.draw(cnode_hl_child);

  glLineWidth(3.f);
  selected->bb.draw(cnode_hl);

  // now perform visualization of the rays
  if (show_rays) {
    glLineWidth(1.f);
    glBegin(GL_LINES);

    for (size_t i = 0; i < rayLog.size(); i += 500) {
      const static double VERY_LONG = 10e4;
      double ray_t = VERY_LONG;

      // color rays that are hits yellow
      // and rays this miss all geometry red
      if (rayLog[i].hit_t >= 0.0) {
        ray_t = rayLog[i].hit_t;
        glColor4f(1.f, 1.f, 0.f, 0.1f);
      } else {
        glColor4f(1.f, 0.f, 0.f, 0.1f);
      }

      Vector3D end = rayLog[i].o + ray_t * rayLog[i].d;

      glVertex3f(rayLog[i].o[0], rayLog[i].o[1], rayLog[i].o[2]);
      glVertex3f(end[0], end[1], end[2]);
    }
    glEnd();
  }

  glDepthMask(GL_TRUE);
  glPopAttrib();
}

void PathTracer::key_press(int key) {
  BVHNode *current = selectionHistory.top();
  switch (key) {
    case ']':
      ns_aa *= 2;
      printf("Samples per pixel changed to %lu\n", ns_aa);
      // tm_key = clamp(tm_key + 0.02f, 0.0f, 1.0f);
      break;
    case '[':
      // tm_key = clamp(tm_key - 0.02f, 0.0f, 1.0f);
      ns_aa /= 2;
      if (ns_aa < 1) ns_aa = 1;
      printf("Samples per pixel changed to %lu\n", ns_aa);
      break;
    case KEYBOARD_UP:
    case '?':
      if (current != bvh->get_root()) {
        selectionHistory.pop();
      }
      break;
    case KEYBOARD_LEFT:
    case '<':
      if (current->l) {
        selectionHistory.push(current->l);
      }
      break;
    case KEYBOARD_RIGHT:
    case '>':
      if (current->l) {
        selectionHistory.push(current->r);
      }
      break;
    case 's':
    case 'S':
      show_rays = !show_rays;
    default:
      return;
  }
}


Spectrum PathTracer::trace_ray(const Ray &r) {
  Intersection isect;

  if (!bvh->intersect(r, &isect)) {
// log ray miss
#ifdef ENABLE_RAY_LOGGING
    log_ray_miss(r);
#endif

    // TODO (PathTracer):
    // (Task 7) If you have an environment map, return the Spectrum this ray
    // samples from the environment map. If you don't return black.

    if(this->envLight)
    {
      Spectrum light_L = this->envLight->sample_dir(r);
      return light_L;
    }
    else
      return Spectrum(0, 0, 0);
  }

// log ray hit
#ifdef ENABLE_RAY_LOGGING
  log_ray_hit(r, isect.t);
#endif

  Spectrum L_out = isect.bsdf->get_emission();  // Le

  // TODO (PathTracer):
  // Instead of initializing this value to a constant color, use the direct,
  // indirect lighting components calculated in the code below. The starter
  // code overwrites L_out by (.5,.5,.5) so that you can test your geometry
  // queries before you implement path tracing.

  //L_out = Spectrum(.5f, .5f, .5f);
  //DirectionalLight dl = DirectionalLight(5, 100);
  

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D hit_n = isect.n;

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  Vector3D w_out = w2o * (r.o - hit_p);
  w_out.normalize();


  if (!isect.bsdf->is_delta()) {
    Vector3D dir_to_light;
    float dist_to_light;
    float pr; 

    // ### Estimate direct lighting integral
    for (SceneLight* light : scene->lights) {

      // no need to take multiple samples from a point/directional source
      int num_light_samples = light->is_delta_light() ? 1 : ns_area_light;
 
      // integrate light over the hemisphere about the normal
      for (int i = 0; i < num_light_samples; i++) {

        // returns a vector 'dir_to_light' that is a direction from
        // point hit_p to the point on the light source.  It also returns
        // the distance from point x to this point on the light source.
        // (pr is the probability of randomly selecting the random
        // sample point on the light source -- more on this in part 2)
        
        const Spectrum& light_L = light->sample_L(hit_p, &dir_to_light, &dist_to_light, &pr);

        // convert direction into coordinate space of the surface, where
        // the surface normal is [0 0 1]
        const Vector3D& w_in = w2o * dir_to_light;

        if (w_in.z < 0) continue;

          // note that computing dot(n,w_in) is simple
        // in surface coordinates since the normal is (0,0,1)
        double cos_theta = w_in.z;
          
        // evaluate surface bsdf
        const Spectrum& f = isect.bsdf->f(w_out, w_in);

        // TODO (PathTracer):
        // (Task 4) Construct a shadow ray and compute whether the intersected surface is
        // in shadow. Only accumulate light if not in shadow.

        Vector3D o = hit_p + EPS_D * dir_to_light;
        float dist = dist_to_light - EPS_D;

        Ray shadow = Ray(o, dir_to_light, dist, 0);
        shadow.min_t = EPS_D;

        if(!bvh->intersect(shadow))
        {
          Spectrum tmp = 1.0*(cos_theta / (num_light_samples * pr)) * f * light_L;
 
          L_out += 1.0*(cos_theta / (num_light_samples * pr)) * f * light_L;
        }
          
      }
    }
  }

  // TODO (PathTracer):
  // ### (Task 5) Compute an indirect lighting estimate using pathtracing with Monte Carlo.


  // Note that Ray objects have a depth field now; you should use this to avoid
  // traveling down one path forever.
  
  // (1) randomly select a new ray direction (it may be
  // reflection or transmittence ray depending on
  // surface type -- see BSDF::sample_f()

  // (2) potentially terminate path (using Russian roulette)

  // (3) evaluate weighted reflectance contribution due 
  // to light from this direction

  if (r.depth > max_ray_depth)
	  return L_out;
  //printf("depth:%d %d\n", r.depth, max_ray_depth);

  // new ray direction and light
  Vector3D wi;
  float pdf;
  Spectrum f = isect.bsdf->sample_f(w_out, &wi, &pdf);

  // terminate
  float terminateProbability = f.illum();
  terminateProbability = terminateProbability > 1 ? 1 : terminateProbability;	
  terminateProbability = 1.f - terminateProbability;

  float randomFloat = ((float)std::rand() / RAND_MAX);
  if (randomFloat < terminateProbability)
	  return L_out;

  wi = o2w * wi;
  wi.normalize();
  Ray newRay = Ray(hit_p + EPS_D * wi, wi, (int)r.depth + 1);
  
  Spectrum newSpectrum = f * trace_ray(newRay) * fabs((float)dot(wi, isect.n) / (pdf * (1.f - terminateProbability)));


  L_out += newSpectrum;
  return L_out;
}

Spectrum PathTracer::raytrace_pixel(size_t x, size_t y) {
  // TODO (PathTracer):
  // Sample the pixel with coordinate (x,y) and return the result spectrum.
  // The sample rate is given by the number of camera rays per pixel.
//	fprintf(stdout, "HERE!\n");
	
  int num_samples = ns_aa;
  Spectrum res = Spectrum(0,0,0);
  double px, py;
  for (int i = 0; i < ns_aa; i++)
  {
	  if (num_samples == 1)
	  {
		  px = (x + 0.5) / sampleBuffer.w;
		  py = (y + 0.5) / sampleBuffer.h;
	  }
	  else
	  {
		  Vector2D tmp = gridSampler->get_sample();
		  px = (x + tmp.x) / sampleBuffer.w;
		  py = (y + tmp.y) / sampleBuffer.h;
	  }

	  res += trace_ray(camera->generate_ray(px, py));
//	  printf("%f %f %f\n", res.r, res.g, res.b);
  }

  return res * (1.0 / num_samples);
}

void PathTracer::raytrace_tile(int tile_x, int tile_y, int tile_w, int tile_h) {
  size_t w = sampleBuffer.w;
  size_t h = sampleBuffer.h;

  size_t tile_start_x = tile_x;
  size_t tile_start_y = tile_y;

  size_t tile_end_x = std::min(tile_start_x + tile_w, w);
  size_t tile_end_y = std::min(tile_start_y + tile_h, h);

  size_t tile_idx_x = tile_x / imageTileSize;
  size_t tile_idx_y = tile_y / imageTileSize;
  size_t num_samples_tile = tile_samples[tile_idx_x + tile_idx_y * num_tiles_w];

  for (size_t y = tile_start_y; y < tile_end_y; y++) {
    if (!continueRaytracing) return;
    for (size_t x = tile_start_x; x < tile_end_x; x++) {
      Spectrum s = raytrace_pixel(x, y);
      sampleBuffer.update_pixel(s, x, y);
    }
  }

  tile_samples[tile_idx_x + tile_idx_y * num_tiles_w] += 1;
  sampleBuffer.toColor(frameBuffer, tile_start_x, tile_start_y, tile_end_x,
                       tile_end_y);
}

void PathTracer::worker_thread() {
  Timer timer;
  timer.start();

  WorkItem work;
#if MPI
  int round_count = 0;
  int chunk_size = workQueue.get_size() / mpi_pcount;
  int chunk_offset = workQueue.get_size() % mpi_pcount;
  int round_start = chunk_size * mpi_id + (mpi_id < chunk_offset ? mpi_id : chunk_offset);
  int round_end = round_start + chunk_size + (mpi_id < chunk_offset);
  while (continueRaytracing && workQueue.try_get_work(&work)) {
    // if (round_count % mpi_pcount == mpi_id)
    if (round_count >= round_start && round_count < round_end)
      raytrace_tile(work.tile_x, work.tile_y, work.tile_w, work.tile_h);
    round_count++;
  }
#else
  while (continueRaytracing && workQueue.try_get_work(&work)) {
    raytrace_tile(work.tile_x, work.tile_y, work.tile_w, work.tile_h);
  }
#endif

#if MPI
  MPI_Barrier(MPI_COMM_WORLD);
  timer.stop();
  if (mpi_id == 0)
  fprintf(stdout, "Done! (%.4fs)\n", timer.duration());
  state = DONE;
#else
  workerDoneCount++;
  if (!continueRaytracing && workerDoneCount == numWorkerThreads) {
    timer.stop();
    fprintf(stdout, "Canceled!\n");
    state = READY;
  }

  if (continueRaytracing && workerDoneCount == numWorkerThreads) {
    timer.stop();
    fprintf(stdout, "Done! (%.4fs)\n", timer.duration());
    state = DONE;
  }
#endif
}

#if MPI
void PathTracer::collect_frame_buffer() {
  // collect framebuffer
  int datalen = frameBuffer.data.size();
  if (mpi_id == 0) {
    fprintf(stdout, "[PathTracer] Collecting frame... ");
    fflush(stdout);
    Timer timer;
    timer.start();
    
    MPI_Request requests[mpi_pcount-1];
    // uint32_t framebufferbuffer[mpi_pcount-1][datalen];
    // memset(framebufferbuffer, 0, sizeof(uint32_t) * (mpi_pcount-1) * datalen);
    // for (int pid = 1; pid < mpi_pcount; pid++) {
    //   fprintf(stdout, "Recving %d...\n", pid);
    //   MPI_Recv(&framebufferbuffer[pid-1], datalen, MPI_UNSIGNED, pid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }

    // // update framebuffer
    // WorkItem work;
    // int round_count = 0;
    // while (workQueue.try_get_work_copy(&work)) {
    //   int mpid = round_count % mpi_pcount;
    //   round_count++;
    //   if (mpid == 0) {
    //     // fprintf(stdout, ".");
    //     continue;
    //   }
    //   mpi_update_tile(framebufferbuffer[mpid-1], work.tile_x, work.tile_y, work.tile_w, work.tile_h);
    // }

    WorkItem work;
    int round_count = 0;
    int chunk_size = workQueue.get_size() / mpi_pcount;
    int chunk_offset = workQueue.get_size() % mpi_pcount;
    int round_start = 0;
    int round_size = chunk_size + (mpi_id < chunk_offset);

    while(round_count < round_size) {
      round_count++;
      workQueue.try_get_work_copy(&work);
    }

    uint32_t framebufferbuffer[datalen];
    memset(framebufferbuffer, 0, sizeof(uint32_t) * datalen);
    for (int pid = 1; pid < mpi_pcount; pid++) {
      MPI_Recv(&framebufferbuffer, datalen, MPI_UNSIGNED, pid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      // update framebuffer
      round_count = 0;
      round_start = chunk_size * pid + (mpi_id < chunk_offset ? mpi_id : chunk_offset);
      round_size = chunk_size + (pid < chunk_offset);
      while(round_count < round_size) {
        round_count++;
        workQueue.try_get_work_copy(&work);
        mpi_update_tile(framebufferbuffer, work.tile_x, work.tile_y, work.tile_w, work.tile_h);
      }
    }

    timer.stop();
    fprintf(stdout, "Done! (%.4f sec)\n", timer.duration());
  }
  else {
    MPI_Request request;
    MPI_Isend(frameBuffer.data.data(), datalen, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
}

void PathTracer::mpi_update_tile(uint32_t *data, int tile_x, int tile_y, int tile_w, int tile_h) {
  size_t w = sampleBuffer.w;
  size_t h = sampleBuffer.h;

  size_t tile_start_x = tile_x;
  size_t tile_start_y = tile_y;

  size_t tile_end_x = std::min(tile_start_x + tile_w, w);
  size_t tile_end_y = std::min(tile_start_y + tile_h, h);

  size_t tile_idx_x = tile_x / imageTileSize;
  size_t tile_idx_y = tile_y / imageTileSize;
  size_t num_samples_tile = tile_samples[tile_idx_x + tile_idx_y * num_tiles_w];

  for (size_t y = tile_start_y; y < tile_end_y; y++) {
    for (size_t x = tile_start_x; x < tile_end_x; x++) {
      frameBuffer.data[x + y * w] = data[x + y * w];
    }
  }
}
#endif

void PathTracer::increase_area_light_sample_count() {
  ns_area_light *= 2;
  fprintf(stdout, "[PathTracer] Area light sample count increased to %zu!\n",
          ns_area_light);
}

void PathTracer::decrease_area_light_sample_count() {
  if (ns_area_light > 1) ns_area_light /= 2;
  fprintf(stdout, "[PathTracer] Area light sample count decreased to %zu!\n",
          ns_area_light);
}

bool PathTracer::is_done() {
  update_screen();
  return (state == DONE);
}

void PathTracer::save_image(string fname) {
  if (state != DONE) return;

#if MPI
  if (mpi_id != 0) return;
#endif

  uint32_t *frame = &frameBuffer.data[0];
  size_t w = frameBuffer.w;
  size_t h = frameBuffer.h;
  uint32_t *frame_out = new uint32_t[w * h];
  for (size_t i = 0; i < h; ++i) {
    memcpy(frame_out + i * w, frame + (h - i - 1) * w, 4 * w);
  }

  fprintf(stderr, "[PathTracer] Saving to file: %s... ", fname.c_str());
  lodepng::encode(fname, (unsigned char *)frame_out, w, h);
  fprintf(stderr, "Done!\n");
}

}  // namespace CMU462
