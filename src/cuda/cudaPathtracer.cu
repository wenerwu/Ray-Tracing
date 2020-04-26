
#include "CMU462/CMU462.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"
#include "CMU462/lodepng.h"

#include "../static_scene/sphere.h"
#include "../static_scene/triangle.h"
#include "../static_scene/light.h" 

#include "cudaPathtracer.h" 

using namespace CMU462;
using namespace StaticScene;

using std::min;
using std::max;


cudaPathTracer::cudaPathTracer(PathTracer* _pathTracer) {
    pathtracer = _pathTracer;

}

cudaPathTracer::~cudaPathTracer() {
    // delete bvh;
    // delete gridSampler;
    // delete hemisphereSampler;
}



void cudaPathTracer::set_scene(Scene *scene) {
    pathtracer->set_scene(scene);
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
    pathtracer->set_camera(camera);
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
  
// void cudaPathTracer::stop() {
  
//     switch (state) {
//       case INIT:
//       case READY:
//         break;
//       case VISUALIZE:
//         while (selectionHistory.size() > 1) {
//           selectionHistory.pop();
//         }
//         state = READY;
//         break;
//       case RENDERING:
//         continueRaytracing = false;
//       case DONE:
//         state = READY;
//         break;
//     }
//   }

//   void cudaPathTracer::clear() {
//     if (state != READY) return;
//     delete bvh;
//     bvh = NULL;
//     scene = NULL;
//     camera = NULL;
//     selectionHistory.pop();
//     sampleBuffer.resize(0, 0);
//     frameBuffer.resize(0, 0);
//     state = INIT;
//   } 
  
void cudaPathTracer::start_raytracing() {
    pathtracer->start_raytracing();
//     if (state != READY) return;
  
//     state = RENDERING;
//     continueRaytracing = true;
   
//     sampleBuffer.clear();
//     frameBuffer.clear();
//     num_tiles_w = sampleBuffer.w / imageTileSize + 1;
//     num_tiles_h = sampleBuffer.h / imageTileSize + 1;
//     tile_samples.resize(num_tiles_w * num_tiles_h);
//     memset(&tile_samples[0], 0, num_tiles_w * num_tiles_h * sizeof(int));
  
//     // populate the tile work queue
    
  
//     // for (size_t y = 0; y < sampleBuffer.h; y += imageTileSize) {
//     //   for (size_t x = 0; x < sampleBuffer.w; x += imageTileSize) {
//     //     //disable thread
//     //   //  workQueue.put_work(WorkItem(x, y, imageTileSize, imageTileSize));
//     //     raytrace_tile(x, y, imageTileSize, imageTileSize);
//     //   }
//     // }
  
//     // launch threads
//     fprintf(stdout, "[PathTracer] Rendering... ");
//     fflush(stdout);
        
// //    worker_thread(); 

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
    return pathtracer->state == DONE;
}
 
  bool cudaPathTracer::is_done_headless() {
        return pathtracer->state == DONE;

  }
