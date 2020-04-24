#ifndef __WORK_QUEUE_H__
#define __WORK_QUEUE_H__

#ifndef MPI
#define MPI 1
#endif

#if MPI
#include <mpi.h>
#endif

#include <mutex>
#include <vector>

/**
 * Note that this queue has no wait-until-more-work-is-added capability; it's
 * intended for more isolated or batch-processing-like situations.
 */
template <class T>
class WorkQueue {
 private:
#if MPI
  std::vector<T> storage_copy;
#endif
  std::vector<T> storage;
  std::mutex lock;

 public:
  WorkQueue() {}

  bool is_empty() {
    lock.lock();
    bool result = storage.empty();
    lock.unlock();
    return result;
  }

  bool try_get_work(T* outPtr) {
    lock.lock();
    if (storage.empty()) {
      lock.unlock();
      return false;
    }
    *outPtr = storage.front();
    storage.erase(storage.begin());
    lock.unlock();
    return true;
  }

#if MPI
  bool try_get_work_copy(T* outPtr) {
    if (storage_copy.empty()) {
      return false;
    }
    *outPtr = storage_copy.front();
    storage_copy.erase(storage_copy.begin());
    return true;
  }
#endif

  void put_work(const T& item) {
    lock.lock();
    storage.push_back(item);
#if MPI
    storage_copy.push_back(item);
#endif
    lock.unlock();
  }

  void clear() {
    lock.lock();
    storage.clear();
#if MPI
    storage_copy.clear();
#endif
    lock.unlock();
  }
};

#endif  // WORK_QUEUE_H_
