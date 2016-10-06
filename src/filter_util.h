#ifndef FASTMF_FILTER_UTIL_H
#define FASTMF_FILTER_UTIL_H

#include <list>
#include "blocking_queue.h"
#include <dmlc/concurrency.h>

namespace mf
{

#define USE_DMLC_QUEUE // Use Dmlc-core's blocking queue

template<typename Object>
class ObjectPool
{
 public:
  ObjectPool(size_t pool_size)
    : pool_size_(pool_size) {
#pragma omp parallel for
    for (size_t x = 0; x < pool_size_; ++x) {
      Object *v = new Object();
      free_object_pool_.Push(v);
    }
  }

  ~ObjectPool() {
    CHECK_EQ(free_object_pool_.Size(), pool_size_);
    while (free_object_pool_.Size()) {
#ifdef USE_DMLC_QUEUE
      Object *v = NULL;
      free_object_pool_.Pop(&v);
#else
      Object *v = free_object_pool_.Pop();
#endif
      if(v) {
        delete v;
      }
    }
  }

  mf::BlockingQueue<Object *> &getFreePool() {
    return free_object_pool_;
  }

  Object *allocateObject() {
#ifdef USE_DMLC_QUEUE
    Object *v = NULL;
    free_object_pool_.Pop(&v);
    return v;
#else
    return free_object_pool_.Pop();
#endif
  }

  void freeObject(Object *obj) {
    CHECK_NOTNULL(obj);
    free_object_pool_.Push(obj);
  }

  IF_CHECK_TIMING(
    void printBlockedTime(const std::string& label, bool reset = false) {
#ifndef USE_DMLC_QUEUE
      free_object_pool_.printBlockedTime(label, reset);
#endif
    }
  )

 private:
  const size_t                pool_size_;
#ifndef USE_DMLC_QUEUE
  mf::BlockingQueue<Object *> free_object_pool_;
#else
  dmlc::ConcurrentBlockingQueue<Object *> free_object_pool_;
#endif
};


class StatusStack {

 public:

  enum StatusCode {
    OK = 0,
    PARSE_ERROR = 0x81000000,
    IO_ERROR,
    POOL_ERROR,
    UNHANDLED_ERROR
  };

  struct Status {
    StatusCode  code;
    std::string message;
  };

  StatusStack(const size_t max_statuses = 10)
  : max_statuses_(max_statuses) {
    CHECK_NE(max_statuses, 0);
  }

  // Currently everything is an error. If warnings or info is added,
  // hash-map by error type and check that key's emptiness.
  bool error() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return !statuses_.empty();
  }

  void addStatus(const StatusCode code, const char *msg = NULL) {
    std::unique_lock<std::mutex> lk(mutex_);
    statuses_.push_back({code, msg ? msg : ""});
    while(statuses_.size() > max_statuses_) {
      statuses_.pop_front();
    }
  }

  StatusCode getLastStatusCode() const {
    std::unique_lock<std::mutex> lk(mutex_);
    if(statuses_.empty()) {
      return OK;
    }
    return statuses_.rbegin()->code;
  }

 private:
  mutable std::mutex  mutex_;
  std::list<Status>   statuses_;
  const size_t        max_statuses_;
};

enum FilterStages {
  FILTER_STAGE_READ,
  FILTER_STAGE_PARSE,
  FILTER_STAGE_CALC
};

class PipelineFilter : public tbb::filter
                     , public StatusStack {
 public:
  PipelineFilter(tbb::filter::mode filter_mode)
  : tbb::filter(filter_mode) {

  }
  virtual ~PipelineFilter() {
  }
  virtual void *execute(void *) = 0;

  virtual void *operator()(void *v) {
    try {
      return execute(v);
    }
    catch(...) {
      addStatus(UNHANDLED_ERROR, "Unhandled exception");
    }
    return NULL; // Filter is finished
  }
};

} // namespace mf

#endif //FASTMF_FILTER_UTIL_H
