#ifndef FASTMF_FILTER_UTIL_H
#define FASTMF_FILTER_UTIL_H

#include <list>
#include "blocking_queue.h"

namespace mf
{

template<typename Object>
class ObjectPool
{
 public:
  ObjectPool(size_t pool_size)
    : pool_size_(pool_size) {
#pragma omp parallel for
    for (size_t x = 0; x < pool_size_; ++x) {
      Object *v = new Object();
      free_object_pool_.push(v);
    }
  }

  ~ObjectPool() {
    CHECK_EQ(free_object_pool_.size(), pool_size_);
    while (!free_object_pool_.empty()) {
      Object *v = free_object_pool_.pop();
      delete v;
    }
  }

  mf::BlockingQueue<Object *> &getFreePool() {
    return free_object_pool_;
  }

  Object *allocateObject() {
    return free_object_pool_.pop();
  }

  void freeObject(Object *obj) {
    CHECK_NOTNULL(obj);
    free_object_pool_.push(obj);
  }

  IF_CHECK_TIMING(
    void printBlockedTime(const std::string& label, bool reset = false) {
      free_object_pool_.printBlockedTime(label, reset);
    }
  )

 private:
  const size_t                pool_size_;
  mf::BlockingQueue<Object *> free_object_pool_;
};


class StatusStack {

 public:

  enum StatusCode {
    OK = 0,
    PARSE_ERROR = 0x81000000,
    IO_ERROR,
    POOL_ERROR
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

} // namespace mf

#endif //FASTMF_FILTER_UTIL_H
