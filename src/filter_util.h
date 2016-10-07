#ifndef FASTMF_FILTER_UTIL_H
#define FASTMF_FILTER_UTIL_H

#include <list>
#include <iostream>
#include <dmlc/concurrency.h>

namespace mf
{

template<typename Object>
class ObjectPool
{
 public:
  ObjectPool(size_t pool_size)
    : pool_size_(pool_size)
    , duration_(0)  {
    all_objects_.reserve(pool_size);
#pragma omp parallel for
    for (size_t x = 0; x < pool_size_; ++x) {
      Object *v = new Object();
      all_objects_.emplace_back(std::unique_ptr<Object>(v));
      free_object_pool_.Push(v);
    }
  }

  ~ObjectPool() {
    CHECK_EQ(free_object_pool_.Size(), pool_size_);
    free_object_pool_.SignalForKill();
  }

  Object *allocateObject() {
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> s
      = std::chrono::high_resolution_clock::now();
    Object *v = NULL;
    free_object_pool_.Pop(&v);
    const long diff = (std::chrono::high_resolution_clock::now() - s).count();
    duration_.fetch_add((uint64_t)diff);
    return v;
  }

  void freeObject(Object *obj) {
    CHECK_NOTNULL(obj);
    free_object_pool_.Push(obj);
  }

  void terminate() {
    free_object_pool_.SignalForKill();
  }

  std::chrono::nanoseconds getBlockedTime() const {
    return std::chrono::nanoseconds(duration_);
  }

  void printBlockedTime(const std::string& label, bool reset = false) {
    if(!label.empty()) {
      std::cout << label << ": ";
    }
    std::cout << mf::perf::toString(NANO2MSF(duration_.load()))
              << " ms" << std::endl << std::flush;
    if(reset) {
      duration_.store(0);
    }
  }

 private:
  const size_t                            pool_size_;
  dmlc::ConcurrentBlockingQueue<Object *> free_object_pool_;
  std::vector<std::unique_ptr<Object>>    all_objects_;
  std::atomic<uint64_t>                   duration_;
};


class StatusStack {

 public:

  enum StatusCode {
    OK = 0,
    PARSE_ERROR = 0x81000000,
    IO_ERROR,
    POOL_ERROR,
    EXECUTION_EXCEPTION,
    UNHANDLED_EXCEPTION
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

  void printAll() const {
    std::unique_lock<std::mutex> lk(mutex_);
    for(auto i = statuses_.rbegin(), e = statuses_.rend(); i != e; ++i) {
      std::cerr << "Status code: " << i->code << " ( " << i->message << " )" << std::endl;
    }
    std::cerr << std::flush;
  }

 private:
  mutable std::mutex  mutex_;
  std::list<Status>   statuses_;
  const size_t        max_statuses_;
};

enum FilterStages {
  FILTER_STAGE_READ,
  FILTER_STAGE_PARSE,
  FILTER_STAGE_CALC,
  FILTER_STAGE_FLUSH
};

struct IFilter {
  virtual bool flush(uint64_t expectedCount, size_t microsecondsSleep) = 0;
};

template<typename ObjectType = void *>
class PipelineFilter : public tbb::filter
                     , public StatusStack
                     , public IFilter {
  // TODO: Do some timing, data-flow metrics (ie downstream filters waiting on data)
 protected:
  typedef ObjectType object_t;

 public:

  PipelineFilter(tbb::filter::mode filter_mode,
                 mf::ObjectPool<ObjectType> *source_buffer_pool)
  : tbb::filter(filter_mode)
    , source_buffer_pool_(source_buffer_pool)
    , items_processed_(0)
    , downstream_finished_(false)
  {
  }

  virtual ~PipelineFilter() {}

  virtual void *execute(object_t *) = 0;

  virtual void *operator()(void *v) {
    bool stop = false;
    void *res = NULL;
    ObjectType *val = reinterpret_cast<ObjectType *>(v);
    try {
      res = execute(val);
    } catch(std::runtime_error& e) {
      addStatus(EXECUTION_EXCEPTION, e.what());
      stop = true;
    } catch(...) {
      addStatus(UNHANDLED_EXCEPTION, "Unhandled exception");
      stop = true;
    }
    if(source_buffer_pool_) {
      if(val) {
        source_buffer_pool_->freeObject(val);
      }
      if(stop) {
        source_buffer_pool_->terminate();
      }
    }
    ++items_processed_;
    if(!res) {
      downstream_finished_.store(true);
    }
    return res; // Signal filter finished
  }

  /**
   * Return when downstream filters are caught up
   * returns false if there was a problem (ie downstream filter has had an error)
   */
  bool flush() {
    return flush(items_processed_.load());
  }

  void addDownstreamFilter(IFilter *f) {
    CHECK_NOTNULL(f);
    if(f) {
      // should not change while running
      CHECK_EQ(items_processed_, 0);
      downstream_filters_.insert(f);
    }
  }

 private:
  /**!
  * \brief Return when downstream is at or past the given iteration
  *        Not meant to be called in a high-performance loop
  */
  bool flush(uint64_t expectedCount, size_t microsecondsSleep = 100) {
    if(!downstream_finished_ || downstream_filters_.empty()) {
      for (IFilter *f : downstream_filters_) {
        if(!f->flush(expectedCount, microsecondsSleep))  {
          return false;
        }
      }
      while(items_processed_.load() < expectedCount) {
        if(downstream_finished_ && !downstream_filters_.empty()) {
          return false;
        }
        usleep(microsecondsSleep);
      }
      return true;
    } else {
      return false;
    }
  }


 private:
  mf::ObjectPool<ObjectType> *    source_buffer_pool_;
  std::atomic<uint64_t>           items_processed_;
  std::atomic<bool>               downstream_finished_;
  std::unordered_set<IFilter *>   downstream_filters_;
};

} // namespace mf

#endif //FASTMF_FILTER_UTIL_H
