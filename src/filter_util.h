#ifndef FASTMF_FILTER_UTIL_H
#define FASTMF_FILTER_UTIL_H

#include <list>
#include <iostream>
#include <dmlc/concurrency.h>

namespace mf
{

/**
 *   ____   _      _              _    _____               _
 *  / __ \ | |    (_)            | |  |  __ \             | |
 * | |  | || |__   _   ___   ___ | |_ | |__) |___    ___  | |
 * | |  | || '_ \ | | / _ \ / __|| __||  ___// _ \  / _ \ | |
 * | |__| || |_) || ||  __/| (__ | |_ | |   | (_) || (_) || |
 *  \____/ |_.__/ | | \___| \___| \__||_|    \___/  \___/ |_|
 *               _/ |
 *              |__/
 */
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

/**
 *   _____  _          _                 _____  _                  _
 *  / ____|| |        | |               / ____|| |                | |
 * | (___  | |_  __ _ | |_  _   _  ___ | |     | |__    ___   ___ | | __
 *  \___ \ | __|/ _` || __|| | | |/ __|| |     | '_ \  / _ \ / __|| |/ /
 *  ____) || |_| (_| || |_ | |_| |\__ \| |____ | | | ||  __/| (__ |   <
 * |_____/  \__|\__,_| \__| \__,_||___/ \_____||_| |_| \___| \___||_|\_\
 *
 *
 */
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

/**
 *  _____  ______  _  _  _
 * |_   _||  ____|(_)| || |
 *   | |  | |__    _ | || |_  ___  _ __
 *   | |  |  __|  | || || __|/ _ \| '__|
 *  _| |_ | |     | || || |_|  __/| |
 * |_____||_|     |_||_| \__|\___||_|
 *
 *
 */
struct IFilter {
  virtual bool flush(uint64_t expectedCount, size_t microsecondsSleep) = 0;
  virtual void onUpstreamError() = 0;
  virtual void addDownstreamFilter(IFilter *f) = 0;
};

/**
 *  _____  _               _  _               ______  _  _  _
 * |  __ \(_)             | |(_)             |  ____|(_)| || |
 * | |__) |_  _ __    ___ | | _  _ __    ___ | |__    _ | || |_  ___  _ __
 * |  ___/| || '_ \  / _ \| || || '_ \  / _ \|  __|  | || || __|/ _ \| '__|
 * | |    | || |_) ||  __/| || || | | ||  __/| |     | || || |_|  __/| |
 * |_|    |_|| .__/  \___||_||_||_| |_| \___||_|     |_||_| \__|\___||_|
 *           | |
 *           |_|
 */
template<typename ObjectType = void *>
class PipelineFilter : public tbb::filter
                     , public StatusStack
                     , public IFilter {
 protected:
  typedef ObjectType object_t;

 public:

  PipelineFilter(tbb::filter::mode filter_mode,
                 mf::ObjectPool<ObjectType> *source_buffer_pool)
  : tbb::filter(filter_mode)
    , source_buffer_pool_(source_buffer_pool)
    , items_processed_(0)
    , downstream_finished_(false)
    , upstream_error_(false)
  {
  }

  virtual ~PipelineFilter() {}

  void setTiming(perf::TimingInstrument *timing) {
    timing_ = timing;
  }

  virtual void *execute(object_t *) = 0;

  virtual void *operator()(void *v) {
    void *res = nullptr;
    if(!upstream_error_) {
      bool stop = false;
      ObjectType *val = reinterpret_cast<ObjectType *>(v);
      try {
        res = execute(val);
      } catch (std::runtime_error &e) {
        addStatus(EXECUTION_EXCEPTION, e.what());
        stop = true;
      } catch (...) {
        addStatus(UNHANDLED_EXCEPTION, "Unhandled exception");
        stop = true;
      }
      if (source_buffer_pool_) {
        if (val) {
          source_buffer_pool_->freeObject(val);
        }
        if (stop) {
          source_buffer_pool_->terminate();
        }
      }
      ++items_processed_;
      if (!res) {
        downstream_finished_.store(true);
      }
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
    if (!upstream_error_) {
      if (!downstream_finished_ || downstream_filters_.empty()) {
        for (IFilter *f : downstream_filters_) {
          if (!f->flush(expectedCount, microsecondsSleep)) {
            return false;
          }
        }
        while (items_processed_.load() < expectedCount) {
          if (upstream_error_ || (downstream_finished_ && !downstream_filters_.empty())) {
            return false;
          }
          usleep(microsecondsSleep);
        }
        return true;
      }
    }
    return false;
  }

  void onUpstreamError() {
    upstream_error_ = true;
    for (IFilter *f : downstream_filters_) {
      f->onUpstreamError();
    }
  }

 private:
  mf::ObjectPool<ObjectType> *    source_buffer_pool_;
  std::atomic<uint64_t>           items_processed_;
  std::atomic<bool>               downstream_finished_;
  std::atomic<bool>               upstream_error_;
  std::unordered_set<IFilter *>   downstream_filters_;
 protected:
  perf::TimingInstrument *        timing_;
};

/**
 *  _____  _               _  _
 * |  __ \(_)             | |(_)
 * | |__) |_  _ __    ___ | | _  _ __    ___
 * |  ___/| || '_ \  / _ \| || || '_ \  / _ \
 * | |    | || |_) ||  __/| || || | | ||  __/
 * |_|    |_|| .__/  \___||_||_||_| |_| \___|
 *           | |
 *           |_|
 */
class Pipeline : protected tbb::pipeline {
 public:
  Pipeline()
    : last_filter_(nullptr) {}

  virtual ~Pipeline() {}

  using tbb::pipeline::run;

  template<typename ObjectType>
  void add_filter(PipelineFilter<ObjectType>& filter) {
    if(last_filter_)  {
      last_filter_->addDownstreamFilter(&filter);
    }
    last_filter_ = &filter;
    filter.setTiming(&timing_);
    tbb::pipeline::add_filter(filter);
  }

  void clear() {
    last_filter_ = nullptr;
    tbb::pipeline::clear();
  }

 public:
  perf::TimingInstrument    timing_;

 private:
  IFilter *                 last_filter_;
};

} // namespace mf

#endif //FASTMF_FILTER_UTIL_H
