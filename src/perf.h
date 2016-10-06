#ifndef _AWSDL_PERF_H
#define _AWSDL_PERF_H

#include <iostream>
#include <sys/time.h>
#include <iomanip>
#include <unordered_set>

namespace mf
{

namespace perf
{

inline uint64_t getTickCount() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return uint64_t(tv.tv_sec) * 1000 + (uint64_t(tv.tv_usec) / 1000);
}

// millionths of a second
inline uint64_t getMicroTickCount() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return uint64_t(tv.tv_sec) * 1000000 + tv.tv_usec;
}

#define MICRO2MS(__micro$)  (((__micro$) + 500)/1000)
#define MICRO2MSF(__micro$) (float(__micro$)/1000)
#define NANO2MSF(__nano$)   (float(__nano$)/1000000)
#define MICRO2S(__micro$)   (((__micro$) + 500000)/1000000)
#define MICRO2SF(__micro$)  (MICRO2MSF(__micro$)/1000)

template<typename T>
inline std::string toString(const T f) {
  std::stringstream buf;
  buf << std::setprecision(4) << std::fixed << f;
  return buf.str();
}

class TimedScope
{
  std::string label;
  const uint64_t startTime;
 public:
  TimedScope(const char *msg = NULL)
    : startTime(getMicroTickCount()) {
    if (msg && *msg) {
      label = msg;
    }
  }

  uint64_t elapsed() const {
    return getMicroTickCount() - startTime;
  }

  void print() const {
    const uint64_t diff = elapsed();
    if (!label.empty()) {
      std::cout << label << " ";
    }
    std::cout << "elapsed time: "
              << std::setprecision(4) << std::fixed << MICRO2MSF(diff) << " ms"
              << std::endl;
  }

  ~TimedScope() {
    print();
  }
};

class TimingInstrument {
 public:
  TimingInstrument(const char *name = "")
    : name_(name) {
  }
  ~TimingInstrument() {

  }
  void startTiming(int id, const char *s) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    std::unordered_map<int, Info>::iterator i = data_.find(id);
    if(i == data_.end()) {
      i = data_.insert(std::make_pair(id, Info(s))).first;
    }
    if(!i->second.nestingCount_++) {
      i->second.baseTime_ = getMicroTickCount();
    }
  }
  void stopTiming(int id) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    std::unordered_map<int, Info>::iterator i = data_.find(id);
    CHECK_NE(i, data_.end()) << "Can't stop timing on an object that we don't know about";
    if(i != data_.end()) {
      CHECK_NE(i->second.nestingCount_, 0) << "While stopping timing, invalid nesting count of 0";
      if(!--i->second.nestingCount_) {
        CHECK_NE(i->second.baseTime_, 0) << "Invalid base time";
        i->second.duration_.fetch_add(getMicroTickCount() - i->second.baseTime_);
        i->second.baseTime_  = 0;
      }
    }
  }
  uint64_t getDuration(int id) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    std::unordered_map<int, Info>::iterator i = data_.find(id);
    if(i != data_.end()) {
      const Info&        info = i->second;
      const uint64_t duration = info.nestingCount_.load()
                                ? info.duration_.load() + (getMicroTickCount() - info.baseTime_.load())
                                : info.duration_.load();
      return duration;
    }
    return 0;
  }
  bool isTiming(int id) {
    std::unordered_map<int, Info>::const_iterator i = data_.find(id);
    if(i != data_.end()) {
      return !!i->second.nestingCount_.load();
    }
    return false;
  }
  void print(bool doReset = false) {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    for(std::unordered_map<int, Info>::const_iterator i = data_.begin(), e = data_.end();
        i != e; ++i) {
      const Info&        info = i->second;
      const uint64_t duration = getDuration(i->first);
      std::cout << name_ << " Timing [" << info.name_ << "] "
                << (info.nestingCount_.load() ? "*" : "")
                << MICRO2MSF(duration) << " ms" << std::endl;
    }
    std::cout << std::flush;
    if(doReset) {
      reset();
    }
  }
  void reset() {
    std::unique_lock<std::recursive_mutex>  lk(mutex_);
    for(std::unordered_map<int, Info>::iterator i = data_.begin(), e = data_.end();
        i != e; ++i) {
      const int id = i->first;
      const bool wasTiming = isTiming(id);
      if(wasTiming) {
        stopTiming(id);
      }
      // need zero count here
      CHECK_EQ(i->second.nestingCount_.load(), 0);
      i->second.duration_ = 0;
      if(wasTiming) {
        startTiming(id, i->second.name_.c_str());
      }
    }
  }
 private:
  struct Info {
    inline Info(const char *s)
      : name_(s ? s : "")
        , baseTime_(0)
        , nestingCount_(0)
        , duration_(0) {}
    inline Info(const Info& o)
      : name_(o.name_)
        , baseTime_(o.baseTime_.load())
        , nestingCount_(o.nestingCount_.load())
        , duration_(o.duration_.load())
    {}
    std::string           name_;
    std::atomic<uint64_t> baseTime_;
    std::atomic<uint64_t> nestingCount_;
    std::atomic<uint64_t> duration_;
  };
  std::string                   name_;
  mutable std::recursive_mutex  mutex_;
  std::unordered_map<int, Info> data_;
};

struct TimingItem {
  TimingItem(TimingInstrument *ti, int id, const char *name)
    : ti_(ti)
    , id_(id) {
    if(ti_) {
      ti_->startTiming(id, name);
    }
  }
  ~TimingItem() {
    if(ti_) {
      ti_->stopTiming(id_);
    }
  }
 private:
  TimingInstrument *ti_;
  const int         id_;
};

#ifndef NDEBUG

/**
 * Maintain a set of items with duplication assertion
 */
template<typename T>
class DebugUsing {
 public:
  void add(const T& v) {
    std::unique_lock<std::mutex> lk(mutex_);
    if(!using_.insert(v).second)
    {
      CHECK_EQ(using_.insert(v).second, true);
    }
  }
  void remove(const T& v) {
    std::unique_lock<std::mutex> lk(mutex_);
    using_.erase(v);
  }
 private:
  std::mutex            mutex_;
  std::unordered_set<T> using_;
};

/**
 * DebugCheckUsing
 * Scoped addition/removal to/from DebugUsing
 */
template<typename T>
class DebugCheckUsing {
 public:
  DebugCheckUsing(const T v, DebugUsing<T>& use)
    : val_(v)
      , use_(use) {
    have_ = true;
    use_.add(v);
  }
  ~DebugCheckUsing() {
    release();
  }
  void release() {
    if(have_.load()) {
      have_ = false;
      use_.remove(val_);
    }
  }
 private:
  std::atomic<bool>     have_;
  const T               val_;
  DebugUsing<T>&        use_;
};
#endif //NDEBUG

} // namespace perf
} // namespace mf

#endif //_AWSDL_PERF_H
