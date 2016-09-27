#ifndef FASTMF_BLOCKING_QUEUE_H
#define FASTMF_BLOCKING_QUEUE_H

//
// TODO: Lockfree version
//

#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace mf
{

#define CHECK_BQUEUE_TIMING

#ifdef CHECK_BQUEUE_TIMING
#define IF_CHECK_TIMING(__t$) __t$
#else
#define IF_CHECK_TIMING(__t$)
#endif

class WeakSemaphore
{
 public:
  inline WeakSemaphore(size_t count = 0)
  : count_(count) {
  }

  inline void notify()
  {
    std::unique_lock<std::mutex> lck(mutex_);
    ++count_;
    cv_.notify_one();
  }

  inline void wait()
  {
    std::unique_lock<std::mutex> lck(mutex_);
    while(count_ == 0)
    {
      cv_.wait(lck);
    }
    --count_;
  }

 private:
  std::mutex                mutex_;
  std::condition_variable   cv_;
  std::atomic<size_t>       count_;
};

template<class Type>
class BlockingQueue
{
 public:
  inline BlockingQueue()
  IF_CHECK_TIMING( : blocked_time_(0) ) {
  }
  inline ~BlockingQueue() {}

  void push(Type& t) {
    {
      std::unique_lock<std::mutex> lck(mutex_);
      q_.push(t);
    }
    sema_.notify();
  }

  Type pop() {
    IF_CHECK_TIMING( const std::chrono::time_point<Time> start = Time::now(); )
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    IF_CHECK_TIMING( { std::unique_lock<std::mutex> lk1(blocked_time_mutex_);
                       blocked_time_ += Time::now() - start; }
    )
    CHECK(!q_.empty());
    Type t = q_.front();
    q_.pop();
    return t;
  }

  Type &pop(Type &t) {
    IF_CHECK_TIMING( const std::chrono::time_point<Time> start = Time::now(); )
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    IF_CHECK_TIMING( { std::unique_lock<std::mutex> lk1(blocked_time_mutex_);
                       blocked_time_ += Time::now() - start; }
    )
    CHECK(!q_.empty());
    t = q_.front();
    q_.pop();
    return t;
  }

  bool empty() const {
    // Would be better if size check were on an atomic var, so that we can avoid the lock
    std::unique_lock<std::mutex> lck(mutex_);
    return q_.empty();
  }

  size_t size() const {
    // Would be better if size check were on an atomic var, so that we can avoid the lock
    std::unique_lock<std::mutex> lck(mutex_);
    return q_.size();
  }

  IF_CHECK_TIMING(

  std::chrono::duration<float> getBlockedTime() const {
    std::unique_lock<std::mutex> lk1(blocked_time_mutex_);
    return blocked_time_;
  }

  void printBlockedTime(const std::string& label, const bool reset) {
    if(!label.empty()) {
      std::cout << label << ": ";
    }
    float dt;
    {
      std::unique_lock<std::mutex> lk1(blocked_time_mutex_);
      dt = blocked_time_.count();
      if(reset) {
        blocked_time_ = std::chrono::duration<float>(0);
      }
    }
    std::cout << dt << std::endl << std::flush;
  }

  )

 private:
  mutable std::mutex            mutex_;
  WeakSemaphore                 sema_;
  std::queue<Type>              q_;
  IF_CHECK_TIMING(
    std::mutex                    blocked_time_mutex_;
    std::chrono::duration<float>  blocked_time_;
  )
};

} // namespace mf

#endif //FASTMF_BLOCKING_QUEUE_H
