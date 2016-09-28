#ifndef FASTMF_BLOCKING_QUEUE_H
#define FASTMF_BLOCKING_QUEUE_H

//
// TODO: Lockfree version
//

#include <queue>
#include <mutex>
#include <atomic>
#include <iostream>
#include <condition_variable>
#include "perf.h"

namespace mf
{

#define CHECK_BQUEUE_TIMING

#ifdef CHECK_BQUEUE_TIMING
#define IF_CHECK_TIMING(__t$) __t$
#else
#define IF_CHECK_TIMING(__t$)
#endif

/**
 * WeakSemaphore
 * Simple semaphore which does not make any attempt to
 * maintain lock/release order
 */
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
  IF_CHECK_TIMING( : blocked_time_(0) )
  {
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
    IF_CHECK_TIMING( const uint64_t start = perf::getMicroTickCount(); )
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    IF_CHECK_TIMING( blocked_time_.fetch_add( perf::getMicroTickCount() - start ) );
    CHECK(!q_.empty());
    Type t = q_.front();
    q_.pop();
    return t;
  }

  Type &pop(Type &t) {
    IF_CHECK_TIMING( const uint64_t start = perf::getMicroTickCount(); )
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    IF_CHECK_TIMING( blocked_time_.fetch_add( perf::getMicroTickCount() - start ) );
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

  uint64_t getBlockedTime() const {
    return blocked_time_.load();
  }

  void printBlockedTime(const std::string& label, const bool reset) {
    if(!label.empty()) {
      std::cout << label << ": ";
    }
    std::cout << perf::toString(MICRO2MSF(blocked_time_.load()))
              << " ms" << std::endl << std::flush;
    if(reset) {
      blocked_time_.store(0);
    }
  }

  );

 private:
  mutable std::mutex            mutex_;
  WeakSemaphore                 sema_;
  std::queue<Type>              q_;
  IF_CHECK_TIMING(
    std::atomic<uint64_t>           blocked_time_;
  )
};

} // namespace mf

#endif //FASTMF_BLOCKING_QUEUE_H
