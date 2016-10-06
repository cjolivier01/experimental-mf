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
#include "sync.h"

#ifndef USE_DMLC_QUEUE

namespace mf
{

#define CHECK_BQUEUE_TIMING

#ifdef CHECK_BQUEUE_TIMING
#define IF_CHECK_TIMING(__t$) __t$
#else
#define IF_CHECK_TIMING(__t$)
#endif

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
    sema_.post();
  }

  Type Pop() {
    IF_CHECK_TIMING( const uint64_t start = mf::perf::getMicroTickCount(); )
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    IF_CHECK_TIMING( blocked_time_.fetch_add( mf::perf::getMicroTickCount() - start ) );
    CHECK(!q_.empty());
    Type t = q_.front();
    q_.pop();
    return t;
  }

  Type &Pop(Type &t) {
    IF_CHECK_TIMING( const uint64_t start = mf::perf::getMicroTickCount(); )
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    IF_CHECK_TIMING( blocked_time_.fetch_add( mf::perf::getMicroTickCount() - start ) );
    CHECK(!q_.empty());
    t = q_.front();
    q_.pop();
    return t;
  }

  bool Empty() const {
    // Would be better if size check were on an atomic var, so that we can avoid the lock
    std::unique_lock<std::mutex> lck(mutex_);
    return q_.empty();
  }

  size_t Size() const {
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
    std::cout << mf::perf::toString(MICRO2MSF(blocked_time_.load()))
              << " ms" << std::endl << std::flush;
    if(reset) {
      blocked_time_.store(0);
    }
  }

  );

 private:
  mutable std::mutex            mutex_;
  mf::semaphore              sema_;
  std::queue<Type>              q_;
  IF_CHECK_TIMING(
    std::atomic<uint64_t>       blocked_time_;
  )
};

} // namespace mf

#endif //USE_DMLC_QUEUE

#endif //FASTMF_BLOCKING_QUEUE_H
