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
  inline BlockingQueue() {}
  inline ~BlockingQueue() {}

  void push(Type& t) {
    {
      std::unique_lock<std::mutex> lck(mutex_);
      q_.push(t);
    }
    sema_.notify();
  }

  Type pop() {
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
    CHECK(!q_.empty());
    Type t = q_.front();
    q_.pop();
    return t;
  }

  Type &pop(Type &t) {
    sema_.wait();
    std::unique_lock<std::mutex> lck(mutex_);
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

 private:
  mutable std::mutex  mutex_;
  WeakSemaphore       sema_;
  std::queue<Type>    q_;
};

} // namespace mf

#endif //FASTMF_BLOCKING_QUEUE_H
