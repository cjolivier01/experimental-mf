#ifndef _AWSDL_SYNC_H
#define _AWSDL_SYNC_H

#include "condition_algorithm_8a.h"
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <semaphore.h>
#include <unordered_set>
#include <semaphore.h>
#include <dmlc/logging.h>

//#define AWSDL_DEBUG_LOG_THREADS

namespace mf
{

/**
 *   _____  _                            _   ____   _      _              _
 *  / ____|| |                          | | / __ \ | |    (_)            | |
 * | (___  | |__    __ _  _ __  ___   __| || |  | || |__   _   ___   ___ | |_
 *  \___ \ | '_ \  / _` || '__|/ _ \ / _` || |  | || '_ \ | | / _ \ / __|| __|
 *  ____) || | | || (_| || |  |  __/| (_| || |__| || |_) || ||  __/| (__ | |_
 * |_____/ |_| |_| \__,_||_|   \___| \__,_| \____/ |_.__/ | | \___| \___| \__|
 *                                                       _/ |
 *                                                      |__/
 */
template<class ObjectType>
class SharedObject
{
 public:

  typedef std::shared_ptr<ObjectType> SharedPtr;
  typedef std::weak_ptr<ObjectType>   WeakPtr;

  // Declarable in child scope without qualifiers

  static SharedPtr create() {
    ObjectType *p = new ObjectType;
    SharedPtr result(p);
    return result;
  }

  template<class A>
  static SharedPtr create(A &arg) {
    ObjectType *p = new ObjectType(arg);
    SharedPtr result(p);
    return result;
  }

  template<class A, class B>
  static SharedPtr create(const A &arg1, const B &arg2) {
    ObjectType *p = new ObjectType(arg1, arg2);
    SharedPtr result(p);
    return result;
  }

  template<class A, class B, class C>
  static SharedPtr create(const A &arg1, const B &arg2, const C &arg3) {
    ObjectType *p = new ObjectType(arg1, arg2, arg3);
    SharedPtr result(p);
    return result;
  }

};

/**
 *                                      _
 *                                     | |
 *  ___   ___  _ __ ___    __ _  _ __  | |__    ___   _ __  ___
 * / __| / _ \| '_ ` _ \  / _` || '_ \ | '_ \  / _ \ | '__|/ _ \
 * \__ \|  __/| | | | | || (_| || |_) || | | || (_) || |  |  __/
 * |___/ \___||_| |_| |_| \__,_|| .__/ |_| |_| \___/ |_|   \___|
 *                              | |
 * Simple semaphore which does not make any attempt to
 * maintain lock/release order
 */

#ifdef __linux__
class semaphore
{
  sem_t sem_;
 public:
  semaphore(int initialCount = 0) {
    const int rc = sem_init(&sem_, 0, initialCount);
    CHECK_EQ(rc, 0);
  }

  ~semaphore() {
    sem_destroy(&sem_);
  }

  void wait() {
    sem_wait(&sem_);
  }

  bool timed_wait(const Time::time_point &abs_time) {
    const uint64_t nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
      abs_time.time_since_epoch()
    ).count();
    timespec ts;
    ts.tv_nsec = nanos % 1000000000;
    ts.tv_sec = nanos / 1000000000;
    return sem_timedwait(&sem_, &ts) == 0;
  }

  bool timed_wait(const int ms) {
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> t1 =
      std::chrono::high_resolution_clock::now();
    t1 += std::chrono::milliseconds(ms);
    const uint64_t nanos = t1.time_since_epoch().count();
    timespec ts;
    ts.tv_nsec = nanos % 1000000000;
    ts.tv_sec = nanos / 1000000000;
    return sem_timedwait(&sem_, &ts) == 0;
  }

/* Test whether SEM is posted.  */
  bool try_wait() {
    return sem_trywait(&sem_) == 0;
  }

  void post() {
    const int rc = sem_post(&sem_);
    CHECK_EQ(rc, 0);
  }

  void post(size_t count) {
    while (count--) {
      post();
    }
  }

  int get_value() {
    int val = 0;
    const int rc = sem_getvalue(&sem_, &val);
    CHECK_EQ(rc, 0);
    return val;
  }

/* Get current value of SEM and store it in *SVAL.  */
//  extern int sem_getvalue (sem_t *__restrict __sem, int *__restrict __sval)
};
typedef semaphore Semaphore;
#else
class WeakSemaphore
{
 public:
  inline WeakSemaphore(size_t count = 0)
    : count_(count) {
  }

  inline void post() {
    std::unique_lock<std::mutex> lck(mutex_);
    cv_.notify_one();
    ++count_;
  }

  void post(size_t count) {
    while (count--) {
      post();
    }
  }

  inline void wait() {
    std::unique_lock<std::mutex> lck(mutex_);
    while (count_ == 0) {
      cv_.wait(lck);
    }
    --count_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<size_t> count_;
};
typedef WeakSemaphore Semaphore;
#endif //__linux__

/**
 *  __  __         _  _    _  ______                   _
 * |  \/  |       | || |  (_)|  ____|                 | |
 * | \  / | _   _ | || |_  _ | |__ __   __ ___  _ __  | |_
 * | |\/| || | | || || __|| ||  __|\ \ / // _ \| '_ \ | __|
 * | |  | || |_| || || |_ | || |____\ V /|  __/| | | || |_
 * |_|  |_| \__,_||_| \__||_||______|\_/  \___||_| |_| \__|
 *
 *
 * Windows-like event object (ie also allows manual reset event)
 */
class MultiEvent : public SharedObject<MultiEvent>
{
  class ConditionMembers
  {
   public:

    struct mutex : public std::mutex
    {
      bool operator!() const { return false; }
    };

    struct atomic_integer : public std::atomic<int>
    {
      typedef int value_type;

      atomic_integer() {}

      atomic_integer(const value_type n)
        : std::atomic<value_type>(n) {}

      atomic_integer(const atomic_integer &o)
        : std::atomic<value_type>(o.load()) {}

      atomic_integer &operator=(const value_type n) {
        exchange(n);
        return *this;
      }

      atomic_integer &operator=(const atomic_integer &n) {
        exchange(n.load());
        return *this;
      }
    };

    typedef semaphore semaphore_type;
    typedef mutex mutex_type;
    typedef atomic_integer integer_type;
   private:
    integer_type    nwaiters_blocked_;
    integer_type    nwaiters_gone_;
    integer_type    nwaiters_to_unblock_;
    semaphore_type  sem_block_queue_;
    semaphore_type  sem_block_lock_;
    mutex_type      mtx_unblock_lock_;
   public:
    integer_type &get_nwaiters_blocked() { return nwaiters_blocked_; }

    integer_type &get_nwaiters_gone() { return nwaiters_gone_; }

    integer_type &get_nwaiters_to_unblock() { return nwaiters_to_unblock_; }

    semaphore_type &get_sem_block_queue() { return sem_block_queue_; }

    semaphore_type &get_sem_block_lock() { return sem_block_lock_; }

    mutex_type &get_mtx_unblock_lock() { return mtx_unblock_lock_; }

    ConditionMembers()
      : nwaiters_blocked_(0)
        , nwaiters_gone_(0)
        , nwaiters_to_unblock_(0)
        , sem_block_queue_(0)
        , sem_block_lock_(1) {}

  };

  std::mutex                              mtx_;
  condition_8a_wrapper<ConditionMembers>  wrapper8a_;
  std::atomic<bool>                       signaled_;
  std::atomic<bool>                       manualReset_;

 public:
  MultiEvent(bool manualReset = true)
    : signaled_(false)
      , manualReset_(manualReset) {}

  void setManualReset(bool manualReset) {
    manualReset_.store(manualReset);
  }

  void signal() {
    std::unique_lock<std::mutex> lk(mtx_);
    signaled_.exchange(true);
    wrapper8a_.notify_all();
  }

  void wait() {
    if (!manualReset_.load() || !signaled_.load()) {
      condition_algorithm_8a<ConditionMembers>::scoped_lock<std::mutex> lk(mtx_);
      if (!signaled_.load()) {
        wrapper8a_.wait(lk);
      }
    }
  }

  bool try_wait(const size_t ms = 0) {
    if (!manualReset_.load() || !signaled_.load()) {
      condition_algorithm_8a<ConditionMembers>::scoped_lock<std::mutex> lk(mtx_);
      if (!signaled_.load()) {
        const Time::time_point tryUntil = Time::now() + Time::duration(ms * 1000000);
        return wrapper8a_.timed_wait(lk, tryUntil);
      }
    }
    return true;
  }

  void reset() {
    condition_algorithm_8a<ConditionMembers>::scoped_lock<std::mutex> lk(mtx_);
    signaled_.exchange(false);
  }
};

/**
 *  _______  _                            _   _____
 * |__   __|| |                          | | / ____|
 *    | |   | |__   _ __  ___   __ _   __| || |  __  _ __  ___   _   _  _ __
 *    | |   | '_ \ | '__|/ _ \ / _` | / _` || | |_ || '__|/ _ \ | | | || '_ \
 *    | |   | | | || |  |  __/| (_| || (_| || |__| || |  | (_) || |_| || |_) |
 *    |_|   |_| |_||_|   \___| \__,_| \__,_| \_____||_|   \___/  \__,_|| .__/
 *                                                                     | |
 *                                                                     |_|
 */
class ThreadGroup : public SharedObject<ThreadGroup>
{
  typedef std::shared_timed_mutex SharedMutex;
  typedef std::unique_lock<SharedMutex> WriteLocker;

  struct ReadLocker
  {
    SharedMutex *m_;

    ReadLocker(SharedMutex &m)
      : m_(&m) { if (m_) m_->lock_shared(); }

    ~ReadLocker() { if (m_) m_->unlock_shared(); }
  };

 public:


  /**
   *  __  __                                        _  _______  _                            _
   * |  \/  |                                      | ||__   __|| |                          | |
   * | \  / |  __ _  _ __    __ _   __ _   ___   __| |   | |   | |__   _ __  ___   __ _   __| |
   * | |\/| | / _` || '_ \  / _` | / _` | / _ \ / _` |   | |   | '_ \ | '__|/ _ \ / _` | / _` |
   * | |  | || (_| || | | || (_| || (_| ||  __/| (_| |   | |   | | | || |  |  __/| (_| || (_| |
   * |_|  |_| \__,_||_| |_| \__,_| \__, | \___| \__,_|   |_|   |_| |_||_|   \___| \__,_| \__,_|
   *                                __/ |
   *                               |___/
   */
  class ManagedThread : public SharedObject<ManagedThread>
  {
    typedef std::thread Thread;

    std::string             name_;
    mutable SharedMutex     csThread_;
    std::atomic<Thread *>   thread_;
    MultiEvent::SharedPtr   evReady_;
    MultiEvent::SharedPtr   evStart_;
    ThreadGroup *           owner_;
    std::atomic<bool>       shutdown_requested_;
    bool                    autoRemove_;

   protected:
    static void startHere(ManagedThread::SharedPtr pThis) {
      pThis->evReady_->signal();
      pThis->run_thread(pThis->thread_);
#if defined(AWSDL_DEBUG_LOG_THREADS) && !defined(NDEBUG)
      LOG(INFO) << "Thread " << pThis->name_ << " exiting";
#endif
      if (pThis->autoRemove_) {
        pThis->owner_->remove_thread(pThis);
      }
    }

   protected:
    virtual void run_thread(Thread *thd) {};

   public:
    ManagedThread(const char *threadName, ThreadGroup *owner, std::thread *thrd = NULL)
      : name_(threadName)
        , thread_(thrd)
        , evReady_(MultiEvent::create())
        , evStart_(MultiEvent::create())
        , owner_(owner)
        , shutdown_requested_(false)
        , autoRemove_(false) {
      CHECK_NOTNULL(owner);
    }

    virtual ~ManagedThread() {
#if defined(AWSDL_DEBUG_LOG_THREADS) && !defined(NDEBUG)
      const std::string name = getName();
            LOG(INFO) << "ManagedThread::~ManagedThread( " << name << " )";
#endif
      if (!is_current_thread()) {
        request_shutdown();
        join();
      }
      WriteLocker guard(csThread_);
      if (thread_) {
        Thread *thrd = thread_;
        thread_ = NULL;
        delete thrd;
      }
    }

    const char *getName() const {
      return name_.c_str();
    }

    static bool launch(ManagedThread::SharedPtr pThis, bool autoRemove = false) {
      WriteLocker guard(pThis->csThread_);
      CHECK_EQ(!pThis->thread_, true);
      CHECK_NOTNULL(pThis->owner_);
      pThis->autoRemove_ = autoRemove;
      pThis->thread_ = new std::thread(startHere, pThis);
      pThis->owner_->add_thread(pThis);
      pThis->evReady_->wait();
      pThis->evStart_->signal();
      return pThis->thread_ != NULL;
    }

    bool is_current_thread() {
      ReadLocker guard(csThread_);
      return thread_.load() ? (thread_.load()->get_id() == std::this_thread::get_id()) : false;
    }

    virtual void request_shutdown() {
      shutdown_requested_ = true;
    }

    virtual bool is_shutdown_requested() const {
      return shutdown_requested_.load();
    }

    bool joinable() const {
      ReadLocker guard(csThread_);
      if (thread_) {
        CHECK_EQ(!autoRemove_, true);  // TODO: If we need this, join needs to be checked by searching the group or exit event.
        return thread_.load()->joinable();
      }
      return false;
    }

    void join() {
#if defined(AWSDL_DEBUG_LOG_THREADS) && !defined(NDEBUG)
      const std::string name = getName();
      LOG(INFO) << "join() on " << name << " ( " << thread_.load()->get_id() << " )";
#endif
      ReadLocker guard(csThread_);
      // should be careful calling (or any function externally) this when in
      // auto-remove mode
      if(thread_ && thread_.load()->get_id() != std::thread::id()) {
        std::thread::id someId;

        CHECK_EQ(!autoRemove_,
                 true);  // TODO: If we need this, join needs to be checked by searching the group or exit event.
        CHECK_NOTNULL(thread_.load());
        if (thread_.load()->joinable()) {
          thread_.load()->join();
        } else {
          LOG(WARNING) << "Thread " << name_ << " ( " << thread_.load()->get_id() << " ) not joinable";
        }
      }
    }

    std::thread::id get_id() const {
      ReadLocker guard(csThread_);
      return thread_.load()->get_id();
    }
  };

  mutable SharedMutex                 m_;
  std::set<ManagedThread::SharedPtr>  threads_;

 public:
  ThreadGroup();

  virtual ~ThreadGroup();

  bool is_this_thread_in();

  bool is_thread_in(ManagedThread::SharedPtr thrd);

  void add_thread(ManagedThread::SharedPtr thrd);

  void remove_thread(ManagedThread::SharedPtr thrd);

  void join_all();

  void request_shutdown_all();

  size_t size() const;

  template<typename F, class T>
  inline ManagedThread::SharedPtr create_thread(const char *threadName, F threadfunc, T data);

  template<typename F>
  inline ManagedThread::SharedPtr create_thread(const char *threadName, F threadfunc);
};

typedef ThreadGroup::ManagedThread ManagedThread;

inline ThreadGroup::ThreadGroup()
//  :   threads_(std::less<ManagedThread::SharedPtr>())
{}

inline ThreadGroup::~ThreadGroup() {}

//
// ThreadGroup implementation
//
inline bool ThreadGroup::is_this_thread_in() {
  std::thread::id id = std::this_thread::get_id();
  ReadLocker guard(m_);
  for (auto it = threads_.begin(), end = threads_.end(); it != end; ++it) {
    ManagedThread::SharedPtr thrd = *it;
    if (thrd->get_id() == id)
      return true;
  }
  return false;
}

inline bool ThreadGroup::is_thread_in(ManagedThread::SharedPtr thrd) {
  if (thrd) {
    std::thread::id id = thrd->get_id();
    ReadLocker guard(m_);
    for (auto it = threads_.begin(), end = threads_.end(); it != end; ++it) {
      ManagedThread::SharedPtr thrd = *it;
      if (thrd->get_id() == id)
        return true;
    }
    return false;
  } else {
    return false;
  }
}

inline void ThreadGroup::add_thread(ManagedThread::SharedPtr thrd) {
  if (thrd) {
    WriteLocker guard(m_);
    threads_.insert(thrd);
  }
}

inline void ThreadGroup::remove_thread(ManagedThread::SharedPtr thrd) {
  WriteLocker guard(m_);
  threads_.erase(thrd);
}

inline void ThreadGroup::join_all() {
  CHECK_EQ(!is_this_thread_in(), true);

  ManagedThread::SharedPtr thrd;
  do {
    {
      ReadLocker guard(m_);
      if (!threads_.empty()) {
        thrd = *threads_.begin();
      } else {
        thrd = ManagedThread::SharedPtr(NULL);
      }
    }
    if (thrd) {
      if (thrd->joinable()) {
        thrd->join();
      }
      WriteLocker guard(m_);
      threads_.erase(thrd);
    }
  } while (thrd);
}

inline void ThreadGroup::request_shutdown_all() {
  ReadLocker guard(m_);
  for (auto it = threads_.begin(), end = threads_.end(); it != end; ++it) {
    ManagedThread::SharedPtr thrd = *it;
    thrd->request_shutdown();
  }
}

inline size_t ThreadGroup::size() const {
  ReadLocker guard(m_);
  return threads_.size();
}

template<typename F, class T>
inline ManagedThread::SharedPtr ThreadGroup::create_thread(const char *threadName, F threadfunc, T data) {
  std::thread *thrd = new std::thread(threadfunc, data);
  ManagedThread::SharedPtr newThread = ManagedThread::create(threadName, this, thrd);
  add_thread(newThread);
  return newThread;
}

template<typename F>
inline ManagedThread::SharedPtr ThreadGroup::create_thread(const char *threadName, F threadfunc) {
  std::thread *thrd = new std::thread(threadfunc);
  ManagedThread::SharedPtr newThread = ManagedThread::create(threadName, this, thrd);
  add_thread(newThread);
  return newThread;
}

} // namespace mf

#endif // _AWSDL_SYNC_H