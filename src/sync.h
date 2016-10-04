#ifndef _FASTMF_SYNC_H
#define _FASTMF_SYNC_H

#include "condition_algorithm_8a.h"
#include <thread>
#include <chrono>
#include <shared_mutex>
#include <semaphore.h>
#include <unordered_set>
#include <semaphore.h>
#include <dmlc/logging.h>

namespace mf
{

template<class ObjectType>
class SharedObject
{
 public:

  typedef std::shared_ptr<ObjectType> SharedPtr;
  typedef std::weak_ptr<ObjectType> WeakPtr;

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
 * semaphore
 * Simple semaphore which does not make any attempt to
 * maintain lock/release order
 */

class semaphore
{
  sem_t sem_;
 public:
  semaphore(int initialCount) {
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

/**
 * MultiEvent
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
    integer_type m_nwaiters_blocked;
    integer_type m_nwaiters_gone;
    integer_type m_nwaiters_to_unblock;
    semaphore_type m_sem_block_queue;
    semaphore_type m_sem_block_lock;
    mutex_type m_mtx_unblock_lock;
   public:
    integer_type &get_nwaiters_blocked() { return m_nwaiters_blocked; }

    integer_type &get_nwaiters_gone() { return m_nwaiters_gone; }

    integer_type &get_nwaiters_to_unblock() { return m_nwaiters_to_unblock; }

    semaphore_type &get_sem_block_queue() { return m_sem_block_queue; }

    semaphore_type &get_sem_block_lock() { return m_sem_block_lock; }

    mutex_type &get_mtx_unblock_lock() { return m_mtx_unblock_lock; }

    ConditionMembers()
      : m_nwaiters_blocked(0)
        , m_nwaiters_gone(0)
        , m_nwaiters_to_unblock(0)
        , m_sem_block_queue(0)
        , m_sem_block_lock(1) {}

  };

  std::mutex m_mtx;
  condition_8a_wrapper<ConditionMembers> m_wrapper8a;
  std::atomic<bool> m_signaled;
  std::atomic<bool> m_manualReset;

 public:
  MultiEvent(bool manualReset = true)
    : m_signaled(false)
      , m_manualReset(manualReset) {}

  void setManualReset(bool manualReset) {
    m_manualReset.store(manualReset);
  }

  void signal() {
    std::unique_lock<std::mutex> lk(m_mtx);
    m_signaled.exchange(true);
    m_wrapper8a.notify_all();
  }

  void wait() {
    if (!m_manualReset.load() || !m_signaled.load()) {
      condition_algorithm_8a<ConditionMembers>::scoped_lock<std::mutex> lk(m_mtx);
      if (!m_signaled.load()) {
        m_wrapper8a.wait(lk);
      }
    }
  }

  bool try_wait(const size_t ms = 0) {
    if (!m_manualReset.load() || !m_signaled.load()) {
      condition_algorithm_8a<ConditionMembers>::scoped_lock<std::mutex> lk(m_mtx);
      if (!m_signaled.load()) {
        const Time::time_point tryUntil = Time::now() + Time::duration(ms * 1000000);
        return m_wrapper8a.timed_wait(lk, tryUntil);
      }
    }
    return true;
  }

  void reset() {
    condition_algorithm_8a<ConditionMembers>::scoped_lock<std::mutex> lk(m_mtx);
    m_signaled.exchange(false);
  }
};

class ThreadGroup : public SharedObject<ThreadGroup>
{
  typedef std::shared_timed_mutex SharedMutex;
  typedef std::unique_lock<SharedMutex> WriteLocker;

  struct ReadLocker
  {
    SharedMutex *m_p;

    ReadLocker(SharedMutex &m)
      : m_p(&m) { if (m_p) m_p->lock_shared(); }

    ~ReadLocker() { if (m_p) m_p->unlock_shared(); }
  };

 public:
  class ManagedThread : public SharedObject<ManagedThread>
  {
    typedef std::thread Thread;

    std::string m_name;
    mutable SharedMutex m_csThread;
    std::atomic<Thread *> m_thread;
    MultiEvent::SharedPtr m_evReady;
    MultiEvent::SharedPtr m_evStart;
    ThreadGroup *m_owner;
    std::atomic<bool> m_shutdown_requested;
    bool m_autoRemove;

   protected:
    static void startHere(ManagedThread::SharedPtr pThis) {
      pThis->m_evReady->signal();
      pThis->run_thread(pThis->m_thread);
      if (pThis->m_autoRemove) {
        pThis->m_owner->remove_thread(pThis);
      }
    }

   protected:
    virtual void run_thread(Thread *thd) {};

   public:
    ManagedThread(const char *threadName, ThreadGroup *owner, std::thread *thrd = NULL)
      : m_name(threadName)
        , m_thread(thrd)
        , m_evReady(MultiEvent::create())
        , m_evStart(MultiEvent::create())
        , m_owner(owner)
        , m_shutdown_requested(false)
        , m_autoRemove(false) {
      CHECK_NOTNULL(owner);
    }

    virtual ~ManagedThread() {
#ifdef DEBUG
      const std::string name = getName();
            GLOG(Grover::Util::GTrace::Sev::NOTI, "ManagedThread::~ManagedThread( %s )\n", name.c_str());
#endif
      if (!is_current_thread()) {
        request_shutdown();
        join();
      }
      WriteLocker guard(m_csThread);
      if (m_thread) {
        Thread *thrd = m_thread;
        m_thread = NULL;
        delete thrd;
      }
    }

    const char *getName() const {
      return m_name.c_str();
    }

    static bool launch(ManagedThread::SharedPtr pThis, bool autoRemove = false) {
      WriteLocker guard(pThis->m_csThread);
      CHECK_EQ(!pThis->m_thread, true);
      CHECK_NOTNULL(pThis->m_owner);
      pThis->m_autoRemove = autoRemove;
      pThis->m_thread = new std::thread(startHere, pThis);
      pThis->m_owner->add_thread(pThis);
      pThis->m_evReady->wait();
      pThis->m_evStart->signal();
      return pThis->m_thread != NULL;
    }

    bool is_current_thread() {
      ReadLocker guard(m_csThread);
      return m_thread.load() ? (m_thread.load()->get_id() == std::this_thread::get_id()) : false;
    }

    virtual void request_shutdown() {
      m_shutdown_requested = true;
    }

    virtual bool is_shutdown_requested() const {
      return m_shutdown_requested.load();
    }

    bool joinable() const {
      ReadLocker guard(m_csThread);
      if (m_thread) {
        CHECK_EQ(!m_autoRemove, true);  // TODO: If we need this, join needs to be checked by searching the group or exit event.
        return m_thread.load()->joinable();
      }
      return false;
    }

    void join() {
#ifdef DEBUG
      const std::string name = getName();
            GLOG(Grover::Util::GTrace::Sev::NOTI, "join() on %s\n", name.c_str());
#endif
      ReadLocker guard(m_csThread);
      // should be careful calling (or any function externally) this when in
      // auto-remove mode
      if (m_thread) {
        CHECK_EQ(!m_autoRemove, true);  // TODO: If we need this, join needs to be checked by searching the group or exit event.
        m_thread.load()->join();
      }
    }

    std::thread::id get_id() const {
      ReadLocker guard(m_csThread);
      return m_thread.load()->get_id();
    }
  };

 private:
  mutable SharedMutex m;
  std::unordered_set<ManagedThread::SharedPtr> m_threads;

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
//  :   m_threads(std::less<ManagedThread::SharedPtr>())
{}

inline ThreadGroup::~ThreadGroup() {}

//
// ThreadGroup implementation
//
inline bool ThreadGroup::is_this_thread_in() {
  std::thread::id id = std::this_thread::get_id();
  ReadLocker guard(m);
  for (auto it = m_threads.begin(), end = m_threads.end(); it != end; ++it) {
    ManagedThread::SharedPtr thrd = *it;
    if (thrd->get_id() == id)
      return true;
  }
  return false;
}

inline bool ThreadGroup::is_thread_in(ManagedThread::SharedPtr thrd) {
  if (thrd) {
    std::thread::id id = thrd->get_id();
    ReadLocker guard(m);
    for (auto it = m_threads.begin(), end = m_threads.end(); it != end; ++it) {
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
    WriteLocker guard(m);
    m_threads.insert(thrd);
  }
}

inline void ThreadGroup::remove_thread(ManagedThread::SharedPtr thrd) {
  WriteLocker guard(m);
  m_threads.erase(thrd);
}

inline void ThreadGroup::join_all() {
  CHECK_EQ(!is_this_thread_in(), true);

  ManagedThread::SharedPtr thrd;
  do {
    {
      ReadLocker guard(m);
      if (!m_threads.empty()) {
        thrd = *m_threads.begin();
      } else {
        thrd = ManagedThread::SharedPtr(NULL);
      }
    }
    if (thrd) {
      if (thrd->joinable()) {
        thrd->join();
      }
      WriteLocker guard(m);
      m_threads.erase(thrd);
    }
  } while (thrd);
}

inline void ThreadGroup::request_shutdown_all() {
  ReadLocker guard(m);
  for (auto it = m_threads.begin(), end = m_threads.end(); it != end; ++it) {
    ManagedThread::SharedPtr thrd = *it;
    thrd->request_shutdown();
  }
}

inline size_t ThreadGroup::size() const {
  ReadLocker guard(m);
  return m_threads.size();
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

#endif //_FASTMF_SYNC_H