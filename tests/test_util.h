#ifndef AWSDL_TEST_UTIL_H
#define AWSDL_TEST_UTIL_H

#include <mutex>
#include "sync.h"

namespace awsdl
{

/**
 * Synchronize test runners
 */
class TestSynchronizer : public SharedObject<TestSynchronizer>
{
  typedef std::unique_lock<std::mutex> scope_lock;

  typedef std::atomic<std::size_t> integer_type;

  integer_type m_expectedCount;
  integer_type m_readyCount;
  integer_type m_startedCount;
  integer_type m_doneCount;

  MultiEvent::SharedPtr m_cond;
  MultiEvent::SharedPtr m_condStart;
  MultiEvent::SharedPtr m_condEnd;

 public:
  TestSynchronizer(std::size_t expectedCount = 0)
    : m_expectedCount(expectedCount)
      , m_readyCount(0)
      , m_startedCount(0)
      , m_doneCount(0)
      , m_cond(MultiEvent::create())
      , m_condStart(MultiEvent::create())
      , m_condEnd(MultiEvent::create()) {}

  /**
   * Set the number of test runners
   */
  void setExpectedCount(size_t count) {
    m_expectedCount = count;
  }

  void ready() {
    assert(m_readyCount.load() < m_expectedCount);
    if (++m_readyCount == m_expectedCount) {
      m_condStart->signal();
    }
    m_cond->wait();
    ++m_startedCount;
  }

  void exit() {
    assert(m_doneCount.load() < m_expectedCount);
    if (++m_doneCount == m_expectedCount) {
      m_condEnd->signal();
    }
  }

  // controller calls
  void startAll() {
    if (m_expectedCount.load() > 0) {
      m_condStart->wait();
    }
    m_cond->signal();
  }

  void join_all() {
    if (m_expectedCount.load() > 0) {
      m_condEnd->wait();
    }
  }

  const integer_type &getExpectedCount() const { return m_expectedCount; }

  const integer_type &getReadyCount() const { return m_readyCount; }

  const integer_type &getDoneCount() const { return m_doneCount; }

  const integer_type &getStartedCount() const { return m_startedCount; }

};

struct STestThreadData : public SharedObject<STestThreadData>
{
  TestSynchronizer::SharedPtr m_p;
  unsigned m_delayLo;
  unsigned m_delayHi;

  inline STestThreadData()
    : m_delayLo(0)
      , m_delayHi(0) {}
};

template<class T>
inline T randomRange(const T lo, const T hi) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(lo, hi);
  return dis(gen);
}

} // namespace awsdl

#endif //AWSDL_TEST_UTIL_H