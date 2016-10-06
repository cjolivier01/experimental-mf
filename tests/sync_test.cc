#include <gtest/gtest.h>
#include "sync.h"
#include "test_util.h"

using namespace mf;

static STestThreadData::SharedPtr sync_test_top(const size_t count)
{
  STestThreadData::SharedPtr data = STestThreadData::create();
  data->m_p = TestSynchronizer::create();
  data->m_p->setExpectedCount(count);
  return data;
}

static void sync_test_bottom(STestThreadData *data)
{
  //LOG(INFO) << "master starting all (no threads starting before here)";
  ASSERT_EQ(data->m_p->getStartedCount(), 0U);
  ASSERT_EQ(data->m_p->getDoneCount(), 0U);
  data->m_p->startAll();
  //LOG(INFO) << "master joining all";
  data->m_p->join_all();
  const size_t count = data->m_p->getExpectedCount();
  ASSERT_EQ(data->m_p->getStartedCount(), count);
  ASSERT_EQ(data->m_p->getDoneCount(), count);
  ASSERT_EQ(data->m_p->getReadyCount(), count);
  //LOG(INFO) << "\nmaster done\n";
}

TEST(Sync, TestSyncObjectsMT)
{
  for(int pass = 0; pass < 50; ++pass) {
    const size_t COUNT = randomRange(2, 1000);

    LOG(INFO) << "Run count: " << COUNT;

    STestThreadData::SharedPtr data = sync_test_top(COUNT);
    data->m_delayHi = 1000;

    ThreadGroup threads;

    for (std::size_t i = 0; i < COUNT; ++i) {
      std::string name = "sync_test ";
      name += std::to_string(pass);
      name += ":";
      name += std::to_string(i);
      threads.create_thread(name.c_str(), [data]() {
        ASSERT_TRUE(data != nullptr);
        ASSERT_TRUE(data->m_p != nullptr);
        usleep(randomRange(data->m_delayLo, data->m_delayHi));
        data->m_p->ready();
        data->m_p->exit();
      });
    }
    usleep(randomRange(0, 10));
    sync_test_bottom(data.get());

    threads.join_all();
  }
}
