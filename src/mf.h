#ifndef _FASTMF_MF_H
#define _FASTMF_MF_H
#include <thread>
#include <iostream>

#include <thread>
#include <cinttypes>
#include <dmlc/io.h>
#include <blocks.pb.h>
#include "model.h"
#include "filter_util.h"
#include "perf.h"
#include "binary_record_source_filter.h"

namespace mf
{

class SgdReadFilter : public BinaryRecordSourceFilter
{

 public:
  SgdReadFilter(MF &mf, dmlc::SeekStream *fr, const mf::Blocks &blocks_test, mf::perf::TimingInstrument *timing)
  : BinaryRecordSourceFilter(mf.data_in_fly_ * 10, fr, timing)
    , mf_(mf)
    , iter_(1)
    , blocks_test_(blocks_test)
  {
  }

  bool onSourceStreamComplete() {
    if(flush()) {
      int nn;
      printf("iter#%d\t%f\ttRMSE=%f\n",
             iter_,
             std::chrono::duration<float>(Time::now() - s_).count(),
             sqrt(mf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
      );

      // Check if we reached the desired number of iterations
      if (iter_ != mf_.iter_) {
        mf_.set_learning_rate(++iter_);
        return true;
      }
    }
    return false;
  }

 private:

  MF&                             mf_;
  int                             iter_;
  const mf::Blocks&               blocks_test_;
};

class ParseFilter : public mf::ObjectPool<mf::Block>,
                    public PipelineFilter< std::vector<char> >
{

 public:
  ParseFilter(size_t fly,
              mf::ObjectPool< std::vector<char> > &free_buffer_pool,
              mf::perf::TimingInstrument *timing)
    : mf::ObjectPool<mf::Block>(fly * 10)
      , PipelineFilter(parallel, &free_buffer_pool)
      , timing_(timing)
  {
  }

  void *execute(std::vector<char> *p) {
    if (p) {
      mf::perf::TimingItem inFunc(timing_, FILTER_STAGE_PARSE, "FILTER_STAGE_PARSE");
      // Get next block object in free queue
      mf::Block *bk = allocateObject();
      CHECK_NOTNULL(bk);
      if (bk) {
        const bool ok = bk->ParseFromArray(p->data(), (int)p->size());
        // Return the buffer
        if (ok) {
          return bk;
        } else {
          // Error, so return the block
          addStatus(PARSE_ERROR, "Error parsing input data");
          freeObject(bk);
        }
      }
    }
    return NULL;
  }

 private:
  char                                pad[CACHE_LINE_SIZE];
  mf::perf::TimingInstrument *     timing_;
};

class SgdFilter : public PipelineFilter<mf::Block>
{

 public:
  SgdFilter(MF &model, mf::ObjectPool<mf::Block> &free_block_pool, mf::perf::TimingInstrument *timing)
    : PipelineFilter(parallel, &free_block_pool)
      , mf_(model)
      , timing_(timing) {
  }
  ~SgdFilter() {
  }

  void *execute(mf::Block *block) {
    if(block) {
      mf::perf::TimingItem inFunc(timing_, FILTER_STAGE_CALC, "FILTER_STAGE_CALC");
      float q[mf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
      padding(mf_.dim_);
      const mf::Block *bk = (mf::Block *) block;
      CHECK_NOTNULL(bk);
      const float lameta = 1.0f - mf_.learning_rate_ * mf_.lambda_;
      int vid, j, i;
      float error, rating, *theta, *phi;
      for (i = 0; i < bk->user_size(); ++i) {
        const mf::User &user = bk->user(i);
        const int uid = user.uid();
        theta = (float *) __builtin_assume_aligned(mf_.theta_[uid], CACHE_LINE_SIZE);
        const int size = user.record_size();
        for (j = 0; j < size - mf_.prefetch_stride_; j++) {
#ifdef FETCH
          const mf::User_Record& rec_fetch = user.record(j+mf_.prefetch_stride_);
          const int vid_fetch = rec_fetch.vid();
          prefetch_range((char*)(mf_.phi_[vid_fetch]), pad*sizeof(float));
#endif
          memset(q, 0, sizeof(float) * mf_.dim_);
          const mf::User_Record &rec = user.record(j);
          vid = rec.vid();
          phi = (float *) __builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
          rating = rec.rating();
          error = rating
                  - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                  - mf_.user_array_[uid] - mf_.video_array_[vid] - mf_.global_bias_;
          error = mf_.learning_rate_ * error;
          cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
          cblas_saxpy(mf_.dim_, lameta - 1.0f, theta, 1, theta, 1);
          cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
          cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
          cblas_scopy(mf_.dim_, q, 1, phi, 1);
          mf_.user_array_[uid] = lameta * mf_.user_array_[uid] + error;
          mf_.video_array_[vid] = lameta * mf_.video_array_[vid] + error;
        }
        //prefetch_range((char*)(mf_.theta_[bk->user(i+1).uid()]), pad*sizeof(float));
        for (; j < size; j++) {
          memset(q, 0.0, sizeof(float) * mf_.dim_);
          const mf::User_Record &rec = user.record(j);
          vid = rec.vid();
          phi = (float *) __builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
          rating = rec.rating();
          error = rating
                  - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                  - mf_.user_array_[uid] - mf_.video_array_[vid] - mf_.global_bias_;
          error = mf_.learning_rate_ * error;
          cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
          cblas_saxpy(mf_.dim_, lameta - 1.0f, theta, 1, theta, 1);
          cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
          cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
          cblas_scopy(mf_.dim_, q, 1, phi, 1);
          mf_.user_array_[uid] = lameta * mf_.user_array_[uid] + error;
          mf_.video_array_[vid] = lameta * mf_.video_array_[vid] + error;

          DCHECK_EQ(isFinite(mf_.user_array_[uid]), true);
          DCHECK_EQ(isFinite(mf_.video_array_[vid]), true);

          DCHECK_LT(mf_.user_array_[uid], 100.0f);
          DCHECK_LT(mf_.video_array_[vid], 100.0f);

        }
      }
    }
    return NULL;
  }

 private:
  const MF&                       mf_;
  mf::perf::TimingInstrument *    timing_;
};

} // namespace mf

#endif //_FASTMF_MF_H
