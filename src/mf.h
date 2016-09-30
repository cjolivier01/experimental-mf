#ifndef _MF_H
#define _MF_H
#include <thread>
#include <iostream>

#include <thread>
#include <cinttypes>
#include <dmlc/io.h>
#include <blocks.pb.h>
#include "model.h"
#include "filter_util.h"
#include "perf.h"

namespace mf
{

class SgdReadFilter : public mf::ObjectPool<std::vector<char> >,
                      public mf::StatusStack,
                      public tbb::filter
{

 public:
  SgdReadFilter(MF &mf, dmlc::SeekStream *fr, const mf::Blocks &blocks_test)
    : mf::ObjectPool<std::vector<char> >(mf.data_in_fly_ * 10)
      , tbb::filter(serial_in_order)
      , mf_(mf)
      , blocks_test_(blocks_test)
      , fr_(fr)
      , stream_(std::unique_ptr<dmlc::istream>(new dmlc::istream(fr , 1 << 20)))
      , iter_(1)
      , pass_(0)
  {
  }

  ~SgdReadFilter() {
  }

  void *operator()(void *) {
    std::chrono::time_point<Time> entryTime = Time::now();
    if(!pass_++) {
      s_ = Time::now();
      in_time_ = in_time_.zero();
    }
    //perf::TimingItem inFunc(timing_, TIMING_READ, "TIMING_READ");
    // TODO: Performance (if realloc of vector becomes a bottleneck): find best size in pool
    std::vector<char> *pbuffer = allocateObject();
    if (pbuffer) {
      if (!stream_->read((char *)&isize_, sizeof(isize_)).fail()) {
        pbuffer->resize(isize_);
        if (!stream_->read(pbuffer->data(), isize_).fail()) {
          in_time_ += Time::now() - entryTime;
          return pbuffer;
        }
        addStatus(IO_ERROR, "Error reading input data object");
        freeObject(pbuffer);
      } else {
        if(stream_->eof()) {
          int nn;
          printf("iter#%d\t%f\ttRMSE=%f\n",
                 iter_,
                 std::chrono::duration<float>(Time::now() - s_).count(),
                 sqrt(mf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
          );
          printf("iter#%d\t%f\ttRMSE=%f\n",
                 iter_,
                 MICRO2SF(timing_.getDuration(TIMING_READ)),
                 sqrt(mf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
          );
          printf("iter#%d\t%f\ttRMSE=%f\n",
                 iter_,
                 in_time_.count(),
                 sqrt(mf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
          );
          timing_.print(true);
          IF_CHECK_TIMING(printBlockedTime("SgdReadFilter buffer queue blocked", false));
          pass_ = 0;
          // Check if we reached the desired number of iterations
          if (iter_ != mf_.iter_) {
            mf_.seteta(++iter_);
            stream_.reset();
            fr_->Seek(0);
            stream_ = std::unique_ptr<dmlc::istream>(new dmlc::istream(fr_, 1 << 20));
            if(!stream_->read((char *) &isize_, sizeof(isize_)).fail()) {
              pbuffer->resize(isize_);
              if (!stream_->read(pbuffer->data(), isize_).fail()) {
                in_time_ += Time::now() - entryTime;
                return pbuffer;
              }
            }
            addStatus(IO_ERROR, "Error reading input data object");
          }
        } else {
          addStatus(IO_ERROR);
        }
      }
      freeObject(pbuffer);
    } else {
      addStatus(POOL_ERROR);
    }
    in_time_ += Time::now() - entryTime;
    return NULL;
  }

 private:

  MF&                             mf_;
  const mf::Blocks&               blocks_test_;
  dmlc::SeekStream *              fr_;
  std::unique_ptr<dmlc::istream>  stream_;
  uint32                          isize_;
  int                             iter_;
  std::atomic<unsigned long>      pass_;
  std::chrono::time_point<Time>   s_;
  std::chrono::duration<float>    in_time_;
 public:
  enum {
    TIMING_READ,
    TIMING_PARSE,
    TIMING_CALC
  };
  static perf::TimingInstrument   timing_;
};

class ParseFilter : public mf::ObjectPool<mf::Block>,
                    public mf::StatusStack,
                    public tbb::filter
{

 public:
  ParseFilter(size_t fly, mf::ObjectPool<std::vector<char> > &free_buffer_pool)
    : mf::ObjectPool<mf::Block>(fly * 10)
      , tbb::filter(parallel)
      , free_buffer_pool_(free_buffer_pool)
  {
  }

  void *operator()(void *chunk) {
    std::vector<char> *p = (std::vector<char> *) chunk;
    if (p) {
      // Get next block object in free queue
      mf::Block *bk = allocateObject();
      CHECK_NOTNULL(bk);
      if (bk) {
        const bool ok = bk->ParseFromArray(p->data(), p->size());
        // Return the buffer
        free_buffer_pool_.freeObject(p);
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
  mf::ObjectPool<std::vector<char> > &free_buffer_pool_;
};

class SgdFilter : public mf::StatusStack,
                  public tbb::filter
{

 public:
  SgdFilter(MF &model, mf::ObjectPool<mf::Block> &free_block_pool)
    : tbb::filter(/*serial_in_order*/ parallel)
      , mf_(model)
      , free_block_pool_(free_block_pool) {
  }

  void *operator()(void *block) {
    //perf::TimingItem inFunc(SgdReadFilter::timing_, SgdReadFilter::TIMING_CALC, "TIMING_CALC");
    float q[mf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
    padding(mf_.dim_);
    const mf::Block *bk = (mf::Block *) block;
    CHECK_NOTNULL(bk);
    const float lameta = 1.0 - mf_.eta_ * mf_.lambda_;
    int vid, j, i;
    float error, rating, *theta, *phi;
    for (i = 0; i < bk->user_size(); ++i) {
      const mf::User &user = bk->user(i);
      const int uid = user.uid();
      //perf::DebugCheckUsing<int> chkUser(uid, using_user_);
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
                - mf_.user_array_[uid] - mf_.video_array_[vid] - mf_.gb_;
        error = mf_.eta_ * error;
        cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
        cblas_saxpy(mf_.dim_, lameta - 1.0, theta, 1, theta, 1);
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
        //perf::DebugCheckUsing<int> chkVid(vid, using_vid_);
        phi = (float *) __builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
        rating = rec.rating();
        error = rating
                - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                - mf_.user_array_[uid] - mf_.video_array_[vid] - mf_.gb_;
        error = mf_.eta_ * error;
        cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
        cblas_saxpy(mf_.dim_, lameta - 1.0, theta, 1, theta, 1);
        cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
        cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
        cblas_scopy(mf_.dim_, q, 1, phi, 1);
        mf_.user_array_[uid] = lameta * mf_.user_array_[uid] + error;
        mf_.video_array_[vid] = lameta * mf_.video_array_[vid] + error;
      }
    }
    // Return block to queue
    mf::Block *b = (mf::Block *)block;
    free_block_pool_.freeObject(b);
    return NULL;
  }

 private:
  const MF&                   mf_;
  mf::ObjectPool<mf::Block>&  free_block_pool_;
  //perf::DebugUsing<int>       using_user_, using_vid_;
};

} // namespace mf

#endif //_FASTMF_MF_H
