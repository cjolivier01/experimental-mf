#ifndef _FASTMF_MF_H
#define _FASTMF_MF_H

#include <thread>
#include <unordered_set>
#include "model.h"
#include "blocks.pb.h"
#include "filter_util.h"

namespace mf
{

class SgdReadFilter : public mf::ObjectPool<std::vector<char> >,
                      public mf::StatusStack,
                      public tbb::filter
{

 public:
  SgdReadFilter(MF &mf, FILE *fr, const mf::Blocks &blocks_test)
    : mf::ObjectPool<std::vector<char> >(mf.data_in_fly_ * 10)
      , tbb::filter(serial_in_order)
      , mf_(mf)
      , blocks_test_(blocks_test)
      , fr_(fr)
      , iter_(1)
      , pass_(0) {
  }

  ~SgdReadFilter() {
  }

  void *operator()(void *) {
    if(!pass_++) {
      s_ = Time::now();
    }
    // TODO: Performance (if realloc of vector becomes a bottleneck): find best size in pool
    std::vector<char> *pbuffer = allocateObject();
    if (pbuffer) {
      std::vector<char> &buffer = *pbuffer;

      if (fread(&isize_, 1, sizeof(isize_), fr_)) {
        buffer.resize(isize_);
        if (fread((char *) buffer.data(), 1, isize_, fr_) == isize_) {
          return pbuffer;
        }
        addStatus(IO_ERROR, "Error reading input data object");
        freeObject(pbuffer);
      } else {
        int nn;
        printf("iter#%d\t%f\ttRMSE=%f\n",
               iter_,
               std::chrono::duration<float>(Time::now() - s_).count(),
               sqrt(mf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
        );
        pass_ = 0;
        //printf("iter#%d\t%f\n", iter_, std::chrono::duration<float>(e-s).count());

        // Check if we reached the desired number of iterations
        if (iter_ != mf_.iter_) {
          mf_.seteta(++iter_);
          // If IO problem, the last fread here will catch it
          fseek(fr_, 0, SEEK_SET);
          fread(&isize_, 1, sizeof(isize_), fr_);
          buffer.resize(isize_);
          if (fread(buffer.data(), 1, isize_, fr_) == isize_) {
            return pbuffer;
          }
          addStatus(IO_ERROR, "Error reading input data object");
        }
        freeObject(pbuffer);
      }
    }
    return NULL;
  }

 private:
  MF&               mf_;
  const mf::Blocks& blocks_test_;
  FILE *            fr_;
  uint32            isize_;
  int               iter_;
  std::atomic<int>              pass_;
  std::chrono::time_point<Time> s_;
};

class ParseFilter : public mf::ObjectPool<mf::Block>,
                    public mf::StatusStack,
                    public tbb::filter
{

 public:
  ParseFilter(size_t fly, mf::ObjectPool<std::vector<char> > &free_buffer_pool)
    : mf::ObjectPool<mf::Block>(fly * 10)
      , tbb::filter(parallel)
      , free_buffer_pool_(free_buffer_pool) {
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
    : tbb::filter(parallel)
      , mf_(model)
      , free_block_pool_(free_block_pool) {
  }

  void *operator()(void *block) {
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
      theta = (float *) __builtin_assume_aligned(mf_.theta_[uid], CACHE_LINE_SIZE);
      const int size = user.record_size();
      for (j = 0; j < size - mf_.prefetch_stride_; j++) {
#ifdef FETCH
        const mf::User_Record& rec_fetch = user.record(j+mf_.prefetch_stride_);
        const int vid_fetch = rec_fetch.vid();
        prefetch_range((char*)(mf_.phi_[vid_fetch]), pad*sizeof(float));
#endif
        memset(q, 0.0, sizeof(float) * mf_.dim_);
        const mf::User_Record &rec = user.record(j);
        vid = rec.vid();
        phi = (float *) __builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
        rating = rec.rating();
        error = float(rating)
                - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                - mf_.bu_[uid] - mf_.bv_[vid] - mf_.gb_;
        error = mf_.eta_ * error;
        cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
        cblas_saxpy(mf_.dim_, lameta - 1.0, theta, 1, theta, 1);
        cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
        cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
        cblas_scopy(mf_.dim_, q, 1, phi, 1);
        mf_.bu_[uid] = lameta * mf_.bu_[uid] + error;
        mf_.bv_[vid] = lameta * mf_.bv_[vid] + error;
      }
      //prefetch_range((char*)(mf_.theta_[bk->user(i+1).uid()]), pad*sizeof(float));
      for (; j < size; j++) {
        memset(q, 0.0, sizeof(float) * mf_.dim_);
        const mf::User_Record &rec = user.record(j);
        vid = rec.vid();
        phi = (float *) __builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
        rating = rec.rating();
        error = float(rating)
                - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                - mf_.bu_[uid] - mf_.bv_[vid] - mf_.gb_;
        error = mf_.eta_ * error;
        cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
        cblas_saxpy(mf_.dim_, lameta - 1.0, theta, 1, theta, 1);
        cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
        cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
        cblas_scopy(mf_.dim_, q, 1, phi, 1);
        mf_.bu_[uid] = lameta * mf_.bu_[uid] + error;
        mf_.bv_[vid] = lameta * mf_.bv_[vid] + error;
      }
    }
    // Return block to queue
    mf::Block *b = (mf::Block *) block;
    free_block_pool_.freeObject(b);
    return NULL;
  }

 private:
  const MF&                   mf_;
  mf::ObjectPool<mf::Block>&  free_block_pool_;
};

} // namespace mf

#endif //_FASTMF_MF_H
