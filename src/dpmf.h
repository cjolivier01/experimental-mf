#ifndef _FASTMF_DPMF_H
#define _FASTMF_DPMF_H

#include "model.h"
#include "filter_util.h"

namespace mf
{

class SgldReadFilter : public mf::ObjectPool< std::vector<char> >,
                       public mf::StatusStack,
                       public tbb::filter
{
 public:

  SgldReadFilter(DPMF &dpmf, dmlc::SeekStream *fr)
    : mf::ObjectPool<std::vector<char> >(dpmf.data_in_fly_ * 10)
      , tbb::filter(serial_in_order)
      , dpmf_(dpmf)
      , fr_(fr)
      , stream_(std::unique_ptr<dmlc::istream>(new dmlc::istream(fr, STREAM_BUFFER_SIZE)))
      , pass_(0)  {
  }

  void *operator()(void *) {
    if(!pass_++) {
      s_ = Time::now();
    }
    if(!stream_->read((char *)&isize_, sizeof(isize_)).fail()) {
      std::vector<char> *pbuffer = allocateObject();
      if(pbuffer) {
        pbuffer->resize(isize_);
        if(!stream_->read(pbuffer->data(), isize_).fail()) {
          return pbuffer;
        }
        addStatus(IO_ERROR);
      } else {
        addStatus(POOL_ERROR);
      }
      freeObject(pbuffer);
    } else {
      if(stream_->eof()) {
        stream_.reset();
        fr_->Seek(0);
      }
      else {
        addStatus(IO_ERROR);
      }
    }
    return NULL;
  }

 public:
  DPMF &                          dpmf_;
  dmlc::SeekStream *              fr_;
  std::unique_ptr<dmlc::istream>  stream_;
  uint32                          isize_;
  std::atomic<int>                pass_;
  std::chrono::time_point<Time>   s_;
};


class SgldFilter : public mf::StatusStack,
                   public tbb::filter
{
 public:
  SgldFilter(DPMF &dpmf, mf::ObjectPool<mf::Block> &free_block_pool)
    : tbb::filter(serial_in_order /*parallel*/)
      , dpmf_(dpmf)
      , free_block_pool_(free_block_pool) {
  }

  void *operator()(void *block) {
    float q[dpmf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
    float p[dpmf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
    mf::Block *bk = (mf::Block *) block;
    const float eta = dpmf_.eta_;
    const float scal = eta * dpmf_.ntrain_ * dpmf_.bound_ * dpmf_.lambda_r_;
    for (int i = 0; i < bk->user_size(); i++) {
      const mf::User &user = bk->user(i);
      const int uid = user.uid();

      //std::cout << "User: " << uid << std::endl << std::flush;

      DCHECK_EQ(isFinite(dpmf_.user_array_[uid]), true);

      //const float userStartValue = dpmf_.user_array_[uid];

      const int size = user.record_size();
      int thetaind = dpmf_.uniform_int_(generator);
      int   phiind = dpmf_.uniform_int_(generator);

      for (int j = 0; j < size; j++) {
        memset(q, 0, sizeof(float) * dpmf_.dim_);
        const mf::User_Record &rec = user.record(j);
        const int vid = rec.vid();

        //std::cout << "\tVid: " << vid << std::endl << std::flush;

        DCHECK_EQ(isFinite(dpmf_.video_array_[vid]), true);
        const float rating = rec.rating();

        dpmf_.gmutex[vid].lock();
        const uint64 gc = dpmf_.gcount.fetch_add(1);
        const uint64 vc = gc - dpmf_.gcountv[vid].exchange(gc);
        dpmf_.gmutex[vid].unlock();
        const uint64 uc = gc - dpmf_.gcountu[uid];
        dpmf_.gcountu[uid] = gc;
        cblas_saxpy(dpmf_.dim_, sqrt(dpmf_.temp_ * eta * uc), dpmf_.noise_ + thetaind, 1, dpmf_.theta_[uid], 1);
        cblas_saxpy(dpmf_.dim_, sqrt(dpmf_.temp_ * eta * vc), dpmf_.noise_ + phiind, 1, dpmf_.phi_[vid], 1);
        dpmf_.user_array_[uid] += sqrt(dpmf_.temp_ * eta * uc) * dpmf_.noise_[thetaind + dpmf_.dim_];
        dpmf_.video_array_[vid] += sqrt(dpmf_.temp_ * eta * vc) * dpmf_.noise_[phiind + dpmf_.dim_];

        float error = rating
                - cblas_sdot(dpmf_.dim_, dpmf_.theta_[uid], 1, dpmf_.phi_[vid], 1)
                - dpmf_.user_array_[uid] - dpmf_.video_array_[vid] - dpmf_.gb_;

        //const float preError = error;
        DCHECK_EQ(isFinite(error), true);

        error = scal * error;

        DCHECK_EQ(isFinite(error), true);

        cblas_saxpy(dpmf_.dim_, error, dpmf_.theta_[uid], 1, q, 1);
        vsMul(dpmf_.dim_, dpmf_.lambda_u_, dpmf_.theta_[uid], p);
        cblas_saxpy(dpmf_.dim_, -eta * dpmf_.ur_[uid] * dpmf_.bound_, p, 1, dpmf_.theta_[uid], 1);
        cblas_saxpy(dpmf_.dim_, error, dpmf_.phi_[vid], 1, dpmf_.theta_[uid], 1);
        vsMul(dpmf_.dim_, dpmf_.lambda_v_, dpmf_.phi_[vid], p);
        cblas_saxpy(dpmf_.dim_, -eta * dpmf_.vr_[vid] * dpmf_.bound_, p, 1, dpmf_.phi_[vid], 1);
        cblas_saxpy(dpmf_.dim_, 1.0, q, 1, dpmf_.phi_[vid], 1);

        //const float vidStartValue = dpmf_.video_array_[vid];

        dpmf_.user_array_[uid] = (float)(1.0 - eta * dpmf_.lambda_ub_ * dpmf_.ur_[uid] * dpmf_.bound_) * dpmf_.user_array_[uid] + error;
        dpmf_.video_array_[vid] = (float)(1.0 - eta * dpmf_.lambda_vb_ * dpmf_.vr_[vid] * dpmf_.bound_) * dpmf_.video_array_[vid] + error;

        DCHECK_EQ(isFinite(dpmf_.user_array_[uid]), true);
        DCHECK_EQ(isFinite(dpmf_.video_array_[vid]), true);

        thetaind = thetaind + dpmf_.dim_ + 1;
        phiind = phiind + dpmf_.dim_ + 1;
      }
    }
    free_block_pool_.freeObject(bk);
    return NULL;
  }

 private:
  DPMF&                         dpmf_;
  mf::ObjectPool<mf::Block>&    free_block_pool_;
};

} // namespace mf

#endif //_FASTMF_DPMF_H