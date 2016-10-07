#ifndef _FASTMF_DPMF_H
#define _FASTMF_DPMF_H

#include "model.h"
#include "filter_util.h"
#include "binary_record_source_filter.h"

namespace mf
{

class SgldReadFilter : public BinaryRecordSourceFilter
{
 public:

  SgldReadFilter(DPMF &dpmf, dmlc::SeekStream *fr, const mf::Blocks& blocks_test, mf::perf::TimingInstrument *timing)
    : BinaryRecordSourceFilter(dpmf.data_in_fly_ * 10, fr, timing)
      , dpmf_(dpmf)
      , blocks_test_(blocks_test)
      , iter_(1)  {
  }

  bool onSourceStreamComplete() {
    if(flush()) {
      if (iter_ != dpmf_.iter_) {
        dpmf_.finish_round(blocks_test_, iter_++, s_);
        return true;
      }
    }
    return false;
  }

 public:
  DPMF &                          dpmf_;
  const mf::Blocks&               blocks_test_;
  int                             iter_;
};


class SgldFilter : public PipelineFilter<mf::Block>
{
 public:
  SgldFilter(DPMF &dpmf,
             mf::ObjectPool<mf::Block> &free_block_pool,
             mf::perf::TimingInstrument *timing)
    : PipelineFilter(serial_in_order /*parallel*/, &free_block_pool)
      , dpmf_(dpmf)
      , timing_(timing)
      , max_noise_(0.0f) {
    seen_users_ = new std::atomic<size_t>[dpmf_.nr_users_];
    memset(seen_users_, 0, dpmf_.nr_users_ * sizeof(seen_users_[0]));
    seen_vid_ = new std::atomic<size_t>[dpmf_.nr_videos_];
    memset(seen_vid_, 0, dpmf_.nr_videos_ * sizeof(seen_vid_[0]));
  }

  void checkNoise() {
#ifndef NDEBUG
    for (int i = 0; i < dpmf_.nr_users_; ++i) {
      for (int j = 0; j < dpmf_.dim_; ++j) {
        const float f = dpmf_.theta_[i][j];
        DCHECK_EQ(mf::isFinite(f), true);
      }
    }
    for (int i = 0; i < dpmf_.nr_videos_; ++i) {
      for (int j = 0; j < dpmf_.dim_; j++) {
        const float f = dpmf_.phi_[i][j];
        DCHECK_EQ(mf::isFinite(f), true);
      }
    }
#endif
  }

  void *execute(mf::Block *block) {
    if(block) {
      mf::perf::TimingItem inFunc(timing_, FILTER_STAGE_CALC, "FILTER_STAGE_CALC");

      float q[dpmf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
      float p[dpmf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
      mf::Block *bk = (mf::Block *) block;
      const float eta = dpmf_.learning_rate_;
      const float scal = eta * dpmf_.ntrain_ * dpmf_.bound_ * dpmf_.lambda_r_;
      DCHECK_LT(scal, 100.0); // should be a small number, like 0.01

      float pt2 = 0.0, pt3 = 0.0;

      for (int i = 0; i < bk->user_size(); i++) {
        const mf::User &user = bk->user(i);
        const int uid = user.uid();
        const size_t seenCount = ++seen_users_[uid];

        const float entryUser = dpmf_.user_array_[uid];

        DCHECK_EQ(isFinite(dpmf_.user_array_[uid]), true);
        DCHECK_LT(fabs(dpmf_.user_array_[uid]), 10.0f);

        int thetaind = dpmf_.uniform_int_(generator);
        int phiind = dpmf_.uniform_int_(generator);

        //checkNoise();
        int lastVid = 0;

        for (size_t j = 0, size = (size_t)user.record_size(); j < size; j++) {
          memset(q, 0, sizeof(float) * dpmf_.dim_);
          const mf::User_Record &rec = user.record(j);
          const int vid = rec.vid();

          const size_t seenVid = ++seen_vid_[vid];

          const float preUser0 = dpmf_.user_array_[uid];
          const float preVid0 = dpmf_.video_array_[vid];

          DCHECK_EQ(isFinite(dpmf_.video_array_[vid]), true);
          DCHECK_LT(fabs(dpmf_.video_array_[vid]), 10.0f);

          const float rating = rec.rating();

          dpmf_.gmutex[vid].lock();
          const uint64_t gc = dpmf_.gcount.fetch_add(1);
          DCHECK_GE(gc, dpmf_.gcountv[vid].load());
          const uint64_t vc = gc - dpmf_.gcountv[vid].exchange(gc);
          dpmf_.gmutex[vid].unlock();

          const uint64_t uc = gc - dpmf_.gcountu[uid];
          DCHECK_GE(gc, dpmf_.gcountu[uid]);
          dpmf_.gcountu[uid] = gc;

          cblas_saxpy(dpmf_.dim_,
                      (float) sqrt(dpmf_.sgld_temperature_ * eta * uc),
                      dpmf_.noise_ + thetaind,
                      1,
                      dpmf_.theta_[uid],
                      1);
          cblas_saxpy(dpmf_.dim_,
                      (float) sqrt(dpmf_.sgld_temperature_ * eta * vc),
                      dpmf_.noise_ + phiind,
                      1,
                      dpmf_.phi_[vid],
                      1);

          max_noise_ = std::max(max_noise_, (float)fabs(dpmf_.noise_[thetaind + dpmf_.dim_]));
          max_noise_ = std::max(max_noise_, (float)fabs(dpmf_.noise_[phiind + dpmf_.dim_]));

          DCHECK_LT(thetaind + dpmf_.dim_, dpmf_.noise_size_); // need to modulus it, i guess, or re-randomize it
          DCHECK_LT(phiind + dpmf_.dim_, dpmf_.noise_size_);

          const float ffu = (float)sqrt(dpmf_.sgld_temperature_ * eta * uc) * dpmf_.noise_[thetaind + dpmf_.dim_];
          const float ffv = (float)sqrt(dpmf_.sgld_temperature_ * eta * vc) * dpmf_.noise_[phiind + dpmf_.dim_];

          dpmf_.user_array_[uid] += sqrt(dpmf_.sgld_temperature_ * eta * uc) * dpmf_.noise_[thetaind + dpmf_.dim_];
          dpmf_.video_array_[vid] += sqrt(dpmf_.sgld_temperature_ * eta * vc) * dpmf_.noise_[phiind + dpmf_.dim_];

          DCHECK_EQ(isFinite(dpmf_.user_array_[uid]), true);
          DCHECK_EQ(isFinite(dpmf_.video_array_[vid]), true);
          DCHECK_LT(fabs(dpmf_.user_array_[uid]), 10.0f);
          DCHECK_LT(fabs(dpmf_.video_array_[vid]), 10.0f);

          pt2 = cblas_sdot(dpmf_.dim_, dpmf_.theta_[uid], 1, dpmf_.phi_[vid], 1);
          pt3 = dpmf_.user_array_[uid] - dpmf_.video_array_[vid] - dpmf_.global_bias_;

          float error = rating
                        - cblas_sdot(dpmf_.dim_, dpmf_.theta_[uid], 1, dpmf_.phi_[vid], 1)
                        - dpmf_.user_array_[uid] - dpmf_.video_array_[vid] - dpmf_.global_bias_;

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

          const float preUser1 = dpmf_.user_array_[uid];
          const float preVid1 = dpmf_.video_array_[vid];

          dpmf_.user_array_[uid] =
            (float) (1.0 - eta * dpmf_.lambda_ub_ * dpmf_.ur_[uid] * dpmf_.bound_) * dpmf_.user_array_[uid] + error;
          dpmf_.video_array_[vid] =
            (float) (1.0 - eta * dpmf_.lambda_vb_ * dpmf_.vr_[vid] * dpmf_.bound_) * dpmf_.video_array_[vid] + error;

          DCHECK_EQ(isFinite(dpmf_.user_array_[uid]), true);
          DCHECK_EQ(isFinite(dpmf_.video_array_[vid]), true);

          DCHECK_LT(fabs(dpmf_.user_array_[uid]), 10.0f);
          DCHECK_LT(fabs(dpmf_.video_array_[vid]), 10.0f);

          thetaind = thetaind + dpmf_.dim_ + 1;
          phiind = phiind + dpmf_.dim_ + 1;

          lastVid = vid;
        } // End of vid loop
        DCHECK_LT(fabs(dpmf_.user_array_[uid]), 10.0f);
      } // End of user loop
    }
    return NULL;
  }

 private:
  DPMF&                        dpmf_;
  mf::perf::TimingInstrument * timing_;
  // TODO: remove seen_users_, seen_vid_
  std::atomic<size_t>        * seen_users_, *seen_vid_;
  float max_noise_;
};

} // namespace mf

#endif //_FASTMF_DPMF_H