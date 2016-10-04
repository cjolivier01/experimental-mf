#ifndef _FASTMF_ADMF_H
#define _FASTMF_ADMF_H

#include "model.h"
#include "filter_util.h"
#include "binary_record_source_filter.h"

namespace mf
{

class AdRegReadFilter : public BinaryRecordSourceFilter
{
 public:
  AdRegReadFilter(AdaptRegMF &admf,
                  dmlc::SeekStream *fr,
                  const mf::Blocks &blocks_test,
                  awsdl::perf::TimingInstrument *timing)
    : BinaryRecordSourceFilter(admf.data_in_fly_ * 10U, fr, timing)
      , admf_(admf)
      , blocks_test_(blocks_test)
      , iter_(1)  {
  }

  bool onSourceStreamComplete() {
    int nn;
    printf("iter#%d\t%f\ttRMSE=%f\n",
           iter_,
           std::chrono::duration<float>(Time::now() - s_).count(),
           sqrt(admf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
    );
    if (iter_ != admf_.iter_) {
      admf_.seteta(++iter_);
      admf_.set_etareg(iter_);
      return true;
    }
    return false;
  }

 private:
  AdaptRegMF&                           admf_;
  const mf::Blocks&                     blocks_test_;
  int                                   iter_;
  static awsdl::perf::TimingInstrument  timing_;
};

class AdRegFilter : public mf::StatusStack,
                    public tbb::filter
{

 public:
  AdRegFilter(AdaptRegMF &model,
              mf::ObjectPool<mf::Block> &free_block_pool,
              awsdl::perf::TimingInstrument *timing)
    : tbb::filter(parallel)
      , admf_(model)
      , free_block_pool_(free_block_pool)
      , timing_(timing)
  {}

  void *operator()(void *block) {
    float q[admf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
    mf::Block *bk = (mf::Block *) block;
    const float eta = admf_.eta_;
    for (int i = 0; i < bk->user_size(); i++) {
      const mf::User &user = bk->user(i);
      const int uid = user.uid();
      DCHECK_EQ(isFinite(admf_.user_array_[uid]), true);
      const int size = user.record_size();
      for (int j = 0; j < size; j++) {
        memset(q, 0.0, sizeof(float) * admf_.dim_);
        const mf::User_Record &rec = user.record(j);
        const int vid = rec.vid();
        DCHECK_EQ(isFinite(admf_.video_array_[vid]), true);
        const float rating = rec.rating();
        cblas_scopy(admf_.dim_, admf_.theta_[uid], 1, admf_.theta_old_[uid], 1);
        cblas_scopy(admf_.dim_, admf_.phi_[vid], 1, admf_.phi_old_[vid], 1);
        const float pred = active(
          cblas_sdot(admf_.dim_, admf_.theta_[uid], 1,
                     admf_.phi_[vid], 1) + admf_.user_array_[uid] + admf_.video_array_[vid] +
          admf_.gb_, admf_.loss_);
        float error = cal_grad(rating, pred, admf_.loss_);
        error = eta * error;
        cblas_saxpy(admf_.dim_, error, admf_.theta_[uid], 1, q, 1);
        cblas_saxpy(admf_.dim_, -eta * admf_.lam_u_, admf_.theta_[uid], 1, admf_.theta_[uid], 1);
        cblas_saxpy(admf_.dim_, error, admf_.phi_[vid], 1, admf_.theta_[uid], 1);
        cblas_saxpy(admf_.dim_, 1.0f - eta * admf_.lam_v_, admf_.phi_[vid], 1, q, 1);
        cblas_scopy(admf_.dim_, q, 1, admf_.phi_[vid], 1);
        admf_.bu_old_[uid] = admf_.user_array_[uid];
        admf_.bv_old_[vid] = admf_.video_array_[vid];
        admf_.user_array_[uid] = (1.0f - eta * admf_.lam_bu_) * admf_.user_array_[uid] + error;
        admf_.video_array_[vid] = (1.0f - eta * admf_.lam_bv_) * admf_.video_array_[vid] + error;
        DCHECK_EQ(isFinite(admf_.user_array_[uid]), true);
        DCHECK_EQ(isFinite(admf_.video_array_[vid]), true);
      }
      const size_t ii = rand() % admf_.recsv_.size();
      admf_.updateReg(admf_.recsv_[ii].u_, admf_.recsv_[ii].v_, admf_.recsv_[ii].r_);
    }
    free_block_pool_.freeObject(bk);
    return NULL;
  }

 private:
  AdaptRegMF&                     admf_;
  mf::ObjectPool<mf::Block>&      free_block_pool_;
  awsdl::perf::TimingInstrument * timing_;
};

} // namespace mf

#endif //_FASTMF_ADMF_H