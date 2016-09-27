#ifndef _FASTMF_ADMF_H
#define _FASTMF_ADMF_H

#include "model.h"
#include "filter_util.h"

namespace mf
{

class AdRegReadFilter : public mf::ObjectPool< std::vector<char> >,
                        public mf::StatusStack,
                        public tbb::filter
{
 public:
  AdRegReadFilter(AdaptRegMF &admf, dmlc::SeekStream *fr, const mf::Blocks &blocks_test)
    : mf::ObjectPool<std::vector<char> >(admf_.data_in_fly_ * 10)
      , tbb::filter(serial_in_order)
      , admf_(admf)
      , blocks_test_(blocks_test)
      , fr_(fr)
      , stream_(std::unique_ptr<dmlc::istream>(new dmlc::istream(fr)))
      , iter_(1)
      , pass_(0) {
  }

  void *operator()(void *) {
    std::vector<char> *pbuffer = allocateObject();
    if(pbuffer) {
      if (!stream_->read((char *)&isize_, sizeof(isize_)).fail()) {
        pbuffer->resize(isize_);
        if(!stream_->read(pbuffer->data(), isize_).fail()) {
          return pbuffer;
        }
        addStatus(IO_ERROR);
      } else {
        if(stream_->eof()) {
          int nn;
          printf("iter#%d\t%f\ttRMSE=%f\n",
                 iter_,
                 std::chrono::duration<float>(Time::now() - s_).count(),
                 sqrt(admf_.calc_mse(blocks_test_, nn) * 1.0 / nn)
          );
          pass_ = 0;
          if (iter_ != admf_.iter_) {
            admf_.seteta(++iter_);
            admf_.set_etareg(iter_);
            stream_.reset();
            fr_->Seek(0);
            stream_ = std::unique_ptr<dmlc::istream>(new dmlc::istream(fr_));
            if(!stream_->read((char *)&isize_, sizeof(isize_)).fail()) {
              pbuffer->resize(isize_);
              if (!stream_->read(pbuffer->data(), isize_).fail()) {
                return pbuffer;
              }
            }
            addStatus(IO_ERROR);
          }
        } else {
          addStatus(IO_ERROR);
        }
        freeObject(pbuffer);
      }
    } else {
      addStatus(POOL_ERROR);
    }
    return NULL;
  }

 private:
  AdaptRegMF&                     admf_;
  const mf::Blocks&               blocks_test_;
  dmlc::SeekStream *              fr_;
  std::unique_ptr<dmlc::istream>  stream_;
  uint32                          isize_;
  int                             iter_;
  std::atomic<int>                pass_;
  std::chrono::time_point<Time>   s_;
};

class AdRegFilter : public mf::StatusStack,
                    public tbb::filter
{

 public:
  AdRegFilter(AdaptRegMF &model, mf::ObjectPool<mf::Block> &free_block_pool)
    : tbb::filter(parallel)
      , admf_(model)
      , free_block_pool_(free_block_pool)
  {}

  void *operator()(void *block) {
    float q[admf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
    mf::Block *bk = (mf::Block *) block;
    const float eta = admf_.eta_;
    int vid, j, i;
    float /*error,*/ rating;
    for (i = 0; i < bk->user_size(); i++) {
      const mf::User &user = bk->user(i);
      const int uid = user.uid();
      const int size = user.record_size();
      for (j = 0; j < size; j++) {
        memset(q, 0.0, sizeof(float) * admf_.dim_);
        const mf::User_Record &rec = user.record(j);
        vid = rec.vid();
        rating = rec.rating();
        cblas_scopy(admf_.dim_, admf_.theta_[uid], 1, admf_.theta_old_[uid], 1);
        cblas_scopy(admf_.dim_, admf_.phi_[vid], 1, admf_.phi_old_[vid], 1);
        float pred = active(
          cblas_sdot(admf_.dim_, admf_.theta_[uid], 1, admf_.phi_[vid], 1) + admf_.bu_[uid] + admf_.bv_[vid] +
          admf_.gb_, admf_.loss_);
        float error = cal_grad(rating, pred, admf_.loss_);
        error = eta * error;
        cblas_saxpy(admf_.dim_, error, admf_.theta_[uid], 1, q, 1);
        cblas_saxpy(admf_.dim_, -eta * admf_.lam_u_, admf_.theta_[uid], 1, admf_.theta_[uid], 1);
        cblas_saxpy(admf_.dim_, error, admf_.phi_[vid], 1, admf_.theta_[uid], 1);
        cblas_saxpy(admf_.dim_, 1.0f - eta * admf_.lam_v_, admf_.phi_[vid], 1, q, 1);
        cblas_scopy(admf_.dim_, q, 1, admf_.phi_[vid], 1);
        admf_.bu_old_[uid] = admf_.bu_[uid];
        admf_.bv_old_[vid] = admf_.bv_[vid];
        admf_.bu_[uid] = (1.0f - eta * admf_.lam_bu_) * admf_.bu_[uid] + error;
        admf_.bv_[vid] = (1.0f - eta * admf_.lam_bv_) * admf_.bv_[vid] + error;
      }
      int ii = rand() % admf_.recsv_.size();
      admf_.updateReg(admf_.recsv_[ii].u_, admf_.recsv_[ii].v_, admf_.recsv_[ii].r_);
    }
    free_block_pool_.freeObject(bk);
    return NULL;
  }

 private:
  AdaptRegMF&                 admf_;
  mf::ObjectPool<mf::Block>&  free_block_pool_;
};

} // namespace mf

#endif //_FASTMF_ADMF_H