#ifndef _FASTMF_MODEL_H
#define _FASTMF_MODEL_H

#include "util.h"
#include <dmlc/concurrency.h>

namespace mf
{

/**
 *  __  __  ______
 * |  \/  ||  ____|
 * | \  / || |__
 * | |\/| ||  __|
 * | |  | || |
 * |_|  |_||_|
 *
 */

class MF
{
 public:
  MF(const std::shared_ptr<mf::TrainConfig> config)
    : theta_(NULL)
      , phi_(NULL)
      , user_array_(NULL)
      , video_array_(NULL)
      , train_data_(config->train())
      , test_data_(config->test())
      , result_(config->result())
      , model_(config->model())
      , global_bias_(config->global_bias())
      , dim_(config->dim())
      , iter_(config->iterations())
      , learning_rate_(config->learning_rate())
      , learning_rate_decay_(config->learning_rate_decay())
      , lambda_(config->regularizer())
      , learning_rate0_(config->learning_rate())
      , nr_users_(config->nu() + 1)   // extra item to accommodate whether data set has a zero id user
      , nr_videos_(config->nv() + 1)  // extra item to accommodate whether data set has a zero id vid
      , data_in_fly_(config->fly())
      , prefetch_stride_(config->prefetch_stride()) {}

  virtual ~MF() {
    if(theta_) {
      free_aligned_alloc(theta_, nr_users_);
      if (phi_) {
        free_aligned_alloc(phi_, nr_videos_);
      }
      free(theta_);
    }
    free(user_array_);
  }

  virtual void init();

  float calc_mse(const mf::Blocks &blocks, int &ndata) const;

  virtual int read_model();
  virtual int save_model(int round);

  void set_learning_rate(int round);

  static constexpr float GAUSSIAN_NOISE_MULTIPLIER = 1e-2;

  float **theta_, **phi_, *user_array_, *video_array_;
  const std::string train_data_, test_data_, result_, model_;
  const float global_bias_;
  const int dim_, iter_;
  std::atomic<float> learning_rate_, learning_rate_decay_;
  const float lambda_, learning_rate0_;
  const int nr_users_, nr_videos_, data_in_fly_, prefetch_stride_;//48B
};

/**
 *  _____   _____   __  __  ______
 * |  __ \ |  __ \ |  \/  ||  ____|
 * | |  | || |__) || \  / || |__
 * | |  | ||  ___/ | |\/| ||  __|
 * | |__| || |     | |  | || |
 * |_____/ |_|     |_|  |_||_|
 *
 *
 */
class DPMF : public MF
{
 public:
  DPMF(const std::shared_ptr<mf::TrainConfig> config)
  : MF(config)
    , hyper_a_(config->hypera())
    , hyper_b_(config->hyperb())
    , sgld_temperature_(config->sgld_temperature())
    , min_learning_rate_(config->min_learning_rate())
    , noise_size_(config->noise_size())
    , max_ratings_(config->max_ratings())
    , dfp_sensitivity_(config->dfp_sensitivity())
    , lambda_r_(1e0)
    , lambda_ub_(1e2)
    , lambda_vb_(1e2)
    , ntrain_(0)
    , ntest_(0) {}

  ~DPMF() {
    if(noise_) {
      free(noise_);
    }
    delete[] gcountu;
    delete[] gcountv;
    delete[] gmutex;
  }

  void init();

  void block_count(int *uc, int *vc, mf::Block *bk);

  int sample_train_and_precompute_weight();

  void set_learning_rate_cutoff(int round);

  int read_model();

  int read_hyper();

  int save_model(int round);

  void finish_noise();

  void finish_round(const mf::Blocks &blocks_test, int round, const std::chrono::time_point<Time>& s);

  void sample_hyper(float mse);

  std::atomic<uint64_t> *gcountu;
  std::atomic<uint64_t> *gcountv;//128
#if 1
  fast_mutex *gmutex;
#else
  std::mutex *gmutex;
#endif
  std::uniform_int_distribution<> uniform_int_;
  float *ur_,
    *vr_,
    *noise_,
    *lambda_u_,
    *lambda_v_;
  const float hyper_a_,
    hyper_b_;//192
  mf::Blocks train_sample_;
  const float sgld_temperature_, min_learning_rate_;
  const int noise_size_;
  int max_ratings_;
  const float dfp_sensitivity_;
  float lambda_r_, lambda_ub_, lambda_vb_;
  long bound_;
  int ntrain_;
  const int ntest_;
  char pad[CACHE_LINE_SIZE];
  std::atomic<uint64> gcount;
};

/**
 *               _                _    _____               __  __  ______
 *     /\       | |              | |  |  __ \             |  \/  ||  ____|
 *    /  \    __| |  __ _  _ __  | |_ | |__) | ___   __ _ | \  / || |__
 *   / /\ \  / _` | / _` || '_ \ | __||  _  / / _ \ / _` || |\/| ||  __|
 *  / ____ \| (_| || (_| || |_) || |_ | | \ \|  __/| (_| || |  | || |
 * /_/    \_\\__,_| \__,_|| .__/  \__||_|  \_\\___| \__, ||_|  |_||_|
 *                        | |                        __/ |
 *                        |_|                       |___/
 */
class AdaptRegMF : public MF
{
 public:

  AdaptRegMF(const std::shared_ptr<mf::TrainConfig> config)
    : MF(config)
      , valid_data_(config->valid())
      , learning_rate_reg_(config->learning_rate_reg())
      , learning_rate0_reg_(config->learning_rate_reg())
      , loss_(config->loss())
      , measure_(config->measure())
      , lam_u_(config->regularizer())
      , lam_v_(config->regularizer())
      , lam_bu_(config->regularizer())
      , lam_bv_(config->regularizer()) {}

  ~AdaptRegMF() {
    if(theta_old_) {
      free_aligned_alloc(theta_old_, nr_users_);
      if(phi_old_) {
        free_aligned_alloc(phi_old_, nr_videos_);
      }
      free(theta_old_);
    }
    if(bu_old_) {
      free(bu_old_);
    }
  }

  void init1();

  inline void updateReg(int uid, int vid, float rating) {
    float pred = active(cblas_sdot(dim_, theta_[uid], 1, phi_[vid], 1) + user_array_[uid] + video_array_[vid] + global_bias_, loss_);
    float grad = cal_grad(rating, pred, loss_);
    updateUV(grad, uid, vid);
    updateBias(grad, uid, vid);
  }

  inline void updateUV(float grad, int uid, int vid) {
    float inner = cblas_sdot(dim_, theta_old_[uid], 1, phi_[vid], 1);
    lam_u_ = std::max(0.0f, lam_u_ - learning_rate_reg_ * learning_rate_ * grad * inner);
    inner = cblas_sdot(dim_, theta_[uid], 1, phi_old_[vid], 1);
    lam_v_ = std::max(0.0f, lam_v_ - learning_rate_reg_ * learning_rate_ * grad * inner);
  }

  inline void updateBias(float grad, int uid, int vid) {
    lam_bu_ = std::max(0.0f, lam_bu_ - learning_rate_reg_ * learning_rate_ * grad * (bu_old_[uid]));
    lam_bv_ = std::max(0.0f, lam_bv_ - learning_rate_reg_ * learning_rate_ * grad * (bv_old_[vid]));
  }

  void set_etareg(int round);

  int plain_read_valid(const std::string& valid);

  std::vector<Record> recsv_;
  float **theta_old_, **phi_old_, *bu_old_, *bv_old_;
  const std::string valid_data_;
  float learning_rate_reg_;
  const float learning_rate0_reg_;
  const int loss_, measure_;
  char pad1[CACHE_LINE_SIZE];
  float lam_u_;
  char pad2[CACHE_LINE_SIZE];
  float lam_v_;
  char pad3[CACHE_LINE_SIZE];
  float lam_bu_;
  char pad4[CACHE_LINE_SIZE];
  float lam_bv_;
};

} // namespace mf

#endif //_FASTMF_MODEL_H
