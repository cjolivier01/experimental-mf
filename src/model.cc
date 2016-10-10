#include <memory>
#include "model.h"

namespace mf
{

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::normal_distribution<float> gaussian(0.0f, 1.0f);

/**
 *  __  __  ______
 * |  \/  ||  ____|
 * | \  / || |__
 * | |\/| ||  __|
 * | |  | || |
 * |_|  |_||_|
 *
 */

/* two chunks, read-only and frequent write, should be seperated.
   Align the address of write chunk to cache line */
void MF::init() {

  CHECK_EQ(std::atomic<float>().is_lock_free(), true);

  // TODO: This should be an aligned alloc
  //alloc
  user_array_ = (float *) malloc((nr_users_ + nr_videos_) * sizeof(float));
  memset(user_array_, 0, (nr_users_ + nr_videos_) * sizeof(float));
  video_array_ = user_array_ + nr_users_;

  const int pad = padding(dim_);

  // TODO: This should be an aligned alloc
  theta_ = (float **) malloc((nr_users_ + nr_videos_) * sizeof(float *));
  memset(theta_, 0, (nr_users_ + nr_videos_) * sizeof(float *));
  phi_ = theta_ + nr_users_;
  align_alloc(theta_, nr_users_, pad);
  align_alloc(phi_, nr_videos_, pad);
  //init

#pragma omp parallel for
  for (int i = 0; i < nr_users_; i++) {
    for (int j = 0; j < dim_; j++) {
      theta_[i][j] = gaussian(generator) * GAUSSIAN_NOISE_MULTIPLIER;
      const float f = theta_[i][j];
      DCHECK_EQ(mf::isFinite(theta_[i][j]), true);
    }
  }
#pragma omp parallel for
  for (int i = 0; i < nr_videos_; i++) {
    for (int j = 0; j < dim_; j++) {
      const float f = phi_[i][j];
      phi_[i][j] = gaussian(generator) * GAUSSIAN_NOISE_MULTIPLIER;
      DCHECK_EQ(mf::isFinite(phi_[i][j]), true);
    }
  }
#pragma omp parallel for
  for (int i = 0; i < nr_users_ + nr_videos_; i++) {
    user_array_[i] = gaussian(generator) * GAUSSIAN_NOISE_MULTIPLIER;
    const float f = user_array_[i];
    DCHECK_EQ(mf::isFinite(user_array_[i]), true);
  }
}

void MF::set_learning_rate(int round) {
  learning_rate_ = (float) (learning_rate0_ * 1.0 / pow(round, learning_rate_decay_));
}

float MF::calc_mse(const mf::Blocks &blocks, int &return_ndata) const {
  std::mutex mlock;
  const int bsize = blocks.block_size();
  volatile float sloss = 0.0;
  volatile int ndata = 0;
#pragma omp parallel for
  for (int i = 0; i < bsize; i++) {
    float sl = 0.0;
    int nn = 0;
    const mf::Block &bk = blocks.block(i);
    const int usize = bk.user_size();
    for (int j = 0; j < usize; j++) {
      const mf::User &user = bk.user(j);
      const int uid = user.uid();

      CHECK_LT(uid, nr_users_);
      DCHECK_EQ(mf::isFinite(user_array_[uid]), true);

      const int rsize = user.record_size();
      nn += rsize;
      for (int k = 0; k < rsize; k++) {
        const mf::User_Record &rec = user.record(k);
        const int vid = rec.vid();
        const float rating = rec.rating();

        CHECK_LT(vid, nr_videos_);
        DCHECK_EQ(mf::isFinite(video_array_[vid]), true);

        const float error = rating - cblas_sdot(dim_, theta_[uid], 1, phi_[vid], 1)
                            - user_array_[uid] - video_array_[vid] - global_bias_;
        sl += error * error;
      }
    }
    mlock.lock();
    sloss += sl;
    ndata += nn;
    mlock.unlock();
  }
  return_ndata = ndata;
  return sloss;
}

#define READVAR(__stream$, __v$) do { \
    if((__stream$).read((char *)&(__v$), sizeof(__v$)).fail()) { \
      return EIO; \
    } \
  } while(0)

#define WRITEVAR(__stream$, __v$) do { \
    if((__stream$).write((const char *)&(__v$), sizeof(__v$)).fail()) { \
      return EIO; \
    } \
  } while(0)


int MF::save_model(int round) {
  int rc = 0;
  if(!result_.empty()) {
    std::string file = result_;
    file += "_";
    file += std::to_string(round);
    mf::dmlc_ostream output(file);
    if (output.is_open()) {
      WRITEVAR(output, nr_users_);
      WRITEVAR(output, nr_videos_);
      WRITEVAR(output, dim_);
      //write lambda
      WRITEVAR(output, lambda_);
      //write gb
      //fwrite(&gb,1,sizeof(float),fp);
      //write v
      for (int i = 0; i < nr_videos_; i++) {
        WRITEVAR(output, video_array_[i]);
      }
      for (int i = 0; i < nr_videos_; i++) {
        for (int j = 0; j < dim_; j++) {
          WRITEVAR(output, phi_[i][j]);
        }
      }
      //write u
      for (int i = 0; i < nr_users_; i++) {
        WRITEVAR(output, user_array_[i]);
      }

      for (int i = 0; i < nr_users_; i++) {
        for (int j = 0; j < dim_; j++) {
          WRITEVAR(output, theta_[i][j]);
        }
      }
    } else {
      rc = errno ? errno : EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

int MF::read_model() {
  int rc = 0;
  if(!model_.empty()) {
    mf::dmlc_istream input(model_);
    if (input.is_open()) {
      READVAR(input, nr_videos_);
      READVAR(input, nr_users_);
      READVAR(input, dim_);
      //read lambda
      READVAR(input, lambda_);
      //read gb
      //fread(&gb,1,sizeof(float),fp);
      //read v
      for (int i = 0; i < nr_videos_; i++) {
        READVAR(input, video_array_[i]);
      }
      for (int i = 0; i < nr_videos_; i++) {
        for (int j = 0; j < dim_; j++) {
          READVAR(input, phi_[i][j]);
        }
      }
      //read u
      for (int i = 0; i < nr_users_; i++) {
        READVAR(input, user_array_[i]);
      }
      for (int i = 0; i < nr_users_; i++) {
        for (int j = 0; j < dim_; j++) {
          READVAR(input, theta_[i][j]);
        }
      }
    } else {
      rc = errno ? errno : EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

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

int DPMF::save_model(int round) {
  int rc = 0;
  if(!result_.empty()) {
    std::string file = result_;
    file += "_";
    file += std::to_string(round);
    mf::dmlc_ostream output(file);
    if (output.is_open()) {
      WRITEVAR(output, nr_videos_);
      WRITEVAR(output, nr_users_);
      WRITEVAR(output, dim_);
      //write lambda
      WRITEVAR(output, lambda_r_);
      WRITEVAR(output, lambda_ub_);
      WRITEVAR(output, lambda_vb_);
      for (int i = 0; i < dim_; i++) {
        WRITEVAR(output, lambda_u_[i]);
      }
      for (int i = 0; i < dim_; i++) {
        WRITEVAR(output, lambda_v_[i]);
      }
      //write gb
      //fwrite(&gb,1,sizeof(float),fp);
      //write v
      for (int i = 0; i < nr_videos_; i++) {
        WRITEVAR(output, video_array_[i]);
      }
      for (int i = 0; i < nr_videos_; i++) {
        for (int j = 0; j < dim_; j++) {
          WRITEVAR(output, phi_[i][j]);
        }
      }
      //write u
      for (int i = 0; i < nr_users_; i++) {
        WRITEVAR(output, user_array_[i]);
      }
      for (int i = 0; i < nr_users_; i++) {
        for (int j = 0; j < dim_; j++) {
          WRITEVAR(output, theta_[i][j]);
        }
      }
    } else {
      rc = errno ? errno : EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

int DPMF::read_hyper() {
  int rc = 0;
  if(!model_.empty()) {
    mf::dmlc_istream input(model_);
    if (input.is_open()) {
      READVAR(input, nr_videos_);
      READVAR(input, nr_users_);
      READVAR(input, dim_);
      //read lambda
      READVAR(input, lambda_r_);
      READVAR(input, lambda_ub_);
      READVAR(input, lambda_vb_);
      for (int i = 0; i < dim_; i++) {
        READVAR(input, lambda_u_[i]);
      }
      for (int i = 0; i < dim_; i++) {
        READVAR(input, lambda_v_[i]);
      }
      //read gb
      //fread(&gb,1,sizeof(float),fp);
    } else {
      rc = errno ? errno : EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

int DPMF::read_model() {
  int rc = 0;
  if(!model_.empty()) {
    mf::dmlc_istream input(model_);
    if (input.is_open()) {
      READVAR(input, nr_videos_);
      READVAR(input, nr_users_);
      READVAR(input, dim_);
      //read lambda
      READVAR(input, lambda_r_);
      READVAR(input, lambda_ub_);
      READVAR(input, lambda_vb_);
      for (int i = 0; i < dim_; i++) {
        READVAR(input, lambda_u_[i]);
      }
      for (int i = 0; i < dim_; i++) {
        READVAR(input, lambda_v_[i]);
      }
      //read gb
      //fread(&gb,1,sizeof(float),fp);
      //read v
      for (int i = 0; i < nr_videos_; i++) {
        READVAR(input, video_array_[i]);
      }
      for (int i = 0; i < nr_videos_; i++) {
        for (int j = 0; j < dim_; j++) {
          READVAR(input, phi_[i][j]);
        }
      }
      //read u
      for (int i = 0; i < nr_users_; i++) {
        READVAR(input, user_array_[i]);
      }
      for (int i = 0; i < nr_users_; i++) {
        for (int j = 0; j < dim_; j++) {
          READVAR(input, theta_[i][j]);
        }
      }
    } else {
      rc = errno ? errno : EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

void DPMF::init() {
  //alloc
  const size_t memSize = (2 * (nr_users_ + nr_videos_) + 2 * dim_) * sizeof(float);
  user_array_ = (float *) malloc(memSize);
  memset(user_array_, 0, memSize);
  video_array_ = user_array_ + nr_users_;
  ur_ = video_array_ + nr_videos_;
  vr_ = ur_ + nr_users_;
  lambda_u_ = vr_ + nr_videos_;
  lambda_v_ = lambda_u_ + dim_;

  const int pad = padding(dim_);

  theta_ = (float **) malloc((nr_users_ + nr_videos_) * sizeof(float *));
  phi_ = theta_ + nr_users_;
  align_alloc(theta_, nr_users_, pad);
  align_alloc(phi_, nr_videos_, pad);

  //bookkeeping
  gcount = 0;
  gcountu = new std::atomic<uint64_t>[nr_users_];
  gcountv = new std::atomic<uint64_t>[nr_videos_];
#if 0
  gmutex  = new dmlc::Spinlock[nr_videos_];
#else
  gmutex  = new std::mutex[nr_videos_];
#endif

  //init
#pragma omp parallel for
  for (int i = 0; i < nr_users_; i++) {
    for (int j = 0; j < dim_; j++) {
      theta_[i][j] = gaussian(generator) * 1e-2f;
      DCHECK_EQ(isFinite(theta_[i][j]), true);
    }
    gcountu[i] = 0;
  }
#pragma omp parallel for
  for (int i = 0; i < nr_videos_; i++) {
    for (int j = 0; j < dim_; j++) {
      phi_[i][j] = gaussian(generator) * 1e-2f;
      DCHECK_EQ(isFinite(phi_[i][j]), true);
    }
    gcountv[i] = 0;
  }
#pragma omp parallel for
  for (int i = 0; i < nr_users_ + nr_videos_; i++) {
    user_array_[i] = gaussian(generator) * 1e-2f;
    DCHECK_EQ(isFinite(user_array_[i]), true);
    DCHECK_LT(user_array_[i], 10.0f);
  }
  for (int i = 0; i < 2 * dim_; i++) {
    lambda_u_[i] = 1e2;
  }

  //noise
  noise_ = (float *) malloc(sizeof(float) * noise_size_);
#pragma omp parallel for
  for (int i = 0; i < noise_size_; i++) {
    noise_[i] = gaussian(generator);
    DCHECK_EQ(isFinite(noise_[i]), true);
  }
  //sample train data and precompute weights according to training data
  sample_train_and_precompute_weight();

  if(ntrain_) {
    // eta is divided over training items
    learning_rate_.store(learning_rate_.load() / ntrain_);
  }

  //differentially private
  if (max_ratings_ <= 0) {
    max_ratings_ = nr_videos_;
  }
  if (dfp_sensitivity_ <= 0.0f) {
    bound_ = 1.0f;
  }
  else {
    bound_ = dfp_sensitivity_ * 1.0f / (4.0f * 25.0f * max_ratings_);
  }
  uniform_int_ = std::uniform_int_distribution<>(0, noise_size_ - max_ratings_ * (dim_ + 1) - 1);
  assert(noise_size_ - max_ratings_ * (dim_ + 1) > 10000);
}

/**
 * Counts the number of training accesses on both users and vids
 * @param uc Array to user access count
 * @param vc Array to vid access count
 * @param bk Current Block item being processed
 */
void DPMF::block_count(int *users, int *vids, mf::Block *bk) {
  for (int i = 0; i < bk->user_size(); i++) {
    const mf::User &user = bk->user(i);
    const int uid = user.uid();
//    if(!uid) {
//      std::cerr << "uid = 0" << std::endl;
//    }
    CHECK_LT(uid, nr_users_);
    const int size = user.record_size();
    for (int j = 0; j < size; j++) {
      const mf::User_Record &rec = user.record(j);
      const int vid = rec.vid();
//      if(!vid) {
//        std::cerr << "vid = 0" << std::endl;
//      }
      CHECK_LT(vid, nr_videos_);
      ++users[uid];
      ++vids[vid];
      ++ntrain_;
    }
  }
}

int DPMF::sample_train_and_precompute_weight() {
  int rc = 0;
  if(!train_data_.empty()) {
    mf::dmlc_istream input(train_data_);
    if (input.is_open()) {
      uint32 isize;
      std::vector<char> buf;
      std::default_random_engine generator;
      std::uniform_real_distribution<float> distribution(0.0, 1.0);

      std::unique_ptr<int> _users(new int[nr_users_]);
      int *users = _users.get();
      std::unique_ptr<int> _vids(new int[nr_videos_]);
      int *vids = _vids.get();
      ntrain_ = 0;
      int pass = 0, train_sample_count = 0;
      while(!rc) {
        float ratio = distribution(generator);
        if (ratio <= 0.1 /*1.0*/ && train_sample_count < 1e6) {
          mf::Block *pbk = train_sample_.add_block();
          ++train_sample_count;
          if(!input.read((char *)&isize, sizeof(isize)).fail()) {
            buf.resize(isize);
            if(!input.read(buf.data(), isize).fail()) {
              if(pbk->ParseFromArray(buf.data(), isize)) {
                block_count(users, vids, pbk);
              } else {
                rc = EINVAL;
                break;
              }
            } else {
              rc = errno ? errno : EIO;
              break;
            }
          } else {
            rc = input.eof() ? 0 : EIO;
            break;
          }
        } else {
          if(!input.read((char *)&isize, sizeof(isize)).fail()) {
            buf.resize(isize);
            if(!input.read(buf.data(), isize).fail()) {
              mf::Block bk;
              if(bk.ParseFromArray(buf.data(), isize)) {
                block_count(users, vids, &bk);
              } else {
                rc = EINVAL;
                break;
              }
            } else {
              rc = errno ? errno : EIO;
              break;
            }
          } else {
            rc = input.eof() ? 0 : EIO;
            break;
          }
        }
        ++pass;
      }
      if(!rc) {
        for (int i = 0; i < nr_users_; i++) {
          // May not have zero-user
          if(users[i]) {
            CHECK_NE(users[i], 0);
            ur_[i] = (float) ntrain_ / users[i];
            //ur_[i] = ((float) users[i]) / ntrain_;
            DCHECK_EQ(mf::isFinite(ur_[i]), true);
          }
        }
        for (int i = 0; i < nr_videos_; i++) {
          if(vids[i]) {
            // May not have zero-vid
            CHECK_NE(vids[i], 0);
            vr_[i] = (float) ntrain_ / vids[i];
            //vr_[i] = ((float) vids[i])/ntrain_;
            DCHECK_EQ(mf::isFinite(vr_[i]), true);
          }
        }
      }
    } else {
      rc = errno ? errno : EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

void DPMF::finish_round(const mf::Blocks &blocks_test, int round, const std::chrono::time_point<Time>& s) {
  finish_noise();
  int ntr, nt;
  const float mse = calc_mse(train_sample_, ntr);
  const float tmse = calc_mse(blocks_test, nt);
  printf("round #%d\tRMSE=%f\ttRMSE=%f\t", round, sqrt(mse * 1.0 / ntr), sqrt(tmse * 1.0 / nt));
  sample_hyper(mse);
  set_learning_rate_cutoff(round + 1);
  printf("%f\n", std::chrono::duration<float>(Time::now() - s).count());
  if (round >= 100 && round % 20 == 0) {
    save_model(round);
  }
}

void DPMF::finish_noise() {
  const size_t gc = gcount.load();
  int rndind;
#pragma omp parallel for
  for (int i = 0; i < nr_users_; i++) {
    rndind = uniform_int_(generator);
    const uint64_t uc = gc - gcountu[i];
    DCHECK_GE(gc, gcountu[i]);
    gcountu[i] = 0;
    DCHECK_LT(rndind + dim_, noise_size_);
    cblas_saxpy(dim_, (float)sqrt(sgld_temperature_ * learning_rate_ * uc), noise_ + rndind, 1, theta_[i], 1);
    DCHECK_EQ(isFinite(user_array_[i]), true);
    const float adder = (float)sqrt(sgld_temperature_ * learning_rate_ * uc) * noise_[rndind + dim_];
    user_array_[i] += (float)sqrt(sgld_temperature_ * learning_rate_ * uc) * noise_[rndind + dim_];
    DCHECK_EQ(isFinite(user_array_[i]), true);
    DCHECK_LT(fabs(user_array_[i]), 10.0f);
  }
#pragma omp parallel for
  for (int i = 0; i < nr_videos_; i++) {
    rndind = uniform_int_(generator);
    const uint64_t vc = gc - gcountv[i].load();
    DCHECK_GE(gc, gcountv[i].load());
    gcountv[i] = 0;
    DCHECK_LT(rndind + dim_, noise_size_);
    cblas_saxpy(dim_, (float)sqrt(sgld_temperature_ * learning_rate_ * vc), noise_ + rndind, 1, phi_[i], 1);
    DCHECK_EQ(isFinite(video_array_[i]), true);
    const float adder = (float)sqrt(sgld_temperature_ * learning_rate_ * vc) * noise_[rndind + dim_];
    video_array_[i] += (float)sqrt(sgld_temperature_ * learning_rate_ * vc) * noise_[rndind + dim_];
    DCHECK_EQ(isFinite(video_array_[i]), true);
    DCHECK_LT(fabs(video_array_[i]), 10.0f);
  }
  gcount = 0;
}


void DPMF::sample_hyper(float mse) {
  gamma_posterior(lambda_r_, hyper_a_, hyper_b_, mse, ntrain_);
  gamma_posterior(lambda_ub_, hyper_a_, hyper_b_, normsqr(user_array_, nr_users_), nr_users_);
  gamma_posterior(lambda_vb_, hyper_a_, hyper_b_, normsqr(video_array_, nr_videos_), nr_videos_);

  float normu[dim_]={0.0}, normv[dim_]={0.0};

  normsqr_col(theta_, dim_, nr_users_, normu);
  normsqr_col(phi_, dim_, nr_videos_, normv);
#pragma omp parallel for
  for (int i = 0; i < dim_; i++) {
    gamma_posterior(lambda_u_[i], hyper_a_, hyper_b_, normu[i], nr_users_);
    gamma_posterior(lambda_v_[i], hyper_a_, hyper_b_, normv[i], nr_videos_);
  }
}

void DPMF::set_learning_rate_cutoff(int round) {
  learning_rate_ = std::max(min_learning_rate_, (float) (learning_rate0_ * 1.0 / pow(round, learning_rate_decay_)));
  if(ntrain_) {
    learning_rate_.store(learning_rate_.load() / ntrain_);
  }
}


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
void AdaptRegMF::init1() {
  init();

  const int pad = padding(dim_);

  bu_old_ = (float *) malloc((nr_users_ + nr_videos_) * sizeof(float));
  bv_old_ = bu_old_ + nr_users_;

  theta_old_ = (float **) malloc((nr_users_ + nr_videos_) * sizeof(float *));
  phi_old_ = theta_old_ + nr_users_;
  align_alloc(theta_old_, nr_users_, pad);
  align_alloc(phi_old_, nr_videos_, pad);

  //init
#pragma omp parallel for
  for (int i = 0; i < nr_users_; i++) {
    for (int j = 0; j < dim_; j++) {
      theta_old_[i][j] = theta_[i][j];
    }
  }
#pragma omp parallel for
  for (int i = 0; i < nr_videos_; i++) {
    for (int j = 0; j < dim_; j++) {
      phi_old_[i][j] = phi_[i][j];
    }
  }
  // TODO: Do with two loops, for Christ's sake (users, then vids)
#pragma omp parallel for
  for (int i = 0; i < nr_users_ + nr_videos_; i++) {
    bu_old_[i] = user_array_[i];
  }
}

void AdaptRegMF::set_etareg(int round) {
  learning_rate_reg_ = (float) (learning_rate0_reg_ * 1.0 / pow(round, learning_rate_decay_));
}

int AdaptRegMF::plain_read_valid(const std::string& valid) {
  int rc = 0;
  if(!valid.empty()) {
    mf::dmlc_istream input(valid);
    if (input.is_open()) {
      std::vector<char> buf;
      uint32 isize;
      mf::Block bk;
      Record rr;
      while (!input.read((char *)&isize, sizeof(isize)).fail()) {
        buf.resize(isize);
        if(!input.read(buf.data(), isize).fail()) {
          bk.ParseFromArray(buf.data(), isize);
          for (int i = 0; i < bk.user_size(); i++) {
            const mf::User &user = bk.user(i);
            const int uid = user.uid();
            for (int j = 0; j < user.record_size(); j++) {
              const mf::User_Record &rec = user.record(j);
              rr.u_ = uid;
              rr.v_ = (int) rec.vid();
              rr.r_ = rec.rating();
              DCHECK_EQ(isFinite(rr.r_), true);
              recsv_.push_back(rr);
            }
          }
          bk.Clear();
        } else {
          if(!input.eof()) {
            rc = EIO;
          }
          break;
        }
      }
      if(!rc) {
        std::random_shuffle(recsv_.begin(), recsv_.end());
      }
    } else {
      rc = EIO;
    }
  } else {
    rc = EINVAL;
  }
  return rc;
}

} //namespace mf