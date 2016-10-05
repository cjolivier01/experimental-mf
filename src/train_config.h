#ifndef FASTMF_TRAIN_CONFIG_H
#define FASTMF_TRAIN_CONFIG_H

#include "blocks.pb.h"

namespace mf
{

class Configuration
{
 public:
  //void setDefaults(mf::Tr
  static std::shared_ptr<mf::TrainConfig> createDefaultTrainConfig() {
    std::shared_ptr<mf::TrainConfig> config(new mf::TrainConfig());
    if (config) {
      config->set_alg("mf");
      config->set_dim(128);
      config->set_iterations(10);
      config->set_fly(4);
      config->set_max_ratings(0);
      config->set_fly(8);
      config->set_prefetch_stride(2);
      config->set_hypera(1.0f);
      config->set_hyperb(100.0);
      config->set_sgld_temperature(1.0f);
      config->set_global_bias(2.76);
      config->set_learning_rate(2e-2);
      config->set_dfp_sensitivity(0.0f);
      config->set_noise_size(20000000);
      config->set_loss(0);
      config->set_measure(0);
      config->set_learning_rate_reg(2e-3f);
      config->set_min_learning_rate(1e-13f);
      config->set_learning_rate_decay(1.0f);
      config->set_regularizer(5e-3f);
    }
    return config;
  }
};

} // namespace mf

#endif //FASTMF_TRAIN_CONFIG_H
