#include <dmlc/io.h>
#include "../src/model.h"
#include "../src/mf.h"
#include "../src/dpmf.h"
#include "../src/admf.h"
#include "../src/train_config.h"
#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#endif

using namespace mf;

static void show_help() {
  printf("Usage:\n");
  printf("./mf\n");
  printf("--train      xxx       : xxx is the file name of the binary training data.\n");
  printf("--nu         int       : number of users.\n");
  printf("--nv         int       : number of items.\n");
  printf("--test       xxx       : xxx is the file name of the binary test data.\n");
  printf("--valid      [xxx]     : xxx is the file name of the binary validation data.\n");
  printf("--result     [xxx]     : save your model in name xxx.\n");
  printf("--model      [xxx]     : read your model in name xxx.\n");
  printf("--alg        [xxx]     : xxx can be {mf, dpmf, admf}.\n");
  printf("--dim        [int]     : low rank of the model.\n");
  printf("--iter       [int]     : number of iterations.\n");
  printf("--fly        [int]     : number of threads.\n");
  printf("--stride     [int]     : prefetch strides.\n");
  printf("--lr         [float]   : learning rate.\n");
  printf("--lambda     [float]   : regularizer.\n");
  printf("--gam        [float]   : decay of learning rate.\n");
  printf("--bias       [float]   : global bias (important for accuracy).\n");
  printf("--mineta     [float]   : minimum learning rate (sometimes used in SGLD).\n");
  printf("--epsilon    [float]   : sensitivity of differentially privacy.\n");
  printf("--max-ratings [int]    : maximum of ratings among all the users (usually after trimming your data).\n");
  printf("--temp       [float]   : temperature in SGLD (can accelerate the convergence).\n");
  printf("--noise-size [int]     : the Gaussian numbers lookup table.\n");
  printf("--lr-reg     [float]   : the learning rate for estimating regularization parameters.\n");
  printf("--loss       [int]     : the loss type can be {least square, 0-1 logistic regression}.\n");
  printf("--measure    [int]     : support RMSE.\n");
}

inline int setOnError(const mf::StatusStack& statusStack, int& rc) {
  const mf::StatusStack::StatusCode code = statusStack.getLastStatusCode();
  if (code != mf::StatusStack::OK) {
    rc = code;
  }
  return rc;
}

//assuming test data can fit into RAM
static int run(MF& mf) {
  int rc = 0;
  mf.init();
  if(!mf.model_.empty()) {
    rc = mf.read_model();
  }

  if(!rc) {
    mf::Blocks blocks_test;
    rc = plain_read(mf.test_data_, blocks_test);
    if (!rc) {
      std::unique_ptr<dmlc::SeekStream> f(dmlc::SeekStream::CreateForRead(mf.train_data_.c_str()));
      if (f.get()) {
        mf::perf::TimingInstrument timing;
        SgdReadFilter read_f(mf, f.get(), blocks_test, &timing);
        ParseFilter parse_f(mf.data_in_fly_, read_f, &timing);
        SgdFilter sgd_f(mf, parse_f, &timing);
        tbb::pipeline p;
        p.add_filter(read_f);
        p.add_filter(parse_f);
        p.add_filter(sgd_f);
        // Check errors in reverse order
        p.run(mf.data_in_fly_);
        read_f.printBlockedTime("read_f queue blocked for");
        parse_f.printBlockedTime("parse_f queue blocked for");
        setOnError(sgd_f, rc);
        setOnError(parse_f, rc);
        setOnError(read_f, rc);
        timing.print();
      } else {
        rc = errno;
      }
    }
  }
  return rc;
}

//assuming test data can fit into RAM
static int run(DPMF& dpmf) {
  int rc = 0;
  dpmf.init();
  if(!dpmf.model_.empty()) {
    rc = dpmf.read_hyper();
  }
  if(!rc) {
    mf::Blocks blocks_test;
    rc = plain_read(dpmf.test_data_, blocks_test);
    if (!rc) {
      std::unique_ptr<dmlc::SeekStream> f(dmlc::SeekStream::CreateForRead(dpmf.train_data_.c_str()));
      if (f.get()) {
        mf::perf::TimingInstrument timing;
        SgldReadFilter read_f(dpmf, f.get(), blocks_test, &timing);
        ParseFilter parse_f(dpmf.data_in_fly_, read_f, &timing);
        SgldFilter sgld_f(dpmf, parse_f, &timing);
        tbb::pipeline p;
        p.add_filter(read_f);
        p.add_filter(parse_f);
        p.add_filter(sgld_f);
        p.run(dpmf.data_in_fly_);
        read_f.printBlockedTime("read_f queue blocked for");
        parse_f.printBlockedTime("parse_f queue blocked for");

        read_f.printAll();
        parse_f.printAll();
        sgld_f.printAll();

        setOnError(sgld_f, rc);
        setOnError(parse_f, rc);
        setOnError(read_f, rc);
        timing.print();
      } else {
        rc = errno;
      }
    }
  }
  return rc;
}

//assuming test, valid data can fit into RAM
static int run(AdaptRegMF& admf) {
  int rc = 0;
  admf.init1();
  mf::Blocks blocks_test;
  rc = plain_read(admf.test_data_, blocks_test);
  if(!rc) {
    rc = admf.plain_read_valid(admf.valid_data_);
    if(!rc) {
      std::unique_ptr<dmlc::SeekStream> f(dmlc::SeekStream::CreateForRead(admf.train_data_.c_str()));
      if (f.get()) {
        mf::perf::TimingInstrument timing;
        AdRegReadFilter read_f(admf, f.get(), blocks_test, &timing);
        ParseFilter parse_f(admf.data_in_fly_, read_f, &timing);
        AdRegFilter admf_f(admf, parse_f, &timing);
        tbb::pipeline p;
        p.add_filter(read_f);
        p.add_filter(parse_f);
        p.add_filter(admf_f);
        p.run(admf.data_in_fly_);
        read_f.printBlockedTime("read_f queue blocked for");
        parse_f.printBlockedTime("parse_f queue blocked for");
        setOnError(admf_f, rc);
        setOnError(parse_f, rc);
        setOnError(read_f, rc);
        timing.print();
      } else {
        rc = errno;
        CHECK_NE(rc, 0);
      }
    }
  }
  return rc;
}

/*
    char *train_data = NULL, *test_data = NULL, *result = NULL, *alg = NULL, *model = NULL;
    int dim = 128, iter = 15, tau = 0, nu = 0, nv = 0, fly = 8, stride = 2;
    float eta = 2e-2, lambda = 5e-3, gam = 1.0f, mineta = 1e-13;
    float epsilon = 0.0f, hypera = 1.0f, hyperb = 100.0f, temp = 1.0f;
    float g_bias = 2.76f;
    int noise_size = 2000000000;
    int loss = 0;
    int measure = 0;
    float eta_reg = 2e-3f;
    char *valid_data = NULL;
    for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--train")) train_data = argv[++i];
      else if (!strcmp(argv[i], "--test")) test_data = argv[++i];
      else if (!strcmp(argv[i], "--valid")) valid_data = argv[++i];
      else if (!strcmp(argv[i], "--result")) result = argv[++i];
      else if (!strcmp(argv[i], "--model")) model = argv[++i];
      else if (!strcmp(argv[i], "--alg")) alg = argv[++i];
      else if (!strcmp(argv[i], "--dim")) dim = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--iter")) iter = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--nu")) nu = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--nv")) nv = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--fly")) fly = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--stride")) stride = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--eta")) eta = atof(argv[++i]);
      else if (!strcmp(argv[i], "--lambda")) lambda = atof(argv[++i]);
      else if (!strcmp(argv[i], "--gam")) gam = atof(argv[++i]);
      else if (!strcmp(argv[i], "--bias")) g_bias = atof(argv[++i]);
      else if (!strcmp(argv[i], "--mineta")) mineta = atof(argv[++i]);
      else if (!strcmp(argv[i], "--epsilon")) epsilon = atof(argv[++i]);
      else if (!strcmp(argv[i], "--tau")) tau = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--hypera")) hypera = atof(argv[++i]);
      else if (!strcmp(argv[i], "--hyperb")) hyperb = atof(argv[++i]);
      else if (!strcmp(argv[i], "--temp")) temp = atof(argv[++i]);
      else if (!strcmp(argv[i], "--noise-size")) noise_size = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--eta-reg")) eta_reg = atof(argv[++i]);
      else if (!strcmp(argv[i], "--loss")) loss = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--measure")) measure = atoi(argv[++i]);
      else {
        printf("%s, unknown parameters, exit\n", argv[i]);
        return 1;
      }
    }
 */

int main(int argc, char** argv) {
  int rc = 0;
  do {
    std::shared_ptr<mf::TrainConfig> config = Configuration::createDefaultTrainConfig();
    if(config) {
      for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--train"))          config->set_train(argv[++i]);
        else if (!strcmp(argv[i], "--test"))      config->set_test(argv[++i]);
        else if (!strcmp(argv[i], "--valid"))     config->set_valid(argv[++i]);
        else if (!strcmp(argv[i], "--result"))    config->set_result(argv[++i]);
        else if (!strcmp(argv[i], "--model"))     config->set_model(argv[++i]);
        else if (!strcmp(argv[i], "--alg"))       config->set_alg(argv[++i]);
        else if (!strcmp(argv[i], "--dim"))       config->set_dim(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--iter"))      config->set_iterations(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--nu"))        config->set_nu(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--nv"))        config->set_nv(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--fly"))       config->set_fly(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--stride"))    config->set_prefetch_stride(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--lr"))        config->set_learning_rate(atof(argv[++i]));
        else if (!strcmp(argv[i], "--lambda"))    config->set_regularizer(atof(argv[++i]));
        else if (!strcmp(argv[i], "--gam"))       config->set_learning_rate_decay(atof(argv[++i]));
        else if (!strcmp(argv[i], "--bias"))      config->set_global_bias(atof(argv[++i]));
        else if (!strcmp(argv[i], "--mineta"))    config->set_min_learning_rate(atof(argv[++i]));
        else if (!strcmp(argv[i], "--epsilon"))   config->set_dfp_sensitivity(atof(argv[++i]));
        else if (!strcmp(argv[i], "--max-ratings")) config->set_max_ratings(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--hypera"))    config->set_hypera(atof(argv[++i]));
        else if (!strcmp(argv[i], "--hyperb"))    config->set_hyperb(atof(argv[++i]));
        else if (!strcmp(argv[i], "--temp"))      config->set_sgld_temperature(atof(argv[++i]));
        else if (!strcmp(argv[i], "--noise-size")) config->set_noise_size(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--lr-reg"))    config->set_learning_rate_reg(atof(argv[++i]));
        else if (!strcmp(argv[i], "--loss"))      config->set_loss(atoi(argv[++i]));
        else if (!strcmp(argv[i], "--measure"))   config->set_measure(atoi(argv[++i]));
        else {
          printf("%s, unknown parameters, exit\n", argv[i]);
          return 1;
        }
      }
      if (!config->has_train() || !config->has_nu() || !config->nu()
          || !config->has_nv() || !config->nv()) {
        printf("Note that train_data/#users/#items are not optional!\n");
        show_help();
        return 1;
      }
      mf::perf::TimedScope timedScope;
      if (!config->has_alg() || config->alg().empty() || config->alg() == "mf") {
        MF mf(config);
        rc = run(mf);
      } else if (config->alg() == "dpmf") {
        DPMF dpmf(config);
        rc = run(dpmf);
      } else if (config->alg() == "admf") {
        AdaptRegMF admf(config);
        rc = run(admf);
      } else {
        printf("Please select a solver: mf | dpmf | admf\n");
        rc = EINVAL;
      }
      if (rc) {
        fprintf(stderr, "Error: %s\n", strerror(rc));
      }
    } else {
      LOG(ERROR) << "Could not create default configuration";
      rc = EINVAL;
      break;
    }
  } while(false);
  if(rc) {
    LOG(ERROR) << "Error: " << strerror(rc) << std::endl << std::flush;
  } else {
#ifdef USE_JEMALLOC
    //malloc_stats_print(NULL, NULL, NULL);
#endif
  }
  return rc;
}
