#include "../src/model.h"
#include "../src/mf.h"
#include "../src/dpmf.h"
#include "../src/admf.h"

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
  printf("--eta        [float]   : learning rate.\n");
  printf("--lambda     [float]   : regularizer.\n");
  printf("--gam        [float]   : decay of learning rate.\n");
  printf("--bias       [float]   : global bias (important for accuracy).\n");
  printf("--mineta     [float]   : minimum learning rate (sometimes used in SGLD).\n");
  printf("--epsilon    [float]   : sensitivity of differentially privacy.\n");
  printf("--tau        [int]     : maximum of ratings among all the users (usually after trimming your data).\n");
  printf("--temp       [float]   : temperature in SGLD (can accelerate the convergence).\n");
  printf("--noise_size [int]     : the Gaussian numbers lookup table.\n");
  printf("--eta_reg    [float]   : the learning rate for estimating regularization parameters.\n");
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
  if(mf.model_ != NULL) {
    mf.read_model();
  }

  // coolivie: TEMPORARY
  //mf.data_in_fly_ = 1;

  mf::Blocks blocks_test;
  rc = plain_read(mf.test_data_, blocks_test);
  if(!rc) {
    FILE *f = fopen(mf.train_data_, "rb");
    if (f) {
      SgdReadFilter read_f(mf, f, blocks_test);
      ParseFilter parse_f(mf.data_in_fly_, read_f);
      SgdFilter sgd_f(mf, parse_f);
      tbb::pipeline p;
      p.add_filter(read_f);
      p.add_filter(parse_f);
      p.add_filter(sgd_f);
      // Check errors in reverse order
      p.run(mf.data_in_fly_);
      fclose(f);
      setOnError(sgd_f, rc);
      setOnError(parse_f, rc);
      setOnError(read_f, rc);
    } else {
      rc = errno;
    }
  }
  return rc;
}

//assuming test data can fit into RAM
static int run(DPMF& dpmf) {
  int rc = 0;
  dpmf.init();
  if(dpmf.model_ != NULL) {
    dpmf.read_hyper();
  }
  mf::Blocks blocks_test;
  rc = plain_read(dpmf.test_data_, blocks_test);
  if(!rc) {
    FILE *f = fopen(dpmf.train_data_, "rb");
    if (f) {
      SgldReadFilter read_f(dpmf, f);
      ParseFilter parse_f(dpmf.data_in_fly_, read_f);
      SgldFilter sgld_f(dpmf);
      tbb::pipeline p;
      p.add_filter(read_f);
      p.add_filter(parse_f);
      p.add_filter(sgld_f);
      const std::chrono::time_point<Time> s = Time::now();
      for (int i = 1; i <= dpmf.iter_; i++) {
        p.run(dpmf.data_in_fly_);
        dpmf.finish_round(blocks_test, i, s);
      }
      fclose(f);
      setOnError(sgld_f, rc);
      setOnError(parse_f, rc);
      setOnError(read_f, rc);
    } else {
      rc = errno;
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
    admf.plain_read_valid(admf.valid_data_);
    FILE *f = fopen(admf.train_data_, "rb");
    if (f) {
      AdRegReadFilter read_f(admf, f, blocks_test);
      ParseFilter parse_f(admf.data_in_fly_, read_f);
      AdRegFilter admf_f(admf);
      tbb::pipeline p;
      p.add_filter(read_f);
      p.add_filter(parse_f);
      p.add_filter(admf_f);
      p.run(admf.data_in_fly_);
      fclose(f);
      setOnError(admf_f, rc);
      setOnError(parse_f, rc);
      setOnError(read_f, rc);
    } else {
      rc = errno;
    }
  }
  return rc;
}

int main(int argc, char** argv) {
  char *train_data=NULL, *test_data=NULL, *result=NULL, *alg=NULL, *model=NULL;
  int dim = 128, iter = 15, tau=0, nu=0, nv=0, fly=8, stride=2;
  float eta = 2e-2, lambda = 5e-3, gam=1.0f, mineta=1e-13;
  float epsilon=0.0f, hypera=1.0f, hyperb=100.0f, temp=1.0f;
  float g_bias = 2.76f;
  int noise_size = 2000000000;
  int loss = 0;
  int measure = 0;
  float eta_reg = 2e-3f;
  char* valid_data=NULL;
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "--train"))            train_data = argv[++i];
    else if(!strcmp(argv[i], "--test"))        test_data = argv[++i];
    else if(!strcmp(argv[i], "--valid"))       valid_data = argv[++i];
    else if(!strcmp(argv[i], "--result"))      result = argv[++i];
    else if(!strcmp(argv[i], "--model"))       model = argv[++i];
    else if(!strcmp(argv[i], "--alg"))         alg = argv[++i];
    else if(!strcmp(argv[i], "--dim"))         dim  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--iter"))        iter = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--nu"))          nu  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--nv"))          nv  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--fly"))         fly  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--stride"))      stride  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--eta"))         eta = atof(argv[++i]);
    else if(!strcmp(argv[i], "--lambda"))      lambda = atof(argv[++i]);
    else if(!strcmp(argv[i], "--gam"))         gam = atof(argv[++i]);
    else if(!strcmp(argv[i], "--bias"))        g_bias = atof(argv[++i]);
    else if(!strcmp(argv[i], "--mineta"))      mineta = atof(argv[++i]);
    else if(!strcmp(argv[i], "--epsilon"))     epsilon = atof(argv[++i]);
    else if(!strcmp(argv[i], "--tau"))         tau = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--hypera"))      hypera = atof(argv[++i]);
    else if(!strcmp(argv[i], "--hyperb"))      hyperb = atof(argv[++i]);
    else if(!strcmp(argv[i], "--temp"))        temp = atof(argv[++i]);
    else if(!strcmp(argv[i], "--noise_size"))  noise_size  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--eta_reg"))     eta_reg = atof(argv[++i]);
    else if(!strcmp(argv[i], "--loss"))        loss  = atoi(argv[++i]);
    else if(!strcmp(argv[i], "--measure"))     measure  = atoi(argv[++i]);
    else {
      printf("%s, unknown parameters, exit\n", argv[i]);
      return 1;
    }
  }
  if (train_data == NULL || nu == 0 || nv == 0) {
    printf("Note that train_data/#users/#items are not optional!\n");
    show_help();
    return 1;
  }
  int rc = 0;
  TimedScope timedScope;
  if(!alg || !*alg || !strcmp(alg, "mf")) {
    MF mf(train_data, test_data, result, model, dim, iter, eta, gam, lambda, \
              g_bias, nu, nv, fly, stride);
    rc = run(mf);
  }
  else if (!strcmp(alg, "dpmf")) {
    DPMF dpmf(train_data, test_data, result, model, dim, iter, eta, gam, lambda, \
                  g_bias, nu, nv, fly, stride, hypera, hyperb, epsilon, tau, \
                  noise_size, temp, mineta);
    rc = run(dpmf);
  }
  else if (!strcmp(alg, "admf")) {
    AdaptRegMF admf(train_data, test_data, valid_data, result, model, dim, iter, eta, gam, \
                        lambda, g_bias, nu, nv, fly, stride, loss, measure, eta_reg);
    rc = run(admf);
  }
  else {
    printf("Please select a solver: mf | dpmf | admf\n");
    rc = EINVAL;
  }
  if(rc) {
    fprintf(stderr, "Error: %s\n", strerror(rc));
  }
  return rc;
}
