#ifndef _FASTMF_UTIL_H
#define _FASTMF_UTIL_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <tbb/pipeline.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "perf.h"

#include "blocks.pb.h"
#ifdef __APPLE__
extern "C"
{
#include <cblas.h>
}
extern "C"
{
void cblas_saxpy(const int N, const float alpha, const float *X,
const int incX, float *Y, const int incY);
void cblas_scopy(const int N, const float *X, const int incX,
float *Y, const int incY);
float cblas_sdot(const int N, const float  *X, const int incX,
const float  *Y, const int incY);
}
#else
#include "mkl.h"
#endif
typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef std::chrono::high_resolution_clock Time;
#ifdef LEVEL1_DCACHE_LINESIZE
#define CACHE_LINE_SIZE LEVEL1_DCACHE_LINESIZE
#else
#define CACHE_LINE_SIZE 64
#endif

namespace mf
{

constexpr size_t STREAM_BUFFER_SIZE = (1 << 20);

/**
 * Wrap simple R/W dmlc stream creation pattern to programmatically
 * match ifstream/ofstream creation pattern
 */
template<typename StreamType, bool WRITING>
class DmlcStream : public StreamType
{
  std::unique_ptr<dmlc::Stream> dmlc_stream;
 public:
  DmlcStream(const std::string &name)
    : StreamType(NULL, mf::STREAM_BUFFER_SIZE) {
    if (!WRITING) {
      dmlc_stream.reset(dmlc::SeekStream::CreateForRead(name.c_str()));
    } else {
      dmlc_stream.reset(dmlc::SeekStream::Create(name.c_str(), "wb", true));
    }
    this->set_stream(dmlc_stream.get());
  }

  inline bool is_open() const {
    return !!dmlc_stream.get();
  }

  virtual ~DmlcStream() {
    this->set_stream(NULL);
  }
};

template<typename StreamType>
class DmlcOutStream : public DmlcStream<StreamType, true>
{
 public:
  DmlcOutStream(const std::string &name)
    : DmlcStream<StreamType, true>(name) {}

  ~DmlcOutStream() { this->flush(); }
};


typedef DmlcStream<dmlc::istream, false> dmlc_istream;
typedef DmlcOutStream<dmlc::ostream> dmlc_ostream;

typedef struct
{
  int u_, v_;
  float r_;
} Record;

extern std::default_random_engine generator;
extern std::normal_distribution<float> gaussian;

#ifdef FETCH
inline void prefetch_range(char *addr, size_t len) {
char *cp;
char *end = addr + len;

for (cp = addr; cp < end; cp += CACHE_LINE_SIZE)
__builtin_prefetch(cp,1,0);
}
#endif

inline void align_alloc(float **u, int nu, int dim) {
  int piece = nu / 1050000 + 1;
  int nn = nu / piece;
  int k;
  for (k = 0; k < piece - 1; k++) {
    u[k * nn] = (float *) mkl_malloc(nn * dim * sizeof(float), CACHE_LINE_SIZE);
    for (int i = 1; i < nn; i++) {
      u[k * nn + i] = u[k * nn + i - 1] + dim;
    }
  }
  u[k * nn] = (float *) mkl_malloc((nn + nu % piece) * dim * sizeof(float), CACHE_LINE_SIZE);
  for (int i = 1; i < nn + nu % piece; i++) {
    u[k * nn + i] = u[k * nn + i - 1] + dim;
  }
}

inline void free_aligned_alloc(float **u, int nu) {
  const int piece = nu / 1050000 + 1;
  const int nn = nu / piece;
  int k;
  for (k = 0; k < piece - 1; k++) {
    mkl_free(u[k * nn]);
  }
  mkl_free(u[k * nn]);
}

inline int plain_read(const std::string &data, mf::Blocks &blocks) {
  int rc = 0;
  if (!data.empty()) {
    mf::dmlc_istream input(data);
    if (input.is_open()) {
      std::vector<char> buf;
      uint32 isize;
      mf::Block *bk;
      while (!input.read((char *) &isize, sizeof(isize)).fail()) {
        buf.resize(isize);
        if (!input.read(buf.data(), isize).fail()) {
          bk = blocks.add_block();
          if (!bk->ParseFromArray(buf.data(), isize)) {
            rc = EINVAL;
            break;
          }
        } else {
          rc = EIO;
          break;
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

inline float active(float val, int type) {
  switch (type) {
    case 0:
      return val;                     //least square
    case 1:
      return 1.0f / (1.0f + expf(-val));  //sigmoid
  }
  CHECK(false); // Should not get here
  return 0.0f;
}

inline float cal_grad(float r, float pred, int type) {
  switch (type) {
    case 0:
      return r - pred;          //least square
    case 1:
      return r - pred;          //0-1 logistic regression
  }
  CHECK(false); // Should not get here
  return 0.0f;
}

inline float next_float() {
  return static_cast<float>( rand()) / (static_cast<float>( RAND_MAX ) + 1.0f);
}

inline float next_float2() {
  return (static_cast<float>( rand()) + 1.0f) / (static_cast<float>(RAND_MAX) + 2.0f);
}

inline float normsqr(float *x, int num) {
  return cblas_sdot(num, x, 1, x, 1);
}

inline float sample_normal() {
  float x, y, s;
  do {
    x = 2 * next_float2() - 1.0f;
    y = 2 * next_float2() - 1.0f;
    s = x * x + y * y;
  } while (s >= 1.0 || s == 0.0);

  return (float) (x * sqrt(-2.0f * log(s) / s));
}

inline float sample_gamma(float alpha, float beta) {
  if (alpha < 1.0) {
    float u;
    do {
      u = next_float();
    } while (u == 0.0);
    return (float) (sample_gamma(alpha + 1.0f, beta) * pow(u, 1.0f / alpha));
  } else {
    float d, c, x, v, u;
    d = alpha - 1.0f / 3.0f;
    c = 1.0f / (float) sqrt(9.0f * d);
    do {
      do {
        x = sample_normal();
        v = 1.0f + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      u = next_float();
    } while ((u >= (1.0 - 0.0331 * (x * x) * (x * x)))
             && (log(u) >= (0.5 * x * x + d * (1.0 - v + log(v)))));
    return d * v / beta;
  }
}

inline void gamma_posterior(float &lambda, float prior_alpha, float prior_beta, float psum_sqr, float psum_cnt) {
  float alpha = prior_alpha + 0.5f * psum_cnt;
  float beta = prior_beta + 0.5f * psum_sqr;
  lambda = sample_gamma(alpha, beta);
}

inline void normsqr_col(float **m, int d, int size, float *norm) {
#pragma omp parallel for
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < size; j++) norm[i] += m[j][i] * m[j][i];
  }
}

inline int padding(int dim) {
  return ((dim * sizeof(float) - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE + CACHE_LINE_SIZE) / sizeof(float);
}

/**
 * isFinite()
 *      Check is the given value is finite (not NAN or infinite, etc.
 *      For debugging, it's sometimes useful to put another "reasonable check" in here
 *      such as some ridiculously large value which indicates a calculation or data bug
 * @param value to check
 * @return true if the value is reasonable
 */
inline bool isFinite(const float &f) {
  if (fabs(f) > 1e10) {
    return false;
  }
  return std::isfinite(f);
}

/**
 * isNan()
 *  Check explicitly for not-a-number
 * @param value to check
 * @return true if the value is not a number
 */
inline bool isNan(const float &f) {
  // nan has special property that it doesn't equal itself
  // This can't be trusted with all compilers in release mode
  return f != f;
}

} // namespace mf

#endif //_FASTMF_UTIL_H
