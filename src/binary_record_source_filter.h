#ifndef _FASTMF_BINARY_RECORD_SOURCE_FILTER_H
#define _FASTMF_BINARY_RECORD_SOURCE_FILTER_H

#include "filter_util.h"

namespace mf
{

/**
 * BinaryRecordSourceFilter
 *
 * Pass binary records from source to tbb pipeline
 */
class BinaryRecordSourceFilter  : public mf::ObjectPool< std::vector<char> >,
                                  public PipelineFilter {
 public:
  BinaryRecordSourceFilter(const size_t bufferCount,
                           dmlc::SeekStream *fr,
                           awsdl::perf::TimingInstrument *timing_ref)
    : mf::ObjectPool<std::vector<char> >(bufferCount)
      , PipelineFilter(serial_in_order)
      , fr_(fr)
      , stream_(new dmlc::istream(fr, STREAM_BUFFER_SIZE))
      , pass_(0)
      , timing_ref_(timing_ref) {
  }

  /**
   * Called when end of file reached
   * @return true:  seek back to the beginning of the file
   *                and start again from the first record
   *         false: finished with source stream
   */
  virtual bool onSourceStreamComplete() = 0;

  virtual void *execute(void *) {
    std::chrono::time_point<Time> entryTime = Time::now();
    if(!pass_++) {
      s_ = Time::now();
      in_time_ = in_time_.zero();
    }
    IF_CHECK_TIMING( awsdl::perf::TimingItem inFunc(timing_ref_, FILTER_STAGE_READ, "FILTER_STAGE_READ") );
    std::vector<char> *pbuffer = allocateObject();
    if (pbuffer) {
      int isize = 0;
      if (!stream_->read((char *)&isize, sizeof(isize)).fail()) {
        pbuffer->resize(isize);
        if (!stream_->read(pbuffer->data(), isize).fail()) {
          in_time_ += Time::now() - entryTime;
          return pbuffer;
        }
        addStatus(IO_ERROR, "Error reading input data object");
        freeObject(pbuffer);
      } else {
        if(stream_->eof()) {
          if(onSourceStreamComplete()) {
            stream_.reset();
            fr_->Seek(0);
            pass_ = 0;
            stream_ = std::unique_ptr<dmlc::istream>(new dmlc::istream(fr_, STREAM_BUFFER_SIZE));
            if (!stream_->read((char *) &isize, sizeof(isize)).fail()) {
              pbuffer->resize(isize);
              if (!stream_->read(pbuffer->data(), isize).fail()) {
                in_time_ += Time::now() - entryTime;
                return pbuffer;
              }
            }
            addStatus(IO_ERROR, "Error reading input data object");
          }
        }
        else {
          addStatus(IO_ERROR);
        }
      }
      freeObject(pbuffer);
    } else {
      addStatus(POOL_ERROR);
    }
    in_time_ += Time::now() - entryTime;
    return NULL;
  }

 protected:
  dmlc::SeekStream *              fr_;
  std::unique_ptr<dmlc::istream>  stream_;
  std::atomic<unsigned long>      pass_;
  std::chrono::time_point<Time>   s_;
  std::chrono::duration<float>    in_time_;
 public:
  awsdl::perf::TimingInstrument * timing_ref_;
};


} // namespace mf

#endif //_FASTMF_BINARY_RECORD_SOURCE_FILTER_H