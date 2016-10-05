#ifndef _FASTMF_RANDOM_MERGER_H
#define _FASTMF_RANDOM_MERGER_H

/**
 * RandomMerger
 *
 * Merge records from several input streams in random order into one large output file
 * It is presumed that each input streams is in random order
 */
template <typename Record, typename ifstream_t, typename ofstream_t>
class RandomMerger
{
 public:
  static void seedRandom() {
    std::srand(time(NULL) * getpid());
  }

  static int random_range(const int floor, const int ceiling) {
    const int range = ceiling - floor;
    const int rnd = floor + int((range * rand()) / (RAND_MAX + 1.0));
    return rnd;
  }

  typedef int (*get_record_t)(ifstream_t& input, Record& buffer);

  template<typename OutStream>
  static int random_merge(std::vector<std::unique_ptr<ifstream_t>> &files,
                          OutStream &outFile,
                          get_record_t get_next,
                          const char *endOfRecord = "\n"
  ) {
    if(files.empty() && !get_next) {
      return EINVAL;
    }
    int rc = 0;
    Record record;
    while(!rc && !files.empty()) {
      const int fileNo = files.size() > 1 ? random_range(0, files.size() - 1) : 0;
      ifstream_t& input = *files[fileNo];
      rc = get_next(input, record);
      if(!rc) {
        if (!record.empty()) {
          outFile << record;
          if(endOfRecord) {
            outFile << endOfRecord;
          }
        }
        if (input.eof()) {
          files.erase(files.begin() + fileNo);
        }
      }
    }
    return rc;
  }

  // Randomly merge files (It is assumed that the filed have already been randomly shuffled)
  static int random_merge(const std::vector<std::string> &files,
                          const std::string &outFile,
                          get_record_t get_next,
                          const char *endOfRecord = "\n") {
    int rc = 0;
    // Open the files
    std::vector<std::unique_ptr<ifstream_t>> streams;
    for(const std::string& fileName : files) {
      if(!fileName.empty()) {
        streams.push_back(std::unique_ptr<ifstream_t>(new ifstream_t(fileName)));
        if((*streams.rbegin())->is_open()) {
        } else {
          rc = errno;
        }
      } else {
        rc = EINVAL;
      }
      if(rc) {
        break;
      }
    }
    if(!rc) {
      if (!outFile.empty()) {
        ofstream_t outStream(outFile);
        if (outStream.is_open()) {
          // Randomly merge the streams
          return random_merge(streams, outStream, get_next);
        } else {
          rc = errno;
        }
      } else {
        rc = EINVAL;
      }
    }
    return rc;
  }
 private:
};


#endif //_FASTMF_RANDOM_MERGER_H