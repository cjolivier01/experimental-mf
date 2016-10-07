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
    size_t lastCount = 0;
    Record record;
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> randomFile(0, (int)files.size());
    while(!rc && !files.empty()) {
      if(files.size() == 1) {
        ++lastCount; // make sure we aren't getting a large # of items from the last file
      }
      const int fileNo = files.size() > 1 ? randomFile(generator) : 0;
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
    std::cout << lastCount << " items from last file\n" << std::flush;
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