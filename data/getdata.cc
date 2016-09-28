#include <iostream>
#include <fstream>
#include <ctime>
#include <list>
#include <locale>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <unistd.h>
#include "util.h"
#include "blocks.pb.h"

#define OFSTREAM_T   std::ofstream

#if 1 && !defined(NDEBUG)
#define DEFAULT_MAX_RECORDS       ((uint64_t)(1e6))
#else
#define DEFAULT_MAX_RECORDS       ((uint64_t)(10e6))
#endif

typedef std::unordered_map<int, std::vector<std::pair<int, float>>> Dict;
typedef Dict::const_iterator DictIt;
typedef std::tuple<int, int, float> Tuple;

static int block_size = 1000;

template <class Record>
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

  typedef int (*get_record_t)(std::ifstream& input, Record& buffer);

  template<typename OutStream>
  static int random_merge(std::vector<std::unique_ptr<std::ifstream>> &files,
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
      std::ifstream& input = *files[fileNo];
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
    std::vector<std::unique_ptr<std::ifstream>> streams;
    for(const std::string& fileName : files) {
      if(!fileName.empty()) {
        streams.push_back(std::unique_ptr<std::ifstream>(new std::ifstream(fileName)));
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
        OFSTREAM_T outStream(outFile);
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

static std::vector<std::string>& tokenize(const std::string& str,
                                          const std::string& delimiters,
                                          std::vector<std::string>& tokens) {
  tokens.reserve((tokens.size() + 4) << 1);
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
  return tokens;
}

static int read_raw(std::ifstream& ins, std::vector<Tuple>& data, size_t maxRecords) {
  std::string line;
  while(std::getline(ins, line) && !line.empty() && line[0] == '%')
    ;
  std::vector<std::string> tokens;
  tokenize(line, " ,\t", tokens);
  const size_t nn = !tokens.empty() ? atoi((*tokens.rbegin()).c_str()) : 0;
  data.reserve(nn);
  while(std::getline(ins, line) && data.size() < maxRecords) {
    tokens.clear();
    tokens.reserve(5);
    tokenize(line, " ,\t", tokens);
    if(tokens.size() >= 3) {
      data.push_back(std::make_tuple(
        atoi(tokens[0].c_str()),
        atoi(tokens[1].c_str()),
        atof(tokens[2].c_str()))
      );
    }
  }
  std::random_shuffle(data.begin(), data.end());
  std::random_shuffle(data.begin(), data.end());
  std::random_shuffle(data.begin(), data.end());
  std::random_shuffle(data.begin(), data.end());
  return 0;
}

static int read_raw(const char* file, std::vector<Tuple>& data, const size_t maxRecords) {
  std::ifstream ins(file);
  if(!ins.is_open())  {
    fprintf(stderr, "Unable to open file for read: %s - %s\n", file, strerror(errno));
    return errno;
  }
  return read_raw(ins, data, maxRecords);
}

static void write_by_dict(Dict& du, FILE* fp) {
  for(DictIt it(du.begin()); it!=du.end(); it++) {
    int u = it->first;
    fprintf(fp, "%d:\n", u);
    auto eles = it->second;
    for(std::vector<std::pair<int, float>>::iterator vit = eles.begin(), e = eles.end();
        vit != e; ++vit) {
      const int   v = vit->first;
      const float r = vit->second;
      fprintf(fp, "%d,%f\n", v, r);
    }
  }
}

static int userwise(const char* write,
                    std::vector<Tuple>& data,
                    size_t itemsPerSplit,
                    size_t numRemainingAfterSplit,
                    size_t numRatingMatrixSplits) {
  int rc = 0;
  if(!data.empty() && itemsPerSplit && numRatingMatrixSplits) {
    size_t i;
    FILE* fp = fopen(write, "w");
    if (fp) {

      for (i = 0; i < numRatingMatrixSplits - 1; ++i) {
        Dict du;
        const size_t count = i * itemsPerSplit + itemsPerSplit;
        for (size_t j = i * itemsPerSplit; j < count; ++j) {
          const auto &ele = data[j];
          const int u = std::get<0>(ele);
          const int v = std::get<1>(ele);
          const float r = std::get<2>(ele);
          du[u].push_back(std::make_pair(v, r));
        }
        // write to output file
        write_by_dict(du, fp);
      }
      // make the last one hold the remainder
      Dict du;
      const size_t count = i * itemsPerSplit + itemsPerSplit + numRemainingAfterSplit;
      CHECK_LE(count, data.size());
      for (size_t j = i * itemsPerSplit; j < count; ++j) {
        const auto &ele = data[j];
        const int u = std::get<0>(ele);
        const int v = std::get<1>(ele);
        const float r = std::get<2>(ele);
        du[u].push_back(std::make_pair(v, r));
      }
      // write to output file
      write_by_dict(du, fp);
      fclose(fp);
    } else {
      rc = errno;
      fprintf(stderr, "Unable to open file for write: %s - %s\n", write, strerror(rc));
    }
  } else {
    fprintf(stderr, "Invalid argument(s)\n");
    rc = EINVAL;
  }
  return rc;
}

struct RangeConverter
{
  struct Range
  {
    int low_, hi_; // inclusive
    int range() const { return hi_ - low_; }
  };
  Range from_;
  Range to_;
  bool scale(const float fIn, float& fOut) const {
    CHECK_GT(from_.hi_, from_.low_) << "Invalid range: low = " <<
                                    from_.low_ << ", high = " << from_.hi_;
    CHECK_GT(to_.hi_, to_.low_)     << "Invalid range: low = " <<
                                    to_.low_ << ", high = " << to_.hi_;
    if(fIn < from_.low_ || fIn > from_.hi_) {
      return false;
    }
    const float pct = (fIn - from_.low_) / from_.range();
    fOut = (pct * to_.range()) + to_.low_;
    return true;
  }
};

static int get_message(const char* read, const char* write, const RangeConverter& ranges)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  std::ifstream ins(read);
  if(!ins.is_open()) {
    fprintf(stderr, "Unable to open file for read: %s - %s\n", read, strerror(errno));
    return errno;
  }
  std::string buf;
  FILE* f_w = fopen(write, "wb");
  if(!f_w)  {
    fprintf(stderr, "Unable to open file for write: %s - %s\n", write, strerror(errno));
    return errno;
  }
  int rc = 0;
  size_t ii=0;
  ::google::protobuf::int32 vid;
  mf::Block block;
  mf::User* user = NULL;
  mf::User_Record* record = NULL;
  std::string uncompressed_buffer;
  while (std::getline(ins, buf)) {
    const size_t len = buf.length();
    if(buf[len-1]==':') {
      if(ii % block_size == 0) {
        if(block.user_size() > 0) {
          if(block.SerializeToString(&uncompressed_buffer)) {
            const uint32 uncompressed_size = uncompressed_buffer.size();
            fwrite(&uncompressed_size, 1, sizeof(uncompressed_size), f_w);
            fwrite(uncompressed_buffer.c_str(), 1, uncompressed_size, f_w);
          } else {
            fprintf(stderr, "Error serializing block to stream\n");
            rc = EINVAL;
            break;
          }
        }
        block.Clear();
        user = block.add_user();
      }
      else {
        user = block.add_user();
      }
      ++ii;
      user->set_uid(stoi(buf));
      continue;
    }
    if(!user) {
      fprintf(stderr, "Found data before user record: \"%s\"\n", buf.c_str());
      break;
    }
    record = user->add_record();
    float rating;
    if(sscanf(buf.c_str(), "%d,%f", &vid, &rating) != 2) {
      fprintf(stderr, "Bad input line: %s\n", buf.c_str());
      rc = 1;
      break;
    }
    record->set_vid(vid);

    float scaledRating;
    if(!ranges.scale(rating, scaledRating)) {
      LOG(WARNING) << "Out of range input score: " << rating;
      continue;
    }
    CHECK_GE(scaledRating, ranges.to_.low_);
    CHECK_LE(scaledRating, ranges.to_.hi_);
    record->set_rating(scaledRating);
  }
  if(!rc) {
    block.SerializeToString(&uncompressed_buffer);
    const uint32 uncompressed_size = uncompressed_buffer.size();
    fwrite(&uncompressed_size, 1, sizeof(uncompressed_size), f_w);
    fwrite(uncompressed_buffer.c_str(), 1, uncompressed_size, f_w);
    fclose(f_w);
  } else {
    fclose(f_w);
    unlink(write);
  }
  google::protobuf::ShutdownProtobufLibrary();
  return rc;
}

class TempFile
{
  std::string name;
 public:
  TempFile(const char *templ) {
    if(templ) {
      name =  "/tmp/";
      name += templ;
      name += "XXXXXX";
      mkstemp((char *)name.c_str());
    }
  }
  void reset() {
    name.clear();
  }
  const std::string& getName() {
    return name;
  }
  ~TempFile() {
    if(!name.empty()) {
      unlink(name.c_str());
    }
  }
};

static int raw_to_protobuf(const char *fileIn,
                           const char *fileOut,
                           const size_t numRatingMatrixSplits,
                           const size_t maxRecords,
                           const RangeConverter& ranges)
{
  RandomMerger<std::string>::seedRandom();
  int rc = 0;
  std::ifstream inFile(fileIn);
  if(!inFile.is_open()) {
    fprintf(stderr, "Unable to open input file: %s - %s\n", fileIn, strerror(errno));
    return errno;
  }
  uint64_t totalCount = 0;
  std::list<std::unique_ptr<TempFile>> tempFiles;
  while(!rc && !inFile.eof()) {
    tempFiles.push_back(std::unique_ptr<TempFile>(new TempFile("raw2proto")));
    const std::string& outFileName = (*tempFiles.rbegin())->getName();
    if(!outFileName.empty()) {
      std::vector<Tuple> data;
      rc = read_raw(inFile, data, maxRecords);
      if(!rc) {
        totalCount += data.size();
        const size_t dataItemCount = data.size();
        size_t itemsPerSplit = dataItemCount / numRatingMatrixSplits;
        size_t numRemainingAfterSplit = dataItemCount % numRatingMatrixSplits;
        rc = userwise(outFileName.c_str(), data, itemsPerSplit, numRemainingAfterSplit, numRatingMatrixSplits);
        if(!rc) {
          std::cout << "Processed " << totalCount << " items...." << std::endl << std::flush;
        }
      }
    } else {
      rc = ENOENT;
    }
  }
  if(!rc) {
    // merge any temp files into destination file
    std::vector<std::string> files;
    files.reserve(tempFiles.size());
    for(const std::unique_ptr<TempFile>& ff : tempFiles) {
      files.push_back(ff->getName());
    }
    TempFile userwiseStage("userwise");
    rc = RandomMerger<std::string>::random_merge(files, userwiseStage.getName().c_str(),
                                                 [](std::ifstream& input, std::string& buffer) {
                                                    buffer.clear();
                                                    std::getline(input, buffer);
                                                    return 0;
                                                  }
    );
    if(!rc) {
      rc = get_message(userwiseStage.getName().c_str(), fileOut, ranges);
    }
  }
  return rc;
}

static void hint() {
  fprintf(stderr, "-i             [input_file_name]\n");
  fprintf(stderr, "-o             [output_file_name]\n");
  fprintf(stderr, "--stage-size   [number of records to split up between stages (for huge files). Default: %" PRIu64 "\n", DEFAULT_MAX_RECORDS);
  fprintf(stderr, "--method       [userwise/raw2proto/protobuf]\n");
  fprintf(stderr, "--split        [number_of_splits_for_rating_matrix]\thints: 1~10 splits are recommended\n");
  fprintf(stderr, "--size         [number_of_users_in_each_block]\thints: 1 fread reads 1 block each time\n");
  fprintf(stderr, "--input-range  [input score inclusive range. Default: 0,5 \n");
  fprintf(stderr, "--output-range [output score inclusive range. low,high. Default: 0,5 \n");
}

int main(int argc, char** argv) {
  const char *read   = NULL;
  const char *write  = NULL;
  const char *method = NULL;

  RangeConverter ranges;
  ranges.from_.low_ = ranges.to_.low_ = 0;
  ranges.from_.hi_  = ranges.to_.hi_  = 5;

  size_t numRatingMatrixSplits = 1;
  int rc = 0;
  size_t stageSize = DEFAULT_MAX_RECORDS;
  for(int i = 1; !rc && i < argc; ++i) {
    const char *option = argv[i];
    if (!strcmp(option, "-i")) {
      if (++i < argc) {
        read = argv[i];
      }
    }
    else if (!strcmp(option, "-o")) {
      if (++i < argc) {
        write = argv[i];
      }
    }
    else if (!strcmp(option, "--method")) {
      if (++i < argc) {
        method = argv[i];
      }
    }
    else if (!strcmp(option, "--split")) {
      if (++i < argc) {
        numRatingMatrixSplits = atoi(argv[i]);
      }
    }
    else if (!strcmp(option, "--size")) {
      if (++i < argc) {
        block_size = atoi(argv[i]);
      }
    }
    else if(!strcmp(option, "--stage-size")) {
      if (++i < argc) {
        stageSize = atoi(argv[i]);
        if(!stageSize) {
          std::cerr << "Invalid stage size: " << stageSize << std::endl;
          rc = EINVAL;
        }
      }
    } else if(!strcmp(option, "--input-range")) {
      if (++i < argc) {
        const std::string s = argv[i];
        int lo, hi;
        if(s.empty() || sscanf(s.c_str(), "%d,%d", &lo, &hi) != 2 || lo >= hi) {
          std::cerr << "Invalid input range: " << s << std::endl;
          rc = EINVAL;
        } else {
          ranges.from_.low_ = lo;
          ranges.from_.hi_  = hi;
        }
      }
    } else if(!strcmp(option, "--output-range")) {
      if (++i < argc) {
        const std::string s = argv[i];
        int lo, hi;
        if(s.empty() || sscanf(s.c_str(), "%d,%d", &lo, &hi) != 2 || lo >= hi) {
          std::cerr << "Invalid output range: " << s << std::endl;
          rc = EINVAL;
        } else {
          ranges.to_.low_ = lo;
          ranges.to_.hi_  = hi;
        }
      }
    }
    else {
      fprintf(stderr, "Unknown parameter: \"%s\"\n\n", option);
      hint();
      rc = 1;
      break;
    }
  }
  if(!rc) {
    if (read == NULL || write == NULL || method == NULL) {
      fprintf(stderr, "Please indicate at least the input, output and method.\n\n");
      hint();
      rc = 1;
    } else {
      const uint64_t startTime = mf::perf::getTickCount();
      if(!strcmp(method, "raw2proto")) {
        rc = raw_to_protobuf(read, write, numRatingMatrixSplits, stageSize, ranges);
      } else if (!strcmp(method, "userwise")) {
        std::vector<Tuple> data;
        rc = read_raw(read, data, ULONG_LONG_MAX);
        if(!rc) {
          const size_t nn = data.size();
          const size_t itemsPerSplit = nn / numRatingMatrixSplits;
          const size_t numRemainingAfterSplit = nn % numRatingMatrixSplits;
          std::string tmp = write;
          rc = userwise(tmp.c_str(), data, itemsPerSplit, numRemainingAfterSplit, numRatingMatrixSplits);
        }
      } else if (!strcmp(method, "protobuf")) {
        rc = get_message(read, write, ranges);
      } else {
        rc = 1;
      }
      std::cout << "Process time: " << (mf::perf::getTickCount() - startTime) << " ms" << std::endl;
    }
  }
  return rc;
}
