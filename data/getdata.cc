#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <unistd.h>
#include "blocks.pb.h"

typedef std::unordered_map<int, std::vector<std::pair<int, float>>> Dict;
typedef Dict::const_iterator DictIt;
typedef std::tuple<int, int, float> Tuple;

static int block_size = 1000;

static std::vector<std::string>& tokenize(const std::string& str,
                                          const std::string& delimiters,
                                          std::vector<std::string>& tokens
) {
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

static int read_raw(const char* file, std::vector<Tuple>& data) {
  std::ifstream ins(file);
  if(!ins.is_open())  {
    fprintf(stderr, "Unable to open file for read: %s - %s\n", file, strerror(errno));
    return errno;
  }
  std::string line;
  while(std::getline(ins, line) && !line.empty() && line[0] == '%')
    ;
  std::vector<std::string> tokens;
  tokenize(line, " ,\t", tokens);
  const size_t nn = !tokens.empty() ? atoi((*tokens.rbegin()).c_str()) : 0;
  data.reserve(nn);
  while(std::getline(ins, line)) {
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

static void write_by_dict(Dict& du, FILE* fp) {
  for(DictIt it(du.begin()); it!=du.end(); it++) {
    int u = it->first;
    fprintf(fp, "%d:\n", u);
    auto eles = it->second;
    for(std::vector<std::pair<int, float>>::iterator vit=eles.begin(); vit!=eles.end(); vit++) {
      int v = vit->first;
      float r = vit->second;
      fprintf(fp, "%d,%f\n", v, r);
    }
  }
}

static int userwise(const char* write, std::vector<Tuple>& data, int nb, int nresd, int bk) {
  int i;
  FILE* fp = fopen(write, "w");
  if(!fp)  {
    fprintf(stderr, "Unable to open file for write: %s - %s\n", write, strerror(errno));
    return errno;
  }
  for(i=0; i<bk-1; i++) {
    Dict du;
    for(int j=i*nb; j<i*nb+nb; j++) {
      auto ele = data[j];
      int u = std::get<0>(ele);
      int v = std::get<1>(ele);
      float r = std::get<2>(ele);
      du[u].push_back(std::make_pair(v,r));
    }
    //write
    write_by_dict(du, fp);
  }
  {
    Dict du;
    for(int j=i*nb; j<i*nb+nb+nresd; j++) {
      auto ele = data[j];
      int u = std::get<0>(ele);
      int v = std::get<1>(ele);
      float r = std::get<2>(ele);
      du[u].push_back(std::make_pair(v,r));
    }
    //write
    write_by_dict(du, fp);
  }
  fclose(fp);
  return 0;
}

static int get_message(const char* read, const char* write)
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
      if(ii%block_size==0) {
        if(block.user_size() > 0) {
          block.SerializeToString(&uncompressed_buffer);
          const size_t uncompressed_size = uncompressed_buffer.size();
          fwrite(&uncompressed_size, 1, sizeof(uncompressed_size), f_w);
          fwrite(uncompressed_buffer.c_str(), 1, uncompressed_size, f_w);
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
    record->set_rating(rating);
  }
  if(!rc) {
    block.SerializeToString(&uncompressed_buffer);
    const size_t uncompressed_size = uncompressed_buffer.size();
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

static void hint() {
  fprintf(stderr, "-i         [input_file_name]\n");
  fprintf(stderr, "-o         [output_file_name]\n");
  fprintf(stderr, "--method   [userwise/raw2proto/protobuf]\n");
  fprintf(stderr, "--split    [number_of_splits_for_rating_matrix]\thints: 1~10 splits are recommended\n");
  fprintf(stderr, "--size     [number_of_users_in_each_block]\thints: 1 fread reads 1 block each time\n");
}

int main(int argc, char** argv) {
  const char *read   = NULL;
  const char *write  = NULL;
  const char *method = NULL;
  int bk=1;
  int rc = 0;
  for(int i = 1; i < argc; ++i) {
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
        bk = atoi(argv[i]);
      }
    }
    else if (!strcmp(option, "--size")) {
      if (++i < argc) {
        block_size = atoi(argv[i]);
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
      const bool isRaw = !strcmp(method, "raw2proto");
      if (isRaw || !strcmp(method, "userwise")) {
        std::vector<Tuple> data;
        rc = read_raw(read, data);
        if(!rc) {
          size_t nn = data.size();
          size_t nb = nn / bk;
          size_t nresd = nn % bk;
          std::string tmp = write;
          if(isRaw) {
            tmp += ".tmp";
          }
          rc = userwise(tmp.c_str(), data, nb, nresd, bk);
          if (!rc && isRaw) {
            rc = get_message(tmp.c_str(), write);
          }
          if(isRaw) {
            unlink(tmp.c_str());
          }
        }
      } else if (!strcmp(method, "protobuf")) {
        rc = get_message(read, write);
      } else {
        rc = 1;
      }
    }
  }
  return rc;
}
