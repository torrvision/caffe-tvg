#ifndef CAFFE_TVG_COMMON_UTILS_HPP
#define CAFFE_TVG_COMMON_UTILS_HPP

#include <string>
#include "caffe/blob.hpp"
#include <boost/lexical_cast.hpp>

namespace tvg {

  namespace CommonUtils {

    template<typename Dtype>
    void read_into_array(const int N, const std::string & source, Dtype * target) {

      std::stringstream iss;
      iss.clear();
      iss << source;
      std::string token;

      for (int i = 0; i < N; ++i) {

        if (std::getline(iss, token, ' ')) {
          target[i] = strtof(token.c_str(), NULL);
        } else {
          throw std::runtime_error(
                  "A malformed string! >" + source + "<. Couldn't read " + boost::lexical_cast<std::string>(N) + " values.");
        }
      }
    }

    template<typename Dtype>
    void read_into_the_diagonal(const std::string & source, caffe::Blob<Dtype> & blob, bool only_read_first = false) {

      const int height = blob.height();
      Dtype * data = blob.mutable_cpu_data();

      caffe::caffe_set(blob.count(), Dtype(0.), data);

      std::stringstream iss;
      iss.clear();
      iss << source;
      std::string token;

      for (int i = 0; i < height; ++i) {

        if (i == 1 && only_read_first){
          break;
        }

        if (std::getline(iss, token, ' ')) {
          data[i * height + i] = strtof(token.c_str(), NULL);
          //data[i * height + i] = std::stof(token);
        } else {
          throw std::runtime_error(
                  "A malformed string! >" + source + "<. Couldn't read " + boost::lexical_cast<std::string>(height) + " values.");
        }
      }
    }

    /*
     * Saves a blob to file
     * Assumes N = 1
     * Saves in the following format WxHxC (displays channel first)
     */
    template<typename Dtype>
    void save_blob_to_file(const caffe::Blob<Dtype> & blob, const std::string& filename, bool is_diff = false)
    {

      if ( blob.num() != 1) { return; }
      const Dtype * data;

      if (!is_diff) {
        data = blob.cpu_data();
      }
      else{
        data = blob.cpu_diff();
      }
      std::ofstream fs(filename.c_str());

      for (size_t h = 0; h < blob.height(); ++h){
        for (size_t w = 0; w < blob.width(); ++w){
          for (size_t c = 0; c < blob.channels(); ++c){
            size_t index =  (c * blob.height() + h) * blob.width() + w;
            fs << data[index] << ( (c+1) % blob.channels() == (0) ? '\n' : ',');
          }
        }
      }
    }

    template<typename Dtype>
    void save_stat_potentials_to_file(std::vector<std::vector<std::vector<Dtype> > > & stat_potentials, const std::string& filename)
    {

      std::ofstream fs(filename.c_str());

      for (size_t layer = 0; layer < stat_potentials.size(); ++layer){
        for (size_t segment = 0; segment < stat_potentials[layer].size(); ++segment){
          for (size_t i = 0; i < stat_potentials[layer][segment].size(); ++i){
            fs << stat_potentials[layer][segment][i] << ( (i+1) % stat_potentials[layer][segment].size() == (0) ? '\n' : ',');
          }
        }
      }
    }

    /*
     * Saves an array to file
     * Saves in the following format WxHxC (displays channel first)
     */
    template<typename Dtype>
    void save_array_to_file(const Dtype* blob, const int N, const std::string& filename)
    {
      std::ofstream fs(filename.c_str());

      for (size_t i = 0; i < N; ++i){
        fs << blob[i] << ( i == (N-1) ? '\n' : ',');
      }
    }


    template<typename Dtype>
    void print_blob(const caffe::Blob <Dtype> & blob, bool print_diff = false, const std::string description = "") {

      const Dtype *data = print_diff ? blob.cpu_diff() : blob.cpu_data();

      if (description!= "") {
        printf("%s: \n", description.c_str());
      }

      for (int n = 0; n < blob.num(); n++) {
        for (int c = 0; c < blob.channels(); c++) {
          for (int h = 0; h < blob.height(); h++) {
            for (int w = 0; w < blob.width(); w++) {
              int offset = ((n * blob.channels() + c) * blob.height() + h) * blob.width() + w;
              printf("%11.6f ", *(data + offset));
            }
            printf("\n");
          }
          printf("\n");
        }
      }

      printf("-- End of Blob --\n\n");
    }

    template<typename Dtype>
    inline void read_and_reshape_from_data(caffe::Blob <Dtype> & blob, const std::string & filename) {

      FILE* fp = fopen(filename.c_str(), "rb");
      if (fp == NULL){
        throw std::runtime_error("Could not open" + filename );
      }

      int N;
      int C;
      int H;
      int W;

      fread(&N, sizeof(int), 1, fp);
      fread(&C, sizeof(int), 1, fp);
      fread(&H, sizeof(int), 1, fp);
      fread(&W, sizeof(int), 1, fp);

      blob.Reshape(N, C, H, W);
      std::cout << "(N,C,H,W) (" << N << ", " << C << "," << H << ", " << W << ")" << std::endl;

      if (fread(blob.mutable_cpu_data(), sizeof (float), N * C * H * W, fp)) {
        std::cout << "Read " << N*C*H*W << " floats from " << filename << std::endl;
        fclose(fp);
      }
      else{
        std::cout << "Something went wrong in reading " << filename << ". (N,C,H,W) = ("<< N << ", " << C << ", " << H << ", " << W << ")"  << std::endl;
        throw std::runtime_error("Could not read " + filename);
      }

    }

  }
}


#endif //CAFFE_TVG_COMMON_UTILS_HPP
