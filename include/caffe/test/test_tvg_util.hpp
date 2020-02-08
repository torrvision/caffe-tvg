#ifndef TVG_TEST_UTIL
#define TVG_TEST_UTIL

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace tvg {

  namespace TestUtils {

    template<typename Dtype>
    void PrintBlob(const caffe::Blob <Dtype> & blob, bool print_diff = false, const char *info = 0) {

      const Dtype *data = print_diff ? blob.cpu_diff() : blob.cpu_data();

      if (info != 0) {
        printf("%s: \n", info);
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
        // printf("=================\n");
      }

      printf("-- End of Blob --\n\n");
    }


    template<typename Dtype>
    void FillWithUpperBound(caffe::Blob<Dtype> *blob, float max_value = 1, bool fill_diff = false) {

      Dtype * target = fill_diff ? blob->mutable_cpu_diff() : blob->mutable_cpu_data();

      for (int i = 0; i < blob->count(); ++i) {
        target[i] = ((double) rand() / RAND_MAX) * max_value;
      }
    }


    template<typename Dtype>
    void FillAsRGB(caffe::Blob <Dtype> *blob) {

      for (int i = 0; i < blob->count(); ++i) {
        blob->mutable_cpu_data()[i] = rand() % 256;
      }
    }


    template<typename Dtype>
    void FillAsProb(caffe::Blob <Dtype> *blob) {

      for (int i = 0; i < blob->count(); ++i) {
        double num = (double) rand() / (double) RAND_MAX;
        blob->mutable_cpu_data()[i] = static_cast<Dtype>((num != 0) ? num : 0.0002);
      }

      for (int n = 0; n < blob->num(); ++n) {
        for (int h = 0; h < blob->height(); ++h) {
          for (int w = 0; w < blob->width(); ++w) {

            Dtype total = 0;

            for (int c = 0; c < blob->channels(); ++c) {
              total += blob->data_at(n, c, h, w);
            }

            for (int c = 0; c < blob->channels(); ++c) {
              blob->mutable_cpu_data()[blob->offset(n, c, h, w)] = blob->data_at(n, c, h, w) / total;
            }
          }
        }
      }
    }

    template<typename Dtype>
    void FillAsLogProb(caffe::Blob <Dtype> *blob) {
      FillAsProb(blob);

      for (int i = 0; i < blob->count(); ++i) {
        blob->mutable_cpu_data()[i] = log(blob->cpu_data()[i]);
      }
    }

    template <typename Dtype>
    void FillFromDat(caffe::Blob<Dtype> *blob, const std::string & filename) {

      FILE* fp = fopen(filename.c_str(), "rb");
      if (fp == NULL){
        throw std::runtime_error("Could not open" + filename );
      }

      const int N = 500 * 500 * 21;

      if (fread(blob->mutable_cpu_data(), sizeof (float), N, fp)) {
        printf("Read %d floats.\n", N);
        fclose(fp);
      }
    }

    template <typename Dtype>
    void fill_from_dat(caffe::Blob<Dtype> *blob, const std::string & filename) {
      FillFromDat(blob, filename);
    }

    template<typename Dtype>
    void FillAsConstant(caffe::Blob <Dtype> *blob, const Dtype constant, bool fill_diff = false) {

      Dtype * data;
      if (fill_diff){
        data = blob->mutable_cpu_diff();
      }else{
        data = blob->mutable_cpu_data();
      }

      for (int i = 0; i < blob->count(); ++i) {
        data[i] = constant;
      }
    }

    template <typename Dtype>
    void GetLabelMap(const caffe::Blob<Dtype> & blob, short * labels) {

      const int num_pixels = blob.height() * blob.width();
      const Dtype * data = blob.cpu_data();

      for (int cur_pixel = 0; cur_pixel < num_pixels; ++cur_pixel) {
        labels[cur_pixel] =  0;
        float cur_max = data[cur_pixel];

        for (int c = 1; c < blob.channels(); ++c) {
          if (data[c * num_pixels + cur_pixel] > cur_max) {
            cur_max = data[c * num_pixels + cur_pixel];
            labels[cur_pixel] = c;
          }
        }
      }
    }

  // for legacy reasons
  template <typename Dtype>
  void get_label_map(const caffe::Blob<Dtype> & blob, short * labels){
    GetLabelMap(blob, labels);
  }

/*
*  Converts a label to an RGB value, using the PASCAL VOC colour map
*/
  inline void label_to_rgb(int label, int * rgb) {
    rgb[0] = rgb[1] = rgb[2] = 0;
    for (int i = 0; label > 0; i++, label >>= 3) {
      rgb[0] |= (unsigned char)(((label >> 0) & 1) << (7 - i));
      rgb[1] |= (unsigned char)(((label >> 1) & 1) << (7 - i));
      rgb[2] |= (unsigned char)(((label >> 2) & 1) << (7 - i));
    }
  }

/*
* The filename should contain the extension of the file as well
*/
  inline void save_image(short * labels, const std::string &filename, const int H, const int W) {

    cv::Mat image;
    image.create(H, W, CV_8UC3);

    int counter = 0;

    cv::MatIterator_<cv::Vec3b> iterator = image.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> end = image.end<cv::Vec3b>();

    int rgb[3];

    for (; iterator != end; ++iterator) {

      int label = labels[counter];
      label_to_rgb(label, &rgb[0]);

      // Since OpenCV stores stuff in BGR order
      (*iterator)[0] = rgb[2];
      (*iterator)[1] = rgb[1];
      (*iterator)[2] = rgb[0];

      ++counter;
    }

    cv::imwrite(filename, image);


    // temp stuff
    std::ofstream myfile;
    std::string matlabout = filename + ".txt";
    myfile.open(matlabout.c_str());

    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        myfile << labels[i * W + j] << " ";
      }
      myfile << std::endl;
    }

    myfile.close();
  }

  template <typename Dtype>
  void read_image( caffe::Blob<Dtype> * blob, const std::string & im_filename,int & real_height, int &real_width) {

    cv::Mat image = cv::imread(im_filename);
    const int num_pixels = 500 * 500;
    real_height = image.rows;
    real_width = image.cols;
    Dtype * data = blob->mutable_cpu_data();
    int pixel_id = 0;
    for (int y = 0; y < image.rows; ++y) {
      for (int x = 0; x < image.cols; ++x) {
        data[pixel_id] = (image.at<cv::Vec3b>(y, x))[0]; // B
        data[num_pixels + pixel_id] = (image.at<cv::Vec3b>(y, x))[1]; // G
        data[2*num_pixels + pixel_id] = (image.at<cv::Vec3b>(y, x))[2]; // R
        ++pixel_id;
      }
    }
  }




  } //namespace TestUtils
} //namespace tvg

#endif