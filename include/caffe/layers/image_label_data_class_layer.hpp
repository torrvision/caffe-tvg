#ifndef CAFFE_IMAGE_LABEL_DATA_CLASS_LAYER_H
#define CAFFE_IMAGE_LABEL_DATA_CLASS_LAYER_H

#include <random>
#include <vector>

#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

  template<typename Dtype>
  class ImageLabelDataClassLayer : public BasePrefetchingDataClassLayer<Dtype> {

  public:
      explicit ImageLabelDataClassLayer(const LayerParameter &param);

      virtual ~ImageLabelDataClassLayer();

      virtual void DataLayerSetUp(const vector<Blob < Dtype> *

      > &bottom,
      const vector<Blob < Dtype> *> &top);

      // DataLayer uses DataReader instead for sharing for parallelism
      virtual inline bool ShareInParallel() const { return false; }

      virtual inline const char *type() const { return "ImageLabelDataClass"; }

      virtual inline int ExactNumBottomBlobs() const { return 0; }

      virtual inline int ExactNumTopBlobs() const { return -1; }

      virtual inline int MaxTopBlobs() const { return 3; }

      virtual inline int MinTopBlobs() const { return 2; }

  protected:
      shared_ptr <Caffe::RNG> prefetch_rng_;

      virtual void ShuffleImages();

      virtual void SampleScale(cv::Mat *image, cv::Mat *label);

      virtual void ApplySampleRotation(cv::Mat &image, cv::Mat &label);

      virtual void ApplySampleGaussianBlur(cv::Mat &image);

      virtual void load_batch(BatchWithClassLabel <Dtype> *batch);

      virtual void MakeClassLabel(const Blob<Dtype> &label, Blob<Dtype> &class_label);
      
      virtual void MakeClassLabel(const cv::Mat &cv_label, Dtype *class_label_data_);

      vector <std::string> image_lines_;
      vector <std::string> label_lines_;
      vector<int> order_;
      int lines_id_;

      Blob <Dtype> transformed_label_;

      int label_margin_h_;
      int label_margin_w_;

      bool hsv_noise_;
      int h_noise_;
      int s_noise_;
      int v_noise_;

      bool pad_centre_;

      bool random_rotate_;
      int min_rotation_angle_;
      int max_rotation_angle_;

      bool random_gaussian_blur_;

      std::mt19937 *rng_;

      int label_space_;
  };

}

#endif //CAFFE_IMAGE_LABEL_DATA_CLASS_LAYER_H
