#ifndef CAFFE_IMAGE_LABEL_DATA_INDEX_DET_LAYER_H
#define CAFFE_IMAGE_LABEL_DATA_INDEX_DET_LAYER_H

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
#include "caffe/detection.hpp"

namespace caffe {

  template<typename Dtype>
  class ImageLabelDataIndexDetLayer : public BasePrefetchingDataIndexDetectionLayer<Dtype> {

  public:
      explicit ImageLabelDataIndexDetLayer(const LayerParameter &param);

      virtual ~ImageLabelDataIndexDetLayer();

      virtual void DataLayerSetUp(const vector<Blob < Dtype> *> &bottom,
      const vector<Blob < Dtype> *> &top);

      // DataLayer uses DataReader instead for sharing for parallelism
      virtual inline bool ShareInParallel() const { return false; }

      virtual inline const char *type() const { return "ImageLabelDataIndexDetection"; }

      virtual inline int ExactNumBottomBlobs() const { return 0; }

      virtual inline int ExactNumTopBlobs() const { return -1; }

      virtual inline int MaxTopBlobs() const { return 4; }

      virtual inline int MinTopBlobs() const { return 2; }

  protected:
      shared_ptr <Caffe::RNG> prefetch_rng_;

      virtual void ShuffleImages();

      virtual void SampleScale(cv::Mat *image, cv::Mat *label, Blob<Dtype>* detection_data = NULL);

      virtual void RotateBbox(const cv::Point2f & pivot, const cv::Point2f & im_dim, Dtype angle, Blob<Dtype>* detection_blob);

      virtual void PadBbox(cv::Mat &image, int min_size, int margin_w, int margin_h, bool pad_centre, Blob<Dtype>* detection_blob);

      virtual void ApplySampleRotation(cv::Mat &image, cv::Mat &label, Blob<Dtype>* detection_blob = NULL);

      virtual void ApplySampleGaussianBlur(cv::Mat &image);

      virtual void ApplyBboxPerturbation(cv::Mat &image, Blob<Dtype> *detection_blob);

      virtual void load_batch(BatchIndexDetection <Dtype> *batch);

      virtual void GetIndex(const std::string & filename, const int item_id, Dtype * prefetch_idx, std::string & img_id);
      virtual void LoadDetection(BatchIndexDetection<Dtype>* batch, const int batch_index, const int det_index, const std::string img_id);

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
      bool random_box_perturb_;

      std::mt19937 *rng_;
  };

}

#endif //CAFFE_IMAGE_LABEL_DATA_INDEX_DET_LAYER_H
