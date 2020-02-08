#ifndef CAFFE_BOX_TERM_LAYER_HPP
#define CAFFE_BOX_TERM_LAYER_HPP

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/detection.hpp"

namespace caffe{
template <typename Dtype>
class BoxTermLayer : public Layer<Dtype> {
public:
  explicit BoxTermLayer(const LayerParameter& param)
          : Layer<Dtype>(param) {}

  // This function is called once, and is basically the "Constructor"
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // This function is called before every call to "Forward"
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Box Term Layer"; }
  //virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<shared_ptr<const tvg::Detection> > detection_box_list_;
  vector<shared_ptr<const tvg::Detection> > detection_pixel_list_;
  std::string detection_boxes_input_dir_;
  std::string detection_pixels_input_dir_;

  int top_channels_;
  int top_height_;
  int top_width_;

  int num_rescored_detections_;
  bool is_no_rescore_baseline_;
  bool is_background_det_;
  Dtype background_det_score_;

  bool read_det_from_blob_;

}; // class BoxTermLayer
} // namespace Caffe


#endif //CAFFE_BOX_TERM_LAYER_HPP
