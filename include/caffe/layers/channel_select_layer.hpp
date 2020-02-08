#ifndef CAFFE_CHANNEL_SELECT_LAYER_HPP
#define CAFFE_CHANNEL_SELECT_LAYER_HPP

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/detection.hpp"

namespace caffe{
template <typename Dtype>
class ChannelSelectLayer : public Layer<Dtype> {
public:
  explicit ChannelSelectLayer(const LayerParameter& param)
          : Layer<Dtype>(param) {}

  // This function is called once, and is basically the "Constructor"
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // This function is called before every call to "Forward"
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ChannelSelect"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<shared_ptr<const tvg::Detection> > detection_box_list_;

  int top_num_;
  int top_channels_;
  int top_height_;
  int top_width_;

}; // class ChannelSelectLayer
} // namespace Caffe


#endif //CAFFE_CHANNEL_SELECT_LAYER_HPP
