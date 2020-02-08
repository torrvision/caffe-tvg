#ifndef CAFFE_ROI_UNPOOLING_HPP
#define CAFFE_ROI_UNPOOLING_HPP

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
template <typename Dtype>
class ROIUnpoolingLayer : public Layer<Dtype> {
public:
  explicit ROIUnpoolingLayer(const LayerParameter& param)
          : Layer<Dtype>(param) {}

  // This function is called once, and is basically the "Constructor"
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // This function is called before every call to "Forward"
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROI Unpooling Layer"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

}; // class ROIUnpooling
} // namespace Caffe


#endif //CAFFE_ROI_UNPOOLING_HPP
