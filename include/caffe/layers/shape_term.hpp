#ifndef CAFFE_SHAPE_TERM_HPP
#define CAFFE_SHAPE_TERM_HPP

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/detection.hpp"
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/layers/transfer_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/roi_unpooling.hpp"

namespace caffe{
template <typename Dtype>
class ShapeTermLayer : public Layer<Dtype> {
public:
  explicit ShapeTermLayer(const LayerParameter& param)
          : Layer<Dtype>(param), transfer_shape_(param), concat_layer_(param), roi_unpool_layer_(param) {}

  // This function is called once, and is basically the "Constructor"
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  // This function is called before every call to "Forward"
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Shape Term Layer"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

#ifndef CPU_ONLY
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
#endif

  vector<shared_ptr<const tvg::Detection> > detection_box_list_;
  std::string detection_boxes_input_dir_;

  shared_ptr<ROIPoolingLayer<Dtype> > roi_pooling_layer_; // std::shared_ptr< ... > will make NVCC complain
  shared_ptr<TransferLayer<Dtype> > transfer_unary_;
  TransferLayer<Dtype> transfer_shape_;
  shared_ptr<EltwiseLayer<Dtype> > eltwise_layer_;
  ConcatLayer<Dtype> concat_layer_;
  ROIUnpoolingLayer<Dtype> roi_unpool_layer_;

  Blob<Dtype> class_unary_blob_;
  std::vector< Blob<Dtype>* > class_unary_rois_; // Need to save this for the backward pass in order to pass the gradients
  std::vector< Blob<Dtype>* > matched_shapes_;
  std::vector< Blob<Dtype>* > shape_terms_orig_;
  std::vector< Blob<Dtype>* > shape_terms_padded_;
  std::vector< Blob<int>* > roi_pooling_switches_;

  std::vector< Blob<Dtype>* > warped_shapes_;

  Blob<Dtype> shape_term_only_;
  Blob<Dtype> shape_warp_;
  Blob<Dtype> zero_;

  std::vector<Blob<Dtype>* > transfer_unary_bottom_;
  std::vector<Blob<Dtype>* > transfer_unary_top_;
  std::vector<Blob<Dtype>* > roi_pool_bottom_;
  std::vector<Blob<Dtype>* > roi_pool_top_;
  std::vector<Blob<Dtype>* > elementwise_bottom_;
  std::vector<Blob<Dtype>* > elementwise_top_;
  std::vector<Blob<Dtype>* > roi_unpool_bottom_;
  std::vector<Blob<Dtype>* > roi_unpool_top_;
  std::vector<Blob<Dtype>* > concat_bottom_;

  std::vector<Blob<Dtype>* > shape_interp_bottom_;

  int top_channels_;
  int top_height_;
  int top_width_;

  bool is_initialised_;

  std::vector<int> shape_a_indices;
  std::vector<int> shape_b_indices;

}; // class ShapeTermLayer
} // namespace Caffe

#endif //CAFFE_SHAPE_TERM_HPP
