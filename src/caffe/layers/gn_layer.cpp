/**
 * Written by Qizhu <liqizhu@robots.ox.ac.uk>
*/

#include <algorithm>
#include <vector>

#include "caffe/layers/gn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Parses epsilon
  const GNParameter& param = this->layer_param_.gn_param();
  gn_eps_ = param.eps();

  // Parses num_groups and channels_per_group
  if (!param.has_num_groups() && !param.has_channels_per_group()){
    LOG(FATAL) << "At least one of num_groups and channels_per_group must be given.";
  }
  num_groups_ = param.has_num_groups() ?
      param.num_groups() : 
      bottom[0]->channels() / param.channels_per_group();
  channels_per_group_ = param.has_channels_per_group() ?
      param.channels_per_group() : 
      bottom[0]->channels() / param.num_groups();
  if (num_groups_ * channels_per_group_ != bottom[0]->channels()) {
    LOG(FATAL) << "Group partitioning error (N_g * C_g != C): " << 
        num_groups_ << " * " << channels_per_group_ << " vs " << bottom[0]->channels(); 
  }
  LOG(INFO) << "num_groups: " << num_groups_ << ", channels_per_group: " << channels_per_group_;

  // Initialize parameters
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    vector<int> shape;
    shape.push_back(1);
    shape.push_back(bottom[0]->channels());
    shape.push_back(1);
    shape.push_back(1);
    // slope
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > slope_filler(GetFiller<Dtype>(param.slope_filler()));
    slope_filler->Fill(this->blobs_[0].get());
    // bias
    this->blobs_[1].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void GNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->ReshapeLike(*(bottom[0]));

  broadcast_buffer_.ReshapeLike(*(bottom[0]));
  spatial_statistic_.Reshape(num_, channels_, 1, 1);
  group_statistic_.Reshape(num_, num_groups_, 1, 1);

  x_norm_.ReshapeLike(*(bottom[0]));
  x_inv_std_.ReshapeLike(group_statistic_);

  spatial_group_sum_multiplier_.Reshape(1, channels_per_group_, height_, width_);
  caffe_set(spatial_group_sum_multiplier_.count(), Dtype(1),
      spatial_group_sum_multiplier_.mutable_cpu_data());
  spatial_sum_multiplier_.Reshape(1, 1, height_, width_);
  caffe_set(spatial_sum_multiplier_.count(), Dtype(1),
      spatial_sum_multiplier_.mutable_cpu_data());
  batch_sum_multiplier_.Reshape(num_, 1, 1, 1);
  caffe_set(batch_sum_multiplier_.count(), Dtype(1),
      batch_sum_multiplier_.mutable_cpu_data());
  
}

template <typename Dtype>
void GNLayer<Dtype>::CalcGroupSum(const Dtype* input, const bool add_to_self, 
      Dtype* output) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * num_groups_, channels_per_group_ * height_ * width_,
      Dtype(1), input, spatial_group_sum_multiplier_.cpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::CalcBatchSum(const Dtype* input, const bool add_to_self, 
    Blob<Dtype>* medium, Dtype* output) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
      Dtype(1), input, spatial_sum_multiplier_.cpu_data(), 
      Dtype(0), medium->mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, 
      Dtype(1), medium->cpu_data(), batch_sum_multiplier_.cpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::BroadcastGroupStats(const Dtype* input, const bool add_to_self, 
    Dtype* output) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * num_groups_,
      channels_per_group_ * height_ * width_, 1, Dtype(1),
      input, spatial_group_sum_multiplier_.cpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::BroadcastChannel(const Dtype* input, const bool add_to_self, 
    Blob<Dtype>* medium, Dtype* output) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.cpu_data(), input,
      Dtype(0), medium->mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      medium->cpu_data(), spatial_sum_multiplier_.cpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype* const_bottom_data = bottom[0]->cpu_data();
  const Dtype* const_top_data = top[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const Dtype* const_scale_data = this->blobs_[0]->cpu_data();
  const Dtype* const_shift_data = this->blobs_[1]->cpu_data();

  // Calculate group means and store in group_statistic_
  CalcGroupSum(const_bottom_data, false, group_statistic_.mutable_cpu_data());
  caffe_scal(group_statistic_.count(), Dtype(1) / (channels_per_group_ * height_ * width_), 
      group_statistic_.mutable_cpu_data());

  // Broadcast group means, and subtract group means from input data
  BroadcastGroupStats(group_statistic_.cpu_data(), false, broadcast_buffer_.mutable_cpu_data());
  caffe_sub(broadcast_buffer_.count(), const_bottom_data, broadcast_buffer_.cpu_data(), top_data);

  // Calculate group variances and store in group_statistic_
  caffe_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
      broadcast_buffer_.mutable_cpu_data());
  CalcGroupSum(broadcast_buffer_.cpu_data(), false, group_statistic_.mutable_cpu_data());
  caffe_scal(group_statistic_.count(), Dtype(1) / (channels_per_group_ * height_ * width_), 
      group_statistic_.mutable_cpu_data());
  
  // Add epsilon to variance, power to (-1/2), and broadcast. Elementwise mult with (x_i - /mu).
  caffe_copy(group_statistic_.count(), group_statistic_.cpu_data(),
      x_inv_std_.mutable_cpu_data());
  caffe_add_scalar(x_inv_std_.count(), gn_eps_, 
      x_inv_std_.mutable_cpu_data());
  caffe_powx(x_inv_std_.count(), x_inv_std_.cpu_data(),
      Dtype(-0.5), x_inv_std_.mutable_cpu_data());
  BroadcastGroupStats(x_inv_std_.cpu_data(), false, broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), const_top_data, broadcast_buffer_.cpu_data(), 
      x_norm_.mutable_cpu_data());

  // Broadcast scale, and elementwise multiply with (x_i - mu)/sqrt(var + epsilon)
  BroadcastChannel(const_scale_data, false, & spatial_statistic_, 
      broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(), broadcast_buffer_.cpu_data(), 
      top_data);

  // Broadcast bias, and elementwise add to scale * (x_i - mu)/sqrt(var + epsilon)
  BroadcastChannel(const_shift_data, false, & spatial_statistic_, 
      broadcast_buffer_.mutable_cpu_data());
  caffe_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.cpu_data(), top_data);
}

template <typename Dtype>
void GNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // const params pointers
  const Dtype* const_scale_data = this->blobs_[0]->cpu_data();

  // const and mutable top and bottom diff pointers
  const Dtype* const_bottom_diff = bottom[0]->cpu_diff();
  const Dtype* const_top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // mutable params pointers
  Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();

  // Calculate dl/d(scale)
  caffe_mul(broadcast_buffer_.count(), const_top_diff, x_norm_.cpu_data(), 
      broadcast_buffer_.mutable_cpu_data());
  CalcBatchSum(broadcast_buffer_.cpu_data(), true, & spatial_statistic_, 
      scale_diff);

  // Calculate dl/d(bias)
  CalcBatchSum(const_top_diff, true, & spatial_statistic_, shift_diff);

  // Calculate dl/dx_hat
  BroadcastChannel(const_scale_data, false, & spatial_statistic_, 
      broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), const_top_diff, broadcast_buffer_.cpu_data(), 
      broadcast_buffer_.mutable_cpu_data());

  // Calculate sum of (dl/dx_hat * dx_hat)
  caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(), broadcast_buffer_.cpu_data(), 
      bottom_diff);
  CalcGroupSum(const_bottom_diff, false, group_statistic_.mutable_cpu_data());

  // Calculate dx_hat * sum of (dl/dx_hat * dx_hat)
  BroadcastGroupStats(group_statistic_.cpu_data(), false, bottom_diff);
  caffe_mul(broadcast_buffer_.count(), x_norm_.cpu_data(), bottom_diff, bottom_diff);
  
  // Calculate dl/dx_hat - E[dl/dx_hat] - x_hat * E[dl/dx_hat * x_hat]
  CalcGroupSum(broadcast_buffer_.cpu_data(), false, group_statistic_.mutable_cpu_data());
  BroadcastGroupStats(group_statistic_.cpu_data(), true, bottom_diff);
  caffe_cpu_axpby(broadcast_buffer_.count(), Dtype(1),
      broadcast_buffer_.cpu_data(), Dtype(-1) / (channels_per_group_ * height_ * width_),
      bottom_diff);
  
  // Calculate 1/sqrt(var + eps) * (dl/dx_hat - E[dl/dx_hat] - x_hat * E[dl/dx_hat * x_hat])
  BroadcastGroupStats(x_inv_std_.cpu_data(), false, broadcast_buffer_.mutable_cpu_data());
  caffe_mul(broadcast_buffer_.count(), broadcast_buffer_.cpu_data(), const_bottom_diff, 
      bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(GNLayer);
#endif

INSTANTIATE_CLASS(GNLayer);
REGISTER_LAYER_CLASS(GN);
}  // namespace caffe
