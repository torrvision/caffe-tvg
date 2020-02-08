#include <algorithm>
#include <vector>

#include "caffe/layers/gn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GNLayer<Dtype>::CalcGroupSum_gpu(const Dtype* input, const bool add_to_self, 
    Dtype* output) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * num_groups_, channels_per_group_ * height_ * width_,
      Dtype(1), input, spatial_group_sum_multiplier_.gpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::CalcBatchSum_gpu(const Dtype* input, const bool add_to_self, 
    Blob<Dtype>* medium, Dtype* output) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
      Dtype(1), input, spatial_sum_multiplier_.gpu_data(), 
      Dtype(0), medium->mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, 
      Dtype(1), medium->gpu_data(), batch_sum_multiplier_.gpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::BroadcastGroupStats_gpu(const Dtype* input, const bool add_to_self, 
    Dtype* output) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * num_groups_,
      channels_per_group_ * height_ * width_, 1, Dtype(1),
      input, spatial_group_sum_multiplier_.gpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::BroadcastChannel_gpu(const Dtype* input, const bool add_to_self, 
    Blob<Dtype>* medium, Dtype* output) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), input,
      Dtype(0), medium->mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      medium->gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(add_to_self), output);
}

template <typename Dtype>
void GNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const Dtype* const_bottom_data = bottom[0]->gpu_data();
  const Dtype* const_top_data = top[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* const_scale_data = this->blobs_[0]->gpu_data();
  const Dtype* const_shift_data = this->blobs_[1]->gpu_data();

  // Calculate group means and store in group_statistic_
  CalcGroupSum_gpu(const_bottom_data, false, group_statistic_.mutable_gpu_data());
  caffe_gpu_scal(group_statistic_.count(), Dtype(1) / (channels_per_group_ * height_ * width_), 
      group_statistic_.mutable_gpu_data());

  // Broadcast group means, and subtract group means from input data
  BroadcastGroupStats_gpu(group_statistic_.gpu_data(), false, broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_sub(broadcast_buffer_.count(), const_bottom_data, broadcast_buffer_.gpu_data(), top_data);

  // Calculate group variances and store in group_statistic_
  caffe_gpu_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
      broadcast_buffer_.mutable_gpu_data());
  CalcGroupSum_gpu(broadcast_buffer_.gpu_data(), false, group_statistic_.mutable_gpu_data());
  caffe_gpu_scal(group_statistic_.count(), Dtype(1) / (channels_per_group_ * height_ * width_), 
      group_statistic_.mutable_gpu_data());
  
  // Add epsilon to variance, power to (-1/2), and broadcast. Elementwise mult with (x_i - /mu).
  caffe_copy(group_statistic_.count(), group_statistic_.gpu_data(),
      x_inv_std_.mutable_gpu_data());
  caffe_gpu_add_scalar(x_inv_std_.count(), gn_eps_, 
      x_inv_std_.mutable_gpu_data());
  caffe_gpu_powx(x_inv_std_.count(), x_inv_std_.gpu_data(),
      Dtype(-0.5), x_inv_std_.mutable_gpu_data());
  BroadcastGroupStats_gpu(x_inv_std_.gpu_data(), false, broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data, broadcast_buffer_.gpu_data(), 
      x_norm_.mutable_gpu_data());

  // Broadcast scale, and elementwise multiply with (x_i - mu)/sqrt(var + epsilon)
  BroadcastChannel_gpu(const_scale_data, false, & spatial_statistic_, 
      broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), broadcast_buffer_.gpu_data(), 
      top_data);

  // Broadcast bias, and elementwise add to scale * (x_i - mu)/sqrt(var + epsilon)
  BroadcastChannel_gpu(const_shift_data, false, & spatial_statistic_, 
      broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top_data);
}

template <typename Dtype>
void GNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // const params pointers
  const Dtype* const_scale_data = this->blobs_[0]->gpu_data();

  // const and mutable top and bottom diff pointers
  const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
  const Dtype* const_top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  // mutable params pointers
  Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();

  // Calculate dl/d(scale)
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff, x_norm_.gpu_data(), 
      broadcast_buffer_.mutable_gpu_data());
  CalcBatchSum_gpu(broadcast_buffer_.gpu_data(), true, & spatial_statistic_, 
      scale_diff);

  // Calculate dl/d(bias)
  CalcBatchSum_gpu(const_top_diff, true, & spatial_statistic_, shift_diff);

  // Calculate dl/dx_hat
  BroadcastChannel_gpu(const_scale_data, false, & spatial_statistic_, 
      broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff, broadcast_buffer_.gpu_data(), 
      broadcast_buffer_.mutable_gpu_data());

  // Calculate sum of (dl/dx_hat * dx_hat)
  caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), broadcast_buffer_.gpu_data(), 
      bottom_diff);
  CalcGroupSum_gpu(const_bottom_diff, false, group_statistic_.mutable_gpu_data());

  // Calculate dx_hat * sum of (dl/dx_hat * dx_hat)
  BroadcastGroupStats_gpu(group_statistic_.gpu_data(), false, bottom_diff);
  caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), bottom_diff, bottom_diff);
  
  // Calculate dl/dx_hat - E[dl/dx_hat] - x_hat * E[dl/dx_hat * x_hat]
  CalcGroupSum_gpu(broadcast_buffer_.gpu_data(), false, group_statistic_.mutable_gpu_data());
  BroadcastGroupStats_gpu(group_statistic_.gpu_data(), true, bottom_diff);
  caffe_gpu_axpby(broadcast_buffer_.count(), Dtype(1),
      broadcast_buffer_.gpu_data(), Dtype(-1) / (channels_per_group_ * height_ * width_),
      bottom_diff);
  
  // Calculate 1/sqrt(var + eps) * (dl/dx_hat - E[dl/dx_hat] - x_hat * E[dl/dx_hat * x_hat])
  BroadcastGroupStats_gpu(x_inv_std_.gpu_data(), false, broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), broadcast_buffer_.gpu_data(), const_bottom_diff, 
      bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(GNLayer);

}  // namespace caffe
