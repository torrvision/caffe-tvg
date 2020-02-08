#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

// #include "caffe/test/test_tvg_util.hpp"
// #include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void ClipData(const int nthreads, const Dtype upper_bound, 
    const Dtype lower_bound, Dtype* data_to_clip) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      if (data_to_clip[index] > upper_bound) {
          data_to_clip[index] = upper_bound;
      } else if (data_to_clip[index] < lower_bound) {
          data_to_clip[index] = lower_bound;
      }
  }
}

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // printf("GPU forward\n\n");
  const Dtype* const_bottom_data = bottom[0]->gpu_data();
  const Dtype* const_top_data = top[0]->gpu_data();
  //Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* scale_data = this->blobs_[0]->gpu_data();
  const Dtype* shift_data = this->blobs_[1]->gpu_data();

  // Mean normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average mean
    caffe_copy(batch_statistic_.count(), this->blobs_[2]->gpu_data(),
        batch_statistic_.mutable_gpu_data());
    // If we don't calculate batch statistics, then set r, d to default values
    // caffe_gpu_set(r_.count(), Dtype(1), r_.mutable_gpu_data());
    // caffe_gpu_set(d_.count(), Dtype(0), d_.mutable_gpu_data());
  } else {
    // Compute the mean by averaging over spatial and batch dimensions.
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), const_bottom_data,
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_,
        Dtype(1) / num_, spatial_statistic_.gpu_data(),
        batch_sum_multiplier_.gpu_data(), Dtype(0),
        batch_statistic_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(batch_statistic_, false, "mu_batch");
    if (use_global_stats_ && !is_gradient_check_){
        // calculate d_ as (mu_batch - mu_global)./std_global
        caffe_gpu_sub(batch_statistic_.count(), batch_statistic_.gpu_data(),
            this->blobs_[2]->gpu_data(),d_.mutable_gpu_data());
        // tvg::TestUtils::PrintBlob(d_, false, "mu_batch - mu_global");
        caffe_gpu_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
            x_inv_std_.mutable_gpu_data());
        caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_, 
            x_inv_std_.mutable_gpu_data());
        caffe_gpu_powx(batch_statistic_.count(), x_inv_std_.gpu_data(),
             Dtype(0.5), x_inv_std_.mutable_gpu_data());
        // tvg::TestUtils::PrintBlob(x_inv_std_, false, "sqrt(var_global + eps)");
        caffe_gpu_div(batch_statistic_.count(), d_.gpu_data(),
            x_inv_std_.gpu_data(),d_.mutable_gpu_data());
        // tvg::TestUtils::PrintBlob(d_, false, "d before clipping");
        if (clip_d_) {
            // printf("d_max is set to %f\n", d_max_);
            // -d_max <= d_ <= d_max
            ClipData<Dtype> <<<CAFFE_GET_BLOCKS(r_.count()), CAFFE_CUDA_NUM_THREADS>>> (
                d_.count(), d_max_, -d_max_, d_.mutable_gpu_data());
            // Dtype* d_data = d_.mutable_cpu_data();
            // for (int i = 0; i < d_.count(); ++i){
            //     d_data[i] = (d_data[i]<d_max_) ? d_data[i] : d_max_;
            //     d_data[i] = (d_data[i]>(-d_max_)) ? d_data[i] : (-d_max_);
            // }
            // tvg::TestUtils::PrintBlob(d_, false, "d after clipping");
        }
    }
    // Add to the moving average
    if (!frozen_) {
      caffe_gpu_axpby(batch_statistic_.count(),
          Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
          bn_momentum_, this->blobs_[2]->mutable_gpu_data());
    }
  }
  // Broadcast the mean vector
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(-1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  // Subtract
  caffe_gpu_add(broadcast_buffer_.count(), const_bottom_data,
      broadcast_buffer_.gpu_data(), top[0]->mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(*top[0], false, "xi - mu_batch");

  // Variance normalization
  if (frozen_ || this->phase_ == TEST) {
    // Use the moving average variance
    caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
        batch_statistic_.mutable_gpu_data());
  } else {
    caffe_gpu_powx(broadcast_buffer_.count(), const_top_data, Dtype(2),
        broadcast_buffer_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(broadcast_buffer_, false, "(xi - mu_batch)^2");
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1) / (height_ * width_), broadcast_buffer_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(spatial_statistic_, false, "sum[(xi - mu_batch)^2]/H/W in spatial dimensions");
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1) / num_,
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(0), batch_statistic_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(batch_statistic_, false, "var_batch = E[(xi - mu_batch)^2]/W/H/N");

    if (use_global_stats_ && !is_gradient_check_){
        // calculate r^2 as (var_batch + eps)./(var_global + eps), (compatible with Batch Renorm)
        caffe_gpu_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
            r_.mutable_gpu_data());
        caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_, 
            r_.mutable_gpu_data());
        // tvg::TestUtils::PrintBlob(r_, false, "var_batch + eps");
        caffe_gpu_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
            x_inv_std_.mutable_gpu_data());
        caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_, 
            x_inv_std_.mutable_gpu_data());
        // tvg::TestUtils::PrintBlob(x_inv_std_, false, "var_global + eps");
        caffe_gpu_div(batch_statistic_.count(),r_.gpu_data(),
            x_inv_std_.gpu_data(), r_.mutable_gpu_data());
        // tvg::TestUtils::PrintBlob(r_, false, "r^2 before clipping");
        if (clip_r_) {
            // Dtype* r_data = r_.mutable_cpu_data();
            Dtype r2_max_ = r_max_ * r_max_;
            // printf("r2_max is set to %f\n", r2_max_);
            ClipData<Dtype> <<<CAFFE_GET_BLOCKS(r_.count()), CAFFE_CUDA_NUM_THREADS>>> (
                r_.count(), r2_max_, (Dtype(1.)/r2_max_), r_.mutable_gpu_data());
            // for (int i = 0; i < r_.count(); ++i){
            //     r_data[i] = (r_data[i]<r2_max_) ? r_data[i] : r2_max_;
            //     r_data[i] = (r_data[i]>(Dtype(1.)/r2_max_)) ? r_data[i] : (Dtype(1.)/r2_max_);
            // }
            // tvg::TestUtils::PrintBlob(r_, false, "r^2 after clipping");
        }
    }
    if (!frozen_){
        // Add to the moving average
        caffe_gpu_axpby(batch_statistic_.count(),
            Dtype(1) - bn_momentum_, batch_statistic_.gpu_data(),
            bn_momentum_, this->blobs_[3]->mutable_gpu_data());
    }
  }
    // Add eps
    caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
        batch_statistic_.mutable_gpu_data());

   if (use_global_stats_ && this->phase_ != TEST && !frozen_){
    // tvg::TestUtils::PrintBlob(batch_statistic_, false, "var_batch");
    // Divide var_batch by r^2 to get var'_batch
    // caffe_gpu_div(batch_statistic_.count(),batch_statistic_.gpu_data(),
    //     r_.gpu_data(), batch_statistic_.mutable_gpu_data());
    caffe_gpu_powx(batch_statistic_.count(), r_.gpu_data(),
        Dtype(-1), x_inv_std_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(r_, false, "r^2 after clipping");
    // tvg::TestUtils::PrintBlob(x_inv_std_, false, "1/r^2");
    caffe_gpu_mul(batch_statistic_.count(), x_inv_std_.gpu_data(),
        batch_statistic_.gpu_data(), batch_statistic_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(batch_statistic_, false, "var'_batch");
    // Square root the above to get std_batch/r
    caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
        Dtype(0.5), x_inv_std_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(x_inv_std_, false, "std_batch/r");
    // Multiply the above with d to correction bias get d*std_batch/r
    caffe_gpu_mul(batch_statistic_.count(), x_inv_std_.gpu_data(),
        d_.gpu_data(), x_inv_std_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(x_inv_std_, false, "d*std_batch/r");
    // Add d*std_batch/r to top
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), x_inv_std_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(spatial_statistic_, false, "broadcast d*std_batch/r");
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(1), top[0]->mutable_gpu_data());
    // tvg::TestUtils::PrintBlob(*top[0], false, "normalised data plus d*std_batch/r");

  }
  // Inverse standard deviation
  caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
        Dtype(-0.5), batch_statistic_.mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(batch_statistic_, false, "inverse std");
  // Broadcast the inverse std
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(spatial_statistic_, false, "inverse std spatial");
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(broadcast_buffer_, false, "inverse std buffer");
  // Multiply with the inverse std
  // tvg::TestUtils::PrintBlob(*top[0], false, "top_data before div by std");
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top[0]->mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(*top[0], false, "output before scale and bias");


  // Save the normalized inputs and std for backprop
  if (!frozen_) {
    caffe_copy(broadcast_buffer_.count(), const_top_data,
        x_norm_.mutable_gpu_data());
    caffe_copy(batch_statistic_.count(), batch_statistic_.gpu_data(),
        x_inv_std_.mutable_gpu_data());
  }

  // Scale
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_mul(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top[0]->mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(*top[0], false, "scaled output");
  // Shift
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
      Dtype(1), batch_sum_multiplier_.gpu_data(), shift_data,
      Dtype(0), spatial_statistic_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
      height_ * width_, 1, Dtype(1),
      spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
      Dtype(0), broadcast_buffer_.mutable_gpu_data());
  caffe_gpu_add(broadcast_buffer_.count(), const_top_data,
      broadcast_buffer_.gpu_data(), top[0]->mutable_gpu_data());
  // tvg::TestUtils::PrintBlob(*top[0], false, "final output");
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (frozen_) {
    if (propagate_down[0]) {
      const Dtype* const_top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      // Use the moving average variance
      caffe_copy(batch_statistic_.count(), this->blobs_[3]->gpu_data(),
          batch_statistic_.mutable_gpu_data());
      caffe_gpu_add_scalar(batch_statistic_.count(), bn_eps_,
          batch_statistic_.mutable_gpu_data());
      caffe_gpu_powx(batch_statistic_.count(), batch_statistic_.gpu_data(),
          Dtype(-0.5), batch_statistic_.mutable_gpu_data());
      // Multiple slope with inverse std
      caffe_gpu_mul(batch_statistic_.count(), this->blobs_[0]->gpu_data(),
          batch_statistic_.gpu_data(), batch_statistic_.mutable_gpu_data());
      // Broadcast
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
          Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
          Dtype(0), spatial_statistic_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
          height_ * width_, 1, Dtype(1),
          spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0), broadcast_buffer_.mutable_gpu_data());
      // Elementwise multiply top grad with (slope / std)
      caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
          broadcast_buffer_.gpu_data(), bottom_diff);
    }
    return;
  }

  // gradient w.r.t. slope
  if (this->param_propagate_down_[0]) {
    const Dtype* const_top_diff = top[0]->gpu_diff();
    Dtype* scale_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(), const_top_diff,
        broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), broadcast_buffer_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(1), scale_diff);
  }

  // gradient w.r.t. bias
  if (this->param_propagate_down_[1]) {
    const Dtype* const_top_diff = top[0]->gpu_diff();
    Dtype* shift_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), const_top_diff, spatial_sum_multiplier_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(1), shift_diff);
  }

  // gradient w.r.t. normalized inputs
  if (propagate_down[0]) {
    const Dtype* const_top_diff = top[0]->gpu_diff();
    const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), scale_data,
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1), spatial_statistic_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), Dtype(0),
        broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_mul(broadcast_buffer_.count(), const_top_diff,
        broadcast_buffer_.gpu_data(), broadcast_buffer_.mutable_gpu_data());

    // sum of x_hat * (dl / dx_hat)
    caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
        broadcast_buffer_.gpu_data(), bottom_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
        Dtype(1), const_bottom_diff, spatial_sum_multiplier_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
        spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
        Dtype(0), batch_statistic_.mutable_gpu_data());
        
    if (use_global_stats_){
        // (x_hat - d)/r^2
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
            Dtype(1), batch_sum_multiplier_.gpu_data(), r_.gpu_data(),
            Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
            height_ * width_, 1, Dtype(1),
            spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
            Dtype(0), bottom_diff);
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
            Dtype(-1), batch_sum_multiplier_.gpu_data(), d_.gpu_data(),
            Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
            height_ * width_, 1, Dtype(1),
            spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
            Dtype(1), x_norm_.mutable_gpu_data());
        caffe_gpu_div(x_norm_.count(), x_norm_.gpu_data(), bottom_diff, 
            x_norm_.mutable_gpu_data());
    }

    // x_hat times the sum
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(0), bottom_diff);

    if (use_global_stats_){
        // sum dl/dx_hat * x_hat - d * sum dl/dx_hat
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
            Dtype(1), broadcast_buffer_.gpu_data(),
            spatial_sum_multiplier_.gpu_data(), Dtype(0),
            spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
            spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
            Dtype(0), batch_statistic_.mutable_gpu_data());
        caffe_gpu_mul(batch_statistic_.count(), batch_statistic_.gpu_data(), 
            d_.gpu_data(), d_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
            Dtype(1), batch_sum_multiplier_.gpu_data(), d_.gpu_data(),
            Dtype(0), spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
            height_ * width_, 1, Dtype(-1),
            spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
            Dtype(1), bottom_diff);
    }
    
    // (x_hat - d)/r^2 .* (sum dl/dx_hat * x_hat - d * sum dl/dx_hat)
    caffe_gpu_mul(broadcast_buffer_.count(), x_norm_.gpu_data(),
        const_bottom_diff, bottom_diff);

    // Subtract the average of x_hat times the sum
    if (!use_global_stats_){
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * channels_, height_ * width_,
            Dtype(1), broadcast_buffer_.gpu_data(),
            spatial_sum_multiplier_.gpu_data(), Dtype(0),
            spatial_statistic_.mutable_gpu_data());
        caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1),
            spatial_statistic_.gpu_data(), batch_sum_multiplier_.gpu_data(),
            Dtype(0), batch_statistic_.mutable_gpu_data());
    }
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), batch_statistic_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(1), bottom_diff);
    caffe_gpu_axpby(broadcast_buffer_.count(), Dtype(1),
        broadcast_buffer_.gpu_data(), Dtype(-1) / (num_ * height_ * width_),
        bottom_diff);

    // Multiply with the inverse std
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1,
        Dtype(1), batch_sum_multiplier_.gpu_data(), x_inv_std_.gpu_data(),
        Dtype(0), spatial_statistic_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * channels_,
        height_ * width_, 1, Dtype(1),
        spatial_statistic_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
        Dtype(0), broadcast_buffer_.mutable_gpu_data());
    caffe_gpu_mul(broadcast_buffer_.count(), const_bottom_diff,
        broadcast_buffer_.gpu_data(), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);

}  // namespace caffe
