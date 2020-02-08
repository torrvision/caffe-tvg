#include <algorithm>
#include <cfloat>
#include <vector>
// #include <thrust/sort.h>
// #include <thrust/execution_policy.h>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          const bool has_prob_thresh, const float prob_thresh,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      Dtype predicted_prob = prob_data[n * dim + label_value * spatial_dim + s];
      if (has_prob_thresh && predicted_prob > prob_thresh){
          loss[index] = 0;
          counts[index] = 0;
      }else {
          loss[index] = -log(max(predicted_prob, Dtype(FLT_MIN)));
          counts[index] = 1;
      }
    }
  }
}

template <typename Dtype>
__global__ void SoftmaxLossOHEM(const int nthreads, 
          const Dtype* ranks, Dtype* loss_data, Dtype* counts, 
          const int ohem_n_){
  CUDA_KERNEL_LOOP(index, nthreads){
    if (ranks[index] >= ohem_n_) {
      loss_data[index] = Dtype(0);
      counts[index] = Dtype(0);
    }
  }
}

// template <typename Dtype>
// __global__ void SoftmaxLossIota(const int nthreads, 
//           Dtype* start, const Dtype start_val) {
//   CUDA_KERNEL_LOOP(index, nthreads){
//     start[index] = start_val + index;
//   }
// }

// template <typename Dtype>
// __global__ void SoftmaxLossSortByKey(const int nthreads, 
//           Dtype* key_start, const int count, Dtype* val_start) {
//   thrust::sort_by_key(thrust::seq, key_start, key_start + count, 
//     val_start, thrust::greater<Dtype>());
// }

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, has_probability_thresh_, probability_thresh_, counts);
  if (use_ohem_){
    caffe_copy(bottom[1]->count(), bottom[0]->gpu_diff(), temp.mutable_gpu_data());    
    // TODO: write a GPU version of RankOfValues
    RankOfValues(temp, temp);
    SoftmaxLossOHEM<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, temp.mutable_gpu_data(), 
      loss_data, prob_.mutable_gpu_diff(), ohem_n_);
  }
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_,
          const bool has_prob_thresh, const float prob_thresh, const Dtype* prob_data,
          Dtype* counts,
          const bool use_ohem_, const Dtype* ranks, const int ohem_n_) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    Dtype prob_value = 0;
    if (has_prob_thresh && label_value != ignore_label_){
      prob_value = prob_data[n * dim + label_value * spatial_dim + s];
    }

    if ( (has_ignore_label_ && label_value == ignore_label_) ||
         (has_prob_thresh && prob_value > prob_thresh) ||
         (use_ohem_ && ranks[n * spatial_dim + s] >= ohem_n_)) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_,
        has_probability_thresh_, probability_thresh_, prob_data,
        counts, use_ohem_, temp.gpu_data(), ohem_n_);

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
