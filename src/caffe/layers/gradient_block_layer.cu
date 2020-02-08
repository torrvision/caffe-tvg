#include <vector>

#include "caffe/layers/gradient_block_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    if (top[i] != bottom[i]) {
      top[i]->ShareData(*bottom[i]);
    }
  }
}

template <typename Dtype>
void GradientBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    caffe_gpu_set(bottom[i]->count(), Dtype(0),
        bottom[i]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientBlockLayer);

}  // namespace caffe
