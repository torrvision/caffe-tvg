#include <vector>

#include "caffe/layers/gradient_block_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    if (top[i] != bottom[i]) {
      // do nothing if in_place
      // set up the top shape if not in_place
      top[i]->ReshapeLike(*bottom[i]);
    }
  }
}

template <typename Dtype>
void GradientBlockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    if (top[i] != bottom[i]) {
      // do nothing if in_place
      // set top to share data with bottom if not in_place
      top[i]->ShareData(*bottom[i]);
    }
  }
}

template <typename Dtype>
void GradientBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    caffe_set(bottom[i]->count(), Dtype(0),
      bottom[i]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientBlockLayer);
#endif

INSTANTIATE_CLASS(GradientBlockLayer);
REGISTER_LAYER_CLASS(GradientBlock);

}  // namespace caffe
