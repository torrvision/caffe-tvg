#include "caffe/layer.hpp"
#include "caffe/layers/channel_select_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

namespace caffe {

/*
 * bottom[0] = Masks ((D+1)xCxHxW)
 * bottom[1] = Detections (1xDx1x6)
 * top[0]    = Selected Masks ((D+1)x1xHxW)
 *
 */
  template<typename Dtype>
  void ChannelSelectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_gpu_data();
  caffe_gpu_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->gpu_data();

  const int num_pixels = unary_blob->width() * unary_blob->height();

  for (int i = 0; i < detection_box_list_.size(); ++i) {

    const std::vector<int> &det_box = detection_box_list_[i]->get_foreground_pixels();

    const int det_label = detection_box_list_[i]->get_label();
    
    const Dtype* bottom_data = unaries_in + unary_blob->offset(i+1, det_label, 0, 0);
    Dtype* top_data = output + output_blob->offset(i+1, 0, 0, 0);

    caffe_gpu_copy(num_pixels, bottom_data, top_data);

  } // for

  // transfer over the 0st channel of 0st num (usually the background channel)
  const Dtype* bottom_data = unaries_in;
  Dtype* top_data = output;
  caffe_gpu_copy(num_pixels, bottom_data, top_data);

} // Forward_gpu

  template <typename Dtype>
  void ChannelSelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    Blob<Dtype>* output_blob = top[0];
    Blob<Dtype>* unary_blob = bottom[0];

    const Dtype* output_diff = output_blob->gpu_diff();
    Dtype* unaries_diff = unary_blob->mutable_gpu_diff();

    // assume batch size of 1
    //const int n = 0;

    const int num_pixels = unary_blob->width() * unary_blob->height();

    // initialise diff to be 0
    caffe_gpu_set(unary_blob->count(), Dtype(0), unaries_diff);

    for (int i = 0; i < detection_box_list_.size(); ++i) {

      const int det_label = detection_box_list_[i]->get_label();

      Dtype* bottom_diff = unaries_diff + unary_blob->offset(i+1, det_label, 0, 0);
      const Dtype* top_diff = output_diff + output_blob->offset(i+1, 0, 0, 0);

      caffe_gpu_copy(num_pixels, top_diff, bottom_diff);

    }

    // transfer over the 0st channel of 0st num  (usually the background channel)
    Dtype* bottom_diff = unaries_diff;
    const Dtype* top_diff = output_diff;
    caffe_gpu_copy(num_pixels, top_diff, bottom_diff);

  }

  INSTANTIATE_LAYER_GPU_FUNCS(ChannelSelectLayer);

} // namespace caffe
