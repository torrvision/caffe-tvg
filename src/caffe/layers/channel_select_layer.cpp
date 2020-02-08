#include "caffe/layer.hpp"
#include "caffe/layers/channel_select_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe {

template<typename Dtype>
void ChannelSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

}

template<typename Dtype>
void ChannelSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  tvg::DetectionUtils::read_detections_from_blob(detection_box_list_, *bottom[1]);

  top_num_ = bottom[0]->num();
  top_channels_ = 1;
  top_height_ = bottom[0]->height();
  top_width_ = bottom[0]->width();
  top[0]->Reshape(top_num_, top_channels_, top_height_, top_width_);
  caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());

}

/*
 * bottom[0] = Masks ((D+1)xCxHxW)
 * bottom[1] = Detections (1xDx1x6)
 * top[0]    = Selected Masks ((D+1)x1xHxW)
 *
 */
template<typename Dtype>
void ChannelSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_cpu_data();
  caffe_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->cpu_data();

  for (int i = 0; i < detection_box_list_.size(); ++i) {
    
    const int det_label = detection_box_list_[i]->get_label();

    for (int x = 0; x < top_width_; ++x) {
      for (int y = 0; y < top_height_; ++y) {
          output[output_blob->offset(i+1, 0, y, x)] = unaries_in[unary_blob->offset(i+1, det_label, y, x)];
      }
    }
  } // for

  for (int x = 0; x < top_width_; ++x) {
    for (int y = 0; y < top_height_; ++y) {
      output[output_blob->offset(0, 0, y, x)] = unaries_in[unary_blob->offset(0, 0, y, x)];
    }
  }

} // Forward_cpu

/*
 * bottom[0] = Masks ((D+1)xCxHxW)
 * bottom[1] = Detections (1xDx1x6)
 * top[0]    = Selected Masks ((D+1)x1xHxW)
 *
 */
template<typename Dtype>
void ChannelSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  const Dtype* output_diff = output_blob->cpu_diff();
  Dtype* unaries_diff = unary_blob->mutable_cpu_diff();
  
  // initialise diff to be 0
  caffe_set(unary_blob->count(), Dtype(0), unaries_diff);

  for (int i = 0; i < detection_box_list_.size(); ++i) {

    const int det_label = detection_box_list_[i]->get_label();

    for (int x = 0; x < top_width_; ++x) {
      for (int y = 0; y < top_height_; ++y) {
        unaries_diff[unary_blob->offset(i+1, det_label, y, x)] = output_diff[output_blob->offset(i+1, 0, y, x)];
      }
    }
  }

  for (int x = 0; x < top_width_; ++x) {
    for (int y = 0; y < top_height_; ++y) {
      unaries_diff[unary_blob->offset(0, 0, y, x)] = output_diff[output_blob->offset(0, 0, y, x)];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ChannelSelectLayer);
#endif

INSTANTIATE_CLASS(ChannelSelectLayer);

REGISTER_LAYER_CLASS(ChannelSelect);
}