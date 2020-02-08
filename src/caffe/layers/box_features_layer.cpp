#include "caffe/layer.hpp"
#include "caffe/layers/box_features_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe {

template<typename Dtype>
void BoxFeaturesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  this->is_background_det_ = this->layer_param().box_term_param().is_background_det();
  this->background_det_score_ = Dtype(this->layer_param().box_term_param().background_det_score());

  if (is_background_det_){
    std::cout << "Background det enabled" << std::endl;
    std::cout << "Background det score: " << background_det_score_ << std::endl;
  }
  else{
    std::cout << "Background det disabled" << std::endl;
  }

}

template<typename Dtype>
void BoxFeaturesLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  tvg::DetectionUtils::read_detections_from_blob(detection_box_list_, *bottom[1]);


  top_num_ = (int)detection_box_list_.size() + 1;
  top_channels_ = bottom[0]->channels();
  top_height_ = bottom[0]->height();
  top_width_ = bottom[0]->width();
  top[0]->Reshape(top_num_, top_channels_, top_height_, top_width_);
  caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());

  if (bottom[0]->num() > 1){
    LOG(FATAL) << "Only a batch size of 1 is currently supported" << std::endl;
  }

}

/*
 * bottom[0] = Unary (1xCxHxW)
 * bottom[1] = Detections (1xDx1x6)
 * top[0]    = Masked output ((D+1)xCxHxW)
 *
 */
template<typename Dtype>
void BoxFeaturesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_cpu_data();
  caffe_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->cpu_data();

  const int unary_batch_idx = 0; // assumes that bottom[0] has a batch size of 1

  for (int i = 0; i < detection_box_list_.size(); ++i) {

    const std::vector<int> &det_box = detection_box_list_[i]->get_foreground_pixels();

    assert(det_box.size() == 4 && "Detection should have exactly four co-ordinates - the top left and bottom right corners of the bounding box");
    int x_start = det_box[0];
    int y_start = det_box[1];
    int x_end = det_box[2];
    int y_end = det_box[3];

    x_start = std::max(0, x_start);
    y_start = std::max(0, y_start);
    x_end = std::min(bottom[0]->width() - 1, x_end);
    y_end = std::min(bottom[0]->height() - 1, y_end);

    Dtype det_score = detection_box_list_[i]->get_score();
    det_score = exp(det_score) / ( exp(det_score) + exp(1-det_score) );

    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int x = x_start; x <= x_end; ++x) {
        for (int y = y_start; y <= y_end; ++y) {
            output[output_blob->offset(i+1, c, y, x)] = unaries_in[unary_blob->offset(unary_batch_idx, c, y, x)] * det_score;
        }
      }
    }

  } // for

  if (is_background_det_){
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int x = 0; x < top_width_; ++x) {
        for (int y = 0; y < top_height_; ++y) {
          output[output_blob->offset(0, c, y, x)] = unaries_in[unary_blob->offset(unary_batch_idx, c, y, x)] * background_det_score_;
        }
      }
    }
  }


} // Forward_cpu

/*
 * bottom[0] = Unary (1xCxHxW)
 * bottom[1] = Detections (1xDx1x6)
 * top[0]    = Masked output ((D+1)xCxHxW)
 *
 */
template<typename Dtype>
void BoxFeaturesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  const Dtype* output_diff = output_blob->cpu_diff();
  Dtype* unaries_diff = unary_blob->mutable_cpu_diff();
  
  // initialise diff to be 0
  caffe_set(unary_blob->count(), Dtype(0), unaries_diff);

  const int unary_batch_idx = 0; // assumes that bottom[0] has a batch size of 1

  for (int i = 0; i < detection_box_list_.size(); ++i) {

    const std::vector<int> &det_box = detection_box_list_[i]->get_foreground_pixels();

    assert(det_box.size() == 4 && "Detection should have exactly four co-ordinates - the top left and bottom right corners of the bounding box");
    int x_start = det_box[0];
    int y_start = det_box[1];
    int x_end = det_box[2];
    int y_end = det_box[3];

    x_start = std::max(0, x_start);
    y_start = std::max(0, y_start);
    x_end = std::min(bottom[0]->width() - 1, x_end);
    y_end = std::min(bottom[0]->height() - 1, y_end);

    Dtype det_score = detection_box_list_[i]->get_score();
    det_score = exp(det_score) / ( exp(det_score) + exp(1-det_score) );

    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int x = x_start; x <= x_end; ++x) {
        for (int y = y_start; y <= y_end; ++y) {
          unaries_diff[unary_blob->offset(unary_batch_idx, c, y, x)] += output_diff[output_blob->offset(i+1, c, y, x)] * det_score;
        }
      }
    }
  }

  if (is_background_det_){
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int x = 0; x < top_width_; ++x) {
        for (int y = 0; y < top_height_; ++y) {
          unaries_diff[unary_blob->offset(unary_batch_idx, c, y, x)] += output_diff[output_blob->offset(0, c, y, x)] * background_det_score_;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BoxFeaturesLayer);
#endif

INSTANTIATE_CLASS(BoxFeaturesLayer);

REGISTER_LAYER_CLASS(BoxFeatures);
}