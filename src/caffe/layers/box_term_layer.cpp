#include "caffe/layer.hpp"
#include "caffe/layers/box_term_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe {

/*
* bottom[0] = Unary
* bottom[1] = Y variables from the CRF with detection potentials
* bottom[2] = Indices for loading detection files (same as meanfield layer)
* bottom[3] = Actual detections as a Nx6 blob, where N is the number of detections. [det_class x0, y0, x1, y1, det_score] are the order of the detection
* top[0]    = Output. Ie, the V distribution
* top[1]    = The final Y variables
*
* This function is called once, and is basically the "Constructor"
*/
template<typename Dtype>
void BoxTermLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  this->detection_boxes_input_dir_ = this->layer_param().box_term_param().detection_box_input_dir();
  this->detection_pixels_input_dir_ = this->layer_param().box_term_param().detection_pixel_input_dir();
  this->is_no_rescore_baseline_ = this->layer_param().box_term_param().is_no_rescore_baseline();

  this->is_background_det_ = this->layer_param().box_term_param().is_background_det();
  this->background_det_score_ = Dtype(this->layer_param().box_term_param().background_det_score());

  this->read_det_from_blob_ = (bottom.size() == 4);

  if (is_background_det_){
    std::cout << "Background det enabled" << std::endl;
    std::cout << "Background det score: " << background_det_score_ << std::endl;
  }
  else{
    std::cout << "Background det disabled" << std::endl;
  }

}

/*
 * This function is called before every call of "Forward"
 */
template<typename Dtype>
void BoxTermLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  detection_box_list_.clear();
  detection_pixel_list_.clear();

  if (!this->read_det_from_blob_) {
    const int image_id = static_cast<int>(bottom[2]->cpu_data()[0]);
    tvg::DetectionUtils::read_detections_from_file(detection_box_list_, detection_boxes_input_dir_ + "/" +
                                                                        boost::lexical_cast<std::string>(image_id) +
                                                                        ".detections"); //TODO: Specify extension, instead of hardcoding ".detections"
    this->num_rescored_detections_ = tvg::DetectionUtils::read_detections_from_file(detection_pixel_list_,
                                                                                    detection_pixels_input_dir_ + "/" +
                                                                                    boost::lexical_cast<std::string>(
                                                                                            image_id) + ".detections",
                                                                                    true); //TODO: Specify extension, instead of hardcoding ".detections"
  }
  else{
    tvg::DetectionUtils::read_detections_from_blob(detection_box_list_, *bottom[3]);
    this->num_rescored_detections_ = 0;
    is_no_rescore_baseline_ = true;
  }

  top_channels_ = (int)detection_box_list_.size()+1; top_height_ = bottom[0]->height(); top_width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), top_channels_, top_height_, top_width_);
  caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());
  top[1]->Reshape(bottom[0]->num(), 1, 1, (int)detection_box_list_.size());

  if (top[0]->num() > 1){
    LOG(FATAL) << "Only a batch size of 1 is currently supported" << std::endl;
  }
}

/*
 * bottom[0] = Unary
 * bottom[1] = Y variables from the CRF with detection potentials
 * bottom[2] = Indices for loading detection files (same as meanfield layer)
 * top[0]    = Output. Ie, the V distribution
 * top[1]    = The final Y variables
 *
 * Only a batch size of 1 is currently supported.
 */
template<typename Dtype>
void BoxTermLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_cpu_data();
  caffe_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->cpu_data();

  // assume batch size of 1
  const int n = 0;

  const Dtype* y_variables_in = bottom[1]->cpu_data();
  Dtype* y_variables_out = top[1]->mutable_cpu_data();

  int det_counter = 0;

  for (int i = 0; i < detection_box_list_.size(); ++i) {

    const std::vector<int> &det_box = detection_box_list_[i]->get_foreground_pixels();
    const int det_label = detection_box_list_[i]->get_label();

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

    if ( !this->is_no_rescore_baseline_
         && (y_variables_in[num_rescored_detections_ + det_counter] != Dtype(0.)) // In cases when detection potentials were not enabled in previous CRF
         && (detection_pixel_list_[i]->get_num_pixels() > 0)
       ){
      det_score = y_variables_in[num_rescored_detections_ + det_counter];
      ++det_counter;
    }

    for (int x = x_start; x <= x_end; ++x) {
        for (int y = y_start; y <= y_end; ++y) {
            output[output_blob->offset(n, i+1, y, x)] = unaries_in[unary_blob->offset(n, det_label, y, x)] * det_score;
        }
    }

    y_variables_out[i] = det_score;

  } // for

  if (is_background_det_){
    for (int x = 0; x < top_width_; ++x) {
      for (int y = 0; y < top_height_; ++y) {
        output[output_blob->offset(n, 0, y, x)] = unaries_in[unary_blob->offset(n, 0, y, x)] * background_det_score_;
      }
    }
  }


} // Forward_cpu

/*
 * top[0] = instance unary
 * top[1] = final y variables
 * bottom[0] = segmentation unary
 * bottom[1] = input y variables
 * bottom[2] = index for reading detections
 */
template<typename Dtype>
void BoxTermLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];
  Blob<Dtype>* y_variable_blob = top[1];

  const Dtype* output_diff = output_blob->cpu_diff();
  Dtype* unaries_diff = unary_blob->mutable_cpu_diff();
  const Dtype* y_variables = y_variable_blob->cpu_data();

  // assume batch size of 1
  const int n = 0;
  
  // initialise diff to be 0
  caffe_set(unary_blob->count(), Dtype(0), unaries_diff);

  for (int i = 0; i < detection_box_list_.size(); ++i) {

    const int det_label = detection_box_list_[i]->get_label();
    CHECK_GT(det_label, 0);

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

    const Dtype det_score = y_variables[i];

    for (int x = x_start; x <= x_end; ++x) {
      for (int y = y_start; y <= y_end; ++y) {
        unaries_diff[unary_blob->offset(n, det_label, y, x)] += output_diff[output_blob->offset(n, i+1, y, x)] * det_score;
      }
    }
  }

  if (is_background_det_){
    for (int x = 0; x < top_width_; ++x) {
      for (int y = 0; y < top_height_; ++y) {
        unaries_diff[unary_blob->offset(n, 0, y, x)] += output_diff[output_blob->offset(n, 0, y, x)] * background_det_score_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BoxTermLayer);
#endif

INSTANTIATE_CLASS(BoxTermLayer);

REGISTER_LAYER_CLASS(BoxTerm);
}
