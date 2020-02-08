#include "caffe/layer.hpp"
#include "caffe/layers/box_term_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"
#include "../../../include/caffe/layers/box_term_layer.hpp"

namespace caffe {

  // For this kernel function, bottom_data (unaries) and top_data(output of "V" distribution term), need to be pointers supplied with the correct offset (ie channel index)
  template <typename Dtype>
  __global__ void BoxTermForward(const int nthreads, const Dtype* bottom_data,
                                 const int width, int x_start, int y_start,
                                 int x_end, int y_end, Dtype det_score, Dtype* top_data) {

    CUDA_KERNEL_LOOP(index, nthreads){
      const int y = index / width;
      const int x = index % width;

      if (x >= x_start && x <= x_end && y >= y_start && y <= y_end){
        top_data[index] = bottom_data[index] * det_score;
      }
    }
  }

  // For this kernel function, bottom_data (unaries) and top_data(output of "V" distribution term), need to be pointers supplied with the correct offset (ie channel index)
  // The way this function is called means that there can be no race conditions by adding to "bottom_diff"
  template <typename Dtype>
  __global__ void BoxTermBackward(const int nthreads, const Dtype* top_diff,
                                 const int width, int x_start, int y_start,
                                 int x_end, int y_end, const Dtype det_score, Dtype* bottom_diff) {

    CUDA_KERNEL_LOOP(index, nthreads){
      const int y = index / width;
      const int x = index % width;

      if (x >= x_start && x <= x_end && y >= y_start && y <= y_end){
        bottom_diff[index] += (top_diff[index] * det_score);
      }
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
  void BoxTermLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  if (this->layer_param().box_term_param().force_cpu()){
      this->Forward_cpu(bottom, top);
      return;
  }

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_gpu_data();
  caffe_gpu_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->gpu_data();

  // assume batch size of 1
  // const int n = 0;

  // Actually easier to update these on CPU than on GPU
  const Dtype* y_variables_in = bottom[1]->cpu_data();
  Dtype* y_variables_out = top[1]->mutable_cpu_data();

  int det_counter = 0;
  const int num_pixels = unary_blob->width() * unary_blob->height();
  const int num_threads = num_pixels;
  const int width = unary_blob->width();

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

    if ( ( y_variables_in[num_rescored_detections_ + det_counter] != Dtype(0.) ) && (detection_pixel_list_[i]->get_num_pixels() > 0)  ){ // First condition is in cases when detection potentials were not enabled in previous CRF
      det_score = y_variables_in[num_rescored_detections_ + det_counter];
      ++det_counter;
    }

    const Dtype* bottom_data = unaries_in + det_label*num_pixels;
    Dtype* top_data = output + (i+1)*num_pixels;

    BoxTermForward<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
      num_threads, bottom_data, width, x_start, y_start, x_end, y_end, det_score, top_data);
    CUDA_POST_KERNEL_CHECK;

    y_variables_out[i] = det_score;

  } // for

  if (is_background_det_){
    const Dtype* bottom_data = unaries_in;
    Dtype* top_data = output;

    int x_start = 0; int x_end = top_width_ - 1;
    int y_start = 0; int y_end = top_height_ - 1;
    Dtype det_score = background_det_score_;

    BoxTermForward<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
    num_threads, bottom_data, width, x_start, y_start, x_end, y_end, det_score, top_data);
    CUDA_POST_KERNEL_CHECK;
  }

  } // Forward_gpu

  template <typename Dtype>
  void BoxTermLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    if (this->layer_param().box_term_param().force_cpu()){
        this->Backward_cpu(bottom, propagate_down, top);
        return;
    }

    Blob<Dtype>* output_blob = top[0];
    Blob<Dtype>* unary_blob = bottom[0];
    Blob<Dtype>* y_variable_blob = top[1];

    const Dtype* output_diff = output_blob->gpu_diff();
    Dtype* unaries_diff = unary_blob->mutable_gpu_diff();
    const Dtype* y_variables = y_variable_blob->cpu_data();

    // assume batch size of 1
    //const int n = 0;

    const int num_pixels = unary_blob->width() * unary_blob->height();
    const int num_threads = num_pixels;
    const int width = unary_blob->width();

    // initialise diff to be 0
    caffe_gpu_set(unary_blob->count(), Dtype(0), unaries_diff);

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
      x_end = std::min(unary_blob->width() - 1, x_end);
      y_end = std::min(unary_blob->height() - 1, y_end);

      const Dtype det_score = y_variables[i];

      const Dtype* top_diff = output_diff + (i+1)*num_pixels;
      Dtype* bottom_diff = unaries_diff + det_label*num_pixels;

      BoxTermBackward <Dtype><<< CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >>>
        (num_threads, top_diff, width, x_start, y_start, x_end, y_end, det_score, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }

    if (is_background_det_){
      const Dtype* top_diff = output_diff;
      Dtype* bottom_diff = unaries_diff;

      int x_start = 0; int x_end = top_width_ - 1;
      int y_start = 0; int y_end = top_height_ - 1;
      Dtype det_score = background_det_score_;

      BoxTermBackward <Dtype><<< CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >>>
      (num_threads, top_diff, width, x_start, y_start, x_end, y_end, det_score, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(BoxTermLayer);

} // namespace caffe