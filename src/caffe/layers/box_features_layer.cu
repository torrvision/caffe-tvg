#include "caffe/layer.hpp"
#include "caffe/layers/box_features_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"
//#include "../../../include/caffe/layers/box_features_layer.hpp"

namespace caffe {

  // For this kernel function, bottom_data (unaries) and top_data(output of "V" distribution term), need to be pointers supplied with the correct offset (ie channel index)
  template <typename Dtype>
  __global__ void BoxFeaturesForward(const int nthreads, const Dtype* bottom_data,
                                  const int num_pixels, const int width, int x_start, int y_start,
                                 int x_end, int y_end, Dtype det_score, Dtype* top_data) {

    CUDA_KERNEL_LOOP(index, nthreads){
      const int index_in_cur_chn = index % num_pixels;
      const int y = index_in_cur_chn / width;
      const int x = index_in_cur_chn % width;

      if (x >= x_start && x <= x_end && y >= y_start && y <= y_end){
        top_data[index] = bottom_data[index] * det_score;
      }
    }
  }

  // For this kernel function, bottom_data (unaries) and top_data(output of "V" distribution term), need to be pointers supplied with the correct offset (ie channel index)
  // The way this function is called means that there can be no race conditions by adding to "bottom_diff"
  template <typename Dtype>
  __global__ void BoxFeaturesBackward(const int nthreads, const Dtype* top_diff,
                                 const int num_pixels, const int width, int x_start, int y_start,
                                 int x_end, int y_end, const Dtype det_score, Dtype* bottom_diff) {

    CUDA_KERNEL_LOOP(index, nthreads){
      const int index_in_cur_chn = index % num_pixels;
      const int y = index_in_cur_chn / width;
      const int x = index_in_cur_chn % width;

      if (x >= x_start && x <= x_end && y >= y_start && y <= y_end){
        bottom_diff[index] += (top_diff[index] * det_score);
      }
    }
  }


/*
 * bottom[0] = Unary (1xCxHxW)
 * bottom[1] = Detections (1xDx1x6)
 * top[0]    = Masked output ((D+1)xCxHxW)
 *
 */
  template<typename Dtype>
  void BoxFeaturesLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_gpu_data();
  caffe_gpu_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->gpu_data();

  const int num_pixels = unary_blob->width() * unary_blob->height();
  const int num_features = unary_blob->width() * unary_blob->height() * unary_blob->channels();
  const int num_threads = num_features;
  const int channels = unary_blob->channels();
  const int width = unary_blob->width();

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

    const Dtype* bottom_data = unaries_in;
    Dtype* top_data = output + (i+1)*num_threads;

    BoxFeaturesForward<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
      num_threads, bottom_data, num_pixels, width, x_start, y_start, x_end, y_end, det_score, top_data);
    CUDA_POST_KERNEL_CHECK;

  } // for

  if (is_background_det_){
    const Dtype* bottom_data = unaries_in;
    Dtype* top_data = output;

    int x_start = 0; int x_end = top_width_ - 1;
    int y_start = 0; int y_end = top_height_ - 1;
    Dtype det_score = background_det_score_;

    BoxFeaturesForward<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
    num_threads, bottom_data, num_pixels, width, x_start, y_start, x_end, y_end, det_score, top_data);
    CUDA_POST_KERNEL_CHECK;
  }

  } // Forward_gpu

  template <typename Dtype>
  void BoxFeaturesLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    Blob<Dtype>* output_blob = top[0];
    Blob<Dtype>* unary_blob = bottom[0];

    const Dtype* output_diff = output_blob->gpu_diff();
    Dtype* unaries_diff = unary_blob->mutable_gpu_diff();

    // assume batch size of 1
    //const int n = 0;

    const int num_pixels = unary_blob->width() * unary_blob->height();
    const int num_features = unary_blob->width() * unary_blob->height() * unary_blob->channels();
    const int num_threads = num_features;
    const int channels = unary_blob->channels();
    const int width = unary_blob->width();

    // initialise diff to be 0
    caffe_gpu_set(unary_blob->count(), Dtype(0), unaries_diff);

    for (int i = 0; i < detection_box_list_.size(); ++i) {

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

      Dtype det_score = detection_box_list_[i]->get_score();
      det_score = exp(det_score) / ( exp(det_score) + exp(1-det_score) );

      Dtype* bottom_diff = unaries_diff;
      const Dtype* top_diff = output_diff + (i+1)*num_threads;
      BoxFeaturesBackward <Dtype><<< CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >>>
        (num_threads, top_diff, num_pixels, width, x_start, y_start, x_end, y_end, det_score, bottom_diff);
      CUDA_POST_KERNEL_CHECK;

    }

    if (is_background_det_){
      Dtype* bottom_diff = unaries_diff;
      const Dtype* top_diff = output_diff;

      int x_start = 0; int x_end = top_width_ - 1;
      int y_start = 0; int y_end = top_height_ - 1;
      Dtype det_score = background_det_score_;

      BoxFeaturesBackward <Dtype><<< CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >>>
      (num_threads, top_diff, num_pixels, width, x_start, y_start, x_end, y_end, det_score, bottom_diff);
      CUDA_POST_KERNEL_CHECK;
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(BoxFeaturesLayer);

} // namespace caffe