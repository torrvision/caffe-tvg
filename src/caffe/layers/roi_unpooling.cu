#include "caffe/layer.hpp"
#include "caffe/layers/roi_unpooling.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

namespace caffe {

  // For this kernel function, bottom_data ("small" data) and top_data(padded data), need to be pointers supplied with the correct offset (ie channel index)
  template <typename Dtype>
  __global__ void RoiUnpoolingForward(const int nthreads, const Dtype* bottom_data,
                                      const int width, int x_start, int y_start,
                                      int x_end, int y_end, int bottom_width, Dtype* top_data) {

    CUDA_KERNEL_LOOP(index, nthreads){
      const int y = index / width;
      const int x = index % width;

      const int bottom_index = (y - y_start)*bottom_width + (x - x_start);

      if (x >= x_start && x < x_end && y >= y_start && y < y_end){
        top_data[index] = bottom_data[bottom_index];
      }
    }
  }

  // For this kernel function, bottom_data ("small" data) and top_data(padded data), need to be pointers supplied with the correct offset (ie channel index)
  template <typename Dtype>
  __global__ void RoiUnpoolingBackward(const int nthreads, const Dtype* top_diff,
                                       const int width, int x_start, int y_start,
                                       int x_end, int y_end, int bottom_width, Dtype* bottom_diff) {

    CUDA_KERNEL_LOOP(index, nthreads){
      const int y = index / width;
      const int x = index % width;

      const int bottom_index = (y - y_start)*bottom_width + (x - x_start);

      if (x >= x_start && x < x_end && y >= y_start && y < y_end){
        bottom_diff[bottom_index] = top_diff[index];
      }
    }
  }

  /*
   * bottom[0] = Unary
   * bottom[1] = 4 numbers. w_desired, h_desired, w_offset, h_offset
   * top[0]    = Output.
   *
   * Only a batch size of 1 is currently supported.
   */
  template<typename Dtype>
  void ROIUnpoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

    Blob<Dtype>* output_blob = top[0];
    Blob<Dtype>* unary_blob = bottom[0];

    Dtype* output = output_blob->mutable_gpu_data();
    caffe_gpu_set(output_blob->count(), Dtype(0), output);
    const Dtype* unaries_in = bottom[0]->gpu_data();

    // assume batch size of 1
    // const int n = 0;

    // Actually easier to update these on CPU than on GPU
    const int bottom_num_pixels = unary_blob->width() * unary_blob->height();
    const int top_num_pixels = output_blob->width() * output_blob->height();
    const int num_threads = top_num_pixels;
    const int top_width = output_blob->width();
    const int bottom_width = unary_blob->width();

    for (int c = 0; c < bottom[1]->channels(); ++c) {

      const int x_start = static_cast<int>(bottom[1]->cpu_data()[2 + 2*c]);
      const int y_start = static_cast<int>(bottom[1]->cpu_data()[3 + 2*c]);
      const int x_end = x_start + bottom[0]->width();
      const int y_end = y_start + bottom[0]->height();

      CHECK_LE(x_end, top[0]->width());
      CHECK_LE(y_end, top[0]->height());

      const Dtype* bottom_data = unaries_in + c*bottom_num_pixels;
      Dtype* top_data = output + c*top_num_pixels;

      RoiUnpoolingForward<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(
      num_threads, bottom_data, top_width, x_start, y_start, x_end, y_end, bottom_width, top_data);
      CUDA_POST_KERNEL_CHECK;

    } // for

  } // Forward_gpu

  template <typename Dtype>
  void ROIUnpoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

    Blob<Dtype>* output_blob = top[0];
    Blob<Dtype>* unary_blob = bottom[0];

    const Dtype* output_diff = output_blob->gpu_diff();
    Dtype* unaries_diff = unary_blob->mutable_gpu_diff();

    // assume batch size of 1
    //const int n = 0;

    const int bottom_num_pixels = unary_blob->width() * unary_blob->height();
    const int top_num_pixels = output_blob->width() * output_blob->height();
    const int num_threads = top_num_pixels;
    const int top_width = output_blob->width();
    const int bottom_width = unary_blob->width();

    // initialise diff to be 0
    caffe_gpu_set(unary_blob->count(), Dtype(0), unaries_diff);

    for (int c = 0; c < bottom[0]->channels(); ++c) {

      const int x_start = static_cast<int>(bottom[1]->cpu_data()[2 + 2*c]);
      const int y_start = static_cast<int>(bottom[1]->cpu_data()[3 + 2*c]);
      const int x_end = x_start + bottom[0]->width();
      const int y_end = y_start + bottom[0]->height();

      CHECK_LE(x_end, top[0]->width());
      CHECK_LE(y_end, top[0]->height());


      const Dtype* top_diff = output_diff + c*top_num_pixels;
      Dtype* bottom_diff = unaries_diff + c*bottom_num_pixels;

      RoiUnpoolingBackward <Dtype><<< CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS >>>
      (num_threads, top_diff, top_width, x_start, y_start, x_end, y_end, bottom_width, bottom_diff);
      CUDA_POST_KERNEL_CHECK;

    }
  } // Backward GPU

  INSTANTIATE_LAYER_GPU_FUNCS(ROIUnpoolingLayer);

} // namespace caffe