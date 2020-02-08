#include "caffe/layer.hpp"
#include "caffe/layers/roi_unpooling.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe {

/*
* bottom[0] = Input (1 x h x w).
* bottom[1] = 4 numbers.
 * First, desired width (w_d). Second, desired height (h_d). Desired width and height must be bigger than the width and height of the input
 * Third, w_offset. Ie from where to start "filling in". Fourth,  h_offset
* top[0]    = Output. Of size 1 x w_d x h_d
* top[1]    = The final Y variables
*
* This function is called once, and is basically the "Constructor"
*/
template<typename Dtype>
void ROIUnpoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

}

/*
 * This function is called before every call of "Forward"
 */
template<typename Dtype>
void ROIUnpoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  int width_out =  static_cast<int>(bottom[1]->cpu_data()[0]);
  int height_out = static_cast<int>(bottom[1]->cpu_data()[1]);

  CHECK_GE(width_out, bottom[0]->width());
  CHECK_GE(height_out, bottom[0]->height());

  CHECK_LE(bottom[0]->width() + bottom[1]->cpu_data()[2], width_out);
  CHECK_LE(bottom[0]->height() + bottom[1]->cpu_data()[3], height_out);

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), height_out, width_out);

  if (top[0]->num() > 1){
    LOG(FATAL) << "Only a batch size of 1 is currently supported" << std::endl;
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
void ROIUnpoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  Dtype* output = output_blob->mutable_cpu_data();
  caffe_set(output_blob->count(), Dtype(0), output);
  const Dtype* unaries_in = bottom[0]->cpu_data();

  const int n = 0;

  for (int c = 0; c < bottom[0]->channels(); ++c) {

    const int x_start = static_cast<int>(bottom[1]->cpu_data()[2 + 2*c]);
    const int y_start = static_cast<int>(bottom[1]->cpu_data()[3 + 2*c]);
    const int x_end = x_start + bottom[0]->width();
    const int y_end = y_start + bottom[0]->height();

    CHECK_LE(x_end, top[0]->width());
    CHECK_LE(y_end, top[0]->height());

    // These checks are actually not necessary
    /*x_start = std::max(0, x_start);
    y_start = std::max(0, y_start);
    x_end = std::min(top[0]->width() - 1, x_end);
    y_end = std::min(top[0]->height() - 1, y_end);*/
    int x_count = 0;

    for (int x = x_start; x < x_end; ++x) {
      int y_count = 0;
      for (int y = y_start; y < y_end; ++y) {
        output[output_blob->offset(n, c, y, x)] = unaries_in[unary_blob->offset(n, c, y_count, x_count)];
        ++y_count;
      }
      ++x_count;
    }

  } // for

} // Forward_cpu

/*
 * bottom[0] = Unary
 * bottom[1] = 4 numbers. w_desired, h_desired, w_offset, h_offset
 * top[0]    = Output.
 *
 * Only a batch size of 1 is currently supported.
 */
template<typename Dtype>
void ROIUnpoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype> *> &bottom) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* unary_blob = bottom[0];

  const Dtype* output_diff = output_blob->cpu_diff();
  Dtype* unaries_diff = unary_blob->mutable_cpu_diff();

  // assume batch size of 1
  const int n = 0;
  
  // initialise diff to be 0
  caffe_set(unary_blob->count(), Dtype(0), unaries_diff);

  for (int c = 0; c < bottom[0]->channels(); ++c) {

    const int x_start = static_cast<int>(bottom[1]->cpu_data()[2 + 2*c]);
    const int y_start = static_cast<int>(bottom[1]->cpu_data()[3 + 2*c]);
    const int x_end = x_start + bottom[0]->width();
    const int y_end = y_start + bottom[0]->height();

    CHECK_LE(x_end, top[0]->width());
    CHECK_LE(y_end, top[0]->height());

    int x_count = 0;

    for (int x = x_start; x < x_end; ++x) {
      int y_count = 0;
      for (int y = y_start; y < y_end; ++y) {
        unaries_diff[unary_blob->offset(n, c, y_count, x_count)] = output_diff[output_blob->offset(n, c, y, x)];
        ++y_count;
      }
      ++x_count;
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU(ROIUnpoolingLayer);
#endif

INSTANTIATE_CLASS(ROIUnpoolingLayer);

REGISTER_LAYER_CLASS(ROIUnpooling);
}
