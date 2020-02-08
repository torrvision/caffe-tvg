#include "caffe/layer.hpp"
#include "caffe/layers/transfer_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe {

    /*
    * bottom[0] = Unary
    * bottom[1] = Indices for loading detection files (same as meanfield layer)
       * Or bottom[1] could be a list of "detection labels". In this case, you would not have to read and parse the detection
       * This is the case if "is_no_detections" is true
    * top[0]    = Output. Ie, the V distribution
    *
    * This function is called once, and is basically the "Constructor"
    */
    template<typename Dtype>
    void TransferLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

        this->detection_boxes_input_dir_ = this->layer_param().transfer_param().detection_box_input_dir();
        this->is_no_detections_ = this->layer_param().transfer_param().no_detections();
        this->copy_bg_ = this->layer_param().transfer_param().copy_bg();
    }

/*
 * This function is called before every call of "Forward"
 */
    template<typename Dtype>
    void TransferLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        detection_box_list_.clear();

        if (!is_no_detections_) {
          const int image_id = static_cast<int>(bottom[1]->cpu_data()[0]);
          tvg::DetectionUtils::read_detections_from_file(detection_box_list_, detection_boxes_input_dir_ + "/" +
                                                                                                           boost::lexical_cast<std::string>(
                                                                                                                   image_id) +
                                                                                                           ".detections"); //TODO: Specify extension, instead of hardcoding ".detections"

          top_channels_ = (int) detection_box_list_.size() + 1;
        }
        else{
          top_channels_ = bottom[1]->count();
        }

        top_height_ = bottom[0]->height();
        top_width_ = bottom[0]->width();
        top[0]->Reshape(bottom[0]->num(), top_channels_, top_height_, top_width_);
        caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());

        if (top[0]->num() > 1) {
            LOG(FATAL) << "Only a batch size of 1 is currently supported" << std::endl;
        }
    }

/*
 * Only a batch size of 1 is currently supported.
 */
    template<typename Dtype>
    void TransferLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

        Blob<Dtype>* output_blob = top[0];
        Blob<Dtype>* unary_blob = bottom[0];

        Dtype* output = top[0]->mutable_cpu_data();
        caffe_set(output_blob->count(), Dtype(0), output);
        const Dtype* unaries_in = bottom[0]->cpu_data();

        // assume batch size of 1
        //const int n = 0;

        const int num_pixels = unary_blob->width() * unary_blob->height();
        // copy unaries for the background (0th) label
        if (copy_bg_) {
          caffe_cpu_copy(num_pixels, unaries_in, output);
        }

        int range = detection_box_list_.size();
        if (is_no_detections_){
          range = bottom[1]->count();
        }

        for (int i = 0; i < range; ++i) {

          int det_label;
          if (is_no_detections_){
            det_label = static_cast<int>(bottom[1]->cpu_data()[i]);
          } else {
            det_label = detection_box_list_[i]->get_label();
          }

          int copy_index = (i+1);
          if (!copy_bg_){
            copy_index = i;
          }

          caffe_cpu_copy(num_pixels, unaries_in + det_label * num_pixels, output + copy_index * num_pixels);

        } // for

    } // Forward_cpu

/*
 * top[0] = instance unary
 * bottom[0] = segmentation unary
 * bottom[1] = index for reading detections
 */
    template<typename Dtype>
    void TransferLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                            const vector<Blob<Dtype> *> &bottom) {

        Blob<Dtype>* output_blob = top[0];
        Blob<Dtype>* unary_blob = bottom[0];

        const Dtype* output_diff = output_blob->cpu_diff();

        Dtype *unaries_diff = unary_blob->mutable_cpu_diff();

        // assume batch size of 1
        // const int n = 0;
        const int num_pixels = unary_blob->width() * unary_blob->height();

        // Initialise diff to be 0
        // Actually, we only do this if "propogate_down[0]" is true
        // Otherwise, we accumulate the gradients
        // This is necessary for the shape_term layer
        if (propagate_down[0]) {
          caffe_set(unary_blob->count(), Dtype(0), unaries_diff);
        }

        // diff for background class
        if (copy_bg_) {
          caffe_cpu_copy(num_pixels, output_diff, unaries_diff);
        }

        int range = detection_box_list_.size();
        if (is_no_detections_){
          range = bottom[1]->count();
        }

        for (int i = 0; i < range; ++i) {

            int det_label;
            if (is_no_detections_){
              det_label = static_cast<int>(bottom[1]->cpu_data()[i]);
              CHECK_GE(det_label, 0);
            }else{
              det_label = detection_box_list_[i]->get_label();
              CHECK_GT(det_label, 0);
            }

            int copy_index = (i+1);
            if (!copy_bg_){
              copy_index = i;
            }

            // y = a*x + b*y
            // https://software.intel.com/en-us/node/520858
            caffe_cpu_axpby(num_pixels, Dtype(1), output_diff + copy_index*num_pixels, Dtype(1), unaries_diff + num_pixels*det_label);

        }
    }

    #ifdef CPU_ONLY
    STUB_GPU(TransferLayer);
    #endif

    INSTANTIATE_CLASS(TransferLayer);

    REGISTER_LAYER_CLASS(Transfer);
}
