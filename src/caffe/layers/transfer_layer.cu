#include "caffe/layer.hpp"
#include "caffe/layers/transfer_layer.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe{

    template<typename Dtype>
    void TransferLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

        if (this->layer_param().transfer_param().force_cpu()){
            this->Forward_cpu(bottom, top);
            return;
        }

        Blob<Dtype>* output_blob = top[0];
        Blob<Dtype>* unary_blob = bottom[0];

        Dtype* output = output_blob->mutable_gpu_data();
        caffe_gpu_set(output_blob->count(), Dtype(0), output);
        const Dtype* unaries_in = unary_blob->gpu_data();

        const int num_pixels = unary_blob->width() * unary_blob->height();

        // copy unaries for the background (0th) label
        if (copy_bg_) {
          caffe_gpu_copy(num_pixels, unaries_in, output);
        }

        int range = detection_box_list_.size();
        if (is_no_detections_){
          range = bottom[1]->count();
        }

        for (int i = 0; i < range; ++i){

          int det_label;
          if (is_no_detections_){
            det_label = static_cast<int>(bottom[1]->cpu_data()[i]);
            CHECK_GE(det_label, 0);
          } else{
            det_label = detection_box_list_[i]->get_label();
            CHECK_GT(det_label, 0);
          }

          int copy_index = (i+1);
          if (!copy_bg_){
            copy_index = i;
          }
          caffe_gpu_copy(num_pixels, unaries_in + det_label * num_pixels, output + copy_index*num_pixels);
        }
    }

    template<typename Dtype>
    void TransferLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {

        if (this->layer_param().transfer_param().force_cpu()){
            this->Backward_cpu(bottom, propagate_down, top);
            return;
        }

        Blob<Dtype>* output_blob = top[0];
        Blob<Dtype>* unary_blob = bottom[0];

        const Dtype* output_diff = output_blob->gpu_diff();

        Dtype* unaries_diff = unary_blob->mutable_gpu_diff();

        const int num_pixels = unary_blob->width() * unary_blob->height();

        // initialise diff to be 0
        if (propagate_down[0]) {
          caffe_gpu_set(unary_blob->count(), Dtype(0), unaries_diff);
        }

        // diff for background class
        if (copy_bg_){
          caffe_gpu_copy(num_pixels, output_diff, unaries_diff);
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
            } else{
              det_label = detection_box_list_[i]->get_label();
              CHECK_GT(det_label, 0);
            }

            int copy_index = (i+1);
            if (!copy_bg_){
              copy_index = i;
            }

            // y = a*x + y
            // https://software.intel.com/en-us/node/520858
            caffe_gpu_axpy(num_pixels, Dtype(1), output_diff + copy_index*num_pixels, unaries_diff + num_pixels*det_label);
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(TransferLayer);
}