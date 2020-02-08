#ifndef CAFFE_INSTANCE_ID_LAYER_HPP
#define CAFFE_INSTANCE_ID_LAYER_HPP

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/detection.hpp"

namespace caffe{
    template <typename Dtype>
    class InstanceIDLayer : public Layer<Dtype> {
    public:
        explicit InstanceIDLayer(const LayerParameter& param)
                : Layer<Dtype>(param) {}

        // This function is called once, and is basically the "Constructor"
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

        // This function is called before every call to "Forward"
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Instance ID Layer"; }
        virtual inline int ExactNumBottomBlobs() const { return 3; }
        virtual inline int ExactNumTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    //    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    //    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        vector<shared_ptr<const tvg::Detection> > detection_box_list_;
        vector<shared_ptr<const tvg::Detection> > detection_pixel_list_;
        std::string detection_boxes_input_dir_;
        std::string detection_pixels_input_dir_;

        // Keeps track of the normalisation constant at every pixel
        Blob<Dtype> channel_sums_;

        int top_channels_;
        int top_height_;
        int top_width_;

        inline int index(int n, int c, int h, int w){
            return ((n * top_channels_ + c) * top_height_ + h) * top_width_ + w;
        }

        inline int index_blob(int n, int c, int h, int w, const Blob<Dtype>* blob){
            return ((n * blob->channels() + c) * blob->height() + h) * blob->width() + w;
        }

        inline void Update_cross_diff(Dtype* unary_diff, const Dtype* top_diff, const Dtype* q_in, Blob<Dtype>* top, Blob<Dtype>* bottom, int det_label_from, int y, int x, Dtype chan_sum, Dtype det_score_to, Dtype det_score_from, int from) {

            const int n = 0; // Already assumed a batch size of 1. before
            const Dtype cross_diff = top_diff[index_blob(n, from, y, x, top)];

            const Dtype q = q_in[index_blob(n, det_label_from, y, x, bottom)];
            Dtype grad_from_zero = -(det_score_to * det_score_from * q) / (chan_sum * chan_sum);
            unary_diff[index_blob(n, 0, y, x, bottom)] += (grad_from_zero * cross_diff);
        }

        Dtype get_y_det_for_background (int y, int x, const Dtype* y_variables_out);

        Dtype background_prob_;
        Dtype epsilon_;

        int num_rescored_detections_;
        bool is_no_rescore_baseline_;

    }; // class InstanceIDLayer
} // namespace Caffe

#endif //CAFFE_INSTANCE_ID_LAYER_HPP
