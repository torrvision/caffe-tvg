#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/iouaccuracy_layer.hpp"

#include <cmath>
#include <string>
#include <stdio.h>
#include <time.h>

#include <cfloat>

namespace caffe {

    template<typename Dtype>
    void IOUAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                             const vector<Blob<Dtype> *> &top) {

        has_ignore_label_ = this->layer_param_.iou_accuracy_param().has_ignore_label();
        if (has_ignore_label_) {
            ignore_label_ = this->layer_param_.iou_accuracy_param().ignore_label();
        }

        /** Init related to IOU accuracy **/
        if (this->layer_param_.iou_accuracy_param().has_test_iterations()) {
            test_iterations_ = this->layer_param_.iou_accuracy_param().test_iterations();
        } else {
            test_iterations_ = Caffe::get_max_iter();
        }
        count_ = bottom[0]->count();
        num_ = bottom[0]->num();
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        pixel_num_ = height_ * width_;

        pred_labels_.reset(new int[pixel_num_]);
        caffe_set(pixel_num_, 0, pred_labels_.get());

        num_classes_ = channels_;
        if ( this->layer_param_.iou_accuracy_param().num_classes() > 0 ){
            num_classes_ = this->layer_param_.iou_accuracy_param().num_classes();
        }

        gt_counts_.reset(new int[num_classes_]);
        pred_counts_.reset(new int[num_classes_]);
        intersect_counts_.reset(new int[num_classes_]);

        caffe_set(num_classes_, 0, gt_counts_.get());
        caffe_set(num_classes_, 0, pred_counts_.get());
        caffe_set(num_classes_, 0, intersect_counts_.get());

        LOG(INFO) << "Configured IOU Accuracy layer. test_iterations " << test_iterations_ << " with " << num_classes_ << " classes. \n";
    }

    template<typename Dtype>
    void IOUAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
        if (top.size() > 1){
          std::cout << "IOU Layer should only have one top" << std::endl;
          LOG(FATAL) << "IOU Layer should only have one top";
        }

        top[0]->Reshape(1,1,1,1 + num_classes_);
        caffe_set(num_classes_, Dtype(0), top[0]->mutable_cpu_data());
    }

/**
 * bottom[0] - Probabilities
 * bottom[1] - Labels
 *
 * top[0] - Softmax loss
 */
    template<typename Dtype>
    void IOUAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {
        Dtype* top_data = top[0]->mutable_cpu_data();

        int cur_iter = Caffe::get_cur_iter();

        // Playing safe.
        if ( cur_iter == 0) {
            caffe_set(num_classes_, 0, gt_counts_.get());
            caffe_set(num_classes_, 0, pred_counts_.get());
            caffe_set(num_classes_, 0, intersect_counts_.get());
        }

        const Dtype *likelihood_data = bottom[0]->cpu_data();
        const Dtype *gt_data = bottom[1]->cpu_data();

        // Store predicted label for each pixel
        for (int i = 0; i < pixel_num_; ++i) {

            if (static_cast<int>(gt_data[i]) == ignore_label_) {
                continue;
            }

            Dtype cur_max = likelihood_data[i];
            int cur_label = 0;
            for (int c = 1; c < bottom[0]->channels(); ++c) {
                Dtype cur_val = likelihood_data[c * pixel_num_ + i];
                if (cur_val > cur_max) {
                    cur_max = cur_val;
                    cur_label = c;
                }
            }
            pred_labels_[i] = cur_label;
        }

        // Update counts
        for (int i = 0; i < pixel_num_; ++i) {

            const int gt_value = static_cast<int>(gt_data[i]);
            if (gt_value == ignore_label_) {
                continue;
            }
            const int pred_value = pred_labels_[i];

            if (gt_value < 0 || gt_value >= num_classes_){
                std::cout << "[" << this->layer_param().name() << "] ";
                std::cout << "Pixel num: " << i << ". Illegal ground truth value of "<< gt_value << std::endl;
                exit(1);
            }

            ++pred_counts_[pred_value];
            ++gt_counts_[gt_value];

            if (pred_value == gt_value) {
                ++intersect_counts_[pred_value];
            }
        }

        if (cur_iter == test_iterations_ - 1) {

            double tot = 0.0;
            int actual_num_classes = 0;

            for (int c = 0; c < num_classes_; ++c) {

                const double denominator = static_cast<double>(gt_counts_[c] + pred_counts_[c] - intersect_counts_[c] );

                if ( denominator != double(0) ){
                    double cur_acc = 100 * (static_cast<double>(intersect_counts_[c]) / denominator );
                    tot += cur_acc;
                    ++actual_num_classes;
                    top_data[c + 1] = cur_acc * test_iterations_;
                    
                    LOG(INFO) << "Accuracy for class " << c << ": " << cur_acc; 
                }
            }

            LOG(INFO) << "IOU Accuracy " << (tot / actual_num_classes);

            // reset variables
            caffe_set(num_classes_, 0, gt_counts_.get());
            caffe_set(num_classes_, 0, pred_counts_.get());
            caffe_set(num_classes_, 0, intersect_counts_.get());

            top_data[0] = (tot / actual_num_classes) * test_iterations_; // Multiply because Caffe averages this number automatically
        }
        else{
            top_data[0] = 0;
        }

    }

    template<typename Dtype>
    void IOUAccuracyLayer<Dtype>::Backward_cpu(
            const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
            const vector<Blob<Dtype> *> &bottom) {

        LOG(INFO) << "Backward_cpu() on IOU Accuracy. Should NOT happen.";
        exit(1);
    }

    INSTANTIATE_CLASS(IOUAccuracyLayer);

    REGISTER_LAYER_CLASS(IOUAccuracy);
}  // namespace caffe

