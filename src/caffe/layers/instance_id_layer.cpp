#include "caffe/layer.hpp"
#include "caffe/layers/instance_id_layer.hpp"

//for debugging
#include "caffe/util/tvg_common_utils.hpp"

#include <boost/lexical_cast.hpp>

namespace caffe{

    /*
    * bottom[0] = Unary
    * bottom[1] = Y Variables
    * bottom[2] = Indices for loading detection files (same as meanfield layer)
    * top[0]    = Output. Ie, the V distribution
    *
    * This function is called once, and is basically the "Constructor"
    */
    template <typename Dtype>
    void InstanceIDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

       this->detection_boxes_input_dir_ = this->layer_param().instance_id_param().detection_box_input_dir();
       this->detection_pixels_input_dir_ = this->layer_param().instance_id_param().detection_pixel_input_dir();
       this->is_no_rescore_baseline_ = this->layer_param().instance_id_param().is_no_rescore_baseline();

       this->background_prob_ =  exp(1) / ( exp(1) + exp(0) );

       this->epsilon_ = 1e-8;
    }

    /*
     * This function is called before every call of "Forward"
     */
    template <typename Dtype>
    void InstanceIDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
       detection_box_list_.clear();
       detection_pixel_list_.clear();

       const int image_id = static_cast<int>(bottom[2]->cpu_data()[0]);
       tvg::DetectionUtils::read_detections_from_file(detection_box_list_, detection_boxes_input_dir_ + "/" +
                                                                       boost::lexical_cast<std::string>(image_id) + ".detections"); //TODO: Specify extension, instead of hardcoding ".detections"
       this->num_rescored_detections_ = tvg::DetectionUtils::read_detections_from_file(detection_pixel_list_, detection_pixels_input_dir_ + "/" +
                                                                           boost::lexical_cast<std::string>(image_id) + ".detections", true); //TODO: Specify extension, instead of hardcoding ".detections"

       top_channels_ = (int)detection_box_list_.size()+1; top_height_ = bottom[0]->height(); top_width_ = bottom[0]->width();
       top[0]->Reshape(bottom[0]->num(), top_channels_, top_height_, top_width_);
       caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());
       top[1]->Reshape(bottom[0]->num(), 1, 1, (int)detection_box_list_.size());

        // Keeps track of the normalisation constant at every pixel
       channel_sums_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
       caffe_set(channel_sums_.count(), Dtype(0.), channel_sums_.mutable_cpu_data());
       Dtype* channel_sums_data = channel_sums_.mutable_cpu_data();

        // If only we could index matrices ... out_data(:,1,:,:) in matlab = following in c++
       Dtype* out_data = top[0]->mutable_cpu_data();
       for (int n = 0; n < top[0]->num(); ++n){
           for (int h = 0; h < top_height_; ++h){
               for (int w = 0; w < top_width_; ++w){
                   out_data[index(n, 0, h, w)] = background_prob_;
                   channel_sums_data[index(n, 0, h, w)] = background_prob_;
               }
           }
       }

       if (top[0]->num() > 1){
           LOG(FATAL) << "Only a batch size of 1 is currently supported" << std::endl;
       }
    }

    /*
     * Only a batch size of 1 is currently supported.
     */
    template <typename Dtype>
    void InstanceIDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        Dtype* out_data = top[0]->mutable_cpu_data();
        const Blob<Dtype>* q = bottom[0];
        const Dtype* q_data = bottom[0]->cpu_data();
        Dtype* channel_sums_data = channel_sums_.mutable_cpu_data();
        const int n = 0; // We have already assumed a batch size of 1!

        const Dtype* y_variables_in = bottom[1]->cpu_data();
        Dtype* y_variables_out = top[1]->mutable_cpu_data();

        int det_counter = 0;

        // assign probabilities
        for( int i = 1; i <= detection_box_list_.size(); ++i){
            const std::vector<int> & det_box = detection_box_list_[i-1]->get_foreground_pixels();

            assert(det_box.size() == 4 && "Detection should have exactly four co-ordinates - the top left and bottom right corners of the bounding box");
            int x_start = det_box[0];
            int y_start = det_box[1];
            int x_end   = det_box[2];
            int y_end   = det_box[3];

            x_start = std::max(0, x_start); y_start = std::max(0, y_start);
            x_end = std::min(q->width()-1, x_end); y_end = std::min(q->height()-1, y_end);

            Dtype det_score = detection_box_list_[i-1]->get_score(); // TODO: This is just the test. We will actually be using the Y variable output when we get it.
            det_score = exp(det_score) / ( exp(det_score) + exp(1-det_score) );

            if (detection_pixel_list_[i-1]->get_num_pixels() > 0 && ( y_variables_in[num_rescored_detections_ + det_counter] != Dtype(0.) ) ){ // Second condition is in cases when detection potentials were not enabled in previous CRF
                det_score = y_variables_in[num_rescored_detections_ + det_counter];
                ++det_counter;
            }

            const Dtype det_label = detection_box_list_[i-1]->get_label();

            for (int x = x_start; x <= x_end; ++x){
                for (int y = y_start; y <= y_end; ++y){
                    out_data[index(n, i, y, x)] = det_score * q_data[index_blob(n, det_label, y, x, q)];

                    if (out_data[index(n,0,y,x)] == background_prob_){ // This condition can only be true if a pixel has not been "visited by a detection" yet.
                        out_data[index(n,0,y,x)] = 0;
                        channel_sums_data[index_blob(n,0,y,x,&channel_sums_)] = 0;
                    }

                    if (out_data[index(n,0,y,x)] == 0) {
                        Dtype term1 = det_score * (1 - q_data[index_blob(n, det_label, y, x, q)]);
                        Dtype term2 = (1 - det_score) * q_data[index_blob(n, 0, y, x, q)];

                        term1 = det_score * q_data[index_blob(n, 0, y, x, q)];
                        term2 = 0;
                        out_data[index(n, 0, y, x)] += (term1 + term2);

                        channel_sums_data[index_blob(n, 0, y, x, &channel_sums_)] += (term1 + term2);
                    }

                    channel_sums_data[index_blob(n, 0, y, x, &channel_sums_)] += out_data[index(n, i, y, x)];
                }


            }

            y_variables_out[i-1] = det_score;
        }

        // normalise probabilities
        // There should be no zero values for us to worry about
        for (int c = 0; c < top[0]->channels(); ++c){
            for (int h = 0; h < top[0]->height(); ++h){
                for (int w = 0; w < top[0]->width(); ++w) {
                    out_data[index(n,c,h,w)] = out_data[index(n,c,h,w)] / channel_sums_data[index_blob(n,0,h,w,&channel_sums_)];
                }
            }
        }

        caffe_add_scalar(top[0]->count(), epsilon_, out_data);
    }

    /*
     * top[0] = instance unary
     * top[1] = final y variable output
     * bottom[0] = segmentation unary
     * bottom[1] = y variable input
     * bottom[2] = index for reading detections
     */
    template <typename Dtype>
    void InstanceIDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_out = top[0]->cpu_data();
        const Dtype* q_in = bottom[0]->cpu_data();
        Dtype* unary_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* y_variables_out = top[1]->cpu_data();
        const Dtype* channel_sums_data = channel_sums_.cpu_data();

        const int n = 0;

        caffe_set(bottom[0]->count(), Dtype(0.), unary_diff);

        // Gradient with respect to the segmentation unaries ...
        for( int i = 1; i <= detection_box_list_.size(); ++i){
            const std::vector<int> & det_box = detection_box_list_[i-1]->get_foreground_pixels();

            CHECK_EQ(det_box.size(), 4) << "Detection should have exactly four co-ordinates - the top left and bottom right corners of the bounding box";

            int x_start = det_box[0];
            int y_start = det_box[1];
            int x_end   = det_box[2];
            int y_end   = det_box[3];

            x_start = std::max(0, x_start); y_start = std::max(0, y_start);
            x_end = std::min(bottom[0]->width()-1, x_end); y_end = std::min(bottom[0]->height()-1, y_end);

            const Dtype det_score = y_variables_out[i-1];
            const Dtype det_label = detection_box_list_[i-1]->get_label();

            for (int x = x_start; x <= x_end; ++x){
                for (int y = y_start; y <= y_end; ++y){

                    // Update for Q(l = l_k) where l_k is the label of the k^th detection
                    // This is not the "cross term"
                    Dtype diff_from_top = top_diff[index_blob(n, i, y, x, top[0])];

                    const Dtype chan_sum = channel_sums_data[index_blob(n, 0, y, x, &channel_sums_)];
                    Dtype q = q_in[index_blob(n, det_label, y, x, bottom[0])];

                    const Dtype denominator = chan_sum * chan_sum;
                    Dtype numerator = det_score*chan_sum - det_score*det_score*q;
                    unary_diff[index_blob(n, det_label, y, x, bottom[0])] = (numerator/denominator) * diff_from_top;

                    // Now we need to update Q(l = 0) as well
                    // Note, in the forward pass, we only updated Q(l=0) for the first overlapping detection.
                    // Here, we do it, but only updating if the current diff is 0
                    if (unary_diff[index_blob(n,0,y,x,bottom[0])] == Dtype(0.)){
                        diff_from_top = top_diff[index_blob(n, 0, y, x, top[0])];

                        q = q_in[index_blob(n, 0, y, x, bottom[0])];
                        numerator = det_score*chan_sum - det_score*det_score*q;
                        unary_diff[index_blob(n, 0, y, x, bottom[0])] = (numerator/denominator) * diff_from_top;
                    }

                    // Cross term update for Q(l = 0).
                    const Dtype cross_diff = top_diff[index_blob(n, i, y, x, top[0])];
                    Dtype cross_det_score = get_y_det_for_background(y, x, y_variables_out);

                    q = q_in[index_blob(n, det_label, y, x, bottom[0])];
                    Dtype grad_from_zero = -(cross_det_score*det_score*q) / (chan_sum * chan_sum);
                    unary_diff[index_blob(n, 0, y, x, bottom[0])] +=  ( grad_from_zero * cross_diff) ;


                    // Add the cross terms between other outputs and this unary Q(i = l_k) now.
                    for (int c = 0; c < top[0]->channels(); ++c) {

                        if ( (top_out[index_blob(n, c, y, x, top[0])] > epsilon_) && (c != i) ) {

                            int cross_det_label = 0;
                            Dtype cross_detection_score;
                            if (c > 0) {
                                cross_det_label = detection_box_list_[c-1]->get_label();
                                cross_detection_score = y_variables_out[c-1]; //TODO: Work this out properly
                            }
                            else{
                                cross_detection_score = get_y_det_for_background(y, x, y_variables_out);
                            }

                            const Dtype cross_diff_from_top = top_diff[index_blob(n, c, y, x, top[0])];

                            q = q_in[index_blob(n, cross_det_label, y, x, bottom[0])];
                            const Dtype cross_grad = -(cross_detection_score * det_score * q) / (chan_sum * chan_sum);
                            unary_diff[index_blob(n, det_label, y, x, bottom[0])] += (cross_grad *
                                                                                      cross_diff_from_top);
                        }
                    }

                }
            }
        }
    }

    /*
     * Due to the way that we work out our detection score for the background, the actual score for the y variable can be ambiguous
     * So we have a function to work this out
     */
    template <typename Dtype>
    Dtype InstanceIDLayer<Dtype>::get_y_det_for_background(int y, int x, const Dtype* y_variables_out) {

        for (int i = 0; i < detection_box_list_.size(); ++i){
            const std::vector<int> & det_box = detection_box_list_[i]->get_foreground_pixels();

            int x_start = det_box[0];
            int y_start = det_box[1];
            int x_end   = det_box[2];
            int y_end   = det_box[3];

            x_start = std::max(0, x_start); y_start = std::max(0, y_start);
            x_end = std::min(top_width_-1, x_end); y_end = std::min(top_height_-1, y_end);

            if ( x >= x_start && x <= x_end && y >= y_start && y <= y_end){
                return y_variables_out[i];
            }
        }
        LOG(FATAL) << ("[InstanceIDLayer::get_y_det_for_background] Should not end up reaching here");
    }

    INSTANTIATE_CLASS(InstanceIDLayer);
    REGISTER_LAYER_CLASS(InstanceID);
}
