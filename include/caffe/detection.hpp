#ifndef DENSECRF_DETECTION_H
#define DENSECRF_DETECTION_H

#include <vector>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include "caffe/blob.hpp"

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace tvg {

    class Detection {

    private:
        int label;
        float score;
        std::vector<int> foreground_pixels;

        /**
         * For debugging purposes.
         */
        friend std::ostream &operator<<(std::ostream &, const Detection &);

        /**
         * Disable copy construction and copy assignment.
         */
        Detection(const Detection &);
        Detection &operator=(const Detection &);

    public:
        explicit Detection(const int label, const float score) : label(label), score(score) { }

        void add_foreground_pixel(const int pixel_id) {
            foreground_pixels.push_back(pixel_id);
        }

        int get_label() const { return label; }

        float get_score() const { return score; }

        float get_y_potentials_multiplier(float epsilon = 1e-6) const {
            return score / ((float) (foreground_pixels.size() + epsilon));
        };

        float get_x_potentials_mutiplier() const {
            return score;
        }

        const std::vector<int> & get_foreground_pixels() const { return foreground_pixels; }

        int get_num_pixels (void) const{
            return (int)foreground_pixels.size();
        }

        const static int num_per_detection = 6;
    };

    class DetectionPotentialConfig {

    private:

        float max_score_;

        float epsilon_;

        /**
        * Disable copy construction and copy assignment.
        */
        DetectionPotentialConfig(const DetectionPotentialConfig &);

        DetectionPotentialConfig &operator=(const DetectionPotentialConfig &);

    public:
        explicit DetectionPotentialConfig() : max_score_(0), epsilon_(0) { }

        float get_max_score() const {
            return max_score_;
        }

        void set_max_score(float max_score) {
            max_score_ = max_score;
        }

        float get_epsilon() const {
            return epsilon_;
        }

        void set_epsilon(float epsilon) {
            epsilon_ = epsilon;
        }
    };

    namespace DetectionUtils {

        /**
         * Reads detections from the given file and populates the given detection_list.
         */
        int read_detections_from_file(std::vector<boost::shared_ptr<const Detection> > &detection_list,
                const std::string &file_name, bool insert_empty_detections = false);

        template<typename Dtype>
        int read_detections_from_blob(std::vector<boost::shared_ptr<const Detection> > & detection_list,
                const caffe::Blob<Dtype> & detections){

            detection_list.clear();
            int num_detections = detections.channels();
            const Dtype* detection_data = detections.cpu_data();

            CHECK_EQ(detections.num(), 1) << "Expect one detection in the blob";
            CHECK_EQ(num_detections * tvg::Detection::num_per_detection, detections.count()) << "Invalid detection blob. Should contain " << num_detections * tvg::Detection::num_per_detection << " elements. Contains " << detections.count();

            int counter = 0;
            for (int i = 0; i < num_detections; ++i){
              const int det_label = static_cast<int>(detection_data[counter++]);
              const int x0 = static_cast<int>(detection_data[counter++]);
              const int y0 = static_cast<int>(detection_data[counter++]);
              const int x1 = static_cast<int>(detection_data[counter++]);
              const int y1 = static_cast<int>(detection_data[counter++]);
              const float det_score = static_cast<float>(detection_data[counter++]);

              boost::shared_ptr<Detection> detection(new Detection(det_label, det_score));
              detection->add_foreground_pixel(x0);
              detection->add_foreground_pixel(y0);
              detection->add_foreground_pixel(x1);
              detection->add_foreground_pixel(y1);

              detection_list.push_back(detection);
            }

          return num_detections;
        }

        /**
         * Exponentiate and normalize for detection binary variable marginals. Note that this function works only for
         * binary random variables. This function assumes that input and output Blobs are of same size.
         */
        template<typename Dtype>
        void exp_and_normalize_q_y(const caffe::Blob<Dtype> &bottom, caffe::Blob<Dtype> &top) {
            // Implementation has to be in the header file.

            const int detection_count = bottom.height() * bottom.width();

            Dtype v[2];

            const Dtype *in_data = bottom.cpu_data();
            Dtype *out_data = top.mutable_cpu_data();

            for (int det_id = 0; det_id < detection_count; ++det_id) {

                const Dtype mx = std::max(in_data[det_id], in_data[detection_count + det_id]);

                v[0] = exp(in_data[det_id] - mx);
                v[1] = exp(in_data[detection_count + det_id] - mx);

                const Dtype tot = v[0] + v[1];
                out_data[det_id] = v[0] / tot;
                out_data[detection_count + det_id] = v[1] / tot;
            }
        }

        /**
         * Works only for binary variables.
         */
        template<typename Dtype>
        void backprop_exp_and_normalize_q_y(const caffe::Blob<Dtype> &top, caffe::Blob<Dtype> &bottom) {

            const Dtype * const top_diff = top.cpu_diff();
            const Dtype * const top_data = top.cpu_data();

            Dtype * const bottom_diff = bottom.mutable_cpu_diff();

            const int channels = top.channels();
            const int spatial_dim = top.height() * top.width();

            Dtype *scale_data = new Dtype[spatial_dim];
            Dtype multiplier_data[2] = {1, 1};

            caffe::caffe_copy(top.count(), top_diff, bottom_diff);

            // compute dot(top_diff, top_data) and subtract them from the bottom diff
            for (int k = 0; k < spatial_dim; ++k) {
                scale_data[k] = caffe::caffe_cpu_strided_dot<Dtype>(channels,
                                                                    bottom_diff + k, spatial_dim,
                                                                    top_data + k, spatial_dim);
            }
            // subtraction
            caffe::caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 1,
                                         -1., multiplier_data, scale_data, 1., bottom_diff);

            // elementwise multiplication
            caffe::caffe_mul(top.count(), bottom_diff, top_data, bottom_diff);

            delete[] scale_data;
        }
    }
}


#endif //DENSECRF_DETECTION_H