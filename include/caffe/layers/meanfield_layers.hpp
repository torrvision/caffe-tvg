#ifndef CAFFE_MF_LAYER_HPP_
#define CAFFE_MF_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/detection.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"

#include "caffe/util/modified_permutohedral.hpp"
#include <boost/shared_array.hpp>


namespace caffe {

    template <typename Dtype>
    class MultiStageMeanfieldLayer;

    template <typename Dtype>
    class MeanfieldIteration {

    public:

        bool is_first_iteration_; // TODO: a nasty hack, fix later.

        /**
         * Every MeanfieldIteration must belong to a {@link MultiStageMeanfieldLayer}.
         */
        explicit MeanfieldIteration(MultiStageMeanfieldLayer<Dtype> * const msmf_parent) :
                is_first_iteration_(false), msmf_parent_(msmf_parent) { }

        /**
         * Must be invoked only once after the construction of the layer.
         */
        void OneTimeSetUp(
                Blob<Dtype> * const unary_terms,
                Blob<Dtype> * const additional_unary_terms,
                Blob<Dtype> * const softmax_input,
                Blob<Dtype> * const output_blob,
                Blob<Dtype> * const detection_y_input,
                Blob<Dtype> * const detection_y_output,
                const shared_ptr<ModifiedPermutohedral> & spatial_lattice,
                const Blob<Dtype> * const spatial_norm);

        /**
         * Must be invoked before invoking {@link Forward_cpu()}
         */
        void PrePass(
                const vector<shared_ptr<Blob<Dtype> > > &  parameters_to_copy_from,
                const shared_ptr<ModifiedPermutohedral> & bilateral_lattice,
                const Blob<Dtype> * const bilateral_norms);

        /*
         * Must be invoked before calling {@link Forward_cpu()}
         * Only valid in the case that "is_no_class_weight" is enabled
         */
        void InitLocalCompatibility(int n, int c, int h, int w);
        void Reshape(int n, int c, int h, int w);
        /**
         * Forward pass - to be called during inference.
         */
        void Forward_cpu();
        void Forward_gpu();

        /**
         * Backward pass - to be called during training.
         */
        void Backward_cpu();
        void Backward_gpu();

        // A quick hack. This should be properly encapsulated.
        vector<shared_ptr<Blob<Dtype> > >& blobs() {
            return blobs_;
        }

        /*
         * For debugging
         */
        void set_iteration_number (int it);
        void print_blobs(std::string prefix = "");


    private:
        void compute_detection_potential_update();
        void compute_detection_potential_diffs();

        vector<shared_ptr<Blob<Dtype> > > blobs_;

        MultiStageMeanfieldLayer<Dtype> * const msmf_parent_;
        int count_;
        int num_;
        int channels_;
        int height_;
        int width_;
        int num_pixels_;

        Blob<Dtype> spatial_out_blob_;
        Blob<Dtype> bilateral_out_blob_;
        Blob<Dtype> pairwise_;
        Blob<Dtype> prob_;
        Blob<Dtype> message_passing_;
        Blob<Dtype> detection_potentials_for_x_;

        vector<Blob<Dtype>*> softmax_top_vec_;
        vector<Blob<Dtype>*> softmax_bottom_vec_;
        vector<Blob<Dtype>*> sum_top_vec_;
        vector<Blob<Dtype>*> sum_bottom_vec_;

        shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
        shared_ptr<EltwiseLayer<Dtype> > sum_layer_;

        shared_ptr<ModifiedPermutohedral> spatial_lattice_;
        shared_ptr<ModifiedPermutohedral> bilateral_lattice_;
        const Blob<Dtype>* spatial_norm_;
        const Blob<Dtype>* bilateral_norms_;

        /**
         * Detection potential related stuff.
         */
        bool detection_potentials_enabled_;
        Blob<Dtype>* detection_y_q_input_;
        Blob<Dtype>* detection_y_q_output_;
        Blob<Dtype> detection_y_q_before_softmax_;

        /**
         * Higher-order potential related stuff.
         */
        bool ho_potentials_enabled_;

        // Other
        bool is_no_class_weights_;
        Blob<Dtype> local_compatibility_matrix_;

        int iteration_number_;

        // for storing temporary data during backward pass of no_class_weights meanfield
        Blob<Dtype> tmp;
        Blob<Dtype> tmp_ones;
        Blob<Dtype> tmp_sum;
    };


    template <typename Dtype>
    class MultiStageMeanfieldLayer : public Layer<Dtype> {

        friend class MeanfieldIteration<Dtype>;

    public:
        explicit MultiStageMeanfieldLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
        virtual ~MultiStageMeanfieldLayer();

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        /*virtual inline LayerParameter_LayerType type() const {
            return LayerParameter_LayerType_MULTI_STAGE_MEANFIELD;
        }*/
        virtual inline const char* type() const {
            return "MultiStageMeanfield";
        }

        virtual inline int ExactNumBottomBlobs() const { return 4; }
        virtual inline int ExactNumTopBlobs() const { return 2; }

        // For debugging
        virtual void print_blob_sizes(void);

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        #ifndef CPU_ONLY
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        #endif

    private:
        void compute_spatial_kernel(float* const output_kernel);
        void compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel);
        void init_param_blobs(const MultiStageMeanfieldParameter & meanfield_param);
        void init_detections(const int image_id);
        void init_detection_y_unaries();
        void init_ho_potentials(const Blob<Dtype> & unaries, const int image_id);
        void init_spatial_lattice(void);
        void init_bilateral_buffers(void);

        int count_;
        int num_;
        int channels_;
        int height_;
        int width_;
        int num_pixels_;

        int num_iterations_;
        Dtype theta_alpha_;
        Dtype theta_beta_;
        Dtype theta_gamma_;

        Dtype* norm_feed_;  // The permutehedral lattice is not templated.
        Blob<Dtype> spatial_norm_;
        Blob<Dtype> bilateral_norms_;

        vector<Blob<Dtype>*> split_layer_bottom_vec_;
        vector<Blob<Dtype>*> split_layer_top_vec_;

        // Blobs owned by this instance of the class.
        vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;
        vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;

        vector<shared_ptr<MeanfieldIteration<Dtype> > > meanfield_iterations_;

        shared_ptr<SplitLayer<Dtype> > split_layer_;

        // Additional unary
        shared_ptr<SplitLayer<Dtype> > au_split_layer_;
        vector<shared_ptr<Blob<Dtype> > > au_blobs_;
        vector<Blob<Dtype> *> au_split_layer_bottom_vec_;
        vector<Blob<Dtype> *> au_split_layer_top_vec_;

        shared_ptr<SoftmaxLayer<Dtype> > ho_softmax_;
        vector<Blob<Dtype> *> ho_softmax_bottom_vec_;
        vector<Blob<Dtype> *> ho_softmax_top_vec_;
        Blob<Dtype> unary_prob_; // FIXME: Not very efficient. Since the same softmax normalisation (with the same input data) will be done in the first mean-field iteration

        // Permutohedral lattice
        shared_ptr<ModifiedPermutohedral> spatial_lattice_;
        //boost::shared_array<float> bilateral_kernel_buffer_;
        float * bilateral_kernel_buffer_;       // Permutohedral lattice is not templated
        shared_ptr<ModifiedPermutohedral> bilateral_lattice_;

        /** start -- stuff related to detection potentials **/
        bool detection_potentials_enabled_;
        bool detection_dat_check_enabled_;
        tvg::DetectionPotentialConfig detection_potentials_config_;
        int train_dataset_size_;
        int cur_train_image_id_;
        std::string detection_dat_dir_;

        vector<shared_ptr<const tvg::Detection> > detection_list_;
        int detection_count_;
        vector<shared_ptr<Blob<Dtype> > > detection_y_qs_;
        shared_ptr<Blob<Dtype> > detection_y_unary_; // Made this a shared_ptr to be consistent with detection_y_qs_.
        /** end -- stuff related to detection potentials **/

        /** start - stuff related to higher order potentials */
        bool ho_potentials_enabled_;
        std::string ho_dat_input_dir_;
        int ho_num_layers_;

        vector< vector< vector<int> > > ho_cliques_;
        vector< vector< vector<float> > > ho_stat_potentials_;
        vector< vector< vector<float> > > ho_stat_potentials_diff_;
        /** end - stuff related to higher order potentials */

        /* GPU/CPU stuff */

        bool init_cpu_;
        bool init_gpu_;

        // Other
        bool is_no_class_weights_;

    };

}//namespace caffe
#endif