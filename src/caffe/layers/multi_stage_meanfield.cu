#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/meanfield_layers.hpp"

#include "caffe/util/tvg_ho_utils.hpp"

namespace caffe {

// Avoid divergence by uncoalescing access
    template <typename Dtype>
    __global__ void  computeBilateralKernel(const  int num_pixels_,
                                            const Dtype* const rgb_blob,
                                            const int width_, const int height_, const int channels_,
                                            float theta_alpha_, float theta_beta_,
                                            const int n, float* const output_kernel) {
        int offset = ((n * channels_ ) * height_) * width_ ;
        CUDA_KERNEL_LOOP(p, num_pixels_) {
            output_kernel[5 * p] = (float)(p % width_) / theta_alpha_;
            output_kernel[5 * p + 1] = (float)(p / width_) / theta_alpha_;
            const Dtype * const rgb_data_start = rgb_blob + offset;
            output_kernel[5 * p + 2] = (float)(rgb_data_start[p] / theta_beta_);
            output_kernel[5 * p + 3] = (float)((rgb_data_start + num_pixels_)[p] / theta_beta_);
            output_kernel[5 * p + 4] = (float)((rgb_data_start + num_pixels_ * 2)[p] / theta_beta_);
        }
    }

    template <typename Dtype>
    __global__ void computeNorm(Dtype* norm_output_data, int num_pixels){
        CUDA_KERNEL_LOOP(i, num_pixels) {
            norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
        }
    }

/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input/Output from the previous iteration (a copy of the unary terms if this is the first stage).
 * bottom[2] - RGB images
 * bottom[3] - Indices used for reading detection and superpixel files
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
    template <typename Dtype>
    void MultiStageMeanfieldLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

        if (init_cpu_ && this->layer_param().multi_stage_meanfield_param().force_cpu()){
            this->Forward_cpu(bottom, top);
            return;
        }

        if(init_cpu_) {
            LOG(FATAL)
            << ("You initialised your network on CPU, please initialise it on GPU.");
        }

        if (detection_potentials_enabled_) {
          init_detections(static_cast<int>(bottom[3]->cpu_data()[0]));
        }

        const Dtype* bottom_data = bottom[2]->gpu_data() ; // The RGB image
        split_layer_bottom_vec_[0] = bottom[0];
        split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

        // Initialize the bilateral lattices.
        computeBilateralKernel<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(
            num_pixels_, bottom_data, width_, height_, channels_,
            theta_alpha_, theta_beta_, 0,
            bilateral_kernel_buffer_); // The 0 refers to the offset (which is the case for a batch size of 1

        CUDA_POST_KERNEL_CHECK;
        bilateral_lattice_.reset(new ModifiedPermutohedral());
        bilateral_lattice_->init_gpu(bilateral_kernel_buffer_, 5, width_, height_);
        // Calculate bilateral filter normalization factors.
        Dtype* norm_output_data = bilateral_norms_.mutable_gpu_data();
        bilateral_lattice_->compute_gpu(norm_output_data, norm_feed_, 1);
        computeNorm<Dtype><<<CAFFE_GET_BLOCKS(num_pixels_), CAFFE_CUDA_NUM_THREADS>>>(norm_output_data, num_pixels_);
        CUDA_POST_KERNEL_CHECK;

        for (int i = 0; i < num_iterations_; ++i) {
            meanfield_iterations_[i]->PrePass(this->blobs_, bilateral_lattice_, &bilateral_norms_);

            if (is_no_class_weights_){
            meanfield_iterations_[i]->InitLocalCompatibility(num_, channels_, height_, width_);
            meanfield_iterations_[i]->Reshape(num_, channels_, height_, width_);
            }

            meanfield_iterations_[i]->Forward_gpu();
        }


        // Output the latent Y variables
        // Actually easier to do this on CPU, since that is where the data is updated
        Dtype * const y_top_data = top[1]->mutable_cpu_data();
        caffe_set(top[1]->count(), Dtype(0.), y_top_data);

        if (detection_potentials_enabled_) {
          const Dtype *const inferred_y_data = detection_y_qs_[num_iterations_]->cpu_data();
          const int aval_y_data_count = detection_y_qs_[num_iterations_]->count();
          for (int i = 0; i < aval_y_data_count; ++i) {
            y_top_data[i] = inferred_y_data[i];
          }
        }
    }

/**
 * Backprop through filter-based mean field inference.
 */

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Backward_gpu(
        const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {

    if (init_cpu_ && this->layer_param().multi_stage_meanfield_param().force_cpu()){
        this->Backward_cpu(bottom, propagate_down, top);
        return;
    }

    if(init_cpu_){
        LOG(FATAL) << ("You initialize your network on CPU, please initialize it on GPU.");
    }

    if (detection_potentials_enabled_ && detection_count_ > 0) {
      caffe_set(detection_y_qs_[num_iterations_]->count(), Dtype(0.), detection_y_qs_[num_iterations_]->mutable_cpu_diff());
    }

    for (int i = (num_iterations_ - 1); i >= 0; --i) {
        meanfield_iterations_[i]->Backward_gpu();
    }

    vector<bool> split_layer_propagate_down(1, true);
    split_layer_->Backward(split_layer_top_vec_, split_layer_propagate_down, split_layer_bottom_vec_);

    // Accumulate diffs from mean field iterations.
    for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {

        Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();

        if (this->param_propagate_down_[blob_id]) {

            caffe_gpu_set(cur_blob->count(), Dtype(0), cur_blob->mutable_gpu_diff());

            for (int i = 0; i < num_iterations_; ++i) {
                const Dtype* diffs_to_add = meanfield_iterations_[i]->blobs()[blob_id]->gpu_diff();
                caffe_gpu_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_gpu_diff());
            }
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiStageMeanfieldLayer);

}  // namespace caffe