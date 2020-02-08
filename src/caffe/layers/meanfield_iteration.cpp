#include <vector>
#include <caffe/util/tvg_ho_utils.hpp>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/meanfield_layers.hpp"
#include "caffe/detection.hpp"

#include "caffe/util/tvg_common_utils.hpp"

namespace caffe {

/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::OneTimeSetUp(
    Blob<Dtype> * const unary_terms,
    Blob<Dtype> * const additional_unary_terms,
    Blob<Dtype> * const softmax_input,
    Blob<Dtype> * const output_blob,
    Blob<Dtype> * const detection_y_q_input,
    Blob<Dtype> * const detection_y_q_output,
    const shared_ptr<ModifiedPermutohedral> & spatial_lattice,
    const Blob<Dtype> * const spatial_norm) {

  detection_potentials_enabled_ = msmf_parent_->detection_potentials_enabled_;
  ho_potentials_enabled_ = msmf_parent_->ho_potentials_enabled_;

  spatial_lattice_ = spatial_lattice;
  spatial_norm_ = spatial_norm;

  count_ = unary_terms->count();
  num_ = unary_terms->num();
  channels_ = unary_terms->channels();
  height_ = unary_terms->height();
  width_ = unary_terms->width();
  num_pixels_ = height_ * width_;

  detection_y_q_input_ = detection_y_q_input;
  detection_y_q_output_ = detection_y_q_output;

  is_no_class_weights_ = msmf_parent_->is_no_class_weights_;

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Meanfield iteration skipping parameter initialization.";
  } else {
    blobs_.resize(7);
    blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // spatial kernel weight
    blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // bilateral kernel weight
    blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // compatibility transform matrix
    blobs_[3].reset(new Blob<Dtype>(1, 1, 1, channels_)); // detection X weights
    blobs_[4].reset(new Blob<Dtype>(1, 1, 1, channels_)); // detection Y weights
    blobs_[5].reset(new Blob<Dtype>(1, 1, 1, msmf_parent_->ho_num_layers_)); // superpixel weights for additional_unaries_1
    blobs_[6].reset(new Blob<Dtype>(1, 1, 1, msmf_parent_->ho_num_layers_)); // legacy blob. retained for compatibility with previous model snapshots
  }

  if (!msmf_parent_->detection_potentials_enabled_) {
    // Uninitialized diffs cause problems.
    caffe_set(blobs_[3]->count(), Dtype(0.), blobs_[3]->mutable_cpu_diff());
    caffe_set(blobs_[4]->count(), Dtype(0.), blobs_[4]->mutable_cpu_diff());
  }

  if (!msmf_parent_->ho_potentials_enabled_) {
    // Uninitialized diffs cause problems.
    caffe_set(blobs_[5]->count(), Dtype(0.), blobs_[5]->mutable_cpu_diff());
  }

  pairwise_.Reshape(num_, channels_, height_, width_);
  spatial_out_blob_.Reshape(num_, channels_, height_, width_);
  bilateral_out_blob_.Reshape(num_, channels_, height_, width_);
  message_passing_.Reshape(num_, channels_, height_, width_);
  if (detection_potentials_enabled_) {
    detection_potentials_for_x_.Reshape(num_, channels_, height_, width_);
  }

  // Softmax layer configuration
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input);

  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);

  LayerParameter softmax_param;
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // Sum layer configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(unary_terms);
  sum_bottom_vec_.push_back(&pairwise_);
  if (detection_potentials_enabled_) {
    sum_bottom_vec_.push_back(&detection_potentials_for_x_);
  }
  if (ho_potentials_enabled_) {
    sum_bottom_vec_.push_back(additional_unary_terms);
  }
  sum_top_vec_.clear();
  sum_top_vec_.push_back(output_blob);

  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.0));
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(-1.0));
  if (detection_potentials_enabled_) {
    sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.0));
  }
  if (ho_potentials_enabled_) {
    sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.0));
  }
  sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);

  if (this->is_no_class_weights_) {
      tmp_sum.Reshape(1, 1, 1, 1);
      tmp.Reshape(spatial_out_blob_.shape());
      tmp_ones.Reshape(spatial_out_blob_.shape());
  }
}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::PrePass(
    const vector<shared_ptr<Blob<Dtype> > > & parameters_to_copy_from,
    const shared_ptr<ModifiedPermutohedral> & bilateral_lattice,
    const Blob<Dtype> * const bilateral_norms) {

  bilateral_lattice_ = bilateral_lattice;
  bilateral_norms_ = bilateral_norms;

  // Get copies of the up-to-date parameters.
  for (int i = 0; i < parameters_to_copy_from.size(); ++i) {
    blobs_[i]->CopyFrom(*(parameters_to_copy_from[i].get()));
  }

}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::Forward_cpu() {

  //------------------------------- Softmax normalization--------------------
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  //-----------------------------------Message passing-----------------------
  Dtype * const spatial_out_data = spatial_out_blob_.mutable_cpu_data();
  const Dtype * const prob_input_data = prob_.cpu_data();

  spatial_lattice_->compute_cpu(spatial_out_data, prob_input_data, channels_, false);
  const Dtype * const spatial_norm_data = spatial_norm_->cpu_data();
  // Pixel-wise normalization.
  for (int channel_id = 0; channel_id < channels_; ++channel_id) {
    caffe_mul(num_pixels_, spatial_norm_data,
              spatial_out_data + channel_id * num_pixels_,
              spatial_out_data + channel_id * num_pixels_);
  }

  Dtype * const bilateral_out_data = bilateral_out_blob_.mutable_cpu_data();

  bilateral_lattice_->compute_cpu(bilateral_out_data, prob_input_data, channels_, false);
  const Dtype *const bilateral_norm_cpu_data = bilateral_norms_->cpu_data();
  // Pixel-wise normalization.
  for (int channel_id = 0; channel_id < channels_; ++channel_id) {
    caffe_mul(num_pixels_, bilateral_norm_cpu_data,
              bilateral_out_data + channel_id * num_pixels_,
              bilateral_out_data + channel_id * num_pixels_);
  }

  Dtype * const message_passing_data = message_passing_.mutable_cpu_data();
  caffe_set(count_, Dtype(0.), message_passing_data);

  if (!is_no_class_weights_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                            this->blobs_[0]->cpu_data(), spatial_out_blob_.cpu_data(),
                            (Dtype) 0., message_passing_data);

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                            this->blobs_[1]->cpu_data(), bilateral_out_blob_.cpu_data(),
                            (Dtype) 1., message_passing_data);
  }
  else{
      caffe_axpy<Dtype>(count_, (this->blobs_[0]->cpu_data())[0],
                        spatial_out_blob_.cpu_data(), message_passing_.mutable_cpu_data());

      caffe_axpy<Dtype>(count_, (this->blobs_[1]->cpu_data())[0],
                        bilateral_out_blob_.cpu_data(), message_passing_.mutable_cpu_data());
  }

  //--------------------------- Compatibility multiplication ----------------
  //Result from message passing needs to be multiplied with compatibility values.
  if(!is_no_class_weights_){
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
                            channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
                            message_passing_.cpu_data(), (Dtype) 0.,
                            pairwise_.mutable_cpu_data());
  }
  else{
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,  channels_, num_pixels_,
                            channels_, (Dtype) 1, local_compatibility_matrix_.cpu_data(),
                            message_passing_.cpu_data(), (Dtype) 0.,
                            pairwise_.mutable_cpu_data());
  }

  //---------------------------- Detection potentials -----------------------
  if (detection_potentials_enabled_ && msmf_parent_->detection_count_ > 0) {
    compute_detection_potential_update();
  } else if (detection_potentials_enabled_ && msmf_parent_->detection_count_ <= 0) {
    caffe_set(count_, Dtype(0.), detection_potentials_for_x_.mutable_cpu_data());
  }

  //------------------------- Adding different potentials, normalization is left to the next iteration --------------
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::compute_detection_potential_update() {

  const Dtype * const prob_input_data = prob_.cpu_data();
  const std::size_t detection_count = msmf_parent_->detection_list_.size();

  detection_y_q_before_softmax_.ReshapeLike((*detection_y_q_input_));
  Dtype * const y_update_data = detection_y_q_before_softmax_.mutable_cpu_data();
  caffe_copy(msmf_parent_->detection_y_unary_->count(), msmf_parent_->detection_y_unary_->cpu_data(), y_update_data);

  Dtype * const x_update_data = detection_potentials_for_x_.mutable_cpu_data();
  caffe_set(detection_potentials_for_x_.count(), Dtype(0.), x_update_data);

  const Dtype * const x_weights_data = blobs_[3]->cpu_data();
  const Dtype * const y_weights_data = blobs_[4]->cpu_data();

  const Dtype * const detection_y_input_data = detection_y_q_input_->cpu_data();

  // Compute the mean field updates for each detection.
  for (int det_id = 0; det_id < detection_count; ++det_id) {

      const shared_ptr<const tvg::Detection> & cur_detection = msmf_parent_->detection_list_[det_id];
      const vector<int> & foreground_pixels = cur_detection->get_foreground_pixels();

      const int det_label = cur_detection->get_label();

      const float x_multiplier = x_weights_data[det_label] * cur_detection->get_x_potentials_mutiplier();
      const float y_multiplier = y_weights_data[det_label] * cur_detection->get_y_potentials_multiplier();

      const float update_from_y_0 = detection_y_input_data[det_id] * x_multiplier;
      const float update_from_y_1 = detection_y_input_data[detection_count + det_id] * x_multiplier;

      const std::size_t num_foreground_pixels = foreground_pixels.size();
      for (int i = 0; i < num_foreground_pixels; ++i) {

        const int pixel_id = foreground_pixels[i];
        const Dtype x_q_at_det_label = prob_input_data[det_label * num_pixels_ + pixel_id];

        // Q(Y_d = 0) update
        y_update_data[det_id] -= x_q_at_det_label * y_multiplier;
        // Q(Y_d = 1) update
        y_update_data[detection_count + det_id] -= (1 - x_q_at_det_label) * y_multiplier;

        // Q(X_i) updates
        for (int label_id = 0; label_id < channels_; ++label_id) {

          if (label_id == det_label) {
            // Q(X_i = l_d) update.
            x_update_data[label_id * num_pixels_ + pixel_id] -= update_from_y_0;
          } else {
            // Q(X_i != l_d) update.
            x_update_data[label_id * num_pixels_ + pixel_id] -= update_from_y_1;
          }
        }
      }
    }

  tvg::DetectionUtils::exp_and_normalize_q_y<Dtype>(detection_y_q_before_softmax_, *detection_y_q_output_);
}


template<typename Dtype>
void MeanfieldIteration<Dtype>::Backward_cpu() {

  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down(sum_bottom_vec_.size(), true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

  Dtype * const prob_diff = prob_.mutable_cpu_diff();
  caffe_set(prob_.count(), Dtype(0.), prob_diff); // Do this regardless of if detections or HOs are enabled or not.

  if (detection_potentials_enabled_ && msmf_parent_->detection_count_ > 0) {
    compute_detection_potential_diffs();
  } else if (detection_potentials_enabled_ && msmf_parent_->detection_count_ <= 0) {
    caffe_set(blobs_[3]->count(), Dtype(0.), blobs_[3]->mutable_cpu_diff());
    caffe_set(blobs_[4]->count(), Dtype(0.), blobs_[4]->mutable_cpu_diff());
  }

  //---------------------------- Update compatibility diffs ------------------
  caffe_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_cpu_diff());

  if (!is_no_class_weights_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                            num_pixels_, (Dtype) 1., pairwise_.cpu_diff(),
                            message_passing_.cpu_data(), (Dtype) 1.,
                            this->blobs_[2]->mutable_cpu_diff());
  }
  //-------------------------- Gradient after compatibility transform--- -----
  if (!is_no_class_weights_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                            channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
                            pairwise_.cpu_diff(), (Dtype) 0.,
                            message_passing_.mutable_cpu_diff());
  }
  else{
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                            channels_, (Dtype) 1., local_compatibility_matrix_.cpu_data(),
                            pairwise_.cpu_diff(), (Dtype) 0.,
                            message_passing_.mutable_cpu_diff());
  }
  // ------------------------- Gradient w.r.t. kernels weights ------------

  caffe_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_cpu_diff());
  caffe_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_cpu_diff());

  if(!is_no_class_weights_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                            num_pixels_, (Dtype) 1., message_passing_.cpu_diff(),
                            spatial_out_blob_.cpu_data(), (Dtype) 1.,
                            this->blobs_[0]->mutable_cpu_diff());

      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                            num_pixels_, (Dtype) 1., message_passing_.cpu_diff(),
                            bilateral_out_blob_.cpu_data(), (Dtype) 1.,
                            this->blobs_[1]->mutable_cpu_diff());

      // TODO: Check whether there's a way to improve the accuracy of this calculation.
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                            this->blobs_[0]->cpu_data(), message_passing_.cpu_diff(),
                            (Dtype) 0., spatial_out_blob_.mutable_cpu_diff());

      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                            this->blobs_[1]->cpu_data(), message_passing_.cpu_diff(),
                            (Dtype) 0., bilateral_out_blob_.mutable_cpu_diff());
  }
  else{
      int count = num_ * channels_ * width_ * height_;
      Dtype* tmp = new Dtype[count];
      caffe_mul<Dtype>(count, message_passing_.cpu_diff(), spatial_out_blob_.cpu_data(), tmp);

      for (int c = 0; c < count; ++c) {
          (this->blobs_[0]->mutable_cpu_diff())[0] += tmp[c];
      }

      caffe_mul<Dtype>(count, message_passing_.cpu_diff(), bilateral_out_blob_.cpu_data(), tmp);
      for (int c = 0; c < count; ++c) {
          (this->blobs_[1]->mutable_cpu_diff())[0] += tmp[c];
      }

      delete[] tmp;

      // TODO: Check whether there's a way to improve the accuracy of this calculation. Unit tests for
      // Gradient calculation w.r.t. softmax inputs are failing when the kernel weights are large.
      caffe_cpu_scale<Dtype>(count, (this->blobs_[0]->cpu_data())[0],
                             message_passing_.cpu_diff(), spatial_out_blob_.mutable_cpu_diff());

      caffe_cpu_scale<Dtype>(count, (this->blobs_[1]->cpu_data())[0],
                             message_passing_.cpu_diff(), bilateral_out_blob_.mutable_cpu_diff());
  }

  //---------------------------- BP thru normalization --------------------------
  Dtype * spatial_out_diff = spatial_out_blob_.mutable_cpu_diff();
  for (int channel_id = 0; channel_id < channels_; ++channel_id) {
    caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
              spatial_out_diff + channel_id * num_pixels_,
              spatial_out_diff + channel_id * num_pixels_);
  }

  Dtype * bilateral_out_diff = bilateral_out_blob_.mutable_cpu_diff();
  for (int channel_id = 0; channel_id < channels_; ++channel_id) {
    caffe_mul(num_pixels_, bilateral_norms_->cpu_data(),
              bilateral_out_diff + channel_id * num_pixels_,
              bilateral_out_diff + channel_id * num_pixels_);
  }

  //--------------------------- Gradient for message passing ---------------
  spatial_lattice_->compute_cpu(prob_diff, spatial_out_blob_.cpu_diff(), channels_, true, true);
  bilateral_lattice_->compute_cpu(prob_diff, bilateral_out_blob_.cpu_diff(), channels_, true, true);

  //--------------------------------------------------------------------------------
  vector<bool> propagate_down(2, true);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::compute_detection_potential_diffs() {

  Dtype * const prob_diff = prob_.mutable_cpu_diff();

  Dtype * const y_input_diff = detection_y_q_input_->mutable_cpu_diff();
  Dtype * const x_weights_diff = blobs_[3]->mutable_cpu_diff();
  Dtype * const y_weights_diff = blobs_[4]->mutable_cpu_diff();

  caffe_set(detection_y_q_input_->count(), Dtype(0.), y_input_diff);
  caffe_set(blobs_[3]->count(), Dtype(0.), x_weights_diff);
  caffe_set(blobs_[4]->count(), Dtype(0.), y_weights_diff);
  // Prob diffs are already zero.

  tvg::DetectionUtils::backprop_exp_and_normalize_q_y(*detection_y_q_output_, detection_y_q_before_softmax_);

  const Dtype * const y_update_diff = detection_y_q_before_softmax_.cpu_diff();
  const Dtype * const x_update_diff = detection_potentials_for_x_.cpu_diff();

  const std::size_t detection_count = msmf_parent_->detection_list_.size();
  const Dtype * const x_weights_data = blobs_[3]->cpu_data();
  const Dtype * const y_weights_data = blobs_[4]->cpu_data();

  const Dtype * const prob_data = prob_.cpu_data();

  // Backprop the mean field updates for each detection.
  for (int det_id = 0; det_id < detection_count; ++det_id) {

      const shared_ptr<const tvg::Detection> &cur_detection = msmf_parent_->detection_list_[det_id];
      const vector<int> &foreground_pixels = cur_detection->get_foreground_pixels();

      const int det_label = cur_detection->get_label();

      const float current_detection_x_mult = cur_detection->get_x_potentials_mutiplier();
      const float x_multiplier = x_weights_data[det_label] * current_detection_x_mult;
      const float y_multiplier = y_weights_data[det_label] * cur_detection->get_y_potentials_multiplier();

      const float y_update_diff_0 = y_update_diff[det_id];
      const float y_update_diff_1 = y_update_diff[detection_count + det_id];

      const float detection_y_q_input_data_0 = detection_y_q_input_->cpu_data()[det_id];
      const float detection_y_q_input_data_1 = detection_y_q_input_->cpu_data()[detection_count + det_id];

      const float prob_diff_update = y_multiplier * (y_update_diff_0 - y_update_diff_1);

      const float cur_label_mult = cur_detection->get_y_potentials_multiplier() * y_update_diff_0;
      const float other_labels_mult = cur_detection->get_y_potentials_multiplier() * y_update_diff_1;

      const float cur_label_x_diff_mult = current_detection_x_mult * detection_y_q_input_data_0;
      const float other_labels_x_diff_mult = current_detection_x_mult * detection_y_q_input_data_1;

      const int num_foreground_pixels = foreground_pixels.size();
      for (int i = 0; i < num_foreground_pixels; ++i) {

        const int pixel_id = foreground_pixels[i];
        const Dtype x_q_at_det_label = prob_data[det_label * num_pixels_ + pixel_id];

        prob_diff[det_label * num_pixels_ + pixel_id] -= prob_diff_update;

        y_weights_diff[det_label] -= (
                cur_label_mult * x_q_at_det_label +
                other_labels_mult * (1 - x_q_at_det_label)
        );

        for (int label_id = 0; label_id < channels_; ++label_id) {
          if (label_id == det_label) {
            y_input_diff[det_id] -= x_multiplier * x_update_diff[label_id * num_pixels_ + pixel_id];

            x_weights_diff[det_label] -= (cur_label_x_diff_mult *
                                          x_update_diff[label_id * num_pixels_ + pixel_id]);

          } else {
            y_input_diff[detection_count + det_id] -= x_multiplier * x_update_diff[label_id * num_pixels_ + pixel_id];

            x_weights_diff[det_label] -= (other_labels_x_diff_mult *
                                          x_update_diff[label_id * num_pixels_ + pixel_id]);
          }
        }
      }
    } // end - for each detection.
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::InitLocalCompatibility(int n, int c, int h, int w) {

    if (!is_no_class_weights_){
        LOG(FATAL) << ("This should only be called when class specific weights are not being used");
    }

    local_compatibility_matrix_.Reshape(1,1,c,c);
    caffe_set(local_compatibility_matrix_.count(), Dtype(0.0), local_compatibility_matrix_.mutable_cpu_data());

    // Initialise to the Potts model
    for (int chan = 0; chan < c; ++chan){
        local_compatibility_matrix_.mutable_cpu_data()[chan + chan*c] = Dtype(-1.0);
    }

    this->channels_ = c;
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::Reshape(int n, int c, int h, int w) {
    pairwise_.Reshape(n,c,h,w);
    spatial_out_blob_.Reshape(n,c,h,w);
    bilateral_out_blob_.Reshape(n,c,h,w);
    message_passing_.Reshape(n,c,h,w);

    tmp.Reshape(spatial_out_blob_.shape());
    tmp_ones.Reshape(spatial_out_blob_.shape());

    this->num_ = n;
    this->channels_ = c;
    this->height_ = h;
    this->width_ = w;
    this->count_ = n * c * h * w;
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::set_iteration_number(int it){
    this->iteration_number_ = it;
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::print_blobs(std::string prefix){
    tvg::CommonUtils::save_blob_to_file(prob_, prefix + "prob_iter_" + std::to_string(iteration_number_) + ".csv");
    tvg::CommonUtils::save_blob_to_file(spatial_out_blob_, prefix + "spatial_out_iter_" + std::to_string(iteration_number_) + ".csv");
    tvg::CommonUtils::save_blob_to_file(bilateral_out_blob_, prefix + "bilateral_out_iter_" + std::to_string(iteration_number_) + ".csv");
    tvg::CommonUtils::save_blob_to_file(message_passing_, prefix + "message_passing_iter_" + std::to_string(iteration_number_) + ".csv");
    tvg::CommonUtils::save_blob_to_file(pairwise_, prefix + "pairwise_iter_" + std::to_string(iteration_number_) + ".csv");
    tvg::CommonUtils::save_blob_to_file(*sum_top_vec_[0], prefix + "output_iter_" + std::to_string(iteration_number_) + ".csv");
    if (this->is_no_class_weights_) {
        tvg::CommonUtils::save_blob_to_file(local_compatibility_matrix_,
                                            prefix + "local_compat_matrix_iter_" + std::to_string(iteration_number_) +
                                            ".csv");
    }
    tvg::CommonUtils::save_blob_to_file(*(this->blobs_[2]), prefix + "compat_matrix_iter_" + std::to_string(iteration_number_) + ".csv");
}

INSTANTIATE_CLASS(MeanfieldIteration);
}  // namespace caffe
