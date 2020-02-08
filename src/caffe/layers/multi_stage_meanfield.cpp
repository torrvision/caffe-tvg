#include <vector>
#include <boost/lexical_cast.hpp>
#include <caffe/util/tvg_ho_utils.hpp>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/meanfield_layers.hpp"
#include "caffe/util/tvg_common_utils.hpp"

namespace caffe {

/*
 * bottom[0] = Unary
 * bottom[1] = Unary
 * bottom[2] = RGB Image
 * bottom[3] = Indices. Used for loading detection and superpixel files
 * top[0]    = Output. Ie, the final Q distribution
 * top[1]    = Final values of latent Y variables
 */
template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  init_cpu_ = false;
  init_gpu_ = false;

  const caffe::MultiStageMeanfieldParameter meanfield_param = this->layer_param_.multi_stage_meanfield_param();

  CHECK_EQ(meanfield_param.has_spatial_filter_weight(), false) << "This parameter is not used anymore";
  CHECK_EQ(meanfield_param.has_bilateral_filter_weight(), false) << "This parameter is not used anymore";
  CHECK_EQ(meanfield_param.has_forced_spatial_filter_weight(), false) << "This parameter is not used anymore";
  CHECK_EQ(meanfield_param.has_forced_bilateral_filter_weight(), false) << "This parameter is not used anymore";

  num_iterations_ = meanfield_param.num_iterations();
  CHECK_GT(num_iterations_, 1) << "Number of iterations must be greater than 1.";

  theta_gamma_ = meanfield_param.theta_gamma();
  theta_alpha_ = meanfield_param.theta_alpha();
  theta_beta_ = meanfield_param.theta_beta();

  is_no_class_weights_ = meanfield_param.is_no_class_weights();

  // Detection potentials related stuff
  detection_potentials_enabled_ = meanfield_param.detection_potentials_enabled();
  detection_dat_dir_ = meanfield_param.detection_dat_input_dir();
  detection_dat_check_enabled_ = meanfield_param.detection_dat_check_enabled();
  train_dataset_size_ = meanfield_param.train_dataset_size();
  detection_potentials_config_.set_max_score(meanfield_param.detection_potentials_max_score());
  detection_potentials_config_.set_epsilon(meanfield_param.detection_potentials_epsilon());
  cur_train_image_id_ = 0;

  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  CHECK_EQ(num_, 1) << "This implementation currently only supports batch size 1";

  top[0]->Reshape(num_, channels_, height_, width_);
  top[1]->Reshape(1, 1, 1, 2 * 100); // TODO: Bit dodgy. Fix later. Can output up to 100 Y variables.

  ho_potentials_enabled_ = meanfield_param.ho_potentials_enabled();
  ho_num_layers_ = meanfield_param.has_ho_num_layers() ? meanfield_param.ho_num_layers() : 1;
  ho_dat_input_dir_ = meanfield_param.ho_dat_input_dir();

  // Initialize the parameters that will updated by backpropagation.
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Multimeanfield layer skipping parameter initialization.";
  } else {
    init_param_blobs(meanfield_param);
  }

  init_spatial_lattice();
  init_bilateral_buffers();

  // Configure the split layer that is used to make copies of the unary term. One copy for each iteration.
  // It may be possible to optimize this calculation later.
  split_layer_bottom_vec_.clear();
  split_layer_bottom_vec_.push_back(bottom[0]);

  split_layer_top_vec_.clear();

  split_layer_out_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    split_layer_out_blobs_[i].reset(new Blob<Dtype>());
    split_layer_top_vec_.push_back(split_layer_out_blobs_[i].get());
  }

  LayerParameter split_layer_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  split_layer_->SetUp(split_layer_bottom_vec_, split_layer_top_vec_);

  // Initialize super-pixel based potentials stuff (the first additional unaries).
  au_split_layer_bottom_vec_.clear();
  au_split_layer_top_vec_.clear();

  au_blobs_.resize(num_iterations_ + 1);
  au_blobs_[0].reset(new Blob<Dtype>(num_, channels_, height_, width_));
  au_split_layer_bottom_vec_.push_back(au_blobs_[0].get());
  for (int i = 1; i < num_iterations_ + 1; ++i) {
    au_blobs_[i].reset(new Blob<Dtype>());
    au_split_layer_top_vec_.push_back(au_blobs_[i].get());
  }
  LayerParameter au_sl_param;
  au_split_layer_.reset(new SplitLayer<Dtype>(au_sl_param));
  au_split_layer_->SetUp(au_split_layer_bottom_vec_, au_split_layer_top_vec_);

  unary_prob_.Reshape(num_, channels_, height_, width_);
  ho_softmax_bottom_vec_.clear();
  ho_softmax_top_vec_.clear();
  ho_softmax_bottom_vec_.push_back(bottom[1]);
  ho_softmax_top_vec_.push_back(&unary_prob_);

  LayerParameter ho_softmax_parameter;
  ho_softmax_.reset(new SoftmaxLayer<Dtype>(ho_softmax_parameter));
  ho_softmax_->SetUp(ho_softmax_bottom_vec_, ho_softmax_top_vec_);

  // Initialize detection potential related stuff.
  detection_y_unary_.reset(new Blob<Dtype>()); // not enough information to provide the blob size.
  detection_y_qs_.resize(num_iterations_ + 1);

  for (int i = 0; i < num_iterations_ + 1; ++i) {
    detection_y_qs_[i].reset(new Blob<Dtype>()); // not enough information to provide the blob size.
  }

  // Make blobs store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
  }

  // Make instances of MeanfieldIteration and initialize them.
  meanfield_iterations_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {

    if (is_no_class_weights_ && meanfield_iterations_[i] != nullptr && (meanfield_iterations_[0]->blobs()[1]->count() == this->blobs_[1]->count()) ){
      LOG(INFO) << ("Skipping initalisation of meanfield iterations");
      printf("Skipping initalisation of meanfield iterations on i = %d \n", i);
      //continue;
    }
    else {
      meanfield_iterations_[i].reset(new MeanfieldIteration<Dtype>(this));
    }
    meanfield_iterations_[i]->OneTimeSetUp(
        split_layer_out_blobs_[i].get(), // unary terms
        au_blobs_[i + 1].get(), // additional unary terms
        (i == 0) ? bottom[1] : iteration_output_blobs_[i - 1].get(), // softmax input
        (i == num_iterations_ - 1) ? top[0] : iteration_output_blobs_[i].get(), // output blob
        detection_y_qs_[i].get(), // y_input
        detection_y_qs_[i + 1].get(), // y_output
        spatial_lattice_, // spatial lattice
        &spatial_norm_); // spatial normalization factors.
  }
  meanfield_iterations_[0]->is_first_iteration_ = true; // TODO: a nasty hack. Fix later.

  this->param_propagate_down_.resize(this->blobs_.size(), true);

  LOG(INFO) << ("MultiStageMeanfieldLayer initialised.");

  if (is_no_class_weights_ && (detection_potentials_enabled_ || ho_potentials_enabled_)){
    LOG(FATAL) << ("Cannot have class weights disabled, and detection potentials enabled");
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

    if (is_no_class_weights_){
      for (int i = 0; i < num_iterations_ - 1; ++i) {
        iteration_output_blobs_[i]->Reshape(num_, channels_, height_, width_);
      }
      top[0]->Reshape(num_, channels_, height_, width_);
      unary_prob_.Reshape(num_, channels_, height_, width_);
    }
}

/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input (a copy of the unary terms)
 * bottom[2] - RGB images
 * bottom[3] - Image indices, useful for higher order potentials.
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if (detection_potentials_enabled_) {
    init_detections(static_cast<int>(bottom[3]->cpu_data()[0]));
  }

  split_layer_bottom_vec_[0] = bottom[0];
  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

  if (this->ho_potentials_enabled_) {
    init_ho_potentials(*bottom[0], static_cast<int>(bottom[3]->cpu_data()[0]));
    au_split_layer_->Forward(au_split_layer_bottom_vec_, au_split_layer_top_vec_);
  }

  // Initialize the bilateral lattice.
  compute_bilateral_kernel(bottom[2], 0, bilateral_kernel_buffer_); // only batch_size = 1 is supported
  bilateral_lattice_.reset(new ModifiedPermutohedral());
  bilateral_lattice_->init_cpu(bilateral_kernel_buffer_, 5, num_pixels_);

  // Calculate bilateral filter normalization factors.
  Dtype *norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(0);
  bilateral_lattice_->compute_cpu(norm_output_data, norm_feed_, 1);
  for (int i = 0; i < num_pixels_; ++i) {
    norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
  }

  for (int i = 0; i < num_iterations_; ++i) {

    meanfield_iterations_[i]->PrePass(this->blobs_, bilateral_lattice_, &bilateral_norms_);
    if (is_no_class_weights_){
      meanfield_iterations_[i]->InitLocalCompatibility(num_, channels_, height_, width_);
      meanfield_iterations_[i]->Reshape(num_, channels_, height_, width_);
    }

    meanfield_iterations_[i]->Forward_cpu();
  }
  
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
void MultiStageMeanfieldLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (ho_potentials_enabled_) {
    caffe_set(ho_num_layers_, Dtype(0.), this->blobs_[5]->mutable_cpu_diff());

    tvg::HOPotentialsUtils::init_to_have_same_size(ho_cliques_, ho_stat_potentials_diff_, channels_, 0);
  }

  if (detection_potentials_enabled_ && detection_count_ > 0) {
    caffe_set(detection_y_qs_[num_iterations_]->count(), Dtype(0.), detection_y_qs_[num_iterations_]->mutable_cpu_diff());
  }
  
  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    meanfield_iterations_[i]->Backward_cpu();
  }

  const vector<bool> split_layer_propagate_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_layer_propagate_down, split_layer_bottom_vec_);

  // BP superpixel based higher order potentials.
  au_split_layer_->Backward(au_split_layer_top_vec_, split_layer_propagate_down, au_split_layer_bottom_vec_);
  tvg::HOPotentialsUtils::bp_compute_additional_unaries(*au_blobs_[0], ho_cliques_, ho_stat_potentials_,
                                                        ho_stat_potentials_diff_, *(this->blobs_[5]));
  tvg::HOPotentialsUtils::bp_average_stats_potentials(*bottom[0], ho_cliques_, ho_stat_potentials_diff_);


  // Accumulate diffs from mean field iterations.
  for (int blob_id = 0; blob_id < this->blobs_.size() - 2; ++blob_id) { // Ignore y-weights and superpixel weights

    Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();

    if (this->param_propagate_down_[blob_id]) {

      caffe_set(cur_blob->count(), Dtype(0), cur_blob->mutable_cpu_diff());

      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add = meanfield_iterations_[i]->blobs()[blob_id]->cpu_diff();
        caffe_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_cpu_diff());
      }
    }
  }
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[5 * p] = static_cast<float>(p % width_) / theta_alpha_;
    output_kernel[5 * p + 1] = static_cast<float>(p / width_) / theta_alpha_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data() + rgb_blob->offset(n);
    output_kernel[5 * p + 2] = static_cast<float>(rgb_data_start[p] / theta_beta_);
    output_kernel[5 * p + 3] = static_cast<float>((rgb_data_start + num_pixels_)[p] / theta_beta_);
    output_kernel[5 * p + 4] = static_cast<float>((rgb_data_start + num_pixels_ * 2)[p] / theta_beta_);
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_spatial_kernel(float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[2*p] = static_cast<float>(p % width_) / theta_gamma_;
    output_kernel[2*p + 1] = static_cast<float>(p / width_) / theta_gamma_;
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_param_blobs(const MultiStageMeanfieldParameter &meanfield_param) {

  // blobs_[0] - spatial kernel weights
  // blobs_[1] - bilateral kernel weights
  // blobs_[2] - compatability matrix
  // blobs_[3] - detection x update weights
  // blobs_[4] - detection y update weights
  // blobs_[5] - super-pixel based HO weight param
  // blobs_[6] - an unused blob. retained for compatibility with older models.
  this->blobs_.resize(7);

  // Allocate space for kernel weights.
  this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
  this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));

  // Initialize the kernels weights.
  tvg::CommonUtils::read_into_the_diagonal(meanfield_param.spatial_filter_weights_str(), *(this->blobs_[0]), is_no_class_weights_);
  tvg::CommonUtils::read_into_the_diagonal(meanfield_param.bilateral_filter_weights_str(), *(this->blobs_[1]), is_no_class_weights_);

  // Initialize the compatibility matrix.
  this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
  caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[2]->mutable_cpu_data());

  // Initialize it to have the Potts model.
  for (int c = 0; c < channels_; ++c) {
    (this->blobs_[2]->mutable_cpu_data())[c * channels_ + c] = Dtype(-1.);
  }

  // Initialize detection weight parameters
  this->blobs_[3].reset(new Blob<Dtype>(1, 1, 1, channels_)); // detection x weights
  this->blobs_[4].reset(new Blob<Dtype>(1, 1, 1, channels_)); // detection y weights

  if (detection_potentials_enabled_) {
    tvg::CommonUtils::read_into_array(channels_, meanfield_param.detection_x_weights_str(),
                                      this->blobs_[3]->mutable_cpu_data());

    tvg::CommonUtils::read_into_array(channels_, meanfield_param.detection_y_weights_str(),
                                      this->blobs_[4]->mutable_cpu_data());
  } else {
    caffe_set(this->blobs_[3]->count(), Dtype(0.), this->blobs_[3]->mutable_cpu_data());
    caffe_set(this->blobs_[3]->count(), Dtype(0.), this->blobs_[3]->mutable_cpu_diff());

    caffe_set(this->blobs_[4]->count(), Dtype(0.), this->blobs_[4]->mutable_cpu_data());
    caffe_set(this->blobs_[4]->count(), Dtype(0.), this->blobs_[3]->mutable_cpu_diff());
  }

  this->blobs_[5].reset(new Blob<Dtype>(1, 1, 1, ho_num_layers_)); // ho_w_param
  this->blobs_[6].reset(new Blob<Dtype>(1, 1, 1, ho_num_layers_)); // legacy blob
  if (ho_potentials_enabled_) {
    tvg::CommonUtils::read_into_array(ho_num_layers_, meanfield_param.ho_w_param_str() , this->blobs_[5]->mutable_cpu_data());
  } else {
    caffe_set(this->blobs_[5]->count(), Dtype(0.), this->blobs_[5]->mutable_cpu_data());
    caffe_set(this->blobs_[5]->count(), Dtype(0.), this->blobs_[5]->mutable_cpu_diff());
  }
  caffe_set(this->blobs_[6]->count(), Dtype(0.), this->blobs_[6]->mutable_cpu_data());
  caffe_set(this->blobs_[6]->count(), Dtype(0.), this->blobs_[6]->mutable_cpu_diff());
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_detections(const int image_id) {

  if (detection_dat_check_enabled_) { // Safety check.
    CHECK_EQ(cur_train_image_id_, image_id) << "Error while intializing detection potentials!";
    if (cur_train_image_id_ != image_id) {
      exit(1);
    }

    if (++cur_train_image_id_ == train_dataset_size_) {
      cur_train_image_id_ = 0;
    }
  }

  detection_list_.clear();
  tvg::DetectionUtils::read_detections_from_file(detection_list_, detection_dat_dir_ + "/" +
          boost::lexical_cast<std::string>(image_id) + ".detections");
  detection_count_ = detection_list_.size();

  if (detection_count_ > 0) {
    detection_y_unary_->Reshape(1, 2, detection_count_, 1);
    for (int i = 0; i < num_iterations_ + 1; ++i) {
      detection_y_qs_[i]->Reshape(1, 2, detection_count_, 1);
    }

    init_detection_y_unaries();

    // Initialize the Q_y input to the first meanfield iteration
    caffe_copy(detection_y_unary_->count(), detection_y_unary_->cpu_data(), detection_y_qs_[0]->mutable_cpu_data());
  }
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_detection_y_unaries() {

  Dtype * const detection_y_unary_data = detection_y_unary_->mutable_cpu_data();
  const size_t detection_count = detection_list_.size();

  // Initilize unaries for Y_det latent variables.
  for (int det_id = 0; det_id < detection_count; ++det_id) {

    float positive_prob = detection_list_[det_id]->get_score() / detection_potentials_config_.get_max_score();
    const float max_positive_prob = 1 - detection_potentials_config_.get_epsilon();

    if (positive_prob > max_positive_prob) { // Prevent saturation and numerical problems.
      positive_prob = max_positive_prob;
    }
    
    detection_y_unary_data[det_id] = (1 - positive_prob); // Prob of Y_d = 0;
    detection_y_unary_data[detection_count + det_id] = positive_prob; // Prob of Y_d = 1
  }
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_ho_potentials(const Blob<Dtype> & unaries, const int image_id) {

  tvg::HOPotentialsUtils::load_superpixels(ho_cliques_,  ho_dat_input_dir_ + "/" +
          boost::lexical_cast<std::string>(image_id) + ".superpixels");

  tvg::HOPotentialsUtils::init_to_have_same_size(ho_cliques_, ho_stat_potentials_, channels_, 0);
  tvg::HOPotentialsUtils::average_stats_potentials(unaries, ho_cliques_, ho_stat_potentials_);
  tvg::HOPotentialsUtils::compute_sp_additional_unaries(*au_blobs_[0], ho_cliques_, ho_stat_potentials_,
                                                        this->blobs_[5]->cpu_data());
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_spatial_lattice(void) {

  // This should be done on GPU if the GPU is available.
  // Right now, the spatial kernel is computed on CPU, then transferred over to the GPU
  float * spatial_kernel = new float[2 * num_pixels_];
  compute_spatial_kernel(spatial_kernel);
  bool force_cpu = this->layer_param().multi_stage_meanfield_param().force_cpu();
  spatial_lattice_.reset(new ModifiedPermutohedral());
  spatial_norm_.Reshape(1, 1, height_, width_);

  if ( Caffe::mode() == Caffe::CPU || force_cpu) {

    spatial_lattice_->init_cpu(spatial_kernel, 2, num_pixels_);

    // Compute normalisation factors
    norm_feed_ = new Dtype[num_pixels_];
    caffe_set(num_pixels_, Dtype(1.0), norm_feed_);
    Dtype* norm_data = spatial_norm_.mutable_cpu_data();
    spatial_lattice_->compute_cpu(norm_data, norm_feed_, 1);

    delete[] spatial_kernel;
    init_cpu_ = true;

  } else if( Caffe::mode() == Caffe::GPU){

    #ifndef CPU_ONLY
    float* spatial_kernel_gpu;
    Dtype* norm_data_gpu;

    CUDA_CHECK( cudaMalloc( (void**)&spatial_kernel_gpu, 2*num_pixels_ * sizeof(float) ) );
    CUDA_CHECK( cudaMemcpy(spatial_kernel_gpu, spatial_kernel, 2*num_pixels_ * sizeof(float), cudaMemcpyHostToDevice ) );
    spatial_lattice_->init_gpu(spatial_kernel_gpu, 2, width_, height_);
    CUDA_CHECK( cudaFree(spatial_kernel_gpu) );

    CUDA_CHECK( cudaMalloc( (void**)&norm_feed_, num_pixels_ * sizeof(float)) );
    caffe_gpu_set(num_pixels_, Dtype(1.0), norm_feed_);
    norm_data_gpu = spatial_norm_.mutable_gpu_data();
    spatial_lattice_->compute_gpu(norm_data_gpu, norm_feed_, 1);

    init_gpu_ = true;
    #endif
  }
  else{
    LOG(FATAL) << "Unknown Caffe mode. Neither CPU nor GPU";
  }

  Dtype* norm_data = spatial_norm_.mutable_cpu_data(); // This value has been computed either on the GPU or CPU. May be more efficient to just do everything on CPU.
  for (int i = 0; i < num_pixels_; ++i) {
    norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
  }
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_bilateral_buffers(void) {

  if (init_cpu_) {
    bilateral_kernel_buffer_ = new float[5 * num_pixels_];
  }
  else if (init_gpu_){
    #ifndef CPU_ONLY
    CUDA_CHECK( cudaMalloc( (void**)&bilateral_kernel_buffer_, 5 * num_pixels_ * sizeof(float) ) );
    #endif
  }
  else{
    LOG(FATAL) << "Should not have been able to get here";
  }
  bilateral_norms_.Reshape(num_, 1, height_, width_);

}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::print_blob_sizes(void) {

  printf("MSMF Check layer: Count of blob[0]: %d\n", this->blobs_[0]->count());
  printf("MSMF Check layer: Count of meanfield_iterations->blob[0]: %d\n", meanfield_iterations_[0]->blobs()[0]->count());

}

template<typename Dtype>
MultiStageMeanfieldLayer<Dtype>::~MultiStageMeanfieldLayer(){
  if(init_cpu_){
    delete[] bilateral_kernel_buffer_;
    delete[] norm_feed_;
  }
#ifndef CPU_ONLY
  if(init_gpu_){
    CUDA_CHECK(cudaFree(bilateral_kernel_buffer_));
    CUDA_CHECK(cudaFree(norm_feed_));
  }
#endif
}


INSTANTIATE_CLASS(MultiStageMeanfieldLayer);
REGISTER_LAYER_CLASS(MultiStageMeanfield);
}  // namespace caffe
