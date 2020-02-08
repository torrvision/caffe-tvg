#include "caffe/layer.hpp"
#include "caffe/layers/shape_term.hpp"
#include "caffe/layers/interp_variable_layer.hpp"
#include "caffe/util/tvg_common_utils.hpp"
#include <boost/lexical_cast.hpp>

namespace caffe {

  /*
  * bottom[0] = Unary
  * bottom[1] = Indices for loading detection files (same as meanfield layer)
  * top[0]    = Output. Ie, the unary shape term
  *
  * This function is called once, and is basically the "Constructor"
  */
  template<typename Dtype>
  void ShapeTermLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

    this->detection_boxes_input_dir_ = this->layer_param().shape_term_param().detection_box_input_dir();

    if (this->blobs_.size() > 0){
      LOG(INFO) << "Skipping parameter initialization";
    }
    else if (this->layer_param_.shape_term_param().init_from_files()){
      std::string prefix = this->layer_param_.shape_term_param().init_prefix();
      int num_files = this->layer_param_.shape_term_param().num_files();

      this->blobs_.resize(num_files);

      for (int i = 0; i < num_files; ++i){
        const std::string filename = prefix + boost::lexical_cast<std::string>(i) + ".dat";
        this->blobs_[i].reset( new Blob<Dtype>() ); // blob will be reshaped when data is read into it
        tvg::CommonUtils::read_and_reshape_from_data( *(this->blobs_[i]), filename);
      }
    }
    else{
      // Initialise parameters
      this->blobs_.resize(5);
    }

    warped_shapes_.resize(this->blobs_.size());
    for (int i = 0; i < this->blobs_.size(); ++i){
      warped_shapes_[i] = new Blob<Dtype>();
    }

    LayerParameter transfer_param;
    transfer_param.mutable_transfer_param()->set_no_detections(true);
    transfer_param.mutable_transfer_param()->set_copy_bg(false);
    transfer_unary_.reset( new TransferLayer<Dtype>(transfer_param) );

    LayerParameter dummy_param;
    roi_pooling_layer_.reset( new ROIPoolingLayer<Dtype>(dummy_param) );

    LayerParameter eltwise_param;
    eltwise_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
    eltwise_layer_.reset( new EltwiseLayer<Dtype>(eltwise_param) );

    is_initialised_ = false;

  }

  /*
   * This function is called before every call of "Forward"
   */
  template<typename Dtype>
  void ShapeTermLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
    detection_box_list_.clear();

    const int image_id = static_cast<int>(bottom[1]->cpu_data()[0]);
    tvg::DetectionUtils::read_detections_from_file(detection_box_list_, detection_boxes_input_dir_ + "/" +
                                                   boost::lexical_cast<std::string>(image_id) + ".detections"); //TODO: Specify extension, instead of hardcoding ".detections"

    top_channels_ = (int) detection_box_list_.size() + 1;
    top_height_ = bottom[0]->height();
    top_width_ = bottom[0]->width();
    top[0]->Reshape(bottom[0]->num(), top_channels_, top_height_, top_width_);
    caffe_set(top[0]->count(), Dtype(0.), top[0]->mutable_cpu_data());

    /* Just calling clear() on a vector of pointers has a memory leak.*/
    for (int i = 0; i < class_unary_rois_.size() ; ++i){
        delete (class_unary_rois_[i]);
        delete (matched_shapes_[i]);
        delete (shape_terms_orig_[i]);
        delete (shape_terms_padded_[i]);
        delete (roi_pooling_switches_[i]);
    }

    class_unary_rois_.clear();
    matched_shapes_.clear();
    shape_terms_orig_.clear();
    shape_terms_padded_.clear();
    roi_pooling_switches_.clear();

    //class_unary_rois_.resize(top_channels_ - 1); //Blob does not allow you to do this
    for (int i = 0; i < top_channels_ - 1; ++i){
      class_unary_rois_.push_back( new Blob<Dtype>() );
      matched_shapes_.push_back( new Blob<Dtype>() );
      shape_terms_orig_.push_back( new Blob<Dtype>() );
      shape_terms_padded_.push_back( new Blob<Dtype>() );
      roi_pooling_switches_.push_back( new Blob<int>() );
    }

    if (top[0]->num() > 1) {
      LOG(FATAL) << "Only a batch size of 1 is currently supported" << std::endl;
    }

    shape_a_indices.clear();
    shape_b_indices.clear();
  }

  /*
   * Only a batch size of 1 is currently supported.
   * bottom[0] = segmentation unaries
   * bottom[1] = detection index (like meanfield layer)
   * top[0] = the output
   *
   * This layer consists of a number of steps
   * 1) "Transfer" the unary, from K channels, to just the channel of interest
   * 2) Do "ROI pooling" and extract the area corresponding to just the detection
   * 3) Interpolate the shape masks (parameters of this layer) to the size of the detection area
   * 4) Find the shape mask which "matches" the best, according to the normalised cross correlation
   * 5) Do an elementwise product between the shape mask and the unary. This is the shape prior
   * 6) Then "ROI Unpool" this back to the original size of the unary, zero-padding as necessary
   * 7) Steps 1 to 6 are done in a loop for all the detections
   * 8) Finally, concatenate all the shape terms together (remembering about the background)
   *
   */
  template<typename Dtype>
  void ShapeTermLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {

    InterpVariableLayer<Dtype> interpLayer(this->layer_param_);
    bool interp_initialised = false;
    std::vector< Blob<Dtype>* > interpBottom;
    std::vector< Blob<Dtype>* > interpTop;

    for (int i = 0; i < detection_box_list_.size(); ++i){

      // ------------- Some preliminarties, reading and parsing the detections ----------------------------

      const std::vector<int> &det_box = detection_box_list_[i]->get_foreground_pixels();
      int x_start = det_box[0];
      int y_start = det_box[1];
      int x_end = det_box[2];
      int y_end = det_box[3];

      x_start = std::max(0, x_start); y_start = std::max(0, y_start);
      x_end = std::min(bottom[0]->width() - 1, x_end); y_end = std::min(bottom[0]->height() - 1, y_end);

      // ------------ 1) "Transfer" the unary, from 21 channels, to just the channel of interest -----------
      Blob<Dtype> class_selector (1,1,1,1);
      class_selector.mutable_cpu_data()[0] = Dtype (detection_box_list_[i]->get_label());

      transfer_unary_bottom_.clear();
      transfer_unary_top_.clear();
      transfer_unary_bottom_.push_back(bottom[0]);
      transfer_unary_bottom_.push_back(&class_selector);
      transfer_unary_top_.push_back(&class_unary_blob_);

      if (!is_initialised_) {
        transfer_unary_->SetUp(transfer_unary_bottom_, transfer_unary_top_);
      }
      transfer_unary_->Reshape(transfer_unary_bottom_, transfer_unary_top_);
      transfer_unary_->Forward(transfer_unary_bottom_, transfer_unary_top_);

      // ----------- 2) Do "ROI pooling" and extract the area corresponding to just the detection -----------
      roi_pool_bottom_.clear();
      roi_pool_bottom_.push_back(&class_unary_blob_);
      roi_pool_top_.clear();

      Blob<Dtype> roi_pool_coords(1, 5, 1, 1);
      roi_pool_coords.mutable_cpu_data()[0] = 0; // the "num" index of the ROI. We assume a batch size of 1
      roi_pool_coords.mutable_cpu_data()[1] = x_start; roi_pool_coords.mutable_cpu_data()[2] = y_start; // x_1, y_1, including coords
      roi_pool_coords.mutable_cpu_data()[3] = x_end - 1; roi_pool_coords.mutable_cpu_data()[4] = y_end - 1; // x_2, y_2, including coords
      roi_pool_bottom_.push_back(&roi_pool_coords);
      roi_pool_top_.push_back(class_unary_rois_[i]);

      roi_pooling_layer_->set_pool_height_width(y_end - y_start, x_end - x_start, Dtype(1));
      roi_pooling_layer_->Reshape(roi_pool_bottom_, roi_pool_top_);
      roi_pooling_layer_->Forward(roi_pool_bottom_, roi_pool_top_);

      // extract the pooling switches, these are needed in the backward pass
      roi_pooling_layer_->transfer_pooling_switches(this->roi_pooling_switches_[i]);

        // ---------- 3) Interpolate the shape masks (parameters of this layer) to the size of the detection area ------

      int width = x_end - x_start; int height = y_end - y_start;

      Blob<Dtype> interpSize(1,1,1,2);
      interpSize.mutable_cpu_data()[0] = Dtype(width);
      interpSize.mutable_cpu_data()[1] = Dtype(height);

      // Warp the shape parameters onto the size of the image
      for (int j = 0; j < this->blobs_.size(); ++j){
        interpBottom.clear();
        interpTop.clear();

        interpBottom.push_back(this->blobs_[j].get());
        interpBottom.push_back(&interpSize);
        interpTop.push_back(warped_shapes_[j]);

        if (!interp_initialised) { interpLayer.SetUp(interpBottom, interpTop); interp_initialised = true;}
        interpLayer.Reshape(interpBottom, interpTop);
        interpLayer.Forward(interpBottom, interpTop);
      }

      // ---------- 4) Find the shape mask which "matches" the best, according to the normalised cross correlation ----

      // Now we need to find out the best shape prior
      int a_index = -1;
      int b_index = -1;
      Dtype max_score = 0;

      for (int j = 0; j < warped_shapes_.size(); ++j){

        Blob<Dtype> ones (1, 1, warped_shapes_[j]->height(), warped_shapes_[j]->width());
        const int N = warped_shapes_[j]->height() * warped_shapes_[j]->width();
        Blob<Dtype> temp ( ones.shape() );
        const Dtype* shapes = warped_shapes_[j]->cpu_data();
        caffe_set(N, Dtype(1), ones.mutable_cpu_data());

        for (int k = 0; k < warped_shapes_[j]->channels(); ++k){

          caffe_mul(N, shapes + N*k, class_unary_rois_[i]->cpu_data(), temp.mutable_cpu_data());
          const Dtype numerator = caffe_cpu_dot(N, ones.cpu_data(), temp.cpu_data());

          // For the denominator, we need to work out the norm
          caffe_mul(N, shapes + N*k, shapes + N*k, temp.mutable_cpu_data());
          const Dtype sum_sq_norm = caffe_cpu_dot(N, ones.cpu_data(), temp.cpu_data());
          const Dtype norm = sqrt(sum_sq_norm);

          const Dtype temp_score = numerator / norm;

          if (temp_score > max_score){
            max_score = temp_score;
            a_index = j;
            b_index = k;
          }
        }
      }

      matched_shapes_[i]->Reshape(1, 1, warped_shapes_[a_index]->height(), warped_shapes_[a_index]->width());
      CHECK_EQ(matched_shapes_[i]->height(), class_unary_rois_[i]->height());
      CHECK_EQ(matched_shapes_[i]->width(), class_unary_rois_[i]->width());
      const Dtype* matched_shape_ptr = warped_shapes_[a_index]->cpu_data();
      const int N = warped_shapes_[a_index]->height() * warped_shapes_[a_index]->width();
      matched_shape_ptr = matched_shape_ptr + b_index * N;
      caffe_cpu_copy(N, matched_shape_ptr, matched_shapes_[i]->mutable_cpu_data());

      shape_a_indices.push_back(a_index);
      shape_b_indices.push_back(b_index);

      // ----- 5) Do an elementwise product between the shape mask and the unary. This is the shape prior --------------

      elementwise_bottom_.clear();
      elementwise_bottom_.push_back( class_unary_rois_[i] );
      elementwise_bottom_.push_back( matched_shapes_[i] );
      elementwise_top_.clear();
      elementwise_top_.push_back( shape_terms_orig_[i] );

      if (!is_initialised_){ eltwise_layer_->SetUp(elementwise_bottom_, elementwise_top_);}
      eltwise_layer_->Reshape(elementwise_bottom_, elementwise_top_);
      eltwise_layer_->Forward(elementwise_bottom_, elementwise_top_);

      // ------ 6) Then "ROI Unpool" this back to the original size of the unary, zero-padding as necessary ------------

      roi_unpool_bottom_.clear();
      roi_unpool_top_.clear();
      roi_unpool_bottom_.push_back( shape_terms_orig_[i]);
      Blob<Dtype> unpooling_coords (1, 1, 1, 4);
      unpooling_coords.mutable_cpu_data()[0] = bottom[0]->width();
      unpooling_coords.mutable_cpu_data()[1] = bottom[0]->height();
      unpooling_coords.mutable_cpu_data()[2] = x_start;
      unpooling_coords.mutable_cpu_data()[3] = y_start;
      roi_unpool_bottom_.push_back( &unpooling_coords);
      roi_unpool_top_.push_back( shape_terms_padded_[i] );

      if (!is_initialised_){roi_unpool_layer_.SetUp(roi_unpool_bottom_, roi_unpool_top_);}
      roi_unpool_layer_.Reshape(roi_unpool_bottom_, roi_unpool_top_);
      roi_unpool_layer_.Forward(roi_unpool_bottom_, roi_unpool_top_);
    }

    // -------- 8) Finally, concatenate all the shape terms together (remembering about the background) --------------
    // Now just concatenate the outputs, and we are done. Remember about the zero!
    zero_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
    caffe_set(zero_.count(), Dtype(0), zero_.mutable_cpu_data());

    concat_bottom_.clear();
    concat_bottom_.push_back( &zero_);
    for (int i = 0; i < detection_box_list_.size(); ++i){
      concat_bottom_.push_back( shape_terms_padded_[i] );
    }

    if (!is_initialised_){concat_layer_.SetUp(concat_bottom_, top);}
    concat_layer_.Reshape(concat_bottom_, top);
    concat_layer_.Forward(concat_bottom_, top);

    if (detection_box_list_.size() > 0) { is_initialised_ = true; }

  } // Forward_cpu

  /*
   * top[0] = instance unary
   * bottom[0] = segmentation unary
   * bottom[1] = index for reading detections
   */
  template<typename Dtype>
  void ShapeTermLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype> *> &bottom) {
    // Initialise all the diffs to 0
    Blob<Dtype>* unary_blob = bottom[0];
    caffe_set(unary_blob->count(), Dtype(0), unary_blob->mutable_cpu_diff());

    for (int i = 0; i < this->blobs_.size(); ++i){
      caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_diff());
    }

    std::vector< Blob<Dtype>* > interpBottom;
    std::vector< Blob<Dtype>* > interpTop;
    InterpVariableLayer<Dtype> interpLayer(this->layer_param_);
    bool interp_initialised = false;
    Blob<Dtype> shape_param_bottom_blob;

    // --------1) Backprop through concatenate -----------
    std::vector<bool> prop_down(concat_bottom_.size(), true);
    concat_layer_.Backward(top, prop_down, concat_bottom_);

    for (int i = 0; i < detection_box_list_.size(); ++i){

      // ------------- Some preliminarties, reading and parsing the detections ----------------------------

      const std::vector<int> &det_box = detection_box_list_[i]->get_foreground_pixels();
      int x_start = det_box[0];
      int y_start = det_box[1];
      int x_end = det_box[2];
      int y_end = det_box[3];

      x_start = std::max(0, x_start); y_start = std::max(0, y_start);
      x_end = std::min(bottom[0]->width() - 1, x_end); y_end = std::min(bottom[0]->height() - 1, y_end);

      // Backprop through the ROI Unpooling
      roi_unpool_top_.clear();
      roi_unpool_top_.push_back( shape_terms_padded_[i] );

      roi_unpool_bottom_.clear();
      roi_unpool_bottom_.push_back( shape_terms_orig_[i]);
      Blob<Dtype> unpooling_coords (1, 1, 1, 4);
      unpooling_coords.mutable_cpu_data()[0] = bottom[0]->width();
      unpooling_coords.mutable_cpu_data()[1] = bottom[0]->height();
      unpooling_coords.mutable_cpu_data()[2] = x_start;
      unpooling_coords.mutable_cpu_data()[3] = y_start;
      roi_unpool_bottom_.push_back( &unpooling_coords);

      roi_unpool_layer_.Backward(roi_unpool_top_, prop_down, roi_unpool_bottom_); // This layer does not care what propagate_down is

      // Backprop through the elementwise layer Unpooling
      prop_down.clear();
      prop_down.resize(elementwise_bottom_.size(), true);

      elementwise_bottom_.clear();
      elementwise_bottom_.push_back( class_unary_rois_[i] );
      elementwise_bottom_.push_back( matched_shapes_[i] );
      elementwise_top_.clear();
      elementwise_top_.push_back( shape_terms_orig_[i] );

      eltwise_layer_->Backward(elementwise_top_, prop_down, elementwise_bottom_);

      // Now we have to backprop to two paths - the input, and also the parameters

      // First, lets backpropogate through to the input
      // Backprop through ROI Pooling
      roi_pool_bottom_.clear();
      roi_pool_bottom_.push_back(&class_unary_blob_);
      roi_pool_top_.clear();

      Blob<Dtype> roi_pool_coords(1, 5, 1, 1);
      roi_pool_coords.mutable_cpu_data()[0] = 0; // the "num" index of the ROI. We assume a batch size of 1
      roi_pool_coords.mutable_cpu_data()[1] = x_start; roi_pool_coords.mutable_cpu_data()[2] = y_start; // x_1, y_1, including coords
      roi_pool_coords.mutable_cpu_data()[3] = x_end - 1; roi_pool_coords.mutable_cpu_data()[4] = y_end - 1; // x_2, y_2, including coords
      roi_pool_bottom_.push_back(&roi_pool_coords);
      roi_pool_top_.push_back(class_unary_rois_[i]);

      roi_pooling_layer_->set_pool_height_width(y_end - y_start, x_end - x_start, Dtype(1));
      prop_down.resize(2); prop_down[0] = true; prop_down[1] = false;
      CHECK_EQ(x_end - x_start, class_unary_rois_[i]->width());
      CHECK_EQ(y_end - y_start, class_unary_rois_[i]->height());

      roi_pooling_layer_->Reshape(roi_pool_bottom_, roi_pool_top_);
      roi_pooling_layer_->set_pooling_switches(this->roi_pooling_switches_[i]);
      roi_pooling_layer_->Backward(roi_pool_top_, prop_down, roi_pool_bottom_); // Fine to have "(true, true)" in propogate down

      // Now, backprop through the transfer layer
      Blob<Dtype> class_selector (1,1,1,1);
      class_selector.mutable_cpu_data()[0] = Dtype (detection_box_list_[i]->get_label());

      transfer_unary_bottom_.clear();
      transfer_unary_top_.clear();
      transfer_unary_bottom_.push_back(bottom[0]);
      transfer_unary_bottom_.push_back(&class_selector);
      transfer_unary_top_.push_back(&class_unary_blob_);
      prop_down.resize(1);
      if (i == 0){ prop_down[0] = true;} else{ prop_down[0] = false; } // "false" tells the layer to accumulate gradients

      transfer_unary_->Backward(transfer_unary_top_, prop_down, transfer_unary_bottom_);

      //
      // Now, we have to backpropagate through to the parameters (ie shape templates) as well
      //
      int shape_a_index = shape_a_indices[i];
      shape_param_bottom_blob.Reshape(1, 1, this->blobs_[shape_a_index]->height(), this->blobs_[shape_a_index]->width());

      const int width = x_end - x_start; const int height = y_end - y_start;

      Blob<Dtype> interpSize(1,1,1,2);
      interpSize.mutable_cpu_data()[0] = Dtype(width);
      interpSize.mutable_cpu_data()[1] = Dtype(height);

      CHECK_EQ(height, matched_shapes_[i]->height());
      CHECK_EQ(width, matched_shapes_[i]->width());

      interpBottom.clear();
      interpTop.clear();
      prop_down[0] = true;

      interpBottom.push_back(&shape_param_bottom_blob);
      interpBottom.push_back(&interpSize);
      interpTop.push_back(matched_shapes_[i]);

      if (!interp_initialised) { interpLayer.SetUp(interpBottom, interpTop); interp_initialised = true;}
      interpLayer.ReInit(interpBottom, interpTop);
      interpLayer.Backward(interpTop, prop_down, interpBottom);

      // Now we need to add to the matched shape prior
      int shape_b_index = shape_b_indices[i];
      const int N = this->blobs_[shape_a_index]->height() * this->blobs_[shape_a_index]->width();
      Dtype* shape_grad_ptr = this->blobs_[shape_a_index]->mutable_cpu_diff();
      shape_grad_ptr = shape_grad_ptr + shape_b_index * N;

      caffe_add(N, shape_param_bottom_blob.cpu_diff(), shape_grad_ptr, shape_grad_ptr);

    }


  } // Backward_cpu

  INSTANTIATE_CLASS(ShapeTermLayer);

  REGISTER_LAYER_CLASS(ShapeTerm);
}
