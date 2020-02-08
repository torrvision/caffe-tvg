#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_tvg_util.hpp"


using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SoftmaxWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  // First, compute the loss with all labels
  scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
      new SoftmaxWithLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new SoftmaxWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);

  // Test when the ignore label is greater than the number of channels
  const int ignore_label = 121003;
  layer_param.mutable_loss_param()->set_ignore_label(ignore_label);
  this->blob_bottom_label_->mutable_cpu_data()[this->blob_bottom_label_->count() - 1] = ignore_label;
  layer.reset(new SoftmaxWithLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype computed_loss = this->blob_top_loss_->cpu_data()[0];
  // EXPECT_NEAR(full_loss, computed_loss, 1e-4);

  vector<bool> propagate_down(2, true); propagate_down[1] = false; // Only backpropogate to the data, but not the labels
  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);

  const int ignore_label = 123123;
  layer_param.mutable_loss_param()->set_ignore_label(ignore_label);
  this->blob_bottom_label_->mutable_cpu_data()[this->blob_bottom_label_->count() - 1] = ignore_label;
  SoftmaxWithLossLayer<Dtype> layer_two(layer_param);
  checker.CheckGradientExhaustive(&layer_two, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientIgnoreProb) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_loss_param()->set_has_probability_thresh(true);
    layer_param.mutable_loss_param()->set_probability_threshold(0.8);

    SoftmaxWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestProbIgnoreForward){

    typedef typename TypeParam::Dtype Dtype;
    // Load the big data
    Blob<Dtype> bottom_data (1, 5, 1, 2);
    Blob<Dtype> bottom_label(1, 1, 1, 2);
    Blob<Dtype> top_result;

    Dtype* bottom_data_cpu = bottom_data.mutable_cpu_data();
    bottom_data_cpu[0] = 1;
    bottom_data_cpu[2] = 2;
    bottom_data_cpu[4] = 3;
    bottom_data_cpu[6] = 4;
    bottom_data_cpu[8] = 6;
    for (int k = 1; k < bottom_data.count(); k = k + 2){
        bottom_data_cpu[k] = bottom_data_cpu[k - 1];
    }

    Dtype* bottom_label_cpu = bottom_label.mutable_cpu_data();
    bottom_label_cpu[0] = 4;
    bottom_label_cpu[1] = 3;

    // Set up the layer
    vector<Blob<Dtype>*> layer_bottom_vec;
    vector<Blob<Dtype>*> layer_top_vec;
    layer_bottom_vec.push_back(&bottom_data);
    layer_bottom_vec.push_back(&bottom_label);
    layer_top_vec.push_back(&top_result);

    LayerParameter layer_param;
    layer_param.mutable_loss_param()->set_normalize(false);
    layer_param.mutable_loss_param()->set_has_probability_thresh(true);
    layer_param.mutable_loss_param()->set_probability_threshold(0.8);

    scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer( new SoftmaxWithLossLayer<Dtype>(layer_param) );
    layer->SetUp(layer_bottom_vec, layer_top_vec);
    layer->Forward(layer_bottom_vec, layer_top_vec);
    Dtype loss = top_result.cpu_data()[0];

    LOG(INFO) << "The loss is " << loss << "\n";

    tvg::TestUtils::PrintBlob(top_result, false, "Output of softmax loss layer");
    tvg::TestUtils::PrintBlob(bottom_data, false, "Input data");
    tvg::TestUtils::PrintBlob(bottom_label, false, "Labels");

    vector<bool> propagate_down(2, true); propagate_down[1] = false; // Only backpropogate to the data, but not the labels
    layer->Backward(layer_top_vec, propagate_down, layer_bottom_vec);
    tvg::TestUtils::PrintBlob(bottom_data, true, "Gradient of loss wrt input data");

    GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
    checker.CheckGradientExhaustive(layer.get(), layer_bottom_vec,
                                    layer_top_vec, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardOHEM) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // set the OHEM number to half of the number of elements in the label blob
  int ohem_n_ = this->blob_bottom_label_->count() / 2;
  layer_param.mutable_loss_param()->set_use_ohem(true);
  layer_param.mutable_loss_param()->set_ohem_n(ohem_n_);
  layer_param.mutable_loss_param()->set_normalize(false);
  // Do Softmax forward
  Blob<Dtype>* softmax_out (new Blob<Dtype>());
  vector<Blob<Dtype>*> softmax_out_vec;
  softmax_out_vec.push_back(softmax_out);

  vector<Blob<Dtype>*> sm_blob_bottom_vec_;
  sm_blob_bottom_vec_.push_back(this->blob_bottom_data_);

  SoftmaxLayer<Dtype> smlayer(layer_param);
  smlayer.SetUp(sm_blob_bottom_vec_, softmax_out_vec);
  smlayer.Forward(sm_blob_bottom_vec_, softmax_out_vec);
  // get the outer_num_ and inner_num_ 
  // for usual scenarios, outer_num_  = N, inner_num_ = H*W
  int softmax_axis_ = this->blob_bottom_data_->CanonicalAxisIndex(layer_param.softmax_param().axis());
  int outer_num_ = this->blob_bottom_data_->count(0, softmax_axis_);
  int inner_num_ = this->blob_bottom_data_->count(softmax_axis_ + 1);
  int dim = softmax_out->count() / outer_num_;
  // loop through the softmax output to gather values that would contribute
  // to final loss, and do elementwise -log(max(FLT_MIN, x)) 
  const Dtype* softmax_out_data = softmax_out->cpu_data();
  const Dtype* label = this->blob_bottom_label_->cpu_data();
  vector<Dtype> loss_entries;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_val = static_cast<int>(label[i * inner_num_ + j]);
      Dtype prob = softmax_out_data[i * dim + label_val * inner_num_ + j];
      loss_entries.push_back(-log(std::max(prob, Dtype(FLT_MIN))));
    }
  }
  // sort the losses in descending order and only add the first ohem_n_ losses
  std::sort(loss_entries.begin(), loss_entries.end(), std::greater<Dtype>());
  Dtype exp_loss = Dtype(0);
  for (int i = 0; i < ohem_n_; i++){
    exp_loss += loss_entries[i];
  }
  exp_loss /= Dtype(outer_num_); // because we set normalize to false, i.e. normalise by BATCH_SIZE
  // Do SoftmaxLoss forward
  SoftmaxWithLossLayer<Dtype> smllayer(layer_param);
  smllayer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  smllayer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype loss = this->blob_top_vec_[0]->cpu_data()[0];

  EXPECT_NEAR(exp_loss, loss, 1e-4);

}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientOHEM) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int ohem_n_ = this->blob_bottom_label_->count() / 2;
  layer_param.mutable_loss_param()->set_use_ohem(true);
  layer_param.mutable_loss_param()->set_ohem_n(ohem_n_);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
