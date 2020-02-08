#include <string>
#include <vector>
#include <math.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/gradient_block_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GradientBlockLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GradientBlockLayerTest()
      : blob_bottom_a_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_bottom_b_(new Blob<Dtype>(2, 3, 5, 5)),
        blob_bottom_c_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_d_(new Blob<Dtype>(2, 3, 3, 5)),
        blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_a_);
    filler.Fill(this->blob_bottom_b_);
    filler.Fill(this->blob_bottom_c_);
    filler.Fill(this->blob_bottom_d_);
    blob_bottom_vec_.push_back(this->blob_bottom_a_);
    blob_top_vec_.push_back(this->blob_top_a_);

    multi_blob_bottom_vec_.push_back(this->blob_bottom_a_);
    multi_blob_bottom_vec_.push_back(this->blob_bottom_b_);
    multi_blob_bottom_vec_.push_back(this->blob_bottom_c_);
    multi_blob_bottom_vec_.push_back(this->blob_bottom_d_);

    multi_blob_top_vec_.push_back(this->blob_top_a_); 
    multi_blob_top_vec_.push_back(this->blob_top_b_);
    multi_blob_top_vec_.push_back(this->blob_bottom_c_); // in_place
    multi_blob_top_vec_.push_back(this->blob_bottom_d_); // in_place

  }
  virtual ~GradientBlockLayerTest() {
    delete blob_bottom_a_;
    delete blob_bottom_b_;
    delete blob_bottom_c_;
    delete blob_bottom_d_;
    delete blob_top_a_;
    delete blob_top_b_;
  }
  Blob<Dtype>* const blob_bottom_a_;
  Blob<Dtype>* const blob_bottom_b_;
  Blob<Dtype>* const blob_bottom_c_;
  Blob<Dtype>* const blob_bottom_d_;
  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> multi_blob_bottom_vec_;
  vector<Blob<Dtype>*> multi_blob_top_vec_;
};

TYPED_TEST_CASE(GradientBlockLayerTest, TestDtypesAndDevices);

TYPED_TEST(GradientBlockLayerTest, TestForwardSimple) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GradientBlockLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
    Dtype bottom_value = this->blob_bottom_a_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_a_->cpu_data()[i]);
  }
}

TYPED_TEST(GradientBlockLayerTest, TestForwardSimpleInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GradientBlockLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_bottom_vec_);
  // cache the bottom data to bottom diff for checking later
  caffe_copy(this->blob_bottom_a_->count(), this->blob_bottom_a_->cpu_data(),
    this->blob_bottom_a_->mutable_cpu_diff());
  layer.Forward(this->blob_bottom_vec_, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
    Dtype bottom_value = this->blob_bottom_a_->cpu_data()[i];
    // check against the cached value stored in bottom diff
    EXPECT_EQ(bottom_value, this->blob_bottom_a_->cpu_diff()[i]);
  }
}

TYPED_TEST(GradientBlockLayerTest, TestForwardMultiMixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GradientBlockLayer<Dtype> layer(layer_param);
  // Test the case where we initialise with 4 bottoms and 4 tops
  // with different dimensions
  // Two are in_place, and the other two are not.
  layer.SetUp(this->multi_blob_bottom_vec_, this->multi_blob_top_vec_);
  // cache the bottom data to bottom diff for checking later
  for (int k = 0; k < this->multi_blob_bottom_vec_.size(); ++k) {
    caffe_copy(this->multi_blob_bottom_vec_[k]->count(), 
        this->multi_blob_bottom_vec_[k]->cpu_data(),
        this->multi_blob_bottom_vec_[k]->mutable_cpu_diff());
  }
  layer.Forward(this->multi_blob_bottom_vec_, this->multi_blob_top_vec_);
  for (int k = 0; k < this->multi_blob_bottom_vec_.size(); ++k) {
    for (int i = 0; i < this->multi_blob_bottom_vec_[k]->count(); ++i) {
        Dtype bottom_value = this->multi_blob_bottom_vec_[k]->cpu_data()[i];
        // check that the top and bottom are equal
        EXPECT_EQ(bottom_value, this->multi_blob_top_vec_[k]->cpu_data()[i]);
        // check against the cached value stored in bottom diff
        EXPECT_EQ(bottom_value, this->multi_blob_bottom_vec_[k]->cpu_diff()[i]);
    }
  }
}

TYPED_TEST(GradientBlockLayerTest, TestGradientSimple) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GradientBlockLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // fill the top_diff with random values
  caffe_rng_gaussian<Dtype>(this->blob_top_a_->count(), Dtype(0),
        Dtype(1), this->blob_top_a_->mutable_cpu_diff());
  // do a forward pass and a backward pass
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  std::vector<bool> propagate_down;
  propagate_down.push_back(true);
  layer.Backward(this->blob_bottom_vec_, propagate_down, this->blob_top_vec_);
  // sum of abs(bottom_diff) should be 0
  Dtype acum_bottom_diff = Dtype(0);
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
      Dtype bottom_diff = this->blob_bottom_a_->cpu_diff()[i];
      acum_bottom_diff += std::fabs(bottom_diff);
  }
  EXPECT_EQ(acum_bottom_diff, Dtype(0));
}

TYPED_TEST(GradientBlockLayerTest, TestGradientSimpleInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GradientBlockLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_bottom_vec_);
  // fill the top_diff with random values
  caffe_rng_gaussian<Dtype>(this->blob_bottom_a_->count(), Dtype(0),
        Dtype(1), this->blob_bottom_a_->mutable_cpu_diff());
  // do a forward pass and a backward pass
  layer.Forward(this->blob_bottom_vec_, this->blob_bottom_vec_);
  std::vector<bool> propagate_down (this->blob_bottom_vec_.size(), true);;
  layer.Backward(this->blob_bottom_vec_, propagate_down, this->blob_bottom_vec_);
  // sum of abs(bottom_diff) should be 0
  Dtype acum_bottom_diff = Dtype(0);
  for (int i = 0; i < this->blob_bottom_a_->count(); ++i) {
      Dtype bottom_diff = this->blob_bottom_a_->cpu_diff()[i];
      acum_bottom_diff += std::fabs(bottom_diff);
  }
  EXPECT_EQ(acum_bottom_diff, Dtype(0));
}

TYPED_TEST(GradientBlockLayerTest, TestGradientMultiMixed) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  GradientBlockLayer<Dtype> layer(layer_param);
  layer.SetUp(this->multi_blob_bottom_vec_, this->multi_blob_top_vec_);
  // fill the top_diff with random values
  for (int k = 0; k < this->multi_blob_bottom_vec_.size(); ++k) {
    caffe_rng_gaussian<Dtype>(this->multi_blob_top_vec_[k]->count(), Dtype(0),
        Dtype(1), this->multi_blob_top_vec_[k]->mutable_cpu_diff());
  }
  // do a forward pass and a backward pass
  layer.Forward(this->multi_blob_bottom_vec_, this->multi_blob_top_vec_);
  std::vector<bool> propagate_down (this->multi_blob_bottom_vec_.size(), true);
  layer.Backward(this->multi_blob_bottom_vec_, propagate_down, this->multi_blob_top_vec_);
  // sum of abs(bottom_diff) should be 0
  Dtype acum_bottom_diff = Dtype(0);
  for (int k = 0; k < this->multi_blob_bottom_vec_.size(); ++k) {
    for (int i = 0; i < this->multi_blob_bottom_vec_[k]->count(); ++i) {
      Dtype bottom_diff = this->multi_blob_bottom_vec_[k]->cpu_diff()[i];
      acum_bottom_diff += std::fabs(bottom_diff);
    }
  }

  EXPECT_EQ(acum_bottom_diff, Dtype(0));
}

}  // namespace caffe
