/**
 * Written by Qizhu <liqizhu@robots.ox.ac.uk>
*/

#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/gn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

  template <typename TypeParam>
  class GNLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    GNLayerTest()
        : blob_bottom_(new Blob<Dtype>(2, 8, 2, 4)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~GNLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(GNLayerTest, TestDtypesAndDevices);

  TYPED_TEST(GNLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    int channels_per_group_ = 2;
    layer_param.mutable_gn_param()->set_channels_per_group(channels_per_group_);
    layer_param.mutable_gn_param()->mutable_slope_filler()->set_type("msra");
    layer_param.mutable_gn_param()->mutable_bias_filler()->set_type("msra");

    GNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype>* const scale = layer.blobs()[0].get();
    Blob<Dtype>* const bias = layer.blobs()[1].get();
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean and var
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    int num_groups_ = channels/channels_per_group_;
    assert(num_groups_ * channels_per_group_ == channels);

    for (int n = 0; n < num; ++n) {
      for (int g = 0; g < num_groups_; ++g) {
        Dtype sum = 0, var = 0;
        int grp_c0 = g * channels_per_group_;
        int grp_c1 = (g+1) * channels_per_group_;
        for (int c = grp_c0; c < grp_c1; ++c){
          Dtype gamma = scale->data_at(0,c,0,0);
          Dtype beta = bias->data_at(0,c,0,0);
          for ( int h = 0; h < height; ++h ) {
            for ( int w = 0; w < width; ++w ) {
              Dtype data = (this->blob_top_->data_at(n, c, h, w) - beta) / gamma;
              sum += data;
              var += data * data;
            }
          }
        }
        sum /= height * width * channels_per_group_;
        var /= height * width * channels_per_group_;

        const Dtype kErrorBound = 0.001;
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
      }
    }
  }

  TYPED_TEST(GNLayerTest, TestForwardInplace) {
    typedef typename TypeParam::Dtype Dtype;
    Blob<Dtype> blob_inplace(2, 8, 6, 4);
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    LayerParameter layer_param;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_inplace);
    blob_bottom_vec.push_back(&blob_inplace);
    blob_top_vec.push_back(&blob_inplace);

    int channels_per_group_ = 2;
    layer_param.mutable_gn_param()->set_channels_per_group(channels_per_group_);
    layer_param.mutable_gn_param()->mutable_slope_filler()->set_type("msra");
    layer_param.mutable_gn_param()->mutable_bias_filler()->set_type("msra");

    GNLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    Blob<Dtype>* const scale = layer.blobs()[0].get();
    Blob<Dtype>* const bias = layer.blobs()[1].get();
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Test mean
    int num = blob_inplace.num();
    int channels = blob_inplace.channels();
    int height = blob_inplace.height();
    int width = blob_inplace.width();

    int num_groups_ = channels/channels_per_group_;
    assert(num_groups_ * channels_per_group_ == channels);


    for (int n = 0; n < num; ++n) {
      for (int g = 0; g < num_groups_; ++g) {
        Dtype sum = 0, var = 0;
        int grp_c0 = g * channels_per_group_;
        int grp_c1 = (g+1) * channels_per_group_;
        for (int c = grp_c0; c < grp_c1; ++c){
          Dtype gamma = scale->data_at(0,c,0,0);
          Dtype beta = bias->data_at(0,c,0,0);
          for ( int h = 0; h < height; ++h ) {
            for ( int w = 0; w < width; ++w ) {
              Dtype data = (blob_inplace.data_at(n, c, h, w) - beta) / gamma;
              sum += data;
              var += data * data;
            }
          }
        }
        sum /= height * width * channels_per_group_;
        var /= height * width * channels_per_group_;

        const Dtype kErrorBound = 0.001;
        // expect zero mean
        EXPECT_NEAR(0, sum, kErrorBound);
        // expect unit variance
        EXPECT_NEAR(1, var, kErrorBound);
      }
    }
  }

  TYPED_TEST(GNLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    int channels_per_group_ = 2;
    layer_param.mutable_gn_param()->set_channels_per_group(channels_per_group_);
    layer_param.mutable_gn_param()->mutable_slope_filler()->set_type("msra");
    layer_param.mutable_gn_param()->mutable_bias_filler()->set_type("msra");

    GNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

}  // namespace caffe
