#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdlib>
// #include <ctime>
#include <cmath>

// #include "caffe/test/test_tvg_util.hpp"
#include "caffe/util/math_functions.hpp"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/bn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {
    template <typename TypeParam>
  class BNLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
    BNLayerTest()
        : blob_bottom_(new Blob<Dtype>(5, 4, 2, 3)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      // Randomise the mean and variance of the Gaussian filler
      std::srand(0);
      Dtype rand_mean_ = std::rand()%1000*0.01;
      Dtype rand_std_ = std::rand()%1000*0.01;
      filler_param.set_mean(rand_mean_);
      filler_param.set_std(rand_std_);
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);

      // Dtype* data = this->blob_bottom_->mutable_cpu_data();
      // data[0] = 1; data[1] = 5; data[2] = 2; data[3] = 3;

      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
    virtual ~BNLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(BNLayerTest, TestDtypesAndDevices);

  TYPED_TEST(BNLayerTest, TestForwardLocalStats) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    // The forward pass carries out the following:
    // x_hat_i = (x_i - mu_batch)/std_batch * r + d;
    // if use_global_stats: 
    //      r = sigma_batch/sigma_global; 
    //      d = (mu_batch-mu_global)/sigma_global;
    //      TODO: test Batch Renorm style clipping of r and d after they are enabled
    // else:
    //      r = 1s; d = 0;
    // y_i = gamma * x_hat_i + beta;

    // By default, frozen: false; use_global_stats: false.
    // This will make r = 1s, and d = 0s.
    // Random initialisation of scale (gamma) and bias (beta) results in:
    // y_i = (x_i - mu_batch)/std_batch * gamma + beta
    // Thus E(y_i) = beta and E(y_i^2) - beta^2 = gamma^2
    BNParameter *bn_param = layer_param.mutable_bn_param();
    bn_param->mutable_slope_filler()->set_type("msra");
    bn_param->mutable_bias_filler()->set_type("msra");
    BNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype>* const gamma = layer.blobs()[0].get();
    Blob<Dtype>* const beta = layer.blobs()[1].get();
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;
      // Find out the values of beta and gamma at the current channel
      Dtype beta_c_ = beta->data_at(0,j,0,0);
      Dtype gamma_c_ = gamma->data_at(0,j,0,0);

      const Dtype kErrorBound = 0.001;
      // expect E(y_i) = beta
      EXPECT_NEAR(beta_c_, sum, kErrorBound);
      // expect E(y_i^2) = gamma^2 + beta^2
      EXPECT_NEAR(beta_c_*beta_c_ + gamma_c_*gamma_c_, var, kErrorBound);
    }
  }


  TYPED_TEST(BNLayerTest, TestForwardGlobalStatsNoClipping) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    int NUM_PASSES = 1;

    // use_global_stats: true.
    // frozen: false
    // This will make r = sigma_batch/sigma_global; 
    //                d = (mu_batch-mu_global)/sigma_global; (no clipping atm)
    // Random initialisation of scale (gamma) and bias (beta) results in:
    // y_i = [(x_i - mu_batch)/std_batch * r + d] * gamma + beta
    // Thus E(y_i) = d*gamma+beta and E(y_i^2) = gamma^2*r^2 + (d*gamma + beta)^2
    // tvg::TestUtils::PrintBlob(*(this->blob_bottom_), false, "bottom blob");
    BNParameter *bn_param = layer_param.mutable_bn_param();
    // printf("Phase: %s\n\n", layer_param.phase()?"TEST":"TRAIN");
    bn_param->set_use_global_stats(true);
    //bn_param->set_frozen(false);
    // printf("Frozen: %d\n\n", bn_param->frozen());
    bn_param->set_momentum(Dtype(1.));
    bn_param->mutable_slope_filler()->set_type("msra");
    bn_param->mutable_bias_filler()->set_type("msra");
    BNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    Blob<Dtype>* const gamma = layer.blobs()[0].get();
    Blob<Dtype>* const beta = layer.blobs()[1].get();
    // tvg::TestUtils::PrintBlob(*gamma, false, "stored scale");
    // tvg::TestUtils::PrintBlob(*beta, false, "stored bias");
    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();
    Dtype mu_global[channels];
    Dtype var_global[channels];
    Dtype eps = Dtype(1e-5);
    FillerParameter filler_param;
    Dtype rand_mean_, rand_std_;
    Dtype mu_batch, var_batch, d, r, beta_c_, gamma_c_, exp_mean, exp_var;
    for (int k = 0; k < NUM_PASSES; k++){
        // Run a few forward passes
        Blob<Dtype>* mu_global_ptr = layer.blobs()[2].get();
        Blob<Dtype>* var_global_ptr = layer.blobs()[3].get();
        // randomly initialise the stored mean and variance
        rand_mean_ = std::rand()%1000*0.01 - 5;
        rand_std_ = std::rand()%1000*0.01;
        filler_param.set_mean(rand_mean_);
        filler_param.set_std(rand_std_);
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(mu_global_ptr);
        rand_mean_ = std::rand()%1000*0.01-5;
        rand_std_ = std::rand()%1000*0.01;
        filler_param.set_mean(rand_mean_);
        filler_param.set_std(rand_std_);
        GaussianFiller<Dtype> filler2(filler_param);
        filler2.Fill(var_global_ptr);
        caffe_powx(var_global_ptr->count(), var_global_ptr->cpu_data(),
          Dtype(2), var_global_ptr->mutable_cpu_data());
        // tvg::TestUtils::PrintBlob(*mu_global_ptr, false, "stored mean before forward pass");
        // tvg::TestUtils::PrintBlob(*var_global_ptr, false, "stored var before forward pass");
        for (int c = 0; c < channels; ++c){
            // make a copy of the global stats before forward pass updates it
            mu_global[c] = mu_global_ptr->data_at(0,c,0,0);
            var_global[c] = var_global_ptr->data_at(0,c,0,0);
        }
        // randomly assign data to bottom for each forward pass
        rand_mean_ = std::rand()%1000*0.01 - 5;
        rand_std_ = std::rand()%1000*0.01;
        filler_param.set_mean(rand_mean_);
        filler_param.set_std(rand_std_);
        GaussianFiller<Dtype> filler3(filler_param);
        filler3.Fill(this->blob_bottom_);

        // do a forward pass
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        // tvg::TestUtils::PrintBlob(*(this->blob_top_), false, "top blob");

        // calculate output mean and variance, input mean and variance
        for (int j = 0; j < channels; ++j) {
            // printf("Testing channel %d\n", j);
            Dtype sum = 0, var = 0, bottom_sum = 0, bottom_var = 0;
            for (int i = 0; i < num; ++i) {
                for ( int k = 0; k < height; ++k ) {
                for ( int l = 0; l < width; ++l ) {
                    Dtype data = this->blob_top_->data_at(i, j, k, l);
                    Dtype bottom_data = this->blob_bottom_->data_at(i,j,k,l);
                    sum += data;
                    var += data * data;
                    bottom_sum += bottom_data;
                    bottom_var += bottom_data * bottom_data;
                }
                }
            }
            sum /= height * width * num;
            var /= height * width * num;
            bottom_sum /= height * width * num;
            bottom_var /= height * width * num;
            // Compute batch mean and batch var for the current channel
            mu_batch = bottom_sum;
            var_batch = bottom_var - mu_batch * mu_batch;
            // Compute r and d for the current channel
            r = sqrt((var_batch + eps) / (var_global[j] + eps));
            d = (mu_batch - mu_global[j]) / sqrt((var_global[j] + eps));
            // Find out the values of beta and gamma at the current channel
            beta_c_ = beta->data_at(0,j,0,0);
            gamma_c_ = gamma->data_at(0,j,0,0);
            // Compute the expected mean and variance at the output
            exp_mean = d * gamma_c_ + beta_c_;
            exp_var = gamma_c_*gamma_c_*r*r + (d*gamma_c_ + beta_c_)*(d*gamma_c_ + beta_c_);

            const Dtype kErrorFactor = 1e-4;
            Dtype mean_scale = std::max<Dtype>(std::max(exp_mean, sum), Dtype(1.));
            Dtype var_scale = std::max<Dtype>(std::max(exp_var, var), Dtype(1.));
            // expect zero mean
            EXPECT_NEAR(exp_mean, sum, kErrorFactor * mean_scale);
            // expect unit variance
            EXPECT_NEAR(exp_var, var, kErrorFactor * var_scale);
      }
    }
  }

TYPED_TEST(BNLayerTest, TestForwardGlobalStatsWithClipping) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    int NUM_PASSES = 1;

    // use_global_stats: true.
    // frozen: false
    // This will make r = sigma_batch/sigma_global; 
    //                d = (mu_batch-mu_global)/sigma_global; (no clipping atm)
    // Random initialisation of scale (gamma) and bias (beta) results in:
    // y_i = [(x_i - mu_batch)/std_batch * r + d] * gamma + beta
    // Thus E(y_i) = d*gamma+beta and E(y_i^2) = gamma^2*r^2 + (d*gamma + beta)^2
    BNParameter *bn_param = layer_param.mutable_bn_param();
    // printf("Phase: %s\n\n", layer_param.phase()?"TEST":"TRAIN");
    bn_param->set_use_global_stats(true);
    //bn_param->set_frozen(false);
    // printf("Frozen: %d\n\n", bn_param->frozen());
    bn_param->set_momentum(Dtype(1.));

    bn_param->set_clip_r(true);
    bn_param->set_clip_d(true);
    Dtype r_max = std::rand()%10*0.01 + 1;
    Dtype d_max = std::rand()%10*0.01;
    // Dtype r_max = Dtype(1.1);
    // Dtype d_max = Dtype(0.1);
    bn_param->set_r_max(r_max);
    bn_param->set_d_max(d_max);

    bn_param->mutable_slope_filler()->set_type("msra");
    // bn_param->mutable_slope_filler()->set_value(1);
    bn_param->mutable_bias_filler()->set_type("msra");
    // bn_param->mutable_bias_filler()->set_value(0);

    BNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    Blob<Dtype>* const gamma = layer.blobs()[0].get();
    Blob<Dtype>* const beta = layer.blobs()[1].get();
    // tvg::TestUtils::PrintBlob(*gamma, false, "stored scale");
    // tvg::TestUtils::PrintBlob(*beta, false, "stored bias");
    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();
    Dtype mu_global[channels];
    Dtype var_global[channels];
    Dtype eps = Dtype(1e-5);
    FillerParameter filler_param;
    Dtype rand_mean_, rand_std_;
    Dtype mu_batch, var_batch, d, r, beta_c_, gamma_c_, exp_mean, exp_var;
    for (int k = 0; k < NUM_PASSES; k++){
        // Run a few forward passes
        Blob<Dtype>* mu_global_ptr = layer.blobs()[2].get();
        Blob<Dtype>* var_global_ptr = layer.blobs()[3].get();
        // randomly initialise the stored mean and variance
        rand_mean_ = std::rand()%1000*0.01 - 5;
        rand_std_ = std::rand()%1000*0.01;
        filler_param.set_mean(rand_mean_);
        filler_param.set_std(rand_std_);
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(mu_global_ptr);
        rand_mean_ = std::rand()%1000*0.01-5;
        rand_std_ = std::rand()%1000*0.01;
        filler_param.set_mean(rand_mean_);
        filler_param.set_std(rand_std_);
        GaussianFiller<Dtype> filler2(filler_param);
        filler2.Fill(var_global_ptr);
        caffe_powx(var_global_ptr->count(), var_global_ptr->cpu_data(),
          Dtype(2), var_global_ptr->mutable_cpu_data());
        // Dtype* mu_data = mu_global_ptr->mutable_cpu_data();
        // Dtype* var_data = var_global_ptr->mutable_cpu_data();
        // mu_data[0] = 2;
        // var_data[0] = 4;
        // tvg::TestUtils::PrintBlob(*mu_global_ptr, false, "stored mean before forward pass");
        // tvg::TestUtils::PrintBlob(*var_global_ptr, false, "stored var before forward pass");
        for (int c = 0; c < channels; ++c){
            // make a copy of the global stats before forward pass updates it
            mu_global[c] = mu_global_ptr->data_at(0,c,0,0);
            var_global[c] = var_global_ptr->data_at(0,c,0,0);
        }
        // randomly assign data to bottom for each forward pass
        rand_mean_ = std::rand()%1000*0.01 - 5;
        rand_std_ = std::rand()%1000*0.01;
        filler_param.set_mean(rand_mean_);
        filler_param.set_std(rand_std_);
        GaussianFiller<Dtype> filler3(filler_param);
        filler3.Fill(this->blob_bottom_);
        // tvg::TestUtils::PrintBlob(*(this->blob_bottom_), false, "bottom blob");

        // do a forward pass
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        // tvg::TestUtils::PrintBlob(*(this->blob_top_), false, "top blob");

        // calculate output mean and variance, input mean and variance
        for (int j = 0; j < channels; ++j) {
            // printf("Testing channel %d\n", j);
            Dtype sum = 0, var = 0, bottom_sum = 0, bottom_var = 0;
            for (int i = 0; i < num; ++i) {
                for ( int k = 0; k < height; ++k ) {
                for ( int l = 0; l < width; ++l ) {
                    Dtype data = this->blob_top_->data_at(i, j, k, l);
                    Dtype bottom_data = this->blob_bottom_->data_at(i,j,k,l);
                    sum += data;
                    var += data * data;
                    bottom_sum += bottom_data;
                    bottom_var += bottom_data * bottom_data;
                }
                }
            }
            sum /= height * width * num;
            var /= height * width * num;
            bottom_sum /= height * width * num;
            bottom_var /= height * width * num;
            // Compute batch mean and batch var for the current channel
            mu_batch = bottom_sum;
            var_batch = bottom_var - mu_batch * mu_batch;
            // Compute r and d for the current channel
            r = sqrt((var_batch + eps) / (var_global[j] + eps));
            // clip r
            r = (r<r_max) ? r : r_max;
            r = (r>(Dtype(1.)/r_max)) ? r : (Dtype(1.)/r_max);

            d = (mu_batch - mu_global[j]) / sqrt((var_global[j] + eps));
            // clip d
            d = (d<d_max) ? d : d_max;
            d = (d>(-d_max)) ? d : (-d_max);
            // Find out the values of beta and gamma at the current channel
            beta_c_ = beta->data_at(0,j,0,0);
            gamma_c_ = gamma->data_at(0,j,0,0);
            // Compute the expected mean and variance at the output
            exp_mean = d * gamma_c_ + beta_c_;
            exp_var = gamma_c_*gamma_c_*r*r + (d*gamma_c_ + beta_c_)*(d*gamma_c_ + beta_c_);

            const Dtype kErrorFactor = 1e-4;
            Dtype mean_scale = std::max<Dtype>(std::max(exp_mean, sum), Dtype(1.));
            Dtype var_scale = std::max<Dtype>(std::max(exp_var, var), Dtype(1.));
            // expect zero mean
            EXPECT_NEAR(exp_mean, sum, kErrorFactor * mean_scale);
            // expect unit variance
            EXPECT_NEAR(exp_var, var, kErrorFactor * var_scale);
      }
    }
  }

  TYPED_TEST(BNLayerTest, TestGradientLocalStats) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BNParameter *bn_param = layer_param.mutable_bn_param();
    bn_param->mutable_slope_filler()->set_type("msra");
    bn_param->mutable_bias_filler()->set_type("msra");

    BNLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

TYPED_TEST(BNLayerTest, TestGradientGlobalStats) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BNParameter *bn_param = layer_param.mutable_bn_param();
    // printf("Phase: %s\n\n", layer_param.phase()?"TEST":"TRAIN");
    //bn_param->set_frozen(false);
    // printf("Frozen: %d\n\n", bn_param->frozen());
    bn_param->set_use_global_stats(true);
    bn_param->set_is_gradient_check(true);
    bn_param->mutable_slope_filler()->set_type("msra");
    // bn_param->mutable_slope_filler()->set_value(1);
    bn_param->mutable_bias_filler()->set_type("msra");
    //bn_param->mutable_bias_filler()->set_value(0);
    // set momentum to 1 to prevent stored mean and var from changing 
    // when we keep top_data_id constant and vary bottom feat_id
    bn_param->set_momentum(1.);

    BNLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // Blob<Dtype> &gamma = *(layer.blobs()[0]);
    // Blob<Dtype> &beta = *(layer.blobs()[1]);
    // tvg::TestUtils::PrintBlob(gamma, false, "stored scale");
    // tvg::TestUtils::PrintBlob(beta, false, "stored bias");
    Blob<Dtype> &mu_global = *(layer.blobs()[2]);
    Blob<Dtype> &var_global = *(layer.blobs()[3]);

    //
    // Blob<Dtype>* bottom_blob = this->blob_bottom_vec_[0];
    // Dtype* data = bottom_blob->mutable_cpu_data();
    // data[0] = 1; data[1] = 5; data[2] = 2; data[3] = 3;
    // tvg::TestUtils::PrintBlob(*(this->blob_bottom_), false, "bottom blob");
  
    // Dtype* mu_data = mu_global.mutable_cpu_data();
    // mu_data[0] = 2;

    // Dtype* var_data = var_global.mutable_cpu_data();
    // var_data[0] = 4 - 1e-5;

    //
    // randomly initialise the stored mean and variance
    FillerParameter filler_param;
    Dtype rand_mean_, rand_std_;
    rand_mean_ = std::rand()%1000*0.01 - 5;
    rand_std_ = std::rand()%1000*0.01;
    filler_param.set_mean(rand_mean_);
    filler_param.set_std(rand_std_);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&mu_global);
    rand_mean_ = std::rand()%1000*0.01-5;
    rand_std_ = std::rand()%1000*0.01;
    filler_param.set_mean(rand_mean_);
    filler_param.set_std(rand_std_);
    GaussianFiller<Dtype> filler2(filler_param);
    filler2.Fill(&var_global);
    caffe_powx(var_global.count(), var_global.cpu_data(),
        Dtype(2), var_global.mutable_cpu_data());
    // tvg::TestUtils::PrintBlob(mu_global, false, "stored mean before forward pass");
    // tvg::TestUtils::PrintBlob(var_global, false, "stored var before forward pass");
    // do a forward pass
    // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // tvg::TestUtils::PrintBlob(*(this->blob_top_), false, "top blob");
    // Dtype* top_diff = this->blob_top_->mutable_cpu_diff();
    // top_diff[0] = 2; top_diff[1] = 0;
    // tvg::TestUtils::PrintBlob(*(this->blob_top_), true, "top diff");
    // vector<bool> propagate_down(1, true);
    // layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
    // tvg::TestUtils::PrintBlob(*bottom_blob, true, "bottom diff");
    // data[0] += 1;
    // printf("Forward data plus\n\n");
    // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // tvg::TestUtils::PrintBlob(*(this->blob_top_), false, "top blob plus");
    // data[0] += -2;
    // printf("Forward data minus\n\n");
    // layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // tvg::TestUtils::PrintBlob(*(this->blob_top_), false, "top blob minus");
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
       this->blob_top_vec_);
  }

} // namespace caffe
