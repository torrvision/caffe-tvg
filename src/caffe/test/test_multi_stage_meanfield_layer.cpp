#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/meanfield_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_tvg_util.hpp"

#include "caffe/test/test_tvg_util.hpp"

namespace caffe {

template <typename TypeParam>
class MultiStageMeanfieldLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiStageMeanfieldLayerTest() {}

  virtual void SetUp() {

  }

  virtual ~MultiStageMeanfieldLayerTest() {

  }
};

TYPED_TEST_CASE(MultiStageMeanfieldLayerTest, TestDtypesAndDevices);


  TYPED_TEST(MultiStageMeanfieldLayerTest, TestInference) {
    typedef typename TypeParam::Dtype Dtype;

    printf("Starting test. Inference with superpixel potentials \n");

    if (sizeof(Dtype) != sizeof(float)) {
      printf("Dtype is not float\n");
      return;
    }

    std::string working_dir = "caffe/src/caffe/test/test_data/mf_layer/";
    std::string image_title = "2007_000663";

    const int channels = 21, height = 500, width = 500;
    int real_height, real_width;

    Blob<Dtype> unary_blob(1, channels, height, width);
    Blob<Dtype> rgb_blob(1, 3, height, width);
    Blob<Dtype> previous_output_blob(1, channels, height, width);
    Blob<Dtype> indices_bob(1, 1, 1, 1);

    tvg::TestUtils::FillFromDat(&unary_blob, working_dir + image_title + ".caffedat");
    previous_output_blob.CopyFrom(unary_blob);
    tvg::TestUtils::read_image(&rgb_blob, working_dir + image_title + ".jpg", real_height, real_width);
    indices_bob.mutable_cpu_data()[0] = 0;

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.push_back(&unary_blob);
    bottom_vec.push_back(&previous_output_blob);
    bottom_vec.push_back(&rgb_blob);
    bottom_vec.push_back(&indices_bob);

    Blob<Dtype> top_blob;
    Blob<Dtype> top_blob2;
    top_vec.push_back(&top_blob);
    top_vec.push_back(&top_blob2);

    LayerParameter layer_param;
    MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
    ms_mf_param->set_num_iterations(5);
    //ms_mf_param->set_spatial_filter_weight(30);
    //ms_mf_param->set_bilateral_filter_weight(15);
    ms_mf_param->set_spatial_filter_weights_str("3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3");
    ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5");

    ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
    ms_mf_param->set_theta_alpha(160);
    ms_mf_param->set_theta_beta(3);
    ms_mf_param->set_theta_gamma(3);

    // Detection potentials
    ms_mf_param->set_detection_potentials_enabled(false);
    ms_mf_param->set_detection_dat_input_dir(working_dir);
    ms_mf_param->set_detection_x_weights_str("32 32 32 32 32 32 32 32 32 1024 32 32 32 32 32 32 32 32 128 32 32");
    ms_mf_param->set_detection_y_weights_str("1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1");
    ms_mf_param->set_detection_potentials_max_score(1);
    ms_mf_param->set_detection_potentials_epsilon(0.0001);

    // Superpixel potentials
    ms_mf_param->set_ho_potentials_enabled(true);
    ms_mf_param->set_ho_dat_input_dir(working_dir);
    ms_mf_param->set_ho_w_param_str("10000 0 0 0");
    ms_mf_param->set_ho_w2_param_str("0 0 0 0");
    ms_mf_param->set_ho_num_layers(4);

    short labels[height * width];
    tvg::TestUtils::GetLabelMap(previous_output_blob, labels);

    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_ho_beforeMF_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_ho_beforeMF_gpu.png", height, width);
    }
    MultiStageMeanfieldLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);
    layer.Forward(bottom_vec, top_vec);


    tvg::TestUtils::get_label_map(top_blob, labels);
    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_ho_afterMF_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_ho_afterMF_gpu.png", height, width);
    }
}



TYPED_TEST(MultiStageMeanfieldLayerTest, TestParameterLoad) {
  typedef typename TypeParam::Dtype Dtype;
  const int n = 1, c = 4, H = 5, W = 5;

  Blob<Dtype> unary_terms_blob(n, c, H, W);
  Blob<Dtype> previous_output_blob(n, c, H, W);
  Blob<Dtype> rgb_blob(n, 3, H, W);
  Blob<Dtype> indices_bob(1, 1, 1, 1);

  vector<Blob<Dtype>*> bottom_vec, top_vec;
  bottom_vec.clear();
  bottom_vec.push_back(&unary_terms_blob);
  bottom_vec.push_back(&previous_output_blob);
  bottom_vec.push_back(&rgb_blob);
  bottom_vec.push_back(&indices_bob);

  Blob<Dtype> top_blob;
  Blob<Dtype> y_variable_blob;
  top_vec.push_back(&top_blob);
  top_vec.push_back(&y_variable_blob);

  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_num_iterations(10);
  ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);
  ms_mf_param->set_spatial_filter_weights_str("1.1 2.2 3.3 4.4");
  ms_mf_param->set_bilateral_filter_weights_str("111.1 222.2 333.3 444.4");

  // Detection potentials
  ms_mf_param->set_detection_potentials_enabled(true);
  ms_mf_param->set_detection_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
  ms_mf_param->set_detection_potentials_max_score(5);
  ms_mf_param->set_detection_potentials_epsilon(0.001);
  ms_mf_param->set_detection_y_weights_str("11 22 33 44");
  ms_mf_param->set_detection_x_weights_str("0.1 0.2 0.3 0.4");

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(bottom_vec, top_vec);

  vector<shared_ptr<Blob<Dtype> > > & param_blobs = layer.blobs();

  tvg::TestUtils::PrintBlob(*param_blobs[0], false, "Spatial weights");
  tvg::TestUtils::PrintBlob(*param_blobs[1], false, "Bilateral weights");
  tvg::TestUtils::PrintBlob(*param_blobs[2], false, "Compatibility matrix");
  tvg::TestUtils::PrintBlob(*param_blobs[3], false, "Detection X weights");
  tvg::TestUtils::PrintBlob(*param_blobs[4], false, "Detection Y weights");

  const float tolerance = 1e-4;
  EXPECT_NEAR(2.2, param_blobs[0]->cpu_data()[5], tolerance);
  EXPECT_NEAR(444.4, param_blobs[1]->cpu_data()[15], tolerance);
  EXPECT_NEAR(-1, param_blobs[2]->cpu_data()[0], tolerance);

  EXPECT_NEAR(0.1, param_blobs[3]->cpu_data()[0], tolerance);
  EXPECT_NEAR(0.2, param_blobs[3]->cpu_data()[1], tolerance);
  EXPECT_NEAR(0.3, param_blobs[3]->cpu_data()[2], tolerance);
  EXPECT_NEAR(0.4, param_blobs[3]->cpu_data()[3], tolerance);

  EXPECT_NEAR(11, param_blobs[4]->cpu_data()[0], tolerance);
  EXPECT_NEAR(22, param_blobs[4]->cpu_data()[1], tolerance);
  EXPECT_NEAR(33, param_blobs[4]->cpu_data()[2], tolerance);
  EXPECT_NEAR(44, param_blobs[4]->cpu_data()[3], tolerance);
}


  TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradientPairwiseOnly) {
    typedef typename TypeParam::Dtype Dtype;
    const int n = 1, c = 4, H = 5, W = 5;

    if (sizeof(Dtype) != sizeof(double))
      return;

    LOG(INFO) << "Running Gradient Test with only Pairwise Potentials";
    printf("Running Gradient Test with only Pairwise Potentials\n");

    Blob<Dtype> unary_terms_blob(n, c, H, W);
    Blob<Dtype> previous_output_blob(n, c, H, W);
    Blob<Dtype> rgb_blob(n, 3, H, W);
    Blob<Dtype> indices_bob(1, 1, 1, 1);

    tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
    tvg::TestUtils::FillAsLogProb(&previous_output_blob);
    tvg::TestUtils::FillAsRGB(&rgb_blob);
    indices_bob.mutable_cpu_data()[0] = 10;

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.clear();
    bottom_vec.push_back(&unary_terms_blob);
    bottom_vec.push_back(&previous_output_blob);
    bottom_vec.push_back(&rgb_blob);
    bottom_vec.push_back(&indices_bob);

    Blob<Dtype> top_blob;
    Blob<Dtype> top_blob2;
    top_vec.push_back(&top_blob);
    top_vec.push_back(&top_blob2);

    LayerParameter layer_param;
    MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
    ms_mf_param->set_num_iterations(5);
    ms_mf_param->set_bilateral_filter_weights_str("1.0 1.0 1.0 1.0 2.0");
    ms_mf_param->set_spatial_filter_weights_str("2.0 2.0 2.0 2.0 3.0");
    ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
    ms_mf_param->set_theta_alpha(5);
    ms_mf_param->set_theta_beta(2);
    ms_mf_param->set_theta_gamma(3);

    // Detection potentials
    ms_mf_param->set_detection_potentials_enabled(false);

    MultiStageMeanfieldLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);

    layer.Forward(bottom_vec, top_vec);
    vector<bool> v(7, true);
    tvg::TestUtils::FillWithUpperBound(top_vec[0], 100, true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
    layer.Backward(top_vec, v, bottom_vec);

    // Parameters to this function are: "blob", print diff if true, else print data, "Title
    tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
    tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
    tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
    tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
    tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");

    // This is where the actual gradient checking occurs. The previous stuff are to manually look at some of the gradients.
    GradientChecker<Dtype> checker(1e-2, 1e-3);

    // Check gradients w.r.t. unary terms
    // This function checks gradients with respect to parameter blobs as well
    // The last parameter specifies which of the layer inputs we want to check gradients with respect to.
    // We do not want to compute gradients with respect to the input RGB image, or the indices.
    checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

    // Check gradients w.r.t. previous outputs
    checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 1, true);
  } // end - TestGradient()


TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradientDetection) {
  typedef typename TypeParam::Dtype Dtype;
  const int n = 1, c = 4, H = 5, W = 5;

  if (sizeof(Dtype) != sizeof(double))
    return;

  LOG(INFO) << "Running Gradient Test with Detection Potentials";

  Blob<Dtype> unary_terms_blob(n, c, H, W);
  Blob<Dtype> previous_output_blob(n, c, H, W);
  Blob<Dtype> rgb_blob(n, 3, H, W);
  Blob<Dtype> indices_bob(1, 1, 1, 1);

  tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
  tvg::TestUtils::FillAsLogProb(&previous_output_blob);
  tvg::TestUtils::FillAsRGB(&rgb_blob);
  indices_bob.mutable_cpu_data()[0] = 10;

  vector<Blob<Dtype>*> bottom_vec, top_vec;
  bottom_vec.clear();
  bottom_vec.push_back(&unary_terms_blob);
  bottom_vec.push_back(&previous_output_blob);
  bottom_vec.push_back(&rgb_blob);
  bottom_vec.push_back(&indices_bob);

  Blob<Dtype> top_blob;
  Blob<Dtype> top_blob2;
  top_vec.push_back(&top_blob);
  top_vec.push_back(&top_blob2);

  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_num_iterations(5);
  ms_mf_param->set_bilateral_filter_weights_str("1.0 1.0 1.0 1.0 2.0");
  ms_mf_param->set_spatial_filter_weights_str("2.0 2.0 2.0 2.0 3.0");
  ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);

  // Detection potentials
  ms_mf_param->set_detection_potentials_enabled(true);
  ms_mf_param->set_detection_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
  ms_mf_param->set_detection_x_weights_str("100 100 100 100 100");
  ms_mf_param->set_detection_y_weights_str("10 5 5 5 10");
  ms_mf_param->set_detection_potentials_max_score(5);
  ms_mf_param->set_detection_potentials_epsilon(0.001);

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(bottom_vec, top_vec);


  layer.Forward(bottom_vec, top_vec);
  vector<bool> v(7, true);
  tvg::TestUtils::FillWithUpperBound(top_vec[0], 100, true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
  layer.Backward(top_vec, v, bottom_vec);

  // Parameters to this function are: "blob", print diff if true, else print data, "Title
  tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
  tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
  tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
  tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
  tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
  tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
  tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
  tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
  tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
  tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");

  GradientChecker<Dtype> checker(1e-2, 1e-3);

  // Check gradients w.r.t. unary terms
  checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

  // Check gradients w.r.t. previous outputs
  checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 1, true);
} // end - TestGradient()*/

  TYPED_TEST(MultiStageMeanfieldLayerTest, TestInferenceDetections) {
      typedef typename TypeParam::Dtype Dtype;

      printf("Starting test\n");

      if (sizeof(Dtype) != sizeof(float)) {
        printf("Dtype is not float\n");
        return;
      }

      std::string working_dir = "caffe/src/caffe/test/test_data/mf_layer/";
      std::string image_title = "2007_000663";

      const int channels = 21, height = 500, width = 500;
      int real_height, real_width;

      Blob<Dtype> unary_blob(1, channels, height, width);
      Blob<Dtype> rgb_blob(1, 3, height, width);
      Blob<Dtype> previous_output_blob(1, channels, height, width);
      Blob<Dtype> indices_bob(1, 1, 1, 1);

      tvg::TestUtils::fill_from_dat(&unary_blob, working_dir + image_title + ".caffedat");
      previous_output_blob.CopyFrom(unary_blob);
      tvg::TestUtils::read_image(&rgb_blob, working_dir + image_title + ".jpg", real_height, real_width);
      indices_bob.mutable_cpu_data()[0] = 0;

      vector<Blob<Dtype>*> bottom_vec, top_vec;
      bottom_vec.push_back(&unary_blob);
      bottom_vec.push_back(&previous_output_blob);
      bottom_vec.push_back(&rgb_blob);
      bottom_vec.push_back(&indices_bob);

      Blob<Dtype> top_blob;
      Blob<Dtype> top_blob2;
      top_vec.push_back(&top_blob);
      top_vec.push_back(&top_blob2);

      LayerParameter layer_param;
      MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
      ms_mf_param->set_num_iterations(5);
      //ms_mf_param->set_spatial_filter_weight(30);
      //ms_mf_param->set_bilateral_filter_weight(15);
      ms_mf_param->set_spatial_filter_weights_str("3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3");
      ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5");

      ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
      ms_mf_param->set_theta_alpha(160);
      ms_mf_param->set_theta_beta(3);
      ms_mf_param->set_theta_gamma(3);

      // Detection potentials
      ms_mf_param->set_detection_potentials_enabled(true);
      ms_mf_param->set_detection_dat_input_dir(working_dir);
      ms_mf_param->set_detection_x_weights_str("32 32 32 32 32 32 32 32 32 1024 32 32 32 32 32 32 32 32 128 32 32");
      ms_mf_param->set_detection_y_weights_str("1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1");
      ms_mf_param->set_detection_potentials_max_score(1);
      ms_mf_param->set_detection_potentials_epsilon(0.0001);

      // Superpixel potentials
      ms_mf_param->set_ho_potentials_enabled(false);
      ms_mf_param->set_ho_dat_input_dir(working_dir);
      ms_mf_param->set_ho_w_param_str("0 0 0 0");
      ms_mf_param->set_ho_w2_param_str("0 0 0 0");
      ms_mf_param->set_ho_num_layers(4);

      short labels[height * width];
      tvg::TestUtils::get_label_map(previous_output_blob, labels);
      if (Caffe::mode() != Caffe::GPU){
        tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_ho_beforeMF_cpu.png", height, width);
      }
      else{
        tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_ho_beforeMF_gpu.png", height, width);
      }

      MultiStageMeanfieldLayer<Dtype> layer(layer_param);
      layer.SetUp(bottom_vec, top_vec);
      layer.Forward(bottom_vec, top_vec);


      tvg::TestUtils::get_label_map(top_blob, labels);
      if (Caffe::mode() != Caffe::GPU){
        tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_det_afterMF_cpu.png", height, width);
      }
      else{
        tvg::TestUtils::save_image(labels, working_dir + image_title + "_caffe_det_afterMF_gpu.png", height, width);
      }
    } // Test inference detections only

    TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradientSuperpixelsOnly) {
      typedef typename TypeParam::Dtype Dtype;
      const int n = 1, c = 4, H = 5, W = 5;

      if (sizeof(Dtype) != sizeof(double))
        return;

      LOG(INFO) << "Running Superpixel Gradient Test";

      Blob<Dtype> unary_terms_blob(n, c, H, W);
      Blob<Dtype> previous_output_blob(n, c, H, W);
      Blob<Dtype> rgb_blob(n, 3, H, W);
      Blob<Dtype> indices_bob(1, 1, 1, 1);

      tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
      tvg::TestUtils::FillAsLogProb(&previous_output_blob);
      tvg::TestUtils::FillAsRGB(&rgb_blob);
      indices_bob.mutable_cpu_data()[0] = 10;

      vector<Blob<Dtype>*> bottom_vec, top_vec;
      bottom_vec.clear();
      bottom_vec.push_back(&unary_terms_blob);
      bottom_vec.push_back(&previous_output_blob);
      bottom_vec.push_back(&rgb_blob);
      bottom_vec.push_back(&indices_bob);

      Blob<Dtype> top_blob;
      Blob<Dtype> top_blob2;
      top_vec.push_back(&top_blob);
      top_vec.push_back(&top_blob2);

      LayerParameter layer_param;
      MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
      ms_mf_param->set_num_iterations(5);
      ms_mf_param->set_bilateral_filter_weights_str("1.0 1.0 1.0 1.0 2.0");
      ms_mf_param->set_spatial_filter_weights_str("2.0 2.0 2.0 2.0 3.0");
      ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
      ms_mf_param->set_theta_alpha(5);
      ms_mf_param->set_theta_beta(2);
      ms_mf_param->set_theta_gamma(3);

      // Detection potentials
      ms_mf_param->set_detection_potentials_enabled(false);
      ms_mf_param->set_detection_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
      ms_mf_param->set_detection_x_weights_str("100 100 100 100 100");
      ms_mf_param->set_detection_y_weights_str("10 5 5 5 10");
      ms_mf_param->set_detection_potentials_max_score(5);
      ms_mf_param->set_detection_potentials_epsilon(0.001);

      // Superpixel potentials
      ms_mf_param->set_ho_potentials_enabled(true);
      ms_mf_param->set_ho_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
      ms_mf_param->set_ho_w_param_str("10 4 2 1");
      //ms_mf_param->set_ho_w2_enabled(true); // This line will cause gradient checks to fail, and output to be different, even though w2 weights are 0.
      ms_mf_param->set_ho_w2_param_str("0 0 0 0");
      ms_mf_param->set_ho_num_layers(4);

      MultiStageMeanfieldLayer<Dtype> layer(layer_param);
      layer.SetUp(bottom_vec, top_vec);


      layer.Forward(bottom_vec, top_vec);
      vector<bool> v(7, true);
      tvg::TestUtils::FillWithUpperBound(top_vec[0], 100, true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
      layer.Backward(top_vec, v, bottom_vec);

      // Parameters to this function are: "blob", print diff if true, else print data, "Title
      tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
      tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
      tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
      tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
      tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");

      GradientChecker<Dtype> checker(1e-2, 1e-3);

      // Check gradients w.r.t. unary terms
      checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

      // Check gradients w.r.t. previous outputs
      checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 1, true);
    } // end - TestGradientSuperpixelsOnly


    TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradientSuperpixels_Det_ForceCPU) {
      typedef typename TypeParam::Dtype Dtype;
      const int n = 1, c = 4, H = 5, W = 5;

      if (sizeof(Dtype) != sizeof(double))
        return;

      LOG(INFO) << "Running Gradient Test: Detections and Superpixels enabled, ForceCPU";

      Blob<Dtype> unary_terms_blob(n, c, H, W);
      Blob<Dtype> previous_output_blob(n, c, H, W);
      Blob<Dtype> rgb_blob(n, 3, H, W);
      Blob<Dtype> indices_bob(1, 1, 1, 1);

      tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
      tvg::TestUtils::FillAsLogProb(&previous_output_blob);
      tvg::TestUtils::FillAsRGB(&rgb_blob);
      indices_bob.mutable_cpu_data()[0] = 10;

      vector<Blob<Dtype>*> bottom_vec, top_vec;
      bottom_vec.clear();
      bottom_vec.push_back(&unary_terms_blob);
      bottom_vec.push_back(&previous_output_blob);
      bottom_vec.push_back(&rgb_blob);
      bottom_vec.push_back(&indices_bob);

      Blob<Dtype> top_blob;
      Blob<Dtype> top_blob2;
      top_vec.push_back(&top_blob);
      top_vec.push_back(&top_blob2);

      LayerParameter layer_param;
      MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
      ms_mf_param->set_num_iterations(5);
      ms_mf_param->set_bilateral_filter_weights_str("1.0 1.0 1.0 1.0 2.0");
      ms_mf_param->set_spatial_filter_weights_str("2.0 2.0 2.0 2.0 3.0");
      ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
      ms_mf_param->set_theta_alpha(5);
      ms_mf_param->set_theta_beta(2);
      ms_mf_param->set_theta_gamma(3);

      // Detection potentials
      ms_mf_param->set_detection_potentials_enabled(true);
      ms_mf_param->set_detection_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
      ms_mf_param->set_detection_x_weights_str("100 100 100 100 100");
      ms_mf_param->set_detection_y_weights_str("10 5 5 5 10");
      ms_mf_param->set_detection_potentials_max_score(5);
      ms_mf_param->set_detection_potentials_epsilon(0.001);

      // Superpixel potentials
      ms_mf_param->set_ho_potentials_enabled(true);
      ms_mf_param->set_ho_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
      ms_mf_param->set_ho_w_param_str("10 4 2 1");
      ms_mf_param->set_ho_w2_param_str("0 0 0 0");
      ms_mf_param->set_ho_num_layers(4);

      // Force CPU meanfield
      ms_mf_param->set_force_cpu(true);

      MultiStageMeanfieldLayer<Dtype> layer(layer_param);
      layer.SetUp(bottom_vec, top_vec);


      layer.Forward(bottom_vec, top_vec);
      vector<bool> v(7, true);
      tvg::TestUtils::FillWithUpperBound(top_vec[0], 100, true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
      layer.Backward(top_vec, v, bottom_vec);

      // Parameters to this function are: "blob", print diff if true, else print data, "Title
      tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
      tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
      tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
      tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
      tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");

      GradientChecker<Dtype> checker(1e-2, 1e-3);

      // Check gradients w.r.t. unary terms
      checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

      // Check gradients w.r.t. previous outputs
      checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 1, true);
    } // end - TestGradientSuperpixelsOnly

    TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradients_Det_ForceCPU) {
      typedef typename TypeParam::Dtype Dtype;
      const int n = 1, c = 4, H = 5, W = 5;

      if (sizeof(Dtype) != sizeof(double))
        return;

      LOG(INFO) << "Running Gradient Test: Detections enabled, ForceCPU";

      Blob<Dtype> unary_terms_blob(n, c, H, W);
      Blob<Dtype> previous_output_blob(n, c, H, W);
      Blob<Dtype> rgb_blob(n, 3, H, W);
      Blob<Dtype> indices_bob(1, 1, 1, 1);

      tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
      tvg::TestUtils::FillAsLogProb(&previous_output_blob);
      tvg::TestUtils::FillAsRGB(&rgb_blob);
      indices_bob.mutable_cpu_data()[0] = 10;

      vector<Blob<Dtype>*> bottom_vec, top_vec;
      bottom_vec.clear();
      bottom_vec.push_back(&unary_terms_blob);
      bottom_vec.push_back(&previous_output_blob);
      bottom_vec.push_back(&rgb_blob);
      bottom_vec.push_back(&indices_bob);

      Blob<Dtype> top_blob;
      Blob<Dtype> top_blob2;
      top_vec.push_back(&top_blob);
      top_vec.push_back(&top_blob2);

      LayerParameter layer_param;
      MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
      ms_mf_param->set_num_iterations(5);
      ms_mf_param->set_bilateral_filter_weights_str("1.0 1.0 1.0 1.0 2.0");
      ms_mf_param->set_spatial_filter_weights_str("2.0 2.0 2.0 2.0 3.0");
      ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
      ms_mf_param->set_theta_alpha(5);
      ms_mf_param->set_theta_beta(2);
      ms_mf_param->set_theta_gamma(3);

      // Detection potentials
      ms_mf_param->set_detection_potentials_enabled(true);
      ms_mf_param->set_detection_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
      ms_mf_param->set_detection_x_weights_str("100 100 100 100 100");
      ms_mf_param->set_detection_y_weights_str("10 5 5 5 10");
      ms_mf_param->set_detection_potentials_max_score(5);
      ms_mf_param->set_detection_potentials_epsilon(0.001);

      // Superpixel potentials
      ms_mf_param->set_ho_potentials_enabled(false);
      ms_mf_param->set_ho_dat_input_dir("caffe/src/caffe/test/test_data/mf_layer/");
      ms_mf_param->set_ho_w_param_str("10 4 2 1");
      ms_mf_param->set_ho_w2_param_str("0 0 0 0");
      ms_mf_param->set_ho_num_layers(4);

      // Force CPU meanfield
      ms_mf_param->set_force_cpu(true);

      MultiStageMeanfieldLayer<Dtype> layer(layer_param);
      layer.SetUp(bottom_vec, top_vec);


      layer.Forward(bottom_vec, top_vec);
      vector<bool> v(7, true);
      tvg::TestUtils::FillWithUpperBound(top_vec[0], 100, true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
      layer.Backward(top_vec, v, bottom_vec);

      // Parameters to this function are: "blob", print diff if true, else print data, "Title
      tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
      tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
      tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
      tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
      tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");

      GradientChecker<Dtype> checker(1e-2, 1e-3);

      // Check gradients w.r.t. unary terms
      checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

      // Check gradients w.r.t. previous outputs
      checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 1, true);
    } // end - TestGradientSuperpixelsOnly

TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradients_NoClassWeights) {
    typedef typename TypeParam::Dtype Dtype;
    const int n = 1, c = 4, H = 5, W = 5;

    if (sizeof(Dtype) != sizeof(double) || (Caffe::mode() == Caffe::CPU) )
    return;

    LOG(INFO) << "Running Gradient Test, with no class weights";

    Blob<Dtype> unary_terms_blob(n, c, H, W);
    Blob<Dtype> previous_output_blob(n, c, H, W);
    Blob<Dtype> rgb_blob(n, 3, H, W);
    Blob<Dtype> indices_bob(1, 1, 1, 1);

    tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
    tvg::TestUtils::FillAsLogProb(&previous_output_blob);
    tvg::TestUtils::FillAsRGB(&rgb_blob);
    indices_bob.mutable_cpu_data()[0] = 10;

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.clear();
    bottom_vec.push_back(&unary_terms_blob);
    bottom_vec.push_back(&previous_output_blob);
    bottom_vec.push_back(&rgb_blob);
    bottom_vec.push_back(&indices_bob);

    Blob<Dtype> top_blob;
    Blob<Dtype> top_blob2;
    top_vec.push_back(&top_blob);
    top_vec.push_back(&top_blob2);

    LayerParameter layer_param;
    MultiStageMeanfieldParameter* ms_mf_param = layer_param.mutable_multi_stage_meanfield_param();
    ms_mf_param->set_num_iterations(5);
    ms_mf_param->set_is_no_class_weights(true);
    ms_mf_param->set_bilateral_filter_weights_str("1.0");
    ms_mf_param->set_spatial_filter_weights_str("2.0 2.0");
    ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
    ms_mf_param->set_theta_alpha(5);
    ms_mf_param->set_theta_beta(2);
    ms_mf_param->set_theta_gamma(3);

    // Detection potentials
    ms_mf_param->set_detection_potentials_enabled(false);
    ms_mf_param->set_detection_dat_input_dir("does/not/matter");
    ms_mf_param->set_detection_x_weights_str("100 100 100 100 100");
    ms_mf_param->set_detection_y_weights_str("10 5 5 5 10");
    ms_mf_param->set_detection_potentials_max_score(5);
    ms_mf_param->set_detection_potentials_epsilon(0.001);

    // Superpixel potentials
    ms_mf_param->set_ho_potentials_enabled(false);
    ms_mf_param->set_ho_dat_input_dir("does/not/matter");
    ms_mf_param->set_ho_w_param_str("10 4 2 1");
    //ms_mf_param->set_ho_w2_enabled(true); // This line will cause gradient checks to fail, and output to be different, even though w2 weights are 0.
    ms_mf_param->set_ho_w2_param_str("0 0 0 0");
    ms_mf_param->set_ho_num_layers(4);

    // Force CPU meanfield
    //ms_mf_param->set_force_cpu(true);

    MultiStageMeanfieldLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);


    layer.Forward(bottom_vec, top_vec);
    vector<bool> v(7, true);
    tvg::TestUtils::FillWithUpperBound(top_vec[0], 100, true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
    layer.Backward(top_vec, v, bottom_vec);

    // Parameters to this function are: "blob", print diff if true, else print data, "Title
    tvg::TestUtils::PrintBlob(top_blob, false, "Output");
    tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
    tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
    tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
    tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
    tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
    tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");

    //if (! Caffe::mode() == Caffe::GPU) {
      unary_terms_blob.Reshape(n, c + 1, H, W);
      previous_output_blob.Reshape(n, c + 1, H, W);
      tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
      tvg::TestUtils::FillAsLogProb(&previous_output_blob);
      layer.Forward(bottom_vec, top_vec);
      layer.print_blob_sizes();
      tvg::TestUtils::FillWithUpperBound(top_vec[0], 100,
                                         true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
      layer.Backward(top_vec, v, bottom_vec);
      layer.print_blob_sizes();

      tvg::TestUtils::PrintBlob(top_blob, false, "Output");
      tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
      tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
      tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
      tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
      tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");
      layer.print_blob_sizes();

      unary_terms_blob.Reshape(n, c - 1, H, W);
      previous_output_blob.Reshape(n, c - 1, H, W);
      tvg::TestUtils::FillAsLogProb(&unary_terms_blob);
      tvg::TestUtils::FillAsLogProb(&previous_output_blob);
      layer.Forward(bottom_vec, top_vec);
      layer.print_blob_sizes();
      tvg::TestUtils::FillWithUpperBound(top_vec[0], 100,
                                         true); // Randomly fill the top blob with some gradients (which is supposed to be the gradient with respect to the loss). Max value of these gradients is 100
      layer.Backward(top_vec, v, bottom_vec);
      layer.print_blob_sizes();

      tvg::TestUtils::PrintBlob(top_blob, false, "Output");
      tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
      tvg::TestUtils::PrintBlob(unary_terms_blob, true, "Unary diff");
      tvg::TestUtils::PrintBlob(previous_output_blob, true, "Previous output blob");
      tvg::TestUtils::PrintBlob(*layer.blobs()[0], true, "Blob 0: Spatial kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[1], true, "Blob 1: Bilteral kernel gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[2], true, "Blob 2: Compatiblility matrix gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[3], true, "Blob 3: Detection X weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[4], true, "Blob 4: Detection Y weights gradients");
      tvg::TestUtils::PrintBlob(*layer.blobs()[5], true, "Blob 5: Superpixel weights 1 gradient");
      tvg::TestUtils::PrintBlob(*layer.blobs()[6], true, "Blob 6: Superpixel weights 2 gradient");
      layer.print_blob_sizes();
   //}

    GradientChecker<Dtype> checker(1e-2, 1e-3);

    // Check gradients w.r.t. unary terms
    printf("Size before gradient checking\n");
    layer.print_blob_sizes();
    checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

    // Check gradients w.r.t. previous outputs
    checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 1, true);
    } // end - TestGradientSuperpixelsOnly

}  // namespace caffe
