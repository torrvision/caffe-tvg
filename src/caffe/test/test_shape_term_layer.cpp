#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/shape_term.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_tvg_util.hpp"

#include "caffe/layers/softmax_layer.hpp"
#include <boost/lexical_cast.hpp>

namespace caffe {

  template<typename TypeParam>
  class ShapeTermLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

  protected:
    ShapeTermLayerTest() { }

    virtual void SetUp() {

    }

    virtual ~ShapeTermLayerTest() {

    }
  };

  TYPED_TEST_CASE(ShapeTermLayerTest, TestDtypesAndDevices);

#if 0
  TYPED_TEST(ShapeTermLayerTest, TestInference) {
    typedef typename TypeParam::Dtype Dtype;

    if (sizeof(Dtype) == sizeof(double)) {
      printf("Skipping test with double\n"); // Unaries are stored as floats
      return;
    }

    // parameters
    const int channels = 21, height = 500, width = 500;
    std::string filename = "2007_000663";
    std::string detection_box_path = working_dir + "detection_box/";
    std::string detection_pixel_path = working_dir + "detection_pixel/";

    //    int real_height, real_width;

    // set up blobs
    Blob<Dtype> unary_blob(1, channels, height, width);
    Blob<Dtype> indices_blob(1, 1, 1, 1);
    Blob<Dtype> input_prob_blob(unary_blob.shape());

    // initialise blobs
    tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
    indices_blob.mutable_cpu_data()[0] = 0;

    // Softmax the unaries before. We need this
    // Softmax layer configuration
    LayerParameter softmax_param;
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);

    vector<Blob<Dtype>*> softmax_bottom_vec;
    softmax_bottom_vec.clear();
    softmax_bottom_vec.push_back(&unary_blob);

    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_top_vec.clear();
    softmax_top_vec.push_back(&input_prob_blob);

    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.push_back(&input_prob_blob);
    bottom_vec.push_back(&indices_blob);

    Blob<Dtype> top_blob; // Layer will reshape it accordingly
    top_vec.push_back(&top_blob);

    // layer parameters
    LayerParameter layer_param;

    ShapeTermParameter* shape_param = layer_param.mutable_shape_term_param();
    shape_param->set_detection_box_input_dir(detection_box_path);
    shape_param->set_init_from_files(true);
    shape_param->set_init_prefix(working_dir + "shapes/shapes_");
    shape_param->set_num_files(5);

    // unary result
    short labels[height * width];
    std::cout << unary_blob.num() << ", " << unary_blob.channels() << ", " << unary_blob.height() << ", " << unary_blob.width() << std::endl;
    tvg::TestUtils::GetLabelMap(unary_blob, labels);
    tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

    // forward pass
    ShapeTermLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);
    layer.Forward(bottom_vec, top_vec);

    // do a backward as well to check it out
    tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
    vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Box Term layer at the moment
    layer.Backward(top_vec, v, bottom_vec);

    // Too much data to print here
    /*tvg::TestUtils::PrintBlob(top_blob, false, "Output");
    tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
    tvg::TestUtils::PrintBlob(input_prob_blob, true, "Derivative wrt segmentation unaries");
    tvg::TestUtils::PrintBlob(y_variables, true, "Diff wrt y variable inputs");
    tvg::TestUtils::PrintBlob(indices_blob, true, "Diff wrt index blob");*/

    // save result
    tvg::TestUtils::GetLabelMap(top_blob, labels);

    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_top_cpu.csv" );
    }
    else{
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_top_gpu.csv" );
    }

    // Now do another image
    indices_blob.mutable_cpu_data()[0] = 1;
    layer.Forward(bottom_vec, top_vec);

    tvg::TestUtils::GetLabelMap(top_blob, labels);
    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_iter2_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_iter2_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_iter2_cpu.csv" );
    }
    else{
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_iter2_gpu.csv" );
    }

  } // end - TestInference()*/
#endif

#if 0
  TYPED_TEST(ShapeTermLayerTest, TestInferenceEmptyDet) {
    typedef typename TypeParam::Dtype Dtype;

    if (sizeof(Dtype) == sizeof(double)) {
      printf("Skipping test with double\n"); // Unaries are stored as floats
      return;
    }

    // parameters
    const int channels = 21, height = 500, width = 500;
    std::string filename = "2007_000663";
    std::string detection_box_path = working_dir + "detection_box/";
    std::string detection_pixel_path = working_dir + "detection_pixel/";

    //    int real_height, real_width;

    // set up blobs
    Blob<Dtype> unary_blob(1, channels, height, width);
    Blob<Dtype> indices_blob(1, 1, 1, 1);
    Blob<Dtype> input_prob_blob(unary_blob.shape());

    // initialise blobs
    tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
    indices_blob.mutable_cpu_data()[0] = 2;

    // Softmax the unaries before. We need this
    // Softmax layer configuration
    LayerParameter softmax_param;
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);

    vector<Blob<Dtype>*> softmax_bottom_vec;
    softmax_bottom_vec.clear();
    softmax_bottom_vec.push_back(&unary_blob);

    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_top_vec.clear();
    softmax_top_vec.push_back(&input_prob_blob);

    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.push_back(&input_prob_blob);
    bottom_vec.push_back(&indices_blob);

    Blob<Dtype> top_blob; // Layer will reshape it accordingly
    top_vec.push_back(&top_blob);

    // layer parameters
    LayerParameter layer_param;

    ShapeTermParameter* shape_param = layer_param.mutable_shape_term_param();
    shape_param->set_detection_box_input_dir(detection_box_path);
    shape_param->set_init_from_files(true);
    shape_param->set_init_prefix(working_dir + "shapes/shapes_");
    shape_param->set_num_files(5);

    // unary result
    short labels[height * width];
    std::cout << unary_blob.num() << ", " << unary_blob.channels() << ", " << unary_blob.height() << ", " << unary_blob.width() << std::endl;
    tvg::TestUtils::GetLabelMap(unary_blob, labels);
    tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

    // forward pass
    ShapeTermLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);
    layer.Forward(bottom_vec, top_vec);

    // do a backward as well to check it out
    tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
    vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Box Term layer at the moment
    layer.Backward(top_vec, v, bottom_vec);

    // Too much data to print here
    /*tvg::TestUtils::PrintBlob(top_blob, false, "Output");
    tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
    tvg::TestUtils::PrintBlob(input_prob_blob, true, "Derivative wrt segmentation unaries");
    tvg::TestUtils::PrintBlob(y_variables, true, "Diff wrt y variable inputs");
    tvg::TestUtils::PrintBlob(indices_blob, true, "Diff wrt index blob");*/

    // save result
    tvg::TestUtils::GetLabelMap(top_blob, labels);

    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_ed_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_ed_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_ed_top_cpu.csv" );
    }
    else{
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_ed_top_gpu.csv" );
    }

    // Now do another image
    indices_blob.mutable_cpu_data()[0] = 1;
    layer.Forward(bottom_vec, top_vec);

    tvg::TestUtils::GetLabelMap(top_blob, labels);
    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_iter2_ed_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_iter2_ed_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_iter2_ed_cpu.csv" );
    }
    else{
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_iter2_ed_gpu.csv" );
    }

  } // end - TestInferenceEmptyDet()*/
#endif

  std::string working_dir = "caffe/src/caffe/test/test_data/shape_term/";

#if 0
  TYPED_TEST(ShapeTermLayerTest, TestInferenceAgain) {
    typedef typename TypeParam::Dtype Dtype;

    if (sizeof(Dtype) == sizeof(double)) {
      printf("Skipping test with double\n"); // Unaries are stored as floats
      return;
    }

    // parameters
    const int channels = 21, height = 500, width = 500;

    std::string filename = "2007_001430";
    std::string detection_box_path = working_dir + "detection_box/";
    std::string detection_pixel_path = working_dir + "detection_pixel/";

    //    int real_height, real_width;

    // set up blobs
    Blob<Dtype> unary_blob(1, channels, height, width);
    Blob<Dtype> indices_blob(1, 1, 1, 1);
    Blob<Dtype> input_prob_blob(unary_blob.shape());

    // initialise blobs
    tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
    indices_blob.mutable_cpu_data()[0] = 71430;

    // Softmax the unaries before. We need this
    // Softmax layer configuration
    LayerParameter softmax_param;
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);

    vector<Blob<Dtype>*> softmax_bottom_vec;
    softmax_bottom_vec.clear();
    softmax_bottom_vec.push_back(&unary_blob);

    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_top_vec.clear();
    softmax_top_vec.push_back(&input_prob_blob);

    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.push_back(&input_prob_blob);
    bottom_vec.push_back(&indices_blob);

    Blob<Dtype> top_blob; // Layer will reshape it accordingly
    top_vec.push_back(&top_blob);

    // layer parameters
    LayerParameter layer_param;

    ShapeTermParameter* shape_param = layer_param.mutable_shape_term_param();
    shape_param->set_detection_box_input_dir(detection_box_path);
    shape_param->set_init_from_files(true);
    shape_param->set_init_prefix(working_dir + "shapes/shapes_");
    shape_param->set_num_files(5);

    // unary result
    short labels[height * width];
    std::cout << unary_blob.num() << ", " << unary_blob.channels() << ", " << unary_blob.height() << ", " << unary_blob.width() << std::endl;
    tvg::TestUtils::GetLabelMap(unary_blob, labels);
    tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

    // forward pass
    ShapeTermLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);
    layer.Forward(bottom_vec, top_vec);

    // do a backward as well to check it out
    tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
    vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Shape Term layer at the moment
    layer.Backward(top_vec, v, bottom_vec);

    // Too much data to print here
    /*tvg::TestUtils::PrintBlob(top_blob, false, "Output");
    tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
    tvg::TestUtils::PrintBlob(input_prob_blob, true, "Derivative wrt segmentation unaries");
    tvg::TestUtils::PrintBlob(y_variables, true, "Diff wrt y variable inputs");
    tvg::TestUtils::PrintBlob(indices_blob, true, "Diff wrt index blob");*/

    // save result
    tvg::TestUtils::GetLabelMap(top_blob, labels);

    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_top_cpu.csv" );
      tvg::CommonUtils::save_blob_to_file(input_prob_blob, working_dir+ filename + "_shape_term_gradient_cpu.csv", true);
    }
    else{
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_top_gpu.csv" );
      tvg::CommonUtils::save_blob_to_file(input_prob_blob, working_dir+ filename + "_shape_term_gradient_gpu.csv", true);
    }

    // Now the gradients of the parameters
    std::vector< shared_ptr<Blob<Dtype> > > & param_blobs = layer.blobs();
    for (int i = 0; i < param_blobs.size(); ++i){
      std::string suffix = "cpu.csv";
      if (Caffe::mode() == Caffe::GPU){ suffix = "gpu.csv"; }

      tvg::CommonUtils::save_blob_to_file(*param_blobs[i], working_dir + "TestInferenceAgain_Param_" + boost::lexical_cast<std::string>(i) + "_data_" + suffix, false);
      tvg::CommonUtils::save_blob_to_file(*param_blobs[i], working_dir + "TestInferenceAgain_Param_" + boost::lexical_cast<std::string>(i) + "_diff_" + suffix, true);

    }


    // Now do another image
    /*indices_blob.mutable_cpu_data()[0] = 1;
    layer.Forward(bottom_vec, top_vec);

    tvg::TestUtils::GetLabelMap(top_blob, labels);
    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_iter2_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_iter2_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_iter2_cpu.csv" );
    }
    else{
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_iter2_gpu.csv" );
    }*/

  } // end - TestInferenceAgain()*/
#endif

  TYPED_TEST(ShapeTermLayerTest, TestInferenceAndGradient){
    typedef typename TypeParam::Dtype Dtype;

    if (sizeof(Dtype) == sizeof(double)) {
      printf("Skipping test with double\n"); // Unaries are stored as floats
      return;
    }

    // parameters
    const int channels = 21, height = 11, width = 11;

    std::string filename = "2007_000663_mini";
    std::string detection_box_path = working_dir + "detection_box/";
    std::string detection_pixel_path = working_dir + "detection_pixel/";

    // set up blobs
    Blob<Dtype> unary_blob(1, channels, height, width);
    Blob<Dtype> indices_blob(1, 1, 1, 1);
    Blob<Dtype> input_prob_blob(unary_blob.shape());

    // initialise blobs
    tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
    indices_blob.mutable_cpu_data()[0] = 1;

    // Softmax the unaries before. We need this
    // Softmax layer configuration
    LayerParameter softmax_param;
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);

    vector<Blob<Dtype>*> softmax_bottom_vec;
    softmax_bottom_vec.clear();
    softmax_bottom_vec.push_back(&unary_blob);

    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_top_vec.clear();
    softmax_top_vec.push_back(&input_prob_blob);

    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.push_back(&input_prob_blob);
    bottom_vec.push_back(&indices_blob);

    Blob<Dtype> top_blob; // Layer will reshape it accordingly
    top_vec.push_back(&top_blob);

    // layer parameters
    LayerParameter layer_param;

    ShapeTermParameter* shape_param = layer_param.mutable_shape_term_param();
    shape_param->set_detection_box_input_dir(detection_box_path);
    shape_param->set_init_from_files(true);
    shape_param->set_init_prefix(working_dir + "shapes/shapes_debug_");
    shape_param->set_num_files(2); // gradient checking with all 5 takes too long

    // unary result
    short labels[height * width];
    std::cout << unary_blob.num() << ", " << unary_blob.channels() << ", " << unary_blob.height() << ", " << unary_blob.width() << std::endl;
    tvg::TestUtils::GetLabelMap(unary_blob, labels);
    tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

    // forward pass
    ShapeTermLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);
    layer.Forward(bottom_vec, top_vec);

    // do a backward as well to check it out
    tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
    vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Shape Term layer at the moment
    layer.Backward(top_vec, v, bottom_vec);

    // save result
    tvg::TestUtils::GetLabelMap(top_blob, labels);

    if (Caffe::mode() != Caffe::GPU){
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_gradtest_cpu.png", height, width);
    }
    else{
      tvg::TestUtils::save_image(labels, working_dir + filename + "_shape_term_gradtest_gpu.png", height, width);
    }

    if (Caffe::mode() != Caffe::GPU){
      tvg::CommonUtils::save_blob_to_file(input_prob_blob, working_dir+ filename + "_shape_term_top_diff_cpu.csv", true);
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_top_data_cpu.csv", false);
    }
    else{
      tvg::CommonUtils::save_blob_to_file(input_prob_blob, working_dir+ filename + "_shape_term_top_diff_gpu.csv", true);
      tvg::CommonUtils::save_blob_to_file(top_blob, working_dir+ filename + "_shape_term_top_data_gpu.csv", false);
    }

    std::cout << "Checking gradients now" << std::endl;
    GradientChecker<Dtype> checker(1e-4, 1e-3);

    checker.CheckGradientExhaustiveReverse(&layer, bottom_vec, top_vec, 0, true);
  } // end - TestInferenceAndGradientFull

#if 0
  TYPED_TEST(ShapeTermLayerTest, TestMemoryUsage) {
    typedef typename TypeParam::Dtype Dtype;

    if (sizeof(Dtype) == sizeof(double)) {
      printf("Skipping test with double\n"); // Unaries are stored as floats
      return;
    }

    // parameters
    const int channels = 21, height = 500, width = 500;

    std::string filename = "2007_001430";
    std::string detection_box_path = working_dir + "detection_box/";
    std::string detection_pixel_path = working_dir + "detection_pixel/";

    //    int real_height, real_width;

    // set up blobs
    Blob<Dtype> unary_blob(1, channels, height, width);
    Blob<Dtype> indices_blob(1, 1, 1, 1);
    Blob<Dtype> input_prob_blob(unary_blob.shape());

    // initialise blobs
    tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
    indices_blob.mutable_cpu_data()[0] = 71430;

    // Softmax the unaries before. We need this
    // Softmax layer configuration
    LayerParameter softmax_param;
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);

    vector<Blob<Dtype>*> softmax_bottom_vec;
    softmax_bottom_vec.clear();
    softmax_bottom_vec.push_back(&unary_blob);

    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_top_vec.clear();
    softmax_top_vec.push_back(&input_prob_blob);

    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);

    vector<Blob<Dtype>*> bottom_vec, top_vec;
    bottom_vec.push_back(&input_prob_blob);
    bottom_vec.push_back(&indices_blob);

    Blob<Dtype> top_blob; // Layer will reshape it accordingly
    top_vec.push_back(&top_blob);

    // layer parameters
    LayerParameter layer_param;

    ShapeTermParameter* shape_param = layer_param.mutable_shape_term_param();
    shape_param->set_detection_box_input_dir(detection_box_path);
    shape_param->set_init_from_files(true);
    shape_param->set_init_prefix(working_dir + "shapes/shapes_");
    shape_param->set_num_files(5);

    // unary result
    short labels[height * width];
    std::cout << unary_blob.num() << ", " << unary_blob.channels() << ", " << unary_blob.height() << ", " << unary_blob.width() << std::endl;
    tvg::TestUtils::GetLabelMap(unary_blob, labels);
    tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

    // forward pass
    ShapeTermLayer<Dtype> layer(layer_param);
    layer.SetUp(bottom_vec, top_vec);

    for (int i = 0; i < 1000 ; ++i) {
      layer.Forward(bottom_vec, top_vec);

      // do a backward as well to check it out
      tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
      vector<bool> v(1,
                     true); // Doesn't actually matter to us. Don't use "propogate down" in our Box Term layer at the moment
      layer.Backward(top_vec, v, bottom_vec);

      std::cout << i << std::endl;

      if (i % 10 == 0){
        std::cout << i << " ";
      }
      if (i % 100 == 0){
        std::cout << std::endl;
      }
    }

  } // end - TestMemoryUsage()*/
#endif

}