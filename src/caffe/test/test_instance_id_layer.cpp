#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/instance_id_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_tvg_util.hpp"

#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

    template <typename TypeParam>
    class InstanceIDLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;

    protected:
        InstanceIDLayerTest() {}

        virtual void SetUp() {

        }

        virtual ~InstanceIDLayerTest() {

        }
    };

    TYPED_TEST_CASE(InstanceIDLayerTest, TestDtypesAndDevices);

    TYPED_TEST(InstanceIDLayerTest, TestInference) {
        typedef typename TypeParam::Dtype Dtype;

        if (sizeof(Dtype) == sizeof(double)) {
            printf("Skipping test with double\n"); // Unaries are stored as floats
            return;
        }

        // parameters
        const int channels = 21, height = 500, width = 500;
        std::string working_dir = "caffe/src/caffe/test/test_data/instance_id_layer/";
        std::string filename = "2007_000663";
        std::string detection_box_path = working_dir + "detection_box/";
        std::string detection_pixel_path = working_dir + "detection_pixel/";

        // set up blobs
        Blob<Dtype> unary_blob(1, channels, height, width);
        Blob<Dtype> y_variables(1, 1, 1, 2*10);
        Blob<Dtype> indices_blob(1, 1, 1, 1);
        Blob<Dtype> input_prob_blob(unary_blob.shape());

        // initialise blobs
        tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
        caffe_set(y_variables.count(), Dtype(0), y_variables.mutable_cpu_data());
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
        bottom_vec.push_back(&y_variables);
        bottom_vec.push_back(&indices_blob);

        Blob<Dtype> top_blob;
        Blob<Dtype> y_variables_out; // Instance ID layer will reshape it appropriately
        top_vec.push_back(&top_blob);
        top_vec.push_back(&y_variables_out);

        // layer parameters
        LayerParameter layer_param;
        InstanceIDParameter* instance_param = layer_param.mutable_instance_id_param();

        instance_param->set_detection_box_input_dir(detection_box_path);
        instance_param->set_detection_pixel_input_dir(detection_pixel_path);

        // unary result
        short labels[height * width];
        tvg::TestUtils::GetLabelMap(unary_blob, labels);
        tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

        // forward pass
        InstanceIDLayer<Dtype> layer(layer_param);
        layer.SetUp(bottom_vec, top_vec);
        layer.Forward(bottom_vec, top_vec);

        // do a backward as well to check it out
        tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
        vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Instance ID layer at the moment
        layer.Backward(top_vec, v, bottom_vec);

        // save result
        tvg::TestUtils::GetLabelMap(top_blob, labels);
        tvg::TestUtils::save_image(labels, working_dir + filename + "_instance_id.png", height, width);

        tvg::CommonUtils::save_blob_to_file(top_blob, working_dir + filename + "_top.csv" );

    } // end - TestInference()*/

    TYPED_TEST(InstanceIDLayerTest, TestGradientSimpler) {
        typedef typename TypeParam::Dtype Dtype;

        // parameters
        const int channels = 4, height = 2, width = 2;
        std::string working_dir = "caffe/src/caffe/test/test_data/instance_id_layer/";
        std::string filename = "2007_000663_mini";
        std::string detection_box_path = working_dir + "detection_box/";
        std::string detection_pixel_path = working_dir + "detection_pixel/";

        //    int real_height, real_width;

        // set up blobs
        Blob<Dtype> unary_blob(1, channels, height, width);
        Blob<Dtype> y_variables(1, 1, 1, 2*10);
        Blob<Dtype> indices_blob(1, 1, 1, 1);

        // initialise blobs
        //tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
        Dtype* unary_blob_data = unary_blob.mutable_cpu_data();
        int c = 0;
        unary_blob_data[c] = 0.9; ++c; unary_blob_data[c] = 0.2; ++c; unary_blob_data[c] = 0.2; ++c; unary_blob_data[c] = 0.1; ++c;
        unary_blob_data[c] = 0.04; ++c; unary_blob_data[c] = 0.7; ++c; unary_blob_data[c] = 0.05; ++c; unary_blob_data[c] = 0.4; ++c;
        unary_blob_data[c] = 0.04; ++c; unary_blob_data[c] = 0.05; ++c; unary_blob_data[c] = 0.65; ++c; unary_blob_data[c] = 0.3; ++c;
        unary_blob_data[c] = 0.02; ++c; unary_blob_data[c] = 0.05; ++c; unary_blob_data[c] = 0.1; ++c; unary_blob_data[c] = 0.2; ++c;

        caffe_set(y_variables.count(), Dtype(0), y_variables.mutable_cpu_data());
        indices_blob.mutable_cpu_data()[0] = 50;

        vector<Blob<Dtype>*> bottom_vec, top_vec;
        bottom_vec.push_back(&unary_blob);
        bottom_vec.push_back(&y_variables);
        bottom_vec.push_back(&indices_blob);

        Blob<Dtype> top_blob;
        Blob<Dtype> y_variables_out; // Instance ID layer will reshape it appropriately
        top_vec.push_back(&top_blob);
        top_vec.push_back(&y_variables_out);

        // layer parameters
        LayerParameter layer_param;
        InstanceIDParameter* instance_param = layer_param.mutable_instance_id_param();

        instance_param->set_detection_box_input_dir(detection_box_path);
        instance_param->set_detection_pixel_input_dir(detection_pixel_path);

        // unary result
        short labels[height * width];
        tvg::TestUtils::GetLabelMap(unary_blob, labels);
        tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

        // forward pass
        InstanceIDLayer<Dtype> layer(layer_param);
        layer.SetUp(bottom_vec, top_vec);
        layer.Forward(bottom_vec, top_vec);

        // do a backward as well to check it out
        tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
        tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
        vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Instance ID layer at the moment
        layer.Backward(top_vec, v, bottom_vec);

        tvg::TestUtils::PrintBlob(top_blob, false, "Output");
        tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
        tvg::TestUtils::PrintBlob(unary_blob, true, "Derivative wrt segmentation unaries");
        tvg::TestUtils::PrintBlob(*top_vec[1], true, "Diff wrt y variable inputs");
        tvg::TestUtils::PrintBlob(indices_blob, true, "Diff wrt index blob");

        printf("Checking gradients now\n");
        GradientChecker<Dtype> checker(1e-4, 1e-3);

        // Check gradients w.r.t. segmentation unaries
        //checker.CheckGradientExhaustiveReverse(&layer, bottom_vec, top_vec, 0);
        checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0);

    } // end - TestGradientSimpler()*/

    TYPED_TEST(InstanceIDLayerTest, TestInferenceAndGradient) {
        typedef typename TypeParam::Dtype Dtype;

        if (sizeof(Dtype) == sizeof(double)) {
            printf("Skipping test with double\n"); // Unaries are stored as floats
            return;
        }

        // parameters
        const int channels = 21, height = 11, width = 11;
        std::string working_dir = "caffe/src/caffe/test/test_data/instance_id_layer/";
        std::string filename = "2007_000663_mini";
        std::string detection_box_path = working_dir + "detection_box/";
        std::string detection_pixel_path = working_dir + "detection_pixel/";

        //    int real_height, real_width;

        // set up blobs
        Blob<Dtype> unary_blob(1, channels, height, width);
        Blob<Dtype> y_variables(1, 1, 1, 2*10);
        Blob<Dtype> indices_blob(1, 1, 1, 1);
        Blob<Dtype> input_prob_blob(unary_blob.shape());

        // initialise blobs
        tvg::TestUtils::FillFromDat(&unary_blob, working_dir + filename + ".caffedat");
        caffe_set(y_variables.count(), Dtype(0), y_variables.mutable_cpu_data());
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
        bottom_vec.push_back(&y_variables);
        bottom_vec.push_back(&indices_blob);

        Blob<Dtype> top_blob;
        Blob<Dtype> y_variables_out; // Instance ID layer will reshape it appropriately
        top_vec.push_back(&top_blob);
        top_vec.push_back(&y_variables_out);

        // layer parameters
        LayerParameter layer_param;
        InstanceIDParameter* instance_param = layer_param.mutable_instance_id_param();

        instance_param->set_detection_box_input_dir(detection_box_path);
        instance_param->set_detection_pixel_input_dir(detection_pixel_path);

        // unary result
        short labels[height * width];
        tvg::TestUtils::GetLabelMap(unary_blob, labels);
        tvg::TestUtils::save_image(labels, working_dir + filename + "_unary.png", height, width);

        // forward pass
        InstanceIDLayer<Dtype> layer(layer_param);
        layer.SetUp(bottom_vec, top_vec);
        layer.Forward(bottom_vec, top_vec);

        // do a backward as well to check it out
        tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
        tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
        vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Instance ID layer at the moment
        layer.Backward(top_vec, v, bottom_vec);

        tvg::TestUtils::PrintBlob(top_blob, false, "Output");
        tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
        tvg::TestUtils::PrintBlob(input_prob_blob, true, "Derivative wrt segmentation unaries");
        tvg::TestUtils::PrintBlob(*top_vec[1], true, "Diff wrt y variable inputs");
        tvg::TestUtils::PrintBlob(indices_blob, true, "Diff wrt index blob");

        // save result
        tvg::TestUtils::GetLabelMap(top_blob, labels);
        tvg::TestUtils::save_image(labels, working_dir + filename + "_instance_id.png", height, width);

        tvg::CommonUtils::save_blob_to_file(top_blob, working_dir + filename + "_top.csv" );

        printf("Checking gradients now\n");
        GradientChecker<Dtype> checker(1e-4, 1e-3);

        // Check gradients w.r.t. segmentation unaries
        checker.CheckGradientExhaustiveReverse(&layer, bottom_vec, top_vec, 0);

    } // end - TestInferenceAndGradient()*/



}  // namespace caffe
