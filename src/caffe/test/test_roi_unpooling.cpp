#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/roi_unpooling.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_tvg_util.hpp"

namespace caffe {

    template <typename TypeParam>
    class ROIUnpoolingLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;

    protected:
      ROIUnpoolingLayerTest() {}

        virtual void SetUp() {

        }

        virtual ~ROIUnpoolingLayerTest() {

        }
    };

    TYPED_TEST_CASE(ROIUnpoolingLayerTest, TestDtypesAndDevices);

    TYPED_TEST(ROIUnpoolingLayerTest, TestInferenceAndGradient) {
        typedef typename TypeParam::Dtype Dtype;

        if (sizeof(Dtype) == sizeof(double)) {
          printf("Skipping test with double\n"); // Unaries are stored as floats
          return;
        }

        // parameters
        const int channels = 1, height = 5, width = 5;

        //    int real_height, real_width;

        // set up blobs
        Blob<Dtype> unary_blob(1, channels, height, width);
        Blob<Dtype> size_blob(1,1,1,4);
        Blob<Dtype> top_blob;

        // initialise blobs
        tvg::TestUtils::FillWithUpperBound(&unary_blob, 10.0f);
        size_blob.mutable_cpu_data()[0] = 12;
        size_blob.mutable_cpu_data()[1] = 10;
        size_blob.mutable_cpu_data()[2] = 3;
        size_blob.mutable_cpu_data()[3] = 4; //6; // Should throw an error

        vector<Blob<Dtype>*> bottom_vec, top_vec;
        bottom_vec.push_back(&unary_blob);
        bottom_vec.push_back(&size_blob);

        top_vec.push_back(&top_blob); // layer will reshape it properly

        // layer parameters
        LayerParameter layer_param;

        // unary result

        // forward pass
        ROIUnpoolingLayer<Dtype> layer(layer_param);
        layer.SetUp(bottom_vec, top_vec);
        layer.Forward(bottom_vec, top_vec);

        // do a backward as well to check it out
        tvg::TestUtils::FillAsConstant(&top_blob, Dtype(1.), true);
        tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
        vector<bool> v(1, true); // Doesn't actually matter to us. Don't use "propogate down" in our Instance ID layer at the moment
        layer.Backward(top_vec, v, bottom_vec);

        tvg::TestUtils::PrintBlob(unary_blob, false, "Input");
        tvg::TestUtils::PrintBlob(top_blob, false, "Output");
        tvg::TestUtils::PrintBlob(top_blob, true, "Input diff");
        tvg::TestUtils::PrintBlob(unary_blob, true, "Derivative wrt input");

        printf("Checking gradients now\n");
        GradientChecker<Dtype> checker(1e-3, 1e-4);

        // Check gradients w.r.t. segmentation unaries
        // Only chech gradient with respect to bottom[0] - the unaries. Not bottom[1] - the index
        // Also, set the "verbose" option to true
        checker.CheckGradientExhaustive(&layer, bottom_vec, top_vec, 0, true);

    } // end - TestInferenceAndGradient()*/


}  // namespace caffe
