#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <math.h>

#include "caffe/layers/image_label_data_index_det_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <boost/lexical_cast.hpp>

namespace {

cv::Mat PadImage(cv::Mat &image, int min_size, double value = -1, bool pad_centre = true) {
  if (image.rows >= min_size && image.cols >= min_size) {
    return image;
  }
  int top, bottom, left, right;
  top = bottom = left = right = 0;
  if (image.rows < min_size) {
    top = (min_size - image.rows) / 2;
    bottom = min_size - image.rows - top;

    if (!pad_centre){
      top = 0;
      bottom = min_size - image.rows;
    }
  }

  if (image.cols < min_size) {
    left = (min_size - image.cols) / 2;
    right = min_size - image.cols - left;

    if (!pad_centre){
      right = 0;
      left = min_size - image.cols;
    }
  }
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h,
                          double value = -1) {
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

void ApplyHSVNoise (cv::Mat &image, const int h_noise, const int s_noise, const int v_noise, std::mt19937 * rng){

  cv::cvtColor(image, image, CV_BGR2HSV);

  int h_delta = std::uniform_int_distribution<int>( -h_noise, h_noise)(*rng);
  int s_delta = std::uniform_int_distribution<int>( -s_noise, s_noise)(*rng);
  int v_delta = std::uniform_int_distribution<int>( -v_noise, v_noise)(*rng);

  for (int y = 0; y < image.rows; ++y){
    for (int x = 0; x < image.cols; ++x){

      int cur1 = image.at<cv::Vec3b>(cv::Point(x,y))[0];
      int cur2 = image.at<cv::Vec3b>(cv::Point(x,y))[1];
      int cur3 = image.at<cv::Vec3b>(cv::Point(x,y))[2];
      cur1 += h_delta;
      cur2 += s_delta;
      cur3 += v_delta;
      // the H output of cvtColor(CV_BGR2HSV) for an 8-bit image is between 0 and 180
      if(cur1 < 0) cur1 += 180; else if(cur1 > 180) cur1 += -180;
      if(cur2 < 0) cur2= 0; else if(cur2 > 255) cur2 = 255;
      if(cur3 < 0) cur3= 0; else if(cur3 > 255) cur3 = 255;

      image.at<cv::Vec3b>(cv::Point(x,y))[0] = cur1;
      image.at<cv::Vec3b>(cv::Point(x,y))[1] = cur2;
      image.at<cv::Vec3b>(cv::Point(x,y))[2] = cur3;

    }
  }

  cv::cvtColor(image, image, CV_HSV2BGR);
}

template <typename Dtype>
void GetLabelSlice(const Dtype *labels, int rows, int cols,
                   const caffe::Slice &label_slice, Dtype *slice_data, int channels) {
  const Dtype* labels_orig = labels;
  for (int c = 0; c < channels; ++c) {
    labels = labels_orig + (c * rows*cols);
    labels += label_slice.offset(0) * cols;
    for (int h = 0; h < label_slice.dim(0); ++h) {
      labels += label_slice.offset(1);
      for (int w = 0; w < label_slice.dim(1); ++w) {
        slice_data[w] = labels[w * label_slice.stride(1)];
      }
      labels += cols * label_slice.stride(0) - label_slice.offset(1);
      slice_data += label_slice.dim(1);
    }
  }
}

} // namespace

namespace caffe {

template <typename Dtype>
ImageLabelDataIndexDetLayer<Dtype>::ImageLabelDataIndexDetLayer(
    const LayerParameter &param) : BasePrefetchingDataIndexDetectionLayer<Dtype>(param) {
  std::random_device rand_dev;
  rng_ = new std::mt19937(rand_dev());
}

template <typename Dtype>
ImageLabelDataIndexDetLayer<Dtype>::~ImageLabelDataIndexDetLayer() {
  this->StopInternalThread();
  delete rng_;
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  auto &data_param = this->layer_param_.image_label_data_param();
  string data_dir = data_param.data_dir();
  string image_dir = data_param.image_dir();
  string label_dir = data_param.label_dir();
  string box_dir = data_param.box_dir();

  if (image_dir == "" && label_dir == "" && data_dir != "") {
    image_dir = data_dir;
    label_dir = data_dir;
  }

  // Read the file with filenames and labels
  const string& image_list_path =
      this->layer_param_.image_label_data_param().image_list_path();
  LOG(INFO) << "Opening image list " << image_list_path;
  std::ifstream infile(image_list_path.c_str());
  if (!infile){
    LOG(FATAL) << "Image list: " << image_list_path << " does not exist";
  }

  string filename;
  while (infile >> filename) {
    if (filename.size() > 0 && filename[0] != '#'){
      image_lines_.push_back(filename);
    }
  }

  const string& label_list_path =
      this->layer_param_.image_label_data_param().label_list_path();
  LOG(INFO) << "Opening label list " << label_list_path;
  std::ifstream in_label(label_list_path.c_str());
  if (!in_label){
    LOG(FATAL) << "Label list: " << label_list_path << " does not exist";
  }

  while (in_label >> filename) {
    if (filename.size() > 0 && filename[0] != '#'){
      label_lines_.push_back(filename);
    }
  }

  LOG(INFO) << "A total of " << image_lines_.size() << " images.";
  LOG(INFO) << "A total of " << label_lines_.size() << " labels.";
  CHECK_EQ(image_lines_.size(), label_lines_.size());

  order_.resize(image_lines_.size());
  for (int i = 0; i < order_.size(); ++i) {
    order_[i] = i;
  }

  if (this->layer_param_.image_label_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_label_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.image_label_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(image_lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.

  cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
  CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }
  cv_img = PadImage(cv_img, crop_size);

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(data_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_label_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  data_shape[0] = batch_size;
  top[0]->Reshape(data_shape);

  /*
  * HSV noise
  */
  hsv_noise_ = this->layer_param_.image_label_data_param().hsv_noise();
  h_noise_ = this->layer_param_.image_label_data_param().h_noise();
  s_noise_ = this->layer_param_.image_label_data_param().s_noise();
  v_noise_ = this->layer_param_.image_label_data_param().v_noise();

  /*
  *pad centre or not
  */
  pad_centre_ = this->layer_param_.image_label_data_param().pad_centre();

  /*
   * Rotation or not
   */
  random_rotate_ = this->layer_param_.image_label_data_param().random_rotate();
  max_rotation_angle_ = this->layer_param_.image_label_data_param().max_rotation_angle();
  min_rotation_angle_ = this->layer_param_.image_label_data_param().min_rotation_angle();

  /*
   * Gaussian blur or not
   */
  random_gaussian_blur_ = this->layer_param_.image_label_data_param().random_gaussian_blur();

    /*
   * Random box perturbation or not
   */
  random_box_perturb_ = this->layer_param_.image_label_data_param().random_box_perturb();

  /*
   * label
   */
  auto &label_slice = this->layer_param_.image_label_data_param().label_slice();
  label_margin_h_ = label_slice.offset(0);
  label_margin_w_ = label_slice.offset(1);
  LOG(INFO) << "Assuming image and label map sizes are the same";
  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = /*1*/ this->layer_param_.image_label_data_param().num_channels();
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  top[1]->Reshape(label_shape);

  /*
   * Index
   */
  vector<int> index_shape(4);
  index_shape[0] = batch_size; index_shape[1] = 1; index_shape[2] = 1; index_shape[3] = 1;
  top[2]->Reshape(index_shape);

  /*
   * Detection
   * Shape is N x D x 1 x 6
   * N is the batch dimension
   * D is the number of detections
   * 6 because it is [detection class, x0, y0, x1, y1, detection score] (x0,y0) is the top left corner, (x1,y1) is the bottom right corner
   */
  vector<int> detection_shape(4);
  detection_shape[0] = batch_size;
  detection_shape[1] = /*1*/ this->layer_param_.image_label_data_param().init_num_dets();
  detection_shape[2] = 1;
  detection_shape[3] = tvg::Detection::num_per_detection;
  if (top.size() >= 4){
    top[3]->Reshape(detection_shape);
  }

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(data_shape);
    this->prefetch_[i].label_.Reshape(label_shape);
    this->prefetch_[i].index_.Reshape(index_shape);
    this->prefetch_[i].detection_.Reshape(detection_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels() << "," << top[1]->height() << ","
  << top[1]->width();
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::ShuffleImages() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
//  LOG(FATAL) <<
//      "ImageLabelDataLayer<Dtype>::ShuffleImages() is not implemented";
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());

  shuffle(order_.begin(), order_.end(), prefetch_rng);
  vector<std::string> new_image_lines(image_lines_.size());
  vector<std::string> new_label_lines(label_lines_.size());

  for (int i = 0; i < order_.size(); ++i) {
    new_image_lines[i] = image_lines_[order_[i]];
    new_label_lines[i] = label_lines_[order_[i]];
  }

  swap(image_lines_, new_image_lines);
  swap(label_lines_, new_label_lines);
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::SampleScale(cv::Mat *image, cv::Mat *label, Blob<Dtype>* detection_blob) {
  ImageLabelDataParameter data_param =
      this->layer_param_.image_label_data_param();
  if (!data_param.rand_scale()) return;
  double scale = std::uniform_real_distribution<double>(
      data_param.min_scale(), data_param.max_scale())(*rng_);
  cv::Size zero_size(0, 0);

  cv::resize(*label, *label, cv::Size(0, 0),
             scale, scale, cv::INTER_NEAREST);

  if (scale > 1) {
    cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_CUBIC);
  } else {
    cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_AREA);
  }

  if (detection_blob != NULL){
    Dtype *detection_data = detection_blob->mutable_cpu_data();
    int step = detection_blob->width();
    for (int index = 0; index < detection_blob->count(); index += step) {
      detection_data[index + 1] = floor(detection_data[index + 1] * scale);
      detection_data[index + 2] = floor(detection_data[index + 2] * scale);
      detection_data[index + 3] = ceil(detection_data[index + 3] * scale);
      detection_data[index + 4] = ceil(detection_data[index + 4] * scale);
    }
  }

}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::PadBbox(cv::Mat &image, int min_size, int margin_w, int margin_h, bool pad_centre, Blob<Dtype>* detection_blob) {
  if (image.rows >= min_size && image.cols >= min_size) {
    return;
  }
  int top, left;
  top = left = 0;
  if (image.rows < min_size) {
    top = (min_size - image.rows) / 2;

    if (!pad_centre){
      top = 0;
    }
  }

  if (image.cols < min_size) {
    left = (min_size - image.cols) / 2;

    if (!pad_centre){
      left = 0;
    }
  }

  top += margin_h;
  left += margin_w;

  Dtype *detection_data = detection_blob->mutable_cpu_data();
  int step = detection_blob->width();
  for (int index = 0; index < detection_blob->count(); index += step) {
    detection_data[index + 1] = detection_data[index + 1] + left;
    detection_data[index + 2] = detection_data[index + 2] + top;
    detection_data[index + 3] = detection_data[index + 3] + left;
    detection_data[index + 4] = detection_data[index + 4] + top;
  }
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::RotateBbox(const cv::Point2f & pivot, const cv::Point2f & im_dim, Dtype angle, Blob<Dtype>* detection_blob){
  angle = angle * M_PI / 180.0f;
  int step = detection_blob->width();
  Dtype *detection_data = detection_blob->mutable_cpu_data();
  Dtype xmin, ymin, xmax, ymax, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4;
  for (int index = 0; index < detection_blob->count(); index += step) {
  
    xmin = detection_data[index + 1];
    ymin = detection_data[index + 2];
    xmax = detection_data[index + 3];
    ymax = detection_data[index + 4];

    // top left
    new_x1 =   (xmin - pivot.x) * cos(angle) + (ymin - pivot.y) * sin(angle) + pivot.x;
    new_y1 = - (xmin - pivot.x) * sin(angle) + (ymin - pivot.y) * cos(angle) + pivot.y;
    // top right
    new_x2 =   (xmax - pivot.x) * cos(angle) + (ymin - pivot.y) * sin(angle) + pivot.x;
    new_y2 = - (xmax - pivot.x) * sin(angle) + (ymin - pivot.y) * cos(angle) + pivot.y;
    // bottom right
    new_x3 =   (xmax - pivot.x) * cos(angle) + (ymax - pivot.y) * sin(angle) + pivot.x;
    new_y3 = - (xmax - pivot.x) * sin(angle) + (ymax - pivot.y) * cos(angle) + pivot.y;
    // bottom left
    new_x4 =   (xmin - pivot.x) * cos(angle) + (ymax - pivot.y) * sin(angle) + pivot.x;
    new_y4 = - (xmin - pivot.x) * sin(angle) + (ymax - pivot.y) * cos(angle) + pivot.y;

    // find out the new mins and max's, clip by image boundaries if necessary
    detection_data[index + 1] = round(std::max(Dtype(0), Dtype(std::min(std::min(std::min(new_x1, new_x2), new_x3), new_x4))));
    detection_data[index + 2] = round(std::max(Dtype(0), Dtype(std::min(std::min(std::min(new_y1, new_y2), new_y3), new_y4))));
    detection_data[index + 3] = round(std::min(Dtype(im_dim.x-1), Dtype(std::max(std::max(std::max(new_x1, new_x2), new_x3), new_x4))));
    detection_data[index + 4] = round(std::min(Dtype(im_dim.y-1), Dtype(std::max(std::max(std::max(new_y1, new_y2), new_y3), new_y4))));

  }
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::ApplySampleRotation(cv::Mat & image, cv::Mat & label, Blob<Dtype>* detection_blob){

  if (!this->random_rotate_ || this->layer_param_.phase() == caffe::Phase::TEST){
    return;
  }
  Dtype angle = std::uniform_real_distribution<Dtype>(this->min_rotation_angle_, this->max_rotation_angle_)(*rng_);

  // Rotate using OpenCV
  cv::Point2f point( image.cols/2.0f, image.rows/2.0f );
  cv::Mat rotation_matrix = cv::getRotationMatrix2D(point, angle, 1.0);

  cv::warpAffine(image, image, rotation_matrix, cv::Size(image.cols, image.rows), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  cv::warpAffine(label, label, rotation_matrix, cv::Size(label.cols, label.rows), cv::INTER_NEAREST, cv::BORDER_CONSTANT, 255);

  if (detection_blob != NULL){
    cv::Point2f im_dim ( image.cols, image.rows );
    RotateBbox(point, im_dim, angle, detection_blob);
  }
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::ApplySampleGaussianBlur(cv::Mat &image){

  float blur_factor = std::uniform_real_distribution<float>(0, 1)(*rng_);
  bool do_blur = (blur_factor < this->layer_param_.image_label_data_param().blur_probability());

  if (!this->random_gaussian_blur_ || this->layer_param_.phase() == caffe::Phase::TEST || !do_blur){
    return;
  }

  const int min_kernel_width = this->layer_param_.image_label_data_param().min_kernel_size();
  const int max_kernel_width = this->layer_param_.image_label_data_param().max_kernel_size();
  int kernel_width = std::uniform_int_distribution<int>(min_kernel_width, max_kernel_width)(*rng_);
  // this also needs to be odd
  if (kernel_width % 2 == 0){
    kernel_width = std::max(3, kernel_width - 1);
  }

  cv::Size kernel_size (kernel_width, kernel_width);

  const float min_sigma_x = this->layer_param_.image_label_data_param().min_sigma_x();
  const float max_sigma_x = this->layer_param_.image_label_data_param().max_sigma_x();
  float sigma_x = std::uniform_real_distribution<float>(min_sigma_x, max_sigma_x)(*rng_);

  const float min_sigma_y = this->layer_param_.image_label_data_param().min_sigma_y();
  const float max_sigma_y = this->layer_param_.image_label_data_param().max_sigma_y();
  float sigma_y = std::uniform_real_distribution<float>(min_sigma_y, max_sigma_y)(*rng_);

  cv::GaussianBlur(image, image, kernel_size, sigma_x, sigma_y);

}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::ApplyBboxPerturbation(cv::Mat &image, Blob<Dtype> *detection_blob){

  if (!this->random_box_perturb_ || this->layer_param_.phase() == caffe::Phase::TEST){
    return;
  }

  const float box_perturb_max_scale = this->layer_param_.image_label_data_param().box_perturb_max_scale_percent() / 100.0f;
  const float box_perturb_max_translate = this->layer_param_.image_label_data_param().box_perturb_max_translate_percent() / 100.0f;

  Dtype x_scale_ratio = std::uniform_real_distribution<Dtype>( -box_perturb_max_scale, box_perturb_max_scale)(*rng_);
  Dtype y_scale_ratio = std::uniform_real_distribution<Dtype>( -box_perturb_max_scale, box_perturb_max_scale)(*rng_);

  Dtype x_translate_ratio = std::uniform_real_distribution<Dtype>( -box_perturb_max_translate, box_perturb_max_translate)(*rng_);
  Dtype y_translate_ratio = std::uniform_real_distribution<Dtype>( -box_perturb_max_translate, box_perturb_max_translate)(*rng_);

  int step = detection_blob->width();
  Dtype *detection_data = detection_blob->mutable_cpu_data();
  Dtype xmin, ymin, xmax, ymax, width, height, x_scale, y_scale, x_translate, y_translate;

  for (int index = 0; index < detection_blob->count(); index += step) {
  
    xmin = detection_data[index + 1];
    ymin = detection_data[index + 2];
    xmax = detection_data[index + 3];
    ymax = detection_data[index + 4];

    width = xmax - xmin + 1;
    height = ymax - ymin + 1;

    // perturbation in absolute terms
    x_scale = floor(width * x_scale_ratio * 0.5);
    y_scale = floor(height * y_scale_ratio * 0.5);
    x_translate = floor(width * x_translate_ratio);
    y_translate = floor(height * y_translate_ratio);

    // the new mins and max's
    xmin = round(xmin - x_scale + x_translate);
    ymin = round(ymin - y_scale + y_translate);
    xmax = round(xmax + x_scale + x_translate);
    ymax = round(ymax + y_scale + y_translate);

    // clip by image boundaries if necessary
    detection_data[index + 1] = std::min(std::max(Dtype(0), xmin), Dtype(image.cols));
    detection_data[index + 2] = std::min(std::max(Dtype(0), ymin), Dtype(image.rows));
    detection_data[index + 3] = std::min(std::max(Dtype(0), xmax), Dtype(image.cols));
    detection_data[index + 4] = std::min(std::max(Dtype(0), ymax), Dtype(image.rows));

  }

}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::load_batch(BatchIndexDetection<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  //LOG(INFO) << "Starting to load batch";
  CHECK(this->transformed_data_.count());
  ImageLabelDataParameter data_param =
      this->layer_param_.image_label_data_param();
  const int batch_size = data_param.batch_size();
  string data_dir = data_param.data_dir();
  string image_dir =
      this->layer_param_.image_label_data_param().image_dir();
  string label_dir =
      this->layer_param_.image_label_data_param().label_dir();

  if (image_dir == "" && label_dir == "" && data_dir != "") {
    image_dir = data_dir;
    label_dir = data_dir;
  }

  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_], true);

  cv_img = PadImage(cv_img, crop_size);

  //LOG(INFO) << "Loading first image";
  CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);


  cv::Mat cv_label = ReadImageToCVMat(label_dir + label_lines_[lines_id_],
                                      this->layer_param_.image_label_data_param().num_channels() > 1);
  //LOG(INFO) << "Loading first label";
  CHECK(cv_label.data) << "Could not load " << label_lines_[lines_id_];
  cv_label = PadImage(cv_label, crop_size);

  vector<int> label_shape = this->data_transformer_->InferBlobShape(cv_label);

  this->transformed_label_.Reshape(label_shape);

  auto &label_slice = this->layer_param_.image_label_data_param().label_slice();

  label_shape[0] = batch_size;
  label_shape[1] = this->layer_param_.image_label_data_param().num_channels();
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_idx = batch->index_.mutable_cpu_data();

  std::string img_id;

  // datum scales
  auto lines_size = image_lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    int num_dets = -1;
    int max_num_dets_allowed = this->layer_param_.image_label_data_param().max_num_dets_allowed();
    int image_offset, label_offset;
    bool valid_image_not_found = true;
    // Load the detection index, and the detection file
    // Do the requested transform to the image, label, and detections
    // In the transformed label map:
    // if no. of detections > max no. of dets allowed, go to the next image until
    // we reach an image with fewer no. of detections.
    while (valid_image_not_found) {
      
      GetIndex(image_lines_[lines_id_], item_id, prefetch_idx, img_id);
      LoadDetection(batch, item_id, prefetch_idx[item_id], img_id);

      cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
      cv::Mat cv_label = ReadImageToCVMat(label_dir + label_lines_[lines_id_],
                                          this->layer_param_.image_label_data_param().num_channels() > 1);

      // do HSV noise here
      if (hsv_noise_){
        ApplyHSVNoise(cv_img, h_noise_, s_noise_, v_noise_, rng_);
      }

      // do the Random Gaussian blur here
      if (random_gaussian_blur_){
        ApplySampleGaussianBlur(cv_img);
      }

      // random scaling if set
      SampleScale(&cv_img, &cv_label, &batch->detection_);

      switch (data_param.padding()) {
        case ImageLabelDataParameter_Padding_ZERO:
          cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, 0);
          // pad the bboxes if we centre-pad the image
          PadBbox(cv_img, crop_size, label_margin_w_, label_margin_h_, pad_centre_, &batch->detection_);
          cv_img = PadImage(cv_img, crop_size, 0, pad_centre_);
          break;
        case ImageLabelDataParameter_Padding_REFLECT:
          cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, -1);
          // pad the bboxes if we centre-pad the image
          PadBbox(cv_img, crop_size, label_margin_w_, label_margin_h_, pad_centre_, &batch->detection_);
          cv_img = PadImage(cv_img, crop_size, -1, pad_centre_);
          break;
        default:
          LOG(FATAL) << "Unknown Padding";
      }

      cv_label = ExtendLabelMargin(cv_label, label_margin_w_, label_margin_h_, 255);
      cv_label = PadImage(cv_label, crop_size, 255, pad_centre_);

      // do the rotation here
      if (random_rotate_){
        ApplySampleRotation(cv_img, cv_label, &batch->detection_);
      }

      if (random_box_perturb_){
        ApplyBboxPerturbation(cv_img, &batch->detection_);
      }

      CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
      CHECK(cv_label.data) << "Could not load " << label_lines_[lines_id_];
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image

      image_offset = batch->data_.offset(item_id);
      label_offset = batch->label_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + image_offset);
      // this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
      this->data_transformer_->Transform(cv_img, cv_label,
                                        &(this->transformed_data_),
                                        &(this->transformed_label_),
                                        &batch->detection_);
      
      num_dets = batch->detection_.channels();

      valid_image_not_found = max_num_dets_allowed > 0 && num_dets > max_num_dets_allowed;
      valid_image_not_found = valid_image_not_found || batch->detection_.cpu_data()[0] == -1; // this signals zero valid dets

      if (valid_image_not_found) {
        if (batch->detection_.cpu_data()[0] == -1) {
          LOG(INFO) << "Data: " << img_id << ": zero valid detections after transformations. Skipping it.";
        } else {
          LOG(INFO) << "Data: " << img_id << ": " << num_dets << " > " << max_num_dets_allowed << " (max no. of dets allowed). Skipping it.";
        }
        // go to the next iter
        lines_id_++;
        if (lines_id_ >= lines_size) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          lines_id_ = 0;
          if (this->layer_param_.image_label_data_param().shuffle()) {
            ShuffleImages();
          }
        } // if (lines_id_ >= lines_size)
      } // if (max_num_dets_allowed > 0 && num_dets > max_num_dets_allowed)

    } // while (num_dets == -1 || (max_num_dets_allowed > 0 && num_dets > max_num_dets_allowed))

    Dtype *label_data = prefetch_label + label_offset;
    const Dtype *t_label_data = this->transformed_label_.cpu_data();

    GetLabelSlice(t_label_data, crop_size, crop_size, label_slice, label_data, this->layer_param_.image_label_data_param().num_channels());

    LOG(INFO) << "Data: " << img_id << ", no. of dets: " << num_dets;

    trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_label_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  batch_timer.Stop();
}

template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::GetIndex(const std::string & filename, const int item_id, Dtype * prefetch_idx, std::string & img_id) {
  Dtype index = lines_id_;

  if ( this->layer_param_.image_label_data_param().filename_to_idx() ){
    unsigned long slice_start = filename.rfind("/") + 1;
    unsigned long slice_end = filename.rfind(".");
    std::string temp = filename.substr(slice_start, slice_end - slice_start);
    img_id = temp;
    std::string final = "";

    // Because of floating point precision, we need to reduce the image index further for VOC
    // This would not be required if we used doubles
    boost::replace_all(temp, "_00", "");
    if (temp.size() > 4){
      std::string year = temp.substr(0, 4);
      if (year == "2007") { boost::replace_all(temp, year, "7"); };
      if (year == "2008") { boost::replace_all(temp, year, "8"); };
      if (year == "2009") { boost::replace_all(temp, year, "9"); };
      if (year == "2010") { boost::replace_all(temp, year, "1"); };
      if (year == "2011") { boost::replace_all(temp, year, "2"); };
      if (year == "2012") { boost::replace_all(temp, year, "3"); };
    }
    
    for (int i = 0; i < temp.size(); ++i){
      char t = temp[i];
      if (t >= '0' && t <= '9'){
        final += t;
      }
    }
    
    if (final.size() != 0){
      index = boost::lexical_cast<Dtype>(final);
    }
  }

  prefetch_idx[item_id] = index;
}

/*
 * Assumes the batch size is only 1.
 */
template <typename Dtype>
void ImageLabelDataIndexDetLayer<Dtype>::LoadDetection(BatchIndexDetection<Dtype>* batch, const int batch_index, const int det_index, const std::string img_id) {
  CHECK_EQ(batch_index, 0) << "Currently, only batch size of one is supported";
  // Load the detection file
  const std::string box_dir = this->layer_param_.image_label_data_param().box_dir();
  const std::string box_ext = this->layer_param_.image_label_data_param().box_extension();
  std::string detection_file_name;
  if (this->layer_param_.image_label_data_param().filename_in_idx()) {
    detection_file_name = box_dir + "/" + boost::lexical_cast<std::string>(det_index) + box_ext;
  } else {
    detection_file_name = box_dir + "/" + img_id + box_ext;
  }

  vector<shared_ptr<const tvg::Detection> > detection_boxes;
  tvg::DetectionUtils::read_detections_from_file(detection_boxes, detection_file_name);

  // Resize the detection blob
  const int num = batch->detection_.num();
  CHECK_EQ(num, 1) << "Currently, only batch size of one is supported";
  const int n_detections = detection_boxes.size();
  const int width = tvg::Detection::num_per_detection; // = 6

  batch->detection_.Reshape(num,n_detections,1,width);

  // Fill up the detection blob
  int counter = 0;
  Dtype* detection_data = batch->detection_.mutable_cpu_data();

  for (int i = 0; i < n_detections; ++i){

    const std::vector<int> &det_box = detection_boxes[i]->get_foreground_pixels();
    const int det_label = detection_boxes[i]->get_label();
    const float det_score = detection_boxes[i]->get_score();

    CHECK_EQ(det_box.size(), 4) << "Detection should have exactly four co-ordinates - the top left and bottom right corners of the bounding box";
    const int x_start = det_box[0];
    const int y_start = det_box[1];
    const int x_end = det_box[2];
    const int y_end = det_box[3];

    detection_data[counter++] = det_label;
    detection_data[counter++] = x_start;
    detection_data[counter++] = y_start;
    detection_data[counter++] = x_end;
    detection_data[counter++] = y_end;
    detection_data[counter++] = det_score;
  }
}

INSTANTIATE_CLASS(ImageLabelDataIndexDetLayer);
REGISTER_LAYER_CLASS(ImageLabelDataIndexDet);

}  // namespace caffe
