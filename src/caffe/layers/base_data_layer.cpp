#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

template <typename Dtype>
BasePrefetchingDataIndexLayer<Dtype>::BasePrefetchingDataIndexLayer(const LayerParameter &param)
  : BaseDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_()
{
  for (int i = 0; i < PREFETCH_COUNT; ++i){
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataIndexLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                      const vector<Blob<Dtype> *> &top) {

  BaseDataLayer<Dtype>::LayerSetUp(bottom, top); // TODO: Do I need BaseDataLayer<Dtype>::...?

  if (top.size() == 3){
    output_index_ = true;
  }
  else{
    output_index_ = false;
  }

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_index_) {
        prefetch_[i].index_.mutable_cpu_data();
      }
  }
  #ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
      for (int i = 0; i < PREFETCH_COUNT; ++i) {
         prefetch_[i].data_.mutable_gpu_data();
         if (this->output_labels_) {
            prefetch_[i].index_.mutable_gpu_data();
         }
      }
  }
  #endif

  DLOG(INFO) << "Initializing prefetch for batches with indices";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized for batches with indices.";
}

template <typename Dtype>
void BasePrefetchingDataIndexLayer<Dtype>::InternalThreadEntry() {
  #ifndef CPU_ONLY
    cudaStream_t stream;
    if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
  #endif

  try {
    while (!must_stop()) {
      BatchWithIndex<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
      #ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      #endif
        prefetch_full_.push(batch);
      }
    }
  catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
  #ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  #endif
}

template <typename Dtype>
void BasePrefetchingDataIndexLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  BatchWithIndex<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
               top[1]->mutable_cpu_data());

  }

  if (this->output_index_) {
    top[2]->ReshapeLike(batch->index_);

    caffe_copy(batch->index_.count(), batch->index_.cpu_data(),
               top[2]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataIndexLayer, Forward);
#endif

INSTANTIATE_CLASS(BasePrefetchingDataIndexLayer);

//

template <typename Dtype>
BasePrefetchingDataIndexDetectionLayer<Dtype>::BasePrefetchingDataIndexDetectionLayer(const LayerParameter &param)
        : BaseDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_()
{
  for (int i = 0; i < PREFETCH_COUNT; ++i){
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataIndexDetectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                      const vector<Blob<Dtype> *> &top) {

  BaseDataLayer<Dtype>::LayerSetUp(bottom, top); // TODO: Do I need BaseDataLayer<Dtype>::...?

  if (top.size() >= 3){
    output_index_ = true;
  }
  else{
    output_index_ = false;
  }

  if (top.size() == 4){
    output_detection_ = true;
  }
  else{
    output_detection_ = false;
  }

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_index_) {
      prefetch_[i].index_.mutable_cpu_data();
    }
    if (this->output_detection_){
      /* TODO: Why do I need to reshape the detection blob? For the other index blob, it is already initialised
         with a count of 1 by this point. */
      prefetch_[i].detection_.Reshape(1,1,1,1);
      prefetch_[i].detection_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].index_.mutable_gpu_data();
      }
      if (this->output_detection_){
        prefetch_[i].detection_.mutable_gpu_data();
      }
    }
  }
#endif

  DLOG(INFO) << "Initializing prefetch for batches with indices";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized for batches with indices.";
}

template <typename Dtype>
void BasePrefetchingDataIndexDetectionLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      BatchIndexDetection<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  }
  catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataIndexDetectionLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  BatchIndexDetection<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
               top[1]->mutable_cpu_data());

  }

  if (this->output_index_) {
    top[2]->ReshapeLike(batch->index_);

    caffe_copy(batch->index_.count(), batch->index_.cpu_data(),
               top[2]->mutable_cpu_data());
  }

  if (this->output_detection_){
    top[3]->ReshapeLike(batch->detection_);

    caffe_copy(batch->detection_.count(), batch->detection_.cpu_data(),
               top[3]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataIndexDetectionLayer, Forward);
#endif

INSTANTIATE_CLASS(BasePrefetchingDataIndexDetectionLayer);


//

template <typename Dtype>
BasePrefetchingDataClassLayer<Dtype>::BasePrefetchingDataClassLayer(const LayerParameter &param)
  : BaseDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_()
{
  for (int i = 0; i < PREFETCH_COUNT; ++i){
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataClassLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                      const vector<Blob<Dtype> *> &top) {

  BaseDataLayer<Dtype>::LayerSetUp(bottom, top); // TODO: Do I need BaseDataLayer<Dtype>::...?

  if (top.size() == 3){
    output_class_label_ = true;
  }
  else{
    output_class_label_ = false;
  }

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_class_label_) {
        prefetch_[i].class_label_.mutable_cpu_data();
      }
  }
  #ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
      for (int i = 0; i < PREFETCH_COUNT; ++i) {
         prefetch_[i].data_.mutable_gpu_data();
         if (this->output_class_label_) {
            prefetch_[i].class_label_.mutable_gpu_data();
         }
      }
  }
  #endif

  DLOG(INFO) << "Initializing prefetch for batches with classification labels";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized for batches with classification labels.";
}

template <typename Dtype>
void BasePrefetchingDataClassLayer<Dtype>::InternalThreadEntry() {
  #ifndef CPU_ONLY
    cudaStream_t stream;
    if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
  #endif

  try {
    while (!must_stop()) {
      BatchWithClassLabel<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
      #ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      #endif
        prefetch_full_.push(batch);
      }
    }
  catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
  #ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
  #endif
}

template <typename Dtype>
void BasePrefetchingDataClassLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  BatchWithClassLabel<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
               top[1]->mutable_cpu_data());

  }

  if (this->output_class_label_) {
    top[2]->ReshapeLike(batch->class_label_);

    caffe_copy(batch->class_label_.count(), batch->class_label_.cpu_data(),
               top[2]->mutable_cpu_data());
  }

  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataClassLayer, Forward);
#endif

INSTANTIATE_CLASS(BasePrefetchingDataClassLayer);


}  // namespace caffe
