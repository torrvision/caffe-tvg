#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

template <typename Dtype>
void BasePrefetchingDataIndexLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BatchWithIndex<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_gpu_copy(batch->data_.count(), batch->data_.gpu_data(),
             top[0]->mutable_gpu_data());

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_gpu_copy(batch->label_.count(), batch->label_.gpu_data(),
               top[1]->mutable_gpu_data());
  }

  if (this->output_index_) {
    top[2]->ReshapeLike(batch->index_);

    caffe_gpu_copy(batch->index_.count(), batch->index_.gpu_data(),
               top[2]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataIndexLayer);

//

template <typename Dtype>
void BasePrefetchingDataIndexDetectionLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BatchIndexDetection<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_gpu_copy(batch->data_.count(), batch->data_.gpu_data(),
                 top[0]->mutable_gpu_data());

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_gpu_copy(batch->label_.count(), batch->label_.gpu_data(),
                   top[1]->mutable_gpu_data());
  }

  if (this->output_index_) {
    top[2]->ReshapeLike(batch->index_);

    caffe_gpu_copy(batch->index_.count(), batch->index_.gpu_data(),
                   top[2]->mutable_gpu_data());
  }

  if (this->output_detection_){
    top[3]->ReshapeLike(batch->detection_);

    caffe_gpu_copy(batch->detection_.count(), batch->detection_.gpu_data(),
                   top[3]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataIndexDetectionLayer);

//

template <typename Dtype>
void BasePrefetchingDataClassLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BatchWithClassLabel<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_gpu_copy(batch->data_.count(), batch->data_.gpu_data(),
             top[0]->mutable_gpu_data());

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_gpu_copy(batch->label_.count(), batch->label_.gpu_data(),
               top[1]->mutable_gpu_data());
  }

  if (this->output_class_label_) {
    top[2]->ReshapeLike(batch->class_label_);

    caffe_gpu_copy(batch->class_label_.count(), batch->class_label_.gpu_data(),
               top[2]->mutable_gpu_data());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataClassLayer);

}  // namespace caffe
