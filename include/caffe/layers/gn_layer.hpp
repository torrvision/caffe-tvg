#ifndef CAFFE_GN_LAYER_HPP_
#define CAFFE_GN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Group normalization: slice the input blob in the channel axis producing
 * multiple "groups" which have an equal number of channels. These groups are 
 * normalised independently, and channel-wise scale and bias are applied.
 * Wu, Yuxin, and He, Kaiming. "Group normalization." In ECCV (2018).
 */
template <typename Dtype>
class GNLayer : public Layer<Dtype> {
 public:
  explicit GNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // CPU functions
  void CalcGroupSum(const Dtype* input, const bool add_to_self, 
      Dtype* output);
  void CalcBatchSum(const Dtype* input, const bool add_to_self, 
      Blob<Dtype>* medium, Dtype* output);
  void BroadcastGroupStats(const Dtype* input, const bool add_to_self, 
      Dtype* output);
  void BroadcastChannel(const Dtype* input, const bool add_to_self, 
      Blob<Dtype>* medium, Dtype* output);

  // GPU functions
  void CalcGroupSum_gpu(const Dtype* input, const bool add_to_self, 
      Dtype* output);
  void CalcBatchSum_gpu(const Dtype* input, const bool add_to_self, 
      Blob<Dtype>* medium, Dtype* output);
  void BroadcastGroupStats_gpu(const Dtype* input, const bool add_to_self, 
      Dtype* output);
  void BroadcastChannel_gpu(const Dtype* input, const bool add_to_self, 
      Blob<Dtype>* medium, Dtype* output);

  Dtype gn_eps_;

  int num_;
  int channels_;
  int height_;
  int width_;
  int num_groups_;
  int channels_per_group_;

  Blob<Dtype> group_statistic_; // N*N_g
  Blob<Dtype> spatial_statistic_; // N*C
  Blob<Dtype> broadcast_buffer_; // N*C*H*W

  Blob<Dtype> x_norm_; // N*C*H*W
  Blob<Dtype> x_inv_std_; // N*N_g

  Blob<Dtype> spatial_group_sum_multiplier_; // C_g*H*W
  Blob<Dtype> spatial_sum_multiplier_; // H*W
  Blob<Dtype> batch_sum_multiplier_; // N
};

}  // namespace caffe

#endif  // CAFFE_GN_LAYER_HPP_
