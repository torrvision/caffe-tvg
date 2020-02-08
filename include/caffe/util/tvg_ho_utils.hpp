#ifndef CAFFE_TVG_HO_UTILS_H
#define CAFFE_TVG_HO_UTILS_H

#include <vector>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"


namespace tvg {

  namespace HOPotentialsUtils {

    using namespace std;
    using namespace boost;
    using namespace caffe;

    /**
     * Read superpixels from a file.
     */
    void load_superpixels(vector< vector <vector<int> > > & cliques,
                          const string & filename, char separator = ' ');

    /**
     * Initialize the target_to_init 3D vector
     */
    void init_to_have_same_size(const vector<vector<vector<int> > > & cliques,
                                vector<vector<vector<float> > > & target_to_init,
                                const int num_labels, const float init_value);

    /**
     * Average stat potentials.
     */
    template<typename Dtype>
    void average_stats_potentials(const Blob<Dtype> & unaries,
                                  const vector< vector< vector<int> > > & cliques,
                                  vector< vector < vector<float> > > & stat_potentials) {

      const int num_pixels = unaries.width() * unaries.height();
      const int channels = unaries.channels();

      const Dtype *unary_data = unaries.cpu_data();

      for (int layer_id = 0; layer_id < cliques.size(); ++layer_id) {

        if (cliques.size() != stat_potentials.size() ) { throw std::runtime_error("Cliques and stat potentials not same size. E1");}

        const vector<vector<int> > & segments_in_layer = cliques[layer_id];
        const size_t segment_count = segments_in_layer.size();

        if (segment_count != stat_potentials[layer_id].size() ) { throw std::runtime_error("Cliques and stat potentials not same size. E2");}

        for (int segment_id = 0; segment_id < segment_count; ++segment_id) {

          const vector<int> & pixels = segments_in_layer[segment_id];

          const size_t segment_size = pixels.size();

          //for (int pixel_id : pixels) { // All versions of CUDA do not support C++ 11
          for (int i_pixel = 0; i_pixel < pixels.size(); ++i_pixel){
            int pixel_id = pixels[i_pixel];

            for (int label_id = 0; label_id < channels; ++label_id) {
              stat_potentials[layer_id][segment_id][label_id] +=
                      unary_data[label_id * num_pixels + pixel_id] / segment_size; // TODO: Although this is faster, this is more likely to have numerical issues, since unary_data[i] could be a small number which you are then dividing by something that could be quite large (1000)
            }
          }
        }
      }
    }

    /**
     * Compute the additional unaries
     */
    template<typename Dtype>
    void compute_sp_additional_unaries(Blob<Dtype> & sp_additional_unaries,
                                       const vector< vector< vector<int> > > & cliques,
                                       const vector< vector< vector<float> > > &stat_potentials,
                                       const Dtype * const ho_w_param_data,
                                       bool takeLog = false) {

      const int num_pixels = sp_additional_unaries.width() * sp_additional_unaries.height();
      const int channels = sp_additional_unaries.channels();

      Dtype * const ho_update_data = sp_additional_unaries.mutable_cpu_data();
      caffe_set(sp_additional_unaries.count(), Dtype(0.), ho_update_data);

      for (int layer_id = 0; layer_id < cliques.size(); ++layer_id) {

        const float ho_w_param = ho_w_param_data[layer_id];

        const vector< vector<int> > & segments_vec = cliques[layer_id];

        const size_t num_segments = segments_vec.size();
        for (int segment_id = 0; segment_id < num_segments; ++segment_id) {

          const vector<int> & segment = segments_vec[segment_id];
          const vector<float> & stat_potentials_for_seg = stat_potentials[layer_id][segment_id];

          const size_t segment_size = segment.size();
          for (int k = 0; k < segment_size; ++k) { //TODO Can be optimized?, use foreach loop
            const int pixel_id = segment[k];

            for (int label_id = 0; label_id < channels; ++label_id) {

              if (takeLog){
                ho_update_data[label_id * num_pixels + pixel_id] += ho_w_param * log(stat_potentials_for_seg[label_id]);
              }
              else {
                ho_update_data[label_id * num_pixels + pixel_id] += (ho_w_param * stat_potentials_for_seg[label_id]);
              }
            }
          }
        }
      }
    }


    template<typename Dtype>
    void bp_compute_additional_unaries(const Blob<Dtype> & sp_additional_unaries,
                                       const vector< vector< vector<int> > > & cliques,
                                       const vector< vector< vector<float> > > & stat_potentials,
                                       vector< vector< vector<float> > > & stat_potentials_diff,
                                       Blob<Dtype> & ho_w_param_blob) {

      const int num_pixels = sp_additional_unaries.width() * sp_additional_unaries.height();
      const int channels = sp_additional_unaries.channels();

      init_to_have_same_size(cliques, stat_potentials_diff, channels, 0);

      const Dtype * const ho_update_diff = sp_additional_unaries.cpu_diff();

      const Dtype * const ho_w_param_data = ho_w_param_blob.cpu_data();
      Dtype * const ho_w_param_diff = ho_w_param_blob.mutable_cpu_diff();

      for (int layer_id = 0; layer_id < cliques.size(); ++layer_id) {

        const vector< vector<int> > & segments_vec = cliques[layer_id];
        const vector< vector<float> > & stat_potentials_vec = stat_potentials[layer_id];
        vector< vector<float> > & stat_potentials_diff_vec = stat_potentials_diff[layer_id];

        const size_t num_segments = segments_vec.size();
        for (int segment_id = 0; segment_id < num_segments; ++segment_id) {

          const vector<int> & segment = segments_vec[segment_id];
          const vector<float> & stat_potentials_for_seg = stat_potentials_vec[segment_id];
          vector<float> & stat_potential_diff_for_seg = stat_potentials_diff_vec[segment_id];

          const size_t segment_size = segment.size();
          for (int k = 0; k < segment_size; ++k) {
            const int pixel_id = segment[k];

            for (int label_id = 0; label_id < channels; ++label_id) {

              const Dtype cur_diff_value = ho_update_diff[label_id * num_pixels + pixel_id];

              ho_w_param_diff[layer_id] +=  cur_diff_value * stat_potentials_for_seg[label_id];
              stat_potential_diff_for_seg[label_id] += cur_diff_value * ho_w_param_data[layer_id];
            }
          }
        }
      }
    }


    /**
     * BP the average stat potentials operation done in tvg::HOPotentialsUtils::average_stats_potentials() function.
     * Note that this function adds to the diffs or unaries blob without clearning existing diffs.
     */
    template<typename Dtype>
    void bp_average_stats_potentials(Blob<Dtype> & unaries,
                                     const vector<vector<vector<int> > > & cliques,
                                     const vector<vector<vector<float> > > & stat_potentials_diff) {

      Dtype * const unary_diff = unaries.mutable_cpu_diff();
      const int num_pixels = unaries.width() * unaries.height();
      const int channels = unaries.channels();

      for (int layer_id = 0; layer_id < cliques.size(); ++layer_id) {

        const vector<vector<int> > &segments_in_layer = cliques[layer_id];
        const size_t segment_count = segments_in_layer.size();

        for (int segment_id = 0; segment_id < segment_count; ++segment_id) {

          const vector<int> &pixels = segments_in_layer[segment_id];
          const size_t segment_size = pixels.size();

          //for (int pixel_id : pixels) {
          for (int i_pixel = 0; i_pixel < pixels.size(); ++i_pixel){
            int pixel_id = pixels[i_pixel];

            for (int label_id = 0; label_id < channels; ++label_id) {

              unary_diff[label_id * num_pixels + pixel_id] +=
                      (stat_potentials_diff[layer_id][segment_id][label_id] / segment_size);
            }
          }
        }
      }
    }

  } // end namespace - tvg::HOPotentialsUtils
} // end namespace - tvg

#endif //CAFFE_TVG_HO_UTILS_H