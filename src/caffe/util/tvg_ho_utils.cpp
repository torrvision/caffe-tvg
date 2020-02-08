//
// Created by vision on 21/10/15.
//

#include "caffe/util/tvg_ho_utils.hpp"

namespace tvg {

  namespace HOPotentialsUtils {

    void load_superpixels(vector< vector <vector<int> > > & cliques,
                          const string & filename, char separator) {

      ifstream infile(filename.c_str());
      stringstream iss;
      iss.clear();
      string line;

      if (!infile){
        printf("Could not open %s \n. Exitting", filename.c_str());
        throw std::invalid_argument("Detection file not found\n");
      }

      getline(infile, line);
      const int num_layers = stoi(line);

      cliques.resize(num_layers);

      for (int layer = 0; layer < num_layers; ++layer) {
        getline(infile, line);
        iss.clear();
        iss << line;

        string token;
        getline(iss, token, separator);
        size_t layer_id = stoi(token);
        getline(iss, token, separator);
        size_t num_segments = stoi(token);

        cliques[layer_id].resize(num_segments);

        for (size_t segment = 0; segment < num_segments; ++segment) {

          getline(infile, line);
          iss.clear();
          iss << line;
          getline(iss, token, separator);
          size_t segment_id = stoi(token);
          if (segment_id != segment) {
            throw runtime_error("Error while parsing superpixel file");
          }
          getline(iss, token, separator);
          size_t num_pixels = stoi(token);

          cliques[layer_id][segment_id].resize(num_pixels);

          getline(infile, line);
          iss.clear();
          iss << line;

          for (size_t pixel_count = 0; pixel_count < num_pixels; ++pixel_count) {
            getline(iss, token, separator);
            cliques[layer_id][segment_id][pixel_count] = stoi(token);
          }
        }
      }

      // print out num superpixels in each layer
      /*cout << "Loaded " << cliques.size() << endl;
      for (size_t i = 0; i < cliques.size(); ++i) {
        cout << "Layer " << i << " has " << cliques[i].size() << " superpixels";
      }*/
    }

    void init_to_have_same_size(const vector<vector<vector<int> > > & cliques,
                                vector<vector<vector<float> > > & target_to_init,
                                const int num_labels, const float init_value) {

      const size_t num_layers = cliques.size();

      target_to_init.resize(num_layers); // not calling clear() to save some memory reallocation if possible.

      for (int layer_id = 0; layer_id < num_layers; ++layer_id) {

        const size_t num_segments = cliques[layer_id].size();
        target_to_init[layer_id].resize(num_segments);

        for (int segment_id = 0; segment_id < num_segments; ++segment_id) {
          target_to_init[layer_id][segment_id].resize(num_labels, init_value);

          // The resize() method does not work as intended here!
          // When a container is shrunk, resize() does not reinitialise old values (http://www.cplusplus.com/reference/vector/vector/resize/)
          // So we have to do the following as well
          for (size_t i = 0; i < num_labels; ++i){
            target_to_init[layer_id][segment_id][i] = init_value;
          }

        }

      }

    }
  }

}
