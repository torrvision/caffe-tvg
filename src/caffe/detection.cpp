#include "caffe/detection.hpp"
#include "caffe/util/math_functions.hpp"

#include <fstream>
#include <sstream>

namespace tvg {

  /**
  * For debugging purposes. Prints contents of the @Detection object.
  */
  std::ostream &operator<<(std::ostream &stream, const Detection &detection) {
    stream << "Detection(label = " << detection.label << ", score = " << detection.score << ", foreground pixels = [";
    for (int i = 0; i < detection.foreground_pixels.size(); ++i) {
      //stream << detection.foreground_pixels[i] << ", ";
    }
    stream << detection.foreground_pixels.size();
    return stream << "]" << std::endl;
  }

  namespace DetectionUtils {

    /**
    * Reads detections from the given file and populates the given detection_list.
    */
    int read_detections_from_file(std::vector<boost::shared_ptr<const Detection> > &detection_list,
                                              const std::string &file_name, bool insert_empty_detections) {

      int detections_read = 0;
      detection_list.clear();
      std::fstream in_file(file_name.c_str(), std::ios_base::in);

      if (!in_file){
          printf("Could not open %s \n. Exitting", file_name.c_str());
          throw std::invalid_argument("Detection file not found\n");
      }

      std::stringstream iss;
      iss.clear();
      std::string line;

      std::getline(in_file, line);
      int det_label = std::stoi(line); //C++ 11 function

      while (det_label != 0) {
          if (det_label < 0 || det_label > 20) {
              throw std::runtime_error("Invalid label in the detection data file.");
          }

          std::getline(in_file, line);
          float det_score = std::stof(line); //C++ 11 function.

          boost::shared_ptr<Detection> detection(new Detection(det_label, det_score));

          std::getline(in_file, line);
          iss.clear();
          iss << line;
          std::string token;
          const char separator = ' ';

          while (std::getline(iss, token, separator)) {
            if (token == " " || token == "") { continue; }
            int foreground_pixel = std::stoi(token);
            detection->add_foreground_pixel(foreground_pixel);
          }

          if (detection->get_foreground_pixels().size() > 0 || insert_empty_detections) {
              detection_list.push_back(detection);
          }

          if (detection->get_foreground_pixels().size() > 0){
              ++detections_read;
          }

          std::getline(in_file, line);
          det_label = std::stoi(line);
      }

      in_file.close();
      return detections_read;
    }


  } // end - namespace DetectionUtils
} // end - namespace tvg