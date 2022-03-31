#include <iostream>
#include <fstream>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

#include "util/eigen_util.h"
#include "tag_mapper_data.h"

namespace carl {
namespace tag_mapper {

Eigen::VectorD<4> load_camparams(const std::filesystem::path& scene_path) {
    std::filesystem::path camera_matrix_path = scene_path / "camera_matrix.txt";
    std::string buf;
    std::ifstream camera_mat_file(camera_matrix_path.string());
    Eigen::VectorD<4> camparams;
    for (int i = 0; i < 2; ++i) {
        std::getline(camera_mat_file, buf);
        const std::vector<absl::string_view> tokens = absl::StrSplit(buf, ' ');
        if (i == 0) {
            // store fx, cx
            bool result = 
                absl::SimpleAtod(tokens[0], &camparams(0)) &&
                absl::SimpleAtod(tokens[2], &camparams(2));
            if(!result) {
                std::cout << "parse error " << buf << "\n";
                exit(-1);
            }
        }
        if (i == 1) {
            // store fy, cy
            bool result = 
                absl::SimpleAtod(tokens[1], &camparams(1)) &&
                absl::SimpleAtod(tokens[2], &camparams(3));
            if(!result) {
                std::cout << "parse error " << buf << "\n";
                exit(-1);
            }
        }
    }
    // std::cout << "Loaded camparams " << camparams.transpose() << "\n";
    return camparams;
}

double Scene::get_tag_side_length(int tag_id) {
    if (tag_side_lengths.count(tag_id)) {
        return tag_side_lengths.at(tag_id);
    }
    return default_tag_side_length;
}

void Scene::load(const std::filesystem::path& scene_path) {
    // load camera matrix
    camparams = load_camparams(scene_path);

    std::string buf;
    
    // load tags
    for (const auto& dir_entry : std::filesystem::directory_iterator{scene_path}) {
        const std::string filename = dir_entry.path().filename().string();
        if (absl::StartsWith(filename, "tags_") && absl::EndsWith(filename, ".txt")) {
            const std::string image_id = filename.substr(/*start=*/5, /*len=*/filename.size() - 9);
            const std::string full_path = dir_entry.path().string();
            // std::cout << "\ttag image_id " << image_id << "\n";

            // BEGIN PARSE TAG FILE ========================================
            std::ifstream tag_file(full_path);
            std::array<Eigen::VectorD<2>, 4> img_points;
            int tag_id;
            for(int lcount = 0; std::getline(tag_file, buf); ++lcount) {
                if (lcount % 5 == 0) {
                    if (absl::SimpleAtoi(buf, &tag_id)) {
                        continue;
                    } else {
                        std::cout << "could not parse tag id " << tag_id << "\n";
                        exit(-1);
                    }                  
                }
                int point_idx = (lcount % 5) - 1;
                const std::vector<absl::string_view> tokens = absl::StrSplit(buf, ' ');
                double x, y;
                bool succ = absl::SimpleAtod(tokens[0], &x) && absl::SimpleAtod(tokens[1], &y);
                if (!succ) {
                    std::cout << "could not parse " << buf << "\n";
                    exit(-1);
                }
                img_points[point_idx] << x, y;
                if (point_idx == 3) {
                    tag_detections[image_id][tag_id] = img_points;
                }
            }
            // END PARSE TAG FILE ========================================

            if (tag_detections[image_id].empty()) {
                std::cout << "Warning: file " <<  full_path << " had no tags\n";
            }
        }
    }

    // load images
    for (const auto& dir_entry : std::filesystem::directory_iterator{scene_path}) {
        std::string filename_stem = dir_entry.path().filename().stem().string(); // chop off image extension
        if (absl::StartsWith(filename_stem, "image_")) {
            std::string image_id = filename_stem.substr(6, filename_stem.size() - 5);
            absl::StripAsciiWhitespace(&image_id);
            image_paths[image_id] = dir_entry.path();
            // std::cout << "scene: installing image " << image_id << " of path " << dir_entry.path() << "\n";
        }
    }

    // load tag side lengths
    for (const auto& dir_entry : std::filesystem::directory_iterator{scene_path}) {
        const std::string filename = dir_entry.path().filename().string();
        if (filename == "tag_side_length.txt") {
            // load tag side length
            std::ifstream tag_file(dir_entry.path().string());
            bool set_default = false;
            for(int lcount = 0; std::getline(tag_file, buf); ++lcount) {
                const std::vector<absl::string_view> tokens = absl::StrSplit(buf, ' ');
                if (tokens.size() == 2) {
                    // special tag side length
                    int tag_id;
                    if (!absl::SimpleAtoi(tokens[0], &tag_id)) {
                        std::cout << "could not parse " << tokens[0] << " as tag id from line " << buf << "\n";
                        exit(-1);
                    }

                    double tag_side_length;
                    if (!absl::SimpleAtod(tokens[1], &tag_side_length)) {
                        std::cout << "could not parse " << tokens[1] << " as side length from " << buf << "\n";
                        exit(-1);
                    }
                    tag_side_lengths[tag_id] = tag_side_length;
                } else if (tokens.size() == 1) {
                    if (set_default) {
                        std::cout << "encountered duplicate default tag side length " << buf << "\n";
                        std::cout << "previously set as " << default_tag_side_length << "\n";
                        exit(-1);
                    }
                    double tag_side_length;
                    if (!absl::SimpleAtod(buf, &tag_side_length)) {
                        std::cout << "could not parse default tag side length from " << buf << "\n";
                        exit(-1);
                    }
                    default_tag_side_length = tag_side_length;
                    set_default = true;
                } else {
                    std::cout << "unexpected data format '" << buf << "'\n";
                    std::cout << "expecting either one item for default side length or two items separated by space for tag_id, side length" << "\n";
                    exit(-1);
                }
            }
            break;
        }
    }

    std::cout << "Loaded tag side lengths" << "\n";
    for (auto [tag_id, length] : tag_side_lengths) {
        std::cout << "\ttag " << tag_id << ": " << length << "\n";
        assert(get_tag_side_length(tag_id) == length);
    }
}

}  // tag_mapper

}  // carl
