#pragma once

#include <filesystem>
#include <map>
#include <unordered_map>
#include "eigen_util.h"

namespace carl {

namespace tag_mapper {

struct Scene {
    // image id -> tag id -> pixel detections
    std::map<std::string, std::map<int, std::array<Eigen::VectorD<2>, 4>>> tag_detections;
    Eigen::VectorD<4> camparams;
    std::map<std::string, std::filesystem::path> image_paths;

    double default_tag_side_length = 0.030;
    std::unordered_map<int, double> tag_side_lengths;

    double get_tag_side_length(int tag_id);

    void load(const std::filesystem::path& scene_path);
};

}

}  // Namespace
