#pragma once
#include "Eigen/Dense"
#include "eigen_util.h"
#include <unordered_map>
#include <array>
#include <memory>

namespace carl {

struct TagMapperConfig {
    Eigen::VectorD<4> default_camparams;
    double default_tag_side_length;
    std::unordered_map<int, Eigen::VectorD<4>> camparam_priors;
    std::unordered_map<int, double> tag_side_lengths;

    double get_tag_side_length(int tag_id);
    std::array<Eigen::VectorD<4>, 4> get_tag_corners(int tag_id);
    Eigen::VectorD<4> get_camparams(int camera_id);
};


// pimpl idion
struct TagMapperImpl; 
struct TagMapper {
    TagMapper();
    ~TagMapper();

    TagMapperConfig& config();

    bool have_camera(int camera_id);
    void init_camera(int camera_id, const Eigen::VectorD<4>& camparams);

    bool have_tag(int tag_id);
    void init_tag(int tag_id, const Eigen::SquareD<4>& tx_world_tag);

    bool have_image(int image_id);
    void init_image(int image_id, const Eigen::SquareD<4>& tx_world_camera);
    
    double update();
    void add_detection(const int camera_id,
                       const int image_id,
                       const int tag_id,
                       const std::array<Eigen::VectorD<2>, 4>& corners);

    
    Eigen::SquareD<4> get_camera_pose(int image_id, Eigen::SquareD<6>* covariance = nullptr);
    Eigen::SquareD<4> get_tag_pose(int tag_id, Eigen::SquareD<6>* covariance = nullptr);
    Eigen::VectorD<4> get_camparams(int camera_id, Eigen::SquareD<4>* covariance = nullptr);    

    std::unique_ptr<TagMapperImpl> impl_;
};

}  // carl
