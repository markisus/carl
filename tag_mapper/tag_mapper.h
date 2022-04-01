#pragma once
#include <unordered_map>
#include <array>
#include <memory>
#include <filesystem>

#include "Eigen/Dense"
#include "util/eigen_util.h"
#include "tag_mapper_data.h"
#include "entt/fwd.hpp"

namespace carl {

namespace tag_mapper {

constexpr uint8_t VAR_TYPE_CAMERA_PARAMS = 0;
constexpr uint8_t VAR_TYPE_IMAGE_POSE = 1;
constexpr uint8_t VAR_TYPE_TAG_POSE = 2;
constexpr const char* VAR_TYPE_STRS[3] = { "Cam Params", "Image Pose", "Tag Pose"};

struct FactorViz {
    Eigen::VectorD<2>* position;
    std::string* image_id;    
    double error;
    int age;
    int tag_id;
};

struct VariableViz {
    Eigen::VectorD<2>* position;
    std::string* id;
    double error;
    int age;
    uint8_t type;
};

struct EdgeViz {
    Eigen::VectorD<2>* factor_position;
    Eigen::VectorD<2>* variable_position;
    double to_factor_residual;
    double to_variable_residual;
};

// pimpl idion
struct TagMapperImpl; 
struct TagMapper {
    TagMapper();
    ~TagMapper();

    const std::vector<std::string>& image_list();
    const std::vector<int>& tag_list();

    void load_scene(const std::filesystem::path& scene_path);
    Scene& get_scene();

    bool have_tag(int tag_id);
    void init_tag(int tag_id, const Eigen::MatrixD<4>& tx_world_tag);
    int entt_to_tag();

    bool have_image(const std::string& image_id);
    void init_image(const std::string& image_id, const Eigen::MatrixD<4>& tx_world_camera);
    std::string entt_to_image();
    
    double update();
    double relinearize();
    double update_layout(bool* converged = nullptr);
    void add_detection(const std::string& image_id,
                       const int tag_id);

    void visit_variable_viz(void(*visiter)(const VariableViz&, void*), void* user_data);
    void visit_factor_viz(void(*visiter)(const FactorViz&, void*), void* user_data);
    void visit_edge_viz(void(*visiter)(const EdgeViz&, void*), void* user_data);

    Eigen::MatrixD<4> get_image_pose(const std::string& image_id, Eigen::MatrixD<6>* covariance = nullptr);
    Eigen::MatrixD<4> get_tag_pose(int tag_id, Eigen::MatrixD<6>* covariance = nullptr);
    Eigen::VectorD<4> get_camparams(Eigen::MatrixD<4>* covariance = nullptr);

    std::array<double, 2> layout_size();

    std::unique_ptr<TagMapperImpl> impl_;
};

}

}  // carl
