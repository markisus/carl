#pragma once
#include <unordered_map>
#include <array>
#include <memory>
#include <filesystem>

#include "Eigen/Dense"
#include "eigen_util.h"
#include "tag_mapper_data.h"

namespace carl {

struct LayoutData;
struct VariableError;
struct FactorError;
struct EdgeResidual;

namespace tag_mapper {

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

    bool have_image(const std::string& image_id);
    void init_image(const std::string& image_id, const Eigen::MatrixD<4>& tx_world_camera);
    
    double update();
    double relinearize();
    double update_layout(bool* converged = nullptr);
    void add_detection(const std::string& image_id,
                       const int tag_id);

    void visit_variable_layout(void(*visiter)(LayoutData*, VariableError*, void*), void* user_data);
    void visit_factor_layout(void(*visiter)(LayoutData*, FactorError*, void*), void* user_data);
    void visit_edge_layout(void(*visiter)(LayoutData*, LayoutData*, EdgeResidual*, void*), void* user_data);

    Eigen::MatrixD<4> get_camera_pose(const std::string& image_id, Eigen::MatrixD<6>* covariance = nullptr);
    Eigen::MatrixD<4> get_tag_pose(int tag_id, Eigen::MatrixD<6>* covariance = nullptr);
    Eigen::VectorD<4> get_camparams(Eigen::MatrixD<4>* covariance = nullptr);

    std::array<double, 2> layout_size();
    double max_factor_change();

    std::unique_ptr<TagMapperImpl> impl_;
};

}

}  // carl
