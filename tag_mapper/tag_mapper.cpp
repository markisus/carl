#include <cassert>

#include "tag_mapper.h"
#include "factor_graph/factor_graph.hpp"
#include "math/geometry.h"
#include "absl/strings/str_format.h"

namespace carl {

std::array<Eigen::VectorD<4>, 4> make_tag_corners(double side_length) {
    const double l_2 = side_length/2; //
    Eigen::VectorD<4> tl = { -l_2, l_2, 0, 1 };
    Eigen::VectorD<4> tr = { l_2, l_2, 0, 1 };
    Eigen::VectorD<4> br = { l_2, -l_2, 0, 1 };
    Eigen::VectorD<4> bl = { -l_2, -l_2, 0, 1 };
    return { tl, tr, br, bl };
}

namespace tag_mapper {

struct TagMapperImpl {
    int RELIN_PERIOD = 5;
    int time_until_relinearize = RELIN_PERIOD;

    bool scene_initted = false;
    Scene scene;

    FactorGraph</*factor_dims=*/Dims<16>, /*variable_dims=*/Dims<4, 6>> graph = {};

    std::unordered_map<std::string, VariableHandle<4>> camera_handles;
    std::unordered_map<std::string, VariableHandle<6>> image_pose_handles;
    std::unordered_map<int, VariableHandle<6>> tag_pose_handles;

    // a variable may be a image pose, tag pose, or camera params
    std::unordered_map<entt::entity, uint8_t> entt_to_type;
    std::unordered_map<entt::entity, std::string> entt_to_variable_id;

    std::unordered_map<entt::entity, int> factor_to_tag_id;
    std::unordered_map<entt::entity, std::string> factor_to_image_id;

    struct CameraObjectData : public FactorData<16, true> {
        std::array<Eigen::VectorD<4>, 4> tag_points;
        std::array<Eigen::VectorD<2>, 4> image_points;
    };

    std::vector<std::string> _image_list;
    std::vector<int> _tag_list;

    void load_scene(const std::filesystem::path& scene_path) {
        assert(!scene_initted);
        scene.load(scene_path);
        init_camera("default_camera_id");
        scene_initted = true;
    }

    Scene& get_scene() {
        assert(scene_initted);
        return scene;
    }

    bool have_camera(const std::string& camera_id) {
        return camera_handles.count(camera_id);
    };

    void init_camera(const std::string& camera_id) {
        if (have_camera(camera_id)) {
            std::cout << "already had camera " << camera_id << "\n";
            exit(-1);
        }
        const auto camera_handle = graph.add_variable(scene.camparams, (Eigen::id<4>() * 1e-6).eval());
        camera_handles.insert({ camera_id,  camera_handle});
        entt_to_variable_id.insert({ camera_handle.entity, camera_id });
        entt_to_type.insert({ camera_handle.entity, VAR_TYPE_CAMERA_PARAMS });
    };

    bool have_tag(int tag_id) {
        return tag_pose_handles.count(tag_id);
    };

    void init_tag(int tag_id, const Eigen::MatrixD<4>& tx_world_tag) {
        assert(!have_tag(tag_id));

        auto se3_world_tag = SE3_log(tx_world_tag);
        Eigen::MatrixD<6> cov = Eigen::id<6>();
        cov.block<3,3>(0,0) *= 0.5;
        cov *= 0.005;
        const auto tag_pose_handle = graph.add_variable(se3_world_tag, cov);
        tag_pose_handles.insert({ tag_id, tag_pose_handle });
        entt_to_variable_id.insert({ tag_pose_handle.entity, std::to_string(tag_id) });
        entt_to_type.insert({ tag_pose_handle.entity, VAR_TYPE_TAG_POSE });
        _tag_list.push_back(tag_id);
    };

    bool have_image(const std::string& image_id) {
        return image_pose_handles.count(image_id);
    };

    void init_image(const std::string& image_id, const Eigen::MatrixD<4>& tx_world_camera) {
        assert(!have_image(image_id));
        auto se3_world_camera = SE3_log(tx_world_camera);
        Eigen::MatrixD<6> cov = Eigen::id<6>();
        cov.block<3,3>(0,0) *= 0.5;
        cov *= 0.005;

        const auto image_pose_handle = graph.add_variable(se3_world_camera, cov);
        image_pose_handles.insert({ image_id, image_pose_handle });
        entt_to_variable_id.insert({ image_pose_handle.entity, image_id });
        entt_to_type.insert({ image_pose_handle.entity, VAR_TYPE_IMAGE_POSE });
        _image_list.push_back(image_id);
    };

    const std::vector<std::string>& image_list() {
        return _image_list;
    }

    const std::vector<int>& tag_list() {
        return _tag_list;
    }

    Eigen::MatrixD<4> get_image_pose(const std::string& image_id, Eigen::MatrixD<6>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(image_pose_handles[image_id]);
        }
        return se3_exp(graph.get_mean(image_pose_handles[image_id]));
    }

    Eigen::MatrixD<4> get_tag_pose(int tag_id, Eigen::MatrixD<6>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(tag_pose_handles[tag_id]);
        }
        return se3_exp(graph.get_mean(tag_pose_handles[tag_id]));
    }

    Eigen::VectorD<4> get_camparams(Eigen::MatrixD<4>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(camera_handles["default_camera_id"]);
        }
        return graph.get_mean(camera_handles["default_camera_id"]);
    }

    double update_layout(bool* converged) {
        double error = graph.update_layout();
        if (converged != nullptr) {
            *converged = graph.layout_converged;
        }
        return error;
    }

    void visit_variable_viz(void(*visiter)(const VariableViz&, void*), void* user_data) {
        for (auto [variable, layout, variable_summary] : graph.variables.view<LayoutData, VariableSummary>().each()) {
            VariableViz viz;
            viz.id = &entt_to_variable_id.at(variable);            
            viz.position = &layout.position;
            viz.error = variable_summary.prior_error;
            viz.age = variable_summary.age;
            viz.type = entt_to_type.at(variable);
            visiter(viz, user_data);
        }
    }

    void visit_factor_viz(void(*visiter)(const FactorViz&, void*), void* user_data) {
        for (auto [factor, layout, factor_summary] : graph.factors.view<LayoutData, FactorSummary>().each()) {
            FactorViz viz;
            viz.image_id = &factor_to_image_id.at(factor);            
            viz.position = &layout.position;
            viz.error = factor_summary.delta + factor_summary.offset;
            viz.age = factor_summary.age;
            viz.tag_id = factor_to_tag_id.at(factor);
            visiter(viz, user_data);
        }
    }

    void visit_edge_viz(void(*visiter)(const EdgeViz&, void*), void* user_data) {
        for (auto [edge, factor_connection, variable_connection, residual] :
                 graph.edges.view<FactorConnection, VariableConnection, EdgeResidual>().each()) {
            auto& v_layout = graph.variables.get<LayoutData>(variable_connection.variable);
            auto& f_layout = graph.factors.get<LayoutData>(factor_connection.factor);
            EdgeViz viz;
            viz.variable_position = &v_layout.position;
            viz.factor_position = &f_layout.position;
            viz.to_factor_residual = residual.to_factor;
            viz.to_variable_residual = residual.to_variable;
            visiter(viz, user_data);
        }
    }

    double relinearize() {                            
        double total_err2 = 0;
        graph.begin_linearization();
        for (auto [_, factor_error, data, lin_point, info_matrix, info_vector] :
                 graph.factors.view<
                 FactorNeedsRelinearize,
                 FactorSummary,
                 CameraObjectData,
                 LinearizationPoint<CameraObjectData::dimension>,
                 InfoMatrix<CameraObjectData::dimension>,
                 InfoVector<CameraObjectData::dimension>>().each()) {
            double err2 = camera_project_factor(
                lin_point,
                data.tag_points,
                data.image_points,
                &info_matrix,
                &info_vector);
            factor_error.offset = err2;
            total_err2 += err2;
        }
        graph.end_linearization();
        return total_err2;
    }

    double update() {
        graph.update_async();
        time_until_relinearize -= 1;
        if (time_until_relinearize <= 0) {
            time_until_relinearize = RELIN_PERIOD;
            relinearize();
        }
        return 0;
    }

    void add_detection(const std::string& image_id,
                       const int tag_id) {
        assert(scene_initted);
        assert(have_image(image_id));
        assert(have_tag(tag_id));

        const auto camera_variable = camera_handles["default_camera_id"];
        const auto image_pose_variable = image_pose_handles[image_id];
        const auto tag_pose_variable = tag_pose_handles[tag_id];

        CameraObjectData data;
        data.image_points = scene.tag_detections[image_id][tag_id];

        const double tag_len = scene.get_tag_side_length(tag_id);
        data.tag_points = make_tag_corners(tag_len);
        auto factor = graph.add_factor(data);
        
        graph.add_edge(camera_variable, factor, 0);
        graph.add_edge(image_pose_variable, factor, 4);
        graph.add_edge(tag_pose_variable, factor, 10);

        Eigen::VectorD<16>& lin_point = graph.factors.get<LinearizationPoint<16>>(factor);
        Eigen::MatrixD<16>& jtj = graph.factors.get<InfoMatrix<16>>(factor);
        Eigen::VectorD<16>& rtj = graph.factors.get<InfoVector<16>>(factor);
        double rtr = camera_project_factor(lin_point, data.tag_points, data.image_points, &jtj, &rtj);
        graph.factors.get<FactorSummary>(factor).offset = rtr;

        graph.add_edges_finish();

        factor_to_image_id.insert({ factor.entity, image_id });
        factor_to_tag_id.insert({ factor.entity, tag_id });
    };

    std::array<double, 2> layout_size() {
        return {graph.layout_width, graph.layout_height};
    }
};

TagMapper::TagMapper() {
    impl_ = std::make_unique<TagMapperImpl>();
}

TagMapper::~TagMapper() = default; 

double TagMapper::update() {
    return impl_->update();
}

double TagMapper::relinearize() {
    return impl_->relinearize();
}

void TagMapper::load_scene(const std::filesystem::path& scene_path) {
    impl_->load_scene(scene_path);
};

void TagMapper::add_detection(const std::string& image_id,
                              const int tag_id) {
    impl_->add_detection(image_id, tag_id);
};

Eigen::MatrixD<4> TagMapper::get_image_pose(const std::string& image_id, Eigen::MatrixD<6>* cov) {
    return impl_->get_image_pose(image_id, cov);
};

Eigen::MatrixD<4> TagMapper::get_tag_pose(int tag_id, Eigen::MatrixD<6>* cov) {
    return impl_->get_tag_pose(tag_id, cov);
};

Eigen::VectorD<4> TagMapper::get_camparams(Eigen::MatrixD<4>* cov) {
    return impl_->get_camparams(cov);
};

bool TagMapper::have_tag(int tag_id) {
    return impl_->have_tag(tag_id);
};

Scene& TagMapper::get_scene() {
    return impl_->get_scene();
};

void TagMapper::init_tag(int tag_id, const Eigen::MatrixD<4>& tx_world_tag) {
    return impl_->init_tag(tag_id, tx_world_tag);
};

bool TagMapper::have_image(const std::string& image_id) {
    return impl_->have_image(image_id);
};

double TagMapper::update_layout(bool* converged) {
    return impl_->update_layout(converged);
};

void TagMapper::init_image(const std::string& image_id, const Eigen::MatrixD<4>& tx_world_camera) {
    return impl_->init_image(image_id, tx_world_camera);
};

void TagMapper::visit_variable_viz(void(*visiter)(const VariableViz&, void*), void* user_data) {
    return impl_->visit_variable_viz(visiter, user_data);
};

void TagMapper::visit_edge_viz(void(*visiter)(const EdgeViz&, void*), void* user_data) {
    return impl_->visit_edge_viz(visiter, user_data);
};

void TagMapper::visit_factor_viz(void(*visiter)(const FactorViz&, void*), void* user_data) {
    return impl_->visit_factor_viz(visiter, user_data);
}

const std::vector<std::string>& TagMapper::image_list() {
    return impl_->image_list();
}

const std::vector<int>& TagMapper::tag_list() {
    return impl_->tag_list();
}


std::array<double,2> TagMapper::layout_size() {
    return impl_->layout_size();
}


}  // tag_mapper


}  // carl
