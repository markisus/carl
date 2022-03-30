#include <cassert>

#include "tag_mapper.h"
#include "factor_graph.hpp"
#include "geometry.h"
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
    static constexpr uint8_t CAMERA_PARAMS = 0;
    static constexpr uint8_t CAMERA_POSE = 1;
    static constexpr uint8_t TAG_POSE = 2;

    int RELIN_PERIOD = 50;
    int time_until_relinearize = RELIN_PERIOD;

    bool scene_initted = false;
    Scene scene;

    FactorGraph</*factor_dims=*/Dims<16>, /*variable_dims=*/Dims<4, 6>> graph = {};
    std::unordered_map<std::string, VariableHandle<4>> camparam_handles;
    std::unordered_map<std::string, VariableHandle<6>> camera_pose_handles;
    std::unordered_map<int, VariableHandle<6>> tag_pose_handles;

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
        return camparam_handles.count(camera_id);
    };

    void init_camera(const std::string& camera_id) {
        if (have_camera(camera_id)) {
            std::cout << "already had camera " << camera_id << "\n";
            exit(-1);
        }
        const auto camparam_handle = graph.add_variable(
            CAMERA_PARAMS, scene.camparams, (Eigen::id<4>() * 1e-6).eval());
        camparam_handles.insert({ camera_id,  camparam_handle});
        graph.set_display_string(camparam_handle, absl::StrFormat("camparams:%s", camera_id));
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
        const auto tag_pose_handle = graph.add_variable(
            TAG_POSE,
            se3_world_tag, cov);
        tag_pose_handles.insert({ tag_id, tag_pose_handle });

        graph.set_display_string(tag_pose_handle, absl::StrFormat("tag:%d", tag_id));
        _tag_list.push_back(tag_id);
    };

    bool have_image(const std::string& image_id) {
        return camera_pose_handles.count(image_id);
    };

    void init_image(const std::string& image_id, const Eigen::MatrixD<4>& tx_world_camera) {
        assert(!have_image(image_id));
        auto se3_world_camera = SE3_log(tx_world_camera);
        Eigen::MatrixD<6> cov = Eigen::id<6>();
        cov.block<3,3>(0,0) *= 0.5;
        cov *= 0.005;

        const auto camera_pose_handle = graph.add_variable(
            CAMERA_POSE,
            se3_world_camera, cov);
        camera_pose_handles.insert({ image_id, camera_pose_handle });

        graph.set_display_string(camera_pose_handle, absl::StrFormat("img:%s", image_id));

        _image_list.push_back(image_id);
    };

    const std::vector<std::string>& image_list() {
        return _image_list;
    }

    const std::vector<int>& tag_list() {
        return _tag_list;
    }

    Eigen::MatrixD<4> get_camera_pose(const std::string& image_id, Eigen::MatrixD<6>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(camera_pose_handles[image_id]);
        }
        return se3_exp(graph.get_mean(camera_pose_handles[image_id]));
    }

    Eigen::MatrixD<4> get_tag_pose(int tag_id, Eigen::MatrixD<6>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(tag_pose_handles[tag_id]);
        }
        return se3_exp(graph.get_mean(tag_pose_handles[tag_id]));
    }

    Eigen::VectorD<4> get_camparams(Eigen::MatrixD<4>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(camparam_handles["default_camera_id"]);
        }
        return graph.get_mean(camparam_handles["default_camera_id"]);
    }

    double update_layout(bool* converged) {
        double error = graph.update_layout();
        if (converged != nullptr) {
            *converged = graph.layout_converged;
        }
        return error;
    }

    void visit_variable_layout(void(*visiter)(LayoutData*, VariableError*, void*), void* user_data) {
        graph.visit_variable_layout(visiter, user_data);
    }    

    void visit_factor_layout(void(*visiter)(LayoutData*, FactorError*, void*), void* user_data) {
        graph.visit_factor_layout(visiter, user_data);
    }

    void visit_edge_layout(void(*visiter)(LayoutData*, LayoutData*, EdgeResidual*, void*), void* user_data) {
        graph.visit_edge_layout(visiter, user_data);
    }

    double relinearize() {
        double total_err2 = 0;
        graph.begin_linearization();
        for (auto [_, factor_error, data, lin_point, info_matrix, info_vector] :
                 graph.factors.view<
                 FactorNeedsRelinearize,
                 FactorError,
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

        const auto camparams_variable = camparam_handles["default_camera_id"];
        const auto camera_pose_variable = camera_pose_handles[image_id];
        const auto tag_pose_variable = tag_pose_handles[tag_id];

        CameraObjectData data;
        data.image_points = scene.tag_detections[image_id][tag_id];

        const double tag_len = scene.get_tag_side_length(tag_id);
        data.tag_points = make_tag_corners(tag_len);
        auto factor = graph.add_factor(data);
        graph.set_display_string(factor, absl::StrFormat("img:%s, tag:%d", image_id, tag_id));
        
        graph.add_edge(camparams_variable, factor, 0);
        graph.add_edge(camera_pose_variable, factor, 4);
        graph.add_edge(tag_pose_variable, factor, 10);

        Eigen::VectorD<16>& lin_point = graph.factors.get<LinearizationPoint<16>>(factor);
        Eigen::MatrixD<16>& jtj = graph.factors.get<InfoMatrix<16>>(factor);
        Eigen::VectorD<16>& rtj = graph.factors.get<InfoVector<16>>(factor);
        double rtr = camera_project_factor(lin_point, data.tag_points, data.image_points, &jtj, &rtj);
        graph.factors.get<FactorError>(factor).offset = rtr;

        graph.add_edges_finish();
    };

    std::array<double, 2> layout_size() {
        return {graph.layout_width, graph.layout_height};
    }

    double max_factor_change() {
        return graph.max_factor_change;
    }
};

TagMapper::TagMapper() {
    impl_ = std::make_unique<TagMapperImpl>();
}

TagMapper::~TagMapper() = default; 

double TagMapper::update() {
    return impl_->update();
}

double TagMapper::max_factor_change() {
    return impl_->max_factor_change();
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

Eigen::MatrixD<4> TagMapper::get_camera_pose(const std::string& image_id, Eigen::MatrixD<6>* cov) {
    return impl_->get_camera_pose(image_id, cov);
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

void TagMapper::visit_variable_layout(void(*visiter)(LayoutData*, VariableError*, void*), void* user_data) {
    return impl_->visit_variable_layout(visiter, user_data);
};

void TagMapper::visit_edge_layout(void(*visiter)(LayoutData*, LayoutData*, EdgeResidual*, void*), void* user_data) {
    return impl_->visit_edge_layout(visiter, user_data);
};

void TagMapper::visit_factor_layout(void(*visiter)(LayoutData*, FactorError*, void*), void* user_data) {
    return impl_->visit_factor_layout(visiter, user_data);
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
