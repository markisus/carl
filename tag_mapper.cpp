#include "tag_mapper.h"
#include "factor_graph.hpp"
#include "geometry.h"

namespace carl {

std::array<Eigen::VectorD<4>, 4> make_tag_corners(double side_length) {
    const double l_2 = side_length/2; //
    Eigen::VectorD<4> tl = { -l_2, l_2, 0, 1 };
    Eigen::VectorD<4> tr = { l_2, l_2, 0, 1 };
    Eigen::VectorD<4> br = { l_2, -l_2, 0, 1 };
    Eigen::VectorD<4> bl = { -l_2, -l_2, 0, 1 };
    return { tl, tr, br, bl };
}

double TagMapperConfig::get_tag_side_length(int tag_id) {
        if (tag_side_lengths.count(tag_id)) {
            return tag_side_lengths[tag_id];
        }
        // std::cout << "returning default tag side length " << default_tag_side_length << "\n";
        return default_tag_side_length;
    }

std::array<Eigen::VectorD<4>, 4> TagMapperConfig::get_tag_corners(int tag_id) {
        const double l = get_tag_side_length(tag_id);
        // std::cout << "getting tag corners for length " << l << "\n";
        return make_tag_corners(l);
    }

Eigen::VectorD<4> TagMapperConfig::get_camparams(int camera_id) {
    if (!camparam_priors.count(camera_id)) {
        return camparam_priors[camera_id];
    }
    return default_camparams;
}

struct TagMapperImpl {
    static constexpr uint8_t CAMERA_PARAMS = 0;
    static constexpr uint8_t CAMERA_POSE = 1;
    static constexpr uint8_t TAG_POSE = 2;

    TagMapperConfig config;

    FactorGraph</*factor_dims=*/Dims<16>, /*variable_dims=*/Dims<4, 6>> graph;
    std::unordered_map<int, VariableHandle<4>> camparam_handles;
    std::unordered_map<int, VariableHandle<6>> camera_pose_handles;
    std::unordered_map<int, VariableHandle<6>> tag_pose_handles;

    struct CameraObjectData : public FactorData<16, true> {
        std::array<Eigen::VectorD<4>, 4> tag_points;
        std::array<Eigen::VectorD<2>, 4> image_points;
    };

    bool have_camera(int camera_id) {
        return camparam_handles.count(camera_id);
    };

    void init_camera(int camera_id, const Eigen::VectorD<4>& camparams) {
        assert(!have_camera(camera_id));
        const auto camparam_handle = graph.add_variable(
            CAMERA_PARAMS, camparams, (Eigen::id<4>() * 1e-6).eval());
        camparam_handles.insert({ camera_id,  camparam_handle});
    };

    bool have_tag(int tag_id) {
        return tag_pose_handles.count(tag_id);
    };

    void init_tag(int tag_id, const Eigen::SquareD<4>& tx_world_tag) {
        assert(!have_tag(tag_id));

        auto se3_world_tag = SE3_log(tx_world_tag);
        Eigen::SquareD<6> cov = Eigen::id<6>();
        cov.block<3,3>(0,0) *= 0.5;
        const auto tag_pose_handle = graph.add_variable(
            TAG_POSE,
            se3_world_tag, cov);
        tag_pose_handles.insert({ tag_id, tag_pose_handle });
    };

    bool have_image(int image_id) {
        return camera_pose_handles.count(image_id);
    };

    void init_image(int image_id, const Eigen::SquareD<4>& tx_world_camera) {
        assert(!have_image(image_id));
        auto se3_world_camera = SE3_log(tx_world_camera);
        Eigen::SquareD<6> cov = Eigen::id<6>();
        cov.block<3,3>(0,0) *= 0.5;

        if (camera_pose_handles.empty()) {
            // special case: first camera ever added
            cov *= 1e-9;
        }

        const auto camera_pose_handle = graph.add_variable(
            CAMERA_POSE,
            se3_world_camera, cov);
        camera_pose_handles.insert({ image_id, camera_pose_handle });
    };

    Eigen::SquareD<4> get_camera_pose(int image_id, Eigen::SquareD<6>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(camera_pose_handles[image_id]);
        }
        return se3_exp(graph.get_mean(camera_pose_handles[image_id]));
    }

    Eigen::SquareD<4> get_tag_pose(int tag_id, Eigen::SquareD<6>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(tag_pose_handles[tag_id]);
        }
        return se3_exp(graph.get_mean(tag_pose_handles[tag_id]));
    }

    Eigen::VectorD<4> get_camparams(int camera_id, Eigen::SquareD<4>* covariance = nullptr) {
        if (covariance) {
            *covariance = graph.get_covariance(camparam_handles[camera_id]);
        }
        return graph.get_mean(camparam_handles[camera_id]);
    }

    double update() {
        double total_err2 = 0;
        graph.begin_linearization();
        for (auto [_, data, lin_point, info_matrix, info_vector] :
                 graph.factors.view<
                 CameraObjectData,
                 LinearizationPoint<CameraObjectData::dimension>,
                 InfoMatrix<CameraObjectData::dimension>,
                 InfoVector<CameraObjectData::dimension>>().each()) {

            // std::cout << "tagmapper update factor\n";
            // for (auto& tag_pt : data.tag_points) {
            //     std::cout << "\t " << tag_pt.transpose() << "\n";
            // }
            
            double err2 = camera_project_factor(
                lin_point,
                data.tag_points,
                data.image_points,
                &info_matrix,
                &info_vector);
            total_err2 += err2;
        }
        graph.end_linearization();


        graph.update_factors();
        graph.update_variables();

        return total_err2;
    }

    void add_detection(const int camera_id,
                       const int image_id,
                       const int tag_id,
                       const std::array<Eigen::VectorD<2>, 4>& corners) {
        assert(have_camera(camera_id));
        assert(have_image(image_id));
        assert(have_tag(tag_id));

        const auto camparams_variable = camparam_handles[camera_id];
        const auto camera_pose_variable = camera_pose_handles[image_id];
        const auto tag_pose_variable = tag_pose_handles[tag_id];

        CameraObjectData data;
        data.image_points = corners;
        data.tag_points = config.get_tag_corners(tag_id);
        auto factor = graph.add_factor(data);
        graph.add_edge(camparams_variable, factor, 0);
        graph.add_edge(camera_pose_variable, factor, 4);
        graph.add_edge(tag_pose_variable, factor, 10);
    };
};

TagMapper::TagMapper() {
    impl_ = std::make_unique<TagMapperImpl>();
}

TagMapper::~TagMapper() = default; 

double TagMapper::update() {
    return impl_->update();
}

void TagMapper::add_detection(const int camera_id,
                              const int image_id,
                              const int tag_id,
                              const std::array<Eigen::VectorD<2>, 4>& corners) {
    impl_->add_detection(camera_id, image_id, tag_id, corners);
};

Eigen::SquareD<4> TagMapper::get_camera_pose(int image_id, Eigen::SquareD<6>* cov) {
    return impl_->get_camera_pose(image_id, cov);
};

Eigen::SquareD<4> TagMapper::get_tag_pose(int tag_id, Eigen::SquareD<6>* cov) {
    return impl_->get_tag_pose(tag_id, cov);
};

Eigen::VectorD<4> TagMapper::get_camparams(int camera_id, Eigen::SquareD<4>* cov) {
    return impl_->get_camparams(camera_id, cov);
};

bool TagMapper::have_camera(int camera_id) {
    return impl_->have_camera(camera_id);
};

void TagMapper::init_camera(int camera_id, const Eigen::VectorD<4>& camparams) {
    return impl_->init_camera(camera_id, camparams);
};

bool TagMapper::have_tag(int tag_id) {
    return impl_->have_tag(tag_id);
};

void TagMapper::init_tag(int tag_id, const Eigen::SquareD<4>& tx_world_tag) {
    return impl_->init_tag(tag_id, tx_world_tag);
};

bool TagMapper::have_image(int image_id) {
    return impl_->have_image(image_id);
};

void TagMapper::init_image(int image_id, const Eigen::SquareD<4>& tx_world_camera) {
    return impl_->init_image(image_id, tx_world_camera);
};

TagMapperConfig& TagMapper::config() {
    return impl_->config;
}


}  // carl
