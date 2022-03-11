#include "factor_graph.hpp"
#include "eigen_util.h"
#include "geometry.h"

using namespace carl;

int main(int argc, char *argv[])
{
    // create a random object
    int num_points = 10;
    std::vector<Eigen::VectorD<4>> object_points;
    std::vector<Eigen::VectorD<2>> image_points;
    for (int i = 0; i < num_points; ++i) {
        object_points.push_back(Eigen::random_homog());
    }

    auto camparams = Eigen::random_vec<4>();
    std::cout << "camparams " << camparams.transpose() << "\n";

    // create a random camera position
    auto se3_world_camera = Eigen::random_vec<6>();
    auto tx_camera_world = se3_exp(-se3_world_camera);
    std::cout << "se3_world_camera " << se3_world_camera.transpose() << "\n";

    // create a random object position
    auto se3_world_object = Eigen::random_vec<6>();
    auto tx_world_object = se3_exp(se3_world_object);
    std::cout << "se3_world_object " << se3_world_object.transpose() << "\n";

    // project object points into the image
    for (const auto& object_point : object_points) {
        auto image_point = camera_project(
            camparams,
            tx_camera_world,
            se3_world_camera,
            tx_world_object,
            se3_world_object,
            object_point);
        image_points.push_back(image_point);
    }

    // create a random object perturb
    auto object_perturb = (Eigen::random_vec<6>() * 0.005).eval();
    auto se3_world_object_perturbed = (se3_world_object + object_perturb).eval();
    auto tx_world_object_perturbed = se3_exp(se3_world_object_perturbed);
    std::cout << "se3_world_object_perturbed " << se3_world_object_perturbed.transpose() << "\n";        

    FactorGraph</*factor_dims=*/Dims<4, 6, 16>, /*variable_dims=*/Dims<4, 6>> graph;

    // variable types
    constexpr uint8_t CAMERA_POSE = 0;
    constexpr uint8_t OBJECT_POSE = 1;
    constexpr uint8_t CAMERA_PARAMS = 2;

    // factor types
    constexpr uint8_t CAMERA_OBJECT = 0;

    // add camera params of known params (low variance prior)
    const auto camparams_variable = graph.add_variable(CAMERA_PARAMS, camparams, (Eigen::id<4>() * 1e-6).eval());

    // add camera pose of known position (low variance prior)
    const auto camera_variable = graph.add_variable(CAMERA_POSE, se3_world_camera, (Eigen::id<6>() * 1e-6).eval());

    // add object pose of unknown position (large variance prior)
    const auto object_variable = graph.add_variable(OBJECT_POSE, se3_world_object_perturbed, (Eigen::id<6>() * 1e-1).eval());

    // connect with a camera object factor
    const auto factor = graph.add_factor<16>(CAMERA_OBJECT, /*nonlinear=*/true);
    graph.add_edge(camparams_variable, factor, 0);
    graph.add_edge(camera_variable, factor, 4);
    graph.add_edge(object_variable, factor, 10);

    // relinearize factor
    std::vector<FactorHandle<16>> camera_object_factors = { factor };

    graph.regularizer = 1e9;
    for (int i = 0; i < 4; ++i) {
        std::cout << "iteration " << i << " ==============\n";
        graph.begin_linearization();
        for (auto factor : camera_object_factors) {
            auto lin_point = graph.get_linearization_point(factor);
            // std::cout << "the lin point was... " << "\n";
            // std::cout << "\t" << lin_point.head<4>().transpose() << "\n";
            // std::cout << "\t" << lin_point.segment<6>(4).transpose() << "\n";
            // std::cout << "\t" << lin_point.segment<6>(10).transpose() << "\n";

            auto* factor_info_mat = graph.get_factor_info_matrix(factor);
            auto* factor_info_vec = graph.get_factor_info_vector(factor);
            double error2 = camera_project_factor(
                lin_point,
                object_points,
                image_points,
                factor_info_mat,
                factor_info_vec);
            std::cout << "projection error2 " << error2 << "\n";
        }
        graph.end_linearization();
        
        graph.update_factors();
        graph.update_variables();
        // std::cout << "camparams " << graph.get_mean(camparams_variable).transpose() << "\n";
        // std::cout << "camera pose " << graph.get_mean(camera_variable).transpose() << "\n";
        std::cout << "object pose " << graph.get_mean(object_variable).transpose() << "\n";
    }

    return 0;
}
