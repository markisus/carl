#include <iostream>
#include <cstdint>

#include "factor_graph.hpp"

using namespace carl;

// get full covariance matrix
// use schur complement to

template <int rows, int cols>
Eigen::Matrix<double, rows*cols, 1> flatten(const Eigen::Matrix<double, rows, cols>& mat) {
    Eigen::Matrix<double, rows*cols, 1> result;
    for (int c = 0; c < mat.cols(); ++c) {
        result.segment<rows>(rows*c) = mat.col(c);
    }
    return result;
}

template <int num_pts>
Eigen::Matrix<double, 3, num_pts> unhomogenize(const Eigen::Matrix<double, 4, num_pts>& points) {
    Eigen::Matrix<double, 3, num_pts> result;
    result = points.block<3,num_pts>(0,0);
    for (int c = 0; c < mat.cols(); ++c) {
        result.col(c) /= points(3,c);
    }
    return result;
}

template <int num_pts>
Eigen::Matrix<double, 4, num_pts> homogenize(const Eigen::Matrix<double, 3, num_pts>& points) {
    Eigen::Matrix<double, 4, num_pts> result;
    result.block<3,num_pts>(0,0) = points;
    for (int c = 0; c < mat.cols(); ++c) {
        result(c,3) = 1;
    }
    return result;
}

template <int rows, int cols>
Eigen::Matrix<double, rows-1, cols> delete_last_row(const Eigen::Matrix<double, rows, cols>& mat) {
    Eigen::Matrix<double, rows-1, cols> result;
    result = mat.block<rows-1, cols>(0,0);
    return result;
}

// forward model of camera params
template <int num_kps>
Eigen::Matrix<double, 2, num_kps> project_keypoints(
    const Eigen::Matrix<double, 4, num_kps>& object_points,
    const Eigen::Matrix<double, 4, 4>& camera_matrix,
    const Eigen::Matrix<double, 4, 4>& tx_world_camera,
    const Eigen::Matrix<double, 4, 4>& tx_world_object) {

    auto tx_camera_world = SE3_inv(tx_world_camera + dcamera);
    auto camera_points = tx_camera_world * tx_world_object * object_points;
    auto image_points = camera_matrix * camera_points;
    return delete_last_row(unhomogenize(image_points));

    // jacobians -
    // dout / dcam
    // dout / dobject
    
}

template <int num_kps>
void keypoints_factor(
    const Eigen::Matrix<double, 4, num_kps>& object_points,    
    const Eigen::Matrix<double, 4, 1>& camera_params_mean,
    const Eigen::Matrix<double, 4, 4>& camera_params_cov,
    const Eigen::Matrix<double, 6, 1>& se3_world_camera_mean,
    const Eigen::Matrix<double, 6, 6>& se3_world_camera_cov,
    const Eigen::Matrix<double, 6, 1>& se3_world_object_mean,
    const Eigen::Matrix<double, 6, 6>& se3_world_object_cov) {

    // factorize camera params cov
    // factorize se3 world_camera_cov 
    // factorize se3 world_object_cov 
    
}

void go() {
    // for i in cam sigma points
    // project_keypoints(cam_sig_i, tx_world_cam,       tx_world_obj      )
    // project_keypoints(cam      , tx_world_cam_sig_i, tx_world_obj      )
    // project_keypoints(cam      , tx_world_cam,       tx_world_obj_sig_i)
    // =============
    // [[ xi xi.t ]]
    // =============
    // fi fi.t => sum to covariance fi
    // [ x | f ] 
    // 
}



int main(int argc, char *argv[])
{
    FactorGraph graph;

    Mean<2> mean1;
    Mean<2> mean2;
    Mean<2> mean3;
    mean1 << 1, 0;
    mean2 << 0, 1;
    mean3 << -1, -1;
    auto covariance1 = Covariance<2>::Identity().eval();
    auto covariance2 = Covariance<2>::Identity().eval();
    auto covariance3 = Covariance<2>::Identity().eval();

    covariance1 *= 1e2;
    covariance2 *= 1e2;
    covariance3 *= 1e-2;

    const auto variable1 = add_variable(mean1, covariance1, graph);
    const auto variable2 = add_variable(mean2, covariance2, graph);
    const auto variable3 = add_variable(mean3, covariance3, graph);


    Eigen::Matrix<double, 2, 4> J;
    J.block<2,2>(0,0).setIdentity();
    J.block<2,2>(0,J.cols()/2).setIdentity();
    J.block<2,2>(0,J.cols()/2) *= -1;
    InfoMatrix<4> info_mat = J.transpose() * J;
    InfoVector<4> info_vec = InfoVector<4>::Zero();

    {
        const auto factor = add_factor(info_vec, info_mat, graph);
        add_edge<2>(variable1, factor, 0, graph);
        add_edge<2>(variable2, factor, info_vec.size()/2, graph);
    }

    {
        const auto factor = add_factor(info_vec, info_mat, graph);
        add_edge<2>(variable1, factor, 0, graph);
        add_edge<2>(variable3, factor, info_vec.size()/2, graph);
    }
    
    for (size_t i = 1; i < 10; ++i) {
        std::cout << "it " << i << "===============\n";
        update_factors<2>(graph);
        update_factors_finish<4>(graph);
        update_variables<2>(graph);
        // pull out the mean
        std::cout << "\tvar1 " << graph.variables.get<Mean<2>>(variable1).transpose() << "\n";
        std::cout << "\tvar2 " << graph.variables.get<Mean<2>>(variable2).transpose() << "\n";
        std::cout << "\tvar3 " << graph.variables.get<Mean<2>>(variable3).transpose() << "\n";
        
    }

    return 0;
}
