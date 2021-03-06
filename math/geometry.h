#pragma once

#include "Eigen/Dense"
#include "util/eigen_util.h"

namespace carl {

Eigen::Matrix<double, 3, 3> so3_vec_to_mat(const Eigen::Matrix<double, 3, 1>& vec);

Eigen::Matrix<double, 3, 3> SO3_left_jacobian_inv(const Eigen::Matrix<double, 3, 1>& so3);
Eigen::Matrix<double, 3, 3> SO3_left_jacobian(const Eigen::Matrix<double, 3, 1>& so3);

Eigen::Matrix<double, 4, 4> se3_exp(const Eigen::Matrix<double, 6, 1>& se3);
Eigen::Matrix<double, 4, 4> se3_vec_to_mat(const Eigen::Matrix<double, 6, 1>& se3);
Eigen::Matrix<double, 6, 1> se3_mat_to_vec(const Eigen::Matrix<double, 4, 4>& se3);

Eigen::VectorD<6> SE3_log(const Eigen::Matrix4d& SE3_element);

Eigen::Matrix<double, 2, 1> apply_camera_matrix(
    const Eigen::Matrix<double, 4, 1>& fxfycxcy,
    const Eigen::Matrix<double, 4, 1>& xyzw,
    Eigen::Matrix<double, 2, 4>* optional_dxy_dcamparams = nullptr,
    Eigen::Matrix<double, 2, 4>* optional_dxy_dxyzw = nullptr);

Eigen::Matrix<double, 6, 6> SE3_left_jacobian_inv(const Eigen::Matrix<double, 6, 1>& se3);
Eigen::Matrix<double, 6, 6> SE3_left_jacobian(const Eigen::Matrix<double, 6, 1>& se3);

// derivative of exp([ẟ]) x wrt ẟ
Eigen::Matrix<double, 4, 6> dxyz_dse3(const Eigen::Matrix<double, 4, 1>& xyzw);

// Derivative wrt body:
// lim t→0 [ (M exp(tẟ) x - M x) / t ] =
// lim t→0 [ M (I + [ẟ]t + [ẟ]²t² + ...) x - M x / t ] =
// M [ẟ] x
//
// Derivate wrt world:
// lim t→0 [ (exp(tẟ) M x - M x) / t ] =
// lim t→0 [ (I + [ẟ]t + [ẟ]²t² + ...) M x - M x / t ] =
// [ẟ] M x
Eigen::Matrix<double, 4, 1> apply_transform(
    const Eigen::Matrix<double, 4, 4>& tx_world_object,
    const Eigen::Matrix<double, 4, 1>& body_xyzw,
    Eigen::Matrix<double, 4, 6>* optional_dxyz_dbody = nullptr,
    Eigen::Matrix<double, 4, 6>* optional_dxyz_dworld = nullptr);

Eigen::VectorD<2> camera_project(
    const Eigen::VectorD<4>& fxfycxcy,
    const Eigen::MatrixD<4>& tx_camera_world,
    const Eigen::VectorD<6>& se3_world_camera,
    const Eigen::MatrixD<4>& tx_world_object,    
    const Eigen::VectorD<6>& se3_world_object,
    const Eigen::VectorD<4>& object_point,
    Eigen::MatrixD<2, 4>* dxy_dcamparams_ptr = nullptr,
    Eigen::MatrixD<2, 6>* dxy_dcamera_ptr = nullptr,
    Eigen::MatrixD<2, 6>* dxy_dobject_ptr = nullptr);

double camera_project_factor(
    const Eigen::VectorD<16>& input,
    const size_t num_points,
    const Eigen::VectorD<4>* object_points,
    const Eigen::VectorD<2>* image_points,
    Eigen::MatrixD<16>* JtJ_ptr,
    Eigen::VectorD<16>* Jtr_ptr);

double camera_project_factor(
    const Eigen::VectorD<4>& fxfycxcy,
    const Eigen::VectorD<6>& se3_world_camera,
    const Eigen::VectorD<6>& se3_world_object,
    const size_t num_points,
    const Eigen::VectorD<4>* object_points,
    const Eigen::VectorD<2>* image_points,
    Eigen::MatrixD<16>* JtJ_ptr = nullptr,
    Eigen::VectorD<16>* Jtr_ptr = nullptr);

// shim for std::vector, std::array style containers
template <typename T, typename U>
inline double camera_project_factor(
    const Eigen::VectorD<4>& fxfycxcy,
    const Eigen::VectorD<6>& se3_world_camera,
    const Eigen::VectorD<6>& se3_world_object,
    const T& object_points,
    const U& image_points,
    Eigen::MatrixD<16>* JtJ_ptr = nullptr,
    Eigen::VectorD<16>* Jtr_ptr = nullptr) {

    assert(object_points.size() == image_points.size());
    assert(!object_points.empty());
    return camera_project_factor(fxfycxcy,
                                 se3_world_camera,
                                 se3_world_object,
                                 object_points.size(),
                                 object_points.data(),
                                 image_points.data(),
                                 JtJ_ptr, Jtr_ptr);
};

// shim for std::vector, std::array style containers
template <typename T, typename U>
inline double camera_project_factor(
    const Eigen::VectorD<16>& lin_point,
    const T& object_points,
    const U& image_points,
    Eigen::MatrixD<16>* JtJ_ptr = nullptr,
    Eigen::VectorD<16>* Jtr_ptr = nullptr) {

    assert(object_points.size() == image_points.size());
    assert(!object_points.empty());
    return camera_project_factor(lin_point,
                                 object_points.size(),
                                 object_points.data(),
                                 image_points.data(),
                                 JtJ_ptr, Jtr_ptr);
};

}  // carl
