#pragma once

#include "Eigen/Dense"

namespace carl {

// some utility functions to cut down on verbosity
template <int dim>
auto id() {
    return Eigen::Matrix<double, dim, dim>::Identity();
}

template <int dim>
Eigen::Matrix<double, dim, dim> zero_mat() {
    return Eigen::Matrix<double, dim, dim>::Zero();
}

template <int dim>
Eigen::Matrix<double, dim, 1> zero_vec() {
    return Eigen::Matrix<double, dim, 1>::Zero();
}

template <int dim>
Eigen::Matrix<double, dim, 1> basis(int i) {
    auto result = zero_vec<dim>();
    result(i) = 1;
    return result;
}

template <int dim>
Eigen::Matrix<double, dim, 1> random_vec() {
    return Eigen::Matrix<double, dim, 1>::Random();
}

Eigen::Matrix<double, 3, 3> so3_vec_to_mat(const Eigen::Matrix<double, 3, 1>& vec);

Eigen::Matrix<double, 3, 3> SO3_left_jacobian_inv(const Eigen::Matrix<double, 3, 1>& so3);
Eigen::Matrix<double, 3, 3> SO3_left_jacobian(const Eigen::Matrix<double, 3, 1>& so3);

Eigen::Matrix<double, 4, 4> se3_exp(const Eigen::Matrix<double, 6, 1>& se3);
Eigen::Matrix<double, 4, 4> se3_vec_to_mat(const Eigen::Matrix<double, 6, 1>& se3);
Eigen::Matrix<double, 6, 1> se3_mat_to_vec(const Eigen::Matrix<double, 4, 4>& se3);

Eigen::Matrix<double, 2, 1> camera_project(
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

}  // carl
