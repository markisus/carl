#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include "gtest/gtest.h"
#include "geometry.h"

using namespace carl;

// TEST(se3_vec_to_mat, test1) {
//     std::cout << "se3_0: \n" << se3_vec_to_mat(basis<6>(0)) << "\n";
//     std::cout << "se3_1: \n" << se3_vec_to_mat(basis<6>(1)) << "\n";
//     std::cout << "se3_2: \n" << se3_vec_to_mat(basis<6>(2)) << "\n";
//     std::cout << "se3_3: \n" << se3_vec_to_mat(basis<6>(3)) << "\n";
//     std::cout << "se3_4: \n" << se3_vec_to_mat(basis<6>(4)) << "\n";
//     std::cout << "se3_5: \n" << se3_vec_to_mat(basis<6>(5)) << "\n";
// }

TEST(apply_transform, implements_matmul) {
    srand(0);
    auto random_se3 = random_vec<6>();
    auto random_SE3 = se3_exp(random_se3);
    auto random_xyzw = random_vec<4>();
    random_xyzw(3) = 1;

    auto result = apply_transform(random_SE3, random_xyzw);
    EXPECT_LT( (result - random_SE3 * random_xyzw).lpNorm<Eigen::Infinity>(), 1e-12);
}

TEST(apply_transform, numerical_deriv_wrt_body) {
    auto random_se3 = random_vec<6>();
    auto random_SE3 = se3_exp(random_se3);
    auto random_xyzw = random_vec<4>();
    random_xyzw(3) = 1;

    Eigen::Matrix<double, 4, 6> dxyz_dbody;
    apply_transform(
        random_SE3,
        random_xyzw,
        &dxyz_dbody);

    Eigen::Matrix<double, 4, 6> dxyz_dbody_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 6; ++i) {
        dxyz_dbody_numeric.col(i) =
            (random_SE3 * se3_exp(basis<6>(i)*eps) * random_xyzw -
             random_SE3 * random_xyzw)/eps;
    }
    
    EXPECT_LT( (dxyz_dbody_numeric - dxyz_dbody).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(apply_transform, numerical_deriv_wrt_world) {
    auto random_se3 = random_vec<6>();
    auto random_SE3 = se3_exp(random_se3);
    auto random_xyzw = random_vec<4>();
    random_xyzw(3) = 1;

    Eigen::Matrix<double, 4, 6> dxyz_dworld;
    apply_transform(
        random_SE3,
        random_xyzw,
        nullptr,
        &dxyz_dworld);

    Eigen::Matrix<double, 4, 6> dxyz_dworld_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 6; ++i) {
        dxyz_dworld_numeric.col(i) =
            (se3_exp(basis<6>(i)*eps)  * random_SE3 * random_xyzw -
             random_SE3 * random_xyzw)/eps;
    }
    
    EXPECT_LT( (dxyz_dworld_numeric - dxyz_dworld).lpNorm<Eigen::Infinity>(), 1e-6);
}


TEST(camera_project, numerical_deriv_wrt_camparams) {
    auto camparams = random_vec<4>();
    auto xyzw = random_vec<4>();
    xyzw(3) = 1;

    Eigen::Matrix<double, 2, 4> dxy_dcamparams;
    auto result = camera_project(camparams, xyzw,
                                 &dxy_dcamparams);

    Eigen::Matrix<double, 2, 4> dxy_dcamparams_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 4; ++i) {
        auto result_p = camera_project((basis<4>(i)*eps + camparams).eval(), xyzw);
        dxy_dcamparams_numeric.col(i) = (result_p - result)/eps;
    }

    EXPECT_LT((dxy_dcamparams - dxy_dcamparams_numeric).lpNorm<Eigen::Infinity>(), 1e-3);
}

TEST(camera_project, numerical_deriv_wrt_xyzw) {
    auto camparams = random_vec<4>();
    auto xyzw = random_vec<4>();
    xyzw(3) = 1;

    Eigen::Matrix<double, 2, 4> dxy_dxyzw;
    auto result = camera_project(camparams, xyzw,
                                 nullptr, &dxy_dxyzw);

    Eigen::Matrix<double, 2, 4> dxy_dxyzw_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 4; ++i) {
        auto result_p = camera_project(camparams, (xyzw + basis<4>(i)*eps).eval());
        dxy_dxyzw_numeric.col(i) = (result_p - result)/eps;
    }

    EXPECT_LT((dxy_dxyzw - dxy_dxyzw_numeric).lpNorm<Eigen::Infinity>(), 1e-3);
}

TEST(se3_exp, vs_generic_eigen) {
    auto input = random_vec<6>().eval();
    auto eigen_result = se3_vec_to_mat(input).exp().eval();
    auto my_result = se3_exp(input);

    // std::cout << "eigen result\n " << eigen_result << "\n";
    // std::cout << "my result\n " << my_result << "\n";

    EXPECT_LT((eigen_result - my_result).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(se3_vec_to_mat, round_trip) {
    auto se3 = random_vec<6>();
    auto se3_mat = se3_vec_to_mat(se3);
    auto se3_b = se3_mat_to_vec(se3_mat);
    EXPECT_LT((se3_b - se3).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(se3_mat_to_vec, round_trip) {
    auto se3 = se3_vec_to_mat(random_vec<6>()); // get an arbitrary se3 mat
    auto se3_vec = se3_mat_to_vec(se3);
    auto se3_b = se3_vec_to_mat(se3_vec);
    EXPECT_LT((se3_b - se3).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(SO3_left_jacobian_inv, inverse_of_SO3_left_jacobian) {
    auto so3 = random_vec<3>();
    auto so3_lj = SO3_left_jacobian(so3);
    auto so3_lji = SO3_left_jacobian_inv(so3);
    EXPECT_LT(((so3_lj*so3_lji) - id<3>()).lpNorm<Eigen::Infinity>(), 1e-9);
}

TEST(SE3_left_jacobian_inv, inverse_of_SE3_left_jacobian) {
    auto se3 = random_vec<6>();
    auto se3_lj = SE3_left_jacobian(se3);
    auto se3_lji = SE3_left_jacobian_inv(se3);
    EXPECT_LT(((se3_lj*se3_lji) - id<6>()).lpNorm<Eigen::Infinity>(), 1e-9);
}

TEST(SE3_left_jacobian_inv, numerical_check) {
    const double eps = 1e-3;
    Eigen::Matrix<double, 6, 1> base = random_vec<6>();
    Eigen::Matrix<double, 6, 6> result;
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> perturb = basis<6>(i);

        // exp(t*d) = exp(b + jli(b) t*d) exp(-b)
        //     t*d  = log(exp(b + jli(b) t*d) exp(-b))
        //       d  = log(exp(b + jli(b) t*d) exp(-b)) / t

        const Eigen::Matrix<double, 6, 6> jlinv = SE3_left_jacobian_inv(base);
        const Eigen::Matrix<double, 4, 4> deriv = (se3_exp(base + jlinv * eps * perturb) * se3_exp(-base)).log()/eps;
        const Eigen::Matrix<double, 6, 1> deriv_vec = se3_mat_to_vec(deriv);
        result.col(i) = deriv_vec;
    }

    EXPECT_LT((result - id<6>()).lpNorm<Eigen::Infinity>(), 1e-3);
}

TEST(SE3_left_jacobian_inv, numerical_check_low_angle) {
    const double eps = 1e-2;
    Eigen::Matrix<double, 6, 1> base = random_vec<6>();
    base.head<3>().normalize();
    base.head<3>() *= 1e-7;

    Eigen::Matrix<double, 6, 6> result;
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> perturb = basis<6>(i);

        // exp(t*d) = exp(b + jli(b) t*d) exp(-b)
        //     t*d  = log(exp(b + jli(b) t*d) exp(-b))
        //       d  = log(exp(b + jli(b) t*d) exp(-b)) / t

        const Eigen::Matrix<double, 6, 6> jlinv = SE3_left_jacobian_inv(base);
        const Eigen::Matrix<double, 4, 4> deriv = (se3_exp(base + jlinv * eps * perturb) * se3_exp(-base)).log()/eps;
        const Eigen::Matrix<double, 6, 1> deriv_vec = se3_mat_to_vec(deriv);
        result.col(i) = deriv_vec;
    }

    EXPECT_LT((result - id<6>()).lpNorm<Eigen::Infinity>(), 1e-3);
}
