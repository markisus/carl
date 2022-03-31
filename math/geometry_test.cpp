#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include "gtest/gtest.h"
#include "geometry.h"
#include "eigen_util.h"

using namespace carl;

#define lpInf(a, b) ((a) - (b)).lpNorm<Eigen::Infinity>()

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
    auto random_se3 = Eigen::random_vec<6>();
    auto random_SE3 = se3_exp(random_se3);
    auto random_xyzw = Eigen::random_vec<4>();
    random_xyzw(3) = 1;

    auto result = apply_transform(random_SE3, random_xyzw);
    EXPECT_LT(lpInf(result, random_SE3 * random_xyzw), 1e-12);
}

TEST(apply_transform, numerical_deriv_wrt_body) {
    auto random_se3 = Eigen::random_vec<6>();
    auto random_SE3 = se3_exp(random_se3);
    auto random_xyzw = Eigen::random_vec<4>();
    random_xyzw(3) = 1;

    Eigen::MatrixD<4, 6> dxyz_dbody;
    apply_transform(
        random_SE3,
        random_xyzw,
        &dxyz_dbody);

    Eigen::MatrixD<4, 6> dxyz_dbody_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 6; ++i) {
        dxyz_dbody_numeric.col(i) =
            (random_SE3 * se3_exp(Eigen::basis<6>(i)*eps) * random_xyzw -
             random_SE3 * random_xyzw)/eps;
    }
    
    EXPECT_LT(lpInf(dxyz_dbody_numeric, dxyz_dbody), 1e-6);
}

TEST(apply_transform, numerical_deriv_wrt_world) {
    auto random_se3 = Eigen::random_vec<6>();
    auto random_SE3 = se3_exp(random_se3);
    auto random_xyzw = Eigen::random_vec<4>();
    random_xyzw(3) = 1;

    Eigen::MatrixD<4, 6> dxyz_dworld;
    apply_transform(
        random_SE3,
        random_xyzw,
        nullptr,
        &dxyz_dworld);

    Eigen::MatrixD<4, 6> dxyz_dworld_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 6; ++i) {
        dxyz_dworld_numeric.col(i) =
            (se3_exp(Eigen::basis<6>(i)*eps)  * random_SE3 * random_xyzw -
             random_SE3 * random_xyzw)/eps;
    }
    
    EXPECT_LT(lpInf(dxyz_dworld_numeric, dxyz_dworld), 1e-6);
}


TEST(apply_camera_matrix, numerical_deriv_wrt_camparams) {
    auto camparams = Eigen::random_vec<4>();
    auto xyzw = Eigen::random_vec<4>();
    xyzw(3) = 1;

    Eigen::MatrixD<2, 4> dxy_dcamparams;
    auto result = apply_camera_matrix(camparams, xyzw,
                                 &dxy_dcamparams);

    Eigen::MatrixD<2, 4> dxy_dcamparams_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 4; ++i) {
        auto result_p = apply_camera_matrix((Eigen::basis<4>(i)*eps + camparams).eval(), xyzw);
        dxy_dcamparams_numeric.col(i) = (result_p - result)/eps;
    }

    EXPECT_LT(lpInf(dxy_dcamparams, dxy_dcamparams_numeric), 1e-3);
}

TEST(apply_camera_matrix, numerical_deriv_wrt_xyzw) {
    auto camparams = Eigen::random_vec<4>();
    auto xyzw = Eigen::random_vec<4>();
    xyzw(3) = 1;

    Eigen::MatrixD<2, 4> dxy_dxyzw;
    auto result = apply_camera_matrix(camparams, xyzw,
                                 nullptr, &dxy_dxyzw);

    Eigen::MatrixD<2, 4> dxy_dxyzw_numeric;
    const double eps = 1e-6;
    for (int i = 0; i < 4; ++i) {
        auto result_p = apply_camera_matrix(camparams, (xyzw + Eigen::basis<4>(i)*eps).eval());
        dxy_dxyzw_numeric.col(i) = (result_p - result)/eps;
    }

    EXPECT_LT(lpInf(dxy_dxyzw, dxy_dxyzw_numeric), 1e-3);
}

TEST(se3_exp, vs_generic_eigen) {
    auto input = Eigen::random_vec<6>().eval();
    auto eigen_result = se3_vec_to_mat(input).exp().eval();
    auto my_result = se3_exp(input);

    // std::cout << "eigen result\n " << eigen_result << "\n";
    // std::cout << "my result\n " << my_result << "\n";

    EXPECT_LT(lpInf(eigen_result, my_result), 1e-6);
}

TEST(SE3_log, vs_generic_eigen) {
    auto input = se3_exp(Eigen::random_vec<6>().eval());
    auto eigen_result = se3_mat_to_vec(input.log().eval());
    auto my_result = SE3_log(input);

    // std::cout << "eigen result\n " << eigen_result << "\n";
    // std::cout << "my result\n " << my_result << "\n";

    EXPECT_LT(lpInf(eigen_result, my_result), 1e-4);
}

TEST(se3_vec_to_mat, round_trip) {
    auto se3 = Eigen::random_vec<6>();
    auto se3_mat = se3_vec_to_mat(se3);
    auto se3_b = se3_mat_to_vec(se3_mat);
    EXPECT_LT((se3_b - se3).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(se3_mat_to_vec, round_trip) {
    auto se3 = se3_vec_to_mat(Eigen::random_vec<6>()); // get an arbitrary se3 mat
    auto se3_vec = se3_mat_to_vec(se3);
    auto se3_b = se3_vec_to_mat(se3_vec);
    EXPECT_LT(lpInf(se3_b, se3), 1e-6);
}

TEST(SO3_left_jacobian_inv, inverse_of_SO3_left_jacobian) {
    auto so3 = Eigen::random_vec<3>();
    auto so3_lj = SO3_left_jacobian(so3);
    auto so3_lji = SO3_left_jacobian_inv(so3);
    EXPECT_LT(lpInf((so3_lj*so3_lji), Eigen::id<3>()), 1e-9);
}

TEST(SE3_left_jacobian_inv, inverse_of_SE3_left_jacobian) {
    auto se3 = Eigen::random_vec<6>();
    auto se3_lj = SE3_left_jacobian(se3);
    auto se3_lji = SE3_left_jacobian_inv(se3);
    EXPECT_LT(lpInf((se3_lj*se3_lji), Eigen::id<6>()), 1e-9);
}

TEST(SE3_left_jacobian_inv, numerical_check) {
    const double eps = 1e-3;
    Eigen::VectorD<6> base = Eigen::random_vec<6>();
    Eigen::MatrixD<6> result;
    for (int i = 0; i < 6; ++i) {
        Eigen::VectorD<6> perturb = Eigen::basis<6>(i);

        // exp(t*d) = exp(b + jli(b) t*d) exp(-b)
        //     t*d  = log(exp(b + jli(b) t*d) exp(-b))
        //       d  = log(exp(b + jli(b) t*d) exp(-b)) / t

        const Eigen::MatrixD<6> jlinv = SE3_left_jacobian_inv(base);
        const Eigen::MatrixD<4> deriv = (se3_exp(base + jlinv * eps * perturb) * se3_exp(-base)).log()/eps;
        const Eigen::VectorD<6> deriv_vec = se3_mat_to_vec(deriv);
        result.col(i) = deriv_vec;
    }

    EXPECT_LT(lpInf(result, Eigen::id<6>()), 1e-3);
}

TEST(SE3_left_jacobian_inv, numerical_check_low_angle) {
    const double eps = 1e-2;
    auto base = Eigen::random_vec<6>();
    base.head<3>().normalize();
    base.head<3>() *= 1e-7;

    Eigen::MatrixD<6> result;
    for (int i = 0; i < 6; ++i) {
        Eigen::VectorD<6> perturb = Eigen::basis<6>(i);

        // exp(t*d) = exp(b + jli(b) t*d) exp(-b)
        //     t*d  = log(exp(b + jli(b) t*d) exp(-b))
        //       d  = log(exp(b + jli(b) t*d) exp(-b)) / t

        const Eigen::MatrixD<6> jlinv = SE3_left_jacobian_inv(base);
        const Eigen::MatrixD<4> deriv = (se3_exp(base + jlinv * eps * perturb) * se3_exp(-base)).log()/eps;
        const Eigen::VectorD<6> deriv_vec = se3_mat_to_vec(deriv);
        result.col(i) = deriv_vec;
    }

    EXPECT_LT(lpInf(result, Eigen::id<6>()), 1e-3);
}

TEST(camera_project, numerical_diff) {
    auto base = Eigen::random_vec<16>();
    auto perturb = Eigen::random_vec<16>();
    const double eps = 1e-4;

    const auto body_point = Eigen::random_homog();

    Eigen::VectorD<16> perturbed = base + perturb*eps;

    auto base_camparams = base.head<4>();
    auto base_se3_world_camera = base.segment<6>(4);
    auto base_se3_world_body = base.tail<6>();
    
    auto perturbed_camparams = perturbed.head<4>();
    auto perturbed_se3_world_camera = perturbed.segment<6>(4);
    auto perturbed_se3_world_body = perturbed.tail<6>();

    Eigen::MatrixD<2, 4> dxy_dcamparams;
    Eigen::MatrixD<2, 6> dxy_dcamera;
    Eigen::MatrixD<2, 6> dxy_dbody;
    
    auto fbase = camera_project(
        base_camparams,
        se3_exp(base_se3_world_camera).inverse(),
        base_se3_world_camera,
        se3_exp(base_se3_world_body),
        base_se3_world_body,
        body_point,
        &dxy_dcamparams,
        &dxy_dcamera,
        &dxy_dbody);

    // std::cout << "fbase " << fbase.transpose() << "\n";
    // std::cout << "dxy_dcamparams\n " << dxy_dcamparams << "\n";
    // std::cout << "dxy_dcamera\n " << dxy_dcamera << "\n";
    // std::cout << "dxy_dbody\n " << dxy_dbody << "\n";

    auto fperturbed = camera_project(
        perturbed_camparams,
        se3_exp(perturbed_se3_world_camera).inverse(),
        perturbed_se3_world_camera,
        se3_exp(perturbed_se3_world_body),
        perturbed_se3_world_body,
        body_point);

    auto delta = (fperturbed - fbase).eval();
    auto deriv_numerical = delta/eps;

    auto deriv_computed = 
        (dxy_dcamparams * perturb.head<4>() +
        dxy_dcamera * perturb.segment<6>(4) +
         dxy_dbody * perturb.tail<6>()).eval();

    EXPECT_LT(lpInf(deriv_computed, deriv_numerical), 1e-3);
}

TEST(camera_project_factor, check_JtJ_rtJ) {
    auto base = Eigen::random_vec<16>();

    const auto body_point = Eigen::random_homog();
    const auto image_point = Eigen::random_vec<2>();

    std::vector<Eigen::VectorD<4>> body_points;
    body_points.push_back(body_point);

    std::vector<Eigen::VectorD<2>> image_points;
    image_points.push_back(image_point);

    auto camparams = base.head<4>();
    auto se3_world_camera = base.segment<6>(4);
    auto se3_world_body = base.tail<6>();

    // leverage camera_project to construct a linearized
    // estimate of the residual 
    Eigen::MatrixD<2, 4> dxy_dcamparams;
    Eigen::MatrixD<2, 6> dxy_dcamera;
    Eigen::MatrixD<2, 6> dxy_dbody;
    auto projected_image_point = camera_project(
        camparams,
        se3_exp(se3_world_camera).inverse(),
        se3_world_camera,
        se3_exp(se3_world_body),
        se3_world_body,
        body_point,
        &dxy_dcamparams,
        &dxy_dcamera,
        &dxy_dbody);
    auto perturb = Eigen::random_vec<16>();
    auto projected_image_point_perturbed =
        projected_image_point +
        dxy_dcamparams * perturb.head<4>() +
        dxy_dcamera * perturb.segment<6>(4) +
        dxy_dbody * perturb.tail<6>();
    const double perturbed_error2 = (projected_image_point_perturbed - image_point).squaredNorm();

    // now compare this estimate to the one given by the factor
    Eigen::MatrixD<16> JtJ;
    Eigen::VectorD<16> rtJ;
    double error2 = camera_project_factor(
        camparams,
        se3_world_camera,
        se3_world_body,
        body_points,
        image_points,
        &JtJ, &rtJ);
    Eigen::MatrixD<1> temp = (perturb.transpose()*JtJ*perturb - 2*rtJ.transpose()*perturb);
    const double perturbed_error2_from_factor = temp(0,0) + error2;

    EXPECT_LT(std::abs(perturbed_error2_from_factor - perturbed_error2), 1e-6);
}

TEST(camera_project_factor, check_JtJ_rtJ_sum) {
    auto base = Eigen::random_vec<16>();
    auto camparams = base.head<4>();
    auto se3_world_camera = base.segment<6>(4);
    auto se3_world_body = base.tail<6>();

    std::array<Eigen::VectorD<4>, 2> body_points_arr { Eigen::random_homog(), Eigen::random_homog() }; 
    std::array<Eigen::VectorD<2>, 2> image_points_arr { Eigen::random_vec<2>(), Eigen::random_vec<2>() }; 
    Eigen::MatrixD<16> sum_JtJs = Eigen::zero_mat<16>();
    Eigen::VectorD<16> sum_rtJs = Eigen::zero_vec<16>();
    double sum_error2s = 0;
    for (int i = 0; i < 2; ++i) {
        Eigen::MatrixD<16> JtJ;
        Eigen::VectorD<16> rtJ;
        std::vector<Eigen::VectorD<4>> body_points;
        body_points.push_back(body_points_arr[i]);
        std::vector<Eigen::VectorD<2>> image_points;
        image_points.push_back(image_points_arr[i]);
        double error2 = camera_project_factor(
            camparams,
            se3_world_camera,
            se3_world_body,
            body_points,
            image_points,
            &JtJ, &rtJ);
        sum_JtJs += JtJ;
        sum_rtJs += rtJ;
        sum_error2s += error2;
    }

    Eigen::MatrixD<16> sum_JtJs_alt = Eigen::zero_mat<16>();
    Eigen::VectorD<16> sum_rtJs_alt = Eigen::zero_vec<16>();
    std::vector<Eigen::VectorD<4>> body_points {body_points_arr[0], body_points_arr[1]};
    std::vector<Eigen::VectorD<2>> image_points {image_points_arr[0], image_points_arr[1]};
    double error2 = camera_project_factor(
        camparams,
        se3_world_camera,
        se3_world_body,
        body_points,
        image_points,
        &sum_JtJs_alt, &sum_rtJs_alt);
    EXPECT_LT(lpInf(sum_JtJs, sum_JtJs_alt), 1e-6);
    EXPECT_LT(lpInf(sum_rtJs, sum_rtJs_alt), 1e-6);
    EXPECT_LT(std::abs(error2 - sum_error2s), 1e-6);
}
