#include "geometry.h"
#include <iostream>

namespace carl {

constexpr double kTol = 1e-9;  // tolerance for almost equal

bool almostEquals(const double a, const double b) {
    return std::abs(a - b) < kTol;
}

struct ScrewData {
    Eigen::Matrix3d omega;  // a unit vector, encoding axis of rotation
    Eigen::Matrix3d omega_squared;
    Eigen::Vector3d v;  // linear velocity of the screw
    Eigen::Vector3d omega_v;
    Eigen::Vector3d omega_squared_v;
};

Eigen::Matrix<double, 3, 3> so3_vec_to_mat(
    const Eigen::Matrix<double, 3, 1>& vec) {

    const double w1 = vec(0);
    const double w2 = vec(1);
    const double w3 = vec(2);

    Eigen::Matrix<double, 3, 3> m;
    // clang-format off
    m <<
    0,  -w3, w2, 
    w3,  0,  -w1, 
    -w2, w1, 0;
    // clang-format on

    return m;
}

Eigen::Matrix<double, 3, 1> so3_mat_to_vec(
    const Eigen::Matrix<double, 3, 3>& mat) {

    Eigen::Matrix<double, 3, 1> result;
    result << mat(2,1), mat(0,2), mat(1,0);

    return result;
}


Eigen::Matrix<double, 4, 4> se3_vec_to_mat(
    const Eigen::Matrix<double, 6, 1>& vec) {

    Eigen::Matrix<double, 4, 4> result;
    result.block<3,3>(0,0) = so3_vec_to_mat(vec.head<3>());
    result.block<3,1>(0,3) = vec.tail<3>();
    result.row(3).setZero();
    return result;
}

Eigen::Matrix<double, 6, 1> se3_mat_to_vec(
    const Eigen::Matrix<double, 4, 4>& mat) {

    Eigen::Matrix<double, 6, 1> result;
    result.tail<3>() = mat.block<3,1>(0,3);
    result.head<3>() = so3_mat_to_vec(mat.block<3,3>(0,0).eval());
    return result;
}

Eigen::Matrix<double, 3, 3> SO3_left_jacobian_inv(const Eigen::Matrix<double, 3, 1>& so3) {
    const double theta = so3.norm();
    const double theta2 = theta*theta;
    const double theta4 = theta2*theta2;

    const auto theta_mat = so3_vec_to_mat(so3);    
    double coeff1;
    if (theta < 1e-6) {
        coeff1 = 1.0/12 + theta2/720 + theta4/30240;
    } else {
        const double cs = std::cos(theta);
        const double sn = std::sin(theta);
        coeff1 = (1.0/theta2) - (1 + cs)/(2*theta*sn);
    }
    Eigen::Matrix<double, 3, 3> result = id<3>() - theta_mat*0.5 + coeff1*theta_mat*theta_mat;
    return result;
}

Eigen::Matrix<double, 3, 3> SO3_left_jacobian(const Eigen::Matrix<double, 3, 1>& so3) {
    const double theta = so3.norm();
    const double theta2 = theta*theta;
    const double theta3 = theta2*theta;
    const double theta4 = theta2*theta2;

    const auto theta_mat = so3_vec_to_mat(so3);    
    double coeff1;
    double coeff2;
    if (theta < 1e-6) {
        coeff1 = 0.5 - theta2/24 + theta4/720;
        coeff2 = 1.0/6 - theta2/120 + theta4/5040;
    } else {
        const double cs = std::cos(theta);
        const double sn = std::sin(theta);
        coeff1 = (1.0 - cs)/theta2;
        coeff2 = (theta - sn)/theta3;
    }
    Eigen::Matrix<double, 3, 3> result = id<3>() + coeff1*theta_mat + coeff2*theta_mat*theta_mat;
    return result;
}

Eigen::Matrix<double, 3, 3> barfoot_Q(const Eigen::Matrix<double, 6, 1>& se3) {
    const auto rho = so3_vec_to_mat(se3.tail<3>());
    const auto theta = so3_vec_to_mat(se3.head<3>());
    const auto rho_theta = (rho*theta).eval();
    const auto rho_theta2 = (rho_theta*theta).eval();
    const auto theta_rho_theta = (theta*rho_theta).eval();
    const auto theta_rho = (theta*rho).eval();
    const auto theta2_rho = (theta*theta_rho).eval();

    double c1;
    double c2;
    double c3;
    const double angle = se3.head<3>().norm();
    if (angle < 1e-6) {
        const double angle2 = angle*angle;
        const double angle4 = angle2*angle2;
        c1 = 1.0/6 - angle2/120 + angle4/5040;
        c2 = 1.0/24 - angle2/720 + angle4/40320;
        c3 = 1.0/120 - angle2/2520 + angle4/120960;
    } else {
        const double angle2 = angle*angle;
        const double angle3 = angle2*angle;
        const double angle4 = angle2*angle2;
        const double angle5 = angle3*angle2;
        const double sn = std::sin(angle);
        const double cs = std::cos(angle);
        c1 = (angle - sn)/angle3;
        c2 = -(1.0 - angle2/2 - cs)/angle4;
        c3 = -0.5*(
            (1.0 - angle2/2 - cs)/angle4 -
            3.0*(angle - sn - angle3/6)/angle5);
    }

    const Eigen::Matrix<double, 3, 3> line1 = 0.5*rho + c1*(theta_rho + rho_theta + theta_rho_theta);
    const Eigen::Matrix<double, 3, 3> line2 = c2*(theta2_rho + rho_theta2 - 3.0*theta_rho_theta);
    const Eigen::Matrix<double, 3, 3> line3 = c3*(theta_rho_theta*theta + theta*theta_rho_theta);
    return line1 + line2 + line3;
}

Eigen::Matrix<double, 6, 6> SE3_left_jacobian_inv(const Eigen::Matrix<double, 6, 1>& se3) {
    const Eigen::Matrix<double, 3, 3> SO3_lji = SO3_left_jacobian_inv(se3.head<3>().eval());
    const Eigen::Matrix<double, 3, 3> Q = barfoot_Q(se3);
    Eigen::Matrix<double, 6, 6> result;
    result.block<3,3>(0,0) = SO3_lji;
    result.block<3,3>(3,3) = SO3_lji;
    result.block<3,3>(0,3).setZero();
    result.block<3,3>(3,0) = -SO3_lji * Q * SO3_lji;
    return result;
}

Eigen::Matrix<double, 6, 6> SE3_left_jacobian(const Eigen::Matrix<double, 6, 1>& se3) {
    const Eigen::Matrix<double, 3, 3> SO3_lj = SO3_left_jacobian(se3.head<3>().eval());
    const Eigen::Matrix<double, 3, 3> Q = barfoot_Q(se3);
    Eigen::Matrix<double, 6, 6> result;
    result.block<3,3>(0,0) = SO3_lj;
    result.block<3,3>(3,3) = SO3_lj;
    result.block<3,3>(0,3).setZero();
    result.block<3,3>(3,0) = Q;
    return result;
}

ScrewData make_screw_data_from_twist(const Eigen::Matrix<double, 6, 1>& twist) {
    ScrewData screw_data;
    screw_data.omega = so3_vec_to_mat(twist.head<3>());
    screw_data.omega_squared = screw_data.omega * screw_data.omega;
    screw_data.v = twist.tail<3>();
    screw_data.omega_v = screw_data.omega * screw_data.v;
    screw_data.omega_squared_v = screw_data.omega * screw_data.omega_v;
    return screw_data;
}

Eigen::Matrix<double, 4, 4> exp_screw(const ScrewData& screw_data,
                                      const double distance) {
    const float s = std::sin(distance);
    const float c = std::cos(distance);

    Eigen::Matrix<double, 4, 4> result;
    result.row(3) << 0, 0, 0, 1;
    result.block<3, 3>(0, 0) =
        id<3>() +
        s * screw_data.omega +
        (1 - c) * screw_data.omega_squared;
    result.block<3, 1>(0, 3) =
        screw_data.v * distance +
        (1 - c) * screw_data.omega_v +
        (distance - s) * screw_data.omega_squared_v;

    return result;
}

Eigen::Matrix<double, 4, 4> se3_exp(const Eigen::Matrix<double, 6, 1>& se3) {
    const double rotation_norm = se3.head<3>().norm();
    if (almostEquals(rotation_norm, 0)) {
        // Easy case: pure translation
        Eigen::Matrix<double, 4, 4> result =
            Eigen::Matrix<double, 4, 4>::Identity();
        result.block<3, 1>(0, 3) = se3.tail<3>();
        return result;
    }
    const Eigen::Matrix<double, 6, 1> screw = se3 / rotation_norm;
    const ScrewData screw_data = make_screw_data_from_twist(screw);
    return exp_screw(screw_data, rotation_norm);
}


// derivative of exp([ẟ]) x wrt ẟ
Eigen::Matrix<double, 4, 6> dxyz_dse3(const Eigen::Matrix<double, 4, 1>& xyzw) {
    const double x = xyzw(0);
    const double y = xyzw(1);
    const double z = xyzw(2);
    const double w = xyzw(3);

    Eigen::Matrix<double, 4, 6> result;
    result.col(0) << 0, -z, y, 0;
    result.col(1) << z, 0, -x, 0;
    result.col(2) << -y, x, 0, 0;
    result.col(3) << w, 0, 0, 0;
    result.col(4) << 0, w, 0, 0;
    result.col(5) << 0, 0, w, 0;    
    return result;
}


Eigen::Matrix<double, 4, 1> apply_transform(
    const Eigen::Matrix<double, 4, 4>& tx_world_body,
    const Eigen::Matrix<double, 4, 1>& body_xyzw,
    Eigen::Matrix<double, 4, 6>* optional_dxyz_dbody,
    Eigen::Matrix<double, 4, 6>* optional_dxyz_dworld) {
    Eigen::Matrix<double, 4, 1> world_xyzw = tx_world_body * body_xyzw;
    if (optional_dxyz_dworld) {
        *optional_dxyz_dworld = dxyz_dse3(world_xyzw);
    }
    if (optional_dxyz_dbody) {
        *optional_dxyz_dbody = tx_world_body * dxyz_dse3(body_xyzw);
    }
    return world_xyzw;
}

Eigen::Matrix<double, 2, 1> camera_project(
    const Eigen::Matrix<double, 4, 1>& fxfycxcy,
    const Eigen::Matrix<double, 4, 1>& xyzw,
    Eigen::Matrix<double, 2, 4>* optional_dxy_dcamparams,
    Eigen::Matrix<double, 2, 4>* optional_dxy_dxyzw) {

    const double x = xyzw(0);
    const double y = xyzw(1);
    const double z = xyzw(2);

    const double fx = fxfycxcy(0);
    const double fy = fxfycxcy(1);
    const double cx = fxfycxcy(2);
    const double cy = fxfycxcy(3);

    double cam_x = fx * x/z + cx;
    double cam_y = fy * y/z + cy;

    Eigen::Matrix<double, 2, 1> result;
    result << cam_x, cam_y;

    if (optional_dxy_dcamparams) {
        auto& dxy_dcamparams = *optional_dxy_dcamparams;
        dxy_dcamparams <<
            x/z,   0, 1, 0,
              0, y/z, 0, 1;
    }

    if (optional_dxy_dxyzw) {
        auto& dxy_dxyzw = *optional_dxy_dxyzw;
        dxy_dxyzw <<
            fx/z,    0, -fx * x/(z*z), 0,
               0, fy/z, -fy * y/(z*z), 0;
    }

    return result;
}

void camera_object_factor(
    const Eigen::Matrix<double, 4, 1>& fxfycxcy,
    const Eigen::Matrix<double, 6, 1>& se3_world_camera,
    const Eigen::Matrix<double, 6, 1>& se3_world_object) {

    auto se3_camera_world = (-se3_world_camera).eval();

    auto tx_camera_world = se3_exp(se3_camera_world);
    auto tx_world_object = se3_exp(se3_world_object);

    auto tx_camera_object = (tx_camera_world * tx_world_object).eval();

    for (int i = 0; i < 8; ++i) {
        // Consider this perturbation.
        // ẟ → tx_camera_world * exp(ẟ) * tx_world_object * model
        // 
        // We can view it as acting on the tx_world_object. Alternatively,
        // we can view it as the negative of a perturbation acting on the
        // tx_world_camera.
        // 
        // ẟ' → (exp(ẟ') * tx_world_camera)⁻¹ * tx_world_object * model
        //    = tx_camera_world * exp(-ẟ') * tx_world_object * model
        //
        // In hindsight it's obvious that applying a perturbation on
        // the body (e.g. moving it forward in the world by 1 mm), will have
        // the exact opposite effect as applying the same perturbation on the
        // camera when viewing the camera frame coordinates of the object.

        Eigen::Matrix<double, 4, 6> dcampoint_dobject;
        Eigen::Matrix<double, 4, 1> world_point = tx_world_object * model.point(i);
        Eigen::Matrix<double, 4, 1> camera_point = apply_transform(tx_camera_world, world_point, &dcampoint_dobject);

        Eigen::Matrix<double, 2, 4> dxy_dcamparams;
        Eigen::Matrix<double, 2, 4> dxy_dcampoint;
        camera_project(fxfycxcy, camera_point, &dxy_dcamparams, &dxy_dcampoint);

        Eigen::Matrix<double, 2, 6> dxy_dobject = dxy_dcampoint * dcampoint_dobject;
        Eigen::Matrix<double, 2, 6> dxy_dcamera = -dxy_dobject;

        // exp(Jl_logM [Δ]) * M ~= exp(logM + [Δ])
        // need to premultiply dxy_dobject by Jl_logdobject_dworld
        // need to premultiply dxy_camera by Jl_logcamera_dworld
        
        Eigen::Matrix<double, 2, 6> dxy_dobject_global = dxy_dobject * SE3_left_jacobian(se3_world_object);
        Eigen::Matrix<double, 2, 6> dxy_dcamera_global = dxy_dobject * SE3_left_jacobian(se3_world_camera);
        
    }
}

}  // carl
