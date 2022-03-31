#include "geometry.h"
#include "util/eigen_util.h"
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

// x * cotan(x) for x
// accurate only in region (-pi/2, pi/2) via taylor series
// https://www.wolframalpha.com/input/?i=taylor+expand+x+cotan%28x%29+around+0
double x_cotx(double x) {
    constexpr double c2 = -1.0 / 3;
    constexpr double c4 = -1.0 / 45;
    constexpr double c6 = -2.0 / 945;
    constexpr double c8 = -1.0 / 4725;
    constexpr double c10 = -2.0 / 93555;
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double x6 = x4 * x2;
    const double x8 = x4 * x4;
    const double x10 = x8 * x2;
    return 1.0 + c2 * x2 + c4 * x4 + c6 * x6 + c8 * x8 + c10 * x10;
}

void SO3_log(const Eigen::Matrix3d& rotation_matrix,
             double* theta,
             Eigen::Matrix3d* omega_hat) {
    // Check for edge case: identity
    // A rotation matrix is identity iff all diagonal entries are 1.0
    bool is_identity = true;
    for (int i = 0; i < 3; ++i) {
        if (!almostEquals(1.0f, rotation_matrix(i, i))) {
            is_identity = false;
            break;
        }
    }
    if (is_identity) {
        *theta = 0;
        Eigen::Vector3d omega_vector;
        omega_vector << 1, 0, 0;
        (*omega_hat) = so3_vec_to_mat(omega_vector);
        return;
    }

    // Check for edge case: rotation of k*PI
    const double trace = rotation_matrix.trace();
    if (almostEquals(-1.0f, trace)) {
        *theta = M_PI;
        Eigen::Vector3d omega_vector;
        const double r33 = rotation_matrix(2, 2);
        const double r22 = rotation_matrix(1, 1);
        const double r11 = rotation_matrix(0, 0);
        if (!almostEquals(1.0f + r33, 0.0f)) {
            (omega_vector)(0) = rotation_matrix(0, 2);
            (omega_vector)(1) = rotation_matrix(1, 2);
            (omega_vector)(2) = 1 + rotation_matrix(2, 2);
            (omega_vector) /= std::sqrt(2 * (1 + r33));
        } else if (!almostEquals(1.0f + r22, 0.0f)) {
            (omega_vector)(0) = rotation_matrix(0, 1);
            (omega_vector)(1) = 1 + rotation_matrix(1, 1);
            (omega_vector)(2) = rotation_matrix(2, 1);
            (omega_vector) /= std::sqrt(2 * (1 + r22));
        } else {
            // 1 + r11 != 0
            (omega_vector)(0) = 1 + rotation_matrix(0, 0);
            (omega_vector)(1) = rotation_matrix(1, 0);
            (omega_vector)(2) = rotation_matrix(2, 0);
            (omega_vector) /= std::sqrt(2 * (1 + r11));
        }
        (*omega_hat) = so3_vec_to_mat(omega_vector);
        return;
    }

    // Normal case

    // htmo means Half of Trace Minus One
    const double htmo = 0.5f * (trace - 1);
    *theta = std::acos(htmo);  // todo: investigate faster approx to acos

    const double sin_acos_htmo = std::sqrt(1.0 - htmo * htmo);
    *omega_hat =
        0.5f / sin_acos_htmo * (rotation_matrix - rotation_matrix.transpose());
}

// Returns a twist
Eigen::VectorD<6> SE3_log(const Eigen::Matrix4d& SE3_element) {
    // exp( (omega theta, v theta) ) = SE3_element
    // omega theta = log(R)
    // v theta = theta Ginv(theta) * p

    double theta;
    Eigen::Matrix3d omega;
    SO3_log(SE3_element.block<3, 3>(0, 0), &theta, &omega);

    // conversion from omega matrix to omega vector, then multiplication by
    // theta
    Eigen::Vector3d omega_theta;
    omega_theta << theta * omega(2, 1), -theta * omega(2, 0),
        theta * omega(1, 0);

    const Eigen::Vector3d p = SE3_element.block<3, 1>(0, 3);

    const Eigen::Vector3d omega_p = omega * p;
    const Eigen::Vector3d v_theta =
        p - 0.5f * theta * omega_p +
        (1.0f - x_cotx(theta / 2)) * omega * omega_p;

    Eigen::VectorD<6> result;
    result.head<3>() = omega_theta;
    result.tail<3>() = v_theta;
    return result;
}

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
    Eigen::Matrix<double, 3, 3> result = Eigen::id<3>() - theta_mat*0.5 + coeff1*theta_mat*theta_mat;
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
    Eigen::Matrix<double, 3, 3> result = Eigen::id<3>() + coeff1*theta_mat + coeff2*theta_mat*theta_mat;
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
    const double s = std::sin(distance);
    const double c = std::cos(distance);

    Eigen::Matrix<double, 4, 4> result;
    result.row(3) << 0, 0, 0, 1;
    result.block<3, 3>(0, 0) =
        Eigen::id<3>() +
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

Eigen::Matrix<double, 2, 1> apply_camera_matrix(
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

Eigen::VectorD<2> camera_project(
    const Eigen::VectorD<4>& fxfycxcy,
    const Eigen::MatrixD<4>& tx_camera_world,
    const Eigen::VectorD<6>& se3_world_camera,
    const Eigen::MatrixD<4>& tx_world_object,
    const Eigen::VectorD<6>& se3_world_object,
    const Eigen::VectorD<4>& object_point,
    Eigen::MatrixD<2, 4>* dxy_dcamparams_ptr,
    Eigen::MatrixD<2, 6>* dxy_dcamera_ptr,
    Eigen::MatrixD<2, 6>* dxy_dobject_ptr) {
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

    Eigen::MatrixD<4, 6> dcampoint_dobject;
    Eigen::MatrixD<4, 1> world_point = tx_world_object * object_point;
    Eigen::MatrixD<4, 1> camera_point = apply_transform(tx_camera_world, world_point, &dcampoint_dobject);

    Eigen::MatrixD<2, 4> dxy_dcampoint;
    Eigen::VectorD<2> projected_image_point = apply_camera_matrix(fxfycxcy, camera_point, dxy_dcamparams_ptr, &dxy_dcampoint);

    Eigen::MatrixD<2, 6> dxy_dobject = dxy_dcampoint * dcampoint_dobject;
    Eigen::MatrixD<2, 6> dxy_dcamera = -dxy_dobject;

    // exp(Jl_logM [Δ]) * M ~= exp(logM + [Δ])
    // need to premultiply dxy_dobject by Jl_logdobject_dworld
    // need to premultiply dxy_camera by Jl_logcamera_dworld
        
    Eigen::MatrixD<2, 6> dxy_dobject_global = dxy_dobject * SE3_left_jacobian(se3_world_object);
    Eigen::MatrixD<2, 6> dxy_dcamera_global = dxy_dcamera * SE3_left_jacobian(se3_world_camera);

    if (dxy_dcamera_ptr) {
        *dxy_dcamera_ptr = dxy_dcamera_global;
    }

    if (dxy_dobject_ptr) {
        *dxy_dobject_ptr = dxy_dobject_global;
    }

    return projected_image_point;
}

double camera_project_factor(
    const Eigen::VectorD<16>& input,
    const size_t num_points,
    const Eigen::VectorD<4>* object_points,
    const Eigen::VectorD<2>* image_points,
    Eigen::MatrixD<16>* JtJ_ptr,
    Eigen::VectorD<16>* Jtr_ptr) {

    // todo : make the other impl call down to this one
    Eigen::VectorD<4> camparams = input.head<4>();
    Eigen::VectorD<6> se3_world_camera = input.segment<6>(4);
    Eigen::VectorD<6> se3_world_object = input.tail<6>();
    return camera_project_factor(camparams,
                                 se3_world_camera,
                                 se3_world_object,
                                 num_points,
                                 object_points,
                                 image_points,
                                 JtJ_ptr, Jtr_ptr);
}

double camera_project_factor(
    const Eigen::VectorD<4>& fxfycxcy,
    const Eigen::VectorD<6>& se3_world_camera,
    const Eigen::VectorD<6>& se3_world_object,
    const std::vector<Eigen::VectorD<4>>& object_points,
    const std::vector<Eigen::VectorD<2>>& image_points,
    Eigen::MatrixD<16>* JtJ_ptr,
    Eigen::VectorD<16>* Jtr_ptr) {
    assert(object_points.size() == image_points.size());
    assert(!object_points.empty());
    return camera_project_factor(
        fxfycxcy,
        se3_world_camera,
        se3_world_object,
        object_points.size(),
        object_points.data(),
        image_points.data(),
        JtJ_ptr, Jtr_ptr);
}

double camera_project_factor(
    const Eigen::VectorD<4>& fxfycxcy,
    const Eigen::VectorD<6>& se3_world_camera,
    const Eigen::VectorD<6>& se3_world_object,
    size_t num_points,
    const Eigen::VectorD<4>* object_points,
    const Eigen::VectorD<2>* image_points,
    Eigen::MatrixD<16>* JtJ_ptr,
    Eigen::VectorD<16>* Jtr_ptr) {

    auto se3_camera_world = (-se3_world_camera).eval();
    auto tx_camera_world = se3_exp(se3_camera_world);
    auto tx_world_object = se3_exp(se3_world_object);

    // r := z - f(x0)
    // || J dx + f(x0) - z ||^2
    // || J dx - r ||^2
    // dx.t J.t J dx - 2 r.t J dx + r.t r

    if (JtJ_ptr) {
        JtJ_ptr->setZero();
    }
    if (Jtr_ptr) {
        Jtr_ptr->setZero();
    }

    double error2 = 0;

    // std::cout << "in camera project factor \n";    
    for (size_t i = 0; i < num_points; ++i) {
        Eigen::MatrixD<2, 4> dxy_dcamparams;
        Eigen::MatrixD<2, 6> dxy_dcamera;
        Eigen::MatrixD<2, 6> dxy_dobject;

        Eigen::VectorD<2> projected_img_point = camera_project(
            fxfycxcy,
            tx_camera_world,
            se3_world_camera,
            tx_world_object,
            se3_world_object,
            object_points[i],
            &dxy_dcamparams,
            &dxy_dcamera,
            &dxy_dobject);

        // std::cout << "\timage point " << i << ": " << image_points[i].transpose() << "\n";
        // std::cout << "\tobject point " << i << ": " << object_points[i].transpose() << "\n";
        // std::cout << "\tprojected point " << i << ": " << projected_img_point.transpose() << "\n";

        auto residual = image_points[i] - projected_img_point;
        // std::cout << "\terr2 " << residual.squaredNorm() << "\n";


        error2 += residual.squaredNorm();

        Eigen::MatrixD<2, 16> J;
        J.block<2, 4>(0, 0) = dxy_dcamparams;
        J.block<2, 6>(0, 4) = dxy_dcamera;
        J.block<2, 6>(0, 10) = dxy_dobject;

        if (JtJ_ptr) {
            (*JtJ_ptr) += J.transpose() * J;
        }
        if (Jtr_ptr) {
            (*Jtr_ptr) += J.transpose() * residual;
        }
    }

    return error2;
}

}  // carl
