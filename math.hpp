#pragma once
#include "Eigen/Dense"
#include <iostream>

namespace carl {

// NOTATION =======================================================================================
// In what follows, the matrices and vectors are assumed to follow some conformable block structure.
// Here are some useful operators for dealing with blocks.
// 
// (.)ᵢ                  is the ith diagonal block of a matrix, or ith block of a vector
// (.)ₓᵢ                 is the ith column block of a matrix, taking all rows
// Eᵢ                    is a column block matrix of zero blocks, except the ith block is identity
//
// You can check that these properties hold.
// - For all matrices M:  Mᵢ  ≣ Eᵢ.t M Eᵢ
// - For all matrices M:  Mₓᵢ ≣ M Eᵢ
// - For all vectors  v:  vᵢ  ≣ Eᵢ.t v
// - Eᵢ M Eᵢ.T lifts a matrix M into a larger matrix whose ith block is M
//   and other blocks are 0.
// - Eᵢ v lifts a vector v into a larger vector whose ith block is v, and
//   other blocks are 0.

// see info_marginalization_fixer() for detailed explanation
// returns the new marginalized mean after perturbing the source information matrix
template <int dim>
Eigen::Matrix<double, dim, 1> perturb_marginalization(
    const Eigen::Matrix<double, dim, dim>& fixer,
    const Eigen::Matrix<double, dim, dim>& marginal_cov_fixed,    
    const Eigen::Matrix<double, dim, 1>& info_vec_add,
    const Eigen::Matrix<double, dim, 1>& marginal_mean) {
    return marginal_mean - fixer * marginal_mean + marginal_cov_fixed * info_vec_add;
}

// see info_marginalization_fixer() for detailed explanation
// returns the new marginalized covariance after perturbing the source information matrix
template <int dim>
Eigen::Matrix<double, dim, dim> perturb_marginalization(
    const Eigen::Matrix<double, dim, dim>& fixer,
    const Eigen::Matrix<double, dim, dim>& marginal_cov) {
    return marginal_cov - fixer * marginal_cov;
}

template <int dim>
Eigen::Matrix<double, dim, dim> info_marginalization_fixer(
    const Eigen::Matrix<double, dim, dim>& marginal_covariance,
    const Eigen::Matrix<double, dim, dim>& info_perturb) {
    // MOTIVATION =============================================================================================
    // Consider a covariance matrix and mean resulting from marginalization of the ith block of information
    // matrix and information vector. If the information matrix / vector's ith block is perturbed, we want to know
    // how the resulting covariance matrix / mean changes.
    // 
    // Λ                     is an information matrix 
    // Σ  = Λ⁻¹              is the covariance matrix corresponding to that information matrix
    // Σᵢ = Λ⁻¹ᵢ             is the marginal covariance of the variable corresponding to the ith block
    // 
    // ν                     is the information vector
    // μ  = Λ⁻¹ ν            is the mean corresponding to the information matrix and vector
    // μᵢ = (Λ'⁻¹ ν)ᵢ        is the marginal mean of the variable corresponding to the ith block
    // 
    // If we perturb the information matrix Λ by adding a matrix V to its ith block, we will end up
    // affecting the marginal covariance of the ith variable.
    // Λ' = Λ + Eᵢ V Eᵢ.t    is the perturbed information matrix
    // Σ' = Λ'⁻¹             is the perturbed covariance matrix
    // Σ'ᵢ = Λ'⁻¹ᵢ           is the perturbed marginal covariance of variable corresponding to the ith block
    //
    // Naively, we would have to invert the block matrix Λ' to compute Σ'ᵢ. But we can do better by taking
    // advantage of the Woodbury Matrix Identity https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    // Σ'ᵢ = Λ'⁻¹ᵢ
    //     = (Λ + Eᵢ V Eᵢ.t)⁻¹ᵢ                                     
    //     = [Λ⁻¹ - Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t Λ⁻¹]ᵢ        Woodbury Formula
    //     = [Λ⁻¹ - Λ⁻¹ Eᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ Eᵢ.t Λ⁻¹]ᵢ               collapse definition of (.)ᵢ operator
    //     = Λ⁻¹ᵢ - (Λ⁻¹ Eᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ Eᵢ.t Λ⁻¹)ᵢ              distribute (.)ᵢ operator
    //     = Λ⁻¹ᵢ - Eᵢ.t Λ⁻¹ Eᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ Eᵢ.t Λ⁻¹) Eᵢ        expand definition of (.)ᵢ operator
    //     = Λ⁻¹ᵢ - Λ⁻¹ᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ Λ⁻¹ᵢ                       collapse definition of (.)ᵢ operator
    //     = Σᵢ - Σᵢ (V⁻¹ + Σᵢ)⁻¹ Σᵢ                               subst Λ⁻¹ᵢ <= Σᵢ
    //     
    // Therefore, we can know the updated covariance by only using the previous covariance Σᵢ and perturbation V.
    // A similar effect occurs with the marginalized mean.
    // 
    // ν'  = ν + Eᵢ v        is the perturbed information vector
    // μ'ᵢ = (Λ'⁻¹ ν')ᵢ      is the perturbed marginal mean of the variable corresponding to the ith block
    //
    // Again, use the Woodbury Matrix Identity.
    // μ'ᵢ = Eᵢ.t (Λ'⁻¹ ν')                                                         expand definition of (.)ᵢ
    //     = Eᵢ.t Λ'⁻¹ (ν + Eᵢ v)                                                   subst ν' = ν + Eᵢ v
    //     = Eᵢ.t Λ'⁻¹ ν + Eᵢ.t Λ'⁻¹ Eᵢ v                                           distribute
    //     = Eᵢ.t Λ'⁻¹ ν + Λ'⁻¹ᵢ v                                                  collapse definition of (.)ᵢ operator
    //     = Eᵢ.t (Λ + Eᵢ V Eᵢ.t)⁻¹ ν + Λ'⁻¹ᵢ v                                     subst Λ' <= Λ + Eᵢ V Eᵢ.t
    //     = Eᵢ.t [Λ⁻¹ - Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t Λ⁻¹] ν + Λ'⁻¹ᵢ v         Woodbury Formula
    //     = Eᵢ.t Λ⁻¹ ν - Eᵢ.t Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t Λ⁻¹ ν + Λ'⁻¹ᵢ v    distribute ν
    //     = Eᵢ.t μ - Eᵢ.t Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t μ  + Λ'⁻¹ᵢ v           subst Λ⁻¹ ν <= μ
    //     = μᵢ - Λ⁻¹ᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ μᵢ + Λ'⁻¹ᵢ v                                  (.)ᵢ operator
    //     = μᵢ - Σᵢ (V⁻¹ + Σᵢ)⁻¹ μᵢ + Σ'ᵢ v                                        subst Λ⁻¹ᵢ <= Σᵢ, Λ'⁻¹ᵢ <= Σᵢ'
    //
    // Therefore, we can know the updated mean by only using the previous marginal covariance Σᵢ and mean μᵢ,
    // updated marginal covariance Σᵢ' and perturbation v.
    //
    // In both mean and covariance cases, the important quantity Σᵢ (V⁻¹ + Σᵢ)⁻¹ arises. We refer this to the fixer
    // matrix and return it as an intermediary.

    const Eigen::Matrix<double, dim, dim> fixer_mat = marginal_covariance * (info_perturb.inverse() + marginal_covariance).inverse();
    return fixer_mat;
}

// benchmark shows naive version is faster
// so this needs to be optimized before using it
template <int dim>
void low_rank_update(
    const Eigen::Map<Eigen::MatrixXd>& info_mat_perturb,
    const Eigen::Map<Eigen::VectorXd>& info_vec_perturb,
    uint8_t slot,
    Eigen::Matrix<double, dim, dim>& covariance,
    Eigen::Matrix<double, dim, 1>& mean) {

    assert(info_mat_perturb.rows() == info_mat_perturb.cols());
    assert(info_mat_perturb.rows() == info_vec_perturb.size());
    assert(info_vec_perturb.size() <= dim);
    
    // MOTIVATION =============================================================================================
    // Consider a covariance matrix and mean resulting from transformation of an information matrix and information
    // vector. If the information matrix / vector's ith block is perturbed, we want to know how the resulting
    // covariance matrix / mean changes.
    // 
    // Λ                     is an information matrix 
    // Σ  = Λ⁻¹              is the covariance matrix corresponding to that information matrix
    // 
    // ν                     is the information vector
    // μ  = Λ⁻¹ ν            is the mean corresponding to the information matrix and vector
    // 
    // If we perturb the information matrix Λ by adding a matrix V to its ith block, we will end up
    // affecting the marginal covariance of the ith variable.
    // Λ' = Λ + Eᵢ V Eᵢ.t    is the perturbed information matrix
    // Σ' = Λ'⁻¹             is the perturbed covariance matrix
    //
    // Naively, we would have to invert the block matrix Λ' to compute Σ'. But we can do better by taking
    // advantage of the Woodbury Matrix Identity https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    // Σ' = Λ'⁻¹
    //    = (Λ + Eᵢ V Eᵢ.t)⁻¹                                     
    //    = Λ⁻¹ - Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t Λ⁻¹           Woodbury Formula
    //    = Λ⁻¹ - Λ⁻¹ Eᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ Eᵢ.t Λ⁻¹                  collapse definition of (.)ᵢ operator
    //    = Σ - Σ Eᵢ (V⁻¹ + Σᵢ)⁻¹ Eᵢ.t Σ                          subst Λ⁻¹ <= Σ
    //    = Σ - Σₓᵢ (V⁻¹ + Σᵢ)⁻¹ Σᵢₓ                              (.)ᵢ operator
    //    
    // Therefore, we can know the updated covariance by only using the prevous covariance Σ and perturbation V.
    // A similar effect occurs with the marginalized mean.
    // 
    // ν' = ν + Eᵢ v        is the perturbed information vector
    // μ' = Λ'⁻¹ ν'         is the perturbed marginal mean of the variable corresponding to the ith block
    //
    // Again, use the Woodbury Matrix Identity.
    // μ' = Λ'⁻¹ ν'                                                             expand definition of (.)ᵢ
    //    = Λ'⁻¹ (ν + Eᵢ v)                                                     subst ν' = ν + Eᵢ v
    //    = Λ'⁻¹ ν + Λ'⁻¹ Eᵢ v                                                  distribute
    //    = (Λ + Eᵢ V Eᵢ.t)⁻¹ ν + Λ'⁻¹ Eᵢ v                                     subst Λ' <= Λ + Eᵢ V Eᵢ.t
    //    = [Λ⁻¹ - Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t Λ⁻¹] ν + Λ'⁻¹ Eᵢ v         Woodbury Formula
    //    = Λ⁻¹ ν - Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t Λ⁻¹ ν + Λ'⁻¹ Eᵢ v         distribute ν
    //    = μ - Λ⁻¹ Eᵢ (V⁻¹ + Eᵢ.t Λ⁻¹ Eᵢ)⁻¹ Eᵢ.t μ  + Λ'⁻¹ Eᵢ v                subst Λ⁻¹ ν <= μ
    //    = μ - Λ⁻¹ Eᵢ (V⁻¹ + Λ⁻¹ᵢ)⁻¹ μᵢ + Λ'⁻¹ Eᵢ v                            (.)ᵢ operator
    //    = μ - Σ Eᵢ (V⁻¹ + Σᵢ)⁻¹ μᵢ + Σ' Eᵢ v                                  subst Λ⁻¹ <= Σ, Λ'⁻¹ <= Σ'
    //    = μ - Σₓᵢ (V⁻¹ + Σᵢ)⁻¹ μᵢ + Σ'ₓᵢ v                                    (.)ᵢ operator
    //
    // Therefore, we can know the updated mean by only using the previous marginal covariance Σ and mean μ,
    // updated marginal covariance Σ' and perturbation v.

    auto sigma_xi = covariance.block(0, slot, dim, info_mat_perturb.cols());

    // double buff1[dim*dim];
    // Eigen::Map<Eigen::MatrixXd> kmat = {buff1, info_mat_perturb.rows(), info_mat_perturb.cols()};

    double buff2[dim*dim];
    Eigen::Map<Eigen::MatrixXd> lmat = {buff2, dim, info_mat_perturb.cols()};
    lmat.noalias() = sigma_xi * (info_mat_perturb.inverse() + covariance.block(slot, slot, info_mat_perturb.rows(), info_mat_perturb.cols())).inverse();

    auto mu_xi = mean.segment(slot, info_vec_perturb.size());

    const Eigen::Matrix<double, dim, 1> mean_update_a = -lmat * mu_xi;
    mean += mean_update_a;

    const Eigen::Matrix<double, dim, dim> cov_update = -lmat * sigma_xi.transpose();
    covariance += cov_update;

    // sigma_xi has now been mutated via underlying reference to covariance
    mean += sigma_xi * info_vec_perturb;
}

// // info getter
// // exp( || x - mu ||^2_sigma ) * exp( || f(x) - z ||^2_sigma
// // unpacking a correlation matrix
// template <int dim_state, int dim_meas>
// void stats_to_factor_info(
//     const Eigen::Matrix<double, dim_state + dim_meas, dim_state + dim_meas>& joint_covariance,
//     const Eigen::Matrix<double, dim_state + dim_meas, 1>& joint_mean,
//     const Eigen::Matrix<double, dim_meas, 1>& measurement,
//     Eigen::Matrix<double, dim_state, dim_state>* factor_info_mat,
//     Eigen::Matrix<double, dim_state, 1>* factor_info_vec) {

//     Eigen::Matrix<double, dim_state + dim_meas, dim_state + dim_meas> info_mat = covariance.inverse();
//     Eigen::Matrix<double, dim_state + dim_meas, dim_state + dim_meas> info_vec = covariance.ldlt().solve(mean);

//     // measurement
//     *factor_info_mat = info_mat.block<dim_state, dim_state>(0,0);
//     *factor_info_vec = -info_mat.block<dim_state, dim_meas>(0, dim_state) * measurement;
// }

// // get joint covariance
// // input samples
// // output samples
// // 

}
