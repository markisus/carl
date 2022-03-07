#include <iostream>
#include "gtest/gtest.h"
#include "math.hpp"

TEST(info_marginalization_fixer, test1) {
    Eigen::Matrix<double, 4, 4> info_mat;
    info_mat <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6,
        5, 4, 3, 2;
    info_mat = info_mat.transpose() * info_mat;
    info_mat += Eigen::Matrix<double, 4, 4>::Identity() * 10;

    Eigen::Matrix<double, 4, 1> info_vec;
    info_vec << -1, 0, 1, 2;

    Eigen::Matrix<double, 4, 4> cov_mat = info_mat.inverse();
    Eigen::Matrix<double, 4, 1> mean = info_mat.ldlt().solve(info_vec);

    // marginalize the 2x2 top left block
    Eigen::Matrix<double, 2, 2> marginal_cov = cov_mat.block<2,2>(0,0);
    Eigen::Matrix<double, 2, 1> marginal_mean = mean.segment<2>(0);

    // perturb the info matrix and info vector
    Eigen::Matrix<double, 2, 2> info_add;
    info_add <<
        1, 2,
        2, 1;
    Eigen::Matrix<double, 2, 1> info_vec_add;
    info_vec_add << 5, 4;
    Eigen::Matrix<double, 4, 4> info_mat_plus = info_mat;
    info_mat_plus.block<2,2>(0,0) += info_add;
    Eigen::Matrix<double, 4, 1> info_vec_plus = info_vec;
    info_vec_plus.segment<2>(0) += info_vec_add;

    // marginalize again
    Eigen::Matrix<double, 4, 4> cov_mat_plus = info_mat_plus.inverse();
    Eigen::Matrix<double, 4, 1> mean_plus = info_mat_plus.ldlt().solve(info_vec_plus);
    Eigen::Matrix<double, 2, 2> marginal_cov_plus = cov_mat_plus.block<2,2>(0, 0);
    Eigen::Matrix<double, 2, 1> marginal_mean_plus = mean_plus.segment<2>(0);
    
    // use the fixer matrix to marginalize again
    Eigen::Matrix<double, 2, 2> fixer = info_marginalization_fixer(marginal_cov,
                                                                   info_add);
    Eigen::Matrix<double, 2, 2> marginal_cov_fixed = perturb_marginalization(fixer, marginal_cov);
    Eigen::Matrix<double, 2, 1> marginal_mean_fixed = perturb_marginalization(fixer, marginal_cov_fixed, info_vec_add, marginal_mean);
    EXPECT_LT((marginal_cov_fixed - marginal_cov_plus).norm(), 1e-6);
    EXPECT_LT((marginal_mean_fixed - marginal_mean_plus).norm(), 1e-6);
}

TEST(low_rank_update, test1) {
    Eigen::Matrix<double, 4, 4> info_mat;
    info_mat <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6,
        5, 4, 3, 2;
    info_mat = info_mat.transpose() * info_mat;
    info_mat += Eigen::Matrix<double, 4, 4>::Identity() * 10;

    Eigen::Matrix<double, 4, 1> info_vec;
    info_vec << -1, 0, 1, 2;

    Eigen::Matrix<double, 4, 4> cov_mat = info_mat.inverse();
    Eigen::Matrix<double, 4, 1> mean = info_mat.ldlt().solve(info_vec);

    // perturb the info matrix and info vector
    Eigen::Matrix<double, 2, 2> info_add;
    info_add <<
        1, 2,
        2, 1;
    Eigen::Matrix<double, 2, 1> info_vec_add;
    info_vec_add << 5, 4;

    Eigen::Matrix<double, 4, 4> info_mat_plus = info_mat;
    info_mat_plus.block<2,2>(0,0) += info_add;
    Eigen::Matrix<double, 4, 1> info_vec_plus = info_vec;
    info_vec_plus.segment<2>(0) += info_vec_add;

    Eigen::Matrix<double, 4, 4> cov_mat_plus = info_mat_plus.inverse();
    Eigen::Matrix<double, 4, 1> mean_plus = info_mat_plus.ldlt().solve(info_vec_plus);

    Eigen::Map<Eigen::MatrixXd> info_add_map {info_add.data(), 2, 2};
    Eigen::Map<Eigen::VectorXd> info_add_vec_map {info_vec_add.data(), 2, 1};

    low_rank_update(info_add_map, info_add_vec_map, 0, cov_mat, mean);

    EXPECT_LT((cov_mat_plus - cov_mat).norm(), 1e-6);
    EXPECT_LT((mean_plus - mean).norm(), 1e-6);    
}

TEST(low_rank_update, test2) {
    Eigen::Matrix<double, 4, 4> info_mat;
    info_mat <<
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 8, 7, 6,
        5, 4, 3, 2;
    info_mat = info_mat.transpose() * info_mat;
    info_mat += Eigen::Matrix<double, 4, 4>::Identity() * 10;

    Eigen::Matrix<double, 4, 1> info_vec;
    info_vec << -1, 0, 1, 2;

    Eigen::Matrix<double, 4, 4> cov_mat = info_mat.inverse();
    Eigen::Matrix<double, 4, 1> mean = info_mat.ldlt().solve(info_vec);

    // perturb the info matrix and info vector
    Eigen::Matrix<double, 2, 2> info_add;
    info_add <<
        1, 2,
        2, 1;
    Eigen::Matrix<double, 2, 1> info_vec_add;
    info_vec_add << 5, 4;

    Eigen::Matrix<double, 4, 4> info_mat_plus = info_mat;
    info_mat_plus.block<2,2>(2,2) += info_add;
    Eigen::Matrix<double, 4, 1> info_vec_plus = info_vec;
    info_vec_plus.segment<2>(2) += info_vec_add;

    Eigen::Matrix<double, 4, 4> cov_mat_plus = info_mat_plus.inverse();
    Eigen::Matrix<double, 4, 1> mean_plus = info_mat_plus.ldlt().solve(info_vec_plus);

    Eigen::Map<Eigen::MatrixXd> info_add_map {info_add.data(), 2, 2};
    Eigen::Map<Eigen::VectorXd> info_add_vec_map {info_vec_add.data(), 2, 1};

    low_rank_update(info_add_map, info_add_vec_map, 2, cov_mat, mean);

    EXPECT_LT((cov_mat_plus - cov_mat).norm(), 1e-6);
    EXPECT_LT((mean_plus - mean).norm(), 1e-6);    
}
