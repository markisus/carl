#include "gaussian_math.hpp"
#include "benchmark/benchmark.h"
#include <memory>

using namespace carl;

template <typename T>
T* non_const(const T* t) {
    return const_cast<T*>(t);
}

template <int dim>
Eigen::Matrix<double, dim, dim> random_pos_def() {
    Eigen::Matrix<double, dim, dim> pos_def =
        Eigen::Matrix<double, dim, dim>::Random();
    pos_def = (pos_def.transpose() * pos_def).eval();
    pos_def += Eigen::Matrix<double, dim, dim>::Identity();
    return pos_def;
}

template <int dim>
Eigen::Matrix<double, dim, 1> random_vec() {
    return Eigen::Matrix<double, dim, 1>::Random();
}

static void BM_cov_update_naive(benchmark::State& state) {
    const auto info_mat = random_pos_def<16>();
    const auto info_vec = random_vec<16>();
    const auto info_mat_add = random_pos_def<6>();
    const auto info_vec_add = random_vec<6>();
    for (auto _ : state) {
        Eigen::Matrix<double, 16, 16> info_mat_perturbed = info_mat;
        info_mat_perturbed.block<6,6>(0,0) += info_mat_add;
        Eigen::Matrix<double, 16, 1> info_vec_perturbed = info_vec;
        info_vec_perturbed.segment<6>(0) += info_vec_add;
        Eigen::Matrix<double, 16, 16> covariance = info_mat_perturbed.inverse();
        Eigen::Matrix<double, 16, 1> mean = info_mat_perturbed.ldlt().solve(info_vec_perturbed);
        benchmark::DoNotOptimize(mean.maxCoeff());
        benchmark::DoNotOptimize(covariance.trace());
    }
}
BENCHMARK(BM_cov_update_naive);

static void BM_cov_update_naive2(benchmark::State& state) {
    const auto info_mat = random_pos_def<16>();
    const auto info_vec = random_vec<16>();
    const auto info_mat_add = random_pos_def<6>();
    const auto info_vec_add = random_vec<6>();
    for (auto _ : state) {
        Eigen::Matrix<double, 16, 16> info_mat_perturbed = info_mat;
        info_mat_perturbed.block<6,6>(0,0) += info_mat_add;
        Eigen::Matrix<double, 16, 1> info_vec_perturbed = info_vec;
        info_vec_perturbed.segment<6>(0) += info_vec_add;

        auto llt = info_mat_perturbed.llt();

        Eigen::Matrix<double, 16, 16> covariance = llt.solve(Eigen::Matrix<double, 16, 16>::Identity());
        Eigen::Matrix<double, 16, 1> mean = llt.solve(info_vec_perturbed);
        benchmark::DoNotOptimize(mean.maxCoeff());
        benchmark::DoNotOptimize(covariance.trace());
    }
}
BENCHMARK(BM_cov_update_naive2);

static void BM_cov_inverse(benchmark::State& state) {
    const auto info_mat = random_pos_def<16>();
    for (auto _ : state) {
        auto inverse = info_mat.inverse().eval();
        benchmark::DoNotOptimize(inverse.trace());
    }
}
BENCHMARK(BM_cov_inverse);

static void BM_cov_llt_inverse(benchmark::State& state) {
    const auto info_mat = random_pos_def<16>();
    for (auto _ : state) {
        auto inverse = info_mat.llt().solve(Eigen::Matrix<double, 16, 16>::Identity()).eval();
        benchmark::DoNotOptimize(inverse.trace());
    }
}
BENCHMARK(BM_cov_llt_inverse);

static void BM_cov_update_low_rank(benchmark::State& state) {
    const auto info_mat = random_pos_def<16>();
    const auto info_vec = random_vec<16>();
    const auto mean_ = info_mat.ldlt().solve(info_vec);
    const auto covariance_ = info_mat.inverse();
    const auto info_mat_add = random_pos_def<6>();
    const auto info_vec_add = random_vec<6>();
    for (auto _ : state) {
        Eigen::Matrix<double, 16, 16> covariance = covariance_;
        Eigen::Matrix<double, 16, 1> mean = mean_;
        Eigen::Map<Eigen::MatrixXd> info_mat_add_map { non_const(info_mat_add.data()), 6, 6 };
        Eigen::Map<Eigen::VectorXd> info_vec_add_map { non_const(info_vec_add.data()), 6, 1 };
        low_rank_update(
            info_mat_add_map,
            info_vec_add_map,
            0, covariance, mean);

        benchmark::DoNotOptimize(mean.maxCoeff());
        benchmark::DoNotOptimize(covariance.trace());
    }
}
BENCHMARK(BM_cov_update_low_rank);

constexpr int mat_dim = 4;
constexpr int num_els = 500;

static void BM_mat_multiply_noncontiguous(benchmark::State& state) {
    std::vector<std::shared_ptr<Eigen::Matrix<double, mat_dim, mat_dim>>> mats;
    std::vector<std::shared_ptr<Eigen::Matrix<double, mat_dim, 1>>> vecs;
    std::vector<std::shared_ptr<Eigen::Matrix<double, mat_dim, 1>>> vecs_dst;
    for (int i = 0; i < num_els; ++i) {
        mats.push_back(std::make_shared<Eigen::Matrix<double, mat_dim, mat_dim>>(Eigen::Matrix<double, mat_dim, mat_dim>::Random().eval()));
        vecs.push_back(std::make_shared<Eigen::Matrix<double, mat_dim, 1>>(Eigen::Matrix<double, mat_dim, 1>::Random().eval()));
        vecs_dst.push_back(std::make_shared<Eigen::Matrix<double, mat_dim, 1>>());
    }

    for (auto _ : state) {
        for (int i = 0; i < num_els; ++i) {
            (*vecs_dst[i]) = (*mats[i]) * (*vecs[i]);
        }
        double use_result = 0;
        for (auto& vec : vecs_dst) {
            use_result += (*vec).sum();
        }
        benchmark::DoNotOptimize(use_result);
    }
}
BENCHMARK(BM_mat_multiply_noncontiguous);


static void BM_mat_multiply_dynamic(benchmark::State& state) {
    std::vector<Eigen::MatrixXd> mats;
    std::vector<Eigen::VectorXd> vecs;
    std::vector<Eigen::VectorXd> vecs_dst;
    for (int i = 0; i < num_els; ++i) {
        mats.push_back(Eigen::Matrix<double, mat_dim, mat_dim>(Eigen::Matrix<double, mat_dim, mat_dim>::Random().eval()));
        vecs.push_back(Eigen::Matrix<double, mat_dim, 1>(Eigen::Matrix<double, mat_dim, 1>::Random().eval()));
        vecs_dst.push_back(Eigen::Matrix<double, mat_dim, 1>());
    }

    for (auto _ : state) {
        for (int i = 0; i < num_els; ++i) {
            vecs_dst[i] = mats[i] * vecs[i];
        }
        double use_result = 0;
        for (auto& vec : vecs_dst) {
            use_result += vec.sum();
        }
        benchmark::DoNotOptimize(use_result);
    }
}
BENCHMARK(BM_mat_multiply_dynamic);


static void BM_mat_multiply_contiguous(benchmark::State& state) {
    std::vector<Eigen::Matrix<double, mat_dim, mat_dim>> mats;
    std::vector<Eigen::Matrix<double, mat_dim, 1>> vecs;
    std::vector<Eigen::Matrix<double, mat_dim, 1>> vecs_dst;
    for (int i = 0; i < num_els; ++i) {
        mats.push_back(Eigen::Matrix<double, mat_dim, mat_dim>::Random().eval());
        vecs.push_back(Eigen::Matrix<double, mat_dim, 1>::Random().eval());
        vecs_dst.push_back(Eigen::Matrix<double, mat_dim, 1>{});
    }

    for (auto _ : state) {
        for (int i = 0; i < num_els; ++i) {
            (vecs_dst)[i] = (mats[i]) * (vecs[i]);
        }
        double use_result = 0;
        for (auto& vec : vecs_dst) {
            use_result += vec.sum();
        }
        benchmark::DoNotOptimize(use_result);
    }
}
BENCHMARK(BM_mat_multiply_contiguous);

static void BM_map_mat_multiply_contiguous(benchmark::State& state) {
    std::vector<Eigen::Matrix<double, mat_dim, mat_dim>> mats;
    std::vector<Eigen::Matrix<double, mat_dim, 1>> vecs;
    std::vector<Eigen::Matrix<double, mat_dim, 1>> vecs_dst;
    for (int i = 0; i < num_els; ++i) {
        mats.push_back(Eigen::Matrix<double, mat_dim, mat_dim>::Random().eval());
        vecs.push_back(Eigen::Matrix<double, mat_dim, 1>::Random().eval());
        vecs_dst.push_back(Eigen::Matrix<double, mat_dim, 1>{});
    }

    for (auto _ : state) {
        for (int i = 0; i < num_els; ++i) {
            Eigen::Map<Eigen::Matrix<double, mat_dim, mat_dim>> mat = {mats[i].data(), mat_dim, mat_dim};
            Eigen::Map<Eigen::Matrix<double, mat_dim, 1>> vec = {vecs[i].data(), mat_dim, 1};
            Eigen::Map<Eigen::Matrix<double, mat_dim, 1>> vec_dst = {vecs_dst[i].data(), mat_dim, 1};
            vec_dst = mat * vec;
        }
        double use_result = 0;
        for (auto& vec : vecs_dst) {
            use_result += vec.sum();
        }
        benchmark::DoNotOptimize(use_result);
    }
}
BENCHMARK(BM_map_mat_multiply_contiguous);



BENCHMARK_MAIN();
