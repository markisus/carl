#include <iostream>
#include <Eigen/Dense>
#include "factor_graph.h"
#include "matrix_mem_eigen.h"

using namespace carl;

template <int dim>
void add_prior(FactorGraph& graph, VariableId variable_id,
               const Eigen::Matrix<double, dim, 1>& prior_mean,
               const Eigen::Matrix<double, dim, dim>& prior_cov) {

    const Eigen::Matrix<double, dim, 1> info_vec = prior_cov.ldlt().solve(prior_mean);
    const Eigen::Matrix<double, dim, dim> info_mat = prior_cov.inverse();

    const FactorId prior_factor_id = graph.factors.of_dimension(dim).create_factor(ref(info_mat), ref(info_vec));
    graph.msg_buffers_of_dimension(dim).add_edge(variable_id, prior_factor_id, 0);

    
}


template <size_t dim>
Eigen::Matrix<double, dim, 1> to_eigen_vec(const std::initializer_list<double>& a) {
    if (a.size() != dim) {
        std::cout << "size mismatch in to_eigen_vec" << "\n";
        exit(-1);
    }
    
    Eigen::Matrix<double, dim, 1> result;

    size_t i = 0;
    for (double d : a) {
        result(i) = d;
        ++i;
    }
    return result;
}

template <size_t dim>
Eigen::Matrix<double, dim, dim> to_eigen_diagonal_cov(const std::initializer_list<double>& a) {
    if (a.size() != dim) {
        std::cout << "size mismatch in to_eigen_diagonal_cov" << "\n";
        exit(-1);
    }
    
    Eigen::Matrix<double, dim, dim> result;
    result.setZero();

    size_t i = 0;
    for (double d : a) {
        result(i,i) = d;
        ++i;
    }

    return result;
}

template <size_t dim>
Eigen::Matrix<double, dim, dim> to_eigen_cov(const std::initializer_list<double>& a) {
    if (a.size() != dim*dim) {
        std::cout << "size mismatch in to_eigen_cov" << "\n";
        exit(-1);
    }
    
    Eigen::Matrix<double, dim, dim> result;

    size_t i = 0;
    for (double d : a) {
        result.data()[i] = d;
        ++i;
    }

    return result;
}

int main(int argc, char *argv[])
{
    std::cout << "Updating graph" << "\n";

    FactorGraph graph;

    auto var1_mean = to_eigen_vec<6>({1, 0, 0, 0, 0, 0});
    auto var1_cov = to_eigen_diagonal_cov<6>({1, 1, 1, 1, 1, 1});

    auto var2_mean = to_eigen_vec<6>({0, 1, 0, 0, 0, 0});
    auto var2_cov = to_eigen_diagonal_cov<6>({1, 1, 1, 1, 1, 1});
    
    VariableId var1 = graph.vars_of_dimension(6).create_variable();
    VariableId var2 = graph.vars_of_dimension(6).create_variable();
    add_prior(graph, var1, var1_mean, var1_cov);
    add_prior(graph, var2, var2_mean, var2_cov);

    // connect these two with a factor
    Eigen::Matrix<double, 6, 12> J;
    J.block<6,6>(0,0).setIdentity();
    J.block<6,6>(0,6).setIdentity();
    J.block<6,6>(0,6) *= -1;
    
    Eigen::Matrix<double, 12, 12> info_mat = J.transpose() * J;
    std::cout << "info mat\n" << info_mat << "\n";

    
    info_mat *= 1000;
    
    Eigen::Matrix<double, 12, 1> info_vec;
    info_vec.setZero();

    FactorId binary_factor = graph.factors.of_dimension(12).create_factor(ref(info_mat),
                                                                          ref(info_vec));
    graph.add_edge(var1, binary_factor, 0);
    graph.add_edge(var2, binary_factor, 6);
    
    graph.commit();
    graph.initial_update();

    for (size_t i = 0; i < 100; ++i)
    {
        graph.update();
        auto [mean1, cov1] = graph.get_mean_cov(var1);
        auto [mean2, cov2] = graph.get_mean_cov(var2);
        std::cout << "mean1\n " << to_eigen_map(static_dim<6>(mean1)).transpose() << "\n";
        std::cout << "mean2\n " << to_eigen_map(static_dim<6>(mean2)).transpose() << "\n";
    }

    return 0;
}
