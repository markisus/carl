#include <iostream>
#include <cstdint>

#include "factor_graph.hpp"

using namespace carl;

int main(int argc, char *argv[])
{
    FactorGraph<std::integer_sequence<uint8_t, 4>,
                std::integer_sequence<uint8_t, 2>> graph;

    Mean<2> mean1;
    Mean<2> mean2;
    Mean<2> mean3;
    mean1 << 1, 0;
    mean2 << 0, 1;
    mean3 << -1, -1;
    auto covariance1 = Covariance<2>::Identity().eval();
    auto covariance2 = Covariance<2>::Identity().eval();
    auto covariance3 = Covariance<2>::Identity().eval();

    covariance1 *= 1e2;
    covariance2 *= 1e2;
    covariance3 *= 1e-2;

    const auto variable1 = graph.add_variable(0, mean1, covariance1);
    const auto variable2 = graph.add_variable(0, mean2, covariance2);
    const auto variable3 = graph.add_variable(0, mean3, covariance3);

    Eigen::Matrix<double, 2, 4> J;
    J.block<2,2>(0,0).setIdentity();
    J.block<2,2>(0,J.cols()/2).setIdentity();
    J.block<2,2>(0,J.cols()/2) *= -1;
    InfoMatrix<4> info_mat = J.transpose() * J;
    InfoVector<4> info_vec = InfoVector<4>::Zero();

    {
        const auto factor = graph.add_factor(0, false, info_vec, info_mat);
        graph.add_edge(variable1, factor, 0);
        graph.add_edge(variable2, factor, info_vec.size()/2);
    }

    {
        const auto factor = graph.add_factor(0, false, info_vec, info_mat);
        graph.add_edge(variable1, factor, 0);
        graph.add_edge(variable3, factor, info_vec.size()/2);
    }
    
    for (size_t i = 1; i < 10; ++i) {
        std::cout << "it " << i << "===============\n";
        graph.update_factors();
        graph.update_variables();
        // pull out the mean
        std::cout << "\tvar1 " << graph.get_mean(variable1).transpose() << "\n";
        std::cout << "\tvar2 " << graph.get_mean(variable2).transpose() << "\n";
        std::cout << "\tvar3 " << graph.get_mean(variable3).transpose() << "\n";
        
    }

    return 0;
}
