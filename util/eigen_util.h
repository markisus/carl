#pragma once
#include "Eigen/Dense"

namespace Eigen {

template <int dim>
using VectorD = Matrix<double, dim, 1>;

template <int rows, int cols=rows>
using MatrixD = Matrix<double, rows, cols>;

template <int dim>
auto id() {
    return MatrixD<dim>::Identity();
}

template <int dim>
MatrixD<dim> zero_mat() {
    return MatrixD<dim>::Zero();
}

template <int rows, int cols>
MatrixD<rows, cols> random_mat() {
    return MatrixD<rows, cols>::Random();
}

template <int dim>
VectorD<dim> zero_vec() {
    return VectorD<dim>::Zero();
}

template <int dim>
VectorD<dim> random_vec() {
    return VectorD<dim>::Random();
}

inline VectorD<4> random_homog() {
    auto result = random_vec<4>();
    result(3) = 1.0;
    return result;
}

template <int dim>
VectorD<dim> basis(int i) {
    auto result = zero_vec<dim>();
    result(i) = 1;
    return result;
}

}  // Eigen
