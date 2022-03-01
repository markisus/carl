#pragma once

#include "matrix_mem.h"
#include "Eigen/Dense"

namespace carl {

inline Eigen::Map<Eigen::MatrixXd> to_eigen_map(const MatrixRef& ref) {
    return Eigen::Map<Eigen::MatrixXd> {ref.begin, ref.dimension, ref.dimension};
}

inline Eigen::Map<Eigen::VectorXd> to_eigen_map(const VectorRef& ref) {
    return Eigen::Map<Eigen::VectorXd> {ref.begin, ref.dimension};
}

template <uint8_t dim>
Eigen::Map<Eigen::Matrix<double, dim, dim>> to_eigen_map(const StaticMatrixRef<dim>& ref) {
    return Eigen::Map<Eigen::Matrix<double, dim, dim>> { ref.begin };
}

template <uint8_t dim>
Eigen::Map<Eigen::Matrix<double, dim, 1>> to_eigen_map(const StaticVectorRef<dim>& ref) {
    return Eigen::Map<Eigen::Matrix<double, dim, 1>> { ref.begin };
}

template <uint8_t dim>
Eigen::Map<Eigen::Matrix<double, dim, dim>> to_eigen_map(const MatrixMem<dim>& mem) {
    return Eigen::Map<Eigen::Matrix<double, dim, dim>> { non_const(mem.data()) };
}

template <uint8_t dim>
Eigen::Map<Eigen::Matrix<double, dim, 1>> to_eigen_map(const VectorMem<dim>& mem) {
    return Eigen::Map<Eigen::Matrix<double, dim, 1>> { non_const(mem.data()) };
}

template <uint8_t dim>
struct StaticEigenMatrixRefs {
    StaticEigenMatrixRefs(const StaticMatrixRefs<dim>& r) : refs_(r) {}
    Eigen::Map<Eigen::Matrix<double, dim, dim>> operator[](size_t idx) {
        return to_eigen_map(refs_[idx]);
    }
    StaticMatrixRefs<dim> refs_;
};

template <uint8_t dim>
inline StaticEigenMatrixRefs<dim> to_eigen_map(const StaticMatrixRefs<dim>& r) {
    return {r};
}

template <uint8_t dim>
struct StaticEigenVectorRefs {
    StaticEigenVectorRefs(const StaticVectorRefs<dim>& r) : refs_(r) {}
    Eigen::Map<Eigen::Matrix<double, dim, 1>> operator[](size_t idx) {
        return to_eigen_map(refs_[idx]);
    }
    StaticVectorRefs<dim> refs_;
};

template <uint8_t dim>
inline StaticEigenVectorRefs<dim> to_eigen_map(const StaticVectorRefs<dim>& r) {
    return {r};
}

struct EigenMatrixRefs {
    EigenMatrixRefs(const MatrixRefs& r) : refs_(r) {}
    Eigen::Map<Eigen::MatrixXd> operator[](size_t idx) {
        return to_eigen_map(refs_[idx]);
    }
    MatrixRefs refs_;
};

inline EigenMatrixRefs to_eigen_map(const MatrixRefs& r) {
    return {r};
}

struct EigenVectorRefs {
    EigenVectorRefs(const VectorRefs& r) : refs_(r) {}
    Eigen::Map<Eigen::VectorXd> operator[](size_t idx) {
        return to_eigen_map(refs_[idx]);
    }
    VectorRefs refs_;
};

inline EigenVectorRefs to_eigen_map(const VectorRefs& r) {
    return {r};
}

template <int dim, typename = std::enable_if_t<(dim > 0)>>
inline StaticMatrixRef<uint8_t(dim)> ref(const Eigen::Matrix<double, dim, dim>& m) {
    return { non_const(m.data()) };
}

template <int dim, typename = std::enable_if_t<(dim > 0)>>
inline StaticVectorRef<uint8_t(dim)> ref(const Eigen::Matrix<double, dim, 1>& m) {
    return { non_const(m.data()) };
}

}  // carl
