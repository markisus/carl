#include "factor_array.h"
#include "matrix_mem_eigen.h"

namespace carl {

constexpr static int FACTOR_ID = 0;
constexpr static int INFO_MAT = 1;
constexpr static int INFO_VEC = 2;
constexpr static int INFO_MAT_DELTA = 3;
constexpr static int INFO_VEC_DELTA = 4;
constexpr static int COV_MAT = 5;
constexpr static int COV_VEC = 6;

template <uint8_t dim>
FactorId FactorArray<dim>::create_factor(const MatrixRef& info_mat, const VectorRef& info_vec) {
    return create_factor(static_dim<dim>(info_mat), static_dim<dim>(info_vec));
}

template <uint8_t dim>
FactorId FactorArray<dim>::create_factor(const StaticMatrixRef<dim>& info_mat, const StaticVectorRef<dim>& info_vec) {
    FactorId id = NEW_FACTOR_ID(dim);
    factors.insert(id,
                   *info_mat, *info_vec,
                   MatrixMem<dim>{}, VectorMem<dim>{},
                   MatrixMem<dim>{}, VectorMem<dim>{});
    return id;
}

template <uint8_t dim>
FactorId FactorArray<dim>::create_factor(const double* info_mat, const double* info_vec) {
    return create_factor(StaticMatrixRef<dim>{non_const(info_mat)},
                         StaticVectorRef<dim>{non_const(info_vec)});
}

template <uint8_t dim>
MatrixRef FactorArray<dim>::get_cov_mat(size_t idx) {
    return get_refs(factors.template get_column<COV_MAT>())[idx];
}

template <uint8_t dim>
VectorRef FactorArray<dim>::get_cov_vec(size_t idx) {
    return get_refs(factors.template get_column<COV_VEC>())[idx];
}

template <uint8_t dim>
MatrixRef FactorArray<dim>::get_info_mat(size_t idx) const {
    return get_refs(factors.template get_column<INFO_MAT>())[idx];
}

template <uint8_t dim>
VectorRef FactorArray<dim>::get_info_vec(size_t idx) const {
    return get_refs(factors.template get_column<INFO_VEC>())[idx];
}

template <uint8_t dim>
MatrixRef FactorArray<dim>::get_info_mat_delta(size_t idx) {
    return get_refs(factors.template get_column<INFO_MAT_DELTA>())[idx];
}

template <uint8_t dim>
VectorRef FactorArray<dim>::get_info_vec_delta(size_t idx) {
    return get_refs(factors.template get_column<INFO_VEC_DELTA>())[idx];
}

template <uint8_t dim>
void FactorArray<dim>::rebuild_cov_mats() {
    auto cov_mats = to_eigen_map(get_refs(factors.template get_column<COV_MAT>()));
    auto cov_vecs = to_eigen_map(get_refs(factors.template get_column<COV_VEC>()));
    auto info_mats = to_eigen_map(get_refs(factors.template get_column<INFO_MAT>()));
    auto info_vecs = to_eigen_map(get_refs(factors.template get_column<INFO_VEC>()));
    auto info_mats_delta = to_eigen_map(get_refs(factors.template get_column<INFO_MAT_DELTA>()));
    auto info_vecs_delta = to_eigen_map(get_refs(factors.template get_column<INFO_VEC_DELTA>()));

    for (size_t i = 0; i < factors.size(); ++i) {
        cov_mats[i] = info_mats[i];            
    }
    for (size_t i = 0; i < factors.size(); ++i) {
        cov_vecs[i] = info_vecs[i];            
    }
    for (size_t i = 0; i < factors.size(); ++i) {
        cov_mats[i] += info_mats_delta[i];
    }
    for (size_t i = 0; i < factors.size(); ++i) {
        cov_vecs[i] += info_vecs_delta[i];
    }
    for (size_t i = 0; i < factors.size(); ++i) {
        Eigen::Matrix<double, dim, dim> result_mat = cov_mats[i].inverse();
        Eigen::Matrix<double, dim, 1> result_vec = cov_mats[i].ldlt().solve(cov_vecs[i]);
        cov_mats[i] = result_mat;
        cov_vecs[i] = result_vec;
    }
}

template <uint8_t dim>
const std::vector<FactorId>& FactorArray<dim>::ids() const {
    return factors.template get_column<FACTOR_ID>();
};

template class FactorArray<4>;
template class FactorArray<6>;
template class FactorArray<12>;
template class FactorArray<16>;


}  // carl
