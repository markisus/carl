#include "variable_array.h"
#include "matrix_mem_eigen.h"

namespace carl {

constexpr static int VAR_ID = 0;
constexpr static int INFO_MAT = 1;
constexpr static int INFO_VEC = 2;
constexpr static int COV_MAT = 3;
constexpr static int COV_VEC = 4;

template <uint8_t dim>
VariableId VariableArray<dim>::create_variable() {
    VariableId id = NEW_VARIABLE_ID(dim);
    vars.insert(id, MatrixMem<dim>{}, VectorMem<dim>{}, MatrixMem<dim>{}, VectorMem<dim>{});
    return id;
}

template <uint8_t dim>
void VariableArray<dim>::zero_all() {
    for (size_t i = 0; i < vars.size(); ++i) {
        to_eigen_map(get_mat(i)).setZero();
    }
    for (size_t i = 0; i < vars.size(); ++i) {
        to_eigen_map(get_vec(i)).setZero();
    }        
}

template <uint8_t dim>
void VariableArray<dim>::rebuild_cov_mats() {
    // std::cout << "Rebulding cov mats dim " << int(dim) << "\n";

    auto info_mats = to_eigen_map(get_refs(vars.template get_column<INFO_MAT>()));
    auto info_vecs = to_eigen_map(get_refs(vars.template get_column<INFO_VEC>()));
    auto cov_mats = to_eigen_map(get_refs(vars.template get_column<COV_MAT>()));
    auto cov_vecs = to_eigen_map(get_refs(vars.template get_column<COV_VEC>()));

    for (size_t i = 0; i < vars.size(); ++i) {
        cov_mats[i] = info_mats[i].inverse();
    }

    for (size_t i = 0; i < vars.size(); ++i) {
        cov_vecs[i] = info_mats[i].ldlt().solve(info_vecs[i]);
    }
}

template <uint8_t dim>
MatrixRef VariableArray<dim>::get_mat(size_t idx) {
    return get_refs(vars.template get_column<INFO_MAT>())[idx];
}

template <uint8_t dim>
VectorRef VariableArray<dim>::get_vec(size_t idx) {
    return get_refs(vars.template get_column<INFO_VEC>())[idx];
}

template <uint8_t dim>
const std::vector<VariableId>& VariableArray<dim>::ids() const {
    return vars.template get_column<VAR_ID>();
}

template <uint8_t dim>
MatrixRef VariableArray<dim>::get_cov(size_t idx) const {
    return get_refs(vars.template get_column<COV_MAT>())[idx];
}

template <uint8_t dim>
VectorRef VariableArray<dim>::get_mean(size_t idx) const {
    return get_refs(vars.template get_column<COV_VEC>())[idx];
}

template class VariableArray<4>;
template class VariableArray<6>;

}  // carl
