#include "variable_array.h"
#include "matrix_mem_eigen.h"

namespace carl {

constexpr static int VAR_ID = 0;
constexpr static int INFO_MAT = 1;
constexpr static int INFO_VEC = 2;

template <uint8_t dim>
Id VariableArray<dim>::create_variable() {
    Id id = NEW_VAR_ID();
    vars.insert(id, MatrixMem<dim>{}, VectorMem<dim>{});
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
MatrixRef VariableArray<dim>::get_mat(size_t idx) {
    return get_refs(vars.template get_column<INFO_MAT>())[idx];
}

template <uint8_t dim>
VectorRef VariableArray<dim>::get_vec(size_t idx) {
    return get_refs(vars.template get_column<INFO_VEC>())[idx];
}

template class VariableArray<4>;
template class VariableArray<6>;

}  // carl
