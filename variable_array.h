#pragma once

#include "Eigen/Dense"
#include "matrix_mem.h"
#include "vapid/soa.h"
#include "id.h"

namespace carl {

class VariableArrayBase {
public:
    virtual Id create_variable() = 0;
    virtual MatrixRef get_mat(size_t idx) = 0;
    virtual VectorRef get_vec(size_t idx) = 0;
    virtual void zero_all() = 0;
};

template <uint8_t dim>
struct VariableArray : public VariableArrayBase {
    // keep sorted by VAR_ID
    vapid::soa<Id, MatrixMem<dim>, VectorMem<dim>> vars;

    Id create_variable() override;
    void zero_all() override;
    MatrixRef get_mat(size_t idx) override;
    VectorRef get_vec(size_t idx) override;
};

}  // carl
