#pragma once

#include "matrix_mem.h"
#include "vapid/soa.h"
#include "id.h"

namespace carl {

class VariableArrayBase {
public:
    virtual VariableId create_variable() = 0;
    virtual const std::vector<VariableId>& ids() const = 0;
    virtual MatrixRef get_mat(size_t idx) = 0;
    virtual VectorRef get_vec(size_t idx) = 0;
    virtual VectorRef get_mean(size_t idx) const = 0;
    virtual MatrixRef get_cov(size_t idx) const = 0;
    virtual void zero_all() = 0;
    virtual void rebuild_cov_mats() = 0;
};

template <uint8_t dim>
struct VariableArray : public VariableArrayBase {
    // keep sorted by VAR_ID
    vapid::soa<VariableId,
               MatrixMem<dim>, VectorMem<dim>,
               MatrixMem<dim>, VectorMem<dim>> vars;

    const std::vector<VariableId>& ids() const override;
    VariableId create_variable() override;
    void zero_all() override;
    void rebuild_cov_mats() override;
    MatrixRef get_mat(size_t idx) override;
    VectorRef get_vec(size_t idx) override;
    MatrixRef get_cov(size_t idx) const override;
    VectorRef get_mean(size_t idx) const override;

};

}  // carl
