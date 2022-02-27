#pragma once

#include "vapid/soa.h"
#include "matrix_mem.h"
#include "id.h"

namespace carl {

class FactorArrayBase {
public:
    virtual MatrixRef get_cov_mat(size_t idx) = 0;
    virtual VectorRef get_cov_vec(size_t idx) = 0;
    virtual MatrixRef get_info_mat_delta(size_t idx) = 0;
    virtual VectorRef get_info_vec_delta(size_t idx) = 0;
    virtual void rebuild_cov_mats() = 0;
};

template <uint8_t dim>
struct FactorArray : public FactorArrayBase {
    // keep sorted by FACTOR_ID
    vapid::soa<FactorId,
               MatrixMem<dim>, VectorMem<dim>,
               MatrixMem<dim>, VectorMem<dim>,
               MatrixMem<dim>, VectorMem<dim>> factors;

    FactorId create_factor(const StaticMatrixRef<dim>& info_mat, const StaticVectorRef<dim>& info_vec);
    FactorId create_factor(const double* info_mat, const double* info_vec);
    MatrixRef get_cov_mat(size_t idx) override;
    VectorRef get_cov_vec(size_t idx) override;
    MatrixRef get_info_mat_delta(size_t idx) override;
    VectorRef get_info_vec_delta(size_t idx) override;
    void rebuild_cov_mats() override;
};

}  // carl
