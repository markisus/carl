#pragma once

#include "vapid/soa.h"
#include "Eigen/Dense"
#include "id.h"
#include "matrix_mem.h"

namespace carl {

struct ToVariableMsg_RetrievalContext {
    size_t num_msgs;
    uint8_t variable_dim;
    FactorId* factor_ids; // for dimensionality of factor
    size_t* factor_idxs;
    uint8_t* factor_slots;
    MatrixRefs matrices;
    VectorRefs vectors;
};

struct ToVariableMsg_SendContext {
    size_t num_msgs;
    uint8_t variable_dim;
    size_t* variable_idxs;
    MatrixRefs matrices;
    VectorRefs vectors;
};

class FactorArrays;

class ToVariableMsgBufferBase {
public:
    virtual uint8_t get_dimension() const = 0;

    virtual void commit_edges(const std::vector<VariableId>& external_variable_ids,
                              const FactorArrays& factors) = 0;
    virtual void add_edge(VariableId variable_id, FactorId factor_id, uint8_t factor_slot) = 0;

    virtual void rebuild_retrieval_buffer() = 0;
    virtual ToVariableMsg_RetrievalContext generate_retrieval_context() const = 0;
    virtual void commit_retrievals() = 0;

    virtual ToVariableMsg_SendContext generate_send_context() const = 0;
};

template <uint8_t dim>
struct ToVariableMsgBuffer : public ToVariableMsgBufferBase {
    vapid::soa<VariableId, FactorId, uint8_t, size_t, size_t, size_t, MatrixMem<dim>, VectorMem<dim>, bool> msgs;
    vapid::soa<VariableId, FactorId, uint8_t, size_t, size_t, size_t> retrieval_buffer_meta;
    vapid::soa<MatrixMem<dim>, VectorMem<dim>> retrieval_buffer_data;

    bool edges_committed = false;

    uint8_t get_dimension() const override;

    void commit_edges(const std::vector<VariableId>& external_variable_ids,
                      const FactorArrays& factors) override;
    void add_edge(VariableId variable_id, FactorId factor_id, uint8_t factor_slot) override;

    ToVariableMsg_RetrievalContext generate_retrieval_context() const override;
    void rebuild_retrieval_buffer() override;
    void commit_retrievals() override;

    ToVariableMsg_SendContext generate_send_context() const;
};

}  // carl
