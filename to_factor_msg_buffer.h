#pragma once

#include "vapid/soa.h"
#include "matrix_mem.h"
#include "id.h"

namespace carl {

struct ToFactorMsg_RetrievalContext {
    size_t num_msgs;
    uint8_t variable_dim;
    size_t* variable_idxs;
    MatrixRefs matrices;
    VectorRefs vectors;
};

struct ToFactorMsg_SendContext {
    size_t num_msgs;
    uint8_t variable_dim;
    FactorId* factor_ids; // for dimensionality of factor
    size_t* factor_idxs;
    uint8_t* factor_slots;
    MatrixRefs matrices;
    VectorRefs vectors;
};

class ToFactorMsgBufferBase {
public:
    virtual uint8_t get_dimension() const = 0;
    virtual void commit_edges(const std::vector<FactorId>& external_factor_ids) = 0;
    virtual void add_edge(Id variable_id, FactorId factor_id, uint8_t factor_slot) = 0;

    virtual ToFactorMsg_RetrievalContext generate_retrieval_context() const = 0;
    virtual void rebuild_retrieval_buffer() = 0;
    virtual ToFactorMsg_SendContext generate_send_context() const = 0;
    virtual void commit_retrievals() = 0;
};

template <uint8_t dim>
struct ToFactorMsgBuffer : ToFactorMsgBufferBase {

    vapid::soa<Id, FactorId, uint8_t, size_t, size_t, MatrixMem<dim>, VectorMem<dim>,  bool> msgs;
    vapid::soa<Id, FactorId, uint8_t, size_t, size_t, MatrixMem<dim>, VectorMem<dim>> retrieval_buffer;

    bool edges_committed = false;

    uint8_t get_dimension() const override;

    void commit_edges(const std::vector<FactorId>& external_factor_ids) override;
    void add_edge(Id variable_id, FactorId factor_id, uint8_t factor_slot) override;

    ToFactorMsg_RetrievalContext generate_retrieval_context() const override;
    void rebuild_retrieval_buffer() override;
    ToFactorMsg_SendContext generate_send_context() const override;
    void commit_retrievals() override;
};


}  // carl
