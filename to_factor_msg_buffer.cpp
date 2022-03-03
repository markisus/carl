#include "to_factor_msg_buffer.h"
#include "factor_arrays.h"
#include "util.h"

namespace carl {

// SOA Indices
constexpr static int VAR_ID = 0;
constexpr static int FACTOR_ID = 1;
constexpr static int FACTOR_SLOT = 2;

constexpr static int INTERNAL_IDX = 3; // used to match retrieval buffer back into the main buffer
constexpr static int EXTERNAL_VAR_IDX = 4; // used to match the message into the external array
constexpr static int EXTERNAL_FACTOR_IDX = 5; // used to match the message into the external array
    
constexpr static int INFO_MAT = 6;
constexpr static int INFO_VEC = 7;
constexpr static int ENABLE_FLAG = 8; // controls which edges enter into the retrieval buffer

template <uint8_t dim>
uint8_t ToFactorMsgBuffer<dim>::get_dimension() const {
    return dim;
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::commit_edges(
    const std::vector<VariableId>& external_variable_ids,
    const FactorArrays& factor_arrays) {

    // build the external variable index
    msgs.template sort_by_view<VAR_ID, FACTOR_ID>();
    rebuild_index(
        external_variable_ids,
        msgs.template get_column<VAR_ID>(),
        msgs.template get_column<EXTERNAL_VAR_IDX>());

    // build the external factor index
    msgs.template sort_by_view<FACTOR_ID, VAR_ID>();
    size_t idx = 0;
    for (uint8_t factor_dim : FactorArrays::factor_dims) {
        idx = rebuild_index(
            factor_arrays.of_dimension(factor_dim).ids(),
            msgs.template get_column<FACTOR_ID>(),
            msgs.template get_column<EXTERNAL_FACTOR_IDX>(),
            idx);
    }

    write_identity_permutation(msgs.template get_column<INTERNAL_IDX>());    
    edges_committed = true;
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::add_edge(VariableId variable_id, FactorId factor_id, uint8_t factor_slot)  {
    msgs.insert(variable_id, factor_id, factor_slot,
                0, 0, 0,
                MatrixMem<dim>{},
                VectorMem<dim>{}, true);
    edges_committed = false;
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::rebuild_retrieval_buffer()  {
    retrieval_buffer.clear();
    for (size_t i = 0; i < msgs.size(); ++i) {
        if (!msgs.template get_column<ENABLE_FLAG>()[i]) {
            continue;
        }
        auto [var_id, factor_id, factor_slot, internal_idx, external_var_idx, external_factor_idx, mat, vec] =
            msgs.template view<VAR_ID, FACTOR_ID, FACTOR_SLOT, INTERNAL_IDX, EXTERNAL_VAR_IDX, EXTERNAL_FACTOR_IDX, INFO_MAT, INFO_VEC>(i);
        retrieval_buffer.insert(var_id, factor_id, factor_slot,
                                internal_idx, external_var_idx, external_factor_idx,
                                mat, vec);
    }

    // retrieve from factors
    retrieval_buffer.template sort_by_view<VAR_ID, FACTOR_ID>();
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::commit_retrievals()  {
    for (size_t i = 0; i < retrieval_buffer.size(); ++i) {
        auto [src_mat, src_vec, buffer_idx] = retrieval_buffer.template view<INFO_MAT, INFO_VEC, INTERNAL_IDX>(i);
        auto& dst_mat = msgs.template get_column<INFO_MAT>()[buffer_idx];
        dst_mat = src_mat;
    }

    for (size_t i = 0; i < retrieval_buffer.size(); ++i) {
        auto [src_mat, src_vec, buffer_idx] = retrieval_buffer.template view<INFO_MAT, INFO_VEC, INTERNAL_IDX>(i);
        auto& dst_vec = msgs.template get_column<INFO_VEC>()[buffer_idx];
        dst_vec = src_vec;
    }

    retrieval_buffer.clear();
}

template <uint8_t dim>
ToFactorMsg_RetrievalContext ToFactorMsgBuffer<dim>::generate_retrieval_context() const  {
    ToFactorMsg_RetrievalContext ctx;
    ctx.num_msgs = retrieval_buffer.size();
    ctx.variable_dim = dim;
    ctx.variable_idxs = non_const(retrieval_buffer.template get_column<EXTERNAL_VAR_IDX>().data());
    ctx.matrices = get_refs(retrieval_buffer.template get_column<INFO_MAT>());
    ctx.vectors = get_refs(retrieval_buffer.template get_column<INFO_VEC>());
    return ctx;
}

template <uint8_t dim>
ToFactorMsg_SendContext ToFactorMsgBuffer<dim>::generate_send_context() const {
    ToFactorMsg_SendContext ctx;
    ctx.num_msgs = msgs.size();
    ctx.variable_dim = dim;
    ctx.factor_ids = non_const(msgs.template get_column<FACTOR_ID>().data());
    ctx.factor_idxs = non_const(msgs.template get_column<EXTERNAL_FACTOR_IDX>().data());
    ctx.factor_slots = non_const(msgs.template get_column<FACTOR_SLOT>().data());
    ctx.matrices = get_refs(msgs.template get_column<INFO_MAT>());
    ctx.vectors = get_refs(msgs.template get_column<INFO_VEC>());
    return ctx;
}

template class ToFactorMsgBuffer<4>;
template class ToFactorMsgBuffer<6>;

}  // carl
