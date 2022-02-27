#include "to_factor_msg_buffer.h"
#include "util.h"

namespace carl {

// SOA Indices
constexpr static int VAR_ID = 0;
constexpr static int FACTOR_ID = 1;
constexpr static int FACTOR_SLOT = 2;

constexpr static int INTERNAL_IDX = 3; // used to match retrieval buffer back into the main buffer
constexpr static int EXTERNAL_IDX = 4; // used to match the var or factor into the external factor array

constexpr static int INFO_MAT = 5;
constexpr static int INFO_VEC = 6;

constexpr static int ENABLE_FLAG = 7; // controls which edges enter into the retrieval buffer

template <uint8_t dim>
uint8_t ToFactorMsgBuffer<dim>::get_dimension() const {
    return dim;
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::commit_edges(const std::vector<FactorId>& external_factor_ids) {
    msgs.template sort_by_view<FACTOR_ID, VAR_ID>();
    write_identity_permutation(msgs.template get_column<INTERNAL_IDX>());
    rebuild_index(
        external_factor_ids,
        msgs.template get_column<FACTOR_ID>(),
        msgs.template get_column<EXTERNAL_IDX>());        
    edges_committed = true;
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::add_edge(Id variable_id, FactorId factor_id, uint8_t factor_slot)  {
    msgs.insert(variable_id, factor_id, factor_slot,
                0, 0,
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
        auto [var_id, factor_id, factor_slot, internal_idx, external_idx, mat, vec] =
            msgs.template view<VAR_ID, FACTOR_ID, FACTOR_SLOT, INTERNAL_IDX, EXTERNAL_IDX, INFO_MAT, INFO_VEC>(i);
        retrieval_buffer.insert(var_id, factor_id, factor_slot,
                                internal_idx, external_idx,
                                mat, vec);
    }

    // retrieve from factors
    retrieval_buffer.template sort_by_view<VAR_ID, FACTOR_ID>();
}

template <uint8_t dim>
void ToFactorMsgBuffer<dim>::commit_retrievals()  {
    for (size_t i = 0; i < retrieval_buffer.size(); ++i) {
        auto [src_mat, src_vec, buffer_idx] = retrieval_buffer.template view<INFO_MAT, INFO_VEC, INTERNAL_IDX>(i);
        auto dst_mat = msgs.template get_column<INFO_MAT>()[buffer_idx];
        dst_mat = src_mat;
    }

    for (size_t i = 0; i < retrieval_buffer.size(); ++i) {
        auto [src_mat, src_vec, buffer_idx] = retrieval_buffer.template view<INFO_MAT, INFO_VEC, INTERNAL_IDX>(i);
        auto dst_vec = msgs.template get_column<INFO_VEC>()[buffer_idx];
        dst_vec = src_vec;
    }

    retrieval_buffer.clear();
}

template <uint8_t dim>
ToFactorMsg_RetrievalContext ToFactorMsgBuffer<dim>::generate_retrieval_context() const  {
    ToFactorMsg_RetrievalContext ctx;
    ctx.num_msgs = retrieval_buffer.size();
    ctx.variable_dim = dim;
    ctx.variable_idxs = non_const(retrieval_buffer.template get_column<EXTERNAL_IDX>().data());
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
    ctx.factor_idxs = non_const(msgs.template get_column<EXTERNAL_IDX>().data());
    ctx.factor_slots = non_const(msgs.template get_column<FACTOR_SLOT>().data());
    ctx.matrices = get_refs(msgs.template get_column<INFO_MAT>());
    ctx.vectors = get_refs(msgs.template get_column<INFO_VEC>());
    return ctx;
}

template class ToFactorMsgBuffer<4>;
template class ToFactorMsgBuffer<6>;

}  // carl
