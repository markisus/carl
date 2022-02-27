#include "to_variable_msg_buffer.h"
#include "util.h"

namespace carl {

// SOA Indices
constexpr static int VAR_ID = 0;
constexpr static int FACTOR_ID = 1;
constexpr static int FACTOR_SLOT = 2;

constexpr static int INTERNAL_IDX = 3; // used to match retrieval buffer back into the main buffer
constexpr static int EXTERNAL_IDX = 4; // used to match the var into the external variable or factor array
    
constexpr static int INFO_MAT = 5;
constexpr static int INFO_VEC = 6;
constexpr static int ENABLE_FLAG = 7; // controls which edges enter into the retrieval buffer

template <uint8_t dim>
uint8_t ToVariableMsgBuffer<dim>::get_dimension() const {
    return dim;
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::commit_edges(const std::vector<Id>& external_variable_ids)  {
    msgs.template sort_by_view<VAR_ID, FACTOR_ID>();
    write_identity_permutation(msgs.template get_column<INTERNAL_IDX>());
    rebuild_index(
        external_variable_ids,
        msgs.template get_column<VAR_ID>(),
        msgs.template get_column<EXTERNAL_IDX>());
    edges_committed = true;
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::add_edge(Id variable_id, FactorId factor_id, uint8_t factor_slot) {
    msgs.insert(variable_id, factor_id, factor_slot,
                0, 0,
                MatrixMem<dim>{},
                VectorMem<dim>{}, true);
    edges_committed = false;
}

template <uint8_t dim>
ToVariableMsg_RetrievalContext ToVariableMsgBuffer<dim>::generate_retrieval_context() const {
    ToVariableMsg_RetrievalContext ctx;
    ctx.num_msgs = retrieval_buffer_meta.size();
    ctx.variable_dim = dim;
    ctx.factor_ids = non_const(retrieval_buffer_meta.template get_column<FACTOR_ID>().data());
    ctx.factor_idxs = non_const(retrieval_buffer_meta.template get_column<EXTERNAL_IDX>().data());
    ctx.factor_slots = non_const(retrieval_buffer_meta.template get_column<FACTOR_SLOT>().data());
    ctx.matrices = get_refs(retrieval_buffer_data.template get_column<0>());
    ctx.vectors = get_refs(retrieval_buffer_data.template get_column<1>());
    return ctx;
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::rebuild_retrieval_buffer()  {
    retrieval_buffer_meta.clear();
    retrieval_buffer_data.clear();

    for (size_t i = 0; i < msgs.size(); ++i) {
        if (!msgs.template get_column<ENABLE_FLAG>()[i]) {
            continue;
        }
        auto [var_id, factor_id, factor_slot, internal_idx, external_idx] =
            msgs.template view<VAR_ID, FACTOR_ID, FACTOR_SLOT, INTERNAL_IDX, EXTERNAL_IDX>(i);
        retrieval_buffer_meta.insert(var_id, factor_id, factor_slot, internal_idx, external_idx);
        retrieval_buffer_data.insert(MatrixMem<dim>{}, VectorMem<dim>{});
    }

    // receive from factors
    retrieval_buffer_meta.template sort_by_view<FACTOR_ID, VAR_ID>();
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::commit_retrievals() {
    for (size_t i = 0; i < retrieval_buffer_meta.size(); ++i) {
        auto [src_mat, src_vec] = retrieval_buffer_data[i];
        auto [buffer_idx] = retrieval_buffer_meta.template view<INTERNAL_IDX>(i);
        auto dst_mat = msgs.template get_column<INFO_MAT>()[buffer_idx];
        dst_mat = src_mat;
    }

    for (size_t i = 0; i < retrieval_buffer_meta.size(); ++i) {
        auto [src_mat, src_vec] = retrieval_buffer_data[i];
        auto [buffer_idx] = retrieval_buffer_meta.template view<INTERNAL_IDX>(i);
        auto dst_vec = msgs.template get_column<INFO_VEC>()[buffer_idx];
        dst_vec = src_vec;
    }

    retrieval_buffer_meta.clear();
    retrieval_buffer_data.clear();
}

template <uint8_t dim>
ToVariableMsg_SendContext ToVariableMsgBuffer<dim>::generate_send_context() const {
    ToVariableMsg_SendContext ctx;
    ctx.num_msgs = msgs.size();
    ctx.variable_dim = dim;
    ctx.matrices = get_refs(msgs.template get_column<INFO_MAT>());
    ctx.vectors = get_refs(msgs.template get_column<INFO_VEC>());
    ctx.variable_idxs = non_const(msgs.template get_column<EXTERNAL_IDX>().data());
    return ctx;
}

template class ToVariableMsgBuffer<4>;
template class ToVariableMsgBuffer<6>;

}  // carl
