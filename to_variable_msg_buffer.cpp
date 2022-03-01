#include "to_variable_msg_buffer.h"
#include "util.h"
#include "matrix_mem_eigen.h"
#include "factor_arrays.h"

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
uint8_t ToVariableMsgBuffer<dim>::get_dimension() const {
    return dim;
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::commit_edges(
    const std::vector<VariableId>& external_variable_ids,
    const FactorArrays& factor_arrays)  {

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

    // build the external variable index
    msgs.template sort_by_view<VAR_ID, FACTOR_ID>();
    rebuild_index(
        external_variable_ids,
        msgs.template get_column<VAR_ID>(),
        msgs.template get_column<EXTERNAL_VAR_IDX>());

    write_identity_permutation(msgs.template get_column<INTERNAL_IDX>());

    /* Special handling for unary factors:
     * [ factor ]=>( variable )
     *
     * Unary factors are special.
     * 1. A unary factor's message to its variable can never change.
     *    We can therefore memorize the message and then disable the edge.
     * 2. All priors are unary factors and priors need to be propagated
     *    before any other messages are sent, or else the graph represents
     *    a degenerate gaussian.
     */

    auto& factor_ids = msgs.template get_column<FACTOR_ID>();
    auto& external_factor_idxs = msgs.template get_column<EXTERNAL_FACTOR_IDX>();
    auto& enable_flags = msgs.template get_column<ENABLE_FLAG>();
    auto& info_mats = msgs.template get_column<INFO_MAT>();
    auto& info_vecs = msgs.template get_column<INFO_VEC>();
    auto& factors = factor_arrays.of_dimension(dim);

    for (size_t i = 0; i < msgs.size(); ++i) {
        auto [factor_dim, _ignore] = factor_ids[i];
        if (factor_dim == dim) {
            // A factor is unary iff its dimension is equal to one of its variable's.
            // So this message is coming from a unary factor.
            enable_flags[i] = false;
            info_mats[i] = *static_dim<dim>(factors.get_info_mat(external_factor_idxs[i]));
            info_vecs[i] = *static_dim<dim>(factors.get_info_vec(external_factor_idxs[i]));
        }
    }
    
    edges_committed = true;
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::add_edge(VariableId variable_id, FactorId factor_id, uint8_t factor_slot) {
    msgs.insert(variable_id, factor_id, factor_slot,
                0, 0, 0,
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
    ctx.factor_idxs = non_const(retrieval_buffer_meta.template get_column<EXTERNAL_FACTOR_IDX>().data());
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
        auto [var_id,
              factor_id,
              factor_slot,
              internal_idx,
              external_var_idx,
              external_factor_idx] =
            msgs.template view<VAR_ID,
                               FACTOR_ID,
                               FACTOR_SLOT,
                               INTERNAL_IDX,
                               EXTERNAL_VAR_IDX,
                               EXTERNAL_FACTOR_IDX>(i);
        retrieval_buffer_meta.insert(var_id,
                                     factor_id,
                                     factor_slot,
                                     internal_idx,
                                     external_var_idx,
                                     external_factor_idx);
        retrieval_buffer_data.insert(MatrixMem<dim>{}, VectorMem<dim>{});
    }

    // receive from factors
    retrieval_buffer_meta.template sort_by_view<FACTOR_ID, VAR_ID>();
}

template <uint8_t dim>
void ToVariableMsgBuffer<dim>::commit_retrievals() {
    // std::cout << "committing retrievals for vars of dim " << int(dim) << "\n";

    for (size_t i = 0; i < retrieval_buffer_meta.size(); ++i) {
        auto [src_mat, src_vec] = retrieval_buffer_data[i];
        auto [buffer_idx] = retrieval_buffer_meta.template view<INTERNAL_IDX>(i);
        auto& dst_mat = msgs.template get_column<INFO_MAT>()[buffer_idx];
        dst_mat = src_mat;

        // std::cout << "\tmoving from retrieval buffer idx " << i << " to edge idx " << buffer_idx << "\n";
        // std::cout << "\tmsg was\n" << to_eigen_map(dst_mat) << "\n";        
    }

    for (size_t i = 0; i < retrieval_buffer_meta.size(); ++i) {
        auto [src_mat, src_vec] = retrieval_buffer_data[i];
        auto [buffer_idx] = retrieval_buffer_meta.template view<INTERNAL_IDX>(i);
        auto& dst_vec = msgs.template get_column<INFO_VEC>()[buffer_idx];
        dst_vec = src_vec;

        // std::cout << "\tmoving from retrieval buffer idx " << i << " to edge idx " << buffer_idx << "\n";
        // std::cout << "\tmsg was " << to_eigen_map(dst_vec).transpose() << "\n";
    }

    retrieval_buffer_meta.clear();
    retrieval_buffer_data.clear();
}

template <uint8_t dim>
ToVariableMsg_SendContext ToVariableMsgBuffer<dim>::generate_send_context() const {
    // std::cout << "making to vars " << int(dim) << " send context" << "\n";

    // for (size_t i = 0; i < msgs.size(); ++i) {
    //     std::cout << "\tmsg[" << i << "] = " << to_eigen_map(msgs.template get_column<INFO_VEC>()[i]).transpose() << "\n";
    // }

    ToVariableMsg_SendContext ctx;
    ctx.num_msgs = msgs.size();
    ctx.variable_dim = dim;
    ctx.matrices = get_refs(msgs.template get_column<INFO_MAT>());
    ctx.vectors = get_refs(msgs.template get_column<INFO_VEC>());
    ctx.variable_idxs = non_const(msgs.template get_column<EXTERNAL_VAR_IDX>().data());
    return ctx;
}

template class ToVariableMsgBuffer<4>;
template class ToVariableMsgBuffer<6>;

}  // carl
