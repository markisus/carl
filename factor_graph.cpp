#include "factor_graph.h"
#include "matrix_mem_eigen.h"

namespace carl {

MsgBuffersBase& FactorGraph::msg_buffers_of_dimension(uint8_t dim) {
    switch (dim) {
    case 4: {
        return msg4s;
    }
    case 6: {
        return msg6s;
    }
    default: {
        std::cout << "No msg buffers of dim " << int(dim) << "\n";
        exit(-1);
    }
    }
}

VariableArrayBase& FactorGraph::vars_of_dimension(uint8_t dim) {
    switch (dim) {
    case 4: {
        return var4s;
    }
    case 6: {
        return var6s;
    }
    default: {
        std::cout << "No vars of dim" << dim << "\n";
        exit(-1);
    }
    }
}

void FactorGraph::execute(const ToVariableMsg_SendContext& ctx) {
    auto& vars = vars_of_dimension(ctx.variable_dim);
    vars.zero_all();

    // update vectors
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto src_vec = to_eigen_map(ctx.vectors[i]);
        auto dst_vec = to_eigen_map(vars.get_vec(ctx.variable_idxs[i]));
        dst_vec += src_vec;
    }

    // update matrices
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto src_mat = to_eigen_map(ctx.matrices[i]);
        auto dst_mat = to_eigen_map(vars.get_mat(ctx.variable_idxs[i]));
        dst_mat += src_mat;
    }
};

template <uint8_t var_dim>
void execute_impl(const ToVariableMsg_RetrievalContext& ctx,
                  FactorGraph& graph) {
    // std::cout << "retreiving msgs to vars of dim " << int(ctx.variable_dim) << "\n";
    
    auto matrices = static_dim<var_dim>(ctx.matrices);
    auto vectors = static_dim<var_dim>(ctx.vectors);

    // retrieve vecs
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const auto [factor_dim, _ignore] = ctx.factor_ids[i];
        const size_t factor_idx = ctx.factor_idxs[i];
        auto dst_vec = to_eigen_map(vectors[i]);
        auto src_vec = to_eigen_map(graph.factors.of_dimension(factor_dim).get_cov_vec(factor_idx));

        dst_vec = src_vec.segment<var_dim>(ctx.factor_slots[i]);
    }

    // retrieve mats
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const auto [factor_dim, _ignore] = ctx.factor_ids[i];
        const size_t factor_idx = ctx.factor_idxs[i];
        auto dst_mat = to_eigen_map(matrices[i]);
        auto src_mat = to_eigen_map(graph.factors.of_dimension(factor_dim).get_cov_mat(factor_idx));

        dst_mat = src_mat.block<var_dim, var_dim>(ctx.factor_slots[i], ctx.factor_slots[i]);
    }

    // info-covariance transform
    for (size_t i = 0; i < matrices.size; ++i) {
        auto dst_mat = to_eigen_map(matrices[i]);
        auto dst_vec = to_eigen_map(vectors[i]);

        const Eigen::Matrix<double, var_dim, 1> result_vec = dst_mat.ldlt().solve(dst_vec); 
        const Eigen::Matrix<double, var_dim, var_dim> result_mat = dst_mat.inverse();

        dst_mat = result_mat;
        dst_vec = result_vec;
    };

}

void FactorGraph::execute(const ToVariableMsg_RetrievalContext& ctx) {
    switch (ctx.variable_dim) {
    case 4: return execute_impl<4>(ctx, *this);
    case 6: return execute_impl<6>(ctx, *this);
    default:
        std::cout << "execute: no variables of dim " << int(ctx.variable_dim) << "\n";
        exit(-1);
    }
}

void FactorGraph::execute(const ToFactorMsg_RetrievalContext& ctx) {
    auto& vars = vars_of_dimension(ctx.variable_dim);

    // retrieve mats
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const size_t var_idx = ctx.variable_idxs[i];
        auto var_mat = to_eigen_map(vars.get_mat(var_idx));
        auto msg_mat = to_eigen_map(ctx.matrices[i]);
        msg_mat *= -1;
        msg_mat += var_mat;
    }

    // retrieve vecs
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const size_t var_idx = ctx.variable_idxs[i];
        auto var_vec = to_eigen_map(vars.get_vec(var_idx));
        auto msg_vec = to_eigen_map(ctx.vectors[i]);
        msg_vec *= -1;
        msg_vec += var_vec;
    }
}

void FactorGraph::execute(const ToFactorMsg_SendContext& ctx) {
    // send mats
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto [factor_dim, _ignore] = ctx.factor_ids[i];
        auto dst = to_eigen_map(factors.of_dimension(factor_dim).get_info_mat_delta(ctx.factor_idxs[i]));
        auto src = to_eigen_map(ctx.matrices[i]);
        dst.block(ctx.factor_slots[i], ctx.factor_slots[i],
                  ctx.variable_dim, ctx.variable_dim) = src;
    }

    // send vecs
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto [factor_dim, _ignore] = ctx.factor_ids[i];
        auto dst = to_eigen_map(factors.of_dimension(factor_dim).get_info_vec_delta(ctx.factor_idxs[i]));
        auto src = to_eigen_map(ctx.vectors[i]);
        dst.segment(ctx.factor_slots[i], ctx.variable_dim) = src;
    }
}

void FactorGraph::initial_update() {
    // install priors, do not try to retrieve from factors since they are not ready yet
    for (uint8_t msg_dim : variable_dims) {
        auto& to_var_msgs = msg_buffers_of_dimension(msg_dim).to_variable_msgs();
        execute(to_var_msgs.generate_send_context());
    }

    // now variables are non-degenerate (if they came with priors)
}

void FactorGraph::update() {
    // send messages to factors
    for (uint8_t msg_dim : variable_dims) {
        auto& to_factor_msgs = msg_buffers_of_dimension(msg_dim).to_factor_msgs();
        to_factor_msgs.rebuild_retrieval_buffer();
        execute(to_factor_msgs.generate_retrieval_context());
        to_factor_msgs.commit_retrievals();
        execute(to_factor_msgs.generate_send_context());
    }

    for (uint8_t factor_dim : FactorArrays::factor_dims) {
        factors.of_dimension(factor_dim).rebuild_cov_mats();
    }

    // send messages to variables
    for (uint8_t msg_dim : variable_dims) {
        auto& to_var_msgs = msg_buffers_of_dimension(msg_dim).to_variable_msgs();
        to_var_msgs.rebuild_retrieval_buffer();
        execute(to_var_msgs.generate_retrieval_context());
        to_var_msgs.commit_retrievals();
        execute(to_var_msgs.generate_send_context());
    }

    for (uint8_t var_dim : variable_dims) {
        vars_of_dimension(var_dim).rebuild_cov_mats();
    }    
}

void FactorGraph::add_edge(VariableId variable_id, FactorId factor_id, uint8_t slot) {
    auto [var_dim, _ignore] = variable_id;
    msg_buffers_of_dimension(var_dim).add_edge(variable_id, factor_id, slot);
}

void FactorGraph::commit() {
    for (uint8_t msg_dim : variable_dims) {
        msg_buffers_of_dimension(msg_dim).to_factor_msgs().commit_edges(
            vars_of_dimension(msg_dim).ids(),
            factors);
    }

    for (uint8_t msg_dim : variable_dims) {
        msg_buffers_of_dimension(msg_dim).to_variable_msgs().commit_edges(
            vars_of_dimension(msg_dim).ids(),
            factors);
    }
}

std::pair<VectorRef, MatrixRef> FactorGraph::get_mean_cov(VariableId variable_id) {
    auto [var_dim, _ignore] = variable_id;
    const auto& vars = vars_of_dimension(var_dim);
    size_t var_idx = SIZE_MAX;
    for (size_t i = 0; i < vars.ids().size(); ++i) {
        if (vars.ids()[i] == variable_id) {
            var_idx = i;
            break;
        }
    }

    if (var_idx == SIZE_MAX) {
        std::cout << "variable not found" << "\n";
        exit(-1);
    }

    return std::make_pair(vars.get_mean(var_idx), vars.get_cov(var_idx));
}

}
