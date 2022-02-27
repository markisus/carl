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
        std::cout << "No msg buffers of dim " << dim << "\n";
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

FactorArrayBase& FactorGraph::factors_of_dimension(uint8_t dim) {
    switch (dim) {
    case 4: {
        return factor4s;
    }
    case 6: {
        return factor6s;
    }
    case 16: {
        return factor16s;
    }
    default:
        std::cout << "No factors of dim " << dim << "\n";
        exit(-1);
    }
}

void FactorGraph::execute(const ToVariableMsg_SendContext& ctx) {
    auto& vars = vars_of_dimension(ctx.variable_dim);
    vars.zero_all();

    // update vectors
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto dst_vec = to_eigen_map(vars.get_vec(ctx.variable_idxs[i]));
        dst_vec += to_eigen_map(ctx.vectors[i]);
    }

    // update matrices
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto dst_mat = to_eigen_map(vars.get_mat(ctx.variable_idxs[i]));
        dst_mat += to_eigen_map(ctx.matrices[i]);
    }
};

template <uint8_t var_dim>
void execute_impl(const ToVariableMsg_RetrievalContext& ctx,
                  FactorGraph& graph) {
    auto matrices = static_dim<var_dim>(ctx.matrices);
    auto vectors = static_dim<var_dim>(ctx.vectors);

    // retrieve vecs
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const auto [factor_dim, _ignore] = ctx.factor_ids[i];
        const size_t factor_idx = ctx.factor_idxs[i];
        auto dst_vec = to_eigen_map(vectors[i]);
        auto src_vec = to_eigen_map(graph.factors_of_dimension(factor_dim).get_cov_vec(factor_idx));
        dst_vec = src_vec.segment<var_dim>(ctx.factor_slots[i]);
    }

    // retrieve mats
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const auto [factor_dim, _ignore] = ctx.factor_ids[i];
        const size_t factor_idx = ctx.factor_idxs[i];
        auto dst_mat = to_eigen_map(matrices[i]);
        auto src_mat = to_eigen_map(graph.factors_of_dimension(factor_dim).get_cov_mat(factor_idx));
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
    }
}

void FactorGraph::execute(const ToFactorMsg_RetrievalContext& ctx) {
    // retrieve mats
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const size_t var_idx = ctx.variable_idxs[i];
        auto var_mat = to_eigen_map(vars_of_dimension(ctx.variable_dim).get_mat(var_idx));
        auto msg_mat = to_eigen_map(ctx.matrices[i]);
        msg_mat *= -1;
        msg_mat += var_mat;
    }

    // retrieve vecs
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        const size_t var_idx = ctx.variable_idxs[i];
        auto var_vec = to_eigen_map(vars_of_dimension(ctx.variable_dim).get_vec(var_idx));
        auto msg_vec = to_eigen_map(ctx.vectors[i]);
        msg_vec *= -1;
        msg_vec += var_vec;
    }
}

void FactorGraph::execute(const ToFactorMsg_SendContext& ctx) {
    // send mats
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto [factor_dim, _ignore] = ctx.factor_ids[i];
        auto dst = to_eigen_map(factors_of_dimension(factor_dim).get_info_mat_delta(ctx.factor_idxs[i]));
        auto src = to_eigen_map(ctx.matrices[i]);
        dst.block(ctx.factor_slots[i], ctx.factor_slots[i],
                  ctx.variable_dim, ctx.variable_dim) = src;
    }

    // send vecs
    for (size_t i = 0; i < ctx.num_msgs; ++i) {
        auto [factor_dim, _ignore] = ctx.factor_ids[i];
        auto dst = to_eigen_map(factors_of_dimension(factor_dim).get_info_vec_delta(ctx.factor_idxs[i]));
        auto src = to_eigen_map(ctx.vectors[i]);
        dst.segment(ctx.factor_slots[i], ctx.variable_dim) = src;
    }
}


void FactorGraph::update() {
    for (uint8_t msg_dim : { 4, 6 }) {
        auto& to_var_msgs = msg_buffers_of_dimension(msg_dim).to_variable_msgs();
        to_var_msgs.rebuild_retrieval_buffer();
        execute(to_var_msgs.generate_retrieval_context());
        to_var_msgs.commit_retrievals();
        execute(to_var_msgs.generate_send_context());
    }

    // send messages to factors
    for (uint8_t msg_dim : { 4, 6 }) {
        auto& to_factor_msgs = msg_buffers_of_dimension(msg_dim).to_factor_msgs();
        to_factor_msgs.rebuild_retrieval_buffer();
        execute(to_factor_msgs.generate_retrieval_context());
        to_factor_msgs.commit_retrievals();
        execute(to_factor_msgs.generate_send_context());
    }

    for (uint8_t factor_dim : { 4, 6, 16 }) {
        auto& factors = factors_of_dimension(factor_dim);
        factors.rebuild_cov_mats();
    }
}

Id FactorGraph::add_camera_matrix(const Eigen::Matrix<double, 4, 1>& mean,
                                  const double fx_fy_stddev,
                                  const double cx_cy_stddev) {

    Eigen::Matrix<double, 4, 4> covariance;
    const double fx_fy_var = std::pow(fx_fy_stddev, 2);
    const double cx_cy_var = std::pow(cx_cy_stddev, 2);
    covariance.setZero();
    covariance(0,0) = fx_fy_var;
    covariance(1,1) = fx_fy_var;
    covariance(2,2) = cx_cy_var;
    covariance(3,3) = cx_cy_var;

    Eigen::Matrix<double, 4, 1> info_vec = covariance.ldlt().solve(mean);
    Eigen::Matrix<double, 4, 4> info_mat = covariance.inverse();

    const Id variable_id = var4s.create_variable();
    const FactorId prior_factor_id = factor4s.create_factor(info_mat.data(), info_vec.data());
    msg4s.add_edge(variable_id, prior_factor_id, 0);

    return variable_id;
}

}
