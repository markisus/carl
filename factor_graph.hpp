#pragma once
#include <iostream>
#include <cstdint>
#include "Eigen/Dense"
#include "entt/entity/registry.hpp"
#include "layout_data.h"
#include "gaussian_math.hpp"
#include "eigen_util.h"

inline Eigen::Map<Eigen::VectorXd> eigen_vector_map(void* data, uint8_t dim) {
    return Eigen::Map<Eigen::VectorXd> {(double*)data, dim, 1};
}
inline Eigen::Map<Eigen::MatrixXd> eigen_matrix_map(void* data, uint8_t dim) {
    return Eigen::Map<Eigen::MatrixXd> {(double*)data, dim, dim};
}

namespace carl {

enum class MatrixTag {
    COVARIANCE_FORM_VEC,
    COVARIANCE_FORM_MAT,
    COVARIANCE_FORM_PRIOR_VEC, // aka prior mean
    INFO_FORM_VEC,
    INFO_FORM_MAT,
    INFO_FORM_DELTA_VEC,
    INFO_FORM_DELTA_VEC_RECENTERED, // recentered around factor linearization point
    INFO_FORM_DELTA_MAT,
    INFO_FORM_PRIOR_VEC,
    INFO_FORM_PRIOR_MAT,
    TO_FACTOR_MESSAGE_VEC,
    TO_FACTOR_MESSAGE_MAT,
    TO_VARIABLE_MESSAGE_VEC,
    TO_VARIABLE_MESSAGE_MAT,
    LINEARIZATION_POINT
};

template <int rows, int cols, MatrixTag tag>
struct TaggedMatrix : public Eigen::Matrix<double, rows, cols> {
    // constructor just passes all arguments up to parent Eigen matrix
    template <typename ...Args>
    TaggedMatrix(Args... args) : Eigen::Matrix<double, rows, cols>(args...) {}
};

template <int dim>
using InfoMatrix = TaggedMatrix<dim, dim, MatrixTag::INFO_FORM_MAT>;
template <int dim>
using InfoVector = TaggedMatrix<dim, 1, MatrixTag::INFO_FORM_VEC>;

template <int dim>
using InfoMatrixPrior = TaggedMatrix<dim, dim, MatrixTag::INFO_FORM_PRIOR_MAT>;
template <int dim>
using InfoVectorPrior = TaggedMatrix<dim, 1, MatrixTag::INFO_FORM_PRIOR_VEC>;

template <int dim>
using InfoMatrixDelta = TaggedMatrix<dim, dim, MatrixTag::INFO_FORM_DELTA_MAT>;
template <int dim>
using InfoVectorDelta = TaggedMatrix<dim, 1, MatrixTag::INFO_FORM_DELTA_VEC>;
template <int dim>
using InfoVectorDeltaRecentered = TaggedMatrix<dim, 1, MatrixTag::INFO_FORM_DELTA_VEC_RECENTERED>;

template <int dim>
using Mean = TaggedMatrix<dim, 1, MatrixTag::COVARIANCE_FORM_VEC>;
template <int dim>
using Covariance = TaggedMatrix<dim, dim, MatrixTag::COVARIANCE_FORM_MAT>;

template <int dim>
using PriorMean = TaggedMatrix<dim, 1, MatrixTag::COVARIANCE_FORM_PRIOR_VEC>;

template <int dim>
struct ToVariableMessage {
    Eigen::MatrixD<dim> matrix = Eigen::zero_mat<dim>();
    Eigen::VectorD<dim> vector = Eigen::zero_vec<dim>();
};

template <int dim>
struct ToFactorMessage {
    Eigen::MatrixD<dim> matrix = Eigen::zero_mat<dim>();
    Eigen::VectorD<dim> vector = Eigen::zero_vec<dim>();
};

template <int dim>
struct ToVariableMessageQueued {
    Eigen::MatrixD<dim> matrix = Eigen::zero_mat<dim>();
    Eigen::VectorD<dim> vector = Eigen::zero_vec<dim>();
};

template <int dim>
struct ToFactorMessageQueued {
    Eigen::MatrixD<dim> matrix = Eigen::zero_mat<dim>();
    Eigen::VectorD<dim> vector = Eigen::zero_vec<dim>();
};

template <typename T, typename Tqd>
double get_heuristic_residual(const Tqd& msg_qd, const T& msg) {
    // todo: do this better
    const auto mean = msg.matrix.llt().solve(msg.vector).eval();
    const auto mean_qd = msg_qd.matrix.llt().solve(msg_qd.vector).eval();
    const double scale = mean.template lpNorm<Eigen::Infinity>();
    const double diff = (mean - mean_qd).template lpNorm<Eigen::Infinity>();

    if ((!mean.allFinite()) || (!mean_qd.allFinite())) {
        // std::cout << "got nonfinite mean in get_heuristic_residual" << "\n";
        // std::cout << "mean" << "\n";
        // std::cout << mean.transpose() << "\n";
        // std::cout << "mean_qd" << "\n";
        // std::cout << mean_qd.transpose() << "\n";
        // std::cout << "info vec" << "\n";
        // std::cout << msg.vector.transpose() << "\n";
        // std::cout << "info vec qd" << "\n";
        // std::cout << msg_qd.vector.transpose() << "\n";
        // std::cout << "info mat" << "\n";
        // std::cout << msg.matrix << "\n";
        // std::cout << "info mat qd" << "\n";
        // std::cout << msg_qd.matrix << "\n";
        // exit(-1);
        return 1000;
    }

    const double result = diff/(scale + 1e-6);
    if (!std::isfinite(result)) {
        std::cout << "returning nonfinite result to get_heuristic_residual" << "\n";
        std::cout << "mean" << "\n";
        std::cout << mean.transpose() << "\n";
        std::cout << "mean_qd" << "\n";
        std::cout << mean_qd.transpose() << "\n";
        std::cout << "info vec" << "\n";
        std::cout << msg.vector.transpose() << "\n";
        std::cout << "info vec qd" << "\n";
        std::cout << msg_qd.vector.transpose() << "\n";
        std::cout << "info mat" << "\n";
        std::cout << msg.matrix << "\n";
        std::cout << "info mat qd" << "\n";
        std::cout << msg_qd.matrix << "\n";
    }
    
    return result;
}

template <int dim>
using LinearizationPoint = TaggedMatrix<dim, 1, MatrixTag::LINEARIZATION_POINT>;

// empty struct to tag edges going
// from variables of dimension var_dim
// to nonlinear factors
template <int var_dim>
struct NonlinearTag {};

struct FactorNeedsInitFlag {};
struct FactorNeedsRelinearize {};
struct EnableFlag {};

struct MatrixTypes {
    entt::id_type info_matrix;
    entt::id_type info_vector;
    entt::id_type info_matrix_delta;
    entt::id_type info_vector_delta;
    entt::id_type info_vector_delta_recentered;
    entt::id_type mean;
    entt::id_type covariance;
    entt::id_type linearization_point;
};

// basically a manually implemented vtable
template <int dim>
constexpr MatrixTypes make_matrix_types() {
    return {
        entt::type_hash<InfoMatrix<dim>>::value(),
        entt::type_hash<InfoVector<dim>>::value(),
        entt::type_hash<InfoMatrixDelta<dim>>::value(),
        entt::type_hash<InfoVectorDelta<dim>>::value(),
        entt::type_hash<InfoVectorDeltaRecentered<dim>>::value(),
        entt::type_hash<Mean<dim>>::value(),
        entt::type_hash<Covariance<dim>>::value(),
        entt::type_hash<LinearizationPoint<dim>>::value()
    };
}

constexpr std::array<MatrixTypes, 21> MATRIX_TYPES {
    make_matrix_types<1>(), // zeroth entry is arbitrary

    make_matrix_types<1>(), make_matrix_types<2>(),
    make_matrix_types<3>(), make_matrix_types<4>(),
    make_matrix_types<5>(), make_matrix_types<6>(),
    make_matrix_types<7>(), make_matrix_types<8>(),
    make_matrix_types<9>(), make_matrix_types<10>(),    
    make_matrix_types<11>(), make_matrix_types<12>(),    
    make_matrix_types<13>(), make_matrix_types<14>(),    
    make_matrix_types<15>(), make_matrix_types<16>(),    
    make_matrix_types<17>(), make_matrix_types<18>(),    
    make_matrix_types<19>(), make_matrix_types<20>()
};

struct FactorInfo {
    uint8_t dimension;
    entt::id_type type;
    bool nonlinear = false;
};

template <int DIM, bool NONLINEAR> 
struct FactorData {
    static constexpr int dimension = DIM;
    static constexpr bool nonlinear = NONLINEAR;
};

struct VariableInfo {
    uint8_t dimension;
    uint8_t type;
};

struct FactorConnection {
    entt::entity factor;
    uint8_t factor_dimension;
    uint8_t factor_slot;
};

struct VariableConnection {
    entt::entity variable;
    uint8_t variable_dimension;
};

template <typename F, typename D>
struct FactorGraph;

template <int dim>
struct VariableHandle {
    entt::entity entity;
    operator entt::entity() const { return entity; }
};

template <int dim>
struct FactorHandle {
    entt::entity entity;
    operator entt::entity() const { return entity; }
};

template <uint8_t... DIMS>
using Dims = std::integer_sequence<uint8_t, DIMS...>;

template <uint8_t... F_DIMS, uint8_t... V_DIMS>
struct FactorGraph<Dims<F_DIMS...>, Dims<V_DIMS...>> {
    entt::registry factors;
    entt::registry variables;
    entt::registry edges;

    double max_factor_change = 0;

    double regularizer = 1.0;
    double total_error = 0;

    bool layout_converged = false;
    double layout_energy = 0;
    double layout_step = 1.0;
    uint8_t layout_progress = 0;
    double layout_width = 1.0;
    double layout_height = 1.0;

    template <int dim>
    constexpr static bool variable_dimension_supported() {
        return ((dim == int(V_DIMS)) || ...);
    }

    template <int dim>
    constexpr static bool factor_dimension_supported() {
        return ((dim == int(F_DIMS)) || ...);
    }

    void visit_variable_layout(void (*visiter)(LayoutData*, VariableError*, void*), void* user_data) {
        for (auto [_, layout, variable_error] : variables.view<LayoutData, VariableError>().each()) {
            visiter(&layout, &variable_error, user_data);
        }
    }

    void visit_edge_layout(void (*visiter)(LayoutData*, LayoutData*, EdgeResidual*, void*), void* user_data) {
        for (auto [edge, factor_connection, variable_connection, residual] :
                 edges.view<FactorConnection, VariableConnection, EdgeResidual>().each()) {
            auto& v_layout = variables.get<LayoutData>(variable_connection.variable);
            auto& f_layout = factors.get<LayoutData>(factor_connection.factor);
            visiter(&f_layout, &v_layout, &residual, user_data);
        }
    }

    void visit_factor_layout(void(*visiter)(LayoutData*, FactorError*, void*), void* user_data) {
        for (auto [_, layout, factor_error] : factors.view<LayoutData, FactorError>().each()) {
            visiter(&layout, &factor_error, user_data);
        }
    }

    void reset_layout_damping() {
        layout_converged = false;
        layout_energy = 0;
        layout_step = 1.0;
        layout_progress = 0;
    }

    double update_layout() {
        // Algorithm based off description here
        // http://yifanhu.net/PUB/graph_draw_small.pdf
        // No quadtree optimizations used, however.

        if (layout_converged) {
            return layout_energy;
        }

        const double C = 0.2;
        const double K = 1.0;
        // reset forces
        for (auto [_, v_layout] : variables.view<LayoutData>().each()) {
            v_layout.force.setZero();
        }
        for (auto [_, f_layout] : factors.view<LayoutData>().each()) {
            f_layout.force.setZero();
        }
        // factor-variable attraction
        for (auto [_, variable_connection, factor_connection] : edges.view<VariableConnection, FactorConnection>().each()) {
            auto& v_layout = variables.get<LayoutData>(variable_connection.variable);
            auto& f_layout = factors.get<LayoutData>(factor_connection.factor);
            const Eigen::VectorD<2> displacement = v_layout.position - f_layout.position;
            const double r = displacement.norm();
            const double force_scale = -r/K;
            v_layout.force += displacement * force_scale;
            f_layout.force -= displacement * force_scale;
        }
        // factor-variable repulsion
        for (auto [f, f_layout] : factors.view<LayoutData>().each()) {
            for (auto [v, v_layout] : variables.view<LayoutData>().each()) {
                const Eigen::VectorD<2> displacement = v_layout.position - f_layout.position;
                const double r = displacement.norm();
                const double force_scale = C*K*K/(r*r) * 0.5;
                v_layout.force += displacement * force_scale;
                f_layout.force -= displacement * force_scale;                
            }
        }
        // variable-variable repulsion
        for (auto [v0, v0_layout] : variables.view<LayoutData>().each()) {
            for (auto [v1, v1_layout] : variables.view<LayoutData>().each()) {
                if (v0 == v1) continue;
                const Eigen::VectorD<2> displacement = v0_layout.position - v1_layout.position;
                const double r = displacement.norm();
                const double force_scale = C*K*K/(r*r) * 0.5;                
                v0_layout.force += displacement * force_scale;
                v1_layout.force -= displacement * force_scale;                
                
            }
        }
        // factor-factor repulsion
        for (auto [f0, f0_layout] : factors.view<LayoutData>().each()) {
            for (auto [f1, f1_layout] : factors.view<LayoutData>().each()) {
                if (f0 == f1) continue;
                const Eigen::VectorD<2> displacement = f0_layout.position - f1_layout.position;
                const double r = displacement.norm();
                const double force_scale = C*K*K/(r*r) * 0.5;                                
                f0_layout.force += displacement * force_scale;
                f1_layout.force -= displacement * force_scale;                
                
            }
        }

        const double step_damp = 0.5;
        double energy = 0;

        Eigen::VectorD<2> center_of_mass = Eigen::zero_vec<2>();
        double layout_min_x = std::numeric_limits<double>::infinity();
        double layout_max_x = -std::numeric_limits<double>::infinity();
        double layout_min_y = std::numeric_limits<double>::infinity();
        double layout_max_y = -std::numeric_limits<double>::infinity();
        
        int num_particles = 0;

        // move particles along forces
        for (auto [_, layout] : variables.view<LayoutData>().each()) {
            layout.position += layout.force.normalized() * layout_step;
            center_of_mass += layout.position;
            num_particles += 1;
            energy += layout.force.squaredNorm();

            layout_min_x = std::min(layout_min_x, layout.position(0));
            layout_max_x = std::max(layout_max_x, layout.position(0));
            layout_min_y = std::min(layout_min_y, layout.position(1));
            layout_max_y = std::max(layout_max_y, layout.position(1));
        }
        for (auto [_, layout] : factors.view<LayoutData>().each()) {
            layout.position += layout.force.normalized() * layout_step;
            center_of_mass += layout.position;
            num_particles += 1;
            energy += layout.force.squaredNorm();

            layout_min_x = std::min(layout_min_x, layout.position(0));
            layout_max_x = std::max(layout_max_x, layout.position(0));
            layout_min_y = std::min(layout_min_y, layout.position(1));
            layout_max_y = std::max(layout_max_y, layout.position(1));
        }

        layout_width = layout_max_x - layout_min_x;
        layout_height = layout_max_y - layout_min_y;
        if (!std::isfinite(layout_width) or layout_width <= 1.0) {
            layout_width = 1.0;
        }
        if (!std::isfinite(layout_height) or layout_height <= 1.0) {
            layout_height = 1.0;
        }

        // recenter all by center of mass
        center_of_mass /= num_particles;
        for (auto [_, layout] : variables.view<LayoutData>().each()) {
            layout.position -= center_of_mass;
        }
        for (auto [_, layout] : factors.view<LayoutData>().each()) {
            layout.position -= center_of_mass;
        }

        if (energy < layout_energy) {
            layout_progress += 1;

            if (layout_progress >= 5) {
                layout_progress = 0;
                layout_step /= step_damp;
            }            
        } else {
            layout_step *= step_damp;
            layout_progress = 0;

        }

        if (layout_step <= 1e-2) {
            layout_converged = true;
        }

        layout_energy = energy;
        return layout_energy;
    }

    template <int dim>
    VariableHandle<dim> add_variable(
        uint8_t type,
        const Eigen::Matrix<double, dim, 1>& prior_mean,
        const Eigen::Matrix<double, dim, dim>& prior_covariance) {
        static_assert(variable_dimension_supported<dim>());

        const auto variable_id = variables.create();

        variables.emplace<EnableFlag>(variable_id);
        variables.emplace<VariableInfo>(variable_id, uint8_t(dim), type);
        variables.emplace<Mean<dim>>(variable_id, prior_mean);
        variables.emplace<PriorMean<dim>>(variable_id, prior_mean);
        variables.emplace<Covariance<dim>>(variable_id, prior_covariance);

        auto llt = prior_covariance.llt();
        const InfoMatrix<dim> info_matrix = llt.solve(Eigen::id<dim>());
        const InfoVector<dim> info_vector = llt.solve(prior_mean);
        variables.emplace<InfoMatrix<dim>>(variable_id, info_matrix);
        variables.emplace<InfoVector<dim>>(variable_id, info_vector);
    
        variables.emplace<InfoMatrixPrior<dim>>(variable_id, info_matrix);
        variables.emplace<InfoVectorPrior<dim>>(variable_id, info_vector);

        variables.emplace<InfoMatrixDelta<dim>>(variable_id, InfoMatrixDelta<dim>::Zero());
        variables.emplace<InfoVectorDelta<dim>>(variable_id, InfoVectorDelta<dim>::Zero());

        variables.emplace<LayoutData>(variable_id);
        variables.emplace<VariableError>(variable_id);

        reset_layout_damping();

        return VariableHandle<dim>{variable_id};
    }

    template <int dim>
    entt::entity add_edge(const VariableHandle<dim> variable_handle,
                          const entt::entity factor,
                          const uint8_t factor_slot) {
        entt::entity variable = variable_handle.entity;
        const auto& factor_info = factors.get<FactorInfo>(factor);
        const uint8_t factor_dimension = factor_info.dimension;
        const uint8_t variable_dimension = variables.get<VariableInfo>(variable).dimension;
        assert((variable_dimension == dim));
        assert((factor_slot < factor_dimension));

        const auto edge_id = edges.create();
        edges.emplace<EnableFlag>(edge_id);
        edges.emplace<FactorNeedsInitFlag>(edge_id);
        edges.emplace<VariableConnection>(edge_id, variable, uint8_t(dim));
        edges.emplace<FactorConnection>(edge_id, factor, factor_dimension, factor_slot);

        edges.emplace<ToVariableMessage<dim>>(edge_id);        
        edges.emplace<ToFactorMessage<dim>>(edge_id);

        edges.emplace<ToVariableMessageQueued<dim>>(edge_id);        
        edges.emplace<ToFactorMessageQueued<dim>>(edge_id);

        edges.emplace<EdgeResidual>(edge_id);

        // build up the factor state supplied by this variable
        Eigen::Map<Eigen::VectorXd> factor_linearization_point = get_factor_linearization_point(factor, factor_dimension);
        const Mean<dim>& variable_mean = variables.get<Mean<dim>>(variable);
        factor_linearization_point.segment<dim>(factor_slot) = variable_mean;
        
        reset_layout_damping();

        return edge_id;
    }

    void add_edges_finish() {
        ((queue_to_factor_messages_impl<V_DIMS, FactorNeedsInitFlag>()), ...);
        ((update_factors_finish_impl_1<V_DIMS, FactorNeedsInitFlag>()), ...);
        ((update_factors_finish_impl_2<F_DIMS, FactorNeedsInitFlag>()), ...);
        ((queue_to_variable_messages_impl<V_DIMS, FactorNeedsInitFlag>()), ...);
        factors.clear<FactorNeedsInitFlag>();
        edges.clear<FactorNeedsInitFlag>();
    }

    Eigen::Map<Eigen::MatrixXd> get_factor_covariance(entt::entity factor, int factor_dim) {
        const MatrixTypes& factor_types = MATRIX_TYPES[factor_dim];
        void* raw = factors.storage(factor_types.covariance)->second.get(factor);
        return eigen_matrix_map(raw, factor_dim);
    };

    Eigen::Map<Eigen::VectorXd> get_factor_mean(entt::entity factor, int factor_dim) {
        const MatrixTypes& factor_types = MATRIX_TYPES[factor_dim];
        void* raw = factors.storage(factor_types.mean)->second.get(factor);
        return eigen_vector_map(raw, factor_dim);
    };

    Eigen::Map<Eigen::MatrixXd> get_factor_info_mat_delta(entt::entity factor, int factor_dim) {
        const MatrixTypes& factor_types = MATRIX_TYPES[factor_dim];
        void* raw = factors.storage(factor_types.info_matrix_delta)->second.get(factor);
        return eigen_matrix_map(raw, factor_dim);
    };

    Eigen::Map<Eigen::VectorXd> get_factor_info_vec_delta(entt::entity factor, int factor_dim) {
        const MatrixTypes& factor_types = MATRIX_TYPES[factor_dim];
        void* raw = factors.storage(factor_types.info_vector_delta)->second.get(factor);
        return eigen_vector_map(raw, factor_dim);
    };

    Eigen::Map<Eigen::VectorXd> get_factor_linearization_point(entt::entity factor, int factor_dim) {
        const MatrixTypes& factor_types = MATRIX_TYPES[factor_dim];
        void* raw = factors.storage(factor_types.linearization_point)->second.get(factor);
        return eigen_vector_map(raw, factor_dim);
    };

    Eigen::Map<Eigen::VectorXd> get_factor_info_vec_delta_recentered(entt::entity factor, int factor_dim) {
        const MatrixTypes& factor_types = MATRIX_TYPES[factor_dim];
        void* raw = factors.storage(factor_types.info_vector_delta_recentered)->second.get(factor);
        return eigen_vector_map(raw, factor_dim);
    };

    void set_display_string(entt::entity factor, const std::string& display_string) {
        factors.get<FactorError>(factor).display_string = display_string;
    }

    template <typename TFactorData>
    auto add_factor(TFactorData&& factor_data) {
        static_assert(factor_dimension_supported<factor_data.dimension>());
        
        constexpr int dim = factor_data.dimension;
        constexpr bool nonlinear = factor_data.nonlinear;

        const auto factor_id = factors.create();

        using TFactorDataDecayed = typename std::decay<TFactorData>::type;
        factors.emplace<TFactorDataDecayed>(factor_id, std::forward<TFactorData>(factor_data));
        factors.emplace<FactorInfo>(factor_id, uint8_t(dim), entt::type_hash<TFactorDataDecayed>::value(), nonlinear);

        factors.emplace<InfoVectorDelta<dim>>(factor_id, InfoVectorDelta<dim>::Zero());
        factors.emplace<InfoVectorDeltaRecentered<dim>>(factor_id, InfoVectorDeltaRecentered<dim>::Zero());        
        factors.emplace<InfoMatrixDelta<dim>>(factor_id, InfoMatrixDelta<dim>::Zero());

        factors.emplace<Mean<dim>>(factor_id, Mean<dim>::Zero());
        factors.emplace<Covariance<dim>>(factor_id, Covariance<dim>::Zero());

        // indeterminate info vectors and matrix
        factors.emplace<InfoVector<dim>>(factor_id);
        factors.emplace<InfoMatrix<dim>>(factor_id);

        factors.emplace<LayoutData>(factor_id);
        factors.emplace<FactorError>(factor_id);
        factors.emplace<LinearizationPoint<dim>>(factor_id);

        factors.emplace<EnableFlag>(factor_id);
        factors.emplace<FactorNeedsInitFlag>(factor_id);

        reset_layout_damping();        

        return FactorHandle<dim>{factor_id};
    }

    template <int variable_dim, typename Flag=EnableFlag>
    void queue_to_factor_messages_impl() {
        // only edges need enabled
        // write the messages into the factor delta entries
        for (auto [_, vconn, to_variable_message, to_factor_message_qd] :
                 edges.template view<Flag,
                 VariableConnection,
                 ToVariableMessage<variable_dim>,
                 ToFactorMessageQueued<variable_dim>>().each()) {
            // prepare to_factor_matrix messages by subtracting
            // out the self information from the factor
            const auto& variable_info_vector = variables.template get<InfoVector<variable_dim>>(vconn.variable);
            const auto& variable_info_matrix = variables.template get<InfoMatrix<variable_dim>>(vconn.variable);
            to_factor_message_qd.matrix = variable_info_matrix - to_variable_message.matrix;
            to_factor_message_qd.vector = variable_info_vector - to_variable_message.vector;
        }

        // update residuals
        for (auto [edge, to_factor_message_qd, to_factor_message, residual] : 
                 edges.template view<Flag,
                 ToFactorMessageQueued<variable_dim>, ToFactorMessage<variable_dim>, EdgeResidual>().each()) {
            residual.to_factor = get_heuristic_residual(to_factor_message_qd, to_factor_message);
            const auto variable = edges.get<VariableConnection>(edge).variable;
            const auto factor = edges.get<FactorConnection>(edge).factor;
            // std::cout << "\tResidualizing edge " << entt::to_entity(edge) << " from var " <<
            //     entt::to_entity(variable) << " to factor " <<
            //     entt::to_entity(factor) << "\n";
            // std::cout << "set residual to " << residual.to_factor << "\n";
        }
    }

    template <int dim>
    void update_factor_errors() {
        max_factor_change = 0;
        // update factor errors
        for (auto [_, factor_error, mean, info_vector, info_matrix] :
                 factors.template view<
                 FactorError, Mean<dim>, InfoVector<dim>, InfoMatrix<dim>>().each()) {
            // || J Δ - r ||² = Δ.t J.tJ Δ - 2 r.t J Δ + r.t r
            //                  ~~~~~~~~~~~~~~~~~~~~~~   ~~~~~
            //                            |                |
            //                           delta           offset
            factor_error.delta = mean.transpose() * info_matrix * mean - 2.0 * info_vector.dot(mean);
        }
    }

    template <int dim, typename Flag=EnableFlag>
    void update_factors_finish_impl_1() {
        // must call update_factors_finish_2 after this
        
        // write the queued updates into the factor deltas
        // write the queued update into the actual update
        for (auto [_, to_factor_message_qd, to_factor_message, residual] : edges.template view<
                 Flag,
                 ToFactorMessageQueued<dim>,
                 ToFactorMessage<dim>,
                 EdgeResidual>().each()) {
            to_factor_message.vector = to_factor_message_qd.vector;
            to_factor_message.matrix = to_factor_message_qd.matrix;
            residual.to_factor = 0;
        }
        
        for (auto [_, fconn, to_factor_message] : edges.template view<
                 Flag,
                 FactorConnection,
                 ToFactorMessage<dim>>().each()) {
            Eigen::Map<Eigen::MatrixXd> info_matrix = get_factor_info_mat_delta(fconn.factor, fconn.factor_dimension);
            info_matrix.block<dim, dim>(fconn.factor_slot, fconn.factor_slot) = to_factor_message.matrix;
            Eigen::Map<Eigen::VectorXd> info_vector = get_factor_info_vec_delta(fconn.factor, fconn.factor_dimension);
            info_vector.segment<dim>(fconn.factor_slot) = to_factor_message.vector;
        }
    }
    

    template <int dim, typename Flag=EnableFlag>
    void update_factors_finish_impl_2() {
        // Actually carry out the update from the incoming message buffers
        // if an edge is enabled, the connected factor must be enabled
        
        for (auto [_, mean, covariance, info_vector,
                   info_vector_delta, info_vector_delta_recentered, linearization_point,
                   info_matrix, info_matrix_delta, factor_error] :
                 factors.template view<
                 Flag, Mean<dim>, Covariance<dim>,                 
                 InfoVector<dim>, InfoVectorDelta<dim>, InfoVectorDeltaRecentered<dim>,
                 LinearizationPoint<dim>,
                 InfoMatrix<dim>, InfoMatrixDelta<dim>, FactorError>().each()) {

            info_vector_delta_recentered = info_vector_delta - info_matrix_delta * linearization_point;
            const Eigen::MatrixD<dim> total_info_matrix = info_matrix + info_matrix_delta;
            const Eigen::VectorD<dim> total_info_vector = info_vector + info_vector_delta_recentered;

            Eigen::VectorD<dim> previous_mean = mean;
            auto llt = total_info_matrix.llt();
            mean = llt.solve(total_info_vector);
            covariance = llt.solve(Eigen::id<dim>());

            if (!covariance.allFinite()) {
                std::cout << "Factor " << entt::to_entity(_) << " finished with non-finite covariance" << "\n";
                exit(-1);
            }

            if (!mean.allFinite()) {
                std::cout << "Factor " << entt::to_entity(_) << " finished with non-finite mean" << "\n";
                exit(-1);
            }

            double change = (mean-previous_mean).template lpNorm<Eigen::Infinity>();
            factor_error.change = change;
            factor_error.age = 0;
        }

        update_factor_errors<dim>();
    }

    void age_variables() {
        // age the variables
        for (auto [_, variable_error]: variables.template view<VariableError>().each()) {
            variable_error.age += 1;
            if (variable_error.age > MAX_AGE) {
                variable_error.age = MAX_AGE;
            }
        }
    }

    void age_factors() {
        // age the factors
        for (auto [_, factor_error]: factors.template view<FactorError>().each()) {
            factor_error.age += 1;
            if (factor_error.age > MAX_AGE) {
                factor_error.age = MAX_AGE;
            }
        }
    }

    template <int dim, typename Flag=EnableFlag>
    void queue_to_variable_messages_impl() {
        // only touches edges
        // std::cout << "Updating variables of dim " << int(dim) << "\n";

        for (auto [_, factor_connection, to_variable_message_qd, to_factor_message] : edges.template view<
                 EnableFlag,
                 FactorConnection, ToVariableMessageQueued<dim>, ToFactorMessage<dim>>().each()) {
            const uint8_t factor_dimension = factor_connection.factor_dimension;
            Eigen::Map<Eigen::VectorXd> mean = get_factor_mean(factor_connection.factor, factor_dimension);
            Eigen::Map<Eigen::MatrixXd> covariance = get_factor_covariance(factor_connection.factor, factor_dimension);
            to_variable_message_qd.vector = mean.segment<dim>(factor_connection.factor_slot);
            to_variable_message_qd.matrix = covariance.block<dim, dim>(factor_connection.factor_slot, factor_connection.factor_slot);

            // sanity checks
            if (!to_variable_message_qd.vector.allFinite()) {
                std::cout << "Variable got non-finite vector message from factor " << entt::to_entity(factor_connection.factor) << "\n";
                exit(-1);
            }

            // transform from delta-based mean to 0-based mean by adding in the linearization point
            to_variable_message_qd.vector +=
                get_factor_linearization_point(factor_connection.factor, factor_dimension).template segment<dim>(factor_connection.factor_slot);

            // fixup the messages by removing the self-contribution (to_factor_matrix and to_factor_vector)
            // add in the linearization point of the factor

            // pull out the data that was sent to the factor
            const Eigen::MatrixD<dim> info_mat_perturb = -to_factor_message.matrix;
            const Eigen::VectorD<dim> info_vec_perturb = -to_factor_message.vector;

            const Eigen::Matrix<double, dim, dim> fixer = info_marginalization_fixer(to_variable_message_qd.matrix, info_mat_perturb);
            to_variable_message_qd.matrix = perturb_marginalization<dim>(fixer, to_variable_message_qd.matrix);
            to_variable_message_qd.vector = perturb_marginalization<dim>(fixer,
                                                                         to_variable_message_qd.matrix,
                                                                         info_vec_perturb,
                                                                         to_variable_message_qd.vector);

            // convert the message from covariance to info form
            auto llt = to_variable_message_qd.matrix.llt();
            const Eigen::VectorD<dim> info_vector = llt.solve(to_variable_message_qd.vector);
            const Eigen::MatrixD<dim> info_matrix = llt.solve(Eigen::id<dim>());
            to_variable_message_qd.matrix = info_matrix;
            to_variable_message_qd.vector = info_vector;
        }

        // update residuals
        for (auto [_, factor_connection, to_variable_message_qd, to_variable_message, residual] : edges.template view<
                 EnableFlag,
                 FactorConnection, ToVariableMessageQueued<dim>, ToVariableMessage<dim>, EdgeResidual>().each()) {
            residual.to_variable = get_heuristic_residual(to_variable_message_qd, to_variable_message);
            if (!std::isfinite(residual.to_variable)) {
                std::cout << "got nonfinite to variable residual" << "\n";
                std::cout << "to variable message" << "\n";
                std::cout << to_variable_message.vector.transpose() << "\n";
                std::cout << to_variable_message.matrix << "\n";
                std::cout << "mean" << "\n";
                std::cout << (to_variable_message.matrix.llt().solve(to_variable_message.vector)).transpose() << "\n";
                std::cout << "to variable message qd" << "\n";
                std::cout << to_variable_message_qd.vector.transpose() << "\n";
                std::cout << to_variable_message_qd.matrix << "\n";
                exit(-1);
            }
        }
    }

    template <int dim, typename Flag=EnableFlag>
    void update_variables_finish_impl() {
        // actually carry out the updates in the message buffers
        // if an edge is enabled, the variable it's connected to must be enabled

        // load variables from queue
        for (auto [_,
                   connection,
                   to_variable_message_qd,
                   to_variable_message,
                   residual] :
                 edges.template view<
                 EnableFlag,
                 VariableConnection,
                 ToVariableMessageQueued<dim>,
                 ToVariableMessage<dim>,
                 EdgeResidual>().each()) {
            if (!to_variable_message_qd.vector.allFinite()) {
                std::cout << "update_variables_finish_impl was nonfinite " << to_variable_message_qd.vector.transpose() << "\n";
                exit(-1);
            }
            
            variables.template get<InfoMatrixDelta<dim>>(connection.variable) += to_variable_message_qd.matrix - to_variable_message.matrix;
            variables.template get<InfoVectorDelta<dim>>(connection.variable) += to_variable_message_qd.vector - to_variable_message.vector;
            to_variable_message.matrix = to_variable_message_qd.matrix;
            to_variable_message.vector = to_variable_message_qd.vector;
            residual.to_variable = 0;
        }
        
        // update the variable mean, covariances
        for (auto [_,
                   mean, covariance,
                   info_vector, info_matrix,
                   info_vector_delta, info_matrix_delta,
                   info_vector_prior, info_matrix_prior] : variables.template view<
                 EnableFlag,
                 Mean<dim>, Covariance<dim>,
                 InfoVector<dim>, InfoMatrix<dim>,
                 InfoVectorDelta<dim>, InfoMatrixDelta<dim>,
                 InfoVectorPrior<dim>, InfoMatrixPrior<dim>>().each()) {

            // std::cout << entt::to_entity(_) << " original mean " << mean.transpose() << "\n";
            info_vector = info_vector_delta + info_vector_prior*regularizer;
            info_matrix = info_matrix_delta + info_matrix_prior*regularizer;

            auto llt = info_matrix.llt();
            mean = llt.solve(info_vector);
            covariance = info_matrix.llt().solve(Eigen::id<dim>());
        }

        // update the prior errors
        // (x - μ).t Λ (x - μ)
        for (auto [_, mean, prior_mean, prior_info_mat, prior_info_vec, variable_error] : variables.template view<
                 EnableFlag,
                 Mean<dim>, PriorMean<dim>, InfoMatrixPrior<dim>, InfoVectorPrior<dim>, VariableError>().each()) {
            auto delta = (mean - prior_mean);
            variable_error.prior_error = delta.transpose() * prior_info_mat * delta;

            // moving the prior mean
            // (x - (μ+d)).t Λ (x - (μ+d)) = 
            // x.t Λ x - 2 (μ+d).t Λ x         + constant
            //         - 2 μ.t Λ - 2 d.t Λ x
            // increment the info vec by d
            auto increment = ((mean - prior_mean) * 0.5).eval();
            prior_mean += increment;
            prior_info_vec += prior_info_mat * increment;

            variable_error.age = 0;
        }
    }

    void disable_all() {
        edges.clear<EnableFlag>();
        variables.clear<EnableFlag>();
        factors.clear<EnableFlag>();
    }

    void enable_all() {
        factors.each([&](auto entity){
            factors.emplace_or_replace<EnableFlag>(entity);
        });
        variables.each([&](auto entity){
            variables.emplace_or_replace<EnableFlag>(entity);
        });
        edges.each([&](auto entity) {
            edges.emplace_or_replace<EnableFlag>(entity);
        });
    }

    void update_async() {
        disable_all();

        edges.sort<EdgeResidual>([](const auto& lhs, const auto& rhs) {
            return lhs.to_variable > rhs.to_variable;
        });

        // update variables connected to most changed edge
        const int MAX_EDGES = 10;
        int num_var_edges = 0;
        for (auto [edge, residual] : edges.template view<EdgeResidual>().each()) {
            edges.emplace<EnableFlag>(edge);
            // const auto factor = edges.get<FactorConnection>(edge).factor;
            const auto variable = edges.get<VariableConnection>(edge).variable;
            variables.template emplace_or_replace<EnableFlag>(variable);
            num_var_edges += 1;
            if (num_var_edges >= MAX_EDGES) {
                break;
            }
        }

        ((update_variables_finish_impl<V_DIMS>()), ...);
        age_variables();

        // update factor messages connected to enabled variables edge
        for (auto [edge, vconn] : edges.template view<VariableConnection>().each()) {
            const auto variable = vconn.variable;
            if (variables.any_of<EnableFlag>(variable)) {
                // const auto variable = edges.get<VariableConnection>(edge).variable;
                // const auto factor = edges.get<FactorConnection>(edge).factor;
                // std::cout << "\tEnabling edge from var " <<
                //     entt::to_entity(variable) << " to factor " <<
                //     entt::to_entity(factor) << "\n";
                edges.template emplace_or_replace<EnableFlag>(edge);
            }
        }
        // refresh the queued factor messages
        // which have all updated due to the variable change
        ((queue_to_factor_messages_impl<V_DIMS>()), ...);

        disable_all();

        edges.sort<EdgeResidual>([](const auto& lhs, const auto& rhs) {
                return lhs.to_factor > rhs.to_factor;
            });

        double max_fac_residual = 0;
        for (auto [_, residual] : edges.template view<EdgeResidual>().each()) {
            max_fac_residual = std::max(max_fac_residual, residual.to_factor);
        }

        // update factors connected to most changed edge
        int num_fac_edges = 0;
        for (auto [edge, residual] : edges.template view<EdgeResidual>().each()) {
            edges.emplace<EnableFlag>(edge);
            // enable variables and variables connected to this edge
            // const auto variable = edges.get<VariableConnection>(edge).variable;
            const auto factor = edges.get<FactorConnection>(edge).factor;
            factors.template emplace_or_replace<EnableFlag>(factor);
            num_fac_edges += 1;
            if (num_fac_edges >= MAX_EDGES) {
                break;
            }
        }
        
        // update factors connected to most changed edge
        ((update_factors_finish_impl_1<V_DIMS>()), ...);
        ((update_factors_finish_impl_2<F_DIMS>()), ...);
        update_factors_finish_impl_3();
        age_factors();

        // update factor messages connected to enabled variables edge
        for (auto [edge, fconn] : edges.template view<FactorConnection>().each()) {
            const auto factor = fconn.factor;
            if (factors.any_of<EnableFlag>(factor)) {
                // const auto variable = edges.get<VariableConnection>(edge).variable;
                // const auto factor = edges.get<FactorConnection>(edge).factor;
                // std::cout << "\tEnabling edge from var " <<
                //     entt::to_entity(variable) << " to factor " <<
                // entt::to_entity(factor) << "\n";
                edges.template emplace_or_replace<EnableFlag>(edge);
            }
        }

        ((queue_to_variable_messages_impl<V_DIMS>()), ...);
        enable_all();
    }

    void update_factors_finish_impl_3() {
        // update total error
        double prev_error = total_error;
        total_error = 0;
        for (auto [_, factor_error] : factors.template view<FactorError>().each()) {
            total_error += factor_error.total();
        }

        // std::cout << "error changed from " << prev_error << "=> " <<total_error << "\n";
        // const double max_regularizer = 1e6;
        // const double min_regularizer = 1e-3;
        // if (total_error < prev_error) {
        //     regularizer *= 0.5;
        // } else {
        //     regularizer *= 3;
        // }
        // regularizer = std::clamp(regularizer, min_regularizer, max_regularizer);
        // std::cout << "regularizer is now " << regularizer << "\n";        

        for (auto [_, factor_error] : factors.template view<FactorError>().each()) {
            max_factor_change = std::max(factor_error.change, max_factor_change);
        }
    }

    void update_variables() {
        ((update_variables_finish_impl<V_DIMS>()), ...);
        ((queue_to_factor_messages_impl<V_DIMS>()), ...);
        age_variables();
    }

    void update_factors() {
        ((update_factors_finish_impl_1<V_DIMS>()), ...);
        ((update_factors_finish_impl_2<F_DIMS>()), ...);
        update_factors_finish_impl_3();
        ((queue_to_variable_messages_impl<V_DIMS>()), ...);
        age_factors();
    }

    template <int dim>
    Eigen::Matrix<double, dim, 1> get_mean(VariableHandle<dim> variable) {
        return variables.get<Mean<dim>>(variable.entity);
    }

    template <int dim>
    Eigen::Matrix<double, dim, 1> get_mean(FactorHandle<dim> factor) {
        return factors.get<Mean<dim>>(factor.entity);
    }

    template <int dim>
    Eigen::Matrix<double, dim, dim> get_covariance(VariableHandle<dim> variable) {
        return variables.get<Covariance<dim>>(variable.entity);
    }

    template <int factor_dim>
    void begin_local_linearization_impl() {
        const double trust_region = 0.005;
        for (auto [factor, linearization_point, mean] : factors.template view<LinearizationPoint<factor_dim>, Mean<factor_dim>>().each()) {
            if (mean.template lpNorm<Eigen::Infinity>() >= trust_region) {
                linearization_point += mean;
                mean.setZero();
                factors.emplace<FactorNeedsRelinearize>(factor);
            }
        }

    }

    void end_linearization() {
        ((update_factors_finish_impl_1<V_DIMS, FactorNeedsRelinearize>()), ...);
        ((update_factors_finish_impl_2<F_DIMS, FactorNeedsRelinearize>()), ...);        
        ((update_factor_errors<F_DIMS>()), ...);
        factors.clear<FactorNeedsRelinearize>();
        edges.clear<FactorNeedsRelinearize>();
    }

    void begin_linearization() {
        // after running this function,
        // caller is responsible for setting
        // - info matrix
        // - info vector
        // - factor error offset
        ((begin_local_linearization_impl<F_DIMS>()), ...);

        // mark edges connected to factors needing relinearize
        for (auto [edge, fconn] : edges.template view<FactorConnection>().each()) {
            if (factors.any_of<FactorNeedsRelinearize>(fconn.factor)) {
                edges.emplace_or_replace<FactorNeedsRelinearize>(edge);
            }
        }
    }

    template <int dim>
    Eigen::Matrix<double, dim, 1> get_linearization_point(FactorHandle<dim> factor) {
        return factors.get<LinearizationPoint<dim>>(factor.entity);
    }

    template <int dim>
    Eigen::Matrix<double, dim, dim>* get_factor_info_matrix(FactorHandle<dim> factor) {
        return &factors.get<InfoMatrix<dim>>(factor.entity);
    }

    template <int dim>
    Eigen::Matrix<double, dim, 1>* get_factor_info_vector(FactorHandle<dim> factor) {
        return &factors.get<InfoVector<dim>>(factor.entity);
    }
    
};


}
