#pragma once
#include <iostream>
#include <cstdint>
#include "Eigen/Dense"
#include "entt/entity/registry.hpp"
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
    INFO_FORM_VEC,
    INFO_FORM_MAT,
    INFO_FORM_DELTA_VEC,
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
using Mean = TaggedMatrix<dim, 1, MatrixTag::COVARIANCE_FORM_MAT>;
template <int dim>
using Covariance = TaggedMatrix<dim, dim, MatrixTag::COVARIANCE_FORM_VEC>;

template <int dim>
using ToVariableMatrix = TaggedMatrix<dim, dim, MatrixTag::TO_VARIABLE_MESSAGE_MAT>;
template <int dim>
using ToVariableVector = TaggedMatrix<dim, 1, MatrixTag::TO_VARIABLE_MESSAGE_VEC>;

template <int dim>
using ToFactorMatrix = TaggedMatrix<dim, dim, MatrixTag::TO_FACTOR_MESSAGE_MAT>;
template <int dim>
using ToFactorVector = TaggedMatrix<dim, 1, MatrixTag::TO_FACTOR_MESSAGE_VEC>;

template <int dim>
using LinearizationPoint = TaggedMatrix<dim, 1, MatrixTag::LINEARIZATION_POINT>;

// empty struct to tag edges going
// from variables of dimension var_dim
// to nonlinear factors
template <int var_dim>
struct NonlinearTag {};

struct FactorTypes {
    entt::id_type info_matrix;
    entt::id_type info_vector;
    entt::id_type info_matrix_delta;
    entt::id_type info_vector_delta;
    entt::id_type mean;
    entt::id_type covariance;
    entt::id_type linearization_point;
};

// basically a manually implemented vtable
template <int dim>
constexpr FactorTypes make_factor_types() {
    return {
        entt::type_hash<InfoMatrix<dim>>::value(),
        entt::type_hash<InfoVector<dim>>::value(),
        entt::type_hash<InfoMatrixDelta<dim>>::value(),
        entt::type_hash<InfoVectorDelta<dim>>::value(),
        entt::type_hash<Mean<dim>>::value(),
        entt::type_hash<Covariance<dim>>::value(),
        entt::type_hash<LinearizationPoint<dim>>::value()
    };
}

constexpr std::array<FactorTypes, 21> FACTOR_TYPES {
    make_factor_types<1>(), // zeroth entry is arbitrary

    make_factor_types<1>(), make_factor_types<2>(),
    make_factor_types<3>(), make_factor_types<4>(),
    make_factor_types<5>(), make_factor_types<6>(),
    make_factor_types<7>(), make_factor_types<8>(),
    make_factor_types<9>(), make_factor_types<10>(),    
    make_factor_types<11>(), make_factor_types<12>(),    
    make_factor_types<13>(), make_factor_types<14>(),    
    make_factor_types<15>(), make_factor_types<16>(),    
    make_factor_types<17>(), make_factor_types<18>(),    
    make_factor_types<19>(), make_factor_types<20>()
};

struct FactorInfo {
    uint8_t dimension;
    uint8_t type;
    bool nonlinear = false;
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

    double regularizer = 1e3;

    template <int dim>
    constexpr static bool variable_dimension_supported() {
        return ((dim == int(V_DIMS)) || ...);
    }

    template <int dim>
    constexpr static bool factor_dimension_supported() {
        return ((dim == int(F_DIMS)) || ...);
    }

    template <int dim>
    VariableHandle<dim> add_variable(
        uint8_t type,
        const Eigen::Matrix<double, dim, 1>& prior_mean,
        const Eigen::Matrix<double, dim, dim>& prior_covariance) {
        static_assert(variable_dimension_supported<dim>());

        const auto variable_id = variables.create();

        variables.emplace<VariableInfo>(variable_id, uint8_t(dim), type);
        variables.emplace<Mean<dim>>(variable_id, prior_mean);
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

        edges.emplace<VariableConnection>(edge_id, variable, uint8_t(dim));
        edges.emplace<ToVariableMatrix<dim>>(edge_id, ToVariableMatrix<dim>::Zero());
        edges.emplace<ToVariableVector<dim>>(edge_id, ToVariableVector<dim>::Zero());

        edges.emplace<FactorConnection>(edge_id, factor, factor_dimension, factor_slot);
        edges.emplace<ToFactorMatrix<dim>>(edge_id, ToFactorMatrix<dim>::Zero());
        edges.emplace<ToFactorVector<dim>>(edge_id, ToFactorVector<dim>::Zero());

        if (factor_info.nonlinear) {
            edges.emplace<NonlinearTag<dim>>(edge_id);
        }

        return edge_id;
    }

    template <int dim>
    FactorHandle<dim> add_factor(uint8_t type, bool nonlinear) {
        static_assert(factor_dimension_supported<dim>());

        const auto factor_id = factors.create();
        factors.emplace<FactorInfo>(factor_id, uint8_t(dim), type, nonlinear);
        factors.emplace<InfoVectorDelta<dim>>(factor_id, InfoVectorDelta<dim>::Zero());
        factors.emplace<InfoMatrixDelta<dim>>(factor_id, InfoMatrixDelta<dim>::Zero());
        factors.emplace<Mean<dim>>(factor_id, Mean<dim>::Zero());
        factors.emplace<Covariance<dim>>(factor_id, Covariance<dim>::Zero());

        // indeterminate info vectors and matrix
        factors.emplace<InfoVector<dim>>(factor_id);
        factors.emplace<InfoMatrix<dim>>(factor_id);

        if (nonlinear) {
            factors.emplace<LinearizationPoint<dim>>(factor_id);
        }

        return FactorHandle<dim>{factor_id};
    }

    template <int dim>
    entt::entity add_factor(
        uint8_t type, bool nonlinear,
        const Eigen::Matrix<double, dim, 1>& info_vector,
        const Eigen::Matrix<double, dim, dim>& info_matrix) {
        const auto factor_id = add_factor<dim>(type, nonlinear);
        factors.get<InfoVector<dim>>(factor_id) = info_vector;
        factors.get<InfoMatrix<dim>>(factor_id) = info_matrix;
        return factor_id;
    }

    template <int variable_dim>
    void update_factors_impl() {
        // prepare to_factor_matrix messages
        for (auto [_, connection, to_variable_matrix, to_factor_matrix] :
                 edges.template view<VariableConnection, ToVariableMatrix<variable_dim>, ToFactorMatrix<variable_dim>>().each()) {
            auto& variable_info_matrix = variables.template get<InfoMatrix<variable_dim>>(connection.variable);
            to_factor_matrix = variable_info_matrix - to_variable_matrix;
        }

        // prepare to_factor_vector messages
        for (auto [_, connection, to_variable_vector, to_factor_vector] :
                 edges.template view<VariableConnection, ToVariableVector<variable_dim>, ToFactorVector<variable_dim>>().each()) {
            auto& variable_info_vector = variables.template get<InfoVector<variable_dim>>(connection.variable);
            to_factor_vector = variable_info_vector - to_variable_vector;
        }

        // write the info matrix block
        for (auto [_, connection, to_factor_matrix, to_factor_vector] :
                 edges.template view<FactorConnection, ToFactorMatrix<variable_dim>, ToFactorVector<variable_dim>>().each()) {
            const uint8_t factor_dimension = connection.factor_dimension;
            const FactorTypes& factor_types = FACTOR_TYPES[factor_dimension];
            void* info_matrix_data = factors.storage(factor_types.info_matrix_delta)->second.get(connection.factor);
            auto info_matrix = eigen_matrix_map(info_matrix_data, factor_dimension);
            info_matrix.block<variable_dim, variable_dim>(connection.factor_slot, connection.factor_slot) = to_factor_matrix;
        }

        // write the info vector block
        for (auto [_, connection, to_factor_matrix, to_factor_vector] :
                 edges.template view<FactorConnection, ToFactorMatrix<variable_dim>, ToFactorVector<variable_dim>>().each()) {
            const uint8_t factor_dimension = connection.factor_dimension;
            const FactorTypes& factor_types = FACTOR_TYPES[factor_dimension];
            void* info_vector_data = factors.storage(factor_types.info_vector_delta)->second.get(connection.factor);
            auto info_vector = eigen_vector_map(info_vector_data, factor_dimension);
            info_vector.segment<variable_dim>(connection.factor_slot) = to_factor_vector;
        }

        // prepare relinearization
        // std::cout << "prepping lin point for var dims " << int(variable_dim) << "\n";
        // for linearization in factors
        // [ pick up the variable state, write it into the linearization point ]
        // to-factor-linearization?
    }

    template <int dim>
    void update_factors_finish_impl() {
        // update mean and covariance
        for (auto [_, factor_info_matrix, factor_info_matrix_delta, covariance] :
                 factors.template view<InfoMatrix<dim>, InfoMatrixDelta<dim>, Covariance<dim>>().each()) {
            covariance = factor_info_matrix + factor_info_matrix_delta;
            // std::cout << "info mat\n" << covariance << "\n";

        }

        for (auto [_, info_vector, info_vector_delta, mean] :
                 factors.template view<InfoVector<dim>, InfoVectorDelta<dim>, Mean<dim>>().each()) {
            mean = info_vector + info_vector_delta;
            // std::cout << "info vec\n" << mean.transpose() << "\n";
        }

        for (auto [_, mean, covariance] : factors.template view<Mean<dim>, Covariance<dim>>().each()) {
            auto llt = covariance.llt();
            Mean<dim> m = llt.solve(mean);
            Covariance<dim> c = llt.solve(Eigen::id<dim>());
            mean = m;
            covariance = c;
            // std::cout << "joint mean " << mean.transpose() << "\n";
        }
    }

    template <int dim>
    void update_variables_impl() {
        // retreive matrix message from factor covariance
        for (auto [_, factor_connection, to_variable_matrix] : edges.template view<FactorConnection, ToVariableMatrix<dim>>().each()) {
            const uint8_t factor_dimension = factor_connection.factor_dimension;
            const FactorTypes& factor_types = FACTOR_TYPES[factor_dimension];
            void* covariance_data = factors.storage(factor_types.covariance)->second.get(factor_connection.factor);
            auto covariance = eigen_matrix_map(covariance_data, factor_dimension);
            to_variable_matrix = covariance.block<dim, dim>(factor_connection.factor_slot, factor_connection.factor_slot);
        }

        // retrieve vector message from factor mean
        for (auto [_, factor_connection, to_variable_vector] : edges.template view<FactorConnection, ToVariableVector<dim>>().each()) {
            const uint8_t factor_dimension = factor_connection.factor_dimension;
            const FactorTypes& factor_types = FACTOR_TYPES[factor_dimension];
            void* mean_data = factors.storage(factor_types.mean)->second.get(factor_connection.factor);
            auto mean = eigen_vector_map(mean_data, factor_dimension);
            to_variable_vector = mean.segment<dim>(factor_connection.factor_slot);
            // std::cout << "pre-fixup to-variable "<< entt::to_entity(_) <<" mean " << to_variable_vector.transpose() << "\n";
        }

        // fixup the messages by removing the self-contribution (to_factor_matrix and to_factor_vector)
        for (auto [_, to_variable_matrix, to_variable_vector, to_factor_matrix, to_factor_vector] :
                 edges.template view<ToVariableMatrix<dim>, ToVariableVector<dim>, ToFactorMatrix<dim>, ToFactorVector<dim>>().each()) {
            const ToFactorMatrix<dim> info_mat_perturb = -to_factor_matrix;
            const ToFactorVector<dim> info_vec_perturb = -to_factor_vector;
            const Eigen::Matrix<double, dim, dim> fixer = info_marginalization_fixer(to_variable_matrix, info_mat_perturb);
            to_variable_matrix = perturb_marginalization<dim>(fixer, to_variable_matrix);
            to_variable_vector = perturb_marginalization<dim>(fixer, to_variable_matrix, info_vec_perturb, to_variable_vector);
            // std::cout << "post-fixup to-variable " << entt::to_entity(_) <<" mean " << to_variable_vector.transpose() << "\n";
        }

        // convert the message from covariance to info form
        for (auto [_, to_variable_matrix, to_variable_vector] :
                 edges.template view<ToVariableMatrix<dim>, ToVariableVector<dim>>().each()) {
            auto llt = to_variable_matrix.llt();
            ToVariableVector<dim> info_vector = llt.solve(to_variable_vector);
            ToVariableMatrix<dim> info_matrix = llt.solve(Eigen::id<dim>());
            to_variable_matrix = info_matrix;
            to_variable_vector = info_vector;
        }

        // update the info deltas
        for (auto [_, matrix] : variables.template view<InfoMatrixDelta<dim>>().each()) {
            matrix.setZero();
        }
        for (auto [_, vector] : variables.template view<InfoVectorDelta<dim>>().each()) {
            vector.setZero();
        }
        for (auto [_, connection, to_variable_matrix] :
                 edges.template view<VariableConnection, ToVariableMatrix<dim>>().each()) {
            variables.template get<InfoMatrixDelta<dim>>(connection.variable) += to_variable_matrix;
        }
        for (auto [_, connection, to_variable_vector] :
                 edges.template view<VariableConnection, ToVariableVector<dim>>().each()) {
            variables.template get<InfoVectorDelta<dim>>(connection.variable) += to_variable_vector;
        }

        // load the priors
        for (auto [_, info_matrix, info_matrix_prior] :
                 variables.template view<InfoMatrix<dim>, InfoMatrixPrior<dim>>().each()) {
            info_matrix = info_matrix_prior;

        }
        for (auto [_, info_vector, info_vector_prior] :
                 variables.template view<InfoVector<dim>, InfoVectorPrior<dim>>().each()) {
            info_vector = info_vector_prior;
        }
    
        // add in the deltas 
        for (auto [_, info_matrix, info_matrix_delta] :
                 variables.template view<InfoMatrix<dim>, InfoMatrixDelta<dim>>().each()) {
            // std::cout << entt::to_entity(_) << " info matrix delta\n" << info_matrix_delta << "\n";
            info_matrix += info_matrix_delta;
        }
        for (auto [_, info_vector, info_vector_delta] :
                 variables.template view<InfoVector<dim>, InfoVectorDelta<dim>>().each()) {
            // std::cout << entt::to_entity(_) << " info vector delta " << info_vector_delta.transpose() << "\n";                 
            info_vector += info_vector_delta;
        }

        // update the variable mean, covariances
        for (auto [_, covariance, info_matrix] : variables.template view<Covariance<dim>, InfoMatrix<dim>>().each()) {
            covariance = info_matrix;
        }
        for (auto [_, mean, info_vector] : variables.template view<Mean<dim>, InfoVector<dim>>().each()) {
            // std::cout << entt::to_entity(_) << " prev mean " << mean.transpose() << "\n";                 
            mean = info_vector;
        }
        for (auto [_, mean, covariance] : variables.template view<Mean<dim>, Covariance<dim>>().each()) {
            auto llt = covariance.llt();
            const Mean<dim> m = llt.solve(mean);
            mean = m;
            // std::cout << entt::to_entity(_) << " updated mean " << mean.transpose() << "\n";                 
            const Covariance<dim> c = llt.solve(Eigen::id<dim>());
            covariance = c;
        }
    }

    void update_variables() {
        ((update_variables_impl<V_DIMS>()), ...);
    }

    void update_factors() {
        ((update_factors_impl<V_DIMS>()), ...);
        ((update_factors_finish_impl<F_DIMS>()), ...);
    }

    template <int dim>
    Eigen::Matrix<double, dim, 1> get_mean(VariableHandle<dim> variable) {
        return variables.get<Mean<dim>>(variable.entity);
    }

    template <int variable_dim>
    void begin_linearization_impl() {
        // Fetch the linearization points of each nonlinear factor.
        // There is a choice here to gather the linearization point from the variable registry
        // or from the to-factor message. The latter choice should be accurate since it subtracts
        // out the information that this factor sent out to the variable. But this
        // requires an info-form to covariance-form conversion. The former choice just uses
        // the fully marginalized estimates of the variable means. That's what we do for now.

        for (auto [_, factor_connection, variable_connection] :
                 edges.template view<FactorConnection, VariableConnection, NonlinearTag<variable_dim>>().each()) {
            const uint8_t factor_dimension = factor_connection.factor_dimension;
            const FactorTypes& factor_types = FACTOR_TYPES[factor_dimension];
            void* linearization_point_data = factors.storage(factor_types.linearization_point)->second.get(factor_connection.factor);
            auto linearization_point = eigen_vector_map(linearization_point_data, factor_dimension);
            const auto& variable_mean = variables.template get<Mean<variable_dim>>(variable_connection.variable);
            // std::cout << "writing " << variable_mean.transpose() << " to " << int(factor_connection.factor_slot) << "\n";
            linearization_point.segment<variable_dim>(factor_connection.factor_slot) = variable_mean;
        }
    }

    template <int factor_dim>
    void end_linearization_impl() {
        // Assuming linearization about input x0, the factor info vectors
        // rtJ are Δ-based, where Δ = x - x0, arising from the following
        // cost function.
        // || JΔ - r ||² = Δ.t J.tJ Δ - 2 r.t J Δ + r.t r
        //
        // This is equivalent to the following.
        // || J(x - x0) - r ||² = (x - x0).t J.tJ (x- x0) - 2 r.t J (x - x0) + r.t r
        // || J(x - x0) - r ||² = x.t J.tJ x - 2 x0.t J.t J x - 2 r.t J x + constant
        // || J(x - x0) - r ||² = x.t J.tJ x - 2 (x0.t J.t J + r.t J) x + constant
        //
        // So the information vector component gets modified by += x0.t J.t J, and
        // J.t J is the information matrix component.

        for (auto [_, linearization_point, info_vector, info_matrix] :
                 factors.template view<LinearizationPoint<factor_dim>, InfoVector<factor_dim>, InfoMatrix<factor_dim>>().each()) {
            info_vector += info_matrix * linearization_point;
        }
    }

    void begin_linearization() {
        ((begin_linearization_impl<V_DIMS>()), ...);
    }

    void end_linearization() {
        ((end_linearization_impl<F_DIMS>()), ...);
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
