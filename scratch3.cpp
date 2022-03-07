#include <iostream>
#include <cstdint>
#include "Eigen/Dense"
#include "entt/entity/registry.hpp"
#include "math.hpp"

struct FactorMeta {
    uint8_t dimension;
    uint8_t type;
};

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

struct FactorTypes {
    entt::id_type info_matrix;
    entt::id_type info_vector;
    entt::id_type info_matrix_delta;
    entt::id_type info_vector_delta;
    entt::id_type mean;
    entt::id_type covariance;
};

template <int dim>
constexpr FactorTypes make_factor_types() {
    return {
        entt::type_hash<InfoMatrix<dim>>::value(),
        entt::type_hash<InfoVector<dim>>::value(),
        entt::type_hash<InfoMatrixDelta<dim>>::value(),
        entt::type_hash<InfoVectorDelta<dim>>::value(),
        entt::type_hash<Mean<dim>>::value(),
        entt::type_hash<Covariance<dim>>::value() };
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
};

struct VariableInfo {
    uint8_t dimension;
};

struct FactorGraph {
    entt::registry factors;
    entt::registry variables;
    entt::registry edges;
};

template <int dim>
entt::entity add_variable(const Eigen::Matrix<double, dim, 1>& prior_mean,
                          const Eigen::Matrix<double, dim, dim>& prior_covariance,
                          FactorGraph& graph) {
    const auto variable_id = graph.variables.create();

    graph.variables.emplace<VariableInfo>(variable_id, uint8_t(dim));
    graph.variables.emplace<Mean<dim>>(variable_id, prior_mean);
    graph.variables.emplace<Covariance<dim>>(variable_id, prior_covariance);

    const InfoMatrix<dim> info_matrix = prior_covariance.inverse();
    const InfoVector<dim> info_vector = prior_covariance.ldlt().solve(prior_mean);
    graph.variables.emplace<InfoMatrix<dim>>(variable_id, info_matrix);
    graph.variables.emplace<InfoVector<dim>>(variable_id, info_vector);
    
    graph.variables.emplace<InfoMatrixPrior<dim>>(variable_id, info_matrix);
    graph.variables.emplace<InfoVectorPrior<dim>>(variable_id, info_vector);

    graph.variables.emplace<InfoMatrixDelta<dim>>(variable_id, InfoMatrixDelta<dim>::Zero());
    graph.variables.emplace<InfoVectorDelta<dim>>(variable_id, InfoVectorDelta<dim>::Zero());
    return variable_id;
}

struct FactorConnection {
    entt::entity factor;
    uint8_t factor_dimension;
    uint8_t factor_slot;
};

struct VariableConnection {
    entt::entity variable;
    uint8_t variable_dimension;
};

template <int dim>
entt::entity add_edge(const entt::entity variable,
                      const entt::entity factor,
                      const uint8_t factor_slot,
                      FactorGraph& graph) {

    const uint8_t factor_dimension = graph.factors.get<FactorInfo>(factor).dimension;
    const uint8_t variable_dimension = graph.variables.get<VariableInfo>(variable).dimension;
    assert((variable_dimension == dim));

    const auto edge_id = graph.edges.create();
    graph.edges.emplace<FactorConnection>(edge_id, factor, factor_dimension, factor_slot);
    graph.edges.emplace<VariableConnection>(edge_id, variable, uint8_t(dim));
    graph.edges.emplace<ToVariableMatrix<dim>>(edge_id, ToVariableMatrix<dim>::Zero());
    graph.edges.emplace<ToVariableVector<dim>>(edge_id, ToVariableVector<dim>::Zero());
    graph.edges.emplace<ToFactorMatrix<dim>>(edge_id, ToFactorMatrix<dim>::Zero());
    graph.edges.emplace<ToFactorVector<dim>>(edge_id, ToFactorVector<dim>::Zero());

    return edge_id;
}

template <int dim>
entt::entity add_factor(
    const Eigen::Matrix<double, dim, 1>& info_vector,
    const Eigen::Matrix<double, dim, dim>& info_matrix,
    FactorGraph& graph) {

    //std::cout << "adding info_matrix\n" << info_matrix << "\n";
    //std::cout << "adding info_vector " << info_vector.transpose() << "\n";


    const auto factor_id = graph.factors.create();
    graph.factors.emplace<FactorInfo>(factor_id, uint8_t(dim));
    graph.factors.emplace<InfoVector<dim>>(factor_id, info_vector);
    graph.factors.emplace<InfoMatrix<dim>>(factor_id, info_matrix);
    graph.factors.emplace<InfoVectorDelta<dim>>(factor_id, InfoVectorDelta<dim>::Zero());
    graph.factors.emplace<InfoMatrixDelta<dim>>(factor_id, InfoMatrixDelta<dim>::Zero());
    graph.factors.emplace<Mean<dim>>(factor_id, Mean<dim>::Zero());
    graph.factors.emplace<Covariance<dim>>(factor_id, Covariance<dim>::Zero());

    return factor_id;
}

inline Eigen::Map<Eigen::VectorXd> eigen_vector_map(void* data, uint8_t dim) {
    return Eigen::Map<Eigen::VectorXd> {(double*)data, dim, 1};
}
inline Eigen::Map<Eigen::MatrixXd> eigen_matrix_map(void* data, uint8_t dim) {
    return Eigen::Map<Eigen::MatrixXd> {(double*)data, dim, dim};
}

template <int variable_dim>
void update_factors(FactorGraph& graph) {
    //std::cout << "updating to factor matrices================" << "\n";
    for (auto [_, connection, to_variable_matrix, to_factor_matrix] :
             graph.edges.view<VariableConnection, ToVariableMatrix<variable_dim>, ToFactorMatrix<variable_dim>>().each()) {
        auto& variable_info_matrix = graph.variables.get<InfoMatrix<variable_dim>>(connection.variable);
        to_factor_matrix = variable_info_matrix - to_variable_matrix;
        //std::cout << "\twrote to_factor_matrix\n" << to_factor_matrix << "\n";
    }

    for (auto [_, variable_connection, to_variable_vector, to_factor_vector] :
             graph.edges.view<VariableConnection, ToVariableVector<variable_dim>, ToFactorVector<variable_dim>>().each()) {
        auto& variable_info_vector = graph.variables.get<InfoVector<variable_dim>>(variable_connection.variable);
        to_factor_vector = variable_info_vector - to_variable_vector;
        //std::cout << "\twrote to_factor_vector\n" << to_factor_vector.transpose() << "\n";
    }

    for (auto [_, factor_connection, to_factor_matrix, to_factor_vector] :
             graph.edges.view<FactorConnection, ToFactorMatrix<variable_dim>, ToFactorVector<variable_dim>>().each()) {
        const uint8_t factor_dimension = factor_connection.factor_dimension;
        const FactorTypes factor_types = FACTOR_TYPES[factor_dimension];
        void* info_matrix_data = graph.factors.storage(factor_types.info_matrix_delta)->second.get(factor_connection.factor);
        void* info_vector_data = graph.factors.storage(factor_types.info_vector_delta)->second.get(factor_connection.factor);
        auto info_matrix = eigen_matrix_map(info_matrix_data, factor_dimension);
        auto info_vector = eigen_vector_map(info_vector_data, factor_dimension);

        //std::cout << "info matrix before\n" << info_matrix << "\n";        
        //std::cout << "info vector before " << info_vector.transpose() << "\n";        
        
        info_matrix.block<variable_dim, variable_dim>(factor_connection.factor_slot, factor_connection.factor_slot) = to_factor_matrix;
        info_vector.segment<variable_dim>(factor_connection.factor_slot) = to_factor_vector;

        //std::cout << "info matrix after\n" << info_matrix << "\n";        
        //std::cout << "info vector after " << info_vector.transpose() << "\n";        
    }
}

template <int variable_dim>
void update_factors_alt(FactorGraph& graph) {
    //std::cout << "updating to factor matrices================" << "\n";
    for (auto [_, connection, to_variable_matrix, to_factor_matrix] :
             graph.edges.view<VariableConnection, ToVariableMatrix<variable_dim>, ToFactorMatrix<variable_dim>>().each()) {
        auto& variable_info_matrix = graph.variables.get<InfoMatrix<variable_dim>>(connection.variable);
        to_factor_matrix = variable_info_matrix - to_variable_matrix;
        //std::cout << "\twrote to_factor_matrix\n" << to_factor_matrix << "\n";
    }

    for (auto [_, variable_connection, to_variable_vector, to_factor_vector] :
             graph.edges.view<VariableConnection, ToVariableVector<variable_dim>, ToFactorVector<variable_dim>>().each()) {
        auto& variable_info_vector = graph.variables.get<InfoVector<variable_dim>>(variable_connection.variable);
        to_factor_vector = variable_info_vector - to_variable_vector;
        //std::cout << "\twrote to_factor_vector\n" << to_factor_vector.transpose() << "\n";
    }

    for (auto [_, factor_connection, to_factor_matrix, to_factor_vector] :
             graph.edges.view<
             FactorConnection,
             ToFactorMatrix<variable_dim>,
             ToFactorVector<variable_dim>>().each()) {

        const uint8_t factor_dimension = factor_connection.factor_dimension;
        const FactorTypes factor_types = FACTOR_TYPES[factor_dimension];
        void* factor_cov_data = graph.factors.storage(factor_types.covariance)->second.get(factor_connection.factor);
        void* factor_mean_data = graph.factors.storage(factor_types.mean)->second.get(factor_connection.factor);
        auto factor_cov = eigen_matrix_map(factor_cov_data, factor_dimension);
        auto factor_mean = eigen_vector_map(factor_mean_data, factor_dimension);

        // rank update
        


    }
}


template <int dim>
void update_factors_finish(FactorGraph& graph) {
    //std::cout << "finishing factor update \n";

    for (auto [_, factor_info_matrix, factor_info_matrix_delta, covariance] :
             graph.factors.view<InfoMatrix<dim>, InfoMatrixDelta<dim>, Covariance<dim>>().each()) {

        //std::cout << "summing info matrix + info matrix delta \n";
        //std::cout << "info matrix\n" << factor_info_matrix << "\n";
        //std::cout << "info matrix delta\n" << factor_info_matrix_delta << "\n";

        covariance = factor_info_matrix + factor_info_matrix_delta;
    }

    for (auto [_, info_vector, info_vector_delta, mean] :
             graph.factors.view<InfoVector<dim>, InfoVectorDelta<dim>, Mean<dim>>().each()) {
        mean = info_vector + info_vector_delta;
    }

    for (auto [_, mean, covariance] : graph.factors.view<Mean<dim>, Covariance<dim>>().each()) {
        //std::cout << "info mat\n" << covariance << "\n";
        //std::cout << "info vec " << mean.transpose() << "\n";
        
        Mean<dim> m = covariance.ldlt().solve(mean);
        Covariance<dim> c = covariance.inverse();
        mean = m;
        covariance = c;

        //std::cout << "covariance " << covariance << "\n";        
        //std::cout << "mean " << mean.transpose() << "\n";        
    }
}

template <int dim>
void update_variables(FactorGraph& graph) {
    //std::cout << "updating variables of dim " << dim << "===================\n";

    for (auto [_, factor_connection, to_variable_matrix, to_variable_vector, to_factor_matrix, to_factor_vector] :
             graph.edges.view<
             FactorConnection,
             ToVariableMatrix<dim>, ToVariableVector<dim>,
             ToFactorMatrix<dim>, ToFactorVector<dim>>().each()) {
        const uint8_t factor_dimension = factor_connection.factor_dimension;
        const FactorTypes factor_types = FACTOR_TYPES[factor_dimension];
        void* covariance_data = graph.factors.storage(factor_types.covariance)->second.get(factor_connection.factor);
        void* mean_data = graph.factors.storage(factor_types.mean)->second.get(factor_connection.factor);
        auto covariance = eigen_matrix_map(covariance_data, factor_dimension);
        auto mean = eigen_vector_map(mean_data, factor_dimension);
        to_variable_matrix = covariance.block<dim, dim>(factor_connection.factor_slot, factor_connection.factor_slot);
        to_variable_vector = mean.segment<dim>(factor_connection.factor_slot);

        // subtract out the information in the factor that came from this edge
        const ToFactorMatrix<dim> info_mat_perturb = -to_factor_matrix;
        const ToFactorVector<dim> info_vec_perturb = -to_factor_vector;
        const Eigen::Matrix<double, dim, dim> fixer = info_marginalization_fixer(to_variable_matrix, info_mat_perturb);

        to_variable_matrix = perturb_marginalization<dim>(fixer, to_variable_matrix).eval();
        to_variable_vector = perturb_marginalization<dim>(fixer, to_variable_matrix, info_vec_perturb, to_variable_vector);
    }

    // convert the message to info form
    for (auto [_, to_variable_matrix, to_variable_vector] :
             graph.edges.view<ToVariableMatrix<dim>, ToVariableVector<dim>>().each()) {
        ToVariableVector<dim> info_vector = to_variable_matrix.ldlt().solve(to_variable_vector);
        ToVariableMatrix<dim> info_matrix = to_variable_matrix.inverse();
        to_variable_matrix = info_matrix;
        to_variable_vector = info_vector;

        //std::cout << "want to send info vector " << to_variable_vector.transpose() << "\n";
        //std::cout << "want to send info mat\n" << to_variable_matrix << "\n";
    }

    // clear the variable info deltas
    for (auto [_, matrix] : graph.variables.view<InfoMatrixDelta<dim>>().each()) {
        matrix.setZero();
    }

    for (auto [_, vector] : graph.variables.view<InfoVectorDelta<dim>>().each()) {
        vector.setZero();
    }

    for (auto [_, info_matrix, info_matrix_prior] :
             graph.variables.view<InfoMatrix<dim>, InfoMatrixPrior<dim>>().each()) {
        info_matrix = info_matrix_prior;
        //std::cout << "setting info matrix to prior\n" << info_matrix << "\n";

    }

    for (auto [_, info_vector, info_vector_prior] :
             graph.variables.view<InfoVector<dim>, InfoVectorPrior<dim>>().each()) {
        info_vector = info_vector_prior;
        //std::cout << "setting info vect to prior " << info_vector.transpose() << "\n";        
    }
    
    // update the variable info deltas
    for (auto [_, connection, to_variable_matrix] :
             graph.edges.view<VariableConnection, ToVariableMatrix<dim>>().each()) {
        graph.variables.get<InfoMatrixDelta<dim>>(connection.variable) += to_variable_matrix;
    }
    // update the variable info deltas
    for (auto [_, connection, to_variable_vector] :
             graph.edges.view<VariableConnection, ToVariableVector<dim>>().each()) {
        graph.variables.get<InfoVectorDelta<dim>>(connection.variable) += to_variable_vector;
    }

    // update the variable info matrices
    for (auto [_, info_matrix, info_matrix_delta] :
             graph.variables.view<InfoMatrix<dim>, InfoMatrixDelta<dim>>().each()) {
        //std::cout << "incrementing info matrix from\n" << info_matrix << "\n";

        info_matrix += info_matrix_delta;
        //std::cout << "to\n" << info_matrix << "\n";        
    }
    
    // update the variable info vectors
    for (auto [_, info_vector, info_vector_delta] :
             graph.variables.view<InfoVector<dim>, InfoVectorDelta<dim>>().each()) {
        //std::cout << "incrementing info vect from " << info_vector.transpose() << "\n";
        info_vector += info_vector_delta;
        //std::cout << "to\n" << info_vector.transpose() << "\n";        
    }

    // update the variable mean, covariances
    for (auto [_, covariance, info_matrix] : graph.variables.view<Covariance<dim>, InfoMatrix<dim>>().each()) {
        covariance = info_matrix;
    }
    for (auto [_, mean, info_vector] : graph.variables.view<Mean<dim>, InfoVector<dim>>().each()) {
        mean = info_vector;
    }
    for (auto [_, mean, covariance] : graph.variables.view<Mean<dim>, Covariance<dim>>().each()) {

        //std::cout << "putting var into mean cov form\n";
        //std::cout << "info vec is " << mean.transpose() << "\n";
        //std::cout << "info mat is\n" << covariance << "\n";


        const Mean<dim> m = covariance.ldlt().solve(mean);
        mean = m;
        const Covariance<dim> c = covariance.inverse();
        covariance = c;

        //std::cout << "mean vec is " << mean.transpose() << "\n";
        //std::cout << "cov mat is\n" << covariance << "\n";
        
    }
}

int main(int argc, char *argv[])
{
    FactorGraph graph;

    Mean<1> mean1;
    Mean<1> mean2;
    mean1 << 1;
    mean2 << -1;
    auto covariance1 = Covariance<1>::Identity().eval();
    auto covariance2 = Covariance<1>::Identity().eval();

    covariance1 *= 1e-2;
    covariance2 *= 1e-2;

    const auto variable1 = add_variable(mean1, covariance1, graph);
    const auto variable2 = add_variable(mean2, covariance2, graph);

    // connect these two with a factor
    Eigen::Matrix<double, 1, 2> J;
    J.block<1,1>(0,0).setIdentity();
    J.block<1,1>(0,1).setIdentity();
    J.block<1,1>(0,1) *= -1;

    InfoMatrix<2> info_mat = J.transpose() * J;
    InfoVector<2> info_vec = InfoVector<2>::Zero();

    const auto factor = add_factor(info_vec, info_mat, graph);

    add_edge<1>(variable1, factor, 0, graph);
    add_edge<1>(variable2, factor, info_vec.size()/2, graph);

    for (size_t i = 1; i < 10; ++i) {
        std::cout << "it " << i << "===============\n";
        update_factors<1>(graph);
        update_factors_finish<2>(graph);
        update_variables<1>(graph);
        // pull out the mean

        std::cout << "\tvar1 " << graph.variables.get<Mean<1>>(variable1).transpose() << "\n";
        std::cout << "\tvar2 " << graph.variables.get<Mean<1>>(variable2).transpose() << "\n";
        
    }

    // entt::id_type type = entt::type_hash<InfoMatrix<6>>::value();
    // auto storage = graph.variables.storage(type);
    // void* data = (storage->second).get(variable);
    // std::cout << *static_cast<InfoMatrix<6>*>(data) << std::endl;

    std::cout << "Hello\n";
    return 0;
}
