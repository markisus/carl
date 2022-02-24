#include <Eigen/Dense>
#include <iostream>
#include <cstdint>

#include "vapid/soa.h"

namespace carl {

using Id = uint32_t;

Id id_ = 0;

Id ID() {
    return id_++;
}

enum class VariableType {
    CameraMatrix,
    CameraPose,
    ObjectPose,
};

enum class FactorType {
    CameraMatrixPrior,
    CameraPosePrior,
    ObjectPosePrior,
    CameraObject,
};

template <uint8_t dim>
struct Gaussian {
    Eigen::Matrix<double, dim, dim> matrix;
    Eigen::Matrix<double, dim, 1> vector;

    static Gaussian<dim> random() {
        Gaussian<dim> result;
        result.matrix.setZero();
        result.vector.setZero();
        return result;
    }

    Gaussian<dim> operator-(const Gaussian<dim>& other) const {
        Gaussian<dim> result = *this;
        result.matrix -= other.matrix;
        result.vector -= other.vector;
        return result;
    }

    Gaussian<dim> operator+(const Gaussian<dim>& other) const {
        Gaussian<dim> result = *this;
        result.matrix += other.matrix;
        result.vector += other.vector;
        return result;
    }

    void operator+=(const Gaussian<dim>& other) {
        auto result = *this + other;
        *this = result;
    }

    void setZero() {
        this->matrix.setZero();
        this->vector.setZero();
    }

    Gaussian<dim> to_covariance() const {
        // assuming this is info form
        // returns the covariance form
        // TODO: fix this
        return *this;
    }

    Gaussian<dim> to_info() const {
        // assuming this is covariance form
        // returns the info form
        // TODO: fix this
        return *this;
    }    
};

template <uint8_t dim>
std::ostream& operator<<(std::ostream& cout,
                         const Gaussian<dim>& gaussian) {
    cout << "Gaussian<" << dim << ">";
    return cout;
}

struct Edge {
    Id variable_id;
    Id factor_id;

    uint8_t variable_dim;
    uint8_t factor_dim;

    // the variable slots into
    // [factor_slot, factor_slot + variable_dim)
    uint8_t factor_slot_begin;
};

std::ostream& operator<<(std::ostream& cout,
                         const Edge& edge) {
    cout << "Edge{var(" << edge.variable_id << ")<=>factor("
         << edge.factor_id <<")}";
    return cout;
}

struct VariableMeta {
    Id variable_id;
    VariableType type;
};

struct FactorMeta {
    Id factor_id;
    FactorType type;
};

size_t get_variable_msgs_end(const std::vector<Edge>& edges, size_t edge_idx) {
    if (edge_idx >= edges.size()) {
        return edge_idx;
    }
    
    Id variable_id = edges[edge_idx].variable_id;
    while (edges[edge_idx].variable_id == variable_id) {
        ++edge_idx;
    }
    return edge_idx;
}

struct VariableMsgSection {
    size_t variable_idx = 0; // idx into varns

    // idx into var{n}s_to_factors, sorted by var
    size_t msgs_start_idx = 0; 
    size_t msgs_end_idx = 0;
};

struct MsgCommit {
    size_t src_idx = 0;
    size_t dst_idx = 0;
};

struct FactorUpdateSchedule {
    std::vector<std::pair<size_t, size_t>> factor4_updates;
    std::vector<std::pair<size_t, size_t>> factor6_updates;
    std::vector<std::pair<size_t, size_t>> factor16_updates;

    const std::vector<std::pair<size_t, size_t>>& for_dimension(uint8_t dim) const {
        switch (dim) {
        case 4: {
            return factor4_updates;
        }
        case 6: {
            return factor6_updates;
        }
        case 16: {
            return factor16_updates;
        }
        default:
            std::cout << "Cannot get fud for " << dim << "\n";
            exit(-1);
        }
    }

    std::vector<std::pair<size_t, size_t>>& for_dimension(uint8_t dim) {
        const auto* const_this = this;
        return const_cast<std::vector<std::pair<size_t, size_t>>&>(const_this->for_dimension(dim));
    }

    void clear() {
        factor4_updates.clear();
        factor6_updates.clear();
        factor16_updates.clear();
    }
};

template <uint8_t factor_dim>
struct FactorArray;

class FactorArrayBase {
public:
    virtual const std::vector<FactorMeta>& get_metas() const = 0;
    virtual Id add_factor(FactorType factor_type) = 0;
    virtual void update_covariances() = 0;

    virtual void update(const FactorUpdateSchedule& update_schedule, 
                        const vapid::soa<Edge, Gaussian<4>>& msgs) = 0;
    virtual void update(const FactorUpdateSchedule& update_schedule, 
                        const vapid::soa<Edge, Gaussian<6>>& msgs) = 0;

    virtual void send_msgs_to_vars(const std::vector<Edge>& edges,
                                   vapid::soa<Edge, Gaussian<4>>& msgs_out) = 0;
    virtual void send_msgs_to_vars(const std::vector<Edge>& edges,
                                   vapid::soa<Edge, Gaussian<6>>& msgs_out) = 0;
};

template <uint8_t factor_dim>
struct FactorArray : public FactorArrayBase {
    // column order: meta, info form, cov form
    vapid::soa<FactorMeta, Gaussian<factor_dim>, Gaussian<factor_dim>> factors;

    const std::vector<FactorMeta>& get_metas() const override {
        return factors.template get_column<0>();
    }

    Id add_factor(FactorType factor_type) override {
        Id factor_id = ID();
        factors.insert(FactorMeta{factor_id, factor_type}, Gaussian<factor_dim>{}, Gaussian<factor_dim>{});
        return factor_id;
    }

    void update(const FactorUpdateSchedule& update_schedule, 
                const vapid::soa<Edge, Gaussian<4>>& msgs) override {
        update_impl(update_schedule, msgs);
    }

    void update(const FactorUpdateSchedule& update_schedule, 
                const vapid::soa<Edge, Gaussian<6>>& msgs) override {
        update_impl(update_schedule, msgs);

    }

    void update_covariances() override {
        for (size_t i = 0; i < factors.size(); ++i) {
            factors.template get_column<2>()[i] =
                factors.template get_column<1>()[i].to_covariance();
        }
    }

    void send_msgs_to_vars(const std::vector<Edge>& edges,
                           vapid::soa<Edge, Gaussian<4>>& msgs_out) override {
        send_msgs_to_vars_impl<4>(edges, msgs_out);
    }

    void send_msgs_to_vars(const std::vector<Edge>& edges,
                           vapid::soa<Edge, Gaussian<6>>& msgs_out) override {
        send_msgs_to_vars_impl<6>(edges, msgs_out);
    }

    template <uint8_t var_dim>
    void send_msgs_to_vars_impl(const std::vector<Edge>& edges,
                                vapid::soa<Edge, Gaussian<var_dim>>& msgs_out) {
        // edges must be sorted by factor id
        msgs_out.clear();
        msgs_out.reserve(edges.size());

        size_t write_head = 0;
        for (const auto& edge : edges) {
            bool factor_found = false;
            for (; write_head < factors.size(); ++write_head) {
                if (factors.template get_column<0>()[write_head].factor_id == edge.factor_id) {
                    factor_found = true;
                    break;
                }
            }
            if (!factor_found) {
                std::cout << "Could not find factor with id " << edge.factor_id << "\n";
            }
            
            Gaussian<factor_dim> covariance; // =?
            Gaussian<var_dim> msg_covariance;
            msg_covariance.matrix = covariance.matrix.template block<var_dim, var_dim>(
                edge.factor_slot_begin, edge.factor_slot_begin);
            msg_covariance.vector = covariance.vector.template segment<var_dim>(edge.factor_slot_begin);
            msgs_out.insert(edge, msg_covariance.to_info());
        }
    }

    template <uint8_t var_dim> 
    void update_impl(const FactorUpdateSchedule& update_schedule, 
                     const vapid::soa<Edge, Gaussian<var_dim>>& msgs) {
        for (auto [src_idx, dst_idx] : update_schedule.for_dimension(factor_dim)) {
            auto& dst_gaussian = factors.template get_column<1>()[dst_idx];
            auto [edge, src_gaussian] = msgs[src_idx];
            const size_t slot_begin = edge.factor_slot_begin;
            dst_gaussian.matrix.template block<var_dim, var_dim>(slot_begin, slot_begin) += src_gaussian.matrix;
            dst_gaussian.vector.template segment<var_dim>(slot_begin) += src_gaussian.vector;
        }


        // // compute info forms for gaussians that have been updated
        // size_t curr_idx = SIZE_MAX;
        // for (auto [_, update_idx] : update_schedule.for_dimension(factor_dim)) {
        //     if (update_idx == curr_idx) {
        //         // already handled this index
        //         continue;
        //     }
        //     curr_idx = update_idx;
        //     const auto& info_gaussian = factors.template get_column<1>()[update_idx];
        //     factors.template get_column<2>()[update_idx] = info_gaussian.to_covariance();
        // }
    }
};

struct Factors {
    FactorArray<4> factor4s;
    FactorArray<6> factor6s;
    FactorArray<16> factor16s;

    const FactorArrayBase& of_dimension(uint8_t factor_dim) const {
        switch (factor_dim) {
        case 4: {
            return factor4s;
        }
        case 6: {
            return factor6s;
        }
        case 16: {
            return factor16s;
        }
        default: {
            std::cout << "No factors of dim " << factor_dim << "\n";
            exit(-1);
        }
        }
    }

    FactorArrayBase& of_dimension(uint8_t factor_dim) {
        const auto* const_this = this;
        return const_cast<FactorArrayBase&>(const_this->of_dimension(factor_dim));
    }
};

size_t find_variable_idx(const std::vector<VariableMeta>& variables, Id var_id, size_t hint = 0) {
    size_t idx = hint;
    for (; idx < variables.size(); ++idx) {
        if (variables[idx].variable_id == var_id) {
            return idx;
        }
    }
    return size_t(-1);
}

size_t find_factor_idx(const std::vector<FactorMeta>& factors, Id factor_id, size_t hint = 0) {
    size_t idx = hint;
    for (; idx < factors.size(); ++idx) {
        if (factors[idx].factor_id == factor_id) {
            return idx;
        }
    }
    return size_t(-1);
}

void rebuild_factor_update_schedule(const Factors& factors,
                                    const std::vector<Edge>& edges,
                                    FactorUpdateSchedule& factor_update_schedule) {
    // edges need to be sorted by [factor_dim, factor_id]
    factor_update_schedule.clear();
    size_t read_head = 0;
    for (uint8_t factor_dim : { 4, 6, 16 }) {
        const auto& factor_metas = factors.of_dimension(factor_dim).get_metas();
        auto& update_schedule = factor_update_schedule.for_dimension(factor_dim);
        for (size_t write_head = 0; write_head < factor_metas.size(); ++write_head) {
            read_head = find_factor_idx(factor_metas, edges[read_head].factor_id, read_head);
            if (read_head == size_t(-1)) {
                std::cout << "Could not match factor " << edges[read_head].factor_id << "\n";
                exit(-1);
            }
            update_schedule.push_back({read_head, write_head});
        }
    }
}

void rebuild_variable_sections(
    const std::vector<Edge>& msgs_to_variables_edges,
    const std::vector<VariableMeta>& variable_metas,    
    std::vector<VariableMsgSection>& msg_sections) {
    msg_sections.clear();
    size_t msgs_start = 0;
    size_t var_idx = 0;
    while (msgs_start < msgs_to_variables_edges.size()) {
        // find the begin and end of a variable
        const size_t msgs_end = get_variable_msgs_end(msgs_to_variables_edges, msgs_start);
        const Id variable_id = msgs_to_variables_edges[msgs_start].variable_id;
        var_idx = find_variable_idx(variable_metas, variable_id, /*hint=*/var_idx);
        if (var_idx == size_t(-1)) {
            std::cout << "variable " << variable_id << " not found" << "\n";
            exit(-1);
        }
        msg_sections.push_back({var_idx, msgs_start, msgs_end});
        msgs_start = msgs_end;
    }
}

void rebuild_commits(
    const std::vector<Edge>& edges,
    const std::vector<Edge>& edges_tmp,
    std::vector<MsgCommit>& commits) {
    commits.clear();
    commits.reserve(edges_tmp.size());
    size_t write_head = 0;
    for (size_t read_head = 0; read_head < edges_tmp.size(); ++read_head) {
        const auto& src_edge = edges_tmp[read_head];
        bool match_found = false;
        for (; write_head < edges.size(); ++write_head) {
            const auto& dst_edge = edges[write_head];
            if (dst_edge.factor_id == src_edge.factor_id &&
                dst_edge.factor_slot_begin == src_edge.factor_slot_begin) {
                match_found = true;
                break;
            }
        }
        if (!match_found) {
            std::cout << "no match found for factor id " << src_edge.factor_id << " slot " << int(src_edge.factor_slot_begin) << "\n";
            // std::cout << "read_head is " << read_head << "\n";
            // for (const auto& edge : edges) {
            //     std::cout << "\tfactor id " << edge.factor_id << " slot " << int(edge.factor_slot_begin) << "\n";
            // }
            exit(-1);
        }
        commits.push_back({ read_head, write_head });
    }
}

template <uint8_t dim>
void commit_msgs(
    const std::vector<MsgCommit>& commits,
    const std::vector<Gaussian<dim>>& src,
    std::vector<Gaussian<dim>>& dst) {
    for (const auto& commit : commits) {
        dst[commit.dst_idx] = src[commit.src_idx];
    }
}

class Variables;

class EdgeArrayBase {
public:
    virtual void update_variables(Variables& variables) = 0;
    virtual void update_factors(Factors& variables) = 0;
    virtual void add_edge(Id variable_id, Id factor_id, uint8_t factor_dim, uint8_t slot) = 0;
    virtual void sort_to_vars_msgs() = 0;
    virtual void sort_to_factors_msgs() = 0;
};

struct Variables {
    vapid::soa<VariableMeta, Gaussian<6>> var6s;
    vapid::soa<VariableMeta, Gaussian<4>> var4s;

    std::vector<VariableMeta>& metas(uint8_t dim) {
        switch (dim) {
        case 4: {
            return var4s.get_column<0>();
        }
        case 6: {
            return var6s.get_column<0>();
        }
        default:
            std::cout << "No vars of dim " << dim << "\n";
            exit(-1);
        }
    }

    vapid::soa<VariableMeta, Gaussian<6>>& of_dimension(std::integral_constant<uint8_t, 6>) {
        return var6s;
    }
    vapid::soa<VariableMeta, Gaussian<4>>& of_dimension(std::integral_constant<uint8_t, 4>) {
        return var4s;
    }

    template <uint8_t variable_dim>
    vapid::soa<VariableMeta, Gaussian<variable_dim>>& of_dimension() {
        return of_dimension(std::integral_constant<uint8_t, variable_dim>{});
    }
};

template <uint8_t var_dim>
struct EdgeArray : public EdgeArrayBase {
    // keep sorted by variable id
    vapid::soa<Edge, Gaussian<var_dim>> to_vars_msgs;
    vapid::soa<Edge, Gaussian<var_dim>> to_vars_msgs_temp;
    std::vector<MsgCommit> to_vars_commits;

    // keep sorted by [factor_var_dim, factor_id]
    vapid::soa<Edge, Gaussian<var_dim>> to_factors_msgs;
    vapid::soa<Edge, Gaussian<var_dim>> to_factors_msgs_temp;
    std::vector<MsgCommit> to_factors_commits;

    std::vector<VariableMsgSection> var_msg_sections;
    FactorUpdateSchedule factor_update_schedule;

    void add_edge(Id variable_id, Id factor_id, uint8_t factor_dim, uint8_t slot) override {
        to_factors_msgs.insert(Edge{variable_id, factor_id, var_dim, factor_dim, slot}, Gaussian<var_dim>{});
        to_vars_msgs.insert(Edge{variable_id, factor_id, var_dim, factor_dim, slot}, Gaussian<var_dim>{});
    }

    static bool to_variable_order (const Edge& e1, const Edge& e2) {
        return e1.variable_id < e2.variable_id;
    };

    static bool to_factor_order(const Edge& e1, const Edge& e2) {
        return std::make_tuple(e1.factor_dim, e1.factor_id, e1.factor_slot_begin) <
            std::make_tuple(e2.factor_dim, e2.factor_id, e2.factor_slot_begin);
    };

    void sort_to_vars_msgs() override {
        to_vars_msgs.template sort_by_field<0>(to_variable_order);
    }

    void sort_to_factors_msgs() override {
        to_factors_msgs.template sort_by_field<0>(to_factor_order);
    }

    void update_variables(Variables& variables) override {
        rebuild_variable_sections(
            to_vars_msgs.template get_column<0>(),
            variables.metas(var_dim),
            var_msg_sections);

        // sum all the messages pertaining to a particular variable into the target
        size_t num_messages = 0;
        
        for (auto [var_idx, msgs_start, msgs_end] : var_msg_sections) {
            auto& dst = variables.of_dimension<var_dim>().template get_column<1>()[var_idx];
            dst.setZero();
            for (size_t msg_idx = msgs_start; msg_idx < msgs_end; ++msg_idx) {
                const auto& src = to_vars_msgs.template get_column<1>()[msg_idx];
                dst += src;
                ++num_messages;
            }
        }

        // send out the factor messages
        // [DANGER!] Manual editing of the buffers inside a soa
        //         will break if another column is added
        to_factors_msgs_temp.clear();
        auto& edges_out = to_factors_msgs_temp.template get_column<0>();
        auto& gaussians_out = to_factors_msgs_temp.template get_column<1>();
        edges_out.reserve(num_messages);
        gaussians_out.reserve(num_messages);

        // handle edges
        for (auto [_, msgs_start, msgs_end] : var_msg_sections) {
            for (size_t msg_idx = msgs_start; msg_idx < msgs_end; ++msg_idx) {
                edges_out.push_back(to_vars_msgs.template get_column<0>()[msg_idx]);
            }
        }

        // handle gaussians
        for (auto [var_idx, msgs_start, msgs_end] : var_msg_sections) {
            const auto& sum_gaussian = variables.of_dimension<var_dim>().template get_column<1>()[var_idx];
            for (size_t msg_idx = msgs_start; msg_idx < msgs_end; ++msg_idx) {
                const auto& gaussian_in = to_vars_msgs.template get_column<1>()[msg_idx];
                Gaussian<var_dim> gaussian_out = sum_gaussian - gaussian_in;
                gaussians_out.push_back(gaussian_out);
            }
        }

        to_factors_msgs_temp.template sort_by_field<0>(to_factor_order);
        rebuild_commits(
            to_factors_msgs.template get_column<0>(),
            to_factors_msgs_temp.template get_column<0>(),
            to_factors_commits);
        commit_msgs<var_dim>(
            to_factors_commits,
            to_factors_msgs_temp.template get_column<1>(),
            to_factors_msgs.template get_column<1>());
    };

    void update_factors(Factors& factors) override {
        rebuild_factor_update_schedule(
            factors,
            to_factors_msgs.template get_column<0>(),
            factor_update_schedule);

        for (uint8_t factor_dim : {4, 6, 16}) {
            factors.of_dimension(factor_dim).update(factor_update_schedule,
                                                    to_factors_msgs);
        }
    }
};

struct Edges {
    EdgeArray<4> edges4;
    EdgeArray<6> edges6;

    EdgeArrayBase& of_dimension(uint8_t dim) {
        switch (dim) {
        case 4: {
            return edges4;
        }
        case 6: {
            return edges6;
        }
        default: {
            std::cout << "No edge collection of dim " << dim << "\n";
            exit(-1);
        }
        }
    }
};

struct FactorGraph {
    Variables vars;
    Factors factors;
    Edges edges;
    
    Id add_camera_matrix_variable() {
        Id var_id = ID();
        vars.of_dimension<4>().insert(VariableMeta{var_id, VariableType::CameraMatrix}, Gaussian<4>{});
        return var_id;
    }
        
    Id add_camera_pose_variable() {
        Id var_id = ID();
        vars.of_dimension<6>().insert(VariableMeta{var_id, VariableType::CameraPose}, Gaussian<6>{});
        return var_id;
    }

    Id add_object_pose_variable() {
        Id var_id = ID();
        vars.of_dimension<6>().insert(VariableMeta{var_id, VariableType::ObjectPose}, Gaussian<6>{});
        return var_id;
    }

    Id add_camera_object_factor(Id camera_matrix, Id camera_pose, Id object_pose) {
        Id factor_id = factors.of_dimension(16).add_factor(FactorType::CameraObject);
        edges.of_dimension(4).add_edge(camera_matrix, factor_id, 16, 0);
        edges.of_dimension(6).add_edge(camera_pose, factor_id, 16, 4);
        edges.of_dimension(6).add_edge(object_pose, factor_id, 16, 10);
        return factor_id;
    }

    Id add_camera_pose_prior(Id camera_pose) {
        Id factor_id = factors.of_dimension(6).add_factor(FactorType::CameraPosePrior);
        edges.of_dimension(6).add_edge(camera_pose, factor_id, 6, 0);
        return factor_id;
    }

    Id add_camera_matrix_prior(Id camera_matrix) {
        Id factor_id = factors.of_dimension(4).add_factor(FactorType::CameraMatrixPrior);
        edges.of_dimension(4).add_edge(camera_matrix, factor_id, 4, 0);
        return factor_id;
    }

    Id add_object_pose_prior(Id object_pose) {
        Id factor_id = factors.of_dimension(6).add_factor(FactorType::ObjectPosePrior);
        edges.of_dimension(6).add_edge(object_pose, factor_id, 6, 0);
        return factor_id;
    }

    void sort() {
        for (uint8_t var_dim : { 4, 6}) {
            edges.of_dimension(var_dim).sort_to_vars_msgs();
            edges.of_dimension(var_dim).sort_to_factors_msgs();
        }
    }

    void send_msgs_to_factors() {
        for (uint8_t var_dim : { 4, 6 }) {
            edges.of_dimension(var_dim).update_factors(factors);
        }
    }

    void update_variables() {
        for (uint8_t var_dim : { 4, 6 }) {
            edges.of_dimension(var_dim).update_variables(vars);
        }
    }
};

}

using namespace carl;

int main(int argc, char *argv[])
{
    std::cout << "Hello world!" << std::endl;

    FactorGraph graph;

    Id cam_matrix = graph.add_camera_matrix_variable();
    graph.add_camera_matrix_prior(cam_matrix);

    Id obj1 = graph.add_object_pose_variable();
    graph.add_object_pose_prior(obj1);
    
    Id obj2 = graph.add_object_pose_variable();
    graph.add_object_pose_prior(obj2);

    Id obj3 = graph.add_object_pose_variable();
    graph.add_object_pose_prior(obj3);

    Id cam1_id = graph.add_camera_pose_variable();
    graph.add_camera_pose_prior(cam1_id);
    
    Id cam2_id = graph.add_camera_pose_variable();
    graph.add_camera_pose_prior(cam2_id);

    Id cam3_id = graph.add_camera_pose_variable();
    graph.add_camera_pose_prior(cam3_id);

    graph.add_camera_object_factor(cam_matrix, cam1_id, obj1);
    graph.add_camera_object_factor(cam_matrix, cam1_id, obj2);

    graph.add_camera_object_factor(cam_matrix, cam2_id, obj2);
    graph.add_camera_object_factor(cam_matrix, cam2_id, obj3);

    graph.add_camera_object_factor(cam_matrix, cam3_id, obj1);
    graph.add_camera_object_factor(cam_matrix, cam3_id, obj3);

    graph.sort();
    graph.update_variables();

    // graph.update_variable_sections();
    // graph.update_variables();
    // graph.update_msgs_to_factors();
    // graph.commit_to_factor_msgs();
    // add msgs into the factor
    
    return 0;
}
