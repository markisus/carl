#pragma once

#include "msg_buffers.h"
#include "variable_array.h"
#include "factor_arrays.h"
#include <array>

namespace carl {

class FactorGraph {
public:
    void update();

    // template <uint8_t dim>
    // VariableId create_variable(const StaticVectorRef<dim>& mean,
    //                            const StaticMatrixRef<dim>& cov);
    // template <uint8_t dim>
    // FactorId create_factor(const StaticVectorRef<dim>& info_vec,
    //                        const StaticMatrixRef<dim>& info_mat);

    void add_edge(VariableId variable_id, FactorId factor_id, uint8_t slot);

    std::pair<VectorRef, MatrixRef> get_mean_cov(VariableId variable_id);

    void commit();

    MsgBuffersBase& msg_buffers_of_dimension(uint8_t dim);
    VariableArrayBase& vars_of_dimension(uint8_t dim);
    FactorArrays factors;

    static constexpr std::array<uint8_t, 2> variable_dims = {4, 6};
    
private:
    MsgBuffers<4> msg4s;
    MsgBuffers<6> msg6s;

    VariableArray<4> var4s;
    VariableArray<6> var6s;

    void execute(const ToVariableMsg_SendContext& ctx);
    void execute(const ToVariableMsg_RetrievalContext& ctx);
    void execute(const ToFactorMsg_RetrievalContext& ctx);
    void execute(const ToFactorMsg_SendContext& ctx);
};

}



