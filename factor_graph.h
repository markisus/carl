#pragma once

#include "msg_buffers.h"
#include "variable_array.h"
#include "factor_array.h"

namespace carl {

class FactorGraph {
public:
    void update();
    Id add_camera_matrix(const Eigen::Matrix<double, 4, 1>& mean,
                         const double fx_fy_stddev,
                         const double cx_cy_stddev);

    MsgBuffersBase& msg_buffers_of_dimension(uint8_t dim);
    VariableArrayBase& vars_of_dimension(uint8_t dim);
    FactorArrayBase& factors_of_dimension(uint8_t dim);
    
private:
    MsgBuffers<4> msg4s;
    MsgBuffers<6> msg6s;

    VariableArray<4> var4s;
    VariableArray<6> var6s;

    FactorArray<4> factor4s;
    FactorArray<6> factor6s;
    FactorArray<16> factor16s;

    void execute(const ToVariableMsg_SendContext& ctx);
    void execute(const ToVariableMsg_RetrievalContext& ctx);
    void execute(const ToFactorMsg_RetrievalContext& ctx);
    void execute(const ToFactorMsg_SendContext& ctx);


};

}



