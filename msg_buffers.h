#pragma once

#include "to_factor_msg_buffer.h"
#include "to_variable_msg_buffer.h"

namespace carl {

class MsgBuffersBase {
public:
    virtual uint8_t get_dimension() const = 0;

    virtual ToVariableMsgBufferBase& to_variable_msgs() = 0;
    virtual ToFactorMsgBufferBase& to_factor_msgs() = 0;

    void add_edge(Id variable_id, FactorId factor_id, uint8_t factor_slot) {
        to_variable_msgs().add_edge(variable_id, factor_id, factor_slot);
        to_factor_msgs().add_edge(variable_id, factor_id, factor_slot);
    };

    void commit_edges(const std::vector<Id>& external_variable_ids,
                      const std::vector<FactorId>& external_factor_ids) {
        to_variable_msgs().commit_edges(external_variable_ids);
        to_factor_msgs().commit_edges(external_factor_ids);
    };
};

template <uint8_t dim>
struct MsgBuffers : public MsgBuffersBase {
    ToFactorMsgBuffer<dim> to_factors;
    ToVariableMsgBuffer<dim> to_vars;

    ToVariableMsgBufferBase& to_variable_msgs() override {
        return to_vars;
    }

    ToFactorMsgBufferBase& to_factor_msgs() override {
        return to_factors;
    }

    uint8_t get_dimension() const override {
        return dim;
    }
};



}  // carl
