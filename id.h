#pragma once

#include <cstdint>
#include <utility>

namespace carl {

using Id = uint32_t;
using FactorId = std::pair<uint8_t, Id>; // 1st component of pair is the dimension of the variable
using VariableId = std::pair<uint8_t, Id>; // 1st component of pair is the dimension of the factor

VariableId NEW_VARIABLE_ID(uint8_t dim);
FactorId NEW_FACTOR_ID(uint8_t dim);

}
