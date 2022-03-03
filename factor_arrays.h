#pragma once

#include "factor_array.h"

namespace carl {

class FactorArrays {
public:    
    FactorArrayBase& of_dimension(uint8_t i);
    const FactorArrayBase& of_dimension(uint8_t i) const;

    static constexpr std::initializer_list<uint8_t> factor_dims = { 4, 6, 12, 16 };
    FactorArray<4> factor4s;
    FactorArray<6> factor6s;
    FactorArray<12> factor12s;
    FactorArray<16> factor16s;
};

}  // carl
