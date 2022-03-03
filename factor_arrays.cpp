#include "factor_arrays.h"

namespace carl {

const FactorArrayBase& FactorArrays::of_dimension(uint8_t dim) const {
    switch (dim) {
    case 4: {
        return factor4s;
    }
    case 6: {
        return factor6s;
    }
    case 12: {
        return factor12s;
    }
    case 16: {
        return factor16s;
    }
    default:
        std::cout << "No factors of dim " << dim << "\n";
        exit(-1);
    }
}

FactorArrayBase& FactorArrays::of_dimension(uint8_t dim) {
    const auto* const_this = this;
    return const_cast<FactorArrayBase&>(const_this->of_dimension(dim));
}

}  
