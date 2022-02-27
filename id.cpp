#include "id.h"

namespace carl {

Id id_ = 0;
Id ID() {
    return id_++;
}

Id NEW_VAR_ID() {
    return ID();
}

FactorId NEW_FACTOR_ID(uint8_t dim) {
    return std::make_pair(dim, ID());
}

}
