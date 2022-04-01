#pragma once
#include "Eigen/Dense"
#include "entt/fwd.hpp"
#include "util/eigen_util.h"

namespace carl {

const int MAX_AGE = 1000;

struct LayoutData {
    Eigen::VectorD<2> position = (Eigen::random_vec<2>()).eval();
    Eigen::VectorD<2> force;
};

struct FactorSummary {
    double offset = 0; // r.t r
    double delta = 0; // x.t H x - 2 r.t x
    double total() const {
        return offset + delta;
    }

    int age = MAX_AGE;
};

struct VariableSummary {
    double prior_error = 0;
    double factor_error = 0;
    double total() const {
        return factor_error + prior_error;
    }

    int age = MAX_AGE;
};

struct EdgeResidual {
    double to_factor = 1000;
    double to_variable = 1000;
};

}  // carl
