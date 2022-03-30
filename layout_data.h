#pragma once
#include "Eigen/Dense"
#include "eigen_util.h"
#include <string>

namespace carl {

const int MAX_AGE = 1000;

struct LayoutData {
    Eigen::VectorD<2> position = (Eigen::random_vec<2>()).eval();
    Eigen::VectorD<2> force;
};

struct FactorError {
    double offset = 0; // r.t r
    double delta = 0; // x.t H x - 2 r.t x
    double change = 0; // change from last iteration
    double total() {
        return offset + delta;
    }

    int age = MAX_AGE;

    std::string display_string;
};

struct VariableError {
    double prior_error = 0;
    double factor_error = 0;
    double change = 0;
    double total() {
        return factor_error + prior_error;
    }

    int age = MAX_AGE;
    std::string display_string;    
};

struct EdgeResidual {
    double to_factor = 1000;
    double to_variable = 1000;
};


}  // carl
