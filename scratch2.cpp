#include <iostream>
#include <Eigen/Dense>
#include "vapid/soa.h"
#include "to_factor_msg_buffer.h"
#include "to_variable_msg_buffer.h"
#include "factor_array.h"
#include "matrix_mem_eigen.h"
#include "variable_array.h"
#include "util.h"
#include "msg_buffers.h"
#include "factor_graph.h"

using namespace carl;



int main(int argc, char *argv[])
{
    std::cout << "Hello world" << "\n";

    std::vector<std::string> strings;
    strings.push_back("A");
    strings.push_back("B");

    std::string* x = strings.data();
    std::cout << "x " << x << "\n";
    std::cout << "x+1 " << x+1 << "\n";
    std::cout << "next_ptr(x) " << next_ptr(sizeof(std::string), x) << "\n";

    FactorGraph graph;
    graph.update();

    return 0;
}
