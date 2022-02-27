load("@rules_python//python:defs.bzl", "py_binary")

cc_library(
    name="factor_graph",
    srcs=["id.cpp",
          "variables.cpp",
          "edges.cpp"],
    hdrs=["id.h",
          "variables.h",
          "factors.h",
          "edge.h",
          "edges.h",
          "gaussian.h",
          "factor_update_schedule.h"],
    deps=[
        "@com_gitlab_libeigen_eigen//:eigen",
        "@com_github_markisus_vapid-soa//:soa",
    ],
)

cc_binary(
    name="scratch",
    srcs=["scratch.cpp"],
    deps=[
        ":factor_graph",
        "@com_gitlab_libeigen_eigen//:eigen",
        "@com_github_markisus_vapid-soa//:soa",
    ]
)

cc_library(
    name="engine",
    srcs=["engine.cpp"],
    hdrs=["engine.h"],
    deps=["@com_gitlab_libeigen_eigen//:eigen"])

cc_binary(
    name="scratch2",
    srcs=["scratch2.cpp",
          "id.cpp",
          "to_factor_msg_buffer.h",
          "to_factor_msg_buffer.cpp",
          "to_variable_msg_buffer.h",
          "to_variable_msg_buffer.cpp",          
          "id.h",
          "factor_graph.h",
          "factor_graph.cpp",
          "factor_array.cpp",
          "factor_array.h",
          "variable_array.cpp",
          "variable_array.h",
          "matrix_mem.h",
          "matrix_mem_eigen.h",
          "msg_buffers.h",
          "util.h"
          ],
    deps=[
        "@com_github_markisus_vapid-soa//:soa",
        "@com_gitlab_libeigen_eigen//:eigen",
    ]
)
    

py_binary(
    name="transpiler",
    srcs=["transpiler.py"])
