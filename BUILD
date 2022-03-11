load("@rules_python//python:defs.bzl", "py_binary")

cc_library(
    name="eigen_util",
    hdrs=["eigen_util.h"],
    deps=[
        "@com_gitlab_libeigen_eigen//:eigen",
    ])

cc_library(
    name="geometry",
    srcs=["geometry.cpp"],
    hdrs=["geometry.h"],
    deps=[
        ":eigen_util",
        "@com_gitlab_libeigen_eigen//:eigen",
    ])

cc_binary(
    name="geometry_test",
    srcs=["geometry_test.cpp"],
    deps=[
        ":geometry",
        ":eigen_util",
        "@com_gitlab_libeigen_eigen//:eigen",
        "@com_github_google_googletest//:gtest_main"]
)

cc_library(
    name = "gaussian_math",
    srcs = ["gaussian_math.hpp"],
    hdrs = ["gaussian_math.hpp"],
    deps = [
        "@com_gitlab_libeigen_eigen//:eigen",
    ])

cc_binary(
    name="math_test",
    srcs=["math_test.cpp"],
    deps=[
        ":gaussian_math",
        "@com_github_google_googletest//:gtest_main"]
)

cc_library(
    name="factor_graph",
    srcs=["factor_graph.hpp"],
    hdrs=["factor_graph.hpp"],
    deps=[
        "@com_github_skypjack_entt//:entt",
        "@com_gitlab_libeigen_eigen//:eigen",
        ":gaussian_math",
        ":eigen_util",
    ])

cc_binary(
    name="scratch3",
    srcs=[
        "scratch3.cpp",
    ],
    deps=[
        ":factor_graph",
        "@com_github_skypjack_entt//:entt",
        "@com_gitlab_libeigen_eigen//:eigen",
    ])    
    
cc_binary(
    name="math_benchmark",
    srcs=[
        "math_benchmark.cpp",
    ],
    deps=[
        ":gaussian_math",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)    

cc_binary(
    name="simulation_a",
    srcs=[
        "simulation_a.cpp",
    ],
    deps=[
        ":factor_graph",
        ":eigen_util",
        ":geometry",
        "@com_github_skypjack_entt//:entt",
        "@com_gitlab_libeigen_eigen//:eigen",
    ])    
    

py_binary(
    name="transpiler",
    srcs=["transpiler.py"])
