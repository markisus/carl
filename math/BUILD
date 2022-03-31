cc_library(
    name="geometry",
    srcs=["geometry.cpp"],
    hdrs=["geometry.h"],
    deps=[
        "//util:eigen_util",
        "@com_gitlab_libeigen_eigen//:eigen",
    ],
    visibility=['//visibility:public'],
)

cc_binary(
    name="geometry_test",
    srcs=["geometry_test.cpp"],
    deps=[
        ":geometry",
        "//util:eigen_util",
        "@com_gitlab_libeigen_eigen//:eigen",
        "@com_github_google_googletest//:gtest_main"]
)

cc_library(
    name = "gaussian_math",
    srcs = ["gaussian_math.hpp"],
    hdrs = ["gaussian_math.hpp"],
    deps = [
        "@com_gitlab_libeigen_eigen//:eigen",
    ],
    visibility=['//visibility:public'],
)

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
