cc_library(
    name="eigen_util",
    hdrs=["eigen_util.h"],
    deps=[
        "@com_gitlab_libeigen_eigen//:eigen",
    ],
    visibility=["//visibility:public"],
)

cc_library(
    name="image",
    srcs=["image.cpp", "image.h"],
    deps=[
        "@com_github_nothings_stb//:stb",
        "@com_github_floooh_sokol//:sokol"],
    visibility=["//visibility:public"],    
)

cc_library(
    name="thread_pool",
    srcs=["thread_pool.cpp",
          "thread_pool.h"],
    hdrs=["thread_pool.h"],
    visibility=["//visibility:public"],
)

cc_library(
    name="imgui_overlayable",
    hdrs=["imgui_overlayable.h"],
    deps=["@com_github_ocornut_imgui//:imgui"],
    visibility=["//visibility:public"],
)

