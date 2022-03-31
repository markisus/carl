workspace(name = "carl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_github_skypjack_entt",
    url = "https://github.com/skypjack/entt/archive/3328c7e78bcf638a78d7a601d3780a10e7dd712c.zip",
    sha256 = "b182fb2625d0da580f27b86bfe07fed0e92154f213b48223d225587e63dd5ea8",
    strip_prefix = "entt-3328c7e78bcf638a78d7a601d3780a10e7dd712c")

http_archive(
    name = "com_github_floooh_sokol",
    url = "https://github.com/floooh/sokol/archive/a18031313cdf7dab461baa296c38b07337ab4d47.zip",
    strip_prefix = "sokol-a18031313cdf7dab461baa296c38b07337ab4d47",
    sha256 = "ce0fec4696bfe68f7ab642df679041bab8253e33c95add52765614b6b236dc45",
    build_file_content = """
cc_library(
    name = 'sokol',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['*.h']) + ['util/sokol_imgui.h'],
    linkopts = ['-lGL', '-lX11', '-ldl', '-lXcursor', '-lXi'],
    visibility = ['//visibility:public'],
    include_prefix = 'sokol'
)
"""
)

stb_version = "af1a5bc352164740c1cc1354942b1c6b72eacb8a"
http_archive(
    name = "com_github_nothings_stb",
    url = "https://github.com/nothings/stb/archive/{}.zip".format(stb_version),
    strip_prefix = "stb-{}".format(stb_version),
    sha256 = "e3d0edbecd356506d3d69b87419de2f9d180a98099134c6343177885f6c2cbef",
    build_file_content = """
cc_library(
    name = 'stb',
    hdrs = ['stb_image.h'], # add others if necessary
    visibility = ['//visibility:public'],
)
"""
)

http_archive(
    name = "com_github_ocornut_imgui",
    url = "https://github.com/ocornut/imgui/archive/refs/tags/v1.87.zip",
    strip_prefix = "imgui-1.87",
    build_file_content = """
cc_library(
    name = 'imgui',
    srcs = glob(['*.cpp', '*.h']),
    hdrs = glob(['*.h']),
    visibility = ['//visibility:public'],    
)
""")

gtest_version = "c9461a9b55ba954df0489bab6420eb297bed846b"
http_archive(
    name = "com_github_google_googletest",
    url = "https://github.com/google/googletest/archive/{}.zip".format(gtest_version),
    strip_prefix = "googletest-{}".format(gtest_version))

http_archive(
    name = "com_github_google_benchmark",
    url = "https://github.com/google/benchmark/archive/refs/tags/v1.6.1.zip",
    strip_prefix = "benchmark-1.6.1")

# rules for eigen
# adapated from https://ceres-solver.googlesource.com/ceres-solver/+/master/WORKSPACE
http_archive(
        name = "com_gitlab_libeigen_eigen",
        strip_prefix = "eigen-3.4",
        url = "https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.zip",
        sha256 = "e55ce8d04938171ab11e8189d7d085c5c215736e9d84839658b6b350d11383fd",
        build_file_content = """
cc_library(
    name = 'eigen',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**', 'unsupported/**']),
    visibility = ['//visibility:public'],
)
"""
)

# Remap system libraries (e.g. installed with apt)
new_local_repository(
  name = "usr",
  path = "/usr",
  build_file = "usr.BUILD",
)

http_archive(
  name = "com_google_absl",
  urls = ["https://github.com/abseil/abseil-cpp/archive/98eb410c93ad059f9bba1bf43f5bb916fc92a5ea.zip"],
  strip_prefix = "abseil-cpp-98eb410c93ad059f9bba1bf43f5bb916fc92a5ea",
  sha256 = "aabf6c57e3834f8dc3873a927f37eaf69975d4b28117fc7427dfb1c661542a87"
)
