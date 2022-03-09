workspace(name = "carl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
rules_python_version = "740825b7f74930c62f44af95c9a4c1bd428d2c53" # Latest @ 2021-06-23

http_archive(
    name = "rules_python",
    strip_prefix = "rules_python-{}".format(rules_python_version),
    url = "https://github.com/bazelbuild/rules_python/archive/{}.zip".format(rules_python_version),
)

vapid_soa_version = "0dbb4cad5cd0a7840c4ad9825236b2024e546f0a"
http_archive(
    name = "com_github_markisus_vapid-soa",
    url = "https://github.com/markisus/vapid-soa/archive/{}.zip".format(vapid_soa_version),
    sha256 = "a8641c84ebacd02f4f238f5ab03f40fde92365ecccc85f629c15fa44b194c9db",
    strip_prefix = "vapid-soa-{}".format(vapid_soa_version))


http_archive(
    name = "com_github_skypjack_entt",
    url = "https://github.com/skypjack/entt/archive/3328c7e78bcf638a78d7a601d3780a10e7dd712c.zip",
    sha256 = "b182fb2625d0da580f27b86bfe07fed0e92154f213b48223d225587e63dd5ea8",
    strip_prefix = "entt-3328c7e78bcf638a78d7a601d3780a10e7dd712c")

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

load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
   name = "python_deps",
   requirements = "//:requirements.txt",
)
