cc_library(
    name = "opencv",
    hdrs = glob(["include/opencv4/**/*"]),
    includes = ["include/opencv4"],
    srcs = [
        "lib/x86_64-linux-gnu/libopencv_core.so",
        "lib/x86_64-linux-gnu/libopencv_imgcodecs.so",
        "lib/x86_64-linux-gnu/libopencv_calib3d.so",
    ],
    visibility = ["//visibility:public"]
)
