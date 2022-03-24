#include <type_traits>
#include <cassert>
#include <iostream>

#include "image.h"
#include "stb_image.h"
#include "sokol_gfx.h"

namespace carl {

// if these fail, then sokol library has changed its implementation of
// sg_image, which was previously just a wrapper for uint32
static_assert(std::is_same<decltype(sg_image::id), uint32_t>::value);
static_assert(sizeof(sg_image)== sizeof(uint32_t));

Image::Image(const std::string& path, bool make_texture) {
    load(path, make_texture);
}

void Image::load(const std::string& path, bool make_texture) {
    assert(texture == 0);
    assert(data == nullptr);
    
    int n = 0;
    data = stbi_load(path.c_str(), &width, &height, &n, 4);
    texture = 0;

    if (data && make_texture) {
        sg_image_desc img_desc {};
        img_desc.width = width;
        img_desc.height = height;
        img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
        img_desc.data.subimage[0][0].ptr = data;
        img_desc.data.subimage[0][0].size = width * height * 4;
        texture = sg_make_image(&img_desc).id;
    }
}

Image::~Image() {
    if (texture) {
        if (sg_isvalid() && sg_query_image_state(sg_image{texture})) {
            sg_destroy_image(sg_image{texture});
        }
    }
    if (data) {
        stbi_image_free(data);
    }
}

}  // carl
