#include <string>
#include <cstdint>

namespace carl {

class Image {
public:
    Image(const std::string& path, bool make_texture = false);
    Image() = default;

    void load(const std::string& path, bool make_texture = false);
    void make_texture(); // needs load() to be called previously

    ~Image();

    int width = 0;
    int height = 0;

    uint8_t* data = nullptr;
    uint32_t texture = 0; // sokol handle
};

}  // carl
