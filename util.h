#pragma once
#include <vector>
#include <iostream>

namespace carl {

template <typename T>
T* non_const(const T* t) {
    return const_cast<T*>(t);
}

// increment void pointer as if
// it points to an object of `size` bytes
inline void* next_ptr(size_t size, void* p) {
    void* result = ((uint8_t*)p) + size;
    return result;
}

template <typename TId>
inline size_t rebuild_index(
    const std::vector<TId>& referred_ids,
    const std::vector<TId>& referring_ids,
    std::vector<size_t>& index,
    size_t referring_start = 0) {
    size_t referred_idx = 0;
    for (size_t i = referring_start; i < referring_ids.size(); ++i) {
        const auto id = referring_ids[i];
        bool match_found = false;
        for (; referred_idx < referred_ids.size(); ++referred_idx) {
            if (referred_ids[referred_idx] == id) {
                match_found = true;
                break;
            }
        }
        if (!match_found) {
            return i;
        }
        index[i] = referred_idx;
    }
    return referring_ids.size();
}

inline void write_identity_permutation(std::vector<size_t>& perm) {
    for (size_t i = 0; i < perm.size(); ++i) {
        perm[i] = i;
    }
}

}
