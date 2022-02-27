#pragma once

#include <array>
#include <cstdint>

#include "util.h"

namespace carl {

template <uint8_t dim>
struct MatrixMem : std::array<double, dim*dim> {
    uint8_t dimension() { return dim; }
};

template <uint8_t dim>
struct VectorMem : std::array<double, dim> {
    uint8_t dimension() { return dim; }
};

struct MatrixRef {
    MatrixRef(uint8_t d, double* b) : dimension(d), begin(b) {}
    
    uint8_t dimension;
    double* begin;
};

template <uint8_t dim>
struct StaticMatrixRef : public MatrixRef {
    StaticMatrixRef(double* b) : MatrixRef(dim, b) {}

    const MatrixMem<dim>& operator*() const {
        return *((const MatrixMem<dim>*)(begin));
    }

    MatrixMem<dim>& operator*() {
        const auto* const_this = this;
        const_cast<MatrixMem<dim>&>(const_this->operator*());
    }    
};

struct VectorRef {
    VectorRef(uint8_t d, double* b) : dimension(d), begin(b) {}

    uint8_t dimension;
    double* begin;
};

template <uint8_t dim>
struct StaticVectorRef : public VectorRef {
    StaticVectorRef(double* b) : VectorRef(dim, b) {}

    const VectorMem<dim>& operator*() const {
        return *((const VectorMem<dim>*)(begin));
    }

    VectorMem<dim>& operator*() {
        const auto* const_this = this;
        const_cast<VectorMem<dim>&>(const_this->operator*());
    }
};

template <uint8_t dim>
StaticMatrixRef<dim> static_dim(const MatrixRef& ref) {
    return { ref.begin };
}

template <uint8_t dim>
StaticVectorRef<dim> static_dim(const VectorRef& ref) {
    return { ref.begin };
}

struct MatrixRefs {
    MatrixRefs(uint8_t d, size_t s, double* b) : dimension(d), size(s), begin(b) {}
    MatrixRefs() {}
    
    uint8_t dimension;
    size_t size;
    double* begin;

    double* raw(size_t idx) const {
        return begin + dimension*dimension*idx;
    }

    MatrixRef operator[](size_t idx) const {
        return MatrixRef(dimension, raw(idx));
    }
};

template <uint8_t dim>
struct StaticMatrixRefs : public MatrixRefs {
    StaticMatrixRefs(size_t s, double* b) : MatrixRefs(dim, s, b) {};

    StaticMatrixRef<dim> operator[](size_t idx) const {
        return static_dim<dim>(MatrixRefs::operator[](idx));
    }
};

template <uint8_t dim>
StaticMatrixRefs<dim> static_dim(const MatrixRefs& mems) {
    return { mems.size, mems.begin };
}

struct VectorRefs {
    VectorRefs(uint8_t d, size_t s, double* b) : dimension(d), size(s), begin(b) {}
    VectorRefs() {};
    
    uint8_t dimension;
    size_t size;
    double* begin;

    double* raw(size_t idx) const {
        return begin + dimension*idx;
    }

    VectorRef operator[](size_t idx) const {
        return VectorRef(dimension, raw(idx));
    }    
};

template <uint8_t dim>
struct StaticVectorRefs : public VectorRefs {
    StaticVectorRefs(size_t s, double* b) : VectorRefs(dim, s, b) {};

    StaticVectorRef<dim> operator[](size_t idx) const {
        return static_dim<dim>(VectorRefs::operator[](idx));
    }
};

template <uint8_t dim>
StaticVectorRefs<dim> static_dim(const VectorRefs& mems) {
    return { mems.size, mems.begin };
}

template <uint8_t dim>
StaticMatrixRefs<dim> get_refs(const std::vector<MatrixMem<dim>>& buff) {
    return { buff.size(), (double*)non_const(buff.data()) };
}

template <uint8_t dim>
StaticVectorRefs<dim> get_refs(const std::vector<VectorMem<dim>>& buff) {
    return { buff.size(), (double*)non_const(buff.data()) };
}



}  // carl
