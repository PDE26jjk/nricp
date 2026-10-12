#pragma once

#include <cmath>

namespace nricp {

// Host-side vector type. The reference path is double precision throughout; the
// accelerated path uses its own single-precision type on the device.
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

// Triangle, as three template/vertex indices.
struct Int3 {
    int v0 = 0;
    int v1 = 0;
    int v2 = 0;
};

// Undirected index pair (deformation-graph edge).
struct Int2 {
    int a = 0;
    int b = 0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }

inline Vec3 operator*(const Vec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator*(double s, const Vec3& a) { return a * s; }

inline Vec3 operator/(const Vec3& a, double s) {
    return {a.x / s, a.y / s, a.z / s};
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline Vec3& operator-=(Vec3& a, const Vec3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double squared_norm(const Vec3& a) { return dot(a, a); }

inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

inline Vec3 normalized(const Vec3& a) {
    const double length = norm(a);
    return length > 0.0 ? a / length : Vec3{};
}

inline bool is_finite(const Vec3& a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

inline int index_of(const Int3& face, int corner) {
    return corner == 0 ? face.v0 : (corner == 1 ? face.v1 : face.v2);
}

}  // namespace nricp
