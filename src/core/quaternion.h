#pragma once

#include <cmath>

#include "core/mat3.h"
#include "core/vec3.h"

namespace nricp {

// Unit quaternion, used wherever rotations are blended or composed without
// going through a matrix decomposition.
struct Quaternion {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline double dot(const Quaternion& a, const Quaternion& b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Quaternion operator+(const Quaternion& a, const Quaternion& b) {
    return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Quaternion operator*(const Quaternion& a, double scale) {
    return {a.w * scale, a.x * scale, a.y * scale, a.z * scale};
}

inline Quaternion& operator+=(Quaternion& a, const Quaternion& b) {
    a.w += b.w;
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline Quaternion normalized(const Quaternion& q) {
    const double length = std::sqrt(dot(q, q));
    if (length <= 0.0) {
        return {};
    }
    return {q.w / length, q.x / length, q.y / length, q.z / length};
}

// Rotation matrix to quaternion (Shepperd's branching, numerically stable).
inline Quaternion quaternion_from_matrix(const Mat3& matrix) {
    const double trace = matrix.at(0, 0) + matrix.at(1, 1) + matrix.at(2, 2);
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        return {0.25 * s, (matrix.at(2, 1) - matrix.at(1, 2)) / s,
                (matrix.at(0, 2) - matrix.at(2, 0)) / s,
                (matrix.at(1, 0) - matrix.at(0, 1)) / s};
    }
    if (matrix.at(0, 0) > matrix.at(1, 1) && matrix.at(0, 0) > matrix.at(2, 2)) {
        const double s =
            std::sqrt(1.0 + matrix.at(0, 0) - matrix.at(1, 1) - matrix.at(2, 2)) * 2.0;
        return {(matrix.at(2, 1) - matrix.at(1, 2)) / s, 0.25 * s,
                (matrix.at(0, 1) + matrix.at(1, 0)) / s,
                (matrix.at(0, 2) + matrix.at(2, 0)) / s};
    }
    if (matrix.at(1, 1) > matrix.at(2, 2)) {
        const double s =
            std::sqrt(1.0 + matrix.at(1, 1) - matrix.at(0, 0) - matrix.at(2, 2)) * 2.0;
        return {(matrix.at(0, 2) - matrix.at(2, 0)) / s,
                (matrix.at(0, 1) + matrix.at(1, 0)) / s, 0.25 * s,
                (matrix.at(1, 2) + matrix.at(2, 1)) / s};
    }
    const double s =
        std::sqrt(1.0 + matrix.at(2, 2) - matrix.at(0, 0) - matrix.at(1, 1)) * 2.0;
    return {(matrix.at(1, 0) - matrix.at(0, 1)) / s,
            (matrix.at(0, 2) + matrix.at(2, 0)) / s,
            (matrix.at(1, 2) + matrix.at(2, 1)) / s, 0.25 * s};
}

inline Mat3 matrix_from_quaternion(const Quaternion& q) {
    const double w = q.w;
    const double x = q.x;
    const double y = q.y;
    const double z = q.z;
    Mat3 matrix;
    matrix.m[0] = 1.0 - 2.0 * (y * y + z * z);
    matrix.m[1] = 2.0 * (x * y - z * w);
    matrix.m[2] = 2.0 * (x * z + y * w);
    matrix.m[3] = 2.0 * (x * y + z * w);
    matrix.m[4] = 1.0 - 2.0 * (x * x + z * z);
    matrix.m[5] = 2.0 * (y * z - x * w);
    matrix.m[6] = 2.0 * (x * z - y * w);
    matrix.m[7] = 2.0 * (y * z + x * w);
    matrix.m[8] = 1.0 - 2.0 * (x * x + y * y);
    return matrix;
}

}  // namespace nricp
