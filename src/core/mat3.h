#pragma once

#include <cmath>

#include "core/vec3.h"

namespace nricp {

// Row-major 3x3 matrix.
struct Mat3 {
    double m[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    double at(int row, int col) const { return m[row * 3 + col]; }
    double& at(int row, int col) { return m[row * 3 + col]; }
};

inline Vec3 operator*(const Mat3& a, const Vec3& v) {
    return {
        a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z,
        a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
        a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z,
    };
}

inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 out;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += a.at(row, k) * b.at(k, col);
            }
            out.at(row, col) = sum;
        }
    }
    return out;
}

inline Mat3 transposed(const Mat3& a) {
    Mat3 out;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            out.at(row, col) = a.at(col, row);
        }
    }
    return out;
}

inline double determinant(const Mat3& a) {
    return a.m[0] * (a.m[4] * a.m[8] - a.m[5] * a.m[7]) -
           a.m[1] * (a.m[3] * a.m[8] - a.m[5] * a.m[6]) +
           a.m[2] * (a.m[3] * a.m[7] - a.m[4] * a.m[6]);
}

}  // namespace nricp
