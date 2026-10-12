#pragma once

#include <cmath>

#include "core/mat3.h"
#include "core/quaternion.h"
#include "core/vec3.h"

namespace nricp {

// Skew-symmetric matrix of a 3-vector, i.e. the matrix with [w] v = w x v.
inline Mat3 skew(const Vec3& w) {
    Mat3 matrix;
    matrix.m[0] = 0.0;
    matrix.m[1] = -w.z;
    matrix.m[2] = w.y;
    matrix.m[3] = w.z;
    matrix.m[4] = 0.0;
    matrix.m[5] = -w.x;
    matrix.m[6] = -w.y;
    matrix.m[7] = w.x;
    matrix.m[8] = 0.0;
    return matrix;
}

// Rotation matrix of a tangent-space increment (Rodrigues' formula), with the
// small-angle series used near zero so the result stays well conditioned.
inline Mat3 exp_so3(const Vec3& omega) {
    const double theta_squared = dot(omega, omega);
    const Mat3 omega_hat = skew(omega);
    const Mat3 omega_hat_squared = omega_hat * omega_hat;
    Mat3 result;
    if (theta_squared < 1e-24) {
        for (int i = 0; i < 9; ++i) {
            result.m[i] += omega_hat.m[i] + 0.5 * omega_hat_squared.m[i];
        }
        return result;
    }
    const double theta = std::sqrt(theta_squared);
    const double a = std::sin(theta) / theta;
    const double b = (1.0 - std::cos(theta)) / theta_squared;
    for (int i = 0; i < 9; ++i) {
        result.m[i] += a * omega_hat.m[i] + b * omega_hat_squared.m[i];
    }
    return result;
}

// Tangent-space vector of a rotation matrix. Goes through the quaternion so it
// stays accurate at any angle, including near a half turn.
inline Vec3 log_so3(const Mat3& rotation) {
    Quaternion q = normalized(quaternion_from_matrix(rotation));
    if (q.w < 0.0) {
        q = q * -1.0;
    }
    const Vec3 vector{q.x, q.y, q.z};
    const double length = norm(vector);
    if (length < 1e-12) {
        return Vec3{};
    }
    const double angle = 2.0 * std::atan2(length, q.w);
    return vector * (angle / length);
}

// Inverse left Jacobian of SO(3). It is the exact derivative of the rotation
// vector with respect to a left perturbation:
//   d log(exp([d]) R) / d d  =  J_l^{-1}(log R).
// Only its first-order term is the identity, so any block that penalizes
// log(R) needs this to stay consistent with a finite-difference check.
inline Mat3 left_jacobian_inverse(const Vec3& omega) {
    const double theta_squared = dot(omega, omega);
    const Mat3 omega_hat = skew(omega);
    const Mat3 omega_hat_squared = omega_hat * omega_hat;
    Mat3 result;
    if (theta_squared < 1e-12) {
        // I - Omega/2 + Omega^2/12
        const double coefficients[3] = {1.0, -0.5, 1.0 / 12.0};
        for (int i = 0; i < 9; ++i) {
            result.m[i] = coefficients[0] * (i % 4 == 0 ? 1.0 : 0.0) +
                          coefficients[1] * omega_hat.m[i] +
                          coefficients[2] * omega_hat_squared.m[i];
        }
        return result;
    }
    const double theta = std::sqrt(theta_squared);
    if (theta > 3.14159265358979323846 - 1e-6) {
        // The inverse left Jacobian is singular at a half turn and the closed
        // form loses its significant digits near it. Past this point the
        // second-order term is meaningless anyway, so the first-order form is
        // used; rotations that large do not occur in a warmed-up solve.
        for (int i = 0; i < 9; ++i) {
            result.m[i] = (i % 4 == 0 ? 1.0 : 0.0) - 0.5 * omega_hat.m[i];
        }
        return result;
    }
    const double second =
        1.0 / theta_squared -
        (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    for (int i = 0; i < 9; ++i) {
        result.m[i] = (i % 4 == 0 ? 1.0 : 0.0) - 0.5 * omega_hat.m[i] +
                      second * omega_hat_squared.m[i];
    }
    return result;
}

}  // namespace nricp
