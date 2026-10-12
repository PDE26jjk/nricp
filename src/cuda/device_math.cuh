#pragma once

#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <sm_20_atomic_functions.h>
#include <sm_32_atomic_functions.h>
#include <sm_35_atomic_functions.h>
#include <sm_60_atomic_functions.h>

#include <cmath>

// ---------------------------------------------------------------------------
// The atomic intrinsics, for the editor
// ---------------------------------------------------------------------------
//
// nvcc declares the atomic functions itself while compiling a .cu file. The
// toolkit headers included above do so only for nvcc: device_atomic_functions.h
// and every sm_*_atomic_functions.h wrap their whole body in
//     #if defined(__cplusplus) && defined(__CUDACC__)
// (the .hpp variants do the same), so a parser that is not nvcc gets no
// declaration at all and marks every atomic red. Measured with the toolkit
// headers on the include path, a plain host compile of
//     #include <cuda_runtime.h>
//     #include <device_atomic_functions.h>
//     float f(float* p) { return atomicAdd(p, 1.0f); }
// still fails with "error C3861: 'atomicAdd': identifier not found".
//
// gridDim/blockIdx/threadIdx/blockDim do not need the same treatment:
// cuda_runtime.h pulls in device_launch_parameters.h, which is host safe.
//
// nvcc never sees this branch, so it cannot affect a device build, and the
// declarations are prototypes rather than definitions, so a parser that does
// reach the toolkit declarations cannot collide with them.
#if !defined(__CUDACC__)

int atomicAdd(int* address, int value);
unsigned int atomicAdd(unsigned int* address, unsigned int value);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int value);
float atomicAdd(float* address, float value);
double atomicAdd(double* address, double value);

int atomicMin(int* address, int value);
unsigned int atomicMin(unsigned int* address, unsigned int value);
int atomicMax(int* address, int value);
unsigned int atomicMax(unsigned int* address, unsigned int value);
unsigned int atomicInc(unsigned int* address, unsigned int value);

int atomicCAS(int* address, int compare, int value);
unsigned int atomicCAS(unsigned int* address, unsigned int compare,
                       unsigned int value);

void __syncthreads();
void __syncwarp();
void __threadfence();
void __threadfence_block();

#endif  // !__CUDACC__

// Device-side scalar types and small linear algebra. The accelerated path is
// single precision throughout (the CPU reference is the double-precision
// arbiter), so everything here is float.
namespace nricp {
namespace gpu {

struct Vec3f {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

__host__ __device__ inline Vec3f operator+(const Vec3f& a, const Vec3f& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline Vec3f operator-(const Vec3f& a, const Vec3f& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ inline Vec3f operator*(const Vec3f& a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}
__host__ __device__ inline Vec3f& operator+=(Vec3f& a, const Vec3f& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline float dot(const Vec3f& a, const Vec3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3f cross(const Vec3f& a, const Vec3f& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ inline float squared_norm(const Vec3f& a) { return dot(a, a); }
__host__ __device__ inline float norm(const Vec3f& a) { return sqrtf(dot(a, a)); }
__host__ __device__ inline Vec3f normalized(const Vec3f& a) {
    const float length = norm(a);
    return length > 0.0f ? a * (1.0f / length) : Vec3f{};
}

// Row-major 3x3, used for node rotations.
struct Mat3f {
    float m[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
};

__host__ __device__ inline Vec3f operator*(const Mat3f& a, const Vec3f& v) {
    return {a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z,
            a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
            a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
}

__host__ __device__ inline Mat3f operator*(const Mat3f& a, const Mat3f& b) {
    Mat3f out;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum += a.m[row * 3 + k] * b.m[k * 3 + column];
            }
            out.m[row * 3 + column] = sum;
        }
    }
    return out;
}

// Row-major 6x6, the block type of the reduced system.
struct Mat6f {
    float m[36] = {};

    __host__ __device__ float& at(int row, int column) { return m[row * 6 + column]; }
    __host__ __device__ float at(int row, int column) const {
        return m[row * 6 + column];
    }
};

// Adds the outer product of two 6-vectors: A += w * lhs * rhs^T.
__device__ inline void add_outer_product(
    Mat6f& block, const float lhs[6], const float rhs[6], float weight) {
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            block.at(row, column) += weight * lhs[row] * rhs[column];
        }
    }
}

// In-place Cholesky factorization of a symmetric positive definite 6x6 block;
// returns false when the block is not positive definite (the caller then damps
// it and tries again). Only the lower triangle is used and written.
__device__ inline bool cholesky6(Mat6f& block) {
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column <= row; ++column) {
            float sum = block.at(row, column);
            for (int k = 0; k < column; ++k) {
                sum -= block.at(row, k) * block.at(column, k);
            }
            if (row == column) {
                if (sum <= 0.0f) {
                    return false;
                }
                block.at(row, column) = sqrtf(sum);
            } else {
                block.at(row, column) = sum / block.at(column, column);
            }
        }
    }
    return true;
}

// Solves a factorized block: x = A^-1 * rhs.
__device__ inline void solve_cholesky6(const Mat6f& factor, const float rhs[6], float x[6]) {
    float y[6];
    for (int row = 0; row < 6; ++row) {
        float sum = rhs[row];
        for (int k = 0; k < row; ++k) {
            sum -= factor.at(row, k) * y[k];
        }
        y[row] = sum / factor.at(row, row);
    }
    for (int row = 5; row >= 0; --row) {
        float sum = y[row];
        for (int k = row + 1; k < 6; ++k) {
            sum -= factor.at(k, row) * x[k];
        }
        x[row] = sum / factor.at(row, row);
    }
}

// Rodrigues' formula for a rotation increment.
__device__ inline Mat3f exp_so3(const Vec3f& omega) {
    const float theta_squared = dot(omega, omega);
    Mat3f omega_hat;
    omega_hat.m[0] = 0.0f;
    omega_hat.m[1] = -omega.z;
    omega_hat.m[2] = omega.y;
    omega_hat.m[3] = omega.z;
    omega_hat.m[4] = 0.0f;
    omega_hat.m[5] = -omega.x;
    omega_hat.m[6] = -omega.y;
    omega_hat.m[7] = omega.x;
    omega_hat.m[8] = 0.0f;
    const Mat3f omega_hat_squared = omega_hat * omega_hat;
    Mat3f result;
    if (theta_squared < 1e-12f) {
        for (int i = 0; i < 9; ++i) {
            result.m[i] += omega_hat.m[i] + 0.5f * omega_hat_squared.m[i];
        }
        return result;
    }
    const float theta = sqrtf(theta_squared);
    const float a = sinf(theta) / theta;
    const float b = (1.0f - cosf(theta)) / theta_squared;
    for (int i = 0; i < 9; ++i) {
        result.m[i] += a * omega_hat.m[i] + b * omega_hat_squared.m[i];
    }
    return result;
}

struct Quaternionf {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

// Rotation matrix to quaternion (Shepperd's branching), used to keep the
// rotation-vector extraction accurate at any angle.
__device__ inline Quaternionf quaternion_from_matrix(const Mat3f& m) {
    const float trace = m.m[0] + m.m[4] + m.m[8];
    if (trace > 0.0f) {
        const float s = sqrtf(trace + 1.0f) * 2.0f;
        return {0.25f * s, (m.m[7] - m.m[5]) / s, (m.m[2] - m.m[6]) / s,
                (m.m[3] - m.m[1]) / s};
    }
    if (m.m[0] > m.m[4] && m.m[0] > m.m[8]) {
        const float s = sqrtf(1.0f + m.m[0] - m.m[4] - m.m[8]) * 2.0f;
        return {(m.m[7] - m.m[5]) / s, 0.25f * s, (m.m[1] + m.m[3]) / s,
                (m.m[2] + m.m[6]) / s};
    }
    if (m.m[4] > m.m[8]) {
        const float s = sqrtf(1.0f + m.m[4] - m.m[0] - m.m[8]) * 2.0f;
        return {(m.m[2] - m.m[6]) / s, (m.m[1] + m.m[3]) / s, 0.25f * s,
                (m.m[5] + m.m[7]) / s};
    }
    const float s = sqrtf(1.0f + m.m[8] - m.m[0] - m.m[4]) * 2.0f;
    return {(m.m[3] - m.m[1]) / s, (m.m[2] + m.m[6]) / s, (m.m[5] + m.m[7]) / s,
            0.25f * s};
}

__device__ inline Vec3f log_so3(const Mat3f& rotation) {
    Quaternionf q = quaternion_from_matrix(rotation);
    const float length = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (length > 0.0f) {
        q.w /= length;
        q.x /= length;
        q.y /= length;
        q.z /= length;
    }
    if (q.w < 0.0f) {
        q.w = -q.w;
        q.x = -q.x;
        q.y = -q.y;
        q.z = -q.z;
    }
    const Vec3f vector{q.x, q.y, q.z};
    const float vector_length = norm(vector);
    if (vector_length < 1e-8f) {
        return Vec3f{};
    }
    const float angle = 2.0f * atan2f(vector_length, q.w);
    return vector * (angle / vector_length);
}

// Inverse left Jacobian of SO(3): the exact derivative of the rotation vector
// with respect to a left perturbation. Clamped near a half turn, where the
// closed form is singular.
__device__ inline Mat3f left_jacobian_inverse(const Vec3f& omega) {
    const float theta_squared = dot(omega, omega);
    Mat3f omega_hat;
    omega_hat.m[0] = 0.0f;
    omega_hat.m[1] = -omega.z;
    omega_hat.m[2] = omega.y;
    omega_hat.m[3] = omega.z;
    omega_hat.m[4] = 0.0f;
    omega_hat.m[5] = -omega.x;
    omega_hat.m[6] = -omega.y;
    omega_hat.m[7] = omega.x;
    omega_hat.m[8] = 0.0f;
    const Mat3f omega_hat_squared = omega_hat * omega_hat;
    Mat3f result;
    if (theta_squared < 1e-12f) {
        for (int i = 0; i < 9; ++i) {
            result.m[i] = (i % 4 == 0 ? 1.0f : 0.0f) - 0.5f * omega_hat.m[i] +
                          (1.0f / 12.0f) * omega_hat_squared.m[i];
        }
        return result;
    }
    const float theta = sqrtf(theta_squared);
    if (theta > 3.14159265f - 1e-3f) {
        for (int i = 0; i < 9; ++i) {
            result.m[i] = (i % 4 == 0 ? 1.0f : 0.0f) - 0.5f * omega_hat.m[i];
        }
        return result;
    }
    const float second =
        1.0f / theta_squared -
        (1.0f + cosf(theta)) / (2.0f * theta * sinf(theta));
    for (int i = 0; i < 9; ++i) {
        result.m[i] = (i % 4 == 0 ? 1.0f : 0.0f) - 0.5f * omega_hat.m[i] +
                      second * omega_hat_squared.m[i];
    }
    return result;
}

}  // namespace gpu
}  // namespace nricp
