#pragma once

#include <string>
#include <vector>

#include "core/vec3.h"

namespace nricp {

struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Int3> faces;
};

struct Bounds {
    Vec3 min;
    Vec3 max;
};

Bounds bounds(const Mesh& mesh);
double bounding_box_diagonal(const Mesh& mesh);

// Area-weighted vertex normals, normalized. Faces are assumed consistently
// wound; a vertex with no usable face gets a zero normal.
std::vector<Vec3> vertex_normals(const Mesh& mesh);

struct ValidationIssues {
    bool ok = true;
    std::vector<std::string> problems;

    void add(std::string problem) {
        ok = false;
        problems.push_back(std::move(problem));
    }
};

// Structural validation: shapes, index ranges, and finiteness. Anything the
// caller can fix is reported by index rather than silently repaired.
ValidationIssues validate(const Mesh& mesh);

struct CleanReport {
    int duplicate_faces = 0;
    int degenerate_faces = 0;
    int isolated_vertices = 0;
};

// Removes duplicate faces (same vertex set), degenerate faces (repeated index or
// zero area at mesh scale), and vertices that no remaining face references.
// The winding of the first occurrence of a face is kept.
CleanReport clean(Mesh& mesh);

// Rigid normalization: center on the bounding-box center and scale so the
// bounding-box diagonal becomes 1. Every solver constant assumes this scale.
struct Normalization {
    Vec3 center;
    double scale = 1.0;
};

Normalization compute_normalization(const Mesh& mesh);
void apply_normalization(Mesh& mesh, const Normalization& normalization);
void undo_normalization(Mesh& mesh, const Normalization& normalization);

}  // namespace nricp
