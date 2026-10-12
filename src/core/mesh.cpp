#include "core/mesh.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_set>

namespace nricp {
namespace {

// A face's identity is its whole vertex set, so the key has to be the complete
// sorted triple. Keying on only the two smallest corners makes two valid faces
// that share an edge look like duplicates, and silently deletes one of them --
// on a real scan that tears holes in the target surface.
struct FaceKey {
    int a;
    int b;
    int c;
    bool operator==(const FaceKey& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& key) const {
        std::size_t hash = static_cast<std::size_t>(key.a) * 73856093u;
        hash ^= static_cast<std::size_t>(key.b) * 19349663u;
        hash ^= static_cast<std::size_t>(key.c) * 83492791u;
        return hash;
    }
};

std::array<int, 3> sorted_corners(const Int3& face) {
    std::array<int, 3> corners{face.v0, face.v1, face.v2};
    std::sort(corners.begin(), corners.end());
    return corners;
}

double triangle_area_squared(const Mesh& mesh, const Int3& face) {
    const Vec3 a = mesh.vertices[face.v0];
    const Vec3 b = mesh.vertices[face.v1];
    const Vec3 c = mesh.vertices[face.v2];
    return squared_norm(cross(b - a, c - a));
}

}  // namespace

Bounds bounds(const Mesh& mesh) {
    Bounds result;
    if (mesh.vertices.empty()) {
        return result;
    }
    result.min = mesh.vertices.front();
    result.max = mesh.vertices.front();
    for (const Vec3& vertex : mesh.vertices) {
        result.min.x = std::min(result.min.x, vertex.x);
        result.min.y = std::min(result.min.y, vertex.y);
        result.min.z = std::min(result.min.z, vertex.z);
        result.max.x = std::max(result.max.x, vertex.x);
        result.max.y = std::max(result.max.y, vertex.y);
        result.max.z = std::max(result.max.z, vertex.z);
    }
    return result;
}

double bounding_box_diagonal(const Mesh& mesh) {
    const Bounds box = bounds(mesh);
    return norm(box.max - box.min);
}

std::vector<Vec3> vertex_normals(const Mesh& mesh) {
    std::vector<Vec3> normals(mesh.vertices.size());
    for (const Int3& face : mesh.faces) {
        const Vec3 a = mesh.vertices[face.v0];
        const Vec3 b = mesh.vertices[face.v1];
        const Vec3 c = mesh.vertices[face.v2];
        // The cross product's length is twice the area, so accumulating it
        // unnormalized already weights each face by area.
        const Vec3 weighted = cross(b - a, c - a);
        normals[face.v0] += weighted;
        normals[face.v1] += weighted;
        normals[face.v2] += weighted;
    }
    for (Vec3& normal : normals) {
        normal = normalized(normal);
    }
    return normals;
}

ValidationIssues validate(const Mesh& mesh) {
    ValidationIssues issues;
    if (mesh.vertices.empty()) {
        issues.add("mesh has no vertices");
    }
    if (mesh.faces.empty()) {
        issues.add("mesh has no faces; closest-point queries need a surface");
    }
    const int vertex_count = static_cast<int>(mesh.vertices.size());
    for (int i = 0; i < vertex_count; ++i) {
        if (!is_finite(mesh.vertices[i])) {
            issues.add("vertex " + std::to_string(i) + " is not finite");
        }
    }
    for (int i = 0; i < static_cast<int>(mesh.faces.size()); ++i) {
        const Int3& face = mesh.faces[i];
        for (int corner = 0; corner < 3; ++corner) {
            const int index = index_of(face, corner);
            if (index < 0 || index >= vertex_count) {
                issues.add("face " + std::to_string(i) + " corner " +
                           std::to_string(corner) + " index " +
                           std::to_string(index) + " is out of range");
            }
        }
    }
    return issues;
}

CleanReport clean(Mesh& mesh) {
    CleanReport report;
    const double scale_squared =
        std::max(bounding_box_diagonal(mesh) * bounding_box_diagonal(mesh), 1e-300);
    const double area_floor = 1e-12 * scale_squared;

    std::unordered_set<FaceKey, FaceKeyHash> seen;
    std::vector<Int3> kept;
    kept.reserve(mesh.faces.size());
    for (const Int3& face : mesh.faces) {
        const int a = face.v0;
        const int b = face.v1;
        const int c = face.v2;
        if (a == b || b == c || a == c) {
            ++report.degenerate_faces;
            continue;
        }
        if (triangle_area_squared(mesh, face) <= area_floor) {
            ++report.degenerate_faces;
            continue;
        }
        const std::array<int, 3> corners = sorted_corners(face);
        if (!seen.insert({corners[0], corners[1], corners[2]}).second) {
            ++report.duplicate_faces;
            continue;
        }
        kept.push_back(face);
    }
    mesh.faces.swap(kept);

    std::vector<int> remap(mesh.vertices.size(), -1);
    for (const Int3& face : mesh.faces) {
        for (int corner = 0; corner < 3; ++corner) {
            remap[static_cast<std::size_t>(index_of(face, corner))] = 0;
        }
    }
    int next = 0;
    std::vector<Vec3> compacted;
    compacted.reserve(mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        if (remap[i] == 0) {
            remap[i] = next++;
            compacted.push_back(mesh.vertices[i]);
        } else {
            ++report.isolated_vertices;
        }
    }
    for (Int3& face : mesh.faces) {
        face = {remap[static_cast<std::size_t>(face.v0)],
                remap[static_cast<std::size_t>(face.v1)],
                remap[static_cast<std::size_t>(face.v2)]};
    }
    mesh.vertices.swap(compacted);
    return report;
}

Normalization compute_normalization(const Mesh& mesh) {
    const Bounds box = bounds(mesh);
    Normalization normalization;
    normalization.center = (box.min + box.max) * 0.5;
    const double diagonal = norm(box.max - box.min);
    normalization.scale = diagonal > 0.0 ? 1.0 / diagonal : 1.0;
    return normalization;
}

void apply_normalization(Mesh& mesh, const Normalization& normalization) {
    for (Vec3& vertex : mesh.vertices) {
        vertex = (vertex - normalization.center) * normalization.scale;
    }
}

void undo_normalization(Mesh& mesh, const Normalization& normalization) {
    for (Vec3& vertex : mesh.vertices) {
        vertex = vertex / normalization.scale + normalization.center;
    }
}

}  // namespace nricp
