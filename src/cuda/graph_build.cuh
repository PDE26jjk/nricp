#pragma once

#include "core/graph.h"
#include "core/mesh.h"

namespace nricp {

// The template graph, built on the device.
//
// Both of the construction's passes compare every template vertex against every
// node of a level, which at a hundred thousand vertices and the pinned
// schedule's fifteen hundred nodes is about three hundred million distance
// evaluations, and that -- not the solve -- is what a single wrap's time is
// made of.
//
// The device version computes in double precision and applies the reference's
// comparisons in the reference's order: the sampling keeps the farthest point
// and breaks ties to the lower index exactly as the sweep does, and the binding
// walks a vertex's candidate nodes in index order with the same "keep the four
// best, ties to the lower index" rule. The seeds, the four nodes and the weights
// therefore come out *identical* to the core layer's, not merely close, which is
// what lets one implementation be checked against the other.
//
// `device_graph_available` reports whether a device is present; the caller falls
// back to the core layer's builder when it is not.
TemplateGraph build_graph_on_device(
    const Mesh& template_mesh, const GraphSettings& settings);

bool device_graph_available();

}  // namespace nricp
