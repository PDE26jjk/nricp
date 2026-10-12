#pragma once

// numpy ships with the interpreter (Blender 5.2 bundles numpy 2.3.4, the
// standalone development interpreter has a newer one). Targeting the numpy 2.0
// API keeps a single build usable by any numpy 2.x the caller happens to have,
// and the module still needs nothing installed into Blender's Python.
#ifndef NPY_TARGET_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API
#endif
