"""Test configuration for the nricp sub-project.

The extension module is built out-of-tree by CMake, so the tests look for it in
the configured build directory (NRICP_BUILD_DIR, defaulting to build/Release)
before falling back to whatever `import nricp_dp` finds. A missing module skips
the suite with a pointer to the build instructions instead of failing every
test with the same import error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIRS = ("build/Release", "build/Debug", "build")


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("NRICP_BUILD_DIR")
    if env:
        dirs.append(Path(env))
    dirs.extend(REPO_ROOT / name for name in DEFAULT_BUILD_DIRS)
    return dirs


def _find_module_dir() -> Path | None:
    for directory in _candidate_dirs():
        if not directory.is_dir():
            continue
        if any(directory.glob("nricp_dp*.pyd")) or any(
            directory.glob("nricp_dp*.so")
        ):
            return directory
    return None


@pytest.fixture(scope="session")
def nricp_dp():
    """The built extension module, or a skip explaining how to build it."""
    module_dir = _find_module_dir()
    if module_dir is not None and str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    try:
        import nricp_dp  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:  # pragma: no cover - depends on local build
        pytest.skip(
            "nricp_dp is not built or not importable "
            f"({exc}); see the build section of README.md",
            allow_module_level=True,
        )
    return nricp_dp
