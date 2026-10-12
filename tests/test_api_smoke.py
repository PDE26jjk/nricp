"""The module must identify itself, and that identity must match the build."""

from __future__ import annotations

import re
import sys


def test_version_is_reported(nricp_dp):
    assert re.fullmatch(r"\d+\.\d+\.\d+", nricp_dp.__version__)


def test_build_info_is_complete(nricp_dp):
    info = nricp_dp.build_info()
    for key in (
        "version",
        "build_type",
        "cxx_compiler",
        "cuda_toolkit",
        "cuda_architectures",
        "accelerated_precision",
        "python_version",
        "pybind11_version",
    ):
        assert key in info, f"build_info() is missing {key}"
        assert isinstance(info[key], str) and info[key]
    assert info["version"] == nricp_dp.__version__


def test_module_matches_the_running_interpreter(nricp_dp):
    info = nricp_dp.build_info()
    python_version = "%d.%d.%d" % sys.version_info[:3]
    assert info["python_version"].startswith(
        "%d.%d" % sys.version_info[:2]
    ), f"module built for {info['python_version']}, running {python_version}"


def test_build_target_is_pinned_toolchain(nricp_dp):
    """The pinned toolkit and architecture are part of the acceptance config."""
    info = nricp_dp.build_info()
    assert info["cuda_toolkit"].startswith("12.1"), info["cuda_toolkit"]
    assert "86" in info["cuda_architectures"], info["cuda_architectures"]
