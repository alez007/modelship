"""Tests for cgroup-aware RAM detection in discover_hardware().

psutil reads the HOST's RAM inside a container; the cgroup pseudo-files hold the
real limit. `_cgroup_memory_limit_bytes` reads them, and discover_hardware takes
min(psutil, cgroup). See modelship/preflight/base.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from modelship.preflight.base import _cgroup_memory_limit_bytes, detect_ram_bytes

_GiB = 1024**3
# cgroup v1's "unlimited" sentinel (PAGE_COUNTER_MAX rounded to the page size).
_V1_UNLIMITED = 0x7FFFFFFFFFFFF000


def _write(tmp_path, name: str, contents: str) -> str:
    p = tmp_path / name
    p.write_text(contents)
    return str(p)


def test_reads_cgroup_v2_numeric_limit(tmp_path):
    v2 = _write(tmp_path, "memory.max", f"{4 * _GiB}\n")
    assert _cgroup_memory_limit_bytes(paths=(v2,)) == 4 * _GiB


def test_cgroup_v2_max_means_unlimited(tmp_path):
    v2 = _write(tmp_path, "memory.max", "max\n")
    assert _cgroup_memory_limit_bytes(paths=(v2,)) is None


def test_reads_cgroup_v1_numeric_limit(tmp_path):
    v1 = _write(tmp_path, "memory.limit_in_bytes", f"{8 * _GiB}")
    assert _cgroup_memory_limit_bytes(paths=(v1,)) == 8 * _GiB


def test_cgroup_v1_unlimited_sentinel_is_returned_but_discarded_by_min(tmp_path):
    # We deliberately do NOT special-case the sentinel here; the caller's min()
    # with the (much smaller) psutil value discards it. Assert the raw read so
    # the contract with discover_hardware stays explicit.
    v1 = _write(tmp_path, "memory.limit_in_bytes", str(_V1_UNLIMITED))
    limit = _cgroup_memory_limit_bytes(paths=(v1,))
    assert limit == _V1_UNLIMITED
    assert min(16 * _GiB, limit) == 16 * _GiB  # min keeps the real host value


def test_missing_files_return_none(tmp_path):
    assert _cgroup_memory_limit_bytes(paths=(str(tmp_path / "nope"),)) is None


def test_non_numeric_contents_return_none(tmp_path):
    junk = _write(tmp_path, "memory.max", "garbage")
    assert _cgroup_memory_limit_bytes(paths=(junk,)) is None


def test_first_existing_path_wins(tmp_path):
    # v2 path listed first and present → used even if a v1 path also exists.
    v2 = _write(tmp_path, "memory.max", f"{2 * _GiB}\n")
    v1 = _write(tmp_path, "memory.limit_in_bytes", f"{8 * _GiB}")
    assert _cgroup_memory_limit_bytes(paths=(v2, v1)) == 2 * _GiB


def test_falls_through_to_second_path_when_first_missing(tmp_path):
    missing = str(tmp_path / "memory.max")
    v1 = _write(tmp_path, "memory.limit_in_bytes", f"{8 * _GiB}")
    assert _cgroup_memory_limit_bytes(paths=(missing, v1)) == 8 * _GiB


# --- detect_ram_bytes: psutil + cgroup interplay ------------------------------


def test_detect_ram_takes_min_of_psutil_and_cgroup():
    with (
        patch("psutil.virtual_memory", return_value=SimpleNamespace(total=16 * _GiB)),
        patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=4 * _GiB),
    ):
        assert detect_ram_bytes() == 4 * _GiB


def test_detect_ram_falls_back_to_cgroup_when_psutil_fails():
    # psutil raising must NOT discard a readable cgroup limit (would return 0).
    with (
        patch("psutil.virtual_memory", side_effect=RuntimeError("boom")),
        patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=4 * _GiB),
    ):
        assert detect_ram_bytes() == 4 * _GiB


def test_detect_ram_zero_only_when_both_unavailable():
    with (
        patch("psutil.virtual_memory", side_effect=RuntimeError("boom")),
        patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=None),
    ):
        assert detect_ram_bytes() == 0
