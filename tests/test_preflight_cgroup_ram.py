"""Tests for cgroup-aware RAM detection in discover_hardware().

psutil reads the HOST's RAM inside a container; the cgroup pseudo-files hold the
real limit. `_cgroup_memory_limit_bytes` reads them, and discover_hardware takes
min(psutil, cgroup). See modelship/preflight/base.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from modelship.preflight.base import (
    _cgroup_memory_available_bytes,
    _cgroup_memory_limit_bytes,
    detect_available_ram_bytes,
    detect_ram_bytes,
)

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


def test_cgroup_v1_unlimited_sentinel_is_recognized_as_none(tmp_path):
    # The near-INT64_MAX "unlimited" sentinel must be treated as no-limit (None)
    # at the source — not leaked for the caller to discard — so it's safe even
    # without a psutil value to min() against.
    v1 = _write(tmp_path, "memory.limit_in_bytes", str(_V1_UNLIMITED))
    assert _cgroup_memory_limit_bytes(paths=(v1,)) is None


def test_cgroup_long_max_sentinel_is_recognized_as_none(tmp_path):
    v1 = _write(tmp_path, "memory.limit_in_bytes", str(0x7FFFFFFFFFFFFFFF))
    assert _cgroup_memory_limit_bytes(paths=(v1,)) is None


def test_cgroup_zero_or_negative_is_none(tmp_path):
    assert _cgroup_memory_limit_bytes(paths=(_write(tmp_path, "memory.max", "0"),)) is None


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


# --- _cgroup_memory_available_bytes: headroom = limit - current + reclaimable ---


def _v2_stat(tmp_path, inactive_file: int, active_file: int) -> str:
    body = f"anon 12345\ninactive_file {inactive_file}\nactive_file {active_file}\nslab 999\n"
    return _write(tmp_path, "memory.stat", body)


def test_cgroup_v2_available_adds_back_reclaimable_cache(tmp_path):
    limit = 8 * _GiB
    current = 6 * _GiB
    inactive, active = 1 * _GiB, 1 * _GiB  # 2 GiB of evictable page cache
    usage = _write(tmp_path, "memory.current", str(current))
    stat = _v2_stat(tmp_path, inactive, active)
    with patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=limit):
        # headroom = 8 - 6 + (1 + 1) = 4 GiB
        assert _cgroup_memory_available_bytes(usage_paths=(usage,), stat_paths=(stat,)) == 4 * _GiB


def test_cgroup_v1_available_uses_total_file_keys(tmp_path):
    limit = 8 * _GiB
    current = 5 * _GiB
    usage = _write(tmp_path, "memory.usage_in_bytes", str(current))
    stat = _write(tmp_path, "memory.stat", f"total_inactive_file {1 * _GiB}\ntotal_active_file {2 * _GiB}\n")
    with patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=limit):
        # headroom = 8 - 5 + (1 + 2) = 6 GiB
        assert _cgroup_memory_available_bytes(usage_paths=(usage,), stat_paths=(stat,)) == 6 * _GiB


def test_cgroup_available_is_none_when_uncapped(tmp_path):
    # No limit (memory.max == "max") → defer to host psutil, not a cgroup figure.
    usage = _write(tmp_path, "memory.current", str(3 * _GiB))
    with patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=None):
        assert _cgroup_memory_available_bytes(usage_paths=(usage,), stat_paths=()) is None


def test_cgroup_available_treats_missing_stat_as_zero_reclaimable(tmp_path):
    limit = 8 * _GiB
    usage = _write(tmp_path, "memory.current", str(6 * _GiB))
    with patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=limit):
        # memory.stat unreadable → reclaimable 0 → headroom = 8 - 6 = 2 GiB (conservative)
        missing_stat = str(tmp_path / "nope.stat")
        assert _cgroup_memory_available_bytes(usage_paths=(usage,), stat_paths=(missing_stat,)) == 2 * _GiB


def test_cgroup_available_is_none_when_usage_unreadable(tmp_path):
    with patch("modelship.preflight.base._cgroup_memory_limit_bytes", return_value=8 * _GiB):
        assert _cgroup_memory_available_bytes(usage_paths=(str(tmp_path / "nope"),), stat_paths=()) is None


# --- detect_available_ram_bytes: psutil + cgroup interplay --------------------


def test_detect_available_takes_min_of_psutil_and_cgroup():
    with (
        patch("psutil.virtual_memory", return_value=SimpleNamespace(available=10 * _GiB)),
        patch("modelship.preflight.base._cgroup_memory_available_bytes", return_value=3 * _GiB),
    ):
        assert detect_available_ram_bytes() == 3 * _GiB


def test_detect_available_falls_back_to_host_when_uncapped():
    # memory.max == "max" → cgroup headroom None → use host psutil available.
    with (
        patch("psutil.virtual_memory", return_value=SimpleNamespace(available=10 * _GiB)),
        patch("modelship.preflight.base._cgroup_memory_available_bytes", return_value=None),
    ):
        assert detect_available_ram_bytes() == 10 * _GiB


def test_detect_available_falls_back_to_cgroup_when_psutil_fails():
    with (
        patch("psutil.virtual_memory", side_effect=RuntimeError("boom")),
        patch("modelship.preflight.base._cgroup_memory_available_bytes", return_value=3 * _GiB),
    ):
        assert detect_available_ram_bytes() == 3 * _GiB
