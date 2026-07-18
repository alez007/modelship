"""Tests for the centralized model-source resolver."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelship.infer.model_resolver import (
    ModelDownloadError,
    PinnedSource,
    _select_patterns,
    check_model_source,
    download_model_source,
    parse_model_ref,
    resolve_model_source,
)


def _repo_info(sha: str = "deadbeef"):
    return MagicMock(sha=sha)


class TestParseModelRef:
    def test_hf_repo_no_selector(self):
        result = parse_model_ref("Qwen/Qwen3-7B")
        assert result.source == "Qwen/Qwen3-7B"
        assert result.selector is None
        assert result.is_local is False

    def test_hf_repo_with_selector(self):
        result = parse_model_ref("lmstudio-community/Qwen2.5-7B-Instruct-GGUF:*Q4_K_M.gguf")
        assert result.source == "lmstudio-community/Qwen2.5-7B-Instruct-GGUF"
        assert result.selector == "*Q4_K_M.gguf"
        assert result.is_local is False

    def test_hf_repo_with_exact_filename(self):
        result = parse_model_ref("nomic-ai/nomic-embed-text-v1.5-GGUF:nomic-embed-text-v1.5.Q4_K_M.gguf")
        assert result.source == "nomic-ai/nomic-embed-text-v1.5-GGUF"
        assert result.selector == "nomic-embed-text-v1.5.Q4_K_M.gguf"

    def test_absolute_path_existing(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        result = parse_model_ref(str(f))
        assert result.source == str(f)
        assert result.selector is None
        assert result.is_local is True

    def test_absolute_path_with_colon_in_name_treated_as_local(self, tmp_path: Path):
        # Absolute paths starting with `/` are not split on `:`, even if the file
        # contains a colon (rare but legal).
        d = tmp_path / "weird_dir"
        d.mkdir()
        result = parse_model_ref(str(d))
        assert result.is_local is True
        assert result.selector is None

    def test_multiple_colons_split_on_first(self):
        result = parse_model_ref("org/repo:path/to/file.gguf")
        assert result.source == "org/repo"
        assert result.selector == "path/to/file.gguf"

    def test_pathy_missing_absolute_path_is_still_local(self, tmp_path: Path):
        # A pathy string (starts with /, ./, ~) is local by syntax alone, not
        # by existence — a typo'd local path must fail with a clear
        # FileNotFoundError, not get silently misread as an HF repo id.
        missing = tmp_path / "does-not-exist"
        result = parse_model_ref(str(missing))
        assert result.source == str(missing)
        assert result.is_local is True

    def test_pathy_missing_path_with_selector_is_still_local(self, tmp_path: Path):
        missing_dir = tmp_path / "does-not-exist"
        result = parse_model_ref(f"{missing_dir}:*.gguf")
        assert result.source == str(missing_dir)
        assert result.selector == "*.gguf"
        assert result.is_local is True

    def test_non_pathy_missing_string_is_not_local(self):
        # No leading /, ./, or ~ — read as an HF repo id, same as today.
        result = parse_model_ref("definitely-not-a-real/repo-id")
        assert result.is_local is False


class TestSelectPatterns:
    def test_safetensors_excludes_bin(self):
        files = ["model.safetensors", "pytorch_model.bin", "config.json"]
        patterns = _select_patterns(files)
        assert "*.safetensors" in patterns
        assert "*.bin" not in patterns
        assert "*.bin.index.json" not in patterns

    def test_no_safetensors_falls_back_to_bin(self):
        files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
        patterns = _select_patterns(files)
        assert "*.bin" in patterns
        assert "*.safetensors" not in patterns

    def test_sharded_safetensors_index(self):
        files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "config.json",
        ]
        patterns = _select_patterns(files)
        assert "*.safetensors" in patterns
        assert "*.safetensors.index.json" in patterns

    def test_trust_remote_code_includes_py(self):
        files = ["model.safetensors", "config.json", "modeling_custom.py"]
        patterns = _select_patterns(files, trust_remote_code=True)
        assert "*.py" in patterns
        assert "**/*.py" in patterns

    def test_trust_remote_code_default_excludes_py(self):
        files = ["model.safetensors", "config.json"]
        patterns = _select_patterns(files, trust_remote_code=False)
        assert "*.py" not in patterns

    def test_always_includes_tokenizer_and_config(self):
        files = ["model.safetensors", "tokenizer.json", "config.json"]
        patterns = _select_patterns(files)
        assert "tokenizer*" in patterns
        assert "*.json" in patterns
        assert "preprocessor_config.json" in patterns


class TestResolveLocalPath:
    """Local paths never touch HF — check_model_source resolves them fully,
    so resolve_model_source (check + download) needs no HF mocking."""

    def test_local_file(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        result = resolve_model_source(str(f))
        assert result == str(f.absolute())

    def test_local_dir(self, tmp_path: Path):
        d = tmp_path / "model_snapshot"
        d.mkdir()
        (d / "config.json").write_text("{}")
        result = resolve_model_source(str(d))
        assert result == str(d.absolute())

    def test_local_dir_with_selector_single_match(self, tmp_path: Path):
        d = tmp_path / "ggufs"
        d.mkdir()
        (d / "model-Q4_K_M.gguf").write_text("dummy")
        (d / "model-Q8_0.gguf").write_text("dummy")
        result = resolve_model_source(f"{d}:*Q4_K_M.gguf")
        assert result.endswith("model-Q4_K_M.gguf")

    def test_local_dir_with_selector_no_match(self, tmp_path: Path):
        d = tmp_path / "ggufs"
        d.mkdir()
        (d / "model-Q8_0.gguf").write_text("dummy")
        with pytest.raises(FileNotFoundError, match="matched no files"):
            resolve_model_source(f"{d}:*Q4_K_M.gguf")

    def test_local_dir_with_selector_multiple_matches_returns_first(self, tmp_path: Path):
        # Sharded GGUF case: selector matches several shards; return the first
        # alphabetically so llama.cpp can auto-load the rest.
        d = tmp_path / "ggufs"
        d.mkdir()
        (d / "model-00002-of-00003.gguf").write_text("dummy")
        (d / "model-00001-of-00003.gguf").write_text("dummy")
        (d / "model-00003-of-00003.gguf").write_text("dummy")
        result = resolve_model_source(f"{d}:model-*.gguf")
        assert result.endswith("model-00001-of-00003.gguf")

    def test_local_path_missing(self, tmp_path: Path):
        # A pathy string is local by syntax alone (see TestParseModelRef), so a
        # missing absolute path fails clearly from the local branch — not a
        # confusing "failed to list files for HF repo" error.
        with pytest.raises(FileNotFoundError, match="Local path not found"):
            resolve_model_source(str(tmp_path / "does-not-exist"))

    def test_download_is_noop_for_local(self, tmp_path: Path):
        f = tmp_path / "model.gguf"
        f.write_text("dummy")
        pinned = check_model_source(str(f))
        assert pinned.resolved_path == str(f.absolute())
        assert download_model_source(pinned) == pinned.resolved_path


class TestCheckHfRepoDoesNoDownload:
    """check_model_source must never fetch weight bytes — only listing +
    metadata calls."""

    def test_no_download_calls(self):
        files = ["model.safetensors", "config.json", "tokenizer.json"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            patch("modelship.infer.model_resolver.hf_hub_download") as mock_dl,
            patch("modelship.infer.model_resolver.snapshot_download") as mock_snap,
        ):
            check_model_source("Qwen/Qwen3-7B")
            mock_dl.assert_not_called()
            mock_snap.assert_not_called()

    def test_pins_commit_sha(self):
        files = ["model.safetensors", "config.json"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info("abc123")),
        ):
            pinned = check_model_source("Qwen/Qwen3-7B")
            assert pinned.revision == "abc123"
            assert pinned.resolved_path is None
            assert pinned.repo == "Qwen/Qwen3-7B"

    def test_revision_lookup_failure_wrapped(self):
        files = ["model.safetensors"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", side_effect=Exception("boom")),
            pytest.raises(RuntimeError, match="Failed to resolve commit revision"),
        ):
            check_model_source("Qwen/Qwen3-7B")


class TestResolveHfRepo:
    """resolve_model_source (check + download) end to end."""

    def test_full_snapshot_calls_universal_filter(self):
        files = ["model.safetensors", "config.json", "tokenizer.json"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            patch("modelship.infer.model_resolver.snapshot_download") as mock_snap,
        ):
            mock_snap.return_value = "/cache/snapshot"
            result = resolve_model_source("Qwen/Qwen3-7B")
            assert result == "/cache/snapshot"
            mock_snap.assert_called_once()
            kwargs = mock_snap.call_args.kwargs
            assert "*.safetensors" in kwargs["allow_patterns"]
            assert "*.bin" not in kwargs["allow_patterns"]
            assert kwargs["revision"] == "deadbeef"

    def test_selector_single_file_uses_hf_hub_download(self):
        files = ["model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            patch("modelship.infer.model_resolver.hf_hub_download") as mock_dl,
        ):
            mock_dl.return_value = "/cache/model-Q4_K_M.gguf"
            result = resolve_model_source("org/repo:*Q4_K_M.gguf")
            assert result == "/cache/model-Q4_K_M.gguf"
            mock_dl.assert_called_once_with("org/repo", "model-Q4_K_M.gguf", revision="deadbeef")

    def test_selector_multiple_matches_returns_first_shard_path(self):
        # Sharded GGUF: download all shards via snapshot_download, then return
        # the first shard's full path (not the snapshot dir) so file-path
        # loaders like llama.cpp work.
        files = ["model-00002-of-00002.gguf", "model-00001-of-00002.gguf"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            patch("modelship.infer.model_resolver.snapshot_download") as mock_snap,
        ):
            mock_snap.return_value = "/cache/snapshot"
            result = resolve_model_source("org/repo:*.gguf")
            assert result == "/cache/snapshot/model-00001-of-00002.gguf"
            mock_snap.assert_called_once_with("org/repo", revision="deadbeef", allow_patterns=["*.gguf"])

    def test_selector_no_match_raises(self):
        files = ["model-Q4_K_M.gguf"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            pytest.raises(FileNotFoundError, match="matched no files"),
        ):
            resolve_model_source("org/repo:*Q8_0.gguf")

    def test_multi_variant_gguf_without_selector_raises(self):
        files = [
            "model-Q2_K.gguf",
            "model-Q4_K_M.gguf",
            "model-Q5_K_M.gguf",
            "model-Q8_0.gguf",
        ]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            pytest.raises(ValueError, match="contains 4 GGUF variants"),
        ):
            resolve_model_source("lmstudio-community/Qwen2.5-7B-Instruct-GGUF")

    def test_single_gguf_without_selector_returns_file_path(self):
        # Single-GGUF repo: resolver must return the file path (not a snapshot
        # dir), because llama_server requires a file path.
        files = ["model.gguf", "config.json"]
        with (
            patch("modelship.infer.model_resolver.list_repo_files", return_value=files),
            patch("modelship.infer.model_resolver.repo_info", return_value=_repo_info()),
            patch("modelship.infer.model_resolver.hf_hub_download") as mock_dl,
            patch("modelship.infer.model_resolver.snapshot_download") as mock_snap,
        ):
            mock_dl.return_value = "/cache/model.gguf"
            result = resolve_model_source("org/single-gguf-repo")
            assert result == "/cache/model.gguf"
            mock_dl.assert_called_once_with("org/single-gguf-repo", "model.gguf", revision="deadbeef")
            mock_snap.assert_not_called()

    def test_list_repo_files_failure_wrapped(self):
        with (
            patch(
                "modelship.infer.model_resolver.list_repo_files",
                side_effect=Exception("auth failure"),
            ),
            pytest.raises(RuntimeError, match="Failed to list files"),
        ):
            resolve_model_source("private/repo")


class TestPinnedSourceResolvesToGguf:
    def test_local_file_gguf(self):
        pinned = PinnedSource("/models/x.gguf", None, None, None, None, None)
        assert pinned.resolves_to_gguf

    def test_local_dir_not_gguf(self):
        pinned = PinnedSource("/models/snapshot", None, None, None, None, None)
        assert not pinned.resolves_to_gguf

    def test_hf_single_file_download_gguf(self):
        pinned = PinnedSource(None, "org/repo", "sha", "model.gguf", None, None)
        assert pinned.resolves_to_gguf

    def test_hf_shard_gguf(self):
        pinned = PinnedSource(None, "org/repo", "sha", None, ["*.gguf"], "model-00001-of-00002.gguf")
        assert pinned.resolves_to_gguf

    def test_hf_full_snapshot_not_gguf(self):
        pinned = PinnedSource(None, "org/repo", "sha", None, ["*.safetensors"], None)
        assert not pinned.resolves_to_gguf


class TestDownloadErrorClassification:
    def test_download_failure_is_not_wrapped_by_download_model_source(self):
        # download_model_source itself raises whatever hf raises; wrapping into
        # ModelDownloadError is BaseInfer.ensure_downloaded's job (it
        # needs the model name for the message), not this function's.
        pinned = PinnedSource(None, "org/repo", "sha", "model.safetensors", None, None)
        with (
            patch("modelship.infer.model_resolver.hf_hub_download", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            download_model_source(pinned)

    def test_model_download_error_is_a_plain_exception(self):
        # Deliberately not a subclass of a "permanent" error type — see its
        # docstring: ModelDeployment.__init__ special-cases this type to skip
        # reporting a fatal error to the coordinator.
        assert issubclass(ModelDownloadError, Exception)
