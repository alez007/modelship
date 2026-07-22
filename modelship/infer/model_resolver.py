import fnmatch
import os
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import hf_hub_download, model_info, snapshot_download

from modelship.logging import get_logger

logger = get_logger("startup")


class ModelDownloadError(Exception):
    """Raised by `download_model_source` when a *validated* source can't be
    downloaded (network blip, transient HF error, disk full, ...). Kept
    distinct from `check_model_source`'s errors so `ModelDeployment` treats
    it as transient rather than fatal."""


class ResolvedSource(NamedTuple):
    """Result of parsing a model reference."""

    source: str  # repo_id or local path
    selector: str | None  # filename or glob pattern
    is_local: bool


def _is_pathy(s: str) -> bool:
    return s.startswith("/") or s.startswith("./") or s.startswith("~")


def _expand(s: str) -> str:
    """expanduser only for pathy strings — Path.resolve() never expands `~`,
    so this must happen here or a valid `~/...` ref 404s downstream."""
    return os.path.expanduser(s) if _is_pathy(s) else s


def parse_model_ref(model: str) -> ResolvedSource:
    """Parses model string into (source, selector, is_local).

    Path-first: if the literal full string is an existing local path, treat it
    as one (covers the rare colon-in-filename case). Otherwise split on the
    first ':' — the part before is the source, the part after is the selector.

    A pathy source (starts with /, ./, or ~) is always local regardless of
    whether it exists, so a missing path fails clearly downstream instead of
    being misread as an HF repo id. `~` is expanded in the returned source."""
    expanded = _expand(model)
    if _is_pathy(model) and Path(expanded).exists():
        return ResolvedSource(source=expanded, selector=None, is_local=True)

    if ":" in model:
        source, selector = model.split(":", 1)
        return ResolvedSource(source=_expand(source), selector=selector, is_local=_is_pathy(source))

    return ResolvedSource(source=expanded, selector=None, is_local=_is_pathy(model))


def _select_patterns(repo_files: list[str], trust_remote_code: bool = False) -> list[str] | None:
    """Universal filter: prefer safetensors over bin/h5/onnx if present."""
    has_safetensors = any(f.endswith(".safetensors") or ".safetensors.index.json" in f for f in repo_files)

    patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "tokenizer*",
        "vocab*",
        "merges*",
        "*.jinja",
        "chat_template*",
        "preprocessor_config.json",
        "generation_config.json",
        "image_processor_config.json",
        "processor_config.json",
    ]

    if trust_remote_code:
        patterns.append("*.py")
        patterns.append("**/*.py")

    if has_safetensors:
        patterns.extend(["*.safetensors", "*.safetensors.index.json", "**/*.safetensors"])
    else:
        # Fallback to bin if no safetensors
        patterns.extend(["*.bin", "*.bin.index.json", "**/*.bin"])

    return patterns


def _format_gguf_variants(repo_files: list[str]) -> str:
    """Format the GGUF files in a repo as a bullet list for error messages."""
    ggufs = sorted(f for f in repo_files if f.endswith(".gguf"))
    return "\n".join(f"  - {f}" for f in ggufs)


class PinnedSource(NamedTuple):
    """Result of `check_model_source` — enough to download the model later
    without touching the network again except for the download itself.

    `resolved_path` is set for local refs (already fully resolved). For HF
    refs it's None, and `download_filename` XOR `download_patterns` tells
    `download_model_source` whether to use `hf_hub_download` or
    `snapshot_download`; `first_shard` picks the entry-point file out of a
    multi-file snapshot."""

    resolved_path: str | None
    repo: str | None
    revision: str | None
    download_filename: str | None
    download_patterns: list[str] | None
    first_shard: str | None

    @property
    def resolves_to_gguf(self) -> bool:
        """Whether this source resolves to a single `.gguf` file, without
        downloading it."""
        if self.resolved_path is not None:
            return self.resolved_path.lower().endswith(".gguf")
        filename = self.download_filename or self.first_shard
        return bool(filename and filename.lower().endswith(".gguf"))


def check_model_source(model_ref: str, trust_remote_code: bool = False) -> PinnedSource:
    """Driver-side: validate model_ref without fetching any weight bytes.

    - Local path: fully resolved here (existence + selector match).
    - HF repo: `repo_info` gives both the file listing (siblings, surfacing
      auth/missing-repo/selector-no-match) and the current commit SHA in one
      call, so every node downloads the same pinned revision later.
    """
    source, selector, is_local = parse_model_ref(model_ref)

    # Re-check localness in case it didn't start with / but exists (e.g. relative path)
    if not is_local and Path(source).exists():
        is_local = True

    if is_local:
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local path not found: {path}")

        if selector and path.is_dir():
            # If selector is provided for a local dir, try to match it
            matches = sorted(path.glob(selector))
            if not matches:
                # Try recursive if not found
                matches = sorted(path.rglob(selector))

            if not matches:
                raise FileNotFoundError(f"Selector {selector!r} matched no files in {path}")
            if len(matches) > 1:
                # Sharded weights (e.g. model-00001-of-00003.gguf): return the
                # first shard sorted alphabetically. llama.cpp auto-loads the
                # rest given the first shard's path.
                logger.info(
                    "Selector %r matched %d files in %s; returning first shard %s",
                    selector,
                    len(matches),
                    path,
                    matches[0].name,
                )
            resolved = matches[0].absolute()
            return PinnedSource(str(resolved), None, None, None, None, None)

        resolved = path.absolute()
        return PinnedSource(str(resolved), None, None, None, None, None)

    # HF Resolve
    try:
        info = model_info(source)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch info for HF repo {source!r}: {e}") from e

    if info.siblings is None:
        raise RuntimeError(f"HF repo {source!r} returned no file listing")

    repo_files = [s.rfilename for s in info.siblings]
    revision = info.sha

    if selector:
        matches = sorted(fnmatch.filter(repo_files, selector))
        if not matches:
            raise FileNotFoundError(f"Selector {selector!r} matched no files in HF repo {source!r}")

        if len(matches) > 1:
            # Sharded weights (e.g. model-00001-of-00003.gguf): pull every shard
            # via snapshot_download, then return the path to the first shard so
            # loaders like llama.cpp (which want a file, not a directory) can
            # auto-load the rest.
            logger.info(
                "Selector %r matched %d files in HF repo %r; will download all shards, resolving to first %s",
                selector,
                source,
                matches[0],
            )
            return PinnedSource(None, source, revision, None, [selector], matches[0])

        # Single match: download via hf_hub_download
        return PinnedSource(None, source, revision, matches[0], None, None)

    # No selector: detect a multi-variant GGUF repo and require an explicit pick.
    # This catches the common `model: org/repo-GGUF` mistake before the loader
    # silently auto-resolves to the wrong quant.
    ggufs = [f for f in repo_files if f.endswith(".gguf")]
    if len(ggufs) > 1:
        raise ValueError(
            f"HF repo {source!r} contains {len(ggufs)} GGUF variants — pick one with the `:filename` "
            f"syntax (glob supported, must match exactly one file):\n"
            f"{_format_gguf_variants(repo_files)}\n"
            f"Example: model: {source}:*Q4_K_M.gguf"
        )

    # Single GGUF in the repo: download it directly to a file path.
    # llama_server requires a file path, not a directory, so snapshot_download
    # would break it. The implicit "the only GGUF" is unambiguous.
    if len(ggufs) == 1:
        logger.info("HF repo %r has a single GGUF (%s); will resolve to its file path", source, ggufs[0])
        return PinnedSource(None, source, revision, ggufs[0], None, None)

    # Full snapshot with universal filter
    patterns = _select_patterns(repo_files, trust_remote_code=trust_remote_code)
    return PinnedSource(None, source, revision, None, patterns, None)


def download_model_source(pinned: PinnedSource) -> str:
    """Download (or confirm already-cached) *pinned* and return its final
    absolute local path. A no-op for local refs. For HF refs,
    `hf_hub_download`/`snapshot_download` check their own cache first, so
    calling this when the files are already present is cheap."""
    if pinned.resolved_path is not None:
        return pinned.resolved_path

    assert pinned.repo is not None  # PinnedSource invariant: local xor repo

    if pinned.download_filename is not None:
        return hf_hub_download(pinned.repo, pinned.download_filename, revision=pinned.revision)

    assert pinned.download_patterns is not None
    snapshot_dir = snapshot_download(pinned.repo, revision=pinned.revision, allow_patterns=pinned.download_patterns)
    if pinned.first_shard is not None:
        return str(Path(snapshot_dir, pinned.first_shard).absolute())
    return snapshot_dir


def resolve_model_source(model_ref: str, trust_remote_code: bool = False) -> str:
    """One-shot check + download, for callers that don't need the
    driver/actor split (e.g. the standalone benchmark entrypoint)."""
    return download_model_source(check_model_source(model_ref, trust_remote_code=trust_remote_code))
