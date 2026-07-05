import fnmatch
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from modelship.logging import get_logger

logger = get_logger("startup")


class ResolvedSource(NamedTuple):
    """Result of parsing a model reference."""

    source: str  # repo_id or local path
    selector: str | None  # filename or glob pattern
    is_local: bool


def _is_pathy(s: str) -> bool:
    return s.startswith("/") or s.startswith("./") or s.startswith("~")


def parse_model_ref(model: str) -> ResolvedSource:
    """Parses model string into (source, selector, is_local).

    Path-first: if the literal full string is an existing local path, treat it
    as one (covers the rare colon-in-filename case). Otherwise split on the
    first ':' — the part before is the source (HF repo or local dir), the part
    after is the selector for picking a file inside it.
    """
    if _is_pathy(model) and Path(model).exists():
        return ResolvedSource(source=model, selector=None, is_local=True)

    if ":" in model:
        source, selector = model.split(":", 1)
        is_local = _is_pathy(source) and Path(source).exists()
        return ResolvedSource(source=source, selector=selector, is_local=is_local)

    return ResolvedSource(source=model, selector=None, is_local=False)


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


def resolve_model_source(model_ref: str, trust_remote_code: bool = False) -> str:
    """Resolves model_ref to an absolute local path.

    - If local path: validates existence and returns absolute path.
    - If HF repo: downloads/checks cache and returns absolute path.
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
            return str(matches[0].absolute())

        return str(path.absolute())

    # HF Resolve
    try:
        repo_files = list_repo_files(source)
    except Exception as e:
        raise RuntimeError(f"Failed to list files for HF repo {source!r}: {e}") from e

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
                "Selector %r matched %d files in HF repo %r; downloading all shards and returning first %s",
                selector,
                len(matches),
                source,
                matches[0],
            )
            snapshot_dir = snapshot_download(source, allow_patterns=[selector])
            return str(Path(snapshot_dir, matches[0]).absolute())

        # Single match: use hf_hub_download
        return hf_hub_download(source, matches[0])

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

    # Single GGUF in the repo: download it directly and return the file path.
    # llama_server requires a file path, not a directory, so snapshot_download
    # would break it. The implicit "the only GGUF" is unambiguous.
    if len(ggufs) == 1:
        logger.info("HF repo %r has a single GGUF (%s); resolving to its file path", source, ggufs[0])
        return hf_hub_download(source, ggufs[0])

    # Full snapshot with universal filter
    patterns = _select_patterns(repo_files, trust_remote_code=trust_remote_code)
    return snapshot_download(source, allow_patterns=patterns)
