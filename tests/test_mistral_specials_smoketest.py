"""Smoke test for the hypothesis that ``[TOOL_CALLS]`` is stripped before our parser sees it.

Background
----------
The ``MistralToolCallParser`` declares ``start_marker = "[TOOL_CALLS]"``.
The transformers loader streams generation through ``TextIteratorStreamer``
with ``skip_special_tokens=True`` ([modelship/infer/transformers/openai/serving_chat.py:221]).

If a Mistral tokenizer registers ``[TOOL_CALLS]`` as an *additional special
token*, then on a real Mistral model run, the marker would be stripped from
the streamed text before reaching our parser — and our parser would never
enter tool-call mode. Streamer-level unit tests don't catch this because
they feed strings directly, bypassing the tokenizer round-trip.

This file confirms or rejects the hypothesis in two parts:

1. **Synthetic** (always runs): build a tokenizer with ``[TOOL_CALLS]``
   registered as an additional special token, encode a sample tool-call
   string, decode with each ``skip_special_tokens`` setting, and verify
   that the stripping behavior is what we feared. This tests the HF
   contract, not Mistral specifically — it tells us whether *any* tokenizer
   that registers the marker as special would lose it on the transformers
   loader path.

2. **Real Mistral** (skipped if the tokenizer can't be loaded): load a
   real Mistral tokenizer and check whether ``[TOOL_CALLS]`` actually
   appears in its specials list. Skipped on missing HF auth or no
   network, so this part is opt-in.
"""

from __future__ import annotations

import os

import pytest


def _sample_tool_call_text() -> str:
    return '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]'


class TestSyntheticAdditionalSpecialTokenStripping:
    """Pure HF behavior check — no network, no auth.

    Take any small public tokenizer, add ``[TOOL_CALLS]`` as an additional
    special token, and confirm that ``skip_special_tokens=True`` strips it
    out of the decoded text. If this passes, the same will happen on any
    real Mistral tokenizer that registers ``[TOOL_CALLS]`` as special —
    which is the second test in this file.
    """

    @pytest.fixture(scope="class")
    def tokenizer_with_tool_calls_special(self):
        from transformers import AutoTokenizer

        # Qwen2.5-0.5B's tokenizer is already used by the project's
        # integration suite (no auth, small download). We only need its
        # vocabulary; we then register `[TOOL_CALLS]` as a special token
        # on top of it to mirror the Mistral configuration.
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        except Exception as e:
            pytest.skip(f"could not load the host tokenizer for the synthetic test: {e}")
        tokenizer.add_special_tokens({"additional_special_tokens": ["[TOOL_CALLS]"]})
        return tokenizer

    def test_roundtrip_keeps_marker_when_skip_special_tokens_false(self, tokenizer_with_tool_calls_special):
        text = _sample_tool_call_text()
        ids = tokenizer_with_tool_calls_special.encode(text, add_special_tokens=False)
        decoded = tokenizer_with_tool_calls_special.decode(ids, skip_special_tokens=False)
        assert "[TOOL_CALLS]" in decoded, (
            "with skip_special_tokens=False the marker MUST round-trip — if this fails, the test setup is wrong"
        )

    def test_roundtrip_strips_marker_when_skip_special_tokens_true(self, tokenizer_with_tool_calls_special):
        """The hypothesis: when ``[TOOL_CALLS]`` is an additional special
        token, ``skip_special_tokens=True`` removes it from the decoded
        text — which is exactly what the transformers loader does today.

        If this assertion holds, the just-merged Mistral parser cannot
        activate on the transformers loader for any tokenizer that
        registers the marker as special. The fix is loader-side
        (per-parser flag flip + noise stripper, deferred from the
        ``llama3_json`` PR).
        """
        text = _sample_tool_call_text()
        ids = tokenizer_with_tool_calls_special.encode(text, add_special_tokens=False)
        decoded = tokenizer_with_tool_calls_special.decode(ids, skip_special_tokens=True)
        assert "[TOOL_CALLS]" not in decoded, (
            "expected `[TOOL_CALLS]` to be stripped by skip_special_tokens=True; "
            f"got decoded={decoded!r}. If this assertion fails, the hypothesis is wrong "
            "and Mistral on the transformers loader is fine as-is."
        )


class TestRealMistralTokenizer:
    """Verify the second half of the hypothesis on a real Mistral tokenizer.

    Skipped unless a Mistral repo is reachable (typically requires
    ``HF_TOKEN`` since ``mistralai/*`` repos are gated).
    """

    @pytest.fixture(scope="class")
    def mistral_tokenizer(self):
        from transformers import AutoTokenizer

        # Try the canonical v0.3 first; fall back to other v3+ Mistral
        # repos that ship the same `[TOOL_CALLS]` marker if v0.3 is
        # unreachable.
        candidates = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-Small-Instruct-2409",
            "mistralai/Mistral-Nemo-Instruct-2407",
        ]
        last_err: Exception | None = None
        for repo in candidates:
            try:
                return AutoTokenizer.from_pretrained(repo)
            except Exception as e:
                last_err = e
        token_present = "set" if os.environ.get("HF_TOKEN") else "unset"
        pytest.skip(
            f"no Mistral tokenizer reachable from {candidates!r} (HF_TOKEN {token_present}); last error: {last_err!r}"
        )

    def test_tool_calls_is_a_special_added_token(self, mistral_tokenizer):
        """Confirm ``[TOOL_CALLS]`` is registered as a special added token.

        Mistral v3+ tokenizers register tool-protocol markers in
        ``added_tokens_decoder`` with ``special=True``, NOT in
        ``all_special_tokens`` (which only carries the bos/eos/unk core).
        ``skip_special_tokens=True`` strips both groups. Checking
        ``all_special_tokens`` alone misses this — the right attribute is
        ``added_tokens_decoder``.
        """
        special_added: dict[int, str] = {
            tid: tok.content for tid, tok in mistral_tokenizer.added_tokens_decoder.items() if tok.special
        }
        assert "[TOOL_CALLS]" in special_added.values(), (
            "hypothesis disproved: `[TOOL_CALLS]` is NOT a special added token on this Mistral tokenizer "
            f"(special_added={special_added!r}); the marker would survive skip_special_tokens=True and "
            "the parser is fine."
        )

    def test_real_tokenizer_strips_marker_with_skip_special_tokens_true(self, mistral_tokenizer):
        """End-to-end confirmation on a real Mistral tokenizer."""
        text = _sample_tool_call_text()
        ids = mistral_tokenizer.encode(text, add_special_tokens=False)
        decoded = mistral_tokenizer.decode(ids, skip_special_tokens=True)
        assert "[TOOL_CALLS]" not in decoded, (
            f"expected `[TOOL_CALLS]` stripped on the real Mistral tokenizer; got decoded={decoded!r}."
        )


class TestLlamaCppHasSameBug:
    """Confirm the same hypothesis on the llama_cpp loader path.

    ``Llama.detokenize`` defaults to ``special=False``
    (``inspect.signature`` proves it), and ``Llama.create_completion``
    calls ``self.detokenize(...)`` without overriding that. So any
    GGUF Mistral model running on the llama_cpp loader will also have
    ``[TOOL_CALLS]`` stripped before our parser sees it.

    This test is a *static* check — we don't need to load a real GGUF
    Mistral to confirm the failure mode. The default value of the
    detokenize parameter is the failure surface.

    Why static and not behavioral: a behavioral test would require
    downloading a Mistral GGUF (multi-GB) and running inference. The
    purpose of the smoke test is to confirm the hypothesis cheaply.
    The transformers fix in this PR addresses the transformers loader;
    the llama_cpp fix is deferred to a follow-up because it requires
    bypassing ``create_completion``'s text-only chunk shape (no token
    IDs are exposed in the chunk dict, so we can't re-detokenize with
    ``special=True`` without restructuring the call site).
    """

    def test_llama_detokenize_defaults_to_stripping_specials(self):
        import inspect

        from llama_cpp import Llama

        sig = inspect.signature(Llama.detokenize)
        special_param = sig.parameters.get("special")
        assert special_param is not None, "Llama.detokenize unexpectedly lacks a `special` parameter"
        assert special_param.default is False, (
            f"hypothesis disproved: Llama.detokenize now defaults to special={special_param.default!r}; "
            "llama_cpp may already preserve special tokens in its streaming text."
        )

    def test_llama_create_completion_does_not_override_special(self):
        """``create_completion`` does not pass ``special=True`` when
        detokenizing. Combined with the default-False above, this means
        ``[TOOL_CALLS]`` is dropped before reaching ``chunk["choices"][0]["text"]``.
        """
        import inspect

        from llama_cpp import Llama

        # Find the streaming generator. Implementation detail name has
        # changed across llama-cpp-python versions, so try both.
        for attr in ("_create_completion", "create_completion"):
            fn = getattr(Llama, attr, None)
            if fn is None:
                continue
            src = inspect.getsource(fn)
            # If `detokenize(` is called anywhere with `special=True`, the
            # bug doesn't apply — but we expect every call site to omit it.
            assert "detokenize(" in src, f"unexpected: {attr} doesn't reference detokenize"
            assert "special=True" not in src, (
                f"hypothesis disproved on {attr}: at least one detokenize call now passes special=True; "
                "llama_cpp may emit specials in streamed text. Re-evaluate the loader fix scope."
            )
            return
        pytest.fail("could not locate the llama-cpp-python streaming completion function")
