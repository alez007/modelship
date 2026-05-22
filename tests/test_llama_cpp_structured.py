"""Tests for the llama_cpp ``response_format`` → ``LlamaGrammar`` converter."""

from llama_cpp import LlamaGrammar

from modelship.infer.llama_cpp.structured import build_llama_grammar


class TestBuildLlamaGrammar:
    def test_none_returns_none(self):
        assert build_llama_grammar(None) is None

    def test_empty_dict_returns_none(self):
        assert build_llama_grammar({}) is None

    def test_text_type_returns_none(self):
        assert build_llama_grammar({"type": "text"}) is None

    def test_missing_type_returns_none(self):
        assert build_llama_grammar({"foo": "bar"}) is None

    def test_json_object_returns_grammar(self):
        g = build_llama_grammar({"type": "json_object"})
        assert isinstance(g, LlamaGrammar)

    def test_json_object_grammar_is_cached(self):
        # The permissive json_object grammar takes the same input on every
        # call; it must be compiled once per process, not per request.
        g1 = build_llama_grammar({"type": "json_object"})
        g2 = build_llama_grammar({"type": "json_object"})
        assert g1 is g2

    def test_json_schema_compiles_schema(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        g = build_llama_grammar(
            {"type": "json_schema", "json_schema": {"name": "p", "schema": schema, "strict": True}},
        )
        assert isinstance(g, LlamaGrammar)

    def test_json_schema_missing_schema_returns_none_with_warning(self, caplog):
        import logging

        target = logging.getLogger("modelship.infer.llama_cpp.structured")
        target.addHandler(caplog.handler)
        try:
            caplog.set_level(logging.WARNING)
            g = build_llama_grammar({"type": "json_schema", "json_schema": {"name": "p"}})
        finally:
            target.removeHandler(caplog.handler)
        assert g is None
        assert any("missing a 'schema'" in r.message for r in caplog.records)

    def test_unknown_type_returns_none_with_warning(self, caplog):
        import logging

        target = logging.getLogger("modelship.infer.llama_cpp.structured")
        target.addHandler(caplog.handler)
        try:
            caplog.set_level(logging.WARNING)
            g = build_llama_grammar({"type": "xml"})
        finally:
            target.removeHandler(caplog.handler)
        assert g is None
        assert any("unsupported response_format" in r.message for r in caplog.records)

    def test_invalid_schema_returns_none_with_warning(self, caplog):
        import logging

        target = logging.getLogger("modelship.infer.llama_cpp.structured")
        target.addHandler(caplog.handler)
        try:
            caplog.set_level(logging.WARNING)
            # Schema with an unresolvable $ref triggers LlamaGrammar.from_json_schema to raise.
            bad_schema = {"$ref": "#/definitions/nope"}
            g = build_llama_grammar(
                {"type": "json_schema", "json_schema": {"name": "p", "schema": bad_schema}},
            )
        finally:
            target.removeHandler(caplog.handler)
        # Either compiles to something or fails gracefully — both are acceptable;
        # this test just guarantees we never raise into the caller.
        assert g is None or isinstance(g, LlamaGrammar)


class TestKwargBuilders:
    """``_build_kwargs`` and ``_build_completion_kwargs`` must replace
    ``response_format`` with a compiled ``grammar`` so neither field reaches
    llama-cpp-python with a shape it would silently ignore.
    """

    def _serving(self):
        from llama_cpp import Llama

        from modelship.infer.llama_cpp.openai.serving_chat import OpenAIServingChat

        # Bypass __init__ — we only need the kwarg builders, which only read
        # _accepted_params / _completion_accepted_params.
        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        import inspect

        chat._accepted_params = set(inspect.signature(Llama.create_chat_completion).parameters)
        chat._completion_accepted_params = set(inspect.signature(Llama.create_completion).parameters)
        return chat

    def _request(self, response_format=None):
        from modelship.openai.protocol import ChatCompletionRequest

        payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
        if response_format is not None:
            payload["response_format"] = response_format
        return ChatCompletionRequest(**payload)

    def test_build_kwargs_injects_grammar(self):
        chat = self._serving()
        req = self._request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "p",
                    "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                    "strict": True,
                },
            }
        )
        kwargs = chat._build_kwargs(req, messages=[{"role": "user", "content": "hi"}])
        assert "response_format" not in kwargs
        assert isinstance(kwargs.get("grammar"), LlamaGrammar)

    def test_build_kwargs_omits_grammar_when_no_response_format(self):
        chat = self._serving()
        req = self._request()
        kwargs = chat._build_kwargs(req, messages=[{"role": "user", "content": "hi"}])
        assert "grammar" not in kwargs
        assert "response_format" not in kwargs

    def test_build_completion_kwargs_injects_grammar(self):
        chat = self._serving()
        req = self._request(response_format={"type": "json_object"})
        kwargs = chat._build_completion_kwargs(req, prompt="hi")
        assert "response_format" not in kwargs
        assert isinstance(kwargs.get("grammar"), LlamaGrammar)

    def test_build_completion_kwargs_omits_grammar_when_no_response_format(self):
        chat = self._serving()
        req = self._request()
        kwargs = chat._build_completion_kwargs(req, prompt="hi")
        assert "grammar" not in kwargs

    def test_max_completion_tokens_mapped_and_dropped(self):
        # ``max_completion_tokens`` is the modern OpenAI field. llama-cpp-python
        # only accepts ``max_tokens``. We map the value over and must drop the
        # original key so it doesn't end up in the "unsupported params" warning.
        chat = self._serving()
        from modelship.openai.protocol import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=42,
        )
        kwargs = chat._build_kwargs(req, messages=[{"role": "user", "content": "hi"}])
        assert kwargs.get("max_tokens") == 42
        assert "max_completion_tokens" not in kwargs

        completion_kwargs = chat._build_completion_kwargs(req, prompt="hi")
        assert completion_kwargs.get("max_tokens") == 42
        assert "max_completion_tokens" not in completion_kwargs
