"""Tests for the llama_cpp tool-call GBNF grammar builder."""

import json

from llama_cpp import LlamaGrammar

from modelship.infer.llama_cpp.tool_grammar import build_tool_call_gbnf, build_tool_call_grammar
from modelship.openai.parsers.tool_calling import HermesToolCallParser, get_parser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "parameters": {
                "type": "object",
                "properties": {"area": {"type": "string"}},
                "required": ["area"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "HassTurnOff",
            "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
        },
    },
]


class TestBuildToolCallGrammar:
    def test_hermes_returns_grammar(self):
        g = build_tool_call_grammar(HermesToolCallParser(), TOOLS)
        assert isinstance(g, LlamaGrammar)

    def test_hermes_grammar_text_has_envelope_and_consts(self):
        parser = HermesToolCallParser()
        text = build_tool_call_gbnf(parser, TOOLS)
        assert text is not None
        # Top-level rules and the renamed inner entry rule. The root allows
        # optional conversational text around the tool calls.
        assert "root ::= ( content )? tool-calls ( content )? | content" in text
        assert "tc-ws ::= [ \\t\\n\\r]*" in text
        assert "tc-json ::=" in text
        assert "root ::= alternative" not in text  # inner root was renamed
        # Envelope markers and the tool names appear as GBNF string literals
        # (consts are escaped inside the GBNF, so match the bare name).
        assert parser.start_marker in text
        assert parser.end_marker in text
        assert "HassTurnOn" in text
        assert "HassTurnOff" in text

    def test_empty_tools_returns_none(self):
        assert build_tool_call_grammar(HermesToolCallParser(), []) is None
        assert build_tool_call_gbnf(HermesToolCallParser(), []) is None

    def test_function_gemma_returns_none(self):
        # Custom-syntax family has no emitter yet.
        assert build_tool_call_grammar(get_parser("function_gemma"), TOOLS) is None

    def test_qwen3_coder_returns_none(self):
        # Shares Hermes markers but an XML body — must not be treated as JSON-family.
        assert build_tool_call_grammar(get_parser("qwen3_coder"), TOOLS) is None

    def test_fresh_instance_per_call(self):
        # LlamaGrammar wraps stateful C-side state; never share across requests.
        g1 = build_tool_call_grammar(HermesToolCallParser(), TOOLS)
        g2 = build_tool_call_grammar(HermesToolCallParser(), TOOLS)
        assert g1 is not g2

    def test_tool_without_parameters_still_builds(self):
        tools = [{"type": "function", "function": {"name": "Now"}}]
        g = build_tool_call_grammar(HermesToolCallParser(), tools)
        assert isinstance(g, LlamaGrammar)

    def test_malformed_tools_are_filtered(self):
        # Entries without a usable function name are dropped, but valid ones
        # still build (a None const would be unsatisfiable).
        tools = [
            {"type": "function", "function": {}},  # no name
            {"type": "function"},  # no function body
            {"type": "function", "function": {"name": "HassTurnOn"}},
        ]
        text = build_tool_call_gbnf(HermesToolCallParser(), tools)
        assert text is not None
        assert "HassTurnOn" in text
        g = build_tool_call_grammar(HermesToolCallParser(), tools)
        assert isinstance(g, LlamaGrammar)

    def test_all_malformed_tools_returns_none(self):
        tools = [{"type": "function", "function": {}}, {"type": "function"}]
        assert build_tool_call_gbnf(HermesToolCallParser(), tools) is None
        assert build_tool_call_grammar(HermesToolCallParser(), tools) is None

    def test_content_rule_excludes_start_marker_first_char(self):
        # The content exclusion is derived from the marker's first char so a
        # free-text answer yields to the tool call once the marker begins.
        text = build_tool_call_gbnf(HermesToolCallParser(), TOOLS)
        assert text is not None
        assert "content ::= [^<]+" in text  # hermes start marker is "<tool_call>"


class TestGrammarShapedOutputRoundTrips:
    """A sample matching the grammar's envelope must parse back to the tools."""

    def test_hermes_roundtrip(self):
        parser = HermesToolCallParser()
        sample = (
            f'{parser.start_marker}\n{{"name": "HassTurnOn", "arguments": {{"area": "kitchen"}}}}\n{parser.end_marker}'
        )
        out = parser.parse(sample)
        assert out.content is None
        assert len(out.tool_calls) == 1
        call = out.tool_calls[0]
        assert call.function.name == "HassTurnOn"
        assert json.loads(call.function.arguments) == {"area": "kitchen"}

    def test_hermes_roundtrip_two_calls(self):
        parser = HermesToolCallParser()
        sample = (
            f"{parser.start_marker}\n"
            '{"name": "HassTurnOn", "arguments": {"area": "kitchen"}}\n'
            f"{parser.end_marker}\n"
            f"{parser.start_marker}\n"
            '{"name": "HassTurnOff", "arguments": {"area": "bedroom"}}\n'
            f"{parser.end_marker}"
        )
        out = parser.parse(sample)
        assert [c.function.name for c in out.tool_calls] == ["HassTurnOn", "HassTurnOff"]
