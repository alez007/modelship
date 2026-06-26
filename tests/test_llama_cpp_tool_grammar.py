"""Tests for the llama_cpp tool-call GBNF grammar builder."""

import json

from llama_cpp import LlamaGrammar

from modelship.infer.llama_cpp.tool_grammar import _MAX_TOOL_CALLS, build_tool_call_gbnf, build_tool_call_grammar
from modelship.openai.parsers.tool_calling import HermesToolCallParser, get_parser

GEMMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "HassTurnOn",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "area": {"enum": ["Living Room", "Small Bedroom"]},
                },
                "required": ["name"],
            },
        },
    },
    {
        # All-optional / no-property intent: must collapse to FUNC{}.
        "type": "function",
        "function": {"name": "HassGetState", "parameters": {"type": "object", "properties": {}}},
    },
    {
        "type": "function",
        "function": {
            "name": "SetTimer",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer"},
                    "on": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
]

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


class TestGemmaToolCallGrammar:
    """The hand-built emitter for the FunctionGemma / Gemma 4 ``call:FUNC{...}`` DSL."""

    def test_function_gemma_returns_grammar(self):
        g = build_tool_call_grammar(get_parser("function_gemma"), GEMMA_TOOLS)
        assert isinstance(g, LlamaGrammar)

    def test_gemma4_returns_grammar(self):
        g = build_tool_call_grammar(get_parser("gemma4"), GEMMA_TOOLS)
        assert isinstance(g, LlamaGrammar)

    def test_envelope_and_markers(self):
        parser = get_parser("function_gemma")
        text = build_tool_call_gbnf(parser, GEMMA_TOOLS)
        assert text is not None
        assert "root ::= ( content )? tool-calls ( content )? | content" in text
        assert f'tool-call ::= "{parser.start_marker}" "call:" call-choice "{parser.end_marker}"' in text
        assert "content ::= [^<]+" in text  # both gemma markers/delims lead with '<'

    def test_call_cap_is_structural(self):
        # The cap is exactly _MAX_TOOL_CALLS calls, concatenated with no separator.
        text = build_tool_call_gbnf(get_parser("function_gemma"), GEMMA_TOOLS)
        assert text is not None
        expected = "tool-calls ::= " + "tool-call" + " ( tool-call )?" * (_MAX_TOOL_CALLS - 1)
        assert expected in text

    def test_every_tool_name_is_a_literal(self):
        text = build_tool_call_gbnf(get_parser("function_gemma"), GEMMA_TOOLS)
        assert text is not None
        for name in ("HassTurnOn", "HassGetState", "SetTimer"):
            assert f'"{name}"' in text
        # All three tools are selectable.
        assert "call-choice ::= tool-0 | tool-1 | tool-2" in text

    def test_enum_values_are_escaped_literals(self):
        # FunctionGemma wraps string values in the <escape> delimiter.
        text = build_tool_call_gbnf(get_parser("function_gemma"), GEMMA_TOOLS)
        assert text is not None
        assert '"<escape>" "Living Room" "<escape>"' in text
        assert '"<escape>" "Small Bedroom" "<escape>"' in text

    def test_gemma4_uses_its_own_delimiter(self):
        # Gemma 4 uses <|"|> rather than <escape>; it appears as an escaped literal.
        text = build_tool_call_gbnf(get_parser("gemma4"), GEMMA_TOOLS)
        assert text is not None
        assert '"<|\\"|>" "Living Room" "<|\\"|>"' in text
        assert "<escape>" not in text

    def test_empty_property_tool_collapses_to_empty_braces(self):
        text = build_tool_call_gbnf(get_parser("function_gemma"), GEMMA_TOOLS)
        assert text is not None
        assert 'tool-1 ::= "HassGetState" "{" "}"' in text

    def test_number_bool_array_value_rules(self):
        text = build_tool_call_gbnf(get_parser("function_gemma"), GEMMA_TOOLS)
        assert text is not None
        assert '"minutes:" ( "-"? [0-9]+ )' in text
        assert '"on:" ( "true" | "false" )' in text
        assert '"[" (' in text  # array rule for `tags`

    def test_all_malformed_tools_returns_none(self):
        tools = [{"type": "function", "function": {}}, {"type": "function"}]
        assert build_tool_call_gbnf(get_parser("function_gemma"), tools) is None
        assert build_tool_call_grammar(get_parser("function_gemma"), tools) is None

    def test_empty_tools_returns_none(self):
        assert build_tool_call_gbnf(get_parser("function_gemma"), []) is None


class TestGemmaGrammarParserAgreement:
    """Canonical strings the grammar can emit must parse back to the right call.

    Guards the emitter and the parser against drifting apart — the grammar
    constrains generation, the parser consumes it, and nothing checks they
    agree except this.
    """

    def test_string_arg_roundtrips(self):
        parser = get_parser("function_gemma")
        sample = f"{parser.start_marker}call:HassTurnOn{{name:{parser.string_delim}small_bedroom_light{parser.string_delim}}}{parser.end_marker}"
        out = parser.parse(sample)
        assert len(out.tool_calls) == 1
        call = out.tool_calls[0]
        assert call.function.name == "HassTurnOn"
        assert json.loads(call.function.arguments) == {"name": "small_bedroom_light"}

    def test_enum_arg_roundtrips(self):
        parser = get_parser("function_gemma")
        d = parser.string_delim
        sample = f"{parser.start_marker}call:HassTurnOn{{name:{d}x{d},area:{d}Living Room{d}}}{parser.end_marker}"
        out = parser.parse(sample)
        assert json.loads(out.tool_calls[0].function.arguments) == {"name": "x", "area": "Living Room"}

    def test_empty_args_roundtrips(self):
        parser = get_parser("function_gemma")
        sample = f"{parser.start_marker}call:HassGetState{{}}{parser.end_marker}"
        out = parser.parse(sample)
        assert out.tool_calls[0].function.name == "HassGetState"
        assert json.loads(out.tool_calls[0].function.arguments) == {}

    def test_number_bool_array_roundtrips(self):
        parser = get_parser("function_gemma")
        d = parser.string_delim
        sample = f"{parser.start_marker}call:SetTimer{{minutes:42,on:true,tags:[{d}a{d},{d}b{d}]}}{parser.end_marker}"
        out = parser.parse(sample)
        assert json.loads(out.tool_calls[0].function.arguments) == {"minutes": 42, "on": True, "tags": ["a", "b"]}
