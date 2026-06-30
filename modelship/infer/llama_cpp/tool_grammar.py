"""Build a GBNF ``LlamaGrammar`` that constrains tool-call output.

Compiled from a request's ``tools`` and passed to
``llama.create_completion(grammar=...)`` — the same ``grammar`` kwarg
:mod:`structured` uses for ``response_format``. Unlike a ``response_format``
grammar (which forces the whole response to be one JSON value), this grammar's
root permits a free-text answer *or* a bounded sequence of enveloped tool
calls wrapped in optional conversational text, so a non-tool answer and a
chatty preamble before a tool call are both reachable.

JSON-family parsers (Hermes) wrap each call's ``{"name", "arguments"}`` JSON in
literal marker tags, so the body is built by reusing llama.cpp's own
``json_schema_to_gbnf`` and only the envelope is hand-written.

Gemma-family parsers (FunctionGemma / Gemma 4) use a non-JSON DSL —
``call:FUNC{key:<delim>val<delim>,n:42}`` — so ``json_schema_to_gbnf`` can't
express it; that body is hand-built by :func:`_build_gemma_tool_call_gbnf`,
which maps each tool's parameter schema to value rules (enum alternations,
delimited strings, bare numbers/bools, arrays, nested objects). The envelope
caps the number of calls (``_MAX_TOOL_CALLS``) to stop the runaway repetition
small FunctionGemma checkpoints fall into. Other families return ``None``.

The Gemma grammar is loose by design: it pins the call count, tool name, and
enum values. Key uniqueness, ordering, and ``required`` arguments are enforced
at the top level, and only nested-object shape remains loose.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from llama_cpp import LlamaGrammar
from llama_cpp.llama_grammar import json_schema_to_gbnf

from modelship.logging import get_logger

if TYPE_CHECKING:
    from modelship.openai.parsers.tool_calling import ToolCallParser

logger = get_logger("infer.llama_cpp.tool_grammar")

# Parsers whose body is a JSON ``{"name", "arguments"}`` object wrapped in literal
# ``start_marker`` / ``end_marker`` tags — the only shape ``json_schema_to_gbnf``
# plus the envelope below can express. Qwen3-Coder shares Hermes' markers but has
# an XML body, so it's excluded; Mistral / llama3_json are JSON but use a
# different envelope shape, not wired here.
_JSON_FAMILY_PARSERS = frozenset({"hermes"})

# Parsers whose call body is a custom ``call:FUNC{...}`` DSL — handled by the
# hand-built emitter, not ``json_schema_to_gbnf``.
_GEMMA_FAMILY_PARSERS = frozenset({"function_gemma", "gemma4"})

# Max enveloped calls the grammar allows per response. Held at 1 to isolate
# single-command correctness and hardest-stop the 270M runaway; raise to relax.
_MAX_TOOL_CALLS = 1

# Recursion guard for nested object/array schemas — beyond this, value rules
# collapse to the generic recursive value rule (``gv``), which accepts any
# string / number / bool / null / array / object the DSL can express.
_MAX_SCHEMA_DEPTH = 5

# Parser names already logged as unconstrained, to avoid repeating per request.
_logged_unconstrained: set[str] = set()


def _gbnf_literal(s: str) -> str:
    """Render ``s`` as a GBNF double-quoted string literal.

    GBNF accepts the same ``\\"`` / ``\\\\`` escapes JSON uses, so ``json.dumps``
    produces a valid literal; ``ensure_ascii=False`` keeps UTF-8 enum values
    intact rather than emitting ``\\uXXXX`` escapes.
    """
    return json.dumps(s, ensure_ascii=False)


def build_tool_call_gbnf(
    parser: ToolCallParser, tools: list[dict[str, Any]], *, require_tool_call: bool = False
) -> str | None:
    """Assemble the GBNF text for the tool-call grammar, or ``None`` if unsupported.

    Split from compilation so the emitted grammar can be inspected/tested as
    text. Returns ``None`` (logged once per parser) for parsers without an
    emitter and on conversion failure. When ``require_tool_call`` is set the
    root drops its free-text escape, so the model must emit a tool call.
    """
    if not tools:
        return None

    if parser.name in _GEMMA_FAMILY_PARSERS:
        return _build_gemma_tool_call_gbnf(parser, tools, require_tool_call=require_tool_call)

    if parser.name not in _JSON_FAMILY_PARSERS:
        if parser.name not in _logged_unconstrained:
            _logged_unconstrained.add(parser.name)
            logger.info(
                "tool-call grammar: no emitter for tool_call_parser %r; left unconstrained",
                parser.name,
            )
        return None

    # One anyOf branch per tool: ``name`` pinned to the tool's name via ``const``,
    # ``arguments`` constrained to its parameter schema. ``additionalProperties:
    # false`` forbids keys outside the envelope. Entries lacking a function name
    # are skipped — a ``None`` const yields an unsatisfiable branch.
    tool_schemas = []
    for t in tools:
        func = t.get("function")
        if not func or not func.get("name"):
            continue
        tool_schemas.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": func["name"]},
                    "arguments": func.get("parameters") or {"type": "object"},
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    if not tool_schemas:
        return None

    meta_schema = {"anyOf": tool_schemas}

    try:
        inner = json_schema_to_gbnf(json.dumps(meta_schema))
    except Exception as exc:
        logger.warning("tool-call grammar: failed to convert tool schemas to GBNF: %s; skipping", exc)
        return None

    # The converter's entry rule is ``root``; rename only its LHS so our own
    # ``root`` can wrap it. Nothing references it, so a line-anchored sub suffices.
    inner = re.sub(r"^\s*root\s*::=", "tc-json ::=", inner, flags=re.M)

    start = json.dumps(parser.start_marker)
    end = json.dumps(parser.end_marker)

    # Exclude the start marker's first char from `content` so free text yields to a
    # tool call when the marker begins. Escape it if it's special inside a GBNF class.
    start_char = parser.start_marker[0] if parser.start_marker else "<"
    escaped_char = f"\\{start_char}" if start_char in "]-^\\" else start_char

    # Allow optional leading/trailing conversational text and optional whitespace
    # around the JSON envelope, so a chatty prefix can't strand the model in the
    # content branch. `tool-calls` caps at two calls so decoding can't loop forever.
    # `require_tool_call` drops the free-text branch so a tool call is mandatory.
    if require_tool_call:
        root_rule = "root ::= tool-calls\n"
        content_rule = ""
    else:
        root_rule = "root ::= ( content )? tool-calls ( content )? | content\n"
        content_rule = f"content ::= [^{escaped_char}]+\n"
    prefix = (
        root_rule + "tool-calls ::= tool-call ( tc-ws tool-call )?\n"
        f"tool-call ::= {start} tc-ws tc-json tc-ws {end}\n" + content_rule + "tc-ws ::= [ \\t\\n\\r]*\n"
    )
    return prefix + inner


def _build_gemma_tool_call_gbnf(
    parser: ToolCallParser, tools: list[dict[str, Any]], *, require_tool_call: bool = False
) -> str | None:
    """Hand-build the GBNF for a Gemma-family ``call:FUNC{...}`` DSL body.

    ``json_schema_to_gbnf`` only speaks JSON, so the per-tool body is emitted
    directly: a name literal, then a brace-wrapped alternation of ``key:value``
    pairs whose value rules come from :func:`_emit_value`. Returns ``None`` when
    no tool carries a usable function name.
    """
    delim = parser.string_delim or ""
    d_lit = _gbnf_literal(delim)
    # String values exclude the marker/delim lead char ('<' for both families)
    # so a value can't swallow the closing delim or end marker mid-generation.
    excl = (delim or parser.start_marker or "<")[0]
    excl_cls = f"\\{excl}" if excl in "]-^\\" else excl
    str_val = f"( {d_lit} [^{excl_cls}]* {d_lit} )"

    # Generic recursive value rules, emitted only when referenced (free-form
    # objects, unconstrained arrays, schemaless props, or past the depth cap).
    # ``gv`` accepts any value the DSL can express, so the grammar never forces
    # a delimited string where the model legitimately needs ``{...}`` / ``[...]``.
    uses_generic = False

    def generic(ref: str) -> str:
        nonlocal uses_generic
        uses_generic = True
        return ref

    def emit_enum_value(v: object) -> str:
        if isinstance(v, bool):
            return '"true"' if v else '"false"'
        if isinstance(v, int | float):
            return _gbnf_literal(str(v))  # bare, unquoted in the wire format
        if v is None:
            return '"null"'
        return f"{d_lit} {_gbnf_literal(str(v))} {d_lit}"

    def emit_value(schema: object, depth: int = 0) -> str:
        """Return a parenthesized GBNF expression matching one value of ``schema``.

        Always parenthesized so it can sit on either side of a ``|`` without
        leaking precedence into the enclosing alternation.
        """
        if not isinstance(schema, dict) or depth > _MAX_SCHEMA_DEPTH:
            return generic("gv")

        enum = schema.get("enum")
        if enum:
            return "( " + " | ".join(emit_enum_value(v) for v in enum) + " )"

        for combinator in ("anyOf", "oneOf"):
            subs = schema.get(combinator)
            if isinstance(subs, list) and subs:
                return "( " + " | ".join(emit_value(s, depth + 1) for s in subs) + " )"

        t = schema.get("type")
        if isinstance(t, list) and t:
            return "( " + " | ".join(emit_value({**schema, "type": tt}, depth + 1) for tt in t) + " )"
        if t == "string":
            return str_val
        if t == "integer":
            return '( "-"? [0-9]+ )'
        if t == "number":
            return '( "-"? [0-9]+ ( "." [0-9]+ )? )'
        if t == "boolean":
            return '( "true" | "false" )'
        if t == "null":
            return '( "null" )'
        if t == "array":
            items = schema.get("items")
            item = emit_value(items, depth + 1) if isinstance(items, dict) else generic("gv")
            return f'( "[" ( {item} ( "," {item} )* )? "]" )'
        if t == "object":
            props = schema.get("properties")
            if isinstance(props, dict) and props:
                pair = (
                    "( "
                    + " | ".join(f"{_gbnf_literal(f'{k}:')} {emit_value(v, depth + 1)}" for k, v in props.items())
                    + " )"
                )
                return f'( "{{" ( {pair} ( "," {pair} )* )? "}}" )'
            # Free-form object (no/empty properties): accept any DSL object.
            return generic("gv-obj")
        return generic("gv")

    tool_rules: list[str] = []
    n_tools = 0
    for t in tools:
        func = t.get("function")
        if not func or not func.get("name"):
            continue
        idx = n_tools
        n_tools += 1
        name_lit = _gbnf_literal(func["name"])
        params = func.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict) and props:
            prop_items = list(props.items())
            for j, (k, v) in enumerate(prop_items):
                tool_rules.append(f"tool-{idx}-p{j} ::= {_gbnf_literal(f'{k}:')} {emit_value(v)}")
            m = len(prop_items)
            # Ordered-optional args: each property appears at most once, in schema order,
            # with correct commas (first present property is bare, every later one carries a
            # leading comma). Enforces key *uniqueness* — a free ``pair ( "," pair )*``
            # alternation let the model repeat a key (e.g. ``name`` twice) -> corrupt call.
            # Enforces top-level ``required`` fields by restricting branches to only valid
            # ordered subsets that include all top-level required properties.
            req_val = params.get("required")
            if isinstance(req_val, str):
                req_keys = {req_val}
            elif isinstance(req_val, list):
                req_keys = {k for k in req_val if isinstance(k, str)}
            else:
                req_keys = set()

            req_set = {j for j, (k, _) in enumerate(prop_items) if k in req_keys}
            first_req = min(req_set) if req_set else None

            branch_js = range(first_req + 1) if first_req is not None else range(m)

            def later(n: int) -> str:
                return f'"," tool-{idx}-p{n}' if n in req_set else f'( "," tool-{idx}-p{n} )?'  # noqa: B023

            branches = " | ".join(
                " ".join([f"tool-{idx}-p{j}"] + [later(n) for n in range(j + 1, m)]) for j in branch_js
            )
            args_rhs = f"( {branches} )?" if first_req is None else f"( {branches} )"
            tool_rules.append(f"tool-{idx}-args ::= {args_rhs}")
            tool_rules.append(f'tool-{idx} ::= {name_lit} "{{" tool-{idx}-args "}}"')
        else:
            # All-optional / no-property intents collapse to FUNC{}.
            tool_rules.append(f'tool-{idx} ::= {name_lit} "{{" "}}"')

    if n_tools == 0:
        return None

    call_choice = " | ".join(f"tool-{i}" for i in range(n_tools))
    # Calls are concatenated back-to-back (no separator) up to the cap.
    tool_calls_rhs = "tool-call" + " ( tool-call )?" * (_MAX_TOOL_CALLS - 1)

    start = _gbnf_literal(parser.start_marker)
    end = _gbnf_literal(parser.end_marker)
    start_char = parser.start_marker[0] if parser.start_marker else "<"
    content_excl = f"\\{start_char}" if start_char in "]-^\\" else start_char

    if require_tool_call:
        root_rule = "root ::= ws tool-calls ws"
        content_rules: list[str] = []
    else:
        # A turn is *either* tool calls or free text — never text wrapped around a call.
        # The old ``( content )? tool-calls ( content )?`` let a small model spill malformed
        # DSL as leading/trailing content (e.g. ``capturePlayercall:HassMediaNext{}``), which
        # then got captured into chat history and poisoned later turns.
        root_rule = "root ::= ws tool-calls ws | content"
        content_rules = [f"content ::= [^{content_excl}]+"]
    envelope = [
        root_rule,
        f"tool-calls ::= {tool_calls_rhs}",
        f'tool-call ::= {start} "call:" call-choice {end}',
        f"call-choice ::= {call_choice}",
        "ws ::= [ \\t\\n\\r]*",
        *content_rules,
    ]

    generic_rules: list[str] = []
    if uses_generic:
        generic_rules = [
            "gv ::= gv-str | gv-num | gv-bool | gv-null | gv-arr | gv-obj",
            f"gv-str ::= {d_lit} [^{excl_cls}]* {d_lit}",
            'gv-num ::= "-"? [0-9]+ ( "." [0-9]+ )?',
            'gv-bool ::= "true" | "false"',
            'gv-null ::= "null"',
            'gv-arr ::= "[" ( gv ( "," gv )* )? "]"',
            'gv-obj ::= "{" ( gv-pair ( "," gv-pair )* )? "}"',
            'gv-pair ::= gv-key ":" gv',
            # Bare key: anything up to the structural ``:`` / separators / delim lead.
            f"gv-key ::= [^:,{{}}\\[\\]{excl_cls}]+",
        ]

    return "\n".join(envelope + tool_rules + generic_rules) + "\n"


def build_tool_call_grammar(
    parser: ToolCallParser, tools: list[dict[str, Any]], *, require_tool_call: bool = False
) -> LlamaGrammar | None:
    """Compile a tool-call-constraining grammar, or ``None`` if unsupported.

    ``parser`` supplies the envelope markers; ``tools`` is the OpenAI-shaped list
    already resolved for the request. ``require_tool_call`` forces a tool call by
    dropping the free-text root branch. Returns ``None`` for parsers without an
    emitter and on any conversion/compile failure, so the caller falls back to
    unconstrained decoding rather than erroring.
    """
    text = build_tool_call_gbnf(parser, tools, require_tool_call=require_tool_call)
    if text is None:
        return None
    try:
        return LlamaGrammar.from_string(text, verbose=False)
    except Exception as exc:
        logger.warning("tool-call grammar: failed to compile: %s; skipping", exc)
        return None
