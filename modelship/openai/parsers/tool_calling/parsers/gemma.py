"""Tool call parsers for Google Gemma models.

Gemma 4 uses a custom serialization format (not JSON) for tool calls:
`<|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>`

FunctionGemma (Gemma 2/3 fine-tune) uses:
`<start_function_call>call:func_name{key:<escape>value<escape>}<end_function_call>`

The two formats share the same envelope shape and only differ in the string
delimiter, so both parsers reuse the same custom-syntax parser parameterized
on ``string_delim``.
"""

from __future__ import annotations

import json

from modelship.openai.parsers.tool_calling.parsers.base import ToolCallParser

STRING_DELIM = '<|"|>'
ESCAPE_DELIM = "<escape>"

# Trailing characters withheld from streaming JSON output so a closing
# brace / quote / structural byte never lands ahead of more args bytes.
# The streamer takes a length-diff of successive returns, so anything
# emitted here must be a monotonic prefix of the final clean output.
_STREAM_STRIP_TAIL = ("}", "]", '"', ",", " ", ":", "<", "|", "\\", ">")


class Gemma4ToolCallParser(ToolCallParser):
    name = "gemma4"
    start_marker = "<|tool_call>"
    end_marker = "<tool_call|>"
    markers_are_specials = True
    # String delimiter used inside the call body. Subclasses override
    # this when the family uses a different delimiter token (e.g.
    # FunctionGemma's ``<escape>``).
    string_delim: str = STRING_DELIM

    def extract_partial_name(self, partial_payload: str) -> str | None:
        """Extract function name from 'call:func_name{...'."""
        if not partial_payload.startswith("call:"):
            return None
        # Name ends at the first '{'
        if "{" not in partial_payload:
            return None
        name_part = partial_payload[5 : partial_payload.find("{")]
        return name_part.strip() or None

    def extract_partial_args(self, partial_payload: str, is_complete: bool = False) -> str | None:
        """Parse the custom syntax into a dict and dump it to JSON."""
        if "{" not in partial_payload:
            return None

        raw_args = partial_payload[partial_payload.find("{") + 1 :]
        # Strip trailing '}' if present (it's the structural end of the call)
        if raw_args.endswith("}"):
            raw_args = raw_args[:-1]

        try:
            # ``partial=not is_complete`` tells the parser to withhold
            # ambiguous trailing bytes (unterminated string values, trailing
            # bare values) so partial output stays a monotonic prefix of the
            # eventual final value.
            args_dict = _parse_args(raw_args, partial=not is_complete, string_delim=self.string_delim)
            if not args_dict:
                return "{}" if is_complete else ""

            full_json = json.dumps(args_dict, ensure_ascii=False)

            if is_complete:
                return full_json

            # The streamer takes a suffix-diff. To prevent it from emitting
            # the closing braces of the JSON object prematurely (which might
            # move if more keys arrive), we withhold trailing JSON structural
            # characters during streaming.
            safe_json = full_json
            while safe_json and safe_json[-1] in _STREAM_STRIP_TAIL:
                safe_json = safe_json[:-1]

            return safe_json
        except Exception:
            return ""

    def split_payload(self, payload: str, is_complete: bool) -> list[tuple[str, bool]]:
        """Multiple calls can be concatenated: `call:a{...}call:b{...}`."""
        # Find all occurrences of "call:" at the top level (not inside braces)
        calls: list[tuple[str, bool]] = []
        pos = 0
        n = len(payload)

        while pos < n:
            start = payload.find("call:", pos)
            if start == -1:
                break

            # Find the end of this call. We look for the NEXT "call:" or end of string.
            # But we must be careful not to find "call:" inside a string or nested object.
            # For simplicity, we search for the next "call:" and assume it's a new call
            # if it's preceded by a "}".
            next_call = payload.find("call:", start + 5)
            while next_call != -1:
                # Basic check: a new call should be preceded by a closing brace
                # from the previous call.
                preceding = payload[start:next_call].rstrip()
                if preceding.endswith("}"):
                    break
                next_call = payload.find("call:", next_call + 5)

            if next_call == -1:
                # This is the last call in the payload.
                sub_payload = payload[start:]
                calls.append((sub_payload, is_complete))
                break
            else:
                sub_payload = payload[start:next_call]
                calls.append((sub_payload, True))  # Inner calls are complete
                pos = next_call

        return calls or [(payload, is_complete)]


class FunctionGemmaToolCallParser(Gemma4ToolCallParser):
    """Parses FunctionGemma output which uses `<escape>` instead of `<|"|>`.

    The envelope and string-delim tokens (``<start_function_call>``,
    ``<end_function_call>``, ``<escape>``) are registered as special tokens
    on the FunctionGemma tokenizer, so ``markers_are_specials = True`` so
    the transformers loader keeps them visible to the parser.
    """

    name = "function_gemma"
    start_marker = "<start_function_call>"
    end_marker = "<end_function_call>"
    markers_are_specials = True
    string_delim = ESCAPE_DELIM


def _parse_value(value_str: str) -> object:
    value_str = value_str.strip()
    if not value_str:
        return value_str
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str.lower() in ("null", "none", "nil"):
        return None
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass
    return value_str


def _safe_overlap(buf: str, marker: str) -> int:
    """Index up to which ``buf`` can be flushed without splitting ``marker``.

    Returns the position past the last byte safe to commit; trailing bytes
    that could be a proper prefix of ``marker`` are excluded so a partial
    parse of an unterminated string value never includes transient bytes
    that will disappear once the delim arrives. This keeps partial output
    a monotonic prefix of the eventual final output, which the streamer's
    suffix-diff relies on.
    """
    if not marker:
        return len(buf)
    max_overlap = min(len(buf), len(marker) - 1)
    for k in range(max_overlap, 0, -1):
        if buf.endswith(marker[:k]):
            return len(buf) - k
    return len(buf)


def _parse_args(args_str: str, *, partial: bool = False, string_delim: str = STRING_DELIM) -> dict:
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)
    delim_len = len(string_delim)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        if i >= n:
            if not partial:
                result[key] = ""
            break

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <delim>...<delim>
        if args_str[i:].startswith(string_delim):
            i += delim_len
            val_start = i
            end_pos = args_str.find(string_delim, i)
            if end_pos == -1:
                val = args_str[val_start:]
                if partial:
                    # The closing delim hasn't arrived yet — trim any trailing
                    # bytes that could be a proper prefix of the delim so the
                    # partial value is a prefix of the eventual final value.
                    val = val[: _safe_overlap(val, string_delim)]
                result[key] = val
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + delim_len
        # Nested object
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(string_delim):
                    i += delim_len
                    nd = args_str.find(string_delim, i)
                    i = n if nd == -1 else nd + delim_len
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_args(args_str[obj_start:i], partial=True, string_delim=string_delim)
            else:
                result[key] = _parse_args(args_str[obj_start : i - 1], string_delim=string_delim)
        # Array
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(string_delim):
                    i += delim_len
                    nd = args_str.find(string_delim, i)
                    i = n if nd == -1 else nd + delim_len
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_array(args_str[arr_start:i], partial=True, string_delim=string_delim)
            else:
                result[key] = _parse_array(args_str[arr_start : i - 1], string_delim=string_delim)
        # Bare value
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_value(args_str[val_start:i])

    return result


def _parse_array(arr_str: str, *, partial: bool = False, string_delim: str = STRING_DELIM) -> list:
    items: list = []
    i = 0
    n = len(arr_str)
    delim_len = len(string_delim)
    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        # String value
        if arr_str[i:].startswith(string_delim):
            i += delim_len
            val_start = i
            end_pos = arr_str.find(string_delim, i)
            if end_pos == -1:
                val = arr_str[val_start:]
                if partial:
                    val = val[: _safe_overlap(val, string_delim)]
                items.append(val)
                break
            items.append(arr_str[val_start:end_pos])
            i = end_pos + delim_len
        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(string_delim):
                    i += delim_len
                    nd = arr_str.find(string_delim, i)
                    i = n if nd == -1 else nd + delim_len
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_args(arr_str[obj_start:i], partial=True, string_delim=string_delim))
            else:
                items.append(_parse_args(arr_str[obj_start : i - 1], string_delim=string_delim))
        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            inner_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(string_delim):
                    i += delim_len
                    nd = arr_str.find(string_delim, i)
                    i = n if nd == -1 else nd + delim_len
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_array(arr_str[inner_start:i], partial=True, string_delim=string_delim))
            else:
                items.append(_parse_array(arr_str[inner_start : i - 1], string_delim=string_delim))
        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_value(arr_str[val_start:i]))
    return items
