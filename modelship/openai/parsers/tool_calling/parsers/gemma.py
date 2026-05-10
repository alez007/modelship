"""Tool call parsers for Google Gemma models.

Gemma 4 uses a custom serialization format (not JSON) for tool calls:
`<|tool_call|>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>`

FunctionGemma (Gemma 2) uses:
`<start_function_call>call:func_name{key:<escape>value<escape>}<end_function_call>`
"""

from __future__ import annotations

import json

from modelship.openai.parsers.tool_calling.parsers.base import ToolCallParser

STRING_DELIM = '<|"|>'
ESCAPE_DELIM = "<escape>"


class Gemma4ToolCallParser(ToolCallParser):
    name = "gemma4"
    start_marker = "<|tool_call>"
    end_marker = "<tool_call|>"
    markers_are_specials = True

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
        """Parse Gemma 4 custom format and return standard JSON string."""
        if "{" not in partial_payload:
            return None

        raw_args = partial_payload[partial_payload.find("{") + 1 :]
        # Strip trailing '}' if present (it's the structural end of the call)
        if raw_args.endswith("}"):
            raw_args = raw_args[:-1]

        try:
            # We parse the custom syntax into a dict and dump it to JSON.
            # partial=not is_complete tells the parser to withhold incomplete values
            # at the very end of the stream to avoid flickering types.
            args_dict = _parse_gemma4_args(raw_args, partial=not is_complete)
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
            while safe_json and safe_json[-1] in ("}", "]", '"', ",", " ", ":", "<", "|", "\\", ">"):
                safe_json = safe_json[:-1]

            return safe_json
        except Exception:
            return ""

    def split_payload(self, payload: str, is_complete: bool) -> list[tuple[str, bool]]:
        """Gemma 4 can concatenate multiple calls: `call:a{...}call:b{...}`."""
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
    """Parses FunctionGemma (Gemma 2) output which uses `<escape>` instead of `<|"|>`."""

    name = "function_gemma"
    start_marker = "<start_function_call>"
    end_marker = "<end_function_call>"
    markers_are_specials = False  # These are generally emitted as literal strings or special tokens depending on the specific tuning. We assume text.

    def extract_partial_args(self, partial_payload: str, is_complete: bool = False) -> str | None:
        """Parse FunctionGemma custom format and return standard JSON string."""
        if "{" not in partial_payload:
            return None

        raw_args = partial_payload[partial_payload.find("{") + 1 :]
        # Strip trailing '}' if present
        if raw_args.endswith("}"):
            raw_args = raw_args[:-1]

        try:
            args_dict = _parse_function_gemma_args(raw_args, partial=not is_complete)
            if not args_dict:
                return "{}" if is_complete else ""

            full_json = json.dumps(args_dict, ensure_ascii=False)

            if is_complete:
                return full_json

            safe_json = full_json
            while safe_json and safe_json[-1] in ("}", "]", '"', ",", " ", ":", "<", "|", "\\", ">"):
                safe_json = safe_json[:-1]

            return safe_json
        except Exception:
            return ""


def _parse_gemma4_value(value_str: str) -> object:
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


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    if not args_str or not args_str.strip():
        return {}

    result = {}
    i = 0
    n = len(args_str)

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

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)
        # Nested object
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = args_str.find(STRING_DELIM, i)
                    i = n if nd == -1 else nd + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])
        # Array
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = args_str.find(STRING_DELIM, i)
                    i = n if nd == -1 else nd + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])
        # Bare value
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    items = []
    i = 0
    n = len(arr_str)
    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)
        elif arr_str[i] == "{":
            # ... port object/array recursion if needed, omitting for brevity in initial implementation
            # but keeping basic structure.
            i += 1  # skip { for now
            pass
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))
    return items


def _parse_function_gemma_args(args_str: str, *, partial: bool = False) -> dict:
    """Same as _parse_gemma4_args but uses <escape>."""
    if not args_str or not args_str.strip():
        return {}

    result = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1

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

        if args_str[i:].startswith(ESCAPE_DELIM):
            i += len(ESCAPE_DELIM)
            val_start = i
            end_pos = args_str.find(ESCAPE_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(ESCAPE_DELIM)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result
