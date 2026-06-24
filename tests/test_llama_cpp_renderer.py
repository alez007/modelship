"""Prompt rendering on the llama.cpp loader (LlamaCppToolCallRenderer).

Regression: reasoning chat templates (Qwen3) test ``'</think>' in message.content``,
which raised "argument of type 'NoneType' is not iterable" on replayed assistant
tool-call messages whose content is legitimately ``None``. render() must coerce
None content to "" so the membership test is a safe no-op.
"""

from modelship.infer.llama_cpp.utils import LlamaCppToolCallRenderer

# Mirrors Qwen3's reasoning-strip: a membership test on content that explodes on None.
_REASONING_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- set content = message['content'] %}"
    "{%- if '</think>' in content %}{%- set content = content.split('</think>')[-1] %}{%- endif %}"
    "{{ message['role'] }}: {{ content }}\n"
    "{%- endfor %}"
)


def _renderer() -> LlamaCppToolCallRenderer:
    return LlamaCppToolCallRenderer(
        chat_template=_REASONING_TEMPLATE,
        bos_token="",
        eos_token="",
        _llama=None,  # type: ignore[arg-type]  # render() never touches _llama
    )


def test_render_tolerates_none_content_on_tool_call_message():
    messages = [
        {"role": "user", "content": "turn off the lamps"},
        {
            "role": "assistant",
            "content": None,  # assistant tool-call turn carries null content
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "HassTurnOff", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
    out = _renderer().render(messages, tools=None)  # pre-fix: raised TypeError -> 400
    assert "assistant:" in out
    assert "turn off the lamps" in out


def test_render_does_not_mutate_caller_messages():
    messages = [{"role": "assistant", "content": None}]
    _renderer().render(messages, tools=None)
    assert messages[0]["content"] is None


# Template branching on a chat_template_kwargs variable, mirroring Qwen3's `enable_thinking`.
_THINKING_TEMPLATE = (
    "{%- if enable_thinking is defined and not enable_thinking %}NOTHINK\n{%- endif %}"
    "{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{%- endfor %}"
)


def _thinking_renderer(template_kwargs: dict) -> LlamaCppToolCallRenderer:
    return LlamaCppToolCallRenderer(
        chat_template=_THINKING_TEMPLATE,
        bos_token="",
        eos_token="",
        _llama=None,  # type: ignore[arg-type]  # render() never touches _llama
        template_kwargs=template_kwargs,
    )


def test_render_forwards_template_kwargs():
    messages = [{"role": "user", "content": "hi"}]
    out = _thinking_renderer({"enable_thinking": False}).render(messages, tools=None)
    assert "NOTHINK" in out


def test_render_omits_kwargs_takes_template_default():
    messages = [{"role": "user", "content": "hi"}]
    out = _thinking_renderer({}).render(messages, tools=None)
    assert "NOTHINK" not in out
