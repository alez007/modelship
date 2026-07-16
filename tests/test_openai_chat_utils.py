import pytest

from modelship.openai.utils.chat import UnsupportedContentError, normalize_chat_messages


def test_string_content_passthrough():
    messages = [{"role": "user", "content": "hello"}]
    assert normalize_chat_messages(messages) == messages


def test_none_content_passthrough():
    messages = [{"role": "assistant", "content": None}]
    assert normalize_chat_messages(messages) == messages


def test_text_only_list_collapses_to_string():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    expected = [{"role": "user", "content": "hello\nworld"}]
    assert normalize_chat_messages(messages) == expected
    assert normalize_chat_messages(messages, supports_image=True) == expected


def test_text_aliases_collapse_to_text():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "a"},
                {"type": "output_text", "text": "b"},
                {"type": "refusal", "refusal": "no"},
                {"type": "thinking", "thinking": "hmm"},
            ],
        }
    ]
    assert normalize_chat_messages(messages) == [{"role": "assistant", "content": "a\nb\nno\nhmm"}]


def test_plain_string_part_treated_as_text():
    messages = [{"role": "user", "content": ["hello", {"type": "text", "text": "world"}]}]
    assert normalize_chat_messages(messages) == [{"role": "user", "content": "hello\nworld"}]


def test_image_rejected_when_not_supported():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ],
        }
    ]
    with pytest.raises(UnsupportedContentError, match="image"):
        normalize_chat_messages(messages, supports_image=False)


def test_image_url_object_passes_through_when_supported():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    result = normalize_chat_messages(messages, supports_image=True)
    assert result[0]["content"][0] == messages[0]["content"][0]
    assert result[0]["content"][1] == {"type": "text", "text": "describe"}


def test_image_url_data_uri():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "what is this"},
            ],
        }
    ]
    result = normalize_chat_messages(messages, supports_image=True)
    assert result[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_image_url_as_bare_string_accepted():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": "https://example.com/x.png"},
                {"type": "text", "text": "what is this"},
            ],
        }
    ]
    result = normalize_chat_messages(messages, supports_image=True)
    assert result[0]["content"][0]["image_url"] == "https://example.com/x.png"


def test_image_url_missing_url_field_rejected():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
    with pytest.raises(UnsupportedContentError, match="url"):
        normalize_chat_messages(messages, supports_image=True)


def test_image_url_empty_string_rejected():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}]
    with pytest.raises(UnsupportedContentError, match="url"):
        normalize_chat_messages(messages, supports_image=True)


def test_image_url_wrong_type_rejected():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": 42}]}]
    with pytest.raises(UnsupportedContentError):
        normalize_chat_messages(messages, supports_image=True)


def test_input_audio_rejected_when_not_supported():
    messages = [
        {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}}],
        }
    ]
    with pytest.raises(UnsupportedContentError, match="audio"):
        normalize_chat_messages(messages, supports_audio=False)


def test_input_audio_passes_through_when_supported():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "AAA", "format": "wav"}},
                {"type": "text", "text": "transcribe"},
            ],
        }
    ]
    result = normalize_chat_messages(messages, supports_audio=True)
    assert result[0]["content"][0] == messages[0]["content"][0]


def test_input_audio_malformed_rejected():
    messages = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AAA"}}]}]
    with pytest.raises(UnsupportedContentError, match="format"):
        normalize_chat_messages(messages, supports_audio=True)


def test_unknown_type_rejected():
    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": "..."}]}]
    with pytest.raises(UnsupportedContentError, match="video_url"):
        normalize_chat_messages(messages)


def test_missing_type_rejected():
    messages = [{"role": "user", "content": [{"text": "hi"}]}]
    with pytest.raises(UnsupportedContentError, match="type"):
        normalize_chat_messages(messages)


def test_part_must_be_dict_or_string():
    messages = [{"role": "user", "content": [123]}]
    with pytest.raises(UnsupportedContentError):
        normalize_chat_messages(messages)


def test_empty_text_part_skipped():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text"},
            ],
        }
    ]
    assert normalize_chat_messages(messages) == [{"role": "user", "content": "hello"}]


def test_empty_image_part_skipped_when_supported():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url"},
            ],
        }
    ]
    assert normalize_chat_messages(messages, supports_image=True) == [{"role": "user", "content": "hi"}]


def test_preserves_other_message_fields():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "abc",
            "content": [{"type": "text", "text": "result"}],
        }
    ]
    assert normalize_chat_messages(messages) == [{"role": "tool", "tool_call_id": "abc", "content": "result"}]


def test_multiple_messages():
    messages = [
        {"role": "system", "content": "you are a bot"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        },
    ]
    assert normalize_chat_messages(messages) == [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "a\nb"},
    ]
