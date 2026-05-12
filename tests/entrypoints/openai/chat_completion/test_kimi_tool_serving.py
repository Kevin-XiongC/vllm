# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vllm import envs
from vllm.entrypoints.chat_utils import get_tool_call_id_type
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

MODEL_NAME = "openai-community/gpt2"


class KimiParamValidator(OpenAIServingChat):
    def __init__(self):
        self.tool_call_id_type = "kimi_k2"


def _build_serving_chat() -> KimiParamValidator:
    return KimiParamValidator()


@pytest.fixture
def kimi_chat(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NOVITA_ENABLE_KIMI_VALIDATIONS", "1")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    return _build_serving_chat()


def _request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "hello"}],
        **kwargs,
    )


def test_kimi_tool_call_id_type_is_model_type_driven():
    kimi_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="kimi_k2"),
        hf_overrides=None,
    )
    non_kimi_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="any"),
        hf_overrides=None,
    )

    assert get_tool_call_id_type(kimi_config) == "kimi_k2"
    assert get_tool_call_id_type(non_kimi_config) == "random"


def test_kimi_param_validation_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    kimi_chat: OpenAIServingChat,
):
    monkeypatch.delenv("NOVITA_ENABLE_KIMI_VALIDATIONS", raising=False)
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()
    req = _request(temperature=1.5, top_p=0.8)

    result = kimi_chat._validate_kimi_params(req)

    assert result is None
    assert req.temperature == 1.5
    assert req.top_p == 0.8
    assert req.chat_template_kwargs is None


def test_kimi_param_defaults_follow_thinking_mode(kimi_chat: OpenAIServingChat):
    req = _request()

    result = kimi_chat._validate_kimi_params(req)

    assert result is None
    assert req.temperature == 1.0
    assert req.top_p == 0.95
    assert req.chat_template_kwargs == {
        "thinking": True,
        "enable_thinking": True,
    }


def test_kimi_non_thinking_defaults(kimi_chat: OpenAIServingChat):
    req = _request(chat_template_kwargs={"thinking": False})

    result = kimi_chat._validate_kimi_params(req)

    assert result is None
    assert req.temperature == 0.6
    assert req.top_p == 0.95
    assert req.chat_template_kwargs["enable_thinking"] is False


@pytest.mark.parametrize("temperature", [0.0, 0.3, 0.6, 1.0])
def test_kimi_custom_temperature_in_unit_interval_allowed(
    kimi_chat: OpenAIServingChat,
    temperature: float,
):
    req = _request(temperature=temperature)

    result = kimi_chat._validate_kimi_params(req)

    assert result is None
    assert req.temperature == temperature


def test_kimi_temperature_above_one_rejected(kimi_chat: OpenAIServingChat):
    req = _request(temperature=1.5)

    result = kimi_chat._validate_kimi_params(req)

    assert isinstance(result, ErrorResponse)
    assert "temperature must be between 0 and 1" in result.error.message


def test_kimi_common_param_errors_are_reported_together(
    kimi_chat: OpenAIServingChat,
):
    req = _request(temperature=1.5, top_p=0.8, presence_penalty=1.0, n=2)

    result = kimi_chat._validate_kimi_params(req)

    assert isinstance(result, ErrorResponse)
    assert "temperature" in result.error.message
    assert "top_p" in result.error.message
    assert "presence_penalty" in result.error.message
    assert "n must be 1" in result.error.message


def test_interleaved_thinking_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NOVITA_ENABLE_KIMI_VALIDATIONS", raising=False)
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        chat_template_kwargs={"thinking": True},
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "functions.get_weather:0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "functions.get_weather:0",
                "content": "sunny",
            },
        ],
    )

    assert request.messages[-1]["role"] == "tool"


def test_interleaved_thinking_requires_reasoning_before_tool_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NOVITA_ENABLE_KIMI_VALIDATIONS", "1")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()

    with pytest.raises(ValidationError, match="Interleaved thinking required"):
        ChatCompletionRequest(
            model=MODEL_NAME,
            chat_template_kwargs={"thinking": True},
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "functions.get_weather:0",
                    "content": "sunny",
                },
            ],
        )


def test_interleaved_thinking_accepts_reasoning_before_tool_results(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NOVITA_ENABLE_KIMI_VALIDATIONS", "1")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()

    request = ChatCompletionRequest(
        model=MODEL_NAME,
        chat_template_kwargs={"thinking": True},
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "reasoning": "I should call the weather tool.",
                "tool_calls": [
                    {
                        "id": "functions.get_weather:0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "functions.get_weather:0",
                "content": "sunny",
            },
        ],
    )

    assert request.messages[1]["reasoning"] == "I should call the weather tool."
