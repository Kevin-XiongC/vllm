# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GLM-4.7 Tool Call Parser.

GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
  - The function name may appear on the same line as ``<tool_call>`` without
    a newline separator before the first ``<arg_key>``.
  - Tool calls may have zero arguments
    (e.g. ``<tool_call>func</tool_call>``).

This parser overrides the parent regex patterns to handle both formats.
"""

import json
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = init_logger(__name__)


class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # GLM-4.7 format: <tool_call>func_name[<arg_key>...]*</tool_call>
        # The function name can be followed by a newline, whitespace, or
        # directly by <arg_key> tags (no separator).  The arg section is
        # optional so that zero-argument calls are supported.
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(\S+?)\s*(<arg_key>.*)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if not request.tools or request.tool_choice == "none":
            return request

        if request.structured_outputs is None:
            tag_json = self._build_structural_tag(request.tools)
            if tag_json is not None:
                request.structured_outputs = StructuredOutputsParams(
                    structural_tag=tag_json
                )

        request.skip_special_tokens = False
        return request

    def _build_structural_tag(self, tools) -> str | None:
        """Build structural_tag for xgrammar guided decoding."""
        tags = []
        for tool in tools:
            name = tool.function.name
            params = tool.function.parameters or {
                "type": "object",
                "properties": {},
            }
            properties = params.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            elements: list[dict[str, Any]] = []
            for param_name, param_schema in properties.items():
                if not isinstance(param_schema, dict):
                    param_schema = {}
                elements.append(
                    {
                        "type": "const_string",
                        "value": f"<arg_key>{param_name}</arg_key><arg_value>",
                    }
                )
                if self._is_string_type(name, param_name, tools):
                    elements.append(
                        {
                            "type": "regex",
                            "pattern": r".*",
                        }
                    )
                else:
                    elements.append(
                        {
                            "type": "json_schema",
                            "json_schema": param_schema,
                        }
                    )
                elements.append(
                    {
                        "type": "const_string",
                        "value": "</arg_value>",
                    }
                )

            tags.append(
                {
                    "type": "tag",
                    "begin": f"<tool_call>{name}",
                    "content": {
                        "type": "sequence",
                        "elements": elements,
                    },
                    "end": "</tool_call>",
                }
            )

        if not tags:
            return None

        structural_tag = {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>"],
                "tags": tags,
                "at_least_one": False,
                "stop_after_first": False,
            },
        }
        return json.dumps(structural_tag)
