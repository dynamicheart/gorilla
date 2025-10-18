import json
import os
import re
import uuid
import time
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError
from overrides import override

TOOL_CALL_SYSTEM_PROMPT = 'You are a helpful assistant with tool calling capabilities. ' + \
        'When a tool call is needed, you MUST use the following format to issue the call:\n' + \
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>FUNCTION_NAME\n' + \
        '```json\n{"param1": "value1", "param2": "value2"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n\n' + \
        'Make sure the JSON is valid.' + \
        '## Tools\n\n### Function\n\nYou have the following functions available:\n\n'

class SimpleDeepSeekCallExtractor:
    """简化版 DeepSeek V3 function call 提取工具"""

    def __init__(self):
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = (
            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>"
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str):
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if not self.has_tool_call(text):
            return normal_text, []

        call_blocks = re.findall(self.func_call_regex, text, flags=re.DOTALL)
        calls = []

        for block in call_blocks:
            match = re.search(self.func_detail_regex, block, flags=re.DOTALL)
            if not match:
                continue

            func_name = match.group(2).strip()
            func_args_str = match.group(3).strip()
            try:
                func_args = json.loads(func_args_str)
            except json.JSONDecodeError:
                func_args = func_args_str
            calls.append({"name": func_name, "arguments": json.dumps(func_args, ensure_ascii=False)})

        return normal_text, calls


def add_tool_prompt(messages, tools):
    tool_prompt = TOOL_CALL_SYSTEM_PROMPT
    for tool in tools:
        if tool['type'] == 'function':
            function = tool['function']
            function_json_object = {
                'description': function["description"],
                'name': function["name"],
                'parameters': function["parameters"],
                'strict': False,
            }
            function_json_str = json.dumps(function_json_object, ensure_ascii=False)
            tool_prompt += f'- `' + function['name'] + '`:\n```json\n' + function_json_str + '\n```\n'

    if messages[0]["role"] == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + tool_prompt
    # Otherwise, use the system prompt template to create a new system prompt.
    else:
        messages.insert(
            0,
            {"role": "system", "content": tool_prompt},
        )

class DeepSeekAPIHandler(OpenAICompletionsHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.client = OpenAI(
            base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        self.tool_call_parser = SimpleDeepSeekCallExtractor()

    # The deepseek API is unstable at the moment, and will frequently give empty responses, so retry on JSONDecodeError is necessary
    @retry_with_backoff(error_type=[RateLimitError, json.JSONDecodeError])
    def generate_with_backoff(self, **kwargs):
        """
        Per the DeepSeek API documentation:
        https://api-docs.deepseek.com/quick_start/rate_limit

        DeepSeek API does NOT constrain user's rate limit. We will try out best to serve every request.
        But please note that when our servers are under high traffic pressure, you may receive 429 (Rate Limit Reached) or 503 (Server Overloaded). When this happens, please wait for a while and retry.

        Thus, backoff is still useful for handling 429 and 503 errors.
        """
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    @override
    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        # Source https://api-docs.deepseek.com/quick_start/pricing
        # This will need to be updated if newer models are released.
        if "DeepSeek-V3" in self.model_name:
            api_model_name = "deepseek-chat"
        elif "DeepSeek-R1" in self.model_name:
            api_model_name = "deepseek-reasoner"
        else:
            raise ValueError(
                f"Model name {self.model_name} not yet supported in this method"
            )

        if len(tools) > 0:
            add_tool_prompt(message, tools)
            return self.generate_with_backoff(
                model=api_model_name,
                messages=message,
                temperature=self.temperature,
            )
        else:
            return self.generate_with_backoff(
                model=api_model_name,
                messages=message,
                temperature=self.temperature,
            )

    @override
    def _query_prompting(self, inference_data: dict):
        """
        This method is intended to be used by the `DeepSeek-R1` models. If used for other models, you will need to modify the code accordingly.

        Reasoning models don't support temperature parameter
        https://api-docs.deepseek.com/guides/reasoning_model

        `DeepSeek-R1` should use `deepseek-reasoner` as the model name in the API
        https://api-docs.deepseek.com/quick_start/pricing
        """
        message: list[dict] = inference_data["message"]
        inference_data["inference_input_log"] = {"message": repr(message)}

        if "DeepSeek-R1" in self.model_name:
            api_model_name = "deepseek-reasoner"
        else:
            raise ValueError(
                f"Model name {self.model_name} not yet supported in this method"
            )

        return self.generate_with_backoff(
            model=api_model_name,
            messages=message,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        # 'deepseek-reasoner does not support successive user messages, so we need to combine them
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": []}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_prompting(api_response)
        self._add_reasoning_content_if_available_prompting(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """
        DeepSeek does not take reasoning content in next turn chat history, for both prompting and function calling mode.
        Error: Error code: 400 - {'error': {'message': 'The reasoning_content is an intermediate result for display purposes only and will not be included in the context for inference. Please remove the reasoning_content from your message to reduce network traffic.', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}
        """
        response_data = super()._parse_query_response_FC(api_response)
        self._add_reasoning_content_if_available_FC(api_response, response_data)
        return response_data

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        _, tool_calls = self.tool_call_parser.detect_and_parse(api_response.choices[0].message.content)
        if len(tool_calls) > 0:
            model_responses = [
                {func_call['name']: func_call['arguments']} for func_call in tool_calls
            ]
            tool_call_ids = [f"call_{uuid.uuid4().hex[:8]}" for _ in tool_calls]
        else:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }