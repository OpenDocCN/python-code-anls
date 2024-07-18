# `.\graphrag\graphrag\llm\openai\openai_chat_llm.py`

```py
# 从 logging 模块中导入日志记录功能
import logging
# 从 json 模块中导入 JSONDecodeError 异常类
from json import JSONDecodeError

# 从 typing_extensions 模块中导入 Unpack 类型
from typing_extensions import Unpack

# 从 graphrag.llm.base 模块中导入 BaseLLM 类
from graphrag.llm.base import BaseLLM
# 从 graphrag.llm.types 模块中导入 CompletionInput 和 CompletionOutput 类型
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)

# 从当前包中的 _json 模块导入 clean_up_json 函数
from ._json import clean_up_json
# 从当前包中的 _prompts 模块导入 JSON_CHECK_PROMPT 常量
from ._prompts import JSON_CHECK_PROMPT
# 从当前包中的 openai_configuration 模块导入 OpenAIConfiguration 类
from .openai_configuration import OpenAIConfiguration
# 从当前包中的 types 模块导入 OpenAIClientTypes 类型
from .types import OpenAIClientTypes
# 从当前包中的 utils 模块导入 get_completion_llm_args 和 try_parse_json_object 函数
from .utils import (
    get_completion_llm_args,
    try_parse_json_object,
)

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 最大生成尝试次数
_MAX_GENERATION_RETRIES = 3
# JSON 生成失败的错误消息
FAILED_TO_CREATE_JSON_ERROR = "Failed to generate valid JSON output"

# Chat-based LLM 类，继承自 BaseLLM 类，接受 CompletionInput 和 CompletionOutput 类型参数
class OpenAIChatLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A Chat-based LLM."""

    # _client 属性，类型为 OpenAIClientTypes
    _client: OpenAIClientTypes
    # _configuration 属性，类型为 OpenAIConfiguration
    _configuration: OpenAIConfiguration

    # 初始化方法，接受 client 和 configuration 参数
    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client  # 设置实例的 client 属性
        self.configuration = configuration  # 设置实例的 configuration 属性

    # 异步方法，执行 LLM
    async def _execute_llm(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> CompletionOutput | None:
        # 获取完成 LLM 所需的参数
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        # 获取历史消息或者使用空列表
        history = kwargs.get("history") or []
        # 创建消息列表，包括历史消息和当前输入
        messages = [
            *history,
            {"role": "user", "content": input},
        ]
        # 调用 OpenAI 客户端创建聊天完成结果
        completion = await self.client.chat.completions.create(
            messages=messages, **args
        )
        # 返回第一个选择的消息内容
        return completion.choices[0].message.content

    # 异步方法，调用生成 JSON 输出
    async def _invoke_json(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        """Generate JSON output."""
        # 获取或设置名称参数，默认为 "unknown"
        name = kwargs.get("name") or "unknown"
        # 获取或设置是否响应有效的函数，默认为始终返回 True
        is_response_valid = kwargs.get("is_response_valid") or (lambda _x: True)

        # 嵌套的异步生成函数
        async def generate(
            attempt: int | None = None,
        ) -> LLMOutput[CompletionOutput]:
            # 设置调用名称
            call_name = name if attempt is None else f"{name}@{attempt}"
            # 根据模型是否支持 JSON 选择调用原生 JSON 或手动 JSON 生成方法
            return (
                await self._native_json(input, **{**kwargs, "name": call_name})
                if self.configuration.model_supports_json
                else await self._manual_json(input, **{**kwargs, "name": call_name})
            )

        # 判断 JSON 是否有效的函数
        def is_valid(x: dict | None) -> bool:
            return x is not None and is_response_valid(x)

        # 第一次尝试生成 JSON 结果
        result = await generate()
        retry = 0
        # 当结果不合法且重试次数小于最大尝试次数时，进行重试
        while not is_valid(result.json) and retry < _MAX_GENERATION_RETRIES:
            result = await generate(retry)
            retry += 1

        # 如果结果合法，返回结果；否则抛出运行时错误
        if is_valid(result.json):
            return result
        raise RuntimeError(FAILED_TO_CREATE_JSON_ERROR)

    # 异步方法，使用原生 JSON 方法生成结果
    async def _native_json(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    # 生成 JSON 输出，利用模型的原生 JSON 输出支持
    async def _generate_json_output(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        # 调用内部方法执行模型推理
        result = await self._invoke(
            input,
            **{
                **kwargs,
                "model_parameters": {
                    **(kwargs.get("model_parameters") or {}),
                    "response_format": {"type": "json_object"},
                },
            },
        )
    
        # 获取原始输出，如果为 None 则为空字符串
        raw_output = result.output or ""
        # 尝试解析原始输出为 JSON 对象
        json_output = try_parse_json_object(raw_output)
    
        # 返回包含原始输出、JSON 输出和历史记录的 LLMOutput 对象
        return LLMOutput[CompletionOutput](
            output=raw_output,
            json=json_output,
            history=result.history,
        )
    
    # 尝试手动处理 JSON 输出
    async def _manual_json(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        # 执行模型推理
        result = await self._invoke(input, **kwargs)
        # 获取历史记录，如果为 None 则为空列表
        history = result.history or []
        # 清理输出并尝试解析为 JSON
        output = clean_up_json(result.output or "")
        try:
            # 尝试解析清理后的输出为 JSON 对象
            json_output = try_parse_json_object(output)
            # 返回包含输出、JSON 输出和历史记录的 LLMOutput 对象
            return LLMOutput[CompletionOutput](
                output=output, json=json_output, history=history
            )
        except (TypeError, JSONDecodeError):
            # 若清理后的 JSON 无法解析，记录警告信息并尝试使用 LLM 重新格式化
            log.warning("error parsing llm json, retrying")
            result = await self._try_clean_json_with_llm(output, **kwargs)
            output = clean_up_json(result.output or "")
            json = try_parse_json_object(output)
    
            # 返回包含重新格式化后的输出、JSON 输出和历史记录的 LLMOutput 对象
            return LLMOutput[CompletionOutput](
                output=output,
                json=json,
                history=history,
            )
    
    # 尝试使用 LLM 重新格式化 JSON 输出
    async def _try_clean_json_with_llm(
        self, output: str, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        # 获取名称参数，默认为 "unknown"
        name = kwargs.get("name") or "unknown"
        # 调用内部方法执行 JSON 校验提示
        return await self._invoke(
            JSON_CHECK_PROMPT,
            **{
                **kwargs,
                "variables": {"input_text": output},
                "name": f"fix_json@{name}",
            },
        )
```