# `.\DB-GPT-src\dbgpt\agent\util\llm\llm_client.py`

```py
# 导入所需的模块
import json
import logging
import traceback
from typing import Callable, Dict, Optional, Union
# 导入 LLMClient 类
from dbgpt.core import LLMClient
# 导入 BaseOutputParser 类
from dbgpt.core.interface.output_parser import BaseOutputParser
# 导入 LLMChatError 类
from dbgpt.util.error_types import LLMChatError
# 导入 root_tracer 函数
from dbgpt.util.tracer import root_tracer

# 设置日志记录器
logger = logging.getLogger(__name__)

# 定义 AIWrapper 类
class AIWrapper:
    """AIWrapper for LLM."""

    # 缓存路径根目录
    cache_path_root: str = ".cache"
    # 额外的参数
    extra_kwargs = {
        "cache_seed",
        "filter_func",
        "allow_format_str_template",
        "context",
        "llm_model",
    }

    # 初始化方法
    def __init__(
        self, llm_client: LLMClient, output_parser: Optional[BaseOutputParser] = None
    ):
        """Create an AIWrapper instance."""
        # 是否打印 LLM 的输出
        self.llm_echo = False
        # 是否启用模型缓存
        self.model_cache_enable = False
        # LLM 客户端
        self._llm_client = llm_client
        # 输出解析器
        self._output_parser = output_parser or BaseOutputParser(is_stream_out=False)

    # 实例化方法
    @classmethod
    def instantiate(
        cls,
        template: Optional[Union[str, Callable]] = None,
        context: Optional[Dict] = None,
        allow_format_str_template: Optional[bool] = False,
    ):
        """Instantiate the template with the context."""
        # 如果上下文为空或模板为空，则返回模板
        if not context or template is None:
            return template
        # 如果模板是字符串类型，则根据上下文格式化模板，如果允许格式化字符串模板，则返回格式化后的模板，否则返回原模板
        if isinstance(template, str):
            return template.format(**context) if allow_format_str_template else template
        # 如果模板是可调用对象，则传入上下文返回结果
        return template(context)
    # 构造创建参数的方法，根据给定的创建配置和额外参数生成一个参数字典
    def _construct_create_params(self, create_config: Dict, extra_kwargs: Dict) -> Dict:
        """Prime the create_config with additional_kwargs."""
        # 从创建配置中获取提示和消息列表
        prompt = create_config.get("prompt")
        messages = create_config.get("messages")
        # 如果提示和消息列表都为空，则抛出数值错误
        if prompt is None and messages is None:
            raise ValueError(
                "Either prompt or messages should be in create config but not both."
            )

        # 从额外参数中获取上下文信息
        context = extra_kwargs.get("context")
        # 如果上下文信息为空，则不需要实例化，直接返回创建配置
        if context is None:
            return create_config

        # 从额外参数中获取是否允许格式化字符串模板的标志，默认为 False
        allow_format_str_template = extra_kwargs.get("allow_format_str_template", False)
        
        # 复制创建配置，以防修改原始对象
        params = create_config.copy()
        
        # 如果存在提示，则实例化提示内容
        if prompt is not None:
            params["prompt"] = self.instantiate(
                prompt, context, allow_format_str_template
            )
        # 如果同时存在上下文、消息列表，并且消息列表是一个列表，则逐个实例化消息内容
        elif context and messages and isinstance(messages, list):
            params["messages"] = [
                (
                    {
                        **m,
                        "content": self.instantiate(
                            m["content"], context, allow_format_str_template
                        ),
                    }
                    if m.get("content")
                    else m
                )
                for m in messages
            ]
        
        return params

    # 将配置分离为创建配置和额外参数的方法
    def _separate_create_config(self, config):
        """Separate the config into create_config and extra_kwargs."""
        # 从配置中提取不包含在额外参数中的键值对，作为创建配置
        create_config = {k: v for k, v in config.items() if k not in self.extra_kwargs}
        # 从配置中提取额外参数中包含的键值对
        extra_kwargs = {k: v for k, v in config.items() if k in self.extra_kwargs}
        return create_config, extra_kwargs

    # 获取配置的唯一标识符的方法，用作字典的键
    def _get_key(self, config):
        """Get a unique identifier of a configuration.

        Args:
            config (dict or list): A configuration.

        Returns:
            tuple: A unique identifier which can be used as a key for a dict.
        """
        # 定义不包含在缓存键中的关键字列表
        non_cache_key = ["api_key", "base_url", "api_type", "api_version"]
        copied = False
        # 遍历非缓存键列表
        for key in non_cache_key:
            # 如果配置中包含当前遍历的键，则复制配置以防止修改原始对象
            if key in config:
                config, copied = config.copy() if not copied else config, True
                # 移除当前遍历的键
                config.pop(key)
        # 返回经过 JSON 序列化后的配置，确保排序键和非 ASCII 字符的输出
        return json.dumps(config, sort_keys=True, ensure_ascii=False)
    async def create(self, verbose: bool = False, **config) -> Optional[str]:
        """Create a response from the input config."""
        # 合并输入的配置参数与配置列表中的第i个配置
        full_config = {**config}
        # 将配置分离为create_config和extra_kwargs
        create_config, extra_kwargs = self._separate_create_config(full_config)

        # 构建创建参数
        params = self._construct_create_params(create_config, extra_kwargs)
        # 获取额外的过滤函数、上下文和语言模型
        filter_func = extra_kwargs.get("filter_func")
        context = extra_kwargs.get("context")
        llm_model = extra_kwargs.get("llm_model")
        try:
            # 调用_completions_create方法生成响应
            response = await self._completions_create(llm_model, params, verbose)
        except LLMChatError as e:
            # 若生成失败，记录错误并重新抛出异常
            logger.debug(f"{llm_model} generate failed!{str(e)}")
            raise e
        else:
            pass_filter = filter_func is None or filter_func(
                context=context, response=response
            )
            if pass_filter:
                # 若通过过滤器，则返回响应
                return response
            else:
                return None

    def _get_span_metadata(self, payload: Dict) -> Dict:
        # 将payload中的键值对复制到metadata中
        metadata = {k: v for k, v in payload.items()}

        # 将metadata中的"messages"转换为字典列表
        metadata["messages"] = list(
            map(lambda m: m if isinstance(m, dict) else m.dict(), metadata["messages"])
        )
        return metadata

    def _llm_messages_convert(self, params):
        gpts_messages = params["messages"]
        # TODO

        return gpts_messages

    async def _completions_create(
        self, llm_model, params, verbose: bool = False
    ) -> str:
        # 构建调用语言模型所需的请求 payload
        payload = {
            "model": llm_model,  # 设定语言模型名称
```