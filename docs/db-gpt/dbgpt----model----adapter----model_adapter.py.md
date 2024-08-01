# `.\DB-GPT-src\dbgpt\model\adapter\model_adapter.py`

```py
# 从未来版本导入注解功能
from __future__ import annotations

# 导入日志记录模块
import logging
# 导入操作系统相关功能
import os
# 导入线程相关功能
import threading
# 导入缓存装饰器
from functools import cache
# 导入类型提示相关功能
from typing import List, Optional, Type

# 导入自定义模块：LLM 模型适配器和获取模型适配器函数
from dbgpt.model.adapter.base import LLMModelAdapter, get_model_adapter
# 导入自定义模块：对话适配器和对话适配器工厂
from dbgpt.model.adapter.template import ConversationAdapter, ConversationAdapterFactory
# 导入自定义模块：模型类型枚举
from dbgpt.model.base import ModelType
# 导入自定义模块：基础模型参数
from dbgpt.model.parameter import BaseModelParameters

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 创建线程本地存储对象
thread_local = threading.local()
# 检查是否启用性能基准测试
_IS_BENCHMARK = os.getenv("DB_GPT_MODEL_BENCHMARK", "False").lower() == "true"

# 已废弃的模型列表
_OLD_MODELS = [
    "llama-cpp",
    # "proxyllm",
    "gptj-6b",
    "codellama-13b-sql-sft",
    "codellama-7b",
    "codellama-7b-sql-sft",
    "codellama-13b",
]

# 使用缓存装饰器定义函数：获取 LLM 模型适配器
@cache
def get_llm_model_adapter(
    model_name: str,
    model_path: str,
    use_fastchat: bool = True,
    use_fastchat_monkey_patch: bool = False,
    model_type: str = None,
) -> LLMModelAdapter:
    # 创建默认对话适配器工厂实例
    conv_factory = DefaultConversationAdapterFactory()

    # 如果模型类型为 VLLM，则返回 VLLMModelAdapterWrapper 适配器
    if model_type == ModelType.VLLM:
        logger.info("Current model type is vllm, return VLLMModelAdapterWrapper")
        from dbgpt.model.adapter.vllm_adapter import VLLMModelAdapterWrapper
        return VLLMModelAdapterWrapper(conv_factory)

    # 导入 NewHFChatModelAdapter，以便注册
    from dbgpt.model.adapter.hf_adapter import NewHFChatModelAdapter
    from dbgpt.model.adapter.proxy_adapter import ProxyLLMModelAdapter

    # 获取指定模型的新适配器
    new_model_adapter = get_model_adapter(
        model_type, model_name, model_path, conv_factory
    )
    if new_model_adapter:
        logger.info(f"Current model {model_name} use new adapter {new_model_adapter}")
        return new_model_adapter

    # 检查模型名是否包含在已废弃模型列表中
    must_use_old = any(m in model_name for m in _OLD_MODELS)
    result_adapter: Optional[LLMModelAdapter] = None

    # 如果使用快速聊天且不必使用旧模型，则使用快速聊天适配器
    if use_fastchat and not must_use_old:
        logger.info("Use fastcat adapter")
        from dbgpt.model.adapter.fschat_adapter import (
            FastChatLLMModelAdapterWrapper,
            _fastchat_get_adapter_monkey_patch,
            _get_fastchat_model_adapter,
        )

        # 获取快速聊天适配器
        adapter = _get_fastchat_model_adapter(
            model_name,
            model_path,
            _fastchat_get_adapter_monkey_patch,
            use_fastchat_monkey_patch=use_fastchat_monkey_patch,
        )
        if adapter:
            result_adapter = FastChatLLMModelAdapterWrapper(adapter)

    # 否则使用旧的 DB-GPT 适配器
    else:
        from dbgpt.app.chat_adapter import get_llm_chat_adapter
        from dbgpt.model.adapter.old_adapter import OldLLMModelAdapterWrapper
        from dbgpt.model.adapter.old_adapter import (
            get_llm_model_adapter as _old_get_llm_model_adapter,
        )

        logger.info("Use DB-GPT old adapter")
        result_adapter = OldLLMModelAdapterWrapper(
            _old_get_llm_model_adapter(model_name, model_path),
            get_llm_chat_adapter(model_name, model_path),
        )
    # 如果存在结果适配器对象，则更新其模型名称、模型路径和会话工厂，并返回适配器对象
    if result_adapter:
        result_adapter.model_name = model_name
        result_adapter.model_path = model_path
        result_adapter.conv_factory = conv_factory
        return result_adapter
    # 如果不存在结果适配器对象，则抛出数值错误，指明找不到相应模型的适配器
    else:
        raise ValueError(f"Can not find adapter for model {model_name}")
@cache
def _auto_get_conv_template(
    model_name: str, model_path: str
) -> Optional[ConversationAdapter]:
    """Auto get the conversation template.

    Args:
        model_name (str): The name of the model.
        model_path (str): The path of the model.

    Returns:
        Optional[ConversationAdapter]: The conversation template.
    """
    try:
        # 调用函数获取适配器对象，使用快速聊天选项
        adapter = get_llm_model_adapter(model_name, model_path, use_fastchat=True)
        # 调用适配器对象的方法获取默认的对话模板
        return adapter.get_default_conv_template(model_name, model_path)
    except Exception as e:
        # 记录调试信息，指示无法获取对话模板的原因
        logger.debug(f"Failed to get conv template for {model_name} {model_path}: {e}")
        return None


class DefaultConversationAdapterFactory(ConversationAdapterFactory):
    def get_by_model(self, model_name: str, model_path: str) -> ConversationAdapter:
        """Get a conversation adapter by model.

        Args:
            model_name (str): The name of the model.
            model_path (str): The path of the model.
        Returns:
            ConversationAdapter: The conversation adapter.
        """
        # 调用函数获取自动对话模板，并返回对应的适配器对象
        return _auto_get_conv_template(model_name, model_path)


def _dynamic_model_parser() -> Optional[List[Type[BaseModelParameters]]]:
    """Dynamic model parser, parse the model parameters from the command line arguments.

    Returns:
        Optional[List[Type[BaseModelParameters]]]: The model parameters class list.
    """
    from dbgpt.model.parameter import (
        EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG,
        EmbeddingModelParameters,
        WorkerType,
    )
    from dbgpt.util.parameter_utils import _SimpleArgParser

    # 创建一个简单的命令行参数解析器对象
    pre_args = _SimpleArgParser("model_name", "model_path", "worker_type", "model_type")
    # 解析命令行参数
    pre_args.parse()
    # 从解析结果中获取模型名称、模型路径、工作类型和模型类型
    model_name = pre_args.get("model_name")
    model_path = pre_args.get("model_path")
    worker_type = pre_args.get("worker_type")
    model_type = pre_args.get("model_type")
    
    # 如果工作类型为 TEXT2VEC，则返回对应的嵌入模型参数类列表
    if worker_type == WorkerType.TEXT2VEC:
        return [
            EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG.get(
                model_name, EmbeddingModelParameters
            )
        ]
    
    # 如果模型名称为空且模型类型不为 VLLM，则返回 None
    if model_name is None and model_type != ModelType.VLLM:
        return None
    
    # 调用函数获取语言模型适配器对象，并获取其模型参数类
    llm_adapter = get_llm_model_adapter(model_name, model_path, model_type=model_type)
    param_class = llm_adapter.model_param_class()
    return [param_class]
```