# `.\DB-GPT-src\dbgpt\model\adapter\proxy_adapter.py`

```py
# 导入日志模块
import logging
# 导入抽象基类模块
from abc import abstractmethod
# 导入类型提示模块
from typing import Optional, Type

# 导入自定义模块
from dbgpt.model.adapter.base import LLMModelAdapter, register_model_adapter
from dbgpt.model.base import ModelType
from dbgpt.model.parameter import ProxyModelParameters
from dbgpt.model.proxy.base import ProxyLLMClient
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 代理模型适配器，继承自LLMModelAdapter基类
class ProxyLLMModelAdapter(LLMModelAdapter):
    # 返回新的适配器实例
    def new_adapter(self, **kwargs) -> "LLMModelAdapter":
        return self.__class__()

    # 返回模型类型
    def model_type(self) -> str:
        return ModelType.PROXY

    # 判断模型是否匹配给定条件
    def match(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        model_name = model_name.lower() if model_name else None
        model_path = model_path.lower() if model_path else None
        return self.do_match(model_name) or self.do_match(model_path)

    # 抽象方法，用于子类实现匹配逻辑
    @abstractmethod
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        raise NotImplementedError()

    # 获取动态LLM客户端类
    def dynamic_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Optional[Type[ProxyLLMClient]]:
        """Get dynamic llm client class

        Parse the llm_client_class from params and return the class

        Args:
            params (ProxyModelParameters): proxy model parameters

        Returns:
            Optional[Type[ProxyLLMClient]]: llm client class
        """

        if params.llm_client_class:
            # 从模块工具中导入并检查字符串所代表的类
            from dbgpt.util.module_utils import import_from_checked_string

            worker_cls: Type[ProxyLLMClient] = import_from_checked_string(
                params.llm_client_class, ProxyLLMClient
            )
            return worker_cls
        return None

    # 获取LLM客户端类
    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        """Get llm client class"""
        # 获取动态LLM客户端类
        dynamic_llm_client_class = self.dynamic_llm_client_class(params)
        if dynamic_llm_client_class:
            return dynamic_llm_client_class
        raise NotImplementedError()

    # 根据参数加载模型
    def load_from_params(self, params: ProxyModelParameters):
        # 获取动态LLM客户端类
        dynamic_llm_client_class = self.dynamic_llm_client_class(params)
        if not dynamic_llm_client_class:
            # 如果没有动态LLM客户端类，则获取LLM客户端类
            dynamic_llm_client_class = self.get_llm_client_class(params)
        # 记录加载模型的日志信息
        logger.info(
            f"Load model from params: {params}, llm client class: {dynamic_llm_client_class}"
        )
        # 创建代理模型客户端
        proxy_llm_client = dynamic_llm_client_class.new_client(params)
        # 创建并返回代理模型对象
        model = ProxyModel(params, proxy_llm_client)
        return model, model


# OpenAI代理LLM模型适配器，继承自ProxyLLMModelAdapter
class OpenAIProxyLLMModelAdapter(ProxyLLMModelAdapter):
    # 支持异步操作
    def support_async(self) -> bool:
        return True

    # 判断模型是否匹配给定条件
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return lower_model_name_or_path in ["chatgpt_proxyllm", "proxyllm"]

    # 获取LLM客户端类
    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 获取动态LLM客户端类
        dynamic_llm_client_class = self.dynamic_llm_client_class(params)
        if dynamic_llm_client_class:
            return dynamic_llm_client_class
        # 如果没有动态LLM客户端类，则抛出未实现错误
        raise NotImplementedError()
    ) -> Type[ProxyLLMClient]:
        """返回LLM客户端类的类型"""
        # 导入OpenAILLMClient类
        from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient

        # 返回OpenAILLMClient类作为LLM客户端类的类型
        return OpenAILLMClient

    def get_async_generate_stream_function(self, model, model_path: str):
        """获取异步生成流功能函数"""
        # 导入chatgpt_generate_stream函数
        from dbgpt.model.proxy.llms.chatgpt import chatgpt_generate_stream

        # 返回chatgpt_generate_stream函数
        return chatgpt_generate_stream
class TongyiProxyLLMModelAdapter(ProxyLLMModelAdapter):
    # TongyiProxyLLMModelAdapter 类，继承自 ProxyLLMModelAdapter 类，用于提供统一的代理LLM模型适配器

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查给定的模型名称或路径是否匹配 "tongyi_proxyllm"
        return lower_model_name_or_path == "tongyi_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 返回 TongyiLLMClient 类，用于与 Tongyi LLM 相关联的客户端类
        from dbgpt.model.proxy.llms.tongyi import TongyiLLMClient
        return TongyiLLMClient

    def get_generate_stream_function(self, model, model_path: str):
        # 返回 tongyi_generate_stream 函数，用于生成 Tongyi LLM 模型的数据流
        from dbgpt.model.proxy.llms.tongyi import tongyi_generate_stream
        return tongyi_generate_stream


class OllamaLLMModelAdapter(ProxyLLMModelAdapter):
    # OllamaLLMModelAdapter 类，继承自 ProxyLLMModelAdapter 类，用于提供 Ollama LLM 的代理模型适配器

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查给定的模型名称或路径是否匹配 "ollama_proxyllm"
        return lower_model_name_or_path == "ollama_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 返回 OllamaLLMClient 类，用于与 Ollama LLM 相关联的客户端类
        from dbgpt.model.proxy.llms.ollama import OllamaLLMClient
        return OllamaLLMClient

    def get_generate_stream_function(self, model, model_path: str):
        # 返回 ollama_generate_stream 函数，用于生成 Ollama LLM 模型的数据流
        from dbgpt.model.proxy.llms.ollama import ollama_generate_stream
        return ollama_generate_stream


class ZhipuProxyLLMModelAdapter(ProxyLLMModelAdapter):
    # ZhipuProxyLLMModelAdapter 类，继承自 ProxyLLMModelAdapter 类，用于提供 Zhipu LLM 的代理模型适配器
    support_system_message = False

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查给定的模型名称或路径是否匹配 "zhipu_proxyllm"
        return lower_model_name_or_path == "zhipu_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 返回 ZhipuLLMClient 类，用于与 Zhipu LLM 相关联的客户端类
        from dbgpt.model.proxy.llms.zhipu import ZhipuLLMClient
        return ZhipuLLMClient

    def get_generate_stream_function(self, model, model_path: str):
        # 返回 zhipu_generate_stream 函数，用于生成 Zhipu LLM 模型的数据流
        from dbgpt.model.proxy.llms.zhipu import zhipu_generate_stream
        return zhipu_generate_stream


class WenxinProxyLLMModelAdapter(ProxyLLMModelAdapter):
    # WenxinProxyLLMModelAdapter 类，继承自 ProxyLLMModelAdapter 类，用于提供 Wenxin LLM 的代理模型适配器

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查给定的模型名称或路径是否匹配 "wenxin_proxyllm"
        return lower_model_name_or_path == "wenxin_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 返回 WenxinLLMClient 类，用于与 Wenxin LLM 相关联的客户端类
        from dbgpt.model.proxy.llms.wenxin import WenxinLLMClient
        return WenxinLLMClient

    def get_generate_stream_function(self, model, model_path: str):
        # 返回 wenxin_generate_stream 函数，用于生成 Wenxin LLM 模型的数据流
        from dbgpt.model.proxy.llms.wenxin import wenxin_generate_stream
        return wenxin_generate_stream


class GeminiProxyLLMModelAdapter(ProxyLLMModelAdapter):
    # GeminiProxyLLMModelAdapter 类，继承自 ProxyLLMModelAdapter 类，用于提供 Gemini LLM 的代理模型适配器
    support_system_message = False

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查给定的模型名称或路径是否匹配 "gemini_proxyllm"
        return lower_model_name_or_path == "gemini_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 返回 GeminiLLMClient 类，用于与 Gemini LLM 相关联的客户端类
        from dbgpt.model.proxy.llms.gemini import GeminiLLMClient
        return GeminiLLMClient

    def get_generate_stream_function(self, model, model_path: str):
        # 返回 gemini_generate_stream 函数，用于生成 Gemini LLM 模型的数据流
        from dbgpt.model.proxy.llms.gemini import gemini_generate_stream
        return gemini_generate_stream


class SparkProxyLLMModelAdapter(ProxyLLMModelAdapter):
    # SparkProxyLLMModelAdapter 类，继承自 ProxyLLMModelAdapter 类，用于提供 Spark LLM 的代理模型适配器
    # 初始化一个布尔变量，用于表示是否支持系统消息
    support_system_message = False

    # 定义一个方法，用于检查传入的模型名或路径是否为 "spark_proxyllm"
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return lower_model_name_or_path == "spark_proxyllm"

    # 定义一个方法，根据给定的参数返回相应的 ProxyLLMClient 类型
    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 导入 SparkLLMClient 类
        from dbgpt.model.proxy.llms.spark import SparkLLMClient

        return SparkLLMClient

    # 定义一个方法，返回处理流生成的函数
    def get_generate_stream_function(self, model, model_path: str):
        # 导入 spark_generate_stream 函数
        from dbgpt.model.proxy.llms.spark import spark_generate_stream

        return spark_generate_stream
class BardProxyLLMModelAdapter(ProxyLLMModelAdapter):
    """Adapter for Bard proxy LLM model.

    This class extends ProxyLLMModelAdapter and provides specific implementations
    for Bard proxy LLM.

    See Also: `Bard Documentation <https://example.com/bard_docs>`_
    """

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        """Check if the given model name or path matches 'bard_proxyllm'."""
        return lower_model_name_or_path == "bard_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        """Get the client class for Bard proxy LLM.

        Returns:
            Type[ProxyLLMClient]: The client class for Bard proxy LLM, which is
            OpenAILLMClient.
        """
        # TODO: Bard proxy LLM not support ProxyLLMClient now, we just return OpenAILLMClient
        from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient

        return OpenAILLMClient

    def get_async_generate_stream_function(self, model, model_path: str):
        """Get the asynchronous generate stream function for Bard proxy LLM.

        Returns:
            Callable: The generate stream function specific to Bard proxy LLM.
        """
        from dbgpt.model.proxy.llms.bard import bard_generate_stream

        return bard_generate_stream


class BaichuanProxyLLMModelAdapter(ProxyLLMModelAdapter):
    """Adapter for Baichuan proxy LLM model.

    This class extends ProxyLLMModelAdapter and provides specific implementations
    for Baichuan proxy LLM.

    See Also: `Baichuan Documentation <https://example.com/baichuan_docs>`_
    """

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        """Check if the given model name or path matches 'bc_proxyllm'."""
        return lower_model_name_or_path == "bc_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        """Get the client class for Baichuan proxy LLM.

        Returns:
            Type[ProxyLLMClient]: The client class for Baichuan proxy LLM, which is
            OpenAILLMClient.
        """
        # TODO: Baichuan proxy LLM not support ProxyLLMClient now, we just return OpenAILLMClient
        from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient

        return OpenAILLMClient

    def get_async_generate_stream_function(self, model, model_path: str):
        """Get the asynchronous generate stream function for Baichuan proxy LLM.

        Returns:
            Callable: The generate stream function specific to Baichuan proxy LLM.
        """
        from dbgpt.model.proxy.llms.baichuan import baichuan_generate_stream

        return baichuan_generate_stream


class YiProxyLLMModelAdapter(ProxyLLMModelAdapter):
    """Yi proxy LLM model adapter.

    This class extends ProxyLLMModelAdapter and provides specific implementations
    for Yi proxy LLM.

    See Also: `Yi Documentation <https://platform.lingyiwanwu.com/docs/>`_
    """

    def support_async(self) -> bool:
        """Indicate if Yi proxy LLM supports asynchronous operations."""
        return True

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        """Check if the given model name or path matches 'yi_proxyllm'."""
        return lower_model_name_or_path in ["yi_proxyllm"]

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        """Get the client class for Yi proxy LLM.

        Returns:
            Type[ProxyLLMClient]: The client class for Yi proxy LLM, which is
            YiLLMClient.
        """
        from dbgpt.model.proxy.llms.yi import YiLLMClient

        return YiLLMClient

    def get_async_generate_stream_function(self, model, model_path: str):
        """Get the asynchronous generate stream function for Yi proxy LLM.

        Returns:
            Callable: The generate stream function specific to Yi proxy LLM.
        """
        from dbgpt.model.proxy.llms.yi import yi_generate_stream

        return yi_generate_stream


class MoonshotProxyLLMModelAdapter(ProxyLLMModelAdapter):
    """Moonshot proxy LLM model adapter.

    This class extends ProxyLLMModelAdapter and provides specific implementations
    for Moonshot proxy LLM.

    See Also: `Moonshot Documentation <https://platform.moonshot.cn/docs/>`_
    """

    def support_async(self) -> bool:
        """Indicate if Moonshot proxy LLM supports asynchronous operations."""
        return True

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        """Check if the given model name or path matches 'moonshot_proxyllm'."""
        return lower_model_name_or_path in ["moonshot_proxyllm"]

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        """Get the client class for Moonshot proxy LLM.

        Returns:
            Type[ProxyLLMClient]: The client class for Moonshot proxy LLM, which is
            MoonshotLLMClient.
        """
        from dbgpt.model.proxy.llms.moonshot import MoonshotLLMClient

        return MoonshotLLMClient

    def get_async_generate_stream_function(self, model, model_path: str):
        """Get the asynchronous generate stream function for Moonshot proxy LLM.

        Returns:
            Callable: The generate stream function specific to Moonshot proxy LLM.
        """
        from dbgpt.model.proxy.llms.moonshot import moonshot_generate_stream

        return moonshot_generate_stream
class DeepseekProxyLLMModelAdapter(ProxyLLMModelAdapter):
    """Deepseek proxy LLM model adapter.

    See Also: `Deepseek Documentation <https://platform.deepseek.com/api-docs/>`_
    """

    def support_async(self) -> bool:
        # 返回 True，表示此适配器支持异步操作
        return True

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查传入的模型名称或路径是否为 "deepseek_proxyllm"
        return lower_model_name_or_path == "deepseek_proxyllm"

    def get_llm_client_class(
        self, params: ProxyModelParameters
    ) -> Type[ProxyLLMClient]:
        # 返回 DeepseekLLMClient 类，用于处理 Deepseek 模型的客户端交互
        from dbgpt.model.proxy.llms.deepseek import DeepseekLLMClient
        return DeepseekLLMClient

    def get_async_generate_stream_function(self, model, model_path: str):
        # 返回 deepseek_generate_stream 函数，用于异步生成流的功能
        from dbgpt.model.proxy.llms.deepseek import deepseek_generate_stream
        return deepseek_generate_stream


register_model_adapter(OpenAIProxyLLMModelAdapter)
register_model_adapter(TongyiProxyLLMModelAdapter)
register_model_adapter(OllamaLLMModelAdapter)
register_model_adapter(ZhipuProxyLLMModelAdapter)
register_model_adapter(WenxinProxyLLMModelAdapter)
register_model_adapter(GeminiProxyLLMModelAdapter)
register_model_adapter(SparkProxyLLMModelAdapter)
register_model_adapter(BardProxyLLMModelAdapter)
register_model_adapter(BaichuanProxyLLMModelAdapter)
register_model_adapter(YiProxyLLMModelAdapter)
register_model_adapter(MoonshotProxyLLMModelAdapter)
register_model_adapter(DeepseekProxyLLMModelAdapter)
```