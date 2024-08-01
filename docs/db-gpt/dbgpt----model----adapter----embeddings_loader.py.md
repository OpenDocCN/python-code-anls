# `.\DB-GPT-src\dbgpt\model\adapter\embeddings_loader.py`

```py
# 导入必要的模块和类
from __future__ import annotations  # 导入用于支持类型注解的特性

import logging  # 导入日志模块
from typing import List, Optional, Type, cast  # 导入类型提示所需的类型定义

# 导入项目中的特定模块和类
from dbgpt.configs.model_config import get_device  # 导入获取设备配置的函数
from dbgpt.core import Embeddings, RerankEmbeddings  # 导入嵌入模型相关的核心类
from dbgpt.model.parameter import (
    BaseEmbeddingModelParameters,  # 导入基础嵌入模型参数类
    EmbeddingModelParameters,  # 导入嵌入模型参数类
    ProxyEmbeddingParameters,  # 导入代理嵌入参数类
)
from dbgpt.util.parameter_utils import (
    EnvArgumentParser,  # 导入用于解析环境参数的类
    _get_dict_from_obj,  # 导入从对象获取字典的辅助函数
)
from dbgpt.util.system_utils import get_system_info  # 导入获取系统信息的函数
from dbgpt.util.tracer import SpanType, SpanTypeRunName, root_tracer  # 导入用于跟踪的类和函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class EmbeddingLoader:
    def __init__(self) -> None:
        pass
    def load(self, model_name: str, param: BaseEmbeddingModelParameters) -> Embeddings:
        # 构建元数据字典，描述模型加载过程中的各种信息
        metadata = {
            "model_name": model_name,
            "run_service": SpanTypeRunName.EMBEDDING_MODEL.value,
            "params": _get_dict_from_obj(param),  # 将参数对象转换为字典
            "sys_infos": _get_dict_from_obj(get_system_info()),  # 获取系统信息并转换为字典
        }
        # 使用根跟踪器创建一个新的跟踪 span，用于监视加载模型的过程
        with root_tracer.start_span(
            "EmbeddingLoader.load", span_type=SpanType.RUN, metadata=metadata
        ):
            # 根据模型名称选择不同的模型加载逻辑
            # 对于代理模型 "proxy_openai" 和 "proxy_azure"
            if model_name in ["proxy_openai", "proxy_azure"]:
                from langchain.embeddings import OpenAIEmbeddings

                from dbgpt.rag.embedding._wrapped import WrappedEmbeddings

                # 返回一个包装后的 OpenAIEmbeddings 模型对象
                return WrappedEmbeddings(OpenAIEmbeddings(**param.build_kwargs()))
            # 对于代理模型 "proxy_http_openapi"
            elif model_name in ["proxy_http_openapi"]:
                from dbgpt.rag.embedding import OpenAPIEmbeddings

                proxy_param = cast(ProxyEmbeddingParameters, param)
                openapi_param = {}
                if proxy_param.proxy_server_url:
                    openapi_param["api_url"] = proxy_param.proxy_server_url
                if proxy_param.proxy_api_key:
                    openapi_param["api_key"] = proxy_param.proxy_api_key
                if proxy_param.proxy_backend:
                    openapi_param["model_name"] = proxy_param.proxy_backend
                # 返回一个 OpenAPIEmbeddings 模型对象
                return OpenAPIEmbeddings(**openapi_param)
            # 对于代理模型 "proxy_tongyi"
            elif model_name in ["proxy_tongyi"]:
                from dbgpt.rag.embedding import TongYiEmbeddings

                proxy_param = cast(ProxyEmbeddingParameters, param)
                tongyi_param = {"api_key": proxy_param.proxy_api_key}
                if proxy_param.proxy_backend:
                    tongyi_param["model_name"] = proxy_param.proxy_backend
                # 返回一个 TongYiEmbeddings 模型对象
                return TongYiEmbeddings(**tongyi_param)
            # 对于代理模型 "proxy_ollama"
            elif model_name in ["proxy_ollama"]:
                from dbgpt.rag.embedding import OllamaEmbeddings

                proxy_param = cast(ProxyEmbeddingParameters, param)
                ollama_param = {}
                if proxy_param.proxy_server_url:
                    ollama_param["api_url"] = proxy_param.proxy_server_url
                if proxy_param.proxy_backend:
                    ollama_param["model_name"] = proxy_param.proxy_backend
                # 返回一个 OllamaEmbeddings 模型对象
                return OllamaEmbeddings(**ollama_param)
            # 默认情况下使用 HuggingFaceEmbeddings 模型
            else:
                from dbgpt.rag.embedding import HuggingFaceEmbeddings

                kwargs = param.build_kwargs(model_name=param.model_path)
                # 返回一个 HuggingFaceEmbeddings 模型对象
                return HuggingFaceEmbeddings(**kwargs)
    # 定义函数签名，指定返回类型为 RerankEmbeddings
    ) -> RerankEmbeddings:
        # 构建元数据字典，包含模型名称、运行服务类型、参数以及系统信息
        metadata = {
            "model_name": model_name,
            "run_service": SpanTypeRunName.EMBEDDING_MODEL.value,
            "params": _get_dict_from_obj(param),
            "sys_infos": _get_dict_from_obj(get_system_info()),
        }
        # 使用根跟踪器开始一个新的跟踪 span，命名为 "EmbeddingLoader.load_rerank_model"，类型为 SpanType.RUN，元数据为 metadata
        with root_tracer.start_span(
            "EmbeddingLoader.load_rerank_model",
            span_type=SpanType.RUN,
            metadata=metadata,
        ):
            # 如果模型名称在 ["rerank_proxy_http_openapi"] 列表中
            if model_name in ["rerank_proxy_http_openapi"]:
                # 导入 OpenAPIRerankEmbeddings 类
                from dbgpt.rag.embedding.rerank import OpenAPIRerankEmbeddings

                # 将 param 强制转换为 ProxyEmbeddingParameters 类型
                proxy_param = cast(ProxyEmbeddingParameters, param)
                openapi_param = {}
                # 如果 proxy_param.proxy_server_url 不为空
                if proxy_param.proxy_server_url:
                    openapi_param["api_url"] = proxy_param.proxy_server_url
                # 如果 proxy_param.proxy_api_key 不为空
                if proxy_param.proxy_api_key:
                    openapi_param["api_key"] = proxy_param.proxy_api_key
                # 如果 proxy_param.proxy_backend 不为空
                if proxy_param.proxy_backend:
                    openapi_param["model_name"] = proxy_param.proxy_backend
                # 返回使用 openapi_param 参数初始化的 OpenAPIRerankEmbeddings 实例
                return OpenAPIRerankEmbeddings(**openapi_param)
            else:
                # 否则，导入 CrossEncoderRerankEmbeddings 类
                from dbgpt.rag.embedding.rerank import CrossEncoderRerankEmbeddings

                # 使用 param.build_kwargs 方法构建参数 kwargs
                kwargs = param.build_kwargs(model_name=param.model_path)
                # 返回使用 kwargs 参数初始化的 CrossEncoderRerankEmbeddings 实例
                return CrossEncoderRerankEmbeddings(**kwargs)
# 解析嵌入模型参数的函数，可以根据给定的模型名称、模型路径、命令行参数和其他关键字参数来创建参数对象
def _parse_embedding_params(
    model_name: Optional[str] = None,                      # 模型名称，可选参数
    model_path: Optional[str] = None,                      # 模型路径，可选参数
    command_args: List[str] = None,                        # 命令行参数列表，可选参数，默认为None
    param_cls: Optional[Type] = EmbeddingModelParameters,  # 参数类，默认为EmbeddingModelParameters
    **kwargs,                                               # 其他关键字参数
):
    # 创建一个环境参数解析器对象
    model_args = EnvArgumentParser()
    # 获取模型名称对应的环境变量前缀
    env_prefix = EnvArgumentParser.get_env_prefix(model_name)
    # 使用模型参数解析器解析参数，并转换为数据类对象，同时传入模型名称、模型路径和其他关键字参数
    model_params: BaseEmbeddingModelParameters = model_args.parse_args_into_dataclass(
        param_cls,
        env_prefixes=[env_prefix],       # 环境变量前缀列表
        command_args=command_args,       # 命令行参数列表
        model_name=model_name,           # 模型名称
        model_path=model_path,           # 模型路径
        **kwargs,                        # 其他关键字参数
    )
    # 如果模型参数中的设备信息为空，则调用get_device()函数获取默认设备，并记录日志信息
    if not model_params.device:
        model_params.device = get_device()
        logger.info(
            f"[EmbeddingsModelWorker] Parameters of device is None, use {model_params.device}"
        )
    # 返回解析后的模型参数对象
    return model_params
```