# `.\DB-GPT-src\dbgpt\rag\embedding\embedding_factory.py`

```py
"""EmbeddingFactory class and DefaultEmbeddingFactory class."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from dbgpt.component import BaseComponent, SystemApp  # 导入必要的模块和类
from dbgpt.core import Embeddings, RerankEmbeddings  # 导入Embeddings和RerankEmbeddings类
from dbgpt.core.awel import DAGVar  # 导入DAGVar类
from dbgpt.core.awel.flow import ResourceCategory, register_resource  # 导入ResourceCategory和register_resource函数
from dbgpt.util.i18n_utils import _  # 导入国际化函数

logger = logging.getLogger(__name__)  # 获取当前模块的logger对象


class EmbeddingFactory(BaseComponent, ABC):
    """Abstract base class for EmbeddingFactory."""

    name = "embedding_factory"  # 设置类属性name为"embedding_factory"

    @abstractmethod
    def create(
        self, model_name: Optional[str] = None, embedding_cls: Optional[Type] = None
    ) -> Embeddings:
        """Create an embedding instance.

        Args:
            model_name (str): The model name.
            embedding_cls (Type): The embedding class.

        Returns:
            Embeddings: The embedding instance.
        """
        # 抽象方法，需要在子类中实现，用于创建嵌入实例


class RerankEmbeddingFactory(BaseComponent, ABC):
    """Class for RerankEmbeddingFactory."""

    name = "rerank_embedding_factory"  # 设置类属性name为"rerank_embedding_factory"

    @abstractmethod
    def create(
        self, model_name: Optional[str] = None, embedding_cls: Optional[Type] = None
    ) -> RerankEmbeddings:
        """Create an embedding instance.

        Args:
            model_name (str): The model name.
            embedding_cls (Type): The embedding class.

        Returns:
            RerankEmbeddings: The embedding instance.
        """
        # 抽象方法，需要在子类中实现，用于创建重新排名嵌入实例


class DefaultEmbeddingFactory(EmbeddingFactory):
    """The default embedding factory."""

    def __init__(
        self,
        system_app: Optional[SystemApp] = None,
        default_model_name: Optional[str] = None,
        default_model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new DefaultEmbeddingFactory."""
        super().__init__(system_app=system_app)  # 调用父类构造函数初始化BaseComponent
        if not default_model_path:
            default_model_path = default_model_name  # 如果未提供default_model_path，则使用default_model_name
        if not default_model_name:
            default_model_name = default_model_path  # 如果未提供default_model_name，则使用default_model_path
        self._default_model_name = default_model_name  # 设置默认模型名
        self._default_model_path = default_model_path  # 设置默认模型路径
        self._kwargs = kwargs  # 存储其它关键字参数
        self._model = self._load_model()  # 调用私有方法加载模型数据

    def init_app(self, system_app):
        """Init the app."""
        pass
        # 初始化应用程序，但此处未实现具体逻辑

    def create(
        self, model_name: Optional[str] = None, embedding_cls: Optional[Type] = None
    ) -> Embeddings:
        """Create an embedding instance.

        Args:
            model_name (str): The model name.
            embedding_cls (Type): The embedding class.
        """
        if embedding_cls:
            raise NotImplementedError  # 如果指定了embedding_cls则抛出未实现错误
        return self._model  # 返回预加载的模型实例
    @classmethod
    def _load_model(self) -> Embeddings:
        # 导入必要的模块和函数
        from dbgpt.model.adapter.embeddings_loader import (
            EmbeddingLoader,
            _parse_embedding_params,
        )
        from dbgpt.model.parameter import (
            EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG,
            BaseEmbeddingModelParameters,
            EmbeddingModelParameters,
        )

        # 根据默认模型名称选择对应的参数类
        param_cls = EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG.get(
            self._default_model_name, EmbeddingModelParameters
        )
        # 解析嵌入模型的参数
        model_params: BaseEmbeddingModelParameters = _parse_embedding_params(
            model_name=self._default_model_name,
            model_path=self._default_model_path,
            param_cls=param_cls,
            **self._kwargs,
        )
        # 记录模型参数信息
        logger.info(model_params)
        
        # 创建嵌入加载器实例
        loader = EmbeddingLoader()
        # 忽略 model_name 参数，选择默认模型名称或从参数中获取的模型名称
        model_name = self._default_model_name or model_params.model_name
        # 如果模型名称未提供，则抛出 ValueError 异常
        if not model_name:
            raise ValueError("model_name must be provided.")
        
        # 使用加载器加载指定模型和参数的嵌入
        return loader.load(model_name, model_params)

    @classmethod
    def openai(
        cls,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        timeout: int = 60,
        **kwargs: Any,
    ) -> Embeddings:
        """Create an OpenAI embeddings.

        If api_url and api_key are not provided, we will try to get them from
        environment variables.

        Args:
            api_url (Optional[str], optional): The api url. Defaults to None.
            api_key (Optional[str], optional): The api key. Defaults to None.
            model_name (str, optional): The model name.
                Defaults to "text-embedding-3-small".
            timeout (int, optional): The timeout. Defaults to 60.

        Returns:
            Embeddings: The embeddings instance.
        """
        # 设置 API URL，默认从环境变量中获取 OPENAI_API_BASE，否则使用默认值
        api_url = (
            api_url
            or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1") + "/embeddings"
        )
        # 获取 API Key，如果未提供则从环境变量 OPENAI_API_KEY 中获取
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        # 如果没有找到 API Key，则抛出 ValueError 异常
        if not api_key:
            raise ValueError("api_key must be provided.")
        
        # 调用 remote 方法，使用提供的参数创建并返回 OpenAI embeddings 实例
        return cls.remote(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def default(
        cls, model_name: str, model_path: Optional[str] = None, **kwargs: Any
    ) -> Embeddings:
        """
        Create a default embeddings instance.

        Args:
            model_name (str): The model name.
            model_path (Optional[str], optional): The model path. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Embeddings: The embeddings instance.
        """
        # 调用 openai 方法创建 embeddings 实例，使用默认参数
        return cls.openai(model_name=model_name, model_path=model_path, **kwargs)
    ) -> Embeddings:
        """
        创建一个默认的嵌入模型。

        尝试从模型名称或模型路径加载模型。

        Args:
            model_name (str): 模型名称。
            model_path (Optional[str], optional): 模型路径。默认为None。
                如果未提供，将使用模型名称作为模型路径来加载模型。

        Returns:
            Embeddings: 嵌入模型的实例。
        """
        return cls(
            default_model_name=model_name, default_model_path=model_path, **kwargs
        ).create()

    @classmethod
    def remote(
        cls,
        api_url: str = "http://localhost:8100/api/v1/embeddings",
        api_key: Optional[str] = None,
        model_name: str = "text2vec",
        timeout: int = 60,
        **kwargs: Any,
    ) -> Embeddings:
        """
        创建一个远程嵌入模型。

        创建一个与OpenAI API兼容的远程嵌入模型。因此，如果您的模型与OpenAI API兼容，
        您可以使用此方法来创建远程嵌入模型。

        Args:
            api_url (str, optional): API的URL。默认为 "http://localhost:8100/api/v1/embeddings"。
            api_key (Optional[str], optional): API密钥。默认为None。
            model_name (str, optional): 模型名称。默认为 "text2vec"。
            timeout (int, optional): 超时时间。默认为60。
        """
        from .embeddings import OpenAPIEmbeddings

        return OpenAPIEmbeddings(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
            **kwargs,
        )
class WrappedEmbeddingFactory(EmbeddingFactory):
    """The default embedding factory."""

    def __init__(
        self,
        system_app: Optional[SystemApp] = None,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new DefaultEmbeddingFactory."""
        # 调用父类的初始化方法，传入系统应用对象
        super().__init__(system_app=system_app)
        # 如果没有提供 embeddings 参数，则抛出数值错误异常
        if not embeddings:
            raise ValueError("embeddings must be provided.")
        # 将 embeddings 参数赋值给对象的 _model 属性
        self._model = embeddings

    def init_app(self, system_app):
        """Init the app."""
        # 此方法目前未实现，只有占位作用
        pass

    def create(
        self, model_name: Optional[str] = None, embedding_cls: Optional[Type] = None
    ) -> Embeddings:
        """Create an embedding instance.

        Args:
            model_name (str): The model name.
            embedding_cls (Type): The embedding class.
        """
        # 如果提供了 embedding_cls 参数，则抛出未实现错误
        if embedding_cls:
            raise NotImplementedError
        # 返回对象的 _model 属性，即嵌入模型对象
        return self._model


@register_resource(
    label=_("Default Embeddings"),
    name="default_embeddings",
    category=ResourceCategory.EMBEDDINGS,
    description=_(
        "Default embeddings(using default embedding model of current system)"
    ),
)
class DefaultEmbeddings(Embeddings):
    """The default embeddings."""

    def __init__(self, embedding_factory: Optional[EmbeddingFactory] = None) -> None:
        """Create a new DefaultEmbeddings."""
        # 将 embedding_factory 参数赋值给对象的 _embedding_factory 属性
        self._embedding_factory = embedding_factory

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings."""
        # 如果 _embedding_factory 属性为空，则获取当前系统应用，并使用其创建 EmbeddingFactory 实例
        if not self._embedding_factory:
            system_app = DAGVar.get_current_system_app()
            if not system_app:
                raise ValueError("System app is not initialized")
            self._embedding_factory = EmbeddingFactory.get_instance(system_app)
        # 调用 _embedding_factory 的 create 方法，返回嵌入模型对象
        return self._embedding_factory.create()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # 调用 embeddings 的 embed_documents 方法，传入文本列表，返回嵌入文档的结果列表
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # 调用 embeddings 的 embed_query 方法，传入查询文本，返回嵌入查询文本的结果向量
        return self.embeddings.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        # 异步调用 embeddings 的 aembed_documents 方法，传入文本列表，返回嵌入文档的结果列表
        return await self.embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        # 异步调用 embeddings 的 aembed_query 方法，传入查询文本，返回嵌入查询文本的结果向量
        return await self.embeddings.aembed_query(text)
```