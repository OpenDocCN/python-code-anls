# `.\DB-GPT-src\dbgpt\app\initialization\embedding_component.py`

```py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Type

from dbgpt.component import ComponentType, SystemApp
from dbgpt.core import Embeddings, RerankEmbeddings
from dbgpt.rag.embedding.embedding_factory import (
    EmbeddingFactory,
    RerankEmbeddingFactory,
)

if TYPE_CHECKING:
    from langchain.embeddings.base import Embeddings
    from dbgpt.app.base import WebServerParameters

logger = logging.getLogger(__name__)


def _initialize_embedding_model(
    param: "WebServerParameters",
    system_app: SystemApp,
    embedding_model_name: str,
    embedding_model_path: str,
):
    # 如果参数指定使用远程嵌入模型，则注册远程嵌入模型工厂并记录日志
    if param.remote_embedding:
        logger.info("Register remote RemoteEmbeddingFactory")
        system_app.register(RemoteEmbeddingFactory, model_name=embedding_model_name)
    else:
        # 否则，注册本地嵌入模型工厂并记录日志
        logger.info(f"Register local LocalEmbeddingFactory")
        system_app.register(
            LocalEmbeddingFactory,
            default_model_name=embedding_model_name,
            default_model_path=embedding_model_path,
        )


def _initialize_rerank_model(
    param: "WebServerParameters",
    system_app: SystemApp,
    rerank_model_name: Optional[str] = None,
    rerank_model_path: Optional[str] = None,
):
    # 如果未指定重排序模型名称，直接返回
    if not rerank_model_name:
        return
    # 如果参数指定使用远程重排序模型，则注册远程重排序嵌入模型工厂并记录日志
    if param.remote_rerank:
        logger.info("Register remote RemoteRerankEmbeddingFactory")
        system_app.register(RemoteRerankEmbeddingFactory, model_name=rerank_model_name)
    else:
        # 否则，注册本地重排序嵌入模型工厂并记录日志
        logger.info(f"Register local LocalRerankEmbeddingFactory")
        system_app.register(
            LocalRerankEmbeddingFactory,
            default_model_name=rerank_model_name,
            default_model_path=rerank_model_path,
        )


class RemoteEmbeddingFactory(EmbeddingFactory):
    def __init__(self, system_app, model_name: str = None, **kwargs: Any) -> None:
        super().__init__(system_app=system_app)
        self._default_model_name = model_name
        self.kwargs = kwargs
        self.system_app = system_app

    def init_app(self, system_app):
        self.system_app = system_app

    def create(
        self, model_name: str = None, embedding_cls: Type = None
    ) -> "Embeddings":
        # 引入相关依赖，如WorkerManagerFactory和RemoteEmbeddings
        from dbgpt.model.cluster import WorkerManagerFactory
        from dbgpt.model.cluster.embedding.remote_embedding import RemoteEmbeddings

        # 如果指定了embedding_cls，则抛出未实现错误
        if embedding_cls:
            raise NotImplementedError
        
        # 获取系统组件WorkerManagerFactory并创建worker_manager实例
        worker_manager = self.system_app.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 忽略传入的model_name参数，创建RemoteEmbeddings实例并返回
        return RemoteEmbeddings(self._default_model_name, worker_manager)


class LocalEmbeddingFactory(EmbeddingFactory):
    def __init__(
        self,
        system_app,
        default_model_name: str = None,
        default_model_path: str = None,
        **kwargs: Any,
    ):
        super().__init__(system_app=system_app)
        self._default_model_name = default_model_name
        self._default_model_path = default_model_path
        self.kwargs = kwargs
        self.system_app = system_app
    ) -> None:
        super().__init__(system_app=system_app)
        self._default_model_name = default_model_name
        self._default_model_path = default_model_path
        self._kwargs = kwargs
        self._model = self._load_model()


        # 调用父类的构造函数，初始化系统应用
        super().__init__(system_app=system_app)
        # 设置默认模型名称和路径
        self._default_model_name = default_model_name
        self._default_model_path = default_model_path
        # 存储传入的额外关键字参数
        self._kwargs = kwargs
        # 调用 _load_model 方法加载模型，并将返回的模型对象存储在 self._model 中
        self._model = self._load_model()


    def init_app(self, system_app):
        pass


        # 初始化应用的方法，这里未实现具体功能，故使用 pass 语句占位
        pass


    def create(
        self, model_name: str = None, embedding_cls: Type = None
    ) -> "Embeddings":
        if embedding_cls:
            raise NotImplementedError
        return self._model


        # 创建方法，根据指定的模型名称和嵌入类（embedding_cls）创建 Embeddings 对象
        if embedding_cls:
            # 如果指定了 embedding_cls，则抛出 NotImplementedError 异常，表示未实现的功能
            raise NotImplementedError
        # 返回存储在 self._model 中的模型对象
        return self._model


    def _load_model(self) -> "Embeddings":
        from dbgpt.model.adapter.embeddings_loader import (
            EmbeddingLoader,
            _parse_embedding_params,
        )
        from dbgpt.model.parameter import (
            EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG,
            BaseEmbeddingModelParameters,
            EmbeddingModelParameters,
        )

        # 获取默认模型名称对应的参数类
        param_cls = EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG.get(
            self._default_model_name, EmbeddingModelParameters
        )
        # 解析模型参数，返回 BaseEmbeddingModelParameters 对象
        model_params: BaseEmbeddingModelParameters = _parse_embedding_params(
            model_name=self._default_model_name,
            model_path=self._default_model_path,
            param_cls=param_cls,
            **self._kwargs,
        )
        # 记录模型参数信息到日志
        logger.info(model_params)
        # 创建 EmbeddingLoader 对象
        loader = EmbeddingLoader()
        # 使用 loader 加载指定模型名称和参数的模型，并返回加载的模型对象
        return loader.load(self._default_model_name, model_params)


        # 加载模型的私有方法，导入所需的模块和类，解析模型参数，加载指定模型名称和参数的模型
        # 返回加载的模型对象
# 定义一个名为 RemoteRerankEmbeddingFactory 的类，继承自 RerankEmbeddingFactory 类
class RemoteRerankEmbeddingFactory(RerankEmbeddingFactory):
    # 初始化方法，接收 system_app 和 model_name 参数以及任意关键字参数
    def __init__(self, system_app, model_name: str = None, **kwargs: Any) -> None:
        # 调用父类 RerankEmbeddingFactory 的初始化方法
        super().__init__(system_app=system_app)
        # 设置默认的模型名称
        self._default_model_name = model_name
        # 将传入的所有关键字参数保存在 self.kwargs 中
        self.kwargs = kwargs
        # 将 system_app 参数保存在 self.system_app 中
        self.system_app = system_app

    # 初始化应用程序的方法，更新 self.system_app 属性
    def init_app(self, system_app):
        self.system_app = system_app

    # 创建方法，用于创建 RerankEmbeddings 对象
    def create(
        self, model_name: str = None, embedding_cls: Type = None
    ) -> "RerankEmbeddings":
        # 导入必要的模块和类
        from dbgpt.model.cluster import WorkerManagerFactory
        from dbgpt.model.cluster.embedding.remote_embedding import (
            RemoteRerankEmbeddings,
        )

        # 如果指定了 embedding_cls，抛出 NotImplementedError
        if embedding_cls:
            raise NotImplementedError
        
        # 获取 WorkerManagerFactory 实例并创建 worker_manager
        worker_manager = self.system_app.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 创建并返回 RemoteRerankEmbeddings 对象，使用 model_name 或者 self._default_model_name 和 worker_manager 参数
        return RemoteRerankEmbeddings(
            model_name or self._default_model_name, worker_manager
        )


# 定义一个名为 LocalRerankEmbeddingFactory 的类，继承自 RerankEmbeddingFactory 类
class LocalRerankEmbeddingFactory(RerankEmbeddingFactory):
    # 初始化方法，接收 system_app、default_model_name、default_model_path 参数以及任意关键字参数
    def __init__(
        self,
        system_app,
        default_model_name: str = None,
        default_model_path: str = None,
        **kwargs: Any,
    ) -> None:
        # 调用父类 RerankEmbeddingFactory 的初始化方法
        super().__init__(system_app=system_app)
        # 设置默认的模型名称和模型路径
        self._default_model_name = default_model_name
        self._default_model_path = default_model_path
        # 将传入的所有关键字参数保存在 self._kwargs 中
        self._kwargs = kwargs
        # 调用 _load_model 方法加载模型，并将结果保存在 self._model 中
        self._model = self._load_model()

    # 初始化应用程序的方法，这里没有实际操作，直接 pass
    def init_app(self, system_app):
        pass

    # 创建方法，用于创建 RerankEmbeddings 对象
    def create(
        self, model_name: str = None, embedding_cls: Type = None
    ) -> "RerankEmbeddings":
        # 如果指定了 embedding_cls，抛出 NotImplementedError
        if embedding_cls:
            raise NotImplementedError
        # 直接返回已加载的模型对象 self._model
        return self._model

    # 加载模型的私有方法，返回加载后的 RerankEmbeddings 对象
    def _load_model(self) -> "RerankEmbeddings":
        # 导入必要的模块和类
        from dbgpt.model.adapter.embeddings_loader import (
            EmbeddingLoader,
            _parse_embedding_params,
        )
        from dbgpt.model.parameter import (
            EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG,
            BaseEmbeddingModelParameters,
            EmbeddingModelParameters,
        )

        # 根据 self._default_model_name 获取对应的参数类
        param_cls = EMBEDDING_NAME_TO_PARAMETER_CLASS_CONFIG.get(
            self._default_model_name, EmbeddingModelParameters
        )
        
        # 解析模型参数，创建 model_params 对象
        model_params: BaseEmbeddingModelParameters = _parse_embedding_params(
            model_name=self._default_model_name,
            model_path=self._default_model_path,
            param_cls=param_cls,
            **self._kwargs,
        )
        
        # 记录模型参数信息到日志
        logger.info(model_params)
        
        # 创建 EmbeddingLoader 实例
        loader = EmbeddingLoader()
        
        # 调用 loader 的 load_rerank_model 方法加载指定名称和参数的 rerank 模型，忽略 model_name 参数
        return loader.load_rerank_model(self._default_model_name, model_params)
```