# `.\DB-GPT-src\dbgpt\model\cluster\manager_base.py`

```py
import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Optional

from dbgpt.component import BaseComponent, ComponentType, SystemApp
from dbgpt.core import ModelMetadata, ModelOutput
from dbgpt.model.base import WorkerApplyOutput, WorkerSupportedModel
from dbgpt.model.cluster.base import WorkerApplyRequest, WorkerStartupRequest
from dbgpt.model.cluster.worker_base import ModelWorker
from dbgpt.model.parameter import ModelParameters, ModelWorkerParameters
from dbgpt.util.parameter_utils import ParameterDescription

@dataclass
class WorkerRunData:
    host: str                          # 主机名
    port: int                          # 端口号
    worker_key: str                    # 工作器键值
    worker: ModelWorker                # 工作器对象
    worker_params: ModelWorkerParameters  # 工作器参数
    model_params: ModelParameters      # 模型参数
    stop_event: asyncio.Event          # 异步事件用于停止信号
    semaphore: asyncio.Semaphore = None  # 异步信号量，可选
    command_args: List[str] = None     # 命令行参数列表，可选
    _heartbeat_future: Optional[Future] = None  # 可选的异步未来对象，心跳
    _last_heartbeat: Optional[datetime] = None  # 可选的最后心跳时间对象

    def _to_print_key(self):
        model_name = self.model_params.model_name  # 获取模型名称
        model_type = (
            self.model_params.model_type          # 获取模型类型，如果存在
            if hasattr(self.model_params, "model_type")
            else "text2vec"
        )
        host = self.host                         # 获取主机名
        port = self.port                         # 获取端口号
        return f"model {model_name}@{model_type}({host}:{port})"  # 返回格式化的打印键名

    @property
    def stopped(self):
        """Check if the worker is stopped"""  # 检查工作器是否停止的属性
        return self.stop_event.is_set()  # 返回停止事件是否被设置

class WorkerManager(ABC):
    @abstractmethod
    async def start(self):
        """Start worker manager

        Raises:
            Exception: if start worker manager not successfully
        """  # 启动工作器管理器的抽象方法，可能会引发异常

    @abstractmethod
    async def stop(self, ignore_exception: bool = False):
        """Stop worker manager"""  # 停止工作器管理器的抽象方法

    @abstractmethod
    def after_start(self, listener: Callable[["WorkerManager"], None]):
        """Add a listener after WorkerManager startup"""  # 在工作器管理器启动后添加监听器的抽象方法

    @abstractmethod
    async def get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        """Asynchronous get model instances by worker type and model name"""  # 异步获取特定工作器类型和模型名称的模型实例列表的抽象方法

    @abstractmethod
    async def get_all_model_instances(
        self, worker_type: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        """Asynchronous get all model instances

        Args:
            worker_type (str): worker type
            healthy_only (bool, optional): only return healthy instances. Defaults to True.

        Returns:
            List[WorkerRunData]: worker run data list
        """  # 异步获取所有模型实例的抽象方法，可按工作器类型和健康状态过滤

    @abstractmethod
    def sync_get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        """Get model instances by worker type and model name"""  # 同步获取特定工作器类型和模型名称的模型实例列表的抽象方法

    @abstractmethod
    async def sync_get_all_model_instances(
        self, worker_type: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        """Asynchronous get all model instances"""  # 异步获取所有模型实例的抽象方法
    # 异步方法：从给定的工作类型和模型名称中选择一个实例，通常选择健康状态的实例
    async def select_one_instance(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> WorkerRunData:
        """Asynchronous select one instance"""

    # 抽象方法：同步方法选择一个实例，根据给定的工作类型和模型名称，通常选择健康状态的实例
    @abstractmethod
    def sync_select_one_instance(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> WorkerRunData:
        """Select one instance"""

    # 抽象方法：异步获取支持的模型列表
    @abstractmethod
    async def supported_models(self) -> List[WorkerSupportedModel]:
        """List supported models"""

    # 抽象方法：启动并创建一个模型实例，并异步开始其运行
    @abstractmethod
    async def model_startup(self, startup_req: WorkerStartupRequest):
        """Create and start a model instance"""

    # 抽象方法：关闭指定模型实例，异步操作
    @abstractmethod
    async def model_shutdown(self, shutdown_req: WorkerStartupRequest):
        """Shutdown model instance"""

    # 抽象方法：生成流式结果，通常用于聊天场景，根据参数生成异步迭代器的模型输出
    @abstractmethod
    async def generate_stream(self, params: Dict, **kwargs) -> Iterator[ModelOutput]:
        """Generate stream result, chat scene"""

    # 抽象方法：生成非流式结果，根据参数生成模型输出
    @abstractmethod
    async def generate(self, params: Dict) -> ModelOutput:
        """Generate non stream result"""

    # 抽象方法：异步嵌入输入，根据参数进行输入的嵌入操作，返回嵌入后的浮点数列表
    @abstractmethod
    async def embeddings(self, params: Dict) -> List[List[float]]:
        """Asynchronous embed input"""

    # 抽象方法：同步嵌入输入，提供给第三方系统调用的同步版本，返回嵌入后的浮点数列表
    @abstractmethod
    def sync_embeddings(self, params: Dict) -> List[List[float]]:
        """Embed input

        This function may be passed to a third-party system call for synchronous calls.
        We must provide a synchronous version.
        """

    # 抽象方法：异步计算提示语的令牌数量，根据参数中的提示和模型信息进行计算
    @abstractmethod
    async def count_token(self, params: Dict) -> int:
        """Count token of prompt

        Args:
            params (Dict): parameters, eg. {"prompt": "hello", "model": "vicuna-13b-v1.5"}

        Returns:
            int: token count
        """

    # 抽象方法：获取模型的元数据信息，根据模型参数返回模型的元数据
    @abstractmethod
    async def get_model_metadata(self, params: Dict) -> ModelMetadata:
        """Get model metadata

        Args:
            params (Dict): parameters, eg. {"model": "vicuna-13b-v1.5"}
        """

    # 抽象方法：工作申请，根据给定的申请请求进行工作申请操作，异步返回工作申请的输出
    @abstractmethod
    async def worker_apply(self, apply_req: WorkerApplyRequest) -> WorkerApplyOutput:
        """Worker apply"""

    # 抽象方法：获取指定模型的参数描述列表，根据工作类型和模型名称返回参数描述列表
    @abstractmethod
    async def parameter_descriptions(
        self, worker_type: str, model_name: str
    ) -> List[ParameterDescription]:
        """Get parameter descriptions of model"""
class WorkerManagerFactory(BaseComponent, ABC):
    # 定义类 WorkerManagerFactory，继承自 BaseComponent 和 ABC（抽象基类）
    name = ComponentType.WORKER_MANAGER_FACTORY.value
    # 设置类属性 name，值为 ComponentType.WORKER_MANAGER_FACTORY 的值

    def init_app(self, system_app: SystemApp):
        # 定义实例方法 init_app，接收参数 system_app，并且不执行任何操作
        pass

    @abstractmethod
    def create(self) -> WorkerManager:
        """Create worker manager"""
        # 抽象方法声明 create，返回类型为 WorkerManager，用于创建 worker manager
```