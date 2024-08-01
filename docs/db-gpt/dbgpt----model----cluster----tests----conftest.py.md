# `.\DB-GPT-src\dbgpt\model\cluster\tests\conftest.py`

```py
from contextlib import asynccontextmanager, contextmanager  # 导入异步和同步上下文管理器
from typing import Dict, Iterator, List, Tuple  # 导入类型提示相关的类和函数

import pytest  # 导入 pytest 测试框架
import pytest_asyncio  # 导入 pytest 异步支持模块

from dbgpt.core import ModelMetadata, ModelOutput  # 导入核心模型元数据和模型输出类
from dbgpt.model.base import ModelInstance  # 导入模型实例基类
from dbgpt.model.cluster.registry import EmbeddedModelRegistry, ModelRegistry  # 导入嵌入式模型注册表和模型注册表
from dbgpt.model.cluster.worker.manager import (  # 导入工作节点管理器相关类和函数
    ApplyFunction,
    DeregisterFunc,
    LocalWorkerManager,
    RegisterFunc,
    SendHeartbeatFunc,
    WorkerManager,
)
from dbgpt.model.cluster.worker_base import ModelWorker  # 导入模型工作节点基类
from dbgpt.model.parameter import (  # 导入模型参数相关类和函数
    ModelParameters,
    ModelWorkerParameters,
    WorkerType,
)

@pytest.fixture
def model_registry(request):
    return EmbeddedModelRegistry()  # 返回一个嵌入式模型注册表的实例作为测试 fixture

@pytest.fixture
def model_instance():
    return ModelInstance(  # 返回一个模型实例作为测试 fixture
        model_name="test_model",
        host="192.168.1.1",
        port=5000,
    )

class MockModelWorker(ModelWorker):
    def __init__(
        self,
        model_parameters: ModelParameters,
        error_worker: bool = False,
        stop_error: bool = False,
        stream_messags: List[str] = None,
        embeddings: List[List[float]] = None,
    ) -> None:
        super().__init__()
        if not stream_messags:
            stream_messags = []  # 如果 stream_messags 为空，则初始化为空列表
        if not embeddings:
            embeddings = []  # 如果 embeddings 为空，则初始化为空列表
        self.model_parameters = model_parameters  # 设置模型参数
        self.error_worker = error_worker  # 设置错误标志位
        self.stop_error = stop_error  # 设置停止错误标志位
        self.stream_messags = stream_messags  # 设置消息流列表
        self._embeddings = embeddings  # 设置嵌入向量列表

    def parse_parameters(self, command_args: List[str] = None) -> ModelParameters:
        return self.model_parameters  # 返回模型参数

    def load_worker(self, model_name: str, model_path: str, **kwargs) -> None:
        pass  # 空方法，用于加载工作节点

    def start(
        self, model_params: ModelParameters = None, command_args: List[str] = None
    ) -> None:
        if self.error_worker:
            raise Exception("Start worker error for mock")  # 如果 error_worker 为真，则抛出异常

    def stop(self) -> None:
        if self.stop_error:
            raise Exception("Stop worker error for mock")  # 如果 stop_error 为真，则抛出异常

    def generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        full_text = ""
        for msg in self.stream_messags:
            full_text += msg  # 将消息串联起来
            yield ModelOutput(text=full_text, error_code=0)  # 生成模型输出对象的迭代器

    def generate(self, params: Dict) -> ModelOutput:
        output = None
        for out in self.generate_stream(params):
            output = out  # 获取最后一个生成的模型输出对象
        return output  # 返回生成的模型输出对象

    def count_token(self, prompt: str) -> int:
        return len(prompt)  # 返回输入字符串的字符数

    def get_model_metadata(self, params: Dict) -> ModelMetadata:
        return ModelMetadata(
            model=self.model_parameters.model_name,
        )  # 返回模型的元数据对象

    def embeddings(self, params: Dict) -> List[List[float]]:
        return self._embeddings  # 返回嵌入向量列表

_TEST_MODEL_NAME = "vicuna-13b-v1.5"
_TEST_MODEL_PATH = "/app/models/vicuna-13b-v1.5"

ClusterType = Tuple[WorkerManager, ModelRegistry]  # 定义集群类型的别名

def _new_worker_params(
    model_name: str = _TEST_MODEL_NAME,
    # 设置模型路径为测试模型的路径，默认为 _TEST_MODEL_PATH
    model_path: str = _TEST_MODEL_PATH,
    # 设置工作类型为语言生成模型的值，默认为 WorkerType.LLM.value
    worker_type: str = WorkerType.LLM.value,
# 创建新的模型工作者参数对象，并返回
def _new_worker_params(
    model_name: str,
    model_path: str,
    worker_type: str = WorkerType.LLM.value,
) -> ModelWorkerParameters:
    return ModelWorkerParameters(
        model_name=model_name, model_path=model_path, worker_type=worker_type
    )

# 创建多个模型工作者及其相关实例，并返回一个元组列表
def _create_workers(
    num_workers: int,
    error_worker: bool = False,
    stop_error: bool = False,
    worker_type: str = WorkerType.LLM.value,
    stream_messags: List[str] = None,
    embeddings: List[List[float]] = None,
    host: str = "127.0.0.1",
    start_port=8001,
) -> List[Tuple[ModelWorker, ModelWorkerParameters, ModelInstance]]:
    workers = []
    for i in range(num_workers):
        model_name = f"test-model-name-{i}"
        model_path = f"test-model-path-{i}"
        
        # 创建模型参数对象
        model_parameters = ModelParameters(model_name=model_name, model_path=model_path)
        
        # 使用模拟模型工作者创建模型工作者对象
        worker = MockModelWorker(
            model_parameters,
            error_worker=error_worker,
            stop_error=stop_error,
            stream_messags=stream_messags,
            embeddings=embeddings,
        )
        
        # 创建模型实例对象
        model_instance = ModelInstance(
            model_name=WorkerType.to_worker_key(model_name, worker_type),
            host=host,
            port=start_port + i,
            healthy=True,
        )
        
        # 创建模型工作者参数对象
        worker_params = _new_worker_params(
            model_name, model_path, worker_type=worker_type
        )
        
        # 将模型工作者、工作者参数及模型实例的元组添加到列表中
        workers.append((worker, worker_params, model_instance))
    return workers


# 异步上下文管理器，用于启动工作者管理器
@asynccontextmanager
async def _start_worker_manager(**kwargs):
    register_func = kwargs.get("register_func")
    deregister_func = kwargs.get("deregister_func")
    send_heartbeat_func = kwargs.get("send_heartbeat_func")
    model_registry = kwargs.get("model_registry")
    workers = kwargs.get("workers")
    num_workers = int(kwargs.get("num_workers", 0))
    start = kwargs.get("start", True)
    stop = kwargs.get("stop", True)
    error_worker = kwargs.get("error_worker", False)
    stop_error = kwargs.get("stop_error", False)
    stream_messags = kwargs.get("stream_messags", [])
    embeddings = kwargs.get("embeddings", [])

    # 创建本地工作者管理器对象
    worker_manager = LocalWorkerManager(
        register_func=register_func,
        deregister_func=deregister_func,
        send_heartbeat_func=send_heartbeat_func,
        model_registry=model_registry,
    )

    # 使用_create_workers函数创建模型工作者，并添加到工作者管理器中
    for worker, worker_params, model_instance in _create_workers(
        num_workers, error_worker, stop_error, stream_messags, embeddings
    ):
        worker_manager.add_worker(worker, worker_params)
    
    # 如果有预先创建的工作者，同样添加到工作者管理器中
    if workers:
        for worker, worker_params, model_instance in workers:
            worker_manager.add_worker(worker, worker_params)

    # 如果需要启动工作者管理器，则异步启动它
    if start:
        await worker_manager.start()

    yield worker_manager  # 返回工作者管理器对象，供上下文管理器使用

    # 如果需要停止工作者管理器，则异步停止它
    if stop:
        await worker_manager.stop()


# 创建模型注册表，并注册所有工作者的模型实例
async def _create_model_registry(
    workers: List[Tuple[ModelWorker, ModelWorkerParameters, ModelInstance]]
) -> ModelRegistry:
    registry = EmbeddedModelRegistry()
    for _, _, inst in workers:
        assert await registry.register_instance(inst) == True
    return registry
# 定义一个异步函数，用于管理两个工作进程
async def manager_2_workers(request):
    # 从请求中获取参数，如果没有则使用空字典
    param = getattr(request, "param", {})
    # 使用_start_worker_manager异步上下文管理器启动两个工作进程的管理器
    async with _start_worker_manager(num_workers=2, **param) as worker_manager:
        # 生成器，返回工作进程管理器
        yield worker_manager


# 使用pytest_asyncio.fixture装饰器定义一个异步的pytest fixture，管理两个工作进程
@pytest_asyncio.fixture
async def manager_with_2_workers(request):
    # 从请求中获取参数，如果没有则使用空字典
    param = getattr(request, "param", {})
    # 创建两个工作进程，并传递流消息参数给它们
    workers = _create_workers(2, stream_messags=param.get("stream_messags", []))
    # 使用_start_worker_manager异步上下文管理器启动这些工作进程的管理器
    async with _start_worker_manager(workers=workers, **param) as worker_manager:
        # 生成器，返回工作进程管理器和工作进程列表
        yield (worker_manager, workers)


# 使用pytest_asyncio.fixture装饰器定义一个异步的pytest fixture，管理两个嵌入式工作进程
@pytest_asyncio.fixture
async def manager_2_embedding_workers(request):
    # 从请求中获取参数，如果没有则使用空字典
    param = getattr(request, "param", {})
    # 创建两个文本向量化工作进程，并传递嵌入向量参数给它们
    workers = _create_workers(
        2, worker_type=WorkerType.TEXT2VEC.value, embeddings=param.get("embeddings", [])
    )
    # 使用_start_worker_manager异步上下文管理器启动这些工作进程的管理器
    async with _start_worker_manager(workers=workers, **param) as worker_manager:
        # 生成器，返回工作进程管理器和工作进程列表
        yield (worker_manager, workers)


# 定义一个异步上下文管理器函数，创建一个新的集群
@asynccontextmanager
async def _new_cluster(**kwargs) -> ClusterType:
    # 获取num_workers参数，默认为0
    num_workers = kwargs.get("num_workers", 0)
    # 根据num_workers参数创建工作进程，并传递流消息参数给它们
    workers = _create_workers(
        num_workers, stream_messags=kwargs.get("stream_messags", [])
    )
    # 如果kwargs中包含num_workers键，则删除它
    if "num_workers" in kwargs:
        del kwargs["num_workers"]
    # 创建模型注册表，等待其完成
    registry = await _create_model_registry(
        workers,
    )
    # 使用_start_worker_manager异步上下文管理器启动这些工作进程的管理器
    async with _start_worker_manager(workers=workers, **kwargs) as worker_manager:
        # 生成器，返回工作进程管理器和模型注册表
        yield (worker_manager, registry)


# 使用pytest_asyncio.fixture装饰器定义一个异步的pytest fixture，管理两个工作进程的集群
@pytest_asyncio.fixture
async def cluster_2_workers(request):
    # 从请求中获取参数，如果没有则使用空字典
    param = getattr(request, "param", {})
    # 创建两个工作进程
    workers = _create_workers(2)
    # 创建模型注册表，等待其完成
    registry = await _create_model_registry(workers)
    # 使用_start_worker_manager异步上下文管理器启动这些工作进程的管理器
    async with _start_worker_manager(workers=workers, **param) as worker_manager:
        # 生成器，返回工作进程管理器和模型注册表
        yield (worker_manager, registry)
```