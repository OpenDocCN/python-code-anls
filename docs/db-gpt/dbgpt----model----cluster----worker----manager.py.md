# `.\DB-GPT-src\dbgpt\model\cluster\worker\manager.py`

```py
import asyncio  # 导入异步IO库，用于处理异步任务
import itertools  # 导入工具函数库，用于生成迭代器
import json  # 导入JSON库，用于处理JSON格式数据
import logging  # 导入日志库，用于记录程序运行时的信息
import os  # 导入操作系统库，用于与操作系统交互
import random  # 导入随机数库，用于生成随机数
import sys  # 导入系统库，用于访问系统相关信息
import time  # 导入时间库，用于时间相关操作
import traceback  # 导入追踪库，用于获取异常的调用堆栈信息
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器，用于并发执行任务
from dataclasses import asdict  # 导入数据类操作函数，用于将数据类转换为字典
from typing import Awaitable, Callable, Iterator  # 导入类型定义，用于声明回调函数的类型

from fastapi import APIRouter  # 导入FastAPI的API路由模块
from fastapi.responses import StreamingResponse  # 导入FastAPI的流响应模块

from dbgpt.component import SystemApp  # 导入调试组件模块中的系统应用类
from dbgpt.configs.model_config import LOGDIR  # 导入调试组件模型配置中的日志目录常量
from dbgpt.core import ModelMetadata, ModelOutput  # 导入调试组件核心模块中的模型元数据和模型输出类
from dbgpt.model.base import ModelInstance, WorkerApplyOutput, WorkerSupportedModel  # 导入调试组件模型基础模块中的模型实例和相关输出类
from dbgpt.model.cluster.base import *  # 导入调试组件集群基础模块中的所有内容
from dbgpt.model.cluster.manager_base import (  # 导入调试组件集群管理基础模块中的相关类和函数
    WorkerManager,
    WorkerManagerFactory,
    WorkerRunData,
)
from dbgpt.model.cluster.registry import ModelRegistry  # 导入调试组件集群注册模块中的模型注册类
from dbgpt.model.cluster.worker_base import ModelWorker  # 导入调试组件集群工作模块中的模型工作类
from dbgpt.model.parameter import ModelWorkerParameters, WorkerType  # 导入调试组件模型参数模块中的相关类和类型
from dbgpt.model.utils.llm_utils import list_supported_models  # 导入调试组件模型工具模块中的列出支持的模型函数
from dbgpt.util.fastapi import create_app, register_event_handler  # 导入调试组件FastAPI工具模块中的创建应用和注册事件处理器函数
from dbgpt.util.parameter_utils import (  # 导入调试组件参数工具模块中的参数解析器和相关工具函数
    EnvArgumentParser,
    ParameterDescription,
    _dict_to_command_args,
    _get_dict_from_obj,
)
from dbgpt.util.system_utils import get_system_info  # 导入调试组件系统工具模块中的获取系统信息函数
from dbgpt.util.tracer import (  # 导入调试组件追踪工具模块中的追踪器初始化和根追踪器对象
    SpanType,
    SpanTypeRunName,
    initialize_tracer,
    root_tracer,
)
from dbgpt.util.utils import setup_http_service_logging, setup_logging  # 导入调试组件工具模块中的设置HTTP服务日志和设置日志函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

RegisterFunc = Callable[[WorkerRunData], Awaitable[None]]  # 定义注册函数类型别名，接受WorkerRunData参数，返回异步无返回值
DeregisterFunc = Callable[[WorkerRunData], Awaitable[None]]  # 定义注销函数类型别名，接受WorkerRunData参数，返回异步无返回值
SendHeartbeatFunc = Callable[[WorkerRunData], Awaitable[None]]  # 定义发送心跳函数类型别名，接受WorkerRunData参数，返回异步无返回值
ApplyFunction = Callable[[WorkerRunData], Awaitable[None]]  # 定义应用函数类型别名，接受WorkerRunData参数，返回异步无返回值


async def _async_heartbeat_sender(
    worker_run_data: WorkerRunData,
    heartbeat_interval,
    send_heartbeat_func: SendHeartbeatFunc,
):
    """
    异步心跳发送器函数，定期发送心跳消息到工作器管理对象。

    Args:
        worker_run_data (WorkerRunData): 工作器运行数据对象，包含运行状态信息和事件对象。
        heartbeat_interval (float): 发送心跳的间隔时间。
        send_heartbeat_func (SendHeartbeatFunc): 发送心跳消息的回调函数。

    """
    while not worker_run_data.stop_event.is_set():
        try:
            await send_heartbeat_func(worker_run_data)  # 调用发送心跳消息的回调函数
        except Exception as e:
            logger.warn(f"Send heartbeat func error: {str(e)}")  # 记录发送心跳消息时的异常信息
        finally:
            await asyncio.sleep(heartbeat_interval)  # 等待指定的心跳间隔时间


class LocalWorkerManager(WorkerManager):
    def __init__(
        self,
        register_func: RegisterFunc = None,
        deregister_func: DeregisterFunc = None,
        send_heartbeat_func: SendHeartbeatFunc = None,
        model_registry: ModelRegistry = None,
        host: str = None,
        port: int = None,
    ) -> None:
        # 初始化一个空字典，用于存储不同类型工作线程的运行数据列表
        self.workers: Dict[str, List[WorkerRunData]] = dict()
        # 创建线程池执行器，最大工作线程数为 CPU 核心数的 5 倍
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 5)
        # 设置注册函数、注销函数、发送心跳函数、模型注册表、主机地址和端口号
        self.register_func = register_func
        self.deregister_func = deregister_func
        self.send_heartbeat_func = send_heartbeat_func
        self.model_registry = model_registry
        self.host = host
        self.port = port
        # 初始化空的启动监听器列表
        self.start_listeners = []

        # 创建 WorkerRunData 对象，用于存储本工作线程的运行数据
        self.run_data = WorkerRunData(
            host=self.host,
            port=self.port,
            # 生成工作线程的唯一标识符，由工作线程类型和模型名称组成
            worker_key=self._worker_key(
                WORKER_MANAGER_SERVICE_TYPE, WORKER_MANAGER_SERVICE_NAME
            ),
            worker=None,
            worker_params=None,
            model_params=None,
            stop_event=asyncio.Event(),
            semaphore=None,
            command_args=None,
        )

    # 根据工作线程类型和模型名称生成唯一的工作线程标识符
    def _worker_key(self, worker_type: str, model_name: str) -> str:
        return WorkerType.to_worker_key(model_name, worker_type)

    # 异步方法，用于运行阻塞的函数
    async def run_blocking_func(self, func, *args):
        # 如果传入的函数是协程函数，抛出异常，因为此方法用于阻塞函数
        if asyncio.iscoroutinefunction(func):
            raise ValueError(f"The function {func} is not blocking function")
        # 获取当前事件循环对象
        loop = asyncio.get_event_loop()
        # 在线程池执行器中运行指定的阻塞函数，并返回其结果
        return await loop.run_in_executor(self.executor, func, *args)

    # 异步方法，用于启动整个工作线程管理器
    async def start(self):
        # 如果存在已注册的工作线程，尝试启动所有工作线程
        if len(self.workers) > 0:
            out = await self._start_all_worker(apply_req=None)
            # 如果启动不成功，抛出异常
            if not out.success:
                raise Exception(out.message)
        # 如果有注册函数，执行注册函数并传入当前工作线程的运行数据
        if self.register_func:
            await self.register_func(self.run_data)
        # 如果有发送心跳函数，创建一个异步任务发送心跳包
        if self.send_heartbeat_func:
            asyncio.create_task(
                _async_heartbeat_sender(self.run_data, 20, self.send_heartbeat_func)
            )
        # 逐个执行启动监听器列表中的监听器函数
        for listener in self.start_listeners:
            if asyncio.iscoroutinefunction(listener):
                await listener(self)
            else:
                listener(self)
    # 异步方法，停止所有工作线程
    async def stop(self, ignore_exception: bool = False):
        # 如果停止事件未设置，则执行以下操作
        if not self.run_data.stop_event.is_set():
            # 记录日志信息
            logger.info("Stop all workers")
            # 清除停止事件
            self.run_data.stop_event.clear()
            # 创建停止任务列表
            stop_tasks = []
            # 添加停止所有工作线程的任务到列表中
            stop_tasks.append(
                self._stop_all_worker(apply_req=None, ignore_exception=ignore_exception)
            )
            # 如果存在注销函数
            if self.deregister_func:
                # 如果 ignore_exception 为 True，则使用异常处理来忽略从 self.deregister_func 抛出的任何异常
                if ignore_exception:
                    # 安全地执行注销函数
                    async def safe_deregister_func(run_data):
                        try:
                            await self.deregister_func(run_data)
                        except Exception as e:
                            logger.warning(
                                f"Stop worker, ignored exception from deregister_func: {e}"
                            )
                    # 将安全的注销函数添加到停止任务列表中
                    stop_tasks.append(safe_deregister_func(self.run_data))
                else:
                    # 将注销函数添加到停止任务列表中
                    stop_tasks.append(self.deregister_func(self.run_data))
            # 并行执行所有停止任务，并获取结果
            results = await asyncio.gather(*stop_tasks)
            # 如果第一个结果不成功且不忽略异常，则抛出异常
            if not results[0].success and not ignore_exception:
                raise Exception(results[0].message)

    # 在启动后执行的方法，添加监听器到启动监听器列表中
    def after_start(self, listener: Callable[["WorkerManager"], None]):
        self.start_listeners.append(listener)

    # 添加工作线程到工作线程管理器中
    def add_worker(
        self,
        worker: ModelWorker,
        worker_params: ModelWorkerParameters,
        command_args: List[str] = None,
    ) -> bool:
        # 如果没有指定命令参数，则使用系统参数
        if not command_args:
            command_args = sys.argv[1:]
        # 加载工作线程
        worker.load_worker(**asdict(worker_params))
        # 如果未指定工作线程类型，则获取默认类型
        if not worker_params.worker_type:
            worker_params.worker_type = worker.worker_type()
        # 如果工作线程类型为枚举类型，则获取其值
        if isinstance(worker_params.worker_type, WorkerType):
            worker_params.worker_type = worker_params.worker_type.value
        # 生成工作线程的唯一标识
        worker_key = self._worker_key(
            worker_params.worker_type, worker_params.model_name
        )
        # 从持久存储加载模型参数
        model_params = worker.parse_parameters(command_args=command_args)
        # 创建工作线程运行数据对象
        worker_run_data = WorkerRunData(
            host=self.host,
            port=self.port,
            worker_key=worker_key,
            worker=worker,
            worker_params=worker_params,
            model_params=model_params,
            stop_event=asyncio.Event(),
            semaphore=asyncio.Semaphore(worker_params.limit_model_concurrency),
            command_args=command_args,
        )
        # 获取工作线程实例列表
        instances = self.workers.get(worker_key)
        # 如果实例列表为空，则初始化并返回 True
        if not instances:
            instances = [worker_run_data]
            self.workers[worker_key] = instances
            logger.info(f"Init empty instances list for {worker_key}")
            return True
        else:
            # TODO 更新工作线程
            logger.warning(f"Instance {worker_key} exist")
            return False
    # 根据给定的 worker_params 删除对应的 worker
    def _remove_worker(self, worker_params: ModelWorkerParameters) -> None:
        # 构造 worker 的唯一标识 key，格式为 worker_type@model_name
        worker_key = self._worker_key(
            worker_params.worker_type, worker_params.model_name
        )
        # 获取指定 worker_key 下的所有实例
        instances = self.workers.get(worker_key)
        # 如果存在实例，则从 self.workers 中删除该 worker
        if instances:
            del self.workers[worker_key]

    async def model_startup(self, startup_req: WorkerStartupRequest):
        """启动模型"""
        # 获取启动请求中的模型名称、工作类型和参数
        model_name = startup_req.model
        worker_type = startup_req.worker_type
        params = startup_req.params
        # 记录调试日志，显示启动模型的详细信息
        logger.debug(
            f"start model, model name {model_name}, worker type {worker_type},  params: {params}"
        )
        # 从参数字典创建 ModelWorkerParameters 对象
        worker_params: ModelWorkerParameters = ModelWorkerParameters.from_dict(
            params, ignore_extra_fields=True
        )
        # 如果参数中未指定 model_name，则使用启动请求中的模型名称
        if not worker_params.model_name:
            worker_params.model_name = model_name
        # 根据 worker_params 构建 worker 对象
        worker = _build_worker(worker_params)
        # 将参数转换为命令行参数形式
        command_args = _dict_to_command_args(params)
        # 使用异步方法运行阻塞函数 self.add_worker，尝试添加 worker
        success = await self.run_blocking_func(
            self.add_worker, worker, worker_params, command_args
        )
        # 如果添加 worker 不成功，则记录警告日志，删除对应的 worker，抛出异常
        if not success:
            msg = f"Add worker {model_name}@{worker_type}, worker instances is exist"
            logger.warning(f"{msg}, worker_params: {worker_params}")
            self._remove_worker(worker_params)
            raise Exception(msg)
        # 获取支持的所有 worker 类型
        supported_types = WorkerType.values()
        # 如果当前 worker_type 不在支持的类型列表中，则删除对应的 worker，并抛出异常
        if worker_type not in supported_types:
            self._remove_worker(worker_params)
            raise ValueError(
                f"Unsupported worker type: {worker_type}, now supported worker type: {supported_types}"
            )
        # 创建一个启动应用请求
        start_apply_req = WorkerApplyRequest(
            model=worker_params.model_name,
            apply_type=WorkerApplyType.START,
            worker_type=worker_type,
        )
        out: WorkerApplyOutput = None
        # 尝试向 worker 应用启动请求，并处理可能的异常
        try:
            out = await self.worker_apply(start_apply_req)
        except Exception as e:
            # 如果出现异常，则删除对应的 worker，并重新抛出异常
            self._remove_worker(worker_params)
            raise e
        # 如果启动应用请求不成功，则删除对应的 worker，并抛出异常
        if not out.success:
            self._remove_worker(worker_params)
            raise Exception(out.message)

    async def model_shutdown(self, shutdown_req: WorkerStartupRequest):
        # 记录信息日志，显示开始关闭模型的请求
        logger.info(f"Begin shutdown model, shutdown_req: {shutdown_req}")
        # 创建一个停止应用请求
        apply_req = WorkerApplyRequest(
            model=shutdown_req.model,
            apply_type=WorkerApplyType.STOP,
            worker_type=shutdown_req.worker_type,
        )
        # 使用异步方法停止所有 worker，并获取操作结果
        out = await self._stop_all_worker(apply_req)
        # 如果操作不成功，则抛出异常
        if not out.success:
            raise Exception(out.message)

    async def supported_models(self) -> List[WorkerSupportedModel]:
        # 使用阻塞函数列出所有支持的模型，并返回 WorkerSupportedModel 列表
        models = await self.run_blocking_func(list_supported_models)
        return [WorkerSupportedModel(host=self.host, port=self.port, models=models)]

    async def get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ):
        # 略，未完成的函数，需要进一步补充完整
    # 返回指定工作者类型、模型名称的所有运行实例列表
    ) -> List[WorkerRunData]:
        return self.sync_get_model_instances(worker_type, model_name, healthy_only)

    # 异步获取指定工作者类型的所有运行实例列表
    async def get_all_model_instances(
        self, worker_type: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 将所有实例列表展开成一个列表
        instances = list(itertools.chain(*self.workers.values()))
        result = []
        # 遍历实例列表
        for instance in instances:
            # 解析实例的工作者键，获取名称和类型
            name, wt = WorkerType.parse_worker_key(instance.worker_key)
            # 如果类型不匹配或者需要健康实例且实例已停止，则跳过
            if wt != worker_type or (healthy_only and instance.stopped):
                continue
            # 将符合条件的实例添加到结果列表中
            result.append(instance)
        return result

    # 同步获取指定工作者类型、模型名称的所有运行实例列表
    def sync_get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 根据工作者类型和模型名称获取工作者键
        worker_key = self._worker_key(worker_type, model_name)
        return self.workers.get(worker_key, [])

    # 简单选择一个实例
    def _simple_select(
        self, worker_type: str, model_name: str, worker_instances: List[WorkerRunData]
    ) -> WorkerRunData:
        # 如果实例列表为空，则抛出异常
        if not worker_instances:
            raise Exception(
                f"Cound not found worker instances for model name {model_name} and worker type {worker_type}"
            )
        # 随机选择一个实例并返回
        worker_run_data = random.choice(worker_instances)
        return worker_run_data

    # 异步选择一个实例
    async def select_one_instance(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> WorkerRunData:
        # 异步获取指定工作者类型、模型名称的所有运行实例列表
        worker_instances = await self.get_model_instances(
            worker_type, model_name, healthy_only
        )
        # 调用简单选择方法选择一个实例并返回
        return self._simple_select(worker_type, model_name, worker_instances)

    # 同步选择一个实例
    def sync_select_one_instance(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> WorkerRunData:
        # 同步获取指定工作者类型、模型名称的所有运行实例列表
        worker_instances = self.sync_get_model_instances(
            worker_type, model_name, healthy_only
        )
        # 调用简单选择方法选择一个实例并返回
        return self._simple_select(worker_type, model_name, worker_instances)

    # 异步获取模型
    async def _get_model(self, params: Dict, worker_type: str = "llm") -> WorkerRunData:
        # 获取参数中的模型名称
        model = params.get("model")
        # 如果模型名称为空，则抛出异常
        if not model:
            raise Exception("Model name count not be empty")
        # 异步选择一个实例并返回
        return await self.select_one_instance(worker_type, model, healthy_only=True)

    # 同步获取模型
    def _sync_get_model(self, params: Dict, worker_type: str = "llm") -> WorkerRunData:
        # 获取参数中的模型名称
        model = params.get("model")
        # 如果模型名称为空，则抛出异常
        if not model:
            raise Exception("Model name count not be empty")
        # 同步选择一个实例并返回
        return self.sync_select_one_instance(worker_type, model, healthy_only=True)

    # 生成流
    async def generate_stream(
        self, params: Dict, async_wrapper=None, **kwargs
    ) -> Iterator[ModelOutput]:
        """Generate stream result, chat scene"""
        # 使用根跟踪器创建一个新的跟踪 span，命名为"WorkerManager.generate_stream"，并使用传入的span_id参数
        with root_tracer.start_span(
            "WorkerManager.generate_stream", params.get("span_id")
        ) as span:
            # 将当前生成的span_id保存到params字典中，以便后续使用
            params["span_id"] = span.span_id
            try:
                # 尝试获取模型数据
                worker_run_data = await self._get_model(params)
            except Exception as e:
                # 如果出现异常，生成一个ModelOutput对象作为输出，指示错误信息
                yield ModelOutput(
                    text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                    error_code=1,
                )
                return
            # 使用worker_run_data中的信号量进入异步上下文管理
            async with worker_run_data.semaphore:
                # 如果worker支持异步操作
                if worker_run_data.worker.support_async():
                    # 使用异步生成流方法生成输出流，并逐个产出输出
                    async for outout in worker_run_data.worker.async_generate_stream(
                        params
                    ):
                        yield outout
                else:
                    # 如果不支持异步操作且未提供async_wrapper，则引入iterate_in_threadpool方法
                    if not async_wrapper:
                        from starlette.concurrency import iterate_in_threadpool

                        async_wrapper = iterate_in_threadpool
                    # 使用async_wrapper方法，在线程池中运行同步生成流方法，逐个产出输出
                    async for output in async_wrapper(
                        worker_run_data.worker.generate_stream(params)
                    ):
                        yield output

    async def generate(self, params: Dict) -> ModelOutput:
        """Generate non stream result"""
        # 使用根跟踪器创建一个新的跟踪 span，命名为"WorkerManager.generate"，并使用传入的span_id参数
        with root_tracer.start_span(
            "WorkerManager.generate", params.get("span_id")
        ) as span:
            # 将当前生成的span_id保存到params字典中，以便后续使用
            params["span_id"] = span.span_id
            try:
                # 尝试获取模型数据
                worker_run_data = await self._get_model(params)
            except Exception as e:
                # 如果出现异常，生成一个ModelOutput对象作为输出，指示错误信息
                return ModelOutput(
                    text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                    error_code=1,
                )
            # 使用worker_run_data中的信号量进入异步上下文管理
            async with worker_run_data.semaphore:
                # 如果worker支持异步操作
                if worker_run_data.worker.support_async():
                    # 使用worker的异步生成方法生成非流结果
                    return await worker_run_data.worker.async_generate(params)
                else:
                    # 使用run_blocking_func方法，在线程池中运行worker的生成方法，得到非流结果
                    return await self.run_blocking_func(
                        worker_run_data.worker.generate, params
                    )

    async def embeddings(self, params: Dict) -> List[List[float]]:
        """Embed input"""
        # 使用根跟踪器创建一个新的跟踪 span，命名为"WorkerManager.embeddings"，并使用传入的span_id参数
        with root_tracer.start_span(
            "WorkerManager.embeddings", params.get("span_id")
        ) as span:
            # 将当前生成的span_id保存到params字典中，以便后续使用
            params["span_id"] = span.span_id
            try:
                # 尝试获取模型数据，指定worker类型为"text2vec"
                worker_run_data = await self._get_model(params, worker_type="text2vec")
            except Exception as e:
                # 如果出现异常，将异常继续抛出
                raise e
            # 使用worker_run_data中的信号量进入异步上下文管理
            async with worker_run_data.semaphore:
                # 如果worker支持异步操作
                if worker_run_data.worker.support_async():
                    # 使用worker的异步嵌入方法处理输入参数，返回嵌入结果
                    return await worker_run_data.worker.async_embeddings(params)
                else:
                    # 使用run_blocking_func方法，在线程池中运行worker的嵌入方法，返回嵌入结果
                    return await self.run_blocking_func(
                        worker_run_data.worker.embeddings, params
                    )
    # 同步更新嵌入向量数据
    def sync_embeddings(self, params: Dict) -> List[List[float]]:
        # 调用内部方法获取文本向量化模型的运行数据
        worker_run_data = self._sync_get_model(params, worker_type="text2vec")
        # 调用工作器对象的嵌入向量方法，返回结果
        return worker_run_data.worker.embeddings(params)

    # 异步计算提示语的标记数量
    async def count_token(self, params: Dict) -> int:
        """Count token of prompt"""
        # 使用分布式追踪器开始一个跟踪 span，命名为 "WorkerManager.count_token"
        with root_tracer.start_span(
            "WorkerManager.count_token", params.get("span_id")
        ) as span:
            # 将 span_id 加入参数中，以便跟踪
            params["span_id"] = span.span_id
            try:
                # 异步获取模型运行数据
                worker_run_data = await self._get_model(params)
            except Exception as e:
                # 捕获异常并抛出
                raise e
            # 获取提示语
            prompt = params.get("prompt")
            # 使用 worker_run_data 中的信号量进行异步上下文管理
            async with worker_run_data.semaphore:
                # 如果 worker 支持异步操作，则调用异步计数方法
                if worker_run_data.worker.support_async():
                    return await worker_run_data.worker.async_count_token(prompt)
                else:
                    # 否则调用阻塞函数来计数标记
                    return await self.run_blocking_func(
                        worker_run_data.worker.count_token, prompt
                    )

    # 异步获取模型元数据
    async def get_model_metadata(self, params: Dict) -> ModelMetadata:
        """Get model metadata"""
        # 使用分布式追踪器开始一个跟踪 span，命名为 "WorkerManager.get_model_metadata"
        with root_tracer.start_span(
            "WorkerManager.get_model_metadata", params.get("span_id")
        ) as span:
            # 将 span_id 加入参数中，以便跟踪
            params["span_id"] = span.span_id
            try:
                # 异步获取模型运行数据
                worker_run_data = await self._get_model(params)
            except Exception as e:
                # 捕获异常并抛出
                raise e
            # 使用 worker_run_data 中的信号量进行异步上下文管理
            async with worker_run_data.semaphore:
                # 如果 worker 支持异步操作，则调用异步获取模型元数据方法
                if worker_run_data.worker.support_async():
                    return await worker_run_data.worker.async_get_model_metadata(params)
                else:
                    # 否则调用阻塞函数来获取模型元数据
                    return await self.run_blocking_func(
                        worker_run_data.worker.get_model_metadata, params
                    )

    # 异步应用工作器操作请求
    async def worker_apply(self, apply_req: WorkerApplyRequest) -> WorkerApplyOutput:
        # 根据请求类型选择对应的应用函数
        apply_func: Callable[[WorkerApplyRequest], Awaitable[str]] = None
        if apply_req.apply_type == WorkerApplyType.START:
            apply_func = self._start_all_worker
        elif apply_req.apply_type == WorkerApplyType.STOP:
            apply_func = self._stop_all_worker
        elif apply_req.apply_type == WorkerApplyType.RESTART:
            apply_func = self._restart_all_worker
        elif apply_req.apply_type == WorkerApplyType.UPDATE_PARAMS:
            apply_func = self._update_all_worker_params
        else:
            # 如果请求类型不支持，则抛出值错误异常
            raise ValueError(f"Unsupported apply type {apply_req.apply_type}")
        # 执行选定的应用函数并返回结果
        return await apply_func(apply_req)

    # 获取工作器参数的描述信息
    async def parameter_descriptions(
        self, worker_type: str, model_name: str
    ) -> List[ParameterDescription]:
        # 调用 self.get_model_instances 方法获取特定类型和模型名称的工作实例列表
        worker_instances = await self.get_model_instances(worker_type, model_name)
        # 如果没有找到工作实例，抛出异常
        if not worker_instances:
            raise Exception(
                f"Not worker instances for model name {model_name} worker type {worker_type}"
            )
        # 获取第一个工作实例的运行数据
        worker_run_data = worker_instances[0]
        # 调用工作实例的 worker.parameter_descriptions() 方法获取参数描述列表
        return worker_run_data.worker.parameter_descriptions()

    async def _apply_worker(
        self, apply_req: WorkerApplyRequest, apply_func: ApplyFunction
    ) -> None:
        """Apply function to worker instances in parallel

        Args:
            apply_req (WorkerApplyRequest): Worker apply request
            apply_func (ApplyFunction): Function to apply to worker instances, now function is async function
        """
        # 记录应用请求和应用函数的信息
        logger.info(f"Apply req: {apply_req}, apply_func: {apply_func}")
        # 如果应用请求存在
        if apply_req:
            # 获取工作者类型和模型名称
            worker_type = apply_req.worker_type.value
            model_name = apply_req.model
            # 调用 self.get_model_instances 方法获取指定类型和模型名称的工作实例列表，包括不健康的实例
            worker_instances = await self.get_model_instances(
                worker_type, model_name, healthy_only=False
            )
            # 如果没有找到工作实例，抛出异常
            if not worker_instances:
                raise Exception(
                    f"No worker instance found for the model {model_name} worker type {worker_type}"
                )
        else:
            # 应用到所有工作实例
            worker_instances = list(itertools.chain(*self.workers.values()))
            logger.info(f"Apply to all workers")
        # 并行应用给定的函数到所有工作实例，使用 asyncio.gather 进行等待
        return await asyncio.gather(
            *(apply_func(worker) for worker in worker_instances)
        )

    async def _start_all_worker(
        self, apply_req: WorkerApplyRequest
    ) -> WorkerApplyOutput:
        from httpx import TimeoutException, TransportError

        # TODO avoid start twice
        start_time = time.time()  # 记录函数开始时间
        logger.info(f"Begin start all worker, apply_req: {apply_req}")  # 记录开始启动所有工作者的信息，包括请求参数 apply_req

        async def _start_worker(worker_run_data: WorkerRunData):
            _start_time = time.time()  # 记录单个工作者启动的开始时间
            info = worker_run_data._to_print_key()  # 获取工作者运行数据的打印信息
            out = WorkerApplyOutput("")  # 初始化工作者应用输出对象
            try:
                await self.run_blocking_func(  # 调用异步方法运行阻塞函数
                    worker_run_data.worker.start,  # 调用工作者对象的启动方法
                    worker_run_data.model_params,  # 传入模型参数
                    worker_run_data.command_args,  # 传入命令行参数
                )
                worker_run_data.stop_event.clear()  # 清除工作者停止事件的状态
                if worker_run_data.worker_params.register and self.register_func:
                    # 如果工作者需要注册且注册函数存在，则注册工作者到控制器
                    await self.register_func(worker_run_data)
                    if (
                        worker_run_data.worker_params.send_heartbeat
                        and self.send_heartbeat_func
                    ):
                        # 如果工作者需要发送心跳且心跳发送函数存在，则创建异步任务发送心跳
                        asyncio.create_task(
                            _async_heartbeat_sender(
                                worker_run_data,
                                worker_run_data.worker_params.heartbeat_interval,
                                self.send_heartbeat_func,
                            )
                        )
                out.message = f"{info} start successfully"  # 设置成功消息
            except TimeoutException as e:
                out.success = False  # 设置输出为失败
                out.message = (
                    f"{info} start failed for network timeout, please make "
                    f"sure your port is available, if you are using global network "
                    f"proxy, please close it"
                )  # 设置超时异常的失败消息
            except TransportError as e:
                out.success = False  # 设置输出为失败
                out.message = (
                    f"{info} start failed for network error, please make "
                    f"sure your port is available, if you are using global network "
                    "proxy, please close it"
                )  # 设置传输错误的失败消息
            except Exception:
                err_msg = traceback.format_exc()  # 获取异常的详细信息
                out.success = False  # 设置输出为失败
                out.message = f"{info} start failed, {err_msg}"  # 设置未知异常的失败消息
            finally:
                out.timecost = time.time() - _start_time  # 计算单个工作者启动消耗的时间
            return out  # 返回工作者应用输出对象

        outs = await self._apply_worker(apply_req, _start_worker)  # 调用应用工作者方法来应用所有工作者
        out = WorkerApplyOutput.reduce(outs)  # 聚合所有工作者的应用输出对象
        out.timecost = time.time() - start_time  # 计算整体启动消耗的时间
        return out  # 返回整体应用输出对象
    ) -> WorkerApplyOutput:
        start_time = time.time()  # 记录函数开始时间

        async def _stop_worker(worker_run_data: WorkerRunData):
            _start_time = time.time()  # 记录内部函数开始时间
            info = worker_run_data._to_print_key()  # 获取工作器运行数据的打印键信息
            out = WorkerApplyOutput("")  # 初始化一个输出对象
            try:
                await self.run_blocking_func(worker_run_data.worker.stop)  # 异步运行阻塞函数停止工作器
                # 设置停止事件
                worker_run_data.stop_event.set()
                if worker_run_data._heartbeat_future:
                    # 等待线程完成
                    worker_run_data._heartbeat_future.result()
                    worker_run_data._heartbeat_future = None
                if (
                    worker_run_data.worker_params.register
                    and self.register_func
                    and self.deregister_func
                ):
                    _deregister_func = self.deregister_func
                    if ignore_exception:
                        
                        async def safe_deregister_func(run_data):
                            try:
                                await self.deregister_func(run_data)
                            except Exception as e:
                                logger.warning(
                                    f"Stop worker, ignored exception from deregister_func: {e}"
                                )
                        
                        _deregister_func = safe_deregister_func
                    await _deregister_func(worker_run_data)  # 异步执行安全的取消注册函数
                # 移除元数据
                self._remove_worker(worker_run_data.worker_params)
                out.message = f"{info} stop successfully"  # 设置输出消息为成功停止信息
            except Exception as e:
                out.success = False  # 标记操作失败
                out.message = f"{info} stop failed, {str(e)}"  # 设置输出消息为停止失败信息
            finally:
                out.timecost = time.time() - _start_time  # 计算内部函数执行时间
            return out  # 返回输出对象

        outs = await self._apply_worker(apply_req, _stop_worker)  # 调用内部函数处理应用请求
        out = WorkerApplyOutput.reduce(outs)  # 对多个输出对象进行合并
        out.timecost = time.time() - start_time  # 计算函数总体执行时间
        return out  # 返回最终输出对象

    async def _restart_all_worker(
        self, apply_req: WorkerApplyRequest
    ) -> WorkerApplyOutput:
        out = await self._stop_all_worker(apply_req, ignore_exception=True)  # 停止所有工作器
        if not out.success:
            return out  # 如果停止失败，则直接返回失败输出对象
        return await self._start_all_worker(apply_req)  # 否则，重新启动所有工作器

    async def _update_all_worker_params(
        self, apply_req: WorkerApplyRequest
    ) -> WorkerApplyOutput:
        # 记录开始执行的时间点
        start_time = time.time()
        # 是否需要重新启动的标志，默认为 False
        need_restart = False

        async def update_params(worker_run_data: WorkerRunData):
            nonlocal need_restart
            # 获取请求中的新参数
            new_params = apply_req.params
            # 如果没有新参数，直接返回
            if not new_params:
                return
            # 更新模型参数，并检查是否有更新
            if worker_run_data.model_params.update_from(new_params):
                need_restart = True

        # 调用内部方法来应用工作参数更新
        await self._apply_worker(apply_req, update_params)
        # 默认消息为成功更新工作参数
        message = f"Update worker params successfully"
        # 计算总耗时
        timecost = time.time() - start_time
        # 如果需要重新启动
        if need_restart:
            # 记录日志，表示模型参数更新成功，并开始重新启动工作进程
            logger.info("Model params update successfully, begin restart worker")
            # 异步重新启动所有工作进程
            await self._restart_all_worker(apply_req)
            # 重新计算总耗时
            timecost = time.time() - start_time
            # 更新消息为成功更新工作参数并重新启动
            message = f"Update worker params and restart successfully"
        # 返回更新结果对象
        return WorkerApplyOutput(message=message, timecost=timecost)
class WorkerManagerAdapter(WorkerManager):
    # WorkerManagerAdapter 类继承自 WorkerManager 类，用于适配工作管理器

    def __init__(self, worker_manager: WorkerManager = None) -> None:
        # 初始化方法，接收一个 worker_manager 参数作为 WorkerManager 类的实例，默认为 None
        self.worker_manager = worker_manager

    async def start(self):
        # 异步方法，调用 worker_manager 的 start 方法并返回结果
        return await self.worker_manager.start()

    async def stop(self, ignore_exception: bool = False):
        # 异步方法，调用 worker_manager 的 stop 方法并返回结果，可以选择是否忽略异常
        return await self.worker_manager.stop(ignore_exception=ignore_exception)

    def after_start(self, listener: Callable[["WorkerManager"], None]):
        # 设置在 worker_manager 启动后执行的回调函数 listener
        if listener is not None:
            self.worker_manager.after_start(listener)

    async def supported_models(self) -> List[WorkerSupportedModel]:
        # 异步方法，获取支持的模型列表，调用 worker_manager 的 supported_models 方法
        return await self.worker_manager.supported_models()

    async def model_startup(self, startup_req: WorkerStartupRequest):
        # 异步方法，启动指定模型，调用 worker_manager 的 model_startup 方法
        return await self.worker_manager.model_startup(startup_req)

    async def model_shutdown(self, shutdown_req: WorkerStartupRequest):
        # 异步方法，关闭指定模型，调用 worker_manager 的 model_shutdown 方法
        return await self.worker_manager.model_shutdown(shutdown_req)

    async def get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 异步方法，获取指定类型和名称的模型实例列表，调用 worker_manager 的 get_model_instances 方法
        return await self.worker_manager.get_model_instances(
            worker_type, model_name, healthy_only
        )

    async def get_all_model_instances(
        self, worker_type: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 异步方法，获取指定类型的所有模型实例列表，调用 worker_manager 的 get_all_model_instances 方法
        return await self.worker_manager.get_all_model_instances(
            worker_type, healthy_only
        )

    def sync_get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 同步方法，获取指定类型和名称的模型实例列表，调用 worker_manager 的 sync_get_model_instances 方法
        return self.worker_manager.sync_get_model_instances(
            worker_type, model_name, healthy_only
        )

    async def select_one_instance(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> WorkerRunData:
        # 异步方法，选择一个模型实例，调用 worker_manager 的 select_one_instance 方法
        return await self.worker_manager.select_one_instance(
            worker_type, model_name, healthy_only
        )

    def sync_select_one_instance(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> WorkerRunData:
        # 同步方法，选择一个模型实例，调用 worker_manager 的 sync_select_one_instance 方法
        return self.worker_manager.sync_select_one_instance(
            worker_type, model_name, healthy_only
        )

    async def generate_stream(self, params: Dict, **kwargs) -> Iterator[ModelOutput]:
        # 异步生成器方法，生成模型输出流，调用 worker_manager 的 generate_stream 方法
        async for output in self.worker_manager.generate_stream(params, **kwargs):
            yield output

    async def generate(self, params: Dict) -> ModelOutput:
        # 异步方法，生成模型输出，调用 worker_manager 的 generate 方法
        return await self.worker_manager.generate(params)

    async def embeddings(self, params: Dict) -> List[List[float]]:
        # 异步方法，获取嵌入向量，调用 worker_manager 的 embeddings 方法
        return await self.worker_manager.embeddings(params)

    def sync_embeddings(self, params: Dict) -> List[List[float]]:
        # 同步方法，获取嵌入向量，调用 worker_manager 的 sync_embeddings 方法
        return self.worker_manager.sync_embeddings(params)

    async def count_token(self, params: Dict) -> int:
        # 异步方法，统计 token 数量，调用 worker_manager 的 count_token 方法
        return await self.worker_manager.count_token(params)
    # 异步方法，获取指定参数的模型元数据
    async def get_model_metadata(self, params: Dict) -> ModelMetadata:
        return await self.worker_manager.get_model_metadata(params)

    # 异步方法，将工作请求应用到工作管理器，返回应用结果
    async def worker_apply(self, apply_req: WorkerApplyRequest) -> WorkerApplyOutput:
        return await self.worker_manager.worker_apply(apply_req)

    # 异步方法，获取指定工作类型和模型名称的参数描述列表
    async def parameter_descriptions(
        self, worker_type: str, model_name: str
    ) -> List[ParameterDescription]:
        return await self.worker_manager.parameter_descriptions(worker_type, model_name)
class _DefaultWorkerManagerFactory(WorkerManagerFactory):
    # _DefaultWorkerManagerFactory 类，继承自 WorkerManagerFactory 类
    def __init__(
        self, system_app: SystemApp | None = None, worker_manager: WorkerManager = None
    ):
        # 初始化函数，接受 system_app 和 worker_manager 两个参数
        super().__init__(system_app)
        # 调用父类的初始化函数，传入 system_app 参数
        self.worker_manager = worker_manager
        # 设置实例变量 worker_manager 为传入的 worker_manager 参数

    def create(self) -> WorkerManager:
        # create 方法，返回一个 WorkerManager 对象
        return self.worker_manager


worker_manager = WorkerManagerAdapter()
# 创建一个 WorkerManagerAdapter 对象实例并赋值给 worker_manager 变量
router = APIRouter()
# 创建一个 APIRouter 对象实例并赋值给 router 变量


async def generate_json_stream(params):
    # 异步生成 JSON 流的函数，接受 params 参数
    from starlette.concurrency import iterate_in_threadpool
    # 从 starlette.concurrency 导入 iterate_in_threadpool 函数

    async for output in worker_manager.generate_stream(
        params, async_wrapper=iterate_in_threadpool
    ):
        # 异步循环，调用 worker_manager 的 generate_stream 方法生成流数据
        yield json.dumps(asdict(output), ensure_ascii=False).encode() + b"\0"
        # 使用 json.dumps 将输出转换为 JSON 字符串并编码为字节流，然后追加空字节（\0）并 yield 出来


@router.post("/worker/generate_stream")
async def api_generate_stream(request: PromptRequest):
    # 处理 "/worker/generate_stream" POST 请求的异步函数，接受 PromptRequest 请求对象
    params = request.dict(exclude_none=True)
    # 从请求对象中获取参数字典，排除值为 None 的项
    span_id = root_tracer.get_current_span_id()
    # 获取当前跟踪器的跟踪 ID
    if "span_id" not in params and span_id:
        # 如果参数中没有 span_id，并且存在根跟踪 ID
        params["span_id"] = span_id
        # 将根跟踪 ID 添加到参数字典中
    generator = generate_json_stream(params)
    # 调用 generate_json_stream 函数生成一个 JSON 流生成器
    return StreamingResponse(generator)
    # 返回一个 StreamingResponse 对象，将 JSON 流生成器作为响应体返回


@router.post("/worker/generate")
async def api_generate(request: PromptRequest):
    # 处理 "/worker/generate" POST 请求的异步函数，接受 PromptRequest 请求对象
    params = request.dict(exclude_none=True)
    # 从请求对象中获取参数字典，排除值为 None 的项
    span_id = root_tracer.get_current_span_id()
    # 获取当前跟踪器的跟踪 ID
    if "span_id" not in params and span_id:
        # 如果参数中没有 span_id，并且存在根跟踪 ID
        params["span_id"] = span_id
        # 将根跟踪 ID 添加到参数字典中
    return await worker_manager.generate(params)
    # 调用 worker_manager 的 generate 方法，传入参数字典，并返回其异步执行的结果


@router.post("/worker/embeddings")
async def api_embeddings(request: EmbeddingsRequest):
    # 处理 "/worker/embeddings" POST 请求的异步函数，接受 EmbeddingsRequest 请求对象
    params = request.dict(exclude_none=True)
    # 从请求对象中获取参数字典，排除值为 None 的项
    span_id = root_tracer.get_current_span_id()
    # 获取当前跟踪器的跟踪 ID
    if "span_id" not in params and span_id:
        # 如果参数中没有 span_id，并且存在根跟踪 ID
        params["span_id"] = span_id
        # 将根跟踪 ID 添加到参数字典中
    return await worker_manager.embeddings(params)
    # 调用 worker_manager 的 embeddings 方法，传入参数字典，并返回其异步执行的结果


@router.post("/worker/count_token")
async def api_count_token(request: CountTokenRequest):
    # 处理 "/worker/count_token" POST 请求的异步函数，接受 CountTokenRequest 请求对象
    params = request.dict(exclude_none=True)
    # 从请求对象中获取参数字典，排除值为 None 的项
    span_id = root_tracer.get_current_span_id()
    # 获取当前跟踪器的跟踪 ID
    if "span_id" not in params and span_id:
        # 如果参数中没有 span_id，并且存在根跟踪 ID
        params["span_id"] = span_id
        # 将根跟踪 ID 添加到参数字典中
    return await worker_manager.count_token(params)
    # 调用 worker_manager 的 count_token 方法，传入参数字典，并返回其异步执行的结果


@router.post("/worker/model_metadata")
async def api_get_model_metadata(request: ModelMetadataRequest):
    # 处理 "/worker/model_metadata" POST 请求的异步函数，接受 ModelMetadataRequest 请求对象
    params = request.dict(exclude_none=True)
    # 从请求对象中获取参数字典，排除值为 None 的项
    span_id = root_tracer.get_current_span_id()
    # 获取当前跟踪器的跟踪 ID
    if "span_id" not in params and span_id:
        # 如果参数中没有 span_id，并且存在根跟踪 ID
        params["span_id"] = span_id
        # 将根跟踪 ID 添加到参数字典中
    return await worker_manager.get_model_metadata(params)
    # 调用 worker_manager 的 get_model_metadata 方法，传入参数字典，并返回其异步执行的结果


@router.post("/worker/apply")
async def api_worker_apply(request: WorkerApplyRequest):
    # 处理 "/worker/apply" POST 请求的异步函数，接受 WorkerApplyRequest 请求对象
    return await worker_manager.worker_apply(request)
    # 调用 worker_manager 的 worker_apply 方法，传入请求对象，并返回其异步执行的结果


@router.get("/worker/parameter/descriptions")
async def api_worker_parameter_descs(
    model: str, worker_type: str = WorkerType.LLM.value
):
    # 处理 "/worker/parameter/descriptions" GET 请求的异步函数，接受 model 和 worker_type 两个参数
    return await worker_manager.parameter_descriptions(worker_type, model)
    # 调用 worker_manager 的 parameter_descriptions 方法，传入 worker_type 和 model 参数，并返回其异步执行的结果


@router.get("/worker/models/supports")
async def api_supported_models():
    """Get all supported models.

    This method reads all models from the configuration file and tries to perform some basic checks on the model (like if the path exists).
    """
    # 处理 "/worker/models/supports" GET 请求的异步函数，获取所有支持的模型信息
    return await worker_manager.supported_models()
    # 调用 worker_manager 的 supported_models 方法，并返回其异步执行的结果
    # 如果是一个 RemoteWorkerManager，这个方法返回整个集群支持的模型列表
    """
    # 返回整个集群支持的模型列表
    return await worker_manager.supported_models()
@router.post("/worker/models/startup")
async def api_model_startup(request: WorkerStartupRequest):
    """Start up a specific model."""
    return await worker_manager.model_startup(request)


@router.post("/worker/models/shutdown")
async def api_model_shutdown(request: WorkerStartupRequest):
    """Shut down a specific model."""
    return await worker_manager.model_shutdown(request)


def _setup_fastapi(
    worker_params: ModelWorkerParameters,
    app=None,
    ignore_exception: bool = False,
    system_app: Optional[SystemApp] = None,
):
    # 如果没有传入应用程序实例，则创建一个新的 FastAPI 应用
    if not app:
        app = create_app()
        # 设置 HTTP 服务日志记录
        setup_http_service_logging()

        # 如果传入了系统应用实例，将 FastAPI 应用设置为其 ASGI 应用
        if system_app:
            system_app._asgi_app = app

    # 如果 worker_params.standalone 为 True，表示运行在独立模式下
    if worker_params.standalone:
        from dbgpt.model.cluster.controller.controller import initialize_controller
        from dbgpt.model.cluster.controller.controller import (
            router as controller_router,
        )

        # 如果没有指定控制器地址，则设置默认地址为本地地址和端口
        if not worker_params.controller_addr:
            # 如果在环境变量中设置了 http_proxy 或 https_proxy，可能导致服务器无法启动，这里将其设为空字符串
            os.environ["http_proxy"] = ""
            os.environ["https_proxy"] = ""
            worker_params.controller_addr = f"http://127.0.0.1:{worker_params.port}"
        # 记录日志，显示 WorkerManager 在独立模式下运行，以及控制器地址
        logger.info(
            f"Run WorkerManager with standalone mode, controller_addr: {worker_params.controller_addr}"
        )
        # 初始化控制器，并将其注册到 FastAPI 应用中
        initialize_controller(app=app, system_app=system_app)
        app.include_router(controller_router, prefix="/api")

    async def startup_event():
        async def start_worker_manager():
            try:
                # 尝试启动 WorkerManager
                await worker_manager.start()
            except Exception as e:
                import signal

                # 如果启动失败，记录错误并发送中断信号以终止进程
                logger.error(f"Error starting worker manager: {str(e)}")
                os.kill(os.getpid(), signal.SIGINT)

        # 由于 WorkerManager 的启动依赖于 FastAPI 应用（已注册到控制器），因此此处不能阻塞
        asyncio.create_task(start_worker_manager())

    async def shutdown_event():
        # 在应用关闭时停止 WorkerManager
        await worker_manager.stop(ignore_exception=ignore_exception)

    # 注册启动和关闭事件处理程序
    register_event_handler(app, "startup", startup_event)
    register_event_handler(app, "shutdown", shutdown_event)
    # 返回配置好的 FastAPI 应用实例
    return app


def _parse_worker_params(
    model_name: str = None, model_path: str = None, **kwargs
) -> ModelWorkerParameters:
    # 使用环境参数解析器创建 Worker 参数实例
    worker_args = EnvArgumentParser()
    env_prefix = None
    if model_name:
        # 如果指定了模型名称，则设置环境变量前缀
        env_prefix = EnvArgumentParser.get_env_prefix(model_name)
    # 使用参数解析器将传入的参数解析为 ModelWorkerParameters 数据类实例
    worker_params: ModelWorkerParameters = worker_args.parse_args_into_dataclass(
        ModelWorkerParameters,
        env_prefixes=[env_prefix],
        model_name=model_name,
        model_path=model_path,
        **kwargs,
    )
    # 再次读取带有模型名称前缀的参数
    env_prefix = EnvArgumentParser.get_env_prefix(worker_params.model_name)
    # 返回解析后的 Worker 参数实例
    # 使用 parse_args_into_dataclass 方法解析 worker_args 中的参数，并返回一个新的 ModelWorkerParameters 实例
    new_worker_params = worker_args.parse_args_into_dataclass(
        ModelWorkerParameters,
        env_prefixes=[env_prefix],
        model_name=worker_params.model_name,
        model_path=worker_params.model_path,
        **kwargs,
    )
    # 使用新的参数更新当前的 worker_params 实例
    worker_params.update_from(new_worker_params)
    # 如果存在 model_alias，则将其赋值给 model_name，以便更新模型名称
    if worker_params.model_alias:
        worker_params.model_name = worker_params.model_alias

    # 记录当前的 worker_params 信息到日志中
    # logger.info(f"Worker params: {worker_params}")
    # 返回更新后的 worker_params 实例
    return worker_params
# 创建本地模型管理器函数，根据给定的参数 worker_params 返回 LocalWorkerManager 对象
def _create_local_model_manager(
    worker_params: ModelWorkerParameters,
) -> LocalWorkerManager:
    # 导入获取 IP 地址的函数
    from dbgpt.util.net_utils import _get_ip_address

    # 确定主机地址，如果未提供注册主机地址，则使用本地 IP 地址
    host = (
        worker_params.worker_register_host
        if worker_params.worker_register_host
        else _get_ip_address()
    )
    # 确定端口号
    port = worker_params.port

    # 如果不需要注册或者未提供控制器地址，则记录日志并返回一个不注册的本地工作管理器
    if not worker_params.register or not worker_params.controller_addr:
        logger.info(
            f"Not register current to controller, register: {worker_params.register}, controller_addr: {worker_params.controller_addr}"
        )
        return LocalWorkerManager(host=host, port=port)
    else:
        # 导入模型注册客户端类
        from dbgpt.model.cluster.controller.controller import ModelRegistryClient

        # 创建模型注册客户端对象
        client = ModelRegistryClient(worker_params.controller_addr)

        # 定义异步注册函数，用于向控制器注册当前工作实例
        async def register_func(worker_run_data: WorkerRunData):
            instance = ModelInstance(
                model_name=worker_run_data.worker_key, host=host, port=port
            )
            return await client.register_instance(instance)

        # 定义异步取消注册函数，用于从控制器取消注册当前工作实例
        async def deregister_func(worker_run_data: WorkerRunData):
            instance = ModelInstance(
                model_name=worker_run_data.worker_key, host=host, port=port
            )
            return await client.deregister_instance(instance)

        # 定义异步发送心跳函数，用于向控制器发送当前工作实例的心跳信息
        async def send_heartbeat_func(worker_run_data: WorkerRunData):
            instance = ModelInstance(
                model_name=worker_run_data.worker_key, host=host, port=port
            )
            return await client.send_heartbeat(instance)

        # 返回一个带有注册、取消注册和发送心跳功能的本地工作管理器对象
        return LocalWorkerManager(
            register_func=register_func,
            deregister_func=deregister_func,
            send_heartbeat_func=send_heartbeat_func,
            host=host,
            port=port,
        )


# 构建工作者函数，根据给定的参数 worker_params 和可选的额外工作者参数 ext_worker_kwargs 构建并返回相应的工作者对象
def _build_worker(
    worker_params: ModelWorkerParameters,
    ext_worker_kwargs: Optional[Dict[str, Any]] = None,
):
    # 确定工作者类
    worker_class = worker_params.worker_class

    # 如果提供了工作者类，则从指定模块中导入相应的工作者类
    if worker_class:
        from dbgpt.util.module_utils import import_from_checked_string

        # 使用导入函数从字符串中导入工作者类
        worker_cls = import_from_checked_string(worker_class, ModelWorker)
        logger.info(f"Import worker class from {worker_class} successfully")
    else:
        # 如果未提供工作者类，则根据工作者类型选择默认的工作者类
        if (
            worker_params.worker_type is None
            or worker_params.worker_type == WorkerType.LLM
        ):
            from dbgpt.model.cluster.worker.default_worker import DefaultModelWorker

            worker_cls = DefaultModelWorker
        elif worker_params.worker_type == WorkerType.TEXT2VEC:
            from dbgpt.model.cluster.worker.embedding_worker import (
                EmbeddingsModelWorker,
            )

            worker_cls = EmbeddingsModelWorker
        else:
            # 如果工作者类型不受支持，则抛出异常
            raise Exception("Unsupported worker type: {worker_params.worker_type}")

    # 如果提供了额外的工作者参数，则使用这些参数创建工作者对象
    if ext_worker_kwargs:
        return worker_cls(**ext_worker_kwargs)
    else:
        # 否则，使用默认构造函数创建工作者对象
        return worker_cls()


# 启动本地工作者函数，接受一个 WorkerManagerAdapter 类型的 worker_manager 参数
    # 定义函数参数 worker_params，类型为 ModelWorkerParameters
    worker_params: ModelWorkerParameters,
    # 定义函数参数 ext_worker_kwargs，类型为可选的字典，键为字符串，值为任意类型
    ext_worker_kwargs: Optional[Dict[str, Any]] = None,
    with root_tracer.start_span(
        "WorkerManager._start_local_worker",
        span_type=SpanType.RUN,
        metadata={
            "run_service": SpanTypeRunName.WORKER_MANAGER,
            "params": _get_dict_from_obj(worker_params),
            "sys_infos": _get_dict_from_obj(get_system_info()),
        },
    ):
        # 使用 root_tracer 创建一个名为 "WorkerManager._start_local_worker" 的新跟踪 span，
        # 设置 span 的类型为 SpanType.RUN，并附带一些元数据信息
        worker = _build_worker(worker_params, ext_worker_kwargs=ext_worker_kwargs)
        # 构建一个新的 worker 对象，根据给定的 worker_params 和额外的扩展参数 ext_worker_kwargs
        if not worker_manager.worker_manager:
            # 如果 worker_manager.worker_manager 不存在，则创建一个本地模型管理器
            worker_manager.worker_manager = _create_local_model_manager(worker_params)
        # 向 worker_manager.worker_manager 添加之前构建的 worker 对象和相应的参数
        worker_manager.worker_manager.add_worker(worker, worker_params)


def _start_local_embedding_worker(
    worker_manager: WorkerManagerAdapter,
    embedding_model_name: str = None,
    embedding_model_path: str = None,
    ext_worker_kwargs: Optional[Dict[str, Any]] = None,
):
    # 如果 embedding_model_name 或 embedding_model_path 为空，则直接返回
    if not embedding_model_name or not embedding_model_path:
        return
    # 根据传入的 embedding_model_name 和 embedding_model_path 构建模型工作参数
    embedding_worker_params = ModelWorkerParameters(
        model_name=embedding_model_name,
        model_path=embedding_model_path,
        worker_type=WorkerType.TEXT2VEC,
        worker_class="dbgpt.model.cluster.worker.embedding_worker.EmbeddingsModelWorker",
    )
    # 使用日志记录器记录启动本地嵌入式 worker 的相关参数信息
    logger.info(
        f"Start local embedding worker with embedding parameters\n{embedding_worker_params}"
    )
    # 调用 _start_local_worker 函数启动本地 worker，传入相应的参数
    _start_local_worker(
        worker_manager, embedding_worker_params, ext_worker_kwargs=ext_worker_kwargs
    )


def initialize_worker_manager_in_client(
    app=None,
    include_router: bool = True,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    run_locally: bool = True,
    controller_addr: Optional[str] = None,
    local_port: int = 5670,
    embedding_model_name: Optional[str] = None,
    embedding_model_path: Optional[str] = None,
    rerank_model_name: Optional[str] = None,
    rerank_model_path: Optional[str] = None,
    start_listener: Optional[Callable[["WorkerManager"], None]] = None,
    system_app: Optional[SystemApp] = None,
):
    """Initialize WorkerManager in client.
    If run_locally is True:
    1. Start ModelController
    2. Start LocalWorkerManager
    3. Start worker in LocalWorkerManager
    4. Register worker to ModelController

    otherwise:
    1. Build ModelRegistryClient with controller address
    2. Start RemoteWorkerManager

    """
    global worker_manager

    # 如果 app 为空，则抛出异常
    if not app:
        raise Exception("app can't be None")

    # 如果 system_app 存在，则记录日志信息，注册默认的 WorkerManagerFactory 到 system_app
    if system_app:
        logger.info(f"Register WorkerManager {_DefaultWorkerManagerFactory.name}")
        system_app.register(_DefaultWorkerManagerFactory, worker_manager)

    # 解析传入的 worker 参数，构建 ModelWorkerParameters 对象
    worker_params: ModelWorkerParameters = _parse_worker_params(
        model_name=model_name, model_path=model_path, controller_addr=controller_addr
    )

    controller_addr = None
    if run_locally:
        # 如果在本地运行，则配置为独立模式，并注册到本地端口，启动日志记录
        worker_params.standalone = True
        worker_params.register = True
        worker_params.port = local_port
        logger.info(f"Worker params: {worker_params}")
        
        # 设置 FastAPI 应用，并启动本地工作进程
        _setup_fastapi(worker_params, app, ignore_exception=True, system_app=system_app)
        _start_local_worker(worker_manager, worker_params)
        
        # 在工作进程启动后执行 start_listener
        worker_manager.after_start(start_listener)
        
        # 启动本地嵌入式工作进程，包括嵌入式模型和重新排序模型
        _start_local_embedding_worker(
            worker_manager, embedding_model_name, embedding_model_path
        )
        _start_local_embedding_worker(
            worker_manager,
            rerank_model_name,
            rerank_model_path,
            ext_worker_kwargs={"rerank_model": True},
        )
    else:
        # 如果非本地运行，则从远程导入所需模块
        from dbgpt.model.cluster.controller.controller import (
            ModelRegistryClient,
            initialize_controller,
        )
        from dbgpt.model.cluster.worker.remote_manager import RemoteWorkerManager

        # 如果未指定控制器地址，则引发异常
        if not worker_params.controller_addr:
            raise ValueError("Controller can't be None")
        
        logger.info(f"Worker params: {worker_params}")
        
        # 使用指定的控制器地址创建 ModelRegistryClient 客户端
        client = ModelRegistryClient(worker_params.controller_addr)
        
        # 配置远程工作管理器
        worker_manager.worker_manager = RemoteWorkerManager(client)
        
        # 在工作进程启动后执行 start_listener
        worker_manager.after_start(start_listener)
        
        # 初始化控制器，配置远程控制器地址和系统应用
        initialize_controller(
            app=app,
            remote_controller_addr=worker_params.controller_addr,
            system_app=system_app,
        )
        
        # 获取事件循环并运行工作管理器的启动过程
        loop = asyncio.get_event_loop()
        loop.run_until_complete(worker_manager.start())

    if include_router and app:
        # 如果要包含路由且存在 FastAPI 应用，则挂载 WorkerManager 路由到 '/api' 前缀
        app.include_router(router, prefix="/api")
# 定义一个函数 run_worker_manager，用于管理工作进程的启动和配置
def run_worker_manager(
    app=None,
    include_router: bool = True,
    model_name: str = None,
    model_path: str = None,
    standalone: bool = False,
    port: int = None,
    embedding_model_name: str = None,
    embedding_model_path: str = None,
    start_listener: Callable[["WorkerManager"], None] = None,
    **kwargs,
):
    # 声明全局变量 worker_manager，用于管理工作进程
    global worker_manager

    # 解析工作进程的参数，返回一个 ModelWorkerParameters 对象
    worker_params: ModelWorkerParameters = _parse_worker_params(
        model_name=model_name,
        model_path=model_path,
        standalone=standalone,
        port=port,
        **kwargs,
    )

    # 设置日志记录，包括日志级别和文件名
    setup_logging(
        "dbgpt",
        logging_level=worker_params.log_level,
        logger_filename=worker_params.log_file,
    )

    # 初始化一个系统应用实例
    system_app = SystemApp()

    # 如果未提供外部的 FastAPI 应用，则配置一个 FastAPI 应用
    if not app:
        embedded_mod = False
        app = _setup_fastapi(worker_params, system_app=system_app)
    system_app._asgi_app = app

    # 初始化分布式追踪器，配置追踪参数和存储方式
    initialize_tracer(
        os.path.join(LOGDIR, worker_params.tracer_file),
        system_app=system_app,
        root_operation_name="DB-GPT-ModelWorker",
        tracer_storage_cls=worker_params.tracer_storage_cls,
        enable_open_telemetry=worker_params.tracer_to_open_telemetry,
        otlp_endpoint=worker_params.otel_exporter_otlp_traces_endpoint,
        otlp_insecure=worker_params.otel_exporter_otlp_traces_insecure,
        otlp_timeout=worker_params.otel_exporter_otlp_traces_timeout,
    )

    # 启动本地工作进程
    _start_local_worker(worker_manager, worker_params)

    # 启动本地嵌入式模型工作进程
    _start_local_embedding_worker(
        worker_manager, embedding_model_name, embedding_model_path
    )

    # 在启动后执行指定的监听器函数
    worker_manager.after_start(start_listener)

    # 如果需要包含路由器，则将路由器包含到 FastAPI 应用中
    if include_router:
        app.include_router(router, prefix="/api")

    # 如果不是嵌入式模式，通过 uvicorn 启动 FastAPI 应用
    if not embedded_mod:
        import uvicorn

        uvicorn.run(
            app, host=worker_params.host, port=worker_params.port, log_level="info"
        )
    else:
        # 如果是嵌入式模式，则在事件循环中运行 worker_manager 的启动
        loop = asyncio.get_event_loop()
        loop.run_until_complete(worker_manager.start())


if __name__ == "__main__":
    # 当作为脚本直接执行时，调用 run_worker_manager 函数
    run_worker_manager()
```