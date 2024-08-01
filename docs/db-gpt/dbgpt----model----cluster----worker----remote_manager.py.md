# `.\DB-GPT-src\dbgpt\model\cluster\worker\remote_manager.py`

```py
import asyncio  # 导入异步IO模块
from typing import Any, Callable  # 导入类型提示模块

from dbgpt.model.base import ModelInstance, WorkerApplyOutput, WorkerSupportedModel  # 导入模型基类
from dbgpt.model.cluster.base import *  # 导入集群基础模块中的所有内容
from dbgpt.model.cluster.registry import ModelRegistry  # 导入模型注册表
from dbgpt.model.cluster.worker.manager import LocalWorkerManager, WorkerRunData, logger  # 导入本地工作管理器和相关类型
from dbgpt.model.cluster.worker.remote_worker import RemoteModelWorker  # 导入远程模型工作器


class RemoteWorkerManager(LocalWorkerManager):
    def __init__(self, model_registry: ModelRegistry = None) -> None:
        super().__init__(model_registry=model_registry)  # 调用父类构造函数初始化

    async def start(self):
        for listener in self.start_listeners:
            if asyncio.iscoroutinefunction(listener):  # 检查监听器是否为协程函数
                await listener(self)  # 如果是协程函数，等待其执行
            else:
                listener(self)  # 如果不是协程函数，直接调用

    async def stop(self, ignore_exception: bool = False):
        pass  # 空的停止方法，暂时不做任何操作

    async def _fetch_from_worker(
        self,
        worker_run_data: WorkerRunData,
        endpoint: str,
        method: str = "GET",
        json: dict = None,
        params: dict = None,
        additional_headers: dict = None,
        success_handler: Callable = None,
        error_handler: Callable = None,
    ) -> Any:
        # Lazy import to avoid high time cost
        import httpx  # 惰性导入httpx模块，避免高时间成本

        url = worker_run_data.worker.worker_addr + endpoint  # 构建完整的请求地址
        headers = {**worker_run_data.worker.headers, **(additional_headers or {})}  # 合并请求头
        timeout = worker_run_data.worker.timeout  # 获取超时时间设定

        async with httpx.AsyncClient() as client:
            request = client.build_request(
                method,
                url,
                json=json,  # 使用json作为数据以确保以application/json格式发送
                params=params,
                headers=headers,
                timeout=timeout,
            )

            response = await client.send(request)  # 发送HTTP请求并等待响应

            if response.status_code != 200:  # 检查响应状态码是否不为200
                if error_handler:
                    return error_handler(response)  # 如果定义了错误处理函数，则调用它处理响应
                else:
                    error_msg = f"Request to {url} failed, error: {response.text}"  # 构建错误消息
                    raise Exception(error_msg)  # 抛出异常

            if success_handler:
                return success_handler(response)  # 如果定义了成功处理函数，则调用它处理响应

            return response.json()  # 返回JSON格式的响应数据

    async def _apply_to_worker_manager_instances(self):
        pass  # 空的应用到工作管理器实例的方法，暂时不做任何操作
    # 异步方法，返回支持的模型列表
    async def supported_models(self) -> List[WorkerSupportedModel]:
        # 获取所有指定类型和名称的工作实例
        worker_instances = await self.get_model_instances(
            WORKER_MANAGER_SERVICE_TYPE, WORKER_MANAGER_SERVICE_NAME
        )

        # 定义内部异步函数，获取每个工作实例支持的模型列表
        async def get_supported_models(worker_run_data) -> List[WorkerSupportedModel]:
            # 处理响应的回调函数，将 JSON 转换为 WorkerSupportedModel 对象列表
            def handler(response):
                return list(WorkerSupportedModel.from_dict(m) for m in response.json())

            # 调用内部方法，从工作实例中获取支持的模型列表
            return await self._fetch_from_worker(
                worker_run_data, "/models/supports", success_handler=handler
            )

        models = []
        # 使用 asyncio.gather 并发获取所有工作实例支持的模型列表
        results = await asyncio.gather(
            *(get_supported_models(worker) for worker in worker_instances)
        )
        # 合并所有结果为一个模型列表
        for res in results:
            models += res
        return models

    # 异步方法，获取符合指定主机和端口的工作实例列表
    async def _get_worker_service_instance(
        self, host: str = None, port: int = None
    ) -> List[WorkerRunData]:
        # 获取所有指定类型和名称的工作实例
        worker_instances = await self.get_model_instances(
            WORKER_MANAGER_SERVICE_TYPE, WORKER_MANAGER_SERVICE_NAME
        )
        # 默认错误信息
        error_msg = f"Cound not found worker instances"
        
        # 如果指定了主机和端口，则筛选符合条件的工作实例
        if host and port:
            worker_instances = [
                ins for ins in worker_instances if ins.host == host and ins.port == port
            ]
            error_msg = f"Cound not found worker instances for host {host} port {port}"
        
        # 如果没有找到符合条件的工作实例，抛出异常
        if not worker_instances:
            raise Exception(error_msg)
        
        # 返回符合条件的工作实例列表
        return worker_instances

    # 异步方法，启动指定工作实例的模型
    async def model_startup(self, startup_req: WorkerStartupRequest):
        # 获取符合指定主机和端口的工作实例列表
        worker_instances = await self._get_worker_service_instance(
            startup_req.host, startup_req.port
        )
        # 选择第一个工作实例进行模型启动
        worker_run_data = worker_instances[0]
        # 记录启动信息到日志
        logger.info(f"Start model remote, startup_req: {startup_req}")
        # 调用内部方法，向工作实例发送模型启动请求
        return await self._fetch_from_worker(
            worker_run_data,
            "/models/startup",
            method="POST",
            json=startup_req.dict(),
            success_handler=lambda x: None,
        )

    # 异步方法，关闭指定工作实例的模型
    async def model_shutdown(self, shutdown_req: WorkerStartupRequest):
        # 获取符合指定主机和端口的工作实例列表
        worker_instances = await self._get_worker_service_instance(
            shutdown_req.host, shutdown_req.port
        )
        # 选择第一个工作实例进行模型关闭
        worker_run_data = worker_instances[0]
        # 记录关闭信息到日志
        logger.info(f"Shutdown model remote, shutdown_req: {shutdown_req}")
        # 调用内部方法，向工作实例发送模型关闭请求
        return await self._fetch_from_worker(
            worker_run_data,
            "/models/shutdown",
            method="POST",
            json=shutdown_req.dict(),
            success_handler=lambda x: None,
        )

    # 构建工作实例列表，用于给定模型名称和实例列表
    def _build_worker_instances(
        self, model_name: str, instances: List[ModelInstance]
    ) -> List[WorkerRunData]:
        worker_instances = []
        # 遍历实例列表，构建单个工作实例，并加入到结果列表中
        for instance in instances:
            worker_instances.append(
                self._build_single_worker_instance(model_name, instance)
            )
        # 返回构建好的工作实例列表
        return worker_instances
    # 创建单个远程模型工作实例的方法，根据给定的模型名称和实例信息
    def _build_single_worker_instance(self, model_name: str, instance: ModelInstance):
        # 初始化一个远程模型工作实例
        worker = RemoteModelWorker()
        # 加载工作实例，指定模型名称、主机和端口
        worker.load_worker(
            model_name, model_name, host=instance.host, port=instance.port
        )
        # 创建 WorkerRunData 对象，包含工作实例的各种信息和参数
        wr = WorkerRunData(
            host=instance.host,
            port=instance.port,
            worker_key=instance.model_name,
            worker=worker,
            worker_params=None,
            model_params=None,
            stop_event=asyncio.Event(),  # 异步事件，用于控制工作实例的停止
            semaphore=asyncio.Semaphore(100),  # 异步信号量，用于控制并发请求的数量
        )
        return wr  # 返回创建的 WorkerRunData 对象

    # 异步方法，获取指定类型和模型名称的所有工作实例
    async def get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 根据工作类型和模型名称获取所有模型实例
        worker_key = self._worker_key(worker_type, model_name)
        instances: List[ModelInstance] = await self.model_registry.get_all_instances(
            worker_key, healthy_only
        )
        # 调用内部方法，构建并返回工作实例列表
        return self._build_worker_instances(model_name, instances)

    # 异步方法，获取所有指定类型的模型实例
    async def get_all_model_instances(
        self, worker_type: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 获取所有指定类型的模型实例列表
        instances: List[
            ModelInstance
        ] = await self.model_registry.get_all_model_instances(healthy_only=healthy_only)
        result = []
        # 遍历每个模型实例，如果匹配指定的工作类型，则创建工作实例并加入结果列表
        for instance in instances:
            name, wt = WorkerType.parse_worker_key(instance.model_name)
            if wt != worker_type:
                continue
            result.append(self._build_single_worker_instance(name, instance))
        return result  # 返回所有匹配的工作实例列表

    # 同步方法，获取指定类型和模型名称的所有工作实例
    def sync_get_model_instances(
        self, worker_type: str, model_name: str, healthy_only: bool = True
    ) -> List[WorkerRunData]:
        # 根据工作类型和模型名称获取所有模型实例（同步方式）
        worker_key = self._worker_key(worker_type, model_name)
        instances: List[ModelInstance] = self.model_registry.sync_get_all_instances(
            worker_key, healthy_only
        )
        # 调用内部方法，构建并返回工作实例列表
        return self._build_worker_instances(model_name, instances)

    # 异步方法，将请求应用到工作实例上，并返回应用结果
    async def worker_apply(self, apply_req: WorkerApplyRequest) -> WorkerApplyOutput:
        async def _remote_apply_func(worker_run_data: WorkerRunData):
            # 内部方法，向工作实例发送应用请求，并处理返回结果
            return await self._fetch_from_worker(
                worker_run_data,
                "/apply",
                method="POST",
                json=apply_req.dict(),
                success_handler=lambda res: WorkerApplyOutput(**res.json()),
                error_handler=lambda res: WorkerApplyOutput(
                    message=res.text, success=False
                ),
            )

        # 调用内部方法，将请求应用到所有工作实例，并获取结果列表
        results = await self._apply_worker(apply_req, _remote_apply_func)
        if results:
            return results[0]  # 返回第一个应用结果
```