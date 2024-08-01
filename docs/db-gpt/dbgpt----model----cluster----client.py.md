# `.\DB-GPT-src\dbgpt\model\cluster\client.py`

```py
import asyncio  # 引入异步编程库 asyncio
from typing import AsyncIterator, List, Optional  # 引入类型提示模块，用于定义函数参数和返回类型

from dbgpt.core.awel import DAGVar  # 从 dbgpt.core.awel 模块导入 DAGVar 类
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource  # 从 dbgpt.core.awel.flow 模块导入 Parameter, ResourceCategory, register_resource
from dbgpt.core.interface.llm import (  # 从 dbgpt.core.interface.llm 模块导入以下类和函数
    DefaultMessageConverter,  # 默认消息转换器类
    LLMClient,  # LLM客户端类
    MessageConverter,  # 消息转换器类
    ModelMetadata,  # 模型元数据类
    ModelOutput,  # 模型输出类
    ModelRequest,  # 模型请求类
)
from dbgpt.model.cluster.manager_base import WorkerManager  # 从 dbgpt.model.cluster.manager_base 导入 WorkerManager 类
from dbgpt.model.parameter import WorkerType  # 从 dbgpt.model.parameter 导入 WorkerType 类
from dbgpt.util.i18n_utils import _  # 导入国际化工具函数 _

# 使用 register_resource 装饰器注册一个资源
@register_resource(
    label=_("Default LLM Client"),  # 资源标签，国际化字符串
    name="default_llm_client",  # 资源名称
    category=ResourceCategory.LLM_CLIENT,  # 资源类别为LLM客户端
    description=_("Default LLM client(Connect to your DB-GPT model serving)"),  # 资源描述，国际化字符串
    parameters=[  # 资源参数列表
        Parameter.build_from(
            _("Auto Convert Message"),  # 参数标签，国际化字符串
            name="auto_convert_message",  # 参数名称
            type=bool,  # 参数类型为布尔型
            optional=True,  # 参数可选
            default=False,  # 参数默认值为False
            description=_(
                "Whether to auto convert the messages that are not supported "
                "by the LLM to a compatible format"
            ),  # 参数描述，国际化字符串
        )
    ],
)
class DefaultLLMClient(LLMClient):
    """Default LLM client implementation.

    Connect to the worker manager and send the request to the worker manager.

    Args:
        worker_manager (WorkerManager): worker manager instance.
        auto_convert_message (bool, optional): auto convert the message to ModelRequest. Defaults to False.
    """

    def __init__(
        self,
        worker_manager: Optional[WorkerManager] = None,
        auto_convert_message: bool = False,
    ):
        self._worker_manager = worker_manager  # 初始化工作管理器实例变量
        self._auto_covert_message = auto_convert_message  # 初始化自动转换消息变量

    @property
    def worker_manager(self) -> WorkerManager:
        """Get the worker manager instance.
        If not set, get the worker manager from the system app. If not set, raise
        ValueError.
        """
        if not self._worker_manager:
            system_app = DAGVar.get_current_system_app()  # 获取当前系统应用
            if not system_app:
                raise ValueError("System app is not initialized")  # 若系统应用未初始化，则抛出异常
            from dbgpt.model.cluster import WorkerManagerFactory  # 动态导入 WorkerManagerFactory

            return WorkerManagerFactory.get_instance(system_app).create()  # 使用工厂创建 WorkerManager 实例
        return self._worker_manager  # 返回现有的工作管理器实例

    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        if not message_converter and self._auto_covert_message:
            message_converter = DefaultMessageConverter()  # 若未提供消息转换器且启用自动转换消息，则使用默认消息转换器
        request = await self.covert_message(request, message_converter)  # 转换消息格式
        return await self.worker_manager.generate(request.to_dict())  # 生成并返回模型输出

    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    async def models(self) -> List[ModelMetadata]:
        # 获取所有健康状态的LLM模型实例
        instances = await self.worker_manager.get_all_model_instances(
            WorkerType.LLM.value, healthy_only=True
        )
        # 用于存放异步任务的列表
        query_metadata_task = []
        # 遍历每个实例，解析出worker_name，并添加获取模型元数据的异步任务
        for instance in instances:
            worker_name, _ = WorkerType.parse_worker_key(instance.worker_key)
            query_metadata_task.append(
                self.worker_manager.get_model_metadata({"model": worker_name})
            )
        # 执行所有获取模型元数据的异步任务，并等待结果
        models: List[ModelMetadata] = await asyncio.gather(*query_metadata_task)
        # 构建模型名称到模型元数据的映射字典
        model_map = {}
        for single_model in models:
            model_map[single_model.model] = single_model
        # 返回按模型名称排序的模型元数据列表
        return [model_map[model_name] for model_name in sorted(model_map.keys())]

    async def count_token(self, model: str, prompt: str) -> int:
        # 调用worker_manager的方法统计特定模型和提示语的token数量
        return await self.worker_manager.count_token({"model": model, "prompt": prompt})
@register_resource(
    label=_("Remote LLM Client"),  # 注册资源，标签为“Remote LLM Client”
    name="remote_llm_client",  # 资源名称为“remote_llm_client”
    category=ResourceCategory.LLM_CLIENT,  # 资源类别为LLM客户端
    description=_("Remote LLM client(Connect to the remote DB-GPT model serving)"),  # 描述为远程LLM客户端，连接到远程的DB-GPT模型服务
    parameters=[  # 参数列表开始
        Parameter.build_from(
            _("Controller Address"),  # 参数名称为“Controller Address”
            name="controller_address",  # 参数变量名为controller_address
            type=str,  # 参数类型为字符串
            optional=True,  # 可选参数
            default=_("http://127.0.0.1:8000"),  # 默认值为http://127.0.0.1:8000
            description=_("Model controller address"),  # 参数描述为模型控制器地址
        ),
        Parameter.build_from(
            _("Auto Convert Message"),  # 参数名称为“Auto Convert Message”
            name="auto_convert_message",  # 参数变量名为auto_convert_message
            type=bool,  # 参数类型为布尔值
            optional=True,  # 可选参数
            default=False,  # 默认值为False
            description=_(
                "Whether to auto convert the messages that are not supported "
                "by the LLM to a compatible format"
            ),  # 参数描述为是否自动转换LLM不支持的消息为兼容格式
        ),
    ],  # 参数列表结束
)
class RemoteLLMClient(DefaultLLMClient):
    """Remote LLM client implementation.

    Connect to the remote worker manager and send the request to the remote worker manager.

    Args:
        controller_address (str): model controller address
        auto_convert_message (bool, optional): auto convert the message to
            ModelRequest. Defaults to False.

    If you start DB-GPT model cluster, the controller address is the address of the
    Model Controller(`dbgpt start controller`, the default port of model controller
    is 8000).
    Otherwise, if you already have a running DB-GPT server(start it by
    `dbgpt start webserver --port ${remote_port}`), you can use the address of the
    `http://${remote_ip}:${remote_port}`.

    """

    def __init__(
        self,
        controller_address: str = "http://127.0.0.1:8000",  # 控制器地址，默认为http://127.0.0.1:8000
        auto_convert_message: bool = False,  # 是否自动转换消息，默认为False
    ):
        """Initialize the RemoteLLMClient."""
        from dbgpt.model.cluster import ModelRegistryClient, RemoteWorkerManager

        # 创建模型注册客户端对象
        model_registry_client = ModelRegistryClient(controller_address)
        
        # 创建远程工作管理器对象，使用模型注册客户端对象
        worker_manager = RemoteWorkerManager(model_registry_client)
        
        # 调用父类的初始化方法，传入远程工作管理器和自动消息转换参数
        super().__init__(worker_manager, auto_convert_message)
```