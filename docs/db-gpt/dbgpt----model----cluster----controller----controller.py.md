# `.\DB-GPT-src\dbgpt\model\cluster\controller\controller.py`

```py
# 导入日志模块
import logging
# 导入操作系统相关功能模块
import os
# 导入抽象基类相关模块
from abc import ABC, abstractmethod
# 导入类型提示相关模块
from typing import List, Literal, Optional

# 导入FastAPI中的API路由器
from fastapi import APIRouter

# 导入自定义组件相关模块
from dbgpt.component import BaseComponent, ComponentType, SystemApp
# 导入模型实例相关模块
from dbgpt.model.base import ModelInstance
# 导入嵌入模型注册表和模型注册表相关模块
from dbgpt.model.cluster.registry import EmbeddedModelRegistry, ModelRegistry
# 导入模型控制器参数相关模块
from dbgpt.model.parameter import ModelControllerParameters
# 导入API工具相关模块
from dbgpt.util.api_utils import APIMixin
# 导入远程API相关模块
from dbgpt.util.api_utils import _api_remote as api_remote
# 导入同步远程API相关模块
from dbgpt.util.api_utils import _sync_api_remote as sync_api_remote
# 导入FastAPI应用创建相关模块
from dbgpt.util.fastapi import create_app
# 导入环境参数解析器相关模块
from dbgpt.util.parameter_utils import EnvArgumentParser
# 导入追踪器初始化和根追踪器相关模块
from dbgpt.util.tracer.tracer_impl import initialize_tracer, root_tracer
# 导入HTTP服务日志设置和日志设置相关模块
from dbgpt.util.utils import setup_http_service_logging, setup_logging

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 定义基础模型控制器类，继承自BaseComponent和抽象基类ABC
class BaseModelController(BaseComponent, ABC):
    # 模型控制器的名称
    name = ComponentType.MODEL_CONTROLLER

    # 初始化应用程序，接收一个系统应用对象作为参数
    def init_app(self, system_app: SystemApp):
        pass

    # 注册给定模型实例的抽象方法，异步返回布尔值
    @abstractmethod
    async def register_instance(self, instance: ModelInstance) -> bool:
        """Register a given model instance"""

    # 注销给定模型实例的抽象方法，异步返回布尔值
    @abstractmethod
    async def deregister_instance(self, instance: ModelInstance) -> bool:
        """Deregister a given model instance."""

    # 获取所有给定模型的实例的抽象方法，异步返回模型实例列表
    @abstractmethod
    async def get_all_instances(
        self, model_name: str = None, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """Fetch all instances of a given model. Optionally, fetch only the healthy instances."""

    # 发送给定模型实例心跳的抽象方法，异步返回布尔值
    @abstractmethod
    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        """Send a heartbeat for a given model instance. This can be used to verify if the instance is still alive and functioning."""

    # 模型应用方法，抛出未实现错误
    async def model_apply(self) -> bool:
        raise NotImplementedError


# 定义本地模型控制器类，继承自BaseModelController
class LocalModelController(BaseModelController):
    # 初始化方法，接收模型注册表对象作为参数
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.deployment = None

    # 注册模型实例的异步方法，返回注册结果布尔值
    async def register_instance(self, instance: ModelInstance) -> bool:
        return await self.registry.register_instance(instance)

    # 注销模型实例的异步方法，返回注销结果布尔值
    async def deregister_instance(self, instance: ModelInstance) -> bool:
        return await self.registry.deregister_instance(instance)

    # 获取所有模型实例的异步方法，根据模型名称和是否仅健康状态进行筛选，返回模型实例列表
    async def get_all_instances(
        self, model_name: str = None, healthy_only: bool = False
    ) -> List[ModelInstance]:
        # 记录信息日志，包含模型名称和是否仅健康状态
        logger.info(
            f"Get all instances with {model_name}, healthy_only: {healthy_only}"
        )
        # 如果没有指定模型名称，则获取所有模型实例（可能仅健康状态）
        if not model_name:
            return await self.registry.get_all_model_instances(
                healthy_only=healthy_only
            )
        # 否则，根据模型名称获取所有实例（可能仅健康状态）
        else:
            return await self.registry.get_all_instances(model_name, healthy_only)

    # 发送模型实例心跳的异步方法，返回发送结果布尔值
    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        return await self.registry.send_heartbeat(instance)


# 定义远程模型控制器类，继承自APIMixin和BaseModelController
class _RemoteModelController(APIMixin, BaseModelController):
    # 略
    # 初始化方法，接受多个参数来配置实例，包括远程服务的 URL 列表、健康检查间隔、超时时间、健康检查开关和选择类型
    def __init__(
        self,
        urls: str,
        health_check_interval_secs: int = 5,
        health_check_timeout_secs: int = 30,
        check_health: bool = True,
        choice_type: Literal["latest_first", "random"] = "latest_first",
    ) -> None:
        # 调用父类 APIMixin 的初始化方法，配置远程服务 URL、健康检查路径、间隔和超时时间、健康检查开关、选择类型
        APIMixin.__init__(
            self,
            urls=urls,
            health_check_path="/api/health",
            health_check_interval_secs=health_check_interval_secs,
            health_check_timeout_secs=health_check_timeout_secs,
            check_health=check_health,
            choice_type=choice_type,
        )
        # 调用 BaseModelController 的初始化方法
        BaseModelController.__init__(self)

    # 注册模型实例的异步方法，向远程服务的 /api/controller/models 发送 POST 请求
    @api_remote(path="/api/controller/models", method="POST")
    async def register_instance(self, instance: ModelInstance) -> bool:
        pass

    # 注销模型实例的异步方法，向远程服务的 /api/controller/models 发送 DELETE 请求
    @api_remote(path="/api/controller/models", method="DELETE")
    async def deregister_instance(self, instance: ModelInstance) -> bool:
        pass

    # 获取所有模型实例的异步方法，向远程服务的 /api/controller/models 发送 GET 请求
    @api_remote(path="/api/controller/models")
    async def get_all_instances(
        self, model_name: str = None, healthy_only: bool = False
    ) -> List[ModelInstance]:
        pass

    # 发送模型实例心跳的异步方法，向远程服务的 /api/controller/heartbeat 发送 POST 请求
    @api_remote(path="/api/controller/heartbeat", method="POST")
    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        pass
class ModelRegistryClient(_RemoteModelController, ModelRegistry):
    async def get_all_model_instances(
        self, healthy_only: bool = False
    ) -> List[ModelInstance]:
        # 调用基类方法获取所有模型实例，支持健康检查筛选
        return await self.get_all_instances(healthy_only=healthy_only)

    @sync_api_remote(path="/api/controller/models")
    def sync_get_all_instances(
        self, model_name: str = None, healthy_only: bool = False
    ) -> List[ModelInstance]:
        # 同步 API 方法，通过远程调用获取所有模型实例
        pass


class ModelControllerAdapter(BaseModelController):
    def __init__(self, backend: BaseModelController = None) -> None:
        self.backend = backend

    async def register_instance(self, instance: ModelInstance) -> bool:
        # 注册模型实例到后端控制器
        return await self.backend.register_instance(instance)

    async def deregister_instance(self, instance: ModelInstance) -> bool:
        # 从后端控制器注销模型实例
        return await self.backend.deregister_instance(instance)

    async def get_all_instances(
        self, model_name: str = None, healthy_only: bool = False
    ) -> List[ModelInstance]:
        # 获取所有模型实例，支持根据模型名称和健康状态筛选
        return await self.backend.get_all_instances(model_name, healthy_only)

    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        # 发送模型实例的心跳信息到后端控制器
        return await self.backend.send_heartbeat(instance)

    async def model_apply(self) -> bool:
        # 向后端控制器应用模型变更
        return await self.backend.model_apply()


router = APIRouter()

controller = ModelControllerAdapter()


def initialize_controller(
    app=None,
    remote_controller_addr: str = None,
    host: str = None,
    port: int = None,
    registry: Optional[ModelRegistry] = None,
    controller_params: Optional[ModelControllerParameters] = None,
    system_app: Optional[SystemApp] = None,
):
    global controller
    if remote_controller_addr:
        # 如果提供了远程控制器地址，则使用远程控制器
        controller.backend = _RemoteModelController(remote_controller_addr)
    else:
        if not registry:
            # 如果未提供注册表，则使用内嵌的模型注册表
            registry = EmbeddedModelRegistry()
        # 否则使用本地模型控制器，传入注册表
        controller.backend = LocalModelController(registry=registry)

    if app:
        # 如果提供了应用程序实例，则将路由器包含到应用程序中
        app.include_router(router, prefix="/api", tags=["Model"])
    # 否则，导入 uvicorn 库，用于异步运行 ASGI 应用程序
    import uvicorn

    # 从模型配置中导入日志目录路径
    from dbgpt.configs.model_config import LOGDIR

    # 设置 HTTP 服务的日志记录配置
    setup_http_service_logging()

    # 创建 ASGI 应用程序实例
    app = create_app()

    # 如果系统应用程序对象未提供，则创建默认的系统应用程序对象
    if not system_app:
        system_app = SystemApp(app)

    # 如果控制器参数未提供，则抛出 ValueError 异常
    if not controller_params:
        raise ValueError("Controller parameters are required.")

    # 初始化分布式跟踪器，配置跟踪器参数
    initialize_tracer(
        os.path.join(LOGDIR, controller_params.tracer_file),
        root_operation_name="DB-GPT-ModelController",
        system_app=system_app,
        tracer_storage_cls=controller_params.tracer_storage_cls,
        enable_open_telemetry=controller_params.tracer_to_open_telemetry,
        otlp_endpoint=controller_params.otel_exporter_otlp_traces_endpoint,
        otlp_insecure=controller_params.otel_exporter_otlp_traces_insecure,
        otlp_timeout=controller_params.otel_exporter_otlp_traces_timeout,
    )

    # 将路由器注册到应用程序，指定 API 路径前缀和标签为 "Model"
    app.include_router(router, prefix="/api", tags=["Model"])

    # 使用 uvicorn 运行 ASGI 应用程序，指定主机、端口和日志级别
    uvicorn.run(app, host=host, port=port, log_level="info")
@router.get("/health")
async def api_health_check():
    """Health check API."""
    # 返回一个简单的状态消息，表明服务状态正常
    return {"status": "ok"}


@router.post("/controller/models")
async def api_register_instance(request: ModelInstance):
    # 使用根跟踪器创建一个新的跟踪 span，用于注册模型实例，并记录请求的元数据
    with root_tracer.start_span(
        "dbgpt.model.controller.register_instance", metadata=request.to_dict()
    ):
        # 调用控制器的注册实例方法，并返回结果
        return await controller.register_instance(request)


@router.delete("/controller/models")
async def api_deregister_instance(model_name: str, host: str, port: int):
    # 创建一个模型实例对象，用于注销操作
    instance = ModelInstance(model_name=model_name, host=host, port=port)
    # 使用根跟踪器创建一个新的跟踪 span，用于注销模型实例，并记录实例的元数据
    with root_tracer.start_span(
        "dbgpt.model.controller.deregister_instance", metadata=instance.to_dict()
    ):
        # 调用控制器的注销实例方法，并返回结果
        return await controller.deregister_instance(instance)


@router.get("/controller/models")
async def api_get_all_instances(model_name: str = None, healthy_only: bool = False):
    # 获取所有模型实例的信息，可选地筛选健康实例
    return await controller.get_all_instances(model_name, healthy_only=healthy_only)


@router.post("/controller/heartbeat")
async def api_model_heartbeat(request: ModelInstance):
    # 发送模型实例的心跳信息，以确保其健康状态
    return await controller.send_heartbeat(request)


def _create_registry(controller_params: ModelControllerParameters) -> ModelRegistry:
    """Create a model registry based on the controller parameters.

    Registry will store the metadata of all model instances, it will be a high
    availability service for model instances if you use a database registry now. Also,
    we can implement more registry types in the future.
    """
    # 根据控制器参数创建一个模型注册表，用于存储所有模型实例的元数据
    registry_type = controller_params.registry_type.strip()
    if controller_params.registry_type == "embedded":
        # 如果注册类型为嵌入式，返回一个嵌入式模型注册表对象
        return EmbeddedModelRegistry(
            heartbeat_interval_secs=controller_params.heartbeat_interval_secs,
            heartbeat_timeout_secs=controller_params.heartbeat_timeout_secs,
        )
    # 如果注册表类型是数据库，则执行以下操作
    elif controller_params.registry_type == "database":
        # 导入必要的模块
        from urllib.parse import quote
        from urllib.parse import quote_plus as urlquote

        # 导入存储模型注册表实现
        from dbgpt.model.cluster.registry_impl.storage import StorageModelRegistry

        # 尝试创建数据库的标志
        try_to_create_db = False

        # 如果注册表数据库类型是 MySQL
        if controller_params.registry_db_type == "mysql":
            # 获取数据库连接参数
            db_name = controller_params.registry_db_name
            db_host = controller_params.registry_db_host
            db_port = controller_params.registry_db_port
            db_user = controller_params.registry_db_user
            db_password = controller_params.registry_db_password

            # 检查是否所有参数都已提供，否则引发异常
            if not db_name:
                raise ValueError(
                    "Registry DB name is required when using MySQL registry."
                )
            if not db_host:
                raise ValueError(
                    "Registry DB host is required when using MySQL registry."
                )
            if not db_port:
                raise ValueError(
                    "Registry DB port is required when using MySQL registry."
                )
            if not db_user:
                raise ValueError(
                    "Registry DB user is required when using MySQL registry."
                )
            if not db_password:
                raise ValueError(
                    "Registry DB password is required when using MySQL registry."
                )

            # 构建 MySQL 数据库的连接 URL
            db_url = (
                f"mysql+pymysql://{quote(db_user)}:"
                f"{urlquote(db_password)}@"
                f"{db_host}:"
                f"{str(db_port)}/"
                f"{db_name}?charset=utf8mb4"
            )

        # 如果注册表数据库类型是 SQLite
        elif controller_params.registry_db_type == "sqlite":
            # 获取 SQLite 数据库文件名
            db_name = controller_params.registry_db_name

            # 检查是否提供了数据库文件名，否则引发异常
            if not db_name:
                raise ValueError(
                    "Registry DB name is required when using SQLite registry."
                )

            # 构建 SQLite 数据库的连接 URL
            db_url = f"sqlite:///{db_name}"

            # 设置标志以尝试创建数据库
            try_to_create_db = True

        # 如果注册表数据库类型未知或不支持，则引发异常
        else:
            raise ValueError(
                f"Unsupported registry DB type: {controller_params.registry_db_type}"
            )

        # 使用数据库连接 URL 创建存储模型注册表对象
        registry = StorageModelRegistry.from_url(
            db_url,
            db_name,
            pool_size=controller_params.registry_db_pool_size,
            max_overflow=controller_params.registry_db_max_overflow,
            try_to_create_db=try_to_create_db,
            heartbeat_interval_secs=controller_params.heartbeat_interval_secs,
            heartbeat_timeout_secs=controller_params.heartbeat_timeout_secs,
        )

        # 返回创建的注册表对象
        return registry

    # 如果注册表类型不是数据库，则引发异常
    else:
        raise ValueError(f"Unsupported registry type: {registry_type}")
# 定义一个函数，用于运行模型控制器的主逻辑
def run_model_controller():
    # 创建一个环境参数解析器实例
    parser = EnvArgumentParser()
    # 设置环境变量的前缀字符串
    env_prefix = "controller_"
    # 使用参数解析器解析环境变量，并将解析结果存储到 ModelControllerParameters 数据类中
    controller_params: ModelControllerParameters = parser.parse_args_into_dataclass(
        ModelControllerParameters,
        env_prefixes=[env_prefix],
    )

    # 设置日志记录的配置，包括日志名称、日志级别和日志文件名
    setup_logging(
        "dbgpt",
        logging_level=controller_params.log_level,
        logger_filename=controller_params.log_file,
    )
    
    # 创建一个注册表，用于存储模型控制器相关的信息
    registry = _create_registry(controller_params)

    # 初始化模型控制器，指定主机、端口、注册表和控制器参数
    initialize_controller(
        host=controller_params.host,
        port=controller_params.port,
        registry=registry,
        controller_params=controller_params,
    )


# 如果当前脚本作为主程序运行，则调用 run_model_controller 函数
if __name__ == "__main__":
    run_model_controller()
```