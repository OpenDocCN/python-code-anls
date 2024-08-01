# `.\DB-GPT-src\dbgpt\model\cluster\registry_impl\storage.py`

```py
# 导入必要的模块：线程管理、时间处理、并发执行器、数据类和日期时间操作
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 导入系统应用组件和存储接口相关内容
from dbgpt.component import SystemApp
from dbgpt.core.interface.storage import (
    QuerySpec,
    ResourceIdentifier,
    StorageInterface,
    StorageItem,
)
# 导入执行器工具函数
from dbgpt.util.executor_utils import blocking_func_to_async

# 导入模型实例和模型注册相关内容
from ...base import ModelInstance
from ..registry import ModelRegistry


@dataclass
class ModelInstanceIdentifier(ResourceIdentifier):
    # 定义模型实例标识符的分隔符和模型名称、主机、端口
    identifier_split: str = field(default="___$$$$___", init=False)
    model_name: str
    host: str
    port: int

    def __post_init__(self):
        """后初始化方法。检查模型名称、主机和端口的有效性，并确保标识符中不包含分隔符。"""
        if self.model_name is None:
            raise ValueError("model_name is required.")
        if self.host is None:
            raise ValueError("host is required.")
        if self.port is None:
            raise ValueError("port is required.")

        if any(
            self.identifier_split in key
            for key in [self.model_name, self.host, str(self.port)]
            if key is not None
        ):
            raise ValueError(
                f"identifier_split {self.identifier_split} is not allowed in "
                f"model_name, host, port."
            )

    @property
    def str_identifier(self) -> str:
        """返回标识符的字符串表示形式，使用分隔符连接模型名称、主机和端口。"""
        return self.identifier_split.join(
            key
            for key in [
                self.model_name,
                self.host,
                str(self.port),
            ]
            if key is not None
        )

    def to_dict(self) -> Dict:
        """将标识符转换为字典形式。返回包含模型名称、主机和端口的字典。"""
        return {
            "model_name": self.model_name,
            "host": self.host,
            "port": self.port,
        }


@dataclass
class ModelInstanceStorageItem(StorageItem):
    # 模型实例存储项数据类，包含模型名称、主机、端口、权重、健康状态等信息
    model_name: str
    host: str
    port: int
    weight: Optional[float] = 1.0
    check_healthy: Optional[bool] = True
    healthy: Optional[bool] = False
    enabled: Optional[bool] = True
    prompt_template: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    _identifier: ModelInstanceIdentifier = field(init=False)

    def __post_init__(self):
        """后初始化方法。将最后心跳时间转换为日期时间对象（如果它是时间戳）。"""
        if isinstance(self.last_heartbeat, (int, float)):
            self.last_heartbeat = datetime.fromtimestamp(self.last_heartbeat)

        # 创建模型实例标识符对象
        self._identifier = ModelInstanceIdentifier(
            model_name=self.model_name,
            host=self.host,
            port=self.port,
        )

    @property
    def identifier(self) -> ModelInstanceIdentifier:
        """返回模型实例的标识符对象。"""
        return self._identifier
    # 将另一个 StorageItem 对象与当前对象进行合并
    def merge(self, other: "StorageItem") -> None:
        # 如果 other 不是 ModelInstanceStorageItem 类型，则抛出数值错误异常
        if not isinstance(other, ModelInstanceStorageItem):
            raise ValueError(f"Cannot merge with {type(other)}")
        # 调用 from_object 方法，将 other 的数据复制到当前对象中
        self.from_object(other)

    # 将对象转换成字典形式
    def to_dict(self) -> Dict:
        # 将最后心跳时间转换成时间戳
        last_heartbeat = self.last_heartbeat.timestamp()
        # 返回包含对象属性的字典
        return {
            "model_name": self.model_name,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "check_healthy": self.check_healthy,
            "healthy": self.healthy,
            "enabled": self.enabled,
            "prompt_template": self.prompt_template,
            "last_heartbeat": last_heartbeat,
        }

    # 从另一个 ModelInstanceStorageItem 对象构建当前对象
    def from_object(self, item: "ModelInstanceStorageItem") -> None:
        """Build the item from another item."""
        # 将 item 的属性复制到当前对象的对应属性上
        self.model_name = item.model_name
        self.host = item.host
        self.port = item.port
        self.weight = item.weight
        self.check_healthy = item.check_healthy
        self.healthy = item.healthy
        self.enabled = item.enabled
        self.prompt_template = item.prompt_template
        self.last_heartbeat = item.last_heartbeat

    # 从 ModelInstance 对象创建一个 ModelInstanceStorageItem 对象
    @classmethod
    def from_model_instance(cls, instance: ModelInstance) -> "ModelInstanceStorageItem":
        # 使用 ModelInstance 对象的属性来初始化一个新的 ModelInstanceStorageItem 对象
        return cls(
            model_name=instance.model_name,
            host=instance.host,
            port=instance.port,
            weight=instance.weight,
            check_healthy=instance.check_healthy,
            healthy=instance.healthy,
            enabled=instance.enabled,
            prompt_template=instance.prompt_template,
            last_heartbeat=instance.last_heartbeat,
        )

    # 将 ModelInstanceStorageItem 对象转换为 ModelInstance 对象
    @classmethod
    def to_model_instance(cls, item: "ModelInstanceStorageItem") -> ModelInstance:
        # 使用 ModelInstanceStorageItem 对象的属性来初始化一个新的 ModelInstance 对象
        return ModelInstance(
            model_name=item.model_name,
            host=item.host,
            port=item.port,
            weight=item.weight,
            check_healthy=item.check_healthy,
            healthy=item.healthy,
            enabled=item.enabled,
            prompt_template=item.prompt_template,
            last_heartbeat=item.last_heartbeat,
        )
# 定义一个存储模型注册表的类，继承自ModelRegistry类
class StorageModelRegistry(ModelRegistry):
    # 初始化方法，接受存储接口storage、系统应用system_app（可选）、执行器executor（可选）、心跳间隔时间heartbeat_interval_secs（默认60秒）、心跳超时时间heartbeat_timeout_secs（默认120秒）
    def __init__(
        self,
        storage: StorageInterface,
        system_app: SystemApp | None = None,
        executor: Optional[Executor] = None,
        heartbeat_interval_secs: float | int = 60,
        heartbeat_timeout_secs: int = 120,
    ):
        # 调用父类ModelRegistry的初始化方法，传入system_app
        super().__init__(system_app)
        # 初始化存储接口
        self._storage = storage
        # 如果未提供执行器，创建一个最大工作线程数为2的线程池执行器
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        # 设置心跳间隔时间
        self.heartbeat_interval_secs = heartbeat_interval_secs
        # 设置心跳超时时间
        self.heartbeat_timeout_secs = heartbeat_timeout_secs
        # 创建一个线程用于心跳检查，目标为自身的_heartbeat_checker方法
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_checker)
        # 将心跳线程设置为守护线程
        self.heartbeat_thread.daemon = True
        # 启动心跳线程
        self.heartbeat_thread.start()

    # 类方法，从数据库URL和数据库名称创建StorageModelRegistry对象
    @classmethod
    def from_url(
        cls,
        db_url: str,
        db_name: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        try_to_create_db: bool = False,
        **kwargs,
    ) -> "StorageModelRegistry":
        # 导入数据库管理器和初始化方法
        from dbgpt.storage.metadata.db_manager import DatabaseManager, initialize_db
        # 导入SQLAlchemy存储、JSON序列化器
        from dbgpt.storage.metadata.db_storage import SQLAlchemyStorage
        from dbgpt.util.serialization.json_serialization import JsonSerializer

        # 导入模型实例实体和适配器
        from .db_storage import ModelInstanceEntity, ModelInstanceItemAdapter

        # 定义引擎参数字典
        engine_args = {
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True,
        }

        # 初始化数据库管理器，返回DatabaseManager对象
        db: DatabaseManager = initialize_db(
            db_url, db_name, engine_args, try_to_create_db=try_to_create_db
        )
        # 创建模型实例项适配器
        storage_adapter = ModelInstanceItemAdapter()
        # 创建JSON序列化器
        serializer = JsonSerializer()
        # 创建SQLAlchemy存储对象，传入数据库管理器、模型实例实体、适配器和序列化器
        storage = SQLAlchemyStorage(
            db,
            ModelInstanceEntity,
            storage_adapter,
            serializer,
        )
        # 返回使用创建的存储对象和其他关键字参数实例化的StorageModelRegistry对象
        return cls(storage, **kwargs)

    # 异步方法，根据模型名称、主机和端口查询模型实例
    async def _get_instances_by_model(
        self, model_name: str, host: str, port: int, healthy_only: bool = False
    ) -> Tuple[List[ModelInstanceStorageItem], List[ModelInstanceStorageItem]]:
        # 创建查询规格对象，限定条件为模型名称
        query_spec = QuerySpec(conditions={"model_name": model_name})
        # 使用阻塞函数转换为异步方式，查询模型实例，返回ModelInstanceStorageItem对象列表
        instances = await blocking_func_to_async(
            self._executor, self._storage.query, query_spec, ModelInstanceStorageItem
        )
        # 如果仅查询健康的模型实例，则筛选出healthy属性为True的实例
        if healthy_only:
            instances = [ins for ins in instances if ins.healthy is True]
        # 筛选出与给定主机和端口匹配的模型实例列表
        exist_ins = [ins for ins in instances if ins.host == host and ins.port == port]
        # 返回所有查询到的模型实例列表和匹配主机端口的模型实例列表
        return instances, exist_ins
    # 定义一个心跳检查器方法，无限循环运行
    def _heartbeat_checker(self):
        # 循环查询存储中的所有模型实例
        while True:
            # 查询条件为空，获取所有 ModelInstanceStorageItem 实例
            all_instances: List[ModelInstanceStorageItem] = self._storage.query(
                QuerySpec(conditions={}), ModelInstanceStorageItem
            )
            # 遍历所有实例
            for instance in all_instances:
                # 如果实例标记为健康且超过心跳超时时间，则标记为不健康
                if (
                    instance.check_healthy
                    and datetime.now() - instance.last_heartbeat
                    > timedelta(seconds=self.heartbeat_timeout_secs)
                ):
                    instance.healthy = False
                    # 更新实例状态到存储
                    self._storage.update(instance)
            # 等待一段时间后再次执行心跳检查
            time.sleep(self.heartbeat_interval_secs)

    # 异步方法：注册一个模型实例
    async def register_instance(self, instance: ModelInstance) -> bool:
        # 提取模型名称、主机和端口信息
        model_name = instance.model_name.strip()
        host = instance.host.strip()
        port = instance.port
        # 根据模型名称、主机和端口获取现有的实例列表
        _, exist_ins = await self._get_instances_by_model(
            model_name, host, port, healthy_only=False
        )
        if exist_ins:
            # 如果存在实例，则更新现有实例的信息
            ins: ModelInstanceStorageItem = exist_ins[0]
            ins.weight = instance.weight
            ins.healthy = True
            ins.prompt_template = instance.prompt_template
            ins.last_heartbeat = datetime.now()
            # 异步更新实例到存储
            await blocking_func_to_async(self._executor, self._storage.update, ins)
        else:
            # 如果不存在实例，则保存新实例信息到存储
            new_inst = ModelInstanceStorageItem.from_model_instance(instance)
            new_inst.healthy = True
            new_inst.last_heartbeat = datetime.now()
            # 异步保存实例到存储
            await blocking_func_to_async(self._executor, self._storage.save, new_inst)
        return True

    # 异步方法：注销一个模型实例
    async def deregister_instance(self, instance: ModelInstance) -> bool:
        """Deregister a model instance.

        If the instance exists, set the instance as unhealthy, nothing to do if the
        instance does not exist.

        Args:
            instance (ModelInstance): The instance to deregister.
        """
        # 提取模型名称、主机和端口信息
        model_name = instance.model_name.strip()
        host = instance.host.strip()
        port = instance.port
        # 根据模型名称、主机和端口获取现有的实例列表
        _, exist_ins = await self._get_instances_by_model(
            model_name, host, port, healthy_only=False
        )
        if exist_ins:
            # 如果存在实例，则将实例标记为不健康
            ins = exist_ins[0]
            ins.healthy = False
            # 异步更新实例到存储
            await blocking_func_to_async(self._executor, self._storage.update, ins)
        return True

    # 异步方法：获取指定模型的所有实例列表
    async def get_all_instances(
        self, model_name: str, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """Get all instances of a model(Async).

        Args:
            model_name (str): The model name.
            healthy_only (bool): Whether only get healthy instances. Defaults to False.
        """
        # 异步获取所有模型实例列表
        return await blocking_func_to_async(
            self._executor, self.sync_get_all_instances, model_name, healthy_only
        )
    def sync_get_all_instances(
        self, model_name: str, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """Get all instances of a model.

        Args:
            model_name (str): The model name.
            healthy_only (bool): Whether only get healthy instances. Defaults to False.

        Returns:
            List[ModelInstance]: The list of instances.
        """
        # 查询存储中符合条件的模型实例
        instances = self._storage.query(
            QuerySpec(conditions={"model_name": model_name}), ModelInstanceStorageItem
        )
        # 如果需要只获取健康的实例，则筛选健康的实例
        if healthy_only:
            instances = [ins for ins in instances if ins.healthy is True]
        # 将存储项转换为模型实例并返回列表
        return [ModelInstanceStorageItem.to_model_instance(ins) for ins in instances]

    async def get_all_model_instances(
        self, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """Get all model instances.

        Args:
            healthy_only (bool): Whether only get healthy instances. Defaults to False.

        Returns:
            List[ModelInstance]: The list of instances.
        """
        # 异步获取所有模型实例
        all_instances = await blocking_func_to_async(
            self._executor,
            self._storage.query,
            QuerySpec(conditions={}),
            ModelInstanceStorageItem,
        )
        # 如果需要只获取健康的实例，则筛选健康的实例
        if healthy_only:
            all_instances = [ins for ins in all_instances if ins.healthy is True]
        # 将存储项转换为模型实例并返回列表
        return [
            ModelInstanceStorageItem.to_model_instance(ins) for ins in all_instances
        ]

    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        """Receive heartbeat from model instance.

        Update the last heartbeat time of the instance. If the instance does not exist,
        register the instance.

        Args:
            instance (ModelInstance): The instance to send heartbeat.

        Returns:
            bool: True if the heartbeat is received successfully.
        """
        # 获取实例的模型名称、主机和端口信息
        model_name = instance.model_name.strip()
        host = instance.host.strip()
        port = instance.port
        # 根据模型名称、主机和端口信息查找实例是否存在
        _, exist_ins = await self._get_instances_by_model(
            model_name, host, port, healthy_only=False
        )
        # 如果实例不存在，则注册新实例并返回 True
        if not exist_ins:
            # 从心跳注册新实例
            await self.register_instance(instance)
            return True
        else:
            # 如果实例存在，则更新最后心跳时间和健康状态
            ins = exist_ins[0]
            ins.last_heartbeat = datetime.now()
            ins.healthy = True
            # 异步更新存储中的实例信息
            await blocking_func_to_async(self._executor, self._storage.update, ins)
            return True
```