# `.\DB-GPT-src\dbgpt\model\cluster\registry.py`

```py
# 导入所需的模块和库
import itertools  # 提供用于操作迭代器的函数
import logging  # 提供日志记录功能
import random  # 提供生成随机数的功能
import threading  # 提供多线程支持
import time  # 提供时间相关的功能
from abc import ABC, abstractmethod  # 引入抽象基类和抽象方法的定义
from collections import defaultdict  # 提供了一种默认字典的实现
from datetime import datetime, timedelta  # 提供日期时间处理功能
from typing import Dict, List, Optional, Tuple  # 提供类型提示功能

# 从特定模块导入需要的类和函数
from dbgpt.component import BaseComponent, ComponentType, SystemApp
from dbgpt.model.base import ModelInstance  # 导入模型实例类

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class ModelRegistry(BaseComponent, ABC):
    """
    Abstract base class for a model registry. It provides an interface
    for registering, deregistering, fetching instances, and sending heartbeats
    for instances.
    """

    # 定义模型注册表的组件类型
    name = ComponentType.MODEL_REGISTRY

    def __init__(self, system_app: SystemApp | None = None):
        # 初始化模型注册表，设置系统应用对象
        self.system_app = system_app
        super().__init__(system_app)

    def init_app(self, system_app: SystemApp):
        """Initialize the component with the main application."""
        # 初始化组件并关联主应用
        self.system_app = system_app

    @abstractmethod
    async def register_instance(self, instance: ModelInstance) -> bool:
        """
        Register a given model instance.

        Args:
        - instance (ModelInstance): The instance of the model to register.

        Returns:
        - bool: True if registration is successful, False otherwise.
        """
        pass

    @abstractmethod
    async def deregister_instance(self, instance: ModelInstance) -> bool:
        """
        Deregister a given model instance.

        Args:
        - instance (ModelInstance): The instance of the model to deregister.

        Returns:
        - bool: True if deregistration is successful, False otherwise.
        """

    @abstractmethod
    async def get_all_instances(
        self, model_name: str, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """
        Fetch all instances of a given model. Optionally, fetch only the healthy instances.

        Args:
        - model_name (str): Name of the model to fetch instances for.
        - healthy_only (bool, optional): If set to True, fetches only the healthy instances.
                                         Defaults to False.

        Returns:
        - List[ModelInstance]: A list of instances for the given model.
        """

    @abstractmethod
    def sync_get_all_instances(
        self, model_name: str, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """Fetch all instances of a given model. Optionally, fetch only the healthy instances."""

    @abstractmethod
    async def get_all_model_instances(
        self, healthy_only: bool = False
    ) -> List[ModelInstance]:
        """
        Fetch all instances of all models, Optionally, fetch only the healthy instances.

        Returns:
        - List[ModelInstance]: A list of instances for the all models.
        """
    # 异步方法：选择给定模型的一个健康且启用的实例
    async def select_one_health_instance(self, model_name: str) -> ModelInstance:
        """
        Selects one healthy and enabled instance for a given model.

        Args:
        - model_name (str): Name of the model.

        Returns:
        - ModelInstance: One randomly selected healthy and enabled instance, or None if no such instance exists.
        """
        # 获取所有健康的实例列表（仅健康状态）
        instances = await self.get_all_instances(model_name, healthy_only=True)
        # 过滤出所有已启用的实例
        instances = [i for i in instances if i.enabled]
        # 如果实例列表为空，则返回 None
        if not instances:
            return None
        # 从符合条件的实例中随机选择一个实例并返回
        return random.choice(instances)

    @abstractmethod
    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        """
        Send a heartbeat for a given model instance. This can be used to
        verify if the instance is still alive and functioning.

        Args:
        - instance (ModelInstance): The instance of the model to send a heartbeat for.

        Returns:
        - bool: True if heartbeat is successful, False otherwise.
        """
    # EmbeddedModelRegistry 类，继承自 ModelRegistry，用于管理嵌入式模型实例的注册和健康检查
    def __init__(
        self,
        system_app: SystemApp | None = None,  # 可选的系统应用对象，用于初始化基类
        heartbeat_interval_secs: int = 60,    # 心跳检测间隔时间，默认60秒
        heartbeat_timeout_secs: int = 120,    # 心跳超时时间，默认120秒
    ):
        super().__init__(system_app)  # 调用基类的初始化方法
        self.registry: Dict[str, List[ModelInstance]] = defaultdict(list)  # 实例注册字典，以模型名为键，值为实例列表
        self.heartbeat_interval_secs = heartbeat_interval_secs  # 设置心跳检测间隔时间
        self.heartbeat_timeout_secs = heartbeat_timeout_secs  # 设置心跳超时时间
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_checker)  # 创建心跳检测线程
        self.heartbeat_thread.daemon = True  # 将心跳检测线程设为守护线程
        self.heartbeat_thread.start()  # 启动心跳检测线程

    # 获取指定模型实例的健康和非健康状态的列表
    def _get_instances(
        self, model_name: str, host: str, port: int, healthy_only: bool = False
    ) -> Tuple[List[ModelInstance], List[ModelInstance]]:
        instances = self.registry[model_name]  # 获取指定模型名的实例列表
        if healthy_only:
            instances = [ins for ins in instances if ins.healthy == True]  # 若仅需健康实例，则过滤出健康状态为 True 的实例
        exist_ins = [ins for ins in instances if ins.host == host and ins.port == port]  # 查找与指定主机和端口匹配的实例列表
        return instances, exist_ins  # 返回所有实例和匹配实例列表

    # 心跳检测线程的运行函数，定期检查实例的健康状态并更新
    def _heartbeat_checker(self):
        while True:
            for instances in self.registry.values():  # 遍历所有模型的实例列表
                for instance in instances:  # 遍历每个模型的实例
                    if (
                        instance.check_healthy  # 若实例支持健康检查
                        and datetime.now() - instance.last_heartbeat  # 计算当前时间与上次心跳时间的时间差
                        > timedelta(seconds=self.heartbeat_timeout_secs)  # 若超过设定的心跳超时时间
                    ):
                        instance.healthy = False  # 将实例标记为不健康
            time.sleep(self.heartbeat_interval_secs)  # 等待下一次心跳检测间隔时间

    # 注册新的模型实例，若已存在则更新实例信息和状态
    async def register_instance(self, instance: ModelInstance) -> bool:
        model_name = instance.model_name.strip()  # 获取模型名并去除空格
        host = instance.host.strip()  # 获取主机地址并去除空格
        port = instance.port  # 获取端口号

        instances, exist_ins = self._get_instances(
            model_name, host, port, healthy_only=False
        )  # 获取指定模型、主机和端口的所有实例及已存在的实例列表
        if exist_ins:
            # 若存在已注册的实例，则更新实例信息
            ins = exist_ins[0]
            ins.weight = instance.weight
            ins.healthy = True
            ins.prompt_template = instance.prompt_template
            ins.last_heartbeat = datetime.now()
        else:
            # 若不存在已注册的实例，则将新实例添加到列表中
            instance.healthy = True
            instance.last_heartbeat = datetime.now()
            instances.append(instance)
        return True  # 返回注册成功的标志

    # 注销模型实例，将指定实例标记为不健康
    async def deregister_instance(self, instance: ModelInstance) -> bool:
        model_name = instance.model_name.strip()  # 获取模型名并去除空格
        host = instance.host.strip()  # 获取主机地址并去除空格
        port = instance.port  # 获取端口号
        _, exist_ins = self._get_instances(model_name, host, port, healthy_only=False)  # 获取指定模型、主机和端口的所有实例及已存在的实例列表
        if exist_ins:
            ins = exist_ins[0]
            ins.healthy = False  # 将找到的实例标记为不健康
        return True  # 返回注销成功的标志

    # 获取指定模型的所有实例列表（包括健康和不健康的）
    async def get_all_instances(
        self, model_name: str, healthy_only: bool = False
    ) -> List[ModelInstance]:
        return self.sync_get_all_instances(model_name, healthy_only)  # 调用同步方法获取实例列表

    # 同步方法：获取指定模型的所有实例列表（包括健康和不健康的）
    def sync_get_all_instances(
        self, model_name: str, healthy_only: bool = False
    ):
    ) -> List[ModelInstance]:
        # 返回给定模型名称对应的实例列表
        instances = self.registry[model_name]
        # 如果只筛选健康的实例，则过滤掉非健康的实例
        if healthy_only:
            instances = [ins for ins in instances if ins.healthy == True]
        # 返回满足条件的实例列表
        return instances

    async def get_all_model_instances(
        self, healthy_only: bool = False
    ) -> List[ModelInstance]:
        # 记录当前注册表的元数据到日志
        logger.debug("Current registry metadata:\n{self.registry}")
        # 获取所有模型实例，并将它们展平为单个列表
        instances = list(itertools.chain(*self.registry.values()))
        # 如果只筛选健康的实例，则过滤掉非健康的实例
        if healthy_only:
            instances = [ins for ins in instances if ins.healthy == True]
        # 返回满足条件的实例列表
        return instances

    async def send_heartbeat(self, instance: ModelInstance) -> bool:
        # 获取与给定实例相关的现存实例
        _, exist_ins = self._get_instances(
            instance.model_name, instance.host, instance.port, healthy_only=False
        )
        # 如果不存在现存实例，则注册新的实例并返回True
        if not exist_ins:
            await self.register_instance(instance)
            return True

        # 否则更新现存实例的心跳时间和健康状态，并返回True
        ins = exist_ins[0]
        ins.last_heartbeat = datetime.now()
        ins.healthy = True
        return True
```