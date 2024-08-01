# `.\DB-GPT-src\dbgpt\agent\resource\manage.py`

```py
"""Resource manager."""

import logging  # 导入日志模块
from collections import defaultdict  # 导入 defaultdict 集合类型
from typing import Any, Dict, List, Optional, Type, Union, cast  # 导入类型提示模块

from dbgpt._private.pydantic import BaseModel, ConfigDict, model_validator  # 导入 Pydantic 模块
from dbgpt.component import BaseComponent, ComponentType, SystemApp  # 导入组件相关类和枚举
from dbgpt.util.parameter_utils import ParameterDescription  # 导入参数描述工具

from .base import AgentResource, Resource, ResourceParameters, ResourceType  # 导入基础资源类和类型
from .pack import ResourcePack  # 导入资源打包模块
from .tool.pack import ToolResourceType, _is_function_tool, _to_tool_list  # 导入工具相关类型和函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class RegisterResource(BaseModel):
    """Register resource model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义模型配置为允许任意类型

    name: Optional[str] = None  # 资源名称，可选
    resource_type: ResourceType  # 资源类型
    resource_type_alias: Optional[str] = None  # 资源类型别名，可选
    resource_cls: Type[Resource]  # 资源类类型
    resource_instance: Optional[Resource] = None  # 资源实例，可选，默认为 None
    is_class: bool = True  # 是否为类资源，默认为 True

    @property
    def key(self) -> str:
        """Return the key."""
        full_cls = f"{self.resource_cls.__module__}.{self.resource_cls.__qualname__}"  # 获取资源类的完整限定名
        name = self.name or full_cls  # 使用资源名称或完整限定名
        resource_type_alias = self.resource_type_alias or self.resource_type.value  # 获取资源类型别名或其值
        return f"{resource_type_alias}:{name}"  # 返回资源的唯一键

    @property
    def type_unique_key(self) -> str:
        """Return the key."""
        resource_type_alias = self.resource_type_alias or self.resource_type.value  # 获取资源类型别名或其值
        return resource_type_alias  # 返回资源类型的唯一键

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values):
        """Pre-fill the model."""
        if not isinstance(values, dict):  # 如果传入值不是字典，则直接返回
            return values
        resource_instance = values.get("resource_instance")  # 获取资源实例
        if resource_instance is not None:  # 如果资源实例不为空
            values["name"] = values["name"] or resource_instance.name  # 使用资源实例的名称填充名称字段
            values["is_class"] = False  # 将 is_class 标记为 False，表示不是类资源
            if not isinstance(resource_instance, Resource):  # 如果资源实例不是 Resource 类型
                raise ValueError(
                    f"resource_instance must be a Resource instance, not "
                    f"{type(resource_instance)}"
                )  # 抛出值错误异常
        if not values.get("resource_type"):  # 如果没有指定资源类型
            values["resource_type"] = values["resource_cls"].type()  # 使用资源类的类型方法获取资源类型
        if not values.get("resource_type_alias"):  # 如果没有指定资源类型别名
            values["resource_type_alias"] = values["resource_cls"].type_alias()  # 使用资源类的类型别名方法获取资源类型别名
        return values  # 返回填充后的值

    def get_parameter_class(self) -> Type[ResourceParameters]:
        """Return the parameter description."""
        if self.is_class:  # 如果是类资源
            return self.resource_cls.resource_parameters_class()  # 返回资源类的参数类
        return self.resource_instance.prefer_resource_parameters_class()  # 返回资源实例的首选参数类（类型不确定的情况下）


class ResourceManager(BaseComponent):
    """Resource manager.

    To manage the resources.
    """

    name = ComponentType.RESOURCE_MANAGER  # 设置组件名称为 RESOURCE_MANAGER
    def __init__(self, system_app: SystemApp):
        """Create a new AgentManager."""
        # 调用父类的构造函数初始化对象
        super().__init__(system_app)
        # 设置实例变量 system_app，表示当前的系统应用
        self.system_app = system_app
        # 初始化资源字典，用于存储所有资源对象
        self._resources: Dict[str, RegisterResource] = {}
        # 初始化类型到资源列表的映射字典，默认值为列表
        self._type_to_resources: Dict[str, List[RegisterResource]] = defaultdict(list)

    def init_app(self, system_app: SystemApp):
        """Initialize the AgentManager."""
        # 设置系统应用对象，用于更新或重新初始化 AgentManager
        self.system_app = system_app

    def after_start(self):
        """Register all resources."""
        # TODO: Register some internal resources
        # 在系统启动后注册所有资源，此处暂时未实现

    def register_resource(
        self,
        resource_cls: Optional[Type[Resource]] = None,
        resource_instance: Optional[Union[Resource, ToolResourceType]] = None,
        resource_type: Optional[ResourceType] = None,
        resource_type_alias: Optional[str] = None,
        ignore_duplicate: bool = False,
    ):
        """Register a resource."""
        # 如果 resource_instance 存在且为函数工具，则转换成工具资源类型列表并取第一个
        if resource_instance and _is_function_tool(resource_instance):
            resource_instance = _to_tool_list(resource_instance)[0]  # type: ignore

        # 如果没有提供 resource_cls 和 resource_instance，抛出 ValueError
        if resource_cls is None and resource_instance is None:
            raise ValueError("Resource class or instance must be provided.")

        name: Optional[str] = None
        # 如果 resource_instance 存在，根据其类型设置 resource_cls 和 name
        if resource_instance is not None:
            resource_cls = resource_cls or type(resource_instance)  # type: ignore
            name = resource_instance.name  # type: ignore

        # 创建 RegisterResource 对象
        resource = RegisterResource(
            name=name,
            resource_cls=resource_cls,
            resource_instance=resource_instance,
            resource_type=resource_type,
            resource_type_alias=resource_type_alias,
        )

        # 检查资源是否已存在，如果存在且不允许重复，则抛出 ValueError
        if resource.key in self._resources:
            if ignore_duplicate:
                return
            else:
                raise ValueError(f"Resource {resource.key} already exists.")

        # 将资源添加到资源字典和类型映射字典中
        self._resources[resource.key] = resource
        self._type_to_resources[resource.type_unique_key].append(resource)

    def get_supported_resources(
        self, version: Optional[str] = None
    ):
        # 获取支持的资源列表，根据版本过滤资源（未实现）
    ) -> Dict[str, Union[List[ParameterDescription], List[str]]]:
        """Return the resources."""
        # 初始化一个默认字典，用于存储资源和参数描述列表或字符串列表的映射关系
        results: Dict[str, Union[List[ParameterDescription], List[str]]] = defaultdict(
            list
        )
        # 遍历资源字典中的每一个键值对
        for key, resource in self._resources.items():
            # 获取资源的参数类
            parameter_class = resource.get_parameter_class()
            # 获取资源的唯一类型键
            resource_type = resource.type_unique_key
            # 调用参数类的方法将参数转换为配置信息
            configs: Any = parameter_class.to_configurations(
                parameter_class, version=version
            )
            # 如果版本为 "v1"，并且转换后的配置信息是列表且不为空，并且第一个元素是 ParameterDescription 类型
            if (
                version == "v1"
                and isinstance(configs, list)
                and len(configs) > 0
                and isinstance(configs[0], ParameterDescription)
            ):
                # 对于版本 "v1"，不与类兼容
                set_configs = set(results[resource_type])
                # 如果资源不是类资源，则将其名称添加到配置集合中
                if not resource.is_class:
                    for r in self._type_to_resources[resource_type]:
                        if not r.is_class:
                            set_configs.add(r.resource_instance.name)  # type: ignore
                configs = list(set_configs)
            # 将资源类型与配置信息映射存入结果字典中
            results[resource_type] = configs

        # 返回存储了资源和配置信息映射关系的字典
        return results

    def build_resource_by_type(
        self,
        type_unique_key: str,
        agent_resource: AgentResource,
        version: Optional[str] = None,
    ) -> Resource:
        """Return the resource by type."""
        # 获取指定类型键对应的资源列表
        item = self._type_to_resources.get(type_unique_key)
        # 如果列表不存在或为空，则抛出 ValueError 异常
        if not item:
            raise ValueError(f"Resource type {type_unique_key} not found.")
        # 获取非类资源实例的列表
        inst_items = [i for i in item if not i.is_class]
        # 如果存在非类资源实例
        if inst_items:
            # 如果版本为 "v1"
            if version == "v1":
                # 遍历非类资源实例列表
                for i in inst_items:
                    # 如果资源实例的名称与代理资源的值相同，则返回该资源实例
                    if (
                        i.resource_instance
                        and i.resource_instance.name == agent_resource.value
                    ):
                        return i.resource_instance
                # 如果未找到匹配的资源实例，则抛出 ValueError 异常
                raise ValueError(
                    f"Resource {agent_resource.value} not found in {type_unique_key}"
                )
            # 返回非类资源实例列表中的第一个资源实例
            return cast(Resource, inst_items[0].resource_instance)
        # 如果非类资源实例列表为空但存在多个实例，则抛出 ValueError 异常
        elif len(inst_items) > 1:
            raise ValueError(
                f"Multiple instances of resource {type_unique_key} found, "
                f"please specify the resource name."
            )
        else:
            # 如果只有一个资源实例，则尝试创建资源实例并返回
            single_item = item[0]
            try:
                # 获取单个资源实例的参数类
                parameter_cls = single_item.get_parameter_class()
                # 从代理资源的字典表示中创建参数对象
                param = parameter_cls.from_dict(agent_resource.to_dict())
                # 使用参数对象的字典形式创建资源实例
                resource_inst = single_item.resource_cls(**param.to_dict())
                # 返回创建的资源实例
                return resource_inst
            except Exception as e:
                # 如果创建失败，则记录警告并抛出 ValueError 异常
                logger.warning(f"Failed to build resource {single_item.key}: {str(e)}")
                raise ValueError(
                    f"Failed to build resource {single_item.key}: {str(e)}"
                )
    # 构建一个资源对象或资源包对象。

    # 如果 agent_resources 为空列表或 None，则返回 None
    if not agent_resources:
        return None
    
    # 初始化一个空列表用于存储依赖资源对象
    dependencies: List[Resource] = []

    # 遍历 agent_resources 列表中的每一个 AgentResource 对象
    for resource in agent_resources:
        # 调用 self.build_resource_by_type 方法，根据 resource.type 构建资源对象
        # 将构建好的资源对象添加到 dependencies 列表中
        resource_inst = self.build_resource_by_type(
            resource.type, resource, version=version
        )
        dependencies.append(resource_inst)
    
    # 如果 dependencies 中只有一个资源对象，则直接返回该资源对象
    if len(dependencies) == 1:
        return dependencies[0]
    else:
        # 如果 dependencies 中有多个资源对象，则将它们打包成一个 ResourcePack 对象并返回
        return ResourcePack(dependencies)
# Optional[SystemApp] 类型的全局变量，用于存储 SystemApp 的实例，可为空
_SYSTEM_APP: Optional[SystemApp] = None


def initialize_resource(system_app: SystemApp):
    """初始化资源管理器。

    Args:
        system_app (SystemApp): 系统应用的实例，用于初始化资源管理器。
    """
    # 将全局变量 _SYSTEM_APP 设置为传入的系统应用实例
    global _SYSTEM_APP
    _SYSTEM_APP = system_app
    # 创建资源管理器对象
    resource_manager = ResourceManager(system_app)
    # 在系统应用中注册资源管理器实例
    system_app.register_instance(resource_manager)


def get_resource_manager(system_app: Optional[SystemApp] = None) -> ResourceManager:
    """返回资源管理器对象。

    Args:
        system_app (Optional[SystemApp], optional): 可选参数，系统应用的实例。如果没有指定，则使用全局变量 _SYSTEM_APP。

    Returns:
        ResourceManager: 资源管理器对象。
    """
    # 如果全局变量 _SYSTEM_APP 为空，则进行初始化
    if not _SYSTEM_APP:
        if not system_app:
            # 如果未提供系统应用实例，则创建一个新的 SystemApp 对象
            system_app = SystemApp()
        # 使用新的系统应用实例来初始化资源
        initialize_resource(system_app)
    # 从全局变量 _SYSTEM_APP 或者传入的 system_app 中获取系统应用实例
    app = system_app or _SYSTEM_APP
    # 返回系统应用实例的资源管理器对象
    return ResourceManager.get_instance(cast(SystemApp, app))
```