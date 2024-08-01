# `.\DB-GPT-src\dbgpt\agent\resource\base.py`

```py
"""Resources for the agent."""

# 导入必要的库和模块
import dataclasses  # 用于创建数据类
import json  # 用于 JSON 操作
from abc import ABC, abstractmethod  # ABC 是抽象基类的模块，abstractmethod 用于定义抽象方法
from enum import Enum  # 用于定义枚举类型
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, cast  # 类型提示相关模块

# 导入自定义模块和类
from dbgpt._private.pydantic import BaseModel, model_to_dict  # 导入基于 Pydantic 的模块
from dbgpt.util.parameter_utils import BaseParameters, _get_parameter_descriptions  # 导入参数相关的工具函数

P = TypeVar("P", bound="ResourceParameters")  # 类型变量 P，绑定到 ResourceParameters 类型
T = TypeVar("T", bound="Resource")  # 类型变量 T，绑定到 Resource 类型


class ResourceType(str, Enum):
    """Resource type enumeration."""

    DB = "database"
    Knowledge = "knowledge"
    Internet = "internet"
    Tool = "tool"
    Plugin = "plugin"
    TextFile = "text_file"
    ExcelFile = "excel_file"
    ImageFile = "image_file"
    AWELFlow = "awel_flow"
    Pack = "pack"  # 资源类型为资源包


@dataclasses.dataclass
class ResourceParameters(BaseParameters):
    """Resource parameters class.

    It defines the parameters for building a resource.
    """

    name: str = dataclasses.field(metadata={"help": "Resource name", "tags": "fixed"})
    # 资源名称，带有说明和标签元数据

    @classmethod
    def _resource_version(cls) -> str:
        """Return the resource version."""
        return "v2"  # 返回资源的版本号为 v2

    @classmethod
    def to_configurations(
        cls, parameters: Type["ResourceParameters"], version: Optional[str] = None
    ) -> Any:
        """Convert the parameters to configurations."""
        return _get_parameter_descriptions(parameters)
        # 将参数转换为配置信息的函数调用


class Resource(ABC, Generic[P]):
    """Resource for the agent."""

    @classmethod
    @abstractmethod
    def type(cls) -> ResourceType:
        """Return the resource type."""
        # 抽象方法，返回资源的类型枚举

    @classmethod
    def type_alias(cls) -> str:
        """Return the resource type alias."""
        return cls.type().value
        # 返回资源类型的别名字符串值

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the resource name."""
        # 抽象属性，返回资源的名称字符串

    @classmethod
    def resource_parameters_class(cls) -> Type[P]:
        """Return the parameters class."""
        return ResourceParameters
        # 返回资源参数类的类型

    def prefer_resource_parameters_class(self) -> Type[P]:
        """Return the parameters class.

        You can override this method to return a different parameters class.
        It will be used to initialize the resource with parameters.
        """
        return self.resource_parameters_class()
        # 返回首选的资源参数类的类型

    def initialize_with_parameters(self, resource_parameters: P):
        """Initialize the resource with parameters."""
        pass
        # 使用给定的参数初始化资源的方法，这里仅占位不做任何实际操作

    def preload_resource(self):
        """Preload the resource."""
        pass
        # 预加载资源的方法，这里仅占位不做任何实际操作

    @classmethod
    def from_resource(
        cls: Type[T],
        resource: Optional["Resource"],
        expected_type: Optional[ResourceType] = None,
    ) -> List[T]:
        """Create a resource from another resource.

        Another resource can be a pack or a single resource, if it is a pack, it will
        return all resources which type is the same as the current resource.

        Args:
            resource(Resource): The resource.
            expected_type(ResourceType): The expected resource type.
        Returns:
            List[Resource]: The resources.
        """
        # 如果资源为空，则返回空列表
        if not resource:
            return []
        # 初始化一个空列表来存储符合期望类型的资源
        typed_resources = []
        # 遍历当前资源的指定类型的子资源，并将其类型转换为 T，然后添加到 typed_resources 中
        for r in resource.get_resource_by_type(expected_type or cls.type()):
            typed_resources.append(cast(T, r))
        # 返回符合条件的资源列表
        return typed_resources

    @abstractmethod
    async def get_prompt(
        self,
        *,
        lang: str = "en",
        prompt_type: str = "default",
        question: Optional[str] = None,
        resource_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get the prompt.

        Args:
            lang(str): The language.
            prompt_type(str): The prompt type.
            question(str): The question.
            resource_name(str): The resource name, just for the pack, it will be used
                to select specific resource in the pack.
        """
        # 抽象方法，用于获取提示信息，需要在子类中实现
        pass

    def execute(self, *args, resource_name: Optional[str] = None, **kwargs) -> Any:
        """Execute the resource."""
        # 执行资源的方法，由子类具体实现
        raise NotImplementedError

    async def async_execute(
        self, *args, resource_name: Optional[str] = None, **kwargs
    ) -> Any:
        """Execute the resource asynchronously."""
        # 异步执行资源的方法，由子类具体实现
        raise NotImplementedError

    @property
    def is_async(self) -> bool:
        """Return whether the resource is asynchronous."""
        # 返回资源是否是异步的标志
        return False

    @property
    def is_pack(self) -> bool:
        """Return whether the resource is a pack."""
        # 返回资源是否是一个 pack 的标志
        return False

    @property
    def sub_resources(self) -> List["Resource"]:
        """Return the resources."""
        # 如果资源不是 pack，则抛出异常
        if not self.is_pack:
            raise ValueError("The resource is not a pack, no sub-resources.")
        # 返回空列表，因为当前资源是一个 pack
        return []

    def get_resource_by_type(self, resource_type: ResourceType) -> List["Resource"]:
        """Get resources by type.

        If the resource is a pack, it will search the sub-resources. Otherwise, it will
        return itself if the type matches.

        Args:
            resource_type(ResourceType): The resource type.

        Returns:
            List[Resource]: The resources.
        """
        # 如果资源不是 pack，则根据类型判断是否返回自身资源
        if not self.is_pack:
            if self.type() == resource_type:
                return [self]
            else:
                return []
        # 如果是 pack，则遍历子资源，返回符合类型的资源列表
        resources = []
        for resource in self.sub_resources:
            if resource.type() == resource_type:
                resources.append(resource)
        return resources
class AgentResource(BaseModel):
    """Agent resource class."""

    # 声明资源的类型，名称，值
    type: str
    name: str
    value: str
    # 指示当前资源是否是预定义的还是动态传入的，默认为False
    is_dynamic: bool = (
        False  # Is the current resource predefined or dynamically passed in?
    )

    def resource_prompt_template(self, **kwargs) -> str:
        """Get the resource prompt template."""
        # 返回资源的提示模板，包含数据类型和介绍信息
        return "{data_type}  --{data_introduce}"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Optional["AgentResource"]:
        """Create an AgentResource object from a dictionary."""
        if d is None:
            return None
        # 从字典中创建AgentResource对象
        return AgentResource(
            type=d.get("type"),
            name=d.get("name"),
            introduce=d.get("introduce"),  # 从字典中获取介绍信息
            value=d.get("value", None),
            is_dynamic=d.get("is_dynamic", False),
            parameters=d.get("parameters", None),
        )

    @staticmethod
    def from_json_list_str(d: Optional[str]) -> Optional[List["AgentResource"]]:
        """Create a list of AgentResource objects from a json string."""
        if d is None:
            return None
        try:
            # 解析JSON字符串为数组
            json_array = json.loads(d)
        except Exception:
            # 抛出值错误异常，如果JSON字符串非法
            raise ValueError(f"Illegal AgentResource json string！{d}")
        if not isinstance(json_array, list):
            # 如果解析后的不是列表，抛出值错误异常
            raise ValueError(f"Illegal AgentResource json string！{d}")
        json_list = []
        # 遍历JSON数组中的每个项，从字典创建AgentResource对象并添加到列表中
        for item in json_array:
            r = AgentResource.from_dict(item)
            if r:
                json_list.append(r)
        return json_list

    def to_dict(self) -> Dict[str, Any]:
        """Convert the AgentResource object to a dictionary."""
        # 将AgentResource对象转换为字典
        temp = model_to_dict(self)
        # 对于枚举类型的字段，将其转换为其值
        for field, value in temp.items():
            if isinstance(value, Enum):
                temp[field] = value.value
        return temp
```