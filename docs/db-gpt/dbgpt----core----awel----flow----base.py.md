# `.\DB-GPT-src\dbgpt\core\awel\flow\base.py`

```py
# 导入必要的模块和类
import abc
import dataclasses
import inspect
from abc import ABC
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

# 导入内部模块和函数
from dbgpt._private.pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_to_dict,
    model_validator,
)
from dbgpt.core.awel.util.parameter_util import BaseDynamicOptions, OptionValue
from dbgpt.core.interface.serialization import Serializable

# 导入自定义异常类
from .exceptions import FlowMetadataException, FlowParameterMetadataException

# 类型注册表，用于存储类型名称和类型对象的映射关系
_TYPE_REGISTRY: Dict[str, Type] = {}

# 允许使用的基本类型字典
_ALLOWED_TYPES: Dict[str, Type] = {
    "str": str,
    "int": int,
}

# 基本数据类型列表
_BASIC_TYPES = [str, int, float, bool, dict, list, set]

# 定义一个类型变量 T，用于泛型约束
T = TypeVar("T", bound="ViewMixin")
TM = TypeVar("TM", bound="TypeMetadata")


def _get_type_name(type_: Type[Any]) -> str:
    """获取类型的完整名称并注册到类型注册表中（如果尚未注册）。

    Args:
        type_ (Type[Any]): 待获取名称的类型对象。

    Returns:
        str: 类型的完整名称。
    """
    type_name = f"{type_.__module__}.{type_.__qualname__}"

    # 如果类型名称尚未注册，则将其注册到类型注册表中
    if type_name not in _TYPE_REGISTRY:
        _TYPE_REGISTRY[type_name] = type_

    return type_name


def _register_alias_types(type_: Type[Any], alias_ids: Optional[List[str]] = None):
    """注册类型的别名到类型注册表中。

    Args:
        type_ (Type[Any]): 要注册的类型对象。
        alias_ids (Optional[List[str]], optional): 类型的别名列表。默认为None。
    """
    if alias_ids:
        for alias_id in alias_ids:
            # 如果别名尚未注册，则将其注册到类型注册表中
            if alias_id not in _TYPE_REGISTRY:
                _TYPE_REGISTRY[alias_id] = type_


def _get_type_cls(type_name: str) -> Type[Any]:
    """根据类型名称从类型注册表中获取对应的类型对象。

    Args:
        type_name (str): 类型的完整名称。

    Returns:
        Type[Any]: 对应的类型对象。

    Raises:
        ValueError: 如果指定类型未注册在类型注册表中。
    """
    # 从兼容性模块中导入函数
    from .compat import get_new_class_name

    # 获取类型的新类名
    new_cls = get_new_class_name(type_name)

    # 如果类型名称在类型注册表中，则返回对应的类型对象
    if type_name in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[type_name]
    # 如果类型的新类名存在且在类型注册表中，则返回对应的类型对象
    elif new_cls and new_cls in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[new_cls]
    else:
        # 否则抛出数值错误异常，指示类型未注册
        raise ValueError(f"Type {type_name} not registered.")


# 注册基本数据类型到类型注册表中
for t in _BASIC_TYPES:
    _get_type_name(t)


class _MISSING_TYPE:
    pass


# 定义一个表示缺失值的类型对象
_MISSING_VALUE = _MISSING_TYPE()


def _serialize_complex_obj(obj: Any) -> Any:
    """序列化复杂对象为可序列化的字典形式。

    Args:
        obj (Any): 待序列化的对象。

    Returns:
        Any: 序列化后的对象或值。
    """
    # 如果对象是可序列化对象，则转换为字典形式
    if isinstance(obj, Serializable):
        return obj.to_dict()
    # 如果对象是数据类对象，则转换为字典形式
    elif dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    # 如果对象是枚举类型，则返回其值
    elif isinstance(obj, Enum):
        return obj.value
    # 如果对象是日期时间类型，则返回 ISO 格式字符串
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        # 否则直接返回对象本身
        return obj


def _serialize_recursive(data: Any) -> Any:
    """递归地序列化复杂数据结构为可序列化的形式。

    Args:
        data (Any): 待序列化的数据。

    Returns:
        Any: 序列化后的数据。
    """
    # 如果数据是字典类型，则递归序列化每个键值对
    if isinstance(data, dict):
        return {key: _serialize_complex_obj(value) for key, value in data.items()}
    # 如果数据是列表类型，则递归序列化每个元素
    elif isinstance(data, list):
        return [_serialize_complex_obj(item) for item in data]
    else:
        # 否则直接序列化对象本身
        return _serialize_complex_obj(data)


class _CategoryDetail:
    """类别详细信息类，用于表示类别的详细信息。"""
    # 初始化函数，用于创建类的实例并初始化其属性
    def __init__(self, label: str, description: str):
        # 将传入的标签赋值给实例变量 label
        self.label = label
        # 将传入的描述赋值给实例变量 description
        self.description = description
# 定义一个常量字典，用于存储操作符类别的详细信息
_OPERATOR_CATEGORY_DETAIL = {
    "trigger": _CategoryDetail("Trigger", "Trigger your AWEL flow"),
    "sender": _CategoryDetail("Sender", "Send the data to the target"),
    "llm": _CategoryDetail("LLM", "Invoke LLM model"),
    "conversion": _CategoryDetail("Conversion", "Handle the conversion"),
    "output_parser": _CategoryDetail("Output Parser", "Parse the output of LLM model"),
    "common": _CategoryDetail("Common", "The common operator"),
    "agent": _CategoryDetail("Agent", "The agent operator"),
    "rag": _CategoryDetail("RAG", "The RAG operator"),
    "experimental": _CategoryDetail("EXPERIMENTAL", "EXPERIMENTAL operator"),
}

# 定义枚举类 `OperatorCategory` 表示操作符的类别
class OperatorCategory(str, Enum):
    """The category of the operator."""

    # 操作符的类别枚举值，每个值对应上述定义的常量字典中的键
    TRIGGER = "trigger"
    SENDER = "sender"
    LLM = "llm"
    CONVERSION = "conversion"
    OUTPUT_PARSER = "output_parser"
    COMMON = "common"
    AGENT = "agent"
    RAG = "rag"
    EXPERIMENTAL = "experimental"

    def label(self) -> str:
        """Get the label of the category."""
        # 返回当前操作符类别的标签，从常量字典中获取并返回
        return _OPERATOR_CATEGORY_DETAIL[self.value].label

    def description(self) -> str:
        """Get the description of the category."""
        # 返回当前操作符类别的描述，从常量字典中获取并返回
        return _OPERATOR_CATEGORY_DETAIL[self.value].description

    @classmethod
    def value_of(cls, value: str) -> "OperatorCategory":
        """Get the category by the value."""
        # 根据给定的值查找并返回对应的操作符类别枚举
        for category in cls:
            if category.value == value:
                return category
        # 如果找不到对应的操作符类别，则抛出 ValueError 异常
        raise ValueError(f"Can't find the category for value {value}")


# 定义一个常量字典，用于存储资源类别的详细信息
_RESOURCE_CATEGORY_DETAIL = {
    "http_body": _CategoryDetail("HTTP Body", "The HTTP body"),
    "llm_client": _CategoryDetail("LLM Client", "The LLM client"),
    "storage": _CategoryDetail("Storage", "The storage resource"),
    "serializer": _CategoryDetail("Serializer", "The serializer resource"),
    "common": _CategoryDetail("Common", "The common resource"),
    "prompt": _CategoryDetail("Prompt", "The prompt resource"),
    "agent": _CategoryDetail("Agent", "The agent resource"),
    "embeddings": _CategoryDetail("Embeddings", "The embeddings resource"),
    "rag": _CategoryDetail("RAG", "The  resource"),
    "vector_store": _CategoryDetail("Vector Store", "The vector store resource"),
}

# 定义枚举类 `ResourceCategory` 表示资源的类别
class ResourceCategory(str, Enum):
    """The category of the resource."""

    # 资源的类别枚举值，每个值对应上述定义的常量字典中的键
    HTTP_BODY = "http_body"
    LLM_CLIENT = "llm_client"
    STORAGE = "storage"
    SERIALIZER = "serializer"
    COMMON = "common"
    PROMPT = "prompt"
    AGENT = "agent"
    EMBEDDINGS = "embeddings"
    RAG = "rag"
    VECTOR_STORE = "vector_store"

    def label(self) -> str:
        """Get the label of the category."""
        # 返回当前资源类别的标签，从常量字典中获取并返回
        return _RESOURCE_CATEGORY_DETAIL[self.value].label
    # 返回该类别的描述信息
    def description(self) -> str:
        """Get the description of the category."""
        # 从预定义的资源类别详细信息字典中，根据当前实例的值获取描述信息
        return _RESOURCE_CATEGORY_DETAIL[self.value].description

    # 根据给定的值获取对应的资源类别实例
    @classmethod
    def value_of(cls, value: str) -> "ResourceCategory":
        """Get the category by the value."""
        # 遍历所有的资源类别枚举值
        for category in cls:
            # 如果找到匹配的值，返回对应的资源类别实例
            if category.value == value:
                return category
        # 如果找不到匹配的资源类别实例，抛出值错误异常
        raise ValueError(f"Can't find the category for value {value}")
# 定义资源类型的枚举类，继承自字符串类型，用于表示资源的类型
class ResourceType(str, Enum):
    """The type of the resource."""

    INSTANCE = "instance"  # 表示实例类型的资源
    CLASS = "class"  # 表示类类型的资源


# 定义参数类型的枚举类，继承自字符串类型，用于表示参数的数据类型
class ParameterType(str, Enum):
    """The type of the parameter."""

    STRING = "str"  # 字符串类型的参数
    INT = "int"  # 整数类型的参数
    FLOAT = "float"  # 浮点数类型的参数
    BOOL = "bool"  # 布尔类型的参数
    DICT = "dict"  # 字典类型的参数
    LIST = "list"  # 列表类型的参数


# 定义参数类别的枚举类，继承自字符串类型，表示参数的类别
class ParameterCategory(str, Enum):
    """The category of the parameter."""

    COMMON = "common"  # 普通类别的参数
    RESOURCER = "resource"  # 资源类别的参数

    @classmethod
    def values(cls) -> List[str]:
        """Get the values of the category."""
        return [category.value for category in cls]

    @classmethod
    def get_category(cls, value: Type[Any]) -> "ParameterCategory":
        """Get the category of the value.

        Args:
            value (Any): The value.

        Returns:
            ParameterCategory: The category of the value.
        """
        if value in _BASIC_TYPES:  # 判断值是否属于基本类型
            return cls.COMMON  # 返回普通类别
        else:
            return cls.RESOURCER  # 返回资源类别


# 定义默认参数类型，可以是字符串、整数、浮点数、布尔值或空值的联合类型
DefaultParameterType = Union[str, int, float, bool, None]


# 定义类型元数据类，继承自BaseModel，表示参数类型的元数据信息
class TypeMetadata(BaseModel):
    """The metadata of the type."""

    type_name: str = Field(
        ..., description="The type short name of the parameter", examples=["str", "int"]
    )

    type_cls: str = Field(
        ...,
        description="The type class of the parameter",
        examples=["builtins.str", "builtins.int"],
    )

    def new(self: TM) -> TM:
        """Copy the metadata."""
        return self.__class__(**self.model_dump(exclude_defaults=True))


# 定义参数类，继承自TypeMetadata和Serializable，表示构建操作符的参数
class Parameter(TypeMetadata, Serializable):
    """Parameter for build operator."""

    label: str = Field(
        ..., description="The label to display in UI", examples=["OpenAI API Key"]
    )
    name: str = Field(
        ..., description="The name of the parameter", examples=["apk_key"]
    )
    is_list: bool = Field(
        default=False,
        description="Whether current parameter is list",
        examples=[True, False],
    )
    category: str = Field(
        ...,
        description="The category of the parameter",
        examples=["common", "resource"],
    )
    resource_type: ResourceType = Field(
        default=ResourceType.INSTANCE,
        description="The type of the resource, just for resource type",
        examples=["instance", "class"],
    )
    optional: bool = Field(
        ..., description="Whether the parameter is optional", examples=[True, False]
    )
    default: Optional[DefaultParameterType] = Field(
        None, description="The default value of the parameter"
    )
    placeholder: Optional[DefaultParameterType] = Field(
        None, description="The placeholder of the parameter"
    )
    description: Optional[str] = Field(
        None, description="The description of the parameter"
    )
    # 参数的选项，可以是动态选项对象或选项值列表的可选类型，用于描述参数的选项
    options: Optional[Union[BaseDynamicOptions, List[OptionValue]]] = Field(
        None, description="The options of the parameter"
    )
    # 参数的值，保存在 DAG 文件中的参数值，可选的任意类型
    value: Optional[Any] = Field(
        None, description="The value of the parameter(Saved in the dag file)"
    )
    # 参数的别名列表，用于兼容旧版本的别名设置
    alias: Optional[List[str]] = Field(
        None, description="The alias of the parameter(Compatible with old version)"
    )

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """预填充元数据。

        将值转换为真实类型。
        """
        if not isinstance(values, dict):
            return values
        type_cls = values.get("type_cls")
        to_handle_values = {
            "value": values.get("value"),
            "default": values.get("default"),
        }
        if type_cls:
            for k, v in to_handle_values.items():
                if v:
                    handled_v = cls._covert_to_real_type(type_cls, v)
                    values[k] = handled_v
        return values

    @classmethod
    def _covert_to_real_type(cls, type_cls: str, v: Any):
        """将值转换为真实类型。

        支持字符串、整数、浮点数和布尔值的转换。
        """
        if type_cls and v is not None:
            try:
                # 尝试将值转换为指定类型
                if type_cls == "builtins.str":
                    return str(v)
                elif type_cls == "builtins.int":
                    return int(v)
                elif type_cls == "builtins.float":
                    return float(v)
                elif type_cls == "builtins.bool":
                    if str(v).lower() in ["false", "0", "", "no", "off"]:
                        return False
                    return bool(v)
            except ValueError:
                raise ValidationError(f"Value '{v}' is not valid for type {type_cls}")
        return v

    def get_typed_value(self) -> Any:
        """获取类型化的值。"""
        return self._covert_to_real_type(self.type_cls, self.value)

    def get_typed_default(self) -> Any:
        """获取类型化的默认值。"""
        return self._covert_to_real_type(self.type_cls, self.default)

    @classmethod
    def build_from(
        cls,
        label: str,
        name: str,
        type: Type,
        optional: bool = False,
        default: Optional[Union[DefaultParameterType, _MISSING_TYPE]] = _MISSING_VALUE,
        placeholder: Optional[DefaultParameterType] = None,
        description: Optional[str] = None,
        options: Optional[Union[BaseDynamicOptions, List[OptionValue]]] = None,
        resource_type: ResourceType = ResourceType.INSTANCE,
        alias: Optional[List[str]] = None,
    ):
        """从给定参数构建对象。

        创建参数对象，包括标签、名称、类型、可选性、默认值、占位符、描述、选项、资源类型和别名。
        """
    ):
        """Build the parameter from the type."""
        # 从类型构建参数对象

        type_name = type.__qualname__
        # 获取类型的限定名称（qualified name）

        type_cls = _get_type_name(type)
        # 调用 _get_type_name 函数获取类型的名称

        category = ParameterCategory.get_category(type)
        # 根据类型获取参数的分类

        if optional and default == _MISSING_VALUE:
            # 如果参数是可选的，并且缺少默认值，则抛出 ValueError 异常
            raise ValueError(f"Default value is missing for optional parameter {name}.")

        if not optional:
            # 如果参数不是可选的，则将默认值设为 None
            default = None

        return cls(
            label=label,
            name=name,
            type_name=type_name,
            type_cls=type_cls,
            category=category.value,
            resource_type=resource_type,
            optional=optional,
            default=default,
            placeholder=placeholder,
            description=description or label,
            options=options,
            alias=alias,
        )

    @classmethod
    def build_from_ui(cls, data: Dict) -> "Parameter":
        """Build the parameter from the type.

        Some fields are not trusted, so we need to check the type.

        Args:
            data (Dict): The parameter data.

        Returns:
            Parameter: The parameter.
        """
        # 从用户界面构建参数对象，需要检查字段的类型

        type_str = data["type_cls"]
        # 获取参数数据中的类型字符串

        type_name = data["type_name"]
        # 获取参数数据中的类型名称

        category = ParameterCategory.get_category(_get_type_cls(type_str))
        # 根据参数数据中的类型字符串获取参数的分类

        return cls(
            label=data["label"],
            name=data["name"],
            type_name=type_name,
            type_cls=type_str,
            category=category.value,
            optional=data["optional"],
            default=data["default"],
            description=data["description"],
            options=data["options"],
            value=data["value"],
        )

    def to_dict(self) -> Dict:
        """Convert current metadata to json dict."""
        # 将当前参数对象转换为 JSON 字典格式

        dict_value = model_to_dict(self, exclude={"options", "alias"})
        # 使用 model_to_dict 函数将对象转换为字典，排除 "options" 和 "alias" 字段

        if not self.options:
            dict_value["options"] = None
        elif isinstance(self.options, BaseDynamicOptions):
            # 如果参数的选项是动态选项类型，则获取其选项值并转换为字典格式
            values = self.options.option_values()
            dict_value["options"] = [value.to_dict() for value in values]
        else:
            # 否则，假设参数的选项是固定选项列表，则将每个选项转换为字典格式
            dict_value["options"] = [value.to_dict() for value in self.options]

        return dict_value

    def get_dict_options(self) -> Optional[List[Dict]]:
        """Get the options of the parameter."""
        # 获取参数的选项列表，并将其转换为字典格式的列表

        if not self.options:
            return None
        elif isinstance(self.options, BaseDynamicOptions):
            # 如果参数的选项是动态选项类型，则获取其选项值并转换为字典格式
            values = self.options.option_values()
            return [value.to_dict() for value in values]
        else:
            # 否则，假设参数的选项是固定选项列表，则将每个选项转换为字典格式
            return [value.to_dict() for value in self.options]

    def to_runnable_parameter(
        self,
        view_value: Any,
        resources: Optional[Dict[str, "ResourceMetadata"]] = None,
        key_to_resource_instance: Optional[Dict[str, Any]] = None,
        ):
        """Prepare parameter to be runnable."""
        # 准备参数以便执行运行时使用
    ) -> Dict:
        """将参数转换为可运行的参数。

        Args:
            view_value (Any): 来自 UI 的值。
            resources (Optional[Dict[str, "ResourceMetadata"]], optional):
                资源。默认为 None。
            key_to_resource_instance (Optional[Dict[str, Any]], optional):

        Returns:
            Dict: 可运行的参数。
        """
        if (
            view_value is not None
            and self.category == ParameterCategory.RESOURCER
            and resources
            and key_to_resource_instance
        ):
            # 资源类型可以有多个参数。
            resource_id = view_value
            resource_metadata = resources[resource_id]
            # 检查类型。
            resource_type = _get_type_cls(resource_metadata.type_cls)
            if self.resource_type == ResourceType.CLASS:
                # 只需要类型，不需要实例。
                value: Any = resource_type
            else:
                if resource_id not in key_to_resource_instance:
                    raise FlowParameterMetadataException(
                        f"The dependency resource {resource_id} not found."
                    )
                resource_inst = key_to_resource_instance[resource_id]
                value = resource_inst
                if value is not None and not isinstance(value, resource_type):
                    raise FlowParameterMetadataException(
                        f"Resource {resource_id} is not an instance of {resource_type}"
                    )
        else:
            value = self.get_typed_default()
            if self.value is not None:
                value = self.value
            if view_value is not None:
                value = view_value
        return {self.name: value}
class BaseResource(Serializable, BaseModel):
    """The base resource."""

    label: str = Field(
        ...,
        description="The label to display in UI",
        examples=["LLM Operator", "OpenAI LLM Client"],
    )
    name: str = Field(
        ...,
        description="The name of the operator",
        examples=["llm_operator", "openai_llm_client"],
    )
    description: str = Field(
        ...,
        description="The description of the field",
        examples=["The LLM operator.", "OpenAI LLM Client"],
    )

    def to_dict(self) -> Dict:
        """Convert current metadata to json dict."""
        return model_to_dict(self)



# 根据类的名称和字段定义，定义了一个基础的资源类，包含标签、名称和描述等字段，可以被序列化并转为字典形式。
class Resource(BaseResource, TypeMetadata):
    """The resource of the operator."""

    pass



# 枚举类型，表示输入或输出字段的类型。
class IOFiledType(str, Enum):
    """The type of the input or output field."""

    STRING = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DICT = "dict"
    LIST = "list"



# 表示运算符的输入或输出字段。
class IOField(Resource):
    """The input or output field of the operator."""

    is_list: bool = Field(
        default=False,
        description="Whether current field is list",
        examples=[True, False],
    )

    @classmethod
    def build_from(
        cls,
        label: str,
        name: str,
        type: Type,
        description: Optional[str] = None,
        is_list: bool = False,
    ):
        """Build the resource from the type."""
        type_name = type.__qualname__
        type_cls = _get_type_name(type)
        return cls(
            label=label,
            name=name,
            type_name=type_name,
            type_cls=type_cls,
            is_list=is_list,
            description=description or label,
        )



# 表示基础元数据，继承自BaseResource，包含了运算符或资源的类别、流类型、图标、文档链接和ID等信息。
class BaseMetadata(BaseResource):
    """The base metadata."""

    category: Union[OperatorCategory, ResourceCategory] = Field(
        ...,
        description="The category of the operator",
        examples=[OperatorCategory.LLM.value, ResourceCategory.LLM_CLIENT.value],
    )
    category_label: str = Field(
        ...,
        description="The category label of the metadata(Just for UI)",
        examples=["LLM", "Resource"],
    )

    flow_type: Optional[str] = Field(
        ..., description="The flow type", examples=["operator", "resource"]
    )
    icon: Optional[str] = Field(
        default=None,
        description="The icon of the operator or resource",
        examples=["public/awel/icons/llm.svg"],
    )
    documentation_url: Optional[str] = Field(
        default=None,
        description="The documentation url of the operator or resource",
        examples=["https://docs.dbgpt.site/docs/awel"],
    )

    id: str = Field(
        description="The id of the operator or resource",
        examples=[
            "operator_llm_operator___$$___llm___$$___v1",
            "resource_dbgpt.model.proxy.llms.chatgpt.OpenAILLMClient",
        ],
    )
    # 定义一个可选的标签列表，用于描述运算符的标签信息
    tags: Optional[List[str]] = Field(
        default=None,
        description="The tags of the operator",
        examples=[["llm", "openai", "gpt3"]],
    )

    # 定义一个参数列表，描述运算符或资源的参数信息
    parameters: List[Parameter] = Field(
        ..., description="The parameters of the operator or resource"
    )

    # 属性方法，用于判断当前元数据是否表示运算符
    @property
    def is_operator(self) -> bool:
        """Whether the metadata is for operator."""
        return self.flow_type == "operator"

    # 方法用于获取可运行的参数列表
    def get_runnable_parameters(
        self,
        view_parameters: Optional[List[Parameter]],
        resources: Optional[Dict[str, "ResourceMetadata"]] = None,
        key_to_resource_instance: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """获取可运行的参数。

        Args:
            view_parameters (Optional[List[Parameter]]):
                来自UI的参数。
            resources (Optional[Dict[str, "ResourceMetadata"]], optional):
                资源。默认为None。
            key_to_resource_instance (Optional[Dict[str, Any]], optional):

        Returns:
            Dict: 可运行的参数。
        """
        runnable_parameters: Dict[str, Any] = {}
        if not self.parameters or not view_parameters:
            return runnable_parameters
        view_required_parameters = {
            parameter.name: parameter
            for parameter in view_parameters
            if not parameter.optional
        }
        current_required_parameters = {
            parameter.name: parameter
            for parameter in self.parameters
            if not parameter.optional
        }
        current_parameters = {}
        current_aliases_parameters = {}
        for parameter in self.parameters:
            current_parameters[parameter.name] = parameter
            if parameter.alias:
                for alias in parameter.alias:
                    if alias in current_aliases_parameters:
                        raise FlowMetadataException(
                            f"Alias {alias} already exists in the metadata."
                        )
                    current_aliases_parameters[alias] = parameter

        if len(view_required_parameters) < len(current_required_parameters):
            # TODO, skip the optional parameters.
            raise FlowParameterMetadataException(
                f"Parameters count not match(current key: {self.id}). "
                f"Expected {len(current_required_parameters)}, "
                f"but got {len(view_required_parameters)} from JSON metadata."
                f"Required parameters: {current_required_parameters.keys()}, "
                f"but got {view_required_parameters.keys()}."
            )
        for view_param in view_parameters:
            view_param_key = view_param.name
            if view_param_key in current_parameters:
                current_parameter = current_parameters[view_param_key]
            elif view_param_key in current_aliases_parameters:
                current_parameter = current_aliases_parameters[view_param_key]
            else:
                raise FlowParameterMetadataException(
                    f"Parameter {view_param_key} not in the metadata."
                )
            runnable_parameters.update(
                current_parameter.to_runnable_parameter(
                    view_param.get_typed_value(), resources, key_to_resource_instance
                )
            )
        return runnable_parameters

    @model_validator(mode="before")
    @classmethod
    # 定义一个类方法，用于预填充元数据
    def base_pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre fill the metadata."""
        # 如果传入的值不是字典类型，则直接返回该值
        if not isinstance(values, dict):
            return values
        # 如果传入的值中不包含 "category_label" 键
        if "category_label" not in values:
            # 获取 "category" 键对应的值
            category = values["category"]
            # 如果 "category" 是字符串类型
            if isinstance(category, str):
                # 根据类是否为 ResourceMetadata 的子类来决定如何处理 category
                if issubclass(cls, ResourceMetadata):
                    # 根据 category 的值获取对应的 ResourceCategory 枚举
                    category = ResourceCategory.value_of(category)
                else:
                    # 根据 category 的值获取对应的 OperatorCategory 枚举
                    category = OperatorCategory.value_of(category)
            # 将获取到的 category 的标签赋给 "category_label" 键
            values["category_label"] = category.label()
        # 返回处理后的字典
        return values

    # 定义一个实例方法，用于获取原始 id
    def get_origin_id(self) -> str:
        """Get the origin id."""
        # 将 self.id 按下划线 "_" 分割，并取除了最后一部分的所有部分
        split_ids = self.id.split("_")
        # 将分割后的部分重新连接起来，形成原始 id
        return "_".join(split_ids[:-1])

    # 定义一个实例方法，用于将当前元数据转换为 JSON 字典
    def to_dict(self) -> Dict:
        """Convert current metadata to json dict."""
        # 使用 model_to_dict 函数将当前对象转换为字典，排除 "parameters" 键
        dict_value = model_to_dict(self, exclude={"parameters"})
        # 将对象的 parameters 属性中的每个 parameter 对象转换为字典，并赋给 "parameters" 键
        dict_value["parameters"] = [
            parameter.to_dict() for parameter in self.parameters
        ]
        # 返回转换后的字典
        return dict_value
class ResourceMetadata(BaseMetadata, TypeMetadata):
    """The metadata of the resource."""

    resource_type: ResourceType = Field(
        default=ResourceType.INSTANCE,
        description="The type of the resource",
        examples=["instance", "class"],
    )

    parent_cls: List[str] = Field(
        default_factory=list,
        description="The parent class of the resource",
        examples=[
            "dbgpt.core.interface.llm.LLMClient",
            "resource_dbgpt.model.proxy.llms.chatgpt.OpenAILLMClient",
        ],
    )

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre fill the metadata."""
        # 如果传入的值不是字典类型，则直接返回该值
        if not isinstance(values, dict):
            return values
        # 如果 values 中没有 'flow_type' 键，则设置 'flow_type' 为 'resource'
        if "flow_type" not in values:
            values["flow_type"] = "resource"
        # 如果 values 中没有 'id' 键，则设置 'id' 为 'flow_type' + '_' + 'type_cls'
        if "id" not in values:
            values["id"] = values["flow_type"] + "_" + values["type_cls"]
        return values

    def new_alias(self, alias: Optional[List[str]] = None) -> List[str]:
        """Get the new alias id."""
        # 如果 alias 为空，则返回空列表
        if not alias:
            return []
        # 否则，生成新的别名列表，每个别名以 'flow_type' + '_' + 别名元素 形式构成
        return [f"{self.flow_type}_{a}" for a in alias]


def register_resource(
    label: str,
    name: Optional[str] = None,
    category: ResourceCategory = ResourceCategory.COMMON,
    parameters: Optional[List[Parameter]] = None,
    description: Optional[str] = None,
    resource_type: ResourceType = ResourceType.INSTANCE,
    alias: Optional[List[str]] = None,
    **kwargs,
):
    """Register the resource.

    Args:
        label (str): The label of the resource.
        name (Optional[str], optional): The name of the resource. Defaults to None.
        category (str, optional): The category of the resource. Defaults to "common".
        parameters (Optional[List[Parameter]], optional): The parameters of the
            resource. Defaults to None.
        description (Optional[str], optional): The description of the resource.
            Defaults to None.
        resource_type (ResourceType, optional): The type of the resource.
        alias (Optional[List[str]], optional): The alias of the resource. Defaults to
            None. For compatibility, we can use the alias to register the resource.

    """
    # 如果资源类型为 CLASS 并且 parameters 不为空，则引发 ValueError 异常
    if resource_type == ResourceType.CLASS and parameters:
        raise ValueError("Class resource can't have parameters.")
    def decorator(cls):
        """定义一个装饰器函数，用于装饰类。"""

        # 如果没有提供描述信息，使用类的文档字符串作为资源描述
        resource_description = description or cls.__doc__

        # 注册类型名
        type_name = cls.__qualname__
        
        # 获取类对应的类型名称
        type_cls = _get_type_name(cls)
        
        # 获取类的方法解析顺序（Method Resolution Order，MRO）
        mro = inspect.getmro(cls)
        
        # 获取所有父类的类型名称，排除 object 和 abc.ABC
        parent_cls = [
            _get_type_name(parent_cls)
            for parent_cls in mro
            if parent_cls != object and parent_cls != abc.ABC
        ]

        # 创建资源元数据对象
        resource_metadata = ResourceMetadata(
            label=label,                         # 标签
            name=name or type_name,              # 名称，如果未提供则使用类型名
            category=category,                   # 类别
            description=resource_description or label,  # 描述，如果未提供则使用标签
            type_name=type_name,                 # 类型名称
            type_cls=type_cls,                   # 类型类名
            parameters=parameters or [],         # 参数列表，如果未提供则为空列表
            parent_cls=parent_cls,               # 父类类型名称列表
            resource_type=resource_type,         # 资源类型
            **kwargs                             # 其他关键字参数
        )

        # 注册资源别名并获取其 ID
        alias_ids = resource_metadata.new_alias(alias)
        
        # 注册别名类型
        _register_alias_types(cls, alias_ids)
        
        # 注册资源
        _register_resource(cls, resource_metadata, alias_ids)
        
        # 将资源元数据附加到类的属性中
        cls._resource_metadata = resource_metadata
        
        # 返回装饰后的类对象
        return cls

    return decorator
class ViewMetadata(BaseMetadata):
    """The metadata of the operator.

    We use this metadata to build the operator in UI and view the operator in UI.
    """

    operator_type: OperatorType = Field(
        default=OperatorType.MAP,
        description="The type of the operator",
        examples=["map", "reduce"],
    )
    inputs: List[IOField] = Field(..., description="The inputs of the operator")
    outputs: List[IOField] = Field(..., description="The outputs of the operator")
    version: str = Field(
        default="v1", description="The version of the operator", examples=["v1", "v2"]
    )

    type_name: Optional[str] = Field(
        default=None,
        description="The type short name of the operator",
        examples=["LLMOperator"],
    )

    type_cls: Optional[str] = Field(
        default=None,
        description="The type class of the operator",
        examples=["dbgpt.model.operators.LLMOperator"],
    )

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre fill the metadata."""
        # 如果传入的值不是字典类型，则直接返回
        if not isinstance(values, dict):
            return values
        # 如果传入的值中没有 "flow_type" 键，则设置其值为 "operator"
        if "flow_type" not in values:
            values["flow_type"] = "operator"
        # 如果传入的值中没有 "id" 键，则根据特定规则生成一个新的唯一 ID
        if "id" not in values:
            key = cls.get_key(
                values["name"], values["category"], values.get("version", "v1")
            )
            values["id"] = values["flow_type"] + "_" + key
        # 处理输入字段列表，确保每个字段是正确的 IOField 类型
        inputs = values.get("inputs")
        if inputs:
            new_inputs = []
            for field in inputs:
                if isinstance(field, dict):
                    new_inputs.append(IOField(**field))
                elif isinstance(field, IOField):
                    new_inputs.append(field)
                else:
                    raise ValueError("Inputs should be IOField.")

            values["inputs"] = new_inputs
        # 处理输出字段列表，确保每个字段是正确的 IOField 类型
        outputs = values.get("outputs")
        if outputs:
            new_outputs = []
            for field in outputs:
                if isinstance(field, dict):
                    new_outputs.append(IOField(**field))
                elif isinstance(field, IOField):
                    new_outputs.append(field)
                else:
                    raise ValueError("Outputs should be IOField.")
            values["outputs"] = new_outputs
        # 返回处理后的值字典
        return values

    def get_operator_key(self) -> str:
        """Get the operator key."""
        # 如果 flow_type 为空，则抛出异常
        if not self.flow_type:
            raise ValueError("Flow type can't be empty")

        # 根据 name、category 和 version 生成操作符的唯一键
        return (
            self.flow_type + "_" + self.get_key(self.name, self.category, self.version)
        )

    @staticmethod
    def get_key(
        name: str,
        category: Union[str, ResourceCategory, OperatorCategory],
        version: str,
    ) -> str:
        """Static method to generate a key based on name, category, and version."""
        # 返回基于传入参数生成的唯一键
        return f"{name}_{category}_{version}"
    ) -> str:
        """获取操作员ID。"""
        # 定义用于拼接的分隔符字符串
        split_str = "___$$___"
        # 如果category是ResourceCategory或OperatorCategory的实例，则获取其值
        if isinstance(category, (ResourceCategory, OperatorCategory)):
            category = category.value
        # 返回格式化后的操作员ID字符串，格式为"name___$$___category___$$___version"
        return f"{name}{split_str}{category}{split_str}{version}"
class ViewMixin(ABC):
    """The mixin of the operator."""

    metadata: Optional[ViewMetadata] = None  # 类属性，存储视图元数据对象，可选类型为ViewMetadata

    def get_view_metadata(self) -> Optional[ViewMetadata]:
        """Get the view metadata.

        Returns:
            Optional[ViewMetadata]: The view metadata of the instance.
        """
        return self.metadata  # 返回当前实例的视图元数据对象

    @classmethod
    def after_define(cls):
        """After define the operator, register the operator."""
        _register_operator(cls)  # 调用全局函数 _register_operator 注册操作符类

    def to_dict(self) -> Dict:
        """Convert current metadata to json.

        Show the metadata in UI.

        Returns:
            Dict: The metadata dict.
        
        Raises:
            ValueError: If the metadata is not set.
        """
        metadata = self.get_view_metadata()
        if not metadata:
            raise ValueError("Metadata is not set.")
        metadata_dict = metadata.to_dict()  # 调用视图元数据对象的 to_dict 方法，将元数据转换为字典
        return metadata_dict

    @classmethod
    def build_from(
        cls: Type[T],
        view_metadata: ViewMetadata,
        key_to_resource: Optional[Dict[str, "ResourceMetadata"]] = None,
    ) -> T:
        """Build the operator from the metadata."""
        operator_key = view_metadata.get_operator_key()  # 获取视图元数据对象的操作符键
        operator_cls: Type[T] = _get_operator_class(operator_key)  # 根据操作符键获取操作符类
        metadata = operator_cls.metadata
        if not metadata:
            raise ValueError("Metadata is not set.")
        runnable_params = metadata.get_runnable_parameters(  # 获取可运行参数
            view_metadata.parameters, key_to_resource
        )
        operator_task: T = operator_cls(**runnable_params)  # 创建操作符类实例
        return operator_task


@dataclasses.dataclass
class _RegistryItem:
    """The registry item."""

    key: str  # 注册项的键，标识唯一性
    cls: Type  # 注册项关联的类
    metadata: Union[ViewMetadata, ResourceMetadata]  # 注册项关联的元数据，可以是视图元数据或资源元数据的联合类型


class FlowRegistry:
    """The registry of the operator and resource."""

    def __init__(self):
        """Init the registry."""
        self._registry: Dict[str, _RegistryItem] = {}  # 初始化注册表字典

    def register_flow(
        self,
        view_cls: Type,
        metadata: Union[ViewMetadata, ResourceMetadata],
        alias_ids: Optional[List[str]] = None,
    ):
        """Register the operator."""
        key = metadata.id  # 获取元数据的唯一标识作为注册表键
        self._registry[key] = _RegistryItem(key=key, cls=view_cls, metadata=metadata)  # 向注册表中添加注册项
        if alias_ids:
            for alias_id in alias_ids:
                self._registry[alias_id] = _RegistryItem(  # 添加别名对应的注册项
                    key=alias_id, cls=view_cls, metadata=metadata
                )

    def get_registry_item(self, key: str) -> Optional[_RegistryItem]:
        """Get the registry item by the key."""
        return self._registry.get(key)  # 根据键获取注册表中的注册项

    def metadata_list(self):
        """Get the metadata list."""
        return [item.metadata.to_dict() for item in self._registry.values()]  # 返回注册表中所有注册项的元数据字典列表


_OPERATOR_REGISTRY: FlowRegistry = FlowRegistry()  # 创建全局的操作符注册表对象


def _get_operator_class(type_key: str) -> Type[T]:
    """Get the operator class by the type name."""
    item = _OPERATOR_REGISTRY.get_registry_item(type_key)  # 根据类型键从全局操作符注册表中获取注册项
    # 如果item为假值（None或者空），抛出FlowMetadataException异常，说明未找到注册的操作符
    if not item:
        raise FlowMetadataException(f"Operator {type_key} not registered.")
    # 从item中获取cls属性，表示操作符的类
    cls = item.cls
    # 如果cls不是ViewMixin的子类，抛出ValueError异常，说明操作符不符合ViewMixin的要求
    if not issubclass(cls, ViewMixin):
        raise ValueError(f"Operator {type_key} is not a ViewMixin.")
    # 返回类型为T的cls，表示操作符的类
    return cast(Type[T], cls)
# 注册操作符到全局注册表中
def _register_operator(view_cls: Optional[Type[T]]):
    # 如果未提供视图类或者视图类没有元数据，则直接返回
    if not view_cls or not view_cls.metadata:
        return
    # 获取视图类的元数据对象
    metadata = view_cls.metadata
    # 设置元数据的类型名称为视图类的限定名称
    metadata.type_name = view_cls.__qualname__
    # 设置元数据的类型类为视图类对应的类型名称
    metadata.type_cls = _get_type_name(view_cls)
    # 将视图类及其元数据注册到全局操作符注册表中
    _OPERATOR_REGISTRY.register_flow(view_cls, metadata)


# 根据类型名称获取对应的注册项
def _get_resource_class(type_key: str) -> _RegistryItem:
    # 根据类型名称从全局操作符注册表中获取注册项
    item = _OPERATOR_REGISTRY.get_registry_item(type_key)
    # 如果未找到对应注册项，则抛出异常
    if not item:
        raise FlowMetadataException(f"Resource {type_key} not registered.")
    # 如果注册项的元数据不是 ResourceMetadata 类型，则抛出值错误异常
    if not isinstance(item.metadata, ResourceMetadata):
        raise ValueError(f"Resource {type_key} is not a ResourceMetadata.")
    # 返回获取到的注册项
    return item


# 注册资源到全局注册表中
def _register_resource(
    cls: Type,
    resource_metadata: ResourceMetadata,
    alias_ids: Optional[List[str]] = None,
):
    # 将指定类及其资源元数据以及可选的别名 ID 注册到全局操作符注册表中
    _OPERATOR_REGISTRY.register_flow(cls, resource_metadata, alias_ids)
```