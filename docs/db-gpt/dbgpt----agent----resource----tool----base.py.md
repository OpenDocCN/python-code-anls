# `.\DB-GPT-src\dbgpt\agent\resource\tool\base.py`

```py
"""Tool resources."""

# 引入异步操作模块 asyncio
import asyncio
# 引入 dataclasses 用于数据类的定义
import dataclasses
# 引入 functools 用于高阶函数的操作
import functools
# 引入 inspect 用于获取对象信息
import inspect
# 引入 json 用于 JSON 数据处理
import json
# 引入 ABC 抽象基类
from abc import ABC, abstractmethod
# 引入类型提示模块
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, Union, cast

# 引入 Pydantic 的基础模型和字段定义
from dbgpt._private.pydantic import BaseModel, Field, model_validator
# 引入配置模块中的常量定义
from dbgpt.util.configure.base import _MISSING, _MISSING_TYPE
# 引入功能工具模块中的函数
from dbgpt.util.function_utils import parse_param_description, type_to_string

# 引入相对路径下的 Resource 类和参数类型定义
from ..base import Resource, ResourceParameters, ResourceType

# 定义工具函数类型为可调用对象的联合类型
ToolFunc = Union[Callable[..., Any], Callable[..., Awaitable[Any]]]

# 定义工具标识符常量
DB_GPT_TOOL_IDENTIFIER = "dbgpt_tool"


@dataclasses.dataclass
class ToolResourceParameters(ResourceParameters):
    """Tool resource parameters class."""
    # 工具资源参数的数据类

    pass  # 空白


class ToolParameter(BaseModel):
    """Parameter for a tool."""
    # 工具参数模型

    name: str = Field(..., description="Parameter name")  # 参数名
    title: str = Field(
        ...,
        description="Parameter title, default to the name with the first letter "
        "capitalized",
    )  # 参数标题，如果未指定则默认为大写首字母的名称
    type: str = Field(..., description="Parameter type", examples=["string", "integer"])  # 参数类型，示例包括字符串和整数
    description: str = Field(..., description="Parameter description")  # 参数描述
    required: bool = Field(True, description="Whether the parameter is required")  # 参数是否必需
    default: Optional[Any] = Field(
        _MISSING, description="Default value for the parameter"
    )  # 参数的默认值，如果未提供则为缺失值

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values):
        """Pre-fill the model."""
        # 模型预填充方法，根据需要填充缺失的字段
        if not isinstance(values, dict):
            return values
        if "title" not in values:
            values["title"] = values["name"].replace("_", " ").title()  # 根据名称生成标题
        if "description" not in values:
            values["description"] = values["title"]  # 根据标题生成描述
        return values


class BaseTool(Resource[ToolResourceParameters], ABC):
    """Base class for a tool."""
    # 工具的基类

    @classmethod
    def type(cls) -> ResourceType:
        """Return the resource type."""
        return ResourceType.Tool  # 返回资源类型为工具

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the tool."""
        # 抽象属性，返回工具的描述信息

    @property
    @abstractmethod
    def args(self) -> Dict[str, ToolParameter]:
        """Return the arguments of the tool."""
        # 抽象属性，返回工具的参数字典

    async def get_prompt(
        self,
        *,
        lang: str = "en",
        prompt_type: str = "default",
        question: Optional[str] = None,
        resource_name: Optional[str] = None,
        **kwargs,
        # 异步方法，获取工具的提示信息
    ):
        """获取提示信息。"""
        # 英文版提示模板
        prompt_template = (
            "{name}: Call this tool to interact with the {name} API. "
            "What is the {name} API useful for? {description} "
            "Parameters: {parameters}"
        )
        # 中文版提示模板
        prompt_template_zh = (
            "{name}：调用此工具与 {name} API进行交互。{name} API 有什么用？{description} "
            "参数：{parameters}"
        )
        # 根据语言选择合适的提示模板
        template = prompt_template if lang == "en" else prompt_template_zh
        # 如果是 OpenAI 的类型
        if prompt_type == "openai":
            # 初始化属性字典和必需属性列表
            properties = {}
            required_list = []
            # 遍历参数字典中的键值对
            for key, value in self.args.items():
                # 将每个参数的类型和描述添加到属性字典中
                properties[key] = {
                    "type": value.type,
                    "description": value.description,
                }
                # 如果参数为必需，则将其键添加到必需属性列表中
                if value.required:
                    required_list.append(key)
            # 构建参数字典
            parameters_dict = {
                "type": "object",
                "properties": properties,
                "required": required_list,
            }
            # 将参数字典转换为 JSON 格式的字符串
            parameters_string = json.dumps(parameters_dict, ensure_ascii=False)
        else:
            # 初始化参数列表
            parameters = []
            # 遍历参数字典中的键值对
            for key, value in self.args.items():
                # 将每个参数的名称、类型、描述和是否必需添加到参数列表中
                parameters.append(
                    {
                        "name": key,
                        "type": value.type,
                        "description": value.description,
                        "required": value.required,
                    }
                )
            # 将参数列表转换为 JSON 格式的字符串
            parameters_string = json.dumps(parameters, ensure_ascii=False)
        # 使用选定的模板格式化并返回提示信息
        return template.format(
            name=self.name,
            description=self.description,
            parameters=parameters_string,
        )

    def __str__(self) -> str:
        """返回工具的字符串表示形式。"""
        # 返回工具的字符串表示形式，包括名称和描述
        return f"Tool: {self.name} ({self.description})"
# 定义一个工具类，继承自BaseTool，用于包装一个函数作为工具
class FunctionTool(BaseTool):
    """Function tool.

    Wrap a function as a tool.
    """

    def __init__(
        self,
        name: str,
        func: ToolFunc,
        description: Optional[str] = None,
        args: Optional[Dict[str, Union[ToolParameter, Dict[str, Any]]]] = None,
        args_schema: Optional[Type[BaseModel]] = None,
    ):
        """Create a tool from a function."""
        # 如果未提供描述，尝试从函数的文档字符串中解析
        if not description:
            description = _parse_docstring(func)
        # 如果仍未提供描述，则抛出值错误
        if not description:
            raise ValueError("The description is required")
        # 初始化工具名称、描述、参数字典和函数对象
        self._name = name
        self._description = cast(str, description)
        self._args: Dict[str, ToolParameter] = _parse_args(func, args, args_schema)
        self._func = func
        # 判断函数是否为异步函数
        self._is_async = asyncio.iscoroutinefunction(func)

    @property
    def name(self) -> str:
        """Return the name of the tool."""
        # 返回工具的名称
        return self._name

    @property
    def description(self) -> str:
        """Return the description of the tool."""
        # 返回工具的描述
        return self._description

    @property
    def args(self) -> Dict[str, ToolParameter]:
        """Return the arguments of the tool."""
        # 返回工具的参数字典
        return self._args

    @property
    def is_async(self) -> bool:
        """Return whether the tool is asynchronous."""
        # 返回工具是否为异步执行
        return self._is_async

    def execute(
        self,
        *args,
        resource_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute the tool.

        Args:
            *args: The positional arguments.
            resource_name (str, optional): The tool name to be executed (not used for specific tool).
            **kwargs: The keyword arguments.
        """
        # 如果工具是异步的，抛出值错误
        if self._is_async:
            raise ValueError("The function is asynchronous")
        # 执行工具函数，并返回结果
        return self._func(*args, **kwargs)

    async def async_execute(
        self,
        *args,
        resource_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute the tool asynchronously.

        Args:
            *args: The positional arguments.
            resource_name (str, optional): The tool name to be executed (not used for specific tool).
            **kwargs: The keyword arguments.
        """
        # 如果工具不是异步的，抛出值错误
        if not self._is_async:
            raise ValueError("The function is synchronous")
        # 异步执行工具函数，并返回结果
        return await self._func(*args, **kwargs)


def tool(
    *decorator_args: Union[str, Callable],
    description: Optional[str] = None,
    args: Optional[Dict[str, Union[ToolParameter, Dict[str, Any]]]] = None,
    args_schema: Optional[Type[BaseModel]] = None,
) -> Callable[..., Any]:
    """Create a tool from a function."""
    def _create_decorator(name: str):
        # 定义一个内部函数 decorator，接受一个 ToolFunc 类型的函数作为参数
        def decorator(func: ToolFunc):
            # 确定工具的名称，如果未提供则使用函数的名称
            tool_name = name or func.__name__
            # 创建 FunctionTool 对象，用于执行功能函数
            ft = FunctionTool(tool_name, func, description, args, args_schema)

            # 同步执行的包装器函数
            @functools.wraps(func)
            def sync_wrapper(*f_args, **kwargs):
                return ft.execute(*f_args, **kwargs)

            # 异步执行的包装器函数
            @functools.wraps(func)
            async def async_wrapper(*f_args, **kwargs):
                return await ft.async_execute(*f_args, **kwargs)

            # 根据函数是否为异步函数选择合适的包装器
            if asyncio.iscoroutinefunction(func):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
            # 将 FunctionTool 对象保存在包装器的 _tool 属性中
            wrapper._tool = ft  # type: ignore
            # 设置属性 DB_GPT_TOOL_IDENTIFIER 为 True，用于标记该函数为工具函数
            setattr(wrapper, DB_GPT_TOOL_IDENTIFIER, True)
            return wrapper

        return decorator

    # 根据 decorator_args 的长度和类型，选择不同的装饰器创建方式
    if len(decorator_args) == 1 and callable(decorator_args[0]):
        # 处理 @tool 装饰器，获取原始函数并应用默认的装饰器
        old_func = decorator_args[0]
        return _create_decorator(old_func.__name__)(old_func)
    elif len(decorator_args) == 1 and isinstance(decorator_args[0], str):
        # 处理 @tool("google_search") 装饰器，使用指定的名称创建装饰器
        return _create_decorator(decorator_args[0])
    elif (
        len(decorator_args) == 2
        and isinstance(decorator_args[0], str)
        and callable(decorator_args[1])
    ):
        # 处理 @tool("google_search", description="Search on Google") 装饰器，使用指定的名称和描述创建装饰器
        return _create_decorator(decorator_args[0])(decorator_args[1])
    elif len(decorator_args) == 0:
        # 处理未传递参数的情况，使用函数名作为工具名称创建装饰器
        def _partial(func: ToolFunc):
            return _create_decorator(func.__name__)(func)

        return _partial
    else:
        # 如果参数不符合预期，抛出 ValueError 异常
        raise ValueError("Invalid usage of @tool")
# 解析函数的文档字符串
def _parse_docstring(func: ToolFunc) -> str:
    """Parse the docstring of the function."""
    docstring = func.__doc__
    if docstring is None:
        return ""
    return docstring.strip()


# 解析函数的参数
def _parse_args(
    func: ToolFunc,
    args: Optional[Dict[str, Union[ToolParameter, Dict[str, Any]]]] = None,
    args_schema: Optional[Type[BaseModel]] = None,
) -> Dict[str, ToolParameter]:
    """Parse the arguments of the function."""
    parsed_args = {}

    # 检查是否所有的参数值都是 ToolParameter 类型
    if args is not None:
        if all(isinstance(v, ToolParameter) for v in args.values()):
            return args  # type: ignore
        # 检查是否所有的参数值都是 dict 类型
        if all(isinstance(v, dict) for v in args.values()):
            for k, v in args.items():
                param_name = v.get("name", k)
                param_title = v.get("title", param_name.replace("_", " ").title())
                param_type = v["type"]
                param_description = v.get("description", param_title)
                param_default = v.get("default", _MISSING)
                param_required = v.get("required", param_default is _MISSING)
                parsed_args[k] = ToolParameter(
                    name=param_name,
                    title=param_title,
                    type=param_type,
                    description=param_description,
                    default=param_default,
                    required=param_required,
                )
            return parsed_args
        # 如果参数类型不符合预期，抛出 ValueError
        raise ValueError("args should be a dict of ToolParameter or dict")

    # 如果提供了参数模式，从模式中解析参数
    if args_schema is not None:
        return _parse_args_from_schema(args_schema)

    # 获取函数的参数签名
    signature = inspect.signature(func)

    # 遍历函数的每个参数
    for param in signature.parameters.values():
        real_type = param.annotation
        param_name = param.name
        param_title = param_name.replace("_", " ").title()

        # 判断参数是否有默认值
        if param.default is not inspect.Parameter.empty:
            param_default = param.default
            param_required = False
        else:
            param_default = _MISSING
            param_required = True

        # 转换参数类型为字符串
        param_type = type_to_string(real_type, "unknown")

        # 解析参数的描述
        param_description = parse_param_description(param_name, real_type)

        # 构建 ToolParameter 对象并添加到 parsed_args 字典中
        parsed_args[param_name] = ToolParameter(
            name=param_name,
            title=param_title,
            type=param_type,
            description=param_description,
            default=param_default,
            required=param_required,
        )

    return parsed_args


# 从 Pydantic 模式中解析参数
def _parse_args_from_schema(args_schema: Type[BaseModel]) -> Dict[str, ToolParameter]:
    """Parse the arguments from a Pydantic schema."""
    pydantic_args = args_schema.schema()["properties"]
    parsed_args = {}
    # 遍历 pydantic_args 字典中的键值对
    for key, value in pydantic_args.items():
        # 将参数名设置为当前键值对的键
        param_name = key
        # 获取参数的标题，如果不存在则将参数名转换为标题格式
        param_title = value.get("title", param_name.replace("_", " ").title())
        # 检查参数定义中是否包含类型信息
        if "type" in value:
            # 如果存在类型信息，则将其赋给 param_type
            param_type = value["type"]
        elif "anyOf" in value:
            # 如果参数定义包含 anyOf 类型，解析其中的类型信息
            any_of: List[Dict[str, Any]] = value["anyOf"]
            # 如果 anyOf 中有一个类型为 "null"，则选择不为 "null" 的类型作为 param_type
            if len(any_of) == 2 and any("null" in t["type"] for t in any_of):
                param_type = next(t["type"] for t in any_of if "null" not in t["type"])
            else:
                # 否则将整个 anyOf 定义转换为 JSON 字符串
                param_type = json.dumps({"anyOf": value["anyOf"]}, ensure_ascii=False)
        else:
            # 如果参数定义既没有 "type" 也没有 "anyOf"，则抛出错误
            raise ValueError(f"Invalid schema for {key}")
        # 获取参数的描述，如果不存在则使用参数的标题
        param_description = value.get("description", param_title)
        # 获取参数的默认值，如果不存在则使用 _MISSING
        param_default = value.get("default", _MISSING)
        # 根据默认值是否为 _MISSING，确定参数是否为必需的
        param_required = False
        if isinstance(param_default, _MISSING_TYPE) and param_default == _MISSING:
            param_required = True

        # 将解析后的参数信息存储到 parsed_args 字典中
        parsed_args[key] = ToolParameter(
            name=param_name,
            title=param_title,
            type=param_type,
            description=param_description,
            default=param_default,
            required=param_required,
        )
    # 返回解析后的参数字典
    return parsed_args
```