# `.\DB-GPT-src\dbgpt\agent\core\action\base.py`

```py
# 导入必要的模块和类
import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

# 导入用于模型操作的私有模块
from dbgpt._private.pydantic import (
    BaseModel,                    # 导入基本模型类
    field_default,                # 导入字段默认值设置函数
    field_description,            # 导入字段描述设置函数
    model_fields,                 # 导入模型字段获取函数
    model_to_dict,                # 导入模型转换为字典的函数
    model_validator,              # 导入模型验证器装饰器
)
# 导入 JSON 相关的工具函数
from dbgpt.util.json_utils import find_json_objects
# 导入可视化相关模块
from dbgpt.vis.base import Vis

# 导入资源相关类和枚举
from ...resource.base import AgentResource, Resource, ResourceType

# 定义类型变量 T，约束为 Union[BaseModel, List[BaseModel], None]
T = TypeVar("T", bound=Union[BaseModel, List[BaseModel], None])

# 定义 JsonMessageType 类型别名，可以是单个字典或字典列表
JsonMessageType = Union[Dict[str, Any], List[Dict[str, Any]]]


class ActionOutput(BaseModel):
    """动作输出模型类。"""

    content: str                    # 动作输出的内容，字符串类型
    is_exe_success: bool = True     # 是否执行成功的布尔值，默认为 True
    view: Optional[str] = None      # 可选的视图字符串
    resource_type: Optional[str] = None  # 可选的资源类型字符串
    resource_value: Optional[Any] = None  # 可选的资源值
    action: Optional[str] = None    # 可选的动作字符串
    thoughts: Optional[str] = None  # 可选的思考字符串
    observations: Optional[str] = None  # 可选的观察字符串

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Any) -> Any:
        """在填充值之前进行预处理。"""
        if not isinstance(values, dict):
            return values
        is_exe_success = values.get("is_exe_success", True)
        if not is_exe_success and "observations" not in values:
            values["observations"] = values.get("content")
        return values

    @classmethod
    def from_dict(
        cls: Type["ActionOutput"], param: Optional[Dict]
    ) -> Optional["ActionOutput"]:
        """从字典转换为 ActionOutput 对象。"""
        if not param:
            return None
        return cls.parse_obj(param)

    def to_dict(self) -> Dict[str, Any]:
        """将对象转换为字典。"""
        return model_to_dict(self)


class Action(ABC, Generic[T]):
    """定义代理动作的基类。"""

    def __init__(self):
        """创建一个动作。"""
        self.resource: Optional[Resource] = None  # 资源对象的可选属性，默认为 None

    def init_resource(self, resource: Optional[Resource]):
        """初始化资源对象。"""
        self.resource = resource

    @property
    def resource_need(self) -> Optional[ResourceType]:
        """返回动作所需的资源类型。"""
        return None  # 默认情况下不需要任何特定的资源类型

    @property
    def render_protocol(self) -> Optional[Vis]:
        """返回渲染协议。"""
        return None  # 默认情况下不需要任何渲染协议

    def render_prompt(self) -> Optional[str]:
        """返回渲染提示信息。"""
        if self.render_protocol is None:
            return None
        else:
            return self.render_protocol.render_prompt()

    def _create_example(
        self,
        model_type: Union[Type[BaseModel], List[Type[BaseModel]]],
        # _create_example 方法，接受一个模型类型或模型类型列表作为参数
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        # 如果模型类型为 None，则返回 None
        if model_type is None:
            return None
        # 获取模型类型的原始类型和参数类型
        origin = get_origin(model_type)
        args = get_args(model_type)
        # 如果模型类型没有原始类型，创建一个空字典作为示例
        if origin is None:
            example = {}
            # 将模型类型强制转换为单一模型类型
            single_model_type = cast(Type[BaseModel], model_type)
            # 遍历单一模型类型的所有字段，并生成字段的描述或默认值
            for field_name, field in model_fields(single_model_type).items():
                description = field_description(field)
                default_value = field_default(field)
                # 根据字段的描述或默认值填充示例字典
                if description:
                    example[field_name] = description
                elif default_value:
                    example[field_name] = default_value
                else:
                    example[field_name] = ""
            return example
        # 如果原始类型是列表或 List，则处理列表类型的模型
        elif origin is list or origin is List:
            element_type = cast(Type[BaseModel], args[0])
            # 检查列表元素类型是否是 BaseModel 的子类
            if issubclass(element_type, BaseModel):
                # 生成列表元素类型的示例，并转换为字典格式
                list_example = self._create_example(element_type)
                typed_list_example = cast(Dict[str, Any], list_example)
                return [typed_list_example]
            else:
                # 如果列表元素类型不是 BaseModel 的子类，则抛出错误
                raise TypeError("List elements must be BaseModel subclasses")
        else:
            # 如果模型类型不符合预期，则抛出值错误
            raise ValueError(
                f"Model type {model_type} is not an instance of BaseModel."
            )

    @property
    def out_model_type(self) -> Optional[Union[Type[T], List[Type[T]]]]:
        """Return the output model type."""
        # 返回输出模型类型，这里始终返回 None
        return None

    @property
    def ai_out_schema(self) -> Optional[str]:
        """Return the AI output schema."""
        # 如果输出模型类型为 None，则返回 None
        if self.out_model_type is None:
            return None

        # 创建输出模型类型的示例，并转换为 JSON 格式的字符串
        json_format_data = json.dumps(
            self._create_example(self.out_model_type), indent=2, ensure_ascii=False
        )
        # 返回带有 JSON 示例格式的说明文本
        return f"""Please response in the following json format:
            {json_format_data}
        Make sure the response is correct json and can be parsed by Python json.loads.
        """

    def _ai_message_2_json(self, ai_message: str) -> JsonMessageType:
        # 从 AI 消息中查找 JSON 对象
        json_objects = find_json_objects(ai_message)
        json_count = len(json_objects)
        # 如果找到的 JSON 对象数量不为 1，则抛出值错误
        if json_count != 1:
            raise ValueError("Unable to obtain valid output.")
        # 返回找到的第一个 JSON 对象
        return json_objects[0]

    def _input_convert(self, ai_message: str, cls: Type[T]) -> T:
        # 将 AI 消息转换为 JSON 对象
        json_result = self._ai_message_2_json(ai_message)
        # 如果目标类型是列表，则处理列表元素类型
        if get_origin(cls) == list:
            inner_type = get_args(cls)[0]
            typed_cls = cast(Type[BaseModel], inner_type)
            # 将每个 JSON 结果项转换为目标列表元素类型的实例
            return [typed_cls.parse_obj(item) for item in json_result]  # type: ignore
        else:
            # 否则，将 JSON 结果转换为目标类型的实例
            typed_cls = cast(Type[BaseModel], cls)
            return typed_cls.parse_obj(json_result)

    @abstractmethod
    async def run(
        self,
        ai_message: str,
        resource: Optional[AgentResource] = None,
        rely_action_out: Optional[ActionOutput] = None,
        need_vis_render: bool = True,
        **kwargs,
        ):
        # 抽象方法：运行 AI 模型
        # 参数说明详见方法定义处
        pass
    ) -> ActionOutput:
        """定义一个方法，该方法接收参数并返回 ActionOutput 类型的结果。"""
```