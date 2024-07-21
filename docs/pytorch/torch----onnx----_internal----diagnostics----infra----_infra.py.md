# `.\pytorch\torch\onnx\_internal\diagnostics\infra\_infra.py`

```py
# mypy: allow-untyped-defs
"""This file defines an additional layer of abstraction on top of the SARIF OM."""

from __future__ import annotations

import dataclasses  # 导入 dataclasses 模块，用于支持数据类的定义
import enum  # 导入 enum 模块，用于定义枚举类型
import logging  # 导入 logging 模块，用于记录日志信息
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple  # 导入类型提示相关的类型定义

from torch.onnx._internal.diagnostics.infra import formatter, sarif  # 导入其他模块中的指定子模块


class Level(enum.IntEnum):
    """The level of a diagnostic.

    This class is used to represent the level of a diagnostic. The levels are defined
    by the SARIF specification, and are not modifiable. For alternative categories,
    please use infra.Tag instead. When selecting a level, please consider the following
    guidelines:

    - NONE: Informational result that does not indicate the presence of a problem.
    - NOTE: An opportunity for improvement was found.
    - WARNING: A potential problem was found.
    - ERROR: A serious problem was found.

    This level is a subclass of enum.IntEnum, and can be used as an integer. Its integer
    value maps to the logging levels in Python's logging module. The mapping is as
    follows:

        Level.NONE = logging.DEBUG = 10
        Level.NOTE = logging.INFO = 20
        Level.WARNING = logging.WARNING = 30
        Level.ERROR = logging.ERROR = 40
    """
    
    NONE = 10  # 定义枚举成员 NONE，对应的整数值为 10
    NOTE = 20  # 定义枚举成员 NOTE，对应的整数值为 20
    WARNING = 30  # 定义枚举成员 WARNING，对应的整数值为 30
    ERROR = 40  # 定义枚举成员 ERROR，对应的整数值为 40


levels = Level  # 将 Level 枚举类赋值给 levels 变量，方便引用


class Tag(enum.Enum):
    """The tag of a diagnostic. This class can be inherited to define custom tags."""


class PatchedPropertyBag(sarif.PropertyBag):
    """Key/value pairs that provide additional information about the object.

    The definition of PropertyBag via SARIF spec is "A property bag is an object (section 3.6)
    containing an unordered set of properties with arbitrary names." However it is not
    reflected in the json file, and therefore not captured by the python representation.
    This patch adds additional **kwargs to the `__init__` method to allow recording
    arbitrary key/value pairs.
    """

    def __init__(self, tags: Optional[List[str]] = None, **kwargs):
        super().__init__(tags=tags)  # 调用父类 sarif.PropertyBag 的 __init__ 方法，初始化 tags
        self.__dict__.update(kwargs)  # 更新对象的属性字典，允许传入任意关键字参数


@dataclasses.dataclass(frozen=True)
class Rule:
    id: str  # 规则的唯一标识符，字符串类型
    name: str  # 规则的名称，字符串类型
    message_default_template: str  # 规则的默认消息模板，字符串类型
    short_description: Optional[str] = None  # 规则的简短描述，可选的字符串类型
    full_description: Optional[str] = None  # 规则的完整描述，可选的字符串类型
    full_description_markdown: Optional[str] = None  # 规则的完整描述的 Markdown 格式，可选的字符串类型
    help_uri: Optional[str] = None  # 规则的帮助文档链接，可选的字符串类型

    @classmethod
    def from_sarif(cls, **kwargs):
        """Returns a rule from the SARIF reporting descriptor."""
        # 从 SARIF 报告描述符中获取短描述文本
        short_description = kwargs.get("short_description", {}).get("text")
        # 从 SARIF 报告描述符中获取完整描述文本
        full_description = kwargs.get("full_description", {}).get("text")
        # 从 SARIF 报告描述符中获取完整描述的 Markdown 格式文本
        full_description_markdown = kwargs.get("full_description", {}).get("markdown")
        # 获取帮助 URI
        help_uri = kwargs.get("help_uri")

        # 创建 Rule 对象，使用传入的参数
        rule = cls(
            id=kwargs["id"],
            name=kwargs["name"],
            message_default_template=kwargs["message_strings"]["default"]["text"],
            short_description=short_description,
            full_description=full_description,
            full_description_markdown=full_description_markdown,
            help_uri=help_uri,
        )
        return rule

    def sarif(self) -> sarif.ReportingDescriptor:
        """Returns a SARIF reporting descriptor of this Rule."""
        # 根据对象的短描述生成 MultiformatMessageString 对象，如果没有短描述则为 None
        short_description = (
            sarif.MultiformatMessageString(text=self.short_description)
            if self.short_description is not None
            else None
        )
        # 根据对象的完整描述和 Markdown 格式生成 MultiformatMessageString 对象，如果没有则为 None
        full_description = (
            sarif.MultiformatMessageString(
                text=self.full_description, markdown=self.full_description_markdown
            )
            if self.full_description is not None
            else None
        )
        # 创建并返回 SARIF 的 ReportingDescriptor 对象
        return sarif.ReportingDescriptor(
            id=self.id,
            name=self.name,
            short_description=short_description,
            full_description=full_description,
            help_uri=self.help_uri,
        )

    def format(self, level: Level, *args, **kwargs) -> Tuple[Rule, Level, str]:
        """Returns a tuple of (rule, level, message) for a diagnostic.

        This method is used to format the message of a diagnostic. The message is
        formatted using the default template of this rule, and the arguments passed in
        as `*args` and `**kwargs`. The level is used to override the default level of
        this rule.
        """
        # 返回由规则、级别和格式化的消息组成的元组，用于诊断
        return (self, level, self.format_message(*args, **kwargs))

    def format_message(self, *args, **kwargs) -> str:
        """Returns the formatted default message of this Rule.

        This method should be overridden (with code generation) by subclasses to reflect
        the exact arguments needed by the message template. This is a helper method to
        create the default message for a diagnostic.
        """
        # 使用默认模板格式化规则的默认消息，并返回结果
        return self.message_default_template.format(*args, **kwargs)
@dataclasses.dataclass
class Location:
    uri: Optional[str] = None  # 资源标识符（可选）
    line: Optional[int] = None  # 行号（可选）
    message: Optional[str] = None  # 消息（可选）
    start_column: Optional[int] = None  # 起始列（可选）
    end_column: Optional[int] = None  # 结束列（可选）
    snippet: Optional[str] = None  # 代码片段（可选）
    function: Optional[str] = None  # 函数名（可选）

    def sarif(self) -> sarif.Location:
        """Returns the SARIF representation of this location."""
        return sarif.Location(
            physical_location=sarif.PhysicalLocation(
                artifact_location=sarif.ArtifactLocation(uri=self.uri),
                region=sarif.Region(
                    start_line=self.line,
                    start_column=self.start_column,
                    end_column=self.end_column,
                    snippet=sarif.ArtifactContent(text=self.snippet),
                ),
            ),
            message=sarif.Message(text=self.message)
            if self.message is not None
            else None,
        )


@dataclasses.dataclass
class StackFrame:
    location: Location  # 堆栈帧的位置信息

    def sarif(self) -> sarif.StackFrame:
        """Returns the SARIF representation of this stack frame."""
        return sarif.StackFrame(location=self.location.sarif())


@dataclasses.dataclass
class Stack:
    """Records a stack trace. The frames are in order from newest to oldest stack frame."""

    frames: List[StackFrame] = dataclasses.field(default_factory=list)  # 堆栈帧列表
    message: Optional[str] = None  # 消息（可选）

    def sarif(self) -> sarif.Stack:
        """Returns the SARIF representation of this stack."""
        return sarif.Stack(
            frames=[frame.sarif() for frame in self.frames],  # 转换每个堆栈帧为 SARIF 格式
            message=sarif.Message(text=self.message)
            if self.message is not None
            else None,
        )


@dataclasses.dataclass
class ThreadFlowLocation:
    """Records code location and the initial state."""

    location: Location  # 线程流的位置信息
    state: Mapping[str, str]  # 初始状态的映射
    index: int  # 索引
    stack: Optional[Stack] = None  # 堆栈信息（可选）

    def sarif(self) -> sarif.ThreadFlowLocation:
        """Returns the SARIF representation of this thread flow location."""
        return sarif.ThreadFlowLocation(
            location=self.location.sarif(),  # 转换位置信息为 SARIF 格式
            state=self.state,  # 使用给定的状态信息
            stack=self.stack.sarif() if self.stack is not None else None,  # 转换堆栈信息为 SARIF 格式（如果存在）
        )


@dataclasses.dataclass
class Graph:
    """A graph of diagnostics.

    This class stores the string representation of a model graph.
    The `nodes` and `edges` fields are unused in the current implementation.
    """

    graph: str  # 图的字符串表示
    name: str  # 图的名称
    description: Optional[str] = None  # 描述（可选）

    def sarif(self) -> sarif.Graph:
        """Returns the SARIF representation of this graph."""
        return sarif.Graph(
            description=sarif.Message(text=self.graph),  # 使用图的字符串表示作为描述
            properties=PatchedPropertyBag(name=self.name, description=self.description),  # 使用名称和描述创建属性包
        )


@dataclasses.dataclass
class RuleCollection:
    _rule_id_name_set: FrozenSet[Tuple[str, str]] = dataclasses.field(init=False)  # 规则ID和名称的不可变集合
    # 初始化方法，在对象创建后立即调用，初始化规则 ID 和名称的不可变集合
    def __post_init__(self) -> None:
        # 使用数据类中默认值为 Rule 类型的字段，创建规则 ID 和名称的不可变集合
        self._rule_id_name_set = frozenset(
            {
                (field.default.id, field.default.name)
                for field in dataclasses.fields(self)
                if isinstance(field.default, Rule)
            }
        )
    
    # 魔法方法，用于判断给定规则是否存在于集合中
    def __contains__(self, rule: Rule) -> bool:
        """Checks if the rule is in the collection."""
        # 检查给定规则的 ID 和名称是否存在于规则 ID 和名称的不可变集合中
        return (rule.id, rule.name) in self._rule_id_name_set
    
    # 类方法，根据给定规则列表创建自定义的规则集合类
    @classmethod
    def custom_collection_from_list(
        cls, new_collection_class_name: str, rules: Sequence[Rule]
    ) -> RuleCollection:
        """Creates a custom class inherited from RuleCollection with the list of rules."""
        # 使用给定的规则列表创建一个新的数据类，并返回相应的规则集合对象
        return dataclasses.make_dataclass(
            new_collection_class_name,
            [
                (
                    formatter.kebab_case_to_snake_case(rule.name),  # 将规则名称转换为蛇形命名格式
                    type(rule),  # 规则对象的类型
                    dataclasses.field(default=rule),  # 设置字段的默认值为给定规则对象
                )
                for rule in rules
            ],
            bases=(cls,),  # 设置新类的基类为当前类
        )()
class Invocation:
    # TODO: Implement this.
    # Tracks top level call arguments and diagnostic options.

    def __init__(self) -> None:
        # 构造函数，抛出未实现错误，提示需要实现该方法
        raise NotImplementedError


@dataclasses.dataclass
class DiagnosticOptions:
    """Options for diagnostic context.

    Attributes:
        verbosity_level: Set the amount of information logged for each diagnostics,
            equivalent to the 'level' in Python logging module.
        warnings_as_errors: When True, warning diagnostics are treated as error diagnostics.
    """

    verbosity_level: int = dataclasses.field(default=logging.INFO)
    """Set the amount of information logged for each diagnostics, equivalent to the 'level' in Python logging module."""

    warnings_as_errors: bool = dataclasses.field(default=False)
    """If True, warning diagnostics are treated as error diagnostics."""
```