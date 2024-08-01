# `.\DB-GPT-src\dbgpt\core\interface\prompt.py`

```py
"""The prompt template interface."""

from __future__ import annotations  # 导入用于支持类型注解的特性

import dataclasses  # 导入用于数据类的模块
import json  # 导入 JSON 模块
from abc import ABC, abstractmethod  # 导入 ABC 抽象基类及其装饰器
from string import Formatter  # 导入字符串格式化工具类 Formatter
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union  # 导入类型注解相关类

from dbgpt._private.pydantic import BaseModel, ConfigDict, model_validator  # 导入 Pydantic 相关类
from dbgpt.core.interface.message import BaseMessage, HumanMessage, SystemMessage  # 导入消息相关类
from dbgpt.core.interface.storage import (  # 导入存储接口相关类
    InMemoryStorage,
    QuerySpec,
    ResourceIdentifier,
    StorageInterface,
    StorageItem,
)
from dbgpt.util.formatting import formatter, no_strict_formatter  # 导入格式化工具类及相关函数

T = TypeVar("T", bound="BasePromptTemplate")  # 定义类型变量 T，绑定于 BasePromptTemplate 类


def _jinja2_formatter(template: str, **kwargs: Any) -> str:
    """Format a template using jinja2."""
    try:
        from jinja2 import Template  # 尝试导入 jinja2 模板类
    except ImportError:
        raise ImportError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )

    return Template(template).render(**kwargs)  # 使用 jinja2 模板类渲染给定模板


_DEFAULT_FORMATTER_MAPPING: Dict[str, Callable] = {
    "f-string": lambda is_strict: (
        formatter.format if is_strict else no_strict_formatter.format
    ),  # 根据 is_strict 参数选择格式化函数
    "jinja2": lambda is_strict: _jinja2_formatter,  # 使用 jinja2 格式化函数
}


class BasePromptTemplate(BaseModel):
    """Base class for all prompt templates, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""


class PromptTemplate(BasePromptTemplate):
    """Prompt template."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义模型配置字典

    template: str
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    response_key: str = "response"

    template_is_strict: bool = True
    """strict template will check template args"""

    response_format: Optional[str] = None

    template_scene: Optional[str] = None

    template_define: Optional[str] = None
    """this template define"""

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "prompt"

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs."""
        if self.response_format:
            kwargs[self.response_key] = json.dumps(
                self.response_format, ensure_ascii=False, indent=4
            )  # 如果定义了 response_format，则将其转换为 JSON 字符串并存入 kwargs
        return _DEFAULT_FORMATTER_MAPPING[self.template_format](
            self.template_is_strict
        )(self.template, **kwargs)  # 使用适当的格式化函数对模板进行格式化

    @classmethod
    def from_template(
        cls: Type[T], template: str, template_format: str = "f-string", **kwargs: Any
    ) -> T:
        """定义一个方法，返回类型为 T。

        从模板字符串创建一个提示模板。
        """
        # 使用指定的模板和格式获取模板变量
        input_variables = get_template_vars(template, template_format)
        # 返回一个类的实例，包括模板字符串、输入变量、模板格式以及其他关键字参数
        return cls(
            template=template,
            input_variables=input_variables,
            template_format=template_format,
            **kwargs,
        )
class BaseChatPromptTemplate(BaseModel, ABC):
    """The base chat prompt template."""

    prompt: BasePromptTemplate

    @property
    def input_variables(self) -> List[str]:
        """Return a list of the names of the variables the prompt template expects."""
        return self.prompt.input_variables

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the prompt with the inputs."""

    @classmethod
    def from_template(
        cls: Type[T],
        template: str,
        template_format: str = "f-string",
        response_format: Optional[str] = None,
        response_key: str = "response",
        template_is_strict: bool = True,
        **kwargs: Any,
    ) -> T:
        """Create a prompt template from a template string."""
        prompt = PromptTemplate.from_template(
            template,
            template_format,
            response_format=response_format,
            response_key=response_key,
            template_is_strict=template_is_strict,
        )
        return cls(prompt=prompt, **kwargs)


class SystemPromptTemplate(BaseChatPromptTemplate):
    """The system prompt template."""

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the prompt with the inputs.

        Returns:
            List[BaseMessage]: The formatted messages.
        """
        content = self.prompt.format(**kwargs)
        return [SystemMessage(content=content)]


class HumanPromptTemplate(BaseChatPromptTemplate):
    """The human prompt template."""

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the prompt with the inputs.

        Returns:
            List[BaseMessage]: The formatted messages.
        """
        content = self.prompt.format(**kwargs)
        return [HumanMessage(content=content)]


class MessagesPlaceholder(BaseModel):
    """The messages placeholder template.

    Mostly used for the chat history.
    """

    variable_name: str

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the prompt with the inputs.

        Just return the messages from the kwargs with the variable name.

        Returns:
            List[BaseMessage]: The messages.
        """
        messages = kwargs.get(self.variable_name, [])
        if not isinstance(messages, list):
            raise ValueError(
                f"Unsupported messages type: {type(messages)}, should be list."
            )
        for message in messages:
            if not isinstance(message, BaseMessage):
                raise ValueError(
                    f"Unsupported message type: {type(message)}, should be BaseMessage."
                )
        return messages

    @property
    def input_variables(self) -> List[str]:
        """Return a list of the names of the variables the prompt template expects.

        Returns:
            List[str]: The input variables.
        """
        return [self.variable_name]
# 定义一个类型别名 MessageType，可以是 BaseChatPromptTemplate、MessagesPlaceholder 或 BaseMessage 中的任意一种
MessageType = Union[BaseChatPromptTemplate, MessagesPlaceholder, BaseMessage]

# 定义一个名为 ChatPromptTemplate 的类，继承自 BasePromptTemplate
class ChatPromptTemplate(BasePromptTemplate):
    """The chat prompt template.

    Examples:
        .. code-block:: python

            prompt_template = ChatPromptTemplate(
                messages=[
                    SystemPromptTemplate.from_template(
                        "You are a helpful AI assistant."
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanPromptTemplate.from_template("{question}"),
                ]
            )
    """

    # messages 属性，是一个 MessageType 类型的列表，存储模板中的消息内容
    messages: List[MessageType]

    # 格式化消息内容的方法，接受关键字参数并返回 BaseMessage 类型的列表
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the prompt with the inputs."""
        result_messages = []
        # 遍历 messages 列表中的每个消息
        for message in self.messages:
            if isinstance(message, BaseMessage):
                # 如果消息是 BaseMessage 类型，则直接添加到结果列表中
                result_messages.append(message)
            elif isinstance(message, (BaseChatPromptTemplate, MessagesPlaceholder)):
                # 如果消息是 BaseChatPromptTemplate 或 MessagesPlaceholder 类型，则传递关键字参数后继续格式化消息
                pass_kwargs = {
                    k: v for k, v in kwargs.items() if k in message.input_variables
                }
                result_messages.extend(message.format_messages(**pass_kwargs))
            else:
                # 如果消息类型不支持，则抛出 ValueError 异常
                raise ValueError(f"Unsupported message type: {type(message)}")
        return result_messages

    # 类方法，使用 model_validator 装饰器，在执行前验证
    @model_validator(mode="before")
    @classmethod
    def base_pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-fill the messages."""
        if not isinstance(values, dict):
            return values
        input_variables = values.get("input_variables", {})
        messages = values.get("messages", [])
        if not input_variables:
            input_variables = set()
            # 遍历 messages 列表，如果消息是 BaseChatPromptTemplate 或 MessagesPlaceholder 类型，则更新输入变量集合
            for message in messages:
                if isinstance(message, (BaseChatPromptTemplate, MessagesPlaceholder)):
                    input_variables.update(message.input_variables)
            # 将输入变量集合按字母顺序排序后存入 values 字典中
            values["input_variables"] = sorted(input_variables)
        return values


# 数据类 PromptTemplateIdentifier，继承自 ResourceIdentifier 类
@dataclasses.dataclass
class PromptTemplateIdentifier(ResourceIdentifier):
    """The identifier of a prompt template."""

    # identifier_split 属性，用于分隔不同部分的标识符，设置为固定值 "___$$$$___"
    identifier_split: str = dataclasses.field(default="___$$$$___", init=False)
    prompt_name: str  # 提示模板的名称
    prompt_language: Optional[str] = None  # 提示模板的语言，可选
    sys_code: Optional[str] = None  # 系统代码，可选
    model: Optional[str] = None  # 模型名称，可选

    # 后初始化方法，用于在实例化后进行一些操作
    def __post_init__(self):
        """Post init method."""
        # 检查 prompt_name 是否为 None，如果是则抛出 ValueError 异常
        if self.prompt_name is None:
            raise ValueError("prompt_name cannot be None")

        # 检查 identifier_split 是否出现在 prompt_name、prompt_language、sys_code 或 model 中，如果出现则抛出 ValueError 异常
        if any(
            self.identifier_split in key
            for key in [
                self.prompt_name,
                self.prompt_language,
                self.sys_code,
                self.model,
            ]
            if key is not None
        ):
            raise ValueError(
                f"identifier_split {self.identifier_split} is not allowed in "
                f"prompt_name, prompt_language, sys_code, model"
            )

    # 属性方法，用于返回 identifier_split 属性的值
    @property
    # 返回由多个标识符组成的字符串标识符
    def str_identifier(self) -> str:
        """Return the string identifier of the identifier."""
        return self.identifier_split.join(
            key
            for key in [
                self.prompt_name,    # 拼接自定义提示名称
                self.prompt_language,    # 拼接自定义语言
                self.sys_code,    # 拼接系统代码
                self.model,    # 拼接模型名称
            ]
            if key is not None    # 只添加非空标识符
        )

    # 将标识符转换为字典形式
    def to_dict(self) -> Dict:
        """Convert the identifier to a dict.

        Returns:
            Dict: The dict of the identifier.
        """
        return {
            "prompt_name": self.prompt_name,    # 将自定义提示名称添加到字典
            "prompt_language": self.prompt_language,    # 将自定义语言添加到字典
            "sys_code": self.sys_code,    # 将系统代码添加到字典
            "model": self.model,    # 将模型名称添加到字典
        }
@dataclasses.dataclass
class StoragePromptTemplate(StorageItem):
    """The storage prompt template."""

    prompt_name: str
    content: Optional[str] = None
    prompt_language: Optional[str] = None
    prompt_format: Optional[str] = None
    input_variables: Optional[str] = None
    model: Optional[str] = None
    chat_scene: Optional[str] = None
    sub_chat_scene: Optional[str] = None
    prompt_type: Optional[str] = None
    user_name: Optional[str] = None
    sys_code: Optional[str] = None
    _identifier: PromptTemplateIdentifier = dataclasses.field(init=False)

    def __post_init__(self):
        """Post init method."""
        # 创建一个唯一标识符对象用于此模板的识别
        self._identifier = PromptTemplateIdentifier(
            prompt_name=self.prompt_name,
            prompt_language=self.prompt_language,
            sys_code=self.sys_code,
            model=self.model,
        )
        # 假设 _check() 是在初始化后需要调用的方法
        self._check()

    def to_prompt_template(self) -> PromptTemplate:
        """Convert the storage prompt template to a prompt template."""
        # 将输入变量字符串分割为列表，如果为空则创建一个空列表
        input_variables = (
            [] if not self.input_variables else self.input_variables.strip().split(",")
        )
        # 确定模板格式，默认为 "f-string"
        template_format = self.prompt_format or "f-string"
        # 返回一个 PromptTemplate 对象，表示存储的提示模板
        return PromptTemplate(
            input_variables=input_variables,
            template=self.content,
            template_scene=self.chat_scene,
            # 注释掉了 prompt_name 参数，可能因为不需要在这里再次传递
            template_format=template_format,
        )

    @staticmethod
    def from_prompt_template(
        prompt_template: PromptTemplate,
        prompt_name: str,
        prompt_language: Optional[str] = None,
        prompt_type: Optional[str] = None,
        sys_code: Optional[str] = None,
        user_name: Optional[str] = None,
        sub_chat_scene: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> "StoragePromptTemplate":
        """
        Convert a prompt template to a storage prompt template.

        Args:
            prompt_template (PromptTemplate): The prompt template to convert from.
            prompt_name (str): The name of the prompt.
            prompt_language (Optional[str], optional): The language of the prompt.
                Defaults to None. e.g. zh-cn, en.
            prompt_type (Optional[str], optional): The type of the prompt.
                Defaults to None. e.g. common, private.
            sys_code (Optional[str], optional): The system code of the prompt.
                Defaults to None.
            user_name (Optional[str], optional): The username of the prompt.
                Defaults to None.
            sub_chat_scene (Optional[str], optional): The sub chat scene of the prompt.
                Defaults to None.
            model (Optional[str], optional): The model name of the prompt.
                Defaults to None.
            kwargs (Dict): Other params to build the storage prompt template.
        """
        # Get input variables from either prompt_template or kwargs
        input_variables = prompt_template.input_variables or kwargs.get(
            "input_variables"
        )
        # If input_variables is a list, convert it to a comma-separated string
        if input_variables and isinstance(input_variables, list):
            input_variables = ",".join(input_variables)
        # Return a new StoragePromptTemplate instance with specified parameters
        return StoragePromptTemplate(
            prompt_name=prompt_name,
            sys_code=sys_code,
            user_name=user_name,
            input_variables=input_variables,
            model=model,
            content=prompt_template.template or kwargs.get("content"),
            prompt_language=prompt_language,
            prompt_format=prompt_template.template_format
            or kwargs.get("prompt_format"),
            chat_scene=prompt_template.template_scene or kwargs.get("chat_scene"),
            sub_chat_scene=sub_chat_scene,
            prompt_type=prompt_type,
        )

    @property
    def identifier(self) -> PromptTemplateIdentifier:
        """
        Return the identifier of the storage prompt template.
        """
        return self._identifier

    def merge(self, other: "StorageItem") -> None:
        """
        Merge the other item into the current item.

        Args:
            other (StorageItem): The other item to merge
        """
        # Check if 'other' is of type StoragePromptTemplate
        if not isinstance(other, StoragePromptTemplate):
            # Raise an error if 'other' is not the same type
            raise ValueError(
                f"Cannot merge {type(other)} into {type(self)} because they are not "
                f"the same type."
            )
        # Merge the attributes of 'other' into the current instance
        self.from_object(other)
    def to_dict(self) -> Dict:
        """将存储的提示模板转换为字典格式。

        Returns:
            Dict: 存储的提示模板的字典表示。
        """
        return {
            "prompt_name": self.prompt_name,  # 提示名称
            "content": self.content,          # 提示内容
            "prompt_language": self.prompt_language,  # 提示语言
            "prompt_format": self.prompt_format,      # 提示格式
            "input_variables": self.input_variables,  # 输入变量
            "model": self.model,                      # 模型
            "chat_scene": self.chat_scene,            # 对话场景
            "sub_chat_scene": self.sub_chat_scene,    # 子对话场景
            "prompt_type": self.prompt_type,          # 提示类型
            "user_name": self.user_name,              # 用户名称
            "sys_code": self.sys_code,                # 系统代码
        }

    def _check(self):
        if self.prompt_name is None:
            raise ValueError("prompt_name cannot be None")  # 如果提示名称为空，则抛出数值错误异常
        if self.content is None:
            raise ValueError("content cannot be None")      # 如果提示内容为空，则抛出数值错误异常

    def from_object(self, template: "StoragePromptTemplate") -> None:
        """从现有的提示模板对象加载提示模板。

        Args:
            template (PromptTemplate): 要加载的提示模板对象。
        """
        self.content = template.content                # 加载内容
        self.prompt_format = template.prompt_format    # 加载提示格式
        self.input_variables = template.input_variables  # 加载输入变量
        self.model = template.model                    # 加载模型
        self.chat_scene = template.chat_scene          # 加载对话场景
        self.sub_chat_scene = template.sub_chat_scene  # 加载子对话场景
        self.prompt_type = template.prompt_type        # 加载提示类型
        self.user_name = template.user_name            # 加载用户名称
    """The manager class for prompt templates.

    Simple wrapper for the storage interface.

    Examples:
        .. code-block:: python

            # Default use InMemoryStorage
            prompt_manager = PromptManager()
            prompt_template = PromptTemplate(
                template="hello {input}",
                input_variables=["input"],
                template_scene="chat_normal",
            )
            prompt_manager.save(prompt_template, prompt_name="hello")
            prompt_template_list = prompt_manager.list()
            prompt_template_list = prompt_manager.prefer_query("hello")

        With a custom storage interface.

        .. code-block:: python

            from dbgpt.core.interface.storage import InMemoryStorage

            prompt_manager = PromptManager(InMemoryStorage())
            prompt_template = PromptTemplate(
                template="hello {input}",
                input_variables=["input"],
                template_scene="chat_normal",
            )
            prompt_manager.save(prompt_template, prompt_name="hello")
            prompt_template_list = prompt_manager.list()
            prompt_template_list = prompt_manager.prefer_query("hello")


    """

    def __init__(
        self, storage: Optional[StorageInterface[StoragePromptTemplate, Any]] = None
    ):
        """Create a new prompt manager."""
        # 初始化 PromptManager 实例时，如果未提供 storage 参数，则默认使用 InMemoryStorage
        if storage is None:
            storage = InMemoryStorage()
        self._storage = storage

    @property
    def storage(self) -> StorageInterface[StoragePromptTemplate, Any]:
        """Return the storage interface for prompt templates."""
        # 返回用于存储提示模板的存储接口对象
        return self._storage

    def prefer_query(
        self,
        prompt_name: str,
        sys_code: Optional[str] = None,
        prefer_prompt_language: Optional[str] = None,
        prefer_model: Optional[str] = None,
        **kwargs,
    # 将提示模板保存到存储中
    def save(self, prompt_template: PromptTemplate, prompt_name: str, **kwargs) -> None:
        """Save a prompt template to storage.

        Examples:
            .. code-block:: python

                prompt_template = PromptTemplate(
                    template="hello {input}",
                    input_variables=["input"],
                    template_scene="chat_normal",
                    prompt_name="hello",
                )
                prompt_manager.save(prompt_template)

            Save with sys_code and username.

            .. code-block:: python

                prompt_template = PromptTemplate(
                    template="hello {input}",
                    input_variables=["input"],
                    template_scene="chat_normal",
                    prompt_name="hello",
                )
                prompt_manager.save(
                    prompt_template, sys_code="sys_code", user_name="user_name"
                )

        Args:
            prompt_template (PromptTemplate): The prompt template to save.
            prompt_name (str): The name of the prompt template.
            kwargs (Dict): Other params to build the storage prompt template.
                More details in :meth:`~StoragePromptTemplate.from_prompt_template`.
        """
        # 从 PromptTemplate 创建 StoragePromptTemplate 对象
        storage_prompt_template = StoragePromptTemplate.from_prompt_template(
            prompt_template, prompt_name, **kwargs
        )
        # 将存储的提示模板保存到存储系统中
        self.storage.save(storage_prompt_template)

    # 查询存储中的提示模板，如果不存在则保存新模板
    def query_or_save(
        self, prompt_template: PromptTemplate, prompt_name: str, **kwargs
    ) -> StoragePromptTemplate:
        """Query a prompt template from storage, if not found, save it.

        Args:
            prompt_template (PromptTemplate): The prompt template to save.
            prompt_name (str): The name of the prompt template.
            kwargs (Dict): Other params to build the storage prompt template.
                More details in :meth:`~StoragePromptTemplate.from_prompt_template`.

        Returns:
            StoragePromptTemplate: The storage prompt template.
        """
        # 从 PromptTemplate 创建 StoragePromptTemplate 对象
        storage_prompt_template = StoragePromptTemplate.from_prompt_template(
            prompt_template, prompt_name, **kwargs
        )
        # 尝试从存储中加载已存在的提示模板
        exist_prompt_template = self.storage.load(
            storage_prompt_template.identifier, StoragePromptTemplate
        )
        # 如果已存在，则直接返回该提示模板
        if exist_prompt_template:
            return exist_prompt_template
        # 否则，保存新的提示模板到存储中
        self.save(prompt_template, prompt_name, **kwargs)
        # 再次加载保存后的提示模板
        prompt = self.storage.load(
            storage_prompt_template.identifier, StoragePromptTemplate
        )
        # 如果加载失败，抛出数值错误异常
        if not prompt:
            raise ValueError("Can't read prompt from storage")
        # 返回加载成功的提示模板
        return prompt
    def list(self, **kwargs) -> List[StoragePromptTemplate]:
        """Retrieve a list of prompt templates from storage.

        Examples:
            Retrieve all prompt templates.
            .. code-block:: python

                all_prompt_templates = prompt_manager.list()

            Retrieve prompt templates with specific filters.
            .. code-block:: python

                templates = prompt_manager.list(
                    sys_code="sys_code", user_name="user_name"
                )

        Args:
            kwargs (Dict): Additional query parameters for filtering.

        Returns:
            List[StoragePromptTemplate]: A list of prompt templates matching the criteria.
        """
        # Create a query specification object with given conditions
        query_spec = QuerySpec(conditions=kwargs)
        # Query the storage with the specified query specification for prompt templates
        return self.storage.query(query_spec, StoragePromptTemplate)

    def delete(
        self,
        prompt_name: str,
        prompt_language: Optional[str] = None,
        sys_code: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Delete a prompt template from storage.

        Examples:
            Delete a prompt template.

            .. code-block:: python

                prompt_manager.delete("hello")

            Delete with specific identifiers.

            .. code-block:: python

                prompt_manager.delete(
                    "hello", sys_code="sys_code", model="model"
                )

        Args:
            prompt_name (str): The name of the prompt template to delete.
            prompt_language (Optional[str], optional): The language of the prompt template.
                Defaults to None.
            sys_code (Optional[str], optional): The system code of the prompt template.
                Defaults to None.
            model (Optional[str], optional): The model of the prompt template.
                Defaults to None.
        """
        # Create an identifier object for the prompt template
        identifier = PromptTemplateIdentifier(
            prompt_name=prompt_name,
            prompt_language=prompt_language,
            sys_code=sys_code,
            model=model,
        )
        # Call the storage to delete the prompt template identified by the identifier
        self.storage.delete(identifier)
# 从模板字符串中获取模板变量集合
def _get_string_template_vars(template_str: str) -> Set[str]:
    # 创建一个空集合用于存储模板变量
    variables = set()
    # 创建一个 Formatter 对象
    formatter = Formatter()

    # 遍历模板字符串中的变量
    for _, variable_name, _, _ in formatter.parse(template_str):
        # 如果变量名不为空，则添加到集合中
        if variable_name:
            variables.add(variable_name)

    # 返回模板变量集合
    return variables


# 从模板字符串中获取模板变量集合
def _get_jinja2_template_vars(template_str: str) -> Set[str]:
    # 导入 jinja2 模块中的 Environment 和 meta
    from jinja2 import Environment, meta

    # 创建一个 Environment 对象
    env = Environment()
    # 解析模板字符串生成 AST
    ast = env.parse(template_str)
    # 从 AST 中找到未声明的变量
    variables = meta.find_undeclared_variables(ast)
    # 返回模板变量集合
    return variables


# 从模板字符串中获取模板变量列表
def get_template_vars(
    template_str: str, template_format: str = "f-string"
) -> List[str]:
    # 根据模板格式选择不同的函数获取模板变量
    if template_format == "f-string":
        result = _get_string_template_vars(template_str)
    elif template_format == "jinja2":
        result = _get_jinja2_template_vars(template_str)
    else:
        # 抛出异常，不支持的模板格式
        raise ValueError(f"Unsupported template format: {template_format}")
    # 对结果进行排序并返回
    return sorted(result)
```