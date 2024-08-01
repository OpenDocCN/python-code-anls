# `.\DB-GPT-src\dbgpt\core\interface\operators\prompt_operator.py`

```py
# 导入必要的模块和类
from abc import ABC
from typing import Any, Dict, List, Optional, Union

# 导入私有模块中的验证函数
from dbgpt._private.pydantic import model_validator

# 导入核心功能模块
from dbgpt.core import (
    ModelMessage,
    ModelMessageRoleType,
    ModelOutput,
    StorageConversation,
)
# 导入 AWEL 模块中的操作符
from dbgpt.core.awel import JoinOperator, MapOperator
from dbgpt.core.awel.flow import (
    IOField,
    OperatorCategory,
    OperatorType,
    Parameter,
    ResourceCategory,
    ViewMetadata,
    register_resource,
)
# 导入消息接口相关模块
from dbgpt.core.interface.message import BaseMessage
from dbgpt.core.interface.operators.llm_operator import BaseLLM
from dbgpt.core.interface.operators.message_operator import BaseConversationOperator
# 导入提示模板相关模块
from dbgpt.core.interface.prompt import (
    BaseChatPromptTemplate,
    ChatPromptTemplate,
    HumanPromptTemplate,
    MessagesPlaceholder,
    MessageType,
    PromptTemplate,
    SystemPromptTemplate,
)
# 导入实用函数和国际化相关工具
from dbgpt.util.function_utils import rearrange_args_by_type
from dbgpt.util.i18n_utils import _

# 注册资源，并定义常用聊天提示模板的类
@register_resource(
    label=_("Common Chat Prompt Template"),
    name="common_chat_prompt_template",
    category=ResourceCategory.PROMPT,
    description=_("The operator to build the prompt with static prompt."),
    parameters=[
        Parameter.build_from(
            label=_("System Message"),
            name="system_message",
            type=str,
            optional=True,
            default="You are a helpful AI Assistant.",
            description=_("The system message."),
        ),
        Parameter.build_from(
            label=_("Message placeholder"),
            name="message_placeholder",
            type=str,
            optional=True,
            default="chat_history",
            description=_("The chat history message placeholder."),
        ),
        Parameter.build_from(
            label=_("Human Message"),
            name="human_message",
            type=str,
            optional=True,
            default="{user_input}",
            placeholder="{user_input}",
            description=_("The human message."),
        ),
    ],
)
class CommonChatPromptTemplate(ChatPromptTemplate):
    """The common chat prompt template."""

    # 在模型验证之前执行的类方法
    @model_validator(mode="before")
    @classmethod
    # 定义一个类方法，用于预填充消息内容
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre fill the messages."""
        # 如果传入的值不是字典类型，则直接返回该值，无需处理
        if not isinstance(values, dict):
            return values
        # 如果传入的字典中不存在键 "system_message"，则设定默认系统消息
        if "system_message" not in values:
            values["system_message"] = "You are a helpful AI Assistant."
        # 如果传入的字典中不存在键 "human_message"，则设定默认人类消息模板
        if "human_message" not in values:
            values["human_message"] = "{user_input}"
        # 如果传入的字典中不存在键 "message_placeholder"，则设定默认消息占位符
        if "message_placeholder" not in values:
            values["message_placeholder"] = "chat_history"
        # 弹出并获取系统消息、人类消息和消息占位符，并从传入的字典中移除这些键
        system_message = values.pop("system_message")
        human_message = values.pop("human_message")
        message_placeholder = values.pop("message_placeholder")
        # 构造消息列表，包括系统提示、消息占位符和人类提示
        values["messages"] = [
            SystemPromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name=message_placeholder),
            HumanPromptTemplate.from_template(human_message),
        ]
        # 返回基类的预填充方法处理后的结果字典
        return cls.base_pre_fill(values)
class BasePromptBuilderOperator(BaseConversationOperator, ABC):
    """The base prompt builder operator."""

    def __init__(self, check_storage: bool, **kwargs):
        """Create a new prompt builder operator."""
        super().__init__(check_storage=check_storage, **kwargs)

    async def format_prompt(
        self, prompt: ChatPromptTemplate, prompt_dict: Dict[str, Any]
    ) -> List[ModelMessage]:
        """Format the prompt.

        Args:
            prompt (ChatPromptTemplate): The prompt.
            prompt_dict (Dict[str, Any]): The prompt dict.

        Returns:
            List[ModelMessage]: The formatted prompt.
        """
        kwargs = {}
        kwargs.update(prompt_dict)
        pass_kwargs = {k: v for k, v in kwargs.items() if k in prompt.input_variables}
        messages = prompt.format_messages(**pass_kwargs)
        model_messages = ModelMessage.from_base_messages(messages)
        # Start new round conversation, and save user message to storage
        await self.start_new_round_conv(model_messages)
        return model_messages

    async def start_new_round_conv(self, messages: List[ModelMessage]) -> None:
        """Start a new round conversation.

        Args:
            messages (List[ModelMessage]): The messages.
        """
        last_user_message = None  # Initialize variable to hold the last user message
        for message in messages[::-1]:  # Iterate through messages in reverse order
            if message.role == ModelMessageRoleType.HUMAN:  # Check if message role is human
                last_user_message = message.content  # Capture the user's message content
                break
        if not last_user_message:  # If no user message was found
            raise ValueError("No user message")  # Raise an error indicating no user message found
        storage_conv: Optional[
            StorageConversation
        ] = await self.get_storage_conversation()  # Get the storage conversation object asynchronously
        if not storage_conv:  # If no storage conversation object exists
            return  # Exit the function
        # Start new round in the conversation storage
        storage_conv.start_new_round()
        storage_conv.add_user_message(last_user_message)  # Add the last user message to storage

    async def after_dag_end(self, event_loop_task_id: int):
        """Execute after the DAG finished."""
        # Save the storage conversation to storage after the whole DAG finished
        storage_conv: Optional[
            StorageConversation
        ] = await self.get_storage_conversation()  # Get the storage conversation object asynchronously

        if not storage_conv:  # If no storage conversation object exists
            return  # Exit the function
        model_output: Optional[
            ModelOutput
        ] = await self.current_dag_context.get_from_share_data(
            BaseLLM.SHARE_DATA_KEY_MODEL_OUTPUT  # Get model output data from shared context
        )
        if model_output:  # If model output data exists
            # Save model output message to storage
            storage_conv.add_ai_message(model_output.text)
            # End current conversation round and flush to storage
            storage_conv.end_current_round()


PromptTemplateType = Union[ChatPromptTemplate, PromptTemplate, MessageType, str]


class PromptBuilderOperator(
    BasePromptBuilderOperator, MapOperator[Dict[str, Any], List[ModelMessage]]
):
    """The operator to build the prompt with static prompt.

    """
    metadata = ViewMetadata(
        label=_("Prompt Builder Operator"),  # 设置视图元数据的标签为“Prompt Builder Operator”
        name="prompt_builder_operator",      # 设置视图元数据的名称为“prompt_builder_operator”
        description=_("Build messages from prompt template."),  # 设置视图元数据的描述为“Build messages from prompt template.”
        category=OperatorCategory.COMMON,    # 设置视图元数据的类别为COMMON
        parameters=[                        # 设置视图元数据的参数列表
            Parameter.build_from(
                _("Chat Prompt Template"),   # 设置参数的标签为“Chat Prompt Template”
                "prompt",                    # 设置参数的名称为“prompt”
                ChatPromptTemplate,          # 设置参数的类型为ChatPromptTemplate
                description=_("The chat prompt template."),  # 设置参数的描述为“The chat prompt template.”
            ),
        ],
        inputs=[                            # 设置视图元数据的输入字段列表
            IOField.build_from(
                _("Prompt Input Dict"),     # 设置输入字段的标签为“Prompt Input Dict”
                "prompt_input_dict",        # 设置输入字段的名称为“prompt_input_dict”
                dict,                       # 设置输入字段的类型为dict
                description=_("The prompt dict."),  # 设置输入字段的描述为“The prompt dict.”
            )
        ],
        outputs=[                           # 设置视图元数据的输出字段列表
            IOField.build_from(
                _("Formatted Messages"),    # 设置输出字段的标签为“Formatted Messages”
                "formatted_messages",       # 设置输出字段的名称为“formatted_messages”
                ModelMessage,               # 设置输出字段的类型为ModelMessage
                is_list=True,               # 设置输出字段为列表形式
                description=_("The formatted messages."),  # 设置输出字段的描述为“The formatted messages.”
            )
        ],
    )
    
    def __init__(self, prompt: PromptTemplateType, **kwargs):
        """Create a new prompt builder operator."""
        if isinstance(prompt, str):
            prompt = ChatPromptTemplate(
                messages=[HumanPromptTemplate.from_template(prompt)]
            )  # 如果传入的prompt是字符串，则创建一个ChatPromptTemplate对象，包含从模板创建的HumanPromptTemplate消息
        elif isinstance(prompt, PromptTemplate):
            prompt = ChatPromptTemplate(
                messages=[HumanPromptTemplate.from_template(prompt.template)]
            )  # 如果传入的prompt是PromptTemplate对象，则创建一个ChatPromptTemplate对象，包含从模板创建的HumanPromptTemplate消息
        elif isinstance(
            prompt, (BaseChatPromptTemplate, MessagesPlaceholder, BaseMessage)
        ):
            prompt = ChatPromptTemplate(messages=[prompt])  # 如果传入的prompt是BaseChatPromptTemplate、MessagesPlaceholder或BaseMessage的实例，则创建一个ChatPromptTemplate对象，包含传入的prompt消息
        self._prompt = prompt  # 将处理后的prompt保存到实例变量self._prompt中
    
        super().__init__(check_storage=False, **kwargs)  # 调用父类ViewOperator的初始化方法，禁用存储检查
        MapOperator.__init__(self, map_function=self.merge_prompt, **kwargs)  # 调用MapOperator类的初始化方法，设置映射函数为self.merge_prompt，传递额外的kwargs参数
    
    @rearrange_args_by_type
    async def merge_prompt(self, prompt_dict: Dict[str, Any]) -> List[ModelMessage]:
        """Format the prompt."""
        return await self.format_prompt(self._prompt, prompt_dict)
class DynamicPromptBuilderOperator(
    BasePromptBuilderOperator, JoinOperator[List[ModelMessage]]
):
    """The operator to build the prompt with dynamic prompt.

    The prompt template is dynamic, and it created by parent operator.
    """

    def __init__(self, **kwargs):
        """Create a new dynamic prompt builder operator."""
        # 调用父类的初始化方法，禁用存储检查，传递其他关键字参数
        super().__init__(check_storage=False, **kwargs)
        # 调用 JoinOperator 的初始化方法，指定合并函数为 merge_prompt，传递其他关键字参数
        JoinOperator.__init__(self, combine_function=self.merge_prompt, **kwargs)

    @rearrange_args_by_type
    async def merge_prompt(
        self, prompt: ChatPromptTemplate, prompt_dict: Dict[str, Any]
    ) -> List[ModelMessage]:
        """Merge the prompt and history."""
        # 调用 format_prompt 方法，将 prompt 和 prompt_dict 作为参数传递，返回格式化后的消息列表
        return await self.format_prompt(prompt, prompt_dict)


class HistoryPromptBuilderOperator(
    BasePromptBuilderOperator, JoinOperator[List[ModelMessage]]
):
    """The operator to build the prompt with static prompt.

    The prompt will pass to this operator.
    """

    metadata = ViewMetadata(
        label=_("History Prompt Builder Operator"),
        name="history_prompt_builder_operator",
        description=_("Build messages from prompt template and chat history."),
        operator_type=OperatorType.JOIN,
        category=OperatorCategory.CONVERSION,
        parameters=[
            Parameter.build_from(
                _("Chat Prompt Template"),
                "prompt",
                ChatPromptTemplate,
                description=_("The chat prompt template."),
            ),
            Parameter.build_from(
                _("History Key"),
                "history_key",
                str,
                optional=True,
                default="chat_history",
                description=_("The key of history in prompt dict."),
            ),
            Parameter.build_from(
                _("String History"),
                "str_history",
                bool,
                optional=True,
                default=False,
                description=_("Whether to convert the history to string."),
            ),
        ],
        inputs=[
            IOField.build_from(
                _("History"),
                "history",
                BaseMessage,
                is_list=True,
                description=_("The history."),
            ),
            IOField.build_from(
                _("Prompt Input Dict"),
                "prompt_input_dict",
                dict,
                description=_("The prompt dict."),
            ),
        ],
        outputs=[
            IOField.build_from(
                _("Formatted Messages"),
                "formatted_messages",
                ModelMessage,
                is_list=True,
                description=_("The formatted messages."),
            )
        ],
    )

    def __init__(
        self,
        prompt: ChatPromptTemplate,
        history_key: str = "chat_history",
        check_storage: bool = True,
        str_history: bool = False,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递 prompt 参数和其他关键字参数
        super().__init__(prompt=prompt, check_storage=check_storage, **kwargs)
        # 初始化历史提示生成操作器的元数据和参数
        self.metadata = self.__class__.metadata
    ):
        """
        创建一个新的历史提示构建操作符。

        Args:
            prompt (ChatPromptTemplate): 提示内容。
            history_key (str, optional): 在提示字典中历史记录的键。默认为 "chat_history"。
            check_storage (bool, optional): 是否检查存储。默认为 True。
            str_history (bool, optional): 是否将历史记录转换为字符串。默认为 False。
        """
        self._prompt = prompt
        self._history_key = history_key
        self._str_history = str_history
        # 调用父类的初始化方法，设置存储检查标志
        BasePromptBuilderOperator.__init__(self, check_storage=check_storage)
        # 调用 JoinOperator 的初始化方法，设置合并函数为 self.merge_history，并传递额外的关键字参数
        JoinOperator.__init__(self, combine_function=self.merge_history, **kwargs)

    @rearrange_args_by_type
    async def merge_history(
        self, history: List[BaseMessage], prompt_dict: Dict[str, Any]
    ) -> List[ModelMessage]:
        """
        合并提示和历史记录。

        Args:
            history (List[BaseMessage]): 历史记录的列表。
            prompt_dict (Dict[str, Any]): 提示字典，用于存储合并后的结果。

        Returns:
            List[ModelMessage]: 合并后的消息列表。
        """
        if self._str_history:
            # 如果设定为字符串历史记录，则将历史记录转换为字符串并存储到提示字典中
            prompt_dict[self._history_key] = BaseMessage.messages_to_string(history)
        else:
            # 否则直接将历史记录存储到提示字典中
            prompt_dict[self._history_key] = history
        # 调用 format_prompt 方法格式化提示，并返回格式化后的结果
        return await self.format_prompt(self._prompt, prompt_dict)
# 历史动态提示构建操作符，继承自基础提示构建操作符和列表连接操作符
class HistoryDynamicPromptBuilderOperator(
    BasePromptBuilderOperator, JoinOperator[List[ModelMessage]]
):
    """The operator to build the prompt with dynamic prompt.

    The prompt template is dynamic, and it created by parent operator.
    """

    def __init__(
        self,
        history_key: str = "chat_history",  # 设置历史记录键，默认为"chat_history"
        check_storage: bool = True,  # 是否检查存储，默认为True
        str_history: bool = False,  # 是否将历史记录转换为字符串，默认为False
        **kwargs,
    ):
        """Create a new history dynamic prompt builder operator."""
        self._history_key = history_key  # 初始化历史记录键
        self._str_history = str_history  # 初始化是否转换历史记录为字符串的标志
        BasePromptBuilderOperator.__init__(self, check_storage=check_storage)  # 调用父类BasePromptBuilderOperator的初始化方法
        JoinOperator.__init__(self, combine_function=self.merge_history, **kwargs)  # 调用父类JoinOperator的初始化方法，传入合并历史记录的函数

    @rearrange_args_by_type
    async def merge_history(
        self,
        prompt: ChatPromptTemplate,  # 提示模板对象
        history: List[BaseMessage],  # 历史消息列表
        prompt_dict: Dict[str, Any],  # 提示字典，用于构建最终的提示
    ) -> List[ModelMessage]:
        """Merge the prompt and history."""
        if self._str_history:
            prompt_dict[self._history_key] = BaseMessage.messages_to_string(history)  # 如果需要转换为字符串，则将历史消息列表转换为字符串后存入提示字典
        else:
            prompt_dict[self._history_key] = history  # 否则直接将历史消息列表存入提示字典
        return await self.format_prompt(prompt, prompt_dict)  # 调用格式化提示的方法，返回格式化后的模型消息列表
```