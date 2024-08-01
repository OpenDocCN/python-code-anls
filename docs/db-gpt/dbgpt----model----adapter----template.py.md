# `.\DB-GPT-src\dbgpt\model\adapter\template.py`

```py
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from enum import Enum  # 导入枚举类型
from typing import TYPE_CHECKING, List, Optional, Tuple, Union  # 导入类型检查和类型提示相关的模块

if TYPE_CHECKING:
    from fastchat.conversation import Conversation  # 条件导入，仅用于类型检查时才导入Conversation类


class PromptType(str, Enum):
    """Prompt type."""
    FSCHAT: str = "fschat"  # 定义枚举类型FSCHAT，值为"fschat"
    DBGPT: str = "dbgpt"    # 定义枚举类型DBGPT，值为"dbgpt"


class ConversationAdapter(ABC):
    """The conversation adapter."""

    @property
    def prompt_type(self) -> PromptType:
        """Return the prompt type of the conversation adapter."""
        return PromptType.FSCHAT  # 默认返回FSCHAT作为对话适配器的提示类型

    @property
    @abstractmethod
    def roles(self) -> Tuple[str]:
        """Get the roles of the conversation.

        Returns:
            Tuple[str]: The roles of the conversation.
        """
        ...

    @property
    def sep(self) -> Optional[str]:
        """Get the separator between messages."""
        return "\n"  # 返回消息之间的分隔符为换行符

    @property
    def stop_str(self) -> Optional[Union[str, List[str]]]:
        """Get the stop criteria."""
        return None  # 默认返回停止条件为None，即无特定停止条件

    @property
    def stop_token_ids(self) -> Optional[List[int]]:
        """Stops generation if meeting any token in this list"""
        return None  # 默认返回停止生成的token列表为None，表示无特定停止token

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the prompt string.

        Returns:
            str: The prompt string.
        """
        ...

    @abstractmethod
    def set_system_message(self, system_message: str) -> None:
        """Set the system message."""
        ...

    @abstractmethod
    def append_message(self, role: str, message: str) -> None:
        """Append a new message.

        Args:
            role (str): The role of the message.
            message (str): The message content.
        """
        ...

    @abstractmethod
    def update_last_message(self, message: str) -> None:
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.

        Args:
            message (str): The message content.
        """
        ...

    @abstractmethod
    def copy(self) -> "ConversationAdapter":
        """Copy the conversation."""
        ...


class ConversationAdapterFactory(ABC):
    """The conversation adapter factory."""

    def get_by_name(
        self,
        template_name: str,
        prompt_template_type: Optional[PromptType] = PromptType.FSCHAT,
    ) -> ConversationAdapter:
        """Get a conversation adapter by name.

        Args:
            template_name (str): The name of the template.
            prompt_template_type (Optional[PromptType]): The type of the prompt template, default to be FSCHAT.

        Returns:
            ConversationAdapter: The conversation adapter.
        """
        raise NotImplementedError()  # 抽象方法，子类需实现具体的获取对话适配器的逻辑
    # 定义一个方法，用于根据模型名称和路径获取对话适配器
    def get_by_model(self, model_name: str, model_path: str) -> ConversationAdapter:
        """Get a conversation adapter by model.

        Args:
            model_name (str): The name of the model. 模型的名称
            model_path (str): The path of the model. 模型的路径

        Returns:
            ConversationAdapter: The conversation adapter. 返回获取到的对话适配器
        """
        # 抛出未实现错误，暂未实现该方法的具体逻辑
        raise NotImplementedError()
def get_conv_template(name: str) -> ConversationAdapter:
    """Get a conversation template.
    
    Args:
        name (str): The name of the template.
        
    Just return the fastchat conversation template for now.
    # TODO: More templates should be supported.
    Returns:
        Conversation: The conversation template.
    """
    # 导入获取对话模板的函数
    from fastchat.conversation import get_conv_template
    
    # 导入 FschatConversationAdapter 类
    from dbgpt.model.adapter.fschat_adapter import FschatConversationAdapter
    
    # 调用获取对话模板的函数，传入指定名称，获取模板对象
    conv_template = get_conv_template(name)
    
    # 使用获取的对话模板对象创建 FschatConversationAdapter 适配器对象
    return FschatConversationAdapter(conv_template)
```