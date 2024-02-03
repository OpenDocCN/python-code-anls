# `.\AutoGPT\autogpts\autogpt\autogpt\core\prompting\schema.py`

```py
# 导入必要的模块
import enum
from pydantic import BaseModel, Field
from autogpt.core.resource.model_providers.schema import (
    ChatMessage,
    ChatMessageDict,
    CompletionModelFunction,
)

# 定义枚举类 LanguageModelClassification，描述模型的功能
class LanguageModelClassification(str, enum.Enum):
    """The LanguageModelClassification is a functional description of the model.

    This is used to determine what kind of model to use for a given prompt.
    Sometimes we prefer a faster or cheaper model to accomplish a task when
    possible.
    """
    FAST_MODEL = "fast_model"
    SMART_MODEL = "smart_model"

# 定义类 ChatPrompt，用于表示聊天提示
class ChatPrompt(BaseModel):
    # 消息列表
    messages: list[ChatMessage]
    # 功能列表，默认为空列表
    functions: list[CompletionModelFunction] = Field(default_factory=list)

    # 返回原始消息列表的字典形式
    def raw(self) -> list[ChatMessageDict]:
        return [m.dict() for m in self.messages]

    # 定义类的字符串表示形式
    def __str__(self):
        return "\n\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in self.messages
        )
```