# `.\Langchain-Chatchat\server\callback_handler\conversation_callback_handler.py`

```py
# 导入必要的模块
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from server.db.repository import update_message

# 定义一个名为ConversationCallbackHandler的类，继承自BaseCallbackHandler类
class ConversationCallbackHandler(BaseCallbackHandler):
    # 设置一个类属性raise_error为True
    raise_error: bool = True

    # 初始化方法，接受conversation_id, message_id, chat_type, query作为参数
    def __init__(self, conversation_id: str, message_id: str, chat_type: str, query: str):
        # 初始化实例属性
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.chat_type = chat_type
        self.query = query
        self.start_at = None

    # 定义一个属性方法always_verbose，返回True
    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    # 定义一个方法on_llm_start，接受serialized, prompts和kwargs作为参数
    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # 如果想存更多信息，则prompts 也需要持久化
        pass

    # 定义一个方法on_llm_end，接受response和kwargs作为参数
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # 从response中获取生成的答案
        answer = response.generations[0][0].text
        # 调用update_message方法，更新消息的答案
        update_message(self.message_id, answer)
```