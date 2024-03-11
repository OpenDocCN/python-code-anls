# `.\Langchain-Chatchat\server\memory\conversation_db_buffer_memory.py`

```
import logging
from typing import Any, List, Dict

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from server.db.repository.message_repository import filter_message
from server.db.models.message_model import MessageModel


class ConversationBufferDBMemory(BaseChatMemory):
    conversation_id: str
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000
    message_limit: int = 10

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        # fetch limited messages desc, and return reversed

        # 从数据库中获取限制数量的消息，并按时间倒序返回
        messages = filter_message(conversation_id=self.conversation_id, limit=self.message_limit)
        # 将消息列表反转，变为正序
        messages = list(reversed(messages))
        chat_messages: List[BaseMessage] = []
        for message in messages:
            chat_messages.append(HumanMessage(content=message["query"]))
            chat_messages.append(AIMessage(content=message["response"]))

        if not chat_messages:
            return []

        # 如果聊天消息超过最大令牌限制，则修剪聊天消息
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))

        return chat_messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]
    # 加载内存变量，返回历史缓冲区
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        # 将缓冲区赋值给变量buffer
        buffer: Any = self.buffer
        # 如果需要返回消息，则将buffer赋值给final_buffer，否则将buffer转换为字符串
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        # 返回包含内存键和最终缓冲区的字典
        return {self.memory_key: final_buffer}

    # 保存上下文，不需要保存或更改任何内容
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        # 什么也不做，保持空白

    # 清除内存，就像一个保险库一样，没有需要清除的内容
    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        # 什么也不做，保持空白
```