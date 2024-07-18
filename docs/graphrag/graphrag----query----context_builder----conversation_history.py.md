# `.\graphrag\graphrag\query\context_builder\conversation_history.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Classes for storing and managing conversation history."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import tiktoken  # 导入tiktoken模块

from graphrag.query.llm.text_utils import num_tokens  # 导入num_tokens函数

"""
Enum for conversation roles
"""


class ConversationRole(str, Enum):
    """Enum for conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @staticmethod
    def from_string(value: str) -> "ConversationRole":
        """Convert string to ConversationRole."""
        if value == "system":
            return ConversationRole.SYSTEM  # 返回系统角色
        if value == "user":
            return ConversationRole.USER  # 返回用户角色
        if value == "assistant":
            return ConversationRole.ASSISTANT  # 返回助手角色

        msg = f"Invalid Role: {value}"
        raise ValueError(msg)

    def __str__(self) -> str:
        """Return string representation of the enum value."""
        return self.value  # 返回枚举值的字符串表示


"""
Data class for storing a single conversation turn
"""


@dataclass
class ConversationTurn:
    """Data class for storing a single conversation turn."""

    role: ConversationRole  # 对话角色，使用ConversationRole枚举类型
    content: str  # 对话内容

    def __str__(self) -> str:
        """Return string representation of the conversation turn."""
        return f"{self.role}: {self.content}"  # 返回对话角色和内容的字符串表示


@dataclass
class QATurn:
    """
    Data class for storing a QA turn.

    A QA turn contains a user question and one more multiple assistant answers.
    """

    user_query: ConversationTurn  # 用户提问，使用ConversationTurn类表示
    assistant_answers: list[ConversationTurn] | None = None  # 助手回答列表，可能为空

    def get_answer_text(self) -> str | None:
        """Get the text of the assistant answers."""
        return (
            "\n".join([answer.content for answer in self.assistant_answers])  # 获取助手回答的文本内容
            if self.assistant_answers
            else None
        )

    def __str__(self) -> str:
        """Return string representation of the QA turn."""
        answers = self.get_answer_text()
        return (
            f"Question: {self.user_query.content}\nAnswer: {answers}"  # 返回问题和回答的字符串表示
            if answers
            else f"Question: {self.user_query.content}"
        )


class ConversationHistory:
    """Class for storing a conversation history."""

    turns: list[ConversationTurn]  # 对话轮次列表

    def __init__(self):
        self.turns = []  # 初始化空的对话轮次列表

    @classmethod
    def from_list(
        cls, conversation_turns: list[dict[str, str]]  # 类方法，从字典列表创建对话轮次列表
        # 参数：conversation_turns - 包含对话角色和内容的字典列表
    ) -> "ConversationHistory":
        """
        Create a conversation history from a list of conversation turns.

        Each turn is a dictionary in the form of {"role": "<conversation_role>", "content": "<turn content>"}
        """
        # 创建一个空的对话历史记录对象
        history = cls()
        # 遍历每一个对话轮次并添加到历史记录中
        for turn in conversation_turns:
            # 将每个对话轮次转换为 ConversationTurn 对象，并添加到历史记录的 turns 列表中
            history.turns.append(
                ConversationTurn(
                    role=ConversationRole.from_string(
                        turn.get("role", ConversationRole.USER)
                    ),
                    content=turn.get("content", ""),
                )
            )
        # 返回创建好的对话历史记录对象
        return history

    def add_turn(self, role: ConversationRole, content: str):
        """Add a new turn to the conversation history."""
        # 向对话历史记录中添加新的对话轮次
        self.turns.append(ConversationTurn(role=role, content=content))

    def to_qa_turns(self) -> list[QATurn]:
        """Convert conversation history to a list of QA turns."""
        # 初始化一个空列表，用于存储 QA 对话轮次
        qa_turns = list[QATurn]()
        # 当前的 QA 对话轮次
        current_qa_turn = None
        # 遍历每一个对话历史记录中的对话轮次
        for turn in self.turns:
            # 如果是用户的对话轮次
            if turn.role == ConversationRole.USER:
                # 如果当前有未完成的 QA 对话轮次，将其添加到 qa_turns 列表中
                if current_qa_turn:
                    qa_turns.append(current_qa_turn)
                # 创建新的 QA 对话轮次，用户查询部分为当前的用户对话轮次，助手回答部分为空列表
                current_qa_turn = QATurn(user_query=turn, assistant_answers=[])
            else:
                # 如果当前有未完成的 QA 对话轮次，将当前的助手回答添加到对应的 QA 对话轮次中
                if current_qa_turn:
                    current_qa_turn.assistant_answers.append(turn)  # type: ignore
        # 将最后一个完成的 QA 对话轮次添加到 qa_turns 列表中
        if current_qa_turn:
            qa_turns.append(current_qa_turn)
        # 返回所有的 QA 对话轮次列表
        return qa_turns

    def get_user_turns(self, max_user_turns: int | None = 1) -> list[str]:
        """Get the last user turns in the conversation history."""
        # 初始化一个空列表，用于存储用户的对话内容
        user_turns = []
        # 从历史记录的最后一个对话轮次开始遍历
        for turn in self.turns[::-1]:
            # 如果是用户的对话轮次，将其内容添加到 user_turns 列表中
            if turn.role == ConversationRole.USER:
                user_turns.append(turn.content)
                # 如果指定了最大的用户对话轮次数，并且已经达到指定数量，结束遍历
                if max_user_turns and len(user_turns) >= max_user_turns:
                    break
        # 返回获取到的用户对话内容列表
        return user_turns

    def build_context(
        self,
        token_encoder: tiktoken.Encoding | None = None,
        include_user_turns_only: bool = True,
        max_qa_turns: int | None = 5,
        max_tokens: int = 8000,
        recency_bias: bool = True,
        column_delimiter: str = "|",
        context_name: str = "Conversation History",
    ) -> tuple[str, dict[str, pd.DataFrame]]:
        """
        准备会话历史作为系统提示的上下文数据。

        Parameters
        ----------
        user_queries_only: 如果为True，只包括用户查询（不包括助手的响应）在上下文中，默认为True。
        max_qa_turns: 包括在上下文中的最大问答轮次数，默认为1。
        recency_bias: 如果为True，反转会话历史的顺序以确保最后的问答被优先处理。
        column_delimiter: 上下文数据中用于分隔列的分隔符，默认为"|"。
        context_name: 上下文的名称，默认为"Conversation History"。

        """
        # 将会话转换为问答轮次列表
        qa_turns = self.to_qa_turns()

        # 如果仅包括用户查询，重新构造问答轮次列表
        if include_user_turns_only:
            qa_turns = [
                QATurn(user_query=qa_turn.user_query, assistant_answers=None)
                for qa_turn in qa_turns
            ]

        # 如果有最近偏好，反转问答轮次列表
        if recency_bias:
            qa_turns = qa_turns[::-1]

        # 如果限制了最大问答轮次数，截取相应数量的问答轮次
        if max_qa_turns and len(qa_turns) > max_qa_turns:
            qa_turns = qa_turns[:max_qa_turns]

        # 构建问答轮次的上下文
        # 添加上下文头部
        if len(qa_turns) == 0 or not qa_turns:
            return ("", {context_name: pd.DataFrame()})

        # 添加表头
        header = f"-----{context_name}-----" + "\n"

        turn_list = []  # 初始化轮次列表
        current_context_df = pd.DataFrame()  # 初始化当前上下文的DataFrame
        for turn in qa_turns:
            # 添加用户查询到轮次列表
            turn_list.append({
                "turn": ConversationRole.USER.__str__(),
                "content": turn.user_query.content,
            })
            # 如果存在助手的响应，添加到轮次列表
            if turn.assistant_answers:
                turn_list.append({
                    "turn": ConversationRole.ASSISTANT.__str__(),
                    "content": turn.get_answer_text(),
                })

            # 创建上下文的DataFrame
            context_df = pd.DataFrame(turn_list)
            # 将上下文DataFrame转换为CSV格式的文本
            context_text = header + context_df.to_csv(sep=column_delimiter, index=False)
            # 如果文本超出最大标记数，则中断
            if num_tokens(context_text, token_encoder) > max_tokens:
                break

            current_context_df = context_df  # 更新当前上下文的DataFrame

        # 将当前上下文DataFrame转换为CSV格式的文本
        context_text = header + current_context_df.to_csv(
            sep=column_delimiter, index=False
        )
        # 返回上下文文本和包含当前上下文DataFrame的字典
        return (context_text, {context_name.lower(): current_context_df})
```