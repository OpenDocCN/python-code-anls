# `MetaGPT\metagpt\actions\talk_action.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/28
@Author  : mashenquan
@File    : talk_action.py
@Desc    : Act as it’s a talk
"""
# 导入必要的模块
from typing import Optional

from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.const import DEFAULT_LANGUAGE
from metagpt.logs import logger
from metagpt.schema import Message

# 定义 TalkAction 类，继承自 Action 类
class TalkAction(Action):
    # 定义类属性
    context: str
    history_summary: str = ""
    knowledge: str = ""
    rsp: Optional[Message] = None

    # 定义 prompt 属性，用于生成提示信息
    @property
    def prompt(self):
        # 生成提示信息
        prompt = ""
        if CONFIG.agent_description:
            prompt = (
                f"You are {CONFIG.agent_description}. Your responses should align with the role-play agreement, "
                f"maintaining the character's persona and habits. When faced with unrelated questions, playfully "
                f"decline to answer without revealing your AI nature to preserve the character's image.\n\n"
            )
        prompt += f"Knowledge:\n{self.knowledge}\n\n" if self.knowledge else ""
        prompt += f"{self.history_summary}\n\n"
        prompt += (
            "If the information is insufficient, you can search in the historical conversation or knowledge above.\n"
        )
        language = CONFIG.language or DEFAULT_LANGUAGE
        prompt += (
            f"Answer the following questions strictly in {language}, and the answers must follow the Markdown format.\n "
            f"{self.context}"
        )
        logger.debug(f"PROMPT: {prompt}")
        return prompt

    # 定义 prompt_gpt4 属性，用于生成提示信息
    @property
    def prompt_gpt4(self):
        # 定义关键词替换字典
        kvs = {
            "{role}": CONFIG.agent_description or "",
            "{history}": self.history_summary or "",
            "{knowledge}": self.knowledge or "",
            "{language}": CONFIG.language or DEFAULT_LANGUAGE,
            "{ask}": self.context,
        }
        prompt = TalkActionPrompt.FORMATION_LOOSE
        # 根据关键词替换生成提示信息
        for k, v in kvs.items():
            prompt = prompt.replace(k, v)
        logger.info(f"PROMPT: {prompt}")
        return prompt

    # 定义 aask_args 属性，用于生成参数
    @property
    def aask_args(self):
        # 生成系统消息
        language = CONFIG.language or DEFAULT_LANGUAGE
        system_msgs = [
            f"You are {CONFIG.agent_description}.",
            "Your responses should align with the role-play agreement, "
            "maintaining the character's persona and habits. When faced with unrelated questions, playfully "
            "decline to answer without revealing your AI nature to preserve the character's image.",
            "If the information is insufficient, you can search in the context or knowledge.",
            f"Answer the following questions strictly in {language}, and the answers must follow the Markdown format.",
        ]
        format_msgs = []
        # 如果有知识，添加到格式化消息中
        if self.knowledge:
            format_msgs.append({"role": "assistant", "content": self.knowledge})
        # 如果有历史总结，添加到格式化消息中
        if self.history_summary:
            format_msgs.append({"role": "assistant", "content": self.history_summary})
        return self.context, format_msgs, system_msgs

    # 定义 run 方法，用于执行动作
    async def run(self, with_message=None, **kwargs) -> Message:
        # 获取参数
        msg, format_msgs, system_msgs = self.aask_args
        # 调用 llm 的 aask 方法，获取回复
        rsp = await self.llm.aask(msg=msg, format_msgs=format_msgs, system_msgs=system_msgs)
        self.rsp = Message(content=rsp, role="assistant", cause_by=self)
        return self.rsp

# 定义 TalkActionPrompt 类
class TalkActionPrompt:
    # 定义 FORMATION 常量
    FORMATION = """Formation: "Capacity and role" defines the role you are currently playing;
  "[HISTORY_BEGIN]" and "[HISTORY_END]" tags enclose the historical conversation;
  "[KNOWLEDGE_BEGIN]" and "[KNOWLEDGE_END]" tags enclose the knowledge may help for your responses;
  "Statement" defines the work detail you need to complete at this stage;
  "[ASK_BEGIN]" and [ASK_END] tags enclose the questions;
  "Constraint" defines the conditions that your responses must comply with.
  "Personality" defines your language style。
  "Insight" provides a deeper understanding of the characters' inner traits.
  "Initial" defines the initial setup of a character.

Capacity and role: {role}
Statement: Your responses should align with the role-play agreement, maintaining the
 character's persona and habits. When faced with unrelated questions, playfully decline to answer without revealing
 your AI nature to preserve the character's image.

[HISTORY_BEGIN]

{history}

[HISTORY_END]

[KNOWLEDGE_BEGIN]

{knowledge}

[KNOWLEDGE_END]

Statement: If the information is insufficient, you can search in the historical conversation or knowledge.
Statement: Unless you are a language professional, answer the following questions strictly in {language}
, and the answers must follow the Markdown format. Strictly excluding any tag likes "[HISTORY_BEGIN]"
, "[HISTORY_END]", "[KNOWLEDGE_BEGIN]", "[KNOWLEDGE_END]" in responses.
 

{ask}
"""

    # 定义 FORMATION_LOOSE 常量
    FORMATION_LOOSE = """Formation: "Capacity and role" defines the role you are currently playing;
  "[HISTORY_BEGIN]" and "[HISTORY_END]" tags enclose the historical conversation;
  "[KNOWLEDGE_BEGIN]" and "[KNOWLEDGE_END]" tags enclose the knowledge may help for your responses;
  "Statement" defines the work detail you need to complete at this stage;
  "Constraint" defines the conditions that your responses must comply with.
  "Personality" defines your language style。
  "Insight" provides a deeper understanding of the characters' inner traits.
  "Initial" defines the initial setup of a character.

Capacity and role: {role}
Statement: Your responses should maintaining the character's persona and habits. When faced with unrelated questions
, playfully decline to answer without revealing your AI nature to preserve the character's image. 

[HISTORY_BEGIN]

{history}

[HISTORY_END]

[KNOWLEDGE_BEGIN]

{knowledge}

[KNOWLEDGE_END]

Statement: If the information is insufficient, you can search in the historical conversation or knowledge.
Statement: Unless you are a language professional, answer the following questions strictly in {language}
, and the answers must follow the Markdown format. Strictly excluding any tag likes "[HISTORY_BEGIN]"
, "[HISTORY_END]", "[KNOWLEDGE_BEGIN]", "[KNOWLEDGE_END]" in responses.


{ask}
"""

```