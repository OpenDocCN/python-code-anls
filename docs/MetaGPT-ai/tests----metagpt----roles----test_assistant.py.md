# `MetaGPT\tests\metagpt\roles\test_assistant.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/25
@Author  : mashenquan
@File    : test_asssistant.py
@Desc    : Used by AgentStore.
"""

# 导入所需的模块
import pytest
from pydantic import BaseModel

# 导入自定义模块
from metagpt.actions.skill_action import SkillAction
from metagpt.actions.talk_action import TalkAction
from metagpt.config import CONFIG
from metagpt.memory.brain_memory import BrainMemory
from metagpt.roles.assistant import Assistant
from metagpt.schema import Message
from metagpt.utils.common import any_to_str

# 使用 pytest 的异步测试标记
@pytest.mark.asyncio
async def test_run():
    # 设置语言为中文
    CONFIG.language = "Chinese"

    # 定义输入数据模型
    class Input(BaseModel):
        memory: BrainMemory
        language: str
        agent_description: str
        cause_by: str

    # 定义测试输入
    inputs = [
        {
            "memory": {
                "history": [
                    {
                        "content": "who is tulin",
                        "role": "user",
                        "id": "1",
                    },
                    {"content": "The one who eaten a poison apple.", "role": "assistant"},
                ],
                "knowledge": [{"content": "tulin is a scientist."}],
                "last_talk": "Do you have a poison apple?",
            },
            "language": "English",
            "agent_description": "chatterbox",
            "cause_by": any_to_str(TalkAction),
        },
        {
            "memory": {
                "history": [
                    {
                        "content": "can you draw me an picture?",
                        "role": "user",
                        "id": "1",
                    },
                    {"content": "Yes, of course. What do you want me to draw", "role": "assistant"},
                ],
                "knowledge": [{"content": "tulin is a scientist."}],
                "last_talk": "Draw me an apple.",
            },
            "language": "English",
            "agent_description": "painter",
            "cause_by": any_to_str(SkillAction),
        },
    ]
    
    # 设置 agent_skills
    CONFIG.agent_skills = [
        {"id": 1, "name": "text_to_speech", "type": "builtin", "config": {}, "enabled": True},
        {"id": 2, "name": "text_to_image", "type": "builtin", "config": {}, "enabled": True},
        {"id": 3, "name": "ai_call", "type": "builtin", "config": {}, "enabled": True},
        {"id": 3, "name": "data_analysis", "type": "builtin", "config": {}, "enabled": True},
        {"id": 5, "name": "crawler", "type": "builtin", "config": {"engine": "ddg"}, "enabled": True},
        {"id": 6, "name": "knowledge", "type": "builtin", "config": {}, "enabled": True},
        {"id": 6, "name": "web_search", "type": "builtin", "config": {}, "enabled": True},
    ]

    # 遍历测试输入
    for i in inputs:
        seed = Input(**i)
        CONFIG.language = seed.language
        CONFIG.agent_description = seed.agent_description
        role = Assistant(language="Chinese")
        role.memory = seed.memory  # Restore historical conversation content.
        while True:
            has_action = await role.think()
            if not has_action:
                break
            msg: Message = await role.act()
            # logger.info(msg)
            assert msg
            assert msg.cause_by == seed.cause_by
            assert msg.content


@pytest.mark.parametrize(
    "memory",
    [
        {
            "history": [
                {
                    "content": "can you draw me an picture?",
                    "role": "user",
                    "id": "1",
                },
                {"content": "Yes, of course. What do you want me to draw", "role": "assistant"},
            ],
            "knowledge": [{"content": "tulin is a scientist."}],
            "last_talk": "Draw me an apple.",
        }
    ],
)
@pytest.mark.asyncio
async def test_memory(memory):
    role = Assistant()
    role.load_memory(memory)

    val = role.get_memory()
    assert val

    await role.talk("draw apple")

    agent_skills = CONFIG.agent_skills
    CONFIG.agent_skills = []
    try:
        await role.think()
    finally:
        CONFIG.agent_skills = agent_skills
    assert isinstance(role.rc.todo, TalkAction)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```