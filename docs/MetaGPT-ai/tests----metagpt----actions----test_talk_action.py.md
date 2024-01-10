# `MetaGPT\tests\metagpt\actions\test_talk_action.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/28
@Author  : mashenquan
@File    : test_talk_action.py
"""

# 导入 pytest 模块
import pytest

# 导入自定义模块
from metagpt.actions.talk_action import TalkAction
from metagpt.config import CONFIG
from metagpt.schema import Message

# 使用 pytest.mark.asyncio 标记异步测试
@pytest.mark.asyncio
# 参数化测试用例
@pytest.mark.parametrize(
    ("agent_description", "language", "context", "knowledge", "history_summary"),
    [
        (
            "mathematician",
            "English",
            "How old is Susie?",
            "Susie is a girl born in 2011/11/14. Today is 2023/12/3",
            "balabala... (useless words)",
        ),
        (
            "mathematician",
            "Chinese",
            "Does Susie have an apple?",
            "Susie is a girl born in 2011/11/14. Today is 2023/12/3",
            "Susie had an apple, and she ate it right now",
        ),
    ],
)
# 定义测试函数
async def test_prompt(agent_description, language, context, knowledge, history_summary):
    # 设置全局配置
    CONFIG.agent_description = agent_description
    CONFIG.language = language

    # 创建 TalkAction 对象
    action = TalkAction(context=context, knowledge=knowledge, history_summary=history_summary)
    # 断言 prompt 中不包含 "{"
    assert "{" not in action.prompt
    # 断言 prompt_gpt4 中不包含 "{"
    assert "{" not in action.prompt_gpt4

    # 运行测试
    rsp = await action.run()
    # 断言返回结果非空且为 Message 类型
    assert rsp
    assert isinstance(rsp, Message)

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```