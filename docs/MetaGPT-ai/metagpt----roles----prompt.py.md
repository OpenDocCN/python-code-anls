# `MetaGPT\metagpt\roles\prompt.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/18 22:43
@Author  : alexanderwu
@File    : prompt.py
"""
# 导入枚举类型
from enum import Enum

# 定义常量 PREFIX，用于存储提示信息
PREFIX = """Answer the questions to the best of your ability. You can use the following tools:"""

# 定义常量 FORMAT_INSTRUCTIONS，用于存储格式说明
FORMAT_INSTRUCTIONS = """Please follow the format below:

Question: The input question you need to answer
Thoughts: You should always think about how to do it
Action: The action to be taken, should be one from [{tool_names}]
Action Input: Input for the action
Observation: Result of the action
... (This Thoughts/Action/Action Input/Observation can be repeated N times)
Thoughts: I now know the final answer
Final Answer: The final answer to the original input question"""

# 定义常量 SUFFIX，用于存储结束提示信息
SUFFIX = """Let's begin!

Question: {input}
Thoughts: {agent_scratchpad}"""

```