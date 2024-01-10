# `MetaGPT\metagpt\actions\execute_task.py`

```

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
"""
@Time    : 2023/9/13 12:26
@Author  : femto Zheng
@File    : execute_task.py
"""
# 文件的注释信息，包括时间、作者和文件名

# 导入需要的模块
from metagpt.actions import Action
from metagpt.schema import Message

# 定义一个名为 ExecuteTask 的类，继承自 Action 类
class ExecuteTask(Action):
    # 类属性：任务名称为 ExecuteTask
    name: str = "ExecuteTask"
    # 类属性：上下文为消息列表
    context: list[Message] = []

    # 异步方法：运行任务
    async def run(self, *args, **kwargs):
        # 空的运行方法，暂时不做任何操作
        pass

```