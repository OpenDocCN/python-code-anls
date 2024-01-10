# `MetaGPT\metagpt\actions\action_output.py`

```

#!/usr/bin/env python
# 指定解释器为 Python
# coding: utf-8
# 指定编码格式为 UTF-8
"""
@Time    : 2023/7/11 10:03
@Author  : chengmaoyu
@File    : action_output
"""
# 文件注释，包括时间、作者、文件名等信息

from pydantic import BaseModel
# 导入 pydantic 模块中的 BaseModel 类

class ActionOutput:
    content: str
    instruct_content: BaseModel
    # 定义 ActionOutput 类，包括 content 和 instruct_content 两个属性

    def __init__(self, content: str, instruct_content: BaseModel):
        # 定义初始化方法，接受 content 和 instruct_content 两个参数
        self.content = content
        self.instruct_content = instruct_content
        # 将参数赋值给实例属性

```