# `MetaGPT\tests\metagpt\actions\test_generate_questions.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 00:26
@Author  : fisherdeng
@File    : test_generate_questions.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.actions.generate_questions 模块中导入 GenerateQuestions 类
from metagpt.actions.generate_questions import GenerateQuestions
# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger

# 定义一个包含主题和记录的上下文字符串
context = """
## topic
如何做一个生日蛋糕

## record
我认为应该先准备好材料，然后再开始做蛋糕。
"""

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_generate_questions():
    # 创建 GenerateQuestions 类的实例
    action = GenerateQuestions()
    # 运行异步方法，获取返回结果
    rsp = await action.run(context)
    # 记录返回结果的内容
    logger.info(f"{rsp.content=}")

    # 断言返回结果中包含 "Questions" 字符串
    assert "Questions" in rsp.content
    # 断言返回结果中包含 "1." 字符串
    assert "1." in rsp.content

```