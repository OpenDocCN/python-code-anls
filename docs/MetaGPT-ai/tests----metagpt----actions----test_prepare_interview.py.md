# `MetaGPT\tests\metagpt\actions\test_prepare_interview.py`

```py

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
"""
@Time    : 2023/9/13 00:26
@Author  : fisherdeng
@File    : test_generate_questions.py
"""
# 文件的时间、作者和名称信息

import pytest
# 导入 pytest 模块

from metagpt.actions.prepare_interview import PrepareInterview
# 从 metagpt.actions.prepare_interview 模块导入 PrepareInterview 类
from metagpt.logs import logger
# 从 metagpt.logs 模块导入 logger 对象

@pytest.mark.asyncio
# 使用 pytest 的 asyncio 标记
async def test_prepare_interview():
    # 定义测试函数 test_prepare_interview
    action = PrepareInterview()
    # 创建 PrepareInterview 实例
    rsp = await action.run("I just graduated and hope to find a job as a Python engineer")
    # 调用 run 方法，传入参数，获取返回结果
    logger.info(f"{rsp.content=}")
    # 记录返回结果的内容

    assert "Questions" in rsp.content
    # 断言返回结果中包含 "Questions"
    assert "1." in rsp.content
    # 断言返回结果中包含 "1."

```