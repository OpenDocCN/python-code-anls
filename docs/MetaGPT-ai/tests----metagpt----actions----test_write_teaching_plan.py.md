# `MetaGPT\tests\metagpt\actions\test_write_teaching_plan.py`

```py

#!/usr/bin/env python
# 指定解释器为python，使得脚本可以直接执行

# -*- coding: utf-8 -*-
# 指定编码格式为UTF-8，以支持中文字符

"""
@Time    : 2023/7/28 17:25
@Author  : mashenquan
@File    : test_write_teaching_plan.py
"""
# 文件的注释信息，包括时间、作者、文件名等

import pytest
# 导入pytest模块，用于测试

from metagpt.actions.write_teaching_plan import WriteTeachingPlanPart
# 从metagpt.actions.write_teaching_plan模块中导入WriteTeachingPlanPart类

@pytest.mark.asyncio
# 使用pytest的装饰器标记为异步测试
@pytest.mark.parametrize(
    ("topic", "context"),
    [("Title", "Lesson 1: Learn to draw an apple."), ("Teaching Content", "Lesson 1: Learn to draw an apple.")],
)
# 使用pytest的参数化装饰器，传入参数列表

async def test_write_teaching_plan_part(topic, context):
    # 定义测试函数，传入topic和context参数
    action = WriteTeachingPlanPart(topic=topic, context=context)
    # 创建WriteTeachingPlanPart对象
    rsp = await action.run()
    # 调用对象的run方法，返回结果
    assert rsp
    # 断言结果为真

if __name__ == "__main__":
    # 如果当前脚本为主程序
    pytest.main([__file__, "-s"])
    # 运行pytest测试，输出详细信息

```