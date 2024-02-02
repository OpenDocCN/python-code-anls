# `MetaGPT\tests\metagpt\tools\test_moderation.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/26 14:46
@Author  : zhanglei
@File    : test_moderation.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.tools.moderation 模块中导入 Moderation 类
from metagpt.tools.moderation import Moderation

# 使用 pytest.mark.asyncio 装饰器标记异步测试
@pytest.mark.asyncio
# 使用 pytest.mark.parametrize 装饰器传入参数化测试的参数
@pytest.mark.parametrize(
    ("content",),
    [
        [
            ["I will kill you", "The weather is really nice today", "I want to hit you"],
        ]
    ],
)
# 定义测试函数 test_amoderation，传入参数 content
async def test_amoderation(content):
    # 前提条件
    assert CONFIG.OPENAI_API_KEY and CONFIG.OPENAI_API_KEY != "YOUR_API_KEY"
    assert not CONFIG.OPENAI_API_TYPE
    assert CONFIG.OPENAI_API_MODEL

    # 创建 Moderation 对象
    moderation = Moderation()
    # 调用 amoderation 方法进行内容审核
    results = await moderation.amoderation(content=content)
    # 断言结果为列表
    assert isinstance(results, list)
    # 断言结果长度与输入内容长度相同
    assert len(results) == len(content)

    # 调用 amoderation_with_categories 方法进行内容审核
    results = await moderation.amoderation_with_categories(content=content)
    # 断言结果为列表
    assert isinstance(results, list)
    # 断言结果不为空
    assert results
    # 遍历结果列表
    for m in results:
        # 断言结果中包含 "flagged" 字段
        assert "flagged" in m
        # 断言结果中包含 "true_categories" 字段
        assert "true_categories" in m

# 当文件被直接运行时执行测试
if __name__ == "__main__":
    # 运行 pytest 测试，并输出结果
    pytest.main([__file__, "-s"])

```