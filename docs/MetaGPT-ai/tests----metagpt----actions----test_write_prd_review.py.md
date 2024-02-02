# `MetaGPT\tests\metagpt\actions\test_write_prd_review.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : test_write_prd_review.py
"""
# 导入 pytest 模块
import pytest

# 导入需要测试的模块
from metagpt.actions.write_prd_review import WritePRDReview

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_write_prd_review():
    # 定义产品需求文档内容
    prd = """
    Introduction: This is a new feature for our product.
    Goals: The goal is to improve user engagement.
    User Scenarios: The expected user group is millennials who like to use social media.
    Requirements: The feature needs to be interactive and user-friendly.
    Constraints: The feature needs to be implemented within 2 months.
    Mockups: There will be a new button on the homepage that users can click to access the feature.
    Metrics: We will measure the success of the feature by user engagement metrics.
    Timeline: The feature should be ready for testing in 1.5 months.
    """

    # 创建 WritePRDReview 对象
    write_prd_review = WritePRDReview(name="write_prd_review")

    # 运行产品需求文档生成函数，并获取结果
    prd_review = await write_prd_review.run(prd)

    # 检查生成的产品需求文档评审是否为字符串且不为空
    assert isinstance(prd_review, str)
    assert len(prd_review) > 0

# 如果当前脚本为主程序，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```