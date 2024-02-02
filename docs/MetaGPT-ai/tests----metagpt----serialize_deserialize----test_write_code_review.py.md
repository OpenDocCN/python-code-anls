# `MetaGPT\tests\metagpt\serialize_deserialize\test_write_code_review.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : unittest of WriteCodeReview SerDeser

# 导入 pytest 模块
import pytest

# 从 metagpt.actions 模块中导入 WriteCodeReview 类
from metagpt.actions import WriteCodeReview
# 从 metagpt.schema 模块中导入 CodingContext 和 Document 类
from metagpt.schema import CodingContext, Document

# 标记该测试函数为异步函数
@pytest.mark.asyncio
async def test_write_code_review_deserialize():
    # 定义代码内容
    code_content = """
def div(a: int, b: int = 0):
    return a / b
"""
    # 创建 CodingContext 对象
    context = CodingContext(
        filename="test_op.py",
        design_doc=Document(content="divide two numbers"),
        code_doc=Document(content=code_content),
    )

    # 创建 WriteCodeReview 对象
    action = WriteCodeReview(context=context)
    # 序列化 WriteCodeReview 对象
    serialized_data = action.model_dump()
    # 断言序列化后的数据中的 name 属性为 "WriteCodeReview"
    assert serialized_data["name"] == "WriteCodeReview"

    # 根据序列化后的数据创建新的 WriteCodeReview 对象
    new_action = WriteCodeReview(**serialized_data)

    # 断言新的 WriteCodeReview 对象的 name 属性为 "WriteCodeReview"
    assert new_action.name == "WriteCodeReview"
    # 异步运行新的 WriteCodeReview 对象
    await new_action.run()

```