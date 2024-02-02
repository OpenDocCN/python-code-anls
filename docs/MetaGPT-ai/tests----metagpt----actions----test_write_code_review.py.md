# `MetaGPT\tests\metagpt\actions\test_write_code_review.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : test_write_code_review.py
"""
# 导入 pytest 模块
import pytest

# 导入需要测试的模块和类
from metagpt.actions.write_code_review import WriteCodeReview
from metagpt.schema import CodingContext, Document

# 标记异步测试
@pytest.mark.asyncio
async def test_write_code_review(capfd):
    # 定义测试用例中的代码片段
    code = """
def add(a, b):
    return a + 
"""
    # 创建编码上下文对象，包括文件名、设计文档和代码文档
    context = CodingContext(
        filename="math.py", design_doc=Document(content="编写一个从a加b的函数，返回a+b"), code_doc=Document(content=code)
    )

    # 运行代码评审操作
    context = await WriteCodeReview(context=context).run()

    # 检查生成的代码评审是否为字符串
    assert isinstance(context.code_doc.content, str)
    assert len(context.code_doc.content) > 0

    # 读取并打印输出内容
    captured = capfd.readouterr()
    print(f"输出内容: {captured.out}")


# @pytest.mark.asyncio
# async def test_write_code_review_directly():
#     code = SEARCH_CODE_SAMPLE
#     write_code_review = WriteCodeReview("write_code_review")
#     review = await write_code_review.run(code)
#     logger.info(review)



```