# `MetaGPT\tests\metagpt\document_store\test_document.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/11 19:46
@Author  : alexanderwu
@File    : test_document.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.const 模块导入 METAGPT_ROOT 常量
from metagpt.const import METAGPT_ROOT
# 从 metagpt.document 模块导入 IndexableDocument 类
from metagpt.document import IndexableDocument

# 定义测试用例
CASES = [
    ("requirements.txt", None, None, 0),
    # ("cases/faq.csv", "Question", "Answer", 1),
    # ("cases/faq.json", "Question", "Answer", 1),
    # ("docx/faq.docx", None, None, 1),
    # ("cases/faq.pdf", None, None, 0),  # 这是因为pdf默认没有分割段落
    # ("cases/faq.txt", None, None, 0),  # 这是因为txt按照256分割段落
]

# 使用 pytest.mark.parametrize 装饰器进行参数化测试
@pytest.mark.parametrize("relative_path, content_col, meta_col, threshold", CASES)
# 定义测试函数
def test_document(relative_path, content_col, meta_col, threshold):
    # 从文件路径创建 IndexableDocument 对象
    doc = IndexableDocument.from_path(METAGPT_ROOT / relative_path, content_col, meta_col)
    # 获取文档和元数据
    rsp = doc.get_docs_and_metadatas()
    # 断言文档数量大于阈值
    assert len(rsp[0]) > threshold
    # 断言元数据数量大于阈值
    assert len(rsp[1]) > threshold

```