# `MetaGPT\tests\metagpt\utils\test_read_docx.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 16:02
@Author  : alexanderwu
@File    : test_read_docx.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.const 模块中导入 METAGPT_ROOT 变量
from metagpt.const import METAGPT_ROOT
# 从 metagpt.utils.read_document 模块中导入 read_docx 函数
from metagpt.utils.read_document import read_docx

# 标记该测试用例为跳过状态，并添加跳过的原因链接
@pytest.mark.skip  # https://copyprogramming.com/howto/python-docx-error-opening-file-bad-magic-number-for-file-header-eoferror
class TestReadDocx:
    # 定义测试用例 test_read_docx
    def test_read_docx(self):
        # 定义 docx_sample 变量为 docx 文件路径
        docx_sample = METAGPT_ROOT / "tests/data/docx_for_test.docx"
        # 调用 read_docx 函数读取 docx 文件内容
        docx = read_docx(docx_sample)
        # 断言读取的 docx 文件内容长度为 6
        assert len(docx) == 6

```