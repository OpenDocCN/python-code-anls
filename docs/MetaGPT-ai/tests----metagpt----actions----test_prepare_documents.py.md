# `MetaGPT\tests\metagpt\actions\test_prepare_documents.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/6
@Author  : mashenquan
@File    : test_prepare_documents.py
@Desc: Unit test for prepare_documents.py
"""
# 导入 pytest 模块
import pytest

# 导入 PrepareDocuments 类、CONFIG 常量、DOCS_FILE_REPO 常量、REQUIREMENT_FILENAME 常量、Message 类和 FileRepository 类
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.config import CONFIG
from metagpt.const import DOCS_FILE_REPO, REQUIREMENT_FILENAME
from metagpt.schema import Message
from metagpt.utils.file_repository import FileRepository

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_prepare_documents():
    # 创建一个 Message 对象
    msg = Message(content="New user requirements balabala...")

    # 如果 CONFIG.git_repo 存在，则删除仓库并将其置为 None
    if CONFIG.git_repo:
        CONFIG.git_repo.delete_repository()
        CONFIG.git_repo = None

    # 运行 PrepareDocuments 类的 run 方法，传入消息列表
    await PrepareDocuments().run(with_messages=[msg])
    # 断言 CONFIG.git_repo 存在
    assert CONFIG.git_repo
    # 获取指定路径下的文件，并断言其存在
    doc = await FileRepository.get_file(filename=REQUIREMENT_FILENAME, relative_path=DOCS_FILE_REPO)
    assert doc
    # 断言文件内容与消息内容相同
    assert doc.content == msg.content

```