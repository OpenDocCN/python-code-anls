# `MetaGPT\tests\metagpt\document_store\test_faiss_store.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/27 20:20
@Author  : alexanderwu
@File    : test_faiss_store.py
"""

# 导入 pytest 模块
import pytest

# 导入相关模块和类
from metagpt.const import EXAMPLE_PATH
from metagpt.document_store import FaissStore
from metagpt.logs import logger
from metagpt.roles import Sales

# 异步测试函数，测试 FaissStore 对 JSON 文件的搜索
@pytest.mark.asyncio
async def test_search_json():
    # 创建 FaissStore 对象，传入 JSON 文件路径
    store = FaissStore(EXAMPLE_PATH / "example.json")
    # 创建 Sales 角色对象，传入配置信息和存储对象
    role = Sales(profile="Sales", store=store)
    # 设置查询字符串
    query = "Which facial cleanser is good for oily skin?"
    # 运行角色的搜索方法，获取结果
    result = await role.run(query)
    # 记录结果
    logger.info(result)

# 异步测试函数，测试 FaissStore 对 XLSX 文件的搜索
@pytest.mark.asyncio
async def test_search_xlsx():
    # 创建 FaissStore 对象，传入 XLSX 文件路径
    store = FaissStore(EXAMPLE_PATH / "example.xlsx")
    # 创建 Sales 角色对象，传入配置信息和存储对象
    role = Sales(profile="Sales", store=store)
    # 设置查询字符串
    query = "Which facial cleanser is good for oily skin?"
    # 运行角色的搜索方法，获取结果
    result = await role.run(query)
    # 记录结果
    logger.info(result)

# 异步测试函数，测试 FaissStore 对 XLSX 文件的写入
@pytest.mark.asyncio
async def test_write():
    # 创建 FaissStore 对象，传入 XLSX 文件路径和元数据列名、内容列名
    store = FaissStore(EXAMPLE_PATH / "example.xlsx", meta_col="Answer", content_col="Question")
    # 调用写入方法，获取 FaissStore 对象
    _faiss_store = store.write()
    # 断言 FaissStore 对象的 docstore 属性存在
    assert _faiss_store.docstore
    # 断言 FaissStore 对象的 index 属性存在
    assert _faiss_store.index

```