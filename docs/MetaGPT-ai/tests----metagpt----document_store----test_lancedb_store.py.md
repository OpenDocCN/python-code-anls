# `MetaGPT\tests\metagpt\document_store\test_lancedb_store.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/9 15:42
@Author  : unkn-wn (Leon Yee)
@File    : test_lancedb_store.py
"""
# 导入随机模块
import random
# 从metagpt.document_store.lancedb_store模块中导入LanceStore类
from metagpt.document_store.lancedb_store import LanceStore

# 定义测试函数test_lance_store
def test_lance_store():
    # 建立与数据库的连接，以便在存在时删除表
    store = LanceStore("test")

    # 删除名为"test"的表
    store.drop("test")

    # 向数据库中写入数据
    store.write(
        data=[[random.random() for _ in range(100)] for _ in range(2)],
        metadatas=[{"source": "google-docs"}, {"source": "notion"}],
        ids=["doc1", "doc2"],
    )

    # 向数据库中添加数据
    store.add(data=[random.random() for _ in range(100)], metadata={"source": "notion"}, _id="doc3")

    # 在数据库中搜索数据
    result = store.search([random.random() for _ in range(100)], n_results=3)
    # 断言搜索结果的长度为3
    assert len(result) == 3

    # 删除数据库中的数据
    store.delete("doc2")
    # 再次在数据库中搜索数据
    result = store.search(
        [random.random() for _ in range(100)], n_results=3, where="source = 'notion'", metric="cosine"
    )
    # 断言搜索结果的长度为1
    assert len(result) == 1

```