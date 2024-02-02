# `MetaGPT\tests\metagpt\tools\test_search_engine_meilisearch.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/27 22:18
@Author  : alexanderwu
@File    : test_search_engine_meilisearch.py
"""
# 导入所需的模块
import subprocess
import time
import pytest
from metagpt.logs import logger
from metagpt.tools.search_engine_meilisearch import DataSource, MeilisearchEngine

# 设置 MASTER_KEY
MASTER_KEY = "116Qavl2qpCYNEJNv5-e0RC9kncev1nr1gt7ybEGVLk"

# 定义 fixture，用于启动和关闭 Meilisearch 服务器
@pytest.fixture()
def search_engine_server():
    # 启动 Meilisearch 服务器
    meilisearch_process = subprocess.Popen(["meilisearch", "--master-key", f"{MASTER_KEY}"], stdout=subprocess.PIPE)
    time.sleep(3)
    yield
    # 关闭 Meilisearch 服务器
    meilisearch_process.terminate()
    meilisearch_process.wait()

# 测试 Meilisearch 引擎
def test_meilisearch(search_engine_server):
    # 创建 Meilisearch 引擎对象
    search_engine = MeilisearchEngine(url="http://localhost:7700", token=MASTER_KEY)

    # 创建一个名为"books"的数据源
    books_data_source = DataSource(name="books", url="https://example.com/books")

    # 创建要添加的文档
    documents = [
        {"id": 1, "title": "Book 1", "content": "This is the content of Book 1."},
        {"id": 2, "title": "Book 2", "content": "This is the content of Book 2."},
        {"id": 3, "title": "Book 1", "content": "This is the content of Book 1."},
        {"id": 4, "title": "Book 2", "content": "This is the content of Book 2."},
        {"id": 5, "title": "Book 1", "content": "This is the content of Book 1."},
        {"id": 6, "title": "Book 2", "content": "This is the content of Book 2."},
    ]

    # 将文档添加到搜索引擎中
    search_engine.add_documents(books_data_source, documents)
    # 打印搜索结果
    logger.info(search_engine.search("Book 1"))

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```