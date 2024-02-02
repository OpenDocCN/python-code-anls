# `MetaGPT\metagpt\tools\search_engine_meilisearch.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/22 21:33
@Author  : alexanderwu
@File    : search_engine_meilisearch.py
"""

# 导入所需的模块
from typing import List
import meilisearch
from meilisearch.index import Index
from metagpt.utils.exceptions import handle_exception

# 定义数据源类，包含名称和URL
class DataSource:
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url

# 定义 MeilisearchEngine 类
class MeilisearchEngine:
    # 初始化方法，接受 URL 和 token 参数
    def __init__(self, url, token):
        # 创建 Meilisearch 客户端
        self.client = meilisearch.Client(url, token)
        self._index: Index = None

    # 设置索引方法
    def set_index(self, index):
        self._index = index

    # 添加文档方法，接受数据源和文档列表作为参数
    def add_documents(self, data_source: DataSource, documents: List[dict]):
        # 构建索引名称
        index_name = f"{data_source.name}_index"
        # 如果索引名称不存在，则创建索引
        if index_name not in self.client.get_indexes():
            self.client.create_index(uid=index_name, options={"primaryKey": "id"})
        # 获取索引对象
        index = self.client.get_index(index_name)
        # 添加文档到索引
        index.add_documents(documents)
        # 设置索引
        self.set_index(index)

    # 搜索方法，接受查询字符串作为参数
    @handle_exception(exception_type=Exception, default_return=[])
    def search(self, query):
        # 执行搜索并返回结果
        search_results = self._index.search(query)
        return search_results["hits"]

```