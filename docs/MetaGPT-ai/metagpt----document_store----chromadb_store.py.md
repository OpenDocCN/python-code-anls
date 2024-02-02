# `MetaGPT\metagpt\document_store\chromadb_store.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/29 14:46
@Author  : alexanderwu
@File    : chromadb_store.py
"""
# 导入 chromadb 模块
import chromadb

# 定义 ChromaStore 类
class ChromaStore:
    """If inherited from BaseStore, or importing other modules from metagpt, a Python exception occurs, which is strange."""

    # 初始化方法，创建 ChromaStore 实例时调用
    def __init__(self, name):
        # 创建 chromadb 客户端
        client = chromadb.Client()
        # 创建指定名称的集合
        collection = client.create_collection(name)
        # 设置实例属性
        self.client = client
        self.collection = collection

    # 查询方法
    def search(self, query, n_results=2, metadata_filter=None, document_filter=None):
        # 使用查询条件查询集合
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=metadata_filter,  # optional filter
            where_document=document_filter,  # optional filter
        )
        return results

    # 持久化方法，抛出未实现异常
    def persist(self):
        """Chroma recommends using server mode and not persisting locally."""
        raise NotImplementedError

    # 写入方法，用于更新文档
    def write(self, documents, metadatas, ids):
        # 添加文档、元数据和 ID 到集合
        return self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    # 添加方法，用于添加单个文档
    def add(self, document, metadata, _id):
        # 添加单个文档、元数据和 ID 到集合
        return self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[_id],
        )

    # 删除方法，用于删除指定 ID 的文档
    def delete(self, _id):
        return self.collection.delete([_id])

```