# `MetaGPT\metagpt\document_store\lancedb_store.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/9 15:42
@Author  : unkn-wn (Leon Yee)
@File    : lancedb_store.py
"""
# 导入所需的模块
import os
import shutil

import lancedb

# 定义 LanceStore 类
class LanceStore:
    def __init__(self, name):
        # 连接到 LanceDB 数据库
        db = lancedb.connect("./data/lancedb")
        self.db = db
        self.name = name
        self.table = None

    def search(self, query, n_results=2, metric="L2", nprobes=20, **kwargs):
        # 这里假设 query 是一个向量嵌入
        # kwargs 可以用于可选的过滤
        # .select - 仅搜索指定的列
        # .where - 用于元数据的 SQL 语法过滤（例如 where("price > 100")）
        # .metric - 指定要使用的距离度量
        # .nprobes - 值会以延迟为代价提高召回率（如果存在，则更有可能找到向量）
        if self.table is None:
            raise Exception("Table not created yet, please add data first.")

        results = (
            self.table.search(query)
            .limit(n_results)
            .select(kwargs.get("select"))
            .where(kwargs.get("where"))
            .metric(metric)
            .nprobes(nprobes)
            .to_df()
        )
        return results

    def persist(self):
        raise NotImplementedError

    def write(self, data, metadatas, ids):
        # 此函数类似于 add()，但用于更通用的更新
        # "data" 是嵌入的列表
        # 通过将元数据扩展为数据框插入表中：[{'vector', 'id', 'meta', 'meta2'}, ...]

        documents = []
        for i in range(len(data)):
            row = {"vector": data[i], "id": ids[i]}
            row.update(metadatas[i])
            documents.append(row)

        if self.table is not None:
            self.table.add(documents)
        else:
            self.table = self.db.create_table(self.name, documents)

    def add(self, data, metadata, _id):
        # 此函数用于添加单个文档
        # 假设您传入了单个向量嵌入、元数据和 id

        row = {"vector": data, "id": _id}
        row.update(metadata)

        if self.table is not None:
            self.table.add([row])
        else:
            self.table = self.db.create_table(self.name, [row])

    def delete(self, _id):
        # 此函数通过 id 删除一行
        # LanceDB 删除语法使用 SQL 语法，因此可以使用 "in" 或 "="
        if self.table is None:
            raise Exception("Table not created yet, please add data first")

        if isinstance(_id, str):
            return self.table.delete(f"id = '{_id}'")
        else:
            return self.table.delete(f"id = {_id}")

    def drop(self, name):
        # 此函数删除表（如果存在）。

        path = os.path.join(self.db.uri, name + ".lance")
        if os.path.exists(path):
            shutil.rmtree(path)

```