# `MetaGPT\tests\metagpt\document_store\test_qdrant_store.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/11 21:08
@Author  : hezhaozhao
@File    : test_qdrant_store.py
"""
# 导入随机数模块
import random

# 导入 qdrant_client 模块中的相关类
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    Range,
    VectorParams,
)

# 导入自定义的 qdrant_store 模块中的相关类
from metagpt.document_store.qdrant_store import QdrantConnection, QdrantStore

# 设置随机数种子
seed_value = 42
random.seed(seed_value)

# 生成随机向量
vectors = [[random.random() for _ in range(2)] for _ in range(10)]

# 生成点结构列表
points = [
    PointStruct(id=idx, vector=vector, payload={"color": "red", "rand_number": idx % 10})
    for idx, vector in enumerate(vectors)
]

# 定义一个函数，用于比较两个数值是否几乎相等
def assert_almost_equal(actual, expected):
    delta = 1e-10
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for ac, exp in zip(actual, expected):
            assert abs(ac - exp) <= delta, f"{ac} is not within {delta} of {exp}"
    else:
        assert abs(actual - expected) <= delta, f"{actual} is not within {delta} of {expected}"

# 定义测试函数
def test_qdrant_store():
    # 创建 Qdrant 连接对象
    qdrant_connection = QdrantConnection(memory=True)
    # 设置向量参数
    vectors_config = VectorParams(size=2, distance=Distance.COSINE)
    # 创建 Qdrant 存储对象
    qdrant_store = QdrantStore(qdrant_connection)
    # 创建集合 "Book"，如果已存在则强制重新创建
    qdrant_store.create_collection("Book", vectors_config, force_recreate=True)
    # 断言集合 "Book" 是否存在
    assert qdrant_store.has_collection("Book") is True
    # 删除集合 "Book"
    qdrant_store.delete_collection("Book")
    # 断言集合 "Book" 是否存在
    assert qdrant_store.has_collection("Book") is False
    # 创建集合 "Book"
    qdrant_store.create_collection("Book", vectors_config)
    # 断言集合 "Book" 是否存在
    assert qdrant_store.has_collection("Book") is True
    # 向集合 "Book" 添加点结构列表
    qdrant_store.add("Book", points)
    # 在集合 "Book" 中搜索指定向量
    results = qdrant_store.search("Book", query=[1.0, 1.0])
    # 断言搜索结果
    assert results[0]["id"] == 2
    assert_almost_equal(results[0]["score"], 0.999106722578389)
    assert results[1]["id"] == 7
    assert_almost_equal(results[1]["score"], 0.9961650411397226)
    # 在集合 "Book" 中搜索指定向量，并返回向量
    results = qdrant_store.search("Book", query=[1.0, 1.0], return_vector=True)
    # 断言搜索结果
    assert results[0]["id"] == 2
    assert_almost_equal(results[0]["score"], 0.999106722578389)
    assert_almost_equal(results[0]["vector"], [0.7363563179969788, 0.6765939593315125])
    assert results[1]["id"] == 7
    assert_almost_equal(results[1]["score"], 0.9961650411397226)
    assert_almost_equal(results[1]["vector"], [0.7662628889083862, 0.6425272226333618])
    # 在集合 "Book" 中搜索指定向量，并应用过滤条件
    results = qdrant_store.search(
        "Book",
        query=[1.0, 1.0],
        query_filter=Filter(must=[FieldCondition(key="rand_number", range=Range(gte=8))]),
    )
    # 断言搜索结果
    assert results[0]["id"] == 8
    assert_almost_equal(results[0]["score"], 0.9100373450784073)
    assert results[1]["id"] == 9
    assert_almost_equal(results[1]["score"], 0.7127610621127889)
    # 在集合 "Book" 中搜索指定向量，并应用过滤条件，同时返回向量
    results = qdrant_store.search(
        "Book",
        query=[1.0, 1.0],
        query_filter=Filter(must=[FieldCondition(key="rand_number", range=Range(gte=8))]),
        return_vector=True,
    )
    # 断言搜索结果
    assert_almost_equal(results[0]["vector"], [0.35037919878959656, 0.9366079568862915])
    assert_almost_equal(results[1]["vector"], [0.9999677538871765, 0.00802854634821415])

```