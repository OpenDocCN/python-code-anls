# `.\DB-GPT-src\tests\intetration_tests\graph_store\test_tugraph_store.py`

```py
# test_tugraph_store.py 文件用于测试 TuGraphStore 类中的功能

import pytest  # 导入 pytest 测试框架

from dbgpt.storage.graph_store.tugraph_store import TuGraphStore  # 导入 TuGraphStore 类


class TuGraphStoreConfig:
    def __init__(self, name):
        self.name = name  # 初始化 TuGraphStoreConfig 类，设置名称属性


@pytest.fixture(scope="module")
def store():
    config = TuGraphStoreConfig(name="TestGraph")
    store = TuGraphStore(config=config)  # 创建 TuGraphStore 实例
    yield store  # 返回 store 实例，作为测试用例的 fixture
    store.conn.close()  # 在测试完成后关闭连接


def test_insert_and_get_triplets(store):
    store.insert_triplet("A", "0", "A")  # 插入三元组
    store.insert_triplet("A", "1", "A")
    store.insert_triplet("A", "2", "B")
    store.insert_triplet("B", "3", "C")
    store.insert_triplet("B", "4", "D")
    store.insert_triplet("C", "5", "D")
    store.insert_triplet("B", "6", "E")
    store.insert_triplet("F", "7", "E")
    store.insert_triplet("E", "8", "F")
    triplets = store.get_triplets("A")  # 获取以 "A" 为主体的三元组
    assert len(triplets) == 3  # 断言三元组数量为 3
    triplets = store.get_triplets("B")  # 获取以 "B" 为主体的三元组
    assert len(triplets) == 3  # 断言三元组数量为 3
    triplets = store.get_triplets("C")  # 获取以 "C" 为主体的三元组
    assert len(triplets) == 1  # 断言三元组数量为 1
    triplets = store.get_triplets("D")  # 获取以 "D" 为主体的三元组
    assert len(triplets) == 0  # 断言三元组数量为 0
    triplets = store.get_triplets("E")  # 获取以 "E" 为主体的三元组
    assert len(triplets) == 1  # 断言三元组数量为 1
    triplets = store.get_triplets("F")  # 获取以 "F" 为主体的三元组
    assert len(triplets) == 1  # 断言三元组数量为 1


def test_query(store):
    query = "MATCH (n)-[r]->(n1) return n,n1,r limit 3"  # 定义查询语句
    result = store.query(query)  # 执行查询
    v_c = result.vertex_count  # 获取顶点计数
    e_c = result.edge_count  # 获取边计数
    assert v_c == 2 and e_c == 3  # 断言顶点数为 2，边数为 3


def test_explore(store):
    subs = ["A", "B"]  # 定义主体列表
    result = store.explore(subs, depth=2, fan=None, limit=10)  # 探索图数据
    v_c = result.vertex_count  # 获取顶点计数
    e_c = result.edge_count  # 获取边计数
    assert v_c == 2 and e_c == 3  # 断言顶点数为 2，边数为 3


# def test_delete_triplet(store):
#     subj = "A"
#     rel = "0"
#     obj = "B"
#     store.delete_triplet(subj, rel, obj)
#     triplets = store.get_triplets(subj)
#     assert len(triplets) == 0
```