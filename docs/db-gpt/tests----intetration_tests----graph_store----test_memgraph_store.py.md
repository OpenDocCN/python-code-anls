# `.\DB-GPT-src\tests\intetration_tests\graph_store\test_memgraph_store.py`

```py
import pytest  # 导入 pytest 模块

from dbgpt.storage.graph_store.memgraph_store import (  # 从特定路径导入 MemoryGraphStore 和 MemoryGraphStoreConfig 类
    MemoryGraphStore,
    MemoryGraphStoreConfig,
)


@pytest.fixture
def graph_store():  # 定义 pytest 的 fixture，返回一个 MemoryGraphStore 对象
    yield MemoryGraphStore(MemoryGraphStoreConfig())  # 使用默认配置创建 MemoryGraphStore 实例


def test_graph_store(graph_store):  # 定义测试函数 test_graph_store，接受 graph_store 作为参数
    graph_store.insert_triplet("A", "0", "A")  # 向图数据库插入三元组 ("A", "0", "A")
    graph_store.insert_triplet("A", "1", "A")  # 向图数据库插入三元组 ("A", "1", "A")
    graph_store.insert_triplet("A", "2", "B")  # 向图数据库插入三元组 ("A", "2", "B")
    graph_store.insert_triplet("B", "3", "C")  # 向图数据库插入三元组 ("B", "3", "C")
    graph_store.insert_triplet("B", "4", "D")  # 向图数据库插入三元组 ("B", "4", "D")
    graph_store.insert_triplet("C", "5", "D")  # 向图数据库插入三元组 ("C", "5", "D")
    graph_store.insert_triplet("B", "6", "E")  # 向图数据库插入三元组 ("B", "6", "E")
    graph_store.insert_triplet("F", "7", "E")  # 向图数据库插入三元组 ("F", "7", "E")
    graph_store.insert_triplet("E", "8", "F")  # 向图数据库插入三元组 ("E", "8", "F")

    subgraph = graph_store.explore(["A"])  # 在图数据库中探索包含节点 "A" 的子图
    print(f"\n{subgraph.graphviz()}")  # 打印子图的 Graphviz 表示
    assert subgraph.edge_count == 9  # 断言子图的边数为 9 条

    graph_store.delete_triplet("A", "0", "A")  # 从图数据库中删除三元组 ("A", "0", "A")
    graph_store.delete_triplet("B", "4", "D")  # 从图数据库中删除三元组 ("B", "4", "D")
    subgraph = graph_store.explore(["A"])  # 再次探索包含节点 "A" 的子图
    print(f"\n{subgraph.graphviz()}")  # 打印更新后的子图的 Graphviz 表示
    assert subgraph.edge_count == 7  # 断言更新后子图的边数为 7 条

    triplets = graph_store.get_triplets("B")  # 获取图数据库中所有以 "B" 为主语的三元组
    print(f"\nTriplets of B: {triplets}")  # 打印以 "B" 为主语的三元组列表
    assert len(triplets) == 2  # 断言以 "B" 为主语的三元组数量为 2

    schema = graph_store.get_schema()  # 获取图数据库的架构信息
    print(f"\nSchema: {schema}")  # 打印图数据库的架构信息
    assert len(schema) == 138  # 断言图数据库的架构信息长度为 138
```