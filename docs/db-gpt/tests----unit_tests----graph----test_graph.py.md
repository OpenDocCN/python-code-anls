# `.\DB-GPT-src\tests\unit_tests\graph\test_graph.py`

```py
# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从指定路径导入 MemoryGraph 类、Edge 类、Vertex 类和 Direction 枚举
from dbgpt.storage.graph_store.graph import MemoryGraph, Edge, Vertex, Direction

# 定义一个 pytest fixture，初始化 MemoryGraph 对象并添加一些边和顶点
@pytest.fixture
def g():
    g = MemoryGraph()
    g.append_edge(Edge("A", "A", label="0"))
    g.append_edge(Edge("A", "A", label="1"))
    g.append_edge(Edge("A", "B", label="2"))
    g.append_edge(Edge("B", "C", label="3"))
    g.append_edge(Edge("B", "D", label="4"))
    g.append_edge(Edge("C", "D", label="5"))
    g.append_edge(Edge("B", "E", label="6"))
    g.append_edge(Edge("F", "E", label="7"))
    g.append_edge(Edge("E", "F", label="8"))
    g.upsert_vertex(Vertex("G"))
    yield g  # 返回 MemoryGraph 对象

# 使用 pytest.mark.parametrize 装饰器为 test_delete 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "action, vc, ec",
    [
        (lambda g: g.del_vertices("G", "G"), 6, 9),
        (lambda g: g.del_vertices("C"), 6, 7),
        (lambda g: g.del_vertices("A", "G"), 5, 6),
        (lambda g: g.del_edges("E", "F", label="8"), 7, 8),
        (lambda g: g.del_edges("A", "B"), 7, 8),
        (lambda g: g.del_neighbor_edges("A", Direction.IN), 7, 7),
    ],
)
# 定义测试函数 test_delete，接受 fixture g 和参数化的 action、vc、ec 参数
def test_delete(g, action, vc, ec):
    action(g)  # 执行传入的 action 函数
    result = g.graphviz()  # 获取当前图形的 Graphviz 表示
    print(f"\n{result}")  # 打印 Graphviz 表示
    assert g.vertex_count == vc  # 断言顶点数符合预期
    assert g.edge_count == ec  # 断言边数符合预期

# 使用 pytest.mark.parametrize 装饰器为 test_search 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "vids, dir, vc, ec",
    [
        (["B"], Direction.OUT, 5, 6),
        (["A"], Direction.IN, 1, 2),
        (["F"], Direction.IN, 4, 6),
        (["B"], Direction.BOTH, 6, 9),
        (["A", "G"], Direction.BOTH, 7, 9),
    ],
)
# 定义测试函数 test_search，接受 fixture g 和参数化的 vids、dir、vc、ec 参数
def test_search(g, vids, dir, vc, ec):
    subgraph = g.search(vids, dir)  # 执行搜索操作，返回子图
    print(f"\n{subgraph.graphviz()}")  # 打印子图的 Graphviz 表示
    assert subgraph.vertex_count == vc  # 断言子图的顶点数符合预期
    assert subgraph.edge_count == ec  # 断言子图的边数符合预期

# 使用 pytest.mark.parametrize 装饰器为 test_search_result_limit 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "vids, dir, ec",
    [
        (["B"], Direction.BOTH, 5),
        (["B"], Direction.OUT, 5),
        (["B"], Direction.IN, 3),
    ],
)
# 定义测试函数 test_search_result_limit，接受 fixture g 和参数化的 vids、dir、ec 参数
def test_search_result_limit(g, vids, dir, ec):
    subgraph = g.search(vids, dir, limit=ec)  # 执行搜索操作，限制结果数量
    print(f"\n{subgraph.graphviz()}")  # 打印子图的 Graphviz 表示
    assert subgraph.edge_count == ec  # 断言子图的边数符合预期

# 使用 pytest.mark.parametrize 装饰器为 test_search_fan_limit 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "vids, dir, fan, ec",
    [
        (["A"], Direction.OUT, 1, 1),
        (["B"], Direction.OUT, 2, 3),
        (["F"], Direction.IN, 1, 4),
    ],
)
# 定义测试函数 test_search_fan_limit，接受 fixture g 和参数化的 vids、dir、fan、ec 参数
def test_search_fan_limit(g, vids, dir, fan, ec):
    subgraph = g.search(vids, dir, fan=fan)  # 执行搜索操作，限制邻居节点数
    print(f"\n{subgraph.graphviz()}")  # 打印子图的 Graphviz 表示
    assert subgraph.edge_count == ec  # 断言子图的边数符合预期

# 使用 pytest.mark.parametrize 装饰器为 test_search_depth_limit 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "vids, dir, dep, ec",
    [
        (["A"], Direction.OUT, 1, 3),
        (["A"], Direction.OUT, 2, 6),
        (["B"], Direction.OUT, 2, 5),
        (["B"], Direction.IN, 1, 1),
        (["D"], Direction.IN, 2, 4),
        (["B"], Direction.BOTH, 1, 4),
        (["B"], Direction.BOTH, 2, 9),
    ],
)
# 定义测试函数 test_search_depth_limit，接受 fixture g 和参数化的 vids、dir、dep、ec 参数
def test_search_depth_limit(g, vids, dir, dep, ec):
    subgraph = g.search(vids, dir, depth=dep)  # 执行搜索操作，限制搜索深度
    print(f"\n{subgraph.graphviz()}")  # 打印子图的 Graphviz 表示
    assert subgraph.edge_count == ec  # 断言子图的边数符合预期
```