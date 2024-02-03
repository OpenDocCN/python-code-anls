# `.\AutoGPT\benchmark\tests\test_get_roots.py`

```py
# 从agbenchmark.utils.dependencies.graphs模块中导入get_roots函数
from agbenchmark.utils.dependencies.graphs import get_roots

# 定义测试函数test_get_roots
def test_get_roots():
    # 定义一个图，包含节点和边的信息
    graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
        ],
    }

    # 调用get_roots函数，获取图中的根节点
    result = get_roots(graph)
    # 断言结果中的根节点为'A'和'D'
    assert set(result) == {
        "A",
        "D",
    }, f"Expected roots to be 'A' and 'D', but got {result}"

# 定义测试函数test_no_roots
def test_no_roots():
    # 定义一个全连接的图，包含节点和边的信息
    fully_connected_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "A"},
        ],
    }

    # 调用get_roots函数，获取全连接图中的根节点
    result = get_roots(fully_connected_graph)
    # 断言结果中没有根节点
    assert not result, "Expected no roots, but found some"

# 定义测试函数test_no_rcoots（该函数注释被注释掉了）
# def test_no_rcoots():
#     fully_connected_graph = {
#         "nodes": [
#             {"id": "A", "data": {"category": []}},
#             {"id": "B", "data": {"category": []}},
#             {"id": "C", "data": {"category": []}},
#         ],
#         "edges": [
#             {"from": "A", "to": "B"},
#             {"from": "D", "to": "C"},
#         ],
#     }
#
#     result = get_roots(fully_connected_graph)
#     assert set(result) == {"A"}, f"Expected roots to be 'A', but got {result}"
```