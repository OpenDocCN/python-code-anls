# `.\AutoGPT\benchmark\tests\test_is_circular.py`

```py
# 导入检测循环依赖的函数
from agbenchmark.utils.dependencies.graphs import is_circular

# 测试存在循环依赖的情况
def test_is_circular():
    cyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # 新节点
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            {"from": "D", "to": "A"},  # 此边创建了一个循环
        ],
    }

    # 检测图中是否存在循环依赖
    result = is_circular(cyclic_graph)
    # 断言应该存在循环依赖，如果没有检测到则抛出异常
    assert result is not None, "Expected a cycle, but none was detected"
    # 断言检测到的循环路径是图中边的一部分
    assert all(
        (
            (result[i], result[i + 1])
            in [(x["from"], x["to"]) for x in cyclic_graph["edges"]]
        )
        for i in range(len(result) - 1)
    ), "The detected cycle path is not part of the graph's edges"

# 测试不存在循环依赖的情况
def test_is_not_circular():
    acyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # 新节点
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            # 从 D 没有回边到任何节点，因此保持无环
        ],
    }

    # 断言在无环图中没有检测到循环依赖，如果检测到则抛出异常
    assert is_circular(acyclic_graph) is None, "Detected a cycle in an acyclic graph"
```