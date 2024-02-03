# `.\AutoGPT\benchmark\tests\test_extract_subgraph.py`

```py
# 导入 pytest 模块
import pytest

# 从 agbenchmark.utils.dependencies.graphs 模块中导入 extract_subgraph_based_on_category 函数
from agbenchmark.utils.dependencies.graphs import extract_subgraph_based_on_category

# 定义一个 fixture 函数，返回一个包含课程图信息的字典
@pytest.fixture
def curriculum_graph():
    return {
        "edges": [
            {"from": "Calculus", "to": "Advanced Calculus"},
            {"from": "Algebra", "to": "Calculus"},
            {"from": "Biology", "to": "Advanced Biology"},
            {"from": "World History", "to": "Modern History"},
        ],
        "nodes": [
            {"data": {"category": ["math"]}, "id": "Calculus", "label": "Calculus"},
            {
                "data": {"category": ["math"]},
                "id": "Advanced Calculus",
                "label": "Advanced Calculus",
            },
            {"data": {"category": ["math"]}, "id": "Algebra", "label": "Algebra"},
            {"data": {"category": ["science"]}, "id": "Biology", "label": "Biology"},
            {
                "data": {"category": ["science"]},
                "id": "Advanced Biology",
                "label": "Advanced Biology",
            },
            {
                "data": {"category": ["history"]},
                "id": "World History",
                "label": "World History",
            },
            {
                "data": {"category": ["history"]},
                "id": "Modern History",
                "label": "Modern History",
            },
        ],
    }

# 定义一个示例的图结构
graph_example = {
    "nodes": [
        {"id": "A", "data": {"category": []}},
        {"id": "B", "data": {"category": []}},
        {"id": "C", "data": {"category": ["math"]}},
    ],
    "edges": [{"from": "B", "to": "C"}, {"from": "A", "to": "C"}],
}

# 定义一个测试函数，测试提取特定类别的子图
def test_dfs_category_math(curriculum_graph):
    # 提取数学类别的子图
    result_graph = extract_subgraph_based_on_category(curriculum_graph, "math")

    # 预期的节点包括：Algebra, Calculus, Advanced Calculus
    # 预期的边包括：Algebra->Calculus, Calculus->Advanced Calculus

    # 预期的节点列表
    expected_nodes = ["Algebra", "Calculus", "Advanced Calculus"]
    # 预期的边列表，包含从一个节点到另一个节点的关系
    expected_edges = [
        {"from": "Algebra", "to": "Calculus"},
        {"from": "Calculus", "to": "Advanced Calculus"},
    ]

    # 断言结果图中节点的 id 集合与预期节点集合相同
    assert set(node["id"] for node in result_graph["nodes"]) == set(expected_nodes)
    # 断言结果图中边的起始节点和结束节点的集合与预期边的起始节点和结束节点的集合相同
    assert set((edge["from"], edge["to"]) for edge in result_graph["edges"]) == set(
        (edge["from"], edge["to"]) for edge in expected_edges
    )
# 测试从给定图中提取基于特定类别的子图的函数
def test_extract_subgraph_math_category():
    # 提取基于"数学"类别的子图
    subgraph = extract_subgraph_based_on_category(graph_example, "math")
    # 断言子图中的节点与原图中的节点具有相同的 id 和类别
    assert set(
        (node["id"], tuple(node["data"]["category"])) for node in subgraph["nodes"]
    ) == set(
        (node["id"], tuple(node["data"]["category"])) for node in graph_example["nodes"]
    )
    # 断言子图中的边与原图中的边具有相同的起始节点和结束节点

    assert set((edge["from"], edge["to"]) for edge in subgraph["edges"]) == set(
        (edge["from"], edge["to"]) for edge in graph_example["edges"]
    )

# 测试提取不存在类别的子图的函数
def test_extract_subgraph_non_existent_category():
    # 提取基于"toto"类别的子图
    result_graph = extract_subgraph_based_on_category(graph_example, "toto")

    # 断言结果图中没有节点和边
    assert len(result_graph["nodes"]) == 0
    assert len(result_graph["edges"]) == 0
```