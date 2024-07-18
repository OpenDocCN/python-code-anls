# `.\graphrag\tests\unit\indexing\graph\utils\test_stable_lcc.py`

```py
# 导入必要的模块和库
import unittest  # 引入unittest模块，用于编写和运行测试用例

import networkx as nx  # 引入networkx库，用于图论和复杂网络的操作

from graphrag.index.graph.utils.stable_lcc import stable_largest_connected_component  # 从特定路径引入稳定最大连通分量函数


class TestStableLCC(unittest.TestCase):
    def test_undirected_graph_run_twice_produces_same_graph(self):
        # 创建一个强连通的无向图作为输入
        graph_in_1 = self._create_strongly_connected_graph()
        # 调用稳定最大连通分量函数，获取输出图
        graph_out_1 = stable_largest_connected_component(graph_in_1)

        # 创建一个边反转的强连通的无向图作为输入
        graph_in_2 = self._create_strongly_connected_graph_with_edges_flipped()
        # 再次调用稳定最大连通分量函数，获取输出图
        graph_out_2 = stable_largest_connected_component(graph_in_2)

        # 确保两次运行的输出图相同
        assert "".join(nx.generate_graphml(graph_out_1)) == "".join(
            nx.generate_graphml(graph_out_2)
        )

    def test_directed_graph_keeps_source_target_intact(self):
        # 创建一个边反转的强连通的有向图作为输入
        graph_in = self._create_strongly_connected_graph_with_edges_flipped(
            digraph=True
        )
        # 调用稳定最大连通分量函数，获取输出图
        graph_out = stable_largest_connected_component(graph_in.copy())

        # 确保边保持相同，并且方向被保留
        edges_1 = [f"{edge[0]} -> {edge[1]}" for edge in graph_in.edges(data=True)]
        edges_2 = [f"{edge[0]} -> {edge[1]}" for edge in graph_out.edges(data=True)]

        assert edges_1 == edges_2

    def test_directed_graph_run_twice_produces_same_graph(self):
        # 创建一个边反转的强连通的有向图作为输入
        graph_in = self._create_strongly_connected_graph_with_edges_flipped(
            digraph=True
        )
        # 第一次调用稳定最大连通分量函数，获取输出图
        graph_out_1 = stable_largest_connected_component(graph_in.copy())
        # 再次调用稳定最大连通分量函数，获取输出图
        graph_out_2 = stable_largest_connected_component(graph_in.copy())

        # 确保多次运行时的输出图相同
        assert "".join(nx.generate_graphml(graph_out_1)) == "".join(
            nx.generate_graphml(graph_out_2)
        )

    def _create_strongly_connected_graph(self, digraph=False):
        # 创建一个无向或有向的强连通图
        graph = nx.Graph() if not digraph else nx.DiGraph()
        graph.add_node("1", node_name=1)
        graph.add_node("2", node_name=2)
        graph.add_node("3", node_name=3)
        graph.add_node("4", node_name=4)
        graph.add_edge("4", "5", degree=4)
        graph.add_edge("3", "4", degree=3)
        graph.add_edge("2", "3", degree=2)
        graph.add_edge("1", "2", degree=1)
        return graph

    def _create_strongly_connected_graph_with_edges_flipped(self, digraph=False):
        # 创建一个边反转的无向或有向的强连通图
        graph = nx.Graph() if not digraph else nx.DiGraph()
        graph.add_node("1", node_name=1)
        graph.add_node("2", node_name=2)
        graph.add_node("3", node_name=3)
        graph.add_node("4", node_name=4)
        graph.add_edge("5", "4", degree=4)
        graph.add_edge("4", "3", degree=3)
        graph.add_edge("3", "2", degree=2)
        graph.add_edge("2", "1", degree=1)
        return graph
```