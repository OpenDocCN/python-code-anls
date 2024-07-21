# `.\pytorch\test\package\test_digraph.py`

```py
# Owner(s): ["oncall: package/deploy"]

# 导入 DiGraph 类，用于创建有向图
from torch.package._digraph import DiGraph
# 导入运行测试的工具函数
from torch.testing._internal.common_utils import run_tests

try:
    # 尝试从当前包中导入 PackageTestCase 类
    from .common import PackageTestCase
except ImportError:
    # 如果导入失败，支持直接运行该文件的情况，从 common 模块导入 PackageTestCase 类
    from common import PackageTestCase

class TestDiGraph(PackageTestCase):
    """Test the DiGraph structure we use to represent dependencies in PackageExporter"""

    def test_successors(self):
        # 创建一个空的有向图对象
        g = DiGraph()
        # 向有向图中添加边，连接 "foo" 和 "bar"
        g.add_edge("foo", "bar")
        # 向有向图中添加边，连接 "foo" 和 "baz"
        g.add_edge("foo", "baz")
        # 向有向图中添加节点 "qux"
        g.add_node("qux")

        # 断言 "foo" 的后继节点中包含 "bar"
        self.assertIn("bar", list(g.successors("foo")))
        # 断言 "foo" 的后继节点中包含 "baz"
        self.assertIn("baz", list(g.successors("foo")))
        # 断言 "qux" 的后继节点数量为 0
        self.assertEqual(len(list(g.successors("qux"))), 0)

    def test_predecessors(self):
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        # 断言 "bar" 的前驱节点中包含 "foo"
        self.assertIn("foo", list(g.predecessors("bar")))
        # 断言 "baz" 的前驱节点中包含 "foo"
        self.assertIn("foo", list(g.predecessors("baz")))
        # 断言 "qux" 的前驱节点数量为 0
        self.assertEqual(len(list(g.predecessors("qux"))), 0)

    def test_successor_not_in_graph(self):
        g = DiGraph()
        # 测试在图中不存在的节点的后继节点
        with self.assertRaises(ValueError):
            g.successors("not in graph")

    def test_predecessor_not_in_graph(self):
        g = DiGraph()
        # 测试在图中不存在的节点的前驱节点
        with self.assertRaises(ValueError):
            g.predecessors("not in graph")

    def test_node_attrs(self):
        g = DiGraph()
        # 添加带有属性的节点 "foo"
        g.add_node("foo", my_attr=1, other_attr=2)
        # 断言节点 "foo" 的 "my_attr" 属性值为 1
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)
        # 断言节点 "foo" 的 "other_attr" 属性值为 2
        self.assertEqual(g.nodes["foo"]["other_attr"], 2)

    def test_node_attr_update(self):
        g = DiGraph()
        # 添加带有属性的节点 "foo"
        g.add_node("foo", my_attr=1)
        # 断言节点 "foo" 的 "my_attr" 属性值为 1
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)

        # 更新节点 "foo" 的 "my_attr" 属性为 "different"
        g.add_node("foo", my_attr="different")
        # 断言节点 "foo" 的 "my_attr" 属性值更新为 "different"
        self.assertEqual(g.nodes["foo"]["my_attr"], "different")

    def test_edges(self):
        g = DiGraph()
        # 添加多条边到有向图
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(1, 3)
        g.add_edge(4, 5)

        # 获取所有边的列表
        edge_list = list(g.edges)
        # 断言边的数量为 4 条
        self.assertEqual(len(edge_list), 4)

        # 断言特定边存在于边列表中
        self.assertIn((1, 2), edge_list)
        self.assertIn((2, 3), edge_list)
        self.assertIn((1, 3), edge_list)
        self.assertIn((4, 5), edge_list)

    def test_iter(self):
        g = DiGraph()
        # 向有向图中添加节点
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        # 使用迭代器获取节点集合
        nodes = set()
        nodes.update(g)

        # 断言节点集合包含特定的节点
        self.assertEqual(nodes, {1, 2, 3})

    def test_contains(self):
        g = DiGraph()
        # 向有向图中添加节点 "yup"
        g.add_node("yup")

        # 断言节点 "yup" 存在于图中
        self.assertTrue("yup" in g)
        # 断言节点 "nup" 不存在于图中
        self.assertFalse("nup" in g)

    def test_contains_non_hashable(self):
        g = DiGraph()
        # 测试不可哈希对象作为节点的情况
        self.assertFalse([1, 2, 3] in g)
    # 定义测试函数，测试有向图的前向传递闭包
    def test_forward_closure(self):
        # 创建一个有向图对象
        g = DiGraph()
        # 添加图的边
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("5", "4")
        g.add_edge("4", "3")
        # 断言：检查从节点 "1" 开始的前向传递闭包是否为 {"1", "2", "3"}
        self.assertTrue(g.forward_transitive_closure("1") == {"1", "2", "3"})
        # 断言：检查从节点 "4" 开始的前向传递闭包是否为 {"4", "3"}
        self.assertTrue(g.forward_transitive_closure("4") == {"4", "3"})

    # 定义测试函数，测试有向图的所有路径
    def test_all_paths(self):
        # 创建一个有向图对象
        g = DiGraph()
        # 添加图的边
        g.add_edge("1", "2")
        g.add_edge("1", "7")
        g.add_edge("7", "8")
        g.add_edge("8", "3")
        g.add_edge("2", "3")
        g.add_edge("5", "4")
        g.add_edge("4", "3")

        # 调用图对象的 all_paths 方法，查找从节点 "1" 到节点 "3" 的所有路径
        result = g.all_paths("1", "3")
        # 处理结果以避免不确定性
        actual = {i.strip("\n") for i in result.split(";")[2:-1]}
        # 预期路径集合
        expected = {
            '"2" -> "3"',
            '"1" -> "7"',
            '"7" -> "8"',
            '"1" -> "2"',
            '"8" -> "3"',
        }
        # 断言：检查实际路径集合与预期路径集合是否相等
        self.assertEqual(actual, expected)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```