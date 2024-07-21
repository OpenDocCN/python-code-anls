# `.\pytorch\test\jit\test_python_bindings.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

# 如果该文件被直接运行，则抛出运行时错误，建议使用特定方式运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TestPythonBindings\n\n"
        "instead."
    )

# 定义测试类，继承自 JitTestCase
class TestPythonBindings(JitTestCase):

    # 测试获取 Python 自定义函数
    def test_cu_get_functions(self):
        @torch.jit.script
        def test_get_python_cu_fn(x: torch.Tensor):
            return 2 * x

        # 获取 Python 自定义函数的相关信息
        cu = torch.jit._state._python_cu
        self.assertTrue(
            "test_get_python_cu_fn" in (str(fn.name) for fn in cu.get_functions())
        )

    # 测试创建自定义函数
    def test_cu_create_function(self):
        @torch.jit.script
        def fn(x: torch.Tensor):
            return 2 * x

        # 创建一个空的 CompilationUnit 对象
        cu = torch._C.CompilationUnit()
        # 在 CompilationUnit 中创建名为 "test_fn" 的函数
        cu.create_function("test_fn", fn.graph)

        # 准备输入数据
        inp = torch.randn(5)

        # 使用创建的函数进行计算并断言结果
        self.assertEqual(inp * 2, cu.find_function("test_fn")(inp))
        # 测试找不到函数时的返回值
        self.assertEqual(cu.find_function("doesnt_exist"), None)
        # 通过类似属性调用的方式使用函数
        self.assertEqual(inp * 2, cu.test_fn(inp))
        # 测试访问不存在的属性时是否抛出异常
        with self.assertRaises(AttributeError):
            cu.doesnt_exist(inp)

    # 测试图的无效性
    def test_invalidation(self):
        @torch.jit.script
        def test_invalidation_fn(x: torch.Tensor):
            return 2 * x

        # 复制图形并插入一个 profile 节点
        gr = test_invalidation_fn.graph.copy()
        n = gr.insertNode(gr.create("prim::profile"))
        v = n.output()
        # 检查节点和输出是否正常工作
        str((n, v))
        # 运行死代码消除优化
        torch._C._jit_pass_dce(gr)
        # 测试节点和输出是否已被标记为无效
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(n)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(v)

    # 测试图的迭代器的生命周期管理
    def test_graph_iterator_keepalive(self):
        @torch.jit.script
        def test_iterator_keepalive_fn(x: torch.Tensor):
            return 2 * x

        # 获取内联图的节点、输入和输出列表，以避免之前的段错误问题
        n = test_iterator_keepalive_fn.inlined_graph.nodes()
        list(n)
        i = test_iterator_keepalive_fn.inlined_graph.inputs()
        list(i)
        o = test_iterator_keepalive_fn.inlined_graph.outputs()
        list(o)

    # 测试别名数据库
    def test_aliasdb(self):
        @torch.jit.script
        def test_aliasdb_fn(x: torch.Tensor):
            return 2 * x

        # 复制图形并获取其别名数据库
        gr = test_aliasdb_fn.graph.copy()
        alias_db = gr.alias_db()
        # 断言别名数据库包含特定字符串
        self.assertTrue("WILDCARD" in str(alias_db))
        # 断言别名数据库能够生成 Graphviz 字符串
        self.assertTrue("digraph alias_db" in alias_db.to_graphviz_str())

    # 测试创建图形时的异常情况
    def test_graph_create(self):
        gr = torch._C.Graph()
        # 断言创建常量时抛出值错误异常
        with self.assertRaises(ValueError):
            gr.create("prim::Constant", [None])

    # 测试添加输入值到图形中
    def test_add_input(self):
        gr = torch._C.Graph()
        # 向图形中添加名为 "foo" 的输入值，并断言其在图形的输入值列表中
        foo_value = gr.addInput("foo")
        assert foo_value in gr.inputs()

    # 测试图的规范化
    def test_canonicalize(self):
        ir = """
graph(%p207 : Tensor,
      %1 : Tensor,
      %p407 : int):
  # 定义一个图形函数，接受三个参数：张量%p207，张量%1，整数%p407

  %11 : Tensor = aten::view_expand_placeholder(%1)
  # 创建一个新张量%11，通过对输入张量%1进行视图扩展操作

  %12 : Tensor = aten::pointwise_placeholder(%11, %p207, %p407)
  # 创建一个新张量%12，通过对%11和%p207、%p407执行逐点操作

  %13 : Tensor = aten::view_expand_placeholder(%12)
  # 创建一个新张量%13，通过对%12进行视图扩展操作

  %14 : Tensor = aten::pointwise_placeholder(%13)
  # 创建一个新张量%14，通过对%13执行逐点操作

  return (%14)
  # 返回张量%14作为函数的结果
        """

graph1 = torch._C.parse_ir(ir)
# 解析给定的IR（Intermediate Representation），得到图形表示graph1

graph1 = torch._C._jit_pass_canonicalize(graph1, True)
# 对graph1进行规范化处理，确保一致性和最佳化，此处启用了所有优化选项

graph2 = torch._C.parse_ir(ir)
# 再次解析给定的IR，得到另一个图形表示graph2

graph2 = torch._C._jit_pass_canonicalize(graph2)
# 对graph2进行规范化处理，采用默认的规范化选项

self.assertEqual(str(graph1), str(graph2))
# 使用断言确保graph1和graph2的字符串表示完全相同

FileCheck().check("%p207").check_not("%14").run(graph1)
# 使用FileCheck工具检查graph1中存在%p207，并且不应包含%14

graph3 = torch._C.parse_ir(ir)
# 再次解析给定的IR，得到另一个图形表示graph3

graph3 = torch._C._jit_pass_canonicalize(graph3, False)
# 对graph3进行规范化处理，此处禁用所有优化选项

FileCheck().check_not("%p207").run(graph3)
# 使用FileCheck工具检查graph3中不应存在%p207
```