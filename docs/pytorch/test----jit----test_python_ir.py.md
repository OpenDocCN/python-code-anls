# `.\pytorch\test\jit\test_python_ir.py`

```py
# Owner(s): ["oncall: jit"]

# 导入单元测试模块
import unittest

# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 导入 PyTorch 库
import torch

# 导入文件检查工具
from torch.testing import FileCheck

# 导入通用工具函数和常量，如 IS_MACOS
from torch.testing._internal.common_utils import IS_MACOS

# 导入 JIT 测试用例基类
from torch.testing._internal.jit_utils import JitTestCase

# 如果脚本直接运行而不是被导入，则抛出运行时错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestPythonIr，继承自 JitTestCase
class TestPythonIr(JitTestCase):

    # 测试函数：test_param_strides
    def test_param_strides(self):
        # 定义一个简单的追踪函数 trace_me
        def trace_me(arg):
            return arg

        # 创建一个形状为 (1, 3, 16, 16) 的全零张量 t
        t = torch.zeros(1, 3, 16, 16)

        # 对 trace_me 函数进行追踪
        traced = torch.jit.trace(trace_me, t)

        # 获取追踪图中参数节点的值
        value = list(traced.graph.param_node().outputs())[0]

        # 获取张量 t 的真实步幅信息
        real_strides = list(t.stride())

        # 获取追踪值的类型步幅信息
        type_strides = value.type().strides()

        # 断言真实步幅和类型步幅相等
        self.assertEqual(real_strides, type_strides)

    # 测试函数：test_permute_inputs_binding
    def test_permute_inputs_binding(self):
        # 定义一个 torch.jit.script 脚本函数 foo
        @torch.jit.script
        def foo(i, j, k):
            pass

        # 获取 foo 函数的计算图
        g = foo.graph

        # 初始化输入索引列表 idxs
        idxs = []

        # 遍历计算图的输入节点，并设置调试名称
        for i, inp in enumerate(g.inputs()):
            inp.setDebugName(f"inp{i}")
            idxs.append(i)

        # 随机排列 idxs 列表
        permuted_idxs = list(np.random.permutation(idxs))

        # 对计算图 g 执行输入重排
        g.permuteInputs(permuted_idxs)

        # 验证每个输入节点的调试名称与重排后的索引相匹配
        for i, inp in enumerate(g.inputs()):
            self.assertEqual(f"inp{permuted_idxs[i]}", inp.debugName())

    # 被 macOS 平台跳过的测试函数：test_python_ir_utils
    @unittest.skipIf(IS_MACOS, "Failing on MacOS only")
    def test_python_ir_utils(self):
        # 定义一个 torch.jit.script 脚本函数 foo
        @torch.jit.script
        def foo(inp):
            x = inp + 1
            y = x / 2
            z = y * y
            return z

        # 查找 foo 函数计算图中的 add 和 div 节点
        add_node = foo.graph.findNode("aten::add")
        div_node = foo.graph.findNode("aten::div")

        # 在 add 节点后插入常量 "goodbye"，在 div 节点后插入常量 "hello"
        with foo.graph.insert_point_guard(add_node):
            with foo.graph.insert_point_guard(div_node):
                foo.graph.insertConstant("goodbye")
            foo.graph.insertConstant("hello")

        # 在 mul 节点后插入常量 "hello"
        with foo.graph.insert_point_guard(foo.graph.findNode("aten::mul")):
            foo.graph.insertConstant("hello")

        # 使用 FileCheck 检查 foo 函数的计算图，验证常量插入是否成功
        FileCheck().check("hello").check("goodbye").check("hello").run(foo.graph)

        # 断言 add_node 的匹配结果为真
        self.assertTrue(add_node.matches(add_node.schema()))

        # 断言 div_node 的匹配结果为假
        self.assertFalse(add_node.matches(div_node.schema()))
    # 定义测试函数 test_python_ir_utils_graph，用于测试以下内容
    def test_python_ir_utils_graph():
        # 定义 torch.jit.script 脚本函数 unrolled_mul，对输入的张量 x 和整数 y 执行乘法操作
        @torch.jit.script
        def unrolled_mul(x: torch.Tensor, y: int):
            # 初始化输出为 x
            out = x
            # 循环 y-1 次，每次将 out 与 x 相加，实现乘法操作的展开
            for _ in range(y - 1):
                out = out + x
            return out

        # 定义 torch.jit.script 脚本函数 foo，对输入 x 执行乘以 4 的操作
        @torch.jit.script
        def foo(x):
            return x * 4

        # 获取 foo 函数的计算图
        g = foo.graph
        # 查找计算图中所有的乘法节点 "aten::mul"
        muls = g.findAllNodes("aten::mul")
        # 筛选出仅包含标量乘法的节点
        scalar_muls = filter(
            lambda x: x.matches("aten::mul(Tensor self, Scalar other) -> Tensor"), muls
        )
        # 筛选出包含常量整数作为乘数的节点
        mul_constant_int = filter(
            lambda x: isinstance(list(x.inputs())[1].toIValue(), int), scalar_muls
        )
        # 遍历每个包含常量整数乘数的节点
        for mul in mul_constant_int:
            # 在乘法节点处设置插入点保护
            with g.insert_point_guard(mul):
                # 插入 unrolled_mul 函数的计算图，使用乘法节点的输入作为 unrolled_mul 的输入
                outputs = g.insertGraph(unrolled_mul.graph, list(mul.inputs()))
                # 断言插入后的输出数量与原乘法节点的输出数量相等
                assert len(outputs) == len(list(mul.outputs()))
                # 用插入的新输出替换原乘法节点的所有使用
                for new_out, old_out in zip(outputs, g.outputs()):
                    old_out.replaceAllUsesWith(new_out)
                # 销毁原乘法节点
                mul.destroy()

        # 使用 FileCheck 对计算图进行验证，检查是否不存在 "aten::mul" 节点，并检查是否存在 "aten::add" 节点
        FileCheck().check_not("aten::mul").check("aten::add").run(foo.graph)
        # 断言 foo 函数对全 1 张量的计算结果等于全 1 张量乘以 4 的结果
        self.assertEqual(foo(torch.ones([2, 2])), torch.ones([2, 2]) * 4)
```