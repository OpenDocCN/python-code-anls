# `.\pytorch\test\jit\test_symbolic_shape_analysis.py`

```py
# Owner(s): ["oncall: jit"]

import operator  # 导入操作符模块
import unittest  # 导入单元测试模块
from textwrap import dedent  # 导入文本包装模块中的dedent函数
from typing import Any, List  # 导入类型提示模块中的Any和List类型

import torch  # 导入PyTorch库
from torch import nn, Tensor  # 从PyTorch中导入神经网络和张量模块
from torch.testing import FileCheck  # 导入PyTorch测试模块中的FileCheck类
from torch.testing._internal.common_methods_invocations import sample_inputs_cat_concat  # 导入PyTorch测试内部的示例输入拼接函数
from torch.testing._internal.common_utils import make_tensor  # 导入PyTorch测试内部的创建张量函数
from torch.testing._internal.jit_utils import execWrapper, JitTestCase  # 导入PyTorch测试内部的执行包装器和JitTestCase测试用例

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# XXX: still in prototype
class TestSymbolicShapeAnalysis(JitTestCase):
    def setUp(self):
        super(JitTestCase, self).setUp()  # 调用父类的setUp方法
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()  # 获取当前符号形状测试模式是否启用的状态
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)  # 设置启用符号形状测试模式为True

    def tearDown(self):
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled  # 恢复之前的符号形状测试模式状态
        )

    def test_shape_analysis(self):
        @torch.jit.script
        def foo(x, y):
            return x * y  # 定义一个torch脚本函数，实现两个参数的乘法操作

        inputs = list(foo.graph.inputs())  # 获取foo函数的输入参数列表

        def prop_shapes_on_graph(inp0, inp1):
            inputs[0].setType(inputs[0].type().with_sizes(inp0))  # 设置第一个输入参数的形状大小
            inputs[1].setType(inputs[1].type().with_sizes(inp1))  # 设置第二个输入参数的形状大小
            torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)  # 在图上传播形状信息

        prop_shapes_on_graph([1, 6, 5], [1, 7, 1, 5])  # 对图进行形状分析和传播
        FileCheck().check("1, 7, 6, 5").run(foo.graph)  # 使用FileCheck检查图的输出形状是否符合预期

        # None implicitly creates a new symbolic symbol
        prop_shapes_on_graph([None, None], [None, None, None])  # 使用None创建新的符号形状
        output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()  # 获取乘法节点的输出符号形状
        inp0_shape = inputs[0].type().symbolic_sizes()  # 获取第一个输入参数的符号形状
        inp1_shape = inputs[1].type().symbolic_sizes()  # 获取第二个输入参数的符号形状

        # output shape dim 0 should be taken from the second inp dim0
        # other two dims we cannot infer and are given a new symbolic shape
        self.assertEqual(output_shape[0], inp1_shape[0])  # 断言输出形状的第一个维度与第二个输入参数的第一个维度相同
        self.assertFalse(output_shape[1] in inp0_shape + inp1_shape)  # 断言输出形状的第二个维度不在输入参数的符号形状中
        self.assertFalse(output_shape[2] in inp0_shape + inp1_shape)  # 断言输出形状的第三个维度不在输入参数的符号形状中

        # XXX: symbolic shapes are represented with an increasing counter of unique
        # values, use `_new_symbolic_shape_symbol` api instead of specifying negative
        # dimensions directly so there is no chance of collision between manual number
        # and current counter value.
        sym1 = torch._C._new_symbolic_shape_symbol()  # 创建新的符号形状符号1
        sym2 = torch._C._new_symbolic_shape_symbol()  # 创建新的符号形状符号2
        sym3 = torch._C._new_symbolic_shape_symbol()  # 创建新的符号形状符号3
        prop_shapes_on_graph([sym1, 1, sym3], [1, sym2, sym3])  # 使用新创建的符号形状对图进行形状分析和传播
        output_shape = foo.graph.findNode("aten::mul").output().type().symbolic_sizes()  # 获取乘法节点的输出符号形状
        self.assertEqual(output_shape[0], sym1)  # 断言输出形状的第一个维度与符号1相同
        self.assertEqual(output_shape[1], sym2)  # 断言输出形状的第二个维度与符号2相同
        self.assertEqual(output_shape[2], sym3)  # 断言输出形状的第三个维度与符号3相同
    def test_shared_shape_graph(self):
        # 定义一个 Torch Script 函数 foo，接受两个参数 x 和 y，返回它们的乘积和除法运算结果
        @torch.jit.script
        def foo(x, y):
            return x * y, x / y

        # 在 foo 函数的计算图中查找乘法节点
        mul_node = foo.graph.findNode("aten::mul")
        # 在 foo 函数的计算图中查找除法节点
        div_node = foo.graph.findNode("aten::div")

        # 获取乘法节点的形状计算子图
        mul_graph = torch._C._jit_shape_compute_graph_for_node(mul_node)
        # 获取除法节点的形状计算子图
        div_graph = torch._C._jit_shape_compute_graph_for_node(div_node)
        # 断言乘法节点和除法节点的形状计算子图是同一个对象
        self.assertIsNotNone(mul_graph)
        self.assertIs(mul_graph, div_graph)

    def test_write(self):
        # 定义一个 Torch Script 函数 foo，接受两个参数 a 和 b，返回它们的乘法结果
        @torch.jit.script
        def foo(a, b):
            return a * b

        # 对 foo 函数的计算图应用形状传播优化
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        # 使用 FileCheck 检查计算图中是否存在乘法操作
        FileCheck().check("Tensor = aten::mul").run(foo.graph)

        # 定义一个 Torch Script 函数 foo，接受一个参数 y，修改数组 x 的第一个元素并返回 y 的视图
        @torch.jit.script
        def foo(y):
            x = [1, 2, 3, 4]
            x[0] = 5
            return y.view(x)

        # 对 foo 函数的计算图应用形状传播优化
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        # 使用 FileCheck 检查计算图中是否存在视图操作
        FileCheck().check("Tensor = aten::view").run(foo.graph)

    def test_if_propagation(self):
        # 定义一个 Torch Script 函数 foo，接受一个整数 i 和一个张量 z
        @torch.jit.script
        def foo(i: int, z):
            x = torch.ones([2, 3, 4, 5])
            # 根据 z 的尺寸和 i 的值创建一个视图 y
            y = z.view([z.size(i), 3, 2, z.size(i)])
            # 如果 i 等于 4，则返回 x
            if i == 4:
                return x
            else:
                return y

        # 对 foo 函数的计算图应用常量传播优化
        torch._C._jit_pass_constant_propagation(foo.graph)
        # 对 foo 函数的计算图应用形状传播优化
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        # 在计算图中查找视图操作节点
        view = foo.graph.findNode("aten::view")

        # 定义一个函数 neg_to_one，将列表中小于 0 的元素替换为 -1，并返回修改后的列表
        def neg_to_one(li):
            return [elem if elem >= 0 else -1 for elem in li]

        # 断言视图操作节点的符号尺寸经过处理后为 [-1, 3, 2, -1]
        self.assertEqual(
            neg_to_one(view.output().type().symbolic_sizes()), [-1, 3, 2, -1]
        )
        # 获取条件语句节点的输出，断言其符号尺寸经过处理后为 [-1, 3, -1, -1]
        if_out = next(foo.graph.findNode("prim::If").outputs())
        self.assertEqual(neg_to_one(if_out.type().symbolic_sizes()), [-1, 3, -1, -1])

    def test_unary_shape_functions(self):
        # 定义一个包含单目运算函数的列表 unary_ops
        unary_ops = [
            torch.nn.functional.hardtanh,
        ]
        # 遍历单目运算函数列表
        for fn in unary_ops:
            # 对函数 fn 进行 Torch Script 追踪，输入为随机生成的 4x4 张量
            t = torch.jit.trace(fn, (torch.rand([4, 4])))
            # 获取追踪图的第一个输入节点
            ten_input = next(t.graph.inputs())
            # 将输入节点的类型设置为指定尺寸为 [2, 2]
            ten_input.setType(ten_input.type().with_sizes([2, 2]))
            # 对追踪图应用形状传播优化
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            # 断言追踪图的输出的符号尺寸为 [2, 2]
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])

    def test_unary_shape_fns_inplace(self):
        # 定义一个原地操作函数 mul_inplace，接受一个张量 x，将其每个元素乘以 2 并返回结果
        def mul_inplace(x: torch.Tensor):
            y = x.mul_(2)
            return y

        # 定义一个包含原地操作函数的列表 unary_ops
        unary_ops = [mul_inplace]
        # 遍历原地操作函数列表
        for fn in unary_ops:
            # 对函数 fn 进行 Torch Script 包装
            t = torch.jit.script(fn)
            # 获取包装后的函数的第一个输入节点
            ten_input = next(t.graph.inputs())
            # 将输入节点的类型设置为指定尺寸为 [2, 2]
            ten_input.setType(ten_input.type().with_sizes([2, 2]))
            # 对包装后函数的计算图应用形状传播优化
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            # 断言包装后函数的输出的符号尺寸为 [2, 2]
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [2, 2])
    def test_binary_shape_functions(self):
        # 定义要测试的二进制操作函数列表
        binary_ops = [
            operator.__mul__,       # 乘法操作
            operator.__truediv__,   # 真除法操作
            operator.__gt__,        # 大于比较操作
            operator.__add__,       # 加法操作
        ]

        # 对于每个操作函数进行测试
        for fn in binary_ops:
            # 定义两个不同大小的张量作为输入
            size_1 = [1, 4, 8]
            size_2 = [4, 1, 8]
            # 使用 torch.jit.trace 对操作函数进行追踪
            t = torch.jit.trace(fn, (torch.rand([4]), torch.rand([4])))
            # 获取计算图的输入节点列表
            inputs = list(t.graph.inputs())
            # 设置第一个输入节点的类型及大小
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            # 设置第二个输入节点的类型及大小
            inputs[1].setType(inputs[1].type().with_sizes(size_2))
            # 在计算图上执行形状传播
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            # 断言输出节点的符号大小
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])

    def test_binary_shape_fns_inplace(self):
        # 定义原地操作函数，如除法和加法
        def div_inplace_tensor(x: torch.Tensor, y: torch.Tensor):
            z = x.div_(y)
            return z

        def add_inplace_tensor(x: torch.Tensor, y: torch.Tensor):
            z = x.add_(y)
            return z

        # 原地操作函数列表
        binary_ops = [
            div_inplace_tensor,
            add_inplace_tensor,
        ]

        # 对于每个原地操作函数进行测试
        for fn in binary_ops:
            # 定义一个固定大小的输入张量
            size_1 = [4, 4, 8]  # x (因为是原地操作，不能广播)
            # 使用 torch.jit.script 对函数进行脚本化
            t = torch.jit.script(fn)
            # 获取计算图的输入节点列表
            inputs = list(t.graph.inputs())
            # 设置第一个输入节点的类型及大小
            inputs[0].setType(inputs[0].type().with_sizes(size_1))
            # 不故意设置第二个输入节点的类型
            torch._C._jit_pass_propagate_shapes_on_graph(t.graph)
            # 断言输出节点的符号大小
            self.assertEqual(next(t.graph.outputs()).type().symbolic_sizes(), [4, 4, 8])

    def test_size_and_sizes(self):
        # 使用 torch.jit.script 定义一个函数 foo
        @torch.jit.script
        def foo(x, y):
            return x.view(y.size(0), 8, y.size(-1))

        # 使用 torch.jit.script 定义另一个函数 foo2
        @torch.jit.script
        def foo2(x, y):
            return x.view(y.size())

        # 对于每个函数的计算图进行测试
        for graph in [foo.graph, foo2.graph]:
            # 获取计算图的输入节点列表
            inputs = list(graph.inputs())
            # 创建一个新的符号形状符号
            sym1 = torch._C._new_symbolic_shape_symbol()

            # 设置第二个输入节点的类型及大小
            inputs[1].setType(inputs[1].type().with_sizes([5, 8, sym1]))
            # 在计算图上执行形状传播
            torch._C._jit_pass_propagate_shapes_on_graph(graph)
            # 断言输出节点的符号大小
            self.assertEqual(
                next(graph.outputs()).type().symbolic_sizes(), [5, 8, sym1]
            )

    def test_adaptive_avg_pool2d(self):
        # 不同的输入列表
        inps = [
            [(1, 64, 8, 9), (5, 7)],
            [(1, 64, 10, 9), (7)],
            [(1, 64, 10, 9), (5, None)],
            [(1, 8, 4, 3), (None, None)],
            [(1, 8, 4, 3), (None, 5)],
        ]

        # 对于每组输入进行测试
        for inp in inps:
            # 创建一个随机张量作为输入
            t = torch.randn(*inp[0])
            # 使用函数 adaptive_avg_pool2d 进行计算并获取输出大小
            out_size = torch.nn.functional.adaptive_avg_pool2d(t, inp[1]).size()

            # 定义一个匿名函数 foo，接收一个输入并调用 adaptive_avg_pool2d 函数
            def foo(x):
                return torch.nn.functional.adaptive_avg_pool2d(x, inp[1])

            # 使用 torch.jit.trace 对函数 foo 进行追踪
            fn = torch.jit.trace(foo, (t,))
            # 擦除非输入形状信息
            torch._C._jit_erase_non_input_shape_information(fn.graph)
            # 进行常量传播
            torch._C._jit_pass_constant_propagation(fn.graph)
            # 进行简单优化
            torch._C._jit_pass_peephole(fn.graph)
            # 检查形状分析
            self.checkShapeAnalysis(out_size, fn.graph, assert_propagation=True)
    def test_arange_shape(self):
        # 定义输入的不同形状
        inps = [
            (10,),              # 一维张量，长度为10
            (10, 10),           # 二维张量，形状为10x10
            (0, 10),            # 零长度的一维张量
            (0, 1000),          # 零长度的一维张量
            (1, -1, -1),        # 三维张量，其中一个维度为1，其他维度使用负数表示需要自动计算
            (1, 0, -1),         # 同上，但其中一个维度为0
            (1, 2, 1),          # 三维张量，每个维度具体指定
            (0.6, 0.89, 0.1),   # 浮点数作为维度，非整数维度
            (1, 10, 0.3),       # 三维张量，其中一个维度为浮点数
            (1, 10, 4),         # 三维张量，具体指定每个维度
            (0.6, 0.7, 0.8),    # 浮点数作为维度，非整数维度
            (1, 10, 0.3),       # 重复项
            # (True,),  TODO: https://github.com/pytorch/pytorch/issues/63405
            # (False,), TODO: https://github.com/pytorch/pytorch/issues/63405
            (0, 5),             # 一维张量，长度为5
            (0, 5, 2),          # 一维张量，步长为2
            (0, 5 + 1e-6),      # 一维张量，长度为5（考虑浮点数精度）
            (0, 5 - 1e-6),      # 一维张量，长度为5（考虑浮点数精度）
            (10, -1 + 1e-6, -1),# 三维张量，其中一个维度为10，其他维度使用负数表示需要自动计算（考虑浮点数精度）
            (10, -1, -1),       # 三维张量，其中一个维度为10，其他维度使用负数表示需要自动计算
            (10, -1 - 1e-6, -1),# 三维张量，其中一个维度为10，其他维度使用负数表示需要自动计算（考虑浮点数精度）
        ]

        for inp in inps:
            # 定义生成张量函数的模板字符串
            funcs_template = dedent(
                """
            def func():
                return torch.arange({args})
            """
            )

            inp_s = str(inp)[1:-1]  # 去除元组括号
            funcs_str = funcs_template.format(args=inp_s)
            scope = {}
            # 在给定的全局和局部作用域中执行函数字符串
            execWrapper(funcs_str, globals(), scope)
            # 使用 torch 的 JIT 编译单元来编译函数字符串
            cu = torch.jit.CompilationUnit(funcs_str)
            # 检查张量形状分析
            self.checkShapeAnalysis(
                list(cu.func().size()),        # 获取函数返回张量的形状并转为列表
                cu.func.graph,                 # 获取编译单元的函数图
                assert_propagation=True,       # 断言形状传播
                constant_prop=False,           # 禁用常量传播
            )
    def test_shape_embedding_bag(self):
        # 测试函数：test_shape_embedding_bag，用于测试嵌入操作的形状

        # TODO: merge into opinfos, having difficulties there
        # TODO：合并到 opinfos 中，遇到了困难

        with torch.no_grad():
            # 使用 torch.no_grad() 上下文管理器，确保在此范围内的操作不会计算梯度

            def make_arg(shape, low=None, high=None):
                # 创建张量的辅助函数，指定形状和数值范围
                return make_tensor(
                    shape,
                    device="cpu",
                    dtype=torch.int64,
                    low=low,
                    high=high,
                    requires_grad=False,
                )

            # 定义嵌入操作的输入和对应的模块
            nn_inps = (
                (
                    make_arg((40,), 0, 9),  # 第一个输入张量
                    torch.nn.Embedding(20, embedding_dim=64, max_norm=1.0),  # 对应的嵌入模块
                ),
                (
                    make_arg((2, 4), 0, 9),  # 第二个输入张量
                    torch.nn.Embedding(10, 20, sparse=True),  # 对应的嵌入模块，稀疏模式
                ),
                (
                    make_arg((0,)),  # 第三个输入张量，空张量
                    torch.nn.Embedding(0, 0, sparse=True),  # 对应的嵌入模块，空张量
                ),
                (
                    make_arg((2, 4), 0, 9),  # 第四个输入张量
                    torch.nn.Embedding(10, 0, sparse=True),  # 对应的嵌入模块，空嵌入
                ),
                (
                    make_arg((4,), 0, 21),  # 第五个输入张量
                    torch.nn.Embedding(22, 5, max_norm=1.0),  # 对应的嵌入模块，最大范数限制
                ),
                (
                    make_arg((2,), 0, 1),  # 第六个输入张量
                    torch.nn.Embedding.from_pretrained(
                        torch.arange(6.0).view(2, 3),  # 预训练权重
                        max_norm=2.0,
                        norm_type=0.5,
                        scale_grad_by_freq=False,
                        sparse=True,
                    ),  # 使用预训练权重创建的嵌入模块
                ),
            )

            for inp, module in nn_inps:
                # 遍历所有输入和对应的嵌入模块

                kwargs = {
                    "weight": module.weight.detach(),
                    "padding_idx": module.padding_idx,
                    "max_norm": module.max_norm,
                    "norm_type": module.norm_type,
                    "scale_grad_by_freq": module.scale_grad_by_freq,
                    "sparse": module.sparse,
                }
                
                # 调用 torch.nn.functional.embedding 进行嵌入操作，并获取输出张量的形状
                out_size = torch.nn.functional.embedding(inp, **kwargs).size()

                def foo(x):
                    # 定义一个简单的函数，返回对输入进行嵌入操作的结果
                    return torch.nn.functional.embedding(inp, **kwargs)

                # 对 foo 函数进行 JIT 编译，用于后续的形状分析
                fn = torch.jit.trace(foo, (inp.detach(),), check_trace=False)

                # 调用自定义函数检查形状分析
                self.checkShapeAnalysis(
                    out_size, fn.graph, assert_propagation=True, constant_prop=False
                )
    def test_convolution_backward(self):
        # No opinfos for ops that are not part of the Python API
        # 该测试不涉及 Python API 中未包含的操作信息
        # Also, as the return shapes are the input, weight, and bias shape, there is no point
        # in a really complicated test
        # 因为返回的形状是输入、权重和偏置的形状，所以没有必要进行非常复杂的测试

        input = torch.randn(
            (16, 16, 8, 8), dtype=torch.float32, device="cpu", requires_grad=True
        )
        # 生成一个随机的输入张量，形状为 (16, 16, 8, 8)，在 CPU 上，需要计算梯度

        weight = torch.randn(
            (8, 4, 3, 3), dtype=torch.float32, device="cpu", requires_grad=True
        )
        # 生成一个随机的权重张量，形状为 (8, 4, 3, 3)，在 CPU 上，需要计算梯度

        out_grad = torch.randn((16, 8, 8, 8), dtype=torch.float32, device="cpu")
        # 生成一个随机的输出梯度张量，形状为 (16, 8, 8, 8)，在 CPU 上

        @torch.jit.script
        def conv_bwd(input, weight, grad):
            bias_sizes = [
                8,
            ]
            # 偏置的尺寸为 [8]
            args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            # 参数设置为 ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            # 调用 torch.ops.aten.convolution_backward 执行卷积的反向传播
            return torch.ops.aten.convolution_backward(
                grad, input, weight, bias_sizes, *args
            )

        # 使用自定义函数 assert_shape_equal_scripted 断言脚本化的卷积反向传播函数的输出形状
        self.assert_shape_equal_scripted(conv_bwd, (input, weight, out_grad))

        @torch.jit.script
        def conv_bwd_2(input, weight, grad):
            bias_sizes = None
            # 偏置尺寸为 None
            args = ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            # 参数设置为 ([1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, True])
            # 调用 torch.ops.aten.convolution_backward 执行卷积的反向传播
            return torch.ops.aten.convolution_backward(
                grad, input, weight, bias_sizes, *args
            )

        # 使用自定义函数 assert_shape_equal_scripted 断言另一个脚本化的卷积反向传播函数的输出形状
        self.assert_shape_equal_scripted(conv_bwd_2, (input, weight, out_grad))
    # 定义一个测试方法，用于测试返回输入符号形状的功能
    def test_returning_input_symbolic_shapes(self):
        # 冻结并脚本化一个 Conv2d 模型，并转为 TorchScript
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        # 获取模型计算图的输入节点列表
        inps = list(mm.graph.inputs())
        # 设置第二个输入节点的类型，指定为具有未知大小的四维张量
        inps[1].setType(inps[1].type().with_sizes([None, None, None, None]))
        # 通过传播形状在计算图上构建形状计算图
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        )
        # 部分评估形状计算图，以便将多个输出变为元组
        g = shape_compute_graph.partial_eval_shape_graph()
        # 从计算图创建一个 TorchScript 函数
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        # 使用指定的输入尺寸运行 TorchScript 函数
        out = func([20, 16, 5, 10])
        # 断言前四个输出应为从输入得到的未知符号形状
        self.assertEqual(out[0:4], [20, 16, 5, 10])
        # 断言最后两个输出应为新的符号维度 - 高度和宽度
        self.assertEqual(out[4:], list(mm(torch.rand([20, 16, 5, 10])).size()[2:]))

    # 定义一个测试方法，用于测试部分评估形状计算图中的 Conv2d 操作
    def test_partial_eval_graph_conv(self):
        # 冻结并脚本化一个 Conv2d 模型，并转为 TorchScript
        mm = torch.jit.freeze(torch.jit.script(nn.Conv2d(16, 33, 3, stride=2).eval()))
        # 通过传播形状在计算图上构建形状计算图
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mm.graph)
        )
        # 获取 Conv2d 操作节点的输出符号大小
        output_sizes = (
            mm.graph.findNode("aten::conv2d").output().type().symbolic_sizes()
        )
        # 断言索引为 0、2 和 3 的输出大小应小于 0
        for i in [0, 2, 3]:
            self.assertTrue(output_sizes[i] < 0)
        # 断言索引为 1 的输出大小应大于等于 0
        self.assertTrue(output_sizes[1] >= 0)
        # 部分评估形状计算图，以便将多个输出变为元组
        g = shape_compute_graph.partial_eval_shape_graph()
        # 从计算图创建一个 TorchScript 函数
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        # 创建一个输入张量
        inp = torch.randn(20, 16, 5, 10)
        # 使用指定的输入尺寸运行 TorchScript 函数
        output = func([20, 16, 5, 10])
        # 获取直接计算的输出形状并进行比较
        output_eager = list(mm(inp).size())
        for o, oe in zip(output, output_eager[0:1] + output_eager[2:]):
            self.assertEqual(o, oe)

    # 定义一个方法，用于检查符号形状计算的结果
    def checkSymShapeCompute(
        self, shape_compute_graph, nodes, node_output_sizes, shape_inputs
    ):
        # 此方法将用于检查符号形状计算的各个参数和输入
    ):
        # 计算图的部分评估形状计算图
        g = shape_compute_graph.partial_eval_shape_graph()
        # 断言输入节点的数量与形状输入的数量相同
        self.assertTrue(len(list(g.inputs())) == len(shape_inputs))
        # 获取图的输出到符号形状维度的映射
        output_sym_map = shape_compute_graph.graph_output_to_symbolic_shape_dim()
        # 符号形状到索引的映射
        sym_shape_to_index = {}
        # 遍历图的输出，建立符号形状到索引的映射
        for index, output in enumerate(g.outputs()):
            sym_shape_to_index[output_sym_map[output]] = index

        # 将多输出的图变为元组
        g.makeMultiOutputIntoTuple()
        # 从图创建一个 Torch 函数
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        # 使用形状输入运行函数，得到符号输出
        sym_outputs = func(*shape_inputs)

        # 遍历节点和其输出大小
        for node, output_shape in zip(nodes, node_output_sizes):
            # 获取节点输出的符号类型大小
            output_type_sizes = node.output().type().symbolic_sizes()
            # 遍历输出的符号形状及其索引
            for i, sym_shape in enumerate(output_type_sizes):
                # 如果符号形状是非负数，则断言其与预期输出形状的对应维度相等
                if sym_shape >= 0:
                    self.assertEqual(sym_shape, output_shape[i])
                # 否则，获取符号形状的索引并断言对应的符号输出与预期输出形状的对应维度相等
                else:
                    sym_shape_index = sym_shape_to_index[sym_shape]
                    self.assertEqual(sym_outputs[sym_shape_index], output_shape[i])

    def test_partial_eval_stitching(self):
        # 创建第一个卷积层
        conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # 创建最大池化层
        max_pool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        # 创建第二个卷积层
        conv2 = nn.Conv2d(
            64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        # 将卷积和池化层组合成序列模块，并冻结为 Torch 脚本
        mod = torch.jit.freeze(
            torch.jit.script(nn.Sequential(conv1, max_pool, conv2).eval())
        )

        # 对第一个卷积层进行前向传播
        conv1_output = conv1(torch.rand(1, 3, 224, 224))
        # 对最大池化层进行前向传播
        max_pool_output = max_pool(conv1_output)
        # 对第二个卷积层进行前向传播
        conv2_output = conv2(max_pool_output)

        # 运行形状传播并构建计算图
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        )
        # 查找最大池化节点和所有卷积节点
        nodes = [mod.graph.findNode("aten::max_pool2d")] + list(
            mod.graph.findAllNodes("aten::conv2d")
        )
        # 设置预期输出形状
        output_shapes = [
            max_pool_output.size(),
            conv1_output.size(),
            conv2_output.size(),
        ]
        # 检查符号形状计算
        self.checkSymShapeCompute(
            shape_compute_graph, nodes, output_shapes, ([1, 3, 224, 224],)
        )
    def test_refinement_through_graph_stitching(self):
        # 定义一个名为TwoConvs的内部类，继承自torch.nn.Module
        class TwoConvs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为7x7，步长为2x2，填充为3x3，无偏置
                self.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                # 第二个卷积层，输入通道数为3，输出通道数为64，卷积核大小、步长和填充与第一层相同，无偏置
                self.conv2 = torch.nn.Conv2d(
                    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )

            def forward(self, x):
                # 对输入x分别应用两个卷积层
                a = self.conv1(x)
                b = self.conv2(x)
                # 返回两个卷积层输出的元素级加法结果
                return a + b

        # 将TwoConvs模型实例化、脚本化并冻结
        mod = torch.jit.freeze(torch.jit.script(TwoConvs()).eval())
        # 获取模型图的第一个输入张量
        inp_tensor = list(mod.graph.inputs())[1]
        # 设置输入张量的符号尺寸为[None, None, None, None]
        inp_tensor.setType(inp_tensor.type().with_sizes([None, None, None, None]))
        # 在模型图上执行形状传播
        torch._C._jit_pass_propagate_shapes_on_graph(mod.graph)
        # 获取模型图的第一个输出节点的输入列表
        outs = list(next(mod.graph.outputs()).node().inputs())
        # 获取第一个输出节点的符号尺寸
        out1 = outs[0].type().symbolic_sizes()
        # 获取第二个输出节点的符号尺寸
        out2 = outs[1].type().symbolic_sizes()
        # 断言第一个输出节点的第二维度与第二个输出节点的第二维度不相等
        self.assertTrue(out1[2] != out2[2])
        # 断言第一个输出节点的第三维度与第二个输出节点的第三维度不相等
        self.assertTrue(out1[3] != out2[3])
        # 通过合并两个卷积层的部分评估图，我们能够识别输出形状是等价的
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        # 再次获取第一个输出节点的符号尺寸
        out1 = outs[0].type().symbolic_sizes()
        # 再次获取第二个输出节点的符号尺寸
        out2 = outs[1].type().symbolic_sizes()
        # 断言两个输出节点的符号尺寸完全相等
        self.assertEqual(out1, out2)

    def test_stitching_multi_output(self):
        # 创建一个最大池化层，池化核大小为3x3，步长为2，填充为1，不进行膨胀，不进行上取整，返回池化索引
        max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
            return_indices=True,
        )
        # 创建一个随机张量，大小为[1, 3, 224, 224]
        tensor = torch.rand(1, 3, 224, 224)
        # 对最大池化层进行追踪，并将模型冻结
        mod = torch.jit.trace(max_pool, (tensor,))
        mod = torch.jit.freeze(mod.eval())
        # 获取模型图的第一个输入张量
        inp = list(mod.graph.inputs())[1]
        # 设置输入张量的符号尺寸为[None, None, None, None]
        inp.setType(inp.type().with_sizes([None, None, None, None]))
        # 获取模型对张量的第一个输出的尺寸
        output_tensor = list(mod(tensor)[0].size())
        # 在模型图上执行"lower_all_tuples"传递
        self.run_pass("lower_all_tuples", mod.graph)
        # 在模型图上执行形状传播，并构建计算
        shape_compute_graph = (
            torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(mod.graph)
        )
        # 查找模型图中的最大池化节点
        max_pool_node = mod.graph.findNode("aten::max_pool2d_with_indices")
        # 获取最大池化节点的所有输出
        outs = list(max_pool_node.outputs())
        # 断言最大池化节点的第一个输出符号尺寸与第二个输出符号尺寸相等
        self.assertEqual(
            outs[0].type().symbolic_sizes(), outs[1].type().symbolic_sizes()
        )
        # 部分评估形状图，并将多个输出转换为元组
        g = shape_compute_graph.partial_eval_shape_graph()
        g.makeMultiOutputIntoTuple()
        # 从形状图创建一个JIT函数
        func = torch._C._create_function_from_graph("partial_eval_graph", g)
        # 获取形状计算图的图输出到符号形状维度的映射
        mapping = shape_compute_graph.graph_output_to_symbolic_shape_dim()
        # 计算输出形状
        output_shape = func(tensor.size())
        # 断言前四个维度与输入张量的尺寸相同
        self.assertEqual(list(output_shape[0:4]), list(tensor.size()))
        # 断言剩余维度与模型输出张量的第二维度开始相同
        self.assertEqual(list(output_shape[4:]), output_tensor[2:])
    # 定义测试函数 test_sym_ir_parsing，用于测试符号 IR 解析功能
    def test_sym_ir_parsing(self):
        # 定义包含图形字符串的变量 graph_str1
        graph_str1 = """graph(%x.1 : Float(SS(-2), SS(-3))):
                        %3 : int = prim::Constant[value=1]()
                        %4 : Tensor = aten::add(%x.1, %x.1, %3)
                        return (%4)"""
        # 使用 torch._C.parse_ir 解析图形字符串，返回一个图形对象 g
        g = torch._C.parse_ir(graph_str1)
        # 获取图形对象 g 的第一个输入节点 inp
        inp = next(g.inputs())
        # 获取输入节点 inp 的类型，并获取其符号化大小
        out = inp.type().symbolic_sizes()
        # 使用 self.assertEqual 断言 out 应为 [-2, -3]
        self.assertEqual(out, [-2, -3])

    # 定义测试函数 test_stitching_concat，用于测试拼接操作的脚本函数
    def test_stitching_concat(self):
        # 定义脚本函数 foo1，接受四个参数 a, b, x, y，执行除法操作并拼接 x, y
        @torch.jit.script
        def foo1(a, b, x, y):
            return (a / b) + torch.cat([x, y])

        # 定义脚本函数 foo2，与 foo1 功能相同，但指定拼接维度为 -2
        @torch.jit.script
        def foo2(a, b, x, y):
            return (a / b) + torch.cat([x, y], dim=-2)

        # 遍历 foo1 和 foo2
        for foo in [foo1, foo2]:
            # 获取函数的计算图 g
            g = foo.graph
            # 遍历计算图的输入节点 inp
            for inp in foo.graph.inputs():
                # 将输入节点的类型设置为具有未指定大小的张量类型
                inp.setType(inp.type().with_sizes([None, None]))

            # 执行传播形状并构建计算图的形状计算图形
            shape_compute_graph = (
                torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
                    foo.graph
                )
            )
            # 定义要检查的节点列表 nodes
            nodes = (
                [g.findNode("aten::div")]
                + [g.findNode("aten::add")]
                + [g.findNode("aten::cat")]
            )

            # 定义输入参数 inps
            inps = [1, 10], [20, 10], [15, 1], [5, 1]
            # 定义预期的输出形状列表 output_shapes
            output_shapes = [[20, 10], [20, 10], [20, 1]]

            # 使用 self.checkSymShapeCompute 方法检查形状计算结果
            self.checkSymShapeCompute(shape_compute_graph, nodes, output_shapes, inps)

    # 定义测试函数 test_shape_function_includes，用于测试形状函数的包含情况
    @unittest.skipIf(
        not hasattr(torch.jit, "_shapes"), "shape functions not loaded in python"
    )
    def test_shape_function_includes(self):
        # 定义输入形状 inp_shape, weight_shape 等
        inp_shape = [1, 16, 5, 10]
        weight_shape = [33, 16, 3, 3]
        bias = None
        stride = [2, 2]
        padding = [0, 0]
        dilation = [1, 1]
        groups = 1
        # 执行 conv2d 形状函数，返回结果 res
        res = torch.jit._shapes.conv2d(
            inp_shape, weight_shape, bias, stride, padding, dilation, groups
        )
        # 使用 self.assertEqual 断言 res 应为 [1, 33, 2, 4]
        self.assertEqual(res, [1, 33, 2, 4])

        # 定义输入形状 m1_shape, m2_shape
        m1_shape = [10, 20]
        m2_shape = [20, 10]
        # 执行 matmul 形状函数，返回结果 res
        res = torch.jit._shapes.matmul(m1_shape, m2_shape)
        # 使用 self.assertEqual 断言 res 应为 [10, 10]
        self.assertEqual(res, [10, 10])
    # 定义测试函数，用于检查注册函数时的错误处理
    def test_register_function_error_checking(self):
        # 使用 torch.jit.script 装饰器将 foo 函数编译为 Torch 脚本
        @torch.jit.script
        def foo(x, y):
            return x + y

        # 查找 foo 函数图中的 "aten::add" 节点
        node = foo.graph.findNode("aten::add")

        # 使用 torch.jit.script 装饰器定义错误类型的输入类型函数 wrong_input_types
        @torch.jit.script
        def wrong_input_types(x, y):
            # 指定 x 为 int 类型的列表
            x: List[int] = []
            return x

        # 断言捕获 RuntimeError 异常，并检查是否包含 "Expected supertype of int" 字符串
        with self.assertRaisesRegex(RuntimeError, "Expected supertype of int"):
            # 将 wrong_input_types 函数的图形注册到 node 节点的形状计算图中
            torch._C._jit_register_shape_compute_graph_for_node(
                node, wrong_input_types.graph
            )

        # 使用 torch.jit.script 装饰器定义错误类型的输出类型函数 wrong_output_types
        @torch.jit.script
        def wrong_output_types(x: List[int], y: List[int]):
            # 指定 x 为 Tensor 类型的列表
            x: List[Tensor] = []
            return x

        # 断言捕获 RuntimeError 异常，并检查是否包含 "but got graph_type" 字符串
        with self.assertRaisesRegex(RuntimeError, "but got graph_type"):
            # 将 wrong_output_types 函数的图形注册到 node 节点的形状计算图中
            torch._C._jit_register_shape_compute_graph_for_node(
                node, wrong_output_types.graph
            )

        # 使用 torch.jit.script 装饰器定义参数过多的输入函数 too_many_inputs
        @torch.jit.script
        def too_many_inputs(x: List[int], y: List[int], z: Any, z2: Any):
            # 指定 x 为 int 类型的列表
            x: List[int] = []
            return x

        # 断言捕获 RuntimeError 异常
        with self.assertRaises(RuntimeError) as error:
            # 将 too_many_inputs 函数的图形注册到 node 节点的形状计算图中
            torch._C._jit_register_shape_compute_graph_for_node(
                node, too_many_inputs.graph
            )

        # 断言异常信息中包含 "fewer arguments than schema" 字符串
        self.assertTrue("fewer arguments than schema" in str(error.exception))

    # 定义测试交叉熵损失函数
    def test_cross_entropy_loss(self):
        # 使用 torch.jit.script 装饰器将 foo 函数编译为 Torch 脚本
        @torch.jit.script
        def foo(x, y):
            return torch.ops.aten.cross_entropy_loss(x, y, reduction=0)

        # 获取 foo 函数图中的输入参数列表
        inputs = list(foo.graph.inputs())
        # 修改第一个输入参数的类型为指定大小 [8, 2]
        inputs[0].setType(inputs[0].type().with_sizes([8, 2]))
        # 修改第二个输入参数的类型为指定大小 [8]
        inputs[1].setType(
            inputs[1]
            .type()
            .with_sizes(
                [
                    8,
                ]
            )
        )
        # 对 foo 函数的图形执行形状传播
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        # 断言 foo 函数的输出类型大小为 [8]
        self.assertEqual(
            next(foo.graph.outputs()).type().sizes(),
            [
                8,
            ],
        )

    # 定义测试挤压维度函数
    def test_squeeze_dims(self):
        # 使用 torch.jit.script 装饰器将 foo 函数编译为 Torch 脚本
        @torch.jit.script
        def foo(x):
            return torch.ops.aten.squeeze(x, dim=0)

        # 获取 foo 函数图中的输入参数
        input = next(foo.graph.inputs())
        # 修改输入参数的类型，使其符号大小为 [5, 8]
        input.setType(input.type().with_sizes([1, 5, 8]))
        # 对 foo 函数的图形执行形状传播
        torch._C._jit_pass_propagate_shapes_on_graph(foo.graph)
        # 断言 foo 函数的输出类型的符号大小为 [5, 8]
        self.assertEqual(next(foo.graph.outputs()).type().symbolic_sizes(), [5, 8])
```