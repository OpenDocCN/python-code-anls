# `.\pytorch\test\jit\test_autodiff_subgraph_slicing.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需的模块和库
import os
import sys
import unittest

import torch
from torch.testing._internal.common_jit import check_against_reference
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    num_profiled_runs,
    ProfilingMode,
)

# Make the helper files in test/ importable
# 获取当前脚本文件的上级目录，并将其添加到系统路径中，使得test目录下的文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from typing import List, Optional, Tuple
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import (
    disable_autodiff_subgraph_inlining,
    JitTestCase,
)

# 如果脚本文件被直接执行，则抛出异常，建议通过指定的方式来运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 跳过测试条件：如果运行在简单执行器下，则跳过，因为简单执行器不支持梯度
@unittest.skipIf(
    GRAPH_EXECUTOR == ProfilingMode.SIMPLE, "Simple Executor doesn't support gradients"
)
class TestAutodiffSubgraphSlicing(JitTestCase):
    # TODO: It is better if we can test directly on graphs instead of the current
    # end-to-end fashion.
    
    # 执行自动微分子图切片的操作函数
    def _perform_ad_subgraph_slicing(self, fn, *input_sizes):
        # 禁用自动微分子图内联
        with disable_autodiff_subgraph_inlining():
            # 启用性能测试模式
            with enable_profiling_mode_for_profiling_tests():
                # 将函数转换为JIT脚本
                ge = torch.jit.script(fn)
                # 生成输入张量，要求计算梯度
                inputs = [torch.randn(size, requires_grad=True) for size in input_sizes]
                # 运行函数，同时进行分析和重播
                ge(*inputs, profile_and_replay=True)
                # 返回生成的计算图
                return ge.graph_for(*inputs)

    # 断言函数生成的计算图的节点数与期望值相等
    def assertGraphSize(self, graph, size):
        nodes = list(
            filter(
                lambda n: (
                    n.kind() != "prim::BailOut"
                    and n.kind() != "prim::BailoutTemplate"
                    and n.kind() != "prim::TypeCheck"
                    and n.kind() != "prim::RequiresGradCheck"
                ),
                graph.nodes(),
            )
        )
        # 断言计算图中节点的数量
        self.assertEqual(len(list(nodes)), size)

    # 测试 torch.chunk 函数在脚本模式下的行为
    def test_chunk_constant_script_ad(self):
        # 定义一个接受输入张量 x 并对其执行 torch.chunk 操作的脚本函数
        @torch.jit.script
        def func(x):
            x1, x2 = torch.chunk(x, 2)
            return (x1, x2)

        # 创建一个形状为 [6, 10] 的随机张量，并要求其计算梯度
        input = torch.rand(6, 10).requires_grad_()
        # 禁用自动微分子图内联
        with disable_autodiff_subgraph_inlining():
            # 启用性能测试模式
            with enable_profiling_mode_for_profiling_tests():
                # 执行脚本函数，并进行分析和重播
                output = func(input, profile_and_replay=True)
                # 使用 FileCheck 检查不包含 "prim::DifferentiableGraph" 的输出
                FileCheck().check_not("prim::DifferentiableGraph").run(
                    func.graph_for(input)
                )

    # 跳过测试条件：如果不运行在性能分析执行器下，则跳过，因为该阈值仅适用于性能分析执行器
    @unittest.skipIf(
        GRAPH_EXECUTOR != ProfilingMode.PROFILING,
        "This threshold is only valid for Profiling Executor",
    )
    def test_diff_graph_inline_threshold(self):
        # 启用性能分析模式以便进行性能测试
        with enable_profiling_mode_for_profiling_tests():
            # 设置运行次数为1次
            NUM_RUNS = 1
            with num_profiled_runs(NUM_RUNS):

                @torch.jit.script
                def foo(x):
                    # 两个节点应该被融合
                    # 参见 https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/runtime/graph_executor_impl.h#L49
                    return torch.sigmoid(torch.sigmoid(x))

                @torch.jit.script
                def bar(x):
                    # 两个节点不应该被融合
                    return torch.sigmoid(x)

                # 创建一个形状为[4, 4]的随机张量，并设置需要计算梯度
                input = torch.rand([4, 4], requires_grad=True)

                # 调用foo函数两次
                foo(input)
                foo(input)

                # 调用bar函数两次
                bar(input)
                bar(input)

                # 断言foo函数的计算图中包含一个 "prim::DifferentiableGraph" 节点
                self.assertGraphContainsExactly(
                    foo.graph_for(input), "prim::DifferentiableGraph", 1
                )
                # 断言bar函数的计算图中包含零个 "prim::DifferentiableGraph" 节点
                self.assertGraphContainsExactly(
                    bar.graph_for(input), "prim::DifferentiableGraph", 0
                )

    def test_bias_as_module_attr(self):
        # 启用性能分析模式以便进行性能测试
        with enable_profiling_mode_for_profiling_tests():

            # 定义一个继承自torch.nn.Module的类M
            class M(torch.nn.Module):
                def __init__(self, has_bias):
                    super().__init__()
                    # 创建一个线性层，输入和输出都是大小为10的向量，根据has_bias参数决定是否有偏置
                    self.ll = torch.nn.Linear(10, 10, has_bias)

                def forward(self, x, y):
                    # 模型的前向传播，计算线性层输出，并进行一些操作
                    return self.ll(x + y) * x + y

            # 创建一个形状为[10, 10]的随机张量，并设置需要计算梯度
            x = torch.rand(10, 10, requires_grad=True)

            # 创建一个没有偏置的M类的实例
            no_bias = M(False)
            # 对没有偏置的M类实例进行脚本化
            scripted_no_bias = torch.jit.script(no_bias)
            # 调用脚本化的没有偏置的模型三次
            scripted_no_bias(x, x)
            scripted_no_bias(x, x)
            scripted_no_bias(x, x)

            # 创建一个有偏置的M类的实例
            has_bias = M(True)
            # 使用参考值对没有偏置的模型进行检查
            check_against_reference(
                self,
                scripted_no_bias,
                no_bias,
                lambda x: x,
                (
                    x,
                    x,
                ),
                check_types=False,
            )
            # 对有偏置的M类实例进行脚本化
            scripted_has_bias = torch.jit.script(has_bias)
            # 调用脚本化的有偏置的模型三次
            scripted_has_bias(x, x)
            scripted_has_bias(x, x)
            scripted_has_bias(x, x)
            # 使用参考值对有偏置的模型进行检查
            check_against_reference(
                self,
                scripted_has_bias,
                has_bias,
                lambda x: x,
                (
                    x,
                    x,
                ),
                check_types=False,
            )
    def test_constructed_bias(self):
        # 在测试方法开始时启用性能分析模式，专门用于性能分析的测试
        with enable_profiling_mode_for_profiling_tests():
            
            # 定义一个方法 method1，接受参数 x, weight, b1, b2，计算线性操作并加入偏置
            def method1(x, weight, b1, b2):
                # 计算偏置，这里是两个张量的乘积
                bias = b1 * b2
                # 使用 PyTorch 的线性函数计算结果，加上指定的偏置
                return torch.nn.functional.linear(x, weight, bias)

            # 设置测试数据维度为 N*N
            N = 10
            # 创建一个随机张量 x，需要梯度计算
            x = torch.rand(N, N, requires_grad=True)
            # 创建一个随机张量 weight，需要梯度计算
            weight = torch.rand(N, N, requires_grad=True)
            # 创建一个随机张量 b1，需要梯度计算
            b1 = torch.rand(N, N, requires_grad=True)
            # 创建一个随机张量 b2，需要梯度计算
            b2 = torch.rand(N, N, requires_grad=True)
            
            # 使用自定义方法 checkScript 对 method1 进行脚本化，并保存结果
            scripted = self.checkScript(method1, (x, weight, b1, b2))
            # 调用 check_against_reference 方法，比较脚本化方法和原始方法的输出
            # check_types 需要在 scripted 上设置 last_graph，这里我们跳过这一步
            check_against_reference(
                self,
                scripted,
                method1,
                lambda x: x,
                (x, weight, b1, b2),
                check_types=False,
            )

    def test_bias_as_arg(self):
        # 在测试方法开始时启用性能分析模式，专门用于性能分析的测试
        with enable_profiling_mode_for_profiling_tests():
            
            # 定义一个方法 method1，接受参数 x, weight, bias（可选的 torch.Tensor 类型）
            def method1(x, weight, bias: Optional[torch.Tensor]):
                # 使用 PyTorch 的线性函数计算结果，加上指定的偏置并应用 ReLU 激活函数再加 2
                return torch.nn.functional.linear(x, weight, bias).relu() + 2

            # 设置测试数据维度为 N*N
            N = 10
            # 创建一个随机张量 x，需要梯度计算
            x = torch.rand(N, N, requires_grad=True)
            # 创建一个随机张量 weight，需要梯度计算
            weight = torch.rand(N, N, requires_grad=True)
            # 初始化 bias 为 None
            bias = None
            # 使用自定义方法 checkScript 对 method1 进行脚本化，并保存结果
            scripted = self.checkScript(method1, (x, weight, bias))
            # 调用 check_against_reference 方法，比较脚本化方法和原始方法的输出
            # check_types 需要在 scripted 上设置 last_graph，这里我们跳过这一步
            check_against_reference(
                self,
                scripted,
                method1,
                lambda x: x,
                (x, weight, bias),
                check_types=False,
            )
            
            # 重新设置 bias 为一个随机张量，需要梯度计算
            bias = torch.rand(N, N, requires_grad=True)
            # 使用自定义方法 checkScript 对 method1 进行脚本化，并保存结果
            scripted = self.checkScript(method1, (x, weight, bias))
            # 调用 check_against_reference 方法，比较脚本化方法和原始方法的输出
            # check_types 需要在 scripted 上设置 last_graph，这里我们跳过这一步
            check_against_reference(
                self,
                scripted,
                method1,
                lambda x: x,
                (x, weight, bias),
                check_types=False,
            )
    def test_requires_grad_for_tensor_list(self):
        with enable_profiling_mode_for_profiling_tests():
            # output & var_list[0] should have requires_grad set to True
            # 定义一个函数 func，接受两个 torch.Tensor 输入，返回一个 torch.Tensor 输出和一个 torch.Tensor 列表
            def func(
                input0: torch.Tensor, input1: torch.Tensor
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
                # 创建包含两个输入 tensor 的列表 var_list
                var_list = [input0, input1]
                # 将 var_list 中的 tensor 连接成一个新的 tensor var
                var = torch.cat(var_list)
                # 对 var 加上常数 1.0，得到输出 output
                output = var + 1.0
                return output, var_list

            # 使用 torch.jit.script 对 func 进行脚本化
            jit_f = torch.jit.script(func)
            # 创建一个随机的 requires_grad=True 的输入 tensor input0
            input0 = torch.randn((2,), requires_grad=True)
            # 创建一个随机的输入 tensor input1
            input1 = torch.randn((2,))
            # 调用 func 获取参考输出 output_ref
            output_ref = func(input0, input1)
            # 针对两个输入进行迭代
            for i in range(2):
                # 使用 jit_f 运行输入，得到输出 output
                output = jit_f(input0, input1)
                # 检查输出的 requires_grad 属性与参考输出的 requires_grad 属性是否相同
                assert output_ref[0].requires_grad == output[0].requires_grad
                # 检查输出列表中第一个 tensor 的 requires_grad 属性与参考输出列表中第一个 tensor 的 requires_grad 属性是否相同
                assert output_ref[1][0].requires_grad == output[1][0].requires_grad
                # 检查输出列表中第二个 tensor 的 requires_grad 属性与参考输出列表中第二个 tensor 的 requires_grad 属性是否相同
                assert output_ref[1][1].requires_grad == output[1][1].requires_grad

    @unittest.skip(
        "disable until we property handle tensor lists with undefined gradients"
    )
    def test_differentiable_graph_ops_requires_grad(self):
        # 创建一个 requires_grad=True 的随机 tensor x
        x = torch.randn(8, 2, dtype=torch.float).requires_grad_()
        # 创建一个随机 tensor y
        y = torch.randn(8, 2, dtype=torch.float)

        # 定义一个函数 t，接受一个 torch.Tensor x，一个 torch.Tensor y，一个 bool 类型的标志 flag
        def t(x: torch.Tensor, y: torch.Tensor, flag: bool):
            # 对 x 加上常数 1.0，得到 o1
            o1 = x + 1.0
            # 对 o1 应用 ReLU 激活函数，得到 o1
            o1 = torch.relu(o1)
            # 对 y 加上常数 1.5，得到 o2
            o2 = y + 1.5
            # 对 o2 应用 ReLU 激活函数，得到 o2
            o2 = torch.relu(o2)
            # 将 o1 和 o2 相加，得到 o3
            o3 = o1 + o2

            # 根据 flag 的值进行条件分支
            if flag:
                # 如果 flag 为 True，执行以下操作
                # 对 o1 加上常数 1.0，得到 oo1
                oo1 = o1 + 1.0
                # 对 oo1 应用 ReLU 激活函数，得到 oo1
                oo1 = torch.relu(oo1)
                # 对 o2 加上常数 2.5，得到 oo2
                oo2 = o2 + 2.5
                # 对 oo2 应用 ReLU 激活函数，得到 oo2
                oo2 = torch.relu(oo2)
                # 将 oo1 和 oo2 相加，得到 oo3
                oo3 = oo1 + oo2
            else:
                # 如果 flag 为 False，执行以下操作
                # 对 o1 乘以常数 1.0，得到 oo1
                oo1 = o1 * 1.0
                # 对 oo1 应用 ReLU 激活函数，得到 oo1
                oo1 = torch.relu(oo1)
                # 对 o2 乘以常数 2.0，得到 oo2
                oo2 = o2 * 2.0
                # 对 oo2 应用 ReLU 激活函数，得到 oo2
                oo2 = torch.relu(oo2)
                # 将 oo1 和 oo2 相加，得到 oo3
                oo3 = oo1 + oo2

            return o1, o2, o3, oo1, oo2, oo3

        with enable_profiling_mode_for_profiling_tests():
            # 使用 torch.jit.script 对函数 t 进行脚本化
            t_jit = torch.jit.script(t)
            # 分别使用 t_jit 和原始函数 t 运行输入 x, y, False
            jit_o = t_jit(x, y, False)
            o = t(x, y, False)

            # 使用 FileCheck 检查 t_jit 对象的图是否包含 "prim::DifferentiableGraph"
            FileCheck().check("prim::DifferentiableGraph").run(
                t_jit.graph_for(x, y, False)
            )

            # 验证不同iableGraphOps 是否正确标记 requires_grad
            for oo, jit_oo in zip(o, jit_o):
                self.assertEqual(oo.requires_grad, jit_oo.requires_grad)
                self.assertEqual(oo, jit_oo)

            # 再次运行一次以触发融合
            jit_o = t_jit(x, y, False)
            for oo, jit_oo in zip(o, jit_o):
                self.assertEqual(oo.dtype, jit_oo.dtype)
                self.assertEqual(oo.requires_grad, jit_oo.requires_grad)
                self.assertEqual(oo, jit_oo)

    @unittest.skipIf(
        GRAPH_EXECUTOR == ProfilingMode.PROFILING,
        "Simple Executor doesn't support gradients",
    )
    def test_prune_grad(self):
        # 定义一个 Torch Script 函数 t，计算 input 和 bias 的 ReLU
        @torch.jit.script
        def t(input, bias):
            return torch.nn.functional.relu(input + bias)

        # 创建一个随机张量 input，需要梯度
        input = torch.randn(2, 8, requires_grad=True)
        # 创建一个随机张量 bias，不需要梯度
        bias = torch.randn(8, requires_grad=False)  # bias does NOT require grad
        NUM_PROFILED_RUNS = 1
        # 设置进行性能分析的运行次数
        with num_profiled_runs(NUM_PROFILED_RUNS):
            WARMUP = 3  # 2 runs to reach backward + 1 to optimize it
            # 进行预热运行，以便优化反向传播计算
            for x in range(WARMUP):
                # 调用 t 函数进行前向传播
                o = t(input, bias)
                # 对输出 o 进行求和并执行反向传播
                o.sum().backward()

            # 获取前向计划的调试状态，并从中获取第一个执行计划的反向图
            fwd_plan = list(t.get_debug_state().execution_plans.values())[0]
            bwd_graph = list(
                fwd_plan.code.grad_executor_states()[0].execution_plans.values()
            )[0].graph
            # 获取反向图的输出并检查其输入节点数是否为1
            tup = next(bwd_graph.outputs())
            self.assertEqual(len(list(tup.node().inputs())), 1)

    def test_simple_merge(self):
        # 定义一个简单的函数 fn，计算 x、y 和 z 的乘积
        # 返回值 b 是通过两次乘法得到的结果
        def fn(x, y, z):
            a = x * y
            b = a * z
            return b

        # 执行自动微分子图分片，并检查结果图的大小和内容
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_simple_no_merge(self):
        # 定义一个函数 fn，计算 x 和 y 的乘积，并创建一个大小由 y 绝对值决定的零张量
        def fn(x, y, z):
            a = x * y
            b = torch.zeros([abs(int(y))])
            return a, b

        # 执行自动微分子图分片，并检查结果图中的内容
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)
        g_str = str(graph)
        FileCheck().check("aten::Int").check("aten::zeros").check_not("aten::mul").run(
            g_str[0 : g_str.find("return")]
        )
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_does_not_merge_unrelated(self):
        # 定义一个函数 fn，计算 x 和 y 的乘积以及 w 和 z 的乘积
        def fn(w, x, y, z):
            a = x * y
            b = w * z
            return a, b

        # 执行自动微分子图分片，并检查结果图的大小和内容
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        self.assertGraphSize(graph, 3)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 2)

    def test_merges_without_cycles(self):
        # 定义一个函数 fn，展示无环的合并关系
        def fn(w, x, y):
            a = w * x
            b = a * y
            c = a * b
            return c

        # 执行自动微分子图分片，并检查结果图的大小和内容
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)

        self.assertGraphSize(graph, 1)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_merges_dense(self):
        # 定义一个函数 fn，展示密集的合并关系
        def fn(x, y):
            a, b = x.chunk(2)
            c, d = y.chunk(2)
            return a + c, b + d

        # 执行自动微分子图分片，并检查结果图的大小和内容
        graph = self._perform_ad_subgraph_slicing(fn, 2, 2)

        self.assertGraphSize(graph, 2)
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)
    def test_does_not_create_cycles(self):
        # 定义一个内部函数 fn，接受参数 w, x, y，并执行如下计算：
        # 计算 a = w * x
        # 根据 a 的绝对值创建一个全零张量 b
        # 计算 c = a * b
        # 返回结果 c
        def fn(w, x, y):
            a = w * x
            b = torch.zeros(abs(int(a)))
            c = a * b
            return c

        # 调用 _perform_ad_subgraph_slicing 方法对 fn 进行子图分析，并返回生成的图对象
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1)
        # 断言生成的图包含 "prim::DifferentiableGraph" 节点 2 次
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 2)

    def test_merges_up(self):
        # 定义一个内部函数 fn，接受参数 w, x, y, z，并执行如下计算：
        # 计算 a = w * x
        # 根据 y 的绝对值创建一个全零张量 b
        # 计算 c = a * z
        # 返回 b 和 c
        def fn(w, x, y, z):
            a = w * x
            b = torch.zeros(abs(int(y)))
            c = a * z
            return b, c

        # 调用 _perform_ad_subgraph_slicing 方法对 fn 进行子图分析，并返回生成的图对象
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)
        # 将生成的图对象转换为字符串形式 g_str
        g_str = str(graph)
        # 使用 FileCheck() 检查字符串 g_str 中 "aten::add" 不出现在返回语句之前的部分
        FileCheck().check_not("aten::add").run(g_str[0 : g_str.find("return")])
        # 断言生成的图包含 "prim::DifferentiableGraph" 节点 1 次
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_merges_down(self):
        # 定义一个内部函数 fn，接受参数 v, w, x, y，并执行如下计算：
        # 计算 a = v * w
        # 创建一个大小为 y 的全一张量 b
        # 计算 c = b * a
        # 返回 a 和 c
        def fn(v, w, x, y):
            a = v * w
            b = torch.ones(int(y))
            c = b * a
            return a, c

        # 调用 _perform_ad_subgraph_slicing 方法对 fn 进行子图分析，并返回生成的图对象
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1, 1, 1)

        # 如果 GRAPH_EXECUTOR 是 ProfilingMode.PROFILING，节点数为 4，否则为 3
        num_nodes = 4 if GRAPH_EXECUTOR == ProfilingMode.PROFILING else 3
        # 在生成的图对象的字符串表示中检查 "aten::add" 不出现在返回语句之前的部分
        g_str = str(graph)
        FileCheck().check_not("aten::add").run(g_str[0 : g_str.find("return")])
        # 断言生成的图包含 "prim::DifferentiableGraph" 节点 1 次
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 1)

    def test_respects_lexical_scoping(self):
        # 定义一个内部函数 fn，接受参数 x 和 k，并执行如下计算：
        # 计算 y = x * 1.1
        # 如果 k 为真值：
        #   计算 k = k + y
        # 计算 z = y * k
        # 返回 z 和 k
        def fn(x, k):
            y = x * 1.1
            if bool(k):
                k = k + y
            z = y * k
            return z, k

        # 调用 _perform_ad_subgraph_slicing 方法对 fn 进行子图分析，并返回生成的图对象
        graph = self._perform_ad_subgraph_slicing(fn, 1, 1)
        # 断言生成的图包含 "prim::DifferentiableGraph" 节点 3 次
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 3)

    def test_merge_respects_aliasing(self):
        # 定义一个内部函数 fn，接受参数 x, k, cond，并执行如下计算：
        # 计算 y = x * 1.1
        # 计算 y = y * k
        # 计算 y = y * 2.2
        # 如果 cond 为真值：
        #   从 y 中选择索引为 0 的值赋给 z1
        #   从 y 中选择索引为 1 的值赋给 z2
        #   z1 加上 3
        #   计算 out = z2 + k + 3.3
        #   计算 out = out * out
        # 返回 out
        def fn(x, k, cond):
            y = x * 1.1
            y = y * k
            y = y * 2.2
            if bool(cond):
                z1 = y[0]
                z2 = y[1]
                z1.add_(3)
                out = z2 + k + 3.3
                out = out * out
                return out

        # 调用 _perform_ad_subgraph_slicing 方法对 fn 进行子图分析，并返回生成的图对象
        graph = self._perform_ad_subgraph_slicing(fn, [2, 2], [2, 2], 1)
        # 在生成的图对象中使用 FileCheck() 检查匹配顺序，确保某些操作符的存在性
        FileCheck().check("prim::If").check("aten::select").check_next(
            "aten::select"
        ).check_next("aten::add_").check("Differentiable").run(graph)
        # 断言生成的图包含 "prim::DifferentiableGraph" 节点 2 次
        self.assertGraphContainsExactly(graph, "prim::DifferentiableGraph", 2)

    def test_aliased_outputs(self):
        with enable_profiling_mode_for_profiling_tests():
            # 启用性能分析模式
            # 情况 1：relu 和 t 之间的别名在 DifferentiableGraph 中。应该可以合并 relu 中的两个 split_with_sizes
            # 在一个图中
            input_str = """
    # 定义一个名为 graph 的函数，接受一个名为 %a 的张量参数
    graph(%a : Tensor):
        # 使用 PyTorch 的 aten::relu 操作对输入张量 %a 进行 ReLU 激活
        %b : Tensor = aten::relu(%a)
        # 对中间结果 %b 进行转置操作
        %2 : Tensor = aten::t(%b)
        # 返回转置后的张量 %2
        return (%2)
    """

    # 解析输入字符串 input_str 中的 IR（Intermediate Representation）
    graph = torch._C.parse_ir(input_str)
    # 在解析得到的图形中创建自动微分子图
    torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
    # 使用 FileCheck 运行验证，检查不同的子图是否存在
    FileCheck().check("with prim::DifferentiableGraph").check(
        "aten::relu"
    ).check("aten::t").run(graph)

    # Case 2: relu 和 split_with_sizes 之间的别名关系
    # 都是 Diff 图的输出。将 split_with_sizes 和 relu 合并到一个图中是无效的，
    # 因此 relu 和 split_with_sizes 应该在不同的可微分图中
    # 即 relu 和 split_with_sizes 应该在不同的可微分图中
    input_str = """
    # 定义一个名为 graph 的函数，接受一个名为 %a 的张量参数
    graph(%a : Tensor):
        # 使用 PyTorch 的 aten::relu 操作对输入张量 %a 进行 ReLU 激活
        %b : Tensor = aten::relu(%a)
        # 定义一个名为 %0 的常量张量，值为 [2, 2, 1]
        %0 : int[] = prim::Constant[value=[2, 2, 1]]()
        # 定义一个名为 %1 的常量整数，值为 0
        %1 : int = prim::Constant[value=0]()
        # 使用 PyTorch 的 aten::split_with_sizes 操作对张量 %b 进行切分
        %2 : Tensor[] = aten::split_with_sizes(%b, %0, %1)
        # 使用 prim::TupleConstruct 将张量 %b 和切分后的张量 %2 合并为元组
        %3 : (Tensor[], Tensor[]) = prim::TupleConstruct(%b, %2)
        # 返回包含元组 %3 的结果
        return (%3)
# Case 3: two aliased nodes in a graph.
# Both `split_with_sizes` should be unfused

# 定义输入字符串，表示一个计算图，输入是 %a Tensor
input_str = """
graph(%a : Tensor):
    %b : Tensor = aten::relu(%a)  # 计算 %b = relu(%a)
    %s1 : int[] = prim::Constant[value=[2, 2, 1]]()  # 创建常量数组 %s1 = [2, 2, 1]
    %s2 : int[] = prim::Constant[value=[3, 1]]()  # 创建常量数组 %s2 = [3, 1]
    %1 : int = prim::Constant[value=0]()  # 创建常量 %1 = 0
    %2 : Tensor[] = aten::split_with_sizes(%b, %s1, %1)  # 执行 split_with_sizes 操作，将 %b 按 %s1 切分
    %3 : Tensor[] = aten::split_with_sizes(%b, %s2, %1)  # 执行 split_with_sizes 操作，将 %b 按 %s2 切分
    %4 : (Tensor, Tensor[]) = prim::TupleConstruct(%b, %2, %3)  # 构造元组 %4，包含 %b, %2, %3
    return (%4)  # 返回元组 %4，包含 %b, %2, %3
"""

# 解析输入字符串生成计算图
graph = torch._C.parse_ir(input_str)
# 在计算图中创建自动微分子图
torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
# 使用 FileCheck 对象验证计算图中的结构
FileCheck().check("Tensor = prim::DifferentiableGraph").check(
    "with prim::DifferentiableGraph"
).check("Tensor = aten::relu").check_not("aten::split_with_sizes").run(
    graph
)
# 运行 FileCheck 对象，检查计算图中的特定模式和条件



# Case 4: the aliased output has a descendant
# Both should be unfused. Note, %3 comes before %2
# to test that we unfuse in the reverse topo order

# 定义输入字符串，表示一个计算图，输入是 %a Tensor
input_str = """
graph(%a : Tensor):
    %b : Tensor = aten::relu(%a)  # 计算 %b = relu(%a)
    %0 : int[] = prim::Constant[value=[2, 2, 1]]()  # 创建常量数组 %0 = [2, 2, 1]
    %1 : int = prim::Constant[value=0]()  # 创建常量 %1 = 0
    %2 : Tensor = aten::t(%b)  # 计算 %2 = t(%b)
    %3 : Tensor = aten::relu(%2)  # 计算 %3 = relu(%2)
    %4 : (Tensor, Tensor, Tensor[]) = prim::TupleConstruct(%b, %3, %2)  # 构造元组 %4，包含 %b, %3, %2
    return (%4)  # 返回元组 %4，包含 %b, %3, %2
"""

# 解析输入字符串生成计算图
graph = torch._C.parse_ir(input_str)
# 在计算图中创建自动微分子图
torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
# 使用 FileCheck 对象验证计算图中的结构
FileCheck().check("Tensor = prim::DifferentiableGraph").check(
    "with prim::DifferentiableGraph"
).check("Tensor = aten::relu").check_not("aten::t").run(graph)
# 运行 FileCheck 对象，检查计算图中的特定模式和条件



# Case 5: multiple aliased groups
# Both should be unfused. Note, %3 comes before %2
# to test that we unfuse in the reverse topo order

# 定义输入字符串，表示一个计算图，输入是 %a Tensor
input_str = """
graph(%a : Tensor):
    %b : Tensor = aten::relu(%a)  # 计算 %b = relu(%a)
    %c : Tensor = aten::abs(%a)  # 计算 %c = abs(%a)
    %0 : int[] = prim::Constant[value=[2, 2, 1]]()  # 创建常量数组 %0 = [2, 2, 1]
    %1 : int = prim::Constant[value=0]()  # 创建常量 %1 = 0
    %d : Tensor = aten::t(%c)  # 计算 %d = t(%c)
    %2 : Tensor = aten::t(%b)  # 计算 %2 = t(%b)
    %3 : Tensor = aten::relu(%2)  # 计算 %3 = relu(%2)
    %4 : (Tensor, Tensor, Tensor[]) = prim::TupleConstruct(%3, %2, %d, %b, %c, %b)  # 构造元组 %4，包含 %3, %2, %d, %b, %c, %b
    return (%4)  # 返回元组 %4，包含 %3, %2, %d, %b, %c, %b
"""

# 解析输入字符串生成计算图
graph = torch._C.parse_ir(input_str)
# 在计算图中创建自动微分子图
torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)
# 使用 FileCheck 对象验证计算图中的结构
FileCheck().check("Tensor = prim::DifferentiableGraph").check(
    "with prim::DifferentiableGraph"
).check("Tensor = aten::relu").check_not("aten::t").run(graph)
# 运行 FileCheck 对象，检查计算图中的特定模式和条件
# 解析输入的 IR 字符串，生成计算图
graph = torch._C.parse_ir(input_str)

# 在计算图中创建自动微分子图，参数 1 表示开启自动微分
torch._C._jit_pass_create_autodiff_subgraphs(graph, 1)

# 使用 FileCheck 运行检查，验证计算图中的特定模式
FileCheck().check("Tensor = prim::DifferentiableGraph").check(
    "with prim::DifferentiableGraph"
).check("Tensor = aten::relu").check_not("aten::t").run(graph)

def test_has_profiled_info_aliasing_outputs(self):
    # 预期 CallFunction 将阻止最终的 profile 节点合并到 DifferentiableGraph 中，
    # 而 create_autodiff_subgraphs 将会将其添加到 %4 的类型中。
    ir = """
    graph(%a : Tensor):
        %1 : Tensor = prim::profile[profiled_type=Float(requires_grad=0)](%a)
        %2 : Tensor = aten::relu(%1)
        %3 : Tensor = prim::profile[profiled_type=Float(requires_grad=0)](%2)
        %4 : Tensor = aten::relu(%3)
        %5 : Tensor = prim::CallFunction(%4)
        %6 : Tensor = prim::profile[profiled_type=Float(requires_grad=0)](%4)
        return (%6)
    """

    # 解析 IR 字符串，生成计算图
    graph = torch._C.parse_ir(ir)

    # 在计算图中创建自动微分子图
    torch._C._jit_pass_create_autodiff_subgraphs(graph)

    # 遍历计算图中的节点，找到类型为 "prim::DifferentiableGraph" 的子图
    for n in graph.nodes():
        if n.kind() == "prim::DifferentiableGraph":
            diff_graph = n.g("Subgraph")

    # 获取自动微分子图的输出节点列表
    outputs = list(diff_graph.outputs())

    # 断言自动微分子图只有一个输出
    self.assertEqual(1, len(outputs))

    # 获取输出节点
    output = outputs[0]

    # 断言输出节点不需要梯度（requiresGrad = False）
    self.assertEqual(False, output.requiresGrad())

    # 使用 FileCheck 运行检查，验证计算图中的特定模式
    FileCheck().check("= prim::DifferentiableGraph").check(
        "with prim::DifferentiableGraph"
    ).check(" = aten::relu").check("requires_grad=0").check("aten::relu").run(graph)
```