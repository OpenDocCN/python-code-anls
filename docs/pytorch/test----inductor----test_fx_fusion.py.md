# `.\pytorch\test\inductor\test_fx_fusion.py`

```
# Owner(s): ["module: inductor"]

# 引入必要的模块和函数类型
from typing import Any, Callable

import torch
from torch._inductor.fx_passes.pre_grad import (
    linear_permute_fusion,
    linear_transpose,
    permute_linear_fusion,
    permute_matmul_fusion,
    sink_cat_after_pointwise,
    transpose_linear,
    transpose_matmul,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.passes.shape_prop import ShapeProp

# 定义函数类型 PassFunc，接受 torch.fx.GraphModule 和任意类型输入，返回 torch.fx.GraphModule
PassFunc = Callable[[torch.fx.GraphModule, Any], torch.fx.GraphModule]

# 定义一个函数 chain_passes，接受多个 PassFunc 类型参数，返回一个 PassFunc 类型
def chain_passes(*passes: PassFunc) -> PassFunc:
    # 定义内部函数 parent_pass，接受 torch.fx.GraphModule 和任意类型输入，返回 torch.fx.GraphModule
    def parent_pass(module: torch.fx.GraphModule, input: Any) -> torch.fx.GraphModule:
        # 遍历每一个传入的 pass_ 函数
        for pass_ in passes:
            # 如果 module 是 torch.fx.GraphModule 类型，则执行 ShapeProp 的 propagate 方法
            if isinstance(module, torch.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            # 将 module 输入当前的 pass_ 函数，并更新 module
            module = pass_(module)
        # 返回经过所有 passes 处理后的 module
        return module

    return parent_pass

# 定义函数 count_call，接受 torch.fx.GraphModule、字符串 op 和任意目标操作 target_op，返回整数
def count_call(module: torch.fx.GraphModule, op: str, target_op: Any) -> int:
    # 统计所有节点中满足条件的数量
    return sum(
        1 if (n.op == op and n.target == target_op) else 0 for n in module.graph.nodes
    )

# 定义函数 count_call_function，调用 count_call 函数，统计 call_function 类型操作的数量
def count_call_function(module: torch.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_function", target_op)

# 定义函数 count_call_method，调用 count_call 函数，统计 call_method 类型操作的数量
def count_call_method(module: torch.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_method", target_op)

# 定义测试类 TestFxFusion，继承自 TestCase
class TestFxFusion(TestCase):
    # 定义测试方法 test_sink_cat_after_pointwise
    def test_sink_cat_after_pointwise(self):
        # 定义多个测试函数，每个函数接受两个参数 x 和 y
        def test_kwarg(x, y):
            return torch.cat([x, y], dim=-1).view(-1).view(128).tanh()

        def test_arg(x, y):
            return torch.cat([x, y], -1).view(-1).view(128).tanh()

        def test_arg2(x, y):
            return torch.cat([x, y]).view(-1).view(128).tanh()

        def test_kwarg2(x, y):
            return torch.cat(tensors=[x, y], dim=0).tanh()

        def test_kwarg3(x, y):
            return torch.cat(tensors=[x, y], dim=0).view(128).tanh()

        # 定义追踪函数 trace_func，应用 symbolic_trace 和 sink_cat_after_pointwise 两个 passes
        trace_func = chain_passes(torch.fx.symbolic_trace, sink_cat_after_pointwise)
        # 定义输入列表 inputs，包含两个随机张量
        inputs = [
            torch.randn(8, 8),
            torch.randn(8, 8),
        ]
        # 对每个测试函数 f 进行追踪，并验证追踪结果和原始函数执行结果的一致性，以及 tanh 方法的调用次数
        for f in [test_kwarg, test_arg, test_arg2, test_kwarg2, test_kwarg3]:
            traced = trace_func(f, inputs)
            self.assertTrue(torch.allclose(f(*inputs), traced(*inputs)))
            self.assertEqual(count_call_method(traced, "tanh"), 2)
    # 定义一个测试函数，用于测试线性和置换融合的功能
    def test_linear_permute_fusion(self):
        # 定义一个测试模块类，继承自torch.nn.Module
        class TestModule(torch.nn.Module):
            # 初始化方法，接收参数 k（输入特征维度）、n（输出特征维度）、has_bias（是否有偏置）
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                # 初始化权重参数，维度为(n, k)
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                # 如果有偏置，则初始化偏置参数，维度为(n,)
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            # 前向传播方法，接收输入张量 input（维度为(batch_size, m, k)）
            def forward(self, input: torch.Tensor):
                # 如果有偏置，则使用带偏置的线性函数进行计算
                if self.has_bias:
                    a0 = torch.nn.functional.linear(input, self.weight, self.bias)
                else:
                    # 否则使用不带偏置的线性函数进行计算
                    a0 = torch.nn.functional.linear(input, self.weight)
                # 将结果张量 a0 进行维度置换，维度变为(batch_size, k, n)
                b0 = a0.permute(0, 2, 1)
                # 返回置换后的结果张量 b0
                return b0

        # 定义测试参数 m=16, k=8, n=4
        m, k, n = 16, 8, 4
        # 定义函数链，将 symbolic_trace 和 linear_permute_fusion 作为函数进行链式调用
        trace_func = chain_passes(torch.fx.symbolic_trace, linear_permute_fusion)
        # 遍历是否有偏置的可能性，True 和 False
        for has_bias in [True, False]:
            # 创建测试模块的实例，设置为评估模式
            module = TestModule(k, n, has_bias).eval()
            # 创建输入张量，维度为(6, m, k)
            input = torch.randn(6, m, k)
            # 对模块进行函数追踪
            traced = trace_func(module, [input])
            # 统计追踪后图中 torch.nn.functional.linear 函数调用的次数
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            # 统计追踪后图中 linear_transpose 函数调用的次数
            num_linear_transpose = count_call_function(traced, linear_transpose)
            # 断言线性函数调用次数为0
            self.assertEqual(num_linear, 0)
            # 断言置换线性函数调用次数为1
            self.assertEqual(num_linear_transpose, 1)
            # 断言模块处理输入张量后的输出与追踪后的输出张量在数值上相等
            self.assertTrue(torch.allclose(module(input), traced(input)))

    # 定义一个测试函数，用于测试置换和线性融合的功能
    def test_permute_linear_fusion(self):
        # 定义一个测试模块类，继承自torch.nn.Module
        class TestModule(torch.nn.Module):
            # 初始化方法，接收参数 k（输入特征维度）、n（输出特征维度）、has_bias（是否有偏置）
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                # 初始化权重参数，维度为(n, k)
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                # 如果有偏置，则初始化偏置参数，维度为(n,)
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            # 前向传播方法，接收输入张量 input（维度为(batch_size, k, m)）
            def forward(self, input: torch.Tensor):
                # 将输入张量进行维度置换，维度变为(batch_size, m, k)
                input1 = input.permute(0, 2, 1)
                # 如果有偏置，则使用带偏置的线性函数进行计算
                if self.has_bias:
                    return torch.nn.functional.linear(input1, self.weight, self.bias)
                # 否则使用不带偏置的线性函数进行计算
                return torch.nn.functional.linear(input1, self.weight)

        # 定义测试参数 m=16, k=8, n=4
        m, k, n = 16, 8, 4
        # 定义函数链，将 symbolic_trace 和 permute_linear_fusion 作为函数进行链式调用
        trace_func = chain_passes(torch.fx.symbolic_trace, permute_linear_fusion)
        # 遍历是否有偏置的可能性，True 和 False
        for has_bias in [True, False]:
            # 创建测试模块的实例，设置为评估模式
            module = TestModule(k, n, has_bias).eval()
            # 创建输入张量，维度为(6, k, m)
            input = torch.randn(6, k, m)
            # 对模块进行函数追踪
            traced = trace_func(module, [input])
            # 统计追踪后图中 torch.nn.functional.linear 函数调用的次数
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            # 统计追踪后图中 transpose_linear 函数调用的次数
            num_transpose_linear = count_call_function(traced, transpose_linear)
            # 断言线性函数调用次数为0
            self.assertEqual(num_linear, 0)
            # 断言置换线性函数调用次数为1
            self.assertEqual(num_transpose_linear, 1)
            # 断言模块处理输入张量后的输出与追踪后的输出张量在数值上相等
            self.assertTrue(torch.allclose(module(input), traced(input)))
    # 定义一个测试类，用于测试矩阵置换与矩阵乘法融合的功能
    def test_permute_bmm_fusion(self):
        # 定义一个继承自torch.nn.Module的测试模块类
        class TestModule(torch.nn.Module):
            # 初始化方法，接受批次数batch，矩阵列数k，矩阵行数n作为参数
            def __init__(self, batch: int, k: int, n: int):
                super().__init__()
                # 随机生成一个形状为(batch, k, n)的张量作为模块的属性other
                self.other = torch.randn(batch, k, n)

            # 前向传播方法，接受一个torch.Tensor类型的输入input
            def forward(self, input: torch.Tensor):
                # 对输入input进行置换，交换第1维与第3维的位置
                input1 = input.permute(0, 2, 1)
                # 执行torch.bmm操作，计算input1与模块属性other的矩阵乘积
                output = torch.bmm(input1, self.other)
                # 返回乘积结果output
                return output

        # 设定测试所需的批次数batch，矩阵列数k，矩阵行数n
        batch, m, k, n = 6, 16, 8, 4

        # 定义一个函数链，trace_func包含torch.fx.symbolic_trace和permute_matmul_fusion
        trace_func = chain_passes(torch.fx.symbolic_trace, permute_matmul_fusion)
        # 创建一个测试模块实例，并设定为评估模式
        module = TestModule(batch, k, n).eval()
        # 随机生成一个形状为(batch, k, m)的张量作为输入
        input = torch.randn(batch, k, m)
        # 对测试模块进行追踪，并应用函数链进行变换
        traced = trace_func(module, [input])
        # 统计在追踪后的模块中torch.bmm函数的调用次数
        num_bmm = count_call_function(traced, torch.bmm)
        # 统计在追踪后的模块中transpose_matmul函数的调用次数
        num_transpose_matmul = count_call_function(traced, transpose_matmul)
        # 断言num_bmm应为0
        self.assertEqual(num_bmm, 0)
        # 断言num_transpose_matmul应为1
        self.assertEqual(num_transpose_matmul, 1)

        # 断言模块的前向传播输出与追踪后的模块前向传播输出的接近程度
        self.assertTrue(torch.allclose(module(input), traced(input)))
# 如果当前脚本作为主程序执行（而不是被导入作为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```