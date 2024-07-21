# `.\pytorch\test\fx\test_subgraph_rewriter.py`

```py
# Owner(s): ["module: fx"]

# 引入标准库模块
import os
import sys

# 引入 PyTorch 库及相关子模块
import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.annotate import annotate

# 使得 test/ 目录下的辅助文件可以被导入
from torch.fx.experimental.rewriter import RewritingTracer

# 获取当前脚本的测试目录路径
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# 将该路径添加到系统路径中，以便导入测试所需模块
sys.path.append(pytorch_test_dir)
# 从内部测试工具中导入 JitTestCase 类
from torch.testing._internal.jit_utils import JitTestCase

# 如果当前脚本被直接运行，则抛出运行时错误提示信息，建议正确的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )

# 使用 torch.fx.wrap 装饰器包装函数 wrapped_gemm_bias_mul，使其支持 FX 图模式
@torch.fx.wrap
def wrapped_gemm_bias_mul(a, b, bias):
    # 计算线性函数的结果
    lin_res = torch.nn.functional.linear(a, b, bias=bias)
    # 将线性函数的结果与 a 元素相乘，得到最终结果
    mul_res = lin_res * a
    return lin_res, mul_res

# 使用 torch.fx.wrap 装饰器包装函数 wrapped_gemm_bias_mul_with_c，使其支持 FX 图模式
@torch.fx.wrap
def wrapped_gemm_bias_mul_with_c(a, b, bias, c):
    # 计算线性函数的结果
    lin_res = torch.nn.functional.linear(a, b, bias=bias)
    # 将线性函数的结果与 c 元素相乘，得到最终结果
    mul_res = lin_res * c
    return lin_res, mul_res

# 定义 JitTestCase 的子类 TestSubgraphRewriter，用于测试子图重写功能
class TestSubgraphRewriter(JitTestCase):
    def test_subgraph_rewriter_preserves_logic(self):
        # 定义一个简单的模块 M，用于示例
        class M(torch.nn.Module):
            def forward(self, x):
                # 计算负值与 ReLU 函数的结果的和
                val = torch.neg(x) + torch.relu(x)
                # 返回结果的两倍
                return torch.add(val, val)

        # 定义一个模式函数 pattern，用于识别特定的子图结构
        def pattern(x):
            return torch.neg(x) + torch.relu(x)

        # 定义一个对比函数 comparison，用于比较实际输出与预期输出
        def comparison(x):
            val = torch.neg(x) + torch.relu(x)
            return torch.add(val, val)

        # 对模块 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比函数 comparison 进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建一个随机输入张量 x
        x = torch.rand(1, 3)

        # 使用 subgraph_rewriter.replace_pattern 替换模块中的子图
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        # 对跟踪得到的图进行检查
        traced.graph.lint()

        # 计算对比函数的输出
        ref_output = comparison_fn(x)
        # 计算跟踪得到的模块的输出
        test_output = traced.forward(x)
        # 断言实际输出与预期输出相等
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_with_oneliner_pattern(self):
        # 定义一个简单的模块 M，用于示例
        class M(torch.nn.Module):
            def forward(self, x):
                # 计算输入张量 x 的负值
                val = torch.neg(x)
                # 返回结果的两倍
                return torch.add(val, val)

        # 定义一个模式函数 pattern，用于识别特定的子图结构
        def pattern(x):
            return torch.neg(x)

        # 定义一个替换函数 replacement，用于替换匹配到的子图
        def replacement(x):
            return torch.relu(x)

        # 定义一个对比函数 comparison，用于比较实际输出与预期输出
        def comparison(x):
            val = torch.relu(x)
            return torch.add(val, val)

        # 对模块 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比函数 comparison 进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建一个随机输入张量 x
        x = torch.rand(1, 3)

        # 使用 subgraph_rewriter.replace_pattern 替换模块中的子图
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪得到的图进行检查
        traced.graph.lint()

        # 计算对比函数的输出
        ref_output = comparison_fn(x)
        # 计算跟踪得到的模块的输出
        test_output = traced.forward(x)
        # 断言实际输出与预期输出相等
        self.assertEqual(ref_output, test_output)
    def test_subgraph_rewriter_with_trivial_replacement(self):
        # 定义一个简单的神经网络模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, x):
                # 对输入 x 取负值
                val = torch.neg(x)
                # 将 val 加上自身，相当于 val * 2
                val = torch.add(val, val)
                # 再次将 val 加上自身，相当于 val * 2
                return torch.add(val, val)

        # 定义一个模式匹配函数，用于查找 torch.add(x, x) 这样的模式
        def pattern(x):
            return torch.add(x, x)

        # 定义一个替换函数，将匹配到的模式替换为 x
        def replacement(x):
            return x

        # 定义一个比较函数，对输入 x 取负值
        def comparison(x):
            return torch.neg(x)

        # 对 M 类进行符号化跟踪，得到 traced 对象
        traced = symbolic_trace(M())
        # 对比较函数进行符号化跟踪，得到 comparison_fn 对象
        comparison_fn = symbolic_trace(comparison)

        # 生成一个形状为 (1, 5) 的随机张量 x
        x = torch.randn(1, 5)

        # 使用 subgraph_rewriter.replace_pattern_with_filters 函数替换模式
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, pattern, replacement, []
        )

        # 对跟踪图进行检查
        traced.graph.lint()

        # 对比较函数应用于输入 x，得到参考输出 ref_output
        ref_output = comparison_fn(x)
        # 对跟踪对象 traced 应用于输入 x，得到测试输出 test_output
        test_output = traced.forward(x)
        # 检查替换是否成功，匹配数为 2 且每个匹配的替换数量为 0
        no_replacements = len(matches) == 2 and len(matches[1].replacements) == 0
        # 使用断言检查 ref_output 和 test_output 是否相等
        self.assertEqual(ref_output, test_output)
        # 使用断言检查 no_replacements 是否为 True
        self.assertTrue(no_replacements)

    def test_subgraph_rewriter_single_pattern_match(self):
        # 定义一个简单的神经网络模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义前向传播函数
            def forward(self, x):
                # 对输入 x 取负值并加上 torch.relu(x)
                val = torch.neg(x) + torch.relu(x)
                # 返回 val * 2
                return torch.add(val, val)

        # 定义一个模式匹配函数，用于查找 torch.neg(x) + torch.relu(x) 这样的模式
        def pattern(x):
            return torch.neg(x) + torch.relu(x)

        # 定义一个替换函数，将匹配到的模式替换为 torch.relu(x)
        def replacement(x):
            return torch.relu(x)

        # 定义一个比较函数，对输入 x 应用 torch.relu(x) 并加上自身
        def comparison(x):
            val = torch.relu(x)
            return torch.add(val, val)

        # 对 M 类进行符号化跟踪，得到 traced 对象
        traced = symbolic_trace(M())
        # 对比较函数进行符号化跟踪，得到 comparison_fn 对象
        comparison_fn = symbolic_trace(comparison)

        # 生成一个形状为 (1, 3) 的随机张量 x
        x = torch.rand(1, 3)

        # 使用 subgraph_rewriter.replace_pattern 函数替换模式
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪图进行检查
        traced.graph.lint()

        # 对比较函数应用于输入 x，得到参考输出 ref_output
        ref_output = comparison_fn(x)
        # 对跟踪对象 traced 应用于输入 x，得到测试输出 test_output
        test_output = traced.forward(x)
        # 使用断言检查 ref_output 和 test_output 是否相等
        self.assertEqual(ref_output, test_output)

    def test_subgraph_rewriter_multiple_pattern_match(self):
        # 定义一个复杂的神经网络模型类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义前向传播函数，接受 x, w1, w2 三个参数
            def forward(self, x, w1, w2):
                # 计算 w1 和 w2 连接后的总和
                m1 = torch.cat([w1, w2]).sum()
                # 计算 w1 和 w2 连接后的总和
                m2 = torch.cat([w1, w2]).sum()
                # 返回 x 加上 m1 的最大值和 m2 的最大值
                return x + torch.max(m1) + torch.max(m2)

        # 定义一个模式匹配函数，用于查找 torch.cat([w1, w2]).sum() 这样的模式
        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        # 定义一个替换函数，将匹配到的模式替换为 torch.stack([w1, w2])
        def replacement(w1, w2):
            return torch.stack([w1, w2])

        # 定义一个比较函数，将 w1 和 w2 连接后堆叠，并返回 x 加上两者堆叠后的最大值
        def comparison(x, w1, w2):
            m1 = torch.stack([w1, w2])
            m2 = torch.stack([w1, w2])
            return x + torch.max(m1) + torch.max(m2)

        # 对 M 类进行符号化跟踪，得到 traced 对象
        traced = symbolic_trace(M())
        # 对比较函数进行符号化跟踪，得到 comparison_fn 对象
        comparison_fn = symbolic_trace(comparison)

        # 生成形状为 (1, 3) 的随机张量 x, w1, w2
        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)

        # 使用 subgraph_rewriter.replace_pattern 函数替换模式
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪图进行检查
        traced.graph.lint()

        # 对比较函数应用于输入 x, w1, w2，得到参考输出 ref_outs
        ref_outs = comparison_fn(x, w1, w2)
        # 对跟踪对象 traced 应用于输入 x, w1, w2，得到测试输出 test_outs
        test_outs = traced.forward(x, w1, w2)
        # 使用断言检查 ref_outs 和 test_outs 是否相等
        self.assertEqual(ref_outs, test_outs)
    def test_subgraph_rewriter_graph_argument_order(self):
        # 定义一个简单的神经网络模块 M，包含一个矩阵乘法作为前向传播函数
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        # 定义一个模式函数 pattern，也包含一个矩阵乘法
        def pattern(x, y):
            return torch.mm(x, y)

        # 定义一个对比函数 comparison，同样包含一个矩阵乘法
        def comparison(x, y):
            return torch.mm(x, y)

        # 对 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比函数也进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建两个随机矩阵 x 和 y
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)

        # 使用 subgraph_rewriter 替换 traced 中的 pattern 函数为另一个 pattern 函数
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        # 对跟踪后的图进行 lint 检查
        traced.graph.lint()

        # 计算对比函数的输出作为参考输出
        ref_outs = comparison_fn(x, y)
        # 计算替换后的 traced 模块的输出
        test_outs = traced.forward(x, y)

        # 断言替换后的输出与对比函数的输出相等
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_correct_output_replacement(self):
        # 定义一个包含复杂操作的神经网络模块 M
        class M(torch.nn.Module):
            def forward(self, x, y):
                val = torch.neg(y) + torch.relu(x)
                return torch.add(val, val)

        # 定义一个模式函数 pattern，只包含 torch.relu 操作
        def pattern(x):
            return torch.relu(x)

        # 定义一个替换函数 replacement，只包含 torch.neg 操作
        def replacement(x):
            return torch.neg(x)

        # 定义一个对比函数 comparison，包含 torch.neg 和 torch.relu 操作
        def comparison(x, y):
            val = torch.neg(y) + torch.neg(x)
            return torch.add(val, val)

        # 对 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比函数也进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建两个随机矩阵 x 和 y
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # 使用 subgraph_rewriter 替换 traced 中的 pattern 函数为 replacement 函数
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的图进行 lint 检查
        traced.graph.lint()

        # 计算对比函数的输出作为参考输出
        ref_outs = comparison_fn(x, y)
        # 计算替换后的 traced 模块的输出
        test_outs = traced.forward(x, y)

        # 断言替换后的输出与对比函数的输出相等
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_traced_as_callable(self):
        # 定义一个简单的神经网络模块 M，包含 torch.neg 和 torch.relu 操作
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        # 定义一个模式函数 Pattern，也包含 torch.neg 和 torch.relu 操作
        class Pattern(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x) + torch.relu(x)

        # 定义一个替换函数 Replacement，包含 torch.sigmoid 操作
        class Replacement(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        # 定义一个对比函数 comparison，只包含 torch.sigmoid 操作
        def comparison(x):
            val = torch.sigmoid(x)
            return torch.add(val, val)

        # 对 M、Pattern、Replacement 和 comparison 进行符号跟踪
        traced = symbolic_trace(M())
        traced_pattern = symbolic_trace(Pattern())
        traced_replacement = symbolic_trace(Replacement())
        comparison_fn = symbolic_trace(comparison)

        # 创建一个随机矩阵 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter 替换 traced 中的 traced_pattern 函数为 traced_replacement 函数
        subgraph_rewriter.replace_pattern(traced, traced_pattern, traced_replacement)

        # 对跟踪后的图进行 lint 检查
        traced.graph.lint()

        # 计算对比函数的输出作为参考输出
        ref_outs = comparison_fn(x)
        # 计算替换后的 traced 模块的输出
        test_outs = traced.forward(x)

        # 断言替换后的输出与对比函数的输出相等
        self.assertEqual(ref_outs, test_outs)
    def test_subgraph_rewriter_pattern_is_entire_graph(self):
        # 定义一个简单的神经网络模块 M，重写 forward 方法
        class M(torch.nn.Module):
            def forward(self, x):
                # 对输入张量 x 取负值
                a = torch.neg(x)
                # 返回将两次取负值的结果相加的张量
                return torch.add(a, a)

        # 定义一个模式函数 pattern，用于匹配输入张量 x
        def pattern(x):
            # 对输入张量 x 取负值
            a = torch.neg(x)
            # 返回将两次取负值的结果相加的张量
            return torch.add(a, a)

        # 定义一个替换函数 replacement，用于替换模式函数匹配到的结构
        def replacement(x):
            # 对输入张量 x 进行 sigmoid 操作
            a = torch.sigmoid(x)
            # 返回两次 sigmoid 操作结果的拼接
            return torch.cat([a, a])

        # 对 M 模块进行符号跟踪
        traced = symbolic_trace(M())
        # 对替换函数进行符号跟踪
        comparison_fn = symbolic_trace(replacement)

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter.replace_pattern 函数替换符号跟踪后的模块 traced 中的模式
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的图形进行静态分析
        traced.graph.lint()

        # 计算参考输出和测试输出
        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)

        # 使用 self.assertEqual 进行断言，验证测试输出与参考输出是否一致
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(
        self,
    ):
        # 定义一个简单的神经网络模块 M，重写 forward 方法
        class M(torch.nn.Module):
            def forward(self, x):
                # 对输入张量 x 进行 relu 操作
                y = torch.relu(x)
                # 返回对 relu 结果取负值后减去原始结果的张量
                return torch.neg(y) - y

        # 定义一个模式函数 pattern，用于匹配输入张量 x 的 relu 操作
        def pattern(x):
            return torch.relu(x)

        # 定义一个替换函数 replacement，用于替换模式函数匹配到的结构
        def replacement(x):
            return torch.sigmoid(x)

        # 定义一个比较函数 comparison，用于验证替换函数的期望输出
        def comparison(x):
            y = torch.sigmoid(x)
            return torch.neg(y) - y

        # 对 M 模块进行符号跟踪
        traced = symbolic_trace(M())
        # 对比较函数进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter.replace_pattern 函数替换符号跟踪后的模块 traced 中的模式
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的图形进行静态分析
        traced.graph.lint()

        # 计算参考输出和测试输出
        ref_outs = comparison_fn(x)
        test_outs = traced.forward(x)

        # 使用 self.assertEqual 进行断言，验证测试输出与参考输出是否一致
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(
        self,
    ):
        # 定义一个复杂的神经网络模块 M，包含多个输入和多个计算步骤
        class M(torch.nn.Module):
            def forward(self, x, w1, w2, b1, b2):
                # 拼接张量 w1 和 w2
                m0 = torch.cat([w1, w2])
                m1 = torch.cat([w1, w2])
                # 拼接输入张量 x 和 b2
                m2 = torch.cat([x, b2])
                # 使用 b1、m1 和 m2 进行 addmm 运算
                t0 = torch.addmm(b1, m1, m2.t())
                # 对 w1 进行沿第一维度求和
                t1 = torch.sum(w1, 1)
                # 再次使用 b1、m1 和 m2 进行 addmm 运算
                t2 = torch.addmm(b1, m1, m2.t())
                # 返回两个求和结果
                return torch.sum(t1), torch.sum(t2)

        # 定义一个模式函数 pattern，用于匹配神经网络模块 M 中的一部分计算
        def pattern(x, w1, w2, b1, b2):
            # 拼接张量 w1 和 w2
            m1 = torch.cat([w1, w2])
            # 拼接输入张量 x 和 b2
            m2 = torch.cat([x, b2])
            # 使用 b1、m1 和 m2 进行 addmm 运算
            return torch.addmm(b1, m1, m2.t())

        # 定义一个替换函数 replacement，用于替换模式函数匹配到的结构
        def replacement(x, w1, w2, b1, b2):
            # 拼接输入张量 x、w1 和 w2
            return torch.cat([x, w1, w2])

        # 对 M 模块进行符号跟踪
        traced = symbolic_trace(M())

        # 使用 subgraph_rewriter.replace_pattern 函数替换符号跟踪后的模块 traced 中的模式
        res = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的图形进行静态分析
        traced.graph.lint()

        # 使用 self.assertEqual 进行断言，验证替换结果是否符合预期
        self.assertEqual(res, [])
    def test_subgraph_rewriter_replaces_referenced_submodules(self):
        # 定义一个测试用例类，测试替换引用子模块的行为

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 在输入张量上加一，然后通过子模块进行处理
                x = x + 1
                return self.submod(self.sigmoid(x))

        class Pattern(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 使用相同的结构，但不同的参数和子模块进行前向传播
                return self.submod(self.sigmoid(x))

        class Replacement(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tanh = torch.nn.Tanh()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 使用替换的子模块对输入进行处理
                return self.submod(self.tanh(x))

        class Comparison(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tanh = torch.nn.Tanh()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 对输入进行操作，用于比较原始模块和替换后模块的输出
                x = x + 1
                return self.submod(self.tanh(x))

        # 对模型M进行符号化追踪
        traced = symbolic_trace(M())
        # 创建一个比较模型的实例
        comparison = Comparison()

        # 生成一个3x4的随机张量
        x = torch.randn(3, 4)

        # 使用子图重写器替换traced中的Pattern()模式为Replacement()模式
        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        # 对traced图进行Lint检查
        traced.graph.lint()

        # 计算comparison模型在输入x上的输出
        ref_outs = comparison(x)
        # 计算traced模型在输入x上的输出
        test_outs = traced.forward(x)
        # 断言替换后的输出与参考输出一致
        self.assertEqual(ref_outs, test_outs)

        # 获取traced模型中名为"tanh"的子模块
        traced.get_submodule("tanh")
        # 预期捕获到AttributeError异常，且异常信息包含"has no attribute"
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            traced.get_submodule("sigmoid")

        # 获取traced模型中名为"submod"的子模块，并断言其类型为torch.nn.ReLU
        submod = traced.get_submodule("submod")
        self.assertEqual(type(submod), torch.nn.ReLU)

    def test_subgraph_rewriter_annotations_int(self):
        # 定义一个测试用例类，测试子图重写器在处理int类型注解时的行为

        class M1(torch.nn.Module):
            def forward(self, x):
                # 将输入x标注为整数类型，并返回x加y的结果
                y: int = x
                return torch.add(x, y)

        class M2(torch.nn.Module):
            def forward(self, x):
                # 使用annotate函数将输入x标注为整数类型，并返回x加y的结果
                y = annotate(x, int)
                return torch.add(x, y)

        # 创建AST重写器实例
        ast_rewriter = RewritingTracer()
        # 对模型M1进行追踪，获取其图结构
        graph = ast_rewriter.trace(M1())

        # 创建M2模型的实例
        module = M2()
        # 对M2模型进行符号化追踪，获取其图结构
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        
        # 遍历symbolic_traced图的节点和graph图的节点，对比占位符节点的类型是否为整数类型
        for n, m in zip(symbolic_traced.graph.nodes, graph.nodes):
            if n.op == "placeholder":
                assert n.type == int
                assert m.type == int
    # 定义测试函数，用于测试替换连续子模块的情况
    def test_subgraph_rewriter_replace_consecutive_submodules(self):
        # 定义简单的函数 f，其中有连续两次的 sigmoid 操作
        def f(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        # 定义模式函数，用于匹配连续的 sigmoid 操作
        def pattern(x):
            return torch.sigmoid(x)

        # 定义替换函数，将匹配到的 sigmoid 操作替换为 exp 操作
        def replacement(x):
            return torch.exp(x)

        # 定义比较函数，用于验证替换后的输出是否正确
        def comparison(x):
            x = torch.exp(x)
            x = torch.exp(x)
            return torch.exp(x)

        # 对函数 f 进行符号跟踪，以便后续的模式匹配和替换操作
        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        # 创建输入张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter.replace_pattern 方法替换 traced 中匹配 pattern 的部分为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪得到的图进行 lint（静态检查）
        traced.graph.lint()

        # 调用 comparison_fn 获取参考输出
        ref_outs = comparison_fn(x)
        # 调用 traced.forward 获取测试输出
        test_outs = traced.forward(x)

        # 使用 self.assertEqual 进行测试，验证替换后的输出是否与预期一致
        self.assertEqual(ref_outs, test_outs)

    # 定义测试函数，用于测试替换包含重叠匹配的情况
    def test_subgraph_rewriter_with_overlapping_matches(self):
        # 定义函数 f，其中有三次连续的 sigmoid 操作
        def f(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        # 定义模式函数，匹配两次连续的 sigmoid 操作
        def pattern(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return x

        # 定义替换函数，将匹配到的 sigmoid 操作替换为 neg 操作
        def replacement(x):
            return torch.neg(x)

        # 定义比较函数，用于验证替换后的输出是否正确
        def comparison(x):
            x = torch.neg(x)
            return torch.neg(x)

        # 对函数 f 进行符号跟踪
        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        # 创建输入张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter.replace_pattern 方法替换 traced 中匹配 pattern 的部分为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪得到的图进行 lint（静态检查）
        traced.graph.lint()

        # 调用 comparison_fn 获取参考输出
        ref_outs = comparison_fn(x)
        # 调用 traced.forward 获取测试输出
        test_outs = traced.forward(x)

        # 使用 self.assertEqual 进行测试，验证替换后的输出是否与预期一致
        self.assertEqual(ref_outs, test_outs)

    # 定义测试函数，用于测试替换带有多个输出的情况
    def test_subgraph_rewriter_replace_with_multiple_outputs(self):
        # 定义函数 f，其中包含 sigmoid 和 relu 操作，并返回它们的和
        def f(x):
            y = torch.sigmoid(x)
            z = torch.relu(x)
            return y + z

        # 定义模式函数，匹配 sigmoid 和 relu 操作并返回它们
        def pattern(a):
            b = torch.sigmoid(a)
            c = torch.relu(a)
            return b, c

        # 定义替换函数，将匹配到的 sigmoid 和 relu 操作分别替换为 exp 和 abs 操作
        def replacement(x):
            return torch.exp(x), torch.abs(x)

        # 定义比较函数，用于验证替换后的输出是否正确
        def comparison(x):
            y = torch.exp(x)
            z = torch.abs(x)
            return y + z

        # 对函数 f 进行符号跟踪
        traced = symbolic_trace(f)
        comparison_fn = symbolic_trace(comparison)

        # 创建输入张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter.replace_pattern 方法替换 traced 中匹配 pattern 的部分为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪得到的图进行 lint（静态检查）
        traced.graph.lint()

        # 调用 comparison_fn 获取参考输出
        ref_outs = comparison_fn(x)
        # 调用 traced.forward 获取测试输出
        test_outs = traced.forward(x)

        # 使用 self.assertEqual 进行测试，验证替换后的输出是否与预期一致
        self.assertEqual(ref_outs, test_outs)
    # 定义一个测试方法，用于测试子图重写器替换带有重复输出的情况
    def test_subgraph_rewriter_replace_with_duplicated_outputs(self):
        # 定义一个函数 f，计算输入 x1 和 x2 的差，然后分别计算 sigmoid 和 relu
        def f(x1, x2):
            x = x1 - x2
            y = torch.sigmoid(x)
            z = torch.relu(x)
            return y + z

        # 定义一个模式匹配函数 pattern，计算输入 a1 和 a2 的差，然后分别计算 sigmoid 和 relu，并返回这些值以及 a 的值
        def pattern(a1, a2):
            a = a1 - a2
            b = torch.sigmoid(a)
            c = torch.relu(a)
            return b, c, a

        # 定义一个替换函数 replacement，计算输入 x1 和 x2 的指数和绝对值，并返回这些值
        def replacement(x1, x2):
            y1 = torch.exp(x1)
            y2 = torch.abs(x2)
            return y2, y2, y1

        # 定义一个比较函数 comparison，计算输入 x1 和 x2 的绝对值之和，并返回这个值
        def comparison(x1, x2):
            y2 = torch.abs(x2)
            return y2 + y2

        # 对函数 f 进行符号化跟踪
        traced = symbolic_trace(f)
        # 对函数 comparison 进行符号化跟踪
        comparison_fn = symbolic_trace(comparison)

        # 生成随机张量作为输入 x1 和 x2
        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)

        # 使用子图重写器将 traced 中匹配 pattern 的部分替换为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的图进行静态检查
        traced.graph.lint()

        # 使用 comparison_fn 计算参考输出 ref_outs
        ref_outs = comparison_fn(x1, x2)
        # 使用 traced.forward 计算测试输出 test_outs
        test_outs = traced.forward(x1, x2)

        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)

    # 定义一个测试方法，用于测试子图重写器处理未使用参数的情况
    def test_subgraph_rewriter_with_unused_args(self):
        # 定义一个简单的神经网络模型 M，其 forward 方法接受三个参数并返回前两个参数的和
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y

        # 定义一个模式匹配函数 pattern，返回输入 x 和 y 的和
        def pattern(x, y):
            return x + y

        # 定义一个替换函数 replacement，返回输入 x 和 y 的差
        def replacement(x, y):
            return x - y

        # 定义一个比较函数 comparison，返回输入 x1 和 x2 的差
        def comparison(x1, x2, x3):
            return x1 - x2

        # 对模型 M 进行符号化跟踪
        traced = symbolic_trace(M())
        # 对比较函数 comparison 进行符号化跟踪
        comparison_fn = symbolic_trace(comparison)

        # 生成随机张量作为输入 x1、x2 和 x3
        x1 = torch.randn(3, 4)
        x2 = torch.randn(3, 4)
        x3 = torch.randn(3, 4)

        # 使用子图重写器将 traced 中匹配 pattern 的部分替换为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的图进行静态检查
        traced.graph.lint()

        # 获取图中所有操作为 placeholder 的节点
        placeholder_nodes = [n for n in traced.graph.nodes if n.op == "placeholder"]
        # 断言 placeholder 节点的数量为 3
        assert len(placeholder_nodes) == 3

        # 使用 comparison_fn 计算参考输出 ref_outs
        ref_outs = comparison_fn(x1, x2, x3)
        # 使用 traced.forward 计算测试输出 test_outs
        test_outs = traced.forward(x1, x2, x3)

        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)

    # 定义一个测试方法，用于测试子图重写器处理调用方法的情况
    def test_subgraph_rewriter_call_method(self):
        # 定义一个简单的神经网络模型 M，其 forward 方法接受一个参数 x，对 x 进行一系列操作并返回结果
        class M(torch.nn.Module):
            def forward(self, x):
                x = x.dequantize()
                x = x.sigmoid()
                x = x.to(torch.float16)
                return x

        # 定义一个模式匹配函数 pattern，接受一个参数 x，对 x 进行一系列操作并返回结果
        def pattern(x):
            x = x.dequantize()
            x = x.sigmoid()
            x = x.to(torch.float16)
            return x

        # 定义一个替换函数 replacement，接受一个参数 x，并直接返回 x
        def replacement(x):
            return x

        # 对模型 M 进行符号化跟踪
        traced = symbolic_trace(M())
        # 对替换函数 replacement 进行符号化跟踪
        comparison_fn = symbolic_trace(replacement)

        # 生成随机张量作为输入 x1
        x1 = torch.randn(3, 4)

        # 使用子图重写器将 traced 中匹配 pattern 的部分替换为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的图进行静态检查
        traced.graph.lint()

        # 使用 comparison_fn 计算参考输出 ref_outs
        ref_outs = comparison_fn(x1)
        # 使用 traced.forward 计算测试输出 test_outs
        test_outs = traced.forward(x1)

        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)
    def test_subgraph_rewriter_nodes_with_kwargs(self):
        # 定义一个测试方法，用于测试带有关键字参数的子图重写功能

        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的类 M
            def __init__(self) -> None:
                super().__init__()
                # 初始化方法
                self.w0 = torch.nn.Parameter(torch.empty([128, 128]))
                # 定义一个大小为 [128, 128] 的参数 w0
                self.b0 = torch.nn.Parameter(torch.empty([128]))
                # 定义一个大小为 [128] 的参数 b0

            def forward(self, in0):
                # 前向传播方法，接受输入参数 in0
                lin_res = torch.nn.functional.linear(in0, self.w0, bias=self.b0)
                # 执行线性操作，使用参数 w0 和 b0
                mul_res = in0 * lin_res
                # 执行元素级乘法操作
                sum_res = mul_res + in0
                # 执行元素级加法操作
                return sum_res
                # 返回结果 sum_res

        def pattern(a, b, bias):
            # 定义一个模式匹配函数 pattern，接受参数 a, b, bias
            lin_res = torch.nn.functional.linear(a, b, bias=bias)
            # 执行线性操作，使用参数 b 和 bias
            mul_res = a * lin_res
            # 执行元素级乘法操作
            return lin_res, mul_res
            # 返回 lin_res 和 mul_res

        def replacement(a, b, bias):
            # 定义一个替换函数 replacement，接受参数 a, b, bias
            lin_res, mul_res = wrapped_gemm_bias_mul(a, b, bias)
            # 调用 wrapped_gemm_bias_mul 函数，获取结果 lin_res 和 mul_res
            return lin_res, mul_res
            # 返回 lin_res 和 mul_res

        traced = symbolic_trace(M())
        # 对类 M 进行符号跟踪
        matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)
        # 在跟踪的符号图中替换模式 pattern 到 replacement，并返回匹配结果

        self.assertEqual(len(matches), 1)
        # 断言匹配结果的数量为 1

        found_repalcement_node = False
        # 初始化一个标志位 found_repalcement_node 为 False
        for node in traced.graph.nodes:
            # 遍历跟踪的符号图的节点
            if node.target == wrapped_gemm_bias_mul:
                # 如果节点的目标是 wrapped_gemm_bias_mul 函数
                found_repalcement_node = True
                # 将 found_repalcement_node 置为 True
                break

        self.assertTrue(found_repalcement_node)
        # 断言 found_repalcement_node 为 True
    def test_replace_pattern_with_filters(self):
        class M(torch.nn.Module):
            def forward(self, x, scale, zero_point):
                # 将输入张量去量化
                x = x.dequantize()
                # 将张量与标量 2 相加
                x = torch.add(x, 2)
                # 应用 ReLU 激活函数
                x = x.relu()
                # 将张量按照给定的量化参数重新量化
                x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

                # 创建新的张量 y，与 x 相加 1
                y = x + 1
                # 不匹配情况，y 不是标量
                x = x.dequantize()
                x = torch.add(x, y)
                x = x.relu()
                x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

                return x

        def BinaryOpScalarReLUPattern(x, num, scale, zero_point):
            # 将输入张量去量化
            x = x.dequantize()
            # 将张量与数值 num 相乘
            x = torch.add(x, num)
            # 应用 ReLU 激活函数
            x = x.relu()
            # 将张量按照给定的量化参数重新量化
            x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
            return x

        def BinaryOpScalarReLUReplacement(x, num, scale, zero_point):
            # 将张量与数值 num 相乘
            x = torch.mul(x, num)
            return x

        def second_input_is_scalar(match, original_graph, pattern_graph):
            """检查匹配到模式图第二个输入的节点是否为标量"""
            input_idx = 0
            for node in pattern_graph.nodes:
                if node.op == "placeholder":
                    if input_idx == 1:
                        num_node = node
                    input_idx += 1
            # 判断匹配到的节点是否为整数或浮点数
            if not isinstance(match.nodes_map[num_node], (int, float)):
                return False
            return True

        def check_replacement_nodes(self, traced, matches):
            # 找到图中目标为 torch.mul 的替换节点
            replacement_nodes_in_graph = [
                node for node in traced.graph.nodes if node.target == torch.mul
            ]
            # 找到匹配结果中的所有替换节点
            replacement_nodes_in_res = [r for m in matches for r in m.replacements]
            # 检查两者数量是否相等
            self.assertEqual(
                len(replacement_nodes_in_graph), len(replacement_nodes_in_res)
            )
            # 检查两者内容是否一致
            self.assertEqual(replacement_nodes_in_graph, replacement_nodes_in_res)
            return len(replacement_nodes_in_graph)

        # 没有过滤条件的匹配，应该找到 2 个匹配结果
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced, BinaryOpScalarReLUPattern, BinaryOpScalarReLUReplacement, None
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 2)

        # 使用过滤条件的匹配，应该找到 1 个匹配结果
        traced = symbolic_trace(M())
        matches = subgraph_rewriter.replace_pattern_with_filters(
            traced,
            BinaryOpScalarReLUPattern,
            BinaryOpScalarReLUReplacement,
            [second_input_is_scalar],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(check_replacement_nodes(self, traced, matches), 1)
    def test_matching_pattern_with_list_type_arg(self):
        # 定义一个内嵌的 PyTorch 模块 M
        class M(torch.nn.Module):
            # 模块的前向传播方法，使用 torch.ops 调用底层函数 _reshape_alias_copy.default
            def forward(self, x):
                return torch.ops.aten._reshape_alias_copy.default(x, [1, 2], [3, 4])

        # 定义一个模式匹配函数 pattern，使用 torch.ops 调用底层函数 _reshape_alias_copy.default
        def pattern(x, arg0, arg1):
            return torch.ops.aten._reshape_alias_copy.default(x, arg0, arg1)

        # 定义一个替换函数 replacement，使用 torch.ops 调用底层函数 _reshape_alias_copy.default
        def replacement(x, arg0, arg1):
            return torch.ops.aten._reshape_alias_copy.default(x, arg1, arg0)

        # 对模块 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 使用替换函数替换匹配到的子图
        matches = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 断言匹配到的替换次数为 1
        self.assertEqual(len(matches), 1)

        # 断言跟踪后的代码符合预期的内联形式
        self.assertExpectedInline(
            traced.code.strip(),
            """\
    # 对输入张量 x 进行操作，使用 torch.ops.aten._reshape_alias_copy.default 函数重塑并复制数据，返回结果给 _reshape_alias_copy_default_1
    _reshape_alias_copy_default_1 = torch.ops.aten._reshape_alias_copy.default(x, [3, 4], [1, 2]);  x = None
    # 返回重塑并复制后的结果 _reshape_alias_copy_default_1
    return _reshape_alias_copy_default_1
```