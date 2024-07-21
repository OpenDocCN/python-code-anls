# `.\pytorch\test\quantization\fx\test_subgraph_rewriter.py`

```
# 导入标准库模块
import os
import sys

# 导入 PyTorch 相关模块
import torch
from torch.fx import symbolic_trace, subgraph_rewriter
from torch.fx.annotate import annotate
# 导入实验性重写追踪器
from torch.fx.experimental.rewriter import RewritingTracer

# 获取当前脚本所在目录的上级目录，添加到系统路径中，使得测试文件中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
# 导入测试框架的基类
from torch.testing._internal.jit_utils import JitTestCase

# 如果该脚本被直接运行，抛出异常，提示不应直接运行该文件，而应使用指定的方式
if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_fx.py TESTNAME\n\n"
                       "instead.")

# 测试类，继承自 JitTestCase
class TestSubgraphRewriter(JitTestCase):

    # 测试函数：测试子图重写器是否保留逻辑
    def test_subgraph_rewriter_preserves_logic(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.neg 和 torch.relu 函数进行操作
                val = torch.neg(x) + torch.relu(x)
                return torch.add(val, val)

        # 定义一个模式函数，与 forward 方法中的操作相同
        def pattern(x):
            return torch.neg(x) + torch.relu(x)

        # 定义一个对照函数，用于比较输出结果是否一致
        def comparison(x):
            val = torch.neg(x) + torch.relu(x)
            return torch.add(val, val)

        # 对模型进行符号跟踪
        traced = symbolic_trace(M())
        # 对比辅助函数进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 生成一个随机输入张量
        x = torch.rand(1, 3)

        # 使用子图重写器，将模式函数替换为相同的模式函数，预期不改变底层逻辑
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        # 对跟踪后的图进行静态分析
        traced.graph.lint()

        # 获取对照函数的输出结果
        ref_output = comparison_fn(x)
        # 获取重写后模型的输出结果
        test_output = traced.forward(x)
        # 断言输出结果应该相同
        self.assertEqual(ref_output, test_output)

    # 测试函数：测试子图重写器使用单行模式
    def test_subgraph_rewriter_with_oneliner_pattern(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.neg 函数进行操作
                val = torch.neg(x)
                return torch.add(val, val)

        # 定义一个模式函数，对应于 forward 方法中的 torch.neg 操作
        def pattern(x):
            return torch.neg(x)

        # 定义一个替换函数，用于替换模式函数中的 torch.neg 操作为 torch.relu 操作
        def replacement(x):
            return torch.relu(x)

        # 定义一个对照函数，用于比较输出结果是否一致
        def comparison(x):
            val = torch.relu(x)
            return torch.add(val, val)

        # 对模型进行符号跟踪
        traced = symbolic_trace(M())
        # 对比辅助函数进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 生成一个随机输入张量
        x = torch.rand(1, 3)

        # 使用子图重写器，将模式函数中的 torch.neg 操作替换为替换函数中的 torch.relu 操作
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的图进行静态分析
        traced.graph.lint()

        # 获取对照函数的输出结果
        ref_output = comparison_fn(x)
        # 获取重写后模型的输出结果
        test_output = traced.forward(x)
        # 断言输出结果应该相同
        self.assertEqual(ref_output, test_output)
    # 定义一个测试用例，用于测试单一模式匹配的子图重写功能
    def test_subgraph_rewriter_single_pattern_match(self):
        # 定义一个简单的神经网络模型类 M
        class M(torch.nn.Module):
            # 模型前向传播函数
            def forward(self, x):
                # 使用 torch.neg 和 torch.relu 函数对输入 x 进行操作
                val = torch.neg(x) + torch.relu(x)
                # 返回对操作结果进行加法运算后的值
                return torch.add(val, val)

        # 定义一个模式函数，用于提取模式 torch.neg(x) + torch.relu(x)
        def pattern(x):
            return torch.neg(x) + torch.relu(x)

        # 定义一个替换函数，用于替换模式 torch.neg(x) + torch.relu(x) 中的部分
        def replacement(x):
            return torch.relu(x)

        # 定义一个比较函数，用于比较替换后的输出与预期输出是否一致
        def comparison(x):
            val = torch.relu(x)
            return torch.add(val, val)

        # 对模型 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比较函数进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建一个形状为 (1, 3) 的随机张量 x
        x = torch.rand(1, 3)

        # 使用子图重写器，替换模型 traced 中的指定模式
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的模型图进行 lint 检查
        traced.graph.lint()

        # 获取比较函数在输入 x 上的输出作为参考输出
        ref_output = comparison_fn(x)
        # 获取替换后模型 traced 在输入 x 上的输出作为测试输出
        test_output = traced.forward(x)
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_output, test_output)

    # 定义一个测试用例，用于测试多模式匹配的子图重写功能
    def test_subgraph_rewriter_multiple_pattern_match(self):
        # 定义一个包含多个输入的神经网络模型类 M
        class M(torch.nn.Module):
            # 模型前向传播函数，接受输入 x, w1, w2
            def forward(self, x, w1, w2):
                # 计算 w1 和 w2 的连接并求和
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                # 返回 x 加上 m1 和 m2 中的最大值
                return x + torch.max(m1) + torch.max(m2)

        # 定义一个模式函数，用于提取模式 torch.cat([w1, w2]).sum()
        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        # 定义一个替换函数，用于替换模式 torch.cat([w1, w2]).sum()
        def replacement(w1, w2):
            return torch.stack([w1, w2])

        # 定义一个比较函数，用于比较替换后的输出与预期输出是否一致
        def comparison(x, w1, w2):
            m1 = torch.stack([w1, w2])
            m2 = torch.stack([w1, w2])
            return x + torch.max(m1) + torch.max(m2)

        # 对模型 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比较函数进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建形状为 (1, 3) 的随机张量 x, w1, w2
        x = torch.rand(1, 3)
        w1 = torch.rand(1, 3)
        w2 = torch.rand(1, 3)

        # 使用子图重写器，替换模型 traced 中的指定模式
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的模型图进行 lint 检查
        traced.graph.lint()

        # 获取比较函数在输入 x, w1, w2 上的输出作为参考输出
        ref_outs = comparison_fn(x, w1, w2)
        # 获取替换后模型 traced 在输入 x, w1, w2 上的输出作为测试输出
        test_outs = traced.forward(x, w1, w2)
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)

    # 定义一个测试用例，用于测试子图重写时的图参数顺序问题
    def test_subgraph_rewriter_graph_argument_order(self):
        # 定义一个简单的神经网络模型类 M，接受输入 x, y，返回它们的矩阵乘法
        class M(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        # 定义一个模式函数，用于提取模式 torch.mm(x, y)
        def pattern(x, y):
            return torch.mm(x, y)

        # 定义一个比较函数，用于比较模式 torch.mm(x, y) 的输出与预期输出是否一致
        def comparison(x, y):
            return torch.mm(x, y)

        # 对模型 M 进行符号跟踪
        traced = symbolic_trace(M())
        # 对比较函数进行符号跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建形状分别为 (3, 4) 和 (4, 5) 的随机张量 x, y
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)

        # 使用子图重写器，替换模型 traced 中的指定模式，注意这里 pattern 和 replacement 是相同的函数
        subgraph_rewriter.replace_pattern(traced, pattern, pattern)

        # 对替换后的模型图进行 lint 检查
        traced.graph.lint()

        # 获取比较函数在输入 x, y 上的输出作为参考输出
        ref_outs = comparison_fn(x, y)
        # 获取替换后模型 traced 在输入 x, y 上的输出作为测试输出
        test_outs = traced.forward(x, y)
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)
    # 定义一个测试方法，用于验证子图重写器在正确替换输出时的行为
    def test_subgraph_rewriter_correct_output_replacement(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 计算 y 的负值加上 x 的 ReLU（整流线性单元）函数的输出
                val = torch.neg(y) + torch.relu(x)
                # 返回 val 的加法运算结果
                return torch.add(val, val)

        # 定义一个模式匹配函数，用于匹配模型中的 ReLU 函数
        def pattern(x):
            return torch.relu(x)

        # 定义一个替换函数，用于将匹配到的 ReLU 函数替换为负值函数
        def replacement(x):
            return torch.neg(x)

        # 定义一个比较函数，用于比较替换后的模型输出与参考输出
        def comparison(x, y):
            # 计算 y 的负值加上 x 的负值的结果
            val = torch.neg(y) + torch.neg(x)
            # 返回 val 的加法运算结果
            return torch.add(val, val)

        # 对模型 M 进行符号跟踪，以获取其计算图
        traced = symbolic_trace(M())
        # 对比较函数进行符号跟踪，以获取其计算图
        comparison_fn = symbolic_trace(comparison)

        # 生成随机输入数据 x 和 y
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # 使用子图重写器，将模型中的 ReLU 函数替换为负值函数
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的模型计算图进行静态分析
        traced.graph.lint()

        # 获取参考输出
        ref_outs = comparison_fn(x, y)
        # 获取测试输出
        test_outs = traced.forward(x, y)
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)

    # 定义一个测试方法，用于验证子图重写器作为可调用对象时的行为
    def test_subgraph_rewriter_traced_as_callable(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 计算 x 的负值加上 x 的 ReLU 函数的输出
                val = torch.neg(x) + torch.relu(x)
                # 返回 val 的加法运算结果
                return torch.add(val, val)

        # 定义一个模式匹配模型，用于匹配模型 M 中的负值加 ReLU 函数
        class Pattern(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                return torch.neg(x) + torch.relu(x)

        # 定义一个替换模型，用于替换匹配到的模式
        class Replacement(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                return torch.sigmoid(x)

        # 定义一个比较函数，用于比较替换后的模型输出与参考输出
        def comparison(x):
            # 计算 x 的 sigmoid 函数的结果
            val = torch.sigmoid(x)
            # 返回 val 的加法运算结果
            return torch.add(val, val)

        # 对模型 M 进行符号跟踪，以获取其计算图
        traced = symbolic_trace(M())
        # 对模式匹配模型进行符号跟踪，以获取其计算图
        traced_pattern = symbolic_trace(Pattern())
        # 对替换模型进行符号跟踪，以获取其计算图
        traced_replacement = symbolic_trace(Replacement())
        # 对比较函数进行符号跟踪，以获取其计算图
        comparison_fn = symbolic_trace(comparison)

        # 生成随机输入数据 x
        x = torch.randn(3, 4)

        # 使用子图重写器，将模型中匹配到的模式替换为替换模型
        subgraph_rewriter.replace_pattern(traced, traced_pattern, traced_replacement)

        # 对跟踪后的模型计算图进行静态分析
        traced.graph.lint()

        # 获取参考输出
        ref_outs = comparison_fn(x)
        # 获取测试输出
        test_outs = traced.forward(x)
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)

    # 定义一个测试方法，用于验证当模式与整个图匹配时的子图重写器行为
    def test_subgraph_rewriter_pattern_is_entire_graph(self):
        # 定义一个简单的神经网络模型
        class M(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 计算 x 的负值，并将结果加起来
                a = torch.neg(x)
                # 返回 a 的加法运算结果
                return torch.add(a, a)

        # 定义一个模式匹配函数，用于匹配整个模型中的负值加法操作
        def pattern(x):
            a = torch.neg(x)
            return torch.add(a, a)

        # 定义一个替换函数，用于替换匹配到的模式
        def replacement(x):
            a = torch.sigmoid(x)
            return torch.cat([a, a])

        # 对模型 M 进行符号跟踪，以获取其计算图
        traced = symbolic_trace(M())
        # 对比较函数进行符号跟踪，以获取其计算图
        comparison_fn = symbolic_trace(replacement)

        # 生成随机输入数据 x
        x = torch.randn(3, 4)

        # 使用子图重写器，将整个模型中的匹配到的模式替换为替换函数
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对跟踪后的模型计算图进行静态分析
        traced.graph.lint()

        # 获取参考输出
        ref_outs = comparison_fn(x)
        # 获取测试输出
        test_outs = traced.forward(x)
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)
    def test_subgraph_rewriter_pattern_output_pattern_node_can_have_users_that_are_not_matched(self):
        # 定义一个简单的神经网络模型类
        class M(torch.nn.Module):
            # 定义模型的前向传播函数
            def forward(self, x):
                # 使用ReLU激活函数处理输入x
                y = torch.relu(x)
                # 对ReLU输出y进行取反并减去自身
                return torch.neg(y) - y

        # 定义模式函数pattern，用于匹配ReLU操作
        def pattern(x):
            return torch.relu(x)

        # 定义替换函数replacement，将匹配到的ReLU操作替换为Sigmoid操作
        def replacement(x):
            return torch.sigmoid(x)

        # 定义比较函数comparison，用于对比替换前后的模型输出
        def comparison(x):
            y = torch.sigmoid(x)
            return torch.neg(y) - y

        # 对模型M进行符号化跟踪
        traced = symbolic_trace(M())
        # 对比较函数comparison进行符号化跟踪
        comparison_fn = symbolic_trace(comparison)

        # 创建一个3x4大小的随机张量作为输入
        x = torch.randn(3, 4)

        # 使用subgraph_rewriter替换模型traced中匹配到的模式为替换函数replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的模型图进行Lint检查
        traced.graph.lint()

        # 计算比较函数的输出作为参考输出
        ref_outs = comparison_fn(x)
        # 计算替换后模型的输出
        test_outs = traced.forward(x)
        # 断言替换后模型的输出与参考输出一致
        self.assertEqual(ref_outs, test_outs)

    def test_subgraph_rewriter_internal_pattern_nodes_cannot_have_users_that_are_not_matched(self):
        # 定义一个复杂的神经网络模型类
        class M(torch.nn.Module):
            # 定义模型的前向传播函数，包含多个操作
            def forward(self, x, w1, w2, b1, b2):
                m0 = torch.cat([w1, w2])
                m1 = torch.cat([w1, w2])
                m2 = torch.cat([x, b2])
                t0 = torch.addmm(b1, m1, m2.t())
                t1 = torch.sum(w1, 1)
                t2 = torch.addmm(b1, m1, m2.t())
                return torch.sum(t1), torch.sum(t2)

        # 定义模式函数pattern，用于匹配模型中的某些操作序列
        def pattern(x, w1, w2, b1, b2):
            m1 = torch.cat([w1, w2])
            m2 = torch.cat([x, b2])
            return torch.addmm(b1, m1, m2.t())

        # 定义替换函数replacement，替换匹配到的模式操作序列
        def replacement(x, w1, w2, b1, b2):
            return torch.cat([x, w1, w2])

        # 对模型M进行符号化跟踪
        traced = symbolic_trace(M())

        # 使用subgraph_rewriter替换模型traced中匹配到的模式为替换函数replacement
        res = subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 对替换后的模型图进行Lint检查
        traced.graph.lint()

        # 断言替换结果为空列表，因为没有匹配成功的模式
        self.assertEqual(res, [])
    def test_subgraph_rewriter_replaces_referenced_submodules(self):
        # 定义一个包含多个子模块的神经网络模型 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模型 M 的子模块：sigmoid 和 submod
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 模型前向传播函数，对输入 x 执行操作并返回结果
                x = x + 1
                return self.submod(self.sigmoid(x))

        # 定义一个模式模型 Pattern，结构与模型 M 类似
        class Pattern(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 模式模型的前向传播函数，对输入 x 执行操作并返回结果
                return self.submod(self.sigmoid(x))

        # 定义一个替换模型 Replacement，结构与模型 M 类似
        class Replacement(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.id = torch.nn.Identity()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 替换模型的前向传播函数，对输入 x 执行操作并返回结果
                return self.submod(self.id(x))

        # 定义一个对比模型 Comparison，结构与模型 M 类似
        class Comparison(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.id = torch.nn.Identity()
                self.submod = torch.nn.ReLU()

            def forward(self, x):
                # 对比模型的前向传播函数，对输入 x 执行操作并返回结果
                x = x + 1
                return self.submod(self.id(x))

        # 对模型 M 进行符号化跟踪
        traced = symbolic_trace(M())
        # 创建一个对比模型实例
        comparison = Comparison()

        # 创建输入张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter 替换 traced 中的 Pattern 模型为 Replacement 模型
        subgraph_rewriter.replace_pattern(traced, Pattern(), Replacement())

        # 对跟踪后的图进行 lint 操作，检查图的合法性
        traced.graph.lint()

        # 计算对比模型的输出
        ref_outs = comparison(x)
        # 计算替换后的 traced 模型的输出
        test_outs = traced.forward(x)
        # 断言替换后的输出与对比模型的输出相等
        self.assertEqual(ref_outs, test_outs)

        # 获取 traced 模型中的子模块 "id"，应成功获取
        traced.get_submodule("id")
        # 断言尝试获取 traced 模型中不存在的子模块 "sigmoid" 时抛出 AttributeError 异常
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            traced.get_submodule("sigmoid")

        # 获取 traced 模型中的子模块 "submod"，并断言其类型为 torch.nn.ReLU
        submod = traced.get_submodule("submod")
        self.assertEqual(type(submod), torch.nn.ReLU)

    def test_subgraph_rewriter_annotations_int(self):
        # 定义一个简单的模型 M1，其 forward 函数中包含类型注解
        class M1(torch.nn.Module):
            def forward(self, x):
                y: int = x
                return torch.add(x, y)

        # 定义一个模型 M2，其 forward 函数中通过 annotate 函数添加类型注解
        class M2(torch.nn.Module):
            def forward(self, x):
                y = annotate(x, int)
                return torch.add(x, y)

        # 创建一个 AST 重写器
        ast_rewriter = RewritingTracer()
        # 对模型 M1 进行 AST 跟踪，获取其图结构
        graph = ast_rewriter.trace(M1())

        # 创建模型 M2 的实例
        module = M2()
        # 对模型 M2 进行符号化跟踪，获取其图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        
        # 逐个比较符号化跟踪得到的节点和 AST 跟踪得到的节点，验证类型注解是否正确
        for n, m in zip(symbolic_traced.graph.nodes, graph.nodes):
            if n.op == 'placeholder':
                assert n.type == int
                assert m.type == int
    # 定义测试函数，用于测试替换连续子模块的功能
    def test_subgraph_writer_replace_consecutive_submodules(self):

        # 定义函数 f，对输入 x 应用两次 sigmoid 函数，并返回结果
        def f(x):
            x = torch.sigmoid(x)
            x = torch.sigmoid(x)
            return torch.sigmoid(x)

        # 定义模式函数 pattern，对输入 x 应用 sigmoid 函数并返回结果
        def pattern(x):
            return torch.sigmoid(x)

        # 定义替换函数 replacement，对输入 x 应用指数函数并返回结果
        def replacement(x):
            return torch.exp(x)

        # 定义对比函数 comparison，对输入 x 应用两次指数函数，并返回结果
        def comparison(x):
            x = torch.exp(x)
            x = torch.exp(x)
            return torch.exp(x)

        # 对函数 f 进行符号化跟踪
        traced = symbolic_trace(f)
        
        # 对比函数 comparison 进行符号化跟踪
        comparison_fn = symbolic_trace(comparison)

        # 生成一个大小为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)

        # 使用 subgraph_rewriter 对 traced 中匹配 pattern 的子图进行替换为 replacement
        subgraph_rewriter.replace_pattern(traced, pattern, replacement)

        # 检查 traced 的计算图是否符合 lint 规范
        traced.graph.lint()

        # 使用 comparison_fn 计算参考输出 ref_outs
        ref_outs = comparison_fn(x)
        
        # 使用 traced 计算测试输出 test_outs
        test_outs = traced.forward(x)
        
        # 断言测试输出与参考输出相等
        self.assertEqual(ref_outs, test_outs)
```