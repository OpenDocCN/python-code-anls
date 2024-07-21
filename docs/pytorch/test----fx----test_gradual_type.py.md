# `.\pytorch\test\fx\test_gradual_type.py`

```
# Owner(s): ["module: fx"]

# 引入单元测试模块
import unittest

# 引入 sympy 库
import sympy

# 引入 PyTorch 库及相关模块
import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.annotate import annotate
from torch.fx.experimental.graph_gradual_typechecker import (
    broadcast_types,
    GraphTypeChecker,
    Refine,
)
from torch.fx.experimental.refinement_types import Equality
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.experimental.unify_refinements import infer_symbolic_types
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.tensor_type import Dyn, is_consistent, is_more_precise, TensorType
from torch.testing._internal.common_utils import TestCase

# 尝试引入 torchvision 中的 resnet50 模型，设置是否存在 torchvision 的标志
try:
    from torchvision.models import resnet50

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 如果没有安装 torchvision，则跳过相关测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """创建一个3x3的带有填充的卷积层"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class AnnotationsTest(TestCase):
    def test_annotations(self):
        """
        测试前向函数中的类型注解。
        注解应出现在生成图中的节点 n.graph 中，
        其中 n 是结果图中对应的节点。
        """

        class M(torch.nn.Module):
            def forward(
                self, x: TensorType((1, 2, 3, Dyn)), y: Dyn, z: TensorType[Dyn, 3, Dyn]
            ):
                return torch.add(x, y) + z

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

        expected_ph_types = [TensorType((1, 2, 3, Dyn)), Dyn, TensorType((Dyn, 3, Dyn))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                assert n.type == next(expected_iter)

    def test_annotate(self):
        """
        测试 annotate 函数的功能。
        """

        class M(torch.nn.Module):
            def forward(self, x):
                y = annotate(x, TensorType((1, 2, 3, Dyn)))
                return torch.add(x, y)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                assert n.type == TensorType((1, 2, 3, Dyn))

    def test_consistency(self):
        """
        测试一致性关系。
        """
        # 断言一致性函数的预期行为
        self.assertTrue(is_consistent(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_consistent(int, Dyn))
        self.assertTrue(is_consistent(int, int))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5))))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), int))
    # 定义一个测试函数，用于测试精度相关的函数
    def test_precision(self):
        """
        Test the consistency relation.
        """
        # 断言：TensorType((1, 2, 3)) 比 TensorType((1, Dyn, 3)) 更精确
        self.assertTrue(is_more_precise(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        # 断言：int 比 Dyn 更精确
        self.assertTrue(is_more_precise(int, Dyn))
        # 断言：int 和 int 相同精度
        self.assertTrue(is_more_precise(int, int))
        # 断言：TensorType((1, 2, 3)) 不比 TensorType((1, 2, 3, 5)) 更精确
        self.assertFalse(
            is_more_precise(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5)))
        )
        # 断言：TensorType((1, 2, 3)) 和 int 不具有相同的精度
        self.assertFalse(is_more_precise(TensorType((1, 2, 3)), int))

    # 定义一个测试函数，用于测试广播类型相关的函数
    def test_broadcasting1(self):
        t1 = TensorType((1, 2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))
        t3 = TensorType(())
        t4 = TensorType((4, 1))
        t5 = TensorType((4, 4, 4))
        # todo switch all code to use list instead of tuple
        # 断言：广播 t1 和 t2 的结果是两个相同的 TensorType
        assert broadcast_types(t1, t2) == (
            TensorType((1, 2, 3, 4)),
            TensorType((1, 2, 3, 4)),
        )
        # 断言：广播 t3 和 t4 的结果是 t4 的两个拷贝
        assert broadcast_types(t3, t4) == (t4, t4)
        # 断言：广播 t5 和 t6 的结果是 t5 的两个拷贝
        assert broadcast_types(t5, t6) == (t5, t5)

    # 定义一个测试函数，用于测试广播类型相关的函数
    def test_broadcasting2(self):
        t1 = TensorType((2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))

        # 断言：广播 t1 和 t2 的结果是两个相同的 TensorType
        assert broadcast_types(t1, t2) == (
            TensorType((1, 2, 3, 4)),
            TensorType((1, 2, 3, 4)),
        )

    # 定义一个测试函数，用于测试广播类型相关的函数
    def test_broadcasting3(self):
        t1 = TensorType((1, 2, 3, Dyn))
        t2 = TensorType((2, 3, 4))
        # 断言：广播 t1 和 t2 的结果分别是 t1 和 t2
        assert broadcast_types(t1, t2) == (
            TensorType((1, 2, 3, Dyn)),
            TensorType((1, 2, 3, 4)),
        )
class TypeCheckerTest(TestCase):
    def test_type_check_add_with_broadcast(self):
        # 定义一个测试用的神经网络模型类
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数，并指定它们的类型注解
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                # 返回两个张量相加的结果
                return torch.add(x, y)

        # 创建模型实例
        module = M()
        # 对模型进行符号化跟踪，得到图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建图类型检查器实例，传入空字典和符号化跟踪后的图模块
        tc = GraphTypeChecker({}, symbolic_traced)
        # 执行类型检查
        tc.type_check()
        # 预期的占位符类型列表
        expected_ph_types = [
            TensorType((1, 2, 3, Dyn)),
            TensorType((2, 3, 4)),
            TensorType((1, 2, 3, Dyn)),
            TensorType((1, 2, 3, Dyn)),
        ]
        # 创建预期类型列表的迭代器
        expected_iter = iter(expected_ph_types)

        # 遍历图模块中的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点的操作是调用函数
            if n.op == "call_function":
                # 断言节点的元数据中包含广播信息
                assert n.meta["broadcast"]
            # 断言节点的类型与预期的类型相符
            assert n.type == next(expected_iter)

    def test_type_check_add_with_scalar(self):
        # 定义一个测试用的神经网络模型类
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，接受一个整数和一个张量参数，并指定它们的类型注解
            def forward(self, x: int, y: TensorType((2, 3, 4))):
                # 返回整数与张量相加的结果
                return torch.add(x, y)

        # 创建模型实例
        module = M()
        # 对模型进行符号化跟踪，得到图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建图类型检查器实例，传入空字典和符号化跟踪后的图模块
        tc = GraphTypeChecker({}, symbolic_traced)
        # 执行类型检查
        tc.type_check()
        # 预期的占位符类型列表
        expected_ph_types = [
            int,
            TensorType((2, 3, 4)),
            TensorType((2, 3, 4)),
            TensorType((2, 3, 4)),
        ]
        # 创建预期类型列表的迭代器
        expected_iter = iter(expected_ph_types)

        # 遍历图模块中的节点
        for n in symbolic_traced.graph.nodes:
            # 断言节点的类型与预期的类型相符
            assert n.type == next(expected_iter)

    def test_type_check_add_false(self):
        # 定义一个测试用的神经网络模型类
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数，并指定它们的类型注解
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((1, 2, 3))):
                # 返回两个张量相加的结果
                return torch.add(x, y)

        # 创建模型实例
        module = M()
        # 对模型进行符号化跟踪，得到图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建图类型检查器实例，传入空字典和符号化跟踪后的图模块
        tc = GraphTypeChecker({}, symbolic_traced)
        # 执行类型检查，预期会引发类型错误异常
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_add_true(self):
        # 定义一个测试用的神经网络模型类
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，接受两个参数，并指定它们的类型注解
            def forward(self, x: TensorType((1, 2, Dyn)), y: TensorType((1, 2, 3))):
                # 返回两个张量相加的结果
                return torch.add(x, y)

        # 创建模型实例
        module = M()
        # 对模型进行符号化跟踪，得到图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建图类型检查器实例，传入空字典和符号化跟踪后的图模块
        tc = GraphTypeChecker({}, symbolic_traced)
        # 执行类型检查，并断言类型检查通过
        self.assertTrue(tc.type_check())

        # 预期的占位符类型列表
        expected_ph_types = [TensorType((1, 2, Dyn)), TensorType((1, 2, 3))]
        # 创建预期类型列表的迭代器
        expected_iter = iter(expected_ph_types)

        # 遍历图模块中的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点的操作是占位符
            if n.op == "placeholder":
                # 断言节点的类型与预期的类型相符
                assert n.type == next(expected_iter)
            # 如果节点的操作是输出
            if n.op == "output":
                # 断言节点的类型与预期的类型相符
                assert n.type == TensorType((1, 2, Dyn))
    def test_type_check_reshape_true(self):
        # 定义一个继承自torch.nn.Module的类M，用于测试类型检查和reshape操作
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，期望输入参数x为形状为(1, 6)的TensorType
            def forward(self, x: TensorType((1, 6))):
                # 调用torch的reshape函数，将输入张量x重新reshape为[1, 2, 3]的形状
                return torch.reshape(x, [1, 2, 3])

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，得到图模块symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建一个GraphTypeChecker对象tc，用于类型检查
        tc = GraphTypeChecker({}, symbolic_traced)
        # 断言类型检查通过
        self.assertTrue(tc.type_check())

        # 遍历symbolic_traced图的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点n的操作为"placeholder"
            if n.op == "placeholder":
                # 断言节点n的类型为TensorType((1, 6))
                assert n.type == TensorType((1, 6))

            # 如果节点n的操作为"call_function"
            if n.op == "call_function":
                # 断言节点n的类型为TensorType((1, 2, 3))
                assert n.type == TensorType((1, 2, 3))

            # 如果节点n的操作为"output"
            if n.op == "output":
                # 断言节点n的类型为TensorType((1, 2, 3))
                assert n.type == TensorType((1, 2, 3))

    def test_type_check_reshape_false(self):
        # 定义一个继承自torch.nn.Module的类M，用于测试类型检查和错误的reshape操作
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，期望输入参数x为形状为(1, 5)的TensorType
            def forward(self, x: TensorType((1, 5))):
                # 调用torch的reshape函数，将输入张量x尝试reshape为[1, 2, 3]的形状
                return torch.reshape(x, [1, 2, 3])

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，得到图模块symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建一个GraphTypeChecker对象tc，用于类型检查
        tc = GraphTypeChecker({}, symbolic_traced)
        # 断言类型检查引发TypeError异常
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_false(self):
        # 定义一个继承自torch.nn.Module的类M，用于测试类型检查和错误的动态reshape操作
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，期望输入参数x为形状为(1, 5)的TensorType
            def forward(self, x: TensorType((1, 5))):
                # 调用torch的reshape函数，将输入张量x尝试reshape为[1, 2, -1]的形状
                return torch.reshape(x, [1, 2, -1])

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，得到图模块symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建一个GraphTypeChecker对象tc，用于类型检查
        tc = GraphTypeChecker({}, symbolic_traced)
        # 断言类型检查引发TypeError异常
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_true(self):
        # 定义一个继承自torch.nn.Module的类M，用于测试类型检查和正确的动态reshape操作
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，期望输入参数x为形状为(1, 15)的TensorType
            def forward(self, x: TensorType((1, 15))):
                # 调用torch的reshape函数，将输入张量x reshape为[1, 5, -1]的形状
                return torch.reshape(x, [1, 5, -1])

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，得到图模块symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建一个GraphTypeChecker对象tc，用于类型检查
        tc = GraphTypeChecker({}, symbolic_traced)
        # 断言类型检查通过
        self.assertTrue(tc.type_check())

    def test_type_check_reshape_dyn_true_param_false(self):
        # 定义一个继承自torch.nn.Module的类M，用于测试类型检查和错误的动态reshape操作（参数中包含Dyn）
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，期望输入参数x为形状为(Dyn, 5)的TensorType
            def forward(self, x: TensorType((Dyn, 5))):
                # 调用torch的reshape函数，将输入张量x尝试reshape为[1, 2, -1]的形状
                return torch.reshape(x, [1, 2, -1])

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，得到图模块symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建一个GraphTypeChecker对象tc，用于类型检查
        tc = GraphTypeChecker({}, symbolic_traced)
        # 断言类型检查引发TypeError异常
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_transpose_true(self):
        # 定义一个继承自torch.nn.Module的类M，用于测试类型检查和transpose操作
        class M(torch.nn.Module):
            # 定义模型的前向传播方法，期望输入参数x为形状为(1, 2, 3, 5)的TensorType
            def forward(self, x: TensorType((1, 2, 3, 5))):
                # 调用torch的transpose函数，对输入张量x进行transpose操作，交换维度0和1
                return torch.transpose(x, 0, 1)

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，得到图模块symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建一个GraphTypeChecker对象tc，用于类型检查
        tc = GraphTypeChecker({}, symbolic_traced)
        # 断言类型检查通过
        self.assertTrue(tc.type_check())

        # 遍历symbolic_traced图的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点n的操作为"call_function"
            if n.op == "call_function":
                # 断言节点n的类型为TensorType([2, 1, 3, 5])
                assert n.type == TensorType([2, 1, 3, 5])
            # 如果节点n的操作为"output"
            if n.op == "output":
                # 断言节点n的类型为TensorType([2, 1, 3, 5])
                assert n.type == TensorType([2, 1, 3, 5])
            # 如果节点n的操作为"x"
            if n.op == "x":
                # 断言节点n的占位符类型为TensorType([1, 2, 3, 5])
                assert n.placeholder == TensorType([1, 2, 3, 5])
    def test_type_check_transpose_False(self):
        # 定义一个测试函数，用于检查类型和转置操作是否正常工作
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5))):
                # 模型的前向传播方法，输入 x 应符合指定的类型 (1, 2, 3, 5)
                return torch.transpose(x, 0, 10)
                # 对输入 x 进行维度转置操作

        module = M()
        # 创建模型实例
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 对模型进行符号化追踪，生成符号图
        tc = GraphTypeChecker({}, symbolic_traced)
        # 创建类型检查器，传入空字典和符号化追踪后的模型
        with self.assertRaises(TypeError):
            # 断言捕获 TypeError 异常
            tc.type_check()

    def test_type_check_batch_norm_2D(self):
        # 定义一个测试函数，用于检查 2D 批量归一化操作的类型
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)
                # 初始化时设置批量归一化层

            def forward(self, x: TensorType((2, 2, 5, 4))):
                # 模型的前向传播方法，输入 x 应符合指定的类型 (2, 2, 5, 4)
                identity = x
                # 复制输入作为身份映射
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                # 对输入 x 进行批量归一化操作，并期望输出类型为 (2, 2, Dyn, 4)
                out += identity
                # 将批量归一化后的输出与身份映射相加
                return out
                # 返回输出结果

        B = BasicBlock(2, 2)
        # 创建 BasicBlock 类的实例
        ast_rewriter = RewritingTracer()
        # 创建重写追踪器
        graph = ast_rewriter.trace(B)
        # 对 BasicBlock 实例进行追踪，生成图形表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建图模块实例，传入根节点、追踪得到的图形和名称
        tc = GraphTypeChecker({}, traced)
        # 创建类型检查器，传入空字典和追踪得到的图模块
        tc.type_check()
        # 执行类型检查

        for n in graph.nodes:
            # 遍历图中的每个节点
            if n.op == "placeholder":
                # 如果节点操作为“placeholder”
                assert n.type == TensorType((2, 2, 5, 4))
                # 断言节点的类型为 (2, 2, 5, 4)
            if n.op == "output":
                # 如果节点操作为“output”
                assert n.type == TensorType((2, 2, 5, 4))
                # 断言节点的类型为 (2, 2, 5, 4)
            if n.op == "call_module":
                # 如果节点操作为“call_module”
                assert n.type == TensorType((2, 2, 5, 4))
                # 断言节点的类型为 (2, 2, 5, 4)
            if n.op == "call_function":
                # 如果节点操作为“call_function”
                assert n.type == TensorType((2, 2, 5, 4))
                # 断言节点的类型为 (2, 2, 5, 4)

    def test_type_check_batch_norm_2D_false(self):
        # 定义一个测试函数，用于检查不符合要求的 2D 批量归一化操作的类型
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)
                # 初始化时设置批量归一化层

            def forward(self, x: TensorType((2, 2, 5))):
                # 模型的前向传播方法，输入 x 应符合指定的类型 (2, 2, 5)
                identity = x
                # 复制输入作为身份映射
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                # 对输入 x 进行批量归一化操作，并期望输出类型为 (2, 2, Dyn, 4)
                out += identity
                # 将批量归一化后的输出与身份映射相加
                return out
                # 返回输出结果

        B = BasicBlock(2, 2)
        # 创建 BasicBlock 类的实例
        ast_rewriter = RewritingTracer()
        # 创建重写追踪器
        graph = ast_rewriter.trace(B)
        # 对 BasicBlock 实例进行追踪，生成图形表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建图模块实例，传入根节点、追踪得到的图形和名称
        tc = GraphTypeChecker({}, traced)
        # 创建类型检查器，传入空字典和追踪得到的图模块
        with self.assertRaises(TypeError):
            # 断言捕获 TypeError 异常
            tc.type_check()
    def test_type_check_batch_norm_2D_broadcast(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化方法，接受输入参数 inplanes 和 planes
            def __init__(self, inplanes, planes):
                super().__init__()
                # 设置 norm_layer 为 BatchNorm2d
                norm_layer = torch.nn.BatchNorm2d
                # 初始化 self.bn1 为 norm_layer(planes)
                self.bn1 = norm_layer(planes)

            # 前向传播方法，接受输入参数 x，其类型为 Dyn
            def forward(self, x: Dyn):
                # 将输入 x 赋值给 identity
                identity = x
                # 对 self.bn1 应用 x，输出为 out，期望类型为 (2, 2, Dyn, 4)
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                # 将 out 与 identity 相加
                out += identity
                # 返回 out
                return out

        # 创建 BasicBlock 实例 B，输入参数为 (2, 2)
        B = BasicBlock(2, 2)
        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 追踪 BasicBlock 的图形表示，存储在 graph 中
        graph = ast_rewriter.trace(B)
        # 创建 GraphModule 实例 traced，使用 ast_rewriter.root 作为根节点，图形为 graph，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建 GraphTypeChecker 实例 tc，用空字典和 traced 进行初始化
        tc = GraphTypeChecker({}, traced)
        # 对图形进行类型检查
        tc.type_check()
        # 遍历图形中的每个节点
        for n in graph.nodes:
            # 如果节点的操作为 "placeholder"
            if n.op == "placeholder":
                # 断言节点的类型为 TensorType((Dyn, Dyn, Dyn, Dyn))
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            # 如果节点的操作为 "call_function"
            if n.op == "call_function":
                # 断言节点的类型为 TensorType((Dyn, Dyn, Dyn, Dyn))
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            # 如果节点的操作为 "output"
            if n.op == "output":
                # 断言节点的类型为 TensorType((Dyn, Dyn, Dyn, Dyn))
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            # 如果节点的操作为 "call_module"
            if n.op == "call_module":
                # 断言节点的类型为 TensorType((2, 2, Dyn, 4))
                assert n.type == TensorType((2, 2, Dyn, 4))

        # 创建 BasicBlock 实例 B，输入参数为 (1, 1)
        B = BasicBlock(1, 1)
        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 追踪 BasicBlock 的图形表示，存储在 graph 中
        graph = ast_rewriter.trace(B)
        # 创建 GraphModule 实例 traced，使用 ast_rewriter.root 作为根节点，图形为 graph，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建 GraphTypeChecker 实例 tc，用空字典和 traced 进行初始化
        tc = GraphTypeChecker({}, traced)
        # 断言调用 tc.type_check() 会引发 TypeError 异常
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_conv2D(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化方法，接受输入参数 inplanes, planes 和 stride（默认为 1）
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                # 设置 norm_layer 为 BatchNorm2d
                norm_layer = torch.nn.BatchNorm2d
                # 初始化 self.conv1 为 conv3x3(inplanes, planes, stride)
                self.conv1 = conv3x3(inplanes, planes, stride)
                # 初始化 self.bn1 为 norm_layer(planes)
                self.bn1 = norm_layer(planes)

            # 前向传播方法，接受输入参数 x，其类型为 Dyn
            def forward(self, x: Dyn):
                # 将输入 x 赋值给 identity
                identity = x
                # 对 self.conv1 应用 x，输出为 out，期望类型为 (2, 2, Dyn, 4)
                out: TensorType((2, 2, Dyn, 4)) = self.conv1(x)
                # 将 out 与 identity 相加
                out += identity
                # 返回 out
                return out

        # 创建 BasicBlock 实例 B，输入参数为 (2, 2)
        B = BasicBlock(2, 2)
        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 追踪 BasicBlock 的图形表示，存储在 graph 中
        graph = ast_rewriter.trace(B)
        # 创建 GraphModule 实例 traced，使用 ast_rewriter.root 作为根节点，图形为 graph，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建 GraphTypeChecker 实例 tc，用空字典和 traced 进行初始化
        tc = GraphTypeChecker({}, traced)
        # 对图形进行类型检查
        tc.type_check()
        # 遍历图形中的每个节点
        for n in graph.nodes:
            # 如果节点的操作为 "placeholder"
            if n.op == "placeholder":
                # 断言节点的类型为 TensorType((Dyn, Dyn, Dyn, Dyn))
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            # 如果节点的操作为 "call_function"
            if n.op == "call_function":
                # 断言节点的类型为 TensorType((Dyn, Dyn, Dyn, Dyn))
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            # 如果节点的操作为 "output"
            if n.op == "output":
                # 断言节点的类型为 TensorType((Dyn, Dyn, Dyn, Dyn))
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            # 如果节点的操作为 "call_module"
            if n.op == "call_module":
                # 断言节点的类型为 TensorType((2, 2, Dyn, 4))
                assert n.type == TensorType((2, 2, Dyn, 4))
    # 定义一个测试方法，用于检查 conv2D 的类型转换
    def test_type_check_conv2D_2(self):
        # 定义一个名为 BasicBlock 的内嵌类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化方法，设置基本参数和卷积层
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            # 前向传播方法，接收类型为 TensorType((5, 2, 3, 4)) 的输入 x
            def forward(self, x: TensorType((5, 2, 3, 4))):
                # 将输入作为身份映射
                identity = x
                # 对输入应用卷积操作
                out = self.conv1(x)
                # 将卷积输出与身份映射相加
                out += identity
                return out

        # 创建 BasicBlock 类的实例 B，传入参数为 (2, 2)
        B = BasicBlock(2, 2)
        # 调用 BasicBlock 类的 forward 方法，并传入随机生成的大小为 (5, 2, 3, 4) 的张量 b
        b = B.forward(torch.rand(5, 2, 3, 4))

        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，生成计算图 graph
        graph = ast_rewriter.trace(B)
        # 创建 GraphModule 实例 traced，传入追踪根节点 ast_rewriter.root 和计算图 graph，并命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建 GraphTypeChecker 实例 tc，传入空字典和追踪后的图 traced
        tc = GraphTypeChecker({}, traced)
        # 进行类型检查
        tc.type_check()

        # 创建一个类型为 TensorType((5, 2, 3, 4)) 的张量 t
        t = TensorType((5, 2, 3, 4))
        # 遍历图中的节点
        for n in graph.nodes:
            # 如果节点的操作是 "placeholder"，断言其类型为 t
            if n.op == "placeholder":
                assert n.type == t
            # 如果节点的操作是 "call_function"，断言其类型为 t
            if n.op == "call_function":
                assert n.type == t
            # 如果节点的操作是 "output"，断言其类型为 b 的形状 torch.Size(n.type.__args__) 应与 b 的形状相同
            if n.op == "output":
                assert torch.Size(n.type.__args__) == b.shape
            # 如果节点的操作是 "call_module"，断言其类型为 t
            if n.op == "call_module":
                assert n.type == t

        # 创建另一个 BasicBlock 类的实例 B，传入参数为 (1, 2)
        B = BasicBlock(1, 2)
        # 重新创建 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对新的 BasicBlock 类进行追踪，生成新的计算图 graph
        graph = ast_rewriter.trace(B)
        # 创建新的 GraphModule 实例 traced，传入追踪根节点 ast_rewriter.root 和新的计算图 graph，并命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建新的 GraphTypeChecker 实例 tc，传入空字典和新的追踪后的图 traced
        tc = GraphTypeChecker({}, traced)
        # 断言类型检查抛出 TypeError 异常
        with self.assertRaises(TypeError):
            tc.type_check()
    def test_typecheck_basicblock(self):
        # 定义一个测试用的 BasicBlock 类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 拓展因子，默认为 1
            expansion = 1

            # 构造函数，初始化 BasicBlock 实例
            def __init__(
                self,
                inplanes,  # 输入平面数
                planes,  # 输出平面数
                stride=1,  # 步幅，默认为 1
                downsample=None,  # 下采样层，默认为 None
                groups=1,  # 分组卷积数，默认为 1
                base_width=64,  # 基础宽度，默认为 64
                dilation=1,  # 膨胀率，默认为 1
            ):
                super().__init__()  # 调用父类的构造函数
                norm_layer = torch.nn.BatchNorm2d  # 规范化层为 BatchNorm2d
                # 如果 groups 不等于 1 或 base_width 不等于 64，则抛出错误
                if groups != 1 or base_width != 64:
                    raise ValueError(
                        "BasicBlock only supports groups=1 and base_width=64"
                    )
                # 如果 dilation 大于 1，则抛出错误
                if dilation > 1:
                    raise NotImplementedError(
                        "Dilation > 1 not supported in BasicBlock"
                    )
                # 第一个卷积层和下采样层在 stride 不等于 1 时会降低输入的维度
                self.conv1 = conv3x3(inplanes, planes, stride)  # 第一个 3x3 卷积层
                self.bn1 = norm_layer(planes)  # 第一个规范化层
                self.relu = torch.nn.ReLU(inplace=True)  # ReLU 激活函数
                self.conv2 = conv3x3(planes, planes)  # 第二个 3x3 卷积层
                self.bn2 = norm_layer(planes)  # 第二个规范化层
                self.downsample = downsample  # 下采样层
                self.stride = stride  # 步幅

            # 前向传播函数，接收一个大小为 (2, 2, 4, 5) 的张量作为输入
            def forward(self, x: TensorType((2, 2, 4, 5))):
                identity = x  # 原始输入作为 identity

                out = self.conv1(x)  # 第一次卷积
                out = self.bn1(out)  # 第一次规范化
                out = self.relu(out)  # 第一次 ReLU 激活

                out = self.conv2(out)  # 第二次卷积
                out = self.bn2(out)  # 第二次规范化

                # 如果有下采样层，则对输入进行下采样
                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity  # 加上残差连接
                out = self.relu(out)  # 最终输出经过 ReLU 激活

                return out  # 返回输出张量

        B = BasicBlock(2, 2)  # 创建 BasicBlock 实例 B，输入输出平面数均为 2

        ast_rewriter = RewritingTracer()  # 创建重写追踪器实例
        graph = ast_rewriter.trace(B)  # 对 BasicBlock B 进行追踪，生成图形表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")  # 创建图形模块实例 traced

        tc = GraphTypeChecker({}, traced)  # 创建图类型检查器实例
        tc.type_check()  # 执行类型检查

        # 遍历图中的节点
        for n in traced.graph.nodes:
            # 如果节点的目标是 "output"
            if n.target == "output":
                # 断言节点的类型是 TensorType
                assert isinstance(n.type, TensorType)
                # 断言节点的类型大小与 BasicBlock 的前向传播结果的大小相同
                assert (
                    torch.Size(n.type.__args__)
                    == B.forward(torch.rand(2, 2, 4, 5)).size()
                )
    def test_type_check_conv2D_maxpool2d_flatten(self):
        # 定义一个名为 BasicBlock 的子类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 构造函数，初始化网络的各个层
            def __init__(self):
                super().__init__()

                # 第一个卷积层，输入通道数为 3，输出通道数为 6，卷积核大小为 5
                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                # 最大池化层，窗口大小为 2x2，步长为 2
                self.pool = torch.nn.MaxPool2d(2, 2)
                # 第二个卷积层，输入通道数为 6，输出通道数为 16，卷积核大小为 5
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                # 全连接层，输入大小为 5，输出大小为 120
                self.fc1 = torch.nn.Linear(5, 120)
                # 自适应平均池化层，输出特征图大小为 6x7
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            # 前向传播函数，接受一个输入张量 x，返回处理后的输出张量
            def forward(self, x: TensorType((4, 3, 32, 32))):
                out = self.conv1(x)  # 第一层卷积操作
                out = self.pool(out)  # 第一层池化操作
                out = self.conv2(out)  # 第二层卷积操作
                out = self.pool(out)  # 第二层池化操作
                out = self.fc1(out)  # 全连接层操作
                out = self.pool2(out)  # 自适应平均池化操作
                out = torch.flatten(out, 1)  # 将输出张量展平为一维
                return out

        # 创建 BasicBlock 实例 B
        B = BasicBlock()
        # 创建 AST 重写器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，生成图形表示
        graph = ast_rewriter.trace(B)
        # 创建 GraphModule 实例 traced，将 AST 根节点和图形对象传入，并命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建类型检查器对象，传入空字典和追踪后的图形模块 traced
        tc = GraphTypeChecker({}, traced)
        # 执行类型检查
        tc.type_check()

        # 预期的占位符类型列表
        expected_ph_types = [
            TensorType((4, 3, 32, 32)),
            TensorType((4, 6, 28, 28)),
            TensorType((4, 6, 14, 14)),
            TensorType((4, 16, 10, 10)),
            TensorType((4, 16, 5, 5)),
            TensorType((4, 16, 5, 120)),
            TensorType((4, 16, 6, 7)),
            TensorType((4, 672)),
            TensorType((4, 672)),
        ]

        # 创建预期类型迭代器
        expected_iter = iter(expected_ph_types)
        # 消除无效代码
        traced.graph.eliminate_dead_code()

        # 遍历图中的节点
        for n in traced.graph.nodes:
            # 断言节点的类型与预期类型迭代器中的下一个类型相等
            assert n.type == next(expected_iter)

    def test_type_check_flatten(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 前向传播函数，接受一个输入张量 x，并声明其类型为 TensorType((1, 2, 3, 5, Dyn))
            def forward(self, x: TensorType((1, 2, 3, 5, Dyn))):
                return torch.flatten(x, 1, 2)

        # 创建 M 的实例 module
        module = M()
        # 使用符号追踪函数 symbolic_trace 对 module 进行追踪，得到图形模块 symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建类型检查器对象，传入空字典和追踪后的图形模块 symbolic_traced
        tc = GraphTypeChecker({}, symbolic_traced)
        # 执行类型检查
        tc.type_check()
        
        # 遍历图中的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点操作为 "output"，则断言节点的类型为 TensorType((1, 6, 5, Dyn))
            if n.op == "output":
                assert n.type == TensorType((1, 6, 5, Dyn))

    def test_type_check_flatten_2(self):
        # 定义一个名为 M 的子类，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 前向传播函数，接受一个输入张量 x，并声明其类型为 TensorType((1, Dyn, 3, 5, Dyn))
            def forward(self, x: TensorType((1, Dyn, 3, 5, Dyn))):
                return torch.flatten(x, 1, 2)

        # 创建 M 的实例 module
        module = M()
        # 使用符号追踪函数 symbolic_trace 对 module 进行追踪，得到图形模块 symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建类型检查器对象，传入空字典和追踪后的图形模块 symbolic_traced
        tc = GraphTypeChecker({}, symbolic_traced)
        # 执行类型检查
        tc.type_check()

        # 遍历图中的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点操作为 "output"，则断言节点的类型为 TensorType((1, Dyn, 5, Dyn))
            if n.op == "output":
                assert n.type == TensorType((1, Dyn, 5, Dyn))
    def test_type_check_flatten3(self):
        # 定义一个继承自torch.nn.Module的模块类M
        class M(torch.nn.Module):
            # 定义模块类的前向传播方法，参数x的类型注解为TensorType((2, 3, 4, 5))
            def forward(self, x: TensorType((2, 3, 4, 5))):
                # 调用torch.flatten函数，对输入x进行扁平化处理，从第1维到第3维
                return torch.flatten(x, start_dim=1, end_dim=3)

        # 创建M类的实例module
        module = M()
        # 使用symbolic_trace对module进行符号化追踪，返回torch.fx.GraphModule类型的对象symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建GraphTypeChecker类的实例tc，传入空字典和symbolic_traced对象
        tc = GraphTypeChecker({}, symbolic_traced)
        # 对tc对象进行类型检查
        tc.type_check()
        # 遍历symbolic_traced图中的节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点n的操作为"output"
            if n.op == "output":
                # 断言节点n的类型为TensorType((2, 60))
                assert n.type == TensorType((2, 60))
        # 创建Refine类的实例r，传入symbolic_traced对象
        r = Refine(symbolic_traced)
        # 对r对象进行精化处理
        r.refine()
        # 获取r对象的约束条件列表，赋值给变量c
        c = r.constraints
        # 断言约束条件c为[Equality(2, 2)]
        assert c == [Equality(2, 2)]

    def test_type_typechecl_maxpool2d_3dinput(self):
        # 定义一个继承自torch.nn.Module的基本块类BasicBlock
        class BasicBlock(torch.nn.Module):
            # 类初始化方法
            def __init__(self):
                super().__init__()
                # 初始化BasicBlock对象的pool属性为torch.nn.MaxPool2d(5, 8)
                self.pool = torch.nn.MaxPool2d(5, 8)

            # 定义模块类的前向传播方法，参数x的类型注解为TensorType((64, 8, 8))
            def forward(self, x: TensorType((64, 8, 8))):
                # 对输入x调用self.pool进行最大池化操作
                out = self.pool(x)
                return out

        # 创建BasicBlock类的实例B
        B = BasicBlock()
        # 创建RewritingTracer类的实例ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对B对象进行追踪，返回一个图graph
        graph = ast_rewriter.trace(B)
        # 创建GraphModule类的实例traced，传入ast_rewriter的根节点、追踪得到的图graph和字符串"gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建GraphTypeChecker类的实例tc，传入空字典和traced对象
        tc = GraphTypeChecker({}, traced)
        # 对tc对象进行类型检查
        tc.type_check()

        # 遍历traced图中的节点
        for n in traced.graph.nodes:
            # 如果节点n的目标为"output"
            if n.target == "output":
                # 断言节点n的类型为TensorType((64, 1, 1))
                assert n.type == TensorType((64, 1, 1))
    def test_flatten_fully_static(self):
        # 定义一个注解列表，用于不同情况下的类型注解
        annotation_list = [
            Dyn,  # 动态类型标记
            TensorType((2, 5, 6, 9)),  # 固定形状的张量类型标记
            TensorType((10, 15, 13, 14)),  # 另一个固定形状的张量类型标记
            TensorType((10, Dyn, 13, 14)),  # 含有动态维度的张量类型标记
            TensorType((Dyn, Dyn, Dyn, 10)),  # 多个动态维度的张量类型标记
        ]
        # 定义输入列表，每个元素是一个元组，表示输入张量的形状
        input_list = [
            (1, 2, 3, 5),
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, 15, 13, 14),
            (2, 2, 10, 10),
        ]

        # 中间类型列表，用于记录中间过程中的类型变化，当前未使用
        intermediate_list = [
            Dyn,
            (2, 5, 6, 9),
            (10, 15, 13, 14),
            (10, 15, 13, 14),
            (2, 2, 10, 10),
        ]

        # 定义起始维度列表和结束维度列表，用于指定在flatten操作中展平的维度范围
        start_dim = [1, 2, 1, 2, 0]
        end_dim = [1, 3, 3, 3, -2]

        # 遍历五种不同情况
        for i in range(5):
            annotation = annotation_list[i]  # 获取当前情况下的类型注解
            input = input_list[i]  # 获取当前情况下的输入形状

            # 定义一个基本的块类，用于测试flatten操作
            class BasicBlock(torch.nn.Module):
                def __init__(self, start, end):
                    super().__init__()
                    self.start = start  # 初始化起始维度
                    self.end = end  # 初始化结束维度

                def forward(self, x):
                    # 对输入张量进行flatten操作，按照指定的起始和结束维度
                    out = torch.flatten(x, self.start, self.end)
                    return out

            # 创建一个BasicBlock实例B，根据当前情况的起始和结束维度
            B = BasicBlock(start_dim[i], end_dim[i])

            # 创建一个重写跟踪器实例
            ast_rewriter = RewritingTracer()

            # 使用重写跟踪器对BasicBlock实例B进行跟踪，获取图形表示
            graph = ast_rewriter.trace(B)

            # 创建一个图模块实例，使用重写跟踪器的根节点和跟踪得到的图形
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # 对图中的占位符节点进行类型注解
            for n in graph.nodes:
                if n.op == "placeholder":
                    n.type = annotation

            # 使用输入input对BasicBlock实例B进行前向传播
            b = B.forward(torch.rand(input))

            # 创建一个图类型检查器实例，使用空字典和traced图模块
            tc = GraphTypeChecker({}, traced)

            # 进行类型检查
            tc.type_check()

            # 遍历图中的节点，如果节点的操作是"output"，则断言其类型与b张量的类型一致
            for n in graph.nodes:
                if n.op == "output":
                    assert is_consistent(n.type, TensorType(b.size()))

    @skipIfNoTorchVision
    def test_resnet50(self):
        # 对 resnet50 模型进行符号化跟踪
        gm_run = symbolic_trace(resnet50())
        # 创建一个样本输入张量
        sample_input = torch.randn(1, 3, 224, 224)

        # 运行节点
        ShapeProp(gm_run).propagate(sample_input)

        # 再次对 resnet50 模型进行符号化跟踪
        gm_static = symbolic_trace(resnet50())

        # 将静态图中所有节点的类型设置为 None
        for n in gm_static.graph.nodes:
            n.type = None

        # 创建图类型检查器实例
        g = GraphTypeChecker({}, gm_static)
        # 进行类型检查
        g.type_check()
        # 消除静态图中的死代码
        gm_static.graph.eliminate_dead_code()
        gm_run.graph.eliminate_dead_code()

        # 在这里我们检查与完全动态节点的一致性
        for n1, n2 in zip(gm_static.graph.nodes, gm_run.graph.nodes):
            assert is_consistent(n1.type, TensorType(n2.meta["tensor_meta"].shape))

        # 使用与运行时相同的输入
        gm_static_with_types = symbolic_trace(resnet50())

        # 初始化占位符
        for n in gm_static_with_types.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType((1, 3, 224, 224))

        # 再次创建图类型检查器实例
        g = GraphTypeChecker({}, gm_static_with_types)
        # 再次进行类型检查
        g.type_check()

        # 检查静态图和运行时图节点的类型是否匹配
        for n1, n2 in zip(gm_static_with_types.graph.nodes, gm_run.graph.nodes):
            assert n1.type == TensorType(n2.meta["tensor_meta"].shape)

        # 对图应用形状推断，并检查所有层的批处理大小是否相等
        infer_symbolic_types(gm_static)

        # 收集批处理大小
        batch_sizes = set()
        gm_static.graph.eliminate_dead_code()
        for n in gm_static.graph.nodes:
            assert isinstance(n.type, TensorType)
            batch_sizes.add(n.type.__args__[0])
        assert len(batch_sizes) == 1

    def test_type_check_batch_norm_symbolic(self):
        # 定义一个 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                identity = x
                # 对 x 应用批量归一化
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        # 创建 BasicBlock 实例 B
        B = BasicBlock(2, 2)
        # 创建 AST 重写器
        ast_rewriter = RewritingTracer()
        # 跟踪 BasicBlock 实例的图形表示
        graph = ast_rewriter.trace(B)
        # 创建图模块，包含 AST 重写器的根节点和跟踪得到的图形
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建图类型检查器实例
        tc = GraphTypeChecker({}, traced)
        # 进行类型检查
        tc.type_check()

        # 对图应用形状推断
        infer_symbolic_types(traced)

        # 准备预期的类型列表
        my_types = iter(
            [
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
                TensorType[(2, 2, sympy.symbols("~7"), 4)],
            ]
        )

        # 验证图中每个节点的类型是否符合预期
        for n in graph.nodes:
            assert n.type == next(my_types)
    def test_symbolic_add_with_broadcast(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                return torch.add(x, y)
        
        module = M()
        # 对模块进行符号化跟踪
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建类型检查器，并初始化为空字典
        tc = GraphTypeChecker({}, symbolic_traced)
        # 进行类型检查
        tc.type_check()
        # 推断符号类型
        infer_symbolic_types(symbolic_traced)
        # 创建精化对象，并进行精化操作
        r = Refine(symbolic_traced)
        r.refine()

        # 断言精化后的约束条件
        assert r.constraints == [Equality(1, 1), Equality(2, 2), Equality(3, 3)]
        # 注意：dyn 和 4 之间没有等式约束，因为 dyn 可能是 4 或 1

        # 再次推断符号类型
        infer_symbolic_types(symbolic_traced)

        # 预期的占位符类型
        expected_ph_types = [
            TensorType((1, 2, 3, sympy.symbols("~0"))),
            TensorType((2, 3, 4)),
            TensorType((1, 2, 3, sympy.symbols("~1"))),
            TensorType((1, 2, 3, sympy.symbols("~1"))),
        ]
        expected_iter = iter(expected_ph_types)

        # 断言每个节点的类型是否符合预期
        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_symbolic_add_with_broadcast_2(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2)), y: TensorType((Dyn, 2))):
                return torch.add(x, y)

        module = M()
        # 对模块进行符号化跟踪
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 创建类型检查器，并初始化为空字典
        tc = GraphTypeChecker({}, symbolic_traced)
        # 进行类型检查
        tc.type_check()
        # 推断符号类型
        infer_symbolic_types(symbolic_traced)
        # 创建精化对象，并进行精化操作
        r = Refine(symbolic_traced)
        r.refine()

        # 预期的占位符类型
        expected_ph_types = [
            TensorType((1, 2)),
            TensorType((sympy.symbols("~1"), 2)),
            TensorType((sympy.symbols("~1"), 2)),
            TensorType((sympy.symbols("~1"), 2)),
        ]
        expected_iter = iter(expected_ph_types)

        # 断言每个节点的类型是否符合预期
        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_type_check_conv2D_types(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                identity = x
                # 执行卷积操作，并期望输出类型为 TensorType((2, 2, Dyn, 4))
                out: TensorType((2, 2, Dyn, 4)) = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        # 创建 AST 重写跟踪器
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 进行跟踪，并生成图形对象
        graph = ast_rewriter.trace(B)
        # 创建图形模块，传入 AST 根节点和生成的图形对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建类型检查器，并初始化为空字典
        tc = GraphTypeChecker({}, traced)
        # 进行类型检查
        tc.type_check()
        # 推断符号类型
        infer_symbolic_types(traced)

        # 断言每个节点的类型是否符合预期
        for n in traced.graph.nodes:
            if n.op == "call_module":
                # 断言类型参数的第三个和第四个是否为 sympy.floor 类型
                assert isinstance(n.type.__args__[2], sympy.floor)
                assert isinstance(n.type.__args__[3], sympy.floor)
    def test_type_check_symbolic_inferenceconv2D_maxpool2d_flatten(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # 定义网络结构的各个层：conv1, pool, conv2, fc1, pool2
                self.conv1 = torch.nn.Conv2d(3, 6, 5)  # 输入通道数为3，输出通道数为6，卷积核大小为5
                self.pool = torch.nn.MaxPool2d(2, 2)   # 最大池化层，池化窗口大小为2，步长为2
                self.conv2 = torch.nn.Conv2d(6, 16, 5) # 输入通道数为6，输出通道数为16，卷积核大小为5
                self.fc1 = torch.nn.Linear(5, 120)     # 线性全连接层，输入特征数为5，输出特征数为120
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))  # 自适应平均池化层，输出尺寸为(6, 7)

            # 定义前向传播函数，参数 x 的类型为 TensorType((4, 3, Dyn, Dyn))
            def forward(self, x: TensorType((4, 3, Dyn, Dyn))):
                out = self.conv1(x)  # 卷积层 conv1 对输入 x 进行卷积操作
                out = self.pool(out)  # 使用池化层 pool 对卷积结果 out 进行池化操作
                out = self.conv2(out)  # 卷积层 conv2 对池化结果 out 进行卷积操作
                out = self.pool(out)   # 再次使用池化层 pool 对卷积结果 out 进行池化操作
                out = self.fc1(out)    # 全连接层 fc1 对池化结果 out 进行全连接操作
                out = self.pool2(out)  # 使用自适应平均池化层 pool2 对全连接结果 out 进行池化操作
                out = torch.flatten(out, 1)  # 将池化后的结果 out 按指定维度进行扁平化
                return out

        B = BasicBlock()  # 创建 BasicBlock 类的实例 B
        ast_rewriter = RewritingTracer()  # 创建重写追踪器的实例 ast_rewriter
        traced = symbolic_trace(B)  # 对 BasicBlock 实例 B 进行符号化跟踪
        tc = GraphTypeChecker({}, traced)  # 创建图类型检查器的实例 tc，传入空字典和符号化跟踪结果 traced
        tc.type_check()  # 进行类型检查
        infer_symbolic_types(traced)  # 对符号化跟踪结果 traced 进行符号类型推断

        # 遍历跟踪图中的节点
        for n in traced.graph.nodes:
            if n.target == "conv1":
                # 断言节点 n 的类型为 TensorType((4, 6, sympy.floor(sympy.symbols("~0") - 4), sympy.floor(sympy.symbols("~1") - 4)))
                assert n.type == TensorType(
                    (
                        4,
                        6,
                        sympy.floor(sympy.symbols("~0") - 4),
                        sympy.floor(sympy.symbols("~1") - 4),
                    )
                )
            elif n.target == "conv2":
                # 断言节点 n 的类型为 TensorType((4, 16, sympy.floor(sympy.symbols("~4") - 4), sympy.floor(sympy.symbols("~5") - 4)))
                assert n.type == TensorType(
                    (
                        4,
                        16,
                        sympy.floor(sympy.symbols("~4") - 4),
                        sympy.floor(sympy.symbols("~5") - 4),
                    )
                )
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行单元测试
if __name__ == "__main__":
    # 调用 unittest 模块的主函数，运行所有的单元测试
    unittest.main()
```