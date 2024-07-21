# `.\pytorch\test\fx\test_z3_gradual_types.py`

```py
# 引入操作符模块，用于操作符相关功能
import operator
# 引入单元测试模块
import unittest

# 引入 PyTorch 深度学习框架
import torch
# 从 torch.fx 中引入图模块和符号化追踪功能
from torch.fx import GraphModule, symbolic_trace
# 从 torch.fx.experimental.meta_tracer 中引入符号化追踪功能
from torch.fx.experimental.meta_tracer import symbolic_trace as meta_symbolic_trace
# 从 torch.fx.experimental.migrate_gradual_types.constraint 中引入类型约束相关类
from torch.fx.experimental.migrate_gradual_types.constraint import (
    BinConstraintT,
    DVar,
    T,
    TVar,
)
# 从 torch.fx.experimental.migrate_gradual_types.constraint_generator 中引入约束生成器
from torch.fx.experimental.migrate_gradual_types.constraint_generator import (
    ConstraintGenerator,
)
# 从 torch.fx.experimental.migrate_gradual_types.constraint_transformation 中引入约束转换功能
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import (
    transform_constraint,
)
# 从 torch.fx.experimental.migrate_gradual_types.operation 中引入操作一致性、匹配性和精度相关功能
from torch.fx.experimental.migrate_gradual_types.operation import (
    op_consistency,
    op_matching,
    op_precision,
)
# 从 torch.fx.experimental.migrate_gradual_types.transform_to_z3 中引入基于约束求值和所有约束转换功能
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (
    evaluate_conditional_with_constraints,
    transform_all_constraints,
)
# 从 torch.fx.experimental.migrate_gradual_types.z3_types 中引入约束求值相关类型
from torch.fx.experimental.migrate_gradual_types.z3_types import D, tensor_type, z3_dyn
# 从 torch.fx.experimental.rewriter 中引入重写追踪器
from torch.fx.experimental.rewriter import RewritingTracer
# 从 torch.fx.tensor_type 中引入动态张量和张量类型
from torch.fx.tensor_type import Dyn, TensorType

# 尝试导入 z3 库，如果失败则设置 HAS_Z3 为 False
try:
    import z3  # type: ignore[import]
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

# 尝试导入 torchvision 模型，如果失败则设置 HAS_TORCHVISION 为 False
try:
    from torchvision import models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 定义 skipIfNoTorchVision 装饰器，用于跳过没有 torchvision 的单元测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


# 定义 TorchDynamoUseCases 类，继承自 unittest.TestCase，用于测试 Torch 动态应用场景
class TorchDynamoUseCases(unittest.TestCase):
    
    # 定义测试函数 test_dim
    def test_dim(self):
        # 定义 BasicBlock 类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数 forward，接收输入 x，类型为 TensorType([1, 2])
            def forward(self, x: TensorType([1, 2])):
                # 对输入张量 x 进行维度查询操作
                y = x.dim()
                # 返回维度结果 y
                return y
        
        # 对 BasicBlock 进行符号化追踪，得到符号化的图模块 symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 对符号化图模块应用约束转换，得到转换后的结果 transformed
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建 z3 求解器对象 s
        s = z3.Solver()
        # 向求解器中添加转换后的约束
        s.add(transformed)
        # 断言求解器的检查结果为 z3.sat，即满足
        self.assertEqual(s.check(), z3.sat)
        # 创建 z3.Int 对象 y_res，值为 2
        y_res = z3.z3.Int(2)
        # 断言求解器模型中的 y_res 变量等于 2
        self.assertEqual(s.model()[y_res], 2)
    
    # 定义测试函数 test_reshape
    def test_reshape(self):
        """
        In this example, we prove that some nodes must
        always have a fixed shape regardless of the input
        """
        # 定义 BasicBlock 类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数 forward，接收输入 x，类型为 Dyn
            def forward(self, x: Dyn):
                # 对输入张量 x 进行形状重塑操作
                y = x.view(100)
                # 获取重塑后张量的第一个维度大小
                tmp = y.size()[0]
                # 返回第一个维度大小 tmp
                return tmp
        
        # 对 BasicBlock 进行符号化追踪，得到符号化的图模块 symbolic_traced
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 对符号化图模块应用约束转换，得到转换后的结果 transformed
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        
        # 创建 z3 求解器对象 s
        s = z3.Solver()
        # 向求解器中添加转换后的约束
        s.add(transformed)
        # 断言求解器的检查结果为 z3.sat，即满足
        self.assertEqual(s.check(), z3.sat)
        # 创建 z3.Int 对象 dim，值为 100
        dim = z3.Int(4)
        # 断言求解器模型中的 dim 变量等于 100
        self.assertEqual(s.model()[dim], 100)
        # 打印求解器模型中的 dim 变量（注释掉的代码）
        # print(s.model()[dim])


# 定义 HFOperations 类，继承自 unittest.TestCase
class HFOperations(unittest.TestCase):
    # 此处可以添加额外的测试函数，未提供示例代码，故不做注释
    pass
    def test_eq_dim(self):
        """
        test dimensions and equalities
        """

        # 定义一个继承自 torch.nn.Module 的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 定义前向传播方法，接受一个输入张量 x，期望维度为 [32, 4, 4]
            def forward(self, x: TensorType([32, 4, 4])):
                # 检查输入张量 x 的维度是否为 3
                eq = x.dim() == 3
                return eq

        # 创建一个 AST 重写器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，生成其计算图
        graph = ast_rewriter.trace(BasicBlock())

        # 遍历计算图的节点
        for n in graph.nodes:
            # 查找目标节点为 operator.eq 的节点
            if n.target == operator.eq:
                node = n

        # 使用给定的约束条件评估条件表达式
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        # 断言正条件为满足
        self.assertEqual(positive, z3.sat)
        # 断言负条件为不满足
        self.assertEqual(negative, z3.unsat)

    def test_conditional_ne_1(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:
        """

        # 定义一个继承自 torch.nn.Module 的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 定义前向传播方法，接受两个输入张量 x 和 y，期望维度为 [32, 4, 4]
            def forward(self, x: TensorType([32, 4, 4]), y: TensorType([32, 4, 4])):
                # 计算输入张量 x 的大小
                size_5 = x.size()
                # 获取大小的第一个元素
                getitem_7 = size_5[0]
                # 获取大小的第二个元素
                getitem_8 = size_5[1]
                # 获取大小的第三个元素
                getitem_9 = size_5[2]
                # 检查张量 y 是否不等于 (getitem_7, getitem_8, getitem_9)
                ne_1 = y != (getitem_7, getitem_8, getitem_9)
                return ne_1

        # 创建一个 AST 重写器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，生成其计算图
        graph = ast_rewriter.trace(BasicBlock())

        # 遍历计算图的节点
        for n in graph.nodes:
            # 查找目标节点为 operator.ne 的节点
            if n.target == operator.ne:
                node = n

        # 使用给定的约束条件评估条件表达式
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        # 断言正条件为不满足
        self.assertEqual(positive, z3.unsat)
        # 断言负条件为满足
        self.assertEqual(negative, z3.sat)

    def test_bmm(self):
        # 定义一个继承自 torch.nn.Module 的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 定义前向传播方法，接受两个输入张量 x 和 y，维度分别为 [Dyn, 2, 3] 和 [1, 3, 2]
            def forward(self, x: TensorType([Dyn, 2, 3]), y: TensorType([1, 3, 2])):
                # 使用 torch.bmm 计算输入张量 x 和 y 的批量矩阵乘积
                bmm = torch.bmm(x, y)
                return bmm

        # 对 BasicBlock 进行符号跟踪，生成符号图
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 调用 BasicBlock 的前向传播方法，传入随机生成的输入张量，得到输出张量 b
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        # 对符号图中的所有约束条件进行转换
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将转换后的约束条件添加到 Solver 中
        s.add(transformed)

        # 创建一个输出张量的常量符号
        output = z3.Const(3, tensor_type)
        # 断言 Solver 返回 z3.sat，表示约束条件满足
        self.assertEqual(s.check(), z3.sat)
        # 断言输出张量的第一个维度与 b 的第一个维度相等
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        # 断言输出张量的第二个维度与 b 的第二个维度相等
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        # 断言输出张量的第三个维度与 b 的第三个维度相等
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])
    def test_bmm2(self):
        # 定义一个继承自torch.nn.Module的基本块类
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数，接受参数x和y，其中x是动态类型，y是形状为[1, 3, 2]的张量类型
            def forward(self, x: Dyn, y: TensorType([1, 3, 2])):
                # 使用torch.bmm进行矩阵乘法运算
                bmm = torch.bmm(x, y)
                # 返回乘法结果
                return bmm

        # 对BasicBlock进行符号跟踪，得到torch.fx.GraphModule对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 调用BasicBlock的forward方法，传入随机生成的张量作为参数，计算bmm结果
        b = BasicBlock().forward(torch.rand(1, 2, 3), torch.rand(1, 3, 2))
        # 对符号跟踪后的图进行约束变换
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个Z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束到求解器中
        s.add(transformed)

        # 创建一个形状为tensor_type的常量output
        output = z3.Const(3, tensor_type)
        # 断言求解器的结果为满足条件
        self.assertEqual(s.check(), z3.sat)
        # 断言output的第一个维度和b的第一维度相等
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        # 断言output的第二个维度和0相等
        self.assertEqual(s.model()[output].arg(1).arg(0), 0)
        # 断言output的第三个维度和b的第三维度相等
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])

    def test_bmm3(self):
        # 定义一个继承自torch.nn.Module的基本块类
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数，接受参数x和y，其中x是形状为[2, 3, 3]的张量类型，y是形状为[1, 3, 2]的张量类型
            def forward(self, x: TensorType([2, 3, 3]), y: TensorType([1, 3, 2])):
                # 使用torch.bmm进行矩阵乘法运算
                bmm = torch.bmm(x, y)
                # 返回乘法结果
                return bmm

        # 对BasicBlock进行符号跟踪，得到torch.fx.GraphModule对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 对符号跟踪后的图进行约束变换
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个Z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束到求解器中
        s.add(transformed)
        # 断言求解器的结果为不满足条件
        self.assertEqual(s.check(), z3.unsat)

    def test_transpose(self):
        # 定义一个继承自torch.nn.Module的基本块类
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数，接受参数x，其中x是形状为[1, 2, 3, 4]的张量类型
            def forward(self, x: TensorType([1, 2, 3, 4])):
                # 使用torch.transpose进行转置操作
                transpose = x.transpose(0, 1)
                # 返回转置结果
                return transpose

        # 对BasicBlock进行符号跟踪，得到torch.fx.GraphModule对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 调用BasicBlock的forward方法，传入随机生成的张量作为参数，计算转置结果
        b = BasicBlock().forward(torch.rand(1, 2, 3, 4))

        # 对符号跟踪后的图进行约束变换
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个Z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束到求解器中
        s.add(transformed)
        # 创建一个形状为tensor_type的常量output
        output = z3.Const(2, tensor_type)
        # 断言求解器的结果为满足条件
        self.assertEqual(s.check(), z3.sat)
        # 断言output的各维度和b的对应维度相等
        self.assertEqual(s.model()[output].arg(0).arg(1), b.shape[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b.shape[1])
        self.assertEqual(s.model()[output].arg(2).arg(1), b.shape[2])
        self.assertEqual(s.model()[output].arg(3).arg(1), b.shape[3])

        # 将注解更改为Dyn
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # 再次对符号跟踪后的图进行约束变换
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个Z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束到求解器中
        s.add(transformed)
        # 断言求解器的结果为满足条件
        self.assertEqual(s.check(), z3.sat)
    def test_index_select(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，接受两个参数 x 和 y
            def forward(self, x: TensorType([2050, 1024]), y: Dyn):
                # 使用 x 的 index_select 方法，根据 y 的索引选择第一维的子集
                index_select = x.index_select(0, y)
                return index_select

        # 对 BasicBlock 进行符号跟踪，得到一个 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 创建 BasicBlock 的实例 b，并调用其 forward 方法
        b = BasicBlock().forward(torch.rand(2050, 1024), torch.ones(8).int())
        # 对符号跟踪结果进行变换，传入 counter=0
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 添加变换后的约束到求解器中
        s.add(transformed)
        # 断言求解器的结果为 z3.sat（可满足）
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个 Z3 的常量 index_select，类型为 tensor_type，值为 3
        index_select = z3.Const(3, tensor_type)

        # 断言模型中 index_select 对应的值的第二个维度应该与 b 的第二维度相同
        self.assertEqual(s.model()[index_select].arg(1).arg(1), b.shape[1])

        # 创建一个 Z3 的常量 replacement_vector，类型为 tensor_type，值为 2
        replacement_vector = z3.Const(2, tensor_type)

        # 创建一个新的 Z3 Solver 对象
        s = z3.Solver()
        # 添加变换后的约束到新的求解器中
        s.add(transformed)
        # 断言新求解器的结果为 z3.sat（可满足）
        self.assertEqual(s.check(), z3.sat)
        
        # 更新 index_select 为新的 Z3 常量，类型为 tensor_type，值为 3
        index_select = z3.Const(3, tensor_type)
        # 添加 replacement_vector 等于 z3_dyn 的约束到求解器中
        s.add(replacement_vector == z3_dyn)
        # 断言求解器的结果为 z3.sat（可满足）

        # 断言模型中 index_select 对应的值的第一个维度的第一个元素为 0
        self.assertEqual(s.model()[index_select].arg(0).arg(0), 0)

    def test_get_attr(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，接受一个参数 x
            def forward(self, x: TensorType([1, 2, 3])):
                # 获取 x 的设备信息
                getattr = x.device
                # 将 x 转换到指定的设备上
                to = x.to(getattr)
                return to

        # 对 BasicBlock 进行符号跟踪，得到一个 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 创建 BasicBlock 的实例 b，并调用其 forward 方法
        b = BasicBlock().forward(torch.rand(1, 2, 3))
        # 对符号跟踪结果进行变换，传入 counter=0
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 添加变换后的约束到求解器中
        s.add(transformed)
        # 断言求解器的结果为 z3.sat（可满足）
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个 Z3 的常量 attr_res，类型为 tensor_type，值为 3
        attr_res = z3.Const(3, tensor_type)
        
        # 断言模型中 attr_res 对应的值的第一个维度的第二个元素应该与 b 的第一个维度相同
        assert s.model()[attr_res].arg(0).arg(1) == b.shape[0]
        # 断言模型中 attr_res 对应的值的第二个维度的第二个元素应该与 b 的第二个维度相同
        assert s.model()[attr_res].arg(1).arg(1) == b.shape[1]
        # 断言模型中 attr_res 对应的值的第三个维度的第二个元素应该与 b 的第三个维度相同
        assert s.model()[attr_res].arg(2).arg(1) == b.shape[2]
    def test_expand(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，输入 x 的类型为 TensorType([1, 4])
            def forward(self, x: TensorType([1, 4])):
                # 获取输入 x 的大小
                size = x.size()
                # 获取 x 的最后一个维度
                getitem = size[-1]
                # 将输入 x 在第一个维度上进行扩展，第一个维度大小为 getitem，第二个维度大小为 4
                expand = x.expand(getitem, 4)
                # 返回扩展后的张量
                return expand

        # 创建 BasicBlock 类的实例 b，调用其 forward 方法并传入一个随机生成的 1x4 的张量
        b = BasicBlock().forward(torch.rand(1, 4))

        # 对 BasicBlock 类进行符号跟踪，得到一个 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 对符号图中的所有约束条件进行转换，counter 设为 0
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个 z3 Solver 对象
        s = z3.Solver()
        # 将转换后的约束条件添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 返回 z3 的满足状态
        self.assertEqual(s.check(), z3.sat)
        # 创建一个 z3 常量 expand_res，表示扩展结果的张量类型
        expand_res = z3.Const(4, tensor_type)
        # 断言 Solver 模型中 expand_res 的第一个参数的第一个元素等于 b 的第一个维度大小
        assert s.model()[expand_res].arg(0).arg(1) == b.shape[0]
        # 断言 Solver 模型中 expand_res 的第二个参数的第一个元素等于 b 的第二个维度大小
        assert s.model()[expand_res].arg(1).arg(1) == b.shape[1]

        # 修改输入的类型注释为 Dyn
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # 重新转换所有约束条件
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个新的 z3 Solver 对象
        s = z3.Solver()
        # 将重新转换后的约束条件添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 返回 z3 的满足状态
        self.assertEqual(s.check(), z3.sat)

        # 断言 Solver 模型中 expand_res 的第二个参数的第一个元素等于 b 的第二个维度大小
        assert s.model()[expand_res].arg(1).arg(1) == b.shape[1]

    def test_getitem_tensor(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，输入 x 的类型为 TensorType([4, 4])
            def forward(self, x: TensorType([4, 4])):
                # 对输入 x 进行切片操作
                getitem = x[
                    (None, None, slice(None, None, None), slice(None, None, None))
                ]
                # 返回切片后的结果
                return getitem

        # 创建 BasicBlock 类的实例 B
        B = BasicBlock()
        # 调用其 forward 方法并传入一个随机生成的 4x4 的张量，得到结果 b
        b = B.forward(torch.rand(4, 4))

        # 对 BasicBlock 类进行符号跟踪，得到一个 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        # 对符号图中的所有约束条件进行转换，counter 设为 0
        transformed = transform_all_constraints(symbolic_traced, counter=0)

        # 创建一个 z3 Solver 对象
        s = z3.Solver()
        # 将转换后的约束条件添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 返回 z3 的满足状态
        self.assertEqual(s.check(), z3.sat)
        # 创建一个 z3 常量 get_item_res，表示切片结果的张量类型
        get_item_res = z3.Const(2, tensor_type)
        # 断言 Solver 模型中 get_item_res 的各个参数的第一个元素分别等于 b 的对应维度大小
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]

        # 修改输入的类型注释为 TensorType([Dyn, 4])
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, 4])

        # 重新转换所有约束条件
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个新的 z3 Solver 对象
        s = z3.Solver()
        # 将重新转换后的约束条件添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 返回 z3 的满足状态
        self.assertEqual(s.check(), z3.sat)
        # 断言 Solver 模型中 get_item_res 的第三个参数的第一个元素等于 0
        assert s.model()[get_item_res].arg(2).arg(0) == 0
    def test_getitem_tensor2(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 重写 forward 方法，接受一个形状为 [4, 4] 的张量 x
            def forward(self, x: TensorType([4, 4])):
                # 使用索引操作获取 x 的子张量，索引为 (None, None)
                getitem = x[(None, None)]
                return getitem

        # 创建 BasicBlock 实例 B
        B = BasicBlock()
        # 调用 B 的 forward 方法，传入一个随机生成的 4x4 的张量 b
        b = B.forward(torch.rand(4, 4))

        # 对 BasicBlock 进行符号跟踪，生成 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        # 对所有约束条件进行转换，返回转换后的对象
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将转换后的约束条件添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 的结果为可满足状态
        self.assertEqual(s.check(), z3.sat)

        # 创建一个 Z3 常量，表示获取的第二个元素的结果
        get_item_res = z3.Const(2, tensor_type)
        # 断言获取的模型中的结果与张量 b 的形状第 0 维相等
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        # 断言获取的模型中的结果与张量 b 的形状第 1 维相等
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        # 断言获取的模型中的结果与张量 b 的形状第 2 维相等
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        # 断言获取的模型中的结果与张量 b 的形状第 3 维相等
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]

    def test_getitem_tensor_3(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 重写 forward 方法，接受一个形状为 [4, 4] 的张量 x
            def forward(self, x: TensorType([4, 4])):
                # 使用索引操作获取 x 的子张量，索引为 (None, slice(None, None, None), None, slice(None, None, None))
                getitem = x[
                    (None, slice(None, None, None), None, slice(None, None, None))
                ]
                return getitem

        # 创建 BasicBlock 实例 B
        B = BasicBlock()
        # 调用 B 的 forward 方法，传入一个随机生成的 4x4 的张量 b
        b = B.forward(torch.rand(4, 4))

        # 对 BasicBlock 进行符号跟踪，生成 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(B)
        # 对所有约束条件进行转换，返回转换后的对象
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将转换后的约束条件添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 的结果为可满足状态
        self.assertEqual(s.check(), z3.sat)

        # 创建一个 Z3 常量，表示获取的第二个元素的结果
        get_item_res = z3.Const(2, tensor_type)
        # 断言获取的模型中的结果与张量 b 的形状第 0 维相等
        assert s.model()[get_item_res].arg(0).arg(1) == b.shape[0]
        # 断言获取的模型中的结果与张量 b 的形状第 1 维相等
        assert s.model()[get_item_res].arg(1).arg(1) == b.shape[1]
        # 断言获取的模型中的结果与张量 b 的形状第 2 维相等
        assert s.model()[get_item_res].arg(2).arg(1) == b.shape[2]
        # 断言获取的模型中的结果与张量 b 的形状第 3 维相等
        assert s.model()[get_item_res].arg(3).arg(1) == b.shape[3]
    # 定义一个测试函数，用于测试 LayerNorm 操作
    def test_layer_norm(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建 LayerNorm 层对象
                self.l = torch.nn.LayerNorm((1024,))

            # 前向传播函数
            def forward(self, x: Dyn):
                return self.l(x)

        # 创建一个 RewritingTracer 对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 创建 GraphModule 对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对所有约束进行转换
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束
        s.add(transformed)
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 将输出设置为大小为 1 的张量，应导致输入的迁移
        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # 输入形状等于输出形状
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))

        # 将注释更改为错误的形状
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([10, 10])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        self.assertEqual(s.check(), z3.unsat)

        # 修正注释
        for n in graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([10, 1024])

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        transformed = transform_all_constraints(traced, counter=0)
        s = z3.Solver()
        s.add(transformed)
        s.check()
        b = BasicBlock().forward(torch.rand(10, 1024)).shape
        self.assertEqual(s.model()[output].arg(0).arg(1), b[0])
        self.assertEqual(s.model()[output].arg(1).arg(1), b[1)

    # 定义一个测试函数，用于测试 LayerNorm 的函数式实现
    def test_layer_norm_functional(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类
        class BasicBlock(torch.nn.Module):
            # 前向传播函数
            def forward(self, x: Dyn):
                return torch.nn.functional.layer_norm(x, (1024,))

        # 创建一个 RewritingTracer 对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 创建 GraphModule 对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对所有约束进行转换
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束
        s.add(transformed)
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 将输出设置为大小为 1 的张量，应导致输入的迁移
        b = BasicBlock().forward(torch.rand(1024))
        input = z3.Const(1, tensor_type)
        output = z3.Const(2, tensor_type)
        s.add(output == tensor_type.tensor1(D(1, 1024)))
        s.check()
        self.assertEqual(s.model()[input], s.model()[output])
        # 输入形状等于输出形状
        self.assertEqual(b.shape[0], s.model()[input].arg(0).arg(1))
    def test_ne_int_long_type_as(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义前向传播方法，接受两个参数 x 和 y，类型为 TensorType([Dyn, Dyn])
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn, Dyn])):
                # 计算 x 和 y 的不等式，结果转换为整数类型
                ne_int = torch.ne(x, y).int()
                # 将 ne_int 的类型转换为 y 的类型
                type_as = ne_int.type_as(y)
                # 将 type_as 转换为长整型
                long = type_as.long()
                # 返回 long
                return long

        # 对 BasicBlock 进行符号化追踪，得到一个 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(BasicBlock())
        # 对追踪得到的图进行约束变换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将变换后的约束添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 的求解结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 将一个参数迁移到完全静态形状，以便进行比较
        # 创建两个输入变量，类型为 tensor_type
        input = z3.Const(1, tensor_type)
        input_2 = z3.Const(2, tensor_type)
        # 创建两个整数变量 s1 和 s2
        s1, s2 = z3.Ints("s1 s2")
        # 创建一个输出变量 output_long，类型为 tensor_type
        output_long = z3.Const(8, tensor_type)
        # 向 Solver 添加两个约束条件
        s.add(input == tensor_type.tensor2(D(1, 2), D(1, 4)))
        s.add(input_2 == tensor_type.tensor2(D(1, s1), D(1, s2)))
        # 断言 Solver 的求解结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)
        # 调用 BasicBlock 的 forward 方法，传入随机生成的两个 2x4 的张量，并获取其形状
        actual_shape = BasicBlock().forward(torch.rand(2, 4), torch.rand(2, 4)).shape
        # 断言 Solver 模型中 output_long 的第一个参数的第二个参数等于 actual_shape 的第一个元素
        self.assertEqual(s.model()[output_long].arg(0).arg(1), actual_shape[0])
        # 断言 Solver 模型中 output_long 的第二个参数的第二个参数等于 actual_shape 的第二个元素
        self.assertEqual(s.model()[output_long].arg(1).arg(1), actual_shape[1])

    def test_ne(self):
        # 创建两个整数变量 s1 和 s2
        s1, s2 = z3.Ints("s1 s2")
        # 创建两个整数变量 s11 和 s22
        s11, s22 = z3.Ints("s11 s22")
        # 创建两个动态维度对象 d1 和 d2
        d1, d2 = D(s11, s1), D(0, s2)

        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义前向传播方法，接受两个参数 x 和 y，类型为 Dyn
            def forward(self, x: Dyn, y: Dyn):
                # 返回 x 和 y 的不等式结果
                return torch.ne(x, y)

        # 创建一个重写跟踪器对象
        ast_rewriter = RewritingTracer()
        # 使用重写跟踪器对 BasicBlock 进行跟踪，并获取跟踪结果的图形
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块对象，将跟踪结果的根节点、图形和字符串 "gm" 传递给它
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对所有约束进行变换，并将变换后的结果传递给 transformed
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将变换后的约束添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 的求解结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 修改注解
        # 遍历图中的所有节点
        for n in graph.nodes:
            # 如果节点的名称为 "x"
            if n.name == "x":
                # 将节点的类型更改为 TensorType([1, 2])
                n.type = TensorType([1, 2])
            # 如果节点的名称为 "y"
            if n.name == "y":
                # 将节点的类型更改为 TensorType([2, Dyn])
                n.type = TensorType([2, Dyn])

        # 再次对所有约束进行变换，并将变换后的结果传递给 transformed
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将变换后的约束添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 的求解结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 强制第二个维度为 Dyn
        # 输出应仍为 TensorType([2, 2])
        # 创建一个输入变量 input，类型为 tensor_type
        input = z3.Const(2, tensor_type)
        # 向 Solver 添加一个约束条件
        s.add(input == tensor_type.tensor2(d1, d2))
        # 断言 Solver 的求解结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)
        # 调用 BasicBlock 的 forward 方法，传入随机生成的两个 1x2 和 2x1 的张量，并将结果赋给 B
        B = BasicBlock().forward(torch.rand(1, 2), torch.rand(2, 1))
        # 创建一个输出变量 output，类型为 tensor_type
        output = z3.Const(3, tensor_type)
        # 断言 Solver 模型中 output 的第一个参数的第二个参数等于 B 的形状的第一个元素
        self.assertEqual(s.model()[output].arg(0).arg(1), B.shape[0])
        # 断言 Solver 模型中 output 的第二个参数的第二个参数等于 B 的形状的第一个元素
        self.assertEqual(s.model()[output].arg(1).arg(1), B.shape[0])
    def test_cumsum(self):
        # 定义一个简单的神经网络模块
        class BasicBlock(torch.nn.Module):
            # 定义模块的前向传播方法，接受一个类型为 [Dyn, 4, 3] 的张量作为输入
            def forward(self, x: TensorType([Dyn, 4, 3])):
                # 对输入张量在第三个维度上进行累加求和操作
                t = torch.cumsum(x, 3)
                return t

        # 使用元符号跟踪技术创建图模块
        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对创建的约束条件进行变换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 求解器对象
        s = z3.Solver()
        # 向求解器中添加变换后的约束条件
        s.add(transformed)

        # 断言求解器的结果为不可满足，因为索引对此注释无效
        self.assertEqual(s.check(), z3.unsat)

        # 将占位符的类型注释修改为 Dyn，预期结果应为可满足
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # 重新变换约束条件
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建新的 Z3 求解器对象
        s = z3.Solver()
        # 向求解器中添加变换后的约束条件
        s.add(transformed)
        # 断言求解器的结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 将占位符的类型注释修改为正确的张量大小
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([1, 2, 3, 4])

        # 验证输入是否等于输出
        B = BasicBlock().forward(torch.rand(1, 2, 3, 4))
        res_shape = B.shape
        # 再次变换约束条件
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建新的 Z3 求解器对象
        s = z3.Solver()
        # 向求解器中添加变换后的约束条件
        s.add(transformed)
        # 断言求解器的结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 确认输出与预期张量匹配
        result = z3.Const(2, tensor_type)
        self.assertEqual(s.model()[result].arg(0).arg(1), res_shape[0])
        self.assertEqual(s.model()[result].arg(1).arg(1), res_shape[1])
        self.assertEqual(s.model()[result].arg(2).arg(1), res_shape[2])
        self.assertEqual(s.model()[result].arg(3).arg(1), res_shape[3])

        # 确认输出不是 Dyn
        self.assertNotEqual(s.model()[result].arg(0).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(1).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(2).arg(0).as_long(), 0)
        self.assertNotEqual(s.model()[result].arg(3).arg(0).as_long(), 0)
    def test_cumsum_kwargs(self):
        # 定义一个名为 BasicBlock 的内嵌类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 实现 forward 方法，接受类型为 TensorType([Dyn, 4, 3]) 的输入张量 x
            def forward(self, x: TensorType([Dyn, 4, 3])):
                # 对输入张量 x 沿着第三个维度进行累加求和
                t = torch.cumsum(x, dim=3)
                return t

        # 对 BasicBlock 进行符号跟踪，得到 symbolic_traced，其类型为 torch.fx.GraphModule
        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对符号跟踪结果进行约束转换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束到求解器中
        s.add(transformed)

        # 断言求解结果应该为 unsat，因为索引对此注释无效
        self.assertEqual(s.check(), z3.unsat)

        # 将注释修改为 Dyn，应该得到 sat 的结果
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # 再次进行约束转换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 重新创建 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束到求解器中
        s.add(transformed)
        # 断言求解结果应该为 sat
        self.assertEqual(s.check(), z3.sat)

    def test_arange(self):
        # 定义一个名为 BasicBlock 的内嵌类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 实现 forward 方法，接受类型为 TensorType([2, 4]) 的输入张量 x
            def forward(self, x: TensorType([2, 4])):
                # 获取输入张量 x 的尺寸
                size = x.size()
                # 获取尺寸的最后一个元素
                getitem = size[-1]
                # 使用 torch.arange 生成一个从 0 到 getitem-1 的张量
                arange = torch.arange(getitem)
                return arange

        # 用随机生成的输入数据对 BasicBlock 进行前向传播，得到 B
        B = BasicBlock().forward(torch.rand(2, 4))

        # 对 BasicBlock 进行符号跟踪，得到 symbolic_traced，其类型为 torch.fx.GraphModule
        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对符号跟踪结果进行约束转换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束到求解器中
        s.add(transformed)
        # 断言求解结果应该为 sat
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个名为 arange_result 的常数，其类型为 tensor_type，值为 5
        arange_result = z3.Const(5, tensor_type)
        # 断言求解器模型中 arange_result 的第一个和第二个参数不为 0 和 B.size()[0]
        self.assertNotEqual(s.model()[arange_result].arg(0).arg(0).as_long(), 0)
        self.assertEqual(s.model()[arange_result].arg(0).arg(1).as_long(), B.size()[0])

        # 将注释修改为 Dyn。这将会迁移到一个任意类型
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # 再次进行约束转换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 重新创建 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束到求解器中
        s.add(transformed)
        # 断言求解结果应该为 sat
        self.assertEqual(s.check(), z3.sat)

        # 将注释修改为 TensorType([Dyn, Dyn, Dyn, Dyn])
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])

        # 再次进行约束转换
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 重新创建 Z3 Solver 对象
        s = z3.Solver()
        # 添加转换后的约束到求解器中
        s.add(transformed)
        # 断言求解结果应该为 sat
        self.assertEqual(s.check(), z3.sat)
    # 定义一个测试方法，用于测试标量加法的功能
    def test_scalar_add(self):
        # 定义一个继承自torch.nn.Module的基本模块类BasicBlock
        class BasicBlock(torch.nn.Module):
            # 实现模块的前向传播方法，接受一个大小为[2, 4]的张量作为输入参数x
            def forward(self, x: TensorType([2, 4])):
                # 获取输入张量x的大小
                size = x.size()
                # 获取张量大小的最后一个维度值
                getitem = size[-1]
                # 创建一个从0到getitem-1的整数张量
                arange = torch.arange(getitem)
                # 将arange中的每个元素加1，并返回结果
                add = arange + 1
                return add

        # 对BasicBlock模块进行符号化跟踪，返回torch.fx.GraphModule对象
        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对所有约束条件进行变换处理
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束条件到求解器中
        s.add(transformed)
        # 断言求解器的求解结果为满足（satisfiable）
        self.assertEqual(s.check(), z3.sat)

        # 创建一个值为5的整数常量，表示arange张量的预期结果
        arange_result = z3.Const(5, tensor_type)
        # 创建一个值为6的整数常量，表示add张量的预期结果
        add_result = z3.Const(6, tensor_type)
        # 断言求解器模型中arange_result和add_result的值相等
        self.assertEqual(s.model()[arange_result], s.model()[add_result])

    # 定义一个测试方法，用于测试常规加法功能
    def test_regular_add_2(self):
        # 定义一个继承自torch.nn.Module的基本模块类BasicBlock
        class BasicBlock(torch.nn.Module):
            # 实现模块的前向传播方法，接受一个大小为[2, 4]的张量作为输入参数x
            def forward(self, x: TensorType([2, 4])):
                # 将输入张量x转换为某种格式（to方法具体实现未给出）
                to = x.to()
                # 获取转换后张量的大小
                size = to.size()
                # 获取张量大小的最后一个维度值
                getitem = size[-1]
                # 将getitem加1，并返回结果
                add = getitem + 1
                return add

        # 创建BasicBlock类的实例，并对其进行前向传播计算，传入一个随机生成的[2, 4]张量作为输入
        b = BasicBlock().forward(torch.rand(2, 4))

        # 对BasicBlock模块进行符号化跟踪，返回torch.fx.GraphModule对象
        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对所有约束条件进行变换处理
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束条件到求解器中
        s.add(transformed)
        # 断言求解器的求解结果为满足（satisfiable）
        self.assertEqual(s.check(), z3.sat)
        # 创建一个值为5的整数常量，表示预期结果
        res = z3.Int(5)
        # 断言求解器模型中res的值等于前向传播计算得到的b的值
        self.assertEqual(s.model()[res], b)

    # 定义一个测试方法，用于测试常规加法功能（另一种形式）
    def test_regular_add_3(self):
        # 定义一个继承自torch.nn.Module的基本模块类BasicBlock
        class BasicBlock(torch.nn.Module):
            # 实现模块的前向传播方法，接受一个大小为[2, 4]的张量作为输入参数x
            def forward(self, x: TensorType([2, 4])):
                # 将输入张量x转换为某种格式（to方法具体实现未给出）
                to = x.to()
                # 获取转换后张量的大小
                size = to.size()
                # 获取张量大小的最后一个维度值
                getitem = size[-1]
                # 将1加上getitem，并返回结果
                add = 1 + getitem
                return add

        # 创建BasicBlock类的实例，并对其进行前向传播计算，传入一个随机生成的[2, 4]张量作为输入
        b = BasicBlock().forward(torch.rand(2, 4))

        # 对BasicBlock模块进行符号化跟踪，返回torch.fx.GraphModule对象
        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对所有约束条件进行变换处理
        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个z3求解器对象
        s = z3.Solver()
        # 添加变换后的约束条件到求解器中
        s.add(transformed)
        # 断言求解器的求解结果为满足（satisfiable）
        self.assertEqual(s.check(), z3.sat)
        # 创建一个值为5的整数常量，表示预期结果
        res = z3.Int(5)
        # 断言求解器模型中res的值等于前向传播计算得到的b的值
        self.assertEqual(s.model()[res], b)
    def test_embedding(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个嵌入层，256008 是嵌入的大小，1024 是嵌入向量的维度，padding_idx=1 表示填充索引为1
                self.embedding = torch.nn.Embedding(256008, 1024, padding_idx=1)

            # 前向传播函数，接受一个形状为 [2, 4] 的张量 x
            def forward(self, x: TensorType([2, 4])):
                return self.embedding(x)

        # 创建 BasicBlock 实例并执行前向传播，获取结果的大小
        B = BasicBlock().forward(torch.ones([2, 4], dtype=torch.long)).size()
        
        # 创建一个重写跟踪器的实例
        ast_rewriter = RewritingTracer()
        
        # 对 BasicBlock 进行图形追踪
        graph = ast_rewriter.trace(BasicBlock())
        
        # 创建图模块，使用重写跟踪器的根节点和图形
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的模块应用所有的转换约束
        transformed = transform_all_constraints(traced, counter=0)
        
        # 创建一个 Z3 求解器
        s = z3.Solver()
        
        # 向求解器添加转换后的约束
        s.add(transformed)
        
        # 断言求解器的结果为可满足状态
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个名为 embedding_result 的 Z3 常量，用于表示张量类型
        embedding_result = z3.Const(2, tensor_type)

        # 断言求解器模型中的 embedding_result 张量的特定索引值与 B 的对应索引值相等
        assert s.model()[embedding_result].arg(0).arg(1) == B[0]
        assert s.model()[embedding_result].arg(1).arg(1) == B[1]
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

        # 修改类型。这个修改仍然应该是可满足的
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn])

        # 重新应用所有转换约束
        transformed = transform_all_constraints(traced, counter=0)
        
        # 重新创建 Z3 求解器
        s = z3.Solver()
        
        # 向求解器添加重新转换后的约束
        s.add(transformed)
        
        # 断言求解器的结果为可满足状态
        self.assertEqual(s.check(), z3.sat)
        
        # 断言求解器模型中的 embedding_result 张量的特定索引值与 B 的对应索引值相等
        assert s.model()[embedding_result].arg(0).arg(0) == 0
        assert s.model()[embedding_result].arg(1).arg(0) == 0
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]

        # 将类型修改为 Dyn。这里我们将得到一个任意的迁移
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                n.type = Dyn

        # 重新应用所有转换约束
        transformed = transform_all_constraints(traced, counter=0)
        
        # 重新创建 Z3 求解器
        s = z3.Solver()
        
        # 向求解器添加重新转换后的约束
        s.add(transformed)
        
        # 断言求解器的结果为可满足状态
        self.assertEqual(s.check(), z3.sat)

    def test_embedding_2(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 前向传播函数，接受形状为 [2, 4] 和 [Dyn, 1024] 的两个张量 x 和 y
            def forward(self, x: TensorType([2, 4]), y: TensorType([Dyn, 1024])):
                # 使用 torch.nn.functional.embedding 函数进行嵌入操作
                return torch.nn.functional.embedding(x, y)

        # 创建 BasicBlock 实例并执行前向传播，获取结果的大小
        B = (
            BasicBlock()
            .forward(torch.ones([2, 4], dtype=torch.long), torch.rand(256008, 1024))
            .size()
        )
        
        # 创建一个重写跟踪器的实例
        ast_rewriter = RewritingTracer()
        
        # 对 BasicBlock 进行图形追踪
        graph = ast_rewriter.trace(BasicBlock())
        
        # 创建图模块，使用重写跟踪器的根节点和图形
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        
        # 对追踪后的模块应用所有的转换约束
        transformed = transform_all_constraints(traced, counter=0)
        
        # 创建一个 Z3 求解器
        s = z3.Solver()
        
        # 向求解器添加转换后的约束
        s.add(transformed)
        
        # 断言求解器的结果为可满足状态
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个名为 embedding_result 的 Z3 常量，用于表示张量类型
        embedding_result = z3.Const(5, tensor_type)

        # 断言求解器模型中的 embedding_result 张量的特定索引值与 B 的对应索引值相等
        assert s.model()[embedding_result].arg(0).arg(1) == B[0]
        assert s.model()[embedding_result].arg(1).arg(1) == B[1]
        assert s.model()[embedding_result].arg(2).arg(1) == B[2]
    def test_size_two_args(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 实现 Module 类的 forward 方法，接受一个类型为 TensorType([Dyn, 2, Dyn]) 的参数 x
            def forward(self, x: TensorType([Dyn, 2, Dyn])):
                # 获取 x 的最后一个维度的大小
                size = x.size(-1)
                return size

        # 创建一个 RewritingTracer 实例
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用追踪生成的图形表示创建 GraphModule 实例
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪生成的约束进行转换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例
        s = z3.Solver()
        # 将转换后的约束添加到 Solver 中
        s.add(transformed)
        # 断言 Solver 的结果为 z3.sat（可满足）
        self.assertEqual(s.check(), z3.sat)

        # 定义多个 Z3 Int 类型的变量
        d1, d2 = z3.Int(39), z3.Int(2)
        d4, d5 = z3.Int("input_d1"), z3.Int("input_d2")

        # 添加约束条件 d1 不等于 0
        s.add(d1 != 0)

        # 断言 Solver 的结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个名为 input 的常量，类型为 tensor_type
        input = z3.Const(1, tensor_type)
        # 添加约束条件 input 等于指定的 tensor_type.tensor3 函数的返回值
        s.add(input == tensor_type.tensor3(D(3, 39), D(1, 2), D(d4, d5)))

        # 断言 Solver 的结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 断言 Solver 模型中 d5 和 d2 的值相等
        self.assertEqual(s.model()[d5], s.model()[d2])
        # 断言 Solver 模型中 d1 和 d4 的值相等
        self.assertEqual(s.model()[d1], s.model()[d4])

    def test_size_getitem(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 实现 Module 类的 forward 方法，接受一个类型为 Dyn 的参数 x
            def forward(self, x: Dyn):
                # 获取 x 的大小
                size = x.size()
                # 获取 size 的倒数第一个元素
                getitem = size[-1]
                return getitem

        # 创建一个 RewritingTracer 实例
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用追踪生成的图形表示创建 GraphModule 实例
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪生成的约束进行转换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例
        s = z3.Solver()
        # 将转换后的约束添加到 Solver 中
        s.add(transformed)

        # 断言 Solver 的结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 强制输入的大小为 4
        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        # 创建一个名为 input 的常量，类型为 tensor_type
        input = z3.Const(1, tensor_type)
        # 添加约束条件 input 等于指定的 tensor_type.tensor4 函数的返回值
        s.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # 断言 Solver 的结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 检查模型中 s1 和 s2 的值是否相等
        self.assertEqual(s.model()[s1], s.model()[s2])

        # 创建一个新的 BasicBlock 内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 实现 Module 类的 forward 方法，接受一个类型为 Dyn 的参数 x
            def forward(self, x: Dyn):
                # 获取 x 的大小
                size = x.size()
                # 获取 size 的倒数第十个元素
                getitem = size[-10]
                return getitem

        # 创建一个 RewritingTracer 实例
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用追踪生成的图形表示创建 GraphModule 实例
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪生成的约束进行转换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例
        s = z3.Solver()
        # 将转换后的约束添加到 Solver 中
        s.add(transformed)

        # 断言 Solver 的结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 添加约束条件 input 不等于 z3_dyn
        s.add(input != z3_dyn)
        # 断言 Solver 的结果为 z3.unsat（不可满足）
        self.assertEqual(s.check(), z3.unsat)
    def test_view_mul(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个包含 256008 个词汇的嵌入层，每个词汇的维度是 1024，其中 padding 的索引为 1
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([2, 4])):
                # 获取输入张量 x 的大小
                size = x.size()
                # 获取 x 张量的最后一个维度
                getitem = size[-1]
                # 对 x 张量进行重新视图，变为大小为 (-1, getitem) 的张量
                view = x.view(-1, getitem)
                # 使用嵌入层 embed_tokens 对 view 进行嵌入操作
                embed_tokens = self.embed_tokens(view)
                # 将嵌入结果乘以 32.0
                mul = embed_tokens * 32.0
                # 返回乘法结果
                return mul

        # 创建一个 RewritingTracer 的实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用追踪得到的图形创建 GraphModule，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的模块进行变换，将其所有约束条件进行转换，初始计数器为 0
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例 s
        s = z3.Solver()
        # 添加变换后的约束条件到 Solver 中
        s.add(transformed)
        # 断言 Solver 的检查结果为 z3.sat，即可满足
        self.assertEqual(s.check(), z3.sat)
        # 创建一个名为 embedding_result 的常量，值为 6，类型为 tensor_type
        embedding_result = z3.Const(6, tensor_type)

        # 输出指示，视图的输出将是：tensor3(dim(0, 0), dim(1, 4), dim(1, 1024))
        # 这是由于重塑约束造成的。可以解除此约束，但需要相应调整类型规则，暂时留下
        assert (s.model()[embedding_result].arg(1).arg(1)) == 4
        assert (s.model()[embedding_result].arg(2).arg(1)) == 1024

        # 创建一个名为 mul_result 的常量，值为 13，类型为 tensor_type
        mul_result = z3.Const(13, tensor_type)
        # 断言 Solver 的模型中 mul_result 的值等于 embedding_result 的值
        assert s.model()[mul_result] == s.model()[embedding_result]

    def test_gt(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, 4])):
                # 获取输入张量 x 的大小
                size = x.size()
                # 获取 x 张量的倒数第二个维度
                getitem_1 = size[-1]
                # 检查 getitem_1 是否大于 1，返回布尔值
                gt = getitem_1 > 1
                return gt

        # 创建一个 RewritingTracer 的实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用追踪得到的图形创建 GraphModule，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的模块进行变换，将其所有约束条件进行转换，初始计数器为 0
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例 s
        s = z3.Solver()
        # 添加变换后的约束条件到 Solver 中
        s.add(transformed)
        # 断言 Solver 的检查结果为 z3.sat，即可满足
        self.assertEqual(s.check(), z3.sat)
        # 创建一个名为 res 的布尔型常量，值为 True，索引为 4
        res = z3.Bool(4)
        # 断言 Solver 的模型中 res 的值等于 True
        self.assertEqual(s.model()[res], True)

    def test_view(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([2, 4])):
                # 对输入张量 x 进行重新视图，变为大小为 (-1, 8) 的张量
                view = x.view(-1, 8)
                return view

        # 创建一个 RewritingTracer 的实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用追踪得到的图形创建 GraphModule，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的模块进行变换，将其所有约束条件进行转换，初始计数器为 0
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例 s
        s = z3.Solver()
        # 添加变换后的约束条件到 Solver 中
        s.add(transformed)
        # 断言 Solver 的检查结果为 z3.sat，即可满足
        self.assertEqual(s.check(), z3.sat)
    def test_lt_tensor(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，接受两个参数 x 和 y，类型分别为 TensorType([2, 4]) 和 Dyn
            def forward(self, x: TensorType([2, 4]), y: Dyn):
                # 创建一个 lt 变量，用于保存 x 是否大于 y 的比较结果
                lt = x > y
                # 返回 lt 变量
                return lt

        # 创建一个 RewritingTracer 对象
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪，得到图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用 GraphModule 将 ast_rewriter 的根和图形转换为模块对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的图形进行约束变换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 z3 Solver 对象
        s = z3.Solver()
        # 将 transformed 添加到 Solver 中作为约束条件
        s.add(transformed)
        # 断言 Solver 的检查结果为可满足
        self.assertEqual(s.check(), z3.sat)

    def test_conditional_wrong_assumption(self):
        """
        Test condition after making the wrong assumption about the input
        """

        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，接受一个参数 x，类型为 Dyn
            def forward(self, x: Dyn):
                # 创建一个 gt 变量，用于保存 x 是否大于 1 的比较结果
                gt = x > 1
                # 返回 gt 变量
                return gt

        # 创建一个 RewritingTracer 对象
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪，得到图形表示
        graph = ast_rewriter.trace(BasicBlock())

        # 遍历图中的节点
        for n in graph.nodes:
            # 如果节点 n 的目标是 operator.gt
            if n.target == operator.gt:
                # 将节点赋值给变量 node
                node = n

        # 使用 evaluate_conditional_with_constraints 函数评估带约束条件的条件分支
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        # 断言 positive 的结果为可满足
        self.assertEqual(positive, z3.sat)
        # 断言 negative 的结果为可满足
        self.assertEqual(negative, z3.sat)
    def test_conditional(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns:

        """

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个大小为 256008 的嵌入层，每个嵌入向量大小为 1024，padding_idx=1
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                # 获取输入张量的大小
                size = x.size()
                # 获取张量的最后一个维度大小
                getitem = size[-1]
                # 将输入张量展平成二维，第一维度为-1，第二维度为getitem
                view = x.view(-1, getitem)
                # 对展平后的张量应用嵌入层，embed_tokens 是之前初始化的嵌入层
                embed_tokens = self.embed_tokens(view)
                # 将嵌入向量乘以32.0
                mul = embed_tokens * 32.0
                # 再次获取输入张量的最后一个维度大小
                getitem_1 = size[-1]
                # 检查获取的最后一个维度大小是否大于1
                gt = getitem_1 > 1
                # 返回比较结果
                return gt

        # 创建一个重写跟踪器实例
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行重写跟踪，返回一个图形结构
        graph = ast_rewriter.trace(BasicBlock())

        # 遍历图形中的节点
        for n in graph.nodes:
            # 如果节点 n 的目标操作符是大于号（gt）
            if n.target == operator.gt:
                # 将当前节点设置为目标节点
                node = n

        # 使用给定的约束评估条件表达式，返回正负解
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        # 断言正解应为可满足（sat）
        self.assertEqual(positive, z3.sat)
        # 断言负解应为不可满足（unsat）
        self.assertEqual(negative, z3.unsat)

        # 将注释更改为 Dyn
        for n in graph.nodes:
            # 如果节点 n 的操作符是 "placeholder"
            if n.op == "placeholder":
                # 将节点类型更改为 Dyn
                n.type = Dyn

        # 再次使用给定的约束评估条件表达式，返回正负解
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        # 断言正解应为可满足（sat）
        self.assertEqual(positive, z3.sat)
        # 断言负解应为可满足（sat）
        self.assertEqual(negative, z3.sat)

        # 将注释更改为 TensorType[Dyn, Dyn]
        for n in graph.nodes:
            # 如果节点 n 的操作符是 "placeholder"
            if n.op == "placeholder":
                # 将节点类型更改为 TensorType[Dyn, Dyn]
                n.type = TensorType([Dyn, Dyn])

        # 第三次使用给定的约束评估条件表达式，返回正负解
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )

        # 断言正解应为可满足（sat）
        self.assertEqual(positive, z3.sat)
        # 断言负解应为可满足（sat）
        self.assertEqual(negative, z3.sat)
    def test_conditional_2(self):
        """
        This test case is for the HFmodels interface.
        A function takes a node and a graph and considers
        the conditional the node represents and its negation
        and solves each formula with the remaining sets of constraints
        Returns the opposite result of the above testcase

        """
        
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Initialize an embedding layer with vocabulary size 256008, embedding size 1024, and padding index 1
                self.embed_tokens = torch.nn.Embedding(256008, 1024, padding_idx=1)

            def forward(self, x: TensorType([Dyn, 4])):
                # Retrieve the size of input tensor x
                size = x.size()
                # Retrieve the last dimension of the tensor size
                getitem = size[-1]
                # Reshape x into a 2-dimensional tensor with -1 rows and getitem columns
                view = x.view(-1, getitem)
                # Apply the embedding layer to the reshaped tensor view
                embed_tokens = self.embed_tokens(view)
                # Multiply the embedded tokens by 32.0
                mul = embed_tokens * 32.0
                # Retrieve again the last dimension of the input tensor x size
                getitem_1 = size[-1]
                # Check if getitem_1 is less than 1 and return the boolean result
                lt = getitem_1 < 1
                return lt

        # Create an instance of the RewritingTracer class
        ast_rewriter = RewritingTracer()
        # Trace the execution of the BasicBlock module and obtain the computation graph
        graph = ast_rewriter.trace(BasicBlock())

        # Identify the node representing the less-than operation in the computation graph
        for n in graph.nodes:
            if n.target == operator.lt:
                node = n

        # Evaluate the conditional represented by the identified node with constraints
        positive, negative = evaluate_conditional_with_constraints(
            ast_rewriter.root, graph, node
        )
        # Assert that the positive evaluation of the conditional is unsatisfiable
        self.assertEqual(positive, z3.unsat)
        # Assert that the negative evaluation of the conditional is satisfiable
        self.assertEqual(negative, z3.sat)
class ComposeOperationsGradualTypes(unittest.TestCase):
    # 定义测试类 ComposeOperationsGradualTypes，继承自 unittest.TestCase

    def test_masked_fill(self):
        # 定义测试方法 test_masked_fill

        class BasicBlock(torch.nn.Module):
            # 定义内部类 BasicBlock，继承自 torch.nn.Module

            def forward(self, x: TensorType([2, 4])):
                # 定义 forward 方法，接受参数 x，其类型为 TensorType([2, 4])

                size = x.size()
                # 获取 x 的尺寸信息

                getitem = size[-1]
                # 获取 x 的最后一个维度大小

                arange = torch.arange(getitem)
                # 创建一个张量，包含从 0 到 getitem-1 的整数

                view = x.view(-1, getitem)
                # 将 x 重塑为形状 (-1, getitem)

                lt = arange > view
                # 创建一个布尔张量，指示 arange 中的元素是否大于 view 中对应位置的元素

                masked_fill = x.masked_fill_(lt, 0)
                # 使用 lt 对 x 进行掩码填充操作，将满足条件的位置设为 0

                return masked_fill
                # 返回填充后的张量 masked_fill

        B = BasicBlock().forward(torch.rand(2, 4))
        # 创建 BasicBlock 实例，并传入形状为 (2, 4) 的随机张量进行 forward 计算

        # print(B.shape)

        symbolic_traced: torch.fx.GraphModule = meta_symbolic_trace(
            BasicBlock(), meta_args={}
        )
        # 对 BasicBlock 进行符号化跟踪，返回一个 torch.fx.GraphModule 对象

        # print(symbolic_traced)

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 对符号化跟踪后的对象应用变换，将其转换为约束形式

        s = z3.Solver()
        # 创建一个 Z3 Solver 对象

        s.add(transformed)
        # 将变换后的约束添加到 Solver 中

        self.assertEqual(s.check(), z3.sat)
        # 使用 Z3 求解器检查约束是否可满足，并断言其结果为 z3.sat

        masked_fill_res = z3.Const(10, tensor_type)
        # 创建一个名为 masked_fill_res 的常量，类型为 tensor_type，初始值为 10

        self.assertEqual(
            s.model()[masked_fill_res].arg(0).arg(1).as_long(), B.size()[0]
        )
        # 断言求解器模型中 masked_fill_res 的第一个参数的第二个元素的长整型值与 B 的第一个维度大小相等

        self.assertEqual(
            s.model()[masked_fill_res].arg(1).arg(1).as_long(), B.size()[1]
        )
        # 断言求解器模型中 masked_fill_res 的第二个参数的第二个元素的长整型值与 B 的第二个维度大小相等

        # change the annotation to Dyn. This will migrate to an arbitrary type
        for n in symbolic_traced.graph.nodes:
            # 遍历符号化跟踪对象的图中的节点

            if n.op == "placeholder":
                # 如果节点的操作为 "placeholder"

                n.type = Dyn
                # 将节点的类型更改为 Dyn

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 再次对修改后的符号化跟踪对象应用变换

        s = z3.Solver()
        # 创建新的 Z3 Solver 对象

        s.add(transformed)
        # 将新的变换后的约束添加到 Solver 中

        self.assertEqual(s.check(), z3.sat)
        # 使用 Z3 求解器检查约束是否可满足，并断言其结果为 z3.sat

        for n in symbolic_traced.graph.nodes:
            # 再次遍历符号化跟踪对象的图中的节点

            if n.op == "placeholder":
                # 如果节点的操作为 "placeholder"

                n.type = TensorType([Dyn, Dyn, Dyn, Dyn])
                # 将节点的类型更改为 TensorType([Dyn, Dyn, Dyn, Dyn])

        transformed = transform_all_constraints(symbolic_traced, counter=0)
        # 再次对修改后的符号化跟踪对象应用变换

        s = z3.Solver()
        # 创建新的 Z3 Solver 对象

        s.add(transformed)
        # 将新的变换后的约束添加到 Solver 中

        self.assertEqual(s.check(), z3.sat)
        # 使用 Z3 求解器检查约束是否可满足，并断言其结果为 z3.sat

    def test_add_reshape_1(self):
        # 定义测试方法 test_add_reshape_1

        class BasicBlock(torch.nn.Module):
            # 定义内部类 BasicBlock，继承自 torch.nn.Module

            def forward(self, x: Dyn, y: Dyn):
                # 定义 forward 方法，接受参数 x 和 y，它们的类型为 Dyn

                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))
                # 返回将 x 和 y 进行重塑后，再相加的结果

        ast_rewriter = RewritingTracer()
        # 创建重写追踪器对象

        graph = ast_rewriter.trace(BasicBlock())
        # 对 BasicBlock 进行追踪，并获取其图形表示

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建图模块对象，使用追踪器的根节点和图形表示

        transformed = transform_all_constraints(traced, counter=0)
        # 对追踪后的图形模块对象应用变换，将其转换为约束形式

        s = z3.Solver()
        # 创建一个 Z3 Solver 对象

        s.add(transformed)
        # 将变换后的约束添加到 Solver 中

        self.assertEqual(s.check(), z3.sat)
        # 使用 Z3 求解器检查约束是否可满足，并断言其结果为 z3.sat

    def test_add_reshape_2(self):
        # 定义测试方法 test_add_reshape_2

        class BasicBlock(torch.nn.Module):
            # 定义内部类 BasicBlock，继承自 torch.nn.Module

            def forward(self, x: Dyn, y: Dyn):
                # 定义 forward 方法，接受参数 x 和 y，它们的类型为 Dyn

                return torch.add(torch.reshape(x, (-1, 2)), torch.reshape(y, (2, 2, 2)))
                # 返回将 x 和 y 进行重塑后，再相加的结果

        ast_rewriter = RewritingTracer()
        # 创建重写追踪器对象

        graph = ast_rewriter.trace(BasicBlock())
        # 对 BasicBlock 进行追踪，并获取其图形表示

        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 创建图模块对象，使用追踪器的根节点和图形表示

        transformed = transform_all_constraints(traced, counter=0)
        # 对追踪后的图形模块对象应用变换，将其转换为约束形式

        s = z3.Solver()
        # 创建一个 Z3 Solver 对象

        s.add(transformed)
        # 将变换后的约束添加到 Solver 中

        self.assertEqual(s.check(), z3.sat)
        # 使用 Z3 求解器检查约束是否可满足，并断言其结果为 z3.sat
    def test_conv_reshape_add_0(self):
        # 定义一个名为 test_conv_reshape_add_0 的测试方法
        class BasicBlock(torch.nn.Module):
            # 定义 BasicBlock 类，继承自 torch.nn.Module
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                # 构造函数，初始化 BasicBlock 实例
                super().__init__()
                # 创建一个二维卷积层对象 conv1
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn, y: Dyn):
                # 前向传播函数，执行卷积操作后，对结果执行 torch.add 操作
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        # 创建一个 BasicBlock 实例 B，参数为 (2, 2, 2, 3, 2, 2, 2)
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock B 进行追踪生成图形表示
        graph = ast_rewriter.trace(B)
        # 使用 ast_rewriter.root 和生成的图形 graph 创建 GraphModule 实例 traced
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对所有约束条件进行转换
        new_transformed_c = transform_all_constraints(traced)
        # 创建一个 z3 Solver 实例 solver
        solver = z3.Solver()
        # 向 solver 中添加转换后的约束条件 new_transformed_c
        solver.add(new_transformed_c)
        # 断言 solver 的结果为 satisfiable（满足条件）
        self.assertEqual(solver.check(), z3.sat)
    def test_conv_reshape_add_0_2(self):
        # 定义一个继承自torch.nn.Module的基本模块类BasicBlock
        class BasicBlock(torch.nn.Module):
            # 初始化函数，设置卷积层参数
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                # 创建一个2维卷积层对象conv1
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            # 前向传播函数，接受输入x和y，并返回torch.add的结果
            def forward(self, x: Dyn, y: TensorType([4, 1])):
                # 对输入x进行reshape操作，然后通过conv1进行卷积，再与y相加
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        # 创建BasicBlock类的实例B，传入特定的参数
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)

        # 使用torch.rand生成输入数据，调用B的forward方法，并获取结果的尺寸
        res = B.forward(torch.rand(20, 20), torch.rand(1, 2, 4, 8)).size()

        # 创建RewritingTracer实例ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对BasicBlock实例B进行追踪，获取图形表示
        graph = ast_rewriter.trace(B)
        # 创建GraphModule实例traced，传入追踪器的根节点和图形表示，命名为"gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对所有约束条件进行转换，得到新的转换后的约束
        new_transformed_c = transform_all_constraints(traced)

        # 创建z3求解器实例solver
        solver = z3.Solver()
        # 添加新转换后的约束到求解器中
        solver.add(new_transformed_c)
        # 断言求解器的检查结果与z3.sat相等
        self.assertEqual(solver.check(), z3.sat)

        # 创建一个常量conv_result和add_result，分别表示卷积和加法的结果
        conv_result = z3.Const(4, tensor_type)
        add_result = z3.Const(9, tensor_type)
        input_2 = z3.Const(2, tensor_type)

        # 创建四个整数变量s1, s2, s3, s4
        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        # 创建四个约束条件d1, d2, d3, d4，表示变量s1, s2, s3, s4的依赖关系
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        # 添加卷积结果的约束到求解器中
        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        # 进行求解器的检查
        solver.check()

        # 断言求解器模型中s1对应的值与前向传播结果res的第一个元素相等
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        # 添加输入2的约束到求解器中
        solver.add(input_2 == tensor_type.tensor2(D(1, 4), D(1, 1)))
        # 断言求解器的检查结果与z3.sat相等
        self.assertEqual(solver.check(), z3.sat)

        # 添加加法结果的约束到求解器中
        solver.add(add_result == tensor_type.tensor4(d1, d2, d3, d4))
        # 断言求解器的检查结果与z3.sat相等
        self.assertEqual(solver.check(), z3.sat)

        # 由于有广播，第一维可能是任意的，所以只检查后三个维度的值是否匹配前向传播结果res
        assert solver.model()[s1] == res[0]
        assert solver.model()[s2] == res[1]
        assert solver.model()[s3] == res[2]
        assert solver.model()[s4] == res[3]
    def test_conv_reshape_add_0_3(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化函数，设置基本的卷积参数和层次结构
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                # 创建一个 2D 卷积层 conv1，用于处理输入数据
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            # 前向传播函数，接收一个 Dyn 类型的输入 x 和一个形状为 [11, 1] 的张量 y
            def forward(self, x: Dyn, y: TensorType([11, 1])):
                # 对输入 x 进行形状重塑，然后经过卷积层 conv1 处理，再加上张量 y 的结果返回
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        # 创建一个 BasicBlock 类的实例 B，指定各种卷积参数
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 创建一个重写追踪器实例
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，生成计算图
        graph = ast_rewriter.trace(B)
        # 使用追踪生成的计算图和追踪器的根节点创建一个 GraphModule 实例 traced
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪得到的约束进行转换处理
        new_transformed_c = transform_all_constraints(traced)
        # 创建一个 z3 求解器实例
        solver = z3.Solver()
        # 将转换后的约束添加到求解器中
        solver.add(new_transformed_c)
        # 断言求解器的结果为不可满足
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv_reshape_add_1(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化函数，设置基本的卷积参数和层次结构
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                # 创建一个 2D 卷积层 conv1，用于处理输入数据
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            # 前向传播函数，接收一个 Dyn 类型的输入 x 和一个形状为 [1, 2, 10, 20] 的张量 y
            def forward(self, x: Dyn, y: TensorType([1, 2, 10, 20])):
                # 对输入 x 进行形状重塑，然后经过卷积层 conv1 处理，再加上张量 y 的结果返回
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)

        # 创建一个 BasicBlock 类的实例 B，指定各种卷积参数
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 创建一个重写追踪器实例
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，生成计算图
        graph = ast_rewriter.trace(B)
        # 使用追踪生成的计算图和追踪器的根节点创建一个 GraphModule 实例 traced
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪得到的约束进行转换处理
        new_transformed_c = transform_all_constraints(traced)
        # 创建一个 z3 求解器实例
        solver = z3.Solver()
        # 将转换后的约束添加到求解器中
        solver.add(new_transformed_c)
        # 断言求解器的结果为不可满足
        self.assertEqual(solver.check(), z3.unsat)
class GradualTypes(unittest.TestCase):
    # 定义测试类 GradualTypes，继承自 unittest.TestCase

    def test_conv_reshape_unsat(self):
        # 定义测试方法 test_conv_reshape_unsat，用于测试卷积和重塑不可满足情况

        class BasicBlock(torch.nn.Module):
            # 定义内部类 BasicBlock，继承自 torch.nn.Module，用于构建基本的神经网络模块

            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                # 构造函数，初始化 BasicBlock 实例

                super().__init__()
                # 调用父类的构造函数

                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )
                # 创建一个二维卷积层 conv1，参数由传入的 in_planes, out_planes, kernel_size,
                # stride, padding, groups, dilation 确定

            def forward(self, x: Dyn):
                # 定义神经网络模块的前向传播函数 forward，输入 x 是一个动态类型 Dyn

                return self.conv1(torch.reshape(x, (1, 2, 10)))
                # 返回对输入 x 进行 reshape 后，再经过 conv1 进行卷积的结果

        # 创建 BasicBlock 类的实例 B，传入特定的参数
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)

        # 创建一个重写跟踪器 RewritingTracer 的实例 ast_rewriter
        ast_rewriter = RewritingTracer()

        # 使用 ast_rewriter 对实例 B 进行跟踪，生成图形表示
        graph = ast_rewriter.trace(B)

        # 使用 ast_rewriter 的根节点和跟踪生成的图形，创建 GraphModule 实例 traced
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对跟踪生成的约束进行转换，得到新的转换后的约束 new_transformed_c
        new_transformed_c = transform_all_constraints(traced)

        # 创建一个求解器 Solver 的实例 solver
        solver = z3.Solver()

        # 向求解器 solver 添加新转换后的约束 new_transformed_c
        solver.add(new_transformed_c)

        # 断言求解器 solver 的检查结果与 z3.unsat 相等
        self.assertEqual(solver.check(), z3.unsat)
    def test_conv_reshape0(self):
        # 定义一个名为 test_conv_reshape0 的测试方法
        class BasicBlock(torch.nn.Module):
            # 定义 BasicBlock 类，继承自 torch.nn.Module
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                # 构造函数，初始化 BasicBlock 实例
                super().__init__()
                # 创建一个 2D 卷积层，设置各种参数
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn):
                # 前向传播函数，执行卷积操作并返回结果
                return self.conv1(torch.reshape(x, (1, 2, 10, 20)))

        # 创建 BasicBlock 的实例 B，设置参数
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 生成一个随机张量，作为输入，执行前向传播，得到输出的尺寸
        res = B.forward(torch.rand(20, 20)).size()
        
        # 创建一个 AST 重写器
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，得到图形表示
        graph = ast_rewriter.trace(B)
        # 创建一个图形模块，使用重写后的根节点和图形
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对图形模块应用转换函数，返回新的约束
        new_transformed_c = transform_all_constraints(traced)

        # 创建一个 Z3 Solver 实例
        solver = z3.Solver()
        # 添加新转换的约束到求解器中
        solver.add(new_transformed_c)
        # 检查求解器的结果，期望是满足 (sat)
        self.assertEqual(solver.check(), z3.sat)
        
        # 创建一个表示卷积结果的常量
        conv_result = z3.Const(3, tensor_type)

        # 创建多个整数变量
        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        # 创建多个 D 对象，表示映射关系
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        # 将卷积结果与张量类型的 4D 张量比较，并添加到求解器中
        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        # 检查求解器状态
        solver.check()

        # 断言求解器模型中的值与卷积结果的对应维度相匹配
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]

        # 创建另一组整数变量
        s1, s2, s3, s4 = z3.Ints("y1 y2 y3 y4")
        s11, s22, s33, s44 = z3.Ints("y11 y22 y33 y44")
        # 创建另一组 D 对象，表示映射关系
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        # 创建一个输入常量，并添加到求解器中
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(d1, d2, d3, d4))

        # 执行断言，期望求解器状态为满足 (sat)
        # assert solver.check() == sat
        # 添加额外的约束条件
        # solver.add(s11 == 1)
        # solver.add(s22 == 1)
        # solver.add(s33 == 1)
        # solver.add(s44 == 1)
        #
        # print(solver.check())
        # print(solver.model())
    def test_conv_reshape1(self):
        # 定义一个继承自 torch.nn.Module 的 BasicBlock 类，用于构建基本的神经网络块
        class BasicBlock(torch.nn.Module):
            # 初始化函数，定义了基本的卷积层参数
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                # 创建一个二维卷积层对象
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            # 前向传播函数，接受输入 x，并对其进行卷积和形状变换操作
            def forward(self, x: TensorType([20, 20])):
                return self.conv1(torch.reshape(x, (1, -1, 10, 20)))

        # 创建 BasicBlock 类的实例 B，传入参数初始化
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 对随机生成的输入数据进行前向传播，获取输出的尺寸信息
        res = B.forward(torch.rand(20, 20)).size()
        
        # 创建一个重写追踪器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行图追踪，获取追踪到的图形对象
        graph = ast_rewriter.trace(B)
        # 使用追踪到的根节点和图对象创建一个图模块对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪到的图形对象进行转换所有约束的操作
        new_transformed_c = transform_all_constraints(traced)

        # 创建一个求解器对象
        solver = z3.Solver()
        # 添加新转换的约束到求解器中
        solver.add(new_transformed_c)
        # 检查求解器的结果是否满足约束条件
        self.assertEqual(solver.check(), z3.sat)
        
        # 创建一个表示卷积结果的常量对象
        conv_result = z3.Const(3, tensor_type)

        # 定义多个整数变量
        s1, s2, s3, s4 = z3.Ints("x1 x2 x3 x4")
        s11, s22, s33, s44 = z3.Ints("x11 x22 x33 x44")
        # 定义多个 D 类型的对象
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        # 添加卷积结果和张量对象的约束到求解器中
        solver.add(conv_result == tensor_type.tensor4(d1, d2, d3, d4))
        # 再次检查求解器的结果
        solver.check()
        # 断言求解器模型中的变量与前向传播的结果一致
        assert solver.model()[s1].as_long() == res[0]
        assert solver.model()[s2].as_long() == res[1]
        assert solver.model()[s3].as_long() == res[2]
        assert solver.model()[s4].as_long() == res[3]
class TestSingleOperation(unittest.TestCase):
    # 单元测试类，继承自unittest.TestCase，用于测试单个操作

    def test_conv_wrong_example(self):
        # 测试函数，测试卷积操作的错误示例

        class BasicBlock(torch.nn.Module):
            # 定义基本模块类，继承自torch.nn.Module

            def __init__(self):
                # 初始化函数

                super().__init__()
                # 调用父类初始化函数

                # 第一个卷积层
                self.conv1 = torch.nn.Conv2d(
                    in_channels=2,
                    out_channels=2,
                    kernel_size=2,
                    stride=2,
                    padding=2,
                    groups=2,
                    bias=False,
                    dilation=2,
                )

                # 第二个卷积层
                self.conv2 = torch.nn.Conv2d(
                    in_channels=4,
                    out_channels=2,
                    kernel_size=2,
                    stride=2,
                    padding=2,
                    groups=2,
                    bias=False,
                    dilation=2,
                )

                # ReLU 激活函数
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x: Dyn):
                # 前向传播函数

                # 第一个卷积层的计算
                y = self.relu(self.conv1(x))
                # 第二个卷积层的计算
                z = self.relu(self.conv2(x))
                return z

        # 创建重写追踪器对象
        ast_rewriter = RewritingTracer()
        # 对BasicBlock类进行追踪
        graph = ast_rewriter.trace(BasicBlock())
        # 创建图模块对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对所有约束条件进行转换
        transformed = transform_all_constraints(traced)

        # 创建Z3求解器对象
        solver3 = z3.Solver()
        # 添加转换后的约束条件到求解器中
        solver3.add(transformed)
        # 打印求解结果
        print(solver3.check())
        # 断言求解结果为满足状态
        assert solver3.check() == z3.sat

        # 定义整数变量s1, s2, s3, s4
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        # 定义整数变量s11, s22, s33, s44
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        # 定义表达式d1, d2, d3, d4
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )
        # 定义常量x，并赋值为tensor_type的张量4
        x = z3.Const(1, tensor_type)
        # 添加等式约束到求解器中
        solver3.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        # 断言求解结果为满足状态
        assert solver3.check() == z3.sat

        # 添加不等式约束到求解器中
        solver3.add(s22 != 0)
        # 断言求解结果为不满足状态
        assert solver3.check() == z3.unsat
    def test_conv_dyn(self):
        # 定义一组整数变量，每组包含四个变量
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        e1, e2, e3, e4 = z3.Ints("e1 e2 e3 e4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        e11, e22, e33, e44 = z3.Ints("e11 e22 e33 e44")
        # 创建一组差分约束，将每对起始和结束变量进行关联
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                # 初始化一个二维卷积层
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )

            def forward(self, x: Dyn):
                # 返回卷积层的前向传播结果
                return self.conv1(x)

        # 创建一个BasicBlock对象，并对随机输入执行前向传播
        BasicBlock(2, 2, 2, 2, 2, 2, 2).forward(torch.rand(4, 2, 3, 4))

        # 创建一个RewritingTracer对象用于AST重写
        ast_rewriter = RewritingTracer()
        # 对BasicBlock对象进行追踪，并生成图形表示
        graph = ast_rewriter.trace(BasicBlock(2, 2, 2, 2, 2, 2, 2))
        # 使用AST根节点和生成的图形创建一个GraphModule对象
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对所有约束条件进行转换
        transformed = transform_all_constraints(traced)

        # 创建一个新的Z3求解器对象
        solver3 = z3.Solver()
        # 将转换后的约束条件添加到求解器中
        solver3.add(transformed)
        # 断言求解器返回结果为可满足
        assert solver3.check() == z3.sat

        # 创建两个常量x和y，并分别用tensor_type的特定形式进行初始化
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        # 添加新的约束条件到求解器中，分别关联x和y到d1, d2, d3, d4以及b1, b2, b3, b4
        solver3.add(
            x == tensor_type.tensor4(d1, d2, d3, d4),
            y == tensor_type.tensor4(b1, b2, b3, b4),
        )

        # 断言求解器返回结果为可满足
        assert solver3.check() == z3.sat
        # 断言求解器模型中的s1和e1相等
        assert solver3.model()[s1].as_long() == solver3.model()[e1].as_long()
        # 断言求解器模型中的s11和e11相等
        assert solver3.model()[s11].as_long() == solver3.model()[e11].as_long()

        # 添加新的约束条件到求解器中，指定s2不等于2
        solver3.add(s2 != 2)
        # 断言求解器返回结果为可满足
        assert solver3.check() == z3.sat
        # 断言求解器模型中的s22为0
        assert solver3.model()[s22].as_long() == 0

        # 添加新的约束条件到求解器中，指定s22不等于0
        solver3.add(s22 != 0)
        # 断言求解器返回结果为不可满足
        self.assertEqual(solver3.check(), z3.unsat)

        # 创建另一个新的Z3求解器对象
        solver2 = z3.Solver()
        # 将转换后的约束条件添加到求解器中
        solver2.add(transformed)
        # 断言求解器返回结果为可满足
        assert solver2.check() == z3.sat
        # 添加新的约束条件到求解器中，关联x到d1, d2, d3的tensor形式
        solver2.add(x == tensor_type.tensor3(d1, d2, d3))
        # 断言求解器返回结果为不可满足
        self.assertEqual(solver2.check(), z3.unsat)
    def test_add(self):
        # 定义多个整数变量，用于表示符号变量
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        # 定义多个约束关系，将符号变量与具体数值关联起来
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        # 定义一个简单的神经网络模块，实现向前传播
        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn, y: Dyn):
                return torch.add(x, y)

        # 创建一个AST重写跟踪器对象
        ast_rewriter = RewritingTracer()
        # 对BasicBlock类进行跟踪，生成对应的计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 将跟踪得到的计算图封装为图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对跟踪后的计算图进行约束变换处理
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个Z3求解器对象
        s = z3.Solver()
        # 添加约束条件到求解器中
        s.add(transformed)
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 设置一个大小为1的张量
        x = z3.Const(1, tensor_type)
        # 添加张量约束条件
        s.add(x == tensor_type.tensor1(D(1, s11)))
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 设置另一个大小为2的张量
        y = z3.Const(2, tensor_type)
        # 添加张量约束条件
        s.add(y == tensor_type.tensor1(D(1, s22)))
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 添加额外的约束条件 s11 == 1
        s.add(s11 == 1)  # tensor[1]
        # 添加额外的约束条件 s22 == 2
        s.add(s22 == 2)  # tensor[2]
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 定义另一个神经网络模块类BasicBlock2
        class BasicBlock2(torch.nn.Module):
            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                return torch.add(x, y)

        # 重新创建一个AST重写跟踪器对象
        ast_rewriter = RewritingTracer()
        # 对BasicBlock2类进行跟踪，生成对应的计算图
        graph = ast_rewriter.trace(BasicBlock2())
        # 将跟踪得到的计算图封装为图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对跟踪后的计算图进行约束变换处理
        transformed = transform_all_constraints(traced)
        # 创建一个新的Z3求解器对象
        s = z3.Solver()
        # 添加约束条件到新的求解器中
        s.add(transformed)
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 设置一个大小为1的张量
        x = z3.Const(1, tensor_type)
        # 添加张量约束条件
        s.add(x == tensor_type.tensor1(D(1, s11)))
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 设置另一个大小为2的张量
        y = z3.Const(2, tensor_type)
        # 添加张量约束条件
        s.add(y == tensor_type.tensor1(D(1, s22)))
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 添加额外的约束条件 s11 == 4
        s.add(s11 == 4)  # tensor[4]
        # 添加额外的约束条件 s22 == 5
        s.add(s22 == 5)  # tensor[5]
        # 断言求解结果为不可满足
        self.assertEqual(s.check(), z3.unsat)

        # 定义另一个神经网络模块类BasicBlock3
        class BasicBlock3(torch.nn.Module):
            def forward(self, x: TensorType((Dyn,)), y: Dyn):
                return torch.add(x, y)

        # 重新创建一个AST重写跟踪器对象
        ast_rewriter = RewritingTracer()
        # 对BasicBlock3类进行跟踪，生成对应的计算图
        graph = ast_rewriter.trace(BasicBlock3())
        # 将跟踪得到的计算图封装为图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对跟踪后的计算图进行约束变换处理
        transformed = transform_all_constraints(traced)
        # 创建一个新的Z3求解器对象
        s = z3.Solver()
        # 添加约束条件到新的求解器中
        s.add(transformed)
        # 设置一个大小为1的张量
        x = z3.Const(1, tensor_type)
        # 添加张量约束条件
        s.add(x == tensor_type.tensor2(d1, d2))
        # 断言求解结果为不可满足
        self.assertEqual(s.check(), z3.unsat)
    def test_add_padding(self):
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType((Dyn,)), y: TensorType((Dyn, Dyn))):
                # 定义神经网络模块的前向传播函数，接收两个张量参数并返回它们的和
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        # 创建 AST 重写跟踪器对象
        graph = ast_rewriter.trace(BasicBlock())
        # 对 BasicBlock 类进行 AST 跟踪，生成计算图
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 将 AST 跟踪器的根节点和计算图封装成图模块对象

        transformed = transform_all_constraints(traced, counter=0)
        # 对跟踪到的计算图应用约束变换，counter 用于记录变换数量

        s = z3.Solver()
        # 创建 Z3 求解器对象

        s.add(transformed)
        # 将约束添加到求解器中

        self.assertEqual(s.check(), z3.sat)
        # 断言求解器返回的结果为 satisfiable

        x = z3.Const(1, tensor_type)
        # 创建一个 Z3 常量 x，类型为 tensor_type

        s.add(x == tensor_type.tensor1(D(1, s1)))
        # 添加 x 的约束条件，限制其为 tensor_type 的第一种张量形式

        self.assertEqual(s.check(), z3.sat)
        # 断言求解器返回的结果为 satisfiable

        # print(s.model())
        # 打印求解器的模型

    def test_add_padding_2(self):
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn, Dyn]), y: TensorType([Dyn])):
                # 定义神经网络模块的前向传播函数，接收两个张量参数并返回它们的和
                return torch.add(x, y)

        ast_rewriter = RewritingTracer()
        # 创建 AST 重写跟踪器对象
        graph = ast_rewriter.trace(BasicBlock())
        # 对 BasicBlock 类进行 AST 跟踪，生成计算图
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 将 AST 跟踪器的根节点和计算图封装成图模块对象

        transformed = transform_all_constraints(traced, counter=0)
        # 对跟踪到的计算图应用约束变换，counter 用于记录变换数量

        s = z3.Solver()
        # 创建 Z3 求解器对象

        s.add(transformed)
        # 将约束添加到求解器中

        self.assertEqual(s.check(), z3.sat)
        # 断言求解器返回的结果为 satisfiable

        # print(s.model())
        # 打印求解器的模型

        x = z3.Const(1, tensor_type)
        # 创建一个 Z3 常量 x，类型为 tensor_type

        s.add(x == tensor_type.tensor2(D(1, s1), D(1, s2)))
        # 添加 x 的约束条件，限制其为 tensor_type 的第二种张量形式

        self.assertEqual(s.check(), z3.sat)
        # 断言求解器返回的结果为 satisfiable

        y = z3.Const(2, tensor_type)
        # 创建一个 Z3 常量 y，类型为 tensor_type

        s.add(y == tensor_type.tensor1(D(0, s3)))
        # 添加 y 的约束条件，限制其为 tensor_type 的第一种张量形式

        self.assertEqual(s.check(), z3.sat)
        # 断言求解器返回的结果为 satisfiable

        add_result = z3.Const(3, tensor_type)
        broadcast_res1, broadcast_res2 = z3.Const(4, tensor_type), z3.Const(
            5, tensor_type
        )

        # print(s.model())
        # 打印求解器的模型

        assert s.model()[broadcast_res1].decl() == tensor_type.tensor2
        # 断言模型中 broadcast_res1 的声明为 tensor_type 的第二种张量形式
        assert s.model()[broadcast_res2].decl() == tensor_type.tensor2
        # 断言模型中 broadcast_res2 的声明为 tensor_type 的第二种张量形式
        assert s.model()[add_result].decl() == tensor_type.tensor2
        # 断言模型中 add_result 的声明为 tensor_type 的第二种张量形式
        assert s.model()[y].decl() == tensor_type.tensor1
        # 断言模型中 y 的声明为 tensor_type 的第一种张量形式

        # print(s.model())
        # 打印求解器的模型

        # prevent broadcasting for that dimension
        # 防止在该维度上进行广播
        s.add(s2 > 1)
        # 添加限制条件：s2 的值大于 1

        assert s.check()
        # 断言求解器可以找到满足所有条件的模型

        # the second dimension of the result is a number, not Dyn.
        # however if the first input dimension had been 1, we would
        # have had dyn in the result, as seen in the next test case
        # 结果的第二个维度是一个数字，而不是 Dyn。
        # 但是，如果第一个输入维度为 1，则结果中会有 dyn，如下一个测试案例所示
        assert s.model()[add_result].arg(1).arg(0).as_long() != 0
        # 断言结果模型中 add_result 的第二个维度的第一个元素不为 0
    def test_add_padding_3(self):
        # 定义四个整数变量 s1, s2, s3, s4
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        # 定义一个继承自torch.nn.Module的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 前向传播函数，接受一个类型为[TensorType([Dyn, 1])]的参数 x 和一个类型为[TensorType([Dyn])]的参数 y
            def forward(self, x: TensorType([Dyn, 1]), y: TensorType([Dyn])):
                # 返回 x 和 y 的元素级加法结果
                return torch.add(x, y)

        # 创建一个重写追踪器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块，将追踪器的根节点和计算图作为参数，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的约束条件进行转换处理，计数器初始值为 0
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 向 Solver 对象添加转换后的约束条件
        s.add(transformed)
        # 断言 Solver 对约束条件的求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 创建一个类型为 tensor_type 的常数对象 x 和 y
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        # 向 Solver 对象添加附加约束条件
        s.add(s2 != 0)
        s.add(x == tensor_type.tensor2(D(0, s1), D(s2, 1)))
        s.add(y == tensor_type.tensor1(D(0, s3)))

        # 断言 Solver 对约束条件的求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 打印模型结果
        # print(s.model())

        # 创建一个类型为 tensor_type 的常数对象 add_result
        add_result = z3.Const(3, tensor_type)
        # 断言 add_result 在模型中的值与预期相符合
        assert s.model()[add_result].arg(0).arg(0).as_long() == 0
        assert s.model()[add_result].arg(1).arg(0).as_long() == 0

    def test_add_padding_4(self):
        # 定义一个继承自torch.nn.Module的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 前向传播函数，接受一个类型为[TensorType([2, 1])]的参数 x 和一个类型为[TensorType([3])]的参数 y
            def forward(self, x: TensorType([2, 1]), y: TensorType([3])):
                # 返回 x 和 y 的元素级加法结果
                return torch.add(x, y)

        # 创建一个重写追踪器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块，将追踪器的根节点和计算图作为参数，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的约束条件进行转换处理，计数器初始值为 0
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 向 Solver 对象添加转换后的约束条件
        s.add(transformed)

        # 断言 Solver 对约束条件的求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 创建一个类型为 tensor_type 的常数对象 add_result
        add_result = z3.Const(3, tensor_type)
        # 断言 add_result 在模型中的值与预期相符合
        assert s.model()[add_result] == tensor_type.tensor2(D(1, 2), D(1, 3))

    def test_add_padding_5(self):
        # 定义一个继承自torch.nn.Module的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 前向传播函数，接受一个类型为[TensorType([2, 2])]的参数 x 和一个类型为[TensorType([3])]的参数 y
            def forward(self, x: TensorType([2, 2]), y: TensorType([3])):
                # 返回 x 和 y 的元素级加法结果
                return torch.add(x, y)

        # 创建一个重写追踪器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块，将追踪器的根节点和计算图作为参数，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的约束条件进行转换处理，计数器初始值为 0
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 向 Solver 对象添加转换后的约束条件
        s.add(transformed)

        # 断言 Solver 对约束条件的求解结果为不可满足
        self.assertEqual(s.check(), z3.unsat)
    def test_add_size_3(self):
        # 定义一个继承自 torch.nn.Module 的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数，接受两个参数 x 和 y，类型为 TensorType([Dyn, Dyn, Dyn])
            def forward(
                self, x: TensorType([Dyn, Dyn, Dyn]), y: TensorType([Dyn, Dyn, Dyn])
            ):
                # 返回 x 和 y 的按元素加法结果
                return torch.add(x, y)

        # 创建一个重写跟踪器实例
        ast_rewriter = RewritingTracer()
        # 使用跟踪器对 BasicBlock 类进行追踪
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块，使用跟踪器的根节点和追踪得到的图
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的约束条件进行转换
        transformed = transform_all_constraints(traced, counter=0)

        # 创建一个 Z3 求解器实例
        s = z3.Solver()
        # 添加转换后的约束条件到求解器中
        s.add(transformed)
        # 断言求解器的结果为 z3.sat（满足）
        self.assertEqual(s.check(), z3.sat)

        # 创建一个常量 x，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 创建常量 s1, s2, s3, s4, s5，类型为整数
        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")

        # 添加额外的约束条件到求解器中
        s.add(x == tensor_type.tensor3(D(1, s1), D(1, 1), D(1, s2)))
        s.add(y == tensor_type.tensor3(D(1, s3), D(1, s4), D(1, s5)))

        # 断言求解器的结果为 z3.sat（满足）
        self.assertEqual(s.check(), z3.sat)
        # 添加额外的约束条件到求解器中
        s.add(s2 == 5)
        # 断言求解器的结果为 z3.sat（满足）
        self.assertEqual(s.check(), z3.sat)
        # 添加额外的约束条件到求解器中
        s.add(s5 == 6)
        # 断言求解器的结果为 z3.unsat（不满足）
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_6(self):
        # 定义一个继承自 torch.nn.Module 的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数，接受两个参数 x 和 y，类型为 TensorType([Dyn]) 和 TensorType([Dyn, Dyn, Dyn])
            def forward(self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn])):
                # 返回 x 和 y 的按元素加法结果
                return torch.add(x, y)

        # 创建一个重写跟踪器实例
        ast_rewriter = RewritingTracer()
        # 使用跟踪器对 BasicBlock 类进行追踪
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块，使用跟踪器的根节点和追踪得到的图
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的约束条件进行转换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 求解器实例
        s = z3.Solver()
        # 添加转换后的约束条件到求解器中
        s.add(transformed)
        # 断言求解器的结果为 z3.sat（满足）
        self.assertEqual(s.check(), z3.sat)

        # 创建一个常量 x，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 创建常量 s1, s2, s3, s4, s5，类型为整数
        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")

        # 添加额外的约束条件到求解器中
        s.add(x == tensor_type.tensor1(D(1, s1)))
        s.add(y == tensor_type.tensor3(D(1, s2), D(1, s3), D(1, s4)))

        # 断言求解器的结果为 z3.sat（满足）
        self.assertEqual(s.check(), z3.sat)

        # 添加额外的约束条件到求解器中
        s.add(s1 == 4)
        s.add(s4 == 5)

        # 断言求解器的结果为 z3.unsat（不满足）
        self.assertEqual(s.check(), z3.unsat)

    def test_add_padding_7(self):
        # 定义一个继承自 torch.nn.Module 的基本块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 定义前向传播函数，接受两个参数 x 和 y，类型为 TensorType([Dyn]) 和 TensorType([Dyn, Dyn, Dyn, Dyn])
            def forward(
                self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])
            ):
                # 返回 x 和 y 的按元素加法结果
                return torch.add(x, y)

        # 创建一个重写跟踪器实例
        ast_rewriter = RewritingTracer()
        # 使用跟踪器对 BasicBlock 类进行追踪
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个图模块，使用跟踪器的根节点和追踪得到的图
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的约束条件进行转换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 求解器实例
        s = z3.Solver()
        # 添加转换后的约束条件到求解器中
        s.add(transformed)
        # 断言求解器的结果为 z3.sat（满足）
        self.assertEqual(s.check(), z3.sat)
        # 创建一个常量 x，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 创建常量 s1, s2, s3, s4, s5，类型为整数
        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")
        # 添加额外的约束条件到求解器中
        s.add(x == tensor_type.tensor2(D(s1, s2), D(s2, s3)))
        # 断言求解器的结果为 z3.unsat（不满足）
        self.assertEqual(s.check(), z3.unsat)
    def test_add_padding_8(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，接受两个参数 x 和 y，并返回它们的和
            def forward(
                self, x: TensorType([Dyn]), y: TensorType([Dyn, Dyn, Dyn, Dyn])
            ):
                return torch.add(x, y)

        # 创建 RewritingTracer 的实例
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪，并生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用 graph 和 ast_rewriter.root 创建 GraphModule 实例，并命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的模型进行约束变换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将 transformed 添加到 Solver 中作为约束条件
        s.add(transformed)
        # 断言 Solver 的检查结果为 z3.sat (满足条件)
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个名为 x 的常量，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 创建一个名为 y 的常量，类型为 tensor_type
        y = z3.Const(2, tensor_type)

        # 创建多个整数常量 s1, s2, s3, s4, s5
        s1, s2, s3, s4, s5 = z3.Ints("s1 s2 s3 s4 s5")
        # 将 x 等于 tensor_type.tensor1(D(s1, 1)) 的约束添加到 Solver 中
        s.add(x == tensor_type.tensor1(D(s1, 1)))
        # 添加 s1 >= 0 的约束
        s.add(s1 >= 0)
        # 断言 Solver 的检查结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 将 y 等于 tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(0, s5)) 的约束添加到 Solver 中
        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(0, s5)))
        # 断言 Solver 的检查结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

    def test_add_padding_9(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 定义 forward 方法，接受两个参数 x 和 y，并返回它们的和
            def forward(self, x: Dyn, y: TensorType([Dyn, Dyn, Dyn, Dyn])):
                return torch.add(x, y)

        # 创建 RewritingTracer 的实例
        ast_rewriter = RewritingTracer()
        # 使用 ast_rewriter 对 BasicBlock 进行追踪，并生成图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 使用 graph 和 ast_rewriter.root 创建 GraphModule 实例，并命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的模型进行约束变换
        transformed = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        s = z3.Solver()
        # 将 transformed 添加到 Solver 中作为约束条件
        s.add(transformed)

        # 断言 Solver 的检查结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)
        
        # 创建一个名为 x 的常量，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 创建一个名为 y 的常量，类型为 tensor_type
        y = z3.Const(2, tensor_type)

        # 创建多个整数常量 s1, s2, s3, s4, s5, s6, s7
        s1, s2, s3, s4, s5, s6, s7 = z3.Ints("s1 s2 s3 s4 s5 s6 s7")
        # 将 x 等于 tensor_type.tensor1(D(s1, s7)) 的约束添加到 Solver 中
        s.add(x == tensor_type.tensor1(D(s1, s7)))
        # 添加 s1 == 1 的约束
        s.add(s1 == 1)
        # 断言 Solver 的检查结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 将 y 等于 tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(s6, s5)) 的约束添加到 Solver 中
        s.add(y == tensor_type.tensor4(D(0, s2), D(0, s3), D(0, s4), D(s6, s5)))
        # 断言 Solver 的检查结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)

        # 添加 s6 == 1 的约束
        s.add(s6 == 1)
        # 断言 Solver 的检查结果为 z3.sat
        self.assertEqual(s.check(), z3.sat)
        
        # 添加 s5 != 1 和 s7 != 1 的约束
        s.add(s5 != 1, s7 != 1)
        # 断言满足约束条件
        assert s.check()

        # 断言 s5 和 s7 在模型中的值相等
        assert s.model()[s5].as_long() == s.model()[s7].as_long()
    # 定义测试函数 test_conv_static
    def test_conv_static(self):
        # 定义整数变量 s1, s2, s3, s4, e1, e2, e3, e4, s11, s22, s33, s44, e11, e22, e33, e44 使用 Z3 提供的 Ints 函数
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        e1, e2, e3, e4 = z3.Ints("e1 e2 e3 e4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        e11, e22, e33, e44 = z3.Ints("e11 e22 e33 e44")
        
        # 创建约束表达式 d1, d2, d3, d4, b1, b2, b3, b4 使用自定义的 D 函数
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )
        b1, b2, b3, b4 = D(e11, e1), D(e22, e2), D(e33, e3), D(e44, e4)

        # 定义神经网络模块 BasicBlock
        class BasicBlock(torch.nn.Module):
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                # 调用父类的初始化方法
                super().__init__()
                # 创建二维卷积层对象 conv1
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )

            def forward(self, x: TensorType((1, 2, 10, 20))):
                # 神经网络前向传播，传入参数 x，返回卷积层 conv1 对输入 x 的处理结果
                return self.conv1(x)

        # 创建重写跟踪器对象 ast_rewriter
        ast_rewriter = RewritingTracer()

        # 创建 BasicBlock 类的实例 B，指定参数
        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 对 B 实例执行前向传播，并获取输出的大小
        res = B.forward(torch.rand(1, 2, 10, 20)).size()

        # 跟踪 BasicBlock 实例 B 的图形表示
        graph = ast_rewriter.trace(B)
        # 创建图模块 traced，基于 ast_rewriter 的根节点和图形表示，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对所有约束进行变换，得到新的转换后约束 new_transformed_c
        new_transformed_c = transform_all_constraints(traced)
        # 创建 Z3 求解器对象 solver
        solver = z3.Solver()
        # 添加新的转换后约束 new_transformed_c 到求解器中
        solver.add(new_transformed_c)
        # 断言求解器的检查结果为 satisfiable（满足）
        self.assertEqual(solver.check(), z3.sat)

        # 定义常量 x, y，使用 tensor_type 类型
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        # 将 d1, d2, d3, d4 分别赋给 x，将 b1, b2, b3, b4 分别赋给 y
        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))
        # 断言求解器的检查结果为 satisfiable（满足）
        self.assertEqual(solver.check(), z3.sat)
        
        # 断言求解器模型中 e3 的整数值等于 res 的第三个元素
        assert solver.model()[e3].as_long() == res[2]
        # 断言求解器模型中 e4 的整数值等于 res 的第四个元素
        assert solver.model()[e4].as_long() == res[3]

        # 创建 BasicBlock 类的另一个实例 B2，指定不同的参数
        B2 = BasicBlock(2, 4, 5, 2, 9, 2, 2)
        # 对 B2 实例执行前向传播，并获取输出的大小
        res2 = B2.forward(torch.rand(1, 2, 10, 20)).size()

        # 再次跟踪 BasicBlock 实例 B2 的图形表示
        graph2 = ast_rewriter.trace(B2)
        # 创建图模块 traced2，基于 ast_rewriter 的根节点和 B2 的图形表示，命名为 "gm"
        traced2 = GraphModule(ast_rewriter.root, graph2, "gm")
        # 对所有约束进行变换，得到新的转换后约束 new_transformed_c
        new_transformed_c = transform_all_constraints(traced2)
        # 创建新的 Z3 求解器对象 solver
        solver = z3.Solver()
        # 添加新的转换后约束 new_transformed_c 到求解器中
        solver.add(new_transformed_c)

        # 将 d1, d2, d3, d4 分别赋给 x，将 b1, b2, b3, b4 分别赋给 y
        solver.add(x == tensor_type.tensor4(d1, d2, d3, d4))
        solver.add(y == tensor_type.tensor4(b1, b2, b3, b4))

        # 断言求解器的检查结果为 satisfiable（满足）
        self.assertEqual(solver.check(), z3.sat)
        # 断言求解器模型中 e3 的整数值等于 res2 的第三个元素
        assert solver.model()[e3].as_long() == res2[2]
        # 断言求解器模型中 e4 的整数值等于 res2 的第四个元素
        assert solver.model()[e4].as_long() == res2[3]
    def test_reshape_dyn(self):
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")

        class BasicBlock(torch.nn.Module):
            def forward(self, x: Dyn):
                # 将输入张量 x 进行形状重塑为 (2, -1)
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 模块进行追踪，生成计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个包含重写根节点的图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        # 对追踪后的计算图进行约束转换
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        # 添加转换后的约束到 Z3 Solver
        s.add(transformed)
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 创建一个名为 x 的常量符号，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 添加一个张量类型的约束到 Z3 Solver，与 tensor_type.tensor1(D(1, s11)) 相等
        s.add(x == tensor_type.tensor1(D(1, s11)))
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 添加一个逻辑或约束，要求 s11 等于 2、4 或 9 中的一个
        s.add(z3.Or([s11 == 2, s11 == 4, s11 == 9]))
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 添加约束要求 s11 必须等于 9
        s.add(s11 == 9)
        # 检查约束是否不可满足
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_annotated(self):
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")
        d1, d2, d3, d4 = (
            D(s11, s1),
            D(s22, s2),
            D(s33, s3),
            D(s44, s4),
        )

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn])):
                # 将输入张量 x 进行形状重塑为 (2, -1)
                return torch.reshape(x, (2, -1))

        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 模块进行追踪，生成计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个包含重写根节点的图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪后的计算图进行约束转换
        transformed = transform_all_constraints(traced)
        s = z3.Solver()
        # 添加转换后的约束到 Z3 Solver
        s.add(transformed)
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 创建一个名为 x 的常量符号，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 添加一个张量类型的约束到 Z3 Solver，与 tensor_type.tensor2(d1, d2) 不相等
        s.add(x == tensor_type.tensor2(d1, d2))
        # 检查约束是否不可满足
        self.assertEqual(s.check(), z3.unsat)

    def test_reshape_static_target(self):
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")

        class BasicBlock(torch.nn.Module):
            def forward(self, x: TensorType([Dyn])):
                # 将输入张量 x 进行形状重塑为 (2, 3)
                return torch.reshape(x, (2, 3))

        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 模块进行追踪，生成计算图
        graph = ast_rewriter.trace(BasicBlock())
        # 创建一个包含重写根节点的图模块
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪后的计算图进行约束转换
        transformed = transform_all_constraints(traced)
        # 创建一个新的 Z3 Solver
        s = z3.Solver()
        # 添加转换后的约束到 Z3 Solver
        s.add(transformed)
        # 检查约束是否可满足
        self.assertEqual(s.check(), z3.sat)

        # 创建一个名为 x 的常量符号，类型为 tensor_type
        x = z3.Const(1, tensor_type)
        # 添加一个张量类型的约束到 Z3 Solver，与 tensor_type.tensor1(D(1, s11)) 相等
        s.add(x == tensor_type.tensor1(D(1, s11)))
        # 执行一次检查
        s.check()
        # 断言 s11 在模型中的值必须为 6
        assert s.model()[s11].as_long() == 6
        # 添加一个约束要求 s11 不等于 6
        s.add(s11 != 6)
        # 检查约束是否不可满足
        self.assertEqual(s.check(), z3.unsat)
    def test_reshape_static_target2(self):
        # 声明四个整数变量作为 Z3 的整数符号
        s11, s22, s33, s44 = z3.Ints("s11 s22 s33 s44")

        # 定义一个继承自 torch.nn.Module 的基本模块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            # 重写 forward 方法，接收一个 Dyn 类型的参数 x
            def forward(self, x: Dyn):
                # 使用 torch.reshape 将 x 重塑为形状 (2, 3, 1, 1)
                return torch.reshape(x, (2, 3, 1, 1))

        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 类进行追踪，获取其图形表示
        graph = ast_rewriter.trace(BasicBlock())
        # 创建 GraphModule 实例 traced，用于管理追踪的根节点和图形
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪得到的约束条件进行转换
        transformed = transform_all_constraints(traced)
        # 创建一个 Z3 Solver 实例 s
        s = z3.Solver()
        # 将转换后的约束条件添加到求解器中
        s.add(transformed)
        # 断言求解结果为可满足
        self.assertEqual(s.check(), z3.sat)

        # 创建一个常量 x，表示一个张量类型的常量值为 1
        x = z3.Const(1, tensor_type)
        # 将约束条件添加到求解器中，要求 s11 的值等于 tensor_type 的第一维的乘积
        s.add(x == tensor_type.tensor1(D(1, s11)))
        # 再次进行求解
        s.check()
        # 断言求解结果中 s11 的值为 6
        assert s.model()[s11].as_long() == 6
        # 添加约束条件 s11 不等于 6
        s.add(s11 != 6)
        # 最后断言求解结果为不可满足
        self.assertEqual(s.check(), z3.unsat)

    def test_conv2D_maxpool2d_flatten(self):
        # 定义一个继承自 torch.nn.Module 的基本模块类 BasicBlock
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模块中的各个层和操作
                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            # 重写 forward 方法，接收一个形状为 (4, 3, 32, 32) 的张量类型的参数 x
            def forward(self, x: TensorType((4, 3, 32, 32))):
                # 进行一系列的卷积、池化、全连接等操作
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                # 将 out 展平为一维张量
                out = torch.flatten(out, 1)
                return out

        # 创建 BasicBlock 类的实例 B
        B = BasicBlock()
        # 对随机生成的形状为 (4, 3, 32, 32) 的张量进行前向传播，并获取输出的形状
        res = B.forward(torch.rand(4, 3, 32, 32)).shape
        # 创建一个 RewritingTracer 实例 ast_rewriter
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，获取其图形表示
        graph = ast_rewriter.trace(B)
        # 创建 GraphModule 实例 traced，用于管理追踪的根节点和图形
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 对追踪得到的约束条件进行转换，设置计数器为 0
        constraints = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 实例 solver
        solver = z3.Solver()
        # 将转换后的约束条件添加到求解器中
        solver.add(constraints)
        # 进行求解
        solver.check()
        # 创建一个常量 input，表示一个张量类型的常量值为 1
        input = z3.Const(1, tensor_type)
        # 将约束条件添加到求解器中，要求 input 的值等于张量类型的第四维的乘积
        solver.add(input == tensor_type.tensor4(D(1, 4), D(1, 3), D(1, 32), D(1, 32)))
        # 再次进行求解
        solver.check()
        # 创建一个常量 output，表示一个张量类型的常量值为 48
        output = z3.Const(48, tensor_type)
        # 断言求解结果中 output 的第一和第二维分别等于 res 的第一个和第二个元素
        assert solver.model()[output].arg(0).arg(1) == res[0]
        assert solver.model()[output].arg(1).arg(1) == res[1]
    def test_conv2D_maxpool2d_flatten_unsat(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化函数，设置神经网络结构
            def __init__(self):
                super().__init__()

                # 第一个卷积层：输入通道数为3，输出通道数为6，卷积核大小为5
                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                # 最大池化层：池化窗口大小为2x2，步幅为2
                self.pool = torch.nn.MaxPool2d(2, 2)
                # 第二个卷积层：输入通道数为6，输出通道数为16，卷积核大小为5
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                # 全连接层：输入特征数为5，输出特征数为120
                self.fc1 = torch.nn.Linear(5, 120)
                # 自适应平均池化层：输出特征图大小为6x7
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            # 前向传播函数，接收输入 x，并返回输出 out
            def forward(self, x: TensorType((4, 3, 32, 32))):
                # 第一层卷积
                out = self.conv1(x)
                # 第一层池化
                out = self.pool(out)
                # 第二层卷积
                out = self.conv2(out)
                # 第二层池化
                out = self.pool(out)
                # 全连接层
                out = self.fc1(out)
                # 自适应平均池化层
                out = self.pool2(out)
                # 将输出扁平化，保留第一维不变
                out = torch.flatten(out, 1)
                return out

        # 创建 BasicBlock 的实例 B
        B = BasicBlock()
        # 创建 AST 重写器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，生成对应的图形结构
        graph = ast_rewriter.trace(B)
        # 使用重写后的根节点和图形创建图模块实例，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 转换所有约束条件为解析器能处理的形式
        constraints = transform_all_constraints(traced, counter=0)
        # 创建 Z3 求解器对象
        solver = z3.Solver()
        # 添加约束条件到求解器
        solver.add(constraints)
        # 检查求解器是否有解，并断言结果为不可满足
        self.assertEqual(solver.check(), z3.unsat)

    def test_conv2D_maxpool2d_flatten_dyn(self):
        # 定义一个名为 BasicBlock 的内部类，继承自 torch.nn.Module
        class BasicBlock(torch.nn.Module):
            # 初始化函数，设置神经网络结构
            def __init__(self):
                super().__init__()

                # 第一个卷积层：输入通道数为3，输出通道数为6，卷积核大小为5
                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                # 最大池化层：池化窗口大小为2x2，步幅为2
                self.pool = torch.nn.MaxPool2d(2, 2)
                # 第二个卷积层：输入通道数为6，输出通道数为16，卷积核大小为5
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                # 全连接层：输入特征数为5，输出特征数为120
                self.fc1 = torch.nn.Linear(5, 120)
                # 自适应平均池化层：输出特征图大小为6x7
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            # 前向传播函数，接收输入 x，并返回输出 out
            def forward(self, x: TensorType((Dyn, 3, 32, 32))):
                # 第一层卷积
                out = self.conv1(x)
                # 第一层池化
                out = self.pool(out)
                # 第二层卷积
                out = self.conv2(out)
                # 第二层池化
                out = self.pool(out)
                # 全连接层
                out = self.fc1(out)
                # 自适应平均池化层
                out = self.pool2(out)
                # 将输出扁平化，保留第一维不变
                out = torch.flatten(out, 1)
                return out

        # 创建 BasicBlock 的实例 B
        B = BasicBlock()
        # 创建 AST 重写器对象
        ast_rewriter = RewritingTracer()
        # 对 BasicBlock 实例 B 进行追踪，生成对应的图形结构
        graph = ast_rewriter.trace(B)
        # 使用重写后的根节点和图形创建图模块实例，命名为 "gm"
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 转换所有约束条件为解析器能处理的形式
        constraints = transform_all_constraints(traced, counter=0)
        # 创建 Z3 求解器对象
        solver = z3.Solver()
        # 添加约束条件到求解器
        solver.add(constraints)
        # 检查求解器是否有解，并断言结果为可满足
        self.assertEqual(solver.check(), z3.sat)
    def test_type_check_flatten(self):
        s1, s2, s3, s4 = z3.Ints("s1 s2 s3 s4")

        # 定义一个继承自 torch.nn.Module 的子类 M
        class M(torch.nn.Module):
            # 定义该类的前向传播方法，输入 x 的类型为 TensorType([2, 3, 4, 5])
            def forward(self, x: TensorType([2, 3, 4, 5])):
                # 调用 torch.flatten 函数，对输入 x 进行扁平化处理，指定起始维度和结束维度
                return torch.flatten(x, start_dim=1, end_dim=3)

        # 创建 M 的实例 module
        module = M()
        # 对 module 进行符号化跟踪，得到 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 转换所有约束条件为特定格式
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 向 solver 中添加约束条件
        solver.add(constraints)
        # 断言 solver 的检查结果为 z3.sat
        self.assertEqual(solver.check(), z3.sat)
        # 创建一个名为 flatten 的常量，类型为 tensor_type，值为 2
        flatten = z3.Const(2, tensor_type)

        # 调用 M 的 forward 方法，输入 torch.rand(2, 3, 4, 5)，获取输出的尺寸大小
        res = M().forward(torch.rand(2, 3, 4, 5)).size()
        # 使用 solver 的模型来验证 flatten 的属性
        assert solver.model()[flatten].arg(0).arg(1) == res[0]
        assert solver.model()[flatten].arg(1).arg(1) == res[1]

        # 定义另一个继承自 torch.nn.Module 的子类 M
        class M(torch.nn.Module):
            # 定义该类的前向传播方法，输入 x 的类型为 TensorType([2, 3, Dyn, 5])
            def forward(self, x: TensorType([2, 3, Dyn, 5])):
                # 调用 torch.flatten 函数，对输入 x 进行扁平化处理，指定起始维度和结束维度
                return torch.flatten(x, start_dim=1, end_dim=3)

        # 创建 M 的实例 module
        module = M()
        # 对 module 进行符号化跟踪，得到 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 转换所有约束条件为特定格式
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 向 solver 中添加约束条件
        solver.add(constraints)
        # 断言 solver 的检查结果为 z3.sat
        self.assertEqual(solver.check(), z3.sat)
        # 创建名为 x 和 y 的常量，类型为 tensor_type，值分别为 1 和 2
        x = z3.Const(1, tensor_type)
        y = z3.Const(2, tensor_type)

        # 向 solver 中添加一个等式约束条件
        solver.add(x == tensor_type.tensor4(D(1, 2), D(1, 3), D(0, s1), D(1, 5)))
        # 断言 solver 的检查结果为 z3.sat
        self.assertEqual(solver.check(), z3.sat)
        # 使用 solver 的模型来验证 y 的属性
        assert solver.model()[y].arg(1).arg(0) == 0

        # 定义另一个继承自 torch.nn.Module 的子类 M
        class M(torch.nn.Module):
            # 定义该类的前向传播方法，输入 x 的类型为 TensorType([2, 3, Dyn])
            def forward(self, x: TensorType([2, 3, Dyn])):
                # 调用 torch.flatten 函数，对输入 x 进行扁平化处理，指定起始维度为 10，结束维度为 0
                return torch.flatten(x, 10, 0)

        # 创建 M 的实例 module
        module = M()
        # 对 module 进行符号化跟踪，得到 torch.fx.GraphModule 对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        # 转换所有约束条件为特定格式
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 向 solver 中添加约束条件
        solver.add(constraints)
        # 断言 solver 的检查结果为 z3.unsat
        self.assertEqual(solver.check(), z3.unsat)
class ConstraintGeneration(unittest.TestCase):
    # 定义约束生成测试类 ConstraintGeneration，继承自 unittest.TestCase
    def test_add_reshape(self):
        # 测试函数：测试 add 和 reshape 的功能
        class BasicBlock(torch.nn.Module):
            # 定义神经网络基本模块 BasicBlock
            def forward(self, x: Dyn, y: Dyn):
                # 前向传播函数，接受两个动态类型参数 x 和 y
                return torch.add(torch.reshape(x, (1, 2)), torch.reshape(y, (2, 2)))
                # 返回将 x 和 y reshape 后的结果相加的张量

        ast_rewriter = RewritingTracer()
        # 创建重写追踪器对象 ast_rewriter
        graph = ast_rewriter.trace(BasicBlock())
        # 使用 ast_rewriter 追踪 BasicBlock 类的对象，获取图形表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 使用追踪的根节点和图形创建 GraphModule 对象 traced

        generator = ConstraintGenerator(traced)
        # 创建约束生成器对象 generator，传入 traced 对象
        new_constraints, counter = generator.generate_constraints(0)
        # 调用生成约束的方法，返回新约束和计数器
        assert len(new_constraints.conjucts) == 11
        # 断言新约束的 conjucts 数量为 11

    def test_conv_reshape_add(self):
        # 测试函数：测试卷积、reshape 和 add 的功能
        class BasicBlock(torch.nn.Module):
            # 定义神经网络基本模块 BasicBlock
            def __init__(
                self,
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups,
                dilation,
            ):
                super().__init__()
                # 调用父类的初始化方法
                self.conv1 = torch.nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                )
                # 创建卷积层对象 conv1

            def forward(self, x: Dyn, y: Dyn):
                # 前向传播函数，接受两个动态类型参数 x 和 y
                return torch.add(self.conv1(torch.reshape(x, (1, 2, 10, 20))), y)
                # 返回将 x reshape 后经过 conv1 处理再加上 y 的结果

        B = BasicBlock(2, 2, 2, 3, 2, 2, 2)
        # 创建 BasicBlock 类的实例 B，传入参数
        ast_rewriter = RewritingTracer()
        # 创建重写追踪器对象 ast_rewriter
        graph = ast_rewriter.trace(B)
        # 使用 ast_rewriter 追踪 B 对象，获取图形表示
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        # 使用追踪的根节点和图形创建 GraphModule 对象 traced

        generator = ConstraintGenerator(traced)
        # 创建约束生成器对象 generator，传入 traced 对象
        new_constraints, counter = generator.generate_constraints(0)
        # 调用生成约束的方法，返回新约束和计数器
        assert len(new_constraints.conjucts) == 16
        # 断言新约束的 conjucts 数量为 16


class TestInternalConstraints(unittest.TestCase):
    # 定义内部约束测试类 TestInternalConstraints，继承自 unittest.TestCase
    def test_precision(self):
        # 测试函数：测试精度约束
        c1 = BinConstraintT(Dyn, TVar("x"), op_precision)
        # 创建动态类型为 Dyn 的二元约束对象 c1，使用 op_precision 操作符
        transformed, _ = transform_constraint(c1, 0)
        # 调用约束转换函数，返回转换后的结果和计数器
        assert transformed == T()
        # 断言转换后的结果为类型 T()

        c2 = BinConstraintT(TensorType([1, Dyn, 3]), TVar("x"), op_precision)
        # 创建包含 TensorType 的二元约束对象 c2，使用 op_precision 操作符
        transformed, counter = transform_constraint(c2, 0)
        # 调用约束转换函数，返回转换后的结果和计数器
        assert len(transformed.conjucts) == 7
        # 断言转换后的结果的 conjucts 数量为 7

    def test_matching(self):
        # 测试函数：测试匹配约束
        c1 = BinConstraintT(
            TVar("x"),
            TensorType([DVar("a"), DVar("b"), DVar("c"), DVar("d")]),
            op_matching,
        )
        # 创建包含匹配约束的二元约束对象 c1，使用 op_matching 操作符
        transformed, _ = transform_constraint(c1, 0)
        # 调用约束转换函数，返回转换后的结果和计数器
        assert len(transformed.disjuncts) == 2
        # 断言转换后的结果的 disjuncts 数量为 2

    def test_consistency(self):
        # 测试函数：测试一致性约束
        c1 = BinConstraintT(
            TVar("x"), TensorType([DVar("a"), DVar("b")]), op_consistency
        )
        # 创建包含一致性约束的二元约束对象 c1，使用 op_consistency 操作符
        transformed, count = transform_constraint(c1, 0)
        # 调用约束转换函数，返回转换后的结果和计数器

        assert len(transformed.disjuncts) == 5
        # 断言转换后的结果的 disjuncts 数量为 5
        transformed, count = transform_constraint(transformed, count)
        # 再次调用约束转换函数，返回转换后的结果和计数器
        assert len(transformed.disjuncts) == 5
        # 断言再次转换后的结果的 disjuncts 数量为 5

    # def test_apply_broadcasting(self):
    #     c1 = ApplyBroadcasting(TVar(1), TVar(2), TVar(3), TVar(4))
    #     # 创建应用广播的约束对象，传入四个 TVar 参数
    # 调用 transform_apply_broadcasting 函数对 c1 进行变换，并获取返回的 transformed 和 count
    transformed, count = transform_apply_broadcasting(c1, 5)
    # 使用断言检查 transformed 对象的 conjucts 属性的长度是否为 41
    assert len(transformed.conjucts) == 41
# 如果未安装 torchvision，则跳过这个测试类
@skipIfNoTorchVision
class TestResNet(unittest.TestCase):

    # 测试未满足条件的 ResNet-50 模型
    def test_resnet50_unsat(self):
        # 对 ResNet-50 模型进行符号化跟踪
        traced = symbolic_trace(models.resnet50())
        # 将图中所有节点的类型设置为动态类型
        for n in traced.graph.nodes:
            n.type = Dyn

        # 转换所有约束条件
        constraints = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 添加约束条件到求解器中
        solver.add(constraints)
        # 创建一个输入变量，并指定其为三维张量
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor3(D(1, 1), D(1, 3), D(1, 224)))
        # 断言求解器检查结果为不可满足
        self.assertEqual(solver.check(), z3.unsat)

    # 测试 ResNet-50 模型
    def test_resnet50(self):
        # 对 ResNet-50 模型进行符号化跟踪
        traced = symbolic_trace(models.resnet50())
        # 将图中所有节点的类型设置为动态类型
        for n in traced.graph.nodes:
            n.type = Dyn

        # 创建一个示例输入
        sample_input = torch.randn(1, 3, 224, 224)
        # 执行 ResNet-50 模型的前向传播，并获取输出大小
        res = models.resnet50().forward(sample_input).size()
        # 转换所有约束条件
        constraints = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 添加约束条件到求解器中
        solver.add(constraints)
        # 断言求解器检查结果为可满足
        self.assertEqual(solver.check(), z3.sat)
        # 创建一个线性变量
        linear = z3.Const(650, tensor_type)
        
        # 创建一个输入变量，并指定其为四维张量
        input = z3.Const(1, tensor_type)
        solver.add(input == tensor_type.tensor4(D(1, 1), D(1, 3), D(1, 224), D(1, 224)))
        # 断言求解器检查结果为可满足
        self.assertEqual(solver.check(), z3.sat)
        # 使用模型输出大小验证线性变量的维度
        assert solver.model()[linear] == tensor_type.tensor2(D(1, res[0]), D(1, res[1]))

    # 测试另一个 ResNet-50 模型
    def test_resnet502(self):
        # 对 ResNet-50 模型进行符号化跟踪
        traced = symbolic_trace(models.resnet50())
        # 将图中所有节点的类型设置为动态类型
        for n in traced.graph.nodes:
            n.type = Dyn

        # 转换所有约束条件
        constraints = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 添加约束条件到求解器中
        solver.add(constraints)
        # 创建一个线性变量
        linear = z3.Const(650, tensor_type)
        # 创建一个输入变量，并指定其为四维张量，其中 batch 大于 4
        input = z3.Const(1, tensor_type)
        batch = z3.Int("b")
        solver.add(
            input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224))
        )
        solver.add(batch > 4)
        solver.check()
        # 断言 batch 变量等于 linear 变量的第一个参数的第二个参数
        assert solver.model()[batch] == solver.model()[linear].arg(0).arg(1)

    # 测试另一个 ResNet-50 模型
    def test_resnet503(self):
        # 对 ResNet-50 模型进行符号化跟踪
        traced = symbolic_trace(models.resnet50())
        # 将图中所有节点的类型设置为动态类型
        for n in traced.graph.nodes:
            n.type = Dyn

        # 转换所有约束条件
        constraints = transform_all_constraints(traced, counter=0)
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        # 添加约束条件到求解器中
        solver.add(constraints)
        # 创建一个线性变量
        linear = z3.Const(650, tensor_type)
        # 创建一个输入变量，并指定其为四维张量，同时验证 linear 的维度
        input = z3.Const(1, tensor_type)
        batch, d1, d2 = z3.Ints("b d1 d2")
        solver.add(
            input == tensor_type.tensor4(D(1, batch), D(1, 3), D(1, 224), D(1, 224))
        )
        solver.add(linear == tensor_type.tensor2(D(1, d1), D(1, d2)))
        # 断言求解器检查结果为可满足
        self.assertEqual(solver.check(), z3.sat)
        # 添加一个约束，使得 batch 变量不等于 d1 变量
        solver.add(batch != d1)
        # 断言求解器检查结果为不可满足
        self.assertEqual(solver.check(), z3.unsat)
    # 测试函数，验证 AlexNet 模型的运行情况
    def test_alexnet1(self):
        # 创建 AlexNet 模型
        alexnet = models.alexnet()
        # 对模型进行符号化跟踪，得到符号化的图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        # 将图中所有节点的类型设为 Dyn
        for n in symbolic_traced.graph.nodes:
            n.type = Dyn

        # 执行模型的前向传播，获取输出的尺寸
        res = alexnet.forward(torch.rand(10, 3, 227, 227)).size()

        # 对符号化跟踪后的模型应用约束变换
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        # 创建 Z3 求解器对象
        solver = z3.Solver()
        # 添加约束到求解器
        solver.add(constraints)
        # 断言求解结果为可满足
        self.assertEqual(solver.check(), z3.sat)

        # 定义输入、卷积层的变量，并添加到求解器中
        input = z3.Const(1, tensor_type)
        conv = z3.Const(2, tensor_type)
        solver.add(
            input == tensor_type.tensor4(D(1, 10), D(1, 3), D(1, 227), D(1, 227))
        )
        # 断言求解器结果中的卷积层与期望的尺寸相匹配
        self.assertEqual(solver.check(), z3.sat)
        assert solver.model()[conv] == tensor_type.tensor4(
            D(1, 10), D(1, 64), D(1, 56), D(1, 56)
        )

        # 定义 ReLU 层的变量，并断言求解器结果中的尺寸与期望相匹配
        relu = z3.Const(7, tensor_type)
        assert solver.model()[relu] == tensor_type.tensor4(
            D(1, 10), D(1, 64), D(1, 56), D(1, 56)
        )

        # 定义第一个最大池化层的变量，并断言求解器结果中的尺寸与期望相匹配
        maxpool = z3.Const(8, tensor_type)
        assert solver.model()[maxpool] == tensor_type.tensor4(
            D(1, 10), D(1, 64), D(1, 27), D(1, 27)
        )

        # 定义第二个最大池化层的变量，并断言求解器结果中的尺寸与期望相匹配
        maxpool2 = z3.Const(42, tensor_type)
        assert solver.model()[maxpool2] == tensor_type.tensor4(
            D(1, 10), D(1, 256), D(1, 6), D(1, 6)
        )

        # 定义展平层的变量，并断言求解器结果中的尺寸与期望相匹配
        flatten = z3.Const(52, tensor_type)
        assert solver.model()[flatten] == tensor_type.tensor2(D(1, 10), D(1, 9216))

        # 定义全连接层的变量，并断言求解器结果中的尺寸与期望相匹配
        linear = z3.Const(64, tensor_type)
        assert solver.model()[linear] == tensor_type.tensor2(D(1, 10), D(1, 4096))

        # 定义第二个全连接层的变量，并断言求解器结果中的尺寸与期望相匹配
        linear2 = z3.Const(109, tensor_type)
        assert solver.model()[linear2] == tensor_type.tensor2(
            D(1, res[0]), D(1, res[1])
        )

    # 测试函数，验证修改占位符后的 AlexNet 模型的运行情况
    def test_alexnet2(self):
        # 创建 AlexNet 模型
        alexnet = models.alexnet()
        # 对模型进行符号化跟踪，得到符号化的图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        # 遍历图中所有节点，将操作为 "placeholder" 的节点类型设为具有部分确定性维度的 TensorType
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, 4, 227, 227])

        # 对符号化跟踪后的模型应用约束变换
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        # 创建 Z3 求解器对象
        solver = z3.Solver()
        # 添加约束到求解器
        solver.add(constraints)
        # 断言求解结果为不可满足
        self.assertEqual(solver.check(), z3.unsat)

    # 测试函数，验证修改占位符后的 AlexNet 模型的运行情况
    def test_alexnet3(self):
        # 创建 AlexNet 模型
        alexnet = models.alexnet()
        # 对模型进行符号化跟踪，得到符号化的图模块
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        # 遍历图中所有节点，将操作为 "placeholder" 的节点类型设为具有动态维度的 TensorType
        for n in symbolic_traced.graph.nodes:
            if n.op == "placeholder":
                n.type = TensorType([Dyn, Dyn, 227, 227])

        # 对符号化跟踪后的模型应用约束变换
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        # 创建 Z3 求解器对象
        solver = z3.Solver()
        # 添加约束到求解器
        solver.add(constraints)
        # 断言求解结果为可满足
        self.assertEqual(solver.check(), z3.sat)
    # 定义一个测试函数 test_alexnet4，用于测试 AlexNet 模型的功能
    def test_alexnet4(self):
        # 创建一个 AlexNet 模型实例
        alexnet = models.alexnet()
        
        # 对 AlexNet 模型进行符号化跟踪，得到一个 Torch 的图模块对象
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(alexnet)

        # 遍历符号化跟踪后的图的所有节点
        for n in symbolic_traced.graph.nodes:
            # 如果节点是占位符节点
            if n.op == "placeholder":
                # 将该节点的类型设置为指定的 TensorType，这里是一个三维张量
                n.type = TensorType([Dyn, Dyn, 227])

        # 对符号化跟踪后的模型应用变换，得到所有约束的转换结果
        constraints = transform_all_constraints(symbolic_traced, counter=0)
        
        # 创建一个 Z3 Solver 对象
        solver = z3.Solver()
        
        # 将所有约束添加到 Solver 中
        solver.add(constraints)
        
        # 断言 Solver 的检查结果与 z3 库中的 unsat（不可满足）相等
        self.assertEqual(solver.check(), z3.unsat)
# 如果当前脚本作为主程序运行，则执行单元测试的主函数
if __name__ == "__main__":
    # 运行 Python 的单元测试框架，执行测试用例
    unittest.main()
```