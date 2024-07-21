# `.\pytorch\test\onnx\test_pytorch_onnx_shape_inference.py`

```
# Owner(s): ["module: onnx"]

import io  # 导入 io 模块

import numpy as np  # 导入 numpy 库，并简称为 np
import onnx  # 导入 onnx 库
import pytorch_test_common  # 导入自定义的 pytorch 测试公共模块
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion  # 从公共模块中导入特定函数

import torch  # 导入 torch 库
from torch.onnx import _constants, utils  # 从 torch.onnx 中导入常量和工具模块
from torch.onnx._globals import GLOBALS  # 从 torch.onnx._globals 中导入 GLOBALS 常量
from torch.onnx._internal import jit_utils  # 从 torch.onnx._internal 中导入 jit_utils 工具
from torch.testing._internal import common_utils  # 从 torch.testing._internal 中导入 common_utils 工具


def expect_tensor(scalar_type, shape=None):
    # 返回一个验证函数，用于验证期望的张量类型和形状
    def verify(actual_type):
        np.testing.assert_equal(actual_type.scalarType(), scalar_type)  # 使用 numpy 测试断言实际类型的标量类型
        if shape is not None:
            np.testing.assert_equal(actual_type.varyingSizes(), shape)  # 如果给定形状，使用 numpy 测试断言实际类型的不同尺寸

    return verify


def as_graphcontext(graph: torch.Graph) -> jit_utils.GraphContext:
    # 将 torch.Graph 转换为 jit_utils.GraphContext 对象
    return jit_utils.GraphContext(
        graph=graph,
        block=graph.block(),
        opset=_constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET,  # 使用 ONNX 导出器的最大操作集
        original_node=None,  # 原始节点为空
        params_dict={},  # 参数字典为空
        env={},  # 环境字典为空
        values_in_env=set(),  # 环境中值的集合为空
    )


def g_op(graph: torch.Graph, op_name: str, *args, **kwargs):
    # 在给定图形上执行操作，返回操作后的图形上下文
    return as_graphcontext(graph).op(op_name, *args, **kwargs)


class TestONNXShapeInference(pytorch_test_common.ExportTestCase):
    def setUp(self):
        # 设置测试环境，导出 ONNX 的最大操作集版本号
        self.opset_version = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        GLOBALS.export_onnx_opset_version = self.opset_version  # 设置全局导出 ONNX 的操作集版本号

    def run_test(self, g, n, type_assertion_funcs):
        # 运行测试，对给定图形执行类型推断，并验证类型断言函数
        if not isinstance(type_assertion_funcs, list):
            type_assertion_funcs = [type_assertion_funcs]

        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)  # 执行 ONNX 图形形状类型推断的 JIT 传递
        for out, type_assertion_func in zip(n.outputs(), type_assertion_funcs):
            type_assertion_func(out.type())  # 对每个输出类型执行类型断言函数的验证

    def create_empty_graph(self):
        # 创建一个空图形，并启动 ConstantMap 的初始化
        g = torch._C.Graph()
        torch._C._jit_pass_onnx_graph_shape_type_inference(g, {}, self.opset_version)  # 执行 ONNX 图形形状类型推断的 JIT 传递
        return g

    def insert_tensor_constant(self, g, tensor):
        # 在图形中插入张量常量节点
        return g_op(g, "Constant", value_t=tensor)

    def test_cast(self):
        # 测试类型转换，使用未知标量类型的输入
        g = self.create_empty_graph()
        input = g.addInput()  # 添加输入节点
        cast_out = g_op(g, "Cast", input, to_i=1)  # 执行 Cast 操作
        self.run_test(g, cast_out.node(), expect_tensor("Float"))  # 运行测试，期望输出为 Float 类型

    def test_constant_of_shape(self):
        # 测试 ConstantOfShape，使用 onnx::Shape 节点作为输入
        g = self.create_empty_graph()
        constant = self.insert_tensor_constant(g, torch.ones(1, 2, 3, 4))  # 插入张量常量节点
        shape = g_op(g, "Shape", constant)  # 执行 Shape 操作
        constant_of_shape = g_op(
            g, "ConstantOfShape", shape, value_t=torch.tensor([2.0])
        )  # 执行 ConstantOfShape 操作
        self.run_test(
            g, constant_of_shape.node(), expect_tensor("Float", shape=(1, 2, 3, 4))
        )  # 运行测试，期望输出为 Float 类型，并且形状为 (1, 2, 3, 4)
    def test_constant_of_shape_static(self):
        # 测试静态张量列表构造的 ConstantOfShape 函数
        rank = 4
        g = self.create_empty_graph()
        # 创建包含静态张量的常量列表
        constants = [
            self.insert_tensor_constant(g, torch.tensor(i + 1)) for i in range(rank)
        ]
        # 构造张量的形状，使用 prim::ListConstruct 操作
        shape = g_op(g, "prim::ListConstruct", *constants)
        shape.setType(torch._C.ListType.ofInts())
        # 使用 ConstantOfShape 操作创建常量张量
        constant_of_shape = g_op(
            g, "ConstantOfShape", shape, value_t=torch.tensor([2.0])
        )
        # 运行测试并期望输出为 Float 类型张量，形状为 (1, 2, 3, 4)
        self.run_test(
            g, constant_of_shape.node(), expect_tensor("Float", shape=(1, 2, 3, 4))
        )

    def test_constant_of_shape_dynamic(self):
        # 测试动态张量列表构造的 ConstantOfShape 函数
        rank = 4
        g = self.create_empty_graph()
        # 创建包含动态张量的输入列表
        inputs = [g.addInput() for i in range(rank)]
        # 构造张量的形状，使用 prim::ListConstruct 操作
        shape = g_op(g, "prim::ListConstruct", *inputs)
        shape.setType(torch._C.ListType.ofInts())
        # 使用 ConstantOfShape 操作创建常量张量
        constant_of_shape = g_op(
            g, "ConstantOfShape", shape, value_t=torch.tensor([2.0])
        )
        # 运行测试并期望输出为 Float 类型张量，形状为 (None, None, None, None)
        self.run_test(
            g,
            constant_of_shape.node(),
            expect_tensor("Float", shape=(None, None, None, None)),
        )

    def test_gather_dynamic_index(self):
        g = self.create_empty_graph()
        input = g.addInput()
        # 设置输入张量类型为 torch.float，大小为 [None, 3, 16, 16]
        input.setType(
            input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16])
        )
        indices = g.addInput()
        # 设置索引张量类型为 torch.int64，大小为 [None]
        indices.setType(indices.type().with_dtype(torch.int64).with_sizes([None]))
        # 执行 Gather 操作
        output = g_op(g, "Gather", input, indices, axis_i=1)
        # 运行测试并期望输出为 Float 类型张量，形状为 ([None, None, 16, 16])
        self.run_test(
            g, output.node(), expect_tensor("Float", shape=([None, None, 16, 16]))
        )

    def test_gather_scalar_index(self):
        g = self.create_empty_graph()
        input = g.addInput()
        # 设置输入张量类型为 torch.float，大小为 [None, 3, 16, 16]
        input.setType(
            input.type().with_dtype(torch.float).with_sizes([None, 3, 16, 16])
        )
        # 插入常量张量索引
        indices = self.insert_tensor_constant(g, torch.tensor(1))
        # 执行 Gather 操作
        output = g_op(g, "Gather", input, indices, axis_i=1)
        # 运行测试并期望输出为 Float 类型张量，形状为 ([None, 16, 16])
        self.run_test(g, output.node(), expect_tensor("Float", shape=([None, 16, 16])))
    # 定义测试函数 test_reshape，用于测试 Reshape 操作
    def test_reshape(self):
        # 创建一个空图形 g
        g = self.create_empty_graph()
        # 在图形 g 中插入一个形状为 (2, 16, 5, 5) 的全为 1 的常量张量，并返回其节点
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 5))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        constant_2 = self.insert_tensor_constant(g, torch.tensor([2, 0, -1]))
        # 执行图操作 g_op，对 constant 和 constant_2 执行 Reshape 操作，并返回结果节点
        shape = g_op(g, "Reshape", constant, constant_2)
        # 在图形 g 上运行测试，检查 shape.node() 返回的节点是否满足期望的浮点张量形状 (2, 16, 25)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(2, 16, 25)))

        # 重置图形 g
        g = self.create_empty_graph()
        # 在图形 g 中插入一个形状为 (2, 16, 5, 4) 的全为 1 的常量张量，并返回其节点
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 4]))
        # 执行图操作 g_op，对 constant 和 constant_2 执行 Reshape 操作，并返回结果节点
        shape = g_op(g, "Reshape", constant, constant_2)
        # 在图形 g 上运行测试，检查 shape.node() 返回的节点是否满足期望的浮点张量形状 (10, 16, 4)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(10, 16, 4)))

        # 重置图形 g
        g = self.create_empty_graph()
        # 在图形 g 中插入一个形状为 (2, 16, 5, 4) 的全为 1 的常量张量，并返回其节点
        constant = self.insert_tensor_constant(g, torch.ones(2, 16, 5, 4))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        constant_2 = self.insert_tensor_constant(g, torch.tensor([-1, 0, 0]))
        # 执行图操作 g_op，对 constant 和 constant_2 执行 Reshape 操作，并返回结果节点
        shape = g_op(g, "Reshape", constant, constant_2)
        # 在图形 g 上运行测试，检查 shape.node() 返回的节点是否满足期望的浮点张量形状 (8, 16, 5)
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(8, 16, 5)))

    # 定义测试函数 test_reshape_symbolic，用于测试带有符号尺寸的 Reshape 操作
    def test_reshape_symbolic(self):
        # 创建一个空图形 g
        g = self.create_empty_graph()
        # 向图形 g 中添加一个输入节点 input
        input = g.addInput()
        # 设置 input 的类型为带有动态尺寸 [None, None, 2, 8]
        input.setType(input.type().with_sizes([None, None, 2, 8]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        constant = self.insert_tensor_constant(g, torch.tensor([0, 0, -1]))
        # 执行图操作 g_op，对 input 和 constant 执行 Reshape 操作，并返回结果节点
        output = g_op(g, "Reshape", input, constant)
        # 在图形 g 上运行测试，检查 output.node() 返回的节点是否满足期望的张量形状 (None, None, 16)
        self.run_test(g, output.node(), expect_tensor(None, shape=(None, None, 16)))

    # 跳过不支持 Opset 版本 14 的测试函数修饰器
    @skipIfUnsupportedMinOpsetVersion(14)
    # 定义测试函数 test_reshape_allowzero，用于测试允许零尺寸的 Reshape 操作
    def test_reshape_allowzero(self):
        # 创建一个空图形 g
        g = self.create_empty_graph()
        # 向图形 g 中添加一个输入节点 input
        input = g.addInput()
        # 设置 input 的类型为带有动态尺寸 [3, 4, 0]
        input.setType(input.type().with_sizes([3, 4, 0]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        constant = self.insert_tensor_constant(g, torch.tensor([0, 4, 3]))
        # 执行图操作 g_op，对 input 和 constant 执行允许零尺寸的 Reshape 操作，并返回结果节点
        output = g_op(g, "Reshape", input, constant, allowzero_i=1)
        # 在图形 g 上运行测试，检查 output.node() 返回的节点是否满足期望的张量形状 (0, 4, 3)
        self.run_test(g, output.node(), expect_tensor(None, shape=(0, 4, 3)))

    # 定义测试函数 test_slice，用于测试 Slice 操作
    def test_slice(self):
        # 创建一个空图形 g
        g = self.create_empty_graph()
        # 向图形 g 中添加一个输入节点 input
        input = g.addInput()
        # 设置 input 的类型为带有动态尺寸 [None, None]
        input.setType(input.type().with_sizes([None, None]))
        # 向图形 g 中添加一个输入节点 start_input
        start_input = g.addInput()
        # 设置 start_input 的类型为带有动态尺寸 [None]
        start_input.setType(start_input.type().with_sizes([None]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        end = self.insert_tensor_constant(g, torch.tensor([3]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        axis = self.insert_tensor_constant(g, torch.tensor([0]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        step = self.insert_tensor_constant(g, torch.tensor([1]))
        # 执行图操作 g_op，对 input、start_input、end、axis 和 step 执行 Slice 操作，并返回结果节点
        slice = g_op(g, "Slice", input, start_input, end, axis, step)
        # 在图形 g 上运行测试，检查 slice.node() 返回的节点是否满足期望的张量形状 (None, None)
        self.run_test(g, slice.node(), expect_tensor(None, shape=(None, None)))

    # 定义测试函数 test_slice_with_dynamic_start_index，用于测试带有动态起始索引的 Slice 操作
    def test_slice_with_dynamic_start_index(self):
        # 创建一个空图形 g
        g = self.create_empty_graph()
        # 在图形 g 中插入一个形状为 (2, 3, 4, 5) 的全为 1 的常量张量，并返回其节点
        input = self.insert_tensor_constant(g, torch.ones(2, 3, 4, 5))
        # 向图形 g 中添加一个输入节点 start_input
        start_input = g.addInput()
        # 设置 start_input 的类型为带有动态尺寸 [2]
        start_input.setType(start_input.type().with_sizes([2]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        end = self.insert_tensor_constant(g, torch.tensor([3, 4]))
        # 在图形 g 中插入一个指定张量数据的常量张量，并返回其节点
        axis = self.insert_tensor_constant(g, torch.tensor([1, -1]))
        # 执行图操作 g_op，对 input、start_input、end 和 axis 执行 Slice 操作，并返回结果节点
        slice = g_op(g, "Slice", input, start_input, end, axis)
        # 在图
    def test_broadcast_matmul(self):
        # 创建一个空的计算图
        g = self.create_empty_graph()
        # 在计算图中插入一个形状为 (5, 1, 2) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        # 在计算图中插入一个形状为 (3, 1, 2, 1) 的全一张量，并返回对应的节点
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        # 执行 MatMul 操作，将 constant 和 constant_2 相乘，返回结果的形状信息
        shape = g_op(g, "MatMul", constant, constant_2)
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 5, 1, 1)))

        # 测试第一个输入张量为一维时的情况
        g = self.create_empty_graph()
        # 在计算图中插入一个形状为 (2,) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(2))
        # 再次插入一个形状为 (3, 1, 2, 1) 的全一张量，并返回对应的节点
        constant_2 = self.insert_tensor_constant(g, torch.ones(3, 1, 2, 1))
        # 执行 MatMul 操作，将 constant 和 constant_2 相乘，返回结果的形状信息
        shape = g_op(g, "MatMul", constant, constant_2)
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(3, 1, 1)))

        # 测试第二个输入张量为一维时的情况
        g = self.create_empty_graph()
        # 在计算图中插入一个形状为 (5, 1, 2) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(5, 1, 2))
        # 再次插入一个形状为 (2,) 的全一张量，并返回对应的节点
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        # 执行 MatMul 操作，将 constant 和 constant_2 相乘，返回结果的形状信息
        shape = g_op(g, "MatMul", constant, constant_2)
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, shape.node(), expect_tensor("Float", shape=(5, 1)))

        # 测试两个输入张量均为一维时的情况
        g = self.create_empty_graph()
        # 在计算图中插入一个形状为 (2,) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(2))
        # 再次插入一个形状为 (2,) 的全一张量，并返回对应的节点
        constant_2 = self.insert_tensor_constant(g, torch.ones(2))
        # 执行 MatMul 操作，将 constant 和 constant_2 相乘，返回结果的形状信息
        shape = g_op(g, "MatMul", constant, constant_2)
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, shape.node(), expect_tensor("Float", shape=()))

    def test_expand(self):
        # 创建一个空的计算图
        g = self.create_empty_graph()
        # 向计算图中添加一个输入节点
        input = g.addInput()
        # 在计算图中插入一个形状为 (2, 4) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(2, 4))
        # 设置输入节点的类型为与 constant 类型相同，但尺寸为任意大小的类型
        input.setType(constant.type().with_sizes([None, None]))
        # 执行 Shape 操作，获取输入节点的形状信息
        shape = g_op(g, "Shape", input)
        # 执行 Expand 操作，将 constant 根据 shape 的信息扩展，返回扩展后的节点
        expand = g_op(g, "Expand", constant, shape)
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, expand.node(), expect_tensor("Float", shape=(None, None)))

    def test_pad(self):
        # 创建一个空的计算图
        g = self.create_empty_graph()
        # 向计算图中添加一个输入节点
        input = g.addInput()
        # 设置输入节点的类型为 float 类型，尺寸为 [3, 320, 100] 的类型
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        # 在计算图中插入一个形状为 (6,) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        # 创建一个空节点
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        # 执行 Pad 操作，对输入节点进行填充，返回填充后的节点
        pad = g_op(g, "Pad", input, constant, none, mode_s="constant")
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(5, 322, 102)))

    def test_pad_with_dynamic_input_shape(self):
        # 创建一个空的计算图
        g = self.create_empty_graph()
        # 向计算图中添加一个输入节点
        input = g.addInput()
        # 设置输入节点的类型为 float 类型，尺寸为 [3, None, None] 的类型
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, None, None]))
        # 在计算图中插入一个形状为 (6,) 的全一张量，并返回对应的节点
        constant = self.insert_tensor_constant(g, torch.ones(6, dtype=torch.long))
        # 创建一个空节点
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        # 执行 Pad 操作，对输入节点进行填充，返回填充后的节点
        pad = g_op(g, "Pad", input, constant, none, mode_s="constant")
        # 运行测试，验证结果节点的形状是否符合预期
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(5, None, None)))
    # 定义一个测试函数，测试动态填充尺寸的情况
    def test_pad_with_dynamic_pad_size(self):
        # 创建一个空图形对象
        g = self.create_empty_graph()
        # 向图形中添加输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为浮点型，大小为 [3, 320, 100]
        input.setType(input.type().with_dtype(torch.float).with_sizes([3, 320, 100]))
        # 向图形中添加输入节点，用于动态填充大小
        pad_size = g.addInput()
        # 设置填充大小节点的数据类型为长整型，大小为 [6]
        pad_size.setType(pad_size.type().with_dtype(torch.long).with_sizes([6]))
        # 创建一个表示空值的节点
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        # 使用输入节点、填充大小节点和空值节点，执行填充操作
        pad = g_op(g, "Pad", input, pad_size, none, mode_s="constant")
        # 运行测试，并期望输出的张量形状为 (None, None, None)
        self.run_test(g, pad.node(), expect_tensor("Float", shape=(None, None, None)))

    # 定义一个测试函数，测试调整大小操作
    def test_resize(self):
        # 创建一个空图形对象
        g = self.create_empty_graph()
        # 向图形中添加输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为浮点型，大小为 [4, 32, 64, 64]
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        # 创建一个表示空值的节点
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        # 插入一个张量常数节点，表示调整大小的比例
        scales = self.insert_tensor_constant(
            g, torch.tensor([1, 1, 2, 2], dtype=torch.float)
        )
        # 使用输入节点、空值节点和比例节点，执行调整大小操作
        resize = g_op(
            g,
            "Resize",
            input,
            none,
            scales,
            coordinate_transformation_mode_s="align_corners",
            cubic_coeff_a_f=-0.75,
            mode_s="linear",
            nearest_mode_s="floor",
        )
        # 运行测试，并期望输出的张量形状为 (4, 32, 128, 128)
        self.run_test(g, resize.node(), expect_tensor("Float", shape=(4, 32, 128, 128)))

    # 定义一个测试函数，测试在连接操作之后进行调整大小操作
    def test_resize_after_concat(self):
        # 创建一个空图形对象
        g = self.create_empty_graph()
        # 向图形中添加输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为浮点型，大小为 [4, 32, 64, 64]
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 32, 64, 64]))
        # 创建一个表示空值的节点
        none = g_op(g, "prim::Constant").setType(torch.NoneType.get())
        # 插入一个张量常数节点，表示第一个调整大小的比例
        scale_1 = self.insert_tensor_constant(
            g, torch.tensor([1, 1], dtype=torch.float)
        )
        # 插入一个张量常数节点，表示第二个调整大小的比例
        scale_2 = self.insert_tensor_constant(
            g, torch.tensor([2, 2], dtype=torch.float)
        )
        # 使用连接节点合并两个比例节点
        scales = g_op(g, "Concat", scale_1, scale_2, axis_i=0)
        # 使用输入节点、空值节点和比例节点，执行调整大小操作
        resize = g_op(
            g,
            "Resize",
            input,
            none,
            scales,
            coordinate_transformation_mode_s="align_corners",
            cubic_coeff_a_f=-0.75,
            mode_s="linear",
            nearest_mode_s="floor",
        )
        # 运行测试，并期望输出的张量形状为 (4, 32, 128, 128)
        self.run_test(g, resize.node(), expect_tensor("Float", shape=(4, 32, 128, 128)))

    # 定义一个测试函数，测试沿指定轴进行的求积操作
    def test_reduce_prod_with_axes(self):
        # 创建一个空图形对象
        g = self.create_empty_graph()
        # 向图形中添加输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为长整型，大小为 [2]
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        # 使用输入节点和指定轴进行求积操作
        reduce_prod = g_op(g, "ReduceProd", input, axes_i=[0])
        # 运行测试，并期望输出的张量形状为 (1,)
        self.run_test(g, reduce_prod.node(), expect_tensor("Long", shape=(1,)))

    # 定义一个测试函数，测试在没有指定轴的情况下进行的求积操作
    def test_reduce_prod_without_axes(self):
        # 创建一个空图形对象
        g = self.create_empty_graph()
        # 向图形中添加输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为长整型，大小为 [2]
        input.setType(input.type().with_dtype(torch.long).with_sizes([2]))
        # 使用输入节点进行求积操作，未指定轴
        reduce_prod = g_op(g, "ReduceProd", input)
        # 运行测试，并期望输出的张量形状为 (1,)
        self.run_test(g, reduce_prod.node(), expect_tensor("Long", shape=(1,)))
    def test_proceeding_nodes_use_prim_pack_padded_output_dtype_correctly(self):
        # 创建一个空图
        g = self.create_empty_graph()
        # 添加一个输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为 torch.float，大小为 [4, 16]
        input.setType(input.type().with_dtype(torch.float).with_sizes([4, 16]))
        # 添加一个长度节点
        length = g.addInput()
        # 设置长度节点的数据类型为 torch.long，大小为 [4]
        length.setType(length.type().with_dtype(torch.long).with_sizes([4]))
        # 调用 g_op 函数执行 "prim::PackPadded" 操作，获取输出的 padded 和 batch_size
        padded, batch_size = g_op(g, "prim::PackPadded", input, length, outputs=2)
        # `prim::PackPadded` 只在跟踪模式下出现，因此其输出从跟踪图继承形状和数据类型
        padded.setType(padded.type().with_dtype(torch.float).with_sizes([None, None]))
        # 设置 batch_size 的数据类型为 torch.long，大小为 [None]
        batch_size.setType(batch_size.type().with_dtype(torch.long).with_sizes([None]))
        # `Gather` 操作应使用 `batch_size` 的数据类型作为其输出的数据类型
        # 插入一个常量张量作为 gather 操作的索引
        gather_idx = self.insert_tensor_constant(g, torch.tensor([0], dtype=torch.long))
        # 调用 g_op 函数执行 "Gather" 操作，获取 gather 的输出
        gather = g_op(g, "Gather", batch_size, gather_idx, axis_i=0)
        # 运行测试，验证 gather 节点的输出是否符合预期的张量形状和数据类型
        self.run_test(g, gather.node(), expect_tensor("Long", shape=(None,)))

    def test_squeeze_after_dynamic_if(self):
        # 导入 squeeze11 函数
        from torch.onnx.symbolic_opset11 import squeeze as squeeze11
        # 创建一个空图
        g = self.create_empty_graph()
        # 添加一个输入节点
        input = g.addInput()
        # 设置输入节点的数据类型为 torch.float，大小为 [1, None, 5]
        input.setType(input.type().with_dtype(torch.float).with_sizes([1, None, 5]))
        
        # 条件节点的数据类型故意不是 bool，以测试添加的 "Cast" 节点不会影响形状推断
        cond = g.addInput()
        # 设置条件节点的数据类型为 torch.int32，大小为 [1]
        cond.setType(input.type().with_dtype(torch.int32).with_sizes([1]))
        
        # 添加带块的 "If" 操作，返回 if_op、if_context、else_context 和 new_node
        if_op, (if_context, else_context), new_node = jit_utils.add_op_with_blocks(
            as_graphcontext(g), "If", cond, n_blocks=2
        )
        
        # 在 if_context 中执行 "Add" 操作，输出为 block1_output
        block1_output = if_context.op("Add", input, input)
        # 在 else_context 中执行 "Identity" 操作，输出为 block2_output
        block2_output = else_context.op("Identity", input)
        
        # 将 block1_output 和 block2_output 添加到各自的块中
        utils._add_output_to_block(if_context.block, block1_output)
        utils._add_output_to_block(else_context.block, block2_output)
        
        # 调整 if_output 以修复 ONNX 控制流节点
        if_output = torch._C._jit_pass_fixup_onnx_controlflow_node(
            new_node, _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        )[0]
        
        # 执行 ONNX 节点形状和类型推断
        torch._C._jit_pass_onnx_node_shape_type_inference(
            new_node, {}, _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        )
        
        # 如果导出器无法确定要挤压的维度是否为大小为 1，则导出器将添加 "If" 而不是原始的 "Squeeze"
        squeezed = squeeze11(as_graphcontext(g), if_output, dim=0)
        # 断言挤压后的节点类型为 "onnx::Squeeze"
        assert squeezed.node().kind() == "onnx::Squeeze"
        # 运行测试，验证 squeezed 节点的输出是否符合预期的张量形状和数据类型
        self.run_test(g, squeezed.node(), expect_tensor("Float", shape=(None, 5)))
class TestONNXCustomOpShapeInference(pytorch_test_common.ExportTestCase):
    # 定义测试类，继承自ExportTestCase用于ONNX导出测试

    def setUp(self):
        # 设置测试环境
        super().setUp()
        self.opset_version = _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET
        # 设置ONNX导出器的操作集版本号为最大值

    def test_setType_maintains_output_shape_for_single_custom_op(self):
        # 测试单个自定义操作的输出形状保持
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)
        # 添加清理函数，用于取消注册自定义操作符号"::linalg_inv"，版本号为9

        class CustomInverse(torch.nn.Module):
            # 定义自定义逆操作的模块
            def forward(self, x):
                return torch.inverse(x) + x
                # 返回输入张量的逆加上自身的结果

        def linalg_inv_settype(g, self):
            # 自定义函数，用于设置类型
            return g.op("com.microsoft::Inverse", self).setType(self.type())
            # 在计算图g上添加"com.microsoft::Inverse"操作，并设置其类型为self.type()

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 9)
        # 注册自定义操作符号"::linalg_inv"，使用linalg_inv_settype函数，版本号为9
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        # 创建一个字节流对象f

        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )
        # 使用ONNX导出模块model的ONNX表示到字节流f，指定操作集版本和自定义操作集

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        # 从字节流f的内容加载ONNX模型协议
        model_value_info = model_proto.graph.value_info
        # 获取模型图中的值信息
        self.assertIsNotNone(model_value_info)
        # 断言值信息不为空
        assert model_value_info
        dims = model_value_info[0].type.tensor_type.shape.dim
        # 获取第一个值信息的张量类型形状维度信息

        for i in range(len(dims)):
            # 遍历维度信息
            # 如果节点输出具有形状信息，则应该有dim_value
            # 否则，具有带有动态形状的dim_param
            self.assertTrue(dims[i].HasField("dim_value"))

        for dim, rank in zip(dims, x.size()):
            # 遍历维度和输入张量的尺寸
            self.assertEqual(dim.dim_value, rank)
            # 断言维度的dim_value与尺寸相等

    def test_no_setType_for_single_custom_op(self):
        # 测试单个自定义操作没有设置类型
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)
        # 添加清理函数，用于取消注册自定义操作符号"::linalg_inv"，版本号为9

        class CustomInverse(torch.nn.Module):
            # 定义自定义逆操作的模块
            def forward(self, x):
                return torch.inverse(x) + x
                # 返回输入张量的逆加上自身的结果

        def linalg_inv_no_settype(g, self):
            # 自定义函数，用于没有设置类型
            return g.op("com.microsoft::Inverse", self)
            # 在计算图g上添加"com.microsoft::Inverse"操作，不设置类型

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_no_settype, 9)
        # 注册自定义操作符号"::linalg_inv"，使用linalg_inv_no_settype函数，版本号为9
        model = CustomInverse()
        x = torch.randn(2, 3, 3)
        f = io.BytesIO()
        # 创建一个字节流对象f

        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
        )
        # 使用ONNX导出模块model的ONNX表示到字节流f，指定操作集版本和自定义操作集

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        # 从字节流f的内容加载ONNX模型协议
        model_value_info = model_proto.graph.value_info
        # 获取模型图中的值信息
        self.assertIsNotNone(model_value_info)
        # 断言值信息不为空
        assert model_value_info
        dims = model_value_info[0].type.tensor_type.shape.dim
        # 获取第一个值信息的张量类型形状维度信息

        for i in range(len(dims)):
            # 遍历维度信息
            # 如果节点输出具有形状信息，则应该有dim_value
            # 否则，具有带有动态形状的dim_param
            self.assertTrue(dims[i].HasField("dim_param"))
    ):
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)
        # 在测试结束时，注册清理函数以取消自定义操作符 "::linalg_inv"

        class CustomInverse(torch.nn.Module):
            # 定义一个自定义的 PyTorch 模块，实现反向操作
            def forward(self, x):
                return torch.inverse(x) + x

        def linalg_inv_settype(g, self):
            # 定义一个函数，设置自定义操作的类型
            return g.op("com.microsoft::Inverse", self).setType(
                self.type().with_dtype(torch.float).with_sizes([None, 3, 3])
            )

        torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 9)
        # 注册自定义的 ONNX 符号操作 "::linalg_inv"，关联到类型设置函数 linalg_inv_settype，并指定 opset 版本为 9

        model = CustomInverse()
        # 创建一个 CustomInverse 模型实例

        x = torch.randn(2, 3, 3)
        # 生成一个形状为 (2, 3, 3) 的随机张量 x

        f = io.BytesIO()
        # 创建一个字节流对象 f

        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=self.opset_version,
            custom_opsets={"com.microsoft": 1},
            input_names=["x"],
            dynamic_axes={"x": {0: "batch"}},
        )
        # 将模型 model 导出为 ONNX 格式，存储到字节流 f 中，设置 opset 版本和自定义操作集合

        model_proto = onnx.load(io.BytesIO(f.getvalue()))
        # 从导出的字节流内容加载 ONNX 模型 proto

        model_value_info = model_proto.graph.value_info
        # 获取 ONNX 模型的值信息

        self.assertIsNotNone(model_value_info)
        # 断言模型值信息不为空

        assert model_value_info
        # 使用断言确认模型值信息存在

        dims = model_value_info[0].type.tensor_type.shape.dim
        # 获取模型输出的维度信息列表

        # The first axe should be dynamic as we defined when exporting
        # 第一个维度应该是动态的，正如我们在导出时定义的一样
        self.assertTrue(dims[0].HasField("dim_param"))

        for i in range(1, len(dims)):
            # If node output has shape info, it should have dim_value
            # Otherwise, it has dim_params with dynamic shape
            # 如果节点输出具有形状信息，则应具有 dim_value
            # 否则，它具有具有动态形状的 dim_params
            self.assertTrue(dims[i].HasField("dim_value"))
            self.assertEqual(dims[i].dim_value, x.size()[i])
    # 注册清理函数，用于在测试结束时注销自定义操作符 "::linalg_inv" 的符号处理器
    self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "::linalg_inv", 9)

    # 定义自定义逆操作的类，继承自 torch.nn.Module
    class CustomInverse(torch.nn.Module):
        def forward(self, x, y, z):
            # 计算输入张量 x 的逆
            x = torch.inverse(x)
            # 返回逆张量 x 与 y、z 的和
            return x + y + z

    # 自定义符号处理器函数，用于设置操作的类型信息
    def linalg_inv_settype(g, self):
        return g.op("com.microsoft::Inverse", self).setType(
            # 设置张量类型，包括数据类型为 torch.float 和大小为 [2, 3, 10, 10]
            self.type().with_dtype(torch.float).with_sizes([2, 3, 10, 10])
        )

    # 注册自定义操作符 "::linalg_inv" 的符号处理器函数，版本号为 9
    torch.onnx.register_custom_op_symbolic("::linalg_inv", linalg_inv_settype, 9)
    # 创建 CustomInverse 类的实例
    model = CustomInverse()
    # 生成随机张量 x、y、z，形状为 [2, 3, 10, 10]
    x = torch.randn(2, 3, 10, 10)
    y = torch.randn(2, 3, 10, 10)
    z = torch.randn(2, 3, 10, 10)
    # 创建一个字节流对象
    f = io.BytesIO()
    # 将模型导出为 ONNX 格式到字节流 f，指定操作集版本和自定义操作集
    torch.onnx.export(
        model,
        (x, y, z),
        f,
        opset_version=self.opset_version,
        custom_opsets={"com.microsoft": 1},
    )

    # 从导出的 ONNX 模型中加载模型定义
    model_proto = onnx.load(io.BytesIO(f.getvalue()))
    # 查找逆操作的输出名称，用于验证其形状信息
    output_name = ""
    for node in model_proto.graph.node:
        if node.op_type == "Inverse":
            output_name = node.output[0]
            break
    # 断言确保找到了输出名称
    assert output_name
    # 获取模型的值信息
    model_value_info = model_proto.graph.value_info
    self.assertIsNotNone(model_value_info)
    assert model_value_info
    # 遍历值信息，查找与输出名称匹配的值信息
    for value_info in model_value_info:
        assert value_info.name
        if value_info.name == output_name:
            # 获取张量的维度信息
            dims = value_info.type.tensor_type.shape.dim
            for i in range(len(dims)):
                # 断言节点输出有形状信息，应该包含 dim_value
                # 否则，使用 dim_params 表示动态形状
                self.assertTrue(dims[i].HasField("dim_value"))
            # 检查维度与输入张量 x 的维度一致
            for dim, rank in zip(dims, x.size()):
                self.assertEqual(dim.dim_value, rank)
# 如果这个模块是作为主程序运行的话，执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于运行测试
    common_utils.run_tests()
```