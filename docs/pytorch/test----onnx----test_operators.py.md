# `.\pytorch\test\onnx\test_operators.py`

```py
# Owner(s): ["module: onnx"]

"""
Usage: python test/onnx/test_operators.py [--no-onnx] [--produce-onnx-test-data]
          --no-onnx: no onnx python dependency
          --produce-onnx-test-data: generate onnx test data
          --accept: accept onnx updates and overwrite models
"""

# 导入所需的标准库和模块
import glob
import inspect
import io
import itertools
import operator
import os
import shutil
import tempfile

# 导入单元测试框架
import unittest

# 从自定义模块导入常用测试工具和变量
from pytorch_test_common import (
    BATCH_SIZE,
    flatten,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
)

# 导入PyTorch相关模块和函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.autograd import Function, Variable
from torch.nn import functional, Module
from torch.onnx._internal import diagnostics
from torch.onnx.symbolic_helper import (
    _get_tensor_dim_size,
    _get_tensor_sizes,
    parse_args,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack

# 设置单元测试最大差异为无限制
unittest.TestCase.maxDiff = None

_onnx_test = False  # 标志是否生成onnx测试用例
_onnx_dep = True  # 标志是否导入onnx包


def export_to_pbtxt(model, inputs, *args, **kwargs):
    """
    将模型导出为可读性较好的PBtxt格式字符串

    Args:
        model: 要导出的PyTorch模型
        inputs: 输入的示例数据
        *args, **kwargs: 其他参数传递给torch.onnx.export_to_pretty_string函数

    Returns:
        str: PBtxt格式的字符串表示
    """
    return torch.onnx.export_to_pretty_string(
        model, inputs, *args, google_printer=True, **kwargs
    )


def export_to_pb(model, inputs, *args, **kwargs):
    """
    将模型导出为PB格式的字节流

    Args:
        model: 要导出的PyTorch模型
        inputs: 输入的示例数据
        *args, **kwargs: 其他参数传递给torch.onnx.export函数

    Returns:
        bytes: PB格式的模型字节流
    """
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class FuncModule(Module):
    """
    定义一个函数模块，用于封装给定函数和参数

    Args:
        f: 要封装的函数
        params: 函数的参数列表，默认为None

    Attributes:
        f: 封装的函数
        params: 参数列表，作为模块的可学习参数
    """

    def __init__(self, f, params=None):
        if params is None:
            params = ()
        super().__init__()
        self.f = f
        self.params = nn.ParameterList(list(params))

    def forward(self, *args):
        """
        模块的前向传播方法

        Args:
            *args: 可变长度的输入参数

        Returns:
            Tensor: 前向传播的结果
        """
        return self.f(*itertools.chain(args, self.params))


class TestOperators(common_utils.TestCase):
    """
    定义测试操作符的单元测试类，继承自common_utils.TestCase
    """

    def setUp(self):
        """
        单元测试初始化方法，在每个测试方法执行前调用
        """
        super().setUp()
        diagnostics.engine.clear()
    # 定义一个名为 assertONNX 的方法，用于验证导出的 ONNX 模型
    def assertONNX(self, f, args, params=None, **kwargs):
        # 如果参数 params 未提供，则设为空元组
        if params is None:
            params = ()
        # 如果 f 是 nn.Module 类型，则直接赋值给 m；否则，使用 FuncModule 类创建 m
        if isinstance(f, nn.Module):
            m = f
        else:
            m = FuncModule(f, params)
        # 将模型设为评估模式
        m.eval()
        # 导出模型为 ONNX 的文本格式
        onnx_model_pbtxt = export_to_pbtxt(m, args, **kwargs)
        # 获取可选参数 subname 的值
        subname = kwargs.pop("subname", None)
        # 调用 assertExpected 方法，验证导出的 ONNX 模型文本
        self.assertExpected(onnx_model_pbtxt, subname)
        
        # 如果 _onnx_dep 可用，则执行以下操作
        if _onnx_dep:
            # 导出模型为 ONNX 的二进制格式
            onnx_model_pb = export_to_pb(m, args, **kwargs)
            import onnx
            import onnx.checker
            import onnx.numpy_helper
            import onnx_test_common

            # 将二进制数据解析为 onnx.ModelProto 对象
            model_def = onnx.ModelProto.FromString(onnx_model_pb)
            # 检查 ONNX 模型的有效性
            onnx.checker.check_model(model_def)
            
            # 如果 _onnx_test 可用，则执行以下操作
            if _onnx_test:
                # 获取调用该方法的测试函数名
                test_function = inspect.stack()[1][0].f_code.co_name
                test_name = test_function[0:4] + "_operator" + test_function[4:]
                # 构建输出目录路径
                output_dir = os.path.join(
                    onnx_test_common.pytorch_operator_dir, test_name
                )
                # 假设：
                #     1) 在执行测试之前应删除旧的测试结果。
                #     2) 每个测试只能有一个 assertONNX，否则将覆盖数据。
                # 检查输出目录是否不存在，如果存在则抛出异常
                assert not os.path.exists(output_dir), f"{output_dir} should not exist!"
                # 创建输出目录
                os.makedirs(output_dir)
                # 将模型序列化为文件存储到 output_dir 目录下的 model.onnx 文件中
                with open(os.path.join(output_dir, "model.onnx"), "wb") as file:
                    file.write(model_def.SerializeToString())
                # 创建数据目录
                data_dir = os.path.join(output_dir, "test_data_set_0")
                os.makedirs(data_dir)
                # 如果 args 是 Variable 类型，则转换为元组
                if isinstance(args, Variable):
                    args = (args,)
                # 遍历展平后的参数列表，将每个参数保存为对应的 input_{index}.pb 文件
                for index, var in enumerate(flatten(args)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(
                        os.path.join(data_dir, f"input_{index}.pb"), "wb"
                    ) as file:
                        file.write(tensor.SerializeToString())
                # 执行模型并获取输出
                outputs = m(*args)
                # 如果输出是 Variable 类型，则转换为元组
                if isinstance(outputs, Variable):
                    outputs = (outputs,)
                # 遍历展平后的输出列表，将每个输出保存为对应的 output_{index}.pb 文件
                for index, var in enumerate(flatten(outputs)):
                    tensor = onnx.numpy_helper.from_array(var.data.numpy())
                    with open(
                        os.path.join(data_dir, f"output_{index}.pb"), "wb"
                    ) as file:
                        file.write(tensor.SerializeToString())
    # 定义一个方法，用于断言导出 ONNX 模型时是否抛出指定错误并匹配正则表达式
    def assertONNXRaisesRegex(self, err, reg, f, args, params=None, **kwargs):
        # 如果未提供参数 params，则将其设置为空元组
        if params is None:
            params = ()
        # 如果参数 f 是 nn.Module 类型，则直接使用 f 作为模型 m
        if isinstance(f, nn.Module):
            m = f
        # 否则，使用 FuncModule 将函数 f 和参数 params 包装成一个模型 m
        else:
            m = FuncModule(f, params)
        # 使用 assertRaisesRegex 上下文管理器断言在执行 export_to_pbtxt 函数时是否抛出指定错误并匹配正则表达式
        with self.assertRaisesRegex(err, reg):
            export_to_pbtxt(m, args, **kwargs)

    # 定义测试基本的 ONNX 导出功能
    def test_basic(self):
        # 创建一个张量 x，要求其梯度追踪
        x = torch.tensor([0.4], requires_grad=True)
        # 创建一个张量 y，要求其梯度追踪
        y = torch.tensor([0.7], requires_grad=True)
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 (x, y)
        self.assertONNX(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), (x, y))

    # 定义测试张量视图操作的 ONNX 导出功能
    def test_view(self):
        # 创建一个张量 x，要求其梯度追踪
        x = torch.tensor([0.0], requires_grad=True)
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 x
        self.assertONNX(lambda x: x.view(1, 1), x)

    # 定义测试张量索引操作的 ONNX 导出功能
    def test_index(self):
        # 创建一个张量 x，要求其梯度追踪
        x = torch.tensor([[0.0]], requires_grad=True)
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 x
        self.assertONNX(lambda x: x[0], x)

    # 定义测试张量类型转换操作的 ONNX 导出功能
    def test_type_as(self):
        # 创建一个张量 x，要求其梯度追踪
        x = torch.tensor([0.0], requires_grad=True)
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 x
        self.assertONNX(lambda x: x.type_as(x), x)

    # 定义测试张量加常数操作的 ONNX 导出功能
    def test_addconstant(self):
        # 创建一个形状为 (2, 3) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(2, 3, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 x
        self.assertONNX(lambda x: x + 1, x)

    # 定义测试张量加法操作的 ONNX 导出功能（涉及广播）
    def test_add_broadcast(self):
        # 创建一个形状为 (2, 3) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(2, 3, requires_grad=True).double()
        # 创建一个形状为 (3,) 的随机张量 y，要求其梯度追踪，数据类型为 double
        y = torch.randn(3, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 operator.add 函数，并传入参数 (x, y)
        self.assertONNX(operator.add, (x, y))

    # 定义测试张量加法操作的 ONNX 导出功能（左操作数涉及广播）
    def test_add_left_broadcast(self):
        # 创建一个形状为 (3,) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(3, requires_grad=True).double()
        # 创建一个形状为 (2, 3) 的随机张量 y，要求其梯度追踪，数据类型为 double
        y = torch.randn(2, 3, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 operator.add 函数，并传入参数 (x, y)
        self.assertONNX(operator.add, (x, y))

    # 定义测试张量加法操作的 ONNX 导出功能（右操作数形状为 (2, 1) 的广播）
    def test_add_size1_broadcast(self):
        # 创建一个形状为 (2, 3) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(2, 3, requires_grad=True).double()
        # 创建一个形状为 (2, 1) 的随机张量 y，要求其梯度追踪，数据类型为 double
        y = torch.randn(2, 1, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 operator.add 函数，并传入参数 (x, y)
        self.assertONNX(operator.add, (x, y))

    # 定义测试张量加法操作的 ONNX 导出功能（右操作数形状为 (1, 3) 的广播）
    def test_add_size1_right_broadcast(self):
        # 创建一个形状为 (2, 3) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(2, 3, requires_grad=True).double()
        # 创建一个形状为 (3,) 的随机张量 y，要求其梯度追踪，数据类型为 double
        y = torch.randn(3, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 operator.add 函数，并传入参数 (x, y)
        self.assertONNX(operator.add, (x, y))

    # 定义测试张量加法操作的 ONNX 导出功能（右操作数形状为 (1, 3) 的广播）
    def test_add_size1_singleton_broadcast(self):
        # 创建一个形状为 (2, 3) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(2, 3, requires_grad=True).double()
        # 创建一个形状为 (1, 3) 的随机张量 y，要求其梯度追踪，数据类型为 double
        y = torch.randn(1, 3, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 operator.add 函数，并传入参数 (x, y)
        self.assertONNX(operator.add, (x, y))

    # 定义测试张量右子操作数形式的 ONNX 导出功能（涉及减法）
    def test_rsub(self):
        # 创建一个形状为 (2, 3) 的随机张量 x，要求其梯度追踪，数据类型为 double
        x = torch.randn(2, 3, requires_grad=True).double()
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 (x,)
        self.assertONNX(lambda x: 1 - x, (x,))

    # 定义测试张量逐元素乘法操作的 ONNX 导出功能（布尔张量）
    def test_mul_bool(self):
        # 创建一个布尔类型张量 x
        x = torch.tensor([True, False, True, False])
        # 创建一个布尔类型张量 y
        y = torch.tensor([True, True, False, False])
        # 断言导出 ONNX 模型，使用 lambda 表达式定义模型，并传入参数 (x, y)
        self.assertONNX(lambda x, y: torch.mul(x, y), (x, y))
    # 定义一个测试方法，测试 torch.split 函数的功能
    def test_split(self):
        # 创建一个2x6的张量
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        # 调用自定义的 assertONNX 方法，验证 torch.split 在指定维度上的输出是否符合预期
        self.assertONNX(lambda x: torch.split(x, 2, 1), x)

    # 定义另一个测试方法，测试 torch.split 函数在给定大小列表的情况下的功能
    def test_split_with_sizes(self):
        # 创建一个2x6的张量
        x = torch.tensor(
            [[0.0, 1.0, 1.0, 0.0, 2.0, 2.0], [2.0, 3.0, 3.0, 2.0, 1.0, 1.0]]
        )
        # 调用自定义的 assertONNX 方法，验证 torch.split 在指定维度上以给定大小列表分割的输出是否符合预期
        self.assertONNX(lambda x: torch.split(x, [2, 1, 3], 1), x)

    # 定义测试方法，测试 torch.cat 函数的功能
    def test_concat2(self):
        # 创建两个2x3的随机张量
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        # 调用自定义的 assertONNX 方法，验证 torch.cat 在指定维度上连接输入张量的输出是否符合预期
        self.assertONNX(lambda inputs: torch.cat(inputs, 1), ((x, y),))

    # 定义测试方法，测试 torch.mm 函数的功能
    def test_mm(self):
        # 创建两个随机张量，分别是 2x3 和 3x4 的
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        # 调用自定义的 assertONNX 方法，验证 torch.mm 函数的输出是否符合预期
        self.assertONNX(torch.mm, (m1, m2))

    # 定义测试方法，测试 torch.addmm 函数的功能
    def test_addmm(self):
        # 创建三个随机张量，分别是 2x3、3x4 和 4x 的
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(3, 4, requires_grad=True)
        m3 = torch.randn(4, requires_grad=True)
        # 调用自定义的 assertONNX 方法，验证 torch.addmm 函数的输出是否符合预期
        self.assertONNX(
            lambda x, y, z: torch.addmm(torch.addmm(z, x, y), x, y), (m1, m2, m3)
        )

    # 定义测试方法，测试 torch.permute 函数的功能
    def test_permute2(self):
        # 创建一个1x1x1x1x1x1的张量
        x = torch.tensor([[[[[[0.0]]]]]], requires_grad=True)
        # 调用自定义的 assertONNX 方法，验证 torch.permute 函数的输出是否符合预期
        self.assertONNX(lambda x: x.permute(0, 1, 4, 2, 5, 3), x)

    # 定义测试方法，测试 nn.ReflectionPad2d 函数的功能
    def test_pad(self):
        # 创建一个1x2x1x4的张量
        x = torch.tensor(
            [[[[0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 7.0, 7.0]]]], requires_grad=True
        )
        # 调用自定义的 assertONNX 方法，验证 nn.ReflectionPad2d 函数的输出是否符合预期
        self.assertONNX(nn.ReflectionPad2d((2, 3, 0, 1)), x)

    # 定义测试方法，测试 torch.sigmoid 和 torch.tanh 函数组合使用的功能
    def test_params(self):
        # 创建一个2x2的张量
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        # 创建一个 nn.Parameter 类型的张量，形状与 x 相同
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        # 调用自定义的 assertONNX 方法，验证组合函数的输出是否符合预期
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            x,
            params=(y,),
            keep_initializers_as_inputs=True,
        )

    # 定义测试方法，测试 torch.sigmoid 和 torch.tanh 函数组合使用的功能，并关闭 keep_initializers_as_inputs
    def test_params_onnx_irv4(self):
        # 创建一个2x2的张量
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        # 创建一个 nn.Parameter 类型的张量，形状与 x 相同
        y = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
        # 调用自定义的 assertONNX 方法，验证组合函数的输出是否符合预期，并关闭 keep_initializers_as_inputs
        self.assertONNX(
            lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))),
            x,
            params=(y,),
            keep_initializers_as_inputs=False,
        )

    # 定义测试方法，测试 MyFun 类的 forward 方法
    def test_symbolic_mismatch(self):
        # 定义一个自定义的 Function 类 MyFun
        class MyFun(Function):
            @staticmethod
            def symbolic(g, x):
                # 该函数内部不应被调用，因为我们会因为参数不匹配而失败
                raise AssertionError

            @staticmethod
            def forward(ctx, x, y):
                return x + y

        # 创建两个2x2的张量
        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        # 使用 assertRaisesRegex 验证在导出到 pbtxt 时是否会抛出类型错误
        with self.assertRaisesRegex(TypeError, "occurred when translating MyFun"):
            export_to_pbtxt(FuncModule(MyFun().apply), (x, y))

    # 提示：这里是一个 TODO，计划进行一个 nn 风格的测试
    # 测试批量归一化（Batch Normalization）在 ONNX 导出中的功能
    def test_batchnorm(self):
        # 创建一个大小为 (2, 2, 2, 2) 的张量 x，要求梯度计算
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        # 使用自定义函数 assertONNX 测试 nn.BatchNorm2d 在给定输入 x 下的导出情况，
        # keep_initializers_as_inputs=True 表示保持初始化器作为输入
        self.assertONNX(nn.BatchNorm2d(2), x, keep_initializers_as_inputs=True)

    # 测试批量归一化在 ONNX IR v4 导出中的功能
    def test_batchnorm_onnx_irv4(self):
        # 创建一个大小为 (2, 2, 2, 2) 的张量 x，要求梯度计算
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        # 使用自定义函数 assertONNX 测试 nn.BatchNorm2d 在给定输入 x 下的导出情况

    # 测试一维批量归一化（Batch Normalization）在 ONNX 导出中的功能
    def test_batchnorm_1d(self):
        # 创建一个大小为 (2, 2) 的张量 x，要求梯度计算
        x = torch.ones(2, 2, requires_grad=True)
        # 使用自定义函数 assertONNX 测试 nn.BatchNorm1d 在给定输入 x 下的导出情况，
        # keep_initializers_as_inputs=True 表示保持初始化器作为输入

    # 测试训练模式下批量归一化在 ONNX 导出中的功能
    def test_batchnorm_training(self):
        # 创建一个大小为 (2, 2, 2, 2) 的张量 x，要求梯度计算
        x = torch.ones(2, 2, 2, 2, requires_grad=True)
        # 使用自定义函数 assertONNX 测试 nn.BatchNorm2d 在给定输入 x 下的导出情况，
        # training=torch.onnx.TrainingMode.TRAINING 表示将模型导出为训练模式，
        # keep_initializers_as_inputs=True 表示保持初始化器作为输入

    # 测试卷积层在 ONNX 导出中的功能
    def test_conv(self):
        # 创建一个大小为 (20, 16, 50, 40) 的张量 x，要求梯度计算
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        # 使用自定义函数 assertONNX 测试 nn.Conv2d 在给定输入 x 下的导出情况，
        # keep_initializers_as_inputs=True 表示保持初始化器作为输入

    # 测试卷积层在 ONNX IR v4 导出中的功能
    def test_conv_onnx_irv4(self):
        # 创建一个大小为 (20, 16, 50, 40) 的张量 x，要求梯度计算
        x = torch.ones(20, 16, 50, 40, requires_grad=True)
        # 使用自定义函数 assertONNX 测试 nn.Conv2d 在给定输入 x 下的导出情况

    # 测试卷积层在 ONNX IR v4 的 opset 8 下导出的功能
    def test_conv_onnx_irv4_opset8(self):
        # 创建一个大小为 (1, 2, 5, 7) 的张量 x，要求梯度计算
        x = torch.ones(1, 2, 5, 7, requires_grad=True)
        # 创建一个 nn.Conv2d 对象 conv_node，设置卷积核权重为全 1
        conv_node = nn.Conv2d(2, 4, 3, bias=False)
        conv_node.weight.data.fill_(1.0)
        # 使用自定义函数 assertONNX 测试 conv_node 在给定输入 x 下的导出情况，
        # opset_version=8 表示使用 ONNX opset 8 版本，
        # keep_initializers_as_inputs=False 表示不保持初始化器作为输入
    # 测试可变长度输入的卷积神经网络转换
    def test_conv_variable_length(self):
        # 创建一个大小为 (5, 3, 6, 6) 的张量，所有值为1，设置梯度计算为True
        x = torch.ones(5, 3, 6, 6, requires_grad=True)
        # 创建一个输入通道数为3，输出通道数为2，卷积核大小为3的二维卷积层模型
        model = torch.nn.Conv2d(3, 2, 3)

        # 定义动态轴的字典，指定输入的维度会动态变化，输出的维度也有自定义的变量名
        dynamic_axes = {
            "input_1": [0, 2, 3],
            "output_1": {0: "output_1_variable_dim_0", 1: "output_1_variable_dim_1"},
        }
        # 创建一个临时文件对象来保存模型的protobuf文件
        model_proto_file = tempfile.NamedTemporaryFile()
        # 将 PyTorch 模型导出为 ONNX 格式
        torch.onnx.export(
            model,
            x,
            model_proto_file.name,
            verbose=True,  # 输出详细信息以便调试
            input_names=["input_1"],  # 指定输入的名称
            output_names=["output_1"],  # 指定输出的名称
            dynamic_axes=dynamic_axes,  # 指定动态轴
        )

        # 导入 ONNX 库
        import onnx

        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(model_proto_file.name)
        # 检查模型的有效性
        onnx.checker.check_model(onnx_model)

        # 断言默认的动态轴名称在未提供自定义名称时被生成
        assert (
            onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
            == "input_1_dynamic_axes_1"
        )
        assert (
            onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_param
            == "input_1_dynamic_axes_2"
        )
        assert (
            onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_param
            == "input_1_dynamic_axes_3"
        )

        # 断言当提供了自定义名称时，它们被正确应用
        assert (
            onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param
            == "output_1_variable_dim_0"
        )
        assert (
            onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_param
            == "output_1_variable_dim_1"
        )
    def test_at_op(self):
        # 创建一个3行4列的张量x，其中的元素服从标准正态分布
        x = torch.randn(3, 4)

        # 定义一个继承自Function的自定义函数MyFun
        class MyFun(Function):
            @staticmethod
            # 定义静态方法symbolic，接收计算图g和输入张量x，返回g中的"add"操作结果
            def symbolic(g, x):
                return g.at("add", x, x)

            @staticmethod
            # 定义静态方法forward，接收上下文对象ctx和输入张量x，返回x加上自身的结果
            def forward(ctx, x):
                return x + x

        # 定义一个继承自Module的自定义模块MyModule
        class MyModule(Module):
            # 定义forward方法，接收输入张量x，返回MyFun的apply方法应用在x上的结果
            def forward(self, x):
                return MyFun.apply(x)

        # 使用self.assertONNX方法验证导出的ONNX模型
        self.assertONNX(
            MyModule(),
            x,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    def test_clip(self):
        # 创建一个形状为(3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(3, 4, requires_grad=True)
        # 使用lambda表达式和torch.clamp函数对输入张量x进行限制在[-0.5, 0.5]之间，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.clamp(x, min=-0.5, max=0.5), x)

    def test_clip_min(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和张量的clamp方法对输入张量x进行下限限制在-0.1，并验证导出的ONNX模型
        self.assertONNX(lambda x: x.clamp(min=-0.1), x)

    def test_clip_max(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和张量的clamp方法对输入张量x进行上限限制在0.1，并验证导出的ONNX模型
        self.assertONNX(lambda x: x.clamp(max=0.1), x)

    def test_hardtanh(self):
        # 创建一个形状为(3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(3, 4, requires_grad=True)
        # 使用lambda表达式和torch.nn.Hardtanh类对输入张量x进行硬切割在[-0.5, 0.5]之间，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.nn.Hardtanh(-0.5, 0.5)(x), x)

    def test_full(self):
        # 创建一个形状为(3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(3, 4, requires_grad=True)
        # 使用lambda表达式和torch.full函数创建一个与输入张量x相同形状的张量，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.full(x.shape, 2.0), x)

    def test_full_like(self):
        # 创建一个形状为(3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(3, 4, requires_grad=True)
        # 使用lambda表达式和torch.full_like函数创建一个与输入张量x相同形状的张量，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.full_like(x, 2), x)

    def test_max(self):
        # 创建两个形状为(3, 4)的张量x和y，元素服从标准正态分布，并要求它们的梯度
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        # 使用lambda表达式和torch.max函数对输入张量x和y进行逐元素比较，并验证导出的ONNX模型
        self.assertONNX(lambda x, y: torch.max(x, y), (x, y))

    def test_min(self):
        # 创建两个形状为(3, 4)的张量x和y，元素服从标准正态分布，并要求它们的梯度
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=True)
        # 使用lambda表达式和torch.min函数对输入张量x和y进行逐元素比较，并验证导出的ONNX模型
        self.assertONNX(lambda x, y: torch.min(x, y), (x, y))

    def test_mean(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和torch.mean函数计算输入张量x的平均值，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.mean(x), x)

    def test_reduced_mean(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和torch.mean函数计算输入张量x在第2维上的平均值，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.mean(x, dim=2), x)

    def test_reduced_mean_keepdim(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和torch.mean函数计算输入张量x在第(2, 3)维上的平均值，并保持维度不变，验证导出的ONNX模型
        self.assertONNX(lambda x: torch.mean(x, dim=(2, 3), keepdim=True), x)

    def test_mean_dtype(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和torch.mean函数计算输入张量x的平均值，结果张量的数据类型为torch.double，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.mean(x, dtype=torch.double), x)

    def test_reduced_mean_dtype(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从标准正态分布，并要求其梯度
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 使用lambda表达式和torch.mean函数计算输入张量x在第0维上的平均值，结果张量的数据类型为torch.double，并验证导出的ONNX模型
        self.assertONNX(lambda x: torch.mean(x, dim=0, dtype=torch.double), x)

    def test_sum(self):
        # 创建一个形状为(1, 2, 3, 4)的张量x，元素服从
    # 测试函数，验证在计算的过程中对张量进行了降维求和操作，并且使用了双精度数据类型
    def test_reduced_sum_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=0, dtype=torch.double), x)

    # 测试函数，验证在计算的过程中对张量进行了多维度降维求和操作
    def test_reduced_sum(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=(1, 2)), x)

    # 测试函数，验证在计算的过程中对张量进行了降维求和操作，并保持了维度
    def test_reduced_sum_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sum(x, dim=2, keepdim=True), x)

    # 测试函数，验证在计算的过程中对张量进行了元素乘积操作
    def test_prod(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x), x)

    # 测试函数，验证在计算的过程中对张量进行了多维度降维乘积操作
    def test_reduced_prod(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=2), x)

    # 测试函数，验证在计算的过程中对张量进行了降维乘积操作，并保持了维度
    def test_reduced_prod_keepdim(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=2, keepdim=True), x)

    # 测试函数，验证在计算的过程中对张量进行了指定数据类型的元素乘积操作
    def test_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dtype=torch.double), x)

    # 测试函数，验证在计算的过程中对张量进行了指定数据类型和多维度降维乘积操作
    def test_reduced_prod_dtype(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.prod(x, dim=0, dtype=torch.double), x)

    # 测试函数，验证在计算的过程中对张量进行了元素级开方操作
    def test_sqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.sqrt(x), x)

    # 测试函数，验证在计算的过程中对张量进行了元素级反平方根操作
    def test_rsqrt(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.rsqrt(x), x)

    # 测试函数，验证在计算的过程中对两个整型张量进行了逐元素相等性比较操作
    def test_equal(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(operator.eq, (x, y))

    # 测试函数，验证在计算的过程中对两个整型张量进行了逐元素小于比较操作
    def test_lt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(operator.lt, (x, y))

    # 测试函数，验证在计算的过程中对两个整型张量进行了逐元素大于比较操作
    def test_gt(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(operator.gt, (x, y))

    # 测试函数，验证在计算的过程中对两个整型张量进行了逐元素小于等于比较操作
    def test_le(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(operator.le, (x, y))

    # 测试函数，验证在计算的过程中对两个整型张量进行了逐元素大于等于比较操作
    def test_ge(self):
        x = torch.randn(3, 4, requires_grad=False).int()
        y = torch.randn(3, 4, requires_grad=False).int()
        self.assertONNX(operator.ge, (x, y))

    # 测试函数，验证在计算的过程中对张量进行了指数函数操作
    def test_exp(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.exp(), x)

    # 测试函数，验证在计算的过程中对张量进行了正弦函数操作
    def test_sin(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.sin(), x)

    # 测试函数，验证在计算的过程中对张量进行了余弦函数操作
    def test_cos(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.cos(), x)

    # 测试函数，验证在计算的过程中对张量进行了正切函数操作
    def test_tan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.tan(), x)
    # 测试 torch.Tensor 对象的反正弦函数计算，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_asin(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.asin(), x)

    # 测试 torch.Tensor 对象的反余弦函数计算，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_acos(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.acos(), x)

    # 测试 torch.Tensor 对象的切片操作，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_slice(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x[:, 1:2], x)

    # 测试 torch.Tensor 对象的动态切片操作，并断言生成的 ONNX 模型与输入的张量 x 等效，使用特定的操作集版本
    def test_slice_dynamic(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x[x.size(0) :, x.size(1) - 3], x, opset_version=10)

    # 测试 torch.Tensor 对象的符号函数计算，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_sign(self):
        x = torch.rand(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.sign(), x)

    # 测试 torch.Tensor 对象的 narrow 函数，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_narrow(self):
        x = torch.randn(3, 3, requires_grad=True)
        self.assertONNX(lambda x: torch.narrow(x, 0, 0, 2), x)

    # 测试 torch.Tensor 对象的反正切函数计算，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_atan(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.atan(), x)

    # 测试 torch.Tensor 对象的视图展平操作，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_view_flatten(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.view(x.size()[0], x.numel() // x.size()[0]), x)

    # 测试 torch.Tensor 对象的全局展平操作，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_flatten(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.flatten(x), x)

    # 测试 torch.Tensor 对象的二维展平操作，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_flatten2D(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.flatten(x, 1), x)

    # 测试 torch.Tensor 对象的检测 NaN 函数，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_isnan(self):
        x = torch.tensor([1, float("nan"), 2])
        self.assertONNX(lambda x: torch.isnan(x), x)

    # 测试 torch.Tensor 对象的指定维度的最大值索引函数，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_argmax(self):
        x = torch.randn(4, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.argmax(x, dim=1), x)

    # 测试 torch.Tensor 对象的对数 softmax 激活函数，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_logsoftmax(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.LogSoftmax(dim=3), x)

    # 测试 torch.Tensor 对象的幂函数计算，并断言生成的 ONNX 模型与输入的张量 x, y 等效
    def test_pow(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x, y: x.pow(y), (x, y))

    # 测试 torch.Tensor 对象的 ELU 激活函数，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_elu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.ELU(), x)

    # 测试 torch.Tensor 对象的 SELU 激活函数，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_selu(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(nn.SELU(), x)

    # 测试 torch.Tensor 对象的复制展开操作，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_repeat(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.repeat(1, 2, 3, 4), x)

    # 测试 torch.Tensor 对象的复制展开操作（维度溢出情况），并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_repeat_dim_overflow(self):
        x = torch.randn(1, 2, requires_grad=True)
        self.assertONNX(lambda x: x.repeat(1, 2, 3, 4), x)

    # 测试 torch.Tensor 对象的 L1 范数计算，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_norm_p1(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=1, dim=2), (x))

    # 测试 torch.Tensor 对象的 L2 范数计算，并断言生成的 ONNX 模型与输入的张量 x 等效
    def test_norm_p2(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.assertONNX(lambda x: x.norm(p=2, dim=2), (x))
    # 定义测试函数，测试最近邻插值的上采样功能
    def test_upsample_nearest_scale(self):
        # 创建一个随机张量 x，形状为 [1, 2, 3, 4]，要求梯度计算
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 断言生成 ONNX 模型时，对张量 x 使用最近邻插值的结果
        self.assertONNX(
            lambda x: nn.functional.interpolate(
                x, scale_factor=2.0, mode="nearest", recompute_scale_factor=False
            ),
            x,
        )

    # 定义测试函数，测试最近邻插值的上采样功能（使用默认的尺度因子）
    def test_upsample_nearest_scale_default_scale_factor(self):
        # 创建一个随机张量 x，形状为 [1, 2, 3, 4]，要求梯度计算
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 断言生成 ONNX 模型时，对张量 x 使用最近邻插值的结果（使用默认的尺度因子 2.0）
        self.assertONNX(
            lambda x: nn.functional.interpolate(x, scale_factor=2.0, mode="nearest"), x
        )

    # 定义测试函数，测试最近邻插值的指定尺寸上采样功能
    def test_upsample_nearest_size(self):
        # 创建一个随机张量 x，形状为 [1, 2, 3, 4]，要求梯度计算
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        # 断言生成 ONNX 模型时，对张量 x 使用最近邻插值，指定输出大小为 16
        self.assertONNX(
            lambda x: nn.functional.interpolate(x, size=16, mode="nearest"), x
        )

    # 定义测试函数，测试在指定维度上对张量进行 unsqueeze 操作
    def test_unsqueeze(self):
        # 创建一个随机张量 x，形状为 [3, 4]，要求梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 断言生成 ONNX 模型时，对张量 x 进行 unsqueeze 操作，增加一个维度
        self.assertONNX(lambda x: x.unsqueeze(len(x.shape)), x)

    # 定义测试函数，测试不带仿射参数的批标准化层
    def test_batchnorm_noaffine(self):
        # 创建一个随机张量 x，形状为 [128, 128, 1, 1]，要求梯度计算
        x = torch.randn(128, 128, 1, 1, requires_grad=True)
        # 断言生成 ONNX 模型时，对输入 x 使用不带仿射参数的批标准化层
        self.assertONNX(
            nn.BatchNorm2d(128, affine=False, momentum=0.3),
            x,
            keep_initializers_as_inputs=True,
        )

    # 定义测试函数，测试嵌入包（EmbeddingBag）的转换为 ONNX
    def test_embedding_bags(self):
        # 创建一个包含 10 个词汇和每个词汇维度为 8 的嵌入包
        emb_bag = nn.EmbeddingBag(10, 8)
        # 创建一个输入张量，包含索引 [1, 2, 3, 4]
        input = torch.tensor([1, 2, 3, 4]).long()
        # 创建一个偏移张量，仅包含值 0
        offset = torch.tensor([0]).long()
        # 断言生成 ONNX 模型时，将嵌入包 emb_bag 应用于输入 (input, offset)
        self.assertONNX(
            emb_bag,
            (input, offset),
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    # 定义测试函数，测试张量的隐式扩展
    def test_implicit_expand(self):
        # 创建一个随机张量 x，形状为 [3, 4]，要求梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 断言生成 ONNX 模型时，对张量 x 执行隐式扩展操作，每个元素加一
        self.assertONNX(lambda x: x + 1, x)

    # 定义测试函数，测试在指定维度上对张量求和
    def test_reduce_sum_negative_indices(self):
        # 创建一个随机张量 x，形状为 [3, 4]，要求梯度计算
        x = torch.randn(3, 4, requires_grad=True)
        # 断言生成 ONNX 模型时，对张量 x 在指定维度 -1（最后一个维度）上求和
        self.assertONNX(lambda x: x.sum(-1), x)

    # 定义测试函数，测试生成服从标准正态分布的随机张量
    def test_randn(self):
        # 创建一个形状为 [1, 2, 3, 4] 的标准正态分布随机张量 x
        x = torch.randn(1, 2, 3, 4)
        # 断言生成 ONNX 模型时，生成另一个标准正态分布随机张量，然后与 x 相加
        self.assertONNX(lambda x: torch.randn(1, 2, 3, 4) + x, x)

    # 定义测试函数，测试生成服从均匀分布的随机张量
    def test_rand(self):
        # 创建一个形状为 [1, 2, 3, 4] 的均匀分布随机张量 x
        x = torch.rand(1, 2, 3, 4)
        # 断言生成 ONNX 模型时，生成另一个均匀分布随机张量，然后与 x 相加
        self.assertONNX(lambda x: torch.rand(1, 2, 3, 4) + x, x)

    # 定义测试函数，测试随机化整流线性单元（RReLU）的转换为 ONNX
    def test_rrelu(self):
        # 创建一个形状为 [1, 2, 3, 4] 的随机张量 x
        x = torch.randn(1, 2, 3, 4)
        # 断言生成 ONNX 模型时，将 RReLU 应用于输入 x
        self.assertONNX(torch.nn.RReLU(), x)

    # 定义测试函数，测试带有参数的参数化整流线性单元（PReLU）的转换为 ONNX
    def test_prelu(self):
        # 创建一个形状为 [1, 2, 3, 4] 的随机张量 x
        x = torch.randn(1, 2, 3, 4)
        # 断言生成 ONNX 模型时，将具有 2 个参数的 PReLU 应用于输入 x
        self.assertONNX(torch.nn.PReLU(2), x, keep_initializers_as_inputs=True)

    # 定义测试函数，测试对数 Sigmoid 函数的转换为 ONNX
    def test_log_sigmoid(self):
        # 创建一个形状为 [1, 2, 3, 4] 的随机张量 x
        x = torch.randn(1, 2, 3, 4)
        # 断言生成 ONNX 模型时，将对数 Sigmoid 函数应用于输入 x
        self.assertONNX(torch.nn.LogSigmoid(), x)

    # 定义测试函数，测试线性变换层的转换为 ONNX
    def test_linear(self):
        # 创建一个形状为 [3, 4] 的随机张量 x
        x = torch.randn(3, 4)
        # 断言生成 ONNX 模型时，将具有输入维度 4 和输出维度 5 的线性层应用于输入 x
        self.assertONNX(
            torch.nn.Linear(4, 5, bias
    # 测试 torch.ones_like 函数是否符合 ONNX 格式，使用随机生成的张量 x
    def test_ones_like(self):
        x = torch.randn(6, 10, requires_grad=True)
        self.assertONNX(lambda x: torch.ones_like(x), x)

    # 测试 torch.Tensor.expand 方法是否符合 ONNX 格式，使用随机生成的张量 x
    def test_expand(self):
        x = torch.randn(6, 1, requires_grad=True)
        self.assertONNX(lambda x: x.expand(4, 6, 2), x)

    # 测试 torch.ne 函数是否符合 ONNX 格式，使用两个随机生成的张量 x 和 y
    def test_ne(self):
        x = torch.randn(1, 2, 3, 1, requires_grad=False).int()
        y = torch.randn(1, 4, requires_grad=False).int()
        self.assertONNX(lambda x, y: torch.ne(x, y), (x, y))

    # 测试 torch.max 函数是否符合 ONNX 格式，使用随机生成的张量 x
    def test_reducemax(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.max(x), x)

    # 测试 torch.min 函数是否符合 ONNX 格式，使用随机生成的张量 x
    def test_reducemin(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: torch.min(x), x)

    # 测试 torch.Tensor.erf 方法是否符合 ONNX 格式，使用随机生成的张量 x
    def test_erf(self):
        x = torch.randn(1, 2, 3, 4)
        self.assertONNX(lambda x: x.erf(), x)

    # 测试 torch.nn.functional.dropout 函数是否符合 ONNX 格式，使用随机生成的张量 x
    def test_dropout(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(lambda x: torch.max(functional.dropout(x, training=False)), x)

    # 测试 torch.nn.functional.dropout 函数在默认设置下是否符合 ONNX 格式，使用随机生成的张量 x
    def test_dropout_default(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(
                functional.dropout(
                    x,
                )
            ),
            x,
        )

    # 测试 torch.nn.functional.dropout 函数在训练模式下是否符合 ONNX 格式，使用随机生成的张量 x
    def test_dropout_training(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            training=torch.onnx.TrainingMode.TRAINING,
        )

    # 测试 torch.nn.functional.dropout 函数在 opset_version=12 下是否符合 ONNX 格式，使用随机生成的张量 x
    def test_dropout_opset12(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x, training=False)),
            x,
            opset_version=12,
        )

    # 测试 torch.nn.functional.dropout 函数在训练模式下且 opset_version=12 下是否符合 ONNX 格式，使用随机生成的张量 x
    def test_dropout_training_opset12(self):
        x = torch.randn(3, 4, requires_grad=True)
        self.assertONNX(
            lambda x: torch.max(functional.dropout(x)),
            x,
            opset_version=12,
            training=torch.onnx.TrainingMode.TRAINING,
        )

    # 测试 torch.Tensor.nonzero 方法是否符合 ONNX 格式，使用指定张量 x
    def test_nonzero(self):
        x = torch.tensor(
            [[[2.0, 2.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]], requires_grad=True
        )
        self.assertONNX(lambda x: torch.nonzero(x), x)

    # 测试 torch.Tensor.gather 方法是否符合 ONNX 格式，使用随机生成的数据张量 data 和索引张量 index
    def test_gather(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(lambda data, index: data.gather(1, index), (data, index))

    # 测试 torch.Tensor.gather 方法在 opset_version=11 下是否符合 ONNX 格式，使用随机生成的数据张量 data 和索引张量 index
    def test_gather_opset11(self):
        data = torch.randn(3, 4, 3, requires_grad=True)
        index = torch.tensor([2, 0]).view(1, 2, 1).expand(3, 2, 3)
        self.assertONNX(
            lambda data, index: data.gather(1, index), (data, index), opset_version=11
        )
    # 定义一个测试函数，测试 scatter_add 方法
    def test_scatter_add(self):
        # 创建一个 3x3 的零张量
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # 定义要进行 scatter_add 操作的索引张量，数据类型为 int64
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 定义要添加的值张量，每个索引对应的添加值
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        # 调用自定义的 assertONNX 方法，测试 data 上的 scatter_add 操作
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
        )

    # 定义一个测试函数，测试 scatter_add 方法，并指定 opset_version 为 11
    def test_scatter_add_opset11(self):
        # 创建一个 3x3 的零张量
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # 定义要进行 scatter_add 操作的索引张量，数据类型为 int64
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        # 定义要添加的值张量，每个索引对应的添加值
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        # 调用自定义的 assertONNX 方法，测试 data 上的 scatter_add 操作，同时指定 opset_version 为 11
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            opset_version=11,
        )

    # 定义一个测试函数，测试 scatter_add 方法，并指定 opset_version 为 16
    def test_scatter_add_opset16(self):
        # 创建一个 3x3 的零张量
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # 定义要进行 scatter_add 操作的索引张量，数据类型为 int64
        indices = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.int64)
        # 定义要添加的值张量，每个索引对应的添加值
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        # 调用自定义的 assertONNX 方法，测试 data 上的 scatter_add 操作，同时指定 opset_version 为 16
        self.assertONNX(
            lambda data, index: data.scatter_add(1, indices, values),
            (data, (indices, values)),
            opset_version=16,
        )

    # 定义一个测试函数，测试加法运算操作符的 ONNX 导出，指定 opset_version 为 10
    def test_master_opset(self):
        # 创建两个形状为 (2, 3) 的随机张量，数据类型为 float
        x = torch.randn(2, 3).float()
        y = torch.randn(2, 3).float()
        # 调用自定义的 assertONNX 方法，测试 torch.add 操作
        self.assertONNX(operator.add, (x, y), opset_version=10)

    # 定义一个测试函数，测试标准差操作的 ONNX 导出
    def test_std(self):
        # 创建一个形状为 (2, 3, 4) 的随机张量，数据类型为 float
        x = torch.randn(2, 3, 4).float()
        # 调用自定义的 assertONNX 方法，测试 torch.std 操作，指定在维度 (0, 1) 上计算标准差
        self.assertONNX(
            lambda x: torch.std(x, dim=(0, 1), unbiased=True, keepdim=True), x
        )

    # 定义一个测试函数，测试累加和操作的 ONNX 导出，指定 opset_version 为 11
    def test_cumsum(self):
        # 创建一个形状为 (2, 3, 4) 的随机张量，同时需要梯度信息
        x = torch.randn(2, 3, 4, requires_grad=True)
        # 调用自定义的 assertONNX 方法，测试 torch.cumsum 操作，沿着维度 1 进行累加和
        self.assertONNX(lambda x: torch.cumsum(x, dim=1), x, opset_version=11)

    # 定义一个测试函数，测试字典操作的 ONNX 导出
    def test_dict(self):
        # 定义一个简单的神经网络模型，输入 x_in 为字典类型
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                # 将输入字典中的第一个键对应的张量和键名相加，生成输出字典
                x_out["test_key_out"] = torch.add(
                    x_in[list(x_in.keys())[0]], list(x_in.keys())[0]
                )
                return x_out

        # 创建一个字典，键为 tensor(1.0)，值为形状为 (1, 2, 3) 的随机张量
        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        # 调用自定义的 assertONNX 方法，测试 MyModel 类的 ONNX 导出
        self.assertONNX(MyModel(), (x, {}))

    # 定义一个测试函数，测试字典操作的 ONNX 导出，输入字典的键为字符串
    def test_dict_str(self):
        # 定义一个简单的神经网络模型，输入 x_in 为字典类型
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                # 将输入字典中 "test_key_in" 对应的张量和常量 2.0 相加，生成输出字典
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.0)
                return x_out

        # 创建一个字典，键为 "test_key_in"，值为形状为 (1, 2, 3) 的随机张量
        x = {"test_key_in": torch.randn(1, 2, 3)}
        # 调用自定义的 assertONNX 方法，测试 MyModel 类的 ONNX 导出
        self.assertONNX(MyModel(), (x, {}))

    # 定义一个测试函数，测试动态 arange 操作的 ONNX 导出，指定 opset_version 为 11
    def test_arange_dynamic(self):
        # 定义一个简单的神经网络模型，输入为任意形状的张量 input
        class TestModel(torch.nn.Module):
            def forward(self, input):
                # 返回一个从 input.shape[0] 开始，到 input.shape[0] + 5 结束的 arange 张量，步长为 0.5
                return torch.arange(input.shape[0], input.shape[0] + 5, 0.5)

        # 创建一个形状为 (5, 3, 2) 的随机张量 input
        input = torch.randn(5, 3, 2)
        # 调用自定义的 assertONNX 方法，测试 TestModel 类的 ONNX 导出，指定 opset_version 为 11
        self.assertONNX(TestModel(), input, opset_version=11)
    # 定义一个测试用例，测试位移操作的功能
    def test_bitshift(self):
        # 定义一个简单的神经网络模型，处理输入并执行右移操作
        class BitshiftModel(torch.nn.Module):
            def forward(self, input):
                return input >> 1, input >> 2

        # 创建一个长度为24的无符号整数张量，重塑为三维张量
        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        # 使用自定义的断言方法验证模型在指定的操作集版本下的ONNX导出结果
        self.assertONNX(BitshiftModel(), input, opset_version=11)

    # 定义一个测试用例，测试按位与操作的功能
    def test_bitwise_and(self):
        # 定义一个简单的神经网络模型，接受两个输入并执行按位与操作
        class BiwiseAndModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.bitwise_and(input, other), input & 2

        # 创建两个随机张量作为输入
        input = torch.randint(0, 100, (2, 3, 4), dtype=torch.uint8)
        other = torch.randint(-50, 50, (2, 3, 4), dtype=torch.int8)
        # 使用自定义的断言方法验证模型在指定的操作集版本下的ONNX导出结果
        self.assertONNX(BiwiseAndModel(), (input, other), opset_version=18)

    # 定义一个测试用例，测试层归一化操作的功能
    def test_layer_norm_aten(self):
        # 创建一个大小为[10, 10]的LayerNorm模型
        model = torch.nn.LayerNorm([10, 10])
        # 创建一个随机张量作为输入
        x = torch.randn(20, 5, 10, 10)
        # 使用自定义的断言方法验证模型在指定的操作集版本下的ONNX导出结果，
        # 并指定使用ATEN后端作为回退选项
        self.assertONNX(
            model,
            x,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    # 定义一个测试用例，测试像素重排操作的功能
    def test_pixel_shuffle(self):
        # 创建一个随机张量作为输入，并对其进行像素重排操作
        x = torch.randn(2, 8, 3, 4).float()
        # 使用自定义的断言方法验证像素重排函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(
            lambda x: torch.pixel_shuffle(x, upscale_factor=2), x, opset_version=11
        )

    # 定义一个测试用例，测试Frobenius范数的功能
    def test_frobenius_norm(self):
        # 创建一个随机张量作为输入，并计算其Frobenius范数
        x = torch.randn(2, 3, 4).float()
        # 使用自定义的断言方法验证Frobenius范数函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(lambda x: torch.norm(x, p="fro", dim=(0, 1), keepdim=True), x)

    # 定义一个测试用例，测试展开操作的功能
    def test_unfold(self):
        # 创建一个带有梯度信息的随机张量作为输入，并执行展开操作
        x = torch.randn(2, 3, 4, requires_grad=True)
        # 使用自定义的断言方法验证展开函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(lambda x: x.unfold(dimension=2, size=2, step=2), x)

    # 定义一个测试用例，测试取余数操作的功能
    def test_remainder(self):
        # 创建两个随机张量作为输入，并执行取余数操作
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        # 使用自定义的断言方法验证取余数函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(lambda x, y: torch.remainder(x, y), (x, y))

    # 定义一个测试用例，测试浮点取余数操作的功能
    def test_fmod(self):
        # 创建两个随机张量作为输入，并执行浮点取余数操作
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 1, 4)
        # 使用自定义的断言方法验证浮点取余数函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(lambda x, y: torch.fmod(x, y), (x, y), opset_version=10)

    # 定义一个测试用例，测试GELU激活函数的功能
    def test_gelu(self):
        # 创建一个带有梯度信息的随机张量作为输入，并应用GELU激活函数
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        # 使用自定义的断言方法验证GELU函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(lambda x: torch.nn.functional.gelu(x), x)

    # 定义一个测试用例，测试唯一值操作的功能
    def test_unique(self):
        # 创建一个随机整数张量作为输入，并执行唯一值操作
        x = torch.randint(3, (2, 3, 4, 5)).float()
        # 使用自定义的断言方法验证唯一值函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(
            lambda x: torch.unique(
                x, dim=0, sorted=True, return_inverse=False, return_counts=True
            ),
            x,
            opset_version=11,
        )

    # 定义一个测试用例，测试网格生成操作的功能
    def test_meshgrid(self):
        # 创建三个带有梯度信息的张量作为输入，并执行网格生成操作
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        # 使用自定义的断言方法验证网格生成函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(lambda x, y, z: torch.meshgrid(x, y, z), (x, y, z))

    # 定义一个测试用例，测试带索引的网格生成操作的功能
    def test_meshgrid_indexing(self):
        # 创建三个带有梯度信息的张量作为输入，并执行带索引的网格生成操作
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        # 使用自定义的断言方法验证带索引的网格生成函数在指定的操作集版本下的ONNX导出结果
        self.assertONNX(
            lambda x, y, z: torch.meshgrid(x, y, z, indexing="xy"),
            (x, y, z),
            opset_version=9,
        )
    # 测试 torch.topk 函数的功能
    def test_topk(self):
        # 创建一个张量 x，包含值为 [1.0, 2.0, 3.0, 4.0, 5.0]，并指定其需要梯度计算
        x = torch.arange(1.0, 6.0, requires_grad=True)
        # 创建一个张量 k，其值为 3
        k = torch.tensor(3)
        # 调用 self.assertONNX 方法，测试 torch.topk 函数在 opset_version=10 下的输出
        self.assertONNX(lambda x, k: torch.topk(x, k), (x, k), opset_version=10)

    # 测试 torch.topk 函数中 smallest=False 和 sorted=False 的情况
    def test_topk_smallest_unsorted(self):
        # 创建一个张量 x，包含值为 [1.0, 2.0, 3.0, 4.0, 5.0]，并指定其需要梯度计算
        x = torch.arange(1.0, 6.0, requires_grad=True)
        # 创建一个张量 k，其值为 3
        k = torch.tensor(3)
        # 调用 self.assertONNX 方法，测试 torch.topk 函数在 smallest=False 和 sorted=False 的情况下，opset_version=11 下的输出
        self.assertONNX(
            lambda x, k: torch.topk(x, k, largest=False, sorted=False),
            (x, k),
            opset_version=11,
        )

    # 测试 torch.baddbmm 函数的功能
    def test_baddbmm(self):
        # 创建一个形状为 (10, 3, 5) 的随机张量 x
        x = torch.randn(10, 3, 5)
        # 创建一个形状为 (10, 3, 4) 的随机张量 b1
        b1 = torch.randn(10, 3, 4)
        # 创建一个形状为 (10, 4, 5) 的随机张量 b2
        b2 = torch.randn(10, 4, 5)
        # 调用 self.assertONNX 方法，测试 torch.baddbmm 函数的输出
        self.assertONNX(lambda x, b1, b2: torch.baddbmm(x, b1, b2), (x, b1, b2))

    # 测试 torch.round 函数的功能
    def test_round(self):
        # 创建一个张量 x，包含值为 [0.9920, -1.0362, -1.5000, 2.5000]，并指定其需要梯度计算
        x = torch.tensor([0.9920, -1.0362, -1.5000, 2.5000], requires_grad=True)
        # 调用 self.assertONNX 方法，测试 torch.round 函数在 opset_version=11 下的输出
        self.assertONNX(lambda x: torch.round(x), x, opset_version=11)

    # 测试 torch.scalar_tensor 和 x.dim() 函数的功能
    def test_dim(self):
        # 创建一个全为 1 的形状为 (2, 2) 的张量 x，并指定其需要梯度计算
        x = torch.ones((2, 2), requires_grad=True)
        # 调用 self.assertONNX 方法，测试 torch.scalar_tensor 和 x.dim() 函数的输出
        self.assertONNX(lambda x: torch.scalar_tensor(x.dim()), x)

    # 使用 skipIfNoLapack 装饰器跳过测试，如果没有 LAPACK 支持
    @skipIfNoLapack
    def test_det(self):
        # 创建一个形状为 (2, 3, 5, 5) 的随机张量 x，位于 CPU 设备上
        x = torch.randn(2, 3, 5, 5, device=torch.device("cpu"))
        # 调用 self.assertONNX 方法，测试 torch.det 函数在 opset_version=11 下的输出
        self.assertONNX(lambda x: torch.det(x), x, opset_version=11)
        # 调用 self.assertONNX 方法，测试 torch.linalg.det 函数在 opset_version=11 下的输出
        self.assertONNX(lambda x: torch.linalg.det(x), x, opset_version=11)

    # 测试 torch.nn.CrossEntropyLoss 函数的功能
    def test_softmaxcrossentropy(self):
        # 创建一个形状为 (3, 5) 的随机张量 x
        x = torch.randn(3, 5)
        # 创建一个形状为 (3,) 的随机张量 y，包含随机整数值，数据类型为 torch.long
        y = torch.empty(3, dtype=torch.long).random_(5)
        # 调用 self.assertONNX 方法，测试 torch.nn.CrossEntropyLoss 函数在 opset_version=12 下的输出
        self.assertONNX(torch.nn.CrossEntropyLoss(), (x, y), opset_version=12)

    # 测试 torch.nn.CrossEntropyLoss 函数中 ignore_index 参数的功能
    def test_softmaxcrossentropy_ignore_index(self):
        # 创建一个形状为 (3, 5) 的随机张量 x
        x = torch.randn(3, 5)
        # 创建一个形状为 (3,) 的随机张量 y，包含随机整数值，数据类型为 torch.long
        y = torch.empty(3, dtype=torch.long).random_(5)
        # 调用 self.assertONNX 方法，测试 torch.nn.CrossEntropyLoss 函数在 ignore_index=1 情况下，opset_version=12 下的输出
        self.assertONNX(
            torch.nn.CrossEntropyLoss(ignore_index=1), (x, y), opset_version=12
        )

    # 测试 torch.nn.CrossEntropyLoss 函数中 weight 参数的功能
    def test_softmaxcrossentropy_weights(self):
        # 创建一个形状为 (3, 5) 的随机张量 x
        x = torch.randn(3, 5)
        # 创建一个形状为 (3,) 的随机张量 y，包含随机整数值，数据类型为 torch.long
        y = torch.empty(3, dtype=torch.long).random_(5)
        # 调用 self.assertONNX 方法，测试 torch.nn.CrossEntropyLoss 函数在 weight 参数为随机张量的情况下，opset_version=12 下的输出
        self.assertONNX(
            torch.nn.CrossEntropyLoss(weight=torch.randn(5)), (x, y), opset_version=12
        )

    # 测试 torch.nn.CrossEntropyLoss 函数处理 3 维输入的功能
    def test_softmaxcrossentropy_3d(self):
        # 创建一个形状为 (3, 5, 2) 的随机张量 x
        x = torch.randn(3, 5, 2)
        # 创建一个形状为 (3, 2) 的随机张量 y，包含随机整数值，数据类型为 torch.long
        y = torch.empty(3, 2, dtype=torch.long).random_(5)
        # 调用 self.assertONNX 方法，测试 torch.nn.CrossEntropyLoss 函数在 3 维输入下的输出，opset_version=12
        self.assertONNX(torch.nn.CrossEntropyLoss(), (x, y), opset_version=12)

    # 测试 torch.nn.CrossEntropyLoss 函数处理 3 维输入并且 reduction="none" 的功能
    def test_softmaxcrossentropy_3d_none(self):
        # 创建一个形状为 (3, 5, 2) 的随机张量 x
        x = torch.randn(3, 5, 2)
        # 创建一个形状为 (3, 2) 的随机张量 y，包含随机整数值，数据类型为 torch.long
        y = torch.empty(3, 2, dtype=torch.long).random_(5)
        # 调用 self.assertONNX 方法，测试 torch.nn.CrossEntropyLoss 函数在 3 维输入下，并且 reduction="none" 的输出，opset_version=12
        self.assertONNX(
            torch.nn.CrossEntropyLoss(reduction="none"), (x, y), opset_version=12
        )

    # 测试 torch.nn.CrossEntropyLoss 函数处理 4 维输入的功能
    def test_softmaxcrossentropy_4d(self):
        # 创建一个形状为 (3, 5, 2, 1) 的随机张量 x
        x = torch.randn(3, 5, 2, 1)
        # 创建一个形状为 (3, 2, 1) 的随机张量 y，包含
    def test_lstm_none_sequence_lens(self):
        """Test symbolic shape inference for LSTM when the input sequence_lens = None."""
        # 创建一个随机张量作为输入数据，形状为 (RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # 创建随机张量作为 LSTM 的初始隐藏状态 h0 和细胞状态 c0，形状为 (1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)

        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 LSTM 层，输入大小为 RNN_INPUT_SIZE，隐藏层大小为 RNN_HIDDEN_SIZE
                # 层数为 1，不是双向的
                self.rnn = torch.nn.LSTM(
                    RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False
                )

            def forward(self, x, h0, c0):
                # 在 LSTM 层上执行前向传播，输出结果为元组 (a, b)
                a, b = self.rnn(x, (h0, c0))
                # 返回一个形状与 b[0] 相同的全为1的张量
                return torch.ones(b[0].shape)

        # 调用 assertONNX 函数来验证模型转换为 ONNX 格式的正确性
        self.assertONNX(
            LSTMModel(),
            (input, h0, c0),
            input_names=["x", "y"],
            # 指定动态维度的名称和索引映射关系
            dynamic_axes={"x": {0: "batch"}},
            opset_version=12,
        )

    def test_dynamic_axes_add(self):
        # 创建两个形状为 (2, 3) 的随机张量 m1 和 m2
        m1 = torch.randn(2, 3, requires_grad=True)
        m2 = torch.randn(2, 1, requires_grad=True)
        # 调用 assertONNX 函数来验证张量相加操作转换为 ONNX 格式的正确性
        self.assertONNX(
            lambda x, y: torch.add(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            # 指定动态维度的名称和索引映射关系
            dynamic_axes={"input_1": {1: "dim_1"}, "input_2": {1: "dim_2"}},
            opset_version=12,
        )

    def test_dynamic_axes_add_inputs_same_symbolic_shape(self):
        # 创建一个形状为 (2, 3) 的随机张量 m1
        m1 = torch.randn(2, 3, requires_grad=True)
        # 调用 assertONNX 函数来验证张量自加操作转换为 ONNX 格式的正确性
        self.assertONNX(
            lambda x: torch.add(x, x),
            (m1,),
            input_names=["input_1"],
            # 指定动态维度的名称和索引映射关系
            dynamic_axes={"input_1": {1: "dim_1"}},
            opset_version=12,
        )

    def test_dynamic_axes_matmul(self):
        # 创建两个随机张量 m1 和 m2，形状分别为 (2, 2, 4) 和 (2, 4, 3)
        m1 = torch.randn(2, 2, 4, requires_grad=True)
        m2 = torch.randn(2, 4, 3, requires_grad=True)
        # 调用 assertONNX 函数来验证矩阵相乘操作转换为 ONNX 格式的正确性
        self.assertONNX(
            lambda x, y: torch.matmul(x, y),
            (m1, m2),
            input_names=["input_1", "input_2"],
            # 指定动态维度的名称和索引映射关系
            dynamic_axes={"input_1": {1: "dim_0"}, "input_2": {2: "dim_1"}},
            opset_version=12,
        )

    def test_dynamic_axes_reduce_mean(self):
        # 创建一个形状为 (2, 3, 4) 的随机张量 m1
        m1 = torch.randn(2, 3, 4, requires_grad=True)
        # 调用 assertONNX 函数来验证张量按指定维度求均值操作转换为 ONNX 格式的正确性
        self.assertONNX(
            lambda x: torch.mean(x, dim=1),
            (m1),
            input_names=["input"],
            # 指定动态维度的名称和索引映射关系
            dynamic_axes={"input": {1: "dim_1", 2: "dim_2"}},
            opset_version=12,
        )

    def test_dynamic_axes_unchange(self):
        """Test ProcessUnchangeNode in symbolic shape inference."""
        # 创建一个形状为 (2, 3) 的随机张量 m1
        m1 = torch.randn(2, 3, requires_grad=True)
        # 调用 assertONNX 函数来验证张量 softmax 操作转换为 ONNX 格式的正确性
        self.assertONNX(
            lambda x: torch.softmax(x, dim=0),
            (m1,),
            input_names=["input"],
            # 指定动态维度的名称和索引映射关系
            dynamic_axes={"input": {1: "dim_1"}},
            opset_version=12,
        )
    def test_aten_embedding_1(self):
        # 设置当前使用的 ONNX 操作集版本为 12
        _onnx_opset_version = 12

        # 定义一个装饰器函数，用于解析参数，生成 ONNX 图中的 embedding 操作
        @parse_args("v", "v", "i", "b", "b")
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            # 构建自定义属性的 JSON 字符串
            custom_attributes_json = (
                "{"
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                "}"
            )
            # 在 ONNX 图中调用 "embedding" 操作，并传入相关参数和自定义属性
            output = g.at(
                "embedding",
                weight,
                indices,
                custom_attributes_json_s=custom_attributes_json,
            )
            return output

        # 在 Torch ONNX 中注册自定义操作符号化函数 "::embedding"
        torch.onnx.register_custom_op_symbolic(
            "::embedding", embedding, _onnx_opset_version
        )

        # 定义一个简单的神经网络模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个维度为 4x8 的 Embedding 层
                self.emb = torch.nn.Embedding(4, 8)

            # 定义前向传播函数
            def forward(self, x, y):
                # 对输入 x 进行 Embedding 操作
                res = self.emb(x)
                # 将 Embedding 结果与输入 y 相加
                res = res + y
                # 返回张量形状的全为 1 的张量
                return torch.ones(res.shape[0])

        # 创建一个 Model 实例
        model = Model()
        # 创建输入张量 x，所有元素为 1，数据类型为 long
        x = torch.ones(32, dtype=torch.long)
        # 创建输入张量 y，形状为 (1, 8)，元素为随机数
        y = torch.randn(1, 8)
        # 使用 assertONNX 方法验证模型的 ONNX 导出结果
        self.assertONNX(model, (x, y), opset_version=_onnx_opset_version)

        # 在 Torch ONNX 中取消注册 "::embedding" 自定义操作符号化函数
        torch.onnx.unregister_custom_op_symbolic("::embedding", _onnx_opset_version)

    # 这是 test_aten_embedding_1 的扩展，包含了对自定义符号化 aten::embedding 的形状推断。
    def test_aten_embedding_2(self):
        _onnx_opset_version = 12

        # 定义一个嵌入层的自定义 ONNX 符号化函数
        @parse_args("v", "v", "i", "b", "b")
        def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
            # 构建包含自定义属性的 JSON 字符串
            custom_attributes_json = (
                "{"
                f'"padding_idx":{str(padding_idx)},'
                f'"scale_grad_by_freq":{str(scale_grad_by_freq).lower()},'
                f'"sparse":{str(sparse).lower()}'
                "}"
            )
            # 在计算图中调用 ONNX 的 embedding 操作
            output = g.at(
                "embedding",
                weight,
                indices,
                custom_attributes_json_s=custom_attributes_json,
            )

            # 进行形状推断并通过 setType 设置输出张量类型
            indices_shape = _get_tensor_sizes(indices)
            if indices_shape is not None and hasattr(weight.type(), "with_sizes"):
                output_type = weight.type().with_sizes(
                    indices_shape + [_get_tensor_dim_size(weight, 1)]
                )
                output.setType(output_type)
            return output

        # 在 Torch ONNX 中注册自定义操作的符号化函数
        torch.onnx.register_custom_op_symbolic(
            "::embedding", embedding, _onnx_opset_version
        )

        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 8)

            # 前向传播函数定义
            def forward(self, x, y):
                res = self.emb(x)
                res = res + y
                return torch.ones(res.shape[0])

        # 创建模型实例
        model = Model()
        x = torch.ones(32, dtype=torch.long)
        y = torch.randn(1, 8)
        # 使用 assertONNX 函数检验模型的 ONNX 输出
        self.assertONNX(
            model,
            (x, y),
            opset_version=_onnx_opset_version,
            input_names=["input_1", "input_2"],
            dynamic_axes={"input_1": {0: "dim_0"}, "input_2": {0: "dim_1", 1: "dim_2"}},
            keep_initializers_as_inputs=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

        # 在 Torch ONNX 中取消注册自定义操作的符号化函数
        torch.onnx.unregister_custom_op_symbolic("::embedding", _onnx_opset_version)

    # 当没有 shapeValueMap 时，ONNX 图如下所示：
    # graph(%0 : Float(*, 1, 128, 1, strides=[128, 128, 1, 1], requires_grad=0, device=cpu)):
    #   %2 : Long(4, strides=[1], device=cpu) = onnx::Shape(%0)
    #   %4 : Long(device=cpu) = onnx::Constant[value={0}]()
    #   %5 : Long(device=cpu) = onnx::Gather[axis=0](%2, %4)
    #   %6 : Long(device=cpu) = onnx::Constant[value={1}]()
    #   %7 : Long(device=cpu) = onnx::Constant[value={2}]()
    #   %8 : Long(device=cpu) = onnx::Constant[value={-1}]()
    #   %9 : int[] = prim::ListConstruct(%5, %6, %7, %8)
    #   %10 : Float(*, *, *, *, strides=[128, 128, 64, 1], requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
    #   ...
    # 当有 shapeValueMap 时，ONNX 图如下所示：
    #   ...
    #   %10 : Float(*, 1, 2, 64, strides=[128, 128, 64, 1], requires_grad=0, device=cpu) = onnx::Reshape(%0, %9)
    #   ...
    # 定义测试类的一个测试方法，用于测试形状值映射
    def test_shape_value_map(self):
        # 定义一个名为 RSoftMax 的内部类，继承自 torch.nn.Module
        class RSoftMax(torch.nn.Module):
            # 初始化方法，接受 radix 和 cardinality 两个参数
            def __init__(self, radix, cardinality):
                super().__init__()
                # 设置对象的 radix 属性
                self.radix = radix
                # 设置对象的 cardinality 属性
                self.cardinality = cardinality

            # 前向传播方法
            def forward(self, x):
                # 获取输入张量的批量大小
                batch = x.size(0)
                # 将输入张量 x 进行形状变换，变换后的维度为 (batch, cardinality, radix, -1)，然后进行维度转置
                x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
                # 对转置后的张量 x 进行 softmax 操作，沿着第一个维度进行计算
                x = F.softmax(x, dim=1)
                # 将张量 x 进行形状变换，将前两个维度合并成一个，其他维度不变
                x = x.reshape(batch, -1)
                # 返回变换后的张量 x
                return x

        # 定义变量 radix 和 cardinality，分别赋值为 2 和 1
        radix = 2
        cardinality = 1
        # 生成一个形状为 (10, 1, 128, 1) 的随机张量 x
        x = torch.randn(10, 1, 128, 1)
        # 调用测试框架的 assertONNX 方法，测试 RSoftMax 类的输出
        self.assertONNX(
            RSoftMax(radix, cardinality),  # 创建 RSoftMax 类的实例
            (x,),  # 传入随机张量 x 作为输入
            input_names=["x"],  # 指定输入的名称为 "x"
            dynamic_axes={"x": {0: "dim_0"}},  # 指定动态维度轴的映射关系
        )
# 如果脚本作为主程序运行
if __name__ == "__main__":
    # 定义一个标志，指示是否存在 "--no-onnx" 参数
    no_onnx_dep_flag = "--no-onnx"
    # 检查 "--no-onnx" 是否不在 common_utils.UNITTEST_ARGS 中，返回布尔值
    _onnx_dep = no_onnx_dep_flag not in common_utils.UNITTEST_ARGS
    # 如果存在 "--no-onnx" 参数，则从 common_utils.UNITTEST_ARGS 中移除它
    if no_onnx_dep_flag in common_utils.UNITTEST_ARGS:
        common_utils.UNITTEST_ARGS.remove(no_onnx_dep_flag)
    # 定义一个标志，指示是否存在 "--produce-onnx-test-data" 参数
    onnx_test_flag = "--produce-onnx-test-data"
    # 检查 "--produce-onnx-test-data" 是否在 common_utils.UNITTEST_ARGS 中，返回布尔值
    _onnx_test = onnx_test_flag in common_utils.UNITTEST_ARGS
    # 如果存在 "--produce-onnx-test-data" 参数，则从 common_utils.UNITTEST_ARGS 中移除它
    if onnx_test_flag in common_utils.UNITTEST_ARGS:
        common_utils.UNITTEST_ARGS.remove(onnx_test_flag)
    # 如果存在 _onnx_test 标志
    if _onnx_test:
        # 将 _onnx_dep 标志设置为 True
        _onnx_dep = True
        # 导入 onnx_test_common 模块
        import onnx_test_common
        # 删除所有符合条件的目录
        for d in glob.glob(
            os.path.join(onnx_test_common.pytorch_operator_dir, "test_operator_*")
        ):
            shutil.rmtree(d)
    # 运行测试用例
    common_utils.run_tests()
```