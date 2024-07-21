# `.\pytorch\test\test_nnapi.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

# 导入所需的库和模块
import ctypes
import os
import unittest
from typing import Tuple

import torch
from torch.backends._nnapi.prepare import convert_model_to_nnapi
from torch.testing._internal.common_quantized import supported_qengines
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义一个函数，用于量化张量
def qpt(t, scale, zero_point, dtype=torch.quint8):
    t = torch.tensor(t)
    return torch.quantize_per_tensor(t, scale, zero_point, dtype)

# 定义一个函数，用于转换张量的内存布局为通道优先形式
def nhwc(t):
    t = t.clone().contiguous(memory_format=torch.channels_last)
    t.nnapi_nhwc = True
    return t

# 创建一个测试类，用于测试NNAPI相关功能
@unittest.skipUnless(
    "qnnpack" in supported_qengines,
    "This Pytorch Build has not been built with or does not support QNNPACK",
)
class TestNNAPI(TestCase):
    # 设置测试准备工作
    def setUp(self):
        # 设置量化后端引擎为qnnpack，避免fbgemm中的饱和
        torch.backends.quantized.engine = "qnnpack"

        # 加载libneuralnetworks库路径，以便支持NNAPI模型运行
        libneuralnetworks_path = os.environ.get("LIBNEURALNETWORKS_PATH")
        if libneuralnetworks_path:
            ctypes.cdll.LoadLibrary(libneuralnetworks_path)
            print("Will attempt to run NNAPI models.")
            self.can_run_nnapi = True
        else:
            self.can_run_nnapi = False

    # 提供一个方法，允许子类通过覆盖来调用转换为NNAPI的功能
    def call_lowering_to_nnapi(self, traced_module, args):
        return convert_model_to_nnapi(traced_module, args)

    # 提供一个方法，允许子类设置是否可以运行NNAPI的标志
    def set_can_run_nnapi(self, can_run):
        self.can_run_nnapi = can_run

    # 定义一个检查函数，用于检查NNAPI功能的测试用例
    def check(
        self,
        module,
        arg_or_args,
        *,
        trace_args=None,
        convert_args=None,
        atol_rtol=None,
        limit=None,
        expected_memory_format=None,
    ):
        # 进入 torch.no_grad 上下文管理器，禁止梯度计算
        with torch.no_grad():
            # 检查参数 arg_or_args 是否为 torch.Tensor 类型，若是则转化为列表 args，否则直接使用 arg_or_args
            if isinstance(arg_or_args, torch.Tensor):
                args = [arg_or_args]
            else:
                args = arg_or_args
            # 将模型设为评估模式
            module.eval()
            # 使用 torch.jit.trace 对模型进行跟踪，trace_args 或 args 作为输入参数
            traced = torch.jit.trace(module, trace_args or args)
            # 调用 self.call_lowering_to_nnapi 方法将追踪得到的模型转换为 NNAPI 模块，使用 convert_args 或 args 作为输入参数
            nnapi_module = self.call_lowering_to_nnapi(traced, convert_args or args)
            # 如果不能运行 NNAPI，则仅测试模型是否成功转换
            if not self.can_run_nnapi:
                # 只返回，不执行后续操作
                return
            # 在 eager 模式下执行模型，获取输出
            eager_output = module(*args)
            # 在 NNAPI 模式下执行模型，获取输出
            nnapi_output = nnapi_module(*args)
            # 设置 kwargs 为空字典
            kwargs = {}
            # 如果 atol_rtol 不为 None，则设置 kwargs 中的 "atol" 和 "rtol" 键值对
            if atol_rtol is not None:
                kwargs["atol"] = atol_rtol[0]
                kwargs["rtol"] = atol_rtol[1]
            # 使用 self.assertEqual 检查 eager_output 和 nnapi_output 是否相等，带上 kwargs 中的容差值
            self.assertEqual(eager_output, nnapi_output, **kwargs)
            # 如果 limit 不为 None
            if limit is not None:
                # 计算两个输出的整数表示之间的差异
                mismatches = eager_output.int_repr().to(
                    torch.int32
                ) - nnapi_output.int_repr().to(torch.int32)
                # 如果差异超过 limit，则再次用零容差重新运行检查以获得详细信息
                if mismatches.count_nonzero() > limit:
                    self.assertEqual(eager_output, nnapi_output, atol=0, rtol=0)
            # 如果 expected_memory_format 不为 None
            if expected_memory_format:
                # 使用 self.assertTrue 检查 nnapi_output 是否符合预期的内存格式
                self.assertTrue(
                    nnapi_output.is_contiguous(memory_format=expected_memory_format)
                )

    def float_and_quant_and_nhwc(self, inp_float, scale, zero_point):
        # 设置随机种子
        torch.manual_seed(29)
        # 对浮点输入 inp_float 进行量化处理，使用 qpt 函数，参数为 0.03 和 128
        inp_quant = qpt(inp_float, 0.03, 128)
        # 返回包含四种不同格式的输入数据的列表
        return [
            ("float", inp_float),
            ("float-nhwc", nhwc(inp_float)),
            ("quant", inp_quant),
            ("quant-nhwc", nhwc(inp_quant)),
        ]

    def test_prelu(self):
        # 创建测试输入 arg
        arg = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        # 创建单个参数的 PReLU 模型
        single_a = torch.nn.PReLU()
        # 使用 self.check 方法测试单个参数的 PReLU 模型
        self.check(single_a, arg)
        # 创建多个参数的 PReLU 模型
        multi_a = torch.nn.PReLU(4)
        # 进入 torch.no_grad 上下文管理器，禁止梯度计算
        with torch.no_grad():
            # 复制权重张量到 multi_a 模型
            multi_a.weight.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        # 使用 self.check 方法测试多个参数的 PReLU 模型，并将输入转换为 NHWC 格式
        self.check(multi_a, nhwc(arg))

        # 测试灵活大小的模型
        self.check(
            multi_a,
            arg,
            trace_args=[torch.zeros(1, 4, 3, 3)],
            convert_args=[nhwc(torch.zeros(1, 4, 0, 0))],
        )

    def test_quantize(self):
        # 使用 self.check 方法测试 Quantize 模块，输入为 NHWC 格式的张量
        self.check(
            torch.ao.nn.quantized.Quantize(0.25, 2, torch.quint8),
            nhwc(torch.tensor([[[[1.0]], [[2.0]]]])),
        )

    def test_dequantize(self):
        # 使用 self.check 方法测试 DeQuantize 模块，输入为经过量化处理后的 NHWC 格式的张量
        self.check(
            torch.ao.nn.quantized.DeQuantize(), nhwc(qpt([[[[1.0]], [[2.0]]]], 0.25, 2))
        )
    def test_unsqueeze(self):
        # 定义一个内部类 UnsqueezeModule，继承自 torch.nn.Module
        class UnsqueezeModule(torch.nn.Module):
            # 初始化方法，接收一个维度参数 dim
            def __init__(self, dim):
                super().__init__()
                # 将参数 dim 存储在实例变量 self.dim 中
                self.dim = dim

            # 前向传播方法，接收一个参数 arg，调用 arg 的 unsqueeze 方法
            # 使用 self.dim 对 arg 进行 unsqueeze 操作，并返回结果
            def forward(self, arg):
                return arg.unsqueeze(self.dim)

        # 使用 self.check 方法测试 UnsqueezeModule 的不同实例与给定的随机张量参数
        self.check(UnsqueezeModule(-2), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(-1), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(0), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(1), torch.randn(4, 2, 2))
        self.check(UnsqueezeModule(2), torch.randn(4, 2, 2))

    def test_reshape(self):
        # 定义一个内部类 ReshapeModule，继承自 torch.nn.Module
        class ReshapeModule(torch.nn.Module):
            # 初始化方法，接收一个形状参数 shape
            def __init__(self, shape):
                super().__init__()
                # 将参数 shape 存储在实例变量 self.shape 中
                self.shape = shape

            # 前向传播方法，接收一个参数 arg，调用 arg 的 reshape 方法
            # 使用 self.shape 对 arg 进行 reshape 操作，并返回结果
            def forward(self, arg):
                return arg.reshape(self.shape)

        # 使用 self.check 方法测试 ReshapeModule 的实例与给定的随机张量参数
        self.check(ReshapeModule((2, 4)), torch.randn(4, 2, 1, 1))

        # 使用 nhwc 函数转换参数后，使用 ReshapeModule 进行测试
        self.check(ReshapeModule((8, -1)), nhwc(torch.randn(4, 2, 1, 1)))

        # 测试抛出异常的情况
        with self.assertRaisesRegex(Exception, "target size"):
            self.check(ReshapeModule((2, 4)), nhwc(torch.randn(4, 2, 1, 1)))

    def test_flatten(self):
        # 遍历不同的 Flatten 模块实例
        for mod in [
            torch.nn.Flatten(),
            torch.nn.Flatten(start_dim=2, end_dim=3),
            torch.nn.Flatten(start_dim=2, end_dim=4),
            torch.nn.Flatten(start_dim=0, end_dim=-2),
            torch.nn.Flatten(start_dim=0, end_dim=4),
        ]:
            # 使用 self.check 方法测试每个 Flatten 模块实例与给定的随机张量参数
            self.check(mod, torch.randn(4, 2, 1, 3, 7))

        # 测试灵活的输入情况
        self.check(
            torch.nn.Flatten(),
            torch.randn(4, 2, 1, 3, 7),
            convert_args=[torch.zeros(0, 2, 1, 3, 7)],
        )

        # 测试通道为最后一维的情况
        self.check(torch.nn.Flatten(), nhwc(torch.randn(2, 1, 4, 7)))
        self.check(torch.nn.Flatten(), nhwc(torch.randn(2, 3, 1, 1)))

        # 测试异常情况
        with self.assertRaisesRegex(Exception, "not supported on NHWC"):
            self.check(torch.nn.Flatten(), nhwc(torch.randn(1, 3, 4, 4)))
        with self.assertRaisesRegex(
            Exception, "Flattening flexible dims is not supported yet"
        ):
            self.check(torch.nn.Flatten(), torch.randn(4, 2, 0, 0, 7))
        with self.assertRaisesRegex(Exception, "Only 1 dim"):
            self.check(
                torch.nn.Flatten(start_dim=1, end_dim=-2), torch.randn(0, 2, 1, 3, 0)
            )
    def test_slice(self):
        # 定义一个测试用例类，测试切片操作的模块
        class SliceModule(torch.nn.Module):
            def __init__(self, start, stop, step):
                super().__init__()
                self.start = start  # 初始化切片的起始位置
                self.stop = stop    # 初始化切片的终止位置
                self.step = step    # 初始化切片的步长

            def forward(self, t):
                # 执行切片操作，返回从索引为1的维度开始，到指定步长和范围的切片
                return t[1:, self.start : self.stop : self.step, :]

        # 定义另一个测试用例类，测试简单的切片操作
        class SliceModule2(torch.nn.Module):
            def forward(self, t):
                return t[3:]  # 返回从索引为3开始的切片

        # 使用自定义的检查函数验证切片模块的输出与输入的兼容性
        self.check(SliceModule(1, 5, 2), torch.randn(4, 6, 2))
        self.check(SliceModule2(), torch.randn(5))

        # 测试灵活的输入情况
        self.check(
            SliceModule(1, 5, 2),
            torch.randn(4, 6, 2),
            convert_args=[torch.zeros(4, 6, 0)],
        )
        # 检查是否引发预期的异常
        with self.assertRaisesRegex(Exception, "slice with flexible shape"):
            self.check(
                SliceModule(1, 5, 2),
                torch.randn(4, 6, 2),
                convert_args=[torch.zeros(0, 0, 0)],
            )

    def test_cat(self):
        # 定义一个测试用例类，测试张量的连接操作
        class CatModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim  # 初始化连接操作的维度

            def forward(self, t1, t2):
                # 执行张量连接操作，沿指定维度连接两个张量
                return torch.cat([t1, t2], self.dim)

        # 检查在不同维度上的张量连接操作
        self.check(
            CatModule(0),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(2, 2, 3, 3),
            ],
        )

        self.check(
            CatModule(1),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(1, 4, 3, 3),
            ],
        )

        self.check(
            CatModule(1),
            [
                nhwc(torch.randn(1, 2, 3, 3)),
                nhwc(torch.randn(1, 4, 3, 3)),
            ],
        )

        # 测试灵活的输入情况
        self.check(
            CatModule(1),
            [
                torch.randn(1, 2, 3, 3),
                torch.randn(1, 4, 3, 3),
            ],
            convert_args=[torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)],
        )

    def test_pointwise_unary(self):
        # 遍历每种一元操作
        for op in ["relu", "sigmoid"]:
            with self.subTest(op):

                # 定义一个测试用例类，测试一元操作的模块
                class UnaryModule(torch.nn.Module):
                    def forward(self, arg):
                        if op == "relu":
                            return torch.nn.functional.relu(arg)  # 执行 ReLU 激活函数
                        if op == "sigmoid":
                            return torch.sigmoid(arg)  # 执行 sigmoid 激活函数
                        raise Exception("Bad op")  # 如果操作不合法，抛出异常

                # 使用自定义的检查函数验证一元操作模块的输出与输入的兼容性
                self.check(UnaryModule(), torch.tensor([-1.0, 1.0]))
                self.check(
                    UnaryModule(),
                    qpt(torch.tensor([-1.0, 1.0]), 1.0 / 256, 0),
                )
    def test_pointwise_binary(self):
        # 遍历操作列表，每个操作(op)进行以下测试
        for op in ["add", "sub", "mul", "div"]:
            # 使用子测试上下文，测试当前操作(op)
            with self.subTest(op):
                
                # 定义一个新的二元操作的模块
                class BinaryModule(torch.nn.Module):
                    def forward(self, lhs, rhs):
                        # 根据操作(op)类型执行相应的二元运算
                        if op == "add":
                            return lhs + rhs
                        if op == "sub":
                            return lhs - rhs
                        if op == "mul":
                            return lhs * rhs
                        if op == "div":
                            return lhs / rhs
                        # 若操作(op)不在预期范围内，抛出异常
                        raise Exception("Bad op")  # noqa: TRY002

                # 使用自定义的检查方法检查BinaryModule的不同输入
                self.check(
                    BinaryModule(),
                    [
                        torch.tensor([1.0, 2.0]),
                        torch.tensor([3.0, 4.0]),
                    ],
                )

                self.check(
                    BinaryModule(),
                    [
                        torch.tensor([[1.0, 2.0]]),
                        torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                    ],
                )

                # 使用断言检查广播不一致的情况
                with self.assertRaisesRegex(Exception, "Non-equal-rank broadcast"):
                    self.check(
                        BinaryModule(),
                        [
                            torch.tensor([1.0, 2.0]),
                            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
                        ],
                    )

    def test_pointwise_binary_const(self):
        # 创建一个随机常量张量
        const = torch.randn(1, 4, 6, 6)

        # 定义将常量加到输入张量的模块
        class ArgPlusConst(torch.nn.Module):
            def forward(self, arg):
                return arg + const

        # 定义将输入张量加到常量的模块
        class ConstPlusArg(torch.nn.Module):
            def forward(self, arg):
                return const + arg

        # 创建连续存储格式和NHWC格式的输入张量
        arg_contig = torch.randn(2, 4, 6, 6)
        arg_nhwc = nhwc(torch.randn(2, 4, 6, 6))

        # 遍历模块类和是否使用NHWC格式的组合
        for mod_class in [ArgPlusConst, ConstPlusArg]:
            for use_nhwc in [False, True]:
                # 使用子测试上下文测试当前模块类和使用NHWC格式的情况
                with self.subTest(mod_class=mod_class.__name__, use_nhwc=use_nhwc):
                    # 根据是否使用NHWC格式选择输入张量
                    arg = arg_nhwc if use_nhwc else arg_contig
                    # 根据是否使用NHWC格式选择内存格式
                    memory_format = (
                        torch.channels_last if use_nhwc else torch.contiguous_format
                    )
                    # 使用自定义的检查方法检查模块输出和预期内存格式
                    self.check(mod_class(), arg, expected_memory_format=memory_format)

    def test_hardtanh(self):
        # 创建输入张量
        inp = torch.tensor([-2.0, -0.5, 0.5, 2.0, 7.0])
        # 使用Hardtanh模块进行检查
        self.check(torch.nn.Hardtanh(), inp)
        # 使用带有指定上下限的Hardtanh模块进行检查
        self.check(torch.nn.Hardtanh(0.0, 6.0), inp)
        # 使用断言检查指定上下限的Hardtanh模块会抛出异常的情况
        with self.assertRaisesRegex(Exception, "hardtanh with args"):
            self.check(torch.nn.Hardtanh(0.0, 5.0), inp)

    def test_softmax(self):
        # 创建输入张量
        inp = torch.tensor([[-2.0, -0.5], [0.5, 2.0]])
        # 使用Softmax模块进行检查
        self.check(torch.nn.Softmax(), inp)
        # 使用指定维度的Softmax模块进行检查
        self.check(torch.nn.Softmax(dim=0), inp)
        # 测试灵活尺寸的Softmax模块，同时传递额外的参数
        self.check(
            torch.nn.Softmax(),
            inp,
            convert_args=[torch.zeros(0, 0)],
        )
    def test_to(self):
        # 定义一个名为 ToCPU 的自定义神经网络模块
        class ToCPU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块，包括 PReLU 激活函数
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                # 将输入张量 x 移动到 CPU 上
                y = x.to("cpu")
                # 返回经过 PReLU 激活函数处理后的输出
                # 输入操作数不能作为输出，因此添加了 PReLU
                return self.prelu(y)

        # 创建一个随机张量作为输入参数
        arg = torch.randn(1, 2, 3, 3)
        # 调用 self.check 方法，测试 ToCPU 模块的输出
        self.check(ToCPU(), arg)
        # 测试灵活大小的输入参数
        self.check(
            ToCPU(),
            arg,
            convert_args=[torch.zeros(1, 2, 0, 0)],
        )

    def test_detach(self):
        # 定义一个名为 DetachModule 的自定义神经网络模块
        class DetachModule(torch.nn.Module):
            def forward(self, x):
                # 分离输入张量 x 的梯度
                y = x.detach()
                # 返回经过 ReLU 激活函数处理后的输出
                return torch.nn.functional.relu(y)

        # 调用 self.check 方法，测试 DetachModule 模块的输出
        self.check(DetachModule(), torch.randn(1, 2, 3, 3))
        # 测试灵活大小的输入参数
        self.check(
            DetachModule(),
            torch.randn(1, 2, 3, 3),
            convert_args=[torch.zeros(1, 2, 0, 0)],
        )

    def test_log_softmax(self):
        # 创建一个随机张量作为输入
        inp = torch.randn(3, 10)
        # 调用 self.check 方法，测试 LogSoftmax 模块的输出
        self.check(torch.nn.LogSoftmax(), inp)
        self.check(torch.nn.LogSoftmax(0), inp)

    def test_mean(self):
        # 定义一个名为 MeanModule 的自定义神经网络模块
        class MeanModule(torch.nn.Module):
            def __init__(self, dim, keep=False):
                super().__init__()
                # 初始化模块，设置维度和是否保持维度
                self.dim = dim
                self.keep = keep

            def forward(self, t):
                # 计算输入张量 t 在指定维度上的均值
                return torch.mean(t, dim=self.dim, keepdim=self.keep)

        # 调用 self.check 方法，测试 MeanModule 模块的输出
        self.check(MeanModule(0), torch.randn(2, 3))
        self.check(MeanModule(1), torch.randn(2, 3))
        self.check(MeanModule([2, 3]), torch.randn(2, 3, 6, 6))
        self.check(MeanModule([2, 3]), nhwc(torch.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2]), nhwc(torch.randn(2, 3, 6, 6)))
        self.check(MeanModule([-1, -2], keep=True), nhwc(torch.randn(2, 3, 6, 6)))

    def test_max_pool2d(self):
        # 遍历使用 float、quantize 和 nhwc 转换后的输入张量
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            with self.subTest(name):
                # 调用 self.check 方法，测试 MaxPool2d 模块的输出
                self.check(torch.nn.MaxPool2d(2), inp)
                self.check(torch.nn.MaxPool2d((3, 4)), inp)
                self.check(torch.nn.MaxPool2d((3, 4), (1, 2)), inp)
    # 定义测试方法，用于测试 AvgPool2d 模块
    def test_avg_pool2d(self):
        # 针对不同的输入数据和量化方式进行测试
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            # 使用子测试名称标识当前测试用例
            with self.subTest(name):
                # 设置误差容限和限制值为初始值
                atol_rtol = None
                limit = None
                # 初始化用于转换的维度和参数
                convert_dims = (2, 3, 0, 0)
                convert_arg = torch.zeros(*convert_dims)

                # 遍历不同的 AvgPool2d 模型配置
                for model in (
                    torch.nn.AvgPool2d(2),
                    torch.nn.AvgPool2d((3, 4)),
                    torch.nn.AvgPool2d((3, 4), (1, 2)),
                ):
                    # 如果模型名称中包含 "quant"，设置特定的误差容限和限制值
                    if "quant" in name:
                        atol_rtol = (1, 0)
                        limit = model(inp).numel()
                        # 使用 qpt 方法生成转换参数
                        convert_arg = qpt(torch.zeros(*convert_dims), 1.0 / 16, 128)
                    # 如果模型名称中包含 "nhwc"，调用 nhwc 方法对转换参数进行处理
                    if "nhwc" in name:
                        convert_arg = nhwc(convert_arg)

                    # 调用 self.check 方法进行模型检验，验证模型输出是否正确
                    self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                    # 再次调用 self.check 方法，使用转换参数进行检验
                    self.check(
                        model,
                        inp,
                        convert_args=[convert_arg],
                        atol_rtol=atol_rtol,
                        limit=limit,
                    )

    # 定义测试方法，用于测试 AdaptiveAvgPool2d 模块
    def test_adaptive_avg_pool2d(self):
        # 针对不同的输入数据和量化方式进行测试
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            # 使用子测试名称标识当前测试用例
            with self.subTest(name):
                # 检验是否能正确处理指定大小的输出
                self.check(torch.nn.AdaptiveAvgPool2d((1, 1)), inp)
                # 检验是否能正确抛出包含特定文本的异常
                with self.assertRaisesRegex(Exception, "with output size"):
                    self.check(torch.nn.AdaptiveAvgPool2d((2, 2)), inp)

    # 定义测试方法，用于测试 UpsamplingNearest2d 模块
    def test_upsample_nearest2d(self):
        # 使用 self.float_and_quant_and_nhwc 方法获取转换参数字典
        convert_args = dict(
            self.float_and_quant_and_nhwc(torch.randn(2, 3, 0, 0), 0.3, 128)
        )
        # 针对不同的输入数据和量化方式进行测试
        for name, inp in self.float_and_quant_and_nhwc(
            torch.randn(2, 3, 12, 16), 0.3, 128
        ):
            # 使用子测试名称标识当前测试用例
            with self.subTest(name):
                # 检验是否能正确上采样到指定大小
                self.check(torch.nn.UpsamplingNearest2d(size=(16, 20)), inp)
                self.check(torch.nn.UpsamplingNearest2d(size=(24, 32)), inp)
                self.check(torch.nn.UpsamplingNearest2d(size=(36, 48)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(1.5, 1.5)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)), inp)
                self.check(torch.nn.UpsamplingNearest2d(scale_factor=(3.0, 3.0)), inp)

                # 使用转换参数字典中的参数调用 self.check 方法进行检验
                self.check(
                    torch.nn.UpsamplingNearest2d(size=(24, 32)),
                    inp,
                    convert_args=[convert_args[name]],
                )
                self.check(
                    torch.nn.UpsamplingNearest2d(scale_factor=(2.0, 2.0)),
                    inp,
                    convert_args=[convert_args[name]],
                )
    # 定义一个测试线性层的方法
    def test_linear(self):
        # 设置随机种子为29，确保结果可重复性
        torch.manual_seed(29)
        # 测试创建一个输入维度为(2, 16)的线性层(16输入，32输出)，并检查其输出
        self.check(torch.nn.Linear(16, 32), torch.randn(2, 16))
        # 测试创建一个输入维度为(2, 16)的线性层(16输入，32输出)，并检查其输出，
        # 同时将额外的参数传递给检查函数，这里传入一个全零的Tensor
        self.check(
            torch.nn.Linear(16, 32),
            torch.randn(2, 16),
            convert_args=[torch.zeros(0, 16)],
        )

    # 定义一个测试转置卷积层的方法
    def test_conv2d_transpose(self):
        # 设置随机种子为29，确保结果可重复性
        torch.manual_seed(29)
        # 定义输入通道数(in_ch)、输出通道数(out_ch)、卷积核大小(kernel)
        in_ch, out_ch, kernel = (5, 7, (2, 2))
        # 定义输入数据的维度(input_dim)，这里是一个四维的Tensor
        input_dim = (4, 5, 3, 3)
        # 定义用于转换的维度(convert_dims)，从input_dim中取前两维，并在后面填充0
        convert_dims = input_dim[:2] + (0, 0)

        # 遍历不同的模式，例如"float", "float-nhwc", "quant", "quant-nhwc"
        for kind in ["float", "float-nhwc", "quant", "quant-nhwc"]:
            # 使用self.subTest(kind)进行子测试，kind为当前测试模式
            with self.subTest(kind):
                # 创建一个输入维度为input_dim的随机Tensor
                inp = torch.randn(input_dim)
                # 创建一个转置卷积层模型，输入通道为in_ch，输出通道为out_ch，卷积核大小为kernel
                model = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel)
                # 计算模型对输入inp的输出元素数量
                output_size = model(inp).numel()
                # 设置绝对误差和相对误差的容忍度
                atol_rtol = (0.0002, 0)
                # 初始化limit为None
                limit = None
                # 创建一个与输入维度相同的全零Tensor，作为转换参数
                convert_arg = torch.zeros(*convert_dims)

                # 如果当前模式中包含"quant"关键字
                if "quant" in kind:
                    # 使用量化后的转置卷积层模型
                    model = torch.ao.nn.quantized.ConvTranspose2d(in_ch, out_ch, kernel)
                    # 设置模型的量化配置
                    model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
                    # 对输入inp进行量化处理
                    inp = qpt(inp, 1.0 / 16, 128)
                    # 设置量化模式下的绝对误差和相对误差的容忍度
                    atol_rtol = (1, 0)
                    # 设置限制值为输出元素数量的10%
                    limit = output_size * 0.1
                    # 对转换参数进行量化处理
                    convert_arg = qpt(convert_arg, 1.0 / 16, 128)

                # 如果当前模式中包含"nhwc"关键字
                if "nhwc" in kind:
                    # 对输入inp进行NHWC格式的转换
                    inp = nhwc(inp)
                    # 对转换参数进行NHWC格式的转换
                    convert_arg = nhwc(convert_arg)

                # 使用self.check方法检查模型的输出
                self.check(model, inp, atol_rtol=atol_rtol, limit=limit)
                # 使用self.check方法检查模型的输出，
                # 同时传入额外的转换参数和误差容忍度
                self.check(
                    model,
                    inp,
                    convert_args=[convert_arg],
                    atol_rtol=atol_rtol,
                    limit=limit,
                )
    def test_qadd(self):
        # 创建量化功能函数对象
        func = torch.ao.nn.quantized.QFunctional()
        # 设置量化参数：缩放因子和零点偏移量
        func.scale = 0.5
        func.zero_point = 120

        # 定义加法模块
        class AddMod(torch.nn.Module):
            def forward(self, lhs, rhs):
                return func.add(lhs, rhs)

        # 定义带ReLU的加法模块
        class AddReluMod(torch.nn.Module):
            def forward(self, lhs, rhs):
                return func.add_relu(lhs, rhs)

        # 定义乘法模块
        class MulMod(torch.nn.Module):
            def forward(self, lhs, rhs):
                return func.mul(lhs, rhs)

        # 遍历测试模块名称和对应的模块类
        for name, mod in [("add", AddMod), ("add_relu", AddReluMod), ("mul", MulMod)]:
            with self.subTest(name):
                # 执行自定义检查函数，测试模块的输出
                self.check(
                    mod(),
                    [
                        # 创建量化张量，指定数据、缩放因子和零点偏移量
                        qpt([1.0, 2.0], 0.25, 128),
                        qpt([3.0, 4.0], 0.25, 128),
                    ],
                )
                # 执行自定义检查函数，测试模块的输出和输入转换参数
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        qpt([[1.0, 2.0]], 0.25, 128),
                        # 创建全零的量化张量，指定缩放因子和零点偏移量
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                    ],
                )
                # 执行自定义检查函数，测试模块的输出和输入转换参数
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        # 创建全零的量化张量，指定缩放因子和零点偏移量
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                )
                # 执行自定义检查函数，测试模块的输出和输入转换参数
                self.check(
                    mod(),
                    [
                        qpt([[1.0, 2.0]], 0.25, 128),
                        qpt([[3.0, 4.0]], 0.25, 128),
                    ],
                    convert_args=[
                        # 创建全零的量化张量，指定缩放因子和零点偏移量
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                        qpt(torch.zeros((1, 2)), 0.25, 128),
                    ],
                )
                # 注意: NNAPI qadd 支持广播，但是 PyTorch 不支持。
    # 定义一个测试方法，用于测试多输出模型的功能
    def test_multi_output(self):
        # 定义一个继承自 torch.nn.Module 的多输出模型类
        class MultiModel(torch.nn.Module):
            # 定义模型的前向传播方法，接收两个输入参数，返回两个张量的元组
            def forward(self, lhs, rhs) -> Tuple[torch.Tensor, torch.Tensor]:
                # 计算输入张量的和
                the_sum = lhs + rhs
                # 计算输入张量的差
                the_diff = lhs - rhs
                # 返回计算结果的元组（和，差）
                return the_sum, the_diff
        
        # 调用 self.check 方法，验证 MultiModel 的输出是否符合预期
        self.check(MultiModel(), [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])])
# 如果这个脚本是作为主程序运行
if __name__ == "__main__":
    # 调用函数 run_tests() 来执行测试
    run_tests()
```