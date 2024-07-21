# `.\pytorch\test\inductor\test_cpu_repro.py`

```
# 导入必要的库和模块
import contextlib  # 提供上下文管理功能的模块
import copy  # 提供对象复制功能的模块
import functools  # 提供函数工具的模块
import itertools  # 提供迭代工具的模块
import math  # 提供数学函数的模块
import platform  # 提供平台信息的模块
import sys  # 提供与 Python 解释器交互的模块
import unittest  # 提供单元测试框架的模块
from typing import Callable  # 引入类型提示中的 Callable 类型
from unittest.mock import patch  # 提供单元测试 mock 功能的模块

import numpy as np  # 数组处理工具包
import sympy  # 符号计算库

import torch  # PyTorch 深度学习框架
from torch import nn  # 神经网络模块
from torch._C import FileCheck  # Torch C++ API 的文件检查工具
from torch._dynamo.testing import rand_strided  # Torch 内部测试工具
from torch._dynamo.utils import same  # Torch 内部工具，用于比较对象是否相同
from torch._inductor import config, cpu_vec_isa, metrics, test_operators  # Torch 感应器模块
from torch._inductor.codegen.common import OptimizationContext  # 优化上下文
from torch._inductor.codegen.cpp import (
    CppOverrides,  # C++ 代码生成的重写
    CppVecKernelChecker,  # C++ 向量化内核检查器
    CppVecOverrides,  # C++ 向量化重写
)
from torch._inductor.compile_fx import (
    compile_fx,  # 编译 FX 模型
    compile_fx_inner,  # 内部 FX 编译函数
    complex_memory_overlap,  # 复杂内存重叠
)
from torch._inductor.graph import GraphLowering  # 图降低
from torch._inductor.ir import InterpreterShim  # 解释器包装
from torch._inductor.utils import timed  # 计时工具
from torch._inductor.virtualized import V  # 虚拟化工具
from torch.fx.experimental.proxy_tensor import make_fx  # 创建 FX 张量的代理
from torch.nn import functional as F  # 神经网络的函数接口
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 实例化参数化测试
    IS_MACOS,  # 是否在 macOS 上
    parametrize,  # 参数化装饰器
    skipIfRocm,  # 如果是 ROCm 平台，则跳过测试
    slowTest,  # 标记为慢速测试
)
from torch.utils._python_dispatch import TorchDispatchMode  # Torch 分发模式

try:
    try:
        from . import test_torchinductor  # 尝试从当前目录导入 test_torchinductor 模块
    except ImportError:
        import test_torchinductor  # 如果导入失败，则从全局导入 test_torchinductor 模块
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)  # 在主程序中，如果是单元测试跳过，则退出程序
    raise  # 否则抛出异常

vec_dtypes = test_torchinductor.vec_dtypes  # 导入向量数据类型列表
_lowp_fp_dtypes = (
    torch.bfloat16,  # 低精度浮点数类型 bfloat16
    torch.float16,  # 半精度浮点数类型 float16
)
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code  # 运行并获取 C++ 代码的函数引用
TestCase = test_torchinductor.TestCase  # 测试用例基类
aten = torch.ops.aten  # 导入 Torch aten 操作
check_model = test_torchinductor.check_model  # 检查模型的函数引用

requires_vectorization = unittest.skipUnless(
    cpu_vec_isa.valid_vec_isa_list(),  # 当 CPU 支持向量化指令集时才执行的装饰器
    "Does not support vectorization"  # 不支持向量化时的提示信息
)


def check_metrics_vec_kernel_count(num_expected_vec_kernels):
    if cpu_vec_isa.valid_vec_isa_list():
        assert metrics.generated_cpp_vec_kernel_count == num_expected_vec_kernels
        # 检查生成的 C++ 向量化内核数量是否符合预期


@contextlib.contextmanager
def set_num_threads(num_threads):
    orig_num_threads = torch.get_num_threads()  # 获取当前线程数
    torch.set_num_threads(num_threads)  # 设置线程数为指定值
    yield  # 执行代码块
    torch.set_num_threads(orig_num_threads)  # 恢复原始线程数


class LstmModule(torch.nn.Module):
    def __init__(
        self,
        input_size,  # 输入大小
        hidden_size,  # 隐藏层大小
        num_layers,  # 层数
        bias=True,  # 是否使用偏置
        bidirectional=False,  # 是否双向
        batch_first=False,  # 是否批处理优先
    ):
        super().__init__()  # 调用父类初始化方法
        self.lstm = torch.nn.LSTM(
            input_size=input_size,  # 设置输入大小
            hidden_size=hidden_size,  # 设置隐藏层大小
            num_layers=num_layers,  # 设置层数
            bias=bias,  # 设置是否使用偏置
            bidirectional=bidirectional,  # 设置是否双向
            batch_first=batch_first,  # 设置是否批处理优先
        )

    def forward(self, x, h=None):
        x, h = self.lstm(x, h)  # LSTM 前向传播
        return x, h  # 返回输出和隐藏层状态


@instantiate_parametrized_tests
class CPUReproTests(TestCase):
    common = check_model  # 使用 check_model 函数作为通用的测试方法

    @skipIfRocm
    # 定义一个测试方法，用于测试卷积层的步长约束
    def test_conv_stride_constraints(self):
        # 针对不同的内存布局格式进行测试：连续格式和通道最后格式
        for fmt in [torch.contiguous_format, torch.channels_last]:
            # 创建一个卷积层，输入通道数为5，输出通道数为6，卷积核大小为3x3
            # 此处仅用于测试目的，不会在我们的 CUDA 调用中工作
            m = torch.nn.Conv2d(5, 6, [3, 3])

            # 定义一个内部函数 fn，用于执行卷积操作
            def fn(inp, weight):
                return (
                    F.conv2d(
                        inp, weight, None, m.stride, m.padding, m.dilation, m.groups
                    ),
                )

            # 创建一个随机输入张量，形状为 [2, 5, 16, 16]
            inp = torch.randn([2, 5, 16, 16])
            # 将输入张量和卷积层权重转换为指定的内存布局格式
            inps = [inp, m.weight.to(memory_format=fmt)]
            # 使用 make_fx 将 fn 函数转换为 FX 函数
            fn_fx = make_fx(fn)(*inps)
            # 编译 FX 函数的内部实现
            fn_compiled = compile_fx_inner(fn_fx, inps)
            # 获取当前测试实例的引用
            test_self = self
            # 初始化卷积层是否被观察到的标志为 False
            conv_seen = False

            # 定义一个用于记录函数调用的类 RecordFunctions，继承自 TorchDispatchMode
            class RecordFunctions(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs if kwargs else {}
                    # 检查是否调用了 torch.ops.aten.convolution.default 函数
                    if func == torch.ops.aten.convolution.default:
                        # 如果当前为 CPU 并且 MKLDNN 可用，则始终使用通道最后格式
                        nonlocal fmt
                        if (
                            torch.backends.mkldnn.enabled
                            and torch.backends.mkldnn.is_available()
                        ):
                            fmt = torch.channels_last
                        # 断言输入的张量在指定的内存布局格式下是连续的
                        test_self.assertTrue(args[0].is_contiguous(memory_format=fmt))
                        test_self.assertTrue(args[1].is_contiguous(memory_format=fmt))
                        # 设置卷积层被观察到的标志为 True
                        nonlocal conv_seen
                        conv_seen = True

                    return func(*args, **kwargs)

            # 使用 RecordFunctions 类记录函数调用
            with RecordFunctions():
                out = fn_compiled(inps)

            # 断言卷积层是否被观察到
            self.assertTrue(conv_seen)

    # 使用 unittest 的 patch 装饰器，禁用 CUDA 并定义一个测试方法，测试混合数据类型的卷积和批归一化
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_bn_mixed_dtype(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个使用 bfloat16 数据类型的卷积层，输入通道数为3，输出通道数为16
                self.conv = torch.nn.Conv2d(
                    3,
                    16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    dtype=torch.bfloat16,
                )
                # 创建一个批归一化层，对输入通道数为16的特征图进行批归一化
                self.bn = torch.nn.BatchNorm2d(
                    16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
                )

            # 定义前向传播函数
            def forward(self, x):
                x = self.conv(x)  # 使用卷积层处理输入
                x = self.bn(x)    # 使用批归一化层处理卷积输出
                return x

        # 创建一个形状为 [1, 3, 64, 64] 的 bfloat16 类型的随机输入张量
        v = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)
        # 实例化模型 Model，并设置为评估模式
        mod = Model().eval()
        # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
        with torch.no_grad():
            # 调用通用的测试函数 common，传入模型和输入数据
            self.common(
                mod,
                (v,),
            )

    # 使用 unittest 的 skipIf 装饰器，检查 MKLDNN 是否可用，并定义一个测试方法
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    # 使用 unittest 的 patch 装饰器，禁用 CUDA 并定义一个测试方法
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_packed(self):
        # 生成所有可能的选项组合，包括输入张量形状、训练模式和填充方式
        options = itertools.product([[3, 56, 56]], [True, False], [0, (0,)])
        # 对每个选项组合执行测试
        for x_shape, mode_train, padding in options:
            # 创建一个包含单个卷积层的序列模型，并设置为指定的训练模式
            mod = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, 3, padding=padding)
            ).train(mode=mode_train)
            # 生成指定形状的随机张量作为输入数据
            v = torch.randn(x_shape, dtype=torch.float32)

            # 在无梯度计算环境下，调用共通的测试方法
            with torch.no_grad():
                self.common(
                    mod,
                    (v,),
                )

    @patch("torch.cuda.is_available", lambda: False)
    def test_conv2d_autocast(self):
        # 生成指定形状的随机张量作为输入数据
        v = torch.randn(1, 3, 28, 18, dtype=torch.float32)
        # 创建一个包含单个卷积层的序列模型，并设置为评估模式
        mod = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 3)).eval()
        # 在无梯度计算环境下，使用自动混合精度进行计算
        with torch.no_grad(), torch.cpu.amp.autocast():
            self.common(
                mod,
                (v,),
            )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_unsupported_conv_transpose(self):
        # 定义一个简单的模型类，包含一个反卷积层
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    3, 6, 3, stride=1, padding=1, output_padding=1
                )

            def forward(self, input_tensor):
                # 执行反卷积操作并对输出进行 Tanh 激活
                x = self.conv_transpose(input_tensor)
                output = torch.tanh(x)
                return output

        # 生成指定形状的随机输入张量
        input = torch.randn(1, 3, 28, 28)
        # 创建并评估模型实例
        m = Model().eval()

        # 在无梯度计算环境下，编译模型并检查是否抛出预期的运行时异常
        with torch.no_grad():
            compiled_m = torch.compile(m)
            with self.assertRaisesRegex(
                RuntimeError,
                "output padding must be smaller than either stride or dilation",
            ):
                compiled_m(input)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_used_from_multiple_places(self):
        # 定义一个包含多处使用相同卷积层的模型类
        class M(torch.nn.Module):
            def __init__(self, conv_in_channel, conv_out_channel) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(conv_in_channel, conv_out_channel, (3, 3))

            def forward(self, x):
                # 执行两次卷积操作，并在中间结果上应用 ReLU 激活
                res = self.conv(x)
                res = F.relu(res)
                res = self.conv(res)
                return res

        # 在无梯度计算环境下，创建并评估模型实例，生成指定形状的随机输入张量
        with torch.no_grad():
            mod = M(3, 3).eval()
            x = torch.randn(1, 3, 224, 224)
            self.common(
                mod,
                (x,),
            )
    def test_linear_used_from_multiple_places(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化方法，接受输入通道数和输出通道数
            def __init__(self, in_channel, out_channel) -> None:
                super().__init__()
                # 创建一个线性层，输入通道数为 in_channel，输出通道数为 out_channel
                self.linear = torch.nn.Linear(in_channel, out_channel)

            # 前向传播方法，接受输入 x
            def forward(self, x):
                # 将输入 x 传入线性层
                res = self.linear(x)
                # 对结果应用 ReLU 激活函数
                res = F.relu(res)
                # 再次将结果传入线性层
                res = self.linear(res)
                return res

        dtypes = []
        # 检查是否支持 MKL-DNN 的 bfloat16 类型，并添加到 dtypes 列表
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 检查是否支持 MKL-DNN 的 float16 类型，并添加到 dtypes 列表
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 遍历 dtypes 列表中的每种数据类型
        for dtype in dtypes:
            with torch.no_grad():
                # 创建 M 类的实例 m，并将其移动到指定的 dtype 类型，并设为评估模式
                m = M(224, 224).to(dtype).eval()
                # 对模型 m 进行编译优化
                m_opt = torch.compile(m)
                # 创建一个指定类型和形状的随机张量 x
                x = torch.randn(224, 224, dtype=dtype)
                # 调用优化后的模型 m_opt 进行推理
                m_opt(x)
                # 使用断言检查模型 m 和 m_opt 在输入 x 上的输出是否相同
                self.assertEqual(m(x), m_opt(x))

    @config.patch(implicit_fallbacks=True)
    def test_multihead_attention_cpu(self):
        # 定义一个函数 fn，实现多头注意力机制
        def fn(
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ):
            # 调用底层的多头注意力实现函数 torch._native_multi_head_attention
            return torch._native_multi_head_attention(
                q,
                k,
                v,
                embed_dim,
                num_heads,
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                mask,
                need_weights,
            )

        B = 1
        T = 3
        embed_dim = 6
        num_heads = 2
        # 创建形状为 [B, T, embed_dim] 的随机张量 q、k 和 v
        q = torch.randn([B, T, embed_dim])
        k = torch.randn([B, T, embed_dim])
        v = torch.randn([B, T, embed_dim])
        # 创建形状为 [3 * embed_dim, embed_dim] 的随机张量 qkv_weight
        qkv_weight = torch.randn([3 * embed_dim, embed_dim])
        # 创建形状为 [3 * embed_dim] 的随机张量 qkv_bias
        qkv_bias = torch.randn([3 * embed_dim])
        # 创建形状为 [3 * embed_dim, embed_dim] 的随机张量 proj_weight
        proj_weight = torch.randn([3 * embed_dim, embed_dim])
        # 创建形状为 [3 * embed_dim] 的随机张量 proj_bias
        proj_bias = torch.randn([3 * embed_dim])
        # 初始化 mask 为 None
        mask = None
        # 设置 need_weights 为 False
        need_weights = False

        # 将输入参数封装为列表 inps
        inps = [
            q,
            k,
            v,
            embed_dim,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            mask,
            need_weights,
        ]
        # 调用 self.common 方法，测试 fn 函数的输出
        self.common(fn, inps)

    @config.patch(freezing=True)
    def test_module_buffer_mutation(self):
        # 定义一个名为 Model 的子类，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为 "foo" 的模型缓冲区，形状为 (3, 10)，内容为随机数
                self.register_buffer("foo", torch.rand((3, 10)))

            def forward(self, x):
                # 创建包含 x、x 的克隆以及 x 的克隆的列表 lx
                lx = [x, x.clone(), x.clone()]
                y = []
                # 遍历 lx 中的每个张量，与模型缓冲区 "foo" 的每行相加
                for i in range(3):
                    y.append(lx[i] + self.foo[i])
                # 将列表 y 中的张量按列拼接成一个张量，并返回
                return torch.cat(y, 1)

        with torch.no_grad():
            # 创建一个示例输入 example_inputs，形状为 (1, 10) 的随机张量
            example_inputs = (torch.rand(1, 10),)
            # 调用 self.common 方法，测试 Model 类的 forward 方法
            self.common(Model(), example_inputs)
    # 使用装饰器跳过测试，如果没有启用 MKLDNN
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    # 使用模拟函数来覆盖 torch.cuda.is_available 函数，返回 False
    @patch("torch.cuda.is_available", lambda: False)
    # 定义测试函数：测试线性层的打包情况
    def test_linear_packed(self):
        # 初始化一个空列表，用于存储数据类型
        dtypes = []
        # 如果支持 MKLDNN 的 BF16 类型，则添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持 MKLDNN 的 FP16 类型，则添加到 dtypes 列表中
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 生成各种选项的组合，包括输入形状、输出维度、偏置、数据类型
        options = itertools.product(
            [[2, 3, 10], [2, 10], [10], [2, 0]], [3, 0], [True, False], dtypes
        )
        # 遍历所有选项
        for input_shape, out_dim, bias, dtype in options:
            # 创建一个包含单个线性层的序列模型，并将其设为评估模式
            mod = torch.nn.Sequential(
                torch.nn.Linear(input_shape[-1], out_dim, bias=bias)
            ).eval()
            # 生成一个具有指定形状的随机张量 v
            v = torch.randn(input_shape)
            # 在无梯度计算的上下文中执行公共方法
            with torch.no_grad():
                self.common(
                    mod.to(dtype),  # 将模型转换为指定的数据类型
                    (v.to(dtype),),  # 将输入数据转换为指定的数据类型
                )

    # 使用装饰器跳过测试，如果没有启用 MKLDNN
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    # 使用模拟函数来覆盖 torch.cuda.is_available 函数，返回 False
    @patch("torch.cuda.is_available", lambda: False)
    # 定义测试函数：测试 ConvTranspose2d 的打包情况（在 CPU 上）
    def test_conv_transpose2d_packed_cpu(self):
        # 生成各种选项的组合，包括输入张量的形状和填充参数
        options = itertools.product([[1, 3, 28, 28], [3, 28, 28]], [0, (0,)])
        # 遍历所有选项
        for x_shape, padding in options:
            # 创建一个包含单个反卷积层的序列模型，并将其设为评估模式
            mod = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(3, 64, 3, 3, padding=padding)
            ).eval()
            # 生成一个指定数据类型的随机张量 v
            v = torch.randn(x_shape, dtype=torch.float32)
            # 在无梯度计算的上下文中执行公共方法
            with torch.no_grad():
                self.common(
                    mod,  # 使用创建的反卷积层模型
                    (v,),  # 输入数据为 v
                )

    # 使用配置装饰器来启用动态形状（dynamic_shapes）和静态假设（assume_static_by_default=False）
    @config.patch(freezing=True)
    # 使用装饰器跳过测试，如果没有启用 MKLDNN
    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    # 使用 Torch Dynamo 配置装饰器来启用动态形状
    @torch._dynamo.config.patch(dynamic_shapes=True)
    # 使用 Torch Dynamo 配置装饰器来禁用默认静态假设
    @torch._dynamo.config.patch(assume_static_by_default=False)
    # 定义测试函数：测试具有动态形状的 Conv2d 中的通道数为 1 的情况
    def test_conv_in_channel_1_dynamic_shapes(self):
        # 定义一个简单的模块类 M，包含一个 Conv2d 层
        class M(torch.nn.Module):
            def __init__(self, in_channel, out_channel) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channel, out_channel, 3)

            def forward(self, x):
                res = self.conv(x)
                res = F.relu(res)
                return res

        # 测试输入通道数为 1 的情况
        # 从 Torchbench 的 maml_omniglot 模型中复现
        in_channel = 1
        out_channel = 3
        # AMP 是否启用的配置列表
        amp_enabled_configs = [False]
        # 如果支持 MKLDNN 的 BF16 类型，则将 True 添加到配置列表中
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            amp_enabled_configs.append(True)
        # 遍历所有 AMP 是否启用的配置
        for amp_enabled in amp_enabled_configs:
            # 创建一个具有指定输入通道数和输出通道数的 M 模块，并将其设为评估模式
            mod = M(in_channel, out_channel).eval()
            # 生成一个指定形状的随机张量 v
            v = torch.randn(5, in_channel, 15, 15)
            # 在无梯度计算的上下文中执行公共方法
            with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
                self.common(
                    mod,  # 使用创建的模块
                    (v,),  # 输入数据为 v
                )
    # 使用装饰器跳过测试，如果未启用 MKLDNN
    @unittest.skipIf(not torch._C._has_mkldnn, "MKLDNN is not enabled")
    # 使用模拟替换函数，始终返回 False，模拟 CUDA 不可用
    @patch("torch.cuda.is_available", lambda: False)
    # 使用动态形状配置修补装饰器
    @torch._dynamo.config.patch(dynamic_shapes=True)
    # 使用默认假设为静态形状配置修补装饰器
    @torch._dynamo.config.patch(assume_static_by_default=False)
    # 允许 RNN 配置修补装饰器
    @torch._dynamo.config.patch(allow_rnn=True)
    # 冻结配置修补装饰器
    @config.patch(freezing=True)
    # 标记为慢速测试
    @slowTest
    # LSTM 打包测试函数
    def test_lstm_packed(self):
        # 参数字典定义不同参数的组合
        params_dict = {
            "unbatched": [True, False],         # 是否不批处理
            "input_size": [1, 2],               # 输入大小
            "hidden_size": [2],                 # 隐藏层大小
            "num_layers": [1, 2],               # LSTM 层数
            "bidirectional": [False, True],     # 是否双向
            "bias": [False, True],              # 是否有偏置
            "empty_state": [False, True],       # 是否空状态
            "batch_first": [True, False],       # 是否批处理优先
            "batch_size": [1, 2],               # 批处理大小
            "seq_len": [1, 2],                  # 序列长度
        }
        # 调用内部方法以测试 LSTM 打包
        self._test_lstm_packed(params_dict)

    # 测试改变输入大小的 LSTM 打包函数（CPU 版本）
    def test_lstm_packed_change_input_sizes_cpu(self):
        # 参数字典定义不同参数的组合
        params_dict = {
            "unbatched": [False],               # 是否不批处理
            "input_size": [2],                  # 输入大小
            "hidden_size": [5],                 # 隐藏层大小
            "num_layers": [3],                  # LSTM 层数
            "bidirectional": [True],            # 是否双向
            "bias": [True],                     # 是否有偏置
            "empty_state": [False],             # 是否空状态
            "batch_first": [False],             # 是否批处理优先
            "batch_size": [2],                  # 批处理大小
            "seq_len": [3],                     # 序列长度
        }
        # 调用内部方法以测试改变输入大小的 LSTM 打包
        self._test_lstm_packed(params_dict, change_input_sizes=True)

    # 使用动态形状配置修补装饰器
    @torch._dynamo.config.patch(dynamic_shapes=True)
    # 使用默认假设为静态形状配置修补装饰器
    @torch._dynamo.config.patch(assume_static_by_default=False)
    # 允许 RNN 配置修补装饰器
    @torch._dynamo.config.patch(allow_rnn=True)
    # 测试打包填充序列 LSTM
    def test_pack_padded_sequence_lstm(self):
        # 嵌入维度
        embedding_dim = 12
        # 隐藏层维度
        hidden_dim = 10
        # 批处理大小
        batch_size = 24
        # LSTM 层数
        num_layers = 1
        # 是否双向
        bidirectional = True
        # 双向数目
        num_direc = 2
        # 最大长度
        max_lens = 96

        # 创建随机张量表示句子的嵌入
        sent = torch.randn(batch_size, max_lens, embedding_dim)
        # 创建具有随机数的张量表示隐藏状态
        hid_0 = torch.rand(num_layers * num_direc, batch_size, hidden_dim)
        hid_1 = torch.randn(num_layers * num_direc, batch_size, hidden_dim)

        # 句子长度张量
        sent_lens = torch.Tensor(
            [1, 2, 3, 4, 5, 1, 3, 2, 96, 5, 3, 1, 1, 2, 1, 2, 3, 6, 1, 2, 4, 6, 2, 1]
        )

        # 断言句子长度张量的形状与批处理大小相等
        assert sent_lens.shape[0] == batch_size
        # 断言句子长度张量的最大值等于最大长度
        assert sent_lens.max().item() == max_lens

        # 克隆隐藏状态张量，不需要梯度
        hidden_0 = hid_0.clone().requires_grad_(False)
        hidden_1 = hid_1.clone().requires_grad_(False)
        
        # 对句子进行打包填充序列处理，按批处理优先，不需要排序
        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            sent, sent_lens, batch_first=True, enforce_sorted=False
        )

        # 创建 LSTM 模块，用于测试
        mod = LstmModule(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bias=True,
            bidirectional=bidirectional,
            batch_first=True,
        ).eval()

        # 在无梯度计算环境中
        with torch.no_grad():
            # 输入参数列表
            inps = [embeds, (hidden_0, hidden_1)]
            # 优化模块功能，获取其生成的 C++ 代码
            fn_opt = torch._dynamo.optimize("inductor")(mod)
            # 断言生成的 C++ 代码中不包含不支持的 MKLDNN LSTM 操作
            _, code = run_and_get_cpp_code(fn_opt, *inps)
            self.assertFalse("torch.ops.mkldnn._lstm" in code)
            # 断言优化后的功能与模块的功能一致
            self.assertEqual(fn_opt(*inps), mod(*inps))
    @patch("torch.cuda.is_available", lambda: False)
    def test_conv_transpose2d_has_output_size_input(self):
        # 定义一个测试函数，用于验证ConvTranspose2d的输出大小是否正确处理输入。
        # 这里是为了解决GitHub上的特定问题：https://github.com/pytorch/pytorch/issues/100344.
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 创建一个ConvTranspose2d层，设置输入通道为3，输出通道为1，内核大小为3，步长为1，填充为1。
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1
                )
    
            def forward(self, x):
                # 在前向传播中使用conv_transpose层，指定输出大小为(10, 10)。
                return self.conv_transpose(x, output_size=(10, 10))
    
        # 创建M类的实例mod，并将其设置为评估模式。
        mod = M().eval()
        # 创建一个形状为(1, 3, 10, 10)的随机张量v，数据类型为torch.float32。
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        # 使用torch.no_grad()上下文管理器，运行测试用例self.common。
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )
    
    def test_pad_with_nan_value(self):
        # 定义一个测试函数，用于验证F.pad函数在填充值为NaN时的行为。
        # 这里是为了解决GitHub上的特定问题：https://github.com/pytorch/pytorch/issues/100988.
        class Model(torch.nn.Module):
            def forward(self, x):
                # 使用F.pad函数，在x的四个边界各填充1个单位，填充值为float("nan")。
                x = F.pad(x, (1, 1, 1, 1), value=float("nan"))
                return x
    
        # 创建Model类的实例mod，并将其设置为评估模式。
        mod = Model().eval()
        # 创建一个形状为(1, 3, 10, 10)的随机张量v，数据类型为torch.float32。
        v = torch.randn(1, 3, 10, 10, dtype=torch.float32)
        # 使用torch.no_grad()上下文管理器，运行测试用例self.common。
        with torch.no_grad():
            self.common(
                mod,
                (v,),
            )
    
    def test_masked_fill_with_inf_or_nan_value(self):
        # 定义一个测试函数，验证torch.masked_fill函数在使用inf或nan值时的行为。
        def fn(value, mask):
            # 使用torch.masked_fill函数，将value中mask为True的元素分别替换为float("inf"), float("-inf"), float("nan")。
            y1 = torch.masked_fill(value, mask, float("inf"))
            y2 = torch.masked_fill(value, mask, float("-inf"))
            y3 = torch.masked_fill(value, mask, float("nan"))
            return y1, y2, y3
    
        # 创建一个形状为(2, 17)的随机张量value。
        value = torch.randn((2, 17))
        # 创建一个形状为(2, 17)的随机布尔掩码mask。
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        # 使用torch.no_grad()上下文管理器，运行测试用例self.common。
        with torch.no_grad():
            self.common(
                fn,
                (value, mask),
            )
    
    def test_relu_with_inf_value(self):
        # 定义一个测试函数，验证torch.relu函数在处理输入为inf值时的行为。
        # 这里是为了解决GitHub上的特定问题：https://github.com/pytorch/pytorch/issues/117544.
    
        def fn(out):
            # 对输入张量out进行sinh操作。
            out = torch.sinh(input=out)
            # 对结果应用torch.relu函数。
            out = torch.relu(input=out)
            return out
    
        # 创建一个包含一组具体值的张量x。
        x = torch.Tensor([-572373.5000, 755109.1250, 330995.5625])
        # 使用torch.no_grad()上下文管理器，运行测试用例self.common。
        with torch.no_grad():
            self.common(
                fn,
                (x,),
            )
    def test_acosh_with_negative_large_input(self):
        # 解决 GitHub 问题 https://github.com/pytorch/pytorch/issues/118267.
        
        def fn(input):
            # 计算反双曲余弦函数
            out = torch.acosh(input)
            return out

        # 创建一个包含重复数据的张量 x
        x = torch.Tensor(
            [
                [
                    -8493.9854,
                    431654.1250,
                    71741.5859,
                    608234.5000,
                    -103814.7500,
                    -699397.0000,
                    -910685.8125,
                    -832737.1875,
                    875343.5000,
                ]
            ]
        ).repeat(3, 9)

        # 对不同数据类型进行循环测试
        for dtype in [torch.float32, torch.bfloat16, torch.double]:
            with torch.no_grad():
                # 重置动态图和度量指标
                torch._dynamo.reset()
                metrics.reset()
                # 将张量 x 转换为指定数据类型 dtype
                _x = x.to(dtype)
                # 调用公共测试函数 common
                self.common(
                    fn,
                    (_x,),
                )

    @config.patch(implicit_fallbacks=True)
    def test_repeat_interleave(self):
        # 解决 GitHub 问题 https://github.com/pytorch/pytorch/issues/93365
        def fn(y):
            # 对张量进行重复插值操作
            return torch.repeat_interleave(y, 2, output_size=8)

        # 创建一个二维张量 a
        a = torch.tensor([[1, 2], [3, 4]])
        # 调用公共测试函数 common
        self.common(
            fn,
            (a,),
        )

    def test_inplace_squeeze_needed(self):
        # 创建一个包含多个层的序列模块 mod
        mod = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.LayerNorm(10),
            torch.nn.ReLU(),
        ).eval()

        def fn(x):
            # 对输入 x 进行模块 mod 的前向传播
            return mod(x)

        # 创建一个随机张量 v
        v = torch.randn(10)
        # TODO: OMP 并行归约顺序不确定，因此精度可能有所上下波动。
        # 因此，增加容差，稍后通过使用 aten 并行进行修复。
        self.common(fn, (v,), atol=5e-1, rtol=5e-1)

    def test_cat_mul(self):
        # 解决 GitHub 问题 https://github.com/pytorch/pytorch/issues/93365
        def fn(p0, p1):
            # 沿指定维度拼接张量 p0 和 p1
            y1 = torch.cat([p0, p1], dim=0)
            # 对拼接后的张量元素逐元素进行平方操作
            y2 = torch.mul(y1, y1)
            return y1, y2

        # 创建两个随机张量 p0 和 p1
        p0 = torch.randn(3, 4)
        p1 = torch.randn(3, 4)
        # 调用公共测试函数 common
        self.common(fn, (p0, p1))

    def test_pow_cos(self):
        # 解决 GitHub 问题 https://github.com/pytorch/pytorch/issues/98149
        def fn(x):
            # 对输入张量 x 进行五次幂运算，然后对结果张量进行余弦函数计算
            t = x.pow(5)
            return torch.cos(t)

        # 创建一个具有单个元素的无符号 8 位整型张量 x
        x = torch.tensor([4], dtype=torch.uint8)
        # 调用公共测试函数 common
        self.common(fn, (x,))

    def test_reduce_with_masked(self):
        # 解决 GitHub 问题 https://github.com/pytorch/pytorch/issues/96484
        def fn(a, b):
            # 对张量 a 进行零填充，然后与张量 b 相加
            a = torch.nn.functional.pad(a, (0, -1))
            c = a + b
            # 返回张量 c 的最小值
            return c.min(0).values

        # 创建两个随机张量 a 和 b
        a = torch.randn([2])
        b = torch.randn([2])
        # 调用公共测试函数 common
        self.common(fn, (a, b))

    def test_scalar_sign_with_min(self):
        # 解决 GitHub 问题 https://github.com/pytorch/pytorch/issues/101340
        def fn(a):
            # 对输入张量 a 进行双曲正切运算，然后取其符号，最后返回它和其最小值之间的较小值
            t1 = torch.tanh(a)
            t2 = torch.sign(t1)
            return torch.min(t1, t2)

        # 创建一个具有形状 (1, 3) 的随机张量 a
        a = torch.randn(1, 3)
        # 调用公共测试函数 common
        self.common(fn, (a,))
    # 定义测试函数：测试索引传播问题 #102065
    def test_index_propagation_issue_102065(self):
        # 定义内部函数 fn，接受参数 x
        def fn(x):
            # 生成一个从 0 到 x 中元素个数的张量 x
            x = torch.arange(x.numel())
            # 计算张量 x 扩展后的张量，求差的平方
            return (x.unsqueeze(0) - x.unsqueeze(1)) ** 2

        # 调用通用测试函数 common，传入 fn 函数和参数元组
        self.common(
            fn,
            (torch.randn(8),),
        )

    # 定义测试函数：测试模块化索引问题 #103133
    def test_ModularIndexing_range_issue_103133(self):
        # 定义内部函数 fn，接受参数 q 和 k
        def fn(q, k):
            # 执行 Einstein 求和运算
            einsum = torch.einsum("bcxd,bcyd->bcxy", (q, k))
            # 使用常量填充 nd 运算
            constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                einsum, [0, 0, 0, 1], 0.0
            )
            # 使用 view 运算重塑张量形状
            view = torch.ops.aten.view.default(constant_pad_nd, [12, 1, 512, 513])
            # 创建新的零张量 y
            y = view.new_zeros((12, 2, 256, 513))
            # 赋值操作：将 view 的部分切片复制到 y
            y[:, :-1, :, 256:] = view[:, :, :256, :257]
            return y

        # 调用通用测试函数 common，传入 fn 函数和参数元组
        self.common(
            fn,
            (
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
                torch.empty_strided((12, 1, 512, 64), (64, 196608, 768, 1)),
            ),
        )

    # 装饰测试函数，模拟 CUDA 不可用状态
    @patch("torch.cuda.is_available", lambda: False)
    def test_max_reduction_lowp_fp(self):
        # 定义函数 fn，接受参数 x
        def fn(x):
            # 执行 aten 模块的最大值降维操作，保持维度
            return torch.ops.aten.max(x, 1, keepdim=True)[0].float()

        # 针对每种低精度浮点数类型循环测试
        for dtype in _lowp_fp_dtypes:
            # 调用通用测试函数 common，传入 fn 函数和参数元组
            self.common(
                fn,
                (torch.randn(1, 32, 4, 4).to(dtype),),
            )

    # 装饰测试函数，模拟 CUDA 不可用状态
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_transpose_lowp_fp(self):
        # 针对每种低精度浮点数类型循环测试
        for dtype in _lowp_fp_dtypes:
            # 定义函数 fn，接受参数 x
            def fn(x):
                # 执行张量转置，使用通道优先的内存格式，然后转换到指定 dtype
                return x.to(memory_format=torch.channels_last).to(dtype)

            # 调用通用测试函数 common，传入 fn 函数和参数元组
            self.common(
                fn,
                (torch.randn(2, 3, 4, 4),),
            )

    # 定义测试函数：加载无穷大 bf16
    def test_load_inf_bf16(self):
        # 定义函数 fn1，接受参数 x
        def fn1(x):
            # 使用 torch.where 函数，大于 0 的位置保留，小于等于 0 的位置设为正无穷
            return torch.where(x > 0, x, math.inf)

        # 定义函数 fn2，接受参数 x
        def fn2(x):
            # 使用 torch.where 函数，大于 0 的位置保留，小于等于 0 的位置设为负无穷
            return torch.where(x > 0, x, -math.inf)

        # 遍历函数列表 [fn1, fn2]
        for fn in [fn1, fn2]:
            # 调用通用测试函数 common，传入 fn 函数和参数元组
            self.common(
                fn,
                (torch.randn(1, 3, 16, 16),),
            )

    # 装饰测试函数，模拟 CUDA 不可用状态
    @patch("torch.cuda.is_available", lambda: False)
    def test_fp32_load_with_to_lowp_fp(self):
        # 定义模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建全零张量 cache_k
                self.cache_k = torch.zeros(8, 4, 2, 2)

            # 前向传播函数，接受参数 x 和 xk
            def forward(self, x, xk):
                # 获取 x 的批大小 bsz 和序列长度 seqlen
                bsz, seqlen, _ = x.shape
                # 将 self.cache_k 转换到与 x 相同的设备上
                self.cache_k = self.cache_k.to(x)
                # 部分赋值操作：将 xk 复制到 self.cache_k 的指定位置
                self.cache_k[:bsz, 1 : 1 + seqlen] = xk
                return self.cache_k

        # 针对每种低精度浮点数类型循环测试
        for dtype in _lowp_fp_dtypes:
            # 创建参考模型 ref_model 和优化模型 opt_model
            ref_model = Model().eval()
            opt_model = torch.compile()(Model().eval())
            # 创建输入张量 x 和 xk，并转换为指定 dtype
            x = torch.randn(4, 2, 2).to(dtype)
            xk = torch.randn(4, 2, 2, 2).to(dtype)
            # 断言优化模型的输出与参考模型的输出相等
            self.assertEqual(opt_model(x, xk), ref_model(x, xk))

    # 装饰测试函数，要求支持向量化操作
    @requires_vectorization
    # 装饰测试函数，模拟 CUDA 不可用状态
    @patch("torch.cuda.is_available", lambda: False)
    def test_sigmoid_with_reduction(self):
        # 定义一个函数 fn，对输入张量进行 sigmoid 操作，并在指定维度上计算均值
        def fn(x):
            x = torch.ops.aten.sigmoid.default(x)
            return torch.ops.aten.mean.dim(x, [-1, -2], True)

        # 生成一个形状为 (1, 8, 8, 8) 的随机张量 x
        x = torch.randn((1, 8, 8, 8))
        
        # 使用指定的配置上下文进行测试
        with config.patch({"cpp.simdlen": None}):
            # 重置 torch._dynamo 和 metrics
            torch._dynamo.reset()
            metrics.reset()
            # 调用 self.common 方法，对 fn 函数进行测试
            self.common(fn, (x,))

    def test_slice_scatter_default_end_value(self):
        # 从 HF AllenaiLongformerBase 中获取的函数 fn，用于处理查询、键和窗口重叠参数
        def fn(query, key, window_overlap):
            # 获取输入张量的形状信息
            batch_size, seq_len, num_heads, head_dim = query.size()
            # 断言序列长度应为 window_overlap 的两倍的倍数，否则抛出异常
            assert (
                seq_len % (window_overlap * 2) == 0
            ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"

            # 计算序列被分成的块数
            chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
            # 创建一个全零张量，用于存储对角线注意力分数
            diagonal_chunked_attention_scores = key
            diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
                (
                    batch_size * num_heads,
                    chunks_count + 1,
                    window_overlap,
                    window_overlap * 2 + 1,
                )
            )
            # 将部分对角线注意力分数赋值给对应位置的张量
            diagonal_attention_scores[
                :, :3, :, window_overlap:
            ] = diagonal_chunked_attention_scores[
                :, :, :window_overlap, : window_overlap + 1
            ]
            # 返回处理后的对角线注意力分数张量
            return diagonal_attention_scores

        # 调用 self.common 方法，对 fn 函数进行测试
        self.common(
            fn,
            (
                torch.randn(1, 1024, 12, 64),
                torch.randn(12, 3, 512, 513),
                256,
            ),
        )

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_to_uint8_rounding_method(self):
        # 定义一个函数 fn，将输入张量 x 转换为 torch.uint8 类型
        def fn(x):
            return x.to(torch.uint8)

        # 数值测试集合
        numerical_testsuit = [4.4, 4.5, 4.6, 5.5]
        # 遍历测试集合中的每个数值
        for numerical_number in numerical_testsuit:
            # 创建一个张量 x，其中每个元素值为 numerical_number 的 17 个元素的张量
            x = torch.ones(17) * numerical_number
            # 使用指定的配置上下文进行测试
            with config.patch({"cpp.simdlen": None}):
                # 重置 torch._dynamo 和 metrics
                torch._dynamo.reset()
                metrics.reset()
                # 调用 self.common 方法，对 fn 函数进行测试
                self.common(fn, (x,))
                # 检查 metrics 中向量化内核计数是否为 1
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    # 定义内部辅助函数，用于测试分解的量化过程（decompose-dequantize-relu-quantize）
    def _test_decomposed_dequant_relu_quant_helper(self, dtype):
        # 内部函数，执行量化/反量化/ReLU/量化过程的具体操作
        def fn(
            x, scale, zero_point, use_dequant, use_quant, quant_min, quant_max, dtype
        ):
            # 如果使用反量化，则将输入张量转换为浮点数，然后执行反量化操作
            if use_dequant:
                x = (x.to(torch.float32) - zero_point) * scale

            # 执行 ReLU 激活函数
            x = torch.relu(x)

            # 如果使用量化，则计算逆标度，将张量按比例量化到整数，并在指定范围内夹紧，最后转换为指定数据类型
            if use_quant:
                inv_scale = 1.0 / scale
                x = torch.clamp(
                    torch.round(x * inv_scale) + zero_point, quant_min, quant_max
                ).to(dtype)
            return x

        # 断言数据类型为 uint8 或 int8
        assert dtype in [torch.uint8, torch.int8]
        # 根据数据类型设置量化的最小值和最大值
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        # 创建使用反量化和量化的布尔值列表
        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        # 遍历所有反量化和量化的组合
        for use_dequant, use_quant in itertools.product(
            use_dequant_list, use_quant_list
        ):
            # 创建输入张量，并在必要时转换为指定的数据类型
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            if use_dequant:
                x = x.to(dtype)
            zero_point = 100
            scale = 0.01
            # 使用特定配置执行以下代码块
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                # 调用公共方法，并传入函数及其参数
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        use_dequant,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                # 检查向量化内核数量的度量值
                check_metrics_vec_kernel_count(1)

    # 要求测试用例具备向量化支持
    @requires_vectorization
    def test_decomposed_dequant_relu_quant_uint8(self):
        # 测试 uint8 数据类型的分解反量化-ReLU-量化函数
        self._test_decomposed_dequant_relu_quant_helper(torch.uint8)

    # 要求测试用例具备向量化支持
    @requires_vectorization
    def test_decomposed_dequant_relu_quant_int8(self):
        # 测试 int8 数据类型的分解反量化-ReLU-量化函数
        self._test_decomposed_dequant_relu_quant_helper(torch.int8)
    # 定义一个内部辅助函数，用于测试量化和反量化的降低功能，针对指定的数据类型
    def _test_dequant_quant_lowering_helper(self, dtype):
        # 定义一个函数 fn，接受多个参数，根据 use_dequant 和 use_quant 条件进行处理
        def fn(
            x, scale, zero_point, use_dequant, use_quant, quant_min, quant_max, dtype
        ):
            # 如果 use_dequant 为 True，则对输入 x 进行反量化操作
            if use_dequant:
                x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    x, scale, zero_point, quant_min, quant_max, dtype
                )

            # 对输入 x 执行 relu 激活函数
            x = torch.relu(x)

            # 如果 use_quant 为 True，则对输入 x 进行量化操作
            if use_quant:
                x = torch.ops.quantized_decomposed.quantize_per_tensor(
                    x, scale, zero_point, quant_min, quant_max, dtype
                )
            # 返回处理后的 x
            return x

        # 定义用于测试的标志列表
        use_dequant_list = [False, True]
        use_quant_list = [False, True]
        use_tensor_overload_list = [False, True]

        # 断言数据类型 dtype 必须是 torch.uint8 或 torch.int8
        assert dtype in [torch.uint8, torch.int8]
        # 根据 dtype 设置量化的最小和最大值
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        # 使用 itertools.product 遍历所有组合情况
        for use_dequant, use_quant, use_tensor_overload in itertools.product(
            use_dequant_list, use_quant_list, use_tensor_overload_list
        ):
            # 生成随机张量 x，并对其进行范围限制
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            # 如果 use_dequant 为 True，则将 x 转换为指定的数据类型 dtype
            if use_dequant:
                x = x.to(dtype)
            # 设置量化的零点和比例尺
            zero_point = 100
            scale = 0.01
            # 如果 use_tensor_overload 为 True，则使用张量来定义零点和比例尺
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            # 使用 config.patch 和 torch._dynamo.reset() 进行环境配置和重置
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                # 调用 self.common 方法，传入 fn 函数和参数元组
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        use_dequant,
                        use_quant,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                )
                # 检查向量化内核数量是否为 1
                check_metrics_vec_kernel_count(1)

    # 装饰器函数，标记该测试函数要求进行向量化处理
    @requires_vectorization
    def test_dequant_quant_lowering_uint8(self):
        # 调用 _test_dequant_quant_lowering_helper 方法，测试 torch.uint8 类型的量化和反量化降低功能
        self._test_dequant_quant_lowering_helper(torch.uint8)

    # 装饰器函数，标记该测试函数要求进行向量化处理
    @requires_vectorization
    def test_dequant_quant_lowering_int8(self):
        # 调用 _test_dequant_quant_lowering_helper 方法，测试 torch.int8 类型的量化和反量化降低功能
        self._test_dequant_quant_lowering_helper(torch.int8)
    # 定义测试函数，用于测试下面两种数据类型的情况：torch.uint8 和 torch.int8
    def _test_dequant_maxpool2d_lowering_helper(self, dtype):
        # 定义内部函数 fn，接收输入张量 x 和量化相关参数，执行反量化操作后，再执行最大池化操作
        def fn(x, scale, zero_point, quant_min, quant_max, dtype):
            # 使用自定义的 quantized_decomposed 操作进行张量 x 的反量化
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, quant_min, quant_max, dtype
            )
            # 调用 aten.max_pool2d_with_indices.default 函数进行最大池化操作，默认参数设置
            max_pool2d_with_indices_default = (
                torch.ops.aten.max_pool2d_with_indices.default(
                    x, [2, 2], [2, 2], [1, 1]
                )[0]
            )
            # 返回最大池化后的结果
            return max_pool2d_with_indices_default
    
        # 断言 dtype 参数只能是 torch.uint8 或 torch.int8
        assert dtype in [torch.uint8, torch.int8]
        # 根据 dtype 确定量化的最小值和最大值
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127
    
        # 定义是否使用张量重载的列表，分别进行测试
        use_tensor_overload_list = [False, True]
        for use_tensor_overload in use_tensor_overload_list:
            # 生成一个形状为 (3, 16, 8, 8) 的随机张量 x，类型为 torch.float32
            x = (
                torch.clamp(
                    torch.randn((3, 16, 8, 8), dtype=torch.float32) * 100,
                    quant_min,
                    quant_max,
                )
                .to(dtype)  # 转换为指定的数据类型
                .contiguous(memory_format=torch.channels_last)  # 转换为内存格式为通道优先
            )
            zero_point = 100  # 设置 zero_point 为常数 100
            scale = 0.01  # 设置 scale 为常数 0.01
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)  # 使用张量重载，转换为 torch.int64 类型
                scale = torch.tensor(scale)  # 使用张量重载，转换为张量类型
            with config.patch({"cpp.simdlen": None}):  # 使用 config.patch 禁用 SIMD
                torch._dynamo.reset()  # 重置 torch._dynamo 模块
                metrics.reset()  # 重置 metrics 模块
                # 调用 self.common 方法执行测试函数 fn，传入相关参数
                self.common(fn, (x, scale, zero_point, quant_min, quant_max, dtype))
                # 检查 metrics 模块的向量内核计数是否为 1
                check_metrics_vec_kernel_count(1)
    
    # 标记需要向量化的测试函数，测试 uint8 类型下的反量化最大池化操作
    @requires_vectorization
    def test_dequant_maxpool2d_lowering_uint8(self):
        self._test_dequant_maxpool2d_lowering_helper(torch.uint8)
    
    # 标记需要向量化的测试函数，测试 int8 类型下的反量化最大池化操作
    @requires_vectorization
    def test_dequant_maxpool2d_lowering_int8(self):
        self._test_dequant_maxpool2d_lowering_helper(torch.int8)
    
    # 标记需要向量化的测试函数，测试 uint8 类型下的加载分解反量化加 relu 量化操作
    @requires_vectorization
    def test_tile2d_load_decomposed_dequant_add_relu_quant_uint8(self):
        self._test_tile2d_load_decomposed_dequant_add_relu_quant_helper(torch.uint8)
    
    # 标记需要向量化的测试函数，测试 int8 类型下的加载分解反量化加 relu 量化操作
    @requires_vectorization
    def test_tile2d_load_decomposed_dequant_add_relu_quant_int8(self):
        self._test_tile2d_load_decomposed_dequant_add_relu_quant_helper(torch.int8)
    def _test_per_tensor_fake_quant_helper(self, dtype):
        # 定义内部函数 fn，用于量化和反量化操作
        def fn(input, scales, zero_points, quant_min, quant_max, dtype):
            # 对输入张量进行量化操作
            input = torch.ops.quantized_decomposed.quantize_per_tensor(
                input, scales, zero_points, quant_min, quant_max, dtype
            )
            # 对量化后的张量进行反量化操作
            input = torch.ops.quantized_decomposed.dequantize_per_tensor(
                input, scales, zero_points, quant_min, quant_max, dtype
            )
            return input

        # 定义使用张量重载的标志列表
        use_tensor_overload_list = [False, True]
        # 遍历每个标志
        for use_tensor_overload in use_tensor_overload_list:
            # 断言 dtype 必须是 torch.uint8 或 torch.int8
            assert dtype in [torch.uint8, torch.int8]
            # 根据 dtype 设置量化范围
            quant_min = 0 if dtype == torch.uint8 else -128
            quant_max = 255 if dtype == torch.uint8 else 127
            # 生成一个符合量化范围的随机张量，并进行数值截断
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            )
            # 设置零点和缩放因子
            zero_point = 100
            scale = 0.01
            # 如果使用张量重载，将零点和缩放因子转换为张量类型
            if use_tensor_overload:
                zero_point = torch.tensor(zero_point, dtype=torch.int64)
                scale = torch.tensor(scale)
            # 使用特定配置运行测试代码块
            with config.patch({"cpp.simdlen": None}):
                torch._dynamo.reset()
                metrics.reset()
                # 调用通用测试函数 common，检查生成的 CPP 向量化内核数量为 1
                self.common(fn, (x, scale, zero_point, quant_min, quant_max, dtype))
                assert metrics.generated_cpp_vec_kernel_count == 1

    @requires_vectorization
    def test_per_tensor_fake_quant_uint8(self):
        # 调用 _test_per_tensor_fake_quant_helper 函数，传入 torch.uint8 类型参数
        self._test_per_tensor_fake_quant_helper(torch.uint8)

    @requires_vectorization
    def test_per_tensor_fake_quant_int8(self):
        # 调用 _test_per_tensor_fake_quant_helper 函数，传入 torch.int8 类型参数
        self._test_per_tensor_fake_quant_helper(torch.int8)

    def _test_per_channel_fake_quant_helper(self, dtype, input_dtype=torch.float32):
        # 定义内部函数 fn，用于通道量化和反量化操作
        def fn(input, scales, zero_points, axis, quant_min, quant_max, dtype):
            # 对输入张量进行通道量化操作
            input = torch.ops.quantized_decomposed.quantize_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max, dtype
            )
            # 对通道量化后的张量进行反量化操作
            input = torch.ops.quantized_decomposed.dequantize_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max, dtype
            )
            return input

        # 断言 dtype 必须是 torch.uint8 或 torch.int8
        assert dtype in [torch.uint8, torch.int8]
        # 根据 dtype 设置量化范围
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127
        # 生成一个符合量化范围的随机张量，并进行数值截断
        x = torch.clamp(
            torch.randn((1, 3, 224, 224), dtype=torch.float32) * 100,
            quant_min,
            quant_max,
        )
        # 如果输入数据类型不是 torch.float32，则将输入张量转换为指定类型
        if input_dtype != torch.float32:
            x = x.to(dtype=input_dtype)
        # 设置通道的缩放因子和零点
        scales = torch.ones((3,))
        zero_points = torch.zeros((3,))
        axis = 1
        # 使用特定配置运行测试代码块
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            # 调用通用测试函数 common，检查生成的 CPP 向量化内核数量为 1
            self.common(fn, (x, scales, zero_points, axis, quant_min, quant_max, dtype))
            check_metrics_vec_kernel_count(1)

    @requires_vectorization
    # 使用 torch.uint8 类型调用辅助函数 _test_per_channel_fake_quant_helper 进行测试
    def test_per_channel_fake_quant_uint8(self):
        self._test_per_channel_fake_quant_helper(torch.uint8)
    
    # 带有 requires_vectorization 装饰器的测试函数，用于测试 per-channel fake quantization 模块的 uint8 版本
    def test_per_channel_fake_quant_module_uint8(self):
        # 定义一个继承自 torch.nn.Module 的模块类 Mod
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块的属性
                self.scales = torch.ones((3,)).to(torch.float64)  # 尺度参数，全为1的张量，类型为 float64
                self.zero_points = torch.zeros((3,)).to(torch.int64)  # 零点参数，全为0的张量，类型为 int64
                self.axis = 1  # 量化轴的索引，设置为1
                self.quant_min = 0  # 量化的最小值
                self.quant_max = 255  # 量化的最大值
                self.dtype = torch.uint8  # 数据类型为 uint8
    
            def forward(self, input):
                # 使用 per-channel 的量化操作对输入进行量化
                input = torch.ops.quantized_decomposed.quantize_per_channel(
                    input,
                    self.scales,
                    self.zero_points,
                    self.axis,
                    self.quant_min,
                    self.quant_max,
                    self.dtype,
                )
                # 对量化后的结果进行反量化
                input = torch.ops.quantized_decomposed.dequantize_per_channel(
                    input,
                    self.scales,
                    self.zero_points,
                    self.axis,
                    self.quant_min,
                    self.quant_max,
                    self.dtype,
                )
                return input
    
        # 创建 Mod 类的实例并设置为评估模式
        m = Mod().eval()
        # 生成随机张量 x，形状为 (1, 3, 224, 224)，类型为 torch.float32，并将值限制在 [0, 255] 范围内
        x = torch.clamp(
            torch.randn((1, 3, 224, 224), dtype=torch.float32) * 100,
            0,
            255,
        )
        # 在特定的配置环境中执行以下测试
        with config.patch({"cpp.simdlen": None}):
            # 重置 torch._dynamo 和 metrics
            torch._dynamo.reset()
            metrics.reset()
            # 执行通用的测试函数 common，传入模块 m 和输入 x
            self.common(m, (x,))
            # 断言生成的 CPP 向量化内核数量为 1
            assert metrics.generated_cpp_vec_kernel_count == 1
    
    # 使用 torch.int8 类型调用辅助函数 _test_per_channel_fake_quant_helper 进行测试
    def test_per_channel_fake_quant_int8(self):
        self._test_per_channel_fake_quant_helper(torch.int8)
    
    # 带有 requires_vectorization 装饰器的测试函数，用于测试 per-channel fake quantization 模块的 uint8 版本，输入类型为 torch.bfloat16
    def test_per_channel_fake_quant_uint8_bf16_input(self):
        self._test_per_channel_fake_quant_helper(
            torch.uint8, input_dtype=torch.bfloat16
        )
    
    # 带有 requires_vectorization 装饰器的测试函数，用于测试 per-channel fake quantization 模块的 int8 版本，输入类型为 torch.bfloat16
    def test_per_channel_fake_quant_int8_bf16_input(self):
        self._test_per_channel_fake_quant_helper(torch.int8, input_dtype=torch.bfloat16)
    # 定义一个测试函数，用于测试非连续加载缓冲区的量化辅助函数，接受一个数据类型参数dtype
    def _test_non_contiguous_load_buf_quant_helper(self, dtype):
        # 内部定义的函数fn，接受多个参数，执行一系列操作后返回处理后的张量x
        def fn(
            x1,
            x2,
            groups,
            quant_min,
            quant_max,
            dtype,
        ):
            # 将输入张量x1和x2在第1维上拼接，形成新的张量x
            x = torch.cat((x1, x2), dim=1)
            # 获取张量x的形状信息
            batchsize, num_channels, height, width = x.size()
            # 计算每组的通道数
            channels_per_group = num_channels // groups
            # 对张量x执行量化反操作，将其转换为浮点张量
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            # 重新调整张量x的形状，以groups为组数进行重新分组
            x = x.view(batchsize, groups, channels_per_group, height, width)
            # 对重新分组后的张量x执行量化操作，转换为指定dtype类型的张量
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            # 再次对张量x执行量化反操作，转换为浮点张量
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, 1.0, 0, quant_min, quant_max, dtype
            )
            # 对张量x进行转置操作，交换第1和第2维，并保证张量是连续的
            x = torch.transpose(x, 1, 2).contiguous()
            # 最后将张量x的形状恢复为原始形状
            x = x.view(batchsize, num_channels, height, width)
            # 返回处理后的张量x
            return x

        # 断言dtype参数必须是torch.uint8或torch.int8之一
        assert dtype in [torch.uint8, torch.int8]
        # 根据dtype确定量化的最小值和最大值
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        # 创建两个形状为(1, 116, 28, 28)的随机整数张量x和x2，指定dtype，并保证张量是连续的，内存格式为通道最后
        x = torch.randint(0, 8, (1, 116, 28, 28), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        x2 = torch.randint(0, 8, (1, 116, 28, 28), dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )

        # 使用特定的配置上下文，执行测试前的初始化操作
        with config.patch({"cpp.simdlen": None}):
            # 重置Dynamo状态
            torch._dynamo.reset()
            # 重置度量指标
            metrics.reset()
            # 调用公共测试函数common，传入fn函数及其参数，执行测试
            self.common(
                fn,
                (
                    x,
                    x2,
                    2,
                    quant_min,
                    quant_max,
                    dtype,
                ),
            )
            # 检查度量指标中向量化内核计数是否为2
            check_metrics_vec_kernel_count(2)

    # 声明一个需要向量化支持的测试函数，测试uint8类型的非连续加载缓冲区的量化操作
    @requires_vectorization
    def test_non_contiguous_load_buf_quant_uint8(self):
        self._test_non_contiguous_load_buf_quant_helper(torch.uint8)

    # 声明一个需要向量化支持的测试函数，测试int8类型的非连续加载缓冲区的量化操作
    @requires_vectorization
    def test_non_contiguous_load_buf_quant_int8(self):
        self._test_non_contiguous_load_buf_quant_helper(torch.int8)
    # 定义一个辅助函数，用于测试二维瓦片存储通道重排的量化输出（支持不同数据类型）
    def _test_tile2d_store_channel_shuffle_cl_quant_output_helper(self, dtype):
        # 定义通道重排函数
        def channel_shuffle(
            x, groups, output_scale, output_zero_point, quant_min, quant_max, dtype
        ):
            # 获取输入张量的维度信息
            batchsize, num_channels, height, width = x.size()
            # 计算每组的通道数
            channels_per_group = num_channels // groups
            # 将输入张量重新排列，以便进行通道重排操作
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            # 将张量展平，以便进行量化操作
            x = x.view(batchsize, -1, height, width)
            # 调用量化函数，对张量进行量化操作
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, output_scale, output_zero_point, quant_min, quant_max, dtype
            )
            # 返回张量，采用通道优先的内存格式（channels_last）
            return x.contiguous(memory_format=torch.channels_last)

        # 断言数据类型为 uint8 或 int8
        assert dtype in [torch.uint8, torch.int8]
        # 根据数据类型设置量化的最小值和最大值
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        # 在特定配置环境下执行测试
        with config.patch({"cpp.simdlen": None}):
            # 重置 Torch 动态图机制
            torch._dynamo.reset()
            # 重置性能指标
            metrics.reset()
            # 创建随机张量作为输入
            x = torch.randn(64, 58, 28, 28)
            # 设置量化输出的零点和比例
            output_zero_point = 3
            output_scale = 0.03
            # 调用通用测试函数，测试通道重排函数的输出
            self.common(
                channel_shuffle,
                (x, 2, output_scale, output_zero_point, quant_min, quant_max, dtype),
            )
            # 检查向量化内核计数的性能指标
            check_metrics_vec_kernel_count(2)

    # 使用 uint8 数据类型测试通道重排函数的量化输出
    @requires_vectorization
    def test_tile2d_store_channel_shuffle_cl_quant_output_uint8(self):
        self._test_tile2d_store_channel_shuffle_cl_quant_output_helper(torch.uint8)

    # 使用 int8 数据类型测试通道重排函数的量化输出
    @requires_vectorization
    def test_tile2d_store_channel_shuffle_cl_quant_output_int8(self):
        self._test_tile2d_store_channel_shuffle_cl_quant_output_helper(torch.int8)
    # 定义内部辅助函数，用于测试量化和反量化操作与 ReLU 函数的结合
    def _test_dequant_relu_quant_dequant_relu_quant_lowering_helper(self, dtype):
        # 定义函数 fn，接受多个量化相关参数，并依次执行反量化、ReLU、量化操作
        def fn(
            x,
            scale,
            zero_point,
            scale2,
            zero_point2,
            scale3,
            zero_point3,
            quant_min,
            quant_max,
            dtype,
        ):
            # 对输入 x 进行反量化操作，使用给定的量化参数
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale, zero_point, quant_min, quant_max, dtype
            )
            # 对 x 执行 ReLU 激活函数
            x = torch.relu(x)
            # 将经过 ReLU 的 x 进行量化操作，使用新的量化参数
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale2, zero_point2, quant_min, quant_max, dtype
            )
            # 再次对量化后的 x 进行反量化操作，保持流程完整性
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x, scale2, zero_point2, quant_min, quant_max, dtype
            )
            # 最后对经过二次反量化的 x 执行 ReLU 激活函数
            x = torch.relu(x)
            # 最后一步，将经过二次 ReLU 的 x 进行最后一次量化操作，使用最后一组量化参数
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x, scale3, zero_point3, quant_min, quant_max, dtype
            )
            return x

        # 断言 dtype 必须是 torch.uint8 或 torch.int8 中的一种
        assert dtype in [torch.uint8, torch.int8]
        # 根据 dtype 确定量化的最小和最大值
        quant_min = 0 if dtype == torch.uint8 else -128
        quant_max = 255 if dtype == torch.uint8 else 127

        # 遍历使用张量重载与否的两种情况
        for use_tensor_overload in [True, False]:
            # 创建一个形状为 (1, 7, 7, 9) 的随机张量 x，数据类型为 dtype，并进行范围限制
            x = torch.clamp(
                torch.randn((1, 7, 7, 9), dtype=torch.float32) * 100,
                quant_min,
                quant_max,
            ).to(dtype)
            # 定义多个零点和缩放因子的列表
            zero_point_list = [100, 101, 102]
            scale_list = [0.01, 0.02, 0.03]
            # 如果使用张量重载，将列表中的每个元素转换为对应的张量类型
            if use_tensor_overload:
                for i in range(len(zero_point_list)):
                    zero_point_list[i] = torch.tensor(
                        zero_point_list[i], dtype=torch.int64
                    )
                    scale_list[i] = torch.tensor(scale_list[i])
            # 将列表中的零点值和缩放因子赋值给对应的变量
            zero_point, zero_point2, zero_point3 = zero_point_list
            scale, scale2, scale3 = scale_list
            # 使用特定配置进行上下文管理
            with config.patch({"cpp.simdlen": None}):
                # 重置内部状态
                torch._dynamo.reset()
                # 重置度量指标
                metrics.reset()
                # 调用公共函数，传入 fn 函数及其参数，进行测试，并设定相对和绝对误差容差
                self.common(
                    fn,
                    (
                        x,
                        scale,
                        zero_point,
                        scale2,
                        zero_point2,
                        scale3,
                        zero_point3,
                        quant_min,
                        quant_max,
                        dtype,
                    ),
                    rtol=1e-2,
                    atol=1e-2,
                )
                # 检查向量化内核计数是否为 1
                check_metrics_vec_kernel_count(1)

    # 要求进行向量化的测试函数装饰器
    @requires_vectorization
    # 测试 uint8 类型的量化与反量化操作和 ReLU 结合的情况
    def test_dequant_relu_quant_dequant_relu_quant_lowering_uint8(self):
        self._test_dequant_relu_quant_dequant_relu_quant_lowering_helper(torch.uint8)

    # 要求进行向量化的测试函数装饰器
    @requires_vectorization
    # 测试 int8 类型的量化与反量化操作和 ReLU 结合的情况
    def test_dequant_relu_quant_dequant_relu_quant_lowering_int8(self):
        self._test_dequant_relu_quant_dequant_relu_quant_lowering_helper(torch.int8)
    # 定义一个测试函数，用于测试 inplace_add_alpha 方法
    def test_inplace_add_alpha(self):
        # 定义一个内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 使用 torch.aten.add_ 方法，在 x 上执行 inplace 加法，带有指定的 alpha 参数
            aten.add_.Tensor(x, y, alpha=0.55)
            # 返回一个元组，包含 x
            return (x,)

        # 创建三个长度为 10 的零张量 x1, x2, x3
        x1 = torch.zeros(10)
        x2 = torch.zeros(10)
        x3 = torch.zeros(10)
        # 创建一个长度为 10 的随机张量 y
        y = torch.randn(10)
        # 对 fn 函数使用 make_fx 包装，然后调用该函数
        fn_fx = make_fx(fn)(x1, y)
        # 编译 fn_fx 内部的计算图，并使用 x1 和 y 作为输入
        fn_compiled = compile_fx_inner(fn_fx, [x1, y])
        # 对 x2 和 y 执行 fn 函数
        fn(x2, y)
        # 对 fn_compiled 函数执行，并传入 x3 和 y 作为参数
        fn_compiled([x3, y])
        # 断言 x2 和 x3 应该是相同的
        assert same(x2, x3)

    # 定义一个测试函数，用于测试整数除法操作
    def test_int_div(self):
        # 定义一个内部函数 fn，接受两个参数 x 和 y
        def fn(x, y):
            # 计算 x 张量的第二维度的大小
            s3 = x.size(1)
            # 创建一个新的零张量 a，其大小为 (1 + s3) // 2
            a = torch.zeros((1 + s3) // 2)
            # 将 y 加到 a 上
            a += y
            # 返回张量 a 和 s3 的值
            return a, s3

        # 创建一个形状为 (1, 8) 的随机整数张量 p0
        p0 = torch.randint(5, (1, 8))
        # 创建一个标量随机张量 p1
        p1 = torch.randn(1)
        # 使用 self.common 函数测试 fn 函数，传入参数 p0 和 p1
        self.common(fn, (p0, p1))

    # 定义一个测试函数，用于测试 no-op 操作的挤压（squeeze）
    def test_no_op_squeeze(self):
        # 定义一个装饰了 torch._dynamo.optimize 的前向函数 forward，接受一个参数 arg0_1
        @torch._dynamo.optimize("inductor")
        def forward(arg0_1):
            # 调用 torch.aten.squeeze.dim 函数，对 arg0_1 执行挤压操作，维度为 1
            return torch.ops.aten.squeeze.dim(arg0_1, 1)

        # 创建一个形状为 (10, 20) 的随机张量 x
        x = torch.randn((10, 20))
        # 使用 self.common 函数测试 forward 函数，传入参数 x
        self.common(forward, (x,))

    # 定义一个测试函数，用于测试并行计算的线程数
    def test_parallel_num_threads(self):
        # 定义一个函数 fn，接受两个参数 x1 和 x2
        def fn(x1, x2):
            # 返回 x1 和 x2 的元素级加法结果
            return x1 + x2

        # 创建两个形状为 (10, 20) 的随机张量 x1 和 x2
        x1 = torch.randn((10, 20))
        x2 = torch.randn((10, 20))
        # 设置线程数为 1，并断言 x1 + x2 与 fn(x1, x2) 的结果相同
        with set_num_threads(1):
            assert same(x1 + x2, fn(x1, x2))
        # 设置线程数为 4，并断言 x1 + x2 与 fn(x1, x2) 的结果相同
        with set_num_threads(4):
            assert same(x1 + x2, fn(x1, x2))

    # 使用 @patch 装饰器模拟 CUDA 不可用环境，测试只在 CPU 上运行的计时函数
    @patch("torch.cuda.is_available", lambda: False)
    def test_timed_cpu_only(self):
        # 使用 timed 函数计时并创建一个形状为 10 的随机张量
        timed(lambda: torch.randn(10), ())

    # 定义一个测试函数，用于测试复杂的内存重叠情况
    def test_complex_memory_overlap(self):
        # 创建一个形状为 (64, 32) 的零张量 dense
        dense = torch.zeros(64, 32)
        # 断言 dense 不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(dense))
        # 断言 dense 的转置也不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(dense.t()))

        # 将 dense 按第二维度分割为 4 份，创建 strided 张量组
        strided = dense.split(4, dim=1)
        # 断言 strided[0] 不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(strided[0]))
        # 断言 strided[0] 的转置也不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(strided[0].t()))

        # 在 dense 上增加一个维度，创建 unsqueezed 张量
        unsqueezed = dense.unsqueeze(1)
        # 断言 unsqueezed 不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(unsqueezed))
        # 断言 unsqueezed 的维度置换也不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(unsqueezed.permute(1, 2, 0)))

        # 使用 index_select 从 dense 中选择特定索引的张量，创建 gathered 张量
        gathered = dense.index_select(0, torch.IntTensor([1, 0, 1]))
        # 断言 gathered 不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(gathered))
        # 断言 gathered 的转置也不具有复杂的内存重叠
        self.assertFalse(complex_memory_overlap(gathered.t()))

    # 使用 @requires_vectorization 装饰器，定义一个测试函数，用于测试动态形状的向量化操作
    @requires_vectorization
    def test_vec_dynamic_shapes(self):
        # 定义一个函数 fn，接受一个参数 x
        def fn(x):
            # 对 x 执行 softmax 操作，维度为 -1
            return torch.softmax(x, -1)

        # 创建一个形状为 (2, 10) 的随机张量 value
        value = torch.randn((2, 10))
        # 使用 config.patch 重置配置，然后测试 fn 函数，传入参数 value
        with config.patch({"cpp.simdlen": None}):
            torch._dynamo.reset()
            metrics.reset()
            self.common(fn, (value,))

    # 使用 @unittest.skipIf 装饰器，如果不在 x86_64 架构或不支持向量化指令集，则跳过测试
    @unittest.skipIf(
        platform.machine() != "x86_64" or not cpu_vec_isa.valid_vec_isa_list(),
        "Does not support vectorization or not x86_64 machine",
    )
    # 使用 @patch 装饰器模拟 CUDA 不可用环境
    @patch("torch.cuda.is_available", lambda: False)
    # 定义一个测试方法，用于测试自动 SIMD 支持情况
    def test_auto_simd(self):
        # 从 CPU 向量 ISA 支持列表中获取 AMX 向量 ISA
        vec_amx = cpu_vec_isa.supported_vec_isa_list[0]
        # 从 CPU 向量 ISA 支持列表中获取 AVX-512 向量 ISA
        vec_avx512 = cpu_vec_isa.supported_vec_isa_list[1]
        # 从 CPU 向量 ISA 支持列表中获取 AVX2 向量 ISA
        vec_avx2 = cpu_vec_isa.supported_vec_isa_list[2]

        # 断言 AMX 向量 ISA 的位宽为 512
        self.assertTrue(vec_amx.bit_width() == 512)
        # 断言 AMX 向量 ISA 的元素个数为 16
        self.assertTrue(vec_amx.nelements() == 16)
        # 断言 AMX 向量 ISA 在使用 bfloat16 数据类型时的元素个数为 32
        self.assertTrue(vec_amx.nelements(torch.bfloat16) == 32)
        # 断言 AVX-512 向量 ISA 的位宽为 512
        self.assertTrue(vec_avx512.bit_width() == 512)
        # 断言 AVX2 向量 ISA 的位宽为 256
        self.assertTrue(vec_avx2.bit_width() == 256)
        # 断言 AVX-512 向量 ISA 的元素个数为 16
        self.assertTrue(vec_avx512.nelements() == 16)
        # 断言 AVX2 向量 ISA 的元素个数为 8
        self.assertTrue(vec_avx2.nelements() == 8)
        # 断言 AVX-512 向量 ISA 在使用 bfloat16 数据类型时的元素个数为 32
        self.assertTrue(vec_avx512.nelements(torch.bfloat16) == 32)
        # 断言 AVX2 向量 ISA 在使用 bfloat16 数据类型时的元素个数为 16
        self.assertTrue(vec_avx2.nelements(torch.bfloat16) == 16)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 None
        with config.patch({"cpp.simdlen": None}):
            # 选择最适合的向量 ISA
            isa = cpu_vec_isa.pick_vec_isa()
            # 如果 AMX 向量 ISA 在有效向量 ISA 列表中，则断言选择的 ISA 是 AMX
            if vec_amx in cpu_vec_isa.valid_vec_isa_list():
                self.assertTrue(isa == vec_amx)
            # 否则，如果 AVX-512 向量 ISA 在有效向量 ISA 列表中，则断言选择的 ISA 是 AVX-512
            elif vec_avx512 in cpu_vec_isa.valid_vec_isa_list():
                self.assertTrue(isa == vec_avx512)
            # 否则，断言选择的 ISA 是 AVX2
            else:
                self.assertTrue(isa == vec_avx2)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 0
        with config.patch({"cpp.simdlen": 0}):
            # 选择最适合的向量 ISA
            isa = cpu_vec_isa.pick_vec_isa()
            # 断言没有选择任何向量 ISA
            self.assertFalse(isa)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 1
        with config.patch({"cpp.simdlen": 1}):
            # 选择最适合的向量 ISA
            isa = cpu_vec_isa.pick_vec_isa()
            # 断言没有选择任何向量 ISA
            self.assertFalse(isa)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 257
        with config.patch({"cpp.simdlen": 257}):
            # 选择最适合的向量 ISA
            isa = cpu_vec_isa.pick_vec_isa()
            # 断言没有选择任何向量 ISA
            self.assertFalse(isa)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 513
        with config.patch({"cpp.simdlen": 513}):
            # 获取有效的向量 ISA 列表
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            # 如果 AVX-512 在 ISA 列表中，则断言没有选择任何向量 ISA
            if vec_avx512 in isa_list:
                self.assertFalse(isa)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 512
        with config.patch({"cpp.simdlen": 512}):
            # 获取有效的向量 ISA 列表
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            # 选择最适合的向量 ISA
            isa = cpu_vec_isa.pick_vec_isa()
            # 如果 AMX 在 ISA 列表中，则断言选择的 ISA 是 AMX
            if vec_amx in isa_list:
                self.assertTrue(isa == vec_amx)
            # 否则，如果 AVX-512 在 ISA 列表中，则断言选择的 ISA 是 AVX-512
            elif vec_avx512 in isa_list:
                self.assertTrue(isa == vec_avx512)

        # 使用配置文件对 "cpp.simdlen" 进行临时修补，设为 256
        with config.patch({"cpp.simdlen": 256}):
            # 获取有效的向量 ISA 列表
            isa_list = cpu_vec_isa.valid_vec_isa_list()
            # 如果 AVX2 在 ISA 列表中，则选择 AVX2 作为向量 ISA
            if vec_avx2 in isa_list:
                isa = cpu_vec_isa.pick_vec_isa()
                # 断言选择的 ISA 是 AVX2
                self.assertTrue(isa == vec_avx2)

    # 要求向量化装饰器，用于标记需要进行向量化的代码
    @requires_vectorization
    # 修补 "torch.cuda.is_available" 方法，使其始终返回 False
    @patch("torch.cuda.is_available", lambda: False)
    def test_masked_fill_softmax(self):
        # 定义一个函数 fn，接受 value 和 mask 两个参数
        def fn(value, mask):
            # 将 mask 转换为布尔类型的张量
            mask = mask.to(torch.bool)
            # 使用 -33.0 对 value 中被 mask 标记的元素进行填充
            x = torch.masked_fill(value, mask, -33.0)
            # 对填充后的张量 x 进行 softmax 操作，沿着最后一个维度进行计算
            return torch.softmax(x, -1)

        # 针对不同的数据类型 dtype 进行测试
        for dtype in vec_dtypes:
            # 生成指定 dtype 的随机张量 value，形状为 (2, 17)
            value = torch.randn((2, 17), dtype=dtype)
            # 生成随机的 mask 张量，形状也为 (2, 17)，数据类型为 torch.uint8
            mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8)
            # 使用 config.patch 对配置进行临时修改
            with config.patch({"cpp.simdlen": None}):
                # 针对 cpp_wrapper_flag 进行两次测试，一次为 True，一次为 False
                for cpp_wrapper_flag in [True, False]:
                    # 再次使用 config.patch 进行配置修改
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        # 重置 torch._dynamo 状态
                        torch._dynamo.reset()
                        # 重置 metrics 指标
                        metrics.reset()
                        # 调用 self.common 方法进行函数 fn 的测试
                        self.common(fn, (value, mask))
                        # 断言生成的 C++ 向量化内核数量至少为 1
                        assert metrics.generated_cpp_vec_kernel_count >= 1

    def test_channels_last_view_as_complex(self):
        # https://github.com/pytorch/pytorch/issues/122448#issuecomment-2046169554

        # 定义一个函数 reduce_example，接受两个参数 x 和 y，返回一个复数张量
        def reduce_example(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Applies the rotary embedding to the query and key tensors."""
            # 将 x 和 y 按照最后一个维度组合成复数张量
            x_out = torch.view_as_complex(torch.stack([x.float(), y.float()], dim=-1))
            return x_out

        # 准备输入参数 args，包含两个随机张量，形状为 (1, 1, 1, 128)
        args = [torch.randn(1, 1, 1, 128), torch.randn(1, 1, 1, 128)]
        # 期望的输出为 reduce_example 函数对 args 的结果
        expected = reduce_example(*args)
        # 使用 torch.compile 对 reduce_example 函数进行编译，fullgraph=True 表示编译整个图
        actual = torch.compile(reduce_example, fullgraph=True)(*args)
        # 断言编译后的结果与期望的结果一致
        self.assertEqual(expected, actual)

    def test_load_same_bool_tensor_twice(self):
        # 使用 torch._dynamo.optimize("inductor") 优化函数 fn
        @torch._dynamo.optimize("inductor")
        def fn(a, b):
            # 对张量 a 使用张量 b 的布尔掩码进行填充，使用 -33.0 替换
            x = torch.masked_fill(a, b, -33.0)
            # 同样的操作再次应用于张量 a 和 b
            y = torch.masked_fill(a, b, -33.0)
            return x, y

        # 生成一个随机张量 value，形状为 (2, 17)
        value = torch.randn((2, 17))
        # 生成随机的 mask 张量，形状为 (2, 17)，数据类型为 torch.uint8，并将其转换为布尔类型
        mask = torch.randint(0, 1, size=(2, 17), dtype=torch.uint8).to(torch.bool)
        # 调用优化后的函数 fn，并传入 value 和 mask 作为参数
        fn(value, mask)
    # 定义一个名为 test_cpu_vec_cosim 的测试方法
    def test_cpu_vec_cosim(self):
        # 初始化空列表，用于存储 CppVecOverrides 类中的静态方法名
        cpp_vec_op_list = []
        # 初始化空列表，用于存储 CppOverrides 类中的静态方法名
        cpp_op_list = []

        # 遍历 CppVecOverrides 类的所有属性名和属性值
        for k, v in CppVecOverrides.__dict__.items():
            # 检查属性值是否为静态方法
            if isinstance(v, staticmethod):
                # 将静态方法名添加到 cpp_vec_op_list 列表中
                cpp_vec_op_list.append(k)
        
        # 遍历 CppOverrides 类的所有属性名和属性值
        for k, v in CppOverrides.__dict__.items():
            # 检查属性值是否为静态方法
            if isinstance(v, staticmethod):
                # 将静态方法名添加到 cpp_op_list 列表中
                cpp_op_list.append(k)

        # 定义包含了一组字符串的列表 diff
        diff = [
            "airy_ai", "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1",
            "modified_bessel_i0", "modified_bessel_i1", "modified_bessel_k0",
            "modified_bessel_k1", "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1", "spherical_bessel_j0", "i1", "i1e",
            "ndtr", "ndtri", "log_ndtr", "erfcx", "gammainc", "gammaincc",
            "igamma", "igammac", "polygamma", "zeta",
            "shifted_chebyshev_polynomial_u", "chebyshev_polynomial_u",
            "chebyshev_polynomial_t", "shifted_chebyshev_polynomial_w",
            "chebyshev_polynomial_w", "shifted_chebyshev_polynomial_t",
            "chebyshev_polynomial_v", "shifted_chebyshev_polynomial_v",
            "hermite_polynomial_he", "laguerre_polynomial_l",
            "hermite_polynomial_h", "legendre_polynomial_p", "constant",
            "index_expr", "signbit", "isinf", "frexp", "mod", "masked",
            "randn", "isnan", "rand", "randint64", "logical_and",
            "logical_not", "logical_or", "logical_xor", "bitwise_and",
            "bitwise_left_shift", "bitwise_not", "bitwise_right_shift",
            "bitwise_or", "bitwise_xor", "to_dtype_bitcast",
        ]

        # 将 cpp_vec_op_list 和 diff 列表的并集存入 union 集合中
        union = {*cpp_vec_op_list, *diff}

        # 使用 assertTrue 断言，验证 cpp_op_list 中的所有元素是否都属于 union
        self.assertTrue(
            set(cpp_op_list).issubset(union), f"unexpected: {set(cpp_op_list) - union}"
        )
    # 定义一个测试函数 test_atomic_add_lowp_fp
    def test_atomic_add_lowp_fp(self):
        # 定义内部函数 fn，接受一个参数 test_args
        def fn(test_args):
            # 使用 torch.gather 函数根据 test_args 进行操作，返回结果 res
            res = torch.gather(**test_args)
            return res

        # 遍历 _lowp_fp_dtypes 列表中的每种数据类型 dtype
        for dtype in _lowp_fp_dtypes:
            # 创建一个用于参考的输入张量 input_tensor_for_ref，包含值为 [[3.0, -5.0]]
            # 使用给定的数据类型 dtype，并允许梯度计算
            input_tensor_for_ref = torch.tensor(
                [[3.0, -5.0]], dtype=dtype, requires_grad=True
            )
            # 创建一个用于优化的输入张量 input_tensor_for_opt，与 input_tensor_for_ref 相同
            input_tensor_for_opt = torch.tensor(
                [[3.0, -5.0]], dtype=dtype, requires_grad=True
            )

            # 准备用于参考执行的测试参数 test_args_for_ref 字典
            test_args_for_ref = {
                "input": input_tensor_for_ref,
                "dim": 1,
                "index": torch.tensor([[1]]),
            }
            # 准备用于优化执行的测试参数 test_args_for_opt 字典
            test_args_for_opt = {
                "input": input_tensor_for_opt,
                "dim": 1,
                "index": torch.tensor([[1]]),
            }

            # 编译内部函数 fn 为 opt_fn
            opt_fn = torch.compile(fn)

            # 分别计算参考前向传播 ref_fwd 和优化前向传播 res_fwd 的结果
            ref_fwd = fn(test_args_for_ref)
            res_fwd = opt_fn(test_args_for_opt)

            # 断言优化后的前向传播结果与参考前向传播结果相等
            self.assertEqual(res_fwd, ref_fwd)

            # 设置随机数种子为 1，用于参考和优化的反向传播张量
            torch.manual_seed(1)
            bwd_tensor_for_ref = torch.randn(ref_fwd.shape, dtype=dtype)
            torch.manual_seed(1)
            bwd_tensor_for_opt = torch.randn(res_fwd.shape, dtype=dtype)

            # 断言参考和优化的反向传播张量相等
            self.assertEqual(bwd_tensor_for_ref, bwd_tensor_for_opt)

            # 对参考前向传播和优化前向传播进行反向传播
            ref_fwd.backward(bwd_tensor_for_ref)
            res_fwd.backward(bwd_tensor_for_opt)

            # 获取参考和优化的输入梯度
            ref_grad = test_args_for_ref["input"].grad
            res_grad = test_args_for_opt["input"].grad

            # 断言参考和优化的输入梯度相等
            self.assertEqual(ref_grad, res_grad)

    # 定义一个测试函数 test_meta_device
    def test_meta_device(self):
        # 使用 @torch.compile(fullgraph=True) 装饰器编译下面的函数 fn
        @torch.compile(fullgraph=True)
        def fn():
            # 创建一个空张量 x，形状为 [1024, 128, 128]
            # 数据类型为 torch.float16，在 "meta" 设备上，不固定内存
            x = torch.ops.aten.empty.memory_format(
                [1024, 128, 128],
                dtype=torch.float16,
                device="meta",
                pin_memory=False,
            )
            # 对张量 x 执行 sin 函数并加 1，返回结果
            return x.sin() + 1

        # 断言 fn 函数返回的张量形状为 [1024, 128, 128]
        self.assertEqual(fn().shape, [1024, 128, 128])
    # 定义测试函数 test_decomposed_fake_quant_per_channel
    def test_decomposed_fake_quant_per_channel(self):
        # 定义 fake quantize 函数 fq，用于对输入进行通道精度的假量化
        def fq(input, scales, zero_points, axis, quant_min, quant_max):
            # 使用 PyTorch 提供的 fake_quantize_per_channel_affine 函数进行假量化
            res = torch.fake_quantize_per_channel_affine(
                input, scales, zero_points, axis, quant_min, quant_max
            )
            return res

        # 定义 quantized decomposed fake quant 函数 qdq，使用自定义的量化操作函数
        def qdq(input, scales, zero_points, axis, quant_min, quant_max):
            # 调用 PyTorch 自定义操作 quantized_decomposed.fake_quant_per_channel 进行假量化
            res = torch.ops.quantized_decomposed.fake_quant_per_channel(
                input, scales, zero_points, axis, quant_min, quant_max
            )
            return res

        # 定义运行 eager 模式下的 aten fake quant 函数
        def run_eager_aten_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            # 清空梯度信息
            input.grad = None
            # 执行 fq 函数进行前向计算
            res = fq(input, scales, zero_points, axis, quant_min, quant_max)
            # 对计算结果进行求和并反向传播
            res.sum().backward()
            return res, input.grad

        # 定义运行 eager 模式下的 decomposed fake quant 函数
        def run_eager_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            # 清空梯度信息
            input.grad = None
            # 执行 qdq 函数进行前向计算
            res = qdq(input, scales, zero_points, axis, quant_min, quant_max)
            # 对计算结果进行求和并反向传播
            res.sum().backward()
            return res, input.grad

        # 定义运行编译后的 decomposed fake quant 函数
        def run_compile_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        ):
            # 清空梯度信息
            input.grad = None
            # 编译 qdq 函数
            compiled_qdq = torch.compile(qdq)
            # 执行编译后的 qdq 函数进行前向计算
            res = compiled_qdq(input, scales, zero_points, axis, quant_min, quant_max)
            # 对计算结果进行求和并反向传播
            res.sum().backward()
            return res, input.grad

        # 创建一个形状为 [2, 3, 224, 224] 的随机张量作为输入
        input = torch.randn(2, 3, 224, 224)
        # 修改输入张量的特定位置的值为 257
        input[1, 2, 3, 4] = 257
        # 设置输入张量需要计算梯度
        input.requires_grad_()
        # 设置量化参数
        scales = torch.ones((3,))
        zero_points = torch.zeros((3,))
        axis = 1
        quant_min = -128
        quant_max = 127

        # 复制输入张量，用于 aten 模式下的计算
        aten_input = copy.deepcopy(input)
        # 复制输入张量，用于编译后的计算
        compiler_input = copy.deepcopy(input)

        # 在 eager 模式下运行 aten 的假量化计算
        res_aten_eager, input_grad_aten_eager = run_eager_aten_fake_quant(
            aten_input, scales, zero_points, axis, quant_min, quant_max
        )
        # 在 eager 模式下运行 decomposed 的假量化计算
        res_decomp_eager, input_grad_decomp_eager = run_eager_decomposed_fake_quant(
            input, scales, zero_points, axis, quant_min, quant_max
        )
        # 在编译后的模式下运行 decomposed 的假量化计算
        res, input_grad = run_compile_decomposed_fake_quant(
            compiler_input, scales, zero_points, axis, quant_min, quant_max
        )

        # 断言结果的一致性
        self.assertEqual(res_aten_eager, res)
        self.assertEqual(res_decomp_eager, res)
        self.assertEqual(input_grad_aten_eager, input_grad)
        self.assertEqual(input_grad_decomp_eager, input_grad)
        # 断言特定位置的梯度为零
        self.assertEqual(input_grad[1, 2, 3, 4], torch.tensor(0.0))
        # 验证内核的前向和反向计数
        check_metrics_vec_kernel_count(2)

    @requires_vectorization
    # 定义测试函数，用于测试布尔类型张量的操作掩码
    def test_ops_masked_with_bool_input(self):
        # 创建一个大小为129的全零张量，数据类型为布尔型
        x = torch.zeros(129, dtype=torch.bool)
        # 设置填充大小为[2, 3]
        size = [2, 3]
        # 使用 torch.constant_pad_nd 函数对张量 x 进行填充，得到结果张量 res_aten_eager
        res_aten_eager = torch.constant_pad_nd(x, size)
        # 编译 torch.constant_pad_nd 函数
        cfn = torch.compile(torch.constant_pad_nd)
        # 使用编译后的函数 cfn 对张量 x 进行填充，得到结果张量 res
        res = cfn(x, size)
        # 断言 res_aten_eager 和 res 相等
        self.assertEqual(res_aten_eager, res)
        # 检查向量化内核计数的指标，确保为1
        check_metrics_vec_kernel_count(1)

    # 定义测试函数，测试整数右移位操作
    def test_bitwise_right_shift(self):
        # 创建一个形状为(1, 1, 1)、设备为CPU、数据类型为64位整数的随机整数张量 x
        x = torch.randint(-1, 0, (1, 1, 1), device="cpu", dtype=torch.int64)
        # 设置右移位数为31
        bit_num = 31
        # 使用 torch.bitwise_right_shift 函数对张量 x 进行右移操作，得到结果张量 res_aten_eager
        res_aten_eager = torch.bitwise_right_shift(x, bit_num)
        # 编译 torch.bitwise_right_shift 函数
        cfn = torch.compile(torch.bitwise_right_shift)
        # 使用编译后的函数 cfn 对张量 x 进行右移操作，得到结果张量 res
        res = cfn(x, bit_num)
        # 断言 res_aten_eager 和 res 相等
        self.assertEqual(res_aten_eager, res)

    # 使用 mock 函数替换 torch.cuda.is_available，测试使用原子加法进行散射操作
    @patch("torch.cuda.is_available", lambda: False)
    def test_scatter_using_atomic_add(self):
        # 定义内部函数 fn，实现张量的散射操作
        def fn(a, dim, index, b):
            return aten.scatter(a, dim, index, b, reduce="add")

        # 设置输入参数
        inps = (
            torch.randn(5, 29, 13),
            2,
            torch.tensor([[[3, 5, 7, 9]]]),
            torch.randn(1, 1, 10),
        )

        # 定义内部检查函数 _internal_check，用于验证函数编译后的代码并进行功能验证
        def _internal_check(
            _fn,
            _inps,
            _target_code_check=None,
            _target_code_check_not=None,
        ):
            # 重置 torch._dynamo 和 metrics
            torch._dynamo.reset()
            metrics.reset()
            # 编译优化函数 _fn，并获取其生成的 C++ 代码
            _fn_opt = torch.compile()(_fn)
            _, code = run_and_get_cpp_code(_fn_opt, *inps)
            # 如果指定了目标代码检查，则使用 FileCheck 进行检查
            if _target_code_check:
                FileCheck().check(_target_code_check).run(code)
            # 如果指定了目标代码不应出现的检查，则使用 FileCheck 进行反向检查
            if _target_code_check_not:
                FileCheck().check_not(_target_code_check_not).run(code)
                # 验证输出代码不为空
                FileCheck().check("Output code:").run(code)

            # 断言原始函数和优化函数在相同输入下的输出结果一致
            self.assertEqual(
                _fn(*_inps),
                _fn_opt(*_inps),
            )

        # 使用 config.patch 设置 cpp.fallback_scatter_reduce_sum 为 False
        with config.patch({"cpp.fallback_scatter_reduce_sum": False}):
            # 执行内部检查函数 _internal_check，验证使用原子加法时的代码生成
            _internal_check(fn, inps, "atomic_add")

        # 使用 config.patch 设置 cpp.fallback_scatter_reduce_sum 为 True
        with config.patch({"cpp.fallback_scatter_reduce_sum": True}):
            # 执行内部检查函数 _internal_check，验证使用标准散射算法时的代码生成
            _internal_check(fn, inps, "aten.scatter_reduce_")

        # 如果 ATen 使用 OpenMP 并行后端
        if "ATen parallel backend: OpenMP" in torch.__config__.parallel_info():
            # 设置线程数为1，验证在单线程情况下是否会使用 C++ 后端而非标准散射算法
            with set_num_threads(1):
                # 使用 config.patch 禁用 fx_graph_cache 和 fx_graph_remote_cache
                with config.patch(
                    {"fx_graph_cache": False, "fx_graph_remote_cache": False}
                ):
                    # 执行内部检查函数 _internal_check，验证不使用标准散射算法的代码生成
                    _internal_check(
                        fn, inps, _target_code_check_not="aten.scatter_reduce_"
                    )

            # 使用 config.patch 设置 cpp.dynamic_threads 为 True，设置线程数为1
            with config.patch({"cpp.dynamic_threads": True}), set_num_threads(1):
                # 执行内部检查函数 _internal_check，验证使用标准散射算法的代码生成
                _internal_check(fn, inps, "aten.scatter_reduce_")

    # 声明装饰器 requires_vectorization，同时使用 mock 函数替换 torch.cuda.is_available
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    # 定义一个测试函数，测试仅限于 CPU 的新向量操作
    def test_new_vec_op_cpu_only(self):
        # 定义一个函数 fn，对输入 x 进行操作：先误差函数，再指数减一，最后对数加一
        def fn(x):
            return torch.log1p(torch.expm1(torch.erf(x)))

        # 对于指定的向量数据类型进行迭代测试
        for dtype in vec_dtypes:
            # 设置随机种子为 0
            torch.manual_seed(0)
            # 生成一个指定类型和形状的随机张量 x
            x = torch.randn((2, 9), dtype=dtype)
            # 将张量 x 的特定元素设置为 NaN
            x[0, 0] = torch.nan
            x[1, -1] = torch.nan

            # 如果数据类型为 torch.bfloat16，则设置容差为 1e-2，否则为 1e-4
            tol = 1e-2 if dtype == torch.bfloat16 else 1e-4

            # 使用指定的配置参数进入代码块
            with config.patch({"cpp.simdlen": None}):
                # 对于 cpp_wrapper_flag 的两个值 True 和 False 进行迭代测试
                for cpp_wrapper_flag in [True, False]:
                    # 根据 cpp_wrapper_flag 的值设置配置参数，并进入代码块
                    with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                        # 重置 dynamo 和 metrics
                        torch._dynamo.reset()
                        metrics.reset()
                        # 调用公共方法 common 执行 fn 函数，并传入 x 作为参数
                        self.common(fn, (x,))
                        # 检查向量内核计数是否为 1
                        check_metrics_vec_kernel_count(1)

    # 标记要求进行向量化，且在 CUDA 不可用时进行 CPU 测试
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_cpu_only_for_all_available_isa(self):
        # 定义一个函数 fn，对输入 x 进行操作：先误差函数，再余弦，最后正弦
        def fn(x):
            return torch.sin(torch.cos(torch.erf(x))))

        # 生成一个指定形状的随机张量 x，并设置部分元素为 NaN
        x = torch.randn((2, 9))
        x[0, 0] = torch.nan
        x[1, -1] = torch.nan

        # 获取所有有效的 CPU 向量指令集的位宽，并包括 None
        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()] + [
            None
        ]
        # 对于 bit_widths 中的每个元素进行迭代测试
        for item in bit_widths:
            # 使用指定的 simdlen 配置参数进入代码块
            with config.patch({"cpp.simdlen": item}):
                # 重置 dynamo 和 metrics
                torch._dynamo.reset()
                metrics.reset()
                # 调用公共方法 common 执行 fn 函数，并传入 x 作为参数
                self.common(fn, (x,))
                # 检查向量内核计数是否为 1
                check_metrics_vec_kernel_count(1)

    # 标记测试为慢速测试，要求进行向量化，并且在 CUDA 不可用时进行 CPU 测试
    @slowTest
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test__adaptive_avg_pool2d(self):
        # 定义一个包装函数 wrap_fn，生成一个函数 fn，对输入 x 执行自适应平均池化操作
        def wrap_fn(oh, ow):
            def fn(x):
                return torch._adaptive_avg_pool2d(x, (oh, ow))

            return fn

        # 获取所有有效的 CPU 向量指令集的位宽
        bit_widths = [isa._bit_width for isa in cpu_vec_isa.valid_vec_isa_list()]
        # 定义输入张量的高度和宽度列表
        ih = [16, 65]
        iw = ih
        oh = ih
        ow = ih
        # 对于 ih, iw, oh, ow, bit_widths 和 vec_dtypes 的笛卡尔积进行迭代测试
        for _ih, _iw, _oh, _ow, _simd_len, dtype in itertools.product(
            ih, iw, oh, ow, bit_widths, vec_dtypes
        ):
            # 生成一个指定形状和数据类型的随机张量 x，并使用通道优先的内存格式
            x = torch.randn(2, 3, _ih, _iw, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            # 生成一个根据 _oh 和 _ow 的值生成的函数 _fn
            _fn = wrap_fn(_oh, _ow)
            # 使用指定的 simdlen 配置参数进入代码块
            with config.patch({"cpp.simdlen": _simd_len}):
                # 重置 dynamo 和 metrics
                torch._dynamo.reset()
                metrics.reset()
                # 调用公共方法 common 执行 _fn 函数，并传入 x 作为参数
                self.common(_fn, (x,))
                # 检查向量内核计数是否为 1
                check_metrics_vec_kernel_count(1)

    # 标记要求进行向量化，且在 CUDA 不可用时进行 CPU 测试
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_vec_logical(self):
        # 定义一个包装函数，接受一个操作符作为参数，并返回一个操作函数，将操作结果转换为 1.0 或 0.0
        def wrap_fn1(op: Callable):
            def fn(x: torch.Tensor):
                return torch.where(op(x), 1.0, 0.0)
            return fn

        # 定义另一个包装函数，接受一个操作符作为参数，并返回一个操作函数，将操作结果转换为 1.0 或 0.0
        def wrap_fn2(op: Callable):
            def fn(x: torch.Tensor, y: torch.Tensor):
                return torch.where(op(x, y), 1.0, 0.0)
            return fn

        # 对于每种向量化数据类型
        for dtype in vec_dtypes:
            # 生成一个随机张量 x 和 y，指定数据类型为当前循环的 dtype
            x = torch.randn(64, dtype=dtype)
            y = torch.randn(64, dtype=dtype)
            # 定义逻辑函数列表
            logical_fns = [
                torch.logical_and,
                torch.logical_not,
                torch.logical_or,
                torch.logical_xor,
            ]
            # 对于每个逻辑函数
            for logical_fn in logical_fns:
                # 重置 Torch 的动态图和度量
                torch._dynamo.reset()
                metrics.reset()
                # 根据逻辑函数选择包装函数和参数
                if logical_fn == torch.logical_not:
                    _fn = wrap_fn1(logical_fn)
                    _args = (x,)
                else:
                    _fn = wrap_fn2(logical_fn)
                    _args = (x, y)
                # 调用共同的测试函数
                self.common(_fn, _args)
                # 检查向量化内核计数的度量值
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    def test_vec_bitwise(self):
        # 对于每种数据类型
        for dtype in [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ]:
            # 生成一个随机张量 x 和 y，数据类型为 torch.float32
            x = torch.randn(64, dtype=torch.float32)
            y = torch.randn(64, dtype=torch.float32)
            # 如果数据类型是 bool
            if dtype == torch.bool:
                x = x > 0
                y = y > 0
            else:
                # 将 x 和 y 转换为当前循环的数据类型
                x = x.to(dtype)
                y = y.to(dtype)
            # 定义位运算函数列表
            bitwise_fns = [
                torch.bitwise_and,
                torch.bitwise_not,
                torch.bitwise_or,
                torch.bitwise_xor,
                torch.bitwise_left_shift,
                torch.bitwise_right_shift,
            ]
            # 对于每个位运算函数
            for bitwise_fn in bitwise_fns:
                # 如果当前位运算函数是左移或右移，并且数据类型是 bool，跳过该函数的测试
                if (
                    bitwise_fn
                    in [
                        torch.bitwise_left_shift,
                        torch.bitwise_right_shift,
                    ]
                    and dtype == torch.bool
                ):
                    continue
                # 重置 Torch 的动态图和度量
                torch._dynamo.reset()
                metrics.reset()
                # 根据位运算函数选择参数
                if bitwise_fn == torch.bitwise_not:
                    _args = (x,)
                else:
                    _args = (x, y)
                # 调用共同的测试函数
                self.common(bitwise_fn, _args)
                # 检查向量化内核计数的度量值
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    # 定义一个测试方法，用于测试只在 CPU 上执行的向量比较操作
    def test_vec_compare_op_cpu_only(self):
        # 定义一个内部函数 fn，接受一个张量 x 作为输入，并进行一系列操作后返回结果张量 x
        def fn(x):
            # 判断 x 中元素是否等于 1.0，返回布尔张量 y1
            y1 = torch.eq(x, 1.0)
            # 根据 y1 的结果，在 x 中选择满足条件的元素或者取相反数，更新 x
            x = torch.where(y1, x, -x)
            # 判断 x 中元素是否不等于 0.0，返回布尔张量 y2
            y2 = torch.ne(x, 0.0)
            # 根据 y2 的结果，在 x 中选择满足条件的元素或者取相反数，更新 x
            x = torch.where(y2, x, -x)
            # 判断 x 中元素是否小于 5.0，返回布尔张量 y3
            y3 = torch.lt(x, 5.0)
            # 根据 y3 的结果，在 x 中选择满足条件的元素或者做减法操作，更新 x
            x = torch.where(y3, x, x - 1.0)
            # 判断 x 中元素是否大于 -2.0，返回布尔张量 y4
            y4 = torch.gt(x, -2.0)
            # 根据 y4 的结果，在 x 中选择满足条件的元素或者做加法操作，更新 x
            x = torch.where(y4, x, x + 1.0)
            # 判断 x 中元素是否小于等于 8.0，返回布尔张量 y5
            y5 = torch.le(x, 8.0)
            # 根据 y5 的结果，在 x 中选择满足条件的元素或者做减法操作，更新 x
            x = torch.where(y5, x, x - 1.0)
            # 判断 x 中元素是否大于等于 -3.0，返回布尔张量 y6
            y6 = torch.ge(x, -3.0)
            # 根据 y6 的结果，在 x 中选择满足条件的元素或者做加法操作，更新 x
            x = torch.where(y6, x, x + 1.0)
            # 判断 x 中元素是否等于 1.0，返回布尔张量 y7
            y7 = x == 1.0
            # 根据 y7 的结果，在 x 中选择满足条件的元素或者取相反数，更新 x
            x = torch.where(y7, x, -x)
            # 判断 x 中元素是否不等于 0.0，返回布尔张量 y8
            y8 = x != 0.0
            # 根据 y8 的结果，在 x 中选择满足条件的元素或者取相反数，更新 x
            x = torch.where(y8, x, -x)
            # 判断 x 中元素是否小于 5.0，返回布尔张量 y9
            y9 = x < 5.0
            # 根据 y9 的结果，在 x 中选择满足条件的元素或者做减法操作，更新 x
            x = torch.where(y9, x, x - 1.0)
            # 判断 x 中元素是否大于 -2.0，返回布尔张量 y10
            y10 = x > -2.0
            # 根据 y10 的结果，在 x 中选择满足条件的元素或者做加法操作，更新 x
            x = torch.where(y10, x, x + 1.0)
            # 判断 x 中元素是否小于等于 8.0，返回布尔张量 y11
            y11 = x <= 8.0
            # 根据 y11 的结果，在 x 中选择满足条件的元素或者做减法操作，更新 x
            x = torch.where(y11, x, x - 1.0)
            # 判断 x 中元素是否大于等于 -3.0，返回布尔张量 y12
            y12 = x >= -3.0
            # 根据 y12 的结果，在 x 中选择满足条件的元素或者做加法操作，更新 x
            x = torch.where(y12, x, x + 1.0)
            # 返回更新后的张量 x
            return x

        # 对指定的向量数据类型进行循环测试
        for dtype in vec_dtypes:
            # 创建一个形状为 (2, 9) 的随机张量 x，并指定数据类型为 dtype
            x = torch.randn((2, 9), dtype=dtype)

            # 在配置中暂时禁用特定的配置项
            with config.patch({"cpp.simdlen": None}):
                # 重置 Torch 的动态机制
                torch._dynamo.reset()
                # 重置性能指标
                metrics.reset()
                # 调用通用测试方法 common，传入 fn 函数和 x 作为参数
                self.common(fn, (x,))
                # 检查向量化内核计数是否为 1
                check_metrics_vec_kernel_count(1)
                # 断言生成的内核计数减去生成的 CPP 向量化内核计数是否为 0
                assert (
                    metrics.generated_kernel_count
                    - metrics.generated_cpp_vec_kernel_count
                ) == 0

    # 定义一个测试方法，用于跳过 CPP 代码生成的测试
    def test_skip_cpp_codegen(self):
        # 在配置中暂时设置禁用 CPP 代码生成
        with config.patch({"disable_cpp_codegen": True}):
            # 创建两个张量数据，一个元素全为 1，一个为随机值
            inps = torch.ones([20]), torch.rand([20])

            # 定义一个函数 f，接受两个参数 x 和 y，并返回它们的和再加上常量 1
            def f(x, y):
                return x + y + torch.tensor(1)

            # 对函数 f 进行 Torch 的编译优化
            f_opt = torch.compile()(f)

            # 运行并获取函数 f_opt 在输入数据 inps 上的 CPP 代码
            _, code = run_and_get_cpp_code(f_opt, inps[0], inps[1])
            # 使用 FileCheck 工具检查生成的 CPP 代码中是否不含有 "void kernel"
            FileCheck().check_not("void kernel").run(code)

            # 断言函数 f 在输入 inps 上的输出与优化后的 f_opt 输出一致
            self.assertEqual(
                f(*inps),
                f_opt(*inps),
            )

            # 定义一个函数 f，接受一个参数 x，并返回 x 中索引为 1 到末尾的元素乘以 2
            # 在 CPP 代码生成被禁用时，常量应当传播
            def f(x):
                return x[torch.tensor(1) :] * 2

            # 对函数 f 进行 Torch 的编译优化
            f_opt = torch.compile()(f)
            # 运行并获取函数 f_opt 在输入数据 inps[0] 上的 CPP 代码
            _, code = run_and_get_cpp_code(f_opt, inps[0])
            # 使用 FileCheck 工具检查生成的 CPP 代码中是否不含有 "void kernel"
            FileCheck().check_not("void kernel").run(code)
            # 断言函数 f 在输入 inps[0] 上的输出与优化后的 f_opt 输出一致
            self.assertEqual(f_opt(inps[0]), f(inps[0]))

            # 定义一个继承自 torch.nn.Module 的模型类 Model
            class Model(torch.nn.Module):
                def __init__(
                    self,
                ):
                    super().__init__()

                # 实现模型的前向传播方法，接受一个名为 v1 的张量参数，并返回处理后的张量 v2
                def forward(self, v1: torch.Tensor):
                    # 计算 v1 每行的最小值，返回结果作为张量 vx
                    vx = v1.min(dim=1).values
                    # 创建一个与 vx 形状相同的随机张量 v2，并返回
                    v2 = torch.randn_like(vx)
                    return v2

            # 创建一个 Model 类的实例 model
    def test_redundant_to_node_elimination_lowp_fp(self):
        # 定义测试函数，对两个输入张量执行加法并计算平均值
        def fn(x, y):
            # 计算加法结果
            res = x + y
            # 计算加法结果的平均值
            res = torch.mean(res)
            return res

        # 遍历低精度浮点数类型的数据类型列表
        for dtype in _lowp_fp_dtypes:
            # 创建指定类型和形状的随机张量 x 和 y
            x = torch.randn((2, 9), dtype=dtype)
            y = torch.randn((2, 9), dtype=dtype)

            # 遍历是否开启 Torch 编译调试模式的选项
            for torch_compile_debug in [True, False]:
                # 设置 Torch 配置项，启用或禁用跟踪，并设置 SIMD 长度为默认
                with config.patch(
                    {"trace.enabled": torch_compile_debug, "cpp.simdlen": None}
                ):
                    # 重置 Torch Dynamo 状态
                    torch._dynamo.reset()
                    # 重置性能指标
                    metrics.reset()
                    # 执行共同操作，传入测试函数和参数 (x, y)
                    self.common(fn, (x, y))
                    # 检查向量化内核计数是否为 1
                    check_metrics_vec_kernel_count(1)

    def test_do_not_insert_to_dtype_for_memory_copy_only_kernel(self):
        # 定义测试函数，对输入张量执行克隆操作
        def fn(x):
            # 使用克隆方法复制输入张量 x
            res = x.clone()
            return res

        # 创建指定形状和数据类型的随机张量 x
        x = torch.randn((100, 100), dtype=torch.bfloat16)

        # 重置 Torch Dynamo 状态
        torch._dynamo.reset()
        # 重置性能指标
        metrics.reset()
        # 执行共同操作，传入测试函数和参数 (x,)
        self.common(fn, (x,))
        # 断言 CPP 到数据类型计数为 0
        assert metrics.cpp_to_dtype_count == 0
        # 检查向量化内核计数是否为 1
        check_metrics_vec_kernel_count(1)

    def test_insert_to_dtype_count(self):
        # 定义测试函数，对输入张量执行 ReLU 激活函数
        def fn(x):
            # 使用 ReLU 函数对输入张量 x 进行激活
            res = x.relu()
            return res

        # 创建指定形状和数据类型的随机张量 x
        x = torch.randn((100, 100), dtype=torch.bfloat16)

        # 重置 Torch Dynamo 状态
        torch._dynamo.reset()
        # 重置性能指标
        metrics.reset()
        # 执行共同操作，传入测试函数和参数 (x,)
        self.common(fn, (x,))
        # 断言 CPP 到数据类型计数为 2
        assert metrics.cpp_to_dtype_count == 2
        # 检查向量化内核计数是否为 1
        check_metrics_vec_kernel_count(1)

    def test_memory_copy_with_fusion(self):
        # 定义测试函数，对输入张量执行 ReLU 激活函数，并进行内存拷贝
        def fn(x):
            # 使用 ReLU 函数对输入张量 x 进行激活
            res = x.relu()
            # 将激活后的结果 res 拷贝回输入张量 x
            x.copy_(res)
            return (res,)

        # 创建指定形状和数据类型的随机张量 x
        x = torch.randn((100, 100), dtype=torch.bfloat16)

        # 重置 Torch Dynamo 状态
        torch._dynamo.reset()
        # 重置性能指标
        metrics.reset()
        # 执行共同操作，传入测试函数和参数 (x,)
        self.common(fn, (x,))
        # 断言 CPP 到数据类型计数为 2
        assert metrics.cpp_to_dtype_count == 2
        # 检查向量化内核计数是否为 1
        check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_maxpool2d_cpu_only(self):
        # 遍历向量化数据类型列表
        for dtype in vec_dtypes:
            # 创建指定形状和数据类型的随机输入张量 input，并使用通道优先内存格式
            input = torch.randn(26, 32, 112, 112, dtype=dtype).to(
                memory_format=torch.channels_last
            )
            # 创建 MaxPool2d 池化层对象
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # 定义函数 func，对输入张量执行 MaxPool2d 操作
            def func(x):
                return maxpool(x)

            # 使用 SIMD 长度为默认值，重置 Torch Dynamo 状态
            with patch.object(config.cpp, "simdlen", None):
                # 重置 Torch Dynamo 状态
                torch._dynamo.reset()
                # 重置性能指标
                metrics.reset()
                # 执行共同操作，传入测试函数和参数 (input,)
                self.common(func, (input,))
                # 检查向量化内核计数是否为 1
                check_metrics_vec_kernel_count(1)

    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    # 定义一个测试函数，用于测试仅在 CPU 上的 MaxPool2d 操作，通过 channels_last 内存格式处理输入
    def test_maxpool2d_with_pre_loop_collapse_cpu_only(self):
        # 创建两个随机张量 x1 和 x2，形状为 (2, 3, 20, 20)，使用 channels_last 内存格式
        x1 = torch.randn(2, 3, 20, 20).to(memory_format=torch.channels_last)
        x2 = torch.randn(2, 3, 20, 20).to(memory_format=torch.channels_last)
        # 创建一个 MaxPool2d 层，设置内核大小为 3，步长为 2，启用 ceil_mode
        maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 定义一个内部函数 func，接收两个输入张量 x1 和 x2，返回它们经过 maxpool 处理后的结果
        def func(x1, x2):
            # 将 x1 和 x2 相加
            y = x1 + x2
            # 对 y 执行最大池化操作
            return maxpool(y)

        # 使用 patch.object 临时替换 config.cpp.simdlen 为 None
        with patch.object(config.cpp, "simdlen", None):
            # 重置 torch._dynamo 状态
            torch._dynamo.reset()
            # 重置 metrics 状态
            metrics.reset()
            # 调用测试辅助函数 self.common，传入 func 函数和参数元组 (x1, x2)
            self.common(func, (x1, x2))
            # 检查生成的向量化内核数量是否为 2
            check_metrics_vec_kernel_count(2)

    # 测试函数，验证处理整数输入的随机数生成函数 get_traj_idx
    def test_randint_symint_input(self):
        # 使用 @torch.compile 注解将 get_traj_idx 函数编译为 TorchScript
        @torch.compile(fullgraph=True)
        def get_traj_idx(lengths: torch.Tensor, num_slices: int) -> torch.Tensor:
            return torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)

        # 创建长度为 10 的零张量 lengths，数据类型为 long
        lengths = torch.zeros(10, dtype=torch.long)
        # 调用 get_traj_idx 函数，传入 lengths 和 num_slices=4
        get_traj_idx(lengths, num_slices=4)
        # 创建长度为 11 的零张量 lengths，数据类型为 long
        lengths = torch.zeros(11, dtype=torch.long)
        # 再次调用 get_traj_idx 函数，传入 lengths 和 num_slices=4
        get_traj_idx(lengths, num_slices=4)

    # 标记需要矢量化的测试函数，并禁用 CUDA 后，测试 torch.sign 函数在 CPU 上的行为
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_sign_cpu_only(self):
        # 定义一个函数 fn，接收一个张量 x，返回其每个元素的符号函数值
        def fn(x):
            return torch.sign(x)

        # 遍历 vec_dtypes 中的数据类型
        for dtype in vec_dtypes:
            # 创建一个随机张量 x，形状为 (2, 9)，数据类型为当前循环的 dtype
            x = torch.randn((2, 9), dtype=dtype)
            # 将 x 的第一个元素设置为 NaN
            x[0, 0] = torch.nan
            # 将 x 的最后一个元素设置为 NaN
            x[1, -1] = torch.nan

            # 使用 config.patch 临时替换 cpp.simdlen 为 None
            with config.patch({"cpp.simdlen": None}):
                # 重置 torch._dynamo 状态
                torch._dynamo.reset()
                # 重置 metrics 状态
                metrics.reset()
                # 调用测试辅助函数 self.common，传入 fn 函数和参数元组 (x,)
                self.common(fn, (x,))
                # 检查生成的向量化内核数量是否为 1
                check_metrics_vec_kernel_count(1)

    # 标记需要矢量化的测试函数，并禁用 CUDA 后，测试 torch.argmax 函数在 CPU 上的行为
    @requires_vectorization
    @patch("torch.cuda.is_available", lambda: False)
    def test_reduction_cpu_only(self):
        # 定义一个函数 fn，接收一个张量 x，返回其在指定维度上的最大值索引
        def fn(x):
            return torch.argmax(x, -1)

        # 遍历 vec_dtypes 中的数据类型
        for dtype in vec_dtypes:
            # 创建一个随机张量 x，形状为 (10, 10)，数据类型为当前循环的 dtype
            x = torch.randn((10, 10), dtype=dtype)

            # 使用 config.patch 临时替换 cpp.simdlen 为 None
            with config.patch({"cpp.simdlen": None}):
                # 重置 torch._dynamo 状态
                torch._dynamo.reset()
                # 重置 metrics 状态
                metrics.reset()
                # 调用测试辅助函数 self.common，传入 fn 函数和参数元组 (x,)
                self.common(fn, (x,))
                # 断言生成的向量化内核数量为 0
                assert metrics.generated_cpp_vec_kernel_count == 0

    # 测试函数，验证在禁用 SIMD 后，处理张量的外部循环融合是否成功
    def test_outer_loop_fusion(self):
        # 定义一个函数 fn，接收一个张量 x，计算其每行元素减去该行最大值后的结果
        def fn(x):
            max = torch.amax(x, dim=-1, keepdim=True)
            return x - max

        # 创建一个形状为 (4, 12, 1023, 1022) 的随机张量 x
        x = torch.randn(4, 12, 1023, 1022)

        # 使用 config.patch 临时替换 cpp.simdlen 为 None
        with config.patch({"cpp.simdlen": None}):
            # 重置 torch._dynamo 状态
            torch._dynamo.reset()
            # 重置 metrics 状态
            metrics.reset()
            # 调用测试辅助函数 self.common，传入 fn 函数和参数元组 (x,)
            self.common(fn, (x,))
            # 断言 metrics 中的 cpp_outer_loop_fused_inner_counts 数组长度为 1
            assert len(metrics.cpp_outer_loop_fused_inner_counts) == 1
            # 断言 metrics 中 cpp_outer_loop_fused_inner_counts 的第一个元素为 2
            assert metrics.cpp_outer_loop_fused_inner_counts[0] == 2

    # 测试函数，验证 torch.argmin 函数在不同数据类型的张量上的行为
    def test_argmin(self):
        # 定义一个函数 fn，接收一个张量 x，返回其在指定维度上的最小值索引
        def fn(x):
            return torch.argmin(x, -1)

        # 遍历 vec_dtypes 中的数据类型
        for dtype in vec_dtypes:
            # 创建一个随机张量 x，形状为 (10, 10)，数据类型为当前循环的 dtype
            x = torch.randn((10, 10), dtype=dtype)
            # 重置 torch._dynamo 状态
            torch._dynamo.reset()
            # 重置 metrics 状态
            metrics.reset()
            # 调用测试辅助函数 self.common，传入 fn 函数和参数元组 (x,)
            self.common(fn, (x,))
            # 断言生成的向量化内核数量为 0
            assert metrics.generated_cpp_vec_kernel_count == 0
    # 定义一个测试函数，用于计算张量 x 的最大值索引
    def test_argmax_argmin_with_nan_value(self):
        # 定义一个函数 fn，返回张量 x 的最大值索引
        def fn(x):
            return torch.argmax(x)

        # 定义一个函数 fn2，返回张量 x 的最小值索引
        def fn2(x):
            return torch.argmin(x)

        # 创建输入张量列表，每个张量包含不同数量和位置的值
        inputs = [
            torch.Tensor([-755832.1250, 100]),
            torch.Tensor([-755832.1250, 100, 200]),
            torch.Tensor([100, -755832.1250]),
            torch.Tensor([100, 200, -755832.1250]),
        ]

        # 对于每个输入张量 x
        for x in inputs:
            # 将张量 x 扩展为 16x16 的形状
            x = x.repeat(16, 16)
            # 对张量 x 应用 log1p 函数，计算 log(1 + x)
            x = torch.log1p(x)

            # 测试 argmax 函数
            torch._dynamo.reset()  # 重置动态图优化器状态
            metrics.reset()  # 重置度量指标
            self.common(fn, (x,))  # 调用共通方法 common，传入函数 fn 和参数 x
            assert metrics.generated_cpp_vec_kernel_count == 0  # 断言生成的 C++ 向量化内核数为 0

            # 测试 argmin 函数
            torch._dynamo.reset()  # 重置动态图优化器状态
            metrics.reset()  # 重置度量指标
            self.common(fn2, (x,))  # 调用共通方法 common，传入函数 fn2 和参数 x
            assert metrics.generated_cpp_vec_kernel_count == 0  # 断言生成的 C++ 向量化内核数为 0

    # 当前已启用 AVX2 和 AVX512 进行向量化。如果平台不支持，向量化将无效并跳过此测试用例。
    # 对于 ARM 或其他支持的平台，只需将 ISA 信息添加到 supported_vector_isa 中，
    # 并包含适当的 aten 向量化头文件。
    @requires_vectorization  # 装饰器，指示需要进行向量化
    @patch("torch.cuda.is_available", lambda: False)  # 使用 lambda 函数模拟 CUDA 不可用
    def test_vec_kernel_cpu_only(self):
        # 定义一个内部函数 fn，用于测试向量化计算的 CPU 版本
        def fn(x1, x2):
            # 对输入 x1 进行绝对值操作
            x = torch.abs(x1)
            # 对 x1 中的每个元素求正弦值
            x = torch.sin(x)
            # 对 x 中的每个元素取负数
            x = torch.neg(x)
            # 对 x 中的每个元素求平方
            x = torch.square(x)
            # 对 x 中的每个元素应用 sigmoid 函数
            x = torch.sigmoid(x)
            # 对 x 中的每个元素应用 ReLU 激活函数
            x = torch.relu(x)
            # 对 x 中的每个元素求余弦值
            x = torch.cos(x)
            # 对 x 中的每个元素求指数
            x = torch.exp(x)
            # 对 x 中的每个元素求平方根
            x = torch.sqrt(x)
            # 将 x 与 x1 逐元素相加
            x = torch.add(x, x1)
            # 将 x 减去 x2
            x = torch.sub(x, x2)
            # 将 x 与 x1 逐元素相乘
            x = torch.mul(x, x1)
            # 将 x 与 x1 逐元素相除
            x = torch.div(x, x1)
            # 将 x 中的每个元素求 10 次方
            x = torch.pow(x, 10)
            # 对 x 中的每个元素求自然对数
            x = torch.log(x)
            # 对 x 中的每个元素向下取整
            x = torch.floor(x)
            # 对 x 中的每个元素向上取整
            x = torch.ceil(x)
            # 对 x 中的每个元素取整数部分
            x = torch.trunc(x)
            # 对 x 中的每个元素求对数 Gamma 函数的自然对数
            x = torch.lgamma(x)
            # 对 x 中的每个元素求除法余数
            x = torch.fmod(x, x2)
            # 返回 x 与 x2 逐元素相加的结果
            res = x + x2
            return res

        # 对不同的数据类型进行测试
        for dtype in vec_dtypes:
            # 设置随机数种子为 0
            torch.manual_seed(0)
            # 生成 dtype 类型的随机张量 x1 和 x2，形状为 (5, 20)
            x1 = torch.randn((5, 20), dtype=dtype)
            x2 = torch.randn((5, 20), dtype=dtype)

            # 根据数据类型设置容差值
            tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
            # 使用 simdlen=1 的配置进行测试
            with config.patch({"cpp.simdlen": 1}):
                # 重置 Torch 内部状态和度量
                torch._dynamo.reset()
                metrics.reset()
                # 调用公共测试函数，测试 fn 函数
                self.common(fn, (x1, x2))
                # 断言生成的 CPP 向量化内核数量为 0
                assert metrics.generated_cpp_vec_kernel_count == 0

            # 使用默认 simdlen 配置进行测试
            with config.patch({"cpp.simdlen": None}):
                # 重置 Torch 内部状态和度量
                torch._dynamo.reset()
                metrics.reset()
                # 调用公共测试函数，测试 fn 函数
                self.common(fn, (x1, x2))
                # 检查生成的 CPP 向量化内核数量为 1
                check_metrics_vec_kernel_count(1)

        # 使用默认 simdlen 配置进行测试
        with config.patch({"cpp.simdlen": None}):
            # 重置 Torch 内部状态和度量
            torch._dynamo.reset()
            metrics.reset()
            # 生成形状为 (10, 20) 的 x1 张量的转置
            x1 = torch.randn(10, 20).permute(1, 0)
            # 生成形状为 (20, 10) 的 x2 张量
            x2 = torch.randn((20, 10))
            # 调用公共测试函数，测试 fn 函数
            self.common(fn, (x1, x2))
            # 检查生成的 CPP 向量化内核数量为 2
            check_metrics_vec_kernel_count(2)

            # 重置 Torch 内部状态和度量
            torch._dynamo.reset()
            metrics.reset()
            # 生成形状为 (10, 7) 的 x1 和 x2 张量
            x1 = torch.randn((10, 7))
            x2 = torch.randn((10, 7))
            # 调用公共测试函数，测试 fn 函数
            self.common(fn, (x1, x2))
            # 检查生成的 CPP 向量化内核数量为 1
            check_metrics_vec_kernel_count(1)

    # 跳过非 Linux 平台的测试，仅在 Linux 平台下支持 CPP 内核性能分析
    @unittest.skipIf(
        sys.platform != "linux", "cpp kernel profile only support linux now"
    )
    # 使用 lambda 函数设置在 GPU 不可用时返回 False
    @patch("torch.cuda.is_available", lambda: False)
    # 使用配置设置启用 CPP 内核性能分析，并设置描述性名称为 original_aten
    @config.patch({"cpp.enable_kernel_profile": True})
    @config.patch({"cpp.descriptive_names": "original_aten"})
    def test_cpp_kernel_profile(self):
        # 导入 torch 的性能分析器模块
        from torch.profiler import profile
        
        # 定义一个被优化的函数 fn，使用 torch._dynamo.optimize 进行优化，禁用 Python 代码执行
        @torch._dynamo.optimize("inductor", nopython=True)
        def fn(a, b):
            return a + b

        # 创建随机张量 a 和 b
        a = torch.rand((100,))
        b = torch.rand((100,))
        
        # 使用性能分析器进行性能分析
        with profile() as prof:
            fn(a, b)

        # 从性能分析结果中提取所有与 "cpp_fused_add_0" 相关的事件
        kernel_profile_events = []
        for e in prof.profiler.function_events:
            if "cpp_fused_add_0" in e.name:
                kernel_profile_events.append(e.name)
        
        # 断言至少有一个与 "cpp_fused_add_0" 相关的事件被记录
        assert len(kernel_profile_events) > 0

    @requires_vectorization
    def test_channel_shuffle_cl_output(self):
        """code and shape extracted from shufflenet_v2_x1_0"""

        # 定义通道混洗函数 channel_shuffle
        def channel_shuffle(x, groups):
            # 获取张量 x 的尺寸信息
            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups
            
            # 重塑张量 x 的形状
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            
            # 使用通道优先的内存格式重新排列张量 x
            return x.contiguous(memory_format=torch.channels_last)

        # 遍历不同的 simdlen 参数进行测试
        for simdlen in (None, 256, 1):
            with config.patch({"cpp.simdlen": simdlen}):
                # 重置 torch._dynamo 和性能度量
                torch._dynamo.reset()
                metrics.reset()
                
                # 创建随机张量 x
                x = torch.randn(64, 58, 28, 28)
                
                # 使用通道混洗函数，并测试通道数为 2 的情况
                self.common(channel_shuffle, (x, 2))
                
                # 如果 simdlen 不为 1，则检查向量化内核计数
                if simdlen != 1:
                    check_metrics_vec_kernel_count(2)

    @slowTest
    @requires_vectorization
    def test_transpose_with_norm(self):
        """a sub-module from TIMM gmlp_s16_224"""

        # 定义一个模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的层结构
                self.linear = torch.nn.Linear(
                    in_features=256, out_features=1536, bias=True
                )
                self.act = torch.nn.GELU()
                self.norm = torch.nn.LayerNorm(768)
                self.proj = torch.nn.Linear(196, 196)
                self.fc = torch.nn.Linear(in_features=768, out_features=256, bias=True)

            # 模型的前向传播
            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                u, v = x.chunk(2, dim=-1)
                v = self.norm(v)
                v = self.proj(v.transpose(-1, -2))
                y = u * v.transpose(-1, -2)
                return self.fc(y)

        # 创建随机张量 x
        x = torch.randn(128, 196, 256)
        
        # 遍历不同的 simdlen 参数进行测试
        for simdlen in (None, 256, 1):
            with config.patch({"cpp.simdlen": simdlen}):
                # 遍历不同的 eval_mode 模式
                for eval_mode in [True, False]:
                    # 重置 torch._dynamo 和性能度量
                    torch._dynamo.reset()
                    metrics.reset()
                    
                    # 创建 Model 实例，如果 eval_mode 为 True 则设置为评估模式
                    m = Model().eval() if eval_mode else Model()
                    
                    # 使用通用测试函数，并传入随机张量 x
                    self.common(m, (x,))
                    
                    # 如果 simdlen 不为 1，则检查向量化内核计数
                    if simdlen != 1:
                        check_metrics_vec_kernel_count(8)
    # 定义一个测试函数，用于测试矩阵转置并复制操作的正确性
    def test_transpose_copy(self):
        # 定义内部函数 fn，接受一个张量 a，并返回其转置后的连续副本
        def fn(a):
            return a.t().contiguous()
    
        # 循环遍历不同的 SIMD 长度配置，包括未指定、256 和 1
        for simdlen in (None, 256, 1):
            # 在每个循环中，通过修改配置，设置当前 SIMD 长度
            with config.patch({"cpp.simdlen": simdlen}):
                # 循环遍历不同的数据类型，包括 torch.float 和 torch.bfloat16
                for dtype in (torch.float, torch.bfloat16):
                    # 循环遍历不同的张量形状
                    for shape in (
                        (7, 7),
                        (8, 8),
                        (9, 9),
                        (16, 16),
                        (17, 17),
                        (32, 32),
                        (33, 33),
                    ):
                        # 重置 Torch Dynamo 状态
                        torch._dynamo.reset()
                        # 重置度量指标
                        metrics.reset()
                        # 生成指定形状和数据类型的随机张量 x
                        x = torch.randn(shape, dtype=dtype)
                        # 调用共享的测试方法 common，传入内部函数 fn 和张量 x
                        self.common(fn, (x,))
                        # 如果 SIMD 长度不为 1，则检查向量内核计数是否为 2
                        if simdlen != 1:
                            check_metrics_vec_kernel_count(2)
    
    # 使用 Torch 的 Dynamo 配置，禁用整数特化，并定义测试函数以解决切片分散问题 #122291
    @torch._dynamo.config.patch(specialize_int=False)
    def test_slice_scatter_issue122291(self):
        # 定义编译为全图模式的函数 fn，接受输入张量 t、源张量 t_src，以及切片参数 dim、start、end 和 step
        @torch.compile(fullgraph=True)
        def fn(t, t_src, dim, start, end, step):
            return t.slice_scatter(t_src, dim, start, end, step)
    
        # 定义张量形状 shape，包括 16x16、16x2、1、4、10、1
        shape = ((16, 16), (16, 2), 1, 4, 10, 1)
        # 创建一个零填充的输入张量 input_tensor，设备为 CPU
        input_tensor = torch.zeros(shape[0], requires_grad=False, device="cpu")
        # 创建一个全一填充的源张量 src_tensor，设备为 CPU
        src_tensor = torch.ones(shape[1], requires_grad=False, device="cpu")
        # 使用断言检查是否抛出了 BackendCompilerFailed 异常，并验证异常信息是否包含“scatter op”形状错误
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed, r".*shape error in scatter op"
        ):
            # 调用 fn 函数，传入输入张量、源张量以及切片参数
            fn(input_tensor, src_tensor, shape[2], shape[3], shape[4], shape[5])
    def test_horizontal_fusion(self):
        # 定义内部函数fn，接受四个参数a、b、c和idx，并使用torch.index_select函数进行索引选择
        def fn(a, b, c, idx):
            _a = torch.index_select(a, dim=0, index=idx)  # 从张量a中按行索引选择数据
            _b = torch.index_select(b, dim=0, index=idx)  # 从张量b中按行索引选择数据
            _c = torch.index_select(c, dim=0, index=idx)  # 从张量c中按行索引选择数据
            return _a, _b, _c

        # 使用config.patch修改配置项"cpp.max_horizontal_fusion_size"为0，限制水平融合的最大大小
        with config.patch({"cpp.max_horizontal_fusion_size": 0}):
            metrics.reset()  # 重置度量指标
            torch._dynamo.reset()  # 重置torch._dynamo模块
            a = torch.randn(size=(4, 16), dtype=torch.bfloat16)  # 创建大小为(4, 16)的随机张量a
            b = torch.randn(size=(4, 16), dtype=torch.bfloat16)  # 创建大小为(4, 16)的随机张量b
            c = torch.randn(size=(4, 16), dtype=torch.bfloat16)  # 创建大小为(4, 16)的随机张量c
            idx = torch.zeros(size=[4], dtype=torch.int64)  # 创建大小为4的零张量idx
            opt_fn = torch._dynamo.optimize("inductor")(fn)  # 使用torch._dynamo优化fn函数
            opt_fn(a, b, c, idx)  # 对a、b、c和idx调用优化后的函数opt_fn
            self.assertEqual(metrics.generated_kernel_count, 3)  # 断言生成的内核数量为3
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))  # 断言原函数fn与优化后的函数opt_fn结果相同

        # 使用config.patch修改配置项"cpp.max_horizontal_fusion_size"为1，限制水平融合的最大大小
        with config.patch({"cpp.max_horizontal_fusion_size": 1}):
            metrics.reset()  # 重置度量指标
            torch._dynamo.reset()  # 重置torch._dynamo模块
            a = torch.randn(size=(4, 32), dtype=torch.bfloat16)  # 创建大小为(4, 32)的随机张量a
            b = torch.randn(size=(4, 32), dtype=torch.bfloat16)  # 创建大小为(4, 32)的随机张量b
            c = torch.randn(size=(4, 32), dtype=torch.bfloat16)  # 创建大小为(4, 32)的随机张量c
            idx = torch.zeros(size=[4], dtype=torch.int64)  # 创建大小为4的零张量idx
            opt_fn = torch._dynamo.optimize("inductor")(fn)  # 使用torch._dynamo优化fn函数
            opt_fn(a, b, c, idx)  # 对a、b、c和idx调用优化后的函数opt_fn
            self.assertEqual(metrics.generated_kernel_count, 3)  # 断言生成的内核数量为3
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))  # 断言原函数fn与优化后的函数opt_fn结果相同

        # 使用config.patch修改配置项"cpp.max_horizontal_fusion_size"为2，限制水平融合的最大大小
        with config.patch({"cpp.max_horizontal_fusion_size": 2}):
            metrics.reset()  # 重置度量指标
            torch._dynamo.reset()  # 重置torch._dynamo模块
            a = torch.randn(size=(4, 64), dtype=torch.bfloat16)  # 创建大小为(4, 64)的随机张量a
            b = torch.randn(size=(4, 64), dtype=torch.bfloat16)  # 创建大小为(4, 64)的随机张量b
            c = torch.randn(size=(4, 64), dtype=torch.bfloat16)  # 创建大小为(4, 64)的随机张量c
            idx = torch.zeros(size=[4], dtype=torch.int64)  # 创建大小为4的零张量idx
            opt_fn = torch._dynamo.optimize("inductor")(fn)  # 使用torch._dynamo优化fn函数
            opt_fn(a, b, c, idx)  # 对a、b、c和idx调用优化后的函数opt_fn
            print(metrics.generated_kernel_count)  # 打印生成的内核数量
            self.assertEqual(metrics.generated_kernel_count, 2)  # 断言生成的内核数量为2
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))  # 断言原函数fn与优化后的函数opt_fn结果相同

        # 使用config.patch修改配置项"cpp.max_horizontal_fusion_size"为3，限制水平融合的最大大小
        with config.patch({"cpp.max_horizontal_fusion_size": 3}):
            metrics.reset()  # 重置度量指标
            torch._dynamo.reset()  # 重置torch._dynamo模块
            a = torch.randn(size=(4, 128), dtype=torch.bfloat16)  # 创建大小为(4, 128)的随机张量a
            b = torch.randn(size=(4, 128), dtype=torch.bfloat16)  # 创建大小为(4, 128)的随机张量b
            c = torch.randn(size=(4, 128), dtype=torch.bfloat16)  # 创建大小为(4, 128)的随机张量c
            idx = torch.zeros(size=[4], dtype=torch.int64)  # 创建大小为4的零张量idx
            opt_fn = torch._dynamo.optimize("inductor")(fn)  # 使用torch._dynamo优化fn函数
            opt_fn(a, b, c, idx)  # 对a、b、c和idx调用优化后的函数opt_fn
            self.assertEqual(metrics.generated_kernel_count, 1)  # 断言生成的内核数量为1
            self.assertTrue(same(fn(a, b, c, idx), opt_fn(a, b, c, idx)))  # 断言原函数fn与优化后的函数opt_fn结果相同
    def test_lowp_fp_neg_abs(self):
        # 定义一个函数 fn(x)，实现对输入张量 x 执行 neg() 和 abs() 操作
        def fn(x):
            return x.neg().abs()

        # 遍历低精度浮点数数据类型列表 _lowp_fp_dtypes
        for dtype in _lowp_fp_dtypes:
            # 重置度量指标 metrics
            metrics.reset()
            # 生成一个随机张量 x，形状为 (100, 100)，并转换为指定数据类型 dtype
            x = torch.randn(100, 100).to(dtype)
            # 使用动态优化器对函数 fn 应用 "inductor" 优化
            opt_fn = torch._dynamo.optimize("inductor")(fn)
            # 断言优化后的函数 opt_fn(x) 和原函数 fn(x) 的输出相同
            self.assertTrue(same(fn(x), opt_fn(x)))
            # 断言 C++ 到数据类型计数为 0
            assert metrics.cpp_to_dtype_count == 0
            # 检查向量内核计数是否为 1
            check_metrics_vec_kernel_count(1)

    def test_transpose_non_contiguous(self):
        # 定义一个函数 fn(a)，实现特定的张量操作
        def fn(a):
            # 使用 torch.ops.aten.as_strided.default 创建新的张量 as_strided
            as_strided = torch.ops.aten.as_strided.default(
                a, [1, 384, 2, 20, 12], [153600, 1, 61440, 384, 7680]
            )
            # 使用 torch.ops.aten.as_strided.default 创建新的张量 as_strided_1
            as_strided_1 = torch.ops.aten.as_strided.default(
                as_strided,
                [1, 384, 2, 2, 12, 12],
                [153600, 1, 61440, 3072, 7680, 384],
            )
            # 使用 torch.ops.aten.clone.default 创建新的张量 clone_1
            clone_1 = torch.ops.aten.clone.default(
                as_strided_1, memory_format=torch.contiguous_format
            )
            # 使用 torch.ops.aten._unsafe_view.default 创建新的张量 _unsafe_view_1
            _unsafe_view_1 = torch.ops.aten._unsafe_view.default(
                clone_1, [8, 48, 4, 144]
            )
            # 使用 torch.ops.aten.permute.default 创建新的张量 permute_2
            permute_2 = torch.ops.aten.permute.default(_unsafe_view_1, [0, 2, 3, 1])
            # 使用 torch.ops.aten.split_with_sizes.default 进行张量分割操作
            split_with_sizes = torch.ops.aten.split_with_sizes.default(
                permute_2, [16, 32], -1
            )
            # 获取分割后的第一个张量 getitem
            getitem = split_with_sizes[0]
            # 获取分割后的第二个张量 getitem_1
            getitem_1 = split_with_sizes[1]
            # 使用 torch.ops.aten.permute.default 创建新的张量 permute_3
            permute_3 = torch.ops.aten.permute.default(getitem, [0, 1, 3, 2])
            # 使用 torch.ops.aten.expand.default 创建新的张量 expand_1
            expand_1 = torch.ops.aten.expand.default(permute_3, [8, 4, 16, 144])
            # 使用 torch.ops.aten.clone.default 创建新的张量 clone_3
            clone_3 = torch.ops.aten.clone.default(
                expand_1, memory_format=torch.contiguous_format
            )
            # 返回张量 clone_3
            return clone_3

        # 重置度量指标 metrics
        metrics.reset()
        # 生成一个随机张量 x，形状为 (1, 384, 20, 20)，并使用通道优先内存格式
        x = torch.randn(1, 384, 20, 20).to(memory_format=torch.channels_last)
        # 调用 self.common 函数，对 fn(x) 进行通用测试
        self.common(fn, (x,))
        # 检查向量内核计数是否为 1
        check_metrics_vec_kernel_count(1)

    def test_non_contiguous_index_with_constant_stride(self):
        # 定义一个函数 fn(x)，实现对输入张量 x 的特定索引和操作
        def fn(x):
            # 使用切片操作 x[:, :, :, ::2] 创建新的张量 x1
            x1 = x[:, :, :, ::2]
            # 使用切片操作 x[:, :, :, 1::2] 创建新的张量 x2
            x2 = x[:, :, :, 1::2]
            # 使用 torch.stack 在最后一个维度上堆叠张量 -x2 和 x1，创建新的张量 x
            x = torch.stack((-x2, x1), dim=-1)
            # 将张量 x 沿着倒数第二维度展平
            return x.flatten(-2)

        # 重置度量指标 metrics
        metrics.reset()
        # 生成一个随机张量 x，形状为 (1, 32, 16, 68)
        x = torch.randn(1, 32, 16, 68)
        # 使用动态优化器对函数 fn 应用 "inductor" 优化
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        # 断言优化后的函数 opt_fn(x) 和原函数 fn(x) 的输出相同
        self.assertTrue(same(fn(x), opt_fn(x)))
        # 使用 FileCheck 检查生成的 C++ 代码中 "cpp_fused" 出现次数是否为 2
        FileCheck().check_count("cpp_fused", 2, exactly=True).run(code)

    def test_invalid_index_of_empty_tensor(self):
        # 定义一个函数 fn(a)，对输入张量 a 执行索引操作
        def fn(a):
            # 使用索引操作 a[[0]] 创建新的张量 b
            b = a[[0]]
            # 返回张量 b
            return b

        # 生成一个空的张量 a
        a = torch.tensor([])
        # 使用 self.assertRaises 检查运行时是否抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 编译函数 fn(a)，并立即执行
            torch.compile(fn)(a)

    # 使用 torch.no_grad 和 torch._inductor.config.patch 对下一个测试函数应用装饰器
    @torch.no_grad()
    @torch._inductor.config.patch(freezing=True)
    def test_issue122380(self):
        # 定义测试函数，用于验证处理张量的函数
        def func(x):
            # 解绑张量 x 的所有维度，返回一个元组
            t1 = torch.unbind(x)
            # 在维度 1 上堆叠张量 t1，返回结果张量
            t2 = torch.stack(t1, dim=1)
            # 对张量 t2 中的元素逐元素计算双曲正切函数
            t3 = torch.tanh(t2)
            # 返回计算结果张量
            return t3

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 使用 torch.compile 编译 func 函数并执行，与直接调用 func(x) 的结果进行比较
        self.assertEqual(torch.compile(func)(x), func(x))

    def test_ir_node_str(self):
        # 定义一个装饰器函数 fn，输入为 torch.Tensor 类型，输出为 torch.Tensor 类型
        @torch.compile
        def fn(x: torch.Tensor) -> torch.Tensor:
            # 返回输入张量 x 的正弦值和在维度 1 上的 softmax 函数值
            return x.sin(), torch.nn.Softmax(dim=1)(x.cos())

        # 定义一个替代函数 run_node_alt，用于运行节点并记录返回值的字符串表示
        def run_node_alt(*args, **kwargs):
            rv = run_node(*args, **kwargs)
            # 将节点运行的返回值 rv 转换成字符串并添加到列表 strings 中
            strings.append(str(rv))
            return rv

        # 初始化空列表 strings
        strings = []
        # 将原始运行函数 GraphLowering.run_node 赋值给变量 run_node
        run_node = GraphLowering.run_node
        # 使用 patch.object 将 GraphLowering 类的 run_node 方法替换为 run_node_alt
        with patch.object(GraphLowering, "run_node", run_node_alt):
            # 执行函数 fn，并传入一个形状为 [8, 128] 的随机张量作为输入
            fn(torch.randn([8, 128]))
        # 断言列表 strings 的长度大于 3
        self.assertGreater(len(strings), 3)

    def test_vertical_sum_cpu_only(self):
        # 定义函数 fn1，对输入张量 a 沿着维度 0 求和
        def fn1(a):
            return a.sum(dim=0)

        # 定义函数 fn2，对输入张量 a 沿着维度 1 求和
        def fn2(a):
            return a.sum(dim=1)

        # 重置指标 metrics
        metrics.reset()
        # 创建一个形状为 (100, 100) 的随机张量 x，并调用 self.common 进行验证
        x = torch.randn(100, 100)
        self.common(fn1, (x,))
        # 检查向量核计数指标是否为 1
        check_metrics_vec_kernel_count(1)

        # 重置指标 metrics
        metrics.reset()
        # 创建一个形状为 (100, 100, 100) 的随机张量 x，并调用 self.common 进行验证
        x = torch.randn(100, 100, 100)
        self.common(fn2, (x,))
        # 检查向量核计数指标是否为 1
        check_metrics_vec_kernel_count(1)

    def test_transpose_vertical_sum_cpu_only(self):
        # 定义函数 fn，对输入张量 a 和 b 对应元素相乘后，沿着维度 1 求和
        def fn(a, b):
            c = a * b
            return c.sum(dim=1)

        # 重置指标 metrics
        metrics.reset()
        # 创建形状为 (100, 50, 50) 的随机张量 x 和 y，y 在维度 1 和 2 进行转置
        x = torch.randn(100, 50, 50)
        y = torch.randn(100, 50, 50).transpose(1, 2)
        # 调用 self.common 运行函数 fn，并传入 x 和 y 作为参数
        self.common(fn, (x, y))
        # 检查向量核计数指标是否为 2
        check_metrics_vec_kernel_count(2)

    def test_transpose_mxn_16_16_bf16_fp16(self):
        # 定义函数 fn，对输入张量 a 和 b 对应元素相乘后，沿着维度 1 求和
        def fn(a, b):
            c = a * b
            return c.sum(dim=1)

        # 针对 torch.bfloat16 和 torch.float16 数据类型进行迭代
        for dtype in [torch.bfloat16, torch.float16]:
            # 重置指标 metrics
            metrics.reset()
            # 创建形状为 (100, 50, 50) 的随机张量 x 和 y，并将数据类型转换为 dtype，y 在维度 1 和 2 进行转置
            x = torch.randn(100, 50, 50).to(dtype)
            y = torch.randn(100, 50, 50).to(dtype).transpose(1, 2)
            # 调用 self.common 运行函数 fn，并传入 x 和 y 作为参数
            self.common(fn, (x, y))
            # 检查向量核计数指标是否为 2
            check_metrics_vec_kernel_count(2)

    def test_transpose_mxn_32_32_bf16_fp16(self):
        # 定义函数 fn，对输入张量 a 执行维度 0 和 2 的转置操作，并返回转置后的张量
        def fn(a):
            return a.permute(0, 2, 1).contiguous()

        # 针对 torch.bfloat16 和 torch.float16 数据类型进行迭代
        for dtype in [torch.bfloat16, torch.float16]:
            # 重置指标 metrics
            metrics.reset()
            # 创建形状为 (2, 9216, 9216) 的随机张量 x，并将数据类型转换为 dtype
            x = torch.randn(2, 9216, 9216).to(dtype)
            # 调用 self.common 运行函数 fn，并传入 x 作为参数
            self.common(fn, (x,))
            # 检查向量核计数指标是否为 2
            check_metrics_vec_kernel_count(2)

    def test_transpose_sum2d_cpu_only(self):
        # 定义函数 fn，对输入张量 a 和 b 对应元素相乘后，对结果张量进行求和
        def fn(a, b):
            c = a * b
            return c.sum()

        # 重置指标 metrics
        metrics.reset()
        # 创建形状为 (50, 50) 的随机张量 x 和 y，y 在维度 0 和 1 进行转置
        x = torch.randn(50, 50)
        y = torch.randn(50, 50).transpose(0, 1)
        # 调用 self.common 运行函数 fn，并传入 x 和 y 作为参数
        self.common(fn, (x, y))
        # 检查向量核计数指标是否为 2
        check_metrics_vec_kernel_count(2)

    def test_transpose_sum_outer(self):
        # 定义函数 fn，对输入张量 a 在维度 2 和 3 进行转置后，沿着维度 1 求和并返回连续存储的张量
        def fn(a):
            return a.transpose(2, 3).sum(dim=1).contiguous()

        # 重置指标 metrics
        metrics.reset()
        # 创建形状为 (10, 50, 50, 50) 的随机张量 x
        x = torch.randn(10, 50, 50, 50)
        # 调用 self.common 运行函数 fn，并传入 x 作为参数
        self.common(fn, (x,))
        # 检查向量核计数指标是否为 1
        check_metrics_vec_kernel_count(1)
    def test_to_dtype_bool_float(self):
        # 测试将布尔类型转换为浮点类型的函数
        def f(a):
            # 使用 torch.where 实现条件选择：
            # 如果条件为真，则返回 torch.zeros_like(a)，否则返回 torch.ones_like(a) * 2
            return torch.where(
                torch.ones_like(a).to(torch.bool),  # 创建与 a 相同形状的全 1 张量，并转换为布尔类型
                torch.zeros_like(a),  # 创建与 a 相同形状的全 0 张量
                torch.ones_like(a) * 2,  # 创建与 a 相同形状的全 2 张量
            )

        self.common(f, (torch.ones(16),))  # 调用公共测试方法，传入函数 f 和参数

    def test_to_dtype_float_bool(self):
        # 测试将浮点类型转换为布尔类型的函数
        def f(a):
            # 根据条件 a >= 0，将 a 乘以相应的浮点数张量
            a = a * torch.tensor(a >= 0, dtype=torch.float32)
            return a

        x = torch.rand(16)  # 创建一个形状为 (16,) 的随机张量
        self.common(f, (x,))  # 调用公共测试方法，传入函数 f 和参数

    def test_constant_store(self):
        # 测试将特定位置的元素替换为负无穷大的函数
        def f(a):
            a[0, [3, 3]] = -float("inf")  # 将 a 的第一行第四列和第一行第五列的元素替换为负无穷大
            return a

        x = torch.rand(4, 5)  # 创建一个形状为 (4, 5) 的随机张量
        self.common(f, (x,))  # 调用公共测试方法，传入函数 f 和参数

    def test_to_channels_last_lowp_fp(self):
        # 测试将张量转换为 channels_last 内存格式的函数
        def f(a):
            return a.to(memory_format=torch.channels_last)  # 将输入张量转换为 channels_last 内存格式

        for dtype in _lowp_fp_dtypes:  # 遍历指定的低精度浮点数数据类型
            x = torch.rand(2, 3, 14, 14).to(dtype)  # 创建指定数据类型和形状的随机张量
            self.common(f, (x,))  # 调用公共测试方法，传入函数 f 和参数

    def test_broadcast_mul_lowp_fp(self):
        # 测试低精度浮点数张量的广播乘法函数
        def f(a, b):
            return a * b  # 对两个输入张量进行逐元素乘法操作

        for dtype in _lowp_fp_dtypes:  # 遍历指定的低精度浮点数数据类型
            a = torch.randn(2, 16, 16).to(dtype)  # 创建指定数据类型和形状的随机张量 a
            b = torch.randn(2, 1, 1).to(dtype)  # 创建指定数据类型和形状的随机张量 b
            self.common(f, (a, b))  # 调用公共测试方法，传入函数 f 和参数

    def test_linear_buffer_reuse(self):
        # 测试线性层缓冲重用的函数
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(16, 16)  # 创建输入大小为 16 输出大小为 16 的线性层
                self.tanh = torch.nn.Tanh()  # 创建 Tanh 激活函数层
                self.linear2 = torch.nn.Linear(16, 16)  # 创建输入大小为 16 输出大小为 16 的线性层

            def forward(self, x):
                x = self.linear1(x)  # 使用第一个线性层处理输入张量 x
                x = self.tanh(x)  # 使用 Tanh 激活函数处理 x
                x = self.linear2(x)  # 使用第二个线性层处理 x
                return x

        mod = M().eval()  # 创建并评估模型
        v = torch.randn(1, 16)  # 创建形状为 (1, 16) 的随机张量

        with torch.no_grad():
            def compile_fx_wrapper(model_, example_inputs_):
                return compile_fx(model_, example_inputs_)

            def run(*ex, **kwargs):
                return mod(*ex, **kwargs)

            run = torch._dynamo.optimize(compile_fx_wrapper)(run)  # 优化运行时函数
            _, code = run_and_get_cpp_code(run, v)  # 获取运行时函数的 C++ 代码
            self.assertFalse("= as_strided(" in code)  # 确保 C++ 代码中不包含 "= as_strided("
            self.assertEqual(run(*v), mod(*v))  # 断言优化后的运行时函数和原模型在输入 v 下的输出相等

    def test_invalid_dropout_args(self):
        # 测试 dropout 函数不合法参数的处理
        class MyModel(torch.nn.Module):
            def forward(self, x):
                x = x * 2  # 输入张量乘以 2
                x = torch.nn.functional.dropout(x, p=0.5)  # 使用 dropout 函数丢弃输入张量的一部分元素
                x = torch.relu(x)  # 使用 ReLU 激活函数处理 x
                return x

        example_inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 创建示例输入张量

        func = MyModel()  # 创建模型实例
        jit_func = torch.compile(func)  # 编译模型

        self.assertRaises(RuntimeError, lambda: func(example_inputs))  # 断言调用未编译的模型会引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: jit_func(example_inputs))  # 断言调用已编译的模型会引发 RuntimeError
    def test_nn_param_assign(self):
        # 定义一个测试方法，用于测试神经网络参数赋值的情况

        # 定义一个简单的神经网络模型
        class Model2(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                # 定义批归一化层，输入通道数为5
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                # 初始化卷积层的权重，形状为(5, 3, 3, 3)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                # 初始化卷积层的偏置，形状为(5,)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                # 设置卷积层的权重为可训练的参数
                self.conv.weight = nn.Parameter(self.conv_weight)
                # 设置卷积层的偏置为不可训练的参数
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                # 将卷积层设置为评估模式
                self.conv.eval()
                # 前向传播过程
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        # 生成一个输入张量，形状为(1, 3, 10, 10)
        input_tensor = torch.randn(1, 3, 10, 10)
        # 创建模型实例，并将其部署到CPU上
        func = Model2().to("cpu")

        # 使用torch.no_grad()上下文管理器，确保在推断模式下运行
        with torch.no_grad():
            # 将模型设置为推断模式
            func.train(False)
            # 计算未编译的模型输出v1
            v1 = func(input_tensor)
            # 编译模型的图形表示，全图编译模式
            jit_func = torch.compile(func, fullgraph=True)
            # 计算编译后模型的输出v2
            v2 = jit_func(input_tensor)
            # 断言v1与v2相等
            self.assertEqual(v1, v2)

    def test_nn_param_assign_wrapped(self):
        # 定义一个测试方法，用于测试包装后的神经网络参数赋值的情况

        # 定义一个简单的神经网络模型
        class Model2(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义卷积层，输入通道数为3，输出通道数为5，卷积核大小为3x3
                self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
                # 定义批归一化层，输入通道数为5
                self.batchnorm = nn.BatchNorm2d(num_features=5)
                # 初始化卷积层的权重，形状为(5, 3, 3, 3)
                self.conv_weight = torch.randn(5, 3, 3, 3)
                # 初始化卷积层的偏置，形状为(5,)
                self.conv_bias = torch.randn(5)

            def forward(self, x):
                # 设置卷积层的权重为可训练的参数
                self.conv.weight = nn.Parameter(self.conv_weight)
                # 设置卷积层的偏置为不可训练的参数
                self.conv.bias = nn.Parameter(self.conv_bias, requires_grad=False)
                # 将卷积层设置为评估模式
                self.conv.eval()
                # 前向传播过程
                x = self.conv(x)
                x = self.batchnorm(x)
                x = F.relu(x)
                return x

        # 生成一个输入张量，形状为(1, 3, 10, 10)
        input_tensor = torch.randn(1, 3, 10, 10)
        # 创建模型实例，并将其部署到CPU上
        func = Model2().to("cpu")

        # 使用functools.wraps(func)包装模型，生成包装后的函数wrapper
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # 使用torch.no_grad()上下文管理器，确保在推断模式下运行
        with torch.no_grad():
            # 将模型设置为推断模式
            func.train(False)
            # 计算未编译的模型输出v1
            v1 = func(input_tensor)
            # 编译模型的图形表示，全图编译模式
            jit_func = torch.compile(wrapper, fullgraph=True)
            # 计算编译后模型的输出v2
            v2 = jit_func(input_tensor)
            # 断言v1与v2相等
            self.assertEqual(v1, v2)

    @config.patch(inplace_buffers=True)
    def test_in_out_buffer(self):
        # 定义一个测试方法，用于测试输入输出缓冲区的情况

        # 定义一个简单的函数，计算两个张量的乘积的转置，并除以8
        def fn(x, y):
            z = torch.matmul(x, y.transpose(-1, -2)) / 8.0
            return z

        # 生成两个输入张量列表，形状分别为(1, 2, 8, 4)
        inps = [torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4)]
        # 对函数fn进行优化，使用torch._dynamo.optimize("inductor")(fn)
        fn_opt = torch._dynamo.optimize("inductor")(fn)
        # 运行并获取优化后的函数fn_opt的C++代码和输出结果
        _, code = run_and_get_cpp_code(fn_opt, *inps)
        # 断言代码中包含字符串"in_out_ptr"
        self.assertTrue("in_out_ptr" in code)
        # 断言优化后的函数fn_opt在给定输入上与原始函数fn的输出相等
        self.assertEqual(fn_opt(*inps), fn(*inps))
    def test_eliminate_meaningless_copy(self):
        # 定义一个函数 fn，接受两个参数 x1 和 x2
        def fn(x1, x2):
            # 使用 torch.ops.aten.permute.default 对 x2 进行维度重排操作
            permute = torch.ops.aten.permute.default(x2, [0, 2, 1, 3])
            # 使用 torch.ops.aten.clone.default 复制 permute，并保持内存格式连续
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            # 使用 torch.ops.aten.view.default 对 clone 进行维度变换，reshape 成 [1024, -1, 32]
            view = torch.ops.aten.view.default(clone, [1024, -1, 32])
            # 使用 torch.ops.aten.bmm.default 对 view 和 x1 进行批量矩阵乘法
            bmm = torch.ops.aten.bmm.default(view, x1)
            # 再次使用 torch.ops.aten.permute.default 对 view 进行维度重排
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            # 返回两个结果：bmm 和 permute
            return (bmm, permute)

        # 重置度量指标
        metrics.reset()
        # 调用 self.common 方法执行 fn 函数
        self.common(
            fn,
            [
                rand_strided(
                    (1024, 32, 128), (4096, 1, 32), device="cpu", dtype=torch.float32
                ),
                rand_strided(
                    (64, 128, 16, 32),
                    (65536, 512, 32, 1),
                    device="cpu",
                    dtype=torch.float32,
                ),
            ],
        )
        # 断言生成的内核数量为 1
        self.assertEqual(metrics.generated_kernel_count, 1)

    def test_attention_size_mismatch(self):
        # 定义一个名为 Attention 的子类，继承自 torch.nn.Module
        class Attention(torch.nn.Module):
            # 初始化方法，接受 hidden_size 和 num_heads 两个参数
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                # 初始化隐藏大小和头数
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                # 计算每个头的大小
                self.head_size = hidden_size // num_heads
                # 定义三个线性变换层：query、key、value
                self.query = torch.nn.Linear(hidden_size, hidden_size)
                self.key = torch.nn.Linear(hidden_size, hidden_size)
                self.value = torch.nn.Linear(hidden_size, hidden_size)
                # 初始化缩放因子为头大小的倒数，不需要梯度
                self.inv_scale = torch.nn.Parameter(
                    torch.Tensor([1 / self.head_size**0.5]), requires_grad=False
                )

            # 前向传播方法，接受输入 x
            def forward(self, x):
                # 对输入 x 进行 query、key、value 的线性变换
                query = self.query(x)
                key = self.key(x)
                value = self.value(x)
                # 获取 query 的形状信息：batch_size、seq_len、hidden_size
                (batch_size, seq_len, hidden_size) = query.size()
                # 将 query、key、value 分别 reshape 成 [batch_size, seq_len, num_heads, head_size] 并进行维度重排
                query = query.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 1, 3)
                key = key.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 3, 1)
                value = value.view(
                    batch_size, seq_len, self.num_heads, self.head_size
                ).permute(0, 2, 1, 3)
                # 计算注意力权重：query 和 key 的矩阵乘积，乘以缩放因子，再进行 softmax 操作
                attention_weights = (
                    torch.matmul(query, key).mul(self.inv_scale).softmax(dim=-1)
                )
                # 计算输出：注意力权重与 value 的矩阵乘积
                output = torch.matmul(attention_weights, value)
                # 返回输出结果
                return output

        # 设定随机种子
        torch.manual_seed(123)
        # 初始化隐藏大小、头数、序列长度、批量大小
        hidden_size = 16
        num_heads = 1
        seq_len = 4
        batch_size = 1
        # 生成随机输入张量 x
        x = torch.randn(batch_size, seq_len, hidden_size)

        # 创建 Attention 类的实例 func，并将其部署到 CPU 上
        func = Attention(hidden_size, num_heads).to("cpu")

        # 关闭梯度计算上下文
        with torch.no_grad():
            # 计算 func 对 x 的前向传播结果 res1
            res1 = func(x)
            # 编译 func 以加速执行，并计算其对 x 的前向传播结果 res2
            jit_func = torch.compile(func)
            res2 = jit_func(x)
        # 断言 res1 和 res2 相等
        self.assertEqual(res1, res2)
    # 定义测试函数 test_scalar_mul_bfloat16，用于测试在 bfloat16 数据类型下的张量乘法操作
    def test_scalar_mul_bfloat16(self):
        # 内部函数 f(x)，用于将输入张量 x 与常数 1.7015043497085571 相乘
        def f(x):
            return torch.ops.aten.mul.Tensor(x, 1.7015043497085571)

        # 重置度量指标
        metrics.reset()
        # 生成一个形状为 (4, 5) 的随机张量 x，数据类型为 torch.bfloat16
        x = torch.randn(4, 5, dtype=torch.bfloat16)
        # 调用共享的测试方法 self.common，传入 f 函数和其参数 x
        self.common(f, (x,))
        # 检查生成的向量化内核计数
        check_metrics_vec_kernel_count(1)

    # 定义测试函数 test_bf16_zeros，用于测试在 torch.bfloat16 数据类型下创建全零张量的操作
    def test_bf16_zeros(self):
        # 内部函数 fn，创建一个形状为 (1, 1, 32) 的全零张量 x，数据类型为 torch.bfloat16
        def fn():
            x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
            return x

        # 调用共享的测试方法 self.common，传入 fn 函数
        self.common(fn, ())

    # 定义测试函数 test_select_tiliing_with_index_expr，测试使用索引表达式进行选取和平铺操作
    def test_select_tiliing_with_index_expr(self):
        # 内部函数 fn，对输入的张量 x 进行视图重塑为 [8, 8, 8, 3136] 的形状
        def fn(x, y):
            x = torch.ops.aten.view.default(x, [8, 8, 8, 3136])
            # 对 x 进行轴置换，调整为 [8, 8, 3136, 8] 的形状
            x = torch.ops.aten.permute.default(x, [0, 1, 3, 2])
            # 将张量 y 与 x 进行元素级乘法
            y = torch.ops.aten.mul.Tensor(y, x)
            # 对 y 进行常数填充，使用 [0, 0, 1, 0, 0, 0] 的填充方式，填充值为 0.0
            return torch.ops.aten.constant_pad_nd.default(y, [0, 0, 1, 0, 0, 0], 0.0)

        # 生成随机张量 x 和 y，分别形状为 (8, 64, 56, 56) 和 (8, 8, 3136, 8)
        x = torch.randn(8, 64, 56, 56)
        y = torch.randn(8, 8, 3136, 8)
        # 调用共享的测试方法 self.common，传入 fn 函数和其参数 x, y
        self.common(fn, (x, y))

    # 使用 unittest.skipIf 装饰器，仅在 MKLDNN 可用时才执行该测试
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKLDNN is not enabled")
    # 使用 patch 装饰器，模拟 torch.cuda.is_available 函数返回 False，同时启用冻结模式
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    # 定义测试函数 test_linear_with_no_default_contiguous_input，测试线性层输入非默认连续情况
    def test_linear_with_no_default_contiguous_input(self):
        # 支持的数据类型列表，包括 torch.float32
        dtypes = [
            torch.float32,
        ]
        # 如果支持 MKLDNN 的 bfloat16，则添加 torch.bfloat16 到数据类型列表
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            dtypes.append(torch.bfloat16)
        # 如果支持 MKLDNN 的 float16，则添加 torch.float16 到数据类型列表
        if torch.ops.mkldnn._is_mkldnn_fp16_supported():
            dtypes.append(torch.float16)
        # 构建包含单个线性层的顺序模型，并设置为评估模式
        mod = torch.nn.Sequential(torch.nn.Linear(16, 16)).eval()
        # 生成一个形状为 (1, 16, 1, 1) 的随机张量 temp
        temp = torch.randn(1, 16, 1, 1)
        # 使用 as_strided 方法创建张量 v，形状为 (1, 16)，步幅为 [0, 1]，偏移为 0
        v = torch.as_strided(temp, [1, 16], [0, 1], 0)
        # 断言 v 是连续的张量
        self.assertTrue(v.is_contiguous())
        # 遍历 dtypes 列表中的每种数据类型
        for dtype in dtypes:
            with torch.no_grad():
                # 调用共享的测试方法 self.common，传入转换为当前数据类型的 mod 和 v
                self.common(
                    mod.to(dtype),
                    (v.to(dtype),),
                )

    # 使用 patch 装饰器，模拟 torch.cuda.is_available 函数返回 False，同时启用冻结模式
    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    # 定义测试函数 test_linear_with_reshape，测试带有重塑操作的线性层
    def test_linear_with_reshape(self):
        # 定义包含线性层的简单模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义不带偏置的线性层
                self.linear = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                # 在前向传播中应用线性层，并将结果重塑为 (4, 4, 4) 的形状
                x = self.linear(x)
                return x.view(4, 4, 4)

        # 创建 M 类的实例 mod，并设置为评估模式
        mod = M().eval()
        # 生成一个形状为 (4, 16) 的随机张量 v
        v = torch.randn(4, 16)
        with torch.no_grad():
            # 重置动态计算图和度量指标
            torch._dynamo.reset()
            metrics.reset()
            # 调用共享的测试方法 self.common，传入 mod 和 v 作为参数
            self.common(
                mod,
                (v,),
            )
            # 断言生成的内核计数为 0
            assert metrics.generated_kernel_count == 0

    # 使用 config.patch 装饰器，启用隐式回退模式
    @config.patch(implicit_fallbacks=True)
    # 测试函数，用于测试 torch.normal 函数在不同数据类型下的行为
    def test_aten_normal_dtype(self):
        # 遍历数据类型列表 [torch.float64, torch.float16, None]
        for dtype in [torch.float64, torch.float16, None]:

            # 定义一个返回 torch.normal 结果的函数
            def fn():
                return torch.normal(2, 3, (10, 10), dtype=dtype, device="cpu")

            # 编译 fn 函数并使用 aot_eager_decomp_partition 后端，检查结果的数据类型
            self.assertEqual(
                torch.compile(fn, backend="aot_eager_decomp_partition")().dtype,
                dtype if dtype else torch.float32,
            )
            # 编译 fn 函数并使用 inductor 后端，检查结果的数据类型
            self.assertEqual(
                torch.compile(fn, backend="inductor")().dtype,
                dtype if dtype else torch.float32,
            )

    # 测试 torch.nn.GroupNorm 在向量输入下的行为
    def test_group_norm_vec(self):
        # 定义一个简单的 Module 包含 GroupNorm 层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.group_norm = torch.nn.GroupNorm(32, 32)

            # 前向传播函数，将输入 x 经过 GroupNorm 层处理后返回结果
            def forward(self, x):
                return self.group_norm(x)

        # 重置性能指标
        metrics.reset()
        # 实例化 M 并设置为评估模式
        mod = M().eval()
        # 创建输入张量 x，大小为 (2, 32, 32, 32)
        x = torch.randn(2, 32, 32, 32)
        # 在无梯度更新的上下文中执行
        with torch.no_grad():
            # 使用 self.common 方法评估 mod 在输入 x 上的性能
            self.common(mod, (x,))
            # 检查生成的内核数目是否为 2（一个用于计算均值和方差，另一个用于最终结果）
            check_metrics_vec_kernel_count(2)

    # 测试 torch.div 在向量输入下的行为
    def test_int_div_vec(self):
        # 定义一个函数 fn，执行 torch.div 操作
        def fn(x, y, mode):
            return torch.div(x, y, rounding_mode=mode)

        # 创建输入张量 x 和 y，大小为 (32, 32)，元素在 [1, 100) 之间
        x = torch.randint(1, 100, (32, 32))
        y = torch.randint(1, 100, (32, 32))
        # 遍历 mode 列表 [None, "trunc", "floor"]
        for mode in [None, "trunc", "floor"]:
            # 在无梯度更新的上下文中执行
            with torch.no_grad():
                # 重置性能指标
                metrics.reset()
                # 使用 self.common 方法评估 fn 在输入 x, y, mode 上的性能
                self.common(fn, (x, y, mode))
                # 检查生成的内核数目是否为 1
                check_metrics_vec_kernel_count(1)

    # 测试 torch.add 在 uint8 数据类型上的行为
    def test_uint8_add(self):
        # 定义一个函数 fn，执行 torch.add 操作并将结果取负数后转换为 torch.int32 类型
        def fn(x, y):
            return torch.add(x, y).neg().to(torch.int32)

        # 创建输入张量 x 和 y，大小为 (3, 3)，数据类型为 torch.uint8，元素在 [0, 255] 之间
        x = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        # 使用 self.common 方法评估 fn 在输入 x, y 上的性能
        self.common(fn, (x, y))

    # 测试 torch.sub 在 uint8 数据类型上的行为
    def test_uint8_sub(self):
        # 定义一个函数 fn，执行 torch.sub 操作并将结果取负数后转换为 torch.int32 类型
        def fn(x, y):
            return torch.sub(x, y).neg().to(torch.int32)

        # 创建输入张量 x 和 y，大小为 (3, 3)，数据类型为 torch.uint8，元素在 [0, 255] 之间
        x = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        y = torch.randint(0, 255, (3, 3), dtype=torch.uint8)
        # 使用 self.common 方法评估 fn 在输入 x, y 上的性能
        self.common(fn, (x, y))

    # 测试包含非连续维度的输入在计算中的行为
    def test_non_contiguous_reduction_store(self):
        # 定义一个简单的 Module 包含 Conv2d 层
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(39, 1, kernel_size=(1, 17), stride=(2, 2))

            # 前向传播函数，输入 x 先沿第三维度求最大值，然后经过 Conv2d 层处理后返回结果
            def forward(self, x):
                return self.conv(x.max(3).values)

        # 实例化 M
        m = M()
        # 创建输入张量 x，大小为 (1, 39, 1, 18, 17)
        x = torch.randn(1, 39, 1, 18, 17)
        # 使用 self.common 方法评估 m 在输入 x 上的性能
        self.common(m, (x,))
    def test_embedding_vec(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，设置一个嵌入层 emb，维度为 (64, 128)
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(64, 128)

            # 前向传播函数，接收 idx 和 x 作为输入，返回嵌入后的结果加上 x
            def forward(self, idx, x):
                return self.emb(idx) + x

        # 生成一个随机整数张量 idx，范围在 [0, 64)，形状为 (4, 32)
        idx = torch.randint(0, 64, (4, 32))
        # 生成一个标准正态分布的张量 x，形状为 (4, 32, 128)
        x = torch.randn(4, 32, 128)
        # 实例化 M 类并设为评估模式
        m = M().eval()
        # 进入无梯度计算环境
        with torch.no_grad():
            # 重置度量指标
            metrics.reset()
            # 调用共有方法 self.common，传入模型 m 和输入数据 (idx, x)
            self.common(m, (idx, x))
            # 检查向量化内核数量是否为 1
            check_metrics_vec_kernel_count(1)

    def test_embedding_vec_bf16(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数，设置一个嵌入层 emb，维度为 (64, 128)
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(64, 128)

            # 前向传播函数，接收 idx 和 x 作为输入，返回嵌入结果
            def forward(self, idx, x):
                return self.emb(idx)

        # 生成一个随机整数张量 idx，范围在 [0, 64)，形状为 (4, 32)
        idx = torch.randint(0, 64, (4, 32))
        # 生成一个标准正态分布的张量 x，形状为 (4, 32, 128)，并转换为 bf16 类型
        x = torch.randn(4, 32, 128).to(torch.bfloat16)
        # 实例化 M 类并设为评估模式
        m = M().eval()
        # 进入无梯度计算环境
        with torch.no_grad():
            # 重置度量指标
            metrics.reset()
            # 调用共有方法 self.common，传入模型 m 和输入数据 (idx, x)
            self.common(m, (idx, x))
            # 检查向量化内核数量是否为 1
            check_metrics_vec_kernel_count(1)

        # 进行直接加载/存储操作，确保不生成多余的类型转换
        m_opt = torch.compile(m)
        # 运行优化后的模型，并获取其 C++ 代码
        _, code = run_and_get_cpp_code(m_opt, idx, x)
        # 断言代码中含有 "Vectorized" 字符串
        self.assertTrue("Vectorized" in code)
        # 断言代码中不含有 "cvt_lowp_fp_to_fp32" 字符串
        self.assertTrue("cvt_lowp_fp_to_fp32" not in code)
        # 断言代码中不含有 "cvt_fp32_to_lowp_fp" 字符串
        self.assertTrue("cvt_fp32_to_lowp_fp" not in code)

    def test_concat_inner_vec(self):
        # 定义一个函数 fn，接收 x 和 y 作为输入，返回 x 和 y 拼接后的结果经过 relu 函数处理
        def fn(x, y):
            return F.relu(torch.cat([x, y], dim=1))

        # 生成两个随机张量 x 和 y，形状分别为 (32, 35) 和 (32, 120)
        x = torch.randn(32, 35)
        y = torch.randn(32, 120)
        # 重置度量指标
        metrics.reset()
        # 调用共有方法 self.common，传入函数 fn 和输入数据 (x, y)
        self.common(fn, (x, y))
        # 检查向量化内核数量是否为 3
        check_metrics_vec_kernel_count(3)

    def test_expr_vec_non_contiguous(self):
        # 定义一个函数 fn，接收 x 作为输入，执行一系列张量操作后返回 softmax 结果
        def fn(x):
            # 执行函数式的 pad 操作，给 x 填充 (0, 31)，然后 reshape 成 (-1, 33, 63)
            y = torch.nn.functional.pad(x, (0, 31)).reshape(-1, 33, 63)
            # 从 y 中切片取出部分，然后 reshape 成 (4, 32, 1, 32, 32)，并扩展维度
            y = y[:, :32, 31:].reshape(4, 32, 1, 32, 32).expand(-1, -1, 32, -1, -1)
            # 转置 y 的部分维度，设置 memory_format 为 torch.contiguous_format
            y = y.permute(0, 3, 1, 4, 2).clone(memory_format=torch.contiguous_format)
            # 将 y reshape 成 (4, 1024, 1024)，并执行 softmax 操作
            y = y.view(4, 1024, 1024)
            return y.softmax(dim=-1)

        # 生成一个随机张量 x，形状为 (128, 2048)
        x = torch.randn(128, 2048)
        # 编译优化函数 fn
        opt_fn = torch.compile(fn)
        # 重置度量指标
        metrics.reset()
        # 运行优化后的函数，并获取其 C++ 代码
        _, code = run_and_get_cpp_code(opt_fn, x)
        # 断言优化后的函数 fn(x) 与原函数 fn(x) 结果相同
        self.assertTrue(same(fn(x), opt_fn(x)))
        # 检查向量化内核数量是否为 4
        check_metrics_vec_kernel_count(4)
        # 使用 FileCheck 检查生成的 C++ 代码中是否有指定字符串出现次数为 0 次
        FileCheck().check_count(
            "Vectorized<int>::loadu(tmpbuf.data())", 0, exactly=True
        ).run(code)
    def test_vec_contiguous_ModularIndexing(self):
        # 测试向量连续性和模块化索引

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # 初始化时创建一个具有指定维度的 LayerNorm 层
                self.norm = torch.nn.LayerNorm(dim * 4)

            def forward(self, x):
                # 前向传播函数
                # 获取输入张量 x 的形状信息 B, H, W, C
                B, H, W, C = x.shape
                # 对输入张量 x 进行形状重塑和维度交换操作
                x = (
                    x.reshape(B, H // 2, 2, W // 2, 2, C)
                    .permute(0, 1, 3, 4, 2, 5)
                    .flatten(3)
                )
                # 对重塑后的张量 x 进行归一化处理
                x = self.norm(x)
                return x

        # 创建一个形状为 (1, 56, 56, 128) 的随机张量 x
        x = torch.randn(1, 56, 56, 128)
        # 创建 M 类的实例 m
        m = M(128)
        # 优化 M 类的实例 m
        opt_m = torch.compile(m)
        
        # 关闭梯度计算的上下文
        with torch.no_grad():
            # 重置指标数据
            metrics.reset()
            # 运行优化后的模型 opt_m，并获取生成的 C++ 代码
            _, code = run_and_get_cpp_code(opt_m, x)
            # 断言优化前后模型输出是否一致
            self.assertTrue(same(m(x), opt_m(x)))
            # 检查向量化内核数量是否为 2
            check_metrics_vec_kernel_count(2)
            # 使用 FileCheck 工具验证生成的 C++ 代码中指定模式的出现次数为 0
            FileCheck().check_count(
                "Vectorized<float>::loadu(tmpbuf.data())", 0, exactly=True
            ).run(code)

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    def test_fp8_cast(self, dtype: torch.dtype, shape: str):
        # 测试浮点类型转换到 fp8 的函数
        def fp8_cast(x):
            # 使用不同的 fp8 转换方式将输入张量 x 转换为指定 dtype 类型的张量
            y0 = x.to(dtype=torch.float8_e4m3fn).to(dtype)
            y1 = x.to(dtype=torch.float8_e5m2).to(dtype)
            return y0, y1

        # 解析形状字符串为一个列表形式的 shape
        shape = [int(dim) for dim in shape.split(",")]
        # 创建一个指定形状和 dtype 的随机张量 x
        x = torch.rand(*shape, device="cpu", dtype=dtype)
        # 调用通用测试函数 common，测试 fp8_cast 函数
        self.common(fp8_cast, (x,))

    def test_logical_op_store_to_lowp_data_dtype(self):
        # 测试逻辑运算结果存储到低精度数据类型的函数
        def fn(out1, out2, input, other):
            # 执行逻辑或和逻辑异或运算，并将结果存储到指定输出张量 out1 和 out2 中
            o1 = torch.logical_or(out=out1, input=input, other=other)
            o2 = torch.logical_xor(out=out2, input=input, other=other)
            return o1, o2

        # 创建两个随机张量 x 和 y
        x = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float)
        y = torch.rand([3, 3, 2, 8, 9, 2], dtype=torch.float)
        # 针对低精度浮点数类型进行迭代
        for dtype in _lowp_fp_dtypes:
            # 创建两个指定 dtype 的随机输出张量 o1 和 o2
            o1 = torch.rand([3, 3, 2, 8, 9, 2], dtype=dtype)
            o2 = torch.rand([3, 3, 2, 8, 9, 2], dtype=dtype)
            # 关闭梯度计算的上下文
            with torch.no_grad():
                # 调用通用测试函数 common，测试 fn 函数
                self.common(fn, (o1, o2, x, y))

    def test_constant_bool_vec(self):
        # 测试常量布尔向量的处理函数
        def fn(x):
            # 创建一个布尔类型的全零向量 mask
            mask = torch.zeros(1, dtype=torch.bool)
            # 根据 mask 的值，在输入张量 x 和 -1.0 之间进行选择
            return torch.where(mask, x, -1.0)

        # 创建一个随机张量 x
        x = torch.rand(1000)
        # 重置指标数据
        metrics.reset()
        # 调用通用测试函数 common，测试 fn 函数
        self.common(fn, (x,))
        # 检查向量化内核数量是否为 1
        check_metrics_vec_kernel_count(1)

    @torch._dynamo.config.patch(dynamic_shapes=True)
    @torch._dynamo.config.patch(assume_static_by_default=False)
    # 测试函数，用于测试符号形状标量值的减少
    def test_symbolic_shape_scalar_value_reduction(self):
        # 定义一个函数 fn，接受参数 x 和 y，返回 y 加上 x 全部元素求和后的结果
        def fn(x, y):
            return y + torch.ones(x).sum()

        # 进入 torch 的无梯度环境
        with torch.no_grad():
            # 重置度量指标
            metrics.reset()
            # 生成一个形状为 (100,) 的随机张量 y
            y = torch.randn(100)
            # 调用通用测试函数 common，传入函数 fn 和参数 (100, y)
            self.common(fn, (100, y))
            # 检查向量化内核函数调用次数是否为 2
            check_metrics_vec_kernel_count(2)

    # 测试函数，用于测试 int32 类型的逐点运算
    def test_int32_pointwise_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 的逐元素平方
        def fn(x):
            return x * x

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 int32 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 检查向量化内核函数调用次数是否为 1
        check_metrics_vec_kernel_count(1)

    # 测试函数，用于测试 int32 类型的降维操作
    def test_int32_reduction_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 按 dim=1 求和的结果
        def fn(x):
            return x.sum(dim=1)

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 int32 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 检查向量化内核函数调用次数是否为 1
        check_metrics_vec_kernel_count(1)

    # 测试函数，用于测试 uint32 类型的逐点运算
    def test_uint32_pointwise_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 的逐元素平方
        def fn(x):
            return x * x

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 uint32 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.uint32)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 断言向量化内核函数调用次数为 0，待优化为使用向量化 uint32 加载后调用一次
        assert metrics.generated_cpp_vec_kernel_count == 0

    # 测试函数，用于测试 uint32 类型的降维操作
    def test_uint32_reduction_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 按 dim=1 求和的结果
        def fn(x):
            return x.sum(dim=1)

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 uint32 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.uint32)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 断言向量化内核函数调用次数为 0，待优化为使用向量化 uint32/uint64 加载后调用一次
        assert metrics.generated_cpp_vec_kernel_count == 0

    # 测试函数，用于测试 int64 类型的逐点运算
    def test_int64_pointwise_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 的逐元素平方
        def fn(x):
            return x * x

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 int64 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 检查向量化内核函数调用次数是否为 1
        check_metrics_vec_kernel_count(1)

    # 测试函数，用于测试 int64 类型的降维操作
    def test_int64_reduction_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 按 dim=1 求和的结果
        def fn(x):
            return x.sum(dim=1)

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 int64 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 检查向量化内核函数调用次数是否为 1
        check_metrics_vec_kernel_count(1)

    # 测试函数，用于测试 uint64 类型的逐点运算
    def test_uint64_pointwise_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 的逐元素平方
        def fn(x):
            return x * x

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 uint64 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.uint64)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 断言向量化内核函数调用次数为 0，待优化为使用向量化 uint64 加载后调用一次
        assert metrics.generated_cpp_vec_kernel_count == 0

    # 测试函数，用于测试 uint64 类型的降维操作
    def test_uint64_reduction_vec(self):
        # 定义一个函数 fn，接受参数 x，返回 x 按 dim=1 求和的结果
        def fn(x):
            return x.sum(dim=1)

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 uint64 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.uint64)
        # 重置度量指标
        metrics.reset()
        # 调用通用测试函数 common，传入函数 fn 和参数 (x,)
        self.common(fn, (x,))
        # 断言向量化内核函数调用次数为 0，待优化为使用向量化 uint64 加载后调用一次
        assert metrics.generated_cpp_vec_kernel_count == 0

    # 测试函数，用于测试将 int32 类型转换为 int64 类型的向量操作
    def test_convert_int32_to_int64_vec(self):
        # 定义一个函数 fn，接受参数 x，将 x 转换为 int64 类型后返回
        def fn(x):
            return x.to(torch.int64)

        # 生成一个形状为 (32, 32)、元素取值在 [0, 100) 的 int32 类型张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int32)
        # 重置度量指标
        metrics.reset()
        #
    def test_convert_int64_to_int32_vec(self):
        # 定义一个函数，将输入张量转换为 torch.int32 类型
        def fn(x):
            return x.to(torch.int32)

        # 创建一个大小为 (32, 32) 的 torch.int64 随机张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        # 重置度量指标
        metrics.reset()
        # 调用公共方法，应用 fn 函数到 x 上
        self.common(fn, (x,))
        # 检查度量指标中向量化内核调用次数为 1
        check_metrics_vec_kernel_count(1)

    def test_convert_fp32_to_int64_vec(self):
        # 定义一个函数，将输入张量转换为 torch.int64 类型
        def fn(x):
            return x.to(torch.int64)

        # 创建一个大小为 (32, 32) 的 torch.float32 随机张量 x
        x = torch.rand(32, 32)
        # 重置度量指标
        metrics.reset()
        # 调用公共方法，应用 fn 函数到 x 上
        self.common(fn, (x,))
        # 检查度量指标中向量化内核调用次数为 1
        check_metrics_vec_kernel_count(1)

    def test_convert_int64_to_fp32_vec(self):
        # 定义一个函数，将输入张量转换为 torch.float32 类型
        def fn(x):
            return x.to(torch.float32)

        # 创建一个大小为 (32, 32) 的 torch.int64 随机张量 x
        x = torch.randint(0, 100, (32, 32), dtype=torch.int64)
        # 重置度量指标
        metrics.reset()
        # 调用公共方法，应用 fn 函数到 x 上
        self.common(fn, (x,))
        # 检查度量指标中向量化内核调用次数为 1
        check_metrics_vec_kernel_count(1)

    def test_no_redundant_to_dtypes_between_fused_scheduler_node(self):
        # https://github.com/pytorch/pytorch/issues/115260
        # 创建一个大小为 (1,) 的 torch.float16 张量 p0
        p0 = torch.tensor([1.0879], dtype=torch.float16)

        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, *args):
                # 在第二维度上连接多个张量 args 的切片
                cat = torch.cat((args[3], args[2], args[1], args[0]), dim=2)
                # 计算 args[4] 和 p0 中的每个元素的最大值
                max_1 = torch.max(args[4], p0)
                # 对 cat 和 max_1 进行按元素乘法
                mul = torch.mul(cat, max_1)
                # 对 mul 中的每个元素计算正切值
                tan = torch.tan(mul)
                # 返回乘法结果 mul 和正切结果 tan
                return (mul, tan)

        # 重置度量指标
        metrics.reset()
        # 创建 Model1 类的实例 m
        m = Model1()
        # 调用公共方法，将输入张量传递给 m
        self.common(
            m,
            (
                torch.randn((17, 5, 1, 7)).half(),
                torch.randn((17, 5, 1, 7)).half(),
                torch.randn((17, 5, 11, 7)).half(),
                torch.randn((17, 5, 1, 7)).half(),
                torch.tensor(4.39, dtype=torch.float16),
            ),
        )

    def test_masked_load_int64_vec(self):
        # https://github.com/pytorch/pytorch/issues/120377
        # 定义一个函数，对输入张量进行零填充
        def fn(x):
            return torch.nn.functional.pad(x, (0, 13))

        # 创建一个大小为 (819,) 的 torch.int64 随机张量 x
        x = torch.randint(0, 100, (819,), dtype=torch.int64)
        # 重置度量指标
        metrics.reset()
        # 调用公共方法，应用 fn 函数到 x 上
        self.common(fn, (x,))
        # 断言度量指标中生成的 C++ 向量化内核调用次数为 1
        assert metrics.generated_cpp_vec_kernel_count == 1

    def test_highp_to_lowp_cse_var_cache_with_store(self):
        # 修复问题：https://github.com/pytorch/pytorch/issues/128263
        # 创建一个大小为 (5, 128) 的 torch.float32 随机张量 input
        input = torch.randn(5, 128, dtype=torch.float32)
        # 创建一个大小为 (5, 128) 的 torch.int8 随机张量 input2
        input2 = torch.randint(0, 10, (5, 128), dtype=torch.int8)
        # 创建一个大小为 (128, 128) 的 torch.float32 随机张量 input3
        input3 = torch.randn(128, 128, dtype=torch.float32)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, x2, x3):
                # 将输入张量 x2 转换为 torch.int32 类型
                x2 = x2.to(torch.int32)
                # 调用 test_operators.realize 函数并将结果存储在 temp 中
                temp = test_operators.realize(x2.to(torch.float16))
                # 将 temp 转换为 torch.float32 类型并存储在 temp2 中
                temp2 = temp.to(torch.float32)
                # 计算 temp2 和输入张量 x 的按元素乘积，并存储在 temp2 中
                temp2 = temp2 * x
                # 返回 temp 和输入张量 x3 转换为 torch.float16 后的矩阵乘积，以及 temp2
                return torch.mm(temp, x3.to(torch.float16)), temp2

        # 重置度量指标
        metrics.reset()
        # 创建 Model 类的实例 m
        m = Model()
        # 调用公共方法，将 input、input2 和 input3 作为输入传递给 m
        self.common(
            m,
            (input, input2, input3),
        )
    def test_reduction_float_to_int64(self):
        # 测试函数：测试将浮点数张量降维为 int64 类型
        # https://github.com/pytorch/pytorch/issues/124821
        def fn(x):
            # 返回张量沿指定维度的最大值
            return x.max(0).values

        # 创建一个形状为 (22, 51) 的 int64 类型的随机整数张量
        x = torch.randint(0, 100, (22, 51), dtype=torch.int64)
        # 重置度量指标
        metrics.reset()
        # 调用公共函数执行测试
        self.common(fn, (x,))
        # 断言生成的 C++ 向量化内核计数为 1
        assert metrics.generated_cpp_vec_kernel_count == 1

    @config.patch({"cpp.dynamic_threads": True})
    def test_reduction_with_dynamic_threads(self):
        # 测试函数：测试动态线程数下的张量降维操作
        def fn(a, b):
            # 返回两个张量的求和结果
            return a.sum(), b.sum()

        # 调用公共函数执行测试，传入两个张量参数
        self.common(
            fn,
            (torch.randn(1000), torch.rand(1000)),
        )

    @patch("torch.cuda.is_available", lambda: False)
    @config.patch(freezing=True)
    def test_linear_float64(self):
        # 测试函数：测试 float64 类型下的线性层运算
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义两个 float64 类型的权重张量和一个偏置张量
                self.weight1 = torch.nn.Parameter(
                    torch.randn(10, 10, dtype=torch.float64)
                )
                self.weight2 = torch.nn.Parameter(
                    torch.randn(10, 10, dtype=torch.float64)
                )
                self.bias = torch.nn.Parameter(torch.randn(10, dtype=torch.float64))

            def forward(self, x1):
                # 计算线性变换结果
                v1 = torch.mm(x1, self.weight1)
                v2 = torch.addmm(self.bias, x1, self.weight2)
                return (v1, v2)

        # 创建并评估一个 float64 类型的模型实例
        mod = M().eval()
        v = torch.randn(10, 10, dtype=torch.float64)
        with torch.no_grad():
            # 调用公共函数执行测试
            self.common(
                mod,
                (v,),
            )

    def test_fused_attention_conv(self):
        # 测试函数：测试融合注意力卷积
        # https://github.com/pytorch/pytorch/issues/121174.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义三个卷积层作为注意力机制的组成部分
                self.q_conv = torch.nn.Conv2d(4, 4, 1)
                self.k_conv = torch.nn.Conv2d(4, 4, 1)
                self.v_conv = torch.nn.Conv2d(4, 4, 1)

            def forward(self, x):
                # 前向传播：计算查询、键、值的卷积结果，并执行注意力机制
                q = self.q_conv(x)
                k = self.k_conv(x)
                v = self.v_conv(x)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False
                )

        # 创建并评估一个模型实例，用于测试融合注意力卷积
        fn = Model()
        x = torch.randn(1, 4, 2, 2)
        # 调用公共函数执行测试
        self.common(fn, (x,))

    @requires_vectorization
    def test_vec_indirect_load_cse_cache(self):
        # 测试函数：测试向量间接加载的 CSE 缓存
        # 导入 infinity（无穷大）常数
        from math import inf
        
        def fn(arg0_1):
            # 调用 ATen 操作：创建指定大小的全 1 张量
            full_default = torch.ops.aten.full.default([209985], 1)
            # 调用 ATen 操作：选择张量中的第一个元素
            select = torch.ops.aten.select.int(arg0_1, 0, 0)
            # 调用 ATen 操作：选择张量中的第二个元素
            select_1 = torch.ops.aten.select.int(arg0_1, 0, 1)
            # 调用 ATen 操作：重塑张量的形状
            view = torch.ops.aten.reshape.default(select_1, [-1])
            # 调用 ATen 操作：扩展张量的维度
            expand = torch.ops.aten.expand.default(view, [209985])
            # 调用 ATen 操作：创建指定大小的全 0 张量
            full_default_1 = torch.ops.aten.full.default([10000], 0)
            # 调用 ATen 操作：执行散列加法
            scatter_add = torch.ops.aten.scatter_add.default(
                full_default_1, 0, expand, full_default
            )
            # 调用 ATen 操作：计算张量的幂次方
            pow_1 = torch.ops.aten.pow.Tensor_Scalar(scatter_add, -0.5)
            # 调用 ATen 操作：比较张量中的元素是否与无穷大相等
            eq = torch.ops.aten.eq.Scalar(pow_1, inf)
            # 调用 ATen 操作：创建空张量
            full_default_2 = torch.ops.aten.full.default([], 0.0)
            # 调用 ATen 操作：根据条件选择张量中的元素
            where = torch.ops.aten.where.self(eq, full_default_2, pow_1)
            # 调用 ATen 操作：根据索引选择张量中的元素
            index = torch.ops.aten.index.Tensor(where, [select])
            # 调用 ATen 操作：根据索引选择张量中的元素
            index_1 = torch.ops.aten.index.Tensor(where, [select_1])
            # 调用 ATen 操作：对张量进行逐元素乘法
            mul_1 = torch.ops.aten.mul.Tensor(index, index_1)
            return (mul_1,)

        # 创建指定大小的零张量，并转换为 int64 类型
        x = torch.zeros(2, 209985).to(torch.int64)
        # 优化函数 fn，并获取优化后的函数及其生成的 C++ 代码
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        _, code = run_and_get_cpp_code(opt_fn, x)
        # 使用 FileCheck 工具检查生成的 C++ 代码中 ".loadu" 的使用次数
        FileCheck().check_count(
            "return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(),",
            4,
            exactly=True,
        ).run(code)

    def test_repeated_exp(self):
        # 测试函数：测试重复的指数函数调用
        def fn(x):
            # 计算输入张量的 sigmoid 函数
            y = x.sigmoid()
            # 返回 sigmoid 函数值加 1 和在指定维度上的元素求和结果
            return y + 1, y.sum(-1)

        # 创建指定大小的随机张量
        x = torch.randn(1000, 1000)
        # 编译函数 fn 并获取生成的 C++ 代码
        opt_fn = torch.compile(fn)
        _, code = run_and_get_cpp_code(opt_fn, x)
        # 使用 FileCheck 工具检查生成的 C++ 代码中 ".exp()" 的使用次数
        FileCheck().check_count(
            ".exp()",
            1,
            exactly=True,
        ).run(code)
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 导入 torch 库中的测试函数 run_tests
    from torch._inductor.test_case import run_tests
    # 从 torch.testing._internal.inductor_utils 导入 HAS_CPU 常量
    from torch.testing._internal.inductor_utils import HAS_CPU
    
    # 如果 HAS_CPU 为 True 并且 IS_MACOS 不为真
    if HAS_CPU and not IS_MACOS:
        # 运行测试函数 run_tests，并要求其中有名为 "filelock" 的依赖项
        run_tests(needs="filelock")
```