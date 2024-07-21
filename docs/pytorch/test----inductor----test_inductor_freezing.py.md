# `.\pytorch\test\inductor\test_inductor_freezing.py`

```py
# Owner(s): ["module: inductor"]
# 引入标准库和第三方库
import contextlib                     # 上下文管理器相关功能
import functools                      # 函数装饰器和高阶函数的实用工具
import importlib                      # 提供对模块的动态加载
import itertools                      # 创建迭代器的函数
import os                            # 提供与操作系统交互的功能
import sys                           # 提供对解释器相关功能的访问和操作
import unittest                      # Python 的单元测试框架
import weakref                       # Python 弱引用对象

import torch                         # PyTorch 深度学习库

from torch import nn                 # 神经网络模块
from torch._inductor import config   # Torch Inductor 配置模块
from torch._inductor.test_case import TestCase as InductorTestCase  # Inductor 测试用例
from torch._inductor.utils import override_lowering, run_and_get_code  # Inductor 工具函数
from torch.testing import FileCheck  # 用于测试文件是否包含某些字符串的实用工具
from torch.testing._internal.common_cuda import SM80OrLater  # CUDA 相关的硬件兼容性检查
from torch.testing._internal.common_utils import skipIfRocm  # 条件测试装饰器，用于跳过 ROCm 平台的测试

# Make the helper files in test/ importable
# 将 test/ 目录下的辅助文件加入到模块搜索路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch.testing._internal.common_utils import (
    IS_CI,                          # 是否处于持续集成环境
    IS_WINDOWS,                     # 是否在 Windows 操作系统上
    TEST_WITH_ASAN,                 # 是否进行 AddressSanitizer 内存检查
    TEST_WITH_ROCM,                 # 是否在 ROCm 平台上运行测试
)

if IS_WINDOWS and IS_CI:
    # 如果在 Windows CI 环境下，缺少 test_torchinductor 的必要依赖
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)  # 程序正常退出
    raise unittest.SkipTest("requires sympy/functorch/filelock")  # 跳过当前测试

from inductor.test_torchinductor import check_model, check_model_cuda, copy_tests  # 导入 Inductor 测试函数

importlib.import_module("functorch")  # 动态导入 functorch 模块
importlib.import_module("filelock")   # 动态导入 filelock 模块

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 导入 CPU 和 CUDA 的硬件支持检查

aten = torch.ops.aten      # PyTorch aten 操作命名空间
prims = torch.ops.prims    # PyTorch prims 操作命名空间
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")  # 仅在有 CUDA 支持时才运行的测试装饰器

# 定义测试用例类，继承自 InductorTestCase
class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()  # 创建上下文管理器堆栈对象
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,  # 调试模式
                    "cpp.min_chunk_size": 1,  # 最小块大小
                    "triton.autotune_pointwise": False,  # 禁用 Triton 自动调优
                    "implicit_fallbacks": False,  # 禁用隐式回退
                    "freezing": True,  # 冻结模式
                    "freezing_discard_parameters": True,  # 冻结时丢弃参数
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()  # 关闭上下文管理器堆栈
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()  # 重置 Torch 内部状态
        super().setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()  # 重置 Torch 内部状态


class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)  # 定义卷积层
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)  # 定义批归一化层

    def forward(self, x):
        return self.bn(self.conv(x))  # 卷积层后接批归一化层的前向传播


class ConvFunctionalBN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        stride=2,
        running_mean=None,
        running_var=None,
        weight=None,
        bn_bias=None,
    ):
        # 调用父类构造函数进行初始化
        super().__init__()
        # 创建卷积层对象，指定输入通道数、输出通道数、是否使用偏置、卷积核大小和步长
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride
        )
        # 设置运行时均值
        self.running_mean = running_mean
        # 设置运行时方差
        self.running_var = running_var
        # 设置权重
        self.weight = weight
        # 设置 Batch Normalization 的偏置项
        self.bias = bn_bias

    def forward(self, x):
        # 执行前向传播：先对输入进行卷积，然后应用批归一化函数
        return torch.nn.functional.batch_norm(
            self.conv(x),   # 将输入 x 经过卷积操作得到特征图
            self.running_mean,  # 使用给定的运行时均值进行批归一化
            self.running_var,   # 使用给定的运行时方差进行批归一化
            self.weight,        # 使用给定的权重进行批归一化
            self.bias,          # 使用给定的偏置进行批归一化
            False,              # 不进行训练模式的 Batch Normalization
            0.1,                # 动量参数设置为 0.1
            1e-5,               # epsilon 参数设置为 1e-5，用于数值稳定性
        )
class ConvMultiBN(torch.nn.Module):
    # 定义一个包含多个卷积层和批归一化层的模块
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()
        # 初始化卷积层，指定输入通道数、输出通道数以及其他参数
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        # 初始化第一个批归一化层，对输出通道数进行归一化处理，设置epsilon值为0.001
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001, dtype=torch.float)
        # 初始化第二个批归一化层，同样对输出通道数进行归一化处理，设置epsilon值为0.1
        self.bn2 = torch.nn.BatchNorm2d(out_channels, eps=0.1, dtype=torch.float)

    def forward(self, x):
        # 对输入数据进行卷积操作，并应用第一个批归一化层
        tmp = self.bn(self.conv(x))
        # 对输入数据进行卷积操作，并应用第二个批归一化层
        tmp2 = self.bn2(self.conv(x))
        # 返回两次处理结果的和
        return tmp + tmp2


class ConvMultiFunctionalBN(torch.nn.Module):
    # 定义一个包含多个功能的卷积层和批归一化层的模块
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        kernel_size=3,
        stride=2,
        running_mean=None,
        running_var=None,
        weight=None,
        bn_bias=None,
        running_mean2=None,
    ):
        super().__init__()
        # 初始化卷积层，指定输入通道数、输出通道数、是否使用偏置、卷积核大小和步长
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, bias=bias, kernel_size=kernel_size, stride=stride
        )
        # 初始化批归一化层的相关参数
        self.running_mean = running_mean
        self.running_var = running_var
        self.weight = weight
        self.bias = bn_bias
        self.running_mean2 = running_mean2

    def forward(self, x):
        # 对输入数据进行卷积操作，并应用批归一化操作，使用第一组归一化参数
        tmp = torch.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )
        # 对输入数据进行卷积操作，并应用批归一化操作，使用第二组归一化参数
        tmp2 = torch.nn.functional.batch_norm(
            self.conv(x),
            self.running_mean2,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.1,
            1e-5,
        )
        # 返回两次处理结果的和
        return tmp + tmp2


class OptimizeForInferenceTemplate(TestCase):
    # 优化推断模板的测试用例
    def test_mutation(self):
        # 定义一个修改过参数的模块
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个可变参数作为模块的参数
                self.mutated_param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self):
                # 在前向传播中对参数进行加法操作
                self.mutated_param.add_(10)
                # 返回修改后的参数
                return self.mutated_param

        with torch.no_grad():
            # 使用无梯度计算上下文
            mod = Mod().to(self.device)
            # 前向计算并获取结果
            out_eager = mod()
            out_eager2 = mod()

            mod = Mod().to(self.device)

            @torch.compile
            def foo(mod):
                # 编译模块并进行前向计算
                return mod()

            # 使用编译后的模块进行前向计算
            out_comp = foo(mod)
            out_comp2 = foo(mod)

            # 断言两种计算方式得到的结果相等
            self.assertEqual(out_eager, out_comp)
            self.assertEqual(out_eager2, out_comp2)
    def test_aliased_param_return(self):
        # 定义一个继承自torch.nn.Module的模型类Mod
        class Mod(torch.nn.Module):
            # 初始化方法，创建一个大小为10x10的零张量，并包装为参数
            def __init__(self):
                super().__init__()
                self.aliased_param = torch.nn.Parameter(torch.zeros([10, 10]))

            # 前向传播方法，返回参数的切片和整个参数
            def forward(self):
                return self.aliased_param[1:], self.aliased_param

        # 创建Mod类的实例，将其移至指定的设备并设置为评估模式
        mod = Mod().to(self.device).eval()

        # 使用torch.compile()装饰器定义函数foo，接受一个模型作为参数，并返回模型的前向传播结果
        @torch.compile()
        def foo(mod):
            return mod()

        # 在无梯度计算环境下，执行模型的前向传播
        with torch.no_grad():
            mod_eager = mod()
            # 断言编译后的模型运行结果与即时计算的结果一致
            self.assertEqual(foo(mod), mod_eager)

    def test_autocast(self):
        # 如果设备为CPU，则跳过测试并抛出unittest.SkipTest异常
        if self.device == "cpu":
            raise unittest.SkipTest("MLKDNN Bug")

        # 创建一个包含10个输入和10个输出的线性层，并将其移至指定的设备并设置为评估模式
        mod = torch.nn.Linear(10, 10).to(self.device).eval()
        # 创建一个大小为10x10的随机张量，并将其转换为半精度数据类型，并移至指定的设备
        inp = torch.rand([10, 10]).to(self.device).to(torch.half)

        # 使用torch.compile()装饰器定义函数foo，接受一个模型和一个输入张量作为参数，并返回模型对输入的计算结果
        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        # 在无梯度计算环境下
        with torch.no_grad():
            # 在自动类型转换上下文中，执行模型对输入的前向传播计算
            with self.autocast():
                out_eager = mod(inp)
                # 运行并获取foo函数的代码和计算结果
                out_compiled, code = run_and_get_code(foo, mod, inp)

                # 检查生成的代码中是否未包含"@triton.jit"标记
                FileCheck().check_not("@triton.jit").run(code[0])
                # 断言即时计算和编译后的结果相等
                self.assertEqual(out_eager, out_compiled)

    # 在内联内置nn模块时，Dynamo追踪内置模块的内部并不修改即时模块。
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_error_on_eager(self):
        # 创建一个卷积层和批归一化层的组合模型，指定输入通道数为3，输出通道数为32，设置核大小为3，步长为2，并将其移至指定设备
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        # 创建一个大小为3x3x32x32的随机张量，并将其移至指定设备
        x = torch.rand(3, 3, 32, 32).to(self.device)

        # 使用torch.compile()装饰器定义函数foo，接受一个模型和一个输入张量作为参数，并返回模型对输入的计算结果
        @torch.compile()
        def foo(mod, x):
            return mod(x)

        # 在无梯度计算环境下
        with torch.no_grad():
            # 执行foo函数的计算，验证即时模块无法在Dynamo冻结后运行时抛出RuntimeError异常
            foo(mod, x)

        # 断言运行即时模块在Dynamo冻结后会抛出RuntimeError异常
        with self.assertRaisesRegex(
            RuntimeError, "Trying to run Pytorch Eager Module after Dynamo Freezing"
        ):
            mod(x)

    def test_rng_op(self):
        # 使用torch.compile()装饰器定义函数foo，返回一个大小为4x4的随机张量并加1
        @torch.compile()
        def foo():
            return torch.rand([4, 4], device=self.device) + 1

        # 在无梯度计算环境下
        with torch.no_grad():
            o1 = foo()
            o2 = foo()
            # 断言两次调用foo函数返回的结果不相等
            self.assertNotEqual(o1, o2)

    def test_symint_not_folded(self):
        # 定义一个接受输入张量a的函数fn，返回a的余弦值和一个与a形状相同的全零张量
        def fn(a):
            return a.cos(), torch.zeros(a.shape[0], a.shape[1])

        # 使用torch._dynamo.optimize()函数对函数fn进行优化，命名优化器为"inductor"，开启动态优化模式
        fn_opt = torch._dynamo.optimize("inductor", dynamic=True)(fn)
        # 创建一个大小为2x4x6的随机张量，并将其移至指定设备
        inp = torch.randn(2, 4, 6).to(self.device)
        # 在输入张量的第0维和第1维上标记为动态维度
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)

        # 在无梯度计算环境下
        with torch.no_grad():
            # 断言原始函数fn和优化后的函数fn_opt在相同输入上的计算结果相等
            self.assertEqual(fn(inp), fn_opt(inp))
            # 创建一个大小为3x5x6的随机张量，并将其移至指定设备
            inp2 = torch.randn(3, 5, 6).to(self.device)
            # 在输入张量的第0维和第1维上标记为动态维度
            torch._dynamo.mark_dynamic(inp2, 0)
            torch._dynamo.mark_dynamic(inp2, 1)
            # 断言原始函数fn和优化后的函数fn_opt在相同输入上的计算结果相等
            self.assertEqual(fn(inp2), fn_opt(inp2))

    @requires_cuda
    def test_conv_multiple_uses(self):
        # 引入 torch 中的 nn 模块，用于神经网络构建
        from torch import nn

        # 定义一个简单的神经网络模型 ToyModel
        class ToyModel(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                # 定义一个 2D 卷积层，输入通道数为 1，输出通道数为 1，核大小为 1
                self.conv1 = nn.Conv2d(1, 1, 1)
                # 定义一个批标准化层，输入通道数为 1
                self.bn1 = nn.BatchNorm2d(1)
                # 初始化批标准化层的权重为正态分布
                self.bn1.weight.data.normal_()

            # 前向传播函数
            def forward(self, x, y):
                # 返回卷积层 conv1 对输入 x 和 y 的处理结果与批标准化层 bn1 对 conv1(y) 的处理结果之和
                return self.conv1(x) + self.bn1(self.conv1(y))

        # 创建一个 ToyModel 实例
        model = ToyModel()
        # 将模型切换为评估模式，并移动到 CUDA 设备上进行加速计算
        model.eval().cuda()

        # 生成随机张量 a 和 b，形状为 [64, 1, 32, 32]，并将它们移动到 CUDA 设备上
        a = torch.rand(64, 1, 32, 32).cuda()
        b = torch.rand(64, 1, 32, 32).cuda()

        # 使用模型进行前向传播，计算输出结果
        output = model(a, b)

        # 在不需要梯度的上下文中，使用编译后的模型进行前向传播，计算输出结果
        with torch.no_grad():
            output2 = torch.compile(model)(a, b)

        # 断言两次前向传播的结果应当相等
        self.assertEqual(output, output2)

    def test_unfolded_bn(self):
        # 创建一个形状为 [3, 32, 15, 15] 的随机张量 x，并将其移动到指定设备上
        x = torch.rand([3, 32, 15, 15]).to(self.device)

        # 创建一个 2D 批标准化层，输入通道数为 32，设置 epsilon 为 0.001，并将其设置为评估模式并移动到指定设备上
        mod = torch.nn.BatchNorm2d(32, eps=0.001).eval().to(self.device)

        # 定义一个使用编译功能的函数 foo，接受一个模型和输入张量，并返回模型对输入张量处理后的结果加上常数 10
        @torch.compile()
        def foo(mod, x):
            return mod(x) + 10

        # 使用 foo 函数对 mod 和 x 进行处理，得到编译后的输出结果（不执行推理）
        out_compiled_no_inference = foo(mod, x)

        # 在不需要梯度的上下文中，使用编译后的模型对输入张量进行处理，并得到输出结果
        with torch.no_grad():
            out_compiled = foo(mod, x)

            # 断言两次处理的结果应当相等
            self.assertEqual(out_compiled_no_inference, out_compiled)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn(self):
        # 使用 itertools.product 生成所有 use_bias 和 dtype 的组合
        for use_bias, dtype in itertools.product(
            [True, False], [torch.float16, torch.bfloat16, torch.float32]
        ):
            # 如果设备为 CPU 且 dtype 为 torch.float16，则跳过当前循环
            if self.device == "cpu" and dtype == torch.float16:
                continue

            # 如果设备为 CUDA 且 dtype 为 torch.bfloat16 且不支持 SM80 或更高版本，则跳过当前循环
            if self.device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
                continue

            # 创建一个 ConvBN 对象，包括一个 3 通道到 32 通道的卷积层和对应的批标准化层，设定相关参数并移动到指定设备和数据类型上
            mod = (
                ConvBN(3, 32, bias=use_bias, kernel_size=3, stride=2)
                .eval()
                .to(self.device)
                .to(dtype)
            )

            # 创建一个形状为 [3, 3, 32, 32] 的随机张量 x，并移动到指定设备和数据类型上
            x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

            # 定义一个使用编译功能的函数 foo，接受一个模型和输入张量，并返回模型对输入张量处理后的结果
            @torch.compile()
            def foo(mod, x):
                return mod(x)

            # 在不需要梯度的上下文中，使用 mod 对输入张量进行处理，得到直接执行的输出结果
            out_eager = mod(x)
            # 运行 foo 函数获取优化后的输出结果以及相应的代码
            out_optimized_for_infernece, code = run_and_get_code(foo, mod, x)

            # 断言优化后的输出结果与直接执行的输出结果相等，允许的误差为 1e-2
            self.assertEqual(
                out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2
            )

            # 检查生成的代码，确保卷积偏置没有融合，且卷积核中只有一个常数
            if self.device == "cuda":
                FileCheck().check_not(".run(").check("conv").check(".run(").check_same(
                    "frozen_param"
                ).check_not("frozen_param").check_next("return").run(code[0])

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_bn_with_module_sharing(self):
        mod = (
            ConvBN(32, 32, bias=True, kernel_size=3, stride=2)  # 创建一个带有卷积和批量归一化的模块，设置参数如卷积核大小、步幅等
            .to(self.device)  # 将模块移动到指定的计算设备（GPU或CPU）
            .to(torch.float32)  # 将模块中的数据类型设置为32位浮点数
        )

        # 更新批量归一化模块的默认参数
        for _ in range(10):
            mod(torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32))  # 对模块进行前向传播，以便更新批量归一化的参数

        mod.eval()  # 设置模块为评估模式，这会影响批量归一化层的行为
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)  # 创建一个输入张量

        def foo(mod, x):
            mod(x)  # 对输入张量进行前向传播，使用模块处理数据
            return mod(x)  # 再次对输入张量进行前向传播，以验证模块是否与上一次结果一致

        with torch.no_grad():
            out_eager = foo(mod, x)  # 在不计算梯度的情况下，使用模块处理输入数据并获取输出
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x  # 编译模块以进行推断优化，并使用编译后的模块处理输入数据
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)  # 检查编译后的模块输出是否与直接运行模块的输出一致

    @torch._inductor.config.patch(layout_optimization=False)
    def test_folded_conv_functional_bn_with_module_sharing(self):
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)  # 创建输入张量并移动到指定设备
        running_mean = torch.mean(x, dim=(0, 2, 3)).to(self.device)  # 计算输入张量的运行均值
        running_var = torch.var(x, dim=(0, 2, 3)).to(self.device)  # 计算输入张量的运行方差

        mod = (
            ConvFunctionalBN(  # 创建一个功能性的带有卷积和批量归一化的模块
                32,
                32,
                bias=True,
                kernel_size=3,
                stride=2,
                running_mean=running_mean,  # 设置批量归一化层的运行均值
                running_var=running_var,  # 设置批量归一化层的运行方差
                weight=torch.ones(32).to(self.device),  # 设置权重张量并移动到指定设备
                bn_bias=torch.zeros(32).to(self.device),  # 设置批量归一化层的偏置张量并移动到指定设备
            )
            .eval()  # 将模块设置为评估模式
            .to(self.device)  # 将模块移动到指定设备
            .to(torch.float32)  # 将模块中的数据类型设置为32位浮点数
        )

        def foo(mod, x):
            mod(x)  # 对输入张量进行前向传播，使用模块处理数据
            return mod(x)  # 再次对输入张量进行前向传播，以验证模块是否与上一次结果一致

        with torch.no_grad():
            out_eager = foo(mod, x)  # 在不计算梯度的情况下，使用模块处理输入数据并获取输出
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x  # 编译模块以进行推断优化，并使用编译后的模块处理输入数据
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)  # 检查编译后的模块输出是否与直接运行模块的输出一致

    @torch._inductor.config.patch(layout_optimization=False)
    def test_conv_bn_with_multi_bn_share_conv(self):
        mod = (
            ConvMultiBN(32, 32, bias=True, kernel_size=3, stride=2)  # 创建一个带有多个批量归一化层的卷积模块，设置参数如卷积核大小、步幅等
            .to(self.device)  # 将模块移动到指定的计算设备（GPU或CPU）
            .to(torch.float32)  # 将模块中的数据类型设置为32位浮点数
        )

        # 更新批量归一化模块的默认参数
        for _ in range(10):
            mod(torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32))  # 对模块进行前向传播，以便更新批量归一化的参数

        mod.eval()  # 设置模块为评估模式，这会影响批量归一化层的行为
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)  # 创建一个输入张量

        def foo(mod, x):
            return mod(x)  # 对输入张量进行前向传播，使用模块处理数据

        with torch.no_grad():
            out_eager = foo(mod, x)  # 在不计算梯度的情况下，使用模块处理输入数据并获取输出
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x  # 编译模块以进行推断优化，并使用编译后的模块处理输入数据
            )

        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)  # 检查编译后的模块输出是否与直接运行模块的输出一致

    @torch._inductor.config.patch(layout_optimization=False)
    def test_conv_functional_bn_with_multi_bn_share_conv(self):
        # 创建输入张量 x，形状为 [3, 32, 32, 32]，随机初始化，移动到指定设备并转换为 float32
        x = torch.rand(3, 32, 32, 32).to(self.device).to(torch.float32)
        # 计算沿指定维度的均值作为 running_mean，并移动到设备
        running_mean = torch.mean(x, dim=(0, 2, 3)).to(self.device)
        # 计算沿指定维度的方差作为 running_var，并移动到设备
        running_var = torch.var(x, dim=(0, 2, 3)).to(self.device)
        # 再次计算均值作为 running_mean2，并移动到设备
        running_mean2 = torch.mean(x, dim=(0, 2, 3)).to(self.device)

        # 创建 ConvMultiFunctionalBN 模块实例 mod
        mod = (
            ConvMultiFunctionalBN(
                32,
                32,
                bias=True,
                kernel_size=3,
                stride=2,
                running_mean=running_mean,
                running_var=running_var,
                weight=torch.ones(32).to(self.device),
                bn_bias=torch.zeros(32).to(self.device),
                running_mean2=running_mean2,
            )
            .eval()  # 设置为评估模式
            .to(self.device)  # 移动到设备
            .to(torch.float32)  # 转换为 float32
        )

        # 定义函数 foo，用于调用 mod 对象处理输入 x
        def foo(mod, x):
            return mod(x)

        # 在没有梯度的上下文中执行以下操作
        with torch.no_grad():
            # 使用 foo 函数直接调用 mod 得到 out_eager
            out_eager = foo(mod, x)
            # 编译 foo 函数并运行得到 out_optimized_for_infernece
            out_optimized_for_infernece, _ = run_and_get_code(
                torch.compile(foo), mod, x
            )
        # 断言优化后的输出和直接调用的输出相等，设置误差容差
        self.assertEqual(out_optimized_for_infernece, out_eager, atol=1e-2, rtol=1e-2)

    @torch._inductor.config.patch(layout_optimization=False)
    def test_dont_change_dtype_folding(self):
        # 根据设备选择数据类型
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

        # 创建 Conv2d 模块实例 mod，设置为评估模式，移动到设备并设置数据类型
        mod = (
            torch.nn.Conv2d(3, 32, bias=None, kernel_size=3, stride=2)
            .eval()
            .to(self.device)
            .to(dtype)
        )
        # 创建输入张量 x，形状为 [3, 3, 32, 32]，随机初始化，移动到设备并设置数据类型
        x = torch.rand(3, 3, 32, 32).to(self.device).to(dtype)

        # 定义函数 foo，用于调用 mod 对象处理输入 x，并乘以常数
        def foo(mod, x):
            return mod(x) * torch.full([1], 2.0, device=self.device)

        # 编译 foo 函数
        foo_c = torch.compile(foo)

        # 在没有梯度的上下文中执行以下操作
        with torch.no_grad():
            # 使用 foo 函数直接调用 mod 得到 out_eager
            out_eager = foo(mod, x)
            # 使用编译后的 foo 函数调用 mod 得到 out_compiled
            out_compiled = foo_c(mod, x)
            # 断言两种方法得到的输出相等
            self.assertEqual(out_eager, out_compiled)

    def test_param_deallocated(self):
        # 如果设备为 CPU，则跳过测试
        if self.device == "cpu":
            raise unittest.SkipTest("NYI CPU")

        # 定义一个简单的 Module 类 Mod
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加一个参数 param，形状为 [10, 10]，初始化为零
                self.param = torch.nn.Parameter(torch.zeros([10, 10]))

            def forward(self, x):
                # 模块的前向传播，返回参数加 10 后的结果加上输入 x
                return (self.param + 10) + x

        # 创建 Mod 的实例 mod，设置为评估模式，并移动到设备
        mod = Mod().eval().to(self.device)
        # 创建输入张量 inp，形状为 [10]，随机初始化，移动到设备
        inp = torch.rand([10], device=self.device)

        # 在没有梯度的上下文中执行以下操作
        with torch.no_grad():
            # 直接调用 mod 处理输入 inp，得到 eager
            eager = mod(inp)

        # 使用 @torch.compile() 装饰器编译 foo 函数
        @torch.compile()
        def foo(mod, inp):
            return mod(inp)

        # 在没有梯度的上下文中执行以下操作
        with torch.no_grad():
            # 使用编译后的 foo 函数调用 mod 处理输入 inp，得到 compiled
            compiled = foo(mod, inp)

        # 断言 eager 和 compiled 相等
        self.assertEqual(eager, compiled)
        # 确保参数 param 已经被释放
        self.assertTrue(weight_ref() is None)

    @skipIfRocm
    def test_conv_with_as_strided(self):
        # 定义一个继承自 nn.Module 的模型类 Model
        class Model(nn.Module):
            # 初始化方法，接收 groups 参数
            def __init__(self, groups):
                super().__init__()
                # 创建一个 2D 卷积层对象，设置输入通道为 256，输出通道为 384，核大小为 (1, 1)，步长为 (1, 1)，无偏置项，分组数为 groups
                self.kv = torch.nn.Conv2d(
                    256,
                    384,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    bias=False,
                    groups=groups,
                )

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 对输入 x 执行卷积操作
                convolution = self.kv(x)
                # 在卷积结果上进行常量填充操作，填充值为 0.0，填充尺寸为 [2, 2, 2, 2]
                constant_pad_nd = torch.ops.aten.constant_pad_nd.default(
                    convolution, [2, 2, 2, 2], 0.0
                )
                # 使用 as_strided 方法创建一个新的张量，形状为 [8, 384, 2, 20, 12]，步幅为 [153600, 400, 160, 1, 20]
                # 此处的参数设置依赖于输入的大小和步幅
                as_strided = torch.ops.aten.as_strided.default(
                    constant_pad_nd, [8, 384, 2, 20, 12], [153600, 400, 160, 1, 20]
                )
                # 再次使用 as_strided 方法创建另一个新的张量，形状为 [8, 384, 2, 2, 12, 12]，步幅为 [153600, 400, 160, 8, 20, 1]
                as_strided_1 = torch.ops.aten.as_strided.default(
                    as_strided, [8, 384, 2, 2, 12, 12], [153600, 400, 160, 8, 20, 1]
                )
                # 使用 clone 方法复制张量 as_strided_1，并设置内存格式为连续
                clone = torch.ops.aten.clone.default(
                    as_strided_1, memory_format=torch.contiguous_format
                )
                # 返回复制后的张量
                return clone

        # 编译器标记函数，接收模型和输入
        @torch.compile()
        def foo(mod, inp):
            # 调用模型处理输入并返回结果
            return mod(inp)

        # 禁用梯度计算
        with torch.no_grad():
            # 创建输入张量 x，形状为 [8, 256, 16, 16]，传输至设备 self.device
            x = torch.randn(8, 256, 16, 16).to(self.device)
            # 对每种分组数进行迭代：1 和 2
            for groups in [1, 2]:
                # 创建模型对象 mod，并设置为评估模式，传输至设备 self.device
                mod = Model(groups).to(self.device).eval()
                # 在模型上处理输入 x 并保存结果
                mod_eager = mod(x)
                # 断言调用 foo 函数返回的结果与模型处理 x 后的结果相等
                self.assertEqual(foo(mod, x), mod_eager)

    # 测试 CPP 封装器功能
    def test_cpp_wrapper(self):
        # 创建 ConvBN 对象 mod，输入通道为 3，输出通道为 32，核大小为 3，步长为 2，设置为评估模式，并传输至设备 self.device
        mod = ConvBN(3, 32, kernel_size=3, stride=2).eval().to(self.device)

        # 创建输入张量 x，形状为 [3, 3, 32, 32]，传输至设备 self.device
        x = torch.rand(3, 3, 32, 32).to(self.device)

        # 编译器标记函数，接收模型和输入，设置选项为启用 CPP 封装器
        @torch.compile(options={"cpp_wrapper": True})
        def foo(mod, x):
            # 调用模型处理输入并返回结果
            return mod(x)

        # 获取模型在输入 x 上的结果
        out_eager = mod(x)

        # 禁用梯度计算
        with torch.no_grad():
            # 断言调用 foo 函数两次返回的结果与模型处理 x 后的结果相等
            self.assertEqual(foo(mod, x), out_eager)
            self.assertEqual(foo(mod, x), out_eager)

    # 测试卷积布局转换与视图结合
    def test_conv_layout_convert_with_view(self):
        # 定义一个继承自 torch.nn.Module 的模型类 Model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 2D 卷积层对象 conv，输入通道为 3，输出通道为 128，核大小为 3，填充为 1，步长为 1，无偏置项
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )
                # 创建一个批归一化层对象 bn，输入通道为 3
                self.bn = nn.BatchNorm2d(3)

            # 前向传播方法，接收输入 x
            def forward(self, x):
                # 在输入 x 上执行批归一化操作
                x = self.bn(x)
                # 在经过批归一化后的张量上执行卷积操作
                x = self.conv(x)
                # 将卷积结果展平，从第二维开始展平
                return torch.flatten(x, 1)

        # 创建模型对象 mod，设置为评估模式，并传输至设备 self.device
        mod = Model().to(self.device).eval()

        # 编译器标记函数，接收模型和输入
        @torch.compile()
        def foo(mod, inp):
            # 调用模型处理输入并返回结果
            return mod(inp)

        # 禁用梯度计算
        with torch.no_grad():
            # 创建输入张量 x，形状为 [2, 3, 5, 5]，传输至设备 self.device
            x = torch.rand(2, 3, 5, 5).to(self.device)
            # 在模型上处理输入 x 并保存结果
            mod_eager = mod(x)
            # 断言调用 foo 函数返回的结果与模型处理 x 后的结果相等
            self.assertEqual(foo(mod, x), mod_eager)

    # 跳过 Rocm 平台的测试
    @skipIfRocm
    # 定义一个测试方法，用于验证卷积权重布局转换的功能
    def test_conv_weight_layout_convert(self):
        # 定义一个内部模型类 Model，继承自 torch.nn.Module
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个 2D 卷积层，输入通道数为 3，输出通道数为 128，核大小为 3，填充为 1，步幅为 1，不使用偏置项
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            # 前向传播方法
            def forward(self, x):
                return self.conv(x)

            # 静态方法，返回一个示例输入的元组，形状为 (2, 3, 5, 5)，并发送到设备上
            @staticmethod
            def get_example_inputs():
                return (torch.rand(2, 3, 5, 5).to(self.device),)

        # 导入用于编译的函数 compile_fx 和 compile_fx_inner
        from torch._inductor.compile_fx import compile_fx, compile_fx_inner

        # 初始化卷积计数器 nconv
        nconv = 0

        # 定义一个内部编译函数 my_inner_compile，接受参数 gm, example_inputs, *args, **kwargs
        def my_inner_compile(gm, example_inputs, *args, **kwargs):
            # 调用内部编译函数 compile_fx_inner 处理 gm 和 example_inputs，返回输出结果 out
            out = compile_fx_inner(gm, example_inputs, *args, **kwargs)

            # 使用 nonlocal 关键字引用外部变量 nconv
            nonlocal nconv
            # 查找所有节点中目标为 aten.convolution.default 的节点，存入列表 convs
            convs = [n for n in gm.graph.nodes if n.target == aten.convolution.default]
            # 更新卷积计数器 nconv，增加 convs 的长度
            nconv += len(convs)
            # 遍历 convs 中的每个节点
            for conv in convs:
                # 获取权重节点，位置为 conv.args[1]
                weight_node = conv.args[1]
                # 从 gm 中获取权重常量张量
                weight_const_tensor = getattr(gm, weight_node.target)
                # 断言权重常量张量在内存中是连续的，内存格式为 torch.channels_last
                self.assertTrue(
                    weight_const_tensor.is_contiguous(memory_format=torch.channels_last)
                )
                # 断言权重节点的元数据 "val" 的张量在内存中是连续的，内存格式为 torch.channels_last
                self.assertTrue(
                    weight_node.meta["val"].is_contiguous(
                        memory_format=torch.channels_last
                    )
                )

            # 返回编译输出结果 out
            return out

        # 创建一个 Model 实例，并编译为模块 mod，转为评估模式并发送到设备
        mod = torch.compile(
            Model().eval().to(self.device),
            backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
        )
        # 获取模块的示例输入 inp
        inp = mod.get_example_inputs()
        # 在无梯度计算环境中，执行模块 mod 的前向传播
        with torch.no_grad():
            mod(*inp)

        # 仅对 CUDA 设备进行断言检查
        # 对于 CPU，可能会在联合图中得到 torch.ops.mkldnn._convolution_pointwise.default
        # 而不是 torch.ops.aten.convolution.default。目前我们仅在布局优化中处理 aten.convolution.default。
        # 这就是为什么在 CPU 上计数可能为 0 的原因。
        if self.device == "cuda":
            # 断言卷积计数器 nconv 的值为 1
            self.assertTrue(nconv == 1)
    # 定义一个测试函数，测试不同偏置和权重的线性函数融合
    def test_unequal_bias_horizontal_addmm_fusion(self):
        # 获取当前设备
        device = self.device

        # 定义一个神经网络模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 第一层权重和偏置
                self.w1 = torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device
                )
                self.b1 = torch.zeros(3, device=device)
                # 第二层权重和偏置
                self.w2 = torch.tensor(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device
                )
                self.b2 = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
                # 第三层权重和偏置
                self.w3 = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device
                )
                self.b3 = torch.tensor([1.0, 2.0, 3.0], device=device)

            # 前向传播函数
            def forward(self, x):
                # 计算第一层输出
                out1 = torch.nn.functional.linear(x, self.w1, self.b1)
                # 计算第二层输出
                out2 = torch.nn.functional.linear(x, self.w2, self.b2)
                # 计算第三层输出
                out3 = torch.nn.functional.linear(x, self.w3, self.b3)
                # 返回三个层的输出作为元组
                return (out1, out2, out3)

        # 创建模型实例并移动到指定设备上，并设置为评估模式
        func = Model().to(device).eval()
        # 输入数据张量
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)

        # 禁止梯度计算上下文管理器
        with torch.no_grad():
            # 在原始模型上执行前向传播
            out_eager = func(x.clone())

            # 编译模型为一个更快速的版本
            func1 = torch.compile(func)
            # 在编译模型上执行前向传播
            out_compiled = func1(x.clone())
            # 断言两种模式下的输出应该相等
            self.assertEqual(out_eager, out_compiled)

    # 如果是在 ROCm 平台上运行，则跳过当前测试用例
    @skipIfRocm
    # 定义一个测试方法，用于验证在布局转换时多余的克隆操作
    def test_redundant_clone_for_layout_convert(self):
        # 定义一个继承自torch.nn.Module的模型类Model
        class Model(torch.nn.Module):
            # 初始化方法，构建模型的结构
            def __init__(self):
                super().__init__()
                # 添加一个二维卷积层，输入通道数为3，输出通道数为128，核大小为3，填充为1，步长为1，无偏置
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            # 前向传播方法，接收输入x，对其进行加一操作，并返回卷积层输出和加一后的值
            def forward(self, x):
                y = x + 1
                return self.conv(x), y

            # 静态方法，返回一个示例输入，这里返回一个形状为(2, 3, 5, 5)的随机张量，发送到模型当前设备上
            @staticmethod
            def get_example_inputs():
                return (torch.rand(2, 3, 5, 5).to(self.device),)

        # 创建Model的实例，设置为评估模式，并将模型发送到当前设备上
        mod = Model().eval().to(self.device)
        # 获取模型的示例输入
        inp = mod.get_example_inputs()
        # 在上下文中不进行梯度计算
        with torch.no_grad():
            # 计算预期输出
            expected_outputs = mod(*inp)

        # 初始化相同步长和不同步长的计数器
        num_same_stride = 0
        num_diff_stride = 0

        # 定义一个调试函数，用于强制步长顺序，并更新同步步长和不同步步长的计数器
        def debug_inductor_force_stride_order(orig_fn, input_tensor, stride):
            nonlocal num_same_stride, num_diff_stride
            input_tensor.realize()
            # 检查输入张量的步长是否与给定的步长相同
            if tuple(input_tensor.get_stride()) == tuple(stride):
                num_same_stride += 1
            else:
                num_diff_stride += 1
            return orig_fn(input_tensor, stride)

        # 使用override_lowering上下文，替换默认的prims.inductor_force_stride_order函数
        with override_lowering(
            prims.inductor_force_stride_order.default, debug_inductor_force_stride_order
        ):
            # 对模型进行编译优化
            opt_mod = torch.compile(mod)
            # 再次不进行梯度计算
            with torch.no_grad():
                # 计算优化模型的实际输出
                actual_outputs = opt_mod(*inp)

        # 断言优化后的实际输出和预期输出的长度相同
        self.assertEqual(len(actual_outputs), len(expected_outputs))
        # 断言优化后的实际输出长度为2
        self.assertEqual(2, len(actual_outputs))
        # 遍历优化后的实际输出和预期输出，逐个进行数值接近的断言检查
        for i, actual, expected in zip(
            itertools.count(), actual_outputs, expected_outputs
        ):
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"{i}th output: expected {expected}, actual {actual}",
            )

        # 如果当前设备为CPU，跳过以下检查
        if self.device == "cpu":
            return

        # 断言优化后的第一个输出张量是连续的
        self.assertTrue(
            actual_outputs[0].is_contiguous(memory_format=torch.contiguous_format)
        )
        # 断言优化后的第二个输出张量是连续的
        self.assertTrue(
            actual_outputs[1].is_contiguous(memory_format=torch.contiguous_format)
        )

        # 对于forward返回的y，步长不发生改变，因此不会有额外的复制
        self.assertTrue(num_same_stride == 1, f"num_same_stride is {num_same_stride}")
        # 对于forward返回的self.conv(x)，步长发生了改变，可能会有额外的复制
        self.assertTrue(num_diff_stride == 1, f"num_diff_stride is {num_diff_stride}")
# 如果正在使用 ROCm 进行测试
if TEST_WITH_ROCM:
    # 设置 PyTorch 中的 layout 优化强制开启
    torch._inductor.config.force_layout_optimization = 1
    # 设置环境变量，建议使用 NHWC 的 MIOPEN
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"

# 如果系统有 CPU 并且不支持 MPS
if HAS_CPU and not torch.backends.mps.is_available():

    # 定义一个用于 CPU 的测试类
    class FreezingCpuTests(TestCase):
        # 公共属性使用 check_model
        common = check_model
        # 设备类型为 CPU
        device = "cpu"
        # 自动混合精度设置为 CPU 的自动混合精度
        autocast = torch.cpu.amp.autocast

    # 复制 OptimizeForInferenceTemplate 的测试到 FreezingCpuTests 中
    copy_tests(OptimizeForInferenceTemplate, FreezingCpuTests, "cpu")

# 如果系统有 CUDA 并且不是使用 ASAN 进行测试
if HAS_CUDA and not TEST_WITH_ASAN:

    # 定义一个用于 CUDA 的测试类
    class FreezingCudaTests(TestCase):
        # 公共属性使用 check_model_cuda
        common = check_model_cuda
        # 设备类型为 CUDA
        device = "cuda"
        # 自动混合精度设置为 CUDA 的自动混合精度
        autocast = torch.cuda.amp.autocast

    # 复制 OptimizeForInferenceTemplate 的测试到 FreezingCudaTests 中
    copy_tests(OptimizeForInferenceTemplate, FreezingCudaTests, "cuda")

# 删除 OptimizeForInferenceTemplate 的引用
del OptimizeForInferenceTemplate

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 导入 torch._inductor.test_case 中的 run_tests 函数
    from torch._inductor.test_case import run_tests

    # 如果系统有 CPU 或者有 CUDA
    if HAS_CPU or HAS_CUDA:
        # 运行测试，需要 filelock
        run_tests(needs="filelock")
```