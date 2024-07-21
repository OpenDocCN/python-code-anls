# `.\pytorch\test\test_flop_counter.py`

```
# 导入必要的模块和类
import functools
import unittest

import torch
import torch.nn.functional as F
import torch.utils.flop_counter
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    skipIfRocm,
)

# 尝试导入torchvision的models模块，如果失败则设置HAS_TORCHVISION为False
try:
    from torchvision import models as torchvision_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# 根据HAS_TORCHVISION的值决定是否跳过测试
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# 检查CUDA是否可用
HAS_CUDA = torch.cuda.is_available()


# 定义FlopCounterMode函数，返回FlopCounterMode对象
def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


# 定义函数get_total_flops，计算给定模式下的总FLOPs并返回字符串表示
def get_total_flops(mode):
    return str(sum(v for _, v in mode.flop_counts["Global"].items()))


# 定义函数T，返回一个具有指定形状的随机张量，可以设置是否需要梯度
def T(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)


# 使用unittest.skipIf装饰器，根据TEST_WITH_TORCHDYNAMO的值决定是否跳过测试
@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO, "torchdynamo doesn't work with __torch_dispatch__ right now"
)
class TestFlopCounter(TestCase):
    # 测试FlopCounter的多样性
    def test_flop_counter_variety(self):
        # 创建一个线性模型
        mod = torch.nn.Linear(9, 10)
        # 使用FlopCounterMode上下文管理器开始计算FLOPs
        with FlopCounterMode() as mode:
            # 计算两个矩阵的乘积
            torch.mm(T(4, 5), T(5, 6))
            # 计算加权矩阵乘积
            torch.addmm(T(4, 6), T(4, 5), T(5, 6), beta=0.5, alpha=0.5)
            # 计算矩阵乘积
            torch.matmul(T(5, 6), T(6, 7))
            # 使用einsum计算张量乘积
            torch.einsum("ab,bc->ac", T(6, 7), T(7, 8))
            # 对模型进行前向传播
            mod(T(8, 9))

        # 断言计算得到的总FLOPs与预期值相符
        self.assertExpectedInline(get_total_flops(mode), """3012""")
    def test_op(self):
        # 进入 FlopCounterMode 上下文
        with FlopCounterMode() as mode:
            # 计算两个矩阵的乘积
            torch.mm(T(4, 5), T(5, 6))
        # 断言预期的 FLOPs 为 240
        self.assertExpectedInline(get_total_flops(mode), """240""")

        # 继续在相同的上下文中计算另一个操作
        with mode:
            # 计算三维矩阵之间的批次矩阵乘积
            torch.bmm(T(3, 4, 5), T(3, 5, 6))
        # 断言预期的 FLOPs 为 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        # 继续在相同的上下文中执行多个操作
        with mode:
            torch.addmm(T(4, 6), T(4, 5), T(5, 6))
            torch.addmm(T(4, 1), T(4, 5), T(5, 6))
            torch.addmm(T(6), T(4, 5), T(5, 6))
        # 断言预期的 FLOPs 为 720（实际应为 240，注释可能有误）
        self.assertExpectedInline(get_total_flops(mode), """720""")

        # 继续在相同的上下文中执行下一个操作
        with mode:
            torch.baddbmm(T(3, 4, 6), T(3, 4, 5), T(3, 5, 6))
        # 断言预期的 FLOPs 为 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        # 继续在相同的上下文中执行二维卷积操作
        with mode:
            torch.conv2d(T(2, 3, 6, 6), T(6, 3, 4, 4), padding=1)
        # 断言预期的 FLOPs 为 28800
        # 注意：该注释提到可能未正确考虑填充（padding）的影响

        self.assertExpectedInline(get_total_flops(mode), """28800""")

        # 继续在相同的上下文中执行一维卷积操作
        with mode:
            torch.conv1d(T(2, 3, 6), T(6, 3, 4), padding=1)
        # 断言预期的 FLOPs 为 1440
        # 注意：该注释提到可能未正确考虑填充（padding）的影响

        self.assertExpectedInline(get_total_flops(mode), """1440""")

    def test_backward(self):
        # 进入 FlopCounterMode 上下文
        with FlopCounterMode() as mode:
            # 创建一个需要梯度的矩阵 a
            a = T(4, 5, requires_grad=True)
            # 计算矩阵 a 与另一个矩阵的乘积
            a = torch.mm(a, T(5, 6))
            # 扩展矩阵 a 的维度，并进行批次矩阵乘积计算
            a = a.unsqueeze(0).expand(7, 4, 6)
            a = torch.bmm(a, T(7, 6, 7))
            # 对所有元素求和，并进行反向传播
            a.sum().backward()

        # 断言预期的 FLOPs 为 5184
        self.assertExpectedInline(get_total_flops(mode), """5184""")

    def test_backward_reset(self):
        # 进入 FlopCounterMode 上下文
        with FlopCounterMode() as mode:
            # 创建一个需要梯度的矩阵 a，并进行矩阵乘积计算
            a = T(4, 5, requires_grad=True)
            a.mm(a.t()).sum().backward()
            # 再次进行相同操作
            a.mm(a.t()).sum().backward()

        # 断言预期的 FLOPs 为 960
        self.assertExpectedInline(get_total_flops(mode), """960""")

    def test_torchscript(self):
        # 定义一个简单的 TorchScript 函数
        def foo(x):
            return torch.mm(x, x)

        # 进入 FlopCounterMode 上下文
        with FlopCounterMode() as mode:
            # 执行 TorchScript 函数
            foo(T(5, 5))
        # 获取非脚本化版本的 FLOPs
        unscripted_flops = get_total_flops(mode)
        # 将函数 foo 转换为 TorchScript，并在相同上下文中执行
        ts_foo = torch.jit.script(foo)
        with mode:
            ts_foo(T(5, 5))
        # 断言非脚本化和脚本化版本的 FLOPs 相等
        self.assertEqual(unscripted_flops, get_total_flops(mode))
    def test_autograd_op(self):
        # 定义一个自定义的 Torch 自动求导操作类 _CustomOp
        class _CustomOp(torch.autograd.Function):
            # 前向传播函数：计算输入矩阵的自身乘积
            @staticmethod
            def forward(ctx, input: torch.Tensor) -> torch.Tensor:
                return torch.mm(input, input)

            # 反向传播函数：计算梯度输出的自身乘积加上两次梯度输出的自身乘积
            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                return torch.mm(grad_output, grad_output) + torch.mm(
                    grad_output, grad_output
                )

        # 创建一个 Torch 张量 a，大小为 5x5，并要求计算梯度
        a = T(5, 5, requires_grad=True)
        # 进入 FlopCounterMode 上下文
        with FlopCounterMode() as mode:
            # 应用自定义操作 _CustomOp 到张量 a
            a = _CustomOp.apply(a)
            # 计算张量 a 所有元素的和，并进行反向传播
            a.sum().backward()

        # 断言期望的总浮点操作数与实际计算的浮点操作数相符
        self.assertExpectedInline(get_total_flops(mode), """750""")

    def test_conv_backwards_as_decomposition(self):
        # 定义一个 Torch 自动求导函数类 onlyConvs，用于卷积反向传播的分解
        class onlyConvs(torch.autograd.Function):
            # 前向传播函数：根据 transposed 参数选择卷积或转置卷积操作
            @staticmethod
            def forward(inp, weight, transposed):
                if not transposed:
                    return F.conv1d(inp, weight)
                else:
                    return F.conv_transpose1d(inp, weight)

            # 设置上下文：保存输入和权重，并记录是否进行转置操作
            @staticmethod
            def setup_context(ctx, inputs, output):
                inp, weight, transposed = inputs
                ctx.save_for_backward(inp, weight)
                ctx.transposed = transposed

            # 反向传播函数：根据之前保存的信息进行梯度计算
            @staticmethod
            def backward(ctx, grad_out):
                inp, weight = ctx.saved_tensors
                if not ctx.transposed:
                    grad_inp = F.conv_transpose1d(grad_out, weight)
                    grad_weight = F.conv1d(inp, grad_out)
                    return grad_inp, grad_weight, None
                else:
                    grad_inp = F.conv1d(grad_out, weight)
                    grad_weight = F.conv1d(
                        grad_out.transpose(1, 0), inp.transpose(1, 0)
                    )
                    return grad_inp, grad_weight.transpose(1, 0), None

        # 导入 torch 的 grad 函数
        from torch.func import grad

        # 创建随机张量 x 和权重张量 weight
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        weight = torch.randn(3, 4, 4, dtype=torch.float64)

        # 定义普通的卷积计算函数 boring_conv
        def boring_conv(x, weight, transposed):
            if not transposed:
                return F.conv1d(x, weight).pow(2).sum()
            else:
                return F.conv_transpose1d(x, weight).pow(2).sum()

        # 定义基于自定义函数 onlyConvs 的卷积计算函数 only_convs
        def only_convs(x, weight, transposed):
            return onlyConvs.apply(x, weight, transposed).pow(2).sum()

        # 计算普通卷积函数和自定义函数的梯度
        boring_grads = grad(boring_conv, argnums=(0, 1))(x, weight, True)
        fun_grads = grad(only_convs, argnums=(0, 1))(x, weight, True)

        # 断言两种方法计算的梯度相等
        self.assertEqual(boring_grads, fun_grads)
    def test_convs(self):
        # 定义一个辅助函数，用于测试卷积操作的前向和反向传播的等效性
        def assert_equivalence(f, expected_forward=None):
            # 进入 FlopCounterMode 上下文，用于计算操作的浮点运算数（FLOPs）
            with FlopCounterMode() as mode:
                f()
            # 获取卷积操作的前向传播和反向传播的 FLOPs
            conv_forward_flops = mode.get_flop_counts()["Global"][
                torch.ops.aten.convolution
            ]
            conv_backward_flops = mode.get_flop_counts()["Global"][
                torch.ops.aten.convolution_backward
            ]

            # 断言前向传播的 FLOPs 应为反向传播的两倍
            self.assertEqual(conv_forward_flops * 2, conv_backward_flops)
            # 如果提供了预期的前向传播 FLOPs，则验证前向传播的 FLOPs 是否符合预期
            if expected_forward is not None:
                self.assertEqual(conv_forward_flops, expected_forward)

        # 初始化输入张量 x 和卷积核 weight，并进行卷积转置操作测试
        x = torch.rand(1, 1, 2, 2, requires_grad=True)
        weight = torch.randn(1, 1, 2, 2, requires_grad=True)
        # 调用 assert_equivalence 函数，验证 conv_transpose2d 的等效性
        assert_equivalence(lambda: F.conv_transpose2d(x, weight).sum().backward(), 32)

        # 初始化输入张量 x 和卷积核 weight，并进行普通卷积操作测试
        x = torch.rand(1, 1, 2, 2, requires_grad=True)
        weight = torch.randn(1, 1, 1, 1, requires_grad=True)
        # 调用 assert_equivalence 函数，验证 conv2d 的等效性
        assert_equivalence(lambda: F.conv2d(x, weight).sum().backward(), 8)

        # 遍历不同的输入通道数、输出通道数和组数的组合
        for in_channels, out_channels, groups in [
            (1, 1, 1),
            (1, 3, 1),
            (3, 1, 1),
            (3, 7, 1),
            (2, 4, 2),
            (4, 2, 2),
        ]:
            # 初始化输入张量 x 和卷积核 weight，并进行普通卷积操作测试
            x = torch.rand(1, in_channels, 4, 4, requires_grad=True)
            weight = torch.randn(out_channels, in_channels, 2, 2, requires_grad=True)
            # 调用 assert_equivalence 函数，验证 conv2d 的等效性
            assert_equivalence(lambda: F.conv2d(x, weight).sum().backward())
            # 初始化转置卷积核 transposed_weight，并进行卷积转置操作测试
            transposed_weight = torch.randn(
                in_channels, out_channels, 2, 2, requires_grad=True
            )
            # 调用 assert_equivalence 函数，验证 conv_transpose2d 的等效性
            assert_equivalence(
                lambda: F.conv_transpose2d(x, transposed_weight).sum().backward()
            )

    @skipIfNoTorchVision
    def test_module(self):
        # 创建 ResNet-18 模型实例 resnet18
        resnet18 = torchvision_models.resnet18()
        # 进入 FlopCounterMode 上下文，计算模型的 FLOPs
        with FlopCounterMode(resnet18) as mode:
            # 创建输入张量 a，并进行模型前向传播及反向传播
            a = T(1, 3, 224, 224, requires_grad=True)
            resnet18(a).sum().backward()

        # 断言模型总的 FLOPs 数量符合预期值
        self.assertExpectedInline(get_total_flops(mode), """10884440064""")
        # 获取 ResNet 第一层卷积的 FLOPs 数量
        layer1_conv_flops = mode.flop_counts["ResNet.layer1"][
            torch.ops.aten.convolution
        ]
        # 获取 ResNet 第一层卷积反向传播的 FLOPs 数量
        layer1_conv_back_flops = mode.flop_counts["ResNet.layer1"][
            torch.ops.aten.convolution_backward
        ]
        # 断言第一层卷积的 FLOPs 数量符合预期值
        self.assertExpectedInline(str(layer1_conv_flops), """924844032""")
        # 断言第一层卷积反向传播的 FLOPs 数量符合预期值
        self.assertExpectedInline(str(layer1_conv_back_flops), """1849688064""")

    def test_conv_transpose_loop(self):
        # 初始化输入张量 x 和转置卷积模型 model
        x = torch.rand(1, 4, 30, 2)
        model = torch.nn.ConvTranspose2d(4, 8, (2, 2), stride=2)

        # 进入 FlopCounterMode 上下文，计算循环中的总 FLOPs
        with FlopCounterMode() as mode:
            for i in range(50):
                # 对输入张量 x 进行转置卷积操作，并进行前向传播及反向传播
                out = model(x)
                out.sum().backward()
        # 断言循环中的总 FLOPs 数量符合预期值
        self.assertExpectedInline(str(mode.get_total_flops()), """1536000""")
    def test_custom(self):
        # 创建一个 FlopCounterMode 对象，使用自定义映射来重定义 torch.ops.aten.add 的计算代价为 5
        mode = FlopCounterMode(
            custom_mapping={torch.ops.aten.add: lambda *args, out_shape: 5}
        )
        # 进入计数模式
        with mode:
            # 创建一个大小为 (4, 5) 的张量 a
            a = T(4, 5)
            # 对张量 a 执行加法操作 a + a
            a + a

        # 断言计算得到的总浮点操作数与预期结果 "5" 相符
        self.assertExpectedInline(get_total_flops(mode), """5""")

        # 定义一个计数函数 count，接受参数 *args 和 out_val
        def count(*args, out_val):
            # 返回输出值 out_val 的元素个数作为计数结果
            return out_val.numel()

        # 将 _get_raw 标志设置为 True
        count._get_raw = True

        # 创建另一个 FlopCounterMode 对象，使用自定义映射来重定义 torch.ops.aten.add 的计算代价为 count 函数
        mode = FlopCounterMode(custom_mapping={torch.ops.aten.add: count})
        # 进入计数模式
        with mode:
            # 创建一个大小为 (4, 5) 的张量 a
            a = T(4, 5)
            # 对张量 a 执行加法操作 a + a
            a + a

        # 断言计算得到的总浮点操作数与预期结果 "20" 相符
        self.assertExpectedInline(get_total_flops(mode), """20""")

    def test_noop(self):
        # 进入 FlopCounterMode 模式
        with FlopCounterMode() as mode:
            # 创建一个大小为 (4, 5) 的张量，调用其 cos 方法
            T(4, 5).cos()

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION
        or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    @skipIfRocm  # Nested tensor
    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION
        or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Does not support all SDPA backends (pre-SM80 hardware on CUDA)",
    )
    def test_addmm_out(self):
        # 定义一个函数 f，接受输入 x
        def f(x):
            # 创建一个大小为 (10, 10) 的全零张量 y
            y = torch.zeros(10, 10)
            # 执行矩阵乘法 x * x，并将结果存入 y
            return torch.mm(x, x, out=y)

        # 进入 FlopCounterMode 模式
        with FlopCounterMode() as mode:
            # 调用函数 f，传入大小为 (10, 10) 的随机张量作为参数
            f(torch.randn(10, 10))

        # 断言计算得到的总浮点操作数与预期结果 "2000" 相符
        self.assertExpectedInline(get_total_flops(mode), """2000""")

    def test_hook_registration(self):
        # 创建一个大小为 (100, 100) 的线性模型
        model = torch.nn.Linear(100, 100)
        # 创建一个大小为 (3, 100) 的随机张量 x
        x = torch.randn(3, 100)

        # 进入 FlopCounterMode 模式
        with FlopCounterMode() as mode:
            # 断言全局前向钩子数量为 1
            self.assertEqual(len(torch.nn.modules.module._global_forward_pre_hooks), 1)
            # 断言全局后向钩子数量为 1
            self.assertEqual(len(torch.nn.modules.module._global_forward_hooks), 1)
            # 对模型输入 x 进行前向计算，并对结果求和再反向传播
            model(x).sum().backward()

        # 断言退出计数模式后，全局前向钩子数量为 0
        self.assertEqual(len(torch.nn.modules.module._global_forward_pre_hooks), 0)
        # 断言退出计数模式后，全局后向钩子数量为 0
        self.assertEqual(len(torch.nn.modules.module._global_forward_hooks), 0)
    def test_pytrees(self):
        # 定义一个继承自torch.nn.Module的子类Foo，重载了forward方法
        class Foo(torch.nn.Module):
            def forward(self, x):
                # 从输入x中获取键为"a"的张量，然后应用ReLU函数
                x = x["a"].relu_()
                # 返回一个字典，包含键"a"对应的张量与其自身的矩阵乘积
                return {"a": torch.mm(x, x)}

        # 定义一个继承自torch.nn.Module的子类Mod，重载了__init__和forward方法
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化两个Foo类的实例a和b
                self.a = Foo()
                self.b = Foo()

            def forward(self, x):
                # 调用实例b的forward方法，并传入实例a对输入x的处理结果
                return self.b(self.a(x))

        # 创建Mod类的一个实例mod
        mod = Mod()
        # 进入FlopCounterMode上下文
        with FlopCounterMode() as mode:
            # 调用mod实例的forward方法，传入包含"a"键的随机张量，并进行处理
            # 然后对处理后的结果中的"a"键对应的张量求和，并执行反向传播
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})["a"].sum().backward()
        # 使用self.assertExpectedInline断言mode.flop_counts["Mod"][torch.ops.aten.mm]为12000
        self.assertExpectedInline(
            (mode.flop_counts["Mod"][torch.ops.aten.mm]), """12000"""
        )

        # 定义一个继承自torch.nn.Module的子类Mod2，重载了forward方法
        class Mod2(torch.nn.Module):
            def forward(self, x):
                # 返回x与其自身的矩阵乘积组成的元组
                return (torch.mm(x, x),)

        # 创建Mod2类的一个实例mod
        mod = Mod2()
        # 进入FlopCounterMode上下文
        with FlopCounterMode() as mode:
            # 调用mod实例的forward方法，传入随机张量，并对其进行处理
            # 然后对处理后结果中的第一个张量（即torch.mm(x, x)的结果）求和，并执行反向传播
            mod(torch.randn(10, 10, requires_grad=True))[0].sum().backward()
        # 使用self.assertExpectedInline断言mode.flop_counts["Mod2"][torch.ops.aten.mm]为6000
        self.assertExpectedInline(
            (mode.flop_counts["Mod2"][torch.ops.aten.mm]), """6000"""
        )

    def test_warning(self):
        # 创建一个torch.nn.Linear对象mod，输入维度为2，输出维度为2
        mod = torch.nn.Linear(2, 2)
        # 使用self.assertWarnsRegex断言在执行FlopCounterMode(mod)时会发出UserWarning警告，
        # 警告信息包含字符串"not needed"
        with self.assertWarnsRegex(UserWarning, "not needed"):
            FlopCounterMode(mod)
# 如果当前脚本被直接执行（而不是被导入到其它模块中），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests() 的函数，用于执行测试或其他逻辑
    run_tests()
```