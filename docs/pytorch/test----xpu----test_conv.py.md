# `.\pytorch\test\xpu\test_conv.py`

```
# Owner(s): ["module: intel"]

# 导入必要的库和模块
import itertools  # 导入 itertools 模块，用于迭代器操作
import math  # 导入 math 模块，提供数学函数
import unittest  # 导入 unittest 模块，用于单元测试
from itertools import product  # 从 itertools 模块导入 product 函数，用于迭代器操作

import torch  # 导入 PyTorch 库
import torch.backends.cudnn as cudnn  # 导入 PyTorch 的 cuDNN 后端
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的函数形式的神经网络模块
from torch.testing import make_tensor  # 从 torch.testing 模块导入 make_tensor 函数
from torch.testing._internal.common_cuda import tf32_is_not_fp32  # 从 common_cuda 模块导入 tf32_is_not_fp32 函数
from torch.testing._internal.common_device_type import (  # 从 common_device_type 模块导入以下对象
    dtypes,  # 数据类型
    instantiate_device_type_tests,  # 设备类型测试实例化
    onlyXPU,  # 仅限 XPU
)
from torch.testing._internal.common_dtype import floating_types_and  # 从 common_dtype 模块导入 floating_types_and 函数
from torch.testing._internal.common_nn import (  # 从 common_nn 模块导入以下对象
    _test_module_empty_input,  # 空输入测试模块
    NNTestCase,  # 神经网络测试用例基类
)
from torch.testing._internal.common_utils import (  # 从 common_utils 模块导入以下对象
    dtype2prec_DONTUSE,  # 数据类型到精度的映射（不要使用）
    gradcheck,  # 梯度检查函数
    gradgradcheck,  # 二阶梯度检查函数
    parametrize as parametrize_test,  # 参数化测试装饰器
    run_tests,  # 运行测试函数
    set_default_dtype,  # 设置默认数据类型函数
    TEST_SCIPY,  # 是否测试 SCIPY
    TEST_WITH_ROCM,  # 是否在 ROCM 上测试
)

# 导入 torch._C._dynamo.guards.assert_size_stride 函数并重命名为 assert_size_stride
assert_size_stride = torch._C._dynamo.guards.assert_size_stride

# 根据测试条件设置 AMPERE_OR_ROCM 变量
AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# 如果 TEST_SCIPY 为真，则导入 scipy.ndimage 和 scipy.signal 模块
if TEST_SCIPY:
    import scipy.ndimage  # 导入 scipy.ndimage 模块
    import scipy.signal  # 导入 scipy.signal 模块

# 定义一个测试类 TestConvolutionNNDeviceType，继承自 NNTestCase
class TestConvolutionNNDeviceType(NNTestCase):

    # 定义一个方法 run_conv_double_back_test，用于测试卷积的双向传播
    def run_conv_double_back_test(
        self,
        kern,
        stride,
        padding,
        chan_in,
        chan_out,
        batch_size,
        inp_size,
        dilation,
        no_weight,
        groups=1,
        use_xpu=False,
        use_bias=True,
        dtype=torch.double,
    ):
        # 根据 use_xpu 的值选择设备类型
        device = torch.device("xpu" if use_xpu else "cpu")
        
        # 生成随机输入张量 x，指定设备和数据类型，并声明需要梯度
        x = torch.randn(
            batch_size,
            chan_in,
            inp_size,
            inp_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        
        # 生成随机权重张量 weight，指定设备和数据类型，并声明是否需要梯度
        weight = torch.randn(
            chan_out,
            chan_in // groups,
            kern,
            kern,
            device=device,
            dtype=dtype,
            requires_grad=not no_weight,
        )
        
        # 如果使用偏置，则生成随机偏置张量 bias，指定设备和数据类型，并声明需要梯度
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
        else:
            bias = None

        # 定义一个函数 func，接受多个输入参数，进行卷积操作，并返回结果
        def func(*inputs):
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        # 根据是否使用偏置，选择相应的输入参数
        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        # 调用 func 函数获取输出结果
        dummy_out = func(*inputs)
        
        # 生成与 dummy_out 相同大小的随机梯度张量 grad_y，指定设备和数据类型，并声明需要梯度
        grad_y = torch.randn_like(
            dummy_out, device=device, dtype=dtype, requires_grad=True
        )

        # 如果数据类型为 torch.float，则进行一阶梯度计算
        if dtype == torch.float:
            (g,) = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad
        
        # 否则进行二阶梯度检查
        return gradgradcheck(func, inputs, (grad_y,))

    # 参数化测试函数，参数为所有浮点类型和 torch.half、torch.bfloat16
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    # 定义一个测试函数，用于测试 Conv2d 对象在大工作空间条件下的行为
    def test_Conv2d_large_workspace(self, device, dtype):
        # 定义输入数据的大小列表
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        # 内部函数，运行 Conv2d 的测试
        def run_test(benchmark):
            # 创建一个 Conv2d 对象，设置输入通道数和输出通道数均为256，卷积核大小为3，padding为1
            conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to(device, dtype)
            # 遍历输入大小列表
            for size in sizes:
                # 生成指定大小的随机输入张量 x，放置在指定的设备上，使用指定的数据类型
                x = torch.randn(size, device=device, dtype=dtype)
                # 对输入张量进行克隆并分离，允许梯度计算
                out = conv(x.detach().clone().requires_grad_())
                # 计算输出的反向传播梯度
                out.backward(torch.ones_like(out))

        # 非基准测试运行
        run_test(benchmark=False)
        # 基准测试运行
        run_test(benchmark=True)

    # 使用指定数据类型进行类型注释，测试 ConvTranspose2d 对象在大输出填充条件下的行为
    @dtypes(torch.half, torch.float)
    def test_ConvTranspose2d_large_output_padding(self, device, dtype):
        # 创建三个 ConvTranspose2d 对象，分别从128通道到64通道，从64通道到32通道，从32通道到3通道
        net1 = torch.nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net2 = torch.nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net3 = torch.nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        # 创建一个随机输入张量 x，形状为 [1, 128, 6, 6]，放置在指定设备上，使用指定数据类型，允许梯度计算
        x = torch.rand(1, 128, 6, 6, device=device, dtype=dtype, requires_grad=True)
        # 依次执行三个 ConvTranspose2d 对象的前向传播
        x = net1(x)
        x = net2(x)
        x = net3(x)
        # 对最终输出张量 x 计算反向传播梯度
        x.backward(torch.randn_like(x))

    # 使用指定数据类型进行类型注释，测试 ConvTranspose2d 对象在浮点、双精度浮点和半精度浮点数下的行为
    @dtypes(torch.float, torch.double, torch.half)
    # 定义一个测试函数，测试深度可分离卷积（depthwise convolution）的简单组合
    def test_Conv2d_depthwise_naive_groups(self, device, dtype):
        # 如果数据类型是半精度浮点数且设备包含"xpu"，则跳过此测试
        if dtype == torch.half and "xpu" in device:
            self.skipTest(
                "The accuracy issue of dtype fp16 would be fixed in oneDNN v3.4"
            )
        
        # 遍历深度倍增因子的可能取值
        for depth_multiplier in [1, 2]:
            # 创建一个深度可分离卷积层对象
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            # 创建一个输入张量 i，并进行缩放和设置梯度
            i = (
                torch.randn(2, 2, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            # 前向传播
            output = m(i)
            # 创建一个梯度输出张量
            grad_output = (
                torch.randn(2, 2 * depth_multiplier, 4, 4, device=device, dtype=dtype)
                / 2
            )
            # 反向传播
            output.backward(grad_output)
            
            # 计算偏移量
            offset = 1 * depth_multiplier
            
            # 创建第一个普通卷积层对象
            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 复制部分权重和偏置
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            # 分离输入张量 i 的部分数据，并设置梯度
            i1 = i.detach()[:, :1].clone().requires_grad_()
            # 第一个卷积层的前向传播
            output1 = m1(i1)
            # 第一个卷积层的反向传播
            output1.backward(grad_output[:, :offset].contiguous())
            
            # 创建第二个普通卷积层对象
            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 复制剩余的权重和偏置
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            # 分离输入张量 i 的剩余数据，并设置梯度
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            # 第二个卷积层的前向传播
            output2 = m2(i2)
            # 第二个卷积层的反向传播
            output2.backward(grad_output[:, offset:].contiguous())
            
            # 断言：验证总输出与两个部分输出的连接是否一致
            self.assertEqual(
                output,
                torch.cat([output1, output2], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 断言：验证输入张量的梯度是否与两个部分梯度的连接一致
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 断言：验证卷积层偏置的梯度是否与两个部分偏置梯度的连接一致
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 断言：验证卷积层权重的梯度是否与两个部分权重梯度的连接一致
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    @dtypes(torch.float, torch.double, torch.half)
    # 定义一个测试函数，测试深度卷积的分组操作
    def test_Conv3d_depthwise_naive_groups(self, device, dtype):
        # 如果数据类型是半精度且设备是 "xpu"，则跳过此测试
        if dtype == torch.half and "xpu" in device:
            self.skipTest(
                "The accuracy issue of dtype fp16 would be fixed in oneDNN v3.4"
            )
        
        # 遍历深度乘数列表 [1, 2]
        for depth_multiplier in [1, 2]:
            # 创建一个 3D 卷积层对象，输入通道数为 2，输出通道数为 2 * depth_multiplier，卷积核大小为 3，分组数为 2
            m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            # 创建一个随机输入张量 i，形状为 (2, 2, 6, 6, 6)，在指定设备上，指定数据类型，除以 2 并设置为需要梯度
            i = (
                torch.randn(2, 2, 6, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            # 将输入张量 i 输入到卷积层 m 中，得到输出张量 output
            output = m(i)
            # 创建一个随机梯度输出张量 grad_output，形状为 (2, 2 * depth_multiplier, 4, 4, 4)，在指定设备上，指定数据类型
            grad_output = (
                torch.randn(
                    2, 2 * depth_multiplier, 4, 4, 4, device=device, dtype=dtype
                )
                / 2
            )
            # 对输出张量 output 进行反向传播，使用 grad_output 作为梯度
            output.backward(grad_output)

            # 设置偏移量为 1 * depth_multiplier
            offset = 1 * depth_multiplier

            # 创建第一个分组卷积层 m1，输入通道数为 1，输出通道数为 1 * depth_multiplier，卷积核大小为 3，放置在指定设备上，指定数据类型
            m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 将 m 的权重数据的前 offset 部分克隆到 m1 的权重数据
            m1.weight.data = m.weight.data[:offset].clone()
            # 将 m 的偏置数据的前 offset 部分克隆到 m1 的偏置数据
            m1.bias.data = m.bias.data[:offset].clone()
            # 从 i 中分离出第一部分，保留梯度信息，并克隆到 i1
            i1 = i.detach()[:, :1].clone().requires_grad_()
            # 将 i1 输入到 m1 中，得到输出张量 output1
            output1 = m1(i1)
            # 对 output1 进行反向传播，使用 grad_output 的前 offset 部分作为梯度，保证数据连续性
            output1.backward(grad_output[:, :offset].contiguous())

            # 创建第二个分组卷积层 m2，输入通道数为 1，输出通道数为 1 * depth_multiplier，卷积核大小为 3，放置在指定设备上，指定数据类型
            m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 将 m 的权重数据的后 offset 部分复制到 m2 的权重数据
            m2.weight.data.copy_(m.weight.data[offset:])
            # 将 m 的偏置数据的后 offset 部分复制到 m2 的偏置数据
            m2.bias.data.copy_(m.bias.data[offset:])
            # 从 i 中分离出第二部分，保留梯度信息，并克隆到 i2
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            # 将 i2 输入到 m2 中，得到输出张量 output2
            output2 = m2(i2)
            # 对 output2 进行反向传播，使用 grad_output 的后 offset 部分作为梯度，保证数据连续性
            output2.backward(grad_output[:, offset:].contiguous())
            
            # 设置绝对容差 atol 和相对容差 rtol
            atol, rtol = (3e-4, 3e-2)

            # 断言 output 等于 torch.cat([output1, output2], 1)，使用指定的绝对容差和相对容差
            self.assertEqual(
                output, torch.cat([output1, output2], 1), atol=atol, rtol=rtol
            )
            # 断言输入张量 i 的梯度数据等于 torch.cat([i1.grad.data, i2.grad.data], 1)，使用指定的绝对容差
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 断言卷积层 m 的偏置梯度数据等于 torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0)，使用指定的绝对容差
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 断言卷积层 m 的权重梯度数据等于 torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0)，使用指定的绝对容差和相对容差
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=atol,
                rtol=rtol,
            )
    # 测试非连续的卷积梯度计算
    def test_noncontig_conv_grad(self, device, dtype):
        # 创建一个卷积层模型，输入通道数为3，输出通道数为5，卷积核大小为3，填充为1，转移到指定设备上
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device, dtype)
        # 创建一个随机输入张量，形状为[2, 3, 10, 10]，指定数据类型和设备，并设置需要计算梯度
        input = torch.randn(
            2, 3, 10, 10, dtype=dtype, device=device, requires_grad=True
        )
        # 将输入张量输入到卷积层中，得到输出张量
        output = module(input)

        # 创建一个随机梯度张量，形状为[2, 2, 5, 10, 10]，不连续
        grad = torch.randn(2, 2, 5, 10, 10, dtype=dtype, device=device)[:, 1]
        # 断言梯度张量不是连续的
        assert not grad.is_contiguous()
        # 反向传播梯度到输入张量，保留计算图
        output.backward(grad, retain_graph=True)
        # 断言输入张量的梯度不为None
        self.assertIsNotNone(input.grad)
        # 克隆梯度张量数据作为结果
        result = input.grad.data.clone()
        # 将输入张量的梯度数据清零
        input.grad.data.zero_()

        # 再次使用连续的梯度张量进行反向传播
        output.backward(grad.contiguous())
        # 断言克隆的结果与输入张量的梯度数据在一定误差范围内相等
        self.assertEqual(
            result, input.grad.data, atol=dtype2prec_DONTUSE[dtype], rtol=0
        )

    # 使用torch.double数据类型进行测试
    @dtypes(torch.double)
    def test_conv_double_backward(self, device, dtype):
        # 设置使用cudnn加速，确定性为真
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 遍历不同的卷积参数组合进行测试
            batch_size = 1
            for kern, inp_size, dilations in [(3, 5, [1, 2]), (4, 9, [1])]:
                for stride, padding, chan_in, chan_out, dilation in product(
                    [1], [2], [2], [3], dilations
                ):
                    # 判断是否不使用权重
                    no_weight = stride == 2
                    # 运行卷积双向梯度测试，并获取测试结果
                    result = self.run_conv_double_back_test(
                        kern,
                        stride,
                        padding,
                        chan_in,
                        chan_out,
                        batch_size,
                        inp_size,
                        dilation,
                        no_weight,
                        use_xpu=True,
                        dtype=dtype,
                    )
                    # 断言测试结果为真，否则输出测试失败信息
                    self.assertTrue(result, "Conv double backward test failed")

    # 测试不使用偏置的卷积双向梯度计算
    def test_conv_double_backward_no_bias(self):
        kern, stride = 3, 2
        chan_in, chan_out = 2, 4
        batch_size, inp_size = 2, 5
        padding, dilation = 1, 1
        no_weight, use_bias = False, True
        # 运行卷积双向梯度测试，并获取测试结果
        result = self.run_conv_double_back_test(
            kern,
            stride,
            padding,
            chan_in,
            chan_out,
            batch_size,
            inp_size,
            dilation,
            no_weight,
            use_bias=use_bias,
        )
        # 断言测试结果为真，否则输出测试失败信息
        self.assertTrue(result, "Conv double backward test failed")

    # 测试使用分组卷积的卷积双向梯度计算
    def test_conv_double_backward_groups(self):
        kern, stride, padding = 3, 1, 2
        chan_in, chan_out = 2, 4
        batch_size, inp_size, dilation = 2, 6, 1
        no_weight = False
        groups = 2
        # 运行卷积双向梯度测试，并获取测试结果
        result = self.run_conv_double_back_test(
            kern,
            stride,
            padding,
            chan_in * groups,
            chan_out * groups,
            batch_size,
            inp_size,
            dilation,
            no_weight,
            groups=groups,
        )
        # 断言测试结果为真，否则输出测试失败信息
        self.assertTrue(result, "Conv double backward test failed")
    # 定义一个测试方法，用于测试卷积层的双向反向传播，并设定批量大小为2
    def test_conv_double_backward_stride(self):
        # 针对不同的卷积核大小、输入大小和空洞率进行循环测试
        batch_size = 2
        for kern, inp_size, dilations in [(3, 5, [1, 2]), (3, 7, [1])]:
            for stride, padding, chan_in, chan_out, dilation in product(
                [2], [0, 1], [1], [2], dilations
            ):
                # 设置是否有权重的标志为False
                no_weight = False
                # 调用自定义方法进行卷积层双向反向传播测试
                self.run_conv_double_back_test(
                    kern,
                    stride,
                    padding,
                    chan_in,
                    chan_out,
                    batch_size,
                    inp_size,
                    dilation,
                    no_weight,
                )

    # 使用torch.float作为数据类型装饰器
    @dtypes(torch.float)
    def test_conv1d_same_padding(self, device, dtype):
        # 定义测试参数的组合列表
        test_args = [
            range(50, 55),
            [1, 2, 3, 8],
            range(1, 4),
            [1],
        ]
        # 针对所有测试参数的组合进行迭代测试
        for in_size, k_size, dilation, stride in itertools.product(*test_args):
            # 创建随机张量x和y，并进行1维卷积计算，使用'same'填充，指定空洞率和步长
            x = torch.rand(1, 1, in_size, device=device, dtype=dtype)
            y = torch.rand(1, 1, k_size, device=device, dtype=dtype)
            z = F.conv1d(x, y, padding="same", dilation=dilation, stride=stride)
            # 断言计算结果的大小符合预期
            self.assertEqual(z.size(2), int(math.ceil(in_size / stride)))

        # 针对不同的张量大小和填充方式进行期望和实际的1维卷积计算，使用相应的填充方式
        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 3, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=1)
        actual = F.conv1d(x, y, padding="same")
        # 断言期望结果与实际结果相等
        self.assertEqual(expect, actual)

        # 针对不同的张量大小和空洞率进行期望和实际的1维卷积计算，使用相应的填充方式和空洞率
        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=3,
    # 定义一个测试函数，用于测试一维卷积操作的有效填充
    def test_conv1d_valid_padding(self, device, dtype):
        # 创建一个形状为 (1, 1, 10) 的随机张量 x，放置在指定设备上，并指定数据类型
        x = torch.rand(1, 1, 10, device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 4) 的随机张量 y，放置在指定设备上，并指定数据类型
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        # 使用 torch.nn.functional 中的 conv1d 函数对 x 和 y 进行一维卷积，返回期望的结果张量 expect
        expect = F.conv1d(x, y)
        # 使用 torch.nn.functional 中的 conv1d 函数对 x 和 y 进行一维卷积，使用有效填充方式，得到实际的结果张量 actual
        actual = F.conv1d(x, y, padding="valid")
        # 断言期望的结果张量和实际的结果张量相等
        self.assertEqual(expect, actual)

    # 使用 torch.testing 模块中的 dtypes 装饰器，定义一个测试函数，用于测试二维卷积操作的有效填充
    @dtypes(torch.float)
    def test_conv2d_valid_padding(self, device, dtype):
        # 创建一个形状为 (1, 1, 1, 10) 的随机张量 x，放置在指定设备上，并指定数据类型
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 1, 4) 的随机张量 y，放置在指定设备上，并指定数据类型
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype)
        # 使用 torch.nn.functional 中的 conv2d 函数对 x 和 y 进行二维卷积，返回期望的结果张量 expect
        expect = F.conv2d(x, y)
        # 使用 torch.nn.functional 中的 conv2d 函数对 x 和 y 进行二维卷积，使用有效填充方式，得到实际的结果张量 actual
        actual = F.conv2d(x, y, padding="valid")
        # 断言期望的结果张量和实际的结果张量相等
        self.assertEqual(expect, actual)

    # 使用 torch.testing 模块中的 dtypes 装饰器，定义一个测试函数，用于测试三维卷积操作的有效填充
    @dtypes(torch.float)
    def test_conv3d_valid_padding(self, device, dtype):
        # 创建一个形状为 (1, 1, 1, 1, 10) 的随机张量 x，放置在指定设备上，并指定数据类型
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device)
        # 创建一个形状为 (1, 1, 1, 1, 4) 的随机张量 y，放置在指定设备上，并指定数据类型
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device)
        # 使用 torch.nn.functional 中的 conv3d 函数对 x 和 y 进行三维卷积，返回期望的结果张量 expect
        expect = F.conv3d(x, y)
        # 使用 torch.nn.functional 中的 conv3d 函数对 x 和 y 进行三维卷积，使用有效填充方式，得到实际的结果张量 actual
        actual = F.conv3d(x, y, padding="valid")
        # 断言期望的结果张量和实际的结果张量相等
        self.assertEqual(expect, actual)

    # 使用 torch.testing 模块中的 dtypes 装饰器，定义一个测试函数，用于测试一维卷积操作的相同填充和反向传播
    @dtypes(torch.float)
    def test_conv1d_same_padding_backward(self, device, dtype):
        # 创建一个形状为 (1, 1, 12) 的随机张量 x，放置在指定设备上，并指定数据类型，要求梯度计算
        x = torch.rand(1, 1, 12, dtype=dtype, device=device, requires_grad=True)
        # 创建一个形状为 (1, 1, 4) 的随机张量 y，放置在指定设备上，并指定数据类型，要求梯度计算
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        # 使用 torch.nn.functional 中的 conv1d 函数对 x 和 y 进行一维卷积，使用特定填充和膨胀，得到结果张量 z
        z = F.conv1d(x, y, padding=3, dilation=2)
        # 对结果张量 z 进行求和并取绝对值，然后进行反向传播
        z.sum().abs().backward()
        # 保存预期的 x 和 y 梯度
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 使用 torch.nn.functional 中的 conv1d 函数对 x 和 y 进行一维卷积，使用相同填充和膨胀，得到结果张量 z
        z = F.conv1d(x, y, padding="same", dilation=2)
        # 对结果张量 z 进行求和并取绝对值，然后进行反向传播
        z.sum().abs().backward()
        # 断言预期的 x 和 y 梯度与实际计算得到的梯度相等
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # 使用 torch.nn.functional 中的 conv1d 函数对 x 和 y 进行一维卷积，使用特定填充后取子集，得到结果张量 z
        z = F.conv1d(x, y, padding=2)[..., 1:]
        # 对结果张量 z 进行求和并取绝对值，然后进行反向传播
        z.sum().abs().backward()
        # 保存预期的 x 和 y 梯度
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 使用 torch.nn.functional 中的 conv1d 函数对 x 和 y 进行一维卷积，使用相同填充，得到结果张量 z
        z = F.conv1d(x, y, padding="same")
        # 对结果张量 z 进行求和并取绝对值，然后进行反向传播
        z.sum().abs().backward()
        # 断言预期的 x 和 y 梯度与实际计算得到的梯度相等
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    # 使用 torch.testing 模块中的 dtypes 装饰器，定义一个测试函数，用于测试二维卷积操作的相同填充和反向传播
    @dtypes(torch.float)
    def test_conv2d_same_padding_backward(self, device, dtype):
        # 创建一个形状为 (1, 1, 10, 11) 的随机张量 x，放置在指定设备上，并指定数据类型，要求梯度计算
        x = torch.rand(1, 1, 10, 11, device=device, dtype=dtype, requires_grad=True)
        # 创建一个形状为 (1, 1, 4, 5) 的随机张量 y，放置在指定设备上，并指定数据类型，要求梯度计算
        y = torch.rand(1, 1, 4, 5, device=device, dtype=dtype, requires_grad=True)

        # 使用 torch.nn.functional 中的 conv2d 函数对 x 和 y 进行二维卷积，
    # 使用装饰器指定该函数的参数类型为 torch.double
    @dtypes(torch.double)
    # 定义一个测试函数，用于测试 3D 卷积的“same”填充反向传播
    def test_conv3d_same_padding_backward(self, device, dtype):
        # 创建输入张量 x 和 y，均为随机值，需计算梯度
        x = torch.rand(1, 1, 1, 11, 12, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 2, 5, dtype=dtype, device=device, requires_grad=True)
        # 进行 3D 卷积，指定填充和扩展参数，计算输出张量 z
        z = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        # 对 z 中所有元素求和并取绝对值，然后进行反向传播
        z.sum().abs().backward()
        # 保存 x 和 y 的梯度期望值
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用 "same" 填充参数再次进行 3D 卷积，并计算梯度
        z = F.conv3d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        # 断言当前梯度与之前保存的期望梯度相等
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None
        # 使用 gradcheck 函数验证 3D 卷积的梯度计算是否正确
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
            (x, y),
            check_forward_ad=True,  # 检查前向传播自动微分
            nondet_tol=1e-5,  # 非确定性容差值
        )
        # 使用 gradgradcheck 函数验证 3D 卷积的二阶梯度计算是否正确
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
            (x, y),
            check_fwd_over_rev=True,  # 检查前向传播和反向传播之间的一致性
        )

        # 更新 y 为新的随机张量，并修改填充参数
        y = torch.rand(1, 1, 1, 4, 4, dtype=dtype, device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=2)[..., 1:, 1:]  # 对计算的 z 进行切片操作
        z.sum().abs().backward()
        # 保存新的梯度期望值
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用 "same" 填充参数再次进行 3D 卷积，并计算梯度
        z = F.conv3d(x, y, padding="same")
        z.sum().abs().backward()
        # 断言当前梯度与之前保存的期望梯度相等
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        # 使用 gradcheck 函数验证 3D 卷积的梯度计算是否正确
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same"),
            (x, y),
            check_forward_ad=True,  # 检查前向传播自动微分
            nondet_tol=1e-5,  # 非确定性容差值
        )
        # 使用 gradgradcheck 函数验证 3D 卷积的二阶梯度计算是否正确
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="same"),
            (x, y),
            check_fwd_over_rev=True,  # 检查前向传播和反向传播之间的一致性
        )
    # 定义测试方法，用于比较一维卷积函数与 Scipy 库的效果
    def test_conv1d_vs_scipy(self, device, dtype, mode):
        # 创建指定设备和数据类型的张量，形状为 (1, 10)
        t = make_tensor((1, 10), device=device, dtype=dtype)
        # 获取张量的特征维度
        feat_dim = t.shape[1]
        # 创建指定设备和数据类型的权重张量，形状为 (1, 1, 4) 和 (1, 1, 5)
        weight_even = make_tensor((1, 1, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 5), device=device, dtype=dtype)

        # 定义内部测试方法，用于执行具体的卷积比较操作
        def _test(t, weight, mode):
            # 将输入张量 t 和权重 weight 展平并转换为 NumPy 数组
            t_a = t.view(-1).cpu().numpy()
            w_a = weight.view(-1).cpu().numpy()
            # 使用 Scipy 库进行卷积操作，返回期望的结果
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            # 初始化卷积参数字典
            kwargs = {"padding": mode}
            # 如果模式为 "same"，则计算需要填充的数量并在输入张量上进行填充操作
            if mode == "same":
                p = weight.shape[2] // 2
                t = torch.nn.functional.pad(t, (p, p))
                # 删除 kwargs 中的 "padding" 键
                kwargs.pop("padding")

            # 反转权重张量并执行 PyTorch 的一维卷积操作，得到实际结果
            weight_flipped = torch.flip(weight, (2,))
            actual = torch.nn.functional.conv1d(t, weight_flipped, **kwargs).squeeze(0)
            # 如果模式为 "same"，则截取与特征维度相同的部分
            if mode == "same":
                actual = actual[:feat_dim]

            # 使用断言比较实际结果与期望结果的近似程度
            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)

        # 使用默认数据类型为 float 运行内部测试方法，分别传入 t、weight_even 和 weight_odd
        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    # 使用 Scipy 库进行二维卷积测试，根据模式选择验证或同维度填充
    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv2d_vs_scipy(self, device, dtype, mode):
        # 创建指定设备和数据类型的三维张量，形状为 (1, 5, 10)
        t = make_tensor((1, 5, 10), device=device, dtype=dtype)
        # 创建指定设备和数据类型的权重张量，形状为 (1, 1, 2, 4) 和 (1, 1, 3, 5)
        weight_even = make_tensor((1, 1, 2, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 3, 5), device=device, dtype=dtype)

        # 定义内部测试方法，用于执行具体的二维卷积比较操作
        def _test(t, weight, mode):
            # 将输入张量 t 去除批次维度并转换为 NumPy 数组
            t_a = t.squeeze(0).cpu().numpy()
            # 将权重张量去除批次和通道维度并转换为 NumPy 数组
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            # 使用 Scipy 库进行二维卷积操作，返回期望的结果
            expected = scipy.signal.convolve2d(t_a, w_a, mode=mode)

            # 初始化二维卷积的参数字典
            kwargs = {"padding": mode}
            # 如果模式为 "same"，计算左右填充和上下填充的数量，并在输入张量上进行填充操作
            if mode == "same":
                left_right_pad = weight.shape[3] // 2
                top_bottom_pad = weight.shape[2] // 2
                p = (left_right_pad, left_right_pad, top_bottom_pad, top_bottom_pad)
                t = torch.nn.functional.pad(t, p)
                # 删除 kwargs 中的 "padding" 键
                kwargs.pop("padding")

            # 反转权重张量并执行 PyTorch 的二维卷积操作，得到实际结果
            weight_flipped = torch.flip(weight, (2, 3))
            actual = torch.nn.functional.conv2d(t, weight_flipped, **kwargs).squeeze(0)
            # 如果模式为 "same"，则截取特定区域
            if mode == "same":
                actual = actual[:5, :10]

            # 使用断言比较实际结果与期望结果的近似程度
            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # 使用默认数据类型为 float 运行内部测试方法，分别传入 t、weight_even 和 weight_odd
        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)
    # 定义一个测试方法，比较自定义的 conv3d 实现和 scipy 库的效果
    def test_conv3d_vs_scipy(self, device, dtype, mode):
        # 创建一个形状为 (1, 5, 5, 10) 的张量 t，指定设备和数据类型
        t = make_tensor((1, 5, 5, 10), device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 2, 2, 4) 的权重张量 weight_even
        weight_even = make_tensor((1, 1, 2, 2, 4), device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 2, 3, 5) 的权重张量 weight_odd

        weight_odd = make_tensor((1, 1, 2, 3, 5), device=device, dtype=dtype)

        # 定义内部函数 _test，用于执行单个测试
        def _test(t, weight, mode):
            # 将张量 t 压缩为一个 numpy 数组 t_a，并将其移到 CPU 上
            t_a = t.squeeze(0).cpu().numpy()
            # 将权重张量 weight 压缩为一个 numpy 数组 w_a，并将其移到 CPU 上
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            # 使用 scipy.signal.convolve 函数计算期望的卷积结果
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            # 根据卷积模式设置不同的填充方式
            kwargs = {"padding": mode}
            if mode == "same":
                # 计算需要的填充量
                left_right_pad = weight.shape[4] // 2
                top_bottom_pad = weight.shape[3] // 2
                front_back_pad = weight.shape[2] // 2
                p = (
                    left_right_pad,
                    left_right_pad,
                    top_bottom_pad,
                    top_bottom_pad,
                    front_back_pad,
                    front_back_pad,
                )
                # 使用 torch.nn.functional.pad 函数对张量 t 进行填充
                t = torch.nn.functional.pad(t, p)
                # 移除 kwargs 中的 padding 键
                kwargs.pop("padding")

            # 翻转权重张量，以便与 conv3d 函数的期望参数匹配
            weight_flipped = torch.flip(weight, (2, 3, 4))
            # 使用 torch.nn.functional.conv3d 函数计算实际的卷积结果，并压缩第一个维度
            actual = torch.nn.functional.conv3d(t, weight_flipped, **kwargs).squeeze(0)

            # 如果模式是 "same"，则截取前 5x5x10 的部分作为最终的实际结果
            if mode == "same":
                actual = actual[:5, :5, :10]

            # 使用 self.assertEqual 函数比较实际结果和期望结果，设置误差容忍度
            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # 使用 set_default_dtype 函数将默认数据类型设置为 torch.float
        with set_default_dtype(torch.float):
            # 分别对 weight_even 和 weight_odd 运行 _test 方法
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    # 使用装饰器指定数据类型为 torch.float，定义一个测试卷积层在 "valid" padding 下反向传播的方法
    @dtypes(torch.float)
    def test_conv2d_valid_padding_backward(self, device, dtype):
        # 创建一个形状为 (1, 1, 1, 10) 的张量 x，和形状为 (1, 1, 1, 4) 的张量 y
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype, requires_grad=True)
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype, requires_grad=True)

        # 对 F.conv2d 函数进行前向传播和反向传播，并计算梯度 gx_expect 和 gy_expect
        F.conv2d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度信息
        x.grad, y.grad = None, None

        # 使用 "valid" padding 对 F.conv2d 函数进行前向传播和反向传播
        F.conv2d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad

        # 使用 self.assertEqual 函数比较 gx_expect 和 gx_actual，gy_expect 和 gy_actual
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    # 使用装饰器指定数据类型为 torch.double，定义一个测试卷积层在 "valid" padding 下反向传播的方法
    @dtypes(torch.double)
    def test_conv3d_valid_padding_backward(self, device, dtype):
        # 创建一个形状为 (1, 1, 1, 1, 10) 的张量 x，和形状为 (1, 1, 1, 1, 4) 的张量 y
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        # 对 F.conv3d 函数进行前向传播和反向传播，并计算梯度 gx_expect 和 gy_expect
        F.conv3d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度信息
        x.grad, y.grad = None, None

        # 使用 "valid" padding 对 F.conv3d 函数进行前向传播和反向传播
        F.conv3d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad

        # 使用 self.assertEqual 函数比较 gx_expect 和 gx_actual，gy_expect 和 gy_actual
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

        # 对 F.conv3d 函数进行梯度检查，检查前向传播和反向传播的一致性
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_forward_ad=True,
        )
        # 对 F.conv3d 函数进行二阶梯度检查，检查前向传播和反向传播的二阶导数一致性
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_fwd_over_rev=True,
        )
    # 使用 @parametrize_test 装饰器为 test_conv_transpose_with_output_size_and_no_batch_dim 方法参数化测试，测试 ConvTransposeNd 模块
    @parametrize_test("N", range(2, 4), name_fn=lambda N: f"ConvTranspose{N}d")
    def test_conv_transpose_with_output_size_and_no_batch_dim(self, device, N):
        # 根据 N 的值选择不同维度的随机输入数据
        inp = torch.randn((1, 15, 13) if N == 2 else (1, 15, 13, 13), device=device)
        # 根据 N 的值选择不同的输出尺寸
        output_size = (1, 240, 200) if N == 2 else (1, 240, 200, 200)
        # 根据 N 动态获取对应维度的 ConvTranspose 类
        ConvTransposeNd = getattr(nn, f"ConvTranspose{N}d")
        # 创建 ConvTransposeNd 模块实例
        m = ConvTransposeNd(
            1, 1, kernel_size=16, stride=16, padding=7, bias=False, device=device
        )
        # 将输入数据 inp 和输出尺寸 output_size 传入模块 m 进行计算
        output = m(inp, output_size=output_size)
        # 断言输出的形状与期望的 output_size 一致
        self.assertEqual(output.shape, output_size)
    
    # 使用 @dtypes 装饰器为 test_conv_empty_channel 方法指定 torch.float 数据类型测试
    @dtypes(torch.float)
    def test_conv_empty_channel(self, device, dtype):
        # 设置输入通道数为 0
        in_channels = 0
        # 创建 Conv1d 模块，输入通道为 0
        mod = torch.nn.Conv1d(in_channels, 8, 2, stride=2, dtype=dtype).to(device)
        # 创建输入数据 inp，通道数为 0
        inp = torch.randn(2, 0, 15, device=device, dtype=dtype)
        # 调用 _test_module_empty_input 函数测试空输入情况
        _test_module_empty_input(self, mod, inp, check_size=False)
    
        # 测试输入通道为 0 时的异常情况
        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 0, device=device, dtype=dtype)
            mod(inp)
    
        # 创建 Conv2d 模块，输入通道为 0
        mod = torch.nn.Conv2d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        # 创建输入数据 inp，通道数为 0
        inp = torch.randn(2, 0, 50, 100, device=device, dtype=dtype)
        # 调用 _test_module_empty_input 函数测试空输入情况
        _test_module_empty_input(self, mod, inp, check_size=False)
    
        # 测试输入通道为 0 时的异常情况
        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 40, 0, device=device, dtype=dtype)
            mod(inp)
    
        # 创建 Conv3d 模块，输入通道为 0
        mod = torch.nn.Conv3d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        # 创建输入数据 inp，通道数为 0
        inp = torch.randn(2, 0, 50, 20, 40, device=device, dtype=dtype)
        # 调用 _test_module_empty_input 函数测试空输入情况
        _test_module_empty_input(self, mod, inp, check_size=False)
    
        # 测试输入通道为 0 时的异常情况
        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 50, 0, 40, device=device, dtype=dtype)
            mod(inp)
    
    # 测试空组卷积（group convolution）的情况
    def test_group_conv_empty(self, device):
        # 创建 Group Conv2d 模块，输入通道为 0
        mod = torch.nn.Conv2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4).to(
            device
        )
        # 创建输入数据 inp，batch 大小为 0
        inp = torch.randn(0, 4, 4, 4, device=device)
        # 调用 _test_module_empty_input 函数测试空输入情况
        _test_module_empty_input(self, mod, inp, check_size=False)
    
    # 测试空组转置卷积（group transposed convolution）的情况
    def test_group_convTranspose_empty(self, device):
        # 创建 Group ConvTranspose2d 模块，输入通道为 0
        mod = torch.nn.ConvTranspose2d(
            4, 4, stride=2, kernel_size=3, padding=1, groups=4
        ).to(device)
        # 创建输入数据 inp，batch 大小为 0
        inp = torch.randn(0, 4, 4, 4, device=device)
        # 调用 _test_module_empty_input 函数测试空输入情况
        _test_module_empty_input(self, mod, inp, check_size=False)
    
    # 测试空转置卷积（transposed convolution）的情况
    def test_convTranspose_empty(self, device):
        # 创建 ConvTranspose2d 模块，输入通道为 0
        mod = torch.nn.ConvTranspose2d(4, 4, stride=2, kernel_size=3, padding=1).to(
            device
        )
        # 创建输入数据 inp，batch 大小为 0
        inp = torch.randn(0, 4, 4, 4, device=device)
        # 调用 _test_module_empty_input 函数测试空输入情况
        _test_module_empty_input(self, mod, inp, check_size=False)
    # 测试大输入数据的卷积操作，不进行分割
    def test_conv_large_nosplit(self, device):
        # 定义数据类型为半精度
        dtype = torch.half
        # 创建一个卷积层，输入通道数为2，输出通道数为2，卷积核大小为8x8
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        # 创建一个大尺寸的输入数据
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        # 对大尺寸输入数据进行卷积操作
        conv1(input_large)
        # 创建另一个卷积层，输入通道数为1，输出通道数为1024，卷积核大小为1x1
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        # 创建另一个大尺寸的输入数据
        input_large = torch.randn(1, 1, 2048, 1024, dtype=dtype, device=device)
        # 对另一个大尺寸输入数据进行卷积操作
        conv2(input_large)

    # 测试非连续权重的卷积操作
    def test_conv_noncontig_weights(self, device):
        # 遍历不同维度
        for dim in (1, 2, 3):
            # 遍历是否分组卷积
            for grouped in (False, True):
                nc = 3
                groups = 3 if grouped else 1
                # 创建随机权重
                w = torch.randn([3] * dim, device=device)
                w = w.expand([nc, int(nc / groups)] + list(w.shape))
                w = w.detach().requires_grad_()
                # 创建随机输入数据
                x = torch.randn([1, nc] + ([5] * dim), device=device, requires_grad=True)
                # 进行卷积操作
                y = getattr(F, f"conv{dim}d")(x, w, groups=groups)
                y.sum().backward()
                # 进行反卷积操作
                y = getattr(F, f"conv_transpose{dim}d")(x, w, groups=groups)
                y.sum().backward()

    # 测试非连续权重和偏置的卷积操作
    def test_conv_noncontig_weights_and_bias(self, device):
        # 遍历是否包含偏置
        for bias in [True, False]:
            # 创建一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为7x7
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=bias).to(
                device, torch.float
            )
            # 创建随机输入数据
            input_nc = torch.randn((1, 3, 224, 224, 2), device=device, dtype=torch.float)[:, :, :, :, 1]
            input_c = input_nc.contiguous()
            # 创建随机权重
            weight_nc = torch.randn((64, 3, 7, 7, 2), device=device, dtype=torch.float)[:, :, :, :, 1]
            conv1.weight = nn.Parameter(weight_nc)
            weight_c = conv1.weight.contiguous()
            # 如果包含偏置，创建随机偏置
            if bias:
                bias_nc = torch.randn((64, 2), device=device, dtype=torch.float)[:, 1]
                conv1.bias = nn.Parameter(bias_nc)
                bias_c = conv1.bias.contiguous()
            # 对输入数据进行卷积操作
            out1 = conv1(input_nc)
            conv1.weight = nn.Parameter(weight_c)
            # 如果包含偏置，设置卷积层的偏置
            if bias:
                conv1.bias = nn.Parameter(bias_c)
            # 对连续数据进行卷积操作
            out2 = conv1(input_c)
            # 断言两次卷积操作的结果是否相等
            self.assertEqual(out1, out2)
    # 定义一个用于测试转置卷积操作的方法，接受一个设备参数
    def test_conv_transposed_large(self, device):
        # 根据设备类型选择数据类型为半精度或单精度
        dtype = torch.half if self.device_type == "cuda" else torch.float
        # 创建一个转置卷积层，输入通道数为1，输出通道数为1，核大小为1x1，步长为1，无偏置项
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        # 创建一个大尺寸的随机输入张量，形状为[4096, 1, 512, 1024]，使用指定的数据类型和设备
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        # 对输入张量进行转置卷积操作
        ret = conv(input_large)
        # 计算不同部分之间的最大差异，确保结果的精确性
        maxdiff0 = (
            (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024)))
            .abs_()  # 取绝对值
            .max()    # 求最大值
            .item()   # 转换为标量值
        )
        maxdiff1 = (
            (ret.narrow(0, 1024, 1024) - conv(input_large.narrow(0, 1024, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff2 = (
            (ret.narrow(0, 2048, 1024) - conv(input_large.narrow(0, 2048, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff3 = (
            (ret.narrow(0, 3072, 1024) - conv(input_large.narrow(0, 3072, 1024)))
            .abs_()
            .max()
            .item()
        )
        # 断言各部分的最大差异为0，即确保转置卷积操作正确无误
        self.assertEqual(maxdiff0, 0)
        self.assertEqual(maxdiff1, 0)
        self.assertEqual(maxdiff2, 0)
        self.assertEqual(maxdiff3, 0)

    # 定义一个用于测试普通卷积操作的方法，接受一个设备参数
    def test_conv_large(self, device):
        # 根据设备类型选择数据类型为半精度或单精度
        dtype = torch.half if self.device_type == "cuda" else torch.float
        # 创建一个普通卷积层，输入通道数为2，输出通道数为2，核大小为8x8，步长为8，无偏置项
        conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
        # 创建一个大尺寸的随机输入张量，形状为[4097, 2, 512, 512]，使用指定的数据类型和设备
        input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
        # 对输入张量进行普通卷积操作
        ret = conv(input_large)
        # 断言卷积结果的前2048个元素与相应部分输入的卷积结果相等
        self.assertEqual(ret[:2048], conv(input_large[:2048]))
        self.assertEqual(ret[2048:4096], conv(input_large[2048:4096]))
        self.assertEqual(ret[4096:], conv(input_large[4096:]))

        # 对卷积层的梯度进行清零
        conv.zero_grad()
        # 对卷积结果进行视图变换，并计算每行最大值的和，然后进行反向传播
        ret.view(4097, -1).max(dim=1).values.sum().backward()
        # 删除卷积结果
        del ret
        # 复制卷积核的梯度值
        grad1 = conv.weight.grad.detach().clone()
        # 再次清零卷积层的梯度
        conv.zero_grad()
        # 分别对不同部分的输入进行卷积操作，并计算每行最大值的和，然后进行反向传播
        conv(input_large[:2048]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[2048:4096]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[4096:]).view(1, -1).max(dim=1).values.sum().backward()
        # 复制卷积核的梯度值
        grad2 = conv.weight.grad.detach().clone()
        # 计算梯度缩放比例
        scale = 1 / grad2.abs().mean()
        grad1 = grad1 * scale
        grad2 = grad2 * scale
        # 断言两次梯度值相等，设置绝对误差和相对误差的容差
        self.assertEqual(grad1, grad2, atol=5e-2, rtol=5e-3)
    # 定义一个测试方法，用于测试 Conv2d 操作的大小为 1 的卷积核
    def test_Conv2d_size_1_kernel(self, device):
        # 生成一个形状为 (2, 3, 5, 5) 的随机张量作为输入数据
        x_cpu = torch.randn(2, 3, 5, 5)
        # 创建一个 Conv2d 模块，输入通道数为 3，输出通道数为 3，卷积核大小为 1
        conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
        # 对输入数据进行卷积操作，得到输出 y_cpu
        y_cpu = conv_cpu(x_cpu)
        # 创建一个与 y_cpu 同样大小的随机张量 y
        y = torch.rand_like(y_cpu)
        # 计算 y_cpu 的反向传播梯度
        y_cpu.backward(y)

        # 在禁用 cuDNN 的上下文中，创建一个在指定设备上的 Conv2d 模块
        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)
            # 将 conv_cpu 的偏置数据复制到 conv_cuda 的偏置中
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            # 将 conv_cpu 的权重数据复制到 conv_cuda 的权重中
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            # 对输入数据 x_cpu 在设备上进行卷积操作，得到输出 y_cuda
            y_cuda = conv_cuda(x_cpu.to(device))
            # 计算 y_cuda 的反向传播梯度
            y_cuda.backward(y.to(device))

        # 使用断言比较 CPU 和 CUDA 上的输出 y_cpu 和 y_cuda 是否相等，指定容差和设备精度
        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        # 使用断言比较 CPU 和 CUDA 上的偏置梯度是否相等，指定容差和设备精度
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        # 使用断言比较 CPU 和 CUDA 上的权重梯度是否相等，指定容差和设备精度
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )

    # 定义一个测试方法，用于测试 ConvTranspose2d 操作的大小为 1 的卷积核
    def test_ConvTranspose2d_size_1_kernel(self, device):
        # 生成一个形状为 (2, 3, 5, 5) 的随机张量作为输入数据
        x_cpu = torch.randn(2, 3, 5, 5)
        # 创建一个 ConvTranspose2d 模块，输入通道数为 3，输出通道数为 3，卷积核大小为 1
        conv_cpu = torch.nn.ConvTranspose2d(3, 3, kernel_size=1)
        # 对输入数据进行转置卷积操作，得到输出 y_cpu
        y_cpu = conv_cpu(x_cpu)
        # 创建一个与 y_cpu 同样大小的随机张量 y
        y = torch.rand_like(y_cpu)
        # 计算 y_cpu 的反向传播梯度
        y_cpu.backward(y)
        
        # 创建一个在指定设备上的 ConvTranspose2d 模块
        conv_cuda = torch.nn.ConvTranspose2d(3, 3, kernel_size=1).to(device)
        # 将 conv_cpu 的偏置数据复制到 conv_cuda 的偏置中
        conv_cuda.bias.data.copy_(conv_cpu.bias.data)
        # 将 conv_cpu 的权重数据复制到 conv_cuda 的权重中
        conv_cuda.weight.data.copy_(conv_cpu.weight.data)
        # 对输入数据 x_cpu 在设备上进行转置卷积操作，得到输出 y_cuda
        y_cuda = conv_cuda(x_cpu.to(device))
        # 计算 y_cuda 的反向传播梯度
        y_cuda.backward(y.to(device))

        # 使用断言比较 CPU 和 CUDA 上的输出 y_cpu 和 y_cuda 是否相等，指定容差和设备精度
        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        # 使用断言比较 CPU 和 CUDA 上的偏置梯度是否相等，指定容差和设备精度
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        # 使用断言比较 CPU 和 CUDA 上的权重梯度是否相等，指定容差和设备精度
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
    # 定义一个测试方法，用于测试 ConvTranspose3d 类的行为和输出大小
    def test_ConvTranspose3d_size_1_kernel(self, device):
        # 使用双精度浮点数作为默认数据类型
        with set_default_dtype(torch.double):
            # 创建一个形状为 (2, 3, 3, 5, 5) 的随机张量 x_cpu
            x_cpu = torch.randn(2, 3, 3, 5, 5)
            # 创建一个 ConvTranspose3d 类对象 conv_cpu，输入通道和输出通道均为 3，卷积核大小为 1
            conv_cpu = torch.nn.ConvTranspose3d(3, 3, kernel_size=1)
            # 对 x_cpu 进行卷积操作，得到输出张量 y_cpu
            y_cpu = conv_cpu(x_cpu)
            # 创建一个形状与 y_cpu 相同的随机张量 y
            y = torch.rand_like(y_cpu)
            # 对 y_cpu 进行反向传播
            y_cpu.backward(y)
            # 在 GPU 设备上创建一个与 conv_cpu 权重和偏置相同的 ConvTranspose3d 对象 conv_cuda
            conv_cuda = torch.nn.ConvTranspose3d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)  # 复制 conv_cpu 的偏置到 conv_cuda
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)  # 复制 conv_cpu 的权重到 conv_cuda
            # 将 x_cpu 移动到指定的设备并在 conv_cuda 上执行卷积操作，得到输出张量 y_cuda
            y_cuda = conv_cuda(x_cpu.to(device))
            # 对 y_cuda 进行反向传播
            y_cuda.backward(y.to(device))

            # 断言两个输出张量 y_cpu 和 y_cuda 在给定的容差范围内相等
            self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
            # 断言 conv_cpu 和 conv_cuda 的偏置梯度在给定的容差范围内相等
            self.assertEqual(
                conv_cpu.bias.grad.data,
                conv_cuda.bias.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )
            # 断言 conv_cpu 和 conv_cuda 的权重梯度在给定的容差范围内相等
            self.assertEqual(
                conv_cpu.weight.grad.data,
                conv_cuda.weight.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )

    # 使用指定的浮点数数据类型对 Conv2d 类进行测试
    @dtypes(torch.float)
    def test_Conv2d_naive_groups(self, device, dtype):
        # 创建一个 Conv2d 类对象 m，输入通道和输出通道均为 4，卷积核大小为 3，分组数为 2
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        # 创建一个形状为 (2, 4, 6, 6) 的随机张量 i，并在指定的设备和数据类型上进行计算
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        # 对输入张量 i 进行卷积操作，得到输出张量 output
        output = m(i)
        # 创建一个形状为 (2, 4, 4, 4) 的随机张量 grad_output，用于反向传播
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        # 对输出张量 output 进行反向传播
        output.backward(grad_output)

        # 创建另一个 Conv2d 类对象 m1，输入通道和输出通道均为 2，卷积核大小为 3，并复制部分 m 的权重和偏置
        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])  # 复制 m 的前两个权重到 m1
        m1.bias.data.copy_(m.bias.data[:2])  # 复制 m 的前两个偏置到 m1
        # 创建一个形状与 i 的前两个通道对应的数据 i1，并设置为需要梯度计算
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        # 对输入张量 i1 进行卷积操作，得到输出张量 output1
        output1 = m1(i1)
        # 对输出张量 output1 进行反向传播
        output1.backward(grad_output[:, :2].contiguous())

        # 创建另一个 Conv2d 类对象 m2，输入通道和输出通道均为 2，卷积核大小为 3，并复制部分 m 的权重和偏置
        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])  # 复制 m 的后两个权重到 m2
        m2.bias.data.copy_(m.bias.data[2:])  # 复制 m 的后两个偏置到 m2
        # 创建一个形状与 i 的后两个通道对应的数据 i2，并设置为需要梯度计算
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        # 对输入张量 i2 进行卷积操作，得到输出张量 output2
        output2 = m2(i2)
        # 对输出张量 output2 进行反向传播
        output2.backward(grad_output[:, 2:].contiguous())

        # 断言总输出 output 等于两部分输出 output1 和 output2 的连接结果
        self.assertEqual(output, torch.cat([output1, output2], 1))
        # 断言输入张量 i 的梯度等于 i1 和 i2 的梯度的连接结果
        self.assertEqual(
            i.grad.data,
            torch.cat([i1.grad.data, i2.grad.data], 1),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        # 断言 m 的偏置梯度等于 m1 和 m2 的偏置梯度的连接结果
        self.assertEqual(
            m.bias.grad.data,
            torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        # 断言 m 的权重梯度等于 m1 和 m2 的权重梯度的连接结果
        self.assertEqual(
            m.weight.grad.data,
            torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
    # 定义一个测试函数，用于测试深度卷积的反向传播
    def test_Conv2d_backward_depthwise(self, device, dtype):
        # 创建随机张量作为输入，设备为指定设备，数据类型为指定类型，并且需要计算梯度
        x = torch.randn(2, 2, 4, 20, device=device, dtype=dtype, requires_grad=True)
        # 创建随机张量作为卷积核，设备和数据类型与输入相同，并且需要计算梯度
        weight = torch.randn(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)

        # 定义一个深度卷积函数，使用 PyTorch 的函数进行卷积操作
        def conv2d_depthwise(x, weight):
            return torch.nn.functional.conv2d(
                x, weight, bias=None, stride=(1, 10), groups=2
            )

        # 使用 PyTorch 的 gradcheck 函数检查 conv2d_depthwise 函数的梯度计算是否正确
        torch.autograd.gradcheck(conv2d_depthwise, (x, weight))

    # 装饰器定义了一个测试函数，用于测试使用 cuDNN 加速的 NHWC 格式的卷积操作
    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_nhwc(self, device, dtype):
        # 定义一个辅助函数，用于执行不同参数下的卷积测试
        def helper(n, c, h, w, out_channels, kernel_size, groups):
            # 创建随机整数张量作为输入，数据类型为指定类型，设备为指定设备，并使用通道优先的内存布局
            input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device).to(
                memory_format=torch.channels_last
            )
            # 需要计算输入张量的梯度
            input.requires_grad_()

            # 创建一个卷积层对象，设置输入通道数、输出通道数、卷积核大小和分组数，并使用通道优先的内存布局
            conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last
            )

            # 将卷积层的参数初始化为与其相同形状的随机整数张量
            for p in conv.parameters():
                p.data = torch.randint_like(p, -3, 3)

            # 创建参考输入，它是输入的副本，并且转换为连续内存布局，数据类型为双精度浮点型，并需要计算梯度
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            # 创建一个与 conv 对象相同结构的参考卷积层
            ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
            # 加载 conv 对象的状态字典到参考卷积层
            ref_conv.load_state_dict(conv.state_dict())
            # 将参考卷积层转换为指定设备和双精度浮点型，使用连续内存布局
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            # 对输入进行卷积操作，得到输出
            out = conv(input)
            # 对参考输入进行卷积操作，得到参考输出
            ref_out = ref_conv(ref_input)

            # 创建随机整数张量作为梯度
            grad = torch.randint_like(out, -3, 3)
            # 将梯度的副本转换为双精度浮点型，并使用连续内存布局
            ref_grad = grad.detach().clone().double().contiguous()

            # 对输出进行反向传播，计算梯度
            out.backward(grad)
            # 对参考输出进行反向传播，计算梯度
            ref_out.backward(ref_grad)

            # 断言输出张量使用通道优先的内存布局
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            # 断言输入张量的梯度使用通道优先的内存布局
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last))
            # 断言卷积层权重的梯度使用通道优先的内存布局
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last)
            )

            # 断言参考输出张量使用连续内存布局
            self.assertTrue(ref_out.is_contiguous())
            # 断言参考输入张量的梯度使用连续内存布局
            self.assertTrue(ref_input.grad.is_contiguous())
            # 断言参考卷积层权重的梯度使用连续内存布局
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            # 断言输出张量与参考输出张量相等，数据类型不必完全相同
            self.assertEqual(out, ref_out, exact_dtype=False)
            # 断言卷积层权重的梯度与参考卷积层权重的梯度相等，数据类型不必完全相同
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            # 断言卷积层偏置的梯度与参考卷积层偏置的梯度相等，数据类型不必完全相同
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            # 断言输入张量的梯度与参考输入张量的梯度相等，数据类型不必完全相同
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        # 分别测试不同参数组合下的卷积操作
        helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=16)
    # 定义一个测试函数，用于测试使用 CuDNN 加速的 NDHWC 格式的卷积操作
    def test_conv_cudnn_ndhwc(self, device, dtype):
        # 定义一个内部辅助函数，用于执行单个卷积测试
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            # 创建一个随机整数张量作为输入，采用 channels_last_3d 内存格式
            input = torch.randint(
                -2, 2, (n, c, d, h, w), dtype=dtype, device=device
            ).to(memory_format=torch.channels_last_3d)
            input.requires_grad_()

            # 创建一个 Conv3d 模块，设置相关参数并使用 channels_last_3d 内存格式
            conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last_3d
            )
            # 初始化卷积核权重为随机整数
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            # 创建参考输入，对其进行深拷贝并转换为 double 类型，保持连续性并要求梯度
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            # 创建一个与 conv 具有相同状态字典的 Conv3d 模块作为参考模型
            ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            # 执行卷积操作，获取输出
            out = conv(input)
            # 使用参考模型进行卷积操作，获取参考输出
            ref_out = ref_conv(ref_input)

            # 创建随机整数梯度张量
            grad = torch.randint_like(out, -2, 2)
            # 对参考输出的梯度进行深拷贝并转换为 double 类型，并保持连续性
            ref_grad = grad.detach().clone().double().contiguous()

            # 对输出进行反向传播
            out.backward(grad)
            # 对参考输出进行反向传播
            ref_out.backward(ref_grad)

            # 断言输出张量在 channels_last_3d 内存格式下是连续的
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            # 断言输入梯度张量在 channels_last_3d 内存格式下是连续的
            self.assertTrue(
                input.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )
            # 断言卷积核权重的梯度在 channels_last_3d 内存格式下是连续的
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )

            # 断言参考输出张量是连续的
            self.assertTrue(ref_out.is_contiguous())
            # 断言参考输入梯度张量是连续的
            self.assertTrue(ref_input.grad.is_contiguous())
            # 断言参考卷积核权重的梯度是连续的
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            # 断言输出与参考输出相等，允许类型不完全匹配
            self.assertEqual(out, ref_out, exact_dtype=False)
            # 断言卷积核权重的梯度与参考卷积核权重的梯度相等，允许类型不完全匹配
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            # 断言卷积的偏置项的梯度与参考卷积的偏置项的梯度相等，允许类型不完全匹配
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            # 断言输入的梯度与参考输入的梯度相等，允许类型不完全匹配
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        # 执行多个测试用例
        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    # 定义一个私有方法，用于运行卷积操作的具体实现
    def _run_conv(
        self,
        layer,
        device,
        inp,
        grad,
        ref_conv,
        ref_input,
        ref_out,
        input_format,
        weight_format,
        grad_format,
        output_format,
    @dtypes(torch.float, torch.double)
    # 使用装饰器声明测试函数支持的数据类型，可以同时测试 float 和 double 类型
    def test_conv_cudnn_nhwc_support(self, device, dtype):
        # 创建一个形状为 (1, 16, 1, 1) 的随机张量作为输入，类型为指定的 dtype，位于指定设备上，并设置需要梯度
        input = torch.randn((1, 16, 1, 1), dtype=dtype, device=device, requires_grad=True)
        # 创建一个形状为 (8, 16, 3, 3) 的随机张量作为卷积核，类型为指定的 dtype，位于指定设备上，并设置需要梯度
        weight = torch.randn((8, 16, 3, 3), dtype=dtype, device=device, requires_grad=True)
        # 将卷积核转换为通道优先的内存格式
        weight = weight.to(memory_format=torch.channels_last)
        # 执行二维卷积操作，输入为 input，卷积核为 weight，其余参数按照指定的步长、填充和 dilation 设置
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        # 断言输出 o 在通道优先的内存格式下是连续的
        self.assertTrue(o.is_contiguous(memory_format=torch.channels_last))
        # 对输出 o 的元素求和，并进行反向传播
        o.sum().backward()
    # 使用装饰器指定此方法仅适用于 torch.float 数据类型
    @dtypes(torch.float)
    # 定义一个测试方法，用于测试没有梯度的 Conv2d 操作
    def test_conv2d_no_grad(self, device, dtype):
        # 对于不同的 batch 大小进行循环测试
        for batch in [1, 2, 3]:
            # 对于不同的分组数进行循环测试
            for groups in [1, 2, 4]:
                # 创建随机输入张量，指定设备和数据类型
                input = torch.rand(batch, groups, 8, 8, dtype=dtype, device=device)
                # 创建 Conv2d 模块，设置分组数等参数
                m = nn.Conv2d(
                    groups,
                    8,
                    kernel_size=(3, 3),
                    groups=groups,
                    dtype=dtype,
                    device=device,
                )
                # 使用 torch.no_grad() 上下文管理器禁用梯度计算，计算输出
                with torch.no_grad():
                    output_ng = m(input)
                # 再次计算输出（有梯度）
                output = m(input)
                # 使用 self.assertEqual 检查两种计算方式的输出是否相等，指定相对和绝对误差
                self.assertEqual(output, output_ng, rtol=1e-2, atol=1e-5)
    
    # 定义一个测试方法，用于测试带有双向传播的 Conv 操作，输入是 3D 张量和权重
    def test_conv_double_backward_strided_with_3D_input_and_weight(self, device):
        # 创建随机输入张量
        input = torch.randn(2, 3, 6, device=device)
        # 创建随机权重张量
        weight = torch.randn(3, 3, 3, device=device)
        # 创建随机偏置张量
        bias = torch.randn(3, device=device)
        # 设置步长（stride）、填充（padding）、膨胀（dilation）和转置标志（transposed）
        stride = (2,)
        padding = (1,)
        dilation = (1,)
        transposed = False
        # 输出填充（output_padding）设置为零
        output_padding = (0,)
        # 分组数设为1
        groups = 1
        # 使用 torch.ops.aten.convolution 进行卷积操作，返回输出张量
        output = torch.ops.aten.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
    
        # 创建随机张量 ggI, ggW, ggB, gO，作为梯度信息
        ggI = torch.randn(input.shape, device=device)
        ggW = torch.randn(weight.shape, device=device)
        ggB = torch.randn(bias.shape, device=device)
        gO = torch.randn(output.shape, device=device)
        # 创建输出掩码列表
        output_mask = [True, True, True]
        # 使用 torch.ops.aten._convolution_double_backward 进行卷积的双向传播计算
        # 返回梯度的梯度（grad_grad_output）、输入梯度（grad_input）、权重梯度（grad_weight）
        (
            grad_grad_output,
            grad_input,
            grad_weight,
        ) = torch.ops.aten._convolution_double_backward(
            ggI,
            ggW,
            ggB,
            gO,
            weight,
            input,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
    
        # 使用 self.assertEqual 检查梯度的梯度、输入梯度和权重梯度的形状是否与输出张量的形状相同
        self.assertEqual(grad_grad_output.shape, gO.shape)
        self.assertEqual(grad_input.shape, input.shape)
        self.assertEqual(grad_weight.shape, weight.shape)
    
    # 使用装饰器指定此方法仅适用于特定设备
    @onlyXPU
    # 使用装饰器指定此方法适用的数据类型范围
    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    # 定义一个测试方法，用于验证通道最后（channels_last）存储格式下的卷积输出步长
    def test_channels_last_ouput_stride(self, device, dtype):
        # 创建一个随机张量作为输入，形状为(2, 3, 16, 16)，在指定设备和数据类型上，并允许梯度计算
        input = torch.randn(
            (2, 3, 16, 16), device=device, dtype=dtype, requires_grad=True
        )
        # 创建一个随机权重张量，形状为(512, 3, 3, 3)，在指定设备和数据类型上，并允许梯度计算
        weight = torch.randn(
            (512, 3, 3, 3), device=device, dtype=dtype, requires_grad=True
        )
        # 将输入张量转换为通道最后存储格式（NHWC）
        input = input.to(memory_format=torch.channels_last)
        # 将权重张量转换为通道最后存储格式（NHWC）
        weight = weight.to(memory_format=torch.channels_last)
        # 执行二维卷积操作，输出张量out，步长为(2, 2)，填充为(0, 0)，卷积步长为(1, 1)，组数为1
        out = torch.conv2d(input, weight, None, (2, 2), (0, 0), (1, 1), 1)

        if dtype is torch.float64:
            # 如果数据类型为float64，大多数卷积后端不支持通道最后存储格式的float64
            # 输入为NHWC，输出为NCHW
            assert_size_stride(out, (2, 512, 7, 7), (25088, 49, 7, 1))
        else:
            # 如果数据类型不是float64
            # 输入为NHWC，输出为NHWC
            assert_size_stride(out, (2, 512, 7, 7), (25088, 1, 3584, 512))
# 实例化设备类型测试，使用 TestConvolutionNNDeviceType 类，将其添加到全局变量中，仅限于 "xpu" 设备类型
instantiate_device_type_tests(TestConvolutionNNDeviceType, globals(), only_for="xpu")

# 如果当前脚本作为主程序运行，则执行测试函数
if __name__ == "__main__":
    run_tests()
```