# `.\pytorch\test\nn\test_pooling.py`

```
# 导入必要的模块和库
# Owner(s): ["module: nn"]
import itertools  # 导入 itertools 模块，用于生成迭代器的函数
import math  # 导入数学函数模块
import operator  # 导入运算符模块，用于函数式编程中的操作符函数
import os  # 导入操作系统功能模块，提供了访问操作系统服务的功能
import random  # 导入随机数模块，用于生成随机数
import subprocess  # 导入子进程管理模块，用于创建和管理子进程
import sys  # 导入系统模块，提供了访问系统相关的功能
import unittest  # 导入单元测试模块，用于编写和运行单元测试
from functools import partial, reduce  # 导入 functools 模块中的 partial 和 reduce 函数
from itertools import repeat  # 从 itertools 模块导入 repeat 函数

import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口模块

from torch import inf, nan  # 从 torch 模块导入 inf 和 nan 常量
from torch.autograd import gradcheck, gradgradcheck  # 从 torch.autograd 模块导入梯度检查函数
from torch.testing import make_tensor  # 从 torch.testing 模块导入 make_tensor 函数
from torch.testing._internal.common_cuda import TEST_CUDA  # 从 torch.testing._internal.common_cuda 导入 TEST_CUDA 常量
from torch.testing._internal.common_device_type import (  # 从 torch.testing._internal.common_device_type 导入多个设备类型相关的常量和函数
    dtypes,
    dtypesIfCUDA,
    expectedFailureMeta,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIfRocm,
    TEST_WITH_ROCM,
)
from torch.testing._internal.common_dtype import floating_types_and  # 从 torch.testing._internal.common_dtype 导入 floating_types_and 函数
from torch.testing._internal.common_nn import (  # 从 torch.testing._internal.common_nn 导入多个神经网络测试相关的函数和类
    _test_bfloat16_ops,
    _test_module_empty_input,
    NNTestCase,
)
from torch.testing._internal.common_utils import (  # 从 torch.testing._internal.common_utils 导入多个实用工具函数和类
    gcIfJetson,
    instantiate_parametrized_tests,
    parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfMps,
    skipIfTorchDynamo,
    slowTest,
    subtest,
    TEST_WITH_UBSAN,
    TestCase,
)

class TestAvgPool(TestCase):
    def _sum_pool2d(self, x, kernel_size):
        # 使用 unfold 函数实现二维池化的窗口滑动操作
        windows = torch.nn.functional.unfold(
            x, kernel_size=kernel_size, stride=kernel_size
        )
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # 由于 unfold 不支持三维滑动窗口，需要将张量分割并计算和
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d 假设张量视图为 (1, 1, n, m)，因此需要两次 unsqueeze
        splited_x = [
            self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:])
            for t in splited_x
        ]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        # 计算二维平均池化结果
        size = reduce(operator.mul, kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        # 计算三维平均池化结果
        size = reduce(operator.mul, kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_doubletensor_avg_pool2d(self):
        n, m = 5, 8
        input = torch.rand(1, 1, n, m, dtype=torch.double)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 执行 avg_pool2d 函数进行测试
                actual = torch.nn.functional.avg_pool2d(input[0], (i, j))
                actual = actual.view(1, actual.numel())
                expected = self._avg_pool2d(input, (i, j))
                # 断言实际输出和期望输出是否一致
                self.assertEqual(actual, expected, rtol=0, atol=1e-5)
    def test_doubletensor_avg_pool2d_with_divisor(self):
        # 定义测试函数，测试在双精度张量上执行平均池化操作
        n, m = 3, 3
        # 创建一个大小为 (1, 1, n, m) 的随机双精度张量输入
        input = torch.rand(1, 1, n, m, dtype=torch.double)
        # 循环遍历池化窗口大小的可能取值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 遍历不同的除数取值，执行平均池化并断言结果
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    # 将池化后的结果重新视图为 (1, numel)
                    actual = actual.view(1, actual.numel())
                    # 计算预期的平均池化结果
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    # 断言实际结果与预期结果的近似性
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_doubletensor_avg_pool3d(self):
        # 定义测试函数，测试在双精度三维张量上执行平均池化操作
        h, w, d = 5, 6, 7
        # 创建一个大小为 (h, w, d) 的随机双精度三维张量输入
        input = torch.rand(h, w, d, dtype=torch.double)
        # 循环遍历池化窗口大小的可能取值
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    # 执行三维平均池化操作并视图化结果
                    actual = torch.nn.functional.avg_pool3d(
                        input.unsqueeze(0), (i, j, k)
                    )
                    actual = actual.view(1, actual.numel())
                    # 计算预期的三维平均池化结果
                    expected = self._avg_pool3d(input, (i, j, k))
                    # 断言实际结果与预期结果的近似性
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_doubletensor_avg_pool3d_with_divisor(self):
        # 定义测试函数，测试在双精度三维张量上执行带除数的平均池化操作
        h, w, d = 6, 5, 7
        # 创建一个大小为 (h, w, d) 的随机双精度三维张量输入
        input = torch.rand(h, w, d, dtype=torch.double)
        # 循环遍历池化窗口大小的可能取值
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    # 遍历不同的除数取值，执行带除数的三维平均池化并断言结果
                    for divisor in [1, 7, i * j]:
                        actual = torch.nn.functional.avg_pool3d(
                            input.unsqueeze(0), (i, j, k), divisor_override=divisor
                        )
                        actual = actual.view(1, actual.numel())
                        # 计算预期的带除数的三维平均池化结果
                        expected = self._sum_pool3d(input, (i, j, k)) / divisor
                        # 断言实际结果与预期结果的近似性
                        self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_avg_pool1d_ceil_mode(self):
        # 回归测试 gh-36977，测试一维平均池化的 ceil_mode
        x = 10 * torch.randn((1, 16, 4))
        # 执行带 ceil_mode 的一维平均池化操作，并确保结果中没有 NaN
        y = torch.nn.functional.avg_pool1d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=1, stride=2
        )
        self.assertTrue(not torch.isnan(y).any())

        # 如果支持 CUDA 测试，则在 CUDA 上执行相同的带 ceil_mode 的一维平均池化操作，并确保结果中没有 NaN
        if TEST_CUDA:
            y = torch.nn.functional.avg_pool1d(
                x.to("cuda"),
                ceil_mode=True,
                count_include_pad=True,
                kernel_size=1,
                stride=2,
            )
            self.assertTrue(not torch.isnan(y).any())
    # 定义一个测试函数，用于测试 avg_pool2d 函数在 ceil_mode=True 时的行为
    def test_avg_pool2d_ceil_mode(self):
        # 用于回归测试 GitHub 问题 gh-36977
        # 创建一个形状为 (1, 16, 4, 4) 的张量 x，其值为标准正态分布随机数乘以 10
        x = 10 * torch.randn((1, 16, 4, 4))
        # 调用 PyTorch 的 avg_pool2d 函数进行平均池化操作
        y = torch.nn.functional.avg_pool2d(
            x,
            ceil_mode=True,  # 使用 ceil_mode=True 进行池化，向上取整计算输出大小
            count_include_pad=True,  # 计算时包括池化核覆盖的 padding 区域
            kernel_size=(1, 2),  # 池化核大小为 (1, 2)
            padding=(0, 1),  # padding 大小为 (0, 1)
            stride=2,  # 池化核的步幅为 2
        )
        # 断言结果张量 y 中没有 NaN 值
        self.assertTrue(not torch.isnan(y).any())

        # 如果支持 CUDA 测试
        if TEST_CUDA:
            # 将输入张量 x 转移到 CUDA 设备上
            y = torch.nn.functional.avg_pool2d(
                x.to("cuda"),
                ceil_mode=True,
                count_include_pad=True,
                kernel_size=(1, 2),
                padding=(0, 1),
                stride=2,
            )
            # 断言 CUDA 下的结果张量 y 中没有 NaN 值
            self.assertTrue(not torch.isnan(y).any())

    # 定义一个测试函数，用于测试 avg_pool3d 函数在 ceil_mode=True 时的行为
    def test_avg_pool3d_ceil_mode(self):
        # 用于回归测试 GitHub 问题 gh-36977
        # 创建一个形状为 (1, 16, 4, 4, 4) 的 5 维张量 x，其值为标准正态分布随机数乘以 10
        x = 10 * torch.randn((1, 16, 4, 4, 4))
        # 调用 PyTorch 的 avg_pool3d 函数进行 3D 平均池化操作
        y = torch.nn.functional.avg_pool3d(
            x, 
            ceil_mode=True,  # 使用 ceil_mode=True 进行池化，向上取整计算输出大小
            count_include_pad=True,  # 计算时包括池化核覆盖的 padding 区域
            kernel_size=(1, 2, 3),  # 池化核大小为 (1, 2, 3)
            stride=2  # 池化核的步幅为 2
        )
        # 断言结果张量 y 中没有 NaN 值
        self.assertTrue(not torch.isnan(y).any())

        # 如果支持 CUDA 测试
        if TEST_CUDA:
            # 将输入张量 x 转移到 CUDA 设备上
            y = torch.nn.functional.avg_pool3d(
                x.to("cuda"),
                ceil_mode=True,
                count_include_pad=True,
                kernel_size=(1, 2, 3),
                stride=2,
            )
            # 断言 CUDA 下的结果张量 y 中没有 NaN 值
            self.assertTrue(not torch.isnan(y).any())
class TestPoolingNN(NNTestCase):
    _do_cuda_memory_leak_check = True  # 设置 CUDA 内存泄漏检查标志为 True
    _do_cuda_non_default_stream = True  # 设置使用非默认 CUDA 流标志为 True

    def test_adaptive_pooling_size_none(self):
        for numel in (2, 3):  # 遍历不同维度的张量
            for pool_type in ("Max", "Avg"):  # 遍历最大池化和平均池化两种类型
                cls_name = f"Adaptive{pool_type}Pool{numel}d"  # 根据池化类型和维度生成类名字符串
                module_cls = getattr(nn, cls_name)  # 根据类名字符串获取对应的类对象
                output_size = (2,) * (numel - 1) + (None,)  # 设置池化操作的输出大小，最后一个维度为 None
                module = module_cls(output_size)  # 使用类对象创建池化模块实例

                input = torch.randn((4,) * (numel + 1))  # 生成指定维度的随机输入张量
                output = module(input)  # 对输入张量进行池化操作
                self.assertEqual(output.size(), (4,) + (2,) * (numel - 1) + (4,))  # 断言输出张量的大小符合预期

    @unittest.skipIf(TEST_WITH_UBSAN, "signed integer overflow error with UBSAN")
    def test_adaptive_pooling_size_overflow(self):
        # 对于 UBSAN 测试，跳过本测试用例
        # 0x0x3fffffffffffffff * 2 * 2 = 0xfffffffffffffffc = -4 as int64_t
        # Tensor::numel() return int64_t, so following check that negative allocs are correctly handled
        # 引发 RuntimeError，验证负数分配是否正确处理
        self.assertRaises(
            RuntimeError,
            lambda: torch.nn.AdaptiveMaxPool1d(0x3FFFFFFFFFFFFFFF)(
                torch.empty([2, 2, 2])
            ),
        )

    def test_adaptive_pooling_avg_nhwc(self):
        device_list = ["cpu"]  # CPU 设备列表
        if TEST_CUDA:
            device_list.append("cuda")  # 如果支持 CUDA，则添加 CUDA 设备到列表

        for device in device_list:
            input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)  # 生成指定设备上的随机整数张量输入
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()  # 转换为通道优先内存格式，启用梯度跟踪
            grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)  # 生成指定设备上的随机整数张量梯度
            pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)  # 创建指定设备上的自适应平均池化层

            ref_input = input.detach().clone().contiguous().requires_grad_(True)  # 参考输入张量，与输入相同的操作
            ref_grad = grad.detach().clone().contiguous()  # 参考梯度张量，与梯度相同的操作
            ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)  # 参考自适应平均池化层，与池化层相同的操作

            out = pool(input)  # 对输入张量进行池化操作
            out.backward(grad)  # 反向传播计算梯度
            ref_out = ref_pool(ref_input)  # 对参考输入张量进行池化操作
            ref_out.backward(ref_grad)  # 反向传播计算参考梯度

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))  # 断言输出张量在通道优先内存格式下是连续的
            self.assertTrue(ref_out.is_contiguous())  # 断言参考输出张量是连续的
            self.assertEqual(out, ref_out)  # 断言输出张量与参考输出张量相等
            self.assertEqual(input.grad, ref_input.grad)  # 断言输入张量的梯度与参考输入张量的梯度相等
    # 定义一个测试函数，用于测试自适应平均池化算法在不连续的 NHWC 格式上的表现
    def test_adaptive_pooling_avg_nhwc_non_contiguous(self):
        # 定义设备列表，包括 CPU，若支持 CUDA 则添加 CUDA
        device_list = ["cpu"]
        if TEST_CUDA:
            device_list.append("cuda")

        # 遍历每个设备
        for device in device_list:
            # 生成一个形状为 (4, 8, 8, 8) 的随机整数张量，类型为 float32，放到指定设备上
            input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
            # 将输入张量转换为通道最后（NHWC）的内存格式
            input = input.contiguous(memory_format=torch.channels_last)
            # 对输入张量进行切片操作，每个通道间隔为2，同时要求梯度计算
            input = input[:, ::2, :, :].requires_grad_()
            # 生成一个形状为 (4, 8, 7, 7) 的随机整数张量作为梯度，类型为 float32，放到指定设备上
            grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
            # 对梯度张量进行切片操作，每个通道间隔为2
            grad = grad[:, ::2, :, :]
            # 创建一个指定大小 (7, 7) 的自适应平均池化层，放到指定设备上
            pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

            # 复制输入张量的无梯度版本，并保持连续性以及开启梯度追踪
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 复制梯度张量的无梯度版本，并保持连续性
            ref_grad = grad.detach().clone().contiguous()
            # 创建一个指定大小 (7, 7) 的自适应平均池化层，放到指定设备上
            ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)

            # 对输入张量进行自适应平均池化操作
            out = pool(input)
            # 对自适应平均池化的输出进行反向传播
            out.backward(grad)
            # 对无梯度版本的输入张量进行自适应平均池化操作
            ref_out = ref_pool(ref_input)
            # 对无梯度版本的自适应平均池化输出进行反向传播
            ref_out.backward(ref_grad)

            # 断言当前输出是否是通道最后（NHWC）内存格式
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            # 断言无梯度版本的输出是否连续
            self.assertTrue(ref_out.is_contiguous())
            # 断言当前输出与无梯度版本的输出是否相等
            self.assertEqual(out, ref_out)
            # 断言输入张量的梯度是否与无梯度版本的输入张量的梯度相等
            self.assertEqual(input.grad, ref_input.grad)
    # 定义测试方法，用于测试自适应池化在低精度情况下的行为
    def test_adaptive_pooling_lower_precision(self):
        # 定义内部测试方法，接受设备、数据类型、模型、内存格式作为参数
        def _test_adaptive_pooling_lower_precision(
            self, device, dtype, mod, memory_format
        ):
            # 创建随机整数张量作为输入，形状为 (3, 19, 8, 8)，数据类型为 float32
            input = torch.randint(1, 10, (3, 19, 8, 8), dtype=torch.float32)
            # 将输入张量移动到指定设备，并设置内存格式
            input = input.to(device).to(memory_format=memory_format).requires_grad_()
            # 根据模型类型创建池化层对象
            pool = mod((7, 7)).to(device)

            # 克隆输入张量并转换数据类型为指定的 dtype，同时需要梯度计算
            input2 = input.detach().clone().to(dtype=dtype).requires_grad_(True)

            # 对第一个输入进行池化操作
            out = pool(input)
            # 计算池化结果的和，并反向传播
            out.sum().backward()
            # 对第二个输入进行池化操作
            out2 = pool(input2)
            # 计算池化结果的和，并反向传播
            out2.sum().backward()

            # 断言第二个输出是否在指定的内存格式下是连续的
            self.assertTrue(out2.is_contiguous(memory_format=memory_format))
            # 断言第二个输出的数据类型是否符合指定的 dtype
            self.assertEqual(out2.dtype, dtype)
            # 断言第二个输入的梯度数据类型是否符合指定的 dtype
            self.assertEqual(input2.grad.dtype, dtype)
            # 断言第一个输出与第二个输出之间的数值相等，允许的误差为 0.1
            self.assertEqual(out, out2.float(), atol=0.1, rtol=0)
            # 断言第一个输入与第二个输入的梯度之间的数值相等，允许的误差为 0.1
            self.assertEqual(input.grad, input2.grad.float(), atol=0.1, rtol=0)

        # 设备列表，仅包含 "cpu"
        device_list = ["cpu"]
        # 遍历设备列表
        for device in device_list:
            # 遍历数据类型列表，包括 torch.bfloat16 和 torch.float16
            for dtype in [torch.bfloat16, torch.float16]:
                # 调用内部测试方法，测试 AdaptiveAvgPool2d 模型在不同内存格式下的行为
                _test_adaptive_pooling_lower_precision(
                    self,
                    device,
                    dtype,
                    torch.nn.AdaptiveAvgPool2d,
                    torch.contiguous_format,
                )
                # 同上，测试 AdaptiveAvgPool2d 模型在 channels_last 内存格式下的行为
                _test_adaptive_pooling_lower_precision(
                    self, device, dtype, torch.nn.AdaptiveAvgPool2d, torch.channels_last
                )
                # 同上，测试 AdaptiveMaxPool2d 模型在不同内存格式下的行为
                _test_adaptive_pooling_lower_precision(
                    self,
                    device,
                    dtype,
                    torch.nn.AdaptiveMaxPool2d,
                    torch.contiguous_format,
                )
                # 同上，测试 AdaptiveMaxPool2d 模型在 channels_last 内存格式下的行为
                _test_adaptive_pooling_lower_precision(
                    self, device, dtype, torch.nn.AdaptiveMaxPool2d, torch.channels_last
                )

    # 如果未启用 CUDA，跳过该测试方法
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 标记为大张量测试，需要至少 12GB 的内存，设备为 "cuda"
    @largeTensorTest("12GB", device="cuda")
    # 测试 AdaptiveAvgPool2d 模型在 nhwc 格式下的启动配置和反向传播行为
    def test_adaptive_pooling_avg_nhwc_launch_config_backward(self):
        # 创建随机整数张量作为输入，形状为 (1, 32, 2^17 + 1, 32)，数据类型为 float32，设备为 "cuda"
        input = torch.randint(
            1, 10, (1, 32, 2**17 + 1, 32), dtype=torch.float32, device="cuda"
        )
        # 设置输入张量的内存格式为 channels_last，并需要梯度计算
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        # 创建随机整数张量作为梯度，形状为 (1, 32, 10, 32)，数据类型为 float32，设备为 "cuda"
        grad = torch.randint(1, 10, (1, 32, 10, 32), dtype=torch.float32, device="cuda")

        # 创建 AdaptiveAvgPool2d 模型，目标输出形状为 (10, 32)，设备为 "cuda"
        pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()

        # 克隆输入张量并设置内存格式为 contiguous，并需要梯度计算
        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        # 克隆梯度张量并设置内存格式为 contiguous
        ref_grad = grad.detach().clone().contiguous()
        # 创建 AdaptiveAvgPool2d 模型，目标输出形状为 (10, 32)，设备为 "cuda"
        ref_pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()

        # 对输入进行 AdaptiveAvgPool2d 操作
        out = pool(input)
        # 对输出进行反向传播，使用给定的梯度
        out.backward(grad)
        # 对参考输入进行 AdaptiveAvgPool2d 操作
        ref_out = ref_pool(ref_input)
        # 对参考输出进行反向传播，使用参考梯度
        ref_out.backward(ref_grad)

        # 断言输出张量是否在 channels_last 内存格式下是连续的
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        # 断言参考输出张量是否在任意内存格式下是连续的
        self.assertTrue(ref_out.is_contiguous())
        # 断言输出张量与参考输出张量之间的数值是否完全相等
        self.assertEqual(out, ref_out)
        # 断言输入张量的梯度与参考输入张量的梯度之间的数值是否完全相等
        self.assertEqual(input.grad, ref_input.grad)

    # 如果未启用 CUDA，跳过该测试方法
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @largeTensorTest("12GB", device="cuda")
    # 使用装饰器指定测试函数需要大量内存和 CUDA 设备支持
    def test_adaptive_pooling_avg_nhwc_launch_config_forward(self):
        # 生成一个在 CUDA 设备上的随机整数张量作为输入
        input = torch.randint(
            1, 10, (1, 32, 16, 16), dtype=torch.float32, device="cuda"
        )
        # 转换输入张量的存储格式为通道为最后一维，并标记为需要梯度
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        # 创建一个在 CUDA 设备上的自适应平均池化层，指定输出尺寸
        pool = torch.nn.AdaptiveAvgPool2d((2**17 + 1, 32)).cuda()
    
        # 创建参考输入，保留梯度信息，并设置存储格式为通道为最后一维
        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        # 创建与 pool 对象相同的自适应平均池化层
        ref_pool = torch.nn.AdaptiveAvgPool2d((2**17 + 1, 32)).cuda()
    
        # 对输入进行池化操作
        out = pool(input)
        # 对参考输入进行池化操作
        ref_out = ref_pool(ref_input)
    
        # 断言输出张量的存储格式为通道为最后一维
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        # 断言参考输出张量的存储格式正常
        self.assertTrue(ref_out.is_contiguous())
        # 断言输出与参考输出相等
        self.assertEqual(out, ref_out)
    
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 如果 CUDA 不可用，则跳过测试
    def test_adaptive_avg_pooling_overflow(self):
        # 生成一个在 CUDA 设备上的随机整数张量作为输入
        input = torch.randint(
            -256, 256, (20, 32, 256, 256), dtype=torch.half, device="cuda"
        )
        # 创建一个自适应平均池化层，指定输出尺寸
        avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        # 对输入进行池化操作
        out = avg_pool(input)
        # 断言输出张量中不包含无穷大的值
        self.assertFalse(torch.isinf(out).any())
        # 断言输出张量中不包含 NaN 值
        self.assertFalse(torch.isnan(out).any())
    
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    # 如果 CUDA 不可用，则跳过测试
    def test_adaptive_avg_pooling_nhwc_overflow(self):
        # 生成一个在 CUDA 设备上的随机整数张量作为输入
        input = torch.randint(
            -256, 256, (20, 32, 256, 256), dtype=torch.half, device="cuda"
        )
        # 将输入张量的存储格式转换为通道为最后一维
        input = input.contiguous(memory_format=torch.channels_last)
        # 创建一个自适应平均池化层，指定输出尺寸
        avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        # 对输入进行池化操作
        out = avg_pool(input)
        # 断言输出张量中不包含无穷大的值
        self.assertFalse(torch.isinf(out).any())
        # 断言输出张量中不包含 NaN 值
        self.assertFalse(torch.isnan(out).any())
    
    def test_MaxUnpool2d_output_size(self):
        # 创建一个最大池化层，指定池化窗口和步幅，并返回池化的索引
        m = nn.MaxPool2d(3, stride=2, return_indices=True)
        # 创建一个最大反池化层，指定池化窗口和步幅
        mu = nn.MaxUnpool2d(3, stride=2)
        # 创建一个随机张量
        big_t = torch.rand(1, 1, 6, 6)
        # 将张量的特定位置值设为 100
        big_t[0][0][4][4] = 100
        # 对大张量进行最大池化操作，返回池化后的输出和池化时的索引
        output_big, indices_big = m(big_t)
        # 引发运行时错误，尝试使用最大反池化层对输出进行反池化操作
        self.assertRaises(RuntimeError, lambda: mu(output_big, indices_big))
    
        # 创建一个较小的随机张量
        small_t = torch.rand(1, 1, 5, 5)
        # 将张量的特定位置值设为 100
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                small_t[:, :, i, j] = 100
        # 对小张量进行最大池化操作，返回池化后的输出和池化时的索引
        output_small, indices_small = m(small_t)
        # 遍历指定尺寸范围，使用最大反池化层对输出进行反池化操作
        for h in range(3, 10):
            for w in range(3, 10):
                if 4 <= h <= 6 and 4 <= w <= 6:
                    size = (h, w)
                    if h == 6:
                        size = (1, 1) + size
    
                    # 尝试使用最大反池化层对输出进行反池化操作，指定输出尺寸
                    mu(output_small, indices_small, output_size=size)
                else:
                    # 引发值错误，尝试使用最大反池化层对输出进行反池化操作
                    self.assertRaises(
                        ValueError, lambda: mu(output_small, indices_small, (h, w))
                    )
    # 定义一个测试函数，用于测试 MaxUnpool2d 操作在 NHWC 格式的 CPU 环境下的表现
    def test_max_unpool2d_nhwc_cpu(self):
        # 生成一个形状为 (2, 10, 9, 9) 的随机张量作为输入，并转换为 float 类型，并在 CPU 上操作
        input = torch.randn(2, 10, 9, 9).float().cpu()
        # 将输入张量转换为 channels_last 内存格式（NHWC）
        input = input.contiguous(memory_format=torch.channels_last)
        # 创建一个输入的副本，并保持其连续性
        ref_input = input.clone().contiguous()

        # 创建一个 CPU 上的 MaxPool2d 层，池化窗口为 3x3，步长为 2，返回池化结果和索引
        pool = nn.MaxPool2d(3, stride=2, return_indices=True).cpu()
        # 创建一个用于参考的 MaxPool2d 层，参数与 pool 相同
        ref_pool = nn.MaxPool2d(3, stride=2, return_indices=True).cpu()

        # 对输入进行池化操作，得到池化后的输出和索引
        out, ind = pool(input)
        # 对参考输入进行池化操作，得到参考输出和索引
        ref_out, ref_ind = ref_pool(ref_input)
        # 设置输出张量需要计算梯度
        out.requires_grad_()
        # 设置参考输出张量需要计算梯度
        ref_out.requires_grad_()

        # 创建一个 CPU 上的 MaxUnpool2d 层，反池化窗口为 3x3，步长为 2
        unpool = nn.MaxUnpool2d(3, stride=2).cpu()
        # 创建一个用于参考的 MaxUnpool2d 层，参数与 unpool 相同
        ref_unpool = nn.MaxUnpool2d(3, stride=2).cpu()

        # 对池化后的输出进行反池化操作，得到反池化后的输出
        upout = unpool(out, ind)
        # 对参考池化后的输出进行反池化操作，得到参考反池化后的输出
        ref_upout = ref_unpool(ref_out, ref_ind)

        # 创建一个形状与 upout 相同的随机张量作为梯度，并在 CPU 上操作
        grad = torch.randn(upout.size()).float().cpu()
        # 将梯度张量转换为 channels_last 内存格式（NHWC）
        grad = grad.contiguous(memory_format=torch.channels_last)
        # 创建一个梯度张量的副本，并保持其连续性
        ref_grad = grad.clone().contiguous()

        # 对反池化后的输出进行反向传播，计算梯度
        upout.backward(grad)
        # 对参考反池化后的输出进行反向传播，计算参考梯度
        ref_upout.backward(ref_grad)

        # 断言反池化后的输出仍然保持 channels_last 内存格式（NHWC）
        self.assertTrue(upout.is_contiguous(memory_format=torch.channels_last))
        # 断言参考反池化后的输出仍然保持默认内存格式
        self.assertTrue(ref_upout.is_contiguous())
        # 断言反池化后的输出与参考反池化后的输出在数值上相近
        self.assertTrue(torch.allclose(upout, ref_upout))
        # 断言池化前的输出的梯度与参考池化前的输出的梯度在数值上相近
        self.assertTrue(torch.allclose(out.grad, ref_out.grad))
    def test_max_unpool(self):
        with set_default_dtype(torch.double):
            # Test 1D
            # 生成一个 1x1x4 的张量，进行 1D 最大池化操作，返回池化结果和索引
            output, indices = F.max_pool1d(
                torch.randn([1, 1, 4]), 2, stride=2, return_indices=True
            )
            # 使用 max_unpool1d 函数对池化结果进行反池化操作，比较两种调用方式的结果
            self.assertEqual(
                F.max_unpool1d(output, indices, 2),
                F.max_unpool1d(output, indices, 2, stride=2),
            )

            # Test list / tuple passed as argument to max_unpool1d
            # 生成一个 1x1x5 的张量，进行 1D 最大池化操作，返回池化结果和索引
            input = torch.randn([1, 1, 5], requires_grad=True)
            output, indices = F.max_pool1d(input, 2, stride=2, return_indices=True)
            # 使用 max_unpool1d 函数对池化结果进行反池化操作，比较两种调用方式的结果
            self.assertEqual(
                F.max_unpool1d(output, indices, 2, stride=2, output_size=input.shape),
                F.max_unpool1d(output, indices, 2, stride=2, output_size=input.size()),
            )
            # 对 max_unpool1d 函数进行梯度检查
            gradcheck(F.max_unpool1d, (output, indices, 2), check_forward_ad=True)

            # Test 2D
            # 生成一个 1x1x4x4 的张量，进行 2D 最大池化操作，返回池化结果和索引
            output, indices = F.max_pool2d(
                torch.randn([1, 1, 4, 4], requires_grad=True),
                2,
                stride=2,
                return_indices=True,
            )
            # 使用 max_unpool2d 函数对池化结果进行反池化操作，比较两种调用方式的结果
            self.assertEqual(
                F.max_unpool2d(output, indices, 2),
                F.max_unpool2d(output, indices, 2, stride=2),
            )
            # 对 max_unpool2d 函数进行梯度检查
            gradcheck(F.max_unpool2d, (output, indices, 2), check_forward_ad=True)

            # Test 3D
            # 生成一个 4x4x4x4x4 的张量，进行 3D 最大池化操作，返回池化结果和索引
            output, indices = F.max_pool3d(
                torch.randn([4, 4, 4, 4, 4], requires_grad=True),
                2,
                stride=2,
                return_indices=True,
            )
            # 使用 max_unpool3d 函数对池化结果进行反池化操作，比较两种调用方式的结果
            self.assertEqual(
                F.max_unpool3d(output, indices, 2),
                F.max_unpool3d(output, indices, 2, stride=2),
            )
            # 对 max_unpool3d 函数进行梯度检查
            gradcheck(F.max_unpool3d, (output, indices, 2), check_forward_ad=True)

    def test_max_unpool3d_input_check(self):
        # 创建一个大小为 [1, 3, 1, 1, 1] 的张量，所有元素初始化为 1
        x = torch.ones(1, 3, 1, 1, 1)
        # 使用 max_unpool3d 函数并传入错误的索引张量，检查是否引发 RuntimeError
        with self.assertRaises(RuntimeError):
            F.max_unpool3d(x, torch.zeros(x.shape, dtype=int), [1, 1])

    def test_quantized_max_pool1d_empty_kernel(self):
        # 当使用空内核调用 torch.quantized_max_pool1d 函数时，此前会导致 segfault
        # 参见 https://github.com/pytorch/pytorch/issues/116323
        # 创建一个随机张量 base，然后将其量化为 temp_tensor
        base = torch.randn(1)
        temp_tensor = torch.quantize_per_tensor(base, 0.1, 10, torch.quint2x4)
        # 使用空内核调用 torch.quantized_max_pool1d 函数，检查是否引发 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.quantized_max_pool1d(temp_tensor, [])
# 定义一个测试类 TestPoolingNNDeviceType，继承自 NNTestCase，用于测试池化层在不同设备类型上的行为
class TestPoolingNNDeviceType(NNTestCase):

    # 标记测试仅适用于原生设备类型，同时指定数据类型为 torch.float 和 torch.double
    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    # 定义测试方法 test_adaptive_pooling_zero_batch，用于测试自适应池化在零批次情况下的行为
    def test_adaptive_pooling_zero_batch(self, dtype, device):
        # 创建一个零输入的张量，大小为 (0, 10)，指定数据类型和设备类型
        inp = torch.ones(0, 10, dtype=dtype, device=device)
        # 创建一个 AdaptiveAvgPool1d 模块，并将其移动到指定的设备上
        mod = torch.nn.AdaptiveAvgPool1d(5).to(device)
        # 调用 _test_module_empty_input 函数来测试模块在空输入下的行为，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 创建一个零输入的张量，大小为 (0, 10, 10)，指定数据类型和设备类型
        inp = torch.ones(0, 10, 10, dtype=dtype, device=device)
        # 创建一个 AdaptiveAvgPool2d 模块，并将其移动到指定的设备上
        mod = torch.nn.AdaptiveAvgPool2d((5, 5)).to(device)
        # 调用 _test_module_empty_input 函数来测试模块在空输入下的行为，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 创建一个零输入的张量，大小为 (0, 10, 10, 10)，指定数据类型和设备类型
        inp = torch.ones(0, 10, 10, 10, dtype=dtype, device=device)
        # 创建一个 AdaptiveAvgPool3d 模块，并将其移动到指定的设备上
        mod = torch.nn.AdaptiveAvgPool3d((5, 5, 5)).to(device)
        # 调用 _test_module_empty_input 函数来测试模块在空输入下的行为，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

    # 这些测试用例用于验证自适应池化和其变种函数在输出大小为 0 时反向传播时是否引发错误
    # 因为 ErrorInputs 不支持反向调用，所以这些测试用例被显式编写
    # Issue: https://github.com/pytorch/pytorch/issues/78868
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    # 定义测试方法 test_adaptive_pooling_empty_output_size，用于测试输出大小为 0 时的自适应池化函数行为
    def test_adaptive_pooling_empty_output_size(self, dtype, device):
        # 设置错误消息，用于检查梯度输出在非批处理维度上的大小是否为非零
        error_msg = (
            "Expected grad_output to have non-zero size for non-batch dimensions"
        )

        # 创建部分函数 make_tensor，生成指定设备和数据类型的张量，并设置 requires_grad 为 True
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        # 创建输入张量，大小为 (1, 64, 10, 9)，作为 make_arg 函数的参数
        input = make_arg((1, 64, 10, 9))
        # 输出大小设为 0
        output_size = 0

        # 定义要测试的自适应池化函数列表
        fns = (
            nn.functional.adaptive_avg_pool2d,
            nn.functional.adaptive_avg_pool3d,
            nn.functional.adaptive_max_pool2d,
            nn.functional.adaptive_max_pool3d,
        )

        # 遍历函数列表，对每个函数进行测试
        for fn in fns:
            # 使用 assertRaisesRegex 检查 RuntimeError 是否抛出指定错误消息
            with self.assertRaisesRegex(RuntimeError, error_msg):
                # 调用自适应池化函数，并对结果求和后进行反向传播
                fn(input, output_size).sum().backward()

        # 定义要测试的另一组自适应池化函数列表
        fns2 = (
            nn.functional.adaptive_avg_pool1d,
            nn.functional.adaptive_max_pool1d,
        )
        # 创建输入张量，大小为 (1, 64)，作为 make_arg 函数的参数
        input2 = make_arg((1, 64))

        # 遍历函数列表，对每个函数进行测试
        for fn in fns2:
            # 使用 assertRaisesRegex 检查 RuntimeError 是否抛出指定错误消息
            with self.assertRaisesRegex(RuntimeError, error_msg):
                # 调用自适应池化函数，并对结果求和后进行反向传播
                fn(input2, output_size).sum().backward()

    # 标记测试仅适用于原生设备类型
    @onlyNativeDeviceTypes
    # 定义测试方法 test_FractionalMaxPool2d_zero_batch，用于测试分数最大池化在零批次情况下的行为
    def test_FractionalMaxPool2d_zero_batch(self, device):
        # 创建 FractionalMaxPool2d 模块，设置输出比例为 (0.5, 0.5)
        mod = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        # 创建一个零输入的张量，大小为 (0, 16, 50, 32)，指定设备类型
        inp = torch.ones(0, 16, 50, 32, device=device)
        # 调用 _test_module_empty_input 函数来测试模块在空输入下的行为，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 使用 assertRaisesRegex 检查 RuntimeError 是否抛出指定错误消息
        with self.assertRaisesRegex(RuntimeError, "Expected input"):
            # 创建一个 1x0x50x32 的随机输入张量，指定设备类型
            inp = torch.randn(1, 0, 50, 32, device=device)
            # 对模块进行调用
            mod(inp)

    # 标记测试仅适用于原生设备类型
    @onlyNativeDeviceTypes
    # 定义测试函数，用于测试在输入为空时的 FractionalMaxPool3d 行为
    def test_FractionalMaxPool3d_zero_batch(self, device):
        # 创建 FractionalMaxPool3d 模块，设置输出比例为 (0.5, 0.5, 0.5)，并将其移动到指定设备上
        mod = nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5)).to(device)
        # 创建一个空的输入张量，形状为 [0, 16, 50, 32, 32]，并进行测试
        inp = torch.ones(0, 16, 50, 32, 32, device=device)
        # 调用辅助函数 _test_module_empty_input，验证模块处理空输入时的行为，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证其消息为 "Expected input"
        with self.assertRaisesRegex(RuntimeError, "Expected input"):
            # 创建一个形状为 [1, 0, 50, 32, 32] 的随机输入张量，并调用 mod 进行处理
            inp = torch.randn(1, 0, 50, 32, 32, device=device)
            mod(inp)

    # 标记为只能在本地设备类型上运行的测试函数
    @onlyNativeDeviceTypes
    # 定义测试函数，用于测试在输出大小为零时的 FractionalMaxPool2d 行为
    def test_FractionalMaxPool2d_zero_out_size(self, device):
        # 创建 FractionalMaxPool2d 模块，设置输出大小为 [0, 1]，并将其移动到指定设备上
        mod = nn.FractionalMaxPool2d([2, 2], output_size=[0, 1])
        # 创建一个随机填充的输入张量，形状为 [16, 50, 32, 32]，并进行测试
        inp = torch.rand([16, 50, 32, 32], device=device)
        # 调用 mod 处理输入，得到输出张量 out
        out = mod(inp)
        # 使用断言验证输出张量 out 的形状与期望的空张量形状 (16, 50, 0, 1) 相同
        self.assertEqual(out, torch.empty((16, 50, 0, 1), device=device))

    # 标记为只能在本地设备类型上运行的测试函数
    @onlyNativeDeviceTypes
    # 定义测试函数，用于测试在输出大小为零时的 FractionalMaxPool3d 行为
    def test_FractionalMaxPool3d_zero_out_size(self, device):
        # 创建 FractionalMaxPool3d 模块，设置输出大小为 [0, 1, 1]，并将其移动到指定设备上
        mod = nn.FractionalMaxPool3d([3, 2, 2], output_size=[0, 1, 1])
        # 创建一个随机填充的输入张量，形状为 [16, 50, 32, 32]，并进行测试
        inp = torch.rand([16, 50, 32, 32], device=device)
        # 调用 mod 处理输入，得到输出张量 out
        out = mod(inp)
        # 使用断言验证输出张量 out 的形状与期望的空张量形状 (16, 0, 1, 1) 相同
        self.assertEqual(out, torch.empty((16, 0, 1, 1), device=device))

    # 标记为只能在本地设备类型上运行的测试函数
    @onlyNativeDeviceTypes
    # 定义测试函数，用于测试在样本数为零时的 FractionalMaxPool2d 行为
    def test_FractionalMaxPool2d_zero_samples(self, device):
        # 创建一个形状为 [0, 16, 2] 的随机样本张量
        samples = torch.rand([0, 16, 2], device=device)
        # 创建 FractionalMaxPool2d 模块，设置输出大小为 [1, 1]，并传入上述随机样本张量
        mod = nn.FractionalMaxPool2d([2, 2], output_size=[1, 1], _random_samples=samples)
        # 创建一个形状为 [0, 16, 32, 32] 的随机输入张量，并进行测试
        inp = torch.randn([0, 16, 32, 32], device=device)
        # 调用 mod 处理输入，得到输出张量 out
        out = mod(inp)
        # 使用断言验证输出张量 out 的形状与期望的空张量形状 (0, 16, 1, 1) 相同
        self.assertEqual(out, torch.empty((0, 16, 1, 1), device=device))

        # 创建一个形状为 [1, 16, 32, 32] 的随机输入张量，并使用 assertRaisesRegex 断言捕获 RuntimeError 异常
        inp1 = torch.randn([1, 16, 32, 32], device=device)
        with self.assertRaisesRegex(RuntimeError, "Expect _random_samples"):
            # 调用 mod 处理输入 inp1，验证模块在缺少 _random_samples 时是否抛出异常
            out1 = mod(inp1)

    # 标记为只能在本地设备类型上运行的测试函数
    @onlyNativeDeviceTypes
    # 定义测试函数，用于测试在样本数为零时的 FractionalMaxPool3d 行为
    def test_FractionalMaxPool3d_zero_samples(self, device):
        # 创建一个形状为 [0, 16, 3] 的随机样本张量
        samples = torch.rand([0, 16, 3], device=device)
        # 创建 FractionalMaxPool3d 模块，设置输出大小为 [1, 1, 1]，并传入上述随机样本张量
        mod = nn.FractionalMaxPool3d([3, 2, 2], output_size=[1, 1, 1], _random_samples=samples)
        # 创建一个形状为 [0, 16, 50, 32, 32] 的随机输入张量，并进行测试
        inp = torch.randn([0, 16, 50, 32, 32], device=device)
        # 调用 mod 处理输入，得到输出张量 out
        out = mod(inp)
        # 使用断言验证输出张量 out 的形状与期望的空张量形状 (0, 16, 1, 1, 1) 相同
        self.assertEqual(out, torch.empty((0, 16, 1, 1, 1), device=device))

        # 创建一个形状为 [1, 16, 50, 32, 32] 的随机输入张量，并使用 assertRaisesRegex 断言捕获 RuntimeError 异常
        inp1 = torch.randn([1, 16, 50, 32, 32], device=device)
        with self.assertRaisesRegex(RuntimeError, "Expect _random_samples"):
            # 调用 mod 处理输入 inp1，验证模块在缺少 _random_samples 时是否抛出异常
            out1 = mod(inp1)
    # 定义一个测试方法，用于测试 MaxPool1d 在输入批次维度为零时的行为
    def test_MaxPool_zero_batch_dim(self, device):
        # 创建一个形状为 (0, 16, 50) 的随机张量作为输入
        inp = torch.randn(0, 16, 50, device=device)
        # 创建一个 MaxPool1d 模块，设置池化窗口大小为 3，步幅为 2，并移动到指定设备
        mod = torch.nn.MaxPool1d(3, stride=2).to(device)
        # 调用辅助函数，测试空输入情况，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 1D 池化在输入元素数为 0 的情况下，应该能正常处理，因此不测试引发错误的情况。

        # 创建一个形状为 (0, 16, 50, 32) 的随机张量作为输入
        inp = torch.randn(0, 16, 50, 32, device=device)
        # 创建一个 MaxPool2d 模块，设置池化窗口大小为 3，步幅为 2，并移动到指定设备
        mod = torch.nn.MaxPool2d(3, stride=2).to(device)
        # 调用辅助函数，测试空输入情况，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 使用断言检查运行时错误是否包含特定字符串，预期引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "Expected"):
            # 创建一个形状为 (1, 0, 50, 32) 的随机张量作为输入
            inp = torch.randn(1, 0, 50, 32, device=device)
            # 将输入传递给 MaxPool2d 模块
            mod(inp)

        # 创建一个形状为 (0, 16, 50, 44, 31) 的全为 1 的张量作为输入
        inp = torch.ones(0, 16, 50, 44, 31, device=device)
        # 创建一个 MaxPool3d 模块，设置池化窗口大小为 3，步幅为 2，并移动到指定设备
        mod = torch.nn.MaxPool3d(3, stride=2).to(device)
        # 调用辅助函数，测试空输入情况，不检查输出大小
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 使用断言检查运行时错误是否包含特定字符串，预期引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "Expected"):
            # 创建一个形状为 (1, 0, 50, 44, 31) 的全为 1 的张量作为输入
            inp = torch.ones(1, 0, 50, 44, 31, device=device)
            # 将输入传递给 MaxPool3d 模块
            mod(inp)

    # 标记为仅在原生设备类型上运行的测试方法
    @onlyNativeDeviceTypes
    # 定义一个测试方法，用于测试 MaxUnpool 在输入批次维度为零时的行为
    def test_MaxUnpool_zero_batch_dim(self, device):
        # 创建一个 MaxPool1d 模块，设置池化窗口大小为 2，步幅为 2，同时返回索引，并移动到指定设备
        pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True).to(device)
        # 创建一个 MaxUnpool1d 模块，设置池化窗口大小为 2，步幅为 2，并移动到指定设备
        unpool = torch.nn.MaxUnpool1d(2, stride=2).to(device)
        # 创建一个形状为 (0, 10, 10) 的随机张量作为输入，要求梯度计算，移动到指定设备
        inp = torch.randn(0, 10, 10, requires_grad=True, device=device)
        # 对输入进行池化操作，并获取池化输出和索引
        output, indices = pool(inp)
        # 将池化输出设置为需要计算梯度
        output.requires_grad_(True)
        # 对池化输出进行反池化操作
        unpool_out = unpool(output, indices)
        # 对反池化输出求和，并进行反向传播
        unpool_out.sum().backward()

        # 使用断言检查输入张量的梯度是否全为零张量
        self.assertEqual(inp.grad, torch.zeros_like(inp))
        # 使用断言检查反池化输出是否为全零张量
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

        # 创建一个 MaxPool2d 模块，设置池化窗口大小为 2，步幅为 2，同时返回索引，并移动到指定设备
        pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True).to(device)
        # 创建一个 MaxUnpool2d 模块，设置池化窗口大小为 2，步幅为 2，并移动到指定设备
        unpool = torch.nn.MaxUnpool2d(2, stride=2).to(device)
        # 创建一个形状为 (0, 10, 10, 10) 的随机张量作为输入，要求梯度计算，移动到指定设备
        inp = torch.randn(0, 10, 10, 10, requires_grad=True, device=device)
        # 对输入进行池化操作，并获取池化输出和索引
        output, indices = pool(inp)
        # 对反池化输出进行反池化操作
        unpool_out = unpool(output, indices)
        # 对反池化输出求和，并进行反向传播
        unpool_out.sum().backward()

        # 使用断言检查输入张量的梯度是否全为零张量
        self.assertEqual(inp.grad, torch.zeros_like(inp))
        # 使用断言检查反池化输出是否为全零张量
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

        # 创建一个 MaxPool3d 模块，设置池化窗口大小为 2，步幅为 2，同时返回索引，并移动到指定设备
        pool = torch.nn.MaxPool3d(2, stride=2, return_indices=True).to(device)
        # 创建一个 MaxUnpool3d 模块，设置池化窗口大小为 2，步幅为 2，并移动到指定设备
        unpool = torch.nn.MaxUnpool3d(2, stride=2).to(device)
        # 创建一个形状为 (0, 10, 10, 10, 10) 的随机张量作为输入，要求梯度计算，移动到指定设备
        inp = torch.randn(0, 10, 10, 10, 10, requires_grad=True, device=device)
        # 对输入进行池化操作，并获取池化输出和索引
        output, indices = pool(inp)
        # 将池化输出设置为需要计算梯度
        output.requires_grad_(True)
        # 对池化输出进行反池化操作
        unpool_out = unpool(output, indices)
        # 对反池化输出求和，并进行反向传播
        unpool_out.sum().backward()

        # 使用断言检查输入张量的梯度是否全为零张量
        self.assertEqual(inp.grad, torch.zeros_like(inp))
        # 使用断言检查反池化输出是否为全零张量
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

    # 标记为缓慢测试
    @slowTest
    # 标记为仅在原生设备类型上运行的测试方法
    @onlyNativeDeviceTypes
    # 如果在 ROCm 平台上不使用 CUDA，则跳过测试
    @skipCUDAIfRocm
    @parametrize_test(
        "module_name,module_size,output_size,test_index,should_error",
        [  # 参数化测试，定义多组参数用于测试
            # Some tests are failing in trunk https://github.com/pytorch/pytorch/issues/103854
            subtest(
                ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), -1, True),  # 子测试案例1
                name="case1",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), 2 * 2 * 4 * 5, True),  # 子测试案例2
                name="case2",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool2d", (2, 2), (1, 3, 4, 5), (2 * 2 * 4 * 5) - 1, False),  # 子测试案例3
                name="case3",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool2d", (2, 3), (2, 1, 4, 2), 2 * 3 * 4 * 2, True),  # 子测试案例4
                name="case4",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool2d", (2, 3), (2, 1, 4, 2), (2 * 3 * 4 * 2) - 1, False),  # 子测试案例5
                name="case5",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool3d", (2, 2, 2), (1, 3, 4, 5), -1, True),  # 子测试案例6
                name="case6",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool3d", (2, 2, 2), (1, 3, 4, 5), 2 * 2 * 2 * 3 * 4 * 5, True),  # 子测试案例7
                name="case7",  # 测试案例名称
            ),
            subtest(
                (
                    "MaxUnpool3d",
                    (2, 2, 2),
                    (1, 3, 4, 5),
                    (2 * 2 * 2 * 3 * 4 * 5) - 1,
                    False,
                ),  # 子测试案例8
                name="case8",  # 测试案例名称
            ),
            subtest(
                ("MaxUnpool3d", (2, 2, 2), (2, 3, 4, 1), 2 * 2 * 2 * 3 * 4 * 1, True),  # 子测试案例9
                name="case9",  # 测试案例名称
            ),
            subtest(
                (
                    "MaxUnpool3d",
                    (2, 2, 2),
                    (2, 3, 4, 1),
                    (2 * 2 * 2 * 3 * 4 * 1) - 1,
                    False,
                ),  # 子测试案例10
                name="case10",  # 测试案例名称
            ),
        ],
    )
    def test_MaxUnpool_index_errors(
        self, device, module_name, module_size, output_size, test_index, should_error
    ):
        # NOTE: CUDA tests need to be run in a subprocess because they cause device asserts
        # 如果运行在 CUDA 设备上，需要在子进程中执行测试，因为它们可能导致设备断言
        if torch.device(device).type == "cuda":
            # 定义 CUDA 设备下的错误消息字典
            error_msgs = {
                "MaxUnpool2d": r"Assertion `maxind >= 0 && maxind < outputImageSize` failed",
                "MaxUnpool3d": r"Assertion `index >= 0 && index < outputImageSize` failed",
            }

            # 构建用于在子进程中运行的脚本
            script = f"""
# 导入 PyTorch 库
import torch
# 使用指定的未命名模块和大小创建一个未初始化的对象，并将其移到指定设备上
unpool = torch.nn.{module_name}({module_size}).to('{device}')
# 创建一个指定大小、在指定设备上的随机张量
output = torch.rand({output_size}, dtype=torch.float32, device='{device}')
# 创建一个指定大小、在指定设备上的零张量，用于存储索引
indices = torch.zeros({output_size}, dtype=torch.int64, device='{device}')
# 将展平后的索引张量的第一个元素设置为指定的测试索引
indices.flatten()[0] = {test_index}
# 对创建的 unpool 对象执行操作，传入输出张量和索引张量
unpool(output, indices)
# 等待 CUDA 操作完成
torch.cuda.synchronize()

"""
    p = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        capture_output=True,
        text=True,
    )

    output = p.stdout + "\n" + p.stderr

    error_msg = error_msgs[module_name]

    # 如果应该出现错误，则检查输出中是否包含预期错误消息
    if should_error:
        self.assertIn(error_msg, output, "The expected error was not found")
    else:
        # 否则，检查输出中是否不包含 "Error"
        self.assertNotIn("Error", output, "Should not have produced an error")
else:
    # 获取指定模块名的模块类，并将其实例化为 unpool 对象
    module_class = getattr(torch.nn, module_name)
    unpool = module_class(module_size).to(device)
    # 创建指定大小、在指定设备上的随机输出张量
    output = torch.rand(output_size, dtype=torch.float32, device=device)
    # 创建指定大小、在指定设备上的零索引张量
    indices = torch.zeros(output_size, dtype=torch.int64, device=device)
    # 将展平后的索引张量的第一个元素设置为指定的测试索引
    indices.flatten()[0] = test_index

    # 如果应该出现错误
    if should_error:
        # 使用断言检查是否引发了预期的 RuntimeError 错误
        with self.assertRaisesRegex(
            RuntimeError, r"Found an invalid max index:"
        ):
            unpool(output, indices)
    else:
        # 否则，正常调用 unpool 对象进行操作
        unpool(output, indices)
    # 定义测试函数，用于测试 AvgPool2d 对象处理空输入的行为
    def test_AvgPool2d_empty(self, device):
        # 创建一个 AvgPool2d 对象，设置池化窗口大小为 3x3，步幅为 2
        avgpool = torch.nn.AvgPool2d(3, stride=2).to(device)
        # 创建一个空的张量作为输入，形状为 [0, 16, 20, 32]，在指定设备上生成随机数
        inp = torch.randn(0, 16, 20, 32, device=device)
        # 调用辅助函数 _test_module_empty_input，验证 AvgPool2d 对象处理空输入的行为
        _test_module_empty_input(self, avgpool, inp, check_size=False)

        # 创建一个在通道最后内存格式下连续的空张量作为输入
        clast_inp = torch.randn(0, 16, 20, 32, device=device).contiguous(
            memory_format=torch.channels_last
        )
        # 再次调用 _test_module_empty_input，验证在通道最后内存格式下处理空输入的行为
        _test_module_empty_input(self, avgpool, clast_inp, check_size=False)

        # 测试处理空的非批量输入，期望抛出 RuntimeError 并包含 "3D or 4D" 的错误信息
        with self.assertRaisesRegex(RuntimeError, "3D or 4D"):
            # 创建一个形状为 [16, 0, 20, 32] 的空张量，在指定设备上生成随机数
            inp = torch.randn(16, 0, 20, 32, device=device)
            # 调用 AvgPool2d 对象处理空输入，预期引发异常
            avgpool(inp)

    # 定义测试函数，用于验证池化操作的输出形状计算
    def test_pooling_shape(self, device):
        """Test the output shape calculation for pooling functions"""

        # 定义检查函数，验证池化函数在不同维度下的输出形状是否符合预期
        def check(expected_out_shape, sizes, *args, **kwargs):
            for kernel in ["max", "avg"]:
                for i in [1, 2, 3]:
                    # 如果 torch.nn.functional 中存在对应的池化函数，获取并执行
                    if hasattr(torch.nn.functional, f"{kernel}_pool{i}d"):
                        op = getattr(torch.nn.functional, f"{kernel}_pool{i}d")
                        # 创建一个随机张量 t，形状根据 sizes 和维度 i 生成，放置在指定设备上
                        t = torch.randn(sizes[: i + 2], device=device)
                        # 断言池化函数的输出形状与期望的输出形状匹配
                        self.assertEqual(
                            op(t, *args, **kwargs).shape, expected_out_shape[: i + 2]
                        )

        # 使用 check 函数验证不同参数组合下的池化输出形状
        check(
            (1, 1, 3, 3, 4),
            (1, 1, 5, 6, 7),
            kernel_size=1,
            stride=2,
            padding=0,
            ceil_mode=True,
        )
        check(
            (1, 1, 2, 3, 3),
            (1, 1, 3, 4, 5),
            kernel_size=2,
            stride=2,
            padding=1,
            ceil_mode=False,
        )
        check(
            (1, 1, 2, 3, 3),
            (1, 1, 3, 4, 5),
            kernel_size=2,
            stride=2,
            padding=1,
            ceil_mode=True,
        )

        # 测试来自 GitHub 问题 https://github.com/pytorch/pytorch/issues/45357 的特定情况
        x = torch.randn(1, 1, 6, 7, device=device)
        # 调用 max_pool2d 函数，使用特定参数进行池化操作，预期输出的形状为 (1, 1, 3, 4)
        y = torch.nn.functional.max_pool2d(
            x, 1, stride=(2, 2), padding=0, ceil_mode=True
        )
        # 断言池化后的输出张量形状是否符合预期
        self.assertEqual(y.size(), (1, 1, 3, 4))

    @onlyNativeDeviceTypes  # TODO: fix on XLA
    # 定义一个测试函数，用于测试 AdaptiveAvgPool2d 输出大小为 (1, 1) 的情况
    def test_adaptive_avg_pool2d_output_size_one(self, device):
        
        # 定义一个辅助函数 helper，用于测试不同情况下的输入
        def helper(size, memory_format):
            # 创建一个张量 x，其形状由 size 指定，数据类型为 float，存储设备为 device，支持梯度计算
            x = torch.randint(
                1, 10, size, dtype=torch.float, device=device, requires_grad=True
            )
            
            # 根据 memory_format 调整张量 x 的存储顺序或子采样
            if memory_format == "non_contiguous":
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            # 创建一个 AdaptiveAvgPool2d 模块 net，目标输出尺寸为 (1, 1)
            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            # 对输入张量 x 进行自适应平均池化
            out = net(x)
            # 计算参考输出 ref_out，先将 x 进行连续化处理后取平均，并调整形状
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            # 对输出 out 执行反向传播以确保不崩溃
            out.sum().backward()

            # 断言输出 out 与参考输出 ref_out 相等
            self.assertEqual(out, ref_out)
            
            # 如果 memory_format 为 torch.channels_last，则验证输出 out 在 channels_last 存储格式下是连续的，并检查其步长
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                # 否则，验证输出 out 是连续的，并检查其步长
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        # 遍历测试的内存格式：torch.contiguous_format、torch.channels_last 和 "non_contiguous"
        for mf in (torch.contiguous_format, torch.channels_last, "non_contiguous"):
            # 调用 helper 函数进行测试
            helper((2, 3, 6, 6), mf)

    # 声明一个仅适用于本地设备类型的测试函数，用于测试 AdaptiveAvgPool3d 输出大小为 (1, 1, 1) 的情况
    @onlyNativeDeviceTypes
    def test_adaptive_avg_pool3d_output_size_one(self, device):
        # 创建一个形状为 (2, 3, 6, 6, 6) 的张量 x，数据类型为 float，存储设备为 device，支持梯度计算
        x = torch.randn(
            (2, 3, 6, 6, 6), dtype=torch.float, device=device, requires_grad=True
        )

        # 创建一个 AdaptiveAvgPool3d 模块 net，目标输出尺寸为 (1, 1, 1)
        net = torch.nn.AdaptiveAvgPool3d(1)
        # 对输入张量 x 进行自适应平均池化
        out = net(x)
        # 计算参考输出 ref_out，先将 x 进行连续化处理后取平均，并调整形状使其与 out 相同
        ref_out = x.contiguous().mean((-1, -2, -3)).view(out.shape)

        # 对输出 out 执行反向传播以确保不崩溃
        out.sum().backward()

        # 断言输出 out 与参考输出 ref_out 相等
        self.assertEqual(out, ref_out)
        
        # 验证输出 out 是连续的，并检查其步长
        self.assertTrue(out.is_contiguous())
        c = out.size(1)
        self.assertEqual(out.stride(), [c, 1, 1, 1, 1])

    # 标记一个预期失败的元数据，测试自适应池化在不支持的输入类型下是否会引发 RuntimeError
    @expectedFailureMeta  # Runtime Error not raised for meta
    @onlyNativeDeviceTypes
    @dtypes(torch.uint8, torch.int8, torch.short, torch.int, torch.long)
    def test_adaptive_pooling_no_suppot_input(self, device, dtype):
        # 遍历 numel 取值为 2 和 3，以及 pool_type 取值为 "Max" 和 "Avg"
        for numel in (2, 3):
            for pool_type in ("Max", "Avg"):
                # 构造自适应池化类名
                cls_name = f"Adaptive{pool_type}Pool{numel}d"
                # 根据类名获取对应的自适应池化模块类
                module_cls = getattr(nn, cls_name)
                # 设定输出大小
                output_size = (2,) * numel
                # 创建自适应池化模块实例
                module = module_cls(output_size)
                # 创建一个形状为 (4,) * (numel + 1) 的输入张量 input，存储设备为 device，数据类型为 dtype
                input = torch.randn((4,) * (numel + 1), device=device).to(dtype)
                # 使用断言验证调用该模块时是否会引发 RuntimeError（"not implemented"）
                with self.assertRaisesRegex(RuntimeError, "not implemented"):
                    output = module(input)

    # 声明一个仅适用于本地设备类型的测试函数，用于测试在 Jetson 平台下是否会进行垃圾回收
    @onlyNativeDeviceTypes
    @gcIfJetson
    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    # 定义测试函数 test_avg_pool2d_nhwc，接受设备和数据类型作为参数
    def test_avg_pool2d_nhwc(self, device, dtype):
        
        # 定义内部辅助函数 helper，用于执行 AvgPool2d 的测试
        def helper(
            n,
            c,
            h,
            w,
            kernel_size,
            stride=None,
            count_include_pad=True,
            divisor_override=None,
            padding=0,
        ):
            # 如果未指定步长，则默认与 kernel_size 相同
            if stride is None:
                stride = kernel_size
            
            # 创建随机输入张量，并设置为通道最后内存格式，需要梯度计算
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            
            # 创建随机梯度张量，大小根据池化后的尺寸确定
            grad = torch.randn(
                n,
                c,
                (h - kernel_size) // stride + 1,
                (w - kernel_size) // stride + 1,
                dtype=dtype,
                device=device,
            )
            
            # 创建 AvgPool2d 对象，指定核大小、步长、是否包含 padding 在内以及覆盖除数
            pool = torch.nn.AvgPool2d(
                kernel_size,
                stride=stride,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            ).to(device)
            
            # 创建参考输入的副本，需要梯度计算
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 创建参考梯度的副本，保持连续性
            ref_grad = grad.detach().clone().contiguous()
            # 创建另一个 AvgPool2d 对象作为参考
            ref_pool = torch.nn.AvgPool2d(
                kernel_size,
                stride=stride,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            ).to(device)
            
            # 对输入进行池化操作
            out = pool(input)
            # 反向传播梯度
            out.backward(grad)
            # 对参考输入进行池化操作
            ref_out = ref_pool(ref_input)
            # 参考输出进行反向传播
            ref_out.backward(ref_grad)
            
            # 断言输出是通道最后内存格式
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            # 断言参考输出是连续的
            self.assertTrue(ref_out.is_contiguous())
            # 断言输出与参考输出相等
            self.assertEqual(out, ref_out)
            # 断言输入的梯度与参考输入的梯度相等
            self.assertEqual(input.grad, ref_input.grad)
        
        # 调用 helper 函数，执行不同的 AvgPool2d 测试用例
        helper(4, 8, 8, 8, 3)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=1)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=2, stride=2)
        helper(4, 8, 8, 8, 3, divisor_override=42)
        helper(4, 8, 8, 8, 7)
        
        # 如果是 ROCm 平台且使用 CUDA 设备，则在运行大规模子测试前清空缓存分配器，以避免内存溢出错误
        if TEST_WITH_ROCM and "cuda" in device:
            torch.cuda.empty_cache()
        
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(4, 8, 7, 7, 3, padding=2, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    # 通过 onlyCPU 装饰器和指定的数据类型对测试函数进行标记
    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_max_pool1d_corner_cases(self, device, dtype):
        # 定义内部函数check，用于验证MaxPool1d的边界情况
        def check(x, args, expected):
            # 创建MaxPool1d模型对象
            model = torch.nn.MaxPool1d(*args)
            # 如果输入x是列表，则转换为张量，并转换期望的输出结果为张量形式
            if isinstance(x, list):
                x = torch.tensor(x, device=device, dtype=dtype)
                expected = torch.tensor(expected, device=device, dtype=dtype)
            # 断言模型的输出与期望值相等
            self.assertEqual(model(x), expected)

        # Pooling args: (kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        # 测试不同的参数组合
        check([[1]], (1, None, 0, 1, False, False), [[1]])
        check([[1]], (2, None, 1, 2, False, False), [[float("-inf")]])
        check(
            [[1], [1]],
            (2, None, 1, 2, False, False),
            [[float("-inf")], [float("-inf")]],
        )
        check([[1, 2]], (2, 1, 1, 2, False, False), [[2, 1]])
        check([[1, 2]], (2, 2, 1, 2, False, True), [[2, 2]])

    @onlyCPU
    @dtypes(torch.float, torch.double)
    @skipIfTorchDynamo("OOMs https://github.com/pytorch/pytorch/issues/111320")
    def test_max_pool1d(self, device, dtype):
        # FIXME For now compare against max_pool1d with indices
        # 定义内部函数check，用于验证MaxPool1d的一般情况
        def check(x, *args, **kwargs):
            # 创建MaxPool1d模型对象和具有返回索引的参考模型对象
            model = torch.nn.MaxPool1d(*args, **kwargs)
            ref_model = torch.nn.MaxPool1d(*args, **kwargs, return_indices=True)
            # 断言模型的输出与参考模型的输出相同
            self.assertEqual(model(x), ref_model(x)[0])

        # 生成测试用的大小、卷积核大小、步幅、扩展率、天花板模式的随机组合
        sizes = [random.sample(range(8, 128), 3) for _ in range(3)]
        kernel_sizes = random.sample(range(1, 5), 3)
        strides = random.sample(range(1, 5), 3)
        dilations = random.sample(range(1, 5), 3)
        ceil_modes = [True, False]

        # 使用itertools生成各种参数组合进行测试
        for size, kernel_size, stride, dilation, ceil_mode in itertools.product(
            sizes, kernel_sizes, strides, dilations, ceil_modes
        ):
            padding = random.sample(range(0, math.floor(kernel_size / 2) + 1), 1)
            # 调用check函数进行测试
            check(
                torch.randn(size, device=device, dtype=dtype),
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode=ceil_mode,
            )

        # Non-contiguous test
        # 测试非连续张量的情况
        tensor = torch.randn(5, 151, 33, device=device, dtype=dtype)[::2, ::3, ::2]
        check(tensor, 3, 2, 1, 2, ceil_mode=True)
        check(tensor.transpose(1, 2), 3, 2, 1, 2, ceil_mode=True)

    @onlyCUDA
    @gcIfJetson
    def test_max_pool2d(self, device):
        # 定义内部辅助函数helper，用于测试MaxPool2d
        def helper(n, c, h, w, ks):
            # 创建大小为(n, c, h, w)的随机张量，并设置requires_grad=True
            x = torch.randn(
                n, c, h, w, device="cuda", dtype=torch.float, requires_grad=True
            )
            # 创建对照张量，与x相同但在CPU上操作
            ref_x = x.detach().clone().cpu().requires_grad_()

            # 创建MaxPool2d对象
            pool = torch.nn.MaxPool2d(kernel_size=ks)

            # 对x和ref_x进行最大池化操作
            y = pool(x)
            ref_y = pool(ref_x)

            # 对y和ref_y进行梯度反向传播
            y.sum().backward()
            ref_y.sum().backward()

            # 断言池化后的结果y与ref_y相等
            self.assertEqual(y, ref_y)
            # 断言x的梯度与ref_x的梯度相等
            self.assertEqual(x.grad, ref_x.grad)

        # 测试不同的输入参数组合
        helper(2, 8, 4, 4, ks=2)
        helper(1, 100000, 32, 32, ks=4)
        helper(1, 100000, 1, 4, ks=(1, 4))  # test for max_pool1d
    # 应用装饰器，限制只在原生设备类型上运行该测试方法
    @onlyNativeDeviceTypes
    # 应用数据类型装饰器，指定输入数据类型为 torch.half, torch.bfloat16, torch.float, torch.double 中的一种
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    # 在 CUDA 设备上运行时，指定支持的数据类型为 torch.half, torch.float, torch.double 中的一种
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    # 如果运行在 Jetson 上，则启用垃圾收集
    @gcIfJetson
    # 定义测试方法 test_max_pool2d_nhwc，接受设备和数据类型参数
    def test_max_pool2d_nhwc(self, device, dtype):
        # 定义辅助函数 helper，用于测试不同参数的最大池化操作
        def helper(n, c, h, w, kernel_size, stride=None):
            # 如果未指定步长，则使用核大小作为步长
            if stride is None:
                stride = kernel_size
            # 创建随机输入张量，指定设备和数据类型
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            # 将输入张量转换为连续格式（内存布局为通道最后），并设置需要梯度计算
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            # 创建随机梯度张量，指定设备和数据类型
            grad = torch.randn(
                n,
                c,
                (h - kernel_size) // stride + 1,
                (w - kernel_size) // stride + 1,
                dtype=dtype,
                device=device,
            )
            # 创建最大池化层对象，指定核大小和步长，返回池化结果和索引
            pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
                device
            )

            # 创建参考输入张量，其梯度计算需要保持可用
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 创建参考梯度张量，并保持连续格式
            ref_grad = grad.detach().clone().contiguous()
            # 创建另一个最大池化层对象作为参考对象
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
                device
            )

            # 对输入进行最大池化操作，返回池化结果和索引，然后进行梯度反向传播
            out, ind = pool(input)
            out.backward(grad)
            # 对参考输入进行相同操作
            ref_out, ref_ind = ref_pool(ref_input)
            ref_out.backward(ref_grad)

            # 断言输出是否保持连续格式
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            # 断言索引张量是否保持连续格式
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            # 断言池化结果和参考结果是否相等
            self.assertEqual(out, ref_out)
            # 断言池化索引和参考索引是否相等
            self.assertEqual(ind, ref_ind)
            # 断言输入张量的梯度是否与参考输入张量的梯度相等
            self.assertEqual(input.grad, ref_input.grad)

        # 使用不同的参数调用 helper 函数进行测试
        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    # 应用装饰器，限制只在 CPU 设备上运行该测试方法
    @onlyCPU
    # 应用数据类型装饰器，指定输入数据类型为 torch.int32 或 torch.int64
    @dtypes(torch.int32, torch.int64)
    # 定义测试方法 test_max_pool2d_corner_cases，接受设备和数据类型参数
    def test_max_pool2d_corner_cases(self, device, dtype):
        # 定义检查函数 check，用于验证最大池化的边界情况
        def check(x, args, expected, memory_format):
            # 创建最大池化层对象 model，使用指定的参数
            model = torch.nn.MaxPool2d(*args)
            # 如果输入是列表，则转换为张量，并设置设备和数据类型，并保持指定的内存格式
            if isinstance(x, list):
                x = torch.tensor(x, device=device, dtype=dtype).to(
                    memory_format=memory_format
                )
                # 创建期望的输出张量，也设置设备和数据类型，并保持指定的内存格式
                expected = torch.tensor(expected, device=device, dtype=dtype).to(
                    memory_format=memory_format
                )
            # 断言最大池化层对输入的输出是否与期望输出相等
            self.assertEqual(model(x), expected)

        # 调用 check 函数，验证不同的输入情况下的最大池化操作
        # 使用不同的内存格式（contiguous_format 或 channels_last）进行测试
        check(
            [[[[-1, -2], [-3, -4]]]],
            (2, 2, 1, 2, False, True),
            [[[[-4, -4], [-4, -4]]]],
            torch.contiguous_format,
        )
        check(
            [[[[-1, -2], [-3, -4]]]],
            (2, 2, 1, 2, False, True),
            [[[[-4, -4], [-4, -4]]]],
            torch.channels_last,
        )

    # 应用装饰器，限制只在原生设备类型上运行该测试方法
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试不同数据类型和设备上的最大池化操作
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @gcIfJetson
    @onlyCPU
    @dtypes(torch.half, torch.bfloat16)
    def test_max_pool_bfloat16_half(self, device, dtype):
        # 定义一个辅助函数，用于执行具体的最大池化测试
        def helper(shape, kernel_size, stride, memory_format, dtype):
            # 生成指定形状和数据类型的随机输入张量，并将其移到指定设备上，设置为需要梯度
            input = torch.randn(shape, dtype=dtype, device=device)
            input = input.to(memory_format=memory_format).requires_grad_()
            
            # 根据输入张量的维度选择合适的最大池化层，设置返回池化结果及索引
            if len(shape) == 4:
                pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(
                    device
                )
            else:
                pool = torch.nn.MaxPool3d(kernel_size, stride, return_indices=True).to(
                    device
                )
    
            # 创建一个与输入张量相同数据的副本，并转换为float类型，同时需要梯度
            input2 = input.detach().clone().float().requires_grad_(True)
    
            # 对两个不同数据类型的输入分别执行最大池化操作，并进行反向传播
            out, ind = pool(input)
            out.sum().backward()
            out2, ind2 = pool(input2)
            out2.sum().backward()
    
            # 断言池化后的输出张量是连续的，并且数据类型与预期一致
            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(out.dtype, dtype)
            
            # 断言输入张量的梯度数据类型与预期一致
            self.assertEqual(input.grad.dtype, dtype)
            
            # 断言两次不同数据类型输入的池化结果一致，并且池化索引相同
            self.assertEqual(out, out2.to(dtype=dtype))
            self.assertEqual(ind, ind2)
            
            # 断言两次不同数据类型输入的梯度一致
            self.assertEqual(input.grad, input2.grad.to(dtype=dtype))
    
        # 对多种输入形状和存储格式下的最大池化进行测试
        helper((4, 30, 8, 8), 7, 1, torch.contiguous_format, dtype)
        helper((4, 65, 8, 8), 7, 1, torch.channels_last, dtype)
        helper((1, 19, 20, 10), 8, 2, torch.contiguous_format, dtype)
        helper((1, 19, 20, 10), 8, 2, torch.channels_last, dtype)
        helper((4, 30, 8, 8), 7, 1, torch.contiguous_format, dtype)
        helper((4, 65, 8, 8), 7, 1, torch.channels_last, dtype)
        helper((1, 19, 10, 10, 10), 8, 2, torch.contiguous_format, dtype)
        helper((1, 19, 10, 9, 14), 8, 2, torch.channels_last_3d, dtype)
        helper((4, 10, 3, 8, 8), 3, 1, torch.contiguous_format, dtype)
        helper((4, 10, 8, 8, 8), 7, 1, torch.channels_last_3d, dtype)
    
    # 仅在CUDA环境下执行的装饰器，用于指定只在GPU设备上执行的测试
    @onlyCUDA
    @gcIfJetson
    # 定义测试函数 test_max_pool2d_indices，接受一个设备参数
    def test_max_pool2d_indices(self, device):
        
        # 嵌套函数 helper，用于测试不同参数的 MaxPool2d 操作
        def helper(n, c, h, w, ks):
            # 根据是否提供 batch size n 创建随机张量 x，设备为 CUDA，数据类型为 float，需要梯度计算
            if n is None:
                x = torch.randn(
                    c, h, w, device="cuda", dtype=torch.float, requires_grad=True
                )
            else:
                x = torch.randn(
                    n, c, h, w, device="cuda", dtype=torch.float, requires_grad=True
                )
            
            # 创建 x 的一个在 CPU 上的副本 ref_x，用于比较
            ref_x = x.detach().clone().cpu().requires_grad_()

            # 创建 MaxPool2d 层，设置核大小为 ks，并返回 pooling 的索引
            pool = torch.nn.MaxPool2d(kernel_size=ks, return_indices=True)

            # 对 x 应用 MaxPool2d，并获取输出 y 和索引 idx
            y, idx = pool(x)
            # 对 ref_x 应用相同的 MaxPool2d 操作，获取 ref_y 和 ref_idx
            ref_y, ref_idx = pool(ref_x)

            # 对 y 的所有元素求和并进行反向传播
            y.sum().backward()
            # 对 ref_y 的所有元素求和并进行反向传播
            ref_y.sum().backward()

            # 断言 y 与 ref_y 相等
            self.assertEqual(y, ref_y)
            # 断言 idx 与 ref_idx 相等（assertEqual 隐式比较张量的形状）
            self.assertEqual(
                idx, ref_idx
            )  # assertEqual implicitly compares shape for tensors
            # 断言 x 的梯度与 ref_x 的梯度相等
            self.assertEqual(x.grad, ref_x.grad)

        # 使用 helper 函数进行两组测试
        helper(2, 8, 4, 4, ks=2)
        helper(None, 3, 50, 50, ks=5)

    # 标记仅在 CPU 上运行的测试，并且数据类型为 torch.half 或 torch.bfloat16
    @onlyCPU
    @dtypes(torch.half, torch.bfloat16)
    # 定义测试函数 test_avg_pool2d_reduced_floating，接受设备和数据类型参数
    def test_avg_pool2d_reduced_floating(self, device, dtype):
        
        # 嵌套函数 helper，用于测试不同参数的 AvgPool2d 操作
        def helper(n, c, h, w, kernel_size, stride, memory_format):
            # 创建随机输入张量 input，数据类型为 float32，设备为指定的 device，并转换为指定的 dtype
            input = torch.randn(n, c, h, w, dtype=torch.float32, device=device).to(
                dtype=dtype
            )
            # 转换 input 张量的存储格式为 memory_format，并要求计算梯度
            input = input.to(memory_format=memory_format).requires_grad_()
            # 创建 AvgPool2d 层，设置核大小为 kernel_size 和步幅为 stride，并将其移至指定的设备
            pool = torch.nn.AvgPool2d(kernel_size, stride).to(device)

            # 对 input 创建一个在 CPU 上的浮点副本 input2，并要求计算梯度
            input2 = input.detach().clone().float().requires_grad_(True)

            # 对 input 应用 AvgPool2d，并获取输出 out，然后对输出进行求和并反向传播
            out = pool(input)
            out.sum().backward()
            
            # 对 input2 应用相同的 AvgPool2d 操作，并获取输出 out2，然后对输出进行求和并反向传播
            out2 = pool(input2)
            out2.sum().backward()

            # 断言 out 在指定的存储格式 memory_format 下是连续的
            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            # 断言 out 的数据类型是指定的 dtype
            self.assertEqual(out.dtype, dtype)
            # 断言 input 的梯度的数据类型是指定的 dtype
            self.assertEqual(input.grad.dtype, dtype)
            # 断言 out 与经 dtype 转换后的 out2 相等
            self.assertEqual(out, out2.to(dtype=dtype))
            # 断言 input 的梯度与经 dtype 转换后的 input2 的梯度相等
            self.assertEqual(input.grad, input2.grad.to(dtype=dtype))

        # 使用 helper 函数进行多组测试，涵盖不同的参数组合和存储格式
        helper(4, 30, 8, 8, 7, 1, torch.contiguous_format)
        helper(4, 65, 8, 8, 7, 1, torch.channels_last)
        helper(1, 19, 20, 10, 8, 2, torch.contiguous_format)
        helper(1, 19, 20, 10, 8, 2, torch.channels_last)

    # 标记数据类型为 torch.float 或 torch.double 的测试
    # 定义测试方法，测试通道自适应最大池化（channels_last_3d 与 3d 分别用不同的 pool）
    def test_adaptive_pooling_max_nhwc(self, device, dtype):
        # 定义辅助函数，用于测试不同输入大小和输出平面大小的情况
        def helper(input_size, output_plane_size, contig):
            # 确定输出平面的维度数
            n_plane_dims = len(output_plane_size)
            # 根据输出平面维度数选择合适的自适应最大池化层类
            mod = (
                torch.nn.AdaptiveMaxPool2d
                if n_plane_dims == 2
                else torch.nn.AdaptiveMaxPool3d
            )
            # 根据输出平面维度数选择合适的内存格式函数
            channels_last = (
                torch.channels_last if n_plane_dims == 2 else torch.channels_last_3d
            )
            # 计算输出尺寸
            output_size = input_size[:2] + output_plane_size
            # 生成随机输入张量
            input = torch.randint(1, 10, input_size, device=device, dtype=dtype)
            # 使用通道最后内存格式进行内存连续化
            input = input.contiguous(memory_format=channels_last)
            # 生成随机梯度张量
            grad = torch.randint(1, 10, output_size, device=device, dtype=dtype)
            # 使用通道最后内存格式进行内存连续化
            grad = grad.contiguous(memory_format=channels_last)
            # 如果不连续，按2的步长切片输入和梯度张量
            if not contig:
                input = input[:, ::2]
                grad = grad[:, ::2]
            # 设置输入张量需要梯度计算
            input.requires_grad_(True)
            # 创建自适应池化层对象
            pool = mod(output_plane_size, return_indices=True).to(device)

            # 复制并连续化参考输入张量
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 复制并连续化参考梯度张量
            ref_grad = grad.detach().clone().contiguous()
            # 创建自适应池化层对象（作为参考）
            ref_pool = mod(output_plane_size, return_indices=True).to(device)

            # 执行池化操作并获取池化输出和指数
            out, ind = pool(input)
            # 对池化输出进行反向传播
            out.backward(grad)
            # 执行参考池化操作并获取参考池化输出和指数
            ref_out, ref_ind = ref_pool(ref_input)
            # 对参考池化输出进行反向传播
            ref_out.backward(ref_grad)

            # 对于 channels_last_3d 情况，不返回 channels_last_3d 输出
            if n_plane_dims == 2:
                # 断言池化输出在通道最后内存格式下是连续的
                self.assertTrue(out.is_contiguous(memory_format=channels_last))
                # 断言池化指数在通道最后内存格式下是连续的
                self.assertTrue(ind.is_contiguous(memory_format=channels_last))
            # 断言参考池化输出是连续的
            self.assertTrue(ref_out.is_contiguous())
            # 断言参考池化指数是连续的
            self.assertTrue(ref_ind.is_contiguous())
            # 断言池化输出与参考池化输出相等
            self.assertEqual(out, ref_out)
            # 断言池化指数与参考池化指数相等
            self.assertEqual(ind, ref_ind)
            # 断言输入梯度与参考输入梯度相等
            self.assertEqual(input.grad, ref_input.grad)

        # 遍历连续与不连续内存格式的情况
        for contig in [True, False]:
            # 测试不同输入大小和输出平面大小的情况
            helper((4, 8, 10, 10), (7, 7), contig)
            helper((4, 8, 9, 14), (5, 8), contig)
            helper((4, 8, 11, 11), (1, 1), contig)
            helper((2, 1, 3, 3), (1, 1), contig)
            helper((4, 8, 10, 10, 10), (7, 7, 7), contig)
            helper((4, 8, 11, 11, 11), (1, 1, 1), contig)
            helper((2, 1, 3, 3, 3), (1, 1, 1), contig)

    # 使用 torch.float 和 torch.double 作为数据类型进行测试
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于测试 NHWC 格式下的最大池化操作
    def test_pooling_max_nhwc(self, device, dtype):
        # 定义一个内部辅助函数，用于执行具体的测试操作
        def helper(n, c, h, w, kernel_size, stride, padding, dilation, contig, device):
            # 计算输出特征图的高度
            output_height = math.floor(
                (h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
                / stride[0]
                + 1
            )
            # 计算输出特征图的宽度
            output_width = math.floor(
                (w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
                / stride[1]
                + 1
            )

            # 创建一个随机填充的输入张量，形状为 (n, c, h, w)，在指定设备上进行处理
            input = torch.randint(1, 10, (n, c, h, w), device=device, dtype=dtype)
            # 将输入张量转换为连续内存格式，使用通道最后的格式
            input = input.contiguous(memory_format=torch.channels_last)
            
            # 创建一个随机填充的梯度张量，形状为 (n, c, output_height, output_width)，在指定设备上进行处理
            grad = torch.randint(
                1, 10, (n, c, output_height, output_width), device=device, dtype=dtype
            )
            # 将梯度张量转换为连续内存格式，使用通道最后的格式
            grad = grad.contiguous(memory_format=torch.channels_last)
            
            # 如果不是连续内存格式，按照一定规则对输入和梯度张量进行切片
            if not contig:
                input = input[:, ::2, :, :]
                grad = grad[:, ::2, :, :]
            
            # 设置输入张量需要计算梯度
            input.requires_grad_(True)
            
            # 创建一个最大池化层对象，指定池化核大小、步长、填充、扩展等参数
            pool = torch.nn.MaxPool2d(
                kernel_size,
                stride,
                padding,
                dilation,
                return_indices=True,
                ceil_mode=False,
            )

            # 创建一个参考输入张量的副本，需要计算梯度，并确保连续内存格式
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            # 创建一个参考梯度张量的副本，并确保连续内存格式
            ref_grad = grad.detach().clone().contiguous()
            # 创建一个在指定设备上的最大池化层对象，用于参考
            ref_pool = torch.nn.MaxPool2d(
                kernel_size,
                stride,
                padding,
                dilation,
                return_indices=True,
                ceil_mode=False,
            ).to(device)

            # 执行池化操作，得到池化后的输出和对应的索引
            out, ind = pool(input)
            # 对输出执行反向传播，使用给定的梯度
            out.backward(grad)
            # 对参考输入执行池化操作，得到参考输出和对应的索引
            ref_out, ref_ind = ref_pool(ref_input)
            # 对参考输出执行反向传播，使用参考梯度
            ref_out.backward(ref_grad)

            # 断言输出张量为连续内存格式，使用通道最后的格式
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            # 断言参考输出张量为连续内存格式
            self.assertTrue(ref_out.is_contiguous())
            # 断言输出索引张量为连续内存格式，使用通道最后的格式
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            # 断言参考输出索引张量为连续内存格式
            self.assertTrue(ref_ind.is_contiguous())
            # 断言输入张量的梯度与参考输入张量的梯度相等
            self.assertEqual(input.grad, ref_input.grad)

        # 遍历不同的内存格式情况，执行具体的测试操作
        for contig in [True, False]:
            helper(4, 8, 10, 10, (2, 2), (1, 1), (1, 1), (2, 2), contig, device)
            helper(4, 8, 9, 14, (2, 2), (1, 1), (1, 1), (2, 2), contig, device)
            helper(4, 8, 11, 11, (4, 4), (2, 2), (2, 2), (2, 2), contig, device)

    @onlyCUDA
    def test_pool3d_size_one_feature_dim(self, device):
        # Tests crazy strides for feature dim of size 1
        # 创建一个形状为 (7, 1, 5, 3, 2) 的张量 x，其中的数据来自于标准正态分布，存储在指定设备上
        x = torch.randn(7, 1, 5, 3, 2, device=device)
        # 定义一个奇怪的步长列表，用于重塑张量 x 的步长
        strange_strides = [30, 1234, 6, 2, 1]
        # 使用奇怪的步长重新生成张量 y，并存储结果
        y = x.as_strided(x.size(), strange_strides)
        # 将张量 x 移回 CPU，并使用奇怪的步长重新生成张量 x，并存储结果
        x = x.cpu().as_strided(x.size(), strange_strides)

        # 定义测试用例字典，每个测试使用不同的池化函数对 y 和 x 执行操作
        to_test = {
            "max_pool3d": lambda t: F.max_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
            "avg_pool3d": lambda t: F.avg_pool3d(t, (5, 1, 1), stride=(5, 1, 1)),
        }

        # 遍历测试用例字典，对每个测试执行相应的函数操作，并进行断言验证
        for test, fn in to_test.items():
            # 应保证不会崩溃
            out_y = fn(y)
            out_x = fn(x)
            self.assertEqual(out_y, out_x.to(device), msg=test)

    @onlyCUDA
    @largeTensorTest("18GB")
    @largeTensorTest("180GB", "cpu")
    def test_pool3d_large_size_int64(self, device):
        # See https://github.com/pytorch/pytorch/issues/52822
        # 创建一个形状为 (70, 32, 100, 100, 100)、数据类型为 torch.half 的张量 x，存储在指定设备上，需要梯度
        x = torch.randn(
            70, 32, 100, 100, 100, dtype=torch.half, device=device, requires_grad=True
        )
        # 对张量 x 进行 3D 最大池化，使用核大小 5，并存储结果
        y = torch.nn.functional.max_pool3d(x, 5)
        # 创建一个形状与 y 相同的张量 g，数据类型为 torch.half，存储在相同设备上
        g = torch.randn_like(y, dtype=torch.half)
        torch.cuda.synchronize()
        # 对 y 执行反向传播，使用梯度 g，并同步 CUDA
        y.backward(g)
        torch.cuda.synchronize()

        # 创建一个与 x 分离且移回 CPU 的张量 ref_x，数据类型为 float
        ref_x = x.detach().cpu().float()  # max_pool3d_cpu is not implemented for half
        ref_x.requires_grad = True
        # 创建一个与 g 相同形状的张量 ref_g，数据类型为 float
        ref_g = g.cpu().float()
        # 对 ref_x 进行 3D 最大池化，使用核大小 5，并存储结果
        ref_y = torch.nn.functional.max_pool3d(ref_x, 5)
        # 对 ref_y 执行反向传播，使用梯度 ref_g
        ref_y.backward(ref_g)

        # 断言 y 与 ref_y 相等，允许数据类型不精确匹配
        self.assertEqual(y, ref_y, exact_dtype=False)
        # 断言 x 的梯度与 ref_x 的梯度相等，允许数据类型不精确匹配
        self.assertEqual(x.grad, ref_x.grad, exact_dtype=False)

    @onlyCUDA
    def test_AvgPool3d_backward_after_cat_dim1_device(self, device):
        # x 必须具有批量大小 1，以便测试连续性检查
        # 创建一个形状为 (1, 3, 4, 4, 4) 的张量 x，存储在指定设备上，需要梯度
        x = torch.randn(1, 3, 4, 4, 4, device=device, requires_grad=True)
        # 对张量 x 进行 3D 平均池化，使用核大小 3、填充大小 1、步长 2，并存储结果
        y = F.avg_pool3d(x, kernel_size=3, padding=1, stride=2)

        # 创建一个与 y 相同形状的张量 grad，存储在相同设备上
        grad = torch.randn(y.size(), device=device)
        # 增加 grad 在维度 0 上的步长，仍然保证张量是连续的，因为 size[0] 为 1
        stride = list(grad.stride())
        stride[0] = stride[0] * 2
        grad.set_(grad.storage(), 0, grad.size(), stride)
        assert grad.is_contiguous()

        # 对 y 执行反向传播，使用梯度 grad
        y.backward(grad)

    def _test_maxpool_indices(
        self, num_dim, adaptive=False, device="cpu", dtype=torch.float
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_MaxPool1d_indices(self, device, dtype):
        # 调用 _test_maxpool_indices 方法，传入维度 1、指定设备和数据类型
        self._test_maxpool_indices(1, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_MaxPool2d_indices(self, device, dtype):
        # 调用 _test_maxpool_indices 方法，传入维度 2、指定设备和数据类型
        self._test_maxpool_indices(2, device=device, dtype=dtype)

    @skipIfMps
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_MaxPool3d_indices(self, device, dtype):
        # 调用 _test_maxpool_indices 方法，传入维度 3、指定设备和数据类型
        self._test_maxpool_indices(3, device=device, dtype=dtype)

    @skipIfMps
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_AdaptiveMaxPool1d_indices(self, device, dtype):
        # 根据指定条件的数据类型和CUDA状态，测试AdaptiveMaxPool1d操作的索引生成
        self._test_maxpool_indices(1, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_AdaptiveMaxPool2d_indices(self, device, dtype):
        # 根据指定条件的数据类型和CUDA状态，测试AdaptiveMaxPool2d操作的索引生成
        self._test_maxpool_indices(2, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_AdaptiveMaxPool3d_indices(self, device, dtype):
        # 根据指定条件的数据类型和CUDA状态，测试AdaptiveMaxPool3d操作的索引生成
        self._test_maxpool_indices(3, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_maxpool_indices_no_batch_dim(self, device, dtype):
        """检查没有批次维度的索引是否与单个批次保持一致。"""
        max_pool_cases = [
            (
                nn.MaxPool1d(3, return_indices=True),
                torch.randn(3, 5, device=device, dtype=dtype),
            ),
            (
                nn.MaxPool2d(3, return_indices=True),
                torch.randn(3, 5, 6, device=device, dtype=dtype),
            ),
            (
                nn.MaxPool3d(3, return_indices=True),
                torch.randn(3, 5, 6, 7, device=device, dtype=dtype),
            ),
            (
                nn.AdaptiveMaxPool1d(3, return_indices=True),
                torch.randn(3, 5, device=device, dtype=dtype),
            ),
            (
                nn.AdaptiveMaxPool2d(3, return_indices=True),
                torch.randn(3, 5, 6, device=device, dtype=dtype),
            ),
            (
                nn.AdaptiveMaxPool3d(3, return_indices=True),
                torch.randn(3, 5, 6, 7, device=device, dtype=dtype),
            ),
        ]

        for module, input in max_pool_cases:
            # 对于每个最大池化操作和输入数据，获取没有批次维度的索引和添加单批次后的索引
            _, indices_no_batch = module(input)
            _, indicies_single_batch = module(input.unsqueeze(0))
            # 断言没有批次维度的索引和添加单批次后的索引是否一致
            self.assertEqual(indices_no_batch, indicies_single_batch.squeeze(0))

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @onlyNativeDeviceTypes  # TODO: XLA 上失败
    @gcIfJetson
    # 定义一个测试函数，用于测试最大池化操作对包含NaN和-inf的张量的处理
    def test_max_pool_nan_inf(self, device, dtype):
        # 循环遍历不同的参数组合进行测试，包括是否自适应和张量维度
        for adaptive in ["", "adaptive_"]:
            for num_dim in [1, 2, 3]:
                # 根据自适应性和张量维度构建函数名
                fn_name = f"{adaptive}max_pool{num_dim}d"
                # 根据函数名获取对应的函数
                fn = getattr(F, fn_name)

                # 创建一个全部为NaN的张量x，指定设备和数据类型，并设置需要梯度计算
                x = torch.full(
                    [1, 1] + num_dim * [3],
                    nan,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                # 对x进行最大池化操作，并进行反向传播，计算梯度
                res = fn(x, 1 if adaptive else 3)
                res.backward(torch.randn_like(res))
                # 断言结果是否为NaN
                self.assertTrue(math.isnan(res.item()))
                # 关闭x的梯度计算
                x.requires_grad_(False)
                # 重新应用最大池化操作，并再次断言结果是否为NaN
                res = fn(x, 1 if adaptive else 3)
                self.assertTrue(math.isnan(res.item()))

                # 创建一个全部为-inf的张量x2，指定设备和数据类型，并设置需要梯度计算
                x2 = torch.full(
                    [1, 1] + num_dim * [3],
                    -inf,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                # 对x2进行最大池化操作，并进行反向传播，计算梯度
                res2 = fn(x2, 1 if adaptive else 3)
                res2.backward(torch.randn_like(res2))
                # 断言结果是否为-inf
                self.assertTrue(math.isinf(res2.item()))
                # 关闭x2的梯度计算
                x2.requires_grad_(False)
                # 重新应用最大池化操作，并再次断言结果是否为-inf
                res2 = fn(x2, 1 if adaptive else 3)
                self.assertTrue(math.isinf(res2.item()))

    # 标记当前测试为预期失败，因为会抛出RuntimeError: Unrecognized tensor type ID: Meta异常
    @expectedFailureMeta
    # 只对原生设备类型进行测试
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试 fractional_max_pool2d 函数的行为
    def test_fractional_max_pool2d(self, device):
        # 设置默认数据类型为双精度浮点数
        with set_default_dtype(torch.double):
            # 生成一个形状为 (1, 2, 7, 7) 的张量 x，要求梯度计算，放置在指定设备上
            x = torch.randn(1, 2, 7, 7, requires_grad=True, device=device)
            # 生成一个与 x 相同设备的新张量 samples，形状为 (1, 2, 2)，均匀分布赋值
            samples = x.new(1, 2, 2).uniform_()

            # 定义一个内部函数 func，调用 F.fractional_max_pool2d 进行池化操作
            def func(x):
                return F.fractional_max_pool2d(
                    x, (2, 2), output_size=(3, 3), _random_samples=samples
                )

            # 断言 func(x) 的输出形状为 (1, 2, 3, 3)
            self.assertEqual(func(x).shape, (1, 2, 3, 3))
            # 使用 gradcheck 函数验证梯度计算的正确性
            gradcheck(func, [x])
            # 使用 gradgradcheck 函数验证二阶梯度计算的正确性
            gradgradcheck(func, [x])

            # 重新生成一个形状为 (2, 7, 7) 的张量 x，要求梯度计算，放置在指定设备上
            x = torch.randn(2, 7, 7, requires_grad=True, device=device)
            # 断言 func(x) 的输出形状为 (2, 3, 3)
            self.assertEqual(func(x).shape, (2, 3, 3))
            
            # 如果设备类型不是 "cuda"
            if self.device_type != "cuda":
                # 在 CUDA 上进行 gradcheck 可能会引发 RuntimeError，这里通过注释说明了这个问题的来源和链接
                # 参考链接：https://github.com/pytorch/pytorch/issues/52427
                # 会引发 RuntimeError: TensorAccessor expected 4 dims but tensor has 3 on CUDA in gradcheck
                gradcheck(func, [x])
                gradgradcheck(func, [x])

            # 遍历不同的 kernel_size 进行测试
            for kernel_size in [(), (1,)]:
                # 使用 assertRaisesRegex 断言捕获 RuntimeError，并验证错误信息中包含特定文本
                with self.assertRaisesRegex(RuntimeError, "kernel_size must either"):
                    # 使用不正确的 kernel_size 调用 fractional_max_pool2d 函数
                    F.fractional_max_pool2d(
                        x,
                        kernel_size=kernel_size,
                        output_size=(3, 3),
                        _random_samples=samples,
                    )

            # 针对不同的 output_size 和错误消息进行测试
            err_large_msg = "too large relative to input "
            err_out_size_msg = "output_size must either"
            for output_size, msg in [
                ((9, 3), err_large_msg + "height"),
                ((3, 9), err_large_msg + "width"),
                ((3,), err_out_size_msg),
                ((), err_out_size_msg),
            ]:
                # 使用 assertRaisesRegex 断言捕获 RuntimeError，并验证错误信息中包含特定文本
                with self.assertRaisesRegex(RuntimeError, msg):
                    # 使用不正确的 output_size 调用 fractional_max_pool2d 函数
                    F.fractional_max_pool2d(
                        x, (2, 2), output_size=output_size, _random_samples=samples
                    )
    # 定义一个测试函数，用于测试 torch.nn.functional 中的 fractional_max_pool3d 函数
    def test_fractional_max_pool3d(self, device):
        # 使用双精度浮点数作为默认数据类型
        with set_default_dtype(torch.double):
            # 创建一个形状为 (1, 2, 7, 7, 7) 的张量 x，其值服从标准正态分布，并且可以计算梯度，位于指定设备上
            x = torch.randn(1, 2, 7, 7, 7, requires_grad=True, device=device)
            # 创建一个与 x 具有相同数据类型的随机均匀分布的张量 samples，形状为 (1, 2, 3)
            samples = x.new(1, 2, 3).uniform_()

            # 定义一个局部函数 func，它调用 torch.nn.functional 中的 fractional_max_pool3d 函数
            def func(x):
                return F.fractional_max_pool3d(
                    x, (2, 2, 2), output_size=(3, 3, 3), _random_samples=samples
                )

            # 断言 func(x) 的输出形状为 (1, 2, 3, 3, 3)
            self.assertEqual(func(x).shape, (1, 2, 3, 3, 3))
            # 对 func 进行梯度检查
            gradcheck(func, [x])
            # 对 func 进行二阶梯度检查
            gradgradcheck(func, [x])

            # 创建一个形状为 (2, 7, 7, 7) 的张量 x，其值服从标准正态分布，并且可以计算梯度，位于指定设备上
            x = torch.randn(2, 7, 7, 7, requires_grad=True, device=device)
            # 断言 func(x) 的输出形状为 (2, 3, 3, 3)
            self.assertEqual(func(x).shape, (2, 3, 3, 3))
            # 对 func 进行梯度检查
            gradcheck(func, [x])
            # 对 func 进行二阶梯度检查
            gradgradcheck(func, [x])

            # 遍历不同的 kernel_size 参数进行测试
            for kernel_size in [(), (1,), (1, 1)]:
                with self.assertRaisesRegex(RuntimeError, "kernel_size must either"):
                    # 错误的 kernel_size 参数
                    F.fractional_max_pool3d(
                        x,
                        kernel_size=kernel_size,
                        output_size=(3, 3, 3),
                        _random_samples=samples,
                    )

            # 错误的 output_size 参数及其对应的错误消息
            err_large_msg = "too large relative to input "
            err_out_size_msg = "output_size must either"
            for output_size, msg in [
                ((9, 3, 3), err_large_msg + "time"),
                ((3, 9, 3), err_large_msg + "height"),
                ((3, 3, 9), err_large_msg + "width"),
                ((3, 3), err_out_size_msg),
                ((3,), err_out_size_msg),
                ((), err_out_size_msg),
            ]:
                with self.assertRaisesRegex(RuntimeError, msg):
                    # 错误的 output_size 参数
                    F.fractional_max_pool3d(
                        x, (2, 2, 2), output_size=output_size, _random_samples=samples
                    )

    # 标记为仅适用于原生设备类型的测试函数（可能与 XLA 有关）
    @onlyNativeDeviceTypes  # TODO: RuntimeError message different on XLA
    # 测试函数，用于检测 fractional_max_pool 函数对 NaN 和 Inf 的处理
    def test_fractional_max_pool_nan_inf(self, device, dtype):
        for num_dim in [2, 3]:
            fn_name = f"FractionalMaxPool{num_dim}d"
            # 获取 torch.nn 中对应维度的 FractionalMaxPool 类实例 fn
            fn = getattr(nn, fn_name)(kernel_size=2, output_size=1)
            # 创建一个形状为 [1, 1, 3, ..., 3] 的张量 x，其值为 NaN，位于指定设备上，指定数据类型
            x = torch.full(
                [1, 1] + num_dim * [3],
                nan,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            # 对 fn(x) 进行前向传播
            res = fn(x)
            # 对 fn(x) 的输出进行反向传播
            res.backward(torch.randn_like(res))
            # 断言 res 的值为 NaN
            self.assertTrue(math.isnan(res.item()))

            # 创建一个形状为 [1, 1, 3, ..., 3] 的张量 x2，其值为 -Inf，位于指定设备上，指定数据类型
            x2 = torch.full(
                [1, 1] + num_dim * [3],
                -inf,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            # 对 fn(x2) 进行前向传播
            res2 = fn(x2)
            # 对 fn(x2) 的输出进行反向传播
            res2.backward(torch.randn_like(res2))
            # 断言 res2 的值为 -Inf
            self.assertTrue(math.isinf(res2.item()))
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)


# 根据当前测试环境选择数据类型，如果在CUDA下使用半精度和bfloat16类型数据，跳过MPS（Memory Pooling System）测试，使用单精度float类型数据进行测试



    def test_pooling_zero_stride(self, device):


# 定义测试函数，用于测试池化操作中 stride 为零的情况



        for op in ("max", "avg"):


# 针对池化操作类型进行循环测试，包括最大池化和平均池化



            for num_dim in [1, 2, 3]:


# 针对不同维度（1维、2维、3维）进行循环测试



                fn_name = f"{op}_pool{num_dim}d"
                fn = getattr(F, fn_name)
                x = torch.ones([1, 2] + num_dim * [4], device=device, dtype=torch.float)
                self.assertRaisesRegex(
                    RuntimeError,
                    r"stride should not be zero|stride must be greater than zero",
                    lambda: fn(x, kernel_size=2, stride=0),
                )


# 构建池化函数的名称，并获取对应的函数对象，创建输入张量 x，如果使用零步幅调用池化函数 fn(x, kernel_size=2, stride=0)，则预期会抛出 RuntimeError 异常，异常信息为 "stride should not be zero" 或 "stride must be greater than zero"



                fn_module_name = f"{op.title()}Pool{num_dim}d"
                fn_module = getattr(nn, fn_module_name)(kernel_size=2, stride=0)
                self.assertRaisesRegex(
                    RuntimeError,
                    r"stride should not be zero|stride must be greater than zero",
                    lambda: fn_module(x),
                )


# 构建池化模块的名称，并获取对应的模块对象，创建输入张量 x，如果使用零步幅调用池化模块 fn_module(x)，则预期会抛出 RuntimeError 异常，异常信息为 "stride should not be zero" 或 "stride must be greater than zero"



    def test_pool_large_size(self, device, dtype):


# 定义测试函数，用于测试输入张量尺寸过大的情况



        for op in ("max", "avg"):


# 针对池化操作类型进行循环测试，包括最大池化和平均池化



            for num_dim in [1, 2, 3]:


# 针对不同维度（1维、2维、3维）进行循环测试



                fn_name = f"{op}_pool{num_dim}d"
                fn = getattr(F, fn_name)
                # 16777217 is the smallest integer not expressible in float32
                x = torch.ones(
                    [1, 1, 16777217] + (num_dim - 1) * [1], device=device, dtype=dtype
                )
                res = fn(x, 1, stride=1, padding=0)
                # check if the output shape was still computed correctly
                self.assertEqual(x.shape[2], res.shape[2])


# 构建池化函数的名称，并获取对应的函数对象，创建一个尺寸超大的输入张量 x（[1, 1, 16777217, 1, ...]），使用指定参数调用池化函数 fn(x, 1, stride=1, padding=0)，并检查输出形状是否正确计算



    @onlyCUDA
    @largeTensorTest("6GB")
    def test_pooling_large(self, device):


# 标记该测试仅在CUDA环境下运行，并且是一个测试大张量情况下的池化操作



        def helper(pool):
            inp = torch.randn(
                2**7 + 10, 2**8, 2**8, 2**8, dtype=torch.half, device="cuda"
            )
            self.assertTrue(inp.numel() > 2**31 - 1)
            out = pool(inp)
            torch.cuda.synchronize()  # asserts test finishes normally without raising errors


# 定义一个辅助函数 helper，接收一个池化操作作为参数，创建一个尺寸非常大的随机张量 inp（[2**7 + 10, 2**8, 2**8, 2**8]），确保张量的元素数量超过2**31 - 1，然后应用池化操作 pool 并同步CUDA流，以确保测试正常完成且不引发错误



        helper(nn.MaxPool2d(4, 4))
        helper(nn.AvgPool2d(4, 4))
        helper(nn.FractionalMaxPool2d(4, 4))
        helper(nn.AdaptiveMaxPool2d((2**6, 2**6)))
        helper(nn.AdaptiveAvgPool2d((2**6, 2**6)))


# 对不同的池化操作（最大池化、平均池化、分数最大池化、自适应最大池化、自适应平均池化）分别调用辅助函数 helper 进行测试
    def test_pool_invalid_size(self, device, dtype):
        # 针对不同的操作和维度数进行测试
        for op in ("max", "avg"):
            for num_dim in [1, 2, 3]:
                # 构造函数名，例如 "max_pool1d"
                fn_name = f"{op}_pool{num_dim}d"
                if op == "max":
                    # 如果是最大池化，添加支持空张量的新实现
                    # TODO(Heitor) 一旦带索引的代码更新，应该移除这段代码
                    fn_name += "_with_indices"
                # 获取函数对象
                fn = getattr(F, fn_name)
                # 创建一个全为1的张量作为输入，维度根据 num_dim 动态生成
                x = torch.ones([1, 1] + num_dim * [4], device=device, dtype=dtype)
                # 使用断言检测是否抛出预期的运行时异常
                with self.assertRaisesRegex(RuntimeError, r"too small|smaller than"):
                    try:
                        # 尝试调用池化函数
                        res = fn(x, 3, stride=2, padding=0, dilation=2)
                    except TypeError:
                        # 捕获可能不支持 dilation 的实现
                        res = fn(x, 6, stride=2, padding=0)

    @onlyCUDA
    def test_pooling_bfloat16(self, device):
        # 测试 bfloat16 数据类型的池化操作
        _test_bfloat16_ops(
            self,
            torch.nn.AvgPool1d(3, stride=2),
            device,
            inp_dims=(8, 4, 16),
            prec=0.05,
        )
        _test_bfloat16_ops(
            self,
            torch.nn.AvgPool2d(3, stride=2),
            device,
            inp_dims=(8, 4, 16, 16),
            prec=0.05,
        )
        _test_bfloat16_ops(
            self,
            torch.nn.AvgPool3d(3, stride=2),
            device,
            inp_dims=(8, 4, 16, 16, 16),
            prec=0.05,
        )
        _test_bfloat16_ops(
            self, torch.nn.AdaptiveAvgPool1d(3), device, inp_dims=(8, 4, 16), prec=0.05
        )
        _test_bfloat16_ops(
            self,
            torch.nn.AdaptiveAvgPool2d((3, 5)),
            device,
            inp_dims=(8, 4, 16, 16),
            prec=0.05,
        )
        _test_bfloat16_ops(
            self,
            torch.nn.AdaptiveAvgPool3d((3, 5, 7)),
            device,
            inp_dims=(8, 4, 16, 16, 16),
            prec=0.05,
        )

    def test_maxpool3d_non_square_backward(self, device):
        # 测试 maxpool3d 非方形输入时的反向传播
        # 以前的 CUDA 实现在计算内核启动网格大小时，最后两个维度会互换，
        # 因此长维度上的尾部会被忽略。在这里测试每个位置是否得到了梯度。
        for dim in (2, 3, 4):
            # 创建具有不同维度形状的随机张量
            shape = tuple(32 if i != dim else 256 for i in range(4))
            x = torch.randn(shape, device=device, requires_grad=True)
            # 对 maxpool3d 的输出进行求和，并执行反向传播
            F.max_pool3d(x, kernel_size=(1, 1, 1)).sum().backward()
            # 断言梯度是否与全1张量相同
            self.assertEqual(x.grad, torch.ones_like(x.grad))

    @slowTest
    # 定义一个测试方法，用于测试自适应池化在奇数尺寸上的行为
    def test_adaptive_pool_odd_size(self, device):
        # 设置输入图像和输出尺寸的高度和宽度
        Ih, Iw, Oh, Ow = 5873, 3693, 3527, 2219
        # 生成随机整数张量作为输入图像，尺寸为 (11, Ih, Iw)，数据类型为浮点型
        imgs = torch.randint(low=0, high=256, size=(11, Ih, Iw), dtype=torch.float)
        # 对输入图像进行二维自适应平均池化到指定的输出尺寸 (Oh, Ow)
        imgs_ = F.adaptive_avg_pool2d(imgs, (Oh, Ow))
        # 对输入图像进行二维自适应最大池化到指定的输出尺寸 (Oh, Ow)

        imgs_ = F.adaptive_max_pool2d(imgs, (Oh, Ow))

        # 设置输入和输出的深度、高度和宽度
        Id, Ih, Iw, Od, Oh, Ow = 3, 5873, 3693, 3, 3527, 2219
        # 生成随机整数张量作为输入图像，尺寸为 (3, Id, Ih, Iw)，数据类型为浮点型
        imgs = torch.randint(low=0, high=256, size=(3, Id, Ih, Iw), dtype=torch.float)
        # 对输入图像进行三维自适应平均池化到指定的输出尺寸 (Od, Oh, Ow)
        imgs_ = F.adaptive_avg_pool3d(imgs, (Od, Oh, Ow))
        # 对输入图像进行三维自适应最大池化到指定的输出尺寸 (Od, Oh, Ow)
        imgs_ = F.adaptive_max_pool3d(imgs, (Od, Oh, Ow))
# 实例化测试设备类型相关的测试，使用 TestPoolingNNDeviceType 类和全局变量
instantiate_device_type_tests(TestPoolingNNDeviceType, globals())

# 实例化参数化测试，使用 TestPoolingNN 类
instantiate_parametrized_tests(TestPoolingNN)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 运行测试
    run_tests()
```