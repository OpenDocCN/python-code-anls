# `.\pytorch\test\nn\test_convolution.py`

```py
# Owner(s): ["module: nn"]
# 引入必要的模块和库
import itertools  # 引入迭代工具模块
import math  # 引入数学函数库
import unittest  # 引入单元测试框架
import warnings  # 引入警告模块
from itertools import product  # 从迭代工具模块中引入 product 函数

import torch  # 引入 PyTorch 库

# 引入 PyTorch 的额外模块和库
import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

# 从 PyTorch 的测试模块中引入必要的函数和类
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    TEST_CUDA,
    TEST_CUDNN,
    tf32_is_not_fp32,
    tf32_on_and_off,
)
from torch.testing._internal.common_device_type import (
    disablecuDNN,
    disableMkldnn,
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    precisionOverride,
    skipCPUIfNoMkldnn,
    skipCUDAIfCudnnVersionLessThan,
    skipCUDAIfMiopen,
    skipCUDAIfNoCudnn,
    skipCUDAIfNoMiopen,
    skipCUDAIfNotMiopenSuggestNHWC,
    skipCUDAIfRocm,
    skipCUDAIfRocmVersionLessThan,
    skipMeta,
)
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
    floating_types_and,
)
from torch.testing._internal.common_nn import _test_module_empty_input, NNTestCase
from torch.testing._internal.common_utils import (
    download_file,
    dtype2prec_DONTUSE,
    gradcheck,
    GRADCHECK_NONDET_TOL,
    gradgradcheck,
    instantiate_parametrized_tests,
    parametrize as parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfNotMiopenSuggestNHWC,
    skipIfRocmVersionLessThan,
    subtest,
    TEST_SCIPY,
    TEST_WITH_ROCM,
)

# 检查是否为 Ampere 架构或者 ROCm 平台
AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# 如果 TEST_SCIPY 为 True，则引入 scipy 的相关模块
if TEST_SCIPY:
    import scipy.ndimage
    import scipy.signal


class TestConvolutionNN(NNTestCase):
    _do_cuda_memory_leak_check = True  # 设置测试内存泄漏
    _do_cuda_non_default_stream = True  # 设置使用非默认 CUDA 流

    def test_conv_backcompat(self):
        from torch.serialization import SourceChangeWarning

        # 下载预先生成的 PyTorch 1.0.1 Python 2 版本的模型文件
        # 参考命令：import torch; from torch import nn; m = nn.Conv2d(1, 1, 1); torch.save(m, 'legacy_conv2d.pt')
        # 注意：该 Pickle 文件也包含一些 Unicode 数据！
        path = download_file("https://download.pytorch.org/test_data/legacy_conv2d.pt")
        
        # 忽略 SourceChangeWarning 警告，并加载模型文件
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SourceChangeWarning)
            m = torch.load(path, encoding="utf-8")
        
        # 创建随机输入张量
        input = torch.randn((1, 1, 1, 1), dtype=torch.float)
        
        # 断言模型输出的张量大小符合预期
        self.assertEqual(m(input).size(), (1, 1, 1, 1))
    # 测试不合法的 Conv1d 操作，遍历不同的数据类型
    def test_invalid_conv1d(self):
        for dtype in [
            torch.half,  # 半精度浮点数数据类型
            torch.bfloat16,  # Bfloat16 浮点数数据类型
            torch.float,  # 单精度浮点数数据类型
            torch.double,  # 双精度浮点数数据类型
            torch.cfloat,  # 复数单精度浮点数数据类型
            torch.cdouble,  # 复数双精度浮点数数据类型
        ]:
            # 创建 Conv1d 模块，设置输入通道为3，输出通道为33，卷积核大小为10，步长为1，包含偏置
            module = nn.Conv1d(
                in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True
            ).to(dtype)
            # 生成一个随机输入张量，并转换为指定的数据类型
            input = torch.randn(1, 3, 4).to(dtype)
            # 使用断言检查是否抛出预期的 RuntimeError 异常，验证卷积核大小不能大于实际输入大小
            with self.assertRaisesRegex(
                RuntimeError,
                r"Calculated padded input size per channel: \(4\). "
                + r"Kernel size: \(10\). Kernel size can\'t be greater than actual input size",
            ):
                module(input)

            # 检查负步长情况
            # 创建 Conv1d 模块，设置输入通道为3，输出通道为6，卷积核大小为3，步长为-1，包含偏置
            module = nn.Conv1d(
                in_channels=3, out_channels=6, kernel_size=3, stride=-1, bias=True
            ).to(dtype)
            # 生成一个随机输入张量，并转换为指定的数据类型
            input = torch.randn(1, 3, 4).to(dtype)
            # 使用断言检查是否抛出预期的 RuntimeError 异常，验证不支持非正步长
            with self.assertRaisesRegex(
                RuntimeError, "non-positive stride is not supported"
            ):
                module(input)

    # 测试不匹配的 Conv2d 输入形状
    def test_mismatch_shape_conv2d(self):
        for dtype in (torch.float, torch.cfloat):
            # 生成一个随机张量 x，形状为 [1, 10, 1, 28, 28]，数据类型为指定的 dtype
            x = torch.randn(1, 10, 1, 28, 28, dtype=dtype)
            # 生成一个随机权重张量 w，形状为 [6, 1, 5, 5]，数据类型为指定的 dtype
            w = torch.randn(6, 1, 5, 5, dtype=dtype)

            # 使用断言检查是否抛出预期的 RuntimeError 异常，验证输入形状不符合 Conv2d 的要求
            with self.assertRaisesRegex(
                RuntimeError,
                r"Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d, but got "
                + r"input of size: \[1, 10, 1, 28, 28\]",
            ):
                F.conv2d(x, w)

    # 测试不连续的 Conv2d 权重
    def test_conv2d_discontiguous_weight(self):
        for dtype in (torch.float, torch.cfloat):
            # 创建一个全为1的张量 x，形状为 [64, 16, 16, 16]，数据类型为指定的 dtype
            x = torch.ones(64, 16, 16, 16, dtype=dtype)
            # 创建一个不连续的权重张量 weight，进行测试特定问题的验证
            weight = (
                torch.arange(0, 1.0, 1 / 2.0**10)  # 生成一系列数值，形成张量
                .reshape(32, 16, 1, 2)  # 重塑张量形状
                .to(dtype)[:, :, :, ::2]  # 按索引切片操作
            )
            self.assertFalse(weight.is_contiguous())  # 断言权重张量不连续
            # 使用函数式接口进行 Conv2d 操作，验证特定问题的修复情况
            y = torch.nn.functional.conv2d(x, weight, None)
            if torch.backends.mkldnn.is_available():
                # 显式禁用 MKLDNN，以便使用 NNPACK 或 THCNN
                with torch.backends.mkldnn.flags(enabled=False):
                    y_ = torch.nn.functional.conv2d(x, weight, None)
                    self.assertEqual(y, y_)  # 断言两次 Conv2d 的结果相等
            self.assertEqual(y.sum(), 4186112.0)  # 断言 y 张量的总和为指定值
    # 定义测试函数，用于测试不合法的二维卷积操作
    def test_invalid_conv2d(self):
        # 遍历不同数据类型
        for dtype in [
            torch.half,
            torch.bfloat16,
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
        ]:
            # 创建一个二维卷积模块，指定参数包括输入通道数、输出通道数、核大小、扩张、步长，并转换为指定数据类型
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(
                dtype
            )
            # 创建一个空的张量作为输入，并将其转换为指定数据类型
            input = torch.empty(1, 1, 4, 4).to(dtype)
            # 断言会抛出运行时异常，lambda 表达式用于延迟执行卷积操作的检查
            self.assertRaises(RuntimeError, lambda: module(input))

            # 创建一个二维卷积模块，指定输入通道数、输出通道数、核大小、步长、是否带偏置，并不指定数据类型
            module = nn.Conv2d(
                in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True
            )
            # 创建一个随机输入张量，形状为(1, 3, 1, 1)
            input = torch.randn(1, 3, 1, 1)
            # 使用上下文管理器检查是否会抛出特定异常，正则表达式用于匹配异常消息
            with self.assertRaisesRegex(
                RuntimeError,
                r"Calculated padded input size per channel: \(1 x 1\). "
                + r"Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size",
            ):
                module(input)

            # 负步长检查
            module = nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True
            ).to(dtype)
            # 创建一个随机输入张量，并将其转换为指定数据类型
            input = torch.randn(1, 3, 4, 4).to(dtype)
            # 使用上下文管理器检查是否会抛出特定异常，异常消息指示不支持非正步长
            with self.assertRaisesRegex(
                RuntimeError, "non-positive stride is not supported"
            ):
                module(input)

            # 零步长检查
            module = nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True
            ).to(dtype)
            # 创建一个随机输入张量，并将其转换为指定数据类型
            input = torch.randn(1, 3, 4, 4).to(dtype)
            # 使用上下文管理器检查是否会抛出特定异常，异常消息指示不支持非正步长
            with self.assertRaisesRegex(
                RuntimeError, "non-positive stride is not supported"
            ):
                module(input)

    # 定义测试函数，用于测试不合法的三维卷积操作
    def test_invalid_conv3d(self):
        # 遍历不同数据类型
        for dtype in [
            torch.half,
            torch.bfloat16,
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
        ]:
            # 创建一个三维卷积模块，指定参数包括输入通道数、输出通道数、核大小、扩张、步长，并转换为指定数据类型
            module = torch.nn.Conv3d(1, 1, kernel_size=3, dilation=2, stride=2).to(
                dtype
            )
            # 创建一个空的张量作为输入，并将其转换为指定数据类型
            input = torch.empty(1, 1, 4, 4, 4).to(dtype)
            # 断言会抛出运行时异常，lambda 表达式用于延迟执行卷积操作的检查
            self.assertRaises(RuntimeError, lambda: module(input))

            # 负步长检查
            module = torch.nn.Conv3d(1, 1, kernel_size=3, stride=-2)
            # 创建一个空的张量作为输入
            input = torch.empty(1, 1, 4, 4, 4)
            # 使用上下文管理器检查是否会抛出特定异常，异常消息指示不支持非正步长
            with self.assertRaisesRegex(
                RuntimeError, "non-positive stride is not supported"
            ):
                module(input)
    # 测试对于无效的 groups 参数是否会引发异常
    def test_conv_invalid_groups(self):
        # 测试 Conv1d 构造函数，当 groups 参数为 0 时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "groups must be a positive integer"):
            torch.nn.Conv1d(1, 1, kernel_size=3, dilation=2, stride=2, groups=0)
        
        # 测试 Conv2d 构造函数，当 groups 参数为负数时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "groups must be a positive integer"):
            torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2, groups=-1)
        
        # 测试 Conv3d 构造函数，当 groups 参数为负数时是否抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "groups must be a positive integer"):
            torch.nn.Conv3d(1, 1, kernel_size=3, dilation=2, stride=2, groups=-2)

    # 测试 Conv1d 模块的 "same" padding 功能
    def test_Conv1d_module_same_padding(self):
        # 创建输入张量 x，大小为 (1, 1, 20)
        x = torch.rand(1, 1, 20)
        
        # 使用 nn.Conv1d 创建 Conv1d 模块，使用 "same" padding
        module = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=10, padding="same"
        )
        
        # 使用 F.conv1d 计算预期结果
        expect = F.conv1d(x, module.weight, module.bias, padding="same")
        
        # 断言 Conv1d 模块的输出与预期结果相等
        self.assertEqual(expect, module(x))

        # 测试 dilation 参数，使用对称的 "same" padding
        module = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=10, padding="same", dilation=2
        )
        expect = F.conv1d(x, module.weight, module.bias, padding="same", dilation=2)
        self.assertEqual(expect, module(x))

        # 测试非零的 padding_mode，需要显式指定 padding
        module = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=10,
            padding="same",
            padding_mode="replicate",
        )
        
        # 使用 F.pad 对输入张量 x 进行填充
        x_padded = F.pad(x, [4, 5], mode="replicate")
        
        # 使用 F.conv1d 计算预期结果，padding 使用 "valid"
        expect = F.conv1d(x_padded, module.weight, module.bias, padding="valid")
        
        # 断言 Conv1d 模块的输出与预期结果相等
        self.assertEqual(expect, module(x))
        
        # 断言输入张量 x 和预期结果的大小相等
        self.assertEqual(x.size(), expect.size())

        # 测试使用无效的 padding 字符串是否会引发异常
        with self.assertRaisesRegex(ValueError, "Invalid padding string"):
            module = nn.Conv1d(
                in_channels=3, out_channels=33, kernel_size=10, padding="foo"
            )
        
        # 测试使用 "same" padding 和 stride 同时设置是否会引发异常
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv1d(
                in_channels=3, out_channels=33, kernel_size=10, padding="same", stride=2
            )
    # 定义测试函数，用于测试 Conv2d 模块的 same padding 功能
    def test_Conv2d_module_same_padding(self):
        # 比较模块与函数式的结果：
        # 无步长/扩展，对称和非对称填充
        x = torch.rand(1, 1, 9, 20)
        
        # 创建一个 Conv2d 模块，设置输入通道数、输出通道数、内核大小和填充方式为 "same"
        module = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(5, 10), padding="same"
        )
        # 使用函数式的 conv2d 进行期望结果的计算
        expect = F.conv2d(x, module.weight, module.bias, padding="same")
        # 断言模块的输出与期望结果相等
        self.assertEqual(expect, module(x))
        
        # 使用扩展（dilation），对称填充
        module = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 4),
            padding="same",
            dilation=(1, 2),
        )
        expect = F.conv2d(
            x, module.weight, module.bias, padding="same", dilation=(1, 2)
        )
        self.assertEqual(expect, module(x))
        
        # 测试非零 padding_mode，需要显式填充
        module = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 4),
            padding="same",
            padding_mode="reflect",
        )
        # 使用函数式的 pad 进行反射填充
        x_padded = F.pad(x, [1, 2, 1, 1], mode="reflect")
        expect = F.conv2d(x_padded, module.weight, module.bias, padding="valid")
        # 断言模块的输出与期望结果相等
        self.assertEqual(expect, module(x))
        # 断言模块的输出与输入大小相等
        self.assertEqual(x.size(), expect.size())
        
        # 测试使用无效填充字符串抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "Invalid padding string"):
            module = nn.Conv2d(
                in_channels=3, out_channels=33, kernel_size=10, padding="foo"
            )
        
        # 测试使用相同填充和步长抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3, out_channels=33, kernel_size=10, padding="same", stride=2
            )
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3,
                out_channels=33,
                kernel_size=10,
                padding="same",
                stride=(1, 3),
            )
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3,
                out_channels=33,
                kernel_size=10,
                padding="same",
                stride=(4, 1),
            )
    # 定义测试方法，验证 Conv3d 模块在 same padding 下的功能
    def test_Conv3d_module_same_padding(self):
        # 创建一个 1x1x4x4x4 的随机张量作为输入
        x = torch.rand(1, 1, 4, 4, 4)
        
        # 创建一个 Conv3d 模块，设置输入通道数、输出通道数、核大小和 padding 方式为 "same"
        module = nn.Conv3d(
            in_channels=1, out_channels=1, kernel_size=(2, 3, 4), padding="same"
        )
        # 使用 functional 模块计算期望输出
        expect = F.conv3d(x, module.weight, module.bias, padding="same")
        # 验证模块输出是否与期望输出相等
        self.assertEqual(expect, module(x))

        # 创建一个带有 dilation 的 Conv3d 模块，设置相同的参数
        module = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 3, 4),
            padding="same",
            dilation=(3, 2, 1),
        )
        # 使用 functional 模块计算期望输出
        expect = F.conv3d(
            x, module.weight, module.bias, padding="same", dilation=(3, 2, 1)
        )
        # 验证模块输出是否与期望输出相等
        self.assertEqual(expect, module(x))

        # 测试使用非零 padding_mode，需要显式设置填充
        module = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 3, 4),
            padding="same",
            padding_mode="circular",
        )
        # 对输入张量进行循环填充
        x_padded = F.pad(x, [1, 2, 1, 1, 0, 1], mode="circular")
        # 使用 functional 模块计算期望输出
        expect = F.conv3d(x_padded, module.weight, module.bias, padding="valid")
        # 验证模块输出是否与期望输出相等
        self.assertEqual(expect, module(x))
        # 验证模块输出的尺寸是否与输入相同
        self.assertEqual(x.size(), expect.size())

        # 测试使用无效的 padding 字符串时是否会引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "Invalid padding string"):
            module = nn.Conv3d(
                in_channels=3, out_channels=33, kernel_size=10, padding="foo"
            )

        # 测试使用 padding='same' 和 strides 同时设置时是否会引发 ValueError 异常
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3, out_channels=33, kernel_size=10, padding="same", stride=2
            )
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3,
                out_channels=33,
                kernel_size=10,
                padding="same",
                stride=(1, 1, 3),
            )
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3,
                out_channels=33,
                kernel_size=10,
                padding="same",
                stride=(1, 4, 1),
            )
        with self.assertRaisesRegex(ValueError, "padding='same'"):
            module = nn.Conv2d(
                in_channels=3,
                out_channels=33,
                kernel_size=10,
                padding="same",
                stride=(5, 1, 1),
            )
    # 定义一个测试方法，用于测试不同的卷积操作
    def test_thnn_conv_strided_padded_dilated(self):
        # 对于每种卷积函数、维度和是否转置的组合，依次进行测试
        for convfn, dims, transposed in (
            (torch.nn.functional.conv2d, 2, False),   # 使用二维卷积函数
            (torch.nn.functional.conv_transpose2d, 2, True),  # 使用二维转置卷积函数
            (torch.nn.functional.conv3d, 3, False),   # 使用三维卷积函数
            (torch.nn.functional.conv_transpose3d, 3, True),  # 使用三维转置卷积函数
        ):
            # 对于每种步长、填充和膨胀的组合，依次进行测试
            for stride, padding, dilation in (
                (2, 0, 1),
                (1, 1, 1),
                (2, 1, 1),
                (1, 0, 2),
            ):
                kwargs = {"stride": stride, "padding": padding, "dilation": dilation}
                # 输入张量的形状
                inp_shape = (1, 2) + dims * (4,)
                # 权重张量的形状
                weight_shape = (2, 2) + dims * (1,)
                # 在CUDA上生成随机输入张量，双精度类型，需要梯度计算
                inputs = torch.randn(
                    inp_shape, dtype=torch.double, device="cuda", requires_grad=True
                )
                # 在CUDA上生成随机权重张量，双精度类型，需要梯度计算
                weight = torch.randn(
                    weight_shape, dtype=torch.double, device="cuda", requires_grad=True
                )
                # 在CUDA上生成随机偏置张量，双精度类型，需要梯度计算
                bias = torch.randn(
                    2, dtype=torch.double, device="cuda", requires_grad=True
                )
                # 禁用cuDNN以确保使用CPU执行卷积计算
                with torch.backends.cudnn.flags(enabled=False):
                    # 执行卷积操作
                    res = convfn(inputs, weight, bias, **kwargs)
                # 将输入、权重和偏置都移到CPU上，再次执行卷积操作
                res_cpu = convfn(inputs.cpu(), weight.cpu(), bias.cpu(), **kwargs)
                # 断言两次卷积结果应该相等
                self.assertEqual(res, res_cpu)
                # 使用gradcheck检查反向传播的数值梯度是否正确
                with torch.backends.cudnn.flags(enabled=False):
                    torch.autograd.gradcheck(
                        lambda x, w, b: convfn(x, w, b, **kwargs),
                        (inputs, weight, bias),
                    )
                    torch.autograd.gradcheck(
                        lambda x, w, b: convfn(x, w, b, **kwargs),
                        (inputs.cpu(), weight.cpu(), bias.cpu()),
                    )

    # 定义一个测试方法，用于测试不同类型的输入在卷积操作时的一致性
    def test_Conv2d_inconsistent_types(self):
        # 生成一个随机输入张量，单精度浮点类型
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float)
        # 生成一个随机权重张量，双精度浮点类型
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double)
        # 预期这种类型不一致应该抛出运行时异常
        self.assertRaises(RuntimeError, lambda: nn.functional.conv2d(inputs, weights))
        # 使用相同类型的输入进行卷积操作应该正常运行
        nn.functional.conv2d(inputs.float(), weights.float())

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_Conv2d_inconsistent_types_on_GPU_without_cudnn(self):
        # 创建一个形状为 (4, 1, 7, 7) 的张量作为输入数据，数据类型为 float，在 GPU 上
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="cuda")
        # 创建一个形状为 (1, 1, 3, 3) 的张量作为卷积核权重，数据类型为 double，在 GPU 上
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="cuda")
        # 创建一个形状为 (1,) 的张量作为偏置，数据类型为 double，在 GPU 上
        bias = torch.randn(1, dtype=torch.double, device="cuda")

        # 禁用 cuDNN 后，应该会因数据类型不一致而引发异常
        with torch.backends.cudnn.flags(enabled=False):
            # 断言调用 conv2d 函数时会抛出 RuntimeError 异常
            self.assertRaises(
                RuntimeError, lambda: nn.functional.conv2d(inputs, weights)
            )
            # 断言调用 conv2d 函数时会抛出 RuntimeError 异常
            self.assertRaises(
                RuntimeError,
                lambda: nn.functional.conv2d(inputs, weights.float(), bias),
            )

            # 但是相同数据类型时应该可以正常工作
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    def test_Conv2d_1x1(self):
        # 输入通道数和输出通道数均为 2
        in_channels = 2
        out_channels = 2
        # 创建一个 1x1 的卷积层模块，不带偏置，数据类型为 double
        mod = torch.nn.Conv2d(2, 2, 1, bias=False).to(dtype=torch.double)
        # 创建一个形状为 (1, 2, 5, 5) 的张量作为输入数据，数据类型为 double，需要梯度计算
        input = torch.randn(
            1, in_channels, 5, 5, requires_grad=True, dtype=torch.double
        )
        # 针对 mkldnn 启用和禁用两种状态进行梯度检查
        for enabled in (False, True):
            with torch.backends.mkldnn.flags(enabled=enabled):
                # 对 conv2d 函数进行梯度检查
                gradcheck(F.conv2d, (input, mod.weight))

    def test_Conv2d_OneDNN(self):
        def run_once(group_val=24, dilation=1):
            # 创建一个形状为 (1, group_val, 6, 6) 的张量作为输入数据，数据类型为 float32
            ifm = torch.ones([1, group_val, 6, 6], dtype=torch.float32)
            # 创建一个形状为 (group_val, 1, 3, 3) 的张量作为卷积核权重，数据类型为 float32
            weights = torch.ones([group_val, 1, 3, 3], dtype=torch.float32)
            # 创建一个卷积层，设置参数如下，并且不带偏置
            op = torch.nn.Conv2d(
                in_channels=group_val,
                out_channels=group_val,
                kernel_size=[3, 3],
                stride=[2, 2],
                padding=[1, 1],
                dilation=[dilation, dilation],
                groups=group_val,
                bias=False,
                padding_mode="zeros",
            )

            # 设置卷积层的权重数据为预设的 weights
            op.weight.data = weights
            # 对输入数据进行卷积操作，得到输出结果 res
            res = op(ifm)
            # 创建一个与 res 相同形状的张量，用于梯度反向传播
            grad_in = torch.ones(res.shape, dtype=torch.float32)
            # 对 res 进行反向传播计算梯度
            res.backward(grad_in)
            # 返回卷积层权重的梯度
            return op.weight.grad

        # 对不同的 group_val 和 dilation 参数组合进行测试
        for group_val in (24, 48, 23, 25):
            for dilation in (1, 2):
                # 在禁用 mkldnn 后运行一次卷积操作
                with torch.backends.mkldnn.flags(enabled=False):
                    without_onednn = run_once(group_val, dilation)

                # 在启用 mkldnn 后运行一次卷积操作
                with torch.backends.mkldnn.flags(enabled=True):
                    with_onednn = run_once(group_val, dilation)

                # 断言两次运行结果的卷积层权重梯度应该一致
                self.assertEqual(without_onednn, with_onednn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_cudnn_non_contiguous(self):
        # 创建一个形状为 (192, 16, 50) 的张量，并将其移动到 CUDA 设备上
        x = torch.randn(192, 16, 50).cuda()
        # 对 x 进行维度变换操作，使其变成连续的
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        # 创建一个一维卷积层，设置输入通道数为 16，输出通道数为 32，卷积核大小为 2，带偏置
        m = torch.nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=2, bias=True
        ).cuda()
        # 对输入数据进行卷积操作
        result = m(x)
    # 定义一个测试方法，验证 CUDNN 不会改变卷积操作的步幅
    def test_cudnn_not_mutate_stride(self):
        # 创建一个随机张量作为卷积的权重，大小为 (64, 64, 1, 1)
        weight = torch.randn(64, 64, 1, 1)
        # 创建一个随机张量作为输入数据，大小为 (2, 64, 10, 10)，并使用通道末尾的内存格式
        x = torch.randn(2, 64, 10, 10).to(memory_format=torch.channels_last)
        # 记录权重张量的步幅
        weight_stride = weight.stride()

        # 定义一个卷积函数，使用 torch.convolution 进行卷积操作
        def conv(x, weight):
            return torch.convolution(
                x,
                weight,
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
                bias=None,
            )

        # 在 NHWC 格式下运行卷积，不应改变输入张量的步幅
        out_nhwc = conv(x, weight)
        # 验证卷积后权重张量的步幅是否保持不变
        self.assertEqual(weight.stride(), weight_stride)
        # 验证输出是否在通道末尾内存格式下是连续的
        self.assertTrue(out_nhwc.is_contiguous(memory_format=torch.channels_last))

        # 将输入张量转换为连续格式
        x = x.contiguous(memory_format=torch.contiguous_format)
        # 使用连续格式的输入进行卷积操作
        out_c = conv(x, weight)
        # 验证输出是否在连续格式下是连续的
        self.assertTrue(out_c.is_contiguous(memory_format=torch.contiguous_format))
        # 验证输出结果是否与 NHWC 格式下的输出相同
        self.assertEqual(out_c, out_nhwc)
        # 再次验证卷积后权重张量的步幅是否保持不变
        self.assertEqual(weight.stride(), weight_stride)

    # 如果 CUDA 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    # 如果 CUDNN 不可用，则跳过测试
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    # 测试在 GPU 上使用 CUDNN 时，不同类型的输入是否会引发异常
    def test_Conv2d_inconsistent_types_on_GPU_with_cudnn(self):
        # 创建一个在 CUDA 上的随机输入张量，大小为 (4, 1, 7, 7)，数据类型为 float
        inputs = torch.randn(4, 1, 7, 7, dtype=torch.float, device="cuda")
        # 创建一个在 CUDA 上的随机权重张量，大小为 (1, 1, 3, 3)，数据类型为 double
        weights = torch.randn(1, 1, 3, 3, dtype=torch.double, device="cuda")
        # 创建一个在 CUDA 上的随机偏置张量，大小为 (1,)，数据类型为 double
        bias = torch.randn(1, dtype=torch.double, device="cuda")

        # 使用启用了 CUDNN 的后端
        with torch.backends.cudnn.flags(enabled=True):
            # 验证不同类型的输入是否会抛出 RuntimeError 异常
            self.assertRaises(
                RuntimeError, lambda: nn.functional.conv2d(inputs, weights)
            )
            self.assertRaises(
                RuntimeError,
                lambda: nn.functional.conv2d(inputs, weights.float(), bias),
            )

            # 但是相同类型的输入应该可以正常运行
            nn.functional.conv2d(inputs.float(), weights.float(), bias.float())

    # 测试 Conv2d 类的一个场景，验证缺少参数时是否会抛出 TypeError 异常
    def test_Conv2d_missing_argument(self):
        # 创建一个 Conv2d 类实例，输入通道为 3，输出通道为 3，卷积核大小为 3
        c = nn.Conv2d(3, 3, 3)
        # 验证调用时未提供参数是否会抛出 TypeError 异常
        self.assertRaises(TypeError, lambda: c(None))

    # 测试 Conv2d 的反向传播是否可以连续调用两次而不抛出异常
    def test_Conv2d_backward_twice(self):
        # 创建一个随机输入张量，大小为 (2, 3, 5, 5)
        input = torch.randn(2, 3, 5, 5)
        # 创建一个 Conv2d 类实例，输入通道为 3，输出通道为 3，卷积核大小为 3
        c = nn.Conv2d(3, 3, 3)
        # 对输入进行一次卷积操作，并对输出结果进行求和并反向传播
        o1 = c(input)
        o1.sum().backward()
        # 验证连续调用两次反向传播是否会抛出 RuntimeError 异常
        self.assertRaisesRegex(
            RuntimeError, "Specify retain_graph=True", lambda: o1.sum().backward()
        )
    def test_conv_modules_raise_error_on_incorrect_input_size(self):
        # 遍历不同的数据类型
        for dtype in [torch.half, torch.bfloat16, torch.double, torch.float]:
            # 创建包含各种卷积模块的列表，每个模块指定数据类型
            modules = [
                nn.Conv1d(3, 8, 3).to(dtype),
                nn.ConvTranspose1d(3, 8, 3).to(dtype),
                nn.Conv2d(3, 8, 3).to(dtype),
                nn.ConvTranspose2d(3, 8, 3).to(dtype),
                nn.Conv3d(3, 8, 3).to(dtype),
                nn.ConvTranspose3d(3, 8, 3).to(dtype),
            ]

            # 不合法的输入维度列表
            invalid_input_dims = [(1, 4), (1, 4), (2, 5), (2, 5), (3, 6), (3, 6)]

            # 遍历不合法维度和对应的模块
            for invalid_dims, module in zip(invalid_input_dims, modules):
                # 遍历每个维度值
                for dims in invalid_dims:
                    # 创建指定维度的空张量作为输入
                    input = torch.empty(torch.Size((3,) * dims))
                    # 断言调用模块时会抛出 RuntimeError 异常
                    self.assertRaises(RuntimeError, lambda: module(input))

    def test_conv_shapecheck(self):
        # 定义测试函数，用于测试模块的形状检查
        def test(should_raise, module, input_size, dtype):
            # 创建指定大小和数据类型的空张量作为输入
            input = torch.empty(3, *input_size).to(dtype)
            # 如果应该抛出异常，则断言调用模块时会抛出 RuntimeError 异常
            if should_raise:
                self.assertRaises(RuntimeError, lambda: module(input))
            else:
                # 否则，运行模块以确保不会抛出异常
                module(input)

        # 遍历不同的数据类型
        for dtype in [
            torch.half,
            torch.bfloat16,
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
        ]:
            # 测试 Conv1d 模块
            test(True, nn.Conv1d(1, 1, 3).to(dtype), (1, 2), dtype)
            test(True, nn.Conv1d(1, 1, 3, stride=2).to(dtype), (1, 2), dtype)
            test(False, nn.Conv1d(1, 1, 2).to(dtype), (1, 2), dtype)
            test(False, nn.Conv1d(1, 1, 2, stride=2).to(dtype), (1, 2), dtype)
            test(
                False, nn.Conv1d(1, 1, 3, stride=2, padding=1).to(dtype), (1, 2), dtype
            )

            # 测试 Conv2d 模块
            test(True, nn.Conv2d(1, 1, (3, 3)).to(dtype), (1, 2, 2), dtype)
            test(False, nn.Conv2d(1, 1, (3, 3)).to(dtype), (1, 3, 3), dtype)
            test(False, nn.Conv2d(1, 1, (3, 3), padding=1).to(dtype), (1, 2, 2), dtype)

            # 测试 Conv3d 模块
            test(True, nn.Conv3d(1, 1, (3, 3, 3)).to(dtype), (1, 2, 2, 2), dtype)
            test(False, nn.Conv3d(1, 1, (3, 3, 3)).to(dtype), (1, 3, 3, 3), dtype)
            test(
                False,
                nn.Conv3d(1, 1, (3, 3, 3), padding=1).to(dtype),
                (1, 2, 2, 2),
                dtype,
            )

    def test_ConvTranspose2d_output_size(self):
        # 创建 ConvTranspose2d 模块
        m = nn.ConvTranspose2d(3, 4, 3, 3, 0, 2)
        # 创建指定大小的随机张量作为输入
        i = torch.randn(2, 3, 6, 6)
        # 遍历不同的高度和宽度
        for h in range(15, 22):
            for w in range(15, 22):
                # 如果高度和宽度在指定范围内
                if 18 <= h <= 20 and 18 <= w <= 20:
                    # 调用模块并断言输出大小符合预期
                    output = m(i, output_size=(h, w))
                    self.assertEqual(output.size()[2:], (h, w))
                else:
                    # 否则，断言调用模块时会抛出 ValueError 异常
                    self.assertRaises(ValueError, lambda: m(i, (h, w)))
    # 定义测试方法：测试 ConvTranspose2d 的输出尺寸在下采样和上采样时的表现
    def test_ConvTranspose2d_output_size_downsample_upsample(self):
        # 设定测试用例的批量大小 b，输入通道数 c，隐藏通道数 hid_c
        b, c, hid_c = 2, 3, 2
        # 循环测试不同的输入高度 h
        for h in range(13, 24):
            # 循环测试不同的输入宽度 w
            for w in range(13, 17):
                # 循环测试不同的卷积核大小 k
                for k in range(2, 5):
                    # 循环测试不同的膨胀系数 d
                    for d in range(1, 5):
                        # 循环测试不同的步长 s
                        for s in range(1, 4):
                            # 循环测试不同的填充 p
                            for p in range(3):
                                # 创建卷积层对象 conv，使用给定的参数
                                conv = nn.Conv2d(
                                    in_channels=c,
                                    out_channels=hid_c,
                                    kernel_size=k,
                                    stride=s,
                                    padding=p,
                                    dilation=d,
                                )
                                # 创建反卷积层对象 t_conv，使用隐藏通道数作为输入通道数，原始通道数作为输出通道数
                                t_conv = nn.ConvTranspose2d(
                                    in_channels=hid_c,
                                    out_channels=c,
                                    kernel_size=k,
                                    stride=s,
                                    padding=p,
                                    dilation=d,
                                )
                                # 生成随机输入张量 i，形状为 (b, c, h, w)
                                i = torch.randn(b, c, h, w)
                                # 使用 conv 对 i 进行卷积操作，然后将结果通过 t_conv 进行反卷积操作，输出尺寸与 i 相同
                                out = t_conv(conv(i), output_size=i.shape)
                                # 断言输出张量 out 的空间维度（高度和宽度）与输入张量 i 的空间维度相同
                                self.assertEqual(out.size()[2:], i.size()[2:])

    # 定义测试方法：验证 ConvTranspose3d 对输出尺寸的正确处理
    def test_ConvTranspose3d_correct_output_size(self):
        # 创建 ConvTranspose3d 模块，输入和输出通道数均为 2，核大小为 2
        m = nn.ConvTranspose3d(2, 2, 2)
        # 创建随机输入张量 i，形状为 (1, 2, 1, 1, 1)
        i = torch.rand(1, 2, 1, 1, 1)
        # 使用 m 对 i 进行反卷积操作，输出尺寸指定为 (1, 2, 2, 2, 2)
        out = m(i, output_size=(1, 2, 2, 2, 2))

    # 标记为跳过测试，条件为 TEST_CUDA 为 False（CUDA 不可用）
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    # 定义测试方法：测试使用 CUDA 加速的 ConvTranspose2d 在半精度计算下的性能
    def test_ConvTranspose2d_half_cublas_gemm(self):
        # 关闭 cudnn 的一些标志，确保使用半精度计算
        with torch.backends.cudnn.flags(enabled=False):
            # 创建在 CUDA 设备上的随机输入张量 inputs，形状为 (1, 1, 16, 16)，数据类型为 torch.half
            inputs = torch.randn(1, 1, 16, 16, device="cuda", dtype=torch.half)
            # 创建卷积转置层对象 deconv，输入通道数为 1，输出通道数为 1，核大小为 3，步长为 2，填充为 1，输出填充为 1
            deconv = (
                nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
                .cuda()  # 将 deconv 移动到 CUDA 设备上
                .half()  # 使用半精度计算
            )
            # 对 inputs 进行反卷积操作，得到输出张量 output
            output = deconv(inputs)
            # 计算 output 的均值，并对其进行反向传播
            output.mean().backward()

    # 标记为与 https://github.com/pytorch/pytorch/pull/1273 相关的测试
    # 几乎与上述 `test_Conv2d_naive_groups` 完全相同
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    # 标记为跳过测试，条件为 TEST_WITH_ROCM 为 True（在 ROCm 上失败）
    @unittest.skipIf(TEST_WITH_ROCM, "Skipped on ROCm, since it is failing on ROCm 5.7")
    # 定义一个测试函数，用于测试没有偏置项的 Conv2d 操作，特别是涉及多个设备和数据类型的情况
    def test_Conv2d_groups_nobias(self):
        # 定义支持的设备和数据类型组合列表
        dev_dtypes = [("cpu", torch.float)]
        if TEST_CUDA:
            dev_dtypes += [("cuda", torch.float), ("cuda", torch.half)]
        if AMPERE_OR_ROCM:
            dev_dtypes += [("cuda", torch.bfloat16)]
        
        # 遍历每个设备和数据类型组合
        for device, dtype in dev_dtypes:
            # 创建一个 Conv2d 模块，设置输入通道数为 4，输出通道数为 4，卷积核大小为 3，分组卷积数为 2，无偏置
            m = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False).to(device, dtype)
            # 生成随机输入张量 i，形状为 (2, 4, 6, 6)，位于指定设备上，指定数据类型，并要求梯度计算
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            # 执行卷积操作
            output = m(i)
            # 创建随机梯度输出张量，形状为 (2, 4, 4, 4)，位于指定设备上，指定数据类型
            grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
            # 反向传播梯度
            output.backward(grad_output)

            # 创建另一个 Conv2d 模块 m1，设置输入通道数为 2，输出通道数为 2，卷积核大小为 3，无偏置
            m1 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            # 将 m 的前两个权重复制给 m1
            m1.weight.data.copy_(m.weight.data[:2])
            # 提取 i 的前两个通道数据，保证数据是连续的，并要求梯度计算
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            # 执行 m1 的卷积操作
            output1 = m1(i1)
            # 对 output1 执行反向传播，只传播前两个通道的梯度
            output1.backward(grad_output[:, :2].contiguous())

            # 创建另一个 Conv2d 模块 m2，设置输入通道数为 2，输出通道数为 2，卷积核大小为 3，无偏置
            m2 = nn.Conv2d(2, 2, kernel_size=3, bias=False).to(device, dtype)
            # 将 m 的后两个权重复制给 m2
            m2.weight.data.copy_(m.weight.data[2:])
            # 提取 i 的后两个通道数据，保证数据是连续的，并要求梯度计算
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            # 执行 m2 的卷积操作
            output2 = m2(i2)
            # 对 output2 执行反向传播，只传播后两个通道的梯度
            output2.backward(grad_output[:, 2:].contiguous())

            # 断言输出应该等于将 output1 和 output2 沿通道维度拼接而成的张量
            self.assertEqual(output, torch.cat([output1, output2], 1))
            # 断言输入的梯度应该等于将 i1 和 i2 的梯度沿通道维度拼接而成的张量
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 断言卷积核的梯度应该等于将 m1 和 m2 的卷积核梯度沿输出通道维度拼接而成的张量
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    # 几乎与上面的 `test_Conv2d_naive_groups` 函数相同
    # 用于覆盖特殊情况，当分组数大于 1，输入通道数除以分组数小于 16，输出通道数是 16 的倍数时
    # 参见 https://github.com/pytorch/pytorch/pull/18463#issuecomment-476563686
    # 以及 https://github.com/pytorch/pytorch/pull/18463#issuecomment-477001024
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    @unittest.skipIf(TEST_WITH_ROCM, "Skipped on ROCm, since it is failing on ROCm 5.7")
    # 定义一个测试方法，用于测试 Conv2d 层的 groups 参数设置为 2，且不使用偏置
    def test_Conv2d_groups_nobias_v2(self):
        # 设置随机种子，确保结果可重复
        torch.manual_seed(123)
        
        # 定义设备和数据类型的组合，仅包括 CPU 上的 float 类型
        dev_dtypes = [("cpu", torch.float)]
        
        # 如果测试 CUDA 可用，添加 CUDA 上的 float 和 half 类型
        if TEST_CUDA:
            dev_dtypes += [("cuda", torch.float), ("cuda", torch.half)]
        
        # 如果是 Ampere 或 ROCm 架构，添加 CUDA 上的 bfloat16 类型
        if AMPERE_OR_ROCM:
            dev_dtypes += [("cuda", torch.bfloat16)]
        
        # 遍历设备和数据类型的组合
        for device, dtype in dev_dtypes:
            # 创建一个 Conv2d 层，输入通道为 4，输出通道为 16，卷积核大小为 3x3，groups 设置为 2，不使用偏置
            m = nn.Conv2d(4, 16, kernel_size=3, groups=2, bias=False).to(device, dtype)
            
            # 生成随机输入张量 i，大小为 (2, 4, 6, 6)，在指定设备和数据类型上，并设置 requires_grad=True 以计算梯度
            i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
            
            # 将输入张量 i 传递给 Conv2d 层 m，得到输出张量 output
            output = m(i)
            
            # 生成随机梯度输出张量 grad_output，大小为 (2, 16, 4, 4)，在指定设备和数据类型上
            grad_output = torch.randn(2, 16, 4, 4, device=device, dtype=dtype)
            
            # 对输出张量 output 进行反向传播，传播梯度 grad_output
            output.backward(grad_output)

            # 创建第二个 Conv2d 层 m1，输入通道为 2，输出通道为 8，卷积核大小为 3x3，不使用偏置
            m1 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            
            # 将 m 的前 8 个输出通道的权重复制给 m1
            m1.weight.data.copy_(m.weight.data[:8])
            
            # 从输入张量 i 中选择前 2 个通道的数据，并保证其连续性，并设置 requires_grad=True 以计算梯度
            i1 = i.data[:, :2].contiguous().requires_grad_(True)
            
            # 将选择后的输入张量 i1 传递给 Conv2d 层 m1，得到输出张量 output1
            output1 = m1(i1)
            
            # 对输出张量 output1 进行反向传播，传播梯度 grad_output 的前 8 个通道
            output1.backward(grad_output[:, :8].contiguous())

            # 创建第三个 Conv2d 层 m2，输入通道为 2，输出通道为 8，卷积核大小为 3x3，不使用偏置
            m2 = nn.Conv2d(2, 8, kernel_size=3, bias=False).to(device, dtype)
            
            # 将 m 的后 8 个输出通道的权重复制给 m2
            m2.weight.data.copy_(m.weight.data[8:])
            
            # 从输入张量 i 中选择后 2 个通道的数据，并保证其连续性，并设置 requires_grad=True 以计算梯度
            i2 = i.data[:, 2:].contiguous().requires_grad_(True)
            
            # 将选择后的输入张量 i2 传递给 Conv2d 层 m2，得到输出张量 output2
            output2 = m2(i2)
            
            # 对输出张量 output2 进行反向传播，传播梯度 grad_output 的后 8 个通道
            output2.backward(grad_output[:, 8:].contiguous())

            # 断言输出张量 output 应为 torch.cat([output1, output2], 1) 的结果
            self.assertEqual(output, torch.cat([output1, output2], 1))
            
            # 断言输入张量 i 的梯度应为 torch.cat([i1.grad.data, i2.grad.data], 1) 的结果，
            # 允许的绝对误差为 dtype2prec_DONTUSE[dtype]，相对误差为 0
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            
            # 断言 m 的权重梯度应为 torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0) 的结果，
            # 允许的绝对误差根据 dtype 的不同而有所不同
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=1e-1 if dtype == torch.half else dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
    # 定义一个测试方法，用于测试 Conv3d 模块在 groups 和无偏置的情况下的行为
    def test_Conv3d_groups_nobias(self):
        # 设置随机种子以确保结果可重复
        torch.manual_seed(123)
        # 创建一个 Conv3d 模块，输入通道数为 4，输出通道数为 16，卷积核大小为 3x3x3，分组数为 2，无偏置，并将其放置在 CPU 上
        m = nn.Conv3d(4, 16, kernel_size=3, groups=2, bias=False).to("cpu", torch.float)
        # 生成一个随机张量作为输入数据，形状为 (2, 4, 6, 6, 6)，放置在 CPU 上，数据类型为浮点型，并设置 requires_grad 为 True
        i = torch.randn(
            2, 4, 6, 6, 6, device="cpu", dtype=torch.float, requires_grad=True
        )
        # 对输入数据进行卷积操作，得到输出结果
        output = m(i)
        # 生成一个随机张量作为输出梯度，形状为 (2, 16, 4, 4, 4)，放置在 CPU 上，数据类型为浮点型
        grad_output = torch.randn(2, 16, 4, 4, 4, device="cpu", dtype=torch.float)
        # 对输出结果进行反向传播，计算梯度
        output.backward(grad_output)

        # 创建另一个 Conv3d 模块，输入通道数为 2，输出通道数为 8，卷积核大小为 3x3x3，无偏置，并将其放置在 CPU 上
        m1 = nn.Conv3d(2, 8, kernel_size=3, bias=False).to("cpu", torch.float)
        # 将 m 模块的前 8 个输出通道的权重复制给 m1 模块
        m1.weight.data.copy_(m.weight.data[:8])
        # 从输入数据 i 中选择前 2 个通道的数据，保证数据在内存中连续，并设置 requires_grad 为 True
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        # 对选择的数据进行卷积操作，得到输出结果
        output1 = m1(i1)
        # 对输出结果进行反向传播，计算梯度
        output1.backward(grad_output[:, :8].contiguous())

        # 创建第二个 Conv3d 模块，输入通道数为 2，输出通道数为 8，卷积核大小为 3x3x3，无偏置，并将其放置在 CPU 上
        m2 = nn.Conv3d(2, 8, kernel_size=3, bias=False).to("cpu", torch.float)
        # 将 m 模块的后 8 个输出通道的权重复制给 m2 模块
        m2.weight.data.copy_(m.weight.data[8:])
        # 从输入数据 i 中选择后 2 个通道的数据，保证数据在内存中连续，并设置 requires_grad 为 True
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        # 对选择的数据进行卷积操作，得到输出结果
        output2 = m2(i2)
        # 对输出结果进行反向传播，计算梯度
        output2.backward(grad_output[:, 8:].contiguous())

        # 断言输出结果应该与将 output1 和 output2 沿通道维度拼接而成的张量相等
        self.assertEqual(output, torch.cat([output1, output2], 1))
        # 断言输入数据的梯度应该与将 i1.grad.data 和 i2.grad.data 沿通道维度拼接而成的张量相等
        self.assertEqual(
            i.grad.data,
            torch.cat([i1.grad.data, i2.grad.data], 1),
            atol=dtype2prec_DONTUSE[torch.float],
            rtol=0,
        )
        # 断言 m 模块的权重梯度应该与将 m1.weight.grad.data 和 m2.weight.grad.data 沿输出通道维度拼接而成的张量相等
        self.assertEqual(
            m.weight.grad.data,
            torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
            atol=dtype2prec_DONTUSE[torch.float],
            rtol=dtype2prec_DONTUSE[torch.float],
        )
    def test_Conv3d_groups_wbias(self):
        # 设置随机种子为123，保证结果可重复性
        torch.manual_seed(123)
        # 创建一个Conv3d模块，输入通道数4，输出通道数16，3x3x3的卷积核，分组数为2，带有偏置项，放置在CPU上
        m = nn.Conv3d(4, 16, kernel_size=3, groups=2, bias=True).to("cpu", torch.float)
        # 生成一个随机输入张量，形状为[2, 4, 6, 6, 6]，放置在CPU上，数据类型为float，需要梯度
        i = torch.randn(
            2, 4, 6, 6, 6, device="cpu", dtype=torch.float, requires_grad=True
        )
        # 将输入张量传入Conv3d模块，得到输出张量
        output = m(i)
        # 创建一个随机梯度张量，形状为[2, 16, 4, 4, 4]，放置在CPU上，数据类型为float
        grad_output = torch.randn(2, 16, 4, 4, 4, device="cpu", dtype=torch.float)
        # 对输出张量进行反向传播，计算梯度
        output.backward(grad_output)

        # 创建另一个Conv3d模块，输入通道数2，输出通道数8，3x3x3的卷积核，带有偏置项，放置在CPU上
        m1 = nn.Conv3d(2, 8, kernel_size=3, bias=True).to("cpu", torch.float)
        # 将第一个Conv3d模块的前8个输出通道的权重复制给当前模块
        m1.weight.data.copy_(m.weight.data[:8])
        # 将第一个Conv3d模块的前8个输出通道的偏置复制给当前模块
        m1.bias.data.copy_(m.bias.data[:8])
        # 从输入张量中选择前两个通道的数据，并确保数据在内存中是连续的，同时需要梯度
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        # 将选定的输入数据传入当前Conv3d模块，计算输出张量
        output1 = m1(i1)
        # 对输出张量进行反向传播，计算梯度
        output1.backward(grad_output[:, :8].contiguous())

        # 创建另一个Conv3d模块，输入通道数2，输出通道数8，3x3x3的卷积核，带有偏置项，放置在CPU上
        m2 = nn.Conv3d(2, 8, kernel_size=3, bias=True).to("cpu", torch.float)
        # 将第一个Conv3d模块的后8个输出通道的权重复制给当前模块
        m2.weight.data.copy_(m.weight.data[8:])
        # 将第一个Conv3d模块的后8个输出通道的偏置复制给当前模块
        m2.bias.data.copy_(m.bias.data[8:])
        # 从输入张量中选择后两个通道的数据，并确保数据在内存中是连续的，同时需要梯度
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        # 将选定的输入数据传入当前Conv3d模块，计算输出张量
        output2 = m2(i2)
        # 对输出张量进行反向传播，计算梯度
        output2.backward(grad_output[:, 8:].contiguous())

        # 断言合并后的输出张量等于两个模块输出张量在通道维度上的连接
        self.assertEqual(output, torch.cat([output1, output2], 1))
        # 断言输入张量的梯度等于两个模块输入张量梯度在通道维度上的连接，使用给定的浮点数精度
        self.assertEqual(
            i.grad.data,
            torch.cat([i1.grad.data, i2.grad.data], 1),
            atol=dtype2prec_DONTUSE[torch.float],
            rtol=dtype2prec_DONTUSE[torch.float],
        )
        # 断言模块权重的梯度等于两个模块权重梯度在通道维度上的连接，使用给定的浮点数精度
        self.assertEqual(
            m.weight.grad.data,
            torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
            atol=dtype2prec_DONTUSE[torch.float],
            rtol=dtype2prec_DONTUSE[torch.float],
        )
        # 断言模块偏置的梯度等于两个模块偏置梯度在通道维度上的连接，使用给定的浮点数精度
        self.assertEqual(
            m.bias.grad.data,
            torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
            atol=dtype2prec_DONTUSE[torch.float],
            rtol=dtype2prec_DONTUSE[torch.float],
        )
    def test_grouped_conv_cudnn_nhwc_support(self):
        # 检测早期 cuDNN 版本中 NHWC 支持中分组卷积的漏洞
        # 创建 CUDA 设备上的随机输入张量，使用通道最后的内存格式
        input = torch.randn((16, 16, 8, 8), dtype=torch.float16, device="cuda").to(
            memory_format=torch.channels_last
        )
        # 创建 CUDA 设备上的随机权重张量，使用通道最后的内存格式
        weight = torch.randn((8, 4, 3, 3), dtype=torch.float16, device="cuda").to(
            memory_format=torch.channels_last
        )
        # 执行卷积操作，检测 NHWC 支持中的分组卷积
        out = torch.convolution(
            input, weight, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 4
        )
        # 创建另一个随机输入张量，并转置通道顺序为通道最后的内存格式
        input = torch.randn((16, 8, 8, 8), dtype=torch.float16, device="cuda").to(
            memory_format=torch.channels_last
        )
        # 执行卷积操作，检测转置后的通道顺序是否正确
        out_transpose = torch.convolution(
            input, weight, None, (1, 1), (1, 1), (1, 1), True, (0, 0), 4
        )

    @unittest.expectedFailure
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @unittest.skipIf(not TEST_CUDNN, "needs cudnn")
    def test_conv_cudnn_memory_layout_dominance(self):
        # 期望的行为是确保 conv.weight 的内存布局优先于输出的布局
        # 当前行为与期望不同，预计在后续的 PR 中修复此问题并移除 `expectedFailure` 标签
        # 创建包含梯度的 CUDA 设备上的随机输入张量
        input = torch.randint(
            1, 10, (2, 8, 4, 4), dtype=torch.float32, device="cuda", requires_grad=True
        )
        # 在 CUDA 设备上创建卷积层，并设置为 float 类型
        conv = nn.Conv2d(8, 4, 3).cuda().float()

        # 对输入执行卷积操作，并断言输出是否连续
        out = conv(input)
        self.assertTrue(out.is_contiguous())

        # 将输入张量转为通道最后的内存格式，并再次执行卷积操作
        input = input.contiguous(memory_format=torch.channels_last)
        out = conv(input)
        self.assertTrue(out.is_contiguous())

        # 将卷积层的权重数据转为通道最后的内存格式，并再次执行卷积操作
        conv.weight.data = conv.weight.contiguous(memory_format=torch.channels_last)
        out = conv(input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

        # 将输入张量恢复为默认的内存格式，并再次执行卷积操作
        input = input.contiguous()
        out = conv(input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_cudnn_noncontiguous_weight(self):
        # 在传递给 cuDNN 之前，非连续的权重必须先调用 contiguous()
        # 创建包含单个通道的双精度张量，并将其视图变形为 (1, 1, 3) 的张量
        input = torch.tensor([1, 1, 1], dtype=torch.double, device="cuda").view(1, 1, 3)
        # 创建扩展为 (1, 1, 2) 的双精度张量权重，这些权重是非连续的
        weights1 = torch.tensor([1], dtype=torch.double, device="cuda").expand(1, 1, 2)
        # 使用 contiguous() 方法创建连续的权重张量
        weights2 = (
            torch.tensor([1], dtype=torch.double, device="cuda")
            .expand(1, 1, 2)
            .contiguous()
        )
        # 使用 conv1d 函数对输入执行卷积操作，比较两种权重下的结果
        self.assertEqual(
            F.conv1d(input, weights1, bias=None, stride=2, dilation=2),
            F.conv1d(input, weights2, bias=None, stride=2, dilation=2),
        )
    # 定义一个测试函数，用于测试梯度卷积操作的前向和后向函数
    def run_grad_conv_test(self, func_forward, func_backward, dim=1, gradient="input"):
        # 针对不同的卷积核大小和输入尺寸进行迭代测试
        for kern, inp_size in [(3, 6), (3, 7), (4, 9)]:
            # 针对不同的批次大小、步长、填充、输入通道数、输出通道数、膨胀率进行组合迭代
            for batch, stride, padding, chan_in, chan_out, dilation in product(
                [1, 2], [1, 2], [0, 1, 2], [2], [3], [1]
            ):
                # 针对是否有偏置进行迭代测试
                for has_bias in [True, False]:
                    input_shape = [batch, chan_in]
                    weight_shape = [chan_out, chan_in]
                    # 根据指定的维度多次迭代，扩展输入和权重的形状
                    for _ in range(dim):
                        input_shape.append(inp_size)
                        weight_shape.append(kern)

                    # 创建随机输入张量，并声明需要计算梯度
                    input = torch.randn(input_shape, requires_grad=True)
                    weight = torch.randn(weight_shape, requires_grad=True)
                    # 如果有偏置，创建随机偏置张量，并声明需要计算梯度
                    if has_bias:
                        bias = torch.randn([chan_out], requires_grad=True)
                    # 调用前向函数计算输出
                    output = func_forward(
                        input,
                        weight,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=bias,
                    )

                    # 创建随机梯度张量
                    gradient_o = torch.randn(output.shape)
                    # 计算输出相对于输入或权重的梯度
                    gradient_w = torch.autograd.grad(
                        output, input if (gradient == "input") else weight, gradient_o
                    )

                    # 使用后向函数计算的梯度与自动求导计算的梯度进行断言比较
                    self.assertEqual(
                        gradient_w[0],
                        func_backward(
                            input_shape if (gradient == "input") else input,
                            weight_shape if (gradient == "weight") else weight,
                            gradient_o,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                        ),
                    )

    # 测试一维卷积输入的梯度
    def test_grad_conv1d_input(self):
        self.run_grad_conv_test(F.conv1d, F.grad.conv1d_input, 1, "input")

    # 测试一维卷积权重的梯度
    def test_grad_conv1d_weight(self):
        self.run_grad_conv_test(F.conv1d, F.grad.conv1d_weight, 1, "weight")

    # 测试二维卷积输入的梯度
    def test_grad_conv2d_input(self):
        self.run_grad_conv_test(F.conv2d, F.grad.conv2d_input, 2, "input")

    # 测试二维卷积权重的梯度
    def test_grad_conv2d_weight(self):
        self.run_grad_conv_test(F.conv2d, F.grad.conv2d_weight, 2, "weight")

    # 测试三维卷积输入的梯度
    def test_grad_conv3d_input(self):
        self.run_grad_conv_test(F.conv3d, F.grad.conv3d_input, 3, "input")

    # 测试三维卷积权重的梯度
    def test_grad_conv3d_weight(self):
        self.run_grad_conv_test(F.conv3d, F.grad.conv3d_weight, 3, "weight")

    # 如果 NNPACK 不可用，跳过该测试
    @unittest.skipIf(not torch._nnpack_available(), "NNPACK unavailable")
    # 定义一个测试函数，用于测试 NNPACK 卷积
    def test_nnpack_conv(self):
        # 遍历不同的卷积核大小和输入大小组合
        for kern, inp_size in [(3, 6), (3, 7), (4, 9)]:
            # 遍历不同的批次大小、步长、填充、输入通道数、输出通道数的组合
            for batch, stride, padding, chan_in, chan_out in product(
                [1, 2, 3, 4], [1, 2], [0, 1, 2], [2], [3]
            ):
                # 遍历是否有偏置的情况
                for has_bias in [True, False]:
                    # 设置输入形状和权重形状
                    input_shape = [batch, chan_in]
                    weight_shape = [chan_out, chan_in]
                    # 根据卷积核大小和输入大小，更新输入形状和权重形状
                    for _ in range(2):
                        input_shape.append(inp_size)
                        weight_shape.append(kern)

                    # 生成随机输入张量和权重张量
                    input = torch.randn(
                        input_shape, requires_grad=True, dtype=torch.float
                    )
                    weight = torch.randn(
                        weight_shape, requires_grad=True, dtype=torch.float
                    )
                    # 如果有偏置，生成偏置张量
                    if has_bias:
                        bias = torch.randn(
                            [chan_out], requires_grad=True, dtype=torch.float
                        )
                    
                    # 进行 NNPACK 空间卷积
                    output = torch._nnpack_spatial_convolution(
                        input, weight, stride=stride, padding=padding, bias=bias
                    )
                    # 计算预期输出
                    output_expected = torch.nn.functional.conv2d(
                        input, weight, stride=stride, padding=padding, bias=bias
                    )
                    # 断言实际输出和预期输出相等，使用给定的容差
                    self.assertEqual(output, output_expected, atol=3e-4, rtol=0)

                    # 生成随机梯度张量
                    gradient_o = torch.randn(output.shape, dtype=torch.float)

                    # 计算实际梯度和预期梯度
                    grads = torch.autograd.grad(output, [input, weight], gradient_o)
                    grads_expected = torch.autograd.grad(
                        output_expected, [input, weight], gradient_o
                    )
                    # 遍历梯度和预期梯度，断言它们相等，使用给定的容差
                    for gr, gr_expected in zip(grads, grads_expected):
                        self.assertEqual(gr, gr_expected, atol=3e-4, rtol=0)

    # 定义测试填充模式的函数
    def test_conv_padding_mode(self):
        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，检查填充模式是否为合法值
        with self.assertRaisesRegex(ValueError, "padding_mode must be one of"):
            nn.Conv2d(3, 3, 3, padding_mode="xyz")

        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，检查填充模式是否为合法值
        with self.assertRaisesRegex(ValueError, "padding_mode must be one of"):
            nn.Conv2d(3, 3, 3, padding_mode=3)

        # 使用 assertRaisesRegex 断言捕获 ValueError 异常，检查填充模式是否为合法值
        with self.assertRaisesRegex(ValueError, 'Only "zeros" '):
            nn.ConvTranspose2d(3, 3, 3, padding_mode="reflect")
    # 定义测试函数，测试卷积操作的梯度计算
    def test_functional_grad_conv(self):
        # Conv 1D
        # 创建一个形状为 (1, 1, 5) 的随机张量，需要计算梯度
        input = torch.randn(1, 1, 5, requires_grad=True)
        # 创建一个形状为 (1, 1, 3) 的随机张量，需要计算梯度
        weight = torch.randn(1, 1, 3, requires_grad=True)
        # 对输入张量进行一维卷积操作，使用给定的权重和扩张参数 dilation=2
        output = F.conv1d(input, weight, dilation=2)
        # 创建一个与输出张量相同形状的随机张量，作为输出的梯度
        grad_output = torch.randn(output.shape)

        # 使用自动求导函数计算输入张量和权重张量的梯度
        grad_input_autograd, grad_weight_autograd = torch.autograd.grad(
            output, (input, weight), grad_output
        )

        # 使用 functional 模块中的 conv1d_input 函数计算输入张量的梯度
        grad_input_functional = torch.nn.grad.conv1d_input(
            input.shape, weight, grad_output, dilation=2
        )
        # 断言 functional 计算得到的输入梯度与自动求导得到的输入梯度相等
        self.assertEqual(grad_input_functional, grad_input_autograd)

        # 使用 functional 模块中的 conv1d_weight 函数计算权重张量的梯度
        grad_weight_functional = torch.nn.grad.conv1d_weight(
            input, weight.shape, grad_output, dilation=2
        )
        # 断言 functional 计算得到的权重梯度与自动求导得到的权重梯度相等
        self.assertEqual(grad_weight_functional, grad_weight_autograd)

        # Conv 2D
        # 创建一个形状为 (1, 1, 5, 5) 的随机张量，需要计算梯度
        input = torch.randn(1, 1, 5, 5, requires_grad=True)
        # 创建一个形状为 (1, 1, 3, 3) 的随机张量，需要计算梯度
        weight = torch.randn(1, 1, 3, 3, requires_grad=True)
        # 对输入张量进行二维卷积操作，使用给定的权重和扩张参数 dilation=2
        output = F.conv2d(input, weight, dilation=2)
        # 创建一个与输出张量相同形状的随机张量，作为输出的梯度
        grad_output = torch.randn(output.shape)

        # 使用自动求导函数计算输入张量和权重张量的梯度
        (grad_input_autograd, grad_weight_autograd) = torch.autograd.grad(
            output, (input, weight), grad_output
        )

        # 使用 functional 模块中的 conv2d_input 函数计算输入张量的梯度
        grad_input_functional = torch.nn.grad.conv2d_input(
            input.shape, weight, grad_output, dilation=2
        )
        # 断言 functional 计算得到的输入梯度与自动求导得到的输入梯度相等
        self.assertEqual(grad_input_functional, grad_input_autograd)

        # 使用 functional 模块中的 conv2d_weight 函数计算权重张量的梯度
        grad_weight_functional = torch.nn.grad.conv2d_weight(
            input, weight.shape, grad_output, dilation=2
        )
        # 断言 functional 计算得到的权重梯度与自动求导得到的权重梯度相等
        self.assertEqual(grad_weight_functional, grad_weight_autograd)

        # Conv 3D
        # 创建一个形状为 (1, 1, 5, 5, 5) 的随机张量，需要计算梯度
        input = torch.randn(1, 1, 5, 5, 5, requires_grad=True)
        # 创建一个形状为 (1, 1, 3, 3, 3) 的随机张量，需要计算梯度
        weight = torch.randn(1, 1, 3, 3, 3, requires_grad=True)
        # 对输入张量进行三维卷积操作，使用给定的权重和扩张参数 dilation=2
        output = F.conv3d(input, weight, dilation=2)
        # 创建一个与输出张量相同形状的随机张量，作为输出的梯度
        grad_output = torch.randn(output.shape)

        # 使用自动求导函数计算输入张量和权重张量的梯度
        (grad_input_autograd, grad_weight_autograd) = torch.autograd.grad(
            output, (input, weight), grad_output
        )

        # 使用 functional 模块中的 conv3d_input 函数计算输入张量的梯度
        grad_input_functional = torch.nn.grad.conv3d_input(
            input.shape, weight, grad_output, dilation=2
        )
        # 断言 functional 计算得到的输入梯度与自动求导得到的输入梯度相等
        self.assertEqual(grad_input_functional, grad_input_autograd)

        # 使用 functional 模块中的 conv3d_weight 函数计算权重张量的梯度
        grad_weight_functional = torch.nn.grad.conv3d_weight(
            input, weight.shape, grad_output, dilation=2
        )
        # 断言 functional 计算得到的权重梯度与自动求导得到的权重梯度相等
        self.assertEqual(grad_weight_functional, grad_weight_autograd)
    def test_functional_grad_conv2d(self):
        # 定义测试函数，用于测试二维卷积的梯度计算功能

        BATCH_SIZE = 4
        IN_CH = 8
        OUT_CH = 16
        SPATIAL = 32

        def _test_conv2d(stride, kernel_size, groups, dilation):
            # 定义内部测试函数，用于测试不同参数下的二维卷积操作的梯度计算

            padding = kernel_size // 2

            # 创建一个具有随机数据的可微张量作为输入
            input = (
                torch.empty(BATCH_SIZE, IN_CH, SPATIAL, SPATIAL)
                .uniform_(-8.0, 8.0)
                .requires_grad_(True)
            )

            # 创建一个具有随机数据的可微张量作为卷积核权重
            weight = (
                torch.empty(OUT_CH, IN_CH // groups, kernel_size, kernel_size)
                .uniform_(-4.0, 4.0)
                .requires_grad_(True)
            )

            # 执行二维卷积操作，计算输出张量
            output = F.conv2d(
                input,
                weight,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )

            # 创建一个随机梯度张量作为输出张量的梯度
            grad_output = torch.randn(output.shape)

            # 使用自动微分功能计算输入张量和权重的梯度
            (grad_input_autograd, grad_weight_autograd) = torch.autograd.grad(
                output, (input, weight), grad_output
            )

            # 使用 functional 接口计算输入张量的梯度
            grad_input_functional = torch.nn.grad.conv2d_input(
                input.shape,
                weight,
                grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            self.assertEqual(grad_input_functional, grad_input_autograd)

            # 使用 functional 接口计算权重的梯度
            grad_weight_functional = torch.nn.grad.conv2d_weight(
                input,
                weight.shape,
                grad_output,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            self.assertEqual(grad_weight_functional, grad_weight_autograd)

        # 定义不同参数的组合进行测试
        strides = [1, 2]
        kernel_sizes = [1, 3, 5]
        groups = [1, 2, 4]
        dilates = [1, 2]

        # 遍历参数组合，调用 _test_conv2d 函数进行测试
        for s, k, g, d in product(strides, kernel_sizes, groups, dilates):
            _test_conv2d(s, k, g, d)

    def test_permute_conv2d_issue_120211(self):
        # 定义用于重现问题的函数，输入一个半径参数
        def reproducer(radius: int):
            # 创建一个随机张量作为输入图像，并对其进行维度置换
            image = torch.rand(1, 1024, 1024, 3)
            image = image.permute(0, 3, 1, 2)
            # 创建一个零张量作为卷积核，并使用 functional 接口进行卷积操作
            kernel_x = torch.zeros([3, 1, 1, radius * 2 + 1], device=image.device)
            image = torch.nn.functional.conv2d(image, kernel_x, groups=image.shape[-3])

        # 对半径从 0 到 127 的范围进行测试
        for i in range(0, 128):
            # 这里不应该出现问题
            reproducer(radius=i)

    def test_conv3d_issue_120406(self):
        # 这里不应该出现问题，执行一个三维卷积操作
        F.conv3d(torch.ones(2, 3, 8, 9, 26), torch.ones(3, 1, 1, 1, 17), groups=3)

    def test_conv1d_issue_120547(self):
        # 创建一个具有全一权重和偏置的一维卷积操作的输入
        weight = torch.ones([16, 1, 32])
        bias = torch.ones([16])
        stride, padding, dilation, groups = (1, 16, 1, 16)
        input = torch.rand((1, 1, 16))
        # 对输入进行维度置换，然后使用一维卷积操作
        input = input.transpose(1, 2)
        # 这里不应该出现问题
        F.conv1d(input, weight, bias, stride, padding, dilation, groups)
# 定义一个测试类 TestConvolutionNNDeviceType，继承自 NNTestCase
class TestConvolutionNNDeviceType(NNTestCase):

    # 定义一个测试方法 run_conv_double_back_test，用于测试卷积神经网络的双向传播
    def run_conv_double_back_test(
        self,
        kern,          # 卷积核大小
        stride,        # 步长
        padding,       # 填充
        chan_in,       # 输入通道数
        chan_out,      # 输出通道数
        batch_size,    # 批大小
        inp_size,      # 输入尺寸
        dilation,      # 膨胀率
        no_weight,     # 是否无权重
        groups=1,      # 分组数，默认为1
        use_cuda=False,   # 是否使用 CUDA，默认不使用
        use_bias=True,    # 是否使用偏置，默认使用
        dtype=torch.double,   # 数据类型，默认双精度
    ):

        # 根据 use_cuda 决定设备类型
        if use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # 创建一个随机张量 x，作为输入，设备类型和数据类型由参数决定
        x = torch.randn(
            batch_size,
            chan_in,
            inp_size,
            inp_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        # 创建一个随机权重张量，设备类型和数据类型由参数决定，是否需要梯度取决于 no_weight 参数
        weight = torch.randn(
            chan_out,
            chan_in // groups,
            kern,
            kern,
            device=device,
            dtype=dtype,
            requires_grad=not no_weight,
        )

        # 如果 use_bias 为 True，则创建一个随机偏置张量，设备类型和数据类型由参数决定，需要梯度
        # 否则，偏置设置为 None
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
        else:
            bias = None

        # 定义一个内部函数 func，接受任意数量的输入
        def func(*inputs):
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            # 在前向传播期间禁用 cudnn，以避免有限差分精度问题
            with cudnn.flags(enabled=False):
                # 执行卷积操作，输入为 lx，权重为 lweight，偏置为 lbias，其他参数由外部传入
                out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        # 根据 use_bias 决定输入参数
        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        # 调用 func 函数，并传入 inputs 参数，获取输出 dummy_out
        dummy_out = func(*inputs)

        # 创建一个与 dummy_out 相同尺寸的随机梯度张量 grad_y，设备类型和数据类型与 dummy_out 相同
        grad_y = torch.randn_like(
            dummy_out, device=device, dtype=dtype, requires_grad=True
        )

        # 如果数据类型为 torch.float，则测试 mkldnn 双向传播，由于精度问题，不运行 gradgradcheck
        if dtype == torch.float:
            (g,) = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad

        # 否则，运行 gradgradcheck 测试双向传播的梯度
        return gradgradcheck(func, inputs, (grad_y,))

    # 以下为装饰器和参数化测试的设置，未提供具体实现
    @onlyCUDA
    @skipCUDAIfNoCudnn
    @dtypes(
        *floating_and_complex_types_and(
            torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else []
        )
    )
    # 定义一个测试函数，用于测试在指定设备和数据类型上使用 Conv2d 模块的确定性行为
    def test_Conv2d_deterministic_cudnn(self, device, dtype):
        # 生成一个随机输入张量，形状为 (2, 3, 5, 5)，在指定设备和数据类型上，并且需要计算梯度
        inputs = torch.randn(2, 3, 5, 5, device=device, dtype=dtype, requires_grad=True)
        
        # 启用 cuDNN，确保使用基准模式和确定性模式
        with cudnn.flags(enabled=True, benchmark=True, deterministic=True):
            # 创建两个 Conv2d 层，每个层包含 3 个输入通道和 3 个输出通道，使用指定的设备和数据类型
            conv1 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            
            # 将 conv2 的偏置数据复制为 conv1 的偏置数据
            conv2.bias.data.copy_(conv1.bias.data)
            # 将 conv2 的权重数据复制为 conv1 的权重数据
            conv2.weight.data.copy_(conv1.weight.data)
            
            # 对输入 inputs 应用 conv1 和 conv2 层，得到输出 out1 和 out2
            out1 = conv1(inputs)
            out2 = conv2(inputs)
            
            # 使用断言确保 out1 和 out2 在给定的数值容差下完全相等
            self.assertEqual(out1, out2, atol=0.0, rtol=0)
            
            # 生成一个与 out1 相同形状的随机张量 y，并在指定设备和数据类型上计算其梯度
            y = torch.randn(out1.size(), device=device, dtype=dtype)
            out1.backward(y)
            out2.backward(y)
            
            # 使用断言确保 conv1 和 conv2 的偏置梯度完全相等
            self.assertEqual(
                conv1.bias.grad.data, conv2.bias.grad.data, atol=0.0, rtol=0
            )
            # 使用断言确保 conv1 和 conv2 的权重梯度完全相等
            self.assertEqual(
                conv1.weight.grad.data, conv2.weight.grad.data, atol=0.0, rtol=0
            )

    # 使用 onlyCUDA 装饰器标记的测试函数，用于测试在 CUDA 设备上使用 Conv2d 模块时的大工作空间需求
    @onlyCUDA
    # 使用 dtypes 装饰器标记，指定测试在浮点数类型以及半精度浮点数类型（若为 AMPERE 或 ROCM 架构）上进行
    @dtypes(
        *floating_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else [])
    )
    def test_Conv2d_large_workspace(self, device, dtype):
        # 定义多个输入大小，这些大小需要巨大的 cuDNN 工作空间，确保选择合理的算法以避免内存耗尽
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        # 定义一个运行测试的内部函数，根据 benchmark 参数设置 cuDNN 的行为
        def run_test(benchmark):
            # 在启用 cuDNN 的情况下，根据 benchmark 参数选择合适的算法
            with torch.backends.cudnn.flags(enabled=True, benchmark=benchmark):
                # 创建一个 Conv2d 层，输入通道数和输出通道数均为 256，内核大小为 3x3，填充为 1
                conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to(
                    device, dtype
                )
                # 遍历所有预定义的输入大小
                for size in sizes:
                    # 生成一个随机张量 x，形状为 size，在指定设备和数据类型上
                    x = torch.randn(size, device=device, dtype=dtype)
                    # 应用 conv 层并计算输出 out，然后对 out 执行反向传播
                    out = conv(x.detach().clone().requires_grad_())
                    out.backward(torch.ones_like(out))

        # 分别以 benchmark 为 False 和 True 运行测试函数
        run_test(benchmark=False)
        run_test(benchmark=True)

    # 使用 onlyCUDA 装饰器标记的测试函数，用于测试在 CUDA 设备上使用 ConvTranspose2d 模块时的大输出填充
    @onlyCUDA
    # 使用 dtypes 装饰器标记，指定测试在浮点数类型（torch.float, torch.double, torch.half）上进行
    @dtypes(torch.float, torch.double, torch.half)
    def test_ConvTranspose2d_large_output_padding(self, device, dtype):
        # 创建三个 ConvTranspose2d 层，每层具有不同的输入输出通道数和内核大小，以及输出填充
        net1 = torch.nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net2 = torch.nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net3 = torch.nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        
        # 生成一个随机张量 x，形状为 (1, 128, 6, 6)，在指定设备和数据类型上，并且需要计算梯度
        x = torch.rand(1, 128, 6, 6, device=device, dtype=dtype, requires_grad=True)
        
        # 应用 net1、net2 和 net3 层到 x 上，依次得到输出，并对最终输出执行反向传播
        x = net1(x)
        x = net2(x)
        x = net3(x)
        x.backward(torch.randn_like(x))
        
        # 同步所有 CUDA 核心，确保计算完成
        torch.cuda.synchronize()
    # 定义一个测试方法，用于测试深度可分离卷积的基本功能
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    @tf32_on_and_off(0.01)
    def test_Conv2d_depthwise_naive_groups(self, device, dtype):
        # 遍历不同的深度倍增器，这里分别为1和2
        for depth_multiplier in [1, 2]:
            # 创建一个深度可分离卷积层，输入通道数为2，输出通道数为2*depth_multiplier，核大小为3，组数为2，并移到指定设备和数据类型
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            # 生成一个随机输入张量，形状为[2, 2, 6, 6]，在CUDA设备上，指定数据类型，并进行归一化处理，同时需要计算梯度
            i = (
                torch.randn(2, 2, 6, 6, device="cuda", dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            # 对输入张量进行深度可分离卷积操作
            output = m(i)
            # 生成一个随机梯度输出张量，形状为[2, 2*depth_multiplier, 4, 4]，在指定设备上，指定数据类型，并进行归一化处理
            grad_output = (
                torch.randn(2, 2 * depth_multiplier, 4, 4, device=device, dtype=dtype)
                / 2
            )
            # 对输出进行反向传播
            output.backward(grad_output)

            # 计算偏移量，用于分割权重和偏置数据
            offset = 1 * depth_multiplier

            # 创建第一个深度可分离卷积层，输入通道数为1，输出通道数为1*depth_multiplier，核大小为3，并移到指定设备和数据类型
            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 将第一个深度可分离卷积层的权重数据设置为前半部分的权重数据的克隆
            m1.weight.data = m.weight.data[:offset].clone()
            # 将第一个深度可分离卷积层的偏置数据设置为前半部分的偏置数据的克隆
            m1.bias.data = m.bias.data[:offset].clone()
            # 对输入张量的前半部分进行分割和克隆，并需要计算梯度
            i1 = i.detach()[:, :1].clone().requires_grad_()
            # 对第一个深度可分离卷积层的输入进行卷积操作
            output1 = m1(i1)
            # 对第一个深度可分离卷积层的输出进行反向传播
            output1.backward(grad_output[:, :offset].contiguous())

            # 创建第二个深度可分离卷积层，输入通道数为1，输出通道数为1*depth_multiplier，核大小为3，并移到指定设备和数据类型
            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 将第二个深度可分离卷积层的权重数据设置为后半部分的权重数据
            m2.weight.data.copy_(m.weight.data[offset:])
            # 将第二个深度可分离卷积层的偏置数据设置为后半部分的偏置数据
            m2.bias.data.copy_(m.bias.data[offset:])
            # 对输入张量的后半部分进行分割和克隆，并需要计算梯度
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            # 对第二个深度可分离卷积层的输入进行卷积操作
            output2 = m2(i2)
            # 对第二个深度可分离卷积层的输出进行反向传播
            output2.backward(grad_output[:, offset:].contiguous())

            # 使用断言验证总输出是否等于两个分部分输出的连接
            self.assertEqual(
                output,
                torch.cat([output1, output2], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 使用断言验证输入张量的梯度是否等于两个分部分输入张量的梯度的连接
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 使用断言验证卷积层的偏置梯度是否等于两个分部分卷积层的偏置梯度的连接
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 使用断言验证卷积层的权重梯度是否等于两个分部分卷积层的权重梯度的连接
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    # 标记仅在CUDA环境下运行的测试方法
    @onlyCUDA
    # 指定数据类型为torch.float、torch.double、torch.half的测试方法
    @dtypes(torch.float, torch.double, torch.half)
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    @tf32_on_and_off(0.01)
    # 定义一个测试函数，用于测试 Conv3d 深度可分离卷积的原始组
    def test_Conv3d_depthwise_naive_groups(self, device, dtype):
        # 遍历不同的深度乘数
        for depth_multiplier in [1, 2]:
            # 创建一个 Conv3d 模型，设置输入通道数为2，输出通道数为2 * depth_multiplier，卷积核大小为3，分组数为2，转移到指定设备上
            m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            # 创建一个输入张量 i，形状为 [2, 2, 6, 6, 6]，在 CUDA 设备上，数据类型为指定的 dtype，并进行归一化处理，要求梯度
            i = (
                torch.randn(2, 2, 6, 6, 6, device="cuda", dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            # 将输入张量 i 输入到模型 m 中，得到输出张量 output
            output = m(i)
            # 创建一个梯度输出张量 grad_output，形状为 [2, 2 * depth_multiplier, 4, 4, 4]，在指定设备上，数据类型为指定的 dtype
            grad_output = (
                torch.randn(
                    2, 2 * depth_multiplier, 4, 4, 4, device=device, dtype=dtype
                )
                / 2
            )
            # 反向传播计算梯度
            output.backward(grad_output)

            # 计算偏移量
            offset = 1 * depth_multiplier

            # 创建另一个 Conv3d 模型 m1，设置输入通道数为1，输出通道数为1 * depth_multiplier，卷积核大小为3，转移到指定设备上
            m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 从 m 中复制权重数据到 m1，只复制前 offset 部分的权重
            m1.weight.data = m.weight.data[:offset].clone()
            # 从 m 中复制偏置数据到 m1，只复制前 offset 部分的偏置
            m1.bias.data = m.bias.data[:offset].clone()
            # 从输入张量 i 的第一个通道中分离数据，克隆为新的张量 i1，并要求梯度
            i1 = i.detach()[:, :1].clone().requires_grad_()
            # 将输入张量 i1 输入到模型 m1 中，得到输出张量 output1
            output1 = m1(i1)
            # 对 output1 进行反向传播计算梯度，只考虑前 offset 部分的梯度，并确保存储连续
            output1.backward(grad_output[:, :offset].contiguous())

            # 创建另一个 Conv3d 模型 m2，设置输入通道数为1，输出通道数为1 * depth_multiplier，卷积核大小为3，转移到指定设备上
            m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            # 从 m 中复制权重数据到 m2，复制从 offset 开始的权重数据
            m2.weight.data.copy_(m.weight.data[offset:])
            # 从 m 中复制偏置数据到 m2，复制从 offset 开始的偏置数据
            m2.bias.data.copy_(m.bias.data[offset:])
            # 从输入张量 i 的第二个通道中分离数据，克隆为新的张量 i2，并要求梯度
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            # 将输入张量 i2 输入到模型 m2 中，得到输出张量 output2
            output2 = m2(i2)
            # 对 output2 进行反向传播计算梯度，只考虑从 offset 开始的梯度，并确保存储连续
            output2.backward(grad_output[:, offset:].contiguous())
            
            # 检查设备是否为 cuda sm86，atol 和 rtol 根据数据类型和设备能力进行设置
            is_cuda_sm86 = device.startswith(
                "cuda"
            ) and torch.cuda.get_device_capability(0) == (8, 6)
            atol, rtol = (
                (3e-4, 3e-2)
                if dtype == torch.float32 and is_cuda_sm86
                else (dtype2prec_DONTUSE[dtype], 0)
            )

            # 使用 self.assertEqual 检查输出是否符合预期，要求输出张量 output 等于拼接后的 output1 和 output2，指定容差值 atol 和 rtol
            self.assertEqual(
                output, torch.cat([output1, output2], 1), atol=atol, rtol=rtol
            )
            # 使用 self.assertEqual 检查输入张量 i 的梯度是否符合预期，要求梯度张量等于拼接后的 i1.grad 和 i2.grad，指定容差值
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 使用 self.assertEqual 检查模型 m 的偏置梯度是否符合预期，要求梯度张量等于拼接后的 m1.bias.grad 和 m2.bias.grad，指定容差值
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            # 使用 self.assertEqual 检查模型 m 的权重梯度是否符合预期，要求梯度张量等于拼接后的 m1.weight.grad 和 m2.weight.grad，指定容差值
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=atol,
                rtol=rtol,
            )

    @onlyCUDA
    @dtypes(
        *floating_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else [])
    )
    def test_noncontig_conv_grad(self, device, dtype):
        # FIXME: remove after adding non-contiguous grad tests for all modules
        # 创建一个卷积模块，包含3个输入通道，输出通道数为5，卷积核大小为3x3，填充为1，转移到指定的设备和数据类型上
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device, dtype)
        # 创建一个形状为[2, 3, 10, 10]的随机张量作为输入，并且要求计算梯度
        input = torch.randn(
            2, 3, 10, 10, dtype=dtype, device=device, requires_grad=True
        )
        # 将输入张量传递给卷积模块，得到输出张量
        output = module(input)

        # 创建一个形状为[2, 2, 5, 10, 10]的随机张量作为梯度，其中第二个维度不是连续的
        grad = torch.randn(2, 2, 5, 10, 10, dtype=dtype, device=device)[:, 1]
        # 检查梯度张量是否连续
        assert not grad.is_contiguous()
        # 对输出张量执行反向传播，传入非连续的梯度，并保留计算图
        output.backward(grad, retain_graph=True)
        # 断言输入张量的梯度不为None
        self.assertIsNotNone(input.grad)
        # 复制输入张量的梯度数据作为预期结果
        result = input.grad.data.clone()
        # 清零输入张量的梯度数据
        input.grad.data.zero_()

        # 再次对输出张量执行反向传播，传入连续的梯度数据
        output.backward(grad.contiguous())
        # 使用指定的绝对误差和相对误差容差断言输入张量的梯度数据与预期结果一致
        self.assertEqual(
            result, input.grad.data, atol=dtype2prec_DONTUSE[dtype], rtol=0
        )

    @onlyCUDA
    @dtypes(torch.double)
    def test_conv_double_backward(self, device, dtype):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # 双向反向传播仅对双精度张量执行，因为精度原因
            batch_size = 1
            # 遍历不同的卷积核大小、输入尺寸、扩张率组合
            for kern, inp_size, dilations in [(3, 5, [1, 2]), (4, 9, [1])]:
                for stride, padding, chan_in, chan_out, dilation in product(
                    [1], [2], [2], [3], dilations
                ):
                    # 确定是否没有权重参数
                    no_weight = stride == 2
                    # 运行双向卷积反向传播测试，返回测试结果
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
                        use_cuda=True,
                        dtype=dtype,
                    )
                    # 断言测试结果为真，否则输出失败信息和测试参数
                    self.assertTrue(
                        result,
                        "Conv double backward test failed with parameters:"
                        + "\nkern: "
                        + str(kern)
                        + "\nstride: "
                        + str(stride)
                        + "\npadding: "
                        + str(padding)
                        + "\nchan_in: "
                        + str(chan_in)
                        + "\nchan_out: "
                        + str(chan_out)
                        + "\nbatch_size: "
                        + str(batch_size)
                        + "\ninp_size: "
                        + str(inp_size)
                        + "\ndilation: "
                        + str(dilation),
                    )
    def test_conv_double_backward_no_bias(self):
        # 定义卷积核大小为3
        kern = 3
        # 定义步幅为2
        stride = 2
        # 定义输入通道数和输出通道数分别为2和4
        chan_in, chan_out = 2, 4
        # 定义批处理大小为2
        batch_size = 2
        # 定义输入尺寸为5
        inp_size = 5
        # 定义填充大小为1
        padding = 1
        # 定义扩张大小为1
        dilation = 1
        # 是否没有权重，默认为False
        no_weight = False
        # 是否使用偏置，默认为True
        use_bias = True

        # 调用测试函数，并传入相应的参数
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

        # 断言测试结果为True，否则输出失败信息和各参数的数值
        self.assertTrue(
            result,
            "Conv double backward test failed with parameters:"
            + "\nkern: "
            + str(kern)
            + "\nstride: "
            + str(stride)
            + "\npadding: "
            + str(padding)
            + "\nchan_in: "
            + str(chan_in)
            + "\nchan_out: "
            + str(chan_out)
            + "\nbatch_size: "
            + str(batch_size)
            + "\ninp_size: "
            + str(inp_size)
            + "\ndilation: "
            + str(dilation),
        )

    def test_conv_double_backward_groups(self):
        # 定义卷积核大小为3
        kern = 3
        # 定义步幅为1
        stride = 1
        # 定义填充大小为2
        padding = 2
        # 定义输入通道数和输出通道数分别为2和4
        chan_in, chan_out = 2, 4
        # 定义批处理大小为2
        batch_size = 2
        # 定义输入尺寸为6
        inp_size = 6
        # 定义扩张大小为1
        dilation = 1
        # 是否没有权重，默认为False
        no_weight = False
        # 定义组数为2
        groups = 2

        # 调用测试函数，并传入相应的参数
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

        # 断言测试结果为True，否则输出失败信息和各参数的数值
        self.assertTrue(
            result,
            "Conv double backward test failed with parameters:"
            + "\nkern: "
            + str(kern)
            + "\nstride: "
            + str(stride)
            + "\npadding: "
            + str(padding)
            + "\nchan_in: "
            + str(chan_in)
            + "\nchan_out: "
            + str(chan_out)
            + "\nbatch_size: "
            + str(batch_size)
            + "\ninp_size: "
            + str(inp_size)
            + "\ndilation: "
            + str(dilation)
            + "\ngroups: "
            + str(groups),
        )

    def test_conv_double_backward_stride(self):
        # 定义批处理大小为2
        batch_size = 2

        # 对于每组(kern, inp_size, dilations)，分别为(3, 5, [1, 2])和(3, 7, [1])
        for kern, inp_size, dilations in [(3, 5, [1, 2]), (3, 7, [1])]:
            # 对于每组(stride, padding, chan_in, chan_out, dilation)的组合
            for stride, padding, chan_in, chan_out, dilation in product(
                [2], [0, 1], [1], [2], dilations
            ):
                # 是否没有权重，默认为False
                no_weight = False
                # 调用测试函数，并传入相应的参数
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

    @dtypes(torch.float, torch.cfloat)
    # 设置 cudnn 后端的标志，启用 cuDNN 加速，禁用基准测试
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    def test_conv1d_same_padding(self, device, dtype):
        # 测试 padding='same' 时输出的正确形状
        test_args = [
            # 输入大小
            range(50, 55),
            # 卷积核大小
            [1, 2, 3, 8],
            # 膨胀率
            range(1, 4),
            # 步长
            [1],
        ]
        # 遍历所有测试参数的组合
        for in_size, k_size, dilation, stride in itertools.product(*test_args):
            # 创建随机输入张量 x 和卷积核张量 y
            x = torch.rand(1, 1, in_size, device=device, dtype=dtype)
            y = torch.rand(1, 1, k_size, device=device, dtype=dtype)
            # 执行 F.conv1d 函数，使用 padding='same'，指定膨胀率和步长
            z = F.conv1d(x, y, padding="same", dilation=dilation, stride=stride)
            # 断言输出张量 z 的第二个维度大小符合预期值，使用 math.ceil 计算
            self.assertEqual(z.size(2), int(math.ceil(in_size / stride)))
    
        # 比较 F.conv1d 使用 padding='same' 输出与手动指定 padding 的结果
        # 没有指定步长和膨胀率
        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 3, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=1)
        actual = F.conv1d(x, y, padding="same")
        self.assertEqual(expect, actual)
    
        # 使用膨胀率
        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=3, dilation=2)
        actual = F.conv1d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual)
    
        # 使用不对称的膨胀率和 padding
        expect = F.conv1d(x, y, padding=5, dilation=3)[..., 1:]
        actual = F.conv1d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual)
    
    @dtypes(torch.float, torch.cfloat)
    def test_conv2d_same_padding(self, device, dtype):
        if dtype is torch.cfloat:
            rtol, atol = 2e-6, 2e-6
        else:
            rtol, atol = None, None
        # 比较 F.conv2d 使用 padding='same' 输出与手动指定 padding 的结果
        # 没有指定步长和膨胀率
        x = torch.rand(1, 1, 10, 11, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, 5, device=device, dtype=dtype)
        expect = F.conv2d(x, y, padding=(2, 2))[..., 1:, :]
        actual = F.conv2d(x, y, padding="same")
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)
    
        # 使用膨胀率
        y = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        expect = F.conv2d(x, y, padding=(2, 3), dilation=2)
        actual = F.conv2d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)
    
        # 使用不对称的膨胀率和 padding
        y = torch.rand(1, 1, 4, 4, device=device, dtype=dtype)
        expect = F.conv2d(x, y, padding=5, dilation=3)[..., 1:, 1:]
        actual = F.conv2d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)
    def test_conv3d_same_padding(self, device, dtype):
        # 如果数据类型是复数，则设置较高的相对和绝对误差容忍度
        if dtype is torch.cfloat:
            rtol, atol = 2e-6, 2e-6
        else:
            rtol, atol = None, None
        
        # 比较使用 padding='same' 的 F.conv3d 输出与手动填充的结果
        # 在没有 strides/dilation 的情况下
        x = torch.rand(1, 1, 10, 11, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 1, 2, 5, device=device, dtype=dtype)
        # 期望的输出，使用手动指定的 padding
        expect = F.conv3d(x, y, padding=(0, 1, 2))[..., :, 1:, :]
        # 实际的输出，使用 padding='same'
        actual = F.conv3d(x, y, padding="same")
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        # 使用 dilation 的情况
        expect = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        actual = F.conv3d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        # 使用不对称的 padding 和 dilation
        y = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=5, dilation=3)[..., 1:, 1:, 1:]
        actual = F.conv3d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)
    def test_conv1d_same_padding_backward(self, device, dtype):
        # Test F.conv1d gradients work with padding='same'
        # 创建一个形状为 (1, 1, 12) 的随机张量 x，用于计算梯度，要求其梯度可计算
        x = torch.rand(1, 1, 12, dtype=dtype, device=device, requires_grad=True)
        # 创建一个形状为 (1, 1, 4) 的随机张量 y，用于计算梯度，要求其梯度可计算
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        # 使用对称填充(padding=3)，进行卷积操作，得到输出张量 z
        z = F.conv1d(x, y, padding=3, dilation=2)
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 保存期望的 x 和 y 的梯度
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用 padding='same' 进行卷积操作，得到输出张量 z
        z = F.conv1d(x, y, padding="same", dilation=2)
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 断言得到的 x 和 y 的梯度与预期相同
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用不对称填充(padding=2)，进行卷积操作，取部分输出张量 z
        z = F.conv1d(x, y, padding=2)[..., 1:]
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 保存期望的 x 和 y 的梯度
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用 padding='same' 进行卷积操作，得到输出张量 z
        z = F.conv1d(x, y, padding="same")
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 断言得到的 x 和 y 的梯度与预期相同
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    @dtypes(torch.float, torch.cfloat)
    @tf32_on_and_off(0.001)
    def test_conv2d_same_padding_backward(self, device, dtype):
        # Test F.conv2d gradients work with padding='same'
        # 创建一个形状为 (1, 1, 10, 11) 的随机张量 x，用于计算梯度，要求其梯度可计算
        x = torch.rand(1, 1, 10, 11, device=device, dtype=dtype, requires_grad=True)
        # 创建一个形状为 (1, 1, 4, 5) 的随机张量 y，用于计算梯度，要求其梯度可计算
        y = torch.rand(1, 1, 4, 5, device=device, dtype=dtype, requires_grad=True)

        # 使用对称填充(padding=(3, 4))，进行卷积操作，得到输出张量 z
        z = F.conv2d(x, y, padding=(3, 4), dilation=2)
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 保存期望的 x 和 y 的梯度
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用 padding='same' 进行卷积操作，得到输出张量 z
        z = F.conv2d(x, y, padding="same", dilation=2)
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 断言得到的 x 和 y 的梯度与预期相同
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用不对称填充(padding=2)，进行卷积操作，取部分输出张量 z
        y = torch.rand(1, 1, 4, 4, device=device, dtype=dtype, requires_grad=True)
        z = F.conv2d(x, y, padding=2)[..., 1:, 1:]
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 保存期望的 x 和 y 的梯度
        gx_expect, gy_expect = x.grad, y.grad
        # 清空 x 和 y 的梯度
        x.grad, y.grad = None, None

        # 使用 padding='same' 进行卷积操作，得到输出张量 z
        z = F.conv2d(x, y, padding="same")
        # 对 z 的所有元素求和并取绝对值，然后反向传播计算梯度
        z.sum().abs().backward()
        # 断言得到的 x 和 y 的梯度与预期相同
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
    def test_conv3d_same_padding_backward(self, device, dtype):
        # 检查是否需要进行前向自动微分检查（不在XLA设备上）
        check_forward_ad = torch.device(device).type != "xla"

        # 创建具有梯度的随机张量 x 和 y，用于测试 F.conv3d 的梯度反向传播
        x = torch.rand(1, 1, 1, 11, 12, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 2, 5, dtype=dtype, device=device, requires_grad=True)

        # 测试使用 padding='same' 时 F.conv3d 的梯度是否有效
        # 对称填充
        z = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 使用 padding='same' 参数进行卷积操作
        z = F.conv3d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # 使用 gradcheck 函数验证梯度是否正确
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
            (x, y),
            check_forward_ad=check_forward_ad,
            nondet_tol=1e-5,
        )

        # 如果设备不是 CUDA，则进行 gradgradcheck
        if torch.device(device).type != "cuda":
            # https://github.com/pytorch/pytorch/issues/70702
            gradgradcheck(
                lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
                (x, y),
                check_fwd_over_rev=True,
            )

        # 非对称填充
        y = torch.rand(1, 1, 1, 4, 4, dtype=dtype, device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=2)[..., 1:, 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 使用 padding='same' 参数进行卷积操作
        z = F.conv3d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

        # 使用 gradcheck 函数验证梯度是否正确
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same"),
            (x, y),
            check_forward_ad=check_forward_ad,
            nondet_tol=1e-5,
        )

        # 如果设备不是 CUDA，则进行 gradgradcheck
        if torch.device(device).type != "cuda":
            # https://github.com/pytorch/pytorch/issues/70702
            gradgradcheck(
                lambda x, y: F.conv3d(x, y, padding="same"),
                (x, y),
                check_fwd_over_rev=True,
            )

    @dtypes(torch.float, torch.cfloat)
    def test_conv1d_valid_padding_backward(self, device, dtype):
        # 测试使用 padding='valid' 时 F.conv1d 的梯度是否有效
        x = torch.rand(1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        # 计算 F.conv1d 的梯度并进行反向传播
        F.conv1d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 使用 padding='valid' 参数进行卷积操作
        F.conv1d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float, torch.cfloat)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv1d_vs_scipy(self, device, dtype, mode):
        # 创建一个形状为 (1, 10) 的张量 t，放置在指定设备上，使用指定数据类型
        t = make_tensor((1, 10), device=device, dtype=dtype)
        # 获取张量的特征维度
        feat_dim = t.shape[1]
        # 创建一个形状为 (1, 1, 4) 的张量 weight_even，放置在指定设备上，使用指定数据类型
        weight_even = make_tensor((1, 1, 4), device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 5) 的张量 weight_odd，放置在指定设备上，使用指定数据类型
        weight_odd = make_tensor((1, 1, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # SciPy 需要两个一维输入。
            # 将张量 t 展平为一维数组，并转换为 numpy 数组
            t_a = t.view(-1).cpu().numpy()
            # 将权重张量 weight 展平为一维数组，并转换为 numpy 数组
            w_a = weight.view(-1).cpu().numpy()
            # 使用 SciPy 进行卷积，期望的结果
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                # 在 PyTorch 的 conv1d 中，`same` 填充与 SciPy 不同
                p = weight.shape[2] // 2
                # 对张量 t 进行填充处理
                t = torch.nn.functional.pad(t, (p, p))
                # 已经处理了填充，从参数中移除填充设置
                kwargs.pop("padding")

            # SciPy 中第二个输入是翻转的
            # 将权重张量按照最后一个维度翻转
            weight_flipped = torch.flip(weight, (2,))
            # 使用 PyTorch 的 conv1d 进行卷积操作，得到实际结果
            actual = torch.nn.functional.conv1d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                # 如果模式为 `same`，则截取实际结果的前 feat_dim 个元素
                actual = actual[:feat_dim]

            # 断言实际结果与期望结果在给定的容差范围内相等
            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)

        # 对于此测试套件，默认的数据类型是 torch.double
        # 这会导致类型提升，并且对于输入为 complex64 的情况，conv1d 输出 complex128。
        with set_default_dtype(torch.float):
            # 分别对 weight_even 和 weight_odd 进行测试
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float, torch.cfloat)
    @parametrize_test("mode", ("valid", "same"))
    # 定义一个测试方法，用于比较 PyTorch 的 conv2d 和 SciPy 的卷积操作
    def test_conv2d_vs_scipy(self, device, dtype, mode):
        # 创建形状为 (1, 5, 10) 的张量 t，设备为指定设备，数据类型为指定类型
        t = make_tensor((1, 5, 10), device=device, dtype=dtype)
        # 创建形状为 (1, 1, 2, 4) 的权重张量 weight_even
        weight_even = make_tensor((1, 1, 2, 4), device=device, dtype=dtype)
        # 创建形状为 (1, 1, 3, 5) 的权重张量 weight_odd

        weight_odd = make_tensor((1, 1, 3, 5), device=device, dtype=dtype)

        # 定义内部函数 _test，用于执行单个测试
        def _test(t, weight, mode):
            # SciPy 期望两个二维输入。
            # 将 t 去除第一个维度后转换为 numpy 数组 t_a
            t_a = t.squeeze(0).cpu().numpy()
            # 将 weight 去除第一个维度和第二个维度后转换为 numpy 数组 w_a
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            # 使用 SciPy 执行二维卷积，返回期望的结果 expected
            expected = scipy.signal.convolve2d(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            # 如果模式为 "same"
            if mode == "same":
                # PyTorch 的 conv2d 的 "same" padding 与 SciPy 不同
                # 计算左右填充和上下填充的数值
                left_right_pad = weight.shape[3] // 2
                top_bottom_pad = weight.shape[2] // 2
                p = (left_right_pad, left_right_pad, top_bottom_pad, top_bottom_pad)
                # 使用 PyTorch 的 pad 函数对 t 进行填充
                t = torch.nn.functional.pad(t, p)
                # 已经处理了填充，从 kwargs 中移除 "padding"
                kwargs.pop("padding")

            # 将 weight 按照 (2, 3) 的维度进行翻转，得到 weight_flipped
            weight_flipped = torch.flip(weight, (2, 3))
            # 使用 PyTorch 的 conv2d 执行卷积操作，得到 actual 结果，并去除第一个维度
            actual = torch.nn.functional.conv2d(t, weight_flipped, **kwargs).squeeze(0)
            # 如果模式为 "same"，则截取 actual 的部分区域以匹配 SciPy 的输出
            if mode == "same":
                actual = actual[:5, :10]

            # 使用断言方法进行实际结果与期望结果的比较，设置相对误差和绝对误差的容差
            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # 在此测试套件中，全局的数据类型为 torch.double
        # 这会导致类型提升的变化，conv1d 对于 complex64 的输入会输出 complex128。
        # 使用默认数据类型设置为 torch.float
        with set_default_dtype(torch.float):
            # 分别对 weight_even 和 weight_odd 执行 _test 函数的测试
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    # 如果未安装 SciPy，则跳过此测试
    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    # 使用 torch.float 和 torch.cfloat 作为测试的数据类型
    @dtypes(torch.float, torch.cfloat)
    # 参数化测试的模式为 "valid" 和 "same"
    @parametrize_test("mode", ("valid", "same"))
    # 定义一个测试函数，比较 PyTorch 的 conv3d 和 SciPy 的卷积效果
    def test_conv3d_vs_scipy(self, device, dtype, mode):
        # 创建一个形状为 (1, 5, 5, 10) 的张量
        t = make_tensor((1, 5, 5, 10), device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 2, 2, 4) 的权重张量
        weight_even = make_tensor((1, 1, 2, 2, 4), device=device, dtype=dtype)
        # 创建一个形状为 (1, 1, 2, 3, 5) 的权重张量
        weight_odd = make_tensor((1, 1, 2, 3, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # SciPy 要求两个三维输入。
            # 将 t 压缩成三维数组并转换为 NumPy 数组
            t_a = t.squeeze(0).cpu().numpy()
            # 将权重数组压缩并转换为 NumPy 数组
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            # 使用 SciPy 进行卷积操作，返回期望的结果
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                # PyTorch 的 conv3d 中的 `same` 填充与 SciPy 不同
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
                # 对输入张量 t 进行填充
                t = torch.nn.functional.pad(t, p)
                # 填充已经处理完毕，不再需要填充参数
                kwargs.pop("padding")

            # 在 SciPy 的卷积中，第二个输入是翻转的
            # 使用 torch.flip 对权重进行翻转
            weight_flipped = torch.flip(weight, (2, 3, 4))
            # 使用 PyTorch 的 conv3d 进行卷积操作，并压缩维度
            actual = torch.nn.functional.conv3d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                # 如果 mode 是 `same`，则截取 actual 的部分结果
                actual = actual[:5, :5, :10]

            # 如果使用的是 torch.float 或 torch.complex64，并且不支持 tf32，则使用指定的容差比较结果
            if tf32_is_not_fp32() and (
                dtype == torch.float or dtype == torch.complex64
            ):
                self.assertEqual(actual, expected, atol=0.05, rtol=0.05)
            else:
                self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # 在这个测试套件中，全局的数据类型为 torch.double
        # 这会影响类型提升和 conv1d 在 complex64 输入时输出 complex128 的行为
        with set_default_dtype(torch.float):
            # 对偶数形状的权重进行测试
            _test(t, weight_even, mode)
            # 对奇数形状的权重进行测试
            _test(t, weight_odd, mode)

    @dtypes(torch.float, torch.complex64)
    # 测试 conv2d 的有效填充情况下的反向传播
    def test_conv2d_valid_padding_backward(self, device, dtype):
        # 创建一个形状为 (1, 1, 1, 10) 的随机张量，并要求计算梯度
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype, requires_grad=True)
        # 创建一个形状为 (1, 1, 1, 4) 的随机张量，并要求计算梯度
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        # 使用 padding='valid' 进行 conv2d，并对结果求和后取绝对值并进行反向传播
        F.conv2d(x, y, padding=0).sum().abs().backward()
        # 记录期望的梯度值
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 使用 padding='valid' 进行 conv2d，并对结果求和后取绝对值并进行反向传播
        F.conv2d(x, y, padding="valid").sum().abs().backward()
        # 记录实际的梯度值
        gx_actual, gy_actual = x.grad, y.grad
        # 比较期望和实际的梯度值
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    @dtypes(torch.double, torch.cdouble)
    def test_conv3d_valid_padding_backward(self, device, dtype):
        # 检查是否需要进行前向自动求导（autograd）
        check_forward_ad = torch.device(device).type != "xla"

        # 创建具有梯度的随机张量 x 和 y，用于测试 F.conv3d 的梯度，padding='valid'
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        # 计算 F.conv3d(x, y, padding=0) 的和的绝对值，并执行反向传播
        F.conv3d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        # 计算 F.conv3d(x, y, padding="valid") 的和的绝对值，并执行反向传播
        F.conv3d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad

        # 断言计算得到的梯度与预期的梯度相等
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

        # 使用 gradcheck 验证梯度是否正确
        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_forward_ad=check_forward_ad,
        )

        # 使用 gradgradcheck 验证双重梯度是否正确
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_fwd_over_rev=check_forward_ad,
        )

    @parametrize_test("N", range(2, 4), name_fn=lambda N: f"ConvTranspose{N}d")
    def test_conv_transpose_with_output_size_and_no_batch_dim(self, device, N):
        # 对于没有批次维度的输入，验证在设置 output_size 时输出的形状是否正确
        # 参考 https://github.com/pytorch/pytorch/issues/75889
        inp = torch.randn((1, 15, 13) if N == 2 else (1, 15, 13, 13), device=device)
        output_size = (1, 240, 200) if N == 2 else (1, 240, 200, 200)

        # 动态获取 ConvTransposeNd 类，并创建实例 m
        ConvTransposeNd = getattr(nn, f"ConvTranspose{N}d")
        m = ConvTransposeNd(
            1, 1, kernel_size=16, stride=16, padding=7, bias=False, device=device
        )

        # 使用 m 进行转置卷积，并验证输出的形状是否与 output_size 一致
        output = m(inp, output_size=output_size)
        self.assertEqual(output.shape, output_size)

    @skipMeta
    @parametrize_test("has_bias", [False, True])
    @parametrize_test("strided", [False, True])
    @parametrize_test("contiguous", [False, True])
    def test_conv_backend(
        self,
        device,
        input_shape,
        has_bias,
        strided,
        contiguous,
        transposed,
        dilated,
        groups,
        layout,
        backend_expected,
    ):
        # 此处缺少部分代码，无法完整注释
    # 定义测试函数 test_conv_contiguous_for_oneDNN，用于测试在不同数据类型下的卷积操作
    def test_conv_contiguous_for_oneDNN(self):
        # 参考 PyTorch GitHub 上的问题 https://github.com/pytorch/pytorch/issues/80837。
        
        # 遍历三种数据类型：torch.float, torch.bfloat16, torch.half
        for dtype in [torch.float, torch.bfloat16, torch.half]:
            # 创建一个二维卷积层对象 conv
            conv = nn.Conv2d(
                1,                      # 输入通道数
                128,                    # 输出通道数
                kernel_size=(5, 2),     # 卷积核大小
                stride=(2, 1),          # 步长
                padding=(0, 1),         # 填充
                dilation=(1, 1),        # 空洞卷积参数
                groups=1,               # 分组卷积数
                bias=True,              # 是否使用偏置
                padding_mode="zeros",   # 填充模式
            ).to(dtype=dtype)           # 设置数据类型
            
            # 创建一个随机张量 x，形状为 [1, 2, 321, 201, 1]，并转置使得第二和最后一个维度交换
            x = torch.rand([1, 2, 321, 201, 1]).to(dtype=dtype)
            x = torch.transpose(x, 1, 4)
            x2 = x[..., 0]  # 取 x 的第一个通道的数据

            # 构建输入列表 inputs，包括 x2, conv.weight, conv.bias 和其他参数
            inputs = [
                x2,                 # 输入数据
                conv.weight,        # 卷积层权重
                conv.bias,          # 卷积层偏置
                (2, 1),             # 步长
                (0, 1),             # 填充
                (1, 1),             # 空洞卷积参数
                False,              # 是否使用 MKLDNN
                (0, 1),             # 填充
                1                   # 其他参数
            ]
            
            # 如果当前系统支持 MKLDNN 加速
            if torch.backends.mkldnn.is_available():
                # 对输入 x2 进行卷积操作，并将结果保存在 y 中
                y = conv(x2)
                # 显式禁用 MKLDNN
                with torch.backends.mkldnn.flags(enabled=False):
                    # 再次对输入 x2 进行卷积操作，并将结果保存在 y_ 中
                    y_ = conv(x2)
                    # 断言 y 和 y_ 相等
                    self.assertEqual(y, y_)

    # 标记为仅在 CPU 上运行的测试函数
    @onlyCPU
    def test_conv_ic1_channels_last_for_oneDNN(self):
        # 参考 PyTorch GitHub 上的问题 https://github.com/pytorch/pytorch/issues/82060, 当 N > 1 时将调用 OneDNN 路径。

        # 遍历两种数据类型：torch.float, torch.bfloat16, torch.half
        for dtype in [torch.float, torch.bfloat16, torch.half]:
            # 创建一个二维卷积层对象 conv，设置输入通道数为 1，输出通道数为 64，卷积核大小为 (3, 3)，填充为 (1, 1)，不使用偏置
            conv = torch.nn.Conv2d(
                1, 64, kernel_size=(3, 3), padding=(1, 1), bias=False
            )
            # 将 conv 对象转换为通道优先的内存格式，并设置数据类型为 dtype
            conv = conv.to(memory_format=torch.channels_last).to(dtype=dtype)

            # 创建一个随机张量 x，形状为 [2, 1, 100, 100]，并设置数据类型为 dtype
            x = torch.rand(2, 1, 100, 100).to(dtype=dtype)

            # 如果当前系统支持 MKLDNN 加速
            if torch.backends.mkldnn.is_available():
                # 对输入 x 进行卷积操作，并将结果保存在 y 中
                y = conv(x)
                # 显式禁用 MKLDNN
                with torch.backends.mkldnn.flags(enabled=False):
                    # 再次对输入 x 进行卷积操作，并将结果保存在 y_ 中
                    y_ = conv(x)
                    # 断言 y 和 y_ 相等
                    self.assertEqual(y, y_)
    
    # 标记为支持的数据类型的测试函数，包括 torch.float 和 torch.cfloat
    @dtypes(torch.float, torch.cfloat)
    # 测试零通道数的情况下的一维卷积
    def test_conv_empty_channel(self, device, dtype):
        # 设置输入通道数为0
        in_channels = 0
        # 创建一维卷积模块，输出通道为8，卷积核大小为2，步长为2，数据类型为dtype，放置在设备device上
        mod = torch.nn.Conv1d(in_channels, 8, 2, stride=2, dtype=dtype).to(device)
        # 创建一个形状为[2, 0, 15]的随机输入张量，放置在设备device上，数据类型为dtype
        inp = torch.randn(2, 0, 15, device=device, dtype=dtype)
        # 调用测试函数，验证空输入的情况
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 断言捕获运行时错误，期望错误信息包含"Given groups=1, weight"
        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            # 创建一个形状为[2, 1, 0]的随机输入张量，放置在设备device上，数据类型为dtype
            inp = torch.randn(2, 1, 0, device=device, dtype=dtype)
            # 将输入张量传递给模块进行计算
            mod(inp)

        # 创建二维卷积模块，输入通道数为0，输出通道为33，卷积核大小为3，步长为2，数据类型为dtype，放置在设备device上
        mod = torch.nn.Conv2d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        # 创建一个形状为[2, 0, 50, 100]的随机输入张量，放置在设备device上，数据类型为dtype
        inp = torch.randn(2, 0, 50, 100, device=device, dtype=dtype)
        # 调用测试函数，验证空输入的情况
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 断言捕获运行时错误，期望错误信息包含"Given groups=1, weight"
        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            # 创建一个形状为[2, 1, 40, 0]的随机输入张量，放置在设备device上，数据类型为dtype
            inp = torch.randn(2, 1, 40, 0, device=device, dtype=dtype)
            # 将输入张量传递给模块进行计算
            mod(inp)

        # 创建三维卷积模块，输入通道数为0，输出通道为33，卷积核大小为3，步长为2，数据类型为dtype，放置在设备device上
        mod = torch.nn.Conv3d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        # 创建一个形状为[2, 0, 50, 20, 40]的随机输入张量，放置在设备device上，数据类型为dtype
        inp = torch.randn(2, 0, 50, 20, 40, device=device, dtype=dtype)
        # 调用测试函数，验证空输入的情况
        _test_module_empty_input(self, mod, inp, check_size=False)

        # 断言捕获运行时错误，期望错误信息包含"Given groups=1, weight"
        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            # 创建一个形状为[2, 1, 50, 0, 40]的随机输入张量，放置在设备device上，数据类型为dtype
            inp = torch.randn(2, 1, 50, 0, 40, device=device, dtype=dtype)
            # 将输入张量传递给模块进行计算
            mod(inp)

    # 测试带组卷积的空输入情况
    def test_group_conv_empty(self, device):
        # 创建带组卷积的二维卷积模块，输入通道数和输出通道数均为4，步长为2，卷积核大小为3，填充为1，分组数为4，放置在设备device上
        mod = torch.nn.Conv2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4).to(
            device
        )
        # 创建一个形状为[0, 4, 4, 4]的随机输入张量，放置在设备device上
        inp = torch.randn(0, 4, 4, 4, device=device)
        # 调用测试函数，验证空输入的情况
        _test_module_empty_input(self, mod, inp, check_size=False)
        
        # 如果设备类型为CUDA并且支持cuDNN，则禁用cuDNN后再次调用测试函数，验证空输入的情况
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp, check_size=False)

    # 测试带组转置卷积的空输入情况
    def test_group_convTranspose_empty(self, device):
        # 创建带组转置卷积的二维卷积模块，输入通道数和输出通道数均为4，步长为2，卷积核大小为3，填充为1，分组数为4，放置在设备device上
        mod = torch.nn.ConvTranspose2d(
            4, 4, stride=2, kernel_size=3, padding=1, groups=4
        ).to(device)
        # 创建一个形状为[0, 4, 4, 4]的随机输入张量，放置在设备device上
        inp = torch.randn(0, 4, 4, 4, device=device)
        # 调用测试函数，验证空输入的情况
        _test_module_empty_input(self, mod, inp, check_size=False)
        
        # 如果设备类型为CUDA并且支持cuDNN，则禁用cuDNN后再次调用测试函数，验证空输入的情况
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp, check_size=False)

    # 测试带转置卷积的空输入情况
    def test_convTranspose_empty(self, device):
        # 创建带转置卷积的二维卷积模块，输入通道数和输出通道数均为4，步长为2，卷积核大小为3，填充为1，放置在设备device上
        mod = torch.nn.ConvTranspose2d(4, 4, stride=2, kernel_size=3, padding=1).to(
            device
        )
        # 创建一个形状为[0, 4, 4, 4]的随机输入张量，放置在设备device上
        inp = torch.randn(0, 4, 4, 4, device=device)
        # 调用测试函数，验证空输入的情况
        _test_module_empty_input(self, mod, inp, check_size=False)
        
        # 如果设备类型为CUDA并且支持cuDNN，则禁用cuDNN后再次调用测试函数，验证空输入的情况
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp, check_size=False)

    # 仅限于CUDA设备的测试装饰器，用于12GB大张量的测试
    # 测试在不拆分的情况下，卷积是否正确地路由到后备实现
    # 换句话说，确保没有崩溃。后备实现的正确性应在其他测试中进行验证
    def test_conv_large_nosplit(self, device):
        # 根据设备类型选择数据类型
        dtype = torch.half if self.device_type == "cuda" else torch.float
        # 创建一个卷积层，输入通道数为2，输出通道数为2，卷积核大小为8x8
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        # 创建一个大尺寸的输入张量，用于测试
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        # 对大尺寸输入进行卷积操作
        conv1(input_large)
        # 创建另一个卷积层，输入通道数为1，输出通道数为1024，卷积核大小为1x1
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        # 创建另一个大尺寸的输入张量，用于测试第二个卷积层
        input_large = torch.randn(1, 1, 2048, 1024, dtype=dtype, device=device)
        # 对第二个大尺寸输入进行卷积操作
        conv2(input_large)

    def test_conv_noncontig_weights(self, device):
        # 针对不连续的权重进行测试，包括1D、2D、3D的情况
        for dim in (1, 2, 3):
            for grouped in (False, True):
                # 设置通道数和组数
                nc = 3
                groups = 3 if grouped else 1
                # 创建随机权重张量，扩展维度以匹配卷积操作要求
                w = torch.randn([3] * dim, device=device)
                w = w.expand([nc, int(nc / groups)] + list(w.shape))
                w = w.detach().requires_grad_()
                # 创建随机输入张量，设置requires_grad=True以支持梯度计算
                x = torch.randn(
                    [1, nc] + ([5] * dim), device=device, requires_grad=True
                )
                # 执行卷积操作并进行反向传播
                y = getattr(F, f"conv{dim}d")(x, w, groups=groups)
                y.sum().backward()
                # 执行转置卷积操作并进行反向传播
                y = getattr(F, f"conv_transpose{dim}d")(x, w, groups=groups)
                y.sum().backward()

    def test_conv_noncontig_weights_and_bias(self, device):
        # 针对不连续的权重和偏置进行测试，需要使用浮点数以测试特定的问题
        for bias in [True, False]:
            # 创建卷积层，指定输入通道数、输出通道数、卷积核大小、步长、填充以及是否包含偏置
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=bias).to(
                device, torch.float
            )

            # 创建随机输入张量，其中一维是多余的，以测试特定问题
            input_nc = torch.randn(
                (1, 3, 224, 224, 2), device=device, dtype=torch.float
            )[:, :, :, :, 1]
            input_c = input_nc.contiguous()

            # 创建随机权重张量，其中一维是多余的，以测试特定问题
            weight_nc = torch.randn((64, 3, 7, 7, 2), device=device, dtype=torch.float)[
                :, :, :, :, 1
            ]
            conv1.weight = nn.Parameter(weight_nc)
            weight_c = conv1.weight.contiguous()

            if bias:
                # 创建随机偏置张量，其中一维是多余的，以测试特定问题
                bias_nc = torch.randn((64, 2), device=device, dtype=torch.float)[:, 1]
                conv1.bias = nn.Parameter(bias_nc)
                bias_c = conv1.bias.contiguous()

            # 对输入张量进行卷积操作，验证非连续权重和偏置的影响
            out1 = conv1(input_nc)
            conv1.weight = nn.Parameter(weight_c)
            if bias:
                conv1.bias = nn.Parameter(bias_c)
            out2 = conv1(input_c)
            # 断言两次卷积结果是否一致
            self.assertEqual(out1, out2)

    @onlyCUDA
    @largeTensorTest("12GB")
    @skipIfRocmVersionLessThan((6, 0))
    # 定义一个测试函数，用于测试大尺寸输入下的转置卷积操作，接受设备参数
    def test_conv_transposed_large(self, device):
        # 根据设备类型选择数据类型，如果是 CUDA 设备则为半精度，否则为单精度
        dtype = torch.half if self.device_type == "cuda" else torch.float
        # 创建一个转置卷积层，输出和输入通道数都为 1，卷积核大小为 1x1，步长为 1，无偏置项
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        # 创建一个大尺寸的随机输入张量，形状为 4096x1x512x1024，指定数据类型和设备
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        # 执行转置卷积操作
        ret = conv(input_large)
        # 计算四个子区域的最大差异
        maxdiff0 = (
            (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024)))
            .abs_()  # 取绝对值
            .max()    # 计算最大值
            .item()   # 转换为 Python 数值类型
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
        if self.device_type == "cuda":
            # 如果是 CUDA 设备，由于 cuDNN 可能使用 FFT 等算法，不保证差异为 0
            # 使用断言检查四个子区域的最大差异是否在给定的容差范围内
            self.assertEqual(maxdiff0, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff1, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff2, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff3, 0, atol=2e-3, rtol=1e-5)
        else:
            # 如果不是 CUDA 设备，直接使用断言检查四个子区域的最大差异是否为 0
            self.assertEqual(maxdiff0, 0)
            self.assertEqual(maxdiff1, 0)
            self.assertEqual(maxdiff2, 0)
            self.assertEqual(maxdiff3, 0)

    @onlyCUDA
    @skipCUDAIfRocm
    @largeTensorTest("12GB")
    # 定义一个测试方法，用于测试大规模卷积运算，接收一个设备参数
    def test_conv_large(self, device):
        # 根据设备类型选择数据类型，如果是 CUDA 设备则选择半精度（torch.half），否则选择单精度（torch.float）
        dtype = torch.half if self.device_type == "cuda" else torch.float
        # 创建一个卷积层对象，输入通道数为 2，输出通道数为 2，核大小为 8x8，没有偏置项
        conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
        # 创建一个大尺寸的输入张量，形状为 4097x2x512x512，数据类型和设备由之前的 dtype 和 device 指定
        input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
        
        # 前向传播
        ret = conv(input_large)
        # 断言前半部分输出是否与部分输入的卷积结果相等
        self.assertEqual(ret[:2048], conv(input_large[:2048]))
        # 断言中间部分输出是否与中间部分输入的卷积结果相等
        self.assertEqual(ret[2048:4096], conv(input_large[2048:4096]))
        # 断言末尾部分输出是否与末尾部分输入的卷积结果相等
        self.assertEqual(ret[4096:], conv(input_large[4096:]))
        
        # 反向传播
        conv.zero_grad()
        # 在计算反向传播时，使用 `max(dim=1)` 方法创建稀疏性，减少数值舍入误差
        # 若无此稀疏性，数值舍入误差可能达到 1e-5，无法满足 1e-6 的 `assertEqual` 标准
        ret.view(4097, -1).max(dim=1).values.sum().backward()
        del ret
        
        # 复制梯度并清零卷积层梯度
        grad1 = conv.weight.grad.detach().clone()
        conv.zero_grad()
        # 对不同部分的输入进行分批次的反向传播
        conv(input_large[:2048]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[2048:4096]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[4096:]).view(1, -1).max(dim=1).values.sum().backward()
        # 复制梯度
        grad2 = conv.weight.grad.detach().clone()
        
        # 梯度的量级较大，需要将其缩放到相近的量级，以便比较
        scale = 1 / grad2.abs().mean()
        grad1 = grad1 * scale
        grad2 = grad2 * scale
        
        # 断言两次反向传播得到的梯度是否相等，允许的绝对误差为 5e-2，相对误差为 5e-3
        self.assertEqual(grad1, grad2, atol=5e-2, rtol=5e-3)

    # 仅在 CUDA 设备上运行的装饰器
    @onlyCUDA
    # 如果是在 ROCm 平台上则跳过测试的装饰器
    @skipCUDAIfRocm
    # 在 CPU 上运行的大张量测试，张量大小为 "20GB"
    @largeTensorTest("20GB", "cpu")
    # 在 CUDA 上运行的大张量测试，张量大小为 "60GB"
    @largeTensorTest("60GB", "cuda")
    # 用于测试单个批次的大规模卷积
    def test_conv_large_batch_1(self, device):
        # 输入通道数
        in_channels = 514
        # 输入图像尺寸
        dim = 2048
        # 输出通道数
        out_channels = 1
        # 卷积核大小
        kernel_size = 3
        # 步长
        stride = 1
        # 填充
        padding = 1
        
        # 创建一个 CUDA 上的全 1 输入张量，数据类型为半精度
        input_tensor = torch.ones(1, in_channels, dim, dim).cuda().half()
        # 创建一个在 CUDA 上的卷积模型，输入通道数、输出通道数、核大小、步长和填充都由之前的变量指定，数据类型为半精度
        model = (
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            .cuda()
            .half()
        )
        # 计算模型在输入张量上的输出
        output = model(input_tensor)
        # 在 CPU 上创建一个单精度的相同模型，计算其在相同输入张量上的输出
        model_cpu = model.cpu().float()
        output_cpu = model(input_tensor.float().cpu())
        # 断言 CUDA 和 CPU 模型输出是否接近，允许的绝对误差和相对误差均为 1e-3
        self.assertEqual(output.cpu().float(), output_cpu, atol=1e-3, rtol=1e-3)

    # 仅在 CUDA 设备上运行的装饰器
    @onlyCUDA
    # 如果没有 CUDNN 则跳过测试的装饰器
    @skipCUDAIfNoCudnn
    def test_contig_wrong_stride_cudnn(self, device):
        # x has to have batch_size 1 to test contiguous checks
        # 创建一个形状为 (1, 16, 5, 5) 的随机张量 x，用于测试连续性检查
        x = torch.randn(1, 16, 5, 5, device=device)
        # 获取张量 x 的步长信息并转换为列表
        stride = list(x.stride())
        # 修改步长列表中的第一个元素为 20
        stride[0] = 20
        # 使用修改后的步长列表来重设张量 x 的步长信息，尽管 size[0] 仍为 1，张量仍保持连续
        x.set_(x.storage(), 0, x.size(), stride)
        # 断言张量 x 是否连续
        self.assertTrue(x.is_contiguous())
        # 对张量 x 进行转置卷积操作，传入一个形状为 (16, 1, 1, 1) 的随机张量
        F.conv_transpose2d(x, torch.randn(16, 1, 1, 1, device=device))
        # 对张量 x 进行普通卷积操作，传入一个形状为 (1, 16, 1, 1) 的随机张量
        F.conv2d(x, torch.randn(1, 16, 1, 1, device=device))

    @onlyCUDA
    @tf32_on_and_off(0.005)
    def test_Conv2d_size_1_kernel(self, device):
        # 创建一个形状为 (2, 3, 5, 5) 的随机张量 x_cpu
        x_cpu = torch.randn(2, 3, 5, 5)
        # 创建一个 kernel_size 为 1 的 Conv2d 层 conv_cpu
        conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
        # 对 x_cpu 应用 conv_cpu 层，得到输出 y_cpu
        y_cpu = conv_cpu(x_cpu)
        # 创建一个与 y_cpu 形状相同的随机张量 y
        y = torch.rand_like(y_cpu)
        # 对 y_cpu 进行反向传播
        y_cpu.backward(y)

        # 禁用 cuDNN 后，在 CUDA 设备上创建一个 kernel_size 为 1 的 Conv2d 层 conv_cuda
        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)
            # 复制 conv_cpu 的偏置数据到 conv_cuda
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            # 复制 conv_cpu 的权重数据到 conv_cuda
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            # 对 x_cpu 在 CUDA 设备上应用 conv_cuda 层，得到输出 y_cuda
            y_cuda = conv_cuda(x_cpu.to(device))
            # 对 y_cuda 在 CUDA 设备上进行反向传播
            y_cuda.backward(y.to(device))

        # 断言 y_cpu 和 y_cuda 在误差范围内相等，不考虑设备上的精确匹配
        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        # 断言 conv_cpu 和 conv_cuda 的偏置梯度数据在误差范围内相等，不考虑设备上的精确匹配
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        # 断言 conv_cpu 和 conv_cuda 的权重梯度数据在误差范围内相等，不考虑设备上的精确匹配
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )

    @onlyCUDA
    @tf32_on_and_off(0.005)
    def test_ConvTranspose2d_size_1_kernel(self, device):
        # 创建一个形状为 (2, 3, 5, 5) 的随机张量 x_cpu
        x_cpu = torch.randn(2, 3, 5, 5)
        # 创建一个 kernel_size 为 1 的 ConvTranspose2d 层 conv_cpu
        conv_cpu = torch.nn.ConvTranspose2d(3, 3, kernel_size=1)
        # 对 x_cpu 应用 conv_cpu 层，得到输出 y_cpu
        y_cpu = conv_cpu(x_cpu)
        # 创建一个与 y_cpu 形状相同的随机张量 y
        y = torch.rand_like(y_cpu)
        # 对 y_cpu 进行反向传播
        y_cpu.backward(y)

        # 禁用 cuDNN 后，在 CUDA 设备上创建一个 kernel_size 为 1 的 ConvTranspose2d 层 conv_cuda
        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.ConvTranspose2d(3, 3, kernel_size=1).to(device)
            # 复制 conv_cpu 的偏置数据到 conv_cuda
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            # 复制 conv_cpu 的权重数据到 conv_cuda
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            # 对 x_cpu 在 CUDA 设备上应用 conv_cuda 层，得到输出 y_cuda
            y_cuda = conv_cuda(x_cpu.to(device))
            # 对 y_cuda 在 CUDA 设备上进行反向传播
            y_cuda.backward(y.to(device))

        # 断言 y_cpu 和 y_cuda 在误差范围内相等，不考虑设备上的精确匹配
        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        # 断言 conv_cpu 和 conv_cuda 的偏置梯度数据在误差范围内相等，不考虑设备上的精确匹配
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        # 断言 conv_cpu 和 conv_cuda 的权重梯度数据在误差范围内相等，不考虑设备上的精确匹配
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
    # 定义一个测试方法，用于测试 ConvTranspose3d 类的行为，使用指定的设备
    def test_ConvTranspose3d_size_1_kernel(self, device):
        # 设置默认的张量数据类型为双精度浮点数
        with set_default_dtype(torch.double):
            # 生成一个大小为 (2, 3, 3, 5, 5) 的随机张量 x_cpu
            x_cpu = torch.randn(2, 3, 3, 5, 5)
            # 创建一个 ConvTranspose3d 层，输入通道数和输出通道数均为 3，卷积核大小为 1
            conv_cpu = torch.nn.ConvTranspose3d(3, 3, kernel_size=1)
            # 对 x_cpu 进行 ConvTranspose3d 操作，得到 y_cpu
            y_cpu = conv_cpu(x_cpu)
            # 生成一个与 y_cpu 同样大小的随机张量 y
            y = torch.rand_like(y_cpu)
            # 对 y_cpu 进行反向传播
            y_cpu.backward(y)

            # 禁用 cudnn 加速的情况下，创建一个在指定设备上的 ConvTranspose3d 层 conv_cuda
            with cudnn.flags(enabled=False):
                conv_cuda = torch.nn.ConvTranspose3d(3, 3, kernel_size=1).to(device)
                # 将 conv_cpu 的偏置数据复制到 conv_cuda 的偏置数据
                conv_cuda.bias.data.copy_(conv_cpu.bias.data)
                # 将 conv_cpu 的权重数据复制到 conv_cuda 的权重数据
                conv_cuda.weight.data.copy_(conv_cpu.weight.data)
                # 对 x_cpu 在设备上执行 ConvTranspose3d 操作，得到 y_cuda
                y_cuda = conv_cuda(x_cpu.to(device))
                # 对 y_cuda 在设备上进行反向传播
                y_cuda.backward(y.to(device))

            # 断言 y_cpu 与 y_cuda 相等，使用绝对容差 1e-5，相对容差 0，设备匹配不需要精确
            self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
            # 断言 conv_cpu 的偏置梯度与 conv_cuda 的偏置梯度相等，使用绝对容差 1e-5，相对容差 0，设备匹配不需要精确
            self.assertEqual(
                conv_cpu.bias.grad.data,
                conv_cuda.bias.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )
            # 断言 conv_cpu 的权重梯度与 conv_cuda 的权重梯度相等，使用绝对容差 1e-5，相对容差 0，设备匹配不需要精确
            self.assertEqual(
                conv_cpu.weight.grad.data,
                conv_cuda.weight.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )

    # 对于 CUDA 的情况，根据条件选择浮点类型，包括半精度和 bfloat16（如果是 AMPERE_OR_ROCM）
    @dtypesIfCUDA(
        *floating_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else [])
    )
    # 限定张量数据类型为单精度浮点数
    @dtypes(torch.float)
    # 启用 cuDNN，并禁用基准测试
    @torch.backends.cudnn.flags(enabled=True, benchmark=False)
    # 如果在 ROCm 上测试，跳过此测试
    @unittest.skipIf(TEST_WITH_ROCM, "Skipped on ROCm, since it is failing on ROCm 5.7")
    # 测试带有分组卷积的情况，确保分组卷积与两个半卷积相匹配
    def test_Conv2d_naive_groups(self, device, dtype):
        # 创建一个分组卷积层，输入通道数为4，输出通道数为4，卷积核大小为3，分组数为2，并转移到指定设备和数据类型上
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        # 创建一个随机输入张量
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        # 对输入张量进行分组卷积操作
        output = m(i)
        # 创建一个随机梯度张量
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        # 对输出进行反向传播
        output.backward(grad_output)

        # 创建第一个半卷积层，输入通道数为2，输出通道数为2，卷积核大小为3，并转移到指定设备和数据类型上
        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        # 复制分组卷积层的前两个通道的权重和偏置
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        # 选择输入张量的前两个通道并确保其连续，需要计算梯度
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        # 对第一个半卷积层进行前向传播
        output1 = m1(i1)
        # 对第一个半卷积层的输出进行反向传播
        output1.backward(grad_output[:, :2].contiguous())

        # 创建第二个半卷积层，输入通道数为2，输出通道数为2，卷积核大小为3，并转移到指定设备和数据类型上
        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        # 复制分组卷积层的后两个通道的权重和偏置
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        # 选择输入张量的后两个通道并确保其连续，需要计算梯度
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        # 对第二个半卷积层进行前向传播
        output2 = m2(i2)
        # 对第二个半卷积层的输出进行反向传播
        output2.backward(grad_output[:, 2:].contiguous())

        # 检查分组卷积的整体输出是否与两个半卷积的拼接输出相等
        self.assertEqual(output, torch.cat([output1, output2], 1))
        # 检查输入张量的梯度是否与两个半卷积输入张量的梯度拼接后相等
        self.assertEqual(
            i.grad.data,
            torch.cat([i1.grad.data, i2.grad.data], 1),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        # 检查分组卷积层的偏置梯度是否与两个半卷积层的偏置梯度拼接后相等
        self.assertEqual(
            m.bias.grad.data,
            torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        # 检查分组卷积层的权重梯度是否与两个半卷积层的权重梯度拼接后相等
        self.assertEqual(
            m.weight.grad.data,
            torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )

    # 使用指定设备和数据类型对深度卷积的反向传播进行测试
    @dtypes(torch.double, torch.cdouble)
    def test_Conv2d_backward_depthwise(self, device, dtype):
        # 创建一个随机输入张量，形状为[2, 2, 4, 20]，并需要计算梯度
        x = torch.randn(2, 2, 4, 20, device=device, dtype=dtype, requires_grad=True)
        # 创建一个随机权重张量，形状为[2, 1, 3, 5]，并需要计算梯度
        weight = torch.randn(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)

        # 定义深度卷积函数，使用torch.nn.functional.conv2d进行计算
        def conv2d_depthwise(x, weight):
            return torch.nn.functional.conv2d(
                x, weight, bias=None, stride=(1, 10), groups=2
            )

        # 针对不同的cudnn启用状态进行梯度检查
        for cudnn_enabled in [False, True]:
            with torch.backends.cudnn.flags(enabled=cudnn_enabled):
                # 使用torch.autograd.gradcheck进行梯度检查
                torch.autograd.gradcheck(conv2d_depthwise, (x, weight))

    # 只在CPU上运行的测试函数修饰器，指定数据类型为torch.float和torch.double
    @onlyCPU
    @dtypes(torch.float, torch.double)
    # 只在CUDA上运行的测试函数修饰器
    @onlyCUDA
    # 如果ROCm版本低于(4, 3)，则跳过在CUDA上运行测试
    @skipCUDAIfRocmVersionLessThan((4, 3))
    # 如果不是使用miopen建议的NHWC格式，则跳过在CUDA上运行测试
    @skipCUDAIfNotMiopenSuggestNHWC
    # 如果CuDNN版本低于7603，则跳过在CUDA上运行测试
    @skipCUDAIfCudnnVersionLessThan(7603)
    # 测试数据类型为torch.half、torch.float和torch.cfloat的情况
    @dtypes(torch.half, torch.float, torch.cfloat)
    # 定义一个测试函数，用于测试使用 cuDNN 和 NHWC 格式的卷积操作
    def test_conv_cudnn_nhwc(self, device, dtype):
        
        # 定义一个内部辅助函数，用于执行单个卷积操作的测试
        def helper(n, c, h, w, out_channels, kernel_size, groups):
            # 生成随机整数张量作为输入，使用 channels_last 内存格式
            input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device).to(
                memory_format=torch.channels_last
            )
            input.requires_grad_()
            
            # 创建一个 Conv2d 模块，指定输出通道数、卷积核大小和分组数，使用 channels_last 内存格式
            conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups).to(
                device="cuda", dtype=dtype, memory_format=torch.channels_last
            )
            # 初始化卷积层参数为随机整数
            for p in conv.parameters():
                p.data = torch.randint_like(p, -3, 3)

            # 使用 channels_first 内存格式创建 FP64 类型的参考输入
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            # 创建一个与 conv 具有相同参数的标准 Conv2d 模块
            ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
            # 从 conv 的状态字典中加载参数到 ref_conv，以恢复 stride 和 memory_layout
            ref_conv.load_state_dict(conv.state_dict())
            # 将 ref_conv 转移到 CUDA 设备上，使用双精度浮点数数据类型和 contiguous 内存格式
            ref_conv = ref_conv.to(
                device="cuda", dtype=torch.double, memory_format=torch.contiguous_format
            )

            # 执行卷积操作和参考卷积操作
            out = conv(input)
            ref_out = ref_conv(ref_input)

            # 生成随机整数梯度作为输出的反向传播梯度
            grad = torch.randint_like(out, -3, 3)
            # 创建 ref_out 的双精度浮点数类型和 contiguous 内存格式的梯度
            ref_grad = grad.detach().clone().double().contiguous()

            # 对 out 和 ref_out 执行反向传播
            out.backward(grad)
            ref_out.backward(ref_grad)

            # 断言输出
    # 定义测试函数，用于测试使用 CuDNN 加速的 NDHWC 格式的卷积
    def test_conv_cudnn_ndhwc(self, device, dtype):
        # 定义内部辅助函数，用于执行单个卷积测试
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            # 创建输入张量，填充随机整数值，使用 channels_last_3d 内存格式
            input = torch.randint(
                -2, 2, (n, c, d, h, w), dtype=dtype, device=device
            ).to(memory_format=torch.channels_last_3d)
            input.requires_grad_()  # 设置输入张量需要计算梯度

            # 创建卷积层，参数随机填充整数值，使用 channels_last_3d 内存格式
            conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups).to(
                device="cuda", dtype=dtype, memory_format=torch.channels_last_3d
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            # 创建参考的 channels-first 卷积层，保持相同状态
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            ref_conv.load_state_dict(conv.state_dict())  # 加载状态以恢复权重和内存布局
            ref_conv = ref_conv.to(
                device="cuda", dtype=torch.double, memory_format=torch.contiguous_format
            )

            # 进行正向传播
            out = conv(input)
            ref_out = ref_conv(ref_input)

            # 创建梯度张量，填充随机整数值
            grad = torch.randint_like(out, -2, 2)
            ref_grad = grad.detach().clone().double().contiguous()

            # 反向传播计算梯度
            out.backward(grad)
            ref_out.backward(ref_grad)

            # 断言保证输出和梯度张量的内存布局符合 channels_last_3d
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(
                input.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )

            # 断言保证参考通道顺序卷积的输出和梯度张量是连续的
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            # 断言比较输出、权重梯度、偏置梯度和输入梯度的数值，允许浮点误差
            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        # 执行多个不同参数的卷积测试
        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    # 定义运行卷积的内部函数，用于执行卷积层的前向和反向传播
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
        ):
            # 根据输入和参考卷积层的尺寸创建新的卷积层对象，并转换为浮点数类型并移动到指定设备上
            conv = (
                layer(inp.size(1), grad.size(1), ref_conv.weight.size(2)).float().to(device)
            )
            # 使用参考卷积层的状态来加载权重和偏置到新的卷积层对象
            conv.load_state_dict(ref_conv.state_dict())
            # 对卷积层权重进行深拷贝，并保持内存格式为指定格式
            weight_data = (
                conv.weight.detach().clone().contiguous(memory_format=weight_format)
            )
            # 调整卷积层权重的大小，并保持内存格式为指定格式
            conv.weight.data = weight_data.resize_(
                weight_data.size(), memory_format=weight_format
            )
            # 对输入数据进行深拷贝，并保持内存格式为指定格式
            input = inp.clone().contiguous(memory_format=input_format)
            # 调整输入数据的大小，并保持内存格式为指定格式
            input.resize_(input.size(), memory_format=input_format)
            # 设置输入数据需要计算梯度
            input = input.requires_grad_()
            # 对梯度数据进行深拷贝，并保持内存格式为指定格式
            grad = grad.contiguous(memory_format=grad_format)
            # 调整梯度数据的大小，并保持内存格式为指定格式
            grad.resize_(grad.size(), memory_format=grad_format)
            # 使用卷积层进行前向传播计算输出
            out = conv(input)
            # 对输出进行反向传播计算梯度
            out.backward(grad)
            # 断言输出张量的内存格式为指定格式
            self.assertTrue(out.is_contiguous(memory_format=output_format))
            # 断言输出与参考输出相等
            self.assertEqual(out, ref_out)
            # 断言卷积层权重的梯度与参考卷积层权重的梯度相等
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad)
            # 断言卷积层偏置的梯度与参考卷积层偏置的梯度相等
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad)
            # 断言输入数据的梯度与参考输入数据的梯度相等
            self.assertEqual(input.grad, ref_input.grad)
    # 定义一个测试函数，用于验证不同卷积层的不同内存格式在 CUDA 环境下的转换
    def _test_conv_cudnn_nhwc_nchw(self, layer, n, c, h, w, k, filter_size, device):
        # 创建随机整数张量作为输入数据，形状为 (n, c, h, w)，数据类型为 float32，设备为指定的 device
        data = torch.randint(1, 10, (n, c, h, w), dtype=torch.float32, device=device)
        # 创建参考输入张量，是输入数据的克隆，要求其是连续的，并且需要梯度追踪
        ref_input = data.clone().contiguous().requires_grad_(True)
        # 创建参考卷积层对象，参数为输入通道数 c、输出通道数 k 和滤波器大小 filter_size，类型为 float，放置在指定的 device 上
        ref_conv = layer(c, k, filter_size).float().to(device)
        # 将参考输入数据输入到参考卷积层中，得到输出结果 ref_out
        ref_out = ref_conv(ref_input)
        # 创建随机整数张量作为梯度，形状与 ref_out 相同，数据类型为 float32，设备为 "cuda"
        grad = torch.randint(1, 10, ref_out.size(), dtype=torch.float32, device="cuda")
        # 对 ref_out 进行反向传播，梯度为 grad
        ref_out.backward(grad)

        # 遍历指定的内存格式转换函数和输入格式
        for w_f in [torch.contiguous_format, torch.channels_last]:
            for g_f in [torch.contiguous_format, torch.channels_last]:
                for input_format in [torch.contiguous_format, torch.channels_last]:
                    output_format = torch.contiguous_format
                    # 对于旧版本的 CudNN，可能不支持 Channels Last 格式
                    if torch.backends.cudnn.version() >= 7603:
                        if input_format == torch.channels_last:
                            output_format = torch.channels_last
                        # 这是因为我们有一个 N111 权重，不能处理模糊的内存格式
                        if w_f == torch.channels_last:
                            if layer == nn.Conv2d and filter_size * c != 1:
                                output_format = torch.channels_last
                            if layer == nn.ConvTranspose2d and filter_size * k != 1:
                                output_format = torch.channels_last
                    # 调用内部函数 _run_conv，传递参数为：卷积层类型 layer、设备 device、数据 data、梯度 grad、参考卷积层 ref_conv、参考输入 ref_input、参考输出 ref_out、输入格式 input_format、权重格式 w_f、梯度格式 g_f、输出格式 output_format
                    self._run_conv(
                        layer,
                        device,
                        data,
                        grad,
                        ref_conv,
                        ref_input,
                        ref_out,
                        input_format,
                        w_f,
                        g_f,
                        output_format,
                    )

    # 在 CUDA 环境下测试卷积层内存格式不匹配的情况
    @onlyCUDA
    @skipCUDAIfRocmVersionLessThan((4, 3))
    @skipCUDAIfNotMiopenSuggestNHWC
    @skipCUDAIfCudnnVersionLessThan(7603)
    @tf32_on_and_off(0.05)
    def test_conv_cudnn_mismatch_memory_format(self, device):
        # 定义多个测试配置，每个配置包含 n、c、h、w、k、filter_size 参数
        configs = [
            [4, 2, 8, 8, 4, 2],
            [4, 1, 8, 8, 4, 2],
            [1, 1, 8, 8, 4, 2],
            [4, 2, 2, 8, 4, 1],
            [4, 2, 1, 8, 4, 1],
            [4, 2, 8, 8, 4, 1],
            [4, 1, 8, 8, 4, 1],
        ]
        # 遍历每个配置进行测试
        for n, c, h, w, k, filter_size in configs:
            # 分别对 nn.Conv2d 和 nn.ConvTranspose2d 两种卷积层进行内存格式不匹配的测试
            self._test_conv_cudnn_nhwc_nchw(
                nn.Conv2d, n, c, h, w, k, filter_size, device
            )
            self._test_conv_cudnn_nhwc_nchw(
                nn.ConvTranspose2d, n, c, h, w, k, filter_size, device
            )

    # 在 CUDA 环境下，如果没有 CUDNN，则跳过特定的测试
    # 指定数据类型为 torch.float 或 torch.double
    @onlyCUDA
    @skipCUDAIfNoCudnn
    @dtypes(torch.float, torch.double)
    # 测试函数，验证CUDA环境下的cuDNN NHWC支持情况
    def test_conv_cudnn_nhwc_support(self, device, dtype):
        # 创建一个形状为(1, 16, 1, 1)的随机张量作为输入，位于CUDA设备上，需要梯度计算
        input = torch.randn((1, 16, 1, 1), dtype=dtype, device="cuda", requires_grad=True)
        # 创建一个形状为(8, 16, 3, 3)的随机张量作为卷积核，位于CUDA设备上，需要梯度计算
        weight = torch.randn((8, 16, 3, 3), dtype=dtype, device="cuda", requires_grad=True)
        # 将卷积核转换为通道最后的内存格式
        weight = weight.to(memory_format=torch.channels_last)
        # 进行二维卷积操作，步长为(1, 1)，填充为(1, 1)，dilation为(1, 1)，组数为1
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        # 断言输出张量采用通道最后的内存格式
        self.assertTrue(o.is_contiguous(memory_format=torch.channels_last))
        # 对输出张量进行求和并反向传播梯度
        o.sum().backward()

    # 测试函数，验证推理阶段使用更快算法是否产生相同结果
    # 验证在 https://github.com/pytorch/pytorch/issues/60176 中报告的depthwise3x3 bug
    @onlyCPU
    @dtypes(torch.float)
    def test_conv2d_no_grad(self, device, dtype):
        # 遍历不同的批次数和分组数
        for batch in [1, 2, 3]:
            for groups in [1, 2, 4]:
                # 创建一个形状为(batch, groups, 8, 8)的随机张量作为输入，位于指定设备上
                input = torch.rand(batch, groups, 8, 8, dtype=dtype, device=device)
                # 创建一个二维卷积层，设置卷积核大小为(3, 3)，组数为groups
                m = nn.Conv2d(
                    groups,
                    8,
                    kernel_size=(3, 3),
                    groups=groups,
                    dtype=dtype,
                    device=device,
                )
                # 使用torch.no_grad()上下文管理器计算输出张量（无梯度）
                with torch.no_grad():
                    output_ng = m(input)
                # 计算输出张量（有梯度）
                output = m(input)
                # 断言两个输出张量相等，设置相对误差为1e-2，绝对误差为1e-5
                self.assertEqual(output, output_ng, rtol=1e-2, atol=1e-5)

    # 测试函数，验证CUDA环境下的cuDNN卷积ReLU操作
    @onlyCUDA
    @skipCUDAIfNoCudnn
    @dtypes(torch.float, torch.float16)
    @precisionOverride({torch.half: 0.002, torch.float: 1e-4})
    def test_cudnn_convolution_relu(self, device, dtype):
        # 使用product()迭代不同的批次、分组、图像大小、卷积核大小和内存格式组合
        for batch, groups, image_size, kernel_size, memory_format in product(
            (1, 2, 3),
            (1, 2, 4),
            ((1, 1), (8, 8)),
            ((1, 1), (3, 3)),
            (torch.channels_last, torch.contiguous_format),
        ):
            if image_size[0] < kernel_size[0]:
                continue
            # 创建一个形状为(batch, groups, *image_size)的随机张量作为输入，位于指定设备上
            inp = torch.rand(batch, groups, *image_size, dtype=dtype, device=device)
            # 创建一个形状为(8, groups, *kernel_size)的随机张量作为卷积核，位于指定设备上
            w = torch.randn(8, groups, *kernel_size, dtype=dtype, device=device)
            # 使用torch.conv2d进行卷积操作
            conv2d_out = torch.conv2d(inp, w, None, (1, 1), (0, 0), (1, 1), 1)
            # 将输入张量和卷积核转换为指定内存格式
            inp = inp.to(memory_format=memory_format)
            w = w.to(memory_format=memory_format)
            # 根据是否支持HIP（AMD GPU加速平台），选择不同的cuDNN卷积ReLU实现
            if torch.version.hip:
                cudnn_out = torch.miopen_convolution_relu(
                    inp, w, None, (1, 1), (0, 0), (1, 1), 1
                )
            else:
                cudnn_out = torch.cudnn_convolution_relu(
                    inp, w, None, (1, 1), (0, 0), (1, 1), 1
                )
            # 断言输出张量采用指定的内存格式
            self.assertTrue(cudnn_out.is_contiguous(memory_format=memory_format))
            # 如果TF32不等于FP32，并且dtype为torch.float，则设置绝对误差为4e-3，相对误差为0.006
            if tf32_is_not_fp32() and dtype == torch.float:
                self.assertEqual(conv2d_out.relu(), cudnn_out, atol=4e-3, rtol=0.006)
            else:
                # 否则直接比较ReLU后的输出张量
                self.assertEqual(conv2d_out.relu(), cudnn_out)

    # 测试函数，验证CUDA环境下的cuDNN卷积ReLU操作（仅限于CUDA环境）
    @onlyCUDA
    @skipCUDAIfNoCudnn
    @dtypes(torch.float, torch.float16)
    # 定义修饰器，覆盖精度要求，针对不同数据类型设置不同的数值精度
    @precisionOverride({torch.half: 0.002, torch.float: 1e-4})
    # 测试 CUDA 下的 CUDNN 卷积加法激活函数操作
    def test_cudnn_convolution_add_relu(self, device, dtype):
        # 使用 product 函数生成多种参数组合的迭代器
        for batch, groups, image_size, kernel_size, memory_format in product(
            (1, 2, 3),                                          # 批次大小
            (1, 2, 4),                                          # 分组数
            ((1, 1), (8, 8)),                                   # 图像大小
            ((1, 1), (3, 3)),                                   # 卷积核大小
            (torch.channels_last, torch.contiguous_format),     # 内存格式
        ):
            # 若图像尺寸小于卷积核尺寸，则跳过当前迭代
            if image_size[0] < kernel_size[0]:
                continue
            # 创建随机输入张量和权重张量
            inp = torch.rand(batch, groups, *image_size, dtype=dtype, device=device)
            w = torch.randn(8, groups, *kernel_size, dtype=dtype, device=device)
            # 使用 torch.conv2d 进行卷积操作
            conv2d_out = torch.conv2d(inp, w, None, (1, 1), (0, 0), (1, 1), 1)
            alpha = 2.0
            # 创建与 conv2d_out 相同大小的随机张量 z
            z = torch.randn_like(conv2d_out)

            # 将输入、权重和 z 张量转换为指定内存格式
            inp = inp.to(memory_format=memory_format)
            w = w.to(memory_format=memory_format)
            z = z.to(memory_format=memory_format)
            # 根据当前环境调用不同的 CUDNN 卷积加法激活函数操作
            if torch.version.hip:
                cudnn_out = torch.miopen_convolution_add_relu(
                    inp, w, z, alpha, None, (1, 1), (0, 0), (1, 1), 1
                )
            else:
                cudnn_out = torch.cudnn_convolution_add_relu(
                    inp, w, z, alpha, None, (1, 1), (0, 0), (1, 1), 1
                )

            # 断言输出张量 cudnn_out 是否按指定内存格式连续
            self.assertTrue(cudnn_out.is_contiguous(memory_format=memory_format))
            # 若当前环境为 TF32 且数据类型为 float，则使用指定的误差范围进行比较
            if tf32_is_not_fp32() and dtype == torch.float:
                self.assertEqual(
                    F.relu(conv2d_out + alpha * z), cudnn_out, atol=2e-3, rtol=0.006
                )
            else:
                self.assertEqual(F.relu(conv2d_out + alpha * z), cudnn_out)

    # 修饰器，仅在 CUDA 环境下运行
    @onlyCUDA
    # 如果是 ROCm 环境，跳过当前测试
    @skipCUDAIfRocm
    # 如果 CUDNN 版本小于 7603，跳过当前测试
    @skipCUDAIfCudnnVersionLessThan(7603)
    # 测试转换 Conv2d 权重的内存格式
    def test_convert_conv2d_weight_memory_format(self, device):
        # 创建随机输入张量
        input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device=device)
        # 创建包含 Conv2d 和 BatchNorm2d 的模型，将其转移到指定设备并设置为 float 类型
        model = nn.Sequential(nn.Conv2d(8, 4, 3), nn.BatchNorm2d(4)).to(device).float()
        # 遍历不同的内存格式
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            # 转换模型的 Conv2d 权重为指定内存格式
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            # 使用转换后的模型进行前向计算
            out = model(input)
            # 断言输出张量 out 是否按指定内存格式连续
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

        # 创建包含 ConvTranspose2d 和 BatchNorm2d 的模型，将其转移到指定设备并设置为 float 类型
        model = (
            nn.Sequential(nn.ConvTranspose2d(8, 4, 3), nn.BatchNorm2d(4))
            .to(device)
            .float()
        )
        # 遍历不同的内存格式
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            # 转换模型的 Conv2d 权重为指定内存格式
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            # 使用转换后的模型进行前向计算
            out = model(input)
            # 断言输出张量 out 是否按指定内存格式连续
            self.assertTrue(out.is_contiguous(memory_format=memory_format))
    # 测试函数：test_convert_conv3d_weight_memory_format，用于测试转换卷积层权重的内存格式
    def test_convert_conv3d_weight_memory_format(self, device):
        # 创建指定设备上的随机整数张量作为输入
        input = torch.randint(
            1, 10, (2, 8, 4, 4, 4), dtype=torch.float32, device=device
        )
        # 创建包含转置卷积层和三维批归一化的模型
        model = (
            nn.Sequential(nn.ConvTranspose3d(8, 4, 3), nn.BatchNorm3d(4))
            .to(device)
            .float()
        )
        # 遍历内存格式选项：通道末尾三维和连续格式
        for memory_format in [torch.channels_last_3d, torch.contiguous_format]:
            # 将模型权重转换为指定的内存格式
            model = nn.utils.convert_conv3d_weight_memory_format(model, memory_format)
            # 使用输入数据计算模型输出
            out = model(input)
            # 断言输出在指定的内存格式下是连续的
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

    # 测试函数：test_conv_double_backward_strided_with_3D_input_and_weight，验证双向卷积梯度的形状
    def test_conv_double_backward_strided_with_3D_input_and_weight(self, device):
        # 创建随机输入张量和权重张量
        input = torch.randn(2, 3, 6, device=device)
        weight = torch.randn(3, 3, 3, device=device)
        bias = torch.randn(3, device=device)
        stride = (2,)
        padding = (1,)
        dilation = (1,)
        transposed = False
        output_padding = (0,)
        groups = 1
        # 执行底层ATen操作，计算卷积输出
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

        # 创建与输入形状相同的随机梯度张量
        ggI = torch.randn(input.shape, device=device)
        ggW = torch.randn(weight.shape, device=device)
        ggB = torch.randn(bias.shape, device=device)
        gO = torch.randn(output.shape, device=device)
        output_mask = [True, True, True]
        # 调用底层ATen函数，计算双向卷积梯度
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

        # 确保计算出的梯度形状正确
        self.assertEqual(grad_grad_output.shape, gO.shape)
        self.assertEqual(grad_input.shape, input.shape)
        self.assertEqual(grad_weight.shape, weight.shape)

    # 测试函数：test_conv3d_64bit_indexing，测试64位索引的三维卷积
    @onlyCUDA
    @largeTensorTest("40GB")
    @largeTensorTest("24GB", "cpu")
    def test_conv3d_64bit_indexing(self, device):
        # 创建形状为(1, 32, 512, 512, 256)的随机张量
        x = torch.rand(1, 32, 512, 512, 256)
        # 创建具有指定参数的三维卷积层，不包含偏置
        m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1, bias=False)
        # 计算在参考设备上的卷积输出
        yref = m(x)
        # 将模型转移到指定设备并计算输出
        y = m.to(device=device)(x.to(device=device))
        # 断言计算结果是否一致
        self.assertEqual(yref, y)
# 实例化设备类型测试，使用 TestConvolutionNNDeviceType 类和全局命名空间
instantiate_device_type_tests(TestConvolutionNNDeviceType, globals())

# 实例化参数化测试，使用 TestConvolutionNN 类
instantiate_parametrized_tests(TestConvolutionNN)

# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则运行测试
if __name__ == "__main__":
    run_tests()
```