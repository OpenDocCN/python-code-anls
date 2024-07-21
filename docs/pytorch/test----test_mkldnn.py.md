# `.\pytorch\test\test_mkldnn.py`

```py
# Owner(s): ["module: mkldnn"]

# 导入必要的模块和函数
import copy  # 导入深拷贝函数
import itertools  # 导入迭代器模块
import functools  # 导入函数工具模块
import unittest  # 导入单元测试模块
from contextlib import nullcontext  # 导入上下文管理模块中的空上下文

try:
    import torchvision  # 尝试导入torchvision模块
    HAS_TORCHVISION = True  # 设置标志，表示成功导入torchvision模块
except ImportError:
    HAS_TORCHVISION = False  # 设置标志，表示未能导入torchvision模块

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")  # 创建一个条件装饰器，如果没有torchvision，则跳过测试

# 导入torch相关模块
import torch
import torch.nn.functional as F
import torch.jit
import torch.backends.mkldnn
from torch.utils import mkldnn as mkldnn_utils
from torch.testing._internal.common_utils import TestCase, \
    run_tests, TemporaryFileName, gradcheck, gradgradcheck, IS_WINDOWS, \
    skipIfTorchDynamo, xfailIfTorchDynamo
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)

# 禁用批次梯度检查功能
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
gradgradcheck = functools.partial(gradgradcheck, check_batched_grad=False)

# 支持的数据类型列表
types = [torch.float, torch.bfloat16, torch.half]

# 根据是否支持MKL-DNN构建选择是否跳过测试
@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
class TestMkldnn(TestCase):
    def test_conversion_byte_char(self):
        int8_types = [torch.int8, torch.uint8]
        for int8_type in int8_types:
            low = -100 if int8_type is torch.int8 else 0
            high = 100
            for cpu_tensor in [torch.randint(
                               low=low,
                               high=high,
                               size=(1, 2, 3, 4),
                               dtype=torch.int64,
                               device=torch.device('cpu')),
                               torch.randint(
                               low=low,
                               high=high,
                               size=(1, 2, 3, 4, 5),
                               dtype=torch.int64,
                               device=torch.device('cpu'))[:, :, :, :, :]]:

                cpu_tensor = cpu_tensor.to(dtype=int8_type)  # 转换CPU上的张量到指定数据类型
                mkldnn_tensor = cpu_tensor.to_mkldnn(int8_type)  # 将CPU上的张量转换为MKL-DNN张量
                self.assertEqual(mkldnn_tensor.dtype, int8_type)  # 断言MKL-DNN张量的数据类型与指定的一致
                cpu_tensor_1 = mkldnn_tensor.to_dense()  # 将稀疏的MKL-DNN张量转换为密集张量
                self.assertEqual(mkldnn_tensor.dtype, cpu_tensor_1.dtype)  # 断言转换后的MKL-DNN张量的数据类型与原始的一致
                self.assertEqual(cpu_tensor, cpu_tensor_1)  # 断言稀疏到密集的转换没有改变张量的值
                self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))  # 断言MKL-DNN张量位于CPU设备上
                self.assertEqual(mkldnn_tensor.size(), cpu_tensor.size())  # 断言MKL-DNN张量的尺寸与原始张量的一致
                self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())  # 断言MKL-DNN张量的元素数与原始张量的一致
                self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size())  # 断言MKL-DNN张量的元素大小与原始张量的一致
                self.assertRaisesRegex(RuntimeError,
                                       "Cannot access data pointer of Tensor that doesn't have storage",
                                       lambda: mkldnn_tensor.data_ptr() != 0)  # 断言尝试访问没有存储的张量的数据指针会引发运行时错误
    # 定义一个名为 test_copy 的测试方法，用于测试 MKLDNN 张量的复制操作
    def test_copy(self):
        # 创建一个大小为 4x5 的随机张量 x，数据类型为 float32
        x = torch.randn(4, 5, dtype=torch.float32)
        # 将 x 转换为 MKLDNN 格式的张量 mkldnn_x
        mkldnn_x = x.to_mkldnn()
        # 创建一个大小为 4x5 的随机张量，并将其转换为 MKLDNN 格式，赋给 mkldnn_y
        mkldnn_y = torch.randn(4, 5, dtype=torch.float32).to_mkldnn()
        # 创建一个大小为 4x10 的随机张量，并将其转换为 MKLDNN 格式，赋给 mkldnn_z
        mkldnn_z = torch.randn(4, 10, dtype=torch.float32).to_mkldnn()
        # 将 mkldnn_x 的数据复制到 mkldnn_y
        mkldnn_y.copy_(mkldnn_x)
        # 断言 x 和 mkldnn_y 的内容相等
        self.assertEqual(x, mkldnn_y.to_dense())
        # 使用 lambda 函数捕获 RuntimeError 异常，检查 mkldnn_z 复制 mkldnn_x 时是否抛出指定错误信息
        self.assertRaisesRegex(RuntimeError,
                               "copy_mkldnn_: only support same size tensor.",
                               lambda: mkldnn_z.copy_(mkldnn_x))
        # 使用 lambda 函数捕获 RuntimeError 异常，检查 x 复制 mkldnn_x 时是否抛出指定错误信息
        self.assertRaisesRegex(RuntimeError,
                               "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! "
                               "Found self type = torch.FloatTensor and src type = Mkldnntorch.FloatTensor",
                               lambda: x.copy_(mkldnn_x))
        # 使用 lambda 函数捕获 RuntimeError 异常，检查 mkldnn_x 复制 x 时是否抛出指定错误信息
        self.assertRaisesRegex(RuntimeError,
                               "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! "
                               "Found self type = Mkldnntorch.FloatTensor and src type = torch.FloatTensor",
                               lambda: mkldnn_x.copy_(x))

    # 定义一个名为 test_unsupported 的测试方法，用于测试不支持的数据类型转换到 MKLDNN 格式的情况
    def test_unsupported(self):
        # 循环遍历不支持的数据类型列表，使用 lambda 函数捕获 RuntimeError 异常
        for dtype in [torch.double, torch.uint8, torch.int8,
                      torch.short, torch.int, torch.long]:
            with self.assertRaises(RuntimeError) as context:
                # 创建指定数据类型和设备的随机张量，并尝试将其转换为 MKLDNN 格式
                torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cpu')).to_mkldnn()
            # 如果 CUDA 可用，再次尝试在 GPU 设备上转换相同数据类型的张量到 MKLDNN 格式
            if torch.cuda.is_available():
                with self.assertRaises(RuntimeError) as context:
                    torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cuda')).to_mkldnn()
        # 如果 CUDA 可用，尝试创建支持的数据类型在 GPU 设备上的随机张量，并尝试将其转换为 MKLDNN 格式
        if torch.cuda.is_available():
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cuda')).to_mkldnn()
        # 循环遍历张量工厂函数列表，尝试使用不支持 MKLDNN 转换的工厂函数创建张量
        for creator in [torch.ones, torch.randn, torch.rand]:
            with self.assertRaises(RuntimeError) as context:
                creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch._mkldnn)
    def test_mkldnn_conv_shapecheck(self):
        input = torch.full((1, 1, 1, 24,), 1, dtype=torch.float32)
        w1 = torch.full((1, 1, 1, 24,), 1, dtype=torch.float32)
        b1 = torch.full((1,), 1, dtype=torch.float32)
        w2 = torch.full((1, 1, 2, 24,), 1, dtype=torch.float32)
        b2 = torch.full((2,), 1, dtype=torch.float32)
        options = zip([-1, 0, 0, 0, 0, 0, 0],  # padding
                      [1, 0, 1, 1, 1, 1, 1],  # stride
                      [1, 1, 0, 1, 1, 1, 1],  # dilation
                      [1, 1, 1, 0, 2, 1, 1],  # groups
                      [w1, w1, w1, w1, w1, w1, w2],  # weight
                      [b1, b1, b1, b1, b1, b2, b1])  # bias
        for pad, st, dil, gr, w, b in options:
            # 在运行中断言语句，确保抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError) as _:
                # 调用 torch 的 MKLDNN 卷积函数进行形状检查
                torch.mkldnn_convolution(input, w, b, [pad] * 2, [st] * 2, [dil] * 2, gr)

    def test_autograd_to_mkldnn(self):
        # MKLDNN 只支持 float32 类型的张量
        root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        def func(root):
            # 将张量转换为 MKLDNN 格式，再转换回稠密张量
            return root.to_mkldnn().to_dense()

        # 因为 MKLDNN 只支持 float32 类型，所以需要降低精度。
        # 这些数值仅是经验结果，似乎有效。
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2))
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2))

    def test_autograd_from_mkldnn(self):
        # MKLDNN 只支持 float32 类型的张量
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        def func(root):
            # 将 MKLDNN 格式的张量转换为稠密张量
            return root.to_dense()

        # 因为 MKLDNN 只支持 float32 类型，所以需要降低精度。
        # 这些数值仅是经验结果，似乎有效。
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2))

    def test_detach(self):
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        # 对张量进行 detach 操作，不再追踪梯度
        detach = root.detach()
        self.assertEqual((4, 5), detach.size())
        self.assertFalse(detach.requires_grad)
        self.assertTrue(root.requires_grad)

        # 原地操作：对张量进行 detach 操作，不再追踪梯度
        detach_ = root.detach_()
        self.assertEqual((4, 5), detach_.size())
        self.assertFalse(detach_.requires_grad)
        self.assertFalse(root.requires_grad)

    def test_repr(self):
        # 测试张量的字符串表示是否包含 'layout=torch._mkldnn'
        self.assertTrue("layout=torch._mkldnn" in str(torch.randn((1, 2, 3, 4),
                                                                  dtype=torch.float, device=torch.device('cpu')).to_mkldnn()))
    # 定义一个用于测试卷积模块的方法，根据给定的维度选择相应的卷积类
    def _test_conv_base(self, dim):
        # 根据维度选择对应的卷积模块类
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        # 定义不同维度下的输入形状
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        # 枚举所有可能的选项组合
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        # 遍历每个选项组合
        for train, bias, dilation, groups in options:
            # 随机生成 N、M、C 的值
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            # 根据当前维度构造输入张量的形状
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)
            # 创建卷积层对象
            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).float()
            # 克隆输入张量 x 以备后用
            x1 = x.clone()
            x2 = x.clone().to_mkldnn()
            # 根据训练状态和维度选择相应的处理方式
            if not train:
                # 若不进行训练，则将普通卷积转为 MKL-DNN 卷积
                mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
            elif train and dim != 1:
                # 若进行训练且维度不为 1，则允许 conv1d 的训练
                x1.requires_grad_()
                x2.requires_grad_()
                mkldnn_conv = copy.deepcopy(conv)
            # 在 MKL-DNN 后端未启用的情况下，计算 Aten 卷积的输出 y_aten
            with torch.backends.mkldnn.flags(enabled=False):
                y_aten = conv(x1)
                # 若进行训练且维度不为 1，则计算 loss1 并反向传播
                if train and dim != 1:
                    loss1 = y_aten.sum()
                    loss1.backward()
            # 若不进行训练或者进行训练但维度不为 1，则计算 MKL-DNN 卷积的输出 y_mkldnn
            if not train or (train and dim != 1):
                y_mkldnn = mkldnn_conv(x2).to_dense()
                # 断言 Aten 卷积和 MKL-DNN 卷积输出相等
                self.assertEqual(y_aten, y_mkldnn)
            # 若不进行训练，则测试卷积对象的序列化
            if not train:
                self._test_serialization(mkldnn_conv, (x.to_mkldnn(),))
                # 测试卷积对象的追踪
                self._test_tracing(mkldnn_conv, (x.to_mkldnn(),))
            # 若进行训练且维度不为 1，则计算 loss2 并反向传播
            elif dim != 1:
                loss2 = y_mkldnn.sum()
                loss2.backward()
                # 断言输入 x2 的梯度为 MKL-DNN 张量
                self.assertTrue(x2.grad.is_mkldnn)
                # 断言 x1 的梯度与 x2 的梯度转换为普通张量后相等
                self.assertEqual(x1.grad, x2.grad.to_dense())
                # 断言卷积权重的梯度相等，允许一定的数值误差
                self.assertEqual(conv.weight.grad,
                                 mkldnn_conv.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                # 若有偏置项，则断言偏置的梯度相等
                if bias:
                    self.assertEqual(conv.bias.grad, mkldnn_conv.bias.grad)

    # 测试 Conv1d 的方法，调用 _test_conv_base 方法，指定维度为 1
    def test_conv1d(self):
        self._test_conv_base(dim=1)

    # 测试 Conv2d 的方法，调用 _test_conv_base 方法，指定维度为 2
    def test_conv2d(self):
        self._test_conv_base(dim=2)

    # 测试 Conv3d 的方法，调用 _test_conv_base 方法，指定维度为 3
    def test_conv3d(self):
        self._test_conv_base(dim=3)
    # 定义一个测试方法，用于测试低精度下的卷积和反卷积基础操作
    def _test_conv_deconv_lower_precision_base(self, dim, conv_module, dtype):
        # 定义不同维度下的输入形状
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        # 组合不同的选项：是否有偏置、膨胀率、分组数
        options = itertools.product([True, False], [1, 2], [1, 4])
        # 遍历所有选项
        for bias, dilation, groups in options:
            # 随机生成 N、M、C 的值
            N = torch.randint(1, 3, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            # 根据当前维度选择输入的形状
            x_shape = (N, C) + input_shapes[dim]
            # 生成随机输入数据 x
            x = torch.randn(x_shape, dtype=torch.float32)
            # 对于不支持组深度卷积的情况，跳过当前循环
            if conv_module in [torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d] and groups > 1 and C == groups:
                continue
            # 创建卷积或反卷积层 conv
            conv = conv_module(in_channels=C,
                               out_channels=M,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               dilation=dilation,
                               bias=bias,
                               groups=groups).float()
            # 将输入 x 转换为指定的数据类型 dtype
            x_lower = x.to(dtype=dtype)
            # 如果是 bfloat16 或者是支持 mkldnn_fp16 的情况，使用 MKL-DNN 执行
            if (dtype == torch.bfloat16 and torch.ops.mkldnn._is_mkldnn_bf16_supported()) or \
               (dtype == torch.half and torch.ops.mkldnn._is_mkldnn_fp16_supported()):
                # 将普通的 conv 转换为 MKL-DNN 格式
                mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
                # 将低精度的 conv 转换为 MKL-DNN 格式
                mkldnn_conv_lower = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), dtype)
                # 使用 MKL-DNN 执行卷积操作，并转换为密集张量
                y = mkldnn_conv(x.to_mkldnn()).to_dense()
                # 在低精度下使用 MKL-DNN 执行卷积操作，并转换为 float32 的密集张量
                y_lower = mkldnn_conv_lower(x_lower.to_mkldnn()).to_dense(torch.float32)
                # 断言两者结果相等，设置容差
                self.assertEqual(y, y_lower, atol=1e-1, rtol=1e-3)
            else:
                # 如果不支持当前 dtype，期望抛出 RuntimeError
                msg = {
                    torch.bfloat16: r"bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq",
                    torch.half: r"fp16 path needs the cpu support avx_ne_convert or avx512_fp16",
                }
                with self.assertRaisesRegex(RuntimeError, msg[dtype]):
                    # 将低精度的 conv 转换为 MKL-DNN 格式
                    mkldnn_conv_lower = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), dtype)
                    # 在低精度下使用 MKL-DNN 执行卷积操作，并转换为 float32 的密集张量
                    y_lower = mkldnn_conv_lower(x_lower.to_mkldnn()).to_dense(torch.float32)
            # 使用 THNN 实现的卷积操作
            conv_lower = copy.deepcopy(conv).to(dtype=dtype)
            conv_ref = copy.deepcopy(conv_lower).float()
            # 禁用 MKL-DNN 后，进行 THNN 实现的卷积操作
            with torch.backends.mkldnn.flags(enabled=False):
                # 克隆 x_lower 并转换为 float32，同时保留梯度信息
                x_ref = x_lower.clone().float().detach().requires_grad_()
                x_lower.requires_grad_()
                # 执行 THNN 实现的卷积操作
                y = conv_ref(x_ref)
                y_lower = conv_lower(x_lower).float()
                # 断言两者结果相等，设置容差
                self.assertEqual(y, y_lower, atol=5e-2, rtol=5e-3)

    # 指定测试用例的数据类型为 float16 和 bfloat16
    @dtypes(torch.float16, torch.bfloat16)
    # 定义一个测试类方法，用于测试一维卷积和反卷积在低精度情况下的行为
    def test_conv_deconv_1d_lower_precision(self, dtype):
        # 调用基础方法测试一维卷积和反卷积在指定数据类型下的行为
        self._test_conv_deconv_lower_precision_base(1, torch.nn.Conv1d, dtype=dtype)
        # 调用基础方法测试一维转置卷积和反卷积在指定数据类型下的行为
        self._test_conv_deconv_lower_precision_base(1, torch.nn.ConvTranspose1d, dtype=dtype)
    
    # 使用指定的数据类型装饰器，定义一个测试类方法，用于测试二维卷积和反卷积在低精度情况下的行为
    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_deconv_2d_lower_precision(self, dtype):
        # 调用基础方法测试二维卷积和反卷积在指定数据类型下的行为
        self._test_conv_deconv_lower_precision_base(2, torch.nn.Conv2d, dtype=dtype)
        # 调用基础方法测试二维转置卷积和反卷积在指定数据类型下的行为
        self._test_conv_deconv_lower_precision_base(2, torch.nn.ConvTranspose2d, dtype=dtype)
    
    # 使用指定的数据类型装饰器，定义一个测试类方法，用于测试三维卷积和反卷积在低精度情况下的行为
    @dtypes(torch.float16, torch.bfloat16)
    def test_conv_deconv_3d_lower_precision(self, dtype):
        # 调用基础方法测试三维卷积和反卷积在指定数据类型下的行为
        self._test_conv_deconv_lower_precision_base(3, torch.nn.Conv3d, dtype=dtype)
        # 调用基础方法测试三维转置卷积和反卷积在指定数据类型下的行为
        self._test_conv_deconv_lower_precision_base(3, torch.nn.ConvTranspose3d, dtype=dtype)
    # 定义测试函数，用于测试卷积和反卷积在不同内存格式下的行为
    def _test_conv_deconv_nhwc_base(self, conv_module, weight_memory_format, dtype, prec=None):
        # 定义不同输入形状的字典
        input_shapes = {2: (55, 55), 3: (14, 14, 14)}
        # 生成所有可能的选项组合
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        
        # 根据卷积模块选择通道顺序格式和输入形状
        if conv_module in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
            cl_format = torch.channels_last
            input_shape = input_shapes[2]
        elif conv_module in [torch.nn.Conv3d, torch.nn.ConvTranspose3d]:
            cl_format = torch.channels_last_3d
            input_shape = input_shapes[3]

        # 遍历所有选项组合进行测试
        for train, bias, dilation, groups in options:
            # 随机生成 N、M、C 的值
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shape
            x = torch.randn(x_shape, dtype=dtype)

            # 创建第一个卷积层 conv1: 在连续内存格式 (nchw) 中进行 MKLDNN 卷积/反卷积
            # 创建第二个卷积层 conv2: 在通道最后内存格式 (nhwc) 中进行 MKLDNN 卷积/反卷积
            conv1 = conv_module(in_channels=C,
                                out_channels=M,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                dilation=dilation,
                                bias=bias,
                                groups=groups).to(dtype=dtype)
            conv2 = copy.deepcopy(conv1).to(memory_format=weight_memory_format)
            
            # 克隆输入张量 x，分别用于两种不同的内存格式
            x1 = x.clone()
            x2 = x.clone().to(memory_format=cl_format)
            
            # 如果需要进行训练，则设置梯度
            if train:
                x1.requires_grad_()
                x2.requires_grad_()
            
            # 分别对两个卷积层进行前向传播
            y1 = conv1(x1)
            y2 = conv2(x2)
            
            # 断言两种内存格式下的输出结果 y1 和 y2 相等，允许的绝对误差和相对误差为 prec
            self.assertEqual(y1, y2, atol=prec, rtol=prec)

            # 如果正在训练阶段，则进行反向传播和梯度检查
            if train:
                y1.sum().backward()
                y2.sum().backward()
                
                # 断言通道最后内存格式下的梯度是连续的
                self.assertTrue(x2.grad.is_contiguous(memory_format=cl_format))
                
                # 检查卷积核权重的梯度是否相等，允许的绝对误差和相对误差为 1e-3
                self.assertEqual(conv1.weight.grad,
                                 conv2.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                
                # 如果有偏置项，则检查偏置项的梯度是否相等，允许的绝对误差和相对误差为 prec
                if bias:
                    self.assertEqual(conv1.bias.grad, conv2.bias.grad, atol=prec, rtol=prec)
                
                # 检查输入张量 x 的梯度是否相等，允许的绝对误差和相对误差为 prec
                self.assertEqual(x1.grad, x2.grad, atol=prec, rtol=prec)

    # 测试卷积层在通道最后内存格式下的行为，使用单精度浮点数（float32）作为数据类型
    def test_conv_nhwc_fp32(self):
        self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.contiguous_format, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.channels_last, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.contiguous_format, dtype=torch.float32)
        self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.channels_last_3d, dtype=torch.float32)

    # 指定数据类型为 torch.float16 和 torch.bfloat16 进行测试
    @dtypes(torch.float16, torch.bfloat16)
    # 测试 NHWC 格式的低精度转换
    def test_conv_nhwc_lower_precision(self, dtype):
        # 当 torch.ops.mkldnn._is_mkldnn_bf16_supported() 或 torch.ops.mkldnn._is_mkldnn_fp16_supported()
        # 返回 false 时，bf16/fp16 CPU conv 会回退到 thnn 实现
        support_checks = {
            torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
            torch.float16: torch.ops.mkldnn._is_mkldnn_fp16_supported
        }
        # 如果当前数据类型支持 MKL-DNN
        if support_checks[dtype]():
            # 测试使用 torch.nn.Conv2d 在 NHWC 格式下的基础操作
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.contiguous_format, dtype=dtype)
            # 测试使用 torch.nn.Conv2d 在通道优先格式下的基础操作
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.channels_last, dtype=dtype)
            # 测试使用 torch.nn.Conv3d 在 NHWC 格式下的基础操作
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.contiguous_format, dtype=dtype)
            # 测试使用 torch.nn.Conv3d 在通道优先格式下的基础操作
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.channels_last_3d, dtype=dtype)

        # BF16/FP16 回退实现分为两部分：im2col+gemm，
        # 中间的数据类型转换次数比 onednn 的直接卷积要多，导致额外的精度损失
        precisions = {
            torch.bfloat16: 1e-2,
            torch.float16: 2e-3,
        }
        # 获取当前数据类型的精度要求
        prec = precisions[dtype]
        # 关闭 MKL-DNN 后进行测试
        with torch.backends.mkldnn.flags(enabled=False):
            # 在 NHWC 格式下测试使用 torch.nn.Conv2d 的基础操作，设定精度要求
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.contiguous_format, dtype=dtype, prec=prec)
            # 在通道优先格式下测试使用 torch.nn.Conv2d 的基础操作，设定精度要求
            self._test_conv_deconv_nhwc_base(torch.nn.Conv2d, torch.channels_last, dtype=dtype, prec=prec)
            # 在 NHWC 格式下测试使用 torch.nn.Conv3d 的基础操作，设定精度要求
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.contiguous_format, dtype=dtype, prec=prec)
            # 在通道优先格式下测试使用 torch.nn.Conv3d 的基础操作，设定精度要求
            self._test_conv_deconv_nhwc_base(torch.nn.Conv3d, torch.channels_last_3d, dtype=dtype, prec=prec)


    # 测试 NHWC 格式下的 FP32 转置卷积
    def test_conv_transpose_nhwc_fp32(self):
        # 测试使用 torch.nn.ConvTranspose2d 在 NHWC 格式下的基础操作，数据类型为 float32
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.contiguous_format, dtype=torch.float32)
        # 测试使用 torch.nn.ConvTranspose2d 在通道优先格式下的基础操作，数据类型为 float32
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.channels_last, dtype=torch.float32)
        # 测试使用 torch.nn.ConvTranspose3d 在 NHWC 格式下的基础操作，数据类型为 float32
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.contiguous_format, dtype=torch.float32)
        # 测试使用 torch.nn.ConvTranspose3d 在通道优先格式下的基础操作，数据类型为 float32
        self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.channels_last_3d, dtype=torch.float32)

    # 使用 torch.float16 和 torch.bfloat16 数据类型进行测试
    @dtypes(torch.float16, torch.bfloat16)
    # 定义一个测试函数，用于测试转置卷积操作在不同数据类型和精度下的表现
    def test_conv_transpose_nhwc_lower_precision(self, dtype):
        # 当 torch.ops.mkldnn._is_mkldnn_bf16_supported() 或 torch.ops.mkldnn._is_mkldnn_fp16_supported() 返回 False 时，
        # bf16/fp16 CPU 卷积将会回退到 thnn 实现
        support_checks = {
            torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
            torch.float16: torch.ops.mkldnn._is_mkldnn_fp16_supported
        }
        # 如果当前数据类型支持相应的 MKLDNN 格式
        if support_checks[dtype]():
            # 测试使用 torch.nn.ConvTranspose2d 进行转置卷积，使用 torch.contiguous_format 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.contiguous_format, dtype=dtype)
            # 测试使用 torch.nn.ConvTranspose2d 进行转置卷积，使用 torch.channels_last 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.channels_last, dtype=dtype)
            # 测试使用 torch.nn.ConvTranspose3d 进行转置卷积，使用 torch.contiguous_format 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.contiguous_format, dtype=dtype)
            # 测试使用 torch.nn.ConvTranspose3d 进行转置卷积，使用 torch.channels_last_3d 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.channels_last_3d, dtype=dtype)

        # BF16/FP16 回退实现分为两部分：col2im+gemm，中间数据类型转换比 onednn 的直接卷积更多，
        # 导致额外的精度损失
        precisions = {
            torch.bfloat16: 2e-2,
            torch.float16: 3e-3,
        }
        # 从精度字典中获取当前数据类型对应的预期精度
        prec = precisions[dtype]
        # 禁用 MKLDNN 后端标志，执行以下测试
        with torch.backends.mkldnn.flags(enabled=False):
            # 使用指定的精度 prec 进行测试，使用 torch.nn.ConvTranspose2d 进行转置卷积，
            # 使用 torch.contiguous_format 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.contiguous_format, dtype=dtype, prec=prec)
            # 使用指定的精度 prec 进行测试，使用 torch.nn.ConvTranspose2d 进行转置卷积，
            # 使用 torch.channels_last 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose2d, torch.channels_last, dtype=dtype, prec=prec)
            # 使用指定的精度 prec 进行测试，使用 torch.nn.ConvTranspose3d 进行转置卷积，
            # 使用 torch.contiguous_format 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.contiguous_format, dtype=dtype, prec=prec)
            # 使用指定的精度 prec 进行测试，使用 torch.nn.ConvTranspose3d 进行转置卷积，
            # 使用 torch.channels_last_3d 格式
            self._test_conv_deconv_nhwc_base(torch.nn.ConvTranspose3d, torch.channels_last_3d, dtype=dtype, prec=prec)
    # 定义一个测试函数，用于测试不同维度的转置卷积操作
    def _test_conv_transpose_base(self, dim):
        # 定义转置卷积模块的字典，根据不同维度选择不同的模块
        conv_module = {
            1: torch.nn.ConvTranspose1d,
            2: torch.nn.ConvTranspose2d,
            3: torch.nn.ConvTranspose3d
        }
        # 输入数据的形状字典，根据不同维度设置不同的输入形状
        input_shapes = {1: (55,), 2: (28, 28), 3: (14, 14, 14)}
        # 使用 itertools 生成所有可能的选项组合
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        # 遍历每一种选项组合
        for train, bias, dilation, groups in options:
            # 随机生成 N、M、C 作为输入数据的维度参数
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            # 根据当前维度设置输入数据 x 的形状
            x_shape = (N, C) + input_shapes[dim]
            # 生成随机输入数据
            data = torch.randn(x_shape, dtype=torch.float32)
            # 创建转置卷积对象 conv，并设置各种参数
            # conv: mkldnn tranpose conv fp32
            # conv_ref: thnn transpose conv fp32
            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).to(dtype=torch.float32)
            # 复制输入数据 x，并为 train 模式下创建梯度
            x = data.clone()
            x_ref = x.clone()
            if train:
                x.requires_grad_()
                x_ref.requires_grad_()
    
            # 深拷贝 conv 对象到 conv_ref
            conv_ref = copy.deepcopy(conv)
            # 禁用 mkldnn 后端，计算 conv_ref 对 x_ref 的输出 y_ref
            with torch.backends.mkldnn.flags(enabled=False):
                y_ref = conv_ref(x_ref)
                if train:
                    # 在 train 模式下，计算 y_ref 的梯度
                    y_ref.sum().backward()
    
            # 计算 conv 对 x 的输出 y
            y = conv(x)
            if train:
                # 在 train 模式下，计算 y 的梯度
                y.sum().backward()
    
            # 使用断言检查转置卷积结果 y 与 y_ref 是否相等
            self.assertEqual(y, y_ref)
            if train:
                # 如果在 train 模式下，还需要检查输入数据 x 的梯度与 conv 对象的梯度是否相等
                self.assertEqual(x.grad, x_ref.grad)
                # 检查 conv 对象的权重梯度与 conv_ref 的权重梯度是否相等，允许的误差为 1e-3
                self.assertEqual(conv.weight.grad,
                                 conv_ref.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                # 如果使用偏置项，还需检查 conv 对象的偏置项梯度与 conv_ref 的偏置项梯度是否相等
                if bias:
                    self.assertEqual(conv.bias.grad, conv_ref.bias.grad)
    
    # 测试 ConvTranspose1d 的函数
    def test_conv_transpose1d(self):
        self._test_conv_transpose_base(dim=1)
    
    # 测试 ConvTranspose2d 的函数
    def test_conv_transpose2d(self):
        self._test_conv_transpose_base(dim=2)
    
    # 测试 ConvTranspose3d 的函数
    def test_conv_transpose3d(self):
        self._test_conv_transpose_base(dim=3)
    def test_conv2d_legacy_jit_model(self):
        """
        MKLDNN integration used to serialize models with 5d weight for grouped
        convolutions, we'd like to preserve this behavior
        """
        # 定义组数为4的2D卷积层
        g = 4
        conv2d = torch.nn.Conv2d(16, 16, 3, groups=g)
        # 将2D卷积层转换为MKLDNN格式
        conv2d_mkldnn = torch.utils.mkldnn.to_mkldnn(conv2d)

        # 构造具有5维权重的传统conv2d模块
        o, i, h, w = conv2d.weight.shape
        weight_5d = conv2d.weight.reshape((g, o // g, i, h, w))
        conv2d_mkldnn.weight = weight_5d.to_mkldnn()

        x = torch.randn(1, 16, 8, 8)

        # 使用临时文件名保存和加载MKLDNN格式的模型
        with TemporaryFileName() as fname:
            torch.jit.save(conv2d_mkldnn, fname)
            conv2d_loaded = torch.jit.load(fname)

            # 检查加载后的权重维度是否为4
            self.assertEqual(conv2d_mkldnn.weight.ndimension(), 5)
            self.assertEqual(conv2d_loaded.weight.ndimension(), 4)
            # 检查模型预测结果是否一致
            self.assertEqual(
                conv2d(x),
                conv2d_loaded(x.to_mkldnn()).to_dense())

    # This test is to check whether 1D conv is supported for mkldnn tensor,
    # which is exposed by Issue https://github.com/pytorch/pytorch/issues/68034.
    def test_conv1d_functional(self):
        # 创建MKLDNN格式的输入、权重和偏置
        input = torch.randn(2, 3, 10).to_mkldnn()
        weight = torch.randn(3, 3, 3).to_mkldnn()
        bias = torch.randn(3).to_mkldnn()
        # 使用torch.nn.functional中的conv1d函数进行1D卷积操作
        output = torch.nn.functional.conv1d(input, weight, bias)
        # 检查输出的尺寸是否为[2, 3, 8]
        self.assertEqual(output.size(), torch.Size([2, 3, 8]))

    def test_relu(self):
        # 创建随机数填充的张量x，并克隆为x1和x2
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        # 计算ReLU激活函数后的输出y1和y2
        y1 = torch.relu(x1)
        y2 = torch.relu(x2).to_dense()
        # 计算y1和y2的和作为损失值
        loss1 = y1.sum()
        loss2 = y2.sum()
        # 反向传播损失值
        loss1.backward()
        loss2.backward()
        # 检查y1和y2是否相等
        self.assertEqual(y1, y2)
        # 检查x1和x2的梯度是否相等
        self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_relu_(self):
        # 创建随机数填充的张量x，并克隆为x1和x2
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        # 原地计算ReLU激活函数后的输出y1和y2
        y1 = torch.relu_(x1.clone())
        y2 = torch.relu_(x2.clone()).to_dense()
        # 计算y1和y2的和作为损失值
        loss1 = y1.sum()
        loss2 = y2.sum()
        # 反向传播损失值
        loss1.backward()
        loss2.backward()
        # 检查y1和y2是否相等
        self.assertEqual(y1, y2)
        # 检查x1和x2的梯度是否相等
        self.assertEqual(x1.grad, x2.grad.to_dense())

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 定义测试方法 _test_relu_bf16_base，接受一个名称参数 name
    def _test_relu_bf16_base(self, name):
        # 创建一个形状为 (4, 5) 的随机张量 x，数据类型为 float32，并乘以 10
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        # 将 x 转换为 bfloat16 类型的张量 x_bf16
        x_bf16 = x.bfloat16()
        # 根据给定的名称获取 torch 模块中对应的函数
        fn = getattr(torch, name)
        # 检查当前环境是否支持 MKL-DNN 的 bfloat16 操作
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # 如果支持，对 x 执行 MKL-DNN 的操作并转换为稠密张量 y
            y = fn(x.to_mkldnn()).to_dense()
            # 对 x_bf16 执行相同的操作并转换为 float32 类型的稠密张量 y_bf16
            y_bf16 = fn(x_bf16.to_mkldnn()).to_dense(torch.float32)
            # 断言 y 和 y_bf16 相等，允许绝对误差为 1e-1，相对误差为 1e-3
            self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
        else:
            # 如果不支持，抛出错误消息指示需要 AVX512BW、AVX512VL 和 AVX512DQ 的 CPU 支持
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: fn(x_bf16.to_mkldnn()))

    # 定义测试方法 test_relu_bf16，调用 _test_relu_bf16_base 测试 relu 函数的 bfloat16 实现
    def test_relu_bf16(self):
        self._test_relu_bf16_base("relu")

    # 定义测试方法 test_relu_inplace_bf16，调用 _test_relu_bf16_base 测试 relu_ 函数的 bfloat16 实现
    def test_relu_inplace_bf16(self):
        self._test_relu_bf16_base("relu_")

    # 定义测试方法 test_gelu，测试 GELU 激活函数的正常操作
    def test_gelu(self):
        # 创建 GELU 激活函数的实例 m
        m = torch.nn.GELU()
        # 创建一个形状为 (4, 5) 的随机张量 x，数据类型为 float32，并乘以 10
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        # 克隆 x 并设置其需要梯度计算
        x1 = x.clone().requires_grad_()
        # 克隆 x 并将其转换为 MKL-DNN 格式，同时设置需要梯度计算
        x2 = x.clone().to_mkldnn().requires_grad_()
        # 对 x1 和 x2 分别应用 GELU 激活函数，y1 为普通张量，y2 转换为稠密张量
        y1 = m(x1)
        y2 = m(x2).to_dense()
        # 计算 y1 的总和作为 loss1
        loss1 = y1.sum()
        # 计算 y2 的总和作为 loss2
        loss2 = y2.sum()
        # 对 loss1 和 loss2 分别执行反向传播
        loss1.backward()
        loss2.backward()
        # 断言 y1 和 y2 相等
        self.assertEqual(y1, y2)
        # 断言 x1 的梯度和 x2 的梯度转换为稠密张量后相等
        self.assertEqual(x1.grad, x2.grad.to_dense())

    # 根据操作系统是否为 Windows 平台决定是否跳过测试方法 test_gelu_bf16
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_gelu_bf16(self):
        # 创建 GELU 激活函数的实例 m
        m = torch.nn.GELU()
        # 创建一个形状为 (4, 5) 的随机张量 x，数据类型为 float32，并乘以 10
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        # 克隆 x 并将其转换为 MKL-DNN 格式，同时设置需要梯度计算
        x1 = x.clone().to_mkldnn().requires_grad_()
        # 克隆 x 并将其转换为 MKL-DNN 格式的 bfloat16，同时设置需要梯度计算
        x2 = x.clone().to_mkldnn(torch.bfloat16).requires_grad_()
        # 检查当前环境是否支持 MKL-DNN 的 bfloat16 操作
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # 如果支持，对 x1 和 x2 分别应用 GELU 激活函数并转换为稠密张量
            y1 = m(x1).to_dense()
            y2 = m(x2).to_dense()
            # 计算 y1 和 y2 的总和作为 loss1 和 loss2
            loss1 = y1.sum()
            loss2 = y2.sum()
            # 对 loss1 和 loss2 分别执行反向传播
            loss1.backward()
            loss2.backward()
            # 断言 y1 和 y2 转换为 float32 类型后相等，允许绝对误差为 1e-1，相对误差为 0
            self.assertEqual(y1, y2.to(torch.float32), atol=1e-1, rtol=0)
            # 断言 x1 的梯度转换为稠密张量后和 x2 的梯度转换为 float32 类型后的梯度相等，允许绝对误差为 1e-2，相对误差为 0
            self.assertEqual(x1.grad.to_dense(), x2.grad.to_dense(torch.float32), atol=1e-2, rtol=0)
        else:
            # 如果不支持，抛出错误消息指示需要 AVX512BW、AVX512VL 和 AVX512DQ 的 CPU 支持
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: m(x2))
    # 定义测试方法，用于测试 PReLU 激活函数的基本功能，接受输入的尺寸和通道数
    def _test_prelu_base(self, size, num_channels):
        # 生成随机张量 x，数据类型为 float32
        x = torch.randn(size, dtype=torch.float32)
        # 克隆张量 x 并声明需要梯度
        x1 = x.clone().requires_grad_()
        # 将张量 x 克隆并转换为 MKL-DNN 张量，同时声明需要梯度
        x2 = x.clone().to_mkldnn().requires_grad_()
        # 同上，另一个 MKL-DNN 张量的克隆
        x3 = x.clone().to_mkldnn().requires_grad_()
        # 创建 PReLU 模块，指定通道数
        m1 = torch.nn.PReLU(num_channels)
        # 将 m1 模块复制到 MKL-DNN
        m2 = mkldnn_utils.to_mkldnn(copy.deepcopy(m1))
        # 深度复制 m1，得到 m3
        m3 = copy.deepcopy(m1)
        # 对 x1 应用 m1，得到输出 y1
        y1 = m1(x1)
        # 对 x2 应用 MKL-DNN 版本的 m2，并转换为稠密张量，得到输出 y2
        y2 = m2(x2).to_dense()
        # 对 x3 应用 m3，并转换为稠密张量，得到输出 y3，只有数据转换为 MKL-DNN，权重仍是 Aten 张量
        y3 = m3(x3).to_dense()
        # 计算 y1 的和作为损失
        loss1 = y1.sum()
        # 反向传播 loss1
        loss1.backward()
        # 计算 y2 的和作为损失
        loss2 = y2.sum()
        # 反向传播 loss2
        loss2.backward()
        # 计算 y3 的和作为损失
        loss3 = y3.sum()
        # 反向传播 loss3
        loss3.backward()
        # 断言 y1 等于 y2
        self.assertEqual(y1, y2)
        # 断言 y1 等于 y3
        self.assertEqual(y1, y3)
        # 断言 x1 的梯度等于 x2 转换为稠密张量的梯度
        self.assertEqual(x1.grad, x2.grad.to_dense())
        # 断言 x1 的梯度等于 x3 转换为稠密张量的梯度
        self.assertEqual(x1.grad, x3.grad.to_dense())

    # 定义 PReLU 测试方法
    def test_prelu(self):
        # 使用不同的尺寸和通道数调用 _test_prelu_base 进行测试
        self._test_prelu_base(torch.Size([16]), 1)
        self._test_prelu_base(torch.Size([16, 64]), 1)
        self._test_prelu_base(torch.Size([16, 64]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112, 112]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112, 112]), 64)
        self._test_prelu_base(torch.Size([16, 64, 112, 112, 1]), 1)
        self._test_prelu_base(torch.Size([16, 64, 112, 112, 1]), 64)

    # 在 Windows 平台上跳过测试
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 定义测试 MKL-DNN BF16 路径下 PReLU 的基本功能方法
    def _test_prelu_bf16_base(self, size, num_channels):
        # 检查当前环境是否支持 MKL-DNN BF16
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # 生成随机张量 x，数据类型为 float32
            x = torch.randn(size, dtype=torch.float32)
            # 将 x 克隆并转换为 MKL-DNN 张量，并声明需要梯度
            x_fp32 = x.clone().to_mkldnn().requires_grad_()
            # 将 x 克隆并转换为 MKL-DNN BF16 张量，并声明需要梯度
            x_bf16 = x.clone().to_mkldnn(torch.bfloat16).requires_grad_()
            # 将标准 PReLU 模块转换为 MKL-DNN 模块
            m = mkldnn_utils.to_mkldnn(torch.nn.PReLU())
            # 将标准 PReLU 模块转换为 MKL-DNN BF16 模块
            m_bf16 = mkldnn_utils.to_mkldnn(torch.nn.PReLU(), torch.bfloat16)

            # 对 x_fp32 应用 MKL-DNN 版本的 m，转换为稠密张量，得到输出 y
            y = m(x_fp32).to_dense()
            # 对 x_bf16 应用 MKL-DNN BF16 版本的 m_bf16，转换为稠密张量，得到输出 y_bf16
            y_bf16 = m_bf16(x_bf16).to_dense()
            # 断言 y 和 y_bf16 转换为 float32 后相等，容错参数为 atol=1e-1, rtol=1e-3
            self.assertEqual(y, y_bf16.to(torch.float32), atol=1e-1, rtol=1e-3)

            # 计算 y 的和作为损失
            loss = y.sum()
            # 反向传播 loss
            loss.backward()
            # 计算 y_bf16 的和作为损失
            loss_bf16 = y_bf16.sum()
            # 反向传播 loss_bf16
            loss_bf16.backward()
            # 断言 x_fp32 转换为稠密张量后的梯度等于 x_bf16 转换为 float32 后的梯度
            self.assertEqual(x_fp32.grad.to_dense(), x_bf16.grad.to_dense(torch.float32))
        else:
            # 在不支持 MKL-DNN BF16 的环境下，生成随机 BF16 张量，并声明需要梯度
            x_bf16 = torch.randn(size, dtype=torch.bfloat16).requires_grad_()
            # 将标准 PReLU 模块转换为 MKL-DNN BF16 模块
            m_bf16 = mkldnn_utils.to_mkldnn(torch.nn.PReLU(), torch.bfloat16)
            # 期望的错误消息
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            # 断言运行时异常，并包含指定的错误消息
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: m_bf16(x_bf16))
    # 测试基于 bf16 数据类型的 PReLU 激活函数
    def test_prelu_bf16(self):
        # 测试不同维度的输入数据
        self._test_prelu_bf16_base(torch.Size([16]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64]), 64)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112]), 64)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112, 112, 1]), 1)
        self._test_prelu_bf16_base(torch.Size([16, 64, 112, 112, 1]), 64)

    # 测试基于不同维度的最大池化操作
    def _test_max_pool_base(self, dim, input):
        # 根据维度选择相应的池化模块
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        # 遍历不同的步长和是否使用 ceil 模式
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                # 创建最大池化层对象
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,
                    stride=stride,
                    padding=1,
                    ceil_mode=ceil_mode)

                # 克隆输入数据并设置梯度计算
                x1 = input.clone().requires_grad_()
                x2 = input.clone().to_mkldnn().requires_grad_()
                # 分别对两种输入计算最大池化结果
                y1 = max_pool(x1)
                y2 = max_pool(x2).to_dense()
                # 计算两种输出结果的和作为损失
                loss1 = y1.sum()
                loss2 = y2.sum()
                # 分别对两种输入进行反向传播
                loss1.backward()
                loss2.backward()
                # 断言两种输入的最大池化结果应当相等
                self.assertEqual(y1, y2)
                # 断言两种输入的梯度计算结果应当相等
                self.assertEqual(x1.grad, x2.grad.to_dense())

    # 测试二维最大池化操作
    def test_max_pool2d(self):
        # 随机生成不同的 N 和 C
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 遍历不同的 H 和 W 组合
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            # 生成相应维度的随机输入数据
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            # 调用 _test_max_pool_base 方法进行测试
            self._test_max_pool_base(dim=2, input=x)

    # 测试三维最大池化操作
    def test_max_pool3d(self):
        # 随机生成不同的 N 和 C
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 遍历不同的 D、H 和 W 组合
        for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
            # 生成相应维度的随机输入数据
            x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
            # 调用 _test_max_pool_base 方法进行测试
            self._test_max_pool_base(dim=3, input=x)

    # 在 Windows 系统下，跳过对 bf16 路径的支持测试
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 定义一个测试函数，用于测试二维或三维最大池化操作在bf16数据类型下的表现
    def _test_max_pool_bf16_base(self, dim, input):
        # 创建一个字典，根据维度选择对应的最大池化模块
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        # 将输入张量转换为bf16数据类型
        x_bf16 = input.bfloat16()
        # 遍历不同的步长和是否使用ceil_mode的组合
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                # 创建最大池化层对象
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,  # 设置核大小，根据ceil_mode决定
                    stride=stride,  # 设置步长
                    padding=1,  # 设置填充
                    ceil_mode=ceil_mode)  # 是否使用ceil_mode

                # 如果当前环境支持MKL-DNN的bf16操作
                if torch.ops.mkldnn._is_mkldnn_bf16_supported():
                    # 在MKL-DNN上执行最大池化操作并转换为密集张量
                    y = max_pool(input.to_mkldnn()).to_dense()
                    # 将bf16类型的输入在MKL-DNN上执行最大池化并转换为float32密集张量
                    y_bf16 = max_pool(x_bf16.to_mkldnn()).to_dense(torch.float32)
                    # 断言两个结果的近似相等性，给定的容差和相对容差
                    self.assertEqual(y, y_bf16, atol=0.1, rtol=1e-3)
                else:
                    # 如果不支持MKL-DNN的bf16操作，抛出特定错误消息
                    msg = "mkldnn_max_pool%dd: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq" % dim
                    self.assertRaisesRegex(RuntimeError,
                                           msg,
                                           lambda: max_pool(x_bf16.to_mkldnn()))

    # 测试二维bf16最大池化操作
    def test_max_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 遍历不同的高度和宽度组合
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            # 创建随机张量作为输入，数据类型为float32，并乘以10
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            # 调用基础最大池化测试函数，维度为2
            self._test_max_pool_bf16_base(dim=2, input=x)

    # 测试三维bf16最大池化操作
    def test_max_pool3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 遍历不同的深度、高度和宽度组合
        for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
            # 创建随机张量作为输入，数据类型为float32，并乘以10
            x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
            # 调用基础最大池化测试函数，维度为3
            self._test_max_pool_bf16_base(dim=3, input=x)

    # 测试二维最大池化操作中stride为None的情况
    def test_max_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 遍历不同的高度和宽度组合
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            # 创建随机张量作为输入，数据类型为float32，并乘以10
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            # 遍历是否使用ceil_mode的情况
            for ceil_mode in [False, True]:
                # 使用PyTorch的F函数执行二维最大池化操作
                y1 = F.max_pool2d(
                    x,
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,  # 设置stride为None
                    padding=1,  # 设置填充
                    ceil_mode=ceil_mode)  # 是否使用ceil_mode

                # 在MKL-DNN上使用PyTorch的F函数执行二维最大池化操作
                y2 = F.max_pool2d(
                    x.to_mkldnn(),
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,  # 设置stride为None
                    padding=1,  # 设置填充
                    ceil_mode=ceil_mode)  # 是否使用ceil_mode

                # 断言两个结果的密集表示是否相等
                self.assertEqual(y1, y2.to_dense())

    # 以下是一个链接到PyTorch GitHub仓库上的issue的注释，标明了一个问题
    @xfailIfTorchDynamo
    def test_max_pool_unsupported(self):
        # OneDNN不支持带有 dilation 的 max pooling，在 v2.0 版本中将可用。
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        # 2维 dilation 情况
        x = torch.randn(N, C, 7, 7, dtype=torch.float32).to_mkldnn()
        max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'mkldnn_max_pool2d does not support dilation case',
                               lambda: max_pool2d(x))

        # 3维 dilation 情况
        x = torch.randn(N, C, 7, 7, 7, dtype=torch.float32).to_mkldnn()
        max_pool3d = torch.nn.MaxPool3d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'mkldnn_max_pool3d does not support dilation case',
                               lambda: max_pool3d(x))

    def _test_avg_pool_base(self, dim, input):
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        for count_include_pad in [True, False]:
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            x1 = input.clone().requires_grad_()
            x2 = input.clone().to_mkldnn().requires_grad_()
            y1 = avg_pool(x1)
            y2 = avg_pool(x2).to_dense()
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            self.assertEqual(y1, y2)
            self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_base(dim=2, input=x)

    def test_avg_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_base(dim=3, input=x)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 定义测试函数，用于测试 bfloat16 数据类型的平均池化操作，针对不同维度的输入
    def _test_avg_pool_bf16_base(self, dim, input):
        # 创建包含 AvgPool2d 和 AvgPool3d 的字典，根据维度选择相应的池化模块
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        # 将输入张量转换为 bfloat16 数据类型
        x_bf16 = input.bfloat16()
        # 遍历是否包括填充的两种情况
        for count_include_pad in [True, False]:
            # 根据维度从字典中选择相应的平均池化模块，并设置参数
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)
            # 如果当前环境支持 MKL-DNN 的 bfloat16 特性
            if torch.ops.mkldnn._is_mkldnn_bf16_supported():
                # 将输入转换为 MKL-DNN 格式进行计算，然后转换为稠密张量
                y = avg_pool(input.to_mkldnn()).to_dense()
                # 将 bfloat16 格式的输入转换为 MKL-DNN 格式进行计算，然后再转换为 float32 格式的稠密张量
                y_bf16 = avg_pool(x_bf16.to_mkldnn()).to_dense(torch.float)
                # 断言两种计算结果相等，允许的绝对误差为 1e-1，相对误差为 1e-3
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                # 如果不支持 bfloat16，抛出异常信息
                msg = "mkldnn_avg_pool%dd: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq" % dim
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: avg_pool(x_bf16.to_mkldnn()))

    # 测试二维 bfloat16 平均池化操作
    def test_avg_pool2d_bf16(self):
        # 随机生成张量维度大小
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 生成指定维度的随机张量，并乘以 10
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        # 调用基础测试函数，维度为 2
        self._test_avg_pool_bf16_base(dim=2, input=x)

    # 测试三维 bfloat16 平均池化操作
    def test_avg_pool3d_bf16(self):
        # 随机生成张量维度大小
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 生成指定维度的随机张量，并乘以 10
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        # 调用基础测试函数，维度为 3
        self._test_avg_pool_bf16_base(dim=3, input=x)

    # 测试二维平均池化的 stride 参数为 None 的情况
    def test_avg_pool2d_stride_none(self):
        # 随机生成张量维度大小
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 生成指定维度的随机张量，并乘以 10
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

        # 遍历是否包括填充的两种情况
        for count_include_pad in [True, False]:
            # 使用 PyTorch 中的平均池化函数进行计算，不指定 stride
            y1 = F.avg_pool2d(
                x,
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)
            # 将输入转换为 MKL-DNN 格式进行计算，不指定 stride
            y2 = F.avg_pool2d(
                x.to_mkldnn(),
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)

            # 断言两种计算结果相等
            self.assertEqual(y1, y2.to_dense())

    # 测试自适应二维平均池化操作
    def test_adaptive_avg_pool2d(self):
        # 随机生成张量维度大小
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 生成指定维度的随机张量，并乘以 100
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        # 创建自适应二维平均池化模块，池化输出大小为 7x7
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)
        # 克隆张量并开启梯度追踪
        x1 = x.clone().requires_grad_()
        # 将张量转换为 MKL-DNN 格式并开启梯度追踪
        x2 = x.clone().to_mkldnn().requires_grad_()
        # 分别对两种格式的输入进行自适应平均池化操作
        y1 = adaptive_avg_pool2d(x1)
        y2 = adaptive_avg_pool2d(x2).to_dense()

        # 计算张量的和，并执行反向传播
        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()

        # 断言两种计算结果相等
        self.assertEqual(y1, y2)
        # 断言两种格式的输入梯度相等
        self.assertEqual(x1.grad, x2.grad.to_dense())

    # 跳过 Windows 平台的测试，限制对 bfloat16 路径的支持
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 定义测试函数，测试自适应平均池化操作对 BF16 数据的影响
    def test_adaptive_avg_pool2d_bf16(self):
        # 随机生成样本数 N 和通道数 C
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        # 生成随机输入张量 x，形状为 N × C × 224 × 224，数据类型为 float32
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        # 将输入张量 x 转换为 BF16 数据类型
        x_bf16 = x.bfloat16()
        # 创建自适应平均池化层对象，目标输出尺寸为 7 × 7
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        # 检查是否支持 MKLDNN 的 BF16 支持，如果支持则进行相应操作
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            # 将输入张量 x 转换为 MKLDNN 张量后进行自适应平均池化并转换回密集张量
            y = adaptive_avg_pool2d(x.to_mkldnn()).to_dense()
            # 类似操作，但最终转换为 float32 后比较两个结果是否相等，容错值为 atol=1e-1, rtol=1e-3
            y_bf16 = adaptive_avg_pool2d(x.to_mkldnn()).to_dense(torch.float32)
            self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
        else:
            # 如果不支持 BF16，抛出运行时异常，提示需要 CPU 支持 avx512bw、avx512vl 和 avx512dq
            msg = "mkldnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: adaptive_avg_pool2d(x_bf16.to_mkldnn()))

    # 用于测试批标准化操作基础的函数，包括 MKLDNN 的转换和一致性检验
    def _test_batch_norm_base(self, dim, channels, input):
        # 根据维度 dim 选择对应的批标准化模块，2 表示二维或 3 表示三维批标准化
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
        # 创建批标准化层对象，设置通道数 channels，将其转换为 float 类型，并设置为推理模式
        bn = bn_module[dim](channels).float().train(False)
        # 将标准化层对象转换为 MKLDNN 张量，并验证转换后的输出是否与普通的密集张量输出一致
        mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
        self.assertEqual(
            bn(input),
            mkldnn_bn(input.to_mkldnn()).to_dense())

        # 测试批标准化层的序列化功能，传入 MKLDNN 张量进行测试
        self._test_serialization(mkldnn_bn, (input.to_mkldnn(),))
        # 测试批标准化层的追踪功能，传入 MKLDNN 张量进行测试
        self._test_tracing(mkldnn_bn, (input.to_mkldnn(),))

    # 用于测试批标准化训练过程的基础函数，包括权重梯度、运行时统计等的一致性检验
    def _test_batch_norm_train_base(self, dim, channels, input):
        # TODO: support 3d batchnorm training.
        bn_module = {2 : torch.nn.BatchNorm2d}
        # TODO: support none affine.
        # 使用 product 函数生成所有可能的 affine 和 track_running_stats 参数组合
        options = itertools.product([True], [True, False])
        for affine, track_running_stats in options:
            # 创建批标准化层对象，设置通道数 channels，是否使用 affine 和 track_running_stats
            bn = bn_module[dim](
                num_features=channels,
                affine=affine,
                track_running_stats=track_running_stats).float().train(True)
            # 深拷贝标准化层对象以创建 MKLDNN 版本
            mkldnn_bn = copy.deepcopy(bn)
            # 克隆输入张量 input 并要求计算梯度
            x1 = input.clone().requires_grad_()
            # 将输入张量 input 转换为 MKLDNN 张量后要求计算梯度
            x2 = input.clone().to_mkldnn().requires_grad_()
            # 分别在普通和 MKLDNN 版本的批标准化层上应用输入并计算输出
            y1 = bn(x1)
            y2 = mkldnn_bn(x2).to_dense()
            # 分别计算普通和 MKLDNN 版本的损失并进行反向传播
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            # 检查两种版本的输出是否一致
            self.assertEqual(y1, y2)
            # 检查普通和 MKLDNN 版本的输入梯度是否一致
            self.assertEqual(x1.grad, x2.grad.to_dense())
            # 检查权重梯度是否一致，容忍度为 rtol=1e-3, atol=1e-3
            self.assertEqual(bn.weight.grad, mkldnn_bn.weight.grad, rtol=1e-3, atol=1e-3)
            # 如果 track_running_stats 开启，检查运行时统计量的一致性
            if track_running_stats:
                self.assertEqual(bn.running_mean, mkldnn_bn.running_mean)
                self.assertEqual(bn.running_var, mkldnn_bn.running_var, rtol=1e-5, atol=1e-5)

    # 测试二维批标准化操作的函数
    def test_batch_norm_2d(self):
        # 随机生成样本数 N 和通道数 C
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        # 生成随机输入张量 x，形状为 N × C × 35 × 45，数据类型为 float32
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        # 调用 _test_batch_norm_base 函数测试二维批标准化操作的基础功能
        self._test_batch_norm_base(dim=2, channels=C, input=x)
        # 调用 _test_batch_norm_train_base 函数测试二维批标准化操作的训练功能
        self._test_batch_norm_train_base(dim=2, channels=C, input=x)
    # 测试三维批量归一化函数
    def test_batch_norm_3d(self):
        # 随机生成一个介于3到9之间的整数，表示样本数N
        N = torch.randint(3, 10, (1,)).item()
        # 随机生成一个介于3到100之间的整数，表示通道数C
        C = torch.randint(3, 100, (1,)).item()
        # 生成一个形状为(N, C, 30, 30, 30)的张量，数据类型为float32，元素服从标准正态分布，再乘以10
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        # 调用测试基础批量归一化函数，维度为3，通道数为C，输入为x
        self._test_batch_norm_base(dim=3, channels=C, input=x)

    # 如果在Windows系统下，跳过使用bf16路径的测试
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_batch_norm_bf16_base(self, dim, channels, input):
        # 定义批量归一化模块字典，键为维度，值为对应的批量归一化类
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
        # 将输入张量转换为bfloat16数据类型
        x_bf16 = input.bfloat16()
        # 循环遍历是否支持训练模式的列表，目前只有False
        # TODO: 支持训练
        for train in [False]:
            # 创建一个对应维度和通道数的批量归一化实例，转换为float类型，并设定训练模式
            bn = bn_module[dim](channels).float().train(train)
            # 使用mkldnn_utils将bn深度复制到mkldnn_bn
            mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
            # 如果当前环境支持MKLDNN的bf16路径
            if torch.ops.mkldnn._is_mkldnn_bf16_supported():
                # 使用bn将输入转换为MKLDNN格式并转为密集张量y
                y = bn(input.to_mkldnn().to_dense())
                # 使用bn将输入转换为MKLDNN格式并转为float的密集张量y_bf16
                y_bf16 = bn(input.to_mkldnn().to_dense(torch.float))
                # 断言y与y_bf16在给定的绝对误差和相对误差下相等
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                # 如果不支持MKLDNN的bf16路径，抛出运行时错误
                msg = "mkldnn_batch_norm: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: bn(x_bf16.to_mkldnn()))

    # 测试二维bf16批量归一化函数
    def test_batch_norm_2d_bf16(self):
        # 随机生成一个介于3到9之间的整数，表示样本数N
        N = torch.randint(3, 10, (1,)).item()
        # 随机生成一个介于3到100之间的整数，表示通道数C
        C = torch.randint(3, 100, (1,)).item()
        # 生成一个形状为(N, C, 35, 45)的张量，数据类型为float32，元素服从标准正态分布，再乘以10
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        # 调用带bf16基础批量归一化函数，维度为2，通道数为C，输入为x
        self._test_batch_norm_bf16_base(dim=2, channels=C, input=x)

    # 测试三维bf16批量归一化函数
    def test_batch_norm_3d_bf16(self):
        # 随机生成一个介于3到9之间的整数，表示样本数N
        N = torch.randint(3, 10, (1,)).item()
        # 随机生成一个介于3到100之间的整数，表示通道数C
        C = torch.randint(3, 100, (1,)).item()
        # 生成一个形状为(N, C, 30, 30, 30)的张量，数据类型为float32，元素服从标准正态分布，再乘以10
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        # 调用带bf16基础批量归一化函数，维度为3，通道数为C，输入为x
        self._test_batch_norm_bf16_base(dim=3, channels=C, input=x)
    def test_add(self):
        # 随机生成张量的维度和数值范围
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        alpha = torch.randn(1, dtype=torch.float32).item()

        # 生成随机张量 x 和 y，维度为 (N, C, 35, 45)，数值范围为 [-10, 10)
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10

        # 将 x 和 y 转换为 MKL-DNN 张量 mx 和 my
        mx = x.to_mkldnn()
        my = y.to_mkldnn()

        # 测试张量相加，验证 MKL-DNN 张量和普通张量的结果是否一致
        self.assertEqual(
            x + y,
            (mx + my).to_dense())

        # 测试带系数的张量相加
        self.assertEqual(
            torch.add(x, y, alpha=alpha),
            torch.add(mx, my, alpha=alpha).to_dense())

        # 原地相加操作
        x += y
        mx += my
        self.assertEqual(x, mx.to_dense())

        # 使用输出张量进行相加操作
        out = x.clone()
        mkldnn_out = out.to_mkldnn()
        torch.add(x, y, alpha=alpha, out=out)
        torch.add(mx, my, alpha=alpha, out=mkldnn_out)
        self.assertEqual(out, mkldnn_out.to_dense())

        # 使用输入张量作为输出张量的原地相加操作：第一个输入
        torch.add(x, y, alpha=alpha, out=x)
        torch.add(mx, my, alpha=alpha, out=mx)
        self.assertEqual(x, mx.to_dense())

        # 使用输入张量作为输出张量的原地相加操作：第二个输入
        torch.add(x, y, alpha=alpha, out=y)
        torch.add(mx, my, alpha=alpha, out=my)
        self.assertEqual(y, my.to_dense())

    def test_mul(self):
        # 随机生成张量的维度和数值范围
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        value = torch.randn(1, dtype=torch.float32).item()

        # 生成随机张量 x 和 y，维度为 (N, C, 35, 45)，数值范围为 [-10, 10)
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10

        # 将 x 和 y 转换为 MKL-DNN 张量 mx 和 my
        mx = x.to_mkldnn()
        my = y.to_mkldnn()

        # 测试张量相乘，验证 MKL-DNN 张量和普通张量的结果是否一致
        self.assertEqual(
            x * y,
            (mx * my).to_dense())

        # 测试张量与标量值相乘
        self.assertEqual(
            x * value,
            (mx * value).to_dense())

        # 测试 torch.mul 函数进行张量相乘
        self.assertEqual(
            torch.mul(x, y),
            torch.mul(mx, my).to_dense())

        # 测试 torch.mul 函数进行张量与标量值相乘
        self.assertEqual(
            torch.mul(x, value),
            torch.mul(mx, value).to_dense())

        # 原地相乘操作
        x *= y
        mx *= my
        self.assertEqual(x, mx.to_dense())

        # 使用输出张量进行相乘操作
        out = x.clone()
        mkldnn_out = out.to_mkldnn()
        torch.mul(x, y, out=out)
        torch.mul(mx, my, out=mkldnn_out)
        self.assertEqual(out, mkldnn_out.to_dense())

        out = x.clone()
        mkldnn_out = out.to_mkldnn()
        torch.mul(x, value, out=out)
        torch.mul(mx, value, out=mkldnn_out)
        self.assertEqual(out, mkldnn_out.to_dense())
    # 测试处理零维张量的情况
    def test_0_dimension_tensor(self):
        # 创建一个形状为 [20, 20, 1, 1] 的随机张量 x，数据类型为 float
        x = torch.rand([20, 20, 1, 1], dtype=torch.float)
        # 创建一个形状为 [20, 20, 0, 1] 的随机张量 y，数据类型为 float
        y = torch.rand([20, 20, 0, 1], dtype=torch.float)

        # 使用 torch.relu 对 y 进行 ReLU 操作，返回张量 out_relu
        out_relu = torch.relu(y)
        # 将 y 转换为 MKLDNN 格式并应用 ReLU，再转换为普通张量格式，得到 out_relu_mkldnn
        out_relu_mkldnn = torch.relu(y.to_mkldnn()).to_dense()
        self.assertEqual(out_relu, out_relu_mkldnn)

        # 计算 x 和 y 的逐元素乘积，得到 out_mul
        out_mul = x * y
        # 将 x 和 y 转换为 MKLDNN 格式，计算其乘积并转换为普通张量格式，得到 out_mul_mkldnn
        out_mul_mkldnn = (x.to_mkldnn() * y.to_mkldnn()).to_dense()
        self.assertEqual(out_mul, out_mul_mkldnn)

        # 计算 x 和 y 的逐元素加法，得到 out_add
        out_add = x + y
        # 将 x 和 y 转换为 MKLDNN 格式，计算其加法并转换为普通张量格式，得到 out_add_mkldnn
        out_add_mkldnn = (x.to_mkldnn() + y.to_mkldnn()).to_dense()
        self.assertEqual(out_add, out_add_mkldnn)

        # 设置 x 和 y 都需要梯度计算
        x.requires_grad_(True)
        y.requires_grad_(True)
        # 使用断言检查在训练过程中是否会出现零维张量的异常
        with self.assertRaisesRegex(RuntimeError, "0-dimension Tensor in training"):
            x.to_mkldnn() + y.to_mkldnn()

        # 使用断言检查是否两个张量的形状不匹配
        with self.assertRaisesRegex(RuntimeError, "must match"):
            torch.rand([5]).to_mkldnn() + torch.rand([0]).to_mkldnn()

        # 定义一个通道数 C，并创建一个 Conv2d 模块 m
        C = 7
        m = torch.nn.Conv2d(C, C, 3)
        # 创建一个形状为 [0, C, C, 8] 的随机张量 x，数据类型为 float
        x = torch.randn(0, C, C, 8, dtype=torch.float)
        # 对 x 应用 m 模块，得到输出 out_eager
        out_eager = m(x)
        # 将 m 模块转换为 MKLDNN 格式后对 x 应用，得到输出 out_mkldnn
        out_mkldnn = mkldnn_utils.to_mkldnn(m)(x)
        self.assertEqual(out_eager, out_mkldnn)

    # 测试 view 操作
    @xfailIfTorchDynamo
    def test_view(self):
        # 创建一个形状为 [3, 4, 5] 的随机张量 x，并转换为 MKLDNN 格式
        x = torch.randn(3, 4, 5, dtype=torch.float32).to_mkldnn()
        # 使用断言检查是否会出现异常提醒使用 reshape 而不是 view
        self.assertRaisesRegex(RuntimeError,
                               "Change to use reshape",
                               lambda: x.view(x.size(0), -1))

    # 测试 reshape 操作
    def test_reshape(self):
        # 创建一个形状为 [3, 4, 5] 的随机张量 x，并乘以 10
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        # 定义一个大小为 (3, -1) 的尺寸
        size = (x.size(0), -1)

        # 使用 reshape 对 x 进行形状变换，与将 x 转换为 MKLDNN 格式后进行 reshape 的结果进行比较
        self.assertEqual(
            x.reshape(size),
            x.to_mkldnn().reshape(size).to_dense(),
        )

        # 测试普通格式张量的共享内存特性
        y = x.to_mkldnn()
        z = y.reshape(size).add_(y.reshape(size))
        self.assertEqual(
            y.reshape(size).to_dense(),
            z.to_dense(),
        )

    # 测试 reshape 在 blocked 格式张量上的操作
    def test_reshape_blocked_format(self):
        # 创建一个通道数为 C 的 Conv2d 模块 m，并转换为 MKLDNN 格式
        C = 7
        m = mkldnn_utils.to_mkldnn(torch.nn.Conv2d(C, C, 3))
        # 创建一个形状为 [1, C, 8, 8] 的随机张量 x，并转换为 MKLDNN 格式
        x = torch.randn(1, C, 8, 8).to_mkldnn()

        # 对 x 应用 m 模块，得到 y_block（MKLDNN blocked 格式张量）
        y_block = m(x)
        # 将 y_block 转换为普通张量格式，得到 y_plain
        y_plain = y_block.to_dense()

        # 对 y_block 和 y_plain 进行 reshape 操作，并比较其结果
        y_block_reshape = y_block.reshape(C, -1)
        y_plain_reshape = y_plain.reshape(C, -1)

        self.assertEqual(y_plain_reshape, y_block_reshape.to_dense())
    def test_reshape_backward(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10  # 创建一个3维张量x，形状为(3, 4, 5)，元素为标准正态分布随机数乘以10
        size = (x.size(0), -1)  # 计算size元组，第一个维度保持不变，第二个维度为-1，即自动计算

        x1 = x.clone().requires_grad_()  # 克隆张量x并启用梯度追踪
        x2 = x.clone().to_mkldnn().requires_grad_()  # 克隆张量x并将其转换为MKL-DNN格式，同时启用梯度追踪
        in_features = 20  # 定义输入特征数量
        out_features = torch.randint(3, 100, (1,)).item()  # 生成一个随机整数，作为输出特征数量
        linear = torch.nn.Linear(in_features, out_features).float()  # 创建一个线性层对象，输入特征数为in_features，输出特征数为out_features

        y1 = linear(x1.reshape(size)).sum()  # 对x1按size重新形状，并计算线性层输出的和
        y2 = linear(x2.reshape(size).to_dense()).sum()  # 将x2按size重新形状，转换为稠密格式，然后计算线性层输出的和
        y1.backward()  # 反向传播，计算梯度
        y2.backward()  # 反向传播，计算梯度
        self.assertEqual(x1.grad, x2.grad.to_dense())  # 断言：x1的梯度应该与x2的稠密格式梯度相等

    def test_clone(self):
        x = torch.randn(4, 5, dtype=torch.float32) * 10  # 创建一个2维张量x，形状为(4, 5)，元素为标准正态分布随机数乘以10
        self.assertEqual(
            x.clone(),  # 克隆张量x
            x.to_mkldnn().clone().to_dense(),  # 将x转换为MKL-DNN格式，再克隆并转换为稠密格式
        )
        # 测试是否共享同一内存
        y = x.to_mkldnn()  # 将x转换为MKL-DNN格式
        z = y.clone().add_(y)  # 克隆y，并将y加到克隆的张量z上
        self.assertNotEqual(
            y.to_dense(),  # 将y转换为稠密格式
            z.to_dense(),  # 将z转换为稠密格式
        )

    def test_transpose(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10  # 创建一个3维张量x，形状为(3, 4, 5)，元素为标准正态分布随机数乘以10
        for dim1 in range(x.ndim):  # 遍历张量x的维度
            for dim2 in range(x.ndim):  # 再次遍历张量x的维度
                self.assertEqual(
                    x.transpose(dim1, dim2),  # 对x进行维度dim1和dim2的转置操作
                    x.to_mkldnn().transpose(dim1, dim2).to_dense(),  # 将x转换为MKL-DNN格式，进行相同的转置操作，然后转换为稠密格式
                )

    def test_transpose_invalid_dime(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32).to_mkldnn()  # 创建一个MKL-DNN格式的3维张量x，形状为(3, 4, 5)，元素为标准正态分布随机数
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):  # 断言：捕获预期的IndexError异常，异常消息包含"Dimension out of range"
            torch._mkldnn_transpose(x, 0, 12)  # 调用_mkldnn_transpose函数对x进行转置，传递的维度参数为0和12

    def test_linear_non_contiguous_weight(self):
        in_features = torch.randint(3, 10, (1,)).item()  # 生成一个随机整数，作为输入特征数量
        out_features = torch.randint(3, 100, (1,)).item()  # 生成一个随机整数，作为输出特征数量
        x = torch.randn(3, in_features, dtype=torch.float32) * 10  # 创建一个3维张量x，形状为(3, in_features)，元素为标准正态分布随机数乘以10
        w = torch.randn(in_features, out_features, dtype=torch.float32)  # 创建一个2维张量w，形状为(in_features, out_features)，元素为标准正态分布随机数
        for bias in [True, False]:  # 遍历bias的可能取值True和False
            x1 = x.clone().requires_grad_()  # 克隆张量x并启用梯度追踪
            x2 = x.clone().to_mkldnn().requires_grad_()  # 克隆张量x并将其转换为MKL-DNN格式，同时启用梯度追踪
            linear = torch.nn.Linear(in_features, out_features).float()  # 创建一个线性层对象，输入特征数为in_features，输出特征数为out_features
            linear.weight = torch.nn.Parameter(w.t())  # 将线性层对象的权重设置为张量w的转置，并封装为Parameter类型
            mkldnn_linear = copy.deepcopy(linear)  # 深度复制线性层对象，得到一个新的对象mkldnn_linear
            y1 = linear(x1).sum()  # 计算线性层对象对x1的输出的和
            y2 = mkldnn_linear(x2).to_dense().sum()  # 计算MKL-DNN格式的线性层对象对x2的输出的和，并转换为稠密格式
            y1.backward()  # 反向传播，计算梯度
            y2.backward()  # 反向传播，计算梯度
            self.assertEqual(x1.grad, x2.grad.to_dense())  # 断言：x1的梯度应该与x2的稠密格式梯度相等
            self.assertEqual(linear.weight.grad, mkldnn_linear.weight.grad)  # 断言：线性层对象的权重梯度应该与MKL-DNN格式的线性层对象的权重梯度相等
            if bias:
                self.assertEqual(linear.bias.grad, mkldnn_linear.bias.grad)  # 如果有偏置项，断言：线性层对象的偏置项梯度应该与MKL-DNN格式的线性层对象的偏置项梯度相等
    # 定义一个测试方法，用于测试线性层的功能
    def test_linear(self):
        # 随机生成输入特征的数量，范围在 [3, 10) 之间
        in_features = torch.randint(3, 10, (1,)).item()
        # 随机生成输出特征的数量，范围在 [3, 100) 之间
        out_features = torch.randint(3, 100, (1,)).item()
        # 生成一个形状为 (3, in_features) 的随机输入张量，数据类型为 float32
        x = torch.randn(3, in_features, dtype=torch.float32) * 10

        # 对于每个是否包含偏置的情况进行测试
        for bias in [True, False]:
            # 创建一个普通的线性层对象，指定输入和输出特征数量，并指定是否包含偏置
            linear = torch.nn.Linear(in_features, out_features, bias=bias).float()
            # 将普通线性层转换为 MKL-DNN 线性层
            mkldnn_linear = mkldnn_utils.to_mkldnn(copy.deepcopy(linear))
            # 断言普通线性层和 MKL-DNN 线性层在相同输入上的输出是否一致
            self.assertEqual(
                linear(x),
                mkldnn_linear(x.to_mkldnn()).to_dense())

            # 测试 MKL-DNN 线性层的序列化与反序列化功能
            self._test_serialization(mkldnn_linear, (x.to_mkldnn(),))
            # 测试 MKL-DNN 线性层的追踪功能
            self._test_tracing(mkldnn_linear, (x.to_mkldnn(),))

    # 定义一个测试方法，用于测试线性层的反向传播功能
    def test_linear_backward(self):
        # 随机生成输入特征的数量，范围在 [3, 10) 之间
        in_features = torch.randint(3, 10, (1,)).item()
        # 随机生成输出特征的数量，范围在 [3, 100) 之间
        out_features = torch.randint(3, 100, (1,)).item()
        # 生成一个形状为 (3, in_features) 的随机输入张量，数据类型为 float32
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        # 对于每个是否包含偏置的情况进行测试
        for bias in [True, False]:
            # 克隆输入张量并要求计算梯度
            x1 = x.clone().requires_grad_()
            # 将输入张量克隆并转换为 MKL-DNN 张量，并要求计算梯度
            x2 = x.clone().to_mkldnn().requires_grad_()
            # 创建一个普通的线性层对象，指定输入和输出特征数量
            linear = torch.nn.Linear(in_features, out_features).float()
            # 深拷贝普通线性层对象以创建 MKL-DNN 线性层对象
            mkldnn_linear = copy.deepcopy(linear)
            # 计算普通线性层在 x1 上的输出并对其求和
            y1 = linear(x1).sum()
            # 计算 MKL-DNN 线性层在 x2 上的输出并将其转换为普通张量后求和
            y2 = mkldnn_linear(x2).to_dense().sum()
            # 对普通线性层和 MKL-DNN 线性层分别进行反向传播
            y1.backward()
            y2.backward()
            # 断言普通张量 x1 的梯度与 MKL-DNN 张量 x2 转换为普通张量后的梯度是否一致
            self.assertEqual(x1.grad, x2.grad.to_dense())
            # 断言普通线性层的权重梯度与 MKL-DNN 线性层的权重梯度是否一致
            self.assertEqual(linear.weight.grad, mkldnn_linear.weight.grad)
            # 如果包含偏置，则断言普通线性层的偏置梯度与 MKL-DNN 线性层的偏置梯度是否一致
            if bias:
                self.assertEqual(linear.bias.grad, mkldnn_linear.bias.grad)

    # 用于测试指定数据类型的装饰器
    @dtypes(torch.float16, torch.bfloat16)
    # 定义测试线性层低精度推理的方法，接受数据类型参数dtype
    def test_linear_lowp(self, dtype):
        # 随机生成输入特征数，范围为[3, 10)
        in_features = torch.randint(3, 10, (1,)).item()
        # 随机生成输出特征数，范围为[3, 100)
        out_features = torch.randint(3, 100, (1,)).item()
        # 生成标准正态分布的张量，形状为(3, in_features)，数据类型为float32，乘以10
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        # 将x转换为指定的dtype类型
        x_lowp = x.to(dtype=dtype)

        # 遍历是否包含偏置的列表
        for bias in [True, False]:
            # 创建标准的线性层，使用float数据类型
            linear = torch.nn.Linear(in_features, out_features, bias=bias).float()
            # 深拷贝线性层并转换为MKLDNN格式
            mkldnn_linear = mkldnn_utils.to_mkldnn(copy.deepcopy(linear))
            # 深拷贝线性层并转换为MKLDNN格式，并指定dtype
            mkldnn_linear_lowp = mkldnn_utils.to_mkldnn(
                copy.deepcopy(linear), dtype
            )
            # 低精度推理支持的字典，包含torch.bfloat16和torch.half的支持函数
            lowp_support = {
                torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
                torch.half: torch.ops.mkldnn._is_mkldnn_fp16_supported,
            }
            # 如果当前dtype支持低精度推理
            if lowp_support[dtype]():
                # 对输入x执行MKLDNN线性层操作，并转换为密集张量
                y = mkldnn_linear(x.to_mkldnn()).to_dense()
                # 对低精度输入x_lowp执行MKLDNN低精度线性层操作，并转换为指定的float32密集张量
                y_lowp = mkldnn_linear_lowp(x_lowp.to_mkldnn()).to_dense(
                    torch.float32
                )
                # 如果dtype为torch.bfloat16，使用指定的容差进行断言比较
                if dtype == torch.bfloat16:
                    self.assertEqual(y, y_lowp, atol=1e-1, rtol=1e-3)
                # 否则使用另一组容差进行断言比较
                else:
                    self.assertEqual(y, y_lowp, atol=5e-3, rtol=1e-3)
            # 如果当前dtype不支持低精度推理
            else:
                # 错误消息字典，根据dtype提供相应的错误信息
                msg = {
                    torch.bfloat16: r"bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq",
                    torch.half: r"fp16 path needs the cpu support avx_ne_convert or avx512_fp16",
                }
                # 断言引发的异常消息与msg[dtype]相匹配
                self.assertRaisesRegex(
                    RuntimeError,
                    msg[dtype],
                    lambda: mkldnn_linear_lowp(x_lowp.to_mkldnn()),
                )

    # 定义测试softmax操作的方法
    def test_softmax(self):
        # 生成标准正态分布的张量，形状为(3, 4, 5)，数据类型为float32，乘以10
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        # 遍历张量的维度
        for dim in range(x.ndim):
            # 创建softmax层，指定维度dim
            softmax = torch.nn.Softmax(dim=dim)
            # 断言MKLDNN和标准操作的结果张量相等
            self.assertEqual(
                softmax(x),
                softmax(x.to_mkldnn()).to_dense())

    # 定义测试sigmoid操作的方法
    def test_sigmoid(self):
        # 生成标准正态分布的张量，形状为(4, 5)，数据类型为float32，乘以10
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        # 将x转换为MKLDNN格式
        mkldnn_x = x.to_mkldnn()
        # 断言MKLDNN和标准操作的sigmoid结果张量相等
        self.assertEqual(
            torch.sigmoid(x),
            torch.sigmoid(mkldnn_x).to_dense(),
        )
        # 原地操作，对x和mkldnn_x执行sigmoid操作后，再次断言它们的值相等
        torch.sigmoid_(x)
        torch.sigmoid_(mkldnn_x)
        self.assertEqual(x, mkldnn_x.to_dense())

    # 定义测试tanh操作的方法
    def test_tanh(self):
        # 生成标准正态分布的张量，形状为(4, 5)，数据类型为float32，乘以10
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        # 将x转换为MKLDNN格式
        mkldnn_x = x.to_mkldnn()
        # 断言MKLDNN和标准操作的tanh结果张量相等
        self.assertEqual(
            torch.tanh(x),
            torch.tanh(mkldnn_x).to_dense(),
        )
        # 原地操作，对x和mkldnn_x执行tanh操作后，再次断言它们的值相等
        torch.tanh_(x)
        torch.tanh_(mkldnn_x)
        self.assertEqual(x, mkldnn_x.to_dense())

    # 定义测试模型序列化的方法，接受模块和输入作为参数
    def _test_serialization(self, module, inputs):
        # 使用临时文件名执行以下操作
        with TemporaryFileName() as fname:
            # 将模块保存到文件
            torch.jit.save(module, fname)
            # 从文件加载模块
            loaded = torch.jit.load(fname)
            # 断言原始模块和加载后的模块对输入执行操作后的结果张量相等
            self.assertEqual(
                module(*inputs).to_dense(),
                loaded(*inputs).to_dense())
    def _test_tracing(self, module, inputs):
        # 使用 torch.jit.trace 方法对模块进行追踪，生成追踪对象
        traced = torch.jit.trace(module, inputs)
        # 断言追踪对象和原始模块在相同输入下的输出是否一致
        self.assertEqual(
            module(*inputs).to_dense(),
            traced(*inputs).to_dense())

    def test_set_data_tensorimpl_type(self):
        # 创建一个普通的密集张量，其实现类型为 `TensorImpl`
        x = torch.randn((1, 2), dtype=torch.float, device=torch.device('cpu'))
        # 将该张量转换为 MKL-DNN 张量
        x_mkldnn = x.to_mkldnn()
        # 使用断言检查赋值操作是否会引发 RuntimeError 异常，并验证异常信息
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_mkldnn

    def test_empty(self):
        # 创建一个普通的未初始化张量，指定形状和数据类型
        x1 = torch.empty(4, 5, 2, 3, dtype=torch.float32)
        # 创建一个 MKL-DNN 张量，指定形状、数据类型和布局
        x2 = torch.empty(4, 5, 2, 3, dtype=torch.float32, layout=torch._mkldnn)
        # 使用断言验证两个张量的形状是否相同
        self.assertEqual(x1.size(), x2.to_dense().size())
        # 使用断言验证两个张量的数据类型是否相同
        self.assertEqual(x1.dtype, x2.to_dense().dtype)

    def test_zero_(self):
        # 创建一个普通的随机张量，并乘以 10
        x1 = torch.randn(4, 5, dtype=torch.float32) * 10
        # 克隆该张量并转换为 MKL-DNN 张量
        x2 = x1.clone().to_mkldnn()
        # 使用断言验证普通张量调用 zero_() 方法后的输出是否与 MKL-DNN 张量调用 zero_() 方法后的结果一致
        self.assertEqual(
            x1.zero_(),
            x2.zero_().to_dense(),
        )

    def test_is_mkldnn(self):
        # 创建一个普通的随机张量
        x = torch.randn(1, dtype=torch.float32)
        # 使用断言验证普通张量不是 MKL-DNN 张量
        self.assertFalse(x.is_mkldnn)
        # 将普通张量转换为 MKL-DNN 张量，并使用断言验证结果
        self.assertTrue(x.to_mkldnn().is_mkldnn)

    # legacy constructor/new doesn't support mkldnn tensors
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1992")
    def test_legacy_new_failure(self):
        # 创建一个普通的随机张量
        x = torch.randn(1, dtype=torch.float32)
        # 将普通张量转换为 MKL-DNN 张量
        x_mkldnn = x.to_mkldnn()
        # 使用 lambda 函数和断言验证使用不支持 MKL-DNN 张量的构造函数和 new 方法是否会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(device='cpu'))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(x.storage()))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(x))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(torch.Size([2, 3])))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new([6]))

    def test_is_mkldnn_jit(self):
        # 创建一个 TorchScript 模块，确保输入张量为 MKL-DNN 张量
        class EnsureMkldnn(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if not x.is_mkldnn:
                    x = x.to_mkldnn()
                return x

        m = EnsureMkldnn()
        x = torch.randn(1, dtype=torch.float32)
        # 使用断言验证模块对普通张量和 MKL-DNN 张量的处理结果
        self.assertTrue(m(x).is_mkldnn)
        self.assertTrue(m(x.to_mkldnn()).is_mkldnn)

    def _test_imagenet_model(self, model):
        # 设置模型为推断模式并转换为 float 类型
        model = model.train(False).float()
        # 将模型转换为 MKL-DNN 模型
        mkldnn_model = mkldnn_utils.to_mkldnn(copy.deepcopy(model))
        # 创建一个随机输入张量
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        # 使用断言验证普通模型和 MKL-DNN 模型在相同输入下的输出是否一致
        with torch.no_grad():
            self.assertEqual(
                model(x),
                mkldnn_model(x.to_mkldnn()).to_dense(),
            )

    @skipIfNoTorchVision
    def test_resnet18(self):
        # 创建一个 ResNet-18 模型
        model = torchvision.models.resnet.resnet18(weights=None)
        # 测试图像模型
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    # 定义一个测试函数，用于测试 ResNeXt-50 32x4d 模型
    def test_resnext50_32x4d(self):
        # 创建一个未加载权重的 ResNeXt-50 32x4d 模型
        model = torchvision.models.resnet.resnext50_32x4d(weights=None)
        # 调用内部函数以测试该模型在 ImageNet 上的表现
        self._test_imagenet_model(model)

    # 定义一个返回 LSTM 参数列表的内部函数
    def _lstm_params_list(self):
        # 定义包含各种 LSTM 参数选择的字典
        params_dict = {
            "input_size": [1, 5],
            "hidden_size": [5, 16],
            "num_layers": [1, 3],
            "bidirectional": [False, True],
            "bias": [False, True],
            "batch_first": [False, True],
            "dropout": [0, 0.4, 0.7, 1],
            "batch_size": [1, 2],
            "seq_len": [1, 3],
            "training": [False, True]
        }

        # 将参数字典的值转换为列表并返回
        params_list = list(params_dict.values())
        return params_list

    # 定义一个函数，根据 bf16 参数将输入张量转换为指定数据类型
    def _cast_dtype(self, input, bf16):
        if bf16:
            # 如果 bf16 为真，则将输入张量转换为 torch.bfloat16 类型
            input = input.to(torch.bfloat16)
        return input

    # 使用 unittest 模块的装饰器标记，当 IS_WINDOWS 为真时跳过测试
    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    # 使用 dtypes 装饰器标记，指定测试的数据类型为 torch.float16 和 torch.bfloat16
    @dtypes(torch.float16, torch.bfloat16)
    # 定义一个测试函数，用于测试矩阵乘法在低精度下的表现
    def test_matmul_lower_precision(self, dtype):
        # 定义检查支持函数的字典，用于检测当前数据类型的 MKL-DNN 支持情况
        support_check = {
            torch.bfloat16: torch.ops.mkldnn._is_mkldnn_bf16_supported,
            torch.float16: torch.ops.mkldnn._is_mkldnn_fp16_supported,
        }

        # 定义一个共用函数，用于执行矩阵操作并验证结果
        def common(self, shape1, shape2, op, dtype):
            # 生成指定形状和数据类型的随机张量 a
            a = torch.randn(shape1, dtype=dtype)
            # 创建 a 的浮点数版本作为参考
            a_ref = a.float()
            # 生成指定形状和数据类型的随机张量 b
            b = torch.randn(shape2, dtype=dtype)
            # 创建 b 的浮点数版本作为参考
            b_ref = b.float()

            # 使用给定的操作符 op 对张量 a 和 b 执行操作
            y = op(a, b)
            # 使用浮点数版本的 a_ref 和 b_ref 执行相同的操作
            y_ref = op(a_ref, b_ref)
            # 断言两个操作的结果相等，不需要严格相等（exact_dtype=False）
            self.assertEqual(y, y_ref, exact_dtype=False)

        # 如果当前数据类型在当前环境中受支持，则执行以下测试
        if support_check[dtype]():
            # 生成形状为 [64, 1, 33] 的随机张量 a1
            a1 = torch.randn([64, 1, 33], dtype=dtype)
            # 创建一个连续的张量 a2，但其步幅不是默认的连续步幅
            a2 = torch.as_strided(a1.clone(), [64, 1, 33], [33, 3, 1])
            # 断言 a2 是否是连续的张量
            self.assertTrue(a2.is_contiguous())
            # 生成形状为 [64, 33, 256] 的随机张量 b，并将其转换为指定的数据类型
            b = torch.randn(64, 33, 256).to(dtype=dtype)
            # 使用 torch.ops.aten.bmm 执行矩阵乘法 y1 = a1 @ b
            y1 = torch.ops.aten.bmm(a1, b)
            # 使用 torch.bmm 执行相同形状的操作 y2 = a2 @ b
            y2 = torch.bmm(a2, b)
            # 断言两种方法得到的结果 y1 和 y2 相等
            self.assertEqual(y1, y2)

            # 对以下形状和操作符进行测试，使用共用函数执行相同的操作
            for shape1, shape2, op in [
                ((33, 77), (77, 22), torch.matmul),
                ((128, 256), (256, 10), torch.matmul),
                ((7, 300), (300, 3), torch.matmul),
                ((1, 100), (100, 60), torch.matmul),
                ((100, 1), (1, 100), torch.matmul),
                ((20, 54, 78), (20, 78, 10), torch.bmm),
                ((1, 300, 1), (1, 1, 300), torch.bmm),
            ]:
                common(self, shape1, shape2, op, dtype)
# 使用指定测试类 TestMkldnn 实例化设备类型测试，并将其注册到全局作用域中
instantiate_device_type_tests(TestMkldnn, globals(), only_for=('cpu',))

# 如果当前脚本作为主程序执行，则运行所有的测试用例
if __name__ == '__main__':
    run_tests()
```