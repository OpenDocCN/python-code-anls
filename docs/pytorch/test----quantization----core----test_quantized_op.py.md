# `.\pytorch\test\quantization\core\test_quantized_op.py`

```
# Owner(s): ["oncall: quantization"]

# 导入必要的模块和库
import copy  # 导入深拷贝模块
import itertools  # 导入迭代工具模块
import numpy as np  # 导入 NumPy 库，并命名为 np
import operator  # 导入运算符模块
import random  # 导入随机数模块
import unittest  # 导入单元测试模块
from typing import NamedTuple, List  # 导入类型提示 NamedTuple 和 List

import torch  # 导入 PyTorch 深度学习库
from torch import _VF  # 导入 PyTorch 私有模块 _VF
import torch.jit  # 导入 PyTorch JIT 模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块 F
from torch.nn.modules.utils import _single, _pair  # 导入 PyTorch 中的单个和对组工具函数

from hypothesis import settings, HealthCheck  # 导入 Hypothesis 的设置和健康检查
from hypothesis import assume, given, note  # 导入假设、给定和笔记函数
from hypothesis import strategies as st  # 导入策略模块，并命名为 st
import torch.testing._internal.hypothesis_utils as hu  # 导入 PyTorch 内部的假设工具模块 hu
hu.assert_deadline_disabled()  # 禁用假设工具的截止时间检查

from torch.testing._internal.common_cuda import SM80OrLater  # 导入测试 CUDA 的工具 SM80OrLater
from torch.testing._internal.common_utils import TestCase  # 导入测试用例模块中的 TestCase 类
from torch.testing._internal.common_utils import IS_PPC, TEST_WITH_UBSAN, IS_MACOS, IS_SANDCASTLE  # 导入平台相关的常量
from torch.testing._internal.common_quantization import skipIfNoFBGEMM, skipIfNoQNNPACK, skipIfNoONEDNN  # 导入量化相关的跳过装饰器和常量
from torch.testing._internal.common_quantized import _quantize, _dequantize, _calculate_dynamic_qparams, \
    override_quantized_engine, supported_qengines, override_qengines, _snr  # 导入量化相关的函数和装饰器

from torch.testing._internal.common_quantized import (
    qengine_is_qnnpack,  # 检查是否使用 QNNPACK 引擎
    qengine_is_onednn,  # 检查是否使用 OneDNN 引擎
)
from torch.ao.quantization import PerChannelMinMaxObserver  # 导入分通道最小最大观察器
from torch.testing._internal.common_cuda import TEST_CUDNN, TEST_CUDNN_VERSION, TEST_CUDA  # 导入 CUDA 测试相关常量
from torch.testing._internal.optests import opcheck  # 导入操作测试函数 opcheck
import torch.backends.xnnpack  # 导入 XNNPack 后端

from torch.utils.cpp_extension import ROCM_HOME  # 导入 ROCM_HOME 变量

from typing import Optional  # 导入可选类型

np_dtype = {
    torch.quint8 : np.uint8,  # 定义 PyTorch quint8 类型对应的 NumPy uint8 类型
    torch.qint8 : np.int8,  # 定义 PyTorch qint8 类型对应的 NumPy int8 类型
    torch.qint32 : np.int32  # 定义 PyTorch qint32 类型对应的 NumPy int32 类型
}

TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None  # 检查是否在 ROCm 环境下进行测试

class PointwisePostOp(NamedTuple):
    binary_attr : str = "none"  # 定义命名元组 PointwisePostOp 的二进制属性，默认为 "none"
    alpha : float = 1.0  # 定义命名元组 PointwisePostOp 的 alpha 属性，默认为 1.0
    unary_attr : str = "none"  # 定义命名元组 PointwisePostOp 的一元属性，默认为 "none"
    scalars : List = []  # 定义命名元组 PointwisePostOp 的标量列表，默认为空列表
    algorithm : str = ""  # 定义命名元组 PointwisePostOp 的算法属性，默认为空字符串

# 确保我们不会在 FBGEMM 使用的 vpmaddubsw 指令中发生溢出。
# 在当前的 Intel x86 架构上，我们需要利用 vpmaddubsw 指令进行 8 位整数乘法。
# 此指令垂直地将每个来自 a 的无符号 8 位整数与来自 b 的相应有符号 8 位整数相乘，产生中间有符号 16 位整数。
# 此函数修改权重以消除有符号 16 位整数的溢出。
def avoid_vpmaddubsw_overflow_linear(
    batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max
):
    # 遍历 np.ndindex 的迭代器，其中 i 为批处理索引，j 为输出通道索引
    for i, j in np.ndindex((batch_size, output_channels)):
        # 遍历每对输入通道索引 k，范围为 0 到 input_channels // 2 * 2（偶数范围）
        for k in range(0, input_channels // 2 * 2, 2):
            # 计算输入张量 X 中的两个值相对于 X_min 的偏移量
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            # 计算权重张量 W 中的两个值减去 128 和 W_min 后的偏移量
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            # 检查是否存在乘积溢出，如果是，则调整权重 W[j, k + 1] 的值
            if x0 * w0 + x1 * w1 < -(1 << 15):
                # 计算调整后的 w1 的值，确保乘积不小于 -(1 << 15)
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                # 更新 W[j, k + 1] 的值，加上 128 和 W_min
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 > (1 << 15) - 1:
                # 计算调整后的 w1 的值，确保乘积不大于 (1 << 15) - 1
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                # 更新 W[j, k + 1] 的值，加上 128 和 W_min
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
    
    # 再次遍历相同的循环以确保没有溢出问题
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            # 使用断言确认乘积不超出 -(1 << 15) 到 (1 << 15) - 1 的范围
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < (1 << 15)
# Reference quantized Linear operator
# 参考量化的线性操作符函数
def qlinear_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp, dtype=np.uint8):
    # 将输入量化的数据 X_q 重新形状为二维数组
    X_q = np.reshape(X_q, (-1, X_q.shape[X_q.ndim - 1]))
    # 计算 X_q 每行的和，并转换为整数类型作为行偏移量
    row_offsets_ref = X_q.sum(axis=1).astype(np.int32).reshape((-1, 1))
    # 计算 W_q 每行的和，并转换为整数类型作为列偏移量
    col_offsets_ref = W_q.sum(axis=1).astype(np.int32).reshape((1, -1))
    # 断言 X_q 必须是二维数组
    assert X_q.ndim == 2
    # 获取批量大小和输入通道数
    batch_size, input_channels = X_q.shape
    # 计算量化后的乘积 Prod_XqWq_ref
    Prod_XqWq_ref = (
        np.matmul(X_q.astype(np.int32), W_q.astype(np.int32).T)
        - W_zp * row_offsets_ref
        - X_zp * col_offsets_ref
        + input_channels * X_zp * W_zp
    )
    # 如果存在偏置 b_q，则加上偏置
    if b_q is not None:
        Prod_XqWq_ref += b_q
    # 对乘积结果进行量化得到输出 Y_q_ref
    Y_q_ref = _quantize(Prod_XqWq_ref, Y_scale / (X_scale * W_scale), Y_zp, dtype=dtype)
    # 返回量化后的输出结果
    return Y_q_ref

"""Computes the output shape given pooling parameters."""
# 根据池化参数计算输出形状
def pool_output_shape(input_size, kernel_size, padding, stride,
                      dilation, ceiling_mode=False):
    # 如果未指定步长，则默认为核大小
    if stride is None:
        stride = kernel_size
    # 计算输出大小
    output_size = (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
         + (stride - 1 if ceiling_mode else 0)) // stride + 1)
    # 如果启用了天花板模式，并且计算后的输出大小大于等于输入大小加上填充，则减小输出大小
    if (ceiling_mode and
            ((output_size - 1) * stride >= input_size + padding)):
        output_size -= 1
    # 返回计算得到的输出大小
    return output_size

"""
Util for creating a random tensor and quantization params when Hypothesis
is undesirable.
"""
# 在假设不可取时，用于创建随机张量和量化参数的实用函数
def _get_random_tensor_and_q_params(shapes, rand_scale, torch_type):
    # 创建随机张量 X
    X = (torch.rand(*shapes, dtype=torch.float) - 0.5) * rand_scale
    # 计算合理的量化参数
    min_val = torch.min(X)
    max_val = torch.max(X)
    if torch_type == torch.qint32:
        X_zero_point = int(torch.randint(-1 * (2 ** 31), 2 ** 31 - 1, (1,)))
        num_bins = 2 ** 32
        X_scale = float(max_val - min_val) / num_bins
    elif torch_type == torch.qint8:
        X_zero_point = int(torch.randint(-128, 127, (1,)))
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    else:  # torch.quint8
        X_zero_point = 127
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    if X_scale == 0:
        X_scale = 1e-10
    # 返回随机张量 X、量化比例 X_scale 和零点 X_zero_point
    return X, X_scale, X_zero_point

class TestQuantizedOps(TestCase):

    """Helper function to test quantized activation functions."""
    """Tests the correctness of the quantized::relu op."""
    @override_qengines
    """Tests the correctness of the quantized::relu op."""
    def test_qrelu(self):
        # 定义了用于测试 quantized relu 函数的配置列表
        relu_test_configs = [
            {
                'quantized_fn': [
                    torch.relu,
                    torch.relu_,
                    torch.nn.functional.relu,
                    torch.nn.functional.relu,
                ],
                'reference_fn': torch.nn.functional.relu
            },
            {
                'quantized_fn': [
                    torch.nn.functional.relu,
                    torch.nn.functional.relu,
                ],
                'reference_fn': torch.nn.functional.relu,
                'extra_kwargs': {
                    'inplace': True
                }
            }
        ]
        # 根据是否支持 CUDA，选择测试设备为 CPU 或 CPU 和 CUDA
        devices = ["cpu", "cuda"] if TEST_CUDA else ["cpu"]
        for device in devices:
            # 定义了用于测试的不同张量形状
            shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
            # 定义了用于测试的不同张量数据类型
            dtypes = (torch.quint8, torch.qint8)
            # 定义了用于测试的不同量化比例因子
            scales = (0.05, 0.1)
            # 定义了用于测试的不同零点偏移值
            zero_points = (0, 5)
            # 生成所有测试用例的组合
            test_cases = itertools.product(shapes, dtypes, scales, zero_points)
            for shape, dtype, scale, zero_point in test_cases:
                # 创建随机张量并与量化参数组合
                X = torch.randn(*shape, device=device)
                X = (X, (scale, zero_point, dtype))
                # 调用测试激活函数的私有方法，测试 relu 函数
                self._test_activation_function(X, 'relu', relu_test_configs)

    """Tests the correctness of the quantized::relu6 op."""
    def test_qrelu6(self):
        # 定义了用于测试 quantized relu6 函数的配置列表
        relu6_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.relu6,
                    torch.ao.nn.quantized.ReLU6(inplace=False),
                    torch.ao.nn.quantized.ReLU6(inplace=True)
                ],
                'reference_fn': torch.nn.functional.relu6
            }
        ]
        # 定义了用于测试的不同张量形状
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        # 定义了用于测试的不同张量数据类型
        dtypes = (torch.quint8, torch.qint8)
        # 定义了用于测试的不同量化比例因子
        scales = (0.05, 0.1)
        # 定义了用于测试的不同零点偏移值
        zero_points = (0, 5)
        # 生成所有测试用例的组合
        test_cases = itertools.product(shapes, dtypes, scales, zero_points)
        for shape, dtype, scale, zero_point in test_cases:
            # 创建随机张量并与量化参数组合
            X = torch.randn(*shape) * 10
            X = (X, (scale, zero_point, dtype))
            # 调用测试激活函数的私有方法，测试 relu6 函数
            self._test_activation_function(X, 'relu6', relu6_test_configs)

    """Tests the correctness of the quantized::sigmoid op."""
    @override_qengines
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_sigmoid_non_observed(self, X):
        # 定义了用于测试 quantized sigmoid 函数的配置列表
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True
            }
        ]
        # 调用测试激活函数的私有方法，测试 sigmoid 函数
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Tests the correctness of the quantized::sigmoid op."""
    # TODO: enable after observed output is supported in qnnpack
    # @override_qengines
    @skipIfNoFBGEMM
    """Defines a test case for the sigmoid activation function.

    Uses hypothesis strategies to generate input tensors X and tests the quantized
    and reference implementations of the sigmoid function across various configurations.
    """
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_sigmoid(self, X):
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'output_is_observed': True,
            }
        ]
        # Calls a helper function to validate the sigmoid activation function
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Defines a test case for sigmoid dequantization rounding error.

    Skips the test if FBGEMM is not available. This test addresses issue #107030
    and checks the quantized and reference implementations of the sigmoid function
    under specific configurations.
    """
    @skipIfNoFBGEMM
    def test_sigmoid_dequantize_rounding_error(self):
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'output_is_observed': True,
            }
        ]
        # Sets up specific input X and calls a helper function to test the sigmoid function
        X = (np.full(64, 514., dtype=np.float32), (1028.02, 255, torch.quint8))
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Tests the correctness of the quantized::hardsigmoid op.

    Overrides quantization engines for the test. This test evaluates the quantized
    and reference implementations of the hardsigmoid function across multiple configurations.
    """
    @override_qengines
    def test_qhardsigmoid(self):
        hardsigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ao.nn.quantized.functional.hardsigmoid,
                ],
                'reference_fn': torch.nn.functional.hardsigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
            },
            {
                'quantized_fn': [
                    torch.ao.nn.quantized.functional.hardsigmoid,
                ],
                'reference_fn': torch.nn.functional.hardsigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'extra_kwargs': {
                    'inplace': True,
                },
            },
        ]
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        test_cases = itertools.product(shapes, dtypes)
        # Iterates over test cases, generating random input X and calling a helper function
        for shape, dtype in test_cases:
            X = (np.random.rand(*shape).astype(np.float32), (1.0, 0, dtype))
            self._test_activation_function(X, 'hardsigmoid', hardsigmoid_test_configs)

    """Defines a test case for the sigmoid activation function.

    Overrides quantization engines and uses hypothesis strategies to generate input tensors X.
    """
    @override_qengines
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    """Tests the correctness of the quantized::relu op."""
    def test_leaky_relu(self):
        # 定义输入数据的形状、数据类型和内存布局格式的组合
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        
        # 使用itertools生成所有可能的测试用例
        test_cases = itertools.product(shapes, dtypes, memory_formats)
        
        # 遍历每一个测试用例
        for shape, dtype, memory_format in test_cases:
            # 如果内存布局格式是channels_last但形状不是4维，则跳过当前测试用例
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            
            # 生成随机输入数据，并根据内存布局格式进行转换
            X, scale, zero_point, torch_type, alpha = \
                torch.randn(*shape), 0.1, 0, dtype, 0.01
            X = X.to(memory_format=memory_format)
            
            # 对输入数据进行量化和反量化操作
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            dqX = qX.dequantize()
            
            # 使用torch.nn.functional中的leaky_relu函数进行操作
            op = torch.nn.functional.leaky_relu
            dqY = op(dqX, negative_slope=alpha)
            
            # 对输出进行再次量化
            qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            
            # 使用leaky_relu函数对量化后的输入数据进行操作
            qY_hat = op(qX, negative_slope=alpha)
            
            # 断言两者的反量化结果是否一致，用于判断leaky_relu函数的正确性
            self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                             msg=f"F.leaky_relu failed ({qY} vs {qY_hat})")
    """Tests the correctness of the quantized::elu op."""
    # 定义用于测试 quantized::elu 操作的函数
    def test_qelu(self, X, alpha):
        # 解包输入数据 X 的元组，包括数据本身，以及量化参数 scale, zero_point 和 torch_type
        X, (scale, zero_point, torch_type) = X
        # 设置输出的量化参数
        output_scale = 0.5
        output_zero_point = 1

        # 将 numpy 数组 X 转换为 Torch 张量
        X = torch.from_numpy(X)
        # 对输入张量 X 进行量化
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 计算 ELU(dqX) 并进行量化
        dqX = qX.dequantize()
        dqY_hat = torch.nn.functional.elu(dqX, alpha)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale, zero_point=output_zero_point,
                                           dtype=torch_type)

        # 使用 quantized::elu 操作计算 qX 的 ELU，结果存储在 qY 中
        qY = torch.ao.nn.quantized.functional.elu(qX, output_scale, output_zero_point, alpha=alpha)
        # 断言量化 ELU 的结果与预期的量化结果 qY_hat 相等，否则输出失败信息
        self.assertEqual(qY, qY_hat,
                         msg=f"F.elu failed ({qY} vs {qY_hat})")


    """Tests the correctness of the quantized::celu op."""
    # 定义用于测试 quantized::celu 操作的函数
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e2, 1e2, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(scale_max=9.999999747378752e-06)),
           alpha=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False))
    def test_qcelu(self, X, alpha):
        # 解包输入数据 X 的元组，包括数据本身，以及量化参数 scale, zero_point 和 torch_type
        X, (scale, zero_point, torch_type) = X
        # 设置输出的量化参数
        output_scale = 0.5
        output_zero_point = 1

        # 将 numpy 数组 X 转换为 Torch 张量
        X = torch.from_numpy(X)
        # 对输入张量 X 进行量化
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 计算 CELU(dqX) 并进行量化
        dqX = qX.dequantize()
        dqY_hat = torch.nn.functional.celu(dqX, alpha)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale, zero_point=output_zero_point,
                                           dtype=torch_type)

        # 使用 quantized::celu 操作计算 qX 的 CELU，结果存储在 qY 中
        qY = torch.ops.quantized.celu(qX, output_scale, output_zero_point, alpha=alpha)
        # 断言量化 CELU 的结果与预期的量化结果 qY_hat 相等，否则输出失败信息
        self.assertEqual(qY, qY_hat,
                         msg=f"F.celu failed ({qY} vs {qY_hat})")

    """Tests the correctness of the quantized::gelu op."""
    # 定义一个测试方法，用于测试 quantized::gelu 操作的正确性
    def test_qgelu(self):
        # 定义不同的输入形状
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        # 定义不同的数据类型，包括量化的整数类型
        dtypes = (torch.quint8, torch.qint8)
        # 定义不同的内存布局格式
        memory_formats = (torch.channels_last, torch.contiguous_format)
        # 定义近似方法的选择，包括 'none' 和 'tanh'
        approximation = ['none', 'tanh']
        # 使用 itertools 生成所有测试用例的组合
        test_cases = itertools.product(shapes, dtypes, memory_formats, approximation)
        # 根据是否支持 CUDA 选择设备列表
        devices = ["cpu", "cuda"] if TEST_CUDA else ["cpu"]
        # 遍历所有测试用例
        for shape, dtype, memory_format, approximate in test_cases:
            # 如果内存布局格式为 channels_last 且形状长度不为 4，则跳过该测试用例
            if memory_format == torch.channels_last and len(shape) != 4:
                continue

            # 生成随机张量 X，设置量化参数 scale、zero_point 和数据类型 torch_type
            X, scale, zero_point, torch_type = \
                torch.randn(*shape), 0.1, 0, dtype
            # 将张量 X 转换为指定的内存布局格式
            X = X.to(memory_format=memory_format)
            # 遍历所有设备进行测试
            for device in devices:
                # 将张量 X 移动到指定设备上
                X = X.to(device=device)
                # 对张量 X 进行量化
                qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)
                # 对量化后的张量进行反量化
                dqX = qX.dequantize()

                # 定义操作为 torch.nn.functional.gelu
                op = torch.nn.functional.gelu
                # 对反量化后的张量应用 gelu 操作，使用指定的近似方法
                dqY = op(dqX, approximate=approximate)
                # 对 gelu 后的结果进行量化
                qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)
                # 使用量化的张量 qX 应用 gelu 操作
                qY_hat = op(qX)
                # 断言量化后的结果与直接应用 gelu 操作的结果相等，用于检验 gelu 操作的正确性
                self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                                 msg=f"F.gelu failed ({qY} vs {qY_hat})")

    # 测试 quantized::prelu 操作的正确性
    """Tests the correctness of the quantized::prelu op."""
    """Tests the correctness of the quantized::qlayer_norm op."""
    # 定义一个测试方法，用于测试量化操作的qlayer_norm操作的正确性

    @skipIfNoFBGEMM
    # 如果没有FBGEMM，跳过这个测试

    """Tests the correctness of the quantized::qnnpack_tanh op."""
    # 测试量化操作的qnnpack_tanh操作的正确性

    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    # 使用hypothesis库定义一个参数化测试，对于X张量，形状从(1, 5, 1, 5)中选择，使用hu.qparams()作为参数

    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    # 使用unittest库的skip装饰器，跳过这个测试，并提供一个消息解释当前测试不可用，需要在CI中删除hypothesis测试
    """Tests the correctness of the quantized::clamp op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8, max_numel=10**5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False))
    # 定义测试函数，用于测试量化后的 clamp 操作的正确性
    def test_qclamp(self, X, min_val, max_val):
        X, (scale, zero_point, torch_type) = X
        # 将 numpy 数组转换为 PyTorch 张量
        X = torch.from_numpy(X)
        # 对输入张量进行量化
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 使用 clamp 函数对反量化后的张量进行截断
        dqX = qX.dequantize()
        dqY_hat = torch.nn.functional.clamp(dqX, min=min_val, max=max_val)
        # 将截断后的张量再次量化
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        # 定义待测试的 clamp 操作及其名称
        ops_under_test = {
            'native': torch.clamp,
            'nn.functional': torch.nn.functional.clamp,
            'ao.nn.quantized.functional': torch.ao.nn.quantized.functional.clamp,
        }

        # 遍历不同的 clamp 实现进行测试
        for name, op in ops_under_test.items():
            qY = op(qX.dequantize(), min=min_val, max=max_val)
            self.assertEqual(qY, qY_hat, msg=f"{name} qclamp failed")
    # 定义一个测试方法，用于测试量化夹紧操作的正确性
    def test_qclamp(self, X, min_val, max_val):
        # 解构元组 X，分别获取数据和元数据 (scale, zero_point, torch_type)
        X, (scale, zero_point, torch_type) = X

        # 假设最小值 min_val 小于等于最大值 max_val
        assume(min_val <= max_val)
        
        # 使用 torch.clamp 函数对从 NumPy 数组 X 转换而来的张量进行夹紧操作，限制在 min_val 和 max_val 之间
        Y_clamp = torch.clamp(torch.from_numpy(X), min=min_val, max=max_val)
        
        # 对夹紧后的张量进行量化操作，使用 scale、zero_point 和 torch_type 参数
        qY_clamp = torch.quantize_per_tensor(Y_clamp, scale=scale,
                                             zero_point=zero_point, dtype=torch_type)

        # 再次将 X 转换为张量
        X = torch.from_numpy(X)
        
        # 对 X 进行量化操作，使用相同的 scale、zero_point 和 torch_type 参数
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        
        # 定义包含被测试操作的字典 ops_under_test
        ops_under_test = {
            'ops.quantized': torch.ops.quantized.clamp,
        }

        # 遍历 ops_under_test 字典中的每个操作及其名称
        for name, op in ops_under_test.items():
            # 使用 torch.ops.quantized.clamp 运行量化夹紧操作，输入 qX、min_val 和 max_val 参数
            qY_clamp_hat = op(qX, min=min_val, max=max_val)
            
            # 断言量化夹紧操作的输出 qY_clamp_hat 与预期的 qY_clamp 相等，若不等则输出自定义消息
            self.assertEqual(qY_clamp, qY_clamp_hat, msg=f"{name} qclamp failed")

        # 如果使用的量化引擎是 'fbgemm'
        if torch.backends.quantized.engine == 'fbgemm':
            # 通过 override_quantized_engine 设置当前的量化引擎为 'fbgemm'
            with override_quantized_engine('fbgemm'):
                # 分别对 X 进行最小值夹紧和最大值夹紧操作，使用 min_val 和 max_val 参数
                Y_min_clamp = torch.clamp(X, min=min_val)
                Y_max_clamp = torch.clamp(X, max=max_val)

                # 将夹紧后的张量分别进行量化操作，使用 scale、zero_point 和 torch_type 参数
                qY_min_clamp = torch.quantize_per_tensor(Y_min_clamp, scale=scale,
                                                         zero_point=zero_point, dtype=torch_type)
                qY_max_clamp = torch.quantize_per_tensor(Y_max_clamp, scale=scale,
                                                         zero_point=zero_point, dtype=torch_type)

                # 再次遍历 ops_under_test 字典中的每个操作及其名称
                for name, op in ops_under_test.items():
                    # 使用 torch.ops.quantized.clamp 运行量化夹紧操作，输入 qX 和 min_val 参数
                    qY_min_clamp_hat = op(qX, min=min_val)
                    # 断言量化夹紧操作的输出 qY_min_clamp_hat 与预期的 qY_min_clamp 相等，若不等则输出自定义消息
                    self.assertEqual(qY_min_clamp, qY_min_clamp_hat, msg=f"{name} qclamp failed")
                    
                    # 使用 torch.ops.quantized.clamp 运行量化夹紧操作，输入 qX 和 max_val 参数
                    qY_max_clamp_hat = op(qX, max=max_val)
                    # 断言量化夹紧操作的输出 qY_max_clamp_hat 与预期的 qY_max_clamp 相等，若不等则输出自定义消息
                    self.assertEqual(qY_max_clamp, qY_max_clamp_hat, msg=f"{name} qclamp failed")
    # 定义测试方法，用于测试 quantized::hardtanh 操作的正确性
    def test_hardtanh(self, X, min_val, max_val):
        # 使用 'fbgemm' 引擎覆盖默认的量化引擎上下文
        with override_quantized_engine('fbgemm'):
            # 解构 X 元组，获取数据和量化相关参数
            X, (scale, zero_point, torch_type) = X

            # 假设最小值 min_val 小于等于最大值 max_val
            assume(min_val <= max_val)

            # 复制 X 数据到 Y
            Y = X.copy()
            # 将 Y 中小于 min_val 的值置为 min_val
            Y[Y < min_val] = min_val
            # 将 Y 中大于 max_val 的值置为 max_val
            Y[Y > max_val] = max_val

            # 将 Y 转换为 torch 张量，并进行量化为 qY
            qY = torch.quantize_per_tensor(torch.from_numpy(Y), scale=scale,
                                           zero_point=zero_point, dtype=torch_type)
            # 将 X 转换为 torch 张量，并进行量化为 qX
            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

            # 待测试的操作字典，包含非原地操作的 hardtanh
            ops_under_test = {
                'ao.nn.quantized.functional.hardtanh':
                    torch.ao.nn.quantized.functional.hardtanh,
            }

            # 遍历 ops_under_test 字典中的操作名和操作函数
            for name, op in ops_under_test.items():
                # 使用 op 对 qX 进行 hardtanh 操作，得到 qY_hat
                qY_hat = op(qX, min_val, max_val)
                # 断言 qY 与 qY_hat 相等，若不相等则输出错误信息
                self.assertEqual(qY, qY_hat, msg=f"{name} hardtanh failed")

            # 待测试的操作字典，包含原地操作的 hardtanh
            ops_under_test_inplace = {
                'inplace ao.nn.quantized.functional.hardtanh':
                    torch.ao.nn.quantized.functional.hardtanh,
            }

            # 遍历 ops_under_test_inplace 字典中的操作名和操作函数
            for name, op_ in ops_under_test_inplace.items():
                # 克隆 qX 到 qY_hat
                qY_hat = qX.clone()
                # 使用 op_ 对 qY_hat 进行原地 hardtanh 操作
                op_(qY_hat, min_val, max_val, inplace=True)
                # 断言 qY 与 qY_hat 相等，若不相等则输出错误信息
                self.assertEqual(qY, qY_hat, msg=f"{name} hardtanh failed")

"""Tests the correctness of the quantized::hardswish op."""
@override_qengines
    def test_hardswish(self):
        # 定义最大边数和边长
        max_sides = (3, 4)
        # 定义边长列表
        side_lens = (1, 7)
        # 定义 torch 类型列表
        torch_types = (torch.quint8, torch.qint8)
        # 定义 Y 缩放因子
        y_scales = (0.1, )
        # 定义 Y 零点
        y_zero_points = (1,)
        # 将所有参数组合成一个列表
        combined = [max_sides, side_lens, torch_types, y_scales, y_zero_points]
        # 使用 itertools 生成所有可能的测试用例组合
        test_cases = itertools.product(*combined)
        # 遍历每个测试用例
        for test_case in test_cases:
            # 解包测试用例
            max_side, side_len, torch_type, Y_scale, Y_zero_point = test_case

            # 如果当前量化后端为 'qnnpack' 且 torch 类型不是 quint8，则跳过当前测试用例
            if torch.backends.quantized.engine == 'qnnpack' and torch_type != torch.quint8:
                continue

            # 根据最大边数和边长创建形状列表
            shapes = [side_len] * max_side
            # 获取随机张量 X 及其量化参数 X_scale 和 X_zero_point
            X, X_scale, X_zero_point = \
                _get_random_tensor_and_q_params(shapes, 2.0, torch_type)
            # 遍历两种内存格式：channels_last 和 contiguous_format
            for memory_format in torch.channels_last, torch.contiguous_format:
                # 如果内存格式为 channels_last 并且形状列表长度为 4，则将张量 X 转换为 channels_last 内存格式
                if memory_format == torch.channels_last and len(shapes) == 4:
                    X = X.to(memory_format=memory_format)
                # 对张量 X 进行量化
                qX = torch.quantize_per_tensor(X, scale=X_scale, zero_point=X_zero_point,
                                               dtype=torch_type)
                # 对量化的张量进行反量化
                dqX = qX.dequantize()

                # 对反量化后的张量应用 hardswish 激活函数
                dqY_hat = F.hardswish(dqX)
                # 将反量化后的张量再次量化为 qY_hat
                qY_hat = torch.quantize_per_tensor(dqY_hat, scale=Y_scale,
                                                   zero_point=Y_zero_point,
                                                   dtype=torch_type)

                # 使用量化版本的 hardswish 函数计算 qY
                qY = torch.ao.nn.quantized.functional.hardswish(
                    qX, scale=Y_scale, zero_point=Y_zero_point)
                # 断言量化后的结果 qY 与 qY_hat 相等
                self.assertEqual(
                    qY, qY_hat,
                    msg=f"Hardswish failed: {qY} vs {qY_hat}, {torch.backends.quantized.engine}")

    """
    Tests the correctness of the binary op + scalar.
    """
    # 定义测试函数，用于测试量化操作与标量加法/乘法的结合及ReLU函数的影响
    def _test_binary_op_scalar_relu(self, A, b, binary_op_name, binary_op, quantized_op, quantized_op_relu):
        import copy  # 导入copy模块，用于复制对象
        op_scalar = quantized_op  # 将量化后的标量操作赋值给op_scalar变量
        op_scalar_relu = quantized_op_relu  # 将带ReLU的量化标量操作赋值给op_scalar_relu变量

        A, (scale, zero_point, dtype) = A  # 解构A，获取数据和量化参数
        A = A.astype(np.float32)  # 将A转换为float32类型
        qA = torch.quantize_per_tensor(torch.from_numpy(A), scale, zero_point, dtype)  # 使用量化参数对A进行量化

        if binary_op_name == 'add':
            # 如果是加法操作，将b量化并加入到qA的反量化结果中
            C = binary_op(qA.dequantize(), round(b / scale) * scale)
        else:
            # 否则，直接使用b进行标量乘法操作
            C = binary_op(qA.dequantize(), b)

        C_relu = copy.deepcopy(C)  # 深复制C到C_relu
        C_relu[C_relu < 0] = 0  # 将C_relu中小于0的元素置为0（即应用ReLU函数）

        # 使用量化标量操作计算C_hat，并将结果量化为与C相同的格式
        C_hat = op_scalar(qA, b)
        C_ref = torch.quantize_per_tensor(C, C_hat.q_scale(), C_hat.q_zero_point(), dtype)  # 创建参考量化结果C_ref

        # 使用带ReLU的量化标量操作计算C_relu_hat，并将结果量化为与C_relu相同的格式
        C_relu_hat = op_scalar_relu(qA, b)
        C_relu_ref = torch.quantize_per_tensor(
            C_relu, C_relu_hat.q_scale(), C_relu_hat.q_zero_point(), dtype)  # 创建带ReLU的参考量化结果C_relu_ref

        # 断言语句，验证C_ref的反量化结果与C_hat的反量化结果是否一致
        self.assertEqual(C_ref.dequantize(), C_hat.dequantize(),
                         msg=f"{binary_op_name}_scalar results don't match: "
                         f"{C_ref.dequantize()} vs {C_hat.dequantize()}")

        # 断言语句，验证C_relu_ref的反量化结果与C_relu_hat的反量化结果是否一致
        self.assertEqual(C_relu_ref.dequantize(), C_relu_hat.dequantize(),
                         msg=f"{binary_op_name}_scalar_relu results don't match: "
                         f"{C_relu_ref.dequantize()} vs {C_relu_hat.dequantize()}")

    # 跳过 macOS 平台下的测试
    @unittest.skipIf(IS_MACOS, "skipping macos test")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    # 测试标量加法及ReLU操作的正确性
    def test_add_scalar_relu(self, A, b):
        self._test_binary_op_scalar_relu(A, b, "add", operator.add, torch.ops.quantized.add, torch.ops.quantized.add_relu)

    # 跳过 macOS 平台下的测试
    @unittest.skipIf(IS_MACOS, "skipping macos test")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    # 测试标量乘法及ReLU操作的正确性
    def test_mul_scalar_relu(self, A, b):
        self._test_binary_op_scalar_relu(A, b, "mul", operator.mul, torch.ops.quantized.mul, torch.ops.quantized.mul_relu)

    """Tests the correctness of the add and add_relu op."""
    # 定义一个测试函数，用于测试具有相同量化参数的量化加法和ReLU操作
    def test_qadd_relu_same_qparams(self):
        # 针对不同的数据类型进行迭代测试
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 获取 quantized.add_relu 操作的引用
            add_relu = torch.ops.quantized.add_relu
            # 获取 quantized.add 操作的引用
            add = torch.ops.quantized.add
            # 获取 quantized.add 操作的输出引用，可能是一个错误，应该是获取 add_out 操作的引用
            add_out = torch.ops.quantized.add
            # 获取 quantized.add_relu 操作的输出引用
            add_relu_out = torch.ops.quantized.add_relu

            # NB: 这个大小选择确保同时测试矢量化实现（每次处理64个元素块）和标量实现
            # 创建一个从 -128 到 129 的浮点数张量 A
            A = torch.arange(-128, 130, dtype=torch.float)
            # 创建一个从 -128 到 129 的浮点数张量 B
            B = torch.arange(-128, 130, dtype=torch.float)
            # 设置量化的缩放因子
            scale = 2.0
            # 设置量化的零点
            zero_point = 127
            # 对张量 A 进行量化
            qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=dtype)
            # 对张量 B 进行量化
            qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                           dtype=dtype)

            # 计算加法和ReLU的正确结果 C
            C = (qA.dequantize() + qB.dequantize()).numpy()
            # 将结果 C 量化为 qC
            qC = _quantize(C, scale, zero_point, dtype=np_dtype[dtype])
            # 使用 quantized.add 计算量化加法的结果 qC_hat
            qC_hat = add(qA, qB, scale=scale, zero_point=zero_point)
            # 断言量化加法的结果是否正确
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized addition failed.")
            # 创建一个空的量化张量 qC_out_hat 用于输出
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale,
                                                       zero_point=zero_point,
                                                       dtype=dtype)
            # 使用 quantized.add 将加法结果输出到 qC_out_hat
            add_out(qA, qB, out=qC_out_hat)
            # 断言输出的 qC_hat 和 qC_out_hat 是否相等
            self.assertEqual(qC_hat, qC_out_hat, msg="Add.out failed")

            # 计算加法后应用ReLU的正确结果 Crelu
            Crelu = C.copy()
            Crelu[C < 0] = 0
            # 将结果 Crelu 量化为 qCrelu
            qCrelu = _quantize(Crelu, scale, zero_point, dtype=np_dtype[dtype])
            # 使用 quantized.add_relu 计算量化加法后应用ReLU的结果 qCrelu_hat
            qCrelu_hat = add_relu(qA, qB, scale=scale, zero_point=zero_point)
            # 断言量化加法后应用ReLU的结果是否正确
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized addition with ReLU failed.")
            # 创建一个空的量化张量 qCrelu_out_hat 用于输出
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale,
                                                           zero_point=zero_point,
                                                           dtype=dtype)
            # 使用 quantized.add_relu 将加法后应用ReLU的结果输出到 qCrelu_out_hat
            add_relu_out(qA, qB, out=qCrelu_out_hat)
            # 断言输出的 qCrelu_hat 和 qCrelu_out_hat 是否相等
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="AddReLU.out failed")

    """Tests the correctness of the cudnn add and add_relu op
    (Similar to test_qadd_relu_different_qparams, will probably merge in the future)"""
    # 如果未启用 cudnn，则跳过此测试
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    # 如果 GPU 架构小于 SM80，则跳过此测试
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")
    # 如果在 ROCm 平台上运行，则跳过此测试
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    # 定义一个测试函数，用于测试 cudnn 下的量化加法和ReLU操作
    def test_qadd_relu_cudnn(self):
        # 指定数据类型为torch.qint8
        dtype = torch.qint8
        # 获取 quantized add_relu 和 add 的操作函数
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        # 在CUDA设备上创建张量A和B，其值从-128到129，数据类型为torch.float
        A = torch.arange(-128, 130, dtype=torch.float).to(torch.device("cuda"))
        B = torch.arange(-128, 130, dtype=torch.float).to(torch.device("cuda"))
        # 定义量化的比例尺度(scale)
        scale_A = 2.5
        scale_B = 6.3
        scale_C = 12.9
        zero_point = 0
        # 对张量A和B进行量化
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                       dtype=dtype)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                       dtype=dtype)

        # 计算未量化的张量A和B的和，然后移动到CPU并转换为NumPy数组
        C = (qA.dequantize() + qB.dequantize()).to(device="cpu").numpy()
        # 对计算得到的C进行量化
        qC = _quantize(C, scale_C, zero_point, dtype=np_dtype[dtype])
        # 使用torch.ops.quantized.add函数计算量化的qA和qB的和，并移动到CPU
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        # 检查计算结果是否与预期一致
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # 计算加上ReLU操作的结果的ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        # 对计算得到的Crelu进行量化
        qCrelu = _quantize(Crelu, scale_C, zero_point, dtype=np_dtype[dtype])
        # 使用torch.ops.quantized.add_relu函数计算加上ReLU操作的量化结果
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        # 检查计算结果是否与预期一致
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the cudnn add and add_relu op for nhwc format"""
    # 如果未启用cudnn，则跳过这个测试
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    # 如果GPU架构不是SM80或更高，则跳过这个测试
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")
    # 如果是在ROCm环境下，则不支持这个测试
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    # 定义测试函数，用于测试在使用 cuDNN 加速的情况下量化加法和加法+ReLU的正确性
    def test_qadd_relu_cudnn_nhwc(self):
        # 设置数据类型为 qint8
        dtype = torch.qint8
        # 载入 quantized.add_relu 和 quantized.add 运算
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        # 生成随机数据 A 和 B，并将其移到 CUDA 设备上
        A = torch.rand(16, 8, 4, 12).to(device="cuda")
        B = torch.rand(16, 8, 4, 12).to(device="cuda")
        scale_A = 2.5
        scale_B = 6.3
        scale_C = 12.9
        zero_point = 0
        # 对 A 和 B 进行量化
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                       dtype=dtype)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                       dtype=dtype)
        
        # 计算加法的标准结果 C，并将其转移到 CPU 并转换为 numpy 数组
        C = (qA.dequantize() + qB.dequantize()).to(device="cpu").numpy()
        # 对标准结果 C 进行量化
        qC = _quantize(C, scale_C, zero_point, dtype=np_dtype[dtype])
        # 使用 quantized.add 进行量化加法，并移回 CPU
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        # 使用 numpy.testing 进行相等断言，验证量化加法的正确性
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # 计算加法+ReLU的标准结果 Crelu，并将其转化为量化结果
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale_C, zero_point, dtype=np_dtype[dtype])
        # 使用 quantized.add_relu 进行量化加法+ReLU，并移回 CPU
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        # 使用 numpy.testing 进行相等断言，验证量化加法+ReLU的正确性
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the add and add_relu op."""


这段代码是一个用于测试在 CUDA 加速环境下的量化加法（`add`）和量化加法+ReLU（`add_relu`）操作的测试函数。
    def test_qadd_relu_different_qparams(self):
        # 遍历三种量化数据类型：torch.quint8, torch.qint8, torch.qint32
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 载入 quantized.add_relu 操作
            add_relu = torch.ops.quantized.add_relu
            # 载入 quantized.add 操作
            add = torch.ops.quantized.add
            # 载入 quantized.add 操作，用于输出
            add_out = torch.ops.quantized.add
            # 载入 quantized.add_relu 操作，用于输出
            add_relu_out = torch.ops.quantized.add_relu

            # NB: 这是一个特殊的大小，以便同时测试矢量化实现（每次64个元素）和标量实现
            A = torch.arange(-128, 130, dtype=torch.float)
            B = torch.arange(-128, 130, dtype=torch.float)
            scale_A = 3.0
            zero_point_A = 7
            scale_B = 5.0
            zero_point_B = 127

            scale_C = 0.5
            zero_point_C = 5

            # 对张量 A 和 B 进行量化
            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                           dtype=dtype)

            # 计算量化后的加法结果的准确值（ground truth）
            C = (qA.dequantize() + qB.dequantize()).numpy()
            qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype[dtype])
            # 使用 quantized.add 计算 qC_hat
            qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point_C)
            # 断言量化加法的准确性
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized addition failed.")

            # 创建一个与 qC 相同形状的空张量 qC_out_hat，使用 quantized.add 输出结果
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale_C,
                                                       zero_point=zero_point_C,
                                                       dtype=dtype)
            add_out(qA, qB, out=qC_out_hat)
            # 断言 add.out 的准确性
            self.assertEqual(qC_hat, qC_out_hat, msg="Add.out failed")

            # 计算加法后应用 ReLU 的准确值（ground truth）
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale_C, zero_point_C, dtype=np_dtype[dtype])
            # 使用 quantized.add_relu 计算 qCrelu_hat
            qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
            # 断言量化加法与 ReLU 的准确性
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized addition with ReLU failed.")

            # 创建一个与 qCrelu 相同形状的空张量 qCrelu_out_hat，使用 quantized.add_relu 输出结果
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale_C,
                                                           zero_point=zero_point_C,
                                                           dtype=dtype)
            add_relu_out(qA, qB, out=qCrelu_out_hat)
            # 断言 add_relu.out 的准确性
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="AddReLU.out failed")

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_relu_same_qparams(self):
        # 遍历数据类型列表，包括torch.quint8、torch.qint8、torch.qint32
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 载入量化操作函数
            mul_relu = torch.ops.quantized.mul_relu
            mul = torch.ops.quantized.mul
            mul_out = torch.ops.quantized.mul  # 这行代码似乎有误，应该是 mul_out = torch.ops.quantized.mul_out
            mul_relu_out = torch.ops.quantized.mul_relu  # 载入量化乘以 ReLU 操作函数

            # 创建两个相同范围的浮点数张量 A 和 B
            A = torch.arange(-100, 100, dtype=torch.float)
            B = torch.arange(-100, 100, dtype=torch.float)
            scale = 2
            zero_point = 127
            # 对张量 A 和 B 进行量化
            qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                           dtype=dtype)

            # 计算乘法和ReLU的基准值
            C = (qA.dequantize() * qB.dequantize()).numpy()
            qC = _quantize(C, scale, zero_point, dtype=np_dtype[dtype])
            # 使用量化乘法函数计算 qC_hat
            qC_hat = mul(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized multiplication failed.")
            # 创建一个空的量化张量 qC_out_hat，并使用 mul_out 函数填充它
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale,
                                                       zero_point=zero_point,
                                                       dtype=dtype)
            mul_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="mul.out failed")

            # 计算乘法加上ReLU的基准值
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale, zero_point, dtype=np_dtype[dtype])
            # 使用量化乘法加ReLU函数计算 qCrelu_hat
            qCrelu_hat = mul_relu(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized multiplication with ReLU failed.")
            # 创建一个空的量化张量 qCrelu_out_hat，并使用 mul_relu_out 函数填充它
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale,
                                                           zero_point=zero_point,
                                                           dtype=dtype)
            mul_relu_out(qA, qB, out=qCrelu_out_hat)
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="mulReLU.out failed")

            # 标量乘法测试
            for b in B:
                C_ref = qA.dequantize().numpy() * b.item()
                qC_hat = torch.ops.quantized.mul(qA, b.item())
                self.assertEqual(C_ref, qC_hat.dequantize())

            # 标量乘法加ReLU测试
            for b in B:
                C_ref = qA.dequantize().numpy() * b.item()
                C_ref[C_ref < 0] = 0
                qC_hat = torch.ops.quantized.mul_relu(qA, b.item())
                self.assertEqual(C_ref, qC_hat.dequantize())

    """Tests the correctness of the mul and mul_relu op."""
    """Tests the correctness of quantized mul and mul_relu operations with different quantization parameters."""

    # 定义一个测试函数，用于测试不同量化参数下的 quantized mul 和 mul_relu 操作
    def test_qmul_relu_different_qparams(self):
        # 对于每种数据类型，依次执行以下测试
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            # 获取 quantized mul_relu 和 mul 的操作函数
            mul_relu = torch.ops.quantized.mul_relu
            mul = torch.ops.quantized.mul
            mul_out = torch.ops.quantized.mul  # 这里应该是 mul_out = torch.ops.quantized.mul_out
            mul_relu_out = torch.ops.quantized.mul_relu

            # 创建输入张量 A 和 B，数据类型为 torch.float
            A = torch.arange(-100, 100, dtype=torch.float)
            B = torch.arange(-100, 100, dtype=torch.float)

            # 设置张量 A 的量化参数
            scale_A = 3.0
            zero_point_A = 7

            # 设置张量 B 的量化参数
            scale_B = 5.0
            zero_point_B = 127

            # 设置输出张量 C 的量化参数
            scale_C = 0.5
            zero_point_C = 5

            # 对张量 A 和 B 进行量化
            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                           dtype=dtype)

            # 计算 mul 的预期结果 C
            C = (qA.dequantize() * qB.dequantize()).numpy()

            # 对预期结果 C 进行量化
            qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype[dtype])

            # 使用 quantized mul 计算 qC_hat
            qC_hat = mul(qA, qB, scale=scale_C, zero_point=zero_point_C)

            # 验证 quantized mul 的输出是否与预期一致
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized multiplication failed.")

            # 创建一个空的量化张量 qC_out_hat，用于接收 mul 的输出
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale_C,
                                                       zero_point=zero_point_C,
                                                       dtype=dtype)

            # 使用 mul_out 将 mul 的结果写入 qC_out_hat
            mul_out(qA, qB, out=qC_out_hat)

            # 验证 mul_out 的输出是否与预期一致
            self.assertEqual(qC_hat, qC_out_hat, msg="mul.out failed")

            # 计算经过 ReLU 处理后的预期结果 Crelu
            Crelu = C.copy()
            Crelu[C < 0] = 0

            # 对预期结果 Crelu 进行量化
            qCrelu = _quantize(Crelu, scale_C, zero_point_C, dtype=np_dtype[dtype])

            # 使用 quantized mul_relu 计算 qCrelu_hat
            qCrelu_hat = mul_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)

            # 验证 quantized mul_relu 的输出是否与预期一致
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized multiplication with ReLU failed.")

            # 创建一个空的量化张量 qCrelu_out_hat，用于接收 mul_relu 的输出
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale_C,
                                                           zero_point=zero_point_C,
                                                           dtype=dtype)

            # 使用 mul_relu_out 将 mul_relu 的结果写入 qCrelu_out_hat
            mul_relu_out(qA, qB, out=qCrelu_out_hat)

            # 验证 mul_relu_out 的输出是否与预期一致
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="mulReLU.out failed")

    """Tests the correctness of the matmul op."""
    # 测试 matmul 操作的正确性
    @given(num_dims=st.integers(2, 5),
           outer_dims=st.lists(st.integers(2, 6), min_size=3, max_size=3),
           m=st.integers(2, 6),
           k=st.integers(2, 6),
           n=st.integers(2, 6),
           dtypes=st.sampled_from(((torch.qint8, np.int8),
                                   (torch.quint8, np.uint8))))
    # 定义一个测试函数，用于测试量化矩阵乘法的正确性
    def test_qmatmul(self, num_dims, outer_dims, m, k, n, dtypes):
        # 解包数据类型元组
        (torch_dtype, np_dtype) = dtypes

        # 根据给定的维度信息构造矩阵 A 和 B
        size_a = outer_dims[:num_dims - 2] + [m, k]
        size_b = outer_dims[:num_dims - 2] + [k, n]
        A = torch.randn(size=size_a, dtype=torch.float32) * 3
        B = torch.randn(size=size_b, dtype=torch.float32) * 3

        # 设置量化参数和零点偏移值
        scale_A = 3.1
        zero_point_A = 7
        scale_B = 5.3
        zero_point_B = 127

        # 设置输出量化的参数和零点偏移值
        scale_C = 1.3
        zero_point_C = 5

        # 使用给定的参数对矩阵 A 和 B 进行量化
        qA = torch.quantize_per_tensor(A,
                                       scale=scale_A,
                                       zero_point=zero_point_A,
                                       dtype=torch_dtype)
        qB = torch.quantize_per_tensor(B,
                                       scale=scale_B,
                                       zero_point=zero_point_B,
                                       dtype=torch_dtype)

        # 计算矩阵乘法的准确结果
        C = torch.matmul(qA.dequantize(), qB.dequantize()).numpy()

        # 将准确结果量化为 qC
        qC = _quantize(C, scale_C, zero_point_C, dtype=(np_dtype))

        # 调用 PyTorch 提供的量化矩阵乘法操作
        qC_hat = torch.ops.quantized.matmul(qA,
                                            qB,
                                            scale=scale_C,
                                            zero_point=zero_point_C)

        # 使用 NumPy 测试库断言量化矩阵乘法的结果是否正确
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized multiplication failed.")

        # 使用通道级量化时，期望失败并抛出 RuntimeError
        axis = 0
        scales_A = torch.rand(size=(A.shape[axis],))
        zero_points_A = torch.randint(low=0, high=5, size=(A.shape[axis],))
        scales_B = torch.rand(size=(B.shape[axis],))
        zero_points_B = torch.randint(low=0, high=5, size=(B.shape[axis],))

        qA = torch.quantize_per_channel(A,
                                        scales=scales_A,
                                        zero_points=zero_points_A,
                                        axis=axis,
                                        dtype=torch.qint8)
        qB = torch.quantize_per_channel(B,
                                        scales=scales_B,
                                        zero_points=zero_points_B,
                                        axis=axis,
                                        dtype=torch.qint8)

        # 使用 per-tensor 的量化矩阵乘法操作时，期望失败并抛出 RuntimeError
        np.testing.assert_raises_regex(RuntimeError,
                                       ".*per-tensor.*",
                                       torch.ops.quantized.matmul,
                                       qA,
                                       qB,
                                       scale_C,
                                       zero_point_C)
    # 定义一个测试方法，用于测试量化 softmax 操作的正确性
    def test_qsoftmax(self, dims):
        # 遍历不同的维度组合和内存格式
        for (num_dims, dim, memory_format) in [
            (2, 1, torch.contiguous_format),  # 2维 softmax，沿着最后一个维度
            (4, 3, torch.contiguous_format),  # 大于2维，沿着最后一个维度的 softmax
            (5, 2, torch.contiguous_format),  # 大于2维，非最后一个维度的 softmax（需要 permute）
            (4, 3, torch.channels_last),      # 大于2维，沿着最后一个维度的 softmax，但是不连续
            (4, 1, torch.channels_last),      # Channels Last，不需要 permute
            (5, 1, torch.channels_last_3d),   # Channels Last 3D，不需要 permute
        ]:
            # 根据给定的维度数量选择大小
            size = dims[:num_dims]
            # 设置 torch 的数据类型为 quint8
            torch_dtype = torch.quint8
            # 设置 numpy 的数据类型为 uint8
            np_dtype = np.uint8

            # 设置量化参数
            scale_X = 1.3
            zero_point_X = 5
            # 创建随机张量 X，按照指定的范围和偏移量
            X = torch.rand(size=size, dtype=torch.float32) * 8 + zero_point_X
            # 根据 memory_format 转换张量 X
            X = X.to(memory_format=memory_format)

            # 设置输出量化参数
            scale_Y = 1 / 256
            zero_point_Y = 0

            # 对张量 X 进行量化
            qX = torch.quantize_per_tensor(X,
                                           scale=scale_X,
                                           zero_point=zero_point_X,
                                           dtype=torch_dtype)

            # 计算未量化的张量 X 的 softmax 作为 ground truth，并转换为 numpy 数组
            Y = torch.softmax(qX.dequantize(), dim=dim).numpy()
            # 对 ground truth 进行量化
            qY = _quantize(Y, scale_Y, zero_point_Y, dtype=np_dtype)
            # 使用 quantized softmax 操作对 qX 进行处理
            qY_hat = torch.ops.quantized.softmax(qX,
                                                 dim=dim,
                                                 output_scale=scale_Y,
                                                 output_zero_point=zero_point_Y)

            # 断言量化 softmax 的输出与预期的量化值相等
            np.testing.assert_equal(qY, qY_hat.int_repr(),
                                    "Quantized softmax failed.")

    """Tests the correctness of the quantized softmax op using qnnpack."""
    # 如果没有 QNNPACK，跳过这个测试
    @skipIfNoQNNPACK
    def test_qsoftmax_qnnpack(self):
        # 使用 qnnpack 引擎覆盖量化引擎，然后运行 test_qsoftmax 测试方法
        with override_quantized_engine('qnnpack'):
            self.test_qsoftmax()

    """Tests the correctness of the mul and mul_relu op."""
    """Tests that quantized multiply and relu work with broadcasting"""
    # 定义函数内的测试函数，用于测试量化乘法和ReLU操作的广播功能
    def test_qmul_broadcast(self):
        # 获取 torch.ops.quantized.mul_relu 的函数引用
        mul_relu = torch.ops.quantized.mul_relu
        # 获取 torch.ops.quantized.mul 的函数引用
        mul = torch.ops.quantized.mul
        # 获取 torch.ops.quantized.mul 的函数引用，命名错误应该为 quantized.mul_out
        mul_out = torch.ops.quantized.mul
        # 获取 torch.ops.quantized.mul_relu 的函数引用
        mul_relu_out = torch.ops.quantized.mul_relu

        # 定义输入张量 A 和 B，使用随机数填充
        A = torch.randn(8, 1, 6, 1)
        B = torch.randn(7, 1, 5)
        # 定义量化参数：张量 A 的缩放因子和零点
        scale_A = 3.0
        zero_point_A = 7
        # 定义量化参数：张量 B 的缩放因子和零点
        scale_B = 5.0
        zero_point_B = 127

        # 定义输出张量的缩放因子和零点
        scale_C = 0.5
        zero_point_C = 5

        # 对张量 A 和 B 进行量化
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                       dtype=torch.quint8)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                       dtype=torch.quint8)

        # 计算乘法的正确结果（ground truth）
        C = (qA.dequantize() * qB.dequantize()).numpy()
        # 对结果 C 进行量化
        qC = _quantize(C, scale_C, zero_point_C)
        # 使用 torch.ops.quantized.mul 计算乘法并量化
        qC_hat = mul(qA, qB, scale=scale_C, zero_point=zero_point_C)
        # 检查是否相等，若不相等则抛出异常
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized multiplication failed.")

    """Tests that quantized add works with broadcasting"""
    # 测试量化加法在广播情况下的工作性能
    def test_qadd_broadcast(self):
        # 定义输入张量 A 和 B，使用随机数填充
        A = torch.randn(1, 1, 4, 4)
        B = torch.randn(2, 1, 4, 4)
        # 对张量 A 和 B 进行量化，设置缩放因子和零点
        qA = torch.quantize_per_tensor(A, 0.02, 0, torch.quint8)
        qB = torch.quantize_per_tensor(B, 0.04, 2, torch.quint8)

        # 定义输出张量的缩放因子和零点
        output_scale = 0.01
        output_zp = 1

        # 计算加法的正确结果（ground truth）
        C = qA.dequantize() + qB.dequantize()
        # 对结果 C 进行量化
        qC = torch.quantize_per_tensor(C, output_scale, output_zp, torch.quint8)

        # 使用 torch.ops.quantized.add 计算加法并量化
        qC_hat_1 = torch.ops.quantized.add(qA, qB, output_scale, output_zp)
        qC_hat_2 = torch.ops.quantized.add(qB, qA, output_scale, output_zp)

        # 断言两个量化结果是否接近，若不接近则抛出异常
        self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_1.dequantize()))
        self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_2.dequantize()))

    """Tests channel shuffle operation on quantized tensors."""
    # 测试量化张量上的通道混洗操作
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=2, max_side=32, max_numel=10**5),
                       qparams=hu.qparams(dtypes=[torch.quint8])),
           groups=st.integers(2, 6))
    # 定义一个测试方法，用于测试通道混洗操作
    def test_channel_shuffle(self, X, groups):
        # 解包 X，获取其内容和元数据 (scale, zero_point, torch_type)
        X, (scale, zero_point, torch_type) = X
        # 获取输入张量 X 的通道数
        channels = X.shape[-3]
        # 获取输入张量 X 的高度和宽度
        iH, iW = X.shape[-2:]
        # 假设通道数能够整除分组数
        assume(channels % groups == 0)

        # 将 X 转换为 Torch 张量
        a = torch.from_numpy(X)
        # 创建一个与 a 相同形状的随机张量 a，并进行通道混洗操作
        a = torch.rand(a.shape)
        a_out = torch.nn.functional.channel_shuffle(a, groups)

        # 对 a_out 进行量化操作，使用给定的量化参数 (scale, zero_point, torch_type)
        a_ref = torch.quantize_per_tensor(a_out, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        # 对量化后的张量 a_ref 进行反量化操作
        a_ref = a_ref.dequantize()
        # 对输入张量 a 进行量化操作，使用给定的量化参数 (scale, zero_point, torch_type)
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 对量化后的张量 qa 进行通道混洗操作
        a_hat = torch.nn.functional.channel_shuffle(qa, groups)
        # 断言 a_ref 和 a_hat 的反量化结果相等，用于检查通道混洗操作是否正确
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="torch.nn.functional.channel_shuffle results are off")

    """Tests 1D max pool operation on quantized tensors."""
    # 使用 hypothesis 框架定义一个测试，用于测试在量化张量上的 1D 最大池化操作
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=2, max_dims=3,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           dilation=st.integers(1, 2),
           padding=st.integers(0, 2),
           ceil_mode=st.booleans())
    # 定义测试函数，用于测试 max_pool1d 操作的各种参数组合
    def test_max_pool1d(self, X, kernel, stride, dilation, padding, ceil_mode):
        # 解包输入的 X，包括数据和量化参数（缩放因子、零点和数据类型）
        X, (scale, zero_point, torch_type) = X
        # 检查约束条件，确保 kernel 不会超出 padding 的边界
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        # 获取输入张量 X 的最后一个维度的大小
        iW = X.shape[-1]
        # 计算池化操作后的输出形状
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        # 假设输出形状 oW 大于 0
        assume(oW > 0)

        # 将 numpy 数组 X 转换为 PyTorch 张量 a
        a = torch.from_numpy(X)
        # 对张量 a 进行 max_pool1d 操作，指定池化核大小、步幅、填充、膨胀、上取整模式
        a_pool = torch.nn.functional.max_pool1d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                ceil_mode=ceil_mode)
        # 对池化后的张量 a_pool 进行量化，使用给定的缩放因子、零点和数据类型
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        # 反量化量化后的张量 a_ref
        a_ref = a_ref.dequantize()
        # 对张量 a 进行量化，使用给定的缩放因子、零点和数据类型
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 定义待测试的操作集合
        ops_under_test = {
            "torch": torch.max_pool1d,
            "nn.functional": torch.nn.functional.max_pool1d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.max_pool1d,
        }

        # 遍历操作集合，逐个进行测试
        for name, op in ops_under_test.items():
            # 使用当前操作 op 对量化后的张量 qa 进行 max_pool1d 操作
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            # 断言操作结果与参考值 a_ref 反量化后的结果相等，否则输出错误消息
            self.assertEqual(a_ref, a_hat.dequantize(),
                             msg=f"{name} results are off")

        # 单独测试 ops.quantized.max_pool1d，因为 None 未被处理
        a_hat = torch.ops.quantized.max_pool1d(
            qa, kernel_size=_single(kernel),
            stride=_single(kernel if stride is None else stride),
            padding=_single(padding), dilation=_single(dilation),
            ceil_mode=ceil_mode)
        # 断言操作结果与参考值 a_ref 反量化后的结果相等，否则输出错误消息
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool1d results are off")

    # TODO: 将此测试与 test_max_pool2d 合并
    """Tests 2D cudnn max pool operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       # cudnn's support for quantized pooling is limited to
                       # int8 currently
                       qparams=hu.qparams(dtypes=[torch.qint8])),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           # currently there is no support for dilation for cudnn
           # pooling
           dilation=st.integers(1, 1),
           padding=st.integers(0, 2),
           ceil_mode=st.booleans())
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skipIf(TEST_CUDNN_VERSION <= 90100, "cuDNN maxpool2d mishandles -128 before v90100")
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    # 定义一个测试函数，用于测试使用 cuDNN 的二维最大池化操作
    def test_max_pool2d_cudnn(self, X, kernel, stride, dilation, padding, ceil_mode):
        # 解包输入 X，包括数据和量化参数
        X, (scale, zero_point, torch_type) = X
        # 假设 kernel // 2 大于等于 padding，确保内核没有悬挂在边界外
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        # 获取输入张量 X 的高度和宽度
        iH, iW = X.shape[-2:]
        # 计算池化操作后输出的高度 oH
        oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
        assume(oH > 0)  # 假设输出高度 oH 大于 0
        # 计算池化操作后输出的宽度 oW
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)  # 假设输出宽度 oW 大于 0

        # 将 numpy 数组 X 转换为 PyTorch 张量，并移动到 CUDA 设备上
        a = torch.from_numpy(X).to(device="cuda")
        # 执行 PyTorch 的二维最大池化操作
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode)
        # 对池化结果进行量化处理，使用之前的量化参数
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        # 反量化量化后的结果，得到参考的浮点数结果
        a_ref = a_ref.dequantize()
        # 再次对输入张量 a 进行量化处理，使用相同的量化参数
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 分别测试 ops.quantized.max_pool2d，因为 None 不会被特别处理
        # 调用 ops.quantized.max_pool2d 执行量化的二维最大池化操作
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        # 使用断言验证 ops.quantized.max_pool2d 的结果是否正确
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool2d results are off")

"""Tests 2D max pool operation on quantized tensors."""
# 使用 hypothesis 提供的装饰器定义一个测试函数，测试量化张量的二维最大池化操作
@given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                          min_side=1, max_side=10),
                   qparams=hu.qparams()),
       kernel=st.sampled_from((3, 5, 7)),
       stride=st.sampled_from((None, 1, 2)),
       dilation=st.integers(1, 2),
       padding=st.integers(0, 2),
       ceil_mode=st.booleans())
    # 定义一个测试函数，用于测试 max_pool2d 函数的不同输入情况
    def test_max_pool2d(self, X, kernel, stride, dilation, padding, ceil_mode):
        # 解包输入的 X 变量，获取数据和量化相关参数
        X, (scale, zero_point, torch_type) = X

        # 检查约束条件，确保 kernel 不会超出 padding 的边界
        assume(kernel // 2 >= padding)

        # 获取输入张量 X 的高度和宽度
        iH, iW = X.shape[-2:]

        # 计算池化操作后的输出高度和宽度
        oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)

        # 将 numpy 数组 X 转换为 PyTorch 张量 a
        a = torch.from_numpy(X)

        # 使用 torch.nn.functional.max_pool2d 进行最大池化操作，得到池化后的张量 a_pool
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode)

        # 对池化后的张量进行量化操作，得到量化后的参考张量 a_ref
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)

        # 取消量化操作，得到非量化的张量 a_ref
        a_ref = a_ref.dequantize()

        # 对输入张量 X 进行量化操作，得到量化后的张量 qa
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # 准备待测试的操作列表
        ops_under_test = {
            "torch": torch.max_pool2d,
            "nn.functional": torch.nn.functional.max_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.max_pool2d,
        }

        # 遍历操作列表，分别对 qa 进行不同操作，并验证结果
        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            self.assertEqual(a_ref, a_hat.dequantize(),
                             msg=f"{name} results are off")

        # 单独测试 ops.quantized.max_pool2d，因为它处理 None 的方式不同
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool2d results are off")
    def test_max_pool2d_pt2e(self):
        # 定义各种参数列表，用于测试不同的池化配置
        kernel_list = [2, 3]
        stride_list = [1, 2]
        padding_list = [0, 2]
        dilation_list = [1, 2]
        ceil_mode_list = [False, True]
        channels_last_input = [False, True]
        
        # 生成所有可能的参数组合
        options = itertools.product(kernel_list, stride_list, padding_list, dilation_list, ceil_mode_list, channels_last_input)
        
        # 遍历所有参数组合进行测试
        for kernel, stride, padding, dilation, ceil_mode, channels_last in options:
            # 如果 padding 大于等于 kernel 的一半，则跳过当前参数组合的测试
            if padding >= (kernel // 2):
                # 继续处理无效输入
                continue
            
            # 创建随机整数张量作为输入
            input = torch.randint(0, 8, (1, 3, 8, 8), dtype=torch.uint8)
            
            # 如果 channels_last 为 True，则需要使用 channels_last 格式
            if channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            # 使用 torch.nn.functional 进行标准的池化操作，并转换输出为 uint8 类型
            a_pool = torch.nn.functional.max_pool2d(input.to(torch.float32), kernel_size=kernel,
                                                    stride=stride, padding=padding, dilation=dilation,
                                                    ceil_mode=ceil_mode).to(torch.uint8)
            
            # 使用 torch.ops.quantized 进行量化池化操作
            a_hat = torch.ops.quantized.max_pool2d(input, kernel_size=_pair(kernel),
                                                   stride=_pair(stride), padding=_pair(padding),
                                                   dilation=_pair(dilation), ceil_mode=ceil_mode)
            
            # 断言输入张量的内存格式与输出张量的内存格式是否一致
            self.assertEqual(input.is_contiguous(), a_hat.is_contiguous(),
                             msg="ops.quantized.max_pool2d input output diff memory format")
            
            # 断言标准池化和量化池化的结果是否一致
            self.assertEqual(a_pool, a_hat,
                             msg="ops.quantized.max_pool2d results are off")
    """Tests max pool operation on NHWC quantized tensors."""
    # 定义一个测试方法，用于测试对 NHWC 格式的量化张量进行最大池化操作

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           # 使用 hypothesis 的给定数据生成器，生成输入张量 X 和量化参数 qparams
           kernel=st.sampled_from((3, 5, 7)),
           # 从指定的核大小中随机选择一个作为 kernel 大小
           stride=st.sampled_from((None, 1, 2)),
           # 从指定的步幅中随机选择一个作为 stride
           dilation=st.integers(1, 2),
           # dilation 在 1 和 2 之间取整数值
           padding=st.integers(0, 2),
           # padding 在 0 和 2 之间取整数值
           ceil_mode=st.booleans())
           # ceil_mode 随机选择 True 或 False
    # 定义一个测试函数，用于测试 max_pool2d 操作，使用 NHWC 格式的输入数据
    def test_max_pool2d_nhwc(self, X, kernel, stride, dilation, padding, ceil_mode):
        # 从输入元组 X 中解包得到 NHWC 格式的数据 X，以及 scale, zero_point, torch_type
        X, (scale, zero_point, torch_type) = X
        
        # 确保进入向量化路径
        # 176 = 128 + 32 + 16
        # 128 使用交织路径（interleaved path）
        # 32 使用非交织路径（non-interleaved path）
        # 16 使用标量路径（scalar path）
        if X.shape[1] < 176:
            # 如果输入数据 X 的通道数小于 176，则通过复制使其达到 176 个通道
            X = np.repeat(X, 176 / X.shape[1], 1)
        
        # 检查约束条件
        assume(kernel // 2 >= padding)  # Kernel 不能超出边界！
        
        # 获取输入数据 X 的高度和宽度
        iH, iW = X.shape[-2:]
        
        # 计算池化层的输出高度和宽度
        oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)

        # 将输入数据 X 转换为 NCHW 格式的数据
        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))
        a = torch.from_numpy(X_nchw).permute([0, 3, 1, 2])
        
        # 对转换后的数据 a 进行最大池化操作
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode)
        
        # 对池化后的结果进行量化
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        
        # 取消量化，得到浮点数结果
        a_ref = a_ref.dequantize()
        
        # 对原始 NHWC 格式的输入数据 X 进行量化，并转换为 NCHW 格式
        qa = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale, zero_point=zero_point,
                                       dtype=torch_type).permute([0, 3, 1, 2])
        
        # 断言量化后的数据的步幅是否有序
        self.assertTrue(qa.stride() != sorted(qa.stride()))

        # 待测试的操作和函数
        ops_under_test = {
            "torch": torch.max_pool2d,
            "nn.functional": torch.nn.functional.max_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.max_pool2d,
        }

        # 遍历每个操作和函数进行测试
        for name, op in ops_under_test.items():
            # 调用每个操作进行 max_pool2d 操作
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            # 断言量化后的结果是否与参考结果 a_ref 一致
            self.assertTrue(a_hat.stride() != sorted(a_hat.stride()))
            self.assertEqual(a_ref, a_hat.dequantize(),
                             msg=f"{name} 的结果不正确")

        # 分开测试 ops.quantized.max_pool2d，因为 None 值不被处理
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        
        # 断言量化后的结果是否与参考结果 a_ref 一致
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool2d 的结果不正确")
    # 定义一个测试函数，用于测试 max_pool3d_nhwc 方法
    def test_max_pool3d_nhwc(self):
        # 定义支持的 Torch 数据类型，包括量化整数类型和无符号量化整数类型
        torch_types = [torch.qint8, torch.quint8]
        # 定义不同的卷积核尺寸
        kernels = [1, 3]
        # 定义不同的步幅
        strides = [1, 3]
        # 定义不同的扩张率
        dilations = [1, 3]
        # 定义不同的填充大小
        paddings = [1, 3]
        # 定义是否使用 ceil 模式的选项
        ceil_modes = [True, False]
        # 生成所有可能的参数组合
        options = itertools.product(torch_types, kernels, strides, dilations, paddings, ceil_modes)
        
        # 遍历所有参数组合
        for torch_type, kernel, stride, dilation, padding, ceil_mode in options:
            # 创建一个随机整数张量 X，形状为 (2, 67, 16, 10, 10)，并转换为浮点型
            X = torch.randint(20, 40, (2, 67, 16, 10, 10)).to(torch.float)
            # 创建 X 的深拷贝 X_copy
            X_copy = copy.deepcopy(X)
            # 将 X 转换为内存格式为通道最后的 3D 张量
            X = X.contiguous(memory_format=torch.channels_last_3d)
            # 设置量化参数的缩放因子和零点
            scale = 15
            zero_point = 20
            
            # 检查无效输入的约束条件
            if not (kernel // 2 >= padding):
                continue
            
            # 获取输入张量的时间维度、高度维度和宽度维度
            iT, iH, iW = X.shape[-3:]
            
            # 计算池化操作后的输出形状的时间维度
            oT = pool_output_shape(iT, kernel, padding, stride, dilation, ceil_mode)
            if not (oT > 0):
                continue
            
            # 计算池化操作后的输出形状的高度维度
            oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
            if not (oH > 0):
                continue
            
            # 计算池化操作后的输出形状的宽度维度
            oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
            if not (oW > 0):
                continue
            
            # 使用 torch.nn.functional.max_pool3d 执行最大池化操作
            a_pool = torch.nn.functional.max_pool3d(X, kernel_size=kernel,
                                                    stride=stride,
                                                    padding=padding, dilation=dilation,
                                                    ceil_mode=ceil_mode)
            
            # 对池化结果进行量化处理，使用指定的缩放因子和零点，以及数据类型 torch_type
            a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                              zero_point=zero_point, dtype=torch_type)
            # 反量化量化后的结果
            a_ref = a_ref.dequantize()
            
            # 对原始数据 X_copy 进行量化处理，使用相同的参数和内存格式为通道最后的 3D 张量
            qa = torch.quantize_per_tensor(X_copy, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            qa = qa.contiguous(memory_format=torch.channels_last_3d)
            
            # 定义测试操作的字典，包括 torch.max_pool3d 和 torch.nn.functional.max_pool3d
            ops_under_test = {
                "torch": torch.max_pool3d,
                "nn.functional": torch.nn.functional.max_pool3d,
            }
            
            # 遍历测试操作字典，执行池化操作并断言结果
            for name, op in ops_under_test.items():
                a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                           dilation=dilation, ceil_mode=ceil_mode)
                self.assertEqual(a_ref, a_hat.dequantize(),
                                 msg=f"{name} results are off")
    def test_avg_pool2d(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        """
        X, (scale, zero_point, torch_type) = X

        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]  # 获取输入张量 X 的高度和宽度
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)  # 计算池化操作后的输出高度
        assume(oH > 0)  # 假设输出高度大于0，即有效的池化操作
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)  # 计算池化操作后的输出宽度
        assume(oW > 0)  # 假设输出宽度大于0，即有效的池化操作

        X = torch.from_numpy(X)  # 将输入张量 X 转换为 PyTorch 的张量
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)  # 对输入张量进行量化

        X = qX.dequantize()  # 将量化后的张量再反量化为浮点数张量，用于参考

        # 运行浮点数张量的参考操作，并将结果量化用于比较
        X_ref = torch.nn.functional.avg_pool2d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        # 待测试的操作集合，包括 PyTorch 的标准和量化的池化操作
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool2d,
        }

        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"

        # 遍历待测试的操作集合，进行测试
        for name, op in ops_under_test.items():
            qX_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, divisor_override=divisor_override)

            # 将参考结果量化，并与待测试的量化结果进行比较
            qX_ref = torch.quantize_per_tensor(X_ref, scale=qX_hat.q_scale(), zero_point=qX_hat.q_zero_point(),
                                               dtype=torch_type)

            # 使用断言比较量化后的整数表示结果，确保结果在允许的误差范围内
            self.assertEqual(qX_ref.int_repr().to(torch.double), qX_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), qX_hat.int_repr()))

            # 比较量化参数的 scale 值
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))

            # 比较量化参数的 zero_point 值
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))
    def test_avg_pool2d_nhwc(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: 1) we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        2) we cannot test the qint32, since the float point precision is much lower than int32 for big number,
        which will make the test be very flaky.
        """
        # 解构输入 X，获取其中的数据和量化参数
        X, (scale, zero_point, torch_type) = X
        # 获取输入数据的高度和宽度
        H, W = X.shape[-2:]

        # 如果输入数据通道数小于 176，进行通道数的扩展
        if X.shape[1] < 176:
            X = np.repeat(X, 176 / X.shape[1], 1)

        # 假设：核大小除以2大于等于填充大小，确保核没有超出边界
        assume(kernel // 2 >= padding)
        # 计算输入数据的高度和宽度
        iH, iW = X.shape[-2:]
        # 计算平均池化后的输出高度和宽度
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)  # 假设输出高度大于0
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)  # 假设输出宽度大于0

        # 将输入数据转换为 NCHW 格式的数组
        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))

        # 对输入数据进行量化
        qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale,
                                       zero_point=zero_point, dtype=torch_type).permute([0, 3, 1, 2])
        # 对量化后的数据进行反量化
        X = qX.dequantize()

        # 运行基准函数以获取参考结果，这里是使用了 PyTorch 的 avg_pool2d 函数
        X_ref = torch.nn.functional.avg_pool2d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        # 检查量化后的数据的步幅是否有序
        self.assertTrue(qX.stride() != sorted(qX.stride()))

        # 待测试的运算函数集合
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool2d,
        }

        # 错误信息模板
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"

        # 遍历每个待测试的函数
        for name, op in ops_under_test.items():
            # 使用当前函数计算结果
            X_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad, divisor_override=divisor_override)
            # 检查计算结果的步幅是否有序
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))

            # 将参考结果量化并转换为双精度浮点数，以便比较
            qX_ref = torch.quantize_per_tensor(X_ref, scale=X_hat.q_scale(), zero_point=X_hat.q_zero_point(),
                                               dtype=torch_type)

            # 检查量化后的整数表示是否与当前计算结果的整数表示一致，允许误差范围为1.0
            self.assertEqual(qX_ref.int_repr().to(torch.double), X_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), X_hat.int_repr()))

            # 检查量化的比例因子是否与期望一致
            self.assertEqual(scale, X_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, X_hat.q_scale()))

            # 检查量化的零点是否与期望一致
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale, X_hat.q_zero_point()))
    # 使用 `@given` 装饰器定义一个参数化测试的测试用例
    @given(
        # 定义参数 X，其形状是一个包含5个维度的数组，每个维度的大小在5到10之间
        X=hu.tensor(shapes=hu.array_shapes(min_dims=5, max_dims=5,
                                           min_side=5, max_side=10),
                    # 定义量化参数为 quint8（8位无符号整数）
                    qparams=hu.qparams(dtypes=torch.quint8)),
        # 定义卷积核的尺寸，可以是3或5
        kernel=st.sampled_from((3, 5)),
        # 定义步长，可以是None、1或2
        stride=st.sampled_from((None, 1, 2)),
        # 定义填充大小，范围在0到2之间的整数
        padding=st.integers(0, 2),
        # 定义是否使用 ceil 模式，可以是True或False
        ceil_mode=st.sampled_from((True, False)),
        # 定义是否包括填充在内，可以是True或False
        count_include_pad=st.sampled_from((True, False)),
        # 定义是否覆盖除法，这里定义为None
        divisor_override=st.sampled_from((None, None))
    )
    def test_avg_pool3d(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        """
        # 解包输入参数 X，包括数据和量化相关信息
        X, (scale, zero_point, torch_type) = X

        # 假设内核大小至少不小于填充大小
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        
        # 获取输入数据的空间维度大小
        iD, iH, iW = X.shape[-3:]
        
        # 计算池化操作后的输出维度
        oD = pool_output_shape(iD, kernel, padding, stride, dilation=1)
        assume(oD > 0)
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)

        # 将 numpy 数组转换为 PyTorch 张量
        X = torch.from_numpy(X)
        
        # 对输入张量进行量化
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        
        # 对量化后的张量进行反量化，以便后续比较
        X = qX.dequantize()
        
        # 在浮点数张量上运行参考的 avg_pool3d 操作，并量化结果以进行比较
        X_ref = torch.nn.functional.avg_pool3d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        # 待测试的操作集合，包括 nn.functional 和 ao.nn.quantized.functional
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool3d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool3d,
        }
        
        # 错误消息模板，用于显示测试结果不符的详细信息
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        
        # 遍历待测试的操作集合，执行量化池化操作并进行比较
        for name, op in ops_under_test.items():
            qX_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, divisor_override=divisor_override)
            
            # 将参考量化结果反量化，并与当前测试的量化结果进行比较
            qX_ref = torch.quantize_per_tensor(X_ref, scale=qX_hat.q_scale(), zero_point=qX_hat.q_zero_point(),
                                               dtype=torch_type)
            
            # 使用 assertEqual 检查整数表示的量化结果是否在容差范围内一致，同时检查量化参数的一致性
            self.assertEqual(qX_ref.int_repr().to(torch.double), qX_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), qX_hat.int_repr()))
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))
    # 使用 hypothesis 库的 @given 装饰器定义一个参数化的测试函数。
    # X 参数是一个张量，形状由 hu.array_shapes 控制，维度为 5，每个维度的大小在 5 到 10 之间，
    #     数据类型是 torch.qint8。
    # kernel 参数是一个从 (4, 5) 中随机选择的整数。
    # stride 参数是一个从 (None, 1, 2) 中随机选择的整数。
    # padding 参数是一个从 0 到 2 的随机整数。
    # ceil_mode 参数是一个布尔值，从 (True, False) 中随机选择。
    # count_include_pad 参数是一个布尔值，从 (True, False) 中随机选择。
    # divisor_override 参数是一个从 (None, None) 中随机选择的值。
    # 定义一个测试方法，用于测试 3D 平均池化操作，采用 NHWC 格式的输入张量
    def test_avg_pool3d_nhwc(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: 1) we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        2) we cannot test the qint32, since the float point precision is much lower than int32 for big number,
        which will make the test be very flaky.
        """
        # 解包输入 X，包括数据和元数据 (scale, zero_point, torch_type)
        X, (scale, zero_point, torch_type) = X
        # 获取输入张量的深度 D，高度 H，宽度 W
        D, H, W = X.shape[-3:]

        # 如果输入张量的通道数小于 176，则使用 np.repeat 进行扩展
        if X.shape[1] < 176:
            X = np.repeat(X, 176 / X.shape[1], 1)

        # 假设核大小的一半大于等于填充大小，确保核不会超出边界
        assume(kernel // 2 >= padding)
        # 获取输入张量的新深度、高度、宽度
        iD, iH, iW = X.shape[-3:]
        # 计算池化后的输出深度、高度、宽度
        oD = pool_output_shape(iD, kernel, padding, stride, dilation=1)
        assume(oD > 0)
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)

        # 将输入张量转换为 NCHW 格式的连续数组
        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 4, 1]))

        # 使用 torch.quantize_per_tensor 对输入张量进行量化，并转换为 NCHW 格式
        qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale,
                                       zero_point=zero_point, dtype=torch_type).permute([0, 4, 1, 2, 3])
        # 对量化后的张量进行反量化操作，得到浮点数张量 X
        X = qX.dequantize()

        # 在 int_repr 上运行参考操作，避免双重舍入误差
        X_ref = torch.nn.functional.avg_pool3d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        # 断言量化后的张量的步长不是按升序排列的
        self.assertTrue(qX.stride() != sorted(qX.stride()))
        
        # 定义待测试的操作及其名称
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool3d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool3d,
        }
        # 错误消息的模板
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"

        # 遍历每个操作及其名称
        for name, op in ops_under_test.items():
            # 对 qX 使用操作 op 进行池化操作，得到 X_hat
            X_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad, divisor_override=divisor_override)
            # 断言 X_hat 的步长不是按升序排列的
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
            
            # 将 X_ref 量化为与 X_hat 相同的量化参数，并转换为与 X_hat 相同的类型
            qX_ref = torch.quantize_per_tensor(X_ref, scale=X_hat.q_scale(), zero_point=X_hat.q_zero_point(),
                                               dtype=torch_type)

            # 断言 qX_ref 的 int_repr 与 X_hat 的 int_repr 相等，允许的绝对误差为 1.0，相对误差为 0
            self.assertEqual(qX_ref.int_repr().to(torch.double), X_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), X_hat.int_repr()))
            # 断言量化参数 scale 与 X_hat 的 q_scale 相等
            self.assertEqual(scale, X_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, X_hat.q_scale()))
            # 断言量化参数 zero_point 与 X_hat 的 q_zero_point 相等
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                             X_hat.q_zero_point()))
    """Tests adaptive average pool operation on NHWC quantized tensors."""
    """Tests adaptive average pool operation on NHWC quantized tensors."""
    # 定义一个测试方法，用于测试在NHWC格式量化张量上的自适应平均池化操作
    def test_qtopk(self):
        # 定义各种测试维度和参数的组合
        x_dims = [3, 4]  # 形状中的元素数量
        sides = [3, 5]  # 生成张量的边长
        dims = [0, 1, 2, 3]  # 执行topk操作的维度
        largest = [False, True]  # 返回最大值或最小值
        sorted = [False, True]  # 返回是否排序
        dtypes = [torch.qint8, torch.quint8]  # 数据类型
        is_nhwc = [False, True]  # 输入是否为NHWC格式

        # 生成所有可能的测试用例组合
        test_cases = itertools.product(x_dims, sides, dims, largest, sorted, dtypes, is_nhwc)
        k = 2
        # 遍历每个测试用例
        for x_dim, side, dim, larg, sort, dtype, nhwc in test_cases:
            if nhwc and x_dim != 4:  # 如果输入是NHWC格式，确保维度为4
                continue
            if dim >= x_dim:  # 确保要执行topk操作的维度存在
                continue
            shape = [side] * x_dim
            # 获取随机张量及其量化参数
            X, scale, zp = _get_random_tensor_and_q_params(shape, 1.0, dtype)
            qX = torch.quantize_per_tensor(X, scale, zp, dtype)

            if nhwc:
                qX = qX.permute([0, 3, 1, 2])  # 调整张量维度顺序为NHWC
                X = np.transpose(X, [0, 3, 1, 2])  # 调整Numpy数组的维度顺序为NHWC

            # 对未量化的张量执行topk操作
            unquantized_out = torch.topk(qX.dequantize(), k, dim=dim, largest=larg, sorted=sort)

            # 创建与X相同的量化张量和索引张量
            values = torch.quantize_per_tensor(X, scale, zp, dtype)
            indices = torch.tensor(X).long()

            # 对量化的张量执行topk操作
            quantized_out = torch.topk(qX, k, dim=dim, largest=larg, sorted=sort)

            # 断言未量化和量化结果的长度相同，并且接近（值部分）或相等（索引部分）
            assert len(unquantized_out) == len(quantized_out)
            torch.testing.assert_close(quantized_out[0].dequantize(), unquantized_out[0])
            torch.testing.assert_close(quantized_out[1], unquantized_out[1])

    """Tests quantize concatenation (both fused and not)."""
    # 使用假设进行张量的拼接量化测试（包括融合和非融合形式）
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           num=st.integers(1, 4),
           dim=st.integers(1, 4),
           relu=st.booleans())
    # 定义一个用于测试 torch.cat 操作的方法，针对量化的输入数据进行处理
    def test_cat(self, X, num, dim, relu):
        # 初始化两个空列表，用于存储量化后的张量
        tensors_q = []
        tensors_ref = []

        # 解构输入元组 X，获取其中的数据、缩放因子、零点值和数据类型
        X, (scale, zero_point, torch_type) = X

        # 假设 dim 小于 X 的维度数
        assume(dim < X.ndim)

        # 将 numpy 数组 X 转换为 PyTorch 张量
        X = torch.from_numpy(X)

        # 创建一个新的形状数组，与 X 的形状相同但指定维度 dim 为 0
        new_shape = np.array(X.shape)
        new_shape[dim] = 0

        # 遍历 num 次，生成量化后的张量 tensors_q 和未量化的张量 tensors_ref
        for idx in range(num):
            tensors_q.append(torch.quantize_per_tensor(X, scale, zero_point, torch_type))
            tensors_ref.append(X)
            # 更新 new_shape[dim]，累加未量化的张量 tensors_ref 的 dim 维度大小

        # 将未量化的张量 tensors_ref 沿指定维度 dim 进行拼接
        cat_ref = torch.cat(tensors_ref, dim=dim)
        # 对拼接后的张量 cat_ref 进行量化处理
        cat_ref = torch.quantize_per_tensor(cat_ref, scale, zero_point, torch_type)
        # 将量化后的张量 cat_ref 还原为浮点数张量
        cat_ref = cat_ref.dequantize()

        # 如果 relu 为 True，则对 cat_ref 执行 ReLU 激活函数
        if relu:
            cat_ref = F.relu(cat_ref)
            # 设置量化操作的函数和输出函数
            q_cat_op = torch.ops.quantized.cat_relu
            q_cat_out_op = torch.ops.quantized.cat_relu_out
        else:
            # 否则设置默认的量化操作的函数和输出函数
            q_cat_op = torch.ops.quantized.cat
            q_cat_out_op = torch.ops.quantized.cat_out

        # 使用量化后的张量 tensors_q 进行 torch.ops.quantized.cat 操作
        cat_q = q_cat_op(tensors_q, dim=dim, scale=scale, zero_point=zero_point)
        # 将量化后的张量 cat_q 还原为浮点数张量
        cat_q = cat_q.dequantize()

        # 使用 NumPy 测试库断言，验证 cat_ref 和 cat_q 数组内容相等
        np.testing.assert_equal(cat_ref.numpy(), cat_q.numpy())

        # 创建一个空的仿射量化张量 cat_q_out，形状由 new_shape 指定，缩放因子和零点值与输入相同
        cat_q_out = torch._empty_affine_quantized(
            list(new_shape), scale=scale, zero_point=zero_point, dtype=torch_type)
        # 使用 torch.ops.quantized.cat_out 操作将 tensors_q 中的张量拼接到 cat_q_out 中
        q_cat_out_op(tensors_q, dim=dim, out=cat_q_out)
        # 将量化后的张量 cat_q_out 还原为浮点数张量
        cat_q_out = cat_q_out.dequantize()
        # 使用 NumPy 测试库断言，验证 cat_ref 和 cat_q_out 数组内容相等
        np.testing.assert_equal(cat_ref.numpy(), cat_q_out.numpy())

        # 测试在通道量化的张量上进行 cat 操作时是否会抛出 RuntimeError 异常
        ch_axis = 1
        scales = torch.from_numpy(np.array([1.0] * X.shape[ch_axis]))
        scales = scales.to(torch.float64)
        zero_points = torch.from_numpy(np.array([0] * X.shape[ch_axis]))
        zero_points = zero_points.to(torch.long)
        # 使用 torch.quantize_per_channel 在通道 ch_axis 上对张量 X 进行量化
        tensors_q[0] = torch.quantize_per_channel(
            X, scales, zero_points, axis=ch_axis, dtype=torch_type)
        # 断言在执行 torch.ops.quantized.cat 操作时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "supported.*cat"):
            cat_q = q_cat_op(tensors_q, dim=ch_axis, scale=scale, zero_point=zero_point)
    def test_interpolate(self, X, size, mode, scale_factor, align_corners, nhwc_layout):
        """
        This test cover upsample_nearest2d and upsample_bilinear2d
        """
        # 解包输入的数据 X，获取 scale、zero_point 和 torch_type
        X, (scale, zero_point, torch_type) = X

        # 如果 scale_factor 不为 None，则 size 设置为 None
        if scale_factor is not None:
            size = None
        # 如果 mode 是 "nearest" 或 "nearest-exact"，则 align_corners 设置为 None
        if mode in ("nearest", "nearest-exact"):
            align_corners = None

        # 如果 nhwc_layout 为 True，则执行以下操作
        if nhwc_layout:
            # 如果 X 的第二维度小于 176，则按照比例增加其大小到 176
            if X.shape[1] < 176:
                X = np.repeat(X, 176 / X.shape[1], 1)

            # 将 X 转置为 NHWC 布局并确保其内存连续性，转换为 Torch 张量格式
            X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))
            X = torch.from_numpy(X_nchw).permute([0, 3, 1, 2])

            # 对 Torch 张量 X 进行量化，并将其维度重新排列为 NCHW
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type).permute([0, 3, 1, 2])
        else:
            # 如果 nhwc_layout 不为 True，则直接将 X 转换为 Torch 张量
            X = torch.from_numpy(X)
            # 对 Torch 张量 X 进行量化
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        # 使用 torch.nn.functional.interpolate 函数进行插值操作，得到参考的输出张量 X_ref
        X_ref = torch.nn.functional.interpolate(
            qX.int_repr().to(torch.float), size=size, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

        # 待测试的操作集合，包括 torch.nn.functional.interpolate 和 torch.ao.nn.quantized.functional.interpolate
        ops_under_test = {
            "nn.functional": torch.nn.functional.interpolate,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.interpolate,
        }
        # 错误消息模板
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        # 遍历每个操作名称和对应的操作函数
        for name, op in ops_under_test.items():
            # 执行操作，得到量化后的输出张量 qX_hat
            qX_hat = op(qX, size=size, scale_factor=scale_factor,
                        mode=mode, align_corners=align_corners)
            # 断言量化后的输出张量 qX_hat 应与参考输出 X_ref 相等，使用指定的容差进行比较
            self.assertEqual(X_ref, qX_hat.int_repr(), atol=1.0, rtol=0,
                             msg=f"{name} results are off: qX_hat={qX_hat.int_repr()} X_ref={X_ref}",
                             exact_dtype=False)
            # 断言量化的 scale 应与预期的 scale 相等
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            # 断言量化的 zero_point 应与预期的 zero_point 相等
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))
    # 定义一个测试函数，用于测试三维插值函数 upsample_nearest3d
    def test_interpolate3d(self, X, size, mode, scale_factor, align_corners, nhwc_layout):
        """
        This test cover upsample_nearest3d
        """
        # 解包参数 X，获取其中的数据、缩放比例、零点和数据类型信息
        X, (scale, zero_point, torch_type) = X
        # 如果 scale_factor 不为 None，则 size 应该为 None
        if scale_factor is not None:
            size = None

        # 将 align_corners 设为 None
        align_corners = None

        # 如果 nhwc_layout 为 True，执行以下操作
        if nhwc_layout:
            # 如果 X 的第二维度小于 176，则将其在第二维度上重复扩展到 176
            if X.shape[1] < 176:
                X = np.repeat(X, 176 / X.shape[1], 1)

            # 将 X 转置为以 NCHW 格式的连续数组，并转换为 Torch 张量
            X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 4, 1]))
            X = torch.from_numpy(X_nchw).permute([0, 4, 1, 2, 3])

            # 对 Torch 张量进行量化处理，按照给定的 scale、zero_point 和 dtype
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type).permute([0, 4, 1, 2, 3])
        else:
            # 如果 nhwc_layout 不为 True，直接将 X 转换为 Torch 张量
            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        # 计算参考结果 X_ref，通过对 qX 进行插值处理
        X_ref = torch.nn.functional.interpolate(
            qX.int_repr().to(torch.float), size=size, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

        # 定义待测试的操作和对应的函数映射
        ops_under_test = {
            "nn.functional": torch.nn.functional.interpolate,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.interpolate,
        }

        # 定义错误消息的格式
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"

        # 遍历 ops_under_test 中的每个操作和对应的函数
        for name, op in ops_under_test.items():
            # 计算当前操作的预测值 qX_hat，根据给定的参数进行操作
            qX_hat = op(qX, size=size, scale_factor=scale_factor,
                        mode=mode, align_corners=align_corners)
            # 断言当前操作的结果与参考值 X_ref 相等，设置允许的误差范围
            self.assertEqual(X_ref, qX_hat.int_repr(), atol=1.0, rtol=0,
                             msg=f"{name} results are off: qX_hat={qX_hat.int_repr()}, X_ref={X_ref}", exact_dtype=False)
            # 检查当前操作的量化参数 scale 是否与预期值 scale 相等
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            # 检查当前操作的量化参数 zero_point 是否与预期值 zero_point 相等
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))

    """Tests quantize concatenation (both fused and not)."""
    # 定义一个测试量化连接的函数，包括融合和非融合版本
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           relu=st.booleans())
    # 测试 NHWC 格式的张量拼接功能，其中 X 是一个元组，包含 NHWC 数据和量化参数
    def test_cat_nhwc(self, X, relu):
        # X 是 NHWC 格式的张量，解包并获取其量化参数 (scale, zero_point, torch_type)
        X, (scale, zero_point, torch_type) = X

        # 将 X 在通道维度上重复，使得通道数大于 64
        X = np.repeat(X, 70 / X.shape[3], 3)
        X = torch.from_numpy(np.ascontiguousarray(X))
        Y = X.clone()  # 克隆 X 到 Y
        Y = torch.from_numpy(np.ascontiguousarray(Y))

        # 在 qcat 中添加了一个快速路径：当输入共享相同的 scale 和 zero_point 时，
        # 将执行直接的内存复制，而不是解量化-拼接-量化的过程。

        # 遍历两种不同的 scale 值组合
        for scaleX, scaleY in ((scale, scale), (scale, scale * 1.1)):
            # 使用 torch.quantize_per_tensor 对 X 和 Y 进行量化，并通过 permute 将 NHWC 转换为 NCHW 形式存储
            qX = torch.quantize_per_tensor(X, scaleX, zero_point, torch_type).permute([0, 3, 1, 2])
            qY = torch.quantize_per_tensor(Y, scaleY, zero_point, torch_type).permute([0, 3, 1, 2])

            # 构建参考值 ref，通过解量化后的 qX 和 qY 拼接得到
            ref = torch.cat([qX.dequantize(), qY.dequantize()], dim=1)
            if relu:
                ref[ref < 0] = 0.0  # 如果启用了 relu，对 ref 中小于 0 的值置为 0

            # 将 ref 重新量化，并使用相同的 scale 和 zero_point
            ref = torch.quantize_per_tensor(ref, scale=scale, zero_point=zero_point, dtype=torch_type)

            if relu:
                # 如果启用了 relu，使用 quantized.cat_relu 函数拼接 qX 和 qY
                out = torch.ops.quantized.cat_relu(
                    [qX, qY], dim=1, scale=scale, zero_point=zero_point)
            else:
                # 否则，使用 quantized.cat 函数拼接 qX 和 qY
                out = torch.ops.quantized.cat([qX, qY], dim=1, scale=scale, zero_point=zero_point)

            # 断言 out 的解量化结果与 ref 的解量化结果相近
            torch.testing.assert_close(out.dequantize(), ref.dequantize())
            self.assertNotEqual(out.stride(), sorted(out.stride()))

    # 重写测试引擎
    @override_qengines
    def test_mean(self):
        # 定义多个测试参数的组合
        scale_list = (1, 0.25)
        zero_point_list = (0, 2)
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4), (4, 4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        dims = ((), (-1,), (0,), (1,), (2,), (3,), (0, 1), (1, 2), (3, 4))
        
        op = torch.mean  # 使用 torch.mean 函数

        # 使用 itertools 生成所有可能的测试用例
        test_cases = itertools.product(scale_list, zero_point_list, shapes, dtypes, dims)
        
        # 遍历所有测试用例
        for scale, zp, shape, dtype, dim in test_cases:
            # 如果 dim 中的任意一个维度超过了张量 shape 的维度数，跳过该测试用例
            if not all(d < len(shape) for d in dim):
                continue
            
            # 创建一个随机数据张量 X，并量化为 qX
            X = torch.randn(*shape) * 10
            qX = torch.quantize_per_tensor(X, scale, zp, dtype)
            
            # 对 qX 进行 torch.mean 操作，并解量化得到 Y
            Y = op(qX.dequantize(), dim)
            Y = torch.quantize_per_tensor(Y, scale, zp, dtype).dequantize()
            
            # 对 qX 进行 torch.mean 量化操作，并与解量化的 Y 进行断言比较
            qY = op(qX, dim)
            self.assertEqual(Y, qY.dequantize())

    # 如果没有 QNNPACK，则跳过测试
    @skipIfNoQNNPACK
    @given(keep=st.booleans())
    def test_quantized_mean_qnnpack(self, keep):
        # 使用 qnnpack 引擎覆盖量化计算
        with override_quantized_engine("qnnpack"):
            # 为了满足 pytorch_q8gavgpool_ukernel_up8xm__sse2() 在 ASAN 下对4字节对齐的需求，使用4的倍数大小
            in_dim = (4, 4, 4, 4)
            # 根据 keep 参数确定输出维度
            if keep:
                out_dim = (4, 4, 1, 1)
            else:
                out_dim = (4, 4)
            # 创建输入张量 X 和输出张量 Y，均为全1
            X = torch.ones(in_dim)
            Y = torch.ones(out_dim)
            # 对张量 X 和 Y 进行量化，设置量化参数 scale=0.2, zero_point=0, 数据类型为 quint8
            XQ = torch.quantize_per_tensor(X, scale=0.2, zero_point=0, dtype=torch.quint8)
            YQ = torch.quantize_per_tensor(Y, scale=0.2, zero_point=0, dtype=torch.quint8)
            # 计算 XQ 的均值，维度为 (2, 3)，根据 keep 参数决定是否保持维度
            MQ = XQ.mean((2, 3), keepdim=keep)
            # 断言 MQ 与 YQ 相等
            self.assertTrue(torch.equal(MQ, YQ))

    @override_qengines
    def test_std(self):
        # 定义测试用例的多种参数组合
        scale_list = (1, 0.25)
        zero_point_list = (0, 2)
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4), (4, 4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        dims = ((), (-1,), (0,), (1,), (2,), (3,), (0, 1), (1, 2), (3, 4))
        unbiased_list = (True, False)
        keep_dim_list = (True, False)
        # 生成所有参数组合的测试用例
        test_cases = itertools.product(scale_list, zero_point_list, shapes,
                                       dtypes, dims, unbiased_list, keep_dim_list)
        # 设置操作为 torch.std
        op = torch.std
        # 遍历测试用例
        for scale, zp, shape, dtype, dim, unbiased, keep_dim in test_cases:
            # 如果维度 dim 中有任何一个超出 shape 的维度范围，则跳过该测试用例
            if not all(d < len(shape) for d in dim):
                continue
            # 生成随机输入张量 X，乘以10以增大数值范围
            X = torch.randn(*shape) * 10
            # 对张量 X 进行量化，使用给定的 scale, zp, dtype 参数
            qX = torch.quantize_per_tensor(X, scale, zp, dtype)
            # 对张量 X 去量化，然后计算标准差
            Y = op(qX.dequantize(), dim, unbiased, keep_dim)
            # 将计算结果 Y 重新量化，并去量化，得到 qY
            Y = torch.quantize_per_tensor(Y, scale, zp, dtype).dequantize()
            qY = op(qX, dim, unbiased, keep_dim)
            # 断言去量化后的 Y 和 qY 相等
            self.assertEqual(Y, qY.dequantize())

    """Tests the correctness of the quantized equal op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()),
           X2=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                        qparams=hu.qparams()),
           X_per_channel=st.booleans(),
           X2_per_channel=st.booleans())
    """Tests quantized equal op with inputs X, X2, X_per_channel, X2_per_channel."""
    def test_equal(self, X, X2, X_per_channel, X2_per_channel):
        # 解包输入参数 X
        X, X_params = X
        (scale, zero_point, torch_type) = X_params
        # 解包输入参数 X2
        X2, X2_params = X2
        (scale2, zero_point2, torch_type2) = X2_params

        # 将 X 转换为 PyTorch 张量
        X = torch.from_numpy(X)
        # 根据 X_per_channel 决定量化策略
        if X_per_channel:
            X_scheme = 'per_channel'
            channels = X.shape[-1]
            # 执行按通道量化
            qX = torch.quantize_per_channel(
                X,
                scales=torch.tensor([scale] * channels),
                zero_points=torch.tensor([zero_point] * channels),
                dtype=torch_type,
                axis=X.ndim - 1)
        else:
            X_scheme = 'per_tensor'
            # 执行整体量化
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        # 将 X2 转换为 PyTorch 张量
        X2 = torch.from_numpy(X2)
        # 根据 X2_per_channel 决定量化策略
        if X2_per_channel:
            X2_scheme = 'per_channel'
            channels = X2.shape[-1]
            # 执行按通道量化
            qX2 = torch.quantize_per_channel(
                X2,
                scales=torch.tensor([scale2] * channels),
                zero_points=torch.tensor([zero_point2] * channels),
                dtype=torch_type2,
                axis=X2.ndim - 1)
        else:
            X2_scheme = 'per_tensor'
            # 执行整体量化
            qX2 = torch.quantize_per_tensor(X2, scale=scale2, zero_point=zero_point2,
                                            dtype=torch_type2)

        # 定义比较函数 equal_ref
        def equal_ref(qX, qX2):
            # 检查量化策略是否相同
            if qX.qscheme() != qX2.qscheme():
                return False
            # 检查形状是否相同
            if qX.shape != qX2.shape:
                return False
            # 检查数据类型是否相同
            if qX.dtype != qX2.dtype:
                return False
            # 对于每种量化策略，进行进一步的比较
            if qX.qscheme() == torch.per_tensor_affine:
                if qX.q_scale() != qX2.q_scale():
                    return False
                if qX.q_zero_point() != qX2.q_zero_point():
                    return False
            elif qX.qscheme() == torch.per_channel_affine:
                if (qX.q_per_channel_scales() !=
                   qX2.q_per_channel_scales()).any():
                    return False
                if (qX.q_per_channel_zero_points() !=
                   qX2.q_per_channel_zero_points()).any():
                    return False
            else:
                raise NotImplementedError("Don't know what to do with",
                                          qX.qscheme())
            # 检查整数表示是否相同
            if (qX.int_repr().to(float) != qX2.int_repr().to(float)).any():
                return False
            return True

        # 断言测试结果
        self.assertEqual(qX.equal(qX), equal_ref(qX, qX))
        self.assertEqual(qX.equal(qX2), equal_ref(qX, qX2))

    """Tests quantized equal op with input of non-quantized tensor."""
    def test_quantized_equal(self,):
        # 创建一个随机张量 x
        x = torch.rand(1)
        # 对 x 进行量化
        y = torch.quantize_per_tensor(x, scale=0.5, zero_point=0, dtype=torch.qint8)
        # 断言 x 和 y 不相等
        self.assertTrue(not torch.equal(x, y))
        self.assertTrue(not torch.equal(y, x))

    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    @skipIfNoFBGEMM
    def test_batch_norm_relu(self):
        # 假设使用假设过慢，手动创建测试用例
        max_sides = (2, 3, 4, 5)  # 定义可能的最大维度组合
        side_lens = (1, 8, 11)  # 定义可能的边长组合
        torch_types = (torch.qint8, torch.quint8)  # 定义可能的数据类型
        combined = [max_sides, side_lens, torch_types]
        test_cases = itertools.product(*combined)  # 生成所有测试用例的组合

        with override_quantized_engine("fbgemm"):  # 使用"fbgemm"引擎覆盖量化引擎
            for test_case in test_cases:  # 遍历每个测试用例
                max_side, side_len, torch_type = test_case
                Y_zero_point = 1  # 设置输出张量的零点
                Y_scale = 0.5  # 设置输出张量的缩放因子

                shapes = [side_len] * max_side  # 根据测试用例生成张量形状
                X, scale_x, zero_point_x = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)  # 获取随机张量和量化参数
                dtype_x = torch_type  # 设置输入张量的数据类型

                c = X.shape[1]  # 获取输入张量的通道数
                mean = torch.rand(c).float()  # 随机生成均值张量
                var = torch.rand(c).float()  # 随机生成方差张量
                weight = torch.rand(c).float()  # 随机生成权重张量
                bias = torch.rand(c).float()  # 随机生成偏置张量
                eps = 0.001  # 设置小的浮点数，用于数值稳定性
                qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)  # 对输入张量进行量化
                if len(X.shape) == 2 or len(X.shape) == 3:
                    qy = torch.ops.quantized.batch_norm1d_relu(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)  # 对二维或三维张量进行批量归一化和ReLU操作
                elif len(X.shape) == 4:
                    qy = torch.ops.quantized.batch_norm2d_relu(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)  # 对四维张量进行批量归一化和ReLU操作
                else:
                    qy = torch.ops.quantized.batch_norm3d_relu(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)  # 对更高维度的张量进行批量归一化和ReLU操作

                float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                         running_mean=mean, running_var=var,
                                         training=False, momentum=0, eps=eps).numpy()  # 计算浮点参考结果

                float_ref_relu = float_ref.copy()  # 复制浮点参考结果
                float_ref_relu[float_ref < 0] = 0  # 应用ReLU操作到浮点参考结果
                quantize_ref = torch.quantize_per_tensor(
                    torch.from_numpy(float_ref_relu), Y_scale, Y_zero_point, dtype_x)  # 对ReLU后的浮点参考结果进行量化
                self.assertEqual(
                    qy.int_repr().numpy(),
                    quantize_ref.int_repr().numpy(),
                    msg=f"{qy} vs {quantize_ref}")  # 断言量化结果与参考结果的一致性

    @skipIfNoFBGEMM
    # 定义测试方法：验证批量归一化操作的正确性
    def test_batch_norm(self):
        # 假设的测试过程太慢，手动创建测试用例
        max_sides = (2, 3, 4, 5)  # 最大边数的组合
        side_lens = (1, 8, 11)  # 边长的不同值
        torch_types = (torch.qint8, torch.quint8)  # 不同的 torch 张量类型
        combined = [max_sides, side_lens, torch_types]  # 组合成一个列表
        test_cases = itertools.product(*combined)  # 生成所有组合的测试用例

        # 使用 "fbgemm" 引擎覆盖量化引擎
        with override_quantized_engine("fbgemm"):
            # 遍历所有测试用例
            for test_case in test_cases:
                max_side, side_len, torch_type = test_case
                Y_zero_point = 1  # 输出张量的零点
                Y_scale = 0.5  # 输出张量的比例尺

                shapes = [side_len] * max_side  # 根据最大边数和边长创建形状列表
                X, scale_x, zero_point_x = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)  # 获取随机张量及其量化参数
                dtype_x = torch_type  # 设置输入张量的数据类型

                c = X.shape[1]  # 获取张量 X 的通道数
                mean = torch.rand(c).float()  # 随机生成均值张量
                var = torch.rand(c).float()  # 随机生成方差张量
                weight = torch.rand(c).float()  # 随机生成权重张量
                bias = torch.rand(c).float()  # 随机生成偏置张量
                eps = 0.001  # 一个很小的数，用于数值稳定性
                qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)  # 对输入张量 X 进行量化
                if len(X.shape) == 2 or len(X.shape) == 3:
                    qy = torch.ops.quantized.batch_norm1d(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)  # 执行一维批量归一化操作
                elif len(X.shape) == 4:
                    qy = torch.ops.quantized.batch_norm2d(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)  # 执行二维批量归一化操作
                elif len(X.shape) == 5:
                    qy = torch.ops.quantized.batch_norm3d(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)  # 执行三维批量归一化操作

                # 计算浮点参考值
                float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                         running_mean=mean, running_var=var, training=False,
                                         momentum=0, eps=eps)
                # 对浮点参考值进行量化
                quantize_ref = torch.quantize_per_tensor(float_ref, Y_scale, Y_zero_point, dtype_x)
                # 断言量化后的输出是否与参考值一致
                self.assertEqual(
                    qy.int_repr().numpy(), quantize_ref.int_repr().numpy(),
                    msg=f"{qy} vs {quantize_ref}")

    # 使用装饰器覆盖量化引擎的测试方法
    @override_qengines
    @override_qengines
    def test_linear_bias_unpack(self):
        """
        验证 LinearPackedParamBase 的 bias() 和 unpack() API 的正确性。
        """
        bias_float = torch.ones(2, dtype=torch.float)  # 创建全为 1 的浮点偏置张量
        w = torch.randn((2, 2), dtype=torch.float)  # 创建随机浮点权重张量
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)  # 对权重张量进行量化
        w_packed = torch.ops.quantized.linear_prepack(qw, bias_float)  # 使用量化权重和浮点偏置进行线性打包

        # 测试 bias() 方法
        self.assertEqual(w_packed.bias(), bias_float)
        # 测试 unpack() 方法
        self.assertEqual(w_packed.unpack()[0], qw)
    def test_advanced_indexing(self):
        """
        Verifies that the x[:, [0], :, :] syntax works for quantized tensors.
        """
        # 遍历不同的数据类型，测试量化张量的高级索引
        for dtype in (torch.qint8, torch.quint8, torch.qint32):
            scale = 0.1
            zp = 0
            # 创建一个随机张量并进行量化
            x_q = torch.quantize_per_tensor(
                torch.randn(1, 4, 4, 4), scale, zp, dtype)
            # 将量化后的张量还原成浮点张量作为参考
            x_fp32 = x_q.dequantize()

            # 单个维度，单个索引
            x_q_s1 = x_q[:, [0], :, :]
            x_fp32_s1 = x_fp32[:, [0], :, :]
            # 使用浮点参考重新量化得到量化张量
            x_fp32_s1_ref = \
                torch.quantize_per_tensor(x_fp32_s1, scale, zp, dtype)
            # 断言量化后的张量与参考张量相等
            self.assertEqual(x_q_s1, x_fp32_s1_ref)

            # 多个维度，单个索引
            x_q_s2 = x_q[:, [0], [2], :]
            x_fp32_s2 = x_fp32[:, [0], [2], :]
            x_fp32_s2_ref = \
                torch.quantize_per_tensor(x_fp32_s2, scale, zp, dtype)
            self.assertEqual(x_q_s2, x_fp32_s2_ref)

            # 单个维度，多个索引
            x_q_s3 = x_q[:, [2, 0, 1], :, :]
            x_fp32_s3 = x_fp32[:, [2, 0, 1], :, :]
            x_fp32_s3_ref = \
                torch.quantize_per_tensor(x_fp32_s3, scale, zp, dtype)
            self.assertEqual(x_q_s3, x_fp32_s3_ref)

            # 多个维度，多个索引
            x_q_s4 = x_q[:, [2, 0, 1], :, [1]]
            x_fp32_s4 = x_fp32[:, [2, 0, 1], :, [1]]
            x_fp32_s4_ref = \
                torch.quantize_per_tensor(x_fp32_s4, scale, zp, dtype)
            self.assertEqual(x_q_s4, x_fp32_s4_ref)
class TestDynamicQuantizedOps(TestCase):
    """Tests the correctness of the dynamic quantized linear and linear_relu op."""

    # 使用 override_qengines 装饰器，用于测试时覆盖量化引擎设置
    @override_qengines
    # 使用 given 装饰器定义参数化测试的参数
    @given(
        batch_size=st.integers(1, 4),  # 批量大小在1到4之间的整数
        input_channels=st.integers(16, 32),  # 输入通道数在16到32之间的整数
        output_channels=st.integers(4, 8),  # 输出通道数在4到8之间的整数
        use_bias=st.booleans(),  # 是否使用偏置的布尔值
        use_relu=st.booleans(),  # 是否使用ReLU的布尔值
        use_multi_dim_input=st.booleans(),  # 是否使用多维输入的布尔值
        use_channelwise=st.booleans(),  # 是否按通道独立量化的布尔值
        reduce_range=st.booleans()  # 是否减少量化范围的布尔值
    )
    # 使用 skipIfNoFBGEMM 装饰器，如果没有FBGEMM，则跳过测试
    @skipIfNoFBGEMM
    # 再次使用 given 装饰器定义另一组参数化测试的参数
    @given(
        batch_size=st.integers(1, 4),  # 批量大小在1到4之间的整数
        input_channels=st.integers(16, 32),  # 输入通道数在16到32之间的整数
        output_channels=st.integers(4, 8),  # 输出通道数在4到8之间的整数
    )
    # 定义测试方法，用于测试量化线性操作的遗留版本
    def test_qlinear_legacy(self, batch_size, input_channels, output_channels):
        # 初始化量化输入的比例因子和零点
        X_scale = 1.0
        X_zp = 0
        # 输入数据的最小值和最大值
        X_value_min = 0
        X_value_max = 255
        # 生成随机的量化输入数据 X_q0，数据类型为 uint8
        X_q0 = np.round(np.random.rand(batch_size, input_channels) * (
            X_value_max - X_value_min) + X_value_min
        ).astype(np.uint8)
        # 将第一行第一列的值设置为最小值
        X_q0[0, 0] = X_value_min
        # 将第一行第二列的值设置为最大值
        X_q0[0, 1] = X_value_max

        # 初始化量化权重的比例因子和零点
        W_scale = 1.0
        W_zp = 0
        # 权重数据的最小值和最大值
        W_value_min = -128
        W_value_max = 127
        # 生成随机的量化权重数据 W_q0，数据类型为 int8
        W_q0 = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.int8)
        # 将第一行第一列的值设置为最小值
        W_q0[0, 0] = W_value_min
        # 将第二行第一列的值设置为最大值
        W_q0[1, 0] = W_value_max

        # 初始化量化偏置的最小值和最大值
        b_value_min = -10
        b_value_max = 10
        # 生成随机的量化偏置数据 b_q0，数据类型为 int32
        b_q0 = np.round(
            np.random.rand(output_channels) * (b_value_max - b_value_min) +
            b_value_min
        ).astype(np.int32)

        # 调用遗留版本的量化线性操作，避免 VPMADDUBSW 溢出
        avoid_vpmaddubsw_overflow_linear(
            batch_size,
            input_channels,
            output_channels,
            X_q0,
            X_value_min,
            X_value_max,
            W_q0,
            W_value_min,
            W_value_max,
        )

        # 将量化后的输入 X_q0 反量化为浮点数张量 X_fp32，并转换为 torch 的浮点数类型
        X_fp32 = torch.from_numpy(_dequantize(X_q0, X_scale, X_zp)).to(dtype=torch.float)
        # 将量化后的权重 W_q0 反量化为浮点数张量 W_fp32，并转换为 torch 的浮点数类型
        W_fp32 = torch.from_numpy(_dequantize(W_q0, W_scale, W_zp)).to(dtype=torch.float)
        # 将量化后的偏置 b_q0 反量化为浮点数张量 b_fp32，并转换为 torch 的浮点数类型
        b_fp32 = torch.from_numpy(
            _dequantize(b_q0, X_scale * W_scale, 0)
        ).to(dtype=torch.float)

        # 计算动态量化参数 W_scale 和 W_zp
        W_scale, W_zp = _calculate_dynamic_qparams(W_fp32, torch.qint8)
        # 对权重 W_fp32 进行量化，使用动态参数 W_scale 和 W_zp
        W_q = torch.quantize_per_tensor(W_fp32, scale=W_scale, zero_point=W_zp, dtype=torch.qint8)

        # 观察 X_fp32 并确定 X_scale 和 X_zero_point，这应与动态线性操作的内部匹配
        X_scale, X_zp = _calculate_dynamic_qparams(X_fp32, torch.quint8)
        # 对输入 X_fp32 进行量化，使用动态参数 X_scale 和 X_zp
        X_q = torch.quantize_per_tensor(X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)

        # 使用 FBGEMM 库对权重 W_q 进行量化线性操作，并利用预打包的权重 W_prepack
        W_int8, col_offsets, W_scale, W_zp = torch.fbgemm_linear_quantize_weight(W_q.dequantize())
        # 对量化后的权重 W_int8 进行打包处理，以提高计算效率
        W_prepack = torch.fbgemm_pack_quantized_matrix(W_int8.clone(), W_int8.size(1), W_int8.size(0))
        # 使用预打包的权重进行量化线性操作
        Y_fp32 = torch.fbgemm_linear_int8_weight(
            X_q.dequantize(), W_q.dequantize(), W_prepack, col_offsets,
            W_scale, W_zp, b_fp32)

        # 使用 PyTorch 内置函数对比计算得到的 Y_fp32 与参考 Y_fp32_ref
        Y_fp32_ref = F.linear(X_q.dequantize(), W_q.dequantize(), b_fp32)
        # 断言量化线性操作的结果 Y_fp32 与参考 Y_fp32_ref 相等
        self.assertEqual(Y_fp32, Y_fp32_ref,
                         msg="torch.ops.quantized.fbgemm_linear_dynamic results are off")

    # 使用 FBGEMM 库时，如果不支持则跳过测试
    @skipIfNoFBGEMM
    @given(
        input_channels=st.integers(16, 32),
        output_channels=st.integers(4, 8),
        exponent=st.integers(0, 8))
    # 测试使用量化线性运算的预打包和解包功能，使用FP16数值
    def test_linear_prepack_fp16_numerics(self, input_channels, output_channels, exponent):
        # 创建随机权重矩阵w，并按照给定的指数缩放
        w = torch.randn(output_channels, input_channels) * 10**exponent
        bias = None
        # 使用torch.ops.quantized.linear_prepack_fp16对权重w进行预打包
        w_packed_fp16 = torch.ops.quantized.linear_prepack_fp16(w, bias)
        # 使用torch.ops.quantized.linear_unpack_fp16对打包后的w进行解包
        w_unpacked_fp16 = torch.ops.quantized.linear_unpack_fp16(w_packed_fp16)
        # 将原始的w转换为FP16格式再转回FP32
        w_fp16 = w.to(torch.float16).to(torch.float32)
        # 断言解包后的w与转换后的w_fp16相等
        self.assertTrue(torch.equal(w_fp16, w_unpacked_fp16[0]))

    @skipIfNoFBGEMM
    # 测试使用动态量化线性运算的FP16版本
    def test_qlinear_dynamic_fp16(self):

        options = itertools.product(
            (2, 4),         # batch_size 批量大小
            (4, 5, 12),     # input_channels 输入通道数
            (4, 7, 8),      # output_channels 输出通道数
            (True, False),  # use_bias 是否使用偏置
            (True, False),  # use_relu 是否使用ReLU
        )
        for batch_size, input_channels, output_channels, use_bias, use_relu in options:
            # 获取量化线性预打包函数的引用
            qlinear_prepack = torch.ops.quantized.linear_prepack_fp16
            if use_relu:
                # 根据是否使用ReLU选择动态量化线性函数
                qlinear_dynamic = torch.ops.quantized.linear_relu_dynamic_fp16
            else:
                qlinear_dynamic = torch.ops.quantized.linear_dynamic_fp16

            x = torch.randn(batch_size, input_channels)
            w = torch.randn(output_channels, input_channels)
            bias = torch.randn(output_channels) if use_bias else None

            # 使用预打包函数对权重w进行打包
            w_packed = qlinear_prepack(w, bias)
            # 对输入x应用动态量化线性运算
            out = qlinear_dynamic(x, w_packed)

            # qlinear_dynamic_fp16使用FP32激活张量和FP16权重张量
            # 输出为FP32
            w_fp16 = w.to(torch.float16).to(torch.float32)
            ref = F.linear(x, w_fp16, bias)
            if use_relu:
                ref.relu_()

            # 断言输出结果out与参考结果ref相等
            self.assertEqual(out, ref)

    @skipIfNoFBGEMM
    # 测试使用未打包权重的动态FP16量化线性运算
    def test_unpacked_qlinear_dynamic_fp16(self):

        options = itertools.product(
            (2, 4),         # batch_size 批量大小
            (4, 5, 12),     # input_channels 输入通道数
            (4, 7, 8),      # output_channels 输出通道数
        )
        for batch_size, input_channels, output_channels in options:
            # 获取动态FP16量化线性运算的引用
            qlinear_dynamic = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight

            x = torch.randn(batch_size, input_channels)
            w = torch.randn(output_channels, input_channels)
            bias = torch.randn(output_channels)

            # 应用未打包权重的动态FP16量化线性运算
            out = qlinear_dynamic(x, w, bias)

            # qlinear_dynamic_fp16使用FP32激活张量和FP16权重张量
            # 输出为FP32
            w_fp16 = w.to(torch.float16).to(torch.float32)
            ref = F.linear(x, w_fp16, bias)

            # 断言输出结果out与参考结果ref相等
            self.assertEqual(out, ref)

    @skipIfNoFBGEMM
    # 测试动态FP16量化线性运算的操作检查
    def test_unpacked_qlinear_dynamic_fp16_opcheck(self):
        # 获取默认动态FP16量化线性运算的引用
        qlinear_dynamic = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight.default

        x = torch.randn(4, 4, device='cpu')
        w = torch.randn(4, 4, device='cpu')
        bias = torch.randn(4, device='cpu')

        # 使用opcheck函数对动态FP16量化线性运算进行操作检查
        opcheck(qlinear_dynamic, (x, w, bias))
    # 定义测试函数，用于测试 wrapped_fbgemm_linear_fp16 函数的功能
    def test_wrapped_fbgemm_linear_fp16(self):
        # 生成所有可能的参数组合，用于测试
        options = itertools.product(
            (2, 4),         # batch_size 批量大小
            (4, 5),         # input_channels 输入通道数
            (4, 7),         # output_channels 输出通道数
        )
        # 遍历参数组合
        for batch_size, input_channels, output_channels in options:
            # 获取 quantized 操作的函数
            pack_op = torch.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16
            linear_op = torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight

            # 创建随机输入张量
            x = torch.randn(batch_size, input_channels)
            # 创建随机权重张量和偏置张量
            w = torch.randn(output_channels, input_channels)
            bias = torch.randn(output_channels)

            # 对权重进行打包操作
            w_packed = pack_op(w)
            # 调用线性操作函数计算输出
            out = linear_op(x, w_packed, bias, output_channels)

            # 将权重转换为半精度浮点数，并转换回单精度浮点数进行参考计算
            w_fp16 = w.to(torch.float16).to(torch.float32)
            ref = F.linear(x, w_fp16, bias)

            # 使用断言检查计算结果是否一致
            self.assertEqual(out, ref)

    # 标记为需要跳过的测试，如果没有 FBGEMM 库
    @skipIfNoFBGEMM
    def test_wrapped_fbgemm_pack_gemm_matrix_fp16_pt2_compliant(self):
        # 此处不使用 opcheck，因为我们测试的操作 _quantized.wrapped_fbgemm_pack_gemm_matrix_fp16
        # 由于其生成的 C 结构，输出不是确定性的
        #
        # 这只是一个临时解决方案，长期来看，我们应该能够原生支持 PT2 和 torchbind
        def func(X, W, B):
            # 对权重进行打包操作
            packed_W = torch.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(W)
            # 调用线性操作函数计算输出
            return torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight(X, packed_W, B, W.size(0))

        # 创建随机输入张量、权重张量和偏置张量
        x = torch.randn(1, 4, device="cpu")
        w = torch.randn(4, 4, device="cpu")
        b = torch.zeros(4, device="cpu")

        # 使用函数计算参考输出
        ref_out = func(x, w, b)

        # 编译函数并计算输出
        compiled = torch.compile(func)
        compiled_out = compiled(x, w, b)

        # 使用断言检查编译后输出与参考输出是否一致
        self.assertEqual(ref_out, compiled_out)

    """Tests the correctness of the dynamic quantized lstm/gru."""
    # 定义一个方法，用于获取RNN的输入数据张量
    def _get_rnn_inputs(self, seq_len, num_batches, input_size, hidden_size, num_directions, reduce_range):
        # 创建一个随机张量作为输入数据，形状为(seq_len, num_batches, input_size)
        X = torch.randn(seq_len, num_batches, input_size)
        # 调用_calculate_dynamic_qparams函数计算输入数据的量化参数s和z，使用torch.quint8类型进行量化
        s, z = _calculate_dynamic_qparams(X, torch.quint8, reduce_range)
        # 对输入数据X进行量化，得到量化后的张量Xq
        Xq = torch.quantize_per_tensor(X, s, z, torch.quint8)

        # 根据num_directions的值初始化隐藏状态H和细胞状态C的张量
        if num_directions == 1:
            # 当num_directions为1时，使用随机张量初始化H和C，形状为(num_directions, num_batches, hidden_size)
            H = torch.randn(num_directions, num_batches, hidden_size)
            C = torch.randn(num_directions, num_batches, hidden_size)
        else:
            # 当num_directions不为1时，使用全零张量初始化H和C，形状为(num_directions, num_batches, hidden_size)
            H = torch.zeros(num_directions, num_batches, hidden_size)
            C = torch.zeros(num_directions, num_batches, hidden_size)

        # 分别计算H和C的量化参数s和z
        s, z = _calculate_dynamic_qparams(H, torch.quint8, reduce_range)
        # 对H进行量化，得到量化后的张量Hq
        Hq = torch.quantize_per_tensor(H, s, z, torch.quint8)
        # 分别计算C的量化参数s和z
        s, z = _calculate_dynamic_qparams(C, torch.quint8, reduce_range)
        # 对C进行量化，得到量化后的张量Cq
        Cq = torch.quantize_per_tensor(C, s, z, torch.quint8)
        # 返回量化后的输入数据Xq、隐藏状态Hq和细胞状态Cq
        return Xq, Hq, Cq

    # 定义一个方法，用于获取RNN的权重和偏置
    def _get_rnn_weights_and_bias(self, input_size, hidden_size, num_directions, per_channel_quant, rnn_type):
        # 根据RNN类型从字典中获取隐藏状态的倍数
        hidden_mult_map = {'LSTM': 4, 'LSTMCell': 4, 'GRU': 3, 'GRUCell': 3, 'RNNTanh': 2, 'RNNReLU': 2}
        hidden_mult = hidden_mult_map[rnn_type]
        # 使用随机张量初始化权重weights1和weights2，形状分别为(hidden_mult * hidden_size, input_size)和(hidden_mult * hidden_size, hidden_size)
        weights1 = torch.randn(hidden_mult * hidden_size, input_size)
        weights2 = torch.randn(hidden_mult * hidden_size, hidden_size)
        # 创建scale1和scale2张量，并初始化为0.1和0.3的全1张量
        scale1 = 0.1 * torch.ones([weights1.size()[0]])
        scale2 = 0.3 * torch.ones([weights2.size()[0]])
        # 创建zero_point1和zero_point2张量，并初始化为全零张量
        zero_point1 = torch.zeros(scale1.size()).to(int)
        zero_point2 = torch.zeros(scale2.size()).to(int)
        # 创建偏置b1张量，并初始化为全零张量
        b1 = torch.zeros(hidden_mult * hidden_size)
        
        # 根据per_channel_quant的值选择不同的量化方式对权重进行量化
        if per_channel_quant:
            # 对权重weights1和weights2进行通道间量化，得到量化后的张量Wq1和Wq2
            Wq1 = torch.quantize_per_channel(weights1, scale1, zero_point1, 0, torch.qint8)
            Wq2 = torch.quantize_per_channel(weights2, scale2, zero_point2, 0, torch.qint8)
        else:
            # 对权重weights1和weights2进行整体量化，使用scale1和zero_point1对weights1进行量化，使用scale2和zero_point2对weights2进行量化
            Wq1 = torch.quantize_per_tensor(weights1, float(scale1[0]), int(zero_point1[0]), torch.qint8)
            Wq2 = torch.quantize_per_tensor(weights2, float(scale2[0]), int(zero_point2[0]), torch.qint8)
        
        # 返回量化后的权重Wq1、Wq2、偏置b1和b1
        return Wq1, Wq2, b1, b1

    # 使用hypothesis的给定参数生成器定义一个测试参数
    @given(
        num_batches=st.integers(1, 4),
        input_size=st.integers(16, 32),
        hidden_size=st.integers(4, 8),
        num_directions=st.integers(1, 2),
        per_channel_quant=st.booleans())
    # 覆盖量化引擎的设置
    @override_qengines
    # 使用hypothesis的给定参数生成器定义另一个测试参数
    @given(
        num_batches=st.integers(1, 4),
        input_size=st.integers(16, 32),
        hidden_size=st.integers(4, 8),
        per_channel_quant=st.booleans())
    # 覆盖量化引擎的设置
    @override_qengines
    # 定义一个测试方法，用于测试动态量化卷积操作的实现
    def _test_qconv_op_impl(self, q_mod, dq_op, dim, dtype):
        # 目标是展示动态操作与计算参数、量化输入、量化操作、反量化输出相同

        # 如果当前使用的量化引擎是 QNNPACK，并且在 PPC 或者 UBSAN 环境中，则不支持该测试
        if qengine_is_qnnpack() and (IS_PPC or TEST_WITH_UBSAN):
            return  # QNNPACK 不支持此功能

        # 根据当前使用的量化引擎设置 reduce_range 变量
        if qengine_is_qnnpack():
            reduce_range = False
        else:
            reduce_range = True

        # 创建一个随机的 float32 张量 X_fp32，维度为 dim 给定的多维度
        X_fp32 = torch.randn(*([2] * dim))
        # 计算动态量化参数 s (缩放因子) 和 z (零点)
        s, z = _calculate_dynamic_qparams(X_fp32, dtype, reduce_range)

        # 使用给定的量化模块 q_mod 创建一个量化后的模块实例
        quantized_module = q_mod(2, 3, 1)
        # 获取量化模块的打包参数
        packed_params = quantized_module._packed_params

        # 设置量化模块的缩放因子和零点
        quantized_module.scale, quantized_module.zero_point = s, z

        # 对输入张量 X_fp32 进行量化，得到量化后的张量 X_q
        X_q = torch.quantize_per_tensor(X_fp32, s, z, dtype)
        # 对量化后的输入 X_q 进行量化模块的前向操作，得到量化后的输出 Y_q_ref
        Y_q_ref = quantized_module(X_q)
        # 对量化后的输出 Y_q_ref 进行反量化操作，得到反量化的输出 Y_ref
        Y_ref = torch.dequantize(Y_q_ref)

        # 对输入张量 X_q 进行反量化操作，得到反量化的输入 X_dq
        X_dq = torch.dequantize(X_q)
        # 使用动态量化操作 dq_op 对反量化的输入 X_dq 进行操作，得到输出 Y
        Y = dq_op(X_dq, packed_params, reduce_range)

        # 断言输出 Y 与反量化的输出 Y_ref 相等
        self.assertEqual(Y, Y_ref)

    # 覆盖量化引擎的设置，用于测试动态卷积1D操作
    @override_qengines
    def test_dynamic_conv1d(self):
        # 设置量化模块和动态量化操作
        q_mod = torch.ao.nn.quantized.Conv1d
        dq_op = torch.ops.quantized.conv1d_dynamic
        dim = 3  # 维度设置为 3
        dtype = torch.quint8  # 数据类型设置为 quint8

        # 调用测试方法 _test_qconv_op_impl 进行测试
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    # 覆盖量化引擎的设置，用于测试动态卷积2D操作
    @override_qengines
    def test_dynamic_conv2d(self):
        # 设置量化模块和动态量化操作
        q_mod = torch.ao.nn.quantized.Conv2d
        dq_op = torch.ops.quantized.conv2d_dynamic
        dim = 4  # 维度设置为 4
        dtype = torch.quint8  # 数据类型设置为 quint8

        # 调用测试方法 _test_qconv_op_impl 进行测试
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    # 覆盖量化引擎的设置，用于测试动态卷积3D操作
    @override_qengines
    def test_dynamic_conv3d(self):
        # 设置量化模块和动态量化操作
        q_mod = torch.ao.nn.quantized.Conv3d
        dq_op = torch.ops.quantized.conv3d_dynamic
        dim = 5  # 维度设置为 5
        dtype = torch.quint8  # 数据类型设置为 quint8

        # 调用测试方法 _test_qconv_op_impl 进行测试
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    # 覆盖量化引擎的设置，用于测试动态转置卷积1D操作
    @override_qengines
    def test_dynamic_convtranspose1d(self):
        # 设置量化模块和动态量化操作
        q_mod = torch.ao.nn.quantized.ConvTranspose1d
        dq_op = torch.ops.quantized.conv_transpose1d_dynamic
        dim = 3  # 维度设置为 3
        dtype = torch.quint8  # 数据类型设置为 quint8

        # 调用测试方法 _test_qconv_op_impl 进行测试
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    # 覆盖量化引擎的设置，用于测试动态转置卷积2D操作
    @override_qengines
    def test_dynamic_convtranspose2d(self):
        # 设置量化模块和动态量化操作
        q_mod = torch.ao.nn.quantized.ConvTranspose2d
        dq_op = torch.ops.quantized.conv_transpose2d_dynamic
        dim = 4  # 维度设置为 4
        dtype = torch.quint8  # 数据类型设置为 quint8

        # 调用测试方法 _test_qconv_op_impl 进行测试
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    # 覆盖量化引擎的设置，用于测试动态转置卷积3D操作
    @override_qengines
    def test_dynamic_convtranspose3d(self):
        # 设置量化模块和动态量化操作
        q_mod = torch.ao.nn.quantized.ConvTranspose3d
        dq_op = torch.ops.quantized.conv_transpose3d_dynamic
        dim = 5  # 维度设置为 5
        dtype = torch.quint8  # 数据类型设置为 quint8

        # 如果当前使用的量化引擎是 QNNPACK，则返回，因为存在问题需要修复
        if qengine_is_qnnpack():
            return  # TODO: fix MakeDeConvOutputShape overflowing for convT3d with qnnpack

        # 调用测试方法 _test_qconv_op_impl 进行测试
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)
class TestQuantizedLinear(TestCase):
    """Tests the correctness of the quantized linear op."""

    @override_qengines
    def test_qlinear(self):
        # 定义测试用例中的不同参数列表
        batch_size_list = [1, 4]
        input_channels_list = [16, 32]
        output_channels_list = [4, 8]
        use_bias_list = [True, False]
        use_multi_dim_input_list = [True, False]
        use_channelwise_list = [True, False]
        post_op = 'none'
        
        # 生成所有可能的测试用例组合
        cases = itertools.product(batch_size_list, input_channels_list, output_channels_list,
                                  use_bias_list, use_multi_dim_input_list, use_channelwise_list)
        
        # 遍历所有测试用例组合
        for batch_size, input_channels, output_channels, use_bias, \
            use_multi_dim_input, use_channelwise in cases:
            
            # 调用具体的测试实现方法来执行测试
            self._test_qlinear_impl(batch_size, input_channels, output_channels,
                                    use_bias, post_op, use_multi_dim_input, use_channelwise)

    """Tests the correctness of the quantized linear_relu op."""

    @override_qengines
    def test_qlinear_relu(self):
        # 定义测试用例中的不同参数列表
        batch_size_list = [1, 4]
        input_channels_list = [16, 32]
        output_channels_list = [4, 8]
        use_bias_list = [True, False]
        use_multi_dim_input_list = [True, False]
        use_channelwise_list = [True, False]
        post_op = 'relu'
        
        # 生成所有可能的测试用例组合
        cases = itertools.product(batch_size_list, input_channels_list, output_channels_list,
                                  use_bias_list, use_multi_dim_input_list, use_channelwise_list)
        
        # 遍历所有测试用例组合
        for batch_size, input_channels, output_channels, use_bias, \
            use_multi_dim_input, use_channelwise in cases:
            
            # 调用具体的测试实现方法来执行测试
            self._test_qlinear_impl(batch_size, input_channels, output_channels,
                                    use_bias, post_op, use_multi_dim_input, use_channelwise)

    @given(batch_size=st.integers(1, 4),
           input_channels=st.integers(16, 32),
           output_channels=st.integers(4, 8),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_multi_dim_input=st.booleans(),
           use_channelwise=st.booleans())
    @skipIfNoFBGEMM
    @given(batch_size=st.integers(1, 4),
           # in cudnn v. 8.4.0, there is a limitation that input channels
           # should be a multiple of 4 for int8 tensors. in cudnn v.8.3.3
           # this should be a multiple of 16
           input_channels=st.sampled_from([4, 8, 12, 16, 32]),
           # constraints on output channels appear to be relax, as it seems we can use any positive integer here
           # except 1. It is not clear why 1 will not work. TODO: check with Yang
           output_channels=st.integers(2, 36),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_multi_dim_input=st.booleans(),
           use_channelwise=st.sampled_from([False]))  # channelwise currently not supported for qlinear cudnn
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    # 使用unittest.skipIf装饰器，根据条件跳过测试用例，条件为TEST_CUDNN和torch.backends.cudnn.version()为90100时跳过
    # 同时也根据条件跳过，条件为不满足SM80OrLater要求时跳过
    # 如果测试运行在ROCM平台上，则跳过测试用例
    # TODO: 与yang确认关于CUDNN标志的问题

    """Tests the correctness of the quantized::linear_unpack op."""
    # 使用@given装饰器和参数W、use_channelwise，用于参数化测试，W包含形状和量化参数，use_channelwise是一个布尔值
    # 使用@override_qengines装饰器，用于指定测试中使用的量化引擎
    def test_qlinear_unpack(self, W, use_channelwise):
        # 解包参数W和量化参数，分别获取W、(W_scale, W_zp, torch_type)
        W, (W_scale, W_zp, torch_type) = W

        # 如果use_channelwise为True，则计算输出通道数，并生成随机的量化参数W_scales和W_zps
        if use_channelwise:
            output_channels = W.shape[0]
            W_scales = torch.rand(output_channels).to(torch.double)
            W_zps = torch.round(torch.rand(output_channels)
                                * 100 - 50).to(torch.int64)

        # 获取量化操作符qlinear_prepack和qlinear_unpack
        qlinear_prepack = torch.ops.quantized.linear_prepack
        qlinear_unpack = torch.ops.quantized.linear_unpack

        # 如果量化引擎为ONEDNN，则进行特定的量化设置
        if qengine_is_onednn():
            if use_channelwise:
                # 如果使用通道间量化，将W_zps置为全零
                W_zps = torch.zeros(output_channels).to(torch.int64)
            else:
                # 否则将W_zp置为0
                W_zp = 0

        # 将参数W转换为PyTorch张量
        W = torch.from_numpy(W)

        # 根据use_channelwise参数选择合适的量化函数对W进行量化，得到量化后的张量W_q
        if use_channelwise:
            W_q = torch.quantize_per_channel(
                W, W_scales, W_zps, 0, dtype=torch_type)
        else:
            W_q = torch.quantize_per_tensor(W, scale=W_scale, zero_point=W_zp,
                                            dtype=torch_type)

        # 使用qlinear_prepack对量化后的权重W_q进行预打包操作
        W_prepack = qlinear_prepack(W_q)

        # 使用qlinear_unpack对预打包的权重W_prepack进行解包操作，获取原始的量化权重W_q_origin
        W_q_origin = qlinear_unpack(W_prepack)[0]

        # 断言量化后的权重W_q和解包后的W_q_origin在整数表示上相等
        np.testing.assert_equal(W_q.int_repr(), W_q_origin.int_repr().numpy())

        # 如果使用通道间量化，则进一步断言量化参数在不同张量之间几乎相等
        if use_channelwise:
            np.testing.assert_array_almost_equal(np.float32(W_q.q_per_channel_scales().numpy()),
                                                 np.float32(
                                                     W_q_origin.q_per_channel_scales().numpy()),
                                                 decimal=4)
            np.testing.assert_equal(W_q.q_per_channel_zero_points(
            ).numpy(), W_q_origin.q_per_channel_zero_points().numpy())
        else:
            # 否则直接断言量化参数在不同张量之间相等
            np.testing.assert_equal(np.float32(
                W_q.q_scale()), np.float32(W_q_origin.q_scale()))
            np.testing.assert_equal(
                W_q.q_zero_point(), W_q_origin.q_zero_point())
    def test_qlinear_qnnpack_free_memory_and_unpack(self, W):
        # 确保当前量化引擎是 QNNPACK
        assert qengine_is_qnnpack
        # 解包输入的权重 W，包括权重数据、缩放因子、零点和数据类型
        W, (W_scale, W_zp, torch_type) = W
        # 获取量化线性预打包和解包操作符
        qlinear_prepack = torch.ops.quantized.linear_prepack
        qlinear_unpack = torch.ops.quantized.linear_unpack

        # 将 numpy 数组 W 转换为 PyTorch 张量
        W = torch.from_numpy(W)
        # 如果当前量化引擎是 ONEDNN，则强制权重的零点为 0
        if qengine_is_onednn():
            W_zp = 0
        # 对权重 W 进行量化，得到量化后的权重 W_q
        W_q = torch.quantize_per_tensor(W, scale=W_scale, zero_point=W_zp, dtype=torch_type)
        # 使用预打包操作符对量化后的权重进行预打包
        W_prepack = qlinear_prepack(W_q)
        # 创建一个用于测试的虚拟输入张量
        dummy_input = torch.randn((1, W.shape[1]))
        # 通过在后端运行矩阵乘法，确保我们释放原始张量
        torch.ops.quantized.linear_dynamic(dummy_input, W_prepack)
        torch.ops.quantized.linear_dynamic(dummy_input, W_prepack)
        # 在此步骤中，应从数据指针中恢复原始张量
        W_q_origin = qlinear_unpack(W_prepack)[0]
        # 断言相等性
        np.testing.assert_equal(W_q.int_repr(), W_q_origin.int_repr().numpy())
        np.testing.assert_equal(np.float32(
            W_q.q_scale()), np.float32(W_q_origin.q_scale()))
        np.testing.assert_equal(
            W_q.q_zero_point(), W_q_origin.q_zero_point())

    @skipIfNoONEDNN
    def test_qlinear_leaky_relu(self):
        # 使用 ONEDNN 引擎执行测试
        with override_quantized_engine('onednn'):
            # 定义各种测试用例的参数组合
            batch_size_list = [1, 4]
            input_channels_list = [16, 32]
            output_channels_list = [4, 8]
            use_bias_list = [True, False]
            use_multi_dim_input_list = [True, False]
            use_channelwise_list = [True, False]
            negative_slopes_list = [0.01, 0.05]
            post_op = 'leaky_relu'
            cases = itertools.product(batch_size_list, input_channels_list, output_channels_list,
                                      use_bias_list, use_multi_dim_input_list,
                                      use_channelwise_list, negative_slopes_list)
            # 对每个测试用例执行量化线性层与 leaky relu 后操作的测试
            for batch_size, input_channels, output_channels, use_bias, \
                    use_multi_dim_input, use_channelwise, neg_slope in cases:
                self._test_qlinear_impl(batch_size, input_channels, output_channels,
                                        use_bias, post_op, use_multi_dim_input,
                                        use_channelwise, negative_slope=neg_slope)

    @skipIfNoONEDNN
    def test_qlinear_tanh(self):
        # 使用 onednn 引擎覆盖量化操作
        with override_quantized_engine('onednn'):
            # 定义测试用例的各种参数列表
            batch_size_list = [1, 4]
            input_channels_list = [16, 32]
            output_channels_list = [4, 8]
            use_bias_list = [True, False]
            use_multi_dim_input_list = [True, False]
            use_channelwise_list = [True, False]
            post_op = 'tanh'
            # 生成测试用例的所有组合
            cases = itertools.product(batch_size_list, input_channels_list,
                                      output_channels_list, use_bias_list,
                                      use_multi_dim_input_list, use_channelwise_list)
            # 遍历所有测试用例
            for batch_size, input_channels, output_channels, use_bias, \
                    use_multi_dim_input, use_channelwise in cases:
                # 调用具体的量化线性操作测试方法
                self._test_qlinear_impl(batch_size, input_channels, output_channels,
                                        use_bias, post_op, use_multi_dim_input,
                                        use_channelwise)

    def _test_qlinear_pt2e_helper(
        self,
        qlinear_op,
        post_op="none",
        unary_post_op_args=(),
        post_op_algorithms=("none"),
    ):
        # 定义量化线性操作的辅助测试方法
        pass

    @skipIfNoONEDNN
    def test_qlinear_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise 操作
        qlinear = torch.ops.onednn.qlinear_pointwise
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "none"
        self._test_qlinear_pt2e_helper(qlinear, "none")

    @skipIfNoONEDNN
    def test_qlinear_relu_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise 操作
        qlinear = torch.ops.onednn.qlinear_pointwise
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "relu"
        self._test_qlinear_pt2e_helper(qlinear, "relu")

    @skipIfNoONEDNN
    def test_qlinear_gelu_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise 操作
        qlinear = torch.ops.onednn.qlinear_pointwise
        # 定义可选的后处理算法列表
        post_op_algorithms = ['none', 'tanh']
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "gelu"，并传入后处理算法列表
        self._test_qlinear_pt2e_helper(qlinear, "gelu", post_op_algorithms=post_op_algorithms)

    @skipIfNoONEDNN
    def test_qlinear_sum_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise.binary 操作
        qlinear = torch.ops.onednn.qlinear_pointwise.binary
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "sum"
        self._test_qlinear_pt2e_helper(qlinear, "sum")

    @skipIfNoONEDNN
    def test_qlinear_sum_relu_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise.binary 操作
        qlinear = torch.ops.onednn.qlinear_pointwise.binary
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "sum_relu"
        self._test_qlinear_pt2e_helper(qlinear, "sum_relu")

    @skipIfNoONEDNN
    def test_qlinear_add_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise.binary 操作
        qlinear = torch.ops.onednn.qlinear_pointwise.binary
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "add"
        self._test_qlinear_pt2e_helper(qlinear, "add")

    @skipIfNoONEDNN
    def test_qlinear_add_relu_pt2e(self):
        # 获取 onednn 下的 qlinear_pointwise.binary 操作
        qlinear = torch.ops.onednn.qlinear_pointwise.binary
        # 调用量化线性操作的辅助测试方法，指定后处理操作为 "add_relu"
        self._test_qlinear_pt2e_helper(qlinear, "add_relu")
# 使用 unittest 装饰器跳过在 macOS 上已知的测试失败情况
@unittest.skipIf(IS_MACOS, "Known test failure on Mac.")
class TestQuantizedEmbeddingOps(TestCase):

    # 定义测试函数 _test_embedding_bag_unpack_impl，用于测试嵌入包解压缩功能
    def _test_embedding_bag_unpack_impl(self, pack_fn, unpack_fn, bit_rate, optimized_qparams, weights):
        # 获取权重数据类型
        data_type = weights.dtype

        # 设置量化类型为 quint8
        qtype = torch.quint8
        # 根据 bit_rate 选择不同的打包函数
        if bit_rate == 8:
            w_packed = pack_fn(weights)
        else:
            w_packed = pack_fn(weights, optimized_qparams=optimized_qparams)
        
        # 解包打包后的权重数据
        w_unpacked = unpack_fn(w_packed)

        # 如果 bit_rate 是 8 或者 4，并且数据类型不是 torch.float16
        if (bit_rate == 8 or bit_rate == 4) and data_type != torch.float16:
            # torch.quantize_per_channel 目前不支持 float16。

            # 观察权重数据，处理 3D 嵌入（例如堆叠的嵌入组合）在通道正交维度上。
            if len(weights.shape) > 2:
                stacked_shape = list(weights.size())
                stacked_shape[1] *= stacked_shape[0]
                obs_weights = weights.reshape(stacked_shape[1:])
            else:
                obs_weights = weights

            # 创建 min-max 观察器以模拟原始函数中的量化过程
            obs = PerChannelMinMaxObserver(dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(obs_weights)
            # 计算权重张量的量化参数（scale 和 zero point）
            qparams = obs.calculate_qparams()

            # 如果 bit_rate 是 4，则设置量化类型为 torch.quint4x2
            if bit_rate == 4:
                qtype = torch.quint4x2
            
            # 对权重进行按通道量化为 8 位
            qweight = torch.quantize_per_channel(obs_weights, qparams[0], qparams[1], axis=0, dtype=qtype)
            
            # 使用 quantized.embedding_bag_prepack 函数打包量化后的权重
            real_packed_weight = torch.ops.quantized.embedding_bag_prepack(qweight)
            
            # 断言 real_packed_weight 是否为 torch._C.ScriptObject 类型
            self.assertEqual(isinstance(real_packed_weight, torch._C.ScriptObject), True)
            
            # 使用 quantized.embedding_bag_unpack 函数解包打包后的权重
            unpacked_weight = torch.ops.quantized.embedding_bag_unpack(real_packed_weight)
            
            # 断言解包后的权重与量化后的权重在整数表示上是否一致
            self.assertEqual(unpacked_weight.int_repr().numpy(), qweight.int_repr().numpy())
            
            # 断言解包后的权重的每通道量化尺度是否与量化后的权重一致
            self.assertEqual(unpacked_weight.q_per_channel_scales(), qweight.q_per_channel_scales())
            
            # 断言解包后的权重的每通道量化零点是否与量化后的权重一致
            self.assertEqual(unpacked_weight.q_per_channel_zero_points(), qweight.q_per_channel_zero_points())
    def _test_embedding_bag_unpack_fn(self, pack_fn, unpack_fn, num_embeddings, embedding_dim, bit_rate,
                                      optimized_qparams, num_batches, data_type=np.float32):
        # 当 num_batches = 1 时，创建一个二维张量
        unsplit_weight = torch.from_numpy((np.random.random_sample((
            num_batches, num_embeddings, embedding_dim)).squeeze() + 1).astype(np.float32))
        
        # 测试未分割的权重数据（内存格式为 `contiguous`）
        self._test_embedding_bag_unpack_impl(pack_fn, unpack_fn, bit_rate, optimized_qparams, unsplit_weight)
        
        # 测试分割后的权重数据（内存格式不是 `contiguous`）
        split_dim = len(unsplit_weight.shape) - 2
        split_weights = torch.split(unsplit_weight, 1, dim=split_dim)
        for weight in split_weights:
            self._test_embedding_bag_unpack_impl(pack_fn, unpack_fn, bit_rate, optimized_qparams, weight)


    """ Tests the correctness of the embedding_bag_8bit quantized operator """
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    # 测试 embedding_bag_8bit 量化操作的正确性
    def test_embedding_bag_byte(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        self.embedding_bag_rowwise_offsets_run(
            8, num_embeddings, embedding_dim, num_offsets,
            use_32bit_indices, use_32bit_offsets,
            enable_per_sample_weights, include_last_offset,
            fallback_to_no_sparse,
            sparsity=sparsity, atol=0.005, rtol=1e-3)

    """ Tests the correctness of the embedding_bag_4bit quantized operator """
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    # 测试 embedding_bag_4bit 量化操作的正确性
    def test_embedding_bag_4bit(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        self.embedding_bag_rowwise_offsets_run(
            4, num_embeddings, embedding_dim, num_offsets,
            use_32bit_indices, use_32bit_offsets,
            enable_per_sample_weights, include_last_offset,
            fallback_to_no_sparse,
            sparsity=sparsity, atol=0.005, rtol=1e-3)
    """ Tests the correctness of the embedding_bag_4bit quantized operator """
    # 测试嵌入袋（embedding_bag）4比特量化运算符的正确性
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 8 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    def test_embedding_bag_4bit(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        # 调用内部方法进行行偏移量运算，测试4比特嵌入袋算子的实现
        self.embedding_bag_rowwise_offsets_run(4, num_embeddings,
                                               embedding_dim, num_offsets,
                                               use_32bit_indices, use_32bit_offsets,
                                               enable_per_sample_weights,
                                               include_last_offset,
                                               fallback_to_no_sparse,
                                               sparsity=sparsity,
                                               atol=0.1, rtol=1e-2)

    """ Tests the correctness of the embedding_bag_2bit quantized operator """
    # 测试嵌入袋（embedding_bag）2比特量化运算符的正确性
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 8 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    def test_embedding_bag_2bit(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        # 调用内部方法进行行偏移量运算，测试2比特嵌入袋算子的实现
        self.embedding_bag_rowwise_offsets_run(2, num_embeddings,
                                               embedding_dim, num_offsets,
                                               use_32bit_indices, use_32bit_offsets,
                                               enable_per_sample_weights,
                                               include_last_offset,
                                               fallback_to_no_sparse,
                                               sparsity=sparsity,
                                               atol=1.0, rtol=1e-1)

    """ Tests the correctness of the quantized 8 bit embedding lookup operator """
    # 测试量化的8比特嵌入查找算子的正确性
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0))
    def test_embedding(self, num_embeddings, embedding_dim):
        # 定义量化数据类型列表
        dtypes = [torch.quint8, torch.quint4x2]
        # 定义量化操作列表
        quant_ops = [torch.ops.quantized.embedding_byte, torch.ops.quantized.embedding_4bit]
        # 定义绝对误差列表
        atols = [0.005, 0.1]
        # 定义相对误差列表
        rtols = [1e-3, 1e-2]
        # 获取预打包操作符
        prepack_op = torch.ops.quantized.embedding_bag_prepack
        # 遍历量化操作、数据类型、误差参数和量化操作
        for quant_op, dtype, atol, rtol in zip(quant_ops, dtypes, atols, rtols):
            # 创建随机权重张量
            weights = torch.from_numpy((np.random.random_sample((
                num_embeddings, embedding_dim)) + 1).astype(np.float32))

            # 创建分通道最小-最大观察器实例
            obs = PerChannelMinMaxObserver(dtype=dtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            # 应用观察器到权重张量
            obs(weights)
            # 计算权重张量的量化参数（比例因子和零点）
            qparams = obs.calculate_qparams()

            # 对权重张量进行8位量化
            qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=dtype)
            # 定义最大段数和最大段长度
            max_segments = 5
            max_segment_length = 20
            # 随机生成段数
            num_lengths = np.random.randint(1, max_segments + 1)
            # 随机生成段长度数组
            lengths = np.random.randint(1, max_segment_length + 1,
                                        size=num_lengths).astype(np.int32)
            # 计算总索引数
            num_indices = np.sum(lengths)
            # 随机生成索引张量
            indices = torch.from_numpy(np.random.randint(
                low=0, high=num_embeddings, size=num_indices, dtype=np.int64))

            # 预打包量化权重
            packed_weight = prepack_op(qweight)
            # 调用量化操作符进行量化嵌入计算
            qresult = quant_op(packed_weight, indices, pruned_weights=False)

            # 使用非量化的torch.embedding函数计算参考结果
            ref = torch.embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
            # 断言量化计算结果与参考结果的接近程度
            torch.testing.assert_close(ref, qresult, atol=atol, rtol=rtol)

    def test_embedding_2d_indices(self):
        """
        Tests the case where 2D indices are passed into the operator
        In this case the operator computes the correct offsets argument.
        Output shape is dependent on the indices dimension.
        """
        # 定义量化操作符
        quant_op = torch.ops.quantized.embedding_byte
        # 获取预打包操作符
        prepack_op = torch.ops.quantized.embedding_bag_prepack

        # 创建包含2D索引的张量
        indices = torch.tensor([[9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8], [3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3]])
        # 创建随机权重张量
        weights = torch.randn(10, 12, dtype=torch.float32)

        # 使用分通道最小-最大观察器实例化观察器
        obs = PerChannelMinMaxObserver(dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
        # 应用观察器到权重张量
        obs(weights)
        # 计算权重张量的量化参数（比例因子和零点）
        qparams = obs.calculate_qparams()

        # 对权重张量进行通道量化
        qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=torch.quint8)
        # 预打包量化权重
        packed_weight = prepack_op(qweight)
        # 调用量化操作符进行量化嵌入计算
        qresult = quant_op(packed_weight, indices, pruned_weights=False)
        # 使用非量化的torch.embedding函数计算参考结果
        ref = torch.embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
        # 断言量化计算结果与参考结果的接近程度
        torch.testing.assert_close(ref, qresult, atol=0.05, rtol=1e-3)
    def test_embedding_bag_2d_indices(self):
        """
        Tests the case where 2D indices are passed into the operator
        In this case the operator computes the correct offsets argument.
        """
        # 定义测试用的2D索引张量
        indices = torch.tensor([[9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8], [3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3]])
        # 随机初始化权重张量
        weights = torch.randn(10, 12, dtype=torch.float32)

        # 创建EmbeddingBag实例，用于计算嵌入向量的和
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=10,
            embedding_dim=12,
            include_last_offset=False, _weight=weights,
            scale_grad_by_freq=False, mode='sum'
        )
        # 对给定的索引进行嵌入计算
        result = embedding_bag(indices)

        # 获取量化操作的函数和预打包的操作函数
        pt_op = torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        pt_prepack_op = torch.ops.quantized.embedding_bag_byte_prepack
        # 对权重进行预打包操作
        q_weights = pt_prepack_op(weights)
        # 使用量化操作函数计算量化结果
        qresult = pt_op(q_weights, indices, mode=0, pruned_weights=False)
        # 断言结果的接近程度
        torch.testing.assert_close(result, qresult, atol=0.05, rtol=1e-3)

        # 测试基于TorchBind的embedding_bag运算符
        # 创建基于通道的最小最大观察器，设定量化的参数
        obs = PerChannelMinMaxObserver(dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
        obs(weights)
        # 计算权重张量的量化参数（尺度和零点）
        qparams = obs.calculate_qparams()

        # 将权重量化为8位
        qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=torch.quint8)

        # 对量化的权重进行预打包
        packed_weight = torch.ops.quantized.embedding_bag_prepack(qweight)
        # 使用量化操作函数计算量化结果
        qresult = torch.ops.quantized.embedding_bag_byte(packed_weight, indices, mode=0)

        # 断言结果的接近程度
        torch.testing.assert_close(result, qresult, atol=0.05, rtol=1e-3)
# 定义一个测试类 TestQuantizedConv，继承自 TestCase
class TestQuantizedConv(TestCase):
    # 定义一个测试函数 _test_qconv_unpack_impl，接受 qconv_prepack_fn、qconv_unpack_fn、inputs、strides、i_pads、o_pads、channelwise 参数
    def _test_qconv_unpack_impl(self, qconv_prepack_fn, qconv_unpack_fn, inputs,
                                strides, i_pads, o_pads, channelwise):
        # 解包 inputs 中的数据
        (X_data, W_data, bias_data, groups, transposed) = inputs
        (X, (X_scale, X_zero_point, X_qtype)) = X_data
        (W, (W_scale, W_zero_point, W_qtype)) = W_data
        (bias, (bias_scale, bias_zero_point, bias_qtype)) = bias_data

        # 将 W 和 bias 转换为 float 类型的 Tensor
        W = torch.from_numpy(W).float()
        bias = torch.from_numpy(bias).float()
        
        # 如果 channelwise 为 True 且 transposed 为 True，则直接返回
        if channelwise and transposed:
            return
        
        # 如果使用 ONEDNN 引擎，则将 W_zero_point 设为 0，o_pads 设为全零列表
        if qengine_is_onednn():
            W_zero_point = 0
            o_pads = len(o_pads) * [0] if o_pads is not None else None
        
        # 如果 channelwise 为 True
        if channelwise:
            # 如果 transposed 为 True，则 output_channels 为 W 的第二维度
            if transposed:
                output_channels = W.shape[1]  # IC OC/G
            else:
                output_channels = W.shape[0]  # OC IC/G
            # 创建 W_scale 和 W_zero_point 的 Tensor
            W_scale = torch.tensor([W_scale] * output_channels)
            W_zero_point = torch.tensor([W_zero_point] * output_channels)
            # 根据 channelwise 进行量化
            W_q = torch.quantize_per_channel(
                W, scales=W_scale, zero_points=W_zero_point,
                axis=int(transposed), dtype=W_qtype)
        else:
            # 根据 per-tensor 进行量化
            W_q = torch.quantize_per_tensor(
                W, scale=W_scale, zero_point=W_zero_point, dtype=W_qtype)

        # 根据 strides 的类型确定 dilations
        if isinstance(strides, int):
            dilations = [1]
        else:
            dilations = (1,) * len(strides)

        # 根据 transposed 调用不同的 qconv_prepack_fn 函数
        if transposed:
            W_packed = qconv_prepack_fn(W_q, bias, strides, i_pads, o_pads,
                                        dilations, groups)
        else:
            W_packed = qconv_prepack_fn(W_q, bias, strides, i_pads, dilations,
                                        groups)
        # 解包 W_packed
        (W_unpacked, bias) = qconv_unpack_fn(W_packed)

        # 断言 W_q 和 W_unpacked 相等
        np.testing.assert_equal(W_q.int_repr().numpy(),
                                W_unpacked.int_repr().numpy())
        
        # 如果 channelwise 为 True，进行额外的断言
        if channelwise:
            np.testing.assert_array_almost_equal(
                np.float32(W_q.q_per_channel_scales().numpy()),
                np.float32(W_unpacked.q_per_channel_scales().numpy()),
                decimal=4)
            np.testing.assert_equal(W_q.q_per_channel_zero_points(
            ).numpy(), W_unpacked.q_per_channel_zero_points().numpy())
        else:
            # 如果 channelwise 为 False，进行额外的断言
            np.testing.assert_equal(np.float32(
                W_q.q_scale()), np.float32(W_unpacked.q_scale()))
            np.testing.assert_equal(
                W_q.q_zero_point(), W_unpacked.q_zero_point())
    def _make_qconv_tensors(
        self, batch_size, input_channels_per_group, input_feature_map_shape,
        output_channels_per_group, groups, kernels, strides, pads, dilations,
        X_scale, X_zero_point, W_scale, W_zero_point,
        use_bias, use_channelwise, use_transpose,
        device=torch.device("cpu"),
        input_dtype=torch.quint8,
        weight_dtype=torch.qint8,
    ):
        """构造量化卷积操作的张量。"""

    def _test_qconv_impl(
        self, qconv_fn, qconv_prepack_fn, conv_op, batch_size,
        input_channels_per_group, input_feature_map_shape,
        output_channels_per_group, groups, kernels, strides, pads, o_pads,
        dilations, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
        Y_zero_point, use_bias, post_op, use_channelwise, use_transpose,
        device=torch.device("cpu"),
        input_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        output_dtype=torch.quint8,
        X2_scale=1.0,
        X2_zero_point=128
    ):
        """测试量化卷积操作的正确性。"""

    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 300),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans())
    @override_qengines
    def test_qconv2d(
            self,
            batch_size,
            input_channels_per_group,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias,
            use_channelwise,
    ):
        input_channels = input_channels_per_group * groups  # 计算输入通道数
        output_channels = output_channels_per_group * groups  # 计算输出通道数
        kernels = (kernel_h, kernel_w)  # 设置卷积核大小
        strides = (stride_h, stride_w)  # 设置步长
        pads = (pad_h, pad_w)  # 设置填充
        dilations = (dilation, dilation)  # 设置扩张率

        qconv = torch.ops.quantized.conv2d  # 获取量化卷积操作符
        qconv_prepack = torch.ops.quantized.conv2d_prepack  # 获取预打包的量化卷积操作符
        conv_op = torch.nn.Conv2d(  # 创建普通的二维卷积层对象
            input_channels,
            output_channels,
            kernels,
            strides,
            pads,
            dilations,
            groups,
        )

        act_qdtypes = [torch.quint8]
        # 仅当 qengine 是 qnnpack 且 xnnpack 已启用时，支持 qint8
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        for X_qdtype in act_qdtypes:
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]  # 若为 qint8，将权重零点初始化为0

            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (height, width),
                output_channels_per_group, groups, kernels, strides, pads, None,
                dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, "none", use_channelwise, False, input_dtype=X_qdtype, output_dtype=X_qdtype)
        ):
        # 计算每个组内的输入通道数和输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        # 定义卷积核大小、步长、填充和扩展参数
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        pads = (pad_h, pad_w)
        dilations = (dilation, dilation)

        # 获取量化卷积操作和预打包函数
        qconv = torch.ops.quantized.conv2d_relu
        qconv_prepack = torch.ops.quantized.conv2d_prepack
        # 创建标准的二维卷积操作
        conv_op = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernels,
            strides,
            pads,
            dilations,
            groups,
        )

        # 定义激活量化数据类型列表，初始包含 torch.quint8
        act_qdtypes = [torch.quint8]
        # 当 qengine 是 qnnpack 并且 torch.backends.xnnpack 启用时，支持 torch.qint8
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        # 遍历激活量化数据类型列表
        for X_qdtype in act_qdtypes:
            # 如果激活量化数据类型为 torch.qint8，则将 W_zero_point 初始化为全零列表
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]

            # 调用测试量化卷积实现的函数，传入相关参数
            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (height, width),
                output_channels_per_group, groups, kernels, strides, pads, None,
                dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, "relu", use_channelwise, False, input_dtype=X_qdtype, output_dtype=X_qdtype)

    @skipIfNoONEDNN
    # 定义测试函数 test_qconv2d_add，用于测试量化卷积加操作
    def test_qconv2d_add(self):
        # 设定批处理大小
        batch_size = 3
        # 定义分组列表，用于分组卷积
        groups_list = [1, 10]
        # 每个分组内的输入通道数
        input_channels_per_group = 2
        # 每个分组内的输出通道数
        output_channels_per_group = 2
        # 输入图像的高度和宽度
        height = 10
        width = 10
        # 卷积核的高度和宽度
        kernel_h = 3
        kernel_w = 3
        # 卷积的步幅
        stride_h = 2
        stride_w = 2
        # 卷积的填充大小
        pad_h = 1
        pad_w = 1
        # 卷积的扩展因子（dilation）
        dilation = 1
        # 输入张量的缩放因子和零点
        X_scale = 1.5
        X_zero_point = 2
        # 权重的缩放因子和零点
        W_scale = [1.5]
        W_zero_point = [-3]
        # 输出张量的缩放因子和零点
        Y_scale = 4.2
        Y_zero_point = 0
        # 是否使用偏置的列表
        use_bias_list = [False, True]
        # 是否使用通道级量化的列表
        use_channelwise_list = [False, True]
        # 第二个输入张量的缩放因子
        X2_scale = 1.2
        # 第二个输入张量的零点列表
        X2_zero_point_list = [0, 4]
        
        # 遍历所有选项的组合
        options = itertools.product(groups_list, use_bias_list, use_channelwise_list, X2_zero_point_list)
        for groups, use_bias, use_channelwise, X2_zero_point in options:
            # 使用 ONEDNN 引擎覆盖量化引擎
            with override_quantized_engine('onednn'):
                # 计算实际的输入通道数和输出通道数
                input_channels = input_channels_per_group * groups
                output_channels = output_channels_per_group * groups
                # 定义卷积核大小、步幅、填充、扩展因子
                kernels = (kernel_h, kernel_w)
                strides = (stride_h, stride_w)
                pads = (pad_h, pad_w)
                dilations = (dilation, dilation)
                
                # 获取量化卷积加法操作的 torch 函数
                qconv = torch.ops.quantized.conv2d_add
                # 获取量化卷积预打包操作的 torch 函数
                qconv_prepack = torch.ops.quantized.conv2d_prepack
                # 创建标准的 PyTorch 卷积层对象
                conv_op = torch.nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernels,
                    strides,
                    pads,
                    dilations,
                    groups,
                )
                
                # 定义输入张量的数据类型为 quint8（8位无符号整数）
                X_qdtype = torch.quint8
                
                # 调用测试函数 _test_qconv_impl，测试量化卷积加操作的实现
                self._test_qconv_impl(
                    qconv, qconv_prepack, conv_op, batch_size,
                    input_channels_per_group, (height, width),
                    output_channels_per_group, groups, kernels, strides, pads, None,
                    dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                    Y_scale, Y_zero_point, use_bias, "add", use_channelwise, False,
                    input_dtype=X_qdtype, output_dtype=X_qdtype, X2_scale=X2_scale, X2_zero_point=X2_zero_point)

    # 当没有 ONEDNN 时跳过此测试
    @skipIfNoONEDNN
    def test_qconv2d_add_relu(self):
        batch_size = 3  # 批处理大小为3
        height = 10  # 输入图像高度为10
        width = 10  # 输入图像宽度为10
        groups_list = [1, 10]  # 分组数量的列表，用于测试不同分组情况
        input_channels_per_group = 2  # 每个分组的输入通道数为2
        output_channels_per_group = 2  # 每个分组的输出通道数为2
        kernel_h = 3  # 卷积核的高度为3
        kernel_w = 3  # 卷积核的宽度为3
        stride_h = 2  # 垂直方向的步长为2
        stride_w = 2  # 水平方向的步长为2
        pad_h = 1  # 垂直方向的填充为1
        pad_w = 1  # 水平方向的填充为1
        dilation = 1  # 卷积核的扩展系数为1
        X_scale = 1.5  # 输入张量的缩放因子为1.5
        X_zero_point = 2  # 输入张量的零点偏移为2
        W_scale = [1.5]  # 权重的缩放因子列表，包含一个元素1.5
        W_zero_point = [-3]  # 权重的零点偏移列表，包含一个元素-3
        Y_scale = 4.2  # 输出张量的缩放因子为4.2
        Y_zero_point = 0  # 输出张量的零点偏移为0
        use_bias_list = [False, True]  # 是否使用偏置的列表，用于测试两种情况
        use_channelwise_list = [False, True]  # 是否使用通道权重的列表，用于测试两种情况
        X2_scale = 1.2  # 第二个输入张量的缩放因子为1.2
        X2_zero_point_list = [0, 4]  # 第二个输入张量的零点偏移的列表，包含0和4两种情况

        options = itertools.product(groups_list, use_bias_list, use_channelwise_list, X2_zero_point_list)
        # 使用itertools生成所有参数组合的迭代器
        for groups, use_bias, use_channelwise, X2_zero_point in options:
            # 遍历每种参数组合
            with override_quantized_engine('onednn'):
                # 使用'onednn'作为量化引擎进行上下文管理

                input_channels = input_channels_per_group * groups
                # 计算当前组合下的输入通道数
                output_channels = output_channels_per_group * groups
                # 计算当前组合下的输出通道数
                kernels = (kernel_h, kernel_w)
                # 定义卷积核大小
                strides = (stride_h, stride_w)
                # 定义步长
                pads = (pad_h, pad_w)
                # 定义填充
                dilations = (dilation, dilation)
                # 定义扩展系数

                qconv = torch.ops.quantized.conv2d_add_relu
                # 获取量化的2D卷积加ReLU操作
                qconv_prepack = torch.ops.quantized.conv2d_prepack
                # 获取预打包的量化2D卷积操作
                conv_op = torch.nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernels,
                    strides,
                    pads,
                    dilations,
                    groups,
                )
                # 定义PyTorch的普通2D卷积操作

                X_qdtype = torch.quint8
                # 定义输入张量的数据类型为量化的8位整数（quint8）
                self._test_qconv_impl(
                    qconv, qconv_prepack, conv_op, batch_size,
                    input_channels_per_group, (height, width),
                    output_channels_per_group, groups, kernels, strides, pads, None,
                    dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                    Y_scale, Y_zero_point, use_bias, "add_relu", use_channelwise, False,
                    input_dtype=X_qdtype, output_dtype=X_qdtype, X2_scale=X2_scale, X2_zero_point=X2_zero_point)
                # 调用测试量化卷积实现的方法，传入所有参数和选项

    # TODO: merge this test with test_qconv2d when CUDNN runtime flags becomes available
    """Tests the correctness of quantized 2D convolution cudnn op."""
    # 测试基于CUDNN的量化2D卷积操作的正确性
    @given(batch_size=st.integers(1, 3),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           input_channels_per_group=st.integers(1, 32),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           output_channels_per_group=st.integers(1, 32),
           groups=st.integers(1, 1),  # currently padding only supports groups=1
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           # result for dilation == 2 is not correct
           # dilation=st.integers(1, 2),
           # currently cudnn has only been verified to work for dilation = 1
           # TODO: check backend works for dilation > 1
           dilation=st.integers(1, 1),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.sampled_from([0]),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(0, 0), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.sampled_from([0]),
           use_bias=st.booleans(),
           # TODO: enable channelwise
           use_channelwise=st.sampled_from([False]))
    @skipIfNoFBGEMM  # 跳过测试，如果没有 FBGEMM 支持
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")  # 如果未启用 cudnn，跳过测试
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")  # 如果不是 SM80 或更新版本的 GPU，跳过测试
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")  # 如果在 ROCm 平台上不支持，跳过测试
    def test_qconv2d_cudnn(
            self,
            batch_size,
            input_channels_per_group,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias,
            use_channelwise,
    ):
        # 计算输入通道数，每个组的输入通道数乘以组数
        input_channels = input_channels_per_group * groups
        # 计算输出通道数，每个组的输出通道数乘以组数
        output_channels = output_channels_per_group * groups
        # 定义卷积核大小为 (kernel_h, kernel_w)
        kernels = (kernel_h, kernel_w)
        # 定义步长大小为 (stride_h, stride_w)
        strides = (stride_h, stride_w)
        # 定义填充大小为 (pad_h, pad_w)
        pads = (pad_h, pad_w)
        # 定义膨胀大小为 (dilation, dilation)
        dilations = (dilation, dilation)

        # 获取量化卷积操作函数
        qconv = torch.ops.quantized.conv2d
        # 创建普通的二维卷积层对象，移动到 CUDA 设备上
        conv_op = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernels,
            strides,
            pads,
            dilations,
            groups,
        ).to(torch.device("cuda"))
        # 调用 _test_qconv_impl 方法，测试量化卷积实现
        self._test_qconv_impl(
            qconv, torch.ops.quantized.conv2d_prepack, conv_op, batch_size,
            input_channels_per_group, (height, width),
            output_channels_per_group, groups, kernels, strides, pads, None,
            dilations, X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, "none", use_channelwise, False,
            device=torch.device("cuda"),
            input_dtype=torch.qint8, weight_dtype=torch.qint8, output_dtype=torch.qint8)

    @given(batch_size=st.integers(1, 3),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           input_channels_per_group=st.integers(1, 32),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           output_channels_per_group=st.integers(1, 32),
           groups=st.integers(1, 1),  # currently padding only supports groups=1
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           # result for dilation == 2 is not correct
           # dilation=st.integers(1, 2),
           # currently cudnn has only been verified to work for dilation = 1
           # TODO: check backend works for dilation > 1
           dilation=st.integers(1, 1),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.sampled_from([0]),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(0, 0), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.sampled_from([0]),
           use_bias=st.booleans(),
           # TODO: enable channelwise
           use_channelwise=st.sampled_from([False]))
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    # 定义测试函数，测试量化卷积操作（包含ReLU激活函数）在CUDA上的实现
    def test_qconv2d_relu_cudnn(
            self,
            batch_size,
            input_channels_per_group,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias,
            use_channelwise,
    ):
        # 计算总输入通道数和总输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)  # 卷积核大小
        strides = (stride_h, stride_w)  # 步长
        pads = (pad_h, pad_w)  # 填充
        dilations = (dilation, dilation)  # 膨胀系数

        qconv = torch.ops.quantized.conv2d_relu  # 量化卷积操作
        # 创建标准卷积操作对象，设置在CUDA上运行
        conv_op = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernels,
            strides,
            pads,
            dilations,
            groups,
        ).to(torch.device("cuda"))
        # 调用测试函数，测试量化卷积操作的实现
        self._test_qconv_impl(
            qconv, torch.ops.quantized.conv2d_prepack, conv_op, batch_size,
            input_channels_per_group, (height, width),
            output_channels_per_group, groups, kernels, strides, pads, None,
            dilations, X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, "relu", use_channelwise, False,
            device=torch.device("cuda"),
            input_dtype=torch.qint8, weight_dtype=torch.qint8, output_dtype=torch.qint8)

    # 跳过该测试用例，用于本地基准测试，当需要运行时请取消注释
    @unittest.skip("used for local benchmarking, comment when we want to run it")
    """Tests the correctness of quantized convolution op."""
    @override_qengines
    """Tests the correctness of quantized convolution op."""
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 300),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           o_pad_h=st.integers(0, 2),
           o_pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans())
    @override_qengines
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    """跳过测试：这段代码在没有修改相关代码的情况下无法正常工作，"
       "我们需要在持续集成中移除假设测试"""
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           time=st.integers(2, 5),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 300),
           kernel_t=st.integers(1, 7),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_t=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_t=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           o_pad_t=st.integers(0, 2),
           o_pad_h=st.integers(0, 2),
           o_pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans())
    """使用假设测试来测试量化卷积操作的正确性，指定了各种参数的取值范围"""
    @override_qengines
    """覆盖量化引擎的装饰器"""
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    """跳过测试：这段代码在没有修改相关代码的情况下无法正常工作，"
       "我们需要在持续集成中移除假设测试"""
    @given(
        inputs=hu.tensor_conv(
            spatial_dim=1, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 4),
            output_channels_per_group_range=(1, 4), feature_map_range=(4, 8),
            kernel_range=(1, 4), max_groups=4,
            can_be_transposed=False,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride=st.integers(1, 3),
        pad=st.integers(1, 2),
        o_pad=st.integers(1, 2),
        channelwise=st.booleans())
    """使用假设测试来测试张量卷积操作的正确性，指定了各种参数的取值范围"""
    @override_qengines
    """覆盖量化引擎的装饰器"""
    # 定义一个测试函数，用于测试 quantized 1D 卷积的解包操作
    def test_qconv1d_unpack(self, inputs, stride, pad, o_pad, channelwise):
        # 获取输入数据中的最后一个元素，通常是转置标志位
        transposed = inputs[-1]
        # 获取当前使用的量化引擎
        qengine = torch.backends.quantized.engine
        # 如果当前引擎不在支持的引擎列表中，则直接返回
        if qengine not in supported_qengines:
            return
        # 如果当前引擎是 'qnnpack'，则假设不支持通道级量化
        if qengine == 'qnnpack':
            assume(not channelwise)  # QNNPACK doesn't support channelwise
        else:
            # 否则，假设不支持转置卷积
            assume(not transposed)  # Only QNNPACK supports transposed conv
        # 如果是转置卷积
        if transposed:
            # 使用 quantized 1D 转置卷积的预打包和解包操作
            qconv_prepack = torch.ops.quantized.conv_transpose1d_prepack
            qconv_unpack = torch.ops.quantized.conv_transpose1d_unpack
        else:
            # 否则使用 quantized 1D 卷积的预打包和解包操作
            qconv_prepack = torch.ops.quantized.conv1d_prepack
            qconv_unpack = torch.ops.quantized.conv1d_unpack
        # 调用内部实现函数，测试量化卷积解包的正确性
        self._test_qconv_unpack_impl(
            qconv_prepack, qconv_unpack, inputs, [stride],
            [pad], [o_pad], channelwise)

    # 使用 hypothesis 库生成参数化测试用例
    @given(
        inputs=hu.tensor_conv(
            spatial_dim=2, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 4),
            output_channels_per_group_range=(1, 4), feature_map_range=(4, 8),
            kernel_range=(1, 4), max_groups=4,
            can_be_transposed=True,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride=st.integers(1, 3),
        pad=st.integers(0, 2),
        o_pad=st.integers(0, 2),
        channelwise=st.booleans())
    # 覆盖量化引擎的装饰器
    @override_qengines
    # 定义一个测试函数，用于测试 quantized 2D 卷积的解包操作
    def test_qconv2d_unpack(self, inputs, stride, pad, o_pad, channelwise):
        # 获取输入数据中的最后一个元素，通常是转置标志位
        transposed = inputs[-1]
        # 获取当前使用的量化引擎
        qengine = torch.backends.quantized.engine
        # 如果当前引擎不在支持的引擎列表中，则直接返回
        if qengine not in supported_qengines:
            return
        # 如果当前引擎是 'qnnpack'，则假设不支持通道级量化
        if qengine == 'qnnpack':
            assume(not channelwise)  # QNNPACK doesn't support channelwise
        # 如果是转置卷积
        if transposed:
            # 使用 quantized 2D 转置卷积的预打包和解包操作
            qconv_prepack = torch.ops.quantized.conv_transpose2d_prepack
            qconv_unpack = torch.ops.quantized.conv_transpose2d_unpack
        else:
            # 否则使用 quantized 2D 卷积的预打包和解包操作
            qconv_prepack = torch.ops.quantized.conv2d_prepack
            qconv_unpack = torch.ops.quantized.conv2d_unpack
        # 调用内部实现函数，测试量化卷积解包的正确性
        self._test_qconv_unpack_impl(
            qconv_prepack, qconv_unpack, inputs, [stride, stride],
            [pad, pad], [o_pad, o_pad], channelwise)
    @given(batch_size=st.integers(1, 6),
           input_channels_per_group=st.sampled_from((2, 4, 5, 8, 16, 32)),
           output_channels_per_group=st.sampled_from((2, 4, 5, 8, 16, 32)),
           groups=st.integers(1, 3),
           length=st.integers(4, 16),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans())
    @override_qengines
    def test_qconv1d(
        self,
        batch_size,
        input_channels_per_group,
        output_channels_per_group,
        groups,
        length,
        kernel,
        stride,
        pad,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_channelwise,
    ):
        # 计算每组的输入通道数和输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        # 如果当前的量化后端引擎是 qnnpack，则禁用通道级量化
        if torch.backends.quantized.engine == 'qnnpack':
            use_channelwise = False
        # 创建 1D 卷积层对象
        conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        )
        # 获取预打包的量化卷积函数和量化卷积函数
        qconv_prepack = torch.ops.quantized.conv1d_prepack
        qconv = torch.ops.quantized.conv1d

        # 定义激活量化数据类型列表，目前仅 qnnpack 引擎支持 qint8
        act_qdtypes = [torch.quint8]
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        # 遍历激活量化数据类型
        for X_qdtype in act_qdtypes:
            # 如果激活量化数据类型是 qint8，则将权重零点初始化为零
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]

            # 调用测试量化卷积的实现方法
            self._test_qconv_impl(
                qconv, qconv_prepack, conv1d, batch_size,
                input_channels_per_group, (length, ),
                output_channels_per_group, groups, kernel, [stride], [pad], None,
                [dilation], X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, "none", use_channelwise, False,
                input_dtype=X_qdtype, output_dtype=X_qdtype)
    # 使用 hypothesis 提供的装饰器 @given 来定义测试参数范围
    @given(batch_size=st.integers(1, 6),
           input_channels_per_group=st.sampled_from((2, 4, 5, 8, 16, 32)),
           output_channels_per_group=st.sampled_from((2, 4, 5, 8, 16, 32)),
           groups=st.integers(1, 3),
           length=st.integers(4, 16),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans())
    @override_qengines
    # 定义测试函数 test_qconv1d_relu，用于测试量化卷积和 ReLU 激活函数的正确性
    def test_qconv1d_relu(
        self,
        batch_size,
        input_channels_per_group,
        output_channels_per_group,
        groups,
        length,
        kernel,
        stride,
        pad,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_channelwise,
    ):
        # 计算实际输入通道数和输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        # 如果当前使用的量化引擎为 qnnpack，则禁用 channelwise 参数
        if torch.backends.quantized.engine == 'qnnpack':
            use_channelwise = False
        # 创建 Conv1d 层对象
        conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        )
        # 获取预打包的量化 Conv1d 操作符
        qconv_prepack = torch.ops.quantized.conv1d_prepack
        # 获取包含 ReLU 的量化 Conv1d 操作符
        qconv = torch.ops.quantized.conv1d_relu

        # 定义激活函数的量化数据类型列表，目前只支持 quint8
        act_qdtypes = [torch.quint8]
        # 如果当前量化引擎是 qnnpack 且启用了 xnnpack，则添加支持 qint8 类型
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        # 遍历所有激活函数的量化数据类型
        for X_qdtype in act_qdtypes:
            # 当激活函数的量化数据类型为 qint8 时，将 W_zero_point 初始化为全零列表
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]

            # 调用内部方法 _test_qconv_impl 来执行量化 Conv1d 和 ReLU 的测试
            self._test_qconv_impl(
                qconv, qconv_prepack, conv1d, batch_size,
                input_channels_per_group, (length, ),
                output_channels_per_group, groups, kernel, [stride], [pad], None,
                [dilation], X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, "relu", use_channelwise, False,
                input_dtype=X_qdtype, output_dtype=X_qdtype)

    # TODO: 将此测试与 test_qconv1d 合并，当 CUDNN 运行时标志变得可用时
    """Tests the correctness of quantized 1D convolution cudnn op."""
    @given(batch_size=st.integers(1, 6),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           input_channels_per_group=st.integers(1, 32),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           output_channels_per_group=st.integers(1, 32),
           groups=st.integers(1, 1),  # currently padding only supports groups=1
           length=st.integers(4, 16),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           # currently cudnn has only been verified to work for dilation = 1
           # TODO: check backend works for dilation > 1
           dilation=st.integers(1, 1),
           X_scale=st.floats(1.2, 1.6),
           # currently conv cudnn backend is only implemented for int8 symmetric
           X_zero_point=st.sampled_from([0]),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           # currently conv cudnn backend is only implemented for int8 symmetric
           W_zero_point=st.lists(st.integers(0, 0), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           # currently conv cudnn backend is only implemented for int8 symmetric
           Y_zero_point=st.sampled_from([0]),
           use_bias=st.booleans(),
           # TODO: enable channelwise
           use_channelwise=st.sampled_from([False]))
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skipIf(not SM80OrLater, "requires sm80 or later.")
    @unittest.skipIf(TEST_ROCM, "not supported on rocm.")
    def test_qconv1d_cudnn(
        self,
        batch_size,
        input_channels_per_group,
        output_channels_per_group,
        groups,
        length,
        kernel,
        stride,
        pad,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_channelwise,
    ):
        # 计算实际输入通道数，考虑了分组
        input_channels = input_channels_per_group * groups
        # 计算实际输出通道数，考虑了分组
        output_channels = output_channels_per_group * groups

        # 创建一个在 CUDA 设备上运行的 1 维卷积层
        conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        ).to(torch.device("cuda"))
        
        # 获取预打包的量化卷积运算符
        qconv_prepack = torch.ops.quantized.conv1d_prepack
        # 获取量化卷积运算符
        qconv = torch.ops.quantized.conv1d

        # 调用内部方法，执行量化卷积的测试
        self._test_qconv_impl(
            qconv, qconv_prepack, conv1d, batch_size,
            input_channels_per_group, (length, ),
            output_channels_per_group, groups, kernel, [stride], [pad], None,
            [dilation], X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, "none", use_channelwise, False,
            device=torch.device("cuda"),
            input_dtype=torch.qint8, weight_dtype=torch.qint8, output_dtype=torch.qint8)
    # 使用 `st.integers` 创建一个假设，表示批处理大小在1到6之间
    # 在注释中指出，由于后端已显式添加了填充，cudnn仅支持4的倍数
    # 使用 `st.integers` 创建一个假设，表示每个组内的输入通道数在1到32之间
    # 使用 `st.integers` 创建一个假设，表示每个组内的输出通道数在1到32之间
    # 使用 `st.integers` 创建一个假设，表示组数为1（当前填充仅支持组数为1）
    # 使用 `st.integers` 创建一个假设，表示长度在4到16之间
    # 使用 `st.integers` 创建一个假设，表示卷积核大小在1到7之间
    # 使用 `st.integers` 创建一个假设，表示步长在1到2之间
    # 使用 `st.integers` 创建一个假设，表示填充在0到2之间
    # 使用 `st.integers` 创建一个假设，表示膨胀率为1（目前仅验证膨胀率为1的后端工作）
    # 使用 `st.floats` 创建一个假设，表示X的缩放因子在1.2到1.6之间
    # 当前卷积 cudnn 后端仅实现了int8对称
    # 使用 `st.sampled_from` 创建一个假设，表示X的零点为0
    # 使用 `st.lists` 和 `st.floats` 创建一个假设，表示W的缩放因子列表在0.2到1.6之间，长度为1到2
    # 当前卷积 cudnn 后端仅实现了int8对称
    # 使用 `st.lists` 和 `st.integers` 创建一个假设，表示W的零点列表为0，长度为1到2
    # 使用 `st.floats` 创建一个假设，表示Y的缩放因子在4.2到5.6之间
    # 当前卷积 cudnn 后端仅实现了int8对称
    # 使用 `st.sampled_from` 创建一个假设，表示Y的零点为0
    # 使用 `st.booleans` 创建一个假设，表示是否使用偏置
    # TODO: 启用通道级的量化
    # 使用 `st.sampled_from` 创建一个假设，表示是否启用通道级量化（目前为False）

    # 如果没有 FBGEMM，跳过这个测试
    # 如果未启用 cudnn，跳过这个测试
    # 如果不是 SM80 或更高版本的 GPU，跳过这个测试
    # 如果在 ROCm 平台上，不支持该测试，跳过

    # 测试函数：测试基于 cudnn 的量化卷积一维ReLU操作
    def test_qconv1d_relu_cudnn(
        self,
        batch_size,
        input_channels_per_group,
        output_channels_per_group,
        groups,
        length,
        kernel,
        stride,
        pad,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_channelwise,
    ):
        # 计算总的输入通道数和输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups

        # 创建一个在CUDA设备上运行的一维卷积层
        conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        ).to(torch.device("cuda"))

        # 获取量化卷积预打包和ReLU操作的函数引用
        qconv_prepack = torch.ops.quantized.conv1d_prepack
        qconv = torch.ops.quantized.conv1d_relu

        # 调用测试函数的内部实现，传递相应的参数和设备信息
        self._test_qconv_impl(
            qconv, qconv_prepack, conv1d, batch_size,
            input_channels_per_group, (length, ),
            output_channels_per_group, groups, kernel, [stride], [pad], None,
            [dilation], X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, "relu", use_channelwise, False,
            device=torch.device("cuda"),
            input_dtype=torch.qint8, weight_dtype=torch.qint8, output_dtype=torch.qint8)
    @given(batch_size=st.integers(1, 4),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),
           D=st.integers(4, 8),
           H=st.integers(4, 8),
           W=st.integers(4, 8),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),
           groups=st.integers(1, 3),
           kernel_d=st.integers(1, 4),
           kernel_h=st.integers(1, 4),
           kernel_w=st.integers(1, 4),
           stride_d=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_d=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qconv3d(
        self,
        batch_size,
        input_channels_per_group,
        D,
        H,
        W,
        output_channels_per_group,
        groups,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_channelwise,
        qengine
    ):
        # 检查量化引擎是否在支持的列表中，如果不在则退出测试
        if qengine not in supported_qengines:
            return

        # 计算总的输入通道数和输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups

        # 定义卷积核大小、步幅、填充和扩张参数
        kernels = (kernel_d, kernel_h, kernel_w)
        strides = (stride_d, stride_h, stride_w)
        pads = (pad_d, pad_h, pad_w)
        dilations = (dilation, dilation, dilation)

        # 使用指定的量化引擎运行以下代码块
        with override_quantized_engine(qengine):
            # 获取量化卷积操作符和预打包操作符
            qconv = torch.ops.quantized.conv3d
            qconv_prepack = torch.ops.quantized.conv3d_prepack

            # 创建标准卷积操作对象
            conv_op = torch.nn.Conv3d(
                input_channels,
                output_channels,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )

            # 调用内部方法来测试量化卷积实现
            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (D, H, W), output_channels_per_group,
                groups, kernels, strides, pads, None, dilations, X_scale,
                X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
                use_bias, "none", use_channelwise, use_transpose=False)
    # 使用 hypothesis 库的 @given 装饰器定义了一个参数化测试函数，用于测试量化卷积操作
    @given(
        batch_size=st.integers(1, 4),  # 批量大小在1到4之间的整数
        input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),  # 输入通道数每组从给定列表中随机选择
        D=st.integers(4, 8),  # 深度（D）维度在4到8之间的整数
        H=st.integers(4, 8),  # 高度（H）维度在4到8之间的整数
        W=st.integers(4, 8),  # 宽度（W）维度在4到8之间的整数
        output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),  # 输出通道数每组从给定列表中随机选择
        groups=st.integers(1, 3),  # 组数在1到3之间的整数
        kernel_d=st.integers(1, 4),  # 深度方向（kernel_d）卷积核大小在1到4之间的整数
        kernel_h=st.integers(1, 4),  # 高度方向（kernel_h）卷积核大小在1到4之间的整数
        kernel_w=st.integers(1, 4),  # 宽度方向（kernel_w）卷积核大小在1到4之间的整数
        stride_d=st.integers(1, 2),  # 深度方向（stride_d）步长在1到2之间的整数
        stride_h=st.integers(1, 2),  # 高度方向（stride_h）步长在1到2之间的整数
        stride_w=st.integers(1, 2),  # 宽度方向（stride_w）步长在1到2之间的整数
        pad_d=st.integers(0, 2),  # 深度方向（pad_d）填充在0到2之间的整数
        pad_h=st.integers(0, 2),  # 高度方向（pad_h）填充在0到2之间的整数
        pad_w=st.integers(0, 2),  # 宽度方向（pad_w）填充在0到2之间的整数
        dilation=st.integers(1, 2),  # 膨胀率在1到2之间的整数
        X_scale=st.floats(1.2, 1.6),  # 输入张量的缩放因子在1.2到1.6之间的浮点数
        X_zero_point=st.integers(0, 4),  # 输入张量的零点在0到4之间的整数
        W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),  # 权重张量的缩放因子列表
        W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),  # 权重张量的零点列表
        Y_scale=st.floats(4.2, 5.6),  # 输出张量的缩放因子在4.2到5.6之间的浮点数
        Y_zero_point=st.integers(0, 4),  # 输出张量的零点在0到4之间的整数
        use_bias=st.booleans(),  # 是否使用偏置项的布尔值
        use_channelwise=st.booleans(),  # 是否使用通道级量化的布尔值
        qengine=st.sampled_from(("qnnpack", "fbgemm"))  # 量化引擎从给定列表中随机选择
    )
    # 定义了一个测试量化卷积操作的方法
    def test_qconv3d_relu(
        self,
        batch_size,  # 批量大小
        input_channels_per_group,  # 每组输入通道数
        D, H, W,  # 深度、高度、宽度维度
        output_channels_per_group,  # 每组输出通道数
        groups,  # 组数
        kernel_d, kernel_h, kernel_w,  # 卷积核大小
        stride_d, stride_h, stride_w,  # 步长
        pad_d, pad_h, pad_w,  # 填充
        dilation,  # 膨胀率
        X_scale, X_zero_point,  # 输入张量的缩放因子和零点
        W_scale, W_zero_point,  # 权重张量的缩放因子和零点
        Y_scale, Y_zero_point,  # 输出张量的缩放因子和零点
        use_bias,  # 是否使用偏置项
        use_channelwise,  # 是否使用通道级量化
        qengine  # 量化引擎
    ):
        # 如果量化引擎不在支持的量化引擎列表中，直接返回
        if qengine not in supported_qengines:
            return
    
        # 计算总的输入通道数和输出通道数
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        # 构建卷积核、步长、填充和膨胀率的元组
        kernels = (kernel_d, kernel_h, kernel_w)
        strides = (stride_d, stride_h, stride_w)
        pads = (pad_d, pad_h, pad_w)
        dilations = (dilation, dilation, dilation)
    
        # 使用指定的量化引擎覆盖当前的量化引擎环境
        with override_quantized_engine(qengine):
            # 获取量化卷积操作函数和预打包函数
            qconv = torch.ops.quantized.conv3d_relu
            qconv_prepack = torch.ops.quantized.conv3d_prepack
            # 创建标准的3D卷积操作
            conv_op = torch.nn.Conv3d(
                input_channels,
                output_channels,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            # 调用内部方法测试量化卷积实现的正确性
            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (D, H, W), output_channels_per_group,
                groups, kernels, strides, pads, None, dilations, X_scale,
                X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
                use_bias, "relu", use_channelwise, use_transpose=False)
    
    """Tests the correctness of the quantized::qconv3d_unpack op."""
    @given(
        inputs=hu.tensor_conv(
            spatial_dim=3, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 3),
            output_channels_per_group_range=(1, 3), feature_map_range=(3, 6),
            kernel_range=(1, 3), max_groups=3,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride_d=st.integers(1, 2), stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_d=st.integers(1, 2), pad_h=st.integers(1, 2),
        pad_w=st.integers(1, 2),
        o_pad=st.integers(0, 2),
        channelwise=st.booleans())
    @override_qengines
    def test_qconv3d_unpack(
        self, inputs, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, o_pad,
        channelwise
    ):
        # 检查当前量化引擎是否为 QNNPACK，若是则直接返回，因为 QNNPACK 不支持该操作
        if qengine_is_qnnpack():
            return  # QNNPACK doesn't support this
        
        # 获取是否需要转置的标志
        transposed = inputs[-1]
        
        # 根据是否需要转置，选择相应的量化卷积预打包和解包函数
        if transposed:
            qconv_prepack = torch.ops.quantized.conv_transpose3d_prepack
            qconv_unpack = torch.ops.quantized.conv_transpose3d_unpack
        else:
            qconv_prepack = torch.ops.quantized.conv3d_prepack
            qconv_unpack = torch.ops.quantized.conv3d_unpack
        
        # 调用内部函数来执行量化卷积解包的测试
        self._test_qconv_unpack_impl(
            qconv_prepack, qconv_unpack, inputs,
            (stride_d, stride_h, stride_w), (pad_d, pad_h, pad_w), (o_pad, o_pad, o_pad),
            channelwise)
    def test_conv_reorder_issue_onednn(self):
        """ 确保 onednn 后端中卷积的重新排序问题已修复。
            Onednn 后端在运行具有动态输入形状的卷积时曾遇到重新排序失败的问题。
            通过 https://github.com/pytorch/pytorch/pull/86876 解决。
        """
        if 'onednn' not in supported_qengines:
            return
        # 使用 onednn 引擎覆盖当前量化引擎
        with override_quantized_engine('onednn'):
            bs = 1  # batch size 设置为 1
            ic, oc = 128, 512  # 输入通道数和输出通道数分别设置为 128 和 512
            kh, kw = 1, 1  # 卷积核大小设置为 1x1
            bias = None  # 偏置为空
            strides, paddings, dilates = (1, 1), (0, 0), (1, 1)  # 步长、填充、扩张分别设置
            for groups in [1, 2]:  # 循环遍历不同的分组数
                ih, iw = 28, 28  # 输入特征图的高度和宽度设置为 28x28
                w = torch.randn((oc * groups, ic, kh, kw))  # 随机初始化卷积核权重
                qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)  # 对权重进行量化
                x = torch.randn((bs, ic * groups, ih, iw))  # 随机初始化输入张量
                qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)  # 对输入张量进行量化
                # 对卷积核进行预打包
                w_packed = torch.ops.quantized.conv2d_prepack(
                    qw, bias, strides, paddings, dilates, groups
                )
                # 执行量化卷积操作
                torch.ops.quantized.conv2d(qx, w_packed, output_scale=1.0, output_zero_point=0)
                ih, iw = 5, 4  # 修改输入特征图的高度和宽度为 5x4
                x = torch.randn((bs, ic * groups, ih, iw))  # 重新初始化输入张量
                qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)  # 重新量化输入张量
                # 当输入形状发生变化时，以下操作应该通过
                torch.ops.quantized.conv2d(qx, w_packed, output_scale=1.0, output_zero_point=0)

    @skipIfNoONEDNN
    def test_conv_transpose_reorder_issue_onednn(self):
        # 使用 onednn 引擎覆盖当前量化引擎
        with override_quantized_engine('onednn'):
            bs = 1  # batch size 设置为 1
            ic, oc = 16, 33  # 输入通道数和输出通道数分别设置为 16 和 33
            kh, kw = 3, 3  # 转置卷积核大小设置为 3x3
            ih, iw = 50, 100  # 输入特征图的高度和宽度设置为 50x100
            bias = None  # 偏置为空
            strides, paddings, output_paddings, dilates, groups = [2, 2], [0, 0], [0, 0], [1, 1], 1  # 步长、填充、输出填充、扩张和分组设置
            w = torch.randn((ic, oc, kh, kw))  # 随机初始化转置卷积核权重
            qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)  # 对权重进行量化
            x = torch.randn((bs, ic, ih, iw))  # 随机初始化输入张量
            qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)  # 对输入张量进行量化
            # 对转置卷积核进行预打包
            w_packed = torch.ops.quantized.conv_transpose2d_prepack(
                qw, bias, strides, paddings, output_paddings, dilates, groups
            )
            # 执行量化转置卷积操作
            torch.ops.quantized.conv_transpose2d(qx, w_packed, output_scale=1.0, output_zero_point=0)
            ih, iw = 5, 4  # 修改输入特征图的高度和宽度为 5x4
            x = torch.randn((bs, ic, ih, iw))  # 重新初始化输入张量
            qx = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)  # 重新量化输入张量
            # 当输入形状发生变化时，以下操作应该通过
            torch.ops.quantized.conv_transpose2d(qx, w_packed, output_scale=1.0, output_zero_point=0)
    # 定义一个测试函数，用于测试量化卷积的实现（CPU 版本）
    def _test_qconv_impl_cpu_tensor(
        self,
        qconv,  # 量化卷积函数
        qconv_prepack,  # 量化卷积预打包函数
        conv_op,  # 标准卷积操作对象
        input_channels_per_group=2,  # 每组输入通道数，默认为2
        input_feature_map_shape=(),  # 输入特征图的形状，默认为空元组
        output_channels_per_group=2,  # 每组输出通道数，默认为2
        groups=1,  # 组数，默认为1
        kernels=3,  # 卷积核大小，默认为3
        strides=(),  # 步长，默认为空元组
        pads=(),  # 填充数，默认为空元组
        dilations=(),  # 膨胀率，默认为空元组
        X_scale=1.3,  # 输入张量的缩放因子，默认为1.3
        X_zero_point=2,  # 输入张量的零点值，默认为2
        W_scale=(1.0,),  # 权重的缩放因子，默认为(1.0,)
        W_zero_point=(0,),  # 权重的零点值，默认为(0,)
        Y_scale=3.2,  # 输出张量的缩放因子，默认为3.2
        Y_zero_point=0,  # 输出张量的零点值，默认为0
        use_bias=True,  # 是否使用偏置，默认为True
        post_op=PointwisePostOp(),  # 点操作后处理对象，默认为PointwisePostOp()的实例
        use_channelwise=True,  # 是否使用通道级量化，默认为True
        X2_scale=1.2,  # 第二个输入张量的缩放因子，默认为1.2
        X2_zero_point=0,  # 第二个输入张量的零点值，默认为0
        qconv_output_dtype=None,  # 量化卷积输出的数据类型，默认为None（torch.float32或torch.bfloat16）
        weight_in_channel_last_format=False,  # 权重是否使用通道最后格式，默认为False
        qconv_x2_dtype=None,  # 第二个输入张量的数据类型，默认为None
    ):
        @skipIfNoONEDNN  # 如果没有ONEDNN支持，则跳过该测试函数
        def test_qconv1d_pt2e(self):
            groups_list = [1, 3]  # 定义组数的列表
            input_channels_per_group = 2  # 每组输入通道数
            output_channels_per_group = 2  # 每组输出通道数
            length = 4  # 输入特征图长度
            kernel = 3  # 卷积核大小
            stride = 1  # 步长
            pad = 1  # 填充数
            dilation = 1  # 膨胀率
            W_scale = [1.5]  # 权重的缩放因子列表
            W_zero_point = [0]  # 权重的零点值列表
            use_bias_list = [False, True]  # 是否使用偏置的列表
            use_channelwise_list = [False, True]  # 是否使用通道级量化的列表
            output_dtype_list = [None, torch.float32, torch.bfloat16]  # 输出数据类型的列表
            options = itertools.product(groups_list, use_bias_list, use_channelwise_list, output_dtype_list)
            # 遍历所有可能的组合选项
            for groups, use_bias, use_channelwise, output_dtype in options:
                if output_dtype is not None and not (use_bias and use_channelwise):
                    # 如果输出数据类型不为空且不同时使用偏置和通道级量化，则跳过该组合的测试以减少单元测试时间
                    continue
                conv1d = torch.nn.Conv1d(
                    input_channels_per_group * groups,
                    output_channels_per_group * groups,
                    kernel,
                    stride,
                    pad,
                    dilation,
                    groups,
                )
                qconv = torch.ops.onednn.qconv1d_pointwise  # 获取量化卷积的操作函数
                qconv_prepack = torch.ops.onednn.qconv_prepack  # 获取量化卷积预打包的操作函数
                pointwise_post_op = PointwisePostOp()  # 创建一个点操作后处理对象的实例
                self._test_qconv_impl_cpu_tensor(
                    qconv,
                    qconv_prepack,
                    conv1d,
                    input_channels_per_group=input_channels_per_group,
                    input_feature_map_shape=(length,),  # 输入特征图的形状为(length,)
                    output_channels_per_group=output_channels_per_group,
                    groups=groups,
                    kernels=kernel,
                    strides=[stride],
                    pads=[pad],
                    dilations=[dilation],
                    W_scale=W_scale,
                    W_zero_point=W_zero_point,
                    use_bias=use_bias,
                    post_op=pointwise_post_op,
                    use_channelwise=use_channelwise,
                    qconv_output_dtype=output_dtype,
                )

        @skipIfNoONEDNN  # 如果没有ONEDNN支持，则跳过该测试函数
    # 定义一个测试函数，用于测试 qconv2d_pt2e 方法
    def test_qconv2d_pt2e(self):
        # 定义一些测试参数的列表
        groups_list = [1, 3]  # 卷积组数的列表
        input_channels_per_group = 2  # 每个组的输入通道数
        output_channels_per_group = 2  # 每个组的输出通道数
        input_feature_map_shape = (10, 10)  # 输入特征图的形状
        kernels = (3, 3)  # 卷积核的大小
        strides = (2, 2)  # 步幅
        pads = (1, 1)  # 填充
        dilations = (1, 1)  # 膨胀率
        W_scale = [1.5]  # 权重缩放因子列表
        W_zero_point = [0]  # 权重零点列表
        use_bias_list = [False, True]  # 是否使用偏置的列表
        use_channelwise_list = [False, True]  # 是否使用通道级别量化的列表
        channel_last_weight_format_list = [False, True]  # 是否使用通道最后权重格式的列表
        output_dtype_list = [None, torch.float32, torch.bfloat16]  # 输出数据类型的列表

        # 生成所有参数组合的迭代器
        options = itertools.product(
            groups_list,
            use_bias_list,
            use_channelwise_list,
            channel_last_weight_format_list,
            output_dtype_list,
        )

        # 遍历参数组合进行测试
        for groups, use_bias, use_channelwise, channel_last_weight_format, output_dtype in options:
            # 如果输出数据类型不为 None 或者使用通道最后权重格式，并且不同时使用偏置和通道级别量化
            if (output_dtype is not None or channel_last_weight_format) and not (use_bias and use_channelwise):
                # 跳过某些测试组合以减少单元测试时间
                continue

            # 获取 torch 中的操作符
            qconv = torch.ops.onednn.qconv2d_pointwise
            qconv_prepack = torch.ops.onednn.qconv_prepack

            # 创建 Conv2d 对象
            conv_op = torch.nn.Conv2d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )

            # 创建 PointwisePostOp 对象
            pointwise_post_op = PointwisePostOp()

            # 调用 _test_qconv_impl_cpu_tensor 方法进行测试
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                qconv_output_dtype=output_dtype,
                weight_in_channel_last_format=channel_last_weight_format,
            )

    # 使用装饰器 @skipIfNoONEDNN 标记的函数，在没有 ONEDNN 支持时跳过执行
    @skipIfNoONEDNN
    # 定义测试函数 test_qconv3d_pt2e，用于测试 QConv3D 算法的不同配置
    def test_qconv3d_pt2e(self):
        # 设置每个组的输入通道数
        input_channels_per_group = 2
        # 设置输入特征图的形状为 6x6x6
        input_feature_map_shape = (6, 6, 6)
        # 设置每个组的输出通道数
        output_channels_per_group = 2
        # 定义组数的列表，用于不同的组配置进行测试
        groups_list = [1, 3]
        # 设置卷积核的大小为 3x3x3
        kernels = (3, 3, 3)
        # 设置卷积的步幅为 2x2x2
        strides = (2, 2, 2)
        # 设置卷积的填充为 1x1x1
        pads = (1, 1, 1)
        # 设置卷积的膨胀率为 1x1x1
        dilations = (1, 1, 1)
        # 设置权重的缩放因子和零点偏移
        W_scale = [1.5]
        W_zero_point = [0]
        # 定义是否使用偏置的列表
        use_bias_list = [False, True]
        # 定义是否使用通道级别权重的列表
        use_channelwise_list = [False, True]
        # 定义是否使用通道为最后一维的权重格式的列表
        channel_last_weight_format_list = [False, True]
        # 定义输出数据类型的列表，包括 None、torch.float32 和 torch.bfloat16
        output_dtype_list = [None, torch.float32, torch.bfloat16]
        
        # 生成所有参数配置的组合
        options = itertools.product(
            groups_list,
            use_bias_list,
            use_channelwise_list,
            channel_last_weight_format_list,
            output_dtype_list,
        )
        
        # 遍历所有参数组合进行测试
        for groups, use_bias, use_channelwise, channel_last_weight_format, output_dtype in options:
            # 如果输出数据类型不为 None 或者使用通道为最后一维的权重格式，并且同时不使用偏置或通道级别权重
            if (output_dtype is not None or channel_last_weight_format) and not (use_bias and use_channelwise):
                # 移除某些测试组合以减少单元测试时间
                continue
            
            # 获取 QConv3D 算法的操作符和预打包操作符
            qconv = torch.ops.onednn.qconv3d_pointwise
            qconv_prepack = torch.ops.onednn.qconv_prepack
            
            # 创建 Conv3d 层对象
            conv_op = torch.nn.Conv3d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            
            # 创建 PointwisePostOp 对象
            pointwise_post_op = PointwisePostOp()
            
            # 调用 _test_qconv_impl_cpu_tensor 方法进行 QConv3D 算法的测试
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                qconv_output_dtype=output_dtype,
                weight_in_channel_last_format=channel_last_weight_format,
            )

    # 带有后操作 relu 的 QConv 测试
    @skipIfNoONEDNN
    # 定义测试函数，用于测试 qconv2d_relu_pt2e 方法
    def test_qconv2d_relu_pt2e(self):
        # 设置每组的输入通道数和输出通道数
        input_channels_per_group = 2
        output_channels_per_group = 2
        # 定义组列表，包含不同的组数
        groups_list = [1, 10]
        # 定义输入特征图的形状
        input_feature_map_shape = (10, 10)
        # 定义卷积核的大小
        kernels = (3, 3)
        # 定义步幅
        strides = (2, 2)
        # 定义填充大小
        pads = (1, 1)
        # 定义扩展大小
        dilations = (1, 1)
        # 定义权重缩放因子列表
        W_scale = [1.5]
        # 定义权重零点列表
        W_zero_point = [0]
        # 定义是否使用偏置的列表
        use_bias_list = [False, True]
        # 定义是否使用通道级操作的列表
        use_channelwise_list = [False, True]
        # 定义输出数据类型的列表
        output_dtype_list = [None, torch.float32, torch.bfloat16]
        # 生成所有参数组合的迭代器
        options = itertools.product(groups_list, use_bias_list, use_channelwise_list, output_dtype_list)
        # 遍历所有参数组合
        for groups, use_bias, use_channelwise, output_dtype in options:
            # 调用 ONEDNN 库的 qconv2d_pointwise 方法
            qconv = torch.ops.onednn.qconv2d_pointwise
            # 调用 ONEDNN 库的 qconv_prepack 方法
            qconv_prepack = torch.ops.onednn.qconv_prepack
            # 创建 PyTorch 的 Conv2d 操作符
            conv_op = torch.nn.Conv2d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            # 创建 PointwisePostOp 对象，指定使用 relu 操作
            pointwise_post_op = PointwisePostOp(unary_attr="relu")
            # 调用内部方法 _test_qconv_impl_cpu_tensor 进行测试
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                qconv_output_dtype=output_dtype,
            )

    # 标记测试 qconv 与 post op 使用 hardtanh 的情况，仅在 ONEDNN 可用时执行
    @skipIfNoONEDNN
    # 定义测试方法，测试一种定点卷积操作
    def test_qconv2d_hardtanh_pt2e(self):
        # 设置每组输入通道数和每组输出通道数
        input_channels_per_group = 2
        output_channels_per_group = 2
        # 定义多组测试参数
        groups_list = [1, 10]
        # 设置输入特征图形状
        input_feature_map_shape = (10, 10)
        # 设置卷积核大小
        kernels = (3, 3)
        # 设置步长
        strides = (2, 2)
        # 设置填充
        pads = (1, 1)
        # 设置膨胀率
        dilations = (1, 1)
        # 设置权重缩放因子
        W_scale = [1.5]
        # 设置权重零点
        W_zero_point = [0]
        # 设置是否使用偏置的测试列表
        use_bias_list = [False, True]
        # 设置是否使用通道级别操作的测试列表
        use_channelwise_list = [False, True]
        # 设置输出数据类型的测试列表
        output_dtype_list = [None, torch.float32, torch.bfloat16]
        # 生成所有可能的组合
        options = itertools.product(groups_list, use_bias_list, use_channelwise_list, output_dtype_list)
        # 遍历所有参数组合进行测试
        for groups, use_bias, use_channelwise, output_dtype in options:
            # 获取定点卷积函数和预打包函数
            qconv = torch.ops.onednn.qconv2d_pointwise
            qconv_prepack = torch.ops.onednn.qconv_prepack
            # 创建标准的二维卷积操作对象
            conv_op = torch.nn.Conv2d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            # 定义点操作的后处理对象，采用hardtanh函数
            pointwise_post_op = PointwisePostOp(unary_attr="hardtanh", scalars=[0.0, 6.0])
            # 调用内部方法执行定点卷积的CPU端测试
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                qconv_output_dtype=output_dtype,
            )

    # 跳过没有ONEDNN支持的情况下进行测试
    @skipIfNoONEDNN
    # 测试带有sum后处理操作的定点卷积
    @skipIfNoONEDNN
    # 定义测试函数，用于测试qconv2d_sum_pt2e方法
    def test_qconv2d_sum_pt2e(self):
        # 定义各种参数列表
        groups_list = [1, 3]  # 卷积组数列表
        input_channels_per_group = 2  # 每组输入通道数
        output_channels_per_group = 2  # 每组输出通道数
        input_feature_map_shape = (10, 10)  # 输入特征图形状
        kernels = (3, 3)  # 卷积核大小
        strides = (2, 2)  # 步长
        pads = (1, 1)  # 填充
        dilations = (1, 1)  # 膨胀率
        W_scale = [1.5]  # 权重比例尺
        W_zero_point = [-3]  # 权重零点
        use_bias_list = [False, True]  # 是否使用偏置列表
        use_channelwise_list = [False, True]  # 是否使用通道级卷积列表
        output_dtype_list = [None, torch.float32, torch.bfloat16]  # 输出数据类型列表
        X2_zero_point_list = [0, 1]  # 第二个输入零点列表
        
        # 生成所有可能的参数组合
        options = itertools.product(
            groups_list, use_bias_list, use_channelwise_list, X2_zero_point_list, output_dtype_list
        )
        
        # 遍历参数组合进行测试
        for groups, use_bias, use_channelwise, X2_zero_point, output_dtype in options:
            # 获取OneDNN库中的二进制点卷积函数
            qconv = torch.ops.onednn.qconv2d_pointwise.binary
            # 获取OneDNN库中的点卷积预打包函数
            qconv_prepack = torch.ops.onednn.qconv_prepack
            # 创建PyTorch的2D卷积层对象
            conv_op = torch.nn.Conv2d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            # 创建二进制点卷积后处理操作对象
            pointwise_post_op = PointwisePostOp(binary_attr="sum")
            
            # 调用内部函数测试CPU tensor的二进制点卷积实现
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                X2_zero_point=X2_zero_point,
                qconv_output_dtype=output_dtype,
                qconv_x2_dtype=output_dtype,
            )

    # 用于测试带有后处理操作sum和relu的qconv函数
    @skipIfNoONEDNN
    # 定义一个测试方法，用于测试带有不同参数的二维量化卷积操作
    def test_qconv2d_sum_relu_pt2e(self):
        # 定义不同的组数列表
        groups_list = [1, 3]
        # 每组内的输入通道数
        input_channels_per_group = 2
        # 每组内的输出通道数
        output_channels_per_group = 2
        # 输入特征图的形状
        input_feature_map_shape = (10, 10)
        # 卷积核的大小
        kernels = (3, 3)
        # 步长
        strides = (2, 2)
        # 填充
        pads = (1, 1)
        # 膨胀
        dilations = (1, 1)
        # 权重的缩放因子
        W_scale = [1.5]
        # 权重的零点
        W_zero_point = [-3]
        # 是否使用偏置的列表
        use_bias_list = [False, True]
        # 是否使用通道级量化的列表
        use_channelwise_list = [False, True]
        # 第二个输入的零点值列表
        X2_zero_point_list = [0, 1]
        # 生成所有参数组合的迭代器
        options = itertools.product(
            groups_list, use_bias_list, use_channelwise_list, X2_zero_point_list
        )
        # 遍历每一种参数组合
        for groups, use_bias, use_channelwise, X2_zero_point in options:
            # 获取点点卷积操作的引用
            qconv = torch.ops.onednn.qconv2d_pointwise.binary
            # 获取预打包卷积操作的引用
            qconv_prepack = torch.ops.onednn.qconv_prepack
            # 创建标准的二维卷积层对象
            conv_op = torch.nn.Conv2d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            # 创建点点卷积后处理操作对象，设置为二进制求和和ReLU
            pointwise_post_op = PointwisePostOp(binary_attr="sum", unary_attr="relu")
            # 调用测试卷积实现的方法，针对CPU上的张量
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                X2_zero_point=X2_zero_point,
            )

    # 带有后处理操作为求和的qconv的测试
    @skipIfNoONEDNN
    # 定义测试函数 test_qconv2d_sum_relu_float_output_pt2e，用于测试量化卷积操作的各种配置
    def test_qconv2d_sum_relu_float_output_pt2e(self):
        # 设置卷积操作的参数
        groups = 1  # 组数设为1
        input_channels_per_group = 2  # 每组输入通道数为2
        output_channels_per_group = 2  # 每组输出通道数为2
        input_feature_map_shape = (10, 10)  # 输入特征图形状为10x10
        kernels = (3, 3)  # 卷积核大小为3x3
        strides = (2, 2)  # 步长为2x2
        pads = (1, 1)  # 填充为1x1
        dilations = (1, 1)  # 膨胀率为1x1
        W_scale = [1.5]  # 权重缩放因子列表，这里只包含一个元素1.5
        W_zero_point = [-3]  # 权重零点列表，这里只包含一个元素-3
        use_bias_list = [False, True]  # 使用偏置的选项列表
        use_channelwise = True  # 使用通道级量化
        output_dtype_list = [torch.float32, torch.bfloat16]  # 输出数据类型列表，包括浮点数和bfloat16
        X2_zero_point = 0  # X2的零点
        use_relu_list = [True, False]  # 是否使用ReLU激活的选项列表
        options = itertools.product(
            use_bias_list, output_dtype_list, use_relu_list
        )  # 组合所有选项，生成配置的组合
        # 遍历所有配置组合进行测试
        for use_bias, output_dtype, use_relu in options:
            qconv_x2_dtype = output_dtype  # 设置量化卷积X2的数据类型
            # 获取量化卷积函数和预打包函数
            qconv = torch.ops.onednn.qconv2d_pointwise.binary
            qconv_prepack = torch.ops.onednn.qconv_prepack
            # 创建标准卷积操作对象
            conv_op = torch.nn.Conv2d(
                input_channels_per_group * groups,
                output_channels_per_group * groups,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            # 设置点积后操作对象，根据是否使用ReLU进行配置
            pointwise_post_op = (
                PointwisePostOp(binary_attr="sum", unary_attr="relu")
                if use_relu
                else PointwisePostOp(binary_attr="sum")
            )
            # 调用内部方法 _test_qconv_impl_cpu_tensor 进行量化卷积的测试
            self._test_qconv_impl_cpu_tensor(
                qconv,
                qconv_prepack,
                conv_op,
                input_channels_per_group=input_channels_per_group,
                input_feature_map_shape=input_feature_map_shape,
                output_channels_per_group=output_channels_per_group,
                groups=groups,
                kernels=kernels,
                strides=strides,
                pads=pads,
                dilations=dilations,
                W_scale=W_scale,
                W_zero_point=W_zero_point,
                use_bias=use_bias,
                post_op=pointwise_post_op,
                use_channelwise=use_channelwise,
                X2_zero_point=X2_zero_point,
                qconv_output_dtype=output_dtype,
                qconv_x2_dtype=qconv_x2_dtype,
            )
class TestPadding(TestCase):
    @given(batch_size=st.integers(1, 64),
           channels=st.integers(1, 64),
           width=st.integers(16, 128),
           qtype=st.sampled_from(hu._ALL_QINT_TYPES))
    def test_reflection_pad1d(self, batch_size, channels, width, qtype):
        # 计算填充量，将输入的宽度除以4
        padding = width // 4

        # 创建一个浮点数张量 x，其元素从 0 到 batch_size * channels * width - 1
        x = torch.arange(batch_size * channels * width).to(torch.float)
        # 调整张量 x 的形状为 (batch_size, channels, width)
        x = x.resize(batch_size, channels, width)
        # 对每个张量进行量化，计算动态量化参数
        scale, zp = _calculate_dynamic_qparams(x, qtype)
        qx = torch.quantize_per_tensor(x, scale, zp, qtype)

        # 创建 ReflectionPad1d 操作对象，用 padding 变量初始化
        padding_op = torch.nn.ReflectionPad1d(padding)

        # 应用 ReflectionPad1d 操作到张量 x，得到 y_ref
        y_ref = padding_op(x)
        # 对 y_ref 进行量化
        qy_ref = torch.quantize_per_tensor(y_ref, scale, zp, qtype)
        # 用 ReflectionPad1d 操作对量化输入 qx 进行操作，得到 qy_hat
        qy_hat = padding_op(qx)
        # 断言量化输出 qy_ref 和 qy_hat 相等
        self.assertEqual(qy_ref, qy_hat)

        # 使用 C++ 后端的 reflection_pad1d 函数进行 Out variant 操作
        qy_hat = torch._C._nn.reflection_pad1d(qx, padding, out=qy_hat)
        # 再次断言量化输出 qy_ref 和 qy_hat 相等
        self.assertEqual(qy_ref, qy_hat)

    @given(batch_size=st.integers(1, 64),
           channels=st.integers(1, 64),
           height=st.integers(16, 128),
           width=st.integers(16, 128),
           qtype=st.sampled_from(hu._ALL_QINT_TYPES))
    def test_reflection_pad2d(self, batch_size, channels, height, width, qtype):
        # 计算填充量，分别为宽度和高度的四分之一
        padding = (width // 4, width // 4, height // 4, height // 4)

        # 创建一个浮点数张量 x，其元素从 0 到 batch_size * channels * height * width - 1
        x = torch.arange(batch_size * channels * height * width).to(torch.float)
        # 调整张量 x 的形状为 (batch_size, channels, height, width)
        x = x.resize(batch_size, channels, height, width)
        # 对每个张量进行量化，计算动态量化参数
        scale, zp = _calculate_dynamic_qparams(x, qtype)
        qx = torch.quantize_per_tensor(x, scale, zp, qtype)

        # 创建 ReflectionPad2d 操作对象，用 padding 变量初始化
        padding_op = torch.nn.ReflectionPad2d(padding)

        # 应用 ReflectionPad2d 操作到张量 x，得到 y_ref
        y_ref = padding_op(x)
        # 对 y_ref 进行量化
        qy_ref = torch.quantize_per_tensor(y_ref, scale, zp, qtype)
        # 用 ReflectionPad2d 操作对量化输入 qx 进行操作，得到 qy_hat
        qy_hat = padding_op(qx)
        # 断言量化输出 qy_ref 和 qy_hat 相等
        self.assertEqual(qy_ref, qy_hat)

        # 使用 C++ 后端的 reflection_pad2d 函数进行 Out variant 操作
        qy_hat = torch._C._nn.reflection_pad2d(qx, padding, out=qy_hat)
        # 再次断言量化输出 qy_ref 和 qy_hat 相等
        self.assertEqual(qy_ref, qy_hat)

    @given(batch_size=st.integers(1, 64),
           channels=st.integers(1, 64),
           hwd=st.integers(1, 16),  # For 3D, max input size would be 16x16x16
           d=st.sampled_from([1, 2, 3]),
           value=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
           qtype=st.sampled_from(hu._ALL_QINT_TYPES))
    # 定义一个测试方法，用于测试 ConstantPadNd 的功能
    def test_constant_padNd(self, batch_size, channels, d, hwd, value, qtype):
        # 根据输入的 hwd 计算 padding 的大小
        padding = hwd // 4

        # 构建张量的形状，初始为 [batch_size, channels, hwd]
        shape = [batch_size, channels, hwd]
        # 初始化操作为 ConstantPad1d
        op = torch.nn.ConstantPad1d
        # 根据维度 d 的大小选择不同的操作和张量形状
        if d >= 2:
            shape.append(hwd)
            op = torch.nn.ConstantPad2d
        if d == 3:
            shape.append(hwd)
            op = torch.nn.ConstantPad3d
        # 计算张量中元素的总数
        numel = np.prod(shape)

        # 创建一个从 0 到 numel-1 的张量，并转换为 float 类型
        x = torch.arange(numel).to(torch.float)
        # 调整张量的形状为 shape
        x = x.resize(*shape)
        # 对每个张量进行量化，得到量化的缩放因子和零点偏移量
        scale, zp = _calculate_dynamic_qparams(x, qtype)
        qx = torch.quantize_per_tensor(x, scale, zp, qtype)

        # 根据 padding 和 value 创建 ConstantPadNd 操作对象
        padding_op = op(padding, value)

        # 使用 padding_op 对 x 进行填充操作，得到参考的结果 y_ref
        y_ref = padding_op(x)
        # 对填充后的结果 y_ref 进行量化
        qy_ref = torch.quantize_per_tensor(y_ref, scale, zp, qtype)
        # 使用 padding_op 对量化的输入 qx 进行填充操作，得到测试结果 qy_hat
        qy_hat = padding_op(qx)

        # 使用断言检查填充后的量化结果 qy_ref 是否等于 qy_hat
        self.assertEqual(qy_ref, qy_hat)
# 如果 qnnpack 引擎在支持的引擎列表中，则跳过该测试；否则给出相应的消息
@unittest.skipUnless('qnnpack' in supported_qengines,
                     "This Pytorch Build has not been built with or does not support QNNPACK")
class TestQNNPackOps(TestCase):
    """Tests the correctness of the quantized::qnnpack_relu op."""

    # 使用 hypothesis 的装饰器定义测试函数，其中 X 是一个张量数据
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=torch.quint8,
                                          zero_point_min=0,
                                          zero_point_max=0)))
    def test_qnnpack_relu(self, X):
        # 通过 override_quantized_engine 切换至 qnnpack 引擎
        with override_quantized_engine('qnnpack'):
            # 解构 X，获取数据、缩放因子、零点和张量类型信息
            X, (scale, zero_point, torch_type) = X
            # 定义 relu 函数
            relu = torch.nn.functional.relu
            # 将 numpy 数组 X 转换为 PyTorch 张量
            X = torch.from_numpy(X)
            # 克隆张量 X 为 Y
            Y = X.clone()

            # 对 X 进行量化，使用给定的缩放因子、零点和数据类型
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch_type)
            # 计算 relu(qX) 的量化结果
            qY_hat = relu(qX)

            # 将 Y 中小于零的元素置为零
            Y[Y < 0] = 0
            # 对 Y 进行量化，使用相同的缩放因子、零点和数据类型
            qY = torch.quantize_per_tensor(Y, scale=scale, zero_point=zero_point, dtype=torch_type)
            # 断言量化后的结果 qY 与预期的量化 relu(qX) 结果 qY_hat 相等
            self.assertEqual(qY, qY_hat)

    """Tests the correctness of the quantized::qnnpack_tanh op."""
    
    # 如果未安装 FBGEMM，则跳过该测试
    @skipIfNoFBGEMM
    def test_qnnpack_tanh(self):
        # 注意：在 QNNPACK 中，输出的缩放因子和零点必须为 2.0/256 和 128，因为它使用一个包含 256 个条目的查找表（LUT）

        # 定义不同形状和内存格式的测试用例
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, memory_formats)
        for shape, memory_format in test_cases:
            # 生成随机张量 X，设置缩放因子为 1.0，零点为 0，数据类型为 torch.quint8
            X, scale, zero_point, torch_type = torch.randn(*shape), 1.0, 0, torch.quint8
            # 如果内存格式为 channels_last 且形状长度不为 4，则跳过当前测试
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            # 将 X 转换为指定内存格式
            X = X.to(memory_format=memory_format)
            # 对 X 进行量化，使用给定的缩放因子、零点和数据类型
            qX = torch.quantize_per_tensor(X, scale=scale,
                                           zero_point=zero_point,
                                           dtype=torch_type)

            # 计算浮点数参考值 Y，对 qX 进行去量化并计算 tanh 函数
            Y = torch.tanh(qX.dequantize())
            # 对 Y 进行量化，使用缩放因子 1.0/128，零点 128，数据类型为 torch.quint8
            qY = torch.quantize_per_tensor(Y, scale=1.0 / 128, zero_point=128,
                                           dtype=torch.quint8)
            # 使用 override_quantized_engine 切换至 fbgemm 引擎
            with override_quantized_engine('fbgemm'):
                qYserver = torch.tanh(qX)
            # 使用 override_quantized_engine 切换至 qnnpack 引擎
            with override_quantized_engine('qnnpack'):
                qY_hat = torch.tanh(qX)
                # 断言 qY_hat 与 qY 相等，若不等则输出相应的错误信息，指明内存格式
                self.assertEqual(
                    qY, qY_hat,
                    msg=f"QNNPACK TanH failed (FP ref), memory_format {memory_format}")
                # 断言 qYserver 与 qY_hat 相等，若不等则输出相应的错误信息，指明内存格式
                self.assertEqual(
                    qYserver, qY_hat,
                    msg=f"QNNPACK TanH failed (FBGEMM ref), memory_format {memory_format}")

    """Tests the correctness of the quantized::qnnpack_sigmoid op."""
    @skipIfNoFBGEMM
    def test_qnnpack_sigmoid(self):
        # 在 QNNPACK 中，输出的 scale 和 zero_point 必须分别为 1.0/256 和 0，
        # 因为它使用了一个具有 256 个 bin 的查找表（LUT）。

        # 定义不同维度的输入形状和内存布局格式
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        memory_formats = (torch.channels_last, torch.contiguous_format)
        
        # 生成所有可能的测试用例
        test_cases = itertools.product(shapes, memory_formats)
        
        # 迭代每个测试用例
        for shape, memory_format in test_cases:
            # 生成随机输入张量 X，并设置 scale、zero_point 和数据类型
            X, scale, zero_point, torch_type = torch.randn(*shape), 1.0, 0, torch.quint8
            
            # 如果内存布局格式为 channels_last 且形状不是四维，则跳过当前测试用例
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            
            # 转换输入张量 X 到指定的内存布局格式
            X = X.to(memory_format=memory_format)
            
            # 对输入张量 X 进行量化，使用给定的 scale、zero_point 和数据类型
            qX = torch.quantize_per_tensor(X, scale=scale,
                                           zero_point=zero_point,
                                           dtype=torch_type)

            # 计算浮点参考输出
            Y = torch.sigmoid(qX.dequantize())
            # 对浮点参考输出进行量化，使用固定的 scale 和 zero_point
            qY = torch.quantize_per_tensor(Y, scale=1.0 / 256, zero_point=0,
                                           dtype=torch.quint8)
            
            # 使用 'fbgemm' 引擎计算 qX 的 sigmoid
            with override_quantized_engine('fbgemm'):
                qYserver = torch.sigmoid(qX)
            
            # 使用 'qnnpack' 引擎计算 qX 的 sigmoid
            with override_quantized_engine('qnnpack'):
                qY_hat = torch.sigmoid(qX)
                
                # 断言 qY 和 qY_hat 的相等性，如果不等则输出指定的错误信息
                self.assertEqual(
                    qY, qY_hat,
                    msg=f"QNNPACK Sigmoid failed (FP ref), memory_format {memory_format}")
                
                # 断言 qYserver 和 qY_hat 的相等性，如果不等则输出指定的错误信息
                self.assertEqual(
                    qYserver, qY_hat,
                    msg=f"QNNPACK Sigmoid failed (FBGEMM ref), memory_format {memory_format}")

    @skipIfNoFBGEMM
    def test_qnnpack_sigmoid_sweep(self):
        # 定义输入参数
        f_min = -4.0
        f_max = 4.0
        scale = (f_max - f_min) / 256.0
        zero_point = 128
        dtype = torch.quint8

        step = scale / 2.0
        # 生成一系列输入值 x
        x = np.arange(f_min, f_max + step, step)
        # 将输入值 x 转换为 PyTorch 的张量类型 float32
        X = torch.from_numpy(x).to(torch.float32)
        
        # 对输入张量 X 进行量化，使用给定的 scale、zero_point 和数据类型
        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=dtype)

        # 对量化后的张量进行反量化，计算浮点参考输出 Y
        dqX = qX.dequantize()
        Y = torch.sigmoid(dqX)
        
        # 对浮点参考输出 Y 进行量化，使用固定的 scale 和 zero_point
        qY = torch.quantize_per_tensor(Y, scale=1.0 / 256, zero_point=0,
                                       dtype=torch.quint8)
        
        # 使用 'fbgemm' 引擎计算 qX 的 sigmoid
        with override_quantized_engine('fbgemm'):
            qYserver = torch.sigmoid(qX)
        
        # 使用 'qnnpack' 引擎计算 qX 的 sigmoid
        with override_quantized_engine('qnnpack'):
            qY_hat = torch.sigmoid(qX)
            
            # 断言 qY 和 qY_hat 的相等性，如果不等则输出指定的错误信息
            self.assertEqual(qY, qY_hat,
                             msg="QNNPACK Sigmoid failed (FP ref)!")
            
            # 断言 qYserver 和 qY_hat 的相等性，如果不等则输出指定的错误信息
            self.assertEqual(qYserver, qY_hat,
                             msg="QNNPACK Sigmoid failed (FBGEMM ref)!")
    # 使用 hypothesis 的 @given 装饰器定义测试函数 test_qnnpack_add，提供输入参数 A, zero_point, scale_A, scale_B, scale_C
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=[torch.quint8, torch.qint8])),
           zero_point=st.sampled_from([0, 2, 5, 15, 127]),
           scale_A=st.sampled_from([0.001, 0.057, 0.889, 12.3]),
           scale_B=st.sampled_from([0.008, 0.0821, 0.67, 7]),
           scale_C=st.sampled_from([0.003, 0.07821, 0.457, 7.34]),)
    # 定义测试函数 test_qnnpack_add，用于测试 QNNPACK 加法操作
    def test_qnnpack_add(self, A, zero_point, scale_A, scale_B, scale_C):
        # 使用 qnnpack 引擎进行量化引擎的覆盖
        with override_quantized_engine('qnnpack'):
            # 将 A 赋值给 A_temp
            A_temp = A
            # 对于 channels_last 在 [True, False] 中循环
            for channels_last in [True, False]:
                # 如果 channels_last 为 True，并且 A_temp 的第一个元素的维度不为 4，则跳过当前循环
                if channels_last and len(A_temp[0].shape) != 4:
                    continue
                # 解包 A_temp，获取 A 和相关参数（scale_a, zero_point_A, torch_type）
                A, (scale_a, zero_point_A, torch_type) = A_temp
                # 解包 A_temp，获取 B 和相关参数（scale_b, zero_point_B, torch_type）
                B, (scale_b, zero_point_B, torch_type) = A_temp
                # 将 A 和 B 转换为 PyTorch 张量
                A = torch.from_numpy(A)
                B = torch.from_numpy(B)

                # 如果 torch_type 为 torch.qint8 并且未启用 xnnpack 后端，则继续下一轮循环
                if torch_type == torch.qint8 and not torch.backends.xnnpack.enabled:
                    continue

                # 如果 channels_last 为 True，则将 A 和 B 转换为 channels_last 内存格式
                if channels_last:
                    A = A.to(memory_format=torch.channels_last)
                    B = B.to(memory_format=torch.channels_last)

                # 假设以下条件成立，否则跳过当前循环
                assume(scale_A // scale_C >= 2**-14)
                assume(scale_A // scale_C < 2**8)
                assume(scale_B // scale_C >= 2**-14)
                assume(scale_B // scale_C < 2**8)

                # 初始化 zero_point_C 和 np_dtype
                zero_point_C = 127
                np_dtype = np.uint8

                # 如果 torch_type 为 torch.qint8，则更新 zero_point_C 和 np_dtype
                if torch_type == torch.qint8:
                    zero_point_C = 0
                    np_dtype = np.int8

                # 对 A 和 B 进行量化操作，得到 qA 和 qB
                qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                               dtype=torch_type)
                qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                               dtype=torch_type)

                # 执行加法操作得到 C 的真值
                C = (qA.dequantize() + qB.dequantize()).numpy()

                # 对 C 进行量化，得到 qC
                qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype)

                # 使用 qnnpack 执行量化加法操作得到 qC_qnnp
                qC_qnnp = torch.ops.quantized.add(qA, qB, scale_C, zero_point_C)

                # 断言 qC 和 qC_qnnp 的整数表示相等，否则输出错误信息
                np.testing.assert_equal(qC, qC_qnnp.int_repr(),
                                        "Quantized addition failed.")

                # 对 C 应用 ReLU 操作得到 Crelu
                Crelu = C.copy()
                Crelu[C < 0] = 0

                # 对 Crelu 进行量化操作得到 qCrelu
                qCrelu = torch.quantize_per_tensor(torch.from_numpy(Crelu), scale_C,
                                                   zero_point_C, dtype=torch_type)

                # 使用 qnnpack 执行带 ReLU 的量化加法操作得到 qCrelu_hat
                qCrelu_hat = torch.ops.quantized.add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)

                # 断言 qCrelu 和 qCrelu_hat 的整数表示相等，否则输出错误信息
                np.testing.assert_equal(qCrelu.int_repr().numpy(), qCrelu_hat.int_repr(),
                                        "Quantized addition with ReLU failed.")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=[torch.quint8, torch.qint8])),
           zero_point=st.sampled_from([0, 2, 5, 15, 127]),
           scale_A=st.sampled_from([0.3, 0.57, 0.889]),
           scale_B=st.sampled_from([0.8, 0.821, 0.67]),
           scale_C=st.sampled_from([0.3, 0.7821, 0.457]),)
    # 定义测试函数，用于测试 QNNPACK 的乘法运算
    def test_qnnpack_mul(self, A, zero_point, scale_A, scale_B, scale_C):
        # 使用 qnnpack 引擎执行以下代码块
        with override_quantized_engine('qnnpack'):
            # 复制 A 作为 A_temp
            A_temp = A
            # 遍历 channels_last 的两种情况
            for channels_last in [True, False]:
                # 如果 channels_last 为 True 且 A_temp 的第一个元素维度不是 4，则跳过本次循环
                if channels_last and len(A_temp[0].shape) != 4:
                    continue
                # 解包 A_temp，获取 A 的数据和量化参数
                A, (scale_a, zero_point_A, torch_type) = A_temp
                # 解包 A_temp，获取 B 的数据和量化参数
                B, (scale_b, zero_point_B, torch_type) = A_temp
                # 将 A 和 B 转换为 Torch 的 Tensor 对象
                A = torch.from_numpy(A)
                B = torch.from_numpy(B)

                # 如果 Torch 类型是 torch.qint8 并且未启用 torch.backends.xnnpack，则跳过本次循环
                if torch_type == torch.qint8 and not torch.backends.xnnpack.enabled:
                    continue

                # 如果 channels_last 为 True，则将 A 和 B 转换为通道优先内存格式
                if channels_last:
                    A = A.to(memory_format=torch.channels_last)
                    B = B.to(memory_format=torch.channels_last)
                # 假设 scale_A / scale_C 大于等于 2^-14
                assume(scale_A // scale_C >= 2**-14)
                # 假设 scale_A / scale_C 小于 2^8
                assume(scale_A // scale_C < 2**8)
                # 假设 scale_B / scale_C 大于等于 2^-14
                assume(scale_B // scale_C >= 2**-14)
                # 假设 scale_B / scale_C 小于 2^8
                assume(scale_B // scale_C < 2**8)

                # 设置 zero_point_C 和 np_dtype 的初始值
                zero_point_C = 127
                np_dtype = np.uint8

                # 如果 Torch 类型是 torch.qint8，则更新 zero_point_C 和 np_dtype 的值
                if torch_type == torch.qint8:
                    zero_point_C = 0
                    np_dtype = np.int8

                # 对 A 和 B 进行量化操作，得到 qA 和 qB
                qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                               dtype=torch_type)
                qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                               dtype=torch_type)

                # 计算 ground truth 结果 C
                C = (qA.dequantize() * qB.dequantize()).numpy()

                # 使用 _quantize 函数对 C 进行量化，得到 qC
                qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype)
                # 使用 Torch 自带的量化乘法运算得到 qC_qnnp
                qC_qnnp = torch.ops.quantized.mul(qA, qB, scale_C, zero_point_C)

                # 断言 qC 和 qC_qnnp 的整数表示是否相等，若不相等则输出错误信息
                np.testing.assert_equal(qC, qC_qnnp.int_repr(),
                                        "Quantized addition failed.")

                # 计算经过 ReLU 处理后的 Crelu
                Crelu = C.copy()
                Crelu[C < 0] = 0
                # 将 Crelu 量化为 qCrelu
                qCrelu = torch.quantize_per_tensor(torch.from_numpy(Crelu), scale_C,
                                                   zero_point_C, dtype=torch_type)
                # 使用 Torch 自带的量化乘法并加 ReLU 处理得到 qCrelu_hat
                qCrelu_hat = torch.ops.quantized.mul_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
                # 断言 qCrelu 和 qCrelu_hat 的整数表示是否相等，若不相等则输出错误信息
                np.testing.assert_equal(qCrelu.int_repr().numpy(), qCrelu_hat.int_repr(),
                                        "Quantized addition with ReLU failed.")

    """Tests that quantized add works with broadcasting """
    """测试 qnnpack_add_broadcast 的正确性。

    定义了一个内部函数 _run_test(A, B)，用于执行测试。
    对输入的张量 A 和 B 进行量化，使用指定的量化参数和数据类型。
    设置输出的量化参数和零点偏移。

    在下面的两个部分分别进行了 ground truth 和 quantized 的计算：
    - ground truth：通过 dequantize() 方法计算张量 C 的值，并进行量化得到 qC。
    - quantized：调用 torch.ops.quantized.add() 进行张量 qA 和 qB 的加法运算，得到 qC_hat_1 和 qC_hat_2。

    使用 assertTrue 进行断言，验证 quantized 计算的结果与 ground truth 的计算结果是否全部接近。

    在 override_quantized_engine("qnnpack") 的上下文中，针对不同的数据类型（torch.qint8 和 torch.quint8）进行测试。
    如果数据类型是 torch.qint8 且不支持 xnnpack，则跳过此次测试。

    遍历 channels_last 变量的值 [True, False]，分别测试 4 维和 5 维的张量：
    - 对于 channels_last 为 True 的情况，将张量 A 和 B 转换为 channels_last 内存格式。
    - 在每个测试中调用 _run_test(A, B) 函数。

    """
    def test_qnnpack_add_broadcast(self):
        def _run_test(A, B):
            qA = torch.quantize_per_tensor(A, 0.02, 0, dtype)
            qB = torch.quantize_per_tensor(B, 0.04, 2, dtype)

            output_scale = 0.01
            output_zp = 1

            # ground truth
            C = qA.dequantize() + qB.dequantize()
            qC = torch.quantize_per_tensor(C, output_scale, output_zp, dtype)

            # quantized
            qC_hat_1 = torch.ops.quantized.add(qA, qB, output_scale, output_zp)
            qC_hat_2 = torch.ops.quantized.add(qB, qA, output_scale, output_zp)

            self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_1.dequantize()))
            self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_2.dequantize()))

        with override_quantized_engine("qnnpack"):
            for dtype in (torch.qint8, torch.quint8):
                if dtype == torch.qint8 and not torch.backends.xnnpack.enabled:
                    continue

                for channels_last in [True, False]:
                    # 4d
                    A = torch.randn(1, 3, 4, 4)
                    B = torch.randn(1, 1, 1, 1)
                    if channels_last:
                        A = A.to(memory_format=torch.channels_last)
                        B = B.to(memory_format=torch.channels_last)
                    _run_test(A, B)

                    # 5d
                    C = torch.randn(1, 3, 4, 4, 4)
                    D = torch.randn(1, 1, 1, 1, 1)
                    if channels_last:
                        C = C.to(memory_format=torch.channels_last_3d)
                        D = D.to(memory_format=torch.channels_last_3d)
                    _run_test(C, D)
    def test_qnnpack_maxpool2d(self, A, kernel, stride, padding):
        import torch.nn.functional as F  # 导入PyTorch的函数库

        with override_quantized_engine('qnnpack'):  # 使用指定的量化引擎进行上下文管理
            A, (scale, zero_point, torch_type) = A  # 解包A参数并获取量化相关的元组信息
            X = torch.from_numpy(A)  # 将NumPy数组A转换为PyTorch张量X
            np_type = np.uint8  # 设置NumPy数据类型为无符号8位整数
            dilation = 1  # 设置膨胀率为1

            assume(kernel // 2 >= padding)  # 假设语句：确保内核不会超出边界

            iH, iW = X.shape[-2:]  # 获取输入张量X的高度和宽度

            oH = pool_output_shape(iH, kernel, padding, stride, dilation)  # 计算池化操作后的输出高度
            assume(oH > 0)  # 假设语句：确保输出高度大于0
            oW = pool_output_shape(iW, kernel, padding, stride, dilation)  # 计算池化操作后的输出宽度
            assume(oW > 0)  # 假设语句：确保输出宽度大于0

            k = (kernel, kernel)  # 内核大小的元组
            s = (stride, stride)  # 步幅的元组
            d = (dilation, dilation)  # 膨胀率的元组
            p = (padding, padding)  # 填充的元组

            q_max_pool = torch.ops.quantized.max_pool2d  # 获取量化最大池化操作的函数引用

            a = scale * (X - zero_point).to(dtype=torch.float)  # 对输入张量进行量化操作
            qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)  # 对a进行张量量化

            a_ref = qa.dequantize()  # 对量化后的张量qa进行反量化操作，得到参考结果a_ref

            a_pool = F.max_pool2d(a_ref, kernel_size=k, stride=s, padding=p,
                                  dilation=d)  # 使用PyTorch的函数库进行最大池化操作

            a_pool_nhwc = a_pool.permute([0, 2, 3, 1])  # 调整池化后张量的维度顺序

            qa_pool = q_max_pool(qa, k, s, p, d, ceil_mode=False)  # 使用量化最大池化操作对量化张量qa进行处理

            qa_pool_int = qa_pool.dequantize()  # 对量化最大池化操作的结果进行反量化操作，得到整数值

            np.testing.assert_equal(a_pool.numpy(), qa_pool_int.numpy())  # 使用NumPy进行数组相等性断言的测试

    @given(batch_size=st.integers(1, 5),
           channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(4, 10),
           width=st.integers(4, 10),
           kernel=st.integers(2, 5),
           stride=st.integers(1, 2),
           padding=st.integers(1, 2),
           scale=st.floats(0.2, 1.6),
           zero_point=st.integers(0, 25)
           )
    def test_avg_pool2d(
            self,
            batch_size,
            channels,
            height,
            width,
            kernel,
            stride,
            padding,
            scale,
            zero_point
            ):
        # 待填充
    ):
        # 使用 'qnnpack' 替换默认的量化引擎
        with override_quantized_engine('qnnpack'):
            # 导入 PyTorch 的函数库
            import torch.nn.functional as F
            # 创建一个随机整数张量作为输入 X_init
            X_init = torch.from_numpy(np.random.randint(
                0, 50, (batch_size, channels, height, width)))
            
            # 对输入张量 X_init 进行量化和缩放处理，转换为 torch.float 类型
            X = scale * (X_init - zero_point).to(dtype=torch.float)

            # 检查约束条件
            assume(kernel // 2 >= padding)  # Kernel 不能超出边界！

            # 获取输入张量 X 的高度和宽度
            iH, iW = X.shape[-2:]

            # 计算池化后的输出高度和宽度
            oH = pool_output_shape(iH, kernel, padding, stride, 1)
            assume(oH > 0)
            oW = pool_output_shape(iW, kernel, padding, stride, 1)
            assume(oW > 0)
            
            # 定义池化核、步长和填充
            k = (kernel, kernel)
            s = (stride, stride)
            p = (padding, padding)

            # 使用 PyTorch 的量化函数进行平均池化操作
            q_avg_pool = torch.ao.nn.quantized.functional.avg_pool2d

            # 对输入张量 X 进行整数量化
            x_q = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                            dtype=torch.quint8)

            # 使用非量化的张量进行平均池化操作
            a_pool = F.avg_pool2d(x_q.dequantize().to(torch.float), kernel_size=k, stride=s, padding=p)
            # 使用量化函数进行平均池化操作
            qa_pool = q_avg_pool(x_q, k, s, p)
            
            # 对平均池化后的输出进行整数量化
            a_pool_q = torch.quantize_per_tensor(a_pool, scale=scale, zero_point=zero_point,
                                                 dtype=torch.quint8)
            
            # 使用 numpy 的函数检查两个量化输出是否几乎相等
            np.testing.assert_array_almost_equal(a_pool_q.int_repr().numpy(),
                                                 qa_pool.int_repr().numpy(), decimal=0)


    @given(batch_size=st.integers(1, 5),
           channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(4, 20),
           width=st.integers(4, 20),
           output_height=st.integers(2, 10),
           output_width=st.integers(2, 10),
           scale=st.floats(0.2, 1.6),
           zero_point=st.integers(0, 25)
           )
    # 定义一个参数化测试函数，用于测试自适应平均池化功能
    def test_adaptive_avg_pool2d(
            self,
            batch_size,
            channels,
            height,
            width,
            output_height,
            output_width,
            scale,
            zero_point
    ):
        # 使用 qnnpack 引擎覆盖量化引擎上下文
        with override_quantized_engine('qnnpack'):
            # 检查约束条件
            assume(height >= output_height)
            assume(width >= output_width)

            import torch.nn.functional as F
            # 从随机整数创建张量 X_init
            X_init = torch.from_numpy(np.random.randint(
                0, 50, (batch_size, channels, height, width)))

            # 对 X_init 进行量化处理，转换为 torch.float 类型
            X = scale * (X_init - zero_point).to(dtype=torch.float)

            # 获取 X 的高度和宽度
            iH, iW = X.shape[-2:]

            # 获取 torch.ao.nn.quantized.functional.adaptive_avg_pool2d 函数
            q_avg_pool = torch.ao.nn.quantized.functional.adaptive_avg_pool2d

            # 对 X 进行量化处理，使用指定的 scale 和 zero_point
            x_q = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                            dtype=torch.quint8)

            # 对量化后的 X 进行自适应平均池化操作，调整为指定的输出尺寸 (output_height, output_width)
            a_pool = F.adaptive_avg_pool2d(x_q.dequantize().to(torch.float), (output_height, output_width))
            # 使用量化版本的自适应平均池化操作
            qa_pool = q_avg_pool(x_q, (output_height, output_width))
            # 对自适应平均池化输出进行量化处理
            a_pool_q = torch.quantize_per_tensor(a_pool, scale=scale, zero_point=zero_point,
                                                 dtype=torch.quint8)
            # 使用 numpy.testing 来比较两个量化张量的值是否几乎相等
            np.testing.assert_array_almost_equal(a_pool_q.int_repr().numpy(),
                                                 qa_pool.int_repr().numpy(), decimal=0)


    @given(batch_size=st.integers(1, 5),
           channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(4, 10),
           width=st.integers(4, 10),
           scale=st.floats(0.02, 2.6),
           zero_point=st.integers(0, 25))
    def test_mean(self, batch_size, channels, height, width, scale, zero_point):
        # 使用 qnnpack 引擎覆盖量化引擎上下文
        with override_quantized_engine('qnnpack'):
            # 设定维度为 (2, 3)
            dim = (2, 3)
            # 从随机整数创建张量 X_init
            X_init = torch.from_numpy(np.random.randint(
                0, 50, (batch_size, channels, height, width)))
            # 对 X_init 进行量化处理，转换为 torch.float 类型
            X = scale * (X_init - zero_point).to(dtype=torch.float)

            # 对 X 进行量化处理，使用指定的 scale 和 zero_point
            qX = torch.quantize_per_tensor(X, scale, zero_point, torch.quint8)
            # 计算 qX 的均值，沿指定的维度 dim
            Y = torch.mean(qX.dequantize(), dim)
            # 对均值张量 Y 进行量化处理，使用指定的 scale 和 zero_point
            Y = torch.quantize_per_tensor(Y, scale, zero_point, torch.quint8)
            # 使用量化版本的均值操作
            qY = torch.mean(qX, dim)
            # 使用 numpy.testing 来比较两个量化张量的值是否几乎相等
            np.testing.assert_array_almost_equal(Y.int_repr().numpy(), qY.int_repr().numpy(), decimal=0)

"""Tests the correctness of the quantized::hardtanh op."""
    # 定义测试方法 test_hardtanh，用于测试量化硬切线函数
    def test_hardtanh(self):
        # 检查是否支持 qnnpack 引擎，若不支持则直接返回
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return
        # 使用 qnnpack 引擎进行量化操作
        with override_quantized_engine('qnnpack'):
            # 定义测试用例的多种形状、内存布局格式、最小值、最大值
            shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
            memory_formats = (torch.channels_last, torch.contiguous_format)
            min_vals = (-0.5, -0.3, 0.5)
            max_vals = (-0.3, 0.3, 0.7)
            # 生成所有可能的测试组合
            test_cases = itertools.product(shapes, memory_formats, min_vals, max_vals)
            # 遍历测试用例
            for shape, memory_format, min_val, max_val in test_cases:
                # 创建随机张量 X，并设定量化参数
                X, scale, zero_point, torch_type = torch.randn(*shape), 1.0, 0, torch.quint8
                # 如果内存布局为 channels_last 但是形状不是四维的，跳过当前测试用例
                if memory_format == torch.channels_last and len(shape) != 4:
                    continue

                # 复制张量 X 到 Y
                Y = X.clone()
                # 对 Y 中小于 min_val 的元素进行修剪为 min_val
                Y[Y < min_val] = min_val
                # 对 Y 中大于 max_val 的元素进行修剪为 max_val
                Y[Y > max_val] = max_val
                # 对修剪后的张量 Y 进行量化
                qY = torch.quantize_per_tensor(Y, scale=scale,
                                               zero_point=zero_point, dtype=torch_type)
                # 对张量 X 进行量化
                qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)

                # 使用量化硬切线函数对 qX 进行操作
                qY_hat = torch.ao.nn.quantized.functional.hardtanh(qX, min_val, max_val)
                # 断言量化后的 qY_hat 与 qY 相等，若不相等则输出错误信息
                self.assertEqual(
                    qY, qY_hat,
                    msg=f"hardtanh failed:\nactual {qY_hat}\nexpected {qY}\nmemory_format {memory_format}")
"""Tests the correctness of the tensor comparators."""
# 定义一个测试类，用于测试张量比较操作的正确性
class TestComparatorOps(TestCase):
    """Tests the element-wise equality ops."""
    # 测试元素级相等操作

    @given(A=hu.tensor(shapes=((3, 4, 5),),
                       qparams=hu.qparams()),
           B=hu.tensor(shapes=((5,), (1, 5), (1, 1, 5), (4, 5), (3, 4, 5)),
                       qparams=hu.qparams()))
    # 使用 hypothesis 的 given 装饰器定义测试函数参数 A 和 B
    def test_compare_tensor_tensor(self, A, B):
        # 解包 A 和 B 的值以及它们的量化参数 (scale, zero_point, dtype)
        A, (scale_a, zero_point_a, dtype_a) = A
        B, (scale_b, zero_point_b, dtype_b) = B
        # 将 numpy 数组 A 和 B 转换为 PyTorch 张量
        tA = torch.from_numpy(A)
        tB = torch.from_numpy(B)

        # 对张量 tA 和 tB 进行量化，使用给定的量化参数 (scale, zero_point, dtype)
        qA = torch.quantize_per_tensor(tA, scale=scale_a, zero_point=zero_point_a,
                                       dtype=dtype_a)
        qB = torch.quantize_per_tensor(tB, scale=scale_b, zero_point=zero_point_b,
                                       dtype=dtype_b)
        # 将量化后的张量反量化为浮点张量
        dqA = qA.dequantize()
        dqB = qB.dequantize()

        # 定义需要测试的操作列表
        ops_under_test = ('__eq__', '__ne__', '__ge__', '__le__', '__gt__',
                          '__lt__', 'eq', 'ne', 'ge', 'le', 'gt', 'lt')

        # 遍历操作列表，对每个操作进行测试
        for op in ops_under_test:
            # 使用 getattr 获取操作函数，并对 dqA 和 dqB 执行操作
            result_ref = getattr(dqA, op)(dqB)
            result = getattr(qA, op)(qB)
            # 使用断言检查操作的结果是否与参考结果一致，如果不一致，输出错误信息
            self.assertEqual(result_ref, result,
                             msg=f"'tensor.{op}(tensor)'' failed")
            # 对 dqB 和 dqA 执行操作，测试广播性质是否翻转
            result_ref = getattr(dqB, op)(dqA)
            result = getattr(qB, op)(qA)
            # 使用断言再次检查反向操作的结果是否一致
            self.assertEqual(result_ref, result,
                             msg=f"'tensor.{op}(tensor)'' failed")
    # 定义一个测试方法，用于比较张量和标量的操作
    def test_compare_tensor_scalar(self, A, b):
        # 解包张量 A，获取其元组内的 (scale_a, zero_point_a, dtype_a)
        A, (scale_a, zero_point_a, dtype_a) = A
        # 将 NumPy 数组 A 转换为 PyTorch 张量 tA
        tA = torch.from_numpy(A)

        # 使用给定的缩放因子 scale_a、零点 zero_point_a 和数据类型 dtype_a 进行量化
        qA = torch.quantize_per_tensor(tA, scale=scale_a, zero_point=zero_point_a,
                                       dtype=dtype_a)
        # 将量化后的张量 qA 还原为浮点数张量 dqA
        dqA = qA.dequantize()

        # 定义可逆操作的名称元组
        ops_under_test_reversible = ('__eq__', '__ne__', '__ge__', '__le__',
                                     '__gt__', '__lt__')
        # 定义不可逆操作的名称元组
        ops_under_test_nonreversible = ('eq', 'ne', 'ge', 'le', 'gt', 'lt')

        # 对于每个可逆操作 op，执行以下操作
        for op in ops_under_test_reversible:
            # 使用 getattr 获取 dqA 对象上 op 方法的结果，与标量 b 进行比较
            result_ref = getattr(dqA, op)(b)
            # 使用 getattr 获取 qA 对象上 op 方法的结果，与标量 b 进行比较
            result = getattr(qA, op)(b)
            # 输出比较结果的日志信息
            note(f"result_ref 1: {result_ref}")
            note(f"result 1: {result}")
            # 使用 self.assertEqual 进行结果验证，如果不相等，输出失败消息
            self.assertEqual(result_ref, result,
                             msg=f"'tensor.{op}(scalar)'' failed")
            # 对 b 和 dqA 进行反向广播操作，获取结果
            result_ref = getattr(b, op)(dqA)
            result = getattr(b, op)(qA)
            # 输出反向广播结果的日志信息
            note(f"result_ref 2: {result_ref}")
            note(f"result 2: {result}")
            # 使用 self.assertEqual 进行结果验证，如果不相等，输出失败消息
            self.assertEqual(result_ref, result,
                             msg=f"'scalar.{op}(tensor)'' failed")

        # 对于每个不可逆操作 op，执行以下操作
        for op in ops_under_test_nonreversible:
            # 使用 getattr 获取 dqA 对象上 op 方法的结果，与标量 b 进行比较
            result_ref = getattr(dqA, op)(b)
            # 使用 getattr 获取 qA 对象上 op 方法的结果，与标量 b 进行比较
            result = getattr(qA, op)(b)
            # 输出比较结果的日志信息
            note(f"result_ref 3: {result_ref}")
            note(f"result 3: {result}")
            # 使用 self.assertEqual 进行结果验证，如果不相等，输出失败消息
            self.assertEqual(result_ref, result,
                             msg=f"'tensor.{op}(scalar)'' failed")
```