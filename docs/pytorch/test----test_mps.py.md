# `.\pytorch\test\test_mps.py`

```py
# Owner(s): ["module: mps"]

import io  # 导入 io 模块，用于处理输入输出流
import platform  # 导入 platform 模块，用于访问平台相关属性和功能
import sys  # 导入 sys 模块，用于访问与 Python 解释器交互的变量和函数
import math  # 导入 math 模块，提供数学函数
import random  # 导入 random 模块，生成伪随机数
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
import warnings  # 导入 warnings 模块，用于管理警告信息
import subprocess  # 导入 subprocess 模块，用于创建新进程和与它们交互
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import os  # 导入 os 模块，提供操作系统相关的功能
import copy  # 导入 copy 模块，用于复制对象
import gc  # 导入 gc 模块，Python 的垃圾回收器接口
import threading  # 导入 threading 模块，提供多线程相关的功能
import torch  # 导入 PyTorch 深度学习库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数式接口模块
import itertools  # 导入 itertools 模块，提供迭代器生成函数
from collections import defaultdict  # 从 collections 模块中导入 defaultdict 类
from torch import inf  # 从 torch 模块中导入 inf 常量，表示正无穷大
from torch.nn import Parameter  # 从 torch.nn 模块中导入 Parameter 类
from torch.testing._internal import opinfo  # 导入 PyTorch 内部测试相关模块
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, parametrize, run_tests, TestCase, download_file, IS_CI,
     NoTest, skipIfSlowGradcheckEnv, suppress_warnings)  # 导入测试工具函数和常量
from torch.testing import make_tensor  # 导入 make_tensor 函数，用于创建测试用张量
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types  # 导入数据类型相关的函数和常量
import torch.backends.mps  # 导入 PyTorch MPS 后端相关模块
from torch.distributions import Uniform, Exponential  # 导入分布相关类
from functools import partial  # 导入 functools 模块中的 partial 函数

from torch.testing._internal.common_methods_invocations import (
    op_db,  # 导入操作数据库
    DecorateInfo,  # 导入装饰器信息类
    UnaryUfuncInfo,  # 导入一元函数信息类
    ReductionOpInfo,  # 导入降维操作信息类
    SpectralFuncInfo,  # 导入频谱函数信息类
    BinaryUfuncInfo,  # 导入二元函数信息类
)
from torch.testing._internal.common_device_type import ops, dtypes, instantiate_device_type_tests, OpDTypes  # 导入设备类型相关的函数和常量
from torch.testing._internal.common_nn import NNTestCase  # 导入神经网络测试用例基类
from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel  # 导入量化相关函数
import numpy as np  # 导入 NumPy 数学库
import torch  # 导入 PyTorch 深度学习库
import torch.utils._pytree as pytree  # 导入 PyTorch 内部的 pytree 模块
from itertools import product  # 从 itertools 模块中导入 product 函数，用于求笛卡尔积
import operator  # 导入 operator 模块，提供 Python 中常见运算符的函数接口

test_consistency_op_db = copy.deepcopy(op_db)  # 深拷贝操作数据库，用于一致性测试
test_error_inputs_op_db = copy.deepcopy(op_db)  # 深拷贝操作数据库，用于错误输入测试

# 从 `test_ops.py` 复制的目的是为了复制 `test_numpy_ref` 的测试
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

def xfailIf(condition):
    def wrapper(func):
        if condition:
            return unittest.expectedFailure(func)  # 如果条件成立，将测试标记为预期失败
        else:
            return func
    return wrapper

def xfailIfMacOS14_4Plus(func):
    return unittest.expectedFailure(func) if product_version > 14.3 else func  # 如果 macOS 版本大于 14.3，则标记测试为预期失败；否则不标记

def mps_ops_grad_modifier(ops):
    }  # 定义函数 mps_ops_grad_modifier，参数 ops，但函数体缺失

MACOS_12_3_XFAILLIST_GRAD = {
    # 不支持的边界填充模式，前向传播成功但作为回退到 CPU
    'grid_sampler_2d': [torch.float32],
    # 未实现的操作
    'logaddexp2': [torch.float32],

}
    # 定义一个字典，记录在早期 macOS 13.3 之前因精度问题而失败的操作列表
    MACOS_BEFORE_13_3_XFAILLIST_GRAD = {
        'masked.softmin': [torch.float32, torch.float16],  # 对于 masked.softmin 操作，支持的数据类型包括 torch.float32 和 torch.float16
        'masked.softmax': [torch.float32, torch.float16],  # 对于 masked.softmax 操作，支持的数据类型包括 torch.float32 和 torch.float16
        'masked.log_softmax': [torch.float32, torch.float16],  # 对于 masked.log_softmax 操作，支持的数据类型包括 torch.float32 和 torch.float16

        # 不支持的边界填充模式，在前向传播中退回到 CPU 执行
        'grid_sampler_2d': [torch.float32],

        # 与 `argsort` 和 `sort` 相同的问题，涉及到重复元素（未定义行为）
        # 在前向传播中通过，因为 `msort` 只返回值，而不是索引，与 CPU 结果匹配
        # 在反向传播中，`sort` 使用了值和索引，导致 CPU 和 MPS 结果不匹配
        # 使用稳定的 `sort` 可以通过
        'msort': [torch.float16],
    }

    # 定义一个字典，记录跳过的梯度操作列表
    SKIPLIST_GRAD = {
        'nn.functional.pairwise_distance': [torch.float16],  # 对于 nn.functional.pairwise_distance 操作，支持的数据类型为 torch.float16
        # 断言失败：目标数据类型必须为 fp32
        'nn.functional.conv1d': [torch.float16],  # 对于 nn.functional.conv1d 操作，支持的数据类型为 torch.float16
        'nn.functional.conv2d': [torch.float16],  # 对于 nn.functional.conv2d 操作，支持的数据类型为 torch.float16
        'nn.functional.conv3d': [torch.float16],  # 对于 nn.functional.conv3d 操作，支持的数据类型为 torch.float16
        'nn.functional.conv_transpose1d': [torch.float16],  # 对于 nn.functional.conv_transpose1d 操作，支持的数据类型为 torch.float16
        'nn.functional.conv_transpose2d': [torch.float16],  # 对于 nn.functional.conv_transpose2d 操作，支持的数据类型为 torch.float16
        'nn.functional.conv_transpose3d': [torch.float16],  # 对于 nn.functional.conv_transpose3d 操作，支持的数据类型为 torch.float16
    }

    # 定义一个字典，记录在 macOS 13.3 及更高版本因重复元素问题导致的梯度失败的操作列表
    MACOS_13_3_XFAILLIST_GRAD = {
        # 与 `argsort` 和 `sort` 相同的问题，涉及到重复元素（未定义行为）
        # 在前向传播中通过，因为 `msort` 只返回值，而不是索引，与 CPU 结果匹配
        # 在反向传播中，`sort` 使用了值和索引，导致 CPU 和 MPS 结果不匹配
        # 使用稳定的 `sort` 可以通过
        'msort': [torch.float16],
    }

    # 定义一个字典，记录在 MPS 后端因下游函数未实现而失败的操作列表
    ON_MPS_XFAILLIST = {
        'linalg.matrix_rank': None,  # 'linalg.matrix_rank' 操作因 MPS 后端下游函数 'aten::_linalg_svd.U' 未实现而失败，暂时跳过

        # 异常：由 MPS 上索引 3 的样本输入引起
        'nn.functional.conv3d': [torch.float32],  # 对于 nn.functional.conv3d 操作，支持的数据类型为 torch.float32
    }

    # 定义一个函数，为操作添加装饰器
    def addDecorator(op, d) -> None:
        op.decorators = list(op.decorators) if op.decorators is not None else []  # 如果 op 已有装饰器列表，则转换为列表形式，否则创建一个空列表
        op.decorators.append(d)  # 将新的装饰器 d 添加到 op 的装饰器列表中
    # 遍历操作列表 `ops`
    for op in ops:
        # 组合操作名称和变体测试名称，形成唯一的键 `key`
        key = op.name + op.variant_test_name
        
        # 如果 `key` 存在于 `XFAILLIST_GRAD` 中，则为操作添加装饰器，标记为预期失败测试，并指定数据类型
        if key in XFAILLIST_GRAD:
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=XFAILLIST_GRAD[key]))

        # 如果 `key` 存在于 `SKIPLIST_GRAD` 中，则为操作添加装饰器，跳过该测试，并指定数据类型
        if key in SKIPLIST_GRAD:
            addDecorator(op, DecorateInfo(
                         unittest.skip,
                         dtypes=SKIPLIST_GRAD[key]))

        # 如果 `key` 存在于 `ON_MPS_XFAILLIST` 中，则为操作添加装饰器，标记为预期失败测试，并指定数据类型
        if key in ON_MPS_XFAILLIST:
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=ON_MPS_XFAILLIST[key]))

        # 如果 `key` 存在于 `MACOS_12_3_XFAILLIST_GRAD` 中，并且当前系统不是 macOS 13 或更新版本，则为操作添加装饰器，标记为预期失败测试，并指定数据类型
        if key in MACOS_12_3_XFAILLIST_GRAD and (not torch.backends.mps.is_macos13_or_newer()):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_12_3_XFAILLIST_GRAD[key]))

        # 如果 `key` 存在于 `MACOS_BEFORE_13_3_XFAILLIST_GRAD` 中，并且当前系统是 macOS 13 或更新版本，并且产品版本小于 13.3，则为操作添加装饰器，标记为预期失败测试，并指定数据类型
        if key in MACOS_BEFORE_13_3_XFAILLIST_GRAD and (torch.backends.mps.is_macos13_or_newer() and product_version < 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_BEFORE_13_3_XFAILLIST_GRAD[key]))

        # 如果 `key` 存在于 `MACOS_13_3_XFAILLIST_GRAD` 中，并且产品版本大于或等于 13.3，则为操作添加装饰器，标记为预期失败测试，并指定数据类型
        if key in MACOS_13_3_XFAILLIST_GRAD and (product_version >= 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_13_3_XFAILLIST_GRAD[key]))
        
        # 生成器，每次迭代返回操作 `op`
        yield op
# 定义函数 mps_ops_modifier，接受一个参数 ops，该参数是一个集合，包含了支持的复杂操作名称列表
def mps_ops_modifier(ops):
    # 支持的复杂操作集合，这些操作名称是字符串类型
    SUPPORTED_COMPLEX_OPS = {
        '__radd__',
        '__rmul__',
        '__getitem__',
        'abs',
        'add',
        'alias_copy',
        'argwhere',
        'atleast_1d',
        'atleast_2d',
        'atleast_3d',
        'as_strided',
        'as_strided_copy',
        'as_strided_scatter',
        'broadcast_tensors',
        'broadcast_to',
        'chalf',
        'cfloat',
        'chunk',
        'clone',
        'conj',
        'conj_physical',
        'contiguous',
        'diag',
        'diag_embed',
        'diagflat',
        'diagonal',
        'diagonal_copy',
        'diagonal_scatter',
        'dsplit',
        'empty',
        'empty_permuted',
        'empty_strided',
        'eye',
        'exp',
        'expand',
        'expand_as',
        'flatten',
        'fill',
        'full',
        'H',
        'hsplit',
        'imag',
        'index_select',
        'isfinite',
        'isinf',
        'isreal',
        'item',
        'kron',
        'linalg.diagonal',
        'linalg.svd',
        'linspace',
        'logspace',
        'linspacetensor_overload',
        'logspacetensor_overload',
        'mH',
        'mT',
        'masked_scatter',
        'masked_select',
        'meshgridlist_of_tensors',
        'meshgridvariadic_tensors',
        'movedim',
        'mul',
        'narrow',
        'narrow_copy',
        'nn.functional.conv1d',
        'nn.functional.conv2d',
        'nn.functional.conv_transpose1d',
        'nn.functional.conv_transpose2d',
        'nn.functional.feature_alpha_dropoutwithout_train',
        'nn.functional.padcircular',
        'nn.functional.tanhshrink',
        'nn.functional.unfold',
        'nonzero',
        'ones',
        'outer',
        'permute',
        'positive',
        'randn',
        'ravel',
        'real',
        'repeat_interleave',
        'reshape_as',
        'reshape',
        'resolve_conj',
        'resolve_neg',
        'scalar_tensor',
        'select',
        'sgn',
        'slice',
        'split',
        'split_with_sizes',
        'split_with_sizes_copy',
        'splitlist_args',
        'squeeze',
        'squeezemultiple',
        'sub',
        'svd',
        't',
        'tanh',
        'tensor_split',
        'transpose',
        'T',
        'unbind',
        'unflatten',
        'unfold',
        'unfold_copy',
        'unsafe_chunk',
        'unsafe_split',
        'unsqueeze',
        'view_as',
        'view_as_real',
        'view',
        'vsplit',
        'zero_',
        'zeros',
    }
    # 定义一个集合，包含在 MacOS 14.0 上支持的复杂操作函数名称
    AFTER_MACOS_14_0_SUPPORTED_COMPLEX_OPS = {
        '__rdiv__',                     # 右除法魔法方法
        '__rmatmul__',                  # 右矩阵乘法魔法方法
        '_chunk_cat',                   # 分块连接操作
        '_unsafe_masked_index',         # 不安全的掩码索引操作
        'acos',                         # 反余弦函数
        'acosh',                        # 反双曲余弦函数
        'all',                          # 判断所有元素是否为真
        'allclose',                     # 比较所有元素是否近似相等
        'any',                          # 判断任一元素是否为真
        'addcdiv',                      # 按元素相加，然后再除以某个张量
        'addcmul',                      # 按元素相加，然后再乘以某个张量
        'addmmdecomposed',              # 分解后的矩阵乘法相加操作
        'addmv',                        # 矩阵向量相加操作
        'asin',                         # 反正弦函数
        'atan',                         # 反正切函数
        'atanh',                        # 反双曲正切函数
        'bfloat16',                     # Bfloat16 类型支持
        'bmm',                          # 批次矩阵乘法
        'bool',                         # 转换为布尔型
        'cartesian_prod',               # 笛卡尔积操作
        'cat',                          # 沿指定维度连接张量
        'char',                         # 字符类型
        'column_stack',                 # 列堆叠操作
        'combinations',                 # 组合操作
        'corrcoef',                     # 相关系数计算
        'constant_pad_nd',              # 多维常数填充操作
        'cos',                          # 余弦函数
        'cosh',                         # 双曲余弦函数
        'count_nonzero',                # 计算非零元素个数
        'diff',                         # 计算差分
        'div',                          # 除法操作
        'divno_rounding_mode',          # 无舍入模式除法操作
        'dot',                          # 点积操作
        'dstack',                       # 深度堆叠操作
        'einsum',                       # Einstein 求和约定操作
        'eq',                           # 相等比较
        'equal',                        # 元素相等比较
        'exp2',                         # 2 的指数函数
        'expm1',                        # 指数减一函数
        'fft.fft',                      # 快速傅里叶变换
        'fft.fft2',                     # 二维快速傅里叶变换
        'fft.fftn',                     # 多维快速傅里叶变换
        'fft.fftshift',                 # 傅里叶变换频移操作
        'fft.ifft',                     # 快速逆傅里叶变换
        'fft.ifft2',                    # 二维快速逆傅里叶变换
        'fft.ifftn',                    # 多维快速逆傅里叶变换
        'fft.ifftshift',                # 逆傅里叶变换频移操作
        'fft.irfftn',                   # 多维快速实逆傅里叶变换
        'fft.irfft2',                   # 二维快速实逆傅里叶变换
        'fft.irfft',                    # 快速实逆傅里叶变换
        'fft.hfftn',                    # 多维快速 Hermite 变换
        'fft.hfft2',                    # 二维快速 Hermite 变换
        'fft.hfft',                     # 快速 Hermite 变换
        'flip',                         # 反转张量的元素
        'fliplr',                       # 左右翻转
        'flipud',                       # 上下翻转
        'float',                        # 转换为浮点型
        'gradient',                     # 计算梯度
        'half',                         # 半精度浮点数类型支持
        'hstack',                       # 水平堆叠操作
        'inner',                        # 内积操作
        'int',                          # 转换为整型
        'isclose',                      # 判断元素是否近似相等
        'isnan',                        # 判断是否为 NaN
        'ldexp',                        # 乘指数后再转换
        'linalg.multi_dot',             # 多个矩阵乘法
        'linalg.pinv',                  # 矩阵的伪逆
        'log10',                        # 10 的对数函数
        'log1p',                        # 自然对数加一函数
        'log2',                         # 2 的对数函数
        'log',                          # 自然对数函数
        'logical_and',                  # 逻辑与操作
        'logical_not',                  # 逻辑非操作
        'logical_or',                   # 逻辑或操作
        'logical_xor',                  # 逻辑异或操作
        'long',                         # 长整型支持
        'masked_fill',                  # 掩码填充操作
        'masked.mean',                  # 掩码平均值操作
        'masked.prod',                  # 掩码乘积操作
        'masked.std',                   # 掩码标准差操作
        'masked.sum',                   # 掩码求和操作
        'masked.var',                   # 掩码方差操作
        'matmul',                       # 矩阵乘法操作
        'mean',                         # 平均值操作
        'mm',                           # 矩阵乘法操作
        'mv',                           # 矩阵向量乘法操作
        'ne',                           # 不等比较操作
        'neg',                          # 取负操作
        'nn.functional.padconstant',    # 常数填充操作
        'nn.functional.padreflect',     # 反射填充操作
        'nn.functional.padreplicate',   # 复制填充操作
        'nn.functional.pixel_shuffle',  # 像素重排操作
        'nn.functional.pixel_unshuffle',# 像素反重排操作
        'nn.functional.rms_norm',       # 均方根归一化操作
        'nn.functional.softsign',       # Softsign 激活函数
        'pinverse',                     # 伪逆操作
        'prod',                         # 计算乘积操作
        'reciprocal',                   # 反数操作
        'roll',                         # 循环移动操作
        'rot90',                        # 旋转 90 度操作
        'rsqrt',                        # 平方根倒数操作
        'short',                        # 短整型支持
        'sigmoid',                      # Sigmoid 激活函数
        'sin',                          # 正弦函数
        'sinh',                         # 双曲正弦函数
        'sqrt',                         # 平方根函数
        'square',                       # 平方操作
        'stack',                        # 堆叠操作
        'stft',                         # 短时傅里叶变换
        'sum',                          # 求和操作
        'sum_to_size',                  # 指定大小求和操作
        'tan',                          # 正切函数
        'tensordot',                    # 张量点积操作
        'trace',                        # 迹操作
        'trapz',                        # 梯形积分操作
        'trapezoid',                    # 梯形积分操作
        'tril',                         # 下三角矩阵操作
        'triu',                         # 上三角矩阵操作
        'true_divide',                  # 真除操作
        'vstack',                       # 垂
    MACOS_BEFORE_13_3_XFAILLIST = {
        # MacOS 13.3 之前的错误列表，因精度问题导致失败（由于 fast-math）。在 MacOS 13.3+ 中已修复。
        'tan': [torch.float32],  # 正切函数在特定浮点数类型下出现问题
        'cdist': [torch.float32],  # 距离计算函数在特定浮点数类型下出现问题

        # CPU 错误：在 CPU 上未能正确处理除以 0.0 的情况
        'atan2': [torch.bool, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],

        # 在 macOS 12 上测试通过，因为回退到 CPU
        # 使用重复索引的 argsort 情况（行为未定义）：
        #  - CPU 输出: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], devuce='cpu')
        #  - MPS 输出: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
        # 索引 30 和 5133 的元素相同。由于 CPU 不使用 stable=True 的 argsort，这些情况导致行为未定义。
        'argsort': [torch.float16, torch.int8, torch.uint8, torch.bool],
        # 与 `argsort` 类似的问题，使用重复索引，检查排序后的值及其索引。
        # 排序张量的值与 CPU 匹配，但返回的索引会导致行为未定义。
        'sort': [torch.int8, torch.uint8, torch.bool, torch.float16],
        # 不支持的数据类型
        'cumsum': [torch.int64],
        'cumprod': [torch.int64],
        'cumulative_trapezoid': [torch.int64],
        'masked.cumsum': [torch.int64],
        'masked.cumprod': [torch.int64],
        'linalg.vander': [torch.int64],
    }

    MACOS_AFTER_13_1_XFAILLIST = {
        # 在 macOS 13.2 之前，回退到 CPU 并通过正向传播
        'grid_sampler_2d': [torch.float32],  # 不支持的边界填充模式

        # CPU 和 MPS 之间的一致性错误，最大绝对容差为 2
        'nn.functional.interpolatebilinear': [torch.uint8],
    }

    MACOS_13_3_XFAILLIST = {
        # 由于 fp16 的精度问题导致失败
        # 在 CPU 和 MPS 上都有可能产生无限结果的测试用例
        # 'nn.functional.pairwise_distance': [torch.float16],

        # 在 macOS 12 上测试通过，因为回退到 CPU
        # 使用重复索引的 argsort 情况（行为未定义）：
        #  - CPU 输出: tensor([2546, 6917, 3181,  ..., 7128, 5133,   30], devuce='cpu')
        #  - MPS 输出: tensor([2546, 6917, 3181,  ..., 7128,   30, 5133], device='mps:0')
        # 索引 30 和 5133 的元素相同。由于 CPU 不使用 stable=True 的 argsort，这些情况导致行为未定义。
        'argsort': [torch.float16, torch.int8, torch.uint8, torch.bool],
        # 与 `argsort` 类似的问题，使用重复索引，检查排序后的值及其索引。
        # 排序张量的值与 CPU 匹配，但返回的索引会导致行为未定义。
        'sort': [torch.int8, torch.uint8, torch.bool, torch.float16],
    }
    # MacOS 版本 14.4 及更早版本的操作失败列表
    MACOS_BEFORE_14_4_XFAILLIST = {
        # 这些操作在 14.4 版本正常工作，但在 14.2 或 13.x 版本中失败
        'fft.hfft2': [torch.complex64],
    }

    # 预计这些操作无法正常工作
    }

    if product_version < 14.0:
        # 在 MacOS 14 版本中新增了 FFT 和 BFloat16 支持
        # 更新未实现操作失败列表，如果操作在 < 14.0 版本中使用则会失败
        UNIMPLEMENTED_XFAILLIST.update({
            'bfloat16': None,
            'fft.fft': None,
            'fft.fft2': None,
            'fft.fftn': None,
            'fft.hfft': None,
            'fft.hfft2': None,
            'fft.hfftn': None,
            'fft.ifft': None,
            'fft.ifft2': None,
            'fft.ifftn': None,
            'fft.ihfft': None,
            'fft.ihfft2': None,
            'fft.ihfftn': None,
            'fft.irfft': None,
            'fft.irfft2': None,
            'fft.irfftn': None,
            'fft.rfft': None,
            'fft.rfft2': None,
            'fft.rfftn': None,
            'stft': None,
            # 在整数情况下，TestConsistencyCPU.test_output_match_isin_cpu 的错误会导致失败，
            # 在后续的操作系统版本中无法复现。如果在 < 14.0 版本中使用这些操作，会增加断言来验证
            'isin': [torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8],
            'nn.functional.max_pool2d': [torch.uint8],
        })

    }

    # MPS（多进程服务）的操作失败列表
    ON_MPS_XFAILLIST = {
        # 由于 MPS 后端缺乏下游函数实现而导致的失败
        # TODO: 一旦下游函数 'aten::_linalg_svd.U' 实现了，移除这些操作
        'linalg.matrix_rank': None,
    }

    # 空操作跳过列表
    EMPTY_OPS_SKIPLIST = {
        # 使用未初始化数据填充张量，在 CPU 上会导致与 MPS 不匹配，
        # 有时候会匹配，因此跳过这些操作
        # 参考 https://github.com/pytorch/pytorch/issues/100175
        'new_empty': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'new_empty_strided': [torch.bool, torch.float16, torch.float32, torch.int16,
                              torch.int32, torch.int64, torch.uint8, torch.int8],
        'empty_strided': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        # CPU: empty 返回全部为 0，并且与 MPS 的分配（MacOS 13）不匹配。
        # 参考 https://pytorch.org/docs/2.0/generated/torch.empty.html
        'empty': [torch.bool, torch.float16, torch.float32, torch.int16,
                  torch.int32, torch.int64, torch.uint8, torch.int8],
        'empty_like': [torch.bool, torch.float16, torch.float32, torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8],
        'empty_permuted': [torch.bool, torch.float16, torch.float32, torch.int16,
                           torch.int32, torch.int64, torch.uint8, torch.int8],
    }
    # 定义一个跳过测试的字典，包含不支持的操作及其不兼容的输入类型
    SKIPLIST = {
        # nn.functional.avg_pool2d 不支持的操作，不兼容 torch.float16 类型
        'nn.functional.avg_pool2d': [torch.float16],

        # nn.functional.conv3d 在 M1 上不工作，在 M2 上部分工作，但不支持 torch.float16
        'nn.functional.conv3d': None,
    }

    # 定义一个函数，用于向操作对象添加装饰器
    def addDecorator(op, d) -> None:
        # 如果操作对象已有装饰器列表，则将其转换为列表形式
        op.decorators = list(op.decorators) if op.decorators is not None else []
        # 向操作对象的装饰器列表中添加新的装饰器
        op.decorators.append(d)

    # 遍历 ops 列表中的每个操作对象
    for op in ops:
        # 构建当前操作对象的唯一键，由操作名称和变体测试名称组成
        key = op.name + op.variant_test_name

        # 如果键存在于 EMPTY_OPS_SKIPLIST 中，则为操作对象添加装饰信息，跳过测试，并指定不支持的数据类型
        if key in EMPTY_OPS_SKIPLIST:
            addDecorator(op, DecorateInfo(
                         unittest.skip("Skipping empty ops."),
                         dtypes=EMPTY_OPS_SKIPLIST[key]))

        # 如果键存在于 SKIPLIST 中，则为操作对象添加装饰信息，跳过测试，并根据键指定的类型列表指定不支持的数据类型
        if key in SKIPLIST:
            addDecorator(op, DecorateInfo(unittest.skip("Skipped!"), dtypes=SKIPLIST[key]))

        # 对于 UNIMPLEMENTED_XFAILLIST、UNDEFINED_XFAILLIST、ON_MPS_XFAILLIST 中的每个列表
        for xfaillist in [UNIMPLEMENTED_XFAILLIST, UNDEFINED_XFAILLIST, ON_MPS_XFAILLIST]:
            # 如果键存在于当前列表中，则为操作对象添加装饰信息，指定预期的测试失败，并根据键指定的类型列表指定失败的数据类型
            if key in xfaillist:
                addDecorator(op, DecorateInfo(
                             unittest.expectedFailure,
                             dtypes=xfaillist[key]))

        # 如果键存在于 MACOS_BEFORE_14_4_XFAILLIST 并且当前产品版本小于 14.4，则为操作对象添加装饰信息，指定预期的测试失败，并根据键指定的类型列表指定失败的数据类型
        if key in MACOS_BEFORE_14_4_XFAILLIST and (product_version < 14.4):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_BEFORE_14_4_XFAILLIST[key]))

        # 如果键存在于 MACOS_BEFORE_13_3_XFAILLIST 并且当前产品版本小于 13.3，则为操作对象添加装饰信息，指定预期的测试失败，并根据键指定的类型列表指定失败的数据类型
        if key in MACOS_BEFORE_13_3_XFAILLIST and (torch.backends.mps.is_macos13_or_newer() and product_version < 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_BEFORE_13_3_XFAILLIST[key]))

        # 如果键存在于 MACOS_AFTER_13_1_XFAILLIST 并且当前产品版本支持 macOS 13.1 或更新版本，则为操作对象添加装饰信息，指定预期的测试失败，并根据键指定的类型列表指定失败的数据类型
        if key in MACOS_AFTER_13_1_XFAILLIST and torch.backends.mps.is_macos13_or_newer(2):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_AFTER_13_1_XFAILLIST[key]))

        # 如果键存在于 MACOS_13_3_XFAILLIST 并且当前产品版本大于或等于 13.3，则为操作对象添加装饰信息，指定预期的测试失败，并根据键指定的类型列表指定失败的数据类型
        if key in MACOS_13_3_XFAILLIST and (product_version >= 13.3):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_13_3_XFAILLIST[key]))

        # 如果键存在于 MACOS_12_3_XFAILLIST 并且当前产品版本不支持 macOS 13 或更高版本，则为操作对象添加装饰信息，指定预期的测试失败，并根据键指定的类型列表指定失败的数据类型
        if key in MACOS_12_3_XFAILLIST and (not torch.backends.mps.is_macos13_or_newer()):
            addDecorator(op, DecorateInfo(
                         unittest.expectedFailure,
                         dtypes=MACOS_12_3_XFAILLIST[key]))

        # 如果键不在 SUPPORTED_COMPLEX_OPS 中，并且（键不在 AFTER_MACOS_14_0_SUPPORTED_COMPLEX_OPS 中 或 产品版本小于 14.0），则为操作对象添加装饰信息，指定预期的测试失败，并指定 torch.complex32 和 torch.complex64 类型
        if key not in SUPPORTED_COMPLEX_OPS and (key not in AFTER_MACOS_14_0_SUPPORTED_COMPLEX_OPS or product_version < 14.0):
            addDecorator(op, DecorateInfo(unittest.expectedFailure, dtypes=[torch.complex32, torch.complex64]))

        # 生成当前操作对象
        yield op
def mps_ops_error_inputs_modifier(ops):
    # 定义一组不支持的操作示例集合
    XFAILLIST = {
        # 不会引发异常
        '__rmod__',
        '__rsub__',
        '__rpow__',
        'bernoulli',
        'clamp_max',
        'clamp_min',
        'masked_scatter',

        # 不支持 float64 类型
        'cat',
        'complex',
        'multinomial',
        'nn.functional.conv1d',
        'nn.functional.conv2d',
        'nn.functional.conv3d',
        'gather',
        'scatter',
        'scatter_add',

        # 不支持复杂类型
        'masked_fill',

        # MPS 不支持超过 16 维的张量维度
        'amax',
        'amin',
        'aminmax',

        # 内存重叠检查
        'index_select',

        # 未实现的操作
        'logcumsumexp',
    }

    # 添加装饰器到操作对象上的辅助函数
    def addDecorator(op, d) -> None:
        op.decorators = list(op.decorators) if op.decorators is not None else []
        op.decorators.append(d)

    # 遍历操作列表
    for op in ops:
        # 如果操作没有错误输入函数，则跳过
        if op.error_inputs_func is None:
            continue
        # 构造操作的唯一标识符
        key = op.name + op.variant_test_name
        # 如果操作在不支持的列表中，则添加期望失败的装饰器
        if key in XFAILLIST:
            addDecorator(op, DecorateInfo(unittest.expectedFailure))
        # 生成操作对象
        yield op

# 检查是否支持 MPS，如果不支持，则输出警告信息并设置相关测试为 NoTest
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    NNTestCase = NoTest  # noqa: F811

# 获取操作系统的产品版本号
product_version = float('.'.join(platform.mac_ver()[0].split('.')[:2]) or -1)
# 获取系统总内存大小
total_memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))

# 确定是否启用 MPS 内存泄漏检查
TEST_MPS_MEM_LEAK_CHECK = os.getenv('PYTORCH_TEST_MPS_MEM_LEAK_CHECK', '0') == '1'

# 创建装饰器函数，根据条件决定是否跳过 MPS 内存泄漏检查
def skipMPSMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_mps_memory_leak_check', True):
            fn._do_mps_memory_leak_check = not condition
        return fn
    return dec

# 定义 MPS 内存泄漏检查的上下文管理器类
class MpsMemoryLeakCheck:
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

    def __enter__(self):
        # 如果存在内存分配，则进行垃圾回收
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > 0:
            gc.collect()
            torch.mps.empty_cache()

        # 在运行测试之前获取缓存分配器和驱动程序的统计信息
        self.caching_allocator_before = torch.mps.current_allocated_memory()
        self.driver_before = torch.mps.driver_allocated_memory()
    # 当执行期间出现异常时，不进行内存泄漏检查
    if exec_type is not None:
        return

    # 比较缓存分配器在统计前后的内存情况
    # 分配内存增加可能表明存在内存泄漏的不一致性
    discrepancy_detected = False
    caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
    if caching_allocator_mem_allocated > self.caching_allocator_before:
        discrepancy_detected = True

    # 如果没有检测到不一致性，则提前结束
    if not discrepancy_detected:
        return

    # 执行垃圾回收和空缓存操作以验证不一致性是否持续存在，并由驱动程序 API 确认
    gc.collect()
    torch.mps.empty_cache()

    # 重新设置不一致性标志
    discrepancy_detected = True

    # 多次查询内存以确保泄漏不是暂时的
    for n in range(3):
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        driver_mem_allocated = torch.mps.driver_allocated_memory()

        caching_allocator_discrepancy = False
        driver_discrepancy = False

        if caching_allocator_mem_allocated > self.caching_allocator_before:
            caching_allocator_discrepancy = True

        if driver_mem_allocated > self.driver_before:
            driver_discrepancy = True

        # 如果既没有缓存分配器的不一致性，也没有驱动程序的不一致性，则认为泄漏是误报，退出循环
        if not (caching_allocator_discrepancy or driver_discrepancy):
            discrepancy_detected = False
            break

    # 如果只有缓存分配器的不一致性而没有驱动程序的不一致性，则发出警告
    if caching_allocator_discrepancy and not driver_discrepancy:
        msg = ("MPS caching allocator reports a memory leak not "
               f"verified by the driver API in {self.name}! "
               f"Caching allocator allocated memory was {self.caching_allocator_before} "
               f"and is now reported as {caching_allocator_mem_allocated}. "
               f"MPS driver allocated memory was {self.driver_before} and is now {driver_mem_allocated}.")
        warnings.warn(msg)
    # 如果缓存分配器和驱动程序的不一致性均被确认，则抛出运行时错误
    elif caching_allocator_discrepancy and driver_discrepancy:
        msg = (f"MPS driver API confirmed a leak in {self.name}! "
               f"Caching allocator allocated memory was {self.caching_allocator_before} "
               f"and is now reported as {caching_allocator_mem_allocated}. "
               f"MPS driver allocated memory was {self.driver_before} and is now {driver_mem_allocated}.")
        raise RuntimeError(msg)
# Expand TestCase class with Memory Leak Detection on MPS device
# 扩展 TestCase 类以在 MPS 设备上检测内存泄漏

class TestCaseMPS(TestCase):
    _do_mps_memory_leak_check = True  # 设置 MPS 内存泄漏检测标志为 True

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # 如果测试方法存在，则根据条件包装测试方法以进行 MPS 内存检查
            if TEST_MPS_MEM_LEAK_CHECK:  # 如果全局变量 TEST_MPS_MEM_LEAK_CHECK 为 True
                if self._do_mps_memory_leak_check:  # 如果类属性 _do_mps_memory_leak_check 为 True
                    self.wrap_with_mps_policy(method_name, self.assertLeaksNoMpsTensors)

    def assertLeaksNoMpsTensors(self, name=None):
        # 断言在运行过程中没有 MPS 张量内存泄漏
        name = self.id() if name is None else name
        return MpsMemoryLeakCheck(self, name)

    def wrap_with_mps_policy(self, method_name, policy):
        # 根据指定的策略对方法进行包装
        test_method = getattr(self, method_name)
        setattr(self, method_name, super().wrap_method_with_policy(test_method, policy))

    # checks for leaks even if TEST_MPS_MEM_LEAK_CHECK is 0
    # 即使 TEST_MPS_MEM_LEAK_CHECK 为 0，也会检查内存泄漏
    def wrap_with_mps_memory_check(self, method):
        return super().wrap_method_with_policy(method, self.assertLeaksNoMpsTensors)


class TestMemoryLeak(TestCaseMPS):
    def test_mps_memory_leak_detection(self):
        l = []

        @self.wrap_with_mps_memory_check
        def no_leak():
            pass

        # 触发一个有意的内存泄漏
        @self.wrap_with_mps_memory_check
        def leak_gpu0():
            # 增加到 8MB 以强制获取新的内存块，克服不同平台上的块大小差异
            l.append(torch.randn(1024 * 1024 * 8, device=torch.device("mps")))

        no_leak()

        # 检查是否发出了关于内存泄漏的运行时错误，以确认内存泄漏检测是否成功
        with self.assertRaisesRegex(RuntimeError, r"MPS driver API confirmed .+"):
            leak_gpu0()

    def test_copy_cast_no_leak(self):

        def step(x):
            x = x.to(device='cpu', dtype=torch.float32)
            x = x.to(device='mps', dtype=torch.float16)

        a = torch.randn(128, 128, device='mps', dtype=torch.float16)
        # 预热 / 预构建 MPS 着色器（否则在 13.2 上检查将失败）
        step(a)
        torch.mps.empty_cache()
        driver_before = torch.mps.driver_allocated_memory()
        step(a)
        torch.mps.empty_cache()
        driver_after = torch.mps.driver_allocated_memory()
        self.assertEqual(driver_before, driver_after, f"Detected {driver_after-driver_before} bytes leak of GPU memory")


class TestPixelShuffle(TestCaseMPS):
    pass


class MPSReluTest(TestCaseMPS):
    def _npRelu(self, np_features):
        return np.maximum(np_features, np.zeros(np_features.shape)).astype(np_features.dtype)
    def testNpRelu(self):
        # 使用 torch.testing.assert_close 来测试两个数组是否在误差范围内相等
        torch.testing.assert_close(
            np.array([[0., 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
            self._npRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7, 0.9]])
            )
        )

    def _testRelu(self, np_features, device):
        # 使用自定义的 _npRelu 方法对输入的 numpy 特征进行 Relu 操作
        np_relu = self._npRelu(np_features)
        # 将 numpy 数组转换为 PyTorch Tensor，并根据参数 "device" 将 Tensor 移动到 CPU/GPU
        py_tensor = torch.from_numpy(np_features).to(device)
        # 使用 PyTorch 的 nn.ReLU 创建一个 ReLU 激活函数对象，inplace=False 表示不原地修改
        py_relu = torch.nn.ReLU(inplace=False)(py_tensor)
        # 将 PyTorch Tensor 移动到 CPU
        py_relu_cpu = py_relu.to("cpu")

        # 断言 numpy 计算的结果与 PyTorch 计算的结果是否一致
        self.assertEqual(np_relu, py_relu_cpu)

    def _testReluInPlace(self, np_features, device):
        # 使用自定义的 _npRelu 方法对输入的 numpy 特征进行 Relu 操作
        np_relu = self._npRelu(np_features)
        # 将 numpy 数组转换为 PyTorch Tensor，并根据参数 "device" 将 Tensor 移动到 CPU/GPU
        py_tensor = torch.from_numpy(np_features).to(device)
        # 使用 PyTorch 的 nn.ReLU 创建一个 ReLU 激活函数对象，inplace=True 表示原地修改
        py_relu = torch.nn.ReLU(inplace=True)(py_tensor)
        # 将 PyTorch Tensor 移动到 CPU
        py_relu_cpu = py_relu.to("cpu")

        # 断言 numpy 计算的结果与 PyTorch 计算的结果是否一致
        self.assertEqual(np_relu, py_relu_cpu)
        # 原地 Relu 修改了初始输入，确保修改后的 Tensor 与 Relu 输出一致
        self.assertEqual(np_relu, py_tensor.to("cpu"))

    def testNumbersCPU(self):
        for t in [np.int32]:
            # 强制在 CPU 上执行，即使对于该类型可能存在 GPU 内核
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu"
            )
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu"
            )

    def testNumbersGPU(self):
        for t in [np.float16, np.float32]:
            # 在 GPU 上执行，指定 "mps" 作为设备
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps"
            )
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps"
            )
            # 测试空的 numpy 数组在 GPU 上的 Relu 操作
            self._testRelu(np.array([]).astype(t), device="mps")
            self._testReluInPlace(np.array([]).astype(t), device="mps")
class MatmulTest(TestCaseMPS):
    # 定义一个测试类 MatmulTest，继承自 TestCaseMPS
    def _helper(self, shape_tensor_1, shape_tensor_2, expand_tensor_1_shape=None, expand_tensor_2_shape=None):
        # 辅助函数 _helper 用于执行矩阵乘法的测试
        if expand_tensor_1_shape:
            # 如果指定了扩展 tensor1 的形状，则生成一个随机张量并进行扩展
            tensor1_mps = torch.randn(shape_tensor_1, device="mps").expand(expand_tensor_1_shape)
        else:
            # 否则生成一个指定形状的随机张量
            tensor1_mps = torch.randn(shape_tensor_1, device="mps")

        if expand_tensor_2_shape:
            # 如果指定了扩展 tensor2 的形状，则生成一个随机张量并进行扩展
            tensor2_mps = torch.randn(shape_tensor_2, device="mps").expand(expand_tensor_2_shape)
        else:
            # 否则生成一个指定形状的随机张量
            tensor2_mps = torch.randn(shape_tensor_2, device="mps")

        # 将 tensor1_mps 和 tensor2_mps 转移到 CPU
        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")

        # 在 CPU 上执行矩阵乘法
        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)
        # 在 MPS 设备上执行矩阵乘法
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        # 断言两个乘法的结果在 CPU 上是相等的
        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

    def test_vector_x_vector(self):
        # 测试向量乘法，使用 torch.matmul 实现，相当于点积
        self._helper(3, 3)

    def test_matrix_x_vector(self):
        # 测试矩阵和向量的乘法，使用 torch.matmul 实现，相当于 addmv 操作
        self._helper((3, 4), 4)

    def test_batched_matrix_x_broadcasted_vector(self):
        # 测试批量矩阵和广播向量的乘法
        self._helper((10, 3, 4), 4)

    def test_batched_matrix_x_batched_matrix(self):
        # 测试批量矩阵和批量矩阵的乘法，使用 torch.matmul 实现，相当于 bmm 操作
        self._helper((10, 3, 4), (10, 4, 5))

    def test_batched_matrix_x_broadcasted_matrix(self):
        # 测试批量矩阵和广播矩阵的乘法
        self._helper((10, 3, 4), (4, 5))


class MPSLeakyReluTest(TestCaseMPS):
    # 定义一个测试类 MPSLeakyReluTest，继承自 TestCaseMPS

    def _npLeakyRelu(self, np_features, negative_slope=0.1):
        # 定义一个 Numpy 实现的 Leaky ReLU 激活函数
        return np.maximum(np_features, negative_slope * np_features).astype(np_features.dtype)

    def testNpLeakyRelu(self):
        # 测试 Numpy 实现的 Leaky ReLU 激活函数
        torch.testing.assert_close(
            np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                      [0.1, -0.03, 0.5, -0.07, 0.9]]),
            self._npLeakyRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]]),
                negative_slope=0.1))

    def _testLeakyRelu(self, shape, dtype, negative_slope, contiguous):
        # 定义测试 Leaky ReLU 激活函数的函数，包括 MPS 设备上的测试

        # 在 CPU 上生成随机张量，并将其转移到 MPS 设备上
        cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
        mps_x = cpu_x.detach().clone().to('mps')

        if not contiguous and not (0 in shape or len(shape) < 2):
            # 如果不是连续的张量，并且不是零维或维度小于 2，则转置张量使其不连续
            cpu_x = cpu_x.transpose(0, 1)
            mps_x = mps_x.transpose(0, 1)
            assert not mps_x.is_contiguous()

        # 在 CPU 张量上启用梯度计算
        cpu_x.requires_grad_()
        mps_x.requires_grad_()

        # 创建 Leaky ReLU 操作符
        relu_op = torch.nn.LeakyReLU(negative_slope)

        # 在 CPU 和 MPS 设备上执行 Leaky ReLU 操作
        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        
        # 断言在 CPU 上计算得到的结果与 MPS 设备上的结果非常接近
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # 测试反向传播

        # 创建一个梯度张量
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')

        # 在 MPS 设备上执行反向传播
        mps_leaky_relu.backward(gradient=mps_grad)
        # 在 CPU 上执行反向传播
        cpu_leaky_relu.backward(gradient=cpu_grad)

        # 断言 CPU 张量的梯度不为空
        assert cpu_x.grad is not None
        # 断言 CPU 和 MPS 设备上的梯度相等
        self.assertEqual(cpu_x.grad, mps_x.grad)
    # 定义一个测试函数，用于测试不同数据类型和形状的张量
    def testNumbersCPU(self):
        # 遍历不同的数据类型
        for t in [torch.float, torch.half]:
            # 遍历不同的张量形状
            for shape in [[], (0,), (0, 3), (4,), (4, 3), (5, 4, 3)]:
                # 遍历是否是连续张量
                for contiguous in [True, False]:
                    # 调用测试 LeakyReLU 函数，传入张量形状、数据类型、负斜率和是否连续的参数
                    self._testLeakyRelu(shape,
                                        dtype=t,
                                        negative_slope=0.2,
                                        contiguous=contiguous)
class TestAvgPool(TestCaseMPS):
    # 定义一个测试类 TestAvgPool，继承自 TestCaseMPS

    def _sum_pool2d(self, x, kernel_size):
        # 定义一个私有方法 _sum_pool2d，接收参数 x 和 kernel_size
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        # 使用 torch.nn.functional.unfold 对输入 x 进行二维池化操作，得到滑动窗口视图 windows
        return torch.sum(windows, dim=1)
        # 对滑动窗口视图 windows 沿着第一个维度进行求和，并返回结果

    def _sum_pool3d(self, x, kernel_size):
        # 定义一个私有方法 _sum_pool3d，接收参数 x 和 kernel_size
        # 因为 unfold 不支持 3D 滑动窗口，所以我们将张量分割成多个小张量并计算它们的和
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # 将输入张量 x 按照 kernel_size[0] 分割，并计算每个分割张量在第一个维度上的和
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        # 对每个分割后的张量 t，调用 _sum_pool2d 方法，先将其扩展两次以适应 sum_pool2d 的输入要求
        joined_x = torch.cat(splited_x)
        # 将所有池化后的结果张量连接起来
        return joined_x.view(1, joined_x.numel())
        # 返回一个将 joined_x 展平成一维后的视图

    def _avg_pool2d(self, x, kernel_size):
        # 定义一个私有方法 _avg_pool2d，接收参数 x 和 kernel_size
        size = reduce(operator.mul, kernel_size)  # noqa: F821
        # 计算 kernel_size 中所有元素的乘积，作为总的池化窗口大小
        return self._sum_pool2d(x, kernel_size) / size
        # 调用 _sum_pool2d 方法对输入 x 进行池化，并返回池化结果除以总大小的结果

    def _avg_pool3d(self, x, kernel_size):
        # 定义一个私有方法 _avg_pool3d，接收参数 x 和 kernel_size
        size = reduce(operator.mul, kernel_size)  # noqa: F821
        # 计算 kernel_size 中所有元素的乘积，作为总的池化窗口大小
        return self._sum_pool3d(x, kernel_size) / size
        # 调用 _sum_pool3d 方法对输入 x 进行池化，并返回池化结果除以总大小的结果

    def test_avg_pool2d_with_zero_divisor(self):
        # 定义测试方法 test_avg_pool2d_with_zero_divisor
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: F.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))
        # 断言在使用 F.avg_pool2d 对一个全零的 3x3x3 张量进行池化时，抛出 RuntimeError 异常，并且异常信息包含 "divisor must be not zero"

    def test_doubletensor_avg_pool2d_with_divisor(self):
        # 定义测试方法 test_doubletensor_avg_pool2d_with_divisor
        n, m = 3, 3
        input = torch.rand(1, 1, n, m)
        # 创建一个随机初始化的 1x1xnxm 张量 input
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    # 循环遍历不同的池化大小和除数组合
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    # 使用 F.avg_pool2d 对 input 的子张量进行池化，指定池化大小和除数
                    actual = actual.view(1, actual.numel())
                    # 将池化后的张量展平成一维
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    # 调用 _sum_pool2d 方法计算预期的池化结果，并除以给定的除数
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)
                    # 断言池化后的实际结果与预期结果相等，允许的相对和绝对误差分别为 0 和 1e-5

    def test_avg_pool2d_ceil_mode(self):
        # 回归测试 gh-36977
        x = 10 * torch.randn((1, 16, 4, 4))
        # 创建一个形状为 (1, 16, 4, 4) 的随机张量 x
        y = torch.nn.functional.avg_pool2d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        # 使用 torch.nn.functional.avg_pool2d 对 x 进行池化，启用 ceil_mode 和 count_include_pad，指定 kernel_size、padding 和 stride
        self.assertFalse(torch.isnan(y).any())
        # 断言池化后的张量 y 中不存在 NaN 值
        y = torch.nn.functional.avg_pool2d(
            x.to('mps'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        # 将 x 转换为 'mps' 后再次进行池化
        self.assertFalse(torch.isnan(y).any())
        # 断言池化后的张量 y 中不存在 NaN 值
    def test_triu_inf(self, device="mps", dtype=torch.float):
        # 循环遍历不同的对角线位置参数
        for diag in [-1, 0, 1]:
            # 创建一个形状为 (3, 6, 6) 的全 `-inf` 的张量
            mask = torch.full((3, 6, 6), float("-inf"))
            # 克隆并分离出一个新的张量，转移到 'mps' 设备上
            mask_mps = mask.clone().detach().to('mps')
            # 对原始张量和 'mps' 张量执行 torch.triu 操作
            cpu_ref = torch.triu(mask, diagonal=diag)
            mps_out = torch.triu(mask_mps, diagonal=diag)
            # 断言两个结果张量是否相等
            self.assertEqual(cpu_ref, mps_out)

    def test_exp1(self, device="mps", dtype=torch.float):
        # 创建一个输入张量，并对其进行指数运算
        input = torch.tensor([-0.1, 1.0, -0.9, 0.1], device=device, dtype=dtype)
        output = torch.exp(input)
        # 将输入张量在 CPU 上再次进行指数运算，用于对比
        output_cpu = torch.exp(input.cpu())
        # 断言两个结果张量是否几乎相等，使用指定的绝对误差和相对误差容忍度
        # 如果在 MPS 设备上使用 exponentWithTensor:M1 运算，可能会有详细的失败信息
        self.assertEqual(output, output_cpu, atol=1e-8, rtol=1e-8)

    def test_exp_strided_output(self):
        # 创建一个形状为 (256, 10) 的 MPS 设备上的随机张量
        x = torch.rand((256, 10), device='mps')
        # 将 MPS 设备上的张量转移到 CPU，并对其进行转置
        x_cpu = x.to("cpu")
        x = x.permute(1, 0)
        x_cpu = x_cpu.permute(1, 0)
        # 对两个转置后的张量分别进行指数运算
        res = x.exp()
        res_cpu = x_cpu.exp()
        # 断言两个结果张量是否相等
        self.assertEqual(res, res_cpu)

    def _testLeakyRelu(self, np_features, negative_slope, device):
        # 从 numpy 数组创建张量，并设置 requires_grad
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        # 创建一个 LeakyReLU 激活函数对象
        relu_op = torch.nn.LeakyReLU(negative_slope)
        # 分别对 CPU 和 MPS 设备上的输入张量应用 LeakyReLU
        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        # 断言两个结果张量的接近程度
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))
        # 对激活后的张量进行反向传播，检查梯度
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        # 断言 CPU 和 MPS 设备上的梯度是否接近
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersGPU(self):
        # 对于给定的 np.float32 数据类型，调用 _testLeakyRelu 方法
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.1,
                device="mps")

    def test_fill(self):

        def helper(val, shape, dtype):
            # 在 MPS 设备上创建一个全零张量，并填充为指定的值
            tensor = torch.zeros(shape, device='mps', dtype=dtype)
            tensor_mps = tensor.fill_(val)
            # 在 CPU 上创建一个全零张量，并填充为指定的值
            tensor_0 = torch.zeros(shape, device='cpu', dtype=dtype)
            tensor_cpu = tensor_0.fill_(val)
            # 断言 MPS 设备上填充后的张量与 CPU 上的填充后的张量是否相等
            self.assertEqual(tensor_mps, tensor_cpu)

        # 使用不同的值、形状和数据类型调用 helper 函数进行测试
        helper(0, [1024], torch.float32)
        helper(0.2, [2, 3], torch.float32)
        helper(0.2 + 0.5j, [2, 3], torch.complex64)
    # 测试函数，验证在指定设备上填充张量的偏移量是否正确
    def test_fill_storage_offset(self):
        # 定义张量的形状
        shape = [2, 10]
        # 定义填充值
        val = 0.2
        # 创建一个在 "mps" 设备上的全一张量
        tensor = torch.ones(shape, device="mps")
        # 填充张量的偏移为 [1] 的切片，并将其填充为指定的值
        tensor_mps = tensor[:][1].fill_(val)
        # 创建一个在 "cpu" 设备上的全一张量
        tensor_0 = torch.ones(shape, device="cpu")
        # 填充张量的偏移为 [1] 的切片，并将其填充为指定的值
        tensor_cpu = tensor_0[:][1].fill_(val)

        # 断言两个张量在值上是否相等
        self.assertEqual(tensor_mps, tensor_cpu)
        # 断言两个张量在值上是否相等
        self.assertEqual(tensor, tensor_0)

        # 重新定义张量的形状
        shape = [1, 10]
        # 定义填充值为 0.0
        val = 0.0
        # 创建一个在 "mps" 设备上的全一张量
        tensor = torch.ones(shape, device="mps")
        # 创建一个在 "mps" 设备上的指定值张量
        val_tensor_mps = torch.tensor(val, device="mps")
        # 填充张量的偏移为 [:, 9] 的切片，并将其填充为指定的值张量
        tensor_mps = tensor[:, 9].fill_(val_tensor_mps)
        # 修复问题 https://github.com/pytorch/pytorch/issues/114692 的回归测试
        # 填充张量的偏移为 [:, 5] 的切片，并将其填充为指定的值张量
        tensor[:, 5].fill_(val_tensor_mps)
        # 创建一个在 "cpu" 设备上的全一张量
        tensor_0 = torch.ones(shape, device="cpu")
        # 创建一个在 "cpu" 设备上的指定值张量
        val_tensor_cpu = torch.tensor(val, device="cpu")
        # 填充张量的偏移为 [:, 9] 的切片，并将其填充为指定的值张量
        tensor_cpu = tensor_0[:, 9].fill_(val_tensor_cpu)
        # 填充张量的偏移为 [:, 5] 的切片，并将其填充为指定的值张量
        tensor_0[:, 5].fill_(val_tensor_cpu)

        # 断言两个张量在值上是否相等（转换至 "cpu" 设备后）
        self.assertEqual(tensor_mps.to(device="cpu"), tensor_cpu)
        # 断言两个张量在值上是否相等（转换至 "cpu" 设备后）
        self.assertEqual(tensor.to(device="cpu"), tensor_0)

    # 测试函数，验证在指定设备上计算大批量数据的欧氏距离是否正确
    def test_cdist_large(self, device="mps"):
        # 遍历计算模式列表
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 在指定设备上生成随机张量 x 和 y
            x = torch.randn(100, 10, device=device)
            y = torch.randn(100, 10, device=device)
            # 使用指定的计算模式计算张量 x 和 y 的欧氏距离
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用暴力方法计算张量 x 和 y 的欧氏距离作为期望值
            expected = self._brute_cdist(x, y, p=2)
            # 断言实际计算结果与期望值是否相等
            self.assertEqual(expected, actual)

    # 测试函数，验证在指定设备上计算大批量数据的批次欧氏距离是否正确
    def test_cdist_large_batch(self, device="mps"):
        # 遍历计算模式列表
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 在指定设备上生成随机张量 x 和 y，包含批次维度
            x = torch.randn(4, 3, 100, 10, device=device)
            y = torch.randn(4, 3, 100, 10, device=device)
            # 使用指定的计算模式计算批次张量 x 和 y 的欧氏距离
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用暴力方法计算批次张量 x 和 y 的欧氏距离作为期望值
            expected = self._brute_cdist(x, y, p=2)
            # 断言实际计算结果与期望值是否相等
            self.assertEqual(expected, actual)
    # 定义一个测试方法，用于测试非连续内存情况下的距离计算函数
    def test_cdist_non_contiguous(self, device="mps"):
        # 对两种计算模式进行循环测试
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 创建随机张量 x 和 y，形状分别为 (5, 7) 和 (5, 3)，并转置非连续内存
            x = torch.randn(5, 7, device=device).mT
            y = torch.randn(5, 3, device=device).mT
            # 使用 torch.cdist 计算 x 和 y 的距离，期望使用给定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用备选的蛮力方法计算期望的距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 和 y 非连续内存
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(expected, actual)

            # 创建随机张量 x 和 y，形状分别为 (7, 5) 和 (5, 3)，并转置非连续内存
            x = torch.randn(7, 5, device=device)
            y = torch.randn(5, 3, device=device).t()
            # 使用 torch.cdist 计算 x 和 y 的距离，期望使用给定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用备选的蛮力方法计算期望的距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 连续内存，y 非连续内存
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(expected, actual)

            # 创建随机张量 x 和 y，形状分别为 (5, 7) 和 (3, 5)，并转置非连续内存
            x = torch.randn(5, 7, device=device).t()
            y = torch.randn(3, 5, device=device)
            # 使用 torch.cdist 计算 x 和 y 的距离，期望使用给定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用备选的蛮力方法计算期望的距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 非连续内存，y 连续内存
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(expected, actual)

    # 定义一个测试方法，用于测试批量非连续内存情况下的距离计算函数
    def test_cdist_non_contiguous_batch(self, device="mps"):
        # 对两种计算模式进行循环测试
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            # 创建随机张量 x 和 y，形状分别为 (4, 3, 2, 5, 7) 和 (4, 3, 2, 5, 3)，并转置非连续内存
            x = torch.randn(4, 3, 2, 5, 7, device=device).mT
            y = torch.randn(4, 3, 2, 5, 3, device=device).mT
            # 使用 torch.cdist 计算 x 和 y 的距离，期望使用给定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用备选的蛮力方法计算期望的距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 和 y 非连续内存
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(expected, actual)

            # 创建随机张量 x 和 y，形状分别为 (7, 2, 7, 5) 和 (7, 2, 5, 3)，并转置非连续内存
            x = torch.randn(7, 2, 7, 5, device=device)
            y = torch.randn(7, 2, 5, 3, device=device).mT
            # 使用 torch.cdist 计算 x 和 y 的距离，期望使用给定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用备选的蛮力方法计算期望的距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 连续内存，y 非连续内存
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(expected, actual)

            # 创建随机张量 x 和 y，形状分别为 (4, 5, 7) 和 (4, 3, 5)，并转置非连续内存
            x = torch.randn(4, 5, 7, device=device).mT
            y = torch.randn(4, 3, 5, device=device)
            # 使用 torch.cdist 计算 x 和 y 的距离，期望使用给定的计算模式 cm
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            # 使用备选的蛮力方法计算期望的距离
            expected = self._brute_cdist(x, y, p=2)
            # 断言 x 非连续内存，y 连续内存
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            # 断言实际计算结果与预期结果相等
            self.assertEqual(expected, actual)
    # 定义测试函数，用于测试大型欧氏距离计算
    def test_cdist_euclidean_large(self, device="mps"):
        # 定义内部函数，用于测试大型欧氏距离计算
        def _test_euclidean_large_cdist(sizex, sizey=None):
            # 如果未指定 sizey，则与 sizex 相同
            if sizey is None:
                sizey = sizex
            # 生成随机张量 x 和 y，设备为指定设备，数据类型为浮点数
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # 避免出现极端值
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            # 计算 x 和 y 之间的欧氏距离
            dist = torch.cdist(x, y, p=2)
            # 计算总距离作为损失
            loss = dist.sum()
            # 执行反向传播以验证对于大型矩阵是有效的
            loss.backward()

        # 调用内部函数，测试指定大小的输入
        _test_euclidean_large_cdist((2000, 5))

    # 定义测试函数，检查在相同输入情况下的 cdist 梯度计算问题
    def test_cdist_same_inputs(self, device="mps"):
        sizex = (1, 27, 32)
        # 遍历不同的 p 值进行测试
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            # 生成随机张量 x，并生成随机梯度 dist_grad
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            # y 是 x 的克隆
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            # 计算 x 和 y 之间的距离
            d = torch.cdist(x, y)
            # 执行反向传播，使用 dist_grad 作为输入梯度
            d.backward(dist_grad)
            # 断言反向传播结果不包含无效值如 nan 或 inf
            assert torch.isfinite(x.grad).all()

    # 定义内部函数，使用 torch.norm 计算欧氏距离
    def _brute_cdist(self, x, y, p=2):
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        # 如果 r1 或 r2 为 0，则返回一个空的张量
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        # 使用 torch.norm 计算 x 和 y 之间的距离
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    # 定义测试函数，检查 cdist 函数对于不同输入和参数的行为是否符合预期
    def test_cdist_norm(self, device="mps"):
        # 分别遍历 r1, m, r2 和 p 的组合进行测试
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 1, 1.5, 2.5, float('inf')]:
                        # 生成随机张量 x 和 y，设备为指定设备
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        if p == 2:
                            # 对于 p=2 的情况，使用两种计算模式进行比较
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                # 计算实际结果和预期结果，并断言它们相等
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            # 对于其他 p 值，计算实际结果和预期结果，并断言它们相等
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)
    # 定义测试函数，用于测试批量计算距离的方法
    def test_cdist_norm_batch(self, device="mps"):
        # 循环遍历不同的参数组合
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 3, 1.5, 2.5, float('inf')]:
                        # 生成随机张量 x 和 y，形状为 (2, 3, 6, r1, m)，并指定设备
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        if p == 2:
                            # 当 p 等于 2 时，分别测试两种计算模式下的结果
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                # 计算使用指定模式的距离
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                # 使用暴力方法计算期望的距离
                                expected = self._brute_cdist(x, y, p=2)
                                # 断言实际计算结果与期望结果相等
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            # 对于其他 p 值，计算距离并比较结果
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    # 定义测试函数，用于测试矩阵乘法操作
    def test_mm(self):
        # 创建形状为 (5, 6) 的全一张量 B，并指定设备为 "mps"
        B = torch.ones(5, 6).to("mps")
        # 创建形状为 (6, 5) 的全一张量 C，并指定设备为 "mps"
        C = torch.ones(6, 5).to("mps")
        # 执行矩阵乘法操作，并将结果移到 CPU 上
        D = torch.mm(B, C).cpu()
        # 断言 D 与形状为 (5, 5)，且所有元素为 6.0 的张量相等
        torch.testing.assert_close(D, torch.full((5, 5), 6.0))
    def test_linalg_cross(self):
        # 定义辅助函数，参数为数据类型 dtype
        def helper(dtype):
            # 设备选择为 "mps"
            device = "mps"
            # 根据数据类型不同，生成不同的张量 x 和 y
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
            else:
                x = torch.rand(100, 3, 100, dtype=dtype, device=device)
                y = torch.rand(100, 3, 100, dtype=dtype, device=device)
            # 将 x 和 y 转移到 CPU 上
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            # 计算在指定维度 dim=1 上的叉积并保存结果到 res1
            res1 = torch.linalg.cross(x, y, dim=1)
            # 创建一个空张量 res2，指定数据类型和设备
            res2 = torch.tensor((), dtype=dtype, device=device)
            # 在 CPU 上计算相同维度的叉积并保存到 res1_cpu
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            # 创建一个空张量 res2_cpu，在 CPU 上使用
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            # 将计算结果直接保存到 res2
            torch.linalg.cross(x, y, dim=1, out=res2)
            # 在 CPU 上进行同样操作，结果保存到 res2_cpu
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            # 使用断言检查 res1 和 res2 的相等性
            self.assertEqual(res1, res2)
            # 使用断言检查 res1 和 res1_cpu 的相等性
            self.assertEqual(res1, res1_cpu)
            # 使用断言检查 res2 和 res2_cpu 的相等性
            self.assertEqual(res2, res2_cpu)

            # 测试广播输入的情况
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (1, 3, 2), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (4, 3, 1), dtype=dtype, device=device)
            else:
                x = torch.rand(1, 3, 2, dtype=dtype, device=device)
                y = torch.rand(4, 3, 1, dtype=dtype, device=device)
            # 将 x 和 y 转移到 CPU 上
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            # 计算在指定维度 dim=1 上的叉积并保存结果到 res1
            res1 = torch.linalg.cross(x, y, dim=1)
            # 创建一个空张量 res2，指定数据类型和设备
            res2 = torch.tensor((), dtype=dtype, device=device)
            # 在 CPU 上计算相同维度的叉积并保存到 res1_cpu
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            # 创建一个空张量 res2_cpu，在 CPU 上使用
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            # 将计算结果直接保存到 res2
            torch.linalg.cross(x, y, dim=1, out=res2)
            # 在 CPU 上进行同样操作，结果保存到 res2_cpu
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            # 使用断言检查 res1 和 res2 的相等性
            self.assertEqual(res1, res2)
            # 使用断言检查 res1 和 res1_cpu 的相等性
            self.assertEqual(res1, res1_cpu)
            # 使用断言检查 res2 和 res2_cpu 的相等性
            self.assertEqual(res2, res2_cpu)
        
        # 对数据类型列表 [torch.int32, torch.int64, torch.float32] 调用辅助函数 helper
        [helper(dtype) for dtype in [torch.int32, torch.int64, torch.float32]]

    def test_cross(self):
        # 创建两个随机张量 a 和 b，指定设备为 "mps"
        a = torch.randn(4, 3, device="mps")
        b = torch.randn(4, 3, device="mps")
        # 将 a 和 b 转移到 CPU 上
        a_cpu = a.to("cpu")
        b_cpu = b.to("cpu")
        # 计算在指定维度 dim=1 上的叉积并保存结果到 res
        res = torch.cross(a, b, dim=1)
        # 在 CPU 上计算相同维度的叉积并保存到 res_cpu
        res_cpu = torch.cross(a_cpu, b_cpu, dim=1)
        # 使用断言检查 res 和 res_cpu 的相等性
        self.assertEqual(res, res_cpu)

    def test_addmm(self):
        # 创建全为 1 的张量 A，B 和 C，设备选择为 "mps"
        A = torch.ones(5, 5).to("mps")
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        # 计算矩阵乘法 A + B @ C 并将结果转移到 CPU 上
        D = torch.addmm(A, B, C).to("cpu")
        # 使用断言检查 D 是否接近全为 7 的张量
        torch.testing.assert_close(D, torch.full((5, 5), 7.0))
    # 测试批矩阵乘积函数 torch.bmm 的功能
    def test_bmm(self):
        # 创建两个随机张量作为输入的 CPU 批
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        # 将 CPU 批张量分别复制并转换为 "mps" 设备上的张量
        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        # 在 CPU 上执行批矩阵乘积
        output_cpu = torch.bmm(batch1_cpu, batch2_cpu)
        # 在 "mps" 设备上执行批矩阵乘积
        output_mps = torch.bmm(batch1_mps, batch2_mps)

        # 断言两个输出张量在值上完全一致
        self.assertEqual(output_cpu, output_mps)
        # 断言两个输出张量的形状完全一致
        self.assertEqual(output_cpu.size(), output_mps.size())

    # 测试 torch.addr 函数的功能
    def test_addr(self):
        # 创建三个全为 1 的 "mps" 设备上的张量
        A = torch.ones(5, 10).to("mps")
        B = torch.ones(5).to("mps")
        C = torch.ones(10).to("mps")
        # 使用 torch.addr 在 "mps" 设备上执行张量加权外积，并将结果转移到 CPU 上
        D = torch.addr(A, B, C).to("cpu")
        # 断言计算结果与全为 2.0 的目标张量 D 在值上非常接近
        torch.testing.assert_close(D, torch.full((5, 10), 2.0))

    # 测试计算矩阵的迹函数 torch.trace 的功能
    def test_trace(self):
        # 创建一个随机矩阵 M_cpu
        M_cpu = torch.randn(3, 3)
        # 将 M_cpu 复制并转换为 "mps" 设备上的张量
        M_mps = M_cpu.detach().clone().to("mps")

        # 计算 CPU 上矩阵的迹
        output_cpu = torch.trace(M_cpu)
        # 计算 "mps" 设备上矩阵的迹
        output_mps = torch.trace(M_mps)

        # 断言两个迹的计算结果在值上完全一致
        self.assertEqual(output_cpu, output_mps)
        # 断言两个迹的计算结果的形状完全一致
        self.assertEqual(output_cpu.size(), output_mps.size())

    # 测试 torch.addbmm 函数的功能
    def test_addbmm(self):
        # 创建一个随机矩阵 M_cpu 和两个随机的 CPU 批
        M_cpu = torch.randn(3, 5)
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        # 将 M_cpu 和两个批张量分别复制并转换为 "mps" 设备上的张量
        M_mps = M_cpu.detach().clone().to("mps")
        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        # 在 CPU 上执行 addbmm 操作
        output_cpu = torch.addbmm(M_cpu, batch1_cpu, batch2_cpu)
        # 在 "mps" 设备上执行 addbmm 操作
        output_mps = torch.addbmm(M_mps, batch1_mps, batch2_mps)

        # 断言两个操作的输出在值上完全一致
        self.assertEqual(output_cpu, output_mps)
        # 断言两个操作的输出的形状完全一致
        self.assertEqual(output_cpu.size(), output_mps.size())

    # 测试 torch.baddbmm 函数的功能
    def test_baddbmm(self):
        # 定义一个帮助函数，用于测试不同形状输入的 baddbmm 功能
        def helper(input_shape, batch1_shape, batch2_shape):
            # 创建一个随机输入矩阵 M_cpu 和两个随机的 CPU 批
            M_cpu = torch.randn(input_shape)
            batch1_cpu = torch.randn(batch1_shape)
            batch2_cpu = torch.randn(batch2_shape)
            alpha = 1.2
            beta = 0.8

            # 将 M_cpu 和两个批张量分别复制并转换为 "mps" 设备上的张量
            M_mps = M_cpu.detach().clone().to("mps")
            batch1_mps = batch1_cpu.detach().clone().to("mps")
            batch2_mps = batch2_cpu.detach().clone().to("mps")

            # 在 CPU 上执行 baddbmm 操作
            output_cpu = torch.baddbmm(M_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
            # 在 "mps" 设备上执行 baddbmm 操作
            output_mps = torch.baddbmm(M_mps, batch1_mps, batch2_mps, beta=beta, alpha=alpha)

            # 断言两个操作的输出在值上完全一致
            self.assertEqual(output_cpu, output_mps)
            # 断言两个操作的输出的形状完全一致
            self.assertEqual(output_cpu.size(), output_mps.size())

        # 使用 helper 函数测试不同形状的输入
        helper(input_shape=(3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(10, 3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(1, 77, 77), batch1_shape=(8, 77, 64), batch2_shape=(8, 64, 77))

    # 测试在本地标量转换时的功能
    def test_local_scalar_dense_mps(self):
        # 创建一个随机 CPU 上的标量张量 x_cpu
        x_cpu = torch.randn(1)
        # 将 x_cpu 转换为 "mps" 设备上的标量张量 y_mps
        y_mps = x_cpu.to("mps")
        # 断言 CPU 上的标量值与 "mps" 设备上的标量值非常接近
        torch.testing.assert_close(x_cpu.item(), y_mps.item())
    def test_linear_1d_weight(self):
        # 设置设备为CPU
        device = 'cpu'
        # 生成一个随机的张量，并将其移到指定设备
        projected = torch.rand([8]).to(device)
        # 生成一个随机的4维张量，并移到指定设备
        x = torch.rand([1, 2, 2, 8]).to(device)
        # 将x张量转换为'mps'设备上的张量
        x_mps = x.to('mps')
        # 将projected张量转换为'mps'设备上的张量
        projected_mps = projected.to('mps')
        # 使用torch.nn.functional中的线性函数计算线性变换
        linear = F.linear(x, projected)
        # 在'mps'设备上使用线性函数计算线性变换
        linear_mps = F.linear(x_mps, projected_mps)

        # 断言两个线性变换结果相等
        self.assertEqual(linear, linear_mps)

        # 重新生成一个形状为[1, 8]的随机张量，并移到指定设备
        projected = torch.rand([1, 8]).to(device)
        # 重新生成一个随机的4维张量，并移到指定设备
        x = torch.rand([1, 2, 2, 8]).to(device)
        # 将x张量转换为'mps'设备上的张量
        x_mps = x.to('mps')
        # 将projected张量转换为'mps'设备上的张量
        projected_mps = projected.to('mps')
        # 使用torch.nn.functional中的线性函数计算线性变换
        linear = F.linear(x, projected)
        # 在'mps'设备上使用线性函数计算线性变换
        linear_mps = F.linear(x_mps, projected_mps)

        # 断言两个线性变换结果相等
        self.assertEqual(linear, linear_mps)

    def test_linear_bias(self):
        def helper(bias_shape):
            # 设置设备为CPU
            device = "cpu"
            # 生成一个形状为[2, 2, 2, 64]的随机张量，并移到指定设备
            x = torch.randn(2, 2, 2, 64, device=device)
            # 创建一个具有64输入和4输出的线性层，并移到指定设备
            linear = torch.nn.Linear(64, 4, device=device)
            # 设置线性层的偏置为随机生成的参数张量
            linear.bias = torch.nn.Parameter(torch.randn(bias_shape, dtype=torch.float32, device=device))
            # 对输入x进行线性变换
            y = linear(x)
            # 将设备更改为'mps'
            device = "mps"
            # 将x张量转换为'mps'设备上的张量
            x_mps = x.to(device)
            # 将线性层移动到'mps'设备上
            linear.to(device)
            # 对'mps'设备上的输入x进行线性变换
            y_mps = linear(x_mps)
            # 断言两个线性变换结果相等
            self.assertEqual(y, y_mps)

        # 调用helper函数，测试不同形状的偏置
        helper(())
        helper((2, 4))

    def test_linear_errors(self):
        # 混合使用CPU和MPS张量
        size = (3, 3)

        # 不支持的数据类型
        with self.assertRaisesRegex(RuntimeError, "does not support linear for non-float weights"):
            # 在'mps'设备上使用非浮点权重进行线性变换
            torch.nn.functional.linear(torch.rand(size, device='mps'),
                                       torch.randint(-10, 10, size, dtype=torch.int8, device='mps'))

        # 权重张量在错误的设备上
        with self.assertRaisesRegex(RuntimeError, "argument weight is on cpu but expected on mps"):
            # 权重张量在CPU设备上，但期望在MPS设备上
            torch.nn.functional.linear(torch.rand(size, device='mps'),
                                       torch.rand(size, device='cpu'))

        # 输入张量在错误的设备上
        with self.assertRaisesRegex(RuntimeError, "argument input is on cpu but expected on mps"):
            # 输入张量在CPU设备上，但期望在MPS设备上
            torch.nn.functional.linear(torch.rand(size, device='cpu'),
                                       torch.rand(size, device='mps'))

    def test_linear1D(self):
        # 调用_helper函数，测试1维线性层的前向传播
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=False)

    def test_linear1D_backward(self):
        # 调用_helper函数，测试1维线性层的反向传播
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=True)

    def test_linear2D(self):
        # 调用_helper函数，测试2维线性层的前向传播
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=False)

    def test_linear2D_backward(self):
        # 调用_helper函数，测试2维线性层的反向传播
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=True)

    def test_linear2D_no_bias(self):
        # 调用_helper函数，测试2维线性层没有偏置的前向传播
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=False)
    # 调用 _linear_helper 方法，测试无偏置的二维线性层的反向传播
    def test_linear2D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=True)

    # 调用 _linear_helper 方法，测试三维输入的线性层
    def test_linear3D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    # 调用 _linear_helper 方法，测试三维输入的线性层的反向传播
    def test_linear3D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    # 调用 _linear_helper 方法，测试无偏置的三维线性层
    def test_linear3D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    # 调用 _linear_helper 方法，测试无偏置的三维线性层的反向传播
    def test_linear3D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    # 创建张量 low 和 high，用于均匀分布的上下界设置，并设置梯度跟踪
    low = torch.zeros(5, 5, requires_grad=True)
    high = (torch.ones(5, 5) * 3).requires_grad_()
    low_1d = torch.zeros(1, requires_grad=True)
    high_1d = (torch.ones(1) * 3).requires_grad_()

    # 断言均匀分布采样结果的形状是否符合预期
    self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
    self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
    self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
    self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
    self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

    # 当采样值超出范围时，检查对数概率的计算结果
    uniform = Uniform(low_1d, high_1d, validate_args=False)
    above_high = torch.tensor([4.0])
    below_low = torch.tensor([-1.0])
    self.assertEqual(uniform.log_prob(above_high).item(), -inf)
    self.assertEqual(uniform.log_prob(below_low).item(), -inf)

    # 当采样值超出范围时，检查累积分布函数的计算结果
    self.assertEqual(uniform.cdf(below_low).item(), 0)
    self.assertEqual(uniform.cdf(above_high).item(), 1)

    # 备份当前的随机数生成器状态，并生成新的随机数
    state = torch.get_rng_state()
    rand = low.new(low.size()).uniform_()
    torch.set_rng_state(state)

    # 使用均匀分布采样并计算反向传播
    u = Uniform(low, high).rsample()
    u.backward(torch.ones_like(u))

    # 断言梯度计算是否正确
    self.assertEqual(low.grad, 1 - rand)
    self.assertEqual(high.grad, rand)

    # 清空梯度，为下一次使用做准备
    low.grad.zero_()
    high.grad.zero_()
    # 测试随机排列函数 randperm
    def test_randperm(self, device="mps"):
        # 初始化 RNG 设备为 None
        rng_device = None
        # 遍历不同的 n 值
        for n in (5, 100, 50000, 100000):
            # 遍历不同的数据类型
            for dtype in (torch.long, torch.half, torch.float):
                # 跳过当 n > 2049 且数据类型为 torch.half 的情况，避免异常
                if n > 2049 and dtype == torch.half:
                    continue
                # 跳过当 n > 256 且数据类型为 torch.bfloat16 的情况
                if n > 256 and dtype == torch.bfloat16:
                    continue
                # 使用指定的 RNG 设备分叉随机数生成器
                with torch.random.fork_rng(devices=rng_device):
                    # 使用 randperm 函数生成随机排列结果 res1
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                # 使用指定的 dtype 和 device 在已分配的空张量上生成随机排列结果 res2
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                # 断言 res1 在 CPU 上排序后的值与 torch.arange(n, device=device) 相等
                self.assertEqual(res1.cpu().sort().values.long(), torch.arange(n, device=device))

        # 默认类型为 long
        for n in (100, 10000):
            # 断言使用默认设备生成的随机排列结果的数据类型为 long
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm 的元素数为 0 时生成空张量
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)  # 此处应为 torch.long
        torch.randperm(0, out=res2)
        # 断言生成的 res1 和 res2 张量元素数为 0
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # 测试非连续张量
        for n in (4, 5, 6, 10, 20):
            # 创建非连续张量
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            # 断言非连续张量确实非连续
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            # 使用指定的 RNG 设备分叉随机数生成器
            with torch.random.fork_rng(devices=rng_device):
                # 使用 randperm 函数生成随机排列结果 res
                res = torch.randperm(n, dtype=torch.long, device=device)
            # 使用指定的 dtype 和 device 在非连续张量上生成随机排列结果
            torch.randperm(n, out=non_contiguous_tensor)
            # 断言 res 在 CPU 上排序后的值与 torch.arange(n, device=device) 相等
            self.assertEqual(res.cpu().sort().values.long(), torch.arange(n, device=device))

    # 测试前向最大池化 maxpool2d
    def test_adaptive_avg_pool2d_output_size_one(self):
        # 辅助函数定义
        def helper(size, memory_format):
            # 创建随机整数张量 x
            x = torch.randint(1, 10, size, dtype=torch.float, device='mps', requires_grad=True)
            # 根据内存格式设置张量 x
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            # 创建自适应平均池化层
            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            # 执行前向计算
            out = net(x)
            # 计算参考输出 ref_out
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            # 执行反向传播，确保不会崩溃
            out.sum().backward()

            # 断言 out 和 ref_out 相等
            self.assertEqual(out, ref_out)
            # 如果使用了通道最后内存格式，断言 out 是连续的
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                # 断言 out 的步长与通道最后内存格式一致
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                # 否则，断言 out 是连续的
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                # 断言 out 的步长是标准的 [c, 1, 1, 1]
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        # 调用辅助函数 helper 进行测试，使用连续内存格式
        helper((2, 3, 6, 6), torch.contiguous_format)
    # 定义测试方法：测试 masked_scatter 函数
    def test_masked_scatter(self):
        
        # 定义内部辅助函数 helper，用于测试不同形状的输入
        def helper(shape):
            # 在 "mps" 设备上生成随机张量 x_mps
            x_mps = torch.randn(shape, device="mps")
            # 将 x_mps 生成一个与之分离的副本，并移到 CPU 上
            x_cpu = x_mps.detach().clone().cpu()

            # 生成一个与 x_mps 形状相同的随机掩码 mask_mps，设备为 "mps"
            mask_mps = torch.rand(shape, device="mps") < 0.6
            # 将 mask_mps 生成一个与之分离的副本，并移到 CPU 上
            mask_cpu = mask_mps.detach().clone().cpu()

            # 在 "mps" 设备上生成随机张量 y_mps
            y_mps = torch.randn(shape, device="mps")
            # 将 y_mps 生成一个与之分离的副本，并移到 CPU 上
            y_cpu = y_mps.detach().clone().cpu()

            # 使用 masked_scatter_ 方法在 y_mps 上根据 mask_mps 进行部分填充操作
            y_mps.masked_scatter_(mask_mps, x_mps)
            # 使用 masked_scatter_ 方法在 y_cpu 上根据 mask_cpu 进行部分填充操作
            y_cpu.masked_scatter_(mask_cpu, x_cpu)

            # 断言 y_mps 和 y_cpu 的值应该相等
            self.assertEqual(y_mps, y_cpu)
        
        # 对不同的 shape 调用 helper 函数进行测试
        helper([2, 5])
        helper([10, 10])
        helper([5, 10, 3])
        helper([10, 5, 10, 3])
        helper([10, 5, 10, 3, 20])

    # 定义测试方法：测试 masked_fill 函数
    def test_masked_fill(self):
        # 指定设备为 "mps"
        device = "mps"
        # 指定数据类型为 torch.float32
        dtype = torch.float32
        # 指定掩码数据类型为 torch.bool
        mask_dtype = torch.bool

        # 使用警告模块捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # 定义目标张量的长度 num_dest
            num_dest = 10
            # 在指定设备和数据类型下生成全零张量 dst
            dst = torch.zeros(num_dest, dtype=dtype, device=device)
            # 生成指定长度的随机整数掩码 mask，数据类型为 mask_dtype，设备为 device
            mask = torch.randint(2, (num_dest,), dtype=mask_dtype, device=device)
            # 随机生成一个浮点数 val
            val = random.random()
            # 生成一个全零张量 dst2，数据类型为 dtype
            dst2 = torch.zeros(num_dest, dtype=dtype)
            # 将 mask 移到 CPU 上生成一个副本 mask_cpu
            mask_cpu = mask.to("cpu")

            # 使用 masked_fill_ 方法在 dst 上根据 mask 进行填充操作
            dst.masked_fill_(mask, val)
            # 对 dst2 进行循环遍历，根据 mask_cpu 进行条件填充操作
            for i in range(num_dest):
                if mask_cpu[i]:
                    dst2[i] = val
            
            # 断言 dst 转移到 CPU 后与 dst2 相等
            self.assertEqual(dst.to("cpu"), dst2, atol=0, rtol=0)

            # 测试非连续情况
            # 生成一个非连续的多维张量 dst
            dst = ((torch.randn(num_dest, num_dest, num_dest) * 10).to(dtype)).permute((2, 0, 1))
            # 生成一个连续的 dst2 张量
            dst2 = dst.contiguous()
            # 根据数据类型是否为复数，生成相应的掩码 mask
            if dtype.is_complex:
                mask = dst.abs() > 0
            else:
                mask = dst > 0
            
            # 断言 dst 是否为非连续张量
            self.assertFalse(dst.is_contiguous())
            # 断言 dst2 是否为连续张量
            self.assertTrue(dst2.is_contiguous())
            
            # 使用 masked_fill_ 方法在 dst 上根据 mask 进行填充操作
            dst.masked_fill_(mask.to(mask_dtype), val)
            # 使用 masked_fill_ 方法在 dst2 上根据 mask 进行填充操作
            dst2.masked_fill_(mask.to(mask_dtype), val)
            # 断言 dst 和 dst2 的值应该相等
            self.assertEqual(dst, dst2, atol=0, rtol=0)

            # 如果 mask_dtype 为 torch.uint8，则断言警告信息长度为 3
            if mask_dtype == torch.uint8:
                self.assertEqual(len(w), 3)

                # 断言警告信息是否符合预期
                warn = 'masked_fill_ received a mask with dtype torch.uint8,'
                for wi in w:
                    self.assertEqual(str(wi.message)[0:52], str(warn))
            else:
                # 否则断言警告信息长度为 0
                self.assertEqual(len(w), 0)

    # 定义测试方法：测试 nhwc_operation 函数
    def test_nhwc_operation(self):
        # 定义内部辅助函数 helper，用于测试指定形状和是否通道优先的张量操作
        def helper(shape, channels_last=False):
            # 设置随机种子
            import numpy as np
            np.random.seed(332)
            # 生成指定形状的随机数组 arr，数值范围在 128 到 256 之间
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            # 在 CPU 上生成张量 cpu_x，数据类型为 torch.float，需要梯度计算
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            # 如果 channels_last 为 True，则将 cpu_x 转换为通道优先存储格式
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                # 保留梯度信息
                cpu_x.retain_grad()
            # 将 cpu_x 生成一个分离副本，并移动到 "mps" 设备上，同时要求梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 断言 x 和 cpu_x 的值应该相等
            self.assertEqual(x, cpu_x)

        # 调用 helper 函数进行测试，指定形状为 (2, 2, 2, 2)，并设置 channels_last=True
        helper((2, 2, 2, 2), True)

    # 测试前向批量归一化
    # 定义测试批量归一化反向传播的函数
    def test_batch_norm_backward(self):
        # 创建一个形状为 (1, 8, 4, 4) 的随机张量输入，要求计算梯度，使用 MPS 设备
        inputs = torch.rand(1, 8, 4, 4, device="mps", requires_grad=True)
        # 在 MPS 设备上创建一个包含 8 个通道的 BatchNorm2d 层
        x = torch.nn.BatchNorm2d(8).to("mps")
        # 在 MPS 设备上创建另一个包含 8 个通道的 BatchNorm2d 层
        y = torch.nn.BatchNorm2d(8).to("mps")
        # 设置 y 的权重不需要计算梯度
        y.weight.requires_grad = False
        # 设置 y 的偏置不需要计算梯度
        y.bias.requires_grad = False
        # 对输入 inputs 应用 BatchNorm2d 层，并将结果保存在 outputs 中
        outputs = y(x(inputs))
        # 进行输出的所有元素之和的反向传播，用于检查之前存在的崩溃问题
        outputs.sum().backward()

    # 定义测试层归一化反向传播的函数
    def test_layer_norm_backward(self):
        # 创建一个形状为 (4, 4) 的随机张量输入，要求计算梯度，使用 MPS 设备
        inputs = torch.rand(4, 4, device="mps", requires_grad=True)
        # 在 MPS 设备上创建一个包含 4 个特征的 LayerNorm 层
        x = torch.nn.LayerNorm(4).to("mps")
        # 在 MPS 设备上创建另一个包含 4 个特征的 LayerNorm 层
        y = torch.nn.LayerNorm(4).to("mps")
        # 设置 y 的权重不需要计算梯度
        y.weight.requires_grad = False
        # 设置 y 的偏置不需要计算梯度
        y.bias.requires_grad = False
        # 对输入 inputs 应用 LayerNorm 层，并将结果保存在 outputs 中
        outputs = y(x(inputs))
        # 进行输出的所有元素之和的反向传播，用于检查之前存在的崩溃问题
        outputs.sum().backward()

    # 定义测试 norm 函数的函数
    def test_norm(self):
        # 创建一个包含 [0, 1, 2, ..., 8] 的张量 a，数据类型为 float，在 MPS 设备上
        a = torch.arange(9, dtype=torch.float, device="mps") - 4
        # 将张量 a 重新形状为 (3, 3) 的张量 b
        b = a.reshape((3, 3))

        # 创建一个包含 [0, 1, 2, ..., 8] 的张量 a_cpu，数据类型为 float，在 CPU 设备上
        a_cpu = torch.arange(9, dtype=torch.float, device="cpu") - 4
        # 将张量 a_cpu 重新形状为 (3, 3) 的张量 b_cpu
        b_cpu = a_cpu.reshape((3, 3))

        # 计算张量 a 的默认 L2 范数
        res = torch.norm(a)
        # 计算张量 a_cpu 的默认 L2 范数
        res_cpu = torch.norm(a_cpu)
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 计算张量 b 的默认 L2 范数
        res = torch.norm(b)
        # 计算张量 b_cpu 的默认 L2 范数
        res_cpu = torch.norm(b_cpu)
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 计算张量 a 的 L-inf 范数
        res = torch.norm(a, float('inf'))
        # 计算张量 a_cpu 的 L-inf 范数
        res_cpu = torch.norm(a_cpu, float('inf'))
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 计算张量 b 的 L-inf 范数
        res = torch.norm(b, float('inf'))
        # 计算张量 b_cpu 的 L-inf 范数
        res_cpu = torch.norm(b_cpu, float('inf'))
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 创建一个形状为 (2, 3) 的张量 c，在 MPS 设备上
        c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float, device="mps")
        # 创建一个形状为 (2, 3) 的张量 c_cpu，在 CPU 设备上
        c_cpu = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float, device="cpu")

        # 计算张量 c 沿第 0 维的 L2 范数
        res = torch.norm(c, dim=0)
        # 计算张量 c_cpu 沿第 0 维的 L2 范数
        res_cpu = torch.norm(c_cpu, dim=0)
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 计算张量 c 沿第 1 维的 L2 范数
        res = torch.norm(c, dim=1)
        # 计算张量 c_cpu 沿第 1 维的 L2 范数
        res_cpu = torch.norm(c_cpu, dim=1)
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 计算张量 c 沿第 1 维的 L1 范数
        res = torch.norm(c, p=1, dim=1)
        # 计算张量 c_cpu 沿第 1 维的 L1 范数
        res_cpu = torch.norm(c_cpu, p=1, dim=1)
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 创建一个形状为 (2, 2, 2) 的张量 d，在 MPS 设备上
        d = torch.arange(8, dtype=torch.float, device="mps").reshape(2, 2, 2)
        # 创建一个形状为 (2, 2, 2) 的张量 d_cpu，在 CPU 设备上
        d_cpu = torch.arange(8, dtype=torch.float, device="cpu").reshape(2, 2, 2)

        # 计算张量 d 沿 (1, 2) 维的 L2 范数
        res = torch.norm(d, dim=(1, 2))
        # 计算张量 d_cpu 沿 (1, 2) 维的 L2 范数
        res_cpu = torch.norm(d_cpu, dim=(1, 2))
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)

        # 计算张量 d 的第 0 个切片的 L2 范数，和第 1 个切片的 L2 范数
        res = torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        # 计算张量 d_cpu 的第 0 个切片的 L2 范数，和第 1 个切片的 L2 范数
        res_cpu = torch.norm(d_cpu[0, :, :]), torch.norm(d_cpu[1, :, :])
        # 断言两个结果相等
        self.assertEqual(res, res_cpu)
    # 定义测试函数，用于测试向量范数计算
    def test_linalg_vector_norm(self):
        # 创建一个张量 x_mps，包含元素 [0, 0, 0, 2, 3]，数据类型为浮点数，在设备 "mps" 上
        x_mps = torch.tensor([0, 0, 0, 2, 3], dtype=torch.float, device="mps")
        # 对 x_mps 进行去除梯度并克隆到 CPU 设备的操作
        x_cpu = x_mps.detach().clone().cpu()

        # 计算 x_mps 和 x_cpu 的 L0 范数（向量中非零元素的个数）
        res_mps = torch.linalg.vector_norm(x_mps, ord=0)
        res_cpu = torch.linalg.vector_norm(x_cpu, ord=0)
        # 断言两个 L0 范数的结果相等
        self.assertEqual(res_mps, res_cpu)

        # 创建一个张量 a_mps，包含元素从 -4 到 22，数据类型为浮点数，在设备 "mps" 上
        a_mps = torch.arange(27, dtype=torch.float, device="mps") - 4
        # 创建一个张量 a_cpu，包含元素从 -4 到 22，数据类型为浮点数，在 CPU 上
        a_cpu = torch.arange(27, dtype=torch.float, device="cpu") - 4

        # 将 a_mps 和 a_cpu 分别重塑为 3x3x3 的张量
        B_mps = a_mps.reshape(3, 3, 3)
        B_cpu = a_cpu.reshape(3, 3, 3)

        # 计算 a_mps 和 a_cpu 的 L3.5 范数
        res_mps = torch.linalg.vector_norm(a_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(a_cpu, ord=3.5)
        # 断言两个 L3.5 范数的结果相等
        self.assertEqual(res_mps, res_cpu)

        # 计算 B_mps 和 B_cpu 的 L3.5 范数
        res_mps = torch.linalg.vector_norm(B_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5)
        # 断言两个 L3.5 范数的结果相等
        self.assertEqual(res_mps, res_cpu)

        # 遍历 B_mps 的每一个维度，计算其 L3.5 范数
        for dim in range(0, B_mps.dim()):
            res_mps = torch.linalg.vector_norm(B_mps, ord=3.5, dim=dim)
            res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5, dim=dim)
            # 断言两个 L3.5 范数的结果相等
            self.assertEqual(res_mps, res_cpu)
    def test_layer_norm(self):
        # TODO: Test non-contiguous
        # 定义一个辅助函数，用于测试 LayerNorm 的功能
        def helper(input_shape, normalized_shape, eps=1e-05, elementwise_affine=True, dtype=torch.float32):
            # 在 CPU 上生成一个随机张量，并设置为需要梯度计算
            cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            # 将 CPU 上的张量复制到 "mps" 设备，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 在 CPU 和 "mps" 设备上分别创建 LayerNorm 操作对象
            cpu_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='cpu', dtype=dtype)
            mps_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='mps', dtype=dtype)
            # 在 CPU 上生成随机权重张量，并设置为需要梯度计算
            cpu_wt = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            # 将 CPU 上的权重张量复制到 "mps" 设备，并设置为需要梯度计算
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()
            # 在 CPU 上生成随机偏置张量，并设置为需要梯度计算
            cpu_bias = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            # 将 CPU 上的偏置张量复制到 "mps" 设备，并设置为需要梯度计算
            bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            # 如果启用了逐元素仿射变换，将权重和偏置参数设置到对应的 LayerNorm 操作对象中
            if (elementwise_affine):
                cpu_op.weight = torch.nn.Parameter(cpu_wt)
                mps_op.weight = torch.nn.Parameter(wt)
                cpu_op.bias = torch.nn.Parameter(cpu_bias)
                mps_op.bias = torch.nn.Parameter(bias)

            # 在 CPU 和 "mps" 设备上分别执行 LayerNorm 操作
            cpu_result = cpu_op(cpu_x)
            result = mps_op(x)

            # 在 CPU 上生成一个随机梯度张量，并将其复制到 "mps" 设备
            cpu_grad = torch.randn(cpu_result.shape)
            grad = cpu_grad.to('mps')

            # 分别在 CPU 和 "mps" 设备上执行反向传播
            cpu_result.backward(cpu_grad)
            result.backward(grad)

            # 断言 "mps" 设备上的结果与 CPU 上的结果一致
            self.assertEqual(result, cpu_result)
            # 断言 "mps" 设备上的输入梯度与 CPU 上的输入梯度一致
            self.assertEqual(x.grad, cpu_x.grad)
            # 如果启用了逐元素仿射变换，断言 "mps" 设备上的权重和偏置的梯度与 CPU 上的一致
            if (elementwise_affine):
                self.assertEqual(mps_op.weight.grad, cpu_op.weight.grad)
                self.assertEqual(mps_op.bias.grad, cpu_op.bias.grad)

        # 对于是否启用逐元素仿射变换的不同情况，调用 helper 函数进行测试
        for elementwise_affine in [True, False]:
            helper((2, 2, 2, 2), (2, 2), elementwise_affine=elementwise_affine)
            helper((2, 3, 4, 5), (4, 5), elementwise_affine=elementwise_affine)
            helper((2, 3, 4, 5, 6), (4, 5, 6), elementwise_affine=elementwise_affine)

        # 回归测试：https://github.com/pytorch/pytorch/issues/96113 的问题
        # 在 "mps" 设备上测试逆快速傅里叶变换（ifft）
        torch.nn.LayerNorm((16,), elementwise_affine=True).to("mps")(torch.randn(1, 2, 16).to("mps", dtype=torch.float16))

    @xfailIf(product_version < 14.0)
    def test_ifft(self):
        # 查看问题：https://github.com/pytorch/pytorch/issues/124096
        # 将设备设置为 "mps"
        device = torch.device("mps")

        N = 64
        # 在 "mps" 设备上生成一个随机信号张量
        signal = torch.rand(N, device=device)
        # 对信号进行实部快速傅里叶变换
        fft_result = torch.fft.rfft(signal)
        # 对快速傅里叶变换结果进行逆实部快速傅里叶变换，长度为信号张量的长度
        ifft_result = torch.fft.irfft(fft_result, n=signal.shape[0])

        # 预期的结果是逆变换后应该得到原始信号
        self.assertEqual(ifft_result, signal)

    # 测试 conv2d
    # 定义一个测试函数，用于测试卷积层的功能
    def test_conv2d_unit(self):
        # 定义一个内部辅助函数，用于执行单元测试的具体操作
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            # 生成随机输入张量，并转换为'MPS'（假设是某种特定设备）上的张量，支持梯度计算
            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 生成随机权重张量，并同样转换为'MPS'上的张量，支持梯度计算
            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            # 如果指定了偏置的形状，生成随机偏置张量，并同样转换为'MPS'上的张量，支持梯度计算
            if (bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            # 使用PyTorch函数进行二维卷积操作，计算输出张量y
            y = torch.nn.functional.conv2d(x, wt, bias=bias, stride=stride,
                                           padding=padding, dilation=dilation, groups=groups)
            # 用CPU上相同参数的张量计算参考输出ref_y
            ref_y = torch.nn.functional.conv2d(cpu_x, cpu_wt, bias=cpu_bias, stride=stride,
                                               padding=padding, dilation=dilation, groups=groups)

            # 生成用于梯度计算的参考梯度张量
            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            # 在输出y上执行反向传播，计算梯度
            y.backward(gradient=grad)
            # 在参考输出ref_y上执行反向传播，计算参考梯度
            ref_y.backward(gradient=cpu_grad)

            # 断言输出张量y与参考输出ref_y在指定的相对和绝对误差范围内相等
            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            # 断言输入张量x的梯度与CPU上相同张量的梯度在指定的相对和绝对误差范围内相等
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            # 断言权重张量wt的梯度与CPU上相同张量的梯度在指定的绝对误差范围内相等，相对误差不进行检查
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)
            # 如果存在偏置张量，断言偏置张量bias的梯度与CPU上相同张量的梯度在指定的绝对误差范围内相等，相对误差不进行检查
            if (bias_shape is not None):
                self.assertEqual(bias.grad, cpu_bias.grad, atol=8e-04, rtol=10.4e-05)

        # 设置第一组测试参数
        N = 1
        C_in = 3
        C_out = 64
        H = 64
        W = 64
        kH = 4
        kW = 4
        stride = 2
        padding = 1

        # 执行辅助函数helper，测试第一组参数下的卷积操作
        helper((N, C_in, H, W), (C_out, C_in, kH, kW), stride=stride, padding=padding)

        # 设置第二组测试参数
        N = 4
        C_in = 16
        H = 32
        W = 32

        C_out = 8
        kH = 3
        kW = 3

        # 遍历不同的groups值，执行辅助函数helper，测试不同参数下的卷积操作
        for groups in [1, 2, 4]:
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)

            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)
    # Test conv transpose 2d
    # Test sigmoid
    def test_sigmoid(self):
        # 定义辅助函数，用于测试不同形状的输入
        def helper(shape):
            # 在 CPU 上生成随机张量并设置梯度追踪
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将张量复制到 'mps' 设备上，并设置梯度追踪
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 创建 Sigmoid 操作对象
            sigmoid_op = torch.nn.Sigmoid()

            # 应用 Sigmoid 操作到输入张量上
            y = sigmoid_op(x)
            # 在 CPU 上应用 Sigmoid 操作到同一输入张量上
            ref_y = sigmoid_op(cpu_x)

            # 创建与 ref_y 形状相同的全 1 张量作为梯度
            cpu_grad = torch.ones_like(ref_y)
            # 将梯度张量转移到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 反向传播梯度到输入张量 x
            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            # 断言两个 Sigmoid 操作的输出张量 y 和 ref_y 相等
            self.assertEqual(y, ref_y)
            # 断言 'mps' 设备上的输入张量 x 的梯度与 CPU 上的 cpu_x.grad 相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 调用 helper 函数，测试不同形状的输入
        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    # Test tanh
    def test_tanh(self):
        # 定义辅助函数，用于测试不同形状的输入
        def helper(shape):
            # 在 CPU 上生成随机张量并设置梯度追踪
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将张量复制到 'mps' 设备上，并设置梯度追踪
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 创建 Tanh 操作对象
            tanh_op = torch.nn.Tanh()

            # 应用 Tanh 操作到输入张量上
            y = tanh_op(x)
            # 在 CPU 上应用 Tanh 操作到同一输入张量上
            ref_y = tanh_op(cpu_x)

            # 创建与 ref_y 形状相同的全 1 张量作为梯度
            cpu_grad = torch.ones_like(ref_y)
            # 将梯度张量转移到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 反向传播梯度到输入张量 x
            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            # 断言两个 Tanh 操作的输出张量 y 和 ref_y 相等
            self.assertEqual(y, ref_y)
            # 断言 'mps' 设备上的输入张量 x 的梯度与 CPU 上的 cpu_x.grad 相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 调用 helper 函数，测试不同形状的输入
        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    def test_threshold(self):
        # 定义辅助函数，用于测试 Threshold 操作
        def helper(threshold, value, num_elems, inplace=False, requires_grad=True):
            # 创建 Threshold 操作对象
            m = nn.Threshold(threshold=threshold, value=value, inplace=inplace)

            # 在 CPU 上生成随机输入张量，并设置梯度追踪
            input_cpu = torch.randn(num_elems, requires_grad=requires_grad, dtype=torch.float)
            # 将输入张量复制到 'mps' 设备上，并设置梯度追踪
            input_mps = input_cpu.detach().clone().to('mps').requires_grad_(requires_grad)

            # 应用 Threshold 操作到输入张量上
            output_cpu = m(input_cpu)
            output_mps = m(input_mps)

            # 创建与输出张量 output_cpu 形状相同的全 1 张量作为梯度
            cpu_grad = torch.ones_like(output_cpu)
            # 将梯度张量转移到 'mps' 设备上
            mps_grad = cpu_grad.to('mps')

            # 断言 'mps' 设备上的输出张量 output_mps 与 CPU 上的 output_cpu 相等
            self.assertEqual(output_cpu, output_mps)

            if requires_grad:
                # 如果需要梯度，反向传播梯度到输入张量 input_cpu 和 input_mps
                output_cpu.backward(gradient=cpu_grad)
                output_mps.backward(gradient=mps_grad)

                # 断言 'mps' 设备上的输入张量 input_mps 的梯度与 CPU 上的 input_cpu.grad 相等
                self.assertEqual(input_cpu.grad, input_mps.grad)

        # 调用 helper 函数，测试不同参数配置下的 Threshold 操作
        helper(threshold=0.1, value=20, num_elems=2)
        helper(threshold=-0.1, value=10, num_elems=10)
        helper(threshold=0.5, value=-15, num_elems=100)
        helper(threshold=1, value=10, num_elems=100, inplace=True, requires_grad=False)
    # 测试幂运算函数
    def test_pow(self):
        # 定义辅助函数，用于测试不同形状的输入数据
        def helper(shape):
            # 生成随机张量，设备为CPU，数据类型为浮点型，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 创建张量的副本并将其转换到'MPS'设备上
            x = cpu_x.detach().clone().to('mps')
            
            # 生成随机张量，设备为CPU，数据类型为浮点型，不需要梯度
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 创建张量的副本并将其转换到'MPS'设备上
            y = cpu_y.detach().clone().to('mps')
            
            # 计算 x 的 y 次幂
            z = torch.pow(x, y)
            # 计算 CPU 上对应张量的 y 次幂作为参考
            ref_z = torch.pow(cpu_x, cpu_y)

            # 断言两个张量是否相等
            self.assertEqual(z, ref_z)

            # 生成随机张量，设备为CPU，数据类型为浮点型，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 创建张量的副本并将其转换到'MPS'设备上
            x = cpu_x.detach().clone().to('mps')
            
            # 生成随机数作为指数
            exp = random.random()
            # 计算 x 的 exp 次幂
            z = torch.pow(x, exp)
            # 计算 CPU 上对应张量的 exp 次幂作为参考
            ref_z = torch.pow(cpu_x, exp)

            # 断言两个张量是否相等
            self.assertEqual(z, ref_z)

            # 生成随机数作为底数
            x = random.random()
            # 生成随机张量，设备为CPU，数据类型为浮点型，不需要梯度
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 创建张量的副本并将其转换到'MPS'设备上
            y = cpu_y.detach().clone().to('mps')
            
            # 计算 x 的 y 次幂
            z = torch.pow(x, y)
            # 计算 CPU 上对应张量的 y 次幂作为参考
            ref_z = torch.pow(x, cpu_y)

            # 断言两个张量是否相等
            self.assertEqual(z, ref_z)

        # 调用辅助函数进行测试，传入特定形状的张量
        helper((2, 8, 4, 5))

    # 测试 addcmul 函数
    def test_addcmul(self):
        # 定义辅助函数，用于测试不同形状和数据类型的输入数据
        def helper(shape, value, xtype=torch.float32, ytype=None, ztype=None):
            # 定义内部辅助函数，生成指定类型的随机张量
            def rand_helper(dtype):
                if dtype.is_floating_point:
                    return torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                return torch.randint(10, shape, dtype=dtype, device='cpu', requires_grad=False)

            # 生成随机张量，根据指定类型
            cpu_x = rand_helper(xtype)
            # 创建张量的副本并将其转换到'MPS'设备上
            x = cpu_x.detach().clone().to('mps')

            # 生成随机张量，根据指定类型或默认与 x 相同的类型
            cpu_y = rand_helper(ytype if ytype is not None else xtype)
            # 创建张量的副本并将其转换到'MPS'设备上
            y = cpu_y.detach().clone().to('mps')

            # 生成随机张量，根据指定类型或默认与 x 相同的类型
            cpu_z = rand_helper(ztype if ztype is not None else xtype)
            # 创建张量的副本并将其转换到'MPS'设备上
            z = cpu_z.detach().clone().to('mps')

            # 使用 addcmul 函数计算结果张量 y
            y = torch.addcmul(x, y, z, value=value)
            # 计算 CPU 上对应张量的结果张量作为参考 ref_y
            ref_y = torch.addcmul(cpu_x, cpu_y, cpu_z, value=value)

            # 断言两个张量是否相等
            self.assertEqual(y, ref_y)

        # 调用辅助函数进行测试，传入不同的形状和值
        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.1)
        helper((2, 3, 4, 5), 0.2)
        helper((2, 8, 4, 5), 0.2)
        # 测试整数类型
        helper((2, 2), 1.0, xtype=torch.int32)
        helper((2, 2), 2.0, xtype=torch.int16)

        # 测试混合类型
        helper((2, 2), 1.0, xtype=torch.float16, ytype=torch.float32)
        helper((3, 2), 1.0, ytype=torch.float16)
        helper((2, 3), 1.0, ztype=torch.float16)
        helper((2, 2), 1.0, xtype=torch.int32, ytype=torch.int16, ztype=torch.uint8)
        helper((2, 2), 1.0, ytype=torch.int16, ztype=torch.uint8)
    def test_addcdiv(self):
        # 定义一个辅助函数，用于测试 torch.addcdiv 函数的功能
        def helper(shape, value):
            # 在 CPU 上生成随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 生成随机张量并进行 clamp 操作，避免除以 0
            cpu_z = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False).clamp_min_(0.1)
            cpu_out = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)

            # 将 CPU 上的张量分离并克隆到 'mps' 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            mps_z = cpu_z.detach().clone().to('mps')
            mps_out = cpu_out.detach().clone().to('mps')

            # 在 'mps' 设备上执行 torch.addcdiv 操作
            result_div_mps = torch.addcdiv(mps_x, mps_y, mps_z, value=value)
            # 在 CPU 上执行相同的 torch.addcdiv 操作
            result_div_cpu = torch.addcdiv(cpu_x, cpu_y, cpu_z, value=value)
            # 断言 'mps' 设备上的结果与 CPU 上的结果相等
            self.assertEqual(result_div_mps, result_div_cpu)
            # 测试带有 .out 变体的 torch.addcdiv 操作
            self.assertEqual(torch.addcdiv(mps_x, mps_y, mps_z, out=mps_out, value=value), result_div_cpu)

        # 调用 helper 函数进行多组测试
        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.2)
        helper((2, 3, 4, 5), 1.0)  # 内部应忽略值为 1.0 的情况

    def test_addcdiv_transpose(self):
        # 回归测试，解决问题 https://github.com/pytorch/pytorch/issues/118115
        # 测试所有输入张量的连续性

        def helper(shape, value):
            # 根据形状生成转置后的形状
            shape_t = shape[::-1]
            # 多重循环用于测试不同情况下的张量
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # 根据条件生成不同的随机张量 x, y, z
                        x = torch.rand(shape, device="cpu") if i == 0 else torch.rand(shape_t, device="cpu").t()
                        y = torch.rand(shape, device="cpu") if j == 0 else torch.rand(shape_t, device="cpu").t()
                        z = torch.rand(shape, device="cpu") if k == 0 else torch.rand(shape_t, device="cpu").t()

                        # 分离并克隆到 'mps' 设备上
                        x_mps = x.detach().clone().to(device="mps")
                        y_mps = y.detach().clone().to(device="mps")
                        z_mps = z.detach().clone().to(device="mps")

                        # 在 CPU 上执行 torch.addcdiv_ 操作，原地修改 x
                        result_cpu = x.addcdiv_(y, z, value=value)
                        # 在 'mps' 设备上执行 torch.addcdiv 操作
                        result_mps = x_mps.addcdiv(y_mps, z_mps, value=value)
                        # 将 CPU 上的结果分离、克隆并转移到 'mps' 设备
                        result_mps_out = result_cpu.detach().clone().to('mps')
                        # 在 'mps' 设备上使用 .out 参数执行 torch.addcdiv 操作
                        torch.addcdiv(x_mps, y_mps, z_mps, out=result_mps_out, value=value)

                        # 断言 'mps' 设备上的结果与 CPU 上的结果相等
                        self.assertEqual(result_cpu, result_mps)
                        self.assertEqual(result_cpu, result_mps_out)

        # 调用 helper 函数进行多组测试
        helper((2, 3), 1.0)
        helper((2, 3), 0.2)
        helper((100, 300), 1.0)
        helper((100, 300), 0.2)
    def test_buffer_size_match(self):
        # 测试缓冲区大小匹配情况，不应该导致任何崩溃
        size = 16
        # 在 CPU 上生成随机张量 cpu_A
        cpu_A = torch.rand(size, device='cpu')
        # 在 CPU 上生成随机三维张量 cpu_F
        cpu_F = torch.rand(size, size, size, device='cpu')

        # 将 cpu_A 转换到 'mps' 设备得到 mps_A
        mps_A = cpu_A.to('mps')
        # 将 cpu_F 转换到 'mps' 设备得到 mps_F
        mps_F = cpu_F.to('mps')
        # 断言两个张量乘积在不同设备上的结果是否相等
        self.assertEqual(cpu_A @ cpu_F, mps_A @ mps_F)

    def test_transpose_inplace(self):
        # 定义一个二维列表 values
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        # 在 CPU 上创建张量 cpu_x
        cpu_x = torch.tensor(values, device='cpu')
        # 在 'mps' 设备上创建张量 mps_x，与 cpu_x 内容相同
        mps_x = torch.tensor(values, device='mps')

        # 将 cpu_x 原地转置
        cpu_x.transpose_(0, 1)
        # 将 mps_x 原地转置
        mps_x.transpose_(0, 1)
        # 断言转置后的 mps_x 转回 CPU 是否与原始的 cpu_x 相等
        self.assertEqual(cpu_x, mps_x.to('cpu'))

    def test_expand_cpu_to_mps_copy(self):
        # 测试将 CPU 张量扩展到 'mps' 设备的复制问题
        # 使用 torch.tensor 创建一个标量张量 x，并扩展到长度为 10 的向量，设备为 'mps'
        x = torch.tensor(1).expand([10]).to("mps")
        # 创建一个标量张量 x_cpu，并扩展到长度为 10 的向量
        x_cpu = torch.tensor(1).expand([10])

        # 断言 x_cpu 是否与 x 的 CPU 版本相等
        self.assertEqual(x_cpu, x.cpu())

    def test_cpu_to_strided_mps_copy(self):
        # 测试从 CPU 到带步幅的 'mps' 设备的复制问题
        # 创建一个张量 a1 在 'mps' 设备上，形状为 [[1, 2], [3, 4], [5, 6]]
        a1 = torch.Tensor([[1, 2], [3, 4], [5, 6]]).to(torch.device("mps"))
        # 创建一个张量 b1，形状为 [-1, -1]
        b1 = torch.Tensor([-1, -1])
        # 将 b1 赋值给 a1 的部分元素
        a1[1:, 1] = b1

        # 创建一个张量 a2 在 'mps' 设备上，形状同 a1
        a2 = torch.Tensor([[1, 2], [3, 4], [5, 6]]).to(torch.device("mps"))
        # 创建一个张量 b2 在 'mps' 设备上，形状同 b1
        b2 = torch.Tensor([-1, -1]).to(torch.device("mps"))
        # 将 b2 赋值给 a2 的部分元素
        a2[1:, 1] = b2

        # 断言 a1 和 a2 在 'mps' 设备上的内容是否相等
        self.assertEqual(a1, a2)

    def test_view_slice_reshape(self):
        # 测试视图、切片和重塑操作
        # 创建一个形状为 [1, 4, 4] 的 'mps' 设备上的随机张量 x
        x = torch.randn([1, 4, 4], device="mps")
        # 从 x 中切片得到 y，形状为 [1, 1, 3]
        y = x[0, :1, 1:]

        # 将 x 转换到 CPU 设备得到 x_cpu
        x_cpu = x.to("cpu")
        # 从 x_cpu 中切片得到 y_cpu，形状同 y
        y_cpu = x_cpu[0, :1, 1:]

        # 对 y 加 1 得到 r
        r = y + 1
        # 对 y_cpu 加 1 得到 r_cpu
        r_cpu = y_cpu + 1
        # 断言 r 和 r_cpu 是否相等
        self.assertEqual(r, r_cpu)

    def test_slice_reshape(self):
        # 测试切片和重塑操作
        # 创建一个形状为 [1, 6, 4, 2]，dtype 为 torch.float，设备为 'mps' 的随机张量 x
        x = torch.randn([1, 6, 4, 2], dtype=torch.float, device="mps")
        # 创建 x 的副本 x_cpu，并转换到 CPU 设备
        x_cpu = x.detach().clone().to("cpu")

        # 对 x 进行切片和重塑操作，形状变为 [2, 3, 4, 1]
        x = x[:, 3:].view(2, 3, 4, 1)
        # 对 x_cpu 进行相同的切片和重塑操作
        x_cpu = x_cpu[:, 3:].view(2, 3, 4, 1)
        # 断言 x 和 x_cpu 是否相等
        self.assertEqual(x, x_cpu)

        # 对 x 和 x_cpu 分别加 2
        x = x + 2
        x_cpu = x_cpu + 2
        # 再次断言 x 和 x_cpu 是否相等
        self.assertEqual(x, x_cpu)
    def test_reshape_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/95883
        # 设置批大小 B 和序列长度 T
        B = 4
        T = 1

        # 创建一个在 CPU 上的线性层
        lin_cpu = nn.Linear(10, 256)
        # 创建一个在 MPS（神经存储系统）上的线性层
        lin_mps = nn.Linear(10, 256, device="mps")

        # 使用与 CPU 上线性层相同的权重和偏置
        lin_mps.weight.data = lin_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        lin_mps.bias.data = lin_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        # 在 MPS 设备上生成随机张量，并要求梯度计算
        x_mps = torch.rand([B, T, 10], device="mps", requires_grad=True)
        # 将 MPS 上的输入张量克隆到 CPU 上，并要求梯度计算
        x_cpu = x_mps.detach().clone().cpu().requires_grad_()
        
        # 在 MPS 线性层上计算输出
        x_mps = lin_mps(x_mps)
        # 在 CPU 线性层上计算输出
        x_cpu = lin_cpu(x_cpu)

        # 断言 MPS 输出的形状
        self.assertEqual(x_mps.shape, (B, T, 256))
        # 断言 CPU 输出的形状
        self.assertEqual(x_cpu.shape, (B, T, 256))

        # 在 MPS 设备上生成类别令牌张量，并要求梯度计算，并重复 B 次
        cls_token_mps = torch.rand([1, 256], device="mps", requires_grad=True).repeat(B, 1, 1)
        # 将 MPS 上的类别令牌张量克隆到 CPU 上
        cls_token_cpu = cls_token_mps.detach().clone().cpu()

        # 在 MPS 上连接类别令牌和输出张量
        x_mps = torch.cat([cls_token_mps, x_mps], dim=1)
        # 在 CPU 上连接类别令牌和输出张量
        x_cpu = torch.cat([cls_token_cpu, x_cpu], dim=1)

        # 调换 MPS 输出的维度 0 和 1
        x_mps = x_mps.transpose(0, 1)
        # 调换 CPU 输出的维度 0 和 1
        x_cpu = x_cpu.transpose(0, 1)

        # 在 MPS 上生成与 x_mps 相同形状的随机张量作为目标
        target_mps = torch.rand_like(x_mps)
        # 将 MPS 上的目标张量克隆到 CPU 上
        target_cpu = target_mps.detach().clone().cpu()

        # 计算 MPS 输出与目标之间的均方误差损失
        loss_mps = F.mse_loss(x_mps, target_mps)
        # 计算 CPU 输出与目标之间的均方误差损失
        loss_cpu = F.mse_loss(x_cpu, target_cpu)

        # 断言 MPS 输出的损失与 CPU 输出的损失相等
        self.assertEqual(loss_mps, loss_cpu)

        # 反向传播 MPS 输出的损失
        loss_mps.backward()
        # 反向传播 CPU 输出的损失
        loss_cpu.backward()

        # 断言 MPS 输出的梯度与 CPU 输出的梯度相等
        self.assertEqual(x_mps.grad, x_cpu.grad)

    def test_stack_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/87856
        # 创建一个在 CPU 上的张量 x_cpu
        x_cpu = torch.tensor([[1, 2]])
        # 将 CPU 上的张量 x_cpu 克隆到 MPS 上
        x_mps = x_cpu.detach().clone().to("mps")

        # 在 CPU 上使用张量 x_cpu 的切片进行堆叠
        y_cpu = torch.stack((x_cpu[:, :1], x_cpu[:, -1:]), dim=-1)
        # 在 MPS 上使用张量 x_mps 的切片进行堆叠
        y_mps = torch.stack((x_mps[:, :1], x_mps[:, -1:]), dim=-1)

        # 断言 CPU 和 MPS 上堆叠后的张量 y 的相等性
        self.assertEqual(y_cpu, y_mps)

        # 在 MPS 设备上创建一个张量 t_mps
        t_mps = torch.tensor([1, 2, 3, 4], device="mps")
        # 将 MPS 上的张量 t_mps 克隆到 CPU 上
        t_cpu = t_mps.detach().cpu().detach()

        # 在 MPS 上对张量 t_mps 进行切片操作
        x_mps = t_mps[2:]
        y_mps = t_mps[:2]

        # 在 CPU 上对张量 t_cpu 进行切片操作
        x_cpu = t_cpu[2:]
        y_cpu = t_cpu[:2]

        # 在 MPS 上堆叠切片后的张量
        res_mps = torch.stack((y_mps, x_mps), dim=-1)
        # 在 CPU 上堆叠切片后的张量
        res_cpu = torch.stack((y_cpu, x_cpu), dim=-1)

        # 断言 MPS 和 CPU 上堆叠后的结果张量的相等性
        self.assertEqual(res_mps, res_cpu)

    def test_unsafe_chunk(self):
        # https://github.com/pytorch/pytorch/issues/91065
        # 在 CPU 上生成一个随机张量 a
        a = torch.rand(5, dtype=torch.float32, device="cpu")
        # 在 CPU 上对张量 a 进行不安全的分块操作
        ret = a.unsafe_chunk(4, 0)
        # 计算不安全分块后的结果
        y = ret[0] * ret[2]

        # 将 CPU 上的张量 a 转换到 MPS 设备上
        a_mps = a.to("mps")
        # 在 MPS 上对张量 a_mps 进行不安全的分块操作
        ret_mps = a_mps.unsafe_chunk(4, 0)
        # 计算不安全分块后的结果
        y_mps = ret_mps[0] * ret_mps[2]

        # 断言 CPU 和 MPS 上不安全分块后的结果的相等性
        self.assertEqual(y, y_mps)
    def test_slice_casting(self):
        # 生成随机的二进制数
        cpu_in = torch.bernoulli(torch.empty(1, 1, 128, 128).uniform_(0, 1)).to(torch.uint8)
        # 将CPU上的张量转移到"mps"设备上
        mps_in = cpu_in.detach().clone().to("mps")
        # 在具有存储偏移的张量上检查 copy_cast(unit8 -> bool)
        cpu_out = cpu_in[:, :, 11 : 12, :12].to(torch.bool)
        mps_out = mps_in[:, :, 11 : 12, :12].to(torch.bool)
        # 断言两个张量是否相等
        self.assertEqual(cpu_out, mps_out)

    def test_slice_reshape_contg_view(self):
        import torch

        x_mps = torch.randn(1, 4800, 2, device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        r_mps = x_mps + 2
        r_cpu = x_cpu + 2

        # 断言两个张量是否相等
        self.assertEqual(r_mps, r_cpu)

    def test_contiguous_slice_2d(self):
        def helper(shape):
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    t_mps = torch.randn(shape, device="mps")
                    t_cpu = t_mps.detach().clone().cpu()

                    y_mps = t_mps[i:, :j]
                    y_cpu = t_cpu[i:, :j]
                    # 断言两个张量是否相等
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[i:, j]
                    y_cpu = t_cpu[i:, j]
                    # 断言两个张量是否相等
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[i, :j]
                    y_cpu = t_cpu[i, :j]
                    # 断言两个张量是否相等
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, :j]
                    y_cpu = t_cpu[:i, :j]
                    # 断言两个张量是否相等
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, j]
                    y_cpu = t_cpu[:i, j]
                    # 断言两个张量是否相等
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, j:]
                    y_cpu = t_cpu[:i, j:]
                    # 断言两个张量是否相等
                    self.assertEqual(y_mps + 1, y_cpu + 1)

        l = []
        for N in range(1, 3):
            l.append(N)
            for C in range(1, 3):
                l.append(C)
                helper(l)
                for D in range(1, 3):
                    l.append(D)
                    helper(l)
                    for H in range(1, 3):
                        l.append(H)
                        helper(l)
                        for W in range(1, 3):
                            l.append(W)
                            helper(l)
                            l.pop()
                        l.pop()
                    l.pop()
                l.pop()
            l.pop()

        helper([9, 15, 4])
        helper([9, 3, 2])
        helper([3, 4, 18, 22])
        helper([3, 4, 18, 22, 150])

    def test_contiguous_slice_3d(self):
        x = torch.randn(2, 3, 3, device="mps")
        x_cpu = x.detach().clone().cpu()
        x = x[:1]
        x_cpu = x_cpu[:1]
        out = x[:, 0:1, 0:1] * x[:, 1:2, 1:2]
        out_cpu = x_cpu[:, 0:1, 0:1] * x_cpu[:, 1:2, 1:2]
        # 断言两个张量是否相等
        self.assertEqual(out, out_cpu)
    def test_view_slice(self):
        # 定义常量 NUM_SAMPLES 为 60
        NUM_SAMPLES = 60
        # 创建包含两个元素的元组 s
        s = (0, 1)

        # 在 CPU 上生成一个形状为 (8000, 3) 的随机张量 X
        X = torch.rand(8000, 3, dtype=torch.float32, device='cpu')
        # 创建 X 的一个副本 X_mps，并将其从计算图中分离并复制到 CPU
        X_mps = X.detach().clone().to("cpu")

        # 生成一个随机整数索引 idx，范围为 [0, X.shape[0])，并重复 len(s) 次
        idx = torch.randint(0, X.shape[0], (1,)).repeat(len(s))
        # 生成一个随机整数张量 pts，形状为 (NUM_SAMPLES, X.shape[1])，范围为 [0, X.shape[0])
        pts = torch.randint(0, X.shape[0], (NUM_SAMPLES, X.shape[1]))
        # 将 idx 转换到设备 "mps"，创建 idx_mps
        idx_mps = idx.to("mps")
        # 将 pts 转换到设备 "mps"，创建 pts_mps
        pts_mps = pts.to("mps")
        # 使用 idx 更新 pts 的前两列（s 所指示的列）
        pts[:, s] = idx
        # 使用 idx_mps 更新 pts_mps 的前两列（s 所指示的列）
        pts_mps[:, s] = idx_mps

        # 创建一个全零张量 actual_pts，形状为 (NUM_SAMPLES, X.shape[1])，数据类型为 float32
        actual_pts = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float)
        # 创建一个全零张量 actual_pts_mps，形状为 (NUM_SAMPLES, X.shape[1])，数据类型为 float32，设备为 "mps"
        actual_pts_mps = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float, device="mps")

        # 遍历 NUM_SAMPLES
        for i in range(NUM_SAMPLES):
            # 遍历 X 的第二个维度（列数）
            for j in range(X.shape[1]):
                # 使用 pts_mps 的索引更新 actual_pts_mps
                actual_pts_mps[i, j] = X_mps[pts_mps[i, j], j]
                # 使用 pts 的索引更新 actual_pts
                actual_pts[i, j] = X[pts[i, j], j]
                # 断言 actual_pts 和 actual_pts_mps 的对应元素相等
                self.assertEqual(actual_pts[i, j], actual_pts_mps[i, j])

    def test_slice_scatter(self):
        # 定义形状为 (4, 4) 的张量 tensor，其中元素为 [0, 10) 的随机整数，设备为 "mps"
        shape = (4, 4)
        tensor = torch.randint(10, shape, device="mps")
        # 创建 tensor 的一个副本 tensor_before
        tensor_before = tensor.clone()
        # 在设备 "mps" 上创建一个形状为 (4, 8) 的空张量，并将 tensor 的内容复制到每隔一个列上
        torch.empty(shape[0], shape[1] * 2, device="mps")[:, ::2].copy_(tensor)
        # 断言 tensor 和 tensor_before 相等
        torch.testing.assert_close(tensor, tensor_before)

    def test_slice(self):
        # 定义一个二维列表 values
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        # 创建一个 CPU 上的张量 cpu_x，元素为 values
        cpu_x = torch.tensor(values, device='cpu')
        # 创建一个设备为 "mps" 的张量 mps_x，元素为 values，数据类型为 float32
        mps_x = torch.tensor(values, device='mps', dtype=torch.float)

        # 对 cpu_x 和 mps_x 进行切片操作，并断言它们相等
        cpu_slice1 = cpu_x[:2, :]
        mps_slice1 = mps_x[:2, :]
        self.assertEqual(cpu_slice1, mps_slice1)

        cpu_slice2 = cpu_x[:, :1]
        mps_slice2 = mps_x[:, :1]
        self.assertEqual(cpu_slice2, mps_slice2)

        cpu_slice3 = cpu_x[1:2, :]
        mps_slice3 = mps_x[1:2, :]
        # 将 mps_slice3 转换到 CPU，并断言它们相等
        self.assertEqual(cpu_slice3, mps_slice3.to('cpu'))

        cpu_slice4 = cpu_x[1, :]
        # 将 mps_x 的行索引为 1 的数据转换到 CPU，并断言它们相等
        mps_slice4 = mps_x[1, :].to('cpu')
        self.assertEqual(cpu_slice4, mps_slice4)

    def test_scalar_from_slice_unary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        # 创建一个包含两个浮点数的张量 tensor_list，设备为 "mps"
        tensor_list = torch.tensor([1.0, 1.2], device="mps")

        # 遍历 tensor_list 中的每个标量
        for scalar in tensor_list:
            # 对 scalar 应用向上取整操作，并在 "mps" 设备上执行
            r_mps = torch.ceil(scalar)
            # 对 scalar 应用向上取整操作，并转换到 CPU 上执行
            r_cpu = torch.ceil(scalar.to("cpu"))
            # 断言 r_mps 和 r_cpu 相等
            self.assertEqual(r_mps.cpu(), r_cpu)

    def test_scalar_from_slice_binary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        # 定义一个辅助函数 helper，接受一个二元操作 binary_op
        def helper(binary_op):
            # 创建一个包含四个浮点数的张量 tensor_list，设备为 "mps"
            tensor_list = torch.tensor([1.0, 1.2, 2.5, 1.0], device="mps")

            # 遍历 tensor_list 中的每个标量
            for scalar in tensor_list:
                # 使用 binary_op 对标量 scalar 和常数 1.0 进行操作，并在 "mps" 设备上执行
                r_mps = binary_op(scalar, 1.0)
                # 将 scalar 转换到 CPU 上，并与常数 1.0 使用 binary_op 操作
                r_cpu = binary_op(scalar.cpu(), 1.0)
                # 断言 r_mps 和 r_cpu 相等
                self.assertEqual(r_mps.cpu(), r_cpu)

        # 调用 helper 函数，传递 torch.sub 函数进行测试
        helper(torch.sub)
        # 调用 helper 函数，传递 torch.add 函数进行测试
        helper(torch.add)
        # 调用 helper 函数，传递 torch.not_equal 函数进行测试
        helper(torch.not_equal)
        # 调用 helper 函数，传递 torch.eq 函数进行测试
        helper(torch.eq)
    def test_slice_contiguous_view(self):
        # 测试函数：检查在PyTorch张量切片时的视图连续性问题
        # https://github.com/pytorch/pytorch/issues/77750

        def helper(operator):
            # 创建在不同设备上的两个PyTorch张量
            t_mps = torch.tensor([1, 2, 3, 4], device="mps")
            t_cpu = torch.tensor([1, 2, 3, 4], device="cpu")

            # 获取连续视图
            x_mps = t_mps[2:]  # 3, 4
            y_mps = t_mps[:2]  # 1, 2

            x_cpu = t_cpu[2:]
            y_cpu = t_cpu[:2]

            res_mps = res_cpu = None
            # 根据不同操作符进行操作
            if operator == "<=":
                res_mps = x_mps <= y_mps
                res_cpu = x_cpu <= y_cpu
            elif operator == "<":
                res_mps = x_mps < y_mps
                res_cpu = x_cpu < y_cpu
            elif operator == ">=":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            elif operator == ">":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            elif operator == "==":
                res_mps = x_mps == y_mps
                res_cpu = x_cpu == y_cpu
            elif operator == "!=":
                res_mps = x_mps != y_mps
                res_cpu = x_cpu != y_cpu
            elif operator == "stack":
                res_mps = torch.stack((y_mps, x_mps), dim=-1)
                res_cpu = torch.stack((y_cpu, x_cpu), dim=-1)

            # 断言两个设备上的结果是否相等
            self.assertEqual(res_mps, res_cpu)

        # 针对每个操作符调用helper函数
        for op in ["<=", "<", ">=", ">", "==", "!=", "stack"]:
            helper(op)

    def test_slice_of_slice(self):
        # 测试函数：检查PyTorch张量切片中切片的使用情况
        x = torch.tensor([0.5, 0.5], device="cpu")
        x_mps = torch.tensor([0.5, 0.5], device="mps")

        # 获取张量的切片后再次切片
        tensor = x[1][None]
        tensor_mps = x_mps[1][None]

        # 计算不等于零的元素
        res = tensor.ne(0)
        res_mps = tensor_mps.ne(0)

        # 断言两个设备上的结果是否相等
        self.assertEqual(res, res_mps)

    def test_index_storage_offset(self):
        # 测试函数：检查PyTorch张量索引和存储偏移的问题
        # https://github.com/pytorch/pytorch/issues/78107

        a = torch.tensor([8.2670e-01, -1.0293e+00])
        b_cpu = a[0]
        c_cpu = a[1]

        # 'b' 和 'c' 都是 'a' 的视图
        # 'b' 的存储偏移为0，而 'c' 的存储偏移为1
        # 当从 'cpu' 复制到 'mps' 时，c 将具有存储偏移为1，这需要考虑在内，
        # 否则将会得到与 'b' 相同的值
        b = b_cpu.to('mps')
        c = c_cpu.to('mps')

        # 比较 'b' 和 'c' 的大小关系
        res_mps = b > c
        res_cpu = b_cpu > c_cpu
        self.assertEqual(res_mps, res_cpu)

        res_mps = c > b
        res_cpu = c_cpu > b_cpu
        self.assertEqual(res_mps, res_cpu)
    # 测试 flatten 方法
    def test_flatten(self):
        # 创建一个三维列表
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        # 在 CPU 上创建张量
        cpu_x = torch.tensor(values, device='cpu')
        # 在自定义设备 "mps" 上创建张量
        mps_x = torch.tensor(values, device='mps')

        # 执行 flatten 操作，返回新的张量（在原设备上）
        cpu_flatten1 = cpu_x.flatten()
        # 执行 flatten 操作，并将结果转移到 CPU 上
        mps_flatten1 = mps_x.flatten().to('cpu')
        # 断言两个张量相等
        self.assertEqual(cpu_flatten1, mps_flatten1)

        # 指定 start_dim 执行 flatten 操作
        cpu_flatten2 = cpu_x.flatten(start_dim=1)
        mps_flatten2 = mps_x.flatten(start_dim=1).to('cpu')
        self.assertEqual(cpu_flatten2, mps_flatten2)

        # 指定 end_dim 执行 flatten 操作
        cpu_flatten3 = cpu_x.flatten(end_dim=1)
        mps_flatten3 = mps_x.flatten(end_dim=1).to('cpu')
        self.assertEqual(cpu_flatten3, mps_flatten3)

    # 测试 repeat 方法
    def test_repeat(self):
        # 定义辅助函数 helper
        def helper(shape, repeats):
            # 在 CPU 上生成随机张量，并启用梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 分离并克隆张量到 "mps" 设备，并启用梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 使用 repeat 方法对张量进行重复操作
            y = x.repeat(repeats)
            ref_y = cpu_x.repeat(repeats)

            # 在 CPU 上生成与 ref_y 形状相同的随机梯度
            cpu_grad = torch.randn(ref_y.shape)
            # 将梯度转移到 "mps" 设备
            grad = cpu_grad.to('mps')

            # 对 y 执行反向传播，使用 "mps" 设备上的梯度
            y.backward(gradient=grad)
            # 对 ref_y 执行反向传播，使用在 CPU 上生成的梯度
            ref_y.backward(gradient=cpu_grad)

            # 断言两个重复操作的结果张量相等
            self.assertEqual(y, ref_y)
            # 断言 "mps" 设备上的梯度与在 CPU 上的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 使用 helper 函数测试不同形状和重复参数的情况
        helper((2, 3, 4, 5), (2, 3, 4, 5))
        helper((2, 3, 4), (4, 3, 2, 5, 7, 2))
        helper((3, 4, 5), (2, 3, 4, 5))
        helper((3, 4, 5), (2, 2, 2))

    # 测试 torch.repeat_interleave 方法
    def test_torch_repeat_interleave(self, device="mps"):
        # 创建一个二维张量 y
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # 调用 repeat_interleave 方法，使用单个参数的函数签名
        temp = y.repeat_interleave(2)
        self.assertEqual(torch.Size([8]), temp.size())

        # 遍历多种数据类型进行测试
        for dtype in [torch.int, torch.long]:
            # 创建长度张量 lengths，并指定数据类型和设备
            lengths = torch.tensor([1, 2], dtype=dtype, device="mps")
            # 计算总输出大小
            output_size = torch.sum(lengths)
            # 使用 torch.repeat_interleave 方法，指定多个参数的函数签名
            a = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
            )
            self.assertEqual(a.dtype, y.dtype)
            self.assertEqual(a.size(), torch.Size([3, 2]))

            # 使用 torch.repeat_interleave 方法，指定 output_size 参数
            a_with_output = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
                output_size=output_size,
            )
            self.assertEqual(a_with_output.dtype, y.dtype)
            self.assertEqual(a_with_output.size(), torch.Size([3, 2]))
    # 定义一个测试函数，用于测试 torch.repeat_interleave 函数的不同用法
    def test_repeat_interleave(self, device="mps"):
        # 创建一个张量 x，包含 [0, 1, 2, 3]，指定设备为 device
        x = torch.tensor([0, 1, 2, 3], device=device)
        # 创建一个期望的张量 expected，包含 [1, 2, 2, 3, 3, 3]，指定设备为 device
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        # 使用 torch.repeat_interleave 函数对张量 x 进行操作，并比较结果是否与 expected 相等
        # 在 macOS 13.3 之前，torch.int64 类型的输入返回 torch.int32 类型
        self.assertEqual(torch.repeat_interleave(x), expected, exact_dtype=product_version >= 13.3)

        # 使用断言检查 torch.repeat_interleave 对于二维张量的错误使用情况
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        # 使用断言检查 torch.repeat_interleave 对于浮点类型张量的错误使用情况
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        # 使用断言检查 torch.repeat_interleave 对于包含负数的张量的错误使用情况
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        # 创建一个二维张量 y，包含 [[1, 2], [3, 4]]，指定设备为 device
        y = torch.tensor([[1, 2], [3, 4]], device=device)

        # 使用 torch.repeat_interleave 函数对张量 y 进行操作，测试不同参数的结果
        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        # 创建一个期望的张量 y1_expect，包含 [1, 1, 2, 2, 3, 3, 4, 4]，指定设备为 device
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        # 比较结果是否与期望的张量 y1_expect 相等
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        # 使用 torch.repeat_interleave 函数对张量 y 进行操作，指定 dim=1
        y2 = torch.repeat_interleave(y, 3, dim=1)
        # 创建一个期望的张量 y2_expect，包含 [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]，指定设备为 device
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        # 比较结果是否与期望的张量 y2_expect 相等
        self.assertEqual(y2, y2_expect)

        # 使用 torch.repeat_interleave 函数对张量 y 进行操作，指定 dim=0 和不同的 repeats 参数
        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        # 创建一个期望的张量 y3_expect，包含 [[1, 2], [3, 4], [3, 4]]，指定设备为 device
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        # 比较结果是否与期望的张量 y3_expect 相等
        self.assertEqual(y3, y3_expect)

        # 使用断言检查 torch.repeat_interleave 对于不合理的 repeats 参数情况
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        # 使用断言检查 torch.repeat_interleave 对于不合理的 repeats 张量参数情况
        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # 测试零大小维度的情况
        # 创建一个零维度的张量 x，设备为 device
        x = torch.zeros((5, 0), device=device)
        # 使用 torch.repeat_interleave 函数对张量 x 进行操作，指定 repeats=3 和 dim=1
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        # 比较结果是否与新建的零维张量相等
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        # 创建一个空张量 x，指定 dtype=torch.int64 和设备为 device
        x = torch.tensor([], dtype=torch.int64, device=device)
        # 使用 torch.repeat_interleave 函数对空张量 x 进行操作，重复张量本身
        y = torch.repeat_interleave(x, x)
        # 比较结果是否与空张量 x 相等
        self.assertEqual(y, x)
    def test_repeat_interleave_simple(self):
        def helper(shape, dtype=torch.float32, num_repeats=torch.Tensor(), dim=None):
            # 生成指定形状和数据类型的随机张量 x，存储在 "mps" 设备上
            x = torch.randn(shape, dtype=dtype, device="mps")
            # 创建 x 的 CPU 克隆副本
            x_cpu = x.detach().clone().cpu()

            # 创建 num_repeats 的 CPU 克隆副本
            num_repeats_cpu = num_repeats.detach().clone().cpu()

            # 在指定维度 dim 上重复插入张量 x，生成 repeats 张量
            repeats = torch.repeat_interleave(x, num_repeats, dim)
            # 在指定维度 dim 上重复插入 CPU 张量 x_cpu，生成 repeats_cpu 张量
            repeats_cpu = torch.repeat_interleave(x_cpu, num_repeats_cpu, dim)

            # 断言 repeats 和 repeats_cpu 张量相等
            self.assertEqual(repeats, repeats_cpu)

        # 测试辅助函数 helper 的不同参数组合
        helper(shape=3, num_repeats=torch.tensor([100], device="mps"))
        helper(shape=(2, 2), num_repeats=torch.tensor([3, 3], device="mps"), dim=0)
        helper(shape=(10, 15, 8), num_repeats=torch.arange(10, device="mps"), dim=0)
        helper(shape=(10, 15, 8), num_repeats=torch.randint(0, 100, (15, ), device="mps"), dim=1)
        helper(shape=(10, 15, 30), num_repeats=torch.randint(0, 100, (30, ), device="mps"), dim=2)

    def test_count_nonzero(self):
        def helper(dtype):
            # 一个包含多个内部列表的 Python 列表 n
            n = [
                [[1, 0, 2], [3, 0, 2], [7, 9, -4]],
                [[0, 2, 3], [3, 2, 1], [2, 0, 0]],
            ]
            # 创建一个 CPU 上的张量 cpu_x，数据类型为 dtype
            cpu_x = torch.tensor(n, dtype=dtype)
            # 将张量 cpu_x 转移到 "mps" 设备上，生成 mps_x 张量
            mps_x = torch.tensor(n, dtype=dtype).to('mps')

            # 检查所有非零元素的数量是否相等
            self.assertEqual(
                torch.count_nonzero(cpu_x),
                torch.count_nonzero(mps_x)
            )

            # 沿着 dim=1 维度检查非零元素的数量是否相等
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=1),
                torch.count_nonzero(mps_x, dim=1)
            )

            # 沿着 dim=(0, 1) 维度检查非零元素的数量是否相等
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=(0, 1)),
                torch.count_nonzero(mps_x, dim=(0, 1))
            )

        # 测试辅助函数 helper 的不同数据类型参数
        helper(torch.int32)
        helper(torch.int64)
        helper(torch.float16)
        helper(torch.float32)

    def _test_module_empty_input(self, module, inp, check_size=True):
        # 将输入张量设置为需要计算梯度
        inp.requires_grad_(True)
        # 使用 module 处理输入 inp，得到输出张量 out
        out = module(inp)
        # 创建一个与 out 相同形状的随机张量 gO
        gO = torch.rand_like(out)
        # 对 out 进行反向传播，计算梯度
        out.backward(gO)

        # 如果 check_size 为真，则断言 out 和 inp 的大小相等
        if check_size:
            self.assertEqual(out.size(), inp.size())

        # 遍历 module 的参数，将需要计算梯度的参数的梯度置为零
        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))

        # 断言输入张量的梯度为零
        self.assertEqual(inp.grad, torch.zeros_like(inp))

    # 测试 dtype 转换，包括同时更改设备的情况
    # 定义测试方法，验证不同数据类型和操作在不同设备上的行为
    def test_to(self):
        # 创建包含两个二维列表的三维列表
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        # 在 CPU 设备上创建张量
        cpu_x = torch.tensor(values, device='cpu')
        # 在 'mps' 设备上创建张量
        mps_x = torch.tensor(values, device='mps')

        # 验证转换为整数类型后在 CPU 上的值是否相等
        self.assertEqual(cpu_x.int(), mps_x.int().cpu())
        # 验证转换为布尔类型后在 CPU 上的值是否相等
        self.assertEqual(cpu_x.bool(), mps_x.bool().cpu())
        # 验证转换为浮点数类型后在 CPU 上的值是否相等
        self.assertEqual(cpu_x.float(), mps_x.float().cpu())

        # 验证在 'mps' 设备上创建的单个浮点数转换为整数后在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(1.3, device='mps').int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        # 验证在 'mps' 设备上创建的单个浮点数转换为布尔类型后在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(0.0, device='mps').bool().cpu(), torch.tensor(False))
        # 验证在 'mps' 设备上创建的单个浮点数转换为布尔类型后在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(0.1, device='mps').bool().cpu(), torch.tensor(True))
        # 验证在 'mps' 设备上创建的单个浮点数转换为布尔类型再转换为整数类型后在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        # 验证在 'mps' 设备上创建的单个浮点数转换为布尔类型再转换为整数类型再转换为浮点数类型后在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().float().cpu(),
                         torch.tensor(1.0))
        # 验证在 'mps' 设备上创建的单个浮点数转换为整数类型后在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(4.25, device='mps').to('cpu', torch.int),
                         torch.tensor(4, dtype=torch.int32))
        # 验证在 CPU 设备上创建的单个浮点数转换为 'mps' 设备后再转换为整数类型在 CPU 上的值是否相等
        self.assertEqual(torch.tensor(4.25, device='cpu').to('mps', torch.int).cpu(),
                         torch.tensor(4, dtype=torch.int32))
        # 验证在 CPU 设备上创建的单个负浮点数转换为 'mps' 设备后再转换为整数类型的行为
        self.assertEqual(torch.tensor(-8.34, device='cpu').to('mps', torch.int),
                         torch.tensor(-8.34, device='cpu').to('mps').to(torch.int))
        # 将 int8 和 uint8 转换为浮点数并比较结果
        # 参考 https://github.com/pytorch/pytorch/issues/80009 获取更多细节
        cpu_byte = torch.tensor([60, 160, 20, 220], dtype=torch.uint8)
        cpu_char = torch.tensor([60, -60, 20, -120], dtype=torch.uint8)
        for x_cpu in [cpu_byte, cpu_char]:
            # 将 CPU 上的张量转换到 'mps' 设备上
            x_mps = x_cpu.to('mps')
            # 验证转换为浮点数后在两个设备上的值是否相等
            self.assertEqual(x_mps.to(torch.float32), x_cpu.to(torch.float32))


    # 定义测试方法，验证在 'mps' 设备上使用标量赋值操作的行为
    def test_setitem_scalar(self) -> None:
        # 设备类型为 'mps'
        device = 'mps'
        # 遍历不同的数据类型
        for dtype in [torch.int32, torch.float32, torch.int64]:
            # 遍历不同的行数
            for i in range(3, 6):
                # 遍历不同的列数
                for j in range(3, 6):
                    # 创建指定大小和数据类型的零张量在 'mps' 设备上
                    t = torch.zeros(i, j, dtype=dtype, device=device)
                    # 验证张量元素之和是否为零
                    self.assertEqual(t.sum(), 0)
                    # 修改张量中的元素值
                    t[1, 1] = 1
                    t[2, 1] = j
                    t[1, 2] = i
                    # 验证张量中特定位置的值是否符合预期
                    self.assertEqual(t[1, 1], 1)
                    self.assertEqual(t[1, 2], i)
                    self.assertEqual(t[2, 1], j)
                    # 验证张量元素之和是否正确计算
                    self.assertEqual(t.sum(), 1 + i + j)


    # 定义测试方法，验证张量的步幅情况
    def test_stride_of_strides(self) -> None:
        # 在 'mps' 设备上创建形状为 (32, 1) 的随机张量
        x = torch.rand(32, 1, device='mps')
        # 创建步幅为 (1, 0) 的 x 的新视图 y
        y = x.as_strided(size=(32, 2), stride=(1, 0))
        # 创建步幅为 (1, 0) 的 y 的新视图并转换到 CPU 上的张量 z
        z = y.as_strided(size=(32, 3), stride=(1, 0)).to("cpu")
        # 验证在 CPU 上创建步幅为 (1, 0) 的 x 的新视图是否与 z 相等
        self.assertEqual(x.to("cpu").as_strided(size=(32, 3), stride=(1, 0)), z)
    def test_type_casting(self):
        # 定义一个测试函数，用于测试类型转换的正确性
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            # 创建一个 CPU 上的张量，使用给定的数据
            a_cpu = torch.tensor(data)
            # 将该张量移动到指定的 MPS 设备上
            a_mps = a_cpu.to(torch.device('mps'))

            # 在 CPU 上进行类型转换，并保存结果
            res_cpu = a_cpu.type(to_dtype)
            # 在 MPS 设备上进行类型转换，并保存结果
            res_mps = a_mps.type(to_dtype)
            # 断言 CPU 和 MPS 设备上的转换结果应该相等
            self.assertEqual(res_cpu, res_mps)

        # 使用 helper 函数测试不同数据类型的类型转换
        helper([9.0, 3.0, 5.0, 4.0], torch.LongTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.FloatTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.IntTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ShortTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.HalfTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.CharTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ByteTensor)

    def test_to_casting(self):
        # 定义一个测试函数，用于测试 to 方法的类型转换的正确性
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            # 创建一个 CPU 上的张量，使用给定的数据
            a_cpu = torch.tensor(data)
            # 将该张量移动到指定的 MPS 设备上
            a_mps = a_cpu.to(torch.device('mps'))

            # 使用 to 方法在 CPU 上进行类型转换，并保存结果
            res_cpu = a_cpu.to(to_dtype)
            # 使用 to 方法在 MPS 设备上进行类型转换，并保存结果
            res_mps = a_mps.to(to_dtype)
            # 断言 CPU 和 MPS 设备上的转换结果应该相等
            self.assertEqual(res_cpu, res_mps)

        # 使用 helper 函数测试不同数据类型的 to 方法类型转换
        helper([9.0, 3.0, 5.0, 4.0], torch.int64)
        helper([9.0, 3.0, 5.0, 4.0], torch.float)
        helper([9.0, 3.0, 5.0, 4.0], torch.int32)
        helper([9.0, 3.0, 5.0, 4.0], torch.short)
        helper([9.0, 3.0, 5.0, 4.0], torch.half)
        helper([9.0, 3.0, 5.0, 4.0], torch.int8)
        helper([9.0, 3.0, 5.0, 4.0], torch.uint8)

    def test_storage_offset_greater_than_src_nbytes(self):
        # 定义一个测试函数，用于测试张量视图创建时的偏移量问题
        # https://github.com/pytorch/pytorch/issues/80844
        # 设置张量个数和每个张量的元素数
        n_tensors = 100
        n_tensor_elems = 784
        # 创建一个包含一系列元素的张量
        elems = torch.arange(n_tensors * n_tensor_elems, dtype=torch.float32)

        tensor_list = []
        # 循环创建连续视图张量的列表（视图张量通过切片操作创建）
        for i in range(0, n_tensors - 1):
            t = elems[n_tensor_elems * i : n_tensor_elems * (i + 1)]
            tensor_list.append(t)

        for i in range(0, n_tensors - 1):
            # 将列表中的每个张量变形为 1 行 n_tensor_elems 列的形状
            t = tensor_list[i].view(1, n_tensor_elems)
            # 将张量移动到 MPS 设备上
            t_mps = t.to("mps")
            # 断言 MPS 设备上的张量应该和 CPU 上的张量相等
            self.assertEqual(t, t_mps.cpu(), f"i={i}")

    # See https://github.com/pytorch/pytorch/issues/82427
    # and https://github.com/pytorch/pytorch/issues/83692
    def test_full_bugs(self):
        # 定义一个测试函数，用于测试 torch.full 方法的 bug
        # Test should not crash
        # 在 MPS 设备上创建一个全是 True 的 3x3 张量
        x = torch.full((3, 3), True, device='mps')
        # 使用 torch.full 方法在 MPS 设备上创建一个 dtype 为 uint8 的 2x2 张量
        y_mps = torch.full((2, 2), 247, device='mps', dtype=torch.uint8)
        # 在 CPU 上创建一个 dtype 为 uint8 的 2x2 张量
        y_cpu = torch.full((2, 2), 247, device='cpu', dtype=torch.uint8)
        # 断言 MPS 设备上的张量应该和 CPU 上的张量相等
        self.assertEqual(y_mps, y_cpu)

    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    # See https://github.com/pytorch/pytorch/issues/84995
    # 测试函数，用于测试 torch.div 函数在不同数据类型和舍入模式下的行为
    def test_div_bugs(self):
        # 对每一种整数类型和舍入模式进行组合测试
        for (dtype, mode) in itertools.product(integral_types(), ['trunc', 'floor']):
            # 排除 torch.int64 类型，因为存在问题
            if dtype != torch.int64:
                # 创建一个张量 x，包含从1到10的整数，设备为 'mps'，指定数据类型为 dtype
                x = torch.tensor(list(range(1, 11)), device='mps', dtype=dtype)
                # 使用 torch.div 函数对 x 进行除法运算，第二个参数为 101，指定舍入模式为 mode
                y = torch.div(x, 101, rounding_mode=mode)
                # 断言 y 的所有元素之和为 0
                self.assertEqual(y.sum(), 0)

    # 参考 https://github.com/pytorch/pytorch/issues/82663
    # 测试函数，用于测试 torch.tensor 的布尔类型张量扩展行为
    def test_bool_expand(self):
        # 创建一个布尔类型的二维张量 x，包含值为 1 和 0
        x = torch.tensor([[1], [0]], dtype=torch.bool, device='mps')
        # 创建一个布尔类型的一维张量 y，包含值 0 和 1
        y = torch.tensor([0, 1], dtype=torch.bool, device='mps')
        # 断言 x 扩展为 2x2 的张量后，与 y 扩展为相同维度的张量不相等
        self.assertFalse(torch.equal(x.expand(2, 2), y.expand(2, 2)))

    # 测试函数，测试空张量的负数运算结果与自身相等
    def test_empty_neg(self):
        # 创建一个空张量 x
        x = torch.tensor([[]], device='mps')
        # 对 x 进行负数运算得到张量 y
        y = -x
        # 断言 x 与 y 相等
        self.assertEqual(x, y)

    # 私有测试函数，测试在不同设备和数据类型下，空标量的唯一性运算结果
    def _test_unique_scalar_empty(self, dtype, device, f):
        # 测试标量情况
        x = torch.tensor(0, dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([0], dtype=dtype, device=device)
        expected_inverse = torch.tensor(0, device=device)
        expected_counts = torch.tensor([1], device=device)
        # 断言计算得到的唯一值、逆向索引和计数与预期值相等
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

        # 测试零大小的张量情况
        x = torch.zeros((0, 0, 3), dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([], dtype=dtype, device=device)
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device=device)
        expected_counts = torch.tensor([], dtype=torch.long, device=device)
        # 断言计算得到的唯一值、逆向索引和计数与预期值相等
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)
    # 定义一个测试方法，用于测试 unique_with_expects 函数的行为
    def _test_unique_with_expects(self, device, dtype, f, x, expected_unique, expected_inverse, expected_counts, additional_shape):
        # 定义一个内部函数，确保输入参数 x 是一个元组
        def ensure_tuple(x):
            # 如果 x 是 torch.Tensor 类型，则返回一个包含 x 的元组
            if isinstance(x, torch.Tensor):
                return (x,)
            # 否则直接返回 x
            return x

        # 遍历 return_inverse 和 return_counts 的可能取值
        for return_inverse in [True, False]:
            for return_counts in [True, False]:
                # 调用被测试的函数 f，并确保返回值的长度符合预期
                ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                # 检查返回的 unique 值是否与预期一致
                self.assertEqual(expected_unique, ret[0])
                # 如果 return_inverse 为 True，则检查返回的 inverse 是否与预期一致
                if return_inverse:
                    self.assertEqual(expected_inverse, ret[1])
                # 如果 return_counts 为 True，则检查返回的 counts 是否与预期一致
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    self.assertEqual(expected_counts, ret[count_index])

                # 在更高维度的张量上测试每个元素的唯一性
                y = x.view(additional_shape)
                # 调用 f 函数，并要求返回 inverse 和 counts
                y_unique, y_inverse, y_counts = f(y, return_inverse=True, return_counts=True)
                # 检查返回的 unique 值是否与预期一致
                self.assertEqual(expected_unique, y_unique)
                # 检查返回的 inverse 是否与预期一致，并且保持与 additional_shape 相同的视图
                self.assertEqual(expected_inverse.view(additional_shape), y_inverse)
                # 检查返回的 counts 是否与预期一致
                self.assertEqual(expected_counts, y_counts)

    # 定义一个测试方法，用于测试 unique 函数的行为
    def test_unique(self):
        # 定义一个内部函数 helper，用于调用 unique 函数并比较其结果
        def helper(x, return_inverse, return_counts):
            # 将输入的张量 cpu_x 赋值给 x，并转移到 'mps' 设备
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            # 调用 torch.unique 函数获取结果
            result = torch.unique(x, return_inverse=return_inverse, return_counts=return_counts)
            # 在 CPU 上调用 torch.unique 函数获取结果
            result_cpu = torch.unique(cpu_x, return_inverse=return_inverse, return_counts=return_counts)

            # 断言两个结果是否相等
            self.assertEqual(result, result_cpu)

        # 对 helper 函数进行多组测试
        helper(torch.tensor([1, 2, 4, 2, 1]), False, False)
        helper(torch.randint(3, (10, )), False, False)
        helper(torch.randint(3, (10, )), True, False)
        helper(torch.randint(3, (10, )), False, True)
        helper(torch.randint(3, (10, )), True, True)
        helper(torch.randint(3, (1, )), True, True)
        helper(torch.randint(3, (0, )), True, True)

        # 对 https://github.com/pytorch/pytorch/issues/104879 的回归测试
        # 创建一个长度为 2 的张量 x，将其 reshape 成 (1, 1, 2)，然后对其进行 unique 操作并断言结果
        x = torch.arange(2, device="mps")
        self.assertEqual(x.reshape(1, 1, 2).unique(), x)
    # 测试函数：test_unique_consecutive
    def test_unique_consecutive(self):
        # 嵌套函数：helper，用于辅助测试不同的输入情况
        def helper(x, dim, return_inverse, return_counts):
            # 备份输入的张量到 cpu_x，并转换为 'mps' 内存格式的张量 x
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            # 使用 torch.unique_consecutive 函数处理张量 x，并返回结果
            result = torch.unique_consecutive(x, dim=dim, return_inverse=return_inverse, return_counts=return_counts)
            # 使用相同的参数在 CPU 上处理张量 cpu_x，作为对比结果
            result_cpu = torch.unique_consecutive(cpu_x, dim=dim, return_inverse=return_inverse, return_counts=return_counts)

            # 断言两个处理结果应该相等
            self.assertEqual(result, result_cpu)

        # 测试不同的输入情况
        helper(torch.tensor([1, 2, 4, 2, 1]), 0, False, False)
        helper(torch.randint(3, (10, )), 0, False, False)
        helper(torch.randint(3, (10, )), 0, True, False)
        helper(torch.randint(3, (10, )), 0, False, True)
        helper(torch.randint(3, (10, )), 0, True, True)
        helper(torch.randint(3, (10, )), 0, True, True)
        helper(torch.randint(3, (1, )), 0, True, True)
        helper(torch.randint(3, (0, )), 0, True, True)

        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 0, False, False)
        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 0, True, True)
        helper(torch.randint(2, (20, 2)), 0, True, True)
        helper(torch.randint(2, (1, 2)), 0, True, True)
        helper(torch.randint(2, (0, 2)), 0, True, True)

        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 1, False, False)
        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 1, True, True)
        helper(torch.randint(2, (2, 20)), 1, True, True)
        helper(torch.randint(2, (2, 1)), 1, True, True)
        helper(torch.randint(2, (2, 0)), 1, True, True)

    # 测试函数：test_cat_non_contiguous
    # 查看 https://github.com/pytorch/pytorch/issues/85675
    def test_cat_non_contiguous(self):
        # 嵌套函数：rotate_subset，用于从数据中旋转子集
        def rotate_subset(data, dim):
            # 从数据中选择前两个维度的子集 x1 和后两个维度的子集 x2
            x1 = data[:, :, :2, :]
            x2 = data[:, :, 2:, :]
            # 断言 x1 和 x2 都不是连续的内存布局
            self.assertFalse(x1.is_contiguous())
            self.assertFalse(x2.is_contiguous())
            # 沿指定维度 dim 连接 x1 和 x2，并返回结果
            return torch.concat((x1, x2), dim=dim)

        # 对不同的数据类型进行测试
        for dtype in MPS_DTYPES:
            # 排除布尔类型的测试
            if dtype == torch.bool:
                continue
            # 创建数据张量 data，形状为 (1, 2, 4, 6)，并设置为 channels_last 内存格式
            data = torch.arange(48, dtype=dtype).reshape(1, 2, 4, 6)
            data = data.to(memory_format=torch.channels_last)
            # 将数据张量转换为 'mps' 内存格式，并进行断言验证
            mps_data = data.to("mps")
            self.assertEqual(data, mps_data)
            # 对数据张量在各个维度上进行旋转，并比较结果
            for dim in range(data.dim()):
                cpu_result = rotate_subset(data, dim)
                mps_result = rotate_subset(mps_data, dim)
                self.assertEqual(cpu_result, mps_result.to("cpu"))
                # TODO: 启用内存格式测试
                # self.assertEqual(cpu_result.is_contiguous(), mps_result.is_contiguous())

    # 测试函数：test_from_numpy_non_contiguous
    # 查看 https://github.com/pytorch/pytorch/issues/85967
    def test_from_numpy_non_contiguous(self):
        # 使用 NumPy 创建二维数组 a，选择其中的前两列作为初始数据
        a = np.arange(9).reshape(3, 3)[:, :2]
        # 创建 CPU 上的张量 t_cpu，并 'mps' 内存格式的张量 t_mps
        t_cpu = torch.tensor(a, device="cpu")
        t_mps = torch.tensor(a, device="mps")
        # 断言 CPU 上的张量与 'mps' 内存格式的张量相等
        self.assertEqual(t_cpu, t_mps.to("cpu"))

    # 查看 https://github.com/pytorch/pytorch/issues/86954
    # 该测试功能尚未实现
    # 测试非连续拷贝的情况
    def test_copy_non_contiguous(self):
        # 创建一个3x3x3的张量，并按照指定顺序重新排列
        x = torch.arange(27).reshape(3, 3, 3).permute(2, 0, 1)
        # 检查张量是否连续存储
        self.assertFalse(x.is_contiguous())
        # 将张量移动到MPS（分布式张量存储）上
        y = x.to('mps')
        # 检查移动后的张量是否连续存储
        self.assertFalse(y.is_contiguous())
        # 比较两个张量是否相等，需要先将y移回CPU上
        self.assertEqual(x, y.to('cpu'))

        # 创建一个4x4x4的张量，并按照指定顺序重新排列，再进行切片操作
        x = torch.arange(4**3).reshape(4, 4, 4).permute((2, 0, 1))[1:, ::2]
        # 将x移动到MPS上
        y = x.to('mps')
        # 比较两个张量是否相等，需要先将y移回CPU上
        self.assertEqual(x, y.to('cpu'))

        # 创建一个4x4x4x4的张量，并填充为固定值，指定设备为CPU
        x = torch.full((4, 4, 4, 4), 13, device="cpu")
        # 创建一个与x相同形状的张量，填充为固定值，指定设备为MPS
        y = torch.full((4, 4, 4, 4), 13, device="mps")
        # 创建一个4x4x4x4的张量，并按照指定顺序重新排列，再进行切片操作
        z = torch.arange(4**4).reshape(4, 4, 4, 4).permute(3, 2, 0, 1)[1::, ::2]
        # 在CPU上对x进行重新排列，并进行切片操作，然后将z赋值给切片后的部分
        x.permute(3, 2, 1, 0)[1::, ::2] = z
        # 由于y在MPS上，z在CPU上，这里会调用复制操作符进行拷贝
        y.permute(3, 2, 1, 0)[1::, ::2] = z
        # 比较两个张量是否相等，需要先将y移回CPU上
        self.assertEqual(x, y.to('cpu'))

    # 参见https://github.com/pytorch/pytorch/issues/95417
    # 测试拷贝时的存储偏移
    def test_copy_storage_offset(self):
        # 在CPU上创建一个全为0的张量
        x_cpu = torch.zeros(5, device="cpu", dtype=torch.float32)
        # 在MPS上创建一个全为0的张量
        x_mps = torch.zeros(5, device="mps", dtype=torch.float32)
        # 创建一个在CPU上的张量，并赋予部分值
        update_cpu = torch.tensor([1, 1], device="cpu", dtype=torch.int64)
        # 创建一个在MPS上的张量，并赋予部分值，这里会发生隐式类型转换和拷贝
        update_mps = torch.tensor([1, 1], device="mps", dtype=torch.int64)
        # 将update_cpu的值复制到x_cpu的指定位置
        x_cpu[2:4] = update_cpu
        # 将update_mps的值复制到x_mps的指定位置，这里会发生隐式设备移动和拷贝
        x_mps[2:4] = update_mps
        # 比较两个张量是否相等
        self.assertEqual(x_cpu, x_mps)

        # 将update_mps的值复制到x_cpu的指定位置，这里会发生隐式设备移动和拷贝
        x_cpu[2:4] = update_mps
        # 比较两个张量是否相等
        self.assertEqual(x_cpu, x_mps)

    # 测试广播拷贝的情况
    def test_copy_broadcasting(self):
        # 定义一个辅助函数，用于测试不同形状和数据类型的拷贝
        def helper(src_shape, dst_shape, src_dtype, dst_dtype):
            # 在CPU上创建一个指定形状和数据类型的随机整数张量
            cpu_src = torch.randint(0, 127, src_shape).to(src_dtype)
            # 在CPU上创建一个指定形状和数据类型的随机整数张量
            cpu_dst = torch.randint(0, 127, dst_shape).to(dst_dtype)
            # 在CPU上进行就地复制操作，并记录结果
            cpu_result = cpu_dst.copy_(cpu_src)
            # 将cpu_src移动到MPS上
            mps_src = cpu_src.to("mps")
            # 将cpu_dst移动到MPS上
            mps_dst = cpu_dst.to("mps")
            # 在MPS上进行就地复制操作，并记录结果
            mps_result = mps_dst.copy_(mps_src)
            # 比较两个结果是否相等
            self.assertEqual(cpu_result, mps_result)

        # 测试所有可能的数据类型组合
        test_dtypes = [torch.float32, torch.int32, torch.int16, torch.int8]

        # 遍历所有数据类型组合，并调用helper函数进行测试
        for (src_dtype, dst_dtype) in itertools.product(test_dtypes, test_dtypes):
            helper((2, 1), (2, 3), src_dtype, dst_dtype)
            helper((2, 1), (2, 2), src_dtype, dst_dtype)
            helper((3, 1, 4, 1), (3, 4, 4, 5), src_dtype, dst_dtype)
            helper((3,), (2, 3), src_dtype, dst_dtype)
            helper((2,), (2, 2), src_dtype, dst_dtype)
            helper((4, 1, 5), (3, 4, 4, 5), src_dtype, dst_dtype)
            helper((4, 1, 5), (4, 0, 5), src_dtype, dst_dtype)
            helper((1, 5), (4, 0, 5), src_dtype, dst_dtype)
            helper((3, 1, 0), (3, 5, 0), src_dtype, dst_dtype)
            helper((0, 1, 0), (0, 5, 0), src_dtype, dst_dtype)

        # 回归测试https://github.com/pytorch/pytorch/issues/107867
        # 检查单个张量在MPS上的值是否正确
        self.assertEqual(torch.tensor([[1]], device='mps').item(), 1.0)

    # 参见https://github.com/pytorch/pytorch/pull/84742
    # 和https://github.com/pytorch/pytorch/pull/78319
    def test_nansum(self):
        def helper(dtype, noncontiguous, dim):
            zero_cpu = torch.zeros((), dtype=dtype)

            # 随机缩放值
            scale = random.randint(10, 100)
            # 创建一个在指定设备上的张量，具有指定的数据类型和非连续性
            x_cpu: torch.Tensor = make_tensor(
                (5, 5), dtype=dtype, device='cpu',
                low=-scale, high=scale, noncontiguous=noncontiguous)

            if dtype.is_floating_point:
                # 创建一个布尔掩码以标识 NaN 值的位置
                nan_mask_cpu = x_cpu < (0.2 * scale)
                # 创建一个不含 NaN 值的张量
                x_no_nan_cpu = torch.where(nan_mask_cpu, zero_cpu, x_cpu)
                # 将 x_cpu 中的一部分值设置为 NaN
                x_cpu[nan_mask_cpu] = np.nan
            else:
                x_no_nan_cpu = x_cpu

            # 将 x_cpu 转换到 'mps' 设备
            x_mps = x_cpu.to('mps')
            # 创建一个空张量以存储计算结果
            actual_out_mps = torch.empty(0, dtype=dtype, device='mps')
            expect_out_cpu = torch.empty(0, dtype=dtype)
            dim_kwargs = {"dim": dim} if dim is not None else {}
            # 计算预期的总和值
            expect = torch.sum(x_no_nan_cpu, **dim_kwargs)

            # 使用 torch.nansum 在 CPU 上计算实际的总和值
            actual_cpu = torch.nansum(x_cpu, **dim_kwargs)
            # 在 CPU 上进行一致性检查
            self.assertEqual(expect, actual_cpu)

            # 在 'mps' 设备上使用 torch.nansum 计算实际的总和值
            actual_mps = torch.nansum(x_mps, **dim_kwargs)
            # 使用 out= 变体计算 'mps' 设备上的总和值
            torch.nansum(x_mps, out=actual_out_mps, **dim_kwargs)
            torch.nansum(x_cpu, out=expect_out_cpu, **dim_kwargs)
            # 进行一致性检查
            self.assertEqual(expect, actual_mps)
            self.assertEqual(expect_out_cpu, actual_out_mps)

        # 为每一组参数调用 helper 函数进行测试
        args = itertools.product(
            (torch.float16, torch.float32, torch.int32, torch.int64),   # dtype
            (True, False),                                              # noncontiguous
            (0, 1, None),                                               # dim
        )

        for dtype, noncontiguous, dim in args:
            with self.subTest(dtype=dtype, noncontiguous=noncontiguous, dim=dim):
                helper(dtype, noncontiguous, dim)

    def test_cumsum_all_dtypes(self):
        def helper(dtype):
            # 创建一个在 'mps' 设备上的张量，并进行累积求和操作
            t = torch.tensor([1, 1, 1, 1], device="mps", dtype=dtype)
            # 创建一个在 'cpu' 设备上的张量，并进行累积求和操作
            t_cpu = torch.tensor([1, 1, 1, 1], device="cpu")

            # 在 'mps' 设备上进行累积求和操作，并存储结果到 a
            a = t.cumsum(0, dtype=dtype)
            # 在 'cpu' 设备上进行累积求和操作，并存储结果到 a_cpu
            a_cpu = t_cpu.cumsum(0, dtype=dtype)

            # 进行结果的一致性检查
            self.assertEqual(a.cpu(), a_cpu)
        # 对于每种数据类型调用 helper 函数进行测试
        [helper(dtype) for dtype in [torch.int8, torch.int16, torch.int32, torch.float32]]

        try:
            # 测试是否抛出了预期的异常
            helper(torch.int64)
        except Exception as e:
            e_string = str(e)
            # 检查异常消息是否符合预期
            self.assertEqual(e_string, "MPS does not support cumsum_out_mps op with int64 input." +
                             " Support has been added in macOS 13.3")

    def test_cumsum_bool(self):
        # 创建一个包含大量 True 值的布尔张量
        a = torch.ones(2**16, dtype=torch.bool)
        # 在 'cpu' 设备上进行累积求和操作
        t_cpu = a.cumsum(0)
        # 将张量转换到 'mps' 设备上，并进行累积求和操作
        t_mps = a.to("mps").cumsum(0)

        # 进行结果的一致性检查
        self.assertEqual(t_cpu, t_mps)
    def test_cumsum_minus_one_axis(self):
        def helper(dtype):
            # Test with axis -1
            # 根据数据类型选择生成随机数据或整数随机数据
            cpu_x = None
            if dtype == torch.float32:
                cpu_x = torch.randn(10, 3, device='cpu', dtype=torch.float32)
            else:
                cpu_x = torch.randint(0, 20, (10, 3), device='cpu', dtype=torch.float32)
            # 将 CPU 上的数据拷贝到 MPS 设备上
            x = cpu_x.detach().clone().to('mps')

            # 在 CPU 上计算累积和
            cpu_y = cpu_x.cumsum(-1)
            # 在 MPS 设备上计算累积和
            y = x.cumsum(-1)

            # 断言 MPS 设备上的计算结果与 CPU 上一致
            self.assertEqual(y, cpu_y)

        # 对不同数据类型调用 helper 函数进行测试
        [helper(dtype) for dtype in [torch.float32, torch.int16, torch.int32, torch.uint8]]

    def test_cumprod_all_dtypes(self):
        def helper(dtype):
            # 创建 MPS 设备上的张量
            t = torch.tensor([1, 1, 1, 1], device="mps", dtype=dtype)
            # 创建 CPU 上的张量
            t_cpu = torch.tensor([1, 1, 1, 1], device="cpu")

            # 在 MPS 设备上计算累积积
            a = t.cumprod(0, dtype=dtype)
            # 在 CPU 上计算累积积
            a_cpu = t_cpu.cumprod(0, dtype=dtype)

            # 断言 MPS 设备上的计算结果与 CPU 上一致
            self.assertEqual(a.cpu(), a_cpu)

        # 对不同数据类型调用 helper 函数进行测试
        [helper(dtype) for dtype in [torch.int8, torch.int16, torch.int32, torch.float32]]

        # 捕获 int64 类型的异常测试
        try:
            helper(torch.int64)
        except Exception as e:
            e_string = str(e)
            # 断言异常信息是否符合预期
            self.assertEqual(e_string, "MPS does not support cumprod_out_mps op with int64 input."
                             + " Support has been added in macOS 13.3")

    def test_cumprod_minus_one_axis(self):
        def helper(dtype):
            # Test with axis -1
            # 根据数据类型选择生成随机数据或整数随机数据
            cpu_x = None
            if dtype == torch.float32:
                cpu_x = torch.randn(10, 3, device='cpu', dtype=torch.float32)
            else:
                cpu_x = torch.randint(0, 20, (10, 3), device='cpu', dtype=torch.float32)
            # 将 CPU 上的数据拷贝到 MPS 设备上
            x = cpu_x.detach().clone().to('mps')

            # 在 CPU 上计算累积乘积
            cpu_y = cpu_x.cumprod(-1)
            # 在 MPS 设备上计算累积乘积
            y = x.cumprod(-1)

            # 断言 MPS 设备上的计算结果与 CPU 上一致
            self.assertEqual(y, cpu_y)

        # 对不同数据类型调用 helper 函数进行测试
        [helper(dtype) for dtype in [torch.float32, torch.int16, torch.int32, torch.uint8]]

    def test_median_int16(self):
        def helper(shape, dtype):
            # 在 CPU 上生成指定形状和数据类型的随机整数数据
            cpu_x = torch.randint(-9999, 9999, shape, device='cpu', dtype=dtype)
            # 将 CPU 上的数据拷贝到 MPS 设备上
            x = cpu_x.detach().clone().to('mps')

            # 在 MPS 设备上计算中位数
            median_result = torch.median(x)
            # 在 CPU 上计算中位数
            median_result_cpu = torch.median(cpu_x)
            
            # 断言 MPS 设备上的计算结果与 CPU 上一致
            self.assertEqual(median_result, median_result_cpu)

        # 调用 helper 函数进行测试，传入指定的形状和数据类型
        helper((2, 8, 4, 5), torch.int16)

    def test_activation_checkpoint_does_not_error(self):
        from torch.utils.checkpoint import checkpoint

        # 循环测试是否能正常使用 activation checkpoint
        for use_reentrant in (True, False):
            # 创建一个 MPS 设备上需要梯度的张量
            a = torch.tensor(1., device="mps", requires_grad=True)

            # 定义一个简单的函数
            def fn(x):
                return x.sin().cos().exp()

            # 使用 activation checkpoint 运行函数
            out = checkpoint(fn, a, use_reentrant=use_reentrant)
            # 对输出进行反向传播
            out.backward()
    # 测试 torch.as_strided() 函数的功能
    def test_as_strided(self):
        # 创建一个包含浮点数的二维列表
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        # 创建另一个二维列表
        values_1 = [[1.0, 1.0], [1.0, 1.0]]
        # 在 CPU 设备上创建张量
        cpu_x = torch.tensor(values, device='cpu')
        # 在自定义设备 'mps' 上创建张量
        ones1 = torch.tensor(values_1, device='mps')
        # 对 cpu_x 进行分离、克隆、转换，并设置梯度跟踪
        x = cpu_x.detach().clone().to('mps').requires_grad_()
        # 使用 torch.as_strided 创建一个步长视图，应用于 cpu_x
        strided_cpu = torch.as_strided(cpu_x, (2, 2), (1, 2))
        # 使用 torch.as_strided 创建一个步长视图，应用于 x
        strided_mps = torch.as_strided(x, (2, 2), (1, 2))
        # 断言两个步长视图是否相等
        self.assertEqual(strided_mps, strided_cpu)
        # 对步长视图 strided_cpu 加上 ones1，并转回 'cpu' 设备
        strided_cpu_out = strided_cpu + ones1.to('cpu')
        # 对步长视图 strided_mps 加上 ones1
        strided_mps_out = strided_mps + ones1
        # 断言两个加法结果是否相等
        self.assertEqual(strided_cpu_out, strided_mps_out)

        # 带有存储偏移的测试
        # 在 CPU 上创建一个随机张量
        cpu_x = torch.rand(3, 3, device='cpu')
        # 将 cpu_x 转换到 'mps' 设备
        mps_x = cpu_x.to('mps')
        # 使用带有存储偏移的 torch.as_strided 创建步长视图
        strided_cpu1 = torch.as_strided(cpu_x, (2, 2), (1, 2), 0)
        strided_mps1 = torch.as_strided(mps_x, (2, 2), (1, 2), 0)
        strided_cpu2 = torch.as_strided(cpu_x, (2, 2), (1, 2), 1)
        strided_mps2 = torch.as_strided(mps_x, (2, 2), (1, 2), 1)
        # 计算两个步长视图的减法结果
        strided_cpu_out = strided_cpu1 - strided_cpu2
        strided_mps_out = strided_mps1 - strided_mps2
        # 断言两个减法结果是否相等
        self.assertEqual(strided_cpu_out, strided_mps_out)

    # 测试 torch.unfold() 函数的功能
    def test_unfold(self):
        # 在默认设备上创建张量 x
        x = torch.arange(1., 8)
        # 在 'mps' 设备上创建张量 x_mps
        x_mps = torch.arange(1., 8, device="mps")

        # 使用 unfold 函数展开张量 x
        y = x.unfold(0, 2, 1)
        # 使用 unfold 函数展开张量 x_mps
        y_mps = x_mps.unfold(0, 2, 1)

        # 断言展开结果是否相等
        self.assertEqual(y, y_mps)

    # 测试 torch.unfold() 在所有设备和数据类型上的功能
    def test_unfold_all_devices_and_dtypes(self):
        # 支持的数据类型列表
        supported_dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.uint8]
        # 遍历每种数据类型
        for dt in supported_dtypes:
            # 在 'mps' 设备上创建一个空张量 x，指定数据类型为 dt
            x = torch.empty((0, 1, 3, 0), dtype=dt, device="mps")
            # 断言 unfold 操作后的张量形状是否符合预期
            self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)

    # 测试 torch.bincount() 函数的简单功能
    def test_bincount_simple(self):
        # 在 'mps' 设备上创建一个随机整数张量 input
        input = torch.randint(0, 8, (5,), dtype=torch.int32, device="mps")
        # 将 input 转回 'cpu' 设备
        input_cpu = input.to("cpu")
        # 在 'mps' 设备上创建一个权重张量 weights
        weights = torch.linspace(0, 1, steps=5, device="mps", dtype=torch.float32)
        # 将 weights 转回 'cpu' 设备
        weights_cpu = weights.to("cpu")

        # 计算 input 的 bincount 结果 x
        x = torch.bincount(input)
        # 计算 input_cpu 的 bincount 结果 x_cpu
        x_cpu = torch.bincount(input_cpu)
        # 断言两者是否相等
        self.assertEqual(x, x_cpu)

        # 使用带权重的 bincount 计算结果 y
        y = input.bincount(weights)
        # 使用带权重的 bincount 计算结果 y_cpu
        y_cpu = input_cpu.bincount(weights_cpu)
        # 断言两者是否相等
        self.assertEqual(y, y_cpu)
    # 定义一个测试方法，用于测试 torch 的 bincount 函数
    def test_bincount(self):
        # 设定设备类型为 "mps"
        device = "mps"
        # 定义输入张量的大小
        input_size = (5000,)
        # 生成一个在指定设备上的随机张量 w，并且数据类型为 float
        w = torch.randn(input_size, dtype=torch.float, device=device)
        # 将 w 复制到 CPU 上
        w_cpu = w.cpu()

        # 生成一个在指定设备上的随机整数张量 t，数据类型为 int8
        t = torch.randint(50, input_size, dtype=torch.int8, device=device)
        # 检查在 CPU 上计算的 t 的 bincount 结果是否与在当前设备上计算的一致
        self.assertEqual(t.cpu().bincount(), t.bincount())
        # 检查在 CPU 上计算的 t 的 bincount 结果是否与在 w_cpu 上计算的一致
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        # 生成一个在指定设备上的随机整数张量 t，数据类型为 int32
        t = torch.randint(500, input_size, dtype=torch.int32, device=device)
        # 同样进行 bincount 的比较
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        # 生成一个在指定设备上的随机整数张量 t，数据类型为 int32
        t = torch.randint(2000, input_size, dtype=torch.int32, device=device)
        # 同样进行 bincount 的比较
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        # 生成一个长度为 10 的全零整数张量 t，数据类型为 int32，设备为指定设备
        t = torch.zeros([10], dtype=torch.int32, device=device)
        # 设置 t 的第一个元素为 35488
        t[0] = 35488
        # 对 t 进行 bincount，同时指定最小长度为 65536
        counted = t.bincount(minlength=65536)
        # 检查 counted 中所有元素之和是否等于 10
        self.assertEqual(torch.sum(counted), 10)

    # 定义一个测试方法，用于测试 torch 的 sum 函数的反向传播
    def test_sum_backward(self):
        # 定义一个辅助函数 helper，接受两个参数 n 和 c
        def helper(n, c):
            # 定义一个二维列表 values
            values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            # 在 CPU 上创建一个张量 cpu_x，并且启用梯度跟踪
            cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
            # 将 cpu_x 分离出来，并克隆到 'mps' 设备上，同时启用梯度跟踪
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算所有元素的总和 all_sum
            all_sum = torch.sum(x)
            # 计算在 CPU 上所有元素的总和 all_sum_cpu
            all_sum_cpu = torch.sum(cpu_x)

            # 执行反向传播
            all_sum.backward()
            all_sum_cpu.backward()

            # 检查 all_sum 是否等于 all_sum_cpu
            self.assertEqual(all_sum, all_sum_cpu)
            # 检查 x 的梯度是否等于 cpu_x 的梯度
            self.assertEqual(x.grad, cpu_x.grad)

        # 调用 helper 函数，参数为 3 和 3
        helper(3, 3)

    # 定义一个测试方法，用于测试 torch 的 L1Loss
    # 参数 shape 表示张量的形状，reduction 表示损失函数的降维方式
    def test_l1_loss(self):
        # 定义一个辅助函数 helper，接受两个参数 shape 和 reduction
        def helper(shape, reduction):
            # 创建 L1 损失函数对象 loss
            loss = torch.nn.L1Loss(reduction=reduction)

            # 在 CPU 上生成一个随机张量 inputCPU，并启用梯度跟踪
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 在 CPU 上生成一个随机张量 targetCPU，并不启用梯度跟踪
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将 inputCPU 分离出来，并克隆到 'mps' 设备上，同时启用梯度跟踪
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            # 将 targetCPU 分离出来，并克隆到 'mps' 设备上，不启用梯度跟踪
            targetMPS = targetCPU.detach().clone().to('mps')

            # 前向传播计算损失值
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            # 检查在 CPU 和 'mps' 设备上计算的损失值是否一致
            self.assertEqual(outputCPU, outputMPS)

            # 如果 reduction 不为 'none'，则进行反向传播
            if reduction != 'none':
                # 选择 2 作为梯度输出值，确保在反向传播时梯度输出大于 1
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                # 检查在 CPU 和 'mps' 设备上计算的梯度是否一致
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 调用 helper 函数，分别测试不同的 shape 和 reduction 组合
        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')
    # 定义一个测试函数，用于测试均方误差损失函数
    def test_mse_loss(self):
        # 定义一个内部辅助函数，接受形状和减少方式作为参数
        def helper(shape, reduction):
            # 创建均方误差损失函数对象，指定减少方式
            loss = torch.nn.MSELoss(reduction=reduction)

            # 在 CPU 上生成随机输入和目标张量，需要梯度计算
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)

            # 将 CPU 上的输入张量克隆到 MPS 设备上，并且需要梯度计算
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # 前向传播
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)

            # 断言 CPU 和 MPS 设备上的输出值相等
            self.assertEqual(outputCPU, outputMPS)

            # 反向传播
            if reduction != 'none':
                # 如果不是 'none' 减少方式，使用梯度值为2进行反向传播
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))

                # 断言 CPU 和 MPS 设备上的输入梯度相等
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 使用 helper 函数测试不同的输入形状和减少方式
        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')

        # 验证改变形状是否会导致缓存图查找问题
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')

    # 定义测试均方误差损失函数输出为跨步的输出
    def test_mse_loss_strided_output(self):
        # 引用 GitHub 上的问题链接，说明此处解决了特定问题
        # 创建均方误差损失函数对象，减少方式为 'none'
        lf = nn.MSELoss(reduction='none')

        # 创建一个包含单个卷积层的模型，在 CPU 上
        model_cpu = nn.Sequential(
            nn.Conv1d(3, 3, 1),
        )

        # 深拷贝 CPU 上的模型到 MPS 设备上
        model_mps = copy.deepcopy(model_cpu).to("mps")

        # 生成一个大小为 (128, 10, 3) 的随机输入张量 x
        x = torch.randn(128, 10, 3)

        # 对输入张量进行维度变换，调整顺序
        x = x.permute(0, 2, 1)

        # 将 CPU 上的输入张量克隆到 MPS 设备上，并且调整顺序
        x_mps = x.detach().clone().to("mps").permute(0, 2, 1)

        # CPU 和 MPS 设备上分别对输入张量进行模型前向传播
        y = model_cpu(x)
        y_mps = model_mps(x_mps)

        # 调整输出张量的维度和大小，仅保留前 5 个元素
        y = y.permute(0, 2, 1)[:, :5, :]
        y_mps = y_mps.permute(0, 2, 1)[:, :5, :]

        # 生成一个大小为 (128, 5, 3) 的随机预测张量 y_hat
        y_hat = torch.randn(128, 5, 3)

        # 将预测张量 y_hat 克隆到 MPS 设备上
        y_hat_mps = y_hat.detach().clone().to("mps")

        # 使用均方误差损失函数计算 CPU 和 MPS 设备上的损失值
        loss = lf(y, y_hat)
        loss_mps = lf(y_mps, y_hat_mps)

        # 断言 CPU 和 MPS 设备上的损失值相等
        self.assertEqual(loss, loss_mps)
    # 定义一个测试 BCE Loss 简单情形的方法
    def test_bce_loss_simple(self):
        # 定义一个辅助函数，用于测试不同形状和减少方式的 BCE Loss

        # 创建 BCELoss 标准，指定减少方式
        loss = torch.nn.BCELoss(reduction=reduction)

        # 输入和目标值必须在 [0..1] 范围内
        input_t = np.random.random_sample(size=shape).astype(np.float32)
        target_t = np.random.random_sample(size=shape).astype(np.float32)
        # 创建在 CPU 上的输入和目标张量
        inputCPU = torch.tensor(input_t, device='cpu', dtype=torch.float, requires_grad=True)
        targetCPU = torch.tensor(target_t, device='cpu', dtype=torch.float, requires_grad=False)
        # 创建 inputCPU 的 MPS 克隆，并且需要梯度
        inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
        # 创建 targetMPS 的 MPS 克隆
        targetMPS = targetCPU.detach().clone().to('mps')

        # 前向传播
        outputCPU = loss(inputCPU, targetCPU)
        outputMPS = loss(inputMPS, targetMPS)
        # 断言 CPU 和 MPS 上的输出是否相等
        self.assertEqual(outputCPU, outputMPS)

        # 反向传播
        if reduction != 'none':
            # 选择梯度为 0.6，以确保 grad_output != 1
            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
            # 断言 inputCPU 和 inputMPS 的梯度是否相等
            self.assertEqual(inputCPU.grad, inputMPS.grad)

    # 测试 BCE Loss 总是非负的情况
    def test_bce_loss_always_nonnegative(self):
        # 创建全为 1 的目标张量和输入张量在 MPS 上
        target = torch.ones(5, device='mps')
        input = torch.ones(5, device='mps')
        # 断言 BCE Loss 输出是否全大于 0
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        # 创建全为 0 的目标张量和输入张量在 MPS 上
        target = torch.zeros(5, device='mps')
        input = torch.zeros(5, device='mps')
        # 断言 BCE Loss 输出是否全大于 0
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    # 测试 BCE Loss 大小不匹配的情况
    def test_bce_loss_size_mismatch(self):
        # 创建一个 BCE Loss 对象
        bceloss = nn.BCELoss()
        # 创建大小不匹配的张量 a 和 b 在 MPS 上
        a = torch.rand(25, device='mps')
        b = torch.rand(25, 1, device='mps')
        # 断言使用不匹配大小的目标会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, r'Using a target size \('):
            bceloss(a, b)
    # 定义一个测试方法，用于验证 BCEWithLogitsLoss 在大张量上的行为与 sigmoid 结合 BCELoss 的结果一致性，并且可以处理梯度
    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self):
        # 定义输入张量的大小
        x_size = 1024
        y_size = 256
        # 生成目标张量，随机初始化在 'mps' 设备上
        target = torch.rand(x_size, y_size, device='mps')

        # 针对不同的 reduction 模式进行循环测试
        for reduction in ['none', 'mean', 'sum']:
            # 创建输出张量，初始化为在 'mps' 设备上的随机值，范围在 [-0.5, 0.5)
            output_sig = torch.rand(x_size, y_size, device='mps') - 0.5
            # 克隆输出张量并分离计算图
            output_logits = output_sig.clone().detach()

            # 设置输出张量需要计算梯度
            output_sig.requires_grad = True
            output_logits.requires_grad = True
            # 生成权重张量，初始化为在 'mps' 设备上的随机值
            weight = torch.rand(y_size, device='mps')

            # 使用 BCELoss 计算基于 sigmoid 的损失
            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                torch.sigmoid(output_sig), target
            )
            # 使用 BCEWithLogitsLoss 计算基于 logits 的损失
            loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                output_logits, target
            )

            # 断言两种损失的计算结果应该一致
            self.assertEqual(loss_logits, loss_sig)

            # 根据 reduction 模式处理梯度计算
            if reduction == 'none':
                # 随机生成梯度张量
                grad = torch.rand(x_size, y_size, device='mps')
                # 分别对两种损失进行反向传播
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                # 对两种损失进行默认的反向传播
                loss_sig.backward()
                loss_logits.backward()

            # 断言输出张量的梯度应该一致
            self.assertEqual(output_sig.grad, output_logits.grad)

    # 验证 BCEWithLogitsLoss 在零输出值处的梯度是否正确
    def test_bce_with_logits_has_correct_grad_at_zero(self):
        # 创建零输出的张量，需要计算梯度，位于 'mps' 设备上
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        # 创建目标张量为零，位于 'mps' 设备上
        target = torch.zeros(3, 1, device='mps')
        # 计算 BCEWithLogitsLoss 的损失并进行反向传播
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        # 期望的梯度应该是填充值为 0.5 的张量
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        # 断言输出张量的梯度应该与期望的梯度一致
        self.assertEqual(output.grad, expected_grad)

    # 验证 BCEWithLogitsLoss 是否正确广播权重
    def test_bce_with_logits_broadcasts_weights(self):
        # 创建目标张量，位于 'mps' 设备上
        target = torch.rand(16, 4, device='mps')
        # 创建输出张量，初始化为在 'mps' 设备上的随机值，范围在 [-0.5, 0.5)
        output = torch.rand(16, 4, device='mps') - 0.5

        # 创建权重张量，初始化为在 'mps' 设备上的随机值
        weight = torch.rand(4, device='mps')
        # 使用不同的权重计算 BCEWithLogitsLoss 的损失
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 将权重张量扩展到与输出张量相同的大小，并确保连续性
        weight = weight.expand(16, 4).contiguous()
        # 使用扩展后的权重计算 BCEWithLogitsLoss 的损失
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 断言两种方式计算的损失应该一致
        self.assertEqual(out1, out2)

        # 创建权重张量，初始化为在 'mps' 设备上的随机值
        weight = torch.rand(16, 1, device='mps')
        # 使用不同的权重计算 BCEWithLogitsLoss 的损失
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 将权重张量扩展到与输出张量相同的大小，并确保连续性
        weight = weight.expand(16, 4).contiguous()
        # 使用扩展后的权重计算 BCEWithLogitsLoss 的损失
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        # 断言两种方式计算的损失应该一致
        self.assertEqual(out1, out2)

    # 验证 BCEWithLogitsLoss 中使用正权重的情况是否与未指定权重的结果一致
    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        # 创建目标张量，位于 'mps' 设备上
        target = torch.rand(64, 4, device='mps')
        # 创建输出张量，初始化为在 'mps' 设备上的随机值，范围在 [-0.5, 0.5)
        output = torch.rand(64, 4, device='mps') - 0.5
        # 创建正权重张量，初始化为全为 1 的张量
        pos_weight = torch.ones(64, 4, device='mps')

        # 断言使用未指定权重和使用全为 1 权重计算的 BCEWithLogitsLoss 结果应该一致
        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))
    # 测试 BCEWithLogitsLoss 在使用不同的正权重（pos_weight）情况下的广播行为
    def test_bce_with_logits_broadcasts_pos_weights(self):
        # 创建随机目标张量和输出张量，设备为 'mps'
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        # 创建随机的正权重张量
        pos_weight = torch.rand(4, device='mps')
        # 使用 pos_weight 计算 BCEWithLogitsLoss 的结果 out1
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        # 将 pos_weight 扩展为形状为 (1, 4) 的张量，计算 BCEWithLogitsLoss 的结果 out2
        pos_weight1 = pos_weight.expand(1, 4)
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        # 将 pos_weight 扩展为形状为 (64, 4) 的张量，计算 BCEWithLogitsLoss 的结果 out3
        pos_weight2 = pos_weight.expand(64, 4)
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        # 断言 out1 与 out2 相等
        self.assertEqual(out1, out2)
        # 断言 out1 与 out3 相等
        self.assertEqual(out1, out3)

    # 测试带正权重（pos_weight）的 BCEWithLogitsLoss 在输出为零时的梯度是否正确
    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        # 创建全零输出张量，需要梯度计算，设备为 'mps'
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        # 创建全零目标张量，设备为 'mps'
        target = torch.zeros(3, 1, device='mps')
        # 创建全一正权重张量，设备为 'mps'
        pos_weight = torch.ones(3, 1, device='mps')
        # 计算带正权重的 BCEWithLogitsLoss 的结果，并进行反向传播
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        # 期望的梯度为全零张量
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        # 获取计算得到的梯度
        grad = output.grad
        # 断言计算得到的梯度与期望的梯度相等
        self.assertEqual(grad, expected_grad)

    # 测试 BCEWithLogitsLoss 在稳定性方面的表现
    def test_bce_with_logits_stability(self):
        # 创建包含两个元素的输出张量和目标张量，设备为 'mps'
        output = torch.tensor([0., -120.], device='mps')
        target = torch.tensor([0., 1.], device='mps')
        # 创建全一正权重张量，设备为 'mps'
        pos_weight = torch.tensor([1., 1.], device='mps')

        # 计算未指定正权重的 BCEWithLogitsLoss 的结果 out1
        out1 = nn.BCEWithLogitsLoss()(output, target)
        # 断言 out1 中所有元素均为有限数
        self.assertTrue(torch.isfinite(out1).all().item())

        # 使用正权重计算 BCEWithLogitsLoss 的结果 out2
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        # 断言 out2 中所有元素均为有限数
        self.assertTrue(torch.isfinite(out2).all().item())

    # 测试 BCELoss 在不同权重情况下的广播行为
    def test_bce_loss_broadcasts_weights(self):
        # 创建 Sigmoid 激活函数
        sigmoid = nn.Sigmoid()
        # 创建随机目标张量和输出张量，设备为 'mps'
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        # 创建随机权重张量，计算带权重的 BCELoss 的结果 out1
        weight = torch.rand(4, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        # 将权重张量扩展为形状为 (16, 4) 的连续张量，计算带权重的 BCELoss 的结果 out2
        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        # 断言 out1 与 out2 相等
        self.assertEqual(out1, out2)

        # 创建形状为 (16, 1) 的随机权重张量，计算带权重的 BCELoss 的结果 out1
        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        # 将权重张量扩展为形状为 (16, 4) 的连续张量，计算带权重的 BCELoss 的结果 out2
        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        # 断言 out1 与 out2 相等
        self.assertEqual(out1, out2)

    # 测试交叉熵损失函数 CrossEntropyLoss
    def test_cross_entropy_loss(self):
        # 创建 CrossEntropyLoss 对象
        loss = nn.CrossEntropyLoss()
        # 创建随机预测张量，目标张量为全一，设备为 'mps'
        pred = torch.randn(3, 5, requires_grad=True, dtype=torch.float16, device='mps')
        target = torch.ones(3, dtype=torch.long, device='mps')
        # 计算交叉熵损失
        output = loss(pred, target)
        # 对损失进行反向传播
        output.backward()
    # 定义一个单元测试方法，用于测试 log_softmax 函数的功能
    def test_log_softmax(self):
        # 创建一个包含两个 2x3 的张量列表
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        # 在 CPU 上创建张量 cpu_x，并设置 requires_grad=True 以便计算梯度
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        # 在 mps 设备（虚拟设备）上创建张量 mps_x，并设置 requires_grad=True
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        # 对 cpu_x 进行 log_softmax 操作，沿着第一个维度（行）进行计算
        cpu_log_softmax = F.log_softmax(cpu_x, dim=0)
        # 对 mps_x 进行 log_softmax 操作，沿着第一个维度（行）进行计算
        mps_log_softmax = F.log_softmax(mps_x, dim=0)
        # 断言两个 log_softmax 的结果在 CPU 上是相等的
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        # 创建一个与 cpu_log_softmax 形状相同的张量，并填充为 1
        cpu_grad = torch.ones_like(cpu_log_softmax)
        # 创建一个与 cpu_log_softmax 形状相同的张量，并在 mps 设备上填充为 1
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        # 对 cpu_log_softmax 执行反向传播，使用 gradient=cpu_grad
        cpu_log_softmax.backward(gradient=cpu_grad)
        # 对 mps_log_softmax 执行反向传播，使用 gradient=mps_grad
        mps_log_softmax.backward(gradient=mps_grad)

        # 断言 cpu_x 的梯度与 mps_x 转到 CPU 后的梯度相等
        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    # 定义一个单元测试方法，用于测试在处理大数时的 log_softmax 函数的功能
    def test_log_softmax_large_numbers(self):
        # 创建包含两个长度为 6 的列表，分别包含正数和负数
        values = [
            [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
            [-10.0, -100.0, -1000.0, -10000.0, -100000.0, -1000000.0]
        ]
        # 在 CPU 上创建张量 cpu_x，并设置 requires_grad=True
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        # 在 mps 设备上创建张量 mps_x，并设置 requires_grad=True
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        # 对 cpu_x 进行 log_softmax 操作，沿着最后一个维度（列）进行计算
        cpu_log_softmax = F.log_softmax(cpu_x, dim=-1)
        # 对 mps_x 进行 log_softmax 操作，沿着最后一个维度（列）进行计算
        mps_log_softmax = F.log_softmax(mps_x, dim=-1)
        # 断言两个 log_softmax 的结果在 CPU 上是相等的
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        # 创建一个与 cpu_log_softmax 形状相同的张量，并填充为 1
        cpu_grad = torch.ones_like(cpu_log_softmax)
        # 创建一个与 cpu_log_softmax 形状相同的张量，并在 mps 设备上填充为 1
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        # 对 cpu_log_softmax 执行反向传播，使用 gradient=cpu_grad
        cpu_log_softmax.backward(gradient=cpu_grad)
        # 对 mps_log_softmax 执行反向传播，使用 gradient=mps_grad
        mps_log_softmax.backward(gradient=mps_grad)

        # 断言 cpu_x 的梯度与 mps_x 转到 CPU 后的梯度相等
        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    # 定义一个单元测试方法，用于测试张量之间的相等性比较
    def test_eq(self):
        # 创建两个张量列表 values1 和 values2
        values1 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        values2 = [[[1.0, 2.0, 15.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 11.0, 12.0]]]
        # 在 mps 设备上创建张量 mps_x 和 mps_y
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        # 在 CPU 上创建张量 cpu_x 和 cpu_y
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        
        # 使用 torch.eq 函数比较 mps_x 和 mps_y 的相等性
        result_mps = torch.eq(mps_x, mps_y)
        # 使用 torch.eq 函数比较 cpu_x 和 cpu_y 的相等性
        result_cpu = torch.eq(cpu_x, cpu_y)

        # 断言在 CPU 上比较的结果与在 mps 设备上转换到 CPU 后的结果相等
        self.assertEqual(result_cpu, result_mps.to('cpu'))

    # 根据产品版本决定是否跳过该测试方法，在 macOS 12 上版本低于 13.0 时跳过
    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    def test_signed_vs_unsigned_comparison(self):
        # 在 CPU 上创建一个 uint8 类型的张量 cpu_x，包含元素 (-1, 2, 3)
        cpu_x = torch.tensor((-1, 2, 3), device='cpu', dtype=torch.uint8)
        # 在 mps 设备上创建一个 uint8 类型的张量 mps_x，包含元素 (-1, 2, 3)
        mps_x = torch.tensor((-1, 2, 3), device='mps', dtype=torch.uint8)
        # 在有符号与无符号比较中，应始终将有符号数转换为无符号数进行比较

        # 断言 cpu_x 中的每个元素是否等于 -1，在 mps_x 中同样进行比较
        self.assertEqual(cpu_x == -1, mps_x == -1)
        # 断言 cpu_x 中的每个元素是否大于 -1，在 mps_x 中同样进行比较
        self.assertEqual(cpu_x > -1, mps_x > -1)
        # 断言 cpu_x 中的每个元素是否小于 -1，在 mps_x 中同样进行比较
        self.assertEqual(cpu_x < -1, mps_x < -1)
    def test_eq_int64(self):
        values1 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        values2 = [[[1, 2, 15], [4, 5, 6]], [[7, 8, 9], [0, 11, 12]]]
        # 创建 MPS 设备上的张量
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        # 创建 CPU 设备上的张量
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        # 在 MPS 设备上比较张量元素是否相等
        result_mps = torch.eq(mps_x, mps_y)
        # 在 CPU 设备上比较张量元素是否相等
        result_cpu = torch.eq(cpu_x, cpu_y)

        # 断言 MPS 设备结果经过 CPU 转换后与 CPU 设备结果相等
        self.assertEqual(result_cpu, result_mps.to('cpu'))

    def test_ne(self):
        def helper(shape):
            # 在 CPU 设备上创建随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量拷贝并转移到 MPS 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            # 在 MPS 设备上比较张量元素是否不相等
            result_mps = torch.ne(mps_x, mps_y)
            # 在 CPU 设备上比较张量元素是否不相等
            result_cpu = torch.ne(cpu_x, cpu_y)

            # 断言 MPS 设备结果经过 CPU 转换后与 CPU 设备结果相等
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ne_scalar(self):
        def helper(shape):
            # 在 CPU 设备上创建随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量拷贝并转移到 MPS 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            # 在 MPS 设备上比较张量元素与标量是否不相等
            result_mps = torch.ne(mps_x, 0.0)
            # 在 CPU 设备上比较张量元素与标量是否不相等
            result_cpu = torch.ne(cpu_x, 0.0)

            # 断言 MPS 设备结果经过 CPU 转换后与 CPU 设备结果相等
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt(self):
        def helper(shape):
            # 在 CPU 设备上创建随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量拷贝并转移到 MPS 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            # 在 MPS 设备上比较张量元素是否小于
            result_mps = torch.lt(mps_x, mps_y)
            # 在 CPU 设备上比较张量元素是否小于
            result_cpu = torch.lt(cpu_x, cpu_y)

            # 断言 MPS 设备结果经过 CPU 转换后与 CPU 设备结果相等
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt_scalar(self):
        def helper(shape):
            # 在 CPU 设备上创建随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量拷贝并转移到 MPS 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            # 在 MPS 设备上比较张量元素是否小于标量
            result_mps = torch.lt(mps_x, 0.0)
            # 在 CPU 设备上比较张量元素是否小于标量
            result_cpu = torch.lt(cpu_x, 0.0)

            # 断言 MPS 设备结果经过 CPU 转换后与 CPU 设备结果相等
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le(self):
        def helper(shape):
            # 在 CPU 设备上创建随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量拷贝并转移到 MPS 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            # 在 MPS 设备上比较张量元素是否小于等于
            result_mps = torch.le(mps_x, mps_y)
            # 在 CPU 设备上比较张量元素是否小于等于
            result_cpu = torch.le(cpu_x, cpu_y)

            # 断言 MPS 设备结果经过 CPU 转换后与 CPU 设备结果相等
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))
    def test_le_scalar(self):
        # 定义测试函数，测试 torch.le 函数对标量情况的处理
        def helper(shape):
            # 生成指定形状的随机张量，设备为 CPU，数据类型为浮点数
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量进行拷贝并分离，然后转移到 'mps' 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            # 使用 torch.le 函数比较 mps_x 和标量 0.0，返回结果张量
            result_mps = torch.le(mps_x, 0.0)
            # 使用 torch.le 函数比较 cpu_x 和标量 0.0，返回结果张量
            result_cpu = torch.le(cpu_x, 0.0)

            # 断言 mps 结果转回 CPU 后与 CPU 结果一致
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        # 调用 helper 函数，测试指定形状的张量
        helper((2, 3, 4, 5))

    def test_ge(self):
        # 定义测试函数，测试 torch.ge 函数对张量情况的处理
        def helper(shape):
            # 生成指定形状的随机张量，设备为 CPU，数据类型为浮点数
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量进行拷贝并分离，然后转移到 'mps' 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            # 使用 torch.ge 函数比较 mps_x 和 mps_y，返回结果张量
            result_mps = torch.ge(mps_x, mps_y)
            # 使用 torch.ge 函数比较 cpu_x 和 cpu_y，返回结果张量
            result_cpu = torch.ge(cpu_x, cpu_y)

            # 断言 mps 结果转回 CPU 后与 CPU 结果一致
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        # 调用 helper 函数，测试指定形状的张量
        helper((2, 3, 4, 5))

    def test_ge_scalar(self):
        # 定义测试函数，测试 torch.ge 函数对标量情况的处理
        def helper(shape):
            # 生成指定形状的随机张量，设备为 CPU，数据类型为浮点数
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量进行拷贝并分离，然后转移到 'mps' 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            # 使用 torch.ge 函数比较 mps_x 和标量 0.0，返回结果张量
            result_mps = torch.ge(mps_x, 0.0)
            # 使用 torch.ge 函数比较 cpu_x 和标量 0.0，返回结果张量
            result_cpu = torch.ge(cpu_x, 0.0)

            # 断言 mps 结果转回 CPU 后与 CPU 结果一致
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        # 调用 helper 函数，测试指定形状的张量
        helper((2, 3, 4, 5))

    def test_gt(self):
        # 定义测试函数，测试 torch.gt 函数对张量情况的处理
        def helper(shape):
            # 生成指定形状的随机张量，设备为 CPU，数据类型为浮点数
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量进行拷贝并分离，然后转移到 'mps' 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            # 使用 torch.gt 函数比较 mps_x 和 mps_y，返回结果张量
            result_mps = torch.gt(mps_x, mps_y)
            # 使用 torch.gt 函数比较 cpu_x 和 cpu_y，返回结果张量
            result_cpu = torch.gt(cpu_x, cpu_y)

            # 断言 mps 结果转回 CPU 后与 CPU 结果一致
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        # 调用 helper 函数，测试指定形状的张量
        helper((2, 3, 4, 5))

    def test_gt_scalar(self):
        # 定义测试函数，测试 torch.gt 函数对标量情况的处理
        def helper(shape):
            # 生成指定形状的随机张量，设备为 CPU，数据类型为浮点数
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 CPU 张量进行拷贝并分离，然后转移到 'mps' 设备上
            mps_x = cpu_x.detach().clone().to('mps')
            # 使用 torch.gt 函数比较 mps_x 和标量 0.0，返回结果张量
            result_mps = torch.gt(mps_x, 0.0)
            # 使用 torch.gt 函数比较 cpu_x 和标量 0.0，返回结果张量
            result_cpu = torch.gt(cpu_x, 0.0)

            # 断言 mps 结果转回 CPU 后与 CPU 结果一致
            self.assertEqual(result_cpu, result_mps.to('cpu'))

        # 调用 helper 函数，测试指定形状的张量
        helper((2, 3, 4, 5))

    def test_argmax(self):
        # 测试 torch.argmax 函数的功能
        # https://github.com/pytorch/pytorch/issues/98191
        # 创建一个 CPU 张量
        cpu_tensor = torch.tensor([[0, 1], [2, 1], [1, 0]])
        # 对 CPU 张量进行沿着维度 1 的 argmax 操作
        res_cpu = torch.argmax(cpu_tensor, dim=1)

        # 将 CPU 张量转移到 'mps' 设备上
        mps_tensor = cpu_tensor.to(torch.device('mps'))
        # 对 'mps' 设备上的张量进行沿着维度 1 的 argmax 操作
        res_mps = torch.argmax(mps_tensor, dim=1)
        # 断言 CPU 和 'mps' 设备上的结果张量一致
        self.assertEqual(res_cpu, res_mps)

        # https://github.com/pytorch/pytorch/issues/92311
        # 生成 'mps' 设备上的随机张量
        mps_tensor = torch.randn(10, 2, device='mps', dtype=torch.float32)
        # 将 'mps' 设备上的张量拷贝并分离，然后转移到 CPU 上
        cpu_tensor = mps_tensor.detach().clone().cpu()

        # 对 CPU 张量进行沿着维度 1 的 argmax 操作
        res_mps = torch.argmax(mps_tensor, dim=1)
        # 对 CPU 张量进行沿着维度 1 的 argmax 操作
        res_cpu = torch.argmax(cpu_tensor, dim=1)
        # 断言 CPU 和 'mps' 设备上的结果张量一致
        self.assertEqual(res_cpu, res_mps)
    #`
    def test_argmin_argmax(self):
        # 定义一个内部辅助函数，进行张量的 argmin 和 argmax 测试
        def helper(n, c, h, w, reduction_type, dtype=torch.float32):
            # 根据 reduction_type 选择 argmin 或 argmax 函数
            if reduction_type == "max":
                arg_reduction_fn = torch.argmax
            else:
                arg_reduction_fn = torch.argmin

            cpu_x = None
            x = None
            # 根据 dtype 分别创建不同类型的张量
            if (dtype not in [torch.float32, torch.bool]):
                # 创建指定 dtype 的整数张量
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                # 将张量从 CPU 移动到 MPS 设备，并克隆一个新的张量
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                # 创建布尔类型的张量
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                # 创建浮点型张量，初始化为标准正态分布
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                # 将张量从 CPU 移动到 MPS 设备，并设置 requires_grad
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算 argmin 或 argmax
            y = arg_reduction_fn(x)
            ref_y = arg_reduction_fn(cpu_x)
            # 检查计算结果与参考结果是否相同
            self.assertEqual(y, ref_y)

            # 在不同维度上进行计算，验证结果是否一致
            y_0 = arg_reduction_fn(x, dim=0)
            refy_0 = arg_reduction_fn(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)

            y_0dim = arg_reduction_fn(x, dim=0, keepdim=True)
            refy_0dim = arg_reduction_fn(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)

            y_1 = arg_reduction_fn(x, dim=1)
            refy_1 = arg_reduction_fn(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)

            y_1dim = arg_reduction_fn(x, dim=1, keepdim=True)
            refy_1dim = arg_reduction_fn(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)

            y_2 = arg_reduction_fn(x, dim=2)
            refy_2 = arg_reduction_fn(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)

            y_2dim = arg_reduction_fn(x, dim=2, keepdim=True)
            refy_2dim = arg_reduction_fn(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)

            y_3 = arg_reduction_fn(x, dim=3)
            refy_3 = arg_reduction_fn(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)

            y_3dim = arg_reduction_fn(x, dim=3, keepdim=True)
            refy_3dim = arg_reduction_fn(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)

        # 调用 helper 函数，测试不同的数据类型和 reduction_type
        helper(2, 8, 4, 4, "max", torch.float32)
        helper(2, 8, 4, 4, "max", torch.int32)
        helper(2, 8, 4, 4, "max", torch.float16)
        helper(2, 8, 4, 4, "max", torch.int64)
        helper(2, 8, 4, 4, "min", torch.float32)
        helper(2, 8, 4, 4, "min", torch.int32)
        helper(2, 8, 4, 4, "min", torch.float16)
        helper(2, 8, 4, 4, "min", torch.int64)

    # 跳过当前测试，直到 macOS 13.3 及以上版本支持 long 数据类型
    @unittest.skipIf(product_version < 13.3, "Long data type supported from macOS 13.3 and above")
    # 定义测试函数，验证对长整数值的求和和最大值的降维操作
    def test_reduction_sum_max_long_val(self):
        # 创建包含系统最大整数值的张量 x_mps，指定设备为 'mps'
        x_mps = torch.tensor([sys.maxsize, sys.maxsize - 10, sys.maxsize - 5, sys.maxsize - 18], device="mps")
        # 创建 x_mps 的副本 x_cpu，并移动到 CPU 设备
        x_cpu = x_mps.detach().clone().cpu()

        # 分别对 x_mps 和 x_cpu 求和
        res_mps = torch.sum(x_mps)
        res_cpu = torch.sum(x_cpu)

        # 断言两者求和结果相等
        self.assertEqual(res_mps, res_cpu)

    # 测试中位数计算
    # 注意 - 目前不测试梯度
    def test_median(self):
        # 定义处理整数类型张量的辅助函数
        def helper_dtype_int32(n1, n2, n3):
            # 在 CPU 设备上生成随机整数张量 cpu_x，范围在 [0, 50)
            cpu_x = torch.randint(50, (n1, n2, n3), device='cpu', dtype=torch.int32)
            # 创建 cpu_x 的 mps_x 副本，并移动到 'mps' 设备
            mps_x = cpu_x.detach().clone().to('mps')

            # 计算 cpu_x 和 mps_x 的中位数
            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            # 断言两者的中位数结果相等
            self.assertEqual(result_cpu, result_mps)

            # 对每个维度和 keepdim 参数组合，计算 cpu_x 和 mps_x 的中位数
            for dim in [0, 1, 2]:
                for keepdim in [True, False]:
                    y, idx = torch.median(cpu_x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.median(mps_x, dim=dim, keepdim=keepdim)
                    # 断言计算结果相等
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

        # 定义处理浮点类型张量的辅助函数
        def helper_dtype_float32(n1, n2, n3):
            # 在 CPU 设备上生成随机浮点数张量 cpu_x，标准正态分布
            cpu_x = torch.randn(n1, n2, n3, device='cpu', dtype=torch.float32)
            # 创建 cpu_x 的 mps_x 副本，并移动到 'mps' 设备
            mps_x = cpu_x.detach().clone().to('mps')

            # 计算 cpu_x 和 mps_x 的中位数
            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            # 断言两者的中位数结果相等
            self.assertEqual(result_cpu, result_mps)

            # 对每个维度和 keepdim 参数组合，计算 cpu_x 和 mps_x 的中位数
            for dim in [0, 1, 2]:
                for keepdim in [True, False]:
                    y, idx = torch.median(cpu_x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.median(mps_x, dim=dim, keepdim=keepdim)
                    # 断言计算结果相等
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

        # 调用整数类型辅助函数，测试不同情况下的中位数计算
        helper_dtype_int32(10, 10, 10)  # median at even place
        helper_dtype_int32(3, 3, 3)    # median at odd place
        helper_dtype_int32(1, 1, 1)    # single element
        helper_dtype_int32(1, 2, 3)    # different shape
        # 调用浮点类型辅助函数，测试不同情况下的中位数计算
        helper_dtype_float32(10, 10, 10)
        helper_dtype_float32(3, 3, 3)
        helper_dtype_float32(1, 1, 1)
    # 定义一个测试方法 test_any，用于测试 torch.any 函数的行为
    def test_any(self):
        # 定义一个辅助函数 helper，用于生成测试数据和进行测试
        def helper(shape):
            # 初始化一个空列表 input_xs 用于存储输入数据
            input_xs = []
            # 初始化 prod 为 1，用于计算 shape 元素的乘积
            prod = 1

            # 计算 shape 中所有元素的乘积
            for i in range(len(shape)):
                prod *= shape[i]
            
            # 生成不同类型的测试数据，并存储到 input_xs 中
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            # 遍历 input_xs 中的数据
            for i, cpu_x in enumerate(input_xs):
                # 将 cpu_x 分离出来并克隆到 'mps' 设备上
                x = cpu_x.detach().clone().to('mps')
                # 使用 torch.any 计算 x 的结果
                y = torch.any(x)
                # 使用 torch.any 计算 cpu_x 的结果作为参考
                ref_y = torch.any(cpu_x)
                # 断言 x 的计算结果与参考结果 ref_y 相等
                self.assertEqual(y, ref_y)

                # 对 x 在 dim=0 上进行 torch.any 操作
                y_0 = torch.any(x, dim=0)
                refy_0 = torch.any(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                # 对 x 在 dim=0 上进行 torch.any 操作，并保持维度
                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                # 重复之前的维度操作
                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                # 对 x 在 dim=1 上进行 torch.any 操作
                y_1 = torch.any(x, dim=1)
                refy_1 = torch.any(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                # 对 x 在 dim=1 上进行 torch.any 操作，并保持维度
                y_1dim = torch.any(x, dim=1, keepdim=True)
                refy_1dim = torch.any(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)

                # 如果 shape 的长度大于 2，则进行额外的维度操作
                if (len(shape) > 2):
                    y_2 = torch.any(x, dim=2)
                    refy_2 = torch.any(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.any(x, dim=2, keepdim=True)
                    refy_2dim = torch.any(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.any(x, dim=3)
                    refy_3 = torch.any(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.any(x, dim=3, keepdim=True)
                    refy_3dim = torch.any(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)

        # 分别测试不同的 shape 参数对 helper 函数的影响
        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))

    # 标记该测试为跳过状态，因为该测试存在问题导致崩溃
    @unittest.skip("Test is crashing")
    # 定义一个测试函数，用于测试在5维张量上的降维操作
    def test_reduction_ops_5D(self):
        # 定义一个辅助函数，用于测试给定的降维函数在指定维度上的操作结果
        def helper(fn, dim):
            # 在CPU上创建一个5维全零张量，并对指定维度进行降维操作
            x_cpu = fn(torch.zeros(1, 1, 1, 1, 1), dim=dim)
            # 在MPS设备（可能是一种特殊的硬件加速器）上创建一个5维全零张量，并对指定维度进行降维操作
            x_mps = fn(torch.zeros(1, 1, 1, 1, 1, device="mps"), dim=dim)
            # 断言CPU和MPS设备上的操作结果相等，将MPS设备上的结果转移到CPU上进行比较
            self.assertEqual(x_cpu, x_mps.to('cpu'))
        
        # 遍历要测试的降维函数列表，这里只有torch.any函数
        for fn in [torch.any]:
            # 遍历可能的降维维度，从维度0到3
            for dim in range(0, 4):
                # 调用helper函数进行测试
                helper(fn, dim)
    def test_all(self):
        # 定义内部辅助函数 `helper`，用于生成不同形状的输入数据并进行测试
        def helper(shape):
            # 初始化空列表 `input_xs` 用于存储不同类型的输入数据
            input_xs = []
            # 计算输入数据的总元素个数 `prod`
            prod = 1

            # 计算总元素个数 `prod` 的乘积
            for i in range(len(shape)):
                prod *= shape[i]
            
            # 将不同类型的数据添加到 `input_xs`
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            # 遍历 `input_xs` 中的每个数据进行测试
            for i, cpu_x in enumerate(input_xs):
                # 将 `cpu_x` 数据拷贝并移动到 'mps' 设备
                x = cpu_x.detach().clone().to('mps')
                
                # 计算 `x` 的所有元素是否为真，与参考值 `ref_y` 比较
                y = torch.all(x)
                ref_y = torch.all(cpu_x)
                self.assertEqual(y, ref_y)

                # 计算 `x` 按第0维的所有元素是否为真，与参考值 `refy_0` 比较
                y_0 = torch.all(x, dim=0)
                refy_0 = torch.all(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                # 计算 `x` 按第0维的所有元素是否为真（保持维度），与参考值 `refy_0dim` 比较
                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                # 再次计算 `x` 按第0维的所有元素是否为真（保持维度），与参考值 `refy_0dim` 比较
                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                # 计算 `x` 按第1维的所有元素是否为真，与参考值 `refy_1` 比较
                y_1 = torch.all(x, dim=1)
                refy_1 = torch.all(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                # 计算 `x` 按第1维的所有元素是否为真（保持维度），与参考值 `refy_1dim` 比较
                y_1dim = torch.all(x, dim=1, keepdim=True)
                refy_1dim = torch.all(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)

                # 如果形状长度大于2，继续进行第2维和第3维的元素是否为真的测试
                if (len(shape) > 2):
                    # 计算 `x` 按第2维的所有元素是否为真，与参考值 `refy_2` 比较
                    y_2 = torch.all(x, dim=2)
                    refy_2 = torch.all(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    # 计算 `x` 按第2维的所有元素是否为真（保持维度），与参考值 `refy_2dim` 比较
                    y_2dim = torch.all(x, dim=2, keepdim=True)
                    refy_2dim = torch.all(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    # 计算 `x` 按第3维的所有元素是否为真，与参考值 `refy_3` 比较
                    y_3 = torch.all(x, dim=3)
                    refy_3 = torch.all(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    # 计算 `x` 按第3维的所有元素是否为真（保持维度），与参考值 `refy_3dim` 比较
                    y_3dim = torch.all(x, dim=3, keepdim=True)
                    refy_3dim = torch.all(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)

        # 分别测试不同形状的数据
        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))
        
        # 创建一个空的布尔类型的 `x_cpu` 张量
        x_cpu = torch.tensor([], dtype=torch.bool)
        # 将 `x_cpu` 张量移动到 'mps' 设备，命名为 `x_mps`
        x_mps = x_cpu.to("mps")
        # 断言 `x_cpu` 张量的所有元素都为真，与 `x_mps` 张量的所有元素（经过 CPU）比较
        assert x_cpu.all() == x_mps.all().cpu()
    # 测试向前传播的求和功能

    def test_sum(self):
        # 定义辅助函数，生成不同类型和形状的张量，并在 'mps' 设备上进行克隆
        def helper(n, c, h, w, dtype=torch.float32):
            cpu_x = None  # 在 CPU 上生成张量的变量
            x = None       # 在 'mps' 设备上生成张量的变量

            # 根据 dtype 类型生成不同类型的张量
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算张量 x 的总和及其在 CPU 上的对应值
            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            # 断言总和值相等
            self.assertEqual(all_sum, all_sum_cpu)

            # 沿空维度求和，并比较 CPU 上的对应值
            nil_dim_sum = torch.sum(x, dim=[])
            nil_dim_sum_cpu = torch.sum(cpu_x, dim=[])

            self.assertEqual(nil_dim_sum, nil_dim_sum_cpu)

            # 沿空维度求和，保持维度，并比较 CPU 上的对应值
            nil_dim_sum_keepdim = torch.sum(x, dim=[], keepdim=True)
            nil_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_sum_keepdim, nil_dim_sum_cpu_keepdim)

            # 沿第一维度求和，并比较 CPU 上的对应值
            zero_dim_sum = torch.sum(x, dim=[0])
            zero_dim_sum_cpu = torch.sum(cpu_x, dim=[0])

            self.assertEqual(zero_dim_sum, zero_dim_sum_cpu)

            # 沿第一维度求和，保持维度，并比较 CPU 上的对应值
            zero_dim_sum_keepdim = torch.sum(x, dim=[0], keepdim=True)
            zero_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_sum_keepdim, zero_dim_sum_cpu_keepdim)

            # 沿第一和第二维度求和，并比较 CPU 上的对应值
            zero_one_dim_sum = torch.sum(x, dim=[0, 1])
            zero_one_dim_sum_cpu = torch.sum(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_sum, zero_one_dim_sum_cpu)

            # 沿第一和第二维度求和，保持维度，并比较 CPU 上的对应值
            zero_one_dim_sum_keepdim = torch.sum(x, dim=[0, 1], keepdim=True)
            zero_one_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_sum_keepdim, zero_one_dim_sum_cpu_keepdim)

            # 沿第三和第四维度求和，并比较 CPU 上的对应值
            two_three_dim_sum = torch.sum(x, dim=[2, 3])
            two_three_dim_sum_cpu = torch.sum(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_sum, two_three_dim_sum_cpu)

            # 沿第三和第四维度求和，保持维度，并比较 CPU 上的对应值
            two_three_keepdim_sum = torch.sum(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepsum_cpu = torch.sum(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_sum, two_three_dim_keepsum_cpu)

        # 测试不同参数下的 helper 函数
        helper(2, 8, 4, 5)
        helper(2, 8, 4, 5, dtype=torch.int32)
        helper(2, 8, 4, 5, dtype=torch.int64)
        helper(2, 8, 4, 5, dtype=torch.bool)

    # 测试向前传播的求积功能
    # 定义测试方法 test_prod，用于测试 torch.prod 函数的行为
    def test_prod(self):
        
        # 定义辅助函数 helper，用于生成不同数据类型和形状的张量，并测试其乘积计算
        def helper(shape, dtype=torch.float32):
            cpu_x = None
            x = None
            
            # 根据 dtype 类型选择不同的张量生成方式
            if (dtype not in [torch.float32, torch.bool]):
                # 生成指定形状的随机整数张量，设备为 CPU，数据类型为 dtype，不需要梯度
                cpu_x = torch.randint(1, 6, shape, device='cpu', dtype=dtype, requires_grad=False)
                # 将 CPU 张量复制到 'mps' 设备上
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                # 生成指定形状的随机布尔张量，设备为 CPU，不需要梯度
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                # 将 CPU 张量复制到 'mps' 设备上
                x = cpu_x.detach().clone().to('mps')
            else:
                # 生成指定形状的随机浮点数张量，设备为 CPU，需要梯度
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                # 将 CPU 张量复制到 'mps' 设备上，并设置需要梯度
                x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 计算 'mps' 设备上张量的所有元素乘积
            all_prod = torch.prod(x)
            # 计算 CPU 上张量的所有元素乘积
            all_prod_cpu = torch.prod(cpu_x)
            
            # 断言 'mps' 设备和 CPU 上的所有元素乘积相等
            self.assertEqual(all_prod, all_prod_cpu)
            
            # 对张量的每个维度进行循环，计算 'mps' 设备和 CPU 上的乘积
            for dim in range(len(shape)):
                # 沿指定维度计算 'mps' 设备上张量的乘积
                dim_prod = torch.prod(x, dim=dim)
                # 沿指定维度计算 CPU 上张量的乘积
                dim_prod_cpu = torch.prod(cpu_x, dim=dim)
                
                # 断言 'mps' 设备和 CPU 上沿指定维度的乘积相等
                self.assertEqual(dim_prod, dim_prod_cpu)
                
                # 沿指定维度计算 'mps' 设备上张量的乘积，并保持维度不变
                dim_prod_keepdim = torch.prod(x, dim=dim, keepdim=True)
                # 沿指定维度计算 CPU 上张量的乘积，并保持维度不变
                dim_prod_cpu_keepdim = torch.prod(cpu_x, dim=dim, keepdim=True)
                
                # 断言 'mps' 设备和 CPU 上沿指定维度并保持维度不变的乘积相等
                self.assertEqual(dim_prod_keepdim, dim_prod_cpu_keepdim)
        
        # 对于给定的数据类型列表，依次调用 helper 函数进行测试
        for dtype in [torch.float32, torch.int32, torch.int64, torch.bool]:
            helper((2, 3), dtype)

    # 测试 forward mean 的功能
    # 定义测试方法 test_mean，用于测试 torch.mean 函数的不同用法
    def test_mean(self):
        # 定义辅助函数 helper，用于生成随机张量并测试其均值计算
        def helper(n, c, h, w):
            # 生成一个指定形状的随机张量 cpu_x，设备为 CPU，数据类型为浮点型，同时需要梯度信息
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=True)
            # 创建张量 x，其值与 cpu_x 一致，设备为 'mps'，同时需要梯度信息
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算张量 x 的全局均值
            all_mean = torch.mean(x)
            # 计算张量 cpu_x 的全局均值
            all_mean_cpu = torch.mean(cpu_x)

            # 断言全局均值应相等
            self.assertEqual(all_mean, all_mean_cpu)

            # 计算张量 x 在空维度上的均值
            nil_dim_mean = torch.mean(x, dim=[])
            # 计算张量 cpu_x 在空维度上的均值
            nil_dim_mean_cpu = torch.mean(cpu_x, dim=[])

            # 断言空维度上的均值应相等
            self.assertEqual(nil_dim_mean, nil_dim_mean_cpu)

            # 计算张量 x 在空维度上的均值，保持维度
            nil_dim_mean_keepdim = torch.mean(x, dim=[], keepdim=True)
            # 计算张量 cpu_x 在空维度上的均值，保持维度
            nil_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[], keepdim=True)

            # 断言保持维度后的空维度均值应相等
            self.assertEqual(nil_dim_mean_keepdim, nil_dim_mean_cpu_keepdim)

            # 计算张量 x 在第一维上的均值
            zero_dim_mean = torch.mean(x, dim=[0])
            # 计算张量 cpu_x 在第一维上的均值
            zero_dim_mean_cpu = torch.mean(cpu_x, dim=[0])

            # 断言第一维上的均值应相等
            self.assertEqual(zero_dim_mean, zero_dim_mean_cpu)

            # 计算张量 x 在第一维上的均值，保持维度
            zero_dim_mean_keepdim = torch.mean(x, dim=[0], keepdim=True)
            # 计算张量 cpu_x 在第一维上的均值，保持维度
            zero_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0], keepdim=True)

            # 断言保持维度后的第一维均值应相等
            self.assertEqual(zero_dim_mean_keepdim, zero_dim_mean_cpu_keepdim)

            # 计算张量 x 在第一和第二维上的均值
            zero_one_dim_mean = torch.mean(x, dim=[0, 1])
            # 计算张量 cpu_x 在第一和第二维上的均值
            zero_one_dim_mean_cpu = torch.mean(cpu_x, dim=[0, 1])

            # 断言第一和第二维上的均值应相等
            self.assertEqual(zero_one_dim_mean, zero_one_dim_mean_cpu)

            # 计算张量 x 在第一和第二维上的均值，保持维度
            zero_one_dim_mean_keepdim = torch.mean(x, dim=[0, 1], keepdim=True)
            # 计算张量 cpu_x 在第一和第二维上的均值，保持维度
            zero_one_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0, 1], keepdim=True)

            # 断言保持维度后的第一和第二维均值应相等
            self.assertEqual(zero_one_dim_mean_keepdim, zero_one_dim_mean_cpu_keepdim)

            # 计算张量 x 在第三和第四维上的均值
            two_three_dim_mean = torch.mean(x, dim=[2, 3])
            # 计算张量 cpu_x 在第三和第四维上的均值
            two_three_dim_mean_cpu = torch.mean(cpu_x, dim=[2, 3])

            # 断言第三和第四维上的均值应相等
            self.assertEqual(two_three_dim_mean, two_three_dim_mean_cpu)

            # 计算张量 x 在第三和第四维上的均值，保持维度
            two_three_keepdim_mean = torch.mean(x, dim=[2, 3], keepdim=True)
            # 计算张量 cpu_x 在第三和第四维上的均值，保持维度
            two_three_dim_keepmean_cpu = torch.mean(cpu_x, dim=[2, 3], keepdim=True)

            # 断言保持维度后的第三和第四维均值应相等
            self.assertEqual(two_three_keepdim_mean, two_three_dim_keepmean_cpu)

        # 调用 helper 函数，测试指定形状的随机张量
        helper(2, 8, 4, 5)

    # 测试 std
    # 测试 var
    def test_var_simple(self):
        def helper():
            # 定义张量的形状
            shape = [2, 3, 4, 5]

            # 在 CPU 上生成随机张量，并将其转换为 'mps' 设备
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            # 遍历不偏估计和保持维度的组合
            for unbiased in [False, True]:
                for keepdim in [False, True]:

                    # 计算沿着最后一个维度的方差，并与 CPU 上的结果比较
                    zero_dim_var = x.var(-1, keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = cpu_x.var(-1, keepdim=keepdim, unbiased=unbiased)
                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    # 计算张量的全局方差，并与 CPU 上的结果比较
                    all_var = torch.var(x, unbiased=unbiased)
                    all_var_cpu = torch.var(cpu_x, unbiased=unbiased)
                    self.assertEqual(all_var, all_var_cpu)

                    # 计算没有维度的方差（即整个张量的方差），并与 CPU 上的结果比较
                    nil_dim_var = torch.var(x, dim=[], keepdim=keepdim, unbiased=unbiased)
                    nil_dim_var_cpu = torch.var(cpu_x, dim=[], keepdim=keepdim, unbiased=unbiased)
                    self.assertEqual(nil_dim_var, nil_dim_var_cpu)

                    # 计算沿着第一个维度的方差，并与 CPU 上的结果比较
                    zero_dim_var = torch.var(x, dim=[0], keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = torch.var(cpu_x, dim=[0], keepdim=keepdim, unbiased=unbiased)
                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    # 计算沿着第一个和最后一个维度的方差，并与 CPU 上的结果比较
                    zero_one_dim_var = torch.var(x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)
                    zero_one_dim_var_cpu = torch.var(cpu_x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)
                    self.assertEqual(zero_one_dim_var, zero_one_dim_var_cpu)

                    # 计算沿着第三和第四个维度的方差，并与 CPU 上的结果比较
                    two_three_dim_var = torch.var(x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)
                    two_three_dim_var_cpu = torch.var(cpu_x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)
                    self.assertEqual(two_three_dim_var, two_three_dim_var_cpu)

        helper()

    # 测试张量的最大值
    def test_amax(self):
        def helper(shape, dim, keepdim):
            # 在 CPU 上生成随机张量，并将其转换为 'mps' 设备，并设置梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算张量在指定维度上的最大值，并与 CPU 上的结果比较
            result = torch.amax(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amax(cpu_x, dim=dim, keepdim=keepdim)

            # 创建一个与 CPU 结果形状相同的随机梯度，并将其转换为 'mps' 设备
            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            # 在张量上执行反向传播，并将梯度应用到 'mps' 设备上的张量
            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            # 断言张量最大值的计算结果与 CPU 上的结果相等
            self.assertEqual(result, result_cpu)
            # 断言张量梯度与 CPU 上的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 遍历不同维度和保持维度的组合进行测试
        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test forward amin
    def test_amin(self):
        # 定义辅助函数helper，用于测试torch.amin函数的功能
        def helper(shape, dim, keepdim):
            # 在CPU上生成随机张量cpu_x，并设置其为浮点类型，需要梯度追踪
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将cpu_x进行分离和克隆，转移到'MPS'设备，并设置需要梯度追踪
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 调用torch.amin函数，在指定维度dim上计算最小值，保持维度keepdim不变
            result = torch.amin(x, dim=dim, keepdim=keepdim)
            # 在CPU上计算相同操作的结果，用于对比验证
            result_cpu = torch.amin(cpu_x, dim=dim, keepdim=keepdim)

            # 创建与result_cpu形状相同的随机梯度cpu_grad
            cpu_grad = torch.randn(result_cpu.shape)
            # 将cpu_grad转移到'MPS'设备
            grad = cpu_grad.to('mps')

            # 对result_cpu进行反向传播，使用cpu_grad作为梯度
            result_cpu.backward(gradient=cpu_grad)
            # 对result进行反向传播，使用grad作为梯度
            result.backward(gradient=grad)

            # 断言两个result相等
            self.assertEqual(result, result_cpu)
            # 断言x的梯度与cpu_x的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 针对不同的维度dim和keepdim进行测试helper函数
        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # 测试torch.minimum和torch.maximum函数
    def test_minimum_maximum(self):
        # 定义辅助函数helper，用于测试torch.minimum和torch.maximum函数的功能
        def helper(n, c, h, w):
            # 在CPU上生成随机张量cpu_x和cpu_y，并设置其为浮点类型，不需要梯度追踪
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            # 将cpu_x和cpu_y进行分离和克隆，转移到'MPS'设备
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')

            # 调用torch.minimum函数，在两个张量上逐元素计算最小值
            minimum_result_cpu = torch.minimum(cpu_x, cpu_y)
            minimum_result_mps = torch.minimum(mps_x, mps_y)
            # 断言两个结果相等
            self.assertEqual(minimum_result_cpu, minimum_result_mps)

            # 调用torch.maximum函数，在两个张量上逐元素计算最大值
            maximum_result_cpu = torch.maximum(cpu_x, cpu_y)
            maximum_result_mps = torch.maximum(mps_x, mps_y)
            # 断言两个结果相等
            self.assertEqual(maximum_result_cpu, maximum_result_mps)

        # 使用指定维度和大小调用helper函数进行测试
        helper(1, 1, 4, 5)

    # 测试torch.clamp函数在混合精度下的表现
    def test_clamp_fp16_fp32(self):
        # 在CPU上生成随机张量cpu_x，并设置其为浮点类型，不需要梯度追踪
        cpu_x = torch.randn(10, device='cpu', dtype=torch.float, requires_grad=False)
        # 将cpu_x进行分离和克隆，转移到'MPS'设备
        x = cpu_x.detach().clone().to('mps')

        # 设置目标数据类型为torch.float16
        dtype = torch.float16

        # 在'MPS'设备上生成最小值和最大值张量，数据类型为torch.float16
        clamp_min_vals_mps = torch.ones(10, device="mps").to(torch.float16)
        clamp_max_vals_mps = torch.ones(10, device="mps").to(torch.float16) * 10
        # 调用torch.clamp函数，将x限制在clamp_min_vals_mps和clamp_max_vals_mps之间
        clamp_result_mps = torch.clamp(x, clamp_min_vals_mps, clamp_max_vals_mps)

        # 在CPU上生成最小值和最大值张量，数据类型为torch.float16
        clamp_min_vals_cpu = torch.ones(10, device="cpu").to(torch.float16)
        clamp_max_vals_cpu = torch.ones(10, device="cpu").to(torch.float16) * 10
        # 调用torch.clamp函数，将cpu_x限制在clamp_min_vals_cpu和clamp_max_vals_cpu之间
        clamp_result_cpu = torch.clamp(cpu_x, clamp_min_vals_cpu, clamp_max_vals_cpu)

        # 断言clamp_result_mps与clamp_result_cpu相等
        self.assertEqual(clamp_result_mps, clamp_result_cpu)
    # 定义一个测试方法，用于测试处理 NaN 值的 clamp 函数
    def test_clamp_nan(self):
        # 创建一个包含 NaN 值的张量，设备为 "mps"
        t_mps = torch.tensor([torch.nan, 1, 2], device="mps")
        # 创建一个包含 NaN 值的张量，设备为 "cpu"
        t_cpu = torch.tensor([torch.nan, 1, 2], device="cpu")

        # 对 t_mps 和 t_cpu 分别使用 clamp 函数，将值限制在 -100 到 100 之间
        clamp_min_max_mps = torch.clamp(t_mps, min=-100, max=100)
        clamp_min_max_cpu = torch.clamp(t_cpu, min=-100, max=100)

        # 断言两个张量的值相等
        self.assertEqual(clamp_min_max_mps, clamp_min_max_cpu)

        # 对 t_mps 和 t_cpu 分别使用 clamp 函数，将值限制在 -100 到正无穷之间
        clamp_min_mps = torch.clamp(t_mps, min=-100)
        clamp_min_cpu = torch.clamp(t_cpu, min=-100)

        # 断言两个张量的值相等
        self.assertEqual(clamp_min_mps, clamp_min_cpu)

        # 对 t_mps 和 t_cpu 分别使用 clamp 函数，将值限制在负无穷到 100 之间
        clamp_max_mps = torch.clamp(t_mps, max=100)
        clamp_max_cpu = torch.clamp(t_cpu, max=100)

        # 断言两个张量的值相等
        self.assertEqual(clamp_max_mps, clamp_max_cpu)

    # Test clamp_min
    # 定义一个测试方法，用于测试 clamp_min 函数
    def test_clamp_min(self):
        # 定义一个辅助函数，生成指定形状的随机张量，设备为 "mps"
        def helper(n, c, h, w):
            # 生成一个随机张量，设备为 "cpu"，并转换为 "mps"
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            # 生成一个随机张量，设备为 "cpu"，并转换为 "mps"
            cpu_min_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            # 对 x 和 cpu_x 分别使用 clamp_min 函数，将值限制在 5.0 以上
            clamp_min_result = torch.clamp_min(x, min=5.0)
            clamp_min_result_cpu = torch.clamp_min(cpu_x, min=5.0)

            # 断言两个张量的值相等
            self.assertEqual(clamp_min_result, clamp_min_result_cpu)

            # 对 x 和 cpu_x 分别使用 clamp_min 函数，将值限制在 min_t 张量的每个元素以上
            clamp_min_t_result = torch.clamp_min(x, min=min_t)
            clamp_min_t_result_cpu = torch.clamp_min(cpu_x, min=cpu_min_t)

            # 断言两个张量的值相等
            self.assertEqual(clamp_min_t_result, clamp_min_t_result_cpu)

        # 调用辅助函数，传入指定的参数
        helper(2, 8, 4, 5)

    # Test clamp_max
    # 定义一个测试方法，用于测试 clamp_max 函数
    def test_clamp_max(self):
        # 定义一个辅助函数，生成指定形状的随机张量，设备为 "mps"
        def helper(n, c, h, w):
            # 生成一个随机张量，设备为 "cpu"，并转换为 "mps"
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            # 生成一个随机张量，设备为 "cpu"，并转换为 "mps"
            cpu_max_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            # 对 x 和 cpu_x 分别使用 clamp_max 函数，将值限制在 100.0 以下
            clamp_max_result = torch.clamp_max(x, max=100.0)
            clamp_max_result_cpu = torch.clamp_max(cpu_x, max=100.0)

            # 断言两个张量的值相等
            self.assertEqual(clamp_max_result, clamp_max_result_cpu)

            # 对 x 和 cpu_x 分别使用 clamp_max 函数，将值限制在 max_t 张量的每个元素以下
            clamp_max_t_result = torch.clamp_max(x, max=max_t)
            clamp_max_t_result_cpu = torch.clamp_max(cpu_x, max=cpu_max_t)

            # 断言两个张量的值相等
            self.assertEqual(clamp_max_t_result, clamp_max_t_result_cpu)

        # 调用辅助函数，传入指定的参数
        helper(2, 8, 4, 5)
    def test_divmode(self):
        # 定义辅助函数，用于测试不同的形状和舍入模式对于不同数据类型的计算结果
        def helper(shape, rounding_mode):
            # 遍历每种数据类型
            for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
                # 检查特定的舍入模式和数据类型组合，根据条件判断是否跳过当前循环
                if ((rounding_mode is not None and "floor" in rounding_mode and dtype == torch.int64) or
                        (rounding_mode is not None and "trunc" in rounding_mode and dtype == torch.float16)) is False:
                    cpu_x = None
                    cpu_y = None
                    # 根据数据类型，生成不同的随机数据或整数数据
                    if (dtype in [torch.float32, torch.float16]):
                        cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                        cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                    else:
                        cpu_x = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)
                        cpu_y = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)

                    # 将CPU上的数据复制到'MPS'设备，并避免除以0的情况
                    mps_x = cpu_x.detach().clone().to('mps')
                    mps_y = cpu_y.detach().clone().to('mps')

                    # 根据舍入模式执行不同的除法运算，比较'MPS'设备和CPU的计算结果
                    if (rounding_mode == "floor_divide"):
                        result_div_cpu = torch.floor_divide(cpu_x, cpu_y)
                        result_div_mps = torch.floor_divide(mps_x, mps_y)
                        self.assertEqual(result_div_mps, result_div_cpu)
                    else:
                        result_div_cpu = torch.div(cpu_x, cpu_y, rounding_mode=rounding_mode)
                        result_div_mps = torch.div(mps_x, mps_y, rounding_mode=rounding_mode)
                        self.assertEqual(result_div_mps, result_div_cpu)

        # 调用helper函数，测试不同形状和舍入模式的组合
        helper((2, 8, 4, 5), None)
        helper((2, 8, 4, 5), "floor")
        helper((2, 8, 4, 5), "trunc")
        helper((2, 8, 4, 5), "floor_divide")

    def test_rounding(self):
        # 定义辅助函数，用于测试不同形状的浮点数张量在'MPS'设备上的四舍五入操作
        def helper(shape):
            # 生成CPU上的随机浮点数张量，并将其复制到'MPS'设备
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')

            # 执行四舍五入、向下取整、向上取整和截断操作，比较'MPS'设备和CPU的计算结果
            result_floor_cpu = torch.floor(cpu_x)
            result_floor_mps = torch.floor(mps_x)
            self.assertEqual(result_floor_mps, result_floor_cpu)

            result_ceil_cpu = torch.ceil(cpu_x)
            result_ceil_mps = torch.ceil(mps_x)
            self.assertEqual(result_ceil_mps, result_ceil_cpu)

            result_trunc_cpu = torch.trunc(cpu_x)
            result_trunc_mps = torch.trunc(mps_x)
            self.assertEqual(result_trunc_mps, result_trunc_cpu)

            result_round_cpu = torch.round(cpu_x)
            result_round_mps = torch.round(mps_x)
            self.assertEqual(result_round_mps, result_round_cpu)

        # 调用helper函数，测试不同形状的浮点数张量在'MPS'设备上的四舍五入操作
        helper((2, 6, 3, 5))
        helper((2, 8, 4, 5))
    def test_remainder(self):
        # 使用torch.remainder计算两个张量的余数，cpu设备上的计算
        res_cpu = torch.remainder(
            torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int32, device="cpu"), torch.tensor(2, device="cpu", dtype=torch.int32))
        # 使用torch.remainder计算两个张量的余数，mps设备上的计算
        res_mps = torch.remainder(
            torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int32, device="mps"), torch.tensor(2, device="mps", dtype=torch.int32))
        # 断言两个计算结果是否相等
        self.assertEqual(res_cpu, res_mps)

        # 使用torch.remainder计算张量与标量之间的余数，cpu设备上的计算
        res_cpu = torch.remainder(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="cpu"), -1.5)
        # 使用torch.remainder计算张量与标量之间的余数，mps设备上的计算
        res_mps = torch.remainder(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="mps"), -1.5)
        # 断言两个计算结果是否相等
        self.assertEqual(res_cpu, res_mps)

    def test_expand(self):
        def helper(n, c):
            # 创建一个CPU上的张量
            values = [[1.0], [4.0], [7.0]]
            cpu_x = torch.tensor(values, device='cpu')
            # 将CPU上的张量复制到mps设备上
            x = cpu_x.detach().clone().to('mps')

            # 在CPU上使用torch.as_strided进行张量的展开
            strided_cpu = torch.as_strided(cpu_x, (3, 4), (1, 0))
            # 在mps设备上使用torch.as_strided进行张量的展开
            strided_mps = torch.as_strided(x, (3, 4), (1, 0))

            # 断言两个展开后的张量是否相等
            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 1)

    def test_im2col(self):
        def helper(x):
            # 使用torch.nn.functional.unfold对输入张量进行图像转换操作
            return torch.nn.functional.unfold(x, kernel_size=(10, 15), dilation=2, padding=5, stride=3)
        # 创建一个CPU上的随机张量
        x_cpu = torch.rand(1, 1, 200, 100)
        # 将CPU上的张量复制到mps设备上
        x = x_cpu.detach().clone().to('mps')
        # 断言两次im2col操作的结果是否相等
        self.assertEqual(helper(x_cpu), helper(x))

    def test_select(self):
        def helper(n, c):
            # 创建一个CPU上的随机张量，带有梯度计算
            cpu_x = torch.randn(n, c, device='cpu', dtype=torch.float, requires_grad=True)
            # 将CPU上的张量复制到mps设备上，并要求计算梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 在CPU上使用torch.as_strided进行张量的选择操作
            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1))
            # 在mps设备上使用torch.as_strided进行张量的选择操作
            strided_mps = torch.as_strided(x, (3, 1), (3, 1))
            # 断言两个选择操作后的张量是否相等
            self.assertEqual(strided_mps, strided_cpu)

            # 在CPU上使用torch.as_strided进行张量的选择操作
            strided_cpu = torch.as_strided(cpu_x, (1, 3), (3, 1))
            # 在mps设备上使用torch.as_strided进行张量的选择操作
            strided_mps = torch.as_strided(x, (1, 3), (3, 1))
            # 断言两个选择操作后的张量是否相等
            self.assertEqual(strided_mps, strided_cpu)

            # 在CPU上使用torch.as_strided进行张量的选择操作，带有偏移
            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1), storage_offset=1)
            # 在mps设备上使用torch.as_strided进行张量的选择操作，带有偏移
            strided_mps = torch.as_strided(x, (3, 1), (3, 1), storage_offset=1)
            # 断言两个选择操作后的张量是否相等
            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 3)
    # 定义名为 test_sort 的测试方法，用于测试排序功能
    def test_sort(self):
        # 针对不同的 SIZE 参数进行测试，分别为 4 和 2049
        for SIZE in (4, 2049):
            # 指定设备为 'mps'
            device = 'mps'
            # 生成一个大小为 (4, SIZE) 的随机张量 x，存储在指定设备上
            x = torch.rand(4, SIZE, device=device)
            # 使用 torch.sort 对 x 进行排序，返回排序后的值和索引
            res1val, res1ind = torch.sort(x)

            # 创建空张量 res2val 和 res2ind，存储在指定设备上，指定数据类型为 torch.long
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            # 使用 torch.sort 将排序结果直接存入 res2val 和 res2ind 中
            torch.sort(x, out=(res2val, res2ind))
            # 断言 res1val 与 res2val 相等，允许的绝对误差和相对误差均为 0
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            # 断言 res1ind 与 res2ind 相等，允许的绝对误差和相对误差均为 0
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            # 断言 torch.argsort(x) 的结果与 res1ind 相等
            self.assertEqual(torch.argsort(x), res1ind)
            # 断言 x.argsort() 的结果与 res1ind 相等
            self.assertEqual(x.argsort(), res1ind)

            # 断言对 torch.tensor((50, 40, 30, 20, 10), device=device) 进行排序的结果
            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0, rtol=0
            )

    # 定义名为 test_upsample_nearest2d 的测试方法
    def test_upsample_nearest2d(self):
        # 定义内部辅助函数 helper，用于测试不同参数下的双线性插值功能
        def helper(N, C, H, W, memory_format):
            # 在 CPU 上生成一个形状为 (N, C, H, W) 的张量 inputCPU，数据类型为 torch.float，允许梯度计算
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W).to(memory_format=memory_format)
            inputCPU.retain_grad()
            # 将 inputCPU 的副本转换到 'mps' 设备，设置为需要梯度
            inputMPS = inputCPU.detach().to('mps').requires_grad_()

            # 定义一组 scale_factor 值
            values = [1, 2, 5, 10, 40]

            # 遍历 values 中的每对值 (i, j)
            for i in values:
                for j in values:
                    # 创建双线性插值层对象 upsample_nearest2d，使用 scale_factor=(i, j)
                    upsample_nearest2d = nn.UpsamplingNearest2d(scale_factor=(i, j))

                    # 对 inputCPU 和 inputMPS 分别进行双线性插值
                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    # 断言 CPU 和 MPS 设备上的插值结果相等
                    self.assertEqual(outputCPU, outputMPS)

                    # 创建双线性插值层对象 upsample_nearest2d，使用 (i * H, j * W) 作为尺寸参数
                    upsample_nearest2d = nn.UpsamplingNearest2d((i * H, j * W))

                    # 对 inputCPU 和 inputMPS 分别进行双线性插值
                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    # 断言 CPU 和 MPS 设备上的插值结果相等
                    self.assertEqual(outputCPU, outputMPS)

                    # 对 outputCPU 和 outputMPS 进行反向传播，梯度为全 0.3
                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    # 断言 inputCPU 和 inputMPS 的梯度相等
                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 遍历两种内存格式：torch.channels_last 和 torch.contiguous_format
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            # 分别使用不同参数调用 helper 函数进行测试
            helper(1, 1, 4, 4, memory_format=memory_format)
            helper(7, 5, 3, 2, memory_format=memory_format)
    # 定义测试方法 test_upsample_bilinear2d，用于测试双线性插值上采样操作
    def test_upsample_bilinear2d(self):
        # 定义内部辅助函数 helper，用于执行不同参数组合下的测试
        def helper(N, C, H, W):
            # 在 CPU 上创建一个张量 inputCPU，包含 N*C*H*W 个元素，每个元素是浮点数，需要梯度计算
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W)
            # 保留 inputCPU 的梯度
            inputCPU.retain_grad()
            # 从 inputCPU 分离并克隆一个新的张量 inputMPS 到 MPS（神经网络硬件加速器），需要梯度计算
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            
            # 定义一组上采样倍数值
            values = [1, 2, 5, 10, 40]

            # 遍历上采样倍数值的组合
            for i in values:
                for j in values:
                    # 创建双线性插值上采样层 upsample_bilinear2d，指定缩放因子为 (i, j)
                    upsample_bilinear2d = nn.UpsamplingBilinear2d(scale_factor=(i, j))

                    # 对 inputCPU 和 inputMPS 进行双线性插值上采样
                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    # 断言 CPU 和 MPS 上采样的输出结果相等
                    self.assertEqual(outputCPU, outputMPS)

                    # 重新设置 upsample_bilinear2d 的参数为 (i * H, j * W)
                    upsample_bilinear2d = nn.UpsamplingBilinear2d((i * H, j * W))

                    # 对 inputCPU 和 inputMPS 进行双线性插值上采样
                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    # 断言 CPU 和 MPS 上采样的输出结果相等
                    self.assertEqual(outputCPU, outputMPS)

                    # 对 outputCPU 和 outputMPS 进行反向传播，使用全1张量作为梯度
                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    # 断言 inputCPU 和 inputMPS 的梯度相等
                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 分别使用不同的输入参数调用 helper 函数进行测试
        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)
    # 定义一个测试方法，用于测试插值函数的各种情况
    def test_interpolate(self):
        
        # 定义一个辅助函数，用于测试不同的输入情况
        def helper(shape, output_size, scales, mode, align_corners=False):
            # 在 CPU 上生成一个随机张量，设置为需要梯度，用于后续计算
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            inputCPU.retain_grad()  # 保持梯度用于后续验证

            # 将 CPU 上的输入张量拷贝到 MPS（Memory Persistence Storage），并设置为需要梯度
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            # 如果 align_corners 参数为 True 并且输入形状大于三维，并且模式为双线性插值
            if (align_corners is True and len(shape) > 3 and mode == 'bilinear'):
                # 如果提供了缩放因子 scales
                if scales is not None:
                    # 使用双线性插值对 CPU 和 MPS 上的输入进行缩放
                    outputCPU = nn.functional.interpolate(inputCPU, scale_factor=scales, mode=mode, align_corners=align_corners)
                    outputMPS = nn.functional.interpolate(inputMPS, scale_factor=scales, mode=mode, align_corners=align_corners)
                else:
                    # 使用双线性插值对 CPU 和 MPS 上的输入进行指定尺寸的调整
                    outputCPU = nn.functional.interpolate(inputCPU, size=output_size, mode=mode, align_corners=align_corners)
                    outputMPS = nn.functional.interpolate(inputMPS, size=output_size, mode=mode, align_corners=align_corners)
            # 如果没有提供缩放因子 scales
            elif scales is not None:
                # 使用指定模式对 CPU 和 MPS 上的输入进行缩放
                outputCPU = nn.functional.interpolate(inputCPU, scale_factor=scales, mode=mode)
                outputMPS = nn.functional.interpolate(inputMPS, scale_factor=scales, mode=mode)
            else:
                # 使用指定模式对 CPU 和 MPS 上的输入进行指定尺寸的调整
                outputCPU = nn.functional.interpolate(inputCPU, size=output_size, mode=mode)
                outputMPS = nn.functional.interpolate(inputMPS, size=output_size, mode=mode)

            # 断言 CPU 和 MPS 上的输出应该相等
            self.assertEqual(outputCPU, outputMPS)

            # 反向传播，使用梯度值 0.6 进行反向传播
            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
            
            # 断言 CPU 和 MPS 上的输入梯度应该相等
            self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 对 1D 插值进行测试
        for mode in ['nearest', 'nearest-exact']:
            helper([2, 3, 4], [3], None, mode)  # 使用指定尺寸进行下采样
            helper([2, 3, 4], [6], None, mode)  # 使用指定尺寸进行上采样
            helper([2, 3, 4], None, [0.6], mode)  # 使用缩放因子进行下采样
            helper([2, 3, 4], None, [1.7], mode)  # 使用缩放因子进行上采样
        
        # 对 2D 插值进行测试
        for mode in ['nearest', 'nearest-exact', 'bilinear']:
            helper([2, 3, 4, 5], [3, 4], None, mode)  # 使用指定尺寸进行下采样（最近邻插值）
            helper([2, 3, 4, 5], [6, 7], None, mode)  # 使用指定尺寸进行上采样（最近邻插值）
            helper([2, 3, 4, 5], None, [0.6, 0.7], mode)  # 使用缩放因子进行下采样（最近邻插值）
            helper([2, 3, 4, 5], None, [1.4, 1.7], mode)  # 使用缩放因子进行上采样（最近邻插值）
        
        # 测试 align_corners=True 的情况
        helper([2, 3, 4, 5], [3, 4], None, 'bilinear', True)  # 使用指定尺寸进行下采样（双线性插值）
        helper([2, 3, 4, 5], None, [1.4, 1.7], 'bilinear', True)  # 使用缩放因子进行上采样（双线性插值）

    # 测试连接前向传播
    # 定义测试方法 test_cat1
    def test_cat1(self):
        # 定义内部辅助函数 helper，用于执行以下操作：
        def helper(shape_x, shape_y, shape_z):
            # 生成具有指定形状的随机张量，存储在 CPU 上，不需要梯度
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            # 将 cpu_x 分离并克隆到 'mps' 设备（假设是一种自定义设备）
            x = cpu_x.detach().clone().to('mps')

            # 生成具有指定形状的随机张量，存储在 CPU 上，不需要梯度
            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            # 将 cpu_y 分离并克隆到 'mps' 设备
            y = cpu_y.detach().clone().to('mps')

            # 生成具有指定形状的随机张量，存储在 CPU 上，不需要梯度
            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            # 将 cpu_z 分离并克隆到 'mps' 设备
            z = cpu_z.detach().clone().to('mps')

            # 在维度 1 上连接张量 x, y, z，形成一个新的张量 cat
            cat = torch.cat([x, y, z], dim=1)
            # 在维度 1 上连接张量 cpu_x, cpu_y, cpu_z，形成一个新的张量 cat_cpu
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            # 使用断言检查两个连接结果张量是否相等
            self.assertEqual(cat, cat_cpu)

        # 调用 helper 函数，测试不同的输入形状组合
        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        helper([2, 2, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([0, 2, 4, 5], [0, 3, 4, 5], [0, 5, 4, 5])
        helper([2, 2, 6, 5], [0], [2, 5, 6, 5])
        helper([0], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 3, 4, 5], [2, 5, 4, 5], [0])
        helper([2, 2, 6, 5], [2, 0, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 0, 6, 5])

    # 测试栈的前向传播
    # 测试栈操作
    def test_stack(self):
        # 定义辅助函数，用于创建张量并进行栈操作比较
        def helper(shape, dtype=torch.float32):
            # 初始化各变量
            x, cpu_x = None, None
            y, cpu_y = None, None
            z, cpu_z = None, None

            # 根据数据类型选择不同的张量初始化方式
            if (dtype not in [torch.float32, torch.bool]):
                # 生成指定形状的随机整数张量，放在CPU上，不需要梯度
                cpu_x = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                # 将CPU张量克隆到'MPS'设备（假设是某种特定的设备）
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            elif (dtype == torch.bool):
                # 生成指定形状的随机布尔型张量，放在CPU上，不需要梯度
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            else:
                # 生成指定形状的随机浮点型张量，放在CPU上，需要梯度
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                # 将CPU张量克隆到'MPS'设备，并标记为需要梯度
                x = cpu_x.detach().clone().to('mps').requires_grad_()
                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                y = cpu_y.detach().clone().to('mps').requires_grad_()
                cpu_z = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                z = cpu_z.detach().clone().to('mps').requires_grad_()

            # 在指定维度上堆叠三个张量
            stack = torch.stack([x, y, z], dim=1)
            stack_cpu = torch.stack([cpu_x, cpu_y, cpu_z], dim=1)

            # 断言堆叠后的结果是否相同
            self.assertEqual(stack, stack_cpu)

        # 调用辅助函数进行多组测试
        helper([2, 8, 4, 5])
        helper([2, 8, 4, 5], dtype=torch.float16)
        helper([2, 8, 4, 5], dtype=torch.int32)
        helper([2, 8, 4, 5], dtype=torch.int64)
        helper([2, 8, 4, 5], dtype=torch.bool)
        # 空张量测试 - 目前失败！未处理空张量情况！
        # helper([0, 2, 4, 5])

    # 测试绝对值函数
    def test_abs(self):
        # 定义辅助函数，计算张量的绝对值并进行比较
        def helper(shape):
            # 生成指定形状的随机浮点型张量，放在CPU上，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将CPU张量克隆到'MPS'设备
            x = cpu_x.detach().clone().to('mps')

            # 计算张量的绝对值
            abs_result = torch.abs(x)
            abs_result_cpu = torch.abs(cpu_x)

            # 断言计算结果是否相同
            self.assertEqual(abs_result, abs_result_cpu)

        # 调用辅助函数进行测试
        helper((2, 8, 4, 5))

    # 测试对数函数
    def test_log(self):
        # 定义辅助函数，计算张量的自然对数并进行比较
        def helper(shape):
            # 生成指定形状的随机浮点型张量，放在CPU上，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将CPU张量克隆到'MPS'设备
            x = cpu_x.detach().clone().to('mps')

            # 计算张量的自然对数
            log_result = torch.log(x)
            log_result_cpu = torch.log(cpu_x)

            # 断言计算结果是否相同
            self.assertEqual(log_result, log_result_cpu)

        # 调用辅助函数进行测试
        helper((2, 8, 4, 5))
    # 定义一个测试函数，用于测试 torch.log10 的功能
    def test_log_ten(self):
        # 定义一个辅助函数，用于生成指定形状的随机张量并进行测试
        def helper(shape):
            # 在 CPU 上生成一个指定形状的随机张量，数据类型为浮点数，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的张量从 CPU 复制到名为 'mps' 的虚拟设备上
            x = cpu_x.detach().clone().to('mps')

            # 计算 x 的对数以 10 为底的结果
            log_ten_result = torch.log10(x)
            # 在 CPU 上计算 cpu_x 的对数以 10 为底的结果
            log_ten_result_cpu = torch.log10(cpu_x)

            # 断言 MPS 设备计算的结果与 CPU 计算的结果相等
            self.assertEqual(log_ten_result, log_ten_result_cpu)

        # 调用辅助函数，测试指定形状的张量
        helper((2, 8, 4, 5))

    # 定义一个测试函数，用于测试 torch.log2 的功能
    def test_log_two(self):
        # 定义一个辅助函数，用于生成指定形状的随机张量并进行测试
        def helper(shape):
            # 在 CPU 上生成一个指定形状的随机张量，数据类型为浮点数，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的张量从 CPU 复制到名为 'mps' 的虚拟设备上
            x = cpu_x.detach().clone().to('mps')

            # 计算 x 的对数以 2 为底的结果
            log_two_result = torch.log2(x)
            # 在 CPU 上计算 cpu_x 的对数以 2 为底的结果
            log_two_result_cpu = torch.log2(cpu_x)

            # 断言 MPS 设备计算的结果与 CPU 计算的结果相等
            self.assertEqual(log_two_result, log_two_result_cpu)

        # 调用辅助函数，测试指定形状的张量
        helper((2, 8, 4, 5))

    # 定义一个测试函数，用于测试 torch.log1p 的功能
    def test_log1p(self):
        # 定义一个辅助函数，用于生成指定形状的随机张量并进行测试
        def helper(shape):
            # 在 CPU 上生成一个指定形状的随机张量，数据类型为浮点数，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的张量从 CPU 复制到名为 'mps' 的虚拟设备上
            x = cpu_x.detach().clone().to('mps')

            # 计算 x 的对数以 e 为底的结果
            log_result = torch.log1p(x)
            # 在 CPU 上计算 cpu_x 的对数以 e 为底的结果
            log_result_cpu = torch.log1p(cpu_x)

            # 断言 MPS 设备计算的结果与 CPU 计算的结果相等
            self.assertEqual(log_result, log_result_cpu)

        # 调用辅助函数，测试指定形状的张量
        helper((2, 8, 4, 5))

    # 定义一个测试函数，用于测试 torch.logaddexp 的功能
    def test_logaddexp(self):
        # 定义一个辅助函数，用于生成指定形状的随机张量并进行测试
        def helper(shape):
            # 在 CPU 上生成两个指定形状的随机张量，数据类型为浮点数，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的张量从 CPU 复制到名为 'mps' 的虚拟设备上
            x = cpu_x.detach().clone().to('mps')
            y = cpu_y.detach().clone().to('mps')

            # 计算 x 和 y 的对数和的结果
            log_result = torch.logaddexp(x, y)
            # 在 CPU 上计算 cpu_x 和 cpu_y 的对数和的结果
            log_result_cpu = torch.logaddexp(cpu_x, cpu_y)

            # 断言 MPS 设备计算的结果与 CPU 计算的结果相等
            self.assertEqual(log_result, log_result_cpu)

        # 调用辅助函数，测试指定形状的张量
        helper((2, 8, 4, 5))

    # 定义一个测试函数，用于测试 torch.logaddexp2 的功能
    def test_logaddexp2(self):
        # 定义一个辅助函数，用于生成指定形状的随机张量并进行测试
        def helper(shape):
            # 在 CPU 上生成两个指定形状的随机张量，数据类型为浮点数，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的张量从 CPU 复制到名为 'mps' 的虚拟设备上
            x = cpu_x.detach().clone().to('mps')
            y = cpu_y.detach().clone().to('mps')

            # 计算 x 和 y 的对数和以 2 为底的结果
            log_result = torch.logaddexp2(x, y)
            # 在 CPU 上计算 cpu_x 和 cpu_y 的对数和以 2 为底的结果
            log_result_cpu = torch.logaddexp2(cpu_x, cpu_y)

            # 断言 MPS 设备计算的结果与 CPU 计算的结果相等
            self.assertEqual(log_result, log_result_cpu)

        # 调用辅助函数，测试指定形状的张量
        helper((2, 8, 4, 5))

    # 测试 concat 的前向传播功能
    def test_cat2(self):
        # 定义一个辅助函数helper1，用于测试torch.cat在多个张量上的拼接操作
        def helper1(shape_x, shape_y, shape_z, shape_w):
            # 生成指定形状的随机张量，并转移到'mps'设备上
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cpu_w = torch.randn(shape_w, device='cpu', dtype=torch.float, requires_grad=False)
            w = cpu_w.detach().clone().to('mps')

            # 使用torch.cat函数在'dim=1'维度上拼接x、y、z、w四个张量
            cat = torch.cat([x, y, z, w], dim=1)
            # 对应的CPU版本操作，用于验证结果
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z, cpu_w], dim=1)

            # 断言两个拼接结果是否相等
            self.assertEqual(cat, cat_cpu)

        # 定义一个辅助函数helper，用于测试torch.cat在三个张量上的拼接操作
        def helper(shape_x, shape_y, shape_z):
            # 生成指定形状的随机张量，并转移到'mps'设备上
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            # 使用torch.cat函数在'dim=1'维度上拼接x、y、z三个张量
            cat = torch.cat([x, y, z], dim=1)
            # 对应的CPU版本操作，用于验证结果
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            # 断言两个拼接结果是否相等
            self.assertEqual(cat, cat_cpu)

        # 调用helper函数进行多组测试
        helper([2, 8, 4, 5], [2, 10, 4, 5], [2, 6, 4, 5])
        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        # 空测试 - 当前失败！未处理空张量！
        # helper([0, 2, 4, 5], [2, 0, 4, 5], [2, 5, 0, 5])

    # Test isnan
    def test_isnan(self):
        # 定义一个辅助函数helper，用于测试torch.isnan函数的功能
        def helper(shape):
            # 生成指定形状的随机张量，并转移到'mps'设备上
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 随机选择一行并将其元素设置为NaN
            nan_index = [random.randrange(0, shape[0])]
            cpu_x.index_put_(indices=[torch.tensor(nan_index)], values=torch.tensor(float('nan')))
            x = cpu_x.detach().clone().to('mps')

            # 使用torch.isnan函数检查张量x中的NaN值
            isnan_result = torch.isnan(x)
            # 对应的CPU版本操作，用于验证结果
            isnan_result_cpu = torch.isnan(cpu_x)

            # 断言两个结果张量是否相等
            self.assertEqual(isnan_result, isnan_result_cpu)

        # 调用helper函数进行测试
        helper((8, 2, 4, 5))

    # Test reciprocal
    # 定义一个测试方法，用于测试 torch.reciprocal 函数的功能
    def test_reciprocal(self):
        # 定义一个辅助函数 helper，用于执行具体的测试操作，输入参数为 shape
        def helper(shape):
            # 在 CPU 上生成随机数，并设置为浮点数类型，需要梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 cpu_x 的克隆转移到 'mps' 设备上，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算 x 的倒数
            reciprocal_result = torch.reciprocal(x)
            # 计算 cpu_x 的倒数
            reciprocal_result_cpu = torch.reciprocal(cpu_x)

            # 设置与 cpu_x 形状相同的全 1 张量作为梯度
            cpu_grad = torch.ones_like(reciprocal_result_cpu)
            # 将 cpu_grad 转移到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 计算 reciprocal_result 相对于 grad 的梯度
            reciprocal_result.backward(gradient=grad)
            # 计算 reciprocal_result_cpu 相对于 cpu_grad 的梯度
            reciprocal_result_cpu.backward(gradient=cpu_grad)

            # 断言 reciprocal_result 和 reciprocal_result_cpu 是否相等
            self.assertEqual(reciprocal_result, reciprocal_result_cpu)
            # 断言 x 的梯度与 cpu_x 的梯度是否相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 调用 helper 函数，传入形状 (2, 8, 4, 5) 进行测试
        helper((2, 8, 4, 5))

    # 测试 torch.sqrt 函数
    def test_sqrt(self):
        # 定义一个辅助函数 helper，用于执行具体的测试操作，输入参数为 shape
        def helper(shape):
            # 在 CPU 上生成随机数，并设置为浮点数类型，需要梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 cpu_x 的克隆转移到 'mps' 设备上，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算 x 的平方根
            sqrt_result = torch.sqrt(x)
            # 计算 cpu_x 的平方根
            sqrt_result_cpu = torch.sqrt(cpu_x)

            # 设置与 cpu_x 形状相同的全 1 张量作为梯度
            cpu_grad = torch.ones_like(sqrt_result_cpu)
            # 将 cpu_grad 转移到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 计算 sqrt_result 相对于 grad 的梯度
            sqrt_result.backward(gradient=grad)
            # 计算 sqrt_result_cpu 相对于 cpu_grad 的梯度
            sqrt_result_cpu.backward(gradient=cpu_grad)

            # 断言 sqrt_result 和 sqrt_result_cpu 是否相等
            self.assertEqual(sqrt_result, sqrt_result_cpu)
            # 断言 x 的梯度与 cpu_x 的梯度是否相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 调用 helper 函数，传入形状 (2, 8, 4, 5) 进行测试
        helper((2, 8, 4, 5))

    # 测试 torch.nn.ELU、torch.nn.CELU 和 torch.nn.SELU 激活函数
    def test_elu(self):
        # 定义一个辅助函数 helper，用于执行具体的测试操作，输入参数为 shape、alpha 和 memory_format
        def helper(shape, alpha=1.0, memory_format=torch.contiguous_format):
            # 在 CPU 上生成随机数，并设置为浮点数类型
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 将 cpu_x 转移到指定内存格式并设置需要梯度计算
            cpu_x = cpu_x.to(memory_format=memory_format).requires_grad_()

            # 将 cpu_x 的克隆转移到 'mps' 设备上，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_(True)

            # 遍历三种激活函数
            for activation_func in [torch.nn.ELU(alpha=alpha), torch.nn.CELU(alpha=alpha), torch.nn.SELU()]:
                # 计算使用激活函数后的结果
                elu_result = activation_func(x)
                elu_result_cpu = activation_func(cpu_x)

                # 生成与 elu_result_cpu 形状相同的随机梯度
                cpu_grad = torch.randn(elu_result_cpu.shape)
                # 将 cpu_grad 转移到 'mps' 设备上
                grad = cpu_grad.to('mps')

                # 计算 elu_result 相对于 grad 的梯度
                elu_result.backward(gradient=grad)
                # 计算 elu_result_cpu 相对于 cpu_grad 的梯度
                elu_result_cpu.backward(gradient=cpu_grad)

                # 断言 elu_result 和 elu_result_cpu 是否相等
                self.assertEqual(elu_result, elu_result_cpu)
                # 断言 x 的梯度与 cpu_x 的梯度是否相等
                self.assertEqual(x.grad, cpu_x.grad)

        # 针对不同的内存格式和形状进行测试
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            for shape in [(2, 8, 4, 5)]:
                for alpha in [0.000001, 1.0, 2.3, 0.34, 23]:
                    helper(shape, alpha, memory_format)

    # 测试 torch.nn.functional.elu 函数
    def test_elu_strided_output(self):
        # 给定输入 elu_input，是一个形状为 (1, 1024, 500) 的随机张量
        elu_input = torch.randn(1, 1024, 500)
        alpha = float(1)
        inplace = False

        # 对非连续内存的 elu_input 进行测试
        elu_input_noncontiguous = elu_input.transpose(1, 2)
        # 断言使用不同设备（'cpu' 和 'mps'）的 F.elu 结果是否相同
        self.assertEqual(
            F.elu(elu_input_noncontiguous.to('cpu'), alpha, inplace),
            F.elu(elu_input_noncontiguous.to('mps'), alpha, inplace)
        )
    # 定义测试函数 test_glu，用于测试 Torch 中的 GLU 激活函数
    def test_glu(self):
        # 定义辅助函数 helper，接受形状和维度参数
        def helper(shape, dim=0):
            # 在 CPU 上生成随机张量，设备为 CPU，数据类型为 float，需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 CPU 上的张量分离并克隆到 'mps' 设备，并声明需要梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 遍历激活函数列表，此处只有一个 GLU 激活函数
            for activation_func in [torch.nn.GLU(dim=dim)]:
                # 对 x 应用 GLU 激活函数，得到结果 glu_result
                glu_result = activation_func(x)
                # 对 cpu_x 应用 GLU 激活函数，得到结果 glu_result_cpu
                glu_result_cpu = activation_func(cpu_x)

                # 生成与 glu_result_cpu 相同形状的随机梯度 cpu_grad
                cpu_grad = torch.randn(glu_result_cpu.shape)
                # 将 cpu_grad 移动到 'mps' 设备
                grad = cpu_grad.to('mps')

                # 对 glu_result 执行反向传播，指定梯度为 grad
                glu_result.backward(gradient=grad)
                # 对 glu_result_cpu 执行反向传播，指定梯度为 cpu_grad
                glu_result_cpu.backward(gradient=cpu_grad)

                # 断言 glu_result 与 glu_result_cpu 相等
                self.assertEqual(glu_result, glu_result_cpu)
                # 断言 x 的梯度与 cpu_x 的梯度相等
                self.assertEqual(x.grad, cpu_x.grad)

        # 对不同形状的输入进行测试
        for shape in [[4], (2, 4), (2, 8, 4, 6)]:
            for dim in range(len(shape)):
                helper(shape, dim)

    # Test softplus
    # 定义测试函数 test_softplus，用于测试 Torch 中的 Softplus 激活函数
    def test_softplus(self):
        # 定义辅助函数 helper，接受形状、beta、threshold、dtype 参数
        def helper(shape, beta, threshold, dtype):
            # 在 CPU 上生成随机张量，设备为 CPU，数据类型为 dtype，需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
            # 将 CPU 上的张量分离并克隆到 'mps' 设备，并声明需要梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 应用 Softplus 激活函数到 x 上，指定 beta 和 threshold
            softplus_result = torch.nn.Softplus(beta=beta, threshold=threshold)(x)
            # 应用 Softplus 激活函数到 cpu_x 上，指定 beta 和 threshold
            softplus_result_cpu = torch.nn.Softplus(beta=beta, threshold=threshold)(cpu_x)

            # 生成与 softplus_result_cpu 相同形状的随机梯度 cpu_grad
            cpu_grad = torch.randn(softplus_result.shape)
            # 将 cpu_grad 移动到 'mps' 设备
            grad = cpu_grad.to('mps')

            # 对 softplus_result 执行反向传播，指定梯度为 grad
            softplus_result.backward(gradient=grad)
            # 对 softplus_result_cpu 执行反向传播，指定梯度为 cpu_grad
            softplus_result_cpu.backward(gradient=cpu_grad)

            # 断言 softplus_result 与 softplus_result_cpu 相等
            self.assertEqual(softplus_result, softplus_result_cpu)
            # 断言 x 的梯度与 cpu_x 的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 对不同形状、beta、threshold 和 dtype 的组合进行测试
        for shape, beta, threshold, dtype in product(
            [(), (2, 3), (10, 10), (2, 3, 4, 5)],
            [0.5, 1, 2, 3, 4],
            [0.5, 20, 30, 40, 50],
            [torch.float16, torch.float32]
        ):
            helper(shape, beta, threshold, dtype)

    # Test silu
    # 定义测试函数 test_silu，用于测试 Torch 中的 SiLU 激活函数
    def test_silu(self):
        # 定义辅助函数 helper，接受形状参数
        def helper(shape):
            # 在 CPU 上生成随机张量，设备为 CPU，数据类型为 float，需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 CPU 上的张量分离并克隆到 'mps' 设备，并声明需要梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 应用 SiLU 激活函数到 x 上
            silu_result = torch.nn.SiLU()(x)
            # 应用 SiLU 激活函数到 cpu_x 上
            silu_result_cpu = torch.nn.SiLU()(cpu_x)

            # 生成与 silu_result_cpu 相同形状的随机梯度 cpu_grad
            cpu_grad = torch.randn(silu_result_cpu.shape)
            # 将 cpu_grad 移动到 'mps' 设备
            grad = cpu_grad.to('mps')

            # 对 silu_result 执行反向传播，指定梯度为 grad
            silu_result.backward(gradient=grad)
            # 对 silu_result_cpu 执行反向传播，指定梯度为 cpu_grad
            silu_result_cpu.backward(gradient=cpu_grad)

            # 断言 silu_result 与 silu_result_cpu 相等
            self.assertEqual(silu_result, silu_result_cpu)
            # 断言 x 的梯度与 cpu_x 的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 对不同形状的输入进行测试
        for shape in [[], (2, 3), (2, 8, 4, 5)]:
            helper(shape)
    # 定义测试函数：测试将 MPS 张量转换为 CPU 张量的功能
    def test_cast_mps_to_cpu(self):
        # 定义内部辅助函数 helper，用于测试不同的源和目标数据类型
        def helper(src_dtype, dst_dtype):
            # 创建一个随机张量作为输入，指定数据类型为 src_dtype
            input = torch.rand((1, 3, 128, 128), dtype=src_dtype)
            # 将输入张量转换为 MPS 格式
            input_cast_mps = input.to('mps')
            # 将 MPS 格式的输入张量转换为 CPU 格式，指定目标数据类型为 dst_dtype
            input_cast_cpu = input_cast_mps.to('cpu', dtype=dst_dtype)

            # 断言：转换后的 CPU 张量应当与初始张量 input 的指定数据类型 dst_dtype 一致
            self.assertEqual(input_cast_cpu, input.to(dtype=dst_dtype))

        # 测试不同数据类型之间的转换
        helper(torch.half, torch.float)
        helper(torch.float, torch.half)

    # 定义测试函数：测试将 MPS 张量转换为 MPS 张量的功能
    def test_cast_mps_to_mps(self):
        # 定义内部辅助函数 helper，用于测试不同的源和目标数据类型
        def helper(src_dtype, dst_dtype):
            # 创建一个随机张量作为输入，指定数据类型为 src_dtype
            input_cpu = torch.rand((1, 3, 128, 128), dtype=src_dtype)
            # 将 CPU 格式的输入张量转换为 MPS 格式
            input_mps = input_cpu.to('mps')
            # 将 MPS 格式的输入张量转换为指定数据类型 dst_dtype
            output_mps = input_mps.to(dtype=dst_dtype)
            # 将 CPU 格式的输入张量转换为指定数据类型 dst_dtype
            output_cpu = input_cpu.to(dtype=dst_dtype)

            # 断言：转换后的 MPS 张量应当与转换后的 CPU 张量相同
            self.assertEqual(output_mps.cpu(), output_cpu)

        # 测试不同数据类型之间的转换
        helper(torch.half, torch.float)
        helper(torch.float, torch.half)
        helper(torch.half, torch.long)
        helper(torch.float, torch.int)

    # 定义测试函数：测试带有 count_include_pad 参数的 2D 平均池化操作
    def test_avg_pool2d_count_include_pad(self):
        # 创建一个在 CPU 上的随机张量，需要梯度计算
        cpu_x = torch.randn((1, 3, 9, 9), device='cpu', dtype=torch.float, requires_grad=True)
        # 对 CPU 张量进行深度复制并将其转换为 MPS 格式，并保留梯度计算
        x = cpu_x.detach().clone().to('mps').requires_grad_()
        # 创建一个 2D 平均池化层，指定核大小、填充、步长、是否包含填充等参数
        pool = torch.nn.AvgPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), ceil_mode=True, count_include_pad=True)
        # 在 CPU 张量上应用平均池化操作作为参考值
        ref_y = pool(cpu_x)
        # 在 MPS 张量上应用平均池化操作
        y = pool(x)

        # 断言：经过平均池化后的 MPS 张量应当与在 CPU 张量上进行平均池化后的结果相同
        self.assertEqual(y, ref_y)

        # 创建一个与参考输出形状相同的随机 CPU 梯度张量
        cpu_grad = torch.randn(ref_y.shape)
        # 将 CPU 梯度张量转换为 MPS 格式
        grad = cpu_grad.to('mps')
        # 在参考输出上执行反向传播，计算 CPU 张量的梯度
        ref_y.backward(gradient=cpu_grad)
        # 在 MPS 输出上执行反向传播，计算 MPS 张量的梯度
        y.backward(gradient=grad)

        # 断言：MPS 格式的输入张量 x 的梯度应当与 CPU 格式的输入张量 cpu_x 的梯度相同
        self.assertEqual(x.grad, cpu_x.grad)
    # 定义测试函数 test_adaptive_avg_pool2d_simple
    def test_adaptive_avg_pool2d_simple(self):
        
        # 定义辅助函数 helper，用于测试不同输入形状和输出形状的情况
        def helper(input_shape, out_shape, channels_last):
            # 生成符合输入形状的随机张量 cpu_x，数据类型为 float，需要梯度计算
            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            
            # 如果 channels_last 为真，则转换为通道在最后的内存格式，并保留梯度
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            
            # 将 cpu_x 克隆并转换到 'mps'（Memory Per Storage）格式，需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 使用 AdaptiveAvgPool2d 对象处理输入张量 x，得到 avg_result
            avg_result = torch.nn.AdaptiveAvgPool2d(out_shape)(x)
            
            # 使用 AdaptiveAvgPool2d 对象处理输入张量 cpu_x，得到 avg_result_cpu
            avg_result_cpu = torch.nn.AdaptiveAvgPool2d(out_shape)(cpu_x)
            
            # 生成与 avg_result_cpu 形状相同的随机张量 cpu_grad
            cpu_grad = torch.randn(avg_result_cpu.shape)
            
            # 将 cpu_grad 转换到 'mps'（Memory Per Storage）格式
            grad = cpu_grad.to('mps')
            
            # 对 avg_result 进行反向传播，传入梯度 grad
            avg_result.backward(gradient=grad)
            
            # 对 avg_result_cpu 进行反向传播，传入梯度 cpu_grad
            avg_result_cpu.backward(gradient=cpu_grad)
            
            # 断言 avg_result 和 avg_result_cpu 相等
            self.assertEqual(avg_result, avg_result_cpu)
            
            # 断言 x 的梯度与 cpu_x 的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)
        
        # 使用 helper 函数进行多组测试
        helper((2, 2, 4, 4), (2, 2), False)
        helper((2, 2, 9, 9), (3, 3), False)
        helper((2, 2, 9, 9), (9, 9), False)
        helper((2, 2, 16, 16), (2, 2), False)
        helper((2, 2, 16, 16), (2, 16), False)
        helper((2, 16, 16), (4, 4), False)
        
        # 输出形状大于输入形状的情况
        helper((2, 2, 4, 4), (8, 8), False)
        helper((2, 2, 2, 2), (4, 4), False)
        helper((2, 2, 3, 3), (9, 9), False)
        helper((2, 2, 2, 2), (16, 16), False)
        helper((2, 2, 2, 16), (16, 16), False)
        helper((2, 4, 4), (16, 16), False)
        
        # 测试抛出异常的情况，期望抛出异常并捕获
        try:
            helper((2, 2, 3, 3), (7, 7), False)
        except Exception as e:
            pass

    # 测试最大均值池化（max avg pool2d）- 当输入大小是输出大小的倍数时
    # 目前不测试 channels last 的情况
    # 定义测试函数 test_adaptive_max_pool2d_simple
    def test_adaptive_max_pool2d_simple(self):
        
        # 定义辅助函数 helper，用于测试不同参数的 AdaptiveMaxPool2d 函数
        def helper(input_shape, out_shape, return_indices, dtype, channels_last=False):
            # 初始化一个 CPU 上的输入张量 cpu_x
            cpu_x = None
            # 根据 dtype 类型生成不同类型的随机张量或整数张量
            if (dtype in [torch.float16, torch.float32]):
                cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            else:
                cpu_x = torch.randint(50, input_shape, device='cpu', dtype=dtype, requires_grad=True)
            # 如果 channels_last 为 True，则将 cpu_x 转换为通道最后格式并保留梯度信息
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            # 将 cpu_x 的克隆副本转换为 'mps' 内存格式，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 初始化变量来存储 AdaptiveMaxPool2d 的输出结果和索引
            max_result, max_indices = None, None
            max_result_cpu, max_indices_cpu = None, None

            # 根据 return_indices 参数选择是否返回索引信息，并分别对 x 和 cpu_x 执行 AdaptiveMaxPool2d 操作
            if (return_indices):
                max_result, max_indices = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(x)
                max_result_cpu, max_indices_cpu = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(cpu_x)
            else:
                max_result = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(x)
                max_result_cpu = torch.nn.AdaptiveMaxPool2d(out_shape, return_indices)(cpu_x)

            # 在 CPU 上创建一个随机梯度张量 cpu_grad，并将其转换为 'mps' 内存格式
            cpu_grad = torch.randn(max_result_cpu.shape)
            grad = cpu_grad.to('mps')

            # 计算 'mps' 格式的 max_result 的反向传播，使用 grad 作为梯度
            max_result.backward(gradient=grad)
            # 计算 CPU 上 max_result_cpu 的反向传播，使用 cpu_grad 作为梯度
            max_result_cpu.backward(gradient=cpu_grad)

            # 断言 'mps' 格式的 max_result 与 CPU 上的 max_result_cpu 相等
            self.assertEqual(max_result, max_result_cpu)
            # 如果 return_indices 为 True，则断言索引信息也相等
            if (return_indices):
                self.assertEqual(max_indices, max_indices_cpu)
            # 断言 'mps' 格式的输入张量 x 的梯度与 CPU 上的 cpu_x 的梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 循环遍历不同的数据类型和 return_indices 参数组合，调用 helper 函数进行测试
        for dtype in [torch.float32]:
            for return_indices in [False, True]:
                helper((2, 2, 4, 4), (2, 2), return_indices, dtype)
                helper((2, 2, 9, 9), (3, 3), return_indices, dtype)
                helper((2, 2, 9, 9), (9, 9), return_indices, dtype)
                helper((2, 2, 16, 16), (2, 2), return_indices, dtype)
                helper((2, 2, 16, 16), (2, 16), return_indices, dtype)
                helper((2, 16, 16), (4, 4), return_indices, dtype)
    # 定义一个测试函数 test_gelu_simple，用于测试 GELU 激活函数在不同条件下的行为
    def test_gelu_simple(self):
        # 定义一个辅助函数 helper，用于测试不同形状、数据类型和是否连续的张量
        def helper(shape, dtype=torch.float, contiguous=True):
            # 在 CPU 上生成一个随机张量 cpu_x，并转换到 'mps' 设备
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
            x = cpu_x.detach().clone().to('mps')

            # 如果张量非连续且不是空张量且维度大于等于2，则进行转置以使张量非连续
            if not contiguous and (0 not in shape and len(shape) >= 2):
                # 转置操作会使张量变为非连续
                cpu_x = cpu_x.transpose(0, 1)
                x = x.transpose(0, 1)
                assert not x.is_contiguous()

            # 在 cpu_x 上启用梯度计算
            cpu_x.requires_grad_()
            x.requires_grad_()

            # 对 x 应用 GELU 激活函数，并将结果存储在 gelu_result 中
            gelu_result = torch.nn.GELU()(x)
            # 由于 GELU 不支持在 CPU 上计算，因此将 cpu_x 转换为 float 后再应用 GELU
            gelu_result_cpu = torch.nn.GELU()(cpu_x.to(torch.float))

            # 创建一个与 gelu_result_cpu 形状相同的全为 1 的张量 cpu_grad，并转换到 'mps' 设备
            cpu_grad = torch.ones_like(gelu_result_cpu)
            grad = cpu_grad.to('mps')

            # 在 gelu_result 上执行反向传播，使用 gradient=grad
            gelu_result.backward(gradient=grad)
            # 在 gelu_result_cpu 上执行反向传播，使用 gradient=cpu_grad
            gelu_result_cpu.backward(gradient=cpu_grad)

            # 设置绝对误差和相对误差的容差值
            atol = 1e-5 if dtype == torch.float else 1e-2
            rtol = 1e-3 if dtype == torch.float else 1e-2

            # 断言 gelu_result 与 gelu_result_cpu 的值在给定的容差范围内相等
            self.assertEqual(gelu_result, gelu_result_cpu.to(dtype), atol=atol, rtol=rtol)

            # 断言 x 的梯度不为 None，即梯度已正确计算
            assert x.grad is not None
            # 断言 x 的梯度与 cpu_x 的梯度在给定的容差范围内相等
            self.assertEqual(x.grad, cpu_x.grad, atol=atol, rtol=rtol)

        # 针对不同的数据类型和形状进行测试
        # 第一组测试：空形状 []
        for dtype in [torch.float, torch.half]:
            for shape in [[], (0,), (0, 3), (4,), (4, 3), (5, 4, 3)]:
                for contiguous in [True, False]:
                    helper(shape, dtype, contiguous)

        # 第二组测试：对整数数据类型进行测试，预期会触发 RuntimeError
        for dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            # 使用 lambda 函数调用 torch.nn.GELU()，传入整数类型的随机张量，期望触发 RuntimeError
            self.assertRaises(RuntimeError, lambda: torch.nn.GELU()(torch.randint(100, (2,), dtype=dtype, device="mps")))
    def test_mish_simple(self):
        # 定义一个辅助函数，用于测试 Mish 激活函数的简单情况
        def helper(shape, dtype=torch.float, contiguous=True):
            # 在 CPU 上生成指定形状的随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
            # 创建一个 detach 的克隆张量并转移到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 如果不是连续的并且张量维度大于等于 2，转置会使张量不再是连续的
            if not contiguous and (0 not in shape and len(shape) >= 2):
                # 转置张量使其不再是连续的
                cpu_x = cpu_x.transpose(0, 1)
                x = x.transpose(0, 1)
                assert not x.is_contiguous()

            # 设置 requires_grad
            cpu_x.requires_grad_()
            x.requires_grad_()

            # 计算在 'mps' 设备上的 Mish 激活函数结果
            mish_result = torch.nn.Mish()(x)
            # 计算在 CPU 上的 Mish 激活函数结果
            mish_result_cpu = torch.nn.Mish()(cpu_x)

            # 创建一个与 mish_result_cpu 形状相同的全为 1 的张量
            cpu_grad = torch.ones_like(mish_result_cpu)
            # 将该张量转移到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 在 mish_result 上执行反向传播，使用指定的梯度
            mish_result.backward(gradient=grad)
            # 在 mish_result_cpu 上执行反向传播，使用全为 1 的梯度
            mish_result_cpu.backward(gradient=cpu_grad)

            # 设置绝对误差和相对误差的阈值
            atol = 1e-5 if dtype == torch.float else 1e-2
            rtol = 1e-3 if dtype == torch.float else 1e-2
            # 断言在 'mps' 设备上的 Mish 激活函数结果与在 CPU 上的结果相等
            self.assertEqual(mish_result, mish_result_cpu.to(dtype), atol=atol, rtol=rtol)

            # 断言 'mps' 设备上的张量 x 的梯度不为空
            assert x.grad is not None
            # 断言 'mps' 设备上的张量 x 的梯度与在 CPU 上的张量 cpu_x 的梯度相等
            self.assertEqual(x.grad, cpu_x.grad, atol=atol, rtol=rtol)

        # 对不同的 dtype、shape 和 contiguous 进行测试
        # 包括空 shape 的测试
        for dtype in [torch.float, torch.half]:
            for shape in [[], (0,), (0, 3), (4,), (4, 3), (5, 4, 3)]:
                for contiguous in [True, False]:
                    helper(shape, dtype, contiguous)

    def test_gelu(self):
        # 测试 Gelu 函数的实现
        def _test_gelu(n, m, dtype, contiguous, atol=None, rtol=None):
            # 根据 dtype 确定 numpy 数据类型
            numpy_dtype = {
                torch.bfloat16: torch.float, torch.float: torch.float, torch.double: torch.double
            }[dtype]
            # 设备列表包括 'cpu' 和 'mps'
            devices = ['cpu']
            devices += ['mps']

            # 定义一个参考实现的 Gelu 函数
            def _gelu_ref(X):
                return X * stats.norm.cdf(X)  # noqa: F821

            # 遍历设备列表
            for d in devices:
                # 在指定设备上生成随机张量 X，并设置 requires_grad=True
                X = torch.rand(n, m, dtype=dtype, requires_grad=True, device=d)[:, ::2]
                res = X
                # 将 X 转换为 numpy 数据类型，并从 CPU 取出其副本作为参考实现
                ref = (X.to(numpy_dtype).cpu().detach().numpy())
                # 断言 res（在 'mps' 设备上的结果）与 ref 相等
                self.assertEqual(res, ref, rtol=rtol, atol=atol, exact_dtype=False)

        # 对不同的 n、m 进行 Gelu 函数的测试
        for n in [1, 5, 10]:
            for m in [1, 5, 10]:
                _test_gelu(n, m, torch.float32, True)
                _test_gelu(n, m, torch.float32, False)

        # 测试多线程情况下的 Gelu 函数
        num_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        try:
            _test_gelu(32, 32, torch.float32, False)
        finally:
            torch.set_num_threads(num_threads)

    def test_gelu_tanh(self):
        # 测试使用 Tanh 近似方法的 Gelu 函数
        def helper(shape):
            # 在 CPU 上生成指定形状的随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            # 创建一个 detach 的克隆张量并转移到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 计算在 'mps' 设备上的 Gelu-Tanh 结果
            gelu_tanh_result = torch.nn.functional.gelu(x, approximate='tanh')
            # 计算在 CPU 上的 Gelu-Tanh 结果
            gelu_tanh_result_cpu = torch.nn.functional.gelu(cpu_x, approximate='tanh')
            # 断言 Gelu-Tanh 结果在 'mps' 设备上与在 CPU 上的结果相等
            self.assertEqual(gelu_tanh_result, gelu_tanh_result_cpu)

        # 测试指定形状的 Gelu-Tanh 函数
        helper((2, 8, 4, 5))
    # Test hardtanh
    # 测试 Hardtanh 函数

    def test_hardtanh(self):
        # 定义辅助函数 helper，用于测试不同参数下的 Hardtanh 函数行为
        def helper(shape, min_val, max_val, inplace=False):
            cpu_x = None
            x = None

            # 如果 inplace 参数为 False，则创建需要梯度的张量 x 和其对应的 CPU 版本 cpu_x
            if (not inplace):
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()
            # 如果 inplace 参数为 True，则创建不需要梯度的张量 x 和其对应的 CPU 版本 cpu_x
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')

            # 使用给定的参数创建 Hardtanh 对象，并对 x 和 cpu_x 进行 Hardtanh 运算
            hardtanh_result = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)(x)
            hardtanh_result_cpu = torch.nn.Hardtanh(min_val=min_val, max_val=max_val, inplace=inplace)(cpu_x)

            # 断言 MPS 和 CPU 版本的 Hardtanh 结果应该相等
            self.assertEqual(hardtanh_result, hardtanh_result_cpu)

            # 如果 inplace 参数为 False，则进行反向传播测试
            if (not inplace):
                cpu_grad = torch.randn(hardtanh_result_cpu.shape)
                grad = cpu_grad.to('mps')
                hardtanh_result.backward(gradient=grad)
                hardtanh_result_cpu.backward(gradient=cpu_grad)
                # 断言 MPS 和 CPU 版本的梯度应该相等
                self.assertEqual(x.grad, cpu_x.grad)

        # 对不同形状和 min_val、max_val 组合进行测试
        # 同时测试 inplace 和非 inplace 模式
        for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
            for min_val, max_val in zip([-1, -2, 3], [1, -1, 4]):
                helper(shape, min_val, max_val)
                helper(shape, min_val, max_val, inplace=True)

    # Test hardswish
    # 测试 Hardswish 函数

    def test_hardswish(self):
        # 定义辅助函数 helper，用于测试不同参数下的 Hardswish 函数行为
        def helper(shape, inplace=False, requires_grad=True):
            m = nn.Hardswish(inplace=inplace)

            # 创建需要梯度的 CPU 张量 input_cpu 和其对应的 MPS 张量 input_mps
            input_cpu = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=requires_grad)
            input_mps = input_cpu.detach().clone().to('mps').requires_grad_(requires_grad)

            # 如果 inplace 和 requires_grad 同时为 True，则检查是否会抛出 RuntimeError
            if inplace and requires_grad:
                self.assertRaises(RuntimeError, lambda: m(input_cpu))
                self.assertRaises(RuntimeError, lambda: m(input_mps))
                return

            # 对 input_cpu 和 input_mps 应用 Hardswish 函数
            output_cpu = m(input_cpu)
            output_mps = m(input_mps)

            # 断言 MPS 和 CPU 版本的输出应该相等
            self.assertEqual(output_cpu, output_mps)

            # 如果 requires_grad 为 True，则进行反向传播测试
            if requires_grad:
                cpu_grad = torch.ones_like(output_cpu)
                mps_grad = cpu_grad.to('mps')

                output_cpu.backward(gradient=cpu_grad)
                output_mps.backward(gradient=mps_grad)

                # 断言 MPS 和 CPU 版本的梯度应该相等
                self.assertEqual(input_cpu.grad, input_mps.grad)

        # 对不同形状和参数组合进行测试
        for shape in [(0, 3), [], (2, 3), (2, 8, 4, 5)]:
            helper(shape, inplace=False, requires_grad=False)
            helper(shape, inplace=True, requires_grad=False)
            helper(shape, inplace=False, requires_grad=True)
            helper(shape, inplace=True, requires_grad=True)
    # 定义测试方法，用于测试二维数组的转置操作
    def test_transpose_2D(self):
        # 定义一个二维数组 values
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        # 定义另一个二维数组 values1，所有元素均为 1.0
        values1 = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # 创建一个 PyTorch 的张量 cpu_x，指定在 CPU 上运行
        cpu_x = torch.tensor(values, device='cpu')
        # 创建一个 PyTorch 的张量 mps_x，指定在 mps 设备（假设为某种特定设备）上运行
        mps_x = torch.tensor(values, device='mps')
        # 创建另一个 PyTorch 的张量 mps_x1，也指定在 mps 设备上运行
        mps_x1 = torch.tensor(values1, device='mps')

        # 对 cpu_x 进行转置操作，交换维度 0 和 1
        cpu_transpose = torch.transpose(cpu_x, 0, 1)
        # 对 mps_x 进行转置操作，交换维度 0 和 1
        mps_transpose = torch.transpose(mps_x, 0, 1)
        # 断言转置后的 cpu_x 和 mps_x 在 CPU 上的结果相等
        self.assertEqual(cpu_transpose, mps_transpose.to('cpu'))

    # 定义测试方法，用于测试三维数组的转置操作
    def test_transpose_3D(self):
        # 定义一个三维数组 values
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        # 创建一个 PyTorch 的张量 cpu_x，指定在 CPU 上运行
        cpu_x = torch.tensor(values, device='cpu')
        # 创建一个 PyTorch 的张量 mps_x，指定在 mps 设备上运行
        mps_x = torch.tensor(values, device='mps')

        # 对 cpu_x 进行转置操作，交换维度 0 和 1
        cpu_transpose1 = torch.transpose(cpu_x, 0, 1)
        # 对 mps_x 进行转置操作，交换维度 0 和 1，并转移到 CPU 上
        mps_transpose1 = torch.transpose(mps_x, 0, 1).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose1, mps_transpose1)

        # 对 cpu_x 进行转置操作，交换维度 0 和 2
        cpu_transpose2 = torch.transpose(cpu_x, 0, 2)
        # 对 mps_x 进行转置操作，交换维度 0 和 2，并转移到 CPU 上
        mps_transpose2 = torch.transpose(mps_x, 0, 2).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose2, mps_transpose2)

        # 对 cpu_x 进行转置操作，交换维度 1 和 2
        cpu_transpose3 = torch.transpose(cpu_x, 1, 2)
        # 对 mps_x 进行转置操作，交换维度 1 和 2，并转移到 CPU 上
        mps_transpose3 = torch.transpose(mps_x, 1, 2).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose3, mps_transpose3)

    # 定义测试方法，用于测试四维数组的转置操作
    def test_transpose_4D(self):
        # 定义一个四维数组 values
        values = [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
                  [[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]]]
        # 创建一个 PyTorch 的张量 cpu_x，指定在 CPU 上运行
        cpu_x = torch.tensor(values, device='cpu')
        # 创建一个 PyTorch 的张量 mps_x，指定在 mps 设备上运行
        mps_x = torch.tensor(values, device='mps')

        # 对 cpu_x 进行转置操作，交换维度 0 和 1
        cpu_transpose1 = torch.transpose(cpu_x, 0, 1)
        # 对 mps_x 进行转置操作，交换维度 0 和 1，并转移到 CPU 上
        mps_transpose1 = torch.transpose(mps_x, 0, 1).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose1, mps_transpose1)

        # 对 cpu_x 进行转置操作，交换维度 0 和 2
        cpu_transpose2 = torch.transpose(cpu_x, 0, 2)
        # 对 mps_x 进行转置操作，交换维度 0 和 2，并转移到 CPU 上
        mps_transpose2 = torch.transpose(mps_x, 0, 2).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose2, mps_transpose2)

        # 对 cpu_x 进行转置操作，交换维度 0 和 3
        cpu_transpose3 = torch.transpose(cpu_x, 0, 3)
        # 对 mps_x 进行转置操作，交换维度 0 和 3，并转移到 CPU 上
        mps_transpose3 = torch.transpose(mps_x, 0, 3).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose3, mps_transpose3)

        # 对 cpu_x 进行转置操作，交换维度 3 和 1
        cpu_transpose4 = torch.transpose(cpu_x, 3, 1)
        # 对 mps_x 进行转置操作，交换维度 3 和 1，并转移到 CPU 上
        mps_transpose4 = torch.transpose(mps_x, 3, 1).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose4, mps_transpose4)

        # 对 cpu_x 进行转置操作，交换维度 3 和 2
        cpu_transpose5 = torch.transpose(cpu_x, 3, 2)
        # 对 mps_x 进行转置操作，交换维度 3 和 2，并转移到 CPU 上
        mps_transpose5 = torch.transpose(mps_x, 3, 2).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose5, mps_transpose5)

        # 对 cpu_x 进行转置操作，交换维度 1 和 2
        cpu_transpose6 = torch.transpose(cpu_x, 1, 2)
        # 对 mps_x 进行转置操作，交换维度 1 和 2，并转移到 CPU 上
        mps_transpose6 = torch.transpose(mps_x, 1, 2).to('cpu')
        # 断言转置后的结果在 CPU 上相等
        self.assertEqual(cpu_transpose6, mps_transpose6)
    # 定义一个测试函数，用于测试 torch.sign 函数的功能
    def test_sign(self):
        # 定义一个辅助函数，生成指定形状的随机张量，包括梯度信息，存储在 CPU 上
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 cpu_x 的数据拷贝到 mps 设备上，并保留梯度信息
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算 x 的符号函数结果
            sign_result = torch.sign(x)
            # 计算 cpu_x 的符号函数结果
            sign_result_cpu = torch.sign(cpu_x)

            # 生成一个与 sign_result_cpu 形状相同的全为 1 的张量，存储在 CPU 上
            cpu_grad = torch.ones_like(sign_result_cpu)
            # 将 cpu_grad 张量的数据拷贝到 mps 设备上
            grad = cpu_grad.to('mps')

            # 对 x 执行反向传播，使用 grad 作为梯度
            sign_result.backward(gradient=grad)
            # 对 cpu_x 执行反向传播，使用 cpu_grad 作为梯度
            sign_result_cpu.backward(gradient=cpu_grad)

            # 断言两者的结果张量相等
            self.assertEqual(sign_result, sign_result_cpu)

        # 调用 helper 函数，传入指定形状参数 (2, 8, 4, 5)
        helper((2, 8, 4, 5))

    # 定义一个测试函数，用于测试 torch.signbit 函数的功能
    def test_signbit(self):
        # 定义一个辅助函数，生成指定形状和数据类型的随机张量，存储在 CPU 上
        def helper(shape, dtype):
            cpu_x = torch.randn(shape, device='cpu').to(dtype)
            # 将 cpu_x 的数据拷贝到 mps 设备上
            x = cpu_x.clone().to('mps')

            # 计算 x 的符号位函数结果
            signbit_result = torch.signbit(x)
            # 计算 cpu_x 的符号位函数结果
            signbit_result_cpu = torch.signbit(cpu_x)

            # 断言两者的结果张量相等
            self.assertEqual(signbit_result, signbit_result_cpu)

        # 调用 helper 函数，传入指定形状和数据类型参数
        helper((2, 8, 4, 5), torch.int)
        helper((2, 8, 4, 5), torch.float)
        helper((2, 8, 4, 5), torch.int64)

    # 定义一个测试函数，用于测试 torch.neg 函数的功能
    def test_neg(self):
        # 定义一个辅助函数，生成指定形状的随机张量，包括梯度信息，存储在 CPU 上
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 cpu_x 的数据拷贝到 mps 设备上，并保留梯度信息
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 计算 x 的负值函数结果
            neg_result = torch.neg(x)
            # 计算 cpu_x 的负值函数结果
            neg_result_cpu = torch.neg(cpu_x)

            # 生成一个与 neg_result_cpu 形状相同的全为 1 的张量，存储在 CPU 上
            cpu_grad = torch.ones_like(neg_result_cpu)
            # 将 cpu_grad 张量的数据拷贝到 mps 设备上
            grad = cpu_grad.to('mps')

            # 对 x 执行反向传播，使用 grad 作为梯度
            neg_result.backward(gradient=grad)
            # 对 cpu_x 执行反向传播，使用 cpu_grad 作为梯度
            neg_result_cpu.backward(gradient=cpu_grad)

            # 断言两者的结果张量相等
            self.assertEqual(neg_result, neg_result_cpu)

        # 调用 helper 函数，传入指定形状参数 (2, 8, 4, 5)
        helper((2, 8, 4, 5))

    # 定义一个测试函数，用于测试 torch.neg 函数在输入为 strided 的情况下的功能
    def test_neg_strided_input(self):
        # 创建一个张量 x，包含从 0 到 17 的数值，存储在 mps 设备上，并重新调整形状为 (2, 3, 3)
        x = torch.arange(18.0, device='mps').reshape(2, 3, 3)
        # 对张量 x 进行维度置换，保留最后一个维度的第二列
        y = x.permute(1, 0, 2)[..., 1]
        # 计算 y 加上其负值的结果，即 y + (-y)
        z = y + y.neg()
        # 断言 z 的绝对值的最大值为 0.0
        self.assertEqual(z.abs().max().item(), 0.0)
    def test_index_add(self):
        def helper(shape, dim, index, source_shape, alpha, x_dtype=torch.float32, idx_dtype=torch.int32):
            # 在 CPU 上生成指定形状的随机张量，并设置不需要梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=x_dtype, requires_grad=False)
            # 将生成的张量复制到 MPS 设备上，且不带梯度
            x = cpu_x.detach().clone().to('mps')

            # 创建 CPU 上的索引张量，指定设备和数据类型
            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            # 将索引张量复制到 MPS 设备上，且不带梯度
            idx = cpu_idx.detach().clone().to('mps')

            # 在 CPU 上生成指定形状的随机源张量，并设置不需要梯度计算
            cpu_source = torch.randn(source_shape, device='cpu', dtype=x_dtype, requires_grad=False)
            # 将生成的源张量复制到 MPS 设备上，且不带梯度
            source = cpu_source.detach().clone().to('mps')

            # 在指定维度上执行索引添加操作，alpha 是可选的缩放因子
            idx_result = torch.index_add(x, dim=dim, index=idx, source=source, alpha=alpha)
            # 在 CPU 张量上执行相同的索引添加操作，以便进行结果比较
            idx_result_cpu = torch.index_add(cpu_x, dim=dim, index=cpu_idx, source=cpu_source, alpha=alpha)
            # 断言 MPS 和 CPU 张量执行索引添加后的结果相等
            self.assertEqual(idx_result, idx_result_cpu)

        # 不同测试用例调用 helper 函数，进行索引添加的测试
        helper((2, 8, 4, 5), 0, [0, 1, 0], (3, 8, 4, 5), 5)
        helper((8, 8, 4, 5), 0, [7], (1, 8, 4, 5), 6.0)
        helper((2, 8, 4, 5), 1, [0, 3, 7], (2, 3, 4, 5), 5)
        helper((2, 8, 4, 5), 2, [3, 0], (2, 8, 2, 5), 3.0)
        helper((2, 8, 4, 5), 3, [2, 3, 0], (2, 8, 4, 3), 4)
        helper((2, 3, 3), -1, [1, 2], (2, 3, 2), 6.0)
        # 测试结果维度为1的情况
        helper((2,), 0, [1], (1,), 6.0)
        helper(2, 0, 1, 1, 6)
        # 测试使用 float16 数据类型的情况
        helper((2,), 0, [1], (1,), 6.0, x_dtype=torch.float16)

    def test_index_64bit(self):
        """ 测试索引操作对于超过4GB的张量是否有效 """
        if product_version < 14.0:
            raise unittest.SkipTest("Sonoma is needed for large tensors, see https://github.com/pytorch/pytorch/issues/84039")
        # 清理内存
        gc.collect()
        torch.mps.empty_cache()
        # 检查索引操作对于超过4GB的张量是否有效
        x = torch.rand(16000, 67120, device="mps")
        self.assertGreater(x.element_size() * x.numel(), 2**32)
        idx = torch.arange(0, 2, device="mps")
        x_sampled = x[:, idx]
        self.assertEqual(x[:, 0], x_sampled[:, 0])
        # 在运行测试后回收内存
        del x
        gc.collect()
        torch.mps.empty_cache()
    # 定义一个测试方法，用于测试对于大于32K索引的矩阵的矩阵乘法运算是否有效
    def test_mm_large(self):
        """ Test that MM works for matrices with index larger than 32K """
        # 创建一个随机张量x，形状为(10, 1)，使用MPS设备加速
        x = torch.rand(10, 1, device="mps")
        # 创建一个随机张量y，形状为(1, 32769)，使用MPS设备加速
        y = torch.rand(1, 32769, device="mps")
        # 这段代码曾经会崩溃，错误信息为:
        # error: subRange.start (24576) is not less than length of dimension[0] (16384)
        # 参考链接：https://github.com/pytorch/pytorch/issues/116769#issuecomment-1888302095
        # 验证乘积的最大绝对值是否不为0
        self.assertNotEqual(torch.mm(x, y[:, 16384:32768]).abs().max().item(), 0.0)

        # 定义一个比较矩阵乘法结果的辅助函数
        def compare_mm(m, n, k, dtype=torch.float):
            # 创建随机张量x，形状为(m, n)，使用MPS设备加速，指定数据类型为dtype
            x = torch.rand(m, n, device="mps", dtype=dtype)
            # 创建随机张量y，形状为(n, k)，使用MPS设备加速，指定数据类型为dtype
            y = torch.rand(n, k, device="mps", dtype=dtype)
            # 计算x和y的矩阵乘法结果，并移动到CPU上
            z = torch.mm(x, y).cpu()
            # 在CPU上计算x和y的矩阵乘法结果
            z_cpu = torch.mm(x.cpu(), y.cpu())
            # 验证两个结果是否相等
            self.assertEqual(z, z_cpu)

        # 在M1芯片上的MacOS 14.3系统上，此测试曾产生不正确的结果，但在Metal上是正确的
        compare_mm(1024, 1, 32769)
        # 反过来测试一次，但维度进行了颠倒
        # 参考链接：https://github.com/pytorch/pytorch/issues/116769#issuecomment-1920066984
        compare_mm(32769, 1, 1025)

        # 如果产品版本大于等于14.0
        if product_version >= 14.0:
            # 测试bfloat16的矩阵乘法
            compare_mm(1024, 1, 32769, torch.bfloat16)

    # 如果总内存小于12GB，则跳过该测试；需要至少12GB内存才能运行该测试
    # 如果产品版本低于14.0，则跳过该测试；在MacOS 13上无法分配4GB张量
    @unittest.skipIf(total_memory < 12_000_000_000, "Needs at least 12Gb RAM to run the test")
    @unittest.skipIf(product_version < 14.0, "Can't allocate 4Gb tensor on MacOS 13")
    # 定义一个测试方法，用于测试复制超过4GB的张量是否有效
    def test_copy_large(self):
        """ Test that copy of 4Gb+ tensors works """
        # 创建一个全为1的张量x，形状为(2^30 + 11,)，数据类型为torch.float32
        x = torch.ones((2**30 + 11,), dtype=torch.float32)
        # 将张量x复制到MPS设备上，命名为y
        y = x.to(device="mps")
        # 验证张量y的所有元素是否等于1.0
        self.assertTrue(torch.all(y == torch.tensor(1.0, device="mps")))
        # 删除y和x张量
        del y
        del x

    # 测试flip操作
    def test_flip(self):
        # 定义一个辅助函数，用于测试flip操作
        def helper(shape, dims):
            # 创建一个随机张量cpu_x，形状为shape，在CPU上，数据类型为torch.float，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将cpu_x从计算图中分离，并克隆到MPS设备上，命名为x
            x = cpu_x.detach().clone().to('mps')

            # 执行flip操作，得到flip_result，同时在CPU上执行flip操作得到flip_result_cpu
            flip_result = torch.flip(x, dims=dims)
            flip_result_cpu = torch.flip(cpu_x, dims=dims)

            # 验证两个flip操作的结果是否相等
            self.assertEqual(flip_result, flip_result_cpu)

        # 分别测试不同的参数组合
        helper((2, 8, 4, 5), [0])
        helper((8, 8, 4, 5), [0, 1])
        helper((2, 8, 4, 5), (0, 1, 2, 3))
        helper((2, 3, 3), (-1,))
        # 测试空的维度
        helper((2, 8, 4, 5), [])
        # 当输入张量的元素个数为1时的测试
        helper((1,), (0,))
        # 当输入张量的元素个数为0时的测试
        helper((0,), (0,))
        # 所有维度都不需要flip的情况
        helper((1, 3), [0])
    def test_index_select(self):
        # 定义内部辅助函数，用于测试 torch.index_select 的功能
        def helper(shape, dim, index, idx_dtype=torch.int32):
            # 在 CPU 上生成指定形状的随机张量，不需要梯度
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的张量克隆到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 创建包含索引值的 CPU 张量
            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            # 将索引张量克隆到 'mps' 设备上
            idx = cpu_idx.detach().clone().to('mps')

            # 使用 torch.index_select 在指定维度上选择数据，返回 'mps' 设备上的结果张量
            idx_result = torch.index_select(x, dim=dim, index=idx)
            # 对比在 CPU 上同样操作的结果
            idx_result_cpu = torch.index_select(cpu_x, dim=dim, index=cpu_idx)

            # 断言 'mps' 设备上的结果与 CPU 上的结果相等
            self.assertEqual(idx_result, idx_result_cpu)

        # 调用 helper 函数进行多种形状和索引的测试
        helper((2, 8, 4, 5), 0, [1])
        helper((8, 8, 4, 5), 0, [0, 3, 2, 7, 6])
        helper((2, 8, 4, 5), 1, [0, 3, 2, 7, 6])
        helper((2, 8, 4, 5), 2, [3, 0, 1])
        helper((2, 8, 4, 5), 3, [2, 3, 0])
        helper((2, 3, 3), -1, [1, 2])
        helper((), 0, [0])
        helper((5), 0, [])

    def test_index_select_scalar(self):
        # 定义处理标量输入的测试函数
        def helper(value, dim, index, idx_dtype=torch.int32):
            # 在 CPU 上生成指定值的标量张量，不需要梯度
            cpu_x = torch.tensor(value, device='cpu', dtype=torch.float, requires_grad=False)
            # 将生成的标量张量克隆到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 创建包含索引值的 CPU 张量
            cpu_idx = torch.tensor(index, device='cpu', dtype=idx_dtype)
            # 将索引张量克隆到 'mps' 设备上
            idx = cpu_idx.detach().clone().to('mps')

            # 使用 torch.index_select 在指定维度上选择数据，返回 'mps' 设备上的结果张量
            idx_result = torch.index_select(x, dim=dim, index=idx)
            # 对比在 CPU 上同样操作的结果
            idx_result_cpu = torch.index_select(cpu_x, dim=dim, index=cpu_idx)

            # 断言 'mps' 设备上的结果与 CPU 上的结果相等
            self.assertEqual(idx_result, idx_result_cpu)

        # 调用 helper 函数测试处理标量输入的情况
        helper(22, 0, [0])
        # 使用断言检查对空列表索引时是否抛出预期的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Index to scalar can have only 1 value"):
            helper(22, 0, [])
    # 定义一个测试函数，用于测试嵌入层和稠密层的反向传播
    def test_embedding_dense_backward(self):
        # 定义内部辅助函数helper，用于执行具体的测试任务
        def helper(n, d, m, idx):
            # 创建一个嵌入层对象embeddingMPS，设定最大范数为True，并指定设备为'mps'
            embeddingMPS = nn.Embedding(n, d, max_norm=True, device='mps')
            # 获取嵌入层的权重，并将其从设备上分离并移动到CPU上
            emedding_weight = embeddingMPS.weight.detach().cpu()
            # 在设备'mps'上创建一个随机张量W_MPS，要求计算其梯度
            W_MPS = torch.randn((m, d), requires_grad=True, device='mps')
            # 创建一个张量idx_MPS，用于索引操作，并移动到设备'mps'
            idx_MPS = torch.tensor(idx, device='mps')
            # 计算a_MPS，使用嵌入层的权重和转置后的W_MPS
            a_MPS = embeddingMPS.weight.clone() @ W_MPS.t()  # weight must be cloned for this to be differentiable
            # 保持a_MPS的梯度以便反向传播
            a_MPS.retain_grad()
            # 计算b_MPS，使用嵌入层根据idx_MPS的索引并乘以W_MPS的转置，此操作会原地修改权重
            b_MPS = embeddingMPS(idx_MPS) @ W_MPS.t()
            # 保持b_MPS的梯度以便反向传播
            b_MPS.retain_grad()
            # 计算out_MPS，将a_MPS和b_MPS相加并在第一维度上添加一个维度
            out_MPS = (a_MPS.unsqueeze(0) + b_MPS)
            # 计算loss_MPS，对out_MPS应用sigmoid函数并计算所有元素的乘积，作为损失值
            loss_MPS = out_MPS.sigmoid().prod()
            # 执行反向传播
            loss_MPS.backward()

            # 在CPU上创建一个新的嵌入层对象embeddingCPU，使用给定的权重
            embeddingCPU = nn.Embedding(n, d, max_norm=True, _weight=emedding_weight)
            # 将W_MPS移动到CPU上并命名为W_CPU
            W_CPU = W_MPS.to('cpu')
            # 创建一个CPU上的索引张量idx_CPU
            idx_CPU = torch.tensor(idx)
            # 计算a_CPU，使用嵌入层的权重和转置后的W_CPU
            a_CPU = embeddingCPU.weight.clone() @ W_CPU.t()  # weight must be cloned for this to be differentiable
            # 保持a_CPU的梯度以便反向传播
            a_CPU.retain_grad()
            # 计算b_CPU，使用嵌入层根据idx_CPU的索引并乘以W_CPU的转置，此操作会原地修改权重
            b_CPU = embeddingCPU(idx_CPU) @ W_CPU.t()
            # 保持b_CPU的梯度以便反向传播
            b_CPU.retain_grad()
            # 计算out_CPU，将a_CPU和b_CPU相加并在第一维度上添加一个维度
            out_CPU = (a_CPU.unsqueeze(0) + b_CPU)
            # 计算loss_CPU，对out_CPU应用sigmoid函数并计算所有元素的乘积，作为损失值
            loss_CPU = out_CPU.sigmoid().prod()
            # 执行反向传播
            loss_CPU.backward()

            # 断言两次运行中的嵌入层权重的梯度应相等
            self.assertEqual(b_CPU.grad, b_MPS.grad)
            self.assertEqual(a_CPU.grad, a_MPS.grad)

        # 使用helper函数执行不同的测试案例
        helper(3, 5, 7, [0, 1, 2])
        helper(3, 6, 7, [0, 1, 2])  # 验证改变形状是否会导致缓存图查找问题
        helper(3, 5, 7, 2)  # 测试标量索引

    # 测试pytorch gather
    # 定义测试函数 test_gather
    def test_gather(self):
        # 定义辅助函数 helper，用于执行 gather 操作的测试
        def helper(shape, dim, idx_shape, idx_dtype=torch.int64):
            # 在 CPU 上生成随机张量 cpu_x，并指定 requires_grad=True 开启梯度跟踪
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 cpu_x 的副本转换到 'mps' 设备上，并继续跟踪梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 生成一个在指定范围内随机整数张量 idx_np，用作 gather 操作的索引
            idx_np = np.random.randint(0, shape[dim], idx_shape)
            
            # 创建 CPU 上的索引张量 cpu_idx，并指定数据类型为 idx_dtype
            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            # 将 cpu_idx 的副本转换到 'mps' 设备上
            idx = cpu_idx.detach().clone().to('mps')

            # 执行 gather 操作，从 x 的 dim 维度中收集索引为 idx 的元素
            gather_result = torch.gather(x, dim=dim, index=idx)
            # 对比使用 cpu_x 的 gather 操作结果，作为参照
            gather_result_cpu = torch.gather(cpu_x, dim=dim, index=cpu_idx)

            # 生成一个在 CPU 上的随机梯度张量 cpu_grad
            cpu_grad = torch.randn(idx_shape, device='cpu', dtype=torch.float)
            # 将 cpu_grad 转换到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 对 gather_result 执行反向传播，使用 grad 作为梯度
            gather_result.backward(gradient=grad)
            # 对比使用 cpu_x 执行的反向传播结果
            gather_result_cpu.backward(gradient=cpu_grad)

            # 断言 gather 操作的结果应当与参照结果 gather_result_cpu 相等
            self.assertEqual(gather_result, gather_result_cpu)
            # 断言 cpu_x 的梯度应当与 x 的梯度相等
            self.assertEqual(cpu_x.grad, x.grad)

        # 调用 helper 函数，测试不同的参数组合
        helper((6, 3, 3), 0, (3, 3, 3))
        helper((2, 3, 3, 3), 0, (10, 3, 3, 3))
        helper((2, 8, 4, 5), 0, (10, 8, 4, 5))
        helper((2, 8, 4, 5), 0, (10, 6, 3, 2))
        helper((8, 8, 4, 5), 0, (6, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (6, 7, 2, 3))
        helper((2, 8, 4, 5), 1, (2, 5, 3, 4))
        helper((2, 8, 4, 5), 2, (1, 8, 10, 3))
        helper((2, 8, 4, 5), 3, (2, 5, 3, 12))

    # 测试 pytorch 的 gather 操作（针对标量情况）
    def test_gather_scalar(self):
        # 定义索引的数据类型为 torch.int64
        idx_dtype = torch.int64
        # 创建一个标量张量 cpu_x，并指定 requires_grad=True 开启梯度跟踪
        cpu_x = torch.tensor(3, device='cpu', dtype=torch.float, requires_grad=True)
        # 将 cpu_x 的副本转换到 'mps' 设备上，并继续跟踪梯度
        x = cpu_x.detach().clone().to('mps').requires_grad_()

        # 指定一个索引列表 idx_np，包含一个元素 [0]
        idx_np = [0]

        # 创建 CPU 上的索引张量 cpu_idx，并指定数据类型为 idx_dtype
        cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
        # 将 cpu_idx 的副本转换到 'mps' 设备上
        idx = cpu_idx.detach().clone().to('mps')

        # 执行 gather 操作，从 x 的 dim=0 维度中收集索引为 idx 的元素
        gather_result = torch.gather(x, dim=0, index=idx)
        # 对比使用 cpu_x 的 gather 操作结果，作为参照
        gather_result_cpu = torch.gather(cpu_x, dim=0, index=cpu_idx)

        # 生成一个在 CPU 上的随机梯度张量 cpu_grad
        cpu_grad = torch.randn([1], device='cpu', dtype=torch.float)
        # 将 cpu_grad 转换到 'mps' 设备上
        grad = cpu_grad.to('mps')

        # 对 gather_result 执行反向传播，使用 grad 作为梯度
        gather_result.backward(gradient=grad)
        # 对比使用 cpu_x 执行的反向传播结果
        gather_result_cpu.backward(gradient=cpu_grad)

        # 断言 gather 操作的结果应当与参照结果 gather_result_cpu 相等
        self.assertEqual(gather_result, gather_result_cpu)
        # 断言 cpu_x 的梯度应当与 x 的梯度相等
        self.assertEqual(cpu_x.grad, x.grad)
    # 定义一个测试函数，用于测试 scatter_add 方法
    def test_scatter_add(self):
        
        # 定义一个辅助函数，用于执行不同形状和类型的测试用例
        def helper(shape, dim, idx_shape, src_shape, idx_dtype=torch.int64, do_add=True):
            # 在 CPU 上生成指定形状的随机张量，并设置为可求导
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将生成的张量复制到 'mps' 设备上，并设置为可求导
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 在 CPU 上生成指定形状的随机源张量，并设置为可求导
            cpu_src = torch.randn(src_shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将生成的源张量复制到 'mps' 设备上，并设置为可求导
            src = cpu_src.detach().clone().to('mps').requires_grad_()
            
            # 如果 do_add 为 True，则随机生成索引，否则使用预定义的索引数组
            idx_np = None
            if (do_add):
                idx_np = np.random.randint(0, shape[dim], idx_shape)
            else:
                idx_np = np.array([[0, 1, 2],
                                   [1, 2, 3],
                                   [2, 3, 4],
                                   [3, 4, 5],
                                   [4, 5, 6]])
            
            # 在 CPU 上生成索引张量，并设置数据类型为 idx_dtype
            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            # 将生成的索引张量复制到 'mps' 设备上
            idx = cpu_idx.detach().clone().to('mps')
            
            # 初始化 scatter 结果变量
            scatter_result = None
            scatter_result_cpu = None
            
            # 根据 do_add 决定调用 scatter_add 或 scatter 方法
            if (do_add):
                scatter_result = torch.scatter_add(x, dim=dim, index=idx, src=src)
                scatter_result_cpu = torch.scatter_add(cpu_x, dim=dim, index=cpu_idx, src=cpu_src)
            else:
                scatter_result = torch.scatter(x, dim=dim, index=idx, src=src)
                scatter_result_cpu = torch.scatter(cpu_x, dim=dim, index=cpu_idx, src=cpu_src)
            
            # 初始化梯度变量
            cpu_grad = None
            grad = None
            
            # 如果索引形状和源张量形状相同，则执行梯度计算和断言
            if (idx_shape == src_shape):
                # 在 CPU 上生成随机梯度张量
                cpu_grad = torch.randn(shape, device='cpu', dtype=torch.float)
                # 将生成的梯度张量复制到 'mps' 设备上
                grad = cpu_grad.to('mps')
                # 对 scatter 结果进行反向传播，使用给定的梯度
                scatter_result.backward(gradient=grad)
                scatter_result_cpu.backward(gradient=cpu_grad)
            
            # 断言 scatter 结果在 'mps' 设备上与在 CPU 上计算的结果一致
            self.assertEqual(scatter_result, scatter_result_cpu)
            # 如果索引形状和源张量形状相同，则断言 'mps' 设备上的梯度与 CPU 上的梯度一致
            if (idx_shape == src_shape):
                self.assertEqual(cpu_x.grad, x.grad)
                self.assertEqual(cpu_src.grad, src.grad)
        
        # 执行一系列测试用例，每个用例包含不同的形状和参数设置
        helper((2, 3), 0, (5, 3), (5, 3))
        helper((2, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5))
        helper((8, 8, 4, 5), 0, (4, 7, 3, 2), (4, 7, 3, 2))
        helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (4, 7, 3, 2))
        helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (8, 8, 4, 5))
        
        helper((2, 8, 4, 5), 1, (2, 20, 4, 5), (2, 20, 4, 5))
        helper((2, 8, 4, 5), 1, (2, 13, 3, 2), (2, 13, 3, 2))
        helper((8, 8, 4, 5), 1, (6, 5, 2, 3), (6, 5, 2, 3))
        helper((8, 8, 4, 5), 1, (3, 4, 2, 2), (6, 5, 2, 3))
        
        helper((4, 5, 9, 8), 2, (4, 5, 13, 8), (4, 5, 13, 8))
        helper((4, 5, 9, 8), 2, (3, 4, 10, 6), (3, 4, 10, 6))
        helper((4, 5, 9, 8), 2, (3, 3, 7, 5), (3, 4, 10, 6))
        
        # 测试 scatter 方法，传入 do_add=False 参数
        helper((8, 3), 0, (5, 3), (5, 3), do_add=False)
        helper((10, 3), 0, (5, 3), (5, 8), do_add=False)
    # 测试 pytorch 的 scatter_add 和 scatter 函数对标量输入的功能
    
    def test_scatter_add_scalar(self):
        # 定义一个辅助函数，用于执行测试
        def helper(idx_dtype=torch.int64, do_add=True):
            # 在 CPU 上创建一个张量，并指定 requires_grad=True 以便计算梯度
            cpu_x = torch.tensor(2, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 CPU 上的张量从计算图中分离，并克隆到 'mps' 设备上，并设置 requires_grad=True
            x = cpu_x.detach().clone().to('mps').requires_grad_()
    
            # 在 CPU 上创建另一个张量，并指定 requires_grad=True
            cpu_src = torch.tensor(3, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 CPU 上的张量从计算图中分离，并克隆到 'mps' 设备上，并设置 requires_grad=True
            src = cpu_src.detach().clone().to('mps').requires_grad_()
    
            # 指定索引，应从进行聚集的轴的范围中获取
            idx_np = [0]
    
            # 在 CPU 上创建索引张量，并指定数据类型为 idx_dtype
            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            # 将 CPU 上的索引张量从计算图中分离，并克隆到 'mps' 设备上
            idx = cpu_idx.detach().clone().to('mps')
    
            # 初始化 scatter 结果变量
            scatter_result = None
            scatter_result_cpu = None
    
            # 根据 do_add 参数选择执行 scatter_add 或 scatter 函数
            if do_add:
                scatter_result = torch.scatter_add(x, dim=0, index=idx, src=src)
                scatter_result_cpu = torch.scatter_add(cpu_x, dim=0, index=cpu_idx, src=cpu_src)
            else:
                scatter_result = torch.scatter(x, dim=0, index=idx, src=src)
                scatter_result_cpu = torch.scatter(cpu_x, dim=0, index=cpu_idx, src=cpu_src)
    
            # 初始化梯度变量
            cpu_grad = None
            grad = None
    
            # 在 CPU 上创建梯度张量，并指定梯度值为 1.2
            cpu_grad = torch.tensor(1.2, device='cpu', dtype=torch.float)
            # 将 CPU 上的梯度张量移动到 'mps' 设备上
            grad = cpu_grad.to('mps')
    
            # 执行反向传播计算梯度
            scatter_result.backward(gradient=grad)
            scatter_result_cpu.backward(gradient=cpu_grad)
    
            # 断言 scatter 结果是否相等
            self.assertEqual(scatter_result, scatter_result_cpu)
            # 断言梯度是否正确传播到原始张量和源张量
            self.assertEqual(cpu_x.grad, x.grad)
            self.assertEqual(cpu_src.grad, src.grad)
    
        # 调用 helper 函数进行测试
        helper()
        helper(do_add=False)
    
    # 测试 pytorch 的 scatter_reduce
    # 定义测试函数，用于验证 scatter 函数在不同参数下的行为是否正确
    def test_scatter_reduce(self):
        # 定义辅助函数，用于执行 scatter 操作的测试
        def helper(shape, dim, idx_shape, src_shape, idx_dtype=torch.int64, reduce_str="sum"):
            # 在 CPU 上生成随机张量，指定梯度需求
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 CPU 上的张量转移到 mps（假设的一种设备）上，并设置需要梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 同上，生成源张量和索引张量
            cpu_src = torch.randn(src_shape, device='cpu', dtype=torch.float, requires_grad=True)
            src = cpu_src.detach().clone().to('mps').requires_grad_()

            # 从指定范围内随机生成索引，以确定 gather 操作的轴
            idx_np = np.random.randint(0, shape[dim], idx_shape)
            cpu_idx = torch.tensor(idx_np, device='cpu', dtype=idx_dtype)
            idx = cpu_idx.detach().clone().to('mps')

            # 执行 scatter 操作，根据给定的维度、索引和源数据进行操作，根据 reduce_str 指定的方式进行汇总
            scatter_result = torch.scatter(x, dim=dim, index=idx, src=src, reduce=reduce_str)
            scatter_result_cpu = torch.scatter(cpu_x, dim=dim, index=cpu_idx, src=cpu_src, reduce=reduce_str)

            # 验证 scatter 操作的结果是否与 CPU 上的结果一致
            self.assertEqual(scatter_result, scatter_result_cpu)

        # 对于不同的 reduce 类型，执行 helper 函数以验证 scatter 函数的行为
        # 注意：这里的 reduce_type 应该是 reduce_str，更正为 reduce_str
        for reduce_str in ["add", "multiply"]:
            helper((2, 3), 0, (5, 3), (5, 3), reduce_str=reduce_str)
            helper((2, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5), reduce_str=reduce_str)
            helper((8, 8, 4, 5), 0, (10, 8, 4, 5), (10, 8, 4, 5), reduce_str=reduce_str)
            helper((8, 8, 4, 5), 0, (4, 7, 3, 2), (4, 7, 3, 2), reduce_str=reduce_str)
            helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (4, 7, 3, 2), reduce_str=reduce_str)
            helper((8, 8, 4, 5), 0, (4, 6, 3, 2), (8, 8, 4, 5), reduce_str=reduce_str)

            helper((2, 8, 4, 5), 1, (2, 20, 4, 5), (2, 20, 4, 5), reduce_str=reduce_str)
            helper((2, 8, 4, 5), 1, (2, 13, 3, 2), (2, 13, 3, 2), reduce_str=reduce_str)
            helper((8, 8, 4, 5), 1, (6, 5, 2, 3), (6, 5, 2, 3), reduce_str=reduce_str)
            helper((8, 8, 4, 5), 1, (3, 4, 2, 2), (6, 5, 2, 3), reduce_str=reduce_str)

            helper((4, 5, 9, 8), 2, (4, 5, 13, 8), (4, 5, 13, 8), reduce_str=reduce_str)
            helper((4, 5, 9, 8), 2, (3, 4, 10, 6), (3, 4, 10, 6), reduce_str=reduce_str)
            helper((4, 5, 9, 8), 2, (3, 3, 7, 5), (3, 4, 10, 6), reduce_str=reduce_str)

    # 测试 torch.is_nonzero 函数的行为
    def test_is_nonzero(self):
        # 验证当输入为零时，函数返回 False
        self.assertFalse(torch.is_nonzero(torch.tensor([0.]).to('mps')))
        # 验证当输入为非零数时，函数返回 True
        self.assertTrue(torch.is_nonzero(torch.tensor([1.5]).to('mps')))
        # 验证当输入为 False 时，函数返回 False
        self.assertFalse(torch.is_nonzero(torch.tensor([False]).to('mps')))
        # 验证当输入为非零整数时，函数返回 True
        self.assertTrue(torch.is_nonzero(torch.tensor([3]).to('mps')))

    # 测试 triu（上三角矩阵函数）
    # Test tril
    def test_tril(self):
        # 定义一个辅助函数来测试 torch.tril 函数
        def helper(shape, diag=0):
            # 在 CPU 上生成随机张量，设置 requires_grad=True，指定数据类型为 float
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将 CPU 上的张量进行分离并克隆到 'mps' 设备上，并设置 requires_grad=True
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 调用 torch.tril 函数生成上三角矩阵，diag 参数指定偏移量
            tril_result = torch.tril(x, diag)
            # 在 CPU 上调用 torch.tril 函数生成上三角矩阵，用于验证
            tril_result_cpu = torch.tril(cpu_x, diag)

            # 生成 CPU 上的随机梯度
            cpu_grad = torch.randn(tril_result_cpu.shape)
            # 将 CPU 上的梯度转移到 'mps' 设备上
            grad = cpu_grad.to('mps')

            # 对 'mps' 设备上的结果张量进行反向传播，使用生成的 'mps' 梯度
            tril_result.backward(gradient=grad)
            # 对 CPU 上的结果张量进行反向传播，使用生成的 CPU 梯度
            tril_result_cpu.backward(gradient=cpu_grad)

            # 断言 'mps' 设备上的结果张量与 CPU 上的结果张量相等
            self.assertEqual(tril_result, tril_result_cpu)
            # 断言 'mps' 设备上的输入张量梯度与 CPU 上的输入张量梯度相等
            self.assertEqual(x.grad, cpu_x.grad)

        # 对不同的形状和 diag 参数值调用辅助函数进行测试
        helper((2, 8, 4, 5))
        helper((2, 8, 4, 5), diag=1)
        helper((2, 8, 4, 5), diag=2)
        helper((2, 8, 4, 5), diag=3)
        helper((2, 8, 4, 5), diag=-1)
        helper((2, 8, 4, 5), diag=-2)
        helper((2, 8, 4, 5), diag=-3)

    # test eye
    def test_eye(self):
        # 定义一个辅助函数来测试 torch.eye 函数
        def helper(n, m, dtype):
            # 初始化 CPU 和 'mps' 设备上的结果为 None
            cpu_result = None
            result = None

            # 根据 n 和 m 的值选择生成单位矩阵的方式，并分别在 CPU 和 'mps' 设备上生成
            if (n == m):
                cpu_result = torch.eye(n, dtype=dtype, device='cpu')
                result = torch.eye(n, dtype=dtype, device='mps')
            else:
                cpu_result = torch.eye(n, m, device='cpu')
                result = torch.eye(n, m, device='mps')

            # 断言 CPU 和 'mps' 设备上生成的结果张量相等
            self.assertEqual(result, cpu_result)

        # 对不同的数据类型和矩阵大小调用辅助函数进行测试
        for dtype in [torch.bool, torch.float16, torch.float32, torch.uint8, torch.int16, torch.int32, torch.int64]:
            helper(2, 2, dtype)
            helper(2, 3, dtype)
            helper(0, 2, dtype)
            helper(0, 0, dtype)
            helper(3, 8, dtype)
            helper(8, 3, dtype)
    def test_diag(self):
        # 定义内部辅助函数，用于测试对角线操作
        def helper(shape, diag=0):
            # 生成具有指定形状的随机张量，位于CPU上，需要梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            # 将张量复制到'MPS'设备，并且需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 对'MPS'设备上的张量进行对角线提取操作
            diag_result = torch.diag(x, diag)
            # 对CPU上的张量进行对角线提取操作
            diag_result_cpu = torch.diag(cpu_x, diag)

            # 断言对角线提取的结果在两个设备上是一致的
            self.assertEqual(diag_result, diag_result_cpu)
            # 注意：以下部分被注释掉的代码段，用于后续梯度计算的测试

        # 遍历不同形状和对角线参数的组合，调用辅助函数进行测试
        for shape in [(5, 5), (5, 6), (6, 5), (5,), (6,)]:
            for diag in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
                helper(shape, diag=diag)

    # Test linspace
    def test_linspace(self):
        # 定义内部辅助函数，用于测试线性空间生成函数
        def helper(start, end, steps, dtype=torch.float32):
            # 在CPU上生成线性空间的张量结果
            cpu_result = torch.tensor(np.linspace(start, end, steps), dtype=dtype)
            # 在'MPS'设备上生成线性空间的张量结果
            result = torch.linspace(start, end, steps, dtype=dtype, device='mps')

            # 断言两种计算结果一致
            self.assertEqual(cpu_result, result)

        # 遍历不同的数据类型，调用辅助函数进行测试
        for dtype in [torch.float32, torch.int32, torch.uint8, torch.int64]:
            helper(2, 5, 10, dtype)
            helper(2, 2, 10, dtype)
            helper(5, 2, 10, dtype)
            helper(2, 2, 0, dtype)

    # Test arange
    def test_arange(self):
        # 断言numpy中的arange与MPS设备上的torch.arange结果一致
        self.assertEqual(np.arange(10), torch.arange(10, device='mps'))
        self.assertEqual(np.arange(7, 1, -1), torch.arange(7, 1, -1, device='mps'))
        self.assertEqual(np.arange(1, 2, .3, dtype=np.float32), torch.arange(1, 2, .3, device='mps'))
        self.assertEqual(np.arange(6.3, dtype=np.float32), torch.arange(6.3, device='mps'))

    def test_arange_empty(self):
        # 在MPS设备上生成空张量，与在CPU上生成空张量进行断言比较
        out_mps = torch.tensor([], device="mps")
        out_cpu = torch.tensor([], device="cpu")

        y_mps = torch.arange(0, 0, 1, out=out_mps)
        y_cpu = torch.arange(0, 0, 1, out=out_cpu)
        self.assertEqual(y_mps, y_cpu)

    # Test range
    def test_range(self):
        # 断言numpy中的range与MPS设备上的torch.range结果一致
        self.assertEqual(np.arange(11, dtype=np.float32), torch.range(0, 10, device='mps'))
        self.assertEqual(np.arange(7, 0, -1, dtype=np.float32), torch.range(7, 1, -1, device='mps'))
        self.assertEqual(np.array([1.0000, 1.3000, 1.6000, 1.9000], dtype=np.float32), torch.range(1, 2, .3, device='mps'))
        self.assertEqual(np.arange(6.3, dtype=np.float32), torch.arange(0, 6.3, device='mps'))

    # Test softmax
    # 定义一个名为 test_softmax 的测试方法
    def test_softmax(self):
        
        # 定义一个内部辅助函数 helper，用于测试 softmax 函数
        def helper(shape, dim, channels_last=False):
            # 在 CPU 上生成一个指定形状的随机张量，数据类型为 float，并允许梯度计算
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            
            # 如果 channels_last 参数为 True，则将张量转换为通道优先的内存格式，并保持梯度计算
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            
            # 创建张量 x，其数据从 cpu_x 复制而来，转换到 'mps' 设备上，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 计算张量 x 的 softmax 结果
            softmax_result = torch.nn.functional.softmax(x, dim=dim)
            # 计算在 CPU 上原始张量 cpu_x 的 softmax 结果
            softmax_result_cpu = torch.nn.functional.softmax(cpu_x, dim=dim)
            
            # 当前不测试 channels last 梯度反向传播
            cpu_grad = None
            grad = None
            
            # 如果 channels_last 参数为 False
            if (not channels_last):
                # 在 CPU 上生成一个指定形状的随机梯度张量
                cpu_grad = torch.randn(shape, device='cpu', dtype=torch.float)
                # 将梯度张量 grad 转换到 'mps' 设备上
                grad = cpu_grad.to('mps')
                
                # 对张量 x 进行 softmax 反向传播，使用梯度 grad
                softmax_result.backward(gradient=grad)
                # 对原始张量 cpu_x 进行 softmax 反向传播，使用梯度 cpu_grad
                softmax_result_cpu.backward(gradient=cpu_grad)
            
            # 断言两种计算方式的 softmax 结果是否一致
            self.assertEqual(softmax_result, softmax_result_cpu)
            # 如果 channels_last 参数为 False，则断言张量 x 的梯度与 cpu_x 的梯度是否一致
            if (not channels_last):
                self.assertEqual(x.grad, cpu_x.grad)
        
        # 定义一个内部辅助函数 helper2，用于测试特定维度的 softmax 函数
        def helper2(dim):
            # 在 CPU 上生成一个标量张量，数据类型为 float，允许梯度计算
            cpu_x = torch.tensor(1.23, device='cpu', dtype=torch.float, requires_grad=True)
            # 创建张量 x，其数据从 cpu_x 复制而来，转换到 'mps' 设备上，并设置为需要梯度计算
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            
            # 计算张量 x 的 softmax 结果
            softmax_result = torch.nn.functional.softmax(x, dim=dim)
            # 计算在 CPU 上原始张量 cpu_x 的 softmax 结果
            softmax_result_cpu = torch.nn.functional.softmax(cpu_x, dim=dim)
            
            # 在 CPU 上生成一个标量张量作为梯度
            cpu_grad = torch.tensor(2.34, device='cpu', dtype=torch.float)
            # 将梯度张量 grad 转换到 'mps' 设备上
            grad = cpu_grad.to('mps')
            
            # 对张量 x 进行 softmax 反向传播，使用梯度 grad
            softmax_result.backward(gradient=grad)
            # 对原始张量 cpu_x 进行 softmax 反向传播，使用梯度 cpu_grad
            softmax_result_cpu.backward(gradient=cpu_grad)
            
            # 断言两种计算方式的 softmax 结果是否一致
            self.assertEqual(softmax_result, softmax_result_cpu)
            # 断言张量 x 的梯度与 cpu_x 的梯度是否一致
            self.assertEqual(x.grad, cpu_x.grad)
        
        # 调用 helper2 函数，测试维度为 0 的 softmax 函数
        helper2(0)
        
        # 对 channels_last 参数和不同形状的输入进行嵌套循环测试
        for channels_last in [False]:
            for shape in [(2, 4, 8, 5), (3, 4, 6, 7, 2)]:
                # 如果形状不是四维且 channels_last 为 True，则跳过当前循环
                if (len(shape) != 4 and channels_last):
                    continue
                
                # 对维度参数 dim 进行循环测试
                for dim in [0, 1, 2, 3, -1, -2, -3]:
                    # 调用 helper 函数，测试不同形状、不同维度和 channels_last 参数的 softmax 函数
                    helper(shape, dim, channels_last)

    # 定义一个名为 test_nan_to_num 的测试方法
    def test_nan_to_num(self):
        # 在 CPU 上创建一个包含特殊值的张量，允许梯度计算
        inputCPU = torch.tensor([float('nan'), float('inf'), -float('inf'), 3.14])
        # 创建张量 inputMPS，其数据从 inputCPU 复制而来，转换到 'mps' 设备上，并设置为需要梯度计算
        inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
        
        # 对 inputCPU 和 inputMPS 分别使用 nan_to_num 函数处理
        outputCPU = torch.nan_to_num(inputCPU, nan=2.0, posinf=1.0, neginf=-1.0)
        outputMPS = torch.nan_to_num(inputMPS, nan=2.0, posinf=1.0, neginf=-1.0)
        
        # 断言处理后的 outputMPS 和 outputCPU 是否相等
        self.assertEqual(outputMPS, outputCPU)
    # 定义测试方法 test_where，用于测试 torch.where 函数的功能
    def test_where(self):
        # 定义辅助函数 helper，用于执行具体的测试逻辑
        def helper(shape, x_shape, y_shape, cond_dtype=torch.bool, x_dtype=torch.float):
            # 在 CPU 上生成随机整数条件向量，并转换为指定设备和数据类型
            cpu_cond = torch.randint(2, shape, device='cpu', dtype=cond_dtype, requires_grad=False)
            # 将条件向量复制到 'mps' 设备上
            cond = cpu_cond.detach().clone().to('mps')

            # 在 CPU 上生成指定形状和数据类型的随机张量 x，并标记需要梯度
            cpu_x = torch.randn(x_shape, device='cpu', dtype=x_dtype, requires_grad=True)
            # 将张量 x 复制到 'mps' 设备上，并标记需要梯度
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # 在 CPU 上生成指定形状和数据类型的随机张量 y，并标记需要梯度
            cpu_y = torch.randn(y_shape, device='cpu', dtype=x_dtype, requires_grad=True)
            # 将张量 y 复制到 'mps' 设备上，并标记需要梯度
            y = cpu_y.detach().clone().to('mps').requires_grad_()

            # 在 CPU 上使用 torch.where 函数根据条件向量选择 x 或 y 的元素组成结果张量，并复制到 'mps' 设备上
            cpu_out = torch.where(cpu_cond, cpu_x, cpu_y)
            out = torch.where(cond, x, y)

            # 在 CPU 上生成随机梯度向量，并复制到 'mps' 设备上
            cpu_grad = torch.randn(cpu_out.shape)
            grad = cpu_grad.to('mps')

            # 对 CPU 上的结果张量 cpu_out 执行反向传播，使用生成的随机梯度向量
            cpu_out.backward(gradient=cpu_grad)
            # 对 'mps' 设备上的结果张量 out 执行反向传播，使用生成的随机梯度向量
            out.backward(gradient=grad)

            # 使用 self.assertEqual 断言测试 'mps' 设备上的结果张量 out 与 CPU 上的结果张量 cpu_out 是否相等
            self.assertEqual(out, cpu_out)
            # 使用 self.assertEqual 断言测试 'mps' 设备上的张量 x 的梯度与 CPU 上的张量 cpu_x 的梯度是否相等
            self.assertEqual(x.grad, cpu_x.grad)
            # 使用 self.assertEqual 断言测试 'mps' 设备上的张量 y 的梯度与 CPU 上的张量 cpu_y 的梯度是否相等
            self.assertEqual(y.grad, cpu_y.grad)

        # 对指定的形状列表进行测试
        for shape in ([(0, 3), [], (2, 3), (9,)]):
            helper(shape, shape, shape)

        # 对额外的不同形状进行测试
        helper((2, 3, 1), (2, 3, 4), (2, 1, 4))
        helper((2, 1, 1), (2, 3, 4), (1, 3, 4))
        helper((1, 1, 1), (1, 1, 4), (2, 3, 1))
        helper([], (1, 1, 4), (2, 3, 1))
        helper([], (2, 3, 4), [])
        helper((5, 2, 3), (2, 3), (2, 3))
        helper((2, 3), (5, 2, 3), (2, 3))
        helper((2, 3), (2, 3), (5, 2, 3))
        helper((2, 3), (5, 2, 3), (6, 5, 2, 3))

        # 测试 torch.where 函数的输出是否正确地调整大小
        # TODO: 当 MPS 上的 out OpInfo 测试启用时移除此行
        output = torch.tensor(0.0, device="mps")
        cond = torch.randint(2, (3, 3), dtype=torch.bool, device="mps")
        inp = torch.rand(3, 3, device="mps")
        other = torch.rand(3, 3, device="mps")
        # 使用 torch.where 函数，并将结果存储在预先分配的 output 张量中
        out = torch.where(cond, inp, other, out=output)
        # 使用 self.assertEqual 断言测试 out 是否与 output 是同一个对象
        self.assertEqual(id(out), id(output))
        # 使用 self.assertEqual 断言测试 out 的形状是否为 (3, 3)
        self.assertEqual(out.shape, (3, 3))

    # 测试常规情况
    def test_normal(self):
        # 定义一个辅助函数，生成指定形状的正态分布随机数张量
        def helper(shape, mean=0.0, std=1.0):
            # 生成指定形状的正态分布随机数张量，存储在 'mps' 设备上
            mps_out = torch.normal(mean, std, shape, device='mps')

            # 创建一个形状相同且值为均值的 numpy 数组
            mean_array = np.ones(shape)
            mean_array *= mean
            # 将 numpy 数组转换为 Torch 张量，存储在 'cpu' 设备上，不需要梯度
            cpu_mean_tensor = torch.tensor(mean_array, device='cpu', dtype=torch.float, requires_grad=False)
            # 将 'cpu' 设备上的张量剥离梯度并克隆到 'mps' 设备上
            mean_tensor = cpu_mean_tensor.detach().clone().to('mps')

            # 创建一个形状相同且值为标准差的 numpy 数组
            std_array = np.ones(shape)
            std_array *= std
            # 将 numpy 数组转换为 Torch 张量，存储在 'cpu' 设备上，不需要梯度
            cpu_std_tensor = torch.tensor(std_array, device='cpu', dtype=torch.float, requires_grad=False)
            # 将 'cpu' 设备上的张量剥离梯度并克隆到 'mps' 设备上
            std_tensor = cpu_std_tensor.detach().clone().to('mps')

            # 使用指定的均值张量和标准差生成正态分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            torch.normal(mean_tensor, std, out=mps_out)

            # 使用指定的均值和标准差张量生成正态分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            torch.normal(mean, std_tensor, out=mps_out)

            # 使用指定的均值张量和标准差张量生成正态分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            torch.normal(mean_tensor, std_tensor, out=mps_out)

            # 使用指定的均值张量和标准差生成正态分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            mps_out = torch.normal(mean_tensor, std)
            # 检查生成的张量形状与均值张量的形状是否相同
            self.assertEqual(mps_out.size(), mean_tensor.size())

            # 使用指定的均值和标准差张量生成正态分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            mps_out = torch.normal(mean, std_tensor)
            # 检查生成的张量形状与标准差张量的形状是否相同
            self.assertEqual(mps_out.size(), std_tensor.size())

            # 推断均值张量和标准差张量的广播形状，并使用它们生成正态分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            inferred_shape = torch.broadcast_shapes(mean_tensor.size(), std_tensor.size())
            mps_out = torch.normal(mean_tensor, std_tensor)
            # 检查生成的张量形状与推断的广播形状是否相同
            self.assertEqual(mps_out.size(), inferred_shape)

        # 使用辅助函数测试指定形状的正态分布生成功能
        helper((2, 3, 4, 5, 6))
        helper((100, 100), 2.5, 1.2)

    def test_bernoulli(self):
        shape = (10, 10)
        # 创建一个形状为 (10, 10) 的全 1 张量，存储在 'mps' 设备上
        all_ones = torch.ones(shape, device='mps')
        # 创建一个形状为 (10, 10) 的全 0 张量，存储在 'mps' 设备上
        all_zeros = torch.zeros(shape, device='mps')

        # 创建一个形状为 (10, 10) 的概率张量，每个元素值为 0.5，存储在 'mps' 设备上
        prob_tensor = all_ones * 0.5
        # 使用概率张量生成伯努利分布随机数，存储在 'mps' 设备上的 mps_out 张量中
        mps_out = torch.bernoulli(prob_tensor)
        # 检查生成的伯努利分布随机数张量在 'cpu' 设备上的均值不等于 0
        self.assertNotEqual(mps_out.to('cpu').mean(), 0.)
        # 检查生成的伯努利分布随机数张量在 'cpu' 设备上的方差平方不等于 0
        self.assertNotEqual(mps_out.to('cpu').std() ** 2, 0.)

        # 使用全 0 张量生成伯努利分布随机数，应返回全 0 张量，存储在 'mps' 设备上的 mps_out 张量中
        mps_out = torch.bernoulli(all_zeros)
        self.assertEqual(mps_out, all_zeros)

        # 使用全 1 张量生成伯努利分布随机数，应返回全 1 张量，存储在 'mps' 设备上的 mps_out 张量中
        mps_out = torch.bernoulli(all_ones)
        self.assertEqual(mps_out, all_ones)

        # 针对不同数据类型循环测试伯努利分布生成功能
        for dtype in [torch.float16, torch.int8, torch.int16, torch.int32, torch.int64]:
            # 使用指定数据类型的全 0 张量生成伯努利分布随机数，存储在 'mps' 设备上的 mps_out 张量中
            mps_out = torch.zeros(shape, device='mps', dtype=dtype).bernoulli(0.5)
            # 检查生成的张量的唯一值是否与指定数据类型的设备上的 [0, 1] 范围内的整数值相等
            if product_version > 13.0:
                uniq = mps_out.unique()
                self.assertEqual(uniq, torch.arange(2, device='mps', dtype=dtype))
            else:
                # 若版本低于 13.0，则检查生成的张量的最小值是否为 0，最大值是否为 1
                self.assertEqual(mps_out.min().item(), 0.)
                self.assertEqual(mps_out.max().item(), 1.)
    def test_mps_generator(self):
        # 创建一个 MPS 生成器并手动设置种子
        g_mps = torch.Generator(device='mps')
        g_mps.manual_seed(999)
        # 使用 MPS 生成器生成随机张量 mps_x
        mps_x = torch.randn(5, device='mps', generator=g_mps)
        # 重新设置相同的种子，生成另一个随机张量 mps_y
        g_mps.manual_seed(999)
        mps_y = torch.randn(5, device='mps', generator=g_mps)
        # 因为种子相同，所以两个随机张量的内容应该相同
        self.assertEqual(mps_x, mps_y)
        # 保存生成器的状态，以便稍后恢复
        g_state = g_mps.get_state()

        # 不使用种子生成随机张量 mps_x
        mps_x = torch.randn(5, device='mps', generator=g_mps)
        # 在这种情况下，新生成的随机结果应该与之前的不同
        self.assertNotEqual(mps_x, mps_y)

        # 恢复先前保存的生成器状态，这时生成的随机张量应该与 mps_y 相同
        g_mps.set_state(g_state)
        mps_x = torch.randn(5, device='mps', generator=g_mps)
        self.assertEqual(mps_x, mps_y)

    def test_default_mps_generator(self):
        # 在“默认”MPS生成器上进行手动种子设置，使用全局的 torch.manual_seed()
        torch.manual_seed(230)
        # 使用“默认”MPS生成器生成随机张量 mps_x
        mps_x = torch.randn(5, device='mps')
        # 使用 torch.mps.manual_seed() 设置“默认”MPS生成器，效果类似于全局的 torch.manual_seed()
        torch.mps.manual_seed(230)
        # 使用“默认”MPS生成器生成随机张量 mps_y
        mps_y = torch.randn(5, device='mps')
        # 因为种子相同，所以两个随机张量的内容应该相同
        self.assertEqual(mps_x, mps_y)

        # 保存“默认”生成器的状态，以便稍后恢复
        g_state = torch.mps.get_rng_state()

        # 不使用种子生成随机张量 mps_x
        mps_x = torch.randn(5, device='mps')
        # 在这种情况下，新生成的随机结果应该与之前的不同
        self.assertNotEqual(mps_x, mps_y)

        # 恢复先前保存的默认生成器状态，这时生成的随机张量应该与 mps_y 相同
        torch.mps.set_rng_state(g_state)
        mps_x = torch.randn(5, device='mps')
        self.assertEqual(mps_x, mps_y)

    def test_device_synchronize(self):
        # 创建一个在 MPS 流上运行的转置卷积网络
        net1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)\
            .to(device='mps', dtype=torch.float)

        # 创建输入张量 x，设备为 MPS，用于梯度计算
        x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
        # 等待 MPS 流完成前面的操作
        torch.mps.synchronize()
        # 在网络上运行输入张量 x
        x = net1(x)
        # 等待 MPS 流完成前面的操作
        torch.mps.synchronize()
        # 计算反向传播
        x.backward(torch.randn_like(x))
        # 等待 MPS 流完成前面的操作
        torch.mps.synchronize()
    # 测试 MPS 分配器模块的功能
    def test_mps_allocator_module(self):
        # 执行垃圾回收并清空缓存的内存块
        gc.collect()
        # 清空 MPS 缓存
        torch.mps.empty_cache()
        # 测量从 MPSAllocator 分配的内存
        current_alloc_before = torch.mps.current_allocated_memory()
        # 垃圾回收和清空缓存后，current_allocated_memory 应为零
        self.assertEqual(current_alloc_before, 0)
        # 测量从 Metal 驱动程序分配的总内存
        driver_alloc_before = torch.mps.driver_allocated_memory()
        # 分配一个新的 8 MB 张量以强制分配新的 Metal 堆
        x = torch.ones(1024 * 1024 * 8, device="mps")
        # 分配张量 x 后的内存分配量
        current_alloc_after = torch.mps.current_allocated_memory()
        driver_alloc_after = torch.mps.driver_allocated_memory()
        # 此时 current_alloc_after 和 driver_alloc_after 应该增加
        self.assertGreater(current_alloc_after, current_alloc_before)
        self.assertGreater(driver_alloc_after, driver_alloc_before)

    # 测试 MPS 分配器模块的统计信息
    def test_mps_allocator_stats(self):
        # 获取推荐的最大内存
        max_memory = torch.mps.recommended_max_memory()
        print(f"Recommended Max Memory : {max_memory/ 1024 ** 3} GB")
        # 确保推荐的最大内存大于零
        self.assertGreater(max_memory, 0)

    # 为验证此测试，运行 XCode Instruments 的 "Metal System Trace" 或 "Logging" 工具，
    # 按下记录，然后运行此 Python 测试，并按下停止。接下来展开 os_signposts->PyTorchMPS，
    # 检查是否记录了事件或时间间隔，例如：
    # "aten::mps_convolution_backward_input:f32[1,128,6,6]:f32[128,64,3,3]:1,128,6,6 (id=G2, run=2)"
    def test_mps_profiler_module(self):
        # 使用 torch.mps.profiler.profile 方法进行事件模式的性能分析
        with torch.mps.profiler.profile(mode="event", wait_until_completed=False) as p:
            # 运行一些操作以捕获用于分析的 OS Signposts 跟踪
            net1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)\
                .to(device='mps', dtype=torch.float)
            x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
            x = net1(x)

        # 开始间隔模式的性能分析
        torch.mps.profiler.start(mode="interval", wait_until_completed=True)
        # 再次运行一些操作以捕获 OS Signposts 跟踪
        x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
        x = net1(x)
        # 停止间隔模式的性能分析
        torch.mps.profiler.stop()
    # 测试 MPS 事件模块的功能
    def test_mps_event_module(self):
        # 创建一个启用计时的 MPS 事件
        startEvent = torch.mps.Event(enable_timing=True)
        # 记录事件开始时间
        startEvent.record()
        # 创建一个 MPS 设备上的转置卷积神经网络层
        net1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)\
            .to(device='mps', dtype=torch.float)
        # 在 MPS 设备上生成随机张量并应用网络层
        x = torch.rand(1, 128, 6, 6, device='mps', dtype=torch.float, requires_grad=True)
        x = net1(x)
        # 创建另一个启用计时的 MPS 事件
        endEvent = torch.mps.Event(enable_timing=True)
        # 记录事件结束时间
        endEvent.record()
        # 计算两个事件之间的经过时间
        elapsedTime = startEvent.elapsed_time(endEvent)
        # 断言经过时间大于0
        self.assertGreater(elapsedTime, 0.0)

    # 测试 JIT 模型的保存与加载
    def test_jit_save_load(self):
        # 创建一个空的神经网络模型
        m = torch.nn.Module()
        # 向模型添加一个张量属性
        m.x = torch.rand(3, 3, device='mps')
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将 JIT 脚本化的模型保存到缓冲区
        torch.jit.save(torch.jit.script(m), buffer)
        # 重置缓冲区位置到起始位置
        buffer.seek(0)
        # 从缓冲区加载模型
        n = torch.jit.load(buffer)
        # 断言加载的模型属性与原始模型属性相等
        self.assertEqual(n.x, m.x)

    # 测试随机生成函数 random_, random_.to 和 random_.from
    def test_random(self):
        # 定义一个辅助函数生成 MPS 设备上的随机张量
        def helper(shape, low, high, dtype=torch.int32):
            mps_out = torch.randint(low, high, shape, dtype=dtype, device='mps')

            # 无法可靠检查均值和标准差，仅确保不返回常量值
            self.assertNotEqual(mps_out.float().mean().item(), 0.)
            self.assertNotEqual(mps_out.float().std().item(), 0.)

        # 使用不同的参数测试 helper 函数
        helper([100, 100], 0, 10)
        helper([100, 100], 23, 89)
        helper([100, 100], 23, 89, dtype=torch.float32)
        helper([100, 100], 23, 89, dtype=torch.int64)
        helper([100, 100], 0, 2, dtype=torch.bool)

        # 测试 random_ 方法
        for dtype in [torch.bool, torch.int8, torch.uint8, torch.int32, torch.float16, torch.float32]:
            x = torch.empty(10, 10, dtype=dtype, device='mps')
            x.random_()
            self.assertNotEqual(x.max().item(), 0)

    # 测试指数分布函数 exponential_
    def test_exponential(self):
        # 定义一个辅助函数在 MPS 设备上生成指数分布随机数
        def helper(shape, lamda, dtype=torch.float32):
            mps_out = torch.zeros(shape, device='mps', dtype=dtype)
            mps_out.exponential_(lamda)

            # 打印 MPS 设备上生成的随机数的平均值和方差与理论值的比较
            print(mps_out.to('cpu').float().mean(), 1 / lamda)
            print(mps_out.to('cpu').float().std() ** 2, 1 / (lamda**2))

        # 使用不同的参数测试 helper 函数
        for dtype in [torch.float32, torch.float16]:
            helper([100, 100], 2, dtype)
            helper([100, 100], 1, dtype)
            helper([100, 100], 3, dtype)
            helper([100, 100], 0.5, dtype)

    def test_exponential_1(self):
        # 测试指数分布类 Exponential 的样本生成
        rate = torch.randn(5, 5).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Exponential(rate).sample().size(), (5, 5))
        self.assertEqual(Exponential(rate).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Exponential(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Exponential(rate_1d).sample().size(), (1,))
        self.assertEqual(Exponential(0.2).sample((1,)).size(), (1,))
        self.assertEqual(Exponential(50.0).sample((1,)).size(), (1,))

    # 测试加法函数 add
    # 定义一个测试方法，用于测试加法和减法操作
    def test_add_sub(self):
        
        # 定义一个辅助函数，用于执行具体的测试
        def helper(shape, alpha, op_name, inplace):
            # 根据操作名称选择相应的加法或减法操作，并根据是否原地操作选择对应的函数
            if op_name == "add":
                op = torch.Tensor.add_ if inplace else torch.add
            elif op_name == "sub":
                op = torch.Tensor.sub_ if inplace else torch.sub
            
            # 遍历浮点数类型，包括 float16 和 float32
            for dtype in [torch.float16, torch.float32]:
                # 在 CPU 上生成随机数据，指定设备和数据类型，不需要梯度
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                # 生成 CPU 上的数据副本，并转移到 'mps' 设备上
                mps_x = cpu_x.detach().clone().to('mps')
                
                # 同样生成另一组随机数据，进行类似的处理
                cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                mps_y = cpu_y.detach().clone().to('mps')
                
                # 在 CPU 和 'mps' 上执行对应的加法或减法操作，带有 alpha 参数
                cpu_out = op(cpu_x, cpu_y, alpha=alpha)
                mps_out = op(mps_x, mps_y, alpha=alpha)
                
                # 如果数据类型是 float16，设置容差 tol，否则不设置
                tol = 2e-3 if dtype is torch.float16 else None
                # 断言 'mps' 和 CPU 上的输出结果近似相等，根据容差设置
                self.assertEqual(mps_out, cpu_out, rtol=tol, atol=tol)
                
                # 如果不是原地操作且 cpu_y 不是标量，则无法广播输出
                if not (cpu_y.shape != () and inplace):  # in-place output cannot be broadcasted.
                    # 创建一个标量张量在 CPU 上
                    cpu_s = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                    # 生成标量张量的 'mps' 副本
                    mps_s = cpu_s.detach().clone().to('mps')
                    # 断言对标量张量和 cpu_y 执行操作的结果在 'mps' 上和 CPU 上相等
                    self.assertEqual(op(cpu_s, cpu_y), op(mps_s, mps_y))
                
                # 创建一个标量张量在 CPU 上
                cpu_s = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                # 生成标量张量的 'mps' 副本
                mps_s = cpu_s.detach().clone().to('mps')
                # 断言对 cpu_x 和标量张量执行操作的结果在 'mps' 上和 CPU 上相等，根据容差设置
                self.assertEqual(op(cpu_x, cpu_s), op(mps_x, mps_s), rtol=tol, atol=tol)
        
        # 使用 product 函数遍历加法和减法操作的所有组合（op_name 和 inplace）
        for op_name, inplace in product(["add", "sub"], [True, False]):
            # 分别对不同形状的张量进行测试，alpha 参数为 0.0, 0.1, 1.0，以及不同的形状
            helper((), 0.0, op_name, inplace)
            helper((2, 8, 4, 5), 0.0, op_name, inplace)
            helper((2, 8, 4, 5), 0.1, op_name, inplace)
            helper((2, 8, 4, 5), 1.0, op_name, inplace)
            helper((2, 8, 3, 5), 0.1, op_name, inplace)
            helper((2, 8, 3, 5), 0.2, op_name, inplace)

    # 测试加法
    # 测试在 MPS 上执行加法操作
    def test_add_scalars(self):
        # 定义辅助函数，用于测试不同的 alpha 值
        def helper(alpha):
            # 遍历不同的数据类型
            for dtype in [torch.float16, torch.float32]:
                # 创建 CPU 上的张量 cpu_x
                cpu_x = torch.tensor(2.3, device='cpu', dtype=dtype, requires_grad=False)
                # 将 cpu_x 分离并克隆到 MPS 上的张量 x
                x = cpu_x.detach().clone().to('mps')

                # 创建 CPU 上的张量 cpu_y
                cpu_y = torch.tensor(3.4, device='cpu', dtype=dtype, requires_grad=False)
                # 将 cpu_y 分离并克隆到 MPS 上的张量 y
                y = cpu_y.detach().clone().to('mps')

                # 在 CPU 上执行加法操作，结果存储在 cpu_out 中
                cpu_out = torch.add(cpu_x, cpu_y, alpha=alpha)
                # 在 MPS 上执行加法操作，结果存储在 out 中
                out = torch.add(x, y, alpha=alpha)
                # 当数据类型为 torch.float16 时，设置容差值为 1e-3，否则为 None
                tol = 1e-3 if dtype is torch.float16 else None
                # 断言 MPS 上的结果 out 与 CPU 上的结果 cpu_out 相等
                self.assertEqual(out, cpu_out, rtol=tol, atol=tol)

        # 测试不同的 alpha 值
        helper(1.0)
        helper(0.0)
        helper(0.1)
        helper(0.2)

        # 测试 int32 张量与 int64 标量相加
        # 参考 https://github.com/pytorch/pytorch/issues/79835#issuecomment-1164984534
        x = torch.ones(4, dtype=torch.int32, device='mps')
        # 断言 int32 张量 x 加 1 的结果与指定的张量相等
        self.assertEqual(x + 1, torch.full((4,), 2, dtype=torch.int32, device='mps'))
        # 断言 int32 张量 x 加 1.5 的结果与指定的张量相等
        self.assertTrue(torch.equal(x + 1.5, torch.full((4,), 2.5, device='mps')))

    # 测试不同类型的二元操作
    def test_types_binary_op(self):
        # Float * Bool
        # 在 CPU 上执行浮点数与布尔值的乘法操作
        cpu_x = torch.arange(5, dtype=torch.float32, device="cpu") * torch.tensor([True, False, True, False, True], device="cpu")
        # 在 MPS 上执行浮点数与布尔值的乘法操作
        mps_x = torch.arange(5, dtype=torch.float32, device="mps") * torch.tensor([True, False, True, False, True], device="mps")
        # 断言 CPU 上的结果 cpu_x 与 MPS 上的结果 mps_x 相等
        self.assertEqual(cpu_x, mps_x)
        
        # Float * Int64
        # 在 CPU 上执行浮点数与 int64 类型的乘法操作
        cpu_y = torch.arange(5, dtype=torch.float32, device="cpu") * torch.tensor([1, 0, 1, 0, 1], device="cpu")
        # 在 MPS 上执行浮点数与 int64 类型的乘法操作
        mps_y = torch.arange(5, dtype=torch.float32, device="mps") * torch.tensor([1, 0, 1, 0, 1], device="mps")
        # 断言 CPU 上的结果 cpu_y 与 MPS 上的结果 mps_y 相等
        self.assertEqual(cpu_y, mps_y)
    # 定义测试一元操作的方法
    def test_unary_ops(self):
        # 定义辅助函数helper，用于执行给定形状和操作的测试
        def helper(shape, op):
            # 测试浮点数类型（torch.float32）
            for dtypef in [torch.float32]:
                # 创建一个在CPU上随机初始化的张量，并确保不需要梯度
                cpu_x = torch.randn(shape, device='cpu', dtype=dtypef, requires_grad=False)
                # 将cpu_x分离出来并克隆到'mps'设备，再测试操作的相等性
                mps_x = cpu_x.detach().clone().to('mps')
                self.assertEqual(op(cpu_x), op(mps_x))

            # 测试整数类型（torch.int32, torch.int16）
            for dtypei in [torch.int32, torch.int16]:
                # 在CPU上生成随机整数张量，范围为0到999（不包括），并确保不需要梯度
                cpu_x = torch.randint(0, 1000, shape, device='cpu', dtype=dtypei, requires_grad=False)
                # 将cpu_x转换到'mps'设备，再测试操作的相等性，设置相对（rtol）和绝对（atol）容差
                mps_x = cpu_x.to('mps')
                self.assertEqual(op(cpu_x), op(mps_x), rtol=1e-4, atol=1e-4)
            
            # 测试切片操作
            for dtypef in [torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtypef, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')
                # 对cpu_x和mps_x进行切片操作，比较切片后的结果是否相等
                cpu_slice = cpu_x[:, ::2, :, :]
                mps_slice = mps_x[:, ::2, :, :]
                self.assertEqual(op(cpu_slice), op(mps_slice))
            
            # 测试视图操作
            for dtypef in [torch.float32]:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtypef, requires_grad=False)
                mps_x = cpu_x.detach().clone().to('mps')
                # 创建一个视图张量，通过减少第三和第四维度来改变形状
                combined_dim = shape[-1] * shape[-2]
                reshaped_dims = list(shape[:-2]) + [combined_dim]
                cpu_view = cpu_x.view(*reshaped_dims)
                mps_view = mps_x.view(*reshaped_dims)
                self.assertEqual(op(cpu_view), op(mps_view))

        # 使用不同形状和操作函数调用helper函数进行测试
        helper((2, 8, 4, 5), torch.exp)
        helper((2, 8, 3, 5), torch.exp2)
        helper((2, 8, 3, 5), torch.expm1)
        helper((2, 8, 3, 5), torch.log)
        helper((2, 8, 3, 5), torch.cos)
        helper((2, 8, 3, 5), torch.erfinv)


    # 定义测试非稠密张量的存储一元操作的方法
    def test_non_dense_in_storage_unary_ops(self):
        # 定义辅助函数helper，用于执行给定操作的测试
        def helper(op):
            # 测试浮点数类型（torch.float32）
            for dtypef in [torch.float32]:
                # 在CPU上随机初始化一个大小为100的张量，并确保不需要梯度
                cpu_x = torch.randn(100, device='cpu', dtype=dtypef, requires_grad=False)
                # 将cpu_x分离出来并克隆到'mps'设备，再测试操作的相等性
                mps_x = cpu_x.detach().clone().to('mps')
                self.assertEqual(op(cpu_x[::2]), op(mps_x[::2]))

            # 测试整数类型（torch.int32, torch.int16, torch.int8）
            for dtypei in [torch.int32, torch.int16, torch.int8]:
                # 在CPU上生成范围在0到126（不包括）之间的随机整数张量，并确保不需要梯度
                cpu_x = torch.randint(127, device='cpu', size=(100,), dtype=dtypei, requires_grad=False)
                # 将cpu_x转换到'mps'设备，再测试操作的相等性，设置相对（rtol）和绝对（atol）容差
                mps_x = cpu_x.to('mps')
                self.assertEqual(op(cpu_x[::2]), op(mps_x[::2]), rtol=1e-4, atol=1e-4)

        # 使用不同操作函数调用helper函数进行测试
        helper(torch.exp)
        helper(torch.exp2)
        helper(torch.expm1)
        helper(torch.log)
        helper(torch.cos)
    def test_unary_ops_storage_offset_strided(self):
        def helper(shape, op, inplace, dtype=torch.float32):
            # 在不同设备上生成指定形状的随机张量
            cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
            # 将CPU张量转换为MPS格式的张量，并分离其计算图后复制
            mps_x = cpu_x.detach().clone().to('mps')
            # 对MPS张量的一个切片应用指定的操作，测试原地操作和存储偏移
            y = op(mps_x[1])
            # 对CPU张量的一个切片应用指定的操作，作为对比
            cpu_y = op(cpu_x[1])
            # 断言MPS和CPU结果相等
            self.assertEqual(y, cpu_y)

            # 查看GitHub上的已知问题说明链接
            if not inplace:
                # 在CPU上生成指定形状的随机张量
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
                # 将CPU张量转换为MPS格式的张量，并分离其计算图后复制
                mps_x = cpu_x.detach().clone().to('mps')
                # 在CPU上生成一个空张量，并转置
                cpu_y = torch.empty(shape, device='cpu', dtype=dtype).t()
                # 将指定操作应用于CPU张量，输出到预先分配的CPU张量
                op(cpu_x, out=cpu_y)
                # 将指定操作应用于MPS张量，输出到预先分配的MPS张量
                op(mps_x, out=mps_y)
                # 断言MPS和CPU输出结果相等
                self.assertEqual(mps_y, cpu_y)

        # 调用helper函数，测试torch.exp操作
        helper((5, 5), torch.exp, False)
        # 调用helper函数，测试torch.cos操作
        helper((5, 5), torch.cos, False)
        # 调用helper函数，测试torch.neg操作
        helper((5, 5), torch.neg, False)
        # 调用helper函数，测试torch.tanh操作
        helper((5, 5), torch.tanh, False)
        # 调用helper函数，测试torch.tanh_操作
        helper((5, 5), torch.tanh_, True)

    def test_atan2(self):
        def helper(shape):
            # 生成指定形状的随机张量，并分离其计算图后复制为MPS格式
            input_cpu = torch.randn(shape)
            input_mps = input_cpu.detach().clone().to("mps")

            # 生成另一个指定形状的随机张量，并分离其计算图后复制为MPS格式
            other_cpu = torch.randn(shape)
            other_mps = other_cpu.detach().clone().to("mps")

            # 计算两个张量之间的反正切值，分别在CPU和MPS格式上
            atan2_cpu = torch.atan2(input_cpu, other_cpu)
            atan2_mps = torch.atan2(input_mps, other_mps)

            # 断言MPS格式的结果与CPU格式的结果相等
            self.assertEqual(atan2_cpu, atan2_mps.to("cpu"))

        # 调用helper函数，测试不同形状的输入
        helper(4)
        helper(10000)
        helper((10000, 40))

    def test_multinomial(self):
        # 测试当num_dist = 1时的多项分布采样
        def helper(probs, compare_mean, compare_var, num_samples=5, replacement=True):
            # 创建CPU张量，包含指定的概率值，并且不要求梯度计算
            cpu_prob_tensor = torch.tensor(probs, device='cpu', dtype=torch.float, requires_grad=False)
            # 将CPU张量转换为MPS格式的张量，并分离其计算图后复制
            prob_tensor = cpu_prob_tensor.detach().clone().to('mps')

            # 在概率张量上进行多项分布采样，输出MPS格式的结果
            mps_out = torch.multinomial(prob_tensor, num_samples, replacement=replacement)
            if (not replacement):
                # 如果不进行替换采样，打印MPS格式的输出
                print(mps_out.to('cpu'))
            else:
                # 如果进行替换采样，比较实际采样值与理论期望值的平均值
                print(mps_out.to('cpu').float().mean(), compare_mean)
                # 比较实际采样值与理论方差值的平方
                print(mps_out.to('cpu').float().std() ** 2, compare_var)

        # TODO: 为数据类型添加测试
        # 调用helper函数，测试多项分布采样，期望结果与理论值比较
        helper(np.array([[0., 0., 0., 0.5, 0.5]]), (3 + 4) / 2, (12.5 - 3.5 ** 2), 100000)
        helper(np.array([[.2, .2, .2, .2, .2]]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([[1, 1, 1, 1, 1]]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([1, 1, 1, 1, 1]), (0 + 1 + 2 + 3 + 4) / 5, (6 - 2 * 2), 10000)
        helper(np.array([[1, 1, 1, 1, 1, 1, 1]]), 0, 0, 7, False)
    # 定义测试方法，用于验证 torch.Tensor 的累积和操作在不同维度上的行为是否符合预期
    def test_cumsum_dim_check(self):
        # 创建一个 3x3 的随机张量 x，指定设备为 "mps"
        x = torch.rand((3, 3), device="mps")
        # 断言在维度 1 上的累积和与维度 -1 上的累积和结果相等
        self.assertEqual(x.cumsum(1), x.cumsum(-1))
        # 断言在维度 0 上的累积和与维度 -2 上的累积和结果相等
        self.assertEqual(x.cumsum(0), x.cumsum(-2))
        # 断言在超出张量维度的情况下会引发 IndexError 异常，维度为 2
        self.assertRaises(IndexError, lambda: x.cumsum(2))
        # 断言在超出张量维度的情况下会引发 IndexError 异常，维度为 -3
        self.assertRaises(IndexError, lambda: x.cumsum(-3))

    # 定义测试方法，用于验证 torch.Tensor 的累积乘积操作在不同维度上的行为是否符合预期
    def test_cumprod_dim_check(self):
        # 创建一个 3x3 的随机张量 x，指定设备为 "mps"
        x = torch.rand((3, 3), device="mps")
        # 断言在维度 1 上的累积乘积与维度 -1 上的累积乘积结果相等
        self.assertEqual(x.cumprod(1), x.cumprod(-1))
        # 断言在维度 0 上的累积乘积与维度 -2 上的累积乘积结果相等
        self.assertEqual(x.cumprod(0), x.cumprod(-2))
        # 断言在超出张量维度的情况下会引发 IndexError 异常，维度为 2
        self.assertRaises(IndexError, lambda: x.cumprod(2))
        # 断言在超出张量维度的情况下会引发 IndexError 异常，维度为 -3
        self.assertRaises(IndexError, lambda: x.cumprod(-3))
# 定义一个名为 TestLogical 的测试类，继承自 TestCaseMPS
class TestLogical(TestCaseMPS):

    # 辅助函数，将给定的张量 x 包装成 torch.tensor 对象，并返回
    def _wrap_tensor(self, x, device="cpu", dtype=None, requires_grad=False):
        return torch.tensor(x, device=device, dtype=dtype, requires_grad=requires_grad)

    # 测试逻辑非操作的函数
    def test_logical_not(self):
        # 辅助函数 helper，接受一个参数 x
        def helper(x):
            # 将参数 x 赋值给 cpu_x
            cpu_x = x
            # 对 cpu_x 进行 detach() 和 clone() 操作，然后将其转移到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 使用 torch.logical_not 计算逻辑非结果
            result = torch.logical_not(x)
            # 对原始输入 cpu_x 应用 torch.logical_not 计算逻辑非结果
            result_cpu = torch.logical_not(cpu_x)

            # 断言两者结果相等
            self.assertEqual(result, result_cpu)

        # 使用辅助函数 helper 对不同类型的输入进行测试
        helper(self._wrap_tensor([1, 1, 0, 0]))  # 测试整数张量
        helper(self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True))  # 测试浮点张量
        helper(self._wrap_tensor([True, True, False, False]))  # 测试布尔张量
        helper(self._wrap_tensor(1))  # 测试标量整数
        helper(self._wrap_tensor(0))  # 测试标量整数
        helper(self._wrap_tensor(True))  # 测试标量布尔值
        helper(self._wrap_tensor(False))  # 测试标量布尔值

    # 测试逻辑与操作的函数
    def test_logical_and(self):
        # 辅助函数 helper，接受两个参数 x 和 other
        def helper(x, other):
            # 将参数 x 赋值给 cpu_x
            cpu_x = x
            # 对 cpu_x 进行 detach() 和 clone() 操作，然后将其转移到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 将参数 other 赋值给 cpu_other
            cpu_other = other
            # 对 cpu_other 进行 detach() 和 clone() 操作，然后将其转移到 'mps' 设备上
            other = cpu_other.detach().clone().to('mps')

            # 使用 torch.logical_and 计算逻辑与结果
            result = torch.logical_and(x, other)
            # 对原始输入 cpu_x 和 cpu_other 应用 torch.logical_and 计算逻辑与结果
            result_cpu = torch.logical_and(cpu_x, cpu_other)

            # 断言两者结果相等
            self.assertEqual(result, result_cpu)

        # 使用辅助函数 helper 对不同类型的输入进行测试
        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor([1, 0, 0, 1]))  # 测试整数张量
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )  # 测试浮点张量
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))  # 测试布尔张量
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))  # 测试张量和标量整数
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))  # 测试张量和标量整数
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))  # 测试张量和标量布尔值
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))  # 测试张量和标量布尔值

    # 测试逻辑或操作的函数
    def test_logical_or(self):
        # 辅助函数 helper，接受两个参数 x 和 other
        def helper(x, other):
            # 将参数 x 赋值给 cpu_x
            cpu_x = x
            # 对 cpu_x 进行 detach() 和 clone() 操作，然后将其转移到 'mps' 设备上
            x = cpu_x.detach().clone().to('mps')

            # 将参数 other 赋值给 cpu_other
            cpu_other = other
            # 对 cpu_other 进行 detach() 和 clone() 操作，然后将其转移到 'mps' 设备上
            other = cpu_other.detach().clone().to('mps')

            # 使用 torch.logical_or 计算逻辑或结果
            result = torch.logical_or(x, other)
            # 对原始输入 cpu_x 和 cpu_other 应用 torch.logical_or 计算逻辑或结果
            result_cpu = torch.logical_or(cpu_x, cpu_other)

            # 断言两者结果相等
            self.assertEqual(result, result_cpu)

        # 使用辅助函数 helper 对不同类型的输入进行测试
        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor([1, 0, 0, 1]))  # 测试整数张量
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )  # 测试浮点张量
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))  # 测试布尔张量
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))  # 测试张量和标量整数
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))  # 测试张量和标量整数
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))  # 测试张量和标量布尔值
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))  # 测试张量和标量布尔值
    def test_logical_xor(self):
        # 定义辅助函数 helper，用于测试逻辑异或操作
        def helper(x, other):
            # 将输入张量 x 的 CPU 版本保存
            cpu_x = x
            # 将 x 转换为 MPS（假设是某种特定硬件/环境的代码） 上的张量，并克隆
            x = cpu_x.detach().clone().to('mps')

            # 将输入张量 other 的 CPU 版本保存
            cpu_other = other
            # 将 other 转换为 MPS 上的张量，并克隆
            other = cpu_other.detach().clone().to('mps')

            # 对 x 和 other 执行逻辑异或操作
            result = torch.logical_xor(x, other)
            # 在 CPU 上执行相同的逻辑异或操作
            result_cpu = torch.logical_xor(cpu_x, cpu_other)

            # 断言 MPS 上的结果与 CPU 上的结果相等
            self.assertEqual(result, result_cpu)

        # 测试不同的输入情况
        helper(self._wrap_tensor([1, 1, 0, 0]), self._wrap_tensor([1, 0, 0, 1]))
        helper(
            self._wrap_tensor([1, 1, 0, 0], dtype=torch.float, requires_grad=True),
            self._wrap_tensor([1, 0, 0, 1], dtype=torch.float)
        )
        helper(self._wrap_tensor([True, True, False, False]), self._wrap_tensor([True, False, False, True]))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(1))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(0))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(True))
        helper(self._wrap_tensor((1, 0, 1, 0)), self._wrap_tensor(False))

    def test_min_max(self):
        # 定义辅助函数 helper，用于测试张量的最大最小值
        def helper(dtype):
            # 针对不同的数据类型 dtype 进行测试
            for _ in range(10):
                if dtype == torch.float32 or dtype == torch.float16:
                    # 如果是浮点型数据类型，生成符合要求的随机张量 x（MPS 环境）
                    x = torch.randn((30, 15), device='mps', dtype=dtype)
                else:
                    # 否则，生成符合要求的整数型随机张量 x（MPS 环境）
                    x = torch.randint(0, 100, (30, 15), device="mps", dtype=dtype)
                # 将 x 转换为 CPU 上的张量
                x_cpu = x.to("cpu")

                # 计算 MPS 上的张量 x 的最大值 y
                y = x.max()
                # 计算 CPU 上的张量 x 的最大值 y_cpu
                y_cpu = x_cpu.max()
                # 断言两者的最大值相等
                self.assertEqual(y, y_cpu)

                # 计算 MPS 上的张量 x 的最小值 z
                z = x.min()
                # 计算 CPU 上的张量 x 的最小值 z_cpu
                z_cpu = x_cpu.min()
                # 断言两者的最小值相等
                self.assertEqual(z, z_cpu)

        # 对多种数据类型进行 helper 函数的测试
        [helper(dtype) for dtype in [torch.float32, torch.float16, torch.int32, torch.int16, torch.uint8, torch.int8, torch.bool]]
    def test_isin(self):
        # 定义辅助函数helper，用于测试不同数据类型的输入
        def helper(dtype):
            # 不同形状的输入组合
            shapes = [([2, 5], [3, 5, 2]), ([10, 3, 5], [20, 1, 3]),
                      ([5], [10]), ([0], [5]), ([5], [0])]
            # 遍历所有形状组合
            for shape_tuple in shapes:
                # 遍历反转标志的两种可能性
                for inverted in [True, False]:
                    # 根据数据类型是否为浮点型选择不同的初始化方式
                    if dtype.is_floating_point:
                        # 如果是浮点数类型，则在 CPU 上生成符合形状的随机张量（使用FP32）
                        A = torch.randn(size=shape_tuple[0], device='cpu', dtype=torch.float32)
                        B = torch.randn(size=shape_tuple[1], device='cpu', dtype=torch.float32)
                    else:
                        # 如果不是浮点数类型，则在 CPU 上生成符合形状的随机整数张量
                        A = torch.randint(0, 100, size=shape_tuple[0], device='cpu', dtype=dtype)
                        B = torch.randint(0, 100, size=shape_tuple[1], device='cpu', dtype=dtype)

                    # 对生成的张量进行 MPS 转换并创建其副本
                    A_mps = A.clone().detach().to('mps')
                    B_mps = B.clone().detach().to('mps')

                    # 在 CPU 上计算参考结果
                    cpu_ref = torch.isin(A, B, invert=inverted)
                    # 如果数据类型为浮点16或bfloat16，则将参考结果转换为对应数据类型
                    if dtype in [torch.float16, torch.bfloat16]:
                        cpu_ref.type(dtype)

                    # 在 MPS 上计算 torch.isin 的输出
                    mps_out = torch.isin(A_mps, B_mps, invert=inverted)
                    # 使用断言验证 MPS 输出与 CPU 参考结果一致
                    self.assertEqual(mps_out, cpu_ref)

        # 定义被测试的数据类型列表
        dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int16, torch.uint8, torch.int8]
        # 如果产品版本小于14.0，在 macOS 上整数类型预计会失败
        if product_version < 14.0:
            dtypes = [torch.float32, torch.float16, torch.bfloat16]

        # 对所有数据类型进行测试
        [helper(dtype) for dtype in dtypes]

    def test_isin_asserts(self):
        # 使用 MPS 设备生成随机张量 A 和 B（A 是 float32，B 是 float16）
        A = torch.randn(size=[1, 4], device='mps', dtype=torch.float32)
        B = torch.randn(size=[1, 4], device='mps', dtype=torch.float16)
        # 使用断言检查是否引发了 RuntimeError，错误消息包含 "Expected elements.dtype()"*
        with self.assertRaisesRegex(RuntimeError, 'Expected elements.dtype()*'):
            out = torch.isin(A, B)

        # 使用 MPS 设备生成随机张量 C 和 D（C 是 float32，D 是在 CPU 上的 float32）
        C = torch.randn(size=[1, 4], device='mps', dtype=torch.float32)
        D = torch.randn(size=[1, 4], device='cpu', dtype=torch.float32)
        # 使用断言检查是否引发了 RuntimeError，错误消息包含 "Expected elements.is_mps()"*
        with self.assertRaisesRegex(RuntimeError, 'Expected elements.is_mps()*'):
            out = torch.isin(C, D)
# 定义一个继承自 TestCaseMPS 的测试类 TestSmoothL1Loss
class TestSmoothL1Loss(TestCaseMPS):

    # 辅助函数，用于测试 smooth_l1_loss 函数的不同参数组合
    def _smooth_l1_loss_helper(self, reduction="mean", requires_grad=False):
        # 创建一个形状为 (4, 7) 的随机张量 input_cpu，并设置是否需要梯度
        input_cpu = torch.randn(4, 7, requires_grad=requires_grad)
        # 创建一个形状为 (4, 7) 的随机张量 target_cpu
        target_cpu = torch.randn(4, 7)

        # 将 input_cpu 分离并克隆到 'mps' 设备，同时要求梯度
        input_mps = input_cpu.detach().clone().to('mps').requires_grad_()
        # 将 target_cpu 克隆到 'mps' 设备
        target_mps = target_cpu.detach().clone().to('mps')

        # 使用 F.smooth_l1_loss 计算 CPU 上的 smooth L1 损失
        smooth_l1_loss_cpu = F.smooth_l1_loss(input_cpu, target_cpu, beta=1.0, reduction=reduction)
        # 使用 F.smooth_l1_loss 计算 'mps' 设备上的 smooth L1 损失
        smooth_l1_loss_mps = F.smooth_l1_loss(input_mps, target_mps, beta=1.0, reduction=reduction)

        # 断言两种设备上的损失值相等
        self.assertEqual(smooth_l1_loss_cpu, smooth_l1_loss_mps)

        # 如果 requires_grad 为 True，则进行反向传播
        if requires_grad:
            smooth_l1_loss_cpu.backward()
            smooth_l1_loss_mps.backward()
            # 断言两种设备上的梯度相等
            self.assertEqual(input_cpu.grad, input_mps.grad.to("cpu"))

        # 返回 CPU 上和 'mps' 设备上的 smooth L1 损失值
        return smooth_l1_loss_cpu, smooth_l1_loss_mps

    # 测试不同 reduction 参数设置下的 smooth L1 损失函数（reduction=None）
    def test_smooth_l1_loss_reduction_none(self):
        self._smooth_l1_loss_helper(reduction="none")

    # 测试不同 reduction 参数设置下的 smooth L1 损失函数（reduction=mean）
    def test_smooth_l1_loss_reduction_mean(self):
        self._smooth_l1_loss_helper(reduction="mean")

    # 测试不同 reduction 参数设置下的 smooth L1 损失函数（reduction=sum）
    def test_smooth_l1_loss_reduction_sum(self):
        self._smooth_l1_loss_helper(reduction="sum")

    # 测试带梯度反向传播的情况下的 smooth L1 损失函数（reduction=mean）
    def test_smooth_l1_loss_reduction_mean_backward(self):
        self._smooth_l1_loss_helper(reduction="mean", requires_grad=True)

    # 测试带梯度反向传播的情况下的 smooth L1 损失函数（reduction=sum）
    def test_smooth_l1_loss_reduction_mean_sum_backward(self):
        self._smooth_l1_loss_helper(reduction="sum", requires_grad=True)

# 定义一个继承自 TestCaseMPS 的测试类 TestNLLLoss
class TestNLLLoss(TestCaseMPS):

    # 测试当 batch 大小不匹配时是否抛出 ValueError 异常
    def test_nll_loss_mismatched_batch(self, device='mps'):
        # 创建一个形状为 (10, 3) 的随机张量 x，并要求梯度，使用指定设备
        x = torch.randn((10, 3), requires_grad=True, device=device)
        # 创建一个形状为 (3,) 的零张量 t，并指定为 int64 类型，使用指定设备
        t = torch.zeros((3,), dtype=torch.int64, device=device)
        # 断言调用 F.nll_loss 函数时是否抛出预期异常信息
        with self.assertRaisesRegex(ValueError, 'Expected.*batch_size'):
            F.nll_loss(x, t)

    # 测试带有 ignore_index 参数设置时的 NLL 损失函数
    def test_nll_loss_out_of_bounds_ignore_index(self):

        # 定义一个辅助函数，用于测试带有 ignore_index 参数设置时的 NLL 损失函数
        def test_nll_loss_out_of_bounds_ignore_index_helper(device):
            output = []
            # 创建一个形状为 (6, 3) 的张量 x，并指定设备
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1],
                              [0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            # 创建两个形状为 (6,) 的张量 t1 和 t2，并指定 int64 类型和设备
            t1 = torch.tensor([0, 1, 255, 0, 1, 2], dtype=torch.int64, device=device)
            t2 = torch.tensor([0, 1, 1, 0, -100, 2], dtype=torch.int64, device=device)
            for reduction in ['mean', 'none']:
                # 测试 ignore_index=255 的情况下的 NLL 损失函数
                output.append(F.nll_loss(x, t1, ignore_index=255, reduction=reduction))
                # 测试默认 ignore_index=-100 的情况下的 NLL 损失函数
                output.append(F.nll_loss(x, t2, reduction=reduction))
            return output

        # 在 CPU 和 'mps' 设备上分别测试 NLL 损失函数的 ignore_index 参数设置
        output_cpu = test_nll_loss_out_of_bounds_ignore_index_helper(device='cpu')
        output_mps = test_nll_loss_out_of_bounds_ignore_index_helper(device='mps')

        # 断言两种设备上的输出结果相等
        for cpu, mps in zip(output_cpu, output_mps):
            self.assertEqual(cpu, mps)
    def test_nll_loss_invalid_target_dim(self):
        # 定义一个测试函数，用于测试目标张量维度不合法的情况
        def _test_nll_loss_invalid_target_dim(device):
            # 创建一个输出列表
            output = []
            # 创建一个张量 x，包含了预测概率分布的数据
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            # 创建一个目标张量 t，其维度是不合法的
            t = torch.zeros((6, 2), dtype=torch.int64, device=device)
            # 使用断言检查是否会抛出 RuntimeError，并且错误信息中包含 "1D target tensor expected"
            with self.assertRaisesRegex(RuntimeError, "1D target tensor expected"):
                F.nll_loss(x, t)

        # 在 CPU 和 MPS 设备上分别执行测试
        _test_nll_loss_invalid_target_dim(device='cpu')
        _test_nll_loss_invalid_target_dim(device='mps')

    def test_nll_loss_invalid_weights(self):
        # 定义一个测试函数，用于测试权重张量不合法的情况
        def _test_nll_loss_invalid_weights(device):
            # 创建一个张量 x，包含了预测概率分布的数据
            x = torch.tensor([[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1], [
                             0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]], device=device)
            # 创建一个目标张量 t，用于指定每个样本的真实类别
            t = torch.tensor([0, 1, 2, 1, 1, 2], dtype=torch.int64, device=device)
            # 创建一组不合法的权重张量列表
            invalid_weights = [
                torch.zeros(4, device=device),
                torch.zeros((1, 3), device=device),
            ]
            # 期望的错误信息
            msg = "weight tensor should be defined either for all 3 classes or no classes"
            # 对每一个不合法的权重张量进行测试
            for weight in invalid_weights:
                # 使用断言检查是否会抛出 RuntimeError，并且错误信息中包含指定的 msg
                with self.assertRaisesRegex(RuntimeError, msg):
                    F.nll_loss(x, t, weight=weight)

        # 在 CPU 和 MPS 设备上分别执行测试
        _test_nll_loss_invalid_weights(device='cpu')
        _test_nll_loss_invalid_weights(device='mps')

    def _nll_loss_helper(self, input_size, reduction, expected):
        # 辅助函数，用于帮助测试 nll_loss 函数的行为

        # 在 CPU 设备上创建输入张量 input，具有指定的大小和梯度要求
        input = torch.rand(input_size, requires_grad=True, device='cpu')
        # 获取输入张量的通道数
        num_channels = input_size[1]
        # 构建目标张量的大小，保持与输入张量 input 相同的批量维度
        target_size = (input_size[0], ) + tuple(input_size[2:])
        # 在 CPU 设备上创建目标张量 target，其元素为随机整数，用于指定真实的类别
        target = torch.randint(num_channels, target_size, device='cpu')
        # 创建权重张量 weights，用于加权损失计算
        weights = torch.randn(num_channels)

        # 在 MPS 设备上创建 input_mps，作为 input 的副本，并设置为需要梯度计算
        input_mps = input.detach().clone().to('mps').requires_grad_()
        # 在 MPS 设备上创建 target_mps，作为 target 的副本
        target_mps = target.detach().clone().to('mps')
        # 在 MPS 设备上创建 weights_mps，作为 weights 的副本
        weights_mps = weights.to("mps")

        # 在 CPU 和 MPS 设备上分别计算 nll_loss
        output_cpu = F.nll_loss(input, target, weight=weights, reduction=reduction)
        output_mps = F.nll_loss(input_mps, target_mps, weight=weights_mps, reduction=reduction)
        # 使用断言检查 CPU 和 MPS 设备上的输出结果是否一致
        self.assertEqual(output_cpu, output_mps.to('cpu'))

        # 对 CPU 和 MPS 设备上的梯度进行反向传播
        output_cpu.sum().backward()
        output_mps.sum().backward()
        # 使用断言检查 CPU 设备上的梯度是否与 MPS 设备上的梯度一致
        self.assertEqual(input.grad, input_mps.grad.to('cpu'))
    # 定义一个辅助方法用于计算一维负对数似然损失函数
    def _nll_loss_1d_helper(self, input_size, reduction):

        # 在 CPU 上生成指定大小的随机张量，需要梯度计算，设备为 CPU
        input = torch.rand(input_size, requires_grad=True, device='cpu')
        # 获取输入张量的通道数
        num_channels = input_size[0]
        # 在 CPU 上生成一个随机整数作为目标值，范围为 [0, num_channels)
        target = torch.randint(num_channels, [], device='cpu')

        # 将 CPU 上的 input 张量拷贝到 MPS（Memory Persistence Storage），并且需要梯度计算
        input_mps = input.detach().clone().to('mps').requires_grad_()
        # 将目标值 target 在 MPS 上的拷贝
        target_mps = target.detach().clone().to('mps')

        # 计算 CPU 上的负对数似然损失，指定损失的减少方式（reduction）
        output_cpu = F.nll_loss(input, target, reduction=reduction)
        # 计算 MPS 上的负对数似然损失，指定损失的减少方式（reduction）
        output_mps = F.nll_loss(input_mps, target_mps, reduction=reduction)
        # 断言 CPU 上的输出与 MPS 上转回 CPU 的输出相等
        self.assertEqual(output_cpu, output_mps.to('cpu'))

        # 对 CPU 上的输出进行求和后反向传播
        output_cpu.sum().backward()
        # 对 MPS 上的输出进行求和后反向传播，然后将梯度转回 CPU
        output_mps.sum().backward()
        # 断言输入张量在 CPU 上的梯度与 MPS 转回 CPU 后的梯度相等
        self.assertEqual(input.grad, input_mps.grad.to('cpu'))

    # 测试一维负对数似然损失函数的不同减少方式
    def test_nll_loss_1d(self, device='cpu'):
        self._nll_loss_1d_helper([10], "none")
        self._nll_loss_1d_helper([10], "mean")
        self._nll_loss_1d_helper([10], "sum")

    # 测试当目标张量为空且减少方式为 "none" 时的一维负对数似然损失
    def test_nll_loss_empty_tensor_reduction_none(self, device='cpu'):
        self._nll_loss_helper([1, 3], "none", torch.empty([0], device=device))
        self._nll_loss_helper([3, 5, 7], "none", torch.empty([5, 7], device=device))
        self._nll_loss_helper([2, 3, 1, 7], "none", torch.empty([2, 1, 7], device=device))
        self._nll_loss_helper([2, 3, 5, 1], "none", torch.empty([2, 5, 1], device=device))
        self._nll_loss_helper([2, 3, 5, 7, 1], "none", torch.empty([2, 5, 7, 1], device=device))

    # 测试当目标张量为空且减少方式为 "mean" 时的一维负对数似然损失
    def test_nll_loss_empty_tensor_reduction_mean(self, device='cpu'):
        nan = torch.tensor(float('nan'), device=device)
        self._nll_loss_helper([1, 3], "mean", nan)
        self._nll_loss_helper([1, 3, 5, 7], "mean", nan)
        self._nll_loss_helper([2, 3, 1, 7], "mean", nan)
        self._nll_loss_helper([2, 3, 5, 1], "mean", nan)
        self._nll_loss_helper([2, 3, 5, 7, 1], "mean", nan)

    # 测试当目标张量为空且减少方式为 "sum" 时的一维负对数似然损失
    def test_nll_loss_empty_tensor_reduction_sum(self, device='cpu'):
        zero = torch.tensor(0, device=device)
        self._nll_loss_helper([1, 3], "sum", zero)
        self._nll_loss_helper([1, 3, 5, 7], "sum", zero)
        self._nll_loss_helper([2, 3, 1, 7], "sum", zero)
        self._nll_loss_helper([2, 3, 5, 1], "sum", zero)
        self._nll_loss_helper([2, 3, 5, 7, 1], "sum", zero)
    # 定义一个测试函数，测试 NLLLoss 对象在字节和长整型目标类型匹配时的行为
    def test_nll_loss_byte_target_matches_long(self, device='cpu'):
        # 定义数据维度和类别数
        N, C = 10, 4
        # 生成随机输入张量，在指定设备上创建并标记为需要梯度
        input = torch.randn(N, C, device=device, requires_grad=True)
        # 生成空的目标张量，数据类型为长整型，在指定设备上随机填充从0到C的整数
        target = torch.empty(N, dtype=torch.long, device=device).random_(0, C)

        # 定义一个内部函数，用于计算结果和梯度
        def compute_result_and_gradient(reduction, target_dtype):
            # 初始化结果字典和梯度字典
            result, grad = {}, {}
            # 遍历设备列表 ['cpu', 'mps']
            for dev in ['cpu', 'mps']:
                # 将输入张量复制到当前设备上
                input_dev = input.to(dev)
                # 分离输入张量，标记为需要梯度
                input_ = input_dev.detach()
                input_.requires_grad_()

                # 将目标张量复制到当前设备上
                target_dev = target.to(dev)

                # 计算输入张量在最后一个维度上的 log softmax
                prob = F.log_softmax(input_, dim=-1)
                # 创建一个 NLLLoss 对象，指定减少方式为 reduction
                loss = nn.NLLLoss(reduction=reduction)
                # 计算损失值并存储到结果字典中
                result[dev] = loss(prob, target_dev.to(target_dtype))
                # 对损失值进行求和并反向传播
                result[dev].sum().backward()
                # 存储输入张量的梯度到梯度字典中
                grad[dev] = input_.grad

            return result, grad

        # 遍历减少方式列表 ["none", "mean", "sum"]
        for reduction in ["none", "mean", "sum"]:
            # 使用长整型目标类型调用 compute_result_and_gradient 函数，获取结果和梯度
            result_long, grad_long = compute_result_and_gradient(reduction, torch.long)
            # 使用字节目标类型调用 compute_result_and_gradient 函数，获取结果和梯度
            result_byte, grad_byte = compute_result_and_gradient(reduction, torch.uint8)

            # 使用断言验证 'mps' 设备上的结果与 'cpu' 设备上的结果是否相等
            self.assertEqual(result_long['mps'].to('cpu'), result_long['cpu'])
            # 使用断言验证 'mps' 设备上的梯度与 'cpu' 设备上的梯度是否相等
            self.assertEqual(grad_long['mps'].to('cpu'), grad_long['cpu'])
class TestTopK(TestCase):
    # 测试 Top-K 操作的单元测试类
    def _test_topk(self, shape, largest):
        # 测试 Top-K 操作的辅助函数
        cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
        # 在 CPU 上生成随机张量
        x = cpu_x.detach().clone().to('mps')
        # 将其克隆并移到 'mps' 设备上
        if isinstance(shape, tuple):
            # 如果形状是元组
            for curr_dim, dim_size in enumerate(shape):
                # 遍历每个维度及其大小
                for k in range(1, dim_size + 1):
                    # 对于每个维度，计算最大（或最小）的 k 个值及其索引
                    topk_values, topk_indices = torch.topk(x, k, dim=curr_dim, largest=largest)
                    topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=curr_dim, largest=largest)
                    # 断言 MPS 和 CPU 计算的结果是否一致
                    self.assertEqual(topk_values, topk_values_cpu)
                    self.assertEqual(topk_indices, topk_indices_cpu)
        else:
            # 如果形状不是元组
            for k in range(1, shape):
                # 计算最大（或最小）的 k 个值及其索引
                topk_values, topk_indices = torch.topk(x, k, dim=0, largest=largest)
                topk_values_cpu, topk_indices_cpu = torch.topk(cpu_x, k, dim=0, largest=largest)
                # 断言 MPS 和 CPU 计算的结果是否一致
                self.assertEqual(topk_values, topk_values_cpu)
                self.assertEqual(topk_indices, topk_indices_cpu)

    def test_topk(self):
        # 测试不同形状和参数 largest 的 Top-K 操作
        largest_vals = [True, False]
        shapes = [
            # 零元素张量
            0,
            (1, 0),
            (0, 1),
            (1, 0, 1),
            # 多元素张量
            1,
            2,
            (5, 1),
            (1, 5),
            (5, 9, 7, 4),
        ]

        for shape in shapes:
            for largest_val in largest_vals:
                with self.subTest(shape=shape, largest_val=largest_val):
                    self._test_topk(shape, largest_val)

class TestNNMPS(NNTestCase):
    # 测试 MPS 网络的单元测试类

    def _create_basic_net(self):
        # 创建基本网络结构的辅助函数
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = Layer()
                self.dummy_param = Parameter(torch.empty(3, 5))
                self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

        l = Layer()
        n = Net()
        s = nn.Sequential(n, n)

        return l, n, s

    def test_requires_grad_(self):
        # 测试网络参数和缓冲区的 requires_grad_ 方法
        m = self._create_basic_net()[-1]
        assert len(list(m.buffers())) > 0, 'invalid test'
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        assert len(list(m.parameters())) > 0, 'invalid test'
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        for requires_grad in (False, True):
            self.assertIs(m.requires_grad_(requires_grad), m)
            for p in m.parameters():
                # 断言参数的 requires_grad 是否符合预期
                self.assertEqual(p.requires_grad, requires_grad)
            for b in m.buffers():
                # 断言缓冲区的 requires_grad 是否为 False
                self.assertFalse(b.requires_grad)
    # 测试模块的向后兼容性
    def test_module_backcompat(self):
        # 导入警告和源变更警告类
        from torch.serialization import SourceChangeWarning
        # 下载文件并获取路径
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        # 忽略源变更警告
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            # 加载模型
            m = torch.load(path)
        # 创建随机输入张量
        input = torch.randn(2, 3, dtype=torch.float)
        # 断言模型输出的尺寸是否符合预期
        self.assertEqual(m(input).size(), (2, 5))

    # 测试卷积操作的向后兼容性
    def test_conv_backcompat(self):
        # 导入警告和源变更警告类
        from torch.serialization import SourceChangeWarning
        # 设置文件路径，此文件是在 Python 2 和 PyTorch 1.0.1 下生成的
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        # 忽略源变更警告
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            # 加载模型，使用 utf-8 编码
            m = torch.load(path, encoding='utf-8')
        # 创建随机输入张量
        input = torch.randn((1, 1, 1, 1), dtype=torch.float)
        # 断言模型输出的尺寸是否符合预期
        self.assertEqual(m(input).size(), (1, 1, 1, 1))

    # 测试卷积扩展操作
    def test_conv_expand(self):
        # 设备选择为 'mps'
        device = 'mps'
        # 创建随机输入张量和卷积核张量
        input_ = torch.rand(2, 3, 16, 16, device=device)
        kernel = torch.rand(1, 1, 3, 11, device=device)
        # 对卷积核张量进行扩展
        tmp_kernel = kernel.expand(-1, 3, -1, -1)
        # 执行卷积操作
        output = F.conv2d(input_, tmp_kernel, groups=1, padding=0, stride=1)

    # 测试 permute 操作
    # 测试不应崩溃
    def test_permute(self):
        # 创建具有随机值的 CPU 张量
        M_cpu = torch.randn(5, 5)
        # 将 CPU 张量转换为 'mps' 设备张量
        M_mps = M_cpu.to('mps')

        # 对 CPU 张量进行维度置换操作
        output_cpu = M_cpu.permute(1, 0)
        # 对 'mps' 设备张量进行维度置换操作
        output_mps = M_mps.permute(1, 0)

        # 断言两个张量的内容是否相等
        self.assertEqual(output_cpu, output_mps)
        # 断言两个张量的尺寸是否相等
        self.assertEqual(output_cpu.size(), output_mps.size())

    # 打印非连续张量，不应崩溃
    def test_print_non_contiguous(self):
        # 打印 'mps' 设备上全为1的张量的非零元素索引
        print(torch.ones(100, 100, device='mps').nonzero())
        # 打印经连续化后的 'mps' 设备上全为1的张量的非零元素索引
        print(torch.ones(100, 100, device='mps').nonzero().contiguous())
    # 定义一个测试函数，用于测试模型梯度清零和梯度计算
    def test_zero_grad(self):
        # 创建一个随机张量作为输入数据，设置 requires_grad=True 开启梯度追踪
        i = torch.randn(2, 5, requires_grad=True)
        # 创建一个线性层模型，输入和输出都是大小为 5 的向量
        module = nn.Linear(5, 5)
        # 将模型参数的 requires_grad 属性设置为 False，不计算这些参数的梯度
        for p in module.parameters():
            p.requires_grad = False
        # 对模型进行梯度清零操作
        module.zero_grad()

        # 将模型权重的 requires_grad 属性设置为 True
        module.weight.requires_grad = True
        # 再次进行梯度清零操作
        module.zero_grad()
        # 断言模型的权重梯度为 None，即未初始化的梯度
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        # 对输入数据进行前向传播和反向传播
        module(i).sum().backward()
        # 断言模型的权重梯度不为 None
        self.assertIsNotNone(module.weight.grad)
        # 断言模型的权重梯度绝对值之和大于 0
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        # 再次进行梯度清零操作
        module.zero_grad()
        # 断言模型的权重梯度为 None
        self.assertIsNone(module.weight.grad)

        # 将模型偏置的 requires_grad 属性设置为 True
        module.bias.requires_grad = True
        # 再次进行梯度清零操作
        module.zero_grad()
        # 断言模型的权重梯度为 None
        self.assertIsNone(module.weight.grad)
        # 断言模型的偏置梯度为 None
        self.assertIsNone(module.bias.grad)
        # 对输入数据进行前向传播和反向传播
        module(i).sum().backward()
        # 断言模型的权重梯度不为 None
        self.assertIsNotNone(module.weight.grad)
        # 断言模型的偏置梯度不为 None
        self.assertIsNotNone(module.bias.grad)
        # 断言模型的权重梯度绝对值之和大于 0
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        # 断言模型的偏置梯度绝对值之和大于 0
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)

        # 强制将模型梯度设置为零，set_to_none=False 表示使用零张量而不是 None
        module.zero_grad(set_to_none=False)
        # 断言模型的权重梯度数据与模型权重数据的零张量相等
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        # 断言模型的偏置梯度数据与模型偏置数据的零张量相等
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        # 再次进行梯度清零操作
        module.zero_grad()
        # 断言模型的权重梯度为 None
        self.assertIsNone(module.weight.grad)
        # 断言模型的偏置梯度为 None


    # 定义一个测试函数，用于测试不计算梯度的情况
    def test_no_grad(self):
        # 针对不同的数据类型进行测试
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            # 创建一个卷积层模型，输入通道为 2，输出通道为 5，卷积核大小为 3x3
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            # 创建一个随机输入张量，并转换为指定数据类型
            input = torch.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            # 对输入数据进行前向传播
            output = module(x)
            # 断言输出张量需要计算梯度
            self.assertTrue(output.requires_grad)
            # 尝试对输出张量进行反向传播，预期会抛出 RuntimeError
            output.backward(torch.ones(1, 5, 10, 10))

            # 使用 torch.no_grad() 上下文管理器
            with torch.no_grad():
                # 对输入数据进行前向传播，不计算梯度
                output2 = module(y)
                # 断言输出张量不需要计算梯度
                self.assertFalse(output2.requires_grad)
                # 尝试对输出张量进行反向传播，预期会抛出 RuntimeError
                self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))


    # 定义一个测试函数，用于测试无效的 Conv1d 参数
    def test_invalid_conv1d(self):
        # 针对不同的数据类型进行测试
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            # 创建一个 Conv1d 模型，输入通道为 3，输出通道为 33，卷积核大小为 10，步长为 1
            module = nn.Conv1d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True).to(dtype)
            # 创建一个随机输入张量，并转换为指定数据类型
            input = torch.randn(1, 3, 4).to(dtype)
            # 使用断言捕获 RuntimeError，并检查错误消息
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(4\). ' +
                                        r'Kernel size: \(10\). Kernel size can\'t be greater than actual input size'):
                # 调用 Conv1d 模型，预期会抛出 RuntimeError
                module(input)

            # 测试负步长的情况
            # 创建一个 Conv1d 模型，输入通道为 3，输出通道为 6，卷积核大小为 3，步长为 -1
            module = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, stride=-1, bias=True).to(dtype)
            # 创建一个随机输入张量，并转换为指定数据类型
            input = torch.randn(1, 3, 4).to(dtype)
            # 使用断言捕获 RuntimeError，并检查错误消息
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                # 调用 Conv1d 模型，预期会抛出 RuntimeError
                module(input)
    def test_invalid_conv2d(self):
        # 循环测试不同数据类型的卷积操作是否会引发异常
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            # 创建指定数据类型的卷积层模块，设置相关参数
            module = torch.nn.Conv2d(1, 1, kernel_size=3, dilation=2, stride=2).to(dtype)
            # 创建指定数据类型的空输入张量
            input = torch.empty(1, 1, 4, 4).to(dtype)
            # 断言在执行模块(input)时会引发 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: module(input))

            # 创建具有特定设置的卷积层模块和输入数据
            module = nn.Conv2d(in_channels=3, out_channels=33, kernel_size=10, stride=1, bias=True)
            input = torch.randn(1, 3, 1, 1)
            # 使用断言检查是否引发了预期的 RuntimeError 异常，匹配特定的错误信息正则表达式
            with self.assertRaisesRegex(RuntimeError,
                                        r'Calculated padded input size per channel: \(1 x 1\). ' +
                                        r'Kernel size: \(10 x 10\). Kernel size can\'t be greater than actual input size'):
                module(input)

            # 创建具有负步长设置的卷积层模块和输入数据
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=-1, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            # 使用断言检查是否引发了预期的 RuntimeError 异常，匹配特定的错误信息字符串
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # 创建具有零步长设置的卷积层模块和输入数据
            module = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=0, bias=True).to(dtype)
            input = torch.randn(1, 3, 4, 4).to(dtype)
            # 使用断言检查是否引发了预期的 RuntimeError 异常，匹配特定的错误信息字符串
            with self.assertRaisesRegex(RuntimeError, 'non-positive stride is not supported'):
                module(input)

            # 使用不同设备的输入和权重，断言是否引发 RuntimeError 异常，匹配特定的错误信息字符串
            self.assertRaisesRegex(RuntimeError,
                                   'must be on the same device',
                                   lambda: torch.conv2d(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 3, 3, device='mps')))
            self.assertRaisesRegex(RuntimeError,
                                   'Input type \\(MPSFloatType\\) and weight type \\(torch\\.FloatTensor\\) should be the same',
                                   lambda: torch.conv2d(torch.rand(1, 3, 32, 32, device='mps'), torch.rand(1, 3, 3, 3)))
    def test_conv2d_valid_padding(self, device='mps'):
        # 测试 F.conv2d 使用 padding='valid' 时与不使用 padding 相同
        x = torch.rand(1, 1, 1, 10, device=device).to(torch.float)
        y = torch.rand(1, 1, 1, 4, device=device).to(torch.float)

        expect = F.conv2d(x, y)
        actual = F.conv2d(x, y, padding='valid')
        self.assertEqual(expect.to('cpu'), actual.to('cpu'))

    def test_conv2d_backward_collision(self):
        # 测试 GitHub 问题 https://github.com/pytorch/pytorch/issues/112998
        x = torch.rand(1, 1, 10, 10, device="mps", requires_grad=True)
        m1 = nn.Conv2d(1, 1, 3, stride=2, padding=1).to("mps")
        m2 = nn.Conv2d(1, 1, 4, stride=2, padding=1).to("mps")
        y1, y2 = m1(x), m2(x)
        self.assertEqual(y1.shape, y2.shape)
        y1.sum().backward()
        # 这行曾经在 MPSNDArrayConvolutionA14.mm:4352 处导致崩溃的断言失败
        y2.sum().backward()

    @unittest.skipIf(product_version < 13.2, "在 macOS 12 上跳过此测试")
    def test_conv3d_backward_collision(self):
        # Conv3D 仅从 macOS 13.2 及以上版本可用
        x = torch.rand(1, 1, 10, 10, 20, device="mps", requires_grad=True)
        m1 = nn.Conv3d(1, 1, 3, stride=2, padding=1).to("mps")
        m2 = nn.Conv3d(1, 1, 4, stride=2, padding=1).to("mps")
        y1, y2 = m1(x), m2(x)
        self.assertEqual(y1.shape, y2.shape)
        y1.sum().backward()
        # 这行曾经在 MPSNDArrayConvolutionA14.mm:4352 处导致崩溃的断言失败
        y2.sum().backward()

    def test_gemm_permute_transpose(self):
        batch_size = 32
        n = 20
        hidden = 768
        num_attention_heads = 12
        attention_head_size = hidden // num_attention_heads

        def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        def attention2(key, *, workaround=False, device):
            key = transpose_for_scores(key)
            res = key.transpose(-1, -2)
            return res

        A = torch.randn(batch_size, n, hidden)
        A_mps = A.detach().clone().to("mps")

        r1 = attention2(A, device="cpu")
        r2 = attention2(A_mps, device="mps")

        r2_cpu = r2.to("cpu")
        self.assertEqual(r1, r2_cpu)
    # 定义一个测试函数，用于测试 GroupNorm 的反向传播功能
    def test_group_norm_backward(self, device='mps'):
        # 在 GitHub 上的 issue 88331 中可以找到更多细节：https://github.com/pytorch/pytorch/issues/88331
        shape = [1, 4, 16, 16]
        # 创建一个形状为 [1, 4, 16, 16] 的张量 x，其值全部为 7.0，设备为指定的 device
        x = torch.full(shape, 7.0, device=device)

        # 创建一个目标张量，形状为 [1, 3, 128, 128]，其值全部为 1，设备同样为指定的 device
        target = torch.ones((1, 3, 128, 128), device=device)

        # 创建一个输入通道数为 4，输出通道数为 128 的二维卷积层 conv_in，设备也为指定的 device
        conv_in = nn.Conv2d(4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
        # 创建一个输入通道数为 128，输出通道数为 3 的二维卷积层 conv_out，设备同样为指定的 device
        conv_out = nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), device=device)
        # 创建一个 GroupNorm 层，分组数为 32，输入通道数为 128，epsilon 设置为 1e-6，启用仿射变换，设备同样为指定的 device
        norm = nn.GroupNorm(32, 128, eps=1e-6, affine=True, device=device)

        # 启用梯度计算的上下文环境
        with torch.enable_grad():
            # 对输入张量 x 执行 detach() 操作，返回一个新的张量并要求计算梯度
            x = x.detach().requires_grad_()
            # 将 x 乘以 5.5
            out = 5.5 * x
            # 将结果传递给 conv_in 层
            out = conv_in(out)
            # 将 out 与 norm(out) 的结果相加
            out = out + norm(out)
            out = out + norm(out)
            out = out + norm(out)
            # 使用最近邻插值将 out 的尺寸放大 8 倍
            out = F.interpolate(out, scale_factor=8.0, mode="nearest")
            # 对放大后的结果应用 norm 层
            out = norm(out)
            # 将结果传递给 conv_out 层
            out = conv_out(out)

            # 计算 out 与 target 之间的欧几里得距离（L2 范数），然后对最后一个维度求和，得到损失值
            loss = (out - target).norm(dim=-1).sum()
            # 使用 autograd 计算损失相对于 x 的梯度
            grad = -torch.autograd.grad(loss, x)[0]
            # 断言梯度中没有任何 NaN 值，否则输出指定的错误信息
            self.assertFalse(grad.detach().isnan().any().item(), 'NaN gradients returned by autograd')
class TestPad(TestCaseMPS):
    # 定义测试类 TestPad，继承自 TestCaseMPS

    def test_constant_pad(self):
        # 定义测试方法 test_constant_pad

        # 创建一个 ConstantPad2d 模块 m，填充为 -2 的常数值为 3.5
        m = torch.nn.ConstantPad2d((-2, -2, -2, -2), 3.5)

        # 生成一个形状为 (1, 16, 16, 16) 的随机张量 input_cpu
        input_cpu = torch.randn(1, 16, 16, 16)

        # 将 input_cpu 克隆为 input_mps，转换到 "mps" 内存格式
        input_mps = input_cpu.detach().clone().to("mps")

        # 对 input_cpu 和 input_mps 分别使用模块 m 进行填充
        r_cpu = m(input_cpu)
        r_mps = m(input_mps)

        # 断言 r_cpu 和 r_mps.to("cpu") 的相等性
        self.assertEqual(r_cpu, r_mps.to("cpu"))

        # 定义任意的 pad 元组和 value 值
        pad = (1, 1, 0, 0, 0, 0)
        value = 3.5

        # 创建一个形状为 (1, 1, 3, 3, 3, 3, 3, 3, 3, 3) 的随机张量 input_cpu
        input_cpu = torch.randn((1, 1, 3, 3, 3, 3, 3, 3, 3, 3))

        # 将 input_cpu 克隆为 input_mps，转换到 "mps" 内存格式
        input_mps = input_cpu.detach().clone().to("mps")

        # 对 input_cpu 和 input_mps 分别使用 F.pad 进行填充
        r_cpu = F.pad(input_cpu, pad=pad, value=value)
        r_mps = F.pad(input_mps, pad=pad, value=value)

        # 断言 r_cpu 和 r_mps.to("cpu") 的相等性
        self.assertEqual(r_cpu, r_mps.to("cpu"))

    def test_circular_pad(self):
        # 定义测试方法 test_circular_pad

        # 创建一个形状为 (3, 3, 9, 9) 的全为 1 的张量 k_cpu，并转换为 "mps" 内存格式
        k_cpu = torch.ones(3, 3, 9, 9)
        k_mps = k_cpu.detach().clone().to("mps")

        # 创建一个形状为 (1, 3, 32, 32) 的随机张量 x_cpu，并转换为 "mps" 内存格式
        x_cpu = torch.rand(1, 3, 32, 32)
        x_mps = x_cpu.detach().clone().to("mps")

        # 对 x_cpu 和 x_mps 分别使用 F.pad 进行环绕填充
        x_pad_cpu = F.pad(x_cpu, (2, 2, 2, 2), mode='circular')
        x_pad_mps = F.pad(x_mps, (2, 2, 2, 2), mode='circular')

        # 对 x_pad_cpu 和 x_pad_mps 分别使用 F.conv2d 进行卷积操作
        y_cpu = F.conv2d(x_pad_cpu, k_cpu)
        y_mps = F.conv2d(x_pad_mps, k_mps)

        # 断言 y_cpu 和 y_mps.cpu() 的相等性
        self.assertEqual(y_cpu, y_mps.cpu())

    def test_constant_pad_4d_warning(self):
        # 定义测试方法 test_constant_pad_4d_warning

        # 创建一个形状为 (1, 2, 2, 2, 1, 1) 的随机张量 inputCPU
        inputCPU = torch.rand((1, 2, 2, 2, 1, 1))

        # 将 inputCPU 克隆为 inputMPS，转换到 "mps" 内存格式
        inputMPS = inputCPU.detach().clone().to('mps')

        # 对 inputCPU 和 inputMPS 分别使用 F.pad 进行填充
        outputCPU = F.pad(inputCPU, [0, 0, 0, 0, 0, 0, 1, 0])
        outputMPS = F.pad(inputMPS, [0, 0, 0, 0, 0, 0, 1, 0])

        # 断言 outputCPU 和 outputMPS 的相等性
        self.assertEqual(outputCPU, outputMPS)

    def test_constant_pad_nd_preserves_memory_format(self):
        # 定义测试方法 test_constant_pad_nd_preserves_memory_format

        # 创建一个形状为 (1, 2, 5, 3) 的随机张量 nchw_tensor
        nchw_tensor = torch.rand((1, 2, 5, 3))

        # 对 nchw_tensor 使用 torch.constant_pad_nd 进行常数填充，保持内存格式为 contiguous_format
        nchw_padded = torch.constant_pad_nd(nchw_tensor, [1, 2], 0.5)

        # 断言 nchw_padded 是否保持了 contiguous_format 的内存格式
        self.assertTrue(nchw_padded.is_contiguous(memory_format=torch.contiguous_format))

        # 将 nchw_tensor 转换为 channels_last 的内存格式 nhwc_tensor
        nhwc_tensor = nchw_tensor.contiguous(memory_format=torch.channels_last)

        # 对 nhwc_tensor 使用 torch.constant_pad_nd 进行常数填充，保持内存格式为 channels_last
        nhwc_padded = torch.constant_pad_nd(nhwc_tensor, [1, 2], 0.5)

        # 断言 nhwc_padded 是否保持了 channels_last 的内存格式
        self.assertTrue(nhwc_padded.is_contiguous(memory_format=torch.channels_last))


class TestLinalgMPS(TestCaseMPS):
    # 定义测试类 TestLinalgMPS，继承自 TestCaseMPS

    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False):
        # 定义私有方法 _test_addmm_addmv，接受参数 f, t, m, v 以及可选参数 alpha, beta, transpose_out

        # 获取张量 t 的数据类型和对应的 numpy 数据类型
        dtype = t.dtype
        numpy_dtype = dtype

        # 设置默认的 alpha 和 beta 值
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta

        # 使用函数 f 对 t, m, v 进行矩阵相乘操作，alpha 和 beta 作为参数传递
        res1 = f(t, m, v, alpha=alpha, beta=beta)

        # 创建一个与 res1 形状相同的全为 NaN 的张量 res2
        res2 = torch.full_like(res1, math.nan)

        # 如果 transpose_out 为 True，则在转置后克隆张量 res2，并保持 contiguous_format 的内存格式
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()

        # 使用函数 f 对 t, m, v 进行矩阵相乘操作，结果存储到 res2 中
        f(t, m, v, alpha=alpha, beta=beta, out=res2)

        # 计算 alpha * (m 转换为 numpy 数据类型的矩阵 @ v 转换为 numpy 数据类型的向量)
        res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())

        # 如果 beta 不为 0，则将 beta * t 加到 res3 上
        if beta != 0:
            res3 += (torch.mul(t, beta)).to(numpy_dtype).cpu().numpy()

        # 将 res3 转换为张量，并使用 t 的数据类型进行存储
        res3 = torch.from_numpy(res3).to(dtype)

        # 断言 res1 和 res2 的相等性
        self.assertEqual(res1, res2)

        # 断言 res1 和 res3 的相等性
        self.assertEqual(res1, res3)
    # 定义测试函数 test_addmm，测试 torch.addmm 函数的行为
    def test_addmm(self, device="mps", dtype=torch.float32):
        # 生成随机矩阵 M，大小为 10x25，指定设备和数据类型
        M = torch.randn(10, 25, device=device).to(dtype)
        # 生成随机矩阵 m1，大小为 10x50，指定设备和数据类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 生成随机矩阵 m2，大小为 50x25，指定设备和数据类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用辅助函数 _test_addmm_addmv，测试 torch.addmm 函数的结果
        self._test_addmm_addmv(torch.addmm, M, m1, m2)

        # 测试情况：beta=0，M 的值为 NaN
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        # 生成随机矩阵 m1，大小为 10x50，指定设备和数据类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 生成随机矩阵 m2，大小为 50x25，指定设备和数据类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用辅助函数 _test_addmm_addmv，测试 torch.addmm 函数的结果，传入 beta=0
        self._test_addmm_addmv(torch.addmm, M, m1, m2, beta=0)

        # 测试转置情况
        # 使用 itertools.product 生成所有 t1, t2, t3, t4 可能的组合
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            # 定义函数 maybe_transpose，根据条件 cond 是否转置矩阵 m
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            # 根据 t1 的值，决定是否对随机生成的矩阵 M 进行转置
            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            # 根据 t2 的值，决定是否对随机生成的矩阵 m1 进行转置
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            # 根据 t3 的值，决定是否对随机生成的矩阵 m2 进行转置
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            # 调用辅助函数 _test_addmm_addmv，测试 torch.addmm 函数的结果，传入是否要输出转置的 t4
            self._test_addmm_addmv(torch.addmm, M, m1, m2, transpose_out=t4)

    # 定义测试函数 _test_addr，测试 torch.addr 函数的行为
    def _test_addr(self, f, t, m, v, alpha=None, beta=None):
        # 获取目标张量 t 的数据类型
        dtype = t.dtype
        # 设置 numpy 中的数据类型为与目标张量 t 相同的数据类型
        numpy_dtype = dtype
        # 如果未指定 alpha，设为默认值 1.2；如果未指定 beta，设为默认值 0.8
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta
        # 调用 torch.addr 函数计算结果 res1
        res1 = f(t, m, v, alpha=alpha, beta=beta)
        # 使用 numpy 计算 addr 函数的期望结果 res2
        res2 = alpha * np.outer(m.to(numpy_dtype).cpu().numpy(), v.to(numpy_dtype).cpu().numpy())
        # 如果 beta 不为 0，则加上 beta 乘以 t 转换为 numpy 数组的结果
        if beta != 0:
            res2 += (torch.mul(t, beta)).to(numpy_dtype).cpu().numpy()
        # 将 res2 转换为与目标张量 t 相同的数据类型，并生成张量
        res2 = torch.from_numpy(res2).to(dtype)
        # 断言 torch.addr 函数的计算结果与预期结果 res2 相等
        self.assertEqual(res1, res2)

    # 定义测试函数 test_addr，测试 torch.addr 函数的行为
    def test_addr(self, device="mps", dtype=torch.float32):
        # 生成随机矩阵 M，大小为 10x25，指定设备和数据类型
        M = torch.randn(10, 25, device=device).to(dtype)
        # 生成随机矩阵 m1，大小为 10，指定设备和数据类型
        m1 = torch.randn(10, device=device).to(dtype)
        # 生成随机矩阵 m2，大小为 25，指定设备和数据类型
        m2 = torch.randn(25, device=device).to(dtype)
        # 调用辅助函数 _test_addr，测试 torch.addr 函数的结果
        self._test_addr(torch.addr, M, m1, m2)

        # 测试情况：beta=0，M 的值为 NaN
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        # 生成随机矩阵 m1，大小为 10，指定设备和数据类型
        m1 = torch.randn(10, device=device).to(dtype)
        # 生成随机矩阵 m2，大小为 25，指定设备和数据类型
        m2 = torch.randn(25, device=device).to(dtype)
        # 调用辅助函数 _test_addr，测试 torch.addr 函数的结果，传入 beta=0
        self._test_addr(torch.addr, M, m1, m2, beta=0)
    # 定义一个测试函数，用于测试矩阵的秩计算功能，支持指定设备和数据类型
    def test_matrix_rank(self, device="mps", dtype=torch.float32):
        # 引用 PyTorch 中的矩阵秩计算函数
        matrix_rank = torch.linalg.matrix_rank

        # 定义一个内部函数，用于运行具体的测试案例
        def run_test(shape0, shape1, batch):
            # 生成指定形状和批次的随机张量 a，使用指定的设备和数据类型
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            # 计算张量 a 的秩
            rank_a = matrix_rank(a)

            # 断言：张量 a 的秩应与其共轭转置的秩相等
            self.assertEqual(rank_a, matrix_rank(a.mH))

            # 计算 a 与其共轭转置的乘积 aaH
            aaH = torch.matmul(a, a.mH)
            # 计算 aaH 的秩
            rank_aaH = matrix_rank(aaH)
            # 使用 Hermitian 参数计算 aaH 的秩
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            # 断言：未使用 Hermitian 参数和使用 Hermitian 参数计算出的秩应相等
            self.assertEqual(rank_aaH, rank_aaH_hermitian)

            # 计算 a 的共轭转置与 a 的乘积 aHa
            aHa = torch.matmul(a.mH, a)
            # 断言：计算 aHa 的秩时，未使用 Hermitian 参数和使用 Hermitian 参数计算出的秩应相等
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # 使用 NumPy 检验上述计算结果的正确性
            self.assertEqual(rank_a, np.linalg.matrix_rank(a.cpu().numpy()))
            self.assertEqual(matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01))
            self.assertEqual(rank_aaH, np.linalg.matrix_rank(aaH.cpu().numpy()))
            self.assertEqual(matrix_rank(aaH, 0.01), np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01))

            # 如果 NumPy 版本支持 Hermitian 参数，则验证使用 Hermitian 参数时的计算结果
            if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
                self.assertEqual(rank_aaH_hermitian,
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), hermitian=True))
                self.assertEqual(matrix_rank(aaH, 0.01, True),
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01, True))

            # 检查使用 out 参数的计算结果
            out = torch.empty(a.shape[:-2], dtype=torch.int64, device=device)
            ans = matrix_rank(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, rank_a)

        # 定义测试用例的形状和批次，通过 product 函数生成形状和批次的组合
        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            # 在遇到下游函数未实现的 NotImplementedError 时，捕获并验证异常信息
            # TODO: 一旦所需函数被实现，移除此处的异常处理逻辑
            try:
                run_test(shape0, shape1, batch)
            except NotImplementedError as e:
                with self.assertRaisesRegex(
                        NotImplementedError,
                        "The operator 'aten::_linalg_svd.U' is not currently implemented for the MPS device."):
                    raise e

    # 使用 parametrize 装饰器对不同参数进行参数化测试
    @parametrize("m", [1, 32, 64])
    @parametrize("n", [48, 64])
    @parametrize("q_group", [32, 64, 128, 256])
    @parametrize("num_groups", [1, 2])
    # 定义一个测试函数，用于测试整型矩阵乘法的功能
    def test__int4_mm(self, m, n, q_group, num_groups):
        # 计算矩阵乘法中的内部维度大小
        k = q_group * num_groups
        inner_k_tiles = 2

        # 设置随机种子
        torch.manual_seed(1)
        # 创建随机的浮点数张量 a_f32 和 b_f32，设备为 "mps"
        a_f32 = torch.rand((m, k), device="mps")
        b_f32 = torch.rand((k, n), device="mps")

        # 定义一个函数，将权重张量转换为 int4pack 格式
        def convert_weight_to_int4pack(b):
            # 使用 _group_quantize_tensor 函数对张量 b 进行量化，得到 int32 格式的 b_int32 和量化参数 b_scales_and_zeros
            b_int32, b_scales_and_zeros = _group_quantize_tensor(
                b, n_bit=4, q_group_size=q_group
            )
            # 使用 torch._convert_weight_to_int4pack 函数将 b_int32 转换为 int4pack 格式的张量 b_int4pack
            b_int4pack = torch._convert_weight_to_int4pack(
                b_int32, inner_k_tiles
            )

            return b_int4pack, b_scales_and_zeros

        # 定义一个函数，执行 int4pack 格式的矩阵乘法
        def weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros):
            # 调用 torch._weight_int4pack_mm 函数执行 int4pack 格式的矩阵乘法
            return torch._weight_int4pack_mm(
                a, b_int4pack, q_group, b_scales_and_zeros
            )

        # 将 b_f32 转换为 int4pack 格式，并获取其量化参数 b_scales_and_zeros_f32
        b_int4pack, b_scales_and_zeros_f32 = convert_weight_to_int4pack(b_f32)

        # 遍历多种数据类型进行测试：torch.float16, torch.float32, 可能还包括 torch.bfloat16（如果产品版本大于 14.0）
        for dtype in [torch.float16, torch.float32] + ([torch.bfloat16] if product_version > 14.0 else []):
            # 将 a_f32 转换为当前数据类型 dtype
            a = a_f32.to(dtype=dtype)
            # 将 b_f32 转换为当前数据类型 dtype
            b = b_f32.to(dtype=dtype)
            # 将 b_scales_and_zeros_f32 转换为当前数据类型 dtype
            b_scales_and_zeros = b_scales_and_zeros_f32.to(dtype=dtype)
            # 计算参考结果 ref，使用 torch.mm 函数执行矩阵乘法
            ref = torch.mm(a, b)
            # 计算测试结果 res，调用 weight_int4pack_mm 函数执行 int4pack 格式的矩阵乘法
            res = weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros)

            # 计算相对误差的均值，用于断言测试结果是否小于 0.05
            mean_err = ((res - ref).abs() / ref).mean()
            self.assertLess(mean_err, 0.05)

    # 使用参数化装饰器 parametrize 来定义多组测试参数 m, k, n
    @parametrize("m", [1, 32, 64])
    @parametrize("k", [32, 64])
    @parametrize("n", [32, 64])
    # 定义测试函数，用于测试 int8 格式的矩阵乘法
    def test__int8_mm(self, m, k, n):
        # 设置随机种子
        torch.manual_seed(1)
        # 创建随机的浮点数张量 a_f32 和 b_f32，设备为 "mps"
        a_f32 = torch.rand((m, k), device="mps")
        b_f32 = torch.rand((n, k), device="mps")

        # 定义一个函数，将权重张量转换为 int8pack 格式
        def convert_weight_to_int8pack(b):
            # 使用 _dynamically_quantize_per_channel 函数对张量 b 进行通道精度动态量化，得到 int8pack 格式的 b_int8pack 和量化参数 b_scales
            b_int8pack, b_scales, _ = _dynamically_quantize_per_channel(
                b, -128, 127, torch.int8
            )
            return b_int8pack, b_scales

        # 定义一个函数，执行 int8pack 格式的矩阵乘法
        def weight_int8pack_mm(a, b_int8pack, b_scales):
            # 调用 torch._weight_int8pack_mm 函数执行 int8pack 格式的矩阵乘法
            return torch._weight_int8pack_mm(a, b_int8pack, b_scales)

        # 将 b_f32 转换为 int8pack 格式，并获取其量化参数 b_scales_f32
        b_int8pack, b_scales_f32 = convert_weight_to_int8pack(b_f32)

        # 遍历多种数据类型进行测试：torch.float16, torch.float32, 可能还包括 torch.bfloat16（如果产品版本大于 14.0）
        for dtype in [torch.float16, torch.float32] + ([torch.bfloat16] if product_version > 14.0 else []):
            # 将 a_f32 转换为当前数据类型 dtype
            a = a_f32.to(dtype=dtype)
            # 将 b_f32 转换为当前数据类型 dtype
            b = b_f32.to(dtype=dtype)
            # 将 b_scales_f32 转换为当前数据类型 dtype
            b_scales = b_scales_f32.to(dtype=dtype)
            # 调用 weight_int8pack_mm 函数执行 int8pack 格式的矩阵乘法，计算测试结果 res
            res = weight_int8pack_mm(a, b_int8pack, b_scales)
            # 计算参考结果 ref，使用 torch.mm 函数执行标准的矩阵乘法
            ref = torch.mm(a, b.transpose(0, 1))

            # 计算相对误差的均值，用于断言测试结果是否小于 0.05
            mean_err = ((res - ref).abs() / ref).mean()
            self.assertLess(mean_err, 0.05)
class TestGatherScatter(TestCaseMPS):
    # 定义一个测试类 TestGatherScatter，继承自 TestCaseMPS
    def test_slicing_with_step(self):
        # 测试切片操作包含步长
        # https://github.com/pytorch/pytorch/issues/78886
        # 创建一个在 MPS 设备上的全零张量 x_mps
        x_mps = torch.zeros(10, dtype=torch.float32, device="mps")
        # 将 x_mps 的偶数索引位置设置为 1.0
        x_mps[::2] = 1.0

        # 创建一个在 CPU 设备上的全零张量 x_cpu
        x_cpu = torch.zeros(10, dtype=torch.float32, device="cpu")
        # 将 x_cpu 的偶数索引位置设置为 1.0
        x_cpu[::2] = 1.0

        # 断言两个张量在值上相等
        self.assertEqual(x_cpu, x_mps)

    def test_cast_gather_scatter(self):
        # 测试类型转换、gather 和 scatter 操作
        for _ in range(0, 50):
            # 随机生成一个 uint8 类型的 5x5x4 数组 input
            input = np.random.randint(0, 255, size=(5, 5, 4), dtype=np.uint8)
            with torch.no_grad():
                # 创建一个在 MPS 设备上的 uint8 类型张量 s，并在第一维度上增加一个维度
                s = torch.tensor(input, dtype=torch.uint8, device="mps").unsqueeze(0)
                # 创建一个在 CPU 设备上的 uint8 类型张量 s_cpu，并在第一维度上增加一个维度
                s_cpu = torch.tensor(input, dtype=torch.uint8, device="cpu").unsqueeze(0)
                
                # 将 s 转换为 long 类型
                s = s.long()
                # 将 s_cpu 转换为 long 类型
                s_cpu = s_cpu.long()
                # 断言两个张量在值上相等
                self.assertEqual(s.cpu(), s_cpu)

                # 将 s 转换为 float 类型
                s = s.float()
                # 将 s_cpu 转换为 float 类型
                s_cpu = s_cpu.float()
                # 断言两个张量在值上相等
                self.assertEqual(s.cpu(), s_cpu)

                # 将 s 中的值除以 255
                s /= 255
                # 将 s_cpu 中的值除以 255
                s_cpu /= 255
                # 断言两个张量在值上相等
                self.assertEqual(s.cpu(), s_cpu)

    def test_slicing_replace_column(self):
        # 测试替换列的切片操作
        # https://github.com/pytorch/pytorch/issues/78074
        def _helper(tensor_data):
            # 根据 tensor_data 创建一个 CPU 上的张量 x_cpu
            x_cpu = torch.tensor(tensor_data)
            # 将 x_cpu 转移到 MPS 设备上得到 x_mps
            x_mps = x_cpu.to('mps')

            # 将 x_cpu 的所有行的第一列元素设置为 7
            x_cpu[:, 0] = 7
            # 将 x_mps 的所有行的第一列元素设置为 7
            x_mps[:, 0] = 7

            # 断言两个张量在值上相等
            self.assertEqual(x_cpu, x_mps)

        # 分别使用不同的 tensor_data 调用 _helper 函数进行测试
        _helper([[1, 2, 3], [4, 5, 6]])
        _helper([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        _helper([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    def test_inplace_scatter(self):
        # 测试原地 scatter 操作
        # https://github.com/pytorch/pytorch/issues/79672
        # 创建一个在 MPS 设备上的全一张量 a_mps 和 b_mps
        a_mps = torch.ones((2, 2),).to(torch.device("mps"))
        b_mps = torch.ones((2, 2),).to(torch.device("mps"))

        # 创建一个在 CPU 设备上的全一张量 a_cpu 和 b_cpu
        a_cpu = torch.ones((2, 2),).to(torch.device("cpu"))
        b_cpu = torch.ones((2, 2),).to(torch.device("cpu"))

        # 在 a_mps 和 b_mps 的第一列上执行原地加法
        a_mps[:, 0] += b_mps[:, 0]
        # 在 a_cpu 和 b_cpu 的第一列上执行原地加法
        a_cpu[:, 0] += b_cpu[:, 0]
        # 断言两个张量在值上相等
        self.assertEqual(a_cpu, a_mps)

        # 在 a_mps 和 b_mps 的第一列上执行原地加法，使用表达式形式
        a_mps[:, 0] = a_mps[:, 0] + b_mps[:, 0]
        # 在 a_cpu 和 b_cpu 的第一列上执行原地加法，使用表达式形式
        a_cpu[:, 0] = a_cpu[:, 0] + b_cpu[:, 0]
        # 断言两个张量在值上相等
        self.assertEqual(a_cpu, a_mps)
    # 测试切片置换的函数
    def test_permute_slicing(self):
        # 测试修复了在 https://github.com/pytorch/pytorch/issues/94190 中报告的崩溃问题
        cpu_x = (torch.randn([3, 2, 2]).float())
        # 将cpu_x的副本转换为mps设备上的张量
        mps_x = cpu_x.detach().clone().to('mps')
        # 对cpu_x进行维度置换并乘以2.0
        cpu_out = cpu_x.permute((2, 0, 1)) * 2.0
        # 对mps_x进行维度置换并乘以2.0
        mps_out = mps_x.permute((2, 0, 1)) * 2.0
        # 在修复PR#94259之前，此打印语句会导致崩溃
        print(torch.zeros_like(mps_out))
        # 测试填充标量mps中fill_scalar_mps()在问题＃94190中提到的修复
        self.assertEqual(torch.zeros_like(cpu_out), torch.zeros_like(mps_out))
        # 对cpu_x和mps_x的特定切片进行填充测试
        self.assertEqual(cpu_x[:, 1, :].fill_(1), mps_x[:, 1, :].fill_(1))

    # 判断是否为某个张量的视图
    def is_view_of(self, base, other):
        if (not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device):
            return False
        # 注意：只验证本地设备类型上的存储，因为一些加速器如XLA不会暴露存储
        if base.device.type == 'mps':
            if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                return False

        return True

    # 返回True如果v1和v2是同一个基张量的视图
    def is_view_of_same_base(self, v1, v2):
        if (not v1._is_view() or v1 is v2):
            return False
        return self.is_view_of(v1._base, v2)

    # 如果contiguous=True，则执行转置操作，否则原样返回输入张量
    def _do_transpose(self, x, contiguous=False, dim0=0, dim1=1):
        if contiguous:
            return x
        else:
            return x.transpose(dim0, dim1)

    # 测试对角视图功能
    def test_diagonal_view(self, device="mps"):
        t = torch.ones((5, 5), device=device)
        v = torch.diagonal(t)
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[0, 0], v[0])

        t = torch.ones((3, 3, 3), device="mps")
        v = torch.diagonal(t, offset=1, dim1=1, dim2=2)
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0, 1], v[0, 0])

    # 测试select函数生成的视图
    def test_select_view(self, device="mps") -> None:
        t = torch.ones((5, 5), device=device)
        v = t.select(0, 2)
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[2, 0], v[0])

    # 测试unbind函数生成的视图
    def test_unbind_view(self, device="mps") -> None:
        t = torch.zeros((5, 5), device=device)
        tup = torch.unbind(t)

        for idx, v in enumerate(tup):
            # 断言v是t的视图
            self.assertTrue(self.is_view_of(t, v))

            v[0] = idx + 1
            self.assertEqual(t[idx, 0], v[0])

    # 测试expand函数生成的视图
    def test_expand_view(self, device="mps") -> None:
        t = torch.ones((5, 1), device=device)
        v = t.expand(5, 5)
        # 断言v是t的视图
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])
    # 测试函数，用于测试 expand_as 方法是否正确生成视图
    def test_expand_as_view(self, device="mps"):
        # 创建一个形状为 (5, 1) 的张量，并指定设备
        t = torch.ones((5, 1), device=device)
        # 创建一个形状为 (5, 5) 的空张量，并指定设备
        e = torch.empty((5, 5), device=device)
        # 使用 t 的形状将其广播扩展为形状与 e 相同的张量
        v = t.expand_as(e)
        # 断言 t 是否是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 中的元素
        v[2, 2] = 0
        # 断言 t 和 v 的对应元素是否相等
        self.assertEqual(t[2, 0], v[2, 2])

    # 测试函数，用于测试 narrow 方法是否正确生成视图
    def test_narrow_view(self, device="mps"):
        # 创建一个形状为 (5, 5) 的张量，并指定设备
        t = torch.ones((5, 5), device=device)
        # 使用 narrow 方法从 t 中选择第 1 维度从索引 2 开始长度为 2 的视图
        v = torch.narrow(t, 1, 2, 2)
        # 断言 t 是否是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 中的元素
        v[0, 0] = 0
        # 断言 t 和 v 的对应元素是否相等
        self.assertEqual(t[0, 2], v[0, 0])

    # 测试函数，用于测试 permute 方法是否正确生成视图
    def test_permute_view(self, device="mps") -> None:
        # 创建一个形状为 (5, 5) 的张量，并指定设备
        t = torch.ones((5, 5), device=device)
        # 使用 permute 方法交换维度，生成 t 的转置视图
        v = t.permute(1, 0)
        # 断言 t 是否是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 中的元素
        v[0, 1] = 0
        # 断言 t 和 v 的对应元素是否相等
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于测试 transpose 相关方法是否正确生成视图
    def test_transpose_view(self, device="mps"):
        # 遍历 transpose 相关方法
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            # 创建一个形状为 (5, 5) 的张量，并指定设备
            t = torch.ones((5, 5), device=device)
            # 使用当前方法 fn 将 t 进行转置操作，生成视图 v
            v = fn(t, 0, 1)
            # 断言 t 是否是 v 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 中的元素
            v[0, 1] = 0
            # 断言 t 和 v 的对应元素是否相等
            self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于测试 inplace 的 transpose 相关方法是否正确生成视图
    def test_transpose_inplace_view(self, device="mps"):
        # 创建一个形状为 (5, 5) 的张量，并指定设备
        t = torch.ones(5, 5, device=device)
        # 创建 t 的视图 v
        v = t.view_as(t)
        # 使用 swapdims_ 方法进行 inplace 的维度交换操作，生成视图 v
        v = v.swapdims_(0, 1)
        # 断言 t 是否是 v 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改 v 中的元素
        v[0, 1] = 0
        # 断言 t 和 v 的对应元素是否相等
        self.assertEqual(t[1, 0], v[0, 1])

        # 同样的测试对 swapaxes_ 和 transpose_ 方法进行
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapaxes_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.transpose_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于测试 t 方法是否正确生成视图
    def test_t_view(self, device="mps"):
        # 创建一个形状为 (5, 5) 的张量，并指定设备
        t = torch.ones((5, 5), device=device)
        # 使用 t 的 t 方法获取其转置视图
        v = t.t()
        # 断言 t 是否是 v 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 中的元素
        v[0, 1] = 0
        # 断言 t 和 v 的对应元素是否相等
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于测试 inplace 的 t 方法是否正确生成视图
    def test_t_inplace_view(self, device="mps"):
        # 创建一个形状为 (5, 5) 的张量，并指定设备
        t = torch.ones(5, 5, device=device)
        # 创建 t 的视图 v
        v = t.view_as(t)
        # 使用 t 的 t_ 方法进行 inplace 的转置操作，生成视图 v
        v = v.t_()
        # 断言 t 是否是 v 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 修改 v 中的元素
        v[0, 1] = 0
        # 断言 t 和 v 的对应元素是否相等
        self.assertEqual(t[1, 0], v[0, 1])

    # 测试函数，用于测试 T 属性方法是否正确生成视图
    def test_T_view(self, device="mps"):
        # 遍历 T 属性相关的方法
        for op in ("T", "H", "mT", "mH"):
            # 创建一个形状为 (5, 5) 的张量，并指定设备
            t = torch.ones((5, 5), device=device)
            # 获取 t 的 op 方法，生成视图 v
            v = getattr(t, op)
            # 断言 t 是否是 v 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 中的元素
            v[0, 1] = 0
            # 断言 t 和 v 的对应元素是否相等
            self.assertEqual(t[1, 0], v[0, 1])
    # 测试在指定设备上创建一个全为1的张量，并对其进行视图展开操作
    def test_unfold_view(self, device="mps"):
        t = torch.ones(10, device=device)  # 创建一个全为1的张量，长度为10，指定设备为device
        v = t.unfold(0, 3, 2)  # 在维度0上展开张量，窗口大小为3，步长为2
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图

        v[1, 0] = 0  # 修改v中索引为(1, 0)的元素为0
        self.assertEqual(t[2], v[1, 0])  # 断言t中索引为2的元素与v中索引为(1, 0)的元素相等

    # 测试在指定设备上创建一个全为1的张量，并对其进行挤压操作
    def test_squeeze_view(self, device="mps"):
        t = torch.ones(5, 1, 5, device=device)  # 创建一个全为1的张量，形状为(5, 1, 5)，指定设备为device
        v = torch.squeeze(t)  # 挤压张量t，去除所有维度为1的维度
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图
        v[0, 1] = 0  # 修改v中索引为(0, 1)的元素为0
        self.assertIs(t, v._base)  # 断言t和v的基张量是同一个对象

    # 测试在指定设备上创建一个全为1的张量，并对其进行原地挤压操作
    def test_squeeze_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        v = t.view_as(t)  # 将张量t按照自身的形状重新视图化
        v = v.squeeze_()  # 对v进行原地挤压操作，去除所有维度为1的维度
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图
        v[0, 1] = 0  # 修改v中索引为(0, 1)的元素为0
        self.assertIs(t, v._base)  # 断言t和v的基张量是同一个对象

    # 测试在指定设备上创建一个全为1的张量，并对其进行挤压操作（增加一个维度）
    def test_unsqueeze_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        v = torch.unsqueeze(t, 1)  # 对张量t在第1维度上进行挤压操作，增加一个维度
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图

        v[0, 0, 1] = 0  # 修改v中索引为(0, 0, 1)的元素为0
        self.assertEqual(t[0, 1], v[0, 0, 1])  # 断言t中索引为(0, 1)的元素与v中索引为(0, 0, 1)的元素相等

    # 测试在指定设备上创建一个全为1的张量，并对其进行原地挤压操作（增加一个维度）
    def test_unsqueeze_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        v = t.view_as(t)  # 将张量t按照自身的形状重新视图化
        v = v.unsqueeze_(1)  # 对v进行原地挤压操作，增加一个维度
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图
        v[0, 0, 1] = 0  # 修改v中索引为(0, 0, 1)的元素为0
        self.assertEqual(t[0, 1], v[0, 0, 1])  # 断言t中索引为(0, 1)的元素与v中索引为(0, 0, 1)的元素相等

    # 测试在指定设备上创建一个全为1的张量，并对其进行按步长展开的操作
    def test_as_strided_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        v = torch.as_strided(t, (25,), (1,))  # 使用as_strided对张量t进行按步长展开操作，得到长度为25的视图
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图

        v[6] = 0  # 修改v中索引为6的元素为0
        self.assertEqual(t[1, 1], v[6])  # 断言t中索引为(1, 1)的元素与v中索引为6的元素相等

    # 测试在指定设备上创建一个全为1的张量，并对其进行原地按步长展开的操作
    def test_as_strided_inplace_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        v = t.view_as(t)  # 将张量t按照自身的形状重新视图化
        v = v.as_strided_((25,), (1,))  # 对v进行原地按步长展开操作，得到长度为25的视图
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图
        v[6] = 0  # 修改v中索引为6的元素为0
        self.assertEqual(t[1, 1], v[6])  # 断言t中索引为(1, 1)的元素与v中索引为6的元素相等

    # 测试在指定设备上创建一个全为1的张量，并对其进行视图化操作
    def test_view_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        v = t.view(25)  # 将张量t按照形状(25,)进行视图化
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图

        v[6] = 0  # 修改v中索引为6的元素为0
        self.assertEqual(t[1, 1], v[6])  # 断言t中索引为(1, 1)的元素与v中索引为6的元素相等

    # 测试在指定设备上创建一个全为1的张量，并对其进行视图化操作（使用另一个张量的形状）
    def test_view_as_view(self, device="mps"):
        t = torch.ones(5, 5, device=device)  # 创建一个全为1的张量，形状为(5, 5)，指定设备为device
        e = torch.empty((25,))  # 创建一个形状为(25,)的空张量e
        v = t.view_as(e)  # 将张量t按照e的形状进行视图化
        self.assertTrue(self.is_view_of(t, v))  # 断言v是否是t的视图

        v[6] = 0  # 修改v中索引为6的元素为0
        self.assertEqual(t[1, 1], v[6])  # 断
    # 测试函数，用于测试 reshape_as 方法是否正确生成视图（view）
    def test_reshape_as_view(self, device="mps"):
        # 创建一个大小为 5x5 的张量 t，设备为指定的 device
        t = torch.ones(5, 5, device=device)
        # 创建一个空的张量 e，大小为 (25,)，设备为指定的 device
        e = torch.empty((25,), device=device)
        # 使用 t 的形状将 e 重塑为一个视图 v
        v = t.reshape_as(e)
        # 断言 v 是否是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改视图 v 的元素，影响原始张量 t 的相应位置
        v[6] = 0
        # 断言 t 和 v 的对应位置元素是否相等
        self.assertEqual(t[1, 1], v[6])

    # 测试函数，用于测试 reshape 方法是否能生成非视图（non-view）
    def test_reshape_nonview(self, device="mps"):
        # 创建一个大小为 5x5 的张量 t，设备为指定的 device
        t = torch.ones(5, 5, device=device)
        # 使用 transpose 方法对 t 进行转置，并将结果重塑为非视图 nv
        nv = torch.reshape(t.t(), (25,))
        # 断言 nv 是否不是 t 的视图
        self.assertFalse(self.is_view_of(t, nv))

        # 修改非视图 nv 的元素，验证其不会影响原始张量 t 的相应位置
        nv[6] = 0
        # 断言 t 和 nv 的对应位置元素不相等
        self.assertNotEqual(t[1, 1], nv[6])

    # 测试函数，用于测试 flatten 方法生成的视图是否正确
    def test_flatten_view(self, device="mps"):
        # 辅助函数，用于测试写入操作是否正确传播
        def test_writes_propagate(t, v):
            # 创建张量 t 和其视图 v 的索引
            idx_t = (0,) * t.ndim
            idx_v = (0,) * v.ndim
            # 将视图 v 的第一个元素设为 0
            v[idx_v] = 0
            # 断言 t 和 v 的相应索引位置元素是否相等
            self.assertEqual(t[idx_t], v[idx_v])

        # 创建一个大小为 1x2x3x4 的张量 t，设备为指定的 device
        t = torch.ones(1, 2, 3, 4, device=device)
        # 使用 flatten 方法将张量 t 展平为视图 v
        v = t.flatten()
        # 断言 v 是否是 t 的视图
        self.assertTrue(self.is_view_of(t, v))
        # 调用辅助函数测试写入操作是否正确传播
        test_writes_propagate(t, v)

        # 创建一个零维张量 t，设备为指定的 device
        t = torch.tensor(1, device=device)
        # 使用 flatten 方法将零维张量 t 展平为视图 v
        v = t.flatten()
        # 调用辅助函数测试写入操作是否正确传播
        test_writes_propagate(t, v)
        # 断言 v 是否是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 使用 transpose 方法对大小为 1x2x3x4 的张量 t 进行转置
        t = torch.ones(1, 2, 3, 4, device=device).transpose(2, 3)
        # 使用 flatten 方法将转置后的张量 t 展平为视图 v
        v = t.flatten(0, 1)
        # 调用辅助函数测试写入操作是否正确传播
        test_writes_propagate(t, v)
        # 断言 v 和 t 是否共享相同的基础张量
        self.assertTrue(self.is_view_of_same_base(t, v))

        # 创建一个步幅满足特定条件的大小为 720 的张量 t，设备为指定的 device
        t = torch.ones(720, device=device) \
            .as_strided((2, 3, 2, 3, 5, 4), (6, 2, 15, 5, 1, 0))
        # 使用 flatten 方法对张量 t 进行多次展平生成视图 v1, v2, v3
        v1 = t.flatten(0, 1)
        v2 = v1.flatten(1, 3)
        v3 = v2.flatten(2, 2)
        # 分别调用辅助函数测试写入操作是否正确传播
        test_writes_propagate(t, v1)
        self.assertTrue(self.is_view_of_same_base(t, v1))
        test_writes_propagate(t, v2)
        self.assertTrue(self.is_view_of_same_base(t, v2))
        test_writes_propagate(t, v3)
        self.assertTrue(self.is_view_of_same_base(t, v3))

    # 测试函数，用于测试 flatten 方法生成的非视图是否正确
    def test_flatten_nonview(self, device="mps"):
        # 辅助函数，用于验证是否为非视图
        def assert_is_nonview(t, nv):
            # 创建张量 t 和非视图 nv 的索引
            idx_t = (0,) * t.ndim
            idx_nv = (0,) * nv.ndim
            # 断言 nv 是否不是视图
            self.assertFalse(nv._is_view())
            # 将非视图 nv 的第一个元素设为 0
            nv[idx_nv] = 0
            # 断言 t 和 nv 的相应索引位置元素不相等
            self.assertNotEqual(t[idx_t], nv[idx_nv])

        # 创建一个大小为 2x3x2x3 的张量 t，设备为指定的 device，并对其进行转置
        t = torch.ones(2, 3, 2, 3, device=device).transpose(2, 3)
        # 使用 flatten 方法对张量 t 进行展平生成非视图 nv
        nv = t.flatten(1, 3)
        # 调用辅助函数验证 nv 是否为非视图
        assert_is_nonview(t, nv)

        # 创建一个大小为 2x2 的张量 t，设备为指定的 device，并对其转置
        t = torch.ones(2, 2, device=device).T
        # 使用 flatten 方法对张量 t 进行展平生成非视图 nv
        nv = t.flatten()
        # 调用辅助函数验证 nv 是否为非视图
        assert_is_nonview(t, nv)

        # 使用 flatten 方法对大小为 2x2 的张量 t 进行展平，起始维度等于结束维度
        t = torch.ones(2, 2, device=device)
        # 使用 flatten 方法展平后，返回原始张量 t
        nv = t.flatten(1, 1)
        # 断言 t 和 nv 是同一个对象
        self.assertIs(t, nv)

    # 测试函数，用于测试基本的索引和切片生成的视图是否正确
    def test_basic_indexing_slice_view(self, device="mps"):
        # 创建一个大小为 5x5 的张量 t，设备为指定的 device
        t = torch.ones(5, 5, device=device)
        # 对张量 t 进行切片操作生成视图 v
        v = t[:2, :3]
        # 断言 v 是否是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改视图 v 的元素，影响原始张量 t 的相应位置
        v[0, 0] = 0
        # 断言 t 和 v 的对应位置元素是否相等
        self.assertEqual(t[0, 0], v[0, 0])
    # 定义测试方法，用于测试基本的索引和视图操作，接受设备参数，默认为 "mps"
    def test_basic_indexing_ellipses_view(self, device="mps"):
        # 创建一个5x5的张量，所有维度上的索引均使用省略号（...）
        t = torch.ones(5, 5, device=device)
        # 使用省略号创建视图 v，包含所有行，但仅前两列的数据
        v = t[..., :2]
        # 断言 v 是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的数据，也会影响到 t 的相应位置
        v[0, 0] = 0
        # 断言 t 和 v 在修改后的位置上的值相等
        self.assertEqual(t[0, 0], v[0, 0])

    # 定义测试方法，用于测试添加了新维度的索引和视图操作，接受设备参数，默认为 "mps"
    def test_basic_indexing_newaxis_view(self, device="mps"):
        # 创建一个5x5的张量
        t = torch.ones(5, 5, device=device)
        # 使用新维度索引创建视图 v，包含所有列的前两行数据
        v = t[None, :2, 3]
        # 断言 v 是 t 的视图
        self.assertTrue(self.is_view_of(t, v))

        # 修改 v 的数据，也会影响到 t 的相应位置
        v[0, 0] = 0
        # 断言 t 和 v 在修改后的位置上的值相等
        self.assertEqual(t[0, 3], v[0, 0])

    # 定义测试方法，用于测试分块视图操作，接受设备参数，默认为 "mps"
    def test_chunk_view(self, device="mps"):
        # 创建一个3x3的零张量
        t = torch.zeros(3, 3, device=device)
        # 使用 torch.chunk 将 t 按行分成3块
        l = torch.chunk(t, 3)

        # 遍历分块后的每个视图 v
        for idx, v in enumerate(l):
            # 断言 v 是 t 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 的数据，也会影响到 t 的相应位置
            v[0, 0] = idx + 1
            # 断言 t 和 v 在修改后的位置上的值相等
            self.assertEqual(t[idx, 0], v[0, 0])

    # 定义测试方法，用于测试分割视图操作，接受设备参数，默认为 "mps"
    def test_split_view(self, device="mps"):
        # 创建一个3x3的零张量
        t = torch.zeros(3, 3, device=device)
        # 使用 torch.split 将 t 按指定长度[1, 1, 1]分割
        l = torch.split(t, [1, 1, 1])

        # 遍历分割后的每个视图 v
        for idx, v in enumerate(l):
            # 断言 v 是 t 的视图
            self.assertTrue(self.is_view_of(t, v))

            # 修改 v 的数据，也会影响到 t 的相应位置
            v[0, 0] = idx + 1
            # 断言 t 和 v 在修改后的位置上的值相等
            self.assertEqual(t[idx, 0], v[0, 0])

    # 定义测试方法，用于测试 movedim 操作的视图，接受设备参数，默认为 "mps"
    def test_movedim_view(self, device="mps"):
        # 定义内部方法 run_test，接受设备参数和操作 op
        def run_test(device, op):
            # 创建一个3x3的零张量
            t = torch.zeros(3, 3, device=device)
            # 执行操作 op 并获得输出 out
            out = op(t)

            # 断言 out 是 t 的视图
            self.assertTrue(self.is_view_of(t, out))

            # 随机修改 out 的值，并验证 t 对应位置的值是否改变
            for _ in range(3):
                idx_1, idx_2 = random.randint(0, 2), random.randint(0, 2)
                out[idx_1, idx_2] = random.random()
                # 断言 t 和 out 在修改后的位置上的值相等
                self.assertEqual(t[idx_2, idx_1], out[idx_1, idx_2])

        # 针对 movedim 和 moveaxis 两个函数分别执行测试
        for fn in [torch.movedim, torch.moveaxis]:
            # 部分应用 fn，定义操作 op
            op = partial(fn, source=(0, 1), destination=(1, 0))
            # 执行测试方法 run_test
            run_test(device, op)

            # 部分应用 fn，定义操作 op
            op = partial(fn, source=0, destination=1)
            # 执行测试方法 run_test
            run_test(device, op)

    # 测试生成的 view_copy 核函数及其导数是否正确实现
    def test_view_copy(self, device="mps"):
        # 创建一个带梯度的4维随机张量 a
        a = torch.randn(4, device=device, requires_grad=True)
        # 克隆 a，断开计算图并设置为需要梯度
        a_ref = a.clone().detach().requires_grad_()
        # 使用 view 将 a_ref 重塑为2x2的视图 a_view
        a_view = a_ref.view(2, 2)
        # 使用 view_copy 将 a 重塑为2x2的张量 a_view_copy
        a_view_copy = torch.view_copy(a, (2, 2))

        # 断言 view_copy 操作不保持视图关系
        self.assertTrue(self.is_view_of(a_ref, a_view))
        self.assertFalse(self.is_view_of(a, a_view_copy))

        # 对 a_view_copy 执行求和并反向传播
        a_view_copy.sum().backward()
        # 对 a_view 执行求和并反向传播
        a_view.sum().backward()

        # 断言前向和反向传播后的形状和结果相同
        self.assertEqual(a_view_copy, a_view)
        self.assertEqual(a.grad, a_ref.grad)
    def test_view_copy_out(self, device="mps"):
        # 创建一个形状为 (2, 2) 的随机张量 a，使用指定的设备
        a = torch.randn(2, 2, device=device)
        # 创建一个形状为 (2,) 的空张量 out，使用指定的设备
        out = torch.empty(2, device=device)

        # 将张量 a 的对角线复制到 out 中
        torch.diagonal_copy(a, out=out)
        # 获取从张量 a 复制得到的对角线张量
        expected = torch.diagonal_copy(a)

        # 断言复制得到的对角线张量与预期的 out 张量相等
        self.assertEqual(expected, out)

        # 创建一个形状为 (4,) 的随机张量 a，使用指定的设备
        a = torch.randn(4, device=device)
        # 创建两个形状为 (2,) 的空张量 out1 和 out2，使用指定的设备
        out1 = torch.empty(2, device=device)
        out2 = torch.empty(2, device=device)

        # 将张量 a 分割为两个张量，分别存储到 out1 和 out2 中
        torch.split_copy(a, 2, out=(out1, out2))
        # 获取从张量 a 分割得到的两个张量
        expected1, expected2 = torch.split_copy(a, 2)

        # 断言分割得到的两个张量与预期的 out1 和 out2 张量相等
        self.assertEqual(expected1, out1)
        self.assertEqual(expected2, out2)

    def test_detached_view_copy(self, device="mps"):
        # 创建一个包含元素 [0, 1] 的张量 x
        x = torch.arange(2)
        # 使用 .detach() 方法从张量 x 中分离出元素 y，并使其不再是视图，而是一个连续的张量，带有非零偏移
        y = x[1].detach()
        # 将张量 y 移动到指定设备上
        z = y.to(device)
        # 断言张量 y 和移动后的张量 z 相等（在 CPU 上）
        self.assertEqual(y, z.cpu())

    def test_empty_reshape(self, device="mps"):
        # 创建一个形状为 (0, 6) 的随机张量 x，使用指定的设备
        x = torch.randn(0, 6, device=device)
        # 断言对形状为 (0, 6) 的张量 x 进行重新形状操作得到形状为 (1, 0, 6, 1, 1) 的张量
        self.assertEqual((1, 0, 6, 1, 1), x.reshape(1, 0, 6, 1, 1).shape)
        # 断言重新形状后的张量 x 与原始张量共享数据指针
        self.assertEqual(x.data_ptr(), x.reshape(1, 0, 6, 1, 1).data_ptr())

        # 匹配 NumPy 语义，不推断具有自由度的维度的大小
        self.assertRaises(RuntimeError, lambda: x.reshape(0, -1))

    def test_expand(self, device="mps"):
        # 创建一个形状为 (1, 8, 1) 的随机张量 tensor，使用指定的设备
        tensor = torch.rand(1, 8, 1, device=device)
        # 创建一个形状为 (5,) 的随机张量 tensor2，使用指定的设备
        tensor2 = torch.rand(5, device=device)
        # 创建一个形状为 (4, 8, 5) 的随机张量 template，使用指定的设备
        template = torch.rand(4, 8, 5, device=device)
        # 获取 template 的目标大小
        target = template.size()
        # 使用 expand_as 将 tensor 扩展为与 template 相同的形状，并检查其形状是否等于 target
        self.assertEqual(tensor.expand_as(template).size(), target)
        # 使用 expand 将 tensor 扩展为 (4, 8, 5) 的形状，并检查其形状是否等于 target
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        # 使用 expand 将 tensor 扩展为与 target 相同的形状，并检查其形状是否等于 target
        self.assertEqual(tensor.expand(target).size(), target)
        # 使用 expand_as 将 tensor2 扩展为与 template 相同的形状，并检查其形状是否等于 target
        self.assertEqual(tensor2.expand_as(template).size(), target)
        # 使用 expand 将 tensor2 扩展为 (4, 8, 5) 的形状，并检查其形状是否等于 target
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        # 使用 expand 将 tensor2 扩展为与 target 相同的形状，并检查其形状是否等于 target
        self.assertEqual(tensor2.expand(target).size(), target)

        # 测试双重扩展
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # 测试非连续张量
        noncontig = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(noncontig.is_contiguous())
        # 使用 expand 将非连续张量 noncontig 扩展为 (2, 5, 4, 3) 的形状，并检查其形状是否等于 repeat 的结果
        self.assertEqual(noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1))

        # 确保与 unsqueeze 兼容
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        self.assertEqual(expanded, unsqueezed)
        self.assertEqual(expanded.stride(), unsqueezed.stride())

        # 测试将 -1 作为目标大小
        self.assertEqual(tensor.expand(4, -1, 5), tensor.expand(4, 8, 5))
        self.assertRaises(RuntimeError, lambda: tensor2.expand(-1, -1))

        # 测试将空张量扩展为空
        self.assertEqual(torch.zeros(0, device=device).expand((0,)), torch.zeros(0, device=device))
    # 定义一个测试方法，用于测试空张量的形状变换操作，设备默认为 "mps"
    def test_view_empty(self, device="mps"):
        # 创建一个形状为 (0, 6) 的随机张量 x，指定设备为 device
        x = torch.randn(0, 6, device=device)
        # 断言 x 经过 view 操作后的形状是否为 (1, 0, 6, 1, 1)
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    # 定义一个测试方法，用于测试张量的 reshape 操作，设备默认为 "mps"
    def test_reshape(self, device="mps"):
        # 创建一个形状为 (3, 3) 的随机张量 x，指定设备为 device
        x = torch.randn(3, 3, device=device)
        # 断言不同 reshape 操作后，张量的数据指针是否保持不变
        self.assertEqual(x.data_ptr(), x.reshape(-1).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape(1, 9, 1).data_ptr())
        self.assertEqual(torch.reshape(x, (9,)), x.reshape(9))
        # 预期抛出 RuntimeError 异常，因为尝试进行非法的 reshape 操作
        self.assertRaises(RuntimeError, lambda: x.reshape(-1, -1))

        # 创建一个形状为 (4, 4, 4) 的随机张量 y，并选择部分数据进行操作，设备默认为 device
        y = torch.randn(4, 4, 4, device=device)[:, 0, :]
        # 如果设备不是 "meta"，则验证 reshape 后数据指针不同
        if device != "meta":
            self.assertNotEqual(y.data_ptr(), y.reshape(-1).data_ptr())
        # 断言经过 contiguous 后的 view 操作与 reshape 操作的结果相同
        self.assertEqual(y.contiguous().view(-1), y.reshape(-1))
        # 断言 reshape 操作后的张量数据指针与原始数据指针相同
        self.assertEqual(y.reshape(2, 2, 4).data_ptr(), y.data_ptr())

        # 创建一个形状为 () 的随机标量张量 s，指定设备为 device
        s = torch.randn((), device=device)
        # 断言标量 s 经过 reshape 操作后的数据指针是否保持不变
        self.assertEqual(s.data_ptr(), s.reshape(()).data_ptr())
        # 断言 reshape 操作后的形状为 (1,)
        self.assertEqual(s.reshape(-1).shape, (1,))
        # 预期抛出 RuntimeError 异常，因为尝试进行非法的 reshape 操作
        self.assertRaises(RuntimeError, lambda: s.reshape(2))

        # 创建一个空张量 empty，指定设备为 device
        empty = torch.tensor([], device=device)
        # 断言空张量经过 reshape 操作后仍然为空
        self.assertEqual(empty, empty.reshape(-1))
        self.assertEqual(empty, empty.reshape([0]))
        # TODO: 一旦支持多维空张量，修复这些断言
        # 断言 reshape 操作后的形状为 (0, 1) 和 (1, 0)
        self.assertEqual(empty.reshape([0, 1]).shape, (0, 1))
        self.assertEqual(empty.reshape([1, -1]).shape, (1, 0))
        # 预期抛出 RuntimeError 异常，因为尝试进行非法的 reshape 操作
        self.assertRaises(RuntimeError, lambda: empty.reshape(1))

        # 再次创建形状为 (3, 3) 的随机张量 x，指定设备为 device
        x = torch.randn(3, 3, device=device)
        # 断言 x 经过 reshape_as 操作后的数据指针是否与随机生成的张量的数据指针相同
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(9)).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(1, 9, 1)).data_ptr())
        # 预期抛出 RuntimeError 异常，因为尝试将 x reshape 成不兼容形状
        self.assertRaises(RuntimeError, lambda: x.reshape_as(torch.rand(10, device=device)))

    # 定义一个测试方法，用于测试张量的 narrow 操作，设备默认为 "mps"
    def test_narrow(self, device="mps"):
        # 创建一个 3x3 的张量 x
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        # 测试 narrow 操作，选择不同的起始位置和长度
        self.assertEqual(x.narrow(0, 0, 1), torch.tensor([[0, 1, 2]]))
        self.assertEqual(x.narrow(0, 0, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(x.narrow(0, 1, 1), torch.tensor([[3, 4, 5]]))
        self.assertEqual(x.narrow(0, -1, 1), torch.tensor([[6, 7, 8]]))
        self.assertEqual(x.narrow(0, -2, 2), torch.tensor([[3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(0, -3, 3), torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(-1, -1, 1), torch.tensor([[2], [5], [8]]))
        self.assertEqual(x.narrow(-2, -1, 1), torch.tensor([[6, 7, 8]]))
    # 测试在给定设备上对张量进行切片操作
    def test_narrow_tensor(self, device="mps"):
        # 创建一个二维张量
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        # 使用 narrow 方法进行切片操作，从第 0 维开始，长度为 1，预期结果是第一行的数据
        self.assertEqual(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]]))
        # 测试当索引是浮点数时是否抛出异常
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.), 1)
        # 测试当索引是包含单个元素的张量时是否抛出异常
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        # 测试当索引是包含多个元素的张量时是否抛出异常
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    # 测试张量的转置操作
    def test_t(self, device="mps"):
        # 测试零维张量的转置
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # 测试一维张量的转置
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # 测试二维张量的转置
        x = torch.rand((2, 2))
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # 测试三维张量的转置
        x = torch.rand((2, 2, 2))
        # 预期在三维张量上调用 t() 方法会引发 RuntimeError 错误，因为只支持二维及以下的转置操作
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 dimensions, but self is 3D'):
            x.t()
        x = x.to_sparse()
        # 预期在稀疏张量的三维情况下调用 t() 方法会引发 RuntimeError 错误
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 sparse and 0 dense dimensions'):
            x.t()

    # 测试张量的分割操作
    def test_split(self, device="mps"):
        tensor = torch.rand(7, 4)
        split_size = 3
        dim = 0
        target_sizes = ([3, 4], [3, 4], [1, 4])
        # 使用 split 方法对张量在指定维度上进行分割，验证每个分割后的子张量大小是否符合预期
        splits = tensor.split(split_size, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            # 使用 narrow 方法检查分割后的子张量是否与原张量在指定维度上的切片相同
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        # 变长分割操作
        tensor = torch.randn(20, 10)
        dim = 0
        split_sizes = [5, 5, 10]
        target_sizes = ([[5, 10], [5, 10], [10, 10]])
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        split_sizes = [2, 2, 6]
        target_sizes = ([20, 2], [20, 2], [20, 6])
        dim = 1
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]
    # 定义一个名为 test_chunk 的测试方法，接受一个可选的设备参数，默认为 "mps"
    def test_chunk(self, device="mps"):
        # 创建一个大小为 (4, 7) 的随机张量 tensor
        tensor = torch.rand(4, 7)
        # 指定要分块的数目为 3
        num_chunks = 3
        # 指定在哪个维度上进行分块，这里是第 1 维
        dim = 1
        # 每个分块目标的大小列表
        target_sizes = ([4, 3], [4, 3], [4, 1])
        # 使用 chunk 方法对 tensor 进行分块，按维度 dim 进行分块，返回分块后的张量列表 splits
        splits = tensor.chunk(num_chunks, dim)
        # 初始化起始索引
        start = 0
        # 遍历目标大小和分块结果
        for target_size, split in zip(target_sizes, splits):
            # 断言每个分块的大小与目标大小一致
            self.assertEqual(split.size(), target_size)
            # 断言分块后的数据与原始 tensor 在指定维度上的切片相等
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split,
                             atol=0, rtol=0)
            # 更新起始索引，以便下一次切片
            start = start + target_size[dim]

        # 检查无效的分块大小是否会触发 RuntimeError 异常
        error_regex = 'chunk expects.*greater than 0'
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(0)
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(-2)

    # 定义一个名为 test_unsqueeze 的测试方法，接受一个可选的设备参数，默认为 "mps"，返回 None
    def test_unsqueeze(self, device="mps") -> None:
        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 在第 1 维上插入一个维度，返回新的张量 y
        y = x.unsqueeze(1)
        # 断言插入维度后的张量 y 与使用 view 方法变换后的张量相等
        self.assertEqual(y, x.view(2, 1, 3, 4))
        # 对 x 进行克隆并在第 2 维上进行原地插入维度操作，返回新的张量 y
        y = x.clone().unsqueeze_(2)
        # 断言插入维度后的张量 y 与使用 view 方法变换后的张量相等
        self.assertEqual(y, x.view(2, 3, 1, 4))

        # 在切片后的张量 x 上进行不连续性检查
        x = x[:, 1]
        # 断言切片后的张量 x 不是连续的
        self.assertFalse(x.is_contiguous())
        # 在不连续的张量 x 上插入一个维度，返回新的张量 y
        y = x.unsqueeze(1)
        # 断言插入维度后的张量 y 与先使 x 连续再进行 view 方法变换后的张量相等
        self.assertEqual(y, x.contiguous().view(2, 1, 4))
        # 对 x 进行克隆并在第 3 维上进行原地插入维度操作，返回新的张量 y
        y = x.clone().unsqueeze_(2)
        # 断言插入维度后的张量 y 与先使 x 连续再进行 view 方法变换后的张量相等
        self.assertEqual(y, x.contiguous().view(2, 4, 1))

    # 单元测试，特殊情况下的转置复制（详见 ATen/native/Copy.cpp）
    def test_big_transpose(self, device="mps"):
        # 在指定设备上创建一个形状为 (456, 789) 的随机张量 t
        t = torch.rand(456, 789, device=device)
        # 对 t 进行转置并确保连续性，返回新的张量 t1
        t1 = t.t().contiguous()
        # 使用 numpy 和 CPU 上的数据创建 t 的转置，返回新的张量 t2
        t2 = torch.from_numpy(t.cpu().numpy().transpose())
        # 断言转置后的张量 t1 与使用 numpy 转置后的张量 t2 相等
        self.assertEqual(t1, t2)

    # 单元测试，测试张量的转置操作，返回 None
    def test_T(self, device="mps"):
        # 在指定设备上创建一个形状为 (2, 3, 4) 的随机张量 a
        a = torch.randn(2, 3, 4, device=device)
        # 使用 T 属性进行转置操作，返回新的张量 t1
        t1 = a.T
        # 使用 permute 方法按指定维度顺序进行转置，返回新的张量 t2
        t2 = a.permute(2, 1, 0)
        # 断言转置后的张量 t1 与 permute 方法得到的张量 t2 相等
        self.assertEqual(t2, t1)
        # 在指定设备上创建一个形状为 (10,) 的随机张量 b
        b = torch.randn(10, device=device)
        # 断言张量 b 与其转置后的张量相等
        self.assertEqual(b, b.T)

    # 单元测试，测试张量的多种转置操作，返回 None
    def test_transposes(self, device="mps", dtype=torch.float32):
        # 遍历转置操作符 op
        for op in ("T", "H", "mT", "mH", "adjoint"):
            # 根据不同的 op 选择不同的形状 shapes
            shapes = ((2, 3), (2, 3, 4)) if op[0] == "m" or op == "adjoint" else ((2, 3),)
            # 遍历不同的形状 shape
            for shape in shapes:
                # 使用 make_tensor 创建指定设备和数据类型的张量 a
                a = make_tensor(shape, device=device, dtype=dtype)
                # 根据 op 获取张量 a 的转置结果 t1
                t1 = getattr(a, op)
                # 如果 op 是 "adjoint"，则再次调用 t1 方法得到真正的转置结果
                if op == "adjoint":
                    t1 = t1()
                # 使用 transpose 方法对张量 a 进行转置，得到张量 t2
                t2 = a
                if a.ndim != 0:
                    t2 = t2.transpose(-2, -1)
                # 如果 op 是 "H" 结尾或者 op 是 "adjoint"，则对 t2 进行共轭操作
                if op[-1] == "H" or op == "adjoint":
                    t2 = t2.conj()
                # 断言转置后的张量 t1 与 t2 相等
                self.assertEqual(t2, t1)

    # 单元测试，测试张量的转置操作引发的异常，返回 None
    def test_transposes_errors(self, device="mps", dtype=torch.float32):
        # 遍历转置操作符 op
        for op in ("H", "mT", "mH", "adjoint"):
            # 根据不同的 op 选择不同的形状 shapes
            shapes = ((2,), (2, 3, 4)) if op == "H" else ((2,),)
            # 遍历不同的形状 shape
            for shape in shapes:
                # 使用 make_tensor 创建指定设备和数据类型的张量 a
                a = make_tensor(shape, device=device, dtype=dtype)
                # 使用 assertRaisesRegex 断言在调用 op 操作时会抛出 RuntimeError 异常
                with self.assertRaisesRegex(RuntimeError, "only supported on matrices"):
                    t1 = getattr(a, op)
                    if op == "adjoint":
                        t1 = t1()
    # 定义一个测试函数，用于测试不同的 Python 数据类型在指定设备上的功能
    def test_python_types(self, device="mps"):
        # 创建两个随机张量 a1 和 a2，形状为 (1, 2)，在指定设备上，数据类型为 torch.float32
        a1 = torch.randn((1, 2), device=device, dtype=torch.float32)
        a2 = torch.randn((1, 2), device=device, dtype=torch.float32)
        # 断言 a1 和 a2 的数据类型相同
        self.assertEqual(a1.dtype, a2.dtype)

        # 创建两个张量 b1 和 b2，使用不同的数据类型和相同的设备
        # b1 是一个整数范围张量，数据类型为 torch.int64
        b1 = torch.arange(10, 20, dtype=torch.int64, device=device)
        # b2 是一个整数范围张量，数据类型为 Python 内置的 int，但会被转换为 torch.int64
        b2 = torch.arange(10, 20, dtype=int, device=device)
        # 断言 b1 和 b2 的数据类型相同
        self.assertEqual(b1.dtype, b2.dtype)

        # 创建两个布尔张量 c1 和 c2，数据类型为 torch.bool，在指定设备上
        c1 = torch.tensor([True, False], dtype=torch.bool, device=device)
        c2 = torch.tensor([True, False], dtype=bool, device=device)
        # 断言 c1 和 c2 的数据类型相同
        self.assertEqual(c1.dtype, c2.dtype)

    # TODO: is resize best put in test_view_ops?
    # 定义一个测试函数，验证 resize_as_ 方法是否保留张量的步幅信息
    def test_resize_as_preserves_strides(self, device="mps"):
        # 创建一个空的张量 x，形状为 (2, 3)，然后对其进行转置
        x = torch.empty(2, 3).t()
        # 记录转置前的步幅信息
        old_strides = x.stride()
        # 使用 resize_as_ 方法将 x 重新调整为其自身的大小
        x.resize_as_(x)
        # 断言调整大小后的张量 x 的步幅信息与调整前保持一致
        self.assertEqual(x.stride(), old_strides)

    # 定义一个测试函数，验证在指定内存格式下使用 resize_as_ 方法是否保持张量的连续性
    def test_memory_format_resize_as(self, device="mps"):
        # 定义一个辅助函数 test_helper，用于测试不同形状、内存格式的张量
        def test_helper(shape, memory_format, device="mps"):
            # 创建一个形状为 shape 的随机张量 xc，在指定设备上保持连续性
            xc = torch.randn(shape, device=device).contiguous(memory_format=memory_format)
            # 创建一个长度与 xc 元素个数相同的随机张量 flat，在指定设备上
            flat = torch.randn(xc.numel(), device=device)
            # 使用 resize_as_ 方法将 flat 调整为与 xc 相同的大小，同时保持内存格式为 torch.preserve_format
            flat.resize_as_(xc, memory_format=torch.preserve_format)
            # 断言调整后的 flat 张量在指定内存格式下仍然是连续的
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        # 分别测试 channels_last 和 channels_last_3d 内存格式下的张量
        test_helper((10, 3, 32, 32), torch.channels_last, device="mps")
        test_helper((3, 10, 3, 32, 32), torch.channels_last_3d, device="mps")

    # 定义一个测试函数，验证在指定内存格式下使用 resize_ 方法是否保持张量的连续性
    def test_memory_format_resize_(self, device="mps"):
        # 定义一个辅助函数 test_helper，用于测试不同形状、元素数量、内存格式的张量
        def test_helper(shape, numel, memory_format, device="mps"):
            # 创建一个长度为 numel 的随机张量 flat，在指定设备上
            flat = torch.randn(numel, device=device)
            # 使用 resize_ 方法将 flat 调整为指定形状 shape，并保持内存格式为 memory_format
            flat.resize_(shape, memory_format=memory_format)
            # 断言调整后的 flat 张量在指定内存格式下仍然是连续的
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        # 分别测试 channels_last 和 channels_last_3d 内存格式下的张量
        test_helper((10, 3, 32, 32), 10 * 3 * 32 * 32, torch.channels_last, device="mps")
        test_helper((3, 10, 3, 32, 32), 3 * 10 * 3 * 32 * 32, torch.channels_last_3d, device="mps")

    # TODO: OpInfo this
    # 定义一个私有方法，用于测试至少函数对不同张量维度的效果
    def _test_atleast(self, device, torch_fn):
        # 0 维张量
        s = torch.tensor(0.5, dtype=torch.double, requires_grad=True)
        # 对 0 维张量 s 使用指定的 torch_fn 函数进行梯度检查和二阶梯度检查
        gradcheck(lambda x: torch_fn(x), s)
        gradgradcheck(lambda x: torch_fn(x), s)

        # 1 维张量
        a = torch.rand(4, dtype=torch.double, requires_grad=True)
        # 对 1 维张量 a 使用指定的 torch_fn 函数进行梯度检查和二阶梯度检查
        gradcheck(lambda x: torch_fn(x), a)
        gradgradcheck(lambda x: torch_fn(x), a)

        # 2, 3, 4 维张量
        b = torch.rand(4, 3, dtype=torch.double, requires_grad=True)
        c = torch.rand(4, 3, 2, dtype=torch.double, requires_grad=True)
        d = torch.rand(4, 3, 2, 1, dtype=torch.double, requires_grad=True)

        input_tuple = (s, a, b, c, d)
        # 对包含多个张量的输入元组 input_tuple 使用指定的 torch_fn 函数进行梯度检查和二阶梯度检查
        gradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)
        gradgradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)

    # 定义一个测试函数，验证至少函数在指定设备上对不同张量维度的效果
    def test_atleast_gradient(self, device="mps"):
        # 分别使用 atleast_1d、atleast_2d、atleast_3d 函数测试至少函数对不同张量维度的效果
        self._test_atleast(device, torch.atleast_1d)
        self._test_atleast(device, torch.atleast_2d)
        self._test_atleast(device, torch.atleast_3d)
    # 测试视图操作，将随机生成的张量 tensor 转换为不同形状的视图，并验证其尺寸是否与目标尺寸相同
    def test_view(self, device="mps"):
        # 创建一个形状为 (15,) 的随机张量 tensor，指定设备为 device
        tensor = torch.rand(15, device=device)
        # 创建一个形状为 (3, 5) 的随机张量 template，指定设备为 device
        template = torch.rand(3, 5, device=device)
        # 创建一个形状为空的张量 empty，指定设备为 device
        empty = torch.empty(0, device=device)
        # 获取 template 的形状作为目标尺寸 target
        target = template.size()
        # 断言通过 view_as 方法转换后的 tensor 的尺寸与 target 相同
        self.assertEqual(tensor.view_as(template).size(), target)
        # 断言通过 view 方法转换后的 tensor 的尺寸与 target 相同
        self.assertEqual(tensor.view(3, 5).size(), target)
        # 断言通过 view 方法传入 torch.Size 对象转换后的 tensor 的尺寸与 target 相同
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        # 断言通过 view 方法传入 -1 和 5 转换后的 tensor 的尺寸与 target 相同
        self.assertEqual(tensor.view(-1, 5).size(), target)
        # 断言通过 view 方法传入 3 和 -1 转换后的 tensor 的尺寸与 target 相同
        self.assertEqual(tensor.view(3, -1).size(), target)
        # 将 tensor 重新以 (5, 3) 的形状进行视图转换，并填充随机数
        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        # 断言空张量 empty 通过 view_as 方法转换后的尺寸仍为 empty
        self.assertEqual(empty.view_as(empty), empty)
        # 断言空张量 empty 通过 view 方法传入 0 转换后的尺寸仍为 empty
        self.assertEqual(empty.view(0), empty)
        # 断言空张量 empty 通过 view 方法传入多个 0 和 3、0、1 转换后的尺寸为 torch.Size([0, 3, 0, 1])
        self.assertEqual(empty.view(0, 3, 0, 1).size(), torch.Size([0, 3, 0, 1]))
        # 断言空张量 empty 通过 view 方法传入 0 转换后的尺寸仍为 empty
        self.assertEqual(empty.view(0, 3, 0, 1).view(0), empty)

        # 测试使用空张量进行尺寸推断
        # 断言空张量 empty 通过 view 方法传入 -1 转换后的尺寸为 torch.Size([0])
        self.assertEqual(empty.view(-1).size(), torch.Size([0]))
        # 断言空张量 empty 通过 view 方法传入 10、3 和 -1 转换后的尺寸为 torch.Size([10, 3, 0])
        self.assertEqual(empty.view(10, 3, -1).size(), torch.Size([10, 3, 0]))

        # 预期 RuntimeError，并验证其异常信息是否包含指定内容
        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(-1, 0)

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(3, 0, -1, 0)

        # 预期 RuntimeError，验证 lambda 表达式中的异常是否被抛出
        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))
    # 定义测试方法，用于测试在指定设备上查看所有数据类型和设备
    def test_view_all_dtypes_and_devices(self, device="mps"):
        # 遍历数据类型列表，包括浮点型和布尔型
        for dt in (torch.float, torch.bool):
            # 创建张量 x，包含二维数组，并指定数据类型和设备
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            # 断言：查看张量 x 重塑为一维后的形状应为 [6]
            self.assertEqual(x.view(6).shape, [6])
class TestConvolutionMPS(TestCaseMPS):
    def test_conv1d_all_strides_paddings(self):
        # 定义一个辅助函数，用于测试不同的步幅和填充值组合
        def helper(stride, padding):
            # 生成一个形状为 (1, 57, 40) 的随机张量
            y_cpu = torch.randn(1, 57, 40)
            # 创建一个 1D 卷积层，输入通道数为 57，输出通道数为 20，设置步幅和填充值
            conv_cpu = nn.Conv1d(57, 20, stride=stride, padding=padding, kernel_size=3, bias=False)
            # 深拷贝卷积层到 GPU（MPS）设备
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
            # 在 CPU 上对 y_cpu 应用 conv_cpu
            x_cpu = conv_cpu(y_cpu)

            # 将 y_cpu 移动到 GPU（MPS）设备
            y_gpu = y_cpu.to(device='mps')
            # 在 GPU 上对 y_gpu 应用 conv_gpu
            x_gpu = conv_gpu(y_gpu)
            # 断言 CPU 和 GPU 计算结果的一致性
            self.assertEqual(x_cpu, x_gpu.cpu())

        # 对步幅和填充值的组合进行测试，步幅范围为 1 到 3，填充值范围也为 1 到 3
        for stride in range(1, 4):
            for padding in range(1, 4):
                helper(stride, padding)


    def test_conv1d_channels_last(self):
        # 创建一个 1D 卷积层，输入通道数为 1，输出通道数为 128，内核大小为 3
        model_cpu = torch.nn.Conv1d(1, 128, 3)
        # 创建一个大小为 (128*176) 的序列，转换为形状为 (128, 176, 1)，并对维度进行置换
        a_cpu = torch.arange((128 * 176), dtype=torch.float32)
        a_cpu = a_cpu.view(128, 176, 1).permute(0, 2, 1)
        # 在 CPU 上对 a_cpu 应用 model_cpu
        out_cpu = model_cpu(a_cpu)

        # 创建 a_cpu 的深拷贝，并将其移动到 MPS（GPU）设备
        a_mps = a_cpu.detach().clone().to("mps")
        # 将 model_cpu 移动到 MPS（GPU）设备
        model_mps = model_cpu.to("mps")
        # 在 MPS（GPU）上对 a_mps 应用 model_mps
        out_mps = model_mps(a_mps)

        # 断言 CPU 和 MPS（GPU）计算结果的一致性，设置相对误差和绝对误差的阈值
        self.assertEqual(out_cpu, out_mps.cpu(), rtol=2.6e-05, atol=2e-04)

    def test_conv_transpose_1d_all_strides(self):
        # 定义一个辅助函数，用于测试不同步幅的转置卷积
        def helper(stride):
            # 创建一个形状为 (1, 1, 2) 的全为 1 的张量
            y_cpu = torch.ones(1, 1, 2)
            # 创建一个 1D 转置卷积层，输入通道数为 1，输出通道数为 1，内核大小为 1，设置步幅和填充值
            deconv_cpu = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=1, stride=stride, bias=False, padding=1)
            # 设置 deconv_cpu 的权重为全为 1
            deconv_cpu.weight.data = torch.ones(1, 1, 2)
            # 深拷贝 deconv_cpu 到 GPU（MPS）设备
            deconv_gpu = copy.deepcopy(deconv_cpu).to(device='mps')
            # 在 CPU 上对 y_cpu 应用 deconv_cpu
            x_cpu = deconv_cpu(y_cpu)

            # 将 y_cpu 移动到 GPU（MPS）设备
            y_gpu = y_cpu.to(device='mps')
            # 在 GPU 上对 y_gpu 应用 deconv_gpu
            x_gpu = deconv_gpu(y_gpu)
            # 断言 CPU 和 GPU 计算结果的一致性
            self.assertEqual(x_cpu, x_gpu.cpu())

        # 对步幅为 1, 2, 3 的转置卷积进行测试
        [helper(stride) for stride in [1, 2, 3]]

    def test_conv_transpose_1d_nn_functional(self):
        # 创建随机张量 tin，tparams 和 tbias
        tin = torch.rand((1, 512, 1245), dtype=torch.float32)
        tparams = torch.rand((512, 256, 16), dtype=torch.float32)
        tbias = torch.rand((256), dtype=torch.float32)

        # 在 CPU 上应用函数式 API 进行 1D 转置卷积
        device = 'cpu'
        tcpu = torch.nn.functional.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

        # 将输入张量和参数移动到 MPS（GPU）设备，并在 MPS（GPU）上应用函数式 API 进行 1D 转置卷积
        device = 'mps'
        tgpu = torch.nn.functional.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

        # 断言 CPU 和 MPS（GPU）计算结果的一致性，设置相对误差和绝对误差的阈值
        self.assertEqual(tcpu, tgpu.cpu(), rtol=2.6e-05, atol=2e-04)
    def test_conv_backward_1d_channels_last(self):
        def helper(shape, in_channels=1, out_channels=1, kernel_size=3, groups=1):
            # 定义一个辅助函数，用于测试反向卷积的计算
            # 创建 CPU 上的卷积层，设置输入通道数、输出通道数、卷积核大小和分组数，并要求梯度计算
            conv_cpu = torch.nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups).requires_grad_()
            # 创建 MPS（Memory Persistence Service）上的卷积层，设置同样的参数，并转移到 MPS，同时要求梯度计算
            conv_mps = torch.nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups).to("mps")
            conv_mps.weight.data = conv_cpu.weight.data.detach().clone().to("mps").requires_grad_(True)
            conv_mps.bias.data = conv_cpu.bias.data.detach().clone().to("mps").requires_grad_(True)

            # 生成指定形状的随机数据张量
            data = torch.rand(shape, dtype=torch.float32)
            # 对数据进行维度置换和连续化，并要求梯度计算
            x_cpu = data.permute(0, 2, 1).contiguous().requires_grad_(True)
            # 将数据张量克隆到 MPS，并进行维度置换和连续化，并要求梯度计算
            x_mps = data.permute(0, 2, 1).detach().clone().to("mps").contiguous().requires_grad_(True)
            # 在 CPU 上对数据进行卷积操作
            res_cpu = conv_cpu(x_cpu)
            # 在 MPS 上对数据进行卷积操作
            res_mps = conv_mps(x_mps)
            # 断言 CPU 和 MPS 上的卷积结果相等
            self.assertEqual(res_cpu, res_mps)
            # 对 CPU 上的卷积结果进行求和并反向传播
            res_cpu = res_cpu.sum().backward()
            # 对 MPS 上的卷积结果进行求和并反向传播
            res_mps = res_mps.sum().backward()

            # 断言 CPU 和 MPS 上的卷积权重梯度相等
            self.assertEqual(conv_cpu.weight.grad, conv_mps.weight.grad, rtol=2.6e-05, atol=2e-04)
            # 断言 CPU 和 MPS 上的输入数据梯度相等
            self.assertEqual(x_cpu.grad, x_mps.grad)

        # 调用 helper 函数进行不同形状的测试
        helper(shape=(1, 176, 1))
        helper(shape=(2, 12, 1))
        helper(shape=(3, 176, 1))
        helper(shape=(4, 376, 1))
        helper(shape=(1024, 376, 9), in_channels=9, out_channels=1, groups=1)
        helper(shape=(1024, 376, 9), in_channels=9, out_channels=9, groups=3)

    def test_conv1d_contiguous(self):
        # 创建一个在 CPU 上的 Conv1d 模型，输入通道为 1，输出通道为 128，卷积核大小为 3
        model_cpu = torch.nn.Conv1d(1, 128, 3)
        # 创建一个形状为 (128, 1, 176) 的全为 1 的数据张量
        a_cpu = torch.ones(128, 1, 176)
        # 在 CPU 上对数据进行卷积操作
        out_cpu = model_cpu(a_cpu)

        # 将数据张量克隆到 MPS，并转移模型到 MPS 上
        a_mps = a_cpu.detach().clone().to("mps")
        model_mps = model_cpu.to("mps")
        # 在 MPS 上对数据进行卷积操作
        out_mps = model_mps(a_mps)

        # 断言 CPU 和 MPS 上的卷积结果形状相等
        self.assertEqual(out_cpu.shape, out_mps.shape)
        # 断言 CPU 和 MPS 上的卷积结果张量值完全相等（需要将 MPS 上的结果转回 CPU 进行比较）
        self.assertEqual(out_cpu, out_mps.cpu())
    # 定义一个测试方法，用于测试不同的卷积操作情况
    def test_conv2d_all_strides_paddings(self):
        # 引用了一个 GitHub 问题链接，解释了为什么需要这个测试
        # https://github.com/pytorch/pytorch/issues/83180
        def helper(N, C, H, W, groups, input_mem_format, weight_mem_format, permute_data):
            # 生成一个 NCHW 格式的随机张量 x_cpu，并设置为需要计算梯度
            x_cpu = torch.randn(N, C, H, W).to(memory_format=input_mem_format).requires_grad_()
            # 使用 x_cpu 的克隆，转换到 mps 设备上，并设置为需要计算梯度
            x_mps = x_cpu.detach().clone().to(device='mps').requires_grad_()

            # 如果 permute_data 为 True，则对数据进行轴置换
            if permute_data:
                x_cpu.permute(0, 2, 3, 1)
                x_mps.permute(0, 2, 3, 1)

            # 遍历 strideX 和 strideY 的可能取值，构建不同配置下的 Conv2d 对象并进行测试
            for strideX in range(1, 4):
                for strideY in range(1, 4):
                    # 创建一个在 CPU 上计算的 Conv2d 对象 conv_cpu，设置输入通道数、输出通道数、卷积核大小、组数和步长
                    conv_cpu = torch.nn.Conv2d(
                        in_channels=N, out_channels=C, kernel_size=H, groups=groups, stride=(strideX, strideY)).requires_grad_()
                    # 将 conv_cpu 的权重数据转换为指定的内存格式，并设置为需要计算梯度
                    conv_cpu.weight.data = conv_cpu.weight.to(memory_format=weight_mem_format).requires_grad_()

                    # 创建一个在 mps 设备上计算的 Conv2d 对象 conv_mps，设置输入通道数、输出通道数、卷积核大小、组数、步长和设备
                    conv_mps = torch.nn.Conv2d(
                        in_channels=N, out_channels=C, kernel_size=H, groups=groups, stride=(strideX, strideY), device="mps")
                    # 将 conv_cpu 的权重数据克隆到 mps 设备上，并设置为需要计算梯度
                    conv_mps.weight.data = conv_cpu.weight.data.detach().clone().to("mps").requires_grad_()
                    # 将 conv_cpu 的偏置数据克隆到 mps 设备上，并设置为需要计算梯度
                    conv_mps.bias.data = conv_cpu.bias.data.detach().clone().to("mps").requires_grad_()

                    # 分别计算在不同设备上的卷积结果 res_cpu 和 res_mps，并进行相等性断言
                    res_cpu = conv_cpu(x_cpu)
                    res_mps = conv_mps(x_mps)
                    self.assertEqual(res_cpu, res_mps.cpu(), rtol=1e-03, atol=1e-05)

                    # 对 res_cpu 和 res_mps 的和进行反向传播，并进行相等性断言
                    res_cpu = res_cpu.sum().backward()
                    res_mps = res_mps.sum().backward()
                    self.assertEqual(res_cpu, res_mps, rtol=2.6e-05, atol=2e-04)
                    # 对 conv_cpu 和 conv_mps 的权重梯度进行相等性断言
                    self.assertEqual(conv_cpu.weight.grad, conv_mps.weight.grad, rtol=2.6e-05, atol=2e-04)
                    # 对 conv_cpu 和 conv_mps 的偏置梯度进行相等性断言
                    self.assertEqual(conv_cpu.bias.grad, conv_mps.bias.grad)
                    # 对 x_cpu 和 x_mps 的梯度进行相等性断言
                    self.assertEqual(x_cpu.grad, x_mps.grad)

        # 遍历不同的内存格式和轴置换设置，对 helper 方法进行测试
        for mem_format_input in [torch.contiguous_format, torch.channels_last]:
            for mem_format_weight in [torch.contiguous_format, torch.channels_last]:
                for permute_data in [True, False]:
                    helper(2, 2, 3, 6, 1, mem_format_input, mem_format_weight, permute_data)
                    helper(10, 10, 4, 6, 2, mem_format_input, mem_format_weight, permute_data)
                    helper(32, 32, 4, 6, 2, mem_format_input, mem_format_weight, permute_data)
    # 定义测试用例：验证二维转置卷积操作在不同存储格式下的正确性
    def test_conv_transpose_2d_strided(self):
        # 定义内部辅助函数，对模型和输入数据进行处理并比较输出结果
        def helper(m_cpu, memory_format):
            # 深拷贝 CPU 模型并设置需要梯度计算
            m_mps = copy.deepcopy(m_cpu).requires_grad_()
            # 将权重和偏置数据转换为 "mps" 设备并设置需要梯度计算
            m_mps.weight.data = m_cpu.weight.data.detach().clone().to("mps").requires_grad_()
            m_mps.bias.data = m_cpu.bias.data.detach().clone().to("mps").requires_grad_()

            # 生成随机输入数据并设置存储格式及需要梯度计算
            input_cpu = torch.randn(20, 16, 50, 100).to(memory_format=memory_format).requires_grad_()
            # 将 CPU 输入数据深拷贝并转换为 "mps" 设备
            input_mps = input_cpu.detach().clone().to("mps")

            # 计算 CPU 模型的输出
            output_cpu = m_cpu(input_cpu)
            # 计算 "mps" 模型的输出
            output_mps = m_mps(input_mps)
            # 断言两种模型的输出应相等
            self.assertEqual(output_cpu, output_mps)

        # 遍历测试用的存储格式：连续存储和通道最后存储
        for mem_format_input in [torch.contiguous_format, torch.channels_last]:
            # 测试正方形核和相同步长的情况
            helper(nn.ConvTranspose2d(16, 33, 3, stride=2).requires_grad_(), mem_format_input)

            # 测试非正方形核、不同步长和有填充的情况
            helper(nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)).requires_grad_(), mem_format_input)

    # 定义测试用例：验证指定输出大小的二维转置卷积操作
    def test_conv_transpose_2d_specified_output(self):
        # 生成随机输入数据
        input_cpu = torch.randn(1, 16, 12, 12)
        # 将 CPU 输入数据深拷贝并转换为 "mps" 设备
        input_mps = input_cpu.detach().clone().to("mps")

        # 创建 CPU 下采样卷积层
        downsample_cpu = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        # 创建 "mps" 设备下采样卷积层
        downsample_mps = nn.Conv2d(16, 16, 3, stride=2, padding=1, device="mps")
        # 将权重和偏置数据转换为 "mps" 设备并设置需要梯度计算
        downsample_mps.weight.data = downsample_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        downsample_mps.bias.data = downsample_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        # 创建 CPU 上采样卷积层
        upsample_cpu = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        # 创建 "mps" 设备上采样卷积层
        upsample_mps = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, device="mps")
        # 将权重和偏置数据转换为 "mps" 设备并设置需要梯度计算
        upsample_mps.weight.data = upsample_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        upsample_mps.bias.data = upsample_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        # 计算 CPU 下采样卷积的输出
        h_cpu = downsample_cpu(input_cpu)
        # 计算 "mps" 设备下采样卷积的输出
        h_mps = downsample_mps(input_mps)
        # 断言两者输出应相等
        self.assertEqual(h_cpu, h_mps)

        # 检查输出大小是否一致
        size_cpu = h_cpu.size()
        size_mps = h_mps.size()
        self.assertEqual(size_cpu, size_mps)

        # 计算 CPU 上采样卷积的输出
        output_cpu = upsample_cpu(h_cpu, output_size=input_cpu.size())
        # 计算 "mps" 设备上采样卷积的输出
        output_mps = upsample_mps(h_mps, output_size=input_mps.size())
        # 断言两者输出应相等
        self.assertEqual(output_cpu, output_mps)
        # 检查输出大小是否一致
        self.assertEqual(output_cpu.size(), output_mps.size())

    # 定义测试用例：验证单一步长的二维卷积操作
    def test_conv2d_single_stride(self):
        # 生成随机输入数据
        y_cpu = torch.randn(2, 2, 3, 6)
        # 将 CPU 输入数据转换为 "mps" 设备
        y_gpu = y_cpu.to(device='mps')

        # 遍历不同步长的范围
        for stride in range(1, 4):
            # 创建 CPU 卷积层
            conv_cpu = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=stride)
            # 深拷贝 CPU 卷积层并将其转换到 "mps" 设备
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')

            # 计算 CPU 卷积的输出
            x_cpu = conv_cpu(y_cpu)
            # 计算 "mps" 设备上的卷积输出
            x_gpu = conv_gpu(y_gpu)
            # 断言两者输出应相等
            self.assertEqual(x_cpu, x_gpu.cpu(), rtol=1e-03, atol=1e-05)

    # 如果产品版本小于 13.2，则跳过这个测试用例，在 macOS 12 上被跳过
    @unittest.skipIf(product_version < 13.2, "Skipped on macOS 12")
    # 定义一个测试方法，用于测试 Conv3d 单步长的情况
    def test_conv3d_single_stride(self):
        # 检查当前操作系统是否支持 Conv3d，要求 MacOS 版本在 13.2 及以上
        # 生成一个随机张量 y_cpu，形状为 (2, 2, 3, 6)
        y_cpu = torch.randn(2, 2, 3, 6)
        # 将 y_cpu 复制到 'mps' 设备上，通常是多进程共享内存（MPS）设备
        y_gpu = y_cpu.to(device='mps')
        
        # 循环遍历步长参数从 1 到 3
        for stride in range(1, 4):
            # 在 CPU 上创建一个 Conv3d 模型，设置输入通道数为 2，输出通道数为 2，卷积核大小为 (2, 2, 2)，步长为当前循环变量 stride
            conv_cpu = torch.nn.Conv3d(in_channels=2, out_channels=2, kernel_size=2, stride=stride)
            # 深度复制 conv_cpu 并将其复制到 'mps' 设备上，得到 conv_gpu
            conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
            
            # 使用 conv_cpu 对 y_cpu 进行卷积操作，得到输出 x_cpu
            x_cpu = conv_cpu(y_cpu)
            # 使用 conv_gpu 对 y_gpu 进行卷积操作，得到输出 x_gpu
            x_gpu = conv_gpu(y_gpu)
            
            # 断言 x_cpu 与 x_gpu.cpu() 的值近似相等，相对误差为 1e-03，绝对误差为 1e-05
            self.assertEqual(x_cpu, x_gpu.cpu(), rtol=1e-03, atol=1e-05)
class TestAdvancedIndexing(TestCaseMPS):
    supported_dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.uint8]
    supported_np_dtypes = [np.float32, np.float16, np.int64, np.int32, np.int16, np.uint8]

    def test_nonzero_no_warning(self):
        # 设定设备为 "mps"
        device = "mps"
        # 创建一个随机张量 t，形状为 (2, 2)，在指定设备上
        t = torch.randn((2, 2), device=device)
        # 使用 warnings 模块捕获所有警告信息
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 调用 torch.nonzero() 函数，返回非零元素的索引
            torch.nonzero(t)
            # 调用张量的 nonzero() 方法，也返回非零元素的索引
            t.nonzero()
            # 断言捕获的警告数量为 0
            self.assertEqual(len(w), 0)

    def test_nonzero(self):
        # 定义一个辅助函数 helper，参数为 dtype
        def helper(dtype):
            # 设定设备为 "mps"
            device = "mps"
            # 定义不同形状的张量
            shapes = [
                torch.Size((12,)),
                torch.Size((12, 1)),
                torch.Size((1, 12)),
                torch.Size((6, 2)),
                torch.Size((3, 2, 2)),
                torch.Size((5, 5, 5)),
            ]

            # 定义生成非平凡输入的函数 gen_nontrivial_input
            def gen_nontrivial_input(shape, dtype, device):
                # 如果 dtype 不是 torch.bfloat16，则使用 torch.randint() 生成随机整数张量
                if dtype != torch.bfloat16:
                    return torch.randint(2, shape, device=device, dtype=dtype)
                else:
                    # 对于 torch.bfloat16，使用 torch.float 生成随机浮点数张量，并转换为 bfloat16
                    return torch.randint(2, shape, device=device, dtype=torch.float).to(dtype)

            # 遍历所有定义的形状
            for shape in shapes:
                # 生成具体的张量 tensor
                tensor = gen_nontrivial_input(shape, dtype, device)
                # 调用 torch.nonzero() 函数，返回非零元素的索引，保存到 dst1
                dst1 = torch.nonzero(tensor, as_tuple=False)
                # 调用张量的 nonzero() 方法，返回非零元素的索引，保存到 dst2
                dst2 = tensor.nonzero(as_tuple=False)
                # 创建一个空张量 dst3，类型为 torch.long，设备为 device
                dst3 = torch.empty([], dtype=torch.long, device=device)
                # 调整 dst3 的大小为 0
                dst3 = dst3.resize_(0)
                # 使用 torch.nonzero() 函数，将非零元素的索引存储到 dst3 中
                torch.nonzero(tensor, out=dst3)
                # 将张量 tensor 转换为 NumPy 数组，如果 dtype 不是 torch.bfloat16，则直接转换为 CPU 上的 NumPy 数组
                np_array = tensor.cpu().numpy() if dtype != torch.bfloat16 else tensor.float().cpu().numpy()
                # 使用 NumPy 的 nonzero() 函数找到非零元素的索引，并转换为 PyTorch 张量，然后转置
                np_result = torch.from_numpy(np.stack(np_array.nonzero())).t()
                # 断言 dst1 的结果与 np_result 相等，atol 和 rtol 设置为 0
                self.assertEqual(dst1.cpu(), np_result, atol=0, rtol=0)
                # 断言 dst2 的结果与 np_result 相等，atol 和 rtol 设置为 0
                self.assertEqual(dst2.cpu(), np_result, atol=0, rtol=0)
                # 断言 dst3 的结果与 np_result 相等，atol 和 rtol 设置为 0
                self.assertEqual(dst3.cpu(), np_result, atol=0, rtol=0)
                # 调用 torch.nonzero() 函数，返回非零元素的索引，并保存为元组 tup1
                tup1 = torch.nonzero(tensor, as_tuple=True)
                # 调用张量的 nonzero() 方法，返回非零元素的索引，并保存为元组 tup2
                tup2 = tensor.nonzero(as_tuple=True)
                # 将 tup1 转换为 PyTorch 张量，并转置，然后转移到 CPU 上
                tup1 = torch.stack(tup1).t().cpu()
                # 将 tup2 转换为 PyTorch 张量，并转置，然后转移到 CPU 上
                tup2 = torch.stack(tup2).t().cpu()
                # 断言 tup1 的结果与 np_result 相等，atol 和 rtol 设置为 0
                self.assertEqual(tup1, np_result, atol=0, rtol=0)
                # 断言 tup2 的结果与 np_result 相等，atol 和 rtol 设置为 0
                self.assertEqual(tup2, np_result, atol=0, rtol=0)
        
        # 对 supported_dtypes 列表中的每种数据类型调用 helper 函数
        [helper(dtype) for dtype in self.supported_dtypes]
    # 测试 torch.nonzero() 在 as_tuple=True 时的异常处理
    def test_nonzero_astuple_out(self):
        device = "mps"
        # 生成一个在指定设备上的随机张量
        t = torch.randn((3, 3, 3), device=device)
        # 创建一个空的长整型张量 out，并将其大小调整为 0
        out = torch.empty([], dtype=torch.long, device=device)
        out = out.resize_(0)

        # 断言在运行时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.nonzero(t, as_tuple=True, out=out)

        # 使用 torch.nonzero() 分别以 as_tuple=False 和给定的 out 参数进行计算，并断言两者结果相等
        self.assertEqual(torch.nonzero(t, as_tuple=False, out=out), torch.nonzero(t, out=out))

        # 验证 JIT 脚本无法处理 as_tuple 关键字的问题
        # 参见 Issue https://github.com/pytorch/pytorch/issues/45499.
        def _foo(t):
            # 使用 as_tuple=True 和 as_tuple=False 两种方式调用 torch.nonzero()
            tuple_result = torch.nonzero(t, as_tuple=True)
            nontuple_result = torch.nonzero(t, as_tuple=False)
            # 创建一个与 nontuple_result 相同大小的空张量 out，并使用 torch.nonzero() 写入结果
            out = torch.empty_like(nontuple_result)
            torch.nonzero(t, as_tuple=False, out=out)
            return tuple_result, nontuple_result, out

        # 断言在运行时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            scripted_foo = torch.jit.script(_foo)

        # 验证 JIT 追踪功能正常工作
        traced_foo = torch.jit.trace(_foo, t)
        traced_tuple, traced_nontuple, traced_out = traced_foo(t)
        # 使用 torch.nonzero() 计算期望的结果，并断言与追踪后的结果相等
        expected_tuple = torch.nonzero(t, as_tuple=True)
        expected_nontuple = torch.nonzero(t)
        self.assertEqual(traced_tuple, expected_tuple)
        self.assertEqual(traced_nontuple, expected_nontuple)
        self.assertEqual(traced_out, expected_nontuple)

    # 测试非连续张量的 nonzero() 函数
    def test_nonzero_discontiguous(self):
        device = "mps"
        shape = (4, 4)
        # 在指定设备上生成一个随机整型张量
        tensor = torch.randint(2, shape, device=device)
        # 创建一个非连续的张量 tensor_nc，并将其内容复制为 tensor 的每一行
        tensor_nc = torch.empty(shape[0], shape[1] * 2, device=device)[:, ::2].copy_(tensor)
        # 使用 torch.nonzero() 分别计算 tensor 和 tensor_nc 的非零元素索引，并断言它们相等
        dst1 = tensor.nonzero(as_tuple=False)
        dst2 = tensor_nc.nonzero(as_tuple=False)
        self.assertEqual(dst1, dst2, atol=0, rtol=0)
        
        # 创建一个与 dst1 相同大小的空张量 dst3，并断言其数据指针与原始 dst1 相同
        dst3 = torch.empty_like(dst1)
        data_ptr = dst3.data_ptr()
        torch.nonzero(tensor, out=dst3)
        self.assertEqual(data_ptr, dst3.data_ptr())
        self.assertEqual(dst1, dst3, atol=0, rtol=0)
        
        # 创建一个非连续的输出张量 dst4，并断言其数据指针和步幅与原始 dst1 相同
        dst4 = torch.empty(dst1.size(0), dst1.size(1) * 2, dtype=torch.long, device=device)[:, ::2]
        data_ptr = dst4.data_ptr()
        strides = dst4.stride()
        torch.nonzero(tensor, out=dst4)
        self.assertEqual(data_ptr, dst4.data_ptr())
        self.assertEqual(dst1, dst4, atol=0, rtol=0)
        self.assertEqual(strides, dst4.stride())

    # 测试非可微张量的 nonzero() 函数
    def test_nonzero_non_diff(self):
        device = "mps"
        # 生成一个具有梯度要求的随机张量 x
        x = torch.randn(10, requires_grad=True)
        # 计算张量 x 的非零元素索引 nz，并断言 nz 不需要梯度
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)

    # 测试多线程环境下的 nonzero() 函数
    def test_nonzero_multi_threading(self):
        # 测试 MPS 下并发调用 nonzero() 不会导致崩溃
        # 参见 https://github.com/pytorch/pytorch/issues/100285
        x = torch.rand(3, 3, device="mps")
        # 创建两个线程，分别调用 torch.nonzero() 函数
        t1 = threading.Thread(target=torch.nonzero, args=(x,))
        t2 = threading.Thread(target=torch.nonzero, args=(x,))
        t1.start()
        t2.start()
    def test_masked_select(self):
        # 创建一个大小为 3x4 的张量，其中元素服从标准正态分布
        x = torch.randn(3, 4)
        # 将张量 x 转换为 "mps" 格式
        x_mps = x.to("mps")
        # 创建一个布尔掩码，标识 x 中大于等于 0.5 的元素位置
        mask = x.ge(0.5)
        # 在 x_mps 中创建与 mask 相同的布尔掩码
        mask_mps = x_mps.ge(0.5)

        # 使用 mask 获取 x 中符合条件的元素，返回一个一维张量
        res = torch.masked_select(x, mask)
        # 使用 mask_mps 获取 x_mps 中符合条件的元素，返回一个一维张量
        res_mps = torch.masked_select(x_mps, mask_mps)

        # 断言两个结果张量是否相等
        self.assertEqual(res, res_mps)

    # examples from https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm
    def test_indexing_get(self):
        # 定义一个辅助函数，参数为数据类型 dtype
        def helper(dtype):
            # 创建一个二维张量 x_cpu，数据类型为 dtype
            x_cpu = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype)
            # 将 x_cpu 从计算图中分离并克隆，然后转换为 "mps" 格式
            x_mps = x_cpu.detach().clone().to("mps")

            # 使用高级索引获取特定位置的元素，并创建一个新的张量 y_cpu
            y_cpu = x_cpu[[0, 1, 2], [0, 1, 0]]
            # 在 x_mps 上执行相同的高级索引操作，并创建新的张量 y_mps
            y_mps = x_mps[[0, 1, 2], [0, 1, 0]]
            # 断言两个张量 y_cpu 和 y_mps 是否相等，输出数据类型信息
            self.assertEqual(y_cpu, y_mps, str(dtype))
        # 对 self.supported_dtypes 中的每种数据类型，调用 helper 函数
        [helper(dtype) for dtype in self.supported_dtypes]

    def test_indexing_select_corners(self):
        # 定义一个辅助函数，参数为数据类型 dtype
        def helper(dtype):
            # 创建一个二维张量 x_cpu，数据类型为 dtype
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            # 将 x_cpu 从计算图中分离并克隆，然后转换为 "mps" 格式
            x_mps = x_cpu.detach().clone().to("mps")

            # 创建一个包含行索引的张量 rows_cpu
            rows_cpu = torch.tensor([[0, 0], [3, 3]])
            # 将 rows_cpu 从计算图中分离并克隆，然后转换为 "mps" 格式
            rows_mps = rows_cpu.detach().clone().to("mps")

            # 创建一个包含列索引的张量 cols_cpu
            cols_cpu = torch.tensor([[0, 2], [0, 2]])
            # 将 cols_cpu 从计算图中分离并克隆，然后转换为 "mps" 格式
            cols_mps = cols_cpu.detach().clone().to("mps")

            # 使用多重索引获取特定位置的元素，并创建一个新的张量 res_cpu
            res_cpu = x_cpu[rows_cpu, cols_cpu]
            # 在 x_mps 上执行相同的多重索引操作，并创建新的张量 res_mps
            res_mps = x_mps[rows_mps, cols_mps]

            # 断言两个张量 res_cpu 和 res_mps 是否相等，输出数据类型信息
            self.assertEqual(res_cpu, res_mps, str(dtype))
        # 对 self.supported_dtypes 中的每种数据类型，调用 helper 函数
        [helper(dtype) for dtype in self.supported_dtypes]

    # FIXME: uint8 fails for this testcase, needs further debugging
    def test_slicing_using_advanced_index_for_column(self):
        # 定义一个辅助函数，参数为数据类型 dtype
        def helper(dtype):
            # 创建一个二维张量 x_cpu，数据类型为 dtype
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            # 将 x_cpu 从计算图中分离并克隆，然后转换为 "mps" 格式
            x_mps = x_cpu.detach().clone().to("mps")

            # 使用切片获取特定行列范围的子张量，并创建一个新的张量 z_cpu
            z_cpu = x_cpu[1:4, 1:3]
            # 在 x_mps 上执行相同的切片操作，并创建新的张量 z_mps
            z_mps = x_mps[1:4, 1:3]
            # 断言两个张量 z_cpu 和 z_mps 是否相等，输出数据类型信息
            self.assertEqual(z_cpu, z_mps, str(dtype))

            # 使用高级索引获取特定行和列的元素，并创建一个新的张量 y_cpu
            y_cpu = x_cpu[1:4, [1, 2]]
            # 在 x_mps 上执行相同的高级索引操作，并创建新的张量 y_mps
            y_mps = x_mps[1:4, [1, 2]]
            # 断言两个张量 y_cpu 和 y_mps 是否相等，输出数据类型信息
            self.assertEqual(y_cpu, y_mps, str(dtype))
        # FIXME: 一旦修复了 uint8，使用 supported_dtypes 替代此处的列表
        [helper(dtype) for dtype in [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16]]

    def test_boolean_array_indexing(self):
        # 定义一个辅助函数，参数为数据类型 dtype
        def helper(dtype):
            # 创建一个二维张量 x_cpu，数据类型为 dtype
            x_cpu = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=dtype)
            # 将 x_cpu 从计算图中分离并克隆，然后转换为 "mps" 格式
            x_mps = x_cpu.detach().clone().to("mps")

            # 使用布尔数组索引获取符合条件的元素，并创建一个新的张量 res_cpu
            res_cpu = x_cpu[x_cpu > 5]
            # 在 x_mps 上执行相同的布尔数组索引操作，并创建新的张量 res_mps
            res_mps = x_mps[x_mps > 5]

            # 断言两个张量 res_cpu 和 res_mps 是否相等，输出数据类型信息
            self.assertEqual(res_cpu, res_mps, str(dtype))
        # 对于 self.supported_dtypes 中的每种数据类型，调用 helper 函数
        for dtype in self.supported_dtypes:
            # 如果产品版本小于 13.0 并且数据类型为 torch.uint8，则跳过此次循环
            if product_version < 13.0 and dtype == torch.uint8:
                continue
            helper(dtype)
    def test_advanced_indexing_3D_get(self):
        # 定义内部辅助函数helper，用于测试索引操作在CPU和"mps"设备上的行为
        def helper(x_cpu):
            # 将x_cpu从计算图中分离并克隆到"mps"设备，然后进行断言比较
            x_mps = x_cpu.detach().clone().to("mps")
            # 断言x_cpu的复合索引[[1, 2], 3, :]与x_mps的相等
            self.assertEqual(x_cpu[[1, 2], 3, :], x_mps[[1, 2], 3, :])
            # 断言x_cpu的复合索引[[0, 2], :, :]与x_mps的相等
            self.assertEqual(x_cpu[[0, 2], :, :], x_mps[[0, 2], :, :])
            # 断言x_cpu的复合索引[:, [1, 0], [1]]与x_mps的相等
            self.assertEqual(x_cpu[:, [1, 0], [1]], x_mps[:, [1, 0], [1]])

        # 创建一个3x4x4的Tensor x_cpu，包含三个3x4的二维矩阵，存储在CPU上，数据类型为torch.float32
        x_cpu = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                               [0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2],
                               [1.3, 1.4, 1.5, 1.6]],

                              [[2.0, 2.1, 2.2, 2.3],
                               [2.4, 2.5, 2.6, 2.7],
                               [2.8, 2.9, 3.0, 3.1],
                               [3.2, 3.3, 3.4, 3.5]],

                              [[4.0, 4.1, 4.2, 4.3],
                               [4.4, 4.5, 4.6, 4.7],
                               [4.8, 4.9, 5.0, 5.1],
                               [5.1, 5.2, 5.3, 5.4]]], device="cpu", dtype=torch.float32)
        # 调用helper函数，传入x_cpu作为参数
        helper(x_cpu)
        # 遍历self.supported_np_dtypes中支持的所有数据类型
        for idx in range(len(self.supported_np_dtypes)):
            # 为当前数据类型生成一个3x4x4的随机numpy数组，然后转换为torch.tensor，
            # 存储在CPU上，数据类型为self.supported_dtypes[idx]
            input_t = np.random.random_sample(size=[3, 4, 4]).astype(self.supported_np_dtypes[idx])
            inputCPU = torch.tensor(input_t, device='cpu', dtype=self.supported_dtypes[idx])

            # 调用helper函数，传入inputCPU作为参数
            helper(inputCPU)
    def test_advanced_indexing_3D_put(self):
        def helper(x_cpu):
            # 获取输入张量的数据类型
            dtype = x_cpu.dtype
            # 将 CPU 上的张量转换为 MPS (Memory Persistence Storage)
            x_mps = x_cpu.detach().clone().to("mps")

            # 创建一个 CPU 上的输出张量
            out_tensor_cpu = torch.tensor([88, 99], dtype=dtype, device="cpu")
            # 获取 CPU 输出张量的视图（除第一个元素外的切片）
            out_tensor_cpu_view = out_tensor_cpu[1:]

            # 创建一个 MPS 上的输出张量
            out_tensor_mps = torch.tensor([88, 99], dtype=dtype, device="mps")
            # 获取 MPS 输出张量的视图（除第一个元素外的切片）
            out_tensor_mps_view = out_tensor_mps[1:]

            # 使用视图替换 CPU 输入张量的指定索引处的值
            x_cpu[[1, 2], 3, :] = out_tensor_cpu_view
            # 使用视图替换 MPS 输入张量的指定索引处的值
            x_mps[[1, 2], 3, :] = out_tensor_mps_view
            # 断言两个张量是否相等
            self.assertEqual(x_cpu, x_mps)

            # 使用视图替换 CPU 输入张量的指定索引处的值
            x_cpu[[0, 2], :, :] = out_tensor_cpu_view
            # 使用视图替换 MPS 输入张量的指定索引处的值
            x_mps[[0, 2], :, :] = out_tensor_mps_view
            # 断言两个张量是否相等
            self.assertEqual(x_cpu, x_mps)

            # 使用视图替换 CPU 输入张量的指定索引处的值
            x_cpu[:, [1, 0], [1]] = out_tensor_cpu_view
            # 使用视图替换 MPS 输入张量的指定索引处的值
            x_mps[:, [1, 0], [1]] = out_tensor_mps_view
            # 断言两个张量是否相等
            self.assertEqual(x_cpu, x_mps)

        # 创建一个 3x4x4 的 CPU 上的浮点型张量
        x_cpu = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                               [0.5, 0.6, 0.7, 0.8],
                               [0.9, 1.0, 1.1, 1.2],
                               [1.3, 1.4, 1.5, 1.6]],

                              [[2.0, 2.1, 2.2, 2.3],
                               [2.4, 2.5, 2.6, 2.7],
                               [2.8, 2.9, 3.0, 3.1],
                               [3.2, 3.3, 3.4, 3.5]],

                              [[4.0, 4.1, 4.2, 4.3],
                               [4.4, 4.5, 4.6, 4.7],
                               [4.8, 4.9, 5.0, 5.1],
                               [5.1, 5.2, 5.3, 5.4]]], device="cpu", dtype=torch.float32)
        # 调用 helper 函数，传入 CPU 上的张量
        helper(x_cpu)

        # 遍历支持的 Numpy 数据类型列表
        for idx in range(len(self.supported_np_dtypes)):
            # torch.randn / torch.rand 不适用于所有数据类型
            # 生成所有 Numpy 数据类型的输入数据，然后转换为 torch 张量
            input_t = np.random.random_sample(size=[3, 4, 4]).astype(self.supported_np_dtypes[idx])
            inputCPU = torch.tensor(input_t, device='cpu', dtype=self.supported_dtypes[idx])

            # 调用 helper 函数，传入 CPU 上的输入张量
            helper(inputCPU)

    def test_index_put_with_view_indices(self):
        def helper(dtype):
            # 创建一个全零的 CPU 张量
            target_cpu = torch.zeros([5, 3], device="cpu", dtype=dtype)
            # 创建一个全零的 MPS 张量
            target_mps = torch.zeros([5, 3], device="mps", dtype=dtype)

            # 创建索引张量，用于 CPU
            indices_cpu = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="cpu")
            # 创建索引张量，用于 MPS
            indices_mps = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="mps")

            # 创建数值张量，用于 CPU
            value_cpu = torch.ones(indices_cpu.shape[0], device="cpu", dtype=dtype)
            # 创建数值张量，用于 MPS
            value_mps = torch.ones(indices_mps.shape[0], device="mps", dtype=dtype)

            # 使用索引操作在 CPU 目标张量上放置数值张量的值（累加）
            target_cpu.index_put_(tuple(indices_cpu.t()), value_cpu, accumulate=True)
            # 使用索引操作在 MPS 目标张量上放置数值张量的值（累加）
            target_mps.index_put_(tuple(indices_mps.t()), value_mps, accumulate=True)

            # 断言两个目标张量是否相等
            self.assertEqual(target_cpu, target_mps)

        # 对于每种数据类型，调用 helper 函数
        [helper(dtype) for dtype in [torch.int32, torch.float]]

    # tests from 'test_indexing.py'
    # 测试高级索引功能，使用指定的设备（默认为"mps"）
    def test_advancedindex_big(self, device="mps"):
        # 创建一个参考张量，包含从0到123343的整数，数据类型为torch.int，使用指定设备
        reference = torch.arange(0, 123344, dtype=torch.int, device=device)

        # 使用高级索引方法，选择指定索引位置的元素，并进行断言比较
        self.assertEqual(reference[[0, 123, 44488, 68807, 123343], ],
                         torch.tensor([0, 123, 44488, 68807, 123343], dtype=torch.int))

    # 测试将张量中的某一项设置为标量张量的情况，使用指定的设备（默认为"mps"）
    def test_set_item_to_scalar_tensor(self, device="mps"):
        # 随机生成矩阵大小的整数m和n
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        # 创建一个随机张量z，形状为[m, n]，数据在指定设备上
        z = torch.randn([m, n], device=device)
        # 将标量a封装为张量w，并标记为需要梯度计算
        a = 1.0
        w = torch.tensor(a, requires_grad=True, device=device)
        # 将张量z的所有行的第一列元素设置为张量w
        z[:, 0] = w
        # 对张量z的所有元素求和并进行反向传播
        z.sum().backward()
        # 断言标量张量w的梯度值为m乘以a
        self.assertEqual(w.grad, m * a)

    # 测试索引为单个整数的情况，使用指定的设备（默认为"mps"）
    def test_single_int(self, device="mps"):
        # 创建一个随机张量v，形状为[5, 7, 3]，数据在指定设备上
        v = torch.randn(5, 7, 3, device=device)
        # 断言选择索引为4的元素后的形状为(7, 3)
        self.assertEqual(v[4].shape, (7, 3))

    # 测试索引为多个整数的情况，使用指定的设备（默认为"mps"）
    def test_multiple_int(self, device="mps"):
        # 创建一个随机张量v，形状为[5, 7, 3]，数据在指定设备上
        v = torch.randn(5, 7, 3, device=device)
        # 断言选择索引为4的元素后的形状为(7, 3)
        self.assertEqual(v[4].shape, (7, 3))
        # 断言选择索引为4，并选择所有行和第1列的元素后的形状为(7,)
        self.assertEqual(v[4, :, 1].shape, (7,))

    # 测试使用None进行索引操作，使用指定的设备（默认为"mps"）
    def test_none(self, device="mps"):
        # 创建一个随机张量v，形状为[5, 7, 3]，数据在指定设备上
        v = torch.randn(5, 7, 3, device=device)
        # 断言选择整个张量后的形状为(1, 5, 7, 3)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        # 断言在所有维度前面添加一个维度后的形状为(5, 1, 7, 3)
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        # 断言在所有维度前面添加两个维度后的形状为(5, 1, 1, 7, 3)
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        # 断言在最后一个维度后面添加一个维度后的形状为(5, 7, 3, 1)
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

    # 测试使用步长进行索引操作，使用指定的设备（默认为"mps"）
    def test_step(self, device="mps"):
        # 创建一个从0到9的整数张量v，数据在指定设备上
        v = torch.arange(10, device=device)
        # 断言取步长为1时，结果应该与张量v相同
        self.assertEqual(v[::1], v)
        # 断言取步长为2时，结果转换为列表后应该为[0, 2, 4, 6, 8]
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        # 断言取步长为3时，结果转换为列表后应该为[0, 3, 6, 9]
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        # 断言取步长为11时，结果转换为列表后应该为[0]
        self.assertEqual(v[::11].tolist(), [0])
        # 断言在索引范围为1到6且步长为2时，结果转换为列表后应该为[1, 3, 5]
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])

    # 测试使用步长进行索引赋值操作，使用指定的设备（默认为"mps"）
    def test_step_assignment(self, device="mps"):
        # 创建一个全零的4x4张量v，数据在指定设备上
        v = torch.zeros(4, 4, device=device)
        # 将第一行、步长为2的元素赋值为[3.0, 4.0]
        v[0, 1::2] = torch.tensor([3., 4.], device=device)
        # 断言第一行的元素列表为[0, 3, 0, 4]
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        # 断言除了第一行之外的所有元素的和为0
        self.assertEqual(v[1:].sum(), 0)

    # 测试使用布尔索引进行选择操作，使用指定的设备（默认为"mps"）
    def test_bool_indices(self, device="mps"):
        # 创建一个随机张量v，形状为[5, 7, 3]，数据在指定设备上
        v = torch.randn(5, 7, 3, device=device)
        # 创建布尔索引张量boolIndices，选择指定位置的元素，并进行断言比较形状
        boolIndices = torch.tensor([True, False, True, True, False], dtype=torch.bool, device=device)
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        # 断言使用布尔索引选取的结果应与手动堆栈的结果相同
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        # 创建布尔张量v和boolIndices，以及uint8索引张量uint8Indices，进行多个断言
        v = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        boolIndices = torch.tensor([True, False, False], dtype=torch.bool, device=device)
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8, device=device)
        # 使用警告记录来捕获异常，进行多个断言比较
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
            self.assertEqual(v[boolIndices], v[uint8Indices])
            self.assertEqual(v[boolIndices], torch.tensor([True], dtype=torch.bool, device=device))
            self.assertEqual(len(w), 2)

    # 如果产品版本小于13.0，则跳过此测试（适用于macOS 12及以下版本）
    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    # 定义一个测试函数，用于测试布尔类型索引在累积情况下的操作
    def test_bool_indices_accumulate(self, device="mps"):
        # 创建一个大小为10的零张量作为掩码，数据类型为torch.uint8，存储于指定设备上
        mask = torch.zeros(size=(10, ), dtype=torch.uint8, device=device)
        # 将掩码转换为布尔类型张量，所有元素均为False
        mask = mask > 0
        # 创建一个大小为(10, 10)的张量，并将所有元素初始化为1，存储于指定设备上
        y = torch.ones(size=(10, 10), device=device)
        # 使用索引放置操作，将掩码对应位置的y值累积写入y张量中
        y.index_put_((mask, ), y[mask], accumulate=True)
        # 断言y张量的结果与全1张量相等
        self.assertEqual(y, torch.ones(size=(10, 10), device=device))

    # 定义一个测试函数，用于测试多个布尔类型索引的情况
    def test_multiple_bool_indices(self, device="mps"):
        # 创建一个大小为(5, 7, 3)的张量，并随机初始化，存储于指定设备上
        v = torch.randn(5, 7, 3, device=device)
        # 创建两个布尔类型的掩码张量
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool, device=device)
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        # 使用布尔类型索引，获取符合条件的子张量，并断言其形状为(3, 7)
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    # 定义一个测试函数，用于测试字节类型掩码的情况
    def test_byte_mask(self, device="mps"):
        # 创建一个大小为(5, 7, 3)的张量，并随机初始化，存储于指定设备上
        v = torch.randn(5, 7, 3, device=device)
        # 创建一个字节类型的掩码张量
        mask = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        # 使用字节类型掩码进行索引操作，并断言结果张量的形状为(3, 7, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[mask].shape, (3, 7, 3))
            # 断言索引后的张量与手动索引到的结果张量相等
            self.assertEqual(v[mask], torch.stack([v[0], v[2], v[3]]))
            # 断言警告信息的数量为2
            self.assertEqual(len(w), 2)

        # 创建一个大小为1的张量，并初始化为1，存储于指定设备上
        v = torch.tensor([1.], device=device)
        # 断言索引到的结果张量为一个空张量
        self.assertEqual(v[v == 0], torch.tensor([], device=device))

    # 定义一个测试函数，用于测试字节类型掩码在累积情况下的操作
    def test_byte_mask_accumulate(self, device="mps"):
        # 创建一个大小为10的零张量作为掩码，数据类型为torch.uint8，存储于指定设备上
        mask = torch.zeros(size=(10, ), dtype=torch.uint8, device=device)
        # 创建一个大小为(10, 10)的张量，并将所有元素初始化为1，存储于指定设备上
        y = torch.ones(size=(10, 10), device=device)
        # 使用索引放置操作，将掩码对应位置的y值累积写入y张量中
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y.index_put_((mask, ), y[mask], accumulate=True)
            # 断言y张量的结果与全1张量相等
            self.assertEqual(y, torch.ones(size=(10, 10), device=device))
            # 断言警告信息的数量为2
            self.assertEqual(len(w), 2)
    # 定义测试方法，用于测试在指定设备上执行索引赋值操作时的累加行为
    def test_index_put_accumulate_expanded_values(self, device="mps"):
        # 创建一个大小为 (5, 2) 的全零张量
        t = torch.zeros((5, 2))
        # 将张量移动到指定设备上
        t_dev = t.to(device)
        # 定义多个索引张量
        indices = [
            torch.tensor([0, 1, 2, 3]),  # 第一个索引张量，包含多个元素
            torch.tensor([1, ]),         # 第二个索引张量，只包含一个元素
        ]
        # 将所有索引张量移动到指定设备上
        indices_dev = [i.to(device) for i in indices]
        # 定义不同形状的数值张量
        values0d = torch.tensor(1.0)     # 标量数值张量
        values1d = torch.tensor([1.0, ])  # 一维数值张量

        # 在指定设备上对 t_dev 进行索引赋值操作，累加数值
        out_mps = t_dev.index_put_(indices_dev, values0d.to(device), accumulate=True)
        # 在 CPU 上对 t 进行索引赋值操作，累加数值
        out_cpu = t.index_put_(indices, values0d, accumulate=True)
        # 断言两个操作结果相等
        self.assertEqual(out_mps.cpu(), out_cpu)

        # 在指定设备上对 t_dev 进行索引赋值操作，累加一维数值张量
        out_mps = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        # 在 CPU 上对 t 进行索引赋值操作，累加一维数值张量
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        # 断言两个操作结果相等
        self.assertEqual(out_mps.cpu(), out_cpu)

        # 重新定义一个大小为 (4, 3, 2) 的全零张量
        t = torch.zeros(4, 3, 2)
        # 将张量移动到指定设备上
        t_dev = t.to(device)

        # 定义复杂形状的索引张量
        indices = [
            torch.tensor([0, ]),           # 第一个索引张量，只包含一个元素
            torch.arange(3)[:, None],     # 第二个索引张量，列向量，长度为 3
            torch.arange(2)[None, :],     # 第三个索引张量，行向量，长度为 2
        ]
        # 将所有索引张量移动到指定设备上
        indices_dev = [i.to(device) for i in indices]
        # 定义不同形状的数值张量
        values1d = torch.tensor([-1.0, -2.0])   # 一维数值张量
        values2d = torch.tensor([[-1.0, -2.0], ])  # 二维数值张量

        # 在指定设备上对 t_dev 进行索引赋值操作，累加一维数值张量
        out_mps = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        # 在 CPU 上对 t 进行索引赋值操作，累加一维数值张量
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        # 断言两个操作结果相等
        self.assertEqual(out_mps.cpu(), out_cpu)

        # 在指定设备上对 t_dev 进行索引赋值操作，累加二维数值张量
        out_mps = t_dev.index_put_(indices_dev, values2d.to(device), accumulate=True)
        # 在 CPU 上对 t 进行索引赋值操作，累加二维数值张量
        out_cpu = t.index_put_(indices, values2d, accumulate=True)
        # 断言两个操作结果相等
        self.assertEqual(out_mps.cpu(), out_cpu)

    # 定义测试方法，用于测试非连续索引时的索引赋值累加操作
    def test_index_put_accumulate_non_contiguous(self, device="mps"):
        # 创建一个大小为 (5, 2, 2) 的全零张量
        t = torch.zeros((5, 2, 2))
        # 将张量移动到指定设备上
        t_dev = t.to(device)
        # 获取 t_dev 的一个切片，使得结果张量 t1 非连续
        t1 = t_dev[:, 0, :]
        # 获取 t 的一个切片，使得结果张量 t2 非连续
        t2 = t[:, 0, :]
        # 断言 t1 和 t2 都是非连续的张量
        self.assertFalse(t1.is_contiguous())
        self.assertFalse(t2.is_contiguous())

        # 定义一个包含单个索引张量的列表
        indices = [torch.tensor([0, 1]), ]
        # 将索引张量移动到指定设备上
        indices_dev = [i.to(device) for i in indices]
        # 定义一个随机数值张量
        value = torch.randn(2, 2)

        # 在指定设备上对 t1 进行索引赋值操作，累加数值张量
        out_mps = t1.index_put_(indices_dev, value.to(device), accumulate=True)
        # 在 CPU 上对 t2 进行索引赋值操作，累加数值张量
        out_cpu = t2.index_put_(indices, value, accumulate=True)
        # 断言 t1 和 t2 仍然是非连续的张量
        self.assertFalse(t1.is_contiguous())
        self.assertFalse(t2.is_contiguous())

        # 断言两个操作结果相等
        self.assertEqual(out_mps.cpu(), out_cpu)
    def test_index_put_accumulate_with_optional_tensors(self, device="mps"):
        # TODO: 替换为更好的解决方案。
        # 目前，这里使用 torchscript 将 None 放入索引中。
        # 在 C++ 中，它将索引作为包含两个可选张量的列表返回：第一个为 null，第二个为有效张量。
        
        @torch.jit.script
        def func(x, i, v):
            # 创建索引列表，第一个元素为 None，第二个为传入的索引 i
            idx = [None, i]
            # 使用索引在张量 x 上进行累积赋值操作
            x.index_put_(idx, v, accumulate=True)
            return x

        n = 4
        # 创建一个张量 t，包含 n*2 个元素，按顺序排列，数据类型为 float32
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        # 将张量 t 移动到指定设备上
        t_dev = t.to(device)
        # 创建一个包含索引值的张量
        indices = torch.tensor([1, 0])
        # 将索引张量移动到指定设备上
        indices_dev = indices.to(device)
        # 创建一个标量张量，值为 10.0
        value0d = torch.tensor(10.0)
        # 创建一个包含两个元素的张量，数据类型为 float32
        value1d = torch.tensor([1.0, 2.0])

        # 调用 func 函数，对 t_dev、indices_dev、value0d 进行操作，并将结果移回 CPU
        out_mps = func(t_dev, indices_dev, value0d.to("mps"))
        # 调用 func 函数，对 t、indices、value0d 进行操作
        out_cpu = func(t, indices, value0d)
        # 断言两个输出张量在 CPU 上的值相等
        self.assertEqual(out_mps.cpu(), out_cpu)

        # 调用 func 函数，对 t_dev、indices_dev、value1d 进行操作，并将结果移回 CPU
        out_mps = func(t_dev, indices_dev, value1d.to("mps"))
        # 调用 func 函数，对 t、indices、value1d 进行操作
        out_cpu = func(t, indices, value1d)
        # 断言两个输出张量在 CPU 上的值相等
        self.assertEqual(out_mps.cpu(), out_cpu)

    def test_index_put_accumulate_duplicate_indices(self, device="mps"):
        # 循环迭代，生成长度从 1 到 127 的索引张量
        for i in range(1, 128):
            # 生成随机游走的增量张量 delta，数据类型为 float32，设备为指定设备
            delta = torch.empty(i, dtype=torch.float32, device=device).uniform_(-1, 1)

            # 计算累积和并转换为 long 类型的索引张量，设备为 "mps"
            indices = delta.cumsum(0).long().to("mps")

            # 计算输入张量的绝对值的最大值，并增加 1，转换为指定设备上的张量
            input = torch.randn(indices.cpu().abs().max().to("mps") + 1, device=device)
            # 创建与 indices 长度相同的随机张量 values，设备为指定设备
            values = torch.randn(indices.size(0), device=device)
            # 使用索引在输入张量上进行累积赋值操作
            output = input.index_put((indices,), values, accumulate=True)

            # 将输入张量转换为列表形式
            input_list = input.tolist()
            # 将索引张量、数值张量转换为列表形式
            indices_list = indices.tolist()
            values_list = values.tolist()
            # 根据索引列表，逐个更新 input_list 中的值
            for i, v in zip(indices_list, values_list):
                input_list[i] += v

            # 断言输出张量与经过更新的 input_list 相等
            self.assertEqual(output, input_list)
    # 定义测试函数 `test_index_put_deterministic`，用于测试索引赋值的行为是否确定性
    def test_index_put_deterministic(self, device="mps"):
        # 定义内部辅助函数 `helper`，用于执行具体的索引赋值测试
        def helper(dtype, accumulate, deterministic, num_tests=128):
            # 预期的累加结果
            acc_expected = torch.tensor([233, 187, 360], device=device, dtype=dtype)
            # 预期的非累加结果
            non_acc_expected = torch.tensor([38, 37, 39], device=device, dtype=dtype)
            # 索引序列
            t_idx = torch.tensor(
                [0, 0, 0, 0, 2, 2, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2,
                 0, 0, 2, 1, 2, 1, 0, 0, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 0, 2]
            )
            # 执行多次测试
            for _ in range(num_tests):
                try:
                    # 设置是否使用确定性算法
                    torch.use_deterministic_algorithms(deterministic)
                    # 创建全零张量 `t`
                    t = torch.zeros(3, dtype=dtype, device=device)
                    # 执行索引赋值操作
                    t.index_put_((t_idx,), torch.arange(len(t_idx), device=device, dtype=dtype), accumulate=accumulate)
                    # 检查累加或非累加结果是否符合预期
                    if accumulate:
                        self.assertEqual(t, acc_expected)
                    else:
                        self.assertEqual(t, non_acc_expected)
                finally:
                    # 恢复默认的非确定性算法设置
                    torch.use_deterministic_algorithms(False)

        # 组合不同的累加和确定性设置进行测试
        for accumulate, deterministic in product((False, True), (False, True)):
            # 根据累加设置选择数据类型
            dtype = torch.float if accumulate else torch.long
            # 如果既不累加也不确定性，则测试应该引发断言错误
            if not accumulate and not deterministic:
                with self.assertRaisesRegex(AssertionError, "Tensor-likes are not equal!"):
                    helper(dtype, accumulate, deterministic)
            else:
                helper(dtype, accumulate, deterministic)

    # 定义测试函数 `test_multiple_byte_mask`，测试多个字节掩码的行为
    def test_multiple_byte_mask(self, device="mps"):
        # 创建随机张量 `v`
        v = torch.randn(5, 7, 3, device=device)
        # 创建第一个字节掩码 `mask1`
        mask1 = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        # 创建第二个字节掩码 `mask2`
        mask2 = torch.ByteTensor([1, 1, 1]).to(device)
        # 使用警告捕获器来检查警告消息
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 检查使用多个字节掩码进行索引的结果形状是否正确
            self.assertEqual(v[mask1, :, mask2].shape, (3, 7))
            # 检查是否捕获到了两条警告消息
            self.assertEqual(len(w), 2)

    # 定义测试函数 `test_byte_mask2d`，测试二维字节掩码的行为
    def test_byte_mask2d(self, device="mps"):
        # 创建随机张量 `v`
        v = torch.randn(5, 7, 3, device=device)
        # 创建随机张量 `c`
        c = torch.randn(5, 7, device=device)
        # 计算张量 `c` 中大于零的元素数量
        num_ones = (c > 0).sum()
        # 根据掩码 `c > 0` 进行索引操作
        r = v[c > 0]
        # 检查结果张量 `r` 的形状是否符合预期
        self.assertEqual(r.shape, (num_ones, 3))

    # 定义测试函数 `test_jit_indexing`，测试 JIT 编译的索引操作
    def test_jit_indexing(self, device="mps"):
        # 定义函数 `fn1`，执行索引小于 50 的元素赋值为 1.0
        def fn1(x):
            x[x < 50] = 1.0
            return x

        # 定义函数 `fn2`，执行索引范围在 0 到 49 的元素赋值为 1.0
        def fn2(x):
            x[0:50] = 1.0
            return x

        # 对函数 `fn1` 和 `fn2` 进行 JIT 编译
        scripted_fn1 = torch.jit.script(fn1)
        scripted_fn2 = torch.jit.script(fn2)
        # 创建从 0 到 99 的浮点数张量 `data`
        data = torch.arange(100, device=device, dtype=torch.float)
        # 执行 JIT 编译后的函数 `scripted_fn1` 对 `data` 的操作，并与参考结果 `ref` 进行比较
        out = scripted_fn1(data.detach().clone())
        ref = torch.tensor(np.concatenate((np.ones(50), np.arange(50, 100))), device=device, dtype=torch.float)
        self.assertEqual(out, ref)
        # 执行 JIT 编译后的函数 `scripted_fn2` 对 `data` 的操作，并与参考结果 `ref` 进行比较
        out = scripted_fn2(data.detach().clone())
        self.assertEqual(out, ref)
    # 测试使用整数索引在张量中选择子集
    def test_int_indices(self, device="mps"):
        # 创建一个形状为 (5, 7, 3) 的张量，并初始化为随机值，使用指定设备
        v = torch.randn(5, 7, 3, device=device)
        # 验证选择具有索引 [0, 4, 2] 的子集后的形状为 (3, 7, 3)
        self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
        # 验证选择所有行，并且列索引为 [0, 4, 2] 的子集后的形状为 (5, 3, 3)
        self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
        # 验证选择所有行，并且行和列索引为 [[0, 1], [4, 3]] 的子集后的形状为 (5, 2, 2, 3)
        self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

    # 测试在原始张量上使用索引放置操作，并验证数据类型
    def test_index_put_src_datatype(self):
        def helper(device, dtype):
            # 创建一个形状为 (3, 2, 4) 的张量，元素值全部为 1，使用指定设备和数据类型
            src = torch.ones(3, 2, 4, device=device, dtype=dtype)
            # 创建一个形状与 src 相同的张量 vals，元素值全部为 1
            vals = torch.ones(3, 2, 4, device=device, dtype=dtype)
            # 定义索引为 (tensor([0, 2, 1]),)，即在第一个维度上索引为 0, 2, 1 的元素
            indices = (torch.tensor([0, 2, 1]),)
            # 在 src 上执行索引放置操作，累加到原始张量上，并返回结果
            res = src.index_put_(indices, vals, accumulate=True)
            # 验证操作后的张量形状与原始张量相同
            self.assertEqual(res.shape, src.shape)
        # 对于指定的设备和数据类型分别调用 helper 函数
        [helper(device="mps", dtype=dtype) for dtype in [torch.float, torch.int32]]

    # 在指定产品版本不低于 13.0 的情况下，测试在原始张量上使用索引操作的数据类型
    @unittest.skipIf(product_version < 13.0, "Skipped on macOS 12")
    def test_index_src_datatype(self):
        def helper(device, dtype):
            # 处理布尔类型特例，将其转换为 uint8 类型
            orig_dtype = dtype
            if dtype is torch.bool:
                dtype = torch.uint8

            # 创建一个形状为 (3, 2, 4) 的张量，元素值全部为 1，使用指定设备和数据类型
            src = torch.ones(3, 2, 4, device=device, dtype=dtype)
            # 如果原始数据类型是布尔型，将 src 转换为布尔型张量
            if orig_dtype is torch.bool:
                src = src == 1
            # 使用整数索引 [0, 2, 1] 选择子集，并验证选择后的形状与原始张量相同
            res = src[[0, 2, 1], :, :]
            self.assertEqual(res.shape, src.shape)
            # 使用索引放置操作，不累加，将 res 的值放回到对应索引的位置，并验证形状与原始张量相同
            src[[0, 2, 1], :, :] = res
            self.assertEqual(res.shape, src.shape)
        # 对于指定的设备和数据类型分别调用 helper 函数
        [helper(device="mps", dtype=dtype) for dtype in [torch.float, torch.float16, torch.long, torch.bool]]

    # 测试在二维张量上使用整数索引进行选择操作，验证返回值是否正确
    def test_int_indices2d(self, device="mps"):
        # 创建一个形状为 (4, 3) 的张量，并初始化为从 0 到 11 的连续值，使用指定设备
        # 然后将其重塑为 (4, 3) 的形状
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 定义行索引和列索引的张量
        rows = torch.tensor([[0, 0], [3, 3]], device=device)
        columns = torch.tensor([[0, 2], [0, 2]], device=device)
        # 使用行索引和列索引选择子集，并验证结果是否与预期相同
        self.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])

    # 测试在二维张量上使用整数索引进行广播操作，验证返回值是否正确
    def test_int_indices_broadcast(self, device="mps"):
        # 创建一个形状为 (4, 3) 的张量，并初始化为从 0 到 11 的连续值，使用指定设备
        # 然后将其重塑为 (4, 3) 的形状
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 定义行索引和列索引的张量
        rows = torch.tensor([0, 3], device=device)
        columns = torch.tensor([0, 2], device=device)
        # 使用行索引进行广播选择子集，并验证结果是否与预期相同
        result = x[rows[:, None], columns]
        self.assertEqual(result.tolist(), [[0, 2], [9, 11]])

    # 测试在空索引上的操作，验证返回的子集是否为空
    def test_empty_index(self, device="mps"):
        # 创建一个形状为 (4, 3) 的张量，并初始化为从 0 到 11 的连续值，使用指定设备
        # 然后将其重塑为 (4, 3) 的形状
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 创建一个空的索引张量
        idx = torch.tensor([], dtype=torch.long, device=device)
        # 验证使用空索引选择的子集元素数量为 0
        self.assertEqual(x[idx].numel(), 0)

        # 对空索引执行赋值操作，不应该有任何效果，但不应该抛出异常
        y = x.clone()
        y[idx] = -1
        self.assertEqual(x, y)

        # 使用布尔掩码索引赋值，不应该有任何效果，但不应该抛出异常
        mask = torch.zeros(4, 3, device=device).bool()
        y[mask] = -1
        self.assertEqual(x, y)
    # 测试空的多维索引操作
    def test_empty_ndim_index(self, device="mps"):
        # 创建一个形状为 (5,) 的张量 x，包含随机数，指定设备
        x = torch.randn(5, device=device)
        # 使用空的索引张量对 x 进行索引，预期结果是一个空的形状为 (0, 2) 的张量
        self.assertEqual(torch.empty(0, 2, device=device), x[torch.empty(0, 2, dtype=torch.int64, device=device)])

        # 创建一个形状为 (2, 3, 4, 5) 的张量 x，包含随机数，指定设备
        x = torch.randn(2, 3, 4, 5, device=device)
        # 使用空的索引张量对 x 的第二个维度进行索引，预期结果是一个形状为 (2, 0, 6, 4, 5) 的张量
        self.assertEqual(torch.empty(2, 0, 6, 4, 5, device=device),
                         x[:, torch.empty(0, 6, dtype=torch.int64, device=device)])

        # 创建一个形状为 (10, 0) 的空张量 x，指定设备
        x = torch.empty(10, 0, device=device)
        # 使用索引 [1, 2] 对 x 进行索引，预期结果的形状是 (2, 0)
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        # 使用空索引对 x 进行索引，预期结果的形状是 (0,)
        self.assertEqual(x[[], []].shape, (0,))
        # 使用带有维度大小为 0 的索引，预期引发 IndexError 异常
        with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
            x[:, [0, 1]]

    # 测试空的多维布尔索引操作
    def test_empty_ndim_index_bool(self, device="mps"):
        # 创建一个形状为 (5,) 的张量 x，包含随机数，指定设备
        x = torch.randn(5, device=device)
        # 使用空的布尔索引张量对 x 进行索引，预期引发 IndexError 异常
        self.assertRaises(IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)])

    # 测试切片操作
    def test_empty_slice(self, device="mps"):
        # 创建一个形状为 (2, 3, 4, 5) 的张量 x，包含随机数，指定设备
        x = torch.randn(2, 3, 4, 5, device=device)
        # 对 x 进行多次切片操作，z 是结果张量
        y = x[:, :, :, 1]
        z = y[:, 1:1, :]
        # 检查 z 的形状是否符合预期 (2, 0, 4)
        self.assertEqual((2, 0, 4), z.shape)
        # 检查 z 的步幅是否符合预期 (60, 20, 5)
        # 这个断言不是必须的，但是与 NumPy 的步幅计算方式相匹配
        self.assertEqual((60, 20, 5), z.stride())
        # 检查 z 是否是连续的张量
        self.assertTrue(z.is_contiguous())

    # 测试索引和获取元素操作（使用布尔值和切片）
    def test_index_getitem_copy_bools_slices(self, device="mps"):
        # 创建布尔值张量 true 和 false，分别表示真和假
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        # 创建包含不同类型张量的列表 tensors
        tensors = [torch.randn(2, 3, device=device), torch.tensor(3., device=device)]

        # 遍历 tensors 列表中的每个张量 a
        for a in tensors:
            # 检查使用布尔值 True 对 a 进行索引后是否生成了新的张量
            self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
            # 使用布尔值 False 对 a 进行索引，预期结果是一个空的张量，形状与 a 相同
            self.assertEqual(torch.empty(0, *a.shape), a[False])
            # 检查使用布尔值 true 对 a 进行索引后是否生成了新的张量
            self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
            # 使用布尔值 false 对 a 进行索引，预期结果是一个空的张量，形状与 a 相同
            self.assertEqual(torch.empty(0, *a.shape), a[false])
            # 使用 None 对 a 进行索引，预期结果与 a 的数据指针相同
            self.assertEqual(a.data_ptr(), a[None].data_ptr())
            # 使用 ... 对 a 进行索引，预期结果与 a 的数据指针相同
            self.assertEqual(a.data_ptr(), a[...].data_ptr())
    def test_index_setitem_bools_slices(self, device="mps"):
        # 创建 torch.uint8 类型的张量 true 和 false，分别代表 1 和 0
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        # 创建包含两个张量的列表 tensors
        tensors = [torch.randn(2, 3, device=device), torch.tensor(3, device=device)]

        # 对于列表中的每个张量 a
        for a in tensors:
            # 创建一个与 a 相同形状的张量，填充为 -1
            neg_ones = torch.ones_like(a) * -1
            # 在 neg_ones 前面添加两个维度为 1 的维度，以确保与 NumPy 兼容，后续的操作可能会自动添加一个 1 的前缀尺寸
            neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
            # 使用 neg_ones_expanded 赋值给 a 中值为真的索引位置
            a[True] = neg_ones_expanded
            # 断言 a 的值与 neg_ones 相等
            self.assertEqual(a, neg_ones)
            # 使用值为假的索引位置赋值为 5，预期 a 的值不变
            a[False] = 5
            self.assertEqual(a, neg_ones)
            # 使用 true （张量值为 1）的索引位置赋值为 neg_ones_expanded 的两倍
            self.assertEqual(a, neg_ones * 2)
            # 使用 false （张量值为 0）的索引位置赋值为 5，预期 a 的值不变
            a[false] = 5
            self.assertEqual(a, neg_ones * 2)
            # 使用 None 赋值给所有索引位置，赋值为 neg_ones_expanded 的三倍
            a[None] = neg_ones_expanded * 3
            self.assertEqual(a, neg_ones * 3)
            # 使用 ... 赋值给所有索引位置，赋值为 neg_ones_expanded 的四倍
            a[...] = neg_ones_expanded * 4
            self.assertEqual(a, neg_ones * 4)
            # 如果 a 的维度为 0，预期引发 IndexError 异常
            if a.dim() == 0:
                with self.assertRaises(IndexError):
                    a[:] = neg_ones_expanded * 5

    def test_index_scalar_with_bool_mask(self, device="mps"):
        # 创建 torch.uint8 类型的张量 true 和 false，分别代表 True 和 False
        a = torch.tensor(1, device=device)
        uintMask = torch.tensor(True, dtype=torch.uint8, device=device)
        boolMask = torch.tensor(True, dtype=torch.bool, device=device)
        # 使用 uintMask 和 boolMask 获取 a 的值，预期结果相等
        self.assertEqual(a[uintMask], a[boolMask])
        # 检查获取的张量类型是否相同
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

        # 创建 torch.bool 类型的张量 a，并再次使用 uintMask 和 boolMask 获取 a 的值，预期结果相等
        a = torch.tensor(True, dtype=torch.bool, device=device)
        self.assertEqual(a[uintMask], a[boolMask])
        # 检查获取的张量类型是否相同
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

    def test_setitem_expansion_error(self, device="mps"):
        # 创建 torch.bool 类型的张量 true
        true = torch.tensor(True, device=device)
        # 创建一个形状为 (2, 3) 的随机张量 a
        a = torch.randn(2, 3, device=device)
        # 尝试在 a 上扩展形状为 (5, 1) + a.size() 的张量 a_expanded
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        # 预期在使用 true 赋值时引发 RuntimeError
        with self.assertRaises(RuntimeError):
            a[True] = a_expanded
        # 再次预期在使用 true 赋值时引发 RuntimeError
        with self.assertRaises(RuntimeError):
            a[true] = a_expanded
    # 测试索引器处理标量情况的方法，设备为 "mps"
    def test_getitem_scalars(self, device="mps"):
        # 创建一个值为 0 的整型张量，数据类型为 torch.int64，在指定设备上
        zero = torch.tensor(0, dtype=torch.int64, device=device)
        # 创建一个值为 1 的整型张量，数据类型为 torch.int64，在指定设备上
        one = torch.tensor(1, dtype=torch.int64, device=device)

        # 创建一个在指定设备上形状为 (2, 3) 的随机张量
        a = torch.randn(2, 3, device=device)
        # 断言索引为 0 的元素等于用 zero 索引的元素
        self.assertEqual(a[0], a[zero])
        # 断言索引为 [0][1] 的元素等于用 zero 和 one 索引的元素
        self.assertEqual(a[0][1], a[zero][one])
        # 断言索引为 [0, 1] 的元素等于用 zero 和 one 索引的元素
        self.assertEqual(a[0, 1], a[zero, one])
        # 断言索引为 [0, one] 的元素等于用 zero 和索引为 1 的元素
        self.assertEqual(a[0, one], a[zero, 1])

        # 断言标量索引应该是切片而不是复制
        # 检查索引为 [0, 1] 的元素的数据指针是否与用 zero 和 one 索引的元素的数据指针相同
        self.assertEqual(a[0, 1].data_ptr(), a[zero, one].data_ptr())
        # 检查索引为 [1] 的元素的数据指针是否与用 one.int() 索引的元素的数据指针相同
        self.assertEqual(a[1].data_ptr(), a[one.int()].data_ptr())
        # 检查索引为 [1] 的元素的数据指针是否与用 one.short() 索引的元素的数据指针相同
        self.assertEqual(a[1].data_ptr(), a[one.short()].data_ptr())

        # 标量索引与标量
        # 创建一个在指定设备上形状为空的随机张量
        r = torch.randn((), device=device)
        # 用 assertRaises 检查是否会引发 IndexError 异常
        with self.assertRaises(IndexError):
            r[:]
        with self.assertRaises(IndexError):
            r[zero]
        # 断言 r 等于 r 的所有切片
        self.assertEqual(r, r[...])

    # 测试标量赋值的方法，设备为 "mps"
    def test_setitem_scalars(self, device="mps"):
        # 创建一个值为 0 的整型张量，数据类型为 torch.int64
        zero = torch.tensor(0, dtype=torch.int64)

        # 创建一个在指定设备上形状为 (2, 3) 的随机张量
        a = torch.randn(2, 3, device=device)
        # 使用 clone 复制 a
        a_set_with_number = a.clone()
        a_set_with_scalar = a.clone()
        # 创建一个在指定设备上形状为 (3,) 的随机张量
        b = torch.randn(3, device=device)

        # 使用数字赋值索引为 0 的元素
        a_set_with_number[0] = b
        # 使用标量赋值索引为 zero 的元素
        a_set_with_scalar[zero] = b
        # 断言两者相等
        self.assertEqual(a_set_with_number, a_set_with_scalar)
        # 修改索引为 [1, 0] 的元素为 7.7
        a[1, zero] = 7.7
        # 断言索引为 [1, 0] 的元素为 7.7
        self.assertEqual(7.7, a[1, 0])

        # 标量索引与标量
        # 创建一个在指定设备上形状为空的随机张量
        r = torch.randn((), device=device)
        # 用 assertRaises 检查是否会引发 IndexError 异常
        with self.assertRaises(IndexError):
            r[:] = 8.8
        with self.assertRaises(IndexError):
            r[zero] = 8.8
        # 使用标量赋值所有切片为 9.9
        r[...] = 9.9
        # 断言 r 等于 9.9
        self.assertEqual(9.9, r)

    # 测试基本和高级索引结合的方法，设备为 "mps"
    def test_basic_advanced_combined(self, device="mps"):
        # 从 NumPy 索引示例中创建一个在指定设备上的张量
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 断言切片 [1:2, 1:3] 等于索引 [1:2, [1, 2]] 的元素
        self.assertEqual(x[1:2, 1:3], x[1:2, [1, 2]])
        # 断言切片 [1:2, 1:3] 转换为列表后等于 [[4, 5]]
        self.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

        # 检查它是否是复制
        # 使用 clone 复制 x
        unmodified = x.clone()
        # 将索引为 [1:2, [1, 2]] 的元素置零
        x[1:2, [1, 2]].zero_()
        # 断言 x 等于未修改前的 unmodified
        self.assertEqual(x, unmodified)

        # 但赋值应修改原始张量
        # 使用 clone 复制 x
        unmodified = x.clone()
        # 将索引为 [1:2, [1, 2]] 的元素赋值为 0
        x[1:2, [1, 2]] = 0
        # 断言 x 不等于未修改前的 unmodified
        self.assertNotEqual(x, unmodified)

    # 测试整数赋值的方法，设备为 "mps"
    def test_int_assignment(self, device="mps"):
        # 创建一个在指定设备上形状为 (2, 2) 的张量
        x = torch.arange(0, 4, device=device).view(2, 2)
        # 将索引为 [1] 的元素赋值为 5
        x[1] = 5
        # 断言 x 转换为列表后等于 [[0, 1], [5, 5]]
        self.assertEqual(x.tolist(), [[0, 1], [5, 5]])

        # 创建一个在指定设备上形状为 (2, 2) 的张量
        x = torch.arange(0, 4, device=device).view(2, 2)
        # 将索引为 [1] 的元素赋值为 torch.arange(5, 7) 的张量
        x[1] = torch.arange(5, 7, device=device)
        # 断言 x 转换为列表后等于 [[0, 1], [5, 6]]
        self.assertEqual(x.tolist(), [[0, 1], [5, 6]])
    # 测试在指定设备上进行字节张量赋值
    def test_byte_tensor_assignment(self, device="mps"):
        # 创建一个 4x4 的张量，从 0 到 15
        x = torch.arange(0., 16, device=device).view(4, 4)
        # 创建一个字节张量，指定位置为 True 或 False，转移到指定设备
        b = torch.ByteTensor([True, False, True, False]).to(device)
        # 创建一个张量，包含值 [3., 4., 5., 6.]，转移到指定设备
        value = torch.tensor([3., 4., 5., 6.], device=device)

        # 捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 根据字节张量 b 修改张量 x 的值为指定的 value
            x[b] = value
            # 断言捕获的警告数量为 1
            self.assertEqual(len(w), 1)

        # 断言张量 x 的特定行的值与 value 相等
        self.assertEqual(x[0], value)
        self.assertEqual(x[1], torch.arange(4., 8, device=device))
        self.assertEqual(x[2], value)
        self.assertEqual(x[3], torch.arange(12., 16, device=device))

    # 测试变量切片操作
    def test_variable_slicing(self, device="mps"):
        # 创建一个 4x4 的张量，从 0 到 15
        x = torch.arange(0, 16, device=device).view(4, 4)
        # 创建一个整数张量，包含索引 [0, 1]，转移到指定设备
        indices = torch.IntTensor([0, 1]).to(device)
        i, j = indices
        # 断言切片操作的结果与预期相等
        self.assertEqual(x[i:j], x[0:1])

    # 测试省略符号张量操作
    def test_ellipsis_tensor(self, device="mps"):
        # 创建一个 3x3 的张量，从 0 到 8
        x = torch.arange(0, 9, device=device).view(3, 3)
        # 创建一个张量，包含索引 [0, 2]，转移到指定设备
        idx = torch.tensor([0, 2], device=device)
        # 断言使用省略符号操作后的结果与预期相等
        self.assertEqual(x[..., idx].tolist(), [[0, 2],
                                                [3, 5],
                                                [6, 8]])
        self.assertEqual(x[idx, ...].tolist(), [[0, 1, 2],
                                                [6, 7, 8]])

    # 测试无效索引操作
    def test_invalid_index(self, device="mps"):
        # 创建一个 4x4 的张量，从 0 到 15
        x = torch.arange(0, 16, device=device).view(4, 4)
        # 断言在尝试使用无效索引时抛出预期的 TypeError 异常
        self.assertRaisesRegex(TypeError, 'slice indices', lambda: x["0":"1"])

    # 测试超出边界索引操作
    def test_out_of_bound_index(self, device="mps"):
        # 创建一个 2x5x10 的张量，从 0 到 99
        x = torch.arange(0, 100, device=device).view(2, 5, 10)
        # 断言在超出维度大小的索引操作时抛出预期的 IndexError 异常
        self.assertRaisesRegex(IndexError, 'index 5 is out of bounds for dimension 1 with size 5', lambda: x[0, 5])
        self.assertRaisesRegex(IndexError, 'index 4 is out of bounds for dimension 0 with size 2', lambda: x[4, 5])
        self.assertRaisesRegex(IndexError, 'index 15 is out of bounds for dimension 2 with size 10',
                               lambda: x[0, 1, 15])
        self.assertRaisesRegex(IndexError, 'index 12 is out of bounds for dimension 2 with size 10',
                               lambda: x[:, :, 12])

    # 测试零维度索引操作
    def test_zero_dim_index(self, device="mps"):
        # 创建一个标量张量为 10，转移到指定设备
        x = torch.tensor(10, device=device)
        # 断言张量 x 的值与其单个元素的值相等
        self.assertEqual(x, x.item())

        # 定义一个内部函数 runner，用于尝试访问索引 0 的元素
        def runner():
            print(x[0])
            return x[0]

        # 断言在访问无效索引时抛出预期的 IndexError 异常
        self.assertRaisesRegex(IndexError, 'invalid index', runner)

    # 测试 CPU 上的索引操作
    def test_cpu_indices(self, device="mps"):
        # 创建一个张量，包含索引 [0, 1]
        idx = torch.tensor([0, 1])
        # 创建一个形状为 (2,) 的零张量，转移到指定设备
        b = torch.zeros(2, device=device)
        # 创建一个长度为 10 的全为 1 的张量，转移到指定设备
        x = torch.ones(10, device=device)
        x[idx] = b  # index_put_
        # 创建一个长度为 10 的全为 1 的张量作为参考
        ref = torch.ones(10, device=device)
        ref[:2] = 0
        # 断言修改后的张量 x 与参考张量 ref 相等
        self.assertEqual(x, ref, atol=0, rtol=0)
        # 获取张量 x 中索引 idx 处的值
        out = x[idx]  # index
        # 断言获取的值与预期的全零张量相等
        self.assertEqual(out, torch.zeros(2, device=device), atol=0, rtol=0)
    # 定义一个测试方法 `test_nextafter`，用于测试 `torch.nextafter` 函数的行为
    def test_nextafter(self, device="mps"):
        # 遍历数据类型列表，包括 torch.float16 和 torch.float32
        for dtype in [torch.float16, torch.float32]:
            # 创建包含特定数据类型的张量 x 和 y，设备为指定的 device
            x = torch.tensor([1, -1, 0, 0, 2, -2], device=device, dtype=dtype)
            y = torch.tensor([2, -2, -1, 1, -3, 3], device=device, dtype=dtype)
            
            # 使用 torch.nextafter 函数计算 x 在给定 y 方向上的下一个浮点数
            na = torch.nextafter(x, y)
            
            # 将 x 和 y 张量转移到 CPU 并使用 torch.nextafter 函数计算
            na_cpu = torch.nextafter(x.cpu(), y.cpu())
            
            # 比较 na 在 MPS 设备上是否大于 x 在 CPU 上的结果
            na_ge_x_mps = na.cpu() > x.cpu()
            
            # 在 CPU 上比较 na_cpu 是否大于 x.cpu()，这里是为了避免 MPS 设备上的错误
            # greater 在 MPS 上存在问题，请参考 https://github.com/pytorch/pytorch/issues/125051
            na_ge_x_cpu = na_cpu > x.cpu()
            
            # 断言 MPS 设备上的结果与 CPU 上的结果是否一致
            self.assertEqual(na_ge_x_mps, na_ge_x_cpu)
# 定义一个测试类 TestRNNMPS，继承自 TestCaseMPS 类
class TestRNNMPS(TestCaseMPS):

    # LSTM_TEST_CASES 是一个包含多个字典的列表，用于测试 LSTM 的不同配置
    LSTM_TEST_CASES = [
        dict(),  # 默认配置
        dict(batch_first=True),  # 设置 batch_first 为 True
        dict(bias=False),  # 设置 bias 为 False
        dict(bidirectional=True),  # 设置 bidirectional 为 True
        dict(batch_first=True, bias=False),  # 同时设置 batch_first 为 True 和 bias 为 False
        dict(bidirectional=True, bias=False),  # 同时设置 bidirectional 为 True 和 bias 为 False
        dict(bidirectional=True, batch_first=True),  # 同时设置 bidirectional 为 True 和 batch_first 为 True
        dict(bidirectional=True, batch_first=True, bias=False)  # 同时设置 bidirectional 为 True, batch_first 为 True 和 bias 为 False
    ]

    # 定义测试 LSTM 正向传播的方法
    def test_lstm_forward(self, device="mps", dtype=torch.float32):
        # 遍历不同的层数 [1, 2, 5]
        for num_layers in [1, 2, 5]:
            # 遍历 LSTM_TEST_CASES 列表中的测试配置
            for test_options in self.LSTM_TEST_CASES:
                # 调用 _lstm_helper 方法进行 LSTM 正向传播测试
                self._lstm_helper(num_layers=num_layers, dtype=dtype, device=device, **test_options)

    # 标记在 MacOS-14.4 上失败，但在 14.2 上可以工作，参见 https://github.com/pytorch/pytorch/issues/125803
    @xfailIfMacOS14_4Plus
    # 定义测试 LSTM 反向传播的方法
    def test_lstm_backward(self, device="mps", dtype=torch.float32):
        # 遍历不同的层数 [1, 2, 5]
        for num_layers in [1, 2, 5]:
            # 遍历 LSTM_TEST_CASES 列表中的测试配置
            for test_options in self.LSTM_TEST_CASES:
                # 调用 _lstm_helper 方法进行 LSTM 反向传播测试，设置 backward=True
                self._lstm_helper(num_layers=num_layers, dtype=dtype, device=device, backward=True, **test_options)

    # 定义测试 RNN cell 没有广播功能的方法
    def test_RNN_cell_no_broadcasting(self):
        # 定义内部函数 test，用于测试单个 cell 模块
        def test(cell_module, input, hx, input_size, hidden_size):
            # 创建指定设备上的 cell 实例
            cell = cell_module(input_size, hidden_size, device='mps')
            # 断言在调用时会抛出 RuntimeError 异常
            self.assertRaises(RuntimeError, lambda: cell(input, hx))

        # 定义内部函数 test_all，用于测试多种 cell 模块
        def test_all(hidden_size, bad_hx, good_hx, input_size, input):
            # 测试 nn.RNNCell、nn.GRUCell 和 nn.LSTMCell 三种 cell 模块
            test(nn.RNNCell, input, bad_hx, input_size, hidden_size)
            test(nn.GRUCell, input, bad_hx, input_size, hidden_size)
            test(nn.LSTMCell, input, (bad_hx, good_hx), input_size, hidden_size)
            test(nn.LSTMCell, input, (good_hx, bad_hx), input_size, hidden_size)

        # 定义输入和隐藏状态的维度
        hidden_size = 20
        input_size = 10
        input = torch.randn(3, input_size, device='mps')
        bad_hx = torch.randn(1, hidden_size, device='mps')
        good_hx = torch.randn(3, hidden_size, device='mps')

        # 测试隐藏状态和输入的批处理大小广播功能
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # 测试隐藏状态的隐藏大小与模块的隐藏大小广播功能
        bad_hx = torch.randn(3, 1)
        test_all(hidden_size, bad_hx, good_hx, input_size, input)

        # 测试输入的输入大小与模块的输入大小广播功能
        bad_input = torch.randn(3, 1)
        test_all(hidden_size, good_hx, good_hx, input_size, bad_input)

    # 定义测试 LSTM cell 的方法
    def test_LSTM_cell(self):
        # 这仅是一个简单的 smoke test；这些模块是通过 autograd 实现的，因此不需要 Jacobian 测试
        # 遍历是否包含偏置参数的两种情况
        for bias in (True, False):
            # 创建输入、隐藏状态和 cell 实例
            input = torch.randn(3, 10, device='mps')
            hx = torch.randn(3, 20, device='mps')
            cx = torch.randn(3, 20, device='mps')
            lstm = nn.LSTMCell(10, 20, bias=bias, device='mps')
            # 运行多次 cell 的前向传播
            for _ in range(6):
                hx, cx = lstm(input, (hx, cx))

            # 对输出进行求和并进行反向传播
            (hx + cx).sum().backward()
    # 测试 LSTM 单元正向传播函数，验证输入维度是否正确
    def test_LSTM_cell_forward_input_size(self):
        # 创建一个形状为 (3, 11) 的随机张量作为输入，设备类型为 'mps'
        input = torch.randn(3, 11, device='mps')
        # 创建一个形状为 (3, 20) 的随机张量作为初始隐藏状态，设备类型为 'mps'
        hx = torch.randn(3, 20, device='mps')
        # 创建一个形状为 (3, 20) 的随机张量作为初始细胞状态，设备类型为 'mps'
        cx = torch.randn(3, 20, device='mps')
        # 创建一个输入维度为 10，隐藏状态维度为 20 的 LSTM 单元，设备类型为 'mps'
        lstm = nn.LSTMCell(10, 20, device='mps')
        # 断言调用 LSTM 单元时输入维度不匹配会抛出异常
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
    
    # 测试 LSTM 单元正向传播函数，验证隐藏状态维度是否正确
    def test_LSTM_cell_forward_hidden_size(self):
        # 创建一个形状为 (3, 10) 的随机张量作为输入，设备类型为 'mps'
        input = torch.randn(3, 10, device='mps')
        # 创建一个形状为 (3, 21) 的随机张量作为初始隐藏状态，设备类型为 'mps'
        hx = torch.randn(3, 21, device='mps')
        # 创建一个形状为 (3, 20) 的随机张量作为初始细胞状态，设备类型为 'mps'
        cx = torch.randn(3, 20, device='mps')
        # 创建一个输入维度为 10，隐藏状态维度为 20 的 LSTM 单元，设备类型为 'mps'
        lstm = nn.LSTMCell(10, 20, device='mps')
        # 断言调用 LSTM 单元时隐藏状态维度不匹配会抛出异常
        self.assertRaises(Exception, lambda: lstm(input, (hx, cx)))
        # 断言调用 LSTM 单元时细胞状态和隐藏状态维度交换会抛出异常
        self.assertRaises(Exception, lambda: lstm(input, (cx, hx)))
class TestFallbackWarning(TestCase):
    # TODO: Remove once test_testing.py is running on MPS devices
    # 在 MPS 设备上运行 test_testing.py 后移除此 TODO

    def test_no_warning_on_import(self):
        # 执行子进程，检查导入 torch 是否会产生警告
        out = subprocess.check_output(
            [sys.executable, "-W", "all", "-c", "import torch"],
            stderr=subprocess.STDOUT,
            # 在 Windows 上，使用默认的当前工作目录会导致 subprocess 执行 `import torch` 失败，
            # 所以将当前工作目录设置为此脚本所在目录
            cwd=os.path.dirname(os.path.realpath(__file__)),
        ).decode("utf-8")
        # 断言输出为空字符串
        self.assertEqual(out, "")

    def _get_not_implemented_op(self):
        # 当实际实现 'lcm' 后可以修改这里的内容
        # 应返回函数 fn, 参数 args, 关键字参数 kwargs, 字符串版本信息
        return (torch.lcm,
                [torch.tensor([1], device='mps'), torch.tensor([2], device='mps')], {},
                "torch.lcm(torch.tensor([1], device='mps'), torch.tensor([2], device='mps'))")

    def test_error_on_not_implemented(self):
        # 获取未实现操作的函数、参数、关键字参数和占位字符串
        fn, args, kwargs, _ = self._get_not_implemented_op()

        # 使用断言检查是否抛出 NotImplementedError，错误消息包含指定文本
        with self.assertRaisesRegex(NotImplementedError, "not currently implemented for the MPS device"):
            fn(*args, **kwargs)

    def test_warn_on_not_implemented_with_fallback(self):
        # 获取未实现操作的函数、参数、关键字参数和操作字符串
        _, _, _, op = self._get_not_implemented_op()
        # 构建脚本字符串，用于测试是否会警告使用 MPS 回退功能
        script = f"""
import os
# 必须在导入 pytorch 之前设置
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import warnings

with warnings.catch_warnings(record=True) as w:
    import torch

if len(w) > 0:
    print(w)
    exit(1)

# 这应该正常运行，并且会触发性能警告
with warnings.catch_warnings(record=True) as w:
    {op}

if len(w) != 1:
    print(w)
    exit(2)

"""
        try:
            # 执行子进程，运行上述脚本
            subprocess.check_output(
                [sys.executable, '-W', 'all', '-c', script],
                stderr=subprocess.STDOUT,
                # 在 Windows 上，使用默认的当前工作目录会导致 subprocess 执行 `import torch` 失败，
                # 所以将当前工作目录设置为此脚本所在目录
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                # 如果返回码为 1，断言失败：设置了 PYTORCH_ENABLE_MPS_FALLBACK 时导入 torch 时有警告
                self.assertTrue(False, "There was a warning when importing torch when PYTORCH_ENABLE_MPS_FALLBACK is set." +
                                       e.output.decode("utf-8"))
            elif e.returncode == 2:
                # 如果返回码为 2，断言失败：在设置了 PYTORCH_ENABLE_MPS_FALLBACK 的情况下，未实现的操作未正好触发一次警告
                self.assertTrue(False, "There wasn't exactly one warning when running not implemented op with "
                                f"PYTORCH_ENABLE_MPS_FALLBACK set. {e.output}")
            else:
                # 其他情况下断言失败：即使设置了 PYTORCH_ENABLE_MPS_FALLBACK，运行未实现的操作仍然失败
                self.assertTrue(False, "Running a not implemented op failed even though PYTORCH_ENABLE_MPS_FALLBACK is set. " +
                                       e.output.decode("utf-8"))

class TestNoRegression(TestCase):
    # 这是一个空测试类，用于检查没有回归
    # 定义一个测试函数，用于验证 assert_close 方法的行为
    def test_assert_close(self):
        # 创建一个包含全为1的张量 a，在 "mps" 设备上
        a = torch.ones(1, device="mps")
        # 创建一个包含全为0的张量 b，在 "mps" 设备上
        b = torch.zeros(1, device="mps")
        # 计算 a / b，得到 inf (无穷大)
        inf = a / b
        # 计算 b / b，得到 nan (非数)
        nan = b / b

        # 使用 assertRaisesRegex 断言捕获 AssertionError 异常，验证 a 和 inf 不近似
        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
            torch.testing.assert_close(a, inf)

        # TODO: 当在 test_mps 中运行所有测试时，NaN 测试失败，但分开运行时通过。
        # 存在可能的内存损坏问题，需要修复才能启用此测试。
        # with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
        #     torch.testing.assert_close(a, nan)

    # 定义一个测试函数，用于验证在尝试使用 float64 时是否会引发 TypeError
    def test_double_error(self):
        # 使用 assertRaisesRegex 断言捕获 TypeError 异常，验证 "mps" 框架不支持 float64 类型
        with self.assertRaisesRegex(TypeError, "the MPS framework doesn't support float64"):
            a = torch.ones(2, dtype=torch.float64, device="mps")

        # 创建一个包含全为1的张量 a，在 "mps" 设备上
        a = torch.ones(2, device="mps")
        # 使用 assertRaisesRegex 断言捕获 TypeError 异常，验证 "mps" 框架不支持 float64 类型
        with self.assertRaisesRegex(TypeError, "the MPS framework doesn't support float64"):
            a = a.double()

    # 定义一个测试函数，用于验证在创建新张量时的行为
    def test_legacy_constructor(self):
        # 创建一个包含全为1的张量 a，在 "mps" 设备上
        a = torch.ones(2, device="mps")
        # 使用 a 的 new 方法创建一个新的张量 b
        b = a.new(1)

    # 定义一个测试函数，用于验证张量在不同设备之间的序列化和反序列化行为
    def test_serialization_map_location(self):
        # 确保可以将 CPU 上的张量加载到 "mps" 设备上
        with tempfile.NamedTemporaryFile() as f:
            # 创建一个包含随机数的张量 x
            x = torch.rand(2)
            # 将张量 x 保存到临时文件 f 中
            torch.save(x, f)

            # 重置文件指针位置到开头
            f.seek(0)
            # 从文件 f 中加载张量 x2，并将其映射到 "mps" 设备
            x2 = torch.load(f, map_location="mps")

            # 使用断言验证 x 和 x2 的值相等
            self.assertEqual(x, x2)
            # 使用断言验证 x2 的设备类型为 "mps"
            self.assertEqual(x2.device.type, "mps")

        # 确保可以将 "mps" 上的张量加载到 "mps" 设备上
        with tempfile.NamedTemporaryFile() as f:
            # 创建一个包含随机数的张量 x，在 "mps" 设备上
            x = torch.rand(2, device="mps")
            # 将张量 x 保存到临时文件 f 中
            torch.save(x, f)

            # 重置文件指针位置到开头
            f.seek(0)
            # 从文件 f 中加载张量 x2
            x2 = torch.load(f)

            # 使用断言验证 x 和 x2 的值相等
            self.assertEqual(x, x2)
            # 使用断言验证 x2 的设备类型为 "mps"
            self.assertEqual(x2.device.type, "mps")

        # 确保可以将 "mps" 上的张量加载到 CPU 设备上
        with tempfile.NamedTemporaryFile() as f:
            # 创建一个包含随机数的张量 x，在 "mps" 设备上
            x = torch.rand(2, device="mps")
            # 将张量 x 保存到临时文件 f 中
            torch.save(x, f)

            # 重置文件指针位置到开头
            f.seek(0)
            # 从文件 f 中加载张量 x2，并将其映射到 CPU 设备
            x2 = torch.load(f, map_location="cpu")

            # 使用断言验证 x 和 x2 的值相等
            self.assertEqual(x, x2)
            # 使用断言验证 x2 的设备类型为 "cpu"

        # 确保可以将 "mps:0" 上的张量加载到 "mps" 设备上
        with tempfile.NamedTemporaryFile() as f:
            # 创建一个包含随机数的张量 x，在 "mps:0" 设备上
            x = torch.rand(2, device="mps:0")
            # 将张量 x 保存到临时文件 f 中
            torch.save(x, f)

            # 重置文件指针位置到开头
            f.seek(0)
            # 从文件 f 中加载张量 x2，并将其映射到 "mps" 设备
            x2 = torch.load(f, map_location="mps:0")

            # 使用断言验证 x 和 x2 的值相等
            self.assertEqual(x, x2)
            # 使用断言验证 x2 的设备类型为 "mps"
# 获取所有支持的数据类型列表
MPS_DTYPES = get_all_dtypes()

# 从MPS_DTYPES中删除指定的数据类型
for t in [torch.double, torch.cdouble, torch.cfloat, torch.bfloat16]:
    del MPS_DTYPES[MPS_DTYPES.index(t)]

# 定义 MPS_GRAD_DTYPES 列表，包含特定的浮点数数据类型
MPS_GRAD_DTYPES = [torch.float32, torch.float16]

# 定义测试类 TestConsistency，继承自 TestCaseMPS 类
class TestConsistency(TestCaseMPS):

    # TODO: This is only used while some ops are being added.
    # This list should contain all ops and dtypes eventually
    # This can be generated automatically in the `new_mps_allowlist.txt` file
    # by doing `EXPECTTEST_ACCEPT=1 python test_mps.py TestConsistencyCPU`
    # You most likely do NOT want to modify this manually

    # 定义 FP16_LOW_PRECISION_LIST，包含需要使用低精度 FP16 运行的操作列表
    FP16_LOW_PRECISION_LIST = {
        'add', 'sub', 'div', 'addcdiv',
        '__rdiv__', '__rmul__',
        'nn.functional.huber_loss',
        'true_divide', 'kron',
        'gradient', 'var', 'std', 'std_mean', 'ldexp',
        'linalg.vector_norm', 'lerp',
        'addr', 'var_mean',
        'var_mean_unbiased',
        'acosh', 'asinh', 'asin',
        'masked.std',
        'nn.functional.normalize',
        'nn.functional.triplet_margin_loss',
        'nn.functional.triplet_margin_with_distance_loss',
        'nn.functional.batch_norm',
        'nn.functional.instance_norm',
        'round', 'xlogy', 'addcmul',
        'nn.functional.cross_entropy',
        'nn.functional.binary_cross_entropy',
        'nn.functional.nll_loss',
        'nn.functional.max_pool2d',
        'nn.functional.gelu',
        'nn.functional.glu',
        '_native_batch_norm_legit',
        '_batch_norm_with_update',
        'native_batch_norm',
        'softmax',
        '_softmax_backward_data',
        'log_softmax',
        'masked.softmax',
        'masked.log_softmax',
        'masked.softmin',
        'nn.functional.kl_div',
        'nn.functional.softmin',
        'cross', 'linalg.cross',
        'prod', 'masked.prod',
        'nextafter',
        'native_layer_norm',
        'nn.functional.layer_norm',
        'nn.functional.interpolate',
        'nn.functional.upsample_bilinear',
        'nn.functional.upsample_nearest',

        # for macOS 12
        'masked.normalize', 'masked.sum', 'masked.var',
        'outer',
        'sum_to_size', 'sum',
        'mul',
        'nansum', 'nanmean',
        'norm',
    }

    # 定义 FP32_LOW_PRECISION_LIST，包含需要使用低精度 FP32 运行的操作列表
    FP32_LOW_PRECISION_LIST = {
        # conv2d and conv_transpose2d results have a very small
        # difference compared to CPU/CUDA, so we use lower precision on FP32
        'nn.functional.conv2d',
        'nn.functional.conv_transpose2d',
        'matmul', '__rmatmul__',
        'linalg.multi_dot',
        'addbmm',
    }
    # 根据操作和数据类型计算容差值，用于断言比较
    def _compute_tolerances(self, op, dtype):
        # 如果操作在FP32低精度列表中，并且数据类型是torch.float32或torch.complex64
        if (op.name in self.FP32_LOW_PRECISION_LIST) and dtype in [torch.float32, torch.complex64]:
            return (1e-4, 3e-5)

        # 如果操作在FP16低精度列表中，并且数据类型是torch.float16
        if op.name in self.FP16_LOW_PRECISION_LIST and dtype == torch.float16:
            return (1e-2, 1e-2)

        # 如果操作名称在指定列表中，并且数据类型是torch.float16
        if op.name in ['nn.functional.conv_transpose1d',
                       'nn.functional.conv_transpose2d',
                       'nn.functional.conv_transpose3d',
                       '__rmatmul__', 'addbmm', 'addmv',
                       'baddbmm', 'cov', 'matmul', 'mv'] and dtype == torch.float16:
            return (5e-2, 5e-2)

        # 如果操作名称为"masked.mean"
        if op.name == "masked.mean":
            return (7e-4, 2e-3)

        # 如果操作名称为"native_layer_norm"
        if op.name == "native_layer_norm":
            return (1e-4, 1.3e-5)

        # 如果操作名称在指定列表中，并且产品版本小于13.3
        if op.name in ["pow", "__rpow__"] and product_version < 13.3:
            # 返回特定的容差值，根据数据类型选择不同的rtol值
            return (1e-6, 2e-3 if dtype == torch.float16 else 4e-6)

        # 如果操作名称为"nn.functional.interpolate"
        if op.name == "nn.functional.interpolate":
            return (1e-3, 1e-4)

        # 如果操作名称在傅立叶变换相关函数列表中
        if op.name in ['fft.rfftn', 'fft.hfftn', 'fft.hfft2', 'fft.fft', 'fft.fftn', 'fft.rfft']:
            # TODO: Investigate why this is needed（为什么需要这样做）
            # 参见https://github.com/pytorch/pytorch/issues/120237
            return (3e-5, 3e-5)

        # 默认情况下返回None，表示无特定容差要求
        return (None, None)

    # 仅用于接受模式
    NEW_ALLOW_LIST = defaultdict(list)
    NEW_ALLOW_LIST_GRAD = defaultdict(list)

    # 使用ops修饰器，测试输出是否匹配
    @ops(mps_ops_modifier(test_consistency_op_db), allowed_dtypes=MPS_DTYPES + [torch.complex64])
    def test_output_match(self, device, dtype, op):
        # 断言设备为CPU
        self.assertEqual(device, "cpu")

        # 定义获取样本数据的函数
        def get_samples():
            return op.sample_inputs(device, dtype, requires_grad=(dtype.is_floating_point or dtype.is_complex))

        # 获取CPU上的样本数据
        cpu_samples = get_samples()

        # 遍历CPU样本数据
        for cpu_sample in cpu_samples:
            #
            # 前向检查
            #
            # 将CPU样本转换为MPS样本，保持梯度属性不变（如果是Tensor）
            mps_sample = cpu_sample.transform(
                lambda x: x.detach().to("mps").requires_grad_(x.requires_grad) if isinstance(x, torch.Tensor) else x)

            # 分别组装CPU和MPS的参数和关键字参数
            cpu_args = [cpu_sample.input] + list(cpu_sample.args)
            cpu_kwargs = cpu_sample.kwargs
            mps_args = [mps_sample.input] + list(mps_sample.args)
            mps_kwargs = mps_sample.kwargs

            # 对于tensor_split()操作，第二个参数必须在CPU上
            if op.name == "tensor_split" and isinstance(mps_args[1], torch.Tensor):
                mps_args[1] = cpu_args[1]

            # 在CPU和MPS上执行操作，并获取输出
            cpu_out = op(*cpu_args, **cpu_kwargs)
            mps_out = op(*mps_args, **mps_kwargs)

            # 计算允许的绝对误差和相对误差
            atol, rtol = self._compute_tolerances(op, dtype)

            # 对于特定操作"nn.functional.upsample_bilinear"且数据类型为torch.uint8，使用特定的容差值
            if op.name == "nn.functional.upsample_bilinear" and dtype == torch.uint8:
                atol = 1.0
                rtol = 0.0

            # 断言CPU输出与MPS输出相等，使用给定的容差值
            self.assertEqual(cpu_out, mps_out, atol=atol, rtol=rtol)
    # 使用装饰器 @ops 对下面的函数进行修饰，传入参数为 mps_ops_grad_modifier(copy.deepcopy(test_consistency_op_db)) 和 allowed_dtypes=MPS_GRAD_DTYPES
    @ops(mps_ops_grad_modifier(copy.deepcopy(test_consistency_op_db)), allowed_dtypes=MPS_GRAD_DTYPES)
class TestErrorInputs(TestCase):
    _ignore_not_implemented_error = True  # 设置一个类属性，用于指示忽略未实现的错误

    @ops(mps_ops_error_inputs_modifier(test_error_inputs_op_db), dtypes=OpDTypes.none)
    def test_error_inputs(self, device, op):
        self.assertEqual(device, "mps:0")  # 断言设备名称必须是 "mps:0"

        mps_samples = op.error_inputs(device)  # 调用被测试操作的 error_inputs 方法，获取错误输入样本列表

        for mps_sample in mps_samples:  # 遍历每个错误输入样本
            mps_sample_input = mps_sample.sample_input  # 获取样本输入
            error_type = mps_sample.error_type  # 获取预期的错误类型
            error_regex = mps_sample.error_regex  # 获取用于匹配错误消息的正则表达式

            mps_args = [mps_sample_input.input] + list(mps_sample_input.args)  # 准备调用操作的参数列表
            mps_kwargs = mps_sample_input.kwargs  # 获取关键字参数

            # 对于 tensor_split() 操作，第二个张量参数 ("tensor_indices_or_sections") 必须在 CPU 上
            if (op.name == "tensor_split" and isinstance(mps_args[1], torch.Tensor)):
                mps_args[1] = mps_args[1].cpu()  # 如果是张量，则将其移到 CPU 上

            with self.assertRaisesRegex(error_type, error_regex):
                op(*mps_args, **mps_kwargs)  # 断言调用操作会引发预期的错误类型和消息

class TestComplex(TestCase):
    def test_tensor_scalar_binops(self):
        # 回归测试 https://github.com/pytorch/pytorch/issues/119088 的案例
        def to_cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) else x  # 如果是张量则移到 CPU 上，否则返回原对象

        # 在 mps 上分配张量
        with torch.device("mps"):
            inputs = [torch.rand(2, dtype=dtype) for dtype in [torch.float, torch.half, torch.cfloat]]
        self.assertTrue(all(x.device.type == "mps" for x in inputs))  # 断言所有输入张量都在 MPS 设备上
        # 添加标量
        inputs.extend([7, 3.14, 2 + 3j, torch.tensor(4 + 5j, dtype=torch.chalf)])

        # 遍历所有类型（整数、浮点数、复数、半精度浮点数）和操作（加法、减法、乘法）的排列组合
        for x, y in itertools.product(inputs, inputs):
            for op_name in ["__add__", "__sub__", "__mul__"]:
                x_cpu, y_cpu = map(to_cpu, (x, y))
                res = getattr(x, op_name)(y)  # 执行操作
                res_cpu = getattr(x_cpu, op_name)(y_cpu)  # 在 CPU 上执行相同操作
                self.assertEqual(to_cpu(res), res_cpu, f"{op_name}({x}, {y}) 产生不同的结果 {res} vs {res_cpu}")

# 从 `test_ops.py` 的 `TestCommon` 复制，以便为 MPS 复制 `test_numpy_ref` 的测试
@skipIfSlowGradcheckEnv
class TestCommon(TestCase):
    exact_dtype = True  # 设置确切的数据类型标志

    # 在类销毁时检查，确保没有 OpInfo 在 CI 中仍然使用动态数据类型
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_CI:
            err_msg = (
                "The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                "This is OK for testing, but be sure to set the dtypes manually before landing your PR!"
            )
            # 确保没有 OpInfo 条目使用动态数据类型
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg  # 断言没有 OpInfo 使用动态数据类型
    # 这是 `test_ops.py` 中 `test_numpy_ref` 的 MPS 版本的等效代码。由于 MPS 在测试框架中仍然需要一些特殊处理，
    # 因此这段代码被放置在这里。
    # 当 MPS 变得更加一致时，可以考虑使用 `@dtypesIfMPS(torch.float32)` 将此测试与原始测试合并，
    # 但目前需要放宽断言本身的条件。

    @suppress_warnings
    # MPS 仅支持 float32 类型
    @ops(_ref_test_ops, allowed_dtypes=(torch.float32,))
    def test_numpy_ref_mps(self, device, dtype, op):
        # 不像 `test_numpy_ref`，这个测试在创建时比较的是 `float32` 类型，
        # 因为在创建该测试时，MPS 不支持 float64 张量。
        # 一些操作目前在其参考输入上存在问题，但在示例输入上没有问题。这些问题应该会得到修复，此时的解决方法可以移除。
        broken_on_ref_inputs = op.name in ['clamp', 'where']
        if not broken_on_ref_inputs:
            inputs = op.reference_inputs(device, dtype)
        else:
            inputs = op.sample_inputs(device, dtype)
        
        # 遍历所有输入样本
        for sample_input in inputs:
            self.compare_with_reference(op, op.ref, sample_input)

    @dtypes(*get_all_dtypes())
    def test_tensor_creation(self, device, dtype):
        # 定义一个创建全1张量的函数
        def ones(device):
            return torch.ones((2, 2), dtype=dtype, device=device)
        
        # 如果数据类型不在 MPS_DTYPES 中，且产品版本大于 14.0，则预期抛出 TypeError 异常
        if dtype not in MPS_DTYPES + ([torch.bfloat16, torch.complex64] if product_version > 14.0 else [torch.complex64]):
            with self.assertRaises(TypeError):
                ones(device)
        else:
            # 在 MPS 上创建张量
            mps_tensor = ones(device)
            # 在 CPU 上创建相同的张量
            cpu_tensor = ones("cpu")
            # 断言 MPS 上的张量与 CPU 上的张量相等
            self.assertEqual(mps_tensor.cpu(), cpu_tensor)
# 在 "mps" 设备上实例化 TestConsistency 测试，以更好地反映其功能。
# 目前，需要在设备通用测试框架中正确注册 mps，但目前尚未完成。
# 可能可以利用 https://github.com/pytorch/pytorch/pull/87342 中引入的 `allow_mps` 来实现这一点。
instantiate_device_type_tests(TestConsistency, globals(), only_for="cpu")

# 在 "mps" 设备上实例化 TestErrorInputs 测试，允许使用 MPS。
instantiate_device_type_tests(TestErrorInputs, globals(), allow_mps=True, only_for="mps")

# 在 "mps" 设备上实例化 TestCommon 测试，允许使用 MPS。
instantiate_device_type_tests(TestCommon, globals(), allow_mps=True, only_for="mps")

# 在 "mps" 设备上实例化 TestLinalgMPS 测试，允许使用 MPS。
instantiate_device_type_tests(TestLinalgMPS, globals(), allow_mps=True, only_for="mps")

# 如果当前脚本作为主程序运行，则执行所有测试。
if __name__ == "__main__":
    run_tests()
```