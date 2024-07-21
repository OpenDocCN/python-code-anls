# `.\pytorch\torch\testing\_internal\common_nn.py`

```
# mypy: ignore-errors

# 从 abc 模块导入 abstractmethod 抽象方法
from abc import abstractmethod
# 导入临时文件模块
import tempfile
# 导入单元测试框架模块
import unittest

# 从 copy 模块导入 deepcopy 深拷贝函数
from copy import deepcopy
# 从 functools 模块导入 reduce 函数和 partial 函数
from functools import reduce, partial
# 从 itertools 模块导入 product 函数
from itertools import product
# 从 operator 模块导入 mul 函数
from operator import mul

# 导入 PyTorch 模块
import torch
# 导入 PyTorch CUDA 支持模块
import torch.cuda
# 导入 PyTorch 的神经网络模块和函数模块
import torch.nn as nn
import torch.nn.functional as F
# 导入 PyTorch 的 _Reduction 模块
from torch.nn import _reduction as _Reduction
# 从 torch.testing._internal.common_utils 模块导入多个函数和类
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
    gradcheck, gradgradcheck, set_default_dtype, skipIfTorchDynamo
# 从 torch.testing._internal.common_cuda 模块导入测试 CUDA 相关函数和类
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
# 从 torch.autograd.gradcheck 模块导入数值梯度检查相关函数
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
# 从 torch.autograd 模块导入 Variable 类
from torch.autograd import Variable
# 从 torch.types 模块导入 _TensorOrTensors 类型
from torch.types import _TensorOrTensors
# 导入 PyTorch 的 cudnn 后端模块
import torch.backends.cudnn

# 从 typing 模块导入多种类型注解
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any

# 定义 TemporaryFile 类为 tempfile 模块中的 TemporaryFile 类
TemporaryFile = tempfile.TemporaryFile
# 定义 PRECISION 常量为 1e-5
PRECISION = 1e-5


# 定义函数 get_reduction，参数为 m
def get_reduction(m):
    # 获取 m 对象的 reduction 属性，如果不存在则返回 None
    result = getattr(m, 'reduction', None)
    if result is None:
        # 如果 result 为 None，则通过 legacy_get_string 方法获取 sizeAverage 属性的字符串表示
        result = _Reduction.legacy_get_string(getattr(m, 'sizeAverage', None), True, emit_warning=False)
    # 断言 result 不为 None
    assert result is not None
    return result


# 定义函数 get_weight，参数为 m
def get_weight(m):
    # 获取 m 对象的 weight 属性，如果存在则返回其值
    result = getattr(m, 'weight', None)
    if result is not None:
        return result
    # 否则返回 m 对象的 weights 属性值（如果存在）
    return getattr(m, 'weights', None)

# NOTE [How to check NN module / functional API parity between Python and C++ frontends]
#
# The way to check API parity is to add parity tests for the NN module / functional of interest.
# Here are the detailed steps:
#
# For NN module:
# 1. Make sure you already have a test dict with the module configuration you want to test.
# 2. Add `cpp_constructor_args` entry to the test dict, with its value exactly matching
#    the Python module constructor arguments. For example, if in the test dict we pass
#    `(10, 8)` to `torch.nn.Linear` constructor, then we should pass `torch::nn::LinearOptions(10, 8)`
#    as the corresponding C++ constructor argument to `torch::nn::Linear`.
# 3. If in the process of performing the above step you referenced any variables
#    in the `cpp_constructor_args` entry, you must add `cpp_var_map` entry
#    to the test dict to make sure that those variables are populated with the right Python values.
#    For example, if the Python constructor call is
#    `torch.nn.FractionalMaxPool2d(2, output_ratio=0.5, _random_samples=random_samples)`,
#    the corresponding C++ constructor argument is
#    `torch::nn::FractionalMaxPool2dOptions(2).output_ratio(0.5)._random_samples(random_samples)`,
#    and the `cpp_var_map` entry must be
#    `{'random_samples': random_samples}` in order to populate the C++ variable `random_samples`
#    used in the C++ constructor argument with the Python tensor value `random_samples`.
#
# For NN functional:
# 1. Make sure you already have a test dict with the functional configuration you want to test.
# 2. If the test dict's `constructor` entry looks like `wrap_functional(F.some_functional_name, ...)`,
module_tests = [
    dict(
        module_name='Linear',  # 模块名称为 'Linear'
        constructor_args=(10, 8),  # 构造函数的参数是 (10, 8)
        cpp_constructor_args='torch::nn::LinearOptions(10, 8)',  # 对应的 C++ 构造函数参数字符串
        input_size=(4, 10),  # 输入数据的大小为 (4, 10)
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8),  # 参考函数的定义
        with_tf32=True,  # 使用 TF32 精度计算
        tf32_precision=0.005,  # TF32 精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为 torch.double
        test_cpp_api_parity=True,  # 执行 C++ 接口的一致性测试，默认为 True
        has_parity=True,  # 期望 C++ 接口测试通过，默认为 True
    ),
]
    # 创建包含模块信息的字典
    dict(
        module_name='Linear',  # 模块名称为 'Linear'
        constructor_args=(10, 8, False),  # 构造函数参数为 (10, 8, False)
        cpp_constructor_args='torch::nn::LinearOptions(10, 8).bias(false)',  # 对应的 C++ 构造函数参数
        input_size=(4, 10),  # 输入大小为 (4, 10)
        desc='no_bias',  # 描述为 'no_bias'
        reference_fn=lambda i, p, _: torch.mm(i, p[0].t()),  # 参考函数为矩阵乘法 torch.mm(i, p[0].t())
        with_tf32=True,  # 支持 TF32
        tf32_precision=0.005,  # TF32 精度为 0.005
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    # 第二个模块信息字典
    dict(
        module_name='RReLU',  # 模块名称为 'RReLU'
        input_size=(1, 2, 2),  # 输入大小为 (1, 2, 2)
        test_cuda=False,  # 不测试 CUDA
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    # 第三个模块信息字典
    dict(
        module_name='RReLU',  # 模块名称为 'RReLU'
        constructor_args=(0.1, 0.9),  # 构造函数参数为 (0.1, 0.9)
        cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)',  # 对应的 C++ 构造函数参数
        input_size=(4, 4, 5),  # 输入大小为 (4, 4, 5)
        desc='with_up_down',  # 描述为 'with_up_down'
        test_cuda=False,  # 不测试 CUDA
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    # 第四个模块信息字典
    dict(
        module_name='Flatten',  # 模块名称为 'Flatten'
        input_size=(2, 3, 4, 5),  # 输入大小为 (2, 3, 4, 5)
        reference_fn=lambda i, *_: torch.flatten(i, 1),  # 参考函数为 torch.flatten(i, 1)
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    # TODO: 参考函数待实现
    # 第五个模块信息字典，标记为待完成
    dict(
        module_name='CrossMapLRN2d',  # 模块名称为 'CrossMapLRN2d'
        constructor_args=(5, 5e-3, 1e-3, 2),  # 构造函数参数为 (5, 5e-3, 1e-3, 2)
        cpp_constructor_args='torch::nn::CrossMapLRN2dOptions(5).alpha(5e-3).beta(1e-3).k(2)',  # 对应的 C++ 构造函数参数
        input_size=(2, 3, 6, 6),  # 输入大小为 (2, 3, 6, 6)
        check_gradgrad=False,  # 不检查 gradgrad
        # TODO(#50743): 解决错误。"RuntimeError: Unrecognized tensor type ID: Batched"
        check_batched_grad=False,  # 不检查批量化梯度
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
# 生成一个具有不相等值的随机张量。这样做可以确保重复值不会导致像最大池化这样的模块测试失败。
# size 应该小一些，否则 randperm 会失败 / 长整型会溢出。
def _rand_tensor_non_equal(*size):
    # 计算张量的总元素数
    total = reduce(mul, size, 1)
    # 使用 torch.randperm 生成随机的排列张量，并将其视图化成指定大小
    return torch.randperm(total).view(*size).double()


def wrap_functional(fn, **kwargs):
    # 定义一个继承自 nn.Module 的函数式模块，用于包装传入的函数
    class FunctionalModule(nn.Module):
        def forward(self, *args):
            return fn(*args, **kwargs)
    return FunctionalModule


def poissonnllloss_no_reduce_test():
    # 生成一个随机的 10x10 张量
    t = torch.randn(10, 10)
    return dict(
        fullname='PoissonNLLLoss_no_reduce',
        # 使用 wrap_functional 包装的匿名函数，调用 F.poisson_nll_loss 函数，reduction 设置为 'none'
        constructor=wrap_functional(
            lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用字符串
        cpp_function_call='F::poisson_nll_loss('
                          'i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))',
        # 生成一个返回随机 10x10 张量的函数
        input_fn=lambda: torch.rand(10, 10),
        # C++ 变量映射，i 映射为 '_get_input()', t 映射为上面定义的随机张量
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，计算指数和差值
        reference_fn=lambda i, *_: i.exp() - t.mul(i),
        pickle=False,
        default_dtype=torch.double)


def bceloss_no_reduce_test():
    # 生成一个随机的 15x10 张量，经过逻辑判断后转换为双精度类型
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    return dict(
        fullname='BCELoss_no_reduce',
        # 使用 wrap_functional 包装的匿名函数，调用 F.binary_cross_entropy 函数，reduction 设置为 'none'
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用字符串
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))',
        # 生成一个返回在特定范围内的随机 15x10 张量的函数
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        # C++ 变量映射，i 映射为 '_get_input()', t 映射为上面定义的随机张量
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，计算二进制交叉熵损失
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False,
        precision=7e-4,
        default_dtype=torch.double)


def bceloss_no_reduce_scalar_test():
    # 生成一个标量的随机张量，经过逻辑判断后转换为双精度类型
    t = torch.randn(()).gt(0).to(torch.double)
    return dict(
        fullname='BCELoss_no_reduce_scalar',
        # 使用 wrap_functional 包装的匿名函数，调用 F.binary_cross_entropy 函数，reduction 设置为 'none'
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用字符串
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))',
        # 生成一个返回在特定范围内的随机标量张量的函数
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        # C++ 变量映射，i 映射为 '_get_input()', t 映射为上面定义的随机张量
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，计算二进制交叉熵损失
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()),
        pickle=False,
        default_dtype=torch.double)


def bceloss_weights_no_reduce_test():
    # 生成一个随机的 15x10 张量，经过逻辑判断后转换为双精度类型
    t = Variable(torch.randn(15, 10, dtype=torch.double).gt(0).to(torch.double))
    # 生成一个随机权重张量
    weights = torch.rand(10, dtype=torch.double)
    # 返回一个包含多个键值对的字典
    return dict(
        # 键名为'fullname'，对应值为'BCELoss_weights_no_reduce'
        fullname='BCELoss_weights_no_reduce',
        # 键名为'constructor'，对应值为一个使用wrap_functional包装的lambda函数
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        # 键名为'cpp_function_call'，对应值为一个C++函数调用的字符串表示
        cpp_function_call='F::binary_cross_entropy('
                          'i, t.to(i.options()), '
                          'F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))',
        # 键名为'input_fn'，对应值为一个lambda函数，返回一个15x10的随机张量，限制在[2.8e-2, 1-2.8e-2]之间
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        # 键名为'cpp_var_map'，对应值为一个字典，包含键值对{'i': '_get_input()', 't': t, 'weights': weights}
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        # 键名为'reference_fn'，对应值为一个lambda函数，计算二元交叉熵损失函数的参考值
        reference_fn=lambda i, p, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        # 键名为'pickle'，对应值为False，表示此字典不会被pickle序列化
        pickle=False,
        # 键名为'precision'，对应值为0.0003，表示精度为3e-4
        precision=3e-4,
        # 键名为'default_dtype'，对应值为torch.double，表示张量的默认数据类型为双精度浮点数
        default_dtype=torch.double,
    )
# 返回一个字典，包含测试 BCELoss_weights_no_reduce_scalar 的相关信息和设置
def bceloss_weights_no_reduce_scalar_test():
    # 创建一个随机张量，元素大于零，并转换为双精度类型
    t = torch.randn(()).gt(0).to(torch.double)
    # 创建一个随机的双精度标量作为权重
    weights = torch.rand((), dtype=torch.double)
    # 返回一个字典，包含完整名称、构造函数、C++函数调用等信息
    return dict(
        fullname='BCELoss_weights_no_reduce_scalar',
        # 使用 wrap_functional 封装的 lambda 函数，用于计算二元交叉熵
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy(i, t.type_as(i),
                                             weight=weights.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用，使用 torch::kNone 表示无缩减模式
        cpp_function_call='''F::binary_cross_entropy(
            i, t.to(i.options()),
            F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        # 定义了在 C++ 中变量映射的字典
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},
        # 返回一个随机张量的函数，元素范围在 (2.8e-2, 1 - 2.8e-2) 之间
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        # 参考函数，用于计算 BCELoss 的参考结果
        reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
        pickle=False,  # 禁用 pickle
        default_dtype=torch.double,  # 默认数据类型为双精度
    )


# 返回一个字典，包含测试 BCEWithLogitsLoss_legacy_enum 的相关信息和设置
def bce_with_logistic_legacy_enum_test():
    # 创建一个变量，形状为 (15, 10)，元素大于零，并转换为双精度类型
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    sigmoid = nn.Sigmoid()
    # 返回一个字典，包含完整名称、构造函数等信息
    return dict(
        fullname='BCEWithLogitsLoss_legacy_enum',
        # 使用 wrap_functional 封装的 lambda 函数，用于计算带 logits 的二元交叉熵
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduce=False)),
        # 对应的 C++ 函数调用，使用 torch::kNone 表示无缩减模式
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        # 返回一个随机张量的函数，元素范围在 (2.8e-2, 1 - 2.8e-2) 之间
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        # 定义了在 C++ 中变量映射的字典
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，用于计算 BCEWithLogitsLoss 的参考结果
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,  # 禁用梯度梯度检查
        pickle=False,  # 禁用 pickle
        default_dtype=torch.double,  # 默认数据类型为双精度
    )


# 返回一个字典，包含测试 BCEWithLogitsLoss_no_reduce 的相关信息和设置
def bce_with_logistic_no_reduce_test():
    # 创建一个变量，形状为 (15, 10)，元素大于零，并转换为双精度类型
    t = Variable(torch.randn(15, 10).gt(0).to(torch.double))
    sigmoid = nn.Sigmoid()
    # 返回一个字典，包含完整名称、构造函数等信息
    return dict(
        fullname='BCEWithLogitsLoss_no_reduce',
        # 使用 wrap_functional 封装的 lambda 函数，用于计算带 logits 的二元交叉熵，无缩减模式
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用，使用 torch::kNone 表示无缩减模式
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        # 返回一个随机张量的函数，元素范围在 (2.8e-2, 1 - 2.8e-2) 之间
        input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
        # 定义了在 C++ 中变量映射的字典
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，用于计算 BCEWithLogitsLoss 的参考结果
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        check_gradgrad=False,  # 禁用梯度梯度检查
        pickle=False,  # 禁用 pickle
        default_dtype=torch.double,  # 默认数据类型为双精度
    )


# 定义一个函数，但未提供具体的返回值和注释
def bce_with_logistic_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).to(torch.double)
    sigmoid = nn.Sigmoid()
    # 返回一个包含各种参数和选项的字典
    return dict(
        # 定义损失函数的全名
        fullname='BCEWithLogitsLoss_no_reduce_scalar',
        # 构造函数，使用函数包装器wrap_functional将lambda函数转换为函数对象
        constructor=wrap_functional(
            lambda i: F.binary_cross_entropy_with_logits(i, t.type_as(i), reduction='none')),
        # C++函数调用字符串表示，用于说明相应的C++函数调用
        cpp_function_call='''F::binary_cross_entropy_with_logits(
            i, t.to(i.options()), F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone))''',
        # 输入函数，返回一个随机数张量并进行范围限制
        input_fn=lambda: torch.rand(()).clamp_(2.8e-2, 1 - 2.8e-2),
        # C++变量映射，将'i'映射为'_get_input()'，'t'映射为变量t
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，计算损失的参考实现
        reference_fn=lambda i, *_: -(t * sigmoid(i).log() + (1 - t) * (1 - sigmoid(i)).log()),
        # 是否检查梯度的梯度
        check_gradgrad=False,
        # 是否支持pickle序列化
        pickle=False,
        # 默认数据类型为torch.double
        default_dtype=torch.double,
    )
# 定义一个测试函数，用于测试 Kullback-Leibler 散度（KL 散度）的损失函数，不进行降维操作
def kldivloss_with_target_no_reduce_test():
    # 创建一个 10x10 的双精度随机张量 t
    t = torch.rand(10, 10, dtype=torch.double)
    # 返回一个字典，包含测试信息和配置
    return dict(
        fullname='KLDivLoss_with_target_no_reduce',  # 完整名称标识损失函数及其设置
        constructor=wrap_functional(  # 使用包装函数构造一个函数，该函数对输入 i 执行 KL 散度计算，目标张量为 t
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.rand(10, 10).log(),  # 返回一个 10x10 的随机张量的对数
        cpp_var_map={'i': '_get_input()', 't': t},  # 定义 C++ 变量映射表，i 表示输入，t 表示目标张量
        reference_fn=lambda i, *_:  # 参考函数，调用 loss_reference_fns 中的 KLDivLoss 函数进行 KL 散度计算
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,  # 支持前向自动微分
        pickle=False,  # 不支持 pickle
        default_dtype=torch.double  # 默认张量类型为双精度
    )


# 定义一个测试函数，测试不带降维的 KL 散度损失函数
def kldivloss_no_reduce_test():
    # 创建一个 10x10 的双精度随机张量 t
    t = torch.rand(10, 10, dtype=torch.double)
    # 返回一个字典，包含测试信息和配置
    return dict(
        fullname='KLDivLoss_no_reduce',  # 完整名称标识损失函数及其设置
        constructor=wrap_functional(  # 使用包装函数构造一个函数，该函数对输入 i 执行 KL 散度计算，目标张量为 t
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.rand(10, 10).log(),  # 返回一个 10x10 的随机张量的对数
        cpp_var_map={'i': '_get_input()', 't': t},  # 定义 C++ 变量映射表，i 表示输入，t 表示目标张量
        reference_fn=lambda i, *_:  # 参考函数，调用 loss_reference_fns 中的 KLDivLoss 函数进行 KL 散度计算
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,  # 支持前向自动微分
        pickle=False,  # 不支持 pickle
        default_dtype=torch.double  # 默认张量类型为双精度
    )


# 定义一个测试函数，测试不带降维的 KL 散度损失函数，其中目标张量为标量
def kldivloss_no_reduce_scalar_test():
    # 创建一个标量的双精度随机张量 t
    t = torch.rand((), dtype=torch.double)
    # 返回一个字典，包含测试信息和配置
    return dict(
        fullname='KLDivLoss_no_reduce_scalar',  # 完整名称标识损失函数及其设置
        constructor=wrap_functional(  # 使用包装函数构造一个函数，该函数对输入 i 执行 KL 散度计算，目标张量为 t
            lambda i: F.kl_div(i, t.type_as(i), reduction='none')),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.rand(()).log(),  # 返回一个标量的对数
        cpp_var_map={'i': '_get_input()', 't': t},  # 定义 C++ 变量映射表，i 表示输入，t 表示目标张量
        reference_fn=lambda i, *_:  # 参考函数，调用 loss_reference_fns 中的 KLDivLoss 函数进行 KL 散度计算
            loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,  # 支持前向自动微分
        pickle=False,  # 不支持 pickle
        default_dtype=torch.double  # 默认张量类型为双精度
    )


# 定义一个测试函数，测试带有对数目标的不带降维的 KL 散度损失函数
def kldivloss_with_log_target_no_reduce_test():
    # 创建一个对数目标的 10x10 的双精度随机张量 t
    t = torch.rand(10, 10, dtype=torch.double).log()
    # 返回一个字典，包含测试信息和配置
    return dict(
        fullname='KLDivLoss_with_log_target_no_reduce',  # 完整名称标识损失函数及其设置
        constructor=wrap_functional(  # 使用包装函数构造一个函数，该函数对输入 i 执行 KL 散度计算，目标张量为 t，且目标已经是对数
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.rand(10, 10).log(),  # 返回一个 10x10 的随机张量的对数
        cpp_var_map={'i': '_get_input()', 't': t},  # 定义 C++ 变量映射表，i 表示输入，t 表示目标张量
        reference_fn=lambda i, *_:  # 参考函数，调用 loss_reference_fns 中的 KLDivLoss_log_target 函数进行 KL 散度计算
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        supports_forward_ad=True,  # 支持前向自动微分
        pickle=False,  # 不支持 pickle
        default_dtype=torch.double  # 默认张量类型为双精度
    )


# 定义一个测试函数，测试带有对数目标的不带降维的 KL 散度损失函数，其中目标张量为标量
def kldivloss_no_reduce_log_target_test():
    t = torch.rand(10, 10, dtype=torch.double).log()
    # 返回一个字典，包含以下键值对：

    fullname='KLDivLoss_no_reduce_log_target',
        # 键名：fullname，字符串值：'KLDivLoss_no_reduce_log_target'

    constructor=wrap_functional(
        lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        # 键名：constructor，值是一个通过 wrap_functional 包装的匿名函数，
        # 函数功能：计算输入张量 i 和目标张量 t 的 KL 散度，设置不进行减少，使用对数目标

    cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        # 键名：cpp_function_call，字符串值：'F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))'
        # 这是一个 C++ 函数调用的示例，用于描述与 PyTorch 中 kl_div 函数的对应关系

    input_fn=lambda: torch.rand(10, 10).log(),
        # 键名：input_fn，值是一个匿名函数，生成一个大小为 10x10 的随机张量，并取其对数

    cpp_var_map={'i': '_get_input()', 't': t},
        # 键名：cpp_var_map，值是一个字典，映射变量 i 到字符串 '_get_input()'，映射变量 t 到实际的张量对象 t

    reference_fn=lambda i, *_:
        loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        # 键名：reference_fn，值是一个匿名函数，调用 loss_reference_fns 中的 KLDivLoss_log_target 函数，
        # 传入参数 i，t.type_as(i)，reduction='none'

    supports_forward_ad=True,
        # 键名：supports_forward_ad，布尔值：True，指示该函数支持前向自动求导

    pickle=False,
        # 键名：pickle，布尔值：False，指示不支持对象的 pickle 操作

    default_dtype=torch.double,
        # 键名：default_dtype，张量的默认数据类型为 torch.double
# 定义了一个函数，用于测试没有减少操作的 Kullback-Leibler 散度损失函数，针对标量 log_target
def kldivloss_no_reduce_scalar_log_target_test():
    # 生成一个随机的双精度张量，并取其对数值，作为 t 的值
    t = torch.rand((), dtype=torch.double).log()
    # 返回一个字典，包含函数的全名、构造函数、C++ 函数调用、输入函数、C++ 变量映射、参考函数、是否支持前向自动求导、是否可序列化、默认数据类型
    return dict(
        fullname='KLDivLoss_no_reduce_scalar_log_target',
        constructor=wrap_functional(
            # 使用 wrap_functional 包装的匿名函数，返回 kl_div 函数的调用结果，传入输入 i、t，设置 reduction='none' 和 log_target=True
            lambda i: F.kl_div(i, t.type_as(i), reduction='none', log_target=True)),
        # 对应的 C++ 函数调用，使用 t.to(i.options()) 转换 t 并调用 kl_div 函数，配置 reduction='none' 和 log_target=true
        cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone).log_target(true))',
        # 输入函数生成一个随机标量张量并取对数
        input_fn=lambda: torch.rand(()).log(),
        # C++ 变量映射，将 'i' 映射到 '_get_input()'，将 't' 映射到之前定义的 t
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，返回使用 'KLDivLoss_log_target' 函数计算的损失，传入 i、t，设置 reduction='none'
        reference_fn=lambda i, *_:
            loss_reference_fns['KLDivLoss_log_target'](i, t.type_as(i), reduction='none'),
        # 支持前向自动求导
        supports_forward_ad=True,
        # 不支持序列化
        pickle=False,
        # 默认数据类型为双精度
        default_dtype=torch.double)


# 定义了一个函数，用于测试没有减少操作的 L1 损失函数
def l1loss_no_reduce_test():
    # 生成一个随机的双精度张量作为 t 的值，形状为 (2, 3, 4)
    t = torch.randn(2, 3, 4, dtype=torch.double)
    # 返回一个字典，包含函数的全名、构造函数、C++ 函数调用、输入函数、C++ 变量映射、参考函数、是否支持前向自动求导、是否可序列化、默认数据类型
    return dict(
        fullname='L1Loss_no_reduce',
        constructor=wrap_functional(
            # 使用 wrap_functional 包装的匿名函数，返回 l1_loss 函数的调用结果，传入输入 i、t，设置 reduction='none'
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用，使用 t.to(i.options()) 转换 t 并调用 l1_loss 函数，配置 reduction='none'
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        # 输入函数生成一个随机张量
        input_fn=lambda: torch.randn(2, 3, 4),
        # C++ 变量映射，将 'i' 映射到 '_get_input()'，将 't' 映射到之前定义的 t
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，返回输入 i 与 t 的差值的绝对值
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        # 支持前向自动求导
        supports_forward_ad=True,
        # 不支持序列化
        pickle=False,
        # 默认数据类型为双精度
        default_dtype=torch.double)


# 定义了一个函数，用于测试没有减少操作的复杂 L1 损失函数
def l1loss_no_reduce_complex_test():
    # 生成一个随机的复双精度张量作为 t 的值，形状为 (2, 3, 4)
    t = torch.randn(2, 3, 4, dtype=torch.cdouble)
    # 返回一个字典，包含函数的全名、构造函数、C++ 函数调用、输入函数、C++ 变量映射、参考函数、是否支持前向自动求导、是否可序列化
    return dict(
        fullname='L1Loss_no_reduce_complex',
        constructor=wrap_functional(
            # 使用 wrap_functional 包装的匿名函数，返回 l1_loss 函数的调用结果，传入输入 i、t，设置 reduction='none'
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用，使用 t.to(i.options()) 转换 t 并调用 l1_loss 函数，配置 reduction='none'
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        # 输入函数生成一个随机复双精度张量
        input_fn=lambda: torch.randn(2, 3, 4, dtype=torch.cdouble),
        # C++ 变量映射，将 'i' 映射到 '_get_input()'，将 't' 映射到之前定义的 t
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，返回输入 i 与 t 的差值的绝对值
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        # 支持前向自动求导
        supports_forward_ad=True,
        # 不支持序列化
        pickle=False)


# 定义了一个函数，用于测试没有减少操作的标量 L1 损失函数
def l1loss_no_reduce_scalar_test():
    # 生成一个随机的双精度标量张量作为 t 的值
    t = torch.randn((), dtype=torch.double)
    # 返回一个字典，包含函数的全名、构造函数、C++ 函数调用、输入函数、C++ 变量映射、参考函数、是否支持前向自动求导、是否可序列化、默认数据类型
    return dict(
        fullname='L1Loss_no_reduce_scalar',
        constructor=wrap_functional(
            # 使用 wrap_functional 包装的匿名函数，返回 l1_loss 函数的调用结果，传入输入 i、t，设置 reduction='none'
            lambda i: F.l1_loss(i, t.type_as(i), reduction='none')),
        # 对应的 C++ 函数调用，使用 t.to(i.options()) 转换 t 并调用 l1_loss 函数，配置 reduction='none'
        cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))',
        # 输入函数生成一个随机标量张量
        input_fn=lambda: torch.randn(()),
        # C++ 变量映射，将 'i' 映射到 '_get_input()'，将 't' 映射到之前定义的 t
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，返回输入 i 与 t 的差值的绝对值
        reference_fn=lambda i, *_: (i - t.type_as(i)).abs(),
        # 支持前向自动求导
        supports_forward_ad=True,
        # 不支持序列化
        pickle=False,
        # 默认数据类型为双精度
        default_dtype=torch.double)


# 定义了一个函数，用于测试没有减少操作的均方误差损失函数
def mseloss_no_reduce_test():
    # 定义输入的尺寸为 (2, 3, 4, 5)
    input_size = (2, 3, 4, 5)
    # 生成一个随机的双精度张量作为目标值，形状与输入尺寸相同
    target = torch.randn(*input_size, dtype=torch.double)
    return dict(
        fullname='MSELoss_no_reduce',  # 设置字典项fullname为字符串'MSELoss_no_reduce'
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),  # 设置字典项constructor为一个使用lambda函数封装的函数对象，用于计算MSE损失
        cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))',  # 设置字典项cpp_function_call为一个C++函数调用的字符串表示
        input_size=input_size,  # 设置字典项input_size为变量input_size的值
        cpp_var_map={'i': '_get_input()', 'target': target},  # 设置字典项cpp_var_map为一个包含键值对的字典，用于映射变量名到其在C++中的表示
        reference_fn=lambda i, *_: (i - target).pow(2),  # 设置字典项reference_fn为一个使用lambda函数封装的函数对象，用于计算参考函数的值
        supports_forward_ad=True,  # 设置字典项supports_forward_ad为布尔值True，表示支持前向自动求导
        pickle=False,  # 设置字典项pickle为布尔值False，表示禁用对象的pickle操作
        default_dtype=torch.double  # 设置字典项default_dtype为torch库中的双精度浮点数类型
    )
# 定义一个函数，用于测试没有减少（reduce）操作的均方误差损失（MSELoss）
def mseloss_no_reduce_scalar_test():
    # 设置输入大小为空元组
    input_size = ()
    # 生成一个随机张量作为目标，数据类型为双精度浮点型
    target = torch.randn(input_size, dtype=torch.double)
    # 返回一个字典，包含测试相关信息
    return dict(
        fullname='MSELoss_no_reduce_scalar',  # 完整名称
        constructor=wrap_functional(
            lambda i: F.mse_loss(i, target.type_as(i), reduction='none')),  # 构造函数包装，计算MSE损失
        cpp_function_call='F::mse_loss(i, target.to(i.options()), F::MSELossFuncOptions().reduction(torch::kNone))',  # 对应C++函数调用
        input_size=input_size,  # 输入大小
        cpp_var_map={'i': '_get_input()', 'target': target},  # C++变量映射
        reference_fn=lambda i, *_: (i - target).pow(2),  # 参考函数，计算损失的参考实现
        supports_forward_ad=True,  # 是否支持前向自动求导
        pickle=False,  # 是否支持pickle序列化
        default_dtype=torch.double  # 默认数据类型为双精度浮点型
    )


# 定义一个函数，用于测试没有减少（reduce）操作的负对数似然损失（NLLLoss）
def nllloss_no_reduce_test():
    # 创建一个长为15的随机整数张量
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    # 设置关键字参数字典，指定没有减少操作
    kwargs = {'reduction': 'none'}
    # 返回一个字典，包含测试相关信息
    return dict(
        fullname='NLLLoss_no_reduce',  # 完整名称
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),  # 构造函数包装，计算NLL损失
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',  # 对应C++函数调用
        input_fn=lambda: torch.rand(15, 10).log(),  # 输入函数，生成输入数据
        cpp_var_map={'i': '_get_input()', 't': t},  # C++变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),  # 参考函数，计算损失的参考实现
        pickle=False,  # 是否支持pickle序列化
        default_dtype=torch.double  # 默认数据类型为双精度浮点型
    )


# 定义一个函数，用于测试带有ignore_index的没有减少（reduce）操作的负对数似然损失（NLLLoss）
def nllloss_no_reduce_ignore_index_test():
    # 创建一个长为15的随机整数张量
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    # 设置关键字参数字典，包括ignore_index为2，没有减少操作
    kwargs: Dict[str, Union[int, str]] = {'ignore_index': 2, 'reduction': 'none'}
    # 返回一个字典，包含测试相关信息
    return dict(
        fullname='NLLLoss_no_reduce_ignore_index',  # 完整名称
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),  # 构造函数包装，计算NLL损失
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(2).reduction(torch::kNone))''',  # 对应C++函数调用
        input_fn=lambda: torch.rand(15, 10).log(),  # 输入函数，生成输入数据
        cpp_var_map={'i': '_get_input()', 't': t},  # C++变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs),  # 参考函数，计算损失的参考实现
        pickle=False,  # 是否支持pickle序列化
        default_dtype=torch.double  # 默认数据类型为双精度浮点型
    )


# 定义一个函数，用于测试带有权重的没有减少（reduce）操作的负对数似然损失（NLLLoss）
def nllloss_no_reduce_weights_test():
    # 创建一个长为15的随机整数张量
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    # 创建一个长度为10的随机张量作为权重
    weight = torch.rand(10)

    # 定义一个函数，返回权重和没有减少（reduce）操作的关键字参数
    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}
    # 返回一个包含多个键值对的字典
    return dict(
        # 定义键名为 'fullname'，对应值为 'NLLLoss_no_reduce_weights'
        fullname='NLLLoss_no_reduce_weights',
        # 定义键名为 'constructor'，对应值为一个函数封装，使用了函数 wrap_functional
        constructor=wrap_functional(
            # 匿名函数，接受参数 i，调用 F.nll_loss 函数，并传递参数
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        # 定义键名为 'cpp_function_call'，对应值为一个字符串，描述了 C++ 函数调用的内容
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        # 定义键名为 'input_fn'，对应值为一个匿名函数，返回一个随机生成的张量
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        # 定义键名为 'cpp_var_map'，对应值为一个字典，映射了变量名到表达式或值
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        # 定义键名为 'reference_fn'，对应值为一个匿名函数，调用 loss_reference_fns 字典中的函数
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        # 定义键名为 'pickle'，对应值为 False，表示不支持序列化
        pickle=False,
        # 定义键名为 'default_dtype'，对应值为 torch.double，指定默认数据类型为双精度浮点数
        default_dtype=torch.double)
def nllloss_no_reduce_weights_ignore_index_test():
    # 创建一个包含15个随机整数的张量，用于作为预测结果的目标
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    # 创建一个包含10个随机数的张量，用于加权损失计算
    weight = torch.rand(10)

    # 定义一个返回损失函数参数字典的函数
    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': 2}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index',
        # 使用 wrap_functional 包装的匿名函数，调用 F.nll_loss 计算损失
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i.data))),
        # 对应的 C++ 函数调用字符串表示
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone).ignore_index(2))''',
        # 生成输入的函数，返回一个形状为(15, 10)的随机张量的对数值
        input_fn=lambda: torch.rand(15, 10).add(1e-2).log(),
        # C++ 变量映射字典，用于传递给 CPP 函数的变量映射
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        # 参考函数，调用 loss_reference_fns['NLLLoss'] 计算参考损失
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def nllloss_no_reduce_weights_ignore_index_neg_test():
    # 创建一个包含15个随机整数的张量，用于作为预测结果的目标
    t = Variable(torch.empty(15).uniform_().mul(10).floor().long())
    # 创建一个包含10个随机数的张量，用于加权损失计算
    weight = torch.rand(10)

    # 定义一个返回损失函数参数字典的函数
    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none',
                'ignore_index': -1}

    return dict(
        fullname='NLLLoss_no_reduce_weights_ignore_index_neg',
        # 使用 wrap_functional 包装的匿名函数，调用 F.nll_loss 计算损失
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        # 对应的 C++ 函数调用字符串表示
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone).ignore_index(-1))''',
        # 输入张量，形状为 (15, 10)，元素为对数值
        input=torch.rand(15, 10, dtype=torch.double).add(1e-2).log(),
        # C++ 变量映射字典，用于传递给 CPP 函数的变量映射
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        # 参考函数，调用 loss_reference_fns['NLLLoss'] 计算参考损失
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLoss'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


def nllloss2d_no_reduce_test():
    # 创建一个形状为 (2, 5, 5) 的张量，用于作为预测结果的目标
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    # 定义损失函数参数字典
    kwargs = {'reduction': 'none'}
    return dict(
        fullname='NLLLoss2d_no_reduce',
        # 使用 wrap_functional 包装的匿名函数，调用 F.nll_loss 计算二维损失
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), reduction=kwargs['reduction'])),
        # 对应的 C++ 函数调用字符串表示
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().reduction(torch::kNone))''',
        # 生成输入的函数，返回一个形状为(2, 3, 5, 5)的随机张量的对数值
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        # C++ 变量映射字典，用于传递给 CPP 函数的变量映射
        cpp_var_map={'i': '_get_input()', 't': t},
        # 参考函数，调用 loss_reference_fns['NLLLossNd'] 计算参考损失
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        pickle=False,
        default_dtype=torch.double)


def nllloss2d_no_reduce_ignore_index_test():
    # 创建一个形状为 (2, 5, 5) 的张量，用于作为预测结果的目标
    t = Variable(torch.rand(2, 5, 5).mul(3).floor().long())
    # 定义损失函数参数字典，包括忽略索引和无损失减少
    kwargs: Dict[str, Union[int, str]] = {'ignore_index': 1, 'reduction': 'none'}
    # 返回一个包含多个键值对的字典对象
    return dict(
        # 设置键 'fullname' 对应的值为 'NLLLoss2d_no_reduce_ignore_index'
        fullname='NLLLoss2d_no_reduce_ignore_index',
        # 设置键 'constructor' 对应的值为一个函数封装
        constructor=wrap_functional(
            # 使用 lambda 函数定义一个函数，接受参数 i，返回 F.nll_loss 的调用结果
            lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']),
                                 reduction=str(kwargs['reduction']))),
        # 设置键 'cpp_function_call' 对应的值为一个字符串，表示 C++ 函数调用的模板
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(1).reduction(torch::kNone))''',
        # 设置键 'input_fn' 对应的值为一个 lambda 函数，生成一个随机张量并应用对数运算
        input_fn=lambda: torch.rand(2, 3, 5, 5).log(),
        # 设置键 'cpp_var_map' 对应的值为一个字典，映射了变量名到对应的 C++ 表达式
        cpp_var_map={'i': '_get_input()', 't': t},
        # 设置键 'reference_fn' 对应的值为一个 lambda 函数，调用 loss_reference_fns 中的函数 'NLLLossNd'
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs),
        # 设置键 'pickle' 对应的值为 False，表示该对象不支持序列化
        pickle=False,
        # 设置键 'default_dtype' 对应的值为 torch.double，表示默认数据类型为双精度浮点数
        default_dtype=torch.double)
def nlllossNd_no_reduce_weights_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    weight = torch.rand(3)

    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    return dict(
        fullname='NLLLossNd_no_reduce_weights',
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        pickle=False,
        default_dtype=torch.double)


注释：


# 定义一个函数，用于测试不带减少（reduction='none'）和权重（weight）的多维负对数似然损失函数（NLLLossNd）
def nlllossNd_no_reduce_weights_test():
    # 创建一个随机的长整型张量 t，形状为 (2, 5, 5, 2, 2)
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    # 创建一个随机的张量 weight，长度为 3
    weight = torch.rand(3)

    # 定义一个内部函数 kwargs，返回一个字典，包含权重和减少方式的参数
    def kwargs(i):
        return {'weight': weight.type_as(i), 'reduction': 'none'}

    # 返回一个字典，包含测试函数的各种信息和配置
    return dict(
        fullname='NLLLossNd_no_reduce_weights',  # 测试的全名
        constructor=wrap_functional(  # 使用包装函数构造的对象，用于函数式风格调用
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),  # 构造负对数似然损失函数对象
        cpp_function_call='''F::nll_loss(  # 对应的 C++ 函数调用形式
            i, t.to(i.options()).to(torch::kLong),  # 将输入 i 和目标 t 转换为适当的类型和形状
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),  # 输入数据的生成函数
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},  # C++ 变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),  # 参考函数调用
        pickle=False,  # 是否支持序列化
        default_dtype=torch.double)  # 默认的数据类型
    # 返回一个字典对象，包含以下键值对：
    return dict(
        # 键 'fullname' 对应字符串 'NLLLossNd_no_reduce_weights'
        fullname='NLLLossNd_no_reduce_weights',
        # 键 'constructor' 对应一个函数，通过 wrap_functional 包装一个匿名函数，
        # 该函数调用 F.nll_loss 函数，使用输入 i 和 t.type_as(i).long()，以及根据 i 的参数调用 kwargs(i) 的结果作为额外参数
        constructor=wrap_functional(
            lambda i: F.nll_loss(i, t.type_as(i).long(), **kwargs(i))),
        # 键 'cpp_function_call' 对应一个描述 C++ 函数调用的字符串，调用 F::nll_loss，
        # 使用 i 和 t.to(i.options()).to(torch::kLong) 作为参数，并设置权重和 reduction 选项
        cpp_function_call='''F::nll_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::NLLLossFuncOptions().weight(weight.to(i.options())).reduction(torch::kNone))''',
        # 键 'input_fn' 对应一个函数，返回一个形状为 (2, 3, 5, 5, 2, 2) 的随机张量的对数值
        input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(),
        # 键 'cpp_var_map' 对应一个字典，映射变量名到其值，其中 'i' 映射到 '_get_input()'，
        # 't' 映射到变量 t，'weight' 映射到变量 weight
        cpp_var_map={'i': '_get_input()', 't': t, 'weight': weight},
        # 键 'reference_fn' 对应一个函数，调用 loss_reference_fns['NLLLossNd']，
        # 使用 i 和 t.type_as(i).long() 作为参数，并根据 i 的参数调用 kwargs(i)
        reference_fn=lambda i, *_:
            loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs(i)),
        # 键 'pickle' 对应布尔值 False，表示禁用对象的 pickle 操作
        pickle=False,
        # 键 'default_dtype' 对应 torch.double，表示默认的数据类型为双精度浮点数
        default_dtype=torch.double)
# 定义一个测试函数，用于测试没有减少项的 Smooth L1 损失函数
def smoothl1loss_no_reduce_test():
    # 生成一个大小为 (2, 3, 4) 的双精度随机张量 t
    t = torch.randn(2, 3, 4, dtype=torch.double)
    # 返回一个字典，包含测试信息
    return dict(
        fullname='SmoothL1Loss_no_reduce',  # 测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装 F.smooth_l1_loss，reduction='none'
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))''',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.randn(2, 3, 4),  # 生成输入张量的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # C++ 变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),  # 参考函数调用
        supports_forward_ad=True,  # 是否支持前向自动微分
        pickle=False,  # 是否支持 pickle
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试没有减少项的 Smooth L1 损失函数（标量输入）
def smoothl1loss_no_reduce_scalar_test():
    # 生成一个双精度的标量随机张量 t
    t = torch.randn((), dtype=torch.double)
    # 返回一个字典，包含测试信息
    return dict(
        fullname='SmoothL1Loss_no_reduce_scalar',  # 测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装 F.smooth_l1_loss，reduction='none'
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))''',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.randn(()),  # 生成输入张量的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # C++ 变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'),  # 参考函数调用
        supports_forward_ad=True,  # 是否支持前向自动微分
        pickle=False,  # 是否支持 pickle
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试带有 beta 参数的 Smooth L1 损失函数
def smoothl1loss_beta_test():
    # 生成一个大小为 (2, 3, 4) 的双精度随机张量 t
    t = torch.randn(2, 3, 4, dtype=torch.double)
    # 返回一个字典，包含测试信息
    return dict(
        fullname='SmoothL1Loss_beta',  # 测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装 F.smooth_l1_loss，reduction='none', beta=0.5
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0.5)),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0.5)''',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.randn(2, 3, 4),  # 生成输入张量的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # C++ 变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0.5),  # 参考函数调用
        supports_forward_ad=True,  # 是否支持前向自动微分
        pickle=False,  # 是否支持 pickle
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试带有零 beta 参数的 Smooth L1 损失函数
def smoothl1loss_zero_beta_test():
    # 生成一个大小为 (2, 3, 4) 的双精度随机张量 t
    t = torch.randn(2, 3, 4, dtype=torch.double)
    # 返回一个字典，包含测试信息
    return dict(
        fullname='SmoothL1Loss_zero_beta',  # 测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装 F.smooth_l1_loss，reduction='none', beta=0
            lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none', beta=0)),
        cpp_function_call='''F::smooth_l1_loss(
            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone), 0)''',  # 对应的 C++ 函数调用
        input_fn=lambda: torch.randn(2, 3, 4),  # 生成输入张量的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # C++ 变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none', beta=0),  # 参考函数调用
        supports_forward_ad=True,  # 是否支持前向自动微分
        pickle=False,  # 是否支持 pickle
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试 Huber 损失函数的 delta 参数
def huberloss_delta_test():
    # 生成一个大小为 (2, 3, 4) 的随机张量 t
    t = torch.randn(2, 3, 4)
    # 返回一个包含各种配置项的字典
    return dict(
        # 设置配置项 fullname 为 'HuberLoss_delta'
        fullname='HuberLoss_delta',
        # 使用 wrap_functional 函数包装的 Lambda 函数，用于计算 Huber 损失
        constructor=wrap_functional(
            lambda i: F.huber_loss(i, t.type_as(i), reduction='none', delta=0.5)),
        # 指定 C++ 函数调用的字符串表示，用于调用 Huber 损失函数
        cpp_function_call='''F::huber_loss(
            i, t.to(i.options()), F::HuberLossFuncOptions().reduction(torch::kNone).delta(0.5))''',
        # 定义一个生成输入的函数，生成一个形状为 (2, 3, 4) 的随机张量
        input_fn=lambda: torch.randn(2, 3, 4),
        # 定义一个 C++ 变量映射，将 'i' 映射为 '_get_input()', 't' 映射为 t
        cpp_var_map={'i': '_get_input()', 't': t},
        # 定义一个参考函数，用于计算 Huber 损失的参考值
        reference_fn=lambda i, *_:
            loss_reference_fns['HuberLoss'](i, t.type_as(i), reduction='none', delta=0.5),
        # 标识该配置支持前向自动求导
        supports_forward_ad=True,
        # 禁用 pickle 功能
        pickle=False,
        # 设置默认数据类型为 torch.double
        default_dtype=torch.double)
# 定义一个测试函数，用于测试没有减少项的多标签边缘损失函数
def multilabelmarginloss_0d_no_reduce_test():
    # 创建一个零维的长整型张量 t
    t = torch.zeros(()).long()
    # 返回包含多个键值对的字典
    return dict(
        fullname='MultiLabelMarginLoss_0d_no_reduce',  # 名称描述为多标签边缘损失函数的零维版本（无减少项）
        constructor=wrap_functional(
            # 使用函数式编程包装，构造一个函数，接受输入 i，并调用多标签边缘损失函数，指定无减少项
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(()),  # 生成一个随机张量作为输入的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # 将输入和 t 映射到 C++ 变量
        reference_fn=lambda i, *_:
            # 调用参考函数来计算多标签边缘损失，指定无减少项
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,  # 检查是否进行了求和减少操作
        check_gradgrad=False,  # 不检查二阶梯度
        pickle=False)  # 不支持 pickle 序列化


def multilabelmarginloss_1d_no_reduce_test():
    # 创建一个包含 10 个元素的长整型变量 t，其值在 [0, 9] 之间
    t = Variable(torch.rand(10).mul(10).floor().long())
    # 返回包含多个键值对的字典
    return dict(
        fullname='MultiLabelMarginLoss_1d_no_reduce',  # 名称描述为多标签边缘损失函数的一维版本（无减少项）
        constructor=wrap_functional(
            # 使用函数式编程包装，构造一个函数，接受输入 i，并调用多标签边缘损失函数，指定无减少项
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(10),  # 生成一个包含 10 个元素的随机张量作为输入的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # 将输入和 t 映射到 C++ 变量
        reference_fn=lambda i, *_:
            # 调用参考函数来计算多标签边缘损失，指定无减少项
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,  # 检查是否进行了求和减少操作
        check_gradgrad=False,  # 不检查二阶梯度
        pickle=False,  # 不支持 pickle 序列化
        default_dtype=torch.double)  # 默认数据类型为双精度浮点数


def multilabelmarginloss_index_neg_test():
    # 创建一个形状为 (5, 10) 的变量 t，其值在 [-1, 19] 之间
    t = Variable(torch.clamp(torch.rand(5, 10).add(-.5).mul(20).floor().long(), min=-1))
    # 返回包含多个键值对的字典
    return dict(
        fullname='MultiLabelMarginLoss_index_neg',  # 名称描述为带有负索引的多标签边缘损失函数
        constructor=wrap_functional(
            # 使用函数式编程包装，构造一个函数，接受输入 i，并调用多标签边缘损失函数，指定无减少项
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),  # 生成一个形状为 (5, 10) 的随机张量作为输入的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # 将输入和 t 映射到 C++ 变量
        reference_fn=lambda i, *_:
            # 调用参考函数来计算多标签边缘损失，指定无减少项
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,  # 检查是否进行了求和减少操作
        check_gradgrad=False,  # 不检查二阶梯度
        pickle=False,  # 不支持 pickle 序列化
        default_dtype=torch.double)  # 默认数据类型为双精度浮点数


def multilabelmarginloss_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(10).floor().long())
    # 此处未返回任何字典，因此需要完成函数的具体实现
    # 返回一个包含各种参数的字典
    return dict(
        # 定义全名为 'MultiLabelMarginLoss_no_reduce'
        fullname='MultiLabelMarginLoss_no_reduce',
        # 将 lambda 函数包装为 PyTorch 的函数
        constructor=wrap_functional(
            lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')),
        # 定义一个 C++ 函数调用字符串
        cpp_function_call='''F::multilabel_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))''',
        # 定义输入数据生成函数
        input_fn=lambda: torch.randn(5, 10),
        # 定义一个 C++ 变量映射字典
        cpp_var_map={'i': '_get_input()', 't': t},
        # 定义参考函数
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        # 检查是否进行了求和操作的标志
        check_sum_reduction=True,
        # 检查二阶梯度的标志
        check_gradgrad=False,
        # 禁用 pickle 功能
        pickle=False,
        # 默认数据类型为双精度浮点数
        default_dtype=torch.double)
# 定义一个测试函数，用于测试不带减少(reduction)的HingeEmbeddingLoss损失函数
def hingeembeddingloss_no_reduce_test():
    # 生成一个包含10个随机数的张量，并进行大于零的比较，转换为双精度张量，乘以2再减1
    t = Variable(torch.randn(10).gt(0).to(torch.double).mul_(2).sub(1))
    # 返回一个字典，包含测试信息
    return dict(
        fullname='HingeEmbeddingLoss_no_reduce',  # 损失函数的全名
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), reduction='none')),  # 构造函数的包装，使用HingeEmbeddingLoss函数，不减少
        cpp_function_call='''F::hinge_embedding_loss(
            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().reduction(torch::kNone))''',  # 对应的C++函数调用
        input_fn=lambda: torch.randn(10),  # 输入函数，生成一个10维随机张量
        cpp_var_map={'i': '_get_input()', 't': t},  # C++变量映射，将'i'映射为'_get_input()', 't'映射为t
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), reduction='none'),  # 参考函数，使用Python的损失函数计算
        check_sum_reduction=True,  # 检查是否进行了减少操作
        pickle=False,  # 是否支持pickle序列化
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试带边界(margin)和不带减少的HingeEmbeddingLoss损失函数
def hingeembeddingloss_margin_no_reduce_test():
    # 生成一个包含10个随机数的张量，并进行大于零的比较，转换为双精度张量，乘以2再减1
    t = Variable(torch.randn(10).gt(0).to(torch.double).mul_(2).sub(1))
    # 返回一个字典，包含测试信息
    return dict(
        fullname='HingeEmbeddingLoss_margin_no_reduce',  # 损失函数的全名
        constructor=wrap_functional(
            lambda i: F.hinge_embedding_loss(i, t.type_as(i), margin=0.5, reduction='none')),  # 构造函数的包装，使用带边界的HingeEmbeddingLoss函数，不减少
        cpp_function_call='''F::hinge_embedding_loss(
            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().margin(0.5).reduction(torch::kNone))''',  # 对应的C++函数调用
        input_fn=lambda: torch.randn(10),  # 输入函数，生成一个10维随机张量
        cpp_var_map={'i': '_get_input()', 't': t},  # C++变量映射，将'i'映射为'_get_input()', 't'映射为t
        reference_fn=lambda i, *_:
            loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), margin=0.5, reduction='none'),  # 参考函数，使用Python的损失函数计算
        check_sum_reduction=True,  # 检查是否进行了减少操作
        pickle=False,  # 是否支持pickle序列化
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试不带减少(reduction)的SoftMarginLoss损失函数
def softmarginloss_no_reduce_test():
    # 生成一个5x5大小的双精度随机张量
    t = torch.randn(5, 5, dtype=torch.double)
    # 返回一个字典，包含测试信息
    return dict(
        fullname='SoftMarginLoss_no_reduce',  # 损失函数的全名
        constructor=wrap_functional(
            lambda i: F.soft_margin_loss(i, t.type_as(i), reduction='none')),  # 构造函数的包装，使用SoftMarginLoss函数，不减少
        cpp_function_call='''F::soft_margin_loss(
            i, t.to(i.options()), F::SoftMarginLossFuncOptions().reduction(torch::kNone))''',  # 对应的C++函数调用
        input_fn=lambda: torch.randn(5, 5),  # 输入函数，生成一个5x5大小的随机张量
        cpp_var_map={'i': '_get_input()', 't': t},  # C++变量映射，将'i'映射为'_get_input()', 't'映射为t
        reference_fn=lambda i, *_:
            loss_reference_fns['SoftMarginLoss'](i, t.type_as(i), reduction='none'),  # 参考函数，使用Python的损失函数计算
        supports_forward_ad=True,  # 是否支持前向自动微分
        pickle=False,  # 是否支持pickle序列化
        default_dtype=torch.double)  # 默认的数据类型为双精度


# 定义一个测试函数，用于测试多标签的不带减少(reduction)的SoftMarginLoss损失函数
def multilabelsoftmarginloss_no_reduce_test():
    # 生成一个5x10大小的张量，其值为0到1之间的随机浮点数乘以2取整
    t = torch.rand(5, 10).mul(2).floor()
    # 返回一个字典对象，包含多个命名参数和值
    return dict(
        # 定义参数 fullname，表示名称为 'MultiLabelSoftMarginLoss_no_reduce'
        fullname='MultiLabelSoftMarginLoss_no_reduce',
        # 定义参数 constructor，使用 wrap_functional 函数封装的 Lambda 表达式
        constructor=wrap_functional(
            lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), reduction='none')),
        # 定义参数 cpp_function_call，表示一个 C++ 函数调用的字符串
        cpp_function_call='''F::multilabel_soft_margin_loss(
            i, t.to(i.options()), F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone))''',
        # 定义参数 input_fn，为一个 Lambda 函数，生成一个 5x10 的随机张量
        input_fn=lambda: torch.randn(5, 10),
        # 定义参数 cpp_var_map，为一个字典，包含键值对 {'i': '_get_input()', 't': t}
        cpp_var_map={'i': '_get_input()', 't': t},
        # 定义参数 reference_fn，为一个 Lambda 函数，计算一个张量的损失函数
        reference_fn=lambda i, *_:
            (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log())).sum(dim=1) / i.size(1),
        # 定义参数 check_gradgrad，表示是否检查梯度的梯度，默认为 False
        check_gradgrad=False,
        # 定义参数 pickle，表示是否支持 pickle，默认为 False
        pickle=False,
        # 定义参数 default_dtype，表示默认的张量数据类型为 torch.double
        default_dtype=torch.double)
def multimarginloss_1d_input_0d_target_no_reduce_test():
    t = torch.rand(()).mul(8).floor().long()
    return dict(
        fullname='MultiMarginLoss_1d_input_0d_target_no_reduce',
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(()),
        cpp_var_map={'i': '_get_input()', 't': t},
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        check_sum_reduction=True,
        check_gradgrad=False,
        pickle=False,
        default_dtype=torch.double)


注释：


# 定义一个测试函数，用于测试处理一维输入和零维目标的多边界损失函数的非降维情况
def multimarginloss_1d_input_0d_target_no_reduce_test():
    # 随机生成一个零维目标张量，乘以8后向下取整并转换为长整型
    t = torch.rand(()).mul(8).floor().long()
    # 返回一个包含测试所需参数和设置的字典
    return dict(
        fullname='MultiMarginLoss_1d_input_0d_target_no_reduce',  # 完整名称标识此测试的类型和设置
        constructor=wrap_functional(
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),  # 使用包装功能构造损失函数对象，不降维
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',  # 对应的 C++ 函数调用方式
        input_fn=lambda: torch.randn(()),  # 返回一个零维输入张量的生成函数
        cpp_var_map={'i': '_get_input()', 't': t},  # 传递给 C++ 的变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),  # 参考函数计算损失，不降维
        check_sum_reduction=True,  # 检查是否进行了求和降维
        check_gradgrad=False,  # 不检查梯度的二阶导数
        pickle=False,  # 不支持序列化
        default_dtype=torch.double)  # 默认张量数据类型为双精度浮点数
    # 返回一个包含多个键值对的字典
    return dict(
        # 键为 fullname，值为特定字符串
        fullname='multimarginloss_1d_input_0d_target_no_reduce',
        # 键为 constructor，值为一个函数包装结果
        constructor=wrap_functional(
            # 使用 Lambda 函数包装的多重边际损失函数调用
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')),
        # 键为 cpp_function_call，值为特定的 C++ 函数调用字符串
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))''',
        # 键为 input_fn，值为生成随机张量的函数
        input_fn=lambda: torch.randn(10),
        # 键为 cpp_var_map，值为一个包含键值对的字典
        cpp_var_map={'i': '_get_input()', 't': t},
        # 键为 reference_fn，值为一个参考函数的调用
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'),
        # 键为 check_sum_reduction，值为布尔值 True，表示是否检查和缩减
        check_sum_reduction=True,
        # 键为 check_gradgrad，值为布尔值 False，表示是否检查梯度的二阶导数
        check_gradgrad=False,
        # 键为 pickle，值为布尔值 False，表示是否允许序列化
        pickle=False,
        # 键为 default_dtype，值为 torch.double，表示默认数据类型为双精度浮点数
        default_dtype=torch.double)
def multimarginloss_p_no_reduce_test():
    # 生成一个包含 5 个随机数的张量，乘以 8 并向下取整，转换为长整型
    t = torch.rand(5).mul(8).floor().long()
    # 返回一个包含测试信息的字典
    return dict(
        fullname='MultiMarginLoss_p_no_reduce',  # 定义测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装的多类边界损失函数，设置 p=2，不进行缩减
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), p=2, reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().p(2).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10).clamp_(1e-2, 1 - 1e-2),  # 生成输入张量的函数，数据范围在 (0.01, 0.99) 之间
        cpp_var_map={'i': '_get_input()', 't': t},  # 定义传递给 C++ 函数的变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), p=2, reduction='none'),  # 引用函数，提供参考计算结果
        check_sum_reduction=True,  # 是否检查缩减求和的正确性
        check_gradgrad=False,  # 是否检查梯度的梯度计算
        pickle=False,  # 是否支持 pickle 序列化
        default_dtype=torch.double  # 默认张量数据类型
    )


def multimarginloss_margin_no_reduce_test():
    # 生成一个包含 5 个随机数的张量，乘以 8 并向下取整，转换为长整型
    t = torch.rand(5).mul(8).floor().long()
    # 返回一个包含测试信息的字典
    return dict(
        fullname='MultiMarginLoss_margin_no_reduce',  # 定义测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装的多类边界损失函数，设置 margin=0.5，不进行缩减
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), margin=0.5, reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::MultiMarginLossFuncOptions().margin(0.5).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),  # 生成输入张量的函数
        cpp_var_map={'i': '_get_input()', 't': t},  # 定义传递给 C++ 函数的变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  margin=0.5, reduction='none'),  # 引用函数，提供参考计算结果
        check_sum_reduction=True,  # 是否检查缩减求和的正确性
        check_gradgrad=False,  # 是否检查梯度的梯度计算
        pickle=False,  # 是否支持 pickle 序列化
        default_dtype=torch.double  # 默认张量数据类型
    )


def multimarginloss_weights_no_reduce_test():
    # 生成一个包含 5 个随机数的张量，乘以 8 并向下取整，转换为长整型
    t = torch.rand(5).mul(8).floor().long()
    # 生成一个包含 10 个随机数的双精度张量
    weights = torch.rand(10, dtype=torch.double)
    # 返回一个包含测试信息的字典
    return dict(
        fullname='MultiMarginLoss_weights_no_reduce',  # 定义测试名称
        constructor=wrap_functional(
            # 使用 lambda 函数封装的多类边界损失函数，设置权重和不进行缩减
            lambda i: F.multi_margin_loss(i, t.type_as(i).long(), weight=weights.type_as(i),
                                          reduction='none')),
        cpp_function_call='''F::multi_margin_loss(
            i, t.to(i.options()).to(torch::kLong),
            F::MultiMarginLossFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))''',
        input_fn=lambda: torch.randn(5, 10),  # 生成输入张量的函数
        cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights},  # 定义传递给 C++ 函数的变量映射
        reference_fn=lambda i, *_:
            loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(),
                                                  weight=weights, reduction='none'),  # 引用函数，提供参考计算结果
        check_sum_reduction=True,  # 是否检查缩减求和的正确性
        check_gradgrad=False,  # 是否检查梯度的梯度计算
        pickle=False,  # 是否支持 pickle 序列化
        default_dtype=torch.double  # 默认张量数据类型
    )


def single_batch_reference_fn(input, parameters, module):
    """Reference function for modules supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    # 定义一个函数，将输入张量或张量列表/元组中的每个张量都在第0维度上添加一个维度（unsqueeze操作）
    def unsqueeze_inp(inp):
        if isinstance(inp, (list, tuple)):
            return [t.unsqueeze(0) for t in inp]
        return inp.unsqueeze(0)

    # 将输入转换为单批次输入，即使输入本身已经是张量也将其放入列表中
    single_batch_input = unsqueeze_inp(input)
    # 如果单批次输入是张量，则将其放入列表中以便处理
    single_batch_input = [single_batch_input] if isinstance(single_batch_input, torch.Tensor) else single_batch_input
    # 冻结随机数生成器状态的上下文管理器，确保结果的可复现性
    with freeze_rng_state():
        # 调用模块并传入单批次输入，然后对输出进行维度压缩以与非批处理输入进行比较
        return module(*single_batch_input).squeeze(0)
new_module_tests = [
    # 执行 poissonnllloss_no_reduce_test 函数，将其结果添加到列表中
    poissonnllloss_no_reduce_test(),
    # 执行 bceloss_no_reduce_test 函数，将其结果添加到列表中
    bceloss_no_reduce_test(),
    # 执行 bceloss_weights_no_reduce_test 函数，将其结果添加到列表中
    bceloss_weights_no_reduce_test(),
    # 执行 bce_with_logistic_legacy_enum_test 函数，将其结果添加到列表中
    bce_with_logistic_legacy_enum_test(),
    # 执行 bce_with_logistic_no_reduce_test 函数，将其结果添加到列表中
    bce_with_logistic_no_reduce_test(),
    # 执行 bceloss_no_reduce_scalar_test 函数，将其结果添加到列表中
    bceloss_no_reduce_scalar_test(),
    # 执行 bceloss_weights_no_reduce_scalar_test 函数，将其结果添加到列表中
    bceloss_weights_no_reduce_scalar_test(),
    # 执行 bce_with_logistic_no_reduce_scalar_test 函数，将其结果添加到列表中
    bce_with_logistic_no_reduce_scalar_test(),
    # 执行 kldivloss_with_target_no_reduce_test 函数，将其结果添加到列表中
    kldivloss_with_target_no_reduce_test(),
    # 执行 kldivloss_no_reduce_test 函数，将其结果添加到列表中
    kldivloss_no_reduce_test(),
    # 执行 kldivloss_no_reduce_scalar_test 函数，将其结果添加到列表中
    kldivloss_no_reduce_scalar_test(),
    # 执行 kldivloss_with_log_target_no_reduce_test 函数，将其结果添加到列表中
    kldivloss_with_log_target_no_reduce_test(),
    # 执行 kldivloss_no_reduce_log_target_test 函数，将其结果添加到列表中
    kldivloss_no_reduce_log_target_test(),
    # 执行 kldivloss_no_reduce_scalar_log_target_test 函数，将其结果添加到列表中
    kldivloss_no_reduce_scalar_log_target_test(),
    # 执行 l1loss_no_reduce_test 函数，将其结果添加到列表中
    l1loss_no_reduce_test(),
    # 执行 l1loss_no_reduce_complex_test 函数，将其结果添加到列表中
    l1loss_no_reduce_complex_test(),
    # 执行 l1loss_no_reduce_scalar_test 函数，将其结果添加到列表中
    l1loss_no_reduce_scalar_test(),
    # 执行 mseloss_no_reduce_test 函数，将其结果添加到列表中
    mseloss_no_reduce_test(),
    # 执行 mseloss_no_reduce_scalar_test 函数，将其结果添加到列表中
    mseloss_no_reduce_scalar_test(),
    # 执行 nllloss_no_reduce_test 函数，将其结果添加到列表中
    nllloss_no_reduce_test(),
    # 执行 nllloss_no_reduce_ignore_index_test 函数，将其结果添加到列表中
    nllloss_no_reduce_ignore_index_test(),
    # 执行 nllloss_no_reduce_weights_test 函数，将其结果添加到列表中
    nllloss_no_reduce_weights_test(),
    # 执行 nllloss_no_reduce_weights_ignore_index_test 函数，将其结果添加到列表中
    nllloss_no_reduce_weights_ignore_index_test(),
    # 执行 nllloss_no_reduce_weights_ignore_index_neg_test 函数，将其结果添加到列表中
    nllloss_no_reduce_weights_ignore_index_neg_test(),
    # 执行 nllloss2d_no_reduce_test 函数，将其结果添加到列表中
    nllloss2d_no_reduce_test(),
    # 执行 nllloss2d_no_reduce_weights_test 函数，将其结果添加到列表中
    nllloss2d_no_reduce_weights_test(),
    # 执行 nllloss2d_no_reduce_ignore_index_test 函数，将其结果添加到列表中
    nllloss2d_no_reduce_ignore_index_test(),
    # 执行 nlllossNd_no_reduce_test 函数，将其结果添加到列表中
    nlllossNd_no_reduce_test(),
    # 执行 nlllossNd_no_reduce_weights_test 函数，将其结果添加到列表中
    nlllossNd_no_reduce_weights_test(),
    # 执行 nlllossNd_no_reduce_ignore_index_test 函数，将其结果添加到列表中
    nlllossNd_no_reduce_ignore_index_test(),
    # 执行 smoothl1loss_no_reduce_test 函数，将其结果添加到列表中
    smoothl1loss_no_reduce_test(),
    # 执行 smoothl1loss_no_reduce_scalar_test 函数，将其结果添加到列表中
    smoothl1loss_no_reduce_scalar_test(),
    # 执行 smoothl1loss_beta_test 函数，将其结果添加到列表中
    smoothl1loss_beta_test(),
    # 执行 smoothl1loss_zero_beta_test 函数，将其结果添加到列表中
    smoothl1loss_zero_beta_test(),
    # 执行 huberloss_delta_test 函数，将其结果添加到列表中
    huberloss_delta_test(),
    # 执行 multilabelmarginloss_0d_no_reduce_test 函数，将其结果添加到列表中
    multilabelmarginloss_0d_no_reduce_test(),
    # 执行 multilabelmarginloss_1d_no_reduce_test 函数，将其结果添加到列表中
    multilabelmarginloss_1d_no_reduce_test(),
    # 执行 multilabelmarginloss_index_neg_test 函数，将其结果添加到列表中
    multilabelmarginloss_index_neg_test(),
    # 执行 multilabelmarginloss_no_reduce_test 函数，将其结果添加到列表中
    multilabelmarginloss_no_reduce_test(),
    # 执行 hingeembeddingloss_no_reduce_test 函数，将其结果添加到列表中
    hingeembeddingloss_no_reduce_test(),
    # 执行 hingeembeddingloss_margin_no_reduce_test 函数，将其结果添加到列表中
    hingeembeddingloss_margin_no_reduce_test(),
    # 执行 softmarginloss_no_reduce_test 函数，将其结果添加到列表中
    softmarginloss_no_reduce_test(),
    # 执行 multilabelsoftmarginloss_no_reduce_test 函数，将其结果添加到列表中
    multilabelsoftmarginloss_no_reduce_test(),
    # 执行 multilabelsoftmarginloss_weights_no_reduce_test 函数，将其结果添加到列表中
    multilabelsoftmarginloss_weights_no_reduce_test(),
    # 执行 multimarginloss_no_reduce_test 函数，将其结果添加到列表中
    multimarginloss_no_reduce_test(),
    # 执行 multimarginloss_1d_no_reduce_test 函数，将其结果添加到列表中
    multimarginloss_1d_no_reduce_test(),
    # 执行 multimarginloss_1d_input_0d_target_no_reduce_test 函数，将其结果添加到列表中
    multimarginloss_1d_input_0d_target_no_reduce_test(),
    # 执行 multimarginloss_p_no_reduce_test 函数，将其结果添加到列表中
    multimarginloss_p_no_reduce_test(),
    # 执行 multimarginloss_margin_no_reduce_test 函数，将其结果添加到列表中
    multimarginloss_margin_no_reduce_test(),
    # 执行 multimarginloss_weights_no_reduce_test 函数，将其结果添加到列表中
    multimarginloss_weights_no_reduce_test(),
    # 创建一个字典，描述 Conv1d 模块的测试用例
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)',
        input_size=(2, 4, 10),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
        default_dtype=torch.double,
    ),
    # 创建一个字典，描述带有步幅参数的 Conv1d 模块的测试用例
    dict(
        module_name='Conv1d',
        constructor_args=(4, 5, 3, 2),
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(2)',
        input_size=(2, 4, 10),
        cudnn=True,
        desc='stride',
        with_tf32=True,
        tf32_precision=0.005,
        default_dtype=torch.double,
    ),
]
    dict(
        module_name='Conv1d',  # 指定使用的模块为 Conv1d
        constructor_args=(4, 5, 3, 1, 1),  # 构造函数的位置参数：输入通道数=4，输出通道数=5，卷积核大小=3，步长=1，填充=1
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(1).padding(1)',  # 对应 C++ 构造函数的参数表示
        input_size=(2, 4, 10),  # 输入数据的大小为 batch=2, channels=4, length=10
        cudnn=True,  # 使用 cuDNN 加速
        desc='pad1',  # 描述：使用填充为 1 的情况
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.01,  # TF32 模式下的精度设置为 0.01
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        module_name='Conv1d',  # 指定使用的模块为 Conv1d
        constructor_args=(4, 5, 5, 1, 2),  # 构造函数的位置参数：输入通道数=4，输出通道数=5，卷积核大小=5，步长=1，填充=2
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 5).stride(1).padding(2)',  # 对应 C++ 构造函数的参数表示
        input_size=(2, 4, 10),  # 输入数据的大小为 batch=2, channels=4, length=10
        cudnn=True,  # 使用 cuDNN 加速
        desc='pad2',  # 描述：使用填充为 2 的情况
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        module_name='Conv1d',  # 指定使用的模块为 Conv1d
        constructor_args=(4, 4, 3, 1, 1),  # 构造函数的位置参数：输入通道数=4，输出通道数=4，卷积核大小=3，步长=1，填充=1
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 3).stride(1).padding(1)',  # 对应 C++ 构造函数的参数表示
        input_size=(1, 4, 1),  # 输入数据的大小为 batch=1, channels=4, length=1
        cudnn=True,  # 使用 cuDNN 加速
        desc='pad1size1',  # 描述：使用填充为 1 的情况，且输入数据长度为 1 的情况
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        module_name='Conv1d',  # 指定使用的模块为 Conv1d
        constructor_args=(4, 4, 5, 1, 2),  # 构造函数的位置参数：输入通道数=4，输出通道数=4，卷积核大小=5，步长=1，填充=2
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 5).stride(1).padding(2)',  # 对应 C++ 构造函数的参数表示
        input_size=(1, 4, 1),  # 输入数据的大小为 batch=1, channels=4, length=1
        cudnn=True,  # 使用 cuDNN 加速
        desc='pad2size1',  # 描述：使用填充为 2 的情况，且输入数据长度为 1 的情况
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        module_name='Conv1d',  # 指定使用的模块为 Conv1d
        constructor_args=(4, 5, 3),  # 构造函数的位置参数：输入通道数=4，输出通道数=5，卷积核大小=3
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)',  # 对应 C++ 构造函数的参数表示
        input_size=(0, 4, 10),  # 输入数据的大小为 batch=0, channels=4, length=10
        cudnn=True,  # 使用 cuDNN 加速
        desc='zero_batch',  # 描述：输入 batch 大小为 0 的情况
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
    ),
    dict(
        fullname='Conv1d_dilated',  # 指定使用的模块为带 dilation 的 Conv1d
        constructor=lambda: nn.Conv1d(4, 5, kernel_size=3, dilation=2),  # 使用 dilation 参数的构造函数
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).dilation(2)',  # 对应 C++ 构造函数的参数表示
        input_size=(2, 4, 10),  # 输入数据的大小为 batch=2, channels=4, length=10
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv1d_groups',  # 指定使用的模块为带 groups 的 Conv1d
        constructor=lambda: nn.Conv1d(4, 6, kernel_size=3, groups=2),  # 使用 groups 参数的构造函数
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 6, 3).groups(2)',  # 对应 C++ 构造函数的参数表示
        input_size=(2, 4, 6),  # 输入数据的大小为 batch=2, channels=4, length=6
        cudnn=True,  # 使用 cuDNN 加速
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv1d_pad_valid',  # 指定使用的模块为 Conv1d，padding 模式为 valid
        constructor=lambda: nn.Conv1d(4, 5, 3, padding="valid"),  # 使用 padding="valid" 的构造函数
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kValid)',  # 对应 C++ 构造函数的参数表示
        input_size=(2, 4, 10),  # 输入数据的大小为 batch=2, channels=4, length=10
        cudnn=True,  # 使用 cuDNN 加速
        with_tf32=True,  # 支持 TF32 模式
        tf32_precision=0.005,  # TF32 模式下的精度设置为 0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv1d_pad_same',  # 指定使用的模块为 Conv1d，padding 模式为 same
        constructor=lambda: nn.Conv1d(4, 5, 3, padding="same"),  # 使用 padding="same
    # 创建一个字典，描述一个 Conv1d 模型的相关参数和配置
    dict(
        fullname='Conv1d_pad_same2',  # 模型名称
        constructor=lambda: nn.Conv1d(4, 5, 4, padding="same"),  # 使用 lambda 表达式定义 Conv1d 模型的构造方法
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 4).padding(torch::kSame)',  # 对应的 C++ 构造方法参数
        input_size=(2, 4, 10),  # 输入张量的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    # 创建另一个字典，描述带有 dilation 参数的 Conv1d 模型的相关配置
    dict(
        fullname='Conv1d_pad_same_dilated',  # 模型名称
        constructor=lambda: nn.Conv1d(4, 5, 4, padding="same", dilation=2),  # 使用 lambda 表达式定义带有 dilation 参数的 Conv1d 模型的构造方法
        cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).padding(torch::kSame).dilation(2)',  # 对应的 C++ 构造方法参数，包括 dilation
        input_size=(2, 4, 10),  # 输入张量的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    # 创建一个字典，描述 ConvTranspose1d 模型的相关参数和配置
    dict(
        fullname='ConvTranspose1d',  # 模型名称
        constructor=lambda: nn.ConvTranspose1d(3, 4, kernel_size=3, stride=(3,), padding=1, output_padding=(1,)),  # 使用 lambda 表达式定义 ConvTranspose1d 模型的构造方法
        cpp_constructor_args='torch::nn::ConvTranspose1dOptions(3, 4, 3).stride(3).padding(1).output_padding(1)',  # 对应的 C++ 构造方法参数
        cudnn=True,  # 是否使用 cuDNN 加速
        input_size=(1, 3, 7),  # 输入张量的尺寸
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    # 创建一个字典，描述不带 bias 参数的 ConvTranspose1d 模型的相关配置
    dict(
        module_name='ConvTranspose1d',  # 模块名称
        constructor_args=(3, 4, 3, 2, 1, 1, 1, False),  # 构造方法的参数元组
        cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(3, 4, 3)
                                .stride(2).padding(1).output_padding(1).groups(1).bias(false)''',  # 对应的 C++ 构造方法参数，包括不带 bias
        input_size=(1, 3, 6),  # 输入张量的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        desc='no_bias',  # 描述信息，表示没有 bias 参数
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    # 创建一个字典，描述带有 dilation 和 bias 参数的 ConvTranspose1d 模型的相关配置
    dict(
        module_name='ConvTranspose1d',  # 模块名称
        constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2),  # 构造方法的参数元组，包括 dilation 和 bias
        cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(3, 4, 3)
                                .stride(2).padding(1).output_padding(1).groups(1).bias(true).dilation(2)''',  # 对应的 C++ 构造方法参数，包括 dilation 和 bias
        input_size=(1, 3, 6),  # 输入张量的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        desc='dilated',  # 描述信息，表示使用了 dilation 参数
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    # 创建一个字典，描述带有 groups 参数的 ConvTranspose1d 模型的相关配置
    dict(
        fullname='ConvTranspose1d_groups',  # 模型名称
        constructor=lambda: nn.ConvTranspose1d(4, 6, 3, stride=(3,), padding=1, output_padding=(1,), groups=2),  # 使用 lambda 表达式定义带有 groups 参数的 ConvTranspose1d 模型的构造方法
        cpp_constructor_args='''torch::nn::ConvTranspose1dOptions(4, 6, 3)
                                .stride(3).padding(1).output_padding(1).groups(2)''',  # 对应的 C++ 构造方法参数，包括 groups
        cudnn=True,  # 是否使用 cuDNN 加速
        input_size=(2, 4, 7),  # 输入张量的尺寸
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    # 创建一个字典，描述 Conv2d 模型的相关参数和配置
    dict(
        module_name='Conv2d',  # 模块名称
        constructor_args=(3, 4, (3, 2)),  # 构造方法的参数元组
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})',  # 对应的 C++ 构造方法参数
        input_size=(2, 3, 7, 5),  # 输入张量的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        check_with_long_tensor=True,  # 是否使用长张量进行检查
        with_tf32=True,  # 是否支持 TF32 精度
        tf32_precision=0.005,  # TF32 精度的设置
        default_dtype=torch.double,  # 默认张量数据类型
    ),
    dict(
        module_name='Conv2d',  # 模块名称为 Conv2d，表示是一个二维卷积层
        constructor_args=(3, 4, (3, 3), (2, 2)),  # 构造函数参数：输入通道数为3，输出通道数为4，卷积核大小为3x3，步长为2x2
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2})',  # 对应的 C++ 构造参数字符串
        input_size=(2, 3, 6, 6),  # 输入数据的尺寸为：批次数为2，通道数为3，高度和宽度均为6
        cudnn=True,  # 是否使用 CuDNN 加速
        desc='strided',  # 描述：带步长的卷积
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
        tf32_precision=0.005,  # TF32 模式的精度
        default_dtype=torch.double,  # 默认张量类型为双精度
    ),
    dict(
        module_name='Conv2d',  # 模块名称为 Conv2d，表示是一个二维卷积层
        constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)),  # 构造函数参数：输入通道数为3，输出通道数为4，卷积核大小为3x3，步长为2x2，填充为1x1
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2}).padding({1, 1})',  # 对应的 C++ 构造参数字符串
        input_size=(2, 3, 6, 6),  # 输入数据的尺寸同上
        cudnn=True,  # 是否使用 CuDNN 加速
        desc='padding',  # 描述：带填充的卷积
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
        tf32_precision=0.005,  # TF32 模式的精度
        default_dtype=torch.double,  # 默认张量类型为双精度
    ),
    dict(
        module_name='Conv2d',  # 模块名称为 Conv2d，表示是一个二维卷积层
        constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)),  # 构造函数参数：输入通道数为3，输出通道数为2，卷积核大小为3x3，步长为2x2，填充为1x1，扩展为2x2
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 2, {3, 3}).stride({2, 2}).padding({1, 1}).dilation({2, 2})',  # 对应的 C++ 构造参数字符串
        input_size=(2, 3, 8, 8),  # 输入数据的尺寸为：批次数为2，通道数为3，高度和宽度均为8
        cudnn=True,  # 是否使用 CuDNN 加速
        desc='dilated',  # 描述：带扩展的卷积
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
        tf32_precision=0.005,  # TF32 模式的精度
        default_dtype=torch.double,  # 默认张量类型为双精度
    ),
    dict(
        module_name='Conv2d',  # 模块名称为 Conv2d，表示是一个二维卷积层
        constructor_args=(3, 4, (3, 2), 1, 0, 1, 1, False),  # 构造函数参数：输入通道数为3，输出通道数为4，卷积核大小为3x2，步长为1，填充为0，扩展为1x1，组数为1，无偏置
        cpp_constructor_args='''torch::nn::Conv2dOptions(3, 4, {3, 2})
                                .stride(1).padding(0).dilation(1).groups(1).bias(false)''',  # 对应的 C++ 构造参数字符串
        input_size=(2, 3, 6, 5),  # 输入数据的尺寸为：批次数为2，通道数为3，高度为6，宽度为5
        cudnn=True,  # 是否使用 CuDNN 加速
        desc='no_bias',  # 描述：无偏置的卷积
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
        tf32_precision=0.015,  # TF32 模式的精度
        default_dtype=torch.double,  # 默认张量类型为双精度
    ),
    dict(
        module_name='Conv2d',  # 模块名称为 Conv2d，表示是一个二维卷积层
        constructor_args=(3, 4, (3, 2)),  # 构造函数参数：输入通道数为3，输出通道数为4，卷积核大小为3x2
        cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})',  # 对应的 C++ 构造参数字符串
        input_size=(0, 3, 7, 5),  # 输入数据的尺寸为：批次数为0，通道数为3，高度为7，宽度为5
        cudnn=True,  # 是否使用 CuDNN 加速
        desc='zero_batch',  # 描述：零批次的卷积
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
    ),
    dict(
        fullname='Conv2d_groups',  # 模块全名为 Conv2d_groups
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),  # 构造函数使用 Lambda 表达式创建，输入通道数为4，输出通道数为6，卷积核大小为3x2，组数为2
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)',  # 对应的 C++ 构造参数字符串
        input_size=(2, 4, 6, 5),  # 输入数据的尺寸为：批次数为2，通道数为4，高度为6，宽度为5
        cudnn=True,  # 是否使用 CuDNN 加速
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
        tf32_precision=0.015,  # TF32 模式的精度
        default_dtype=torch.double,  # 默认张量类型为双精度
    ),
    dict(
        fullname='Conv2d_groups_thnn',  # 模块全名为 Conv2d_groups_thnn
        constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2),  # 构造函数使用 Lambda 表达式创建，输入通道数为4，输出通道数为6，卷积核大小为3x2，组数为2
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)',  # 对应的 C++ 构造参数字符串
        input_size=(2, 4, 6, 5),  # 输入数据的尺寸同上
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 模式
    dict(
        fullname='Conv2d_pad_valid',
        constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="valid"),
        cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kValid)',
        input_size=(2, 2, 6, 5),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.005,
        default_dtype=torch.double,
    ),

    # 定义一个包含卷积参数和设置的字典，用于创建 "valid" padding 的 2D 卷积层
    dict(
        fullname='Conv2d_pad_same',
        constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="same"),
        cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kSame)',
        input_size=(2, 2, 6, 5),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.01,
        default_dtype=torch.double,
    ),

    # 定义一个包含卷积参数和设置的字典，用于创建 "same" padding 的 2D 卷积层
    dict(
        fullname='Conv2d_pad_same_dilated',
        constructor=lambda: nn.Conv2d(2, 4, (3, 4), padding="same", dilation=2),
        cpp_constructor_args='torch::nn::Conv2dOptions(2, 4, {3, 4}).padding(torch::kSame).dilation(2)',
        input_size=(2, 2, 6, 5),
        cudnn=True,
        with_tf32=True,
        tf32_precision=0.01,
        default_dtype=torch.double,
    ),

    # 定义一个包含卷积参数和设置的字典，用于创建 "same" padding 和 dilation 的 2D 卷积层
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (3, 2), 1, (1, 1)),
        cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                .stride({3, 2}).padding(1).output_padding({1, 1})''',
        cudnn=True,
        input_size=(1, 3, 7, 6),
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
        default_dtype=torch.double,
    ),

    # 定义一个包含转置卷积参数和设置的字典，用于创建转置卷积层
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)),
        cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                .stride({2, 3})
                                .padding(1)
                                .output_padding({1, 1})
                                .groups(1)
                                .bias(false)
                                .dilation({2, 2})''',
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='dilated',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
        default_dtype=torch.double,
    ),

    # 定义一个包含转置卷积参数和设置的字典，用于创建转置卷积层，不包含偏置
    dict(
        module_name='ConvTranspose2d',
        constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False),
        cpp_constructor_args='''torch::nn::ConvTranspose2dOptions(3, 4, 3)
                                .stride({2, 3}).padding(1).output_padding({1, 1}).groups(1).bias(false)''',
        input_size=(1, 3, 6, 7),
        cudnn=True,
        desc='no_bias',
        check_with_long_tensor=True,
        with_tf32=True,
        tf32_precision=0.01,
        default_dtype=torch.double,
    ),
    dict(
        fullname='ConvTranspose2d_groups',  # 定义一个字典项，表示转置卷积层的组卷积设置
        constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2),  # 使用nn.ConvTranspose2d创建转置卷积层，设置输入通道为2，输出通道为4，卷积核大小为(2, 3)，使用2个分组
        cpp_constructor_args='torch::nn::ConvTranspose2dOptions(2, 4, {2, 3}).groups(2)',  # 对应的C++构造选项，设置输入通道为2，输出通道为4，卷积核大小为{2, 3}，使用2个分组
        input_size=(1, 2, 4, 5),  # 输入数据的大小为(1, 2, 4, 5)，表示(batch_size, channels, height, width)
        cudnn=True,  # 使用CuDNN加速
        check_with_long_tensor=True,  # 使用长整型张量进行检查
        with_tf32=True,  # 支持TF32精度
        tf32_precision=0.01,  # TF32精度设置为0.01
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv2d_depthwise',  # 定义一个字典项，表示深度可分离卷积层
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), groups=4),  # 使用nn.Conv2d创建深度可分离卷积层，设置输入输出通道都为4，卷积核大小为(3, 3)，使用4个分组
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).groups(4)',  # 对应的C++构造选项，设置输入输出通道都为4，卷积核大小为{3, 3}，使用4个分组
        input_size=(2, 4, 6, 6),  # 输入数据的大小为(2, 4, 6, 6)，表示(batch_size, channels, height, width)
        with_tf32=True,  # 支持TF32精度
        tf32_precision=0.005,  # TF32精度设置为0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv2d_depthwise_with_multiplier',  # 定义一个字典项，表示带乘数的深度可分离卷积层
        constructor=lambda: nn.Conv2d(4, 8, (3, 3), groups=4),  # 使用nn.Conv2d创建深度可分离卷积层，设置输入通道为4，输出通道为8，卷积核大小为(3, 3)，使用4个分组
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 8, {3, 3}).groups(4)',  # 对应的C++构造选项，设置输入通道为4，输出通道为8，卷积核大小为{3, 3}，使用4个分组
        input_size=(2, 4, 6, 6),  # 输入数据的大小为(2, 4, 6, 6)，表示(batch_size, channels, height, width)
        with_tf32=True,  # 支持TF32精度
        tf32_precision=0.005,  # TF32精度设置为0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv2d_depthwise_strided',  # 定义一个字典项，表示带步幅的深度可分离卷积层
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), stride=(2, 2), groups=4),  # 使用nn.Conv2d创建深度可分离卷积层，设置输入输出通道都为4，卷积核大小为(3, 3)，步幅为(2, 2)，使用4个分组
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).stride({2, 2}).groups(4)',  # 对应的C++构造选项，设置输入输出通道都为4，卷积核大小为{3, 3}，步幅为{2, 2}，使用4个分组
        input_size=(2, 4, 6, 6),  # 输入数据的大小为(2, 4, 6, 6)，表示(batch_size, channels, height, width)
        with_tf32=True,  # 支持TF32精度
        tf32_precision=0.005,  # TF32精度设置为0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv2d_depthwise_padded',  # 定义一个字典项，表示带填充的深度可分离卷积层
        constructor=lambda: nn.Conv2d(4, 4, (3, 3), padding=(1, 1), groups=4),  # 使用nn.Conv2d创建深度可分离卷积层，设置输入输出通道都为4，卷积核大小为(3, 3)，填充为(1, 1)，使用4个分组
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).padding({1, 1}).groups(4)',  # 对应的C++构造选项，设置输入输出通道都为4，卷积核大小为{3, 3}，填充为{1, 1}，使用4个分组
        input_size=(2, 4, 6, 6),  # 输入数据的大小为(2, 4, 6, 6)，表示(batch_size, channels, height, width)
        with_tf32=True,  # 支持TF32精度
        tf32_precision=0.005,  # TF32精度设置为0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        fullname='Conv2d_depthwise_dilated',  # 定义一个字典项，表示带扩张的深度可分离卷积层
        constructor=lambda: nn.Conv2d(4, 4, (2, 2), dilation=(2, 2), groups=4),  # 使用nn.Conv2d创建深度可分离卷积层，设置输入输出通道都为4，卷积核大小为(2, 2)，扩张率为(2, 2)，使用4个分组
        cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {2, 2}).dilation({2, 2}).groups(4)',  # 对应的C++构造选项，设置输入输出通道都为4，卷积核大小为{2, 2}，扩张率为{2, 2}，使用4个分组
        input_size=(2, 4, 5, 5),  # 输入数据的大小为(2, 4, 5, 5)，表示(batch_size, channels, height, width)
        with_tf32=True,  # 支持TF32精度
        tf32_precision=0.005,  # TF32精度设置为0.005
        default_dtype=torch.double,  # 默认数据类型为双精度浮点型
    ),
    dict(
        module_name='Conv3d',  # 定义一个字典项，表示3D卷积层
        constructor_args=(2, 3, (2, 3, 2)),  # 使用nn.Conv3d创建3D卷积层，设置输入通道为2，输出通道为3，卷积核大小为(2, 3, 2)
        cpp_constructor_args='torch::nn::Conv3dOptions(2, 3, {2, 3,
    dict(
        module_name='Conv3d',  # 指定模块名称为 Conv3d
        constructor_args=(2, 3, (1, 1, 1), 1, 0, 1, 1, False),  # Conv3d 构造函数参数：输入通道数 2，输出通道数 3，卷积核大小 (1, 1, 1)，步长 1，填充 0，膨胀系数 1，分组数 1，不使用偏置
        cpp_constructor_args='''torch::nn::Conv3dOptions(2, 3, {2, 3, 4})  # 对应的 C++ 构造函数参数
                                .stride(1).padding(0).dilation(1).groups(1).bias(false)''',  # 设置步长、填充、膨胀、分组及是否使用偏置
        input_size=(1, 2, 3, 4, 5),  # 输入数据的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        desc='1x1x1_no_bias',  # 描述：1x1x1 没有偏置
        check_with_long_tensor=False,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 精度
        tf32_precision=0.05,  # TF32 精度
        default_dtype=torch.double,  # 默认的数据类型为双精度
    ),
    dict(
        module_name='Conv3d',  # 指定模块名称为 Conv3d
        constructor_args=(3, 4, 2, 2),  # Conv3d 构造函数参数：输入通道数 3，输出通道数 4，卷积核大小 2，步长 2
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2)',  # 对应的 C++ 构造函数参数，设置步长
        input_size=(2, 3, 5, 5, 5),  # 输入数据的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        desc='stride',  # 描述：带有步长
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 精度
        tf32_precision=0.05,  # TF32 精度
        default_dtype=torch.double,  # 默认的数据类型为双精度
    ),
    dict(
        module_name='Conv3d',  # 指定模块名称为 Conv3d
        constructor_args=(3, 4, 2, 2, 1),  # Conv3d 构造函数参数：输入通道数 3，输出通道数 4，卷积核大小 2，步长 2，填充 1
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2).padding(1)',  # 对应的 C++ 构造函数参数，设置步长和填充
        input_size=(2, 3, 5, 5, 5),  # 输入数据的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        desc='stride_padding',  # 描述：带有步长和填充
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 精度
        tf32_precision=0.05,  # TF32 精度
        default_dtype=torch.double,  # 默认的数据类型为双精度
    ),
    dict(
        module_name='Conv3d',  # 指定模块名称为 Conv3d
        constructor_args=(3, 4, (2, 3, 4)),  # Conv3d 构造函数参数：输入通道数 3，输出通道数 4，卷积核大小 (2, 3, 4)
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4})',  # 对应的 C++ 构造函数参数，设置卷积核大小
        input_size=(0, 3, 3, 4, 5),  # 输入数据的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        desc='zero_batch',  # 描述：零批次
        with_tf32=True,  # 是否使用 TF32 精度
    ),
    dict(
        fullname='Conv3d_groups',  # 完整名称为 Conv3d_groups
        constructor=lambda: nn.Conv3d(2, 4, kernel_size=3, groups=2),  # 使用 lambda 函数定义的 Conv3d 构造函数，指定输入通道数 2，输出通道数 4，卷积核大小 3，分组数 2
        cpp_constructor_args='torch::nn::Conv3dOptions(2, 4, 3).groups(2)',  # 对应的 C++ 构造函数参数，设置分组数
        input_size=(1, 2, 4, 5, 4),  # 输入数据的尺寸
        cudnn=True,  # 是否使用 cuDNN 加速
        check_with_long_tensor=True,  # 是否使用长整型张量进行检查
        with_tf32=True,  # 是否使用 TF32 精度
        tf32_precision=0.005,  # TF32 精度
        default_dtype=torch.double,  # 默认的数据类型为双精度
    ),
    dict(
        fullname='Conv3d_dilated',  # 完整名称为 Conv3d_dilated
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2),  # 使用 lambda 函数定义的 Conv3d 构造函数，指定输入通道数 3，输出通道数 4，卷积核大小 2，膨胀系数 2
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2)',  # 对应的 C++ 构造函数参数，设置膨胀系数
        input_size=(2, 3, 5, 5, 5),  # 输入数据的尺寸
        with_tf32=True,  # 是否使用 TF32 精度
        tf32_precision=0.05,  # TF32 精度
        default_dtype=torch.double,  # 默认的数据类型为双精度
    ),
    dict(
        fullname='Conv3d_dilated_strided',  # 完整名称为 Conv3d_dilated_strided
        constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2, stride=2),  # 使用 lambda 函数定义的 Conv3d 构造函数，指定输入通道数 3，输出通道数 4，卷积核大小 2，膨胀系数 2，步长 2
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2).stride(2)',  # 对应的 C++ 构造函数参数，设置膨胀系数和步长
        input_size=(2, 3, 5, 5, 5),  # 输入数据的尺寸
        with_tf32=True,  # 是否使用 TF32 精度
        tf32_precision=0.05,  # TF32 精度
        default_dtype=torch.double,  # 默认的数据类型为双精度
    ),
    dict(
        fullname='Conv3d_pad_valid',  # 完整名称为 Conv3d_pad_valid
        constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="valid"),  # 使用 lambda 函数定义的 Conv3d 构造函数，指定输入通道数 3，输出通道数 4，卷积核大小 (2, 3, 4)，使用 valid padding
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kValid)',  # 对应的 C++ 构造函数参数，设置卷积核大小和 padding 类型
        input
    # 创建一个包含多个键值对的字典，描述了不同的神经网络模块和它们的配置
    dict(
        fullname='Conv3d_pad_same',  # 模块名称为 Conv3d_pad_same
        constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="same"),  # 使用 nn.Conv3d 创建一个 3D 卷积层对象，输入通道数为 3，输出通道数为 4，卷积核大小为 (2, 3, 4)，使用 "same" 方式填充
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kSame)',  # 对应的 C++ 构造器参数
        input_size=(2, 3, 6, 5, 4),  # 输入数据的大小为 (2, 3, 6, 5, 4)
        cudnn=True,  # 支持 cudnn 加速
        with_tf32=True,  # 使用 TF32 混合精度加速
        tf32_precision=0.05,  # TF32 操作的精度
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    ),
    dict(
        fullname='Conv3d_pad_same_dilated',  # 模块名称为 Conv3d_pad_same_dilated
        constructor=lambda: nn.Conv3d(3, 4, (2, 3, 4), padding="same", dilation=2),  # 使用 nn.Conv3d 创建一个 3D 卷积层对象，输入通道数为 3，输出通道数为 4，卷积核大小为 (2, 3, 4)，使用 "same" 方式填充，并设置 dilation 参数为 2
        cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4}).padding(torch::kSame).dilation(2)',  # 对应的 C++ 构造器参数，包括 dilation 设置
        input_size=(2, 3, 6, 5, 4),  # 输入数据的大小为 (2, 3, 6, 5, 4)
        cudnn=True,  # 支持 cudnn 加速
        with_tf32=True,  # 使用 TF32 混合精度加速
        tf32_precision=0.05,  # TF32 操作的精度
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    ),
    dict(
        module_name='ConvTranspose3d',  # 模块名称为 ConvTranspose3d
        constructor_args=(2, 3, (2, 3, 2)),  # 使用 nn.ConvTranspose3d 创建对象，输入通道数为 2，输出通道数为 3，卷积核大小为 (2, 3, 2)
        cpp_constructor_args='torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})',  # 对应的 C++ 构造器参数
        cudnn=True,  # 支持 cudnn 加速
        input_size=(1, 2, 4, 5, 4),  # 输入数据的大小为 (1, 2, 4, 5, 4)
        with_tf32=True,  # 使用 TF32 混合精度加速
        tf32_precision=0.05,  # TF32 操作的精度
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    ),
    dict(
        module_name='ConvTranspose3d',  # 模块名称为 ConvTranspose3d
        constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)),  # 使用 nn.ConvTranspose3d 创建对象，输入通道数为 2，输出通道数为 3，卷积核大小为 (2, 3, 2)，设置 stride=1, padding=0, output_padding=0, groups=1, bias=True，并设置 dilation=(2, 2, 2)
        cpp_constructor_args='''torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})
                                .stride(1).padding(0).output_padding(0).groups(1).bias(true).dilation({2, 2, 2})''',  # 对应的 C++ 构造器参数，包括各个参数的设置
        cudnn=True,  # 支持 cudnn 加速
        input_size=(1, 2, 4, 5, 4),  # 输入数据的大小为 (1, 2, 4, 5, 4)
        desc='dilated',  # 描述为 dilated
        with_tf32=True,  # 使用 TF32 混合精度加速
        tf32_precision=0.05,  # TF32 操作的精度
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    ),
    dict(
        module_name='ReplicationPad3d',  # 模块名称为 ReplicationPad3d
        constructor_args=((1, 2, 3, 3, 2, 1),),  # 使用 nn.ReplicationPad3d 创建对象，填充参数为 (1, 2, 3, 3, 2, 1)
        cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',  # 对应的 C++ 构造器参数
        input_size=(2, 3, 2, 2, 2),  # 输入数据的大小为 (2, 3, 2, 2, 2)
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    ),
    dict(
        module_name='ReplicationPad3d',  # 模块名称为 ReplicationPad3d
        constructor_args=((1, 2, 3, 3, 2, 1),),  # 使用 nn.ReplicationPad3d 创建对象，填充参数为 (1, 2, 3, 3, 2, 1)
        cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',  # 对应的 C++ 构造器参数
        input_size=(3, 2, 2, 2),  # 输入数据的大小为 (3, 2, 2, 2)
        reference_fn=single_batch_reference_fn,  # 参考函数为 single_batch_reference_fn
        desc='no_batch_dim',  # 描述为 no_batch_dim
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    ),
    dict(
        module_name='ReplicationPad3d',  # 模块名称为 ReplicationPad3d
        constructor_args=((1, 2, 3, 3, 2, 1),),  # 使用 nn.ReplicationPad3d 创建对象，填充参数为 (1, 2, 3, 3, 2, 1)
        cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 3, 2, 1})',  # 对应的 C++ 构造器参数
        input_fn=lambda: torch.rand(2, 3, 2, 2, 2, dtype=torch.complex128, requires_grad=True),  # 输入数据的生成函数，生成大小为 (2, 3, 2, 2, 2) 的复杂数据张量，数据类型为复数双精度浮点数，需要计算梯度
        skip_half=True,  # 跳过一半的操作
        desc='complex'  # 描述为 complex
    ),
    dict(
        module_name='Embedding',  # 模块名称为 Embedding
        constructor_args=(4, 3),  # 使用 nn.Embedding 创建对象，词典大小为 4，嵌入维度为 3
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',  # 对应的 C++ 构造器参数
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),  # 输入数据的生成函数，生成大小为 (2, 3) 的长整型
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',
        input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),
        check_gradgrad=False,
        desc='discontiguous',
        default_dtype=torch.double,
        decorator=skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/117971")
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
        desc='mean',
        default_dtype=torch.double,
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',
        input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),
        check_gradgrad=False,
        desc='discontiguous',
        default_dtype=torch.double,
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3, None, 2., False, 'sum'),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum)''',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
        desc='sum',
        default_dtype=torch.double,
    ),
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3, None, 2., False, 'max'),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax)''',
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),
        check_gradgrad=False,
        desc='max',
        default_dtype=torch.double,
    ),
    dict(
        fullname='EmbeddingBag_mean_padding_idx',
        constructor=lambda: nn.EmbeddingBag(4, 3, padding_idx=1),
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3).padding_idx(1)',
        input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
        check_gradgrad=False,
        default_dtype=torch.double,
    ),
    dict(
        fullname='EmbeddingBag_sum_padding_idx',
        constructor=lambda: nn.EmbeddingBag(4, 3, None, 2., False, 'sum', padding_idx=1),
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum).padding_idx(1)''',
        input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),
        check_gradgrad=False,
        default_dtype=torch.double,
    ),


注释：


    # 定义一个字典，描述一个 Embedding 模块的配置和参数
    dict(
        module_name='Embedding',
        constructor_args=(4, 3),  # 构造函数参数，分别是嵌入矩阵的大小和维度
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)',  # 对应 C++ 接口的构造函数参数
        input_fn=lambda: torch.empty(1, 512, dtype=torch.long).random_(4).expand(7, 512),  # 生成输入数据的函数
        check_gradgrad=False,  # 是否检查二阶导数的梯度
        desc='discontiguous',  # 描述该配置的性质，此处为不连续性
        default_dtype=torch.double,  # 默认的数据类型
        decorator=skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/117971")  # 装饰器，根据条件跳过测试
    ),
    # 定义一个字典，描述一个 EmbeddingBag 模块的配置和参数
    dict(
        module_name='EmbeddingBag',
        constructor_args=(4, 3),  # 构造函数参数，分别是嵌入矩阵的大小和维度
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)',  # 对应 C++ 接口的构造函数参数
        input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4),  # 生成输入数据的函数
        check_gradgrad=False,  # 是否检查二阶导数的梯度
        desc='mean',  # 描述该配置的性质，此处为均值计算
        default_dtype=torch.double,  # 默认的数据类型
    ),
    # 后续的字典以相同的方式描述不同的 EmbeddingBag 模块配置和参数，依次类推
    dict(
        fullname='EmbeddingBag_max_padding_idx',  # 设置字典条目的全名
        constructor=lambda: nn.EmbeddingBag(4, 3, None, 2., False, 'max', padding_idx=1),  # 创建 EmbeddingBag 模块的构造函数
        cpp_constructor_args='''torch::nn::EmbeddingBagOptions(4, 3)
                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax).padding_idx(1)''',  # 对应 C++ 构造函数的参数
        input_fn=lambda: torch.stack([torch.randperm(3), torch.randperm(3)]),  # 定义输入函数，生成随机排列的张量堆栈
        check_gradgrad=False,  # 关闭梯度检查
        default_dtype=torch.double,  # 设置默认的数据类型为双精度浮点数
    ),
    dict(
        fullname='EmbeddingBag_sparse',  # 设置字典条目的全名
        constructor=lambda: nn.EmbeddingBag(4, 3, sparse=True, dtype=torch.double),  # 创建支持稀疏张量的 EmbeddingBag 模块的构造函数
        cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3).sparse(true)._weight(torch::rand({4, 3}).to(torch::kFloat64))',  # 对应 C++ 构造函数的参数
        input_fn=lambda: torch.randperm(2).repeat(1, 2),  # 定义输入函数，生成随机排列的张量，并重复堆叠
        check_gradgrad=False,  # 关闭梯度检查
        has_sparse_gradients=True,  # 指示是否支持稀疏梯度
    ),
    dict(
        constructor=lambda: nn.Embedding(4, 3, dtype=torch.double, sparse=True),  # 创建支持稀疏张量的 Embedding 模块的构造函数
        cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3).sparse(true)._weight(torch::rand({4, 3}).to(torch::kFloat64))',  # 对应 C++ 构造函数的参数
        input_fn=lambda: torch.randperm(2).repeat(1, 2),  # 定义输入函数，生成随机排列的张量，并重复堆叠
        fullname='Embedding_sparse',  # 设置字典条目的全名
        check_gradgrad=False,  # 关闭梯度检查
        has_sparse_gradients=True,  # 指示是否支持稀疏梯度
    ),
    dict(
        module_name='PixelShuffle',  # 设置模块名称
        constructor_args=(3,),  # 设置构造函数的参数
        cpp_constructor_args='torch::nn::PixelShuffleOptions(3)',  # 对应 C++ 构造函数的参数
        input_size=(1, 9, 4, 4),  # 设置输入张量的大小
        default_dtype=torch.double,  # 设置默认的数据类型为双精度浮点数
    ),
    dict(
        module_name='PixelUnshuffle',  # 设置模块名称
        constructor_args=(3,),  # 设置构造函数的参数
        cpp_constructor_args='torch::nn::PixelUnshuffleOptions(3)',  # 对应 C++ 构造函数的参数
        input_size=(1, 1, 12, 12),  # 设置输入张量的大小
        default_dtype=torch.double,  # 设置默认的数据类型为双精度浮点数
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),  # 使用函数包装器创建插值函数的构造函数
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)''',  # 对应 C++ 构造函数的参数
        input_size=(1, 2, 4),  # 设置输入张量的大小
        fullname='interpolate_nearest_1d',  # 设置字典条目的全名
        pickle=False,  # 禁用 Pickle 序列化
        default_dtype=torch.double,  # 设置默认的数据类型为双精度浮点数
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),  # 使用函数包装器创建插值函数的构造函数
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)''',  # 对应 C++ 构造函数的参数
        input_size=(0, 2, 4),  # 设置输入张量的大小
        fullname='interpolate_nearest_1d_zero_dim',  # 设置字典条目的全名
        pickle=False,  # 禁用 Pickle 序列化
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'),  # 使用函数包装器创建插值函数的构造函数
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)''',  # 对应 C++ 构造函数的参数
        input_size=(1, 2, 3),  # 设置输入张量的大小
        fullname='interpolate_nearest_tuple_1d',  # 设置字典条目的全名
        pickle=False,  # 禁用 Pickle 序列化
        default_dtype=torch.double,  # 设置默认的数据类型为双精度浮点数
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        # 对应的 C++ 选项参数字符串
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt).scale_factor(std::vector<double>({4.})).mode(torch::kNearest)''',
        # 输入的张量大小
        input_size=(1, 2, 4),
        # 函数名称
        fullname='interpolate_nearest_scale_1d',
        # 是否序列化
        pickle=False,
        # 默认的数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
        # 对应的 C++ 选项参数字符串
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        # 输入的张量大小
        input_size=(1, 2, 4),
        # 函数名称
        fullname='interpolate_linear_1d',
        # 是否序列化
        pickle=False,
        # 默认的数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=(4, ), scale_factor=None, mode='linear', align_corners=False),
        # 对应的 C++ 选项参数字符串
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        # 输入的张量大小
        input_size=(1, 2, 3),
        # 函数名称
        fullname='interpolate_linear_tuple_1d',
        # 是否序列化
        pickle=False,
        # 默认的数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=False),
        # 对应的 C++ 选项参数字符串
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4.}))
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        # 输入的张量大小
        input_size=(1, 2, 4),
        # 函数名称
        fullname='interpolate_linear_scale_1d',
        # 是否序列化
        pickle=False,
        # 默认的数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False),
        # 对应的 C++ 选项参数字符串
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(false)''',
        # 输入的张量大小
        input_size=(0, 2, 4),
        # 函数名称
        fullname='interpolate_linear_1d_zero_dim',
        # 是否序列化
        pickle=False,
    ),
    # 创建一个字典，包含不同配置的插值函数调用所需的参数和选项
    dict(
        # 使用 wrap_functional 包装 F.interpolate 函数，设置插值大小为 12，不指定缩放因子，插值模式为线性，角落对齐为真
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=True),
        # 定义用于 C++ 选项的参数字符串，指定插值函数的配置选项
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kLinear)
                            .align_corners(true)''',
        # 输入数据的大小为 (1, 2, 4)
        input_size=(1, 2, 4),
        # 函数的完整名称为 interpolate_linear_1d_align_corners
        fullname='interpolate_linear_1d_align_corners',
        # 不支持 pickle 操作
        pickle=False,
        # 默认数据类型为双精度浮点型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 包装 F.interpolate 函数，设置插值大小为 None，缩放因子为 4.0，插值模式为线性，角落对齐为真
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='linear', align_corners=True),
        # 定义用于 C++ 选项的参数字符串，指定插值函数的配置选项
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4.}))
                            .mode(torch::kLinear)
                            .align_corners(true)''',
        # 输入数据的大小为 (1, 2, 4)
        input_size=(1, 2, 4),
        # 函数的完整名称为 interpolate_linear_scale_1d_align_corners
        fullname='interpolate_linear_scale_1d_align_corners',
        # 不支持 pickle 操作
        pickle=False,
        # 默认数据类型为双精度浮点型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 包装 F.interpolate 函数，设置插值大小为 2，不指定缩放因子，插值模式为最近邻
        constructor=wrap_functional(F.interpolate, size=2, scale_factor=None, mode='nearest'),
        # 定义用于 C++ 选项的参数字符串，指定插值函数的配置选项
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({2, 2}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        # 输入数据的大小为 (1, 128, 1, 1)
        input_size=(1, 128, 1, 1),
        # 函数的完整名称为 interpolate_nearest_2d_launch_configs
        fullname='interpolate_nearest_2d_launch_configs',
        # 不支持 pickle 操作
        pickle=False,
        # 默认数据类型为双精度浮点型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 包装 F.interpolate 函数，设置插值大小为 12，不指定缩放因子，插值模式为最近邻
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        # 定义用于 C++ 选项的参数字符串，指定插值函数的配置选项
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        # 输入数据的大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数的完整名称为 interpolate_nearest_2d
        fullname='interpolate_nearest_2d',
        # 不支持 pickle 操作
        pickle=False,
        # 默认数据类型为双精度浮点型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 包装 F.interpolate 函数，设置插值大小为 (12, 16)，不指定缩放因子，插值模式为最近邻
        constructor=wrap_functional(F.interpolate, size=(12, 16), scale_factor=None, mode='nearest'),
        # 定义用于 C++ 选项的参数字符串，指定插值函数的配置选项
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 16}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        # 输入数据的大小为 (1, 2, 3, 4)
        input_size=(1, 2, 3, 4),
        # 函数的完整名称为 interpolate_nearest_tuple_2d
        fullname='interpolate_nearest_tuple_2d',
        # 不支持 pickle 操作
        pickle=False,
        # 默认数据类型为双精度浮点型
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kNearest)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_nearest_scale_2d',
        pickle=False,
        default_dtype=torch.double,
    ),

    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        input_size=(0, 2, 4, 4),
        fullname='interpolate_nearest_2d_zero_dim',
        pickle=False,
    ),

    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_2d',
        pickle=False,
        default_dtype=torch.double,
    ),

    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(0, 2, 4, 4),
        fullname='interpolate_bilinear_2d_zero_dim',
        pickle=False,
    ),

    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 2, 3),
        fullname='interpolate_bilinear_tuple_2d',
        pickle=False,
        default_dtype=torch.double,
    ),



# 解释每个字典的内容，构建了不同的插值参数和选项，用于PyTorch的插值功能。
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.,
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_2d',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 2.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_shared_2d',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=False),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_scale_tuple_skewed_2d',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=True),
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(true)''',
        input_size=(1, 2, 4, 4),
        fullname='interpolate_bilinear_tuple_2d_align_corners',
        pickle=False,
        default_dtype=torch.double,
    ),


注释：


    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设定参数：尺寸为 None，缩放因子为 4.0，
        # 模式为双线性插值，不对齐角点
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.,
                                    mode='bilinear', align_corners=False),
        # 使用 C++ 配置选项参数字符串，指定插值选项：尺寸为默认空值，缩放因子为 (4.0, 4.0)，
        # 模式为双线性插值，不对齐角点
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bilinear_scale_2d
        fullname='interpolate_bilinear_scale_2d',
        # 不支持数据的 pickle 操作
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设定参数：尺寸为 None，缩放因子为 (2.0, 2.0)，
        # 模式为双线性插值，不对齐角点
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bilinear', align_corners=False),
        # 使用 C++ 配置选项参数字符串，指定插值选项：尺寸为默认空值，缩放因子为 (2.0, 2.0)，
        # 模式为双线性插值，不对齐角点
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 2.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bilinear_scale_tuple_shared_2d
        fullname='interpolate_bilinear_scale_tuple_shared_2d',
        # 不支持数据的 pickle 操作
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设定参数：尺寸为 None，缩放因子为 (2.0, 1.0)，
        # 模式为双线性插值，不对齐角点
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=False),
        # 使用 C++ 配置选项参数字符串，指定插值选项：尺寸为默认空值，缩放因子为 (2.0, 1.0)，
        # 模式为双线性插值，不对齐角点
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBilinear)
                            .align_corners(false)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bilinear_scale_tuple_skewed_2d
        fullname='interpolate_bilinear_scale_tuple_skewed_2d',
        # 不支持数据的 pickle 操作
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，设定参数：尺寸为 (4, 6)，缩放因子为默认空值，
        # 模式为双线性插值，对齐角点
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=True),
        # 使用 C++ 配置选项参数字符串，指定插值选项：尺寸为 (4, 6)，缩放因子为默认空值，
        # 模式为双线性插值，对齐角点
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBilinear)
                            .align_corners(true)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bilinear_tuple_2d_align_corners
        fullname='interpolate_bilinear_tuple_2d_align_corners',
        # 不支持数据的 pickle 操作
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 方法，配置如下参数：
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bilinear', align_corners=True),
        # 定义 C++ 选项参数字符串，配置如下选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBilinear)
                            .align_corners(true)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 'interpolate_bilinear_scale_tuple_skewed_2d_align_corners'
        fullname='interpolate_bilinear_scale_tuple_skewed_2d_align_corners',
        # 不进行 pickle
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 方法，配置如下参数：
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
        # 定义 C++ 选项参数字符串，配置如下选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 'interpolate_bicubic_2d'
        fullname='interpolate_bicubic_2d',
        # 不进行 pickle
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 方法，配置如下参数：
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False),
        # 定义 C++ 选项参数字符串，配置如下选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        # 输入数据大小为 (0, 2, 4, 4)
        input_size=(0, 2, 4, 4),
        # 函数名称为 'interpolate_bicubic_2d_zero_dim'
        fullname='interpolate_bicubic_2d_zero_dim',
        # 不进行 pickle
        pickle=False,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 方法，配置如下参数：
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None,
                                    mode='bicubic', align_corners=False),
        # 定义 C++ 选项参数字符串，配置如下选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        # 输入数据大小为 (1, 2, 2, 3)
        input_size=(1, 2, 2, 3),
        # 函数名称为 'interpolate_bicubic_tuple_2d'
        fullname='interpolate_bicubic_tuple_2d',
        # 不进行 pickle
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 方法，配置如下参数：
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='bicubic', align_corners=False),
        # 定义 C++ 选项参数字符串，配置如下选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4.}))
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        # 输入数据大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 'interpolate_bicubic_scale_2d'
        fullname='interpolate_bicubic_scale_2d',
        # 不进行 pickle
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，配置参数如下：
        # size=None 表示不指定输出大小
        # scale_factor=(2., 2.) 表示沿两个维度放大两倍
        # mode='bicubic' 表示使用双三次插值
        # align_corners=False 表示不对齐角点
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 2.),
                                    mode='bicubic', align_corners=False),
        # cpp_options_args 包含 C++ 函数选项的字符串表示
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 2.}))
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        # 输入数据的大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bicubic_scale_tuple_shared_2d
        fullname='interpolate_bicubic_scale_tuple_shared_2d',
        # 不支持序列化
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，配置参数如下：
        # size=None 表示不指定输出大小
        # scale_factor=(2., 1.) 表示沿两个维度一个方向放大两倍，另一个方向不变
        # mode='bicubic' 表示使用双三次插值
        # align_corners=False 表示不对齐角点
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bicubic', align_corners=False),
        # cpp_options_args 包含 C++ 函数选项的字符串表示
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBicubic)
                            .align_corners(false)''',
        # 输入数据的大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bicubic_scale_tuple_skewed_2d
        fullname='interpolate_bicubic_scale_tuple_skewed_2d',
        # 不支持序列化
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，配置参数如下：
        # size=(4, 6) 表示输出大小为 (4, 6)
        # scale_factor=None 表示不进行尺度因子的放大
        # mode='bicubic' 表示使用双三次插值
        # align_corners=True 表示对齐角点
        constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bicubic', align_corners=True),
        # cpp_options_args 包含 C++ 函数选项的字符串表示
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kBicubic)
                            .align_corners(true)''',
        # 输入数据的大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bicubic_tuple_2d_align_corners
        fullname='interpolate_bicubic_tuple_2d_align_corners',
        # 不支持序列化
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，配置参数如下：
        # size=None 表示不指定输出大小
        # scale_factor=(2., 1.) 表示沿两个维度一个方向放大两倍，另一个方向不变
        # mode='bicubic' 表示使用双三次插值
        # align_corners=True 表示对齐角点
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2., 1.),
                                    mode='bicubic', align_corners=True),
        # cpp_options_args 包含 C++ 函数选项的字符串表示
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({2., 1.}))
                            .mode(torch::kBicubic)
                            .align_corners(true)''',
        # 输入数据的大小为 (1, 2, 4, 4)
        input_size=(1, 2, 4, 4),
        # 函数名称为 interpolate_bicubic_scale_tuple_skewed_2d_align_corners
        fullname='interpolate_bicubic_scale_tuple_skewed_2d_align_corners',
        # 不支持序列化
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 封装 F.interpolate 函数，配置参数如下：
        # size=12 表示输出大小为 (12, 12, 12)
        # scale_factor=None 表示不进行尺度因子的放大
        # mode='nearest' 表示使用最近邻插值
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        # cpp_options_args 包含 C++ 函数选项的字符串表示
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        # 输入数据的大小为 (1, 2, 4, 4, 4)
        input_size=(1, 2, 4, 4, 4),
        # 函数名称为 interpolate_nearest_3d
        fullname='interpolate_nearest_3d',
        # 不支持序列化
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'),
        # 使用 wrap_functional 封装 F.interpolate 函数，设置 size=12，scale_factor=None，mode='nearest'
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        # 为 C++ 选项参数创建字符串，指定 size=(12, 12, 12)，scale_factor=c10::nullopt，mode=torch::kNearest
        input_size=(0, 2, 4, 4, 4),
        # 设置输入尺寸为 (0, 2, 4, 4, 4)
        fullname='interpolate_nearest_3d_zero_dim',
        # 设置全名为 'interpolate_nearest_3d_zero_dim'
        pickle=False,
        # 禁用 pickle
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=(12, 16, 16), scale_factor=None, mode='nearest'),
        # 使用 wrap_functional 封装 F.interpolate 函数，设置 size=(12, 16, 16)，scale_factor=None，mode='nearest'
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 16, 16}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kNearest)''',
        # 为 C++ 选项参数创建字符串，指定 size=(12, 16, 16)，scale_factor=c10::nullopt，mode=torch::kNearest
        input_size=(1, 2, 3, 4, 4),
        # 设置输入尺寸为 (1, 2, 3, 4, 4)
        fullname='interpolate_nearest_tuple_3d',
        # 设置全名为 'interpolate_nearest_tuple_3d'
        pickle=False,
        # 禁用 pickle
        default_dtype=torch.double,
        # 设置默认数据类型为 torch.double
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=4., mode='nearest'),
        # 使用 wrap_functional 封装 F.interpolate 函数，设置 size=None，scale_factor=4.0，mode='nearest'
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({4., 4., 4.}))
                            .mode(torch::kNearest)''',
        # 为 C++ 选项参数创建字符串，指定 size=c10::nullopt，scale_factor=(4.0, 4.0, 4.0)，mode=torch::kNearest
        input_size=(1, 2, 4, 4, 4),
        # 设置输入尺寸为 (1, 2, 4, 4, 4)
        fullname='interpolate_nearest_scale_3d',
        # 设置全名为 'interpolate_nearest_scale_3d'
        pickle=False,
        # 禁用 pickle
        default_dtype=torch.double,
        # 设置默认数据类型为 torch.double
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
        # 使用 wrap_functional 封装 F.interpolate 函数，设置 size=12，scale_factor=None，mode='trilinear'，align_corners=False
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        # 为 C++ 选项参数创建字符串，指定 size=(12, 12, 12)，scale_factor=c10::nullopt，mode=torch::kTrilinear，align_corners=false
        input_size=(1, 2, 4, 4, 4),
        # 设置输入尺寸为 (1, 2, 4, 4, 4)
        fullname='interpolate_trilinear_3d',
        # 设置全名为 'interpolate_trilinear_3d'
        pickle=False,
        # 禁用 pickle
        default_dtype=torch.double,
        # 设置默认数据类型为 torch.double
    ),
    dict(
        constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False),
        # 使用 wrap_functional 封装 F.interpolate 函数，设置 size=12，scale_factor=None，mode='trilinear'，align_corners=False
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({12, 12, 12}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        # 为 C++ 选项参数创建字符串，指定 size=(12, 12, 12)，scale_factor=c10::nullopt，mode=torch::kTrilinear，align_corners=false
        input_size=(0, 2, 4, 4, 4),
        # 设置输入尺寸为 (0, 2, 4, 4, 4)
        fullname='interpolate_trilinear_3d_zero_dim',
        # 设置全名为 'interpolate_trilinear_3d_zero_dim'
        pickle=False,
        # 禁用 pickle
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=(4, 6, 6),
                                    scale_factor=None, mode='trilinear', align_corners=False),
        # 用于 C++ 参数的字符串表示，指定了插值操作的选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        # 输入张量的大小
        input_size=(1, 2, 2, 3, 3),
        # 函数的全名标识
        fullname='interpolate_trilinear_tuple_3d',
        # 是否支持序列化
        pickle=False,
        # 默认数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=False),
        # 用于 C++ 参数的字符串表示，指定了插值操作的选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({3., 3., 3.}))
                            .mode(torch::kTrilinear)
                            .align_corners(false)''',
        # 输入张量的大小
        input_size=(1, 2, 3, 4, 5),
        # 函数的全名标识
        fullname='interpolate_trilinear_scale_3d',
        # 精度设置，参考指定的 GitHub 问题链接
        precision=3e-4,
        # 是否支持序列化
        pickle=False,
        # 默认数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=(4, 6, 6), scale_factor=None,
                                    mode='trilinear', align_corners=True),
        # 用于 C++ 参数的字符串表示，指定了插值操作的选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>({4, 6, 6}))
                            .scale_factor(c10::nullopt)
                            .mode(torch::kTrilinear)
                            .align_corners(true)''',
        # 输入张量的大小
        input_size=(1, 2, 2, 3, 3),
        # 函数的全名标识
        fullname='interpolate_trilinear_tuple_3d_align_corners',
        # 是否支持序列化
        pickle=False,
        # 默认数据类型
        default_dtype=torch.double
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.interpolate 函数，设置参数如下：
        constructor=wrap_functional(F.interpolate, size=None, scale_factor=3., mode='trilinear', align_corners=True),
        # 用于 C++ 参数的字符串表示，指定了插值操作的选项：
        cpp_options_args='''F::InterpolateFuncOptions()
                            .size(c10::nullopt)
                            .scale_factor(std::vector<double>({3., 3., 3.}))
                            .mode(torch::kTrilinear)
                            .align_corners(true)''',
        # 输入张量的大小
        input_size=(1, 2, 3, 4, 4),
        # 函数的全名标识
        fullname='interpolate_trilinear_scale_3d_align_corners',
        # 精度设置，参考指定的 GitHub 问题链接
        precision=3e-4,
        # 是否支持序列化
        pickle=False,
        # 默认数据类型
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数包装 F.softmax 函数，设置参数如下：
        constructor=wrap_functional(F.softmax, dim=-1),
        # 用于 C++ 参数的字符串表示，指定了 softmax 操作的选项：
        cpp_options_args='F::SoftmaxFuncOptions(-1)',
        # 输入张量的大小，触发 CUDA 中的最后维度算法
        input_size=(2, 128),
        # 函数的全名标识
        fullname='softmax_lastdim',
        # 是否支持序列化
        pickle=False,
        # 默认数据类型
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
        input_size=(2, 128),
        fullname='softmax_lastdim_dtype',
        pickle=False,
        test_cuda=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        cpp_options_args='F::SoftmaxFuncOptions(1)',
        input_size=(2, 128, 2, 2),  # 触发空间 CUDA 算法的特殊情况
        fullname='softmax_spatial_special',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1),
        cpp_options_args='F::SoftmaxFuncOptions(1)',
        input_size=(2, 2, 4, 4),  # 常规空间算法
        fullname='softmax_spatial',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64),
        cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)',
        input_size=(2, 2, 4, 4),  # 常规空间算法
        fullname='softmax_spatial_dtype',
        pickle=False,
        test_cuda=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=0),
        cpp_options_args='F::SoftmaxFuncOptions(0)',
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim0',
        test_cuda=False,
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=3),
        cpp_options_args='F::SoftmaxFuncOptions(3)',
        input_size=(2, 3, 4, 5),
        fullname='softmax_functional_dim3',
        test_cuda=False,
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.softmax, dim=-1),
        cpp_options_args='F::SoftmaxFuncOptions(-1)',
        input_size=(),
        fullname='softmax_functional_scalar',
        test_cuda=False,
        pickle=False,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=-1),
        cpp_options_args='F::LogSoftmaxFuncOptions(-1)',
        input_size=(2, 128),  # 触发 CUDA 中的最后一维算法
        fullname='log_softmax_lastdim',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        cpp_options_args='F::LogSoftmaxFuncOptions(1)',
        input_size=(2, 128, 2, 2),  # 触发空间 CUDA 算法的特殊情况
        fullname='log_softmax_spatial_special',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        constructor=wrap_functional(F.log_softmax, dim=1),
        cpp_options_args='F::LogSoftmaxFuncOptions(1)',
        input_size=(2, 2, 4, 4),  # 常规空间算法
        fullname='log_softmax_spatial',
        pickle=False,
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数封装 F.log_softmax，指定维度为0的对数softmax函数
        constructor=wrap_functional(F.log_softmax, dim=0),
        # 对应的 C++ 函数选项参数字符串表示
        cpp_options_args='F::LogSoftmaxFuncOptions(0)',
        # 输入张量的大小为 (2, 3, 4, 5)
        input_size=(2, 3, 4, 5),
        # 完整名称为 'log_softmax_dim0'
        fullname='log_softmax_dim0',
        # 不进行pickle操作
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数封装 F.log_softmax，指定维度为3的对数softmax函数
        constructor=wrap_functional(F.log_softmax, dim=3),
        # 对应的 C++ 函数选项参数字符串表示
        cpp_options_args='F::LogSoftmaxFuncOptions(3)',
        # 输入张量的大小为 (2, 3, 4, 5)
        input_size=(2, 3, 4, 5),
        # 完整名称为 'log_softmax_dim3'
        fullname='log_softmax_dim3',
        # 不进行pickle操作
        pickle=False,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 使用 wrap_functional 函数封装 F.log_softmax，指定维度为0的对数softmax函数
        constructor=wrap_functional(F.log_softmax, dim=0),
        # 对应的 C++ 函数选项参数字符串表示
        cpp_options_args='F::LogSoftmaxFuncOptions(0)',
        # 输入张量的大小为空元组，表示标量
        input_size=(),
        # 完整名称为 'log_softmax_scalar'
        fullname='log_softmax_scalar',
        # 不进行pickle操作
        pickle=False,
    ),
    dict(
        # 完整名称为 'Unfold'，使用 nn.Unfold 构造函数生成对象
        fullname='Unfold',
        # 使用 lambda 函数构造 nn.Unfold 对象，指定参数：kernel_size=(2, 2)，dilation=(1, 1)，padding=(0, 0)，stride=(1, 1)
        constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)),
        # 对应的 C++ 构造函数选项参数字符串表示
        cpp_constructor_args='torch::nn::UnfoldOptions({2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        # 输入张量的大小为 (2, 4, 3, 3)
        input_size=(2, 4, 3, 3),
        # 不检查梯度二阶导数
        check_gradgrad=False,
        # 在 CUDA 上进行测试
        test_cuda=True,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 完整名称为 'Fold'，使用 nn.Fold 构造函数生成对象
        fullname='Fold',
        # 使用 lambda 函数构造 nn.Fold 对象，指定参数：output_size=(3, 3)，kernel_size=(2, 2)，dilation=(1, 1)，padding=(0, 0)，stride=(1, 1)
        constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
        # 对应的 C++ 构造函数选项参数字符串表示
        cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        # 输入张量的大小为 (2, 16, 4)
        input_size=(2, 16, 4),
        # 不检查梯度二阶导数
        check_gradgrad=False,
        # 在 CUDA 上进行测试
        test_cuda=True,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 完整名称为 'Fold_no_batch_dim_input'，使用 nn.Fold 构造函数生成对象
        fullname='Fold_no_batch_dim_input',
        # 使用 lambda 函数构造 nn.Fold 对象，指定参数：output_size=(3, 3)，kernel_size=(2, 2)，dilation=(1, 1)，padding=(0, 0)，stride=(1, 1)
        constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)),
        # 对应的 C++ 构造函数选项参数字符串表示
        cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})',
        # 输入张量的大小为 (16, 4)
        input_size=(16, 4),
        # 不检查梯度二阶导数
        check_gradgrad=False,
        # 使用单批次参考函数作为参考
        ref=single_batch_reference_fn,
        # 在 CUDA 上进行测试
        test_cuda=True,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 完整名称为 'Unfold_int_input'，使用 nn.Unfold 构造函数生成对象
        fullname='Unfold_int_input',
        # 使用 lambda 函数构造 nn.Unfold 对象，指定参数：kernel_size=2，dilation=1，padding=0，stride=1
        constructor=lambda: nn.Unfold(2, 1, 0, 1),
        # 对应的 C++ 构造函数选项参数字符串表示
        cpp_constructor_args='torch::nn::UnfoldOptions(2).dilation(1).padding(0).stride(1)',
        # 输入张量的大小为 (2, 4, 3, 3)
        input_size=(2, 4, 3, 3),
        # 不检查梯度二阶导数
        check_gradgrad=False,
        # 在 CUDA 上进行测试
        test_cuda=True,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 完整名称为 'Fold_int_input'，使用 nn.Fold 构造函数生成对象
        fullname='Fold_int_input',
        # 使用 lambda 函数构造 nn.Fold 对象，指定参数：output_size=3，kernel_size=2，dilation=1，padding=0，stride=1
        constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
        # 对应的 C++ 构造函数选项参数字符串表示
        cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)',
        # 输入张量的大小为 (2, 16, 4)
        input_size=(2, 16, 4),
        # 不检查梯度二阶导数
        check_gradgrad=False,
        # 在 CUDA 上进行测试
        test_cuda=True,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        # 完整名称为 'Fold_no_batch_dim_int_input'，使用 nn.Fold 构造函数生成对象
        fullname='Fold_no_batch_dim_int_input',
        # 使用 lambda 函数构造 nn.Fold 对象，指定参数：output_size=3，kernel_size=2，dilation=1，padding=0，stride=1
        constructor=lambda: nn.Fold(3, 2, 1, 0, 1),
        # 对应的 C++ 构造函数选项参数字符串表示
        cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)',
        # 输入张量的大小为 (16, 4)
        input_size=(16, 4),
        # 不检查梯度二阶导数
        check_gradgrad=False,
        # 使用单批次参考函数作为参考
        ref=single_batch_reference_fn,
        # 在 CUDA 上进行测试
        test_cuda=True,
        # 默认数据类型为 torch.double
        default_dtype=torch.double,
    ),
    dict(
        module_name='RReLU',  # 模块名为 'RReLU'
        constructor_args=(0.1, 0.9),  # 构造函数参数为 (0.1, 0.9)
        cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)',  # 对应的 C++ 构造函数参数
        input_size=(),  # 输入大小为空元组
        desc='with_up_down_scalar',  # 描述为 'with_up_down_scalar'
        test_cuda=False,  # 不测试 CUDA
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='PairwiseDistance',  # 模块名为 'PairwiseDistance'
        input_fn=lambda: (torch.randn(10, 8), torch.randn(10, 8)),  # 输入函数生成输入数据
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='PairwiseDistance',  # 模块名为 'PairwiseDistance'
        input_fn=lambda: (torch.randn(10, 1), torch.randn(10, 8)),  # 输入函数生成输入数据
        desc='broadcast_lhs',  # 描述为 'broadcast_lhs'
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='PairwiseDistance',  # 模块名为 'PairwiseDistance'
        input_fn=lambda: (torch.randn(10, 8), torch.randn(1, 8)),  # 输入函数生成输入数据
        desc='broadcast_rhs',  # 描述为 'broadcast_rhs'
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='PairwiseDistance',  # 模块名为 'PairwiseDistance'
        constructor_args=(1.5, 1e-05, True),  # 构造函数参数为 (1.5, 1e-05, True)
        cpp_constructor_args='torch::nn::PairwiseDistanceOptions().p(1.5).eps(1e-05).keepdim(true)',  # 对应的 C++ 构造函数参数
        input_fn=lambda: (torch.randn(10, 8), torch.randn(10, 8)),  # 输入函数生成输入数据
        desc='with_non_default_args',  # 描述为 'with_non_default_args'
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='PairwiseDistance',  # 模块名为 'PairwiseDistance'
        input_fn=lambda: (torch.randn(8), torch.randn(8)),  # 输入函数生成输入数据（没有批次维度）
        reference_fn=single_batch_reference_fn,  # 参考函数为 single_batch_reference_fn
        desc='no_batch_dim',  # 描述为 'no_batch_dim'
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='TransformerEncoderLayer',  # 模块名为 'TransformerEncoderLayer'
        constructor_args=(4, 2, 16, 0.0),  # 构造函数参数为 (4, 2, 16, 0.0)
        cpp_constructor_args='''torch::nn::TransformerEncoderLayerOptions(4, 2)
                                .dim_feedforward(16)
                                .dropout(0.0)''',  # 对应的 C++ 构造函数参数
        input_size=(2, 3, 4),  # 输入大小为 (2, 3, 4)
        desc='relu_activation',  # 描述为 'relu_activation'
        with_tf32=True,  # 使用 TF32
        tf32_precision=0.1,  # TF32 精度为 0.1
        # TODO(#50743): figure out the error
        # RuntimeError: The size of tensor a (6) must match the size of tensor b (4)
        # at non-singleton dimension 2
        check_batched_grad=False,  # 不检查批量梯度
        check_gradgrad=False,  # 不检查 gradgrad
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='TransformerEncoderLayer',  # 模块名为 'TransformerEncoderLayer'
        constructor_args=(4, 2, 8, 0.0, F.gelu),  # 构造函数参数为 (4, 2, 8, 0.0, F.gelu)
        cpp_constructor_args='''torch::nn::TransformerEncoderLayerOptions(4, 2)
                                .dim_feedforward(8)
                                .dropout(0.0)
                                .activation(torch::kGELU)''',  # 对应的 C++ 构造函数参数
        input_size=(2, 3, 4),  # 输入大小为 (2, 3, 4)
        check_gradgrad=False,  # 不检查 gradgrad
        desc='gelu_activation',  # 描述为 'gelu_activation'
        with_tf32=True,  # 使用 TF32
        tf32_precision=0.08 if SM90OrLater else 0.05,  # 根据条件设置 TF32 精度
        default_dtype=torch.double,  # 默认数据类型为 torch.double
    ),
    dict(
        module_name='TransformerDecoderLayer',  # 模块名称为 TransformerDecoderLayer
        constructor_args=(4, 2, 8, 0.0),  # 构造函数参数为 (4, 2, 8, 0.0)
        cpp_constructor_args='''torch::nn::TransformerDecoderLayerOptions(4, 2)
                                .dim_feedforward(8)
                                .dropout(0.0)''',  # 对应的 C++ 构造函数参数字符串
        input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4)),  # 输入数据生成函数，返回两个随机张量
        check_gradgrad=False,  # 是否检查梯度的梯度，默认为 False
        desc='relu_activation',  # 描述字段，表示使用的是 relu 激活函数
        with_tf32=True,  # 是否启用 TF32 加速
        tf32_precision=0.05,  # TF32 加速时的精度
        default_dtype=torch.double,  # 默认张量数据类型为 double
    ),
    dict(
        module_name='TransformerDecoderLayer',  # 模块名称为 TransformerDecoderLayer
        constructor_args=(4, 2, 8, 0.0, F.gelu),  # 构造函数参数为 (4, 2, 8, 0.0, F.gelu)
        cpp_constructor_args='''torch::nn::TransformerDecoderLayerOptions(4, 2)
                                .dim_feedforward(8)
                                .dropout(0.0)
                                .activation(torch::kGELU)''',  # 对应的 C++ 构造函数参数字符串
        input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4)),  # 输入数据生成函数，返回两个随机张量
        check_gradgrad=False,  # 是否检查梯度的梯度，默认为 False
        desc='gelu_activation',  # 描述字段，表示使用的是 gelu 激活函数
        with_tf32=True,  # 是否启用 TF32 加速
        tf32_precision=0.05,  # TF32 加速时的精度
        default_dtype=torch.double,  # 默认张量数据类型为 double
    ),
    dict(
        module_name='Transformer',  # 模块名称为 Transformer
        constructor_args=(4, 2, 2, 2, 8, 0.0, F.relu),  # 构造函数参数为 (4, 2, 2, 2, 8, 0.0, F.relu)
        cpp_constructor_args='''torch::nn::TransformerOptions()
                                .d_model(4)
                                .nhead(2)
                                .num_encoder_layers(2)
                                .num_decoder_layers(2)
                                .dim_feedforward(8)
                                .dropout(0.0)
                                .activation(torch::kReLU)''',  # 对应的 C++ 构造函数参数字符串
        input_fn=lambda: (torch.rand(3, 3, 4), torch.rand(2, 3, 4), torch.rand(3, 3)),  # 输入数据生成函数，返回三个随机张量
        check_gradgrad=False,  # 是否检查梯度的梯度，默认为 False
        desc='multilayer_coder',  # 描述字段，表示使用的是多层编码器
        with_tf32=True,  # 是否启用 TF32 加速
        tf32_precision=0.05 if SM90OrLater else 0.03,  # TF32 加速时的精度，根据条件选择不同的值
        default_dtype=torch.double,  # 默认张量数据类型为 double
    ),
    dict(
        module_name='Linear',  # 模块名称为 Linear
        constructor_args=(3, 5),  # 构造函数参数为 (3, 5)
        cpp_constructor_args='torch::nn::LinearOptions(3, 5)',  # 对应的 C++ 构造函数参数字符串
        input_fn=lambda: torch.rand(3),  # 输入数据生成函数，返回一个大小为 3 的随机张量
        reference_fn=lambda i, p, _: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1],  # 参考函数，执行线性变换
        desc="no_batch_dim",  # 描述字段，表示没有批次维度
        with_tf32=True,  # 是否启用 TF32 加速
        tf32_precision=0.005,  # TF32 加速时的精度
        default_dtype=torch.double,  # 默认张量数据类型为 double
    ),
    dict(
        module_name='Flatten',  # 模块名称为 Flatten
        cpp_constructor_args='torch::nn::FlattenOptions().start_dim(-3).end_dim(-1)',  # 对应的 C++ 构造函数参数字符串
        constructor_args=(-3, -1),  # 构造函数参数为 (-3, -1)
        input_size=(3, 4, 5),  # 输入尺寸为 (3, 4, 5)
        reference_fn=single_batch_reference_fn,  # 参考函数，执行单批处理的参考函数
        desc="no_batch_dim",  # 描述字段，表示没有批次维度
        default_dtype=torch.double,  # 默认张量数据类型为 double
    ),
    dict(
        module_name='Unflatten',  # 模块名称为 Unflatten
        cpp_constructor_args='torch::nn::UnflattenOptions(-2, {2, 2})',  # 对应的 C++ 构造函数参数字符串
        constructor_args=(-2, torch.Size([2, 2])),  # 构造函数参数为 (-2, [2, 2])
        input_size=(3, 4, 5),  # 输入尺寸为 (3, 4, 5)
        reference_fn=single_batch_reference_fn,  # 参考函数，执行单批处理的参考函数
        desc="no_batch_dim",  # 描述字段，表示没有批次维度
        default_dtype=torch.double,  # 默认张量数据类型为 double
    ),
    # 创建一个字典对象，包含以下键值对
    dict(
        # 模块名称为 'LayerNorm'
        module_name='LayerNorm',
        # 构造函数的参数为 ([56, 56, 56], 1e-5, False)
        constructor_args=([56, 56, 56], 1e-5, False),
        # 对应的 C++ 构造函数参数字符串表示为 'torch::nn::LayerNormOptions({56, 56, 56}).eps(1e-5).elementwise_affine(false)'
        cpp_constructor_args='torch::nn::LayerNormOptions({56, 56, 56}).eps(1e-5).elementwise_affine(false)',
        # 输入大小为 (4, 56, 56, 56)
        input_size=(4, 56, 56, 56),
        # 使用 cuDNN 加速
        cudnn=True,
        # 检查评估模式
        check_eval=True,
        # 使用快速梯度检查模式
        gradcheck_fast_mode=True,
        # 检查半精度浮点数
        check_half=True,
        # 描述为 '3d_no_affine_large_feature'
        desc='3d_no_affine_large_feature',
    ),
# add conv padding mode tests:
# 添加卷积填充模式的测试

for padding_mode, cpp_padding_mode in zip(
        ['reflect', 'circular', 'replicate', 'zeros'],
        ['torch::kReflect', 'torch::kCircular', 'torch::kReplicate', 'torch::kZeros']):
    # 对于每种填充模式和对应的 C++ 填充模式进行迭代

    # conv signature:
    #     in_channels, out_channels, kernel_size, stride=1,
    #     padding=0, dilation=1, groups=1,
    #     bias=True, padding_mode='zeros'
    # 卷积函数签名说明，包括输入通道数、输出通道数、核大小、步幅、填充、膨胀、分组、偏置和填充模式

    for d in (1, 2, 3):
        # 对于每个维度 d (1维、2维、3维)

        if d == 3 and padding_mode == 'reflect':
            # 如果维度为3且填充模式为'reflect'
            # FIXME: remove after implementing reflection pad 3d
            #        https://github.com/pytorch/pytorch/issues/27655
            # FIXME：在实现3D反射填充后移除此部分，详见链接

            continue  # 跳过当前循环，不执行以下代码

        padding = tuple(range(1, d + 1))
        # 生成长度为 d 的填充元组，例如 (1,)、(1, 2)、(1, 2, 3)

        cpp_padding = '{' + ', '.join(map(str, padding)) + '}'
        # 生成 C++ 中填充模式的字符串表示，例如 '{1}'、'{1, 2}'、'{1, 2, 3}'

        input_size = (2, 2) + (4,) * d
        # 输入的大小，例如对于二维：(2, 2, 4, 4)，三维：(2, 2, 4, 4, 4)

        output_size = (2, 3) + tuple(p + 1 for p in padding)
        # 输出的大小，简化自 '(4 + 2 * p - 3) // 2 + 1'，例如对于二维：(2, 3, 2, 2)，三维类似

        new_module_tests.append(
            dict(
                module_name=f'Conv{d}d',
                # 模块名称，例如 'Conv1d'、'Conv2d'、'Conv3d'
                constructor_args=(2, 3, 3, 2, padding, 1, 1, True, padding_mode),
                # 构造函数参数，包括输入通道、输出通道、核大小、步幅、填充、膨胀、分组、偏置和填充模式

                cpp_constructor_args=f'''torch::nn::Conv{d}dOptions(2, 3, 3)
                                        .stride(2)
                                        .padding({cpp_padding})
                                        .dilation(1)
                                        .groups(1)
                                        .bias(true)
                                        .padding_mode({cpp_padding_mode})''',
                # 对应的 C++ 构造函数参数，作为字符串提供

                input_size=input_size,
                # 输入的大小

                output_size=output_size,
                # 输出的大小

                cudnn=True,
                # 是否使用 cuDNN 加速

                desc=f'{padding_mode}_stride2_pad2',
                # 描述信息，例如 'reflect_stride2_pad2'

                with_tf32=True,
                # 是否支持 TF32 精度

                tf32_precision=0.05,
                # TF32 精度

                default_dtype=torch.double,
                # 默认的数据类型为双精度浮点数
            ),
        )

# Check that non linear activations work with no batch dimensions
# 检查非线性激活函数在无批处理维度时是否工作正常

non_linear_activations_no_batch = [
    'ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'Hardswish', 'LeakyReLU',
    'LogSigmoid', 'PReLU', 'ReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'GLU',
    'Sigmoid', 'SiLU', 'Mish', 'Softplus', 'Softshrink', 'Softsign', 'Tanh',
    'Tanhshrink', 'Threshold'
]
# 包含非线性激活函数名称的列表，无批处理维度时应工作正常

non_linear_activations_extra_info: Dict[str, dict] = {
    'CELU': {'constructor_args': (2.,), 'default_dtype': torch.double},
    # 额外信息字典，例如 'CELU' 的构造函数参数和默认数据类型为双精度浮点数

    'Threshold': {'constructor_args': (2., 1.)},
    # 'Threshold' 的构造函数参数为 (2., 1.)

    'Hardsigmoid': {'check_gradgrad': False, 'check_jit': False, 'default_dtype': torch.double},
    # 'Hardsigmoid' 的额外信息，不检查 gradgrad 和 jit，默认数据类型为双精度浮点数

    'Hardswish': {'check_gradgrad': False, 'check_jit': False, 'default_dtype': torch.double},
    # 'Hardswish' 的额外信息，不检查 gradgrad 和 jit，默认数据类型为双精度浮点数

    # For RRelu, test that compare CPU and GPU results fail because RNG
    # is different between CPU and GPU
    # 对于 RReLU，测试比较 CPU 和 GPU 结果失败，因为随机数生成器在 CPU 和 GPU 之间不同

    'RReLU': {'test_cuda': False, 'default_dtype': torch.double},
    # 'RReLU' 的额外信息，不测试 CUDA 加速，默认数据类型为双精度浮点数

    'ELU': {'default_dtype': torch.double},
    # 'ELU' 的额外信息，默认数据类型为双精度浮点数

    'GELU': {'default_dtype': torch.double},
    # 'GELU' 的额外信息，默认数据类型为双精度浮点数

    'GLU': {'default_dtype': torch.double},
    # 'GLU' 的额外信息，默认数据类型为双精度浮点数

    'Hardshrink': {'default_dtype': torch.double},
    # 'Hardshrink' 的额外信息，默认数据类型为双精度浮点数

    'Hardtanh': {'default_dtype': torch.double},
    # 'Hardtanh' 的额外信息，默认数据类型为双精度浮点数

    'LeakyReLU': {'default_dtype': torch.double},
    # 'LeakyReLU' 的额外信息，默认数据类型为双精度浮点数
    'LogSigmoid': {'default_dtype': torch.double},
    'Mish': {'default_dtype': torch.double},
    'PReLU': {'default_dtype': torch.double},
    'ReLU6': {'default_dtype': torch.double},
    'ReLU': {'default_dtype': torch.double},
    'SELU': {'default_dtype': torch.double},
    'SiLU': {'default_dtype': torch.double},
    'Sigmoid': {'default_dtype': torch.double},
    'Softplus': {'default_dtype': torch.double},
    'Softshrink': {'default_dtype': torch.double},
    'Softsign': {'default_dtype': torch.double},
    'Tanh': {'default_dtype': torch.double},
    'Tanhshrink': {'default_dtype': torch.double},



    'LogSigmoid': {'default_dtype': torch.double},
    # 'LogSigmoid' 激活函数的默认数据类型设定为双精度(torch.double)
    'Mish': {'default_dtype': torch.double},
    # 'Mish' 激活函数的默认数据类型设定为双精度(torch.double)
    'PReLU': {'default_dtype': torch.double},
    # 'PReLU' 激活函数的默认数据类型设定为双精度(torch.double)
    'ReLU6': {'default_dtype': torch.double},
    # 'ReLU6' 激活函数的默认数据类型设定为双精度(torch.double)
    'ReLU': {'default_dtype': torch.double},
    # 'ReLU' 激活函数的默认数据类型设定为双精度(torch.double)
    'SELU': {'default_dtype': torch.double},
    # 'SELU' 激活函数的默认数据类型设定为双精度(torch.double)
    'SiLU': {'default_dtype': torch.double},
    # 'SiLU' 激活函数的默认数据类型设定为双精度(torch.double)
    'Sigmoid': {'default_dtype': torch.double},
    # 'Sigmoid' 激活函数的默认数据类型设定为双精度(torch.double)
    'Softplus': {'default_dtype': torch.double},
    # 'Softplus' 激活函数的默认数据类型设定为双精度(torch.double)
    'Softshrink': {'default_dtype': torch.double},
    # 'Softshrink' 激活函数的默认数据类型设定为双精度(torch.double)
    'Softsign': {'default_dtype': torch.double},
    # 'Softsign' 激活函数的默认数据类型设定为双精度(torch.double)
    'Tanh': {'default_dtype': torch.double},
    # 'Tanh' 激活函数的默认数据类型设定为双精度(torch.double)
    'Tanhshrink': {'default_dtype': torch.double},
    # 'Tanhshrink' 激活函数的默认数据类型设定为双精度(torch.double)
}
# 对于非线性激活函数列表中的每个激活函数进行测试设置
for non_linear_activation in non_linear_activations_no_batch:
    # 创建激活函数测试信息的字典
    activation_test_info = dict(
        module_name=non_linear_activation,
        input_size=(4,),
        reference_fn=single_batch_reference_fn,
        desc='no_batch_dim',
        test_cpp_api_parity=False,
    )
    # 获取非线性激活函数额外信息，若存在则更新测试信息
    extra_info = non_linear_activations_extra_info.get(non_linear_activation, {})
    activation_test_info.update(extra_info)
    # 将更新后的测试信息添加到新的模块测试列表中
    new_module_tests.append(activation_test_info)


def kldivloss_reference(input, target, reduction='mean', log_target=False):
    # 根据参数计算 KL 散度损失
    if log_target:
        result = torch.exp(target) * (target - input)
    else:
        result = target * (target.log() - input)
    # 根据 reduction 参数返回相应的损失值
    if reduction == 'mean':
        return result.mean()
    elif reduction == 'sum':
        return result.sum()
    elif reduction == 'batchmean' and result.dim() != 0:
        return result.sum() / result.size(0)
    return result


def nlllossNd_reference(input, target, weight=None, ignore_index=-100,
                        reduction='mean'):
    # 断言输入张量维度至少为3
    assert input.dim() >= 3
    # 获取输入张量的大小信息
    N = input.size(0)
    C = input.size(1)
    out_size = (N,) + input.size()[2:]
    # 创建与输入张量相同类型的全零张量
    output = torch.zeros(out_size).type_as(input)

    # 如果未提供权重，则使用全1的权重张量
    if weight is None:
        weight = torch.ones(C).type_as(input)
    total_weight = 0
    # 遍历输出大小的笛卡尔积，计算 NLL 损失
    for tup in product(*[range(size) for size in out_size]):
        t_nx = target[tup]
        norm = 0. if ignore_index == t_nx else weight[t_nx].item()
        input_index = list(tup)
        input_index.insert(1, t_nx)
        output[tup] = -input[tuple(input_index)] * norm
        total_weight += norm

    # 根据 reduction 参数返回相应的损失值
    if reduction == 'mean':
        return output.sum() / total_weight
    elif reduction == 'sum':
        return output.sum()
    return output


def cross_entropy_loss_prob_target_reference(input, target, weight=None, reduction='mean',
                                             label_smoothing=0.0):
    # 断言输入张量维度至少为2
    assert input.dim() >= 2

    # 对输入张量进行 log_softmax 操作
    input = torch.log_softmax(input, 1)
    C = input.size(1)
    # 如果未提供权重，则使用全1的权重张量
    if weight is None:
        weight = torch.ones(C).type_as(input)
    weight = weight.view(1, C, *(1 for _ in input.shape[2:]))

    # 如果设置了 label_smoothing 参数，则进行标签平滑处理
    if label_smoothing > 0.0:
        assert label_smoothing <= 1.0
        target = (target * (1 - label_smoothing) + label_smoothing / C)

    # 计算交叉熵损失
    output = -(input * target * weight).sum(dim=1)
    # 根据 reduction 参数返回相应的损失值
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


def cross_entropy_loss_indices_target_reference(input, target, weight=None, ignore_index=-100,
                                                reduction='mean', label_smoothing=0.0):
    # 对输入张量进行 log_softmax 操作
    log_softmax_input = torch.log_softmax(input, 1)
    # 使用 PyTorch 提供的 nll_loss 函数计算损失
    nllloss = F.nll_loss(
        log_softmax_input,
        target,
        weight,
        ignore_index=ignore_index,
        reduction=reduction)

    # 如果设置了 label_smoothing 参数，则进行标签平滑处理
    if label_smoothing == 0.0:
        return nllloss

    # 断言 label_smoothing 参数在 (0, 1] 范围内
    assert 0.0 < label_smoothing <= 1.0

    # 再次对输入张量进行 log_softmax 操作
    input = torch.log_softmax(input, 1)
    C = input.size(1)
    # 如果给定了权重，将输入张量按照权重进行加权
    if weight is not None:
        input = input * weight.view(1, C, *(1 for _ in input.shape[2:]))

    # 计算平滑损失，即对输入张量在第一个维度上求和的负值
    smooth_loss = -torch.sum(input, 1)

    # 创建一个忽略掩码，标记目标张量中与忽略索引相等的位置
    ignore_mask = target == ignore_index
    # 将平滑损失中对应忽略掩码位置的值设置为0
    smooth_loss.masked_fill_(ignore_mask, 0.0)

    # 根据指定的减少方式计算最终的损失值
    if reduction == 'mean':
        if weight is not None:
            # 如果给定了权重，按照权重对损失进行加权平均，以保持与nll_loss_nd一致
            ret = torch.sum(smooth_loss) / weight.gather(0, target.masked_select(ignore_mask.logical_not()).flatten()).sum()
        else:
            # 否则，直接对不被忽略的位置进行平均计算
            ret = torch.mean(smooth_loss.masked_select(ignore_mask.logical_not()))
    elif reduction == 'sum':
        # 对所有损失值进行求和
        ret = torch.sum(smooth_loss)
    else:
        # 如果没有指定减少方式，则返回平滑损失张量本身
        ret = smooth_loss

    # 返回最终的损失，结合标签平滑系数和交叉熵损失nllloss
    return (1 - label_smoothing) * nllloss + ret * (label_smoothing / C)
# 根据输入和目标的形状判断使用哪种交叉熵损失函数，返回损失值
def cross_entropy_loss_reference(input, target, weight=None, ignore_index=-100, reduction='mean',
                                 label_smoothing=0.0):
    if input.shape == target.shape:
        # 如果输入和目标形状相同，则调用基于概率目标的交叉熵损失函数
        return cross_entropy_loss_prob_target_reference(
            input,
            target,
            weight=weight,
            reduction=reduction,
            label_smoothing=label_smoothing)
    else:
        # 否则调用基于索引目标的交叉熵损失函数
        return cross_entropy_loss_indices_target_reference(
            input, target, weight=weight, reduction=reduction,
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )


# 计算负对数似然损失函数，支持权重和忽略索引
def nllloss_reference(input, target, weight=None, ignore_index=-100,
                      reduction='mean'):

    # 辅助函数，计算单个目标的负对数似然损失和权重
    def nll_loss_helper(input, target, weight, ignore_index):
        if target == ignore_index:
            return (0, 0)
        norm = 1 if weight is None else weight[target]
        result = -input[target] * norm
        return (result, norm)

    # 对每个输入和目标计算负对数似然损失和权重
    losses_and_weights = [nll_loss_helper(i, t, weight, ignore_index)
                          for i, t in zip(input, target)]
    losses, weights = zip(*losses_and_weights)
    losses_tensor = input.new_tensor(losses)
    if reduction == 'mean':
        return sum(losses_tensor) / sum(weights)
    elif reduction == 'sum':
        return sum(losses_tensor)
    else:
        return losses_tensor


# 平滑 L1 损失函数，支持均值或总和的降维选项
def smoothl1loss_reference(input, target, reduction='mean', beta=1.0):
    abs_diff = (input - target).abs()
    ge_beta_mask = (abs_diff >= beta).type_as(abs_diff)
    lt_beta_mask = (abs_diff < beta).type_as(abs_diff)
    # 当 beta <= 0 时，直接使用 L1 损失
    if beta == 0:
        output = abs_diff
    else:
        output = ge_beta_mask * (abs_diff - 0.5 * beta) + lt_beta_mask * 0.5 * (abs_diff ** 2) / beta
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


# Huber 损失函数，支持均值或总和的降维选项
def huberloss_reference(input, target, reduction='mean', delta=1.0):
    abs_diff = (input - target).abs()
    ge_delta_mask = (abs_diff >= delta)
    lt_delta_mask = (abs_diff < delta)
    output = ge_delta_mask * delta * (abs_diff - 0.5 * delta) + lt_delta_mask * 0.5 * (abs_diff ** 2)
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    return output


# 多标签边界损失函数的参考实现
def _multilabelmarginloss_reference(input, target):
    targets = []
    for target_index in target:
        if target_index < 0:
            break
        targets.append(target_index)

    sum = 0
    for target_index in targets:
        for i in range(0, len(input)):
            if i not in targets:
                sum += max(0, 1 - input[target_index] + input[i])

    return sum


# 多标签边界损失函数，处理输入维度，支持均值或总和的降维选项
def multilabelmarginloss_reference(input, target, reduction='mean'):
    # 将输入维度设置为二维
    input_dim = input.dim()
    # 如果输入的张量维度小于 2，确保目标张量也维度小于 2
    if input.dim() < 2:
        assert target.dim() < 2
        # 如果输入张量是 1 维，则添加一个维度使其变为 (1, 1) 或 (1, C)
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)
        # 如果目标张量是 1 维，则添加一个维度使其变为 (1, 1) 或 (1, C)
        target = target.unsqueeze(0) if target.dim() == 1 else target.unsqueeze(0).unsqueeze(0)

    # 获取输入张量的第一维和第二维大小
    n = input.size(0)
    dim = input.size(1)
    # 创建一个与输入张量相同类型的张量，全部初始化为 0
    output = input.new(n).zero_()
    # 遍历每个样本，计算多标签边际损失
    for i in range(0, n):
        output[i] = _multilabelmarginloss_reference(input[i], target[i])

    # 根据指定的减少方式计算最终损失
    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    elif input_dim < 2:
        # 当输入维度小于 2 时，squeeze 函数将使结果张量返回正确的维度
        return output.squeeze() / dim
    else:
        # 默认情况下直接返回损失结果
        return output / dim
# 计算 Hinge Embedding Loss，根据输入和目标值计算损失，将小于边界的部分设为0并转换为与输入相同的类型
def hingeembeddingloss_reference(input, target, margin=1.0, reduction='mean'):
    # 计算小于边界的部分，并转换为与输入相同的数据类型
    margin_clamp = (margin - input).clamp(min=0).type_as(input)
    # 根据目标值选择输入值或者小于边界的部分
    output = torch.where(target == 1, input, margin_clamp)

    if reduction == 'mean':
        return output.mean()  # 返回平均值
    elif reduction == 'sum':
        return output.sum()   # 返回总和
    return output             # 返回未减少的输出


# 计算 Soft Margin Loss，根据输入和目标值计算损失
def softmarginloss_reference(input, target, reduction='mean'):
    # 计算 Soft Margin Loss
    output = (1 + (-input * target).exp()).log()

    if reduction == 'mean':
        return output.mean()  # 返回平均值
    elif reduction == 'sum':
        return output.sum()   # 返回总和
    return output             # 返回未减少的输出


# 计算 Multi Margin Loss，根据输入、目标索引和参数计算损失
def _multimarginloss_reference(input, target_idx, p, margin, weight):
    if weight is None:
        weight = input.new(len(input)).fill_(1)

    output = 0
    for i in range(0, len(input)):
        if i != target_idx:
            output += weight[target_idx] * (max(0, (margin - input[target_idx] + input[i])) ** p)
    return output


# 计算 Multi Margin Loss，根据输入、目标值和参数计算损失
def multimarginloss_reference(input, target, p=1, margin=1, weight=None, reduction='mean'):
    if input.dim() < 2:
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)

    target_dim = target.dim()
    if target.dim() == 0:
        target = target.unsqueeze(0)

    n = input.size(0)
    dim = input.size(1)
    output = input.new(n)
    for x in range(0, n):
        output[x] = _multimarginloss_reference(input[x], target[x], p, margin, weight)

    if reduction == 'mean':
        return output.mean() / dim  # 返回平均值除以维度
    elif reduction == 'sum':
        return output.sum() / dim   # 返回总和除以维度
    elif target_dim == 0:
        return output.squeeze(0) / dim  # 如果目标维度为0，返回挤压后的输出除以维度
    return output / dim                 # 返回输出除以维度


# 计算 Cosine Embedding Loss，根据两个输入和目标值计算损失
def cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    # 计算余弦相似度函数
    def _cos(a, b):
        cos = a.new(a.size(0))
        for i in range(0, a.size(0)):
            cos[i] = (a[i] * b[i]).sum() / ((((a[i] * a[i]).sum() + 1e-12) * ((b[i] * b[i]).sum() + 1e-12)) ** 0.5)
        return cos

    # 根据目标值选择计算 1-cosine 相似度或者带有边界的 cosine 相似度
    output = torch.where(target == 1, 1 - _cos(input1, input2), (_cos(input1, input2) - margin).clamp(min=0))

    if reduction == 'mean':
        return output.mean()  # 返回平均值
    elif reduction == 'sum':
        return output.sum()   # 返回总和
    return output             # 返回未减少的输出


# 计算 Triplet Margin Loss，根据锚点、正例、负例和参数计算损失
def tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False,
                                reduction='mean'):
    # 计算锚点和正例之间的距离以及锚点和负例之间的距离
    d_p = torch.pairwise_distance(anchor, positive, p, eps)
    d_n = torch.pairwise_distance(anchor, negative, p, eps)
    if swap:
        # 如果指定了交换，则计算正例和负例之间的距离，并取两者之间较小的距离
        d_s = torch.pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    # 计算 Triplet Margin Loss
    output = torch.clamp(margin + d_p - d_n, min=0.0)

    if reduction == 'mean':
        return output.mean()  # 返回平均值
    elif reduction == 'sum':
        return output.sum()   # 返回总和
    return output             # 返回未减少的输出


# 计算 Margin Ranking Loss，根据两个输入和目标值计算损失
def marginrankingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    # 计算 Margin Ranking Loss
    output = (-target * (input1 - input2) + margin).clamp(min=0)

    if reduction == 'mean':
        return output.mean()  # 返回平均值
    # 如果 reduction 参数为 'sum'，则返回输出张量的所有元素的和
    elif reduction == 'sum':
        return output.sum()
    # 如果 reduction 参数不为 'sum'，直接返回输出张量
    return output
# 根据 Graves 等人的论文编写的 CTC 损失函数，与生产实现不同，不使用对数空间
def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    # 将输入长度和目标长度转换为 PyTorch 的张量类型
    input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.as_tensor(target_lengths, dtype=torch.long)
    # 记录输入 log_probs 的数据类型
    dt = log_probs.dtype
    # 将 log_probs 转换为 double 类型以提高精度（因为不在对数空间）
    log_probs = log_probs.double()  # 我们需要准确度，因为不在对数空间
    # 将 targets 转换为 long 类型
    targets = targets.long()
    # 计算累积目标长度
    cum_target_lengths = target_lengths.cumsum(0)
    # 初始化损失列表
    losses = []
    # 对每个时间步进行损失计算
    for i in range(log_probs.size(1)):
        # 获取当前输入序列的长度和目标序列的长度
        input_length = input_lengths[i].item()
        target_length = target_lengths[i].item()
        cum_target_length = cum_target_lengths[i].item()
        # 根据是否为二维目标数据选择不同的处理方式
        targets_prime = targets.new_full((2 * target_length + 1,), blank)
        if targets.dim() == 2:
            targets_prime[1::2] = targets[i, :target_length]
        else:
            targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
        # 计算当前时间步的概率
        probs = log_probs[:input_length, i].exp()
        # 初始化 alpha 向量
        alpha = log_probs.new_zeros((target_length * 2 + 1,))
        alpha[0] = probs[0, blank]
        alpha[1] = probs[0, targets_prime[1]]
        mask_third = (targets_prime[:-2] != targets_prime[2:])
        # 进行前向传播计算 alpha
        for t in range(1, input_length):
            alpha_next = alpha.clone()
            alpha_next[1:] += alpha[:-1]
            alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
            alpha = probs[t, targets_prime] * alpha_next
        # 将当前时间步计算得到的损失加入损失列表
        losses.append(-alpha[-2:].sum().log()[None])
    # 将所有时间步的损失拼接成一个张量
    output = torch.cat(losses, 0)
    # 根据 reduction 参数计算最终的输出损失
    if reduction == 'mean':
        output = (output / target_lengths.to(dtype=output.dtype, device=output.device)).mean()
    elif reduction == 'sum':
        output = output.sum()
    # 将输出损失转换回原始数据类型并返回
    output = output.to(dt)
    return output


# 定义损失函数的参考字典，包含各种损失函数的名称及其对应的参考实现函数
loss_reference_fns: Dict['str', Callable] = {
    'KLDivLoss': kldivloss_reference,
    'KLDivLoss_log_target': partial(kldivloss_reference, log_target=True),
    'NLLLoss': nllloss_reference,
    'NLLLossNd': nlllossNd_reference,
    'SmoothL1Loss': smoothl1loss_reference,
    'HuberLoss': huberloss_reference,
    'MultiLabelMarginLoss': multilabelmarginloss_reference,
    'HingeEmbeddingLoss': hingeembeddingloss_reference,
    'SoftMarginLoss': softmarginloss_reference,
    'MultiMarginLoss': multimarginloss_reference,
    'CosineEmbeddingLoss': cosineembeddingloss_reference,
    'TripletMarginLoss': tripletmarginloss_reference,
    'MarginRankingLoss': marginrankingloss_reference,
    'CTCLoss': ctcloss_reference,  # 将上面定义的 CTC 损失函数添加到字典中
    'CrossEntropyLoss': cross_entropy_loss_reference
}


# 定义用于测试的准则函数列表
criterion_tests = []


def single_batch_reference_criterion_fn(*args):
    """Reference function for criterion supporting no batch dimensions.

    The criterion is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    criterion = args[-1]
    # 定义一个函数，用于将输入张量或列表/元组中的每个张量在第0维度上增加一个维度
    def unsqueeze_inp(inp):
        # 如果输入是列表或元组，则对每个元素调用unsqueeze(0)，即在第0维度上增加一个维度
        if isinstance(inp, (list, tuple)):
            return [t.unsqueeze(0) for t in inp]
        # 如果输入是单个张量，则在第0维度上增加一个维度
        return inp.unsqueeze(0)

    # 定义一个函数，用于将嵌套的列表或元组展平为一个单层列表
    def flatten(xs):
        result = []
        # 如果输入是列表或元组，则递归地将每个元素添加到结果列表中
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(flatten(x))
        else:
            # 如果输入是单个元素，则直接将其添加到结果列表中
            result.append(xs)
        return result

    # 对参数 args 中除最后一个参数外的所有参数执行 unsqueeze_inp，并将结果列表展平为单层列表
    single_batch_input_args = flatten([unsqueeze_inp(input) for input in args[:-1]])

    # 使用 criterion 函数计算输出，传入单批次输入参数列表
    output = criterion(*single_batch_input_args)
    
    # 获取 criterion 的减少（reduction）方式（例如：'none'、'sum' 或 'mean'）
    reduction = get_reduction(criterion)

    # 如果减少方式是 'none'，则从输出中去除第0维度（假设输出是批次大小为1的情况）
    if reduction == 'none':
        return output.squeeze(0)
    
    # 如果减少方式是 'sum' 或 'mean'，则输出结果应该是一个标量，直接返回输出
    # 因为在 'sum' 或 'mean' 的情况下，输出应该已经是一个标量（单个值）
    return output
# 定义不带批处理维度的回归损失函数名称列表
regression_criterion_no_batch = [
    'L1Loss', 'MSELoss', 'PoissonNLLLoss', 'HuberLoss', 'SmoothL1Loss'
]
# 定义不同的减少(reduction)方式列表
reductions = ['none', 'mean', 'sum']
# 遍历回归损失函数和减少方式的笛卡尔积，生成测试信息并添加到测试列表
for name, reduction in product(regression_criterion_no_batch, reductions):
    # 构造单个回归测试信息的字典
    regression_test_info = dict(
        fullname=f"{name}_no_batch_dim_{reduction}",  # 定义测试名称
        constructor=lambda *args, name=name: getattr(nn, name)(reduction=reduction),  # 匿名函数，构造损失函数对象
        input_size=(3, ),  # 输入大小为3的元组
        target_size=(3, ),  # 目标大小为3的元组
        reference_fn=single_batch_reference_criterion_fn,  # 参考函数
        test_cpp_api_parity=False,  # 不测试C++ API的一致性
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    )
    criterion_tests.append(regression_test_info)  # 将测试信息添加到测试列表中


# 遍历减少方式列表，生成KLDivLoss的测试信息并添加到测试列表
for reduction in reductions:
    regression_test_info = dict(
        fullname=f"KLDivLoss_no_batch_dim_{reduction}",  # 定义测试名称
        constructor=lambda: nn.KLDivLoss(reduction=reduction),  # 匿名函数，构造KLDivLoss对象
        input_fn=lambda: torch.rand((3,)).log(),  # 输入函数，生成大小为3的随机张量的对数
        target_fn=lambda: torch.rand((3,)),  # 目标函数，生成大小为3的随机张量
        reference_fn=single_batch_reference_criterion_fn,  # 参考函数
        test_cpp_api_parity=False,  # 不测试C++ API的一致性
        default_dtype=torch.double,  # 默认数据类型为双精度浮点数
    )
    criterion_tests.append(regression_test_info)  # 将测试信息添加到测试列表中


# 检查不带批处理维度的分类损失函数是否正常工作
# 列表中包含各种分类损失函数的名称、输入函数和目标函数
classification_criterion_no_batch = [
    (
        'BCELoss',
        lambda: torch.sigmoid(torch.randn(9, dtype=torch.double)),  # 输入函数，生成大小为9的随机张量的sigmoid值
        lambda: torch.randn(9, dtype=torch.double).gt(0).to(torch.double)  # 目标函数，生成大小为9的随机张量并转换为双精度张量
    ),
    ('BCEWithLogitsLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.randn(9, dtype=torch.double)),  # BCEWithLogitsLoss的输入函数和目标函数
    ('HingeEmbeddingLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.tensor([-1, 1, 1] * 3)),  # HingeEmbeddingLoss的输入函数和目标函数
    ('MultiLabelMarginLoss', lambda: torch.randn(4, dtype=torch.double), lambda: torch.tensor([3, 0, -1, 1])),  # MultiLabelMarginLoss的输入函数和目标函数
    ('SoftMarginLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.tensor([-1, 1, 1] * 3)),  # SoftMarginLoss的输入函数和目标函数
    ('NLLLoss', lambda: F.log_softmax(torch.randn(3, dtype=torch.double), dim=0), lambda: torch.tensor(1)),  # NLLLoss的输入函数和目标函数
    (
        'CosineEmbeddingLoss',
        lambda: (torch.randn(9, dtype=torch.double), torch.randn(9, dtype=torch.double)),  # 输入函数，生成大小为9的两个随机张量
        lambda: torch.tensor(1, dtype=torch.double)  # 目标函数，生成双精度张量1
    ),
    ('MarginRankingLoss', lambda: (torch.randn(()), torch.randn(())), lambda: torch.randn(()).sign()),  # MarginRankingLoss的输入函数和目标函数
    (
        'TripletMarginLoss',
        lambda: (torch.randn(9, dtype=torch.double), torch.randn(9, dtype=torch.double)),  # 输入函数，生成大小为9的两个随机张量
        lambda: torch.randn(9, dtype=torch.double)  # 目标函数，生成大小为9的随机张量
    ),
    ('MultiLabelSoftMarginLoss', lambda: torch.randn(9, dtype=torch.double), lambda: torch.randn(9)),  # MultiLabelSoftMarginLoss的输入函数和目标函数
]

# classification_criterion_no_batch_extra_info字典，指定了某些分类损失函数的额外信息
classification_criterion_no_batch_extra_info: Dict[str, dict] = {
    'MultiLabelMarginLoss': {'check_gradgrad': False},  # 对于MultiLabelMarginLoss，禁用gradgrad检查
}

# classification_cpp_parity字典，指定了需要修复的C++ API一致性问题
# TODO : Fix these discrepancies
classification_cpp_parity = {
    'BCELoss': False,  # BCELoss存在一致性问题，需要修复
}
    'BCEWithLogitsLoss': False,  # BCEWithLogitsLoss 损失函数是否启用的标志，此处设为 False 表示未启用
    'HingeEmbeddingLoss': False,  # HingeEmbeddingLoss 损失函数是否启用的标志，此处设为 False 表示未启用
    'NLLLoss': False,  # NLLLoss 损失函数是否启用的标志，此处设为 False 表示未启用
    'SoftMarginLoss': False,  # SoftMarginLoss 损失函数是否启用的标志，此处设为 False 表示未启用
}
# 定义一个包含三元组和减少方式的迭代器
reductions = ['none', 'mean', 'sum']
for (name, input_fn, target_fn), reduction in product(classification_criterion_no_batch,
                                                      reductions):
    # 构造分类测试信息字典
    classification_test_info = dict(
        # 设置测试信息的完整名称
        fullname=f"{name}_no_batch_dim_{reduction}",
        # 使用 lambda 函数延迟构造分类器对象
        constructor=lambda *args, name=name: getattr(nn, name)(reduction=reduction),
        # 使用 lambda 函数延迟获取输入数据的函数
        input_fn=lambda f=input_fn: f(),
        # 使用 lambda 函数延迟获取目标数据的函数
        target_fn=lambda f=target_fn: f(),
        # 设置参考函数为单批量参考标准函数
        reference_fn=single_batch_reference_criterion_fn,
        # 启用 C++ API 的功能检查
        test_cpp_api_parity=True,
        # 检查分类器是否具有 C++ API 的功能对应性
        has_parity=classification_cpp_parity.get(name, True)
    )
    # 获取额外的分类准则信息
    extra_info = classification_criterion_no_batch_extra_info.get(name, {})
    # 更新分类测试信息字典
    classification_test_info.update(extra_info)
    # 将分类测试信息字典添加到测试准则列表中
    criterion_tests.append(classification_test_info)


class NNTestCase(TestCase):

    # _forward 在继承 NNTestCase 的类中定义
    @abstractmethod
    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    # _get_parameters 在继承 NNTestCase 的类中定义
    @abstractmethod
    def _get_parameters(self, module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        raise NotImplementedError

    # _zero_grad_parameters 在继承 NNTestCase 的类中定义
    @abstractmethod
    def _zero_grad_parameters(self, module: nn.Module) -> None:
        raise NotImplementedError

    # _backward 在继承 NNTestCase 的类中定义
    @abstractmethod
    def _backward(self, module: nn.Module,
                  input: _TensorOrTensors, output: torch.Tensor,
                  grad_output: Union[torch.Tensor, Sequence[torch.Tensor]],
                  create_graph: bool = False):
        raise NotImplementedError

    # 计算输入的雅可比矩阵
    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            # 如果输入是元组，则逐个计算元素的雅可比矩阵
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            # 如果输入是列表，则逐个计算元素的雅可比矩阵
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            # 如果输入是张量，则返回一个全零的张量作为雅可比矩阵
            return torch.zeros(input.nelement(), num_out)

    # 展平输入的张量
    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                # 如果输入是稀疏张量，则转为稠密张量并展平
                return x.to_dense().view(-1)
            else:
                # 否则直接展平张量
                return x.view(-1)
        else:
            # 如果输入不是张量，则逐层展平
            return tuple(self._flatten_tensors(a) for a in x)

    # 将输入的梯度归零
    def _zero_grad_input(self, input):
        if isinstance(input, torch.Tensor):
            if input.requires_grad and input.grad is not None:
                # 如果输入是张量且需要梯度计算，则将梯度归零并分离
                input.grad.zero_()
                input.grad.detach_()
        else:
            # 如果输入不是张量，则递归地将每个元素的梯度归零
            for i in input:
                self._zero_grad_input(i)
    # 计算给定模块的解析雅可比矩阵，对于输入和参数均有选择性地计算
    def _analytical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        # 调用模型的前向传播方法，获取输出结果
        output = self._forward(module, input)
        # 计算输出张量的元素个数
        output_size = output.nelement()

        # 如果需要计算输入的雅可比矩阵
        if jacobian_input:
            # 计算输入向量关于输出的雅可比矩阵
            jacobian_inp = self._jacobian(input, output_size)
            # 将雅可比矩阵扁平化为张量列表
            flat_jacobian_input = list(_iter_tensors(jacobian_inp))

        # 如果需要计算参数的雅可比矩阵
        if jacobian_parameters:
            # 获取模块的参数，并计算所有参数的总元素个数
            num_param = sum(p.numel() for p in self._get_parameters(module)[0])
            # 创建一个全零的参数雅可比矩阵
            jacobian_param = torch.zeros(num_param, output_size)

        # 遍历输出张量的每个元素
        for i in range(output_size):
            # 获取模块的参数及其梯度
            param, d_param = self._get_parameters(module)
            # 将没有梯度的参数设置为全零张量
            d_param = [torch.zeros_like(p) if d is None else d for (p, d) in zip(param, d_param)]

            # 创建一个与输出张量相同形状的全零张量，并将第 i 个位置设为 1
            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1

            # 如果需要计算参数的雅可比矩阵，重置模块参数的梯度
            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # 如果需要计算输入的雅可比矩阵，重置输入的梯度
            if jacobian_input:
                self._zero_grad_input(input)
            # 计算反向传播，获取输入的梯度
            d_input = self._backward(module, input, output, d_out)

            # 如果需要计算输入的雅可比矩阵，将输入的梯度更新到扁平化的雅可比矩阵中
            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, _iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x.contiguous().view(-1)
            # 如果需要计算参数的雅可比矩阵，将参数的梯度更新到参数雅可比矩阵中
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)

        # 返回结果，包括输入和参数的雅可比矩阵
        res: Tuple[torch.Tensor, ...] = tuple()
        if jacobian_input:
            res += jacobian_inp,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    # 计算给定模块的数值雅可比矩阵，对于输入和参数均有选择性地计算
    def _numerical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        # 定义一个函数，该函数返回模块的前向传播结果并且与计算图分离
        def fw(*input):
            return self._forward(module, input).detach()

        # 初始化结果元组
        res: Tuple[torch.Tensor, ...] = tuple()
        # 如果需要计算输入的数值雅可比矩阵
        if jacobian_input:
            # 调用函数获取输入的数值雅可比矩阵，并将结果加入到结果元组中
            res += _get_numerical_jacobian(fw, input, eps=1e-6),

        # 如果需要计算参数的数值雅可比矩阵
        if jacobian_parameters:
            # 获取模块的参数
            param, _ = self._get_parameters(module)
            to_cat = []
            # 遍历每个参数
            for p in param:
                # 调用函数获取参数的数值雅可比矩阵，并将结果加入到列表中
                jacobian = _get_numerical_jacobian(fw, input, target=p, eps=1e-6)
                # get_numerical_jacobian 返回的是一个列表，但我们需要一个张量
                to_cat.append(jacobian[0][0])
            # 将所有结果连接成一个张量，并将其加入到结果元组中
            res += (torch.cat(to_cat, 0),)

        return res
    # 检查给定模块的雅可比矩阵是否正确
    def check_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True):
        # 检查模块是否具有可学习参数
        jacobian_parameters = bool(self._get_parameters(module)[0])
        # 计算解析法求得的雅可比矩阵
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        # 计算数值法求得的雅可比矩阵
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        # 将解析法和数值法求得的矩阵展平为列表
        analytical_t = list(_iter_tensors(analytical))
        numerical_t = list(_iter_tensors(numerical))

        # 存储解析法和数值法求得矩阵中每对对应元素之间的差异
        differences = []
        for a, n in zip(analytical_t, numerical_t):
            if a.numel() != 0:
                # 计算对应元素之间的差异的绝对值的最大值
                differences.append(a.add(n, alpha=-1).abs().max())
            # TODO: 比较结构（确保解析法得到的雅可比矩阵具有正确的形状）
        # 如果存在差异，则断言最大差异不超过预设精度值
        if len(differences) > 0:
            self.assertLessEqual(max(differences), PRECISION)  # type: ignore[type-var]
# 定义一个基础测试类 TestBase
class TestBase:

    # 需要的参数名称集合，作为类变量
    _required_arg_names = {'constructor_args', 'input', 'extra_args'}

    # 初始化方法，接受构造函数、描述、参考函数、全名和其他关键字参数
    def __init__(self, constructor, desc='', reference_fn=None, fullname=None, **kwargs):
        self.desc = desc  # 设置描述信息
        self.fullname = fullname  # 设置全名
        self.constructor = constructor  # 设置构造函数
        self.reference_fn = reference_fn  # 设置参考函数
        # 遍历必需的参数名称集合
        for name in self._required_arg_names:
            # 检查是否缺少某个必需的参数，如果是则根据情况进行处理
            if name not in kwargs and name + '_fn' not in kwargs and name + '_size' not in kwargs:
                if name in {'constructor_args', 'extra_args'}:
                    kwargs[name] = tuple()  # 如果是构造参数或额外参数，则默认为空元组
                else:
                    raise ValueError(f"{self.get_name()}: Specify {name} by a value, a function to generate it, or it's size!")
        self._extra_kwargs = kwargs  # 将剩余的关键字参数存储到 _extra_kwargs 中
        self._arg_cache = {}  # 初始化参数缓存字典

    # 获取测试名称的方法
    def get_name(self):
        if self.fullname is not None:
            return 'test_' + self.fullname  # 如果有全名，则返回以 test_ 开头的全名
        test_name = 'test_' + self.constructor.__name__  # 否则返回以 test_ 开头的构造函数名称
        if self.desc:
            test_name += '_' + self.desc  # 如果有描述信息，则加上下划线和描述信息
        return test_name  # 返回组装好的测试名称

    # 解包数据的方法，处理成 torch.Tensor 类型或迭代类型
    def _unpack(self, value):
        if isinstance(value, torch.Tensor):
            return value  # 如果是 Tensor 类型直接返回
        elif is_iterable(value):
            return type(value)(self._unpack(v) for v in value)  # 如果是可迭代类型，则递归解包其中的元素
        else:
            return value  # 否则直接返回原值

    # 获取构造参数的属性方法
    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', True)  # 调用 _get_arg 方法获取构造参数，需要解包

    # 获取额外参数的属性方法
    @property
    def extra_args(self):
        return self._get_arg('extra_args', True)  # 调用 _get_arg 方法获取额外参数，需要解包

    # 获取参数的内部方法，根据名称获取参数值并进行缓存
    def _get_arg(self, name, unpack):
        assert name in self._required_arg_names  # 断言参数名称在必需参数集合中

        # 如果参数名称不在缓存中，则根据情况获取参数值并进行缓存
        if name not in self._arg_cache:
            fn_name = name + '_fn'  # 参数对应的函数名称
            size_name = name + '_size'  # 参数对应的尺寸名称

            # 根据不同情况获取参数值并存入缓存
            if name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[name]
            elif fn_name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[fn_name]()
            else:
                assert size_name in self._extra_kwargs, \
                    f"Missing `{name}`, `{size_name}` or `{fn_name}` for {self.get_name()}"
                
                # 定义一个函数，根据尺寸生成张量数据
                def map_tensor_sizes(sizes):
                    if isinstance(sizes, list):
                        return [map_tensor_sizes(s) for s in sizes]
                    elif isinstance(sizes, torch.Tensor):
                        return sizes.double()
                    else:
                        return torch.randn(sizes)

                self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])

        # 如果需要解包，则调用 _unpack 方法解包参数值，否则直接返回参数值
        return self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name]

    # 获取输入数据的方法，支持解包操作
    def _get_input(self, unpack=True):
        return self._get_arg('input', unpack)  # 调用 _get_arg 方法获取输入数据，支持解包操作

    # 调用实例时的方法，抛出未实现错误
    def __call__(self, test_case):
        raise NotImplementedError


# 定义一个模块测试类 ModuleTest，继承自 TestBase
class ModuleTest(TestBase):

    # 抽象方法，需要子类实现具体测试逻辑
    @abstractmethod
    def _do_test(self, test_case: Any, module: nn.Module, input: Any) -> Any:
        raise NotImplementedError  # 抛出未实现错误
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 初始化属性 jacobian_input，默认为 True
        self.jacobian_input = kwargs.get('jacobian_input', True)
        # 初始化属性 should_test_cuda，默认为 True
        self.should_test_cuda = kwargs.get('test_cuda', True)
        # 初始化属性 should_test_pickle，默认为 True
        self.should_test_pickle = kwargs.get('pickle', True)
        # 初始化属性 check_gradgrad，默认为 True
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        # 初始化属性 FIXME_no_cuda_gradgrad_comparison，默认为 False
        self.FIXME_no_cuda_gradgrad_comparison = \
            kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        # 初始化属性 precision，默认为 2e-4
        self.precision = kwargs.get('precision', 2e-4)
        # 初始化属性 check_forward_only，默认为 False
        self.check_forward_only = kwargs.get('check_forward_only', False)
        # 初始化属性 default_dtype，默认为当前的默认数据类型，如果未指定则使用 torch 的默认数据类型
        self.default_dtype = kwargs.get('default_dtype', None)
        # 如果未指定 default_dtype，则设为当前 torch 的默认数据类型
        if self.default_dtype is None:
            self.default_dtype = torch.get_default_dtype()
    
    # 调用实例对象时执行的方法，接受一个 test_case 参数
    def __call__(self, test_case):
        # 设置默认的数据类型为 self.default_dtype 的上下文管理器
        with set_default_dtype(self.default_dtype):
            # 使用 self.constructor 和 self.constructor_args 创建一个模块实例
            module = self.constructor(*self.constructor_args)
            # 获取输入数据
            input = self._get_input()
    
            # 如果存在 reference_fn 属性，则进行参考输出的测试
            if self.reference_fn is not None:
                # 对模块和输入数据进行前向传播，获取输出
                out = test_case._forward(module, input)
                # 深拷贝输入数据和模块
                ref_input = deepcopy(input)
                ref_module = deepcopy(module)
                # 使用 reference_fn 对深拷贝的输入数据、模块和模块的参数进行预期输出的计算
                expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0], ref_module)
                # 断言实际输出与预期输出相等
                test_case.assertEqual(out, expected_out, exact_dtype=False)
            
            # 如果 check_forward_only 属性为 True，则直接返回
            if self.check_forward_only:
                return
            
            # 对非连续数据的测试
            self.test_noncontig(test_case, module, input)
    
            # 如果 should_test_pickle 属性为 True，则进行对象序列化和反序列化的测试
            if self.should_test_pickle:
                # 使用临时文件 f 进行对象的序列化和反序列化测试
                with tempfile.TemporaryFile() as f:
                    # 对模块和输入数据进行前向传播
                    test_case._forward(module, input)
                    # 将模块保存到临时文件 f 中
                    torch.save(module, f)
                    # 将文件指针移到文件开头
                    f.seek(0)
                    # 从文件中加载模块的副本
                    module_copy = torch.load(f)
                    # 断言原模块和加载的副本在相同输入下的输出相等
                    test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))
            
            # 执行测试方法 _do_test
            self._do_test(test_case, module, input)
    
    # 使给定对象的数据变为非连续的方法
    def noncontiguize(self, obj):
        # 如果对象是列表，则递归处理列表中的每个元素
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        # 如果对象是元组，则递归处理元组中的每个元素
        elif isinstance(obj, tuple):
            return tuple(self.noncontiguize(o) for o in obj)
        # 如果对象是张量，则将张量的某个维度设置为非连续的
        tensor = obj
        ndim = tensor.dim()
        # 找到一个维度使得张量在该维度上的大小大于 1，然后将其设置为非连续的
        dim = ndim
        for d in range(ndim):
            if tensor.size(d) > 1:
                dim = d + 1
                break
        noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
        # 断言非连续的张量元素个数为 1 或者为 0，或者张量本身就不是连续的
        assert noncontig.numel() == 1 or noncontig.numel() == 0 or not noncontig.is_contiguous()
        # 设置非连续张量的梯度需求与原张量相同
        noncontig.requires_grad = tensor.requires_grad
        return noncontig
    # 定义测试方法，用于测试非连续数据的情况
    def test_noncontig(self, test_case, module, input):
        # 检查输入是否为 torch.Tensor 类型且维度为 0
        # 如果是，则表示无法创建非连续数据，直接返回
        if isinstance(input, torch.Tensor) and input.dim() == 0:
            return
        # 检查是否存在维度为 0 的张量，如果有，则表示无法创建非连续数据，直接返回
        if any(i.dim() == 0 for i in input if isinstance(i, torch.Tensor)):
            return

        # 将 module 的参数梯度置零
        test_case._zero_grad_parameters(module)
        # 将输入数据的梯度置零
        test_case._zero_grad_input(input)
        # 冻结随机数生成器状态
        with freeze_rng_state():
            # 执行 module 的前向传播，得到输出
            output = test_case._forward(module, input)
            # 如果 module 具有 return_indices 属性，则取输出的第一个元素
            if getattr(module, "return_indices", False):
                output = output[0]
            # 生成一个与输出形状相同的正态分布张量作为梯度输出的深拷贝
            grad_output = output.new(output.shape).normal_()
            # 对输出进行克隆
            output = output.clone()
            # 对 module 进行反向传播，得到输入数据的梯度的深拷贝
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            # 得到 module 的参数的深拷贝
            d_param = deepcopy(test_case._get_parameters(module)[1])

        # 将输入数据非连续化
        nc_input = self.noncontiguize(input)
        # 将梯度输出数据非连续化
        nc_grad_output = self.noncontiguize(grad_output)
        # 对连续和非连续的输入数据和梯度输出数据进行组合遍历
        for contig_i, contig_g in product((True, False), repeat=2):
            # 根据 contig_i 确定使用连续还是非连续的输入数据
            i = input if contig_i else nc_input
            # 对梯度输出进行深拷贝，如果 contig_g 为 False，则使用非连续的梯度输出
            # 一些操作（例如 nn.Flatten）可能返回与梯度输出共享存储的梯度，因此这里需要进行拷贝操作
            go = deepcopy(grad_output if contig_g else nc_grad_output)
            # 将 module 的参数梯度置零
            test_case._zero_grad_parameters(module)
            # 将输入数据的梯度置零
            test_case._zero_grad_input(i)
            # 冻结随机数生成器状态
            with freeze_rng_state():
                # 执行 module 的前向传播，得到输出
                out = test_case._forward(module, i)
                # 如果 module 具有 return_indices 属性，则取输出的第一个元素
                if getattr(module, "return_indices", False):
                    out = out[0]
                # 执行 module 的反向传播，得到梯度
                grad = test_case._backward(module, i, out, go)

                # 断言输出是否与预期的 output 相等
                test_case.assertEqual(out, output)
                # 断言梯度是否与预期的 d_input 相等，允许的绝对误差为 1e-4，相对误差为 0
                test_case.assertEqual(grad, d_input, atol=1e-4, rtol=0)
                # 断言 module 的参数是否与预期的 d_param 相等
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)
class InputVariableMixin:
    # 定义一个 mixin 类，用于处理输入变量的准备工作
    def _get_input(self):
        # 调用父类 TestBase 的 _get_input 方法，获取输入数据
        input = TestBase._get_input(self, False)  # type: ignore[arg-type]

        # 定义一个递归函数 map_variables，用于处理输入数据中的 torch.Tensor 对象
        def map_variables(i):
            # 如果是 torch.Tensor 类型
            if isinstance(i, torch.Tensor):
                # 如果是浮点数或复数类型的 Tensor，则设置 requires_grad 为 True
                if i.is_floating_point() or i.is_complex():
                    i.requires_grad = True
                return i
            else:
                # 如果不是 Tensor 类型，递归处理其元素
                return type(i)(map_variables(elem) for elem in i)

        # 返回处理后的输入数据
        return map_variables(input)


class NewModuleTest(InputVariableMixin, ModuleTest):  # type: ignore[misc]
    # 定义一个新的测试模块 NewModuleTest，继承 InputVariableMixin 和 ModuleTest 类
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，并接受额外的参数
        super().__init__(*args, **kwargs)
        
        # 根据参数设置类的属性值，默认值为 False 或指定值
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.skip_double = kwargs.get('skip_double', False)
        self.skip_half = kwargs.get('skip_half', False)
        self.with_tf32 = kwargs.get('with_tf32', False)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.has_sparse_gradients = kwargs.get('has_sparse_gradients', False)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        self.gradcheck_fast_mode = kwargs.get('gradcheck_fast_mode', None)
        self.supports_forward_ad = kwargs.get('supports_forward_ad', False)
        self.supports_fwgrad_bwgrad = kwargs.get('supports_fwgrad_bwgrad', False)
    # 定义一个方法来检查梯度
    def _check_gradients(self, test_case, module, input_tuple):
        # 从 module 中获取所有参数并组成元组
        params = tuple(x for x in module.parameters())
        # 获取输入的元组的长度，即输入的参数个数
        num_inputs = len(input_tuple)

        # 定义一个用于梯度检查的函数 fn_to_gradcheck
        def fn_to_gradcheck(*inputs_and_params, **kwargs):
            # 断言 kwargs 为空
            assert not kwargs
            # 调用 test_case 的 _forward 方法进行前向传播
            return test_case._forward(module, inputs_and_params[:num_inputs])

        # 如果支持稀疏梯度
        if self.has_sparse_gradients:
            # 断言只有一个输入参数
            assert num_inputs == 1
            # 检查输入是否为浮点型，用于测试输入的雅可比矩阵
            test_input_jacobian = torch.is_floating_point(input_tuple[0])
            # 调用 test_case 的 check_jacobian 方法检查雅可比矩阵
            test_case.check_jacobian(module, input_tuple[0], test_input_jacobian)
        else:
            # 使用 gradcheck 函数检查梯度
            test_case.assertTrue(gradcheck(fn_to_gradcheck, input_tuple + params,
                                           check_batched_grad=self.check_batched_grad,
                                           fast_mode=self.gradcheck_fast_mode,
                                           check_forward_ad=self.supports_forward_ad))

        # 如果需要检查 gradgrad（二阶梯度）
        if self.check_gradgrad:
            # 使用 gradgradcheck 函数检查二阶梯度
            test_case.assertTrue(gradgradcheck(fn_to_gradcheck, input_tuple + params,
                                               check_batched_grad=self.check_batched_grad,
                                               fast_mode=self.gradcheck_fast_mode,
                                               check_fwd_over_rev=self.supports_fwgrad_bwgrad))

    # 获取目标属性的值
    def _get_target(self):
        return self._get_arg('target', False)

    # 获取构造函数的参数
    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)
class CriterionTest(InputVariableMixin, TestBase):  # type: ignore[misc]
    # 类CriterionTest继承自InputVariableMixin和TestBase类，并忽略类型检查警告

    # TODO: check that criterions don't ignore grad_output
    # 待办事项：检查评判标准是否未忽略梯度输出

    _required_arg_names = TestBase._required_arg_names.union({'target'})
    # 设置_required_arg_names为TestBase._required_arg_names和{'target'}的并集

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 调用父类的构造方法，初始化继承的属性和方法

        self.should_test_cuda = kwargs.get('test_cuda', True)
        # 设置是否测试CUDA，如果kwargs中包含'test_cuda'，则使用其值，否则默认为True
        self.check_forward_only = kwargs.get('check_forward_only', False)
        # 设置是否仅检查前向传播，如果kwargs中包含'check_forward_only'，则使用其值，否则默认为False
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        # 设置是否检查梯度梯度，如果kwargs中包含'check_gradgrad'，则使用其值，否则默认为True
        self.check_half = kwargs.get('check_half', True)
        # 设置是否检查半精度，如果kwargs中包含'check_half'，则使用其值，否则默认为True
        self.check_bfloat16 = kwargs.get('check_bfloat16', False)
        # 设置是否检查Bfloat16精度，如果kwargs中包含'check_bfloat16'，则使用其值，否则默认为False
        self.check_complex = kwargs.get('check_complex', False)
        # 设置是否检查复杂类型，如果kwargs中包含'check_complex'，则使用其值，否则默认为False
        self.test_cpu = kwargs.get('test_cpu', True)
        # 设置是否测试CPU，如果kwargs中包含'test_cpu'，则使用其值，否则默认为True
        self.with_tf32 = kwargs.get('with_tf32', True)
        # 设置是否使用TF32精度，如果kwargs中包含'with_tf32'，则使用其值，否则默认为True
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        # 设置TF32精度值，如果kwargs中包含'tf32_precision'，则使用其值，否则默认为0.001
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        # 设置是否检查批处理梯度，如果kwargs中包含'check_batched_grad'，则使用其值，否则默认为True
        self.default_dtype = kwargs.get('default_dtype', None)
        # 设置默认数据类型，默认从torch获取当前默认数据类型
        if self.default_dtype is None:
            self.default_dtype = torch.get_default_dtype()

    def __call__(self, test_case):
        # 实现了__call__方法，使得实例对象可以像函数一样被调用
        with set_default_dtype(self.default_dtype):
            # 设置默认数据类型为self.default_dtype

            module = self.constructor(*self.constructor_args)
            # 使用self.constructor及其参数构造模块对象

            input = self._get_input()
            # 获取输入数据

            # Check that these methods don't raise errors
            # 检查这些方法是否不会引发错误
            module.__repr__()
            str(module)

            target = self._get_target()
            # 获取目标数据

            if self.reference_fn is not None:
                # 如果有参考函数
                out = test_case._forward_criterion(module, input, target, extra_args=self.extra_args)
                # 使用模块、输入数据、目标数据以及额外参数计算前向评判
                ref_args = (deepcopy(input), deepcopy(target)) + self.extra_args + (module,)
                # 创建参考参数元组
                expected_out = self.reference_fn(*ref_args)
                # 使用参考参数调用参考函数
                test_case.assertEqual(out, expected_out)
                # 断言前向评判结果与期望输出结果相等

            if self.check_forward_only:
                # 如果仅需检查前向传播
                return

            params = tuple(x for x in module.parameters())
            # 获取模块的参数元组
            if not isinstance(input, tuple):
                # 如果输入数据不是元组
                inputs = (input,) + params + (target,)
                # 构造输入元组

                def apply_fn(input, target, *params):
                    # 定义应用函数，接受输入数据、目标数据和其他参数
                    return module(input, target)
                    # 返回模块的输出结果
            else:
                # 如果输入数据是元组
                inputs = input + params + (target,)

                def apply_fn(input1, input2, target, *params):  # type: ignore[misc]
                    # 定义应用函数，接受两个输入数据、目标数据和其他参数
                    return module(input1, input2, target)

            gradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)
            # 调用gradcheck函数，检查梯度

            if self.check_gradgrad:
                # 如果需要检查梯度梯度
                gradgradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)
                # 调用gradgradcheck函数，检查梯度梯度
    # 定义一个测试 CUDA 的方法，接受测试用例、数据类型和额外参数
    def test_cuda(self, test_case, dtype, extra_args=None):
        # 定义一个内部函数，用于将对象转换为指定数据类型，并可选择是否需要梯度
        def convert_dtype(obj, dtype, requires_grad=False):
            if isinstance(obj, torch.Tensor):
                return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
            elif isinstance(obj, tuple):
                return tuple(convert_dtype(o, dtype, requires_grad) for o in obj)
            else:
                return obj

        # 如果不支持 CUDA 测试或者不需要测试 CUDA，则跳过此测试
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')

        # 设置默认数据类型为指定的数据类型
        with set_default_dtype(self.default_dtype):
            # 获取 CPU 上的输入数据、目标数据和构造函数参数
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args)

            # 将 CPU 上的输入数据、目标数据和模型参数转换为指定的数据类型
            cpu_input = convert_dtype(cpu_input, dtype, True)
            if cpu_target.is_floating_point() or cpu_target.is_complex():
                cpu_target = convert_dtype(cpu_target, dtype)
            cpu_module.type(dtype)
            gpu_module.type(dtype)

            # 在 GPU 上设置输入数据、目标数据和模型参数
            gpu_input = to_gpu(cpu_input)
            gpu_target = to_gpu(cpu_target)
            gpu_module.cuda()

            # 如果数据类型是 torch.half 或者 torch.bfloat16，则回到默认数据类型进行操作
            if dtype in {torch.half, torch.bfloat16}:
                cpu_input = self._get_input()
                cpu_target = self._get_target()
                # 损失模块需要一致的输入和权重类型
                cpu_module = self.constructor(*self.constructor_args)

            # 在 CPU 和 GPU 上分别计算前向传播结果
            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            # 如果数据类型曾经可以是 None，则设置精度而不是精度映射
            test_case.assertEqual(cpu_output, gpu_output,
                                  atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4, rtol=0, exact_dtype=False)

            # 在 CPU 和 GPU 上分别计算反向传播梯度
            cpu_gradInput = test_case._backward_criterion(
                cpu_module, cpu_input, cpu_output, cpu_target, extra_args=extra_args)
            gpu_gradInput = test_case._backward_criterion(
                gpu_module, gpu_input, gpu_output, gpu_target, extra_args=extra_args)
            # 如果数据类型曾经可以是 None，则设置精度而不是精度映射
            test_case.assertEqual(cpu_gradInput, gpu_gradInput,
                                  atol=1e-1 if dtype in {torch.half, torch.bfloat16} else 4e-4, rtol=0, exact_dtype=False)

    # 获取目标数据的方法，返回存储在实例中的 'target' 参数
    def _get_target(self):
        return self._get_arg('target', False)

    # 返回存储在实例中的 'constructor_args' 参数作为属性
    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)

    @property
    # 定义一个方法 extra_args，用于获取对象的额外参数
    def extra_args(self):
        # 调用 _get_arg 方法，获取名为 'extra_args' 的参数值，默认为 False
        return self._get_arg('extra_args', False)
# 定义一个函数用于测试 bfloat16 运算的操作
def _test_bfloat16_ops(test_case, op, device, inp_dims=(), prec=1e-2, scale_factor=None):
    # 使用 torch.randn 创建指定维度的浮点数张量 input1，设备为指定设备，开启梯度追踪
    input1 = torch.randn(inp_dims, dtype=torch.float32, device=device, requires_grad=True)
    # 如果提供了 scale_factor，则用随机数生成与 input1 维度相同的 bfloat16 张量，并乘以 scale_factor，然后转换为浮点数张量，并开启梯度追踪
    if scale_factor is not None:
        input1 = (torch.rand(inp_dims, dtype=torch.bfloat16, device=device) * scale_factor).float().requires_grad_()
    # 使用 op 对 input1 进行操作，得到 out1
    out1 = op(input1)
    # 使用与 out1 相同维度的随机张量 grad_input1
    grad_input1 = torch.randn_like(out1, device=device)
    # 对 out1 进行反向传播
    out1.backward(grad_input1)

    # 使用 bfloat16 进行计算
    op_bfp16 = op.bfloat16()
    # 对 input1 进行分离（detach），转换为 bfloat16 张量，并开启梯度追踪
    input2 = input1.detach().bfloat16().requires_grad_()
    # 对 grad_input1 转换为 bfloat16 张量
    grad_input2 = grad_input1.bfloat16()
    # 使用 op_bfp16 对 input2 进行操作，得到 out2
    out2 = op_bfp16(input2)
    # 对 out2 进行反向传播
    out2.backward(grad_input2)

    # 断言两次操作得到的 out1 和 out2 相等，允许的误差为 prec
    test_case.assertEqual(out1, out2, atol=prec, rtol=prec, exact_dtype=False)
    # 断言两次操作得到的 input1 和 input2 的梯度数据相等，允许的误差为 prec
    test_case.assertEqual(input1.grad.data, input2.grad.data, atol=prec, rtol=prec, exact_dtype=False)


# 定义一个函数用于测试模块对空输入的反应
def _test_module_empty_input(test_case, module, inp, check_size=True, inference=False):
    # 如果不是推理模式，则开启 inp 的梯度追踪
    if not inference:
        inp.requires_grad_(True)
    # 将 inp 输入模块 module，得到 out
    out = module(inp)
    # 如果不是推理模式，则创建与 out 相同维度的随机张量 gO，并对 out 进行反向传播
    if not inference:
        gO = torch.rand_like(out)
        out.backward(gO)
    # 如果 check_size 为 True，则断言 out 的尺寸与 inp 的尺寸相同
    if check_size:
        test_case.assertEqual(out.size(), inp.size())
    # 如果不是推理模式，则检查 module 中所有需要梯度的参数，确保它们的梯度为零张量
    if not inference:
        for p in module.parameters():
            if p.requires_grad:
                test_case.assertEqual(p.grad, torch.zeros_like(p.grad))
        # 断言 inp 的梯度为零张量
        test_case.assertEqual(inp.grad, torch.zeros_like(inp))


# 定义一个函数用于创建一个基本的神经网络
def _create_basic_net():
    # 定义一个内部类 Layer，继承自 nn.Module
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            # 定义一个 nn.Parameter 参数 layer_dummy_param，形状为 (3, 5)
            self.layer_dummy_param = nn.Parameter(torch.empty(3, 5))
            # 注册一个缓冲区 layer_dummy_buf，形状为 (1, 3, 3, 7)，初始化为零张量
            self.register_buffer('layer_dummy_buf', torch.zeros(1, 3, 3, 7))

    # 定义一个神经网络类 Net，继承自 nn.Module
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # 实例化一个 Layer 类，并赋值给属性 l1
            self.l1 = Layer()
            # 定义一个 nn.Parameter 参数 dummy_param，形状为 (3, 5)
            self.dummy_param = nn.Parameter(torch.empty(3, 5))
            # 注册一个缓冲区 dummy_buf，形状为 (7, 3, 3, 1)，初始化为零张量
            self.register_buffer('dummy_buf', torch.zeros(7, 3, 3, 1))

    # 创建 Layer 类的实例 l
    l = Layer()
    # 创建 Net 类的实例 n
    n = Net()
    # 创建一个 nn.Sequential 容器 s，包含两个 n 实例
    s = nn.Sequential(n, n)

    # 返回创建的对象 l, n, s
    return l, n, s
```