# `.\pytorch\torch\_numpy\random.py`

```py
# mypy: ignore-errors

"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""
# 导入必要的模块和库
from __future__ import annotations

import functools
from math import sqrt
from typing import Optional

import torch

# 从本地模块导入特定功能
from . import _dtypes_impl, _util
from ._normalizations import array_or_scalar, ArrayLike, normalizer


__all__ = [
    "seed",
    "random_sample",
    "sample",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
    "randint",
    "shuffle",
    "uniform",
]

# 确定是否使用 NumPy 的随机数流
def use_numpy_random():
    # 避免循环引用，局部导入配置模块
    import torch._dynamo.config as config

    return config.use_numpy_random_stream

# 装饰器函数，用于根据配置选择使用 NumPy 或 Torch 的随机数生成函数
def deco_stream(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if not use_numpy_random():
            return func(*args, **kwds)
        else:
            import numpy

            from ._ndarray import ndarray

            # 获取对应的 NumPy 随机数生成函数
            f = getattr(numpy.random, func.__name__)

            # 将输入参数中的 Torch 张量转换为 NumPy ndarray
            args = tuple(
                arg.tensor.numpy() if isinstance(arg, ndarray) else arg for arg in args
            )
            kwds = {
                key: val.tensor.numpy() if isinstance(val, ndarray) else val
                for key, val in kwds.items()
            }

            # 调用 NumPy 随机数生成函数
            value = f(*args, **kwds)

            # `value` 可能是 NumPy ndarray 或 Python 标量（或 None）
            if isinstance(value, numpy.ndarray):
                value = ndarray(torch.as_tensor(value))

            return value

    return inner

# 设置随机种子的函数装饰器
@deco_stream
def seed(seed=None):
    if seed is not None:
        torch.random.manual_seed(seed)

# 生成随机样本的函数装饰器
@deco_stream
def random_sample(size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_()
    return array_or_scalar(values, return_scalar=size == ())

# 生成随机样本的函数别名
sample = random_sample
random = random_sample

# 生成指定形状随机数的函数
def rand(*size):
    if size == ():
        size = None
    return random_sample(size)

# 生成指定区间均匀分布随机数的函数装饰器
@deco_stream
def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_(low, high)
    return array_or_scalar(values, return_scalar=size == ())

# 生成指定形状正态分布随机数的函数装饰器
@deco_stream
def randn(*size):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.randn(size, dtype=dtype)
    return array_or_scalar(values, return_scalar=size == ())

# 生成指定参数正态分布随机数的函数装饰器
@deco_stream
def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).normal_(loc, scale)
    return array_or_scalar(values, return_scalar=size == ())

# 洗牌输入数组的函数装饰器，不进行归一化处理
@deco_stream
def shuffle(x):
    # no @normalizer because we do not cast e.g. lists to tensors
    # 从当前包中导入 _ndarray 模块中的 ndarray 类
    from ._ndarray import ndarray
    
    # 检查 x 是否为 torch.Tensor 类型
    if isinstance(x, torch.Tensor):
        # 如果是，则直接将 x 赋给 tensor
        tensor = x
    # 如果 x 是 ndarray 类型
    elif isinstance(x, ndarray):
        # 则将 x 的 tensor 属性赋给 tensor
        tensor = x.tensor
    else:
        # 如果 x 不是上述两种类型，则抛出未实现的异常，提示不支持对列表进行原地打乱
        raise NotImplementedError("We do not random.shuffle lists in-place")
    
    # 生成一个随机排列的索引，范围是 tensor 的第一个维度
    perm = torch.randperm(tensor.shape[0])
    
    # 根据 perm 数组重新排列 tensor 的行，并将结果赋给 xp
    xp = tensor[perm]
    
    # 将 xp 的内容复制到 tensor，实现原地修改
    tensor.copy_(xp)
# 装饰器函数，用于将 randint 函数与流处理装饰器结合
@deco_stream
# 用于生成指定范围随机整数或数组的函数
def randint(low, high=None, size=None):
    # 如果 size 未指定，则设为空元组
    if size is None:
        size = ()
    # 如果 size 不是 tuple 或 list 类型，则转换为单元素的元组
    if not isinstance(size, (tuple, list)):
        size = (size,)
    # 如果 high 未指定，则将 low 设为 0，high 设为 low，并生成指定大小的随机整数或数组
    if high is None:
        low, high = 0, low
    values = torch.randint(low, high, size=size)  # 使用 torch.randint 生成随机数
    return array_or_scalar(values, int, return_scalar=size == ())

# 装饰器函数，用于将 choice 函数与流处理装饰器结合，同时进行正则化处理
@deco_stream
@normalizer
# 从指定的数组中进行随机抽样的函数
def choice(a: ArrayLike, size=None, replace=True, p: Optional[ArrayLike] = None):
    # 如果 a 只有一个元素，则将其转换为范围内的整数序列
    if a.numel() == 1:
        a = torch.arange(a)

    # TODO: check a.dtype is integer -- cf np.random.choice(3.4) which raises

    # 确定抽样数量
    if size is None:
        num_el = 1
    elif _util.is_sequence(size):
        num_el = 1
        for el in size:
            num_el *= el
    else:
        num_el = size

    # 准备概率分布
    if p is None:
        p = torch.ones_like(a) / a.shape[0]

    # 检查概率分布的总和是否接近 1
    atol = sqrt(torch.finfo(p.dtype).eps)
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # 实际进行抽样操作
    indices = torch.multinomial(p, num_el, replacement=replace)

    # 如果 size 是一个序列，则对 indices 进行形状调整
    if _util.is_sequence(size):
        indices = indices.reshape(size)

    # 根据抽样结果取出样本
    samples = a[indices]

    return samples
```