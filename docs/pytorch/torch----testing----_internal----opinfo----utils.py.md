# `.\pytorch\torch\testing\_internal\opinfo\utils.py`

```
# mypy: ignore-errors

# 导入必要的模块和库
import collections  # 导入 collections 模块
import warnings     # 导入 warnings 模块
from functools import partial, wraps  # 导入 functools 模块中的 partial 和 wraps 函数
from typing import Sequence  # 导入 typing 模块中的 Sequence 类型

import numpy as np  # 导入 numpy 库，并使用 np 作为别名

import torch  # 导入 torch 库
from torch.testing._internal.common_cuda import TEST_CUDA  # 从 torch.testing._internal.common_cuda 导入 TEST_CUDA 变量
from torch.testing._internal.common_dtype import (
    _dispatch_dtypes,  # 从 torch.testing._internal.common_dtype 导入 _dispatch_dtypes 类
    all_types, all_types_and, all_types_and_complex, all_types_and_half,  # 导入多个数据类型集合
    complex_types, floating_and_complex_types, floating_and_complex_types_and,
    floating_types, floating_types_and, floating_types_and_half,
    integral_types, integral_types_and,
)
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict  # 从 torch.testing._internal.common_utils 导入 torch_to_numpy_dtype_dict 函数

# 定义常量 COMPLETE_DTYPES_DISPATCH 和 EXTENSIBLE_DTYPE_DISPATCH
COMPLETE_DTYPES_DISPATCH = (
    all_types, all_types_and_complex, all_types_and_half,  # 包含多种数据类型的元组
    floating_types, floating_and_complex_types, floating_types_and_half,
    integral_types, complex_types,
)

EXTENSIBLE_DTYPE_DISPATCH = (
    all_types_and_complex_and,  # 包含多种数据类型的元组
    floating_types_and, floating_and_complex_types_and,
    integral_types_and, all_types_and,
)

# 获取设备列表 DEVICES，包含 "cpu" 和 "cuda"（如果 TEST_CUDA 为 True）
DEVICES = ["cpu"] + (["cuda"] if TEST_CUDA else [])

# 定义 _dynamic_dispatch_dtypes 类，用于标记动态生成的数据类型
class _dynamic_dispatch_dtypes(_dispatch_dtypes):
    pass  # 空的类定义，继承自 _dispatch_dtypes

def get_supported_dtypes(op, sample_inputs_fn, device_type):
    # 返回给定操作符和设备类型支持的数据类型集合
    assert device_type in ["cpu", "cuda"]  # 断言设备类型为 "cpu" 或 "cuda"
    if not TEST_CUDA and device_type == "cuda":
        warnings.warn(
            "WARNING: CUDA is not available, empty_dtypes dispatch will be returned!"
        )
        return _dynamic_dispatch_dtypes(())  # 如果 CUDA 不可用，返回一个空的 _dynamic_dispatch_dtypes 对象

    supported_dtypes = set()  # 创建一个空集合，用于存储支持的数据类型
    for dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
        try:
            samples = sample_inputs_fn(op, device_type, dtype, False)  # 尝试获取给定操作符和数据类型的样本输入
        except RuntimeError:
            # 如果 sample_inputs_fn 不支持某个数据类型的样本生成，记录警告并继续下一个数据类型的检查
            warnings.warn(
                f"WARNING: Unable to generate sample for device:{device_type} and dtype:{dtype}"
            )
            continue

        # 假设数据类型支持，仅当所有样本输入都通过时，才认为数据类型是支持的
        supported = True
        for sample in samples:
            try:
                op(sample.input, *sample.args, **sample.kwargs)  # 调用操作符，并传入样本输入及其参数
            except RuntimeError as re:
                # 如果操作符抛出 RuntimeError，则说明该数据类型不支持，设置 supported 为 False 并跳出循环
                supported = False
                break

        if supported:
            supported_dtypes.add(dtype)  # 如果数据类型支持，则将其添加到 supported_dtypes 集合中

    return _dynamic_dispatch_dtypes(supported_dtypes)  # 返回一个 _dynamic_dispatch_dtypes 对象，包含支持的数据类型集合

def dtypes_dispatch_hint(dtypes):
    # 根据给定的数据类型集合，返回适当的分发函数（从 COMPLETE_DTYPES_DISPATCH 和 EXTENSIBLE_DTYPE_DISPATCH 中选择）
    pass  # 函数体暂时为空，需要根据具体需求实现
    # 创建一个具名元组(return_type)，包含两个字段：dispatch_fn 和 dispatch_fn_str，用于返回不同类型的分发函数和它们的字符串表示。
    return_type = collections.namedtuple("return_type", "dispatch_fn dispatch_fn_str")

    # 如果传入的 dtypes 列表为空，则说明 CUDA 不可用，直接返回空的分发函数和空字符串表示。
    if len(dtypes) == 0:
        return return_type((), str(tuple()))

    # 将 dtypes 转换为集合，方便后续操作。
    set_dtypes = set(dtypes)

    # 遍历完全匹配的分发函数列表 COMPLETE_DTYPES_DISPATCH。
    for dispatch in COMPLETE_DTYPES_DISPATCH:
        # 如果当前分发函数的输出集合与 set_dtypes 完全相等，则找到了完全匹配，返回该分发函数和其名称字符串表示。
        if set(dispatch()) == set_dtypes:
            return return_type(dispatch, dispatch.__name__ + "()")

    # 初始化最佳匹配分发函数和其对应的分数。
    chosen_dispatch = None
    chosen_dispatch_score = 0.0

    # 遍历可扩展匹配的分发函数列表 EXTENSIBLE_DTYPE_DISPATCH。
    for dispatch in EXTENSIBLE_DTYPE_DISPATCH:
        # 获取当前分发函数的输出集合。
        dispatch_dtypes = set(dispatch())
        # 如果 dispatch_dtypes 不是 set_dtypes 的子集，则跳过当前分发函数。
        if not dispatch_dtypes.issubset(set_dtypes):
            continue

        # 计算当前分发函数的分数，即其输出集合的长度。
        score = len(dispatch_dtypes)
        # 如果当前分数比已记录的最佳匹配分数更高，则更新最佳匹配分发函数和分数。
        if score > chosen_dispatch_score:
            chosen_dispatch_score = score
            chosen_dispatch = dispatch

    # 如果没有找到匹配的分发函数，则返回一个空的分发函数和 dtypes 的字符串表示。
    if chosen_dispatch is None:
        return return_type((), str(dtypes))

    # 构造最终返回的具名元组(return_type)，其中 dispatch_fn 是部分应用的 chosen_dispatch 函数，其参数为 set(dtypes) 与 set(dispatch()) 的差集。
    # dispatch_fn_str 是 chosen_dispatch 函数的名称字符串表示，加上差集的字符串形式。
    return return_type(
        partial(dispatch, *tuple(set(dtypes) - set(dispatch()))),
        dispatch.__name__ + str(tuple(set(dtypes) - set(dispatch()))),
    )
# 检查操作是否使用动态获取的数据类型信息
def is_dynamic_dtype_set(op):
    # 返回操作的动态数据类型标记
    return op.dynamic_dtypes


# 格式化字符串，描述操作的动态数据类型信息
def str_format_dynamic_dtype(op):
    fmt_str = f"""
        OpInfo({op.name},
               dtypes={dtypes_dispatch_hint(op.dtypes).dispatch_fn_str},
               dtypesIfCUDA={dtypes_dispatch_hint(op.dtypesIfCUDA).dispatch_fn_str},
        )
        """
    return fmt_str


# NumPy 一元通用函数整数提升包装器
def np_unary_ufunc_integer_promotion_wrapper(fn):
    # 包装器，当输入为整数时，传递PyTorch的默认标量类型作为参数给包装后的NumPy一元ufunc函数。
    # 这模仿了PyTorch将整数类型提升为浮点类型的方式。
    
    # 判断是否需要类型提升的辅助函数
    def is_integral(dtype):
        return dtype in [
            np.bool_,
            bool,
            np.uint8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]

    @wraps(fn)
    def wrapped_fn(x):
        # 当函数被调用时，获取默认的dtype，因为默认dtype可能会改变。
        np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]

        # 如果输入是整数类型，则使用默认dtype对其进行类型提升
        if is_integral(x.dtype):
            return fn(x.astype(np_dtype))
        # 否则直接调用函数
        return fn(x)

    return wrapped_fn


# 包装一个NumPy的归约操作函数
def reference_reduction_numpy(f, supports_keepdims=True):
    """包装一个NumPy的归约操作函数。

    包装函数将转发dim、keepdim、mask和identity参数给被包装的函数，作为NumPy中相应的axis、keepdims、where和initial参数。

    Args:
        f: 需要包装的NumPy归约操作函数
        supports_keepdims (bool, optional): NumPy操作函数是否接受keepdims参数。
            如果不接受，包装器会手动扩展被调用时keepdim=True时的减少维度。默认为True。

    Returns:
        包装后的函数

    """

    @wraps(f)
    # 定义一个装饰器函数，接受一个 NumPy 数组和其他参数，并返回一个处理后的结果
    def wrapper(x: np.ndarray, *args, **kwargs):
        # 将关键字参数的键复制到一个集合中
        keys = set(kwargs.keys())

        # 从 kwargs 中弹出 "dim" 参数，如果没有则返回 None
        dim = kwargs.pop("dim", None)
        
        # 从 kwargs 中弹出 "keepdim" 参数，如果没有则默认为 False
        keepdim = kwargs.pop("keepdim", False)

        # 如果 keys 集合中包含 "dim" 键
        if "dim" in keys:
            # 如果 dim 是序列对象，则转换为元组
            dim = tuple(dim) if isinstance(dim, Sequence) else dim
            
            # 如果输入数组 x 是标量（ndim == 0），且 dim 在 {0, -1, (0,), (-1,)} 中
            # 则将 kwargs 中的 "axis" 设置为 None，因为 NumPy 不接受 dim=0 的标量输入
            else:
                kwargs["axis"] = dim

        # 如果 keys 集合中包含 "keepdim" 键，并且 supports_keepdims 为 True
        if "keepdim" in keys and supports_keepdims:
            # 将 kwargs 中的 "keepdims" 设置为 keepdim 的值
            kwargs["keepdims"] = keepdim

        # 如果 keys 集合中包含 "mask" 键
        if "mask" in keys:
            # 从 kwargs 中弹出 "mask" 参数
            mask = kwargs.pop("mask")
            # 如果 mask 不为 None，则断言 mask 的布局是 torch.strided
            assert mask.layout == torch.strided
            # 将 kwargs 中的 "where" 设置为 mask 在 CPU 上的 NumPy 数组表示
            kwargs["where"] = mask.cpu().numpy()

        # 如果 keys 集合中包含 "identity" 键
        if "identity" in keys:
            # 从 kwargs 中弹出 "identity" 参数
            identity = kwargs.pop("identity")
            # 如果 identity 不为 None
            if identity is not None:
                # 如果 identity 的数据类型是 torch.bfloat16
                if identity.dtype is torch.bfloat16:
                    # 将 identity 转换为 torch.float32 类型并移到 CPU 上
                    identity = identity.cpu().to(torch.float32)
                else:
                    # 将 identity 移到 CPU 上
                    identity = identity.cpu()
                # 将 kwargs 中的 "initial" 设置为 identity 的 NumPy 数组表示
                kwargs["initial"] = identity.numpy()

        # 调用函数 f，传入 x 和其他参数以及处理后的 kwargs
        result = f(x, *args, **kwargs)

        # 如果 keepdim 为 True，并且 supports_keepdims 为 False，且输入数组 x 的维度大于 0
        if keepdim and not supports_keepdims and x.ndim > 0:
            # 如果 dim 是 None，则使用 x 的所有维度索引；否则使用 dim 指定的维度索引
            dim = list(range(x.ndim)) if dim is None else dim
            # 在结果 result 的 dim 维度上扩展维度
            result = np.expand_dims(result, dim)

        # 返回处理后的结果
        return result

    # 返回装饰器函数 wrapper
    return wrapper
# 定义一个函数，用于计算输入数组的乘积，根据输入数组的数据类型选择合适的整数类型以避免整数溢出问题
def prod_numpy(a, *args, **kwargs):
    """
    The function will call np.prod with type as np.int64 if the input type
    is int or uint64 if is uint. This is necessary because windows np.prod uses by default
    int32 while on linux it uses int64.
    This is for fixing integer overflow https://github.com/pytorch/pytorch/issues/77320

    Returns:
        np.prod of input
    """
    # 如果用户没有指定 dtype 关键字参数
    if "dtype" not in kwargs:
        # 如果数组 a 的数据类型是有符号整数类型
        if np.issubdtype(a.dtype, np.signedinteger):
            # 将数组 a 转换为 np.int64 类型
            a = a.astype(np.int64)
        # 如果数组 a 的数据类型是无符号整数类型
        elif np.issubdtype(a.dtype, np.unsignedinteger):
            # 将数组 a 转换为 np.uint64 类型
            a = a.astype(np.uint64)

    # 获取参考的 numpy 函数，此处为 reference_reduction_numpy(np.prod)
    fn = reference_reduction_numpy(np.prod)
    # 调用参考的 numpy 函数 fn，计算数组 a 的乘积，并返回结果
    return fn(a, *args, **kwargs)
```