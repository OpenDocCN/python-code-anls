# `.\pytorch\torch\_numpy\_util.py`

```py
# 忽略 mypy 的错误信息
# 导入 torch 和标准库以外的各种实用工具
"""Assorted utilities, which do not need anything other then torch and stdlib.
"""

# 导入运算符模块
import operator

# 导入 torch 库
import torch

# 导入本地的 _dtypes_impl 模块
from . import _dtypes_impl


# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
# 判断给定的序列是否是可迭代的序列（非字符串）
def is_sequence(seq):
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


# 自定义的异常类，继承自 ValueError 和 IndexError
class AxisError(ValueError, IndexError):
    pass


# 自定义的异常类，继承自 TypeError 和 RuntimeError
class UFuncTypeError(TypeError, RuntimeError):
    pass


# 根据需要进行数据类型转换，如果 dtype 不为 None 并且 tensor 的数据类型与 dtype 不同，则进行转换
def cast_if_needed(tensor, dtype):
    # 注意：如果 dtype 为 None，则不进行数据类型转换
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


# 将整数和布尔值转换为默认的浮点数数据类型
def cast_int_to_float(x):
    # 如果 x 的数据类型的分类小于 2（即整数或布尔值），则将其转换为默认的浮点数数据类型
    if _dtypes_impl._category(x.dtype) < 2:
        x = x.to(_dtypes_impl.default_dtypes().float_dtype)
    return x


# 根据数组的维度 ndim 对轴索引 ax 进行归一化处理
def normalize_axis_index(ax, ndim, argname=None):
    if not (-ndim <= ax < ndim):
        raise AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")
    if ax < 0:
        ax += ndim
    return ax


# 从 https://github.com/numpy/numpy/blob/main/numpy/core/numeric.py#L1378
# 将轴参数 axis 标准化为非负整数轴的元组
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.
    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`
    """
    # 如果 axis 不是元组或列表，则尝试将其转换为包含一个整数的列表
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # 对 axis 中的每个元素进行轴索引的归一化处理
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    # 如果不允许重复轴，并且 axis 中存在重复的轴索引，则抛出 ValueError
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        else:
            raise ValueError("repeated axis")
    return axis


# 允许只有单个轴参数的函数
def allow_only_single_axis(axis):
    if axis is None:
        return axis
    # 检查 axis 的长度是否为 1，如果不是，则抛出未实现错误异常
    if len(axis) != 1:
        raise NotImplementedError("does not handle tuple axis")
    # 返回 axis 列表中的第一个元素
    return axis[0]
# 根据给定的轴向扩展张量的形状
def expand_shape(arr_shape, axis):
    # 如果轴向不是列表或元组，转换为元组
    if type(axis) not in (list, tuple):
        axis = (axis,)
    # 计算输出的维度数
    out_ndim = len(axis) + len(arr_shape)
    # 标准化轴向元组，确保在正确范围内
    axis = normalize_axis_tuple(axis, out_ndim)
    # 使用生成器创建形状迭代器
    shape_it = iter(arr_shape)
    # 根据轴向信息创建扩展后的形状列表
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return shape


# 根据指定轴向和维度将张量应用于 keepdims 操作
def apply_keepdims(tensor, axis, ndim):
    if axis is None:
        # 如果张量是标量，创建全为1的形状
        shape = (1,) * ndim
        # 扩展张量的形状并确保连续性
        tensor = tensor.expand(shape).contiguous()
    else:
        # 根据轴向信息扩展张量的形状
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor


# 如果 axis 为 None，则将数组展平
def axis_none_flatten(*tensors, axis=None):
    if axis is None:
        # 对每个数组进行展平操作并返回
        tensors = tuple(ar.flatten() for ar in tensors)
        return tensors, 0
    else:
        # 如果指定了轴向，则直接返回原始数组和轴向
        return tensors, axis


# 将张量转换为指定的目标数据类型
def typecast_tensor(t, target_dtype, casting):
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to cast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

     Returns
     -------
    `torch.Tensor` of the `target_dtype` dtype

     Raises
     ------
     ValueError
        if the argument cannot be cast according to the `casting` rule

    """
    # 检查是否可以按照指定的规则进行类型转换
    can_cast = _dtypes_impl.can_cast_impl

    if not can_cast(t.dtype, target_dtype, casting=casting):
        # 如果无法按照规则进行类型转换，抛出类型错误
        raise TypeError(
            f"Cannot cast array data from {t.dtype} to"
            f" {target_dtype} according to the rule '{casting}'"
        )
    # 根据需要进行强制类型转换
    return cast_if_needed(t, target_dtype)


# 将一组张量转换为指定的目标数据类型
def typecast_tensors(tensors, target_dtype, casting):
    # 对每个张量进行类型转换，并返回转换后的结果组成的元组
    return tuple(typecast_tensor(t, target_dtype, casting) for t in tensors)


# 尝试将对象转换为张量
def _try_convert_to_tensor(obj):
    try:
        # 尝试将对象转换为张量
        tensor = torch.as_tensor(obj)
    except Exception as e:
        # 如果转换失败，抛出未实现错误，包含详细错误信息
        mesg = f"failed to convert {obj} to ndarray. \nInternal error is: {str(e)}."
        raise NotImplementedError(mesg)  # noqa: B904
    return tensor


# 强制将对象转换为张量的核心逻辑
def _coerce_to_tensor(obj, dtype=None, copy=False, ndmin=0):
    """The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not
    ndmin : int
        The results as least this many dimensions
    is_weak : bool
        Whether obj is a weakly typed python scalar.

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coersion function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    """
    # 如果 obj 已经是张量，则直接返回
    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        # 如果输入的 tensor 的元素不精确地可表示为 float32 类型，可能会丢失精度：
        # >>> torch.as_tensor(1e12).item() - 1e12
        # -4096.0
        # 设置默认的数据类型为 float32，以便于转换
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtypes_impl.get_default_dtype_for(torch.float32))
        try:
            # 尝试将 obj 转换为 tensor
            tensor = _try_convert_to_tensor(obj)
        finally:
            # 恢复默认的数据类型设置
            torch.set_default_dtype(default_dtype)

    # 如果需要，进行类型转换
    tensor = cast_if_needed(tensor, dtype)

    # 如果需要，调整 tensor 的维度
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)

    # 如果需要，执行拷贝操作
    if copy:
        tensor = tensor.clone()

    # 返回处理后的 tensor
    return tensor
def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    # 导入ndarray类
    from ._ndarray import ndarray

    # 如果没有输入参数，返回一个ValueError异常
    if len(inputs) == 0:
        return ValueError()
    # 如果只有一个输入参数
    elif len(inputs) == 1:
        input_ = inputs[0]
        # 如果输入参数是ndarray类型，则返回其对应的tensor
        if isinstance(input_, ndarray):
            return input_.tensor
        # 如果输入参数是元组
        elif isinstance(input_, tuple):
            result = []
            # 遍历元组中的每个元素，递归调用ndarrays_to_tensors函数
            for sub_input in input_:
                sub_result = ndarrays_to_tensors(sub_input)
                result.append(sub_result)
            # 返回转换后的元组
            return tuple(result)
        # 对于其他类型的输入参数，直接返回
        else:
            return input_
    # 如果有多个输入参数
    else:
        # 断言输入参数是元组，用于检查程序运行时的健全性
        assert isinstance(inputs, tuple)  # sanity check
        # 递归调用ndarrays_to_tensors函数，处理整个元组作为输入
        return ndarrays_to_tensors(inputs)
```