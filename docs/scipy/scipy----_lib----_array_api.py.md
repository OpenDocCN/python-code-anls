# `D:\src\scipysrc\scipy\scipy\_lib\_array_api.py`

```
"""
Utility functions to use Python Array API compatible libraries.

For the context about the Array API see:
https://data-apis.org/array-api/latest/purpose_and_scope.html

The SciPy use case of the Array API is described on the following page:
https://data-apis.org/array-api/latest/use_cases.html#use-case-scipy
"""
# 导入未来支持的语法特性，用于类型注解
from __future__ import annotations

import os  # 导入操作系统功能模块
import warnings  # 导入警告模块

from types import ModuleType  # 导入 ModuleType 类型
from typing import Any, Literal, TYPE_CHECKING  # 导入类型注解相关内容

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 类型注解

from scipy._lib import array_api_compat  # 导入 SciPy 的 Array API 兼容模块
from scipy._lib.array_api_compat import (
    is_array_api_obj,  # 导入检查是否为 Array API 对象的函数
    size,  # 导入获取对象大小的函数
    numpy as np_compat,  # 导入 NumPy 的别名 np_compat
    device  # 导入获取设备信息的函数
)

__all__ = ['array_namespace', '_asarray', 'size', 'device']


# 用于启用 Array API 和严格的类数组输入验证
SCIPY_ARRAY_API: str | bool = os.environ.get("SCIPY_ARRAY_API", False)
# 用于控制默认设备，仅在测试套件中使用
SCIPY_DEVICE = os.environ.get("SCIPY_DEVICE", "cpu")

_GLOBAL_CONFIG = {
    "SCIPY_ARRAY_API": SCIPY_ARRAY_API,
    "SCIPY_DEVICE": SCIPY_DEVICE,
}

# 如果在类型检查模式下
if TYPE_CHECKING:
    Array = Any  # 将 Array 定义为 Any 类型，稍后将更改为 Protocol 类型（参见 array-api#589）
    ArrayLike = Array | npt.ArrayLike  # ArrayLike 定义为 Array 或者 npt.ArrayLike 类型


def compliance_scipy(arrays: list[ArrayLike]) -> list[Array]:
    """Raise exceptions on known-bad subclasses.

    The following subclasses are not supported and raise and error:
    - `numpy.ma.MaskedArray`
    - `numpy.matrix`
    - NumPy arrays which do not have a boolean or numerical dtype
    - Any array-like which is neither array API compatible nor coercible by NumPy
    - Any array-like which is coerced by NumPy to an unsupported dtype
    """
    # 遍历数组列表
    for i in range(len(arrays)):
        array = arrays[i]
        # 检查是否为 MaskedArray 类型
        if isinstance(array, np.ma.MaskedArray):
            raise TypeError("Inputs of type `numpy.ma.MaskedArray` are not supported.")
        # 检查是否为 matrix 类型
        elif isinstance(array, np.matrix):
            raise TypeError("Inputs of type `numpy.matrix` are not supported.")
        # 如果是 ndarray 或者 generic 类型
        if isinstance(array, (np.ndarray, np.generic)):
            dtype = array.dtype
            # 如果 dtype 不是布尔类型或者数值类型，抛出异常
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                raise TypeError(f"An argument has dtype `{dtype!r}`; "
                                f"only boolean and numerical dtypes are supported.")
        # 如果不是 Array API 兼容对象
        elif not is_array_api_obj(array):
            try:
                array = np.asanyarray(array)
            except TypeError:
                raise TypeError("An argument is neither array API compatible nor "
                                "coercible by NumPy.")
            dtype = array.dtype
            # 如果 dtype 不是布尔类型或者数值类型，抛出异常
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                message = (
                    f"An argument was coerced to an unsupported dtype `{dtype!r}`; "
                    f"only boolean and numerical dtypes are supported."
                )
                raise TypeError(message)
            arrays[i] = array  # 将转换后的数组重新赋值给数组列表
    return arrays  # 返回处理后的数组列表
# 检查数组是否包含 NaN 或者 Inf
def _check_finite(array: Array, xp: ModuleType) -> None:
    """Check for NaNs or Infs."""
    msg = "array must not contain infs or NaNs"
    try:
        # 使用 array API 中的 xp 模块检查数组是否全部为有限值
        if not xp.all(xp.isfinite(array)):
            raise ValueError(msg)
    except TypeError:
        raise ValueError(msg)


# 获取数组 API 兼容的命名空间，用于一组数组 *arrays 的命名空间推断
def array_namespace(*arrays: Array) -> ModuleType:
    """Get the array API compatible namespace for the arrays xs.

    Parameters
    ----------
    *arrays : sequence of array_like
        Arrays used to infer the common namespace.

    Returns
    -------
    namespace : module
        Common namespace.

    Notes
    -----
    Thin wrapper around `array_api_compat.array_namespace`.

    1. Check for the global switch: SCIPY_ARRAY_API. This can also be accessed
       dynamically through ``_GLOBAL_CONFIG['SCIPY_ARRAY_API']``.
    2. `compliance_scipy` raise exceptions on known-bad subclasses. See
       its definition for more details.

    When the global switch is False, it defaults to the `numpy` namespace.
    In that case, there is no compliance check. This is a convenience to
    ease the adoption. Otherwise, arrays must comply with the new rules.
    """
    # 如果 SCIPY_ARRAY_API 全局开关为 False，则返回 numpy 兼容的命名空间 np_compat
    if not _GLOBAL_CONFIG["SCIPY_ARRAY_API"]:
        # 这里可以根据需要包装命名空间
        return np_compat

    # 筛选出非空的数组
    _arrays = [array for array in arrays if array is not None]

    # 对数组进行兼容性检查
    _arrays = compliance_scipy(_arrays)

    # 返回数组 API 兼容的命名空间
    return array_api_compat.array_namespace(*_arrays)


# SciPy 特有的 np.asarray 替代品，支持 order、check_finite 和 subok 参数
def _asarray(
        array: ArrayLike,
        dtype: Any = None,
        order: Literal['K', 'A', 'C', 'F'] | None = None,
        copy: bool | None = None,
        *,
        xp: ModuleType | None = None,
        check_finite: bool = False,
        subok: bool = False,
    ) -> Array:
    """SciPy-specific replacement for `np.asarray` with `order`, `check_finite`, and
    `subok`.

    Memory layout parameter `order` is not exposed in the Array API standard.
    `order` is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.

    `check_finite` is also not a keyword in the array API standard; included
    here for convenience rather than that having to be a separate function
    call inside SciPy functions.

    `subok` is included to allow this function to preserve the behaviour of
    `np.asanyarray` for NumPy based inputs.
    """
    # 如果 xp 为 None，则使用 array 的命名空间推断 xp
    if xp is None:
        xp = array_namespace(array)
    
    # 如果 xp 模块的名称为 "numpy" 或 "scipy._lib.array_api_compat.numpy"
    if xp.__name__ in {"numpy", "scipy._lib.array_api_compat.numpy"}:
        # 使用 NumPy API 来支持 order 参数
        if copy is True:
            array = np.array(array, order=order, dtype=dtype, subok=subok)
        elif subok:
            array = np.asanyarray(array, order=order, dtype=dtype)
        else:
            array = np.asarray(array, order=order, dtype=dtype)

        # 此时 array 是一个 NumPy ndarray，将其转换为与输入命名空间一致的数组容器
        array = xp.asarray(array)
    else:
        try:
            # 尝试将 array 转换为指定的 dtype 类型的数组，使用 xp.asarray() 方法
            array = xp.asarray(array, dtype=dtype, copy=copy)
        except TypeError:
            # 如果出现 TypeError 异常，则执行以下操作：
            # 创建一个 array_namespace 对象并强制转换为数组
            coerced_xp = array_namespace(xp.asarray(3))
            # 使用 coerced_xp 对象的 asarray 方法再次尝试将 array 转换为指定的 dtype 类型的数组
            array = coerced_xp.asarray(array, dtype=dtype, copy=copy)

    if check_finite:
        # 如果 check_finite 为 True，则调用 _check_finite 函数检查 array 是否包含有限数值
        _check_finite(array, xp)

    # 返回转换后的 array 数组
    return array
# 递归地扩展数组的维度，以至少达到指定的维数 `ndim`
def atleast_nd(x: Array, *, ndim: int, xp: ModuleType | None = None) -> Array:
    """Recursively expand the dimension to have at least `ndim`."""
    # 如果 xp 为 None，则使用 array_namespace(x) 来确定数组操作的命名空间
    if xp is None:
        xp = array_namespace(x)
    # 将 x 转换为 xp 中的数组表示
    x = xp.asarray(x)
    # 如果 x 的维度小于 ndim，则在第 0 轴上扩展维度
    if x.ndim < ndim:
        x = xp.expand_dims(x, axis=0)
        # 递归调用 atleast_nd，确保 x 达到至少 ndim 维度
        x = atleast_nd(x, ndim=ndim, xp=xp)
    return x


# 复制一个数组
def copy(x: Array, *, xp: ModuleType | None = None) -> Array:
    """
    Copies an array.

    Parameters
    ----------
    x : array
        要复制的数组

    xp : array_namespace
        数组操作的命名空间

    Returns
    -------
    copy : array
        复制的数组

    Notes
    -----
    此复制函数不提供 `np.copy` 的所有语义，例如 `subok` 和 `order` 关键字未使用。
    """
    # 注意：如果 xp 是 numpy，则 xp.asarray 会失败。
    if xp is None:
        xp = array_namespace(x)

    return _asarray(x, copy=True, xp=xp)


# 判断给定的模块是否为 numpy
def is_numpy(xp: ModuleType) -> bool:
    return xp.__name__ in ('numpy', 'scipy._lib.array_api_compat.numpy')


# 判断给定的模块是否为 cupy
def is_cupy(xp: ModuleType) -> bool:
    return xp.__name__ in ('cupy', 'scipy._lib.array_api_compat.cupy')


# 判断给定的模块是否为 torch
def is_torch(xp: ModuleType) -> bool:
    return xp.__name__ in ('torch', 'scipy._lib.array_api_compat.torch')


# 判断给定的模块是否为 jax
def is_jax(xp):
    return xp.__name__ in ('jax.numpy', 'jax.experimental.array_api')


# 判断给定的模块是否为 array_api_strict
def is_array_api_strict(xp):
    return xp.__name__ == 'array_api_strict'


# 执行严格检查，确保 actual 与 desired 在命名空间、数据类型和形状上匹配
def _strict_check(actual, desired, xp,
                  check_namespace=True, check_dtype=True, check_shape=True,
                  allow_0d=False):
    __tracebackhide__ = True  # 在 py.test 中隐藏 traceback
    if check_namespace:
        _assert_matching_namespace(actual, desired)

    # 检查 desired 是否是标量，如果是，则转换为 xp 中的数组表示
    was_scalar = np.isscalar(desired)
    desired = xp.asarray(desired)

    # 检查数据类型是否匹配
    if check_dtype:
        _msg = f"dtypes do not match.\nActual: {actual.dtype}\nDesired: {desired.dtype}"
        assert actual.dtype == desired.dtype, _msg

    # 检查形状是否匹配
    if check_shape:
        _msg = f"Shapes do not match.\nActual: {actual.shape}\nDesired: {desired.shape}"
        assert actual.shape == desired.shape, _msg
        # 如果 desired 是标量，则检查标量匹配性
        _check_scalar(actual, desired, xp, allow_0d=allow_0d, was_scalar=was_scalar)

    # 将 desired 广播到 actual 的形状
    desired = xp.broadcast_to(desired, actual.shape)
    return desired


# 断言 actual 和 desired 的命名空间匹配
def _assert_matching_namespace(actual, desired):
    __tracebackhide__ = True  # 在 py.test 中隐藏 traceback
    actual = actual if isinstance(actual, tuple) else (actual,)
    # 获取 desired 的数组命名空间
    desired_space = array_namespace(desired)
    for arr in actual:
        # 获取 arr 的数组命名空间
        arr_space = array_namespace(arr)
        _msg = (f"Namespaces do not match.\n"
                f"Actual: {arr_space.__name__}\n"
                f"Desired: {desired_space.__name__}")
        assert arr_space == desired_space, _msg


# 检查标量匹配性
def _check_scalar(actual, desired, xp, *, allow_0d, was_scalar):
    __tracebackhide__ = True  # 在 py.test 中隐藏 traceback
    # 只有当 desired.shape != () 或者 xp 不是 numpy 时，才需要进行形状检查
    if desired.shape != () or not is_numpy(xp):
        return
    # 如果 `was_scalar` 为 True，表示 `desired` 可能是一个0维数组，我们希望将其转换为标量
    if was_scalar:
        desired = desired[()]

    # 如果 `allow_0d` 为 True，我们允许 `actual` 和 `desired` 可能是标量或0维数组类型
    # 否则，我们期望 `actual` 是标量，因为很多函数在处理0维数组时会返回标量
    _msg = ("Types do not match:\n Actual: "
            f"{type(actual)}\n Desired: {type(desired)}")
    assert ((xp.isscalar(actual) and xp.isscalar(desired))
            or (not xp.isscalar(actual) and not xp.isscalar(desired))), _msg

    # 如果 `allow_0d` 为 False，这段代码会抛出一个错误信息，指出结果是一个NumPy 0维数组，
    # 但很多SciPy函数和NumPy函数的约定是当应该返回0维数组时返回标量。`xp_assert_` 函数默认
    # 是保守的，不会默认接受0维数组。如果正确的结果可能是一个NumPy 0维数组，应该传递 `allow_0d=True`。
    else:
        _msg = ("Result is a NumPy 0d array. Many SciPy functions intend to follow "
                "the convention of many NumPy functions, returning a scalar when a "
                "0d array would be correct. `xp_assert_` functions err on the side of "
                "caution and do not accept 0d arrays by default. If the correct result "
                "may be a 0d NumPy array, pass `allow_0d=True`.")
        assert xp.isscalar(actual), _msg
# 定义一个函数 xp_assert_equal，用于比较两个输入对象的相等性
def xp_assert_equal(actual, desired, check_namespace=True, check_dtype=True,
                    check_shape=True, allow_0d=False, err_msg='', xp=None):
    __tracebackhide__ = True  # 隐藏 pytest 的回溯信息
    # 如果 xp 参数为 None，则使用 actual 的命名空间创建 xp 对象
    if xp is None:
        xp = array_namespace(actual)
    # 对 desired 进行严格检查，确保与 actual 匹配
    desired = _strict_check(actual, desired, xp, check_namespace=check_namespace,
                            check_dtype=check_dtype, check_shape=check_shape,
                            allow_0d=allow_0d)
    # 如果 xp 是 Cupy 对象
    if is_cupy(xp):
        # 使用 Cupy 的断言函数检查两个数组是否完全相等
        return xp.testing.assert_array_equal(actual, desired, err_msg=err_msg)
    # 如果 xp 是 Torch 对象
    elif is_torch(xp):
        # PyTorch 建议使用 rtol=0, atol=0 进行精确相等性测试
        err_msg = None if err_msg == '' else err_msg
        return xp.testing.assert_close(actual, desired, rtol=0, atol=0, equal_nan=True,
                                       check_dtype=False, msg=err_msg)
    # 否则，默认使用 NumPy 的断言函数检查两个数组是否完全相等
    return np.testing.assert_array_equal(actual, desired, err_msg=err_msg)


# 定义一个函数 xp_assert_close，用于比较两个输入对象的接近程度
def xp_assert_close(actual, desired, rtol=None, atol=0, check_namespace=True,
                    check_dtype=True, check_shape=True, allow_0d=False,
                    err_msg='', xp=None):
    __tracebackhide__ = True  # 隐藏 pytest 的回溯信息
    # 如果 xp 参数为 None，则使用 actual 的命名空间创建 xp 对象
    if xp is None:
        xp = array_namespace(actual)
    # 对 desired 进行严格检查，确保与 actual 匹配
    desired = _strict_check(actual, desired, xp, check_namespace=check_namespace,
                            check_dtype=check_dtype, check_shape=check_shape,
                            allow_0d=allow_0d)

    # 检查 actual 是否为浮点数类型
    floating = xp.isdtype(actual.dtype, ('real floating', 'complex floating'))
    # 如果 rtol 未提供且 actual 是浮点数类型
    if rtol is None and floating:
        # 对于 np.float64，这里使用 4 倍的 sqrt(eps) 将默认的 rtol 放在
        # sqrt(eps) 和 `numpy.testing.assert_allclose` 的默认值 1e-7 之间
        rtol = xp.finfo(actual.dtype).eps**0.5 * 4
    # 如果 rtol 未提供，默认设置为 1e-7
    elif rtol is None:
        rtol = 1e-7

    # 如果 xp 是 Cupy 对象
    if is_cupy(xp):
        # 使用 Cupy 的断言函数检查两个数组的接近程度
        return xp.testing.assert_allclose(actual, desired, rtol=rtol,
                                          atol=atol, err_msg=err_msg)
    # 如果 xp 是 Torch 对象
    elif is_torch(xp):
        err_msg = None if err_msg == '' else err_msg
        # 使用 PyTorch 的断言函数检查两个数组的接近程度
        return xp.testing.assert_close(actual, desired, rtol=rtol, atol=atol,
                                       equal_nan=True, check_dtype=False, msg=err_msg)
    # 否则，默认使用 NumPy 的断言函数检查两个数组的接近程度
    return np.testing.assert_allclose(actual, desired, rtol=rtol,
                                      atol=atol, err_msg=err_msg)


# 定义一个函数 xp_assert_less，用于比较第一个输入对象是否小于第二个输入对象
def xp_assert_less(actual, desired, check_namespace=True, check_dtype=True,
                   check_shape=True, allow_0d=False, err_msg='', verbose=True, xp=None):
    __tracebackhide__ = True  # 隐藏 pytest 的回溯信息
    # 如果 xp 参数为 None，则使用 actual 的命名空间创建 xp 对象
    if xp is None:
        xp = array_namespace(actual)
    # 对 desired 进行严格检查，确保与 actual 匹配
    desired = _strict_check(actual, desired, xp, check_namespace=check_namespace,
                            check_dtype=check_dtype, check_shape=check_shape,
                            allow_0d=allow_0d)
    # 如果使用的是 Cupy 库（一个用于数组计算的库），调用其测试函数 assert_array_less 比较 actual 和 desired
    if is_cupy(xp):
        return xp.testing.assert_array_less(actual, desired,
                                            err_msg=err_msg, verbose=verbose)
    # 如果使用的是 PyTorch 库，需要检查 actual 和 desired 是否在 GPU 上，若是，则移到 CPU 上进行比较
    elif is_torch(xp):
        if actual.device.type != 'cpu':
            actual = actual.cpu()  # 将 actual 数据移动到 CPU
        if desired.device.type != 'cpu':
            desired = desired.cpu()  # 将 desired 数据移动到 CPU
    # 如果使用的是 JAX 库（一个用于数值计算的库），使用其 np.testing 模块中的 assert_array_less 函数比较 actual 和 desired
    # （JAX 框架不直接调用库函数，而是通过 np.testing 来调用 NumPy 测试函数）
    return np.testing.assert_array_less(actual, desired,
                                        err_msg=err_msg, verbose=verbose)
# 计算协方差矩阵，返回结果
def cov(x: Array, *, xp: ModuleType | None = None) -> Array:
    # 如果未指定特定的计算库，则使用默认的数组命名空间来初始化 xp
    if xp is None:
        xp = array_namespace(x)

    # 复制输入数组 x，使用指定的计算库 xp
    X = copy(x, xp=xp)
    # 推断结果数据类型为 float64
    dtype = xp.result_type(X, xp.float64)

    # 将 X 至少转换为二维数组，使用指定的计算库 xp
    X = atleast_nd(X, ndim=2, xp=xp)
    # 将 X 转换为指定数据类型的数组，使用指定的计算库 xp
    X = xp.asarray(X, dtype=dtype)

    # 计算 X 按行的平均值，使用指定的计算库 xp
    avg = xp.mean(X, axis=1)
    # 计算 X 的列数减 1，作为分母
    fact = X.shape[1] - 1

    # 如果分母 <= 0，则发出警告，并将 fact 设置为 0.0
    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    # 从 X 的每行减去平均值，使用广播操作，使得 X 变为零均值
    X -= avg[:, None]
    # 计算 X 的转置
    X_T = X.T
    # 如果 X_T 的数据类型为复数浮点数，则取其共轭
    if xp.isdtype(X_T.dtype, 'complex floating'):
        X_T = xp.conj(X_T)
    # 计算 X 和其转置的乘积，得到协方差矩阵 c
    c = X @ X_T
    # 将 c 除以 fact，得到标准化的协方差矩阵
    c /= fact
    # 寻找 c 中长度为 1 的维度，将其作为轴来压缩（去除）
    axes = tuple(axis for axis, length in enumerate(c.shape) if length == 1)
    return xp.squeeze(c, axis=axes)


# 返回一个错误消息，说明提供的参数不支持当前的计算库
def xp_unsupported_param_msg(param: Any) -> str:
    return f'Providing {param!r} is only supported for numpy arrays.'


# 检查数组是否为复数类型
def is_complex(x: Array, xp: ModuleType) -> bool:
    return xp.isdtype(x.dtype, 'complex floating')


# 获取指定计算库中可用设备的列表
def get_xp_devices(xp: ModuleType) -> list[str] | list[None]:
    """Returns a list of available devices for the given namespace."""
    devices: list[str] = []
    # 如果使用的是 PyTorch 计算库
    if is_torch(xp):
        # 将 'cpu' 添加到设备列表
        devices += ['cpu']
        import torch # type: ignore[import]
        # 获取 CUDA 设备的数量，并将其编号形式添加到设备列表
        num_cuda = torch.cuda.device_count()
        for i in range(0, num_cuda):
            devices += [f'cuda:{i}']
        # 如果支持多进程服务，则将 'mps' 添加到设备列表
        if torch.backends.mps.is_available():
            devices += ['mps']
        return devices
    # 如果使用的是 CuPy 计算库
    elif is_cupy(xp):
        import cupy # type: ignore[import]
        # 获取 CUDA 设备的数量，并将其编号形式添加到设备列表
        num_cuda = cupy.cuda.runtime.getDeviceCount()
        for i in range(0, num_cuda):
            devices += [f'cuda:{i}']
        return devices
    # 如果使用的是 JAX 计算库
    elif is_jax(xp):
        import jax # type: ignore[import]
        # 获取 CPU 设备的数量，并将其编号形式添加到设备列表
        num_cpu = jax.device_count(backend='cpu')
        for i in range(0, num_cpu):
            devices += [f'cpu:{i}']
        # 获取 GPU 设备的数量，并将其编号形式添加到设备列表
        num_gpu = jax.device_count(backend='gpu')
        for i in range(0, num_gpu):
            devices += [f'gpu:{i}']
        # 获取 TPU 设备的数量，并将其编号形式添加到设备列表
        num_tpu = jax.device_count(backend='tpu')
        for i in range(0, num_tpu):
            devices += [f'tpu:{i}']
        return devices

    # 如果给定的计算库没有已知的可用设备列表，则返回 `[None]`
    # 这样可以在 `device=None` 的测试中使用
    return [None]


# 返回一个非 NumPy 计算库的 `scipy` 类似命名空间
def scipy_namespace_for(xp: ModuleType) -> ModuleType | None:
    """Return the `scipy`-like namespace of a non-NumPy backend

    That is, return the namespace corresponding with backend `xp` that contains
    `scipy` sub-namespaces like `linalg` and `special`. If no such namespace
    exists, return ``None``. Useful for dispatching.
    """

    # 如果使用的是 CuPy 计算库，则返回 cupyx.scipy 命名空间
    if is_cupy(xp):
        import cupyx  # type: ignore[import-not-found,import-untyped]
        return cupyx.scipy

    # 如果使用的是 JAX 计算库，则返回 jax.scipy 命名空间
    if is_jax(xp):
        import jax  # type: ignore[import-not-found]
        return jax.scipy

    # 如果使用的是 PyTorch 计算库，则返回 xp 本身（即 PyTorch 命名空间）
    if is_torch(xp):
        return xp

    # 如果使用的计算库不是已知的 NumPy 后端，则返回 `None`
    return None
# 使用数组命名空间函数来获取特定的数组库或命名空间
def xp_minimum(x1: Array, x2: Array, /) -> Array:
    xp = array_namespace(x1, x2)
    # 检查 xp 对象是否具有 'minimum' 属性
    if hasattr(xp, 'minimum'):
        # 若有 'minimum' 方法，则使用该方法计算 x1 和 x2 的元素级最小值
        return xp.minimum(x1, x2)
    # 若 xp 没有 'minimum' 方法，则使用广播数组的方式计算最小值
    x1, x2 = xp.broadcast_arrays(x1, x2)
    # 创建布尔索引，用于标识 x2 中小于 x1 或为 NaN 的元素
    i = (x2 < x1) | xp.isnan(x2)
    # 使用 xp.where 函数根据布尔索引选择 x2 或 x1 的元素构成结果数组 res
    res = xp.where(i, x2, x1)
    # 若 res 是零维数组，则返回其标量值；否则返回数组 res
    return res[()] if res.ndim == 0 else res


# 临时替代 xp.clip 的函数，因为并非所有后端都支持或被 array_api_compat 覆盖
def xp_clip(
        x: Array,
        /,
        min: int | float | Array | None = None,
        max: int | float | Array | None = None,
        *,
        xp: ModuleType | None = None) -> Array:
    xp = array_namespace(x) if xp is None else xp
    # 将 min 和 max 转换为与 x 相同类型的数组，并使用 xp.asarray 函数
    a, b = xp.asarray(min, dtype=x.dtype), xp.asarray(max, dtype=x.dtype)
    # 检查 xp 对象是否具有 'clip' 方法
    if hasattr(xp, 'clip'):
        # 若有 'clip' 方法，则使用该方法对 x 进行裁剪
        return xp.clip(x, a, b)
    # 否则，使用广播数组的方式进行裁剪
    x, a, b = xp.broadcast_arrays(x, a, b)
    # 创建 x 的副本 y，并将其视为 xp 数组
    y = xp.asarray(x, copy=True)
    # 根据条件 ia 将 y 中小于 a 的元素设为 a
    ia = y < a
    y[ia] = a[ia]
    # 根据条件 ib 将 y 中大于 b 的元素设为 b
    ib = y > b
    y[ib] = b[ib]
    # 若 y 是零维数组，则返回其标量值；否则返回数组 y
    return y[()] if y.ndim == 0 else y


# 临时替代 xp.moveaxis 的函数，因为并非所有后端都支持或被 array_api_compat 覆盖
def xp_moveaxis_to_end(
        x: Array,
        source: int,
        /, *,
        xp: ModuleType | None = None) -> Array:
    xp = array_namespace(xp) if xp is None else xp
    # 创建包含 x.ndim 个轴索引的列表 axes
    axes = list(range(x.ndim))
    # 从 axes 中移除 source 索引，并将其添加到末尾
    temp = axes.pop(source)
    axes = axes + [temp]
    # 使用 xp.permute_dims 函数根据新的轴顺序 axes 进行维度重排
    return xp.permute_dims(x, axes)


# 临时替代 xp.copysign 的函数，因为并非所有后端都支持或被 array_api_compat 覆盖
def xp_copysign(x1: Array, x2: Array, /, *, xp: ModuleType | None = None) -> Array:
    # 不考虑特殊情况的实现
    xp = array_namespace(x1, x2) if xp is None else xp
    # 计算 x1 的绝对值
    abs_x1 = xp.abs(x1)
    # 使用 xp.where 函数根据 x2 的正负号对 abs_x1 进行符号设置
    return xp.where(x2 >= 0, abs_x1, -abs_x1)


# 临时替代 xp.sign 的函数，因为并非所有后端都支持或被 array_api_compat 覆盖
# 这个函数也未处理 NaN 的特殊情况（https://github.com/data-apis/array-api-compat/issues/136）
def xp_sign(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    xp = array_namespace(x) if xp is None else xp
    # 如果是 NumPy 后端，则调用 xp.sign 处理特殊情况
    if is_numpy(xp):
        return xp.sign(x)
    # 否则，创建一个与 x 相同形状的数组 sign，并用 NaN 填充
    sign = xp.full_like(x, xp.nan)
    # 创建数组 one，并将其设为 x 的数据类型的 1
    one = xp.asarray(1, dtype=x.dtype)
    # 使用 xp.where 函数根据 x 的正负号设置 sign 数组的值
    sign = xp.where(x > 0, one, sign)
    sign = xp.where(x < 0, -one, sign)
    sign = xp.where(x == 0, 0*one, sign)
    # 返回 sign 数组
    return sign
```