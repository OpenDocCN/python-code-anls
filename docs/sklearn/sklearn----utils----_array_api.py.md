# `D:\src\scipysrc\scikit-learn\sklearn\utils\_array_api.py`

```
# 导入所需模块和库
"""Tools to support array_api."""

# 导入迭代工具
import itertools
# 导入数学函数库
import math
# 导入装饰器模块
from functools import wraps

# 导入 NumPy 库
import numpy
# 导入 SciPy 的特殊函数模块
import scipy.special as special

# 从当前包的配置模块导入配置信息获取函数
from .._config import get_config
# 从修复模块导入版本解析函数
from .fixes import parse_version

# 定义 NumPy 相关的命名空间集合
_NUMPY_NAMESPACE_NAMES = {"numpy", "array_api_compat.numpy"}


# 生成支持的命名空间
def yield_namespaces(include_numpy_namespaces=True):
    """Yield supported namespace.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    include_numpy_namespaces : bool, default=True
        If True, also yield numpy namespaces.

    Yields
    ------
    array_namespace : str
        The name of the Array API namespace.
    """
    for array_namespace in [
        # 以下用于测试 array_api_compat 封装器在启用 array_api_dispatch 时的效果：
        # 特别是测试中使用的数组是普通的 NumPy 数组，没有任何 "device" 属性。
        "numpy",
        # 更严格的基于 NumPy 的 Array API 实现。array_api_strict.Array 实例始终具有虚拟的 "device" 属性。
        "array_api_strict",
        "cupy",
        "cupy.array_api",
        "torch",
    ]:
        if not include_numpy_namespaces and array_namespace in _NUMPY_NAMESPACE_NAMES:
            continue
        yield array_namespace


# 生成命名空间、设备和数据类型组合，用于测试
def yield_namespace_device_dtype_combinations(include_numpy_namespaces=True):
    """Yield supported namespace, device, dtype tuples for testing.

    Use this to test that an estimator works with all combinations.

    Parameters
    ----------
    include_numpy_namespaces : bool, default=True
        If True, also yield numpy namespaces.

    Yields
    ------
    array_namespace : str
        The name of the Array API namespace.

    device : str
        The name of the device on which to allocate the arrays. Can be None to
        indicate that the default value should be used.

    dtype_name : str
        The name of the data type to use for arrays. Can be None to indicate
        that the default value should be used.
    """
    for array_namespace in yield_namespaces(
        include_numpy_namespaces=include_numpy_namespaces
    ):
        if array_namespace == "torch":
            for device, dtype in itertools.product(
                ("cpu", "cuda"), ("float64", "float32")
            ):
                yield array_namespace, device, dtype
            yield array_namespace, "mps", "float32"
        else:
            yield array_namespace, None, None


# 检查 array_api_compat 是否已安装，并检查 NumPy 版本是否兼容
def _check_array_api_dispatch(array_api_dispatch):
    """Check that array_api_compat is installed and NumPy version is compatible.

    array_api_compat follows NEP29, which has a higher minimum NumPy version than
    scikit-learn.
    """
    # 如果 array_api_dispatch 变量为真，则执行以下代码块
    if array_api_dispatch:
        # 尝试导入 array_api_compat 模块，忽略 Flake8 的 noqa 标记
        try:
            import array_api_compat  # noqa
        # 如果导入失败，抛出 ImportError 异常
        except ImportError:
            raise ImportError(
                "array_api_compat is required to dispatch arrays using the API"
                " specification"
            )

        # 解析当前 NumPy 的版本号
        numpy_version = parse_version(numpy.__version__)
        # 设置最低支持的 NumPy 版本为 "1.21"
        min_numpy_version = "1.21"
        # 如果当前 NumPy 版本低于最低要求的版本
        if numpy_version < parse_version(min_numpy_version):
            # 抛出 ImportError 异常，要求 NumPy 版本必须大于或等于 "1.21"
            raise ImportError(
                f"NumPy must be {min_numpy_version} or newer to dispatch array using"
                " the API specification"
            )
# 返回给定数组所在的硬件设备。

def _single_array_device(array):
    """Hardware device where the array data resides on."""
    # 如果数组是 NumPy 的 ndarray 或者标量类型，或者没有 `device` 属性，则返回 "cpu"
    if isinstance(array, (numpy.ndarray, numpy.generic)) or not hasattr(
        array, "device"
    ):
        return "cpu"
    else:
        # 否则返回数组的设备属性
        return array.device


def device(*array_list, remove_none=True, remove_types=(str,)):
    """Hardware device where the array data resides on.

    If the hardware device is not the same for all arrays, an error is raised.

    Parameters
    ----------
    *array_list : arrays
        List of array instances from NumPy or an array API compatible library.

    remove_none : bool, default=True
        Whether to ignore None objects passed in array_list.

    remove_types : tuple or list, default=(str,)
        Types to ignore in array_list.

    Returns
    -------
    out : device
        `device` object (see the "Device Support" section of the array API spec).
    """
    # 移除非数组对象，并确保 array_list 不为空
    array_list = _remove_non_arrays(
        *array_list, remove_none=remove_none, remove_types=remove_types
    )

    # 注意: _remove_non_arrays 确保 array_list 不为空。
    # 获取第一个数组的设备
    device_ = _single_array_device(array_list[0])

    # 注意: 这里不能简单地使用 Python 的 `set`，因为数组 API 设备对象不一定是可哈希的。
    # 特别是，在撰写本注释时，CuPy 设备不可哈希。
    # 检查所有数组的设备是否相同
    for array in array_list[1:]:
        device_other = _single_array_device(array)
        if device_ != device_other:
            # 如果不同，抛出 ValueError
            raise ValueError(
                f"Input arrays use different devices: {str(device_)}, "
                f"{str(device_other)}"
            )

    return device_


def size(x):
    """Return the total number of elements of x.

    Parameters
    ----------
    x : array
        Array instance from NumPy or an array API compatible library.

    Returns
    -------
    out : int
        Total number of elements.
    """
    # 返回数组 x 中元素的总数
    return math.prod(x.shape)


def _is_numpy_namespace(xp):
    """Return True if xp is backed by NumPy."""
    # 如果 xp 的名称在 _NUMPY_NAMESPACE_NAMES 中，则返回 True
    return xp.__name__ in _NUMPY_NAMESPACE_NAMES


def _union1d(a, b, xp):
    # 如果 xp 是由 NumPy 支持的，则返回 a 和 b 的并集作为 NumPy 数组
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.union1d(a, b))
    assert a.ndim == b.ndim == 1
    # 否则，确保 a 和 b 是一维数组，并返回它们的唯一值的并集
    return xp.unique_values(xp.concat([xp.unique_values(a), xp.unique_values(b)]))


def isdtype(dtype, kind, *, xp):
    """Returns a boolean indicating whether a provided dtype is of type "kind".

    Included in the v2022.12 of the Array API spec.
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    """
    # 如果 kind 是一个元组，则检查 dtype 是否是元组中任一类型的数据类型
    if isinstance(kind, tuple):
        return any(_isdtype_single(dtype, k, xp=xp) for k in kind)
    else:
        # 否则，检查 dtype 是否是类型 kind
        return _isdtype_single(dtype, kind, xp=xp)


def _isdtype_single(dtype, kind, *, xp):
    # 检查给定的 dtype 是否是指定 kind 类型的数据类型
    # xp 参数表示使用的数组 API
    # 检查 `kind` 是否为字符串类型
    if isinstance(kind, str):
        # 如果 `kind` 是 "bool"，返回是否为布尔类型
        if kind == "bool":
            return dtype == xp.bool
        # 如果 `kind` 是 "signed integer"，返回是否为有符号整数类型
        elif kind == "signed integer":
            return dtype in {xp.int8, xp.int16, xp.int32, xp.int64}
        # 如果 `kind` 是 "unsigned integer"，返回是否为无符号整数类型
        elif kind == "unsigned integer":
            return dtype in {xp.uint8, xp.uint16, xp.uint32, xp.uint64}
        # 如果 `kind` 是 "integral"，返回是否为整数类型（有符号或无符号）
        elif kind == "integral":
            return any(
                _isdtype_single(dtype, k, xp=xp)
                for k in ("signed integer", "unsigned integer")
            )
        # 如果 `kind` 是 "real floating"，返回是否为浮点数类型
        elif kind == "real floating":
            return dtype in supported_float_dtypes(xp)
        # 如果 `kind` 是 "complex floating"，返回是否为复数类型
        elif kind == "complex floating":
            # 一些命名空间（如 cupy.array_api）可能没有复数类型
            complex_dtypes = set()
            if hasattr(xp, "complex64"):
                complex_dtypes.add(xp.complex64)
            if hasattr(xp, "complex128"):
                complex_dtypes.add(xp.complex128)
            return dtype in complex_dtypes
        # 如果 `kind` 是 "numeric"，返回是否为数值类型（整数、浮点数或复数）
        elif kind == "numeric":
            return any(
                _isdtype_single(dtype, k, xp=xp)
                for k in ("integral", "real floating", "complex floating")
            )
        else:
            # 如果 `kind` 不属于已知的数据类型类别，抛出 ValueError 异常
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        # 如果 `kind` 不是字符串类型，则直接比较 `dtype` 是否与 `kind` 相等
        return dtype == kind
# 返回支持的浮点数类型列表，考虑到 Array API 规范
def supported_float_dtypes(xp):
    """Supported floating point types for the namespace.

    Note: float16 is not officially part of the Array API spec at the
    time of writing but scikit-learn estimators and functions can choose
    to accept it when xp.float16 is defined.

    https://data-apis.org/array-api/latest/API_specification/data_types.html
    """
    # 检查 xp 是否具有 float16 属性，如果有则包括 float16，否则不包括
    if hasattr(xp, "float16"):
        return (xp.float64, xp.float32, xp.float16)
    else:
        return (xp.float64, xp.float32)


# 确保所有数组与参考数组具有相同的命名空间和设备
def ensure_common_namespace_device(reference, *arrays):
    """Ensure that all arrays use the same namespace and device as reference.

    If necessary the arrays are moved to the same namespace and device as
    the reference array.

    Parameters
    ----------
    reference : array
        Reference array.

    *arrays : array
        Arrays to check.

    Returns
    -------
    arrays : list
        Arrays with the same namespace and device as reference.
    """
    # 获取参考数组的命名空间和是否为 Array API
    xp, is_array_api = get_namespace(reference)

    if is_array_api:
        # 获取参考数组的设备信息
        device_ = device(reference)
        # 将所有数组移动到与参考数组相同的命名空间和设备
        return [xp.asarray(a, device=device_) for a in arrays]
    else:
        return arrays


class _ArrayAPIWrapper:
    """sklearn specific Array API compatibility wrapper

    This wrapper makes it possible for scikit-learn maintainers to
    deal with discrepancies between different implementations of the
    Python Array API standard and its evolution over time.

    The Python Array API standard specification:
    https://data-apis.org/array-api/latest/

    Documentation of the NumPy implementation:
    https://numpy.org/neps/nep-0047-array-api-standard.html
    """

    def __init__(self, array_namespace):
        self._namespace = array_namespace

    def __getattr__(self, name):
        return getattr(self._namespace, name)

    def __eq__(self, other):
        return self._namespace == other._namespace

    def isdtype(self, dtype, kind):
        return isdtype(dtype, kind, xp=self._namespace)


# 检查设备是否为 CPU，如果不是则引发 ValueError 异常
def _check_device_cpu(device):  # noqa
    if device not in {"cpu", None}:
        raise ValueError(f"Unsupported device for NumPy: {device!r}")


# 装饰器函数，确保函数接受的 device 参数为 "cpu" 或 None
def _accept_device_cpu(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        _check_device_cpu(kwargs.pop("device", None))
        return func(*args, **kwargs)

    return wrapped_func


class _NumPyAPIWrapper:
    """Array API compat wrapper for any numpy version

    NumPy < 2 does not implement the namespace. NumPy 2 and later should
    progressively implement more an more of the latest Array API spec but this
    is still work in progress at this time.

    This wrapper makes it possible to write code that uses the standard Array
    API while working with any version of NumPy supported by scikit-learn.

    See the `get_namespace()` public function for more details.
    """
    # TODO: once scikit-learn drops support for NumPy < 2, this class can be
    # removed, assuming Array API compliance of NumPy 2 is actually sufficient
    # for scikit-learn's needs.

    # Creation functions in spec:
    # https://data-apis.org/array-api/latest/API_specification/creation_functions.html
    # 定义创建函数集合，包括各种数组的创建方式
    _CREATION_FUNCS = {
        "arange",
        "empty",
        "empty_like",
        "eye",
        "full",
        "full_like",
        "linspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }
    # Data types in spec
    # https://data-apis.org/array-api/latest/API_specification/data_types.html
    # 定义数据类型集合，包括整数、浮点数和复数类型
    _DTYPES = {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        # XXX: float16 is not part of the Array API spec but exposed by
        # some namespaces.
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    }

    def __getattr__(self, name):
        # 获取 NumPy 模块中的属性，并返回对应的属性对象
        attr = getattr(numpy, name)

        # Support device kwargs and make sure they are on the CPU
        # 对于创建函数，支持设备参数，并确保数据位于 CPU 上
        if name in self._CREATION_FUNCS:
            return _accept_device_cpu(attr)

        # Convert to dtype objects
        # 将字符串类型的数据类型转换为 NumPy 的 dtype 对象
        if name in self._DTYPES:
            return numpy.dtype(attr)
        return attr

    @property
    def bool(self):
        # 返回 NumPy 中的布尔类型对象
        return numpy.bool_

    def astype(self, x, dtype, *, copy=True, casting="unsafe"):
        # astype 方法在顶层 NumPy 命名空间中未定义，这里调用其实现类型转换
        return x.astype(dtype, copy=copy, casting=casting)

    def asarray(self, x, *, dtype=None, device=None, copy=None):  # noqa
        _check_device_cpu(device)
        # 在 NumPy 命名空间中支持复制操作
        if copy is True:
            return numpy.array(x, copy=True, dtype=dtype)
        else:
            return numpy.asarray(x, dtype=dtype)

    def unique_inverse(self, x):
        # 返回数组 x 的唯一值和反向索引
        return numpy.unique(x, return_inverse=True)

    def unique_counts(self, x):
        # 返回数组 x 中每个唯一值的计数
        return numpy.unique(x, return_counts=True)

    def unique_values(self, x):
        # 返回数组 x 中的唯一值
        return numpy.unique(x)

    def unique_all(self, x):
        # 返回数组 x 的唯一值、索引、反向索引以及计数
        return numpy.unique(
            x, return_index=True, return_inverse=True, return_counts=True
        )

    def concat(self, arrays, *, axis=None):
        # 沿指定轴连接数组数组 arrays
        return numpy.concatenate(arrays, axis=axis)

    def reshape(self, x, shape, *, copy=None):
        """Gives a new shape to an array without changing its data.

        The Array API specification requires shape to be a tuple.
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.reshape.html
        """
        if not isinstance(shape, tuple):
            # 如果 shape 不是元组，则抛出类型错误异常
            raise TypeError(
                f"shape must be a tuple, got {shape!r} of type {type(shape)}"
            )

        if copy is True:
            # 如果需要复制数组，则执行复制操作
            x = x.copy()
        # 返回具有新形状的数组，但不更改其数据
        return numpy.reshape(x, shape)
    # 定义一个方法 isdtype，用于检查给定的数据类型 dtype 是否属于指定的种类 kind
    def isdtype(self, dtype, kind):
        # 调用外部的 isdtype 函数，传入参数 dtype、kind 和当前对象的引用 xp=self
        return isdtype(dtype, kind, xp=self)

    # 定义一个方法 pow，用于计算 x1 的 x2 次方
    def pow(self, x1, x2):
        # 调用 numpy 库的 power 函数，计算 x1 的 x2 次方并返回结果
        return numpy.power(x1, x2)
# 创建一个全局变量 _NUMPY_API_WRAPPER_INSTANCE，初始化为 _NumPyAPIWrapper 的实例
_NUMPY_API_WRAPPER_INSTANCE = _NumPyAPIWrapper()

# 定义一个函数 _remove_non_arrays，用于过滤掉 None 和特定类型的数组
def _remove_non_arrays(*arrays, remove_none=True, remove_types=(str,)):
    """Filter arrays to exclude None and/or specific types.

    Raise ValueError if no arrays are left after filtering.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    remove_none : bool, default=True
        Whether to ignore None objects passed in arrays.

    remove_types : tuple or list, default=(str,)
        Types to ignore in the arrays.

    Returns
    -------
    filtered_arrays : list
        List of arrays with None and types removed.
    """
    # 初始化空列表，用于存放过滤后的数组
    filtered_arrays = []
    # 将 remove_types 转换为元组
    remove_types = tuple(remove_types)
    # 遍历传入的 arrays
    for array in arrays:
        # 如果 remove_none=True 并且当前数组为 None，则跳过
        if remove_none and array is None:
            continue
        # 如果当前数组的类型在 remove_types 中，则跳过
        if isinstance(array, remove_types):
            continue
        # 将通过过滤条件的数组添加到 filtered_arrays 中
        filtered_arrays.append(array)

    # 如果 filtered_arrays 为空，则抛出 ValueError 异常
    if not filtered_arrays:
        raise ValueError(
            f"At least one input array expected after filtering with {remove_none=}, "
            f"remove_types=[{', '.join(t.__name__ for t in remove_types)}]. Got none. "
            f"Original types: [{', '.join(type(a).__name__ for a in arrays)}]."
        )
    # 返回过滤后的数组列表
    return filtered_arrays


# 定义一个函数 get_namespace，用于获取数组的命名空间
def get_namespace(*arrays, remove_none=True, remove_types=(str,), xp=None):
    """Get namespace of arrays.

    Introspect `arrays` arguments and return their common Array API compatible
    namespace object, if any.

    See: https://numpy.org/neps/nep-0047-array-api-standard.html

    If `arrays` are regular numpy arrays, an instance of the `_NumPyAPIWrapper`
    compatibility wrapper is returned instead.

    Namespace support is not enabled by default. To enabled it call:

      sklearn.set_config(array_api_dispatch=True)

    or:

      with sklearn.config_context(array_api_dispatch=True):
          # your code here

    Otherwise an instance of the `_NumPyAPIWrapper` compatibility wrapper is
    always returned irrespective of the fact that arrays implement the
    `__array_namespace__` protocol or not.

    Parameters
    ----------
    *arrays : array objects
        Array objects.

    remove_none : bool, default=True
        Whether to ignore None objects passed in arrays.

    remove_types : tuple or list, default=(str,)
        Types to ignore in the arrays.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    namespace : module
        Namespace shared by array objects. If any of the `arrays` are not arrays,
        the namespace defaults to NumPy.

    is_array_api_compliant : bool
        True if the arrays are containers that implement the Array API spec.
        Always False when array_api_dispatch=False.
    """
    # 获取当前配置中的 array_api_dispatch 参数
    array_api_dispatch = get_config()["array_api_dispatch"]
    # 如果 array_api_dispatch 为假值（如None、空列表等），执行以下逻辑
    if not array_api_dispatch:
        # 如果 xp 不为空，则返回 xp 和 False
        if xp is not None:
            return xp, False
        # 否则返回预定义的 _NUMPY_API_WRAPPER_INSTANCE 和 False
        else:
            return _NUMPY_API_WRAPPER_INSTANCE, False

    # 如果 xp 不为空，则返回 xp 和 True
    if xp is not None:
        return xp, True

    # 移除 arrays 中非数组的元素
    arrays = _remove_non_arrays(
        *arrays, remove_none=remove_none, remove_types=remove_types
    )

    # 检查 array_api_dispatch 是否符合预期，如果不符合会抛出异常
    _check_array_api_dispatch(array_api_dispatch)

    # array-api-compat 是 scikit-learn 配置 array_api_dispatch=True 时所需的依赖项，
    # 其导入应当受 _check_array_api_dispatch 保护，以便在缺失时显示信息丰富的错误消息。
    import array_api_compat

    # 获取 arrays 的命名空间和一个标志，表明是否符合数组 API 兼容性
    namespace, is_array_api_compliant = array_api_compat.get_namespace(*arrays), True

    # 如果命名空间的名称在 {"cupy.array_api"} 中，对其进行额外的包装以平滑实现之间的小差异
    if namespace.__name__ in {"cupy.array_api"}:
        namespace = _ArrayAPIWrapper(namespace)

    # 返回处理后的命名空间和数组 API 兼容性标志
    return namespace, is_array_api_compliant
# 将多个数组列表参数传递给 _remove_non_arrays 函数，移除非数组对象
array_list = _remove_non_arrays(
    *array_list, remove_none=remove_none, remove_types=remove_types
)

# 创建一个跳过移除参数的字典，用于 get_namespace 函数调用
skip_remove_kwargs = dict(remove_none=False, remove_types=[])

# 调用 get_namespace 函数获取命名空间 xp 和 is_array_api 标志
xp, is_array_api = get_namespace(*array_list, **skip_remove_kwargs)

# 如果 is_array_api 为 True，则返回 xp、is_array_api 和 device 函数处理后的结果
if is_array_api:
    return (
        xp,
        is_array_api,
        device(*array_list, **skip_remove_kwargs),
    )
# 如果 is_array_api 为 False，则返回 xp、False 和 None
else:
    return xp, False, None


def _expit(X, xp=None):
    # 调用 get_namespace 函数获取命名空间 xp 和 is_array_api 标志
    xp, _ = get_namespace(X, xp=xp)
    
    # 如果 xp 是 numpy 命名空间，则调用 numpy 和 special 模块计算 expit 函数
    if _is_numpy_namespace(xp):
        return xp.asarray(special.expit(numpy.asarray(X)))

    # 否则，使用 xp.exp 计算 expit 函数
    return 1.0 / (1.0 + xp.exp(-X))


def _add_to_diagonal(array, value, xp):
    # 对于 numpy.array_api 的兼容性问题，使用 xp.asarray 将 value 转换为指定的数据类型
    value = xp.asarray(value, dtype=array.dtype)
    
    # 如果 xp 是 numpy 命名空间，则转换 array 为 numpy 数组并在对角线上增加值
    if _is_numpy_namespace(xp):
        array_np = numpy.asarray(array)
        array_np.flat[:: array.shape[0] + 1] += value
        return xp.asarray(array_np)
    
    # 否则，如果 value 是一维数组，则遍历 array 的对角线并逐个增加对应的值
    elif value.ndim == 1:
        for i in range(array.shape[0]):
            array[i, i] += value[i]
    # 否则，如果 value 是标量，则遍历 array 的对角线并逐个增加标量值
    else:
        for i in range(array.shape[0]):
            array[i, i] += value


def _find_matching_floating_dtype(*arrays, xp):
    """在处理数组时找到合适的浮点数据类型。

    如果输入的数组中有任何浮点数组，则根据官方类型提升规则返回最高精度的浮点数据类型：
    https://data-apis.org/array-api/latest/API_specification/type_promotion.html

    如果没有浮点输入数组（例如全部为整数输入），则返回命名空间的默认浮点数据类型。
    """
    # 从输入数组中筛选出带有 dtype 属性的数组
    dtyped_arrays = [a for a in arrays if hasattr(a, "dtype")]
    
    # 筛选出实浮点类型的数组，并获取它们的 dtype
    floating_dtypes = [
        a.dtype for a in dtyped_arrays if xp.isdtype(a.dtype, "real floating")
    ]
    
    # 如果存在浮点类型的数组，则返回精度最高的浮点数据类型
    if floating_dtypes:
        return xp.result_type(*floating_dtypes)

    # 如果输入数组没有浮点 dtype，则它们必须全部是整数数组或 Python 标量容器，返回命名空间的默认浮点数据类型
    return xp.asarray(0.0).dtype


def _average(a, axis=None, weights=None, normalize=True, xp=None):
    """部分移植自 np.average 函数以支持 Array API。

    它尽力模仿 https://numpy.org/doc/stable/reference/generated/numpy.average.html 中描述的返回 dtype 规则，
    但仅限于 scikit-learn 中所需的常见情况。
    """
    # 调用 get_namespace_and_device 函数获取命名空间 xp 和 device_
    xp, _, device_ = get_namespace_and_device(a, weights)
    # 检查是否使用了 numpy 的命名空间
    if _is_numpy_namespace(xp):
        # 如果需要归一化，使用 numpy.average 计算加权平均值并返回数组
        if normalize:
            return xp.asarray(numpy.average(a, axis=axis, weights=weights))
        # 如果未指定 axis 但有 weights，则使用 numpy.dot 计算加权和并返回数组
        elif axis is None and weights is not None:
            return xp.asarray(numpy.dot(a, weights))

    # 将输入数组 a 转换为指定设备上的数组
    a = xp.asarray(a, device=device_)
    # 如果有 weights，则将 weights 也转换为指定设备上的数组
    if weights is not None:
        weights = xp.asarray(weights, device=device_)

    # 如果 weights 存在且其形状与 a 不匹配，则进行相应的类型错误或值错误检查和处理
    if weights is not None and a.shape != weights.shape:
        # 如果未指定 axis，则抛出类型错误异常，说明 a 和 weights 的形状不匹配
        if axis is None:
            raise TypeError(
                f"Axis must be specified when the shape of a {tuple(a.shape)} and "
                f"weights {tuple(weights.shape)} differ."
            )
        # 如果 weights 的维数不是 1，则抛出类型错误异常，期望 weights 是 1 维数组
        if weights.ndim != 1:
            raise TypeError(
                f"1D weights expected when a.shape={tuple(a.shape)} and "
                f"weights.shape={tuple(weights.shape)} differ."
            )
        # 如果 weights 的长度与 a 在指定 axis 上的长度不兼容，则抛出值错误异常
        if size(weights) != a.shape[axis]:
            raise ValueError(
                f"Length of weights {size(weights)} not compatible with "
                f" a.shape={tuple(a.shape)} and {axis=}."
            )

        # 如果 weights 是 1 维数组，则为了广播操作在其它维度添加单例维度
        shape = [1] * a.ndim
        shape[axis] = a.shape[axis]
        weights = xp.reshape(weights, shape)

    # 如果数组 a 的数据类型是复数浮点型，则抛出未实现错误，不支持复数浮点型数据
    if xp.isdtype(a.dtype, "complex floating"):
        raise NotImplementedError(
            "Complex floating point values are not supported by average."
        )
    # 如果 weights 存在且其数据类型是复数浮点型，则抛出未实现错误，不支持复数浮点型数据
    if weights is not None and xp.isdtype(weights.dtype, "complex floating"):
        raise NotImplementedError(
            "Complex floating point values are not supported by average."
        )

    # 查找适合 a 和 weights 的浮点数数据类型，用于转换
    output_dtype = _find_matching_floating_dtype(a, weights, xp=xp)
    a = xp.astype(a, output_dtype)

    # 如果没有 weights，则根据 normalize 返回均值或总和
    if weights is None:
        return (xp.mean if normalize else xp.sum)(a, axis=axis)

    # 将 weights 转换为适合的浮点数数据类型
    weights = xp.astype(weights, output_dtype)

    # 计算加权和并返回数组
    sum_ = xp.sum(xp.multiply(a, weights), axis=axis)

    # 如果不进行归一化，则直接返回加权和
    if not normalize:
        return sum_

    # 计算 weights 的总和
    scale = xp.sum(weights, axis=axis)
    # 如果 weights 的总和为零，则抛出零除异常，无法进行归一化
    if xp.any(scale == 0.0):
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")

    # 返回归一化后的加权和
    return sum_ / scale
# 计算数组 X 沿指定轴的最小值，支持 NaN 安全计算
def _nanmin(X, axis=None, xp=None):
    # 得到适当的计算命名空间 xp 和忽略符号
    xp, _ = get_namespace(X, xp=xp)
    # 如果命名空间为 NumPy，则使用 NumPy 的 nanmin 函数计算最小值
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmin(X, axis=axis))

    else:
        # 创建一个 NaN 掩码
        mask = xp.isnan(X)
        # 将所有 NaN 所在的切片用 +inf 替换，然后计算最小值
        X = xp.min(xp.where(mask, xp.asarray(+xp.inf, device=device(X)), X), axis=axis)
        # 如果有任何一个切片全为 NaN，则用 NaN 替换对应的最小值
        mask = xp.all(mask, axis=axis)
        if xp.any(mask):
            X = xp.where(mask, xp.asarray(xp.nan), X)
        return X


# 计算数组 X 沿指定轴的最大值，支持 NaN 安全计算
def _nanmax(X, axis=None, xp=None):
    # 得到适当的计算命名空间 xp 和忽略符号
    xp, _ = get_namespace(X, xp=xp)
    # 如果命名空间为 NumPy，则使用 NumPy 的 nanmax 函数计算最大值
    if _is_numpy_namespace(xp):
        return xp.asarray(numpy.nanmax(X, axis=axis))

    else:
        # 创建一个 NaN 掩码
        mask = xp.isnan(X)
        # 将所有 NaN 所在的切片用 -inf 替换，然后计算最大值
        X = xp.max(xp.where(mask, xp.asarray(-xp.inf, device=device(X)), X), axis=axis)
        # 如果有任何一个切片全为 NaN，则用 NaN 替换对应的最大值
        mask = xp.all(mask, axis=axis)
        if xp.any(mask):
            X = xp.where(mask, xp.asarray(xp.nan), X)
        return X


# 根据指定的命名空间将输入数组转换为数组，并支持 NumPy 的内存布局参数
def _asarray_with_order(
    array, dtype=None, order=None, copy=None, *, xp=None, device=None
):
    """Helper to support the order kwarg only for NumPy-backed arrays

    Memory layout parameter `order` is not exposed in the Array API standard,
    however some input validation code in scikit-learn needs to work both
    for classes and functions that will leverage Array API only operations
    and for code that inherently relies on NumPy backed data containers with
    specific memory layout constraints (e.g. our own Cython code). The
    purpose of this helper is to make it possible to share code for data
    container validation without memory copies for both downstream use cases:
    the `order` parameter is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.
    """
    xp, _ = get_namespace(array, xp=xp)
    # 如果命名空间为 NumPy，则使用 NumPy 的 asarray 或者 array 函数支持 order 参数
    if _is_numpy_namespace(xp):
        if copy is True:
            array = numpy.array(array, order=order, dtype=dtype)
        else:
            array = numpy.asarray(array, order=order, dtype=dtype)

        # 现在 array 是一个 NumPy ndarray。我们将其转换为与输入命名空间一致的数组容器。
        return xp.asarray(array)
    else:
        # 对于非 NumPy 命名空间，忽略 order 参数，直接转换为数组
        return xp.asarray(array, dtype=dtype, copy=copy, device=device)


# Array API 兼容版本的 np.ravel 函数
def _ravel(array, xp=None):
    """Array API compliant version of np.ravel.

    For non numpy namespaces, it just returns a flattened array, that might
    be or not be a copy.
    """
    xp, _ = get_namespace(array, xp=xp)
    # 如果命名空间为 NumPy，则使用 NumPy 的 ravel 函数将数组展平
    if _is_numpy_namespace(xp):
        array = numpy.asarray(array)
        return xp.asarray(numpy.ravel(array, order="C"))

    # 对于非 NumPy 命名空间，直接返回一个展平后的数组，可能是复制的或者是视图
    return xp.reshape(array, shape=(-1,))
def _convert_to_numpy(array, xp):
    """Convert X into a NumPy ndarray on the CPU."""
    xp_name = xp.__name__  # 获取 xp 对象的名称（通常是库的名称）

    if xp_name in {"array_api_compat.torch", "torch"}:  # 如果 xp 名称是 torch 或者 array_api_compat.torch
        return array.cpu().numpy()  # 使用 array 的 cpu() 方法将其转换为 NumPy 数组并返回
    elif xp_name == "cupy.array_api":  # 如果 xp 名称是 cupy.array_api
        return array._array.get()  # 使用 array 的 _array 属性的 get() 方法获取数据并返回
    elif xp_name in {"array_api_compat.cupy", "cupy"}:  # 如果 xp 名称是 cupy 或者 array_api_compat.cupy（覆盖率不涵盖）
        return array.get()  # 使用 array 的 get() 方法获取数据并返回

    return numpy.asarray(array)  # 默认情况下，使用 NumPy 的 asarray() 将 array 转换为 NumPy 数组并返回


def _estimator_with_converted_arrays(estimator, converter):
    """Create new estimator which converting all attributes that are arrays.

    The converter is called on all NumPy arrays and arrays that support the
    `DLPack interface <https://dmlc.github.io/dlpack/latest/>`__.

    Parameters
    ----------
    estimator : Estimator
        Estimator to convert

    converter : callable
        Callable that takes an array attribute and returns the converted array.

    Returns
    -------
    new_estimator : Estimator
        Convert estimator
    """
    from sklearn.base import clone  # 导入 sklearn.base 模块中的 clone 函数

    new_estimator = clone(estimator)  # 克隆给定的 estimator 对象
    for key, attribute in vars(estimator).items():  # 遍历 estimator 对象的所有属性和属性值
        if hasattr(attribute, "__dlpack__") or isinstance(attribute, numpy.ndarray):
            # 如果属性 attribute 具有 __dlpack__ 属性或者是 NumPy 数组
            attribute = converter(attribute)  # 使用 converter 函数对 attribute 进行转换
        setattr(new_estimator, key, attribute)  # 设置 new_estimator 的属性 key 为 attribute

    return new_estimator  # 返回转换后的新 estimator 对象


def _atol_for_type(dtype):
    """Return the absolute tolerance for a given numpy dtype."""
    return numpy.finfo(dtype).eps * 100  # 返回给定 NumPy dtype 的绝对容差


def indexing_dtype(xp):
    """Return a platform-specific integer dtype suitable for indexing.

    On 32-bit platforms, this will typically return int32 and int64 otherwise.

    Note: using dtype is recommended for indexing transient array
    datastructures. For long-lived arrays, such as the fitted attributes of
    estimators, it is instead recommended to use platform-independent int32 if
    we do not expect to index more 2B elements. Using fixed dtypes simplifies
    the handling of serialized models, e.g. to deploy a model fit on a 64-bit
    platform to a target 32-bit platform such as WASM/pyodide.
    """
    # Currently this is implemented with simple hack that assumes that
    # following "may be" statements in the Array API spec always hold:
    # > The default integer data type should be the same across platforms, but
    # > the default may vary depending on whether Python is 32-bit or 64-bit.
    # > The default array index data type may be int32 on 32-bit platforms, but
    # > the default should be int64 otherwise.
    # https://data-apis.org/array-api/latest/API_specification/data_types.html#default-data-types
    # TODO: once sufficiently adopted, we might want to instead rely on the
    # newer inspection API: https://github.com/data-apis/array-api/issues/640
    return xp.asarray(0).dtype  # 返回 xp 对象创建的数组的数据类型


def _searchsorted(xp, a, v, *, side="left", sorter=None):
    # Temporary workaround needed as long as searchsorted is not widely
    # adopted by implementers of the Array API spec. This is a quite
    # 检查 xp 对象是否具有 searchsorted 方法，如果有，则调用该方法进行搜索
    if hasattr(xp, "searchsorted"):
        return xp.searchsorted(a, v, side=side, sorter=sorter)

    # 将 a 转换为 NumPy 数组（如果尚未是），使用指定的 xp 库
    a_np = _convert_to_numpy(a, xp=xp)
    # 将 v 转换为 NumPy 数组（如果尚未是），使用指定的 xp 库
    v_np = _convert_to_numpy(v, xp=xp)
    # 在 a_np 中搜索 v_np 的插入点索引，返回索引数组
    indices = numpy.searchsorted(a_np, v_np, side=side, sorter=sorter)
    # 将索引数组转换为指定设备上的 xp 数组，并返回结果
    return xp.asarray(indices, device=device(a))
# 导入必要的模块或函数库后，定义了一个名为 `_setdiff1d` 的函数，用于计算两个数组的差集。
def _setdiff1d(ar1, ar2, xp, assume_unique=False):
    """Find the set difference of two arrays.

    Return the unique values in `ar1` that are not in `ar2`.
    """
    # 检查当前命名空间是否为 NumPy，如果是则调用 NumPy 提供的 `setdiff1d` 函数
    if _is_numpy_namespace(xp):
        return xp.asarray(
            numpy.setdiff1d(
                ar1=ar1,
                ar2=ar2,
                assume_unique=assume_unique,
            )
        )

    # 如果不是 NumPy 命名空间，根据 `assume_unique` 参数处理数组 `ar1` 和 `ar2`
    if assume_unique:
        ar1 = xp.reshape(ar1, (-1,))
    else:
        ar1 = xp.unique_values(ar1)  # 获取唯一值
        ar2 = xp.unique_values(ar2)  # 获取唯一值
    # 调用 `_in1d` 函数获取 `ar1` 中存在于 `ar2` 中但不存在的值
    return ar1[_in1d(ar1=ar1, ar2=ar2, xp=xp, assume_unique=True, invert=True)]


# 导入必要的模块或函数库后，定义了一个名为 `_isin` 的函数，用于检查元素是否在测试元素中。
def _isin(element, test_elements, xp, assume_unique=False, invert=False):
    """Calculates ``element in test_elements``, broadcasting over `element`
    only.

    Returns a boolean array of the same shape as `element` that is True
    where an element of `element` is in `test_elements` and False otherwise.
    """
    # 检查当前命名空间是否为 NumPy，如果是则调用 NumPy 提供的 `isin` 函数
    if _is_numpy_namespace(xp):
        return xp.asarray(
            numpy.isin(
                element=element,
                test_elements=test_elements,
                assume_unique=assume_unique,
                invert=invert,
            )
        )

    # 如果不是 NumPy 命名空间，处理元素 `element` 和测试元素 `test_elements`
    original_element_shape = element.shape
    element = xp.reshape(element, (-1,))
    test_elements = xp.reshape(test_elements, (-1,))
    # 调用 `_in1d` 函数检查 `element` 中的元素是否存在于 `test_elements` 中
    return xp.reshape(
        _in1d(
            ar1=element,
            ar2=test_elements,
            xp=xp,
            assume_unique=assume_unique,
            invert=invert,
        ),
        original_element_shape,
    )


# 注意：这是 `_isin` 和 `_setdiff1d` 函数的辅助函数，不应直接调用。
def _in1d(ar1, ar2, xp, assume_unique=False, invert=False):
    """Checks whether each element of an array is also present in a
    second array.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    This function has been adapted using the original implementation
    present in numpy:
    https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/arraysetops.py#L524-L758
    """
    xp, _ = get_namespace(ar1, ar2, xp=xp)

    # 这段代码用于提高性能
    if ar2.shape[0] < 10 * ar1.shape[0] ** 0.145:
        if invert:
            mask = xp.ones(ar1.shape[0], dtype=xp.bool, device=device(ar1))
            for a in ar2:
                mask &= ar1 != a
        else:
            mask = xp.zeros(ar1.shape[0], dtype=xp.bool, device=device(ar1))
            for a in ar2:
                mask |= ar1 == a
        return mask

    # 如果不假设数组唯一性，则处理数组 `ar1` 和 `ar2`，并调用相关的 NumPy 函数
    if not assume_unique:
        ar1, rev_idx = xp.unique_inverse(ar1)
        ar2 = xp.unique_values(ar2)

    ar = xp.concat((ar1, ar2))  # 连接数组 `ar1` 和 `ar2`
    device_ = device(ar)
    # 需要稳定排序
    order = xp.argsort(ar, stable=True)
    reverse_order = xp.argsort(order, stable=True)
    sar = xp.take(ar, order, axis=0)
    if invert:
        bool_ar = sar[1:] != sar[:-1]
    # 否则，比较数组 sar 的相邻元素是否相等，返回布尔数组
    bool_ar = sar[1:] == sar[:-1]
    
# 使用 xp.concat() 将 bool_ar 和 invert 转换为同一设备上的张量，存储在 flag 中
flag = xp.concat((bool_ar, xp.asarray([invert], device=device_)))

# 按照 reverse_order 数组的顺序，从 flag 中取出元素组成 ret
ret = xp.take(flag, reverse_order, axis=0)

# 如果 assume_unique 为真，返回 ret 的前 ar1 数组的长度部分
if assume_unique:
    return ret[: ar1.shape[0]]
# 否则，按照 rev_idx 数组的顺序，从 ret 中取出元素返回
else:
    return xp.take(ret, rev_idx, axis=0)
```