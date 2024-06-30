# `D:\src\scipysrc\scipy\scipy\_lib\_util.py`

```
import re  # 导入 re 模块，用于正则表达式操作
from contextlib import contextmanager  # 导入 contextmanager 上下文管理器
import functools  # 导入 functools 模块，用于高阶函数操作
import operator  # 导入 operator 模块，提供了许多内置操作符的函数实现
import warnings  # 导入 warnings 模块，用于警告管理
import numbers  # 导入 numbers 模块，处理数字抽象基类
from collections import namedtuple  # 导入 namedtuple 类型，用于创建命名元组
import inspect  # 导入 inspect 模块，用于检查类和函数的定义和调用
import math  # 导入 math 模块，提供数学函数
from typing import (  # 导入 typing 模块，用于类型提示
    Optional,
    Union,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np  # 导入 NumPy 库，重命名为 np
from scipy._lib._array_api import array_namespace, is_numpy, size as xp_size  # 导入 SciPy 库的数组 API 相关内容

AxisError: type[Exception]  # 定义 AxisError 类型为 Exception 类型的子类
ComplexWarning: type[Warning]  # 定义 ComplexWarning 类型为 Warning 类型
VisibleDeprecationWarning: type[Warning]  # 定义 VisibleDeprecationWarning 类型为 Warning 类型

# 根据 NumPy 版本选择性地导入异常类型
if np.lib.NumpyVersion(np.__version__) >= '1.25.0':
    from numpy.exceptions import (
        AxisError, ComplexWarning, VisibleDeprecationWarning,
        DTypePromotionError
    )
else:
    from numpy import (  # type: ignore[attr-defined, no-redef]
        AxisError, ComplexWarning, VisibleDeprecationWarning  # noqa: F401
    )
    DTypePromotionError = TypeError  # type: ignore

np_long: type  # 声明 np_long 变量，类型为 type
np_ulong: type  # 声明 np_ulong 变量，类型为 type

# 根据 NumPy 版本选择性地定义 np_long 和 np_ulong 的类型
if np.lib.NumpyVersion(np.__version__) >= "2.0.0.dev0":
    try:
        with warnings.catch_warnings():  # 使用 warnings 模块捕获警告
            warnings.filterwarnings(
                "ignore",
                r".*In the future `np\.long` will be defined as.*",
                FutureWarning,
            )
            np_long = np.long  # type: ignore[attr-defined]
            np_ulong = np.ulong  # type: ignore[attr-defined]
    except AttributeError:
        np_long = np.int_
        np_ulong = np.uint
else:
    np_long = np.int_
    np_ulong = np.uint

IntNumber = Union[int, np.integer]  # 定义 IntNumber 类型为 int 或 np.integer 的联合类型
DecimalNumber = Union[float, np.floating, np.integer]  # 定义 DecimalNumber 类型为 float、np.floating 或 np.integer 的联合类型

copy_if_needed: Optional[bool]  # 声明 copy_if_needed 变量，类型为可选的布尔值

# 根据 NumPy 版本选择性地定义 copy_if_needed 变量
if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    copy_if_needed = None
elif np.lib.NumpyVersion(np.__version__) < "1.28.0":
    copy_if_needed = False
else:
    # 处理 2.0.0 dev 版本中可能存在的 copy 参数
    try:
        np.array([1]).__array__(copy=None)  # type: ignore[call-overload]
        copy_if_needed = None
    except TypeError:
        copy_if_needed = False

# 为了向后兼容，如果存在 TYPE_CHECKING，定义 SeedType 和 GeneratorType 类型
if TYPE_CHECKING:
    SeedType = Optional[Union[IntNumber, np.random.Generator,
                              np.random.RandomState]]
    GeneratorType = TypeVar("GeneratorType", bound=Union[np.random.Generator,
                                                         np.random.RandomState])

try:
    from numpy.random import Generator as Generator  # 尝试导入 Generator 类
except ImportError:
    class Generator:  # type: ignore[no-redef]
        pass

def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """Return elements chosen from two possibilities depending on a condition

    Equivalent to ``f(*arrays) if cond else fillvalue`` performed elementwise.

    Parameters
    ----------
    cond : array
        The condition (expressed as a boolean array).
    arrays : tuple of array
        Arguments to `f` (and `f2`). Must be broadcastable with `cond`.
    f : callable
        Where `cond` is True, output will be ``f(arr1[cond], arr2[cond], ...)``

    """
    fillvalue : object
        如果提供了，用来填充输出数组中 `cond` 不为 True 的位置。
    f2 : callable
        如果提供了，输出将会是 ``f2(arr1[cond], arr2[cond], ...)``，其中 `cond` 不为 True。

    Returns
    -------
    out : array
        一个数组，其元素来自于 `f` 的输出，其中 `cond` 为 True；或者来自于 `fillvalue`（或 `f2` 的输出）的元素，其他位置。返回的数组的数据类型由 `f` 的输出和 `fillvalue`（或 `f2` 的输出）决定。

    Notes
    -----
    ``xp.where(cond, x, fillvalue)`` 要求显式地形成 `x`，即使 `cond` 为 False。这个函数只在 `cond` 为 True 时评估 ``f(arr1[cond], arr2[cond], ...)``。

    Examples
    --------
    >>> import numpy as np
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    """
    xp = array_namespace(cond, *arrays)  # 使用 array_namespace 函数命名空间来处理条件和数组

    if (f2 is fillvalue is None) or (f2 is not None and fillvalue is not None):
        raise ValueError("Exactly one of `fillvalue` or `f2` must be given.")

    args = xp.broadcast_arrays(cond, *arrays)  # 对条件和所有数组进行广播处理
    bool_dtype = xp.asarray([True]).dtype  # numpy 1.xx 不支持 `bool`，这里用来获取布尔类型的 dtype
    cond, arrays = xp.astype(args[0], bool_dtype, copy=False), args[1:]  # 将条件转换为布尔类型

    temp1 = xp.asarray(f(*(arr[cond] for arr in arrays)))  # 计算满足条件的数组的函数 `f` 的输出

    if f2 is None:
        # 如果 `fillvalue` 是 Python 标量并且我们转换为 `xp.asarray`，它会获得 `xp` 的默认 `int` 或 `float` 类型，因此 `result_type` 可能会错。
        # `result_type` 应该处理混合的数组/Python标量；当它这样做时，移除这个特殊逻辑。
        if type(fillvalue) in {bool, int, float, complex}:
            with np.errstate(invalid='ignore'):
                dtype = (temp1 * fillvalue).dtype
        else:
           dtype = xp.result_type(temp1.dtype, fillvalue)
        out = xp.full(cond.shape, dtype=dtype,
                      fill_value=xp.asarray(fillvalue, dtype=dtype))  # 根据 `fillvalue` 的数据类型创建填充数组
    else:
        ncond = ~cond  # 取反条件
        temp2 = xp.asarray(f2(*(arr[ncond] for arr in arrays)))  # 计算不满足条件的数组的函数 `f2` 的输出
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty(cond.shape, dtype=dtype)  # 根据 `temp1` 和 `temp2` 的数据类型创建空数组
        out[ncond] = temp2  # 将不满足条件的部分赋值给 `out`

    out[cond] = temp1  # 将满足条件的部分赋值给 `out`

    return out  # 返回计算结果的数组
# 为了模仿 `np.select(condlist, choicelist)` 函数。
def _lazyselect(condlist, choicelist, arrays, default=0):
    """
    Mimic `np.select(condlist, choicelist)`.

    Notice, it assumes that all `arrays` are of the same shape or can be
    broadcasted together.

    All functions in `choicelist` must accept array arguments in the order
    given in `arrays` and must return an array of the same shape as broadcasted
    `arrays`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> np.select([x <3, x > 3], [x**2, x**3], default=0)
    array([  0,   1,   4,   0,  64, 125])

    >>> _lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,))
    array([   0.,    1.,    4.,   0.,   64.,  125.])

    >>> a = -np.ones_like(x)
    >>> _lazyselect([x < 3, x > 3],
    ...             [lambda x, a: x**2, lambda x, a: a * x**3],
    ...             (x, a), default=np.nan)
    array([   0.,    1.,    4.,   nan,  -64., -125.])

    """
    # 广播输入的数组，使它们具有相同的形状
    arrays = np.broadcast_arrays(*arrays)
    # 确定数组的最小类型码
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    # 创建一个用默认值填充的输出数组，形状与广播后的数组相同
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    # 对每对条件和函数进行迭代
    for func, cond in zip(choicelist, condlist):
        # 如果条件全部为 False，则继续下一个迭代
        if np.all(cond is False):
            continue
        # 对条件和第一个数组进行广播
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        # 从数组中提取满足条件的元素组成元组
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        # 使用函数处理提取的元组，并将结果放置到输出数组中
        np.place(out, cond, func(*temp))
    # 返回最终的输出数组
    return out


# 分配一个具有对齐内存的新 ndarray
def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """Allocate a new ndarray with aligned memory.

    Primary use case for this currently is working around a f2py issue
    in NumPy 1.9.1, where dtype.alignment is such that np.zeros() does
    not necessarily create arrays aligned up to it.

    """
    # 将 dtype 转换为 numpy 的 dtype 对象
    dtype = np.dtype(dtype)
    # 如果没有指定对齐方式，使用 dtype 的对齐值
    if align is None:
        align = dtype.alignment
    # 如果 shape 不是一个长度对象，将其转换为元组
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    # 计算所需内存块的总大小
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    # 分配一个足够大的缓冲区，确保能够对齐
    buf = np.empty(size + align + 1, np.uint8)
    # 计算偏移量以对齐数据
    offset = buf.__array_interface__['data'][0] % align
    if offset != 0:
        offset = align - offset
    # 切片操作以确保数据对齐，并分配正确的大小
    buf = buf[offset:offset+size+1][:-1]
    # 创建一个新的 ndarray 对象，使用指定的 dtype、内存缓冲区和排序顺序
    data = np.ndarray(shape, dtype, buf, order=order)
    # 将数组填充为零
    data.fill(0)
    # 返回对齐后的数组
    return data


# 返回与输入数组等效的数组，如果输入数组是更大数组的视图，则将其内容复制到新分配的数组中
def _prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """
    # 如果数组有基础数组并且大小小于基础数组的一半，复制其内容到新数组
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    # 否则返回原数组
    return array


# 计算阶乘并返回为浮点数，当结果对于双精度浮点数来说太大时返回无穷大
def float_factorial(n: int) -> float:
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    # 使用 math.factorial 计算阶乘，如果 n 小于 171，返回浮点数，否则返回无穷大
    return float(math.factorial(n)) if n < 171 else np.inf
# change this to scipy.stats._qmc.check_random_state once numpy 1.16 is dropped
# 定义一个函数，用于检查和处理随机数种子，以生成一个合适的随机数生成器实例
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    # 如果 seed 是 None 或者是 np.random，则返回 np.random.mtrand._rand 单例对象
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    # 如果 seed 是整数类型（Integral 或 np.integer），则创建一个新的 RandomState 实例并使用 seed 初始化
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    # 如果 seed 已经是 Generator 或 RandomState 实例，则直接返回 seed
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed

    # 如果 seed 不符合上述条件，则抛出 ValueError 异常
    raise ValueError(f"'{seed}' cannot be used to seed a numpy.random.RandomState"
                     " instance")


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.

    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.

    Returns
    -------
    ret : ndarray
        The converted validated array.

    """
    # 如果 sparse_ok 参数为 False，并且输入数组是稀疏矩阵，则抛出 ValueError 异常
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    # 如果 mask_ok 参数为 False，并且输入数组是掩码数组，则抛出 ValueError 异常
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    # 根据 check_finite 参数选择相应的转换函数来处理输入数组 a
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    # 如果对象不符合预期，抛出值错误异常
    if not objects_ok:
        # 检查数组元素的数据类型是否为对象类型
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    
    # 如果需要将数组转换为浮点数类型
    if as_inexact:
        # 检查数组的数据类型是否为浮点数或其子类型
        if not np.issubdtype(a.dtype, np.inexact):
            # 将数组转换为指定的浮点数类型（默认为 np.float64）
            a = toarray(a, dtype=np.float64)
    
    # 返回处理后的数组
    return a
def _validate_int(k, name, minimum=None):
    """
    Validate a scalar integer.

    This function validates whether the input `k` is a scalar integer. It uses
    `operator.index` to ensure the value is an integer (e.g., TypeError is raised
    for k=2.0).

    Parameters
    ----------
    k : int
        The value to be validated.
    name : str
        The name of the parameter.
    minimum : int, optional
        An optional lower bound.

    Returns
    -------
    int
        The validated integer value `k`.

    Raises
    ------
    TypeError
        If `k` is not an integer.
    ValueError
        If `minimum` is provided and `k` is less than `minimum`.
    """
    try:
        # Ensure `k` is an integer using `operator.index`
        k = operator.index(k)
    except TypeError:
        raise TypeError(f'{name} must be an integer.') from None
    if minimum is not None and k < minimum:
        raise ValueError(f'{name} must be an integer not less '
                         f'than {minimum}') from None
    return k


# Add a replacement for inspect.getfullargspec()/
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846.

# Note an inconsistency between inspect.getfullargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not*
# list `self` as a first argument, while the former *does*.
# Hence, cook up a common ground replacement: `getfullargspec_no_self` which
# mimics `inspect.getfullargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getfullargspec or a bright and shiny .signature.

FullArgSpec = namedtuple('FullArgSpec',
                         ['args', 'varargs', 'varkw', 'defaults',
                          'kwonlyargs', 'kwonlydefaults', 'annotations'])


def getfullargspec_no_self(func):
    """
    inspect.getfullargspec replacement using inspect.signature.

    If `func` is a bound method, exclude the 'self' parameter.

    Parameters
    ----------
    func : callable
        A callable to inspect

    Returns
    -------
    FullArgSpec
        Full argument specification including args, varargs, varkw, defaults,
        kwonlyargs, kwonlydefaults, and annotations.

    Notes
    -----
    If `func`'s first parameter is 'self', it is not included in `args`.
    This maintains consistency with inspect.getargspec() in Python 2.x and
    inspect.signature() in Python 3.x.
    """
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY]
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = tuple(
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
            p.default is not p.empty)
    )
    # 如果 kwdefaults 不为空，则返回 kwdefaults；否则返回 None
    kwonlyargs = [
        p.name for p in sig.parameters.values()  # 遍历参数对象的值，取参数名，仅限 KEYWORD_ONLY 类型的参数
        if p.kind == inspect.Parameter.KEYWORD_ONLY  # 仅选择参数类型为 KEYWORD_ONLY 的参数
    ]
    # 构建一个字典，包含所有 KEYWORD_ONLY 类型参数的默认值，如果默认值不为空的话
    kwdefaults = {p.name: p.default for p in sig.parameters.values()
                  if p.kind == inspect.Parameter.KEYWORD_ONLY and
                  p.default is not p.empty}
    # 构建一个字典，包含所有参数的注解，但排除那些注解为空的情况
    annotations = {p.name: p.annotation for p in sig.parameters.values()
                   if p.annotation is not p.empty}
    # 返回一个 FullArgSpec 对象，其中包含了所有参数的详细规范信息
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                       kwdefaults or None, annotations)
class _FunctionWrapper:
    """
    Object to wrap user's function, allowing picklability
    """
    # 初始化函数，接收用户定义的函数对象和参数列表作为参数
    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    # 调用函数，传入参数 x，调用被包装的函数 self.f，并传入参数列表 self.args
    def __call__(self, x):
        return self.f(x, *self.args)


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.

    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilize all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelization.
    """
    # 初始化函数，根据传入的参数 pool 进行并行化封装
    def __init__(self, pool=1):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        # 如果 pool 是可调用对象，则使用该对象作为并行化函数
        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            from multiprocessing import Pool
            # 如果 pool 是整数 -1，则使用所有可用 CPU
            if int(pool) == -1:
                self.pool = Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            # 如果 pool 是整数 1，则不使用并行处理
            elif int(pool) == 1:
                pass
            # 如果 pool 是大于 1 的整数，则使用指定数量的进程池
            elif int(pool) > 1:
                self.pool = Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True
            else:
                # 如果 pool 不满足上述条件，抛出运行时错误
                raise RuntimeError("Number of workers specified must be -1,"
                                   " an int >= 1, or an object with a 'map' "
                                   "method")

    # 进入上下文管理器时调用，返回自身对象
    def __enter__(self):
        return self

    # 终止并行处理池
    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    # 等待所有进程池任务完成
    def join(self):
        if self._own_pool:
            self.pool.join()

    # 关闭进程池
    def close(self):
        if self._own_pool:
            self.pool.close()

    # 退出上下文管理器时调用，关闭并终止进程池
    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    # 调用对象时，接受一个函数和一个可迭代对象作为参数，使用并行化函数执行映射操作
    def __call__(self, func, iterable):
        try:
            return self._mapfunc(func, iterable)
        except TypeError as e:
            # 如果参数错误，抛出类型错误
            raise TypeError("The map-like callable must be of the"
                            " form f(func, iterable)") from e


def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    """
    """
    Return random integers from a discrete uniform distribution, with an option
    to include or exclude the high endpoint.
    
    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, uses np.random.RandomState singleton.
    low : int or array-like of ints
        Lowest integers to draw from (0 if high=None).
    high : int or array-like of ints
        If provided, one above the largest integer to draw.
    size : array-like of ints, optional
        Shape of output array. If not provided, returns a single integer.
    dtype : {str, dtype}, optional
        Desired data type of the output. Default is 'int64'.
    endpoint : bool, optional
        Whether to include high endpoint in samples. Defaults to False.
    
    Returns
    -------
    out: int or ndarray of ints
        Array of random integers or a single integer if size not provided.
    """
    if isinstance(gen, Generator):
        # Use Generator's integers method for random integers
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # Use default RandomState from np.random
            gen = np.random.mtrand._rand
        if endpoint:
            # Inclusive of endpoint
            # Ensure low and high are not modified in place, especially if they are arrays
            if high is None:
                # Special case when high is None
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                # General case when high is provided
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)
    
        # Exclusive of endpoint
        return gen.randint(low, high=high, size=size, dtype=dtype)
@contextmanager
def _fixed_default_rng(seed=1638083107694713882823079058616272161):
    """Context manager that temporarily fixes the seed of np.random.default_rng."""
    # Save the original np.random.default_rng function
    orig_fun = np.random.default_rng
    # Redefine np.random.default_rng to always use the specified seed
    np.random.default_rng = lambda seed=seed: orig_fun(seed)
    try:
        yield  # Yield control back to the caller
    finally:
        # Restore the original np.random.default_rng function
        np.random.default_rng = orig_fun


def _rng_html_rewrite(func):
    """Decorator function to rewrite HTML rendering of np.random.default_rng.

    This function decorates the output of SphinxDocString._str_examples, modifying
    instances of np.random.default_rng(seed) to np.random.default_rng().
    """
    # Regular expression pattern to match hexadecimal or number seed
    pattern = re.compile(r'np.random.default_rng\((0x[0-9A-F]+|\d+)\)', re.I)

    def _wrapped(*args, **kwargs):
        # Call the original decorated function
        res = func(*args, **kwargs)
        # Modify each line of the result to replace seed values with ()
        lines = [
            re.sub(pattern, 'np.random.default_rng()', line)
            for line in res
        ]
        return lines

    return _wrapped


def _argmin(a, keepdims=False, axis=None):
    """
    Return indices of the minimum values along an axis.

    This function extends np.argmin by optionally keeping dimensions and handling
    specific cases, as described in the issue link provided.
    """
    # Call np.argmin to get indices of minimum values
    res = np.argmin(a, axis=axis)
    # If keepdims is True and axis is not None, expand dimensions of the result
    if keepdims and axis is not None:
        res = np.expand_dims(res, axis=axis)
    return res


def _first_nonnan(a, axis):
    """
    Return the first non-nan value along the given axis.

    If a slice is all nan, nan is returned for that slice.
    """
    # Get indices of nan values along the axis
    k = _argmin(np.isnan(a), axis=axis, keepdims=True)
    # Return the elements of a corresponding to the minimum indices
    return np.take_along_axis(a, k, axis=axis)


def _nan_allsame(a, axis, keepdims=False):
    """
    Determine if the values along an axis are all the same, ignoring nan values.

    This function checks if all non-nan values along the specified axis are identical.
    """
    # If axis length is 0, return True (no different values)
    # For all other cases, use np.isnan to find nan values and check uniformity
    pass  # Function definition not completed in the provided snippet
    # 如果轴参数为 None，则检查数组是否为空，若是则返回 True
    if axis is None:
        if a.size == 0:
            return True
        # 将数组展平以便处理
        a = a.ravel()
        axis = 0
    else:
        # 获取数组的形状
        shp = a.shape
        # 如果指定轴的长度为 0，则返回填充值为 True 的数组，保持原形状
        if shp[axis] == 0:
            shp = shp[:axis] + (1,) * keepdims + shp[axis + 1:]
            return np.full(shp, fill_value=True, dtype=bool)
    
    # 获取数组中每个子数组的第一个非 NaN 元素
    a0 = _first_nonnan(a, axis=axis)
    # 检查数组中每个元素是否与其对应的第一个非 NaN 元素相等或者是否为 NaN，返回结果作为布尔值数组
    return ((a0 == a) | np.isnan(a)).all(axis=axis, keepdims=keepdims)
# 检查数组 `a` 中是否包含 NaN 值，并根据指定的策略处理
def _contains_nan(a, nan_policy='propagate', policies=None, *,
                  xp_omit_okay=False, xp=None):
    # 如果未指定 `xp`，则根据数组 `a` 的类型确定使用的数组命名空间
    if xp is None:
        xp = array_namespace(a)
    # 检查是否不是 NumPy 数组
    not_numpy = not is_numpy(xp)

    # 如果策略列表未提供，则默认包含 'propagate', 'raise', 'omit' 三种策略
    if policies is None:
        policies = {'propagate', 'raise', 'omit'}
    # 如果指定的 nan_policy 不在策略列表中，则引发 ValueError 异常
    if nan_policy not in policies:
        raise ValueError(f"nan_policy must be one of {set(policies)}.")

    # 判断数组 `a` 是否属于浮点数或复数类型
    inexact = (xp.isdtype(a.dtype, "real floating")
               or xp.isdtype(a.dtype, "complex floating"))
    # 如果数组 `a` 的尺寸为 0，则不包含 NaN 值
    if xp_size(a) == 0:
        contains_nan = False
    # 如果属于浮点数或复数类型，则使用快速方法判断数组中是否存在 NaN 值
    elif inexact:
        # 比使用 xp.any(xp.isnan(a)) 更快且内存消耗更少的方法
        contains_nan = xp.isnan(xp.max(a))
    # 如果是 NumPy 数组且数据类型为对象，假设不包含 NaN 值
    elif is_numpy(xp) and np.issubdtype(a.dtype, object):
        contains_nan = False
        # 遍历数组中的每个元素，如果有数值类型且为 NaN，则设置 contains_nan 为 True
        for el in a.ravel():
            # 对于非数值元素，isnan 不起作用
            if np.issubdtype(type(el), np.number) and np.isnan(el):
                contains_nan = True
                break
    else:
        # 只有对象数组和浮点数数组可能包含 NaN 值
        contains_nan = False

    # 如果发现 NaN 值且策略为 'raise'，则引发 ValueError 异常
    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    # 如果不允许 `xp_omit_okay`，且不是 NumPy 数组且包含 NaN 值且策略为 'omit'，则引发 ValueError 异常
    if not xp_omit_okay and not_numpy and contains_nan and nan_policy=='omit':
        message = "`nan_policy='omit' is incompatible with non-NumPy arrays."
        raise ValueError(message)

    # 返回包含是否包含 NaN 值和所使用的策略的元组
    return contains_nan, nan_policy


def _rename_parameter(old_name, new_name, dep_version=None):
    """
    Generate decorator for backward-compatible keyword renaming.

    Apply the decorator generated by `_rename_parameter` to functions with a
    recently renamed parameter to maintain backward-compatibility.

    After decoration, the function behaves as follows:
    If only the new parameter is passed into the function, behave as usual.
    If only the old parameter is passed into the function (as a keyword), raise
    a DeprecationWarning if `dep_version` is provided, and behave as usual
    otherwise.
    If both old and new parameters are passed into the function, raise a
    DeprecationWarning if `dep_version` is provided, and raise the appropriate
    TypeError (function got multiple values for argument).

    Parameters
    ----------
    old_name : str
        Old name of parameter
    new_name : str
        New name of parameter
    dep_version : str, optional
        Version of SciPy in which old parameter was deprecated in the format
        'X.Y.Z'. If supplied, the deprecation message will indicate that
        support for the old parameter will be removed in version 'X.Y+2.Z'

    Notes
    -----
    """
    """
    定义一个装饰器函数，用于处理函数参数的重命名和弃用警告。

    Parameters:
    - fun: 要装饰的函数对象

    Returns:
    - wrapper: 装饰后的函数对象

    Raises:
    - TypeError: 如果函数同时使用了旧参数名和新参数名，则抛出类型错误

    Warnings:
    - DeprecationWarning: 如果使用了被弃用的旧参数名，则发出弃用警告
    """
    def decorator(fun):
        # 定义装饰器内部的包装函数
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            # 检查是否传入了旧参数名
            if old_name in kwargs:
                # 如果需要向后兼容，生成未来版本的版本号
                if dep_version:
                    end_version = dep_version.split('.')
                    end_version[1] = str(int(end_version[1]) + 2)
                    end_version = '.'.join(end_version)
                    # 生成弃用警告消息
                    message = (f"Use of keyword argument `{old_name}` is "
                               f"deprecated and replaced by `{new_name}`.  "
                               f"Support for `{old_name}` will be removed "
                               f"in SciPy {end_version}.")
                    # 发出警告
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                # 检查是否同时使用了新参数名
                if new_name in kwargs:
                    # 如果同时使用了新旧参数名，抛出类型错误
                    message = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                    raise TypeError(message)
                # 将旧参数名对应的值移动到新参数名下
                kwargs[new_name] = kwargs.pop(old_name)
            # 调用原始函数，传递参数和关键字参数
            return fun(*args, **kwargs)
        # 返回装饰后的包装函数
        return wrapper
    # 返回装饰器函数
    return decorator
# 从给定的 RNG（随机数生成器）生成 `n_children` 个独立的 RNG
def _rng_spawn(rng, n_children):
    # 获取父 RNG 的底层比特生成器
    bg = rng._bit_generator
    # 获取比特生成器的种子序列
    ss = bg._seed_seq
    # 使用种子序列生成 `n_children` 个新的随机数生成器，每个使用相同类型的比特生成器
    child_rngs = [np.random.Generator(type(bg)(child_ss))
                  for child_ss in ss.spawn(n_children)]
    return child_rngs


def _get_nan(*data, xp=None):
    # 如果 xp 为 None，则使用默认的数组命名空间函数 array_namespace(*data)，否则使用给定的 xp
    xp = array_namespace(*data) if xp is None else xp
    # 将所有数据项转换为 xp 的数组表示
    data = [xp.asarray(item) for item in data]
    try:
        # 尝试获取数据项的最小浮点数类型，至少为 float 的类型
        min_float = getattr(xp, 'float16', xp.float32)
        # 确定数据项的结果类型，必须至少是一个浮点数
        dtype = xp.result_type(*data, min_float)
    except DTypePromotionError:
        # 如果类型提升错误，则回退到 float64
        dtype = xp.float64
    # 返回一个包含 NaN 值的数组，类型为 dtype
    return xp.asarray(xp.nan, dtype=dtype)[()]


def normalize_axis_index(axis, ndim):
    # 检查 `axis` 是否在正确的范围内，并将其归一化
    if axis < -ndim or axis >= ndim:
        # 如果 `axis` 超出了数组维度 `ndim` 的范围，抛出 AxisError 异常
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)

    if axis < 0:
        # 如果 `axis` 是负数，则将其转换为对应的正数索引
        axis = axis + ndim
    return axis


def _call_callback_maybe_halt(callback, res):
    """调用包装后的回调函数；如果算法应该停止，则返回 True。

    Parameters
    ----------
    callback : callable or None
        使用 `_wrap_callback` 包装过的用户提供的回调函数
    res : OptimizeResult
        当前迭代的信息

    Returns
    -------
    halt : bool
        如果最小化应该停止，则为 True

    """
    if callback is None:
        # 如果回调函数为 None，则直接返回 False，表示不需要停止
        return False
    try:
        # 调用回调函数并传入 res
        callback(res)
        return False
    except StopIteration:
        # 如果回调函数抛出 StopIteration 异常，则标记 callback.stop_iteration 为 True，返回 True
        callback.stop_iteration = True
        return True


class _RichResult(dict):
    """带有漂亮打印输出的多个输出的容器类"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]
    # 返回对象的字符串表示形式，通常用于调试和日志记录
    def __repr__(self):
        # 定义默认的键顺序，用于排序对象属性
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin',
                      'converged', 'flag', 'function_calls', 'iterations',
                      'root']
        # 获取对象可能存在的自定义键顺序，若无则使用默认顺序
        order_keys = getattr(self, '_order_keys', order_keys)
        
        # 指定要忽略的键集合，这些键不会包含在输出中
        # 'slack', 'con' 是冗余的，因为可以使用 residuals 代替
        # 'crossover_nit' 对大多数用户可能不感兴趣
        omit_keys = {'slack', 'con', 'crossover_nit', '_order_keys'}

        # 定义用于排序属性的函数
        def key(item):
            try:
                # 尝试根据键的顺序返回索引，如果键不在 order_keys 中则返回无穷大
                return order_keys.index(item[0].lower())
            except ValueError:  # 如果键不在顺序列表中则返回无穷大
                return np.inf

        # 过滤掉要忽略的冗余键，返回过滤后的属性生成器
        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        # 对属性字典进行排序处理，先过滤再排序
        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        # 如果对象存在属性，则返回格式化后的字符串表示形式
        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            # 如果对象没有任何属性，则返回类名作为字符串表示形式
            return self.__class__.__name__ + "()"

    # 返回对象的所有属性名称列表
    def __dir__(self):
        return list(self.keys())
def _indenter(s, n=0):
    """
    Ensures that lines after the first are indented by the specified amount
    
    Args:
        s (str): The input string to be indented.
        n (int, optional): The number of spaces to indent by. Defaults to 0.
    
    Returns:
        str: The indented string.
    """
    split = s.split("\n")  # Split the input string by newline characters
    indent = " "*n  # Create an indentation string with n spaces
    return ("\n" + indent).join(split)  # Join split lines with newline and indentation


def _float_formatter_10(x):
    """
    Returns a string representation of a float with exactly ten characters
    
    Args:
        x (float): The floating-point number to be formatted.
    
    Returns:
        str: The formatted string representation of the float.
    """
    if np.isposinf(x):  # Check if x is positive infinity
        return "       inf"  # Return formatted string for positive infinity
    elif np.isneginf(x):  # Check if x is negative infinity
        return "      -inf"  # Return formatted string for negative infinity
    elif np.isnan(x):  # Check if x is NaN (Not a Number)
        return "       nan"  # Return formatted string for NaN
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)
    # Format the float using scientific notation with specified precision and padding


def _dict_formatter(d, n=0, mplus=1, sorter=None):
    """
    Pretty printer for dictionaries
    
    Args:
        d (dict or other): The dictionary or other object to be formatted.
        n (int, optional): Starting indentation level. Defaults to 0.
        mplus (int, optional): Additional left padding applied to keys. Defaults to 1.
        sorter (function, optional): Function used to sort dictionary items. Defaults to None.
    
    Returns:
        str: Formatted string representation of the dictionary or object.
    """
    if isinstance(d, dict):  # Check if d is a dictionary
        m = max(map(len, list(d.keys()))) + mplus  # Calculate maximum width for keys
        s = '\n'.join([k.rjust(m) + ': ' +  # Right justify keys with width m
                       _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
                       for k, v in sorter(d)])  # Format and indent each key-value pair
    else:
        # Format non-dictionary objects (likely NumPy arrays)
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)  # Convert d to string using specified NumPy options
    return s  # Return the formatted string
```