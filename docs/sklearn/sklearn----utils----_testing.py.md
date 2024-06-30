# `D:\src\scipysrc\scikit-learn\sklearn\utils\_testing.py`

```
# 导入标准库和第三方库
import atexit  # 用于注册退出函数的模块
import contextlib  # 用于创建上下文管理器的模块
import functools  # 提供了创建偏函数（partial function）的功能
import importlib  # 提供了导入和重新加载模块的功能
import inspect  # 提供了解析源代码、检查类和函数、获取对象信息的功能
import os  # 提供了与操作系统交互的功能
import os.path as op  # os 模块的路径操作的别名
import re  # 提供了正则表达式的功能
import shutil  # 提供了高级的文件操作功能
import sys  # 提供了对 Python 解释器的访问和控制
import tempfile  # 提供了创建临时文件和目录的功能
import unittest  # 提供了编写和运行单元测试的功能
import warnings  # 提供了警告相关的功能

# 从标准库和第三方库导入特定模块和函数
from collections.abc import Iterable  # 导入 collections.abc 模块的 Iterable 类
from dataclasses import dataclass  # 导入 dataclasses 模块的 dataclass 装饰器
from functools import wraps  # 导入 functools 模块的 wraps 装饰器
from inspect import signature  # 导入 inspect 模块的 signature 函数
from subprocess import (STDOUT, CalledProcessError, TimeoutExpired, check_output)
# 导入 subprocess 模块的 STDOUT 常量，以及 CalledProcessError、TimeoutExpired、check_output 函数

# 从第三方库导入特定模块和函数
import joblib  # 导入 joblib 库，用于并行执行任务
import numpy as np  # 导入 NumPy 库，用于科学计算
import scipy as sp  # 导入 SciPy 库，用于科学计算
from numpy.testing import assert_allclose as np_assert_allclose
# 从 numpy.testing 模块导入 assert_allclose 函数，并重命名为 np_assert_allclose
from numpy.testing import (
    assert_almost_equal,  # 导入 numpy.testing 模块的 assert_almost_equal 函数
    assert_approx_equal,  # 导入 numpy.testing 模块的 assert_approx_equal 函数
    assert_array_almost_equal,  # 导入 numpy.testing 模块的 assert_array_almost_equal 函数
    assert_array_equal,  # 导入 numpy.testing 模块的 assert_array_equal 函数
    assert_array_less,  # 导入 numpy.testing 模块的 assert_array_less 函数
    assert_no_warnings,  # 导入 numpy.testing 模块的 assert_no_warnings 函数
)

# 导入 scikit-learn 库及其子模块和函数
import sklearn  # 导入 scikit-learn 库
from sklearn.utils._array_api import _check_array_api_dispatch
# 导入 scikit-learn 库的 _array_api 模块的 _check_array_api_dispatch 函数
from sklearn.utils.fixes import (
    _IS_32BIT,  # 导入 scikit-learn 库的 fixes 模块的 _IS_32BIT 常量
    VisibleDeprecationWarning,  # 导入 scikit-learn 库的 fixes 模块的 VisibleDeprecationWarning 警告类
    _in_unstable_openblas_configuration,  # 导入 scikit-learn 库的 fixes 模块的 _in_unstable_openblas_configuration 函数
    parse_version,  # 导入 scikit-learn 库的 fixes 模块的 parse_version 函数
    sp_version,  # 导入 scikit-learn 库的 fixes 模块的 sp_version 函数
)
from sklearn.utils.multiclass import check_classification_targets
# 导入 scikit-learn 库的 multiclass 模块的 check_classification_targets 函数
from sklearn.utils.validation import (
    check_array,  # 导入 scikit-learn 库的 validation 模块的 check_array 函数
    check_is_fitted,  # 导入 scikit-learn 库的 validation 模块的 check_is_fitted 函数
    check_X_y,  # 导入 scikit-learn 库的 validation 模块的 check_X_y 函数
)

__all__ = [
    "assert_raises",  # 将 assert_raises 添加到 __all__ 列表中
    "assert_raises_regexp",  # 将 assert_raises_regexp 添加到 __all__ 列表中
    "assert_array_equal",  # 将 assert_array_equal 添加到 __all__ 列表中
    "assert_almost_equal",  # 将 assert_almost_equal 添加到 __all__ 列表中
    "assert_array_almost_equal",  # 将 assert_array_almost_equal 添加到 __all__ 列表中
    "assert_array_less",  # 将 assert_array_less 添加到 __all__ 列表中
    "assert_approx_equal",  # 将 assert_approx_equal 添加到 __all__ 列表中
    "assert_allclose",  # 将 assert_allclose 添加到 __all__ 列表中
    "assert_run_python_script_without_output",  # 将 assert_run_python_script_without_output 添加到 __all__ 列表中
    "assert_no_warnings",  # 将 assert_no_warnings 添加到 __all__ 列表中
    "SkipTest",  # 将 SkipTest 添加到 __all__ 列表中
]

_dummy = TestCase("__init__")  # 创建一个 TestCase 的实例 _dummy
assert_raises = _dummy.assertRaises  # 将 _dummy 实例的 assertRaises 方法赋值给 assert_raises
SkipTest = unittest.case.SkipTest  # 将 unittest.case 中的 SkipTest 类赋值给 SkipTest
assert_dict_equal = _dummy.assertDictEqual  # 将 _dummy 实例的 assertDictEqual 方法赋值给 assert_dict_equal

assert_raises_regex = _dummy.assertRaisesRegex
# 将 _dummy 实例的 assertRaisesRegex 方法赋值给 assert_raises_regex
# assert_raises_regexp 在 Python 3.4 中已弃用，推荐使用 assert_raises_regex，但为了兼容性，保留 assert_raises_regexp 名称

# 函数定义：忽略警告的上下文管理器和装饰器
def ignore_warnings(obj=None, category=Warning):
    """Context manager and decorator to ignore warnings.

    Note: Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging, this is not your tool of choice.

    Parameters
    ----------
    obj : callable, default=None
        callable where you want to ignore the warnings.
    category : warning class, default=Warning
        The category to filter. If Warning, all categories will be muted.

    Examples
    --------
    >>> import warnings
    >>> from sklearn.utils._testing import ignore_warnings
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')

    >>> def nasty_warn():
    ...     warnings.warn('buhuhuhu')
    ...     print(42)

    >>> ignore_warnings(nasty_warn)()
    42
    """
    # 检查 obj 是否是类型且是 Warning 的子类
    if isinstance(obj, type) and issubclass(obj, Warning):
        # 避免常见错误，即将类别作为第一个位置参数传递，
        # 这会导致测试未能运行
        warning_name = obj.__name__
        # 抛出值错误，提示应当传入可调用对象而非警告类
        raise ValueError(
            "'obj' 应当是一个可调用对象以忽略警告。"
            "您传入了一个警告类: 'obj={warning_name}'. "
            "如果您想传入一个警告类以忽略警告，"
            "您应当使用 'category={warning_name}'".format(warning_name=warning_name)
        )
    # 如果 obj 是可调用对象，则使用 _IgnoreWarnings 装饰器来忽略警告
    elif callable(obj):
        return _IgnoreWarnings(category=category)(obj)
    else:
        # 如果 obj 既不是警告类也不是可调用对象，则抛出错误
        return _IgnoreWarnings(category=category)
class _IgnoreWarnings:
    """Improved and simplified Python warnings context manager and decorator.

    This class allows the user to ignore the warnings raised by a function.
    Copied from Python 2.7.5 and modified as required.

    Parameters
    ----------
    category : tuple of warning class, default=Warning
        The category to filter. By default, all the categories will be muted.
    """

    def __init__(self, category):
        # 初始化 _IgnoreWarnings 实例
        self._record = True
        # 获取 warnings 模块对象
        self._module = sys.modules["warnings"]
        # 进入状态标志
        self._entered = False
        # 日志列表
        self.log = []
        # 要忽略的警告类别
        self.category = category

    def __call__(self, fn):
        """Decorator to catch and hide warnings without visual nesting."""

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # 使用 catch_warnings 上下文管理器捕获警告
            with warnings.catch_warnings():
                # 设置警告过滤器忽略指定类别的警告
                warnings.simplefilter("ignore", self.category)
                # 调用被装饰的函数
                return fn(*args, **kwargs)

        return wrapper

    def __repr__(self):
        # 返回 _IgnoreWarnings 实例的字符串表示形式
        args = []
        if self._record:
            args.append("record=True")
        if self._module is not sys.modules["warnings"]:
            args.append("module=%r" % self._module)
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def __enter__(self):
        # 进入上下文管理器
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        # 备份当前警告过滤器
        self._filters = self._module.filters
        # 备份当前 showwarning 函数
        self._showwarning = self._module.showwarning
        # 设置警告过滤器忽略指定类别的警告
        warnings.simplefilter("ignore", self.category)

    def __exit__(self, *exc_info):
        # 退出上下文管理器
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        # 恢复原始的警告过滤器
        self._module.filters = self._filters
        # 恢复原始的 showwarning 函数
        self._module.showwarning = self._showwarning
        # 清空日志列表
        self.log[:] = []


def assert_raise_message(exceptions, message, function, *args, **kwargs):
    """Helper function to test the message raised in an exception.

    Given an exception, a callable to raise the exception, and
    a message string, tests that the correct exception is raised and
    that the message is a substring of the error thrown. Used to test
    that the specific message thrown during an exception is correct.

    Parameters
    ----------
    exceptions : exception or tuple of exception
        An Exception object.

    message : str
        The error message or a substring of the error message.

    function : callable
        Callable object to raise error.

    *args : the positional arguments to `function`.

    **kwargs : the keyword arguments to `function`.
    """
    try:
        # 调用指定的函数，期望其抛出异常
        function(*args, **kwargs)
    except exceptions as e:
        # 获取捕获的异常的字符串表示
        error_message = str(e)
        # 检查预期的消息是否在捕获的异常信息中
        if message not in error_message:
            # 如果不在，抛出断言错误
            raise AssertionError(
                "Error message does not include the expected"
                " string: %r. Observed error message: %r" % (message, error_message)
            )
    else:
        # 如果exceptions是一个元组，则连接异常类名
        if isinstance(exceptions, tuple):
            names = " or ".join(e.__name__ for e in exceptions)
        else:
            # 否则，直接使用异常类的名称
            names = exceptions.__name__

        # 抛出断言错误，指明特定的异常没有被特定的函数所触发
        raise AssertionError("%s not raised by %s" % (names, function.__name__))
# 自定义断言函数，用于比较两个数组或稀疏矩阵的近似相等性
def assert_allclose(
    actual, desired, rtol=None, atol=0.0, equal_nan=True, err_msg="", verbose=True
):
    """dtype-aware variant of numpy.testing.assert_allclose

    This variant introspects the least precise floating point dtype
    in the input argument and automatically sets the relative tolerance
    parameter to 1e-4 float32 and use 1e-7 otherwise (typically float64
    in scikit-learn).

    `atol` is always left to 0. by default. It should be adjusted manually
    to an assertion-specific value in case there are null values expected
    in `desired`.

    The aggregate tolerance is `atol + rtol * abs(desired)`.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional, default=None
        Relative tolerance.
        If None, it is set based on the provided arrays' dtypes.
    atol : float, optional, default=0.
        Absolute tolerance.
    equal_nan : bool, optional, default=True
        If True, NaNs will compare equal.
    err_msg : str, optional, default=''
        The error message to be printed in case of failure.
    verbose : bool, optional, default=True
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.

    See Also
    --------
    numpy.testing.assert_allclose

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils._testing import assert_allclose
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> assert_allclose(x, y, rtol=1e-5, atol=0)
    >>> a = np.full(shape=10, fill_value=1e-5, dtype=np.float32)
    >>> assert_allclose(a, 1e-5)
    """
    # 初始化一个空列表，用于存储数组或矩阵的数据类型
    dtypes = []

    # 将输入的 actual 和 desired 转换为 NumPy 数组，保证统一的数据类型处理方式
    actual, desired = np.asanyarray(actual), np.asanyarray(desired)
    # 获取 actual 和 desired 的数据类型，并存储在 dtypes 列表中
    dtypes = [actual.dtype, desired.dtype]

    # 如果用户没有提供 rtol 参数，则根据数据类型设置默认的相对容差 rtol
    if rtol is None:
        # 对于每种数据类型，选择不同的默认相对容差值
        rtols = [1e-4 if dtype == np.float32 else 1e-7 for dtype in dtypes]
        rtol = max(rtols)

    # 调用 NumPy 的 assert_allclose 函数进行比较
    np_assert_allclose(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )


def assert_allclose_dense_sparse(x, y, rtol=1e-07, atol=1e-9, err_msg=""):
    """Assert allclose for sparse and dense data.

    Both x and y need to be either sparse or dense, they
    can't be mixed.

    Parameters
    ----------
    x : {array-like, sparse matrix}
        First array to compare.

    y : {array-like, sparse matrix}
        Second array to compare.

    rtol : float, default=1e-07
        relative tolerance; see numpy.allclose.

    atol : float, default=1e-9
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.

    err_msg : str, default=''
        Error message to raise.
    """
    # 检查输入的 x 和 y 是否都是稀疏矩阵，并且都是稀疏矩阵
    if sp.sparse.issparse(x) and sp.sparse.issparse(y):
        # 将稀疏矩阵 x 和 y 转换为 CSR 格式
        x = x.tocsr()
        y = y.tocsr()
        # 对转换后的 CSR 格式的稀疏矩阵去除重复的元素
        x.sum_duplicates()
        y.sum_duplicates()
        # 检查稀疏矩阵的非零元素的列索引是否相等
        assert_array_equal(x.indices, y.indices, err_msg=err_msg)
        # 检查稀疏矩阵的行指针数组是否相等
        assert_array_equal(x.indptr, y.indptr, err_msg=err_msg)
        # 检查稀疏矩阵的非零元素值是否在给定的容差范围内相等
        assert_allclose(x.data, y.data, rtol=rtol, atol=atol, err_msg=err_msg)
    # 如果 x 和 y 都不是稀疏矩阵（即都是密集矩阵）
    elif not sp.sparse.issparse(x) and not sp.sparse.issparse(y):
        # 比较两个密集矩阵的元素是否在给定的容差范围内相等
        assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)
    else:
        # 如果一个是稀疏矩阵，另一个是数组，则抛出 ValueError 异常
        raise ValueError(
            "Can only compare two sparse matrices, not a sparse matrix and an array."
        )
# 设置估算器的随机状态，如果估算器具有 `random_state` 参数的话
def set_random_state(estimator, random_state=0):
    if "random_state" in estimator.get_params():  # 检查估算器是否支持 `random_state` 参数
        estimator.set_params(random_state=random_state)  # 设置估算器的随机状态为指定的值


try:
    _check_array_api_dispatch(True)  # 检查数组 API 的分派情况
    ARRAY_API_COMPAT_FUNCTIONAL = True  # 设置数组 API 兼容性标志为 True
except ImportError:
    ARRAY_API_COMPAT_FUNCTIONAL = False  # 如果导入失败，设置数组 API 兼容性标志为 False

try:
    import pytest  # 尝试导入 pytest 测试框架

    # 定义测试装饰器，如果平台是 32 位则跳过
    skip_if_32bit = pytest.mark.skipif(_IS_32BIT, reason="skipped on 32bit platforms")
    
    # 定义测试装饰器，如果 OpenBLAS 配置不稳定则标记为失败
    fails_if_unstable_openblas = pytest.mark.xfail(
        _in_unstable_openblas_configuration(),
        reason="OpenBLAS is unstable for this configuration",
    )
    
    # 定义测试装饰器，如果 joblib 不处于并行模式则跳过
    skip_if_no_parallel = pytest.mark.skipif(
        not joblib.parallel.mp, reason="joblib is in serial mode"
    )
    
    # 定义测试装饰器，如果数组 API 兼容性未配置则跳过
    skip_if_array_api_compat_not_configured = pytest.mark.skipif(
        not ARRAY_API_COMPAT_FUNCTIONAL,
        reason="requires array_api_compat installed and a new enough version of NumPy",
    )

    # 测试装饰器，用于同时涉及 BLAS 调用和多进程的测试
    if_safe_multiprocessing_with_blas = pytest.mark.skipif(
        sys.platform == "darwin", reason="Possible multi-process bug with some BLAS"
    )
except ImportError:
    pass  # 如果导入 pytest 失败，则跳过


def check_skip_network():
    if int(os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", 0)):  # 检查环境变量是否跳过网络测试
        raise SkipTest("Text tutorial requires large dataset download")  # 抛出跳过测试的异常


def _delete_folder(folder_path, warn=False):
    """Utility function to cleanup a temporary folder if still existing.

    Copy from joblib.pool (for independence).
    """
    try:
        if os.path.exists(folder_path):  # 检查临时文件夹是否存在
            shutil.rmtree(folder_path)  # 删除临时文件夹及其内容（递归删除）
    except Exception:
        # 在 Windows 下可能会失败，但在退出时会成功
        pass  # 捕获异常，不做任何处理
    # 如果发生 OSError 异常（操作系统错误）
    except OSError:
        # 如果 warn 参数为真，则发出警告，提示无法删除临时文件夹
        if warn:
            # 发出警告信息，指示无法删除临时文件夹的路径
            warnings.warn("Could not delete temporary folder %s" % folder_path)
class TempMemmap:
    """
    Parameters
    ----------
    data
        数据对象，用于创建临时内存映射文件
    mmap_mode : str, default='r'
        内存映射模式，默认为只读 ('r')
    """

    def __init__(self, data, mmap_mode="r"):
        # 初始化函数，设置内存映射模式和数据对象
        self.mmap_mode = mmap_mode
        self.data = data

    def __enter__(self):
        # 进入上下文管理器时调用的方法
        # 创建基于内存映射的数据和临时文件夹
        data_read_only, self.temp_folder = create_memmap_backed_data(
            self.data, mmap_mode=self.mmap_mode, return_folder=True
        )
        return data_read_only

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 离开上下文管理器时调用的方法
        # 删除临时文件夹及其内容
        _delete_folder(self.temp_folder)


def create_memmap_backed_data(data, mmap_mode="r", return_folder=False):
    """
    Parameters
    ----------
    data
        要保存为内存映射的数据对象
    mmap_mode : str, default='r'
        内存映射模式，默认为只读 ('r')
    return_folder :  bool, default=False
        是否返回临时文件夹路径

    Returns
    -------
    result : object or tuple
        如果return_folder为True，则返回内存映射数据对象及临时文件夹路径；
        否则，仅返回内存映射数据对象
    """
    # 创建一个带有随机前缀的临时文件夹
    temp_folder = tempfile.mkdtemp(prefix="sklearn_testing_")
    # 注册一个函数，在退出时删除临时文件夹
    atexit.register(functools.partial(_delete_folder, temp_folder, warn=True))
    # 在临时文件夹中生成一个数据文件的完整路径
    filename = op.join(temp_folder, "data.pkl")
    # 使用joblib将数据对象保存到指定路径中
    joblib.dump(data, filename)
    # 使用指定的内存映射模式加载数据文件
    memmap_backed_data = joblib.load(filename, mmap_mode=mmap_mode)
    # 根据return_folder的值返回相应的结果
    result = (
        memmap_backed_data if not return_folder else (memmap_backed_data, temp_folder)
    )
    return result


# Utils to test docstrings


def _get_args(function, varargs=False):
    """Helper to get function arguments.

    Parameters
    ----------
    function : callable
        要检查参数的函数对象
    varargs : bool, default=False
        是否包括可变位置参数

    Returns
    -------
    args : list
        函数的参数列表
    varargs : list or None
        可变位置参数的名称列表，如果不存在则返回None
    """
    try:
        # 尝试获取函数的参数信息
        params = signature(function).parameters
    except ValueError:
        # 如果是内置的C函数，会抛出错误
        # 返回一个空列表
        return []
    # 从参数信息中提取参数名称，排除可变位置参数和关键字参数
    args = [
        key
        for key, param in params.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    ]
    if varargs:
        # 如果需要包括可变位置参数，再次遍历参数信息
        varargs = [
            param.name
            for param in params.values()
            if param.kind == param.VAR_POSITIONAL
        ]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args


def _get_func_name(func):
    """Get function full name.

    Parameters
    ----------
    func : callable
        要获取全名的函数对象

    Returns
    -------
    name : str
        函数的完整名称，包括模块名称和限定名称
    """
    # 初始化一个部分名称列表
    parts = []
    # 获取函数所在的模块对象
    module = inspect.getmodule(func)
    if module:
        # 如果模块存在，将模块名称添加到部分名称列表中
        parts.append(module.__name__)

    # 获取函数的限定名称
    qualname = func.__qualname__
    # 如果限定名称不等于函数名，将限定名称的第一个点之前的部分添加到列表中
    if qualname != func.__name__:
        parts.append(qualname[: qualname.find(".")])

    # 最后添加函数的名称到部分名称列表中，并将它们用点号连接起来形成完整的名称
    parts.append(func.__name__)
    return ".".join(parts)


def check_docstring_parameters(func, doc=None, ignore=None):
    """Helper to check docstring.

    Parameters
    ----------
    func : callable
        要测试的函数对象
    doc : str, default=None
        手动传递的函数文档字符串
    ignore : list, default=None
        需要忽略的参数列表

    Returns
    -------
    incorrect : list
        描述不正确结果的字符串列表
    """
    from numpydoc import docscrape

    # 初始化一个不正确结果的列表
    incorrect = []
    # 如果ignore为None，则初始化一个空列表
    ignore = [] if ignore is None else ignore

    # 获取函数的完整名称
    func_name = _get_func_name(func)
    # 检查函数名称是否以'sklearn.'开头且不以'sklearn.externals'开头
    if not func_name.startswith("sklearn.") or func_name.startswith(
        "sklearn.externals"
    ):
        # 如果参数名为空，则返回错误列表
        return incorrect
    # 对于属性函数，不检查文档字符串
    if inspect.isdatadescriptor(func):
        # 返回错误列表
        return incorrect
    # 对于 pytest 的 setup / teardown 函数，不检查文档字符串
    if func_name.split(".")[-1] in ("setup_module", "teardown_module"):
        # 返回错误列表
        return incorrect
    # 不检查 estimator_checks 模块的函数
    if func_name.split(".")[2] == "estimator_checks":
        # 返回错误列表
        return incorrect
    # 获取函数签名中的参数列表，排除 ignore 中的参数
    param_signature = list(filter(lambda x: x not in ignore, _get_args(func)))
    # 如果参数列表中包含 self，则移除
    if len(param_signature) > 0 and param_signature[0] == "self":
        param_signature.remove("self")

    # 分析函数的文档字符串
    if doc is None:
        records = []
        # 捕获警告信息
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", UserWarning)
            try:
                # 尝试获取函数的文档信息
                doc = docscrape.FunctionDoc(func)
            except UserWarning as exp:
                # 处理特定的警告信息，例如 numpydoc 1.2 中的错误提示
                if "potentially wrong underline length" in str(exp):
                    message = str(exp).split("\n")[:3]
                    # 添加错误信息到列表并返回
                    incorrect += [f"In function: {func_name}"] + message
                    return incorrect
                records.append(str(exp))
            except Exception as exp:
                # 处理其他异常情况
                incorrect += [func_name + " parsing error: " + str(exp)]
                return incorrect
        if len(records):
            # 如果有记录，抛出运行时错误
            raise RuntimeError("Error for %s:\n%s" % (func_name, records[0]))

    param_docs = []
    # 遍历函数文档中的参数信息
    for name, type_definition, param_doc in doc["Parameters"]:
        # 如果类型定义为空，则检查参数名是否以 : 结尾
        if not type_definition.strip():
            if ":" in name and name[: name.index(":")][-1:].strip():
                # 参数名和冒号之间没有空格
                incorrect += [
                    func_name
                    + " There was no space between the param name and colon (%r)" % name
                ]
            elif name.rstrip().endswith(":"):
                # 参数类型规范为空，移除冒号
                incorrect += [
                    func_name
                    + " Parameter %r has an empty type spec. Remove the colon"
                    % (name.lstrip())
                ]

        # 将参数名添加到参数文档列表中
        if "*" not in name:
            param_docs.append(name.split(":")[0].strip("` "))

    # 如果存在错误信息，则返回错误列表
    if len(incorrect) > 0:
        return incorrect

    # 从参数文档列表中移除应忽略的参数
    param_docs = list(filter(lambda x: x not in ignore, param_docs))

    # 以下代码片段来源于 pytest，版权归 Holger Krekel 及其他人所有，使用 MIT 许可证
    # 初始化空列表，用于存储错误信息
    message = []
    # 遍历参数文档和参数签名，比较它们的长度，取较小值作为范围
    for i in range(min(len(param_docs), len(param_signature))):
        # 检查当前索引处的参数名称是否不匹配
        if param_signature[i] != param_docs[i]:
            # 若不匹配，生成错误消息并添加到消息列表中
            message += [
                "There's a parameter name mismatch in function"
                " docstring w.r.t. function signature, at index %s"
                " diff: %r != %r" % (i, param_signature[i], param_docs[i])
            ]
            # 发现第一个不匹配后立即终止循环
            break
    # 如果参数签名长度大于参数文档长度，说明参数文档缺少条目
    if len(param_signature) > len(param_docs):
        # 生成相应错误消息并添加到消息列表中
        message += [
            "Parameters in function docstring have less items w.r.t."
            " function signature, first missing item: %s"
            % param_signature[len(param_docs)]
        ]
    # 如果参数签名长度小于参数文档长度，说明参数文档有多余条目
    elif len(param_signature) < len(param_docs):
        # 生成相应错误消息并添加到消息列表中
        message += [
            "Parameters in function docstring have more items w.r.t."
            " function signature, first extra item: %s"
            % param_docs[len(param_signature)]
        ]

    # 如果消息列表长度为0，表示参数本身没有差异，返回空列表
    if len(message) == 0:
        return []

    # 导入用于比较文本差异的模块
    import difflib
    import pprint

    # 格式化参数文档和参数签名，使其易于比较，并按行拆分为列表
    param_docs_formatted = pprint.pformat(param_docs).splitlines()
    param_signature_formatted = pprint.pformat(param_signature).splitlines()

    # 添加完整差异的标签到消息列表中
    message += ["Full diff:"]

    # 使用difflib比较格式化后的参数文档和参数签名，将差异行添加到消息列表中
    message.extend(
        line.strip()
        for line in difflib.ndiff(param_signature_formatted, param_docs_formatted)
    )

    # 将错误信息列表添加到外部列表（incorrect）
    incorrect.extend(message)

    # 在错误信息列表开头添加函数名作为标识
    incorrect = ["In function: " + func_name] + incorrect

    # 返回包含错误信息的列表
    return incorrect
# 定义一个函数，用于在独立的 Python 子进程中运行脚本并进行断言检查
def assert_run_python_script_without_output(source_code, pattern=".+", timeout=60):
    """Utility to check assertions in an independent Python subprocess.

    The script provided in the source code should return 0 and the stdtout +
    stderr should not match the pattern `pattern`.

    This is a port from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    source_code : str
        The Python source code to execute.
    pattern : str
        Pattern that the stdout + stderr should not match. By default, unless
        stdout + stderr are both empty, an error will be raised.
    timeout : int, default=60
        Time in seconds before timeout.
    """
    # 创建临时文件，将提供的源代码写入
    fd, source_file = tempfile.mkstemp(suffix="_src_test_sklearn.py")
    os.close(fd)
    try:
        with open(source_file, "wb") as f:
            f.write(source_code.encode("utf-8"))
        
        # 准备执行命令，使用系统的 Python 解释器执行临时文件
        cmd = [sys.executable, source_file]
        
        # 设置当前工作目录为 sklearn 的父目录，并配置环境变量 PYTHONPATH
        cwd = op.normpath(op.join(op.dirname(sklearn.__file__), ".."))
        env = os.environ.copy()
        try:
            env["PYTHONPATH"] = os.pathsep.join([cwd, env["PYTHONPATH"]])
        except KeyError:
            env["PYTHONPATH"] = cwd
        
        # 如果正在运行代码覆盖工具，传递配置文件到子进程
        coverage_rc = os.environ.get("COVERAGE_PROCESS_START")
        kwargs = {"cwd": cwd, "stderr": STDOUT, "env": env}
        if coverage_rc:
            kwargs["env"]["COVERAGE_PROCESS_START"] = coverage_rc
        
        # 设置子进程超时时间
        kwargs["timeout"] = timeout
        
        try:
            try:
                # 执行子进程命令，并获取标准输出和错误输出
                out = check_output(cmd, **kwargs)
            except CalledProcessError as e:
                raise RuntimeError(
                    "script errored with output:\n%s" % e.output.decode("utf-8")
                )
            
            # 解码子进程输出
            out = out.decode("utf-8")
            
            # 检查输出是否匹配指定的模式
            if re.search(pattern, out):
                if pattern == ".+":
                    expectation = "Expected no output"
                else:
                    expectation = f"The output was not supposed to match {pattern!r}"
                
                # 如果输出与模式匹配，则引发断言错误
                message = f"{expectation}, got the following output instead: {out!r}"
                raise AssertionError(message)
        
        # 捕获超时异常
        except TimeoutExpired as e:
            raise RuntimeError(
                "script timeout, output so far:\n%s" % e.output.decode("utf-8")
            )
    
    finally:
        # 删除临时创建的源文件
        os.unlink(source_file)
    columns_name : index or array-like, default=None
        # 列名或类似索引的对象，用于指定数据容器的列名
        For pandas container supporting `columns_names`, it will affect
        specific names.
        # 对于支持 `columns_names` 的 Pandas 容器，它会影响特定的列名。

    dtype : dtype, default=None
        # 强制指定容器的数据类型。不适用于 `"slice"` 类型的容器。
        Force the dtype of the container. Does not apply to `"slice"`
        container.

    minversion : str, default=None
        # 安装包的最低版本要求。
        Minimum version for package to install.

    categorical_feature_names : list of str, default=None
        # 需要转换为分类数据类型的列名列表。
        List of column names to cast to categorical dtype.

    Returns
    -------
    converted_container
    """
    # 根据构造函数名称进行不同类型的容器转换
    if constructor_name == "list":
        if dtype is None:
            # 返回列表形式的容器
            return list(container)
        else:
            # 返回指定数据类型的 NumPy 数组，并转换为列表形式
            return np.asarray(container, dtype=dtype).tolist()
    elif constructor_name == "tuple":
        if dtype is None:
            # 返回元组形式的容器
            return tuple(container)
        else:
            # 返回指定数据类型的 NumPy 数组，并转换为元组形式
            return tuple(np.asarray(container, dtype=dtype).tolist())
    elif constructor_name == "array":
        # 返回指定数据类型的 NumPy 数组
        return np.asarray(container, dtype=dtype)
    elif constructor_name in ("pandas", "dataframe"):
        pd = pytest.importorskip("pandas", minversion=minversion)
        # 使用 Pandas 创建 DataFrame 容器
        result = pd.DataFrame(container, columns=columns_name, dtype=dtype, copy=False)
        if categorical_feature_names is not None:
            # 将指定列名转换为分类数据类型
            for col_name in categorical_feature_names:
                result[col_name] = result[col_name].astype("category")
        return result
    elif constructor_name == "pyarrow":
        pa = pytest.importorskip("pyarrow", minversion=minversion)
        array = np.asarray(container)
        if columns_name is None:
            # 如果列名为空，则生成默认列名
            columns_name = [f"col{i}" for i in range(array.shape[1])]
        # 使用 PyArrow 创建 Table 容器
        data = {name: array[:, i] for i, name in enumerate(columns_name)}
        result = pa.Table.from_pydict(data)
        if categorical_feature_names is not None:
            # 将指定列名转换为分类数据类型
            for col_idx, col_name in enumerate(result.column_names):
                if col_name in categorical_feature_names:
                    result = result.set_column(
                        col_idx, col_name, result.column(col_name).dictionary_encode()
                    )
        return result
    elif constructor_name == "polars":
        pl = pytest.importorskip("polars", minversion=minversion)
        # 使用 Polars 创建 DataFrame 容器
        result = pl.DataFrame(container, schema=columns_name, orient="row")
        if categorical_feature_names is not None:
            # 将指定列名转换为分类数据类型
            for col_name in categorical_feature_names:
                result = result.with_columns(pl.col(col_name).cast(pl.Categorical))
        return result
    elif constructor_name == "series":
        pd = pytest.importorskip("pandas", minversion=minversion)
        # 返回 Pandas Series 容器
        return pd.Series(container, dtype=dtype)
    elif constructor_name == "polars_series":
        pl = pytest.importorskip("polars", minversion=minversion)
        # 返回 Polars Series 容器
        return pl.Series(values=container)
    elif constructor_name == "index":
        pd = pytest.importorskip("pandas", minversion=minversion)
        # 返回 Pandas Index 容器
        return pd.Index(container, dtype=dtype)
    # 如果构造器名称是 "slice"，返回一个切片对象，使用 container 的第一个和第二个元素作为参数
    elif constructor_name == "slice":
        return slice(container[0], container[1])
    # 如果构造器名称包含 "sparse"
    elif "sparse" in constructor_name:
        # 如果 container 不是稀疏矩阵，确保将其至少转换为二维数组
        if not sp.sparse.issparse(container):
            # 对于 scipy >= 1.13，从 1 维数组构造的稀疏数组可能是 1 维的，也可能引发异常。
            # 为了避免这种情况，确保输入的 container 是二维的。详细信息请参见链接
            container = np.atleast_2d(container)

        # 如果构造器名称包含 "array" 并且 scipy 版本小于 1.8
        if "array" in constructor_name and sp_version < parse_version("1.8"):
            # 抛出数值错误，指出需要 scipy >= 1.8.0 版本
            raise ValueError(
                f"{constructor_name} is only available with scipy>=1.8.0, got "
                f"{sp_version}"
            )
        
        # 根据构造器名称不同，返回相应的稀疏矩阵对象
        if constructor_name in ("sparse", "sparse_csr"):
            # 对于 "sparse" 和 "sparse_csr"，返回 CSR 格式的稀疏矩阵
            return sp.sparse.csr_matrix(container, dtype=dtype)
        elif constructor_name == "sparse_csr_array":
            # 对于 "sparse_csr_array"，返回 CSR 格式的稀疏矩阵，使用指定的数据类型
            return sp.sparse.csr_array(container, dtype=dtype)
        elif constructor_name == "sparse_csc":
            # 对于 "sparse_csc"，返回 CSC 格式的稀疏矩阵，使用指定的数据类型
            return sp.sparse.csc_matrix(container, dtype=dtype)
        elif constructor_name == "sparse_csc_array":
            # 对于 "sparse_csc_array"，返回 CSC 格式的稀疏矩阵，使用指定的数据类型
            return sp.sparse.csc_array(container, dtype=dtype)
# 创建一个上下文管理器用于确保代码块内部会抛出指定的异常

def raises(expected_exc_type, match=None, may_pass=False, err_msg=None):
    """Context manager to ensure exceptions are raised within a code block.

    This is similar to and inspired from pytest.raises, but supports a few
    other cases.

    This is only intended to be used in estimator_checks.py where we don't
    want to use pytest. In the rest of the code base, just use pytest.raises
    instead.

    Parameters
    ----------
    excepted_exc_type : Exception or list of Exception
        The exception that should be raised by the block. If a list, the block
        should raise one of the exceptions.
    match : str or list of str, default=None
        A regex that the exception message should match. If a list, one of
        the entries must match. If None, match isn't enforced.
    may_pass : bool, default=False
        If True, the block is allowed to not raise an exception. Useful in
        cases where some estimators may support a feature but others must
        fail with an appropriate error message. By default, the context
        manager will raise an exception if the block does not raise an
        exception.
    err_msg : str, default=None
        If the context manager fails (e.g. the block fails to raise the
        proper exception, or fails to match), then an AssertionError is
        raised with this message. By default, an AssertionError is raised
        with a default error message (depends on the kind of failure). Use
        this to indicate how users should fix their estimators to pass the
        checks.

    Attributes
    ----------
    raised_and_matched : bool
        True if an exception was raised and a match was found, False otherwise.
    """
    # 返回一个 _Raises 上下文管理器的实例，用于具体的异常检查
    return _Raises(expected_exc_type, match, may_pass, err_msg)


class _Raises(contextlib.AbstractContextManager):
    # see raises() for parameters
    def __init__(self, expected_exc_type, match, may_pass, err_msg):
        # 将 expected_exc_type 转换为迭代对象（列表），以便处理多种异常情况
        self.expected_exc_types = (
            expected_exc_type
            if isinstance(expected_exc_type, Iterable)
            else [expected_exc_type]
        )
        # 将 match 参数转换为列表，以便处理多个匹配条件
        self.matches = [match] if isinstance(match, str) else match
        # 是否允许通过（即代码块未抛出异常时是否算通过）
        self.may_pass = may_pass
        # 错误消息，如果上下文管理器检测失败，则抛出 AssertionError，并使用该消息
        self.err_msg = err_msg
        # 标记是否抛出了期望的异常并且匹配了条件
        self.raised_and_matched = False
    # __exit__ 方法用于处理 with 语句块中的异常情况，参考 PEP 343
    def __exit__(self, exc_type, exc_value, _):
        # 如果没有异常抛出
        if exc_type is None:
            # 如果允许通过（may_pass 为 True），返回 True，表示上下文管理器正常退出
            if self.may_pass:
                return True  # CM is happy
            else:
                # 否则，如果不允许通过，抛出断言错误，指明未抛出期望的异常类型
                err_msg = self.err_msg or f"Did not raise: {self.expected_exc_types}"
                raise AssertionError(err_msg)

        # 如果抛出的异常类型不在期望的异常类型列表中
        if not any(
            issubclass(exc_type, expected_type)
            for expected_type in self.expected_exc_types
        ):
            # 如果有自定义错误消息，抛出断言错误并附带原始异常信息
            if self.err_msg is not None:
                raise AssertionError(self.err_msg) from exc_value
            else:
                return False  # 将重新抛出原始异常

        # 如果设置了匹配规则 matches
        if self.matches is not None:
            # 构造错误消息，说明错误消息应包含匹配规则中的任意一种模式
            err_msg = self.err_msg or (
                "The error message should contain one of the following "
                "patterns:\n{}\nGot {}".format("\n".join(self.matches), str(exc_value))
            )
            # 如果异常值不包含任何匹配规则中的模式，抛出断言错误并附带原始异常信息
            if not any(re.search(match, str(exc_value)) for match in self.matches):
                raise AssertionError(err_msg) from exc_value
            self.raised_and_matched = True

        # 返回 True 表示异常已处理
        return True
# 定义一个最小的分类器类，不继承自 BaseEstimator。

class MinimalClassifier:
    """Minimal classifier implementation without inheriting from BaseEstimator.
    
    This estimator should be tested with:
    
    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    _estimator_type = "classifier"

    def __init__(self, param=None):
        self.param = param  # 初始化参数 param

    def get_params(self, deep=True):
        return {"param": self.param}  # 返回当前参数的字典表示

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)  # 设置参数
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)  # 确保输入数据 X 和标签 y 合法
        check_classification_targets(y)  # 检查分类目标的有效性
        self.classes_, counts = np.unique(y, return_counts=True)  # 获取类别和对应的计数
        self._most_frequent_class_idx = counts.argmax()  # 找到最频繁出现的类别的索引
        return self  # 返回自身作为训练后的分类器对象

    def predict_proba(self, X):
        check_is_fitted(self)  # 检查模型是否已经拟合
        X = check_array(X)  # 确保输入数据 X 合法
        proba_shape = (X.shape[0], self.classes_.size)
        y_proba = np.zeros(shape=proba_shape, dtype=np.float64)  # 初始化概率预测数组
        y_proba[:, self._most_frequent_class_idx] = 1.0  # 预测每个样本属于最频繁类别的概率为1
        return y_proba  # 返回预测的概率数组

    def predict(self, X):
        y_proba = self.predict_proba(X)  # 调用 predict_proba 方法得到预测的概率数组
        y_pred = y_proba.argmax(axis=1)  # 找到每个样本预测概率最大的类别索引
        return self.classes_[y_pred]  # 返回预测的类别标签

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        
        return accuracy_score(y, self.predict(X))  # 使用准确率评估预测效果的得分


class MinimalRegressor:
    """Minimal regressor implementation without inheriting from BaseEstimator.
    
    This estimator should be tested with:
    
    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    _estimator_type = "regressor"

    def __init__(self, param=None):
        self.param = param  # 初始化参数 param

    def get_params(self, deep=True):
        return {"param": self.param}  # 返回当前参数的字典表示

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)  # 设置参数
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)  # 确保输入数据 X 和标签 y 合法
        self.is_fitted_ = True  # 设置拟合状态为 True
        self._mean = np.mean(y)  # 计算标签 y 的均值作为预测值
        return self  # 返回自身作为训练后的回归器对象

    def predict(self, X):
        check_is_fitted(self)  # 检查模型是否已经拟合
        X = check_array(X)  # 确保输入数据 X 合法
        return np.ones(shape=(X.shape[0],)) * self._mean  # 返回每个样本预测值均为训练时计算的均值

    def score(self, X, y):
        from sklearn.metrics import r2_score
        
        return r2_score(y, self.predict(X))  # 使用 R^2 分数评估预测效果的得分


class MinimalTransformer:
    """Minimal transformer implementation without inheriting from
    BaseEstimator.
    
    This estimator should be tested with:
    
    * `check_estimator` in `test_estimator_checks.py`;
    * within a `Pipeline` in `test_pipeline.py`;
    * within a `SearchCV` in `test_search.py`.
    """

    def __init__(self, param=None):
        self.param = param  # 初始化参数 param

    def get_params(self, deep=True):
        return {"param": self.param}  # 返回当前参数的字典表示
    # 定义一个方法用于设置对象的参数，允许通过关键字参数传入多个参数
    def set_params(self, **params):
        # 遍历参数字典，使用 setattr 方法将每个参数设置为对象的属性
        for key, value in params.items():
            setattr(self, key, value)
        # 返回设置参数后的对象自身
        return self

    # 定义一个训练方法，用于标记对象已经进行了训练
    def fit(self, X, y=None):
        # 检查输入的数据 X 是否合法
        check_array(X)
        # 将对象的 is_fitted_ 属性设置为 True，表示对象已经训练完成
        self.is_fitted_ = True
        # 返回训练后的对象自身
        return self

    # 定义一个转换方法，用于对输入的数据 X 进行转换操作
    def transform(self, X, y=None):
        # 检查对象是否已经完成了训练，如果未完成将会引发异常
        check_is_fitted(self)
        # 检查并转换输入的数据 X，确保其符合要求
        X = check_array(X)
        # 返回转换后的数据 X
        return X

    # 定义一个方法，结合 fit 和 transform 方法，先进行训练再进行转换
    def fit_transform(self, X, y=None):
        # 调用 fit 方法进行训练，并立即调用 transform 方法对数据 X 进行转换
        return self.fit(X, y).transform(X, y)
# 定义一个函数，用于获取测试中所需的数组 API 模块
def _array_api_for_tests(array_namespace, device):
    try:
        # 尝试导入给定的数组命名空间模块
        array_mod = importlib.import_module(array_namespace)
    except ModuleNotFoundError:
        # 如果模块未找到，抛出跳过测试的异常
        raise SkipTest(
            f"{array_namespace} is not installed: not checking array_api input"
        )
    try:
        # 尝试导入 array_api_compat 模块（兼容性模块）
        import array_api_compat  # noqa
    except ImportError:
        # 如果导入失败，抛出跳过测试的异常
        raise SkipTest(
            "array_api_compat is not installed: not checking array_api input"
        )

    # 创建一个使用选择的数组模块的数组，然后根据它获取兼容性封装的数组命名空间
    # 这是因为 `cupy` 不同于其兼容性封装的 CuPy 数组命名空间。
    xp = array_api_compat.get_namespace(array_mod.asarray(1))

    # 检查特定情况下是否需要跳过测试
    if (
        array_namespace == "torch"
        and device == "cuda"
        and not xp.backends.cuda.is_built()
    ):
        # 如果是 PyTorch 测试，并且需要 CUDA 但 CUDA 不可用，抛出跳过测试的异常
        raise SkipTest("PyTorch test requires cuda, which is not available")
    elif array_namespace == "torch" and device == "mps":
        if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            # 对于 MPS 设备，需要设置环境变量 PYTORCH_ENABLE_MPS_FALLBACK=1，
            # 否则抛出跳过测试的异常
            raise SkipTest(
                "Skipping MPS device test because PYTORCH_ENABLE_MPS_FALLBACK is not "
                "set."
            )
        if not xp.backends.mps.is_built():
            # 如果当前的 PyTorch 安装未启用 MPS，抛出跳过测试的异常
            raise SkipTest(
                "MPS is not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
    elif array_namespace in {"cupy", "cupy.array_api"}:  # pragma: nocover
        import cupy

        # 如果是 CuPy 测试，并且 CUDA 不可用，抛出跳过测试的异常
        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise SkipTest("CuPy test requires cuda, which is not available")

    # 返回兼容性封装的数组命名空间对象 xp
    return xp


# 定义一个函数，用于获取警告过滤器信息列表
def _get_warnings_filters_info_list():
    @dataclass
    class WarningInfo:
        action: "warnings._ActionKind"
        message: str = ""
        category: type[Warning] = Warning

        def to_filterwarning_str(self):
            # 将警告类别转换为过滤器警告字符串的方法
            if self.category.__module__ == "builtins":
                category = self.category.__name__
            else:
                category = f"{self.category.__module__}.{self.category.__name__}"

            # 返回格式化的过滤器警告字符串
            return f"{self.action}:{self.message}:{category}"
    # 返回一个列表，包含多个 WarningInfo 对象，用于控制不同类型警告的处理方式
    return [
        # 创建一个 WarningInfo 对象，处理 DeprecationWarning 类别的警告
        WarningInfo("error", category=DeprecationWarning),
        # 创建一个 WarningInfo 对象，处理 FutureWarning 类别的警告
        WarningInfo("error", category=FutureWarning),
        # 创建一个 WarningInfo 对象，处理 VisibleDeprecationWarning 类别的警告
        WarningInfo("error", category=VisibleDeprecationWarning),
        # TODO: pyamg 版本大于 5.0.1 时移除此处设置
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 pyamg 中 pkg_resources 的使用而产生的警告
        WarningInfo(
            "ignore",
            message="pkg_resources is deprecated as an API",
            category=DeprecationWarning,
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 pyamg 中对 pkg_resources 的调用而产生的警告
        WarningInfo(
            "ignore",
            message="Deprecated call to `pkg_resources",
            category=DeprecationWarning,
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免 pytest-cov 中的特定问题
        WarningInfo(
            "ignore",
            message=(
                "The --rsyncdir command line argument and rsyncdirs config variable are"
                " deprecated"
            ),
            category=DeprecationWarning,
        ),
        # XXX: 临时忽略 pandas Pyarrow DeprecationWarning，参考 https://github.com/pandas-dev/pandas/issues/54466 了解更多详情
        WarningInfo(
            "ignore",
            message=r"\s*Pyarrow will become a required dependency",
            category=DeprecationWarning,
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 dateutil 中的问题而产生的警告
        WarningInfo(
            "ignore",
            message="datetime.datetime.utcfromtimestamp",
            category=DeprecationWarning,
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 joblib 中的问题而产生的警告
        WarningInfo(
            "ignore", message="ast.Num is deprecated", category=DeprecationWarning
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 joblib 中的问题而产生的警告
        WarningInfo(
            "ignore", message="Attribute n is deprecated", category=DeprecationWarning
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 sphinx-gallery 中的问题而产生的警告
        WarningInfo(
            "ignore", message="ast.Str is deprecated", category=DeprecationWarning
        ),
        # 创建一个 WarningInfo 对象，忽略特定的 DeprecationWarning，避免由于 sphinx-gallery 中的问题而产生的警告
        WarningInfo(
            "ignore", message="Attribute s is deprecated", category=DeprecationWarning
        ),
    ]
# 获取 Pytest 的警告过滤器信息列表
def get_pytest_filterwarning_lines():
    # 调用内部函数 _get_warnings_filters_info_list() 获取警告过滤器信息列表
    warning_filters_info_list = _get_warnings_filters_info_list()
    # 使用列表推导式将每个警告信息对象转换为过滤器警告字符串，并返回结果列表
    return [
        warning_info.to_filterwarning_str()
        for warning_info in warning_filters_info_list
    ]


# 将警告转换为错误处理
def turn_warnings_into_errors():
    # 调用内部函数 _get_warnings_filters_info_list() 获取警告过滤器信息列表
    warnings_filters_info_list = _get_warnings_filters_info_list()
    # 遍历警告过滤器信息列表，依次注册警告过滤器
    for warning_info in warnings_filters_info_list:
        warnings.filterwarnings(
            warning_info.action,        # 设置警告的动作（例如 'error' 表示将警告转为错误）
            message=warning_info.message,   # 指定警告消息文本
            category=warning_info.category,  # 指定警告类别
        )
```