# `D:\src\scipysrc\scipy\scipy\special\_testutils.py`

```
# 导入必要的模块
import os  # 导入操作系统模块
import functools  # 导入 functools 模块
import operator  # 导入 operator 模块
from scipy._lib import _pep440  # 导入 scipy._lib 中的 _pep440 模块

# 导入 NumPy 库并将其命名为 np
import numpy as np
# 从 NumPy.testing 模块中导入 assert_ 函数
from numpy.testing import assert_
# 导入 pytest 库
import pytest

# 导入 scipy.special 库并将其命名为 sc
import scipy.special as sc

# 定义模块的公共接口
__all__ = ['with_special_errors', 'assert_func_equal', 'FuncData']

#------------------------------------------------------------------------------
# 检查测试中是否存在特定模块
#------------------------------------------------------------------------------

class MissingModule:
    def __init__(self, name):
        self.name = name

# 检查模块的版本是否符合要求
def check_version(module, min_ver):
    if type(module) == MissingModule:
        return pytest.mark.skip(reason=f"{module.name} is not installed")
    return pytest.mark.skipif(
        _pep440.parse(module.__version__) < _pep440.Version(min_ver),
        reason=f"{module.__name__} version >= {min_ver} required"
    )

#------------------------------------------------------------------------------
# 启用收敛和精度丢失警告 -- 逐个关闭
#------------------------------------------------------------------------------

def with_special_errors(func):
    """
    启用特殊函数错误（如下溢、上溢、精度丢失等）
    """
    @functools.wraps(func)
    def wrapper(*a, **kw):
        with sc.errstate(all='raise'):
            res = func(*a, **kw)
        return res
    return wrapper

#------------------------------------------------------------------------------
# 在许多数据点上比较函数值，并提供有用的错误报告
#------------------------------------------------------------------------------

def assert_func_equal(func, results, points, rtol=None, atol=None,
                      param_filter=None, knownfailure=None,
                      vectorized=True, dtype=None, nan_ok=False,
                      ignore_inf_sign=False, distinguish_nan_and_inf=True):
    if hasattr(points, 'next'):
        # 如果 points 是一个生成器，则转换为列表
        points = list(points)

    # 将 points 转换为 NumPy 数组
    points = np.asarray(points)
    # 如果 points 是一维数组，则转换为二维数组
    if points.ndim == 1:
        points = points[:,None]
    nparams = points.shape[1]

    if hasattr(results, '__name__'):
        # 如果 results 具有 '__name__' 属性，表明是函数
        data = points
        result_columns = None
        result_func = results
    else:
        # 否则认为是数据集
        data = np.c_[points, results]
        result_columns = list(range(nparams, data.shape[1]))
        result_func = None

    # 创建 FuncData 对象
    fdata = FuncData(func, data, list(range(nparams)),
                     result_columns=result_columns, result_func=result_func,
                     rtol=rtol, atol=atol, param_filter=param_filter,
                     knownfailure=knownfailure, nan_ok=nan_ok, vectorized=vectorized,
                     ignore_inf_sign=ignore_inf_sign,
                     distinguish_nan_and_inf=distinguish_nan_and_inf)
    # 执行检查函数
    fdata.check()

class FuncData:
    """
    用于检查特殊函数的数据集。

    Parameters
    ----------
    func : function
        要测试的函数
    """
    data : numpy array
        # 用于测试的列数据
        columnar data to use for testing

    param_columns : int or tuple of ints
        # 函数 `func` 参数所在的列索引。
        # 可以是虚数索引，表示参数应转换为复数。
        Columns indices in which the parameters to `func` lie.
        Can be imaginary integers to indicate that the parameter
        should be cast to complex.

    result_columns : int or tuple of ints, optional
        # `func` 的预期结果所在的列索引。
        Column indices for expected results from `func`.
    
    result_func : callable, optional
        # 获取结果的函数。
        Function to call to obtain results.

    rtol : float, optional
        # 所需的相对容差。默认为 5*eps。
        Required relative tolerance. Default is 5*eps.

    atol : float, optional
        # 所需的绝对容差。默认为 5*tiny。
        Required absolute tolerance. Default is 5*tiny.

    param_filter : function, or tuple of functions/Nones, optional
        # 用于排除某些参数范围的过滤函数。
        # 如果省略，则不执行过滤。
        Filter functions to exclude some parameter ranges.
        If omitted, no filtering is done.

    knownfailure : str, optional
        # 在运行测试时引发的已知失败错误消息。
        # 如果省略，则不引发异常。
        Known failure error message to raise when the test is run.
        If omitted, no exception is raised.

    nan_ok : bool, optional
        # 是否始终接受 NaN 作为结果。
        If nan is always an accepted result.

    vectorized : bool, optional
        # 所有传入函数是否都是矢量化的。
        Whether all functions passed in are vectorized.

    ignore_inf_sign : bool, optional
        # 是否忽略无穷大的符号。
        # （对于复值函数无关紧要。）
        Whether to ignore signs of infinities.
        (Doesn't matter for complex-valued functions.)

    distinguish_nan_and_inf : bool, optional
        # 如果为 True，则将包含 NaN 或 Inf 的数字视为相等。
        # 设置 ignore_inf_sign 为 True。
        If True, treat numbers which contain nans or infs as
        equal. Sets ignore_inf_sign to be True.
    # 返回给定数据类型的容差参数
    def get_tolerances(self, dtype):
        # 如果给定的数据类型不是浮点数类型，则将其转换为浮点数类型
        if not np.issubdtype(dtype, np.inexact):
            dtype = np.dtype(float)
        # 获取给定数据类型的数值范围信息
        info = np.finfo(dtype)
        # 设置相对容差 rtol 和绝对容差 atol 初始值
        rtol, atol = self.rtol, self.atol
        # 如果 rtol 为 None，则设置为 5 倍的数据类型的机器精度
        if rtol is None:
            rtol = 5 * info.eps
        # 如果 atol 为 None，则设置为 5 倍的数据类型的最小正数
        if atol is None:
            atol = 5 * info.tiny
        # 返回计算得到的相对容差 rtol 和绝对容差 atol
        return rtol, atol

    # 返回对象的字符串表示形式，用于美观打印，特别是用于 Nose 测试框架输出
    def __repr__(self):
        # 检查参数列中是否有复数对象
        if np.any(list(map(np.iscomplexobj, self.param_columns))):
            is_complex = " (complex)"
        else:
            is_complex = ""
        # 如果存在数据名称，则返回包含函数名、是否复数以及数据名称的字符串表示形式
        if self.dataname:
            return "<Data for {}{}: {}>".format(self.func.__name__, is_complex,
                                                os.path.basename(self.dataname))
        # 否则，只返回包含函数名和是否复数的字符串表示形式
        else:
            return f"<Data for {self.func.__name__}{is_complex}>"
```