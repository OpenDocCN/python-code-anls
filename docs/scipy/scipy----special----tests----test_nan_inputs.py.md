# `D:\src\scipysrc\scipy\scipy\special\tests\test_nan_inputs.py`

```
"""Test how the ufuncs in special handle nan inputs.

"""
# 引入必要的模块和函数
from typing import Callable  # 引入Callable类型的支持
import numpy as np  # 引入NumPy库，用于数值计算
from numpy.testing import assert_array_equal, assert_, suppress_warnings  # 引入NumPy测试模块中的函数和类
import pytest  # 引入pytest测试框架
import scipy.special as sc  # 引入SciPy库中的special模块，用于特殊函数计算


KNOWNFAILURES: dict[str, Callable] = {}  # 初始化已知失败的测试用例字典

POSTPROCESSING: dict[str, Callable] = {}  # 初始化后处理函数字典


def _get_ufuncs():
    # 获取所有在scipy.special模块中的ufunc函数
    ufuncs = []
    ufunc_names = []
    for name in sorted(sc.__dict__):
        obj = sc.__dict__[name]
        # 如果不是NumPy的ufunc对象，则跳过
        if not isinstance(obj, np.ufunc):
            continue
        # 检查是否有已知的失败情况，如果有则标记为xfail
        msg = KNOWNFAILURES.get(obj)
        if msg is None:
            ufuncs.append(obj)
            ufunc_names.append(name)
        else:
            fail = pytest.mark.xfail(run=False, reason=msg)
            ufuncs.append(pytest.param(obj, marks=fail))
            ufunc_names.append(name)
    return ufuncs, ufunc_names


UFUNCS, UFUNC_NAMES = _get_ufuncs()  # 获取所有ufunc函数及其名称


@pytest.mark.parametrize("func", UFUNCS, ids=UFUNC_NAMES)
def test_nan_inputs(func):
    # 构造所有参数为NaN的元组
    args = (np.nan,) * func.nin
    with suppress_warnings() as sup:
        # 忽略关于旧包装器的不安全类型转换的警告
        sup.filter(RuntimeWarning, "floating point number truncated to an integer")
        try:
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                res = func(*args)  # 调用ufunc函数并传入NaN参数
        except TypeError:
            # 如果参数类型错误，则直接返回
            return
    if func in POSTPROCESSING:
        # 如果有后处理函数，则对结果进行后处理
        res = POSTPROCESSING[func](*res)

    msg = f"got {res} instead of nan"
    # 断言结果应为NaN
    assert_array_equal(np.isnan(res), True, err_msg=msg)


def test_legacy_cast():
    with suppress_warnings() as sup:
        # 忽略关于旧包装器的不安全类型转换的警告
        sup.filter(RuntimeWarning, "floating point number truncated to an integer")
        res = sc.bdtrc(np.nan, 1, 0.5)  # 调用特定的bdtrc函数，传入NaN参数
        assert_(np.isnan(res))  # 断言结果应为NaN
```