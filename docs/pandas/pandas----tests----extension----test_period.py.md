# `D:\src\scipysrc\pandas\pandas\tests\extension\test_period.py`

```
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""

# 引入将来版本兼容性模块
from __future__ import annotations

# 引入类型检查标记
from typing import TYPE_CHECKING

# 引入 NumPy 库
import numpy as np
# 引入 Pytest 测试框架
import pytest

# 引入 Pandas 核心库中的 Period 和 iNaT
from pandas._libs import (
    Period,
    iNaT,
)
# 引入跨平台兼容性检查工具
from pandas.compat import is_platform_windows
# 引入 NumPy 版本检查工具
from pandas.compat.numpy import np_version_gte1p24

# 引入 Pandas 核心数据类型中的 PeriodDtype
from pandas.core.dtypes.dtypes import PeriodDtype

# 引入 Pandas 测试工具模块
import pandas._testing as tm
# 引入 Pandas 核心数组模块中的 PeriodArray
from pandas.core.arrays import PeriodArray
# 引入扩展测试基类
from pandas.tests.extension import base

# 如果类型检查为真，引入 Pandas 核心库
if TYPE_CHECKING:
    import pandas as pd


# 定义夹具函数，返回特定频率的 PeriodDtype 对象
@pytest.fixture(params=["D", "2D"])
def dtype(request):
    return PeriodDtype(freq=request.param)


# 定义夹具函数，返回特定数据的 PeriodArray 对象
@pytest.fixture
def data(dtype):
    return PeriodArray(np.arange(1970, 2070), dtype=dtype)


# 定义夹具函数，返回用于排序的 PeriodArray 对象
@pytest.fixture
def data_for_sorting(dtype):
    return PeriodArray([2018, 2019, 2017], dtype=dtype)


# 定义夹具函数，返回带有缺失值的 PeriodArray 对象
@pytest.fixture
def data_missing(dtype):
    return PeriodArray([iNaT, 2017], dtype=dtype)


# 定义夹具函数，返回用于排序的带有缺失值的 PeriodArray 对象
@pytest.fixture
def data_missing_for_sorting(dtype):
    return PeriodArray([2018, iNaT, 2017], dtype=dtype)


# 定义夹具函数，返回用于分组的 PeriodArray 对象
@pytest.fixture
def data_for_grouping(dtype):
    B = 2018
    NA = iNaT
    A = 2017
    C = 2019
    return PeriodArray([B, B, NA, NA, A, A, B, C], dtype=dtype)


# 测试类，继承自 base.ExtensionTests 基类
class TestPeriodArray(base.ExtensionTests):

    # 覆盖父类方法，返回期望的异常对象
    def _get_expected_exception(self, op_name, obj, other):
        if op_name in ("__sub__", "__rsub__"):
            return None
        return super()._get_expected_exception(op_name, obj, other)

    # 检查是否支持累积操作
    def _supports_accumulation(self, ser, op_name: str) -> bool:
        return op_name in ["cummin", "cummax"]

    # 检查是否支持缩减操作
    def _supports_reduction(self, obj, op_name: str) -> bool:
        return op_name in ["min", "max", "median"]

    # 检查缩减操作，特别是中位数操作
    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        if op_name == "median":
            # 调用中位数操作
            res_op = getattr(ser, op_name)

            # 将序列转换为整型序列
            alt = ser.astype("int64")

            # 获取整型序列的中位数操作
            exp_op = getattr(alt, op_name)

            # 执行结果和期望的中位数比较
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
            
            # 获取序列的频率属性
            freq = ser.dtype.freq  # type: ignore[union-attr]
            
            # 根据序列的频率属性创建 Period 对象
            expected = Period._from_ordinal(int(expected), freq=freq)
            
            # 使用测试工具函数检查结果几乎相等
            tm.assert_almost_equal(result, expected)

        else:
            # 调用父类的缩减操作检查方法
            return super().check_reduce(ser, op_name, skipna)

    # 参数化测试，使用不同的期数值进行参数化
    @pytest.mark.parametrize("periods", [1, -2])
    # 定义测试方法，用于比较数据差异
    def test_diff(self, data, periods):
        # 如果当前平台是 Windows 且 NumPy 版本大于等于 1.24
        if is_platform_windows() and np_version_gte1p24:
            # 断言期望产生 RuntimeWarning 警告，忽略栈级别检查
            with tm.assert_produces_warning(RuntimeWarning, check_stacklevel=False):
                # 调用父类的 test_diff 方法进行数据差异比较
                super().test_diff(data, periods)
        else:
            # 否则，直接调用父类的 test_diff 方法
            super().test_diff(data, periods)

    # 使用 pytest 的参数化装饰器，定义测试方法 test_map
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        # 对数据进行映射操作，传入的映射函数为 lambda 函数，na_action 参数根据装饰器参数化设置
        result = data.map(lambda x: x, na_action=na_action)
        # 断言扩展数组的相等性
        tm.assert_extension_array_equal(result, data)
# 定义一个测试类 Test2DCompat，继承自 base.NDArrayBacked2DTests
class Test2DCompat(base.NDArrayBacked2DTests):
    # pass 语句表示此类没有额外定义，直接继承父类的测试用例
    pass
```