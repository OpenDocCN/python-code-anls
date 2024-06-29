# `D:\src\scipysrc\pandas\pandas\tests\arrays\period\test_astype.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 PyTest 库

# 导入 Pandas 相关模块和类
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import period_array

# 参数化测试函数，测试不同的数据类型转换
@pytest.mark.parametrize("dtype", [int, np.int32, np.int64, "uint32", "uint64"])
def test_astype_int(dtype):
    # 对于 Period/Datetime/Timedelta 的 astype，我们选择忽略整数的符号和大小
    arr = period_array(["2000", "2001", None], freq="D")

    # 如果 dtype 不是 np.int64 类型
    if np.dtype(dtype) != np.int64:
        # 断言抛出 TypeError，并匹配错误消息中是否包含 "Do obj.astype('int64')"
        with pytest.raises(TypeError, match=r"Do obj.astype\('int64'\)"):
            arr.astype(dtype)
        return

    # 执行 astype 转换
    result = arr.astype(dtype)
    # 期望的结果是将 arr 转换为 np.int64 类型的视图
    expected = arr._ndarray.view("i8")
    # 断言结果是否符合预期
    tm.assert_numpy_array_equal(result, expected)


# 测试不同的拷贝策略
def test_astype_copies():
    arr = period_array(["2000", "2001", None], freq="D")
    # 使用 copy=False 参数进行类型转换
    result = arr.astype(np.int64, copy=False)

    # 断言 result 的基础数据是 arr._ndarray
    assert result.base is arr._ndarray

    # 使用 copy=True 参数进行类型转换
    result = arr.astype(np.int64, copy=True)
    # 断言 result 不是 arr._ndarray 的引用
    assert result is not arr._ndarray
    # 断言结果是否符合预期
    tm.assert_numpy_array_equal(result, arr._ndarray.view("i8"))


# 测试类型转换为分类类型
def test_astype_categorical():
    arr = period_array(["2000", "2001", "2001", None], freq="D")
    # 执行类型转换为 'category'
    result = arr.astype("category")
    # 创建预期的分类对象
    categories = pd.PeriodIndex(["2000", "2001"], freq="D")
    expected = pd.Categorical.from_codes([0, 1, 1, -1], categories=categories)
    # 断言结果是否符合预期
    tm.assert_categorical_equal(result, expected)


# 测试类型转换为 PeriodDtype
def test_astype_period():
    arr = period_array(["2000", "2001", None], freq="D")
    # 执行类型转换为 PeriodDtype("M")
    result = arr.astype(PeriodDtype("M"))
    # 创建预期的 PeriodArray 对象
    expected = period_array(["2000", "2001", None], freq="M")
    # 断言结果是否符合预期
    tm.assert_period_array_equal(result, expected)


# 参数化测试函数，测试不同的 datetime 类型转换
@pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
def test_astype_datetime(dtype):
    arr = period_array(["2000", "2001", None], freq="D")
    # 如果 dtype 是 "timedelta64[ns]"，则切掉 "[ns]" 部分以匹配正则表达式
    if dtype == "timedelta64[ns]":
        with pytest.raises(TypeError, match=dtype[:-4]):
            arr.astype(dtype)
    else:
        # 对于其他 datetime 类型，执行类型转换
        result = arr.astype(dtype)
        # 创建预期的 DatetimeIndex 对象
        expected = pd.DatetimeIndex(["2000", "2001", pd.NaT], dtype=dtype)._data
        # 断言结果是否符合预期
        tm.assert_datetime_array_equal(result, expected)
```