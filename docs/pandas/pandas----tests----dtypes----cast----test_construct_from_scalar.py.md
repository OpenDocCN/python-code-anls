# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_construct_from_scalar.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar  # 从 Pandas 库中导入单个标量转换为数组的函数
from pandas.core.dtypes.dtypes import CategoricalDtype  # 从 Pandas 库中导入分类数据类型

from pandas import (  # 从 Pandas 库中导入多个模块
    Categorical,  # 导入分类数据类型
    Timedelta,  # 导入时间增量类型
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


def test_cast_1d_array_like_from_scalar_categorical():
    # 测试函数：test_cast_1d_array_like_from_scalar_categorical
    # 查看 GitHub issue-19565
    #
    # 当从标量创建的分类数据未能保持
    # 类别和传递数据类型的顺序。
    cats = ["a", "b", "c"]  # 定义分类的类别列表
    cat_type = CategoricalDtype(categories=cats, ordered=False)  # 创建分类数据类型对象，指定类别和无序
    expected = Categorical(["a", "a"], categories=cats)  # 创建预期的分类数据对象

    result = construct_1d_arraylike_from_scalar("a", len(expected), cat_type)  # 使用标量创建一维数组，转换为分类数据类型
    tm.assert_categorical_equal(result, expected)  # 断言：验证结果与预期是否相等


def test_cast_1d_array_like_from_timestamp(fixed_now_ts):
    # 测试函数：test_cast_1d_array_like_from_timestamp
    # 检查不会丢失纳秒级精度
    ts = fixed_now_ts + Timedelta(1)  # 创建一个时间增量对象
    res = construct_1d_arraylike_from_scalar(ts, 2, np.dtype("M8[ns]"))  # 使用时间戳创建一维数组，指定数据类型为纳秒级时间
    assert res[0] == ts  # 断言：验证结果的第一个元素与预期时间戳相等


def test_cast_1d_array_like_from_timedelta():
    # 测试函数：test_cast_1d_array_like_from_timedelta
    # 检查不会丢失纳秒级精度
    td = Timedelta(1)  # 创建一个时间增量对象
    res = construct_1d_arraylike_from_scalar(td, 2, np.dtype("m8[ns]"))  # 使用时间增量创建一维数组，指定数据类型为纳秒级时间
    assert res[0] == td  # 断言：验证结果的第一个元素与预期时间增量对象相等


def test_cast_1d_array_like_mismatched_datetimelike():
    # 测试函数：test_cast_1d_array_like_mismatched_datetimelike
    td = np.timedelta64("NaT", "ns")  # 创建一个纳秒级空时间增量对象
    dt = np.datetime64("NaT", "ns")  # 创建一个纳秒级空日期时间对象

    with pytest.raises(TypeError, match="Cannot cast"):  # 使用 pytest 断言捕获 TypeError 异常，匹配指定错误信息
        construct_1d_arraylike_from_scalar(td, 2, dt.dtype)  # 尝试从时间增量对象转换为日期时间对象

    with pytest.raises(TypeError, match="Cannot cast"):  # 使用 pytest 断言捕获 TypeError 异常，匹配指定错误信息
        construct_1d_arraylike_from_scalar(np.timedelta64(4, "ns"), 2, dt.dtype)  # 尝试从时间增量对象转换为日期时间对象

    with pytest.raises(TypeError, match="Cannot cast"):  # 使用 pytest 断言捕获 TypeError 异常，匹配指定错误信息
        construct_1d_arraylike_from_scalar(dt, 2, td.dtype)  # 尝试从日期时间对象转换为时间增量对象

    with pytest.raises(TypeError, match="Cannot cast"):  # 使用 pytest 断言捕获 TypeError 异常，匹配指定错误信息
        construct_1d_arraylike_from_scalar(np.datetime64(4, "ns"), 2, td.dtype)  # 尝试从日期时间对象转换为时间增量对象
```