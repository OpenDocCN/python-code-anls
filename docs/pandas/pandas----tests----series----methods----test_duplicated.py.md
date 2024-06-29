# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_duplicated.py`

```
# 导入所需的库和模块
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

from pandas import (  # 从 Pandas 中导入以下模块：
    NA,  # NA 值表示缺失值
    Categorical,  # 用于处理分类数据的类
    Series,  # Pandas 中的 Series 数据结构
)
import pandas._testing as tm  # 导入 Pandas 测试模块


@pytest.mark.parametrize(  # 使用 Pytest 的参数化测试装饰器，定义参数化测试用例
    "keep, expected",  # 参数包括 keep 和 expected
    [  # 参数化测试数据列表
        ("first", [False, False, True, False, True]),  # 第一组参数化测试数据
        ("last", [True, True, False, False, False]),  # 第二组参数化测试数据
        (False, [True, True, True, False, True]),  # 第三组参数化测试数据
    ],
)
def test_duplicated_keep(keep, expected):
    ser = Series(["a", "b", "b", "c", "a"], name="name")  # 创建一个 Series 对象

    result = ser.duplicated(keep=keep)  # 调用 Series 的 duplicated 方法进行测试
    expected = Series(expected, name="name")  # 创建预期的结果 Series 对象
    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的方法验证结果


@pytest.mark.parametrize(  # 使用 Pytest 的参数化测试装饰器，定义参数化测试用例
    "keep, expected",  # 参数包括 keep 和 expected
    [  # 参数化测试数据列表
        ("first", [False, False, True, False, True]),  # 第一组参数化测试数据
        ("last", [True, True, False, False, False]),  # 第二组参数化测试数据
        (False, [True, True, True, False, True]),  # 第三组参数化测试数据
    ],
)
def test_duplicated_nan_none(keep, expected):
    ser = Series([np.nan, 3, 3, None, np.nan], dtype=object)  # 创建一个包含 NaN 和 None 的 Series 对象

    result = ser.duplicated(keep=keep)  # 调用 Series 的 duplicated 方法进行测试
    expected = Series(expected)  # 创建预期的结果 Series 对象
    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的方法验证结果


def test_duplicated_categorical_bool_na(nulls_fixture):
    # GH#44351
    ser = Series(  # 创建一个包含分类数据的 Series 对象
        Categorical(  # 创建分类数据对象
            [True, False, True, False, nulls_fixture],  # 分类数据内容包括 True、False 和 nulls_fixture
            categories=[True, False],  # 分类的可能取值
            ordered=True,  # 分类是否有序
        )
    )
    result = ser.duplicated()  # 调用 Series 的 duplicated 方法进行测试
    expected = Series([False, False, True, True, False])  # 创建预期的结果 Series 对象
    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的方法验证结果


@pytest.mark.parametrize(  # 使用 Pytest 的参数化测试装饰器，定义参数化测试用例
    "keep, vals",  # 参数包括 keep 和 vals
    [  # 参数化测试数据列表
        ("last", [True, True, False]),  # 第一组参数化测试数据
        ("first", [False, True, True]),  # 第二组参数化测试数据
        (False, [True, True, True]),  # 第三组参数化测试数据
    ],
)
def test_duplicated_mask(keep, vals):
    # GH#48150
    ser = Series([1, 2, NA, NA, NA], dtype="Int64")  # 创建一个包含 NA 值的整数类型的 Series 对象
    result = ser.duplicated(keep=keep)  # 调用 Series 的 duplicated 方法进行测试
    expected = Series([False, False] + vals)  # 创建预期的结果 Series 对象
    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的方法验证结果


def test_duplicated_mask_no_duplicated_na(keep):
    # GH#48150
    ser = Series([1, 2, NA], dtype="Int64")  # 创建一个包含 NA 值的整数类型的 Series 对象
    result = ser.duplicated(keep=keep)  # 调用 Series 的 duplicated 方法进行测试
    expected = Series([False, False, False])  # 创建预期的结果 Series 对象
    tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块的方法验证结果
```