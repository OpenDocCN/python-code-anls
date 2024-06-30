# `D:\src\scipysrc\seaborn\tests\_core\test_rules.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析

import pytest  # 导入 pytest 库，用于编写和运行单元测试

from seaborn._core.rules import (  # 从 seaborn 库中导入相关函数和类
    VarType,  # 导入 VarType 类，用于定义变量类型
    variable_type,  # 导入 variable_type 函数，用于推断变量类型
    categorical_order,  # 导入 categorical_order 函数，用于获取分类变量的排序
)


def test_vartype_object():
    # 测试 VarType 类的初始化，断言 v 的值为 "numeric"
    v = VarType("numeric")
    assert v == "numeric"
    assert v != "categorical"
    # 断言 VarType 类不接受 "number" 的初始化参数，会抛出 AssertionError
    with pytest.raises(AssertionError):
        v == "number"
    # 断言 VarType 类不接受 "date" 的初始化参数，会抛出 AssertionError
    with pytest.raises(AssertionError):
        VarType("date")


def test_variable_type():
    # 创建一个包含浮点数的 Pandas Series
    s = pd.Series([1., 2., 3.])
    # 断言 variable_type 函数能正确推断出 Series s 的类型为 "numeric"
    assert variable_type(s) == "numeric"
    # 断言 variable_type 函数能正确推断出将整数类型的 Series 转换为 "numeric"
    assert variable_type(s.astype(int)) == "numeric"
    # 断言 variable_type 函数能正确推断出将对象类型的 Series 转换为 "numeric"
    assert variable_type(s.astype(object)) == "numeric"

    # 创建一个包含对象类型和 NaN 值的 Pandas Series
    s = pd.Series([1, 2, 3, np.nan], dtype=object)
    # 断言 variable_type 函数能正确推断出包含对象类型的 Series 的类型为 "numeric"
    assert variable_type(s) == "numeric"

    # 创建一个包含 NaN 值的 Pandas Series
    s = pd.Series([np.nan, np.nan])
    # 断言 variable_type 函数能正确推断出包含 NaN 值的 Series 的类型为 "numeric"
    assert variable_type(s) == "numeric"

    # 创建一个包含 pd.NA 值的 Pandas Series
    s = pd.Series([pd.NA, pd.NA])
    # 断言 variable_type 函数能正确推断出包含 pd.NA 值的 Series 的类型为 "numeric"
    assert variable_type(s) == "numeric"

    # 创建一个包含整数和 pd.NA 值的 Pandas Series，并指定 dtype 为 "Int64"
    s = pd.Series([1, 2, pd.NA], dtype="Int64")
    # 断言 variable_type 函数能正确推断出包含整数和 pd.NA 值的 Series 的类型为 "numeric"
    assert variable_type(s) == "numeric"

    # 创建一个包含整数和 pd.NA 值的 Pandas Series，并指定 dtype 为 object
    s = pd.Series([1, 2, pd.NA], dtype=object)
    # 断言 variable_type 函数能正确推断出包含整数和 pd.NA 值的 Series 的类型为 "numeric"
    assert variable_type(s) == "numeric"

    # 创建一个包含字符串的 Pandas Series
    s = pd.Series(["1", "2", "3"])
    # 断言 variable_type 函数能正确推断出包含字符串的 Series 的类型为 "categorical"
    assert variable_type(s) == "categorical"

    # 创建一个包含布尔值的 Pandas Series
    s = pd.Series([True, False, False])
    # 断言 variable_type 函数能正确推断出包含布尔值的 Series 的类型为 "numeric"
    assert variable_type(s) == "numeric"
    # 断言 variable_type 函数能正确推断出将布尔值视为分类变量时的类型为 "categorical"
    assert variable_type(s, boolean_type="categorical") == "categorical"
    # 断言 variable_type 函数能正确推断出将布尔值视为布尔类型时的类型为 "boolean"
    assert variable_type(s, boolean_type="boolean") == "boolean"

    # 创建一个时间间隔的 Pandas Series
    s = pd.timedelta_range(1, periods=3, freq="D").to_series()
    # 断言 variable_type 函数将时间间隔的 Series 推断为 "categorical"
    assert variable_type(s) == "categorical"

    # 将 Series 转换为分类类型
    s_cat = s.astype("category")
    # 断言 variable_type 函数能正确推断出分类类型的 Series 的类型为 "categorical"
    assert variable_type(s_cat, boolean_type="categorical") == "categorical"
    assert variable_type(s_cat, boolean_type="numeric") == "categorical"
    assert variable_type(s_cat, boolean_type="boolean") == "categorical"

    # 创建一个包含整数的 Pandas Series
    s = pd.Series([1, 0, 0])
    # 断言 variable_type 函数能正确推断出将布尔值视为布尔类型时的类型为 "boolean"
    assert variable_type(s, boolean_type="boolean") == "boolean"
    # 断言 variable_type 函数在 strict_boolean=True 时能正确推断出类型为 "numeric"
    assert variable_type(s, boolean_type="boolean", strict_boolean=True) == "numeric"

    # 创建一个包含整数的 Pandas Series
    s = pd.Series([1, 0, 0])
    # 断言 variable_type 函数能正确推断出将布尔值视为布尔类型时的类型为 "boolean"
    assert variable_type(s, boolean_type="boolean") == "boolean"

    # 创建一个包含 Timestamp 的 Pandas Series
    s = pd.Series([pd.Timestamp(1), pd.Timestamp(2)])
    # 断言 variable_type 函数能正确推断出包含 Timestamp 的 Series 的类型为 "datetime"
    assert variable_type(s) == "datetime"
    # 断言 variable_type 函数能正确推断出将对象类型的 Series 转换为 "datetime"
    assert variable_type(s.astype(object)) == "datetime"


def test_categorical_order():
    # 创建一个包含字符串的 Pandas Series
    x = pd.Series(["a", "c", "c", "b", "a", "d"])
    # 创建一个包含整数的 Pandas Series
    y = pd.Series([3, 2, 5, 1, 4])
    # 创建一个排序列表
    order = ["a", "b", "c", "d"]

    # 调用 categorical_order 函数获取 Series x 的分类顺序
    out = categorical_order(x)
    # 断言 categorical_order 函数返回的结果与预期一致
    assert out == ["a", "c", "b", "d"]

    # 使用指定的排序列表调用 categorical_order 函数获取 Series x 的分类顺序
    out = categorical_order(x, order)
    # 断言 categorical_order 函数返回的结果与预期一致
    assert out == order

    # 使用新的排序列表调用 categorical_order 函数获取 Series x 的分类顺序
    out = categorical_order(x, ["b", "a"])
    # 断言 categorical_order 函数返回的结果与预期一致
    assert out == ["b", "a"]

    # 调用 categorical_order 函数获取 Series y 的分类顺序
    out = categorical_order(y)
    # 断言 categorical_order 函数返回的结果与预期一致
    assert out == [1, 2, 3, 4, 5]

    # 调用 categorical_order 函数获取 pd.Series(y) 的分类顺序
    out = categorical_order(pd.Series(y))
    # 断言 categorical_order 函数返回的结果与预期一致
    assert out == [1, 2, 3, 4, 5]

    # 将 Series y 转换为分类类型后调用 categorical_order 函数获取分类顺序
    y_cat = pd.Series(pd.Categorical(y, y))
    out = categorical_order(y_cat)
    # 断言 categorical_order 函数返回的结果与预期一致
    assert out == list(y)

    # 将 Series x 转换为分类类型后调用 categorical_order 函数获取分类顺序
    x = pd.Series(x).astype("category")
    out = categorical_order(x)
    # 断言 categorical_order 函数返回的结果与预期一致
    assert
    # 确保变量 `out` 等于 `x.cat.categories` 转换为列表后的结果
    assert out == list(x.cat.categories)
    
    # 调用函数 `categorical_order`，传入参数 `x` 和指定的顺序列表 ["b", "a"]，并将返回结果赋给 `out`
    out = categorical_order(x, ["b", "a"])
    # 确保 `out` 等于预期的顺序列表 ["b", "a"]
    assert out == ["b", "a"]
    
    # 创建一个 Pandas Series 对象 `x`，包含指定的数据 ["a", np.nan, "c", "c", "b", "a", "d"]
    x = pd.Series(["a", np.nan, "c", "c", "b", "a", "d"])
    # 调用函数 `categorical_order`，传入参数 `x`，并将返回结果赋给 `out`
    out = categorical_order(x)
    # 确保 `out` 等于预期的顺序列表 ["a", "c", "b", "d"]
    assert out == ["a", "c", "b", "d"]
```