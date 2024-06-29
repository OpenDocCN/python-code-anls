# `D:\src\scipysrc\pandas\pandas\tests\strings\test_string_array.py`

```
# 导入必要的库
import numpy as np
import pytest

# 从 pandas._libs 库中导入 lib 模块
from pandas._libs import lib

# 从 pandas 库中导入以下模块
from pandas import (
    NA,
    DataFrame,
    Series,
    _testing as tm,
    option_context,
)

# 标记测试以忽略特定警告信息
@pytest.mark.filterwarnings("ignore:Falling back")
def test_string_array(nullable_string_dtype, any_string_method):
    # 解包 any_string_method 元组中的方法名、参数和关键字参数
    method_name, args, kwargs = any_string_method

    # 创建包含字符串和缺失值的数据列表
    data = ["a", "bb", np.nan, "ccc"]
    # 创建 Series 对象 a，指定对象类型为 object
    a = Series(data, dtype=object)
    # 创建 Series 对象 b，指定对象类型为 nullable_string_dtype
    b = Series(data, dtype=nullable_string_dtype)

    # 如果方法名为 "decode"，则验证是否会引发 TypeError 异常
    if method_name == "decode":
        with pytest.raises(TypeError, match="a bytes-like object is required"):
            getattr(b.str, method_name)(*args, **kwargs)
        return

    # 获取预期结果，调用 a.str 对象的 method_name 方法，并传入参数和关键字参数
    expected = getattr(a.str, method_name)(*args, **kwargs)
    # 获取实际结果，调用 b.str 对象的 method_name 方法，并传入参数和关键字参数
    result = getattr(b.str, method_name)(*args, **kwargs)

    # 如果预期结果是 Series 对象
    if isinstance(expected, Series):
        # 如果预期结果的 dtype 是 "object" 并且 lib.is_string_array 判断为真
        if expected.dtype == "object" and lib.is_string_array(
            expected.dropna().values,
        ):
            # 验证结果的 dtype 是否为 nullable_string_dtype
            assert result.dtype == nullable_string_dtype
            # 将结果转换为 object 类型
            result = result.astype(object)

        # 如果预期结果的 dtype 是 "object" 并且 lib.is_bool_array 判断为真
        elif expected.dtype == "object" and lib.is_bool_array(
            expected.values, skipna=True
        ):
            # 验证结果的 dtype 是否为 "boolean"
            assert result.dtype == "boolean"
            # 将结果转换为 object 类型
            result = result.astype(object)

        # 如果预期结果的 dtype 是 "bool"
        elif expected.dtype == "bool":
            # 验证结果的 dtype 是否为 "boolean"
            assert result.dtype == "boolean"
            # 将结果转换为 "bool" 类型
            result = result.astype("bool")

        # 如果预期结果的 dtype 是 "float" 并且存在缺失值
        elif expected.dtype == "float" and expected.isna().any():
            # 验证结果的 dtype 是否为 "Int64"
            assert result.dtype == "Int64"
            # 将结果转换为 "float" 类型
            result = result.astype("float")

        # 如果预期结果的 dtype 是 object 类型
        if expected.dtype == object:
            # GH#18463 注释：在预期结果中填充缺失值 NA

    # 如果预期结果是 DataFrame 对象
    elif isinstance(expected, DataFrame):
        # 选择预期结果中 dtype 是 "object" 的列名
        columns = expected.select_dtypes(include="object").columns
        # 验证结果中所有列的 dtype 是否为 nullable_string_dtype
        assert all(result[columns].dtypes == nullable_string_dtype)
        # 将结果中的列转换为 object 类型
        result[columns] = result[columns].astype(object)
        # 在选项上下文中启用特定设置，设置未来下转型时不发出警告
        with option_context("future.no_silent_downcasting", True):
            # 在预期结果中的列上填充缺失值 NA，GH#18463 注释
            expected[columns] = expected[columns].fillna(NA)

    # 使用测试框架验证结果是否与预期一致
    tm.assert_equal(result, expected)


# 使用参数化测试来测试不同的方法和预期结果
@pytest.mark.parametrize(
    "method,expected",
    [
        ("count", [2, None]),
        ("find", [0, None]),
        ("index", [0, None]),
        ("rindex", [2, None]),
    ],
)
def test_string_array_numeric_integer_array(nullable_string_dtype, method, expected):
    # 创建 Series 对象 s，包含字符串和缺失值，指定对象类型为 nullable_string_dtype
    s = Series(["aba", None], dtype=nullable_string_dtype)
    # 调用 s.str 对象的 method 方法，传入参数 "a"
    result = getattr(s.str, method)("a")
    # 创建预期结果 Series 对象，指定对象类型为 "Int64"
    expected = Series(expected, dtype="Int64")
    # 使用测试框架验证结果是否与预期一致
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试不同的方法和预期结果
@pytest.mark.parametrize(
    "method,expected",
    [
        ("isdigit", [False, None, True]),
        ("isalpha", [True, None, False]),
        ("isalnum", [True, None, True]),
        ("isnumeric", [False, None, True]),
    ],
)
def test_string_array_boolean_array(nullable_string_dtype, method, expected):
    # 创建 Series 对象 s，包含字符串和缺失值，指定对象类型为 nullable_string_dtype
    s = Series(["a", None, "1"], dtype=nullable_string_dtype)
    # 调用 s.str 对象的 method 方法
    result = getattr(s.str, method)()
    # 创建预期结果 Series 对象，指定对象类型为 "boolean"
    expected = Series(expected, dtype="boolean")
    # 使用测试框架验证结果是否与预期一致
    tm.assert_series_equal(result, expected)
def test_string_array_extract(nullable_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/30969
    # 仅当 expand=False 和多个组时才会失败，这是一个已知的问题

    # 创建包含特定数据类型的 Pandas Series 对象 a
    a = Series(["a1", "b2", "cc"], dtype=nullable_string_dtype)
    # 创建包含对象类型的 Pandas Series 对象 b
    b = Series(["a1", "b2", "cc"], dtype="object")
    # 设置正则表达式模式
    pat = r"(\w)(\d)"

    # 对 Series a 应用正则表达式模式，不扩展结果
    result = a.str.extract(pat, expand=False)
    # 对 Series b 应用相同的正则表达式模式，不扩展结果
    expected = b.str.extract(pat, expand=False)
    # 将 NaN 值填充为 NA（可能是处理缺失值的特定标识）
    expected = expected.fillna(NA)  # GH#18463
    # 断言结果的数据类型符合 nullable_string_dtype
    assert all(result.dtypes == nullable_string_dtype)

    # 将结果转换为对象类型
    result = result.astype(object)
    # 使用测试框架中的 assert_equal 函数比较 result 和 expected
    tm.assert_equal(result, expected)
```