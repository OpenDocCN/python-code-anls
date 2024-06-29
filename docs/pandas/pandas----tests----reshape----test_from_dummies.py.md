# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_from_dummies.py`

```
# 导入 NumPy 库并使用别名 np
import numpy as np
# 导入 pytest 库进行单元测试
import pytest

# 从 pandas 库中导入 DataFrame、Series、from_dummies 和 get_dummies 函数
from pandas import (
    DataFrame,
    Series,
    from_dummies,
    get_dummies,
)
# 导入 pandas 内部测试工具模块
import pandas._testing as tm

# 定义一个 pytest fixture，返回一个基本的虚拟 DataFrame，包含多列数据
@pytest.fixture
def dummies_basic():
    return DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )

# 定义一个 pytest fixture，返回一个带有未分配值的虚拟 DataFrame
@pytest.fixture
def dummies_with_unassigned():
    return DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )

# 定义测试函数 test_error_wrong_data_type，测试当传入的数据类型不符合预期时是否会抛出 TypeError 异常
def test_error_wrong_data_type():
    # 准备测试数据
    dummies = [0, 1, 0]
    # 断言抛出 TypeError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        TypeError,
        match=r"Expected 'data' to be a 'DataFrame'; Received 'data' of type: list",
    ):
        from_dummies(dummies)

# 定义测试函数 test_error_no_prefix_contains_unassigned，测试当 DataFrame 中存在未分配值时是否会抛出 ValueError 异常
def test_error_no_prefix_contains_unassigned():
    # 准备测试数据
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    # 断言抛出 ValueError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)

# 定义测试函数 test_error_no_prefix_wrong_default_category_type，测试当传入的默认类别参数类型不符合预期时是否会抛出 TypeError 异常
def test_error_no_prefix_wrong_default_category_type():
    # 准备测试数据
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    # 断言抛出 TypeError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        from_dummies(dummies, default_category=["c", "d"])

# 定义测试函数 test_error_no_prefix_multi_assignment，测试当 DataFrame 中存在多重分配值时是否会抛出 ValueError 异常
def test_error_no_prefix_multi_assignment():
    # 准备测试数据
    dummies = DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
    # 断言抛出 ValueError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 2"
        ),
    ):
        from_dummies(dummies)

# 定义测试函数 test_error_no_prefix_contains_nan，测试当 DataFrame 中存在 NaN 值时是否会抛出 ValueError 异常
def test_error_no_prefix_contains_nan():
    # 准备测试数据
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, np.nan]})
    # 断言抛出 ValueError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'b'"
    ):
        from_dummies(dummies)

# 定义测试函数 test_error_contains_non_dummies，测试当 DataFrame 中包含非虚拟变量数据时是否会抛出 TypeError 异常
def test_error_contains_non_dummies():
    # 准备测试数据
    dummies = DataFrame(
        {"a": [1, 6, 3, 1], "b": [0, 1, 0, 2], "c": ["c1", "c2", "c3", "c4"]}
    )
    # 断言抛出 TypeError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        TypeError,
        match=r"Passed DataFrame contains non-dummy data",
    ):
        from_dummies(dummies)

# 定义测试函数 test_error_with_prefix_multiple_seperators，测试当指定了错误的分隔符时是否会抛出 ValueError 异常
def test_error_with_prefix_multiple_seperators():
    # 准备测试数据
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [0, 1, 0],
            "col2-a": [0, 1, 0],
            "col2-b": [1, 0, 1],
        },
    )
    # 断言抛出 ValueError 异常，并验证异常信息中包含特定文本
    with pytest.raises(
        ValueError,
        match=(r"Separator not specified for column: col2-a"),
    ):
        from_dummies(dummies, sep="_")

# 定义测试函数 test_error_with_prefix_sep_wrong_type，测试当传入的分隔符参数类型不符合预期时是否会抛出 TypeError 异常
def test_error_with_prefix_sep_wrong_type(dummies_basic):
    # 使用 pytest.raises 上下文管理器来捕获 TypeError 异常，并检查其匹配的错误消息
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'sep' to be of type 'str' or 'None'; "
            r"Received 'sep' of type: list"
        ),
    ):
        # 调用 from_dummies 函数，并传入 dummies_basic 和 sep=["_"] 作为参数
        from_dummies(dummies_basic, sep=["_"])
# 测试函数：测试当输入的 DataFrame 包含未赋值的值时是否会触发 ValueError 异常
def test_error_with_prefix_contains_unassigned(dummies_with_unassigned):
    # 使用 pytest 来检查是否会引发 ValueError 异常，并检查异常信息中的内容匹配特定模式
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains unassigned value\(s\); "
            r"First instance in row: 2"
        ),
    ):
        # 调用 from_dummies 函数来处理带有未赋值的 DataFrame，并指定分隔符为 "_"
        from_dummies(dummies_with_unassigned, sep="_")


# 测试函数：测试当指定的 'default_category' 类型错误时是否会触发 TypeError 异常
def test_error_with_prefix_default_category_wrong_type(dummies_with_unassigned):
    # 使用 pytest 来检查是否会引发 TypeError 异常，并检查异常信息中的内容匹配特定模式
    with pytest.raises(
        TypeError,
        match=(
            r"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; "
            r"Received 'default_category' of type: list"
        ),
    ):
        # 调用 from_dummies 函数来处理带有未赋值的 DataFrame，指定分隔符为 "_"，
        # 并传入一个错误类型的 'default_category' 参数
        from_dummies(dummies_with_unassigned, sep="_", default_category=["x", "y"])


# 测试函数：测试当 'default_category' 字典长度不匹配时是否会触发 ValueError 异常
def test_error_with_prefix_default_category_dict_not_complete(
    dummies_with_unassigned,
):
    # 使用 pytest 来检查是否会引发 ValueError 异常，并检查异常信息中的内容匹配特定模式
    with pytest.raises(
        ValueError,
        match=(
            r"Length of 'default_category' \(1\) did not match "
            r"the length of the columns being encoded \(2\)"
        ),
    ):
        # 调用 from_dummies 函数来处理带有未赋值的 DataFrame，指定分隔符为 "_"，
        # 并传入一个长度不匹配的 'default_category' 参数
        from_dummies(dummies_with_unassigned, sep="_", default_category={"col1": "x"})


# 测试函数：测试当 DataFrame 中包含 NaN 值时是否会触发 ValueError 异常
def test_error_with_prefix_contains_nan(dummies_basic):
    # 将 "col2_c" 列转换为 float64 类型，以避免设置 np.nan 时的类型转换
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype("float64")
    # 在第 2 行 "col2_c" 列中设置 NaN 值
    dummies_basic.loc[2, "col2_c"] = np.nan
    # 使用 pytest 来检查是否会引发 ValueError 异常，并检查异常信息中的内容匹配特定模式
    with pytest.raises(
        ValueError, match=r"Dummy DataFrame contains NA value in column: 'col2_c'"
    ):
        # 调用 from_dummies 函数来处理包含 NaN 值的 DataFrame，指定分隔符为 "_"
        from_dummies(dummies_basic, sep="_")


# 测试函数：测试当 DataFrame 包含非虚拟变量数据时是否会触发 TypeError 异常
def test_error_with_prefix_contains_non_dummies(dummies_basic):
    # 将 "col2_c" 列转换为 object 类型，以避免设置 "str" 时的类型转换
    dummies_basic["col2_c"] = dummies_basic["col2_c"].astype(object)
    # 在第 2 行 "col2_c" 列中设置 "str" 值
    dummies_basic.loc[2, "col2_c"] = "str"
    # 使用 pytest 来检查是否会引发 TypeError 异常，并检查异常信息中的内容匹配特定模式
    with pytest.raises(TypeError, match=r"Passed DataFrame contains non-dummy data"):
        # 调用 from_dummies 函数来处理包含非虚拟变量数据的 DataFrame，指定分隔符为 "_"
        from_dummies(dummies_basic, sep="_")


# 测试函数：测试当 DataFrame 包含重复赋值时是否会触发 ValueError 异常
def test_error_with_prefix_double_assignment():
    # 创建一个包含重复赋值的 DataFrame
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 1],
            "col1_b": [1, 1, 0],
            "col2_a": [0, 1, 0],
            "col2_b": [1, 0, 0],
            "col2_c": [0, 0, 1],
        },
    )
    # 使用 pytest 来检查是否会引发 ValueError 异常，并检查异常信息中的内容匹配特定模式
    with pytest.raises(
        ValueError,
        match=(
            r"Dummy DataFrame contains multi-assignment\(s\); "
            r"First instance in row: 0"
        ),
    ):
        # 调用 from_dummies 函数来处理包含重复赋值的 DataFrame，指定分隔符为 "_"
        from_dummies(dummies, sep="_")


# 测试函数：测试将 Series 转换为 DataFrame 再转换回来是否得到预期结果
def test_roundtrip_series_to_dataframe():
    # 创建一个 Series 对象
    categories = Series(["a", "b", "c", "a"])
    # 调用 get_dummies 函数将 Series 转换为虚拟变量 DataFrame
    dummies = get_dummies(categories)
    # 调用 from_dummies 函数将虚拟变量 DataFrame 转换回 Series
    result = from_dummies(dummies)
    # 创建一个预期的 DataFrame 对象
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    # 使用 assert_frame_equal 来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试将单列 DataFrame 转换为虚拟变量再转换回来是否得到预期结果
def test_roundtrip_single_column_dataframe():
    # 创建一个单列 DataFrame 对象
    categories = DataFrame({"": ["a", "b", "c", "a"]})
    # 调用 get_dummies 函数将单列 DataFrame 转换为虚拟变量 DataFrame
    dummies = get_dummies(categories)
    # 调用 from_dummies 函数将虚拟变量 DataFrame 转换回单列 DataFrame，指定分隔符为 "_"
    result = from_dummies(dummies, sep="_")
    # 预期结果与输入的单列 DataFrame 相同
    expected = categories
    # 使用 assert_frame_equal 来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试使用具有前缀的 DataFrame 进行往返转换是否得到预期结果
def test_roundtrip_with_prefixes():
    # 创建一个具有两列的 DataFrame 对象
    categories = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    # 使用 get_dummies 函数将分类变量 categories 转换为虚拟变量 DataFrame
    dummies = get_dummies(categories)
    # 使用 from_dummies 函数将虚拟变量 DataFrame 转换为原始分类变量 DataFrame，使用下划线作为分隔符
    result = from_dummies(dummies, sep="_")
    # 将原始分类变量 categories 赋值给 expected 变量
    expected = categories
    # 使用 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 测试函数，用于测试处理没有前缀的字符串分类数据的基本情况
def test_no_prefix_string_cats_basic():
    # 创建包含虚拟变量的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    # 期望的结果数据框，包含一个空字符串列和分类变量的名称
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试处理具有布尔值的没有前缀的字符串分类数据的基本情况
def test_no_prefix_string_cats_basic_bool_values():
    # 创建包含布尔值的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame(
        {"a": [True, False, False, True], "b": [False, True, False, False], "c": [False, False, True, False]}
    )
    # 期望的结果数据框，包含一个空字符串列和分类变量的名称
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试处理具有混合布尔值的没有前缀的字符串分类数据的基本情况
def test_no_prefix_string_cats_basic_mixed_bool_values():
    # 创建包含混合布尔值的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame(
        {"a": [1, 0, 0, 1], "b": [False, True, False, False], "c": [0, 0, 1, 0]}
    )
    # 期望的结果数据框，包含一个空字符串列和分类变量的名称
    expected = DataFrame({"": ["a", "b", "c", "a"]})
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试处理没有前缀的整数分类数据的基本情况
def test_no_prefix_int_cats_basic():
    # 创建包含整数的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame(
        {1: [1, 0, 0, 0], 25: [0, 1, 0, 0], 2: [0, 0, 1, 0], 5: [0, 0, 0, 1]}
    )
    # 期望的结果数据框，包含一个空字符串列和分类变量的整数值
    expected = DataFrame({"": [1, 25, 2, 5]})
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试处理没有前缀的浮点数分类数据的基本情况
def test_no_prefix_float_cats_basic():
    # 创建包含浮点数的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame(
        {1.0: [1, 0, 0, 0], 25.0: [0, 1, 0, 0], 2.5: [0, 0, 1, 0], 5.84: [0, 0, 0, 1]}
    )
    # 期望的结果数据框，包含一个空字符串列和分类变量的浮点数值
    expected = DataFrame({"": [1.0, 25.0, 2.5, 5.84]})
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试处理没有前缀的混合分类数据的基本情况
def test_no_prefix_mixed_cats_basic():
    # 创建包含混合数据类型的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame(
        {
            1.23: [1, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0],
            2: [0, 0, 1, 0, 0],
            False: [0, 0, 0, 1, 0],
            None: [0, 0, 0, 0, 1],
        }
    )
    # 期望的结果数据框，包含一个空字符串列和分类变量的值，类型为对象
    expected = DataFrame({"": [1.23, "c", 2, False, None]}, dtype="object")
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，用于测试处理没有前缀的字符串分类数据，并包含具有NaN列的情况
def test_no_prefix_string_cats_contains_get_dummies_NaN_column():
    # 创建包含NaN列的数据框，每列表示一个分类变量的虚拟编码
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0], "NaN": [0, 0, 1]})
    # 期望的结果数据框，包含一个空字符串列和分类变量的名称，其中包含NaN列
    expected = DataFrame({"": ["a", "b", "NaN"]})
    # 调用被测试函数，将虚拟变量数据框转换为结果数据框
    result = from_dummies(dummies)
    # 使用测试框架验证结果数据框与期望数据框是否相等
    tm.assert_frame_equal(result, expected)


# 参数化测试，测试默认类别和期望结果
    [
        pytest.param(
            "c",  # 参数值为字符串 "c"
            {"": ["a", "b", "c"]},  # 期望返回一个空键对应的列表包含 "a", "b", "c"
            id="default_category is a str",  # 测试用例的标识
        ),
        pytest.param(
            1,  # 参数值为整数 1
            {"": ["a", "b", 1]},  # 期望返回一个空键对应的列表包含 "a", "b", 1
            id="default_category is a int",  # 测试用例的标识
        ),
        pytest.param(
            1.25,  # 参数值为浮点数 1.25
            {"": ["a", "b", 1.25]},  # 期望返回一个空键对应的列表包含 "a", "b", 1.25
            id="default_category is a float",  # 测试用例的标识
        ),
        pytest.param(
            0,  # 参数值为整数 0
            {"": ["a", "b", 0]},  # 期望返回一个空键对应的列表包含 "a", "b", 0
            id="default_category is a 0",  # 测试用例的标识
        ),
        pytest.param(
            False,  # 参数值为布尔值 False
            {"": ["a", "b", False]},  # 期望返回一个空键对应的列表包含 "a", "b", False
            id="default_category is a bool",  # 测试用例的标识
        ),
        pytest.param(
            (1, 2),  # 参数值为元组 (1, 2)
            {"": ["a", "b", (1, 2)]},  # 期望返回一个空键对应的列表包含 "a", "b", (1, 2)
            id="default_category is a tuple",  # 测试用例的标识
        ),
    ],
# 定义一个测试函数，用于测试在没有前缀的情况下使用默认类别进行字符串合并
def test_no_prefix_string_cats_default_category(
    default_category, expected, using_infer_string
):
    # 创建一个包含虚拟数据的数据帧，列名为"a"和"b"
    dummies = DataFrame({"a": [1, 0, 0], "b": [0, 1, 0]})
    # 调用from_dummies函数，生成结果数据帧result
    result = from_dummies(dummies, default_category=default_category)
    # 将期望的输出转换为数据帧格式
    expected = DataFrame(expected)
    # 如果使用推断字符串类型
    if using_infer_string:
        # 将期望输出的空列转换为字符串类型（使用pyarrow_numpy）
        expected[""] = expected[""].astype("string[pyarrow_numpy]")
    # 断言结果数据帧与期望数据帧相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试在具有前缀的基本情况下的字符串合并
def test_with_prefix_basic(dummies_basic):
    # 创建一个包含期望结果的数据帧，列名为"col1"和"col2"
    expected = DataFrame({"col1": ["a", "b", "a"], "col2": ["b", "a", "c"]})
    # 调用from_dummies函数，生成结果数据帧result，使用下划线作为分隔符
    result = from_dummies(dummies_basic, sep="_")
    # 断言结果数据帧与期望数据帧相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试在具有前缀的情况下，包含NaN列的字符串合并
def test_with_prefix_contains_get_dummies_NaN_column():
    # 创建一个包含虚拟数据的数据帧，具有多个列和NaN值
    dummies = DataFrame(
        {
            "col1_a": [1, 0, 0],
            "col1_b": [0, 1, 0],
            "col1_NaN": [0, 0, 1],
            "col2_a": [0, 1, 0],
            "col2_b": [0, 0, 0],
            "col2_c": [0, 0, 1],
            "col2_NaN": [1, 0, 0],
        },
    )
    # 创建一个包含期望结果的数据帧，列名为"col1"和"col2"
    expected = DataFrame({"col1": ["a", "b", "NaN"], "col2": ["NaN", "a", "c"]})
    # 调用from_dummies函数，生成结果数据帧result，使用下划线作为分隔符
    result = from_dummies(dummies, sep="_")
    # 断言结果数据帧与期望数据帧相等
    tm.assert_frame_equal(result, expected)


# 使用pytest的参数化装饰器定义多个测试用例，用于测试具有前缀的情况下，使用不同的默认类别进行字符串合并
@pytest.mark.parametrize(
    "default_category, expected",
    [
        # 测试默认类别为字符串"x"时的情况
        pytest.param(
            "x",
            {"col1": ["a", "b", "x"], "col2": ["x", "a", "c"]},
            id="default_category is a str",
        ),
        # 测试默认类别为整数0时的情况
        pytest.param(
            0,
            {"col1": ["a", "b", 0], "col2": [0, "a", "c"]},
            id="default_category is a 0",
        ),
        # 测试默认类别为布尔值False时的情况
        pytest.param(
            False,
            {"col1": ["a", "b", False], "col2": [False, "a", "c"]},
            id="default_category is a False",
        ),
        # 测试默认类别为字典时，包含整数和浮点数值的情况
        pytest.param(
            {"col2": 1, "col1": 2.5},
            {"col1": ["a", "b", 2.5], "col2": [1, "a", "c"]},
            id="default_category is a dict with int and float values",
        ),
        # 测试默认类别为字典时，包含布尔值和None值的情况
        pytest.param(
            {"col2": None, "col1": False},
            {"col1": ["a", "b", False], "col2": [None, "a", "c"]},
            id="default_category is a dict with bool and None values",
        ),
        # 测试默认类别为字典时，包含列表和元组值的情况
        pytest.param(
            {"col2": (1, 2), "col1": [1.25, False]},
            {"col1": ["a", "b", [1.25, False]], "col2": [(1, 2), "a", "c"]},
            id="default_category is a dict with list and tuple values",
        ),
    ],
)
def test_with_prefix_default_category(
    dummies_with_unassigned, default_category, expected
):
    # 调用from_dummies函数，生成结果数据帧result，使用下划线作为分隔符，指定默认类别
    result = from_dummies(
        dummies_with_unassigned, sep="_", default_category=default_category
    )
    # 将期望的输出转换为数据帧格式
    expected = DataFrame(expected)
    # 断言结果数据帧与期望数据帧相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试在类别为"string[python]"时的分类操作
def test_ea_categories():
    # 创建一个包含分类数据的数据帧，列名为"a", "b", "c"
    df = DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]})
    # 将数据帧的列类型转换为"string[python]"
    df.columns = df.columns.astype("string[python]")
    # 调用from_dummies函数，生成结果数据帧result
    result = from_dummies(df)
    # 创建一个包含期望结果的数据帧，列名为空，类型为"string[python]"
    expected = DataFrame({"": Series(list("abca"), dtype="string[python]")})
    # 断言结果数据帧与期望数据帧相等
    tm.assert_frame_equal(result, expected)
    # GH 54300 的测试用例，用于验证从虚拟变量（dummy variables）中恢复原始分类变量的功能
    
    # 创建一个 DataFrame，包含虚拟变量的数据
    df = DataFrame(
        {
            "col1_a": [1, 0, 1],    # 第一列的虚拟变量 'col1' 的取值
            "col1_b": [0, 1, 0],    # 第一列的虚拟变量 'col1' 的取值
            "col2_a": [0, 1, 0],    # 第二列的虚拟变量 'col2' 的取值
            "col2_b": [1, 0, 0],    # 第二列的虚拟变量 'col2' 的取值
            "col2_c": [0, 0, 1],    # 第二列的虚拟变量 'col2' 的取值
        }
    )
    
    # 将 DataFrame 的列名转换为 string[python] 类型，这是特定于 Pandas 的类型声明
    df.columns = df.columns.astype("string[python]")
    
    # 调用 from_dummies 函数，将虚拟变量的 DataFrame 转换为原始的分类变量 DataFrame
    result = from_dummies(df, sep="_")
    
    # 创建预期的 DataFrame，包含原始分类变量的数据
    expected = DataFrame(
        {
            "col1": Series(list("aba"), dtype="string[python]"),    # 第一列的预期原始分类变量 'col1' 的取值
            "col2": Series(list("bac"), dtype="string[python]"),    # 第二列的预期原始分类变量 'col2' 的取值
        }
    )
    
    # 将预期 DataFrame 的列名转换为 string[python] 类型
    expected.columns = expected.columns.astype("string[python]")
    
    # 使用 tm.assert_frame_equal 比较 result 和 expected，确保它们相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试特定功能
def test_maintain_original_index():
    # GH 54300：这个测试函数与 GitHub 上的 issue 编号 54300 相关联

    # 创建一个 DataFrame 对象 df，包含三列 'a', 'b', 'c'，每列对应的值为列表中的元素，行索引为列表 "abcd"
    df = DataFrame(
        {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]}, index=list("abcd")
    )

    # 调用 from_dummies 函数处理 DataFrame df，返回结果存储在 result 变量中
    result = from_dummies(df)

    # 创建一个期望的 DataFrame 对象 expected，包含一列为空列 ""，其值为列表 "abca"，行索引为列表 "abcd"
    expected = DataFrame({"": list("abca")}, index=list("abcd"))

    # 使用 tm.assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
```