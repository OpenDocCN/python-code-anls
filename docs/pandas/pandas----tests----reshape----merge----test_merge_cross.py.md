# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_merge_cross.py`

```
# 导入pytest库，用于单元测试
import pytest

# 从pandas库中导入DataFrame和Series类
from pandas import (
    DataFrame,
    Series,
)

# 导入pandas内部测试模块
import pandas._testing as tm

# 从pandas核心库中导入异常类MergeError和函数merge
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)

# 使用pytest的参数化装饰器，定义参数化测试用例
@pytest.mark.parametrize(
    ("input_col", "output_cols"), [("b", ["a", "b"]), ("a", ["a_x", "a_y"])]
)
# 定义测试函数test_merge_cross，测试merge函数的交叉连接功能
def test_merge_cross(input_col, output_cols):
    # 创建左侧DataFrame对象，包含列"a"，值为[1, 3]
    left = DataFrame({"a": [1, 3]})
    # 创建右侧DataFrame对象，根据输入参数决定列名和值
    right = DataFrame({input_col: [3, 4]})
    # 复制左右两侧DataFrame对象，用于后续比较
    left_copy = left.copy()
    right_copy = right.copy()
    # 使用merge函数进行左右DataFrame的交叉连接操作，返回合并后的结果
    result = merge(left, right, how="cross")
    # 根据预期结果创建DataFrame对象，用于断言比较
    expected = DataFrame({output_cols[0]: [1, 1, 3, 3], output_cols[1]: [3, 4, 3, 4]})
    # 使用pandas内部测试模块的assert_frame_equal函数，断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)
    # 检查左侧DataFrame对象是否在操作后保持不变
    tm.assert_frame_equal(left, left_copy)
    # 检查右侧DataFrame对象是否在操作后保持不变
    tm.assert_frame_equal(right, right_copy)


# 使用pytest的参数化装饰器，定义参数化测试用例
@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True},
        {"right_index": True},
        {"on": "a"},
        {"left_on": "a"},
        {"right_on": "b"},
    ],
)
# 定义测试函数test_merge_cross_error_reporting，测试merge函数在错误参数情况下的异常报告功能
def test_merge_cross_error_reporting(kwargs):
    # 创建左侧DataFrame对象，包含列"a"，值为[1, 3]
    left = DataFrame({"a": [1, 3]})
    # 创建右侧DataFrame对象，包含列"b"，值为[3, 4]
    right = DataFrame({"b": [3, 4]})
    # 定义错误信息字符串
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or "
        "left_index=True"
    )
    # 使用pytest的raises上下文管理器，检查是否抛出MergeError异常，并且异常信息与msg匹配
    with pytest.raises(MergeError, match=msg):
        # 调用merge函数，传入左右DataFrame对象和参数化的kwargs字典
        merge(left, right, how="cross", **kwargs)


# 定义测试函数test_merge_cross_mixed_dtypes，测试merge函数在不同数据类型下的交叉连接功能
def test_merge_cross_mixed_dtypes():
    # 创建左侧DataFrame对象，包含列"A"，值为['a', 'b', 'c']
    left = DataFrame(["a", "b", "c"], columns=["A"])
    # 创建右侧DataFrame对象，包含列"B"，值为[0, 1]
    right = DataFrame(range(2), columns=["B"])
    # 使用merge函数进行左右DataFrame的交叉连接操作，返回合并后的结果
    result = merge(left, right, how="cross")
    # 根据预期结果创建DataFrame对象，用于断言比较
    expected = DataFrame({"A": ["a", "a", "b", "b", "c", "c"], "B": [0, 1, 0, 1, 0, 1]})
    # 使用pandas内部测试模块的assert_frame_equal函数，断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_merge_cross_more_than_one_column，测试merge函数在多列情况下的交叉连接功能
def test_merge_cross_more_than_one_column():
    # 创建左侧DataFrame对象，包含列"A"和"B"，值分别为['a', 'b']和[2, 1]
    left = DataFrame({"A": list("ab"), "B": [2, 1]})
    # 创建右侧DataFrame对象，包含列"C"和"D"，值分别为[0, 1]和[4, 5]
    right = DataFrame({"C": range(2), "D": range(4, 6)})
    # 使用merge函数进行左右DataFrame的交叉连接操作，返回合并后的结果
    result = merge(left, right, how="cross")
    # 根据预期结果创建DataFrame对象，用于断言比较
    expected = DataFrame(
        {
            "A": ["a", "a", "b", "b"],
            "B": [2, 2, 1, 1],
            "C": [0, 1, 0, 1],
            "D": [4, 5, 4, 5],
        }
    )
    # 使用pandas内部测试模块的assert_frame_equal函数，断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_merge_cross_null_values，测试merge函数在空值情况下的交叉连接功能
def test_merge_cross_null_values(nulls_fixture):
    # 创建左侧DataFrame对象，包含列"a"，值为[1, nulls_fixture]
    left = DataFrame({"a": [1, nulls_fixture]})
    # 创建右侧DataFrame对象，包含列"b"和"c"，值分别为['a', 'b']和[1.0, 2.0]
    right = DataFrame({"b": ["a", "b"], "c": [1.0, 2.0]})
    # 使用merge函数进行左右DataFrame的交叉连接操作，返回合并后的结果
    result = merge(left, right, how="cross")
    # 根据预期结果创建DataFrame对象，用于断言比较
    expected = DataFrame(
        {
            "a": [1, 1, nulls_fixture, nulls_fixture],
            "b": ["a", "b", "a", "b"],
            "c": [1.0, 2.0, 1.0, 2.0],
        }
    )
    # 使用pandas内部测试模块的assert_frame_equal函数，断言结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_join_cross_error_reporting，测试join函数在错误参数情况下的异常报告功能
def test_join_cross_error_reporting():
    # 创建左侧DataFrame对象，包含列"a"，值为[1, 3]
    left = DataFrame({"a": [1, 3]})
    # 创建右侧DataFrame对象，包含列"a"，值为[3, 4]
    right = DataFrame({"a": [3, 4]})
    # 定义错误信息字符串
    msg = (
        "Can not pass on, right_on, left_on or set right_index=True or "
        "left_index=True"
    )
    # 使用pytest的raises上下文管理器，检查是否抛出MergeError异常，并且异常信息与msg匹配
    with pytest.raises(MergeError, match=msg):
        # 调用左侧DataFrame对象的join方法，进行交叉连接操作，传入右侧DataFrame对象和错误参数
        left.join(right, how="cross", on="a")


# 定义测试函数test_merge_cross_series，测试merge函数在Series对象上的交叉连接功能
def test_merge_cross_series():
    # 创建Series对象ls，包含索引[1, 2, 3, 4]，值为[1, 2, 3, 4]
    ls = Series([1, 2, 3, 4], index=[1, 2, 3, 4], name="left")
    # 创建一个 Pandas Series 对象 `rs`，包含整数 [3, 4, 5, 6]，并指定这些整数为索引，名称为 "right"
    rs = Series([3, 4, 5, 6], index=[3, 4, 5, 6], name="right")
    
    # 使用 `merge` 函数将左侧的 Series `ls` 和右侧的 Series `rs` 进行交叉连接（笛卡尔积），结果存储在 `res` 中
    res = merge(ls, rs, how="cross")
    
    # 使用 `ls` 和 `rs` 分别转换为 DataFrame，然后使用 `merge` 函数将它们进行交叉连接，预期结果存储在 `expected` 中
    expected = merge(ls.to_frame(), rs.to_frame(), how="cross")
    
    # 使用 `tm.assert_frame_equal` 函数断言 `res` 和 `expected` 的 DataFrame 相等，确保交叉连接的结果正确
    tm.assert_frame_equal(res, expected)
```