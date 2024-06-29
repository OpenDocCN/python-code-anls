# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_merge_index_as_string.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

@pytest.fixture
def df1():
    """Fixture函数：返回一个DataFrame对象，用于测试"""
    return DataFrame(
        {
            "outer": [1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4],  # 创建'outer'列
            "inner": [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2],  # 创建'inner'列
            "v1": np.linspace(0, 1, 11),  # 创建'v1'列，包含从0到1的11个均匀间隔数值
        }
    )


@pytest.fixture
def df2():
    """Fixture函数：返回另一个DataFrame对象，用于测试"""
    return DataFrame(
        {
            "outer": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3],  # 创建'outer'列
            "inner": [1, 2, 2, 3, 3, 4, 2, 3, 1, 1, 2, 3],  # 创建'inner'列
            "v2": np.linspace(10, 11, 12),  # 创建'v2'列，包含从10到11的12个均匀间隔数值
        }
    )


@pytest.fixture(params=[[], ["outer"], ["outer", "inner"]])
def left_df(request, df1):
    """Fixture函数：根据参数返回不同索引设置的左侧测试DataFrame

    Parameters
    ----------
    request : pytest.FixtureRequest
        pytest中的请求对象，包含参数化的参数
    df1 : DataFrame
        左侧原始的DataFrame对象

    Returns
    -------
    DataFrame
        根据参数设置后的DataFrame对象
    """
    levels = request.param
    if levels:
        df1 = df1.set_index(levels)  # 根据参数设置DataFrame的索引

    return df1


@pytest.fixture(params=[[], ["outer"], ["outer", "inner"]])
def right_df(request, df2):
    """Fixture函数：根据参数返回不同索引设置的右侧测试DataFrame

    Parameters
    ----------
    request : pytest.FixtureRequest
        pytest中的请求对象，包含参数化的参数
    df2 : DataFrame
        右侧原始的DataFrame对象

    Returns
    -------
    DataFrame
        根据参数设置后的DataFrame对象
    """
    levels = request.param

    if levels:
        df2 = df2.set_index(levels)  # 根据参数设置DataFrame的索引

    return df2


def compute_expected(df_left, df_right, on=None, left_on=None, right_on=None, how=None):
    """
    Compute the expected merge result for the test case.

    This method computes the expected result of merging two DataFrames on
    a combination of their columns and index levels. It does so by
    explicitly dropping/resetting their named index levels, performing a
    merge on their columns, and then finally restoring the appropriate
    index in the result.

    Parameters
    ----------
    df_left : DataFrame
        The left DataFrame (may have zero or more named index levels)
    df_right : DataFrame
        The right DataFrame (may have zero or more named index levels)
    on : list of str
        The on parameter to the merge operation
    left_on : list of str
        The left_on parameter to the merge operation
    right_on : list of str
        The right_on parameter to the merge operation
    how : str
        The how parameter to the merge operation

    Returns
    -------
    DataFrame
        The expected merge result
    """
    # Handle on param if specified
    if on is not None:
        left_on, right_on = on, on  # 如果指定了'on'参数，则left_on和right_on都设置为'on'

    # Compute input named index levels
    left_levels = [n for n in df_left.index.names if n is not None]  # 获取左侧DataFrame的命名索引级别
    right_levels = [n for n in df_right.index.names if n is not None]  # 获取右侧DataFrame的命名索引级别

    # Compute output named index levels
    output_levels = [i for i in left_on if i in right_levels and i in left_levels]  # 计算输出的命名索引级别

    # Drop index levels that aren't involved in the merge
    drop_left = [n for n in left_levels if n not in left_on]  # 需要在左侧DataFrame中删除的索引级别
    if drop_left:
        df_left = df_left.reset_index(drop_left, drop=True)  # 根据需要删除的索引级别重置左侧DataFrame

    drop_right = [n for n in right_levels if n not in right_on]  # 需要在右侧DataFrame中删除的索引级别
    if drop_right:
        df_right = df_right.reset_index(drop_right, drop=True)  # 根据需要删除的索引级别重置右侧DataFrame

    # Convert remaining index levels to columns
    # 在左侧数据框中重置索引，仅包括在 left_levels 中存在的列
    reset_left = [n for n in left_levels if n in left_on]
    if reset_left:
        df_left = df_left.reset_index(level=reset_left)

    # 在右侧数据框中重置索引，仅包括在 right_levels 中存在的列
    reset_right = [n for n in right_levels if n in right_on]
    if reset_right:
        df_right = df_right.reset_index(level=reset_right)

    # 执行数据框的合并操作，使用指定的列名进行左右连接
    expected = df_left.merge(df_right, left_on=left_on, right_on=right_on, how=how)

    # 如果指定了输出索引级别，则根据这些级别重新设置合并后数据框的索引
    if output_levels:
        expected = expected.set_index(output_levels)

    # 返回合并后的结果数据框
    return expected
@pytest.mark.parametrize(
    "on,how",
    [
        (["outer"], "inner"),  # 参数化测试用例，定义了 'on' 和 'how' 的组合
        (["inner"], "left"),
        (["outer", "inner"], "right"),
        (["inner", "outer"], "outer"),
    ],
)
def test_merge_indexes_and_columns_on(left_df, right_df, on, how):
    # 构造预期结果
    expected = compute_expected(left_df, right_df, on=on, how=how)

    # 执行合并操作
    result = left_df.merge(right_df, on=on, how=how)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "left_on,right_on,how",
    [
        (["outer"], ["outer"], "inner"),  # 参数化测试用例，定义了 'left_on'、'right_on' 和 'how' 的组合
        (["inner"], ["inner"], "right"),
        (["outer", "inner"], ["outer", "inner"], "left"),
        (["inner", "outer"], ["inner", "outer"], "outer"),
    ],
)
def test_merge_indexes_and_columns_lefton_righton(
    left_df, right_df, left_on, right_on, how
):
    # 构造预期结果
    expected = compute_expected(
        left_df, right_df, left_on=left_on, right_on=right_on, how=how
    )

    # 执行合并操作
    result = left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how)
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize("left_index", ["inner", ["inner", "outer"]])
def test_join_indexes_and_columns_on(df1, df2, left_index, join_type):
    # 构造 left_df
    left_df = df1.set_index(left_index)

    # 构造 right_df
    right_df = df2.set_index(["outer", "inner"])

    # 构造预期结果
    expected = (
        left_df.reset_index()
        .join(
            right_df, on=["outer", "inner"], how=join_type, lsuffix="_x", rsuffix="_y"
        )
        .set_index(left_index)
    )

    # 执行连接操作
    result = left_df.join(
        right_df, on=["outer", "inner"], how=join_type, lsuffix="_x", rsuffix="_y"
    )

    tm.assert_frame_equal(result, expected, check_like=True)
```