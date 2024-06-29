# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_all_methods.py`

```
"""
Tests that apply to all groupby operation methods.

The only tests that should appear here are those that use the `groupby_func` fixture.
Even if it does use that fixture, prefer a more specific test file if it available
such as:

 - test_categorical
 - test_groupby_dropna
 - test_groupby_subclass
 - test_raises
"""

import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 pandas 库并使用 pd 别名
from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 内部测试模块中的 _testing
from pandas.tests.groupby import get_groupby_method_args  # 从 pandas 测试模块中的 groupby 子模块导入 get_groupby_method_args 函数


def test_multiindex_group_all_columns_when_empty(groupby_func):
    # GH 32464 测试空 DataFrame 在多级索引情况下的 groupby 行为
    df = DataFrame({"a": [], "b": [], "c": []}).set_index(["a", "b", "c"])  # 创建空 DataFrame 并设置多级索引
    gb = df.groupby(["a", "b", "c"], group_keys=False)  # 对 DataFrame 进行 groupby 操作，不生成分组键
    method = getattr(gb, groupby_func)  # 获取指定方法的调用方式
    args = get_groupby_method_args(groupby_func, df)  # 获取 groupby 方法的参数
    if groupby_func == "corrwith":
        warn = FutureWarning  # 如果是 corrwith 方法，警告类型为 FutureWarning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"  # 相应的警告消息
    else:
        warn = None  # 否则警告为空
        warn_msg = ""  # 警告消息为空字符串
    with tm.assert_produces_warning(warn, match=warn_msg):  # 断言产生相应的警告
        result = method(*args).index  # 执行方法并获取其索引
    expected = df.index  # 期望的结果索引为原 DataFrame 的索引
    tm.assert_index_equal(result, expected)  # 断言结果与期望相等


def test_duplicate_columns(request, groupby_func, as_index):
    # GH#50806 测试在存在重复列时的 groupby 方法
    if groupby_func == "corrwith":
        msg = "GH#50845 - corrwith fails when there are duplicate columns"  # 如果是 corrwith 方法，添加失败时的消息
        request.applymarker(pytest.mark.xfail(reason=msg))  # 对此测试标记为预期失败
    df = DataFrame([[1, 3, 6], [1, 4, 7], [2, 5, 8]], columns=list("abb"))  # 创建包含重复列的 DataFrame
    args = get_groupby_method_args(groupby_func, df)  # 获取 groupby 方法的参数
    gb = df.groupby("a", as_index=as_index)  # 对 DataFrame 进行按列分组，根据 as_index 参数设置是否生成索引
    result = getattr(gb, groupby_func)(*args)  # 调用指定方法进行 groupby 操作

    expected_df = df.set_axis(["a", "b", "c"], axis=1)  # 重设 DataFrame 的列名
    expected_args = get_groupby_method_args(groupby_func, expected_df)  # 获取重设后的 DataFrame 的 groupby 方法参数
    expected_gb = expected_df.groupby("a", as_index=as_index)  # 对重设后的 DataFrame 进行按列分组
    expected = getattr(expected_gb, groupby_func)(*expected_args)  # 调用指定方法进行 groupby 操作
    if groupby_func not in ("size", "ngroup", "cumcount"):
        expected = expected.rename(columns={"c": "b"})  # 如果方法不是 size、ngroup、cumcount，重命名列名为 b
    tm.assert_equal(result, expected)  # 断言结果与期望相等


@pytest.mark.parametrize(
    "idx",
    [
        pd.Index(["a", "a"], name="foo"),  # 创建具有重复标签的单索引
        pd.MultiIndex.from_tuples((("a", "a"), ("a", "a")), names=["foo", "bar"]),  # 创建具有重复标签的多级索引
    ],
)
def test_dup_labels_output_shape(groupby_func, idx):
    if groupby_func in {"size", "ngroup", "cumcount"}:
        pytest.skip(f"Not applicable for {groupby_func}")  # 如果是 size、ngroup、cumcount 方法，跳过测试

    df = DataFrame([[1, 1]], columns=idx)  # 创建具有指定索引的 DataFrame
    grp_by = df.groupby([0])  # 根据第一列对 DataFrame 进行分组

    args = get_groupby_method_args(groupby_func, df)  # 获取 groupby 方法的参数
    if groupby_func == "corrwith":
        warn = FutureWarning  # 如果是 corrwith 方法，警告类型为 FutureWarning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"  # 相应的警告消息
    else:
        warn = None  # 否则警告为空
        warn_msg = ""  # 警告消息为空字符串
    with tm.assert_produces_warning(warn, match=warn_msg):  # 断言产生相应的警告
        result = getattr(grp_by, groupby_func)(*args)  # 调用指定方法进行 groupby 操作

    assert result.shape == (1, 2)  # 断言结果的形状为 (1, 2)
    tm.assert_index_equal(result.columns, idx)  # 断言结果的列名与指定的索引相等
```