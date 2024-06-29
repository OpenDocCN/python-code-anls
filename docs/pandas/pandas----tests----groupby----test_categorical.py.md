# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_categorical.py`

```
from datetime import datetime  # 导入datetime模块中的datetime类

import numpy as np  # 导入numpy模块并重命名为np
import pytest  # 导入pytest模块

import pandas as pd  # 导入pandas模块并重命名为pd
from pandas import (  # 从pandas模块中导入多个子模块和类
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    qcut,
)
import pandas._testing as tm  # 导入pandas._testing模块并重命名为tm
from pandas.api.typing import SeriesGroupBy  # 从pandas.api.typing模块导入SeriesGroupBy类
from pandas.tests.groupby import get_groupby_method_args  # 导入pandas.tests.groupby模块中的get_groupby_method_args函数


def cartesian_product_for_groupers(result, args, names, fill_value=np.nan):
    """为分组器创建笛卡尔积的重新索引，保留每个分组器的特性（如Categorical）"""

    def f(a):
        """将输入a转换为Categorical类型，如果a是CategoricalIndex或Categorical"""
        if isinstance(a, (CategoricalIndex, Categorical)):
            categories = a.categories
            a = Categorical.from_codes(
                np.arange(len(categories)), categories=categories, ordered=a.ordered
            )
        return a

    index = MultiIndex.from_product(map(f, args), names=names)  # 使用map函数将args中的每个元素都应用f函数，然后创建MultiIndex索引
    return result.reindex(index, fill_value=fill_value).sort_index()  # 对结果result进行重新索引并排序


_results_for_groupbys_with_missing_categories = {
    # 下面的映射将内置的groupby函数映射到它们在观察到=False时，在分类分组器上调用时的预期输出。
    # 一些函数预期返回NaN，一些预期返回零。
    # 这些预期值可以在多个测试中使用（即它们对SeriesGroupBy和DataFrameGroupBy是相同的），
    # 但它们应该只在一个地方硬编码。
    "all": True,
    "any": False,
    "count": 0,
    "corrwith": np.nan,
    "first": np.nan,
    "idxmax": np.nan,
    "idxmin": np.nan,
    "last": np.nan,
    "max": np.nan,
    "mean": np.nan,
    "median": np.nan,
    "min": np.nan,
    "nth": np.nan,
    "nunique": 0,
    "prod": 1,
    "quantile": np.nan,
    "sem": np.nan,
    "size": 0,
    "skew": np.nan,
    "std": np.nan,
    "sum": 0,
    "var": np.nan,
}


def test_apply_use_categorical_name(df):
    """测试apply方法在使用分类名称时的行为"""

    cats = qcut(df.C, 4)  # 对df中的C列进行四分位数分箱

    def get_stats(group):
        """计算分组的统计信息"""
        return {
            "min": group.min(),
            "max": group.max(),
            "count": group.count(),
            "mean": group.mean(),
        }

    result = df.groupby(cats, observed=False).D.apply(get_stats)  # 对分类cats进行分组并应用get_stats函数
    assert result.index.names[0] == "C"  # 断言结果的第一个索引名称为"C"


def test_basic():
    """基本的测试用例"""

    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})  # 创建DataFrame对象data

    exp_index = CategoricalIndex(list("abcd"), name="b", ordered=True)  # 创建预期索引对象exp_index
    expected = DataFrame({"a": [1, 2, 4, np.nan]}, index=exp_index)  # 创建预期结果DataFrame对象expected
    result = data.groupby("b", observed=False).mean()  # 对数据按照'b'列进行分组并计算均值
    tm.assert_frame_equal(result, expected)  # 使用测试模块tm断言结果与预期值相等


def test_basic_single_grouper():
    """单个分组器的基本测试用例"""

    cat1 = Categorical(["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True)
    cat2 = Categorical(["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True)
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})  # 创建DataFrame对象df

    gb = df.groupby("A", observed=False)  # 对df按照'A'列进行分组，不考虑未观察到的类别
    # 创建一个分类索引对象，包含值为 ["a", "b", "z"] 的索引，名称为"A"，并且是有序的
    exp_idx = CategoricalIndex(["a", "b", "z"], name="A", ordered=True)
    
    # 创建一个数据框对象，包含一个名为"values"的序列，序列的索引为 exp_idx，值分别为 [3, 7, 0]
    expected = DataFrame({"values": Series([3, 7, 0], index=exp_idx)})
    
    # 对于给定的分组对象 gb，对其进行数值求和操作，仅考虑数值列
    result = gb.sum(numeric_only=True)
    
    # 使用测试工具 tm 中的 assert_frame_equal 函数，比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，测试基本的字符串操作
def test_basic_string(using_infer_string):
    # GH 8623
    # 创建一个包含两列的 DataFrame，第一列为数字，第二列为字符串
    x = DataFrame(
        [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
        columns=["person_id", "person_name"],
    )
    # 将 person_name 列转换为分类数据类型
    x["person_name"] = Categorical(x.person_name)

    # 根据 person_id 列分组 DataFrame
    g = x.groupby(["person_id"], observed=False)
    # 对每个分组应用一个恒等函数
    result = g.transform(lambda x: x)
    # 断言转换后的 DataFrame 与原 DataFrame 中仅包含 person_name 列的 DataFrame 相等
    tm.assert_frame_equal(result, x[["person_name"]])

    # 去除 person_name 列中重复的行
    result = x.drop_duplicates("person_name")
    # 期望的结果是原 DataFrame 的前两行
    expected = x.iloc[[0, 1]]
    tm.assert_frame_equal(result, expected)

    # 定义一个函数 f，用于应用到 DataFrameGroupBy 对象上
    def f(x):
        return x.drop_duplicates("person_name").iloc[0]

    # 设置警告消息
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在执行 g.apply(f) 时会产生 DeprecationWarning 警告，且警告消息匹配 msg
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = g.apply(f)
    # 期望的结果与原 DataFrame 的前两行相同，但是重新索引为 [1, 2]，并将 person_name 列转换为特定类型（根据 using_infer_string 参数决定）
    expected = x.iloc[[0, 1]].copy()
    expected.index = Index([1, 2], name="person_id")
    dtype = "string[pyarrow_numpy]" if using_infer_string else object
    expected["person_name"] = expected["person_name"].astype(dtype)
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试基本的单调函数操作
def test_basic_monotonic():
    # GH 9921
    # 创建一个包含一列 'a' 的 DataFrame
    df = DataFrame({"a": [5, 15, 25]})
    # 对 'a' 列进行分箱操作
    c = pd.cut(df.a, bins=[0, 10, 20, 30, 40])

    # 对 'a' 列根据分箱结果进行分组，并对每组应用 sum 函数
    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])

    # 使用 lambda 函数对每组应用 np.sum 函数
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )

    # 对整个 DataFrame 根据分箱结果进行分组，并对每组应用 sum 函数
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)

    # 对分组对象 gbc 应用 lambda 函数，计算每组中的最大值
    gbc = df.groupby(c, observed=False)
    result = gbc.transform(lambda xs: np.max(xs, axis=0))
    tm.assert_frame_equal(result, df[["a"]])

    # 使用不同方式计算每组中的最大值，并进行断言
    result2 = gbc.transform(lambda xs: np.max(xs, axis=0))
    result3 = gbc.transform(max)
    result4 = gbc.transform(np.maximum.reduce)
    result5 = gbc.transform(lambda xs: np.maximum.reduce(xs))
    tm.assert_frame_equal(result2, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result3, df[["a"]], check_dtype=False)
    tm.assert_frame_equal(result4, df[["a"]])
    tm.assert_frame_equal(result5, df[["a"]])

    # 过滤操作，断言经过过滤后的结果与原始 DataFrame 的 'a' 列相等
    tm.assert_series_equal(df.a.groupby(c, observed=False).filter(np.all), df["a"])
    tm.assert_frame_equal(df.groupby(c, observed=False).filter(np.all), df)


# 定义一个测试函数，测试基本的非单调函数操作
def test_basic_non_monotonic():
    # 创建一个包含一列 'a' 的 DataFrame
    df = DataFrame({"a": [5, 15, 25, -5]})
    # 对 'a' 列进行分箱操作
    c = pd.cut(df.a, bins=[-10, 0, 10, 20, 30, 40])

    # 对 'a' 列根据分箱结果进行分组，并对每组应用 sum 函数
    result = df.a.groupby(c, observed=False).transform(sum)
    tm.assert_series_equal(result, df["a"])

    # 使用 lambda 函数对每组应用 np.sum 函数
    tm.assert_series_equal(
        df.a.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df["a"]
    )

    # 对整个 DataFrame 根据分箱结果进行分组，并对每组应用 sum 函数
    result = df.groupby(c, observed=False).transform(sum)
    expected = df[["a"]]
    tm.assert_frame_equal(result, expected)

    # 使用 lambda 函数对整个 DataFrame 根据分组对象 gbc 应用，计算每组中的和
    tm.assert_frame_equal(
        df.groupby(c, observed=False).transform(lambda xs: np.sum(xs)), df[["a"]]
    )


# 定义一个测试函数，测试基本的分箱分组操作
def test_basic_cut_grouping():
    # GH 9603
    # 创建一个包含一列 'a' 的 DataFrame
    df = DataFrame({"a": [1, 0, 0, 0]})
    # 对 'a' 列进行分箱操作，并指定标签
    c = pd.cut(df.a, [0, 1, 2, 3, 4], labels=Categorical(list("abcd")))
    # 使用 DataFrame `df` 按列 `c` 进行分组，并对每个分组应用 `len` 函数，返回每个分组的长度
    result = df.groupby(c, observed=False).apply(len)
    
    # 使用 `CategoricalIndex` 对象创建 `exp_index`，该对象由 `c.values.categories` 中的类别和 `c.values.ordered` 中的排序信息组成
    exp_index = CategoricalIndex(c.values.categories, ordered=c.values.ordered)
    
    # 创建一个 Series `expected`，其值为 [1, 0, 0, 0]，索引为 `exp_index`
    expected = Series([1, 0, 0, 0], index=exp_index)
    
    # 设置 `expected` 的索引名称为 "a"
    expected.index.name = "a"
    
    # 使用 `tm.assert_series_equal` 函数断言 `result` 和 `expected` Series 相等
    tm.assert_series_equal(result, expected)
def test_more_basic():
    # 创建一个包含四个水平的列表
    levels = ["foo", "bar", "baz", "qux"]
    # 使用随机数生成器创建一个包含10个元素的整数数组，范围在0到3之间
    codes = np.random.default_rng(2).integers(0, 4, size=10)

    # 使用分类方法根据codes和levels创建一个有序分类变量
    cats = Categorical.from_codes(codes, levels, ordered=True)

    # 使用随机标准正态分布数据创建一个包含10行4列的数据帧
    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

    # 按照分类变量cats对数据帧data进行分组，并计算每组的均值
    result = data.groupby(cats, observed=False).mean()

    # 以numpy数组的形式按照分类变量cats对数据帧data进行分组，并计算每组的均值
    expected = data.groupby(np.asarray(cats), observed=False).mean()
    # 根据新的分类索引exp_idx对预期结果进行重新索引
    exp_idx = CategoricalIndex(levels, categories=cats.categories, ordered=True)
    expected = expected.reindex(exp_idx)

    # 使用测试工具比较result和expected的数据帧是否相等
    tm.assert_frame_equal(result, expected)

    # 再次按照分类变量cats对数据帧data进行分组，不进行观察，获得分组对象
    grouped = data.groupby(cats, observed=False)
    # 对分组对象进行描述性统计
    desc_result = grouped.describe()

    # 对分类变量cats的编码进行排序，并获取排序后的标签和数据
    idx = cats.codes.argsort()
    ord_labels = np.asarray(cats).take(idx)
    ord_data = data.take(idx)

    # 使用排序后的标签ord_labels创建一个有序分类变量exp_cats
    exp_cats = Categorical(
        ord_labels, ordered=True, categories=["foo", "bar", "baz", "qux"]
    )
    # 根据exp_cats对排序后的数据ord_data进行分组，不排序，不进行观察，并进行描述性统计
    expected = ord_data.groupby(exp_cats, sort=False, observed=False).describe()
    # 使用测试工具比较desc_result和expected的数据帧是否相等
    tm.assert_frame_equal(desc_result, expected)

    # GH 10460
    # 根据重复的范围值创建一个分类变量expc，使用指定的levels和有序属性
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    # 创建一个分类索引exp，使用expc
    exp = CategoricalIndex(expc)
    # 使用测试工具比较描述结果的堆叠索引的第一级是否等于exp
    tm.assert_index_equal(desc_result.stack().index.get_level_values(0), exp)
    # 创建一个包含指定索引名称的索引exp
    exp = Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4)
    # 使用测试工具比较描述结果的堆叠索引的第二级是否等于exp
    tm.assert_index_equal(desc_result.stack().index.get_level_values(1), exp)


def test_level_get_group(observed):
    # GH15155
    # 创建一个多级索引的数据帧df，其中包含两个级别，一个是分类变量，一个是范围值
    df = DataFrame(
        data=np.arange(2, 22, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(10)],
            codes=[[0] * 5 + [1] * 5, range(10)],
            names=["Index1", "Index2"],
        ),
    )
    # 根据Index1级别对数据帧df进行分组，根据参数observed来决定是否进行观察
    g = df.groupby(level=["Index1"], observed=observed)

    # 期望结果应与test.loc[["a"]]相等
    # GH15166
    # 创建一个期望的数据帧expected，包含指定数据和多级索引
    expected = DataFrame(
        data=np.arange(2, 12, 2),
        index=MultiIndex(
            levels=[CategoricalIndex(["a", "b"]), range(5)],
            codes=[[0] * 5, range(5)],
            names=["Index1", "Index2"],
        ),
    )
    # 获取分组结果g中("a",)组的数据，与期望的数据帧expected进行比较
    result = g.get_group(("a",))
    # 使用测试工具比较result和expected的数据帧是否相等
    tm.assert_frame_equal(result, expected)


def test_sorting_with_different_categoricals():
    # GH 24271
    # 创建一个数据帧df，包含三列：group, dose和outcomes
    df = DataFrame(
        {
            "group": ["A"] * 6 + ["B"] * 6,
            "dose": ["high", "med", "low"] * 4,
            "outcomes": np.arange(12.0),
        }
    )

    # 将df中的dose列转换为分类变量，指定分类的顺序
    df.dose = Categorical(df.dose, categories=["low", "med", "high"], ordered=True)

    # 按照group列分组，然后统计dose列中每个值的计数
    result = df.groupby("group")["dose"].value_counts()
    # 按照指定的顺序级别对结果进行排序
    result = result.sort_index(level=0, sort_remaining=True)
    # 创建一个指定顺序的分类变量index
    index = ["low", "med", "high", "low", "med", "high"]
    index = Categorical(index, categories=["low", "med", "high"], ordered=True)
    index = [["A", "A", "A", "B", "B", "B"], CategoricalIndex(index)]
    # 使用index数组创建一个多级索引对象index
    index = MultiIndex.from_arrays(index, names=["group", "dose"])
    # 创建一个期望的Series对象expected，包含指定数据和多级索引
    expected = Series([2] * 6, index=index, name="count")
    # 使用测试工具比较result和expected的Series对象是否相等
    tm.assert_series_equal(result, expected)
    # GH 10138
    
    # 创建一个有序的分类变量，取值为 'a', 'b', 'c'
    dense = Categorical(list("abc"), ordered=ordered)
    
    # 创建一个分类变量，取值为 'a', 'a', 'a'，指定可能的类别为 ["a", "b"]，有序性与 dense 相同
    missing = Categorical(list("aaa"), categories=["a", "b"], ordered=ordered)
    
    # 创建一个数据框，包含三列："missing" 列为 missing 变量，"dense" 列为 dense 变量，"values" 列为 values 数组
    values = np.arange(len(dense))
    df = DataFrame({"missing": missing, "dense": dense, "values": values})
    
    # 根据 "missing" 和 "dense" 列进行分组，保留未观察到的分类
    grouped = df.groupby(["missing", "dense"], observed=True)
    
    # 构建一个多级索引，索引的第一级为 missing，第二级为 dense，名称分别为 "missing" 和 "dense"
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    
    # 创建一个期望的数据框，包含一列 "values"，索引为 idx，值为 [0, 1, 2.0]
    expected = DataFrame([0, 1, 2.0], index=idx, columns=["values"])
    
    # 对分组后的数据应用函数 np.mean，并将结果与期望的数据框 expected 进行比较
    result = grouped.apply(lambda x: np.mean(x, axis=0))
    tm.assert_frame_equal(result, expected)
    
    # 对分组后的数据计算均值，并将结果与期望的数据框 expected 进行比较
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)
    
    # 对分组后的数据应用 np.mean 函数，并将结果与期望的数据框 expected 进行比较
    result = grouped.agg(np.mean)
    tm.assert_frame_equal(result, expected)
    
    # 对于 transform 操作，预期返回原始索引
    idx = MultiIndex.from_arrays([missing, dense], names=["missing", "dense"])
    # 创建一个预期的系列，索引为 idx，值为 1
    expected = Series(1, index=idx)
    # 检查 DataFrameGroupBy.apply 是否在分组列上操作，预期产生 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = grouped.apply(lambda x: 1)
    tm.assert_series_equal(result, expected)
# 定义一个函数用于测试观察参数对分组操作的影响
def test_observed(observed):
    # 创建一个有序分类变量 cat1 和 cat2，每个有两个级别，与实际数据可能有不同
    cat1 = Categorical(["a", "a", "b", "b"], categories=["a", "b", "z"], ordered=True)
    cat2 = Categorical(["c", "d", "c", "d"], categories=["c", "d", "y"], ordered=True)
    
    # 创建一个 DataFrame df 包含 cat1, cat2 和 values 列
    df = DataFrame({"A": cat1, "B": cat2, "values": [1, 2, 3, 4]})
    # 添加一列 C，其值为 'foo' 和 'bar' 交替
    df["C"] = ["foo", "bar"] * 2

    # 根据多个列进行分组，使用参数 observed 控制是否扩展分组键的输出空间
    gb = df.groupby(["A", "B", "C"], observed=observed)
    
    # 创建期望的索引对象 exp_index，包含 cat1, cat2 和 ['foo', 'bar'] * 2 的组合
    exp_index = MultiIndex.from_arrays(
        [cat1, cat2, ["foo", "bar"] * 2], names=["A", "B", "C"]
    )
    # 创建期望的结果 DataFrame expected，包含 values 列和 exp_index 索引，并按索引排序
    expected = DataFrame({"values": Series([1, 2, 3, 4], index=exp_index)}).sort_index()
    # 对分组结果进行求和
    result = gb.sum()
    
    # 如果 observed 参数为 False，则使用 cartesian_product_for_groupers 函数扩展期望结果
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2, ["foo", "bar"]], list("ABC"), fill_value=0
        )

    # 断言分组后的结果与期望的结果相等
    tm.assert_frame_equal(result, expected)

    # 根据两个列进行分组
    gb = df.groupby(["A", "B"], observed=observed)
    # 创建期望的索引对象 exp_index，包含 cat1 和 cat2 的组合
    exp_index = MultiIndex.from_arrays([cat1, cat2], names=["A", "B"])
    # 创建期望的结果 DataFrame expected，包含 values 列和 C 列，并按索引排序
    expected = DataFrame(
        {"values": [1, 2, 3, 4], "C": ["foo", "bar", "foo", "bar"]}, index=exp_index
    )
    # 对分组结果进行求和
    result = gb.sum()
    
    # 如果 observed 参数为 False，则使用 cartesian_product_for_groupers 函数扩展期望结果
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [cat1, cat2], list("AB"), fill_value=0
        )

    # 断言分组后的结果与期望的结果相等
    tm.assert_frame_equal(result, expected)


# 定义一个函数用于测试单列观察情况下的分组操作
def test_observed_single_column(observed):
    # 创建一个字典 d，包含分类变量 cat，整数列 ints 和值列 val
    d = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    # 创建 DataFrame df
    df = DataFrame(d)

    # 根据单个列进行分组，使用参数 observed 控制是否扩展分组键的输出空间
    groups_single_key = df.groupby("cat", observed=observed)
    # 对分组结果求均值
    result = groups_single_key.mean()

    # 创建期望的索引对象 exp_index，包含 ['a', 'b'] 和所有可能的类别 'abc'
    exp_index = CategoricalIndex(
        list("ab"), name="cat", categories=list("abc"), ordered=True
    )
    # 创建期望的结果 DataFrame expected，包含 ints 列和 val 列，并按索引匹配
    expected = DataFrame({"ints": [1.5, 1.5], "val": [20.0, 30]}, index=exp_index)
    # 如果 observed 参数为 False，则重新索引期望结果
    if not observed:
        index = CategoricalIndex(
            list("abc"), name="cat", categories=list("abc"), ordered=True
        )
        expected = expected.reindex(index)

    # 断言分组后的结果与期望的结果相等
    tm.assert_frame_equal(result, expected)


# 定义一个函数用于测试双列观察情况下的分组操作
def test_observed_two_columns(observed):
    # 创建一个字典 d，包含分类变量 cat，整数列 ints 和值列 val
    d = {
        "cat": Categorical(
            ["a", "b", "a", "b"], categories=["a", "b", "c"], ordered=True
        ),
        "ints": [1, 1, 2, 2],
        "val": [10, 20, 30, 40],
    }
    # 创建 DataFrame df
    df = DataFrame(d)
    
    # 根据两个列进行分组，使用参数 observed 控制是否扩展分组键的输出空间
    groups_double_key = df.groupby(["cat", "ints"], observed=observed)
    # 对分组结果进行均值计算
    result = groups_double_key.agg("mean")
    
    # 创建期望的结果 DataFrame expected，包含 val 列和 cat, ints 列作为索引
    expected = DataFrame(
        {
            "val": [10.0, 30.0, 20.0, 40.0],
            "cat": Categorical(
                ["a", "a", "b", "b"], categories=["a", "b", "c"], ordered=True
            ),
            "ints": [1, 2, 1, 2],
        }
    ).set_index(["cat", "ints"])
    # 如果没有观察到任何结果，则通过 cartesian_product_for_groupers 函数为期望值生成笛卡尔积
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [df.cat.values, [1, 2]], ["cat", "ints"]
        )

    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 result 和 expected 的数据帧是否相等
    tm.assert_frame_equal(result, expected)

    # GH 10132: 这段代码解决 GitHub 上的 issue 10132，对于指定的键值对列表执行以下操作
    for key in [("a", 1), ("b", 2), ("b", 1), ("a", 2)]:
        c, i = key
        # 获取具有双重键值的组数据并赋给 result
        result = groups_double_key.get_group(key)
        # 从 DataFrame df 中选择符合条件 cat == c 和 ints == i 的数据作为期望值
        expected = df[(df.cat == c) & (df.ints == i)]
        # 比较 result 和 expected 的数据帧是否相等
        tm.assert_frame_equal(result, expected)
def test_observed_with_as_index(observed):
    # 定义一个测试函数，用于测试 observed 参数的影响
    # gh-8869

    # 准备一个包含多列数据的字典
    d = {
        "foo": [10, 8, 4, 8, 4, 1, 1],
        "bar": [10, 20, 30, 40, 50, 60, 70],
        "baz": ["d", "c", "e", "a", "a", "d", "c"],
    }

    # 根据字典创建一个 DataFrame 对象
    df = DataFrame(d)

    # 对 'foo' 列进行分箱处理
    cat = pd.cut(df["foo"], np.linspace(0, 10, 3))

    # 将分箱结果作为新的列 'range' 添加到 DataFrame 中
    df["range"] = cat

    # 使用 groupby 函数按照 'range' 和 'baz' 列进行分组
    # observed 参数用于指定如何处理不在列中的观察值
    groups = df.groupby(["range", "baz"], as_index=False, observed=observed)

    # 对分组后的结果应用均值函数
    result = groups.agg("mean")

    # 以 as_index=True 的方式再次进行分组
    groups2 = df.groupby(["range", "baz"], as_index=True, observed=observed)

    # 对第二次分组的结果应用均值函数，并重置索引
    expected = groups2.agg("mean").reset_index()

    # 使用 pytest 的 assert_frame_equal 函数比较结果和预期
    tm.assert_frame_equal(result, expected)


def test_observed_codes_remap(observed):
    # 定义一个测试函数，用于测试 observed 参数对代码重映射的影响

    # 准备一个包含多列数据的字典
    d = {"C1": [3, 3, 4, 5], "C2": [1, 2, 3, 4], "C3": [10, 100, 200, 34]}

    # 根据字典创建一个 DataFrame 对象
    df = DataFrame(d)

    # 对 'C1' 列进行分箱处理
    values = pd.cut(df["C1"], [1, 2, 3, 6])
    values.name = "cat"

    # 使用 groupby 函数按照 'values' 和 'C2' 列进行分组
    groups_double_key = df.groupby([values, "C2"], observed=observed)

    # 为结果创建一个多级索引
    idx = MultiIndex.from_arrays([values, [1, 2, 3, 4]], names=["cat", "C2"])

    # 根据 observed 参数决定是否使用 cartesian_product_for_groupers 函数
    expected = DataFrame(
        {"C1": [3.0, 3.0, 4.0, 5.0], "C3": [10.0, 100.0, 200.0, 34.0]}, index=idx
    )
    if not observed:
        expected = cartesian_product_for_groupers(
            expected, [values.values, [1, 2, 3, 4]], ["cat", "C2"]
        )

    # 对分组后的结果应用均值函数
    result = groups_double_key.agg("mean")

    # 使用 pytest 的 assert_frame_equal 函数比较结果和预期
    tm.assert_frame_equal(result, expected)


def test_observed_perf():
    # 定义一个测试函数，用于测试 observed 参数对性能的影响
    # we create a cartesian product, so this is
    # non-performant if we don't use observed values
    # gh-14942

    # 创建一个包含多列数据的 DataFrame 对象
    df = DataFrame(
        {
            "cat": np.random.default_rng(2).integers(0, 255, size=30000),
            "int_id": np.random.default_rng(2).integers(0, 255, size=30000),
            "other_id": np.random.default_rng(2).integers(0, 10000, size=30000),
            "foo": 0,
        }
    )

    # 将 'cat' 列的数据类型转换为字符串并转换为分类类型
    df["cat"] = df.cat.astype(str).astype("category")

    # 使用 groupby 函数按照 'cat', 'int_id', 'other_id' 列进行分组
    # observed 参数为 True，表示仅考虑观察到的值
    grouped = df.groupby(["cat", "int_id", "other_id"], observed=True)

    # 对分组后的结果应用计数函数
    result = grouped.count()

    # 使用 assert 语句检查索引级别的唯一值是否符合预期
    assert result.index.levels[0].nunique() == df.cat.nunique()
    assert result.index.levels[1].nunique() == df.int_id.nunique()
    assert result.index.levels[2].nunique() == df.other_id.nunique()


def test_observed_groups(observed):
    # 定义一个测试函数，用于测试 observed 参数对分组结果的影响
    # gh-20583
    # test that we have the appropriate groups

    # 创建一个分类数据对象
    cat = Categorical(["a", "c", "a"], categories=["a", "b", "c"])

    # 根据分类数据和 'vals' 列创建一个 DataFrame 对象
    df = DataFrame({"cat": cat, "vals": [1, 2, 3]})

    # 使用 groupby 函数按照 'cat' 列进行分组
    g = df.groupby("cat", observed=observed)

    # 获取分组的结果
    result = g.groups

    # 根据 observed 参数决定预期的分组结果
    if observed:
        expected = {"a": Index([0, 2], dtype="int64"), "c": Index([1], dtype="int64")}
    else:
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "c": Index([1], dtype="int64"),
        }

    # 使用 pytest 的 assert_dict_equal 函数比较结果和预期
    tm.assert_dict_equal(result, expected)
    # 包含三个元组的列表，每个元组描述一个数据结构
    [
        # 第一个元组包含名称为"a"的列数据和对应的索引对象
        ("a", [15, 9, 0], CategoricalIndex([1, 2, 3], name="a")),
        # 第二个元组包含名称为"a"和"b"的列数据以及它们各自的索引对象
        (
            ["a", "b"],
            [7, 8, 0, 0, 0, 9, 0, 0, 0],
            [CategoricalIndex([1, 2, 3], name="a"), Index([4, 5, 6])],
        ),
        # 第三个元组包含名称为"a"和"a2"的列数据以及它们各自的索引对象
        (
            ["a", "a2"],
            [15, 0, 0, 0, 9, 0, 0, 0, 0],
            [
                CategoricalIndex([1, 2, 3], name="a"),
                CategoricalIndex([1, 2, 3], name="a"),
            ],
        ),
    ],
# 使用 pytest 的装饰器来标记测试用例，参数化测试以验证不同情况下的预期行为
@pytest.mark.parametrize("test_series", [True, False])
def test_unobserved_in_index(keys, expected_values, expected_index_levels, test_series):
    # GH#49354 - ensure unobserved cats occur when grouping by index levels
    # 创建一个 DataFrame 对象，包括 'a' 和 'a2' 两个分类列，以及 'b' 和 'c' 两个普通列，并将 'a' 和 'a2' 设置为索引
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "a2": Categorical([1, 1, 2], categories=[1, 2, 3]),
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    ).set_index(["a", "a2"])

    if "b" not in keys:
        # 如果 'b' 不在 keys 中，则从 DataFrame 中移除 'b' 列以保持结果中列的一致性
        df = df.drop(columns="b")

    # 根据 keys 对 DataFrame 进行分组，observed=False 表示允许未观察到的分类值
    gb = df.groupby(keys, observed=False)

    if test_series:
        # 如果 test_series 为 True，则只保留 'c' 列作为结果
        gb = gb["c"]

    # 对分组后的结果进行求和操作
    result = gb.sum()

    if len(keys) == 1:
        # 如果 keys 的长度为 1，则使用 expected_index_levels 作为索引
        index = expected_index_levels
    else:
        # 否则创建一个 MultiIndex，使用指定的 expected_index_levels 和 codes
        codes = [[0, 0, 0, 1, 1, 1, 2, 2, 2], 3 * [0, 1, 2]]
        index = MultiIndex(
            expected_index_levels,
            codes=codes,
            names=keys,
        )

    # 创建一个期望的 DataFrame，包括 'c' 列和指定的 index
    expected = DataFrame({"c": expected_values}, index=index)

    if test_series:
        # 如果 test_series 为 True，则只保留 'c' 列作为期望结果
        expected = expected["c"]

    # 使用 pytest 的断言函数 tm.assert_equal 检查 result 和 expected 是否相等
    tm.assert_equal(result, expected)


def test_observed_groups_with_nan(observed):
    # GH 24740
    # 创建一个 DataFrame 包括 'cat' 列和 'vals' 列，其中 'cat' 是一个分类列，observed 参数根据测试函数的输入而定
    df = DataFrame(
        {
            "cat": Categorical(["a", np.nan, "a"], categories=["a", "b", "d"]),
            "vals": [1, 2, 3],
        }
    )

    # 根据 'cat' 列进行分组，observed 参数根据测试函数的输入而定
    g = df.groupby("cat", observed=observed)

    # 获取分组后的结果的 groups 属性
    result = g.groups

    if observed:
        # 如果 observed 为 True，则期望的结果包含未观察到的 'b' 和 'd' 类别
        expected = {"a": Index([0, 2], dtype="int64")}
    else:
        # 如果 observed 为 False，则期望的结果包含所有三个类别 'a', 'b', 'd'
        expected = {
            "a": Index([0, 2], dtype="int64"),
            "b": Index([], dtype="int64"),
            "d": Index([], dtype="int64"),
        }

    # 使用 pytest 的断言函数 tm.assert_dict_equal 检查 result 和 expected 是否相等
    tm.assert_dict_equal(result, expected)


def test_observed_nth():
    # GH 26385
    # 创建一个包含 'cat' 和 'ser' 列的 DataFrame，其中 'cat' 是一个分类列，包含 NaN 值
    cat = Categorical(["a", np.nan, np.nan], categories=["a", "b", "c"])
    ser = Series([1, 2, 3])
    df = DataFrame({"cat": cat, "ser": ser})

    # 对 DataFrame 按 'cat' 列进行分组，observed=False 表示允许未观察到的分类值
    result = df.groupby("cat", observed=False)["ser"].nth(0)

    # 获取第一个分组的 'ser' 列作为期望结果
    expected = df["ser"].iloc[[0]]

    # 使用 pytest 的断言函数 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_dataframe_categorical_with_nan(observed):
    # GH 21151
    # 创建一个包含 's1' 和 's2' 列的 DataFrame，其中 's1' 是一个分类列，包含 NaN 值
    s1 = Categorical([np.nan, "a", np.nan, "a"], categories=["a", "b", "c"])
    s2 = Series([1, 2, 3, 4])
    df = DataFrame({"s1": s1, "s2": s2})

    # 根据 's1' 列进行分组，observed 参数根据测试函数的输入而定，并取每组的第一个值，并重置索引
    result = df.groupby("s1", observed=observed).first().reset_index()

    if observed:
        # 如果 observed 为 True，则期望结果中只包含 'a' 类别
        expected = DataFrame(
            {"s1": Categorical(["a"], categories=["a", "b", "c"]), "s2": [2]}
        )
    else:
        # 如果 observed 为 False，则期望结果中包含所有三个类别 'a', 'b', 'c'，并且 's2' 列中的 NaN 值表示未观察到的类别
        expected = DataFrame(
            {
                "s1": Categorical(["a", "b", "c"], categories=["a", "b", "c"]),
                "s2": [2, np.nan, np.nan],
            }
        )

    # 使用 pytest 的断言函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_dataframe_categorical_ordered_observed_sort(ordered, observed, sort):
    # GH 25871: Fix groupby sorting on ordered Categoricals
    # GH 25167: Groupby with observed=True doesn't sort
    # 测试用例参数化，测试 DataFrame 中有序分类列的分组排序行为，以及 observed=True 时分组不排序的情况
    # 使用 Categorical 类创建一个带有未观察到的类别 ('missing') 的分类变量
    # 并创建一个具有相同值的 Series
    label = Categorical(
        ["d", "a", "b", "a", "d", "b"],
        categories=["a", "b", "missing", "d"],
        ordered=ordered,
    )
    # 创建一个 Series 包含值 ["d", "a", "b", "a", "d", "b"]
    val = Series(["d", "a", "b", "a", "d", "b"])
    # 用 label 和 val 构建一个 DataFrame
    df = DataFrame({"label": label, "val": val})

    # 根据 Categorical 变量进行聚合
    result = df.groupby("label", observed=observed, sort=sort)["val"].aggregate("first")

    # 如果排序正常，我们期望索引标签与聚合结果相等，
    # 对于 'observed=False' 的情况，预期 'missing' 标签的聚合结果为 None
    label = Series(result.index.array, dtype="object")
    aggr = Series(result.array)
    # 如果 observed=False，将缺失值对应的聚合结果设置为 'missing'
    if not observed:
        aggr[aggr.isna()] = "missing"
    # 如果标签与聚合结果不一致，生成错误消息并抛出 pytest 异常
    if not all(label == aggr):
        msg = (
            "Labels and aggregation results not consistently sorted\n"
            f"for (ordered={ordered}, observed={observed}, sort={sort})\n"
            f"Result:\n{result}"
        )
        pytest.fail(msg)
def test_datetime():
    # GH9049: ensure backward compatibility
    # 创建日期范围，从 "2014-01-01" 开始，持续 4 个时间段
    levels = pd.date_range("2014-01-01", periods=4)
    # 生成一个随机整数数组，范围在 [0, 4)，大小为 10
    codes = np.random.default_rng(2).integers(0, 4, size=10)

    # 使用 codes 和 levels 创建一个有序的分类变量
    cats = Categorical.from_codes(codes, levels, ordered=True)

    # 创建一个随机数据的 DataFrame
    data = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    # 对数据根据分类变量进行分组，并计算均值
    result = data.groupby(cats, observed=False).mean()

    # 预期结果：对数据按照分类变量的编码进行分组，并计算均值，再根据 levels 重新索引
    expected = data.groupby(np.asarray(cats), observed=False).mean()
    expected = expected.reindex(levels)
    # 将索引转换为有序的分类索引
    expected.index = CategoricalIndex(
        expected.index, categories=expected.index, ordered=True
    )

    # 断言结果 DataFrame 和预期结果相等
    tm.assert_frame_equal(result, expected)

    # 对数据按照分类变量进行分组，计算描述统计信息
    grouped = data.groupby(cats, observed=False)
    desc_result = grouped.describe()

    # 对分类编码进行排序，获取排序后的标签和数据
    idx = cats.codes.argsort()
    ord_labels = cats.take(idx)
    ord_data = data.take(idx)
    # 预期结果：对排序后的数据按照排序后的标签进行分组，计算描述统计信息
    expected = ord_data.groupby(ord_labels, observed=False).describe()
    # 断言结果 DataFrame 和预期结果相等
    tm.assert_frame_equal(desc_result, expected)
    # 断言索引相等
    tm.assert_index_equal(desc_result.index, expected.index)
    tm.assert_index_equal(
        desc_result.index.get_level_values(0), expected.index.get_level_values(0)
    )

    # GH 10460
    # 生成一个重复的分类变量，共 4 个值重复 8 次
    expc = Categorical.from_codes(np.arange(4).repeat(8), levels, ordered=True)
    # 预期结果：创建一个有序的分类索引
    exp = CategoricalIndex(expc)
    # 断言结果的索引等于预期的分类索引
    tm.assert_index_equal((desc_result.stack().index.get_level_values(0)), exp)
    # 预期结果：创建一个索引，包含多个标签
    exp = Index(["count", "mean", "std", "min", "25%", "50%", "75%", "max"] * 4)
    # 断言结果的索引等于预期的索引
    tm.assert_index_equal((desc_result.stack().index.get_level_values(1)), exp)


def test_categorical_index():
    s = np.random.default_rng(2)
    levels = ["foo", "bar", "baz", "qux"]
    codes = s.integers(0, 4, size=20)
    # 用 codes 和 levels 创建一个有序的分类变量
    cats = Categorical.from_codes(codes, levels, ordered=True)
    # 创建一个 DataFrame，包含重复的整数值和分类变量列
    df = DataFrame(np.repeat(np.arange(20), 4).reshape(-1, 4), columns=list("abcd"))
    df["cats"] = cats

    # 使用分类变量作为索引进行分组，并求和
    result = df.set_index("cats").groupby(level=0, observed=False).sum()
    # 预期结果：对 df 中的列按照分类变量的编码进行分组，并求和
    expected = df[list("abcd")].groupby(cats.codes, observed=False).sum()
    # 将预期结果的索引转换为有序的分类索引
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True), name="cats"
    )
    # 断言结果 DataFrame 和预期结果相等
    tm.assert_frame_equal(result, expected)

    # 使用分类变量列进行分组，并求和
    result = df.groupby("cats", observed=False).sum()
    # 预期结果：对 df 中的列按照分类变量的编码进行分组，并求和
    expected = df[list("abcd")].groupby(cats.codes, observed=False).sum()
    # 将预期结果的索引转换为有序的分类索引
    expected.index = CategoricalIndex(
        Categorical.from_codes([0, 1, 2, 3], levels, ordered=True), name="cats"
    )
    # 断言结果 DataFrame 和预期结果相等
    tm.assert_frame_equal(result, expected)


def test_describe_categorical_columns():
    # GH 11558
    # 创建一个有序的分类索引，指定了 categories 和 ordered=True
    cats = CategoricalIndex(
        ["qux", "foo", "baz", "bar"],
        categories=["foo", "bar", "baz", "qux"],
        ordered=True,
    )
    # 创建一个包含随机数据的 DataFrame
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)), columns=cats)
    # 对 DataFrame 按照 [1, 2, 3, 4] * 5 进行分组，并计算描述统计信息
    result = df.groupby([1, 2, 3, 4] * 5).describe()

    # 断言结果的列索引等于 cats
    tm.assert_index_equal(result.stack().columns, cats)
    # 断言分类等于预期的分类值
    tm.assert_categorical_equal(result.stack().columns.values, cats.values)


def test_unstack_categorical():
    # 这个函数暂时没有提供代码，请补充完整后添加注释。
    # GH11558 (example is taken from the original issue)
    # 创建一个 DataFrame 对象，包含三列数据："a" 是范围为 0 到 9 的整数序列，
    # "medium" 是长度为 10 的列表，交替包含字符串 "A" 和 "B"，"artist" 是长度为 10 的列表，交替包含字符 "X" 和 "Y"
    df = DataFrame(
        {"a": range(10), "medium": ["A", "B"] * 5, "artist": list("XYXXY") * 2}
    )
    
    # 将 "medium" 列的数据类型转换为分类数据
    df["medium"] = df["medium"].astype("category")

    # 对 DataFrame 进行分组，按 "artist" 和 "medium" 列分组，并计算每组中 "a" 列的计数，生成一个透视表
    gcat = df.groupby(["artist", "medium"], observed=False)["a"].count().unstack()
    
    # 对透视表进行描述统计，生成描述统计结果
    result = gcat.describe()

    # 创建预期的分类索引对象，包含元素 ["A", "B"]，无序，命名为 "medium"
    exp_columns = CategoricalIndex(["A", "B"], ordered=False, name="medium")
    
    # 检查 result 的列索引是否等于预期的分类索引 exp_columns
    tm.assert_index_equal(result.columns, exp_columns)
    
    # 检查 result 的列值是否与预期的分类索引 exp_columns 的值相等
    tm.assert_categorical_equal(result.columns.values, exp_columns.values)

    # 计算透视表 gcat 中 "A" 和 "B" 列的和，生成结果 Series
    result = gcat["A"] + gcat["B"]
    
    # 创建预期的 Series 对象，包含元素 [6, 4]，索引为 ["X", "Y"]，命名为 "artist"
    expected = Series([6, 4], index=Index(["X", "Y"], name="artist"))
    
    # 检查 result 是否与预期的 Series 对象 expected 相等
    tm.assert_series_equal(result, expected)
def test_bins_unequal_len():
    # GH3011: Test case for GitHub issue 3011

    # 创建包含 NaN 值和整数的 Pandas Series 对象
    series = Series([np.nan, np.nan, 1, 1, 2, 2, 3, 3, 4, 4])

    # 基于删除 NaN 值后的 Series 数据创建分箱对象 bins
    bins = pd.cut(series.dropna().values, 4)

    # 在分组过程中 bins 和 series 的长度不一致时，预期抛出 ValueError 异常
    with pytest.raises(ValueError, match="Grouper and axis must be same length"):
        series.groupby(bins).mean()


@pytest.mark.parametrize(
    ["series", "data"],
    [
        # 长度和索引与 grouper 相同的 Series 分组
        (Series(range(4)), {"A": [0, 3], "B": [1, 2]}),

        # 长度与 grouper 相同但索引不同的 Series 分组
        (Series(range(4)).rename(lambda idx: idx + 1), {"A": [2], "B": [0, 1]}),

        # GH44179: 长度与 grouper 不同的 Series 分组
        (Series(range(7)), {"A": [0, 3], "B": [1, 2]}),
    ],
)
def test_categorical_series(series, data):
    # 使用分类数据类型的 Series 对象进行分组，其中组 A 包含索引 0 和 3，组 B 包含索引 1 和 2，
    # 并根据给定数据映射对应的值。
    groupby = series.groupby(Series(list("ABBA"), dtype="category"), observed=False)
    result = groupby.aggregate(list)

    # 期望的结果是一个 Pandas Series，其数据和索引与给定的 data 对应
    expected = Series(data, index=CategoricalIndex(data.keys()))
    tm.assert_series_equal(result, expected)


def test_as_index():
    # GH13204: Test case for GitHub issue 13204

    # 创建一个包含分类数据的 DataFrame 对象
    df = DataFrame(
        {
            "cat": Categorical([1, 2, 2], [1, 2, 3]),
            "A": [10, 11, 11],
            "B": [101, 102, 103],
        }
    )

    # 使用指定的列进行分组，并确保不将其作为结果的索引
    result = df.groupby(["cat", "A"], as_index=False, observed=True).sum()

    # 期望的结果是一个 Pandas DataFrame，与给定的 expected 相匹配
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 11],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )
    tm.assert_frame_equal(result, expected)

    # 使用函数作为 grouper
    f = lambda r: df.loc[r, "A"]
    result = df.groupby(["cat", f], as_index=False, observed=True).sum()

    # 期望的结果是一个 Pandas DataFrame，与给定的 expected 相匹配
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "level_1": [10, 11],
            "A": [10, 22],
            "B": [101, 205],
        },
    )
    tm.assert_frame_equal(result, expected)

    # 另一个不在轴上的 grouper（在索引中存在冲突的名称）
    s = Series(["a", "b", "b"], name="cat")
    result = df.groupby(["cat", s], as_index=False, observed=True).sum()

    # 期望的结果是一个 Pandas DataFrame，与给定的 expected 相匹配
    expected = DataFrame(
        {
            "cat": ["a", "b"],
            "A": [10, 22],
            "B": [101, 205],
        },
    )
    tm.assert_frame_equal(result, expected)

    # 原始索引是否被丢弃？
    group_columns = ["cat", "A"]
    expected = DataFrame(
        {
            "cat": Categorical([1, 2], categories=df.cat.cat.categories),
            "A": [10, 11],
            "B": [101, 205],
        },
        columns=["cat", "A", "B"],
    )
    # 使用循环遍历给定的名称列表 [None, "X", "B"]
    for name in [None, "X", "B"]:
        # 将 DataFrame 的索引设置为带有指定名称的 Index 对象，名称为循环变量 name
        df.index = Index(list("abc"), name=name)
        # 对 DataFrame 进行按组分组，并计算每组的和，结果存储在 result 中
        result = df.groupby(group_columns, as_index=False, observed=True).sum()

        # 使用断言检查 result 是否与期望的结果 expected 相等
        tm.assert_frame_equal(result, expected)
# 定义测试函数，用于验证保留分类数据的行为
def test_preserve_categories():
    # GH-13179
    # 创建包含字符 'a', 'b', 'c' 的列表作为分类的标签
    categories = list("abc")

    # 创建 DataFrame，其中列"A"包含分类数据，使用有序分类并指定初始数据为 'ba'
    df = DataFrame({"A": Categorical(list("ba"), categories=categories, ordered=True)})
    
    # 创建有序分类索引对象，其类别与初始数据相同
    sort_index = CategoricalIndex(categories, categories, ordered=True, name="A")
    
    # 创建另一个有序分类索引对象，类别顺序为 'bac'
    nosort_index = CategoricalIndex(list("bac"), categories, ordered=True, name="A")
    
    # 断言按"A"列分组并排序后的第一个索引与 sort_index 相等
    tm.assert_index_equal(
        df.groupby("A", sort=True, observed=False).first().index, sort_index
    )
    
    # GH#42482 - 当 sort=False 时，即使 ordered=True 也不对结果排序
    # 断言按"A"列分组并不排序的第一个索引与 nosort_index 相等
    tm.assert_index_equal(
        df.groupby("A", sort=False, observed=False).first().index, nosort_index
    )


# 定义测试函数，用于验证不同 ordered 设置下保留分类数据的行为
def test_preserve_categories_ordered_false():
    # GH-13179
    # 创建包含字符 'a', 'b', 'c' 的列表作为分类的标签
    categories = list("abc")
    
    # 创建 DataFrame，其中列"A"包含分类数据，使用无序分类并指定初始数据为 'ba'
    df = DataFrame({"A": Categorical(list("ba"), categories=categories, ordered=False)})
    
    # 创建无序分类索引对象，类别与初始数据相同
    sort_index = CategoricalIndex(categories, categories, ordered=False, name="A")
    
    # GH#48749 - 不改变分类的顺序
    # GH#42482 - 当 sort=False 时，即使 ordered=True 也不对结果排序
    # 创建另一个无序分类索引对象，类别顺序为 'bac'
    nosort_index = CategoricalIndex(list("bac"), list("abc"), ordered=False, name="A")
    
    # 断言按"A"列分组并排序后的第一个索引与 sort_index 相等
    tm.assert_index_equal(
        df.groupby("A", sort=True, observed=False).first().index, sort_index
    )
    
    # 断言按"A"列分组并不排序的第一个索引与 nosort_index 相等
    tm.assert_index_equal(
        df.groupby("A", sort=False, observed=False).first().index, nosort_index
    )


# 定义参数化测试函数，用于验证保留分类数据类型的行为
@pytest.mark.parametrize("col", ["C1", "C2"])
def test_preserve_categorical_dtype(col):
    # GH13743, GH13854
    # 创建包含分类数据的 DataFrame，其中"C1"列为无序分类，"C2"列为有序分类
    df = DataFrame(
        {
            "A": [1, 2, 1, 1, 2],
            "B": [10, 16, 22, 28, 34],
            "C1": Categorical(list("abaab"), categories=list("bac"), ordered=False),
            "C2": Categorical(list("abaab"), categories=list("bac"), ordered=True),
        }
    )
    
    # 创建期望的 DataFrame，其中"C1"列和"C2"列分别为包含类别 'bac' 的分类数据
    exp_full = DataFrame(
        {
            "A": [2.0, 1.0, np.nan],
            "B": [25.0, 20.0, np.nan],
            "C1": Categorical(list("bac"), categories=list("bac"), ordered=False),
            "C2": Categorical(list("bac"), categories=list("bac"), ordered=True),
        }
    )
    
    # 对 DataFrame 按指定列 col 进行分组并计算均值，as_index=False 表示不保留分组键作为索引
    result1 = df.groupby(by=col, as_index=False, observed=False).mean(numeric_only=True)
    
    # 对 DataFrame 按指定列 col 进行分组并计算均值，as_index=True 表示保留分组键作为索引
    result2 = (
        df.groupby(by=col, as_index=True, observed=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    
    # 根据期望的列顺序重建期望的 DataFrame
    expected = exp_full.reindex(columns=result1.columns)
    
    # 断言两个结果 DataFrame 与期望的 DataFrame 相等
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)


# 定义参数化测试函数，用于验证有序分类数据上操作时保持分类特性的行为
@pytest.mark.parametrize(
    "func, values",
    [
        ("first", ["second", "first"]),
        ("last", ["fourth", "third"]),
        ("min", ["fourth", "first"]),
        ("max", ["second", "third"]),
    ],
)
def test_preserve_on_ordered_ops(func, values):
    # gh-18502
    # 在操作中保留分类数据的特性
    c = Categorical(["first", "second", "third", "fourth"], ordered=True)
    
    # 创建包含分类列"col"和"payload"的 DataFrame
    df = DataFrame({"payload": [-1, -2, -1, -2], "col": c})
    
    # 对 DataFrame 按"payload"列分组
    g = df.groupby("payload")
    
    # 调用指定的聚合函数，例如 "first"、"last"、"min"、"max"
    result = getattr(g, func)()
    # 创建一个预期的 DataFrame 对象，其中包含一个名为 'payload' 的列和一个 Series 对象，该 Series 包含特定数据类型的值
    expected = DataFrame(
        {"payload": [-2, -1], "col": Series(values, dtype=c.dtype)}
    ).set_index("payload")
    
    # 使用 tm.assert_frame_equal 检查两个 DataFrame 对象（result 和 expected）是否相等
    tm.assert_frame_equal(result, expected)
    
    # 对于 SeriesGroupBy 对象，需要保留分类信息
    # 使用 df.groupby("payload")["col"] 创建一个 SeriesGroupBy 对象 sgb
    sgb = df.groupby("payload")["col"]
    
    # 调用 SeriesGroupBy 对象 sgb 的某个函数（func），获取结果
    result = getattr(sgb, func)()
    
    # 从预期的 DataFrame 对象中提取 'col' 列作为预期的 Series 对象
    expected = expected["col"]
    
    # 使用 tm.assert_series_equal 检查两个 Series 对象（result 和 expected）是否相等
    tm.assert_series_equal(result, expected)
def test_categorical_no_compress():
    # 创建一个包含9个随机标准正态分布数据的 Series
    data = Series(np.random.default_rng(2).standard_normal(9))

    # 创建一个包含9个分类编码的 numpy 数组
    codes = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    # 使用分类编码创建有序的 Categorical 对象
    cats = Categorical.from_codes(codes, [0, 1, 2], ordered=True)

    # 根据分类进行分组计算平均值
    result = data.groupby(cats, observed=False).mean()
    # 使用分类编码进行分组计算平均值作为期望结果
    exp = data.groupby(codes, observed=False).mean()

    # 将期望结果的索引转换为 CategoricalIndex 类型，保持分类信息
    exp.index = CategoricalIndex(
        exp.index, categories=cats.categories, ordered=cats.ordered
    )
    # 使用测试工具比较结果和期望值是否相等
    tm.assert_series_equal(result, exp)

    # 创建另一个包含9个分类编码的 numpy 数组
    codes = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3])
    # 使用这些分类编码创建有序的 Categorical 对象
    cats = Categorical.from_codes(codes, [0, 1, 2, 3], ordered=True)

    # 根据分类进行分组计算平均值
    result = data.groupby(cats, observed=False).mean()
    # 使用分类编码进行分组计算平均值，然后重新索引到 cats 的分类
    exp = data.groupby(codes, observed=False).mean().reindex(cats.categories)
    # 将期望结果的索引转换为 CategoricalIndex 类型，保持分类信息
    exp.index = CategoricalIndex(
        exp.index, categories=cats.categories, ordered=cats.ordered
    )
    # 使用测试工具比较结果和期望值是否相等
    tm.assert_series_equal(result, exp)


def test_categorical_no_compress_string():
    # 创建一个包含字符串的 Categorical 对象
    cats = Categorical(
        ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        categories=["a", "b", "c", "d"],
        ordered=True,
    )
    # 创建一个包含两列数据的 DataFrame，其中一列使用了上面创建的 cats 对象
    data = DataFrame({"a": [1, 1, 1, 2, 2, 2, 3, 4, 5], "b": cats})

    # 根据 "b" 列进行分组计算平均值
    result = data.groupby("b", observed=False).mean()
    # 提取结果中 "a" 列的数值部分
    result = result["a"].values
    # 期望结果是一个包含特定数值和 NaN 的 numpy 数组
    exp = np.array([1, 2, 4, np.nan])
    # 使用测试工具比较结果和期望值是否相等
    tm.assert_numpy_array_equal(result, exp)


def test_groupby_empty_with_category():
    # GH-9614
    # 对于空值进行分组时，确保不会将分类数据类型转换为浮点数
    # 创建一个包含两列数据的 DataFrame，其中 "A" 列包含三个空值，"B" 列使用了 Categorical 对象
    df = DataFrame({"A": [None] * 3, "B": Categorical(["train", "train", "test"])})
    # 对 "A" 列进行分组并取每组的第一个值，然后获取 "B" 列的结果
    result = df.groupby("A").first()["B"]
    # 创建期望结果，其中 "B" 列的值是空的 Categorical 对象，其分类为 ["test", "train"]
    expected = Series(
        Categorical([], categories=["test", "train"]),
        index=Series([], dtype="object", name="A"),
        name="B",
    )
    # 使用测试工具比较结果和期望值是否相等
    tm.assert_series_equal(result, expected)


def test_sort():
    # https://stackoverflow.com/questions/23814368/sorting-pandas-
    #        categorical-labels-after-groupby
    # 该测试确保结果按照分类顺序正确排序，以便绘制的图表 x 轴有序

    # 创建一个包含随机整数的 DataFrame
    df = DataFrame({"value": np.random.default_rng(2).integers(0, 10000, 10)})
    # 创建一组标签，用于创建 Categorical 对象
    labels = [f"{i} - {i+499}" for i in range(0, 10000, 500)]
    cat_labels = Categorical(labels, labels)

    # 根据 "value" 列对 DataFrame 进行升序排序
    df = df.sort_values(by=["value"], ascending=True)
    # 使用 pd.cut 函数将 "value" 列的数值划分为不同的区间，使用 cat_labels 作为标签
    df["value_group"] = pd.cut(
        df.value, range(0, 10500, 500), right=False, labels=cat_labels
    )

    # 根据 "value_group" 列进行分组，并计算每组的数量
    res = df.groupby(["value_group"], observed=False)["value_group"].count()
    # 期望结果是根据区间的起始值进行排序后的 Series
    exp = res[sorted(res.index, key=lambda x: float(x.split()[0]))]
    # 将期望结果的索引转换为 CategoricalIndex 类型
    exp.index = CategoricalIndex(exp.index, name=exp.index.name)
    # 使用测试工具比较结果和期望值是否相等
    tm.assert_series_equal(res, exp)


@pytest.mark.parametrize("ordered", [True, False])
def test_sort2(sort, ordered):
    # dataframe groupby sort was being ignored # GH 8868
    # GH#48749 - don't change order of categories
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    pass  # 这是一个未实现的测试函数，因此不需要添加注释
    # 创建一个 DataFrame 对象，包含指定的数据和列名
    df = DataFrame(
        [
            ["(7.5, 10]", 10, 10],
            ["(7.5, 10]", 8, 20],
            ["(2.5, 5]", 5, 30],
            ["(5, 7.5]", 6, 40],
            ["(2.5, 5]", 4, 50],
            ["(0, 2.5]", 1, 60],
            ["(5, 7.5]", 7, 70],
        ],
        columns=["range", "foo", "bar"],
    )
    
    # 将 "range" 列转换为有序分类类型，使用传入的 ordered 参数
    df["range"] = Categorical(df["range"], ordered=ordered)
    
    # 根据 "range" 列进行分组，如果 sort 参数为 True，则排序结果；observed 参数为 False 表示不要求包含所有可能的值
    result = df.groupby("range", sort=sort, observed=False).first()
    
    # 如果 sort 参数为 True，则按照指定顺序准备预期的数据和索引；否则按照另一种顺序
    if sort:
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = ["(0, 2.5]", "(2.5, 5]", "(5, 7.5]", "(7.5, 10]"]
    else:
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = ["(7.5, 10]", "(2.5, 5]", "(5, 7.5]", "(0, 2.5]"]
    
    # 创建一个 DataFrame 对象，表示预期的结果，包括数据、列名和有序的分类索引
    expected = DataFrame(
        data_values,
        columns=["foo", "bar"],
        index=CategoricalIndex(index_values, name="range", ordered=ordered),
    )
    
    # 使用测试工具 tm.assert_frame_equal 检查计算结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("ordered", [True, False])
# 使用 pytest 的 parametrize 装饰器，允许按参数化方式运行测试函数，参数为 ordered，分别为 True 和 False
def test_sort_datetimelike(sort, ordered):
    # GH10505
    # GH#42482 - don't sort result when sort=False, even when ordered=True
    # GH10505 和 GH#42482 的相关说明：当 sort=False 时，即使 ordered=True，也不对结果进行排序

    # use same data as test_groupby_sort_categorical, which category is
    # corresponding to datetime.month
    # 使用与 test_groupby_sort_categorical 相同的数据，其中类别对应于 datetime.month
    df = DataFrame(
        {
            "dt": [
                datetime(2011, 7, 1),
                datetime(2011, 7, 1),
                datetime(2011, 2, 1),
                datetime(2011, 5, 1),
                datetime(2011, 2, 1),
                datetime(2011, 1, 1),
                datetime(2011, 5, 1),
            ],
            "foo": [10, 8, 5, 6, 4, 1, 7],
            "bar": [10, 20, 30, 40, 50, 60, 70],
        },
        columns=["dt", "foo", "bar"],
    )

    # ordered=True
    # 设置 df["dt"] 为有序分类变量，其 ordered 参数根据传入的 ordered 决定
    df["dt"] = Categorical(df["dt"], ordered=ordered)
    if sort:
        # 当 sort=True 时，定义预期的数据值和索引值
        data_values = [[1, 60], [5, 30], [6, 40], [10, 10]]
        index_values = [
            datetime(2011, 1, 1),
            datetime(2011, 2, 1),
            datetime(2011, 5, 1),
            datetime(2011, 7, 1),
        ]
    else:
        # 当 sort=False 时，定义不同的预期数据值和索引值
        data_values = [[10, 10], [5, 30], [6, 40], [1, 60]]
        index_values = [
            datetime(2011, 7, 1),
            datetime(2011, 2, 1),
            datetime(2011, 5, 1),
            datetime(2011, 1, 1),
        ]
    # 构建预期的 DataFrame
    expected = DataFrame(
        data_values,
        columns=["foo", "bar"],
        index=CategoricalIndex(index_values, name="dt", ordered=ordered),
    )
    # 使用 groupby 方法对 df 进行分组，并选择第一个值作为结果
    result = df.groupby("dt", sort=sort, observed=False).first()
    # 使用 assert_frame_equal 函数比较结果和预期，确保它们相等
    tm.assert_frame_equal(result, expected)


def test_empty_sum():
    # https://github.com/pandas-dev/pandas/issues/18678
    # 使用 pandas 问题追踪系统中的链接作为注释

    # 创建包含分类变量和整数列的 DataFrame
    df = DataFrame(
        {"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"]), "B": [1, 2, 1]}
    )
    # 预期的索引为分类变量的所有类别
    expected_idx = CategoricalIndex(["a", "b", "c"], name="A")

    # 0 by default
    # 对 "A" 列进行分组，并对 "B" 列求和，observed=False 表示使用所有可能的类别
    result = df.groupby("A", observed=False).B.sum()
    # 创建预期的 Series
    expected = Series([3, 1, 0], index=expected_idx, name="B")
    # 使用 assert_series_equal 函数比较结果和预期，确保它们相等
    tm.assert_series_equal(result, expected)

    # min_count=0
    # 同上，但此时 min_count=0，表示对于空组，返回结果为 0
    result = df.groupby("A", observed=False).B.sum(min_count=0)
    # 创建预期的 Series
    expected = Series([3, 1, 0], index=expected_idx, name="B")
    # 使用 assert_series_equal 函数比较结果和预期，确保它们相等
    tm.assert_series_equal(result, expected)

    # min_count=1
    # 同上，但此时 min_count=1，表示对于空组，返回结果为 NaN
    result = df.groupby("A", observed=False).B.sum(min_count=1)
    # 创建预期的 Series
    expected = Series([3, 1, np.nan], index=expected_idx, name="B")
    # 使用 assert_series_equal 函数比较结果和预期，确保它们相等
    tm.assert_series_equal(result, expected)

    # min_count>1
    # 同上，但此时 min_count=2，表示对于非空组至少有两个元素，否则返回结果为 NaN
    result = df.groupby("A", observed=False).B.sum(min_count=2)
    # 创建预期的 Series
    expected = Series([3, np.nan, np.nan], index=expected_idx, name="B")
    # 使用 assert_series_equal 函数比较结果和预期，确保它们相等
    tm.assert_series_equal(result, expected)


def test_empty_prod():
    # https://github.com/pandas-dev/pandas/issues/18678
    # 使用 pandas 问题追踪系统中的链接作为注释

    # 创建包含分类变量和整数列的 DataFrame
    df = DataFrame(
        {"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"]), "B": [1, 2, 1]}
    )

    expected_idx = CategoricalIndex(["a", "b", "c"], name="A")

    # 1 by default
    # 对 "A" 列进行分组，并对 "B" 列求积，observed=False 表示使用所有可能的类别
    result = df.groupby("A", observed=False).B.prod()
    # 创建预期的 Series
    expected = Series([2, 1, 1], index=expected_idx, name="B")
    # 使用 assert_series_equal 函数比较结果和预期，确保它们相等
    tm.assert_series_equal(result, expected)
    # 断言验证结果 series 是否与期望结果相等
    tm.assert_series_equal(result, expected)

    # 使用 groupby 按列 "A" 进行分组，计算每组中列 "B" 的乘积，如果组中元素数量少于 1 则结果为 NaN
    result = df.groupby("A", observed=False).B.prod(min_count=0)
    # 期望的结果 series，表示每个组中列 "B" 的乘积，不存在缺失值
    expected = Series([2, 1, 1], expected_idx, name="B")
    # 断言验证结果 series 是否与期望结果相等
    tm.assert_series_equal(result, expected)

    # 使用 groupby 按列 "A" 进行分组，计算每组中列 "B" 的乘积，如果组中元素数量少于 2 则结果为 NaN
    result = df.groupby("A", observed=False).B.prod(min_count=1)
    # 期望的结果 series，表示每个组中列 "B" 的乘积，其中有组中元素数量少于 2 的情况下结果为 NaN
    expected = Series([2, 1, np.nan], expected_idx, name="B")
    # 断言验证结果 series 是否与期望结果相等
    tm.assert_series_equal(result, expected)
def test_groupby_multiindex_categorical_datetime():
    # 测试函数，用于验证多级索引、分类数据和日期时间的分组聚合功能

    # 创建一个 DataFrame，包含 key1 和 key2 两列，以及 values 列
    df = DataFrame(
        {
            "key1": Categorical(list("abcbabcba")),
            "key2": Categorical(
                list(pd.date_range("2018-06-01 00", freq="1min", periods=3)) * 3
            ),
            "values": np.arange(9),
        }
    )

    # 对 DataFrame 进行分组聚合操作，计算均值
    result = df.groupby(["key1", "key2"], observed=False).mean()

    # 创建预期结果的 MultiIndex
    idx = MultiIndex.from_product(
        [
            Categorical(["a", "b", "c"]),
            Categorical(pd.date_range("2018-06-01 00", freq="1min", periods=3)),
        ],
        names=["key1", "key2"],
    )
    
    # 创建预期结果的 DataFrame
    expected = DataFrame({"values": [0, 4, 8, 3, 4, 5, 6, np.nan, 2]}, index=idx)
    
    # 使用测试工具比较实际结果和预期结果
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "as_index, expected",
    [
        (
            True,
            Series(
                index=MultiIndex.from_arrays(
                    [Series([1, 1, 2], dtype="category"), [1, 2, 2]], names=["a", "b"]
                ),
                data=[1, 2, 3],
                name="x",
            ),
        ),
        (
            False,
            DataFrame(
                {
                    "a": Series([1, 1, 2], dtype="category"),
                    "b": [1, 2, 2],
                    "x": [1, 2, 3],
                }
            ),
        ),
    ],
)
def test_groupby_agg_observed_true_single_column(as_index, expected):
    # 测试函数，验证分组聚合并根据不同参数生成不同的索引结果
    # GH-23970
    
    # 创建包含三列的 DataFrame，其中一列是分类数据
    df = DataFrame(
        {"a": Series([1, 1, 2], dtype="category"), "b": [1, 2, 2], "x": [1, 2, 3]}
    )

    # 对 DataFrame 进行分组聚合，并返回指定列的和
    result = df.groupby(["a", "b"], as_index=as_index, observed=True)["x"].sum()

    # 使用测试工具比较实际结果和预期结果
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("fill_value", [None, np.nan, pd.NaT])
def test_shift(fill_value):
    # 测试函数，验证分类数据的 shift 操作
    # 创建一个分类数据对象
    ct = Categorical(
        ["a", "b", "c", "d"], categories=["a", "b", "c", "d"], ordered=False
    )
    
    # 创建预期结果的分类数据对象
    expected = Categorical(
        [None, "a", "b", "c"], categories=["a", "b", "c", "d"], ordered=False
    )
    
    # 对分类数据进行 shift 操作
    res = ct.shift(1, fill_value=fill_value)
    
    # 使用测试工具比较实际结果和预期结果
    tm.assert_equal(res, expected)


@pytest.fixture
def df_cat(df):
    """
    Fixture 函数，返回包含多个分类列和一个整数列的 DataFrame，用于测试 GroupBy 对象的 observed 参数功能。

    Parameters
    ----------
    df: DataFrame
        另一个 Fixture 提供的非分类数据 DataFrame，用于生成此 DataFrame

    Returns
    -------
    df_cat: DataFrame
    """
    # 复制原始 DataFrame 的前四行，用于测试
    df_cat = df.copy()[:4]
    
    # 将列 'A' 和 'B' 转换为分类类型
    df_cat["A"] = df_cat["A"].astype("category")
    df_cat["B"] = df_cat["B"].astype("category")
    
    # 添加一列 'C'，包含整数数据
    df_cat["C"] = Series([1, 2, 3, 4])
    
    # 删除列 'D'
    df_cat = df_cat.drop(["D"], axis=1)
    
    return df_cat


@pytest.mark.parametrize("operation", ["agg", "apply"])
def test_seriesgroupby_observed_true(df_cat, operation):
    # 测试函数，验证 SeriesGroupBy 对象在 observed=True 参数下的行为
    # GH#24880
    # GH#49223 - order of results was wrong when grouping by index levels
    # 创建名为 lev_a 的索引对象，包含元素 ["bar", "bar", "foo", "foo"]，使用 df_cat["A"] 的数据类型，并命名为 "A"
    lev_a = Index(["bar", "bar", "foo", "foo"], dtype=df_cat["A"].dtype, name="A")
    # 创建名为 lev_b 的索引对象，包含元素 ["one", "three", "one", "two"]，使用 df_cat["B"] 的数据类型，并命名为 "B"
    lev_b = Index(["one", "three", "one", "two"], dtype=df_cat["B"].dtype, name="B")
    # 使用 lev_a 和 lev_b 创建多级索引对象 index
    index = MultiIndex.from_arrays([lev_a, lev_b])
    # 创建一个预期的 Series 对象，数据为 [2, 4, 1, 3]，索引为 index，名称为 "C"，并按索引排序
    expected = Series(data=[2, 4, 1, 3], index=index, name="C").sort_index()

    # 使用 df_cat 按 ["A", "B"] 列进行分组，观察所有可能的组合，选取 "C" 列作为分组后的对象 grouped
    grouped = df_cat.groupby(["A", "B"], observed=True)["C"]
    # 对 grouped 对象执行指定的操作（例如 sum、mean 等），并将结果保存到 result 中
    result = getattr(grouped, operation)(sum)
    # 使用测试模块 tm 来比较 result 和预期的结果 expected 是否相等
    tm.assert_series_equal(result, expected)
# 使用 pytest 的标记参数化测试函数，测试 operation 参数为 "agg" 或 "apply" 时的不同情况
@pytest.mark.parametrize("operation", ["agg", "apply"])
# 使用 pytest 的标记参数化测试函数，测试 observed 参数为 False 或 None 时的不同情况
@pytest.mark.parametrize("observed", [False, None])
def test_seriesgroupby_observed_false_or_none(df_cat, observed, operation):
    # GH 24880
    # GH#49223 - order of results was wrong when grouping by index levels
    # 创建多级索引对象 index，其中 A 和 B 是分类索引，不排序
    index, _ = MultiIndex.from_product(
        [
            CategoricalIndex(["bar", "foo"], ordered=False),
            CategoricalIndex(["one", "three", "two"], ordered=False),
        ],
        names=["A", "B"],
    ).sortlevel()

    # 创建预期的 Series 对象 expected，数据为 [2, 4, 0, 1, 0, 3]，使用创建的 index，名称为 "C"
    expected = Series(data=[2, 4, 0, 1, 0, 3], index=index, name="C")
    
    # 对 df_cat 按 ["A", "B"] 分组，根据 observed 参数获取相应的 Series 对象 grouped
    grouped = df_cat.groupby(["A", "B"], observed=observed)["C"]
    
    # 对 grouped 对象应用 operation 操作（sum），得到结果 result
    result = getattr(grouped, operation)(sum)
    
    # 使用 pytest 的断言方法，验证 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的参数化标记，测试 observed 参数为 True、False、None 时的不同情况
@pytest.mark.parametrize(
    "observed, index, data",
    [
        (
            True,
            # 创建多级索引对象 index，其中 A 和 B 是类别索引，名称为 "A" 和 "B"
            MultiIndex.from_arrays(
                [
                    Index(["bar"] * 4 + ["foo"] * 4, dtype="category", name="A"),
                    Index(
                        ["one", "one", "three", "three", "one", "one", "two", "two"],
                        dtype="category",
                        name="B",
                    ),
                    Index(["min", "max"] * 4),
                ]
            ),
            # 数据为 [2, 2, 4, 4, 1, 1, 3, 3]
            [2, 2, 4, 4, 1, 1, 3, 3],
        ),
        (
            False,
            # 创建多级索引对象 index，其中 A 和 B 是不排序的类别索引，名称为 "A" 和 "B"，无名称为 None
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            # 数据为 [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3]
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
        (
            None,
            # 创建多级索引对象 index，其中 A 和 B 是不排序的类别索引，名称为 "A" 和 "B"，无名称为 None
            MultiIndex.from_product(
                [
                    CategoricalIndex(["bar", "foo"], ordered=False),
                    CategoricalIndex(["one", "three", "two"], ordered=False),
                    Index(["min", "max"]),
                ],
                names=["A", "B", None],
            ),
            # 数据为 [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3]
            [2, 2, 4, 4, np.nan, np.nan, 1, 1, np.nan, np.nan, 3, 3],
        ),
    ],
)
def test_seriesgroupby_observed_apply_dict(df_cat, observed, index, data):
    # GH 24880
    # 创建预期的 Series 对象 expected，数据为 data，使用 index，名称为 "C"
    expected = Series(data=data, index=index, name="C")
    
    # 对 df_cat 按 ["A", "B"] 分组，根据 observed 参数应用 lambda 函数生成字典的 apply 操作，得到结果 result
    result = df_cat.groupby(["A", "B"], observed=observed)["C"].apply(
        lambda x: {"min": x.min(), "max": x.max()}
    )
    
    # 使用 pytest 的断言方法，验证 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_series_dataframe_consistent(df_cat):
    # GH 20416
    # 创建预期的 Series 对象 expected，对 df_cat 按 ["A", "B"] 分组，observed=False，计算 "C" 列的均值
    expected = df_cat.groupby(["A", "B"], observed=False)["C"].mean()
    
    # 对 df_cat 按 ["A", "B"] 分组，observed=False，计算整个 DataFrame 的均值，再取 "C" 列
    result = df_cat.groupby(["A", "B"], observed=False).mean()["C"]
    
    # 使用 pytest 的断言方法，验证 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


def test_groupby_cat_preserves_structure(observed, ordered):
    # GH 28787
    # 创建 DataFrame 对象 df，包含 "Name" 列和 "Item" 列，"Name" 列为有序或无序的分类变量
    df = DataFrame(
        {"Name": Categorical(["Bob", "Greg"], ordered=ordered), "Item": [1, 2]},
        columns=["Name", "Item"],
    )
    
    # 创建预期的 DataFrame 对象 expected，与 df 相同
    expected = df.copy()
    # 使用 Pandas DataFrame 对象 df 按照 "Name" 列进行分组，并进行聚合操作，观察现有的值
    result = (
        df.groupby("Name", observed=observed)
        .agg(DataFrame.sum, skipna=True)  # 对每个分组进行求和操作，跳过 NaN 值
        .reset_index()  # 重置索引，将分组后的结果重新索引化
    )
    
    # 使用 pytest 中的 assert_frame_equal 函数比较 result 和期望的结果 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_get_nonexistent_category():
    # 创建一个包含变量 'var' 和 'val' 的 DataFrame 对象
    df = DataFrame({"var": ["a", "a", "b", "b"], "val": range(4)})
    # 使用 pytest 来检测是否会抛出 KeyError，匹配错误消息 "'vau'"
    with pytest.raises(KeyError, match="'vau'"):
        # 对 DataFrame 进行分组，并应用 lambda 函数
        df.groupby("var").apply(
            lambda rows: DataFrame(
                # 创建一个新的 DataFrame，包含单个变量 'var' 和不存在的 'vau' 列
                {"var": [rows.iloc[-1]["var"]], "val": [rows.iloc[-1]["vau"]]}
            )
        )


def test_series_groupby_on_2_categoricals_unobserved(reduction_func, observed):
    # GH 17605
    # 如果 reduction_func 是 "ngroup"，则跳过测试，因为它不是真正的聚合函数
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")

    # 创建一个包含 'cat_1', 'cat_2', 'value' 列的 DataFrame 对象
    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABCD")),
            "cat_2": Categorical(list("AB") * 2, categories=list("ABCD")),
            "value": [0.1] * 4,
        }
    )
    # 获取用于 groupby 方法的参数
    args = get_groupby_method_args(reduction_func, df)

    # 如果 observed 为 False，则预期长度为 16，否则为 4
    expected_length = 4 if observed else 16

    # 对 DataFrame 进行分组，并获取 SeriesGroupBy 对象
    series_groupby = df.groupby(["cat_1", "cat_2"], observed=observed)["value"]

    # 如果 reduction_func 是 "corrwith"，则断言 SeriesGroupBy 对象不具有该方法
    if reduction_func == "corrwith":
        assert not hasattr(series_groupby, reduction_func)
        return

    # 获取 reduction_func 对应的聚合函数
    agg = getattr(series_groupby, reduction_func)

    # 如果 observed 为 False，并且 reduction_func 是 "idxmin" 或 "idxmax"，
    # 则预期抛出 ValueError，匹配错误消息 "empty group due to unobserved categories"
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            agg(*args)
        return

    # 执行聚合函数，并获取结果
    result = agg(*args)

    # 断言结果的长度符合预期长度
    assert len(result) == expected_length


def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(
    reduction_func, request
):
    # GH 17605
    # 检测结果中未观察到的类别是否包含 0 或 NaN

    # 如果 reduction_func 是 "ngroup"，则跳过测试，因为它不是真正的聚合函数
    if reduction_func == "ngroup":
        pytest.skip("ngroup is not truly a reduction")

    # 如果 reduction_func 是 "corrwith"，标记测试为失败，原因是功能尚未实现
    if reduction_func == "corrwith":  # GH 32293
        mark = pytest.mark.xfail(
            reason="TODO: implemented SeriesGroupBy.corrwith. See GH 32293"
        )
        request.applymarker(mark)

    # 创建一个包含 'cat_1', 'cat_2', 'value' 列的 DataFrame 对象
    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("AB") * 2, categories=list("ABC")),
            "value": [0.1] * 4,
        }
    )
    # 创建包含未观察到的类别元组的列表
    unobserved = [tuple("AC"), tuple("BC"), tuple("CA"), tuple("CB"), tuple("CC")]
    # 获取用于 groupby 方法的参数
    args = get_groupby_method_args(reduction_func, df)

    # 对 DataFrame 进行分组，并获取 SeriesGroupBy 对象
    series_groupby = df.groupby(["cat_1", "cat_2"], observed=False)["value"]
    # 获取 reduction_func 对应的聚合函数
    agg = getattr(series_groupby, reduction_func)

    # 如果 reduction_func 是 "idxmin" 或 "idxmax"，则预期抛出 ValueError，匹配错误消息 "empty group due to unobserved categories"
    if reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            agg(*args)
        return

    # 执行聚合函数，并获取结果
    result = agg(*args)

    # 获取 _results_for_groupbys_with_missing_categories 中的填充值
    missing_fillin = _results_for_groupbys_with_missing_categories[reduction_func]
    # 遍历未观察到的索引列表 unobserved
    for idx in unobserved:
        # 获取结果 DataFrame 中索引 idx 处的数值
        val = result.loc[idx]
        # 断言条件：如果 missing_fillin 和 val 都是 NaN，或者它们相等
        assert (pd.isna(missing_fillin) and pd.isna(val)) or (val == missing_fillin)

    # 如果我们期望未观察到的值为零，则也期望数据类型为整数（int）
    # 除非是在 reduction_func 为 "sum" 时，如果观察到的类别求和结果为浮点数（即有小数点），
    # 那么缺失类别的零值也应该是浮点数。
    if missing_fillin == 0:
        if reduction_func in ["count", "nunique", "size"]:
            # 断言条件：结果 result 的数据类型应为整数（int）
            assert np.issubdtype(result.dtype, np.integer)
        else:
            # 断言条件：reduction_func 在 ["sum", "any"] 中
            assert reduction_func in ["sum", "any"]
# 定义一个测试函数，测试在 observed=True 时，对两个分类变量进行分组聚合操作是否忽略不在数据框中的分类
def test_dataframe_groupby_on_2_categoricals_when_observed_is_true(reduction_func):
    # GH 23865
    # GH 27075
    # 如果 reduction_func 是 "ngroup"，则跳过测试，因为 ngroup 不会返回索引上的分类
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")

    # 创建一个数据框 df，包含两个分类变量 cat_1 和 cat_2，以及一个数值列 value
    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("1111"), categories=list("12")),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    
    # 未观察到的分类组合
    unobserved_cats = [("A", "2"), ("B", "2"), ("C", "1"), ("C", "2")]

    # 使用分类变量 cat_1 和 cat_2 对数据框进行分组，observed=True 表示只考虑实际观察到的分类值
    df_grp = df.groupby(["cat_1", "cat_2"], observed=True)

    # 获取分组方法的参数
    args = get_groupby_method_args(reduction_func, df)

    # 如果 reduction_func 是 "corrwith"，则发出 FutureWarning
    if reduction_func == "corrwith":
        warn = FutureWarning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""

    # 使用 pytest 的 assert_produces_warning 上下文来检查是否产生了警告信息
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 调用 reduction_func 对应的方法进行聚合操作
        res = getattr(df_grp, reduction_func)(*args)

    # 断言未观察到的分类组合不在结果的索引中
    for cat in unobserved_cats:
        assert cat not in res.index


# 使用参数化测试，测试在 observed=False 或 observed=None 时，对两个分类变量进行分组聚合操作是否返回未在数据框中的分类
@pytest.mark.parametrize("observed", [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(
    reduction_func, observed
):
    # GH 23865
    # GH 27075
    # 如果 reduction_func 是 "ngroup"，则跳过测试，因为 ngroup 不会返回索引上的分类
    if reduction_func == "ngroup":
        pytest.skip("ngroup does not return the Categories on the index")

    # 创建一个数据框 df，包含两个分类变量 cat_1 和 cat_2，以及一个数值列 value
    df = DataFrame(
        {
            "cat_1": Categorical(list("AABB"), categories=list("ABC")),
            "cat_2": Categorical(list("1111"), categories=list("12")),
            "value": [0.1, 0.1, 0.1, 0.1],
        }
    )
    
    # 未观察到的分类组合
    unobserved_cats = [("A", "2"), ("B", "2"), ("C", "1"), ("C", "2")]

    # 使用分类变量 cat_1 和 cat_2 对数据框进行分组，observed 参数决定是否包括未观察到的分类值
    df_grp = df.groupby(["cat_1", "cat_2"], observed=observed)

    # 获取分组方法的参数
    args = get_groupby_method_args(reduction_func, df)

    # 如果 observed 为 False 且 reduction_func 是 ["idxmin", "idxmax"] 中的一个，则断言会抛出 ValueError
    if not observed and reduction_func in ["idxmin", "idxmax"]:
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(df_grp, reduction_func)(*args)
        return

    # 如果 reduction_func 是 "corrwith"，则发出 FutureWarning
    if reduction_func == "corrwith":
        warn = FutureWarning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""

    # 使用 pytest 的 assert_produces_warning 上下文来检查是否产生了警告信息
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 调用 reduction_func 对应的方法进行聚合操作
        res = getattr(df_grp, reduction_func)(*args)

    # 获取预期的结果字典
    expected = _results_for_groupbys_with_missing_categories[reduction_func]

    # 如果预期结果是 np.nan，则断言结果中未观察到的分类组合均为 NaN
    if expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        # 否则断言结果中未观察到的分类组合与预期结果一致
        assert (res.loc[unobserved_cats] == expected).all().all()


# 定义一个测试函数，测试在使用分类变量进行分组聚合时通过索引获取元素是否正常工作
def test_series_groupby_categorical_aggregation_getitem():
    # GH 8870
    # 创建一个字典 d，包含三个键值对，每个值是一个列表
    d = {"foo": [10, 8, 4, 1], "bar": [10, 20, 30, 40], "baz": ["d", "c", "d", "c"]}
    
    # 使用 pandas 的 DataFrame 构造函数将字典 d 转换为 DataFrame 对象 df
    df = DataFrame(d)
    
    # 对 df 中的 "foo" 列进行分段，分段的依据是将该列的值划分为 0 到 20 之间均匀间隔的四个区间
    cat = pd.cut(df["foo"], np.linspace(0, 20, 5))
    
    # 将分段结果存储在 df 的新列 "range" 中
    df["range"] = cat
    
    # 根据 "range" 和 "baz" 两列进行分组，将数据按照这两列的值进行分组，as_index=True 表示分组的键作为索引，sort=True 表示排序结果，observed=False 表示不使用未观察到的组
    groups = df.groupby(["range", "baz"], as_index=True, sort=True, observed=False)
    
    # 对分组后的 "foo" 列计算平均值，返回一个 Series 对象 result
    result = groups["foo"].agg("mean")
    
    # 对整个分组后的 DataFrame 计算每组的平均值，然后取出 "foo" 列，返回一个 Series 对象 expected
    expected = groups.agg("mean")["foo"]
    
    # 使用测试框架中的函数 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "func, expected_values",
    [(Series.nunique, [1, 1, 2]), (Series.count, [1, 2, 2])],
)
# 定义参数化测试函数，分别测试 Series.nunique 和 Series.count 函数
def test_groupby_agg_categorical_columns(func, expected_values):
    # 创建一个 DataFrame 对象 df，包含三列：id, groups, value
    # 其中 value 列使用 Categorical 类型存储
    df = DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "groups": [0, 1, 1, 2, 2],
            "value": Categorical([0, 0, 0, 0, 1]),
        }
    ).set_index("id")
    # 对 df 按 groups 列进行分组，并对分组后的结果应用 func 函数
    result = df.groupby("groups").agg(func)

    # 创建一个预期的 DataFrame 对象 expected，包含一列 value，使用 expected_values 初始化
    expected = DataFrame(
        {"value": expected_values}, index=Index([0, 1, 2], name="groups")
    )
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试处理非数值类型的数据
def test_groupby_agg_non_numeric():
    # 创建一个 DataFrame 对象 df，包含一列 A，使用 Categorical 类型存储
    df = DataFrame({"A": Categorical(["a", "a", "b"], categories=["a", "b", "c"])})
    # 创建预期的 DataFrame 对象 expected，包含一列 A，使用 [2, 1] 初始化
    expected = DataFrame({"A": [2, 1]}, index=np.array([1, 2]))

    # 对 df 按 [1, 2, 1] 列进行分组，并对分组后的结果应用 Series.nunique 函数
    result = df.groupby([1, 2, 1]).agg(Series.nunique)
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected
    tm.assert_frame_equal(result, expected)

    # 对 df 按 [1, 2, 1] 列进行分组，并直接调用 nunique() 方法
    result = df.groupby([1, 2, 1]).nunique()
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
# 定义参数化测试函数，测试 groupby 对象在应用 first 和 last 函数时返回分类类型而不是 DataFrame 的问题
def test_groupby_first_returned_categorical_instead_of_dataframe(func):
    # 创建一个 DataFrame 对象 df，包含两列 A 和 B
    df = DataFrame({"A": [1997], "B": Series(["b"], dtype="category").cat.as_ordered()})
    # 对 df 按 A 列进行分组，并选择 B 列
    df_grouped = df.groupby("A")["B"]
    # 对分组后的结果应用 func 函数
    result = getattr(df_grouped, func)()

    # 创建一个预期的 Series 对象 expected，其类型应保持为有序分类类型
    expected = Series(
        ["b"], index=Index([1997], name="A"), name="B", dtype=df["B"].dtype
    )
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected
    tm.assert_series_equal(result, expected)


# 定义测试函数，测试读取只读分类数据时不排序的问题
def test_read_only_category_no_sort():
    # 创建一个只读的分类数组 cats
    cats = np.array([1, 2])
    cats.flags.writeable = False
    # 创建一个 DataFrame 对象 df，包含两列 a 和 b
    df = DataFrame(
        {"a": [1, 3, 5, 7], "b": Categorical([1, 1, 2, 2], categories=Index(cats))}
    )
    # 创建一个预期的 DataFrame 对象 expected，包含一列 a，使用 [2.0, 6.0] 初始化
    expected = DataFrame(data={"a": [2.0, 6.0]}, index=CategoricalIndex(cats, name="b"))
    # 对 df 按 b 列进行分组，不排序，不考虑观测值，然后计算均值
    result = df.groupby("b", sort=False, observed=False).mean()
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试处理排序的缺失分类值的问题
def test_sorted_missing_category_values():
    # 创建一个 DataFrame 对象 df，包含两列 foo 和 bar
    df = DataFrame(
        {
            "foo": [
                "small",
                "large",
                "large",
                "large",
                "medium",
                "large",
                "large",
                "medium",
            ],
            "bar": ["C", "A", "A", "C", "A", "C", "A", "C"],
        }
    )
    # 将 foo 列转换为有序分类类型，指定分类值顺序
    df["foo"] = (
        df["foo"]
        .astype("category")
        .cat.set_categories(["tiny", "small", "medium", "large"], ordered=True)
    )

    # 创建一个预期的 DataFrame 对象 expected，包含四列 tiny, small, medium, large
    expected = DataFrame(
        {
            "tiny": {"A": 0, "C": 0},
            "small": {"A": 0, "C": 1},
            "medium": {"A": 1, "C": 1},
            "large": {"A": 3, "C": 2},
        }
    )
    # 将行索引命名为 bar，以便与预期的索引进行比较
    expected = expected.rename_axis("bar", axis="index")
    # 设置 expected 对象的列索引为 CategoricalIndex 类型
    expected.columns = CategoricalIndex(
        # 指定索引的名称为 "foo"
        ["tiny", "small", "medium", "large"],  # 列名列表
        categories=["tiny", "small", "medium", "large"],  # 列的分类列表
        ordered=True,  # 列是否有序
        name="foo",  # 列索引的名称
        dtype="category",  # 列的数据类型为分类数据
    )
    
    # 对 DataFrame df 进行按照 "bar" 和 "foo" 列分组，不考虑未观察到的组合
    result = df.groupby(["bar", "foo"], observed=False).size().unstack()
    
    # 使用测试框架中的函数验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_agg_cython_category_not_implemented_fallback():
    # GitHub issue reference: https://github.com/pandas-dev/pandas/issues/31450
    # 创建包含数值列 'col_num' 的数据框
    df = DataFrame({"col_num": [1, 1, 2, 3]})
    # 将 'col_num' 列转换为分类数据类型并添加到数据框
    df["col_cat"] = df["col_num"].astype("category")

    # 按 'col_num' 列分组，并取每组的 'col_cat' 列的第一个值
    result = df.groupby("col_num").col_cat.first()

    # 创建预期结果，包含数值列 [1, 2, 3]，索引为 'col_num'，列名为 'col_cat'，数据类型与 df["col_cat"] 一致
    expected = Series(
        [1, 2, 3],
        index=Index([1, 2, 3], name="col_num"),
        name="col_cat",
        dtype=df["col_cat"].dtype,
    )
    # 断言结果与预期结果相等
    tm.assert_series_equal(result, expected)

    # 按 'col_num' 列分组，并对 'col_cat' 列应用 'first' 聚合函数
    result = df.groupby("col_num").agg({"col_cat": "first"})
    # 将预期结果转换为数据框格式
    expected = expected.to_frame()
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)


def test_aggregate_categorical_with_isnan():
    # GitHub issue reference: GH 29837
    # 创建包含数值、对象和分类列的数据框
    df = DataFrame(
        {
            "A": [1, 1, 1, 1],
            "B": [1, 2, 1, 2],
            "numerical_col": [0.1, 0.2, np.nan, 0.3],
            "object_col": ["foo", "bar", "foo", "fee"],
            "categorical_col": ["foo", "bar", "foo", "fee"],
        }
    )

    # 将 'categorical_col' 列转换为分类数据类型
    df = df.astype({"categorical_col": "category"})

    # 按 ['A', 'B'] 列分组，并计算每组中各列中 NaN 值的数量
    result = df.groupby(["A", "B"]).agg(lambda df: df.isna().sum())
    # 创建预期结果的索引为 MultiIndex，包含 ('A', 'B') 列的值
    index = MultiIndex.from_arrays([[1, 1], [1, 2]], names=("A", "B"))
    # 创建预期结果的数据框，包含 ['numerical_col', 'object_col', 'categorical_col'] 列
    expected = DataFrame(
        data={
            "numerical_col": [1, 0],
            "object_col": [0, 0],
            "categorical_col": [0, 0],
        },
        index=index,
    )
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)


def test_categorical_transform():
    # GitHub issue reference: GH 29037
    # 创建包含 'package_id' 和 'status' 列的数据框
    df = DataFrame(
        {
            "package_id": [1, 1, 1, 2, 2, 3],
            "status": [
                "Waiting",
                "OnTheWay",
                "Delivered",
                "Waiting",
                "OnTheWay",
                "Waiting",
            ],
        }
    )

    # 定义状态列的有序分类数据类型
    delivery_status_type = pd.CategoricalDtype(
        categories=["Waiting", "OnTheWay", "Delivered"], ordered=True
    )
    # 将 'status' 列转换为定义的分类数据类型
    df["status"] = df["status"].astype(delivery_status_type)
    # 按 'package_id' 列分组，并对 'status' 列应用 'max' 聚合函数
    df["last_status"] = df.groupby("package_id")["status"].transform(max)
    # 将结果复制给变量 result
    result = df.copy()

    # 创建预期结果的数据框，包含 'package_id', 'status', 'last_status' 列
    expected = DataFrame(
        {
            "package_id": [1, 1, 1, 2, 2, 3],
            "status": [
                "Waiting",
                "OnTheWay",
                "Delivered",
                "Waiting",
                "OnTheWay",
                "Waiting",
            ],
            "last_status": [
                "Waiting",
                "Waiting",
                "Waiting",
                "Waiting",
                "Waiting",
                "Waiting",
            ],
        }
    )

    # 将 'status' 列转换为定义的分类数据类型
    expected["status"] = expected["status"].astype(delivery_status_type)
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["first", "last"])
def test_series_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
):
    # GitHub issue reference: GH 34951
    # 创建包含 [0, 0, 1, 1] 的分类数据列
    cat = Categorical([0, 0, 1, 1])
    # 创建一个包含四个整数的列表
    val = [0, 1, 1, 0]
    # 使用 DataFrame 构造函数创建一个数据框，包含列 "a" 和 "b"，以及列 "c"，其值为之前定义的列表 val
    df = DataFrame({"a": cat, "b": cat, "c": val})
    
    # 创建一个分类变量 cat2，包含值 [0, 1]
    cat2 = Categorical([0, 1])
    # 使用 MultiIndex 的 from_product 方法创建一个多级索引 idx，索引由 cat2 和 cat2 的笛卡尔积构成，命名为 "a" 和 "b"
    idx = MultiIndex.from_product([cat2, cat2], names=["a", "b"])
    
    # 创建一个预期的字典 expected_dict，包含两个键值对，分别是 "first" 和 "last"
    # 每个值是一个 Series 对象，分别包含四个元素的数组 [0, np.nan, np.nan, 1] 和 [1, np.nan, np.nan, 0]
    # 这些数组使用之前创建的多级索引 idx，每个 Series 的名称为 "c"
    expected_dict = {
        "first": Series([0, np.nan, np.nan, 1], idx, name="c"),
        "last": Series([1, np.nan, np.nan, 0], idx, name="c"),
    }
    
    # 根据函数名 func 选择预期结果 expected
    expected = expected_dict[func]
    # 如果 observed 为真，则从预期结果中删除 NaN 值并将其转换为 np.int64 类型
    if observed:
        expected = expected.dropna().astype(np.int64)
    
    # 使用数据框 df 按列 "a" 和 "b" 进行分组，生成一个 SeriesGroupBy 对象 srs_grp
    # 对象的聚合操作由参数 func 指定
    srs_grp = df.groupby(["a", "b"], observed=observed)["c"]
    # 调用 SeriesGroupBy 对象的 func 方法，返回聚合结果 result
    result = getattr(srs_grp, func)()
    # 使用测试工具 tm.assert_series_equal 检查聚合结果 result 是否与预期结果 expected 相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("func", ["first", "last"])
# 使用 pytest 的参数化标记，测试函数将会多次运行，每次使用不同的 func 参数
def test_df_groupby_first_on_categorical_col_grouped_on_2_categoricals(
    func: str, observed: bool
):
    # GH 34951
    # 创建一个包含分类数据的 Categorical 对象
    cat = Categorical([0, 0, 1, 1])
    # 创建一个数值列表
    val = [0, 1, 1, 0]
    # 创建 DataFrame，包含列 'a' 和 'b' 的分类数据以及列 'c' 的数值
    df = DataFrame({"a": cat, "b": cat, "c": val})

    # 创建第二个分类对象
    cat2 = Categorical([0, 1])
    # 创建多级索引，使用两个分类的笛卡尔积，并指定索引的名称
    idx = MultiIndex.from_product([cat2, cat2], names=["a", "b"])
    # 创建预期的结果字典，包含 "first" 和 "last" 函数的预期输出 Series 对象
    expected_dict = {
        "first": Series([0, np.nan, np.nan, 1], idx, name="c"),
        "last": Series([1, np.nan, np.nan, 0], idx, name="c"),
    }

    # 从预期字典中选择特定函数的预期输出，并将其转换为 DataFrame 对象
    expected = expected_dict[func].to_frame()
    # 如果 observed 为 True，则对预期结果进行处理：删除 NaN 值并转换为 np.int64 类型
    if observed:
        expected = expected.dropna().astype(np.int64)

    # 根据列 'a' 和 'b' 对 DataFrame 进行分组，使用 observed 参数指定是否使用观察到的分类
    df_grp = df.groupby(["a", "b"], observed=observed)
    # 调用指定的聚合函数（根据 func 参数选择 "first" 或 "last"）
    result = getattr(df_grp, func)()
    # 断言聚合后的结果与预期结果相等
    tm.assert_frame_equal(result, expected)


def test_groupby_categorical_indices_unused_categories():
    # GH#38642
    # 创建包含分类数据和数值列 'col' 的 DataFrame
    df = DataFrame(
        {
            "key": Categorical(["b", "b", "a"], categories=["a", "b", "c"]),
            "col": range(3),
        }
    )
    # 根据 'key' 列对 DataFrame 进行分组，指定不排序且不使用观察到的分类
    grouped = df.groupby("key", sort=False, observed=False)
    # 获取分组后的索引字典，键为分组的类别值，值为相应的索引数组
    result = grouped.indices
    # 预期的索引字典，包含每个类别值对应的空间分配数组
    expected = {
        "b": np.array([0, 1], dtype="intp"),
        "a": np.array([2], dtype="intp"),
        "c": np.array([], dtype="intp"),
    }
    # 断言结果的键集合与预期的键集合相等，并逐一比较每个键对应的索引数组
    assert result.keys() == expected.keys()
    for key in result.keys():
        tm.assert_numpy_array_equal(result[key], expected[key])


@pytest.mark.parametrize("func", ["first", "last"])
# 使用 pytest 的参数化标记，测试函数将会多次运行，每次使用不同的 func 参数
def test_groupby_last_first_preserve_categoricaldtype(func):
    # GH#33090
    # 创建包含列 'a' 的 DataFrame
    df = DataFrame({"a": [1, 2, 3]})
    # 将列 'a' 转换为分类类型，并将其赋给列 'b'
    df["b"] = df["a"].astype("category")
    # 根据列 'a' 进行分组，并调用指定的聚合函数（根据 func 参数选择 "first" 或 "last"）
    result = getattr(df.groupby("a")["b"], func)()
    # 创建预期的 Series 对象，包含分类数据
    expected = Series(
        Categorical([1, 2, 3]), name="b", index=Index([1, 2, 3], name="a")
    )
    # 断言聚合后的结果与预期结果相等
    tm.assert_series_equal(expected, result)


def test_groupby_categorical_observed_nunique():
    # GH#45128
    # 创建包含列 'a', 'b', 'c' 的 DataFrame
    df = DataFrame({"a": [1, 2], "b": [1, 2], "c": [10, 11]})
    # 将列 'a' 和 'b' 转换为分类类型
    df = df.astype(dtype={"a": "category", "b": "category"})
    # 根据列 'a' 和 'b' 进行分组，计算每个组合下 'c' 列的唯一值数量
    result = df.groupby(["a", "b"], observed=True).nunique()["c"]
    # 创建预期的 Series 对象，包含唯一值数量
    expected = Series(
        [1, 1],
        index=MultiIndex.from_arrays(
            [CategoricalIndex([1, 2], name="a"), CategoricalIndex([1, 2], name="b")]
        ),
        name="c",
    )
    # 断言聚合后的结果与预期结果相等
    tm.assert_series_equal(result, expected)


def test_groupby_categorical_aggregate_functions():
    # GH#37275
    # 创建包含列 'grp' 和 'description' 的 DataFrame，其中 'description' 列为分类类型
    dtype = pd.CategoricalDtype(categories=["small", "big"], ordered=True)
    df = DataFrame(
        [[1, "small"], [1, "big"], [2, "small"]], columns=["grp", "description"]
    ).astype({"description": dtype})

    # 根据 'grp' 列进行分组，并对 'description' 列应用 max 函数
    result = df.groupby("grp")["description"].max()
    # 创建预期的 Series 对象，包含应用 max 函数后的结果
    expected = Series(
        ["big", "small"],
        index=Index([1, 2], name="grp"),
        name="description",
        dtype=pd.CategoricalDtype(categories=["small", "big"], ordered=True),
    )

    # 断言聚合后的结果与预期结果相等
    tm.assert_series_equal(result, expected)
    # GH#48645 - dropna should have no impact on the result when there are no NA values
    
    # 创建一个包含值为 [1, 2] 的分类变量 cat，其类别为 [1, 2, 3]
    cat = Categorical([1, 2], categories=[1, 2, 3])
    
    # 创建一个数据框 df，包含列 "x" 和 "y"，其中 "x" 是一个分类变量，"y" 是 [3, 4]
    df = DataFrame({"x": Categorical([1, 2], categories=[1, 2, 3]), "y": [3, 4]})
    
    # 根据 "x" 列进行分组，observed 参数用于指定是否使用观察到的类别，dropna 参数用于指定是否删除缺失值
    gb = df.groupby("x", observed=observed, dropna=dropna)
    
    # 对分组后的结果进行求和
    result = gb.sum()
    
    # 如果 observed 参数为 True，则创建一个期望的数据框 expected，索引为 cat，包含列 "y" 和值 [3, 4]
    if observed:
        expected = DataFrame({"y": [3, 4]}, index=cat)
    # 如果 observed 参数为 False，则创建一个期望的数据框 expected，索引为 [1, 2, 3]，包含列 "y" 和值 [3, 4, 0]
    else:
        index = CategoricalIndex([1, 2, 3], [1, 2, 3])
        expected = DataFrame({"y": [3, 4, 0]}, index=index)
    
    # 设置期望数据框 expected 的索引名称为 "x"
    expected.index.name = "x"
    
    # 使用测试模块 tm 进行结果 result 和期望 expected 的比较，确保它们相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_reducer(
    request, as_index, sort, observed, reduction_func, index_kind, ordered
):
    # GH#48749
    # 如果 reduction_func 是 "corrwith" 并且 as_index 是 False 并且 index_kind 不是 "single"，则标记为预期失败
    if reduction_func == "corrwith" and not as_index and index_kind != "single":
        msg = "GH#49950 - corrwith with as_index=False may not have grouping column"
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif index_kind != "range" and not as_index:
        # 如果 index_kind 不是 "range" 并且 as_index 是 False，则跳过测试，因为结果没有分类，无需测试
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    
    # 创建一个 DataFrame 对象
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        # 将 DataFrame 设置为以 "a" 列为索引
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        # 添加 "a2" 列作为多级索引的一部分，并将 DataFrame 设置为以 "a", "a2" 列为索引
        df["a2"] = df["a"]
        df = df.set_index(keys)
    
    # 获取 GroupBy 操作的参数
    args = get_groupby_method_args(reduction_func, df)
    # 对 DataFrame 进行分组操作，返回 GroupBy 对象
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)

    if not observed and reduction_func in ["idxmin", "idxmax"]:
        # 对于未观察到的分类，idxmin 和 idxmax 会因空输入而失败
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            getattr(gb, reduction_func)(*args)
        return
    
    if reduction_func == "corrwith":
        # 如果是 corrwith 操作，产生未来警告
        warn = FutureWarning
        warn_msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        warn_msg = ""
    
    # 断言产生警告信息
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 执行 GroupBy 对象上的指定函数操作
        op_result = getattr(gb, reduction_func)(*args)
    
    if as_index:
        # 如果 as_index 是 True，则从操作结果中获取 "a" 列的分类
        result = op_result.index.get_level_values("a").categories
    else:
        # 否则，从操作结果的 "a" 列中获取分类
        result = op_result["a"].cat.categories
    
    # 预期的结果
    expected = Index([1, 4, 3, 2])
    # 断言结果与预期相等
    tm.assert_index_equal(result, expected)

    if index_kind == "multi":
        # 如果 index_kind 是 "multi"，则还需验证 "a2" 列的分类
        result = op_result.index.get_level_values("a2").categories
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("index_kind", ["single", "multi"])
@pytest.mark.parametrize("ordered", [True, False])
def test_category_order_transformer(
    as_index, sort, observed, transformation_func, index_kind, ordered
):
    # GH#48749
    # 创建一个 DataFrame 对象
    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    
    if index_kind == "single":
        keys = ["a"]
        # 将 DataFrame 设置为以 "a" 列为索引
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        # 添加 "a2" 列作为多级索引的一部分，并将 DataFrame 设置为以 "a", "a2" 列为索引
        df["a2"] = df["a"]
        df = df.set_index(keys)
    
    # 获取 GroupBy 操作的参数
    args = get_groupby_method_args(transformation_func, df)
    # 对 DataFrame 进行分组操作，返回 GroupBy 对象
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    
    if transformation_func == "fillna":
        # 如果是 fillna 操作，产生未来警告
        warn = FutureWarning
        msg = "DataFrameGroupBy.fillna is deprecated"
    else:
        warn = None
        msg = ""
    # 使用 `assert_produces_warning` 上下文管理器来检查特定的警告消息是否被产生
    with tm.assert_produces_warning(warn, match=msg):
        # 调用 `getattr` 函数从 `gb` 对象中获取特定名称的方法，并执行
        op_result = getattr(gb, transformation_func)(*args)
    
    # 从 `op_result` 中获取索引标签为 "a" 的级别的分类信息，并赋值给 `result`
    result = op_result.index.get_level_values("a").categories
    
    # 创建一个预期的索引对象，包含指定的整数列表 [1, 4, 3, 2]
    expected = Index([1, 4, 3, 2])
    
    # 使用 `assert_index_equal` 函数比较 `result` 和 `expected`，确保它们相等
    tm.assert_index_equal(result, expected)
    
    # 如果索引类型是 "multi"
    if index_kind == "multi":
        # 从 `op_result` 中获取索引标签为 "a2" 的级别的分类信息，并赋值给 `result`
        result = op_result.index.get_level_values("a2").categories
        
        # 使用 `assert_index_equal` 函数比较 `result` 和 `expected`，确保它们相等
        tm.assert_index_equal(result, expected)
@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
# 参数化测试：index_kind 可以是 "range", "single", "multi" 中的一个
@pytest.mark.parametrize("method", ["head", "tail"])
# 参数化测试：method 可以是 "head" 或者 "tail"
@pytest.mark.parametrize("ordered", [True, False])
# 参数化测试：ordered 可以是 True 或者 False

def test_category_order_head_tail(
    as_index, sort, observed, method, index_kind, ordered
):
    # GH#48749
    # 用例参考编号 GH#48749

    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    # 创建一个 DataFrame，包含两列："a" 是有序分类数据，"b" 是整数范围 [0, 1, 2, 3]

    if index_kind == "range":
        keys = ["a"]
        # 如果 index_kind 是 "range"，则 keys 为 ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
        # 如果 index_kind 是 "single"，则将 DataFrame 按 "a" 列设为索引
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
        # 如果 index_kind 是 "multi"，则将 DataFrame 按 ["a", "a2"] 列设为多级索引

    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    # 根据 keys 列对 DataFrame 进行分组，as_index, sort, observed 参数由测试参数决定

    op_result = getattr(gb, method)()
    # 调用 groupby 对象 gb 的 method 方法（例如 head 或 tail），获取操作结果

    if index_kind == "range":
        result = op_result["a"].cat.categories
        # 如果 index_kind 是 "range"，则获取操作结果中 "a" 列的分类类别
    else:
        result = op_result.index.get_level_values("a").categories
        # 否则获取操作结果的索引中级别 "a" 的分类类别

    expected = Index([1, 4, 3, 2])
    # 预期的结果是包含元素 [1, 4, 3, 2] 的 Index 对象

    tm.assert_index_equal(result, expected)
    # 使用 pytest 的 assert_index_equal 方法比较 result 和 expected

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        # 如果 index_kind 是 "multi"，则获取操作结果的索引中级别 "a2" 的分类类别
        tm.assert_index_equal(result, expected)
        # 使用 pytest 的 assert_index_equal 方法比较 result 和 expected


@pytest.mark.parametrize("index_kind", ["range", "single", "multi"])
# 参数化测试：index_kind 可以是 "range", "single", "multi" 中的一个
@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
# 参数化测试：method 可以是 "apply", "agg", "transform" 中的一个
@pytest.mark.parametrize("ordered", [True, False])
# 参数化测试：ordered 可以是 True 或者 False

def test_category_order_apply(as_index, sort, observed, method, index_kind, ordered):
    # GH#48749
    # 用例参考编号 GH#48749

    if (method == "transform" and index_kind == "range") or (
        not as_index and index_kind != "range"
    ):
        pytest.skip("No categories in result, nothing to test")
        # 如果是 "transform" 方法并且 index_kind 是 "range"，或者不使用 as_index 且 index_kind 不是 "range"，则跳过测试

    df = DataFrame(
        {
            "a": Categorical([2, 1, 2, 3], categories=[1, 4, 3, 2], ordered=ordered),
            "b": range(4),
        }
    )
    # 创建一个 DataFrame，包含两列："a" 是有序分类数据，"b" 是整数范围 [0, 1, 2, 3]

    if index_kind == "range":
        keys = ["a"]
        # 如果 index_kind 是 "range"，则 keys 为 ["a"]
    elif index_kind == "single":
        keys = ["a"]
        df = df.set_index(keys)
        # 如果 index_kind 是 "single"，则将 DataFrame 按 "a" 列设为索引
    elif index_kind == "multi":
        keys = ["a", "a2"]
        df["a2"] = df["a"]
        df = df.set_index(keys)
        # 如果 index_kind 是 "multi"，则将 DataFrame 按 ["a", "a2"] 列设为多级索引

    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=observed)
    # 根据 keys 列对 DataFrame 进行分组，as_index, sort, observed 参数由测试参数决定

    warn = DeprecationWarning if method == "apply" and index_kind == "range" else None
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(warn, match=msg):
        op_result = getattr(gb, method)(lambda x: x.sum(numeric_only=True))
        # 使用 getattr 调用 groupby 对象 gb 的 method 方法（例如 apply, agg, transform），对每组数据应用 lambda 函数

    if (method == "transform" or not as_index) and index_kind == "range":
        result = op_result["a"].cat.categories
        # 如果是 "transform" 方法或者不使用 as_index 且 index_kind 是 "range"，则获取操作结果中 "a" 列的分类类别
    else:
        result = op_result.index.get_level_values("a").categories
        # 否则获取操作结果的索引中级别 "a" 的分类类别

    expected = Index([1, 4, 3, 2])
    # 预期的结果是包含元素 [1, 4, 3, 2] 的 Index 对象

    tm.assert_index_equal(result, expected)
    # 使用 pytest 的 assert_index_equal 方法比较 result 和 expected

    if index_kind == "multi":
        result = op_result.index.get_level_values("a2").categories
        # 如果 index_kind 是 "multi"，则获取操作结果的索引中级别 "a2" 的分类类别
        tm.assert_index_equal(result, expected)
        # 使用 pytest 的 assert_index_equal 方法比较 result 和 expected
    # GH#48749 - Test when the grouper has many categories
    # 如果索引类型不是"range"且不是作为索引，跳过测试，因为结果没有类别，无需测试
    if index_kind != "range" and not as_index:
        pytest.skip(reason="Result doesn't have categories, nothing to test")
    
    # 创建一个包含9999到0的类别数组
    categories = np.arange(9999, -1, -1)
    
    # 使用指定的类别数组创建一个分类数据对象
    grouper = Categorical([2, 1, 2, 3], categories=categories, ordered=ordered)
    
    # 创建一个DataFrame对象，包含两列：一列使用grouper作为数据，一列是简单的整数范围
    df = DataFrame({"a": grouper, "b": range(4)})
    
    # 根据索引类型选择合适的键
    if index_kind == "range":
        keys = ["a"]
    elif index_kind == "single":
        keys = ["a"]
        # 将DataFrame设置为以选定的键作为索引
        df = df.set_index(keys)
    elif index_kind == "multi":
        keys = ["a", "a2"]
        # 创建一个新的列"a2"，其内容与列"a"相同
        df["a2"] = df["a"]
        # 将DataFrame设置为以多重索引键作为索引
        df = df.set_index(keys)
    
    # 根据指定的键对DataFrame进行分组操作，返回一个GroupBy对象
    gb = df.groupby(keys, as_index=as_index, sort=sort, observed=True)
    
    # 对分组后的结果进行求和操作，返回一个DataFrame对象
    result = gb.sum()

    # 设置预期的数据列表，如果sort为True，则数据为[3, 2, 1]，否则为[2, 1, 3]
    data = [3, 2, 1] if sort else [2, 1, 3]
    
    # 创建一个CategoricalIndex对象，使用指定的数据、类别和排序信息
    index = CategoricalIndex(
        data, categories=grouper.categories, ordered=ordered, name="a"
    )
    
    # 根据as_index的值选择预期的DataFrame对象
    if as_index:
        expected = DataFrame({"b": data})
        if index_kind == "multi":
            # 如果索引类型是多重索引，将预期的索引设置为由两列DataFrame构建的MultiIndex
            expected.index = MultiIndex.from_frame(DataFrame({"a": index, "a2": index}))
        else:
            # 否则，直接设置预期的索引为单一索引对象
            expected.index = index
    elif index_kind == "multi":
        # 如果索引类型是多重索引，预期的DataFrame包含"a"、"a2"和"b"三列数据
        expected = DataFrame({"a": Series(index), "a2": Series(index), "b": data})
    else:
        # 否则，预期的DataFrame包含"a"和"b"两列数据
        expected = DataFrame({"a": Series(index), "b": data})
    
    # 使用测试工具tm.assert_frame_equal检查结果DataFrame与预期DataFrame是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("test_series", [True, False])
# 使用 pytest.mark.parametrize 装饰器为 test_agg_list 函数参数 test_series 创建参数化测试，分别测试 True 和 False 两种情况
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
# 使用 pytest.mark.parametrize 装饰器为 test_agg_list 函数参数 keys 创建参数化测试，分别测试包含 ["a1"] 和 ["a1", "a2"] 两种情况
def test_agg_list(request, as_index, observed, reduction_func, test_series, keys):
    # GH#52760
    # 如果 test_series 为 True 并且 reduction_func 为 "corrwith"
    if test_series and reduction_func == "corrwith":
        # 断言 SeriesGroupBy 没有属性 "corrwith"
        assert not hasattr(SeriesGroupBy, "corrwith")
        # 如果条件满足，则跳过测试，并给出相应提示
        pytest.skip("corrwith not implemented for SeriesGroupBy")
    # 如果 reduction_func 为 "corrwith"
    elif reduction_func == "corrwith":
        # 提示信息
        msg = "GH#32293: attempts to call SeriesGroupBy.corrwith"
        # 将当前测试标记为预期失败，并提供失败原因
        request.applymarker(pytest.mark.xfail(reason=msg))

    # 创建包含示例数据的 DataFrame
    df = DataFrame({"a1": [0, 0, 1], "a2": [2, 3, 3], "b": [4, 5, 6]})
    # 将 DataFrame 列 a1 和 a2 转换为分类数据类型
    df = df.astype({"a1": "category", "a2": "category"})
    # 如果 keys 列表中不包含 "a2" 列
    if "a2" not in keys:
        # 则删除 DataFrame 中的 "a2" 列
        df = df.drop(columns="a2")
    # 根据 keys 参数对 DataFrame 进行分组，创建 GroupBy 对象
    gb = df.groupby(by=keys, as_index=as_index, observed=observed)
    # 如果 test_series 为 True
    if test_series:
        # 只保留 GroupBy 对象的 "b" 列
        gb = gb["b"]
    # 获取用于 groupby 操作的参数
    args = get_groupby_method_args(reduction_func, df)

    # 如果 observed 为 False，且 reduction_func 是 ["idxmin", "idxmax"] 之一，且 keys 为 ["a1", "a2"]
    if not observed and reduction_func in ["idxmin", "idxmax"] and keys == ["a1", "a2"]:
        # 使用 pytest.raises 断言抛出 ValueError 异常，并匹配给定的错误信息
        with pytest.raises(
            ValueError, match="empty group due to unobserved categories"
        ):
            # 执行 gb.agg([reduction_func], *args) 操作，期望抛出异常
            gb.agg([reduction_func], *args)
        # 函数返回，结束当前测试
        return

    # 对 GroupBy 对象执行聚合操作，返回结果
    result = gb.agg([reduction_func], *args)
    # 使用 getattr 获取 GroupBy 对象上 reduction_func 方法的结果
    expected = getattr(gb, reduction_func)(*args)

    # 如果 as_index 为 True 且 (test_series 为 True 或 reduction_func 为 "size")
    if as_index and (test_series or reduction_func == "size"):
        # 将 expected 转换为 DataFrame，设置列名为 reduction_func
        expected = expected.to_frame(reduction_func)
    # 如果 test_series 为 False
    if not test_series:
        # 创建 MultiIndex，将 expected 列名设置为 (ind, "") 对形式，最后一列列名设置为 ("b", reduction_func)
        expected.columns = MultiIndex.from_tuples(
            [(ind, "") for ind in expected.columns[:-1]] + [("b", reduction_func)]
        )
    # 如果 test_series 为 True 且 as_index 为 False
    elif not as_index:
        # 设置 expected 列名为 keys + [reduction_func]
        expected.columns = keys + [reduction_func]

    # 使用 pandas.testing.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_equal(result, expected)
```