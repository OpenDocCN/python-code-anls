# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_value_counts.py`

```
"""
这些代码系统地测试了所有参数组合对 value_counts 的影响，以确保排序的稳定性和参数的正确处理。
"""

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器模块

from pandas import (  # 从 pandas 中导入以下对象
    Categorical,
    CategoricalIndex,
    DataFrame,
    Grouper,
    Index,
    MultiIndex,
    Series,
    date_range,
    to_datetime,
)
import pandas._testing as tm  # 导入 pandas 测试工具模块
from pandas.util.version import Version  # 导入 pandas 版本信息模块


def tests_value_counts_index_names_category_column():
    # GH44324 缺失索引类别列的名称
    df = DataFrame(  # 创建包含两列的 DataFrame
        {
            "gender": ["female"],  # 设置 'gender' 列为包含一个元素的列表
            "country": ["US"],  # 设置 'country' 列为包含一个元素的列表
        }
    )
    df["gender"] = df["gender"].astype("category")  # 将 'gender' 列转换为分类类型
    result = df.groupby("country")["gender"].value_counts()  # 对 'country' 分组并计算 'gender' 列的值计数

    # 构建预期的特定多级索引
    df_mi_expected = DataFrame([["US", "female"]], columns=["country", "gender"])  # 创建预期的 DataFrame
    df_mi_expected["gender"] = df_mi_expected["gender"].astype("category")  # 将 'gender' 列转换为分类类型
    mi_expected = MultiIndex.from_frame(df_mi_expected)  # 从 DataFrame 创建多级索引
    expected = Series([1], index=mi_expected, name="count")  # 创建预期的 Series 对象

    tm.assert_series_equal(result, expected)  # 使用测试工具断言两个 Series 对象是否相等


def seed_df(seed_nans, n, m):
    days = date_range("2015-08-24", periods=10)  # 创建一个日期范围对象

    frame = DataFrame(  # 创建 DataFrame 对象
        {
            "1st": np.random.default_rng(2).choice(list("abcd"), n),  # 随机生成 '1st' 列数据
            "2nd": np.random.default_rng(2).choice(days, n),  # 随机生成 '2nd' 列数据
            "3rd": np.random.default_rng(2).integers(1, m + 1, n),  # 随机生成 '3rd' 列数据
        }
    )

    if seed_nans:
        # 显式将 '3rd' 列转换为浮点型，以避免设置 nan 时的隐式转换
        frame["3rd"] = frame["3rd"].astype("float")
        # 设置部分行的值为 NaN
        frame.loc[1::11, "1st"] = np.nan
        frame.loc[3::17, "2nd"] = np.nan
        frame.loc[7::19, "3rd"] = np.nan
        frame.loc[8::19, "3rd"] = np.nan
        frame.loc[9::19, "3rd"] = np.nan

    return frame  # 返回生成的 DataFrame 对象


@pytest.mark.slow  # 将此测试标记为慢速测试
@pytest.mark.parametrize("seed_nans", [True, False])  # 参数化 seed_nans 参数，分别测试 True 和 False 情况
@pytest.mark.parametrize("num_rows", [10, 50])  # 参数化 num_rows 参数，分别测试 10 和 50 行数据
@pytest.mark.parametrize("max_int", [5, 20])  # 参数化 max_int 参数，分别测试最大整数为 5 和 20
@pytest.mark.parametrize("keys", ["1st", "2nd", ["1st", "2nd"]], ids=repr)  # 参数化 keys 参数，测试不同的键组合
@pytest.mark.parametrize("bins", [None, [0, 5]], ids=repr)  # 参数化 bins 参数，测试不同的分组范围
@pytest.mark.parametrize("isort", [True, False])  # 参数化 isort 参数，分别测试排序和不排序的情况
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])  # 参数化 normalize 和 name 参数
def test_series_groupby_value_counts(
    seed_nans,
    num_rows,
    max_int,
    keys,
    bins,
    isort,
    normalize,
    name,
    sort,
    ascending,
    dropna,
):
    df = seed_df(seed_nans, num_rows, max_int)  # 根据参数生成 DataFrame 对象

    def rebuild_index(df):
        arr = list(map(df.index.get_level_values, range(df.index.nlevels)))  # 重新构建 DataFrame 的索引
        df.index = MultiIndex.from_arrays(arr, names=df.index.names)  # 使用新的索引数组创建多级索引
        return df  # 返回重建索引后的 DataFrame 对象

    kwargs = {
        "normalize": normalize,  # 是否归一化
        "sort": sort,  # 是否排序
        "ascending": ascending,  # 是否升序
        "dropna": dropna,  # 是否丢弃 NaN
        "bins": bins,  # 分组范围
    }

    gr = df.groupby(keys, sort=isort)  # 根据指定键分组
    left = gr["3rd"].value_counts(**kwargs)  # 对分组后的 '3rd' 列进行值计数
    # 根据指定的键(keys)对DataFrame进行分组，并根据isort参数决定是否排序
    gr = df.groupby(keys, sort=isort)
    # 对分组后的结果中的"3rd"列应用Series.value_counts函数，使用kwargs作为额外参数
    right = gr["3rd"].apply(Series.value_counts, **kwargs)
    # 将right的索引名修改为除最后一个以外的索引名，最后一个索引名设置为"3rd"
    right.index.names = right.index.names[:-1] + ["3rd"]
    # 根据指定的名称(name)对right进行重命名
    right = right.rename(name)

    # 由于数值的不稳定排序，必须根据索引排序
    # 使用rebuild_index函数对left和right进行索引重建
    left, right = map(rebuild_index, (left, right))  # xref GH9212
    # 使用tm.assert_series_equal函数断言left和right在索引排序后是否相等
    tm.assert_series_equal(left.sort_index(), right.sort_index())
@pytest.mark.parametrize("utc", [True, False])
# 使用 pytest 的 parametrize 标记，用于多次运行同一个测试函数，参数为 utc 值的列表 [True, False]
def test_series_groupby_value_counts_with_grouper(utc):
    # GH28479
    # 创建 DataFrame 对象 df，包含 "Timestamp" 和 "Food" 列，其中 "Timestamp" 列有多个时间戳
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    # 根据 "Timestamp" 列创建 "Datetime" 列，转换为日期时间格式，根据 utc 参数决定时区
    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s")
    # 按照每日频率对 DataFrame 进行分组，以 "Datetime" 列为关键列
    dfg = df.groupby(Grouper(freq="1D", key="Datetime"))

    # 对分组后的 "Food" 列进行值计数，并按索引排序，以解决值不稳定排序问题（参考 GH9212）
    result = dfg["Food"].value_counts().sort_index()
    # 期望的结果是对分组后的 "Food" 列应用值计数，并按索引排序
    expected = dfg["Food"].apply(Series.value_counts).sort_index()
    # 将期望结果的索引名设置为与结果相同
    expected.index.names = result.index.names
    # 重命名期望结果的数据列为 "count"
    expected = expected.rename("count")

    # 使用 pytest 的 tm 模块断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
# 使用 pytest 的 parametrize 标记，参数为 columns 列表的列表，用于多次运行同一个测试函数
def test_series_groupby_value_counts_empty(columns):
    # GH39172
    # 创建空的 DataFrame 对象 df，列名为参数 columns
    df = DataFrame(columns=columns)
    # 根据除最后一列外的所有列分组
    dfg = df.groupby(columns[:-1])

    # 对最后一列的值进行分组后计数
    result = dfg[columns[-1]].value_counts()
    # 创建一个空的 Series 作为期望结果，与 result 的数据类型和名称相同
    expected = Series([], dtype=result.dtype, name="count")
    # 使用空的 MultiIndex 作为期望结果的索引，索引名为 columns
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)

    # 使用 pytest 的 tm 模块断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("columns", [["A", "B"], ["A", "B", "C"]])
# 使用 pytest 的 parametrize 标记，参数为 columns 列表的列表，用于多次运行同一个测试函数
def test_series_groupby_value_counts_one_row(columns):
    # GH42618
    # 创建 DataFrame 对象 df，包含一行数据，列名为参数 columns
    df = DataFrame(data=[range(len(columns))], columns=columns)
    # 根据除最后一列外的所有列分组
    dfg = df.groupby(columns[:-1])

    # 对最后一列的值进行分组后计数
    result = dfg[columns[-1]].value_counts()
    # 期望结果是对整个 DataFrame 进行值计数
    expected = df.value_counts()

    # 使用 pytest 的 tm 模块断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_on_categorical():
    # GH38672
    # 创建一个 Categorical 类型的 Series 对象 s，包含一个元素 "a"，可选值为 ["a", "b"]
    s = Series(Categorical(["a"], categories=["a", "b"]))
    # 对 Series 对象 s 按照第一个索引进行分组，然后对值进行计数
    result = s.groupby([0]).value_counts()

    # 创建期望的 Series 对象，包含两个索引级别
    expected = Series(
        data=[1, 0],
        index=MultiIndex.from_arrays(
            [
                np.array([0, 0]),
                CategoricalIndex(
                    ["a", "b"], categories=["a", "b"], ordered=False, dtype="category"
                ),
            ]
        ),
        name="count",
    )

    # 使用 pytest 的 tm 模块断言两个 Series 对象相等
    tm.assert_series_equal(result, expected)


def test_series_groupby_value_counts_no_sort():
    # GH#50482
    # 创建一个包含三列的 DataFrame 对象 df，列名为 "gender", "education", "country"
    df = DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    # 按照 ["country", "gender"] 列分组，不对分组结果排序，选择 "education" 列进行值计数
    gb = df.groupby(["country", "gender"], sort=False)["education"]
    # 对分组后的 "education" 列进行值计数，不排序
    result = gb.value_counts(sort=False)
    # 创建一个多级索引对象 `index`
    index = MultiIndex(
        # 定义多级索引的层级及其对应的标签列表
        levels=[["US", "FR"], ["male", "female"], ["low", "medium", "high"]],
        # 指定每个层级对应的代码以构建索引
        codes=[[0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 1, 2, 0, 2]],
        # 指定每个层级的名称
        names=["country", "gender", "education"],
    )
    # 创建一个预期的 Series 对象 `expected`，包含指定的数据和索引
    expected = Series([1, 1, 1, 2, 1], index=index, name="count")
    # 使用测试框架中的方法验证 `result` 和 `expected` 是否相等
    tm.assert_series_equal(result, expected)
@pytest.fixture
def education_df():
    # 返回一个包含教育数据的DataFrame对象，包括gender、education和country三列
    return DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )


def test_bad_subset(education_df):
    # 对education_df按照"country"列进行分组
    gp = education_df.groupby("country")
    # 检查是否会抛出值错误异常，并匹配包含"subset"的错误信息
    with pytest.raises(ValueError, match="subset"):
        # 调用value_counts方法，但给出了不支持的subset参数
        gp.value_counts(subset=["country"])


def test_basic(education_df, request):
    # gh43564
    # 如果numpy的版本大于等于1.25，将会标记该测试用例为xfail状态
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    # 对education_df按照"country"分组，计算每个组内"gender"和"education"的值的频次，并进行归一化
    result = education_df.groupby("country")[["gender", "education"]].value_counts(
        normalize=True
    )
    # 期望的结果是一个包含频次归一化值的Series对象，使用MultiIndex作为索引
    expected = Series(
        data=[0.5, 0.25, 0.25, 0.5, 0.5],
        index=MultiIndex.from_tuples(
            [
                ("FR", "male", "low"),
                ("FR", "female", "high"),
                ("FR", "male", "medium"),
                ("US", "female", "high"),
                ("US", "male", "low"),
            ],
            names=["country", "gender", "education"],
        ),
        name="proportion",
    )
    # 使用测试工具方法assert_series_equal比较计算结果和期望结果是否相同
    tm.assert_series_equal(result, expected)


def _frame_value_counts(df, keys, normalize, sort, ascending):
    # 调用DataFrame的value_counts方法，根据传入的keys参数进行计数统计
    return df[keys].value_counts(normalize=normalize, sort=sort, ascending=ascending)


@pytest.mark.parametrize("groupby", ["column", "array", "function"])
@pytest.mark.parametrize("normalize, name", [(True, "proportion"), (False, "count")])
@pytest.mark.parametrize(
    "sort, ascending",
    [
        (False, None),
        (True, True),
        (True, False),
    ],
)
@pytest.mark.parametrize("frame", [True, False])
def test_against_frame_and_seriesgroupby(
    education_df, groupby, normalize, name, sort, ascending, as_index, frame, request
):
    # 测试所有参数组合:
    # - 使用列、数组或函数作为by参数
    # - 是否进行归一化
    # - 是否进行排序及排序方式
    # - 是否将groupby结果作为索引
    # - 通过不同方式进行比较:
    #   - 使用DataFrame的value_counts方法
    #   - 使用SeriesGroupBy的value_counts方法
    if Version(np.__version__) >= Version("1.25") and frame and sort and normalize:
        # 如果numpy版本大于等于1.25，并且测试中使用了DataFrame，并且开启了排序和归一化，标记该测试为xfail状态
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    # 根据groupby参数选择相应的by参数值
    by = {
        "column": "country",
        "array": education_df["country"].values,
        "function": lambda x: education_df["country"][x] == "US",
    }[groupby]
    # 根据指定的列进行分组，并计算每组中每个组合的频数
    gp = education_df.groupby(by=by, as_index=as_index)
    # 对分组后的结果列进行计数，并返回计数结果
    result = gp[["gender", "education"]].value_counts(
        normalize=normalize, sort=sort, ascending=ascending
    )
    if frame:
        # 如果指定比较 DataFrameGroupBy 的 apply 方法和 value_counts 方法的结果
        # 根据 groupby 参数确定是否抛出 DeprecationWarning
        warn = DeprecationWarning if groupby == "column" else None
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言 apply 方法的结果与 _frame_value_counts 函数的结果相等
        with tm.assert_produces_warning(warn, match=msg):
            expected = gp.apply(
                _frame_value_counts, ["gender", "education"], normalize, sort, ascending
            )
        
        if as_index:
            # 如果 as_index 为 True，直接比较 Series 结果
            tm.assert_series_equal(result, expected)
        else:
            # 否则，重命名预期结果的列，并根据情况插入额外的列
            name = "proportion" if normalize else "count"
            expected = expected.reset_index().rename({0: name}, axis=1)
            if groupby in ["array", "function"] and (not as_index and frame):
                expected.insert(loc=0, column="level_0", value=result["level_0"])
            else:
                expected.insert(loc=0, column="country", value=result["country"])
            # 断言 DataFrame 结果的相等性
            tm.assert_frame_equal(result, expected)
    else:
        # 否则，比较 SeriesGroupBy 的 value_counts 方法的结果
        # 创建新列 "both"，其中包含 "gender" 和 "education" 的组合
        education_df["both"] = education_df["gender"] + "-" + education_df["education"]
        # 计算 "both" 列的频数，并设置名称为 name
        expected = gp["both"].value_counts(
            normalize=normalize, sort=sort, ascending=ascending
        )
        expected.name = name
        if as_index:
            # 如果 as_index 为 True，对索引进行重命名和重组
            index_frame = expected.index.to_frame(index=False)
            index_frame["gender"] = index_frame["both"].str.split("-").str.get(0)
            index_frame["education"] = index_frame["both"].str.split("-").str.get(1)
            del index_frame["both"]
            index_frame = index_frame.rename({0: None}, axis=1)
            expected.index = MultiIndex.from_frame(index_frame)
            # 断言 Series 结果的相等性
            tm.assert_series_equal(result, expected)
        else:
            # 否则，重命名列并删除不必要的列
            expected.insert(1, "gender", expected["both"].str.split("-").str.get(0))
            expected.insert(2, "education", expected["both"].str.split("-").str.get(1))
            del expected["both"]
            # 断言 DataFrame 结果的相等性
            tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "dtype",  # 参数化测试的参数名为 dtype
    [  # 参数化的取值列表
        object,  # 第一个取值为 Python 的 object 类型
        pytest.param("string[pyarrow_numpy]", marks=td.skip_if_no("pyarrow")),  # 第二个取值为字符串 "string[pyarrow_numpy]"，并添加了一个条件标记
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),  # 第三个取值为字符串 "string[pyarrow]"，同样添加了一个条件标记
    ],
)
@pytest.mark.parametrize("normalize", [True, False])  # 参数化测试的参数名为 normalize，取值为 True 和 False
@pytest.mark.parametrize(  # 再次使用 parametrize 装饰器定义多个参数化测试
    "sort, ascending, expected_rows, expected_count, expected_group_size",  # 定义多个参数名称
    [  # 参数化的取值列表
        (False, None, [0, 1, 2, 3, 4], [1, 1, 1, 2, 1], [1, 3, 1, 3, 1]),  # 第一个测试用例的具体取值
        (True, False, [3, 0, 1, 2, 4], [2, 1, 1, 1, 1], [3, 1, 3, 1, 1]),  # 第二个测试用例的具体取值
        (True, True, [0, 1, 2, 4, 3], [1, 1, 1, 1, 2], [1, 3, 1, 1, 3]),  # 第三个测试用例的具体取值
    ],
)
def test_compound(  # 定义名为 test_compound 的测试函数，用于测试复合功能
    education_df,  # 测试函数的参数，名为 education_df，用作测试数据框
    normalize,  # 测试函数的参数，名为 normalize，用于指定是否进行归一化
    sort,  # 测试函数的参数，名为 sort，用于指定是否排序
    ascending,  # 测试函数的参数，名为 ascending，用于指定排序顺序
    expected_rows,  # 测试函数的参数，名为 expected_rows，用于指定预期行索引
    expected_count,  # 测试函数的参数，名为 expected_count，用于指定预期计数
    expected_group_size,  # 测试函数的参数，名为 expected_group_size，用于指定预期组大小
    dtype,  # 测试函数的参数，名为 dtype，用于指定数据类型
):
    education_df = education_df.astype(dtype)  # 将 education_df 数据框中的列转换为指定的数据类型
    education_df.columns = education_df.columns.astype(dtype)  # 将 education_df 数据框的列索引也转换为指定的数据类型
    gp = education_df.groupby(  # 对 education_df 数据框进行分组，指定多个列为分组键，不排序
        ["country", "gender"], as_index=False, sort=False
    )
    result = gp["education"].value_counts(  # 对分组后的数据框中的 "education" 列进行值计数
        normalize=normalize, sort=sort, ascending=ascending
    )
    expected = DataFrame()  # 创建一个空的 DataFrame 对象，用于存储预期结果
    for column in ["country", "gender", "education"]:  # 遍历指定的列
        expected[column] = [education_df[column][row] for row in expected_rows]  # 根据预期的行索引获取对应的数据
        expected = expected.astype(dtype)  # 将 expected 的数据类型转换为指定的数据类型
        expected.columns = expected.columns.astype(dtype)  # 将 expected 的列索引也转换为指定的数据类型
    if normalize:  # 如果 normalize 参数为 True
        expected["proportion"] = expected_count  # 设置预期结果中的 "proportion" 列为预期的计数值
        expected["proportion"] /= expected_group_size  # 将 "proportion" 列除以预期的组大小
        if dtype == "string[pyarrow]":  # 如果数据类型为 "string[pyarrow]"
            expected["proportion"] = expected["proportion"].convert_dtypes()  # 将 "proportion" 列转换为适合的数据类型
    else:  # 如果 normalize 参数为 False
        expected["count"] = expected_count  # 设置预期结果中的 "count" 列为预期的计数值
        if dtype == "string[pyarrow]":  # 如果数据类型为 "string[pyarrow]"
            expected["count"] = expected["count"].convert_dtypes()  # 将 "count" 列转换为适合的数据类型
    tm.assert_frame_equal(result, expected)  # 使用测试框架中的函数比较 result 和 expected 是否相等


@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器定义参数化测试
    "sort, ascending, normalize, name, expected_data, expected_index",  # 参数化测试的参数名
    [  # 参数化的取值列表
        (False, None, False, "count", [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]),  # 第一个测试用例的具体取值
        (True, True, False, "count", [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]),  # 第二个测试用例的具体取值
        (True, False, False, "count", [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]),  # 第三个测试用例的具体取值
        (
            True,
            False,
            True,
            "proportion",
            [0.5, 0.25, 0.25],
            [(1, 1, 1), (4, 2, 6), (0, 2, 0)],
        ),  # 第四个测试用例的具体取值
    ],
)
def test_data_frame_value_counts(  # 定义名为 test_data_frame_value_counts 的测试函数，用于测试 DataFrame 的值计数功能
    sort, ascending, normalize, name, expected_data, expected_index
):
    # 3-way compare with :meth:`~DataFrame.value_counts`
    # Tests from frame/methods/test_value_counts.py
    animals_df = DataFrame(  # 创建一个名为 animals_df 的 DataFrame 对象
        {"key": [1, 1, 1, 1], "num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},  # 设置数据框的列及其数据
        index=["falcon", "dog", "cat", "ant"],  # 设置数据框的索引
    )
    result_frame = animals_df.value_counts(  # 对 animals_df 数据框进行值计数
        sort=sort, ascending=ascending, normalize=normalize
    )
    # 创建一个 Series 对象，用于存储预期的数据
    expected = Series(
        data=expected_data,  # 设置数据部分为 expected_data 变量的值
        index=MultiIndex.from_arrays(
            expected_index,  # 使用 expected_index 变量创建一个 MultiIndex 对象作为索引
            names=["key", "num_legs", "num_wings"]  # 设置 MultiIndex 的各个层级的名称
        ),
        name=name,  # 设置 Series 对象的名称为 name 变量的值
    )
    
    # 使用测试工具函数 tm.assert_series_equal 检查 result_frame 和 expected Series 是否相等
    tm.assert_series_equal(result_frame, expected)
    
    # 对 animals_df 根据 "key" 列进行分组，并计算每组中每个值的出现次数，结果存储在 result_frame_groupby 中
    result_frame_groupby = animals_df.groupby("key").value_counts(
        sort=sort,  # 根据 sort 变量指定的顺序排序结果
        ascending=ascending,  # 根据 ascending 变量指定的顺序升序或降序排序
        normalize=normalize  # 如果 normalize 为 True，则返回相对频率而不是计数
    )
    
    # 使用测试工具函数 tm.assert_series_equal 检查 result_frame_groupby 和 expected Series 是否相等
    tm.assert_series_equal(result_frame_groupby, expected)
# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试参数
@pytest.mark.parametrize(
    "group_dropna, count_dropna, expected_rows, expected_values",
    [
        (
            False,
            False,
            [0, 1, 3, 5, 7, 6, 8, 2, 4],  # 预期行索引列表，指示期望结果的行顺序
            [0.5, 0.5, 1.0, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0],  # 预期值列表，对应于每行的预期值
        ),
        (
            False,
            True,
            [0, 1, 3, 5, 2, 4],  # 预期行索引列表，指示期望结果的行顺序
            [0.5, 0.5, 1.0, 1.0, 1.0, 1.0],  # 预期值列表，对应于每行的预期值
        ),
        (
            True,
            False,
            [0, 1, 5, 7, 6, 8],  # 预期行索引列表，指示期望结果的行顺序
            [0.5, 0.5, 0.25, 0.25, 0.25, 0.25],  # 预期值列表，对应于每行的预期值
        ),
        (
            True,
            True,
            [0, 1, 5],  # 预期行索引列表，指示期望结果的行顺序
            [0.5, 0.5, 1.0],  # 预期值列表，对应于每行的预期值
        ),
    ],
)
def test_dropna_combinations(
    group_dropna, count_dropna, expected_rows, expected_values, request
):
    # 检查 numpy 版本是否大于等于 1.25 且 group_dropna 为 False，如果是则标记为预期失败
    if Version(np.__version__) >= Version("1.25") and not group_dropna:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    
    # 创建包含空值的 DataFrame
    nulls_df = DataFrame(
        {
            "A": [1, 1, np.nan, 4, np.nan, 6, 6, 6, 6],
            "B": [1, 1, 3, np.nan, np.nan, 6, 6, 6, 6],
            "C": [1, 2, 3, 4, 5, 6, np.nan, 8, np.nan],
            "D": [1, 2, 3, 4, 5, 6, 7, np.nan, np.nan],
        }
    )
    
    # 根据 ["A", "B"] 列进行分组，根据 group_dropna 决定是否丢弃空值
    gp = nulls_df.groupby(["A", "B"], dropna=group_dropna)
    
    # 对分组后的结果使用 value_counts 方法，计算各组的比例，并按要求排序和丢弃空值
    result = gp.value_counts(normalize=True, sort=True, dropna=count_dropna)
    
    # 创建一个空的 DataFrame，用于存储根据 expected_rows 指定的预期行索引的列数据
    columns = DataFrame()
    for column in nulls_df.columns:
        columns[column] = [nulls_df[column][row] for row in expected_rows]
    
    # 根据 columns 构建 MultiIndex 对象
    index = MultiIndex.from_frame(columns)
    
    # 创建预期结果的 Series，包括预期的值和索引名称
    expected = Series(data=expected_values, index=index, name="proportion")
    
    # 使用 pytest 的断言方法，验证 result 是否与 expected 相等
    tm.assert_series_equal(result, expected)


# 参数化测试函数，测试 DataFrame 的 value_counts 方法在不同 dropna 设置下的行为
@pytest.mark.parametrize(
    "dropna, expected_data, expected_index",
    [
        (
            True,
            [1, 1],  # 预期数据值列表
            MultiIndex.from_arrays(
                [(1, 1), ("Beth", "John"), ("Louise", "Smith")],  # 预期索引
                names=["key", "first_name", "middle_name"],  # 索引名称
            ),
        ),
        (
            False,
            [1, 1, 1, 1],  # 预期数据值列表
            MultiIndex(
                levels=[
                    Index([1]),
                    Index(["Anne", "Beth", "John"]),
                    Index(["Louise", "Smith", np.nan]),
                ],
                codes=[[0, 0, 0, 0], [0, 1, 2, 2], [2, 0, 1, 2]],  # 索引编码
                names=["key", "first_name", "middle_name"],  # 索引名称
            ),
        ),
    ],
)
@pytest.mark.parametrize("normalize, name", [(False, "count"), (True, "proportion")])
def test_data_frame_value_counts_dropna(
    nulls_fixture, dropna, normalize, name, expected_data, expected_index
):
    # GH 41334
    # 3-way compare with :meth:`~DataFrame.value_counts`
    # Tests with nulls from frame/methods/test_value_counts.py
    # 创建包含空值的 DataFrame，用于测试 value_counts 方法
    names_with_nulls_df = DataFrame(
        {
            "key": [1, 1, 1, 1],
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )
    # 使用 value_counts() 函数计算 DataFrame 列中各元素的频数，并返回一个 Series 对象
    result_frame = names_with_nulls_df.value_counts(dropna=dropna, normalize=normalize)
    
    # 创建一个 Series 对象，指定数据、索引和名称
    expected = Series(
        data=expected_data,
        index=expected_index,
        name=name,
    )
    
    # 如果 normalize 参数为 True，则将 expected Series 中的数据除以数据长度的浮点数形式
    if normalize:
        expected /= float(len(expected_data))
    
    # 使用 assert_series_equal() 函数比较 result_frame 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result_frame, expected)

    # 根据 "key" 列对 DataFrame 进行分组，并使用 groupby() 和 value_counts() 计算分组后各组的频数
    result_frame_groupby = names_with_nulls_df.groupby("key").value_counts(
        dropna=dropna, normalize=normalize
    )
    
    # 使用 assert_series_equal() 函数比较 result_frame_groupby 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result_frame_groupby, expected)
@pytest.mark.parametrize("observed", [False, True])
# 参数化测试，测试observed参数取值为False和True的情况
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [
        (
            False,
            "count",
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
# 参数化测试，测试normalize、name、expected_data参数的多种组合
def test_categorical_single_grouper_with_only_observed_categories(
    education_df, as_index, observed, normalize, name, expected_data, request
):
    # 测试仅包含观察到的分类组的单个分类分组器
    # 当非分组器也是分类时

    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )
    # 根据numpy版本应用标记，标记测试在特定条件下预期失败

    gp = education_df.astype("category").groupby(
        "country", as_index=as_index, observed=observed
    )
    # 将education_df按照"country"列分组为分类数据，并根据参数设置观察或未观察到的分类

    result = gp.value_counts(normalize=normalize)
    # 对分组后的结果进行值计数，根据normalize参数设置是否进行标准化

    expected_index = MultiIndex.from_tuples(
        [
            ("FR", "male", "low"),
            ("FR", "female", "high"),
            ("FR", "male", "medium"),
            ("FR", "female", "low"),
            ("FR", "female", "medium"),
            ("FR", "male", "high"),
            ("US", "female", "high"),
            ("US", "male", "low"),
            ("US", "female", "low"),
            ("US", "female", "medium"),
            ("US", "male", "high"),
            ("US", "male", "medium"),
        ],
        names=["country", "gender", "education"],
    )
    # 创建预期的多级索引，包含"country"、"gender"、"education"三个层级

    expected_series = Series(
        data=expected_data,
        index=expected_index,
        name=name,
    )
    # 创建预期的Series对象，数据为expected_data，索引为expected_index，名称为name

    for i in range(3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )
    # 将预期Series的每个层级索引转换为分类索引

    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        # 如果不使用索引，则将预期Series重置为DataFrame，并根据normalize参数命名列为"proportion"或"count"

        tm.assert_frame_equal(result, expected)
        # 断言结果DataFrame与预期DataFrame相等


def assert_categorical_single_grouper(
    education_df, as_index, observed, expected_index, normalize, name, expected_data
):
    # 测试单个分类分组器，当非分组器也是分类时

    education_df = education_df.copy().astype("category")
    # 将education_df复制一份，并将所有列转换为分类数据类型

    education_df["country"] = education_df["country"].cat.add_categories(["ASIA"])
    # 将"country"列添加新的分类类别"ASIA"

    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    # 将education_df按照"country"列分组，并根据参数设置观察或未观察到的分类

    result = gp.value_counts(normalize=normalize)
    # 对分组后的结果进行值计数，根据normalize参数设置是否进行标准化
    # 创建一个 Series 对象，指定数据和索引
    expected_series = Series(
        data=expected_data,  # 用给定的数据初始化 Series
        index=MultiIndex.from_tuples(
            expected_index,  # 使用给定的元组列表创建 MultiIndex，指定索引的多个层级和名称
            names=["country", "gender", "education"],  # 指定每个索引层级的名称
        ),
        name=name,  # 设置 Series 的名称
    )
    
    # 遍历前三个索引层级
    for i in range(3):
        # 创建一个 CategoricalIndex 对象，使用预期 Series 的每个索引层级
        index_level = CategoricalIndex(expected_series.index.levels[i])
        
        # 如果当前是第一个索引层级
        if i == 0:
            # 将第一个索引层级设置为指定的分类列表
            index_level = index_level.set_categories(
                education_df["country"].cat.categories  # 使用 education_df 数据框的国家分类
            )
        
        # 将预期 Series 的当前索引层级设置为新创建的 index_level
        expected_series.index = expected_series.index.set_levels(index_level, level=i)
    
    # 如果需要作为索引比较
    if as_index:
        # 使用 tm.assert_series_equal 检查结果和预期 Series 是否相等
        tm.assert_series_equal(result, expected_series)
    else:
        # 否则，将预期 Series 转换为数据框并重置索引，设置列名为 name
        expected = expected_series.reset_index(name=name)
        # 使用 tm.assert_frame_equal 检查结果和预期的数据框是否相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [  # 参数化测试，定义测试用例的参数
        (
            False,  # 是否进行归一化
            "count",  # 测试数据名称
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),  # 预期数据数组
        ),
        (
            True,  # 是否进行归一化
            "proportion",  # 测试数据名称
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),  # 预期数据数组
        ),
    ],
)
def test_categorical_single_grouper_observed_true(
    education_df, as_index, normalize, name, expected_data, request
):
    # GH#46357
    # 如果 NumPy 的版本大于等于 1.25，则标记为预期失败，提供失败原因
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,  # 允许宽松的失败标记
            )
        )

    expected_index = [  # 预期的索引值列表
        ("FR", "male", "low"),
        ("FR", "female", "high"),
        ("FR", "male", "medium"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("FR", "male", "high"),
        ("US", "female", "high"),
        ("US", "male", "low"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("US", "male", "high"),
        ("US", "male", "medium"),
    ]

    assert_categorical_single_grouper(  # 调用断言函数，验证分类数据的单一分组器
        education_df=education_df,
        as_index=as_index,
        observed=True,  # 指定观察模式为真
        expected_index=expected_index,
        normalize=normalize,
        name=name,
        expected_data=expected_data,
    )


@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [  # 参数化测试，定义测试用例的参数
        (
            False,  # 是否进行归一化
            "count",  # 测试数据名称
            np.array(
                [2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64  # 预期数据数组
            ),
        ),
        (
            True,  # 是否进行归一化
            "proportion",  # 测试数据名称
            np.array(
                [
                    0.5,
                    0.25,
                    0.25,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),  # 预期数据数组
        ),
    ],
)
def test_categorical_single_grouper_observed_false(
    education_df, as_index, normalize, name, expected_data, request
):
    # GH#46357
    # 如果 NumPy 的版本大于等于 1.25，则标记为预期失败，提供失败原因
    if Version(np.__version__) >= Version("1.25"):
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,  # 允许宽松的失败标记
            )
        )
    expected_index = [
        ("FR", "male", "low"),            # 预期的索引值列表，每个元素是一个元组，表示国家、性别和教育水平的组合
        ("FR", "female", "high"),
        ("FR", "male", "medium"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("FR", "male", "high"),
        ("US", "female", "high"),
        ("US", "male", "low"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("US", "male", "high"),
        ("US", "male", "medium"),
        ("ASIA", "female", "high"),
        ("ASIA", "female", "low"),
        ("ASIA", "female", "medium"),
        ("ASIA", "male", "high"),
        ("ASIA", "male", "low"),
        ("ASIA", "male", "medium"),
    ]

    assert_categorical_single_grouper(
        education_df=education_df,         # 参数：教育数据的数据框
        as_index=as_index,                 # 参数：是否使用列作为索引
        observed=False,                    # 参数：是否观察到预期数据
        expected_index=expected_index,     # 参数：预期的索引列表，用于断言检查
        normalize=normalize,               # 参数：是否进行归一化处理
        name=name,                         # 参数：名称标识
        expected_data=expected_data,       # 参数：预期的数据
    )
@pytest.mark.parametrize(
    "observed, expected_index",
    [  # 参数化测试的参数定义：observed 表示是否观察到，expected_index 表示预期的索引
        (
            False,
            [
                ("FR", "high", "female"),
                ("FR", "high", "male"),
                ("FR", "low", "male"),
                ("FR", "low", "female"),
                ("FR", "medium", "male"),
                ("FR", "medium", "female"),
                ("US", "high", "female"),
                ("US", "high", "male"),
                ("US", "low", "male"),
                ("US", "low", "female"),
                ("US", "medium", "female"),
                ("US", "medium", "male"),
            ],
        ),
        (
            True,
            [
                ("FR", "high", "female"),
                ("FR", "low", "male"),
                ("FR", "medium", "male"),
                ("US", "high", "female"),
                ("US", "low", "male"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "normalize, name, expected_data",
    [  # 另一个参数化测试的参数定义：normalize 表示是否进行归一化，name 表示名称，expected_data 表示预期的数据
        (
            False,
            "count",
            np.array([1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int64),
        ),
        (
            True,
            "proportion",
            # NaN values corresponds to non-observed groups
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_categorical_multiple_groupers(
    education_df, as_index, observed, expected_index, normalize, name, expected_data
):
    # GH#46357
    # 用于跟踪 GitHub 问题编号

    # Test multiple categorical groupers when non-groupers are non-categorical
    # 当非分组键非分类时，测试多个分类分组器
    education_df = education_df.copy()
    education_df["country"] = education_df["country"].astype("category")
    education_df["education"] = education_df["education"].astype("category")

    # 进行分组操作，按照指定参数进行分组
    gp = education_df.groupby(
        ["country", "education"], as_index=as_index, observed=observed
    )
    # 获取分组后的值计数结果
    result = gp.value_counts(normalize=normalize)

    # 创建预期的 Series 对象
    expected_series = Series(
        data=expected_data[expected_data > 0.0] if observed else expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "education", "gender"],
        ),
        name=name,
    )

    # 将 Series 的索引转换为分类索引
    for i in range(2):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    # 根据是否使用索引参数，比较结果
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)
    [
        (
            False,  # 第一个元组的第一个元素，表示是否为比例数据
            "count",  # 第一个元组的第二个元素，标识数据类型为计数
            np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),  # 数组，包含计数数据
        ),
        (
            True,  # 第二个元组的第一个元素，表示是否为比例数据
            "proportion",  # 第二个元组的第二个元素，标识数据类型为比例
            # NaN values corresponds to non-observed groups
            np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),  # 数组，包含比例数据，NaN表示未观察到的组
        ),
    ],
)  # 这里有一个语法错误，括号不匹配

def test_categorical_non_groupers(
    education_df, as_index, observed, normalize, name, expected_data, request
):
    # GH#46357 Test non-observed categories are included in the result,
    # regardless of `observed`

    if Version(np.__version__) >= Version("1.25"):
        # 如果 NumPy 版本大于等于 1.25，则应用 xfail 标记
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "pandas default unstable sorting of duplicates"
                    "issue with numpy>=1.25 with AVX instructions"
                ),
                strict=False,
            )
        )

    education_df = education_df.copy()
    # 将 'gender' 和 'education' 列转换为分类类型
    education_df["gender"] = education_df["gender"].astype("category")
    education_df["education"] = education_df["education"].astype("category")

    # 根据 'country' 列分组，根据参数设置确定是否作为索引和观察值
    gp = education_df.groupby("country", as_index=as_index, observed=observed)
    # 对分组后的数据进行值计数，可以选择是否进行归一化
    result = gp.value_counts(normalize=normalize)

    expected_index = [
        ("FR", "male", "low"),
        ("FR", "female", "high"),
        ("FR", "male", "medium"),
        ("FR", "female", "low"),
        ("FR", "female", "medium"),
        ("FR", "male", "high"),
        ("US", "female", "high"),
        ("US", "male", "low"),
        ("US", "female", "low"),
        ("US", "female", "medium"),
        ("US", "male", "high"),
        ("US", "male", "medium"),
    ]
    # 创建预期的 MultiIndex 结构的 Series 对象
    expected_series = Series(
        data=expected_data,
        index=MultiIndex.from_tuples(
            expected_index,
            names=["country", "gender", "education"],
        ),
        name=name,
    )
    # 将预期 Series 对象的索引的每个级别设置为分类类型
    for i in range(1, 3):
        expected_series.index = expected_series.index.set_levels(
            CategoricalIndex(expected_series.index.levels[i]), level=i
        )

    if as_index:
        # 如果 as_index 为 True，则断言结果是预期的 Series 对象
        tm.assert_series_equal(result, expected_series)
    else:
        # 如果 as_index 为 False，则将预期 Series 转换为 DataFrame 并进行断言
        expected = expected_series.reset_index(
            name="proportion" if normalize else "count"
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label, expected_values",
    [
        (False, "count", [1, 1, 1]),
        (True, "proportion", [0.5, 0.5, 1.0]),
    ],
)
def test_mixed_groupings(normalize, expected_label, expected_values):
    # Test multiple groupings
    # 创建一个 DataFrame 对象 df
    df = DataFrame({"A": [1, 2, 1], "B": [1, 2, 3]})
    # 根据多个键或函数进行分组，as_index 设为 False
    gp = df.groupby([[4, 5, 4], "A", lambda i: 7 if i == 1 else 8], as_index=False)
    # 对分组后的数据进行值计数，并指定排序和是否归一化
    result = gp.value_counts(sort=True, normalize=normalize)
    # 创建预期的 DataFrame 对象 expected
    expected = DataFrame(
        {
            "level_0": np.array([4, 4, 5], dtype=int),
            "A": [1, 1, 2],
            "level_2": [8, 8, 7],
            "B": [1, 3, 2],
            expected_label: expected_values,
        }
    )
    # 断言分组后的结果与预期的 DataFrame 对象 expected 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test, columns, expected_names",
    [
        ("repeat", list("abbde"), ["a", None, "d", "b", "b", "e"]),
        ("level", list("abcd") + ["level_1"], ["a", None, "d", "b", "c", "level_1"]),
    ],
)
def test_column_label_duplicates(test, columns, expected_names, as_index):
    # GH 44992
    # Test for duplicate input column labels and generated duplicate labels
    
    # 创建一个 DataFrame，用于测试重复的列标签和生成的重复标签
    df = DataFrame([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], columns=columns)
    
    # 预期的数据格式
    expected_data = [(1, 0, 7, 3, 5, 9), (2, 1, 8, 4, 6, 10)]
    
    # 分组的键，包括字符串和 NumPy 数组
    keys = ["a", np.array([0, 1], dtype=np.int64), "d"]
    
    # 对 DataFrame 进行分组并计算值的出现次数
    result = df.groupby(keys, as_index=as_index).value_counts()
    
    if as_index:
        # 如果 as_index 为 True，则预期的结果是一个 Series
        expected = Series(
            data=(1, 1),
            index=MultiIndex.from_tuples(
                expected_data,
                names=expected_names,
            ),
            name="count",
        )
        tm.assert_series_equal(result, expected)
    else:
        # 如果 as_index 为 False，则预期的结果是一个 DataFrame
        expected_data = [list(row) + [1] for row in expected_data]
        expected_columns = list(expected_names)
        expected_columns[1] = "level_1"
        expected_columns.append("count")
        expected = DataFrame(expected_data, columns=expected_columns)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "normalize, expected_label",
    [
        (False, "count"),
        (True, "proportion"),
    ],
)
def test_result_label_duplicates(normalize, expected_label):
    # Test for result column label duplicating an input column label
    
    # 创建一个 DataFrame 并进行分组，检查结果列标签是否重复输入列标签
    gb = DataFrame([[1, 2, 3]], columns=["a", "b", expected_label]).groupby(
        "a", as_index=False
    )
    
    # 出现 ValueError 异常的消息
    msg = f"Column label '{expected_label}' is duplicate of result column"
    
    # 断言应该抛出 ValueError 异常并匹配预期的消息
    with pytest.raises(ValueError, match=msg):
        gb.value_counts(normalize=normalize)


def test_ambiguous_grouping():
    # Test that groupby is not confused by groupings length equal to row count
    
    # 创建一个包含重复分组键的 DataFrame，测试 groupby 方法不会因为分组键的长度等于行数而混淆
    df = DataFrame({"a": [1, 1]})
    
    # 根据数组进行分组
    gb = df.groupby(np.array([1, 1], dtype=np.int64))
    
    # 计算分组后每组的值出现次数
    result = gb.value_counts()
    
    # 期望的结果是一个 Series
    expected = Series(
        [2], index=MultiIndex.from_tuples([[1, 1]], names=[None, "a"]), name="count"
    )
    tm.assert_series_equal(result, expected)


def test_subset_overlaps_gb_key_raises():
    # GH 46383
    # Test that subset of keys cannot overlap with groupby column keys
    
    # 创建一个 DataFrame，测试子集的键不能与 groupby 列的键重叠
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    
    # 预期的错误消息
    msg = "Keys {'c1'} in subset cannot be in the groupby column keys."
    
    # 断言应该抛出 ValueError 异常并匹配预期的消息
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c1"])


def test_subset_doesnt_exist_in_frame():
    # GH 46383
    # Test that subset of keys must exist in the DataFrame
    
    # 创建一个 DataFrame，测试子集的键必须存在于 DataFrame 中
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    
    # 预期的错误消息
    msg = "Keys {'c3'} in subset do not exist in the DataFrame."
    
    # 断言应该抛出 ValueError 异常并匹配预期的消息
    with pytest.raises(ValueError, match=msg):
        df.groupby("c1").value_counts(subset=["c3"])


def test_subset():
    # GH 46383
    # Test subset argument for groupby value_counts
    
    # 创建一个 DataFrame，测试 groupby 后使用 subset 参数进行值计数
    df = DataFrame({"c1": ["a", "b", "c"], "c2": ["x", "y", "y"]}, index=[0, 1, 1])
    
    # 根据索引级别进行分组并计算值的出现次数
    result = df.groupby(level=0).value_counts(subset=["c2"])
    
    # 期望的结果是一个 Series
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays([[0, 1], ["x", "y"]], names=[None, "c2"]),
        name="count",
    )
    tm.assert_series_equal(result, expected)


def test_subset_duplicate_columns():
    # Test subset argument with duplicate columns
    
    # 在将 DataFrame 的列作为子集键时，测试处理重复列名的情况
    # GH 46383
    # 创建一个 DataFrame 对象，包含三行数据和三列列名，其中第二列列名重复
    df = DataFrame(
        [["a", "x", "x"], ["b", "y", "y"], ["b", "y", "y"]],
        index=[0, 1, 1],
        columns=["c1", "c2", "c2"],
    )
    
    # 对 DataFrame 进行分组，按照第一级索引进行分组，并计算每组中"c2"列值的出现次数
    result = df.groupby(level=0).value_counts(subset=["c2"])
    
    # 创建一个预期的 Series 对象，包含两个元素，使用 MultiIndex 指定了索引的多层结构
    expected = Series(
        [1, 2],
        index=MultiIndex.from_arrays(
            [[0, 1], ["x", "y"], ["x", "y"]],
            names=[None, "c2", "c2"]
        ),
        name="count",
    )
    
    # 使用测试工具（tm.assert_series_equal）比较计算得到的结果和预期结果，确保它们相等
    tm.assert_series_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器定义参数化测试，测试utc为True和False两种情况
@pytest.mark.parametrize("utc", [True, False])
def test_value_counts_time_grouper(utc, unit):
    # GH#50486
    # 创建包含时间戳和食物类型的数据框
    df = DataFrame(
        {
            "Timestamp": [
                1565083561,
                1565083561 + 86400,
                1565083561 + 86500,
                1565083561 + 86400 * 2,
                1565083561 + 86400 * 3,
                1565083561 + 86500 * 3,
                1565083561 + 86400 * 4,
            ],
            "Food": ["apple", "apple", "banana", "banana", "orange", "orange", "pear"],
        }
    ).drop([3])

    # 根据时间戳创建 Datetime 列，并按给定的单位转换
    df["Datetime"] = to_datetime(df["Timestamp"], utc=utc, unit="s").dt.as_unit(unit)
    # 按照日期频率进行分组
    gb = df.groupby(Grouper(freq="1D", key="Datetime"))
    # 对分组结果进行值计数
    result = gb.value_counts()
    # 创建指定日期和时间戳的 Datetime 对象列表
    dates = to_datetime(
        ["2019-08-06", "2019-08-07", "2019-08-09", "2019-08-10"], utc=utc
    ).as_unit(unit)
    # 获取时间戳的唯一值
    timestamps = df["Timestamp"].unique()
    # 创建 MultiIndex 对象，指定层级、代码和名称
    index = MultiIndex(
        levels=[dates, timestamps, ["apple", "banana", "orange", "pear"]],
        codes=[[0, 1, 1, 2, 2, 3], range(6), [0, 0, 1, 2, 2, 3]],
        names=["Datetime", "Timestamp", "Food"],
    )
    # 创建预期的 Series 对象，包含计数值
    expected = Series(1, index=index, name="count")
    # 断言分组结果与预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，测试整数列的值计数情况
def test_value_counts_integer_columns():
    # GH#55627
    # 创建包含整数列的数据框
    df = DataFrame({1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"]})
    # 按照多列进行分组
    gp = df.groupby([1, 2], as_index=False, sort=False)
    # 对分组结果的第三列进行值计数
    result = gp[3].value_counts()
    # 创建预期的 DataFrame 对象，包含计数值
    expected = DataFrame(
        {1: ["a", "a", "a"], 2: ["a", "a", "d"], 3: ["a", "b", "c"], "count": 1}
    )
    # 断言分组结果与预期结果相等
    tm.assert_frame_equal(result, expected)


# 使用两个参数化装饰器定义测试函数，测试值计数排序和标准化
@pytest.mark.parametrize("vc_sort", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts_sort(sort, vc_sort, normalize):
    # GH#55951
    # 创建包含两列的数据框
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]})
    # 按照 'a' 列进行分组，可以选择是否排序
    gb = df.groupby("a", sort=sort)
    # 对分组结果进行值计数，可以选择排序和标准化
    result = gb.value_counts(sort=vc_sort, normalize=normalize)

    # 根据是否标准化选择预期的值列表
    if normalize:
        values = [2 / 3, 1 / 3, 1.0]
    else:
        values = [2, 1, 1]
    # 创建 MultiIndex 对象，指定层级和代码
    index = MultiIndex(
        levels=[[1, 2], [3, 4]], codes=[[0, 0, 1], [0, 1, 0]], names=["a", 0]
    )
    # 创建预期的 Series 对象，包含计数或比例值
    expected = Series(values, index=index, name="proportion" if normalize else "count")
    # 根据排序和值计数排序选择结果
    if sort and vc_sort:
        taker = [0, 1, 2]
    elif sort and not vc_sort:
        taker = [0, 1, 2]
    elif not sort and vc_sort:
        taker = [0, 2, 1]
    else:
        taker = [2, 1, 0]
    expected = expected.take(taker)

    # 断言分组结果与预期结果相等
    tm.assert_series_equal(result, expected)


# 使用两个参数化装饰器定义测试函数，测试分类数据的值计数排序和标准化
@pytest.mark.parametrize("vc_sort", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts_sort_categorical(sort, vc_sort, normalize):
    # GH#55951
    # 创建包含两列的分类数据框
    df = DataFrame({"a": [2, 1, 1, 1], 0: [3, 4, 3, 3]}, dtype="category")
    # 按照 'a' 列进行分组，可以选择是否排序和是否观察到所有分类
    gb = df.groupby("a", sort=sort, observed=True)
    # 对分组结果进行值计数，可以选择排序和标准化
    result = gb.value_counts(sort=vc_sort, normalize=normalize)

    # 根据是否标准化选择预期的值列表
    if normalize:
        values = [2 / 3, 1 / 3, 1.0, 0.0]
    else:
        values = [2, 1, 1, 0]
    # 根据条件选择要使用的列名，根据 normalize 变量确定选择 "proportion" 或 "count"
    name = "proportion" if normalize else "count"
    # 创建一个 DataFrame 对象，包含三列："a", 0 和根据 name 变量确定的第三列，用 values 初始化
    expected = DataFrame(
        {
            "a": Categorical([1, 1, 2, 2]),
            0: Categorical([3, 4, 3, 4]),
            name: values,
        }
    )
    # 设置 DataFrame 的多级索引为 ["a", 0]，并选取第 name 列作为 Series 对象 expected
    expected = expected.set_index(["a", 0])[name]
    
    # 根据 sort 和 vc_sort 变量的值选择索引的顺序
    if sort and vc_sort:
        taker = [0, 1, 2, 3]  # 如果 sort 和 vc_sort 均为 True，则保持原始顺序
    elif sort and not vc_sort:
        taker = [0, 1, 2, 3]  # 如果 sort 为 True 而 vc_sort 为 False，则保持原始顺序
    elif not sort and vc_sort:
        taker = [0, 2, 1, 3]  # 如果 sort 为 False 而 vc_sort 为 True，则调整顺序
    else:
        taker = [2, 1, 0, 3]  # 如果 sort 和 vc_sort 均为 False，则调整顺序

    # 根据 taker 列表的顺序重新排列 expected Series 对象
    expected = expected.take(taker)

    # 使用 test_helper 模块的 assert_series_equal 函数比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
```