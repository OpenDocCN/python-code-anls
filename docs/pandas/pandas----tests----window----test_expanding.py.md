# `D:\src\scipysrc\pandas\pandas\tests\window\test_expanding.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入多个模块和函数
    DataFrame,  # 数据帧结构
    DatetimeIndex,  # 时间日期索引
    Index,  # 索引对象
    MultiIndex,  # 多级索引
    Series,  # 数据序列
    isna,  # 判断缺失值的函数
    notna,  # 判断非缺失值的函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


def test_doc_string():
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})  # 创建包含 NaN 的数据帧
    df  # 打印数据帧（此处注释不需要）
    df.expanding(2).sum()  # 对数据帧进行扩展运算，计算累积和


def test_constructor(frame_or_series):
    # GH 12669
    c = frame_or_series(range(5)).expanding  # 获取数据帧或序列对象的扩展属性

    # valid
    c(min_periods=1)  # 使用最小观测期数为 1 进行扩展操作


@pytest.mark.parametrize("w", [2.0, "foo", np.array([2])])
def test_constructor_invalid(frame_or_series, w):
    # not valid

    c = frame_or_series(range(5)).expanding  # 获取数据帧或序列对象的扩展属性
    msg = "min_periods must be an integer"  # 异常信息
    with pytest.raises(ValueError, match=msg):  # 捕获 ValueError 异常并匹配消息
        c(min_periods=w)  # 使用非整数值进行扩展操作，应抛出异常


@pytest.mark.parametrize(
    "expander",
    [
        1,  # 整数类型的扩展器
        pytest.param(
            "ls",
            marks=pytest.mark.xfail(
                reason="GH#16425 expanding with offset not supported"
            ),
        ),  # 使用 pytest 的参数化方式，标记测试用例为预期失败
    ],
)
def test_empty_df_expanding(expander):
    # GH 15819 Verifies that datetime and integer expanding windows can be
    # applied to empty DataFrames

    expected = DataFrame()  # 创建预期结果为空的数据帧
    result = DataFrame().expanding(expander).sum()  # 对空数据帧应用扩展窗口并计算累积和
    tm.assert_frame_equal(result, expected)  # 断言结果与预期一致

    # Verifies that datetime and integer expanding windows can be applied
    # to empty DataFrames with datetime index
    expected = DataFrame(index=DatetimeIndex([]))  # 创建具有空日期时间索引的预期结果数据帧
    result = DataFrame(index=DatetimeIndex([])).expanding(expander).sum()  # 对具有空日期时间索引的数据帧应用扩展窗口并计算累积和
    tm.assert_frame_equal(result, expected)  # 断言结果与预期一致


def test_missing_minp_zero():
    # https://github.com/pandas-dev/pandas/pull/18921
    # minp=0
    x = Series([np.nan])  # 创建包含 NaN 的数据序列
    result = x.expanding(min_periods=0).sum()  # 使用最小观测期数为 0 进行扩展操作并计算累积和
    expected = Series([0.0])  # 创建预期结果为包含单个值的数据序列
    tm.assert_series_equal(result, expected)  # 断言结果序列与预期一致

    # minp=1
    result = x.expanding(min_periods=1).sum()  # 使用最小观测期数为 1 进行扩展操作并计算累积和
    expected = Series([np.nan])  # 创建预期结果为包含单个 NaN 值的数据序列
    tm.assert_series_equal(result, expected)  # 断言结果序列与预期一致


def test_expanding():
    # see gh-23372.
    df = DataFrame(np.ones((10, 20)))  # 创建一个 10x20 的数据帧，所有元素为 1

    expected = DataFrame(  # 创建预期结果的数据帧
        {i: [np.nan] * 2 + [float(j) for j in range(3, 11)] for i in range(20)}  # 使用列表推导式生成每列的预期结果
    )
    result = df.expanding(3).sum()  # 对数据帧应用扩展窗口并计算累积和
    tm.assert_frame_equal(result, expected)  # 断言结果数据帧与预期一致


def test_expanding_count_with_min_periods(frame_or_series):
    # GH 26996
    result = frame_or_series(range(5)).expanding(min_periods=3).count()  # 使用最小观测期数为 3 进行扩展操作并计算累积计数
    expected = frame_or_series([np.nan, np.nan, 3.0, 4.0, 5.0])  # 创建预期结果的数据序列
    tm.assert_equal(result, expected)  # 断言结果序列与预期一致


def test_expanding_count_default_min_periods_with_null_values(frame_or_series):
    # GH 26996
    values = [1, 2, 3, np.nan, 4, 5, 6]  # 创建包含 NaN 的数据列表
    expected_counts = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]  # 创建预期结果的计数值列表

    result = frame_or_series(values).expanding().count()  # 对数据序列应用默认最小观测期数的扩展操作并计算累积计数
    expected = frame_or_series(expected_counts)  # 创建预期结果的数据序列
    tm.assert_equal(result, expected)  # 断言结果序列与预期一致


def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series):
    # GH 25857
    result = frame_or_series(range(5)).expanding(min_periods=6).count()  # 使用大于序列长度的最小观测期数进行扩展操作并计算累积计数
    # 创建一个期望的对象，其中包含一个由 np.nan 组成的列表，用于测试
    expected = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    # 使用测试框架的断言方法，验证结果与期望是否相等
    tm.assert_equal(result, expected)
@pytest.mark.parametrize(
    "df,expected,min_periods",
    [  # 参数化测试：提供不同的输入和预期输出
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},  # 输入的 DataFrame 格式数据
            [
                ({"A": [1], "B": [4]}, [0]),  # 预期的输出格式：包含的 DataFrame 和对应的索引列表
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            3,  # 最小期数参数
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            2,
        ),
        (
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            [
                ({"A": [1], "B": [4]}, [0]),
                ({"A": [1, 2], "B": [4, 5]}, [0, 1]),
                ({"A": [1, 2, 3], "B": [4, 5, 6]}, [0, 1, 2]),
            ],
            1,
        ),
        ({"A": [1], "B": [4]}, [], 2),
        (None, [({}, [])], 1),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            3,
        ),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            2,
        ),
        (
            {"A": [1, np.nan, 3], "B": [np.nan, 5, 6]},
            [
                ({"A": [1.0], "B": [np.nan]}, [0]),
                ({"A": [1, np.nan], "B": [np.nan, 5]}, [0, 1]),
                ({"A": [1, np.nan, 3], "B": [np.nan, 5, 6]}, [0, 1, 2]),
            ],
            1,
        ),
    ],
)
def test_iter_expanding_dataframe(df, expected, min_periods):
    # GH 11704
    df = DataFrame(df)  # 将输入的 df 转换为 DataFrame 格式
    expecteds = [DataFrame(values, index=index) for (values, index) in expected]  # 构建预期的 DataFrame 格式数据列表

    for expected, actual in zip(expecteds, df.expanding(min_periods)):  # 遍历预期输出和实际输出的 DataFrame
        tm.assert_frame_equal(actual, expected)  # 使用测试框架的方法比较实际输出和预期输出的 DataFrame


@pytest.mark.parametrize(
    "ser,expected,min_periods",
    [  # 参数化测试：提供不同的输入和预期输出
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 3),  # 输入的 Series 数据
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 2),
        (Series([1, 2, 3]), [([1], [0]), ([1, 2], [0, 1]), ([1, 2, 3], [0, 1, 2])], 1),
        (Series([1, 2]), [([1], [0]), ([1, 2], [0, 1])], 2),
        (Series([np.nan, 2]), [([np.nan], [0]), ([np.nan, 2], [0, 1])], 2),
        (Series([], dtype="int64"), [], 2),
    ],
)
def test_iter_expanding_series(ser, expected, min_periods):
    # GH 11704
    expecteds = [Series(values, index=index) for (values, index) in expected]  # 构建预期的 Series 格式数据列表

    for expected, actual in zip(expecteds, ser.expanding(min_periods)):  # 遍历预期输出和实际输出的 Series
        tm.assert_series_equal(actual, expected)  # 使用测试框架的方法比较实际输出和预期输出的 Series
def test_center_invalid():
    # 标记：GH 20647
    # 创建一个空的 DataFrame 对象
    df = DataFrame()
    # 使用 pytest 检查是否会抛出 TypeError 异常，且异常消息包含特定文本
    with pytest.raises(TypeError, match=".* got an unexpected keyword"):
        # 调用 DataFrame 对象的 expanding 方法，传入 center=True 参数
        df.expanding(center=True)


def test_expanding_sem(frame_or_series):
    # GH: 26476
    # 根据传入的 frame_or_series 函数创建一个对象
    obj = frame_or_series([0, 1, 2])
    # 调用对象的 expanding 方法，并计算其标准误差
    result = obj.expanding().sem()
    # 如果结果是 DataFrame 类型，则转换成 Series 类型，取第一个值
    if isinstance(result, DataFrame):
        result = Series(result[0].values)
    # 期望的结果是一个 Series，包含特定的数值
    expected = Series([np.nan] + [0.707107] * 2)
    # 使用 pandas 的 assert_series_equal 方法比较结果和期望值
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_expanding_skew_kurt_numerical_stability(method):
    # GH: 6929
    # 创建一个具有随机数据的 Series 对象
    s = Series(np.random.default_rng(2).random(10))
    # 获取对象的 expanding(3) 的 skew 或 kurtosis 值
    expected = getattr(s.expanding(3), method)()
    # 将 Series 对象的值加上 5000
    s = s + 5000
    # 再次获取对象的 expanding(3) 的 skew 或 kurtosis 值
    result = getattr(s.expanding(3), method)()
    # 使用 pandas 的 assert_series_equal 方法比较结果和期望值
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("window", [1, 3, 10, 20])
@pytest.mark.parametrize("method", ["min", "max", "average"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize("test_data", ["default", "duplicates", "nans"])
def test_rank(window, method, pct, ascending, test_data):
    # 设置长度为 20
    length = 20
    # 根据不同的 test_data 类型生成不同的 Series 数据
    if test_data == "default":
        ser = Series(data=np.random.default_rng(2).random(length))
    elif test_data == "duplicates":
        ser = Series(data=np.random.default_rng(2).choice(3, length))
    elif test_data == "nans":
        ser = Series(
            data=np.random.default_rng(2).choice(
                [1.0, 0.25, 0.75, np.nan, np.inf, -np.inf], length
            )
        )

    # 计算 expanding 窗口内每个元素的排名
    expected = ser.expanding(window).apply(
        lambda x: x.rank(method=method, pct=pct, ascending=ascending).iloc[-1]
    )
    # 获取 expanding 窗口内每个元素的排名
    result = ser.expanding(window).rank(method=method, pct=pct, ascending=ascending)

    # 使用 pandas 的 assert_series_equal 方法比较结果和期望值
    tm.assert_series_equal(result, expected)


def test_expanding_corr(series):
    # 删除 NaN 值后的 Series A
    A = series.dropna()
    # 生成一个与 A 相同长度的随机数列 B
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]

    # 计算 A 和 B 的 expanding 窗口内的相关系数
    result = A.expanding().corr(B)

    # 使用 rolling 方法计算 A 和 B 的滚动窗口内的相关系数
    rolling_result = A.rolling(window=len(A), min_periods=1).corr(B)

    # 使用 pandas 的 assert_almost_equal 方法比较结果和期望值
    tm.assert_almost_equal(rolling_result, result)


def test_expanding_count(series):
    # 计算 Series 的 expanding 窗口内的计数
    result = series.expanding(min_periods=0).count()
    # 使用 rolling 方法计算 Series 的滚动窗口内的计数
    tm.assert_almost_equal(
        result, series.rolling(window=len(series), min_periods=0).count()
    )


def test_expanding_quantile(series):
    # 计算 Series 的 expanding 窗口内的分位数（0.5）
    result = series.expanding().quantile(0.5)

    # 使用 rolling 方法计算 Series 的滚动窗口内的分位数（0.5）
    rolling_result = series.rolling(window=len(series), min_periods=1).quantile(0.5)

    # 使用 pandas 的 assert_almost_equal 方法比较结果和期望值
    tm.assert_almost_equal(result, rolling_result)


def test_expanding_cov(series):
    # Series A
    A = series
    # 生成一个与 A 相同长度的随机数列 B
    B = (A + np.random.default_rng(2).standard_normal(len(A)))[:-5]

    # 计算 A 和 B 的 expanding 窗口内的协方差
    result = A.expanding().cov(B)

    # 使用 rolling 方法计算 A 和 B 的滚动窗口内的协方差
    rolling_result = A.rolling(window=len(A), min_periods=1).cov(B)

    # 使用 pandas 的 assert_almost_equal 方法比较结果和期望值
    tm.assert_almost_equal(rolling_result, result)


def test_expanding_cov_pairwise(frame):
    # 计算 DataFrame 的 expanding 窗口内的成对协方差
    result = frame.expanding().cov()

    # 使用 rolling 方法计算 DataFrame 的滚动窗口内的成对协方差
    rolling_result = frame.rolling(window=len(frame), min_periods=1).cov()

    # 使用 pandas 的 assert_frame_equal 方法比较结果和期望值
    tm.assert_frame_equal(result, rolling_result)
def test_expanding_corr_pairwise(frame):
    # 计算DataFrame的展开窗口相关系数
    result = frame.expanding().corr()

    # 计算DataFrame的滚动窗口相关系数，窗口大小为DataFrame的长度，最小有效期为1
    rolling_result = frame.rolling(window=len(frame), min_periods=1).corr()
    # 使用断言库比较展开窗口和滚动窗口的相关系数DataFrame是否相等
    tm.assert_frame_equal(result, rolling_result)


@pytest.mark.parametrize(
    "func,static_comp",
    [
        ("sum", lambda x: np.sum(x, axis=0)),  # 求和函数
        ("mean", lambda x: np.mean(x, axis=0)),  # 求平均值函数
        ("max", lambda x: np.max(x, axis=0)),  # 求最大值函数
        ("min", lambda x: np.min(x, axis=0)),  # 求最小值函数
    ],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_func(func, static_comp, frame_or_series):
    # 创建一个包含空值的DataFrame或Series数据
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))

    # 根据数据类型创建展开窗口对象
    obj = data.expanding(min_periods=1)
    # 应用指定的统计函数（如求和、平均值、最大值、最小值）到展开窗口中
    result = getattr(obj, func)()
    # 断言结果是DataFrame或Series类型
    assert isinstance(result, frame_or_series)

    # 计算预期结果
    expected = static_comp(data[:11])
    if frame_or_series is Series:
        # 使用断言库比较展开窗口应用函数后的结果和预期结果的近似性
        tm.assert_almost_equal(result[10], expected)
    else:
        tm.assert_series_equal(result.iloc[10], expected, check_names=False)


@pytest.mark.parametrize(
    "func,static_comp",
    [("sum", np.sum), ("mean", np.mean), ("max", np.max), ("min", np.min)],
    ids=["sum", "mean", "max", "min"],
)
def test_expanding_min_periods(func, static_comp):
    # 创建一个包含随机标准正态分布数据的Series
    ser = Series(np.random.default_rng(2).standard_normal(50))

    # 应用展开窗口到Series，并应用指定的统计函数（如求和、平均值、最大值、最小值）
    result = getattr(ser.expanding(min_periods=30), func)()
    # 断言前30个值是NaN
    assert result[:29].isna().all()
    # 使用断言库比较展开窗口应用函数后的结果和预期结果的近似性
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))

    # 验证min_periods参数的功能
    result = getattr(ser.expanding(min_periods=15), func)()
    assert pd.isna(result.iloc[13])
    assert pd.notna(result.iloc[14])

    # 创建另一个包含随机标准正态分布数据的Series
    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = getattr(ser2.expanding(min_periods=5), func)()
    assert pd.isna(result[3])
    assert pd.notna(result[4])

    # 验证min_periods=0的情况
    result0 = getattr(ser.expanding(min_periods=0), func)()
    result1 = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result0, result1)

    result = getattr(ser.expanding(min_periods=1), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))


def test_expanding_apply(engine_and_raw, frame_or_series):
    engine, raw = engine_and_raw
    # 创建一个包含空值的DataFrame或Series数据
    data = frame_or_series(np.array(list(range(10)) + [np.nan] * 10))
    # 应用展开窗口到数据，并应用指定的函数（这里是求均值）
    result = data.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    # 断言结果是DataFrame或Series类型
    assert isinstance(result, frame_or_series)

    if frame_or_series is Series:
        # 使用断言库比较展开窗口应用函数后的结果和预期结果的近似性
        tm.assert_almost_equal(result[9], np.mean(data[:11], axis=0))
    else:
        tm.assert_series_equal(
            result.iloc[9], np.mean(data[:11], axis=0), check_names=False
        )


def test_expanding_min_periods_apply(engine_and_raw):
    engine, raw = engine_and_raw
    # 创建一个包含随机标准正态分布数据的Series
    ser = Series(np.random.default_rng(2).standard_normal(50))
    # 应用展开窗口到Series，并应用指定的函数（这里是求均值）
    result = ser.expanding(min_periods=30).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    # 断言前30个值是NaN
    assert result[:29].isna().all()
    # 使用断言库比较展开窗口应用函数后的结果和预期结果的近似性
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))
    # 使用 Series 对象的 expanding 方法进行累积计算，应用 lambda 函数计算均值
    result = ser.expanding(min_periods=15).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    # 断言第 14 个元素为空值（NaN）
    assert isna(result.iloc[13])
    # 断言第 15 个元素不为空值（非NaN）
    assert notna(result.iloc[14])

    # 创建一个包含随机标准正态分布数据的 Series 对象
    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    # 使用 expanding 方法进行累积计算，应用 lambda 函数计算均值
    result = ser2.expanding(min_periods=5).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    # 断言第 4 个元素为空值（NaN）
    assert isna(result[3])
    # 断言第 5 个元素不为空值（非NaN）
    assert notna(result[4])

    # 使用 expanding 方法进行累积计算，应用 lambda 函数计算均值，设置 min_periods=0
    result0 = ser.expanding(min_periods=0).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    result1 = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    # 断言累积计算的结果近似相等
    tm.assert_almost_equal(result0, result1)

    # 使用 expanding 方法进行累积计算，应用 lambda 函数计算均值，设置 min_periods=1
    result = ser.expanding(min_periods=1).apply(
        lambda x: x.mean(), raw=raw, engine=engine
    )
    # 断言累积计算的最后一个元素近似等于 ser 的前 50 个元素的均值
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))
# 使用 pytest.mark.parametrize 装饰器，为 test_moment_functions_zero_length_pairwise 函数添加参数化测试
@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.expanding(min_periods=5).cov(x, pairwise=True)),  # 定义 lambda 函数，计算指定列的展开协方差（带对数法）
        lambda x: (x.expanding(min_periods=5).corr(x, pairwise=True)),  # 定义 lambda 函数，计算指定列的展开相关系数（带对数法）
    ],
)
# 定义测试函数 test_moment_functions_zero_length_pairwise，接受一个函数 f 作为参数
def test_moment_functions_zero_length_pairwise(f):
    # 创建空的 DataFrame df1
    df1 = DataFrame()
    # 创建具有特定列和索引的 DataFrame df2
    df2 = DataFrame(columns=Index(["a"], name="foo"), index=Index([], name="bar"))
    # 将 df2 中的列 'a' 转换为 float64 类型
    df2["a"] = df2["a"].astype("float64")

    # 创建预期的 DataFrame df1_expected，索引为 df1 的行列的笛卡尔积
    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    # 创建预期的 DataFrame df2_expected，索引为 df2 的索引和列名为 ['bar', 'foo'] 的笛卡尔积，列为 ['a']，数据类型为 float64
    df2_expected = DataFrame(
        index=MultiIndex.from_product([df2.index, df2.columns], names=["bar", "foo"]),
        columns=Index(["a"], name="foo"),
        dtype="float64",
    )

    # 计算 df1 的结果 df1_result，使用参数化的函数 f 对 df1 进行操作
    df1_result = f(df1)
    # 断言 df1_result 与预期的 df1_expected 相等
    tm.assert_frame_equal(df1_result, df1_expected)

    # 计算 df2 的结果 df2_result，使用参数化的函数 f 对 df2 进行操作
    df2_result = f(df2)
    # 断言 df2_result 与预期的 df2_expected 相等
    tm.assert_frame_equal(df2_result, df2_expected)


# 使用 pytest.mark.parametrize 装饰器，为 test_moment_functions_zero_length 函数添加参数化测试
@pytest.mark.parametrize(
    "f",
    [
        lambda x: x.expanding().count(),  # 计算扩展窗口中的非 NA 值数量
        lambda x: x.expanding(min_periods=5).cov(x, pairwise=False),  # 计算扩展窗口中指定列的展开协方差（不带对数法）
        lambda x: x.expanding(min_periods=5).corr(x, pairwise=False),  # 计算扩展窗口中指定列的展开相关系数（不带对数法）
        lambda x: x.expanding(min_periods=5).max(),  # 计算扩展窗口中指定列的展开最大值
        lambda x: x.expanding(min_periods=5).min(),  # 计算扩展窗口中指定列的展开最小值
        lambda x: x.expanding(min_periods=5).sum(),  # 计算扩展窗口中指定列的展开总和
        lambda x: x.expanding(min_periods=5).mean(),  # 计算扩展窗口中指定列的展开平均值
        lambda x: x.expanding(min_periods=5).std(),  # 计算扩展窗口中指定列的展开标准差
        lambda x: x.expanding(min_periods=5).var(),  # 计算扩展窗口中指定列的展开方差
        lambda x: x.expanding(min_periods=5).skew(),  # 计算扩展窗口中指定列的展开偏度
        lambda x: x.expanding(min_periods=5).kurt(),  # 计算扩展窗口中指定列的展开峰度
        lambda x: x.expanding(min_periods=5).quantile(0.5),  # 计算扩展窗口中指定列的展开分位数（中位数）
        lambda x: x.expanding(min_periods=5).median(),  # 计算扩展窗口中指定列的展开中位数
        lambda x: x.expanding(min_periods=5).apply(sum, raw=False),  # 应用自定义函数 sum 到扩展窗口中指定列的展开数据（非原始数据）
        lambda x: x.expanding(min_periods=5).apply(sum, raw=True),  # 应用自定义函数 sum 到扩展窗口中指定列的展开数据（原始数据）
    ],
)
# 定义测试函数 test_moment_functions_zero_length，接受一个函数 f 作为参数
def test_moment_functions_zero_length(f):
    # GH 8056
    # 创建空的 Series s，数据类型为 np.float64
    s = Series(dtype=np.float64)
    # 预期的 Series s_expected 等于 s
    s_expected = s
    # 创建空的 DataFrame df1
    df1 = DataFrame()
    # 预期的 DataFrame df1_expected 等于 df1
    df1_expected = df1
    # 创建具有列 'a' 的 DataFrame df2
    df2 = DataFrame(columns=["a"])
    # 将 df2 中的列 'a' 转换为 float64 类型
    df2["a"] = df2["a"].astype("float64")
    # 预期的 DataFrame df2_expected 等于 df2
    df2_expected = df2

    # 计算 s 的结果 s_result，使用参数化的函数 f 对 s 进行操作
    s_result = f(s)
    # 断言 s_result 与预期的 s_expected 相等
    tm.assert_series_equal(s_result, s_expected)

    # 计算 df1 的结果 df1_result，使用参数化的函数 f 对 df1 进行操作
    df1_result = f(df1)
    # 断言 df1_result 与预期的 df1_expected 相等
    tm.assert_frame_equal(df1_result, df1_expected)

    # 计算 df2 的结果 df2_result，使用参数化的函数 f 对 df2 进行操作
    df2_result = f(df2)
    # 断言 df2_result 与预期的 df2_expected 相等
    tm.assert_frame_equal(df2_result, df2_expected)


# 定义测试函数 test_expanding_apply_empty_series，接受 engine_and_raw 作为参数
def test_expanding_apply_empty_series(engine_and_raw):
    # 获取 engine_and_raw 中的 engine 和 raw
    engine, raw = engine_and_raw
    # 创建空的 Series ser，数据类型为 np.float64
    ser = Series([], dtype=np.float64)
    # 断言 ser 与 ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine) 的结果相等
    tm.assert_series_equal(
        ser, ser.expanding().apply(lambda x: x.mean(), raw=raw, engine=engine)
    )


# 定义测试函数 test_expanding_apply_min_periods_0，接受 engine_and_raw 作为参数
def test_expanding_apply_min_periods_0(engine_and_raw):
    # GH 8080
    # 获取 engine_and_raw 中的 engine 和 raw
    engine, raw = engine_and_raw
    # 创建包含 None 值的 Series s
    s = Series([None, None, None])
    # 计算 s 的结果 result，使用 s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    result = s.expanding(min_periods=0).apply(lambda x: len(x), raw=raw, engine=engine)
    # 预期的 Series expected 等于 Series([1.0, 2.0, 3.0])
    expected = Series([1.0, 2.0, 3.0])
    # 断言 result 与 expected 相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_expanding_cov_diff_index
def test_expanding_cov_diff_index():
    # GH 7512
    # 创建 Series s1，索引为 [0, 1, 2]，值为 [1, 2, 3]
    s1 = Series([1, 2, 3], index=[0,
    # 使用测试工具库中的函数检查两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 创建一个包含 NaN 的 Series 对象 s2a，指定索引
    s2a = Series([1, None, 3], index=[0, 1, 2])
    # 计算 s1 和 s2a 的扩展窗口协方差
    result = s1.expanding().cov(s2a)
    # 使用测试工具库中的函数检查计算结果是否符合预期
    tm.assert_series_equal(result, expected)

    # 创建两个 Series 对象 s1 和 s2，指定索引
    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    # 计算 s1 和 s2 的扩展窗口协方差
    result = s1.expanding().cov(s2)
    # 创建预期的 Series 对象，包含预期结果
    expected = Series([None, None, None, 4.5])
    # 使用测试工具库中的函数检查计算结果是否符合预期
    tm.assert_series_equal(result, expected)
def test_expanding_corr_diff_index():
    # GH 7512
    # 创建包含值和索引的 Series s1 和 s2
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    # 使用 expanding() 方法计算 s1 和 s2 的相关性
    result = s1.expanding().corr(s2)
    # 创建期望的 Series 对象 expected，包含预期的相关性值
    expected = Series([None, None, 1.0])
    # 使用 assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 修改 s2 使其包含缺失值，再次计算相关性
    s2a = Series([1, None, 3], index=[0, 1, 2])
    result = s1.expanding().corr(s2a)
    # 使用 assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 修改 s1 和 s2 的值和索引，再次计算相关性
    s1 = Series([7, 8, 10], index=[0, 1, 3])
    s2 = Series([7, 9, 10], index=[0, 2, 3])
    result = s1.expanding().corr(s2)
    # 创建期望的 Series 对象 expected，包含预期的相关性值
    expected = Series([None, None, None, 1.0])
    # 使用 assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_expanding_cov_pairwise_diff_length():
    # GH 7512
    # 创建包含值和列名的 DataFrame df1, df1a, df2, df2a
    df1 = DataFrame([[1, 5], [3, 2], [3, 9]], columns=Index(["A", "B"], name="foo"))
    df1a = DataFrame(
        [[1, 5], [3, 9]], index=[0, 2], columns=Index(["A", "B"], name="foo")
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]], columns=Index(["X", "Y"], name="foo")
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=[0, 2], columns=Index(["X", "Y"], name="foo")
    )
    # xref gh-15826
    # 使用 expanding() 方法计算 df1 和 df2 的协方差，保留列名，按对计算
    result1 = df1.expanding().cov(df2, pairwise=True).loc[2]
    result2 = df1.expanding().cov(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().cov(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().cov(df2a, pairwise=True).loc[2]
    # 创建期望的 DataFrame 对象 expected，包含预期的协方差值
    expected = DataFrame(
        [[-3.0, -6.0], [-5.0, -10.0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(["X", "Y"], name="foo"),
    )
    # 使用 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_corr_pairwise_diff_length():
    # GH 7512
    # 创建包含值和列名的 DataFrame df1, df1a, df2, df2a
    df1 = DataFrame(
        [[1, 2], [3, 2], [3, 4]], columns=["A", "B"], index=Index(range(3), name="bar")
    )
    df1a = DataFrame(
        [[1, 2], [3, 4]], index=Index([0, 2], name="bar"), columns=["A", "B"]
    )
    df2 = DataFrame(
        [[5, 6], [None, None], [2, 1]],
        columns=["X", "Y"],
        index=Index(range(3), name="bar"),
    )
    df2a = DataFrame(
        [[5, 6], [2, 1]], index=Index([0, 2], name="bar"), columns=["X", "Y"]
    )
    # 使用 expanding() 方法计算 df1 和 df2 的相关性，保留列名，按对计算
    result1 = df1.expanding().corr(df2, pairwise=True).loc[2]
    result2 = df1.expanding().corr(df2a, pairwise=True).loc[2]
    result3 = df1a.expanding().corr(df2, pairwise=True).loc[2]
    result4 = df1a.expanding().corr(df2a, pairwise=True).loc[2]
    # 创建期望的 DataFrame 对象 expected，包含预期的相关性值
    expected = DataFrame(
        [[-1.0, -1.0], [-1.0, -1.0]], columns=["A", "B"], index=Index(["X", "Y"])
    )
    # 使用 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result1, expected)
    tm.assert_frame_equal(result2, expected)
    tm.assert_frame_equal(result3, expected)
    tm.assert_frame_equal(result4, expected)


def test_expanding_apply_args_kwargs(engine_and_raw):
    def mean_w_arg(x, const):
        return np.mean(x) + const

    engine, raw = engine_and_raw
    # 创建一个包含随机数据的 DataFrame，形状为 (20, 3)
    df = DataFrame(np.random.default_rng(2).random((20, 3)))
    
    # 使用扩展窗口函数计算 DataFrame 的展开均值，指定计算引擎和原始数据选项，然后加上 20.0
    expected = df.expanding().apply(np.mean, engine=engine, raw=raw) + 20.0
    
    # 使用自定义函数 mean_w_arg 对扩展窗口中的 DataFrame 应用，并传入额外的参数 20
    result = df.expanding().apply(mean_w_arg, engine=engine, raw=raw, args=(20,))
    # 使用断言确保 result 与 expected 的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
    
    # 再次使用自定义函数 mean_w_arg 对扩展窗口中的 DataFrame 应用，这次使用 kwargs 传递常量参数为 20
    result = df.expanding().apply(mean_w_arg, raw=raw, kwargs={"const": 20})
    # 使用断言确保 result 与 expected 的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
# 定义测试函数，用于测试在特定的算术窗口操作下，DataFrame 或 Series 对象的扩展方法
def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # GH#46560
    # 使用传入的算术窗口操作符作为内核
    kernel = arithmetic_win_operators
    # 创建一个 DataFrame 对象，包含列 'a', 'b', 'c'，其中 'c' 的类型被强制转换为对象类型
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    # 使用 expanding() 方法创建一个扩展对象
    expanding = df.expanding()
    # 获取扩展对象的特定内核方法，如果不存在则为 None
    op = getattr(expanding, kernel, None)
    if op is not None:
        # 执行特定内核方法，根据 numeric_only 参数选择是否只考虑数值列
        result = op(numeric_only=numeric_only)

        # 根据 numeric_only 参数选择不同的列进行聚合操作，并将结果转换为 float 类型的 DataFrame
        columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        # 断言预期的结果列名与选择的列名相符
        assert list(expected.columns) == columns

        # 断言结果与预期结果的 DataFrame 相等
        tm.assert_frame_equal(result, expected)


# 使用参数化标记指定不同的内核和使用参数
@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # GH#46560
    # 创建一个 DataFrame 对象，包含列 'a', 'b', 'c'，其中 'c' 的类型被强制转换为对象类型
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    df["c"] = df["c"].astype(object)
    # 根据 use_arg 参数选择是否将 df 包装为元组
    arg = (df,) if use_arg else ()
    # 使用 expanding() 方法创建一个扩展对象
    expanding = df.expanding()
    # 获取扩展对象的特定内核方法
    op = getattr(expanding, kernel)
    # 执行特定内核方法，根据 numeric_only 参数选择是否只考虑数值列
    result = op(*arg, numeric_only=numeric_only)

    # 根据 numeric_only 参数选择不同的列进行浮点类型转换，并构建预期结果
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    df2 = df[columns].astype(float)
    arg2 = (df2,) if use_arg else ()
    expanding2 = df2.expanding()
    op2 = getattr(expanding2, kernel)
    expected = op2(*arg2, numeric_only=numeric_only)

    # 断言结果与预期结果的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 使用参数化标记指定不同的数据类型
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # GH#46560
    # 使用传入的算术窗口操作符作为内核
    kernel = arithmetic_win_operators
    # 创建一个 Series 对象，包含一个元素为 1 的列表，并指定数据类型为 dtype
    ser = Series([1], dtype=dtype)
    # 使用 expanding() 方法创建一个扩展对象
    expanding = ser.expanding()
    # 获取扩展对象的特定内核方法
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        # 如果 numeric_only 为 True 且数据类型为对象，则预期会抛出 NotImplementedError 异常
        msg = f"Expanding.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        # 否则执行特定内核方法，根据 numeric_only 参数选择是否只考虑数值列
        result = op(numeric_only=numeric_only)
        # 根据特定内核方法聚合 Series，并将结果转换为 float 类型的 Series
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        # 断言结果与预期结果的 Series 相等
        tm.assert_series_equal(result, expected)


# 使用参数化标记指定不同的内核、使用参数和数据类型
@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # GH#46560
    # 创建一个 Series 对象，包含元素为 1, 2, 3 的列表，并指定数据类型为 dtype
    ser = Series([1, 2, 3], dtype=dtype)
    # 根据 use_arg 参数选择是否将 ser 包装为元组
    arg = (ser,) if use_arg else ()
    # 使用 expanding() 方法创建一个扩展对象
    expanding = ser.expanding()
    # 获取扩展对象的特定内核方法
    op = getattr(expanding, kernel)
    if numeric_only and dtype is object:
        # 如果 numeric_only 为 True 且数据类型为对象，则预期会抛出 NotImplementedError 异常
        msg = f"Expanding.{kernel} does not implement numeric_only"
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        # 对传入的操作 op 进行调用，并将结果存储在 result 中
        result = op(*arg, numeric_only=numeric_only)

        # 将序列 ser 转换为浮点类型的序列 ser2
        ser2 = ser.astype(float)
        
        # 如果 use_arg 为 True，则将 ser2 封装为一个元组；否则为空元组
        arg2 = (ser2,) if use_arg else ()
        
        # 创建一个扩展对象 expanding2，用于执行扩展操作
        expanding2 = ser2.expanding()
        
        # 根据 kernel 获取 expanding2 对象的相应操作函数 op2
        op2 = getattr(expanding2, kernel)
        
        # 调用 op2 执行操作，传入参数 arg2，并将结果存储在 expected 中
        expected = op2(*arg2, numeric_only=numeric_only)
        
        # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
```