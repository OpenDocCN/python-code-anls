# `D:\src\scipysrc\pandas\pandas\tests\window\test_ewm.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入以下模块和函数
    DataFrame,  # 数据帧对象，用于存储和操作二维数据
    DatetimeIndex,  # 日期时间索引对象
    Series,  # 系列对象，一维标签数组，支持多种数据类型
    date_range,  # 创建日期范围
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

def test_doc_string():
    df = DataFrame({"B": [0, 1, 2, np.nan, 4]})
    df  # 创建一个包含 NaN 值的数据帧
    df.ewm(com=0.5).mean()  # 对数据帧进行指数加权移动平均计算

def test_constructor(frame_or_series):
    c = frame_or_series(range(5)).ewm  # 使用给定的数据创建 ewm 对象

    # valid
    c(com=0.5)  # 设置 com 参数进行有效计算
    c(span=1.5)  # 设置 span 参数进行有效计算
    c(alpha=0.5)  # 设置 alpha 参数进行有效计算
    c(halflife=0.75)  # 设置 halflife 参数进行有效计算
    c(com=0.5, span=None)  # 设置多个参数进行有效计算
    c(alpha=0.5, com=None)  # 设置多个参数进行有效计算
    c(halflife=0.75, alpha=None)  # 设置多个参数进行有效计算

    # not valid: mutually exclusive
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):  # 检查互斥参数组合的有效性
        c(com=0.5, alpha=0.5)
    with pytest.raises(ValueError, match=msg):  # 检查互斥参数组合的有效性
        c(span=1.5, halflife=0.75)
    with pytest.raises(ValueError, match=msg):  # 检查互斥参数组合的有效性
        c(alpha=0.5, span=1.5)

    # not valid: com < 0
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):  # 检查 com 参数小于零的情况
        c(com=-0.5)

    # not valid: span < 1
    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):  # 检查 span 参数小于一的情况
        c(span=0.5)

    # not valid: halflife <= 0
    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):  # 检查 halflife 参数小于或等于零的情况
        c(halflife=0)

    # not valid: alpha <= 0 or alpha > 1
    msg = "alpha must satisfy: 0 < alpha <= 1"
    for alpha in (-0.5, 1.5):  # 遍历不合法的 alpha 参数范围
        with pytest.raises(ValueError, match=msg):  # 检查 alpha 参数不合法的情况
            c(alpha=alpha)

def test_ewma_times_not_datetime_type():
    msg = r"times must be datetime64 dtype."
    with pytest.raises(ValueError, match=msg):  # 检查 times 参数不是 datetime64 数据类型的情况
        Series(range(5)).ewm(times=np.arange(5))

def test_ewma_times_not_same_length():
    msg = "times must be the same length as the object."
    with pytest.raises(ValueError, match=msg):  # 检查 times 参数长度与对象不匹配的情况
        Series(range(5)).ewm(times=np.arange(4).astype("datetime64[ns]"))

def test_ewma_halflife_not_correct_type():
    msg = "halflife must be a timedelta convertible object"
    with pytest.raises(ValueError, match=msg):  # 检查 halflife 参数类型不正确的情况
        Series(range(5)).ewm(halflife=1, times=np.arange(5).astype("datetime64[ns]"))

def test_ewma_halflife_without_times(halflife_with_times):
    msg = "halflife can only be a timedelta convertible argument if times is not None."
    with pytest.raises(ValueError, match=msg):  # 检查没有 times 参数时设置 halflife 的情况
        Series(range(5)).ewm(halflife=halflife_with_times)

@pytest.mark.parametrize(
    "times",
    [
        np.arange(10).astype("datetime64[D]").astype("datetime64[ns]"),  # 不同格式的日期时间对象
        date_range("2000", freq="D", periods=10),  # 生成日期范围
        date_range("2000", freq="D", periods=10).tz_localize("UTC"),  # 生成带时区信息的日期范围
    ],
)
@pytest.mark.parametrize("min_periods", [0, 2])
def test_ewma_with_times_equal_spacing(halflife_with_times, times, min_periods):
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    df = DataFrame({"A": data})  # 创建包含 NaN 值的数据帧
    result = df.ewm(halflife=halflife, min_periods=min_periods, times=times).mean()  # 对数据帧应用 ewm 方法进行计算
    # 使用指数加权移动平均（Exponential Weighted Moving Average, EWMA）计算数据框架 df 的期望值，
    # 使用半衰期为 1.0，并指定最小期数为 min_periods。
    expected = df.ewm(halflife=1.0, min_periods=min_periods).mean()
    
    # 使用测试框架中的 assert_frame_equal 函数比较计算得到的结果 result 和期望值 expected，
    # 确保它们在内容上完全相等。
    tm.assert_frame_equal(result, expected)
# 定义测试函数，用于测试带有可变间隔时间的指数加权移动平均（EWMA）
def test_ewma_with_times_variable_spacing(tz_aware_fixture, unit):
    # 从测试夹具中获取时区信息
    tz = tz_aware_fixture
    # 设置半衰期为 "23 days"
    halflife = "23 days"
    # 创建一个包含三个日期时间的时间索引，将其本地化为指定时区，并按指定单位进行调整
    times = (
        DatetimeIndex(["2020-01-01", "2020-01-10T00:04:05", "2020-02-23T05:00:23"])
        .tz_localize(tz)
        .as_unit(unit)
    )
    # 创建一个包含三个元素的 NumPy 数组
    data = np.arange(3)
    # 使用 NumPy 数组创建一个 DataFrame
    df = DataFrame(data)
    # 计算 DataFrame 的指数加权移动平均，使用指定的半衰期和时间点
    result = df.ewm(halflife=halflife, times=times).mean()
    # 创建一个预期的 DataFrame，包含预期的结果
    expected = DataFrame([0.0, 0.5674161888241773, 1.545239952073459])
    # 断言计算结果与预期结果相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，测试在存在 NaT 值时 EWMA 是否会引发异常
def test_ewm_with_nat_raises(halflife_with_times):
    # GH#38535
    # 创建一个包含单个元素的 Series
    ser = Series(range(1))
    # 创建一个包含单个 NaT 值的时间索引
    times = DatetimeIndex(["NaT"])
    # 使用 pytest 断言检查调用 EWMA 时是否会引发 ValueError，并检查异常信息
    with pytest.raises(ValueError, match="Cannot convert NaT values to integer"):
        ser.ewm(com=0.1, halflife=halflife_with_times, times=times)


# 定义测试函数，测试在 EWMA 中对 times 执行索引操作
def test_ewm_with_times_getitem(halflife_with_times):
    # GH 40164
    # 将 halflife 设置为传入的 halflife_with_times 值
    halflife = halflife_with_times
    # 创建一个包含 10 个浮点数的 NumPy 数组，并将其中的偶数索引位置设为 NaN
    data = np.arange(10.0)
    data[::2] = np.nan
    # 创建一个包含 10 个日期的时间索引
    times = date_range("2000", freq="D", periods=10)
    # 使用包含两列的 DataFrame 创建对象 df
    df = DataFrame({"A": data, "B": data})
    # 对 df 进行指数加权移动平均，针对列"A"，并获取其均值
    result = df.ewm(halflife=halflife, times=times)["A"].mean()
    # 计算全局 halflife 为 1.0 时，列"A"的均值
    expected = df.ewm(halflife=1.0)["A"].mean()
    # 断言结果与预期结果相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的参数化装饰器定义测试函数，测试 EWMA 对象属性在索引时是否被保留
@pytest.mark.parametrize("arg", ["com", "halflife", "span", "alpha"])
def test_ewm_getitem_attributes_retained(arg, adjust, ignore_na):
    # GH 40164
    # 准备参数字典，其中包含一个属性和两个附加参数
    kwargs = {arg: 1, "adjust": adjust, "ignore_na": ignore_na}
    # 创建一个包含两列的 DataFrame，对其进行 EWMA 操作
    ewm = DataFrame({"A": range(1), "B": range(1)}).ewm(**kwargs)
    # 创建一个期望的属性字典，包含 EWMA 对象的全部属性
    expected = {attr: getattr(ewm, attr) for attr in ewm._attributes}
    # 对 EWMA 对象的某列进行索引操作，获取对应的新 EWMA 对象
    ewm_slice = ewm["A"]
    # 创建一个结果字典，包含 ewm_slice 对象的全部属性
    result = {attr: getattr(ewm, attr) for attr in ewm_slice._attributes}
    # 使用断言检查结果字典与预期字典是否相等
    assert result == expected


# 定义测试函数，测试在 adjust=False 时是否引发异常
def test_ewma_times_adjust_false_raises():
    # GH 40098
    # 使用 pytest 断言检查在 adjust=False 时是否会引发 NotImplementedError，检查异常信息
    with pytest.raises(
        NotImplementedError, match="times is not supported with adjust=False."
    ):
        Series(range(1)).ewm(
            0.1, adjust=False, times=date_range("2000", freq="D", periods=1)
        )


# 定义测试函数，测试在 times 列为字符串类型时是否引发异常
def test_times_string_col_raises():
    # GH 43265
    # 创建一个包含两列的 DataFrame，其中一列是时间序列
    df = DataFrame(
        {"A": np.arange(10.0), "time_col": date_range("2000", freq="D", periods=10)}
    )
    # 使用 pytest 断言检查在 times 列为字符串类型时是否会引发 ValueError，检查异常信息
    with pytest.raises(ValueError, match="times must be datetime64"):
        df.ewm(halflife="1 day", min_periods=0, times="time_col")


# 定义测试函数，测试在 adjust=False 时调用 sum() 方法是否引发异常
def test_ewm_sum_adjust_false_notimplemented():
    # 创建一个 Series 对象，并对其进行 EWMA 操作，设置 adjust=False
    data = Series(range(1)).ewm(com=1, adjust=False)
    # 使用 pytest 断言检查调用 sum() 方法是否会引发 NotImplementedError，检查异常信息
    with pytest.raises(NotImplementedError, match="sum is not"):
        data.sum()


# 使用 pytest 的参数化装饰器定义测试函数，测试在 times 参数存在时，部分方法是否实现
@pytest.mark.parametrize("method", ["sum", "std", "var", "cov", "corr"])
def test_times_only_mean_implemented(frame_or_series, method):
    # GH 51695
    # 设置半衰期为 "1 day"
    halflife = "1 day"
    # 创建一个包含 10 个日期的时间索引
    times = date_range("2000", freq="D", periods=10)
    # 对 frame_or_series 中的数据进行 EWMA 操作，使用指定的半衰期和时间点
    ewm = frame_or_series(range(10)).ewm(halflife=halflife, times=times)
    # 使用 pytest 断言检查部分方法（如 sum、std、var 等）是否在使用 times 参数时抛出 NotImplementedError
    with pytest.raises(
        NotImplementedError, match=f"{method} is not implemented with times"
    ):
        getattr(ewm, method)()
    # 定义一个嵌套的列表，每个子列表包含两部分数据：一个包含数值的列表和一个布尔值
    [[[10.0, 5.0, 2.5, 11.25], False], [[10.0, 5.0, 5.0, 12.5], True]],
def test_ewm_sum(expected_data, ignore):
    # xref from Numbagg tests
    # https://github.com/numbagg/numbagg/blob/v0.2.1/numbagg/test/test_moving.py#L50
    # 创建一个包含一些数值和 NaN 值的 Series 对象
    data = Series([10, 0, np.nan, 10])
    # 对 Series 应用指数加权移动平均，并求和
    result = data.ewm(alpha=0.5, ignore_na=ignore).sum()
    # 创建一个预期的 Series 对象，用于比较结果
    expected = Series(expected_data)
    # 使用测试模块中的函数验证结果与预期是否一致
    tm.assert_series_equal(result, expected)


def test_ewma_adjust():
    # 创建一个包含 1000 个零的 Series 对象，并在索引 5 处设置值为 1
    vals = Series(np.zeros(1000))
    vals[5] = 1
    # 对 Series 应用指数加权移动平均，设置调整参数为 False，并求均值后求和
    result = vals.ewm(span=100, adjust=False).mean().sum()
    # 使用断言验证结果是否接近 1
    assert np.abs(result - 1) < 1e-2


def test_ewma_cases(adjust, ignore_na):
    # 尝试不同的调整和忽略 NaN 参数组合

    # 创建一个包含一些数值的 Series 对象
    s = Series([1.0, 2.0, 4.0, 8.0])

    if adjust:
        # 如果调整参数为 True，则预期结果是调整后的值
        expected = Series([1.0, 1.6, 2.736842, 4.923077])
    else:
        # 如果调整参数为 False，则预期结果是未调整的值
        expected = Series([1.0, 1.333333, 2.222222, 4.148148])

    # 对 Series 应用指数加权移动平均，并求均值
    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()
    # 使用测试模块中的函数验证结果与预期是否一致
    tm.assert_series_equal(result, expected)


def test_ewma_nan_handling():
    # 创建一个包含 NaN 值的 Series 对象，并对其应用指数加权移动平均
    s = Series([1.0] + [np.nan] * 5 + [1.0])
    result = s.ewm(com=5).mean()
    # 使用测试模块中的函数验证结果与预期是否一致
    tm.assert_series_equal(result, Series([1.0] * len(s)))

    # 创建一个包含 NaN 值的 Series 对象，并对其应用指数加权移动平均
    s = Series([np.nan] * 2 + [1.0] + [np.nan] * 2 + [1.0])
    result = s.ewm(com=5).mean()
    # 使用测试模块中的函数验证结果与预期是否一致
    tm.assert_series_equal(result, Series([np.nan] * 2 + [1.0] * 4))


@pytest.mark.parametrize(
    "s, adjust, ignore_na, w",
    ],
)
def test_ewma_nan_handling_cases(s, adjust, ignore_na, w):
    # GH 7603
    # 将输入的 s 转换为 Series 对象
    s = Series(s)
    # 计算预期的加权平均值
    expected = (s.multiply(w).cumsum() / Series(w).cumsum()).ffill()
    # 对 Series 应用指数加权移动平均，并求均值
    result = s.ewm(com=2.0, adjust=adjust, ignore_na=ignore_na).mean()

    # 使用测试模块中的函数验证结果与预期是否一致
    tm.assert_series_equal(result, expected)
    if ignore_na is False:
        # 如果忽略 NaN 参数设置为 False，则再次验证默认情况下的行为
        result = s.ewm(com=2.0, adjust=adjust).mean()
        tm.assert_series_equal(result, expected)


def test_ewm_alpha():
    # GH 10789
    # 创建一个随机数组，包含 NaN 值
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan

    # 创建一个包含随机数组的 Series 对象
    s = Series(arr)
    # 分别使用 alpha、com、span 和 halflife 参数应用指数加权移动平均，并求均值
    a = s.ewm(alpha=0.61722699889169674).mean()
    b = s.ewm(com=0.62014947789973052).mean()
    c = s.ewm(span=2.240298955799461).mean()
    d = s.ewm(halflife=0.721792864318).mean()
    # 使用测试模块中的函数验证所有结果是否一致
    tm.assert_series_equal(a, b)
    tm.assert_series_equal(a, c)
    tm.assert_series_equal(a, d)


def test_ewm_domain_checks():
    # GH 12492
    # 创建一个包含随机数组的 Series 对象，包含 NaN 值
    arr = np.random.default_rng(2).standard_normal(100)
    locs = np.arange(20, 40)
    arr[locs] = np.nan

    # 创建一个包含随机数组的 Series 对象
    s = Series(arr)
    
    # 验证 com 参数的域检查，确保抛出异常
    msg = "comass must satisfy: comass >= 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(com=-0.1)
    s.ewm(com=0.0)
    s.ewm(com=0.1)

    # 验证 span 参数的域检查，确保抛出异常
    msg = "span must satisfy: span >= 1"
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=-0.1)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.0)
    with pytest.raises(ValueError, match=msg):
        s.ewm(span=0.9)
    s.ewm(span=1.0)
    s.ewm(span=1.1)

    # 验证 halflife 参数的域检查，确保抛出异常
    msg = "halflife must satisfy: halflife > 0"
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=-0.1)
    # 使用 pytest 来测试抛出 ValueError 异常，验证 halflife 参数为 0.0 时的情况
    with pytest.raises(ValueError, match=msg):
        s.ewm(halflife=0.0)
    # 调用 ewm 方法，设置 halflife 参数为 0.1
    s.ewm(halflife=0.1)

    # 设置错误信息字符串，用于验证 alpha 参数不在有效范围内时抛出 ValueError 异常
    msg = "alpha must satisfy: 0 < alpha <= 1"
    # 使用 pytest 来测试抛出 ValueError 异常，验证 alpha 参数为负数时的情况
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=-0.1)
    # 使用 pytest 来测试抛出 ValueError 异常，验证 alpha 参数为 0.0 时的情况
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=0.0)
    # 调用 ewm 方法，设置 alpha 参数为 0.1，位于有效范围内
    s.ewm(alpha=0.1)
    # 调用 ewm 方法，设置 alpha 参数为 1.0，位于有效范围内
    s.ewm(alpha=1.0)
    # 使用 pytest 来测试抛出 ValueError 异常，验证 alpha 参数为超过 1.0 时的情况
    with pytest.raises(ValueError, match=msg):
        s.ewm(alpha=1.1)
@pytest.mark.parametrize("method", ["mean", "std", "var"])
def test_ew_empty_series(method):
    # 创建一个空的 Series，数据类型为 np.float64
    vals = Series([], dtype=np.float64)

    # 计算指数加权移动平均 (EWM) 对象
    ewm = vals.ewm(3)
    # 调用指定方法（mean, std, var）计算 EWM 的结果
    result = getattr(ewm, method)()
    # 断言计算结果与原始 Series 相近
    tm.assert_almost_equal(result, vals)


@pytest.mark.parametrize("min_periods", [0, 1])
@pytest.mark.parametrize("name", ["mean", "var", "std"])
def test_ew_min_periods(min_periods, name):
    # 生成一个包含随机数据的 Series，并设置部分数据为 NaN
    arr = np.random.default_rng(2).standard_normal(50)
    arr[:10] = np.nan
    arr[-10:] = np.nan
    s = Series(arr)

    # 检查 min_periods 参数设置是否正确排除 NaN 值
    result = getattr(s.ewm(com=50, min_periods=2), name)()
    assert result[:11].isna().all()
    assert not result[11:].isna().any()

    result = getattr(s.ewm(com=50, min_periods=min_periods), name)()
    if name == "mean":
        assert result[:10].isna().all()
        assert not result[10:].isna().any()
    else:
        # 对于 ewm.std 和 ewm.var（使用 bias=False），至少需要两个值
        assert result[:11].isna().all()
        assert not result[11:].isna().any()

    # 检查长度为 0 的 Series
    result = getattr(Series(dtype=object).ewm(com=50, min_periods=min_periods), name)()
    tm.assert_series_equal(result, Series(dtype="float64"))

    # 检查长度为 1 的 Series
    result = getattr(Series([1.0]).ewm(50, min_periods=min_periods), name)()
    if name == "mean":
        tm.assert_series_equal(result, Series([1.0]))
    else:
        # 对于 ewm.std 和 ewm.var（使用 bias=False），至少需要两个值
        tm.assert_series_equal(result, Series([np.nan]))

    # 传入整数作为参数
    result2 = getattr(Series(np.arange(50)).ewm(span=10), name)()
    assert result2.dtype == np.float64


@pytest.mark.parametrize("name", ["cov", "corr"])
def test_ewm_corr_cov(name):
    # 生成两个随机 Series，用于计算相关性和协方差
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)

    A[:10] = np.nan
    B.iloc[-10:] = np.nan

    # 计算指数加权移动相关性或协方差
    result = getattr(A.ewm(com=20, min_periods=5), name)(B)
    assert np.isnan(result.values[:14]).all()
    assert not np.isnan(result.values[14:]).any()


@pytest.mark.parametrize("min_periods", [0, 1, 2])
@pytest.mark.parametrize("name", ["cov", "corr"])
def test_ewm_corr_cov_min_periods(name, min_periods):
    # 生成两个随机 Series，用于计算相关性和协方差
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    B = A[2:] + np.random.default_rng(2).standard_normal(48)

    A[:10] = np.nan
    B.iloc[-10:] = np.nan

    # 计算指数加权移动相关性或协方差，根据 min_periods 参数检查结果
    assert np.isnan(result.values[:11]).all()
    assert not np.isnan(result.values[11:]).any()

    # 检查长度为 0 的 Series
    empty = Series([], dtype=np.float64)
    result = getattr(empty.ewm(com=50, min_periods=min_periods), name)(empty)
    tm.assert_series_equal(result, empty)

    # 检查长度为 1 的 Series
    # 这部分代码还待完成，需补充完整的测试用例和断言
    # 使用getattr函数从Series对象中获取指定名称的属性或方法，并传入参数Series([1.0]).ewm(com=50, min_periods=min_periods)
    result = getattr(Series([1.0]).ewm(com=50, min_periods=min_periods), name)(
        Series([1.0])
    )
    # 调用tm.assert_series_equal函数，比较result与Series([np.nan])是否相等
    tm.assert_series_equal(result, Series([np.nan]))
@pytest.mark.parametrize("name", ["cov", "corr"])
# 参数化测试，name 可以是 "cov" 或 "corr"
def test_different_input_array_raise_exception(name):
    # 创建一个 Series 对象 A，其中包含50个标准正态分布的随机数，前10个设置为 NaN
    A = Series(np.random.default_rng(2).standard_normal(50), index=range(50))
    A[:10] = np.nan

    msg = "other must be a DataFrame or Series"
    # 检查是否会引发 ValueError 异常，异常消息应为 "other must be a DataFrame or Series"
    with pytest.raises(ValueError, match=msg):
        # 调用 A 对象的 ewm(com=20, min_periods=5) 方法，并对其结果调用 name 参数指定的方法
        getattr(A.ewm(com=20, min_periods=5), name)(
            np.random.default_rng(2).standard_normal(50)
        )


@pytest.mark.parametrize("name", ["var", "std", "mean"])
# 参数化测试，name 可以是 "var", "std", 或 "mean"
def test_ewma_series(series, name):
    # 调用 series 对象的 ewm(com=10) 方法，并对其结果调用 name 参数指定的方法
    series_result = getattr(series.ewm(com=10), name)()
    # 断言 series_result 的类型为 Series
    assert isinstance(series_result, Series)


@pytest.mark.parametrize("name", ["var", "std", "mean"])
# 参数化测试，name 可以是 "var", "std", 或 "mean"
def test_ewma_frame(frame, name):
    # 调用 frame 对象的 ewm(com=10) 方法，并对其结果调用 name 参数指定的方法
    frame_result = getattr(frame.ewm(com=10), name)()
    # 断言 frame_result 的类型为 DataFrame
    assert isinstance(frame_result, DataFrame)


def test_ewma_span_com_args(series):
    # 调用 series 对象的 ewm(com=9.5) 方法，并计算其均值
    A = series.ewm(com=9.5).mean()
    # 调用 series 对象的 ewm(span=20) 方法，并计算其均值
    B = series.ewm(span=20).mean()
    # 使用 pytest 的 assert_almost_equal 函数比较 A 和 B 的值
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    # 检查是否会引发 ValueError 异常，异常消息应为 "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(com=9.5, span=20) 方法
        series.ewm(com=9.5, span=20)

    msg = "Must pass one of comass, span, halflife, or alpha"
    # 检查是否会引发 ValueError 异常，异常消息应为 "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm() 方法
        series.ewm().mean()


def test_ewma_halflife_arg(series):
    # 调用 series 对象的 ewm(com=13.932726172912965) 方法，并计算其均值
    A = series.ewm(com=13.932726172912965).mean()
    # 调用 series 对象的 ewm(halflife=10.0) 方法，并计算其均值
    B = series.ewm(halflife=10.0).mean()
    # 使用 pytest 的 assert_almost_equal 函数比较 A 和 B 的值
    tm.assert_almost_equal(A, B)
    msg = "comass, span, halflife, and alpha are mutually exclusive"
    # 检查是否会引发 ValueError 异常，异常消息应为 "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(span=20, halflife=50) 方法
        series.ewm(span=20, halflife=50)
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(com=9.5, halflife=50) 方法
        series.ewm(com=9.5, halflife=50)
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(com=9.5, span=20, halflife=50) 方法
        series.ewm(com=9.5, span=20, halflife=50)
    msg = "Must pass one of comass, span, halflife, or alpha"
    # 检查是否会引发 ValueError 异常，异常消息应为 "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm() 方法
        series.ewm()


def test_ewm_alpha_arg(series):
    # GH 10789
    s = series
    msg = "Must pass one of comass, span, halflife, or alpha"
    # 检查是否会引发 ValueError 异常，异常消息应为 "Must pass one of comass, span, halflife, or alpha"
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm() 方法
        s.ewm()

    msg = "comass, span, halflife, and alpha are mutually exclusive"
    # 检查是否会引发 ValueError 异常，异常消息应为 "comass, span, halflife, and alpha are mutually exclusive"
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(com=10.0, alpha=0.5) 方法
        s.ewm(com=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(span=10.0, alpha=0.5) 方法
        s.ewm(span=10.0, alpha=0.5)
    with pytest.raises(ValueError, match=msg):
        # 调用 series 对象的 ewm(halflife=10.0, alpha=0.5) 方法
        s.ewm(halflife=10.0, alpha=0.5)


@pytest.mark.parametrize("func", ["cov", "corr"])
# 参数化测试，func 可以是 "cov" 或 "corr"
def test_ewm_pairwise_cov_corr(func, frame):
    # 调用 frame 对象的 ewm(span=10, min_periods=5) 方法，并计算其 func 参数指定的函数
    result = getattr(frame.ewm(span=10, min_periods=5), func)()
    # 选择 result 中第1列和第5行的数据
    result = result.loc[(slice(None), 1), 5]
    # 删除 result 索引的第二级
    result.index = result.index.droplevel(1)
    # 调用 frame 对象的 ewm(span=10, min_periods=5) 方法，并计算 func 参数指定的函数，传入 frame[5] 作为参数
    expected = getattr(frame[1].ewm(span=10, min_periods=5), func)(frame[5])
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 的值，不检查名称
    tm.assert_series_equal(result, expected, check_names=False)


def test_numeric_only_frame(arithmetic_win_operators, numeric_only):
    # GH#46560
    # kernel 变量赋值为 arithmetic_win_operators
    kernel = arithmetic_win_operators
    # 创建一个 DataFrame 对象 df，包含列 "a", "b", "c"，分别对应值 [1], 2, 3
    df = DataFrame({"a": [1], "b": 2, "c": 3})
    # 将数据框 df 中的 "c" 列转换为对象类型
    df["c"] = df["c"].astype(object)
    
    # 使用指数加权移动平均（Exponential Weighted Moving Average, EWM）方法，设置 span=2 和最小周期数为1
    ewm = df.ewm(span=2, min_periods=1)
    
    # 获取指定的核函数（kernel），如 mean、std 等
    op = getattr(ewm, kernel, None)
    
    # 如果找到了指定的核函数
    if op is not None:
        # 对数据应用该核函数，返回结果
        result = op(numeric_only=numeric_only)
    
        # 根据 numeric_only 确定要使用的列
        columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
        
        # 计算预期的聚合结果，以核函数为聚合函数，然后重置索引并转换为浮点类型
        expected = df[columns].agg([kernel]).reset_index(drop=True).astype(float)
        
        # 断言预期的结果列与实际计算的列相同
        assert list(expected.columns) == columns
    
        # 使用 testtools 模块的 assert_frame_equal 方法检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
# 使用 pytest 框架的 parametrize 装饰器，参数化测试函数的输入 kernel 和 use_arg
@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
def test_numeric_only_corr_cov_frame(kernel, numeric_only, use_arg):
    # GH#46560：测试标识
    # 创建一个 DataFrame，包含列 'a', 'b', 'c'，其中 'b' 和 'c' 的值为标量，'c' 的类型为 object
    df = DataFrame({"a": [1, 2, 3], "b": 2, "c": 3})
    # 将 'c' 列的数据类型转换为 object
    df["c"] = df["c"].astype(object)
    # 根据 use_arg 的值，构建参数元组 arg
    arg = (df,) if use_arg else ()
    # 创建一个指数加权移动窗口对象 ewm，设置 span=2, min_periods=1
    ewm = df.ewm(span=2, min_periods=1)
    # 获取 ewm 对象的指定内核方法（kernel 对应的方法）
    op = getattr(ewm, kernel)
    # 调用指定方法 op，传入参数 arg 和 numeric_only 参数，获取结果
    result = op(*arg, numeric_only=numeric_only)

    # 比较 result 和 expected，使用 float 数据类型，如果 numeric_only 为 True 则不包含 'c' 列
    columns = ["a", "b"] if numeric_only else ["a", "b", "c"]
    # 从 DataFrame df 中选择指定列，并转换为 float 类型的 DataFrame df2
    df2 = df[columns].astype(float)
    # 根据 df2 创建新的 ewm 对象 ewm2
    ewm2 = df2.ewm(span=2, min_periods=1)
    # 获取 ewm2 对象的指定内核方法（kernel 对应的方法）
    op2 = getattr(ewm2, kernel)
    # 计算预期结果 expected
    expected = op2(*arg2, numeric_only=numeric_only)
    # 使用 pytest 的 assert_frame_equal 断言比较 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 框架的 parametrize 装饰器，参数化测试函数的输入 dtype
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_series(arithmetic_win_operators, numeric_only, dtype):
    # GH#46560：测试标识
    # kernel 变量取自 arithmetic_win_operators
    kernel = arithmetic_win_operators
    # 创建一个 Series，包含一个元素，数据类型由 dtype 指定
    ser = Series([1], dtype=dtype)
    # 创建一个指数加权移动窗口对象 ewm，设置 span=2, min_periods=1
    ewm = ser.ewm(span=2, min_periods=1)
    # 获取 ewm 对象的指定内核方法（kernel 对应的方法）
    op = getattr(ewm, kernel, None)
    # 如果 op 为 None，则跳过测试
    if op is None:
        pytest.skip("No op to test")
    # 如果 numeric_only 为 True 且 dtype 为 object，则期望抛出 NotImplementedError 异常
    if numeric_only and dtype is object:
        msg = f"ExponentialMovingWindow.{kernel} does not implement numeric_only"
        # 使用 pytest 的 raises 断言检查是否抛出指定异常及消息
        with pytest.raises(NotImplementedError, match=msg):
            op(numeric_only=numeric_only)
    else:
        # 否则，调用 op 方法，传入 numeric_only 参数，获取结果
        result = op(numeric_only=numeric_only)
        # 计算预期结果 expected，聚合结果并转换为 float 类型的 Series
        expected = ser.agg([kernel]).reset_index(drop=True).astype(float)
        # 使用 pytest 的 assert_series_equal 断言比较 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)


# 使用 pytest 框架的 parametrize 装饰器，参数化测试函数的输入 kernel、use_arg 和 dtype
@pytest.mark.parametrize("kernel", ["corr", "cov"])
@pytest.mark.parametrize("use_arg", [True, False])
@pytest.mark.parametrize("dtype", [int, object])
def test_numeric_only_corr_cov_series(kernel, use_arg, numeric_only, dtype):
    # GH#46560：测试标识
    # 创建一个 Series，包含元素 [1, 2, 3]，数据类型由 dtype 指定
    ser = Series([1, 2, 3], dtype=dtype)
    # 根据 use_arg 的值，构建参数元组 arg
    arg = (ser,) if use_arg else ()
    # 创建一个指数加权移动窗口对象 ewm，设置 span=2, min_periods=1
    ewm = ser.ewm(span=2, min_periods=1)
    # 获取 ewm 对象的指定内核方法（kernel 对应的方法）
    op = getattr(ewm, kernel)
    # 如果 numeric_only 为 True 且 dtype 为 object，则期望抛出 NotImplementedError 异常
    if numeric_only and dtype is object:
        msg = f"ExponentialMovingWindow.{kernel} does not implement numeric_only"
        # 使用 pytest 的 raises 断言检查是否抛出指定异常及消息
        with pytest.raises(NotImplementedError, match=msg):
            op(*arg, numeric_only=numeric_only)
    else:
        # 否则，调用 op 方法，传入参数 arg 和 numeric_only 参数，获取结果
        result = op(*arg, numeric_only=numeric_only)

        # 创建一个新的 Series ser2，数据类型转换为 float
        ser2 = ser.astype(float)
        # 根据 ser2 创建新的 ewm 对象 ewm2
        ewm2 = ser2.ewm(span=2, min_periods=1)
        # 获取 ewm2 对象的指定内核方法（kernel 对应的方法）
        op2 = getattr(ewm2, kernel)
        # 计算预期结果 expected
        expected = op2(*arg2, numeric_only=numeric_only)
        # 使用 pytest 的 assert_series_equal 断言比较 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)
```