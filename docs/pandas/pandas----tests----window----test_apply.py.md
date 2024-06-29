# `D:\src\scipysrc\pandas\pandas\tests\window\test_apply.py`

```
import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    isna,
    notna,
)
import pandas._testing as tm

from pandas.tseries import offsets

# 忽略关于空切片的警告，因为我们故意在测试中使用了长度为 0 的 Series
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*(empty slice|0 for slice).*:RuntimeWarning"
)

# 定义一个函数 f，用于计算非 NaN 值的平均数
def f(x):
    return x[np.isfinite(x)].mean()

# 参数化测试，测试当 raw 参数不合法时是否会抛出 ValueError 异常
@pytest.mark.parametrize("bad_raw", [None, 1, 0])
def test_rolling_apply_invalid_raw(bad_raw):
    with pytest.raises(ValueError, match="raw parameter must be `True` or `False`"):
        Series(range(3)).rolling(1).apply(len, raw=bad_raw)

# 测试滚动应用函数在超出数据边界时的行为
def test_rolling_apply_out_of_bounds(engine_and_raw):
    # 获取测试引擎和原始数据标志
    engine, raw = engine_and_raw

    vals = Series([1, 2, 3, 4])

    # 对长度为 4 的 Series 执行滚动窗口大小为 10 的求和操作
    result = vals.rolling(10).apply(np.sum, engine=engine, raw=raw)
    assert result.isna().all()

    # 对长度为 4 的 Series 执行滚动窗口大小为 10，最小数据点为 1 的求和操作
    result = vals.rolling(10, min_periods=1).apply(np.sum, engine=engine, raw=raw)
    expected = Series([1, 3, 6, 10], dtype=float)
    tm.assert_almost_equal(result, expected)

# 参数化测试，测试滚动应用函数在不同窗口大小下的行为
@pytest.mark.parametrize("window", [2, "2s"])
def test_rolling_apply_with_pandas_objects(window):
    # 创建一个 DataFrame df，包含两列 A 和 B
    df = DataFrame(
        {
            "A": np.random.default_rng(2).standard_normal(5),
            "B": np.random.default_rng(2).integers(0, 10, size=5),
        },
        index=date_range("20130101", periods=5, freq="s"),
    )

    # 定义函数 f，用于处理滚动应用函数的行为
    def f(x):
        if x.index[0] == df.index[0]:
            return np.nan
        return x.iloc[-1]

    # 对 DataFrame 执行滚动窗口大小为 window 的函数 f 操作
    result = df.rolling(window).apply(f, raw=False)
    expected = df.iloc[2:].reindex_like(df)
    tm.assert_frame_equal(result, expected)

    # 测试使用 raw=True 参数执行滚动应用函数时是否引发 AttributeError 异常
    with tm.external_error_raised(AttributeError):
        df.rolling(window).apply(f, raw=True)

# 测试滚动应用函数的行为，包括空 Series 和指定步长
def test_rolling_apply(engine_and_raw, step):
    engine, raw = engine_and_raw

    expected = Series([], dtype="float64")
    result = expected.rolling(10, step=step).apply(
        lambda x: x.mean(), engine=engine, raw=raw
    )
    tm.assert_series_equal(result, expected)

    # 测试包含 None 值的 Series 执行滚动窗口大小为 2 的长度操作
    s = Series([None, None, None])
    result = s.rolling(2, min_periods=0, step=step).apply(
        lambda x: len(x), engine=engine, raw=raw
    )
    expected = Series([1.0, 2.0, 2.0])[::step]
    tm.assert_series_equal(result, expected)

    # 测试包含 None 值的 Series 执行滚动窗口大小为 2 的长度操作
    result = s.rolling(2, min_periods=0, step=step).apply(len, engine=engine, raw=raw)
    tm.assert_series_equal(result, expected)

# 测试所有滚动应用函数的行为
def test_all_apply(engine_and_raw):
    engine, raw = engine_and_raw

    # 创建一个包含时间序列索引的 DataFrame df
    df = (
        DataFrame(
            {"A": date_range("20130101", periods=5, freq="s"), "B": range(5)}
        ).set_index("A")
        * 2
    )
    er = df.rolling(window=1)
    r = df.rolling(window="1s")

    # 对 DataFrame 执行窗口大小为 1 的滚动应用函数
    result = r.apply(lambda x: 1, engine=engine, raw=raw)
    expected = er.apply(lambda x: 1, engine=engine, raw=raw)
    # 使用 pytest 模块中的 assert_frame_equal 函数比较 result 和 expected 两个数据框架的内容是否相等
    tm.assert_frame_equal(result, expected)
# 定义测试函数 test_ragged_apply，接受一个包含引擎和原始数据的元组作为参数
def test_ragged_apply(engine_and_raw):
    # 解包元组，分别赋值给 engine 和 raw
    engine, raw = engine_and_raw

    # 创建一个 DataFrame 对象 df，包含列 B 和索引时间戳
    df = DataFrame({"B": range(5)})
    df.index = [
        Timestamp("20130101 09:00:00"),
        Timestamp("20130101 09:00:02"),
        Timestamp("20130101 09:00:03"),
        Timestamp("20130101 09:00:05"),
        Timestamp("20130101 09:00:06"),
    ]

    # 定义函数 f，其返回值始终为 1
    f = lambda x: 1

    # 对 df 执行滚动窗口为 "1s"、最小观测期为 1 的滚动应用，使用给定的引擎和原始数据参数
    result = df.rolling(window="1s", min_periods=1).apply(f, engine=engine, raw=raw)
    # 创建预期的 DataFrame 对象 expected，与 df 相同，但列 B 的值均设为 1.0
    expected = df.copy()
    expected["B"] = 1.0
    # 断言结果 result 与预期 expected 相等
    tm.assert_frame_equal(result, expected)

    # 类似地，对滚动窗口为 "2s" 和 "5s" 的情况进行相同的操作和断言
    result = df.rolling(window="2s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)

    result = df.rolling(window="5s", min_periods=1).apply(f, engine=engine, raw=raw)
    expected = df.copy()
    expected["B"] = 1.0
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_invalid_engine，验证当引擎参数不合法时是否引发 ValueError 异常
def test_invalid_engine():
    with pytest.raises(ValueError, match="engine must be either 'numba' or 'cython'"):
        # 创建一个 Series 对象，对其执行窗口大小为 1 的滚动应用，使用未知的引擎参数 "foo"
        Series(range(1)).rolling(1).apply(lambda x: x, engine="foo")


# 定义测试函数 test_invalid_engine_kwargs_cython，验证在使用 cython 引擎时传递 engine_kwargs 是否引发 ValueError 异常
def test_invalid_engine_kwargs_cython():
    with pytest.raises(ValueError, match="cython engine does not accept engine_kwargs"):
        # 创建一个 Series 对象，对其执行窗口大小为 1 的滚动应用，使用 cython 引擎和 engine_kwargs 参数
        Series(range(1)).rolling(1).apply(
            lambda x: x, engine="cython", engine_kwargs={"nopython": False}
        )


# 定义测试函数 test_invalid_raw_numba，验证在使用 numba 引擎时是否要求 raw 参数为 True，否则引发 ValueError 异常
def test_invalid_raw_numba():
    with pytest.raises(
        ValueError, match="raw must be `True` when using the numba engine"
    ):
        # 创建一个 Series 对象，对其执行窗口大小为 1 的滚动应用，使用 numba 引擎但 raw 参数为 False
        Series(range(1)).rolling(1).apply(lambda x: x, raw=False, engine="numba")


# 使用 pytest.mark.parametrize 标记的参数化测试函数 test_rolling_apply_args_kwargs，验证滚动应用对 args 和 kwargs 参数的处理
@pytest.mark.parametrize("args_kwargs", [[None, {"par": 10}], [(10,), None]])
def test_rolling_apply_args_kwargs(args_kwargs):
    # 定义一个求和函数 numpysum，接受参数 x 和 par，返回 x 加上 par 的总和
    def numpysum(x, par):
        return np.sum(x + par)

    # 创建一个 DataFrame 对象 df，包含列 gr 和 a
    df = DataFrame({"gr": [1, 1], "a": [1, 2]})

    # 创建一个索引 idx，包含 gr 和 a 两列
    idx = Index(["gr", "a"])
    # 创建预期的 DataFrame 对象 expected，包含计算后的结果
    expected = DataFrame([[11.0, 11.0], [11.0, 12.0]], columns=idx)

    # 对 df 执行滚动窗口为 1 的应用，传递 args 和 kwargs 参数给 numpysum 函数
    result = df.rolling(1).apply(numpysum, args=args_kwargs[0], kwargs=args_kwargs[1])
    # 断言结果 result 与预期 expected 相等
    tm.assert_frame_equal(result, expected)

    # 创建一个 MultiIndex 对象 midx，包含 gr 和 None
    midx = MultiIndex.from_tuples([(1, 0), (1, 1)], names=["gr", None])
    # 创建预期的 Series 对象 expected，包含计算后的结果
    expected = Series([11.0, 12.0], index=midx, name="a")

    # 使用 groupby 方法对 df 按 gr 列分组，然后对 a 列执行滚动窗口为 1 的应用
    gb_rolling = df.groupby("gr")["a"].rolling(1)

    # 对 gb_rolling 执行滚动应用，传递 args 和 kwargs 参数给 numpysum 函数
    result = gb_rolling.apply(numpysum, args=args_kwargs[0], kwargs=args_kwargs[1])
    # 断言结果 result 与预期 expected 相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_nans，验证滚动应用对包含 NaN 值的 Series 对象的处理
def test_nans(raw):
    # 创建一个随机生成的 Series 对象 obj，包含 50 个标准正态分布的随机数，其中前 10 个和后 10 个设置为 NaN
    obj = Series(np.random.default_rng(2).standard_normal(50))
    obj[:10] = np.nan
    obj[-10:] = np.nan

    # 对 obj 执行窗口大小为 50、最小观测期为 30 的滚动应用，使用给定的原始数据参数
    result = obj.rolling(50, min_periods=30).apply(f, raw=raw)
    # 断言最后一个元素的结果 result.iloc[-1] 与 obj 中 10 到 -10 之间的均值 np.mean(obj[10:-10]) 几乎相等
    tm.assert_almost_equal(result.iloc[-1], np.mean(obj[10:-10]))

    # 对 obj 执行窗口大小为 20、最小观测期为 15 的滚动应用，使用给定的原始数据参数
    result = obj.rolling(20, min_periods=15).apply(f, raw=raw)
    # 断言结果 result.iloc[23] 为 NaN，result.iloc[24] 不为 NaN
    assert pd.isna(result.iloc[23])
    assert not pd.isna(result.iloc[24])

    # 断言结果 result.iloc[-6] 不为 NaN，result.iloc[-5] 为 NaN
    assert not pd.isna(result.iloc[-6])
    assert pd.isna(result.iloc[-5])

    # 创建另一个随机生成的 Series 对象 obj2，包含 20 个标准正态分布的随机数
    obj2 = Series(np.random.default_rng(2).standard_normal(20))
    # 对 obj2 执行窗口大小为 10、最小观测期为 5 的滚动应用，使用给定的原始数据参数
    result = obj2.rolling(10, min_periods=5).apply(f, raw=raw)
    # 断言：检查 result 的第四行是否为空值
    assert isna(result.iloc[3])
    # 断言：检查 result 的第五行是否不为空值
    assert notna(result.iloc[4])
    
    # 对 obj 应用滚动窗口大小为20的 rolling 计算，并指定最小观测期数为0，使用函数 f 处理数据（不进行优化处理）
    result0 = obj.rolling(20, min_periods=0).apply(f, raw=raw)
    # 对 obj 应用滚动窗口大小为20的 rolling 计算，并指定最小观测期数为1，使用函数 f 处理数据（不进行优化处理）
    result1 = obj.rolling(20, min_periods=1).apply(f, raw=raw)
    # 使用断言确保 result0 和 result1 的计算结果在精度上几乎相等
    tm.assert_almost_equal(result0, result1)
# 测试滚动窗口中心化处理的函数，接收一个序列 `raw` 作为输入
def test_center(raw):
    # 创建一个包含50个随机标准正态分布值的序列对象 `obj`
    obj = Series(np.random.default_rng(2).standard_normal(50))
    # 将序列的前10个和后10个值设置为NaN
    obj[:10] = np.nan
    obj[-10:] = np.nan

    # 对序列 `obj` 应用滚动窗口为20，最小观测期为15，中心化为True的函数 `f`
    result = obj.rolling(20, min_periods=15, center=True).apply(f, raw=raw)
    
    # 构建期望的结果序列 `expected`：
    # 1. 将序列 `obj` 与一个包含9个NaN值的序列合并
    # 2. 对合并后的序列应用滚动窗口为20，最小观测期为15的函数 `f`
    # 3. 从结果中删除前9行数据，并重置索引
    expected = (
        concat([obj, Series([np.nan] * 9)])
        .rolling(20, min_periods=15)
        .apply(f, raw=raw)
        .iloc[9:]
        .reset_index(drop=True)
    )
    
    # 使用测试框架检查 `result` 和 `expected` 是否相等
    tm.assert_series_equal(result, expected)


# 测试序列对象上的滚动窗口应用函数，接收一个序列 `raw` 和一个 `series` 作为输入
def test_series(raw, series):
    # 对输入序列 `series` 应用滚动窗口为50的函数 `f`
    result = series.rolling(50).apply(f, raw=raw)
    
    # 使用断言检查 `result` 是否为 `Series` 类型
    assert isinstance(result, Series)
    
    # 使用测试框架检查 `result` 的最后一个值是否接近于 `series` 最后50个值的平均值
    tm.assert_almost_equal(result.iloc[-1], np.mean(series[-50:]))


# 测试数据帧对象上的滚动窗口应用函数，接收一个序列 `raw` 和一个 `frame` 作为输入
def test_frame(raw, frame):
    # 对输入数据帧 `frame` 应用滚动窗口为50的函数 `f`
    result = frame.rolling(50).apply(f, raw=raw)
    
    # 使用断言检查 `result` 是否为 `DataFrame` 类型
    assert isinstance(result, DataFrame)
    
    # 使用测试框架检查 `result` 的最后一行是否与 `frame` 最后50行的列均值相等，忽略列名检查
    tm.assert_series_equal(
        result.iloc[-1, :],
        frame.iloc[-50:, :].apply(np.mean, axis=0, raw=raw),
        check_names=False,
    )


# 测试时间规则下序列对象的滚动窗口应用函数，接收一个序列 `raw` 和一个 `series` 作为输入
def test_time_rule_series(raw, series):
    win = 25
    minp = 10
    # 对序列 `series` 每隔2个值进行取样，并按工作日("B")重新采样后求均值
    ser = series[::2].resample("B").mean()
    # 对重新采样后的序列应用滚动窗口为25，最小观测期为10的函数 `f`
    series_result = ser.rolling(window=win, min_periods=minp).apply(f, raw=raw)
    # 获取结果序列的最后一个日期
    last_date = series_result.index[-1]
    # 计算前24个工作日前的日期
    prev_date = last_date - 24 * offsets.BDay()

    # 对原始序列 `series` 每隔2个值进行取样，并截取从 `prev_date` 到 `last_date` 之间的数据
    trunc_series = series[::2].truncate(prev_date, last_date)
    # 使用测试框架检查 `series_result` 的最后一个值是否接近于截取序列 `trunc_series` 的均值
    tm.assert_almost_equal(series_result.iloc[-1], np.mean(trunc_series))


# 测试时间规则下数据帧对象的滚动窗口应用函数，接收一个序列 `raw` 和一个 `frame` 作为输入
def test_time_rule_frame(raw, frame):
    win = 25
    minp = 10
    # 对数据帧 `frame` 每隔2行进行取样，并按工作日("B")重新采样后求均值
    frm = frame[::2].resample("B").mean()
    # 对重新采样后的数据帧应用滚动窗口为25，最小观测期为10的函数 `f`
    frame_result = frm.rolling(window=win, min_periods=minp).apply(f, raw=raw)
    # 获取结果数据帧的最后一个日期
    last_date = frame_result.index[-1]
    # 计算前24个工作日前的日期
    prev_date = last_date - 24 * offsets.BDay()

    # 对原始数据帧 `frame` 每隔2行进行取样，并截取从 `prev_date` 到 `last_date` 之间的数据
    trunc_frame = frame[::2].truncate(prev_date, last_date)
    # 使用测试框架检查 `frame_result` 中最后一个日期的序列是否接近于截取数据帧 `trunc_frame` 的列均值
    tm.assert_series_equal(
        frame_result.xs(last_date),
        trunc_frame.apply(np.mean, raw=raw),
        check_names=False,
    )


# 使用不同的 `minp` 参数值进行测试的函数，接收一个序列 `raw` 和一个 `series` 作为输入
@pytest.mark.parametrize("minp", [0, 99, 100])
def test_min_periods(raw, series, minp, step):
    # 对输入序列 `series` 应用滚动窗口为 `len(series) + 1`，最小观测期为 `minp` 的函数 `f`
    result = series.rolling(len(series) + 1, min_periods=minp, step=step).apply(
        f, raw=raw
    )
    # 对输入序列 `series` 应用滚动窗口为 `len(series)`，最小观测期为 `minp` 的函数 `f`
    expected = series.rolling(len(series), min_periods=minp, step=step).apply(
        f, raw=raw
    )
    # 创建一个包含 `result` 和 `expected` 结果中NaN值的布尔掩码
    nan_mask = isna(result)
    # 使用测试框架检查 `nan_mask` 是否与 `expected` 中NaN值的一致性
    tm.assert_series_equal(nan_mask, isna(expected))

    # 反转 `nan_mask`
    nan_mask = ~nan_mask
    # 使用测试框架检查 `result` 中非NaN值是否接近于 `expected` 中非NaN值
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])


# 测试序列对象上的滚动窗口中心化重新索引处理函数，接收一个序列 `raw` 和一个 `series` 作为输入
def test_center_reindex_series(raw, series):
    # 创建一个字符串列表 `s`，包含从 'x0' 到 'x11' 的格式化字符串
    s = [f"x{x:d}" for x in range(12)]
    minp = 10

    # 将序列 `series` 重新索引为原索引加上字符串列表 `s`，
    # 并对结果应用滚动窗口为25，最小观测期为10的函数 `f`，最后向前偏移12个位置，并重新索引为原始序列的索引
    series_xp = (
        series.reindex(list(series.index) + s)
        .rolling(window=25, min_periods=minp)
        .apply(f, raw=raw)
        .shift(-12)
        .reindex(series.index)
    )
    # 对序列 `series` 应用滚动窗口为25，最小观测期为10，中心化为True的函数 `f`
    series_rs
    # 创建新的 DataFrame `frame_xp`，其索引为原始索引加上序列 `s` 的列表，并且重新索引原始 DataFrame `frame`。
    # 对新创建的 DataFrame `frame_xp` 应用滚动窗口为 25 的 rolling 计算，并使用函数 `f` 进行处理，raw 参数为 raw。
    # 将处理后的结果向前偏移 12 个位置。
    # 再次按照原始 DataFrame `frame` 的索引重新索引 `frame_xp`。
    frame_xp = (
        frame.reindex(list(frame.index) + s)  # 重新索引，扩展索引列表
        .rolling(window=25, min_periods=minp)  # 应用滚动窗口为 25 的 rolling 计算
        .apply(f, raw=raw)  # 应用函数 f 进行处理
        .shift(-12)  # 向前偏移 12 个位置
        .reindex(frame.index)  # 按照原始索引重新索引
    )
    
    # 创建新的 DataFrame `frame_rs`，对原始 DataFrame `frame` 应用滚动窗口为 25 的 rolling 计算，并使用函数 `f` 进行处理，raw 参数为 raw。
    # 设置滚动窗口为 25，最小期数为 minp，居中对齐。
    frame_rs = frame.rolling(window=25, min_periods=minp, center=True).apply(f, raw=raw)
    
    # 使用 `tm.assert_frame_equal` 断言函数，比较 `frame_xp` 和 `frame_rs` 是否相等，用于测试和验证两个 DataFrame 是否一致。
    tm.assert_frame_equal(frame_xp, frame_rs)
```