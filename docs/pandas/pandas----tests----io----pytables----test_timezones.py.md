# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_timezones.py`

```
# 导入必要的模块和函数
from datetime import (
    date,        # 导入日期对象
    timedelta,   # 导入时间间隔对象
)

import numpy as np   # 导入NumPy库，用于数值计算
import pytest         # 导入pytest库，用于单元测试

# 导入时区处理相关函数
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td

# 导入pandas库并使用简化别名pd
import pandas as pd
from pandas import (
    DataFrame,        # 导入DataFrame类
    DatetimeIndex,    # 导入DatetimeIndex类
    Series,           # 导入Series类
    Timestamp,        # 导入Timestamp类
    date_range,       # 导入日期范围生成函数
)
import pandas._testing as tm   # 导入pandas测试相关模块
from pandas.tests.io.pytables.common import (
    _maybe_remove,     # 导入辅助函数用于处理文件删除
    ensure_clean_store,  # 导入确保存储干净的函数
)


def _compare_with_tz(a, b):
    tm.assert_frame_equal(a, b)  # 使用pandas测试模块比较DataFrame a 和 b

    # 比较每个元素的时区
    for c in a.columns:
        for i in a.index:
            a_e = a.loc[i, c]   # 获取DataFrame a 在 (i, c) 处的元素
            b_e = b.loc[i, c]   # 获取DataFrame b 在 (i, c) 处的元素
            # 如果时区不相同则抛出异常
            if not (a_e == b_e and a_e.tz == b_e.tz):
                raise AssertionError(f"invalid tz comparison [{a_e}] [{b_e}]")


# 使用maybe_get_tz替代dateutil.tz.gettz处理Windows文件名问题
gettz_dateutil = lambda x: maybe_get_tz("dateutil/" + x)
gettz_pytz = lambda x: x


@pytest.mark.parametrize("gettz", [gettz_dateutil, gettz_pytz])
def test_append_with_timezones(setup_path, gettz):
    # 作为列的DataFrame示例

    # 单一时区，无夏令时转换
    df_est = DataFrame(
        {
            "A": [
                Timestamp("20130102 2:00:00", tz=gettz("US/Eastern")).as_unit("ns")
                + timedelta(hours=1) * i
                for i in range(5)
            ]
        }
    )

    # 所有列具有相同时区的DataFrame示例，但处于不同夏令时转换的两侧
    df_crosses_dst = DataFrame(
        {
            "A": Timestamp("20130102", tz=gettz("US/Eastern")).as_unit("ns"),
            "B": Timestamp("20130603", tz=gettz("US/Eastern")).as_unit("ns"),
        },
        index=range(5),
    )

    # 混合时区的DataFrame示例
    df_mixed_tz = DataFrame(
        {
            "A": Timestamp("20130102", tz=gettz("US/Eastern")).as_unit("ns"),
            "B": Timestamp("20130102", tz=gettz("EET")).as_unit("ns"),
        },
        index=range(5),
    )

    # 不同时区的DataFrame示例
    df_different_tz = DataFrame(
        {
            "A": Timestamp("20130102", tz=gettz("US/Eastern")).as_unit("ns"),
            "B": Timestamp("20130102", tz=gettz("CET")).as_unit("ns"),
        },
        index=range(5),
    )
    # 使用 ensure_clean_store 上下文管理器创建一个存储对象 store，确保环境干净
    with ensure_clean_store(setup_path) as store:
        # 可能移除存储中的 "df_tz" 数据
        _maybe_remove(store, "df_tz")
        # 向存储中追加 df_est 数据，使用列 "A" 作为数据列
        store.append("df_tz", df_est, data_columns=["A"])
        # 从存储中获取名为 "df_tz" 的数据
        result = store["df_tz"]
        # 比较存储中的结果与 df_est 数据的时区敏感性
        _compare_with_tz(result, df_est)
        # 断言存储中的结果与 df_est 相等
        tm.assert_frame_equal(result, df_est)

        # 使用时区感知进行选择操作
        expected = df_est[df_est.A >= df_est.A[3]]
        result = store.select("df_tz", where="A>=df_est.A[3]")
        # 比较存储中的结果与预期结果的时区敏感性
        _compare_with_tz(result, expected)

        # 确保在这里包含夏令时和标准时间的日期
        _maybe_remove(store, "df_tz")
        # 向存储中追加 df_crosses_dst 数据
        store.append("df_tz", df_crosses_dst)
        # 从存储中获取名为 "df_tz" 的数据
        result = store["df_tz"]
        # 比较存储中的结果与 df_crosses_dst 数据的时区敏感性
        _compare_with_tz(result, df_crosses_dst)
        # 断言存储中的结果与 df_crosses_dst 相等
        tm.assert_frame_equal(result, df_crosses_dst)

        # 设置错误消息的模式匹配，用于捕获 ValueError 异常
        msg = (
            r"invalid info for \[values_block_1\] for \[tz\], "
            r"existing_value \[(dateutil/.*)?(US/Eastern|America/New_York)\] "
            r"conflicts with new value \[(dateutil/.*)?EET\]"
        )
        # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            store.append("df_tz", df_mixed_tz)

        # 可以正常执行的操作
        _maybe_remove(store, "df_tz")
        # 向存储中追加 df_mixed_tz 数据，使用列 "A" 和 "B" 作为数据列
        store.append("df_tz", df_mixed_tz, data_columns=["A", "B"])
        # 从存储中获取名为 "df_tz" 的数据
        result = store["df_tz"]
        # 比较存储中的结果与 df_mixed_tz 数据的时区敏感性
        _compare_with_tz(result, df_mixed_tz)
        # 断言存储中的结果与 df_mixed_tz 相等
        tm.assert_frame_equal(result, df_mixed_tz)

        # 无法追加具有不同时区的数据
        msg = (
            r"invalid info for \[B\] for \[tz\], "
            r"existing_value \[(dateutil/.*)?EET\] "
            r"conflicts with new value \[(dateutil/.*)?CET\]"
        )
        # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            store.append("df_tz", df_different_tz)
@pytest.mark.parametrize("gettz", [gettz_dateutil, gettz_pytz])
# 使用参数化测试，测试函数 gettz 可以是 gettz_dateutil 或 gettz_pytz 两者之一
def test_append_with_timezones_as_index(setup_path, gettz):
    # 测试函数，确保索引带有时区信息的数据框的附加和选择操作
    # GH#4098 示例

    dti = date_range("2000-1-1", periods=3, freq="h", tz=gettz("US/Eastern"))
    # 创建一个带有时区信息的日期时间索引对象 dti，频率为每小时，时区为美国东部

    dti = dti._with_freq(None)  # freq doesn't round-trip
    # 将 dti 的频率设置为 None，以测试频率不能回环

    df = DataFrame({"A": Series(range(3), index=dti)})
    # 创建一个数据框 df，包含一列名为 "A" 的序列数据，索引为 dti

    with ensure_clean_store(setup_path) as store:
        # 使用确保清洁存储环境的上下文管理器
        _maybe_remove(store, "df")
        # 如果存在，则移除名称为 "df" 的对象

        store.put("df", df)
        # 将数据框 df 放入存储中，键为 "df"

        result = store.select("df")
        # 从存储中选择名称为 "df" 的对象，将结果赋给 result

        tm.assert_frame_equal(result, df)
        # 使用测试框架的函数检查 result 是否与 df 相等

        _maybe_remove(store, "df")
        # 再次检查并移除名称为 "df" 的对象

        store.append("df", df)
        # 在存储中附加数据框 df，键为 "df"

        result = store.select("df")
        # 从存储中选择名称为 "df" 的对象，将结果赋给 result

        tm.assert_frame_equal(result, df)
        # 使用测试框架的函数检查 result 是否与 df 相等


def test_roundtrip_tz_aware_index(setup_path, unit):
    # 测试函数，确保带有时区信息的索引的往返存储操作
    # GH 17618 示例

    ts = Timestamp("2000-01-01 01:00:00", tz="US/Eastern")
    # 创建一个带有美国东部时区信息的时间戳对象 ts

    dti = DatetimeIndex([ts]).as_unit(unit)
    # 创建一个日期时间索引对象 dti，包含 ts，时间单位由参数 unit 指定

    df = DataFrame(data=[0], index=dti)
    # 创建一个数据框 df，包含数据为 [0]，索引为 dti

    with ensure_clean_store(setup_path) as store:
        # 使用确保清洁存储环境的上下文管理器
        store.put("frame", df, format="fixed")
        # 将数据框 df 以固定格式放入存储中，键为 "frame"

        recons = store["frame"]
        # 从存储中选择名称为 "frame" 的对象，将结果赋给 recons

        tm.assert_frame_equal(recons, df)
        # 使用测试框架的函数检查 recons 是否与 df 相等

    value = recons.index[0]._value
    # 从 recons 的索引中获取第一个元素的值

    denom = {"ns": 1, "us": 1000, "ms": 10**6, "s": 10**9}[unit]
    # 根据 unit 确定分母的值

    assert value == 946706400000000000 // denom
    # 断言 value 等于计算结果 946706400000000000 除以 denom 的整数部分


def test_store_index_name_with_tz(setup_path):
    # 测试函数，确保存储带有时区信息的索引名称
    # GH 13884 示例

    df = DataFrame({"A": [1, 2]})
    # 创建一个数据框 df，包含一列名为 "A" 的数据

    df.index = DatetimeIndex([1234567890123456787, 1234567890123456788])
    # 将 df 的索引设置为包含两个日期时间的日期时间索引对象，这两个时间戳是 1234567890123456787 和 1234567890123456788

    df.index = df.index.tz_localize("UTC")
    # 将 df 的索引本地化为 UTC 时区

    df.index.name = "foo"
    # 将 df 的索引名称设置为 "foo"

    with ensure_clean_store(setup_path) as store:
        # 使用确保清洁存储环境的上下文管理器
        store.put("frame", df, format="table")
        # 将数据框 df 以表格格式放入存储中，键为 "frame"

        recons = store["frame"]
        # 从存储中选择名称为 "frame" 的对象，将结果赋给 recons

        tm.assert_frame_equal(recons, df)
        # 使用测试框架的函数检查 recons 是否与 df 相等


def test_tseries_select_index_column(setup_path):
    # 测试函数，确保选择带有时间序列索引列的操作
    # GH7777 示例
    # 选择 UTC datetimeindex 列之前未保留 UTC tzinfo

    # 检查没有时区信息是否仍然有效
    rng = date_range("1/1/2000", "1/30/2000")
    # 创建一个日期范围对象 rng，从 "2000-01-01" 到 "2000-01-30"

    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )
    # 创建一个数据框 frame，包含随机数据，索引为 rng

    with ensure_clean_store(setup_path) as store:
        # 使用确保清洁存储环境的上下文管理器
        store.append("frame", frame)
        # 在存储中附加数据框 frame，键为 "frame"

        result = store.select_column("frame", "index")
        # 从存储中选择名称为 "frame" 的对象的 "index" 列，将结果赋给 result

        assert rng.tz == DatetimeIndex(result.values).tz
        # 断言 rng 的时区与 result 的日期时间索引的时区相同

    # 检查 UTC 时区
    rng = date_range("1/1/2000", "1/30/2000", tz="UTC")
    # 创建一个日期范围对象 rng，从 "2000-01-01" 到 "2000-01-30"，时区为 UTC

    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )
    # 创建一个数据框 frame，包含随机数据，索引为 rng

    with ensure_clean_store(setup_path) as store:
        # 使用确保清洁存储环境的上下文管理器
        store.append("frame", frame)
        # 在存储中附加数据框 frame，键为 "frame"

        result = store.select_column("frame", "index")
        # 从存储中选择名称为 "frame" 的对象的 "index" 列，将结果赋给 result

        assert rng.tz == result.dt.tz
        # 断言 rng 的时区与 result 的日期时间索引的时区相同

    # 再次检查非 UTC 时区
    rng = date_range("1/1/2000", "1/30/2000", tz="US/Eastern")
    # 创建一个日期范围对象 rng，从 "2000-01-01" 到 "2000-01-30"，时区为美国东部

    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )
    # 创建一个数据框 frame，包含随机数据，索引为 rng

    with ensure_clean_store(setup_path) as store:
        # 使用确保清洁存储环境的上下文管理器
        store.append("frame", frame)
        # 在存储中附加数据框 frame，键为 "frame"

        result =
    # 使用 ensure_clean_store 上下文管理器创建临时存储，确保操作后存储处于干净状态
    with ensure_clean_store(setup_path) as store:
        # index
        # 创建日期范围，从 "1/1/2000" 到 "1/30/2000"，使用美国东部时区
        rng = date_range("1/1/2000", "1/30/2000", tz="US/Eastern")
        # 修改日期范围对象的频率为 None，因为频率不支持往返
        rng = rng._with_freq(None)
        # 创建一个 DataFrame，形状为 (日期范围长度, 4)，填充随机正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
        )
        # 将 DataFrame 存储在临时存储中，键名为 "df"
        store["df"] = df
        # 从临时存储中获取 "df" 键对应的值，赋给 result
        result = store["df"]
        # 使用测试工具比较 result 和 df，确保它们相等
        tm.assert_frame_equal(result, df)

        # as data
        # GH11411
        # 在临时存储中移除名为 "df" 的项
        _maybe_remove(store, "df")
        # 创建一个新的 DataFrame，包含列 "A"、"B"、"C" 和 "D"，索引为 rng
        df = DataFrame(
            {
                "A": rng,
                "B": rng.tz_convert("UTC").tz_localize(None),
                "C": rng.tz_convert("CET"),
                "D": range(len(rng)),
            },
            index=rng,
        )
        # 将新的 DataFrame 存储在临时存储中，键名为 "df"
        store["df"] = df
        # 从临时存储中获取 "df" 键对应的值，赋给 result
        result = store["df"]
        # 使用测试工具比较 result 和 df，确保它们相等
        tm.assert_frame_equal(result, df)
# GH 20594
# 使用指定的时区创建 DatetimeTZDtype 类型的对象
dtype = pd.DatetimeTZDtype(tz=tz_aware_fixture)

obj = Series(dtype=dtype, name="A")
# 如果 frame_or_series 是 DataFrame 类型，则将 Series 对象转换为 DataFrame 对象
if frame_or_series is DataFrame:
    obj = obj.to_frame()

# 使用 ensure_clean_store 函数确保存储路径 setup_path 是干净的
with ensure_clean_store(setup_path) as store:
    # 将 obj 存储到 store 中，键名为 "obj"
    store["obj"] = obj
    # 从 store 中读取 "obj"，并将结果赋给 result
    result = store["obj"]
    # 使用测试模块中的 assert_equal 函数比较 result 和 obj
    tm.assert_equal(result, obj)


# GH 20594
# 使用指定的时区创建 DatetimeTZDtype 类型的对象
dtype = pd.DatetimeTZDtype(tz=tz_aware_fixture)

# 使用 ensure_clean_store 函数确保存储路径 setup_path 是干净的
with ensure_clean_store(setup_path) as store:
    # 创建一个包含单个元素的 Series 对象 s，数据类型为 dtype
    s = Series([0], dtype=dtype)
    # 将 s 存储到 store 中，键名为 "s"
    store["s"] = s
    # 从 store 中读取 "s"，并将结果赋给 result
    result = store["s"]
    # 使用测试模块中的 assert_series_equal 函数比较 result 和 s
    tm.assert_series_equal(result, s)


def test_fixed_offset_tz(setup_path):
    # 创建一个时间范围 rng，从 "1/1/2000 00:00:00-07:00" 到 "1/30/2000 00:00:00-07:00"
    rng = date_range("1/1/2000 00:00:00-07:00", "1/30/2000 00:00:00-07:00")
    # 创建一个 DataFrame 对象 frame，数据为随机数矩阵，索引为 rng
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng
    )

    # 使用 ensure_clean_store 函数确保存储路径 setup_path 是干净的
    with ensure_clean_store(setup_path) as store:
        # 将 frame 存储到 store 中，键名为 "frame"
        store["frame"] = frame
        # 从 store 中读取 "frame"，并将结果赋给 recons
        recons = store["frame"]
        # 使用测试模块中的 assert_index_equal 函数比较 recons.index 和 rng
        tm.assert_index_equal(recons.index, rng)
        # 检查 recons.index 的时区是否与 rng 的时区相同
        assert rng.tz == recons.index.tz


@td.skip_if_windows
def test_store_timezone(setup_path):
    # GH2852
    # 处理存储带时区的 datetime.date 时可能遇到的问题，读取时区可能会重置

    # 使用 ensure_clean_store 函数确保存储路径 setup_path 是干净的
    with ensure_clean_store(setup_path) as store:
        # 使用 datetime.date 创建一个 DataFrame df，将其存储为 "obj1"
        today = date(2013, 9, 10)
        df = DataFrame([1, 2, 3], index=[today, today, today])
        store["obj1"] = df
        # 从 store 中读取 "obj1"，并将结果赋给 result
        result = store["obj1"]
        # 使用测试模块中的 assert_frame_equal 函数比较 result 和 df
        tm.assert_frame_equal(result, df)

    # 使用时区设置 "EST5EDT" 执行相同的存储和读取操作
    with ensure_clean_store(setup_path) as store:
        with tm.set_timezone("EST5EDT"):
            today = date(2013, 9, 10)
            df = DataFrame([1, 2, 3], index=[today, today, today])
            store["obj1"] = df

        # 使用时区设置 "CST6CDT" 读取存储的数据
        with tm.set_timezone("CST6CDT"):
            result = store["obj1"]

        # 使用测试模块中的 assert_frame_equal 函数比较 result 和 df
        tm.assert_frame_equal(result, df)


def test_dst_transitions(setup_path):
    # 确保不会在时区转换时失败

    # 使用 ensure_clean_store 函数确保存储路径 setup_path 是干净的
    with ensure_clean_store(setup_path) as store:
        # 创建一个时间范围 times，从 "2013-10-26 23:00" 到 "2013-10-27 01:00"，
        # 时区为 "Europe/London"，频率为每小时，根据情况推断时间歧义
        times = date_range(
            "2013-10-26 23:00",
            "2013-10-27 01:00",
            tz="Europe/London",
            freq="h",
            ambiguous="infer",
        )
        # 将 times 的频率设置为 None，以避免频率不完整
        times = times._with_freq(None)

        # 对于每个时间范围 times 和 times + 10 分钟
        for i in [times, times + pd.Timedelta("10min")]:
            # 如果存在 "df"，则移除它
            _maybe_remove(store, "df")
            # 创建一个 DataFrame df，包含列 "A" 和 "B"，索引为 i
            df = DataFrame({"A": range(len(i)), "B": i}, index=i)
            # 将 df 追加到 store 中的 "df"
            store.append("df", df)
            # 从 store 中选择 "df"，并将结果赋给 result
            result = store.select("df")
            # 使用测试模块中的 assert_frame_equal 函数比较 result 和 df
            tm.assert_frame_equal(result, df)


def test_read_with_where_tz_aware_index(tmp_path, setup_path):
    # GH 11926
    # 创建一个时间范围 dts，从 "20151201" 开始，期间为 10 天，频率为每天，时区为 "UTC"
    periods = 10
    dts = date_range("20151201", periods=periods, freq="D", tz="UTC")
    # 创建一个 MultiIndex mi，包含两个层级：日期和序号
    mi = pd.MultiIndex.from_arrays([dts, range(periods)], names=["DATE", "NO"])
    # 创建一个 DataFrame expected，包含列 "MYCOL"，索引为 mi
    expected = DataFrame({"MYCOL": 0}, index=mi)

    # 设置存储的键名为 "mykey"，路径为 tmp_path / setup_path
    key = "mykey"
    path = tmp_path / setup_path
    # 使用 `pd.HDFStore` 打开 HDF5 文件，使用上下文管理器确保在退出代码块时文件被正确关闭
    with pd.HDFStore(path) as store:
        # 向 HDF5 文件中的指定键（`key`）追加数据 `expected`，使用表格格式存储，追加模式为真
        store.append(key, expected, format="table", append=True)
    
    # 从 HDF5 文件中读取指定键 `key` 的数据，只选择满足条件 "DATE > 20151130" 的数据
    result = pd.read_hdf(path, key, where="DATE > 20151130")
    
    # 使用 `tm.assert_frame_equal` 函数断言 `result` 和 `expected` 两个数据帧是否相等
    tm.assert_frame_equal(result, expected)
```