# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_indexing.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import re  # 导入 re 模块，用于正则表达式操作

import numpy as np  # 导入 NumPy 库，并使用 np 别名
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入以下对象：
    Index,  # 索引对象
    NaT,  # "Not a Time" 占位符
    Timedelta,  # 时间增量对象
    TimedeltaIndex,  # 时间增量索引对象
    Timestamp,  # 时间戳对象
    notna,  # 检查对象是否非 NaT 或 NaN 的函数
    offsets,  # 偏移量对象
    timedelta_range,  # 生成时间增量范围的函数
    to_timedelta,  # 转换为时间增量的函数
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块，并使用 tm 别名


class TestGetItem:
    def test_getitem_slice_keeps_name(self):
        # 测试用例：确保切片后的名称保持不变，参见 GitHub 问题编号 GH#4226, GH#59051
        msg = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            tdi = timedelta_range("1d", "5d", freq="h", name="timebucket")
        assert tdi[1:].name == tdi.name  # 断言切片后的名称与原始名称相同

    def test_getitem(self):
        idx1 = timedelta_range("1 day", "31 day", freq="D", name="idx")

        for idx in [idx1]:
            result = idx[0]  # 获取索引第一个元素
            assert result == Timedelta("1 day")  # 断言结果为 1 天的时间增量

            result = idx[0:5]  # 获取索引的前5个元素
            expected = timedelta_range("1 day", "5 day", freq="D", name="idx")
            tm.assert_index_equal(result, expected)  # 使用测试工具断言索引相等
            assert result.freq == expected.freq  # 断言频率相同

            result = idx[0:10:2]  # 获取索引的每隔2个取一个的前10个元素
            expected = timedelta_range("1 day", "9 day", freq="2D", name="idx")
            tm.assert_index_equal(result, expected)  # 使用测试工具断言索引相等
            assert result.freq == expected.freq  # 断言频率相同

            result = idx[-20:-5:3]  # 获取索引倒数第20到倒数第5个元素，步长为3
            expected = timedelta_range("12 day", "24 day", freq="3D", name="idx")
            tm.assert_index_equal(result, expected)  # 使用测试工具断言索引相等
            assert result.freq == expected.freq  # 断言频率相同

            result = idx[4::-1]  # 反向获取索引的前5个元素
            expected = TimedeltaIndex(
                ["5 day", "4 day", "3 day", "2 day", "1 day"], freq="-1D", name="idx"
            )
            tm.assert_index_equal(result, expected)  # 使用测试工具断言索引相等
            assert result.freq == expected.freq  # 断言频率相同

    @pytest.mark.parametrize(
        "key",
        [
            Timestamp("1970-01-01"),  # 时间戳参数
            Timestamp("1970-01-02"),  # 时间戳参数
            datetime(1970, 1, 1),  # datetime 对象参数
            Timestamp("1970-01-03").to_datetime64(),  # 转换后的时间戳参数
            # 非匹配的 NA 值
            np.datetime64("NaT"),
        ],
    )
    def test_timestamp_invalid_key(self, key):
        # 测试用例：确保当使用不合法的时间戳作为键时抛出 KeyError 异常，参见 GitHub 问题编号 GH#20464
        tdi = timedelta_range(0, periods=10)
        with pytest.raises(KeyError, match=re.escape(repr(key))):
            tdi.get_loc(key)


class TestGetLoc:
    def test_get_loc_key_unit_mismatch(self):
        idx = to_timedelta(["0 days", "1 days", "2 days"])
        key = idx[1].as_unit("ms")  # 将索引的第二个元素转换为毫秒单位
        loc = idx.get_loc(key)  # 获取索引中键为 key 的位置
        assert loc == 1  # 断言位置为 1

    def test_get_loc_key_unit_mismatch_not_castable(self):
        tdi = to_timedelta(["0 days", "1 days", "2 days"]).astype("m8[s]")  # 将索引转换为秒单位的时间增量
        assert tdi.dtype == "m8[s]"  # 断言索引的数据类型为秒单位的时间增量
        key = tdi[0].as_unit("ns") + Timedelta(1)  # 将索引的第一个元素转换为纳秒单位，并加上 1 纳秒

        with pytest.raises(KeyError, match=r"Timedelta\('0 days 00:00:00.000000001'\)"):
            tdi.get_loc(key)  # 断言获取 key 的位置会抛出 KeyError 异常

        assert key not in tdi  # 断言 key 不在索引中
    # 定义测试方法，用于测试 TimedeltaIndex 的 get_loc 方法
    def test_get_loc(self):
        # 创建一个时间增量数组，包含三个时间增量
        idx = to_timedelta(["0 days", "1 days", "2 days"])

        # GH 16909：验证在时间增量数组中查找时间增量的位置
        assert idx.get_loc(idx[1].to_timedelta64()) == 1

        # GH 16896：验证在时间增量数组中查找字符串 '0 days' 的位置
        assert idx.get_loc("0 days") == 0

    # 定义测试方法，用于测试包含自然元素的 TimedeltaIndex 的 get_loc 方法
    def test_get_loc_nat(self):
        # 创建一个时间增量索引数组，包含三个时间增量，其中一个是 NaT（Not a Time）
        tidx = TimedeltaIndex(["1 days 01:00:00", "NaT", "2 days 01:00:00"])

        # 验证在时间增量索引数组中查找 NaT 的位置，预期返回位置 1
        assert tidx.get_loc(NaT) == 1

        # 验证在时间增量索引数组中查找 None 的位置，预期返回位置 1
        assert tidx.get_loc(None) == 1

        # 验证在时间增量索引数组中查找 float("nan") 的位置，预期返回位置 1
        assert tidx.get_loc(float("nan")) == 1

        # 验证在时间增量索引数组中查找 np.nan 的位置，预期返回位置 1
        assert tidx.get_loc(np.nan) == 1
class TestGetIndexer:
    def test_get_indexer(self):
        # 创建时间增量对象，表示0天、1天、2天
        idx = to_timedelta(["0 days", "1 days", "2 days"])
        # 断言返回的索引数组与预期的相等，应为 [0, 1, 2]，数据类型为 np.intp
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        # 创建目标时间增量对象，表示-1小时、12小时、1天1小时
        target = to_timedelta(["-1 hour", "12 hours", "1 day 1 hour"])
        # 断言根据目标时间获取的索引数组与预期的相等，使用 "pad" 填充方式，应为 [-1, 0, 1]，数据类型为 np.intp
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        # 断言根据目标时间获取的索引数组与预期的相等，使用 "backfill" 填充方式，应为 [0, 1, 2]，数据类型为 np.intp
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        # 断言根据目标时间获取的索引数组与预期的相等，使用 "nearest" 填充方式，应为 [0, 1, 1]，数据类型为 np.intp
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )

        # 使用 "nearest" 填充方式和 1小时容差获取索引数组，应为 [0, -1, 1]，数据类型为 np.intp
        res = idx.get_indexer(target, "nearest", tolerance=Timedelta("1 hour"))
        tm.assert_numpy_array_equal(res, np.array([0, -1, 1], dtype=np.intp))


class TestWhere:
    def test_where_doesnt_retain_freq(self):
        # 创建时间增量索引对象，包含3个周期为1天的时间间隔
        tdi = timedelta_range("1 day", periods=3, freq="D", name="idx")
        # 创建条件列表
        cond = [True, True, False]
        # 创建预期的时间增量索引对象，忽略频率信息，名称为 "idx"
        expected = TimedeltaIndex([tdi[0], tdi[1], tdi[0]], freq=None, name="idx")

        # 使用条件进行筛选，返回结果与预期相等
        result = tdi.where(cond, tdi[::-1])
        tm.assert_index_equal(result, expected)

    def test_where_invalid_dtypes(self, fixed_now_ts):
        # 创建时间增量索引对象，包含3个周期为1天的时间间隔
        tdi = timedelta_range("1 day", periods=3, freq="D", name="idx")

        # 获取末尾部分的时间增量对象
        tail = tdi[2:].tolist()
        # 创建索引对象，包含 NaT 值
        i2 = Index([NaT, NaT] + tail)
        # 创建非空值掩码
        mask = notna(i2)

        # 创建预期的索引对象，数据类型为 object，名称为 "idx"
        expected = Index([NaT._value, NaT._value] + tail, dtype=object, name="idx")
        assert isinstance(expected[0], int)
        # 使用掩码对时间增量索引对象进行筛选，返回结果与预期相等
        result = tdi.where(mask, i2.asi8)
        tm.assert_index_equal(result, expected)

        # 创建时间戳对象，并与掩码对时间增量索引对象进行筛选，返回结果与预期相等
        ts = i2 + fixed_now_ts
        expected = Index([ts[0], ts[1]] + tail, dtype=object, name="idx")
        result = tdi.where(mask, ts)
        tm.assert_index_equal(result, expected)

        # 将时间戳转换为周期对象，并与掩码对时间增量索引对象进行筛选，返回结果与预期相等
        per = (i2 + fixed_now_ts).to_period("D")
        expected = Index([per[0], per[1]] + tail, dtype=object, name="idx")
        result = tdi.where(mask, per)
        tm.assert_index_equal(result, expected)

        # 使用固定的当前时间戳与掩码对时间增量索引对象进行筛选，返回结果与预期相等
        ts = fixed_now_ts
        expected = Index([ts, ts] + tail, dtype=object, name="idx")
        result = tdi.where(mask, ts)
        tm.assert_index_equal(result, expected)

    def test_where_mismatched_nat(self):
        # 创建时间增量索引对象，包含3个周期为1天的时间间隔
        tdi = timedelta_range("1 day", periods=3, freq="D", name="idx")
        # 创建布尔数组作为条件
        cond = np.array([True, False, False])

        # 创建特殊的 NaT 对象
        dtnat = np.datetime64("NaT", "ns")
        # 创建预期的索引对象，数据类型为 object，名称为 "idx"
        expected = Index([tdi[0], dtnat, dtnat], dtype=object, name="idx")
        assert expected[2] is dtnat
        # 使用条件和特殊 NaT 对象对时间增量索引对象进行筛选，返回结果与预期相等
        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)
    # 定义测试函数 test_take(self)，用于测试 timedelta_range 的 take 方法
    def test_take(self):
        # GH 10295
        # 创建一个时间增量范围对象，从 "1 day" 到 "31 day"，频率为每日 ("D")，名称为 "idx"
        idx1 = timedelta_range("1 day", "31 day", freq="D", name="idx")

        # 遍历包含单个时间增量范围对象的列表 [idx1]
        for idx in [idx1]:
            # 调用 take 方法，从索引列表 [0] 中获取索引对应的时间增量，断言结果为 Timedelta("1 day")
            result = idx.take([0])
            assert result == Timedelta("1 day")

            # 调用 take 方法，从索引列表 [-1] 中获取索引对应的时间增量，断言结果为 Timedelta("31 day")
            result = idx.take([-1])
            assert result == Timedelta("31 day")

            # 调用 take 方法，从索引列表 [0, 1, 2] 中获取索引对应的时间增量，预期结果为从 "1 day" 到 "3 day" 的时间增量范围
            expected = timedelta_range("1 day", "3 day", freq="D", name="idx")
            result = idx.take([0, 1, 2])
            # 使用 tm.assert_index_equal 断言结果与预期相符
            tm.assert_index_equal(result, expected)
            # 断言结果的频率与预期相符
            assert result.freq == expected.freq

            # 调用 take 方法，从索引列表 [0, 2, 4] 中获取索引对应的时间增量，预期结果为从 "1 day" 到 "5 day" 的时间增量范围，频率为每 2 天
            expected = timedelta_range("1 day", "5 day", freq="2D", name="idx")
            result = idx.take([0, 2, 4])
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            # 调用 take 方法，从索引列表 [7, 4, 1] 中获取索引对应的时间增量，预期结果为从 "8 day" 到 "2 day" 的时间增量范围，每隔 -3 天
            expected = timedelta_range("8 day", "2 day", freq="-3D", name="idx")
            result = idx.take([7, 4, 1])
            tm.assert_index_equal(result, expected)
            assert result.freq == expected.freq

            # 调用 take 方法，从索引列表 [3, 2, 5] 中获取索引对应的时间增量，预期结果为 ["4 day", "3 day", "6 day"] 的时间增量索引，没有指定频率
            expected = TimedeltaIndex(["4 day", "3 day", "6 day"], name="idx")
            result = idx.take([3, 2, 5])
            tm.assert_index_equal(result, expected)
            # 断言结果的频率为 None
            assert result.freq is None

            # 调用 take 方法，从索引列表 [-3, 2, 5] 中获取索引对应的时间增量，预期结果为 ["29 day", "3 day", "6 day"] 的时间增量索引，没有指定频率
            expected = TimedeltaIndex(["29 day", "3 day", "6 day"], name="idx")
            result = idx.take([-3, 2, 5])
            tm.assert_index_equal(result, expected)
            # 断言结果的频率为 None
            assert result.freq is None

    # 定义测试函数 test_take_invalid_kwargs(self)，用于测试 timedelta_range 的 take 方法在传入无效参数时的行为
    def test_take_invalid_kwargs(self):
        # 创建一个时间增量范围对象，从 "1 day" 到 "31 day"，频率为每日 ("D")，名称为 "idx"
        idx = timedelta_range("1 day", "31 day", freq="D", name="idx")
        # 定义索引列表
        indices = [1, 6, 5, 9, 10, 13, 15, 3]

        # 用 pytest 的 pytest.raises 检查传入无效参数 'foo' 时是否会抛出 TypeError 异常，异常信息包含 "take() got an unexpected keyword argument 'foo'"
        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)

        # 用 pytest 的 pytest.raises 检查传入无效参数 'out' 时是否会抛出 ValueError 异常，异常信息为 "the 'out' parameter is not supported"
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)

        # 用 pytest 的 pytest.raises 检查传入无效参数 'mode' 时是否会抛出 ValueError 异常，异常信息为 "the 'mode' parameter is not supported"
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode="clip")

    # 定义测试函数 test_take_equiv_getitem(self)，测试 timedelta_range 的 take 方法与直接使用索引操作符 [] 的等效性
    def test_take_equiv_getitem(self):
        # 定义一组时间增量字符串列表
        tds = ["1day 02:00:00", "1 day 04:00:00", "1 day 10:00:00"]
        # 创建一个时间增量范围对象，从 "1D" 到 "2D"，频率为每小时 ("h")，名称为 "idx"
        idx = timedelta_range(start="1D", end="2D", freq="h", name="idx")
        # 创建预期结果，为一个时间增量索引对象，包含 tds 中的时间增量字符串，没有指定频率
        expected = TimedeltaIndex(tds, freq=None, name="idx")

        # 调用 take 方法，从索引列表 [2, 4, 10] 中获取索引对应的时间增量索引，结果为 taken1
        taken1 = idx.take([2, 4, 10])
        # 直接使用索引操作符 []，从索引列表 [2, 4, 10] 中获取索引对应的时间增量索引，结果为 taken2
        taken2 = idx[[2, 4, 10]]

        # 遍历包含 taken1 和 taken2 的列表，分别进行以下断言：
        for taken in [taken1, taken2]:
            # 使用 tm.assert_index_equal 断言结果与预期相符
            tm.assert_index_equal(taken, expected)
            # 断言结果为 TimedeltaIndex 对象
            assert isinstance(taken, TimedeltaIndex)
            # 断言结果的频率为 None
            assert taken.freq is None
            # 断言结果的名称与预期相符
            assert taken.name == expected.name
    # 定义一个测试方法，用于测试 TimedeltaIndex 类的 take 方法
    def test_take_fill_value(self):
        # GH 12631，这是 GitHub 上的问题编号，指明此处是为了解决特定的问题
        idx = TimedeltaIndex(["1 days", "2 days", "3 days"], name="xxx")
        # 调用 take 方法，根据给定的索引数组取出对应的元素，构造结果
        result = idx.take(np.array([1, 0, -1]))
        # 期望的结果 TimedeltaIndex
        expected = TimedeltaIndex(["2 days", "1 days", "3 days"], name="xxx")
        # 断言两个索引对象是否相等
        tm.assert_index_equal(result, expected)

        # 使用 fill_value 参数进行索引，当索引超出边界时使用填充值
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        # 期望的结果 TimedeltaIndex，当索引为 -1 时使用填充值 NaT
        expected = TimedeltaIndex(["2 days", "1 days", "NaT"], name="xxx")
        tm.assert_index_equal(result, expected)

        # allow_fill=False，禁止使用填充值，索引超出边界时抛出异常
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        # 期望的结果 TimedeltaIndex，与第一个断言结果相同
        expected = TimedeltaIndex(["2 days", "1 days", "3 days"], name="xxx")
        tm.assert_index_equal(result, expected)

        # 测试索引超出边界且 allow_fill=True 且 fill_value 不为空时的异常情况
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        # 使用 pytest 检查是否会抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 测试索引超出边界时是否会抛出 IndexError 异常
        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        # 使用 pytest 检查是否会抛出 IndexError 异常，并匹配特定的错误消息
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))
    class TestMaybeCastSliceBound:
        # 定义测试类 TestMaybeCastSliceBound
        @pytest.fixture(params=["increasing", "decreasing", None])
        # 定义参数化 fixture，提供参数 "increasing", "decreasing", None
        def monotonic(self, request):
            # 返回 fixture monotonic 的值，根据 request 参数确定
            return request.param

        @pytest.fixture
        # 定义普通 fixture
        def tdi(self, monotonic):
            # 生成一个时间增量范围为 "1 Day"，包含 10 个时间点的 TimedeltaIndex 对象 tdi
            tdi = timedelta_range("1 Day", periods=10)
            if monotonic == "decreasing":
                # 如果 monotonic 是 "decreasing"，则反转 tdi 的顺序
                tdi = tdi[::-1]
            elif monotonic is None:
                # 如果 monotonic 是 None，则生成一个包含 10 个元素的整数数组 taker
                taker = np.arange(10, dtype=np.intp)
                # 使用随机种子 2 对 taker 进行打乱
                np.random.default_rng(2).shuffle(taker)
                # 根据 taker 选择 tdi 中的元素，重新赋值给 tdi
                tdi = tdi.take(taker)
            return tdi

        def test_maybe_cast_slice_bound_invalid_str(self, tdi):
            # 测试 _maybe_cast_slice_bound 方法，确保捕获预期的 TypeError 异常及其消息
            msg = (
                "cannot do slice indexing on TimedeltaIndex with these "
                r"indexers \[foo\] of type str"
            )
            # 使用 pytest 检查是否抛出 TypeError 异常，且消息与预期匹配
            with pytest.raises(TypeError, match=msg):
                tdi._maybe_cast_slice_bound("foo", side="left")
            with pytest.raises(TypeError, match=msg):
                tdi.get_slice_bound("foo", side="left")
            with pytest.raises(TypeError, match=msg):
                tdi.slice_locs("foo", None, None)

        def test_slice_invalid_str_with_timedeltaindex(
            self, tdi, frame_or_series, indexer_sl
        ):
            # 创建一个对象 obj，其索引为 tdi，测试确保捕获预期的 TypeError 异常及其消息
            obj = frame_or_series(range(10), index=tdi)

            msg = (
                "cannot do slice indexing on TimedeltaIndex with these "
                r"indexers \[foo\] of type str"
            )
            # 使用 pytest 检查是否抛出 TypeError 异常，且消息与预期匹配
            with pytest.raises(TypeError, match=msg):
                indexer_sl(obj)["foo":]
            with pytest.raises(TypeError, match=msg):
                indexer_sl(obj)["foo":-1]
            with pytest.raises(TypeError, match=msg):
                indexer_sl(obj)[:"foo"]
            with pytest.raises(TypeError, match=msg):
                indexer_sl(obj)[tdi[0] : "foo"]


    class TestContains:
        # 定义测试类 TestContains

        def test_contains_nonunique(self):
            # 测试在 TimedeltaIndex 中确保第一个元素存在于索引中
            # GH#9512
            for vals in (
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, -1],
                ["00:01:00", "00:01:00", "00:02:00"],
                ["00:01:00", "00:01:00", "00:00:01"],
            ):
                idx = TimedeltaIndex(vals)
                # 断言索引中的第一个元素存在于索引对象中
                assert idx[0] in idx

        def test_contains(self):
            # 测试检查是否包含任何 NaT 类型的对象
            # GH#13603, GH#59051
            msg = "'d' is deprecated and will be removed in a future version."
            # 使用 tm.assert_produces_warning 确保出现 FutureWarning 警告并匹配消息
            with tm.assert_produces_warning(FutureWarning, match=msg):
                td = to_timedelta(range(5), unit="d") + offsets.Hour(1)
            for v in [NaT, None, float("nan"), np.nan]:
                # 断言 v 不在 td 中
                assert v not in td

            td = to_timedelta([NaT])
            for v in [NaT, None, float("nan"), np.nan]:
                # 断言 v 在 td 中
                assert v in td
```