# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_transpose.py`

```
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
    bdate_range,
    date_range,
    timedelta_range,
)
import pandas._testing as tm


class TestTranspose:
    def test_transpose_td64_intervals(self):
        # GH#44917
        # 创建一个时间间隔序列
        tdi = timedelta_range("0 Days", "3 Days")
        # 从时间间隔序列创建区间索引
        ii = IntervalIndex.from_breaks(tdi)
        # 在区间索引中插入一个 NaN 值
        ii = ii.insert(-1, np.nan)
        # 创建包含区间索引的 DataFrame
        df = DataFrame(ii)

        # 对 DataFrame 进行转置操作
        result = df.T
        # 创建预期的转置后的 DataFrame
        expected = DataFrame({i: ii[i : i + 1] for i in range(len(ii))})
        # 断言转置后的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)

    def test_transpose_empty_preserves_datetimeindex(self):
        # GH#41382
        # 创建一个空的日期时间索引
        dti = DatetimeIndex([], dtype="M8[ns]")
        # 创建一个使用空日期时间索引的 DataFrame
        df = DataFrame(index=dti)

        # 创建预期的空日期时间索引
        expected = DatetimeIndex([], dtype="datetime64[ns]", freq=None)

        # 获取转置后的 DataFrame 的索引
        result1 = df.T.sum().index
        # 获取按列求和后的 DataFrame 的索引
        result2 = df.sum(axis=1).index

        # 断言转置后的索引与预期的索引相等
        tm.assert_index_equal(result1, expected)
        tm.assert_index_equal(result2, expected)

    def test_transpose_tzaware_1col_single_tz(self):
        # GH#26825
        # 创建一个带有时区信息的日期时间索引
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")

        # 创建一个使用日期时间索引的 DataFrame
        df = DataFrame(dti)
        # 断言 DataFrame 的所有列都是相同的数据类型
        assert (df.dtypes == dti.dtype).all()
        # 对 DataFrame 进行转置操作
        res = df.T
        # 断言转置后的 DataFrame 的所有列都是相同的数据类型
        assert (res.dtypes == dti.dtype).all()

    def test_transpose_tzaware_2col_single_tz(self):
        # GH#26825
        # 创建一个带有时区信息的日期时间索引
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")

        # 创建一个包含两列的 DataFrame，每列使用相同的日期时间索引
        df3 = DataFrame({"A": dti, "B": dti})
        # 断言 DataFrame 的所有列都是相同的数据类型
        assert (df3.dtypes == dti.dtype).all()
        # 对 DataFrame 进行转置操作
        res3 = df3.T
        # 断言转置后的 DataFrame 的所有列都是相同的数据类型
        assert (res3.dtypes == dti.dtype).all()

    def test_transpose_tzaware_2col_mixed_tz(self):
        # GH#26825
        # 创建一个带有时区信息的日期时间索引
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        # 将日期时间索引转换为另一个时区
        dti2 = dti.tz_convert("US/Pacific")

        # 创建一个包含两列的 DataFrame，每列使用不同的日期时间索引
        df4 = DataFrame({"A": dti, "B": dti2})
        # 断言 DataFrame 的所有列分别具有相应的数据类型
        assert (df4.dtypes == [dti.dtype, dti2.dtype]).all()
        # 断言转置后的 DataFrame 的所有列都是对象类型
        assert (df4.T.dtypes == object).all()
        # 使用测试模块的方法断言转置两次后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(df4.T.T, df4.astype(object))

    @pytest.mark.parametrize("tz", [None, "America/New_York"])
    def test_transpose_preserves_dtindex_equality_with_dst(self, tz):
        # GH#19970
        # 创建一个带有夏令时变化的日期时间索引
        idx = date_range("20161101", "20161130", freq="4h", tz=tz)
        # 创建一个包含两列的 DataFrame，每列使用不同的整数序列作为数据
        df = DataFrame({"a": range(len(idx)), "b": range(len(idx))}, index=idx)
        # 对 DataFrame 进行转置操作，比较转置前后的相等性
        result = df.T == df.T
        # 创建预期的结果 DataFrame
        expected = DataFrame(True, index=list("ab"), columns=idx)
        # 断言转置后的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)

    def test_transpose_object_to_tzaware_mixed_tz(self):
        # GH#26825
        # 创建一个带有时区信息的日期时间索引
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        # 将日期时间索引转换为另一个时区
        dti2 = dti.tz_convert("US/Pacific")

        # 创建一个包含两行的 DataFrame，每行是一个日期时间索引对象
        df2 = DataFrame([dti, dti2])
        # 断言 DataFrame 的所有列都是对象类型
        assert (df2.dtypes == object).all()
        # 对 DataFrame 进行转置操作
        res2 = df2.T
        # 断言转置后的 DataFrame 的所有列都是对象类型
        assert (res2.dtypes == object).all()
    # 定义一个测试函数，用于测试 DataFrame 的 uint64 数据类型的转置操作
    def test_transpose_uint64(self):
        # 创建一个 DataFrame，包含两列：A列为从0到2的整数，B列为三个大整数（uint64类型）
        df = DataFrame(
            {"A": np.arange(3), "B": [2**63, 2**63 + 5, 2**63 + 10]},
            dtype=np.uint64,
        )
        # 对 DataFrame 进行转置操作
        result = df.T
        # 创建一个期望的 DataFrame，其值为转置后的 df.values，并设置索引为["A", "B"]
        expected = DataFrame(df.values.T)
        expected.index = ["A", "B"]
        # 使用测试工具函数 tm.assert_frame_equal 检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试 DataFrame 中的浮点数转置操作
    def test_transpose_float(self, float_frame):
        # 从参数中获取一个浮点数类型的 DataFrame
        frame = float_frame
        # 对 DataFrame 进行转置操作
        dft = frame.T
        # 遍历转置后的 DataFrame 的每一列及其对应的 Series
        for idx, series in dft.items():
            for col, value in series.items():
                # 如果某个值是 NaN，则验证原始 DataFrame 中对应位置也是 NaN
                if np.isnan(value):
                    assert np.isnan(frame[col][idx])
                else:
                    # 否则，验证转置后的值与原始 DataFrame 中对应位置的值相等
                    assert value == frame[col][idx]

    # 定义一个测试函数，用于测试混合类型的 DataFrame 转置操作
    def test_transpose_mixed(self):
        # 创建一个混合类型的 DataFrame
        mixed = DataFrame(
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
                "D": bdate_range("1/1/2009", periods=5),
            },
            index=Index(["a", "b", "c", "d", "e"], dtype=object),
        )

        # 对混合类型的 DataFrame 进行转置操作
        mixed_T = mixed.T
        # 遍历转置后的 DataFrame 的每一列及其对应的 Series
        for col, s in mixed_T.items():
            # 验证每一列的数据类型是否为 np.object_
            assert s.dtype == np.object_

    # 定义一个测试函数，用于测试 DataFrame 转置后的视图获取和修改操作
    def test_transpose_get_view(self, float_frame):
        # 对浮点数类型的 DataFrame 进行转置操作
        dft = float_frame.T
        # 修改转置后 DataFrame 的部分数据
        dft.iloc[:, 5:10] = 5
        # 验证修改后的数据是否反映在原始 DataFrame 中
        assert (float_frame.values[5:10] != 5).all()

    # 定义一个测试函数，用于测试转置后的 DataFrame 在日期时间类型中的操作
    def test_transpose_get_view_dt64tzget_view(self):
        # 创建一个带有时区的日期范围
        dti = date_range("2016-01-01", periods=6, tz="US/Pacific")
        # 将日期范围转换为二维数组
        arr = dti._data.reshape(3, 2)
        # 创建一个 DataFrame
        df = DataFrame(arr)
        # 验证 DataFrame 的内部块数为1
        assert df._mgr.nblocks == 1

        # 对 DataFrame 进行转置操作
        result = df.T
        # 验证转置后的 DataFrame 的内部块数为1
        assert result._mgr.nblocks == 1

        # 获取转置后 DataFrame 的内部块数据，并验证与原始 DataFrame 的数据是否共享内存
        rtrip = result._mgr.blocks[0].values
        assert np.shares_memory(df._mgr.blocks[0].values._ndarray, rtrip._ndarray)

    # 定义一个测试函数，用于测试不推断日期时间类型的 DataFrame 转置操作
    def test_transpose_not_inferring_dt(self):
        # 创建一个包含日期时间戳的对象类型 DataFrame
        df = DataFrame(
            {
                "a": [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            },
            dtype=object,
        )
        # 对 DataFrame 进行转置操作
        result = df.T
        # 创建一个期望的 DataFrame，其值为转置后的 df.values，设置列名为[0, 1]，索引为["a"]
        expected = DataFrame(
            [[Timestamp("2019-12-31"), Timestamp("2019-12-31")]],
            columns=[0, 1],
            index=["a"],
            dtype=object,
        )
        # 使用测试工具函数 tm.assert_frame_equal 检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试不推断日期时间类型且包含混合块的 DataFrame 转置操作
    def test_transpose_not_inferring_dt_mixed_blocks(self):
        # 创建一个包含日期时间戳和对象的混合类型 DataFrame
        df = DataFrame(
            {
                "a": Series(
                    [Timestamp("2019-12-31"), Timestamp("2019-12-31")], dtype=object
                ),
                "b": [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            }
        )
        # 对 DataFrame 进行转置操作
        result = df.T
        # 创建一个期望的 DataFrame，其值为转置后的 df.values，设置列名为[0, 1]，索引为["a", "b"]
        expected = DataFrame(
            [
                [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
                [Timestamp("2019-12-31"), Timestamp("2019-12-31")],
            ],
            columns=[0, 1],
            index=["a", "b"],
            dtype=object,
        )
        # 使用测试工具函数 tm.assert_frame_equal 检查结果与期望是否相等
        tm.assert_frame_equal(result, expected)
    # 使用 pytest 的参数化功能，为测试用例提供多组参数组合
    @pytest.mark.parametrize("dtype1", ["Int64", "Float64"])
    @pytest.mark.parametrize("dtype2", ["Int64", "Float64"])
    # 定义一个测试方法，用于测试 DataFrame 的转置操作
    def test_transpose(self, dtype1, dtype2):
        # GH#57315 - transpose should have F contiguous blocks
        # 创建一个 DataFrame 对象，包含两列，每列使用给定的数据类型
        df = DataFrame(
            {
                "a": pd.array([1, 1, 2], dtype=dtype1),
                "b": pd.array([3, 4, 5], dtype=dtype2),
            }
        )
        # 对 DataFrame 进行转置操作
        result = df.T
        # 遍历转置后结果的内部数据块
        for blk in result._mgr.blocks:
            # 当 dtype1 和 dtype2 不同时，blk.values 返回的是 NumPy 对象数组
            data = blk.values._data if dtype1 == dtype2 else blk.values
            # 断言数据块是否按列（Fortran）顺序存储，即是否是 F contiguous
            assert data.flags["F_CONTIGUOUS"]
```