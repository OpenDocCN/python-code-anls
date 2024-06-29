# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_join.py`

```
# 导入 numpy 库，并为了方便使用起别名 np
import numpy as np
# 导入 pytest 库，用于单元测试
import pytest

# 导入 pandas 库中的异常类 IncompatibleFrequency
from pandas._libs.tslibs import IncompatibleFrequency

# 从 pandas 库中导入以下模块：
# DataFrame：用于操作数据的二维表格
# Index：用于表示 pandas 对象的索引
# PeriodIndex：用于表示一组时间段的索引
# date_range：生成一组日期时间数据
# period_range：生成一组时间段数据
from pandas import (
    DataFrame,
    Index,
    PeriodIndex,
    date_range,
    period_range,
)
# 导入 pandas 内部测试工具集
import pandas._testing as tm


# 定义一个测试类 TestJoin，用于测试 PeriodIndex 的 join 操作
class TestJoin:
    # 定义测试方法 test_join_outer_indexer，测试 _outer_indexer 方法
    def test_join_outer_indexer(self):
        # 创建一个 PeriodIndex 对象 pi，包含从 "1/1/2000" 到 "1/20/2000" 的每日时间段
        pi = period_range("1/1/2000", "1/20/2000", freq="D")

        # 调用 pi 对象的 _outer_indexer 方法，返回结果存入 result
        result = pi._outer_indexer(pi)
        
        # 使用测试工具集中的方法验证 result[0] 等于 pi._values
        tm.assert_extension_array_equal(result[0], pi._values)
        # 使用测试工具集中的方法验证 result[1] 等于一个整数数组，表示长度为 len(pi)
        tm.assert_numpy_array_equal(result[1], np.arange(len(pi), dtype=np.intp))
        # 使用测试工具集中的方法验证 result[2] 等于一个整数数组，表示长度为 len(pi)
        tm.assert_numpy_array_equal(result[2], np.arange(len(pi), dtype=np.intp))

    # 定义测试方法 test_joins，测试 PeriodIndex 的 join 操作，接受一个参数 join_type
    def test_joins(self, join_type):
        # 创建一个 PeriodIndex 对象 index，包含从 "1/1/2000" 到 "1/20/2000" 的每日时间段
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        # 调用 index 对象的 join 方法，将 index 与 index[:-5] 进行连接，连接方式由 join_type 决定
        joined = index.join(index[:-5], how=join_type)

        # 断言 joined 是一个 PeriodIndex 对象
        assert isinstance(joined, PeriodIndex)
        # 断言 joined 的频率与 index 的频率相同
        assert joined.freq == index.freq

    # 定义测试方法 test_join_self，测试 PeriodIndex 的自身 join 操作，接受一个参数 join_type
    def test_join_self(self, join_type):
        # 创建一个 PeriodIndex 对象 index，包含从 "1/1/2000" 到 "1/20/2000" 的每日时间段
        index = period_range("1/1/2000", "1/20/2000", freq="D")

        # 调用 index 对象的 join 方法，将 index 与自身进行连接，连接方式由 join_type 决定
        res = index.join(index, how=join_type)
        
        # 断言连接后的结果对象与原始的 index 对象是同一个对象
        assert index is res

    # 定义测试方法 test_join_does_not_recur，测试 DataFrame 的列索引与行索引的 join 操作
    def test_join_does_not_recur(self):
        # 创建一个 DataFrame 对象 df，包含值全为 1 的 3 行 2 列数据
        df = DataFrame(
            np.ones((3, 2)),
            index=date_range("2020-01-01", periods=3),
            columns=period_range("2020-01-01", periods=2),
        )
        # 选择 df 的前两行第一列，形成一个 Series 对象 ser
        ser = df.iloc[:2, 0]

        # 调用 ser 对象的 index 属性的 join 方法，将 ser 的索引与 df 的列索引进行外连接
        res = ser.index.join(df.columns, how="outer")
        # 创建一个预期的 Index 对象 expected，包含 ser 的索引和 df 的列索引
        expected = Index(
            [ser.index[0], ser.index[1], df.columns[0], df.columns[1]], object
        )
        # 使用测试工具集中的方法验证 res 与 expected 相等
        tm.assert_index_equal(res, expected)

    # 定义测试方法 test_join_mismatched_freq_raises，测试频率不匹配时的 join 操作
    def test_join_mismatched_freq_raises(self):
        # 创建一个 PeriodIndex 对象 index，包含从 "1/1/2000" 到 "1/20/2000" 的每日时间段
        index = period_range("1/1/2000", "1/20/2000", freq="D")
        # 创建一个 PeriodIndex 对象 index3，包含从 "1/1/2000" 到 "1/20/2000" 的每隔两日时间段
        index3 = period_range("1/1/2000", "1/20/2000", freq="2D")
        # 创建一个匹配错误频率的异常消息字符串
        msg = r".*Input has different freq=2D from Period\(freq=D\)"
        # 使用 pytest 的上下文管理语句，检查 join 操作是否引发 IncompatibleFrequency 异常，并匹配异常消息
        with pytest.raises(IncompatibleFrequency, match=msg):
            index.join(index3)
```