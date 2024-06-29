# `D:\src\scipysrc\pandas\pandas\tests\indexes\timedeltas\test_delete.py`

```
from pandas import (
    TimedeltaIndex,
    timedelta_range,
)
import pandas._testing as tm

class TestTimedeltaIndexDelete:
    def test_delete(self):
        # 创建一个时间增量索引，从1天开始，共5个周期，频率为每天一次，命名为"idx"
        idx = timedelta_range(start="1 Days", periods=5, freq="D", name="idx")

        # 保持频率不变的预期结果
        expected_0 = timedelta_range(start="2 Days", periods=4, freq="D", name="idx")
        expected_4 = timedelta_range(start="1 Days", periods=4, freq="D", name="idx")

        # 将频率重置为None的预期结果
        expected_1 = TimedeltaIndex(
            ["1 day", "3 day", "4 day", "5 day"], freq=None, name="idx"
        )

        # 构建测试用例字典，每个键是要测试的索引删除操作，对应的值是预期结果
        cases = {
            0: expected_0,
            -5: expected_0,
            -1: expected_4,
            4: expected_4,
            1: expected_1,
        }
        # 遍历测试用例字典
        for n, expected in cases.items():
            # 执行索引删除操作
            result = idx.delete(n)
            # 断言删除后的索引与预期结果相等
            tm.assert_index_equal(result, expected)
            # 断言删除后的索引名与预期结果相同
            assert result.name == expected.name
            # 断言删除后的索引频率与预期结果相同
            assert result.freq == expected.freq

        # 检查是否会引发索引错误或值错误异常
        with tm.external_error_raised((IndexError, ValueError)):
            # 根据numpy版本的不同可能会引发异常，删除索引5
            idx.delete(5)

    def test_delete_slice(self):
        # 创建一个时间增量索引，从1天开始，共10个周期，频率为每天一次，命名为"idx"
        idx = timedelta_range(start="1 days", periods=10, freq="D", name="idx")

        # 保持频率不变的预期结果
        expected_0_2 = timedelta_range(start="4 days", periods=7, freq="D", name="idx")
        expected_7_9 = timedelta_range(start="1 days", periods=7, freq="D", name="idx")

        # 将频率重置为None的预期结果
        expected_3_5 = TimedeltaIndex(
            ["1 D", "2 D", "3 D", "7 D", "8 D", "9 D", "10D"], freq=None, name="idx"
        )

        # 构建测试用例字典，每个键是要测试的索引删除操作，对应的值是预期结果
        cases = {
            (0, 1, 2): expected_0_2,
            (7, 8, 9): expected_7_9,
            (3, 4, 5): expected_3_5,
        }
        # 遍历测试用例字典
        for n, expected in cases.items():
            # 执行索引删除操作
            result = idx.delete(n)
            # 断言删除后的索引与预期结果相等
            tm.assert_index_equal(result, expected)
            # 断言删除后的索引名与预期结果相同
            assert result.name == expected.name
            # 断言删除后的索引频率与预期结果相同
            assert result.freq == expected.freq

            # 执行切片删除操作
            result = idx.delete(slice(n[0], n[-1] + 1))
            # 断言删除后的索引与预期结果相等
            tm.assert_index_equal(result, expected)
            # 断言删除后的索引名与预期结果相同
            assert result.name == expected.name
            # 断言删除后的索引频率与预期结果相同
            assert result.freq == expected.freq

    def test_delete_doesnt_infer_freq(self):
        # GH#30655 行为与DatetimeIndex匹配

        # 创建一个时间增量索引，包含字符串"1 Day", "2 Days", None, "3 Days", "4 Days"
        tdi = TimedeltaIndex(["1 Day", "2 Days", None, "3 Days", "4 Days"])
        # 执行索引删除操作
        result = tdi.delete(2)
        # 断言删除后的索引频率为None
        assert result.freq is None
```