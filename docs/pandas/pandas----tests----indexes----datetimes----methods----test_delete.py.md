# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_delete.py`

```
import pytest  # 导入 pytest 测试框架

from pandas import (  # 导入 pandas 库中的 DatetimeIndex, Series, date_range 函数
    DatetimeIndex,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试模块

class TestDelete:  # 定义测试类 TestDelete
    def test_delete(self, unit):  # 定义测试方法 test_delete，接受参数 unit
        idx = date_range(  # 创建日期范围对象 idx
            start="2000-01-01", periods=5, freq="ME", name="idx", unit=unit
        )

        # 保持频率不变的预期结果
        expected_0 = date_range(  # 创建日期范围对象 expected_0，保持频率不变
            start="2000-02-01", periods=4, freq="ME", name="idx", unit=unit
        )
        expected_4 = date_range(  # 创建日期范围对象 expected_4，保持频率不变
            start="2000-01-01", periods=4, freq="ME", name="idx", unit=unit
        )

        # 将频率重置为 None 的预期结果
        expected_1 = DatetimeIndex(  # 创建 DatetimeIndex 对象 expected_1，将频率重置为 None
            ["2000-01-31", "2000-03-31", "2000-04-30", "2000-05-31"],
            freq=None,
            name="idx",
        ).as_unit(unit)

        cases = {  # 创建测试用例字典 cases
            0: expected_0,
            -5: expected_0,
            -1: expected_4,
            4: expected_4,
            1: expected_1,
        }
        for n, expected in cases.items():  # 遍历测试用例
            result = idx.delete(n)  # 调用 idx 的 delete 方法进行删除操作
            tm.assert_index_equal(result, expected)  # 使用 pandas 测试模块断言索引相等
            assert result.name == expected.name  # 断言结果的名称与预期相同
            assert result.freq == expected.freq  # 断言结果的频率与预期相同

        with pytest.raises((IndexError, ValueError), match="out of bounds"):  # 使用 pytest 的断言捕获异常
            # 取决于 numpy 的版本
            idx.delete(5)  # 尝试删除超出索引边界的元素

    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Pacific"])  # 参数化测试，测试时会传入不同的时区 tz
    def test_delete2(self, tz):  # 定义第二个测试方法 test_delete2，接受参数 tz
        idx = date_range(  # 创建日期范围对象 idx
            start="2000-01-01 09:00", periods=10, freq="h", name="idx", tz=tz
        )

        expected = date_range(  # 创建日期范围对象 expected，删除第一个元素后的预期结果
            start="2000-01-01 10:00", periods=9, freq="h", name="idx", tz=tz
        )
        result = idx.delete(0)  # 调用 idx 的 delete 方法删除第一个元素
        tm.assert_index_equal(result, expected)  # 使用 pandas 测试模块断言索引相等
        assert result.name == expected.name  # 断言结果的名称与预期相同
        assert result.freqstr == "h"  # 断言结果的频率字符串为 "h"
        assert result.tz == expected.tz  # 断言结果的时区与预期相同

        expected = date_range(  # 创建日期范围对象 expected，删除最后一个元素后的预期结果
            start="2000-01-01 09:00", periods=9, freq="h", name="idx", tz=tz
        )
        result = idx.delete(-1)  # 调用 idx 的 delete 方法删除最后一个元素
        tm.assert_index_equal(result, expected)  # 使用 pandas 测试模块断言索引相等
        assert result.name == expected.name  # 断言结果的名称与预期相同
        assert result.freqstr == "h"  # 断言结果的频率字符串为 "h"
        assert result.tz == expected.tz  # 断言结果的时区与预期相同
    # 定义测试方法，用于测试删除切片操作
    def test_delete_slice(self, unit):
        # 创建一个日期范围索引对象，从"2000-01-01"开始，包含10天，频率为每天("D")，名称为"idx"，单位为unit
        idx = date_range(
            start="2000-01-01", periods=10, freq="D", name="idx", unit=unit
        )

        # 保持频率不变的预期结果（删除索引0到2的数据）
        expected_0_2 = date_range(
            start="2000-01-04", periods=7, freq="D", name="idx", unit=unit
        )
        # 保持频率不变的预期结果（删除索引7到9的数据）
        expected_7_9 = date_range(
            start="2000-01-01", periods=7, freq="D", name="idx", unit=unit
        )

        # 将频率重置为None的预期结果（删除索引3到5的数据）
        expected_3_5 = DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
                "2000-01-07",
                "2000-01-08",
                "2000-01-09",
                "2000-01-10",
            ],
            freq=None,
            name="idx",
        ).as_unit(unit)

        # 创建测试用例字典，每个键值对包含索引位置和预期结果
        cases = {
            (0, 1, 2): expected_0_2,
            (7, 8, 9): expected_7_9,
            (3, 4, 5): expected_3_5,
        }
        # 遍历测试用例字典，对每个测试用例执行删除操作并验证结果
        for n, expected in cases.items():
            # 删除指定位置n的索引
            result = idx.delete(n)
            # 断言删除后的索引与预期结果相等
            tm.assert_index_equal(result, expected)
            # 断言删除后的索引名称与预期结果相等
            assert result.name == expected.name
            # 断言删除后的索引频率与预期结果相等
            assert result.freq == expected.freq

            # 删除切片n[0]到n[-1]+1的索引
            result = idx.delete(slice(n[0], n[-1] + 1))
            # 断言删除后的索引与预期结果相等
            tm.assert_index_equal(result, expected)
            # 断言删除后的索引名称与预期结果相等
            assert result.name == expected.name
            # 断言删除后的索引频率与预期结果相等
            assert result.freq == expected.freq

    # TODO: 属于Series.drop测试？
    @pytest.mark.parametrize("tz", [None, "Asia/Tokyo", "US/Pacific"])
    # 定义测试方法，用于测试删除切片操作2
    def test_delete_slice2(self, tz, unit):
        # 创建一个日期范围索引对象，从"2000-01-01 09:00"开始，包含10个小时，频率为每小时("h")，名称为"idx"，时区为tz，单位为unit
        dti = date_range(
            "2000-01-01 09:00", periods=10, freq="h", name="idx", tz=tz, unit=unit
        )
        # 创建一个Series对象，每个元素为1，索引为dti
        ts = Series(
            1,
            index=dti,
        )

        # 保持频率不变的预期结果（删除前5个索引）
        result = ts.drop(ts.index[:5]).index
        expected = dti[5:]
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq
        assert result.tz == expected.tz

        # 将频率重置为None的预期结果（删除索引[1, 3, 5, 7, 9]）
        result = ts.drop(ts.index[[1, 3, 5, 7, 9]]).index
        expected = dti[::2]._with_freq(None)
        tm.assert_index_equal(result, expected)
        assert result.name == expected.name
        assert result.freq == expected.freq
        assert result.tz == expected.tz
```