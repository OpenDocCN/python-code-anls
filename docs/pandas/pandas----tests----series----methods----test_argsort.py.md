# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_argsort.py`

```
import numpy as np  # 导入 NumPy 库，用于数组和数值计算
import pytest  # 导入 Pytest 测试框架

from pandas import (  # 从 Pandas 库中导入以下模块：
    Series,  # 数据结构：Series，用于一维标记数据
    Timestamp,  # 时间戳类，处理日期时间数据
    isna,  # 函数，用于检测缺失值
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestSeriesArgsort:  # 定义测试类 TestSeriesArgsort
    def test_argsort_axis(self):  # 定义测试方法 test_argsort_axis
        # GH#54257
        ser = Series(range(3))  # 创建一个 Series 对象，包含整数范围 [0, 1, 2]

        msg = "No axis named 2 for object type Series"
        with pytest.raises(ValueError, match=msg):  # 使用 Pytest 断言引发 ValueError 异常，并匹配错误信息
            ser.argsort(axis=2)  # 调用 Series 的 argsort 方法，传入 axis=2 参数

    def test_argsort_numpy(self, datetime_series):  # 定义测试方法 test_argsort_numpy，接受 datetime_series 参数
        ser = datetime_series  # 将参数赋值给 ser
        res = np.argsort(ser).values  # 使用 NumPy 对 ser 进行排序，并获取排序后的值
        expected = np.argsort(np.array(ser))  # 对 ser 的 NumPy 数组版本进行排序
        tm.assert_numpy_array_equal(res, expected)  # 使用 Pandas 测试模块检查 res 和 expected 是否相等

    def test_argsort_numpy_missing(self):  # 定义测试方法 test_argsort_numpy_missing
        data = [0.1, np.nan, 0.2, np.nan, 0.3]  # 创建包含浮点数和 NaN 的列表
        ser = Series(data)  # 创建 Series 对象
        result = np.argsort(ser)  # 使用 NumPy 对 Series 进行排序
        expected = np.argsort(np.array(data))  # 对原始数据的 NumPy 数组版本进行排序

        tm.assert_numpy_array_equal(result.values, expected)  # 使用 Pandas 测试模块检查排序结果是否符合预期

    def test_argsort(self, datetime_series):  # 定义测试方法 test_argsort，接受 datetime_series 参数
        argsorted = datetime_series.argsort()  # 对 datetime_series 进行排序
        assert issubclass(argsorted.dtype.type, np.integer)  # 断言排序后的数据类型为 np.integer

    def test_argsort_dt64(self, unit):  # 定义测试方法 test_argsort_dt64，接受 unit 参数
        # GH#2967 (introduced bug in 0.11-dev I think)
        ser = Series(  # 创建一个 Series 对象，包含日期时间戳
            [Timestamp(f"201301{i:02d}") for i in range(1, 6)], dtype=f"M8[{unit}]"
        )
        assert ser.dtype == f"datetime64[{unit}]"  # 断言 Series 的 dtype 符合指定的日期时间格式
        shifted = ser.shift(-1)  # 对 Series 进行向前位移操作
        assert shifted.dtype == f"datetime64[{unit}]"  # 断言位移后的 dtype 符合指定的日期时间格式
        assert isna(shifted[4])  # 断言位移后的第五个元素是否为 NaN

        result = ser.argsort()  # 对原始 Series 进行排序
        expected = Series(range(5), dtype=np.intp)  # 创建预期的排序结果 Series
        tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块检查排序结果是否符合预期

        result = shifted.argsort()  # 对位移后的 Series 进行排序
        expected = Series(list(range(4)) + [4], dtype=np.intp)  # 创建预期的排序结果 Series
        tm.assert_series_equal(result, expected)  # 使用 Pandas 测试模块检查排序结果是否符合预期

    def test_argsort_stable(self):  # 定义测试方法 test_argsort_stable
        ser = Series(np.random.default_rng(2).integers(0, 100, size=10000))  # 创建包含随机整数的 Series 对象
        mindexer = ser.argsort(kind="mergesort")  # 使用 mergesort 稳定排序对 Series 进行索引排序
        qindexer = ser.argsort()  # 使用默认的 quicksort 排序对 Series 进行索引排序

        mexpected = np.argsort(ser.values, kind="mergesort")  # 对 Series 数据的 NumPy 数组版本进行 mergesort 排序
        qexpected = np.argsort(ser.values, kind="quicksort")  # 对 Series 数据的 NumPy 数组版本进行 quicksort 排序

        tm.assert_series_equal(mindexer.astype(np.intp), Series(mexpected))  # 使用 Pandas 测试模块检查排序结果是否符合预期
        tm.assert_series_equal(qindexer.astype(np.intp), Series(qexpected))  # 使用 Pandas 测试模块检查排序结果是否符合预期
        msg = (
            r"ndarray Expected type <class 'numpy\.ndarray'>, "  # 定义错误信息的正则表达式
            r"found <class 'pandas\.core\.series\.Series'> instead"
        )
        with pytest.raises(AssertionError, match=msg):  # 使用 Pytest 断言引发 AssertionError 异常，并匹配错误信息
            tm.assert_numpy_array_equal(qindexer, mindexer)  # 使用 Pandas 测试模块检查两个排序结果是否相等

    def test_argsort_preserve_name(self, datetime_series):  # 定义测试方法 test_argsort_preserve_name，接受 datetime_series 参数
        result = datetime_series.argsort()  # 对 datetime_series 进行排序
        assert result.name == datetime_series.name  # 断言排序后的 Series 名称与原始 Series 相同
```