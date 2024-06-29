# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_unique.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算

from pandas import (  # 从 Pandas 库中导入以下模块：
    Categorical,  # 用于处理分类数据的类
    IntervalIndex,  # 用于处理间隔索引的类
    Series,  # Pandas 中的基本数据结构，类似于带标签的一维数组
    date_range,  # 生成日期范围的函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于测试功能的一致性和正确性


class TestUnique:
    def test_unique_uint64(self):
        ser = Series([1, 2, 2**63, 2**63], dtype=np.uint64)  # 创建一个 Pandas Series 对象，包含无符号 64 位整数
        res = ser.unique()  # 获取 Series 中唯一值的 ndarray
        exp = np.array([1, 2, 2**63], dtype=np.uint64)  # 期望的唯一值数组
        tm.assert_numpy_array_equal(res, exp)  # 使用测试模块验证结果与期望是否相等

    def test_unique_data_ownership(self):
        # it works! GH#1807
        Series(Series(["a", "c", "b"]).unique()).sort_values()  # 创建一个 Series 对象，对其唯一值进行排序

    def test_unique(self):
        # GH#714 also, dtype=float
        ser = Series([1.2345] * 100)  # 创建一个包含重复浮点数的 Series 对象
        ser[::2] = np.nan  # 将每隔一个元素设为 NaN
        result = ser.unique()  # 获取 Series 中的唯一值
        assert len(result) == 2  # 断言结果中的唯一值个数为 2

        # explicit f4 dtype
        ser = Series([1.2345] * 100, dtype="f4")  # 创建一个指定数据类型为单精度浮点数的 Series 对象
        ser[::2] = np.nan  # 将每隔一个元素设为 NaN
        result = ser.unique()  # 获取 Series 中的唯一值
        assert len(result) == 2  # 断言结果中的唯一值个数为 2

    def test_unique_nan_object_dtype(self):
        # NAs in object arrays GH#714
        ser = Series(["foo"] * 100, dtype="O")  # 创建一个包含对象类型数据的 Series 对象
        ser[::2] = np.nan  # 将每隔一个元素设为 NaN
        result = ser.unique()  # 获取 Series 中的唯一值
        assert len(result) == 2  # 断言结果中的唯一值个数为 2

    def test_unique_none(self):
        # decision about None
        ser = Series([1, 2, 3, None, None, None], dtype=object)  # 创建一个包含 None 的 Series 对象
        result = ser.unique()  # 获取 Series 中的唯一值
        expected = np.array([1, 2, 3, None], dtype=object)  # 期望的唯一值数组
        tm.assert_numpy_array_equal(result, expected)  # 使用测试模块验证结果与期望是否相等

    def test_unique_categorical(self):
        # GH#18051
        cat = Categorical([])  # 创建一个空的分类变量对象
        ser = Series(cat)  # 创建一个包含分类变量的 Series 对象
        result = ser.unique()  # 获取 Series 中的唯一值
        tm.assert_categorical_equal(result, cat)  # 使用测试模块验证分类变量的一致性

        cat = Categorical([np.nan])  # 创建一个包含 NaN 的分类变量对象
        ser = Series(cat)  # 创建一个包含分类变量的 Series 对象
        result = ser.unique()  # 获取 Series 中的唯一值
        tm.assert_categorical_equal(result, cat)  # 使用测试模块验证分类变量的一致性

    def test_tz_unique(self):
        # GH 46128
        dti1 = date_range("2016-01-01", periods=3)  # 创建一个日期范围对象
        ii1 = IntervalIndex.from_breaks(dti1)  # 使用日期范围创建间隔索引对象
        ser1 = Series(ii1)  # 创建一个包含间隔索引的 Series 对象
        uni1 = ser1.unique()  # 获取 Series 中的唯一值
        tm.assert_interval_array_equal(ser1.array, uni1)  # 使用测试模块验证间隔数组的一致性

        dti2 = date_range("2016-01-01", periods=3, tz="US/Eastern")  # 创建一个带时区的日期范围对象
        ii2 = IntervalIndex.from_breaks(dti2)  # 使用带时区的日期范围创建间隔索引对象
        ser2 = Series(ii2)  # 创建一个包含间隔索引的 Series 对象
        uni2 = ser2.unique()  # 获取 Series 中的唯一值
        tm.assert_interval_array_equal(ser2.array, uni2)  # 使用测试模块验证间隔数组的一致性

        assert uni1.dtype != uni2.dtype  # 断言两个唯一值数组的数据类型不同
```