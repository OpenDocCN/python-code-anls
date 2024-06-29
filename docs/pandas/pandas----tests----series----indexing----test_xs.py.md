# `D:\src\scipysrc\pandas\pandas\tests\series\indexing\test_xs.py`

```
import numpy as np  # 导入 numpy 库，用于科学计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    MultiIndex,  # 多级索引对象
    Series,  # 系列对象，类似于带标签的一维数组
    date_range,  # 生成时间序列
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块


def test_xs_datetimelike_wrapping():
    # GH#31630 不应该在这种情况下将 datetime64 包装成 Timestamp 的一个案例
    arr = date_range("2016-01-01", periods=3)._data._ndarray  # 创建一个日期范围的 ndarray 数组

    ser = Series(arr, dtype=object)  # 用 arr 创建一个 dtype 为 object 的 Series 对象
    for i in range(len(ser)):
        ser.iloc[i] = arr[i]  # 将 arr 的每个元素赋值给 ser 的对应位置
    assert ser.dtype == object  # 确保 ser 的 dtype 为 object
    assert isinstance(ser[0], np.datetime64)  # 确保 ser 的第一个元素是 np.datetime64 类型

    result = ser.xs(0)  # 获取 ser 的第一个元素
    assert isinstance(result, np.datetime64)  # 确保结果 result 是 np.datetime64 类型


class TestXSWithMultiIndex:
    def test_xs_level_series(self, multiindex_dataframe_random_data):
        df = multiindex_dataframe_random_data  # 从测试数据中获取多级索引的 DataFrame
        ser = df["A"]  # 获取 DataFrame 中的 "A" 列
        expected = ser[:, "two"]  # 选择 "two" 级别的所有数据作为期望结果
        result = df.xs("two", level=1)["A"]  # 使用 xs 方法获取 "two" 级别的 "A" 列数据
        tm.assert_series_equal(result, expected)  # 使用测试工具比较结果和期望值

    def test_series_getitem_multiindex_xs_by_label(self):
        # GH#5684 多级索引中使用 xs 方法根据标签获取 Series
        idx = MultiIndex.from_tuples(
            [("a", "one"), ("a", "two"), ("b", "one"), ("b", "two")]
        )  # 创建一个多级索引对象 idx
        ser = Series([1, 2, 3, 4], index=idx)  # 创建一个具有多级索引的 Series 对象
        return_value = ser.index.set_names(["L1", "L2"], inplace=True)  # 设置索引的名称
        assert return_value is None  # 确保返回值为 None
        expected = Series([1, 3], index=["a", "b"])  # 创建期望的 Series 对象
        return_value = expected.index.set_names(["L1"], inplace=True)  # 设置期望的索引名称
        assert return_value is None  # 确保返回值为 None

        result = ser.xs("one", level="L2")  # 使用 xs 方法获取 "one" 标签的数据
        tm.assert_series_equal(result, expected)  # 使用测试工具比较结果和期望值

    def test_series_getitem_multiindex_xs(self):
        # GH#6258 多级索引中使用 xs 方法获取 Series
        dt = list(date_range("20130903", periods=3))  # 生成一个日期范围的列表
        idx = MultiIndex.from_product([list("AB"), dt])  # 创建一个产品形式的多级索引
        ser = Series([1, 3, 4, 1, 3, 4], index=idx)  # 创建一个具有多级索引的 Series 对象
        expected = Series([1, 1], index=list("AB"))  # 创建期望的 Series 对象

        result = ser.xs("20130903", level=1)  # 使用 xs 方法获取 "20130903" 日期的数据
        tm.assert_series_equal(result, expected)  # 使用测试工具比较结果和期望值

    def test_series_xs_droplevel_false(self):
        # GH: 19056 在多级索引中使用 xs 方法，设置 drop_level=False
        mi = MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "x")], names=["level1", "level2"]
        )  # 创建一个具有命名的多级索引对象
        ser = Series([1, 1, 1], index=mi)  # 创建一个具有多级索引的 Series 对象
        result = ser.xs("a", axis=0, drop_level=False)  # 使用 xs 方法获取 "a" 标签的数据，保留索引级别
        expected = Series(
            [1, 1],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y")], names=["level1", "level2"]
            ),
        )  # 创建期望的 Series 对象
        tm.assert_series_equal(result, expected)  # 使用测试工具比较结果和期望值

    def test_xs_key_as_list(self):
        # GH#41760 多级索引中使用 xs 方法，键值为列表时抛出 TypeError 异常
        mi = MultiIndex.from_tuples([("a", "x")], names=["level1", "level2"])  # 创建一个具有命名的多级索引对象
        ser = Series([1], index=mi)  # 创建一个具有多级索引的 Series 对象
        with pytest.raises(TypeError, match="list keys are not supported"):
            ser.xs(["a", "x"], axis=0, drop_level=False)  # 使用 xs 方法尝试使用列表作为键值抛出 TypeError 异常

        with pytest.raises(TypeError, match="list keys are not supported"):
            ser.xs(["a"], axis=0, drop_level=False)  # 使用 xs 方法尝试使用列表作为键值抛出 TypeError 异常
```