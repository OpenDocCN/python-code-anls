# `D:\src\scipysrc\pandas\pandas\tests\series\test_subclass.py`

```
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

# 标记 pytest 的警告过滤器，忽略特定的 DeprecationWarning
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)

# 定义一个测试类 TestSeriesSubclassing
class TestSeriesSubclassing:
    
    # 参数化测试：测试索引操作的切片
    @pytest.mark.parametrize(
        "idx_method, indexer, exp_data, exp_idx",
        [
            ["loc", ["a", "b"], [1, 2], "ab"],  # 使用 loc 方法，索引为 ['a', 'b']，期望数据为 [1, 2]，期望索引为 'ab'
            ["iloc", [2, 3], [3, 4], "cd"],    # 使用 iloc 方法，索引为 [2, 3]，期望数据为 [3, 4]，期望索引为 'cd'
        ],
    )
    def test_indexing_sliced(self, idx_method, indexer, exp_data, exp_idx):
        # 创建一个 SubclassedSeries 对象 s，数据为 [1, 2, 3, 4]，索引为 ['a', 'b', 'c', 'd']
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"))
        # 执行索引操作，并获取结果
        res = getattr(s, idx_method)[indexer]
        # 创建期望的 SubclassedSeries 对象，数据和索引与预期一致
        exp = tm.SubclassedSeries(exp_data, index=list(exp_idx))
        # 使用测试框架断言两个 Series 对象是否相等
        tm.assert_series_equal(res, exp)

    # 测试 Series 对象转为 DataFrame 对象
    def test_to_frame(self):
        # 创建一个 SubclassedSeries 对象 s，数据为 [1, 2, 3, 4]，索引为 ['a', 'b', 'c', 'd']，名称为 "xxx"
        s = tm.SubclassedSeries([1, 2, 3, 4], index=list("abcd"), name="xxx")
        # 将 Series 对象转为 DataFrame 对象
        res = s.to_frame()
        # 创建期望的 SubclassedDataFrame 对象，数据为 {"xxx": [1, 2, 3, 4]}，索引为 ['a', 'b', 'c', 'd']
        exp = tm.SubclassedDataFrame({"xxx": [1, 2, 3, 4]}, index=list("abcd"))
        # 使用测试框架断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(res, exp)

    # 测试 Series 对象执行 unstack 操作
    def test_subclass_unstack(self):
        # 创建一个 SubclassedSeries 对象 s，数据为 [1, 2, 3, 4]，索引为 [['a', 'a', 'b', 'b'], ['x', 'y', 'x', 'y']]
        s = tm.SubclassedSeries([1, 2, 3, 4], index=[list("aabb"), list("xyxy")])
        # 执行 unstack 操作
        res = s.unstack()
        # 创建期望的 SubclassedDataFrame 对象，数据为 {"x": [1, 3], "y": [2, 4]}，索引为 ['a', 'b']
        exp = tm.SubclassedDataFrame({"x": [1, 3], "y": [2, 4]}, index=["a", "b"])
        # 使用测试框架断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(res, exp)

    # 测试 SubclassedSeries 对象的空表示
    def test_subclass_empty_repr(self):
        # 创建一个空的 SubclassedSeries 对象
        sub_series = tm.SubclassedSeries()
        # 使用断言检查对象的 __repr__ 方法是否包含 "SubclassedSeries" 字符串
        assert "SubclassedSeries" in repr(sub_series)

    # 测试 Series 对象的 asof 方法
    def test_asof(self):
        N = 3
        # 创建一个 SubclassedSeries 对象 s，数据为 {"A": [NaN, NaN, NaN]}，索引为指定的日期时间索引 rng
        rng = pd.date_range("1/1/1990", periods=N, freq="53s")
        s = tm.SubclassedSeries({"A": [np.nan, np.nan, np.nan]}, index=rng)
        # 执行 asof 方法，获取结果
        result = s.asof(rng[-2:])
        # 使用断言检查结果是否为 SubclassedSeries 对象
        assert isinstance(result, tm.SubclassedSeries)

    # 测试 Series 对象的 explode 方法
    def test_explode(self):
        # 创建一个 SubclassedSeries 对象 s，数据包含嵌套列表 [[1, 2, 3], "foo", [], [3, 4]]
        s = tm.SubclassedSeries([[1, 2, 3], "foo", [], [3, 4]])
        # 执行 explode 方法
        result = s.explode()
        # 使用断言检查结果是否为 SubclassedSeries 对象
        assert isinstance(result, tm.SubclassedSeries)

    # 测试 Series 对象的 equals 方法
    def test_equals(self):
        # https://github.com/pandas-dev/pandas/pull/34402
        # 创建两个 Series 对象 s1 和 s2，分别使用不同的构造函数
        s1 = pd.Series([1, 2, 3])
        s2 = tm.SubclassedSeries([1, 2, 3])
        # 使用断言检查两个 Series 对象是否相等
        assert s1.equals(s2)
        assert s2.equals(s1)


# 定义一个 SubclassedSeries 类，继承自 pandas 的 Series 类
class SubclassedSeries(pd.Series):
    # 定义 _constructor 属性，返回一个新的构造函数
    @property
    def _constructor(self):
        def _new(*args, **kwargs):
            # 自定义的构造函数逻辑，根据 Series 的名称决定返回的对象类型
            if self.name == "test":
                return pd.Series(*args, **kwargs)
            return SubclassedSeries(*args, **kwargs)

        return _new


# 测试从字典构造 SubclassedSeries 对象的函数
def test_constructor_from_dict():
    # https://github.com/pandas-dev/pandas/issues/52445
    # 使用字典构造 SubclassedSeries 对象
    result = SubclassedSeries({"a": 1, "b": 2, "c": 3})
    # 使用断言检查结果是否为 SubclassedSeries 对象
    assert isinstance(result, SubclassedSeries)
```