# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_get_level_values.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值计算

import pandas as pd  # 导入 Pandas 库，用于数据分析和操作
from pandas import (  # 从 Pandas 库中导入以下模块
    CategoricalIndex,  # 分类索引
    Index,  # 索引
    MultiIndex,  # 多级索引
    Timestamp,  # 时间戳
    date_range,  # 生成日期范围
)
import pandas._testing as tm  # 导入 Pandas 测试模块

class TestGetLevelValues:
    def test_get_level_values_box_datetime64(self):
        dates = date_range("1/1/2000", periods=4)  # 生成一个日期范围，从 2000-01-01 开始，4个周期
        levels = [dates, [0, 1]]  # 定义多级索引的层级
        codes = [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]]  # 定义多级索引的代码

        index = MultiIndex(levels=levels, codes=codes)  # 创建多级索引对象

        assert isinstance(index.get_level_values(0)[0], Timestamp)  # 断言第一个层级值为时间戳类型


def test_get_level_values(idx):
    result = idx.get_level_values(0)  # 获取第一个层级的值
    expected = Index(["foo", "foo", "bar", "baz", "qux", "qux"], name="first")  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同
    assert result.name == "first"  # 断言结果的名称为 "first"

    result = idx.get_level_values("first")  # 通过层级名称获取层级值
    expected = idx.get_level_values(0)  # 预期的索引值与通过位置获取的结果相同
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同

    # GH 10460
    index = MultiIndex(
        levels=[CategoricalIndex(["A", "B"]), CategoricalIndex([1, 2, 3])],  # 创建多级索引，包含分类索引
        codes=[np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 2, 0, 1, 2])],  # 设置多级索引的代码
    )

    exp = CategoricalIndex(["A", "A", "A", "B", "B", "B"])  # 预期的分类索引
    tm.assert_index_equal(index.get_level_values(0), exp)  # 断言第一个层级值与预期相同
    exp = CategoricalIndex([1, 2, 3, 1, 2, 3])  # 预期的分类索引
    tm.assert_index_equal(index.get_level_values(1), exp)  # 断言第二个层级值与预期相同


def test_get_level_values_all_na():
    # GH#17924 当层级完全由 NaN 组成时
    arrays = [[np.nan, np.nan, np.nan], ["a", np.nan, 1]]  # 设置包含 NaN 的数组
    index = MultiIndex.from_arrays(arrays)  # 从数组创建多级索引
    result = index.get_level_values(0)  # 获取第一个层级的值
    expected = Index([np.nan, np.nan, np.nan], dtype=np.float64)  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同

    result = index.get_level_values(1)  # 获取第二个层级的值
    expected = Index(["a", np.nan, 1], dtype=object)  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同


def test_get_level_values_int_with_na():
    # GH#17924
    arrays = [["a", "b", "b"], [1, np.nan, 2]]  # 设置包含 NaN 的整数数组
    index = MultiIndex.from_arrays(arrays)  # 从数组创建多级索引
    result = index.get_level_values(1)  # 获取第二个层级的值
    expected = Index([1, np.nan, 2])  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同

    arrays = [["a", "b", "b"], [np.nan, np.nan, 2]]  # 设置包含 NaN 的整数数组
    index = MultiIndex.from_arrays(arrays)  # 从数组创建多级索引
    result = index.get_level_values(1)  # 获取第二个层级的值
    expected = Index([np.nan, np.nan, 2])  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同


def test_get_level_values_na():
    arrays = [[np.nan, np.nan, np.nan], ["a", np.nan, 1]]  # 设置包含 NaN 的数组
    index = MultiIndex.from_arrays(arrays)  # 从数组创建多级索引
    result = index.get_level_values(0)  # 获取第一个层级的值
    expected = Index([np.nan, np.nan, np.nan])  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同

    result = index.get_level_values(1)  # 获取第二个层级的值
    expected = Index(["a", np.nan, 1])  # 预期的索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同

    arrays = [["a", "b", "b"], pd.DatetimeIndex([0, 1, pd.NaT])]  # 设置包含 NaN 的日期时间数组
    index = MultiIndex.from_arrays(arrays)  # 从数组创建多级索引
    result = index.get_level_values(1)  # 获取第二个层级的值
    expected = pd.DatetimeIndex([0, 1, pd.NaT])  # 预期的日期时间索引值
    tm.assert_index_equal(result, expected)  # 断言结果与预期相同

    arrays = [[], []]  # 空数组
    index = MultiIndex.from_arrays(arrays)  # 从数组创建多级索引
    # 获取索引的第一级值并赋给变量 result
    result = index.get_level_values(0)
    # 创建一个空的索引对象，并指定其数据类型为 object
    expected = Index([], dtype=object)
    # 使用断言函数检查变量 result 和变量 expected 是否相等
    tm.assert_index_equal(result, expected)
def test_get_level_values_when_periods():
    # GH33131. See also discussion in GH32669.
    # This test can probably be removed when PeriodIndex._engine is removed.
    # 从 pandas 库导入 Period 和 PeriodIndex 类
    from pandas import (
        Period,
        PeriodIndex,
    )

    # 创建一个 MultiIndex，包含一个 PeriodIndex，每个 Period 包含 "2019Q1" 和 "2019Q2" 两个时期
    idx = MultiIndex.from_arrays(
        [PeriodIndex([Period("2019Q1"), Period("2019Q2")], name="b")]
    )

    # 使用列表推导式创建 idx2，其中每个元素是对应 idx 中级别的 level_values
    idx2 = MultiIndex.from_arrays(
        [idx._get_level_values(level) for level in range(idx.nlevels)]
    )

    # 断言 idx2.levels 中的每个 MultiIndex 级别都是单调递增的
    assert all(x.is_monotonic_increasing for x in idx2.levels)


def test_values_loses_freq_of_underlying_index():
    # GH#49054
    # 创建一个包含三个工作日的 DatetimeIndex，从 "20200101" 开始
    idx = pd.DatetimeIndex(date_range("20200101", periods=3, freq="BME"))
    
    # 深拷贝 idx 到 expected
    expected = idx.copy(deep=True)
    
    # 创建一个简单的 Index 对象，包含整数 [1, 2, 3]
    idx2 = Index([1, 2, 3])
    
    # 创建一个 MultiIndex 对象 midx，包含 levels 为 [idx, idx2]，codes 分别为 [0, 1, 2] 和 [0, 1, 2]
    midx = MultiIndex(levels=[idx, idx2], codes=[[0, 1, 2], [0, 1, 2]])
    
    # 访问 midx 的 values 属性
    midx.values
    
    # 断言 idx 的频率不为 None
    assert idx.freq is not None
    
    # 使用 tm.assert_index_equal 断言 idx 和 expected 相等
    tm.assert_index_equal(idx, expected)
```