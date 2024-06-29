# `D:\src\scipysrc\pandas\pandas\tests\indexes\numeric\test_setops.py`

```
# 导入必要的模块和库
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

# 导入 pandas 内部测试模块
import pandas._testing as tm
# 导入 pandas 核心索引相关 API
from pandas.core.indexes.api import (
    Index,
    RangeIndex,
)

@pytest.fixture
def index_large():
    # 用于 TestUInt64Index 测试的大数值，不需要与 int64/float64 兼容
    large = [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25]
    return Index(large, dtype=np.uint64)

class TestSetOps:
    @pytest.mark.parametrize("dtype", ["f8", "u8", "i8"])
    def test_union_non_numeric(self, dtype):
        # 边界情况，非数值型
        index = Index(np.arange(5, dtype=dtype), dtype=dtype)
        assert index.dtype == dtype

        # 创建一个包含 datetime 对象的索引
        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        # 执行 union 操作
        result = index.union(other)
        expected = Index(np.concatenate((index, other)))
        # 断言索引相等
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    def test_intersection(self):
        # 创建一个整数索引
        index = Index(range(5), dtype=np.int64)

        other = Index([1, 2, 3, 4, 5])
        # 执行 intersection 操作
        result = index.intersection(other)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    def test_int_float_union_dtype(self, dtype):
        # https://github.com/pandas-dev/pandas/issues/26778
        # [u]int | float -> float
        # 创建一个整数索引和一个浮点数索引，测试它们的 union 操作
        index = Index([0, 2, 3], dtype=dtype)
        other = Index([0.5, 1.5], dtype=np.float64)
        expected = Index([0.0, 0.5, 1.5, 2.0, 3.0], dtype=np.float64)
        result = index.union(other)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_float_union_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/26778
        # 创建一个范围索引和一个浮点数索引，测试它们的 union 操作
        index = RangeIndex(start=0, stop=3)
        other = Index([0.5, 1.5], dtype=np.float64)
        result = index.union(other)
        expected = Index([0.0, 0.5, 1, 1.5, 2.0], dtype=np.float64)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_uint64_union_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/26778
        # 创建一个范围索引和一个 uint64 类型的索引，测试它们的 union 操作
        index = RangeIndex(start=0, stop=3)
        other = Index([0, 10], dtype=np.uint64)
        result = index.union(other)
        expected = Index([0, 1, 2, 10], dtype=object)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)
    # 测试 Index 对象的浮点数索引差异功能
    def test_float64_index_difference(self):
        # 创建一个浮点数类型的 Index 对象
        float_index = Index([1.0, 2, 3])
        # 创建一个字符串类型的 Index 对象
        string_index = Index(["1", "2", "3"])

        # 求两个 Index 对象的差异
        result = float_index.difference(string_index)
        # 断言结果与 float_index 相同
        tm.assert_index_equal(result, float_index)

        # 求两个 Index 对象的差异（反向）
        result = string_index.difference(float_index)
        # 断言结果与 string_index 相同
        tm.assert_index_equal(result, string_index)

    # 测试 Index 对象的交集功能，包括超出 int64 范围的 uint64 类型索引
    def test_intersection_uint64_outside_int64_range(self):
        # 创建一个包含超出 int64 范围的 uint64 类型索引对象
        index_large = Index(
            [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25],
            dtype=np.uint64,
        )
        # 创建另一个 Index 对象
        other = Index([2**63, 2**63 + 5, 2**63 + 10, 2**63 + 15, 2**63 + 20])
        
        # 求两个 Index 对象的交集
        result = index_large.intersection(other)
        # 创建预期结果的 Index 对象，使用 numpy 函数处理
        expected = Index(np.sort(np.intersect1d(index_large.values, other.values)))
        tm.assert_index_equal(result, expected)

        # 求两个 Index 对象的交集（反向）
        result = other.intersection(index_large)
        # 创建预期结果的 Index 对象，使用 numpy 函数处理
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index_large.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    # 使用 pytest 参数化装饰器，测试 Index 对象的交集功能，验证是否单调
    @pytest.mark.parametrize(
        "index2_name,keeps_name",
        [
            ("index", True),  # index2_name 为 "index"，keeps_name 为 True
            ("other", False),  # index2_name 为 "other"，keeps_name 为 False
        ],
    )
    def test_intersection_monotonic(self, index2_name, keeps_name, sort):
        # 创建两个 Index 对象，并设置名称
        index2 = Index([4, 7, 6, 5, 3], name=index2_name)
        index1 = Index([5, 3, 2, 4, 1], name="index")
        expected = Index([5, 3, 4])

        # 如果 keeps_name 为 True，则设置预期结果的名称为 "index"
        if keeps_name:
            expected.name = "index"

        # 求两个 Index 对象的交集，并根据 sort 参数进行排序
        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    # 测试 Index 对象的对称差异功能
    def test_symmetric_difference(self, sort):
        # 创建两个 Index 对象，其中一个带有名称 "index1"
        index1 = Index([5, 2, 3, 4], name="index1")
        index2 = Index([2, 3, 4, 1])
        # 求两个 Index 对象的对称差异，并根据 sort 参数进行排序
        result = index1.symmetric_difference(index2, sort=sort)
        expected = Index([5, 1])
        if sort is not None:
            tm.assert_index_equal(result, expected)
        else:
            tm.assert_index_equal(result, expected.sort_values())
        # 断言结果对象的名称为 None
        assert result.name is None
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)
class TestSetOpsSort:
    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_other_special(self, slice_):
        # 测试用例链接：https://github.com/pandas-dev/pandas/issues/24959
        # 创建一个索引对象 idx 包含元素 [1, 0, 2]
        idx = Index([1, 0, 2])
        
        # 使用给定的切片参数 slice_ 对索引进行切片操作
        other = idx[slice_]
        
        # 断言 idx.union(other) 的结果与 idx 相等
        tm.assert_index_equal(idx.union(other), idx)
        
        # 断言 other.union(idx) 的结果与 idx 相等
        tm.assert_index_equal(other.union(idx), idx)
        
        # 使用 sort=False 参数进行 union 操作，并断言结果与 idx 相等
        tm.assert_index_equal(idx.union(other, sort=False), idx)

    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_special_true(self, slice_):
        # 创建一个索引对象 idx 包含元素 [1, 0, 2]
        idx = Index([1, 0, 2])
        
        # 使用给定的切片参数 slice_ 对索引进行切片操作
        other = idx[slice_]

        # 使用 sort=True 参数进行 union 操作
        result = idx.union(other, sort=True)
        
        # 预期的排序后索引对象，包含元素 [0, 1, 2]
        expected = Index([0, 1, 2])
        
        # 断言 union 操作后的结果与预期的排序后索引对象相等
        tm.assert_index_equal(result, expected)
```