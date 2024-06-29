# `D:\src\scipysrc\pandas\pandas\tests\indexes\base_class\test_indexing.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

from pandas._libs import index as libindex  # 从 Pandas 私有库中导入 index 模块

import pandas as pd  # 导入 Pandas 库并用 pd 别名表示
from pandas import (  # 从 Pandas 导入 Index 和 NaT 类
    Index,
    NaT,
)
import pandas._testing as tm  # 导入 Pandas 测试模块作为 tm 别名

class TestGetSliceBounds:  # 定义测试类 TestGetSliceBounds
    @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])  # 使用 Pytest 的参数化装饰器，指定两组参数
    def test_get_slice_bounds_within(self, side, expected):
        index = Index(list("abcdef"))  # 创建 Index 对象，包含字符 'abcdef'
        result = index.get_slice_bound("e", side=side)  # 调用 Index 对象的 get_slice_bound 方法
        assert result == expected  # 断言结果符合预期

    @pytest.mark.parametrize("side", ["left", "right"])  # 参数化装饰器，指定 side 参数的两个值
    @pytest.mark.parametrize(
        "data, bound, expected", [(list("abcdef"), "x", 6), (list("bcdefg"), "a", 0)]
    )
    def test_get_slice_bounds_outside(self, side, expected, data, bound):
        index = Index(data)  # 创建 Index 对象，使用传入的 data 参数
        result = index.get_slice_bound(bound, side=side)  # 调用 Index 对象的 get_slice_bound 方法
        assert result == expected  # 断言结果符合预期

    def test_get_slice_bounds_invalid_side(self):
        with pytest.raises(ValueError, match="Invalid value for side kwarg"):  # 使用 Pytest 的上下文管理断言抛出特定异常
            Index([]).get_slice_bound("a", side="middle")  # 调用 Index 对象的 get_slice_bound 方法


class TestGetIndexerNonUnique:  # 定义测试类 TestGetIndexerNonUnique
    def test_get_indexer_non_unique_dtype_mismatch(self):
        # GH#25459
        indexes, missing = Index(["A", "B"]).get_indexer_non_unique(Index([0]))  # 调用 Index 对象的 get_indexer_non_unique 方法
        tm.assert_numpy_array_equal(np.array([-1], dtype=np.intp), indexes)  # 使用 Pandas 测试模块的方法进行数组相等性断言
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), missing)

    @pytest.mark.parametrize(
        "idx_values,idx_non_unique",
        [
            ([np.nan, 100, 200, 100], [np.nan, 100]),  # 参数化装饰器，指定两组参数
            ([np.nan, 100.0, 200.0, 100.0], [np.nan, 100.0]),
        ],
    )
    def test_get_indexer_non_unique_int_index(self, idx_values, idx_non_unique):
        indexes, missing = Index(idx_values).get_indexer_non_unique(Index([np.nan]))  # 调用 Index 对象的 get_indexer_non_unique 方法
        tm.assert_numpy_array_equal(np.array([0], dtype=np.intp), indexes)  # 使用 Pandas 测试模块的方法进行数组相等性断言
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)

        indexes, missing = Index(idx_values).get_indexer_non_unique(
            Index(idx_non_unique)
        )  # 再次调用 Index 对象的 get_indexer_non_unique 方法
        tm.assert_numpy_array_equal(np.array([0, 1, 3], dtype=np.intp), indexes)  # 使用 Pandas 测试模块的方法进行数组相等性断言
        tm.assert_numpy_array_equal(np.array([], dtype=np.intp), missing)


class TestGetLoc:  # 定义测试类 TestGetLoc
    @pytest.mark.slow  # 标记为慢速测试
    def test_get_loc_tuple_monotonic_above_size_cutoff(self, monkeypatch):
        # Go through the libindex path for which using
        # _bin_search vs ndarray.searchsorted makes a difference

        with monkeypatch.context():  # 使用 Monkeypatch 上下文
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", 100)  # 修改 libindex 模块的 _SIZE_CUTOFF 属性
            lev = list("ABCD")
            dti = pd.date_range("2016-01-01", periods=10)

            mi = pd.MultiIndex.from_product([lev, range(5), dti])  # 创建多级索引对象
            oidx = mi.to_flat_index()  # 获得扁平化的索引

            loc = len(oidx) // 2  # 计算索引位置的中点
            tup = oidx[loc]  # 获取中点位置的索引元组

            res = oidx.get_loc(tup)  # 调用 get_loc 方法获取元组的位置
        assert res == loc  # 断言结果符合预期
    # 定义一个测试函数，测试获取包含非单调非唯一对象 dtype 的 Index 对象中的元素位置
    def test_get_loc_nan_object_dtype_nonmonotonic_nonunique(self):
        # 创建一个 Index 对象 idx，包含字符串 "foo"、np.nan、None、"foo"、1.0、None
        idx = Index(["foo", np.nan, None, "foo", 1.0, None], dtype=object)

        # 获取 np.nan 在 idx 中的位置，预期结果是索引 1
        res = idx.get_loc(np.nan)
        assert res == 1

        # 获取 None 在 idx 中的位置，预期结果是一个布尔数组，只有第三个和最后一个元素为 True
        res = idx.get_loc(None)
        expected = np.array([False, False, True, False, False, True])
        tm.assert_numpy_array_equal(res, expected)

        # 尝试获取 NaT（不匹配的 NA 值），预期会抛出 KeyError 异常，异常信息应包含 "NaT"
        with pytest.raises(KeyError, match="NaT"):
            idx.get_loc(NaT)
# 定义一个测试函数，用于测试布尔型索引器在Series上的行为
def test_getitem_boolean_ea_indexer():
    # GH#45806：关联GitHub issue编号，说明这段测试的背景或目的
    # 创建一个包含True、False和pd.NA（缺失值）的Series，指定dtype为"boolean"
    ser = pd.Series([True, False, pd.NA], dtype="boolean")
    # 通过布尔型索引器，从Series的索引中选取True值对应的索引位置
    result = ser.index[ser]
    # 预期的结果是一个Index对象，包含索引位置0
    expected = Index([0])
    # 使用测试框架的断言函数，验证result与expected是否相等
    tm.assert_index_equal(result, expected)
```