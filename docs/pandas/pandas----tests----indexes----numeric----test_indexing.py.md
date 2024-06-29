# `D:\src\scipysrc\pandas\pandas\tests\indexes\numeric\test_indexing.py`

```
import numpy as np
import pytest

from pandas.errors import InvalidIndexError

from pandas import (
    NA,
    Index,
    RangeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowExtensionArray,
    FloatingArray,
)

class TestGetLoc:
    # 定义测试索引定位方法的测试类

    def test_get_loc(self):
        # 测试获取索引位置的基本用例
        index = Index([0, 1, 2])
        assert index.get_loc(1) == 1

    def test_get_loc_raises_bad_label(self):
        # 测试当索引包含无效标签时抛出异常的情况
        index = Index([0, 1, 2])
        with pytest.raises(InvalidIndexError, match=r"\[1, 2\]"):
            index.get_loc([1, 2])

    def test_get_loc_float64(self):
        # 测试 float64 类型索引的特殊情况
        idx = Index([0.0, 1.0, 2.0], dtype=np.float64)

        with pytest.raises(KeyError, match="^'foo'$"):
            idx.get_loc("foo")
        with pytest.raises(KeyError, match=r"^1\.5$"):
            idx.get_loc(1.5)
        with pytest.raises(KeyError, match="^True$"):
            idx.get_loc(True)
        with pytest.raises(KeyError, match="^False$"):
            idx.get_loc(False)

    def test_get_loc_na(self):
        # 测试包含 NaN 值的索引情况
        idx = Index([np.nan, 1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 1
        assert idx.get_loc(np.nan) == 0

        idx = Index([np.nan, 1, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1

        # 通过切片表示 [0:2:2] 的情况
        msg = "'Cannot get left slice bound for non-unique label: nan'"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)
        # 无法通过切片表示的情况
        idx = Index([np.nan, 1, np.nan, np.nan], dtype=np.float64)
        assert idx.get_loc(1) == 1
        msg = "'Cannot get left slice bound for non-unique label: nan"
        with pytest.raises(KeyError, match=msg):
            idx.slice_locs(np.nan)

    def test_get_loc_missing_nan(self):
        # 测试索引中缺少 NaN 值时的情况
        # GH#8569
        idx = Index([1, 2], dtype=np.float64)
        assert idx.get_loc(1) == 0
        with pytest.raises(KeyError, match=r"^3$"):
            idx.get_loc(3)
        with pytest.raises(KeyError, match="^nan$"):
            idx.get_loc(np.nan)
        with pytest.raises(InvalidIndexError, match=r"\[nan\]"):
            # listlike/non-hashable raises TypeError
            idx.get_loc([np.nan])

    @pytest.mark.parametrize("vals", [[1], [1.0], [Timestamp("2019-12-31")], ["test"]])
    def test_get_loc_float_index_nan_with_method(self, vals):
        # 测试带有特定值类型的参数化浮点索引情况
        # GH#39382
        idx = Index(vals)
        with pytest.raises(KeyError, match="nan"):
            idx.get_loc(np.nan)

    @pytest.mark.parametrize("dtype", ["f8", "i8", "u8"])
    def test_get_loc_numericindex_none_raises(self, dtype):
        # 测试数值类型索引情况下的异常情况
        # case that goes through searchsorted and key is non-comparable to values
        arr = np.arange(10**7, dtype=dtype)
        idx = Index(arr)
        with pytest.raises(KeyError, match="None"):
            idx.get_loc(None)
    # 测试 get_loc 方法当索引值溢出时是否能正确抛出 KeyError 异常
    def test_get_loc_overflows(self):
        # 创建一个索引对象，包含非单调但唯一的索引值，用于通过 IndexEngine.mapping.get_item 方法
        idx = Index([0, 2, 1])

        # 创建一个超出 np.int64 数据类型范围的值
        val = np.iinfo(np.int64).max + 1

        # 测试 get_loc 方法是否能正确抛出 KeyError 异常，匹配异常信息为超出范围的值
        with pytest.raises(KeyError, match=str(val)):
            idx.get_loc(val)
        # 测试 IndexEngine 的 get_loc 方法是否能正确抛出 KeyError 异常，匹配异常信息为超出范围的值
        with pytest.raises(KeyError, match=str(val)):
            idx._engine.get_loc(val)
    # 定义一个测试类 TestGetIndexer，用于测试 Index 对象的 get_indexer 方法
    class TestGetIndexer:
        
        # 测试 get_indexer 方法的基本功能
        def test_get_indexer(self):
            # 创建两个 Index 对象 index1 和 index2
            index1 = Index([1, 2, 3, 4, 5])
            index2 = Index([2, 4, 6])
            
            # 调用 index1 的 get_indexer 方法，传入 index2 作为参数
            r1 = index1.get_indexer(index2)
            # 预期的结果数组 e1
            e1 = np.array([1, 3, -1], dtype=np.intp)
            # 使用测试框架检查 r1 和 e1 是否几乎相等
            tm.assert_almost_equal(r1, e1)

        # 使用参数化测试，测试不同的方法和预期结果
        @pytest.mark.parametrize("reverse", [True, False])
        @pytest.mark.parametrize(
            "expected,method",
            [
                ([-1, 0, 0, 1, 1], "pad"),
                ([-1, 0, 0, 1, 1], "ffill"),
                ([0, 0, 1, 1, 2], "backfill"),
                ([0, 0, 1, 1, 2], "bfill"),
            ],
        )
        def test_get_indexer_methods(self, reverse, expected, method):
            # 创建两个 Index 对象 index1 和 index2
            index1 = Index([1, 2, 3, 4, 5])
            index2 = Index([2, 4, 6])
            # 将预期结果转换为 numpy 数组
            expected = np.array(expected, dtype=np.intp)
            # 如果 reverse 标志为 True，则反转 index1 和 expected
            if reverse:
                index1 = index1[::-1]
                expected = expected[::-1]

            # 调用 index2 的 get_indexer 方法，传入 index1 和 method 作为参数
            result = index2.get_indexer(index1, method=method)
            # 使用测试框架检查 result 和 expected 是否几乎相等
            tm.assert_almost_equal(result, expected)

        # 测试 get_indexer 方法在参数为无效值时是否引发异常
        def test_get_indexer_invalid(self):
            # 创建一个包含 0 到 9 的 Index 对象
            index = Index(np.arange(10))

            # 使用 pytest 框架确保调用 get_indexer 方法时会引发 ValueError 异常，匹配特定的错误消息
            with pytest.raises(ValueError, match="tolerance argument"):
                index.get_indexer([1, 0], tolerance=1)

            with pytest.raises(ValueError, match="limit argument"):
                index.get_indexer([1, 0], limit=1)

        # 使用参数化测试，测试不同的方法、容差和预期结果
        @pytest.mark.parametrize(
            "method, tolerance, indexer, expected",
            [
                ("pad", None, [0, 5, 9], [0, 5, 9]),
                ("backfill", None, [0, 5, 9], [0, 5, 9]),
                ("nearest", None, [0, 5, 9], [0, 5, 9]),
                ("pad", 0, [0, 5, 9], [0, 5, 9]),
                ("backfill", 0, [0, 5, 9], [0, 5, 9]),
                ("nearest", 0, [0, 5, 9], [0, 5, 9]),
                ("pad", None, [0.2, 1.8, 8.5], [0, 1, 8]),
                ("backfill", None, [0.2, 1.8, 8.5], [1, 2, 9]),
                ("nearest", None, [0.2, 1.8, 8.5], [0, 2, 9]),
                ("pad", 1, [0.2, 1.8, 8.5], [0, 1, 8]),
                ("backfill", 1, [0.2, 1.8, 8.5], [1, 2, 9]),
                ("nearest", 1, [0.2, 1.8, 8.5], [0, 2, 9]),
                ("pad", 0.2, [0.2, 1.8, 8.5], [0, -1, -1]),
                ("backfill", 0.2, [0.2, 1.8, 8.5], [-1, 2, -1]),
                ("nearest", 0.2, [0.2, 1.8, 8.5], [0, 2, -1]),
            ],
        )
        def test_get_indexer_nearest(self, method, tolerance, indexer, expected):
            # 创建一个包含 0 到 9 的 Index 对象
            index = Index(np.arange(10))

            # 调用 index 的 get_indexer 方法，传入 method、tolerance 和 indexer 作为参数
            actual = index.get_indexer(indexer, method=method, tolerance=tolerance)
            # 使用测试框架检查 actual 和 expected 是否准确匹配
            tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

        # 使用参数化测试，测试不同的容差列表类型和预期结果
        @pytest.mark.parametrize("listtype", [list, tuple, Series, np.array])
        @pytest.mark.parametrize(
            "tolerance, expected",
            [
                [[0.3, 0.3, 0.1], [0, 2, -1]],
                [[0.2, 0.1, 0.1], [0, -1, -1]],
                [[0.1, 0.5, 0.5], [-1, 2, 9]],
            ],
        )
        def test_get_indexer_nearest_listlike_tolerance(
            self, tolerance, expected, listtype
        ):
            # 创建一个包含 0 到 9 的 Index 对象
            index = Index(np.arange(10))

            # 调用 index 的 get_indexer 方法，传入 tolerance 和 expected 作为参数
            actual = index.get_indexer(tolerance, tolerance=tolerance)
            # 使用测试框架检查 actual 和 expected 是否准确匹配
            tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))
    @pytest.mark.parametrize(
        "method,expected",
        [("pad", [8, 7, 0]), ("backfill", [9, 8, 1]), ("nearest", [9, 7, 0])],
    )
    # 参数化测试方法，给定不同的method和期望结果expected
    def test_get_indexer_nearest_decreasing(self, method, expected):
        # 创建一个逆序排列的索引对象，包含数字0到9
        index = Index(np.arange(10))[::-1]

        # 对索引对象调用get_indexer方法，使用给定的method参数进行索引计算
        actual = index.get_indexer([0, 5, 9], method=method)
        # 断言计算得到的索引结果与预期的numpy数组相等
        tm.assert_numpy_array_equal(actual, np.array([9, 4, 0], dtype=np.intp))

        # 对索引对象调用get_indexer方法，使用给定的method参数进行索引计算，此处测试浮点数输入
        actual = index.get_indexer([0.2, 1.8, 8.5], method=method)
        # 断言计算得到的索引结果与预期的numpy数组相等
        tm.assert_numpy_array_equal(actual, np.array(expected, dtype=np.intp))

    @pytest.mark.parametrize("idx_dtype", ["int64", "float64", "uint64", "range"])
    @pytest.mark.parametrize("method", ["get_indexer", "get_indexer_non_unique"])
    # 参数化测试方法，对不同的索引数据类型和方法进行测试
    def test_get_indexer_numeric_index_boolean_target(self, method, idx_dtype):
        # 创建一个数字类型索引对象，根据不同的idx_dtype参数选择不同的索引类型
        if idx_dtype == "range":
            numeric_index = RangeIndex(4)
        else:
            numeric_index = Index(np.arange(4, dtype=idx_dtype))

        # 创建一个包含布尔值的索引对象
        other = Index([True, False, True])

        # 调用numeric_index对象的指定方法（method），将other作为参数传入
        result = getattr(numeric_index, method)(other)
        # 创建预期结果的numpy数组，期望所有匹配项的索引都是-1
        expected = np.array([-1, -1, -1], dtype=np.intp)
        # 如果method是"get_indexer"，则断言result与expected相等
        if method == "get_indexer":
            tm.assert_numpy_array_equal(result, expected)
        else:
            # 否则，预期第一项结果是expected，第二项结果是missing
            missing = np.arange(3, dtype=np.intp)
            tm.assert_numpy_array_equal(result[0], expected)
            tm.assert_numpy_array_equal(result[1], missing)

    @pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
    # 参数化测试方法，对不同的填充方法进行测试
    def test_get_indexer_with_method_numeric_vs_bool(self, method):
        # 创建两个索引对象，一个包含数字，一个包含布尔值
        left = Index([1, 2, 3])
        right = Index([True, False])

        # 使用pytest.raises断言捕获TypeError异常，匹配异常信息"Cannot compare"
        with pytest.raises(TypeError, match="Cannot compare"):
            # 调用左侧索引对象的get_indexer方法，将右侧索引对象和method参数传入
            left.get_indexer(right, method=method)

        # 使用pytest.raises断言捕获TypeError异常，匹配异常信息"Cannot compare"
        with pytest.raises(TypeError, match="Cannot compare"):
            # 调用右侧索引对象的get_indexer方法，将左侧索引对象和method参数传入
            right.get_indexer(left, method=method)
    # 定义一个测试方法，用于测试索引器在数值索引和布尔索引之间的行为
    def test_get_indexer_numeric_vs_bool(self):
        # 创建一个整数索引对象 `left`
        left = Index([1, 2, 3])
        # 创建一个布尔索引对象 `right`
        right = Index([True, False])

        # 使用 `left` 对象获取 `right` 的索引器结果
        res = left.get_indexer(right)
        # 创建一个预期结果数组，全部为 -1 的整数数组
        expected = -1 * np.ones(len(right), dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res, expected)

        # 使用 `right` 对象获取 `left` 的索引器结果
        res = right.get_indexer(left)
        # 创建一个预期结果数组，全部为 -1 的整数数组，与 left 的长度相同
        expected = -1 * np.ones(len(left), dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res, expected)

        # 使用 `left` 对象获取 `right` 的非唯一索引器结果
        res = left.get_indexer_non_unique(right)[0]
        # 创建一个预期结果数组，全部为 -1 的整数数组，与 right 的长度相同
        expected = -1 * np.ones(len(right), dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res, expected)

        # 使用 `right` 对象获取 `left` 的非唯一索引器结果
        res = right.get_indexer_non_unique(left)[0]
        # 创建一个预期结果数组，全部为 -1 的整数数组，与 left 的长度相同
        expected = -1 * np.ones(len(left), dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(res, expected)

    # 定义一个测试方法，用于测试浮点数索引器的行为
    def test_get_indexer_float64(self):
        # 创建一个浮点数索引对象 `idx`
        idx = Index([0.0, 1.0, 2.0], dtype=np.float64)
        # 断言 `idx` 对象与自身的索引器结果相等
        tm.assert_numpy_array_equal(
            idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp)
        )

        # 创建一个目标数组 `target`
        target = [-0.1, 0.5, 1.1]
        # 使用 "pad" 方法获取 `target` 相对于 `idx` 的索引器结果
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "pad"), np.array([-1, 0, 1], dtype=np.intp)
        )
        # 使用 "backfill" 方法获取 `target` 相对于 `idx` 的索引器结果
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "backfill"), np.array([0, 1, 2], dtype=np.intp)
        )
        # 使用 "nearest" 方法获取 `target` 相对于 `idx` 的索引器结果
        tm.assert_numpy_array_equal(
            idx.get_indexer(target, "nearest"), np.array([0, 1, 1], dtype=np.intp)
        )

    # 定义一个测试方法，用于测试 NaN 值在索引器中的行为
    def test_get_indexer_nan(self):
        # 对于包含 NaN 的浮点数索引对象，获取 NaN 的索引器结果
        # GH#7820 是一个特定的 GitHub 问题参考号
        result = Index([1, 2, np.nan], dtype=np.float64).get_indexer([np.nan])
        # 创建一个预期结果数组，包含索引 2（对应 NaN 的索引）
        expected = np.array([2], dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，用于测试整数索引器的行为
    def test_get_indexer_int64(self):
        # 创建一个整数索引对象 `index`
        index = Index(range(0, 20, 2), dtype=np.int64)
        # 创建一个目标整数索引对象 `target`
        target = Index(np.arange(10), dtype=np.int64)
        # 获取 `target` 相对于 `index` 的索引器结果
        indexer = index.get_indexer(target)
        # 创建一个预期结果数组，按照 `target` 在 `index` 中的索引顺序
        expected = np.array([0, -1, 1, -1, 2, -1, 3, -1, 4, -1], dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(indexer, expected)

        # 使用 "pad" 方法获取 `target` 相对于 `index` 的索引器结果
        indexer = index.get_indexer(target, method="pad")
        # 创建一个预期结果数组，使用前向填充方法索引 `target`
        expected = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(indexer, expected)

        # 使用 "backfill" 方法获取 `target` 相对于 `index` 的索引器结果
        indexer = index.get_indexer(target, method="backfill")
        # 创建一个预期结果数组，使用后向填充方法索引 `target`
        expected = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.intp)
        # 使用测试框架断言两个 NumPy 数组相等
        tm.assert_numpy_array_equal(indexer, expected)
    # 定义一个测试方法，用于测试针对 uint64 类型数据的索引功能
    def test_get_indexer_uint64(self):
        # 创建一个包含大整数的 Index 对象，数据类型为 np.uint64
        index_large = Index(
            [2**63, 2**63 + 10, 2**63 + 15, 2**63 + 20, 2**63 + 25],
            dtype=np.uint64,
        )
        # 创建一个目标 Index 对象，包含从 2**63 开始的 uint64 数组
        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        # 使用 index_large 对象获取目标数组的索引
        indexer = index_large.get_indexer(target)
        # 预期的索引结果数组，包含对应的索引位置或者 -1（如果目标值不在 index_large 中）
        expected = np.array([0, -1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        # 断言索引结果与预期结果相等
        tm.assert_numpy_array_equal(indexer, expected)

        # 重设目标 Index 对象为相同的 uint64 数据，但是使用 "pad" 方法获取索引
        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        # 使用 index_large 对象获取目标数组的索引，使用 "pad" 方法填充缺失值
        indexer = index_large.get_indexer(target, method="pad")
        # 预期的索引结果数组，使用 "pad" 方法填充后的索引
        expected = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4, 4], dtype=np.intp)
        # 断言索引结果与预期结果相等
        tm.assert_numpy_array_equal(indexer, expected)

        # 重设目标 Index 对象为相同的 uint64 数据，但是使用 "backfill" 方法获取索引
        target = Index(np.arange(10).astype("uint64") * 5 + 2**63)
        # 使用 index_large 对象获取目标数组的索引，使用 "backfill" 方法填充缺失值
        indexer = index_large.get_indexer(target, method="backfill")
        # 预期的索引结果数组，使用 "backfill" 方法填充后的索引
        expected = np.array([0, 1, 1, 2, 3, 4, -1, -1, -1, -1], dtype=np.intp)
        # 断言索引结果与预期结果相等
        tm.assert_numpy_array_equal(indexer, expected)

    @pytest.mark.parametrize("val, val2", [(4, 5), (4, 4), (4, NA), (NA, NA)])
    # 定义一个参数化测试方法，测试对含有 NA 值的索引功能
    def test_get_loc_masked(self, val, val2, any_numeric_ea_and_arrow_dtype):
        # 创建一个包含数值和 NA 值的 Index 对象，数据类型为 any_numeric_ea_and_arrow_dtype
        idx = Index([1, 2, 3, val, val2], dtype=any_numeric_ea_and_arrow_dtype)
        # 获取数值 2 的位置
        result = idx.get_loc(2)
        # 断言获取的位置为 1
        assert result == 1

        # 使用 pytest 的断言，期望抛出 KeyError，匹配错误信息中包含 "9"
        with pytest.raises(KeyError, match="9"):
            idx.get_loc(9)

    # 定义一个测试方法，测试对含有 NA 值的索引功能
    def test_get_loc_masked_na(self, any_numeric_ea_and_arrow_dtype):
        # 创建一个包含数值和 NA 值的 Index 对象，数据类型为 any_numeric_ea_and_arrow_dtype
        idx = Index([1, 2, NA], dtype=any_numeric_ea_and_arrow_dtype)
        # 获取 NA 值的位置
        result = idx.get_loc(NA)
        # 断言获取的位置为 2
        assert result == 2

        # 创建一个包含多个 NA 值的 Index 对象，数据类型为 any_numeric_ea_and_arrow_dtype
        idx = Index([1, 2, NA, NA], dtype=any_numeric_ea_and_arrow_dtype)
        # 获取 NA 值的位置
        result = idx.get_loc(NA)
        # 断言获取的位置数组与预期结果相等
        tm.assert_numpy_array_equal(result, np.array([False, False, True, True]))

        # 创建一个不含 NA 值的 Index 对象，数据类型为 any_numeric_ea_and_arrow_dtype
        idx = Index([1, 2, 3], dtype=any_numeric_ea_and_arrow_dtype)
        # 使用 pytest 的断言，期望抛出 KeyError，匹配错误信息中包含 "NA"
        with pytest.raises(KeyError, match="NA"):
            idx.get_loc(NA)

    # 定义一个测试方法，测试对同时含有 NA 和 NaN 值的索引功能
    def test_get_loc_masked_na_and_nan(self):
        # 创建一个包含浮点数和 NA 值的 Index 对象，数据类型为 FloatingArray
        idx = Index(
            FloatingArray(
                np.array([1, 2, 1, np.nan]), mask=np.array([False, False, True, False])
            )
        )
        # 获取 NA 值的位置
        result = idx.get_loc(NA)
        # 断言获取的位置为 2
        assert result == 2
        # 获取 NaN 值的位置
        result = idx.get_loc(np.nan)
        # 断言获取的位置为 3
        assert result == 3

        # 创建一个包含浮点数和 NA 值的 Index 对象，数据类型为 FloatingArray
        idx = Index(
            FloatingArray(np.array([1, 2, 1.0]), mask=np.array([False, False, True]))
        )
        # 获取 NA 值的位置
        result = idx.get_loc(NA)
        # 断言获取的位置为 2
        assert result == 2
        # 使用 pytest 的断言，期望抛出 KeyError，匹配错误信息中包含 "nan"
        with pytest.raises(KeyError, match="nan"):
            idx.get_loc(np.nan)

        # 创建一个包含浮点数和 NaN 值的 Index 对象，数据类型为 FloatingArray
        idx = Index(
            FloatingArray(
                np.array([1, 2, np.nan]), mask=np.array([False, False, False])
            )
        )
        # 获取 NaN 值的位置
        result = idx.get_loc(np.nan)
        # 断言获取的位置为 2
        assert result == 2
        # 使用 pytest 的断言，期望抛出 KeyError，匹配错误信息中包含 "NA"
        with pytest.raises(KeyError, match="NA"):
            idx.get_loc(NA)

    @pytest.mark.parametrize("val", [4, 2])
    # 测试函数，验证在包含缺失值和特定数据类型的情况下，get_indexer_for 方法的行为
    def test_get_indexer_masked_na(self, any_numeric_ea_and_arrow_dtype, val):
        # GH#39133：参考GitHub上的issue编号
        # 创建一个包含整数和缺失值的索引对象
        idx = Index([1, 2, NA, 3, val], dtype=any_numeric_ea_and_arrow_dtype)
        # 获取目标值 [1, NA, 5] 在索引中的位置索引
        result = idx.get_indexer_for([1, NA, 5])
        # 预期结果是 [0, 2, -1]
        expected = np.array([0, 2, -1])
        # 使用测试工具函数验证结果数组与预期数组相等
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    # 使用参数化测试，验证在不同数据类型下，处理缺失值的方法
    @pytest.mark.parametrize("dtype", ["boolean", "bool[pyarrow]"])
    def test_get_indexer_masked_na_boolean(self, dtype):
        # GH#39133：参考GitHub上的issue编号
        # 如果数据类型是 "bool[pyarrow]"，则需要先确保导入 pyarrow 库
        if dtype == "bool[pyarrow]":
            pytest.importorskip("pyarrow")
        # 创建一个包含布尔值和缺失值的索引对象
        idx = Index([True, False, NA], dtype=dtype)
        # 获取 False 的位置索引
        result = idx.get_loc(False)
        # 断言结果应为 1
        assert result == 1
        # 获取 NA 的位置索引
        result = idx.get_loc(NA)
        # 断言结果应为 2
        assert result == 2

    # 测试函数，验证在处理 Arrow 扩展数组目标时的索引行为
    def test_get_indexer_arrow_dictionary_target(self):
        # 导入 pyarrow 库，如未安装则跳过测试
        pa = pytest.importorskip("pyarrow")
        # 创建一个 Arrow 扩展数组作为索引的目标
        target = Index(
            ArrowExtensionArray(
                pa.array([1, 2], type=pa.dictionary(pa.int8(), pa.int8()))
            )
        )
        # 创建一个普通的索引对象
        idx = Index([1])
        
        # 获取索引对象 idx 相对于目标索引 target 的位置索引
        result = idx.get_indexer(target)
        # 预期结果是 [0, -1]
        expected = np.array([0, -1], dtype=np.int64)
        # 使用测试工具函数验证结果数组与预期数组相等
        tm.assert_numpy_array_equal(result, expected)
        
        # 获取索引对象 idx 相对于目标索引 target 的非唯一位置索引
        result_1, result_2 = idx.get_indexer_non_unique(target)
        # 预期结果分别是 [0, -1] 和 [1]
        expected_1, expected_2 = (
            np.array([0, -1], dtype=np.int64),
            np.array([1], dtype=np.int64),
        )
        # 使用测试工具函数验证结果数组与预期数组相等
        tm.assert_numpy_array_equal(result_1, expected_1)
        tm.assert_numpy_array_equal(result_2, expected_2)
class TestWhere:
    # 测试用例：where 方法的测试
    @pytest.mark.parametrize(
        "index",
        [
            # 参数化测试：使用不同的 Index 实例
            Index(np.arange(5, dtype="float64")),  # 创建浮点数类型的索引
            Index(range(0, 20, 2), dtype=np.int64),  # 创建整数类型的索引
            Index(np.arange(5, dtype="uint64")),  # 创建无符号长整数类型的索引
        ],
    )
    def test_where(self, listlike_box, index):
        # 准备条件数组，所有值为 True
        cond = [True] * len(index)
        expected = index
        # 测试 where 方法，期望返回原始的 Index 对象
        result = index.where(listlike_box(cond))

        # 准备条件数组，除了第一个值为 False，其余值为 True
        cond = [False] + [True] * (len(index) - 1)
        # 创建期望的 Index 对象，将第一个元素替换为 NaN（特定值）
        expected = Index([index._na_value] + index[1:].tolist(), dtype=np.float64)
        # 再次测试 where 方法，期望返回修改后的 Index 对象
        result = index.where(listlike_box(cond))
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected)

    # 测试用例：测试 where 方法处理 uint64 类型数据
    def test_where_uint64(self):
        # 创建 uint64 类型的 Index 对象
        idx = Index([0, 6, 2], dtype=np.uint64)
        # 创建布尔掩码
        mask = np.array([False, True, False])
        # 创建替换数组
        other = np.array([1], dtype=np.int64)

        # 期望的 Index 对象
        expected = Index([1, 6, 1], dtype=np.uint64)

        # 测试 where 方法，期望返回替换后的 Index 对象
        result = idx.where(mask, other)
        tm.assert_index_equal(result, expected)

        # 测试 putmask 方法，期望返回替换后的 Index 对象
        result = idx.putmask(~mask, other)
        tm.assert_index_equal(result, expected)

    # 测试用例：确保 where 方法可以推断类型，而不是尝试将字符串转换为浮点数（GH 32413）
    def test_where_infers_type_instead_of_trying_to_convert_string_to_float(self):
        # 创建包含 NaN 的 Index 对象
        index = Index([1, np.nan])
        # 创建条件数组，检查是否为 NaN
        cond = index.notna()
        # 创建替换数组，包含字符串类型的数据
        other = Index(["a", "b"], dtype="string")

        # 期望的 Index 对象，保留浮点数类型和字符串类型数据
        expected = Index([1.0, "b"])
        # 测试 where 方法，期望返回替换后的 Index 对象
        result = index.where(cond, other)

        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected)


class TestTake:
    # 测试用例：测试 take 方法保留名称属性
    @pytest.mark.parametrize("idx_dtype", [np.float64, np.int64, np.uint64])
    def test_take_preserve_name(self, idx_dtype):
        # 创建带有名称的 Index 对象
        index = Index([1, 2, 3, 4], dtype=idx_dtype, name="foo")
        # 调用 take 方法
        taken = index.take([3, 0, 1])
        # 断言索引对象的名称属性保持不变
        assert index.name == taken.name

    # 测试用例：测试 take 方法使用 float64 类型数据填充值（GH 12631）
    def test_take_fill_value_float64(self):
        # 创建 float64 类型的 Index 对象
        idx = Index([1.0, 2.0, 3.0], name="xxx", dtype=np.float64)
        # 调用 take 方法
        result = idx.take(np.array([1, 0, -1]))
        # 期望的 Index 对象，包含特定的 float64 类型数据
        expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name="xxx")
        # 断言两个 Index 对象相等
        tm.assert_index_equal(result, expected)

        # 使用 fill_value 参数进行测试
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index([2.0, 1.0, np.nan], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        # 使用 allow_fill=False 参数进行测试
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index([2.0, 1.0, 3.0], dtype=np.float64, name="xxx")
        tm.assert_index_equal(result, expected)

        # 测试异常情况：当 allow_fill=True 且 fill_value 不为空时，所有索引必须 >= -1
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 测试异常情况：索引 -5 超出了索引范围
        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    # 以下省略的部分属于代码结构的一部分，不需要添加注释
    # 定义一个测试方法，用于测试在给定数据类型 `dtype` 的情况下的索引操作
    def test_take_fill_value_ints(self, dtype):
        # 注释: 解释这是一个关于 GitHub issue #12631 的测试用例
        idx = Index([1, 2, 3], dtype=dtype, name="xxx")
        # 执行索引操作，返回索引值为 [1, 0, -1] 的元素组成的结果
        result = idx.take(np.array([1, 0, -1]))
        # 预期的索引结果，应为 [2, 1, 3]
        expected = Index([2, 1, 3], dtype=dtype, name="xxx")
        # 断言两个索引对象是否相等
        tm.assert_index_equal(result, expected)

        # 获取索引对象类型的名称
        name = type(idx).__name__
        # 构造填充值失败的错误信息
        msg = f"Unable to fill values because {name} cannot contain NA"

        # 使用 pytest 断言，期望会抛出 ValueError 异常，并且匹配预期的错误信息消息
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -1]), fill_value=True)

        # 使用 allow_fill=False 参数执行索引操作，期望返回与 `expected` 相同的结果
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        tm.assert_index_equal(result, expected)

        # 再次使用 pytest 断言，期望抛出 ValueError 异常，匹配预期的错误信息消息
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 构造索引超出边界的错误信息
        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        # 使用 pytest 断言，期望抛出 IndexError 异常，并匹配预期的错误信息消息
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))
    # 定义一个测试类 TestContains，用于测试 Index 对象的包含操作
    class TestContains:
        
        # 使用 pytest 的参数化装饰器，为测试方法 test_contains_none 提供不同的数据类型作为参数
        @pytest.mark.parametrize("dtype", [np.float64, np.int64, np.uint64])
        def test_contains_none(self, dtype):
            # 创建一个 Index 对象，包含整数 0 到 4，使用指定的数据类型
            index = Index([0, 1, 2, 3, 4], dtype=dtype)
            # 断言空值 None 不在 index 中
            assert None not in index

        # 测试 Index 对象包含浮点数 NaN 的情况
        def test_contains_float64_nans(self):
            # 创建一个 Index 对象，包含浮点数和 NaN，使用 np.float64 数据类型
            index = Index([1.0, 2.0, np.nan], dtype=np.float64)
            # 断言 NaN 在 index 中
            assert np.nan in index

        # 测试 Index 对象包含非 NaN 浮点数的情况
        def test_contains_float64_not_nans(self):
            # 创建一个 Index 对象，包含浮点数和 NaN，使用 np.float64 数据类型
            index = Index([1.0, 2.0, np.nan], dtype=np.float64)
            # 断言浮点数 1.0 在 index 中
            assert 1.0 in index


    # 定义一个测试类 TestSliceLocs，用于测试 Index 对象的切片定位功能
    class TestSliceLocs:
        
        # 使用 pytest 的参数化装饰器，为测试方法 test_slice_locs 提供不同的数据类型作为参数
        @pytest.mark.parametrize("dtype", [int, float])
        def test_slice_locs(self, dtype):
            # 创建一个 Index 对象，包含指定数据类型的数组
            index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
            n = len(index)

            # 断言从指定起始位置开始的切片定位结果
            assert index.slice_locs(start=2) == (2, n)
            assert index.slice_locs(start=3) == (3, n)
            assert index.slice_locs(3, 8) == (3, 6)
            assert index.slice_locs(5, 10) == (3, n)
            assert index.slice_locs(end=8) == (0, 6)
            assert index.slice_locs(end=9) == (0, 7)

            # 创建一个反向的 Index 对象
            index2 = index[::-1]
            # 断言反向对象上的切片定位结果
            assert index2.slice_locs(8, 2) == (2, 6)
            assert index2.slice_locs(7, 3) == (2, 5)

        # 使用 pytest 的参数化装饰器，为测试方法 test_slice_locs_float_locs 提供不同的数据类型作为参数
        @pytest.mark.parametrize("dtype", [int, float])
        def test_slice_locs_float_locs(self, dtype):
            # 创建一个 Index 对象，包含指定数据类型的数组
            index = Index(np.array([0, 1, 2, 5, 6, 7, 9, 10], dtype=dtype))
            n = len(index)
            # 断言根据浮点数起始和结束位置的切片定位结果
            assert index.slice_locs(5.0, 10.0) == (3, n)
            assert index.slice_locs(4.5, 10.5) == (3, 8)

            # 创建一个反向的 Index 对象
            index2 = index[::-1]
            # 断言反向对象上根据浮点数起始和结束位置的切片定位结果
            assert index2.slice_locs(8.5, 1.5) == (2, 6)
            assert index2.slice_locs(10.5, -1) == (0, n)

        # 使用 pytest 的参数化装饰器，为测试方法 test_slice_locs_dup_numeric 提供不同的数据类型作为参数
        @pytest.mark.parametrize("dtype", [int, float])
        def test_slice_locs_dup_numeric(self, dtype):
            # 创建一个 Index 对象，包含指定数据类型的数组
            index = Index(np.array([10, 12, 12, 14], dtype=dtype))
            # 断言重复数值的切片定位结果
            assert index.slice_locs(12, 12) == (1, 3)
            assert index.slice_locs(11, 13) == (1, 3)

            # 创建一个反向的 Index 对象
            index2 = index[::-1]
            # 断言反向对象上重复数值的切片定位结果
            assert index2.slice_locs(12, 12) == (1, 3)
            assert index2.slice_locs(13, 11) == (1, 3)

        # 测试包含 NaN 的 Index 对象的切片定位
        def test_slice_locs_na(self):
            # 创建一个 Index 对象，包含 NaN 和其他数值
            index = Index([np.nan, 1, 2])
            # 断言根据 NaN 的切片定位结果
            assert index.slice_locs(1) == (1, 3)
            assert index.slice_locs(np.nan) == (0, 3)

            # 创建一个 Index 对象，包含 NaN 和其他数值
            index = Index([0, np.nan, np.nan, 1, 2])
            # 断言根据 NaN 的切片定位结果
            assert index.slice_locs(np.nan) == (1, 5)

        # 测试在包含 NaN 的 Index 对象上使用错误参数引发异常
        def test_slice_locs_na_raises(self):
            # 创建一个 Index 对象，包含 NaN 和其他数值
            index = Index([np.nan, 1, 2])
            # 使用 pytest 的 assert 异常上下文管理器，验证 start 参数引发 KeyError 异常
            with pytest.raises(KeyError, match=""):
                index.slice_locs(start=1.5)

            # 使用 pytest 的 assert 异常上下文管理器，验证 end 参数引发 KeyError 异常
            with pytest.raises(KeyError, match=""):
                index.slice_locs(end=1.5)


    # 定义一个测试类 TestGetSliceBounds，用于测试 Index 对象的切片边界获取功能
    class TestGetSliceBounds:
        
        # 使用 pytest 的参数化装饰器，为测试方法 test_get_slice_bounds_within 提供不同的侧边和预期结果作为参数
        @pytest.mark.parametrize("side, expected", [("left", 4), ("right", 5)])
        def test_get_slice_bounds_within(self, side, expected):
            # 创建一个 Index 对象，包含从 0 到 5 的整数
            index = Index(range(6))
            # 获取指定侧边的切片边界
            result = index.get_slice_bound(4, side=side)
            # 断言获取的结果与预期结果相等
            assert result == expected

        # 使用 pytest 的参数化装饰器，为测试方法 test_get_slice_bounds_within 提供不同的侧边和边界、预期结果作为参数
        @pytest.mark.parametrize("side", ["left", "right"])
        @pytest.mark.parametrize("bound, expected", [(-1, 0), (10, 6)])
    # 定义一个测试方法，用于测试 Index 类的 get_slice_bound 方法在给定参数下的行为
    def test_get_slice_bounds_outside(self, side, expected, bound):
        # 创建一个 Index 对象，传入一个包含整数 0 到 5 的范围对象
        index = Index(range(6))
        # 调用 Index 对象的 get_slice_bound 方法，传入参数 bound 和 side，获取结果
        result = index.get_slice_bound(bound, side=side)
        # 使用断言来验证调用结果是否与期望值 expected 相等
        assert result == expected
```