# `D:\src\scipysrc\pandas\pandas\tests\indexes\object\test_indexing.py`

```
from decimal import Decimal  # 导入 Decimal 类

import numpy as np  # 导入 numpy 库，并简称为 np
import pytest  # 导入 pytest 库

from pandas._libs.missing import (  # 导入 pandas 内部库中的 NA 和 is_matching_na 函数
    NA,
    is_matching_na,
)
import pandas.util._test_decorators as td  # 导入 pandas 的测试装饰器

import pandas as pd  # 导入 pandas 库，并简称为 pd
from pandas import Index  # 从 pandas 中导入 Index 类
import pandas._testing as tm  # 导入 pandas 内部测试工具


class TestGetIndexer:
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义参数化测试方法
        "method,expected",
        [
            ("pad", [-1, 0, 1, 1]),  # 参数化测试的测试用例，使用方法 'pad' 时的预期结果
            ("backfill", [0, 0, 1, -1]),  # 参数化测试的测试用例，使用方法 'backfill' 时的预期结果
        ],
    )
    def test_get_indexer_strings(self, method, expected):
        expected = np.array(expected, dtype=np.intp)  # 将预期结果列表转换为 numpy 数组
        index = Index(["b", "c"])  # 创建一个 Index 对象，包含字符串列表 ["b", "c"]
        actual = index.get_indexer(["a", "b", "c", "d"], method=method)  # 调用 get_indexer 方法，获取索引器

        tm.assert_numpy_array_equal(actual, expected)  # 使用测试工具 tm 进行 numpy 数组的相等性断言

    def test_get_indexer_strings_raises(self, using_infer_string):
        index = Index(["b", "c"])  # 创建一个 Index 对象，包含字符串列表 ["b", "c"]

        if using_infer_string:
            import pyarrow as pa  # 条件满足时，导入 pyarrow 库并简称为 pa

            msg = "has no kernel"
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                index.get_indexer(["a", "b", "c", "d"], method="nearest")  # 使用最近方法引发 ArrowNotImplementedError 异常

            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                index.get_indexer(["a", "b", "c", "d"], method="pad", tolerance=2)  # 使用 pad 方法和容差值 2 引发异常

            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                index.get_indexer(
                    ["a", "b", "c", "d"], method="pad", tolerance=[2, 2, 2, 2]
                )  # 使用 pad 方法和容差值列表引发异常

        else:
            msg = r"unsupported operand type\(s\) for -: 'str' and 'str'"
            with pytest.raises(TypeError, match=msg):
                index.get_indexer(["a", "b", "c", "d"], method="nearest")  # 使用最近方法引发 TypeError 异常

            with pytest.raises(TypeError, match=msg):
                index.get_indexer(["a", "b", "c", "d"], method="pad", tolerance=2)  # 使用 pad 方法和容差值 2 引发异常

            with pytest.raises(TypeError, match=msg):
                index.get_indexer(
                    ["a", "b", "c", "d"], method="pad", tolerance=[2, 2, 2, 2]
                )  # 使用 pad 方法和容差值列表引发异常

    def test_get_indexer_with_NA_values(
        self, unique_nulls_fixture, unique_nulls_fixture2
    ):
        # GH#22332
        # 检查两两组合的 NA 值不会混淆
        if unique_nulls_fixture is unique_nulls_fixture2:
            return  # 如果两个值相等，则跳过测试
        arr = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)  # 创建包含 NA 值的对象数组
        index = Index(arr, dtype=object)  # 创建 Index 对象，使用对象数组
        result = index.get_indexer(
            Index(
                [unique_nulls_fixture, unique_nulls_fixture2, "Unknown"], dtype=object
            )  # 调用 get_indexer 方法获取索引器，使用包含 NA 值和 "Unknown" 的 Index 对象
        )
        expected = np.array([0, 1, -1], dtype=np.intp)  # 创建预期的索引器数组
        tm.assert_numpy_array_equal(result, expected)  # 使用测试工具 tm 进行 numpy 数组的相等性断言


class TestGetIndexerNonUnique:
    def test_get_indexer_non_unique_nas(
        self, nulls_fixture, request, using_infer_string
    ):
        # 即使这不是非唯一的，但这应该仍然有效
        if using_infer_string and (nulls_fixture is None or nulls_fixture is NA):
            # 如果使用推断字符串并且 nulls_fixture 为 None 或 NA，则应用 xfail 标记
            request.applymarker(pytest.mark.xfail(reason="NAs are cast to NaN"))

        # 创建包含三个元素的 Index 对象，其中包括 nulls_fixture
        index = Index(["a", "b", nulls_fixture])
        # 获取针对非唯一索引的索引器和缺失项数组
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])

        # 期望的索引器数组，仅包含一个元素
        expected_indexer = np.array([2], dtype=np.intp)
        # 期望的缺失项数组，为空数组
        expected_missing = np.array([], dtype=np.intp)
        # 断言获取的索引器和缺失项数组与期望的数组相等
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # 创建包含四个元素的 Index 对象，其中包括两个 nulls_fixture
        index = Index(["a", nulls_fixture, "b", nulls_fixture])
        # 获取针对非唯一索引的索引器和缺失项数组
        indexer, missing = index.get_indexer_non_unique([nulls_fixture])

        # 期望的索引器数组，包含两个索引值
        expected_indexer = np.array([1, 3], dtype=np.intp)
        # 断言获取的索引器和缺失项数组与期望的数组相等
        tm.assert_numpy_array_equal(indexer, expected_indexer)
        tm.assert_numpy_array_equal(missing, expected_missing)

        # 如果 nulls_fixture 和 float("NaN") 匹配但不完全相同
        if is_matching_na(nulls_fixture, float("NaN")):
            # 创建包含四个元素的 Index 对象，其中包括两个 float("NaN")
            index = Index(["a", float("NaN"), "b", float("NaN")])
            # 设置 match_but_not_identical 标志为 True
            match_but_not_identical = True
        # 如果 nulls_fixture 和 Decimal("NaN") 匹配但不完全相同
        elif is_matching_na(nulls_fixture, Decimal("NaN")):
            # 创建包含四个元素的 Index 对象，其中包括两个 Decimal("NaN")
            index = Index(["a", Decimal("NaN"), "b", Decimal("NaN")])
            # 设置 match_but_not_identical 标志为 True
            match_but_not_identical = True
        else:
            # 否则，将 match_but_not_identical 标志设置为 False
            match_but_not_identical = False

        # 如果 match_but_not_identical 标志为 True
        if match_but_not_identical:
            # 获取针对非唯一索引的索引器和缺失项数组
            indexer, missing = index.get_indexer_non_unique([nulls_fixture])

            # 期望的索引器数组，包含两个索引值
            expected_indexer = np.array([1, 3], dtype=np.intp)
            # 断言获取的索引器和缺失项数组与期望的数组相等
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)
    def test_get_indexer_non_unique_np_nats(self, np_nat_fixture, np_nat_fixture2):
        # 创建一个空的预期缺失值数组，数据类型为 np.intp
        expected_missing = np.array([], dtype=np.intp)
        # 检查 np_nat_fixture 和 np_nat_fixture2 是否匹配但不相同的 NaT
        if is_matching_na(np_nat_fixture, np_nat_fixture2):
            # 确保 np_nat_fixture 和 np_nat_fixture2 是不同的对象
            index = Index(
                np.array(
                    ["2021-10-02", np_nat_fixture.copy(), np_nat_fixture2.copy()],
                    dtype=object,
                ),
                dtype=object,
            )
            # 将 Index 对象作为索引传递，以防止目标被转换为 DatetimeIndex
            indexer, missing = index.get_indexer_non_unique(
                Index([np_nat_fixture], dtype=object)
            )
            # 预期的 indexer 数组，数据类型为 np.intp
            expected_indexer = np.array([1, 2], dtype=np.intp)
            # 检查 indexer 和 missing 是否与预期值相等
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)
        # 处理 dt64nat vs td64nat 情况
        else:
            try:
                # 检查 np_nat_fixture 和 np_nat_fixture2 是否相等
                np_nat_fixture == np_nat_fixture2
            except (TypeError, OverflowError):
                # 如果是不可比较的类型，如 np.datetime64('NaT', 'Y') 和 np.datetime64('NaT', 'ps')
                # 抛出异常后直接返回，参考 https://github.com/numpy/numpy/issues/22762
                return
            index = Index(
                np.array(
                    [
                        "2021-10-02",
                        np_nat_fixture,
                        np_nat_fixture2,
                        np_nat_fixture,
                        np_nat_fixture2,
                    ],
                    dtype=object,
                ),
                dtype=object,
            )
            # 将 Index 对象作为索引传递，以防止目标被转换为 DatetimeIndex
            indexer, missing = index.get_indexer_non_unique(
                Index([np_nat_fixture], dtype=object)
            )
            # 预期的 indexer 数组，数据类型为 np.intp
            expected_indexer = np.array([1, 3], dtype=np.intp)
            # 检查 indexer 和 missing 是否与预期值相等
            tm.assert_numpy_array_equal(indexer, expected_indexer)
            tm.assert_numpy_array_equal(missing, expected_missing)
class TestSliceLocs:
    # 参数化测试，测试不同的数据类型
    @pytest.mark.parametrize(
        "dtype",
        [
            "object",
            pytest.param("string[pyarrow_numpy]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    # 参数化测试，测试不同的切片和期望结果
    @pytest.mark.parametrize(
        "in_slice,expected",
        [
            # 错误：切片索引必须是整数或None
            (pd.IndexSlice[::-1], "yxdcb"),
            (pd.IndexSlice["b":"y":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["b"::-1], "b"),  # type: ignore[misc]
            (pd.IndexSlice[:"b":-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"y":-1], "y"),  # type: ignore[misc]
            (pd.IndexSlice["y"::-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice["y"::-4], "yb"),  # type: ignore[misc]
            # 缺少标签
            (pd.IndexSlice[:"a":-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"a":-2], "ydb"),  # type: ignore[misc]
            (pd.IndexSlice["z"::-1], "yxdcb"),  # type: ignore[misc]
            (pd.IndexSlice["z"::-3], "yc"),  # type: ignore[misc]
            (pd.IndexSlice["m"::-1], "dcb"),  # type: ignore[misc]
            (pd.IndexSlice[:"m":-1], "yx"),  # type: ignore[misc]
            (pd.IndexSlice["a":"a":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["z":"z":-1], ""),  # type: ignore[misc]
            (pd.IndexSlice["m":"m":-1], ""),  # type: ignore[misc]
        ],
    )
    # 测试负步长的切片位置计算
    def test_slice_locs_negative_step(self, in_slice, expected, dtype):
        # 创建索引对象
        index = Index(list("bcdxy"), dtype=dtype)

        # 计算切片起始和结束位置
        s_start, s_stop = index.slice_locs(in_slice.start, in_slice.stop, in_slice.step)
        # 根据计算的位置获取切片结果
        result = index[s_start : s_stop : in_slice.step]
        # 创建期望的索引对象
        expected = Index(list(expected), dtype=dtype)
        # 断言切片结果是否符合预期
        tm.assert_index_equal(result, expected)

    # 跳过测试，如果没有安装pyarrow
    @td.skip_if_no("pyarrow")
    def test_slice_locs_negative_step_oob(self):
        # 创建索引对象
        index = Index(list("bcdxy"), dtype="string[pyarrow_numpy]")

        # 获取超出边界的切片结果
        result = index[-10:5:1]
        # 断言切片结果是否符合预期
        tm.assert_index_equal(result, index)

        # 获取负步长切片的结果
        result = index[4:-10:-1]
        # 创建期望的索引对象
        expected = Index(list("yxdcb"), dtype="string[pyarrow_numpy]")
        # 断言切片结果是否符合预期
        tm.assert_index_equal(result, expected)

    # 测试重复标签的切片位置计算
    def test_slice_locs_dup(self):
        # 创建索引对象
        index = Index(["a", "a", "b", "c", "d", "d"])
        # 断言切片位置是否正确
        assert index.slice_locs("a", "d") == (0, 6)
        assert index.slice_locs(end="d") == (0, 6)
        assert index.slice_locs("a", "c") == (0, 4)
        assert index.slice_locs("b", "d") == (2, 6)

        # 创建倒序的索引对象
        index2 = index[::-1]
        # 断言倒序索引对象的切片位置是否正确
        assert index2.slice_locs("d", "a") == (0, 6)
        assert index2.slice_locs(end="a") == (0, 6)
        assert index2.slice_locs("d", "b") == (0, 4)
        assert index2.slice_locs("c", "a") == (2, 6)
```