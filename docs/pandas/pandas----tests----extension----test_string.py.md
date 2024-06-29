# `D:\src\scipysrc\pandas\pandas\tests\extension\test_string.py`

```
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""

# 导入必要的模块和库
from __future__ import annotations

import string
from typing import cast

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base


# 定义一个函数，根据条件可能分割数组
def maybe_split_array(arr, chunked):
    # 如果不需要分割，则直接返回数组
    if not chunked:
        return arr
    # 如果数组的存储类型不是 "pyarrow"，也直接返回数组
    elif arr.dtype.storage != "pyarrow":
        return arr

    # 导入并检查是否存在 pyarrow
    pa = pytest.importorskip("pyarrow")

    # 获取 Arrow 数组
    arrow_array = arr._pa_array
    # 分割数组为两半
    split = len(arrow_array) // 2
    # 创建一个新的分块数组
    arrow_array = pa.chunked_array(
        [*arrow_array[:split].chunks, *arrow_array[split:].chunks]
    )
    # 确保分块数组有两个分块
    assert arrow_array.num_chunks == 2
    # 返回与原始数组类型相同的新数组
    return type(arr)(arrow_array)


# 定义一个夹具函数，根据参数返回是否分块
@pytest.fixture(params=[True, False])
def chunked(request):
    return request.param


# 定义一个夹具函数，根据字符串存储类型返回 StringDtype 对象
@pytest.fixture
def dtype(string_storage):
    return StringDtype(storage=string_storage)


# 定义一个夹具函数，根据 dtype 和 chunked 参数生成随机字符串数组
@pytest.fixture
def data(dtype, chunked):
    # 生成随机字符串数组
    strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
    while strings[0] == strings[1]:
        strings = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)

    # 构造字符串数组对象并返回可能分割后的结果
    arr = dtype.construct_array_type()._from_sequence(strings, dtype=dtype)
    return maybe_split_array(arr, chunked)


# 定义一个夹具函数，根据 dtype 和 chunked 参数生成包含缺失值的数组
@pytest.fixture
def data_missing(dtype, chunked):
    """Length 2 array with [NA, Valid]"""
    # 构造包含 NA 和有效值的字符串数组对象并返回可能分割后的结果
    arr = dtype.construct_array_type()._from_sequence([pd.NA, "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


# 定义一个夹具函数，根据 dtype 和 chunked 参数生成用于排序的数组
@pytest.fixture
def data_for_sorting(dtype, chunked):
    # 构造用于排序的字符串数组对象并返回可能分割后的结果
    arr = dtype.construct_array_type()._from_sequence(["B", "C", "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


# 定义一个夹具函数，根据 dtype 和 chunked 参数生成用于排序的包含缺失值的数组
@pytest.fixture
def data_missing_for_sorting(dtype, chunked):
    # 构造用于排序的包含缺失值的字符串数组对象并返回可能分割后的结果
    arr = dtype.construct_array_type()._from_sequence(["B", pd.NA, "A"], dtype=dtype)
    return maybe_split_array(arr, chunked)


# 定义一个夹具函数，根据 dtype 和 chunked 参数生成用于分组的数组
@pytest.fixture
def data_for_grouping(dtype, chunked):
    # 构造用于分组的字符串数组对象并返回可能分割后的结果
    arr = dtype.construct_array_type()._from_sequence(
        ["B", "B", pd.NA, pd.NA, "A", "A", "B", "C"], dtype=dtype
    )
    return maybe_split_array(arr, chunked)


# 定义一个测试类，继承自 base.ExtensionTests
class TestStringArray(base.ExtensionTests):
    # 定义测试方法，测试字符串相等性
    def test_eq_with_str(self, dtype):
        # 断言字符串类型与存储类型的匹配
        assert dtype == f"string[{dtype.storage}]"
        # 调用父类方法执行字符串相等性测试
        super().test_eq_with_str(dtype)
    def test_is_not_string_type(self, dtype):
        # 检查给定的数据类型是否不是字符串类型
        # 与 BaseDtypeTests.test_is_not_string_type 不同，
        # 因为 StringDtype 是字符串类型
        assert is_string_dtype(dtype)

    def test_view(self, data, request, arrow_string_storage):
        # 如果数据的 dtype 存储方式在 arrow_string_storage 中，
        # 则跳过测试，因为 ArrowStringArray 尚未实现 2D 支持
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        # 调用父类的 test_view 方法进行测试
        super().test_view(data)

    def test_from_dtype(self, data):
        # 基础测试使用 dtype 的字符串表示形式
        # 这里暂时不做任何操作
        pass

    def test_transpose(self, data, request, arrow_string_storage):
        # 如果数据的 dtype 存储方式在 arrow_string_storage 中，
        # 则跳过测试，因为 ArrowStringArray 尚未实现 2D 支持
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        # 调用父类的 test_transpose 方法进行测试
        super().test_transpose(data)

    def test_setitem_preserves_views(self, data, request, arrow_string_storage):
        # 如果数据的 dtype 存储方式在 arrow_string_storage 中，
        # 则跳过测试，因为 ArrowStringArray 尚未实现 2D 支持
        if data.dtype.storage in arrow_string_storage:
            pytest.skip(reason="2D support not implemented for ArrowStringArray")
        # 调用父类的 test_setitem_preserves_views 方法进行测试
        super().test_setitem_preserves_views(data)

    def test_dropna_array(self, data_missing):
        # 对 data_missing 执行 dropna 操作
        result = data_missing.dropna()
        # 预期结果是仅包含非空值的 data_missing 的子集
        expected = data_missing[[1]]
        # 断言 dropna 的结果与预期结果相等
        tm.assert_extension_array_equal(result, expected)

    def test_fillna_no_op_returns_copy(self, data):
        # 从 data 中移除空值
        data = data[~data.isna()]

        # 获取有效值作为填充值
        valid = data[0]
        # 使用有效值填充 data 的空值，返回填充后的副本
        result = data.fillna(valid)
        # 断言 result 不是 data 的同一对象
        assert result is not data
        # 断言填充后的结果与原始 data 相等
        tm.assert_extension_array_equal(result, data)

    def _get_expected_exception(
        self, op_name: str, obj, other
        # 返回预期的异常对象，用于测试异常情况
    ) -> type[Exception] | None:
        # 如果操作名称为 "__divmod__" 或 "__rdivmod__"
        if op_name in ["__divmod__", "__rdivmod__"]:
            # 如果对象是 pandas Series 类型，并且其数据类型是 StringDtype，并且存储格式为 "pyarrow" 或 "pyarrow_numpy"
            if isinstance(obj, pd.Series) and cast(
                StringDtype, tm.get_dtype(obj)
            ).storage in [
                "pyarrow",
                "pyarrow_numpy",
            ]:
                # TODO: 是否应该重新抛出为 TypeError？
                return NotImplementedError
            # 如果其他对象是 pandas Series 类型，并且其数据类型是 StringDtype，并且存储格式为 "pyarrow" 或 "pyarrow_numpy"
            elif isinstance(other, pd.Series) and cast(
                StringDtype, tm.get_dtype(other)
            ).storage in [
                "pyarrow",
                "pyarrow_numpy",
            ]:
                # TODO: 是否应该重新抛出为 TypeError？
                return NotImplementedError
            # 返回 TypeError
            return TypeError
        # 如果操作名称为 "__mod__", "__rmod__", "__pow__", "__rpow__"
        elif op_name in ["__mod__", "__rmod__", "__pow__", "__rpow__"]:
            # 如果对象的数据类型是 StringDtype，并且存储格式为 "pyarrow" 或 "pyarrow_numpy"
            if cast(StringDtype, tm.get_dtype(obj)).storage in [
                "pyarrow",
                "pyarrow_numpy",
            ]:
                return NotImplementedError
            # 返回 TypeError
            return TypeError
        # 如果操作名称为 "__mul__", "__rmul__"
        elif op_name in ["__mul__", "__rmul__"]:
            # 只能将字符串与整数相乘，返回 TypeError
            return TypeError
        # 如果操作名称为 "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__", "__sub__", "__rsub__"
        elif op_name in [
            "__truediv__",
            "__rtruediv__",
            "__floordiv__",
            "__rfloordiv__",
            "__sub__",
            "__rsub__",
        ]:
            # 如果对象的数据类型是 StringDtype，并且存储格式为 "pyarrow" 或 "pyarrow_numpy"
            if cast(StringDtype, tm.get_dtype(obj)).storage in [
                "pyarrow",
                "pyarrow_numpy",
            ]:
                import pyarrow as pa

                # TODO: 是否应该重新抛出为 TypeError？
                return pa.ArrowNotImplementedError
            # 返回 TypeError
            return TypeError

        # 默认情况返回 None
        return None

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # 如果操作名称为 "min" 或 "max"，或者数据类型存储格式为 "pyarrow_numpy" 且操作名称为 "any" 或 "all"
        return (
            op_name in ["min", "max"]
            or ser.dtype.storage == "pyarrow_numpy"  # type: ignore[union-attr]
            and op_name in ("any", "all")
        )

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        # 获取对象的数据类型
        dtype = cast(StringDtype, tm.get_dtype(obj))
        # 如果操作名称为 "__add__" 或 "__radd__"
        if op_name in ["__add__", "__radd__"]:
            # 转换为 dtype 类型
            cast_to = dtype
        # 如果数据类型存储格式为 "pyarrow"
        elif dtype.storage == "pyarrow":
            cast_to = "boolean[pyarrow]"  # type: ignore[assignment]
        # 如果数据类型存储格式为 "pyarrow_numpy"
        elif dtype.storage == "pyarrow_numpy":
            cast_to = np.bool_  # type: ignore[assignment]
        else:
            # 否则转换为 "boolean"
            cast_to = "boolean"  # type: ignore[assignment]
        # 将 pointwise_result 转换为 cast_to 类型并返回
        return pointwise_result.astype(cast_to)

    def test_compare_scalar(self, data, comparison_op):
        # 创建一个 pandas Series 对象
        ser = pd.Series(data)
        # 调用 _compare_other 方法比较 ser 对象与 data、comparison_op 和 "abc"
        self._compare_other(ser, data, comparison_op, "abc")

    @pytest.mark.filterwarnings("ignore:Falling back:pandas.errors.PerformanceWarning")
    def test_groupby_extension_apply(self, data_for_grouping, groupby_apply_op):
        # 调用父类的 test_groupby_extension_apply 方法，传递 data_for_grouping 和 groupby_apply_op
        super().test_groupby_extension_apply(data_for_grouping, groupby_apply_op)
class Test2DCompat(base.Dim2CompatTests):
    @pytest.fixture(autouse=True)
    def arrow_not_supported(self, data):
        # 如果数据类型为 ArrowStringArray，则跳过测试，因为对 ArrowStringArray 尚未实现 2D 支持
        if isinstance(data, ArrowStringArray):
            pytest.skip(reason="2D support not implemented for ArrowStringArray")


def test_searchsorted_with_na_raises(data_for_sorting, as_series):
    # GH50447
    # 将 data_for_sorting 中的值解包给 b, c, a
    b, c, a = data_for_sorting
    # 从 data_for_sorting 中选择索引为 [2, 0, 1] 的值，以得到顺序为 [a, b, c] 的数组
    arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]
    # 将数组 arr 的最后一个元素设置为 pd.NA
    arr[-1] = pd.NA

    if as_series:
        # 如果 as_series 为 True，则将 arr 转换为 Pandas Series
        arr = pd.Series(arr)

    # 错误消息，说明带有 NA 值的数组无法排序
    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    # 使用 pytest 断言，检查是否会引发 ValueError，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)
```