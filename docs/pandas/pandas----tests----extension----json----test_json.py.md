# `D:\src\scipysrc\pandas\pandas\tests\extension\json\test_json.py`

```
import collections
import operator
import sys

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

# We intentionally don't run base.BaseSetitemTests because pandas'
# internals has trouble setting sequences of values into scalar positions.
unhashable = pytest.mark.xfail(reason="Unhashable")


@pytest.fixture
def dtype():
    return JSONDtype()


@pytest.fixture
def data():
    """Length-100 PeriodArray for semantics test."""
    # 生成用于语义测试的长度为100的 PeriodArray 数据
    data = make_data()

    # Why the while loop? NumPy is unable to construct an ndarray from
    # equal-length ndarrays. Many of our operations involve coercing the
    # EA to an ndarray of objects. To avoid random test failures, we ensure
    # that our data is coercible to an ndarray. Several tests deal with only
    # the first two elements, so that's what we'll check.
    # 为什么使用 while 循环？NumPy 无法从长度相等的 ndarrays 构建 ndarray。
    # 我们的许多操作涉及将 EA 强制转换为对象的 ndarray。为避免随机测试失败，
    # 我们确保我们的数据可转换为 ndarray。几个测试只涉及前两个元素，因此我们将检查它们。
    while len(data[0]) == len(data[1]):
        data = make_data()

    return JSONArray(data)


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    # 包含两个元素的数组，[NA, Valid]
    return JSONArray([{}, {"a": 10}])


@pytest.fixture
def data_for_sorting():
    # 用于排序的 JSONArray 数据
    return JSONArray([{"b": 1}, {"c": 4}, {"a": 2, "c": 3}])


@pytest.fixture
def data_missing_for_sorting():
    # 用于排序的 JSONArray 数据，包含缺失值
    return JSONArray([{"b": 1}, {}, {"a": 4}])


@pytest.fixture
def na_cmp():
    # 比较操作符，用于比较缺失值
    return operator.eq


@pytest.fixture
def data_for_grouping():
    # 用于分组操作的 JSONArray 数据
    return JSONArray(
        [
            {"b": 1},
            {"b": 1},
            {},
            {},
            {"a": 0, "c": 2},
            {"a": 0, "c": 2},
            {"b": 1},
            {"c": 2},
        ]
    )


class TestJSONArray(base.ExtensionTests):
    @pytest.mark.xfail(
        reason="comparison method not implemented for JSONArray (GH-37867)"
    )
    def test_contains(self, data):
        # GH-37867
        # 测试包含操作，预期失败，因为 JSONArray 尚未实现比较方法
        super().test_contains(data)

    @pytest.mark.xfail(reason="not implemented constructor from dtype")
    def test_from_dtype(self, data):
        # construct from our dtype & string dtype
        # 从我们的 dtype 和 string dtype 构造对象，预期失败
        super().test_from_dtype(data)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_no_data_with_index(self, dtype, na_value):
        # RecursionError: maximum recursion depth exceeded in comparison
        # 递归错误：比较中超出了最大递归深度
        rec_limit = sys.getrecursionlimit()
        try:
            # Limit to avoid stack overflow on Windows CI
            # 限制以避免在 Windows CI 上的堆栈溢出
            sys.setrecursionlimit(100)
            super().test_series_constructor_no_data_with_index(dtype, na_value)
        finally:
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="RecursionError, GH-33900")
    def test_series_constructor_scalar_na_with_index(self, dtype, na_value):
        # 设置递归深度限制，以避免在比较中出现递归错误
        rec_limit = sys.getrecursionlimit()
        try:
            # 临时将递归深度限制设置为100，以避免在Windows CI中发生堆栈溢出
            sys.setrecursionlimit(100)
            # 调用父类方法来测试带索引的标量 NA 值的系列构造
            super().test_series_constructor_scalar_na_with_index(dtype, na_value)
        finally:
            # 恢复原始的递归深度限制
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="collection as scalar, GH-33901")
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # TypeError: 所有的值必须是<class 'collections.abc.Mapping'>类型
        rec_limit = sys.getrecursionlimit()
        try:
            # 临时将递归深度限制设置为100，以避免在Windows CI中发生堆栈溢出
            sys.setrecursionlimit(100)
            # 调用父类方法来测试带索引的标量的系列构造
            super().test_series_constructor_scalar_with_index(data, dtype)
        finally:
            # 恢复原始的递归深度限制
            sys.setrecursionlimit(rec_limit)

    @pytest.mark.xfail(reason="Different definitions of NA")
    def test_stack(self):
        """
        测试 .astype(object).stack() 方法。
        如果 `data` 中有任何缺失值，由于我们将 `{}` 视为 NA，但 `.astype(object)` 不视为 NA，因此可能会得到不同的行。
        """
        super().test_stack()

    @pytest.mark.xfail(reason="dict for NA")
    def test_unstack(self, data, index):
        # 基本测试中，预期的 NA 值为 NaN。
        # 其他情况都匹配
        return super().test_unstack(data, index)

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_series(self):
        """我们在 fillna 中将字典视为映射，而不是标量。"""
        super().test_fillna_series()

    @pytest.mark.xfail(reason="Setting a dict as a scalar")
    def test_fillna_frame(self):
        """我们在 fillna 中将字典视为映射，而不是标量。"""
        super().test_fillna_frame()

    def test_fillna_with_none(self, data_missing):
        # GH#57723
        # 如果 EA（预期值）对 None 没有特殊逻辑，则会引发异常，与 pandas 不同，后者将 None 解释为 dtype 的 NA 值。
        with pytest.raises(AssertionError):
            super().test_fillna_with_none(data_missing)

    @pytest.mark.xfail(reason="fill value is a dictionary, takes incorrect code path")
    def test_fillna_limit_frame(self, data_missing):
        # GH#58001
        super().test_fillna_limit_frame(data_missing)

    @pytest.mark.xfail(reason="fill value is a dictionary, takes incorrect code path")
    def test_fillna_limit_series(self, data_missing):
        # GH#58001
        super().test_fillna_limit_frame(data_missing)
    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [  # 参数化测试参数，定义了不同的测试用例
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),  # 外部区域测试用例
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),  # 外部区域测试用例
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),  # 外部区域测试用例
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),  # 外部区域测试用例
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),  # 内部区域测试用例
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),  # 内部区域测试用例
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),  # 内部区域测试用例
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),  # 内部区域测试用例
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # GH#56616
        # 使用 pytest 的 raises 断言检查是否抛出 NotImplementedError 异常，匹配特定的错误消息
        msg = "JSONArray does not implement limit_area"
        with pytest.raises(NotImplementedError, match=msg):
            # 调用父类的 test_ffill_limit_area 方法，验证是否抛出预期异常
            super().test_ffill_limit_area(
                data_missing, limit_area, input_ilocs, expected_ilocs
            )

    @unhashable
    def test_value_counts(self, all_data, dropna):
        # 调用父类的 test_value_counts 方法，使用传入的 all_data 和 dropna 参数
        super().test_value_counts(all_data, dropna)

    @unhashable
    def test_value_counts_with_normalize(self, data):
        # 调用父类的 test_value_counts_with_normalize 方法，使用传入的 data 参数
        super().test_value_counts_with_normalize(data)

    @unhashable
    def test_sort_values_frame(self):
        # TODO (EA.factorize): see if _values_for_factorize allows this.
        # 调用父类的 test_sort_values_frame 方法，有一个待办事项提醒
        super().test_sort_values_frame()

    @pytest.mark.xfail(reason="combine for JSONArray not supported")
    def test_combine_le(self, data_repeated):
        # 调用父类的 test_combine_le 方法，标记为预期失败，原因是 JSONArray 不支持该操作
        super().test_combine_le(data_repeated)

    @pytest.mark.xfail(
        reason="combine for JSONArray not supported - "
        "may pass depending on random data",
        strict=False,
        raises=AssertionError,
    )
    def test_combine_first(self, data):
        # 调用父类的 test_combine_first 方法，标记为预期失败，可能取决于随机数据
        super().test_combine_first(data)

    @pytest.mark.xfail(reason="broadcasting error")
    def test_where_series(self, data, na_value):
        # 调用父类的 test_where_series 方法，标记为预期失败，由于广播错误
        super().test_where_series(data, na_value)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_searchsorted(self, data_for_sorting):
        # 调用父类的 test_searchsorted 方法，标记为预期失败，由于无法比较字典
        super().test_searchsorted(data_for_sorting)

    @pytest.mark.xfail(reason="Can't compare dicts.")
    def test_equals(self, data, na_value, as_series):
        # 调用父类的 test_equals 方法，标记为预期失败，由于无法比较字典
        super().test_equals(data, na_value, as_series)

    @pytest.mark.skip("fill-value is interpreted as a dict of values")
    def test_fillna_copy_frame(self, data_missing):
        # 调用父类的 test_fillna_copy_frame 方法，跳过测试，填充值被解释为字典的值
        super().test_fillna_copy_frame(data_missing)

    @pytest.mark.xfail(reason="Fails with CoW")
    def test_equals_same_data_different_object(self, data):
        # 调用父类的 test_equals_same_data_different_object 方法，标记为预期失败，由于 CoW（Copy-on-Write）问题
        super().test_equals_same_data_different_object(data)

    @pytest.mark.xfail(reason="failing on np.array(self, dtype=str)")
    def test_searchsorted(self, data_for_sorting):
        # 调用父类的 test_searchsorted 方法，标记为预期失败，由于在 np.array(self, dtype=str) 上失败
        super().test_searchsorted(data_for_sorting)
    def test_astype_str(self):
        """
        This currently fails in NumPy on np.array(self, dtype=str) with

        *** ValueError: setting an array element with a sequence
        """
        # 调用父类的 test_astype_str 方法
        super().test_astype_str()

    @unhashable
    def test_groupby_extension_transform(self):
        """
        This currently fails in Series.name.setter, since the
        name must be hashable, but the value is a dictionary.
        I think this is what we want, i.e. `.name` should be the original
        values, and not the values for factorization.
        """
        # 调用父类的 test_groupby_extension_transform 方法
        super().test_groupby_extension_transform()

    @unhashable
    def test_groupby_extension_apply(self):
        """
        This fails in Index._do_unique_check with

        >   hash(val)
        E   TypeError: unhashable type: 'UserDict' with

        I suspect that once we support Index[ExtensionArray],
        we'll be able to dispatch unique.
        """
        # 调用父类的 test_groupby_extension_apply 方法
        super().test_groupby_extension_apply()

    @unhashable
    def test_groupby_extension_agg(self):
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        # 调用父类的 test_groupby_extension_agg 方法
        super().test_groupby_extension_agg()

    @unhashable
    def test_groupby_extension_no_sort(self):
        """
        This fails when we get to tm.assert_series_equal when left.index
        contains dictionaries, which are not hashable.
        """
        # 调用父类的 test_groupby_extension_no_sort 方法
        super().test_groupby_extension_no_sort()

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        """
        If the length of the first data element is not 1, mark this test as expected to fail
        due to coercion issues in converting to Series.
        """
        if len(data[0]) != 1:
            mark = pytest.mark.xfail(reason="raises in coercing to Series")
            request.applymarker(mark)
        # 调用父类的 test_arith_frame_with_scalar 方法
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_compare_array(self, data, comparison_op, request):
        """
        If the comparison operation name is 'eq' or 'ne', mark this test as expected to fail
        due to unimplemented comparison methods.
        """
        if comparison_op.__name__ in ["eq", "ne"]:
            mark = pytest.mark.xfail(reason="Comparison methods not implemented")
            request.applymarker(mark)
        # 调用父类的 test_compare_array 方法
        super().test_compare_array(data, comparison_op)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_mixed(self, data):
        # 调用父类的 test_setitem_loc_scalar_mixed 方法，标记为预期失败
        super().test_setitem_loc_scalar_mixed(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        # 调用父类的 test_setitem_loc_scalar_multiple_homogoneous 方法，标记为预期失败
        super().test_setitem_loc_scalar_multiple_homogoneous(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_mixed(self, data):
        # 调用父类的 test_setitem_iloc_scalar_mixed 方法，标记为预期失败
        super().test_setitem_iloc_scalar_mixed(data)

    @pytest.mark.xfail(reason="ValueError: Must have equal len keys and value")
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        # 调用父类的 test_setitem_iloc_scalar_multiple_homogoneous 方法，标记为预期失败
        super().test_setitem_iloc_scalar_multiple_homogoneous(data)
    # 使用 pytest 的 parametrize 装饰器定义多个测试参数组合，对应的参数包括 mask 和 idx
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),  # numpy 数组，包含布尔值
            pd.array([True, True, True, False, False], dtype="boolean"),  # pandas boolean 数组
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),  # pandas boolean 数组，包含 NA 值
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],  # 参数组合的标识
    )
    # 测试方法：测试使用 mask 进行设置项目时的行为
    def test_setitem_mask(self, data, mask, box_in_series, request):
        # 如果 box_in_series 为 True，则标记测试为预期失败，原因是使用不同长度的类列表索引器无法设置
        if box_in_series:
            mark = pytest.mark.xfail(
                reason="cannot set using a list-like indexer with a different length"
            )
            request.applymarker(mark)
        # 如果 mask 不是 numpy 数组，则标记测试为预期失败，原因是可能引发不希望的 DeprecationWarning
        elif not isinstance(mask, np.ndarray):
            mark = pytest.mark.xfail(reason="Issues unwanted DeprecationWarning")
            request.applymarker(mark)
        # 调用超类的测试方法，测试使用 mask 设置项目时的行为
        super().test_setitem_mask(data, mask, box_in_series)

    # 测试方法：测试使用 mask 设置项目时引发异常的情况
    def test_setitem_mask_raises(self, data, box_in_series, request):
        # 如果 box_in_series 为 False，则标记测试为预期失败，原因是预期未引发异常
        if not box_in_series:
            mark = pytest.mark.xfail(reason="Fails to raise")
            request.applymarker(mark)
        # 调用超类的测试方法，测试使用 mask 设置项目时引发异常的情况
        super().test_setitem_mask_raises(data, box_in_series)

    # 测试方法：测试使用带有 NA 值的布尔数组 mask 设置项目时的行为，预期为测试失败
    @pytest.mark.xfail(
        reason="cannot set using a list-like indexer with a different length"
    )
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        # 调用超类的测试方法，测试使用带有 NA 值的布尔数组 mask 设置项目时的行为
        super().test_setitem_mask_boolean_array_with_na(data, box_in_series)

    # 使用 pytest 的 parametrize 装饰器定义多个测试参数组合，对应的参数包括 idx 和 box_in_series
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],  # 索引器参数的多种类型：列表、pandas Int64 数组、numpy 数组
        ids=["list", "integer-array", "numpy-array"],  # 参数组合的标识
    )
    # 测试方法：测试使用整数数组索引器设置项目时的行为
    def test_setitem_integer_array(self, data, idx, box_in_series, request):
        # 如果 box_in_series 为 True，则标记测试为预期失败，原因是使用不同长度的类列表索引器无法设置
        if box_in_series:
            mark = pytest.mark.xfail(
                reason="cannot set using a list-like indexer with a different length"
            )
            request.applymarker(mark)
        # 调用超类的测试方法，测试使用整数数组索引器设置项目时的行为
        super().test_setitem_integer_array(data, idx, box_in_series)

    # 使用 pytest 的 parametrize 装饰器定义多个测试参数组合，对应的参数包括 idx 和 box_in_series
    @pytest.mark.xfail(reason="list indices must be integers or slices, not NAType")
    @pytest.mark.parametrize(
        "idx, box_in_series",
        [
            ([0, 1, 2, pd.NA], False),  # 索引器为包含 NA 值的列表，box_in_series 为 False
            pytest.param(
                [0, 1, 2, pd.NA], True, marks=pytest.mark.xfail(reason="GH-31948")
            ),  # 索引器为包含 NA 值的列表，box_in_series 为 True，标记为预期失败
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), False),  # 索引器为包含 NA 值的 pandas Int64 数组，box_in_series 为 False
            (pd.array([0, 1, 2, pd.NA], dtype="Int64"), True),  # 索引器为包含 NA 值的 pandas Int64 数组，box_in_series 为 True
        ],
        ids=["list-False", "list-True", "integer-array-False", "integer-array-True"],  # 参数组合的标识
    )
    # 测试方法：测试使用包含 NA 值的索引器设置项目时引发异常的情况
    def test_setitem_integer_with_missing_raises(self, data, idx, box_in_series):
        # 调用超类的测试方法，测试使用包含 NA 值的索引器设置项目时引发异常的情况
        super().test_setitem_integer_with_missing_raises(data, idx, box_in_series)

    # 测试方法：测试使用标量键序列引发异常的情况，预期为测试失败
    @pytest.mark.xfail(reason="Fails to raise")
    def test_setitem_scalar_key_sequence_raise(self, data):
        # 调用超类的测试方法，测试使用标量键序列引发异常的情况
        super().test_setitem_scalar_key_sequence_raise(data)
    # 如果测试函数名中包含 "full_slice"，则标记该测试为预期失败，原因是“slice is not iterable”
    if "full_slice" in request.node.name:
        mark = pytest.mark.xfail(reason="slice is not iterable")
        request.applymarker(mark)

    # 调用父类的 test_setitem_with_expansion_dataframe_column 方法来执行测试
    super().test_setitem_with_expansion_dataframe_column(data, full_indexer)

@pytest.mark.xfail(reason="slice is not iterable")
# 标记该测试函数为预期失败，原因是“slice is not iterable”
def test_setitem_frame_2d_values(self, data):
    # 调用父类的 test_setitem_frame_2d_values 方法来执行测试
    super().test_setitem_frame_2d_values(data)

@pytest.mark.xfail(reason="cannot set using a list-like indexer with a different length")
# 标记该测试函数为预期失败，原因是“cannot set using a list-like indexer with a different length”
@pytest.mark.parametrize("setter", ["loc", None])
def test_setitem_mask_broadcast(self, data, setter):
    # 调用父类的 test_setitem_mask_broadcast 方法来执行测试
    super().test_setitem_mask_broadcast(data, setter)

@pytest.mark.xfail(reason="cannot set using a slice indexer with a different length")
# 标记该测试函数为预期失败，原因是“cannot set using a slice indexer with a different length”
def test_setitem_slice(self, data, box_in_series):
    # 调用父类的 test_setitem_slice 方法来执行测试
    super().test_setitem_slice(data, box_in_series)

@pytest.mark.xfail(reason="slice object is not iterable")
# 标记该测试函数为预期失败，原因是“slice object is not iterable”
def test_setitem_loc_iloc_slice(self, data):
    # 调用父类的 test_setitem_loc_iloc_slice 方法来执行测试
    super().test_setitem_loc_iloc_slice(data)

@pytest.mark.xfail(reason="slice object is not iterable")
# 标记该测试函数为预期失败，原因是“slice object is not iterable”
def test_setitem_slice_mismatch_length_raises(self, data):
    # 调用父类的 test_setitem_slice_mismatch_length_raises 方法来执行测试
    super().test_setitem_slice_mismatch_length_raises(data)

@pytest.mark.xfail(reason="slice object is not iterable")
# 标记该测试函数为预期失败，原因是“slice object is not iterable”
def test_setitem_slice_array(self, data):
    # 调用父类的 test_setitem_slice_array 方法来执行测试
    super().test_setitem_slice_array(data)

@pytest.mark.xfail(reason="Fail to raise")
# 标记该测试函数为预期失败，原因是“Fail to raise”
def test_setitem_invalid(self, data, invalid_scalar):
    # 调用父类的 test_setitem_invalid 方法来执行测试
    super().test_setitem_invalid(data, invalid_scalar)

@pytest.mark.xfail(reason="only integer scalar arrays can be converted")
# 标记该测试函数为预期失败，原因是“only integer scalar arrays can be converted”
def test_setitem_2d_values(self, data):
    # 调用父类的 test_setitem_2d_values 方法来执行测试
    super().test_setitem_2d_values(data)

@pytest.mark.xfail(reason="data type 'json' not understood")
# 标记该测试函数为预期失败，原因是“data type 'json' not understood”
@pytest.mark.parametrize("engine", ["c", "python"])
def test_EA_types(self, engine, data, request):
    # 调用父类的 test_EA_types 方法来执行测试
    super().test_EA_types(engine, data, request)
# 自定义函数，用于比较两个 Pandas Series 对象是否相等，如果不相等则抛出 AssertionError
def custom_assert_series_equal(left, right, *args, **kwargs):
    # NumPy doesn't handle an array of equal-length UserDicts.
    # NumPy 无法处理等长 UserDicts 数组。
    # The default assert_series_equal eventually does a
    # Series.values, which raises. We work around it by
    # converting the UserDicts to dicts.
    # 默认的 assert_series_equal 最终会执行 Series.values 操作，导致异常。我们通过将 UserDicts 转换为 dicts 来解决这个问题。
    if left.dtype.name == "json":
        # 如果 left 的数据类型是 "json"，则进行如下处理：
        assert left.dtype == right.dtype
        # 断言 left 和 right 的数据类型相同
        left = pd.Series(
            JSONArray(left.values.astype(object)), index=left.index, name=left.name
        )
        # 将 left 转换为 Pandas Series 对象，处理数据类型为 "json" 的情况
        right = pd.Series(
            JSONArray(right.values.astype(object)),
            index=right.index,
            name=right.name,
        )
        # 将 right 转换为 Pandas Series 对象，处理数据类型为 "json" 的情况
    # 使用 Pandas 的 assert_series_equal 函数比较 left 和 right 是否相等，*args 和 **kwargs 用于传递额外参数
    tm.assert_series_equal(left, right, *args, **kwargs)


# 自定义函数，用于比较两个 Pandas DataFrame 对象是否相等，如果不相等则抛出 AssertionError
def custom_assert_frame_equal(left, right, *args, **kwargs):
    obj_type = kwargs.get("obj", "DataFrame")
    # 获取关键字参数 obj 的值，默认为 "DataFrame"
    tm.assert_index_equal(
        left.columns,
        right.columns,
        exact=kwargs.get("check_column_type", "equiv"),
        check_names=kwargs.get("check_names", True),
        check_exact=kwargs.get("check_exact", False),
        check_categorical=kwargs.get("check_categorical", True),
        obj=f"{obj_type}.columns",
    )
    # 使用 Pandas 的 assert_index_equal 函数比较 left 和 right 的列索引是否相等，*args 和 **kwargs 用于传递额外参数

    jsons = (left.dtypes == "json").index
    # 获取数据类型为 "json" 的列的索引列表

    for col in jsons:
        # 遍历所有数据类型为 "json" 的列
        custom_assert_series_equal(left[col], right[col], *args, **kwargs)
        # 使用自定义函数比较 left 和 right 的每个 "json" 列是否相等，*args 和 **kwargs 用于传递额外参数

    left = left.drop(columns=jsons)
    # 从 left 中删除所有数据类型为 "json" 的列
    right = right.drop(columns=jsons)
    # 从 right 中删除所有数据类型为 "json" 的列
    tm.assert_frame_equal(left, right, *args, **kwargs)
    # 使用 Pandas 的 assert_frame_equal 函数比较 left 和 right 是否相等，*args 和 **kwargs 用于传递额外参数


# 测试自定义断言函数的功能
def test_custom_asserts():
    # This would always trigger the KeyError from trying to put
    # an array of equal-length UserDicts inside an ndarray.
    # 这将始终触发 KeyError，因为试图将等长 UserDicts 数组放入 ndarray 中会导致异常。
    data = JSONArray(
        [
            collections.UserDict({"a": 1}),
            collections.UserDict({"b": 2}),
            collections.UserDict({"c": 3}),
        ]
    )
    # 创建包含 UserDict 对象的 JSONArray 对象 data

    a = pd.Series(data)
    # 创建 Pandas Series 对象 a，包含 data 的内容
    custom_assert_series_equal(a, a)
    # 使用自定义函数比较 a 和 a 是否相等

    custom_assert_frame_equal(a.to_frame(), a.to_frame())
    # 使用自定义函数比较 a 和 a 的 DataFrame 形式是否相等

    b = pd.Series(data.take([0, 0, 1]))
    # 创建另一个 Pandas Series 对象 b，包含 data 的部分内容
    msg = r"Series are different"
    # 错误消息字符串

    with pytest.raises(AssertionError, match=msg):
        # 断言抛出 AssertionError，并且错误消息包含指定内容
        custom_assert_series_equal(a, b)

    with pytest.raises(AssertionError, match=msg):
        # 断言抛出 AssertionError，并且错误消息包含指定内容
        custom_assert_frame_equal(a.to_frame(), b.to_frame())
```