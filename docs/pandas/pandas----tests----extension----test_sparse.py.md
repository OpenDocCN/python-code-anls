# `D:\src\scipysrc\pandas\pandas\tests\extension\test_sparse.py`

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

# Import necessary libraries and modules
import numpy as np
import pytest

import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.arrays import SparseArray
from pandas.tests.extension import base


def make_data(fill_value):
    # Initialize a random number generator
    rng = np.random.default_rng(2)
    # Generate random data based on the fill_value type
    if np.isnan(fill_value):
        data = rng.uniform(size=100)  # Generate uniform data if fill_value is NaN
    else:
        data = rng.integers(1, 100, size=100, dtype=int)  # Generate integer data within range if fill_value is a number
        if data[0] == data[1]:
            data[0] += 1  # Ensure the first two elements are not identical

    data[2::3] = fill_value  # Assign fill_value to every third element starting from index 2
    return data


@pytest.fixture
def dtype():
    return SparseDtype()


@pytest.fixture(params=[0, np.nan])
def data(request):
    """Length-100 PeriodArray for semantics test."""
    # Create a SparseArray with data generated using make_data and specified fill_value
    res = SparseArray(make_data(request.param), fill_value=request.param)
    return res


@pytest.fixture
def data_for_twos():
    # Create a SparseArray filled with the value 2
    return SparseArray(np.ones(100) * 2)


@pytest.fixture(params=[0, np.nan])
def data_missing(request):
    """Length 2 array with [NA, Valid]"""
    # Return a SparseArray with specified missing value and a valid value
    return SparseArray([np.nan, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_repeated(request):
    """Return different versions of data for count times"""

    def gen(count):
        for _ in range(count):
            yield SparseArray(make_data(request.param), fill_value=request.param)

    return gen


@pytest.fixture(params=[0, np.nan])
def data_for_sorting(request):
    # Return a SparseArray for sorting tests with specified fill_value
    return SparseArray([2, 3, 1], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_missing_for_sorting(request):
    # Return a SparseArray with missing value for sorting tests
    return SparseArray([2, np.nan, 1], fill_value=request.param)


@pytest.fixture
def na_cmp():
    # Return a lambda function to compare for NaN values
    return lambda left, right: pd.isna(left) and pd.isna(right)


@pytest.fixture(params=[0, np.nan])
def data_for_grouping(request):
    # Return a SparseArray with data suitable for grouping, including NaN values
    return SparseArray([1, 1, np.nan, np.nan, 2, 2, 1, 3], fill_value=request.param)


@pytest.fixture(params=[0, np.nan])
def data_for_compare(request):
    # Return a SparseArray with data suitable for comparison, including NaN values
    return SparseArray([0, 0, np.nan, -2, -1, 4, 2, 3, 0, 0], fill_value=request.param)


class TestSparseArray(base.ExtensionTests):
    def _supports_reduction(self, obj, op_name: str) -> bool:
        # Always return True for supporting reduction operations
        return True

    @pytest.mark.parametrize("skipna", [True, False])
    # 定义测试函数，用于对数值序列进行缩减操作的测试
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        # 如果缩减操作为以下任意一种，标记为预期失败，添加失败原因到测试标记
        if all_numeric_reductions in [
            "prod",
            "median",
            "var",
            "std",
            "sem",
            "skew",
            "kurt",
        ]:
            mark = pytest.mark.xfail(
                reason="This should be viable but is not implemented"
            )
            request.node.add_marker(mark)
        # 如果缩减操作为以下任意一种，并且数据类型为浮点数，并且不跳过NA值，则标记为预期失败
        elif (
            all_numeric_reductions in ["sum", "max", "min", "mean"]
            and data.dtype.kind == "f"
            and not skipna
        ):
            mark = pytest.mark.xfail(reason="getting a non-nan float")
            request.node.add_marker(mark)

        # 调用父类的测试函数来执行测试
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    # 参数化测试函数，用于对数据框进行缩减操作的测试
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        # 如果缩减操作为以下任意一种，标记为预期失败，添加失败原因到测试标记
        if all_numeric_reductions in [
            "prod",
            "median",
            "var",
            "std",
            "sem",
            "skew",
            "kurt",
        ]:
            mark = pytest.mark.xfail(
                reason="This should be viable but is not implemented"
            )
            request.node.add_marker(mark)
        # 如果缩减操作为以下任意一种，并且数据类型为浮点数，并且不跳过NA值，则标记为预期失败
        elif (
            all_numeric_reductions in ["sum", "max", "min", "mean"]
            and data.dtype.kind == "f"
            and not skipna
        ):
            mark = pytest.mark.xfail(reason="ExtensionArray NA mask are different")
            request.node.add_marker(mark)

        # 调用父类的测试函数来执行测试
        super().test_reduce_frame(data, all_numeric_reductions, skipna)

    # 检查数据是否为稀疏类型的整数，若是则跳过测试
    def _check_unsupported(self, data):
        if data.dtype == SparseDtype(int, 0):
            pytest.skip("Can't store nan in int array.")

    # 测试混合数据类型的数据框连接操作
    def test_concat_mixed_dtypes(self, data):
        # 创建三个数据框，分别包含不同类型的数据列
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # 连接三个数据框
        result = pd.concat(dfs)
        # 构建预期结果，将每个数据框中的列转换为对象类型后连接
        expected = pd.concat(
            [x.apply(lambda s: np.asarray(s).astype(object)) for x in dfs]
        )
        # 断言连接结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 对测试函数进行参数化，并标记忽略警告信息
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    @pytest.mark.parametrize(
        "columns",
        [
            ["A", "B"],
            pd.MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["outer", "inner"]
            ),
        ],
    )
    @pytest.mark.parametrize("future_stack", [True, False])
    def test_stack(self, data, columns, future_stack):
        # 调用父类的测试函数来执行测试
        super().test_stack(data, columns, future_stack)
    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_concat_columns 方法，执行数据列的拼接操作
    super().test_concat_columns(data, na_value)

    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_concat_extension_arrays_copy_false 方法，执行拼接扩展数组操作（拷贝设置为假）
    super().test_concat_extension_arrays_copy_false(data, na_value)

    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_align 方法，执行数据对齐操作
    super().test_align(data, na_value)

    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_align_frame 方法，执行数据框架对齐操作
    super().test_align_frame(data, na_value)

    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_align_series_frame 方法，执行序列框架对齐操作
    super().test_align_series_frame(data, na_value)

    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_merge 方法，执行数据合并操作
    super().test_merge(data, na_value)

    # 根据给定数据创建 Pandas Series 对象，指定索引为原索引的两倍
    ser = pd.Series(data, index=[2 * i for i in range(len(data))])
    # 如果 Series 中存在 NaN 值，验证 get 方法和 iloc 方法的返回值均为 NaN
    if np.isnan(ser.values.fill_value):
        assert np.isnan(ser.get(4)) and np.isnan(ser.iloc[2])
    else:
        # 否则，验证 get 方法和 iloc 方法的返回值是否相等
        assert ser.get(4) == ser.iloc[2]
    # 验证 get 方法和 iloc 方法的返回值是否正确
    assert ser.get(2) == ser.iloc[1]

    # 检查数据是否支持，如果不支持则抛出异常
    self._check_unsupported(data)
    # 调用父类的 test_reindex 方法，执行重新索引操作
    super().test_reindex(data, na_value)

    # 使用 data_missing 创建 SparseArray 对象
    sarr = SparseArray(data_missing)
    # 根据 data_missing 的类型创建预期的数据类型
    expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
    # 创建预期的 SparseArray 对象
    expected = SparseArray([True, False], dtype=expected_dtype)
    # 调用 SparseArray 的 isna 方法，得到结果
    result = sarr.isna()
    # 验证结果与预期是否相等
    tm.assert_sp_array_equal(result, expected)

    # 将 SparseArray 对象中的缺失值填充为 0
    sarr = sarr.fillna(0)
    # 根据 data_missing 的类型创建预期的数据类型
    expected_dtype = SparseDtype(bool, pd.isna(data_missing.dtype.fill_value))
    # 创建预期的 SparseArray 对象，填充值为 False
    expected = SparseArray([False, False], fill_value=False, dtype=expected_dtype)
    # 验证 isna 方法的结果与预期是否相等
    tm.assert_equal(sarr.isna(), expected)

    # 调用父类的 test_fillna_no_op_returns_copy 方法，测试填充缺失值且不产生副本的情况
    super().test_fillna_no_op_returns_copy(data)

@pytest.mark.xfail(reason="Unsupported")
def test_fillna_series(self, data_missing):
    # 此测试用例标记为失败，因为不支持填充 Series 的操作
    # TODO: 当我们传递 data_missing 时，0 填充案例将通过测试
    super().test_fillna_series()

# 使用 data_missing 创建 DataFrame 对象，填充缺失值
def test_fillna_frame(self, data_missing):
    # 必须重写以指定 fill_value 将发生变化
    fill_value = data_missing[1]

    # 使用 data_missing 创建 DataFrame 对象，并填充缺失值
    result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

    # 根据 data_missing 的 fill_value 是否为 NaN，创建预期的数据类型
    if pd.isna(data_missing.fill_value):
        dtype = SparseDtype(data_missing.dtype, fill_value)
    else:
        dtype = data_missing.dtype

    # 创建预期的 DataFrame 对象
    expected = pd.DataFrame(
        {
            "A": data_missing._from_sequence([fill_value, fill_value], dtype=dtype),
            "B": [1, 2],
        }
    )

    # 验证结果 DataFrame 与预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
    # 测试在DataFrame或Series中使用fillna方法时，限制参数不为None时抛出值错误异常
    def test_fillna_limit_frame(self, data_missing):
        # GH#58001
        with pytest.raises(ValueError, match="limit must be None"):
            super().test_fillna_limit_frame(data_missing)

    # 测试在Series中使用fillna方法时，限制参数不为None时抛出值错误异常
    def test_fillna_limit_series(self, data_missing):
        # GH#58001
        with pytest.raises(ValueError, match="limit must be None"):
            super().test_fillna_limit_frame(data_missing)

    # 预期_combine_le方法返回的数据类型为稀疏布尔型
    _combine_le_expected_dtype = "Sparse[bool]"

    # 测试在DataFrame中使用fillna方法时，copy参数为False时的情况
    def test_fillna_copy_frame(self, data_missing):
        arr = data_missing.take([1, 1])
        df = pd.DataFrame({"A": arr}, copy=False)

        filled_val = df.iloc[0, 0]
        result = df.fillna(filled_val)

        if hasattr(df._mgr, "blocks"):
            assert df.values.base is result.values.base
        assert df.A._values.to_dense() is arr.to_dense()

    # 测试在Series中使用fillna方法时，copy参数为False时的情况
    def test_fillna_copy_series(self, data_missing):
        arr = data_missing.take([1, 1])
        ser = pd.Series(arr, copy=False)

        filled_val = ser[0]
        result = ser.fillna(filled_val)

        assert ser._values is result._values
        assert ser._values.to_dense() is arr.to_dense()

    # 标记该测试为预期失败，原因是“不适用”
    @pytest.mark.xfail(reason="Not Applicable")
    def test_fillna_length_mismatch(self, data_missing):
        super().test_fillna_length_mismatch(data_missing)

    # 测试在Series中使用where方法时的情况
    def test_where_series(self, data, na_value):
        assert data[0] != data[1]
        cls = type(data)
        a, b = data[:2]

        ser = pd.Series(cls._from_sequence([a, a, b, b], dtype=data.dtype))

        cond = np.array([True, True, False, False])
        result = ser.where(cond)

        new_dtype = SparseDtype("float", 0.0)
        expected = pd.Series(
            cls._from_sequence([a, a, na_value, na_value], dtype=new_dtype)
        )
        tm.assert_series_equal(result, expected)

        other = cls._from_sequence([a, b, a, b], dtype=data.dtype)
        cond = np.array([True, False, True, True])
        result = ser.where(cond, other)
        expected = pd.Series(cls._from_sequence([a, b, b, b], dtype=data.dtype))
        tm.assert_series_equal(result, expected)

    # 测试在性能警告存在的情况下使用searchsorted方法
    def test_searchsorted(self, performance_warning, data_for_sorting, as_series):
        with tm.assert_produces_warning(performance_warning, check_stacklevel=False):
            super().test_searchsorted(data_for_sorting, as_series)

    # 测试在数据进行shift操作时，periods参数为0时返回的对象是副本而不是相同对象
    def test_shift_0_periods(self, data):
        # GH#33856 shifting with periods=0 should return a copy, not same obj
        result = data.shift(0)

        data._sparse_values[0] = data._sparse_values[1]
        assert result._sparse_values[0] != result._sparse_values[1]

    # 标记为参数化测试，测试在所有数据为缺失值的情况下使用argmin或argmax方法
    @pytest.mark.parametrize("method", ["argmax", "argmin"])
    def test_argmin_argmax_all_na(self, method, data, na_value):
        # overriding because Sparse[int64, 0] cannot handle na_value
        self._check_unsupported(data)
        super().test_argmin_argmax_all_na(method, data, na_value)

    # 标记为参数化测试，测试在不同的数据类型（pd.array, pd.Series, pd.DataFrame）下的行为
    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        # 检查是否支持给定的数据类型
        self._check_unsupported(data)
        # 调用父类方法进行相等性测试
        super().test_equals(data, na_value, as_series, box)

    @pytest.mark.parametrize(
        "func, na_action, expected",
        [
            (lambda x: x, None, SparseArray([1.0, np.nan])),
            (lambda x: x, "ignore", SparseArray([1.0, np.nan])),
            (str, None, SparseArray(["1.0", "nan"], fill_value="nan")),
            (str, "ignore", SparseArray(["1.0", np.nan])),
        ],
    )
    def test_map(self, func, na_action, expected):
        # GH52096：相关的 GitHub issue 编号
        data = SparseArray([1, np.nan])
        # 使用 map 方法进行映射操作
        result = data.map(func, na_action=na_action)
        # 断言映射后的结果与期望结果相等
        tm.assert_extension_array_equal(result, expected)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map_raises(self, data, na_action):
        # GH52096：相关的 GitHub issue 编号
        # 测试在特定情况下 map 方法是否会引发 ValueError 异常
        msg = "fill value in the sparse values not supported"
        with pytest.raises(ValueError, match=msg):
            data.map(lambda x: np.nan, na_action=na_action)

    @pytest.mark.xfail(raises=TypeError, reason="no sparse StringDtype")
    def test_astype_string(self, data, nullable_string_dtype):
        # TODO: this fails bc we do not pass through nullable_string_dtype;
        #  If we did, the 0-cases would xpass
        # 测试将稀疏数组转换为字符串类型时是否会引发 TypeError 异常
        super().test_astype_string(data)

    series_scalar_exc = None
    frame_scalar_exc = None
    divmod_exc = None
    series_array_exc = None

    def _skip_if_different_combine(self, data):
        if data.fill_value == 0:
            # 当 fill_value 为 0 时，跳过测试，因为组合操作不能如预期一样执行
            pytest.skip("Incorrected expected from Series.combine and tested elsewhere")

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # 如果数据不同，跳过组合操作测试
        self._skip_if_different_combine(data)
        # 调用父类方法测试稀疏系列与标量的算术操作
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # 如果数据不同，跳过组合操作测试
        self._skip_if_different_combine(data)
        # 调用父类方法测试稀疏系列与数组的算术操作
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        if data.dtype.fill_value != 0:
            pass
        elif all_arithmetic_operators.strip("_") not in [
            "mul",
            "rmul",
            "floordiv",
            "rfloordiv",
            "truediv",
            "rtruediv",
            "pow",
            "mod",
            "rmod",
        ]:
            # 如果数据类型的 fill_value 不为 0，或者算术运算符不在指定列表中，标记为预期失败
            mark = pytest.mark.xfail(reason="result dtype.fill_value mismatch")
            request.applymarker(mark)
        # 调用父类方法测试稀疏数据帧与标量的算术操作
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def _compare_other(
        self, ser: pd.Series, data_for_compare: SparseArray, comparison_op, other
        # 用于比较其他数据的私有方法，接受一个 Pandas Series 和一个 SparseArray 进行比较
    ):
        # 将比较操作符赋给变量 op
        op = comparison_op

        # 使用 op 对 data_for_compare 和 other 进行比较操作，结果赋给 result
        result = op(data_for_compare, other)
        
        # 如果 other 是 pd.Series 类型，则断言 result 也是 pd.Series 类型
        if isinstance(other, pd.Series):
            assert isinstance(result, pd.Series)
            # 断言 result 的数据类型为 SparseDtype
            assert isinstance(result.dtype, SparseDtype)
        else:
            # 如果 other 不是 pd.Series 类型，则断言 result 是 SparseArray 类型
            assert isinstance(result, SparseArray)
        
        # 断言 result 的数据类型的子类型为 np.bool_
        assert result.dtype.subtype == np.bool_

        # 根据 other 的类型不同，构造期望的 SparseArray 对象 expected
        if isinstance(other, pd.Series):
            # 对于 pd.Series 类型的 other
            # 计算 fill_value，并使用 op 对 data_for_compare 和 other._values.fill_value 进行操作
            fill_value = op(data_for_compare.fill_value, other._values.fill_value)
            # 期望的 SparseArray 对象，使用 op 对 data_for_compare.to_dense() 和 np.asarray(other) 进行操作
            expected = SparseArray(
                op(data_for_compare.to_dense(), np.asarray(other)),
                fill_value=fill_value,
                dtype=np.bool_,
            )

        else:
            # 对于非 pd.Series 类型的 other
            # 计算 fill_value，并使用 op 对 data_for_compare.fill_value 和 other 进行操作
            fill_value = np.all(
                op(np.asarray(data_for_compare.fill_value), np.asarray(other))
            )
            # 期望的 SparseArray 对象，使用 op 对 data_for_compare.to_dense() 和 np.asarray(other) 进行操作
            expected = SparseArray(
                op(data_for_compare.to_dense(), np.asarray(other)),
                fill_value=fill_value,
                dtype=np.bool_,
            )
        
        # 如果 other 是 pd.Series 类型，则处理 expected 为 pd.Series 类型（忽略类型检查错误）
        if isinstance(other, pd.Series):
            # 错误：赋值时类型不兼容
            expected = pd.Series(expected)  # type: ignore[assignment]
        
        # 使用 pytest 框架的 assert_equal 方法断言 result 等于 expected
        tm.assert_equal(result, expected)

    def test_scalar(self, data_for_compare: SparseArray, comparison_op):
        # 创建 pd.Series 对象 ser，以 data_for_compare 为数据
        ser = pd.Series(data_for_compare)
        # 分别使用 _compare_other 方法进行比较测试
        self._compare_other(ser, data_for_compare, comparison_op, 0)
        self._compare_other(ser, data_for_compare, comparison_op, 1)
        self._compare_other(ser, data_for_compare, comparison_op, -1)
        self._compare_other(ser, data_for_compare, comparison_op, np.nan)

    def test_array(self, data_for_compare: SparseArray, comparison_op, request):
        # 如果 data_for_compare 的填充值为 0，且比较操作为 "eq", "ge", "le" 中的一种
        if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ in [
            "eq",
            "ge",
            "le",
        ]:
            # 标记为预期失败，理由是填充值不正确
            mark = pytest.mark.xfail(reason="Wrong fill_value")
            request.applymarker(mark)

        # 创建一个长度为 10 的 numpy 数组 arr，范围从 -4 到 5
        arr = np.linspace(-4, 5, 10)
        # 创建 pd.Series 对象 ser，以 data_for_compare 为数据
        ser = pd.Series(data_for_compare)
        # 使用 _compare_other 方法进行比较测试
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    def test_sparse_array(self, data_for_compare: SparseArray, comparison_op, request):
        # 如果 data_for_compare 的填充值为 0，且比较操作不是 "gt"
        if data_for_compare.dtype.fill_value == 0 and comparison_op.__name__ != "gt":
            # 标记为预期失败，理由是填充值不正确
            mark = pytest.mark.xfail(reason="Wrong fill_value")
            request.applymarker(mark)

        # 创建 pd.Series 对象 ser，以 data_for_compare 为数据
        ser = pd.Series(data_for_compare)
        # 创建 data_for_compare + 1 的数组 arr，并使用 _compare_other 方法进行比较测试
        arr = data_for_compare + 1
        self._compare_other(ser, data_for_compare, comparison_op, arr)
        # 创建 data_for_compare * 2 的数组 arr，并使用 _compare_other 方法进行比较测试
        arr = data_for_compare * 2
        self._compare_other(ser, data_for_compare, comparison_op, arr)

    @pytest.mark.xfail(reason="Different repr")
    def test_array_repr(self, data, size):
        # 调用父类的 test_array_repr 方法
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason="result does not match expected")
    @pytest.mark.parametrize("as_index", [True, False])
    # 定义一个测试方法 `test_groupby_extension_agg`，该方法继承并调用了其父类（超类）的同名方法，
    # 传入参数 `as_index` 和 `data_for_grouping` 给父类方法。
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        # 调用父类的同名方法 `test_groupby_extension_agg`，传入参数 `as_index` 和 `data_for_grouping`。
        super().test_groupby_extension_agg(as_index, data_for_grouping)
# 定义一个函数，用于测试给定类型的数组类型是否为稀疏数组类型
def test_array_type_with_arg(dtype):
    # 使用断言验证给定类型的数组的构造数组类型是否为 SparseArray
    assert dtype.construct_array_type() is SparseArray
```