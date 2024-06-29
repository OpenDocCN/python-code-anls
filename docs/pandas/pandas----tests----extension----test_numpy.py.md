# `D:\src\scipysrc\pandas\pandas\tests\extension\test_numpy.py`

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

Note: we do not bother with base.BaseIndexTests because NumpyExtensionArray
will never be held in an Index.
"""

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import NumpyEADtype

import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_object_dtype
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.tests.extension import base

orig_assert_attr_equal = tm.assert_attr_equal


def _assert_attr_equal(attr: str, left, right, obj: str = "Attributes"):
    """
    patch tm.assert_attr_equal so NumpyEADtype("object") is closed enough to
    np.dtype("object")
    """
    if attr == "dtype":
        lattr = getattr(left, "dtype", None)
        rattr = getattr(right, "dtype", None)
        if isinstance(lattr, NumpyEADtype) and not isinstance(rattr, NumpyEADtype):
            left = left.astype(lattr.numpy_dtype)
        elif isinstance(rattr, NumpyEADtype) and not isinstance(lattr, NumpyEADtype):
            right = right.astype(rattr.numpy_dtype)

    orig_assert_attr_equal(attr, left, right, obj)


@pytest.fixture(params=["float", "object"])
def dtype(request):
    return NumpyEADtype(np.dtype(request.param))


@pytest.fixture
def allow_in_pandas(monkeypatch):
    """
    A monkeypatch to tells pandas to let us in.

    By default, passing a NumpyExtensionArray to an index / series / frame
    constructor will unbox that NumpyExtensionArray to an ndarray, and treat
    it as a non-EA column. We don't want people using EAs without
    reason.

    The mechanism for this is a check against ABCNumpyExtensionArray
    in each constructor.

    But, for testing, we need to allow them in pandas. So we patch
    the _typ of NumpyExtensionArray, so that we evade the ABCNumpyExtensionArray
    check.
    """
    with monkeypatch.context() as m:
        m.setattr(NumpyExtensionArray, "_typ", "extension")
        m.setattr(tm.asserters, "assert_attr_equal", _assert_attr_equal)
        yield


@pytest.fixture
def data(allow_in_pandas, dtype):
    if dtype.numpy_dtype == "object":
        return pd.Series([(i,) for i in range(100)]).array
    return NumpyExtensionArray(np.arange(1, 101, dtype=dtype._dtype))


@pytest.fixture
def data_missing(allow_in_pandas, dtype):
    if dtype.numpy_dtype == "object":
        return NumpyExtensionArray(np.array([np.nan, (1,)], dtype=object))
    # 返回一个 NumpyExtensionArray 对象，其包含一个 NumPy 数组，数组中有两个元素：NaN（Not a Number，表示缺失值）和 1.0。
    return NumpyExtensionArray(np.array([np.nan, 1.0]))
@pytest.fixture
# 定义名为 na_cmp 的 pytest fixture
def na_cmp():
    # 定义一个比较函数 cmp，用于比较两个值是否都是 NaN
    def cmp(a, b):
        return np.isnan(a) and np.isnan(b)

    # 返回比较函数 cmp
    return cmp


@pytest.fixture
# 定义名为 data_for_sorting 的 pytest fixture
def data_for_sorting(allow_in_pandas, dtype):
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    if dtype.numpy_dtype == "object":
        # 对于 object 类型的数据，返回经过特定处理的 NumpyExtensionArray
        # 通过空元组开始，然后移除，以禁用 np.array 的形状推断
        return NumpyExtensionArray(np.array([(), (2,), (3,), (1,)], dtype=object)[1:])
    # 对于其他类型的数据，返回经过特定处理的 NumpyExtensionArray
    return NumpyExtensionArray(np.array([1, 2, 0]))


@pytest.fixture
# 定义名为 data_missing_for_sorting 的 pytest fixture
def data_missing_for_sorting(allow_in_pandas, dtype):
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    if dtype.numpy_dtype == "object":
        # 对于 object 类型的数据，返回经过特定处理的 NumpyExtensionArray
        return NumpyExtensionArray(np.array([(1,), np.nan, (0,)], dtype=object))
    # 对于其他类型的数据，返回经过特定处理的 NumpyExtensionArray
    return NumpyExtensionArray(np.array([1, np.nan, 0]))


@pytest.fixture
# 定义名为 data_for_grouping 的 pytest fixture
def data_for_grouping(allow_in_pandas, dtype):
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    if dtype.numpy_dtype == "object":
        # 对于 object 类型的数据，定义元组 a, b, c，并赋值
        a, b, c = (1,), (2,), (3,)
    else:
        # 对于其他类型的数据，创建 numpy 数组 a, b, c
        a, b, c = np.arange(3)
    # 返回经过特定处理的 NumpyExtensionArray
    return NumpyExtensionArray(
        np.array([b, b, np.nan, np.nan, a, a, b, c], dtype=dtype.numpy_dtype)
    )


@pytest.fixture
# 定义名为 data_for_twos 的 pytest fixture
def data_for_twos(dtype):
    # 如果 dtype 的种类是对象型，则跳过测试
    if dtype.kind == "O":
        pytest.skip(f"{dtype} is not a numeric dtype")
    # 创建一个长度为 100 的由 2 组成的数组 arr
    arr = np.ones(100) * 2
    # 返回经过特定处理的 NumpyExtensionArray
    return NumpyExtensionArray._from_sequence(arr, dtype=dtype)


@pytest.fixture
# 定义名为 skip_numpy_object 的 pytest fixture
def skip_numpy_object(dtype, request):
    """
    Tests for NumpyExtensionArray with nested data. Users typically won't create
    these objects via `pd.array`, but they can show up through `.array`
    on a Series with nested data. Many of the base tests fail, as they aren't
    appropriate for nested data.

    This fixture allows these tests to be skipped when used as a usefixtures
    marker to either an individual test or a test class.
    """
    # 如果 dtype 是对象型
    if dtype == "object":
        # 将测试标记为预期失败，原因是不适用于对象类型的数据
        mark = pytest.mark.xfail(reason="Fails for object dtype")
        # 应用 pytest marker
        request.applymarker(mark)


# 使用 skip_nested 标记的 pytest fixture
skip_nested = pytest.mark.usefixtures("skip_numpy_object")


class TestNumpyExtensionArray(base.ExtensionTests):
    @pytest.mark.skip(reason="We don't register our dtype")
    # 我们不希望注册我们的数据类型，因此跳过此测试
    def test_from_dtype(self, data):
        pass

    @skip_nested
    def test_series_constructor_scalar_with_index(self, data, dtype):
        # 抛出 ValueError：传入的值的长度为 1，但索引表明应该有 3 个值
        # 这个测试的目的是验证在带索引的情况下，用标量创建 Series 是否会引发错误
        super().test_series_constructor_scalar_with_index(data, dtype)
    # 检查数据的 dtype 是否为 "object" 类型
    def test_check_dtype(self, data, request, using_infer_string):
        if data.dtype.numpy_dtype == "object":
            # 如果是对象类型，标记为预期失败，给出失败原因
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"NumpyExtensionArray expectedly clashes with a "
                    f"NumPy name: {data.dtype.numpy_dtype}"
                )
            )
        # 调用父类的相同测试方法
        super().test_check_dtype(data)

    # 检查 dtype 是否不是 "object" 类型
    def test_is_not_object_type(self, dtype, request):
        if dtype.numpy_dtype == "object":
            # 如果是对象类型，检查是否是由于特定的 NumpyEADtype(object)引起的
            assert is_object_dtype(dtype)
        else:
            # 否则调用父类的相同测试方法
            super().test_is_not_object_type(dtype)

    # 跳过嵌套测试，测试获取单个元素的操作
    @skip_nested
    def test_getitem_scalar(self, data):
        # 断言错误
        super().test_getitem_scalar(data)

    # 跳过嵌套测试，测试移位操作
    @skip_nested
    def test_shift_fill_value(self, data):
        # np.array 的形状推断。移位实现失败。
        super().test_shift_fill_value(data)

    # 跳过嵌套测试，测试填充缺失值的限制条件（数据为DataFrame）
    @skip_nested
    def test_fillna_limit_frame(self, data_missing):
        # GH#58001
        # 此数组的“标量”不是标量。
        super().test_fillna_limit_frame(data_missing)

    # 跳过嵌套测试，测试填充缺失值的限制条件（数据为Series）
    @skip_nested
    def test_fillna_limit_series(self, data_missing):
        # GH#58001
        # 此数组的“标量”不是标量。
        super().test_fillna_limit_series(data_missing)

    # 跳过嵌套测试，测试DataFrame的fillna方法（拷贝方式）
    @skip_nested
    def test_fillna_copy_frame(self, data_missing):
        # 此数组的“标量”不是标量。
        super().test_fillna_copy_frame(data_missing)

    # 跳过嵌套测试，测试Series的fillna方法（拷贝方式）
    @skip_nested
    def test_fillna_copy_series(self, data_missing):
        # 此数组的“标量”不是标量。
        super().test_fillna_copy_series(data_missing)

    # 跳过嵌套测试，测试searchsorted方法
    @skip_nested
    def test_searchsorted(self, data_for_sorting, as_series):
        # TODO: NumpyExtensionArray.searchsorted 调用了 ndarray.searchsorted，
        #  在嵌套数据情况下，这不是我们想要的。我们需要调整类似于 libindex._bin_search 的方法。
        super().test_searchsorted(data_for_sorting, as_series)

    # 标记为预期失败的测试方法，因为 NumpyExtensionArray.diff 在某些 dtype 下可能失败
    @pytest.mark.xfail(reason="NumpyExtensionArray.diff may fail on dtype")
    def test_diff(self, data, periods):
        return super().test_diff(data, periods)

    # 测试插入操作
    def test_insert(self, data, request):
        if data.dtype.numpy_dtype == object:
            # 如果数据类型是 object 类型，标记为预期失败，给出失败原因
            mark = pytest.mark.xfail(reason="Dimension mismatch in np.concatenate")
            request.applymarker(mark)

        # 调用父类的相同测试方法
        super().test_insert(data)

    # 跳过嵌套测试，测试插入无效值的情况
    @skip_nested
    def test_insert_invalid(self, data, invalid_scalar):
        # NumpyExtensionArray[object] 可以容纳任何值，因此跳过测试
        super().test_insert_invalid(data, invalid_scalar)

    # 初始化异常变量
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None
    # 定义一个测试方法，用于测试 divmod 操作
    def test_divmod(self, data):
        # 初始化 divmod_exc 变量为 None
        divmod_exc = None
        # 如果 data 的数据类型是对象类型 ("O")，则设置 divmod_exc 为 TypeError
        if data.dtype.kind == "O":
            divmod_exc = TypeError
        # 将 divmod_exc 赋值给 self.divmod_exc 属性
        self.divmod_exc = divmod_exc
        # 调用父类的 test_divmod 方法进行测试
        super().test_divmod(data)

    # 定义一个测试方法，测试 Series 和数组之间的 divmod 操作
    def test_divmod_series_array(self, data):
        # 将 data 转换为 Pandas Series 对象
        ser = pd.Series(data)
        # 初始化 exc 变量为 None
        exc = None
        # 如果 data 的数据类型是对象类型 ("O")，则设置 exc 为 TypeError
        if data.dtype.kind == "O":
            exc = TypeError
            # 将 exc 赋值给 self.divmod_exc 属性
            self.divmod_exc = exc
        # 调用 _check_divmod_op 方法，测试 Series 和数组之间的 divmod 操作
        self._check_divmod_op(ser, divmod, data)

    # 定义一个测试方法，测试 Series 和标量之间的所有算术操作
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        # 获取算术操作的名称
        opname = all_arithmetic_operators
        # 初始化 series_scalar_exc 变量为 None
        series_scalar_exc = None
        # 如果 data 的 numpy 数据类型是对象类型 ("O")
        if data.dtype.numpy_dtype == object:
            # 对于乘法操作和右乘法操作，标记为预期失败
            if opname in ["__mul__", "__rmul__"]:
                mark = pytest.mark.xfail(
                    reason="the Series.combine step raises but not the Series method."
                )
                # 将 mark 标记添加到当前测试用例的节点
                request.node.add_marker(mark)
            # 设置 series_scalar_exc 为 TypeError
            series_scalar_exc = TypeError
        # 将 series_scalar_exc 赋值给 self.series_scalar_exc 属性
        self.series_scalar_exc = series_scalar_exc
        # 调用父类的 test_arith_series_with_scalar 方法进行测试
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    # 定义一个测试方法，测试 Series 和数组之间的所有算术操作
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # 获取算术操作的名称
        opname = all_arithmetic_operators
        # 初始化 series_array_exc 变量为 None
        series_array_exc = None
        # 如果 data 的 numpy 数据类型是对象类型 ("O")，且不是加法操作和右加法操作
        if data.dtype.numpy_dtype == object and opname not in ["__add__", "__radd__"]:
            # 设置 series_array_exc 为 TypeError
            series_array_exc = TypeError
        # 将 series_array_exc 赋值给 self.series_array_exc 属性
        self.series_array_exc = series_array_exc
        # 调用父类的 test_arith_series_with_array 方法进行测试
        super().test_arith_series_with_array(data, all_arithmetic_operators)

    # 定义一个测试方法，测试 DataFrame 和标量之间的所有算术操作
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # 获取算术操作的名称
        opname = all_arithmetic_operators
        # 初始化 frame_scalar_exc 变量为 None
        frame_scalar_exc = None
        # 如果 data 的 numpy 数据类型是对象类型 ("O")
        if data.dtype.numpy_dtype == object:
            # 对于乘法操作和右乘法操作，标记为预期失败
            if opname in ["__mul__", "__rmul__"]:
                mark = pytest.mark.xfail(
                    reason="the Series.combine step raises but not the Series method."
                )
                # 将 mark 标记添加到当前测试用例的节点
                request.node.add_marker(mark)
            # 设置 frame_scalar_exc 为 TypeError
            frame_scalar_exc = TypeError
        # 将 frame_scalar_exc 赋值给 self.frame_scalar_exc 属性
        self.frame_scalar_exc = frame_scalar_exc
        # 调用父类的 test_arith_frame_with_scalar 方法进行测试
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    # 检查是否支持对 Series 进行指定操作的归约
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # 如果 Series 的数据类型是对象类型 ("O")，则判断操作是否在支持的操作集合中
        if ser.dtype.kind == "O":
            return op_name in ["sum", "min", "max", "any", "all"]
        # 其他情况返回 True
        return True

    # 执行归约操作并检查结果
    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # 获取 Series 对象的指定操作的结果方法
        res_op = getattr(ser, op_name)
        # 获取与比较的数据类型，避免将 int 转换为 float，直接转换为实际的 numpy 类型
        cmp_dtype = ser.dtype.numpy_dtype  # type: ignore[union-attr]
        # 将 Series 转换为指定的数据类型
        alt = ser.astype(cmp_dtype)
        # 获取转换后 Series 对象的指定操作的结果方法
        exp_op = getattr(alt, op_name)
        # 如果操作名称为 "count"，则分别计算结果和期望值
        if op_name == "count":
            result = res_op()
            expected = exp_op()
        else:
            # 否则，根据 skipna 参数计算结果和期望值
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)
        # 使用 pytest 的 assert_almost_equal 方法断言结果与期望值的近似相等性
        tm.assert_almost_equal(result, expected)
    # 标记此测试为跳过状态，因为测试尚未编写
    @pytest.mark.skip("TODO: tests not written yet")
    # 参数化测试函数，测试跳过 NaN 处理时是否跳过 NaN 值
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna):
        pass

    # 标记此测试为跳过状态，并调用父类的 test_fillna_series 方法
    @skip_nested
    def test_fillna_series(self, data_missing):
        # 非标量的 "标量" 值。
        super().test_fillna_series(data_missing)

    # 标记此测试为跳过状态，并调用父类的 test_fillna_frame 方法
    @skip_nested
    def test_fillna_frame(self, data_missing):
        # 非标量的 "标量" 值。
        super().test_fillna_frame(data_missing)

    # 标记此测试为跳过状态，并调用父类的 test_setitem_invalid 方法
    @skip_nested
    def test_setitem_invalid(self, data, invalid_scalar):
        # 对象类型可以包含任何内容，因此不会引发异常
        super().test_setitem_invalid(data, invalid_scalar)

    # 标记此测试为跳过状态，并调用父类的 test_setitem_sequence_broadcasts 方法
    @skip_nested
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        # ValueError: 无法使用与值长度不同的类似列表索引器设置
        super().test_setitem_sequence_broadcasts(data, box_in_series)

    # 标记此测试为跳过状态，并参数化测试函数中的 setter 参数
    @skip_nested
    @pytest.mark.parametrize("setter", ["loc", None])
    def test_setitem_mask_broadcast(self, data, setter):
        # ValueError: 无法使用与值长度不同的类似列表索引器设置
        super().test_setitem_mask_broadcast(data, setter)

    # 标记此测试为跳过状态，并调用父类的 test_setitem_scalar_key_sequence_raise 方法
    def test_setitem_scalar_key_sequence_raise(self, data):
        # 未通过测试：未引发 <class 'ValueError'>
        super().test_setitem_scalar_key_sequence_raise(data)

    # TODO: 目前 NumpyExtensionArray 存在问题，因此暂时跳过 setitem 测试，并稍后修复（GH 31446）

    # 标记此测试为跳过状态，并参数化测试函数中的 mask 参数
    @skip_nested
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        # 调用父类的 test_setitem_mask 方法
        super().test_setitem_mask(data, mask, box_in_series)

    # 标记此测试为跳过状态，并参数化测试函数中的 idx 参数
    @skip_nested
    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    def test_setitem_integer_array(self, data, idx, box_in_series):
        # 调用父类的 test_setitem_integer_array 方法
        super().test_setitem_integer_array(data, idx, box_in_series)

    # 标记此测试为跳过状态，并调用父类的 test_setitem_slice 方法
    @skip_nested
    def test_setitem_slice(self, data, box_in_series):
        # 调用父类的 test_setitem_slice 方法
        super().test_setitem_slice(data, box_in_series)

    # 标记此测试为跳过状态，并调用父类的 test_setitem_loc_iloc_slice 方法
    @skip_nested
    def test_setitem_loc_iloc_slice(self, data):
        # 调用父类的 test_setitem_loc_iloc_slice 方法
        super().test_setitem_loc_iloc_slice(data)
    # 定义一个测试方法，用于测试扩展数据帧列的设置
    def test_setitem_with_expansion_dataframe_column(self, data, full_indexer):
        # 链接到 GitHub 上的问题页面，详细说明了问题的背景和相关信息
        # https://github.com/pandas-dev/pandas/issues/32395
        # 创建一个包含单列数据的数据帧，并赋给 df 和 expected
        df = expected = pd.DataFrame({"data": pd.Series(data)})
        # 创建一个空数据帧，索引与 df 相同
        result = pd.DataFrame(index=df.index)

        # 因为 result 的 dtype 是 object，尝试在 inplace 设置时成功，并保持 object dtype
        key = full_indexer(df)
        result.loc[key, "data"] = df["data"]

        # 基类方法中 expected 等于 df；NumpyExtensionArray 在这些测试中的行为有点奇怪，因为我们修改了这些测试的 _typ。
        if data.dtype.numpy_dtype != object:
            # 如果数据的 dtype 不是 object，且 key 不是全切片，则将 expected 重新赋为一个包含 numpy 数组的数据帧
            if not isinstance(key, slice) or key != slice(None):
                expected = pd.DataFrame({"data": data.to_numpy()})
        # 使用测试框架中的 assert_frame_equal 函数比较 result 和 expected 的内容，忽略列类型的检查
        tm.assert_frame_equal(result, expected, check_column_type=False)

    # 标记为预期失败的测试用例，原因是 NumpyEADtype 被展开了
    @pytest.mark.xfail(reason="NumpyEADtype is unpacked")
    def test_index_from_listlike_with_dtype(self, data):
        # 调用基类的 test_index_from_listlike_with_dtype 方法，并传入 data 参数
        super().test_index_from_listlike_with_dtype(data)

    # 标记为跳过的嵌套测试用例
    @skip_nested
    # 使用参数化装饰器，对 engine 参数进行参数化，分别测试 "c" 和 "python" 两种引擎
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, request):
        # 调用基类的 test_EA_types 方法，并传入 engine, data, request 参数
        super().test_EA_types(engine, data, request)
# 定义一个类 Test2DCompat，继承自 base.NDArrayBacked2DTests，用于测试二维数组的兼容性
class Test2DCompat(base.NDArrayBacked2DTests):
    pass
```