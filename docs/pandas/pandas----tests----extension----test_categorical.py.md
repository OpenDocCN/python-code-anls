# `D:\src\scipysrc\pandas\pandas\tests\extension\test_categorical.py`

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

# Import necessary modules
import string

import numpy as np  # Import NumPy library
import pytest  # Import pytest for testing framework

from pandas._config import using_pyarrow_string_dtype  # Import specific configuration from pandas

import pandas as pd  # Import pandas library
from pandas import Categorical  # Import Categorical type from pandas
import pandas._testing as tm  # Import testing utilities from pandas
from pandas.api.types import CategoricalDtype  # Import CategoricalDtype from pandas.api.types
from pandas.tests.extension import base  # Import base module from pandas.tests.extension

# Function to generate random data
def make_data():
    while True:
        values = np.random.default_rng(2).choice(list(string.ascii_letters), size=100)
        # ensure we meet the requirements
        # 1. first two not null
        # 2. first and second are different
        if values[0] != values[1]:
            break
    return values

# Fixture to provide a default CategoricalDtype
@pytest.fixture
def dtype():
    return CategoricalDtype()

# Fixture to provide random categorical data
@pytest.fixture
def data():
    """Length-100 array for this type.

    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return Categorical(make_data())

# Fixture to provide categorical data with missing values
@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    return Categorical([np.nan, "A"])

# Fixture to provide sorted categorical data
@pytest.fixture
def data_for_sorting():
    return Categorical(["A", "B", "C"], categories=["C", "A", "B"], ordered=True)

# Fixture to provide sorted categorical data with missing values
@pytest.fixture
def data_missing_for_sorting():
    return Categorical(["A", None, "B"], categories=["B", "A"], ordered=True)

# Fixture to provide categorical data for grouping
@pytest.fixture
def data_for_grouping():
    return Categorical(["a", "a", None, None, "b", "b", "a", "c"])

# Test class inheriting from base.ExtensionTests
class TestCategorical(base.ExtensionTests):
    
    def test_contains(self, data, data_missing):
        # GH-37867
        # na value handling in Categorical.__contains__ is deprecated.
        # See base.BaseInterFaceTests.test_contains for more details.

        na_value = data.dtype.na_value  # Get the na_value from data's dtype
        # Ensure data without missing values
        data = data[~data.isna()]

        # Assert statements to validate containment
        assert data[0] in data
        assert data_missing[0] in data_missing

        # Check the presence of na_value
        assert na_value in data_missing
        assert na_value not in data

        # Categoricals can contain other nan-likes than na_value
        for na_value_obj in tm.NULL_OBJECTS:
            if na_value_obj is na_value:
                continue
            assert na_value_obj not in data
            # Additional assertion if not using pyarrow string dtype
            if not using_pyarrow_string_dtype():
                assert na_value_obj in data_missing
    def test_empty(self, dtype):
        # 使用给定的 dtype 构建数组类型的类对象
        cls = dtype.construct_array_type()
        # 调用 _empty 方法创建一个指定形状的空数组
        result = cls._empty((4,), dtype=dtype)

        # 断言结果是 cls 类的实例
        assert isinstance(result, cls)
        # 传入的 dtype 是未初始化的，因此不会与结果中的 dtype 匹配
        assert result.dtype == CategoricalDtype([])

    @pytest.mark.skip(reason="Backwards compatibility")
    def test_getitem_scalar(self, data):
        # CategoricalDtype.type 不是 "正确的"，因为它应该是元素（object）的父类。
        # 但是为了不破坏现有代码，暂时不改变它。
        super().test_getitem_scalar(data)

    def test_combine_add(self, data_repeated):
        # GH 20825
        # 在 combine 操作中，当对分类数据进行加法运算时，结果是字符串
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        
        # 使用 lambda 函数对 s1 和 s2 进行元素级别的加法操作
        result = s1.combine(s2, lambda x1, x2: x1 + x2)
        # 期望的结果是原始数据的每对元素相加得到的 Series
        expected = pd.Series(
            [a + b for (a, b) in zip(list(orig_data1), list(orig_data2))]
        )
        tm.assert_series_equal(result, expected)

        # 取 s1 的第一个元素
        val = s1.iloc[0]
        # 使用 lambda 函数将 s1 的每个元素与 val 相加
        result = s1.combine(val, lambda x1, x2: x1 + x2)
        # 期望的结果是将 val 加到 s1 的每个元素上得到的 Series
        expected = pd.Series([a + val for a in list(orig_data1)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        # 使用 map 方法对数据进行映射操作，na_action 参数指定了处理缺失值的方式
        result = data.map(lambda x: x, na_action=na_action)
        # 断言映射后的结果与原始数据相等
        tm.assert_extension_array_equal(result, data)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # frame & scalar
        op_name = all_arithmetic_operators
        if op_name == "__rmod__":
            request.applymarker(
                pytest.mark.xfail(
                    reason="rmod never called when string is first argument"
                )
            )
        # 调用父类的 test_arith_frame_with_scalar 方法进行测试
        super().test_arith_frame_with_scalar(data, op_name)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        op_name = all_arithmetic_operators
        if op_name == "__rmod__":
            request.applymarker(
                pytest.mark.xfail(
                    reason="rmod never called when string is first argument"
                )
            )
        # 调用父类的 test_arith_series_with_scalar 方法进行测试
        super().test_arith_series_with_scalar(data, op_name)

    def _compare_other(self, ser: pd.Series, data, op, other):
        op_name = f"__{op.__name__}__"
        if op_name not in ["__eq__", "__ne__"]:
            msg = "Unordered Categoricals can only compare equality or not"
            # 断言非等式比较会抛出 TypeError 异常，错误信息为 msg
            with pytest.raises(TypeError, match=msg):
                op(data, other)
        else:
            # 否则调用父类的 _compare_other 方法进行比较操作
            return super()._compare_other(ser, data, op, other)

    @pytest.mark.xfail(reason="Categorical overrides __repr__")
    @pytest.mark.parametrize("size", ["big", "small"])
    def test_array_repr(self, data, size):
        # 调用父类的 test_array_repr 方法进行测试，测试数组的字符串表示形式
        super().test_array_repr(data, size)

    @pytest.mark.xfail(reason="TBD")
    # 使用 pytest 的 parametrize 装饰器为下面的测试方法提供多组参数化输入，测试是否能正确处理索引设置。
    @pytest.mark.parametrize("as_index", [True, False])
    # 定义一个测试方法，测试对给定数据执行分组聚合操作时的行为。
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        # 调用父类的相同方法，以测试分组聚合操作在不同索引设置下的行为。
        super().test_groupby_extension_agg(as_index, data_for_grouping)
class Test2DCompat(base.NDArrayBacked2DTests):
    # 继承自 base.NDArrayBacked2DTests 的测试类 Test2DCompat

    def test_repr_2d(self, data):
        # 定义测试方法 test_repr_2d，接受参数 data

        # 对于 data 调整为 1 行多列的形状，获取其字符串表示，并检查是否包含一次 "\nCategories"
        res = repr(data.reshape(1, -1))
        assert res.count("\nCategories") == 1

        # 对于 data 调整为多行 1 列的形状，获取其字符串表示，并检查是否包含一次 "\nCategories"
        res = repr(data.reshape(-1, 1))
        assert res.count("\nCategories") == 1
```