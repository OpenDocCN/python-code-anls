# `D:\src\scipysrc\pandas\pandas\tests\indexes\numeric\test_numeric.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 测试框架

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 导入 Pandas 中的 Index 和 Series 类
    Index,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestFloatNumericIndex:  # 定义测试类 TestFloatNumericIndex

    @pytest.fixture(params=[np.float64, np.float32])  # 参数化测试装置，测试浮点数类型
    def dtype(self, request):
        return request.param  # 返回参数中的数据类型

    @pytest.fixture  # 测试装置，生成混合索引
    def mixed_index(self, dtype):
        return Index([1.5, 2, 3, 4, 5], dtype=dtype)  # 返回混合类型索引对象

    @pytest.fixture  # 测试装置，生成浮点数索引
    def float_index(self, dtype):
        return Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=dtype)  # 返回浮点数类型索引对象

    @pytest.mark.parametrize(  # 参数化测试，测试不同的索引数据
        "index_data",
        [
            [1.5, 2, 3, 4, 5],
            [0.0, 2.5, 5.0, 7.5, 10.0],
            [5, 4, 3, 2, 1.5],
            [10.0, 7.5, 5.0, 2.5, 0.0],
        ],
        ids=["mixed", "float", "mixed_dec", "float_dec"],  # 参数标识
    )
    def test_repr_roundtrip(self, index_data, dtype):
        index = Index(index_data, dtype=dtype)  # 创建索引对象
        tm.assert_index_equal(eval(repr(index)), index, exact=True)  # 断言索引对象的序列化和反序列化结果一致

    def check_coerce(self, a, b, is_float_index=True):
        assert a.equals(b)  # 断言索引 a 和 b 相等
        tm.assert_index_equal(a, b, exact=False)  # 断言索引 a 和 b 大致相等
        if is_float_index:
            assert isinstance(b, Index)  # 如果是浮点数索引，断言 b 是 Index 类型
        else:
            assert type(b) is Index  # 否则断言 b 是 Index 类型

    def test_constructor_from_list_no_dtype(self):
        index = Index([1.5, 2.5, 3.5])  # 创建索引对象
        assert index.dtype == np.float64  # 断言索引对象的数据类型是 np.float64

    def test_constructor(self, dtype):
        index_cls = Index  # 索引类别设为 Index

        # explicit construction
        index = index_cls([1, 2, 3, 4, 5], dtype=dtype)  # 显式构造索引对象
        assert isinstance(index, index_cls)  # 断言 index 是 index_cls 类型
        assert index.dtype == dtype  # 断言 index 的数据类型是 dtype

        expected = np.array([1, 2, 3, 4, 5], dtype=dtype)  # 预期的 NumPy 数组
        tm.assert_numpy_array_equal(index.values, expected)  # 断言 index 的值和预期的数组值相等

        index = index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)  # 创建索引对象
        assert isinstance(index, index_cls)  # 断言 index 是 index_cls 类型
        assert index.dtype == dtype  # 断言 index 的数据类型是 dtype

        index = index_cls([1.0, 2, 3, 4, 5], dtype=dtype)  # 创建索引对象
        assert isinstance(index, index_cls)  # 断言 index 是 index_cls 类型
        assert index.dtype == dtype  # 断言 index 的数据类型是 dtype

        index = index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)  # 创建索引对象
        assert isinstance(index, index_cls)  # 断言 index 是 index_cls 类型
        assert index.dtype == dtype  # 断言 index 的数据类型是 dtype

        # nan handling
        result = index_cls([np.nan, np.nan], dtype=dtype)  # 处理 NaN 值的索引对象
        assert pd.isna(result.values).all()  # 断言 result 的所有值都是 NaN

        result = index_cls(np.array([np.nan]), dtype=dtype)  # 处理 NaN 值的索引对象
        assert pd.isna(result.values).all()  # 断言 result 的所有值都是 NaN

    def test_constructor_invalid(self):
        index_cls = Index  # 索引类别设为 Index
        cls_name = index_cls.__name__  # 获取索引类的名称
        # invalid
        msg = (
            rf"{cls_name}\(\.\.\.\) must be called with a collection of "
            r"some kind, 0\.0 was passed"
        )  # 异常消息内容
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常，并匹配指定的消息
            index_cls(0.0)  # 使用 0.0 创建索引对象
    # 定义测试函数，验证 Index 类的构造器函数的强制类型转换功能
    def test_constructor_coerce(self, mixed_index, float_index):
        # 检查混合索引，验证是否被正确强制转换为浮点数索引
        self.check_coerce(mixed_index, Index([1.5, 2, 3, 4, 5]))
        # 检查浮点数索引，验证是否被正确强制转换为特定的浮点数索引
        self.check_coerce(float_index, Index(np.arange(5) * 2.5))

        # 创建一个包含对象类型的索引，验证其数据类型是否为对象
        result = Index(np.array(np.arange(5) * 2.5, dtype=object))
        assert result.dtype == object  # 自 2.0 版本开始，以匹配 Series 的行为
        # 再次验证浮点数索引是否被正确强制转换为浮点数类型
        self.check_coerce(float_index, result.astype("float64"))

    # 定义测试函数，验证 Index 类的显式构造器函数的功能
    def test_constructor_explicit(self, mixed_index, float_index):
        # 这些数据不会自动转换
        self.check_coerce(
            float_index, Index((np.arange(5) * 2.5), dtype=object), is_float_index=False
        )
        self.check_coerce(
            mixed_index, Index([1.5, 2, 3, 4, 5], dtype=object), is_float_index=False
        )

    # 定义测试函数，验证类型强制转换失败的情况
    def test_type_coercion_fail(self, any_int_numpy_dtype):
        # 查看 GitHub issue-15832
        msg = "Trying to coerce float values to integers"
        # 使用 pytest 检测是否会引发值错误，并验证错误消息是否匹配预期消息
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3.5], dtype=any_int_numpy_dtype)

    # 定义测试函数，验证索引对象的相等性比较（数值类型）
    def test_equals_numeric(self):
        index_cls = Index

        # 创建浮点数索引对象
        idx = index_cls([1.0, 2.0])
        assert idx.equals(idx)  # 验证对象是否与自身相等
        assert idx.identical(idx)  # 验证对象是否与自身完全相同

        # 创建另一个与 idx 相同的浮点数索引对象
        idx2 = index_cls([1.0, 2.0])
        assert idx.equals(idx2)  # 验证两个对象的值是否相等

        # 创建包含 NaN 的浮点数索引对象
        idx = index_cls([1.0, np.nan])
        assert idx.equals(idx)  # 验证对象是否与自身相等
        assert idx.identical(idx)  # 验证对象是否与自身完全相同

        # 创建另一个包含 NaN 的浮点数索引对象
        idx2 = index_cls([1.0, np.nan])
        assert idx.equals(idx2)  # 验证两个对象的值是否相等

    # 使用 pytest 的参数化标记，定义测试函数，验证索引对象的相等性比较（其他索引类型）
    @pytest.mark.parametrize(
        "other",
        (
            Index([1, 2], dtype=np.int64),
            Index([1.0, 2.0], dtype=object),
            Index([1, 2], dtype=object),
        ),
    )
    def test_equals_numeric_other_index_type(self, other):
        # 创建浮点数索引对象
        idx = Index([1.0, 2.0])
        assert idx.equals(other)  # 验证当前对象与参数化的其他对象是否相等
        assert other.equals(idx)  # 验证参数化的其他对象与当前对象是否相等

    # 使用 pytest 的参数化标记，定义测试函数，验证不同类型的数值序列索引
    @pytest.mark.parametrize(
        "vals",
        [
            pd.date_range("2016-01-01", periods=3),
            pd.timedelta_range("1 Day", periods=3),
        ],
    )
    # 测试函数：检查包含日期时间或时间间隔值时，确保它们被正确包装
    def test_lookups_datetimelike_values(self, vals, dtype):
        # 创建 Series 对象，使用给定的值和索引
        ser = Series(vals, index=range(3, 6))
        # 将索引转换为指定的数据类型
        ser.index = ser.index.astype(dtype)

        # 期望的结果是 vals[1] 的值
        expected = vals[1]

        # 测试通过浮点数键访问元素
        result = ser[4.0]
        assert isinstance(result, type(expected)) and result == expected
        # 测试通过整数键访问元素
        result = ser[4]
        assert isinstance(result, type(expected)) and result == expected

        # 测试通过浮点数标签使用 .loc 访问元素
        result = ser.loc[4.0]
        assert isinstance(result, type(expected)) and result == expected
        # 测试通过整数标签使用 .loc 访问元素
        result = ser.loc[4]
        assert isinstance(result, type(expected)) and result == expected

        # 测试通过浮点数标签使用 .at 访问元素
        result = ser.at[4.0]
        assert isinstance(result, type(expected)) and result == expected
        # GH#31329 .at[4] 应该转换为 4.0，与 .loc 的行为一致
        # 测试通过整数标签使用 .at 访问元素
        result = ser.at[4]
        assert isinstance(result, type(expected)) and result == expected

        # 测试通过整数位置使用 .iloc 访问元素
        result = ser.iloc[1]
        assert isinstance(result, type(expected)) and result == expected

        # 测试通过整数位置使用 .iat 访问元素
        result = ser.iat[1]
        assert isinstance(result, type(expected)) and result == expected
# 定义一个测试类 TestNumericInt，用于测试 Index 类的数值整数特性
class TestNumericInt:
    
    # 定义 pytest 的 fixture，返回指定整数类型的 NumPy 数据类型
    @pytest.fixture
    def dtype(self, any_int_numpy_dtype):
        return np.dtype(any_int_numpy_dtype)
    
    # 定义 pytest 的 fixture，返回一个简单的 Index 对象，包含范围为 0 到 18 的偶数值
    @pytest.fixture
    def simple_index(self, dtype):
        return Index(range(0, 20, 2), dtype=dtype)
    
    # 测试函数，测试 Index 对象是否单调递增
    def test_is_monotonic(self):
        # 将 Index 类赋值给 index_cls 变量
        index_cls = Index
        
        # 创建一个包含整数列表的 Index 对象
        index = index_cls([1, 2, 3, 4])
        # 断言 Index 对象是否单调递增
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_decreasing is False
        
        # 创建一个包含逆序整数列表的 Index 对象
        index = index_cls([4, 3, 2, 1])
        # 断言 Index 对象是否单调递减
        assert index.is_monotonic_increasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is True
        
        # 创建一个仅包含一个整数的 Index 对象
        index = index_cls([1])
        # 断言 Index 对象是否单调递增和单调递减
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_increasing is True
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_increasing is True
        assert index._is_strictly_monotonic_decreasing is True

    # 测试函数，测试 Index 对象是否严格单调
    def test_is_strictly_monotonic(self):
        # 将 Index 类赋值给 index_cls 变量
        index_cls = Index
        
        # 创建一个包含非严格单调递增整数列表的 Index 对象
        index = index_cls([1, 1, 2, 3])
        # 断言 Index 对象是否单调递增但不是严格单调递增
        assert index.is_monotonic_increasing is True
        assert index._is_strictly_monotonic_increasing is False
        
        # 创建一个包含非严格单调递减整数列表的 Index 对象
        index = index_cls([3, 2, 1, 1])
        # 断言 Index 对象是否单调递减但不是严格单调递减
        assert index.is_monotonic_decreasing is True
        assert index._is_strictly_monotonic_decreasing is False
        
        # 创建一个包含重复值的 Index 对象
        index = index_cls([1, 1])
        # 断言 Index 对象是否单调递增和单调递减，但不是严格单调
        assert index.is_monotonic_increasing
        assert index.is_monotonic_decreasing
        assert not index._is_strictly_monotonic_increasing
        assert not index._is_strictly_monotonic_decreasing

    # 测试函数，测试简单的索引对象是否在逻辑上与其值兼容
    def test_logical_compat(self, simple_index):
        # 获取 simple_index 的别名为 idx 的对象
        idx = simple_index
        # 断言 Index 对象的所有元素是否与其值的所有元素逻辑相同
        assert idx.all() == idx.values.all()
        # 断言 Index 对象的任一元素是否与其值的任一元素逻辑相同
        assert idx.any() == idx.values.any()

    # 测试函数，测试 Index 对象的身份是否相同
    def test_identical(self, simple_index, dtype):
        # 获取 simple_index 的别名为 index 的对象
        index = simple_index
        
        # 创建一个具有相同值的新 Index 对象 idx
        idx = Index(index.copy())
        # 断言 idx 和 index 对象的身份是否相同
        assert idx.identical(index)
        
        # 创建一个与 idx 具有相同值但类型不同的 Index 对象
        same_values_different_type = Index(idx, dtype=object)
        # 断言 idx 和 same_values_different_type 的身份是否不同
        assert not idx.identical(same_values_different_type)
        
        # 将 index 对象转换为 object 类型，并重命名为 "foo"
        idx = index.astype(dtype=object)
        idx = idx.rename("foo")
        # 创建一个与 idx 具有相同值和类型的 Index 对象
        same_values = Index(idx, dtype=object)
        # 断言 idx 和 same_values 的身份是否相同
        assert same_values.identical(idx)
        
        # 断言 idx 和 index 的身份是否不同
        assert not idx.identical(index)
        # 断言具有相同值和类型的两个 Index 对象的身份是否相同
        assert Index(same_values, name="foo", dtype=object).identical(idx)
        
        # 断言将 index 和 index.astype(dtype=object) 转换为不同类型的 Index 对象的身份是否不同
        assert not index.astype(dtype=object).identical(index.astype(dtype=dtype))

    # 测试函数，测试不能或不应该转换的情况
    def test_cant_or_shouldnt_cast(self, dtype):
        # 定义错误消息字符串，用于匹配值错误的异常消息
        msg = r"invalid literal for int\(\) with base 10: 'foo'"
        
        # 无法转换的情况
        data = ["foo", "bar", "baz"]
        # 使用 pytest 的上下文管理器，断言是否引发 ValueError 异常，并检查异常消息是否匹配
        with pytest.raises(ValueError, match=msg):
            Index(data, dtype=dtype)
    # 定义一个测试方法，用于测试查看索引功能
    def test_view_index(self, simple_index):
        # 从参数中获取简单索引对象
        index = simple_index
        # 设置错误消息，用于匹配类型错误异常时的断言
        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        # 使用 pytest 断言，预期会抛出类型错误异常，并匹配指定消息
        with pytest.raises(TypeError, match=msg):
            # 尝试将索引对象视图为 Index 类型，应该会引发异常
            index.view(Index)
    
    # 定义一个测试方法，用于测试阻止类型转换功能
    def test_prevent_casting(self, simple_index):
        # 从参数中获取简单索引对象
        index = simple_index
        # 将索引对象转换为对象类型
        result = index.astype("O")
        # 使用断言验证转换后的数据类型为 NumPy 对象类型
        assert result.dtype == np.object_
class TestIntNumericIndex:
    @pytest.fixture(params=[np.int64, np.int32, np.int16, np.int8])
    def dtype(self, request):
        # 返回一个参数化的测试数据类型
        return request.param

    def test_constructor_from_list_no_dtype(self):
        # 创建一个包含整数的索引对象
        index = Index([1, 2, 3])
        # 断言索引对象的数据类型为 np.int64
        assert index.dtype == np.int64

    def test_constructor(self, dtype):
        index_cls = Index

        # scalar raise Exception
        # 准备错误消息字符串，用于测试 TypeError 异常
        msg = (
            rf"{index_cls.__name__}\(\.\.\.\) must be called with a collection of some "
            "kind, 5 was passed"
        )
        # 使用 pytest 检查是否引发了 TypeError 异常，并验证错误消息
        with pytest.raises(TypeError, match=msg):
            index_cls(5)

        # copy
        # 复制索引的值数组，并创建新的索引对象
        # 通过列表传递数据，在没有强制转换的情况下完成
        index = index_cls([-5, 0, 1, 2], dtype=dtype)
        arr = index.values.copy()
        new_index = index_cls(arr, copy=True)
        # 使用 pytest 的方法检查两个索引对象是否相等
        tm.assert_index_equal(new_index, index, exact=True)
        val = int(arr[0]) + 3000

        # this should not change index
        # 如果数据类型不是 np.int8，则执行以下代码块
        if dtype != np.int8:
            # NEP 50 won't allow assignment that would overflow
            # 修改数组第一个元素的值，但不会改变新索引的第一个元素值
            arr[0] = val
            assert new_index[0] != val

        if dtype == np.int64:
            # pass list, coerce fine
            # 创建索引对象，强制转换整数数据类型为 np.int64
            index = index_cls([-5, 0, 1, 2], dtype=dtype)
            expected = Index([-5, 0, 1, 2], dtype=dtype)
            # 使用 pytest 的方法检查两个索引对象是否相等
            tm.assert_index_equal(index, expected)

            # from iterable
            # 从可迭代对象创建索引对象
            index = index_cls(iter([-5, 0, 1, 2]), dtype=dtype)
            expected = index_cls([-5, 0, 1, 2], dtype=dtype)
            # 使用 pytest 的方法检查两个索引对象是否完全相等
            tm.assert_index_equal(index, expected, exact=True)

            # interpret list-like
            # 解释类列表的数据，创建索引对象并验证其相等性
            expected = index_cls([5, 0], dtype=dtype)
            for cls in [Index, index_cls]:
                for idx in [
                    cls([5, 0], dtype=dtype),
                    cls(np.array([5, 0]), dtype=dtype),
                    cls(Series([5, 0]), dtype=dtype),
                ]:
                    tm.assert_index_equal(idx, expected)

    def test_constructor_corner(self, dtype):
        index_cls = Index

        arr = np.array([1, 2, 3, 4], dtype=object)

        # 创建包含对象类型数组的索引对象
        index = index_cls(arr, dtype=dtype)
        # 断言索引对象的值数组数据类型与索引对象自身的数据类型相同
        assert index.values.dtype == index.dtype
        if dtype == np.int64:
            without_dtype = Index(arr)
            # as of 2.0 we do not infer a dtype when we get an object-dtype
            #  ndarray of numbers, matching Series behavior
            # 当我们得到一个包含数字的对象类型 ndarray 时，不会推断其数据类型
            assert without_dtype.dtype == object

            # 使用 pytest 的方法检查两个索引对象是否相等
            tm.assert_index_equal(index, without_dtype.astype(np.int64))

        # preventing casting
        # 防止强制转换
        arr = np.array([1, "2", 3, "4"], dtype=object)
        msg = "Trying to coerce object values to integers"
        # 使用 pytest 检查是否引发了 ValueError 异常，并验证错误消息
        with pytest.raises(ValueError, match=msg):
            index_cls(arr, dtype=dtype)

    def test_constructor_coercion_signed_to_unsigned(
        self,
        any_unsigned_int_numpy_dtype,
    ):
        # see gh-15832
        msg = "|".join(
            [
                "Trying to coerce negative values to unsigned integers",
                "The elements provided in the data cannot all be casted",
            ]
        )
        # 使用 pytest 检查是否会抛出 OverflowError 异常，并验证异常消息是否匹配预期
        with pytest.raises(OverflowError, match=msg):
            Index([-1], dtype=any_unsigned_int_numpy_dtype)

    def test_constructor_np_signed(self, any_signed_int_numpy_dtype):
        # GH#47475
        # 创建一个指定有符号整数类型的标量
        scalar = np.dtype(any_signed_int_numpy_dtype).type(1)
        # 使用该标量创建 Index 对象
        result = Index([scalar])
        # 创建预期的 Index 对象，确保类型和值正确
        expected = Index([1], dtype=any_signed_int_numpy_dtype)
        # 使用 assert_index_equal 检查结果是否与预期相等
        tm.assert_index_equal(result, expected, exact=True)

    def test_constructor_np_unsigned(self, any_unsigned_int_numpy_dtype):
        # GH#47475
        # 创建一个指定无符号整数类型的标量
        scalar = np.dtype(any_unsigned_int_numpy_dtype).type(1)
        # 使用该标量创建 Index 对象
        result = Index([scalar])
        # 创建预期的 Index 对象，确保类型和值正确
        expected = Index([1], dtype=any_unsigned_int_numpy_dtype)
        # 使用 assert_index_equal 检查结果是否与预期相等
        tm.assert_index_equal(result, expected, exact=True)

    def test_coerce_list(self):
        # coerce things
        # 创建一个包含整数的 Index 对象
        arr = Index([1, 2, 3, 4])
        # 断言 arr 是 Index 类型的实例
        assert isinstance(arr, Index)

        # but not if explicit dtype passed
        # 如果传入了显式的 dtype 参数，则不进行类型强制转换
        arr = Index([1, 2, 3, 4], dtype=object)
        # 使用 type 函数验证 arr 的类型确实是 Index
        assert type(arr) is Index
class TestFloat16Index:
    # 定义测试类 TestFloat16Index，用于测试 float16 索引的行为
    # GH 49535 是关于该问题的 GitHub issue 编号

    def test_constructor(self):
        # 测试构造函数

        index_cls = Index  # 使用 Index 类进行索引操作
        dtype = np.float16  # 指定数据类型为 np.float16

        msg = "float16 indexes are not supported"  # 错误消息

        # 显式构造方式测试
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1, 2, 3, 4, 5], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1.0, 2, 3, 4, 5], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1.0, 2, 3, 4, 5], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)

        # 处理 NaN 值的测试
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([np.nan, np.nan], dtype=dtype)

        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([np.nan]), dtype=dtype)


@pytest.mark.parametrize(
    "box",
    [list, lambda x: np.array(x, dtype=object), lambda x: Index(x, dtype=object)],
)
def test_uint_index_does_not_convert_to_float64(box):
    # 测试 uint 索引不会转换为 float64 类型

    # 创建 Series 对象，包含特定索引
    series = Series(
        [0, 1, 2, 3, 4, 5],
        index=[
            7606741985629028552,
            17876870360202815256,
            17876870360202815256,
            13106359306506049338,
            8991270399732411471,
            8991270399732411472,
        ],
    )

    # 对特定 box 进行索引操作
    result = series.loc[box([7606741985629028552, 17876870360202815256])]

    # 期望的索引结果
    expected = Index(
        [7606741985629028552, 17876870360202815256, 17876870360202815256],
        dtype="uint64",
    )
    tm.assert_index_equal(result.index, expected)  # 断言索引结果符合预期

    tm.assert_equal(result, series.iloc[:3])  # 断言结果与原始 series 的切片操作一致


def test_float64_index_equals():
    # 测试 float64 索引的 equals 方法

    float_index = Index([1.0, 2, 3])  # 创建 float64 类型的索引
    string_index = Index(["1", "2", "3"])  # 创建字符串类型的索引

    result = float_index.equals(string_index)  # 比较两个索引是否相等
    assert result is False  # 断言结果为 False

    result = string_index.equals(float_index)  # 再次比较，保证对称性
    assert result is False  # 断言结果为 False


def test_map_dtype_inference_unsigned_to_signed():
    # 测试 map 方法对无符号整数到有符号整数的类型推断

    # 创建包含无符号整数的索引对象
    idx = Index([1, 2, 3], dtype=np.uint64)

    # 对索引进行映射操作
    result = idx.map(lambda x: -x)

    # 期望的映射结果
    expected = Index([-1, -2, -3], dtype=np.int64)

    tm.assert_index_equal(result, expected)  # 断言映射结果符合预期


def test_map_dtype_inference_overflows():
    # 测试 map 方法对溢出情况的类型推断

    # 创建包含 int8 类型数据的索引对象
    idx = Index(np.array([1, 2, 3], dtype=np.int8))

    # 对索引进行映射操作
    result = idx.map(lambda x: x * 1000)

    # 期望的映射结果
    expected = Index([1000, 2000, 3000], dtype=np.int64)

    # TODO: 我们可能需要尝试推断到 int16 类型
    # 使用测试工具模块中的 assert_index_equal 函数来比较 result 和 expected 两个对象
    tm.assert_index_equal(result, expected)
def test_view_to_datetimelike():
    # GH#55710
    # 创建一个整数索引对象
    idx = Index([1, 2, 3])
    # 将索引对象转换为时间单位为秒的日期时间索引
    res = idx.view("m8[s]")
    # 从整数索引的视图创建一个TimedeltaIndex对象
    expected = pd.TimedeltaIndex(idx.values.view("m8[s]"))
    # 断言两个索引对象是否相等
    tm.assert_index_equal(res, expected)

    # 将索引对象转换为时间单位为天的日期时间视图
    res2 = idx.view("m8[D]")
    # 从整数索引的视图创建一个日期时间视图
    expected2 = idx.values.view("m8[D]")
    # 断言两个NumPy数组是否相等
    tm.assert_numpy_array_equal(res2, expected2)

    # 将索引对象转换为时间单位为小时的日期时间视图
    res3 = idx.view("M8[h]")
    # 从整数索引的视图创建一个日期时间视图
    expected3 = idx.values.view("M8[h]")
    # 断言两个NumPy数组是否相等
    tm.assert_numpy_array_equal(res3, expected3)
```