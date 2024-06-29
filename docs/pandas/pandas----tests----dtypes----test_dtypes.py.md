# `D:\src\scipysrc\pandas\pandas\tests\dtypes\test_dtypes.py`

```
import re  # 导入正则表达式模块
import weakref  # 导入弱引用模块，用于创建对象的弱引用

import numpy as np  # 导入NumPy库并重命名为np
import pytest  # 导入pytest测试框架

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 从pandas库中导入NpyDatetimeUnit数据类型

from pandas.core.dtypes.base import _registry as registry  # 从pandas核心模块中导入_registry并重命名为registry
from pandas.core.dtypes.common import (  # 从pandas核心模块中导入多个数据类型判断函数
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_dtype_equal,
    is_interval_dtype,
    is_period_dtype,
    is_string_dtype,
)

from pandas.core.dtypes.dtypes import (  # 从pandas核心模块中导入不同的数据类型类别
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)

import pandas as pd  # 导入pandas库并重命名为pd
from pandas import (  # 从pandas库中导入多个类和函数
    Categorical,
    CategoricalIndex,
    DatetimeIndex,
    IntervalIndex,
    Series,
    SparseDtype,
    date_range,
)
import pandas._testing as tm  # 导入pandas测试工具模块并重命名为tm
from pandas.core.arrays.sparse import SparseArray  # 从pandas核心模块中导入SparseArray类


class Base:
    def test_hash(self, dtype):
        hash(dtype)  # 计算给定对象的哈希值

    def test_equality_invalid(self, dtype):
        assert not dtype == "foo"  # 断言dtype对象不等于字符串"foo"
        assert not is_dtype_equal(dtype, np.int64)  # 断言dtype对象不等于np.int64类型

    def test_numpy_informed(self, dtype):
        # npdev 2020-02-02 changed from "data type not understood" to
        #  "Cannot interpret 'foo' as a data type"
        msg = "|".join(  # 将多个字符串用竖线连接起来
            ["data type not understood", "Cannot interpret '.*' as a data type"]
        )
        with pytest.raises(TypeError, match=msg):  # 使用pytest断言抛出TypeError，并匹配指定的异常信息msg
            np.dtype(dtype)  # 尝试创建NumPy dtype对象

        assert not dtype == np.str_  # 断言dtype对象不等于np.str_
        assert not np.str_ == dtype  # 断言np.str_对象不等于dtype

    def test_pickle(self, dtype):
        # make sure our cache is NOT pickled

        # clear the cache
        type(dtype).reset_cache()  # 调用dtype类型的reset_cache方法，清空缓存
        assert not len(dtype._cache_dtypes)  # 断言dtype对象的_cache_dtypes属性长度为0

        # force back to the cache
        result = tm.round_trip_pickle(dtype)  # 使用tm工具模块对dtype对象进行序列化和反序列化
        if not isinstance(dtype, PeriodDtype):
            # Because PeriodDtype has a cython class as a base class,
            #  it has different pickle semantics, and its cache is re-populated
            #  on un-pickling.
            assert not len(dtype._cache_dtypes)  # 断言dtype对象的_cache_dtypes属性长度为0
        assert result == dtype  # 断言序列化和反序列化的结果与原始dtype对象相等


class TestCategoricalDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestCategoricalDtype
        """
        return CategoricalDtype()  # 返回一个CategoricalDtype对象作为测试用例的dtype

    def test_hash_vs_equality(self, dtype):
        dtype2 = CategoricalDtype()  # 创建另一个CategoricalDtype对象dtype2
        assert dtype == dtype2  # 断言两个dtype对象相等
        assert dtype2 == dtype  # 断言两个dtype对象相等
        assert hash(dtype) == hash(dtype2)  # 断言两个dtype对象的哈希值相等

    def test_equality(self, dtype):
        assert dtype == "category"  # 断言dtype对象等于字符串"category"
        assert is_dtype_equal(dtype, "category")  # 断言dtype对象与字符串"category"类型相等
        assert "category" == dtype  # 断言字符串"category"等于dtype对象
        assert is_dtype_equal("category", dtype)  # 断言字符串"category"类型与dtype对象相等

        assert dtype == CategoricalDtype()  # 断言dtype对象等于新创建的CategoricalDtype对象
        assert is_dtype_equal(dtype, CategoricalDtype())  # 断言dtype对象与新创建的CategoricalDtype对象类型相等
        assert CategoricalDtype() == dtype  # 断言新创建的CategoricalDtype对象等于dtype对象
        assert is_dtype_equal(CategoricalDtype(), dtype)  # 断言新创建的CategoricalDtype对象类型与dtype对象相等

        assert dtype != "foo"  # 断言dtype对象不等于字符串"foo"
        assert not is_dtype_equal(dtype, "foo")  # 断言dtype对象与字符串"foo"类型不相等
        assert "foo" != dtype  # 断言字符串"foo"不等于dtype对象
        assert not is_dtype_equal("foo", dtype)  # 断言字符串"foo"类型与dtype对象不相等
    # 测试从字符串构造 CategoricalDtype 对象的方法
    def test_construction_from_string(self, dtype):
        # 使用字符串构造 CategoricalDtype 对象，检查结果是否符合预期的 dtype
        result = CategoricalDtype.construct_from_string("category")
        assert is_dtype_equal(dtype, result)
        # 测试用非法字符串构造 CategoricalDtype 对象时是否会抛出 TypeError 异常
        msg = "Cannot construct a 'CategoricalDtype' from 'foo'"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype.construct_from_string("foo")

    # 测试构造函数传入非法参数时是否会抛出 TypeError 异常
    def test_constructor_invalid(self):
        msg = "Parameter 'categories' must be list-like"
        with pytest.raises(TypeError, match=msg):
            CategoricalDtype("category")

    # 创建两个不同的 CategoricalDtype 对象
    dtype1 = CategoricalDtype(["a", "b"], ordered=True)
    dtype2 = CategoricalDtype(["x", "y"], ordered=False)
    # 创建一个包含 Categorical 对象的变量 c
    c = Categorical([0, 1], dtype=dtype1)

    # 使用参数化测试标记来测试 _from_values_or_dtype 方法
    @pytest.mark.parametrize(
        "values, categories, ordered, dtype, expected",
        [
            [None, None, None, None, CategoricalDtype()],  # 测试默认情况
            [None, ["a", "b"], True, None, dtype1],       # 测试指定 categories 和 ordered 的情况
            [c, None, None, dtype2, dtype2],              # 测试使用 Categorical 对象作为 values 的情况
            [c, ["x", "y"], False, None, dtype2],         # 测试指定 categories 的情况
        ],
    )
    def test_from_values_or_dtype(self, values, categories, ordered, dtype, expected):
        # 调用 _from_values_or_dtype 方法，检查返回结果是否符合预期
        result = CategoricalDtype._from_values_or_dtype(
            values, categories, ordered, dtype
        )
        assert result == expected

    # 使用参数化测试标记来测试 _from_values_or_dtype 方法抛出异常的情况
    @pytest.mark.parametrize(
        "values, categories, ordered, dtype",
        [
            [None, ["a", "b"], True, dtype2],          # 测试同时指定 categories 和 dtype 的情况
            [None, ["a", "b"], None, dtype2],          # 测试指定 categories 的情况
            [None, None, True, dtype2],                # 测试指定 ordered 的情况
        ],
    )
    def test_from_values_or_dtype_raises(self, values, categories, ordered, dtype):
        # 测试当同时指定 categories 或 ordered 与 dtype 时是否会抛出 ValueError 异常
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)

    # 测试当指定非法 dtype 时是否会抛出 ValueError 异常
    def test_from_values_or_dtype_invalid_dtype(self):
        msg = "Cannot not construct CategoricalDtype from <class 'object'>"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype._from_values_or_dtype(None, None, None, object)

    # 测试 is_dtype 方法的多种情况
    def test_is_dtype(self, dtype):
        assert CategoricalDtype.is_dtype(dtype)                   # 测试传入 dtype 是否为 CategoricalDtype 类型
        assert CategoricalDtype.is_dtype("category")             # 测试传入字符串 "category" 是否为 CategoricalDtype 类型
        assert CategoricalDtype.is_dtype(CategoricalDtype())     # 测试传入 CategoricalDtype 对象是否为 CategoricalDtype 类型
        assert not CategoricalDtype.is_dtype("foo")              # 测试传入非 CategoricalDtype 相关的字符串是否为 CategoricalDtype 类型
        assert not CategoricalDtype.is_dtype(np.float64)         # 测试传入非 CategoricalDtype 相关的类型是否为 CategoricalDtype 类型

    # 测试基本的 Categorical 对象方法
    def test_basic(self, dtype):
        msg = "is_categorical_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_categorical_dtype(dtype)                   # 检查是否为 CategoricalDtype 类型

            factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"])

            s = Series(factor, name="A")

            # dtypes
            assert is_categorical_dtype(s.dtype)                 # 检查 Series 对象的 dtype 是否为 CategoricalDtype
            assert is_categorical_dtype(s)                       # 检查 Series 对象是否为 Categorical 类型
            assert not is_categorical_dtype(np.dtype("float64")) # 检查非 CategoricalDtype 类型是否返回 False
    # 定义测试方法，验证 CategoricalDtype 对象对元组类别的处理是否正确
    def test_tuple_categories(self):
        # 创建包含元组的类别列表
        categories = [(1, "a"), (2, "b"), (3, "c")]
        # 调用 CategoricalDtype 构造函数，传入类别列表
        result = CategoricalDtype(categories)
        # 断言结果对象的类别属性与预期的类别列表相等
        assert all(result.categories == categories)

    # 使用 pytest 的参数化装饰器，定义多组输入数据和预期结果，对 test_is_boolean 方法进行测试
    @pytest.mark.parametrize(
        "categories, expected",
        [
            ([True, False], True),        # 类别包含布尔值，预期结果为 True
            ([True, False, None], True),  # 类别包含布尔值和 None，预期结果为 True
            ([True, False, "a", "b'"], False),  # 类别包含布尔值和字符串，预期结果为 False
            ([0, 1], False),              # 类别包含整数，预期结果为 False
        ],
    )
    # 定义测试方法，验证 Categorical 对象在不同类别情况下的布尔类型判断功能
    def test_is_boolean(self, categories, expected):
        # 创建 Categorical 对象
        cat = Categorical(categories)
        # 断言类别的 dtype 属性的 _is_boolean 属性与预期结果相等
        assert cat.dtype._is_boolean is expected
        # 断言 is_bool_dtype 函数对 Categorical 对象的结果与预期结果相等
        assert is_bool_dtype(cat) is expected
        # 断言 is_bool_dtype 函数对 Categorical 对象的 dtype 属性的结果与预期结果相等
        assert is_bool_dtype(cat.dtype) is expected

    # 定义测试方法，验证 Categorical 对象在处理特定数据类型的情况下的行为
    def test_dtype_specific_categorical_dtype(self):
        # 指定预期的数据类型字符串
        expected = "datetime64[ns]"
        # 创建空的 DatetimeIndex 对象，指定数据类型为预期值
        dti = DatetimeIndex([], dtype=expected)
        # 创建 Categorical 对象，并获取其类别的数据类型字符串表示
        result = str(Categorical(dti).categories.dtype)
        # 断言结果与预期值相等
        assert result == expected

    # 定义测试方法，验证 CategoricalDtype 对象在不是字符串情况下的行为
    def test_not_string(self):
        # 尽管 CategoricalDtype 具有对象类型，但它不能是字符串类型
        assert not is_string_dtype(CategoricalDtype())

    # 定义测试方法，验证 CategoricalDtype 对象在处理范围类别时的字符串表示
    def test_repr_range_categories(self):
        # 创建包含范围数据的索引对象
        rng = pd.Index(range(3))
        # 创建 CategoricalDtype 对象，指定范围数据和无序性
        dtype = CategoricalDtype(categories=rng, ordered=False)
        # 获取对象的字符串表示
        result = repr(dtype)
        # 设置预期的字符串表示，包括类别范围和无序性信息
        expected = (
            "CategoricalDtype(categories=range(0, 3), ordered=False, "
            "categories_dtype=int64)"
        )
        # 断言结果与预期值相等
        assert result == expected

    # 定义测试方法，验证 CategoricalDtype 对象在更新数据类型时的行为
    def test_update_dtype(self):
        # GH 27338
        # 调用 CategoricalDtype 的 update_dtype 方法，传入新的 Categorical 对象
        result = CategoricalDtype(["a"]).update_dtype(Categorical(["b"], ordered=True))
        # 设置预期的 CategoricalDtype 对象
        expected = CategoricalDtype(["b"], ordered=True)
        # 断言结果与预期值相等
        assert result == expected

    # 定义测试方法，验证 Categorical 对象的字符串表示
    def test_repr(self):
        # 创建包含整数的索引对象
        cat = Categorical(pd.Index([1, 2, 3], dtype="int32"))
        # 获取 Categorical 对象的 dtype 属性的字符串表示
        result = cat.dtype.__repr__()
        # 设置预期的字符串表示，包括类别和无序性信息
        expected = (
            "CategoricalDtype(categories=[1, 2, 3], ordered=False, "
            "categories_dtype=int32)"
        )
        # 断言结果与预期值相等
        assert result == expected
class TestDatetimeTZDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestDatetimeTZDtype
        """
        # 返回一个 DatetimeTZDtype 实例，时间单位为纳秒，时区为 US/Eastern
        return DatetimeTZDtype("ns", "US/Eastern")

    def test_alias_to_unit_raises(self):
        # 测试当传入不正确的别名时是否会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Passing a dtype alias"):
            DatetimeTZDtype("datetime64[ns, US/Central]")

    def test_alias_to_unit_bad_alias_raises(self):
        # 测试当传入不正确的类型别名时是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match=""):
            DatetimeTZDtype("this is a bad string")

        # 测试当传入不正确的时区别名时是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match=""):
            DatetimeTZDtype("datetime64[ns, US/NotATZ]")

    def test_hash_vs_equality(self, dtype):
        # 确保 DatetimeTZDtype 类实现了哈希和相等性语义
        dtype2 = DatetimeTZDtype("ns", "US/Eastern")
        dtype3 = DatetimeTZDtype(dtype2)
        assert dtype == dtype2
        assert dtype2 == dtype
        assert dtype3 == dtype
        assert hash(dtype) == hash(dtype2)
        assert hash(dtype) == hash(dtype3)

        # 测试不同时区的 DatetimeTZDtype 实例的相等性和哈希值
        dtype4 = DatetimeTZDtype("ns", "US/Central")
        assert dtype2 != dtype4
        assert hash(dtype2) != hash(dtype4)

    def test_construction_non_nanosecond(self):
        # 测试以非纳秒为单位构造 DatetimeTZDtype 实例的情况
        res = DatetimeTZDtype("ms", "US/Eastern")
        assert res.unit == "ms"
        assert res._creso == NpyDatetimeUnit.NPY_FR_ms.value
        assert res.str == "|M8[ms]"
        assert str(res) == "datetime64[ms, US/Eastern]"
        assert res.base == np.dtype("M8[ms]")

    def test_day_not_supported(self):
        # 测试当传入天单位时是否会抛出 ValueError 异常
        msg = "DatetimeTZDtype only supports s, ms, us, ns units"
        with pytest.raises(ValueError, match=msg):
            DatetimeTZDtype("D", "US/Eastern")

    def test_subclass(self):
        # 测试从字符串构造 DatetimeTZDtype 实例的子类关系
        a = DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]")
        b = DatetimeTZDtype.construct_from_string("datetime64[ns, CET]")

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))

    def test_compat(self, dtype):
        # 测试与兼容性相关的函数和方法，同时验证警告信息是否产生
        msg = "is_datetime64tz_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)
            assert is_datetime64tz_dtype("datetime64[ns, US/Eastern]")
        assert is_datetime64_any_dtype(dtype)
        assert is_datetime64_any_dtype("datetime64[ns, US/Eastern]")
        assert is_datetime64_ns_dtype(dtype)
        assert is_datetime64_ns_dtype("datetime64[ns, US/Eastern]")
        assert not is_datetime64_dtype(dtype)
        assert not is_datetime64_dtype("datetime64[ns, US/Eastern]")

    def test_construction_from_string(self, dtype):
        # 测试从字符串构造 DatetimeTZDtype 实例的正确性
        result = DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]")
        assert is_dtype_equal(dtype, result)
    @pytest.mark.parametrize(
        "string",
        [
            "foo",
            "datetime64[ns, notatz]",
            # non-nano unit
            "datetime64[ps, UTC]",
            # dateutil str that returns None from gettz
            "datetime64[ns, dateutil/invalid]",
        ],
    )
    def test_construct_from_string_invalid_raises(self, string):
        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        # 使用 pytest 来参数化测试，测试不合法的输入是否会引发 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=re.escape(msg)):
            DatetimeTZDtype.construct_from_string(string)

    def test_construct_from_string_wrong_type_raises(self):
        msg = "'construct_from_string' expects a string, got <class 'list'>"
        # 测试当输入类型错误时是否会引发 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            DatetimeTZDtype.construct_from_string(["datetime64[ns, notatz]"])

    def test_is_dtype(self, dtype):
        # 断言 DatetimeTZDtype.is_dtype 方法的预期行为
        assert not DatetimeTZDtype.is_dtype(None)
        assert DatetimeTZDtype.is_dtype(dtype)
        assert DatetimeTZDtype.is_dtype("datetime64[ns, US/Eastern]")
        assert DatetimeTZDtype.is_dtype("M8[ns, US/Eastern]")
        assert not DatetimeTZDtype.is_dtype("foo")
        assert DatetimeTZDtype.is_dtype(DatetimeTZDtype("ns", "US/Pacific"))
        assert not DatetimeTZDtype.is_dtype(np.float64)

    def test_equality(self, dtype):
        # 断言 is_dtype_equal 函数对不同的日期时间类型的正确比较结果
        assert is_dtype_equal(dtype, "datetime64[ns, US/Eastern]")
        assert is_dtype_equal(dtype, "M8[ns, US/Eastern]")
        assert is_dtype_equal(dtype, DatetimeTZDtype("ns", "US/Eastern"))
        assert not is_dtype_equal(dtype, "foo")
        assert not is_dtype_equal(dtype, DatetimeTZDtype("ns", "CET"))
        assert not is_dtype_equal(
            DatetimeTZDtype("ns", "US/Eastern"), DatetimeTZDtype("ns", "US/Pacific")
        )

        # numpy 兼容性测试
        assert is_dtype_equal(np.dtype("M8[ns]"), "datetime64[ns]")

        assert dtype == "M8[ns, US/Eastern]"

    def test_basic(self, dtype):
        msg = "is_datetime64tz_dtype is deprecated"
        # 测试是否会产生 DeprecationWarning 警告消息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(dtype)

        dr = date_range("20130101", periods=3, tz="US/Eastern")
        s = Series(dr, name="A")

        # dtypes
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_datetime64tz_dtype(s.dtype)
            assert is_datetime64tz_dtype(s)
            assert not is_datetime64tz_dtype(np.dtype("float64"))
            assert not is_datetime64tz_dtype(1.0)

    def test_dst(self):
        dr1 = date_range("2013-01-01", periods=3, tz="US/Eastern")
        s1 = Series(dr1, name="A")
        assert isinstance(s1.dtype, DatetimeTZDtype)

        dr2 = date_range("2013-08-01", periods=3, tz="US/Eastern")
        s2 = Series(dr2, name="A")
        assert isinstance(s2.dtype, DatetimeTZDtype)
        assert s1.dtype == s2.dtype

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern"])
    @pytest.mark.parametrize("constructor", ["M8", "datetime64"])
    # 使用 pytest 的参数化功能来测试不同的时区和日期时间构造器
    # 定义一个测试方法，用于测试时区和构造函数的解析功能
    def test_parser(self, tz, constructor):
        # pr #11245
        # 根据构造函数和时区生成描述时区的字符串
        dtz_str = f"{constructor}[ns, {tz}]"
        # 使用DatetimeTZDtype的方法根据字符串构造DatetimeTZDtype对象
        result = DatetimeTZDtype.construct_from_string(dtz_str)
        # 生成期望的DatetimeTZDtype对象
        expected = DatetimeTZDtype("ns", tz)
        # 断言结果与期望值相等
        assert result == expected

    # 定义一个测试方法，用于测试空值情况
    def test_empty(self):
        # 使用pytest的上下文管理器来验证抛出TypeError异常，并匹配给定的错误信息
        with pytest.raises(TypeError, match="A 'tz' is required."):
            # 调用DatetimeTZDtype的构造函数，期望它抛出TypeError异常
            DatetimeTZDtype()

    # 定义一个测试方法，用于测试时区标准化功能
    def test_tz_standardize(self):
        # GH 24713
        # 导入pytest，如果导入失败则跳过这个测试
        pytz = pytest.importorskip("pytz")
        # 获取美国东部时区对象
        tz = pytz.timezone("US/Eastern")
        # 生成一个包含时区的日期范围对象
        dr = date_range("2013-01-01", periods=3, tz=tz)
        # 使用DatetimeTZDtype构造函数创建一个DatetimeTZDtype对象，指定时区
        dtype = DatetimeTZDtype("ns", dr.tz)
        # 断言对象的时区与预期的时区对象相等
        assert dtype.tz == tz
        # 使用DatetimeTZDtype构造函数创建另一个DatetimeTZDtype对象，指定时区为日期范围的第一个日期的时区
        dtype = DatetimeTZDtype("ns", dr[0].tz)
        # 断言对象的时区与预期的时区对象相等
        assert dtype.tz == tz
class TestPeriodDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestPeriodDtype
        """
        return PeriodDtype("D")

    def test_hash_vs_equality(self, dtype):
        # 确保满足等价性语义
        dtype2 = PeriodDtype("D")
        dtype3 = PeriodDtype(dtype2)
        assert dtype == dtype2  # 检查相等性
        assert dtype2 == dtype  # 检查相等性
        assert dtype3 == dtype   # 检查相等性
        assert dtype is not dtype2   # 检查对象不同
        assert dtype2 is not dtype   # 检查对象不同
        assert dtype3 is not dtype   # 检查对象不同
        assert hash(dtype) == hash(dtype2)  # 检查哈希值相等性
        assert hash(dtype) == hash(dtype3)  # 检查哈希值相等性

    def test_construction(self):
        with pytest.raises(ValueError, match="Invalid frequency: xx"):
            PeriodDtype("xx")  # 测试无效频率抛出异常

        for s in ["period[D]", "Period[D]", "D"]:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day()  # 检查频率为天的情况

        for s in ["period[3D]", "Period[3D]", "3D"]:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Day(3)  # 检查频率为3天的情况

        for s in [
            "period[26h]",
            "Period[26h]",
            "26h",
            "period[1D2h]",
            "Period[1D2h]",
            "1D2h",
        ]:
            dt = PeriodDtype(s)
            assert dt.freq == pd.tseries.offsets.Hour(26)  # 检查频率为26小时的情况

    def test_cannot_use_custom_businessday(self):
        # GH#52534
        msg = "C is not supported as period frequency"
        msg1 = "<CustomBusinessDay> is not supported as period frequency"
        msg2 = r"PeriodDtype\[B\] is deprecated"
        with pytest.raises(ValueError, match=msg):
            PeriodDtype("C")  # 检查不支持 "C" 作为频率时是否抛出异常
        with pytest.raises(ValueError, match=msg1):
            with tm.assert_produces_warning(FutureWarning, match=msg2):
                PeriodDtype(pd.offsets.CustomBusinessDay())  # 检查不支持自定义工作日频率时是否抛出异常和警告

    def test_subclass(self):
        a = PeriodDtype("period[D]")
        b = PeriodDtype("period[3D]")

        assert issubclass(type(a), type(a))  # 检查类型 a 是其自身的子类
        assert issubclass(type(a), type(b))  # 检查类型 a 是类型 b 的子类

    def test_identity(self):
        assert PeriodDtype("period[D]") == PeriodDtype("period[D]")  # 检查相同频率的类型相等
        assert PeriodDtype("period[D]") is not PeriodDtype("period[D]")  # 检查相同频率的类型不是同一个对象

        assert PeriodDtype("period[3D]") == PeriodDtype("period[3D]")  # 检查相同频率的类型相等
        assert PeriodDtype("period[3D]") is not PeriodDtype("period[3D]")  # 检查相同频率的类型不是同一个对象

        assert PeriodDtype("period[1s1us]") == PeriodDtype("period[1000001us]")  # 检查相同频率的类型相等
        assert PeriodDtype("period[1s1us]") is not PeriodDtype("period[1000001us]")  # 检查相同频率的类型不是同一个对象

    def test_compat(self, dtype):
        assert not is_datetime64_ns_dtype(dtype)  # 检查给定类型不是 datetime64 纳秒类型
        assert not is_datetime64_ns_dtype("period[D]")  # 检查 "period[D]" 不是 datetime64 纳秒类型
        assert not is_datetime64_dtype(dtype)  # 检查给定类型不是 datetime64 类型
        assert not is_datetime64_dtype("period[D]")  # 检查 "period[D]" 不是 datetime64 类型
    # 测试从字符串构造 PeriodDtype 对象
    def test_construction_from_string(self, dtype):
        # 使用字符串 "period[D]" 构造 PeriodDtype 对象，断言与给定的 dtype 是否相等
        result = PeriodDtype("period[D]")
        assert is_dtype_equal(dtype, result)
        # 使用字符串 "period[D]" 构造 PeriodDtype 对象，断言与给定的 dtype 是否相等
        result = PeriodDtype.construct_from_string("period[D]")
        assert is_dtype_equal(dtype, result)

        # 使用 pytest 的 raises 断言，验证当传入一个列表时会引发 TypeError 异常，异常信息包含 "list"
        with pytest.raises(TypeError, match="list"):
            PeriodDtype.construct_from_string([1, 2, 3])

    # 使用 pytest 的 parametrize 装饰器，参数化测试字符串输入
    @pytest.mark.parametrize(
        "string",
        [
            "foo",
            "period[foo]",
            "foo[D]",
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
        ],
    )
    # 测试使用不合法的字符串构造 PeriodDtype 对象时是否会引发 TypeError 异常
    def test_construct_dtype_from_string_invalid_raises(self, string):
        # 构造异常消息
        msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        # 使用 pytest 的 raises 断言，验证当传入无效字符串时会引发 TypeError 异常，异常消息符合预期
        with pytest.raises(TypeError, match=re.escape(msg)):
            PeriodDtype.construct_from_string(string)

    # 测试 PeriodDtype 的 is_dtype 方法
    def test_is_dtype(self, dtype):
        # 验证给定的 dtype 是 PeriodDtype 类型
        assert PeriodDtype.is_dtype(dtype)
        assert PeriodDtype.is_dtype("period[D]")
        assert PeriodDtype.is_dtype("period[3D]")
        assert PeriodDtype.is_dtype(PeriodDtype("3D"))
        assert PeriodDtype.is_dtype("period[us]")
        assert PeriodDtype.is_dtype("period[s]")
        assert PeriodDtype.is_dtype(PeriodDtype("us"))
        assert PeriodDtype.is_dtype(PeriodDtype("s"))

        # 验证给定的内容不是 PeriodDtype 类型
        assert not PeriodDtype.is_dtype("D")
        assert not PeriodDtype.is_dtype("3D")
        assert not PeriodDtype.is_dtype("U")
        assert not PeriodDtype.is_dtype("s")
        assert not PeriodDtype.is_dtype("foo")
        assert not PeriodDtype.is_dtype(np.object_)
        assert not PeriodDtype.is_dtype(np.int64)
        assert not PeriodDtype.is_dtype(np.float64)

    # 测试 PeriodDtype 的相等性比较
    def test_equality(self, dtype):
        # 验证两个 PeriodDtype 对象或字符串表示相等
        assert is_dtype_equal(dtype, "period[D]")
        assert is_dtype_equal(dtype, PeriodDtype("D"))
        assert is_dtype_equal(dtype, PeriodDtype("D"))
        assert is_dtype_equal(PeriodDtype("D"), PeriodDtype("D"))

        # 验证两个 PeriodDtype 对象或字符串表示不相等
        assert not is_dtype_equal(dtype, "D")
        assert not is_dtype_equal(PeriodDtype("D"), PeriodDtype("2D"))

    # 测试 PeriodDtype 的基本功能
    def test_basic(self, dtype):
        # 验证在使用被弃用的 is_period_dtype 函数时会产生警告消息
        msg = "is_period_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 验证给定的 dtype 是否是 Period 类型
            assert is_period_dtype(dtype)

            # 创建一个 Period 索引对象，并验证其 dtype 是 PeriodDtype 类型
            pidx = pd.period_range("2013-01-01 09:00", periods=5, freq="h")
            assert is_period_dtype(pidx.dtype)
            assert is_period_dtype(pidx)

            # 创建一个 Series 对象，并验证其 dtype 是 PeriodDtype 类型
            s = Series(pidx, name="A")
            assert is_period_dtype(s.dtype)
            assert is_period_dtype(s)

            # 验证给定的内容不是 PeriodDtype 类型
            assert not is_period_dtype(np.dtype("float64"))
            assert not is_period_dtype(1.0)
    def test_freq_argument_required(self):
        # 测试频率参数的必要性
        msg = "missing 1 required positional argument: 'freq'"
        # 使用 pytest 检测是否会抛出 TypeError，并匹配特定的错误信息
        with pytest.raises(TypeError, match=msg):
            PeriodDtype()

        msg = "PeriodDtype argument should be string or BaseOffset, got NoneType"
        # 使用 pytest 检测是否会抛出 TypeError，并匹配特定的错误信息
        with pytest.raises(TypeError, match=msg):
            # 创建一个 PeriodDtype 实例，传入 None 作为参数
            PeriodDtype(None)

    def test_not_string(self):
        # 虽然 PeriodDtype 的类型为 object，但不能是字符串
        assert not is_string_dtype(PeriodDtype("D"))

    def test_perioddtype_caching_dateoffset_normalize(self):
        # GH 24121
        # 创建一个 PeriodDtype 实例，使用年末日期偏移，并将 normalize 设置为 True
        per_d = PeriodDtype(pd.offsets.YearEnd(normalize=True))
        # 断言 per_d 的频率属性是否为 normalize

        # 创建一个 PeriodDtype 实例，使用年末日期偏移，并将 normalize 设置为 False
        per_d2 = PeriodDtype(pd.offsets.YearEnd(normalize=False))
        # 断言 per_d2 的频率属性是否不为 normalize

    def test_dont_keep_ref_after_del(self):
        # GH 54184
        # 创建一个 PeriodDtype 实例，频率为每日
        dtype = PeriodDtype("D")
        # 使用弱引用 ref 来引用 dtype
        ref = weakref.ref(dtype)
        # 删除 dtype 对象
        del dtype
        # 断言弱引用 ref 是否已经指向 None
        assert ref() is None
class TestIntervalDtype(Base):
    @pytest.fixture
    def dtype(self):
        """
        Class level fixture of dtype for TestIntervalDtype
        """
        return IntervalDtype("int64", "right")

    def test_hash_vs_equality(self, dtype):
        # make sure that we satisfy is semantics
        dtype2 = IntervalDtype("int64", "right")
        dtype3 = IntervalDtype(dtype2)
        assert dtype == dtype2  # Assert equality between dtype and dtype2
        assert dtype2 == dtype   # Assert equality between dtype2 and dtype
        assert dtype3 == dtype   # Assert equality between dtype3 and dtype
        assert dtype is not dtype2  # Assert that dtype and dtype2 are not the same object
        assert dtype2 is not dtype3  # Assert that dtype2 and dtype3 are not the same object
        assert dtype3 is not dtype   # Assert that dtype3 and dtype are not the same object
        assert hash(dtype) == hash(dtype2)  # Assert hash equality between dtype and dtype2
        assert hash(dtype) == hash(dtype3)  # Assert hash equality between dtype and dtype3

        dtype1 = IntervalDtype("interval")
        dtype2 = IntervalDtype(dtype1)
        dtype3 = IntervalDtype("interval")
        assert dtype2 == dtype1  # Assert equality between dtype2 and dtype1
        assert dtype2 == dtype2  # Assert dtype2 is equal to itself
        assert dtype2 == dtype3  # Assert equality between dtype2 and dtype3
        assert dtype2 is not dtype1  # Assert that dtype2 and dtype1 are not the same object
        assert dtype2 is dtype2  # Assert that dtype2 is the same object as itself
        assert dtype2 is not dtype3  # Assert that dtype2 and dtype3 are not the same object
        assert hash(dtype2) == hash(dtype1)  # Assert hash equality between dtype2 and dtype1
        assert hash(dtype2) == hash(dtype2)  # Assert dtype2 hash is equal to itself
        assert hash(dtype2) == hash(dtype3)  # Assert hash equality between dtype2 and dtype3

    @pytest.mark.parametrize(
        "subtype", ["interval[int64]", "Interval[int64]", "int64", np.dtype("int64")]
    )
    def test_construction(self, subtype):
        i = IntervalDtype(subtype, closed="right")
        assert i.subtype == np.dtype("int64")  # Assert subtype is 'int64'
        msg = "is_interval_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(i)  # Assert i is recognized as an interval dtype

    @pytest.mark.parametrize(
        "subtype", ["interval[int64]", "Interval[int64]", "int64", np.dtype("int64")]
    )
    def test_construction_allows_closed_none(self, subtype):
        # GH#38394
        dtype = IntervalDtype(subtype)
        assert dtype.closed is None  # Assert closed attribute is None

    def test_closed_mismatch(self):
        msg = "'closed' keyword does not match value specified in dtype string"
        with pytest.raises(ValueError, match=msg):
            IntervalDtype("interval[int64, left]", "right")  # Expecting ValueError due to mismatch

    @pytest.mark.parametrize("subtype", [None, "interval", "Interval"])
    def test_construction_generic(self, subtype):
        # generic
        i = IntervalDtype(subtype)
        assert i.subtype is None  # Assert subtype is None for generic construction
        msg = "is_interval_dtype is deprecated"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            assert is_interval_dtype(i)  # Assert i is recognized as an interval dtype

    @pytest.mark.parametrize(
        "subtype",
        [
            CategoricalDtype(list("abc"), False),
            CategoricalDtype(list("wxyz"), True),
            object,
            str,
            "<U10",
            "interval[category]",
            "interval[object]",
        ],
    )
    # 测试 IntervalDtype 类构造函数在不支持的子类型上抛出 TypeError 异常
    def test_construction_not_supported(self, subtype):
        # GH 19016
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalDtype"
        )
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并验证异常消息是否匹配
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    # 使用参数化测试检验 IntervalDtype 构造函数在错误情况下是否抛出 TypeError 异常
    @pytest.mark.parametrize("subtype", ["xx", "IntervalA", "Interval[foo]"])
    def test_construction_errors(self, subtype):
        msg = "could not construct IntervalDtype"
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并验证异常消息是否匹配
        with pytest.raises(TypeError, match=msg):
            IntervalDtype(subtype)

    # 检验 IntervalDtype 类的构造中，dtype.closed 参数必须匹配的情况
    def test_closed_must_match(self):
        # GH#37933
        dtype = IntervalDtype(np.float64, "left")

        msg = "dtype.closed and 'closed' do not match"
        # 使用 pytest 检查是否抛出预期的 ValueError 异常，并验证异常消息是否匹配
        with pytest.raises(ValueError, match=msg):
            IntervalDtype(dtype, closed="both")

    # 测试在指定无效的 closed 参数时，是否抛出 ValueError 异常
    def test_closed_invalid(self):
        with pytest.raises(ValueError, match="closed must be one of"):
            IntervalDtype(np.float64, "foo")

    # 测试通过字符串构造 IntervalDtype 对象的方法，并验证结果是否符合预期
    def test_construction_from_string(self, dtype):
        result = IntervalDtype("interval[int64, right]")
        assert is_dtype_equal(dtype, result)
        result = IntervalDtype.construct_from_string("interval[int64, right]")
        assert is_dtype_equal(dtype, result)

    # 使用参数化测试检验在非字符串参数传递时，是否抛出预期的 TypeError 异常
    @pytest.mark.parametrize("string", [0, 3.14, ("a", "b"), None])
    def test_construction_from_string_errors(self, string):
        # 这些是完全无效的情况
        msg = f"'construct_from_string' expects a string, got {type(string)}"

        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并验证异常消息是否匹配
        with pytest.raises(TypeError, match=re.escape(msg)):
            IntervalDtype.construct_from_string(string)

    # 使用参数化测试检验在无效子类型字符串传递时，是否抛出预期的 TypeError 异常
    @pytest.mark.parametrize("string", ["foo", "foo[int64]", "IntervalA"])
    def test_construction_from_string_error_subtype(self, string):
        # 这是一个无效的子类型
        msg = (
            "Incorrectly formatted string passed to constructor. "
            r"Valid formats include Interval or Interval\[dtype\] "
            "where dtype is numeric, datetime, or timedelta"
        )

        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并验证异常消息是否匹配
        with pytest.raises(TypeError, match=msg):
            IntervalDtype.construct_from_string(string)

    # 检验 IntervalDtype 类是否正确继承自其自身类型
    def test_subclass(self):
        a = IntervalDtype("interval[int64, right]")
        b = IntervalDtype("interval[int64, right]")

        assert issubclass(type(a), type(a))
        assert issubclass(type(a), type(b))
    @pytest.mark.parametrize(
        "subtype",
        [
            None,  # 参数化测试的子类型，这里是 None，即不指定具体子类型
            "interval",  # 参数化测试的子类型，指定为 "interval"
            "Interval",  # 参数化测试的子类型，指定为 "Interval"
            "int64",  # 参数化测试的子类型，指定为 "int64"
            "uint64",  # 参数化测试的子类型，指定为 "uint64"
            "float64",  # 参数化测试的子类型，指定为 "float64"
            "complex128",  # 参数化测试的子类型，指定为 "complex128"
            "datetime64",  # 参数化测试的子类型，指定为 "datetime64"
            "timedelta64",  # 参数化测试的子类型，指定为 "timedelta64"
            PeriodDtype("Q"),  # 参数化测试的子类型，指定为 PeriodDtype("Q")
        ],
    )
    def test_equality_generic(self, subtype):
        # GH 18980
        closed = "right" if subtype is not None else None  # 如果 subtype 不是 None，则设置 closed 为 "right"，否则为 None
        dtype = IntervalDtype(subtype, closed=closed)  # 使用给定的 subtype 和 closed 创建 IntervalDtype 对象
        assert is_dtype_equal(dtype, "interval")  # 断言新创建的 dtype 与 "interval" 相等
        assert is_dtype_equal(dtype, IntervalDtype())  # 断言新创建的 dtype 与默认的 IntervalDtype 相等
    def test_name_repr(self, subtype):
        # GH 18980
        # 根据 subtype 是否为 None 设置 closed 变量为 "right" 或者 None
        closed = "right" if subtype is not None else None
        # 使用 IntervalDtype 类创建 dtype 对象，设置 subtype 和 closed 属性
        dtype = IntervalDtype(subtype, closed=closed)
        # 期望的字符串表示形式
        expected = f"interval[{subtype}, {closed}]"
        # 断言 dtype 对象的字符串表示形式符合预期
        assert str(dtype) == expected
        # 断言 dtype 对象的名称为 "interval"
        assert dtype.name == "interval"

    @pytest.mark.parametrize("subtype", [None, "interval", "Interval"])
    def test_name_repr_generic(self, subtype):
        # GH 18980
        # 使用 IntervalDtype 类创建 dtype 对象，设置 subtype 属性
        dtype = IntervalDtype(subtype)
        # 断言 dtype 对象的字符串表示形式为 "interval"
        assert str(dtype) == "interval"
        # 断言 dtype 对象的名称为 "interval"
        assert dtype.name == "interval"

    def test_basic(self, dtype):
        # 检查警告信息
        msg = "is_interval_dtype is deprecated"
        # 确认在上下文中生成 DeprecationWarning 类型的警告信息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 检查 dtype 是否为 interval 数据类型
            assert is_interval_dtype(dtype)

            # 创建一个 IntervalIndex 对象 ii，从 0 到 2 的范围
            ii = IntervalIndex.from_breaks(range(3))

            # 检查 ii 的数据类型是否为 interval 数据类型
            assert is_interval_dtype(ii.dtype)
            # 检查 ii 对象本身是否为 interval 数据类型
            assert is_interval_dtype(ii)

            # 创建一个 Series 对象 s，使用 ii 作为数据，名称为 "A"
            s = Series(ii, name="A")

            # 检查 s 的数据类型是否为 interval 数据类型
            assert is_interval_dtype(s.dtype)
            # 检查 s 对象本身是否为 interval 数据类型
            assert is_interval_dtype(s)

    def test_basic_dtype(self):
        # 检查警告信息
        msg = "is_interval_dtype is deprecated"
        # 确认在上下文中生成 DeprecationWarning 类型的警告信息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 检查字符串 "interval[int64, both]" 是否为 interval 数据类型
            assert is_interval_dtype("interval[int64, both]")
            # 检查 IntervalIndex 对象是否为 interval 数据类型
            assert is_interval_dtype(IntervalIndex.from_tuples([(0, 1)]))
            # 检查 IntervalIndex 对象是否为 interval 数据类型
            assert is_interval_dtype(IntervalIndex.from_breaks(np.arange(4)))
            # 检查 IntervalIndex 对象是否为 interval 数据类型
            assert is_interval_dtype(
                IntervalIndex.from_breaks(date_range("20130101", periods=3))
            )
            # 检查字符串 "U" 不是 interval 数据类型
            assert not is_interval_dtype("U")
            # 检查字符串 "S" 不是 interval 数据类型
            assert not is_interval_dtype("S")
            # 检查字符串 "foo" 不是 interval 数据类型
            assert not is_interval_dtype("foo")
            # 检查 np.object_ 不是 interval 数据类型
            assert not is_interval_dtype(np.object_)
            # 检查 np.int64 不是 interval 数据类型
            assert not is_interval_dtype(np.int64)
            # 检查 np.float64 不是 interval 数据类型
            assert not is_interval_dtype(np.float64)

    def test_caching(self):
        # GH 54184: Caching not shown to improve performance
        # 重置 IntervalDtype 的缓存
        IntervalDtype.reset_cache()
        # 使用 "int64" 和 "right" 创建 IntervalDtype 对象 dtype
        dtype = IntervalDtype("int64", "right")
        # 断言 IntervalDtype 的缓存为空
        assert len(IntervalDtype._cache_dtypes) == 0

        # 创建一个简单的 IntervalDtype 对象
        IntervalDtype("interval")
        # 再次断言 IntervalDtype 的缓存为空
        assert len(IntervalDtype._cache_dtypes) == 0

        # 再次重置 IntervalDtype 的缓存
        IntervalDtype.reset_cache()
        # 对 dtype 对象进行 pickle 转换
        tm.round_trip_pickle(dtype)
        # 最后断言 IntervalDtype 的缓存为空
        assert len(IntervalDtype._cache_dtypes) == 0

    def test_not_string(self):
        # GH30568: though IntervalDtype has object kind, it cannot be string
        # 断言 IntervalDtype 类型不是字符串类型
        assert not is_string_dtype(IntervalDtype())

    def test_unpickling_without_closed(self):
        # GH#38394
        # 使用 "interval" 创建 IntervalDtype 对象 dtype
        dtype = IntervalDtype("interval")
        # 断言 dtype 的 _closed 属性为 None
        assert dtype._closed is None
        # 对 dtype 进行 pickle 转换
        tm.round_trip_pickle(dtype)

    def test_dont_keep_ref_after_del(self):
        # GH 54184
        # 使用 "int64" 和 "right" 创建 IntervalDtype 对象 dtype
        dtype = IntervalDtype("int64", "right")
        # 创建对 dtype 的弱引用 ref
        ref = weakref.ref(dtype)
        # 删除 dtype 对象
        del dtype
        # 断言 ref 引用为 None，即对象已被删除
        assert ref() is None
class TestCategoricalDtypeParametrized:
    # 使用 pytest 的参数化装饰器标记，为测试方法 test_basic 提供多组参数
    @pytest.mark.parametrize(
        "categories",
        [
            list("abcd"),  # 测试用例：字符列表 ['a', 'b', 'c', 'd']
            np.arange(1000),  # 测试用例：从0到999的整数数组
            ["a", "b", 10, 2, 1.3, True],  # 测试用例：混合类型列表
            [True, False],  # 测试用例：布尔值列表
            date_range("2017", periods=4),  # 测试用例：包含4个日期的时间范围
        ],
    )
    # 测试基本功能，验证类别和有序性
    def test_basic(self, categories, ordered):
        c1 = CategoricalDtype(categories, ordered=ordered)
        tm.assert_index_equal(c1.categories, pd.Index(categories))
        assert c1.ordered is ordered

    # 测试顺序对结果对象的影响
    def test_order_matters(self):
        categories = ["a", "b"]
        c1 = CategoricalDtype(categories, ordered=True)
        c2 = CategoricalDtype(categories, ordered=False)
        c3 = CategoricalDtype(categories, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    # 使用 pytest 的参数化装饰器标记，为测试方法 test_unordered_same 提供多组参数
    @pytest.mark.parametrize("ordered", [False, None])
    # 测试无序情况下的哈希值是否相等
    def test_unordered_same(self, ordered):
        c1 = CategoricalDtype(["a", "b"], ordered=ordered)
        c2 = CategoricalDtype(["b", "a"], ordered=ordered)
        assert hash(c1) == hash(c2)

    # 测试类别设定功能
    def test_categories(self):
        result = CategoricalDtype(["a", "b", "c"])
        tm.assert_index_equal(result.categories, pd.Index(["a", "b", "c"]))
        assert result.ordered is False

    # 测试相等但类型不同的情况
    def test_equal_but_different(self):
        c1 = CategoricalDtype([1, 2, 3])
        c2 = CategoricalDtype([1.0, 2.0, 3.0])
        assert c1 is not c2
        assert c1 != c2

    # 测试相等且混合类型的情况
    def test_equal_but_different_mixed_dtypes(self):
        c1 = CategoricalDtype([1, 2, "3"])
        c2 = CategoricalDtype(["3", 1, 2])
        assert c1 is not c2
        assert c1 == c2

    # 测试相等空类别且有序的情况
    def test_equal_empty_ordered(self):
        c1 = CategoricalDtype([], ordered=True)
        c2 = CategoricalDtype([], ordered=True)
        assert c1 is not c2
        assert c1 == c2

    # 测试相等空类别且无序的情况
    def test_equal_empty_unordered(self):
        c1 = CategoricalDtype([])
        c2 = CategoricalDtype([])
        assert c1 is not c2
        assert c1 == c2

    # 使用 pytest 的参数化装饰器标记，为测试方法 test_order_hashes_different 提供多组参数
    @pytest.mark.parametrize("v1, v2", [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1])])
    # 测试不同参数下的哈希值是否不同
    def test_order_hashes_different(self, v1, v2):
        c1 = CategoricalDtype(v1, ordered=False)
        c2 = CategoricalDtype(v2, ordered=True)
        c3 = CategoricalDtype(v1, ordered=None)
        assert c1 is not c2
        assert c1 is not c3

    # 测试空值 nan 是否会引发异常
    def test_nan_invalid(self):
        msg = "Categorical categories cannot be null"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype([1, 2, np.nan])

    # 测试非唯一类别是否会引发异常
    def test_non_unique_invalid(self):
        msg = "Categorical categories must be unique"
        with pytest.raises(ValueError, match=msg):
            CategoricalDtype([1, 2, 1])

    # 测试相同类别但顺序不同的情况
    def test_same_categories_different_order(self):
        c1 = CategoricalDtype(["a", "b"], ordered=True)
        c2 = CategoricalDtype(["b", "a"], ordered=True)
        assert c1 is not c2

    # 使用 pytest 的参数化装饰器标记，为测试方法 test_unordered_same 提供多组参数
    @pytest.mark.parametrize("ordered2", [True, False, None])
    def test_categorical_equality(self, ordered, ordered2):
        # 测试分类数据类型的相等性

        # 相同的分类和顺序
        # 任何组合的 None/False 都相等
        # 只有 True/True 的组合是相等的
        c1 = CategoricalDtype(list("abc"), ordered)
        c2 = CategoricalDtype(list("abc"), ordered2)
        result = c1 == c2
        expected = bool(ordered) is bool(ordered2)
        assert result is expected

        # 相同的分类，不同的顺序
        # 任何组合的 None/False 都相等（顺序无关紧要）
        # 任何包含 True 的组合都不相等（分类顺序不同）
        c1 = CategoricalDtype(list("abc"), ordered)
        c2 = CategoricalDtype(list("cab"), ordered2)
        result = c1 == c2
        expected = (bool(ordered) is False) and (bool(ordered2) is False)
        assert result is expected

        # 不同的分类
        c2 = CategoricalDtype([1, 2, 3], ordered2)
        assert c1 != c2

        # 空分类
        c1 = CategoricalDtype(list("abc"), ordered)
        c2 = CategoricalDtype(None, ordered2)
        c3 = CategoricalDtype(None, ordered)
        assert c1 != c2
        assert c2 != c1
        assert c2 == c3

    def test_categorical_dtype_equality_requires_categories(self):
        # 测试分类数据类型的相等性要求有分类数据

        # categories=None 的 CategoricalDtype 不等于任何完全初始化的 CategoricalDtype
        first = CategoricalDtype(["a", "b"])
        second = CategoricalDtype()
        third = CategoricalDtype(ordered=True)

        assert second == second
        assert third == third

        assert first != second
        assert second != first
        assert first != third
        assert third != first
        assert second == third
        assert third == second

    @pytest.mark.parametrize("categories", [list("abc"), None])
    @pytest.mark.parametrize("other", ["category", "not a category"])
    def test_categorical_equality_strings(self, categories, ordered, other):
        # 测试分类数据类型和字符串的相等性

        c1 = CategoricalDtype(categories, ordered)
        result = c1 == other
        expected = other == "category"
        assert result is expected

    def test_invalid_raises(self):
        # 测试无效参数时的异常抛出

        with pytest.raises(TypeError, match="ordered"):
            CategoricalDtype(["a", "b"], ordered="foo")

        with pytest.raises(TypeError, match="'categories' must be list-like"):
            CategoricalDtype("category")

    def test_mixed(self):
        # 测试混合类型的哈希值

        a = CategoricalDtype(["a", "b", 1, 2])
        b = CategoricalDtype(["a", "b", "1", "2"])
        assert hash(a) != hash(b)

    def test_from_categorical_dtype_identity(self):
        # 测试从 Categorical 类型转换回来的身份验证

        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # 没有更改时的身份测试
        c2 = CategoricalDtype._from_categorical_dtype(c1)
        assert c2 is c1
    def test_from_categorical_dtype_categories(self):
        # 创建一个包含 [1, 2] 的 Categorical 对象，指定其类别为 [1, 2, 3]，并且是有序的
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # 覆盖类别信息
        result = CategoricalDtype._from_categorical_dtype(c1, categories=[2, 3])
        # 断言结果与预期的 CategoricalDtype([2, 3], ordered=True) 相等
        assert result == CategoricalDtype([2, 3], ordered=True)

    def test_from_categorical_dtype_ordered(self):
        # 创建一个包含 [1, 2] 的 Categorical 对象，指定其类别为 [1, 2, 3]，并且是有序的
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # 覆盖有序属性
        result = CategoricalDtype._from_categorical_dtype(c1, ordered=False)
        # 断言结果与预期的 CategoricalDtype([1, 2, 3], ordered=False) 相等
        assert result == CategoricalDtype([1, 2, 3], ordered=False)

    def test_from_categorical_dtype_both(self):
        # 创建一个包含 [1, 2] 的 Categorical 对象，指定其类别为 [1, 2, 3]，并且是有序的
        c1 = Categorical([1, 2], categories=[1, 2, 3], ordered=True)
        # 同时覆盖类别和有序属性
        result = CategoricalDtype._from_categorical_dtype(
            c1, categories=[1, 2], ordered=False
        )
        # 断言结果与预期的 CategoricalDtype([1, 2], ordered=False) 相等
        assert result == CategoricalDtype([1, 2], ordered=False)

    def test_str_vs_repr(self, ordered, using_infer_string):
        # 创建一个包含 ["a", "b"] 的 CategoricalDtype 对象，根据 ordered 参数指定是否有序
        c1 = CategoricalDtype(["a", "b"], ordered=ordered)
        # 断言 str(c1) 结果为 "category"
        assert str(c1) == "category"
        # 根据 using_infer_string 参数选择 dtype 类型
        dtype = "string" if using_infer_string else "object"
        # 生成匹配 CategoricalDtype 对象 repr 表示的正则表达式模式
        pat = (
            r"CategoricalDtype\(categories=\[.*\], ordered={ordered}, "
            rf"categories_dtype={dtype}\)"
        )
        # 使用正则表达式匹配 repr(c1) 结果是否符合 pat 模式
        assert re.match(pat.format(ordered=ordered), repr(c1))

    def test_categorical_categories(self):
        # GH17884 测试
        # 创建一个包含 ["a", "b"] 的 Categorical 对象，并生成对应的 CategoricalDtype 对象
        c1 = CategoricalDtype(Categorical(["a", "b"]))
        # 断言 c1 的类别与预期的 pd.Index(["a", "b"]) 相等
        tm.assert_index_equal(c1.categories, pd.Index(["a", "b"]))
        # 创建一个包含 ["a", "b"] 的 CategoricalIndex 对象，并生成对应的 CategoricalDtype 对象
        c1 = CategoricalDtype(CategoricalIndex(["a", "b"]))
        # 断言 c1 的类别与预期的 pd.Index(["a", "b"]) 相等
        tm.assert_index_equal(c1.categories, pd.Index(["a", "b"]))

    @pytest.mark.parametrize(
        "new_categories", [list("abc"), list("cba"), list("wxyz"), None]
    )
    @pytest.mark.parametrize("new_ordered", [True, False, None])
    def test_update_dtype(self, ordered, new_categories, new_ordered):
        # 创建一个类别为 ['a', 'b', 'c'] 的 CategoricalDtype 对象，根据 ordered 参数指定是否有序
        original_categories = list("abc")
        dtype = CategoricalDtype(original_categories, ordered)
        # 创建一个新的 CategoricalDtype 对象，根据 new_categories 和 new_ordered 参数
        new_dtype = CategoricalDtype(new_categories, new_ordered)

        # 执行 dtype 的 update_dtype 方法，返回更新后的结果
        result = dtype.update_dtype(new_dtype)
        # 根据 new_categories 或者 original_categories 生成预期的类别 Index
        expected_categories = pd.Index(new_categories or original_categories)
        # 根据 new_ordered 或者 dtype 的 ordered 属性生成预期的 ordered 属性值
        expected_ordered = new_ordered if new_ordered is not None else dtype.ordered

        # 断言 result 的类别与预期的类别相等
        tm.assert_index_equal(result.categories, expected_categories)
        # 断言 result 的 ordered 属性与预期的 ordered 属性值相等
        assert result.ordered is expected_ordered

    def test_update_dtype_string(self, ordered):
        # 创建一个类别为 ['a', 'b', 'c'] 的 CategoricalDtype 对象，根据 ordered 参数指定是否有序
        dtype = CategoricalDtype(list("abc"), ordered)
        # 复制 dtype 的类别和有序属性作为预期结果
        expected_categories = dtype.categories
        expected_ordered = dtype.ordered
        # 执行 dtype 的 update_dtype 方法，传入字符串 "category"，返回更新后的结果
        result = dtype.update_dtype("category")
        # 断言 result 的类别与预期的类别相等
        tm.assert_index_equal(result.categories, expected_categories)
        # 断言 result 的 ordered 属性与预期的 ordered 属性值相等
        assert result.ordered is expected_ordered

    @pytest.mark.parametrize("bad_dtype", ["foo", object, np.int64, PeriodDtype("Q")])
    # 定义一个测试方法，用于测试更新数据类型时的错误情况
    def test_update_dtype_errors(self, bad_dtype):
        # 创建一个不允许排序的分类数据类型对象，包含字符集合["a", "b", "c"]
        dtype = CategoricalDtype(list("abc"), False)
        # 定义错误消息，要求传递一个 CategoricalDtype 对象以执行更新操作
        msg = "a CategoricalDtype must be passed to perform an update, "
        # 使用 pytest 来验证是否会引发 ValueError 异常，并检查异常消息是否匹配定义的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用被测试的方法 update_dtype，传入一个错误的数据类型 bad_dtype
            dtype.update_dtype(bad_dtype)
@pytest.mark.parametrize(
    "dtype", [CategoricalDtype, IntervalDtype, DatetimeTZDtype, PeriodDtype]
)
def test_registry(dtype):
    # 断言被测试的 dtype 类型存在于注册表 registry.dtypes 中
    assert dtype in registry.dtypes


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("int64", None),
        ("interval", IntervalDtype()),
        ("interval[int64, neither]", IntervalDtype()),
        ("interval[datetime64[ns], left]", IntervalDtype("datetime64[ns]", "left")),
        ("period[D]", PeriodDtype("D")),
        ("category", CategoricalDtype()),
        ("datetime64[ns, US/Eastern]", DatetimeTZDtype("ns", "US/Eastern")),
    ],
)
def test_registry_find(dtype, expected):
    # 断言 registry.find(dtype) 的结果等于 expected
    assert registry.find(dtype) == expected


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (str, False),
        (int, False),
        (bool, True),
        (np.bool_, True),
        (np.array(["a", "b"]), False),
        (Series([1, 2]), False),
        (np.array([True, False]), True),
        (Series([True, False]), True),
        (SparseArray([True, False]), True),
        (SparseDtype(bool), True),
    ],
)
def test_is_bool_dtype(dtype, expected):
    # 调用 is_bool_dtype 函数，并断言其结果为 expected
    result = is_bool_dtype(dtype)
    assert result is expected


def test_is_bool_dtype_sparse():
    # 测试 is_bool_dtype 函数处理稀疏数据类型的情况
    result = is_bool_dtype(Series(SparseArray([True, False])))
    assert result is True


@pytest.mark.parametrize(
    "check",
    [
        is_categorical_dtype,
        is_datetime64tz_dtype,
        is_period_dtype,
        is_datetime64_ns_dtype,
        is_datetime64_dtype,
        is_interval_dtype,
        is_datetime64_any_dtype,
        is_string_dtype,
        is_bool_dtype,
    ],
)
def test_is_dtype_no_warning(check):
    # 创建一个包含数据列 "A" 的 DataFrame
    data = pd.DataFrame({"A": [1, 2]})

    # 设置警告和消息
    warn = None
    msg = f"{check.__name__} is deprecated"
    
    # 如果检查函数在特定的几种情况下，会产生 DeprecationWarning
    if (
        check is is_categorical_dtype
        or check is is_interval_dtype
        or check is is_datetime64tz_dtype
        or check is is_period_dtype
    ):
        warn = DeprecationWarning

    # 使用 assert_produces_warning 上下文管理器来检查警告
    with tm.assert_produces_warning(warn, match=msg):
        check(data)

    with tm.assert_produces_warning(warn, match=msg):
        check(data["A"])


def test_period_dtype_compare_to_string():
    # 测试 PeriodDtype 类型与字符串进行比较
    # 参考：https://github.com/pandas-dev/pandas/issues/37265
    dtype = PeriodDtype(freq="M")
    assert (dtype == "period[M]") is True
    assert (dtype != "period[M]") is False


def test_compare_complex_dtypes():
    # 测试复杂数据类型的比较
    # 参考：GH 28050
    df = pd.DataFrame(np.arange(5).astype(np.complex128))
    msg = "'<' not supported between instances of 'complex' and 'complex'"

    # 使用 pytest.raises 来断言期望的异常被抛出
    with pytest.raises(TypeError, match=msg):
        df < df.astype(object)

    with pytest.raises(TypeError, match=msg):
        df.lt(df.astype(object))


def test_cast_string_to_complex():
    # 测试将字符串转换为复数类型
    # 参考：GH 4895
    expected = pd.DataFrame(["1.0+5j", "1.5-3j"], dtype=complex)
    result = pd.DataFrame(["1.0+5j", "1.5-3j"]).astype(complex)
    tm.assert_frame_equal(result, expected)


def test_categorical_complex():
    # 测试复杂类型的分类数据
    result = Categorical([1, 2 + 2j])
    # 创建一个预期的分类数据对象，包含复数和实数部分
    expected = Categorical([1.0 + 0.0j, 2.0 + 2.0j])
    # 使用测试工具比较两个分类数据对象是否相等
    tm.assert_categorical_equal(result, expected)
    
    # 创建一个新的分类数据对象，包含整数和复数
    result = Categorical([1, 2, 2 + 2j])
    # 创建一个预期的分类数据对象，包含实数和复数，其中一项为 2.0 + 0.0j
    expected = Categorical([1.0 + 0.0j, 2.0 + 0.0j, 2.0 + 2.0j])
    # 使用测试工具比较两个分类数据对象是否相等
    tm.assert_categorical_equal(result, expected)
# 测试多列数据类型赋值
def test_multi_column_dtype_assignment():
    # GitHub issue #27583
    # 创建一个包含两列的DataFrame，'a'列是包含一个浮点数的列表，'b'列包含一个浮点数
    df = pd.DataFrame({"a": [0.0], "b": 0.0})
    # 创建预期结果的DataFrame，'a'列是包含一个整数的列表，'b'列包含一个整数
    expected = pd.DataFrame({"a": [0], "b": 0})

    # 将DataFrame的'a'和'b'列同时赋值为0
    df[["a", "b"]] = 0
    # 断言修改后的DataFrame与预期结果相等
    tm.assert_frame_equal(df, expected)

    # 将DataFrame的'b'列赋值为0
    df["b"] = 0
    # 断言修改后的DataFrame与预期结果相等
    tm.assert_frame_equal(df, expected)


# 测试在空标签位置使用loc进行赋值时不进行数据类型转换
def test_loc_setitem_empty_labels_no_dtype_conversion():
    # GitHub issue #29707

    # 创建一个包含'a'列的DataFrame，列中包含两个整数
    df = pd.DataFrame({"a": [2, 3]})
    # 复制原始DataFrame作为预期结果
    expected = df.copy()
    # 断言DataFrame的'a'列的数据类型为'int64'
    assert df.a.dtype == "int64"
    
    # 使用空的标签位置loc赋值为0.1
    df.loc[[]] = 0.1

    # 断言DataFrame的'a'列的数据类型仍为'int64'，并且修改后的DataFrame与预期结果相等
    assert df.a.dtype == "int64"
    tm.assert_frame_equal(df, expected)
```