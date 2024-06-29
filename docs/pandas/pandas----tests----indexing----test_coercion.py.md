# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_coercion.py`

```
###############################################################
# Index / Series common tests which may trigger dtype coercions
###############################################################

# 导入必要的模块和库
from __future__ import annotations

from datetime import (
    datetime,         # 导入 datetime 模块中的 datetime 类
    timedelta,        # 导入 datetime 模块中的 timedelta 类
)
import itertools      # 导入 itertools 库，用于生成迭代器的函数

import numpy as np    # 导入 NumPy 库，用于科学计算

import pytest         # 导入 Pytest 库，用于编写和运行测试用例

from pandas._config import using_pyarrow_string_dtype   # 导入 pandas 库中的配置项

from pandas.compat import (
    IS64,                    # 导入 pandas 兼容性模块中的 IS64 常量
    is_platform_windows,     # 导入 pandas 兼容性模块中的 is_platform_windows 函数
)
from pandas.compat.numpy import np_version_gt2           # 导入 pandas 兼容性模块中的 np_version_gt2 函数

import pandas as pd          # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 测试模块

###############################################################
# Fixture to check comprehensiveness of test coverage
###############################################################

@pytest.fixture(autouse=True, scope="class")
def check_comprehensiveness(request):
    # 检查测试覆盖率的完整性
    cls = request.cls
    combos = itertools.product(cls.klasses, cls.dtypes, [cls.method])

    def has_test(combo):
        klass, dtype, method = combo
        cls_funcs = request.node.session.items
        return any(
            klass in x.name and dtype in x.name and method in x.name for x in cls_funcs
        )

    opts = request.config.option
    if opts.lf or opts.keyword:
        # 如果运行参数指定为 "last-failed" 或者 -k foo，则只运行部分测试
        yield
    else:
        for combo in combos:
            if not has_test(combo):
                raise AssertionError(
                    f"test method is not defined: {cls.__name__}, {combo}"
                )

        yield


###############################################################
# Base class defining common attributes for coercion tests
###############################################################

class CoercionBase:
    klasses = ["index", "series"]   # 定义类属性 klasses，包含 index 和 series 字符串列表
    dtypes = [                     # 定义类属性 dtypes，包含多种数据类型字符串
        "object",
        "int64",
        "float64",
        "complex128",
        "bool",
        "datetime64",
        "datetime64tz",
        "timedelta64",
        "period",
    ]

    @property
    def method(self):
        raise NotImplementedError(self)


###############################################################
# Test class for specific coercion method 'setitem'
###############################################################

class TestSetitemCoercion(CoercionBase):
    method = "setitem"   # 定义属性 method 为字符串 "setitem"

    # 禁用完整性测试，因为大部分已经移到 SetitemCastingEquivalents 子类的 tests.series.indexing.test_setitem 中
    klasses: list[str] = []

    def test_setitem_series_no_coercion_from_values_list(self):
        # 测试用例：确保在使用 np.array(ser.values) 时不会将整数转换为字符串
        ser = pd.Series(["a", 1])
        ser[:] = list(ser.values)

        expected = pd.Series(["a", 1])

        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(
        self, original_series, loc_key, expected_index, expected_dtype
        # 辅助方法：用于测试索引转换的一致性
    ):
        """test index's coercion triggered by assign key"""
        # 复制原始 Series 对象
        temp = original_series.copy()
        # GH#33469 pre-2.0 with int loc_key and temp.index.dtype == np.float64
        #  `temp[loc_key] = 5` treated loc_key as positional
        # 将 loc_key 对应的值设为 5
        temp[loc_key] = 5
        # 创建预期的 Series 对象
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        # 断言 temp 和 exp 相等
        tm.assert_series_equal(temp, exp)
        # 明确检查索引的数据类型
        assert temp.index.dtype == expected_dtype

        # 复制原始 Series 对象
        temp = original_series.copy()
        # 将 loc_key 对应的值设为 5
        temp.loc[loc_key] = 5
        # 创建预期的 Series 对象
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        # 断言 temp 和 exp 相等
        tm.assert_series_equal(temp, exp)
        # 明确检查索引的数据类型
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize(
        "val,exp_dtype", [("x", object), (5, IndexError), (1.1, object)]
    )
    def test_setitem_index_object(self, val, exp_dtype):
        # 创建具有对象类型索引的 Series 对象
        obj = pd.Series([1, 2, 3, 4], index=pd.Index(list("abcd"), dtype=object))
        assert obj.index.dtype == object

        # 创建预期的索引
        exp_index = pd.Index(list("abcd") + [val], dtype=object)
        # 调用 _assert_setitem_index_conversion 方法进行设置索引测试
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.int64), (1.1, np.float64), ("x", object)]
    )
    def test_setitem_index_int64(self, val, exp_dtype):
        # 创建具有 np.int64 类型索引的 Series 对象
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64

        # 创建预期的索引
        exp_index = pd.Index([0, 1, 2, 3, val])
        # 调用 _assert_setitem_index_conversion 方法进行设置索引测试
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.float64), (5.1, np.float64), ("x", object)]
    )
    def test_setitem_index_float64(self, val, exp_dtype, request):
        # 创建具有 np.float64 类型索引的 Series 对象
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64

        # 创建预期的索引
        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        # 调用 _assert_setitem_index_conversion 方法进行设置索引测试
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_series_period(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64tz(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_period(self):
        raise NotImplementedError
class TestInsertIndexCoercion(CoercionBase):
    # 定义测试类 TestInsertIndexCoercion，继承自 CoercionBase
    klasses = ["index"]
    # 定义类属性 klasses，值为包含字符串 "index" 的列表
    method = "insert"
    # 定义类属性 method，其值为字符串 "insert"

    def _assert_insert_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by insert"""
        # 定义私有方法 _assert_insert_conversion，用于测试插入操作触发的类型转换
        target = original.copy()
        # 复制传入的原始对象 original
        res = target.insert(1, value)
        # 在复制后的对象 target 的索引 1 处插入 value 值，并返回结果
        tm.assert_index_equal(res, expected)
        # 使用测试模块 tm 来断言 res 与期望的索引 expected 相等
        assert res.dtype == expected_dtype
        # 断言 res 的数据类型等于期望的数据类型 expected_dtype

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, object),
            (1.1, 1.1, object),
            (False, False, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_object(self, insert, coerced_val, coerced_dtype):
        # 定义测试方法 test_insert_index_object，用于测试对象索引插入时的类型转换
        obj = pd.Index(list("abcd"), dtype=object)
        # 创建一个包含字符 'abcd' 的对象索引 obj，数据类型为 object
        assert obj.dtype == object
        # 断言 obj 的数据类型为 object

        exp = pd.Index(["a", coerced_val, "b", "c", "d"], dtype=object)
        # 创建一个期望的对象索引 exp，插入 coerced_val 到索引位置，数据类型为 object
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
        # 调用 _assert_insert_conversion 方法，测试插入操作的类型转换是否符合预期

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, None),
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_int_index(
        self, any_int_numpy_dtype, insert, coerced_val, coerced_dtype
    ):
        # 定义测试方法 test_insert_int_index，测试整数索引插入时的类型转换
        dtype = any_int_numpy_dtype
        # 从参数 any_int_numpy_dtype 获取数据类型 dtype
        obj = pd.Index([1, 2, 3, 4], dtype=dtype)
        # 创建一个整数对象索引 obj，数据类型为 dtype
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        # 如果 coerced_dtype 不为 None，则使用其值；否则使用 dtype

        exp = pd.Index([1, coerced_val, 2, 3, 4], dtype=coerced_dtype)
        # 创建一个期望的对象索引 exp，插入 coerced_val 到索引位置，数据类型为 coerced_dtype
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
        # 调用 _assert_insert_conversion 方法，测试插入操作的类型转换是否符合预期

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1.0, None),
            # 当 float_numpy_dtype=float32 时，这不是情况
            # 请参见下面的更正
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_float_index(
        self, float_numpy_dtype, insert, coerced_val, coerced_dtype
    ):
        # 定义测试方法 test_insert_float_index，测试浮点数索引插入时的类型转换
        dtype = float_numpy_dtype
        # 从参数 float_numpy_dtype 获取数据类型 dtype
        obj = pd.Index([1.0, 2.0, 3.0, 4.0], dtype=dtype)
        # 创建一个浮点数对象索引 obj，数据类型为 dtype
        coerced_dtype = coerced_dtype if coerced_dtype is not None else dtype
        # 如果 coerced_dtype 不为 None，则使用其值；否则使用 dtype

        if np_version_gt2 and dtype == "float32" and coerced_val == 1.1:
            # 如果 np_version_gt2 为真，并且 dtype 是 "float32"，coerced_val 是 1.1
            # 在第二个测试用例中，由于 1.1 可以无损地转换为 float32
            # 如果原始 dtype 是 float32，则期望的数据类型将是 float32
            coerced_dtype = np.float32
        exp = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0], dtype=coerced_dtype)
        # 创建一个期望的对象索引 exp，插入 coerced_val 到索引位置，数据类型为 coerced_dtype
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)
        # 调用 _assert_insert_conversion 方法，测试插入操作的类型转换是否符合预期

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    # 使用 pytest 的 parametrize 装饰器，为 test_insert_index_datetimes 方法定义多组参数化测试
    @pytest.mark.parametrize(
        "insert_value",
        [pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01", tz="Asia/Tokyo"), 1],
    )
    # 测试插入日期时间索引的方法，验证索引数据类型与预期类型是否一致
    def test_insert_index_datetimes(self, fill_val, exp_dtype, insert_value):
        # 创建一个日期时间索引对象 obj，基于指定的填充值时区，并转换为纳秒单位
        obj = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], tz=fill_val.tz
        ).as_unit("ns")
        assert obj.dtype == exp_dtype

        # 创建预期的日期时间索引 exp，包括插入值 fill_val.date()，并转换为纳秒单位
        exp = pd.DatetimeIndex(
            ["2011-01-01", fill_val.date(), "2011-01-02", "2011-01-03", "2011-01-04"],
            tz=fill_val.tz,
        ).as_unit("ns")
        # 调用 _assert_insert_conversion 方法验证插入操作的结果是否符合预期
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)

        if fill_val.tz:
            # 如果填充值有时区信息
            # 创建一个指定时刻的时间戳 ts
            ts = pd.Timestamp("2012-01-01")
            # 执行插入操作，返回结果 result
            result = obj.insert(1, ts)
            # 预期结果是将 obj 转换为对象类型后再插入 ts
            expected = obj.astype(object).insert(1, ts)
            # 验证预期结果的数据类型为对象类型
            assert expected.dtype == object
            # 使用 assert_index_equal 方法验证 result 和 expected 是否相等
            tm.assert_index_equal(result, expected)

            # 创建一个带时区的时间戳 ts
            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo")
            # 执行插入操作，返回结果 result
            result = obj.insert(1, ts)
            # 一旦弃用生效：预期结果是在插入时将 ts 转换为 obj 的数据类型时区
            expected = obj.insert(1, ts.tz_convert(obj.dtype.tz))
            # 验证预期结果的数据类型与 obj 的数据类型相同
            assert expected.dtype == obj.dtype
            # 使用 assert_index_equal 方法验证 result 和 expected 是否相等
            tm.assert_index_equal(result, expected)

        else:
            # 如果填充值没有时区信息
            # 创建一个带时区的时间戳 ts
            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo")
            # 执行插入操作，返回结果 result
            result = obj.insert(1, ts)
            # 预期结果是将 obj 转换为对象类型后再插入 ts
            expected = obj.astype(object).insert(1, ts)
            # 验证预期结果的数据类型为对象类型
            assert expected.dtype == object
            # 使用 assert_index_equal 方法验证 result 和 expected 是否相等
            tm.assert_index_equal(result, expected)

        # 创建一个普通的对象 item
        item = 1
        # 执行插入操作，返回结果 result
        result = obj.insert(1, item)
        # 预期结果是将 obj 转换为对象类型后再插入 item
        expected = obj.astype(object).insert(1, item)
        # 验证预期结果的第一个元素是否等于 item
        assert expected[1] == item
        # 验证预期结果的数据类型为对象类型
        assert expected.dtype == object
        # 使用 assert_index_equal 方法验证 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)

    # 测试插入时间增量索引的方法
    def test_insert_index_timedelta64(self):
        # 创建一个时间增量索引对象 obj，包括多个时间增量字符串
        obj = pd.TimedeltaIndex(["1 day", "2 day", "3 day", "4 day"])
        # 验证 obj 的数据类型是否为 "timedelta64[ns]"
        assert obj.dtype == "timedelta64[ns]"

        # 创建预期的时间增量索引 exp，包括插入值 pd.Timedelta("10 day")
        exp = pd.TimedeltaIndex(["1 day", "10 day", "2 day", "3 day", "4 day"])
        # 调用 _assert_insert_conversion 方法验证插入操作的结果是否符合预期
        self._assert_insert_conversion(
            obj, pd.Timedelta("10 day"), exp, "timedelta64[ns]"
        )

        # 遍历插入值列表，每次将一个值插入 obj
        for item in [pd.Timestamp("2012-01-01"), 1]:
            # 执行插入操作，返回结果 result
            result = obj.insert(1, item)
            # 预期结果是将 obj 转换为对象类型后再插入 item
            expected = obj.astype(object).insert(1, item)
            # 验证预期结果的数据类型为对象类型
            assert expected.dtype == object
            # 使用 assert_index_equal 方法验证 result 和 expected 是否相等
            tm.assert_index_equal(result, expected)

    # 使用 pytest 的 parametrize 装饰器，为多种插入值定义参数化测试
    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            # 插入 pd.Period 对象，预期的值为 "2012-01"，数据类型为 "period[M]"
            (pd.Period("2012-01", freq="M"), "2012-01", "period[M]"),
            # 插入 pd.Timestamp 对象，预期的值为 pd.Timestamp("2012-01-01")，数据类型为 object
            (pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01"), object),
            # 插入整数 1，预期的值为 1，数据类型为 object
            (1, 1, object),
            # 插入字符串 "x"，预期的值为 "x"，数据类型为 object
            ("x", "x", object),
        ],
    )
    # 定义测试方法，用于测试插入期间索引
    def test_insert_index_period(self, insert, coerced_val, coerced_dtype):
        # 创建一个 PeriodIndex 对象，包含多个月份的周期索引
        obj = pd.PeriodIndex(["2011-01", "2011-02", "2011-03", "2011-04"], freq="M")
        # 断言对象的数据类型为 "period[M]"
        assert obj.dtype == "period[M]"

        # 准备数据列表，包括 Period 对象和其他被强制转换的值
        data = [
            pd.Period("2011-01", freq="M"),
            coerced_val,
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        # 如果插入的对象是 Period 类型
        if isinstance(insert, pd.Period):
            # 期望的结果是一个新的 PeriodIndex 对象
            exp = pd.PeriodIndex(data, freq="M")
            # 调用内部方法，断言插入操作后的转换结果符合预期
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

            # 对可以解析为合适 PeriodDtype 的字符串进行插入操作
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)

        else:
            # 执行插入操作，返回插入后的新对象
            result = obj.insert(0, insert)
            # 期望的结果是将对象转换为对象类型后执行插入操作
            expected = obj.astype(object).insert(0, insert)
            # 断言插入后的索引对象相等
            tm.assert_index_equal(result, expected)

            # TODO: ATM inserting '2012-01-01 00:00:00' when we have obj.freq=="M"
            #  casts that string to Period[M], not clear that is desirable
            # 如果插入的对象不是 Timestamp 类型
            if not isinstance(insert, pd.Timestamp):
                # 尝试插入一个不可强制转换的字符串
                result = obj.insert(0, str(insert))
                # 期望的结果是将对象转换为对象类型后执行字符串插入操作
                expected = obj.astype(object).insert(0, str(insert))
                # 断言插入后的索引对象相等
                tm.assert_index_equal(result, expected)

    # 标记为预期失败的测试方法，暂未实现
    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_complex128(self):
        raise NotImplementedError

    # 标记为预期失败的测试方法，暂未实现
    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_bool(self):
        raise NotImplementedError
# 定义一个测试类 TestWhereCoercion，继承自 CoercionBase
class TestWhereCoercion(CoercionBase):
    # 类变量 method 表示方法名为 "where"
    method = "where"
    # 类变量 _cond 是一个包含布尔值的 NumPy 数组
    _cond = np.array([True, False, True, False])

    # 定义一个测试方法 _assert_where_conversion，用于验证 where 方法的强制类型转换
    def _assert_where_conversion(
        self, original, cond, values, expected, expected_dtype
    ):
        """test coercion triggered by where"""
        # 复制传入的对象 original
        target = original.copy()
        # 调用 where 方法进行条件运算
        res = target.where(cond, values)
        # 使用测试工具函数 tm.assert_equal 验证结果 res 是否等于预期值 expected
        tm.assert_equal(res, expected)
        # 使用 assert 语句验证结果 res 的数据类型是否等于预期数据类型 expected_dtype
        assert res.dtype == expected_dtype

    # 定义一个辅助方法 _construct_exp，根据不同的 fill_val 构建预期值
    def _construct_exp(self, obj, klass, fill_val, exp_dtype):
        # 根据 fill_val 的类型选择不同的 values 值
        if fill_val is True:
            values = klass([True, False, True, True])
        elif isinstance(fill_val, (datetime, np.datetime64)):
            values = pd.date_range(fill_val, periods=4)
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])

        # 使用 klass 类构建一个新的对象 exp，其包含 obj 的部分元素和 values 的部分元素
        exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
        return values, exp

    # 定义一个测试方法 _run_test，执行测试条件下的 where 方法调用
    def _run_test(self, obj, fill_val, klass, exp_dtype):
        # 使用类变量 _cond 构建条件 cond
        cond = klass(self._cond)

        # 根据 fill_val 构建预期的结果 exp
        exp = klass([obj[0], fill_val, obj[2], fill_val], dtype=exp_dtype)
        # 调用 _assert_where_conversion 方法验证 where 方法的结果
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        # 使用 _construct_exp 方法构建 values 和新的预期结果 exp
        values, exp = self._construct_exp(obj, klass, fill_val, exp_dtype)
        # 再次调用 _assert_where_conversion 方法验证 where 方法的结果
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    # 定义一个参数化测试方法 test_where_object，测试对象类型为 object
    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_where_object(self, index_or_series, fill_val, exp_dtype):
        # 获取测试对象的类 klass
        klass = index_or_series
        # 构建一个包含字符的对象 obj
        obj = klass(list("abcd"), dtype=object)
        # 使用 assert 语句验证 obj 的数据类型是否为 object
        assert obj.dtype == object
        # 调用 _run_test 方法执行测试
        self._run_test(obj, fill_val, klass, exp_dtype)

    # 定义一个参数化测试方法 test_where_int64，测试对象类型为 np.int64
    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_int64(self, index_or_series, fill_val, exp_dtype, request):
        # 获取测试对象的类 klass
        klass = index_or_series

        # 构建一个包含整数的对象 obj
        obj = klass([1, 2, 3, 4])
        # 使用 assert 语句验证 obj 的数据类型是否为 np.int64
        assert obj.dtype == np.int64
        # 调用 _run_test 方法执行测试
        self._run_test(obj, fill_val, klass, exp_dtype)

    # 定义一个参数化测试方法 test_where_float64，测试对象类型为 np.float64
    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_float64(self, index_or_series, fill_val, exp_dtype, request):
        # 获取测试对象的类 klass
        klass = index_or_series

        # 构建一个包含浮点数的对象 obj
        obj = klass([1.1, 2.2, 3.3, 4.4])
        # 使用 assert 语句验证 obj 的数据类型是否为 np.float64
        assert obj.dtype == np.float64
        # 调用 _run_test 方法执行测试
        self._run_test(obj, fill_val, klass, exp_dtype)

    # 定义一个参数化测试方法 test_where_complex128，测试对象类型为 np.complex128
    @pytest.mark.parametrize(
        "fill_val, exp_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_where_complex128(self, index_or_series, fill_val, exp_dtype):
        # 获取测试对象的类 klass
        klass = index_or_series
        # 构建一个包含复数的对象 obj
        obj = klass([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        # 使用 assert 语句验证 obj 的数据类型是否为 np.complex128
        assert obj.dtype == np.complex128
        # 调用 _run_test 方法执行测试
        self._run_test(obj, fill_val, klass, exp_dtype)
    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_where_series_bool 提供多组参数组合
    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)],
    )
    # 定义测试函数 test_where_series_bool，参数 index_or_series 作为测试数据来源
    def test_where_series_bool(self, index_or_series, fill_val, exp_dtype):
        # 从参数 index_or_series 中获取类别信息
        klass = index_or_series

        # 创建一个包含布尔值的实例对象 obj
        obj = klass([True, False, True, False])
        # 断言 obj 对象的数据类型为 np.bool_
        assert obj.dtype == np.bool_
        # 调用内部方法 _run_test 执行测试
        self._run_test(obj, fill_val, klass, exp_dtype)

    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_where_datetime64 提供多组参数组合
    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), object),
        ],
        # 为参数组合指定标识符，便于识别不同的测试场景
        ids=["datetime64", "datetime64tz"],
    )
    # 定义测试函数 test_where_datetime64，参数 index_or_series 作为测试数据来源
    def test_where_datetime64(self, index_or_series, fill_val, exp_dtype):
        # 从参数 index_or_series 中获取类别信息
        klass = index_or_series

        # 创建一个包含日期范围的实例对象 obj
        obj = klass(pd.date_range("2011-01-01", periods=4, freq="D")._with_freq(None))
        # 断言 obj 对象的数据类型为 "datetime64[ns]"
        assert obj.dtype == "datetime64[ns]"

        # 将 fill_val 赋值给 fv
        fv = fill_val
        # 根据 exp_dtype 的不同情况，选择不同的时间标量进行测试
        if exp_dtype == "datetime64[ns]":
            # 如果 exp_dtype 是 "datetime64[ns]"，则使用每个可用的时间标量进行测试
            for scalar in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
                self._run_test(obj, scalar, klass, exp_dtype)
        else:
            # 如果 exp_dtype 不是 "datetime64[ns]"，则使用 fv 和其转换后的时间标量进行测试
            for scalar in [fv, fv.to_pydatetime()]:
                self._run_test(obj, fill_val, klass, exp_dtype)

    # 使用 pytest 的 @xfail 装饰器标记测试为预期失败，并给出失败原因
    @pytest.mark.xfail(reason="Test not implemented")
    # 定义测试函数 test_where_index_complex128，未实现具体的测试逻辑
    def test_where_index_complex128(self):
        raise NotImplementedError

    # 使用 pytest 的 @xfail 装饰器标记测试为预期失败，并给出失败原因
    @pytest.mark.xfail(reason="Test not implemented")
    # 定义测试函数 test_where_index_bool，未实现具体的测试逻辑
    def test_where_index_bool(self):
        raise NotImplementedError

    # 使用 pytest 的 @xfail 装饰器标记测试为预期失败，并给出失败原因
    @pytest.mark.xfail(reason="Test not implemented")
    # 定义测试函数 test_where_series_timedelta64，未实现具体的测试逻辑
    def test_where_series_timedelta64(self):
        raise NotImplementedError

    # 使用 pytest 的 @xfail 装饰器标记测试为预期失败，并给出失败原因
    @pytest.mark.xfail(reason="Test not implemented")
    # 定义测试函数 test_where_series_period，未实现具体的测试逻辑
    def test_where_series_period(self):
        raise NotImplementedError

    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_where_index_timedelta64 提供多组参数组合
    @pytest.mark.parametrize(
        "value", [pd.Timedelta(days=9), timedelta(days=9), np.timedelta64(9, "D")]
    )
    # 定义测试函数 test_where_index_timedelta64，参数 value 作为测试数据来源
    def test_where_index_timedelta64(self, value):
        # 创建一个时间间隔对象 tdi
        tdi = pd.timedelta_range("1 Day", periods=4)
        # 创建一个布尔条件数组 cond
        cond = np.array([True, False, False, True])

        # 创建预期的时间间隔索引对象 expected
        expected = pd.TimedeltaIndex(["1 Day", value, value, "4 Days"])
        # 使用 tdi 对象的 where 方法进行条件筛选，并将结果与 expected 进行比较
        result = tdi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # 创建错误数据类型的 NaT 对象 dtnat
        dtnat = np.datetime64("NaT", "ns")
        # 创建预期的索引对象 expected，包含了 dtnat 对象
        expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
        # 断言 expected[1] 是 dtnat 对象
        assert expected[1] is dtnat

        # 使用 tdi 对象的 where 方法进行条件筛选，并将结果与 expected 进行比较
        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)
    # 定义一个测试函数，测试 Pandas 时间索引的 where 方法的不同用例

    # 创建一个日期时间索引，从 "2016-01-01" 开始，每季度开始（"QS"）的前三个周期
    dti = pd.date_range("2016-01-01", periods=3, freq="QS")
    
    # 将日期时间索引转换为季度周期索引
    pi = dti.to_period("Q")
    
    # 创建一个包含布尔值的 NumPy 数组作为条件
    cond = np.array([False, True, False])

    # 使用 where 方法，传入一个有效的标量值
    value = pi[-1] + pi.freq * 10
    expected = pd.PeriodIndex([value, pi[1], value])
    result = pi.where(cond, value)
    tm.assert_index_equal(result, expected)

    # 测试传入一个由 Period 对象组成的 ndarray[object] 的情况
    other = np.asarray(pi + pi.freq * 10, dtype=object)
    result = pi.where(cond, other)
    expected = pd.PeriodIndex([other[0], pi[1], other[2]])
    tm.assert_index_equal(result, expected)

    # 测试传入一个类型不匹配的标量 -> 自动转换为 object 类型
    td = pd.Timedelta(days=4)
    expected = pd.Index([td, pi[1], td], dtype=object)
    result = pi.where(cond, td)
    tm.assert_index_equal(result, expected)

    # 测试传入一个 Period 对象作为替换值
    per = pd.Period("2020-04-21", "D")
    expected = pd.Index([per, pi[1], per], dtype=object)
    result = pi.where(cond, per)
    tm.assert_index_equal(result, expected)
# 定义一个测试类 TestFillnaSeriesCoercion，继承自 CoercionBase 类
class TestFillnaSeriesCoercion(CoercionBase):
    # 这是一个类级别的属性，指定方法名称为 "fillna"
    method = "fillna"

    # 标记此测试为预期失败，原因是测试尚未实现
    @pytest.mark.xfail(reason="Test not implemented")
    def test_has_comprehensive_tests(self):
        # 抛出未实现错误，表示测试尚未完成
        raise NotImplementedError

    # 定义一个私有方法，用于断言 fillna 方法的转换结果
    def _assert_fillna_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by fillna"""
        # 复制原始对象，填充缺失值，生成结果对象
        target = original.copy()
        res = target.fillna(value)
        # 使用测试工具断言结果对象与预期对象相等
        tm.assert_equal(res, expected)
        # 断言结果对象的数据类型与预期数据类型相等
        assert res.dtype == expected_dtype

    # 使用参数化测试，测试填充缺失值为对象类型的情况
    @pytest.mark.parametrize(
        "fill_val, fill_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_fillna_object(self, index_or_series, fill_val, fill_dtype):
        # 获取测试对象类型
        klass = index_or_series
        # 创建包含 NaN 的对象数组，数据类型为 object
        obj = klass(["a", np.nan, "c", "d"], dtype=object)
        # 断言对象数组的数据类型为 object
        assert obj.dtype == object

        # 创建预期的对象数组，填充 NaN 值
        exp = klass(["a", fill_val, "c", "d"], dtype=object)
        # 调用断言填充函数，验证填充结果
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    # 使用参数化测试，测试填充缺失值为 np.float64 类型的情况
    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_fillna_float64(self, index_or_series, fill_val, fill_dtype):
        # 获取测试对象类型
        klass = index_or_series
        # 创建包含 NaN 的对象数组，数据类型为 np.float64
        obj = klass([1.1, np.nan, 3.3, 4.4])
        # 断言对象数组的数据类型为 np.float64
        assert obj.dtype == np.float64

        # 创建预期的对象数组，填充 NaN 值
        exp = klass([1.1, fill_val, 3.3, 4.4])
        # 调用断言填充函数，验证填充结果
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    # 使用参数化测试，测试填充缺失值为 np.complex128 类型的情况
    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_fillna_complex128(self, index_or_series, fill_val, fill_dtype):
        # 获取测试对象类型
        klass = index_or_series
        # 创建包含 NaN 的对象数组，数据类型为 np.complex128
        obj = klass([1 + 1j, np.nan, 3 + 3j, 4 + 4j], dtype=np.complex128)
        # 断言对象数组的数据类型为 np.complex128
        assert obj.dtype == np.complex128

        # 创建预期的对象数组，填充 NaN 值
        exp = klass([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
        # 调用断言填充函数，验证填充结果
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    # 使用参数化测试，测试填充缺失值为日期时间类型的情况
    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[s]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), object),
            (1, object),
            ("x", object),
        ],
        ids=["datetime64", "datetime64tz", "object", "object"],
    )
    # 定义一个测试函数，用于测试填充缺失值为日期时间类型的情况
    def test_fillna_datetime(self, index_or_series, fill_val, fill_dtype):
        # 获取传入参数的类
        klass = index_or_series
        # 创建一个对象，包含四个日期时间数据，其中一个为缺失值
        obj = klass(
            [
                pd.Timestamp("2011-01-01"),
                pd.NaT,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        # 断言对象的数据类型为 "datetime64[s]"
        assert obj.dtype == "datetime64[s]"

        # 创建期望的对象，用于比较填充后的结果
        exp = klass(
            [
                pd.Timestamp("2011-01-01"),
                fill_val,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        # 调用辅助方法，验证填充操作后的转换
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    # 使用 pytest 的 parametrize 装饰器，对不同的填充值和填充数据类型进行参数化测试
    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[s, US/Eastern]"),
            (pd.Timestamp("2012-01-01"), object),
            # 在 2.0 之前，由于时区不匹配，我们会得到对象结果
            (pd.Timestamp("2012-01-01", tz="Asia/Tokyo"), "datetime64[s, US/Eastern]"),
            (1, object),
            ("x", object),
        ],
    )
    # 定义测试填充日期时间带时区的情况
    def test_fillna_datetime64tz(self, index_or_series, fill_val, fill_dtype):
        # 获取传入参数的类
        klass = index_or_series
        # 设置时区
        tz = "US/Eastern"

        # 创建对象，包含四个带时区的日期时间数据，其中一个为缺失值
        obj = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.NaT,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        # 断言对象的数据类型为 "datetime64[s, US/Eastern]"
        assert obj.dtype == "datetime64[s, US/Eastern]"

        # 根据填充值是否带有时区信息，进行时区的转换或直接赋值
        if getattr(fill_val, "tz", None) is None:
            fv = fill_val
        else:
            fv = fill_val.tz_convert(tz)
        
        # 创建期望的对象，用于比较填充后的结果
        exp = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                fv,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        # 调用辅助方法，验证填充操作后的转换
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    # 使用 pytest 的 parametrize 装饰器，对不同的填充值进行参数化测试
    @pytest.mark.parametrize(
        "fill_val",
        [
            1,
            1.1,
            1 + 1j,
            True,
            pd.Interval(1, 2, closed="left"),
            pd.Timestamp("2012-01-01", tz="US/Eastern"),
            pd.Timestamp("2012-01-01"),
            pd.Timedelta(days=1),
            pd.Period("2016-01-01", "D"),
        ],
    )
    # 定义测试填充间隔的情况
    def test_fillna_interval(self, index_or_series, fill_val):
        # 创建间隔范围对象，包含 NaN 值
        ii = pd.interval_range(1.0, 5.0, closed="right").insert(1, np.nan)
        # 断言对象的数据类型为 pd.IntervalDtype
        assert isinstance(ii.dtype, pd.IntervalDtype)
        # 使用传入参数创建对象
        obj = index_or_series(ii)

        # 创建期望的对象，用于比较填充后的结果，数据类型为 object
        exp = index_or_series([ii[0], fill_val, ii[2], ii[3], ii[4]], dtype=object)

        # 填充数据类型为 object
        fill_dtype = object
        # 调用辅助方法，验证填充操作后的转换
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    # 使用 xfail 标记，声明该测试尚未实现
    @pytest.mark.xfail(reason="Test not implemented")
    # 声明测试填充 Series 类型的 int64 数据的情况
    def test_fillna_series_int64(self):
        # 抛出 NotImplementedError 异常
        raise NotImplementedError

    # 使用 xfail 标记，声明该测试尚未实现
    @pytest.mark.xfail(reason="Test not implemented")
    # 声明测试填充索引类型的 int64 数据的情况
    def test_fillna_index_int64(self):
        # 抛出 NotImplementedError 异常
        raise NotImplementedError
    # 声明一个测试函数，用于测试 Series 类型数据的填充操作（测试预期会失败）
    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_bool(self):
        # 抛出未实现错误，表示测试尚未实现
        raise NotImplementedError
    
    # 声明一个测试函数，用于测试索引类型数据的布尔填充操作（测试预期会失败）
    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_bool(self):
        # 抛出未实现错误，表示测试尚未实现
        raise NotImplementedError
    
    # 声明一个测试函数，用于测试 Series 类型数据的 timedelta64 填充操作（测试预期会失败）
    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_timedelta64(self):
        # 抛出未实现错误，表示测试尚未实现
        raise NotImplementedError
    
    # 声明一个参数化测试函数，用于对填充操作进行多种值的测试
    @pytest.mark.parametrize(
        "fill_val",
        [
            1,  # 整数值
            1.1,  # 浮点数值
            1 + 1j,  # 复数值
            True,  # 布尔值
            pd.Interval(1, 2, closed="left"),  # 区间对象
            pd.Timestamp("2012-01-01", tz="US/Eastern"),  # 带时区的时间戳
            pd.Timestamp("2012-01-01"),  # 时间戳
            pd.Timedelta(days=1),  # 时间增量
            pd.Period("2016-01-01", "W"),  # 时间段对象
        ],
    )
    def test_fillna_series_period(self, index_or_series, fill_val):
        # 创建一个包含 NaN 值的周期范围对象
        pi = pd.period_range("2016-01-01", periods=4, freq="D").insert(1, pd.NaT)
        # 断言周期数据类型为 PeriodDtype
        assert isinstance(pi.dtype, pd.PeriodDtype)
        # 根据输入的索引或者系列对象创建相应的对象
        obj = index_or_series(pi)
    
        # 创建一个期望的填充后的对象，其中 NaN 填充为 fill_val
        exp = index_or_series([pi[0], fill_val, pi[2], pi[3], pi[4]], dtype=object)
    
        # 填充的数据类型为对象类型
        fill_dtype = object
        # 调用内部函数来断言填充后的转换
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)
    
    # 声明一个测试函数，用于测试索引类型数据的 timedelta64 填充操作（测试预期会失败）
    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_timedelta64(self):
        # 抛出未实现错误，表示测试尚未实现
        raise NotImplementedError
    
    # 声明一个测试函数，用于测试索引类型数据的周期填充操作（测试预期会失败）
    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_period(self):
        # 抛出未实现错误，表示测试尚未实现
        raise NotImplementedError
class TestReplaceSeriesCoercion(CoercionBase):
    # 定义要测试的数据类型
    klasses = ["series"]
    # 定义要测试的方法
    method = "replace"

    # 初始化替换字典
    rep: dict[str, list] = {}
    # 设置替换规则：将 "object" 类型替换为 ["a", "b"]
    rep["object"] = ["a", "b"]
    # 设置替换规则：将 "int64" 类型替换为 [4, 5]
    rep["int64"] = [4, 5]
    # 设置替换规则：将 "float64" 类型替换为 [1.1, 2.2]
    rep["float64"] = [1.1, 2.2]
    # 设置替换规则：将 "complex128" 类型替换为 [1+1j, 2+2j]
    rep["complex128"] = [1 + 1j, 2 + 2j]
    # 设置替换规则：将 "bool" 类型替换为 [True, False]
    rep["bool"] = [True, False]
    # 设置替换规则：将 "datetime64[ns]" 类型替换为日期时间戳列表
    rep["datetime64[ns]"] = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-03")]

    # 遍历时区列表，为 "datetime64[ns, {tz}]" 类型设置替换规则
    for tz in ["UTC", "US/Eastern"]:
        key = f"datetime64[ns, {tz}]"
        rep[key] = [
            pd.Timestamp("2011-01-01", tz=tz),
            pd.Timestamp("2011-01-03", tz=tz),
        ]

    # 设置替换规则：将 "timedelta64[ns]" 类型替换为时间增量列表
    rep["timedelta64[ns]"] = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]

    @pytest.fixture(params=["dict", "series"])
    def how(self, request):
        # 返回测试用例参数化的参数
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        ]
    )
    def from_key(self, request):
        # 返回替换字典的键作为测试用例参数化的参数
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        ],
        ids=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64",
            "datetime64tz",
            "datetime64tz",
            "timedelta64",
        ],
    )
    def to_key(self, request):
        # 返回替换字典的键作为测试用例参数化的参数，同时指定其 ID
        return request.param

    @pytest.fixture
    def replacer(self, how, from_key, to_key):
        """
        Object we will pass to `Series.replace`
        """
        # 根据传入的 how 参数选择不同的替换方式
        if how == "dict":
            # 如果 how 为 "dict"，则返回以 from_key 和 to_key 为键值对的字典
            replacer = dict(zip(self.rep[from_key], self.rep[to_key]))
        elif how == "series":
            # 如果 how 为 "series"，则返回将 to_key 的值作为索引与 from_key 的值构成的 Series 对象
            replacer = pd.Series(self.rep[to_key], index=self.rep[from_key])
        else:
            # 如果 how 参数既不是 "dict" 也不是 "series"，则抛出数值错误异常
            raise ValueError
        return replacer

    # 标记此测试用例为跳过状态，若使用了 pyarrow 字符串数据类型，则跳过执行，并附加原因说明
    @pytest.mark.skipif(using_pyarrow_string_dtype(), reason="TODO: test is to complex")
    # 定义测试方法，用于替换 Series 对象的值，包括如何替换、目标键、源键以及替换规则
    def test_replace_series(self, how, to_key, from_key, replacer):
        # 创建索引为 [3, 4]，名称为 "xxx" 的索引对象
        index = pd.Index([3, 4], name="xxx")
        # 使用 self.rep 中的 from_key 对应的值创建一个 Series 对象，索引为 index，名称为 "yyy"
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        # 将 Series 对象转换为指定的数据类型 from_key
        obj = obj.astype(from_key)
        # 断言 obj 的数据类型为 from_key
        assert obj.dtype == from_key

        # 如果源键和目标键都以 "datetime" 开头，则跳过以下测试
        if from_key.startswith("datetime") and to_key.startswith("datetime"):
            # tested below
            return
        # 如果源键在指定列表中，则跳过以下测试
        elif from_key in ["datetime64[ns, US/Eastern]", "datetime64[ns, UTC]"]:
            # tested below
            return

        # 如果条件满足，则根据不同情况跳过测试
        if (from_key == "float64" and to_key in ("int64")) or (
            from_key == "complex128" and to_key in ("int64", "float64")
        ):
            # 如果不是 64 位系统或者是 Windows 平台，则跳过测试，并附带跳过的原因
            if not IS64 or is_platform_windows():
                pytest.skip(f"32-bit platform buggy: {from_key} -> {to_key}")

            # 预期结果：不通过替换降低数据类型
            exp = pd.Series(self.rep[to_key], index=index, name="yyy", dtype=from_key)

        else:
            # 根据 to_key 从 self.rep 中创建 Series 对象 exp，索引为 index，名称为 "yyy"
            exp = pd.Series(self.rep[to_key], index=index, name="yyy")

        # 对 obj 进行替换操作，得到结果 Series 对象 result
        result = obj.replace(replacer)
        # 使用 tm.assert_series_equal 函数断言 result 与 exp 相等，不检查数据类型
        tm.assert_series_equal(result, exp, check_dtype=False)

    # 使用参数化测试装饰器标记测试方法，测试替换 Series 对象的日期时间带时区的情况
    @pytest.mark.parametrize(
        "to_key",
        ["timedelta64[ns]", "bool", "object", "complex128", "float64", "int64"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key", ["datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"], indirect=True
    )
    def test_replace_series_datetime_tz(
        self, how, to_key, from_key, replacer, using_infer_string
    ):
        # 创建索引为 [3, 4]，名称为 "xyz" 的索引对象
        index = pd.Index([3, 4], name="xyz")
        # 使用 self.rep 中的 from_key 对应的值创建一个 Series 对象 obj，索引为 index，名称为 "yyy"，并将其转换为纳秒单位
        obj = pd.Series(self.rep[from_key], index=index, name="yyy").dt.as_unit("ns")
        # 断言 obj 的数据类型为 from_key
        assert obj.dtype == from_key

        # 根据 to_key 从 self.rep 中创建 Series 对象 exp，索引为 index，名称为 "yyy"
        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        # 如果使用 infer_string 并且 to_key 是 "object"，则断言 exp 的数据类型为 "string"，否则为 to_key
        if using_infer_string and to_key == "object":
            assert exp.dtype == "string"
        else:
            assert exp.dtype == to_key

        # 对 obj 进行替换操作，得到结果 Series 对象 result
        result = obj.replace(replacer)
        # 使用 tm.assert_series_equal 函数断言 result 与 exp 相等，不检查数据类型
        tm.assert_series_equal(result, exp, check_dtype=False)

    # 使用参数化测试装饰器标记测试方法，测试替换 Series 对象的日期时间的情况
    @pytest.mark.parametrize(
        "to_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    # 定义测试方法，用于测试替换 Series 中的日期时间数据
    def test_replace_series_datetime_datetime(self, how, to_key, from_key, replacer):
        # 创建索引对象，指定名称为 "xyz"
        index = pd.Index([3, 4], name="xyz")
        # 使用给定的数据和索引创建 Series 对象，并将其数据类型转换为纳秒单位的日期时间
        obj = pd.Series(self.rep[from_key], index=index, name="yyy").dt.as_unit("ns")
        # 断言 Series 对象的数据类型与参数 from_key 相同
        assert obj.dtype == from_key

        # 创建期望的 Series 对象，使用指定的数据和索引
        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        # 如果原始数据和期望数据的数据类型都是 pd.DatetimeTZDtype 类型，并且时区不匹配
        if isinstance(obj.dtype, pd.DatetimeTZDtype) and isinstance(
            exp.dtype, pd.DatetimeTZDtype
        ):
            # 保持原始数据的数据类型不变，如版本 2.0 之前的行为
            exp = exp.astype(obj.dtype)
        # 如果目标数据类型与原始数据类型相同
        elif to_key == from_key:
            # 将期望数据的日期时间单位转换为纳秒
            exp = exp.dt.as_unit("ns")

        # 对 Series 对象执行替换操作
        result = obj.replace(replacer)
        # 使用断言库检查替换后的结果与期望的结果是否相等，忽略数据类型的检查
        tm.assert_series_equal(result, exp, check_dtype=False)

    # 标记此测试为预期失败，原因是测试尚未实现
    @pytest.mark.xfail(reason="Test not implemented")
    def test_replace_series_period(self):
        # 抛出 NotImplementedError 异常，表示该测试尚未实现
        raise NotImplementedError
```