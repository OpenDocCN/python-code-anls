# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_replace.py`

```
# 导入所需的模块和库
import re  # 导入正则表达式模块
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from pandas._config import using_pyarrow_string_dtype  # 导入pandas配置模块

import pandas as pd  # 导入pandas库，并简写为pd
import pandas._testing as tm  # 导入pandas测试模块
from pandas.core.arrays import IntervalArray  # 导入pandas核心数组模块IntervalArray

# 定义一个测试类TestSeriesReplace，用于测试pd.Series的替换功能
class TestSeriesReplace:
    
    # 测试替换操作，当用户显式传递value=None时，返回相应结果
    def test_replace_explicit_none(self):
        # 创建一个包含整数、空字符串的Series对象
        ser = pd.Series([0, 0, ""], dtype=object)
        # 对空字符串进行替换，期望替换为None
        result = ser.replace("", None)
        # 创建一个期望的Series对象，空字符串替换为None
        expected = pd.Series([0, 0, None], dtype=object)
        # 使用测试模块tm中的方法验证结果与期望是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个3x3的DataFrame对象，初始值为0，并将第2列转换为对象类型以避免隐式转换
        df = pd.DataFrame(np.zeros((3, 3))).astype({2: object})
        # 将第(2, 2)位置设置为空字符串
        df.iloc[2, 2] = ""
        # 对空字符串进行替换，期望替换为None
        result = df.replace("", None)
        # 创建一个期望的DataFrame对象，第(2, 2)位置的值应为None
        expected = pd.DataFrame(
            {
                0: np.zeros(3),
                1: np.zeros(3),
                2: np.array([0.0, 0.0, None], dtype=object),
            }
        )
        # 使用assert语句验证期望结果中第(2, 2)位置的值为None
        assert expected.iloc[2, 2] is None
        # 使用测试模块tm中的方法验证DataFrame对象的相等性
        tm.assert_frame_equal(result, expected)

        # 创建一个包含整数和字符串的Series对象
        ser = pd.Series([10, 20, 30, "a", "a", "b", "a"])
        # 对字符串'a'进行替换，期望替换为None
        result = ser.replace("a", None)
        # 创建一个期望的Series对象，字符串'a'替换为None
        expected = pd.Series([10, 20, 30, None, None, "b", None])
        # 使用assert语句验证期望结果中最后一个位置的值为None
        assert expected.iloc[-1] is None
        # 使用测试模块tm中的方法验证结果与期望是否相等
        tm.assert_series_equal(result, expected)

    # 测试替换操作，如果没有实际替换操作，则不应该降低数据类型
    def test_replace_noop_doesnt_downcast(self):
        # 创建一个包含None和时间戳的Series对象，数据类型为对象类型
        ser = pd.Series([None, None, pd.Timestamp("2021-12-16 17:31")], dtype=object)
        # 对None进行替换，不应有实际替换操作
        res = ser.replace({np.nan: None})  # 应该是一个空操作
        # 使用测试模块tm中的方法验证结果与原始Series对象是否相等
        tm.assert_series_equal(res, ser)
        # 使用assert语句验证结果的数据类型为对象类型
        assert res.dtype == object

        # 使用不同的调用方式进行相同的替换操作
        res = ser.replace(np.nan, None)
        # 使用测试模块tm中的方法验证结果与原始Series对象是否相等
        tm.assert_series_equal(res, ser)
        # 使用assert语句验证结果的数据类型为对象类型
        assert res.dtype == object
    # 定义测试方法，用于测试替换功能
    def test_replace(self):
        # 设置数据数量 N
        N = 50
        # 创建一个包含 N 个随机标准正态分布值的 Pandas Series
        ser = pd.Series(np.random.default_rng(2).standard_normal(N))
        # 将前四个元素设为 NaN
        ser[0:4] = np.nan
        # 将第 6 到 9 个元素设为 0
        ser[6:10] = 0

        # 使用单个值 -1 替换列表中的 NaN，且在原地进行替换
        return_value = ser.replace([np.nan], -1, inplace=True)
        # 断言替换操作返回 None
        assert return_value is None

        # 用 -1 填充 NaN 值后的期望结果
        exp = ser.fillna(-1)
        # 断言替换后的 Series 等于填充后的期望结果
        tm.assert_series_equal(ser, exp)

        # 将所有值为 0.0 替换为 NaN，并与手动设置的操作等效
        rs = ser.replace(0.0, np.nan)
        ser[ser == 0.0] = np.nan
        tm.assert_series_equal(rs, ser)

        # 创建一个包含 N 个非负随机标准正态分布值的 Pandas Series，附带日期索引
        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        # 将前五个元素设为 NaN
        ser[:5] = np.nan
        # 将第 6 到 9 个元素设为 "foo"
        ser[6:10] = "foo"
        # 将第 20 到 29 个元素设为 "bar"

        # 使用单个值 -1 替换列表中的 NaN、"foo" 和 "bar"
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        # 断言前五个元素都被替换为 -1
        assert (rs[:5] == -1).all()
        # 断言第 6 到 9 个元素都被替换为 -1
        assert (rs[6:10] == -1).all()
        # 断言第 20 到 29 个元素都被替换为 -1
        assert (rs[20:30] == -1).all()
        # 断言原始 Series 中前五个元素仍为 NaN
        assert (pd.isna(ser[:5])).all()

        # 使用不同的值 -1、-2 和 -3 替换列表中的 NaN、"foo" 和 "bar"
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        # 断言前五个元素都被替换为 -1
        assert (rs[:5] == -1).all()
        # 断言第 6 到 9 个元素都被替换为 -2
        assert (rs[6:10] == -2).all()
        # 断言第 20 到 29 个元素都被替换为 -3
        assert (rs[20:30] == -3).all()
        # 断言原始 Series 中前五个元素仍为 NaN
        assert (pd.isna(ser[:5])).all()

        # 使用两个列表进行替换操作
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        # 断言 rs 和 rs2 的结果相同
        tm.assert_series_equal(rs, rs2)

        # 在原地进行替换操作
        return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        # 断言替换操作返回 None
        assert return_value is None

        # 断言前五个元素都被替换为 -1
        assert (ser[:5] == -1).all()
        # 断言第 6 到 9 个元素都被替换为 -1
        assert (ser[6:10] == -1).all()
        # 断言第 20 到 29 个元素都被替换为 -1
        assert (ser[20:30] == -1).all()

    # 测试将 NaN 替换为 inf 的功能
    def test_replace_nan_with_inf(self):
        # 创建包含 NaN、0 和 inf 的 Pandas Series
        ser = pd.Series([np.nan, 0, np.inf])
        # 断言将 NaN 替换为 0 后的 Series 与填充 0 的结果相同
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        # 创建包含 NaN、0、"foo"、"bar"、inf、None 和 NaT 的 Pandas Series
        ser = pd.Series([np.nan, 0, "foo", "bar", np.inf, None, pd.NaT])
        # 断言将 NaN 替换为 0 后的 Series 与填充 0 的结果相同
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        # 创建填充后的 Series 副本
        filled = ser.copy()
        filled[4] = 0
        # 断言将 inf 替换为 0 后的 Series 与填充 0 的结果相同
        tm.assert_series_equal(ser.replace(np.inf, 0), filled)

    # 测试将列表形式的值替换为列表形式的目标值的功能
    def test_replace_listlike_value_listlike_target(self, datetime_series):
        # 创建包含日期时间索引的 Pandas Series
        ser = pd.Series(datetime_series.index)
        # 断言将 NaN 替换为 0 后的 Series 与填充 0 的结果相同
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        # 测试异常情况，替换列表和目标列表的长度不匹配
        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3], [np.nan, 0])

        # 因为 ser 是 datetime64 类型，无法包含 1 或 2，因此此替换操作无效
        result = ser.replace([1, 2], [np.nan, 0])
        tm.assert_series_equal(result, ser)

        # 创建包含 0 到 4 的 Pandas Series
        ser = pd.Series([0, 1, 2, 3, 4])
        # 将 0 到 4 替换为 4 到 0
        result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        # 断言结果与期望的 Series 相同
        tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))
    # 测试函数，用于验证在 Series.replace 中替换 NaN 值时的异常处理
    def test_replace_gh5319(self):
        # API 变更自版本 0.12？
        # GitHub issue 5319
        # 创建一个包含 NaN 的 Pandas Series 对象
        ser = pd.Series([0, np.nan, 2, 3, 4])
        # 错误消息字符串，用于匹配异常信息
        msg = (
            "Series.replace must specify either 'value', "
            "a dict-like 'to_replace', or dict-like 'regex'"
        )
        # 验证是否抛出 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            ser.replace([np.nan])

        # 再次验证是否抛出 ValueError 异常，匹配相同错误消息
        with pytest.raises(ValueError, match=msg):
            ser.replace(np.nan)

    # 测试函数，用于验证在 Series.replace 中替换日期时间对象的功能
    def test_replace_datetime64(self):
        # GitHub issue 5797
        # 创建一个包含日期时间序列的 Pandas Series 对象
        ser = pd.Series(pd.date_range("20130101", periods=5))
        # 复制原始序列以备后用
        expected = ser.copy()
        # 将第二个日期替换为指定的新日期
        expected.loc[2] = pd.Timestamp("20120101")
        # 使用字典形式的替换操作，将指定日期替换为新日期
        result = ser.replace({pd.Timestamp("20130103"): pd.Timestamp("20120101")})
        # 验证替换后的结果是否与预期结果一致
        tm.assert_series_equal(result, expected)
        # 使用单一值替换操作，将指定日期替换为新日期
        result = ser.replace(pd.Timestamp("20130103"), pd.Timestamp("20120101"))
        # 再次验证替换后的结果是否与预期结果一致
        tm.assert_series_equal(result, expected)

    # 测试函数，用于验证在 Series.replace 中替换 NaT 对象的功能
    def test_replace_nat_with_tz(self):
        # GitHub issue 11792：测试在包含时区信息的列表中替换 NaT 对象
        # 创建一个带有时区信息的时间戳对象
        ts = pd.Timestamp("2015/01/01", tz="UTC")
        # 创建一个 Pandas Series 对象，其中包含 NaT 和具体时间戳对象
        s = pd.Series([pd.NaT, pd.Timestamp("2015/01/01", tz="UTC")])
        # 使用新值替换列表中的 NaT 和 NaN 对象
        result = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
        # 创建一个预期的 Pandas Series 对象，替换后的值应为指定的最小时间戳
        expected = pd.Series([pd.Timestamp.min, ts], dtype=object)
        # 验证替换操作后的结果是否与预期结果一致
        tm.assert_series_equal(expected, result)

    # 测试函数，用于验证在 Series.replace 中替换时间增量对象的功能
    def test_replace_timedelta_td64(self):
        # 创建一个时间增量序列
        tdi = pd.timedelta_range(0, periods=5)
        # 创建一个包含时间增量的 Pandas Series 对象
        ser = pd.Series(tdi)
        # 使用单一字典参数进行替换操作，将序列中的一个时间增量替换为另一个
        result = ser.replace({ser[1]: ser[3]})
        # 创建一个预期的 Pandas Series 对象，其中相应位置的增量已被替换
        expected = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
        # 验证替换操作后的结果是否与预期结果一致
        tm.assert_series_equal(result, expected)

    # 测试函数，用于验证在 Series.replace 中使用单一列表进行替换时的异常处理
    def test_replace_with_single_list(self):
        # 创建一个整数序列的 Pandas Series 对象
        ser = pd.Series([0, 1, 2, 3, 4])
        # 错误消息字符串，用于匹配异常信息
        msg = (
            "Series.replace must specify either 'value', "
            "a dict-like 'to_replace', or dict-like 'regex'"
        )
        # 验证是否抛出 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3])

        # 复制原始序列以备后用
        s = ser.copy()
        # 使用 inplace=True 参数调用替换操作，验证是否抛出相同的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            s.replace([1, 2, 3], inplace=True)
    # 测试替换混合类型数据的函数
    def test_replace_mixed_types(self):
        # 创建一个包含整数数据的 Pandas Series 对象
        ser = pd.Series(np.arange(5), dtype="int64")

        # 定义一个内部函数用于检查替换操作的结果
        def check_replace(to_rep, val, expected):
            # 复制原始的 Series 对象
            sc = ser.copy()
            # 执行替换操作，并返回替换后的结果
            result = ser.replace(to_rep, val)
            # 在原地执行替换操作，返回值应为 None
            return_value = sc.replace(to_rep, val, inplace=True)
            # 断言原地替换的返回值为 None
            assert return_value is None
            # 使用测试工具检查替换后的结果是否符合预期
            tm.assert_series_equal(expected, result)
            # 使用测试工具检查原始 Series 对象是否被正确替换
            tm.assert_series_equal(expected, sc)

        # 说明：3.0 可以被 int64 类型的 Series 对象容纳，因此不需要升级为 float，参见 GH#44940
        tr, v = [3], [3.0]
        # 测试替换操作
        check_replace(tr, v, ser)

        # 注意：这与使用标量 3 和 3.0 的结果是一致的
        check_replace(tr[0], v[0], ser)

        # 必须升级为 float 类型
        e = pd.Series([0, 1, 2, 3.5, 4])
        tr, v = [3], [3.5]
        check_replace(tr, v, e)

        # 升级为 object 类型
        e = pd.Series([0, 1, 2, 3.5, "a"])
        tr, v = [3, 4], [3.5, "a"]
        check_replace(tr, v, e)

        # 再次升级为 object 类型
        e = pd.Series([0, 1, 2, 3.5, pd.Timestamp("20130101")])
        tr, v = [3, 4], [3.5, pd.Timestamp("20130101")]
        check_replace(tr, v, e)

        # 升级为 object 类型
        e = pd.Series([0, 1, 2, 3.5, True], dtype="object")
        tr, v = [3, 4], [3.5, True]
        check_replace(tr, v, e)

        # 测试包含日期、浮点数、整数和字符串的对象
        dr = pd.Series(pd.date_range("1/1/2001", "1/10/2001", freq="D"))
        # 将日期序列转换为对象类型，然后进行替换操作
        result = dr.astype(object).replace([dr[0], dr[1], dr[2]], [1.0, 2, "a"])
        expected = pd.Series([1.0, 2, "a"] + dr[3:].tolist(), dtype=object)
        # 使用测试工具检查结果是否符合预期
        tm.assert_series_equal(result, expected)

    # 测试在不执行操作的情况下替换布尔值为字符串的函数
    def test_replace_bool_with_string_no_op(self):
        # 创建包含布尔值的 Series 对象
        s = pd.Series([True, False, True])
        # 替换操作不会改变 Series 对象本身
        result = s.replace("fun", "in-the-sun")
        # 使用测试工具检查结果是否符合预期
        tm.assert_series_equal(s, result)

    # 测试替换布尔值为字符串的函数
    def test_replace_bool_with_string(self):
        # 创建包含布尔值的 Series 对象
        s = pd.Series([True, False, True])
        # 替换所有 True 值为 "2u"
        result = s.replace(True, "2u")
        expected = pd.Series(["2u", False, "2u"])
        # 使用测试工具检查结果是否符合预期
        tm.assert_series_equal(expected, result)

    # 测试替换布尔值为布尔值的函数
    def test_replace_bool_with_bool(self):
        # 创建包含布尔值的 Series 对象
        s = pd.Series([True, False, True])
        # 替换所有 True 值为 False
        result = s.replace(True, False)
        expected = pd.Series([False] * len(s))
        # 使用测试工具检查结果是否符合预期
        tm.assert_series_equal(expected, result)

    # 测试替换包含布尔键的字典的函数
    def test_replace_with_dict_with_bool_keys(self):
        # 创建包含布尔值的 Series 对象
        s = pd.Series([True, False, True])
        # 使用字典进行替换，键为布尔值 True
        result = s.replace({"asdf": "asdb", True: "yes"})
        expected = pd.Series(["yes", False, "yes"])
        # 使用测试工具检查结果是否符合预期
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试将整数替换为缺失值的情况
    def test_replace_Int_with_na(self, any_int_ea_dtype):
        # GH 38267
        # 创建一个包含两个元素的 Series，元素为 0 和 None，数据类型为传入的 any_int_ea_dtype
        result = pd.Series([0, None], dtype=any_int_ea_dtype).replace(0, pd.NA)
        # 期望的结果是一个包含两个 pd.NA 元素的 Series，数据类型与之前相同
        expected = pd.Series([pd.NA, pd.NA], dtype=any_int_ea_dtype)
        # 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
        
        # 创建一个包含两个元素的 Series，元素为 0 和 1，数据类型为传入的 any_int_ea_dtype
        result = pd.Series([0, 1], dtype=any_int_ea_dtype).replace(0, pd.NA)
        # 将 1 替换为 pd.NA，同时保持 inplace=True
        result.replace(1, pd.NA, inplace=True)
        # 检查 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)

    # 定义第二个测试方法
    def test_replace2(self):
        N = 50
        # 创建一个包含 N 个随机数的 Series，数据类型为对象类型，索引为从 "2020-01-01" 开始的 N 个日期
        ser = pd.Series(
            np.fabs(np.random.default_rng(2).standard_normal(N)),
            pd.date_range("2020-01-01", periods=N),
            dtype=object,
        )
        # 将前 5 个元素设置为 np.nan
        ser[:5] = np.nan
        # 将索引为 6 到 9 的元素设置为字符串 "foo"
        ser[6:10] = "foo"
        # 将索引为 20 到 29 的元素设置为字符串 "bar"

        # 使用列表替换为单个值
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        # 检查 rs 的前 5 个元素是否全部为 -1
        assert (rs[:5] == -1).all()
        # 检查 rs 的索引为 6 到 9 的元素是否全部为 -1
        assert (rs[6:10] == -1).all()
        # 检查 rs 的索引为 20 到 29 的元素是否全部为 -1
        assert (rs[20:30] == -1).all()
        # 检查 ser 的前 5 个元素是否全部为 pd.NA
        assert (pd.isna(ser[:5])).all()

        # 使用不同的替换值
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        # 检查 rs 的前 5 个元素是否全部为 -1
        assert (rs[:5] == -1).all()
        # 检查 rs 的索引为 6 到 9 的元素是否全部为 -2
        assert (rs[6:10] == -2).all()
        # 检查 rs 的索引为 20 到 29 的元素是否全部为 -3
        assert (rs[20:30] == -3).all()
        # 检查 ser 的前 5 个元素是否全部为 pd.NA
        assert (pd.isna(ser[:5])).all()

        # 使用两个列表进行替换
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        # 检查 rs 和 rs2 是否相等
        tm.assert_series_equal(rs, rs2)

        # 在原地进行替换
        return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        # 检查 inplace 替换后返回值是否为 None
        assert return_value is None
        # 检查 ser 的前 5 个元素是否全部为 -1
        assert (ser[:5] == -1).all()
        # 检查 ser 的索引为 6 到 9 的元素是否全部为 -1
        assert (ser[6:10] == -1).all()
        # 检查 ser 的索引为 20 到 29 的元素是否全部为 -1

    @pytest.mark.parametrize("inplace", [True, False])
    # 定义测试方法，测试级联替换的行为
    def test_replace_cascade(self, inplace):
        # GH #50778
        # 创建一个包含 [1, 2, 3] 的 Series
        ser = pd.Series([1, 2, 3])
        # 创建一个期望的 Series，其元素为 [2, 3, 4]
        expected = pd.Series([2, 3, 4])

        # 执行替换操作，根据 inplace 参数决定是否原地替换
        res = ser.replace([1, 2, 3], [2, 3, 4], inplace=inplace)
        # 如果 inplace=True，则检查 ser 是否与 expected 相等
        if inplace:
            tm.assert_series_equal(ser, expected)
        else:
            # 如果 inplace=False，则检查返回的结果 res 是否与 expected 相等
            tm.assert_series_equal(res, expected)

    # 定义测试方法，测试在字符串数据类型上使用类字典的替换操作
    def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype):
        # GH 32621, GH#44940
        # 创建一个包含 ["one", "two", np.nan] 的 Series，数据类型为可空字符串
        ser = pd.Series(["one", "two", np.nan], dtype=nullable_string_dtype)
        # 创建一个期望的 Series，其元素为 ["1", "2", np.nan]
        expected = pd.Series(["1", "2", np.nan], dtype=nullable_string_dtype)
        # 执行替换操作，使用类字典的形式进行替换
        result = ser.replace({"one": "1", "two": "2"})
        # 检查期望的结果与实际结果是否相等
        tm.assert_series_equal(expected, result)

    # 定义测试方法，测试空类字典的替换操作
    def test_replace_with_empty_dictlike(self):
        # GH 15289
        # 创建一个包含 ['a', 'b', 'c', 'd'] 的 Series
        s = pd.Series(list("abcd"))
        # 检查用空字典替换后的 Series 是否与原始 Series 相等
        tm.assert_series_equal(s, s.replace({}))

        # 创建一个空 Series
        empty_series = pd.Series([])
        # 检查用空 Series 替换后的 Series 是否与原始 Series 相等
        tm.assert_series_equal(s, s.replace(empty_series))

    # 定义测试方法，测试字符串替换为数字的情况
    def test_replace_string_with_number(self):
        # GH 15743
        # 创建一个包含 [1, 2, 3] 的 Series
        s = pd.Series([1, 2, 3])
        # 将字符串 "2" 替换为 np.nan
        result = s.replace("2", np.nan)
        # 创建一个期望的 Series，其元素为 [1, 2, 3]
        expected = pd.Series([1, 2, 3])
        # 检查期望的结果与实际结果是否相等
        tm.assert_series_equal(expected, result)
    def test_replace_replacer_equals_replacement(self):
        # GH 20656
        # 确保所有替换器与原始值匹配
        s = pd.Series(["a", "b"])
        expected = pd.Series(["b", "a"])
        result = s.replace({"a": "b", "b": "a"})
        tm.assert_series_equal(expected, result)

    def test_replace_unicode_with_number(self):
        # GH 15743
        # 将 Unicode 替换为数字
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_mixed_types_with_string(self):
        # 测试混合类型替换为字符串
        s = pd.Series([1, 2, 3, "4", 4, 5])
        result = s.replace([2, "4"], np.nan)
        expected = pd.Series([1, np.nan, 3, np.nan, 4, 5], dtype=object)
        tm.assert_series_equal(expected, result)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't fill 0 in string")
    @pytest.mark.parametrize(
        "categorical, numeric",
        [
            (["A"], [1]),
            (["A", "B"], [1, 2]),
        ],
    )
    def test_replace_categorical(self, categorical, numeric):
        # GH 24971, GH#23305
        # 分类替换测试
        ser = pd.Series(pd.Categorical(categorical, categories=["A", "B"]))
        result = ser.cat.rename_categories({"A": 1, "B": 2})
        expected = pd.Series(numeric).astype("category")
        if 2 not in expected.cat.categories:
            # 即使没有"B"，类别应为[1, 2]
            # GH#44940
            expected = expected.cat.add_categories(2)
        tm.assert_series_equal(expected, result, check_categorical=False)

    def test_replace_categorical_inplace(self):
        # GH 53358
        # 分类替换测试（就地替换）
        data = ["a", "b", "c"]
        data_exp = ["b", "b", "c"]
        result = pd.Series(data, dtype="category")
        result.replace(to_replace="a", value="b", inplace=True)
        expected = pd.Series(pd.Categorical(data_exp, categories=data))
        tm.assert_series_equal(result, expected)

    def test_replace_categorical_single(self):
        # GH 26988
        # 单一分类替换测试
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        s = pd.Series(dti)
        c = s.astype("category")

        expected = c.copy()
        expected = expected.cat.add_categories("foo")
        expected[2] = "foo"
        expected = expected.cat.remove_unused_categories()
        assert c[2] != "foo"

        result = c.cat.rename_categories({c.values[2]: "foo"})
        tm.assert_series_equal(expected, result)
        assert c[2] != "foo"  # 确保非就地调用不会改变原始数据
    def test_replace_with_no_overflowerror(self):
        # GH 25616
        # 在无溢出错误的情况下将整数转换为对象
        s = pd.Series([0, 1, 2, 3, 4])
        result = s.replace([3], ["100000000000000000000"])
        expected = pd.Series([0, 1, 2, "100000000000000000000", 4])
        tm.assert_series_equal(result, expected)

        s = pd.Series([0, "100000000000000000000", "100000000000000000001"])
        result = s.replace(["100000000000000000000"], [1])
        expected = pd.Series([0, 1, "100000000000000000001"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, to_replace, exp",
        [
            ([1, 2, 3], {1: 2, 2: 3, 3: 4}, [2, 3, 4]),
            (["1", "2", "3"], {"1": "2", "2": "3", "3": "4"}, ["2", "3", "4"]),
        ],
    )
    def test_replace_commutative(self, ser, to_replace, exp):
        # GH 16051
        # 当值为非数值时，DataFrame.replace() 覆盖操作

        series = pd.Series(ser)

        expected = pd.Series(exp)
        result = series.replace(to_replace)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp", [([1, 2, 3], [1, True, 3]), (["x", 2, 3], ["x", True, 3])]
    )
    def test_replace_no_cast(self, ser, exp):
        # GH 9113
        # BUG: 将 int64 类型替换为布尔型会强制转换为 int64

        series = pd.Series(ser)
        result = series.replace(2, True)
        expected = pd.Series(exp)

        tm.assert_series_equal(result, expected)

    def test_replace_invalid_to_replace(self):
        # GH 18634
        # API: 若给定无效参数，则 replace() 应引发异常
        series = pd.Series(["a", "b", "c "])
        msg = (
            r"Expecting 'to_replace' to be either a scalar, array-like, "
            r"dict or None, got invalid type.*"
        )
        with pytest.raises(TypeError, match=msg):
            series.replace(lambda x: x.strip())

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_nonbool_regex(self, frame):
        obj = pd.Series(["a", "b", "c "])
        if frame:
            obj = obj.to_frame()

        msg = "'to_replace' must be 'None' if 'regex' is not a bool"
        with pytest.raises(ValueError, match=msg):
            obj.replace(to_replace=["a"], regex="foo")

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_empty_copy(self, frame):
        obj = pd.Series([], dtype=np.float64)
        if frame:
            obj = obj.to_frame()

        res = obj.replace(4, 5, inplace=True)
        assert res is None

        res = obj.replace(4, 5, inplace=False)
        tm.assert_equal(res, obj)
        assert res is not obj
    # 定义一个测试函数，用于测试 Series 对象的 replace 方法，使用单个类字典样式参数
    def test_replace_only_one_dictlike_arg(self, fixed_now_ts):
        # GH#33340

        # 创建一个包含整数、字符串、日期、布尔值的 Series 对象
        ser = pd.Series([1, 2, "A", fixed_now_ts, True])

        # 定义要替换的字典样式的映射关系和替换的值
        to_replace = {0: 1, 2: "A"}
        value = "foo"

        # 当使用字典样式的 to_replace 和非 None 值的 value 时，期望抛出 ValueError 异常
        msg = "Series.replace cannot use dict-like to_replace and non-None value"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

        # 更新 to_replace 和 value 变量，以测试另一种情况
        to_replace = 1
        value = {0: "foo", 2: "bar"}

        # 当使用非 None 的 to_replace 和字典值的 value 时，期望抛出 ValueError 异常
        msg = "Series.replace cannot use dict-value and non-None to_replace"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

    # 定义一个测试函数，用于测试 Series 对象的 replace 方法，测试替换字符串空值不引发异常的情况
    def test_replace_extension_other(self, frame_or_series):
        # https://github.com/pandas-dev/pandas/issues/34530

        # 创建一个包含整数的 Series 对象
        obj = frame_or_series(pd.array([1, 2, 3], dtype="Int64"))

        # 替换空字符串，预期不会引起数据类型变化
        result = obj.replace("", "")  # no exception

        # 断言结果和原始对象相等，不应该改变数据类型
        tm.assert_equal(obj, result)

    # 定义一个测试函数，测试 Series 对象的 replace 方法是否能够使用编译的正则表达式进行替换
    def test_replace_with_compiled_regex(self):
        # https://github.com/pandas-dev/pandas/issues/35680

        # 创建一个包含字符串的 Series 对象
        s = pd.Series(["a", "b", "c"])

        # 编译一个正则表达式
        regex = re.compile("^a$")

        # 使用编译的正则表达式替换符合条件的值
        result = s.replace({regex: "z"}, regex=True)

        # 预期的替换结果
        expected = pd.Series(["z", "b", "c"])

        # 断言结果和预期相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，测试 Series 对象的 replace 方法是否能够正确处理 NA 值的替换
    def test_pandas_replace_na(self):
        # GH#43344
        # GH#56599

        # 创建一个包含字符串的 Series 对象，包括 NA 值
        ser = pd.Series(["AA", "BB", "CC", "DD", "EE", "", pd.NA, "AA"], dtype="string")

        # 定义一个正则表达式映射关系，用于替换特定的字符串
        regex_mapping = {
            "AA": "CC",
            "BB": "CC",
            "EE": "CC",
            "CC": "CC-REPL",
        }

        # 使用正则表达式替换映射关系进行替换
        result = ser.replace(regex_mapping, regex=True)

        # 预期的替换结果
        exp = pd.Series(
            ["CC", "CC", "CC-REPL", "DD", "CC", "", pd.NA, "CC"], dtype="string"
        )

        # 断言结果和预期相等
        tm.assert_series_equal(result, exp)
    @pytest.mark.parametrize(
        "dtype, input_data, to_replace, expected_data",
        [
            ("bool", [True, False], {True: False}, [False, False]),
            ("int64", [1, 2], {1: 10, 2: 20}, [10, 20]),
            ("Int64", [1, 2], {1: 10, 2: 20}, [10, 20]),
            ("float64", [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
            ("Float64", [1.1, 2.2], {1.1: 10.1, 2.2: 20.5}, [10.1, 20.5]),
            ("string", ["one", "two"], {"one": "1", "two": "2"}, ["1", "2"]),
            (
                pd.IntervalDtype("int64"),
                IntervalArray([pd.Interval(1, 2), pd.Interval(2, 3)]),
                {pd.Interval(1, 2): pd.Interval(10, 20)},
                IntervalArray([pd.Interval(10, 20), pd.Interval(2, 3)]),
            ),
            (
                pd.IntervalDtype("float64"),
                IntervalArray([pd.Interval(1.0, 2.7), pd.Interval(2.8, 3.1)]),
                {pd.Interval(1.0, 2.7): pd.Interval(10.6, 20.8)},
                IntervalArray([pd.Interval(10.6, 20.8), pd.Interval(2.8, 3.1)]),
            ),
            (
                pd.PeriodDtype("M"),
                [pd.Period("2020-05", freq="M")],
                {pd.Period("2020-05", freq="M"): pd.Period("2020-06", freq="M")},
                [pd.Period("2020-06", freq="M")],
            ),
        ],
    )
    def test_replace_dtype(self, dtype, input_data, to_replace, expected_data):
        # GH#33484
        # 创建一个参数化测试函数，测试替换操作在不同数据类型和数据上的行为
        ser = pd.Series(input_data, dtype=dtype)
        # 使用输入数据和指定的数据类型创建 Pandas Series 对象
        result = ser.replace(to_replace)
        # 执行替换操作
        expected = pd.Series(expected_data, dtype=dtype)
        # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(result, expected)
        # 使用测试工具来比较实际结果和预期结果

    def test_replace_string_dtype(self):
        # GH#40732, GH#44940
        # 测试替换操作对字符串类型数据的影响
        ser = pd.Series(["one", "two", np.nan], dtype="string")
        # 创建一个字符串类型的 Pandas Series 对象，包括 NaN 值
        res = ser.replace({"one": "1", "two": "2"})
        # 执行替换操作
        expected = pd.Series(["1", "2", np.nan], dtype="string")
        # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(res, expected)
        # 使用测试工具来比较实际结果和预期结果

        # GH#31644
        ser2 = pd.Series(["A", np.nan], dtype="string")
        # 创建另一个字符串类型的 Pandas Series 对象，包括 NaN 值
        res2 = ser2.replace("A", "B")
        # 执行替换操作
        expected2 = pd.Series(["B", np.nan], dtype="string")
        # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(res2, expected2)

        ser3 = pd.Series(["A", "B"], dtype="string")
        # 创建另一个字符串类型的 Pandas Series 对象
        res3 = ser3.replace("A", pd.NA)
        # 执行替换操作
        expected3 = pd.Series([pd.NA, "B"], dtype="string")
        # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(res3, expected3)

    def test_replace_string_dtype_list_to_replace(self):
        # GH#41215, GH#44940
        # 测试替换操作对字符串类型数据中列表形式的多个替换值的影响
        ser = pd.Series(["abc", "def"], dtype="string")
        # 创建一个字符串类型的 Pandas Series 对象
        res = ser.replace(["abc", "any other string"], "xyz")
        # 执行替换操作
        expected = pd.Series(["xyz", "def"], dtype="string")
        # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(res, expected)

    def test_replace_string_dtype_regex(self):
        # GH#31644
        # 测试替换操作对字符串类型数据使用正则表达式的影响
        ser = pd.Series(["A", "B"], dtype="string")
        # 创建一个字符串类型的 Pandas Series 对象
        res = ser.replace(r".", "C", regex=True)
        # 执行替换操作
        expected = pd.Series(["C", "C"], dtype="string")
        # 创建预期结果的 Pandas Series 对象
        tm.assert_series_equal(res, expected)
    def test_replace_nullable_numeric(self):
        # 测试用例针对替换可空数值的操作

        # 创建包含浮点数的 Series，指定数据类型为 Float64Dtype
        floats = pd.Series([1.0, 2.0, 3.999, 4.4], dtype=pd.Float64Dtype())
        # 替换值为 1.0 的元素为 9，并检查返回的 Series 数据类型是否与原始相同
        assert floats.replace({1.0: 9}).dtype == floats.dtype
        # 同上，替换值为 1.0 的元素为 9，并检查返回的 Series 数据类型是否与原始相同
        assert floats.replace(1.0, 9).dtype == floats.dtype
        # 替换值为 1.0 的元素为 9.0，并检查返回的 Series 数据类型是否与原始相同
        assert floats.replace({1.0: 9.0}).dtype == floats.dtype
        # 同上，替换值为 1.0 的元素为 9.0，并检查返回的 Series 数据类型是否与原始相同
        assert floats.replace(1.0, 9.0).dtype == floats.dtype

        # 替换多个值为指定值，将 1.0 替换为 9.0，2.0 替换为 10.0，并检查返回的 Series 数据类型是否与原始相同
        res = floats.replace(to_replace=[1.0, 2.0], value=[9.0, 10.0])
        assert res.dtype == floats.dtype

        # 创建包含整数的 Series，指定数据类型为 Int64Dtype
        ints = pd.Series([1, 2, 3, 4], dtype=pd.Int64Dtype())
        # 替换值为 1 的元素为 9，并检查返回的 Series 数据类型是否与原始相同
        assert ints.replace({1: 9}).dtype == ints.dtype
        # 同上，替换值为 1 的元素为 9，并检查返回的 Series 数据类型是否与原始相同
        assert ints.replace(1, 9).dtype == ints.dtype
        # 替换值为 1 的元素为 9.0，并检查返回的 Series 数据类型是否与原始相同
        assert ints.replace({1: 9.0}).dtype == ints.dtype
        # 同上，替换值为 1 的元素为 9.0，并检查返回的 Series 数据类型是否与原始相同
        assert ints.replace(1, 9.0).dtype == ints.dtype

        # 当尝试将整数替换为浮点数时，预期引发 TypeError 异常，匹配错误消息为 "Invalid value"
        with pytest.raises(TypeError, match="Invalid value"):
            ints.replace({1: 9.5})
        with pytest.raises(TypeError, match="Invalid value"):
            ints.replace(1, 9.5)



    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't fill 1 in string")
    @pytest.mark.parametrize("regex", [False, True])
    def test_replace_regex_dtype_series(self, regex):
        # 测试用例针对 Series 的正则表达式替换操作，特定于 GH-48644

        # 创建包含字符串 "0" 的 Series
        series = pd.Series(["0"])
        # 期望的结果是将 "0" 替换为整数 1，并指定数据类型为 object
        expected = pd.Series([1], dtype=object)
        # 执行替换操作，将 "0" 替换为整数 1，根据 regex 参数指定是否为正则表达式替换
        result = series.replace(to_replace="0", value=1, regex=regex)
        # 断言替换后的结果与期望结果相同
        tm.assert_series_equal(result, expected)



    def test_replace_different_int_types(self, any_int_numpy_dtype):
        # 测试用例针对替换不同整数类型的操作，特定于 GH#45311

        # 创建包含不同整数类型的 Series，数据类型由 any_int_numpy_dtype 参数指定
        labs = pd.Series([1, 1, 1, 0, 0, 2, 2, 2], dtype=any_int_numpy_dtype)

        # 创建另一个 Series，用于映射原始整数值到新的整数值
        maps = pd.Series([0, 2, 1], dtype=any_int_numpy_dtype)
        map_dict = dict(zip(maps.values, maps.index))

        # 使用 map_dict 字典将 labs Series 中的值替换，并与期望的结果进行比较
        result = labs.replace(map_dict)
        expected = labs.replace({0: 0, 2: 1, 1: 2})
        tm.assert_series_equal(result, expected)



    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val):
        # 测试用例针对替换数值型数据中的特定值为 None 的操作，特定于 GH#48231

        # 创建包含数值的 Series，包括一个特定值 val，数据类型自动推断
        ser = pd.Series([1, val])
        # 将特定值 val 替换为 None，并与期望的结果进行比较，期望的结果指定数据类型为 object
        result = ser.replace(val, None)
        expected = pd.Series([1, None], dtype=object)
        tm.assert_series_equal(result, expected)



    def test_replace_change_dtype_series(self, using_infer_string):
        # 测试用例针对修改 Series 数据类型的替换操作，特定于 GH#25797

        # 从字典创建 DataFrame，包含一个名为 "Test" 的列，列中包含字符串和布尔值
        df = pd.DataFrame.from_dict({"Test": ["0.5", True, "0.6"]})
        # 根据 using_infer_string 参数是否为 True，生成警告类型为 FutureWarning 或不生成警告
        warn = FutureWarning if using_infer_string else None
        # 使用 assert_produces_warning 上下文管理器，检查将布尔值替换为 np.nan 是否引发警告
        with tm.assert_produces_warning(warn, match="Downcasting"):
            df["Test"] = df["Test"].replace([True], [np.nan])
        # 期望的结果是将 None 替换为 np.nan 后的 DataFrame
        expected = pd.DataFrame.from_dict({"Test": ["0.5", np.nan, "0.6"]})
        tm.assert_frame_equal(df, expected)

        # 创建另一个 DataFrame，包含一个名为 "Test" 的列，列中包含字符串和 None 值
        df = pd.DataFrame.from_dict({"Test": ["0.5", None, "0.6"]})
        # 将 None 值替换为 np.nan，并与期望的结果进行比较
        df["Test"] = df["Test"].replace([None], [np.nan])
        tm.assert_frame_equal(df, expected)

        # 创建另一个 DataFrame，包含一个名为 "Test" 的列，列中包含字符串和 None 值
        df = pd.DataFrame.from_dict({"Test": ["0.5", None, "0.6"]})
        # 将 None 值填充为 np.nan，并与期望的结果进行比较
        df["Test"] = df["Test"].fillna(np.nan)
        tm.assert_frame_equal(df, expected)
    # 使用 pytest.mark.parametrize 装饰器，允许在测试方法中使用不同的参数进行参数化测试
    @pytest.mark.parametrize("dtype", ["object", "Int64"])
    # 定义一个测试方法，用于测试在对象列中替换缺失值的行为
    def test_replace_na_in_obj_column(self, dtype):
        # 创建一个 Pandas Series 对象，包含整数和缺失值，数据类型由参数 dtype 指定
        ser = pd.Series([0, 1, pd.NA], dtype=dtype)
        # 创建期望的 Pandas Series 对象，其中第二个元素被替换为 2，数据类型与 ser 相同
        expected = pd.Series([0, 2, pd.NA], dtype=dtype)
        # 调用 replace 方法，将 ser 中的值 1 替换为 2，返回结果存储在 result 中
        result = ser.replace(to_replace=1, value=2)
        # 使用测试框架的 assert_series_equal 方法比较 result 和 expected，确保它们相等
        tm.assert_series_equal(result, expected)
    
        # 在原地修改 ser，将值 1 替换为 2，此时 ser 的值会发生改变
        ser.replace(to_replace=1, value=2, inplace=True)
        # 再次使用 assert_series_equal 方法比较修改后的 ser 和 expected，确保它们相等
        tm.assert_series_equal(ser, expected)
    
    # 使用 pytest.mark.parametrize 装饰器，允许在测试方法中使用不同的参数进行参数化测试
    @pytest.mark.parametrize("val", [0, 0.5])
    # 定义一个测试方法，用于测试将数值列中的数值替换为缺失值的行为
    def test_replace_numeric_column_with_na(self, val):
        # 创建一个 Pandas Series 对象，包含一个数值和一个整数，其中数值由参数 val 指定
        ser = pd.Series([val, 1])
        # 创建期望的 Pandas Series 对象，将第二个元素替换为缺失值 pd.NA
        expected = pd.Series([val, pd.NA])
        # 调用 replace 方法，将 ser 中的值 1 替换为 pd.NA，返回结果存储在 result 中
        result = ser.replace(to_replace=1, value=pd.NA)
        # 使用测试框架的 assert_series_equal 方法比较 result 和 expected，确保它们相等
        tm.assert_series_equal(result, expected)
    
        # 在原地修改 ser，将值 1 替换为 pd.NA，此时 ser 的值会发生改变
        ser.replace(to_replace=1, value=pd.NA, inplace=True)
        # 再次使用 assert_series_equal 方法比较修改后的 ser 和 expected，确保它们相等
        tm.assert_series_equal(ser, expected)
    
    # 定义一个测试方法，用于测试将浮点数列中的值替换为布尔值的行为
    def test_replace_ea_float_with_bool(self):
        # 创建一个 Pandas Series 对象，包含一个浮点数 0.0，数据类型为 "Float64"
        ser = pd.Series([0.0], dtype="Float64")
        # 复制 ser，创建期望的 Pandas Series 对象
        expected = ser.copy()
        # 调用 replace 方法，将 ser 中的 False 替换为 1.0，返回结果存储在 result 中
        result = ser.replace(False, 1.0)
        # 使用测试框架的 assert_series_equal 方法比较 result 和 expected，确保它们相等
        tm.assert_series_equal(result, expected)
    
        # 创建一个 Pandas Series 对象，包含一个布尔值 False，数据类型为 "boolean"
        ser = pd.Series([False], dtype="boolean")
        # 复制 ser，创建期望的 Pandas Series 对象
        expected = ser.copy()
        # 调用 replace 方法，将 ser 中的 0.0 替换为 True，返回结果存储在 result 中
        result = ser.replace(0.0, True)
        # 使用测试框架的 assert_series_equal 方法比较 result 和 expected，确保它们相等
        tm.assert_series_equal(result, expected)
```