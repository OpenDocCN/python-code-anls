# `D:\src\scipysrc\pandas\pandas\tests\indexes\period\test_period.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下类和函数
    Index,  # 数据结构索引类
    NaT,  # Not a Time，表示缺失时间数据
    Period,  # 表示时期的类
    PeriodIndex,  # 表示时期索引的类
    Series,  # 表示一维数据的类
    date_range,  # 生成日期范围的函数
    offsets,  # 偏移量对象
    period_range,  # 生成时期范围的函数
)
import pandas._testing as tm  # 导入 pandas 内部测试工具模块

class TestPeriodIndex:
    def test_view_asi8(self):
        idx = PeriodIndex([], freq="M")  # 创建一个空的 PeriodIndex 对象，频率为月份

        exp = np.array([], dtype=np.int64)  # 期望的结果是空的 int64 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.view("i8"), exp)  # 断言 idx.view("i8") 的结果等于 exp
        tm.assert_numpy_array_equal(idx.asi8, exp)  # 断言 idx.asi8 的结果等于 exp

        idx = PeriodIndex(["2011-01", NaT], freq="M")  # 创建一个包含 "2011-01" 和 NaT 的 PeriodIndex 对象，频率为月份

        exp = np.array([492, -9223372036854775808], dtype=np.int64)  # 期望的结果是包含特定整数的 int64 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.view("i8"), exp)  # 断言 idx.view("i8") 的结果等于 exp
        tm.assert_numpy_array_equal(idx.asi8, exp)  # 断言 idx.asi8 的结果等于 exp

        exp = np.array([14975, -9223372036854775808], dtype=np.int64)  # 期望的结果是包含特定整数的 int64 类型的 NumPy 数组
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")  # 创建一个包含 "2011-01-01" 和 NaT 的 PeriodIndex 对象，频率为天
        tm.assert_numpy_array_equal(idx.view("i8"), exp)  # 断言 idx.view("i8") 的结果等于 exp
        tm.assert_numpy_array_equal(idx.asi8, exp)  # 断言 idx.asi8 的结果等于 exp

    def test_values(self):
        idx = PeriodIndex([], freq="M")  # 创建一个空的 PeriodIndex 对象，频率为月份

        exp = np.array([], dtype=object)  # 期望的结果是空的 object 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.values, exp)  # 断言 idx.values 的结果等于 exp
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)  # 断言 idx.to_numpy() 的结果等于 exp

        exp = np.array([], dtype=np.int64)  # 期望的结果是空的 int64 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.asi8, exp)  # 断言 idx.asi8 的结果等于 exp

        idx = PeriodIndex(["2011-01", NaT], freq="M")  # 创建一个包含 "2011-01" 和 NaT 的 PeriodIndex 对象，频率为月份

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)  # 期望的结果是包含 Period 和 NaT 对象的 object 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.values, exp)  # 断言 idx.values 的结果等于 exp
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)  # 断言 idx.to_numpy() 的结果等于 exp
        exp = np.array([492, -9223372036854775808], dtype=np.int64)  # 期望的结果是包含特定整数的 int64 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.asi8, exp)  # 断言 idx.asi8 的结果等于 exp

        idx = PeriodIndex(["2011-01-01", NaT], freq="D")  # 创建一个包含 "2011-01-01" 和 NaT 的 PeriodIndex 对象，频率为天

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)  # 期望的结果是包含 Period 和 NaT 对象的 object 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.values, exp)  # 断言 idx.values 的结果等于 exp
        tm.assert_numpy_array_equal(idx.to_numpy(), exp)  # 断言 idx.to_numpy() 的结果等于 exp
        exp = np.array([14975, -9223372036854775808], dtype=np.int64)  # 期望的结果是包含特定整数的 int64 类型的 NumPy 数组
        tm.assert_numpy_array_equal(idx.asi8, exp)  # 断言 idx.asi8 的结果等于 exp
    @pytest.mark.parametrize(
        "periodindex",
        [
            # 参数化测试数据：年度频率，从 "2001年1月1日" 到 "2005年12月1日" 的期间范围
            period_range(freq="Y", start="1/1/2001", end="12/1/2005"),
            # 参数化测试数据：季度频率，从 "2001年1月1日" 到 "2002年12月1日" 的期间范围
            period_range(freq="Q", start="1/1/2001", end="12/1/2002"),
            # 参数化测试数据：月度频率，从 "2001年1月1日" 到 "2002年1月1日" 的期间范围
            period_range(freq="M", start="1/1/2001", end="1/1/2002"),
            # 参数化测试数据：每日频率，从 "2001年12月1日" 到 "2001年6月1日" 的期间范围
            period_range(freq="D", start="12/1/2001", end="6/1/2001"),
            # 参数化测试数据：每小时频率，从 "2001年12月31日" 23:00 到 "2002年1月1日" 23:00 的期间范围
            period_range(freq="h", start="12/31/2001", end="1/1/2002 23:00"),
            # 参数化测试数据：每分钟频率，从 "2001年12月31日" 00:00 到 "2002年1月1日" 00:20 的期间范围
            period_range(freq="Min", start="12/31/2001", end="1/1/2002 00:20"),
            # 参数化测试数据：每秒频率，从 "2001年12月31日" 00:00:00 到 "2001年12月31日" 00:05:00 的期间范围
            period_range(
                freq="s", start="12/31/2001 00:00:00", end="12/31/2001 00:05:00"
            ),
            # 参数化测试数据：从 "2006年12月31日" 10个周期的期间范围
            period_range(end=Period("2006-12-31", "W"), periods=10),
        ],
    )
    def test_fields(self, periodindex, field):
        # 将 periodindex 转换为列表
        periods = list(periodindex)
        # 创建 Series 对象
        ser = Series(periodindex)

        # 获取 periodindex 中 field 对应的索引
        field_idx = getattr(periodindex, field)
        # 断言 periodindex 的长度与 field 索引的长度相等
        assert len(periodindex) == len(field_idx)
        # 遍历 periods 和 field_idx，断言每个 period 的 field 属性与 field_idx 中对应值相等
        for x, val in zip(periods, field_idx):
            assert getattr(x, field) == val

        # 如果 ser 的长度为0，则返回
        if len(ser) == 0:
            return

        # 获取 ser.dt 中 field 对应的属性
        field_s = getattr(ser.dt, field)
        # 断言 periodindex 的长度与 field_s 的长度相等
        assert len(periodindex) == len(field_s)
        # 遍历 periods 和 field_s，断言每个 period 的 field 属性与 field_s 中对应值相等
        for x, val in zip(periods, field_s):
            assert getattr(x, field) == val

    def test_is_(self):
        # 创建一个函数 create_index，返回一个从 "2001年1月1日" 到 "2009年12月1日" 的 period_range 对象
        create_index = lambda: period_range(freq="Y", start="1/1/2001", end="12/1/2009")
        # 创建 index 对象
        index = create_index()
        # 断言 index 与自身相等
        assert index.is_(index)
        # 断言 index 与 create_index() 创建的对象不相等
        assert not index.is_(create_index())
        # 断言 index 与 index.view() 相等
        assert index.is_(index.view())
        # 多次 view() 操作，断言结果仍与 index 相等
        assert index.is_(index.view().view().view().view().view())
        # 修改 index 的 name 属性为 "Apple"，断言 index 与 ind2 相等
        index.name = "Apple"
        ind2 = index.view()
        assert ind2.is_(index)
        # 断言 index 与其切片对象不相等
        assert not index.is_(index[:])
        # 断言 index 与 asfreq("M") 后的对象不相等
        assert not index.is_(index.asfreq("M"))
        # 断言 index 与 asfreq("Y") 后的对象不相等
        assert not index.is_(index.asfreq("Y"))

        # 断言 index 与 index - 2 后的对象不相等
        assert not index.is_(index - 2)
        # 断言 index 与 index - 0 后的对象不相等
        assert not index.is_(index - 0)

    def test_index_unique(self):
        # 创建一个 PeriodIndex 对象 idx
        idx = PeriodIndex([2000, 2007, 2007, 2009, 2009], freq="Y-JUN")
        # 创建一个期望的 PeriodIndex 对象 expected
        expected = PeriodIndex([2000, 2007, 2009], freq="Y-JUN")
        # 断言 idx.unique() 返回的对象与 expected 相等
        tm.assert_index_equal(idx.unique(), expected)
        # 断言 idx 的唯一值数量为 3
        assert idx.nunique() == 3

    def test_pindex_fieldaccessor_nat(self):
        # 创建一个 PeriodIndex 对象 idx
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2012-03", "2012-04"], freq="D", name="name"
        )

        # 创建一个预期的 Index 对象 exp
        exp = Index([2011, 2011, -1, 2012, 2012], dtype=np.int64, name="name")
        # 断言 idx.year 返回的对象与 exp 相等
        tm.assert_index_equal(idx.year, exp)

        # 创建一个预期的 Index 对象 exp
        exp = Index([1, 2, -1, 3, 4], dtype=np.int64, name="name")
        # 断言 idx.month 返回的对象与 exp 相等
        tm.assert_index_equal(idx.month, exp)
    # 定义测试函数，测试 PeriodIndex 对象的生成和功能
    def test_pindex_multiples(self):
        # 预期的 PeriodIndex 对象，包含指定的时间序列和频率
        expected = PeriodIndex(
            ["2011-01", "2011-03", "2011-05", "2011-07", "2011-09", "2011-11"],
            freq="2M",
        )

        # 生成一个 PeriodIndex 对象 pi，起始时间为 '1/1/11'，结束时间为 '12/31/11'，频率为 '2M'
        pi = period_range(start="1/1/11", end="12/31/11", freq="2M")
        # 检查生成的 pi 是否与预期的 expected 相等
        tm.assert_index_equal(pi, expected)
        # 检查 pi 的频率是否为 MonthEnd(2)
        assert pi.freq == offsets.MonthEnd(2)
        # 检查 pi 的频率字符串是否为 '2M'

        assert pi.freqstr == "2M"

        # 重新生成一个 PeriodIndex 对象 pi，起始时间为 '1/1/11'，包含 6 个周期，频率为 '2M'
        pi = period_range(start="1/1/11", periods=6, freq="2M")
        # 检查生成的 pi 是否与预期的 expected 相等
        tm.assert_index_equal(pi, expected)
        # 检查 pi 的频率是否为 MonthEnd(2)
        assert pi.freq == offsets.MonthEnd(2)
        # 检查 pi 的频率字符串是否为 '2M'

        assert pi.freqstr == "2M"

    # 使用 pytest 的标记，忽略特定的警告消息
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    @pytest.mark.filterwarnings("ignore:Period with BDay freq:FutureWarning")
    # 定义测试函数，测试 PeriodIndex 对象的迭代功能
    def test_iteration(self):
        # 生成一个 PeriodIndex 对象 index，起始时间为 '1/1/10'，包含 4 个周期，频率为 'B'（工作日）
        index = period_range(start="1/1/10", periods=4, freq="B")

        # 将 PeriodIndex 对象 index 转换为列表 result
        result = list(index)
        # 检查 result 的第一个元素是否为 Period 对象
        assert isinstance(result[0], Period)
        # 检查 result 的第一个元素的频率是否与 index 的频率相同
        assert result[0].freq == index.freq

    # 定义测试函数，测试带有多重索引的 Series 对象的生成和功能
    def test_with_multi_index(self):
        # 生成一个 DatetimeIndex 对象 index，从 '1/1/2012' 开始，包含 4 个时间点，频率为 '12h'（12 小时）
        index = date_range("1/1/2012", periods=4, freq="12h")
        # 将 index 转换为以 'D'（天）频率的 PeriodIndex 对象和小时数组的列表 index_as_arrays
        index_as_arrays = [index.to_period(freq="D"), index.hour]

        # 使用 index_as_arrays 创建一个 Series 对象 s，值为 [0, 1, 2, 3]
        s = Series([0, 1, 2, 3], index_as_arrays)

        # 检查 s 的第一级索引是否为 PeriodIndex 对象
        assert isinstance(s.index.levels[0], PeriodIndex)

        # 检查 s 的第一个索引值的第一个元素是否为 Period 对象
        assert isinstance(s.index.values[0][0], Period)

    # 定义测试函数，测试 PeriodIndex 对象的 map 方法
    def test_map(self):
        # 生成一个 PeriodIndex 对象 index，包含年份 [2005, 2007, 2009]，频率为 'Y'（年）
        index = PeriodIndex([2005, 2007, 2009], freq="Y")
        # 对 index 中的每个元素执行 lambda 函数 x: x.ordinal，得到结果 result
        result = index.map(lambda x: x.ordinal)
        # 生成一个期望的 Index 对象 exp，包含 index 中每个元素的 ordinal 属性值
        exp = Index([x.ordinal for x in index])
        # 检查 result 是否与 exp 相等
        tm.assert_index_equal(result, exp)
# 定义一个测试函数，用于测试可能的时间差转换
def test_maybe_convert_timedelta():
    # 创建一个日期索引对象，包含两个日期字符串 "2000" 和 "2001"，频率为每天
    pi = PeriodIndex(["2000", "2001"], freq="D")
    # 创建一个时间偏移对象，表示两天的偏移量
    offset = offsets.Day(2)
    # 断言调用期间索引对象的 _maybe_convert_timedelta 方法，返回值应为 2
    assert pi._maybe_convert_timedelta(offset) == 2
    # 断言调用期间索引对象的 _maybe_convert_timedelta 方法，输入参数为 2 时，返回值应为 2
    assert pi._maybe_convert_timedelta(2) == 2

    # 创建一个工作日时间偏移对象
    offset = offsets.BusinessDay()
    # 定义一个错误消息，指示输入的频率与期间索引对象的频率不同
    msg = r"Input has different freq=B from PeriodIndex\(freq=D\)"
    # 使用 pytest 的上下文管理，断言调用期间索引对象的 _maybe_convert_timedelta 方法会引发 ValueError 异常，且异常消息匹配预期消息
    with pytest.raises(ValueError, match=msg):
        pi._maybe_convert_timedelta(offset)


# 使用 pytest 的参数化装饰器，定义一个参数化测试函数，测试对象为 array 参数（True 和 False）
@pytest.mark.parametrize("array", [True, False])
def test_dunder_array(array):
    # 创建一个日期索引对象，包含两个日期字符串 "2000-01-01" 和 "2001-01-01"，频率为每天
    obj = PeriodIndex(["2000-01-01", "2001-01-01"], freq="D")
    # 根据 array 参数决定是否使用索引对象的 _data 属性
    if array:
        obj = obj._data

    # 创建一个预期的 NumPy 数组，其元素为索引对象的第一个和第二个元素，数据类型为对象
    expected = np.array([obj[0], obj[1]], dtype=object)
    # 调用 np.array 函数，将索引对象转换为 NumPy 数组，并断言转换结果与预期数组相等
    result = np.array(obj)
    tm.assert_numpy_array_equal(result, expected)

    # 调用 np.asarray 函数，将索引对象转换为 NumPy 数组，并断言转换结果与预期数组相等
    result = np.asarray(obj)
    tm.assert_numpy_array_equal(result, expected)

    # 获取索引对象的 asi8 属性作为预期结果
    expected = obj.asi8
    # 遍历不同的数据类型字符串，如 "i8", "int64", np.int64
    for dtype in ["i8", "int64", np.int64]:
        # 调用 np.array 函数，指定数据类型为当前遍历的 dtype，断言抛出 TypeError 异常，且异常消息包含预期字符串
        with pytest.raises(TypeError, match="argument must be"):
            np.array(obj, dtype=dtype)
        # 调用 np.array 函数，指定数据类型为 np 中对应的 dtype 对象，断言抛出 TypeError 异常，且异常消息包含预期字符串
        with pytest.raises(TypeError, match="argument must be"):
            np.array(obj, dtype=getattr(np, dtype))

    # 遍历不同的数据类型字符串，如 "float64", "int32", "uint64"
    for dtype in ["float64", "int32", "uint64"]:
        # 定义一个错误消息，指示 np.array 函数的参数类型错误
        msg = "argument must be"
        # 使用 pytest 的上下文管理，断言调用 np.array 函数时会引发 TypeError 异常，且异常消息包含预期字符串
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=dtype)
        # 使用 pytest 的上下文管理，断言调用 np.array 函数时会引发 TypeError 异常，且异常消息包含预期字符串
        with pytest.raises(TypeError, match=msg):
            np.array(obj, dtype=getattr(np, dtype))
```