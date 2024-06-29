# `D:\src\scipysrc\pandas\pandas\tests\arrays\test_period.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架

# 导入 pandas 库的特定模块和类
from pandas._libs.tslibs import iNaT  # 导入 iNaT 类
from pandas._libs.tslibs.period import IncompatibleFrequency  # 导入 IncompatibleFrequency 类

# 导入 pandas 核心数据类型的注册表和特定数据类型类
from pandas.core.dtypes.base import _registry as registry  # 导入注册表 _registry
from pandas.core.dtypes.dtypes import PeriodDtype  # 导入 PeriodDtype 类

# 导入 pandas 库及其测试模块
import pandas as pd  # 导入 Pandas 库
import pandas._testing as tm  # 导入测试工具模块
from pandas.core.arrays import PeriodArray  # 导入 PeriodArray 类

# ----------------------------------------------------------------------------
# Dtype


def test_registered():
    assert PeriodDtype in registry.dtypes  # 断言 PeriodDtype 在注册表 dtypes 中
    result = registry.find("Period[D]")  # 查找注册表中 Period[D] 的对象
    expected = PeriodDtype("D")  # 创建期望的 PeriodDtype("D") 对象
    assert result == expected  # 断言查找结果与期望相同


# ----------------------------------------------------------------------------
# period_array


def test_asi8():
    result = PeriodArray._from_sequence(["2000", "2001", None], dtype="period[D]").asi8
    expected = np.array([10957, 11323, iNaT])  # 期望的 NumPy 数组结果
    tm.assert_numpy_array_equal(result, expected)  # 使用测试工具验证结果数组是否与期望相同


def test_take_raises():
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")  # 创建 PeriodArray 对象
    with pytest.raises(IncompatibleFrequency, match="freq"):  # 捕获 IncompatibleFrequency 异常，并匹配异常消息中的 "freq"
        arr.take([0, -1], allow_fill=True, fill_value=pd.Period("2000", freq="W"))  # 尝试使用指定值填充索引位置

    msg = "value should be a 'Period' or 'NaT'. Got 'str' instead"
    with pytest.raises(TypeError, match=msg):  # 捕获 TypeError 异常，并匹配异常消息
        arr.take([0, -1], allow_fill=True, fill_value="foo")  # 尝试使用字符串填充索引位置


def test_fillna_raises():
    arr = PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")  # 创建 PeriodArray 对象
    with pytest.raises(ValueError, match="Length"):  # 捕获 ValueError 异常，并匹配异常消息中的 "Length"
        arr.fillna(arr[:2])  # 尝试使用部分数组填充缺失值


def test_fillna_copies():
    arr = PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")  # 创建 PeriodArray 对象
    result = arr.fillna(pd.Period("2000", "D"))  # 使用指定的 Period 值填充缺失值
    assert result is not arr  # 断言返回的结果对象不是原始对象的引用


# ----------------------------------------------------------------------------
# setitem


@pytest.mark.parametrize(  # 参数化测试
    "key, value, expected",  # 参数名称
    [  # 参数列表
        ([0], pd.Period("2000", "D"), [10957, 1, 2]),  # 第一个参数组合
        ([0], None, [iNaT, 1, 2]),  # 第二个参数组合
        ([0], np.nan, [iNaT, 1, 2]),  # 第三个参数组合
        ([0, 1, 2], pd.Period("2000", "D"), [10957] * 3),  # 第四个参数组合
        (  # 第五个参数组合
            [0, 1, 2],
            [
                pd.Period("2000", "D"),
                pd.Period("2001", "D"),
                pd.Period("2002", "D")
            ],
            [10957, 11323, 11688],  # 第五个参数组合的期望结果
        ),
    ],
)
def test_setitem(key, value, expected):
    arr = PeriodArray(np.arange(3), dtype="period[D]")  # 创建 PeriodArray 对象
    expected = PeriodArray(expected, dtype="period[D]")  # 创建期望的 PeriodArray 对象
    arr[key] = value  # 设置数组的索引位置为指定值
    tm.assert_period_array_equal(arr, expected)  # 使用测试工具验证期望的数组与结果数组是否相同


def test_setitem_raises_incompatible_freq():
    arr = PeriodArray(np.arange(3), dtype="period[D]")  # 创建 PeriodArray 对象
    with pytest.raises(IncompatibleFrequency, match="freq"):  # 捕获 IncompatibleFrequency 异常，并匹配异常消息中的 "freq"
        arr[0] = pd.Period("2000", freq="Y")  # 尝试设置不兼容频率的值为年度

    other = PeriodArray._from_sequence(["2000", "2001"], dtype="period[Y]")  # 创建另一个 PeriodArray 对象
    with pytest.raises(IncompatibleFrequency, match="freq"):  # 捕获 IncompatibleFrequency 异常，并匹配异常消息中的 "freq"
        arr[[0, 1]] = other  # 尝试将另一个频率的数组赋给当前数组的多个索引位置


def test_setitem_raises_length():
    arr = PeriodArray(np.arange(3), dtype="period[D]")  # 创建 PeriodArray 对象
    with pytest.raises(ValueError, match="length"):  # 捕获 ValueError 异常，并匹配异常消息中的 "length"
        arr[[0, 1]] = [pd.Period("2000", freq="D")]  # 尝试设置长度不匹配的值数组
# 当设置项引发类型错误时，测试函数
def test_setitem_raises_type():
    # 创建一个 PeriodArray 对象，使用整数数组初始化，并指定数据类型为 "period[D]"
    arr = PeriodArray(np.arange(3), dtype="period[D]")
    # 使用 pytest 检查是否引发了 TypeError 异常，并匹配错误信息 "int"
    with pytest.raises(TypeError, match="int"):
        # 尝试将索引位置为 0 的元素设置为整数 1，预期会引发类型错误异常
        arr[0] = 1


# ----------------------------------------------------------------------------
# Ops


# 测试 PeriodArray 对象与不兼容频率的 Period 对象相减时是否引发异常
def test_sub_period():
    # 创建一个 PeriodArray 对象，从日期字符串序列 ["2000", "2001"] 初始化，数据类型为 "period[D]"
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    # 创建一个 Period 对象，表示 2000 年，频率为月份 "M"
    other = pd.Period("2000", freq="M")
    # 使用 pytest 检查是否引发 IncompatibleFrequency 异常，并匹配错误信息 "freq"
    with pytest.raises(IncompatibleFrequency, match="freq"):
        # 尝试计算 PeriodArray 对象与 Period 对象的差，预期会引发频率不兼容的异常
        arr - other


# 测试 PeriodArray 对象与超出整数范围的 Period 对象相减时是否引发溢出异常
def test_sub_period_overflow():
    # 创建一个日期时间索引，从 "1677-09-22" 开始的 2 个日期，频率为天 "D"
    dti = pd.date_range("1677-09-22", periods=2, freq="D")
    # 将日期时间索引转换为 PeriodArray 对象，数据类型为 "ns"
    pi = dti.to_period("ns")

    # 创建一个超出整数范围的 Period 对象，使用 _from_ordinal 方法
    per = pd.Period._from_ordinal(10**14, pi.freq)

    # 使用 pytest 检查是否引发 OverflowError 异常，并匹配错误信息 "Overflow in int64 addition"
    with pytest.raises(OverflowError, match="Overflow in int64 addition"):
        # 尝试计算 PeriodArray 对象与超出整数范围的 Period 对象的差，预期会引发溢出异常
        pi - per

    # 同样地，尝试反向计算，检查是否同样引发溢出异常
    with pytest.raises(OverflowError, match="Overflow in int64 addition"):
        # 尝试计算超出整数范围的 Period 对象与 PeriodArray 对象的差，预期会引发溢出异常
        per - pi


# ----------------------------------------------------------------------------
# Methods


# 测试 PeriodArray 对象的 where 方法在频率不同的情况下是否引发异常
@pytest.mark.parametrize(
    "other",
    [
        pd.Period("2000", freq="h"),
        PeriodArray._from_sequence(["2000", "2001", "2000"], dtype="period[h]"),
    ],
)
def test_where_different_freq_raises(other):
    # 创建一个 Series 对象，其值为 PeriodArray 对象，从日期字符串序列 ["2000", "2001", "2002"] 初始化，数据类型为 "period[D]"
    ser = pd.Series(
        PeriodArray._from_sequence(["2000", "2001", "2002"], dtype="period[D]")
    )
    # 创建一个布尔数组作为条件
    cond = np.array([True, False, True])

    # 使用 pytest 检查是否引发 IncompatibleFrequency 异常，并匹配错误信息 "freq"
    with pytest.raises(IncompatibleFrequency, match="freq"):
        # 尝试调用 PeriodArray 对象的 _where 方法，传入条件和不兼容频率的 other 对象，预期会引发异常
        ser.array._where(cond, other)

    # 将 Series 对象的 where 方法的结果与预期结果进行比较
    res = ser.where(cond, other)
    expected = ser.astype(object).where(cond, other)
    # 使用 tm.assert_series_equal 检查结果是否与预期相等
    tm.assert_series_equal(res, expected)


# ----------------------------------------------------------------------------
# Printing


# 测试 PeriodArray 对象在长度较小情况下的字符串表示是否与预期相符
def test_repr_small():
    # 创建一个 PeriodArray 对象，从日期字符串序列 ["2000", "2001"] 初始化，数据类型为 "period[D]"
    arr = PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]")
    # 调用 PeriodArray 对象的字符串表示方法，将结果与预期字符串进行比较
    result = str(arr)
    expected = (
        "<PeriodArray>\n['2000-01-01', '2001-01-01']\nLength: 2, dtype: period[D]"
    )
    assert result == expected


# 测试 PeriodArray 对象在长度较大情况下的字符串表示是否与预期相符
def test_repr_large():
    # 创建一个 PeriodArray 对象，从日期字符串序列 ["2000", "2001"] 重复 500 次初始化，数据类型为 "period[D]"
    arr = PeriodArray._from_sequence(["2000", "2001"] * 500, dtype="period[D]")
    # 调用 PeriodArray 对象的字符串表示方法，将结果与预期字符串进行比较
    result = str(arr)
    expected = (
        "<PeriodArray>\n"
        "['2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01',\n"
        " ...\n"
        " '2000-01-01', '2001-01-01', '2000-01-01', '2001-01-01', "
        "'2000-01-01',\n"
        " '2001-01-01', '2000-01-01', '2001-01-01', '2000-01-01', "
        "'2001-01-01']\n"
        "Length: 1000, dtype: period[D]"
    )
    assert result == expected
```