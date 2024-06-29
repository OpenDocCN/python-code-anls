# `D:\src\scipysrc\pandas\pandas\tests\arrays\period\test_arrow_compat.py`

```
# 导入pytest库，用于测试框架
import pytest

# 从pandas.compat.pyarrow导入pa_version_under10p1，用于检查pyarrow版本是否低于10.1
from pandas.compat.pyarrow import pa_version_under10p1

# 从pandas.core.dtypes.dtypes导入PeriodDtype，用于处理周期类型数据
from pandas.core.dtypes.dtypes import PeriodDtype

# 导入pandas库，并将其命名为pd，用于数据处理和分析
import pandas as pd

# 导入pandas的测试工具集_pandas.testing，并将其命名为tm，用于数据测试
import pandas._testing as tm

# 从pandas.core.arrays导入PeriodArray和period_array，用于处理周期数组
from pandas.core.arrays import (
    PeriodArray,
    period_array,
)

# 将pytestmark设置为pytest.mark.filterwarnings，忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 导入pytest库中的importorskip函数，并将其命名为pa，用于导入pyarrow并检查其是否可用
pa = pytest.importorskip("pyarrow")


# 定义测试函数test_arrow_extension_type，用于测试ArrowPeriodType类型的扩展
def test_arrow_extension_type():
    # 从pandas.core.arrays.arrow.extension_types导入ArrowPeriodType，用于处理Arrow的周期类型
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    # 创建三个ArrowPeriodType对象，分别表示天(D)和月(M)的周期类型
    p1 = ArrowPeriodType("D")
    p2 = ArrowPeriodType("D")
    p3 = ArrowPeriodType("M")

    # 断言p1的频率为'D'
    assert p1.freq == "D"
    # 断言p1与p2相等
    assert p1 == p2
    # 断言p1与p3不相等
    assert p1 != p3
    # 断言p1与p2的哈希值相等
    assert hash(p1) == hash(p2)
    # 断言p1与p3的哈希值不相等
    assert hash(p1) != hash(p3)


# 标记为预期测试失败（xfail），当pyarrow版本小于10.1时，因其行为可能有误
@pytest.mark.xfail(not pa_version_under10p1, reason="Wrong behavior with pyarrow 10")
# 参数化测试函数test_arrow_array，用于测试Arrow数组的操作
@pytest.mark.parametrize(
    "data, freq",
    [
        # 测试使用pd.date_range生成的日期数据，频率为'D'（天）
        (pd.date_range("2017", periods=3), "D"),
        # 测试使用pd.date_range生成的日期数据，频率为'YE'（年末）
        (pd.date_range("2017", periods=3, freq="YE"), "Y-DEC"),
    ],
)
# 定义测试函数test_arrow_array，用于测试Arrow数组的创建和操作
def test_arrow_array(data, freq):
    # 从pandas.core.arrays.arrow.extension_types导入ArrowPeriodType，用于处理Arrow的周期类型
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    # 使用period_array函数根据给定数据和频率创建PeriodArray对象periods
    periods = period_array(data, freq=freq)
    # 使用pa.array将PeriodArray转换为pyarrow的Array对象result
    result = pa.array(periods)
    # 断言result的类型为ArrowPeriodType
    assert isinstance(result.type, ArrowPeriodType)
    # 断言result的频率与输入的频率freq相等
    assert result.type.freq == freq
    # 使用type="int64"创建期望的pyarrow Array对象expected
    expected = pa.array(periods.asi8, type="int64")
    # 断言result的存储内容与expected相等
    assert result.storage.equals(expected)

    # 将result转换为其存储类型
    result = pa.array(periods, type=pa.int64())
    # 断言result与expected相等
    assert result.equals(expected)

    # 不支持的转换操作
    msg = "Not supported to convert PeriodArray to 'double' type"
    # 使用pytest.raises断言抛出TypeError异常，并检查异常消息是否匹配msg
    with pytest.raises(TypeError, match=msg):
        pa.array(periods, type="float64")

    # 使用pytest.raises断言抛出TypeError异常，并检查异常消息是否包含'different 'freq''
    with pytest.raises(TypeError, match="different 'freq'"):
        pa.array(periods, type=ArrowPeriodType("T"))


# 定义测试函数test_arrow_array_missing，用于测试包含缺失值的Arrow数组
def test_arrow_array_missing():
    # 从pandas.core.arrays.arrow.extension_types导入ArrowPeriodType，用于处理Arrow的周期类型
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    # 创建包含周期类型的PeriodArray对象arr，其中包含一个缺失值NaT
    arr = PeriodArray([1, 2, 3], dtype="period[D]")
    arr[1] = pd.NaT

    # 使用pa.array将PeriodArray转换为pyarrow的Array对象result
    result = pa.array(arr)
    # 断言result的类型为ArrowPeriodType
    assert isinstance(result.type, ArrowPeriodType)
    # 断言result的频率为'D'
    assert result.type.freq == "D"
    # 使用type="int64"创建期望的pyarrow Array对象expected
    expected = pa.array([1, None, 3], type="int64")
    # 断言result的存储内容与expected相等
    assert result.storage.equals(expected)


# 定义测试函数test_arrow_table_roundtrip，用于测试Arrow表格的往返转换
def test_arrow_table_roundtrip():
    # 从pandas.core.arrays.arrow.extension_types导入ArrowPeriodType，用于处理Arrow的周期类型
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    # 创建包含周期类型的PeriodArray对象arr，其中包含一个缺失值NaT
    arr = PeriodArray([1, 2, 3], dtype="period[D]")
    arr[1] = pd.NaT
    # 创建DataFrame对象df，包含一个名为'a'的列，列数据为arr
    df = pd.DataFrame({"a": arr})

    # 使用pa.table将DataFrame对象df转换为pyarrow的Table对象table
    table = pa.table(df)
    # 断言table中字段'a'的类型为ArrowPeriodType
    assert isinstance(table.field("a").type, ArrowPeriodType)
    # 使用table.to_pandas将Table对象table转换为DataFrame对象result
    result = table.to_pandas()
    # 断言result中列'a'的数据类型为PeriodDtype
    assert isinstance(result["a"].dtype, PeriodDtype)
    # 使用_pandas.testing.assert_frame_equal断言result与df相等
    tm.assert_frame_equal(result, df)

    # 使用pa.concat_tables将两个相同的Table对象table连接为table2
    table2 = pa.concat_tables([table, table])
    # 将table2转换为DataFrame对象result
    result = table2.to_pandas()
    # 创建DataFrame对象expected，包含两个df的拼接结果
    expected = pd.concat([df, df], ignore_index=True)
    # 使用_pandas.testing.assert_frame_equal断言result与expected相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_arrow_load_from_zero_chunks，用于测试从空块加载Arrow数组
def test_arrow_load_from_zero_chunks():
    # GH-41040

    # 从pandas.core.arrays.arrow.extension_types导入ArrowPeriodType，用于处理Arrow的周期类型
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

    # 创建空的PeriodArray对象arr，数据类型为'period[D]'
    arr = PeriodArray([], dtype="period[D]")
    # 创建DataFrame对象df，包含一个名为'a'的列，列数据为arr
    df = pd.DataFrame({"a": arr})

    # 使用pa.table将DataFrame对象df转换为pyarrow的Table对象table
    table = pa.table(df)
    # 断言验证表的字段 "a" 的类型是否为 ArrowPeriodType 类型
    assert isinstance(table.field("a").type, ArrowPeriodType)
    
    # 使用空的分块数组创建一个新的表，并保持第一个列的数据类型与原表相同，生成的表结构由原表的 schema 决定
    table = pa.table(
        [pa.chunked_array([], type=table.column(0).type)], schema=table.schema
    )
    
    # 将 Arrow 表格转换为 Pandas 数据框
    result = table.to_pandas()
    
    # 断言验证结果数据框中列 "a" 的数据类型是否为 PeriodDtype 类型
    assert isinstance(result["a"].dtype, PeriodDtype)
    
    # 使用测试工具（test tool）验证 Pandas 数据框是否与预期的数据框 df 相等
    tm.assert_frame_equal(result, df)
# 定义一个测试函数，验证 Arrow 表格在没有元数据的情况下来回转换的正确性
def test_arrow_table_roundtrip_without_metadata():
    # 创建一个 PeriodArray 对象，包含三个时间周期（每小时）的数据，其中第二个值设为 NaT（Not a Time）
    arr = PeriodArray([1, 2, 3], dtype="period[h]")
    # 将第二个值设置为 NaT
    arr[1] = pd.NaT
    # 使用 PeriodArray 创建一个 DataFrame，列名为 'a'
    df = pd.DataFrame({"a": arr})

    # 将 DataFrame 转换为 Arrow 表格
    table = pa.table(df)
    
    # 移除表格的元数据
    table = table.replace_schema_metadata()
    
    # 断言表格的 schema 元数据为 None
    assert table.schema.metadata is None

    # 将 Arrow 表格转换为 Pandas DataFrame
    result = table.to_pandas()
    
    # 断言 'a' 列的数据类型为 PeriodDtype
    assert isinstance(result["a"].dtype, PeriodDtype)
    
    # 使用测试工具（testtools）断言转换后的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(result, df)
```