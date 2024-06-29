# `D:\src\scipysrc\pandas\pandas\tests\arrays\interval\test_interval_pyarrow.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部测试工具
from pandas.core.arrays import IntervalArray  # 从 Pandas 库的数组模块中导入 IntervalArray 类


def test_arrow_extension_type():
    pa = pytest.importorskip("pyarrow")  # 导入 PyArrow 库，如果库不存在则跳过测试

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType  # 从 Pandas 的 Arrow 扩展类型中导入 ArrowIntervalType 类

    p1 = ArrowIntervalType(pa.int64(), "left")  # 创建一个 ArrowIntervalType 实例，指定整数类型和左闭区间
    p2 = ArrowIntervalType(pa.int64(), "left")  # 创建另一个 ArrowIntervalType 实例，与 p1 相同
    p3 = ArrowIntervalType(pa.int64(), "right")  # 创建第三个 ArrowIntervalType 实例，指定右闭区间

    assert p1.closed == "left"  # 断言 p1 实例的闭合方向为左
    assert p1 == p2  # 断言 p1 等于 p2
    assert p1 != p3  # 断言 p1 不等于 p3
    assert hash(p1) == hash(p2)  # 断言 p1 和 p2 的哈希值相同
    assert hash(p1) != hash(p3)  # 断言 p1 和 p3 的哈希值不同


def test_arrow_array():
    pa = pytest.importorskip("pyarrow")  # 导入 PyArrow 库，如果库不存在则跳过测试

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType  # 从 Pandas 的 Arrow 扩展类型中导入 ArrowIntervalType 类

    intervals = pd.interval_range(1, 5, freq=1).array  # 创建一个 IntervalArray，包含整数区间范围 [1, 5)，步长为1

    result = pa.array(intervals)  # 使用 PyArrow 创建一个包含 intervals 的数组
    assert isinstance(result.type, ArrowIntervalType)  # 断言 result 的类型是 ArrowIntervalType
    assert result.type.closed == intervals.closed  # 断言 result 的闭合方向与 intervals 相同
    assert result.type.subtype == pa.int64()  # 断言 result 的子类型为整数类型

    # 断言 result 的存储字段 "left" 和 "right" 分别等于指定的整数数组
    assert result.storage.field("left").equals(pa.array([1, 2, 3, 4], type="int64"))
    assert result.storage.field("right").equals(pa.array([2, 3, 4, 5], type="int64"))

    # 期望的结果是一个包含结构化数据的 PyArrow 数组
    expected = pa.array([{"left": i, "right": i + 1} for i in range(1, 5)])
    assert result.storage.equals(expected)

    # 将 intervals 转换为指定的存储类型
    result = pa.array(intervals, type=expected.type)
    assert result.equals(expected)

    # 不支持的类型转换，断言会抛出 TypeError 异常
    with pytest.raises(TypeError, match="Not supported to convert IntervalArray"):
        pa.array(intervals, type="float64")

    with pytest.raises(TypeError, match="Not supported to convert IntervalArray"):
        pa.array(intervals, type=ArrowIntervalType(pa.float64(), "left"))


def test_arrow_array_missing():
    pa = pytest.importorskip("pyarrow")  # 导入 PyArrow 库，如果库不存在则跳过测试

    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType  # 从 Pandas 的 Arrow 扩展类型中导入 ArrowIntervalType 类

    arr = IntervalArray.from_breaks([0.0, 1.0, 2.0, 3.0])  # 使用间隔断点创建 IntervalArray
    arr[1] = None  # 将第二个元素设置为 None

    result = pa.array(arr)  # 使用 PyArrow 创建一个包含 arr 的数组
    assert isinstance(result.type, ArrowIntervalType)  # 断言 result 的类型是 ArrowIntervalType
    assert result.type.closed == arr.closed  # 断言 result 的闭合方向与 arr 相同
    assert result.type.subtype == pa.float64()  # 断言 result 的子类型为浮点数类型

    # 断言存储字段 "left" 和 "right" 分别等于指定的浮点数数组，包含缺失值而非 NaN
    left = pa.array([0.0, None, 2.0], type="float64")
    right = pa.array([1.0, None, 3.0], type="float64")
    assert result.storage.field("left").equals(left)
    assert result.storage.field("right").equals(right)

    # 结构化数组本身也包含数组级别的缺失值
    vals = [
        {"left": 0.0, "right": 1.0},
        {"left": None, "right": None},
        {"left": 2.0, "right": 3.0},
    ]
    expected = pa.StructArray.from_pandas(vals, mask=np.array([False, True, False]))
    assert result.storage.equals(expected)


@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
@pytest.mark.parametrize(
    "breaks",
    [[0.0, 1.0, 2.0, 3.0], pd.date_range("2017", periods=4, freq="D")],
    ids=["float", "datetime64[ns]"],
)
def test_arrow_table_roundtrip(breaks):
    pa = pytest.importorskip("pyarrow")  # 导入 PyArrow 库，如果库不存在则跳过测试
    # 导入 ArrowIntervalType 类型，用于处理箭头库中的时间间隔数据类型
    from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
    
    # 根据给定的断点数组创建一个 IntervalArray 对象
    arr = IntervalArray.from_breaks(breaks)
    
    # 将 IntervalArray 中索引为 1 的位置设置为 None
    arr[1] = None
    
    # 使用 IntervalArray 创建一个 DataFrame，列名为"a"
    df = pd.DataFrame({"a": arr})
    
    # 将 DataFrame 转换为 PyArrow 的表格对象
    table = pa.table(df)
    
    # 断言表格中字段"a"的类型是 ArrowIntervalType 类型
    assert isinstance(table.field("a").type, ArrowIntervalType)
    
    # 将表格对象转换为 Pandas 的 DataFrame
    result = table.to_pandas()
    
    # 断言结果 DataFrame 的"a"列的数据类型是 pd.IntervalDtype 类型
    assert isinstance(result["a"].dtype, pd.IntervalDtype)
    
    # 断言结果 DataFrame 与原始 DataFrame df 相等
    tm.assert_frame_equal(result, df)
    
    # 将两个表格对象合并为一个新的表格对象
    table2 = pa.concat_tables([table, table])
    
    # 将合并后的表格对象转换为 Pandas 的 DataFrame
    result = table2.to_pandas()
    
    # 创建预期的合并结果 DataFrame，忽略索引
    expected = pd.concat([df, df], ignore_index=True)
    
    # 断言合并后的结果 DataFrame 与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
    
    # GH#41040 注释：创建一个空表格对象，其结构与已有表格对象的第一列相同
    table = pa.table(
        [pa.chunked_array([], type=table.column(0).type)], schema=table.schema
    )
    
    # 将表格对象转换为 Pandas 的 DataFrame
    result = table.to_pandas()
    
    # 断言转换后的结果与预期的空 DataFrame 相等
    tm.assert_frame_equal(result, expected[0:0])
@pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)
# 使用 pytest 的标记来忽略特定的警告信息
@pytest.mark.parametrize(
    "breaks",
    [[0.0, 1.0, 2.0, 3.0], pd.date_range("2017", periods=4, freq="D")],
    # 参数化测试，分别传入浮点数列表和日期时间列表
    ids=["float", "datetime64[ns]"],
)
def test_arrow_table_roundtrip_without_metadata(breaks):
    # 导入 pyarrow 库，如果不存在则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 从断点列表创建 IntervalArray
    arr = IntervalArray.from_breaks(breaks)
    arr[1] = None
    # 使用 IntervalArray 创建 Pandas DataFrame
    df = pd.DataFrame({"a": arr})

    # 使用 pyarrow 将 DataFrame 转换为 Table
    table = pa.table(df)
    # 移除 Table 的元数据
    table = table.replace_schema_metadata()
    # 断言 Table 的元数据为 None
    assert table.schema.metadata is None

    # 将 Table 转换回 Pandas DataFrame
    result = table.to_pandas()
    # 断言结果 DataFrame 的 'a' 列数据类型为 pd.IntervalDtype
    assert isinstance(result["a"].dtype, pd.IntervalDtype)
    # 使用 Pandas 测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, df)


def test_from_arrow_from_raw_struct_array():
    # 在 pyarrow 丢失 Interval 扩展类型时（例如在 Parquet 往返过程中使用 datetime64[ns] 子类型，见 GH-45881），仍然允许从 arrow 转换为 IntervalArray
    pa = pytest.importorskip("pyarrow")

    # 使用 pyarrow 创建一个包含结构化字典的数组
    arr = pa.array([{"left": 0, "right": 1}, {"left": 1, "right": 2}])
    # 定义期望的 IntervalDtype 类型
    dtype = pd.IntervalDtype(np.dtype("int64"), closed="neither")

    # 使用 IntervalDtype 的 __from_arrow__ 方法从 arrow 数组中创建结果
    result = dtype.__from_arrow__(arr)
    # 创建期望的 IntervalArray
    expected = IntervalArray.from_breaks(
        np.array([0, 1, 2], dtype="int64"), closed="neither"
    )
    # 使用 Pandas 测试工具比较两个 ExtensionArray 是否相等
    tm.assert_extension_array_equal(result, expected)

    # 使用 chunked_array 方法处理 chunked array，并比较结果与期望值是否相等
    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)
```