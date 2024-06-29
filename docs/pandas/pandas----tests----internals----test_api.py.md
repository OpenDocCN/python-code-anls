# `D:\src\scipysrc\pandas\pandas\tests\internals\test_api.py`

```
"""
Tests for the pseudo-public API implemented in internals/api.py and exposed
in core.internals
"""

# 导入必要的模块
import datetime  # 日期时间操作
import numpy as np  # 数组操作
import pytest  # 测试框架

import pandas as pd  # 数据分析库
import pandas._testing as tm  # 测试辅助工具
from pandas.api.internals import create_dataframe_from_blocks  # 从数据块创建数据框
from pandas.core import internals  # Pandas 内部功能
from pandas.core.internals import api  # Pandas 内部 API

# 测试 internals 模块中的 make_block 函数是否等于 api 模块中的 make_block 函数
def test_internals_api():
    assert internals.make_block is api.make_block


# 测试 internals 模块的命名空间是否包含预期的模块和函数
def test_namespace():
    # SUBJECT TO CHANGE

    modules = [
        "blocks",  # 模块列表中应包含 "blocks"
        "concat",  # 模块列表中应包含 "concat"
        "managers",  # 模块列表中应包含 "managers"
        "construction",  # 模块列表中应包含 "construction"
        "api",  # 模块列表中应包含 "api"
        "ops",  # 模块列表中应包含 "ops"
    ]
    expected = [
        "make_block",  # 预期包含函数 "make_block"
        "BlockManager",  # 预期包含类 "BlockManager"
        "SingleBlockManager",  # 预期包含类 "SingleBlockManager"
        "concatenate_managers",  # 预期包含函数 "concatenate_managers"
    ]

    # 获取 internals 模块中除私有成员外的所有成员，以列表形式存储在 result 中
    result = [x for x in dir(internals) if not x.startswith("__")]
    assert set(result) == set(expected + modules)


# 测试 make_block 函数在处理带有日期时间索引的二维数据时是否产生警告并返回正确的数据块
def test_make_block_2d_with_dti():
    # GH#41168
    dti = pd.date_range("2012", periods=3, tz="UTC")

    msg = "make_block is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        blk = api.make_block(dti, placement=[0])

    assert blk.shape == (1, 3)
    assert blk.values.shape == (1, 3)


# 测试从数据块创建数据框的函数 create_dataframe_from_blocks 是否按预期工作
def test_create_dataframe_from_blocks(float_frame):
    block = float_frame._mgr.blocks[0]
    index = float_frame.index.copy()
    columns = float_frame.columns.copy()

    result = create_dataframe_from_blocks(
        [(block.values, block.mgr_locs.as_array)], index=index, columns=columns
    )
    tm.assert_frame_equal(result, float_frame)


# 测试从数据块创建数据框的函数 create_dataframe_from_blocks 在处理不同数据类型时是否按预期工作
def test_create_dataframe_from_blocks_types():
    df = pd.DataFrame(
        {
            "int": list(range(1, 4)),
            "uint": np.arange(3, 6).astype("uint8"),
            "float": [2.0, np.nan, 3.0],
            "bool": np.array([True, False, True]),
            "boolean": pd.array([True, False, None], dtype="boolean"),
            "string": list("abc"),
            "datetime": pd.date_range("20130101", periods=3),
            "datetimetz": pd.date_range("20130101", periods=3).tz_localize(
                "Europe/Brussels"
            ),
            "timedelta": pd.timedelta_range("1 day", periods=3),
            "period": pd.period_range("2012-01-01", periods=3, freq="D"),
            "categorical": pd.Categorical(["a", "b", "a"]),
            "interval": pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (3, 4)]),
        }
    )

    result = create_dataframe_from_blocks(
        [(block.values, block.mgr_locs.as_array) for block in df._mgr.blocks],
        index=df.index,
        columns=df.columns,
    )
    tm.assert_frame_equal(result, df)


# 测试从数据块创建数据框的函数 create_dataframe_from_blocks 在处理日期时间相关数据时是否按预期工作
def test_create_dataframe_from_blocks_datetimelike():
    # extension dtypes that have an exact matching numpy dtype can also be
    # be passed as a numpy array
    index, columns = pd.RangeIndex(3), pd.Index(["a", "b", "c", "d"])

    block_array1 = np.arange(
        datetime.datetime(2020, 1, 1),
        datetime.datetime(2020, 1, 7),
        step=datetime.timedelta(1),
    ).reshape((2, 3))
    # 使用 NumPy 的 arange 函数生成一个二维数组 block_array2，表示时间间隔为 1 天到 7 天，步长为 1 天，reshape 成 (2, 3) 的形状
    block_array2 = np.arange(
        datetime.timedelta(1), datetime.timedelta(7), step=datetime.timedelta(1)
    ).reshape((2, 3))
    
    # 调用 create_dataframe_from_blocks 函数创建 DataFrame，传入两个数据块及其对应的索引和列信息
    result = create_dataframe_from_blocks(
        [(block_array1, np.array([0, 2])), (block_array2, np.array([1, 3]))],
        index=index,
        columns=columns,
    )
    
    # 创建预期的 DataFrame 对象 expected，包含四列：a, b, c, d，每列使用不同的时间序列生成方式
    expected = pd.DataFrame(
        {
            "a": pd.date_range("2020-01-01", periods=3, unit="us"),
            "b": pd.timedelta_range("1 days", periods=3, unit="us"),
            "c": pd.date_range("2020-01-04", periods=3, unit="us"),
            "d": pd.timedelta_range("4 days", periods=3, unit="us"),
        }
    )
    
    # 使用 assert_frame_equal 检查 result 和 expected 是否相等，以确认生成的 DataFrame 结果正确性
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "array",
    [  # 使用 pytest 的 parametrize 标记，定义参数化测试，传入不同的日期数组
        pd.date_range("2020-01-01", periods=3),  # 创建一个包含三个日期的日期范围
        pd.date_range("2020-01-01", periods=3, tz="UTC"),  # 创建一个带有时区的日期范围
        pd.period_range("2012-01-01", periods=3, freq="D"),  # 创建一个周期范围，每天频率
        pd.timedelta_range("1 day", periods=3),  # 创建一个时间间隔范围，每天间隔
    ],
)
def test_create_dataframe_from_blocks_1dEA(array):
    # ExtensionArrays can be passed as 1D even if stored under the hood as 2D
    # 创建一个 DataFrame，将参数化测试中的日期数组作为列 'a' 的值
    df = pd.DataFrame({"a": array})

    # 从 DataFrame 的内部管理器中获取第一个数据块
    block = df._mgr.blocks[0]
    
    # 调用函数 create_dataframe_from_blocks，将数据块的值和位置信息作为元组列表传入
    # 创建一个新的 DataFrame，使用与原始 DataFrame 相同的索引和列名
    result = create_dataframe_from_blocks(
        [(block.values[0], block.mgr_locs.as_array)], index=df.index, columns=df.columns
    )
    
    # 使用 pytest 中的 assert_frame_equal 断言函数比较 result 和原始 DataFrame df 是否相等
    tm.assert_frame_equal(result, df)
```