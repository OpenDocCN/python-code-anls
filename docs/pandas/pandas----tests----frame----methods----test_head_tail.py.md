# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_head_tail.py`

```
import numpy as np  # 导入NumPy库，用于处理数值数据

from pandas import DataFrame  # 从Pandas库中导入DataFrame类，用于处理表格数据
import pandas._testing as tm  # 导入Pandas测试模块，提供了各种测试函数和工具


def test_head_tail_generic(index, frame_or_series):
    # GH#5370
    # 根据传入的frame_or_series类型确定对象的维度
    ndim = 2 if frame_or_series is DataFrame else 1
    # 计算对象的形状
    shape = (len(index),) * ndim
    # 生成符合正态分布的随机数据
    vals = np.random.default_rng(2).standard_normal(shape)
    # 根据随机数据和索引创建DataFrame或Series对象
    obj = frame_or_series(vals, index=index)

    # 测试头部数据是否与iloc方法切片的前5行数据相等
    tm.assert_equal(obj.head(), obj.iloc[:5])
    # 测试尾部数据是否与iloc方法切片的后5行数据相等
    tm.assert_equal(obj.tail(), obj.iloc[-5:])

    # 测试头部数据是否与空切片操作相等
    tm.assert_equal(obj.head(0), obj.iloc[0:0])
    # 测试尾部数据是否与空切片操作相等
    tm.assert_equal(obj.tail(0), obj.iloc[0:0])

    # 测试头部数据是否与包含全部数据的切片操作相等
    tm.assert_equal(obj.head(len(obj) + 1), obj)
    # 测试尾部数据是否与包含全部数据的切片操作相等
    tm.assert_equal(obj.tail(len(obj) + 1), obj)

    # 测试头部数据是否与负索引操作得到的数据相等
    tm.assert_equal(obj.head(-3), obj.head(len(index) - 3))
    # 测试尾部数据是否与负索引操作得到的数据相等
    tm.assert_equal(obj.tail(-3), obj.tail(len(index) - 3))


def test_head_tail(float_frame):
    # 测试DataFrame头部数据是否与切片操作得到的前5行数据相等
    tm.assert_frame_equal(float_frame.head(), float_frame[:5])
    # 测试DataFrame尾部数据是否与切片操作得到的后5行数据相等
    tm.assert_frame_equal(float_frame.tail(), float_frame[-5:])

    # 测试DataFrame头部数据是否与空切片操作得到的数据相等
    tm.assert_frame_equal(float_frame.head(0), float_frame[0:0])
    # 测试DataFrame尾部数据是否与空切片操作得到的数据相等
    tm.assert_frame_equal(float_frame.tail(0), float_frame[0:0])

    # 测试DataFrame头部数据是否与负索引操作得到的数据相等
    tm.assert_frame_equal(float_frame.head(-1), float_frame[:-1])
    # 测试DataFrame尾部数据是否与负索引操作得到的数据相等
    tm.assert_frame_equal(float_frame.tail(-1), float_frame[1:])
    # 测试DataFrame头部数据是否与切片操作得到的前1行数据相等
    tm.assert_frame_equal(float_frame.head(1), float_frame[:1])
    # 测试DataFrame尾部数据是否与切片操作得到的最后1行数据相等
    tm.assert_frame_equal(float_frame.tail(1), float_frame[-1:])
    # 使用浮点索引测试
    df = float_frame.copy()
    df.index = np.arange(len(float_frame)) + 0.1
    # 测试DataFrame头部数据是否与iloc方法切片的前5行数据相等
    tm.assert_frame_equal(df.head(), df.iloc[:5])
    # 测试DataFrame尾部数据是否与iloc方法切片的后5行数据相等
    tm.assert_frame_equal(df.tail(), df.iloc[-5:])
    # 测试DataFrame头部数据是否与空切片操作得到的数据相等
    tm.assert_frame_equal(df.head(0), df[0:0])
    # 测试DataFrame尾部数据是否与空切片操作得到的数据相等
    tm.assert_frame_equal(df.tail(0), df[0:0])
    # 测试DataFrame头部数据是否与负索引操作得到的数据相等
    tm.assert_frame_equal(df.head(-1), df.iloc[:-1])
    # 测试DataFrame尾部数据是否与负索引操作得到的数据相等
    tm.assert_frame_equal(df.tail(-1), df.iloc[1:])


def test_head_tail_empty():
    # 测试空DataFrame的头部和尾部是否与自身相等
    empty_df = DataFrame()
    tm.assert_frame_equal(empty_df.tail(), empty_df)
    tm.assert_frame_equal(empty_df.head(), empty_df)
```