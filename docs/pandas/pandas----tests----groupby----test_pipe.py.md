# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_pipe.py`

```
# 导入必要的库：numpy 和 pandas
import numpy as np
import pandas as pd
from pandas import (
    DataFrame,
    Index,
)
import pandas._testing as tm

# 定义一个测试函数 test_pipe
def test_pipe():
    # 测试 DataFrameGroupBy 的 pipe 方法。
    # 这是解决问题 #17871 的一个例子。

    # 设定随机数生成器的种子
    random_state = np.random.default_rng(2)

    # 创建一个 DataFrame 对象 df，包含三列：A、B、C
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": random_state.standard_normal(8),  # 列 B 包含随机标准正态分布的数据
            "C": random_state.standard_normal(8),  # 列 C 包含随机标准正态分布的数据
        }
    )

    # 定义一个函数 f，输入为 DataFrameGroupBy 对象 dfgb，返回值为计算 B 列的最大值与 C 列最小值的差的最小值
    def f(dfgb):
        return dfgb.B.max() - dfgb.C.min().min()

    # 定义一个函数 square，输入为 Series 对象 srs，返回值为每个元素的平方
    def square(srs):
        return srs ** 2

    # 使用 groupby 方法按列 A 对 df 进行分组，然后依次调用 f 和 square 函数
    # 注意这里的转换链条是 GroupBy -> Series -> Series
    # 这样链接了 GroupBy.pipe 和 NDFrame.pipe 方法
    result = df.groupby("A").pipe(f).pipe(square)

    # 生成预期的 Series 对象 expected，包含了按 A 列分组后 B 列的预期结果
    index = Index(["bar", "foo"], dtype="object", name="A")
    expected = pd.Series([3.749306591013693, 6.717707873081384], name="B", index=index)

    # 使用 assert_series_equal 函数验证 result 和 expected 是否相等
    tm.assert_series_equal(expected, result)


def test_pipe_args():
    # 测试向 DataFrameGroupBy 的 pipe 方法传递参数。
    # 这是解决问题 #17871 的一个例子。

    # 创建一个 DataFrame 对象 df，包含三列：group、x、y
    df = DataFrame(
        {
            "group": ["A", "A", "B", "B", "C"],
            "x": [1.0, 2.0, 3.0, 2.0, 5.0],
            "y": [10.0, 100.0, 1000.0, -100.0, -1000.0],
        }
    )

    # 定义一个函数 f，输入为 DataFrameGroupBy 对象 dfgb 和参数 arg1，返回值为按组过滤后的 DataFrameGroupBy 对象
    def f(dfgb, arg1):
        filtered = dfgb.filter(lambda grp: grp.y.mean() > arg1, dropna=False)
        return filtered.groupby("group")

    # 定义一个函数 g，输入为 DataFrameGroupBy 对象 dfgb 和参数 arg2，返回值为计算 dfgb.sum() 后的比例并加上 arg2 的结果
    def g(dfgb, arg2):
        return dfgb.sum() / dfgb.sum().sum() + arg2

    # 定义一个函数 h，输入为 DataFrame 对象 df 和参数 arg3，返回值为计算 x 列加上 y 列再减去 arg3 的结果
    def h(df, arg3):
        return df.x + df.y - arg3

    # 使用 groupby 方法按 group 列对 df 进行分组，依次调用 f、g、h 函数，并传递相应的参数
    result = df.groupby("group").pipe(f, 0).pipe(g, 10).pipe(h, 100)

    # 生成预期的 Series 对象 expected，包含了按 group 列分组后计算结果的预期值
    index = Index(["A", "B"], name="group")
    expected = pd.Series([-79.5160891089, -78.4839108911], index=index)

    # 使用 assert_series_equal 函数验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 测试 SeriesGroupby 的 pipe 方法
    ser = pd.Series([1, 1, 2, 2, 3, 3])
    # 使用 lambda 函数作为参数，对 ser 进行分组并计算每组的和乘以组大小的结果
    result = ser.groupby(ser).pipe(lambda grp: grp.sum() * grp.count())

    # 生成预期的 Series 对象 expected，包含了按组计算结果的预期值
    expected = pd.Series([4, 8, 12], index=Index([1, 2, 3], dtype=np.int64))

    # 使用 assert_series_equal 函数验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```