# `D:\src\scipysrc\pandas\pandas\tests\frame\test_npfuncs.py`

```
"""
Tests for np.foo applied to DataFrame, not necessarily ufuncs.
"""

# 导入需要的库和模块
import numpy as np

# 从 pandas 库中导入特定的类和函数
from pandas import (
    Categorical,
    DataFrame,
)
import pandas._testing as tm

# 定义测试类 TestAsArray
class TestAsArray:
    # 测试方法：测试将 DataFrame 转换为数组时，处理同质数据的情况
    def test_asarray_homogeneous(self):
        # 创建一个 DataFrame，包含两列分别为 Categorical 类型的数据
        df = DataFrame({"A": Categorical([1, 2]), "B": Categorical([1, 2])})
        # 将 DataFrame 转换为数组
        result = np.asarray(df)
        # 期望的数组结果，包含同样的数据，数据类型为 object
        expected = np.array([[1, 1], [2, 2]], dtype="object")
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected)

    # 测试方法：测试 np.sqrt 函数在 DataFrame 上的应用
    def test_np_sqrt(self, float_frame):
        # 在忽略所有错误状态下计算 DataFrame 的平方根
        with np.errstate(all="ignore"):
            result = np.sqrt(float_frame)
        # 断言结果的类型与原始 DataFrame 类型一致
        assert isinstance(result, type(float_frame))
        # 断言结果的行索引与原始 DataFrame 的行索引一致
        assert result.index.is_(float_frame.index)
        # 断言结果的列索引与原始 DataFrame 的列索引一致
        assert result.columns.is_(float_frame.columns)
        # 断言 DataFrame 的平方根计算结果与使用 np.sqrt 应用的结果一致
        tm.assert_frame_equal(result, float_frame.apply(np.sqrt))

    # 测试方法：测试 np.sum 在指定轴上的行为
    def test_sum_axis_behavior(self):
        # 创建一个随机数组成的 DataFrame
        arr = np.random.default_rng(2).standard_normal((4, 3))
        df = DataFrame(arr)

        # 对 DataFrame 沿着所有轴的元素求和
        res = np.sum(df)
        # 计算期望的总和
        expected = df.to_numpy().sum(axis=None)
        # 断言结果与期望一致
        assert res == expected

    # 测试方法：测试 np.ravel 函数的行为
    def test_np_ravel(self):
        # 创建一个二维数组
        arr = np.array(
            [
                [0.11197053, 0.44361564, -0.92589452],
                [0.05883648, -0.00948922, -0.26469934],
            ]
        )

        # 使用 np.ravel 将 DataFrame 批量扁平化
        result = np.ravel([DataFrame(batch.reshape(1, 3)) for batch in arr])
        # 期望的扁平化结果
        expected = np.array(
            [
                0.11197053,
                0.44361564,
                -0.92589452,
                0.05883648,
                -0.00948922,
                -0.26469934,
            ]
        )
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected)

        # 使用 np.ravel 将指定列名的 DataFrame 扁平化
        result = np.ravel(DataFrame(arr[0].reshape(1, 3), columns=["x1", "x2", "x3"]))
        # 期望的扁平化结果
        expected = np.array([0.11197053, 0.44361564, -0.92589452])
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected)

        # 使用 np.ravel 将批量包含指定列名的 DataFrame 扁平化
        result = np.ravel(
            [
                DataFrame(batch.reshape(1, 3), columns=["x1", "x2", "x3"])
                for batch in arr
            ]
        )
        # 期望的扁平化结果
        expected = np.array(
            [
                0.11197053,
                0.44361564,
                -0.92589452,
                0.05883648,
                -0.00948922,
                -0.26469934,
            ]
        )
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected)
```