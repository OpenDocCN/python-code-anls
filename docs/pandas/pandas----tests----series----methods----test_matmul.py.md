# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_matmul.py`

```
import operator  # 导入操作符模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

from pandas import (  # 从pandas库中导入DataFrame和Series
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入pandas测试模块

class TestMatmul:
    def test_matmul(self):
        # matmul test is for GH#10259
        # 创建一个Series对象a，包含4个标准正态分布的随机数，指定索引
        a = Series(
            np.random.default_rng(2).standard_normal(4), index=["p", "q", "r", "s"]
        )
        # 创建一个DataFrame对象b，包含3行4列的标准正态分布随机数矩阵，指定行索引和列索引，并转置
        b = DataFrame(
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["1", "2", "3"],
            columns=["p", "q", "r", "s"],
        ).T

        # Series @ DataFrame -> Series
        result = operator.matmul(a, b)  # 使用matmul操作符计算Series @ DataFrame的结果
        expected = Series(np.dot(a.values, b.values), index=["1", "2", "3"])  # 计算期望结果
        tm.assert_series_equal(result, expected)  # 断言结果与期望结果相等

        # DataFrame @ Series -> Series
        result = operator.matmul(b.T, a)  # 使用matmul操作符计算DataFrame @ Series的结果
        expected = Series(np.dot(b.T.values, a.T.values), index=["1", "2", "3"])  # 计算期望结果
        tm.assert_series_equal(result, expected)  # 断言结果与期望结果相等

        # Series @ Series -> scalar
        result = operator.matmul(a, a)  # 使用matmul操作符计算Series @ Series的结果
        expected = np.dot(a.values, a.values)  # 计算期望结果
        tm.assert_almost_equal(result, expected)  # 断言结果与期望结果近似相等

        # GH#21530
        # vector (1D np.array) @ Series (__rmatmul__)
        result = operator.matmul(a.values, a)  # 使用matmul操作符计算向量与Series的结果
        expected = np.dot(a.values, a.values)  # 计算期望结果
        tm.assert_almost_equal(result, expected)  # 断言结果与期望结果近似相等

        # GH#21530
        # vector (1D list) @ Series (__rmatmul__)
        result = operator.matmul(a.values.tolist(), a)  # 使用matmul操作符计算列表和Series的结果
        expected = np.dot(a.values, a.values)  # 计算期望结果
        tm.assert_almost_equal(result, expected)  # 断言结果与期望结果近似相等

        # GH#21530
        # matrix (2D np.array) @ Series (__rmatmul__)
        result = operator.matmul(b.T.values, a)  # 使用matmul操作符计算矩阵和Series的结果
        expected = np.dot(b.T.values, a.values)  # 计算期望结果
        tm.assert_almost_equal(result, expected)  # 断言结果与期望结果近似相等

        # GH#21530
        # matrix (2D nested lists) @ Series (__rmatmul__)
        result = operator.matmul(b.T.values.tolist(), a)  # 使用matmul操作符计算嵌套列表和Series的结果
        expected = np.dot(b.T.values, a.values)  # 计算期望结果
        tm.assert_almost_equal(result, expected)  # 断言结果与期望结果近似相等

        # mixed dtype DataFrame @ Series
        a["p"] = int(a.p)  # 将Series中索引为'p'的元素转换为整数
        result = operator.matmul(b.T, a)  # 使用matmul操作符计算混合数据类型的DataFrame @ Series的结果
        expected = Series(np.dot(b.T.values, a.T.values), index=["1", "2", "3"])  # 计算期望结果
        tm.assert_series_equal(result, expected)  # 断言结果与期望结果相等

        # different dtypes DataFrame @ Series
        a = a.astype(int)  # 将Series的数据类型转换为整数类型
        result = operator.matmul(b.T, a)  # 使用matmul操作符计算不同数据类型的DataFrame @ Series的结果
        expected = Series(np.dot(b.T.values, a.T.values), index=["1", "2", "3"])  # 计算期望结果
        tm.assert_series_equal(result, expected)  # 断言结果与期望结果相等

        msg = r"Dot product shape mismatch, \(4,\) vs \(3,\)"  # 设置异常消息
        # exception raised is of type Exception
        with pytest.raises(Exception, match=msg):  # 断言引发异常，并匹配异常消息
            a.dot(a.values[:3])  # 使用dot方法计算向量点积
        msg = "matrices are not aligned"  # 设置异常消息
        with pytest.raises(ValueError, match=msg):  # 断言引发异常，并匹配异常消息
            a.dot(b.T)  # 使用dot方法计算矩阵点积
```