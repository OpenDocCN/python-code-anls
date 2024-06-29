# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_matmul.py`

```
import operator  # 导入 Python 中的 operator 模块，用于进行运算符操作

import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 Pytest 测试框架，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入 DataFrame, Index, Series 类
    DataFrame,  # Pandas 中的二维表格数据结构
    Index,  # Pandas 中的索引对象
    Series,  # Pandas 中的一维标签数组
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块，用于测试辅助

class TestMatMul:  # 定义测试类 TestMatMul
    def test_matmul(self):  # 定义测试方法 test_matmul
        # matmul test is for GH#10259
        # 准备测试数据
        a = DataFrame(  # 创建 DataFrame 对象 a，填充随机标准正态分布数据
            np.random.default_rng(2).standard_normal((3, 4)),
            index=["a", "b", "c"],
            columns=["p", "q", "r", "s"],
        )
        b = DataFrame(  # 创建 DataFrame 对象 b，填充随机标准正态分布数据
            np.random.default_rng(2).standard_normal((4, 2)),
            index=["p", "q", "r", "s"],
            columns=["one", "two"],
        )

        # DataFrame @ DataFrame
        result = operator.matmul(a, b)  # 执行 DataFrame 对象之间的矩阵乘法运算
        expected = DataFrame(  # 创建期望结果的 DataFrame 对象
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)  # 使用测试辅助工具检验结果是否符合期望

        # DataFrame @ Series
        result = operator.matmul(a, b.one)  # 执行 DataFrame 与 Series 的矩阵乘法运算
        expected = Series(  # 创建期望结果的 Series 对象
            np.dot(a.values, b.one.values), index=["a", "b", "c"]
        )
        tm.assert_series_equal(result, expected)  # 使用测试辅助工具检验结果是否符合期望

        # np.array @ DataFrame
        result = operator.matmul(a.values, b)  # 执行 NumPy 数组与 DataFrame 的矩阵乘法运算
        assert isinstance(result, DataFrame)  # 检查结果是否为 DataFrame 类型
        assert result.columns.equals(b.columns)  # 检查结果的列是否与 b 的列相等
        assert result.index.equals(Index(range(3)))  # 检查结果的索引是否符合预期
        expected = np.dot(a.values, b.values)  # 计算期望结果
        tm.assert_almost_equal(result.values, expected)  # 使用测试辅助工具检验结果是否近似相等

        # nested list @ DataFrame (__rmatmul__)
        result = operator.matmul(a.values.tolist(), b)  # 执行嵌套列表与 DataFrame 的矩阵乘法运算（反向乘法）
        expected = DataFrame(  # 创建期望结果的 DataFrame 对象
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_almost_equal(result.values, expected.values)  # 使用测试辅助工具检验结果是否近似相等

        # mixed dtype DataFrame @ DataFrame
        a["q"] = a.q.round().astype(int)  # 修改 DataFrame a 的列 q，将其四舍五入并转换为整数类型
        result = operator.matmul(a, b)  # 执行数据类型不同的 DataFrame 与 DataFrame 的矩阵乘法运算
        expected = DataFrame(  # 创建期望结果的 DataFrame 对象
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)  # 使用测试辅助工具检验结果是否符合期望

        # different dtypes DataFrame @ DataFrame
        a = a.astype(int)  # 将 DataFrame a 中的所有数据类型转换为整数类型
        result = operator.matmul(a, b)  # 执行数据类型完全不同的 DataFrame 与 DataFrame 的矩阵乘法运算
        expected = DataFrame(  # 创建期望结果的 DataFrame 对象
            np.dot(a.values, b.values), index=["a", "b", "c"], columns=["one", "two"]
        )
        tm.assert_frame_equal(result, expected)  # 使用测试辅助工具检验结果是否符合期望

        # unaligned
        df = DataFrame(  # 创建 DataFrame 对象 df，填充随机标准正态分布数据
            np.random.default_rng(2).standard_normal((3, 4)),
            index=[1, 2, 3],
            columns=range(4),
        )
        df2 = DataFrame(  # 创建 DataFrame 对象 df2，填充随机标准正态分布数据
            np.random.default_rng(2).standard_normal((5, 3)),
            index=range(5),
            columns=[1, 2, 3],
        )

        with pytest.raises(ValueError, match="aligned"):  # 使用 Pytest 的断言检查是否引发值错误异常
            operator.matmul(df, df2)  # 执行不对齐的 DataFrame 与 DataFrame 的矩阵乘法运算
    def test_matmul_message_shapes(self):
        # 定义测试方法，验证矩阵乘法异常消息反映原始形状，
        # 而非转置后的形状
        # 创建一个 10x4 的随机数矩阵 a
        a = np.random.default_rng(2).random((10, 4))
        # 使用相同种子创建一个 5x3 的随机数矩阵 b
        b = np.random.default_rng(2).random((5, 3))

        # 将矩阵 b 转换为 pandas 的 DataFrame 对象
        df = DataFrame(b)

        # 预期的异常消息，反映矩阵乘法时形状不匹配
        msg = r"shapes \(10, 4\) and \(5, 3\) not aligned"
        
        # 使用 pytest 检查异常，验证矩阵 a 与 DataFrame df 的乘法
        with pytest.raises(ValueError, match=msg):
            a @ df
        # 将矩阵 a 转换为列表后再与 DataFrame df 做乘法，同样验证异常
        with pytest.raises(ValueError, match=msg):
            a.tolist() @ df
```