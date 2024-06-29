# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_util.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵的数学运算
import pytest  # 导入 Pytest 库，用于编写和运行测试

from pandas import (  # 从 Pandas 库中导入特定模块和函数
    Index,  # 导入 Index 类，用于处理索引对象
    date_range,  # 导入 date_range 函数，用于生成日期范围
)
import pandas._testing as tm  # 导入 Pandas 内部的测试工具模块
from pandas.core.reshape.util import cartesian_product  # 导入 Pandas 中的笛卡尔积函数


class TestCartesianProduct:
    def test_simple(self):
        x, y = list("ABC"), [1, 22]  # 定义两个输入列表 x 和 y
        result1, result2 = cartesian_product([x, y])  # 调用 cartesian_product 函数计算笛卡尔积
        expected1 = np.array(["A", "A", "B", "B", "C", "C"])  # 预期的第一个结果数组
        expected2 = np.array([1, 22, 1, 22, 1, 22])  # 预期的第二个结果数组
        tm.assert_numpy_array_equal(result1, expected1)  # 断言结果1与预期1相等
        tm.assert_numpy_array_equal(result2, expected2)  # 断言结果2与预期2相等

    def test_datetimeindex(self):
        # GitHub 问题 #6439 的回归测试
        # 确保 DateTimeIndex 的顺序是一致的
        x = date_range("2000-01-01", periods=2)  # 生成一个日期范围对象 x
        result1, result2 = (Index(y).day for y in cartesian_product([x, x]))  # 计算两个日期范围对象的笛卡尔积
        expected1 = Index([1, 1, 2, 2], dtype=np.int32)  # 预期的第一个结果索引对象
        expected2 = Index([1, 2, 1, 2], dtype=np.int32)  # 预期的第二个结果索引对象
        tm.assert_index_equal(result1, expected1)  # 断言结果1与预期1相等
        tm.assert_index_equal(result2, expected2)  # 断言结果2与预期2相等

    def test_tzaware_retained(self):
        x = date_range("2000-01-01", periods=2, tz="US/Pacific")  # 生成一个带时区信息的日期范围对象 x
        y = np.array([3, 4])  # 定义一个 NumPy 数组 y
        result1, result2 = cartesian_product([x, y])  # 计算 x 和 y 的笛卡尔积
        expected = x.repeat(2)  # 预期的结果，重复 x 两次
        tm.assert_index_equal(result1, expected)  # 断言结果1与预期相等

    def test_tzaware_retained_categorical(self):
        x = date_range("2000-01-01", periods=2, tz="US/Pacific").astype("category")  # 生成一个带时区信息的日期范围对象，并转换为分类类型
        y = np.array([3, 4])  # 定义一个 NumPy 数组 y
        result1, result2 = cartesian_product([x, y])  # 计算 x 和 y 的笛卡尔积
        expected = x.repeat(2)  # 预期的结果，重复 x 两次
        tm.assert_index_equal(result1, expected)  # 断言结果1与预期相等

    @pytest.mark.parametrize("x, y", [[[], []], [[0, 1], []], [[], ["a", "b", "c"]]])
    def test_empty(self, x, y):
        # 空因子的笛卡尔积
        expected1 = np.array([], dtype=np.asarray(x).dtype)  # 预期的第一个结果为空数组
        expected2 = np.array([], dtype=np.asarray(y).dtype)  # 预期的第二个结果为空数组
        result1, result2 = cartesian_product([x, y])  # 计算空因子的笛卡尔积
        tm.assert_numpy_array_equal(result1, expected1)  # 断言结果1与预期1相等
        tm.assert_numpy_array_equal(result2, expected2)  # 断言结果2与预期2相等

    def test_empty_input(self):
        # 空的产品（空输入）：
        result = cartesian_product([])  # 计算空输入的笛卡尔积
        expected = []  # 预期的结果为空列表
        assert result == expected  # 断言结果与预期相等

    @pytest.mark.parametrize(
        "X", [1, [1], [1, 2], [[1], 2], "a", ["a"], ["a", "b"], [["a"], "b"]]
    )
    def test_invalid_input(self, X):
        msg = "Input must be a list-like of list-likes"  # 错误消息内容

        with pytest.raises(TypeError, match=msg):
            cartesian_product(X=X)  # 断言调用 cartesian_product 函数时会抛出 TypeError 异常，且异常信息匹配预期的消息内容

    def test_exceed_product_space(self):
        # GH31355: 当生成空间过大时，抛出有用的错误信息
        msg = "Product space too large to allocate arrays!"  # 错误消息内容

        dims = [np.arange(0, 22, dtype=np.int16) for i in range(12)] + [
            (np.arange(15128, dtype=np.int16)),
        ]  # 定义一个维度列表，其中包含多个数组，用于测试大型笛卡尔积的边界情况
        with pytest.raises(ValueError, match=msg):
            cartesian_product(X=dims)  # 断言调用 cartesian_product 函数时会抛出 ValueError 异常，且异常信息匹配预期的消息内容
```