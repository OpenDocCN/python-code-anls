# `D:\src\scipysrc\scipy\scipy\stats\tests\test_censored_data.py`

```
# Tests for the CensoredData class.

import pytest  # 导入 pytest 模块，用于编写和运行测试
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_equal, assert_array_equal  # 导入 NumPy 的断言函数
from scipy.stats import CensoredData  # 导入 CensoredData 类，用于处理被审查的数据


class TestCensoredData:

    def test_basic(self):
        uncensored = [1]  # 未被审查的数据
        left = [0]  # 左审查边界的数据
        right = [2, 5]  # 右审查边界的数据
        interval = [[2, 3]]  # 区间审查的数据
        # 创建 CensoredData 对象，传入不同类型的审查数据
        data = CensoredData(uncensored, left=left, right=right,
                            interval=interval)
        assert_equal(data._uncensored, uncensored)  # 断言未被审查数据正确存储
        assert_equal(data._left, left)  # 断言左审查边界数据正确存储
        assert_equal(data._right, right)  # 断言右审查边界数据正确存储
        assert_equal(data._interval, interval)  # 断言区间审查数据正确存储

        udata = data._uncensor()  # 执行去审查操作
        # 断言去审查后的数据与预期连接结果一致
        assert_equal(udata, np.concatenate((uncensored, left, right,
                                            np.mean(interval, axis=1))))

    def test_right_censored(self):
        x = np.array([0, 3, 2.5])  # 数据数组
        is_censored = np.array([0, 1, 0], dtype=bool)  # 数据的审查状态
        # 调用 right_censored 方法创建 CensoredData 对象
        data = CensoredData.right_censored(x, is_censored)
        assert_equal(data._uncensored, x[~is_censored])  # 断言未被审查数据正确存储
        assert_equal(data._right, x[is_censored])  # 断言右审查边界数据正确存储
        assert_equal(data._left, [])  # 断言左审查边界数据为空
        assert_equal(data._interval, np.empty((0, 2)))  # 断言区间审查数据为空数组

    def test_left_censored(self):
        x = np.array([0, 3, 2.5])  # 数据数组
        is_censored = np.array([0, 1, 0], dtype=bool)  # 数据的审查状态
        # 调用 left_censored 方法创建 CensoredData 对象
        data = CensoredData.left_censored(x, is_censored)
        assert_equal(data._uncensored, x[~is_censored])  # 断言未被审查数据正确存储
        assert_equal(data._left, x[is_censored])  # 断言左审查边界数据正确存储
        assert_equal(data._right, [])  # 断言右审查边界数据为空
        assert_equal(data._interval, np.empty((0, 2)))  # 断言区间审查数据为空数组

    def test_interval_censored_basic(self):
        a = [0.5, 2.0, 3.0, 5.5]  # 区间审查的下限
        b = [1.0, 2.5, 3.5, 7.0]  # 区间审查的上限
        # 调用 interval_censored 方法创建 CensoredData 对象
        data = CensoredData.interval_censored(low=a, high=b)
        assert_array_equal(data._interval, np.array(list(zip(a, b))))  # 断言区间审查数据正确存储
        assert data._uncensored.shape == (0,)  # 断言未被审查数据为空数组
        assert data._left.shape == (0,)  # 断言左审查边界数据为空数组
        assert data._right.shape == (0,)  # 断言右审查边界数据为空数组

    def test_interval_censored_mixed(self):
        # 这个测试包含未被审查、左审查、右审查和区间审查的混合数据。
        # 检查当使用 interval_censored 类方法时，数据是否正确分离到相应的数组中。
        a = [0.5, -np.inf, -13.0, 2.0, 1.0, 10.0, -1.0]  # 区间审查的下限
        b = [0.5, 2500.0, np.inf, 3.0, 1.0, 11.0, np.inf]  # 区间审查的上限
        # 调用 interval_censored 方法创建 CensoredData 对象
        data = CensoredData.interval_censored(low=a, high=b)
        assert_array_equal(data._interval, [[2.0, 3.0], [10.0, 11.0]])  # 断言区间审查数据正确存储
        assert_array_equal(data._uncensored, [0.5, 1.0])  # 断言未被审查数据正确存储
        assert_array_equal(data._left, [2500.0])  # 断言左审查边界数据正确存储
        assert_array_equal(data._right, [-13.0, -1.0])  # 断言右审查边界数据正确存储
    def test_interval_to_other_types(self):
        # interval 参数可以表示未被审查的、左-或右被审查的数据。测试将这样一个例子转换为规范形式，
        # 其中不同类型已分割成不同的数组。
        interval = np.array([[0, 1],        # 区间被审查
                             [2, 2],        # 未被审查
                             [3, 3],        # 未被审查
                             [9, np.inf],   # 右被审查
                             [8, np.inf],   # 右被审查
                             [-np.inf, 0],  # 左被审查
                             [1, 2]])       # 区间被审查
        data = CensoredData(interval=interval)
        assert_equal(data._uncensored, [2, 3])
        assert_equal(data._left, [0])
        assert_equal(data._right, [9, 8])
        assert_equal(data._interval, [[0, 1], [1, 2]])

    def test_empty_arrays(self):
        data = CensoredData(uncensored=[], left=[], right=[], interval=[])
        assert data._uncensored.shape == (0,)
        assert data._left.shape == (0,)
        assert data._right.shape == (0,)
        assert data._interval.shape == (0, 2)
        assert len(data) == 0

    def test_invalid_constructor_args(self):
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(uncensored=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(left=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a one-dimensional'):
            CensoredData(right=[[1, 2, 3]])
        with pytest.raises(ValueError, match='must be a two-dimensional'):
            CensoredData(interval=[[1, 2, 3]])

        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(uncensored=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(left=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(right=[1, np.nan, 2])
        with pytest.raises(ValueError, match='must not contain nan'):
            CensoredData(interval=[[1, np.nan], [2, 3]])

        with pytest.raises(ValueError,
                           match='both values must not be infinite'):
            CensoredData(interval=[[1, 3], [2, 9], [np.inf, np.inf]])

        with pytest.raises(ValueError,
                           match='left value must not exceed the right'):
            CensoredData(interval=[[1, 0], [2, 2]])

    @pytest.mark.parametrize('func', [CensoredData.left_censored,
                                      CensoredData.right_censored])


注释：
第一个函数 `test_interval_to_other_types` 测试了将不同类型的被审查数据转换为规范形式的功能。其中，`interval` 是一个包含不同类型数据的 NumPy 数组。第二个函数 `test_empty_arrays` 测试了当给定空数组时，`CensoredData` 对象的预期行为。最后一个函数 `test_invalid_constructor_args` 测试了在不同情况下，`CensoredData` 构造函数对于无效参数的异常处理。
    # 测试函数，用于检查传递给函数的参数是否合法（左右截尾参数无效）
    def test_invalid_left_right_censored_args(self, func):
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `x` 是否为一维数组
        with pytest.raises(ValueError,
                           match='`x` must be one-dimensional'):
            func([[1, 2, 3]], [0, 1, 1])
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `censored` 是否为一维数组
        with pytest.raises(ValueError,
                           match='`censored` must be one-dimensional'):
            func([1, 2, 3], [[0, 1, 1]])
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `x` 是否包含 NaN 值
        with pytest.raises(ValueError, match='`x` must not contain'):
            func([1, 2, np.nan], [0, 1, 1])
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `x` 和 `censored` 是否具有相同的长度
        with pytest.raises(ValueError, match='must have the same length'):
            func([1, 2, 3], [0, 0, 1, 1])

    # 测试函数，用于检查传递给函数的参数是否合法（截尾参数无效）
    def test_invalid_censored_args(self):
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `low` 是否为一维数组
        with pytest.raises(ValueError,
                           match='`low` must be a one-dimensional'):
            CensoredData.interval_censored(low=[[3]], high=[4, 5])
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `high` 是否为一维数组
        with pytest.raises(ValueError,
                           match='`high` must be a one-dimensional'):
            CensoredData.interval_censored(low=[3], high=[[4, 5]])
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `low` 是否包含 NaN 值
        with pytest.raises(ValueError, match='`low` must not contain'):
            CensoredData.interval_censored([1, 2, np.nan], [0, 1, 1])
        # 使用 pytest 的断言检查是否会引发 ValueError 异常，检查 `low` 和 `high` 是否具有相同的长度
        with pytest.raises(ValueError, match='must have the same length'):
            CensoredData.interval_censored([1, 2, 3], [0, 0, 1, 1])

    # 测试函数，用于检查计数截尾数据的方法
    def test_count_censored(self):
        # 定义一个列表 x
        x = [1, 2, 3]
        # 创建 CensoredData 类的实例 data1，传入无截尾数据
        data1 = CensoredData(x)
        # 断言 data1 实例中截尾数据的数量为 0
        assert data1.num_censored() == 0
        # 创建 CensoredData 类的实例 data2，传入截尾数据
        data2 = CensoredData(uncensored=[2.5], left=[10], interval=[[0, 1]])
        # 断言 data2 实例中截尾数据的数量为 2
        assert data2.num_censored() == 2
```