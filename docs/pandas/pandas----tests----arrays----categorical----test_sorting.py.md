# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_sorting.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 Pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入以下类和函数
    Categorical,  # 用于处理分类数据的类
    Index,  # Pandas 数据结构中的索引类
)
import pandas._testing as tm  # 导入 Pandas 测试模块作为 tm 别名


class TestCategoricalSort:  # 定义一个测试类 TestCategoricalSort
    def test_argsort(self):  # 定义测试方法 test_argsort
        c = Categorical([5, 3, 1, 4, 2], ordered=True)  # 创建有序分类对象 c

        expected = np.array([2, 4, 1, 3, 0])  # 期望的排序结果数组
        tm.assert_numpy_array_equal(  # 使用 Pandas 测试模块验证 NumPy 数组相等性
            c.argsort(ascending=True), expected, check_dtype=False  # 验证按升序排序的结果
        )

        expected = expected[::-1]  # 将期望的排序结果数组反转
        tm.assert_numpy_array_equal(  # 再次验证 NumPy 数组相等性
            c.argsort(ascending=False), expected, check_dtype=False  # 验证按降序排序的结果
        )

    def test_numpy_argsort(self):  # 定义测试方法 test_numpy_argsort
        c = Categorical([5, 3, 1, 4, 2], ordered=True)  # 创建有序分类对象 c

        expected = np.array([2, 4, 1, 3, 0])  # 期望的排序结果数组
        tm.assert_numpy_array_equal(np.argsort(c), expected, check_dtype=False)  # 验证使用 NumPy 的 argsort 函数的结果

        tm.assert_numpy_array_equal(  # 再次验证 NumPy 数组相等性
            np.argsort(c, kind="mergesort"), expected, check_dtype=False  # 验证指定 mergesort 排序方法的结果
        )

        msg = "the 'axis' parameter is not supported"  # 错误消息：不支持 'axis' 参数
        with pytest.raises(ValueError, match=msg):  # 使用 Pytest 断言引发 ValueError 异常，且异常消息匹配特定模式
            np.argsort(c, axis=0)  # 尝试使用 'axis' 参数调用 argsort 函数

        msg = "the 'order' parameter is not supported"  # 错误消息：不支持 'order' 参数
        with pytest.raises(ValueError, match=msg):  # 使用 Pytest 断言引发 ValueError 异常，且异常消息匹配特定模式
            np.argsort(c, order="C")  # 尝试使用 'order' 参数调用 argsort 函数

    def test_sort_values(self):  # 定义测试方法 test_sort_values
        # 无序分类对象可排序
        cat = Categorical(["a", "b", "b", "a"], ordered=False)  # 创建无序分类对象 cat
        cat.sort_values()  # 对分类对象进行排序

        cat = Categorical(["a", "c", "b", "d"], ordered=True)  # 创建有序分类对象 cat

        # sort_values 方法的测试
        res = cat.sort_values()  # 对有序分类对象进行排序，返回排序后的结果
        exp = np.array(["a", "b", "c", "d"], dtype=object)  # 期望的排序结果数组
        tm.assert_numpy_array_equal(res.__array__(), exp)  # 验证排序结果的 NumPy 数组相等性
        tm.assert_index_equal(res.categories, cat.categories)  # 验证排序结果的索引相等性

        cat = Categorical(  # 创建有序分类对象 cat，指定其类别和顺序
            ["a", "c", "b", "d"], categories=["a", "b", "c", "d"], ordered=True
        )
        res = cat.sort_values()  # 对有序分类对象进行排序
        exp = np.array(["a", "b", "c", "d"], dtype=object)  # 期望的排序结果数组
        tm.assert_numpy_array_equal(res.__array__(), exp)  # 验证排序结果的 NumPy 数组相等性
        tm.assert_index_equal(res.categories, cat.categories)  # 验证排序结果的索引相等性

        res = cat.sort_values(ascending=False)  # 对有序分类对象进行降序排序
        exp = np.array(["d", "c", "b", "a"], dtype=object)  # 期望的降序排序结果数组
        tm.assert_numpy_array_equal(res.__array__(), exp)  # 验证降序排序结果的 NumPy 数组相等性
        tm.assert_index_equal(res.categories, cat.categories)  # 验证排序结果的索引相等性

        # sort 方法（原地排序）
        cat1 = cat.copy()  # 复制分类对象 cat 到 cat1
        orig_codes = cat1._codes  # 获取原始代码（分类的内部表示）
        cat1.sort_values(inplace=True)  # 在原地对分类对象进行排序
        assert cat1._codes is orig_codes  # 验证原地排序后的分类代码不变
        exp = np.array(["a", "b", "c", "d"], dtype=object)  # 期望的排序结果数组
        tm.assert_numpy_array_equal(cat1.__array__(), exp)  # 验证原地排序结果的 NumPy 数组相等性
        tm.assert_index_equal(res.categories, cat.categories)  # 验证排序结果的索引相等性

        # reverse 排序
        cat = Categorical(["a", "c", "c", "b", "d"], ordered=True)  # 创建有序分类对象 cat
        res = cat.sort_values(ascending=False)  # 对有序分类对象进行降序排序
        exp_val = np.array(["d", "c", "c", "b", "a"], dtype=object)  # 期望的降序排序结果数组
        exp_categories = Index(["a", "b", "c", "d"])  # 期望的排序后的类别索引
        tm.assert_numpy_array_equal(res.__array__(), exp_val)  # 验证降序排序结果的 NumPy 数组相等性
        tm.assert_index_equal(res.categories, exp_categories)  # 验证排序结果的类别索引相等性
    def test_sort_values_na_position(self):
        # 根据 GitHub issue gh-12882 编写的测试函数，测试分类变量的排序功能

        # 创建一个有序的分类变量对象，包括整数和缺失值
        cat = Categorical([5, 2, np.nan, 2, np.nan], ordered=True)
        # 预期的分类值列表
        exp_categories = Index([2, 5])

        # 预期的排序结果，包含缺失值在最后
        exp = np.array([2.0, 2.0, 5.0, np.nan, np.nan])
        # 执行默认参数的排序操作
        res = cat.sort_values()
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        # 预期的排序结果，缺失值在首位
        exp = np.array([np.nan, np.nan, 2.0, 2.0, 5.0])
        # 执行指定参数的排序操作，升序排列，缺失值在首位
        res = cat.sort_values(ascending=True, na_position="first")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        # 预期的排序结果，缺失值在首位，降序排列
        exp = np.array([np.nan, np.nan, 5.0, 2.0, 2.0])
        # 执行指定参数的排序操作，降序排列，缺失值在首位
        res = cat.sort_values(ascending=False, na_position="first")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        # 预期的排序结果，缺失值在末尾，升序排列
        exp = np.array([2.0, 2.0, 5.0, np.nan, np.nan])
        # 执行指定参数的排序操作，升序排列，缺失值在末尾
        res = cat.sort_values(ascending=True, na_position="last")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        # 预期的排序结果，缺失值在末尾，降序排列
        exp = np.array([5.0, 2.0, 2.0, np.nan, np.nan])
        # 执行指定参数的排序操作，降序排列，缺失值在末尾
        res = cat.sort_values(ascending=False, na_position="last")
        tm.assert_numpy_array_equal(res.__array__(), exp)
        tm.assert_index_equal(res.categories, exp_categories)

        # 创建一个有序的分类变量对象，包括字符串和缺失值
        cat = Categorical(["a", "c", "b", "d", np.nan], ordered=True)
        # 执行降序排列，缺失值在末尾
        res = cat.sort_values(ascending=False, na_position="last")
        # 预期的排序结果，包含缺失值在末尾
        exp_val = np.array(["d", "c", "b", "a", np.nan], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)

        # 执行降序排列，缺失值在首位
        res = cat.sort_values(ascending=False, na_position="first")
        # 预期的排序结果，缺失值在首位
        exp_val = np.array([np.nan, "d", "c", "b", "a"], dtype=object)
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_numpy_array_equal(res.__array__(), exp_val)
        tm.assert_index_equal(res.categories, exp_categories)
```