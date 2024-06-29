# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_reindex.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    Categorical,  # 用于处理分类数据的类
    CategoricalIndex,  # 用于处理分类索引的类
    Index,  # 用于创建索引的类
    Interval,  # 用于表示间隔数据的类
)
import pandas._testing as tm  # 导入 pandas 内部测试模块，用于执行测试断言

class TestReindex:
    def test_reindex_list_non_unique(self):
        # 测试用例: GH#11586
        msg = "cannot reindex on an axis with duplicate labels"  # 设置预期错误信息
        ci = CategoricalIndex(["a", "b", "c", "a"])  # 创建一个包含重复标签的分类索引
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言预期抛出 ValueError 异常，并匹配指定的错误信息
            ci.reindex(["a", "c"])  # 调用 reindex 方法进行索引重建操作

    def test_reindex_categorical_non_unique(self):
        msg = "cannot reindex on an axis with duplicate labels"  # 设置预期错误信息
        ci = CategoricalIndex(["a", "b", "c", "a"])  # 创建一个包含重复标签的分类索引
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言预期抛出 ValueError 异常，并匹配指定的错误信息
            ci.reindex(Categorical(["a", "c"]))  # 调用 reindex 方法进行索引重建操作

    def test_reindex_list_non_unique_unused_category(self):
        msg = "cannot reindex on an axis with duplicate labels"  # 设置预期错误信息
        ci = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])  # 创建一个包含未使用分类的分类索引
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言预期抛出 ValueError 异常，并匹配指定的错误信息
            ci.reindex(["a", "c"])  # 调用 reindex 方法进行索引重建操作

    def test_reindex_categorical_non_unique_unused_category(self):
        msg = "cannot reindex on an axis with duplicate labels"  # 设置预期错误信息
        ci = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])  # 创建一个包含未使用分类的分类索引
        with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言预期抛出 ValueError 异常，并匹配指定的错误信息
            ci.reindex(Categorical(["a", "c"]))  # 调用 reindex 方法进行索引重建操作

    def test_reindex_duplicate_target(self):
        # 测试用例: See GH25459
        cat = CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c", "d"])  # 创建一个分类索引对象
        res, indexer = cat.reindex(["a", "c", "c"])  # 调用 reindex 方法对索引进行重建操作
        exp = Index(["a", "c", "c"])  # 创建预期的索引对象
        tm.assert_index_equal(res, exp, exact=True)  # 使用测试工具函数断言索引对象相等
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))  # 使用测试工具函数断言索引器数组相等

        res, indexer = cat.reindex(
            CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        )  # 调用 reindex 方法对分类索引进行重建操作
        exp = CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])  # 创建预期的分类索引对象
        tm.assert_index_equal(res, exp, exact=True)  # 使用测试工具函数断言分类索引对象相等
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))  # 使用测试工具函数断言索引器数组相等

    def test_reindex_empty_index(self):
        # 测试用例: See GH16770
        c = CategoricalIndex([])  # 创建一个空的分类索引对象
        res, indexer = c.reindex(["a", "b"])  # 调用 reindex 方法对索引进行重建操作
        tm.assert_index_equal(res, Index(["a", "b"]), exact=True)  # 使用测试工具函数断言索引对象相等
        tm.assert_numpy_array_equal(indexer, np.array([-1, -1], dtype=np.intp))  # 使用测试工具函数断言索引器数组相等

    def test_reindex_categorical_added_category(self):
        # GH 42424
        ci = CategoricalIndex(
            [Interval(0, 1, closed="right"), Interval(1, 2, closed="right")],
            ordered=True,
        )  # 创建一个包含间隔数据的有序分类索引对象
        ci_add = CategoricalIndex(
            [
                Interval(0, 1, closed="right"),
                Interval(1, 2, closed="right"),
                Interval(2, 3, closed="right"),
                Interval(3, 4, closed="right"),
            ],
            ordered=True,
        )  # 创建一个包含扩展间隔数据的有序分类索引对象
        result, _ = ci.reindex(ci_add)  # 调用 reindex 方法对分类索引进行重建操作
        expected = ci_add  # 设置预期的分类索引对象
        tm.assert_index_equal(expected, result)  # 使用测试工具函数断言分类索引对象相等
```