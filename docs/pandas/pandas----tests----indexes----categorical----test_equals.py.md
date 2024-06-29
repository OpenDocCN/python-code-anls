# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_equals.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行单元测试

from pandas import (  # 从pandas库中导入以下模块：
    Categorical,  # 用于处理分类数据的类
    CategoricalIndex,  # 分类数据的索引类
    Index,  # 用于一般索引的类
    MultiIndex,  # 多级索引的类
)


class TestEquals:
    def test_equals_categorical(self):
        ci1 = CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)  # 创建有序的分类索引对象ci1
        ci2 = CategoricalIndex(["a", "b"], categories=["a", "b", "c"], ordered=True)  # 创建有序的分类索引对象ci2

        assert ci1.equals(ci1)  # 断言ci1等于自身
        assert not ci1.equals(ci2)  # 断言ci1不等于ci2
        assert ci1.equals(ci1.astype(object))  # 断言ci1与其对象类型转换后的对象相等
        assert ci1.astype(object).equals(ci1)  # 断言ci1对象类型转换后的对象与ci1相等

        assert (ci1 == ci1).all()  # 断言ci1中所有元素与自身相等
        assert not (ci1 != ci1).all()  # 断言ci1中所有元素与自身不不相等
        assert not (ci1 > ci1).all()  # 断言ci1中所有元素不大于自身
        assert not (ci1 < ci1).all()  # 断言ci1中所有元素不小于自身
        assert (ci1 <= ci1).all()  # 断言ci1中所有元素小于或等于自身
        assert (ci1 >= ci1).all()  # 断言ci1中所有元素大于或等于自身

        assert not (ci1 == 1).all()  # 断言ci1中所有元素不等于1
        assert (ci1 == Index(["a", "b"])).all()  # 断言ci1与包含["a", "b"]的索引对象相等
        assert (ci1 == ci1.values).all()  # 断言ci1与其值数组相等

        # invalid comparisons
        with pytest.raises(ValueError, match="Lengths must match"):  # 断言引发值错误异常，且异常消息匹配"Lengths must match"
            ci1 == Index(["a", "b", "c"])

        msg = "Categoricals can only be compared if 'categories' are the same"
        with pytest.raises(TypeError, match=msg):  # 断言引发类型错误异常，且异常消息匹配msg
            ci1 == ci2
        with pytest.raises(TypeError, match=msg):  # 断言引发类型错误异常，且异常消息匹配msg
            ci1 == Categorical(ci1.values, ordered=False)
        with pytest.raises(TypeError, match=msg):  # 断言引发类型错误异常，且异常消息匹配msg
            ci1 == Categorical(ci1.values, categories=list("abc"))

        # tests
        # make sure that we are testing for category inclusion properly
        ci = CategoricalIndex(list("aabca"), categories=["c", "a", "b"])  # 创建分类索引对象ci
        assert not ci.equals(list("aabca"))  # 断言ci不等于包含字符的列表
        # Same categories, but different order
        # Unordered
        assert ci.equals(CategoricalIndex(list("aabca")))  # 断言ci等于另一个具有相同分类的分类索引对象
        # Ordered
        assert not ci.equals(CategoricalIndex(list("aabca"), ordered=True))  # 断言ci不等于另一个具有相同分类和有序属性的分类索引对象
        assert ci.equals(ci.copy())  # 断言ci等于其自身的复制对象

        ci = CategoricalIndex(list("aabca") + [np.nan], categories=["c", "a", "b"])  # 创建包含NaN的分类索引对象ci
        assert not ci.equals(list("aabca"))  # 断言ci不等于包含字符的列表
        assert not ci.equals(CategoricalIndex(list("aabca")))  # 断言ci不等于具有相同分类的另一个分类索引对象
        assert ci.equals(ci.copy())  # 断言ci等于其自身的复制对象

        ci = CategoricalIndex(list("aabca") + [np.nan], categories=["c", "a", "b"])  # 创建包含NaN的分类索引对象ci
        assert not ci.equals(list("aabca") + [np.nan])  # 断言ci不等于包含字符和NaN的列表
        assert ci.equals(CategoricalIndex(list("aabca") + [np.nan]))  # 断言ci等于具有相同分类和NaN的分类索引对象
        assert not ci.equals(CategoricalIndex(list("aabca") + [np.nan], ordered=True))  # 断言ci不等于具有相同分类、NaN和有序属性的分类索引对象
        assert ci.equals(ci.copy())  # 断言ci等于其自身的复制对象

    def test_equals_categorical_unordered(self):
        # https://github.com/pandas-dev/pandas/issues/16603
        a = CategoricalIndex(["A"], categories=["A", "B"])  # 创建具有'A'分类的分类索引对象a
        b = CategoricalIndex(["A"], categories=["B", "A"])  # 创建具有'A'分类但顺序不同的分类索引对象b
        c = CategoricalIndex(["C"], categories=["B", "A"])  # 创建具有'C'分类的分类索引对象c
        assert a.equals(b)  # 断言a等于b
        assert not a.equals(c)  # 断言a不等于c
        assert not b.equals(c)  # 断言b不等于c
    # 定义测试函数，用于测试 CategoricalIndex 的 equals 方法在非分类数据中的行为
    def test_equals_non_category(self):
        # 创建一个包含四个元素的 CategoricalIndex，包括两个 np.nan
        ci = CategoricalIndex(["A", "B", np.nan, np.nan])
        # 创建一个包含四个元素的普通 Index，包括一个不在 ci 中的值 "D" 和两个 np.nan
        other = Index(["A", "B", "D", np.nan])

        # 断言 ci 不等于 other
        assert not ci.equals(other)

    # 定义测试函数，用于测试 CategoricalIndex 的 equals 方法在 MultiIndex 中的行为
    def test_equals_multiindex(self):
        # 创建一个 MultiIndex，包含两个级别，第一个级别是 ["A", "B", "C", "D"]，第二个级别是 [0, 1, 2, 3]
        mi = MultiIndex.from_arrays([["A", "B", "C", "D"], range(4)])
        # 将 MultiIndex 转换为扁平化索引，并将其类型转换为 "category"
        ci = mi.to_flat_index().astype("category")

        # 断言 ci 不等于原始的 MultiIndex mi
        assert not ci.equals(mi)

    # 定义测试函数，用于测试 CategoricalIndex 的 equals 方法在字符串类型数据中的行为
    def test_equals_string_dtype(self, any_string_dtype):
        # 创建一个具有名称 "B" 的 CategoricalIndex，包含字符列表 ["a", "b", "c"]
        idx = CategoricalIndex(list("abc"), name="B")
        # 创建一个具有名称 "B"、dtype 为 any_string_dtype 的普通 Index，包含字符串列表 ["a", "b", "c"]
        other = Index(["a", "b", "c"], name="B", dtype=any_string_dtype)

        # 断言 idx 等于 other
        assert idx.equals(other)
```