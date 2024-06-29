# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_constructors.py`

```
import numpy as np  # 导入 NumPy 库，通常用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行单元测试

from pandas import (  # 从 Pandas 库中导入以下模块：
    Categorical,  # 用于处理分类数据的类
    CategoricalDtype,  # 分类数据类型的定义类
    CategoricalIndex,  # 分类索引类
    Index,  # 通用的索引类
)
import pandas._testing as tm  # 导入 Pandas 内部的测试模块

class TestCategoricalIndexConstructors:
    def test_construction_disallows_scalar(self):
        msg = "must be called with a collection of some kind"  # 错误消息的文本
        # 使用 pytest 检查下面的代码块是否会引发 TypeError，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            CategoricalIndex(data=1, categories=list("abcd"), ordered=False)
        # 同样使用 pytest 检查下面的代码块是否会引发 TypeError，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            CategoricalIndex(categories=list("abcd"), ordered=False)
    # 测试 CategoricalIndex 类的构造函数和相关功能
    def test_construction(self):
        # 创建一个 CategoricalIndex 对象 ci，使用字符列表 "aabbca" 作为数据，字符列表 "abcd" 作为类别，无序
        ci = CategoricalIndex(list("aabbca"), categories=list("abcd"), ordered=False)
        # 获取 CategoricalIndex 对象 ci 的类别列表
        categories = ci.categories

        # 使用 CategoricalIndex 对象 ci 创建一个 Index 对象 result，并验证它们相等
        result = Index(ci)
        tm.assert_index_equal(result, ci, exact=True)
        # 验证 result 不是有序的
        assert not result.ordered

        # 使用 CategoricalIndex 对象 ci 的值创建一个 Index 对象 result，并验证它们相等
        result = Index(ci.values)
        tm.assert_index_equal(result, ci, exact=True)
        # 验证 result 不是有序的
        assert not result.ordered

        # 创建一个空的 CategoricalIndex 对象 result，类别为之前创建的 categories 列表
        result = CategoricalIndex([], categories=categories)
        # 验证 result 的类别列表与 Index(categories) 相等
        tm.assert_index_equal(result.categories, Index(categories))
        # 验证 result 的 codes 数组为空数组
        tm.assert_numpy_array_equal(result.codes, np.array([], dtype="int8"))
        # 验证 result 不是有序的
        assert not result.ordered

        # 创建一个 CategoricalIndex 对象 result，使用字符列表 "aabbca"，类别为之前创建的 categories 列表
        result = CategoricalIndex(list("aabbca"), categories=categories)
        # 验证 result 的类别列表与 Index(categories) 相等
        tm.assert_index_equal(result.categories, Index(categories))
        # 验证 result 的 codes 数组与预期一致
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )

        # 创建一个 Categorical 对象 c，使用字符列表 "aabbca"
        c = Categorical(list("aabbca"))
        # 使用 Categorical 对象 c 创建一个 CategoricalIndex 对象 result
        result = CategoricalIndex(c)
        # 验证 result 的类别列表与 Index(list("abc")) 相等
        tm.assert_index_equal(result.categories, Index(list("abc")))
        # 验证 result 的 codes 数组与预期一致
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )
        # 验证 result 不是有序的
        assert not result.ordered

        # 使用 Categorical 对象 c 创建一个 CategoricalIndex 对象 result，类别为之前创建的 categories 列表
        result = CategoricalIndex(c, categories=categories)
        # 验证 result 的类别列表与 Index(categories) 相等
        tm.assert_index_equal(result.categories, Index(categories))
        # 验证 result 的 codes 数组与预期一致
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )
        # 验证 result 不是有序的
        assert not result.ordered

        # 使用 CategoricalIndex 对象 ci，类别列表为 "abcd" 创建一个新的 CategoricalIndex 对象 ci
        ci = CategoricalIndex(ci, categories=list("abcd"))
        # 使用 CategoricalIndex 对象 ci 创建一个 CategoricalIndex 对象 result
        result = CategoricalIndex(ci)
        # 验证 result 的类别列表与 Index(categories) 相等
        tm.assert_index_equal(result.categories, Index(categories))
        # 验证 result 的 codes 数组与预期一致
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, 2, 0], dtype="int8")
        )
        # 验证 result 不是有序的
        assert not result.ordered

        # 使用 CategoricalIndex 对象 ci，类别列表为 "ab" 创建一个新的 CategoricalIndex 对象 result
        result = CategoricalIndex(ci, categories=list("ab"))
        # 验证 result 的类别列表与 Index(list("ab")) 相等
        tm.assert_index_equal(result.categories, Index(list("ab")))
        # 验证 result 的 codes 数组与预期一致
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, -1, 0], dtype="int8")
        )
        # 验证 result 不是有序的
        assert not result.ordered

        # 使用 CategoricalIndex 对象 ci，类别列表为 "ab"，并指定有序性创建一个新的 CategoricalIndex 对象 result
        result = CategoricalIndex(ci, categories=list("ab"), ordered=True)
        # 验证 result 的类别列表与 Index(list("ab")) 相等
        tm.assert_index_equal(result.categories, Index(list("ab")))
        # 验证 result 的 codes 数组与预期一致
        tm.assert_numpy_array_equal(
            result.codes, np.array([0, 0, 1, 1, -1, 0], dtype="int8")
        )
        # 验证 result 是有序的
        assert result.ordered

        # 使用 CategoricalIndex 对象 ci，类别列表为 "ab"，并指定有序性创建一个新的 CategoricalIndex 对象 result
        result = CategoricalIndex(ci, categories=list("ab"), ordered=True)
        # 创建一个预期的 CategoricalIndex 对象 expected，与 result 相同的参数，指定 dtype="category"
        expected = CategoricalIndex(
            ci, categories=list("ab"), ordered=True, dtype="category"
        )
        # 验证 result 与预期的 expected 相等
        tm.assert_index_equal(result, expected, exact=True)

        # 将 CategoricalIndex 对象 ci 转换为一个 Index 对象 result
        result = Index(np.array(ci))
        # 验证 result 是 Index 类型的对象
        assert isinstance(result, Index)
        # 验证 result 不是 CategoricalIndex 类型的对象
        assert not isinstance(result, CategoricalIndex)
    def test_construction_with_dtype(self):
        # 指定数据类型为 CategoricalIndex
        ci = CategoricalIndex(list("aabbca"), categories=list("abc"), ordered=False)

        # 使用 np.array 创建 Index 对象，并指定数据类型为 'category'
        result = Index(np.array(ci), dtype="category")
        # 断言 result 与 ci 相等，确保精确匹配
        tm.assert_index_equal(result, ci, exact=True)

        # 将 np.array(ci) 转换为列表后创建 Index 对象，指定数据类型为 'category'
        result = Index(np.array(ci).tolist(), dtype="category")
        # 断言 result 与 ci 相等，确保精确匹配
        tm.assert_index_equal(result, ci, exact=True)

        # 当类别重新排序时，通常这两者相等
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)

        # 使用 np.array 创建 Index 对象，并使用 reorder_categories 方法重新排序
        result = Index(np.array(ci), dtype="category").reorder_categories(ci.categories)
        # 断言 result 与 ci 相等，确保精确匹配
        tm.assert_index_equal(result, ci, exact=True)

        # 确保处理索引时的情况
        idx = Index(range(3))
        # 期望的 CategoricalIndex 对象，使用 Index 对象作为类别参数，有序
        expected = CategoricalIndex([0, 1, 2], categories=idx, ordered=True)
        # 使用 Index 对象创建 CategoricalIndex 对象，指定类别和有序性
        result = CategoricalIndex(idx, categories=idx, ordered=True)
        # 断言 result 与 expected 相等，确保精确匹配
        tm.assert_index_equal(result, expected, exact=True)

    def test_construction_empty_with_bool_categories(self):
        # 参考 GitHub issue #22702
        # 创建一个空的 CategoricalIndex 对象，指定类别为布尔值 [True, False]
        cat = CategoricalIndex([], categories=[True, False])
        # 将类别列表排序
        categories = sorted(cat.categories.tolist())
        # 断言排序后的类别列表为 [False, True]
        assert categories == [False, True]

    def test_construction_with_categorical_dtype(self):
        # 使用 CategoricalDtype 进行构造
        # 参考 GitHub issue #18109
        data, cats, ordered = "a a b b".split(), "c b a".split(), True
        # 创建 CategoricalDtype 对象，指定类别和有序性
        dtype = CategoricalDtype(categories=cats, ordered=ordered)

        # 使用 data 和 dtype 创建 CategoricalIndex 对象
        result = CategoricalIndex(data, dtype=dtype)
        # 期望的 CategoricalIndex 对象，使用指定的类别和有序性
        expected = CategoricalIndex(data, categories=cats, ordered=ordered)
        # 断言 result 与 expected 相等，确保精确匹配
        tm.assert_index_equal(result, expected, exact=True)

        # 另一个 GitHub issue #19032
        # 使用 data 和 dtype 创建 Index 对象
        result = Index(data, dtype=dtype)
        # 断言 result 与 expected 相等，确保精确匹配
        tm.assert_index_equal(result, expected, exact=True)

        # 在同时指定 categories/ordered 和 dtype 参数时，应当报错
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."
        # 确保在同时使用 categories/ordered 和 dtype 参数时抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            CategoricalIndex(data, categories=cats, dtype=dtype)

        with pytest.raises(ValueError, match=msg):
            CategoricalIndex(data, ordered=ordered, dtype=dtype)
```