# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_api.py`

```
# 导入所需的模块
import re  # 导入正则表达式模块
import numpy as np  # 导入NumPy模块，并使用别名np
import pytest  # 导入pytest测试框架

from pandas.compat import PY311  # 从pandas兼容模块中导入PY311常量

# 从pandas库中导入多个类和函数
from pandas import (
    Categorical,  # 导入Categorical类
    CategoricalIndex,  # 导入CategoricalIndex类
    DataFrame,  # 导入DataFrame类
    Index,  # 导入Index类
    Series,  # 导入Series类
    StringDtype,  # 导入StringDtype类
)

import pandas._testing as tm  # 导入pandas测试模块，并使用别名tm
from pandas.core.arrays.categorical import recode_for_categories  # 从pandas核心数组的categorical模块中导入recode_for_categories函数


class TestCategoricalAPI:
    def test_ordered_api(self):
        # GH 9347
        # 创建一个无序的Categorical对象cat1，包含字符列表['a', 'c', 'b']
        cat1 = Categorical(list("acb"), ordered=False)
        # 断言cat1的categories属性与Index(["a", "b", "c"])相等
        tm.assert_index_equal(cat1.categories, Index(["a", "b", "c"]))
        # 断言cat1不是有序的
        assert not cat1.ordered

        # 创建一个无序的Categorical对象cat2，包含字符列表['a', 'c', 'b']，并指定自定义的categories和无序
        cat2 = Categorical(list("acb"), categories=list("bca"), ordered=False)
        # 断言cat2的categories属性与Index(["b", "c", "a"])相等
        tm.assert_index_equal(cat2.categories, Index(["b", "c", "a"]))
        # 断言cat2不是有序的
        assert not cat2.ordered

        # 创建一个有序的Categorical对象cat3，包含字符列表['a', 'c', 'b']
        cat3 = Categorical(list("acb"), ordered=True)
        # 断言cat3的categories属性与Index(["a", "b", "c"])相等
        tm.assert_index_equal(cat3.categories, Index(["a", "b", "c"]))
        # 断言cat3是有序的
        assert cat3.ordered

        # 创建一个有序的Categorical对象cat4，包含字符列表['a', 'c', 'b']，并指定自定义的categories和有序
        cat4 = Categorical(list("acb"), categories=list("bca"), ordered=True)
        # 断言cat4的categories属性与Index(["b", "c", "a"])相等
        tm.assert_index_equal(cat4.categories, Index(["b", "c", "a"]))
        # 断言cat4是有序的
        assert cat4.ordered

    def test_set_ordered(self):
        # 创建一个有序的Categorical对象cat，包含字符列表['a', 'b', 'c', 'a']
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        # 调用as_unordered方法，返回一个无序的Categorical对象cat2，并断言其不是有序的
        cat2 = cat.as_unordered()
        assert not cat2.ordered
        # 再次调用as_ordered方法，返回一个有序的Categorical对象cat2，并断言其是有序的
        cat2 = cat.as_ordered()
        assert cat2.ordered

        # 调用set_ordered方法，传入True参数，返回一个有序的Categorical对象cat2，并断言其是有序的
        assert cat2.set_ordered(True).ordered
        # 调用set_ordered方法，传入False参数，返回一个无序的Categorical对象cat2，并断言其不是有序的
        assert not cat2.set_ordered(False).ordered

        # 在pandas 0.19.0版本中移除了以下操作
        msg = (
            "property 'ordered' of 'Categorical' object has no setter"
            if PY311
            else "can't set attribute"
        )
        # 使用pytest的raises断言捕获AttributeError异常，断言异常消息与msg匹配
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = True
        with pytest.raises(AttributeError, match=msg):
            cat.ordered = False

    def test_rename_categories(self):
        # 创建一个Categorical对象cat，包含字符列表['a', 'b', 'c', 'a']
        cat = Categorical(["a", "b", "c", "a"])

        # 调用rename_categories方法，传入新的categories列表[1, 2, 3]，返回一个新的Categorical对象res，并断言其数据与预期相等
        res = cat.rename_categories([1, 2, 3])
        tm.assert_numpy_array_equal(
            res.__array__(), np.array([1, 2, 3, 1], dtype=np.int64)
        )
        # 断言res的categories属性与Index([1, 2, 3])相等
        tm.assert_index_equal(res.categories, Index([1, 2, 3]))

        # 创建一个预期的字符列表exp_cat，用于验证cat对象数据未改变
        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        # 断言cat的数据与exp_cat相等
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # 创建一个预期的Index对象exp_cat，用于验证cat的categories属性未改变
        exp_cat = Index(["a", "b", "c"])
        # 断言cat的categories属性与exp_cat相等
        tm.assert_index_equal(cat.categories, exp_cat)

        # GH18862 (让rename_categories接受可调用对象)
        # 调用rename_categories方法，传入lambda函数，将所有字符转为大写，并返回结果result
        result = cat.rename_categories(lambda x: x.upper())
        # 创建一个期望的Categorical对象expected，包含字符列表['A', 'B', 'C', 'A']
        expected = Categorical(["A", "B", "C", "A"])
        # 使用tm.assert_categorical_equal断言result与expected相等
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("new_categories", [[1, 2, 3, 4], [1, 2]])
    def test_rename_categories_wrong_length_raises(self, new_categories):
        # 创建一个Categorical对象cat，包含字符列表['a', 'b', 'c', 'a']
        cat = Categorical(["a", "b", "c", "a"])
        # 创建一个错误消息字符串msg，用于验证pytest.raises捕获的异常消息
        msg = (
            "new categories need to have the same number of items as the "
            "old categories!"
        )
        # 使用pytest.raises断言捕获ValueError异常，断言异常消息与msg匹配
        with pytest.raises(ValueError, match=msg):
            cat.rename_categories(new_categories)
    # 测试用例：使用 Series 对象重命名分类数据
    def test_rename_categories_series(self):
        # GitHub 上的 issue 链接：https://github.com/pandas-dev/pandas/issues/17981
        # 创建一个分类变量 c，包含值 "a" 和 "b"
        c = Categorical(["a", "b"])
        # 使用 Series 对象作为新的分类值进行重命名
        result = c.rename_categories(Series([0, 1], index=["a", "b"]))
        # 期望的结果是一个包含整数值的 Categorical 对象
        expected = Categorical([0, 1])
        # 使用测试工具方法来比较两个分类对象是否相等
        tm.assert_categorical_equal(result, expected)

    # 测试用例：使用字典重命名分类数据
    def test_rename_categories_dict(self):
        # GitHub issue 17336
        # 创建一个包含值 "a", "b", "c", "d" 的分类变量 cat
        cat = Categorical(["a", "b", "c", "d"])
        # 使用字典来重命名分类变量的值
        res = cat.rename_categories({"a": 4, "b": 3, "c": 2, "d": 1})
        # 期望的结果是一个包含整数索引的 Index 对象
        expected = Index([4, 3, 2, 1])
        # 使用测试工具方法来比较结果的分类对象的 categories 与期望的 Index 是否相等
        tm.assert_index_equal(res.categories, expected)

        # 测试字典长度较小的情况
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 1, "c": 3})
        expected = Index([1, "b", 3, "d"])
        tm.assert_index_equal(res.categories, expected)

        # 测试字典长度较大的情况
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6})
        expected = Index([1, 2, 3, 4])
        tm.assert_index_equal(res.categories, expected)

        # 测试字典中不包含任何旧分类项的情况
        cat = Categorical(["a", "b", "c", "d"])
        res = cat.rename_categories({"f": 1, "g": 3})
        expected = Index(["a", "b", "c", "d"])
        tm.assert_index_equal(res.categories, expected)

    # 测试用例：重新排序分类变量的分类顺序
    def test_reorder_categories(self):
        # 创建一个有序的分类变量 cat，包含值 "a", "b", "c", "a"
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        # 备份原始的分类变量 cat
        old = cat.copy()
        # 创建一个新的分类变量 new，包含重新排序后的分类值
        new = Categorical(
            ["a", "b", "c", "a"], categories=["c", "b", "a"], ordered=True
        )
        # 对 cat 进行重新排序操作
        res = cat.reorder_categories(["c", "b", "a"])
        # 断言：cat 在重新排序后应该与原始值相同
        tm.assert_categorical_equal(cat, old)
        # 断言：只有 res 被修改为新值
        tm.assert_categorical_equal(res, new)

    # 参数化测试用例：测试重新排序分类变量时引发的异常情况
    @pytest.mark.parametrize(
        "new_categories",
        [
            ["a"],  # 新分类不包含所有旧分类
            ["a", "b", "d"],  # 新分类仍然不包含所有旧分类
            ["a", "b", "c", "d"],  # 新分类包含所有旧分类，但长度过长
        ],
    )
    def test_reorder_categories_raises(self, new_categories):
        # 创建一个有序的分类变量 cat，包含值 "a", "b", "c", "a"
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        # 期望引发异常的错误消息
        msg = "items in new_categories are not the same as in old categories"
        # 断言：当重排序的新分类不满足旧分类的条件时，应该引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            cat.reorder_categories(new_categories)
    # 定义测试方法，验证在有序情况下添加分类是否正确
    def test_add_categories(self):
        # 创建一个有序的分类对象，包含四个元素："a", "b", "c", "a"
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        # 备份当前分类对象
        old = cat.copy()
        # 创建一个新的有序分类对象，指定新的分类为 ["a", "b", "c", "d"]
        new = Categorical(
            ["a", "b", "c", "a"], categories=["a", "b", "c", "d"], ordered=True
        )

        # 在原始分类对象上添加新的分类 "d"，并验证不改变原始对象，返回新对象与预期的新分类对象相同
        res = cat.add_categories("d")
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        # 同样，在原始分类对象上添加多个新的分类 ["d"]，验证结果与预期相同
        res = cat.add_categories(["d"])
        tm.assert_categorical_equal(cat, old)
        tm.assert_categorical_equal(res, new)

        # GH 9927
        # 创建一个有序的分类对象，包含三个元素："a", "b", "c"
        cat = Categorical(list("abc"), ordered=True)
        # 预期的新分类对象，包含五个元素："a", "b", "c", "d", "e"
        expected = Categorical(list("abc"), categories=list("abcde"), ordered=True)
        # 使用 Series, np.array, Index, list 等不同类型的数据添加新的分类 ["d", "e"]，验证结果与预期相同
        res = cat.add_categories(Series(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(np.array(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(Index(["d", "e"]))
        tm.assert_categorical_equal(res, expected)
        res = cat.add_categories(["d", "e"])
        tm.assert_categorical_equal(res, expected)

    # 定义测试方法，验证添加已存在分类时是否触发异常
    def test_add_categories_existing_raises(self):
        # 创建一个有序的分类对象，包含四个元素："a", "b", "c", "d"
        cat = Categorical(["a", "b", "c", "d"], ordered=True)
        # 定义异常消息的正则表达式模式
        msg = re.escape("new categories must not include old categories: {'d'}")
        # 使用 pytest 检查在添加已存在的分类 ["d"] 时是否触发 ValueError 异常，异常消息与预期相匹配
        with pytest.raises(ValueError, match=msg):
            cat.add_categories(["d"])

    # 定义测试方法，验证添加分类是否能正确处理 dtype 信息丢失的情况
    def test_add_categories_losing_dtype_information(self):
        # GH#48812
        # 创建一个包含整数的 Series 对象，创建有序的整数分类对象
        cat = Categorical(Series([1, 2], dtype="Int64"))
        # 创建一个整数的 Series 对象
        ser = Series([4], dtype="Int64")
        # 使用 Series [4] 添加到分类对象中，验证结果与预期相同
        result = cat.add_categories(ser)
        expected = Categorical(
            Series([1, 2], dtype="Int64"), categories=Series([1, 2, 4], dtype="Int64")
        )
        tm.assert_categorical_equal(result, expected)

        # 创建一个包含字符串的 Series 对象，创建有序的字符串分类对象
        cat = Categorical(Series(["a", "b", "a"], dtype=StringDtype()))
        # 创建一个字符串的 Series 对象
        ser = Series(["d"], dtype=StringDtype())
        # 使用 Series ["d"] 添加到分类对象中，验证结果与预期相同
        result = cat.add_categories(ser)
        expected = Categorical(
            Series(["a", "b", "a"], dtype=StringDtype()),
            categories=Series(["a", "b", "d"], dtype=StringDtype()),
        )
        tm.assert_categorical_equal(result, expected)
    def test_set_categories(self):
        # 创建一个有序的分类变量对象，初始类别为 ["a", "b", "c", "a"]
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        # 预期的类别顺序为 ["c", "b", "a"]
        exp_categories = Index(["c", "b", "a"])
        # 预期的数值数组为 ["a", "b", "c", "a"]
        exp_values = np.array(["a", "b", "c", "a"], dtype=np.object_)

        # 修改分类变量的类别顺序为 ["c", "b", "a"]
        cat = cat.set_categories(["c", "b", "a"])
        # 修改后的结果应该与初始的预期类别顺序一致
        res = cat.set_categories(["a", "b", "c"])
        # 检查 cat 的类别是否与预期一致
        tm.assert_index_equal(cat.categories, exp_categories)
        # 检查 cat 转换为 numpy 数组后是否与预期一致
        tm.assert_numpy_array_equal(cat.__array__(), exp_values)
        # 检查 res 的类别是否与修改后的预期类别顺序一致
        exp_categories_back = Index(["a", "b", "c"])
        tm.assert_index_equal(res.categories, exp_categories_back)
        # 检查 res 转换为 numpy 数组后是否与预期一致
        tm.assert_numpy_array_equal(res.__array__(), exp_values)

        # 将所有不包含在新类别中的旧类别标记为 np.nan
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        res = cat.set_categories(["a"])
        tm.assert_numpy_array_equal(res.codes, np.array([0, -1, -1, 0], dtype=np.int8))

        # 仍然有旧类别不在新类别中
        res = cat.set_categories(["a", "b", "d"])
        tm.assert_numpy_array_equal(res.codes, np.array([0, 1, -1, 0], dtype=np.int8))
        tm.assert_index_equal(res.categories, Index(["a", "b", "d"]))

        # 所有旧类别都包含在新类别中
        cat = cat.set_categories(["a", "b", "c", "d"])
        exp_categories = Index(["a", "b", "c", "d"])
        tm.assert_index_equal(cat.categories, exp_categories)

        # 内部实现细节...
        c = Categorical([1, 2, 3, 4, 1], categories=[1, 2, 3, 4], ordered=True)
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, 2, 3, 0], dtype=np.int8))
        tm.assert_index_equal(c.categories, Index([1, 2, 3, 4]))

        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)

        # 将所有指向 '4' 的指针从 3 改为 0，位置发生变化
        c = c.set_categories([4, 3, 2, 1])

        # 检查位置是否已经改变
        tm.assert_numpy_array_equal(c._codes, np.array([3, 2, 1, 0, 3], dtype=np.int8))

        # 检查类别是否按新顺序排列
        tm.assert_index_equal(c.categories, Index([4, 3, 2, 1]))

        # 输出应该保持不变
        exp = np.array([1, 2, 3, 4, 1], dtype=np.int64)
        tm.assert_numpy_array_equal(np.asarray(c), exp)
        assert c.min() == 4
        assert c.max() == 1

        # 如果指定了排序，set_categories 应该设置排序
        c2 = c.set_categories([4, 3, 2, 1], ordered=False)
        assert not c2.ordered

        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))

        # set_categories 应该传递排序设置
        c2 = c.set_ordered(False).set_categories([4, 3, 2, 1])
        assert not c2.ordered

        tm.assert_numpy_array_equal(np.asarray(c), np.asarray(c2))
    @pytest.mark.parametrize(
        "values, categories, new_categories",
        [
            # No NaNs, same cats, same order
            (["a", "b", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["a", "b", "a"], ["a", "b"], ["b", "a"]),
            # Same, unsorted
            (["b", "a", "a"], ["a", "b"], ["a", "b"]),
            # No NaNs, same cats, different order
            (["b", "a", "a"], ["a", "b"], ["b", "a"]),
            # NaNs
            (["a", "b", "c"], ["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "b"], ["b", "a"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            (["b", "a", "c"], ["a", "b"], ["b", "a"]),
            # Introduce NaNs
            (["a", "b", "c"], ["a", "b"], ["a"]),
            (["a", "b", "c"], ["a", "b"], ["b"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            (["b", "a", "c"], ["a", "b"], ["b"]),
            # No overlap
            (["a", "b", "c"], ["a", "b"], ["d", "e"]),
        ],
    )
    # 参数化测试方法，用于多次运行相同测试方法，每次使用不同的参数组合
    def test_set_categories_many(self, values, categories, new_categories, ordered):
        # 创建分类对象 c，使用给定的 values 和 categories
        c = Categorical(values, categories)
        # 创建预期结果对象 expected，使用相同的 values 和 new_categories
        expected = Categorical(values, new_categories, ordered)
        # 调用被测试方法 set_categories，传入新的 categories 和 ordered 参数
        result = c.set_categories(new_categories, ordered=ordered)
        # 使用测试工具方法 tm.assert_categorical_equal 验证 result 和 expected 是否相等
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_rename_less(self):
        # GH 24675
        # 创建分类对象 cat，使用给定的 ["A", "B"]
        cat = Categorical(["A", "B"])
        # 调用被测试方法 set_categories，使用新的 categories ["A"] 并启用 rename 选项
        result = cat.set_categories(["A"], rename=True)
        # 创建预期结果对象 expected，使用 ["A", np.nan]
        expected = Categorical(["A", np.nan])
        # 使用测试工具方法 tm.assert_categorical_equal 验证 result 和 expected 是否相等
        tm.assert_categorical_equal(result, expected)

    def test_set_categories_private(self):
        # 创建分类对象 cat，使用给定的 ["a", "b", "c"] 和 categories ["a", "b", "c", "d"]
        cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        # 调用私有方法 _set_categories，使用新的 categories ["a", "c", "d", "e"]
        cat._set_categories(["a", "c", "d", "e"])
        # 创建预期结果对象 expected，使用 ["a", "c", "d"] 和 categories ["a", "c", "d", "e"]
        expected = Categorical(["a", "c", "d"], categories=list("acde"))
        # 使用测试工具方法 tm.assert_categorical_equal 验证 cat 和 expected 是否相等
        tm.assert_categorical_equal(cat, expected)

        # fastpath
        # 创建分类对象 cat，使用给定的 ["a", "b", "c"] 和 categories ["a", "b", "c", "d"]
        cat = Categorical(["a", "b", "c"], categories=["a", "b", "c", "d"])
        # 调用私有方法 _set_categories，使用新的 categories ["a", "c", "d", "e"] 并启用 fastpath 选项
        cat._set_categories(["a", "c", "d", "e"], fastpath=True)
        # 创建预期结果对象 expected，使用 ["a", "c", "d"] 和 categories ["a", "c", "d", "e"]
        expected = Categorical(["a", "c", "d"], categories=list("acde"))
        # 使用测试工具方法 tm.assert_categorical_equal 验证 cat 和 expected 是否相等
        tm.assert_categorical_equal(cat, expected)

    def test_remove_categories(self):
        # 创建有序分类对象 cat，使用给定的 ["a", "b", "c", "a"]
        cat = Categorical(["a", "b", "c", "a"], ordered=True)
        # 复制当前分类对象到 old
        old = cat.copy()
        # 创建新分类对象 new，使用 ["a", "b", np.nan, "a"] 和 categories ["a", "b"] 并保持有序
        new = Categorical(["a", "b", np.nan, "a"], categories=["a", "b"], ordered=True)

        # 调用被测试方法 remove_categories，移除 "c"
        res = cat.remove_categories("c")
        # 使用测试工具方法 tm.assert_categorical_equal 验证 cat 和 old 是否相等
        tm.assert_categorical_equal(cat, old)
        # 使用测试工具方法 tm.assert_categorical_equal 验证 res 和 new 是否相等
        tm.assert_categorical_equal(res, new)

        # 调用被测试方法 remove_categories，移除 ["c"]
        res = cat.remove_categories(["c"])
        # 使用测试工具方法 tm.assert_categorical_equal 验证 cat 和 old 是否相等
        tm.assert_categorical_equal(cat, old)
        # 使用测试工具方法 tm.assert_categorical_equal 验证 res 和 new 是否相等
        tm.assert_categorical_equal(res, new)

    @pytest.mark.parametrize("removals", [["c"], ["c", np.nan], "c", ["c", "c"]])
    # 测试函数，用于测试在从分类数据中移除类别时是否会引发异常
    def test_remove_categories_raises(self, removals):
        # 创建一个包含重复类别的分类数据对象
        cat = Categorical(["a", "b", "a"])
        # 构建用于匹配异常消息的正则表达式模式
        message = re.escape("removals must all be in old categories: {'c'}")

        # 使用 pytest 库检测是否会抛出 ValueError 异常，并验证异常消息是否符合预期
        with pytest.raises(ValueError, match=message):
            cat.remove_categories(removals)

    # 测试函数，用于验证移除未使用类别的功能
    def test_remove_unused_categories(self):
        # 创建一个包含指定类别的分类数据对象，并设定预期结果的类别集合
        c = Categorical(["a", "b", "c", "d", "a"], categories=["a", "b", "c", "d", "e"])
        exp_categories_all = Index(["a", "b", "c", "d", "e"])
        exp_categories_dropped = Index(["a", "b", "c", "d"])

        # 断言当前分类数据对象的类别与预期一致
        tm.assert_index_equal(c.categories, exp_categories_all)

        # 执行移除未使用类别的操作，并断言操作后的类别与预期一致，同时原对象的类别保持不变
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, exp_categories_dropped)
        tm.assert_index_equal(c.categories, exp_categories_all)

        # 处理包含 NaN 值的分类数据情况，验证移除未使用类别功能的正确性
        c = Categorical(["a", "b", "c", np.nan], categories=["a", "b", "c", "d", "e"])
        res = c.remove_unused_categories()
        tm.assert_index_equal(res.categories, Index(np.array(["a", "b", "c"])))
        exp_codes = np.array([0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(res.codes, exp_codes)
        tm.assert_index_equal(c.categories, exp_categories_all)

        # 创建包含随机值和 NaN 值的分类数据对象，验证移除未使用类别功能的正确性
        val = ["F", np.nan, "D", "B", "D", "F", np.nan]
        cat = Categorical(values=val, categories=list("ABCDEFG"))
        out = cat.remove_unused_categories()
        tm.assert_index_equal(out.categories, Index(["B", "D", "F"]))
        exp_codes = np.array([2, -1, 1, 0, 1, 2, -1], dtype=np.int8)
        tm.assert_numpy_array_equal(out.codes, exp_codes)
        assert out.tolist() == val

        # 创建包含大量随机数据的分类数据对象，验证移除未使用类别功能的性能和正确性
        alpha = list("abcdefghijklmnopqrstuvwxyz")
        val = np.random.default_rng(2).choice(alpha[::2], 10000).astype("object")
        val[np.random.default_rng(2).choice(len(val), 100)] = np.nan

        cat = Categorical(values=val, categories=alpha)
        out = cat.remove_unused_categories()
        assert out.tolist() == val.tolist()
class TestCategoricalAPIWithFactor:
    def test_describe(self):
        # 创建一个有序的分类变量实例
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"], ordered=True)
        # 调用 describe 方法获取描述信息
        desc = factor.describe()
        # 断言分类变量实例是有序的
        assert factor.ordered
        # 创建预期的分类索引对象
        exp_index = CategoricalIndex(
            ["a", "b", "c"], name="categories", ordered=factor.ordered
        )
        # 创建预期的数据帧对象
        expected = DataFrame(
            {"counts": [3, 2, 3], "freqs": [3 / 8.0, 2 / 8.0, 3 / 8.0]}, index=exp_index
        )
        # 使用测试框架检查描述结果与预期是否相等
        tm.assert_frame_equal(desc, expected)

        # 检查未使用的分类
        cat = factor.copy()
        # 设置新的分类列表，包括未使用的分类 'd'
        cat = cat.set_categories(["a", "b", "c", "d"])
        # 再次调用 describe 方法获取描述信息
        desc = cat.describe()

        # 创建预期的分类索引对象，包括新增的分类 'd'
        exp_index = CategoricalIndex(
            list("abcd"), ordered=factor.ordered, name="categories"
        )
        # 创建预期的数据帧对象，包括未使用分类的计数和频率
        expected = DataFrame(
            {"counts": [3, 2, 3, 0], "freqs": [3 / 8.0, 2 / 8.0, 3 / 8.0, 0]},
            index=exp_index,
        )
        # 使用测试框架检查描述结果与预期是否相等
        tm.assert_frame_equal(desc, expected)

        # 检查整数类型的分类变量
        cat = Categorical([1, 2, 3, 1, 2, 3, 3, 2, 1, 1, 1])
        # 再次调用 describe 方法获取描述信息
        desc = cat.describe()
        # 创建预期的分类索引对象
        exp_index = CategoricalIndex([1, 2, 3], ordered=cat.ordered, name="categories")
        # 创建预期的数据帧对象
        expected = DataFrame(
            {"counts": [5, 3, 3], "freqs": [5 / 11.0, 3 / 11.0, 3 / 11.0]},
            index=exp_index,
        )
        # 使用测试框架检查描述结果与预期是否相等
        tm.assert_frame_equal(desc, expected)

        # 测试包含 NaN 的情况
        cat = Categorical([np.nan, 1, 2, 2])
        # 再次调用 describe 方法获取描述信息
        desc = cat.describe()
        # 创建预期的数据帧对象，包括 NaN 值的计数和频率
        expected = DataFrame(
            {"counts": [1, 2, 1], "freqs": [1 / 4.0, 2 / 4.0, 1 / 4.0]},
            index=CategoricalIndex(
                [1, 2, np.nan], categories=[1, 2], name="categories"
            ),
        )
        # 使用测试框架检查描述结果与预期是否相等
        tm.assert_frame_equal(desc, expected)


class TestPrivateCategoricalAPI:
    def test_codes_immutable(self):
        # 测试代码属性为只读
        c = Categorical(["a", "b", "c", "a", np.nan])
        # 创建预期的代码数组
        exp = np.array([0, 1, 2, 0, -1], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)

        # 尝试对代码属性进行赋值应该引发异常
        msg = (
            "property 'codes' of 'Categorical' object has no setter"
            if PY311
            else "can't set attribute"
        )
        with pytest.raises(AttributeError, match=msg):
            c.codes = np.array([0, 1, 2, 0, 1], dtype="int8")

        # 尝试修改代码数组中的元素应该引发异常
        codes = c.codes
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            codes[4] = 1

        # 即使获取了代码数组，原始数组仍然应该是可写的！
        # 修改原始分类变量中的值后，检查预期的代码数组
        c[4] = "a"
        exp = np.array([0, 1, 2, 0, 0], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)
        c._codes[4] = 2
        exp = np.array([0, 1, 2, 0, 2], dtype="int8")
        tm.assert_numpy_array_equal(c.codes, exp)
    # 使用 pytest 的参数化装饰器，定义多组参数进行测试
    @pytest.mark.parametrize(
        "codes, old, new, expected",
        [
            # 测试用例1: codes=[0, 1], old=["a", "b"], new=["a", "b"], expected=[0, 1]
            ([0, 1], ["a", "b"], ["a", "b"], [0, 1]),
            # 测试用例2: codes=[0, 1], old=["b", "a"], new=["b", "a"], expected=[0, 1]
            ([0, 1], ["b", "a"], ["b", "a"], [0, 1]),
            # 测试用例3: codes=[0, 1], old=["a", "b"], new=["b", "a"], expected=[1, 0]
            ([0, 1], ["a", "b"], ["b", "a"], [1, 0]),
            # 测试用例4: codes=[0, 1], old=["b", "a"], new=["a", "b"], expected=[1, 0]
            ([0, 1], ["b", "a"], ["a", "b"], [1, 0]),
            # 测试用例5: codes=[0, 1, 0, 1], old=["a", "b"], new=["a", "b", "c"], expected=[0, 1, 0, 1]
            ([0, 1, 0, 1], ["a", "b"], ["a", "b", "c"], [0, 1, 0, 1]),
            # 测试用例6: codes=[0, 1, 2, 2], old=["a", "b", "c"], new=["a", "b"], expected=[0, 1, -1, -1]
            ([0, 1, 2, 2], ["a", "b", "c"], ["a", "b"], [0, 1, -1, -1]),
            # 测试用例7: codes=[0, 1, -1], old=["a", "b", "c"], new=["a", "b", "c"], expected=[0, 1, -1]
            ([0, 1, -1], ["a", "b", "c"], ["a", "b", "c"], [0, 1, -1]),
            # 测试用例8: codes=[0, 1, -1], old=["a", "b", "c"], new=["b"], expected=[-1, 0, -1]
            ([0, 1, -1], ["a", "b", "c"], ["b"], [-1, 0, -1]),
            # 测试用例9: codes=[0, 1, -1], old=["a", "b", "c"], new=["d"], expected=[-1, -1, -1]
            ([0, 1, -1], ["a", "b", "c"], ["d"], [-1, -1, -1]),
            # 测试用例10: codes=[0, 1, -1], old=["a", "b", "c"], new=[], expected=[-1, -1, -1]
            ([0, 1, -1], ["a", "b", "c"], [], [-1, -1, -1]),
            # 测试用例11: codes=[-1, -1], old=[], new=["a", "b"], expected=[-1, -1]
            ([-1, -1], [], ["a", "b"], [-1, -1]),
            # 测试用例12: codes=[1, 0], old=["b", "a"], new=["a", "b"], expected=[0, 1]
            ([1, 0], ["b", "a"], ["a", "b"], [0, 1]),
        ],
    )
    # 定义测试函数，用于测试 recode_for_categories 函数
    def test_recode_to_categories(self, codes, old, new, expected):
        # 将 codes 转换为 numpy 数组，dtype 为 np.int8
        codes = np.asanyarray(codes, dtype=np.int8)
        # 将 expected 转换为 numpy 数组，dtype 为 np.int8
        expected = np.asanyarray(expected, dtype=np.int8)
        # 创建 Index 对象 old，用于处理旧类别
        old = Index(old)
        # 创建 Index 对象 new，用于处理新类别
        new = Index(new)
        # 调用 recode_for_categories 函数进行类别重编码
        result = recode_for_categories(codes, old, new)
        # 断言 numpy 数组 result 与 expected 相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试大数据集情况下的函数
    def test_recode_to_categories_large(self):
        N = 1000
        # 生成从 0 到 N-1 的 numpy 数组 codes
        codes = np.arange(N)
        # 创建 Index 对象 old，用于处理旧类别
        old = Index(codes)
        # 生成从 N-1 到 0 的 numpy 数组 expected，dtype 为 np.int16
        expected = np.arange(N - 1, -1, -1, dtype=np.int16)
        # 创建 Index 对象 new，用于处理新类别
        new = Index(expected)
        # 调用 recode_for_categories 函数进行类别重编码
        result = recode_for_categories(codes, old, new)
        # 断言 numpy 数组 result 与 expected 相等
        tm.assert_numpy_array_equal(result, expected)
```