# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_indexing.py`

```
import numpy as np  # 导入NumPy库，用于处理数组和数值计算

import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas.errors import InvalidIndexError  # 从pandas.errors模块导入InvalidIndexError异常类

import pandas as pd  # 导入pandas库，用于数据分析和处理

from pandas import (  # 从pandas库中导入多个类和函数
    CategoricalIndex,   # 类别索引
    Index,              # 普通索引
    IntervalIndex,      # 区间索引
    Timestamp,          # 时间戳
)

import pandas._testing as tm  # 导入pandas._testing模块，用于测试辅助函数和类


class TestTake:
    def test_take_fill_value(self):
        # GH 12631
        # GH 12631问题的测试用例

        # numeric category
        idx = CategoricalIndex([1, 2, 3], name="xxx")  # 创建一个数值类别索引对象
        result = idx.take(np.array([1, 0, -1]))  # 使用索引数组获取索引对象的子集
        expected = CategoricalIndex([2, 1, 3], name="xxx")  # 预期的结果索引对象
        tm.assert_index_equal(result, expected)  # 断言结果索引和预期索引相等
        tm.assert_categorical_equal(result.values, expected.values)  # 断言结果索引的值和预期索引的值相等

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)  # 使用填充值获取索引对象的子集
        expected = CategoricalIndex([2, 1, np.nan], categories=[1, 2, 3], name="xxx")  # 预期的结果索引对象，包含填充值
        tm.assert_index_equal(result, expected)  # 断言结果索引和预期索引相等
        tm.assert_categorical_equal(result.values, expected.values)  # 断言结果索引的值和预期索引的值相等

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)  # 不允许填充值获取索引对象的子集
        expected = CategoricalIndex([2, 1, 3], name="xxx")  # 预期的结果索引对象
        tm.assert_index_equal(result, expected)  # 断言结果索引和预期索引相等
        tm.assert_categorical_equal(result.values, expected.values)  # 断言结果索引的值和预期索引的值相等

        # object category
        idx = CategoricalIndex(
            list("CBA"), categories=list("ABC"), ordered=True, name="xxx"
        )  # 创建一个对象类别索引对象
        result = idx.take(np.array([1, 0, -1]))  # 使用索引数组获取索引对象的子集
        expected = CategoricalIndex(
            list("BCA"), categories=list("ABC"), ordered=True, name="xxx"
        )  # 预期的结果索引对象
        tm.assert_index_equal(result, expected)  # 断言结果索引和预期索引相等
        tm.assert_categorical_equal(result.values, expected.values)  # 断言结果索引的值和预期索引的值相等

        # fill_value
        result = idx.take(np.array([1, 0, -1]), fill_value=True)  # 使用填充值获取索引对象的子集
        expected = CategoricalIndex(
            ["B", "C", np.nan], categories=list("ABC"), ordered=True, name="xxx"
        )  # 预期的结果索引对象，包含填充值
        tm.assert_index_equal(result, expected)  # 断言结果索引和预期索引相等
        tm.assert_categorical_equal(result.values, expected.values)  # 断言结果索引的值和预期索引的值相等

        # allow_fill=False
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)  # 不允许填充值获取索引对象的子集
        expected = CategoricalIndex(
            list("BCA"), categories=list("ABC"), ordered=True, name="xxx"
        )  # 预期的结果索引对象
        tm.assert_index_equal(result, expected)  # 断言结果索引和预期索引相等
        tm.assert_categorical_equal(result.values, expected.values)  # 断言结果索引的值和预期索引的值相等

        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )  # 错误消息字符串
        with pytest.raises(ValueError, match=msg):  # 使用pytest断言抛出指定错误消息的值错误
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):  # 使用pytest断言抛出指定错误消息的值错误
            idx.take(np.array([1, 0, -5]), fill_value=True)

        msg = "index -5 is out of bounds for (axis 0 with )?size 3"  # 错误消息字符串
        with pytest.raises(IndexError, match=msg):  # 使用pytest断言抛出指定错误消息的索引错误
            idx.take(np.array([1, -5]))
    # 测试在填充值为日期时间类型时的行为
    def test_take_fill_value_datetime(self):
        # 创建一个日期时间索引对象
        idx = pd.DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"], name="xxx")
        # 将日期时间索引转换为分类索引对象
        idx = CategoricalIndex(idx)
        # 使用给定的索引数组获取索引值，返回结果与期望的日期时间索引对象相等
        result = idx.take(np.array([1, 0, -1]))
        expected = pd.DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx"
        )
        expected = CategoricalIndex(expected)
        tm.assert_index_equal(result, expected)

        # 使用填充值选项获取索引值，期望结果为填充了缺失值的日期时间索引对象
        result = idx.take(np.array([1, 0, -1]), fill_value=True)
        expected = pd.DatetimeIndex(["2011-02-01", "2011-01-01", "NaT"], name="xxx")
        exp_cats = pd.DatetimeIndex(["2011-01-01", "2011-02-01", "2011-03-01"])
        expected = CategoricalIndex(expected, categories=exp_cats)
        tm.assert_index_equal(result, expected)

        # 当 allow_fill=False 时，获取索引值，期望结果与未填充值时相同
        result = idx.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = pd.DatetimeIndex(
            ["2011-02-01", "2011-01-01", "2011-03-01"], name="xxx"
        )
        expected = CategoricalIndex(expected)
        tm.assert_index_equal(result, expected)

        # 使用不合法的索引值时，当 allow_fill=True 且填充值不为 None 时，抛出 ValueError 异常
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -2]), fill_value=True)
        with pytest.raises(ValueError, match=msg):
            idx.take(np.array([1, 0, -5]), fill_value=True)

        # 使用超出索引范围的索引值时，抛出 IndexError 异常
        msg = "index -5 is out of bounds for (axis 0 with )?size 3"
        with pytest.raises(IndexError, match=msg):
            idx.take(np.array([1, -5]))

    # 测试不合法的关键字参数
    def test_take_invalid_kwargs(self):
        # 创建一个分类索引对象
        idx = CategoricalIndex([1, 2, 3], name="foo")
        indices = [1, 0, -1]

        # 当传递未预期的关键字参数 'foo' 时，抛出 TypeError 异常
        msg = r"take\(\) got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=msg):
            idx.take(indices, foo=2)

        # 当传递不支持的 'out' 参数时，抛出 ValueError 异常
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, out=indices)

        # 当传递不支持的 'mode' 参数时，抛出 ValueError 异常
        msg = "the 'mode' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            idx.take(indices, mode="clip")
class TestGetLoc:
    def test_get_loc(self):
        # 创建一个具有自定义类别的分类索引对象 cidx1，从列表 ["abcde"] 中创建，类别为 ["edabc"]
        cidx1 = CategoricalIndex(list("abcde"), categories=list("edabc"))
        # 创建一个标准索引对象 idx1，从列表 ["abcde"] 中创建
        idx1 = Index(list("abcde"))
        # 断言 cidx1 中字符 "a" 的位置与 idx1 中字符 "a" 的位置相同
        assert cidx1.get_loc("a") == idx1.get_loc("a")
        # 断言 cidx1 中字符 "e" 的位置与 idx1 中字符 "e" 的位置相同
        assert cidx1.get_loc("e") == idx1.get_loc("e")

        # 对于 cidx1 和 idx1，检查当查找不存在的键 'NOT-EXIST' 时是否引发 KeyError 异常
        for i in [cidx1, idx1]:
            with pytest.raises(KeyError, match="'NOT-EXIST'"):
                i.get_loc("NOT-EXIST")

        # 创建一个具有非唯一值的分类索引对象 cidx2，从列表 ["aacded"] 中创建，类别为 ["edabc"]
        cidx2 = CategoricalIndex(list("aacded"), categories=list("edabc"))
        # 创建一个标准索引对象 idx2，从列表 ["aacded"] 中创建
        idx2 = Index(list("aacded"))

        # 获取字符 "d" 在 cidx2 中的位置，并将结果与 idx2 中字符 "d" 的位置进行比较
        # 结果以布尔数组形式返回
        res = cidx2.get_loc("d")
        tm.assert_numpy_array_equal(res, idx2.get_loc("d"))
        tm.assert_numpy_array_equal(
            res, np.array([False, False, False, True, False, True])
        )
        # 获取字符 "e" 在 cidx2 中的位置，并将结果与 idx2 中字符 "e" 的位置进行比较
        # 结果以标量形式返回
        res = cidx2.get_loc("e")
        assert res == idx2.get_loc("e")
        assert res == 4

        # 对于 cidx2 和 idx2，检查当查找不存在的键 'NOT-EXIST' 时是否引发 KeyError 异常
        for i in [cidx2, idx2]:
            with pytest.raises(KeyError, match="'NOT-EXIST'"):
                i.get_loc("NOT-EXIST")

        # 创建一个具有非唯一值的分类索引对象 cidx3，从列表 ["aabbb"] 中创建，类别为 ["abc"]
        cidx3 = CategoricalIndex(list("aabbb"), categories=list("abc"))
        # 创建一个标准索引对象 idx3，从列表 ["aabbb"] 中创建
        idx3 = Index(list("aabbb"))

        # 获取字符 "a" 在 cidx3 中的位置，并将结果与 idx3 中字符 "a" 的位置进行比较
        # 结果以切片形式返回
        res = cidx3.get_loc("a")
        assert res == idx3.get_loc("a")
        assert res == slice(0, 2, None)

        # 获取字符 "b" 在 cidx3 中的位置，并将结果与 idx3 中字符 "b" 的位置进行比较
        # 结果以切片形式返回
        res = cidx3.get_loc("b")
        assert res == idx3.get_loc("b")
        assert res == slice(2, 5, None)

        # 对于 cidx3 和 idx3，检查当查找不存在的键 'c' 时是否引发 KeyError 异常
        for i in [cidx3, idx3]:
            with pytest.raises(KeyError, match="'c'"):
                i.get_loc("c")

    def test_get_loc_unique(self):
        # 创建一个具有唯一值的分类索引对象 cidx，从列表 ["abc"] 中创建
        cidx = CategoricalIndex(list("abc"))
        # 获取字符 "b" 在 cidx 中的位置，并断言其为 1
        result = cidx.get_loc("b")
        assert result == 1

    def test_get_loc_monotonic_nonunique(self):
        # 创建一个具有非单调且非唯一值的分类索引对象 cidx，从列表 ["abbc"] 中创建
        cidx = CategoricalIndex(list("abbc"))
        # 获取字符 "b" 在 cidx 中的位置，并与预期的切片对象进行比较
        expected = slice(1, 3, None)
        result = cidx.get_loc("b")
        assert result == expected

    def test_get_loc_nonmonotonic_nonunique(self):
        # 创建一个具有非单调且非唯一值的分类索引对象 cidx，从列表 ["abcb"] 中创建
        cidx = CategoricalIndex(list("abcb"))
        # 获取字符 "b" 在 cidx 中的位置，并将结果与预期的布尔数组进行比较
        expected = np.array([False, True, False, True], dtype=bool)
        result = cidx.get_loc("b")
        tm.assert_numpy_array_equal(result, expected)

    def test_get_loc_nan(self):
        # 创建一个具有 NaN 值的分类索引对象 ci，从列表 ["A", "B", np.nan] 中创建
        ci = CategoricalIndex(["A", "B", np.nan])
        # 获取 NaN 在 ci 中的位置，并断言其为 2
        res = ci.get_loc(np.nan)
        assert res == 2


class TestGetIndexer:
    def test_get_indexer_base(self):
        # 创建一个具有指定类别顺序的分类索引对象 idx，从列表 ["cab"] 中创建，类别为 ["cab"]
        idx = CategoricalIndex(list("cab"), categories=list("cab"))
        # 创建一个预期的整数数组，用于与 idx.get_indexer 方法返回的实际结果进行比较
        expected = np.arange(len(idx), dtype=np.intp)

        # 调用 idx.get_indexer 方法，传入 idx 作为参数，获取实际结果并与预期结果进行比较
        actual = idx.get_indexer(idx)
        tm.assert_numpy_array_equal(expected, actual)

        # 使用 pytest 框架断言调用 idx.get_indexer 方法时使用无效的填充方法会引发 ValueError 异常
        with pytest.raises(ValueError, match="Invalid fill method"):
            idx.get_indexer(idx, method="invalid")
    # 测试函数：test_get_indexer_requires_unique
    def test_get_indexer_requires_unique(self):
        # 创建一个非有序的分类索引对象 ci，包含字符"aabbca"，使用的分类列表为"cab"
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        # 根据 ci 创建一个索引对象 oidx
        oidx = Index(np.array(ci))

        # 错误消息字符串，用于异常断言
        msg = "Reindexing only valid with uniquely valued Index objects"

        # 对于列表中的每个 n 值，进行测试
        for n in [1, 2, 5, len(ci)]:
            # 从 oidx 中随机选择 n 个元素作为 finder
            finder = oidx[np.random.default_rng(2).integers(0, len(ci), size=n)]

            # 使用 pytest 断言应该抛出 InvalidIndexError 异常，并且异常消息应该匹配 msg
            with pytest.raises(InvalidIndexError, match=msg):
                ci.get_indexer(finder)

        # 注释：关于 GitHub 问题 gh-17323
        #
        # 即使索引器 finder 等于索引中的成员，我们也应该
        # 尊重重复项，而不是采取快速路径。
        for finder in [list("aabbca"), list("aababca")]:
            # 使用 pytest 断言应该抛出 InvalidIndexError 异常，并且异常消息应该匹配 msg
            with pytest.raises(InvalidIndexError, match=msg):
                ci.get_indexer(finder)

    # 测试函数：test_get_indexer_non_unique
    def test_get_indexer_non_unique(self):
        # 创建两个分类索引对象 idx1 和 idx2
        idx1 = CategoricalIndex(list("aabcde"), categories=list("edabc"))
        idx2 = CategoricalIndex(list("abf"))

        # 对于每个 indexer 进行测试
        for indexer in [idx2, list("abf"), Index(list("abf"))]:
            # 错误消息字符串，用于异常断言
            msg = "Reindexing only valid with uniquely valued Index objects"

            # 使用 pytest 断言应该抛出 InvalidIndexError 异常，并且异常消息应该匹配 msg
            with pytest.raises(InvalidIndexError, match=msg):
                idx1.get_indexer(indexer)

            # 获取非唯一值的索引器结果
            r1, _ = idx1.get_indexer_non_unique(indexer)
            expected = np.array([0, 1, 2, -1], dtype=np.intp)
            # 使用测试工具方法 tm.assert_almost_equal 检查结果与预期是否接近
            tm.assert_almost_equal(r1, expected)

    # 测试函数：test_get_indexer_method
    def test_get_indexer_method(self):
        # 创建两个分类索引对象 idx1 和 idx2
        idx1 = CategoricalIndex(list("aabcde"), categories=list("edabc"))
        idx2 = CategoricalIndex(list("abf"))

        # 错误消息字符串，用于异常断言
        msg = "method pad not yet implemented for CategoricalIndex"
        # 使用 pytest 断言应该抛出 NotImplementedError 异常，并且异常消息应该匹配 msg
        with pytest.raises(NotImplementedError, match=msg):
            idx2.get_indexer(idx1, method="pad")

        msg = "method backfill not yet implemented for CategoricalIndex"
        # 使用 pytest 断言应该抛出 NotImplementedError 异常，并且异常消息应该匹配 msg
        with pytest.raises(NotImplementedError, match=msg):
            idx2.get_indexer(idx1, method="backfill")

        msg = "method nearest not yet implemented for CategoricalIndex"
        # 使用 pytest 断言应该抛出 NotImplementedError 异常，并且异常消息应该匹配 msg
        with pytest.raises(NotImplementedError, match=msg):
            idx2.get_indexer(idx1, method="nearest")

    # 测试函数：test_get_indexer_array
    def test_get_indexer_array(self):
        # 创建一个包含 Timestamp 对象的 numpy 数组 arr 和对应的分类列表 cats
        arr = np.array(
            [Timestamp("1999-12-31 00:00:00"), Timestamp("2000-12-31 00:00:00")],
            dtype=object,
        )
        cats = [Timestamp("1999-12-31 00:00:00"), Timestamp("2000-12-31 00:00:00")]
        # 创建一个非有序的分类索引对象 ci
        ci = CategoricalIndex(cats, categories=cats, ordered=False, dtype="category")
        # 获取 arr 在 ci 中的索引器结果
        result = ci.get_indexer(arr)
        expected = np.array([0, 1], dtype="intp")
        # 使用测试工具方法 tm.assert_numpy_array_equal 检查结果与预期是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：test_get_indexer_same_categories_same_order
    def test_get_indexer_same_categories_same_order(self):
        # 创建一个包含两个元素的分类索引对象 ci，使用相同的分类列表
        ci = CategoricalIndex(["a", "b"], categories=["a", "b"])

        # 获取一个另外的分类索引对象的索引器结果
        result = ci.get_indexer(CategoricalIndex(["b", "b"], categories=["a", "b"]))
        expected = np.array([1, 1], dtype="intp")
        # 使用测试工具方法 tm.assert_numpy_array_equal 检查结果与预期是否相等
        tm.assert_numpy_array_equal(result, expected)
    # 定义测试方法：验证当类别索引中的类别顺序不同但内容相同时的行为
    def test_get_indexer_same_categories_different_order(self):
        # GitHub 上的问题链接，描述了此测试方法的背景
        # https://github.com/pandas-dev/pandas/issues/19551
        # 创建一个分类索引对象 ci，包含 ["a", "b"] 两个类别，类别顺序为 ["a", "b"]
        ci = CategoricalIndex(["a", "b"], categories=["a", "b"])

        # 调用 ci 对象的 get_indexer 方法，使用另一个分类索引对象作为参数
        result = ci.get_indexer(CategoricalIndex(["b", "b"], categories=["b", "a"]))
        
        # 预期的结果是一个 numpy 数组，内容为 [1, 1]，数据类型为 intp
        expected = np.array([1, 1], dtype="intp")
        
        # 使用测试框架中的方法验证 result 是否与 expected 相等
        tm.assert_numpy_array_equal(result, expected)

    # 定义测试方法：验证当类别索引中包含 NaN 值时的行为
    def test_get_indexer_nans_in_index_and_target(self):
        # GitHub 上的问题编号，描述了此测试方法的背景
        # GH 45361
        # 创建一个分类索引对象 ci，包含 [1, 2, NaN, 3] 四个值
        ci = CategoricalIndex([1, 2, np.nan, 3])
        
        # 创建另一个列表 other1，包含 [2, 3, 4, NaN]，用于作为参数调用 get_indexer 方法
        other1 = [2, 3, 4, np.nan]
        
        # 调用 ci 对象的 get_indexer 方法，使用 other1 作为参数
        res1 = ci.get_indexer(other1)
        
        # 预期的结果是一个 numpy 数组，内容为 [1, 3, -1, 2]，数据类型为 intp
        expected1 = np.array([1, 3, -1, 2], dtype=np.intp)
        
        # 使用测试框架中的方法验证 res1 是否与 expected1 相等
        tm.assert_numpy_array_equal(res1, expected1)
        
        # 创建另一个列表 other2，包含 [1, 4, 2, 3]，用于作为参数调用 get_indexer 方法
        other2 = [1, 4, 2, 3]
        
        # 调用 ci 对象的 get_indexer 方法，使用 other2 作为参数
        res2 = ci.get_indexer(other2)
        
        # 预期的结果是一个 numpy 数组，内容为 [0, -1, 1, 3]，数据类型为 intp
        expected2 = np.array([0, -1, 1, 3], dtype=np.intp)
        
        # 使用测试框架中的方法验证 res2 是否与 expected2 相等
        tm.assert_numpy_array_equal(res2, expected2)
class TestWhere:
    # 测试 where 方法
    def test_where(self, listlike_box):
        # 将 listlike_box 赋给 klass
        klass = listlike_box

        # 创建一个非有序的分类索引对象 i，内容为 "aabbca"，类别为 "cab"
        i = CategoricalIndex(list("aabbca"), categories=list("cab"), ordered=False)
        # 创建一个长度与 i 相同的条件列表 cond，均为 True
        cond = [True] * len(i)
        # 预期结果为 i 本身
        expected = i
        # 使用 where 方法进行条件筛选，将结果赋给 result
        result = i.where(klass(cond))
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

        # 将第一个元素设置为 False，其余为 True，形成条件列表 cond
        cond = [False] + [True] * (len(i) - 1)
        # 预期结果为将第一个元素替换为 NaN，其余保持不变
        expected = CategoricalIndex([np.nan] + i[1:].tolist(), categories=i.categories)
        # 使用 where 方法进行条件筛选，将结果赋给 result
        result = i.where(klass(cond))
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

    # 测试 where 方法对非分类索引对象的影响
    def test_where_non_categories(self):
        # 创建一个非分类索引对象 ci，内容为 ["a", "b", "c", "d"]
        ci = CategoricalIndex(["a", "b", "c", "d"])
        # 创建一个布尔掩码 mask，用于条件筛选
        mask = np.array([True, False, True, False])

        # 使用 where 方法，将 mask 中为 True 的位置替换为 2
        result = ci.where(mask, 2)
        # 预期结果为 ["a", 2, "c", 2]，类型为 object
        expected = Index(["a", 2, "c", 2], dtype=object)
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

        # 定义错误消息
        msg = "Cannot setitem on a Categorical with a new category"
        # 使用 pytest 的断言抛出 TypeError 异常，消息匹配 msg
        with pytest.raises(TypeError, match=msg):
            # 测试 CategoricalIndex 的 _where 方法，直接调用方法
            ci._data._where(mask, 2)


class TestContains:
    # 测试 contains 方法
    def test_contains(self):
        # 创建一个非有序的分类索引对象 ci，内容为 "aabbca"，类别为 "cabdef"
        ci = CategoricalIndex(list("aabbca"), categories=list("cabdef"), ordered=False)

        # 断言 "a" 在 ci 中
        assert "a" in ci
        # 断言 "z" 不在 ci 中
        assert "z" not in ci
        # 断言 "e" 不在 ci 中
        assert "e" not in ci
        # 断言 np.nan 不在 ci 中
        assert np.nan not in ci

        # 断言代码 0 不在索引中
        assert 0 not in ci
        # 断言代码 1 不在索引中
        assert 1 not in ci

    # 测试 contains 方法对含有 NaN 的情况
    def test_contains_nan(self):
        # 创建一个含有 NaN 的分类索引对象 ci，内容为 "aabbca" + [NaN]，类别为 "cabdef"
        ci = CategoricalIndex(list("aabbca") + [np.nan], categories=list("cabdef"))
        # 断言 NaN 在 ci 中
        assert np.nan in ci

    # 使用 unwrap 参数化测试 contains 方法
    @pytest.mark.parametrize("unwrap", [True, False])
    def test_contains_na_dtype(self, unwrap):
        # 创建一个日期范围 dti，并插入 NaT 值
        dti = pd.date_range("2016-01-01", periods=100).insert(0, pd.NaT)
        # 将 dti 转换为周期类型 pi
        pi = dti.to_period("D")
        # 创建一个时间差 tdi
        tdi = dti - dti[-1]
        # 创建一个分类索引对象 ci，内容为 dti
        ci = CategoricalIndex(dti)

        # 根据 unwrap 参数选择 obj
        obj = ci
        if unwrap:
            obj = ci._data

        # 断言 NaN 在 obj 中
        assert np.nan in obj
        # 断言 None 在 obj 中
        assert None in obj
        # 断言 pd.NaT 在 obj 中
        assert pd.NaT in obj
        # 断言 np.datetime64("NaT") 在 obj 中
        assert np.datetime64("NaT") in obj
        # 断言 np.timedelta64("NaT") 不在 obj 中
        assert np.timedelta64("NaT") not in obj

        # 创建另一个分类索引对象 obj2，内容为 tdi
        obj2 = CategoricalIndex(tdi)
        if unwrap:
            obj2 = obj2._data

        # 断言 NaN 在 obj2 中
        assert np.nan in obj2
        # 断言 None 在 obj2 中
        assert None in obj2
        # 断言 pd.NaT 在 obj2 中
        assert pd.NaT in obj2
        # 断言 np.datetime64("NaT") 不在 obj2 中
        assert np.datetime64("NaT") not in obj2
        # 断言 np.timedelta64("NaT") 在 obj2 中
        assert np.timedelta64("NaT") in obj2

        # 创建另一个分类索引对象 obj3，内容为 pi
        obj3 = CategoricalIndex(pi)
        if unwrap:
            obj3 = obj3._data

        # 断言 NaN 在 obj3 中
        assert np.nan in obj3
        # 断言 None 在 obj3 中
        assert None in obj3
        # 断言 pd.NaT 在 obj3 中
        assert pd.NaT in obj3
        # 断言 np.datetime64("NaT") 不在 obj3 中
        assert np.datetime64("NaT") not in obj3
        # 断言 np.timedelta64("NaT") 不在 obj3 中
        assert np.timedelta64("NaT") not in obj3

    # 使用参数化测试 item 和 expected
    @pytest.mark.parametrize(
        "item, expected",
        [
            (pd.Interval(0, 1), True),
            (1.5, True),
            (pd.Interval(0.5, 1.5), False),
            ("a", False),
            (Timestamp(1), False),
            (pd.Timedelta(1), False),
        ],
        ids=str,
    )
    # 测试方法：检查元素是否存在于分类索引中，并与期望结果比较
    def test_contains_interval(self, item, expected):
        # GH 23705
        # 创建一个包含间隔索引的分类索引对象
        ci = CategoricalIndex(IntervalIndex.from_breaks(range(3)))
        # 检查 item 是否在分类索引 ci 中
        result = item in ci
        # 断言检查结果是否符合期望
        assert result is expected

    # 测试方法：检查列表是否存在于分类索引中
    def test_contains_list(self):
        # GH#21729
        # 创建一个包含整数列表的分类索引对象
        idx = CategoricalIndex([1, 2, 3])

        # 断言字符串 "a" 不在分类索引 idx 中
        assert "a" not in idx

        # 使用 pytest 来检测以下情况是否会引发 TypeError，匹配错误消息为 "unhashable type"
        with pytest.raises(TypeError, match="unhashable type"):
            # 断言列表 ["a"] 不在分类索引 idx 中
            ["a"] in idx

        with pytest.raises(TypeError, match="unhashable type"):
            # 断言列表 ["a", "b"] 不在分类索引 idx 中
            ["a", "b"] in idx
```