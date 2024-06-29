# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_append.py`

```
import pytest  # 导入 pytest 库

from pandas import (  # 从 pandas 库中导入以下模块
    CategoricalIndex,  # 分类索引模块
    Index,  # 索引模块
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestAppend:  # 定义测试类 TestAppend

    @pytest.fixture  # 使用 pytest 的装饰器定义测试的 fixture
    def ci(self):  # 定义 fixture ci，返回一个 CategoricalIndex 对象
        categories = list("cab")  # 定义分类的列表
        return CategoricalIndex(list("aabbca"), categories=categories, ordered=False)  # 返回一个 CategoricalIndex 对象

    def test_append(self, ci):  # 定义测试方法 test_append，接受参数 ci
        # append cats with the same categories
        result = ci[:3].append(ci[3:])  # 将前三个元素和后三个元素拼接起来
        tm.assert_index_equal(result, ci, exact=True)  # 断言结果和原始 ci 相等

        foos = [ci[:1], ci[1:3], ci[3:]]  # 将 ci 分成三部分
        result = foos[0].append(foos[1:])  # 将 foos 列表的内容拼接起来
        tm.assert_index_equal(result, ci, exact=True)  # 断言结果和原始 ci 相等

    def test_append_empty(self, ci):  # 定义测试方法 test_append_empty，接受参数 ci
        # empty
        result = ci.append([])  # 将空列表追加到 ci 中
        tm.assert_index_equal(result, ci, exact=True)  # 断言结果和原始 ci 相等

    def test_append_mismatched_categories(self, ci):  # 定义测试方法 test_append_mismatched_categories，接受参数 ci
        # appending with different categories or reordered is not ok
        msg = "all inputs must be Index"  # 错误消息
        with pytest.raises(TypeError, match=msg):  # 断言引发 TypeError 异常，并检查错误消息是否匹配
            ci.append(ci.values.set_categories(list("abcd")))  # 尝试追加不同类别或重新排序的内容
        with pytest.raises(TypeError, match=msg):  # 再次断言引发 TypeError 异常，并检查错误消息是否匹配
            ci.append(ci.values.reorder_categories(list("abc")))  # 尝试追加重新排序的内容

    def test_append_category_objects(self, ci):  # 定义测试方法 test_append_category_objects，接受参数 ci
        # with objects
        result = ci.append(Index(["c", "a"]))  # 将另一个 Index 对象追加到 ci 中
        expected = CategoricalIndex(list("aabbcaca"), categories=ci.categories)  # 预期的结果
        tm.assert_index_equal(result, expected, exact=True)  # 断言结果和预期结果相等

    def test_append_non_categories(self, ci):  # 定义测试方法 test_append_non_categories，接受参数 ci
        # invalid objects -> cast to object via concat_compat
        result = ci.append(Index(["a", "d"]))  # 将另一个 Index 对象追加到 ci 中
        expected = Index(["a", "a", "b", "b", "c", "a", "a", "d"])  # 预期的结果
        tm.assert_index_equal(result, expected, exact=True)  # 断言结果和预期结果相等

    def test_append_object(self, ci):  # 定义测试方法 test_append_object，接受参数 ci
        # GH#14298 - if base object is not categorical -> coerce to object
        result = Index(["c", "a"]).append(ci)  # 将另一个 Index 对象追加到 ci 中
        expected = Index(list("caaabbca"))  # 预期的结果
        tm.assert_index_equal(result, expected, exact=True)  # 断言结果和预期结果相等

    def test_append_to_another(self):  # 定义测试方法 test_append_to_another
        # hits Index._concat
        fst = Index(["a", "b"])  # 创建第一个 Index 对象
        snd = CategoricalIndex(["d", "e"])  # 创建第二个 CategoricalIndex 对象
        result = fst.append(snd)  # 将第二个对象追加到第一个对象中
        expected = Index(["a", "b", "d", "e"])  # 预期的结果
        tm.assert_index_equal(result, expected)  # 断言结果和预期结果相等
```