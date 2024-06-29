# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_dtypes.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from pandas.core.dtypes.dtypes import CategoricalDtype  # 导入CategoricalDtype类，处理分类数据类型

from pandas import (  # 导入多个Pandas模块和类
    Categorical,
    CategoricalIndex,
    Index,
    IntervalIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入Pandas测试模块


class TestCategoricalDtypes:
    def test_categories_match_up_to_permutation(self):
        # 测试分类数据类型之间的比较

        c1 = Categorical(list("aabca"), categories=list("abc"), ordered=False)
        c2 = Categorical(list("aabca"), categories=list("cab"), ordered=False)
        c3 = Categorical(list("aabca"), categories=list("cab"), ordered=True)
        assert c1._categories_match_up_to_permutation(c1)  # 断言分类c1与自身匹配
        assert c2._categories_match_up_to_permutation(c2)  # 断言分类c2与自身匹配
        assert c3._categories_match_up_to_permutation(c3)  # 断言分类c3与自身匹配
        assert c1._categories_match_up_to_permutation(c2)  # 断言分类c1与c2匹配
        assert not c1._categories_match_up_to_permutation(c3)  # 断言分类c1与c3不匹配
        assert not c1._categories_match_up_to_permutation(Index(list("aabca")))  # 断言分类c1与Index对象不匹配
        assert not c1._categories_match_up_to_permutation(c1.astype(object))  # 断言分类c1与转换为object类型后的c1不匹配
        assert c1._categories_match_up_to_permutation(CategoricalIndex(c1))  # 断言分类c1与其CategoricalIndex匹配
        assert c1._categories_match_up_to_permutation(
            CategoricalIndex(c1, categories=list("cab"))
        )  # 断言分类c1与指定categories的CategoricalIndex匹配
        assert not c1._categories_match_up_to_permutation(
            CategoricalIndex(c1, ordered=True)
        )  # 断言分类c1与有序的CategoricalIndex不匹配

        # GH 16659
        s1 = Series(c1)
        s2 = Series(c2)
        s3 = Series(c3)
        assert c1._categories_match_up_to_permutation(s1)  # 断言分类c1与其Series匹配
        assert c2._categories_match_up_to_permutation(s2)  # 断言分类c2与其Series匹配
        assert c3._categories_match_up_to_permutation(s3)  # 断言分类c3与其Series匹配
        assert c1._categories_match_up_to_permutation(s2)  # 断言分类c1与c2的Series匹配
        assert not c1._categories_match_up_to_permutation(s3)  # 断言分类c1与c3的Series不匹配
        assert not c1._categories_match_up_to_permutation(s1.astype(object))  # 断言分类c1与转换为object类型后的c1的Series不匹配

    def test_set_dtype_same(self):
        c = Categorical(["a", "b", "c"])
        result = c._set_dtype(CategoricalDtype(["a", "b", "c"]))
        tm.assert_categorical_equal(result, c)  # 使用tm.assert_categorical_equal断言result与c相等

    def test_set_dtype_new_categories(self):
        c = Categorical(["a", "b", "c"])
        result = c._set_dtype(CategoricalDtype(list("abcd")))
        tm.assert_numpy_array_equal(result.codes, c.codes)  # 使用tm.assert_numpy_array_equal断言result的codes与c的codes相等
        tm.assert_index_equal(result.dtype.categories, Index(list("abcd")))  # 使用tm.assert_index_equal断言result的dtype.categories与指定的Index相等
    @pytest.mark.parametrize(
        "values, categories, new_categories",
        [
            # 测试用例参数化：无 NaN，相同分类，顺序相同
            (["a", "b", "a"], ["a", "b"], ["a", "b"]),
            # 测试用例参数化：无 NaN，相同分类，顺序不同
            (["a", "b", "a"], ["a", "b"], ["b", "a"]),
            # 测试用例参数化：相同值，未排序
            (["b", "a", "a"], ["a", "b"], ["a", "b"]),
            # 测试用例参数化：无 NaN，相同分类，顺序不同
            (["b", "a", "a"], ["a", "b"], ["b", "a"]),
            # 测试用例参数化：含有 NaN
            (["a", "b", "c"], ["a", "b"], ["a", "b"]),
            (["a", "b", "c"], ["a", "b"], ["b", "a"]),
            (["b", "a", "c"], ["a", "b"], ["a", "b"]),
            (["b", "a", "c"], ["a", "b"], ["b", "a"]),
            # 测试用例参数化：引入 NaN
            (["a", "b", "c"], ["a", "b"], ["a"]),
            (["a", "b", "c"], ["a", "b"], ["b"]),
            (["b", "a", "c"], ["a", "b"], ["a"]),
            (["b", "a", "c"], ["a", "b"], ["b"]),
            # 测试用例参数化：无重叠
            (["a", "b", "c"], ["a", "b"], ["d", "e"]),
        ],
    )
    def test_set_dtype_many(self, values, categories, new_categories, ordered):
        # 创建分类数据对象 c
        c = Categorical(values, categories)
        # 创建预期结果的分类数据对象 expected
        expected = Categorical(values, new_categories, ordered)
        # 调用被测试方法 _set_dtype
        result = c._set_dtype(expected.dtype)
        # 使用测试框架检查 result 是否与 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_set_dtype_no_overlap(self):
        # 创建分类数据对象 c，包含不在原始分类中的值
        c = Categorical(["a", "b", "c"], ["d", "e"])
        # 调用被测试方法 _set_dtype，指定新的分类数据类型
        result = c._set_dtype(CategoricalDtype(["a", "b"]))
        # 创建期望的分类数据对象 expected
        expected = Categorical([None, None, None], categories=["a", "b"])
        # 使用测试框架检查 result 是否与 expected 相等
        tm.assert_categorical_equal(result, expected)

    def test_codes_dtypes(self):
        # GH 8453
        # 创建分类数据对象 result，包含字符串数据
        result = Categorical(["foo", "bar", "baz"])
        # 使用断言检查 result 的 codes 属性的数据类型是否为 int8
        assert result.codes.dtype == "int8"

        # 创建分类数据对象 result，包含大量字符串数据
        result = Categorical([f"foo{i:05d}" for i in range(400)])
        # 使用断言检查 result 的 codes 属性的数据类型是否为 int16
        assert result.codes.dtype == "int16"

        # 创建分类数据对象 result，包含非常大量字符串数据
        result = Categorical([f"foo{i:05d}" for i in range(40000)])
        # 使用断言检查 result 的 codes 属性的数据类型是否为 int32
        assert result.codes.dtype == "int32"

        # 在原分类数据对象 result 上添加新的分类
        result = Categorical(["foo", "bar", "baz"])
        assert result.codes.dtype == "int8"
        result = result.add_categories([f"foo{i:05d}" for i in range(400)])
        # 使用断言检查 result 的 codes 属性的数据类型是否为 int16
        assert result.codes.dtype == "int16"

        # 在原分类数据对象 result 上移除部分分类
        result = result.remove_categories([f"foo{i:05d}" for i in range(300)])
        # 使用断言检查 result 的 codes 属性的数据类型是否为 int8
        assert result.codes.dtype == "int8"

    def test_iter_python_types(self):
        # GH-19909
        # 创建包含整数的分类数据对象 cat
        cat = Categorical([1, 2])
        # 使用断言检查分类数据对象的第一个元素是否为整数类型
        assert isinstance(next(iter(cat)), int)
        # 使用断言检查通过 tolist 方法转换后的第一个元素是否为整数类型
        assert isinstance(cat.tolist()[0], int)

    def test_iter_python_types_datetime(self):
        # 创建包含时间戳的分类数据对象 cat
        cat = Categorical([Timestamp("2017-01-01"), Timestamp("2017-01-02")])
        # 使用断言检查分类数据对象的第一个元素是否为 Timestamp 类型
        assert isinstance(next(iter(cat)), Timestamp)
        # 使用断言检查通过 tolist 方法转换后的第一个元素是否为 Timestamp 类型
        assert isinstance(cat.tolist()[0], Timestamp)
    def test_interval_index_category(self):
        # 定义一个测试方法，用于测试区间索引的类别
        # GH 38316 是指 GitHub 上的 issue 编号，可能是与此测试相关的问题编号

        # 创建一个 IntervalIndex 对象，根据给定的断点数组 np.arange(3, dtype="uint64")
        index = IntervalIndex.from_breaks(np.arange(3, dtype="uint64"))

        # 使用 IntervalIndex 创建一个 CategoricalIndex 对象，并获取其 dtype 的类别
        result = CategoricalIndex(index).dtype.categories

        # 创建一个期望的 IntervalIndex 对象，根据给定的边界数组 [0, 1] 和 [1, 2]，类型为 "interval[uint64, right]"
        expected = IntervalIndex.from_arrays(
            [0, 1], [1, 2], dtype="interval[uint64, right]"
        )

        # 使用断言方法验证 result 和 expected 是否相等
        tm.assert_index_equal(result, expected)
```