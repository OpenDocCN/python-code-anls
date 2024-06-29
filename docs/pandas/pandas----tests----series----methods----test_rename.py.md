# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_rename.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库，并用 np 别名表示
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下对象：
    Index,  # 索引对象
    MultiIndex,  # 多重索引对象
    Series,  # 系列对象
    array,  # 数组对象
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestRename:  # 定义测试类 TestRename

    def test_rename(self, datetime_series):  # 定义测试方法 test_rename，接受一个 datetime_series 参数
        ts = datetime_series  # 将参数 datetime_series 赋值给变量 ts
        renamer = lambda x: x.strftime("%Y%m%d")  # 定义一个 lambda 函数 renamer，将日期格式化为年月日字符串
        renamed = ts.rename(renamer)  # 对 ts 应用 renamer 函数进行重命名操作
        assert renamed.index[0] == renamer(ts.index[0])  # 断言操作，验证重命名后的第一个索引是否符合预期

        # dict
        rename_dict = dict(zip(ts.index, renamed.index))  # 创建一个字典，将原索引和重命名后的索引进行配对
        renamed2 = ts.rename(rename_dict)  # 使用字典进行重命名操作
        tm.assert_series_equal(renamed, renamed2)  # 使用测试模块 tm，断言两个 Series 对象是否相等

    def test_rename_partial_dict(self):  # 定义测试方法 test_rename_partial_dict
        # partial dict
        ser = Series(np.arange(4), index=["a", "b", "c", "d"], dtype="int64")  # 创建一个 Series 对象
        renamed = ser.rename({"b": "foo", "d": "bar"})  # 使用部分字典进行重命名
        tm.assert_index_equal(renamed.index, Index(["a", "foo", "c", "bar"]))  # 断言操作，验证重命名后的索引是否符合预期

    def test_rename_retain_index_name(self):  # 定义测试方法 test_rename_retain_index_name
        # index with name
        renamer = Series(  # 创建一个带有名称的 Series 对象
            np.arange(4),  # 数组数据为 0 到 3
            index=Index(["a", "b", "c", "d"], name="name"),  # 设置索引为 a, b, c, d，索引名称为 name
            dtype="int64"  # 数据类型为 int64
        )
        renamed = renamer.rename({})  # 使用空字典进行重命名操作，保持索引名称不变
        assert renamed.index.name == renamer.index.name  # 断言操作，验证重命名后的索引名称是否与原始索引名称相同

    def test_rename_by_series(self):  # 定义测试方法 test_rename_by_series
        ser = Series(range(5), name="foo")  # 创建一个 Series 对象，数据为 0 到 4，名称为 foo
        renamer = Series({1: 10, 2: 20})  # 创建一个 Series 对象作为重命名映射
        result = ser.rename(renamer)  # 使用 Series 对象进行重命名操作
        expected = Series(range(5), index=[0, 10, 20, 3, 4], name="foo")  # 创建预期的 Series 对象
        tm.assert_series_equal(result, expected)  # 断言操作，验证重命名后的结果是否与预期相同

    def test_rename_set_name(self, using_infer_string):  # 定义测试方法 test_rename_set_name，接受 using_infer_string 参数
        ser = Series(range(4), index=list("abcd"))  # 创建一个 Series 对象，数据为 0 到 3，索引为 a, b, c, d
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:  # 迭代不同类型的名称
            result = ser.rename(name)  # 使用不同类型的名称进行重命名操作
            assert result.name == name  # 断言操作，验证重命名后的名称是否符合预期
            if using_infer_string:  # 根据 using_infer_string 参数选择不同的断言函数
                tm.assert_extension_array_equal(result.index.values, ser.index.values)
            else:
                tm.assert_numpy_array_equal(result.index.values, ser.index.values)
            assert ser.name is None  # 断言操作，验证原始 Series 对象的名称为 None

    def test_rename_set_name_inplace(self, using_infer_string):  # 定义测试方法 test_rename_set_name_inplace，接受 using_infer_string 参数
        ser = Series(range(3), index=list("abc"))  # 创建一个 Series 对象，数据为 0 到 2，索引为 a, b, c
        for name in ["foo", 123, 123.0, datetime(2001, 11, 11), ("foo",)]:  # 迭代不同类型的名称
            ser.rename(name, inplace=True)  # 使用不同类型的名称进行原地重命名操作
            assert ser.name == name  # 断言操作，验证重命名后的名称是否符合预期
            exp = np.array(["a", "b", "c"], dtype=np.object_)  # 创建预期的索引数组
            if using_infer_string:  # 根据 using_infer_string 参数选择不同的断言函数
                exp = array(exp, dtype="string[pyarrow_numpy]")
                tm.assert_extension_array_equal(ser.index.values, exp)
            else:
                tm.assert_numpy_array_equal(ser.index.values, exp)

    def test_rename_axis_supported(self):  # 定义测试方法 test_rename_axis_supported
        # Supporting axis for compatibility, detailed in GH-18589
        ser = Series(range(5))  # 创建一个包含 0 到 4 的 Series 对象
        ser.rename({}, axis=0)  # 对索引轴为 0 进行重命名操作
        ser.rename({}, axis="index")  # 对索引轴为 "index" 进行重命名操作

        with pytest.raises(ValueError, match="No axis named 5"):  # 使用 pytest 断言抛出 ValueError 异常
            ser.rename({}, axis=5)  # 对不存在的索引轴进行重命名操作
    def test_rename_inplace(self, datetime_series):
        # 定义一个 lambda 函数 renamer，用于将时间序列的索引转换为指定格式的字符串
        renamer = lambda x: x.strftime("%Y%m%d")
        # 计算期望的重命名结果，即将时间序列的第一个索引项按 renamer 转换的结果
        expected = renamer(datetime_series.index[0])

        # 在原地修改时间序列的索引，使用 renamer 函数进行重命名
        datetime_series.rename(renamer, inplace=True)
        # 断言修改后的时间序列的第一个索引是否等于预期的值
        assert datetime_series.index[0] == expected

    def test_rename_with_custom_indexer(self):
        # GH 27814
        # 定义一个自定义的索引器类 MyIndexer
        class MyIndexer:
            pass

        # 创建一个 MyIndexer 实例 ix
        ix = MyIndexer()
        # 使用 MyIndexer 实例 ix 对 Series 进行重命名
        ser = Series([1, 2, 3]).rename(ix)
        # 断言重命名后的 Series 的名称是否为 ix
        assert ser.name is ix

    def test_rename_with_custom_indexer_inplace(self):
        # GH 27814
        # 定义一个自定义的索引器类 MyIndexer
        class MyIndexer:
            pass

        # 创建一个 MyIndexer 实例 ix
        ix = MyIndexer()
        # 对 Series 进行原地重命名，使用 MyIndexer 实例 ix
        ser = Series([1, 2, 3])
        ser.rename(ix, inplace=True)
        # 断言原地重命名后的 Series 的名称是否为 ix
        assert ser.name is ix

    def test_rename_callable(self):
        # GH 17407
        # 创建一个 Series，指定索引和名称
        ser = Series(range(1, 6), index=Index(range(2, 7), name="IntIndex"))
        # 使用 str 函数对 Series 进行重命名
        result = ser.rename(str)
        # 使用 lambda 函数对 Series 进行重命名，期望结果与 result 相同
        expected = ser.rename(lambda i: str(i))
        # 断言两次重命名结果是否相等
        tm.assert_series_equal(result, expected)
        # 断言重命名后的 Series 的名称是否相同
        assert result.name == expected.name

    def test_rename_none(self):
        # GH 40977
        # 创建一个 Series，指定数据和名称
        ser = Series([1, 2], name="foo")
        # 使用 None 对 Series 进行重命名
        result = ser.rename(None)
        # 创建一个预期的 Series，数据与原 Series 相同
        expected = Series([1, 2])
        # 断言重命名后的 Series 是否与预期的 Series 相等
        tm.assert_series_equal(result, expected)

    def test_rename_series_with_multiindex(self):
        # issue #43659
        # 创建一个多级索引的数组
        arrays = [
            ["bar", "baz", "baz", "foo", "qux"],
            ["one", "one", "two", "two", "one"],
        ]

        # 根据数组创建一个多级索引对象 index
        index = MultiIndex.from_arrays(arrays, names=["first", "second"])
        # 创建一个 Series，指定数据和索引
        ser = Series(np.ones(5), index=index)
        # 使用指定的参数对 Series 的索引进行重命名
        result = ser.rename(index={"one": "yes"}, level="second", errors="raise")

        # 创建一个预期的多级索引的数组
        arrays_expected = [
            ["bar", "baz", "baz", "foo", "qux"],
            ["yes", "yes", "two", "two", "yes"],
        ]

        # 根据预期的数组创建预期的多级索引对象 index_expected
        index_expected = MultiIndex.from_arrays(
            arrays_expected, names=["first", "second"]
        )
        # 创建一个预期的 Series，数据与原 Series 相同，但索引已经重命名
        series_expected = Series(np.ones(5), index=index_expected)

        # 断言重命名后的 Series 是否与预期的 Series 相等
        tm.assert_series_equal(result, series_expected)

    def test_rename_series_with_multiindex_keeps_ea_dtypes(self):
        # GH21055
        # 创建一个包含分类数据类型的索引数组
        arrays = [
            Index([1, 2, 3], dtype="Int64").astype("category"),
            Index([1, 2, 3], dtype="Int64"),
        ]
        # 根据数组创建一个多级索引对象 mi
        mi = MultiIndex.from_arrays(arrays, names=["A", "B"])
        # 创建一个 Series，指定数据和索引
        ser = Series(1, index=mi)
        # 使用指定的参数对 Series 的索引进行重命名
        result = ser.rename({1: 4}, level=1)

        # 创建一个预期的包含分类数据类型的索引数组
        arrays_expected = [
            Index([1, 2, 3], dtype="Int64").astype("category"),
            Index([4, 2, 3], dtype="Int64"),
        ]
        # 根据预期的数组创建预期的多级索引对象 mi_expected
        mi_expected = MultiIndex.from_arrays(arrays_expected, names=["A", "B"])
        # 创建一个预期的 Series，数据与原 Series 相同，但索引已经重命名
        expected = Series(1, index=mi_expected)

        # 断言重命名后的 Series 是否与预期的 Series 相等
        tm.assert_series_equal(result, expected)

    def test_rename_error_arg(self):
        # GH 46889
        # 创建一个包含字符串的 Series
        ser = Series(["foo", "bar"])
        # 设置一个正则表达式字符串，用于匹配预期的异常信息
        match = re.escape("[2] not found in axis")
        # 使用 pytest 框架断言在重命名时是否抛出预期的 KeyError 异常
        with pytest.raises(KeyError, match=match):
            ser.rename({2: 9}, errors="raise")
    def test_rename_copy_false(self):
        # 定义一个测试函数，用于测试重命名且不复制的行为
        # GH 46889 是 GitHub 上的 issue 编号
        # 创建一个 Series 对象，包含字符串 "foo" 和 "bar"
        ser = Series(["foo", "bar"])
        # 复制原始 Series 对象，以备后续比较使用
        ser_orig = ser.copy()
        # 使用 rename 方法对 Series 进行浅复制，将索引 1 重命名为 9
        shallow_copy = ser.rename({1: 9})
        # 修改原始 Series 的第一个元素为 "foobar"
        ser[0] = "foobar"
        # 断言：原始 Series 和浅复制后的 Series 的第一个元素应该相等
        assert ser_orig[0] == shallow_copy[0]
        # 断言：原始 Series 的第二个元素和浅复制后的 Series 的第九个元素应该相等
        assert ser_orig[1] == shallow_copy[9]
```