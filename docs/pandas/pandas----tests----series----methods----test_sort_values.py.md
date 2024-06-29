# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_sort_values.py`

```
import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm

class TestSeriesSortValues:
    def test_sort_values_categorical(self):
        # 创建一个包含非有序分类数据的 Series 对象
        cat = Series(Categorical(["a", "b", "b", "a"], ordered=False))

        # 按照分类的顺序排序
        expected = Series(
            Categorical(["a", "a", "b", "b"], ordered=False), index=[0, 3, 1, 2]
        )
        result = cat.sort_values()
        tm.assert_series_equal(result, expected)

        # 创建一个包含有序分类数据的 Series 对象
        cat = Series(Categorical(["a", "c", "b", "d"], ordered=True))
        res = cat.sort_values()

        # 期望结果是按字母顺序排列的 numpy 数组
        exp = np.array(["a", "b", "c", "d"], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)

        # 创建一个有序分类数据，并显式指定其分类顺序的 Series 对象
        cat = Series(
            Categorical(
                ["a", "c", "b", "d"], categories=["a", "b", "c", "d"], ordered=True
            )
        )
        res = cat.sort_values()

        # 验证结果为按字母顺序排列的 numpy 数组
        exp = np.array(["a", "b", "c", "d"], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)

        # 按降序排列分类数据
        res = cat.sort_values(ascending=False)
        exp = np.array(["d", "c", "b", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(res.__array__(), exp)

        # 创建未排序的分类数据对象
        raw_cat1 = Categorical(
            ["a", "b", "c", "d"], categories=["a", "b", "c", "d"], ordered=False
        )
        raw_cat2 = Categorical(
            ["a", "b", "c", "d"], categories=["d", "c", "b", "a"], ordered=True
        )
        s = ["a", "b", "c", "d"]

        # 创建一个包含多列数据的 DataFrame 对象
        df = DataFrame(
            {"unsort": raw_cat1, "sort": raw_cat2, "string": s, "values": [1, 2, 3, 4]}
        )

        # 按 'string' 列的值降序排序 DataFrame
        res = df.sort_values(by=["string"], ascending=False)
        exp = np.array(["d", "c", "b", "a"], dtype=np.object_)
        tm.assert_numpy_array_equal(res["sort"].values.__array__(), exp)
        assert res["sort"].dtype == "category"

        # 按 'sort' 列的值降序排序 DataFrame，并验证其它列的顺序
        res = df.sort_values(by=["sort"], ascending=False)
        exp = df.sort_values(by=["string"], ascending=True)
        tm.assert_series_equal(res["values"], exp["values"])
        assert res["sort"].dtype == "category"
        assert res["unsort"].dtype == "category"

        # 对未排序的分类数据进行排序（这种操作是被允许的）
        df.sort_values(by=["unsort"], ascending=False)

        # 多列排序示例
        # GH#7848
        df = DataFrame(
            {"id": [6, 5, 4, 3, 2, 1], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
        )
        df["grade"] = Categorical(df["raw_grade"], ordered=True)
        df["grade"] = df["grade"].cat.set_categories(["b", "e", "a"])

        # 按 'grade' 列的值排序，并验证结果
        result = df.sort_values(by=["grade"])
        expected = df.iloc[[1, 2, 5, 0, 3, 4]]
        tm.assert_frame_equal(result, expected)

        # 多列排序示例
        result = df.sort_values(by=["grade", "id"])
        expected = df.iloc[[2, 1, 5, 4, 3, 0]]
        tm.assert_frame_equal(result, expected)
    # 使用 pytest 的参数化装饰器，测试 inplace 参数为 True 和 False 时的情况
    @pytest.mark.parametrize("inplace", [True, False])
    # 参数化测试用例，验证排序前后的列表、是否忽略索引以及预期输出的索引
    @pytest.mark.parametrize(
        "original_list, sorted_list, ignore_index, output_index",
        [
            ([2, 3, 6, 1], [6, 3, 2, 1], True, [0, 1, 2, 3]),
            ([2, 3, 6, 1], [6, 3, 2, 1], False, [2, 1, 0, 3]),
        ],
    )
    # 测试排序函数的 ignore_index 参数功能
    def test_sort_values_ignore_index(
        self, inplace, original_list, sorted_list, ignore_index, output_index
    ):
        # GH 30114
        # 创建 Series 对象
        ser = Series(original_list)
        # 创建预期的 Series 对象，指定索引
        expected = Series(sorted_list, index=output_index)
        # 准备排序参数
        kwargs = {"ignore_index": ignore_index, "inplace": inplace}

        if inplace:
            # 如果 inplace 为 True，则对副本进行排序操作
            result_ser = ser.copy()
            result_ser.sort_values(ascending=False, **kwargs)
        else:
            # 否则对原始 Series 进行排序
            result_ser = ser.sort_values(ascending=False, **kwargs)

        # 验证排序后的结果与预期是否一致
        tm.assert_series_equal(result_ser, expected)
        # 验证原始 Series 是否保持不变
        tm.assert_series_equal(ser, Series(original_list))

    # 测试稳定排序算法 mergesort 在降序排序中的稳定性
    def test_mergesort_descending_stability(self):
        # GH 28697
        # 创建 Series 对象
        s = Series([1, 2, 1, 3], ["first", "b", "second", "c"])
        # 使用 mergesort 算法进行降序排序
        result = s.sort_values(ascending=False, kind="mergesort")
        # 创建预期的 Series 对象，指定索引
        expected = Series([3, 2, 1, 1], ["c", "b", "first", "second"])
        # 验证排序后的结果与预期是否一致
        tm.assert_series_equal(result, expected)

    # 测试 sort_values 方法在传递错误的 ascending 参数时是否抛出 ValueError
    def test_sort_values_validate_ascending_for_value_error(self):
        # GH41634
        # 创建 Series 对象
        ser = Series([23, 7, 21])

        # 准备错误信息字符串
        msg = 'For argument "ascending" expected type bool, received type str.'
        # 验证传递错误的 ascending 参数时是否抛出 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            ser.sort_values(ascending="False")

    # 测试 sort_values 方法在传递正确的 ascending 参数时的功能性验证
    def test_sort_values_validate_ascending_functional(self, ascending):
        # GH41634
        # 创建 Series 对象
        ser = Series([23, 7, 21])
        # 创建预期的排序后的 numpy 数组
        expected = np.sort(ser.values)

        # 根据传入的 ascending 参数进行排序
        sorted_ser = ser.sort_values(ascending=ascending)
        if not ascending:
            # 如果 ascending 为 False，则反转预期的数组
            expected = expected[::-1]

        # 获取排序后的结果数组
        result = sorted_ser.values
        # 验证排序后的结果数组与预期是否一致
        tm.assert_numpy_array_equal(result, expected)
class TestSeriesSortingKey:
    # 测试类 TestSeriesSortingKey，用于测试 Series 排序关键字功能

    def test_sort_values_key(self):
        # 定义测试方法 test_sort_values_key，测试排序方法 sort_values 的默认行为
        series = Series(np.array(["Hello", "goodbye"]))

        # 调用 sort_values 方法，默认按值排序
        result = series.sort_values(axis=0)
        expected = series
        # 断言排序结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 调用 sort_values 方法，使用自定义关键字 key=lambda x: x.str.lower() 进行排序
        result = series.sort_values(axis=0, key=lambda x: x.str.lower())
        expected = series[::-1]
        # 断言排序结果与预期结果相等
        tm.assert_series_equal(result, expected)

    def test_sort_values_key_nan(self):
        # 定义测试方法 test_sort_values_key_nan，测试处理 NaN 值的排序方法
        series = Series(np.array([0, 5, np.nan, 3, 2, np.nan]))

        # 调用 sort_values 方法，默认按值排序
        result = series.sort_values(axis=0)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        # 断言排序结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 调用 sort_values 方法，使用 key=lambda x: x + 5 进行排序
        result = series.sort_values(axis=0, key=lambda x: x + 5)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        # 断言排序结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 调用 sort_values 方法，使用 key=lambda x: -x 进行排序，同时指定降序排序
        result = series.sort_values(axis=0, key=lambda x: -x, ascending=False)
        expected = series.iloc[[0, 4, 3, 1, 2, 5]]
        # 断言排序结果与预期结果相等
        tm.assert_series_equal(result, expected)
```