# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_missing.py`

```
import collections  # 导入 collections 模块

import numpy as np  # 导入 numpy 库，使用 np 别名
import pytest  # 导入 pytest 库

from pandas.core.dtypes.dtypes import CategoricalDtype  # 从 pandas 库中导入 CategoricalDtype 类型

import pandas as pd  # 导入 pandas 库，使用 pd 别名
from pandas import (  # 从 pandas 库中导入多个模块
    Categorical,
    Index,
    Series,
    isna,
)
import pandas._testing as tm  # 导入 pandas._testing 模块，使用 tm 别名


class TestCategoricalMissing:
    def test_isna(self):
        exp = np.array([False, False, True])  # 创建一个预期的 numpy 数组
        cat = Categorical(["a", "b", np.nan])  # 创建一个 Categorical 类型的对象
        res = cat.isna()  # 调用 Categorical 对象的 isna() 方法，返回结果

        tm.assert_numpy_array_equal(res, exp)  # 使用 pandas._testing 模块中的 assert_numpy_array_equal 方法进行结果比较

    def test_na_flags_int_categories(self):
        # #1457

        categories = list(range(10))  # 创建一个包含整数 0 到 9 的列表
        labels = np.random.default_rng(2).integers(0, 10, 20)  # 生成一个包含 20 个随机整数的 numpy 数组，取值范围为 0 到 9
        labels[::5] = -1  # 将 labels 数组中每隔五个元素的位置设置为 -1

        cat = Categorical(labels, categories)  # 创建一个基于 labels 和 categories 的 Categorical 对象
        repr(cat)  # 打印该 Categorical 对象的字符串表示形式

        tm.assert_numpy_array_equal(isna(cat), labels == -1)  # 使用 pandas._testing 模块中的 assert_numpy_array_equal 方法进行结果比较

    def test_nan_handling(self):
        # Nans are represented as -1 in codes
        c = Categorical(["a", "b", np.nan, "a"])  # 创建一个包含 NaN 值的 Categorical 对象
        tm.assert_index_equal(c.categories, Index(["a", "b"]))  # 使用 pandas._testing 模块中的 assert_index_equal 方法进行结果比较
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0], dtype=np.int8))  # 使用 pandas._testing 模块中的 assert_numpy_array_equal 方法进行结果比较
        c[1] = np.nan  # 将索引为 1 的元素设置为 NaN
        tm.assert_index_equal(c.categories, Index(["a", "b"]))  # 使用 pandas._testing 模块中的 assert_index_equal 方法进行结果比较
        tm.assert_numpy_array_equal(c._codes, np.array([0, -1, -1, 0], dtype=np.int8))  # 使用 pandas._testing 模块中的 assert_numpy_array_equal 方法进行结果比较

        # Adding nan to categories should make assigned nan point to the
        # category!
        c = Categorical(["a", "b", np.nan, "a"])  # 创建一个包含 NaN 值的 Categorical 对象
        tm.assert_index_equal(c.categories, Index(["a", "b"]))  # 使用 pandas._testing 模块中的 assert_index_equal 方法进行结果比较
        tm.assert_numpy_array_equal(c._codes, np.array([0, 1, -1, 0], dtype=np.int8))  # 使用 pandas._testing 模块中的 assert_numpy_array_equal 方法进行结果比较

    def test_set_dtype_nans(self):
        c = Categorical(["a", "b", np.nan])  # 创建一个包含 NaN 值的 Categorical 对象
        result = c._set_dtype(CategoricalDtype(["a", "c"]))  # 调用 Categorical 对象的 _set_dtype 方法，返回结果
        tm.assert_numpy_array_equal(result.codes, np.array([0, -1, -1], dtype="int8"))  # 使用 pandas._testing 模块中的 assert_numpy_array_equal 方法进行结果比较

    def test_set_item_nan(self):
        cat = Categorical([1, 2, 3])  # 创建一个包含整数的 Categorical 对象
        cat[1] = np.nan  # 将索引为 1 的元素设置为 NaN

        exp = Categorical([1, np.nan, 3], categories=[1, 2, 3])  # 创建一个预期的 Categorical 对象
        tm.assert_categorical_equal(cat, exp)  # 使用 pandas._testing 模块中的 assert_categorical_equal 方法进行结果比较

    @pytest.mark.parametrize("named", [True, False])
    def test_fillna_iterable_category(self, named):
        # https://github.com/pandas-dev/pandas/issues/21097
        if named:
            Point = collections.namedtuple("Point", "x y")  # 如果 named 为 True，则创建一个命名元组 Point
        else:
            Point = lambda *args: args  # 如果 named 为 False，则创建一个元组 Point

        cat = Categorical(np.array([Point(0, 0), Point(0, 1), None], dtype=object))  # 创建一个包含对象的 Categorical 对象
        result = cat.fillna(Point(0, 0))  # 调用 Categorical 对象的 fillna 方法，返回结果
        expected = Categorical([Point(0, 0), Point(0, 1), Point(0, 0)])  # 创建一个预期的 Categorical 对象

        tm.assert_categorical_equal(result, expected)  # 使用 pandas._testing 模块中的 assert_categorical_equal 方法进行结果比较

        # Case where the Point is not among our categories; we want ValueError,
        #  not NotImplementedError GH#41914
        cat = Categorical(np.array([Point(1, 0), Point(0, 1), None], dtype=object))  # 创建一个包含对象的 Categorical 对象
        msg = "Cannot setitem on a Categorical with a new category"  # 定义一个错误信息字符串
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 的 raises 方法捕获 TypeError 异常，验证错误信息
            cat.fillna(Point(0, 0))  # 调用 Categorical 对象的 fillna 方法
    def test_fillna_array(self):
        # 测试填充缺失值的功能，接受包含合适数值的分类或 ndarray 值
        cat = Categorical(["A", "B", "C", None, None])

        # 使用 "C" 填充缺失值
        other = cat.fillna("C")
        # 使用填充后的结果再次填充
        result = cat.fillna(other)
        # 断言填充后的结果与填充值相同
        tm.assert_categorical_equal(result, other)
        # 断言原始分类对象的最后一个元素仍然是缺失值
        assert isna(cat[-1])  # 没有在原地修改原始数据

        # 使用 ndarray ["A", "B", "C", "B", "A"] 填充缺失值
        other = np.array(["A", "B", "C", "B", "A"])
        # 期望的填充后的分类对象
        expected = Categorical(["A", "B", "C", "B", "A"], dtype=cat.dtype)
        # 断言填充后的结果与期望值相同
        tm.assert_categorical_equal(result, expected)
        # 断言原始分类对象的最后一个元素仍然是缺失值
        assert isna(cat[-1])  # 没有在原地修改原始数据

    @pytest.mark.parametrize(
        "a1, a2, categories",
        [
            (["a", "b", "c"], [np.nan, "a", "b"], ["a", "b", "c"]),
            ([1, 2, 3], [np.nan, 1, 2], [1, 2, 3]),
        ],
    )
    def test_compare_categorical_with_missing(self, a1, a2, categories):
        # GH 28384：针对缺失值进行分类比较的测试

        # 使用指定的分类类型创建 Series，并比较不相等情况
        cat_type = CategoricalDtype(categories)
        result = Series(a1, dtype=cat_type) != Series(a2, dtype=cat_type)
        expected = Series(a1) != Series(a2)
        tm.assert_series_equal(result, expected)

        # 使用指定的分类类型创建 Series，并比较相等情况
        result = Series(a1, dtype=cat_type) == Series(a2, dtype=cat_type)
        expected = Series(a1) == Series(a2)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "na_value, dtype",
        [
            (pd.NaT, "datetime64[s]"),
            (None, "float64"),
            (np.nan, "float64"),
            (pd.NA, "float64"),
        ],
    )
    def test_categorical_only_missing_values_no_cast(self, na_value, dtype):
        # GH#44900：仅处理缺失值，不进行类型转换的分类测试

        # 创建包含缺失值的 Categorical 对象
        result = Categorical([na_value, na_value])
        # 断言结果的 categories 为空 Index，且其数据类型为指定的 dtype
        tm.assert_index_equal(result.categories, Index([], dtype=dtype))
```