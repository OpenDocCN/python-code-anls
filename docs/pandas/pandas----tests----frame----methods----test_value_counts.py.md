# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_value_counts.py`

```
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

# 定义一个测试函数，用于测试未排序的数据框值计数
def test_data_frame_value_counts_unsorted():
    # 创建一个包含动物和它们的腿数和翅膀数的数据框
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    # 对数据框进行值计数，不排序
    result = df.value_counts(sort=False)
    # 预期结果是一个包含计数的序列，使用多级索引表示腿数和翅膀数
    expected = pd.Series(
        data=[1, 2, 1],
        index=pd.MultiIndex.from_arrays(
            [(2, 4, 6), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试升序排列的数据框值计数
def test_data_frame_value_counts_ascending():
    # 创建一个包含动物和它们的腿数和翅膀数的数据框
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    # 对数据框进行值计数，升序排列
    result = df.value_counts(ascending=True)
    # 预期结果是一个包含计数的序列，使用多级索引表示腿数和翅膀数
    expected = pd.Series(
        data=[1, 1, 2],
        index=pd.MultiIndex.from_arrays(
            [(2, 6, 4), (2, 0, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试默认排序的数据框值计数
def test_data_frame_value_counts_default():
    # 创建一个包含动物和它们的腿数和翅膀数的数据框
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    # 对数据框进行默认排序的值计数
    result = df.value_counts()
    # 预期结果是一个包含计数的序列，使用多级索引表示腿数和翅膀数
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="count",
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试归一化的数据框值计数
def test_data_frame_value_counts_normalize():
    # 创建一个包含动物和它们的腿数和翅膀数的数据框
    df = pd.DataFrame(
        {"num_legs": [2, 4, 4, 6], "num_wings": [2, 0, 0, 0]},
        index=["falcon", "dog", "cat", "ant"],
    )

    # 对数据框进行归一化的值计数
    result = df.value_counts(normalize=True)
    # 预期结果是一个包含比例的序列，使用多级索引表示腿数和翅膀数
    expected = pd.Series(
        data=[0.5, 0.25, 0.25],
        index=pd.MultiIndex.from_arrays(
            [(4, 2, 6), (0, 2, 0)], names=["num_legs", "num_wings"]
        ),
        name="proportion",
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试只包含单列的数据框默认排序的值计数
def test_data_frame_value_counts_single_col_default():
    # 创建一个包含只有腿数的数据框
    df = pd.DataFrame({"num_legs": [2, 4, 4, 6]})

    # 对数据框进行默认排序的值计数
    result = df.value_counts()
    # 预期结果是一个包含计数的序列，使用单级索引表示腿数
    expected = pd.Series(
        data=[2, 1, 1],
        index=pd.MultiIndex.from_arrays([[4, 2, 6]], names=["num_legs"]),
        name="count",
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试空数据框的值计数
def test_data_frame_value_counts_empty():
    # 创建一个没有列的空数据框
    df_no_cols = pd.DataFrame()

    # 对空数据框进行值计数
    result = df_no_cols.value_counts()
    # 预期结果是一个空的序列，数据类型为 int64，索引为空数组
    expected = pd.Series(
        [], dtype=np.int64, name="count", index=np.array([], dtype=np.intp)
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试空数据框的归一化值计数
def test_data_frame_value_counts_empty_normalize():
    # 创建一个没有列的空数据框
    df_no_cols = pd.DataFrame()

    # 对空数据框进行归一化的值计数
    result = df_no_cols.value_counts(normalize=True)
    # 预期结果是一个空的序列，数据类型为 float64，索引为空数组
    expected = pd.Series(
        [], dtype=np.float64, name="proportion", index=np.array([], dtype=np.intp)
    )

    # 断言函数用于比较结果和预期值是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试在存在空值的情况下是否能够正确处理
def test_data_frame_value_counts_dropna_true(nulls_fixture):
    # GH 41334
    # 创建一个 Pandas 数据框，包含两列：first_name 和 middle_name，分别存储名字和中间名数据
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],  # 列 first_name 包含四个名字
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],  # 列 middle_name 包含特定值和 nulls_fixture 变量的值
        },
    )
    
    # 对数据框 df 进行 value_counts() 操作，生成一个包含计数结果的 Pandas Series
    result = df.value_counts()
    
    # 创建一个预期的 Pandas Series 对象 expected，包含如下数据：
    # - 数据为 [1, 1]
    # - 索引为 pd.MultiIndex 对象，由两个数组构成，分别为 ("Beth", "John") 和 ("Louise", "Smith")，指定了多级索引的名字为 "first_name" 和 "middle_name"
    # - Series 的名字为 "count"
    expected = pd.Series(
        data=[1, 1],
        index=pd.MultiIndex.from_arrays(
            [("Beth", "John"), ("Louise", "Smith")], names=["first_name", "middle_name"]
        ),
        name="count",
    )
    
    # 使用 Pandas 的测试工具 tm 来比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试 DataFrame 的 value_counts 方法在 dropna 参数设置为 False 时的行为
def test_data_frame_value_counts_dropna_false(nulls_fixture):
    # 创建一个 DataFrame，包含两列数据："first_name" 和 "middle_name"
    df = pd.DataFrame(
        {
            "first_name": ["John", "Anne", "John", "Beth"],
            "middle_name": ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )

    # 对 DataFrame 使用 value_counts 方法，设置 dropna 参数为 False，返回结果保存在 result 中
    result = df.value_counts(dropna=False)

    # 创建预期的 Series 对象 expected，包含数据和多级索引
    expected = pd.Series(
        data=[1, 1, 1, 1],  # 数据部分，每个值对应一个组合
        index=pd.MultiIndex(  # 多级索引对象
            levels=[  # 索引的层级，分别为 ["Anne", "Beth", "John"] 和 ["Louise", "Smith", np.nan]
                pd.Index(["Anne", "Beth", "John"]),
                pd.Index(["Louise", "Smith", np.nan]),
            ],
            codes=[[0, 1, 2, 2], [2, 0, 1, 2]],  # 每个层级的代码，与数据对应
            names=["first_name", "middle_name"],  # 每个层级的名称
        ),
        name="count",  # Series 的名称
    )

    # 使用测试模块中的 assert_series_equal 方法，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用参数化装饰器标记的测试函数，用于测试 DataFrame 的 value_counts 方法在指定列名或索引位置时的行为
@pytest.mark.parametrize("columns", (["first_name", "middle_name"], [0, 1]))
def test_data_frame_value_counts_subset(nulls_fixture, columns):
    # 创建一个 DataFrame，根据传入的列名或索引位置创建列，分别为 ["John", "Anne", "John", "Beth"] 和 ["Smith", nulls_fixture, nulls_fixture, "Louise"]
    df = pd.DataFrame(
        {
            columns[0]: ["John", "Anne", "John", "Beth"],
            columns[1]: ["Smith", nulls_fixture, nulls_fixture, "Louise"],
        },
    )

    # 对 DataFrame 使用 value_counts 方法，指定列名或索引位置为 columns[0]，返回结果保存在 result 中
    result = df.value_counts(columns[0])

    # 创建预期的 Series 对象 expected，包含数据和单级索引
    expected = pd.Series(
        data=[2, 1, 1],  # 数据部分，每个值对应一个唯一的索引值
        index=pd.Index(["John", "Anne", "Beth"], name=columns[0]),  # 索引对象，名称为 columns[0]
        name="count",  # Series 的名称
    )

    # 使用测试模块中的 assert_series_equal 方法，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试 DataFrame 中的分类数据在 value_counts 方法中的行为是否引发未来警告
def test_value_counts_categorical_future_warning():
    # 创建一个 DataFrame，包含名为 "a" 的列，列数据类型为 category，并有三个分类值 [1, 2, 3]
    df = pd.DataFrame({"a": [1, 2, 3]}, dtype="category")

    # 对 DataFrame 使用 value_counts 方法，返回结果保存在 result 中
    result = df.value_counts()

    # 创建预期的 Series 对象 expected，包含数据和多级索引
    expected = pd.Series(
        1,  # 单一值的数据部分
        index=pd.MultiIndex.from_arrays(  # 使用数组创建多级索引对象
            [pd.Index([1, 2, 3], name="a", dtype="category")]  # 包含分类名称和类型的单一层级
        ),
        name="count",  # Series 的名称
    )

    # 使用测试模块中的 assert_series_equal 方法，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试 DataFrame 中含有缺失分类数据的情况下，value_counts 方法的行为
def test_value_counts_with_missing_category():
    # 创建一个 DataFrame，包含名为 "a" 的列，列数据类型为分类数据类型，有四个数据项，其中一个数据项缺失
    df = pd.DataFrame({"a": pd.Categorical([1, 2, 4], categories=[1, 2, 3, 4])})

    # 对 DataFrame 使用 value_counts 方法，返回结果保存在 result 中
    result = df.value_counts()

    # 创建预期的 Series 对象 expected，包含数据和多级索引
    expected = pd.Series(
        [1, 1, 1, 0],  # 数据部分，对应每个分类值的计数
        index=pd.MultiIndex.from_arrays(  # 使用数组创建多级索引对象
            [pd.CategoricalIndex([1, 2, 4, 3], categories=[1, 2, 3, 4], name="a")]  # 包含分类名称和分类数据的多级索引
        ),
        name="count",  # Series 的名称
    )

    # 使用测试模块中的 assert_series_equal 方法，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```