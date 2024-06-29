# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_indexing.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

from pandas.core.dtypes.common import is_float_dtype  # 从 Pandas 中导入判断数据类型的函数

import pandas as pd  # 导入 Pandas 库，并命名为 pd
from pandas import (  # 从 Pandas 中导入 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.tests.copy_view.util import get_array  # 从 Pandas 测试工具中导入 get_array 函数


@pytest.fixture(params=["numpy", "nullable"])
def backend(request):
    if request.param == "numpy":
        # 如果请求参数为 "numpy"，定义使用 NumPy 的 make_dataframe 和 make_series 函数

        def make_dataframe(*args, **kwargs):
            return DataFrame(*args, **kwargs)  # 创建并返回 DataFrame 对象

        def make_series(*args, **kwargs):
            return Series(*args, **kwargs)  # 创建并返回 Series 对象

    elif request.param == "nullable":
        # 如果请求参数为 "nullable"，定义使用 Pandas 的 convert_dtypes 后的 make_dataframe 和 make_series 函数

        def make_dataframe(*args, **kwargs):
            df = DataFrame(*args, **kwargs)  # 创建 DataFrame 对象
            df_nullable = df.convert_dtypes()  # 将 DataFrame 转换为可空数据类型
            # 如果转换类型导致浮点数转换为整数，则恢复为浮点数
            for col in df.columns:
                if is_float_dtype(df[col].dtype) and not is_float_dtype(
                    df_nullable[col].dtype
                ):
                    df_nullable[col] = df_nullable[col].astype("Float64")
            return df_nullable.copy()  # 返回深拷贝后的 DataFrame

        def make_series(*args, **kwargs):
            ser = Series(*args, **kwargs)  # 创建 Series 对象
            return ser.convert_dtypes().copy()  # 转换为可空数据类型并进行深拷贝

    return request.param, make_dataframe, make_series  # 返回请求参数和相应的函数


# -----------------------------------------------------------------------------
# Indexing operations taking subset + modifying the subset/parent


def test_subset_column_selection(backend):
    # 测试用例：对 DataFrame 的列进行子集选择后进行修改
    _, DataFrame, _ = backend  # 解包 backend，获取 DataFrame 函数
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()  # 复制原始 DataFrame

    subset = df[["a", "c"]]  # 选择 DataFrame 的子集包含列 'a' 和 'c'

    # subset 与 df 的 'a' 列共享内存 ...
    assert np.shares_memory(get_array(subset, "a"), get_array(df, "a"))
    # ... 但在修改时使用 Copy-on-Write（CoW）机制
    subset.iloc[0, 0] = 0  # 修改 subset 的第一行第一列元素

    assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))  # 确认 'a' 列不再共享内存

    expected = DataFrame({"a": [0, 2, 3], "c": [0.1, 0.2, 0.3]})
    tm.assert_frame_equal(subset, expected)  # 使用 Pandas 测试工具比较 subset 和预期结果
    tm.assert_frame_equal(df, df_orig)  # 使用 Pandas 测试工具比较 df 和原始 DataFrame


def test_subset_column_selection_modify_parent(backend):
    # 测试用例：对 DataFrame 的列进行子集选择后修改原始 DataFrame
    _, DataFrame, _ = backend  # 解包 backend，获取 DataFrame 函数
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})

    subset = df[["a", "c"]]  # 选择 DataFrame 的子集包含列 'a' 和 'c'

    # subset 与 df 的 'a' 列共享内存 ...
    assert np.shares_memory(get_array(subset, "a"), get_array(df, "a"))
    # ... 但在 df 修改时， 'a' 列使用父 DataFrame 的 CoW 机制
    df.iloc[0, 0] = 0  # 修改 df 的第一行第一列元素

    assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))  # 确认 'a' 列不再共享内存
    # 不同的列/块仍然共享内存
    assert np.shares_memory(get_array(subset, "c"), get_array(df, "c"))

    expected = DataFrame({"a": [1, 2, 3], "c": [0.1, 0.2, 0.3]})
    # 比较 subset 和预期结果
    # 使用测试工具 `tm` 中的函数 `assert_frame_equal` 来比较 `subset` 和 `expected` 两个数据框（DataFrame）是否相等。
    tm.assert_frame_equal(subset, expected)
# 定义一个测试函数，用于测试通过切片取DataFrame的子集行
def test_subset_row_slice(backend):
    # 用于存储测试用的DataFrame、DataFrame类和数据类型的元组解包
    _, DataFrame, _ = backend
    # 创建一个DataFrame对象，包含三列（'a', 'b', 'c'），每列有三个元素
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 备份原始的DataFrame对象
    df_orig = df.copy()

    # 通过切片获取df的子集，包含第1行到第2行（索引为1和2）
    subset = df[1:3]
    # 验证子集的数据块管理是否完整
    subset._mgr._verify_integrity()

    # 断言子集的列'a'与df的列'a'共享内存
    assert np.shares_memory(get_array(subset, "a"), get_array(df, "a"))

    # 修改子集的第1行第1列的元素为0，断言此时子集的列'a'与df的列'a'不共享内存
    subset.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(subset, "a"), get_array(df, "a"))

    # 再次验证子集的数据块管理是否完整
    subset._mgr._verify_integrity()

    # 创建预期的DataFrame对象，包含两列（'a', 'b'），两行数据与子集对应（索引为1和2）
    expected = DataFrame({"a": [0, 3], "b": [5, 6], "c": [0.2, 0.3]}, index=range(1, 3))
    # 断言子集与预期的DataFrame相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始的父DataFrame没有被修改（写时复制）
    tm.assert_frame_equal(df, df_orig)


# 使用参数化装饰器进行多组测试参数化配置
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
# 定义一个测试函数，用于测试通过切片取DataFrame的子集列
def test_subset_column_slice(backend, dtype):
    # 解包测试后端数据类型、DataFrame类和数据类型的元组
    dtype_backend, DataFrame, _ = backend
    # 创建一个DataFrame对象，包含三列（'a', 'b', 'c'），数据类型根据dtype参数指定
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    # 备份原始的DataFrame对象
    df_orig = df.copy()

    # 通过iloc切片获取df的子集，包含所有行（:）和从第1列开始的所有列
    subset = df.iloc[:, 1:]
    # 验证子集的数据块管理是否完整
    subset._mgr._verify_integrity()

    # 断言子集的列'b'与df的列'b'共享内存
    assert np.shares_memory(get_array(subset, "b"), get_array(df, "b"))

    # 修改子集的第1行第1列的元素为0，断言此时子集的列'b'与df的列'b'不共享内存
    subset.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(subset, "b"), get_array(df, "b"))

    # 创建预期的DataFrame对象，包含列'b'和'c'，数据类型根据dtype参数指定
    expected = DataFrame({"b": [0, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)})
    # 断言子集与预期的DataFrame相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始的父DataFrame没有被修改（对于BlockManager情况除外，单块除外）
    tm.assert_frame_equal(df, df_orig)


# 使用参数化装饰器进行多组测试参数化配置
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
# 使用参数化装饰器进行多组测试参数化配置
@pytest.mark.parametrize(
    "row_indexer",
    [slice(1, 2), np.array([False, True, True]), np.array([1, 2])],
    ids=["slice", "mask", "array"],
)
# 使用参数化装饰器进行多组测试参数化配置
@pytest.mark.parametrize(
    "column_indexer",
    [slice("b", "c"), np.array([False, True, True]), ["b", "c"]],
    ids=["slice", "mask", "array"],
)
# 定义一个测试函数，用于测试通过.loc获取DataFrame的子集行和列
def test_subset_loc_rows_columns(
    backend,
    dtype,
    row_indexer,
    column_indexer,
):
    # 解包测试后端数据类型、DataFrame类和数据类型的元组
    dtype_backend, DataFrame, _ = backend
    # 创建一个DataFrame对象，包含三列（'a', 'b', 'c'），数据类型根据dtype参数指定
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    # 备份原始的DataFrame对象
    df_orig = df.copy()

    # 通过.loc获取df的子集，行和列根据row_indexer和column_indexer参数指定
    subset = df.loc[row_indexer, column_indexer]

    # 修改子集的第1行第1列的元素为0，确保对子集的修改不会影响父DataFrame
    subset.iloc[0, 0] = 0
    # 创建预期的 DataFrame 对象，包含列 'b' 和 'c'，以及指定的数据和索引范围
    expected = DataFrame(
        {"b": [0, 6], "c": np.array([8, 9], dtype=dtype)}, index=range(1, 3)
    )
    # 使用测试工具库中的函数比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(subset, expected)
    # 使用测试工具库中的函数比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(df, df_orig)
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
@pytest.mark.parametrize(
    "row_indexer",
    [slice(1, 3), np.array([False, True, True]), np.array([1, 2])],
    ids=["slice", "mask", "array"],
)
@pytest.mark.parametrize(
    "column_indexer",
    [slice(1, 3), np.array([False, True, True]), [1, 2]],
    ids=["slice", "mask", "array"],
)
# 定义测试函数，用于测试从 DataFrame 中取子集，并修改子集后不影响原始 DataFrame
def test_subset_iloc_rows_columns(
    backend,
    dtype,
    row_indexer,
    column_indexer,
):
    # Case: taking a subset of the rows+columns of a DataFrame using .iloc
    # + afterwards modifying the subset
    # Generic test for several combinations of row/column indexers, not all
    # of those could actually return a view / need CoW (so this test is not
    # checking memory sharing, only ensuring subsequent mutation doesn't
    # affect the parent dataframe)
    # 获取后端类型和 DataFrame 类，以及一些测试工具
    dtype_backend, DataFrame, _ = backend
    # 创建一个 DataFrame 包含三列 'a', 'b', 'c'，用给定的 dtype
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 从 DataFrame 中取子集，使用给定的行和列索引器
    subset = df.iloc[row_indexer, column_indexer]

    # 修改子集，验证修改不影响原始 DataFrame
    subset.iloc[0, 0] = 0

    # 预期的子集 DataFrame，仅包含特定的行和列
    expected = DataFrame(
        {"b": [0, 6], "c": np.array([8, 9], dtype=dtype)}, index=range(1, 3)
    )
    # 断言子集与预期结果相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始 DataFrame 与备份相等，验证修改不影响原始 DataFrame
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "indexer",
    [slice(0, 2), np.array([True, True, False]), np.array([0, 1])],
    ids=["slice", "mask", "array"],
)
# 定义测试函数，测试在视图子集上使用行索引器进行设置值的情况
def test_subset_set_with_row_indexer(backend, indexer_si, indexer):
    # Case: setting values with a row indexer on a viewing subset
    # subset[indexer] = value and subset.iloc[indexer] = value
    _, DataFrame, _ = backend
    # 创建一个 DataFrame 包含三列 'a', 'b', 'c'，并备份原始 DataFrame
    df = DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "c": [0.1, 0.2, 0.3, 0.4]})
    df_orig = df.copy()
    # 从 DataFrame 中取视图子集
    subset = df[1:4]

    # 根据不同情况使用索引器设置子集的值为 0
    if (
        indexer_si is tm.setitem
        and isinstance(indexer, np.ndarray)
        and indexer.dtype == "int"
    ):
        pytest.skip("setitem with labels selects on columns")

    indexer_si(subset)[indexer] = 0

    # 预期的子集 DataFrame
    expected = DataFrame(
        {"a": [0, 0, 4], "b": [0, 0, 7], "c": [0.0, 0.0, 0.4]}, index=range(1, 4)
    )
    # 断言子集与预期结果相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始 DataFrame 与备份相等，验证修改不影响原始 DataFrame
    tm.assert_frame_equal(df, df_orig)


# 定义测试函数，测试在视图子集上使用掩码设置值的情况
def test_subset_set_with_mask(backend):
    # Case: setting values with a mask on a viewing subset: subset[mask] = value
    _, DataFrame, _ = backend
    # 创建一个 DataFrame 包含三列 'a', 'b', 'c'，并备份原始 DataFrame
    df = DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7], "c": [0.1, 0.2, 0.3, 0.4]})
    df_orig = df.copy()
    # 从 DataFrame 中取视图子集
    subset = df[1:4]

    # 根据掩码设置子集的值为 0
    mask = subset > 3
    subset[mask] = 0

    # 预期的子集 DataFrame
    expected = DataFrame(
        {"a": [2, 3, 0], "b": [0, 0, 0], "c": [0.20, 0.3, 0.4]}, index=range(1, 4)
    )
    # 断言子集与预期结果相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始 DataFrame 与备份相等，验证修改不影响原始 DataFrame
    tm.assert_frame_equal(df, df_orig)


# 定义测试函数，测试在视图子集上设置单列值的情况
def test_subset_set_column(backend):
    # Case: setting a single column on a viewing subset -> subset[col] = value
    # 从 backend 中解包出 dtype_backend, DataFrame, _ 三个变量
    dtype_backend, DataFrame, _ = backend
    # 创建一个 DataFrame 对象 df，包含三列数据 "a", "b", "c"
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制 df，生成 df_orig，保留原始数据副本
    df_orig = df.copy()
    # 从 df 中取出索引为 1 到 2 的子集，存入 subset
    subset = df[1:3]
    
    # 根据 dtype_backend 的值选择相应的数组类型创建 arr
    if dtype_backend == "numpy":
        arr = np.array([10, 11], dtype="int64")
    else:
        arr = pd.array([10, 11], dtype="Int64")
    
    # 将 arr 的值赋给 subset 的 "a" 列
    subset["a"] = arr
    # 检查 subset 内部的数据管理器确保数据完整性
    subset._mgr._verify_integrity()
    
    # 创建期望的 DataFrame 对象 expected，与 subset 数据相符
    expected = DataFrame(
        {"a": [10, 11], "b": [5, 6], "c": [0.2, 0.3]}, index=range(1, 3)
    )
    # 使用断言检查 subset 是否等于 expected
    tm.assert_frame_equal(subset, expected)
    # 使用断言检查 df 是否等于 df_orig，验证操作是否影响原始数据
    tm.assert_frame_equal(df, df_orig)
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
# 参数化测试：测试不同的数据类型（int64 和 float64），并为每种情况指定标识符（单一块、混合块）
def test_subset_set_column_with_loc(backend, dtype):
    # 情况：在视图子集上使用 loc 设置单列
    # -> subset.loc[:, col] = value
    _, DataFrame, _ = backend
    # 创建一个包含不同数据类型列的 DataFrame 对象
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    # 备份原始 DataFrame
    df_orig = df.copy()
    # 从原 DataFrame 中获取一个子集
    subset = df[1:3]

    # 在子集上使用 loc 设置列 "a" 的值为指定的数组
    subset.loc[:, "a"] = np.array([10, 11], dtype="int64")

    # 验证数据完整性
    subset._mgr._verify_integrity()
    # 期望的结果 DataFrame，应包含修改后的数据
    expected = DataFrame(
        {"a": [10, 11], "b": [5, 6], "c": np.array([8, 9], dtype=dtype)},
        index=range(1, 3),
    )
    # 断言子集与期望结果 DataFrame 相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始父 DataFrame 未被修改（写时复制）
    tm.assert_frame_equal(df, df_orig)


def test_subset_set_column_with_loc2(backend):
    # 情况：在视图子集上使用 loc 设置单列
    # -> subset.loc[:, col] = value
    # 单独测试仅包含单列的 DataFrame 的情况，使用不同的代码路径
    _, DataFrame, _ = backend
    # 创建一个包含单列 "a" 的 DataFrame 对象
    df = DataFrame({"a": [1, 2, 3]})
    # 备份原始 DataFrame
    df_orig = df.copy()
    # 从原 DataFrame 中获取一个子集
    subset = df[1:3]

    # 在子集上使用 loc 设置列 "a" 的值为 0
    subset.loc[:, "a"] = 0

    # 验证数据完整性
    subset._mgr._verify_integrity()
    # 期望的结果 DataFrame，应包含修改后的数据
    expected = DataFrame({"a": [0, 0]}, index=range(1, 3))
    # 断言子集与期望结果 DataFrame 相等
    tm.assert_frame_equal(subset, expected)
    # 断言原始父 DataFrame 未被修改（写时复制）
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
# 参数化测试：测试不同的数据类型（int64 和 float64），并为每种情况指定标识符（单一块、混合块）
def test_subset_set_columns(backend, dtype):
    # 情况：在视图子集上设置多列数据
    # -> subset[[col1, col2]] = value
    dtype_backend, DataFrame, _ = backend
    # 创建一个包含不同数据类型列的 DataFrame 对象
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    # 备份原始 DataFrame
    df_orig = df.copy()
    # 从原 DataFrame 中获取一个子集
    subset = df[1:3]

    # 在子集上设置列 "a" 和 "c" 的值为 0
    subset[["a", "c"]] = 0

    # 验证数据完整性
    subset._mgr._verify_integrity()
    # 断言子集与期望结果 DataFrame 相等
    expected = DataFrame({"a": [0, 0], "b": [5, 6], "c": [0, 0]}, index=range(1, 3))
    if dtype_backend == "nullable":
        # 如果后端支持 nullable 类型，则重写列时设置标量默认为 numpy 数据类型，即使原始列是 nullable
        expected["a"] = expected["a"].astype("int64")
        expected["c"] = expected["c"].astype("int64")

    tm.assert_frame_equal(subset, expected)
    # 断言原始父 DataFrame 未被修改（写时复制）
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "indexer",
    [slice("a", "b"), np.array([True, True, False]), ["a", "b"]],
    ids=["slice", "mask", "array"],
)
# 参数化测试：测试不同的索引器类型（切片、掩码、数组）
def test_subset_set_with_column_indexer(backend, indexer):
    # 情况：在视图子集上使用列索引器设置多列数据
    # -> subset.loc[:, [col1, col2]] = value
    _, DataFrame, _ = backend
    # 创建一个包含列 "a", "b", "c" 的 DataFrame 对象
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": [4, 5, 6]})
    # 备份原始 DataFrame
    df_orig = df.copy()
    # 从DataFrame中选择第1到第2行（索引为1到2，不包括索引3），创建一个子集
    subset = df[1:3]
    
    # 在子集中的所有行上，针对索引器所指示的列，将所有元素设置为0
    subset.loc[:, indexer] = 0
    
    # 验证子集的内部一致性，确保操作后的DataFrame符合预期
    subset._mgr._verify_integrity()
    
    # 创建一个预期的DataFrame，包含指定列和索引范围内的值
    expected = DataFrame({"a": [0, 0], "b": [0.0, 0.0], "c": [5, 6]}, index=range(1, 3))
    
    # 使用测试工具比较子集和预期DataFrame，确认它们相等
    tm.assert_frame_equal(subset, expected)
    
    # 使用测试工具比较原始DataFrame和操作前的备份DataFrame，确认它们相等
    tm.assert_frame_equal(df, df_orig)
@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    "method",  # 参数名称为 method
    [  # 参数取值列表，每个元素是一个 lambda 函数，用于生成不同的 DataFrame 或 Series 子集
        lambda df: df[["a", "b"]][0:2],  # 选择列 'a' 和 'b'，然后选择前两行的子集
        lambda df: df[0:2][["a", "b"]],  # 选择前两行，然后选择列 'a' 和 'b' 的子集
        lambda df: df[["a", "b"]].iloc[0:2],  # 使用 iloc 选择前两行，然后选择列 'a' 和 'b' 的子集
        lambda df: df[["a", "b"]].loc[0:1],  # 使用 loc 按标签选择行 '0' 和 '1'，然后选择列 'a' 和 'b' 的子集
        lambda df: df[0:2].iloc[:, 0:2],  # 使用 iloc 选择前两行，然后再使用 iloc 选择前两列的子集
        lambda df: df[0:2].loc[:, "a":"b"],  # 使用 loc 按标签选择前两行，并选择列 'a' 到 'b' 的子集
    ],
    ids=[  # 每个参数取值的标识符列表，对应方法列表中的每个方法
        "row-getitem-slice",  # 第一个方法标识符：行-getitem-切片
        "column-getitem",  # 第二个方法标识符：列-getitem
        "row-iloc-slice",  # 第三个方法标识符：行-iloc-切片
        "row-loc-slice",  # 第四个方法标识符：行-loc-切片
        "column-iloc-slice",  # 第五个方法标识符：列-iloc-切片
        "column-loc-slice",  # 第六个方法标识符：列-loc-切片
    ],
)
@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器再次定义参数化测试
    "dtype",  # 参数名称为 dtype
    ["int64", "float64"],  # 参数取值列表为 "int64" 和 "float64"
    ids=["single-block", "mixed-block"],  # 对应取值的标识符列表
)
def test_subset_chained_getitem(  # 定义测试函数 test_subset_chained_getitem
    request,  # pytest 的 request 对象，用于访问测试请求上下文
    backend,  # 测试参数，表示测试后端
    method,  # 参数 method，用于生成 DataFrame 或 Series 的子集
    dtype,  # 参数 dtype，表示数据类型 "int64" 或 "float64"
):
    # Case: creating a subset using multiple, chained getitem calls using views
    # still needs to guarantee proper CoW behaviour
    _, DataFrame, _ = backend  # 解包 backend，获取 DataFrame 类型
    df = DataFrame(  # 创建 DataFrame df
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}  # DataFrame 的数据和列类型
    )
    df_orig = df.copy()  # 复制 df，以备后续比较使用

    # modify subset -> don't modify parent
    subset = method(df)  # 使用参数 method 生成 df 的子集
    subset.iloc[0, 0] = 0  # 修改 subset 的第一行第一列元素为 0
    tm.assert_frame_equal(df, df_orig)  # 断言 df 未被修改，与原始 df 相等

    # modify parent -> don't modify subset
    subset = method(df)  # 再次使用参数 method 生成 df 的子集
    df.iloc[0, 0] = 0  # 修改 df 的第一行第一列元素为 0
    expected = DataFrame({"a": [1, 2], "b": [4, 5]})  # 预期的修改后 DataFrame
    tm.assert_frame_equal(subset, expected)  # 断言 subset 与预期 DataFrame 相等


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    "dtype",  # 参数名称为 dtype
    ["int64", "float64"],  # 参数取值列表为 "int64" 和 "float64"
    ids=["single-block", "mixed-block"],  # 对应取值的标识符列表
)
def test_subset_chained_getitem_column(  # 定义测试函数 test_subset_chained_getitem_column
    backend,  # 测试参数，表示测试后端
    dtype,  # 参数 dtype，表示数据类型 "int64" 或 "float64"
):
    # Case: creating a subset using multiple, chained getitem calls using views
    # still needs to guarantee proper CoW behaviour
    dtype_backend, DataFrame, Series = backend  # 解包 backend，获取 DataFrame 和 Series 类型
    df = DataFrame(  # 创建 DataFrame df
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}  # DataFrame 的数据和列类型
    )
    df_orig = df.copy()  # 复制 df，以备后续比较使用

    # modify subset -> don't modify parent
    subset = df[:]["a"][0:2]  # 使用链式 getitem 调用选择 df 的子集
    subset.iloc[0] = 0  # 修改 subset 的第一个元素为 0
    tm.assert_frame_equal(df, df_orig)  # 断言 df 未被修改，与原始 df 相等

    # modify parent -> don't modify subset
    subset = df[:]["a"][0:2]  # 再次使用链式 getitem 调用选择 df 的子集
    df.iloc[0, 0] = 0  # 修改 df 的第一行第一列元素为 0
    expected = Series([1, 2], name="a")  # 预期的修改后 Series
    tm.assert_series_equal(subset, expected)  # 断言 subset 与预期 Series 相等


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器定义参数化测试
    "method",  # 参数名称为 method
    [  # 参数取值列表，每个元素是一个 lambda 函数，用于生成不同的 Series 子集
        lambda s: s["a":"c"]["a":"b"],  # 选择索引标签 'a' 到 'c' 的子集，然后再选择列 'a' 到 'b' 的子集
        lambda s: s.iloc[0:3].iloc[0:2],  # 使用 iloc 选择前三行，然后再选择前两行的子集
        lambda s: s.loc["a":"c"].loc["a":"b"],  # 使用 loc 选择标签 'a' 到 'c' 的子集，然后再选择标签 'a' 到 'b' 的子集
        lambda s: s.loc["a":"c"]  # 使用 loc 选择标签 'a' 到 'c' 的子集
        .iloc[0:3]  # 再使用 iloc 选择前三行
        .iloc[0:2]  # 再使用 iloc 选择前两行
        .loc["a":"b"]  # 最后使用 loc 选择标签 'a' 到 'b' 的子集
        .iloc[0:1],  # 最终使用 iloc 选择第一行
    ],
    ids=[  # 每个参数取值的标识符列表
        "getitem",  # 第一个方法标识符：getitem
        "iloc",  # 第二个方法标识符：iloc
        "loc",  # 第三个方法标识符：loc
        "long-chain",  # 第四个方法标识符：long-chain
    ],
)
def test_subset_chained_getitem_series(  # 定义测试函数 test_subset_chained_getitem_series
    backend,  # 测试参数，表示测试后端
    method,  # 参数 method，用于生成 Series 的子集
):
    # Case: creating a subset using multiple, chained getitem calls using views
    # still needs to guarantee proper CoW behaviour
    # 从 Series s 中选择子集，包括第 0 至 2 行（不包括第 3 行），再从中选择第 0 至 1 行（不包括第 2 行）
    subset = s.iloc[0:3].iloc[0:2]
    
    # 将 Series s 中第 0 行的值设为 0
    s.iloc[0] = 0
    
    # 创建一个期望的 Series，包含值为 1 和 2，索引分别为 "a" 和 "b"
    expected = Series([1, 2], index=["a", "b"])
    
    # 使用测试框架中的方法比较 subset 和 expected 是否相等
    tm.assert_series_equal(subset, expected)
def test_subset_chained_single_block_row():
    # 不针对 dtype 后端进行参数化，因为这里显式测试单个块
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()

    # 修改子集 -> 不修改原始数据框
    subset = df[:].iloc[0].iloc[0:2]  # 选择数据框的子集，深复制后取第一行再取第一到第二列的子集
    subset.iloc[0] = 0  # 修改子集的第一个元素为0
    tm.assert_frame_equal(df, df_orig)  # 断言确认修改没有影响原始数据框

    # 修改原始数据框 -> 不修改子集
    subset = df[:].iloc[0].iloc[0:2]  # 重新选择数据框的子集
    df.iloc[0, 0] = 0  # 修改原始数据框的第一个元素为0
    expected = Series([1, 4], index=["a", "b"], name=0)
    tm.assert_series_equal(subset, expected)  # 断言确认子集没有被修改


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df[:],
        lambda df: df.loc[:, :],
        lambda df: df.loc[:],
        lambda df: df.iloc[:, :],
        lambda df: df.iloc[:],
    ],
    ids=["getitem", "loc", "loc-rows", "iloc", "iloc-rows"],
)
def test_null_slice(backend, method):
    # 场景：测试使用空切片(:)进行的各种索引操作，应返回新对象以确保正确使用 CoW（写时复制）进行结果处理
    dtype_backend, DataFrame, _ = backend
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df_orig = df.copy()

    df2 = method(df)  # 使用参数化方法选择数据框的新副本

    # 我们始终返回新对象（浅复制），无论是否使用 CoW
    assert df2 is not df

    # 修改这些对象会触发 CoW
    df2.iloc[0, 0] = 0  # 修改新副本的第一个元素为0
    tm.assert_frame_equal(df, df_orig)  # 断言确认原始数据框没有被修改


@pytest.mark.parametrize(
    "method",
    [
        lambda s: s[:],
        lambda s: s.loc[:],
        lambda s: s.iloc[:],
    ],
    ids=["getitem", "loc", "iloc"],
)
def test_null_slice_series(backend, method):
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()

    s2 = method(s)  # 使用参数化方法选择系列的新副本

    # 我们始终返回新对象，无论是否使用 CoW
    assert s2 is not s

    # 修改这些对象会触发 CoW
    s2.iloc[0] = 0  # 修改新副本的第一个元素为0
    tm.assert_series_equal(s, s_orig)  # 断言确认原始系列没有被修改


# TODO add more tests modifying the parent


# -----------------------------------------------------------------------------
# Series -- Indexing operations taking subset + modifying the subset/parent


def test_series_getitem_slice(backend):
    # 场景：获取系列的切片并修改子集
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()

    subset = s[:]
    assert np.shares_memory(get_array(subset), get_array(s))  # 断言确认子集与原始系列共享内存

    subset.iloc[0] = 0  # 修改子集的第一个元素为0

    assert not np.shares_memory(get_array(subset), get_array(s))  # 断言确认修改后子集与原始系列不再共享内存

    expected = Series([0, 2, 3], index=["a", "b", "c"])
    tm.assert_series_equal(subset, expected)  # 断言确认子集的值符合预期

    # 原始父系列没有被修改（CoW）
    tm.assert_series_equal(s, s_orig)  # 断言确认原始系列没有被修改


def test_series_getitem_ellipsis():
    # 场景：使用省略号获取系列的视图，并修改子集
    s = Series([1, 2, 3])
    s_orig = s.copy()

    subset = s[...]  # 使用省略号获取整个系列的视图
    assert np.shares_memory(get_array(subset), get_array(s))  # 断言确认子集与原始系列共享内存

    subset.iloc[0] = 0  # 修改子集的第一个元素为0
    # 确保 subset 和 s 不共享内存，即它们是独立的对象
    assert not np.shares_memory(get_array(subset), get_array(s))
    
    # 验证 subset 和期望的 Series 对象相等
    expected = Series([0, 2, 3])
    tm.assert_series_equal(subset, expected)
    
    # 确保原始的父级 Series 没有被修改（写时复制机制）
    tm.assert_series_equal(s, s_orig)
@pytest.mark.parametrize(
    "indexer",
    [slice(0, 2), np.array([True, True, False]), np.array([0, 1])],
    ids=["slice", "mask", "array"],
)
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_series_subset_set_with_indexer 参数化设置多组参数，并指定每组参数的标识符

def test_series_subset_set_with_indexer(backend, indexer_si, indexer):
    # Case: setting values in a viewing Series with an indexer
    # 创建 Series 对象 s，包含数据 [1, 2, 3] 和索引 ["a", "b", "c"]，并复制给 s_orig
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()
    # 创建 s 的切片 subset
    subset = s[:]

    # 根据不同条件执行不同操作
    if (
        indexer_si is tm.setitem
        and isinstance(indexer, np.ndarray)
        and indexer.dtype.kind == "i"
    ):
        # In 3.0 we treat integers as always-labels
        # 如果 indexer_si 是 tm.setitem，且 indexer 是 np.ndarray 且其数据类型为整数，抛出 KeyError 异常
        with pytest.raises(KeyError):
            indexer_si(subset)[indexer] = 0
        return

    # 使用 indexer_si 对 subset 应用索引器 indexer，并将其设置为 0
    indexer_si(subset)[indexer] = 0
    # 创建期望结果 Series 对象 expected，包含数据 [0, 0, 3] 和索引 ["a", "b", "c"]
    expected = Series([0, 0, 3], index=["a", "b", "c"])
    # 断言 subset 和 expected 相等
    tm.assert_series_equal(subset, expected)
    # 断言 s 和 s_orig 相等
    tm.assert_series_equal(s, s_orig)


# -----------------------------------------------------------------------------
# del operator


def test_del_frame(backend):
    # Case: deleting a column with `del` on a viewing child dataframe should
    # not modify parent + update the references
    # 获取 DataFrame 类型和其它对象的引用
    dtype_backend, DataFrame, _ = backend
    # 创建 DataFrame 对象 df，包含列 "a", "b", "c" 的数据和索引
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    # 创建 df 的切片 df2
    df2 = df[:]

    # 断言 df 和 df2 的 "a" 列共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 使用 del 删除 df2 的 "b" 列
    del df2["b"]

    # 断言 df 和 df2 的 "a" 列仍然共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    # 断言 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)
    # 断言 df2 和 df_orig 的子集相等，包含列 "a", "c"
    tm.assert_frame_equal(df2, df_orig[["a", "c"]])
    # 验证 df2 的数据完整性
    df2._mgr._verify_integrity()

    # 修改 df 的 "b" 列中第一个元素的值为 200
    df.loc[0, "b"] = 200
    # 断言 df 和 df2 的 "a" 列仍然共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    # 更新 df_orig
    df_orig = df.copy()

    # 修改 df2 的 "a" 列中第一个元素的值为 100
    # 删除列后修改子对象仍不更新父对象
    df2.loc[0, "a"] = 100
    # 断言 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


def test_del_series(backend):
    # 创建 Series 对象 s 和其复制 s_orig
    _, _, Series = backend
    s = Series([1, 2, 3], index=["a", "b", "c"])
    s_orig = s.copy()
    # 创建 s 的切片 s2
    s2 = s[:]

    # 断言 s 和 s2 的数据共享内存
    assert np.shares_memory(get_array(s), get_array(s2))

    # 使用 del 删除 s2 的 "a" 元素
    del s2["a"]

    # 断言 s 和 s2 的数据不共享内存
    assert not np.shares_memory(get_array(s), get_array(s2))
    # 断言 s 和 s_orig 相等
    tm.assert_series_equal(s, s_orig)
    # 断言 s2 和 s_orig 的子集相等，包含索引 "b", "c"
    tm.assert_series_equal(s2, s_orig[["b", "c"]])

    # 由于 `del` 操作，修改 s2 不需要写时复制（由新数组支持）
    values = s2.values
    s2.loc["b"] = 100
    assert values[0] == 100


# -----------------------------------------------------------------------------
# Accessing column as Series


def test_column_as_series(backend):
    # Case: selecting a single column now also uses Copy-on-Write
    # 获取后端数据类型，DataFrame 和 Series 的引用
    dtype_backend, DataFrame, Series = backend
    # 创建 DataFrame 对象 df，包含列 "a", "b", "c" 的数据和索引
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()

    # 选择 df 的 "a" 列作为 Series 对象 s
    s = df["a"]

    # 断言 s 和 df 的 "a" 列共享内存
    assert np.shares_memory(get_array(s, "a"), get_array(df, "a"))
    # 修改 s 的第一个元素值为 0
    s[0] = 0

    # 创建期望结果 Series 对象 expected，包含数据 [0, 2, 3] 和名称 "a"
    expected = Series([0, 2, 3], name="a")
    # 断言 s 和 expected 相等
    tm.assert_series_equal(s, expected)
    # 断言 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)
    # 确保通过索引获取的缓存系列不是已更改的系列
    tm.assert_series_equal(df["a"], df_orig["a"])
# 定义测试函数，测试列作为 Series 时的设定行为，使用给定的后端（backend）功能
def test_column_as_series_set_with_upcast(backend):
    # Case: selecting a single column now also uses Copy-on-Write -> when
    # setting a value causes an upcast, we don't need to update the parent
    # DataFrame through the cache mechanism
    dtype_backend, DataFrame, Series = backend
    # 创建一个 DataFrame 包含三列数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame
    df_orig = df.copy()

    # 选择 DataFrame 的列 'a' 并赋值给变量 s
    s = df["a"]
    # 根据后端的数据类型选择不同的测试方式
    if dtype_backend == "nullable":
        # 使用 pytest 验证设定非法值会抛出 TypeError 异常，并匹配指定的错误信息
        with pytest.raises(TypeError, match="Invalid value"):
            s[0] = "foo"
        # 期望的结果是一个 Series，包含原始数据，列名为 'a'
        expected = Series([1, 2, 3], name="a")
    else:
        # 使用 tm.assert_produces_warning 验证设定值会产生 FutureWarning 警告，并匹配指定的警告信息
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            s[0] = "foo"
        # 期望的结果是一个 Series，包含设定后的数据，数据类型为 object，列名为 'a'
        expected = Series(["foo", 2, 3], dtype=object, name="a")

    # 验证 s 是否与期望的结果 expected 相等
    tm.assert_series_equal(s, expected)
    # 验证整个 DataFrame 是否与原始的 df_orig 相等
    tm.assert_frame_equal(df, df_orig)
    # 确保通过获取列 'a' 返回的缓存 Series 不是已更改的 Series
    tm.assert_series_equal(df["a"], df_orig["a"])


# 使用参数化装饰器标记的测试函数，分别测试不同的索引方法对 Series 的影响
@pytest.mark.parametrize(
    "method",
    [
        lambda df: df["a"],
        lambda df: df.loc[:, "a"],
        lambda df: df.iloc[:, 0],
    ],
    ids=["getitem", "loc", "iloc"],
)
def test_column_as_series_no_item_cache(request, backend, method):
    # Case: selecting a single column (which now also uses Copy-on-Write to protect
    # the view) should always give a new object (i.e. not make use of a cache)
    dtype_backend, DataFrame, _ = backend
    # 创建一个 DataFrame 包含三列数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame
    df_orig = df.copy()

    # 使用给定的 method 函数从 DataFrame 中选择列，并赋值给 s1 和 s2
    s1 = method(df)
    s2 = method(df)

    # 断言 s1 和 s2 不是同一个对象
    assert s1 is not s2

    # 修改 s1 的第一个元素为 0
    s1.iloc[0] = 0

    # 验证 s2 是否与原始 DataFrame 中的列 'a' 相等
    tm.assert_series_equal(s2, df_orig["a"])
    # 验证整个 DataFrame 是否与原始的 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


# 定义测试函数，测试从 Series 添加新列到 DataFrame 的行为
def test_dataframe_add_column_from_series(backend):
    # Case: adding a new column to a DataFrame from an existing column/series
    # -> delays copy under CoW
    _, DataFrame, Series = backend
    # 创建一个 DataFrame 包含两列数据
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})

    # 创建一个 Series 包含三个元素
    s = Series([10, 11, 12])
    # 将 Series 添加为 DataFrame 的新列 'new'
    df["new"] = s
    # 断言新添加的列 'new' 和 Series s 使用了相同的内存
    assert np.shares_memory(get_array(df, "new"), get_array(s))

    # 修改 Series s 的第一个元素为 0
    s[0] = 0
    # 期望的结果 DataFrame 包含三列数据，'new' 列的第一个值没有被修改
    expected = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "new": [10, 11, 12]})
    tm.assert_frame_equal(df, expected)


# 使用参数化装饰器标记的测试函数，测试设置值时的行为，验证只有必要的列会被复制
@pytest.mark.parametrize("val", [100, "a"])
@pytest.mark.parametrize(
    "indexer_func, indexer",
    [
        (tm.loc, (0, "a")),
        (tm.iloc, (0, 0)),
        (tm.loc, ([0], "a")),
        (tm.iloc, ([0], 0)),
        (tm.loc, (slice(None), "a")),
        (tm.iloc, (slice(None), 0)),
    ],
)
@pytest.mark.parametrize(
    "col", [[0.1, 0.2, 0.3], [7, 8, 9]], ids=["mixed-block", "single-block"]
)
def test_set_value_copy_only_necessary_column(indexer_func, indexer, val, col):
    # When setting inplace, only copy column that is modified instead of the whole
    # block (by splitting the block)
    # 创建一个 DataFrame 包含三列数据，其中 'c' 列的数据由参数化装饰器指定
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": col})
    # 复制原始数据框 df 到 df_orig
    df_orig = df.copy()
    
    # 创建视图 view，视图与 df 相同
    view = df[:]
    
    # 如果 val 等于 "a"，则执行以下操作
    if val == "a":
        # 使用 assert_produces_warning 上下文管理器，检查未来警告，匹配指定警告信息
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype is deprecated"
        ):
            # 使用 indexer_func 函数获取索引器并将其应用于 df 的 indexer 位置，设置为 val
            indexer_func(df)[indexer] = val
    
    # 使用 indexer_func 函数获取索引器并将其应用于 df 的 indexer 位置，设置为 val
    indexer_func(df)[indexer] = val
    
    # 断言 df 中的 "b" 列和 view 中的 "b" 列共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(view, "b"))
    
    # 断言 df 中的 "a" 列和 view 中的 "a" 列不共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(view, "a"))
    
    # 使用 tm.assert_frame_equal 函数断言视图 view 与原始数据框 df_orig 相等
    tm.assert_frame_equal(view, df_orig)
def test_series_midx_slice():
    # 创建一个带有多级索引的 Pandas Series 对象
    ser = Series([1, 2, 3], index=pd.MultiIndex.from_arrays([[1, 1, 2], [3, 4, 5]]))
    # 备份原始的 Series 对象
    ser_orig = ser.copy()
    # 对 Series 进行切片操作，选择索引为 1 的部分
    result = ser[1]
    # 断言新切片的数据与原始数据共享内存
    assert np.shares_memory(get_array(ser), get_array(result))
    # 修改切片后的数据的第一个元素为 100
    result.iloc[0] = 100
    # 断言修改后的 Series 与原始 Series 相等
    tm.assert_series_equal(ser, ser_orig)


def test_getitem_midx_slice():
    # 创建一个带有多级索引的 Pandas DataFrame 对象
    df = DataFrame({("a", "x"): [1, 2], ("a", "y"): 1, ("b", "x"): 2})
    # 备份原始的 DataFrame 对象
    df_orig = df.copy()
    # 对 DataFrame 进行切片操作，选择索引为 ("a",) 的部分
    new_df = df[("a",)]

    # 断言新切片的数据不是没有引用
    assert not new_df._mgr._has_no_reference(0)

    # 断言原 DataFrame 和新切片的数据共享内存
    assert np.shares_memory(get_array(df, ("a", "x")), get_array(new_df, "x"))
    # 修改新切片的数据的第一个元素为 100
    new_df.iloc[0, 0] = 100
    # 断言修改后的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(df_orig, df)


def test_series_midx_tuples_slice():
    # 创建一个带有元组作为多级索引的 Pandas Series 对象
    ser = Series(
        [1, 2, 3],
        index=pd.MultiIndex.from_tuples([((1, 2), 3), ((1, 2), 4), ((2, 3), 4)]),
    )
    # 对 Series 进行元组索引的切片操作，选择索引为 (1, 2) 的部分
    result = ser[(1, 2)]
    # 断言新切片的数据与原始数据共享内存
    assert np.shares_memory(get_array(ser), get_array(result))
    # 修改切片后的数据的第一个元素为 100
    result.iloc[0] = 100
    # 创建预期的 Series 对象，用于断言修改后的 Series 与预期的相等
    expected = Series(
        [1, 2, 3],
        index=pd.MultiIndex.from_tuples([((1, 2), 3), ((1, 2), 4), ((2, 3), 4)]),
    )
    tm.assert_series_equal(ser, expected)


def test_midx_read_only_bool_indexer():
    # 创建一个带有多级索引的 Pandas DataFrame 对象
    def mklbl(prefix, n):
        return [f"{prefix}{i}" for i in range(n)]

    idx = pd.MultiIndex.from_product(
        [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
    )
    cols = pd.MultiIndex.from_tuples(
        [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
    )
    df = DataFrame(1, index=idx, columns=cols).sort_index().sort_index(axis=1)

    # 创建一个布尔索引器
    mask = df[("a", "foo")] == 1
    expected_mask = mask.copy()
    # 对 DataFrame 使用布尔索引器进行 loc 定位和切片操作
    result = df.loc[pd.IndexSlice[mask, :, ["C1", "C3"]], :]
    expected = df.loc[pd.IndexSlice[:, :, ["C1", "C3"]], :]
    tm.assert_frame_equal(result, expected)
    tm.assert_series_equal(mask, expected_mask)


def test_loc_enlarging_with_dataframe():
    # 创建一个 Pandas DataFrame 对象
    df = DataFrame({"a": [1, 2, 3]})
    # 创建一个右手边的 Pandas DataFrame 对象
    rhs = DataFrame({"b": [1, 2, 3], "c": [4, 5, 6]})
    # 备份右手边的 DataFrame 对象
    rhs_orig = rhs.copy()
    # 使用 loc 扩展的方式将 rhs 的列 'b' 和 'c' 赋值给 df
    df.loc[:, ["b", "c"]] = rhs
    # 断言 df 的列 'b' 与 rhs 的列 'b' 共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))
    # 断言 df 的列 'c' 与 rhs 的列 'c' 共享内存
    assert np.shares_memory(get_array(df, "c"), get_array(rhs, "c"))
    # 断言 df 的第二列不是没有引用
    assert not df._mgr._has_no_reference(1)

    # 修改 df 的第一个元素的第二列为 100
    df.iloc[0, 1] = 100
    # 断言修改后的 rhs 与原始 rhs 相等
    tm.assert_frame_equal(rhs, rhs_orig)
```