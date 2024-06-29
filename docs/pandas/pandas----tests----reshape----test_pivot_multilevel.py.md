# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_pivot_multilevel.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库及其内部模块
from pandas._libs import lib
import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器定义多组参数化测试用例
@pytest.mark.parametrize(
    # 参数化测试的参数，包括输入的索引、列、数值、期望结果的值、列和索引等
    "input_index, input_columns, input_values, "
    "expected_values, expected_columns, expected_index",
    [
        (
            ["lev4"],  # 输入的索引为包含字符串 'lev4' 的列表
            "lev3",    # 输入的列为字符串 'lev3'
            "values",  # 输入的数值为字符串 'values'
            # 期望的数值结果，是一个包含 NaN 和具体数值的二维数组
            [
                [0.0, np.nan],
                [np.nan, 1.0],
                [2.0, np.nan],
                [np.nan, 3.0],
                [4.0, np.nan],
                [np.nan, 5.0],
                [6.0, np.nan],
                [np.nan, 7.0],
            ],
            Index([1, 2], name="lev3"),  # 期望的列索引对象
            Index([1, 2, 3, 4, 5, 6, 7, 8], name="lev4"),  # 期望的行索引对象
        ),
        (
            ["lev4"],  # 输入的索引为包含字符串 'lev4' 的列表
            "lev3",    # 输入的列为字符串 'lev3'
            lib.no_default,  # 输入的数值为 lib.no_default，表示特殊值
            # 期望的数值结果，是一个包含 NaN 和具体数值的二维数组
            [
                [1.0, np.nan, 1.0, np.nan, 0.0, np.nan],
                [np.nan, 1.0, np.nan, 1.0, np.nan, 1.0],
                [1.0, np.nan, 2.0, np.nan, 2.0, np.nan],
                [np.nan, 1.0, np.nan, 2.0, np.nan, 3.0],
                [2.0, np.nan, 1.0, np.nan, 4.0, np.nan],
                [np.nan, 2.0, np.nan, 1.0, np.nan, 5.0],
                [2.0, np.nan, 2.0, np.nan, 6.0, np.nan],
                [np.nan, 2.0, np.nan, 2.0, np.nan, 7.0],
            ],
            MultiIndex.from_tuples(
                # 期望的列索引对象，包含了不同层级的元组索引
                [
                    ("lev1", 1),
                    ("lev1", 2),
                    ("lev2", 1),
                    ("lev2", 2),
                    ("values", 1),
                    ("values", 2),
                ],
                names=[None, "lev3"],  # 列索引的名称
            ),
            Index([1, 2, 3, 4, 5, 6, 7, 8], name="lev4"),  # 期望的行索引对象
        ),
        (
            ["lev1", "lev2"],  # 输入的索引为包含字符串 'lev1' 和 'lev2' 的列表
            "lev3",             # 输入的列为字符串 'lev3'
            "values",           # 输入的数值为字符串 'values'
            # 期望的数值结果，是一个包含具体数值的二维数组
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            Index([1, 2], name="lev3"),  # 期望的列索引对象
            MultiIndex.from_tuples(
                # 期望的行索引对象，包含了不同层级的元组索引
                [(1, 1), (1, 2), (2, 1), (2, 2)],
                names=["lev1", "lev2"],  # 行索引的名称
            ),
        ),
        (
            ["lev1", "lev2"],  # 输入的索引为包含字符串 'lev1' 和 'lev2' 的列表
            "lev3",             # 输入的列为字符串 'lev3'
            lib.no_default,     # 输入的数值为 lib.no_default，表示特殊值
            # 期望的数值结果，是一个包含具体数值的二维数组
            [[1, 2, 0, 1], [3, 4, 2, 3], [5, 6, 4, 5], [7, 8, 6, 7]],
            MultiIndex.from_tuples(
                # 期望的列索引对象，包含了不同层级的元组索引
                [("lev4", 1), ("lev4", 2), ("values", 1), ("values", 2)],
                names=[None, "lev3"],  # 列索引的名称
            ),
            MultiIndex.from_tuples(
                # 期望的行索引对象，包含了不同层级的元组索引
                [(1, 1), (1, 2), (2, 1), (2, 2)],
                names=["lev1", "lev2"],  # 行索引的名称
            ),
        ),
    ],
)
def test_pivot_list_like_index(
    input_index,
    input_columns,
    input_values,
    expected_values,
    expected_columns,
    expected_index,
):
    # GH 21425, test when index is given a list
    # 创建 DataFrame 对象，测试索引是一个列表的情况
    df = pd.DataFrame(
        {
            "lev1": [1, 1, 1, 1, 2, 2, 2, 2],    # 第一层级索引的数据
            "lev2": [1, 1, 2, 2, 1, 1, 2, 2],    # 第二层级索引的数据
            "lev3": [1, 2, 1, 2, 1, 2, 1, 2],    # 列索引的数据
            "lev4": [1, 2, 3, 4, 5, 6, 7, 8],    # 行索引的数据
            "values": [0, 1, 2, 3, 4, 5, 6, 7],  # 值的数据
        }
    )
    # 使用 pandas 的 pivot 函数，根据指定的索引和列重新组织数据框架，生成结果数据框架
    result = df.pivot(index=input_index, columns=input_columns, values=input_values)
    # 使用 pandas 的 DataFrame 函数创建一个期望的数据框架，指定列名和索引
    expected = pd.DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较结果数据框架和期望数据框架是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "input_index, input_columns, input_values, "
    "expected_values, expected_columns, expected_index",
    [  # 参数化测试的参数列表
        (
            "lev4",  # 第一个参数 input_index，用于设置行索引
            ["lev3"],  # 第二个参数 input_columns，用于设置列索引
            "values",  # 第三个参数 input_values，用于填充数据的列名
            [  # 期望的数据值列表
                [0.0, np.nan],  # 第一行数据
                [np.nan, 1.0],  # 第二行数据
                [2.0, np.nan],  # 第三行数据
                [np.nan, 3.0],  # 第四行数据
                [4.0, np.nan],  # 第五行数据
                [np.nan, 5.0],  # 第六行数据
                [6.0, np.nan],  # 第七行数据
                [np.nan, 7.0],  # 第八行数据
            ],
            Index([1, 2], name="lev3"),  # 期望的列索引对象
            Index([1, 2, 3, 4, 5, 6, 7, 8], name="lev4"),  # 期望的行索引对象
        ),
        (
            ["lev1", "lev2"],  # 第一个参数 input_index，用于设置行索引
            ["lev3"],  # 第二个参数 input_columns，用于设置列索引
            "values",  # 第三个参数 input_values，用于填充数据的列名
            [[0, 1], [2, 3], [4, 5], [6, 7]],  # 期望的数据值列表
            Index([1, 2], name="lev3"),  # 期望的列索引对象
            MultiIndex.from_tuples(  # 期望的行索引对象，创建一个多重索引
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
            ),
        ),
        (
            ["lev1"],  # 第一个参数 input_index，用于设置行索引
            ["lev2", "lev3"],  # 第二个参数 input_columns，用于设置列索引
            "values",  # 第三个参数 input_values，用于填充数据的列名
            [[0, 1, 2, 3], [4, 5, 6, 7]],  # 期望的数据值列表
            MultiIndex.from_tuples(  # 期望的行索引对象，创建一个多重索引
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev2", "lev3"]
            ),
            Index([1, 2], name="lev1"),  # 期望的列索引对象
        ),
        (
            ["lev1", "lev2"],  # 第一个参数 input_index，用于设置行索引
            ["lev3", "lev4"],  # 第二个参数 input_columns，用于设置列索引
            "values",  # 第三个参数 input_values，用于填充数据的列名
            [  # 期望的数据值列表
                [0.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # 第一行数据
                [np.nan, np.nan, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan],  # 第二行数据
                [np.nan, np.nan, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan],  # 第三行数据
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0],  # 第四行数据
            ],
            MultiIndex.from_tuples(  # 期望的行索引对象，创建一个多重索引
                [(1, 1), (2, 2), (1, 3), (2, 4), (1, 5), (2, 6), (1, 7), (2, 8)],
                names=["lev3", "lev4"],
            ),
            MultiIndex.from_tuples(  # 期望的列索引对象，创建一个多重索引
                [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
            ),
        ),
    ],
)
def test_pivot_list_like_columns(
    input_index,  # 输入的行索引
    input_columns,  # 输入的列索引
    input_values,  # 用于填充数据的列名
    expected_values,  # 期望的数据值列表
    expected_columns,  # 期望的列索引对象
    expected_index,  # 期望的行索引对象
):
    # GH 21425, test when columns is given a list
    df = pd.DataFrame(  # 创建测试用的 DataFrame 对象
        {
            "lev1": [1, 1, 1, 1, 2, 2, 2, 2],  # lev1 列数据
            "lev2": [1, 1, 2, 2, 1, 1, 2, 2],  # lev2 列数据
            "lev3": [1, 2, 1, 2, 1, 2, 1, 2],  # lev3 列数据
            "lev4": [1, 2, 3, 4, 5, 6, 7, 8],  # lev4 列数据
            "values": [0, 1, 2, 3, 4, 5, 6, 7],  # values 列数据
        }
    )

    result = df.pivot(index=input_index, columns=input_columns, values=input_values)  # 执行数据透视操作
    expected = pd.DataFrame(  # 创建期望结果的 DataFrame 对象
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)  # 断言结果与期望值相等


def test_pivot_multiindexed_rows_and_cols():
    # GH 36360，用于测试多重索引的行和列数据透视
    # 创建一个 Pandas 数据帧（DataFrame），包含指定数据和多级列与索引
    df = pd.DataFrame(
        data=np.arange(12).reshape(4, 3),  # 使用 reshape 生成 4x3 的数据数组
        columns=MultiIndex.from_tuples(
            [(0, 0), (0, 1), (0, 2)], names=["col_L0", "col_L1"]  # 创建多级列索引
        ),
        index=MultiIndex.from_tuples(
            [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 0, 0)],  # 创建多级行索引
            names=["idx_L0", "idx_L1", "idx_L2"],
        ),
    )
    
    # 对数据帧进行数据透视，按照指定的索引和列进行聚合操作，并计算特定列的和
    res = df.pivot_table(
        index=["idx_L0"],  # 设置透视表的行索引
        columns=["idx_L1"],  # 设置透视表的列索引
        values=[(0, 1)],  # 设置透视表的值，这里使用元组指定多级列
        aggfunc=lambda col: col.values.sum(),  # 聚合函数，计算每列的和
    )
    
    # 创建预期的 Pandas 数据帧，包含特定的数据、多级列和索引
    expected = pd.DataFrame(
        data=[[5, np.nan], [10, 7.0]],  # 设置数据数组，包含缺失值和浮点数
        columns=MultiIndex.from_tuples(
            [(0, 1, 0), (0, 1, 1)], names=["col_L0", "col_L1", "idx_L1"]  # 创建多级列索引
        ),
        index=Index([0, 1], dtype="int64", name="idx_L0"),  # 创建索引
    )
    expected = expected.astype("float64")  # 将数据类型转换为浮点型
    
    # 使用测试框架中的 assert_frame_equal 函数来比较实际结果和预期结果的数据帧是否相等
    tm.assert_frame_equal(res, expected)
# 定义一个测试函数，用于测试 DataFrame 多级索引和无索引的情况
def test_pivot_df_multiindex_index_none():
    # GH 23955: GitHub issue编号，用于追踪相关问题
    # 创建一个包含数据的 DataFrame，列分别为 index_1, index_2, label, value
    df = pd.DataFrame(
        [
            ["A", "A1", "label1", 1],
            ["A", "A2", "label2", 2],
            ["B", "A1", "label1", 3],
            ["B", "A2", "label2", 4],
        ],
        columns=["index_1", "index_2", "label", "value"],
    )
    # 将 index_1 和 index_2 列设为 DataFrame 的多级索引
    df = df.set_index(["index_1", "index_2"])

    # 对 DataFrame 进行透视，以 label 列为列名，value 列为数值
    result = df.pivot(columns="label", values="value")
    
    # 创建预期的 DataFrame，包含了透视后的数据
    expected = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 2.0], [3.0, np.nan], [np.nan, 4.0]],
        index=df.index,  # 使用原始 DataFrame 的索引作为行索引
        columns=pd.Index(["label1", "label2"], name="label"),  # 设置列索引的名称为 label
    )
    
    # 使用测试工具检查结果和预期是否相等
    tm.assert_frame_equal(result, expected)
```