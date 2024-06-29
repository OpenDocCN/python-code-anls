# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_conversion.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
from pandas import (  # 导入 Pandas 中的特定模块和类
    DataFrame,  # 数据框类，用于表示二维数据
    MultiIndex,  # 多级索引类，用于处理多级行或列索引
    RangeIndex,  # 范围索引类，用于表示指定范围的整数索引
)
import pandas._testing as tm  # 导入 Pandas 内部的测试工具模块

def test_to_numpy(idx):
    result = idx.to_numpy()  # 将传入的 idx 对象转换为 NumPy 数组
    exp = idx.values  # 获取 idx 对象的值作为期望结果
    tm.assert_numpy_array_equal(result, exp)  # 使用测试工具比较两个 NumPy 数组是否相等

def test_to_frame():
    tuples = [(1, "one"), (1, "two"), (2, "one"), (2, "two")]

    index = MultiIndex.from_tuples(tuples)  # 根据元组列表创建多级索引对象
    result = index.to_frame(index=False)  # 将索引转换为数据帧（DataFrame），不包含索引列
    expected = DataFrame(tuples)  # 创建期望的数据帧对象
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    result = index.to_frame()  # 将索引转换为数据帧（DataFrame），包含索引列
    expected.index = index  # 设置期望的数据帧索引为原始索引对象
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    tuples = [(1, "one"), (1, "two"), (2, "one"), (2, "two")]
    index = MultiIndex.from_tuples(tuples, names=["first", "second"])  # 创建带有命名级别的多级索引对象
    result = index.to_frame(index=False)  # 将索引转换为数据帧（DataFrame），不包含索引列
    expected = DataFrame(tuples)  # 创建期望的数据帧对象
    expected.columns = ["first", "second"]  # 设置期望的列名
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    result = index.to_frame()  # 将索引转换为数据帧（DataFrame），包含索引列
    expected.index = index  # 设置期望的数据帧索引为原始索引对象
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    # See GH-22580
    index = MultiIndex.from_tuples(tuples)
    result = index.to_frame(index=False, name=["first", "second"])  # 将索引转换为数据帧（DataFrame），不包含索引列，并指定列名
    expected = DataFrame(tuples)  # 创建期望的数据帧对象
    expected.columns = ["first", "second"]  # 设置期望的列名
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    result = index.to_frame(name=["first", "second"])  # 将索引转换为数据帧（DataFrame），包含索引列，并指定列名
    expected.index = index  # 设置期望的数据帧索引为原始索引对象
    expected.columns = ["first", "second"]  # 设置期望的列名
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    msg = "'name' must be a list / sequence of column names."
    with pytest.raises(TypeError, match=msg):  # 断言捕获到 TypeError 异常，并且异常消息匹配给定的模式
        index.to_frame(name="first")  # 尝试将索引转换为数据帧，但未提供列表或序列形式的列名

    msg = "'name' should have same length as number of levels on index."
    with pytest.raises(ValueError, match=msg):  # 断言捕获到 ValueError 异常，并且异常消息匹配给定的模式
        index.to_frame(name=["first"])  # 尝试将索引转换为数据帧，但提供的列名长度与索引级别数不匹配

    # Tests for datetime index
    index = MultiIndex.from_product([range(5), pd.date_range("20130101", periods=3)])  # 创建包含日期时间索引的多级索引对象
    result = index.to_frame(index=False)  # 将索引转换为数据帧（DataFrame），不包含索引列
    expected = DataFrame(  # 创建期望的数据帧对象
        {
            0: np.repeat(np.arange(5, dtype="int64"), 3),  # 第一列，重复数组值
            1: np.tile(pd.date_range("20130101", periods=3), 5),  # 第二列，平铺日期范围
        }
    )
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    result = index.to_frame()  # 将索引转换为数据帧（DataFrame），包含索引列
    expected.index = index  # 设置期望的数据帧索引为原始索引对象
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    # See GH-22580
    result = index.to_frame(index=False, name=["first", "second"])  # 将索引转换为数据帧（DataFrame），不包含索引列，并指定列名
    expected = DataFrame(  # 创建期望的数据帧对象
        {
            "first": np.repeat(np.arange(5, dtype="int64"), 3),  # 第一列，重复数组值
            "second": np.tile(pd.date_range("20130101", periods=3), 5),  # 第二列，平铺日期范围
        }
    )
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等

    result = index.to_frame(name=["first", "second"])  # 将索引转换为数据帧（DataFrame），包含索引列，并指定列名
    expected.index = index  # 设置期望的数据帧索引为原始索引对象
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据帧是否相等


def test_to_frame_dtype_fidelity():
    # GH 22420
    # 使用 MultiIndex 类的 from_arrays 方法创建一个多级索引对象 mi
    mi = MultiIndex.from_arrays(
        [
            pd.date_range("19910905", periods=6, tz="US/Eastern"),  # 创建包含6个日期的日期范围，带有时区信息
            [1, 1, 1, 2, 2, 2],  # 创建一个整数数组作为第二级索引
            pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),  # 创建有序的分类数据作为第三级索引
            ["x", "x", "y", "z", "x", "y"],  # 创建字符串数组作为第四级索引
        ],
        names=["dates", "a", "b", "c"],  # 设置每个级别的名称
    )
    
    # 创建一个字典，记录原始 MultiIndex 对象中每个级别的数据类型
    original_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}
    
    # 创建一个预期的 DataFrame 对象，与 mi 的结构和数据相匹配
    expected_df = DataFrame(
        {
            "dates": pd.date_range("19910905", periods=6, tz="US/Eastern"),  # 与 mi 中的日期范围相同的日期范围
            "a": [1, 1, 1, 2, 2, 2],  # 与 mi 中的第二级索引相同的整数数组
            "b": pd.Categorical(["a", "a", "b", "b", "c", "c"], ordered=True),  # 与 mi 中的第三级索引相同的有序分类数据
            "c": ["x", "x", "y", "z", "x", "y"],  # 与 mi 中的第四级索引相同的字符串数组
        }
    )
    
    # 将 MultiIndex 对象 mi 转换为普通的 DataFrame 对象 df，不包含索引
    df = mi.to_frame(index=False)
    
    # 获取 DataFrame df 中各列的数据类型，并转换为字典形式
    df_dtypes = df.dtypes.to_dict()
    
    # 使用 pytest 的 assert_frame_equal 方法比较 DataFrame df 和预期的 DataFrame expected_df 是否相等
    tm.assert_frame_equal(df, expected_df)
    
    # 断言原始 MultiIndex 对象中每个级别的数据类型是否与 DataFrame df 中对应列的数据类型相同
    assert original_dtypes == df_dtypes
# 测试函数，验证转换为DataFrame后的列顺序是否符合预期
def test_to_frame_resulting_column_order():
    # GH 22420
    # 预期的列顺序
    expected = ["z", 0, "a"]
    # 创建一个多级索引对象，从数组构建，指定列名为 expected
    mi = MultiIndex.from_arrays(
        [["a", "b", "c"], ["x", "y", "z"], ["q", "w", "e"]], names=expected
    )
    # 将多级索引转换为DataFrame后，获取其列并转换为列表
    result = mi.to_frame().columns.tolist()
    # 断言结果是否与预期一致
    assert result == expected


# 测试函数，验证处理重复标签时的行为
def test_to_frame_duplicate_labels():
    # GH 45245
    # 数据和重复的列名
    data = [(1, 2), (3, 4)]
    names = ["a", "a"]
    # 从元组数据和重复的列名创建多级索引对象
    index = MultiIndex.from_tuples(data, names=names)
    # 使用 pytest 断言检查转换为DataFrame时是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match="Cannot create duplicate column labels"):
        index.to_frame()

    # 测试允许重复列名的情况
    result = index.to_frame(allow_duplicates=True)
    expected = DataFrame(data, index=index, columns=names)
    # 使用测试模块中的方法比较结果和预期DataFrame是否相等
    tm.assert_frame_equal(result, expected)

    # 另一种情况，列名包含 None 和整数0
    names = [None, 0]
    index = MultiIndex.from_tuples(data, names=names)
    # 同样使用 pytest 断言检查转换为DataFrame时是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match="Cannot create duplicate column labels"):
        index.to_frame()

    # 再次测试允许重复列名的情况
    result = index.to_frame(allow_duplicates=True)
    expected = DataFrame(data, index=index, columns=[0, 0])
    # 使用测试模块中的方法比较结果和预期DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证转换为DataFrame后的列是否为 RangeIndex
def test_to_frame_column_rangeindex():
    # 从数组创建多级索引对象
    mi = MultiIndex.from_arrays([[1, 2], ["a", "b"]])
    # 将多级索引转换为DataFrame后，获取其列索引
    result = mi.to_frame().columns
    # 预期的结果应该是一个 RangeIndex
    expected = RangeIndex(2)
    # 使用测试模块中的方法比较结果和预期列索引是否相等
    tm.assert_index_equal(result, expected, exact=True)


# 测试函数，验证转换为扁平索引的行为
def test_to_flat_index(idx):
    # 预期的扁平化索引
    expected = pd.Index(
        (
            ("foo", "one"),
            ("foo", "two"),
            ("bar", "one"),
            ("baz", "two"),
            ("qux", "one"),
            ("qux", "two"),
        ),
        tupleize_cols=False,
    )
    # 执行转换为扁平索引
    result = idx.to_flat_index()
    # 使用测试模块中的方法比较结果和预期扁平索引是否相等
    tm.assert_index_equal(result, expected)
```