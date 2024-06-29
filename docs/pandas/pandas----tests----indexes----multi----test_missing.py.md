# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_missing.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库及其子模块
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm

# 定义单元测试函数 test_fillna，用于测试 MultiIndex 的 fillna 方法
def test_fillna(idx):
    # GH 11343
    # 定义错误信息
    msg = "isna is not defined for MultiIndex"
    # 使用 pytest 检查是否会抛出 NotImplementedError，且错误信息匹配 msg
    with pytest.raises(NotImplementedError, match=msg):
        idx.fillna(idx[0])

# 定义单元测试函数 test_dropna，用于测试 MultiIndex 的 dropna 方法
def test_dropna():
    # GH 6194
    # 创建一个 MultiIndex 对象 idx，包含 NaN 值
    idx = MultiIndex.from_arrays(
        [
            [1, np.nan, 3, np.nan, 5],
            [1, 2, np.nan, np.nan, 5],
            ["a", "b", "c", np.nan, "e"],
        ]
    )

    # 期望的结果 exp，使用 dropna 方法删除 NaN 值后的 MultiIndex
    exp = MultiIndex.from_arrays([[1, 5], [1, 5], ["a", "e"]])
    tm.assert_index_equal(idx.dropna(), exp)
    tm.assert_index_equal(idx.dropna(how="any"), exp)

    # 期望的结果 exp，使用 dropna 方法指定 how="all" 参数后的 MultiIndex
    exp = MultiIndex.from_arrays(
        [[1, np.nan, 3, 5], [1, 2, np.nan, 5], ["a", "b", "c", "e"]]
    )
    tm.assert_index_equal(idx.dropna(how="all"), exp)

    # 测试异常情况，期望抛出 ValueError 异常，错误信息为 msg
    msg = "invalid how option: xxx"
    with pytest.raises(ValueError, match=msg):
        idx.dropna(how="xxx")

    # GH26408
    # 创建一个 MultiIndex 对象 idx，包含 NaN 值
    idx = MultiIndex(
        levels=[[np.nan, None, pd.NaT, "128", 2], [np.nan, None, pd.NaT, "128", 2]],
        codes=[[0, -1, 1, 2, 3, 4], [0, -1, 3, 3, 3, 4]],
    )
    # 期望的结果 expected，使用 dropna 方法删除 NaN 值后的 MultiIndex
    expected = MultiIndex.from_arrays([["128", 2], ["128", 2]])
    tm.assert_index_equal(idx.dropna(), expected)
    tm.assert_index_equal(idx.dropna(how="any"), expected)

    # 期望的结果 expected，使用 dropna 方法指定 how="all" 参数后的 MultiIndex
    expected = MultiIndex.from_arrays(
        [[np.nan, np.nan, "128", 2], ["128", "128", "128", 2]]
    )
    tm.assert_index_equal(idx.dropna(how="all"), expected)

# 定义单元测试函数 test_nulls，对 MultiIndex 的 isna 方法进行测试
def test_nulls(idx):
    # 这实际上是 isna 方法的烟雾测试，因为这些方法在其他地方已经充分测试过

    # 定义错误信息
    msg = "isna is not defined for MultiIndex"
    # 使用 pytest 检查是否会抛出 NotImplementedError，且错误信息匹配 msg
    with pytest.raises(NotImplementedError, match=msg):
        idx.isna()

# 标记为预期失败的单元测试函数 test_hasnans_isnans，测试 MultiIndex 的 hasnans 和 isnans 方法
@pytest.mark.xfail(reason="isna is not defined for MultiIndex")
def test_hasnans_isnans(idx):
    # GH 11343, 为 hasnans / isnans 添加测试

    # 复制 idx 到 index
    index = idx.copy()

    # 期望结果为所有索引都不包含 NaN
    expected = np.array([False] * len(index), dtype=bool)
    tm.assert_numpy_array_equal(index._isnan, expected)
    assert index.hasnans is False

    # 再次复制 idx 到 index，并将其值中的第二个元素设置为 NaN
    index = idx.copy()
    values = index.values
    values[1] = np.nan

    index = type(idx)(values)

    # 期望结果中有一个元素为 True
    expected = np.array([False] * len(index), dtype=bool)
    expected[1] = True
    tm.assert_numpy_array_equal(index._isnan, expected)
    assert index.hasnans is True

# 定义单元测试函数 test_nan_stays_float，测试 MultiIndex 中 NaN 值保持为浮点数类型
def test_nan_stays_float():
    # GH 7031

    # 创建两个 MultiIndex 对象 idx0 和 idx1
    idx0 = MultiIndex(levels=[["A", "B"], []], codes=[[1, 0], [-1, -1]], names=[0, 1])
    idx1 = MultiIndex(levels=[["C"], ["D"]], codes=[[0], [0]], names=[0, 1])
    
    # 使用 join 方法将 idx0 和 idx1 进行外连接，得到 idxm
    idxm = idx0.join(idx1, how="outer")
    
    # 断言 idx0 中第二级别索引的所有值都是 NaN
    assert pd.isna(idx0.get_level_values(1)).all()
    # 以下在 0.14.1 版本中失败
    # 断言 idxm 中除最后一个值外第二级别索引的所有值都是 NaN
    assert pd.isna(idxm.get_level_values(1)[:-1]).all()

    # 创建两个 DataFrame 对象 df0 和 df1，分别使用 idx0 和 idx1 作为索引
    df0 = pd.DataFrame([[1, 2]], index=idx0)
    df1 = pd.DataFrame([[3, 4]], index=idx1)
    
    # 使用 df0 减去 df1 得到 dfm
    dfm = df0 - df1
    # 断言：检查 DataFrame df0 的第二级索引的所有值是否都是缺失值（NaN）
    assert pd.isna(df0.index.get_level_values(1)).all()
    
    # 断言：在版本 0.14.1 中以下操作失败
    assert pd.isna(dfm.index.get_level_values(1)[:-1]).all()
# 定义一个测试函数，用于检查多级索引中是否包含缺失值
def test_tuples_have_na():
    # 创建一个多级索引对象，包括两个层级，分别是 [1, 0] 和 [0, 1, 2, 3]
    index = MultiIndex(
        levels=[[1, 0], [0, 1, 2, 3]],
        # 指定每个级别的编码，其中第一个级别有一个非法值 -1
        codes=[[1, 1, 1, 1, -1, 0, 0, 0], [0, 1, 2, 3, 0, 1, 2, 3]],
    )

    # 断言：检查索引中第 4 行第 0 列是否为 NaN（缺失值）
    assert pd.isna(index[4][0])
    # 断言：检查索引的值中第 4 行第 0 列是否为 NaN
    assert pd.isna(index.values[4][0])
```