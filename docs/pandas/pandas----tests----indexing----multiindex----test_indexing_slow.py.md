# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_indexing_slow.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库，用于数据处理
from pandas import (  # 从 Pandas 库中导入 DataFrame 和 Series 类
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

# 定义测试 fixture，返回值为整数 5
@pytest.fixture
def m():
    return 5

# 定义测试 fixture，返回值为整数 100
@pytest.fixture
def n():
    return 100

# 定义测试 fixture，返回值为包含字符串的列表
@pytest.fixture
def cols():
    return ["jim", "joe", "jolie", "joline", "jolia"]

# 定义测试 fixture，使用参数 n，返回一个元组列表
@pytest.fixture
def vals(n):
    vals = [
        np.random.default_rng(2).integers(0, 10, n),  # 生成长度为 n 的随机整数数组
        np.random.default_rng(2).choice(list("abcdefghij"), n),  # 从字符列表中随机选择 n 个字符
        np.random.default_rng(2).choice(
            pd.date_range("20141009", periods=10).tolist(), n  # 从日期范围中随机选择 n 个日期
        ),
        np.random.default_rng(2).choice(list("ZYXWVUTSRQ"), n),  # 从字符列表中随机选择 n 个字符
        np.random.default_rng(2).standard_normal(n),  # 生成长度为 n 的标准正态分布随机数数组
    ]
    vals = list(map(tuple, zip(*vals)))  # 转置列表中的元组，以便于 DataFrame 的创建
    return vals

# 定义测试 fixture，使用参数 n 和 m，返回一个键的元组列表
@pytest.fixture
def keys(n, m, vals):
    # bunch of keys for testing
    keys = [
        np.random.default_rng(2).integers(0, 11, m),  # 生成长度为 m 的随机整数数组
        np.random.default_rng(2).choice(list("abcdefghijk"), m),  # 从字符列表中随机选择 m 个字符
        np.random.default_rng(2).choice(
            pd.date_range("20141009", periods=11).tolist(), m  # 从日期范围中随机选择 m 个日期
        ),
        np.random.default_rng(2).choice(list("ZYXWVUTSRQP"), m),  # 从字符列表中随机选择 m 个字符
    ]
    keys = list(map(tuple, zip(*keys)))  # 转置列表中的元组，以便于 DataFrame 的创建
    keys += [t[:-1] for t in vals[:: n // m]]  # 添加一些额外的键，以增加多样性
    return keys

# 定义测试 fixture，使用 vals 和 cols，返回一个 DataFrame 对象
@pytest.fixture
def df(vals, cols):
    return DataFrame(vals, columns=cols)

# 定义测试 fixture，使用 df，返回一个与 df 连接后的 DataFrame 对象
@pytest.fixture
def a(df):
    return pd.concat([df, df])

# 定义测试 fixture，使用 df 和 cols，返回一个去重后的 DataFrame 对象
@pytest.fixture
def b(df, cols):
    return df.drop_duplicates(subset=cols[:-1])

# 标记测试函数，忽略特定警告
@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
# 参数化测试函数，测试 lexsort_depth 参数从 0 到 4 的情况
@pytest.mark.parametrize("lexsort_depth", list(range(5)))
# 参数化测试函数，测试 frame_fixture 参数为 "a" 和 "b" 的情况
@pytest.mark.parametrize("frame_fixture", ["a", "b"])
def test_multiindex_get_loc(request, lexsort_depth, keys, frame_fixture, cols):
    # GH7724, GH2646
    # 根据 GitHub issue 号码标识测试相关的问题

    # 获取指定的 fixture 值作为测试数据框架
    frame = request.getfixturevalue(frame_fixture)
    if lexsort_depth == 0:
        df = frame.copy(deep=False)  # 如果 lexsort_depth 为 0，浅复制数据框架
    else:
        df = frame.sort_values(by=cols[:lexsort_depth])  # 否则，按前 lexsort_depth 列排序数据框架

    mi = df.set_index(cols[:-1])  # 将数据框架设置为多级索引，排除最后一列
    assert not mi.index._lexsort_depth < lexsort_depth  # 断言多级索引的 lexsort 深度不小于 lexsort_depth
    # 对给定的每个键进行迭代处理
    for key in keys:
        # 创建一个布尔掩码，长度与数据框 df 的行数相同，初始值为 True
        mask = np.ones(len(df), dtype=bool)

        # 对当前键 key 的所有部分进行测试
        for i, k in enumerate(key):
            # 更新掩码，以过滤出与当前部分键 k 匹配的行
            mask &= df.iloc[:, i] == k

            # 如果没有匹配的行，则断言当前部分键不在索引 mi 中
            if not mask.any():
                assert key[: i + 1] not in mi.index
                continue

            # 否则，断言当前部分键在索引 mi 中
            assert key[: i + 1] in mi.index

            # 从数据框中复制匹配的子集数据
            right = df[mask].copy(deep=False)

            # 如果当前处理的是部分键（非最后一部分）
            if i + 1 != len(key):
                # 删除部分键对应的列，并在原地进行修改
                return_value = right.drop(cols[: i + 1], axis=1, inplace=True)
                assert return_value is None
                # 将剩余部分键设为索引，并在原地进行修改
                return_value = right.set_index(cols[i + 1 : -1], inplace=True)
                assert return_value is None
                # 断言处理后的子集数据与索引 mi 中对应的部分键数据一致
                tm.assert_frame_equal(mi.loc[key[: i + 1]], right)

            else:  # 如果当前处理的是完整键（最后一部分）
                # 将剩余部分键设为索引，并在原地进行修改
                return_value = right.set_index(cols[:-1], inplace=True)
                assert return_value is None

                # 如果匹配到的行数为 1，则为单个命中
                if len(right) == 1:
                    # 创建 Series 对象，以匹配索引 mi 中对应部分键的期望结果
                    right = Series(
                        right["jolia"].values, name=right.index[0], index=["jolia"]
                    )
                    # 断言单个命中的结果与索引 mi 中对应部分键的期望结果一致
                    tm.assert_series_equal(mi.loc[key[: i + 1]], right)

                else:  # 如果匹配到的行数大于 1，则为多个命中
                    # 断言多个命中的子集数据与索引 mi 中对应部分键的期望结果一致
                    tm.assert_frame_equal(mi.loc[key[: i + 1]], right)
```