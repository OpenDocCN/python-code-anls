# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_internals.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 导入 pandas 测试模块
import pandas._testing as tm
# 从 pandas 测试模块的复制视图工具中导入 get_array 函数
from pandas.tests.copy_view.util import get_array


# 定义测试函数 test_consolidate
def test_consolidate():
    # 创建一个未合并的 DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df["c"] = [4, 5, 6]

    # 获取 DataFrame 的视图子集
    subset = df[:]

    # 确保每个 subset 中的块引用了 df 中的块
    assert all(blk.refs.has_reference() for blk in subset._mgr.blocks)

    # 在原地合并两个 int64 类型的块
    subset._consolidate_inplace()

    # float64 类型的块仍然引用原始块，因为它仍然是一个视图
    assert subset._mgr.blocks[0].refs.has_reference()
    # 断言 df["b"].values 和 subset["b"].values 共享内存，避免缓存 df["b"]
    assert np.shares_memory(get_array(df, "b"), get_array(subset, "b"))

    # 新合并的 int64 类型块不再引用其他块
    assert not subset._mgr.blocks[1].refs.has_reference()

    # 父 DataFrame 现在只有 float 列还与其他块相关联
    assert not df._mgr.blocks[0].refs.has_reference()
    assert df._mgr.blocks[1].refs.has_reference()
    assert not df._mgr.blocks[2].refs.has_reference()

    # 修改 subset 不会修改父 DataFrame
    subset.iloc[0, 1] = 0.0
    assert not df._mgr.blocks[1].refs.has_reference()
    assert df.loc[0, "b"] == 0.1


# 使用 pytest 的参数化装饰器，定义多个测试参数
@pytest.mark.parametrize("dtype", [np.intp, np.int8])
@pytest.mark.parametrize(
    "locs, arr",
    [
        ([0], np.array([-1, -2, -3])),
        ([1], np.array([-1, -2, -3])),
        ([5], np.array([-1, -2, -3])),
        ([0, 1], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([0, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([0, 1, 2], np.array([[-1, -2, -3], [-4, -5, -6], [-4, -5, -6]]).T),
        ([1, 2], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
        ([1, 3], np.array([[-1, -2, -3], [-4, -5, -6]]).T),
    ],
)
def test_iset_splits_blocks_inplace(locs, arr, dtype):
    # 目前没有调用 iset，并且 inplace=True 情况下的 loc 多于 1（只有 inplace=False 才会发生）
    # 确保它能正常工作
    df = DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12],
            "e": [13, 14, 15],
            "f": ["a", "b", "c"],
        },
    )
    arr = arr.astype(dtype)
    df_orig = df.copy()
    df2 = df.copy(deep=False)  # 触发 CoW（如果启用），否则进行复制
    df2._mgr.iset(locs, arr, inplace=True)

    # 断言 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)
    # 对于每列，如果不在 locs 中，则确保 get_array(df, col) 和 get_array(df2, col) 共享内存
    for i, col in enumerate(df.columns):
        if i not in locs:
            assert np.shares_memory(get_array(df, col), get_array(df2, col))


# 定义测试函数 test_exponential_backoff
def test_exponential_backoff():
    # GH#55518
    df = DataFrame({"a": [1, 2, 3]})
    for i in range(490):
        df.copy(deep=False)

    # 确保 df._mgr.blocks[0].refs.referenced_blocks 的长度为 491
    assert len(df._mgr.blocks[0].refs.referenced_blocks) == 491

    df = DataFrame({"a": [1, 2, 3]})

    # 函数结束
    # 创建一个包含510个DataFrame副本的列表
    dfs = [df.copy(deep=False) for i in range(510)]

    # 复制DataFrame对象，但是没有存储结果
    for i in range(20):
        df.copy(deep=False)

    # 断言DataFrame第一个管理块的引用计数等于531
    assert len(df._mgr.blocks[0].refs.referenced_blocks) == 531
    # 断言DataFrame第一个管理块的清除计数等于1000
    assert df._mgr.blocks[0].refs.clear_counter == 1000

    # 500次复制DataFrame对象，但是没有存储结果
    for i in range(500):
        df.copy(deep=False)

    # 断言DataFrame第一个管理块的清除计数仍然等于1000，因为仍有超过500个对象存活
    assert df._mgr.blocks[0].refs.clear_counter == 1000

    # 将dfs列表缩减为300个元素
    dfs = dfs[:300]

    # 再次进行500次DataFrame对象的复制，但是没有存储结果
    for i in range(500):
        df.copy(deep=False)

    # 断言DataFrame第一个管理块的清除计数现在等于500，因为存活的对象少于500个
    assert df._mgr.blocks[0].refs.clear_counter == 500
```