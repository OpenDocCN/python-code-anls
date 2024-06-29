# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_reindex.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理
from pandas import (  # 从 Pandas 中导入 Index 和 MultiIndex 类
    Index,
    MultiIndex,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块

def test_reindex(idx):
    # 测试函数，验证 reindex 方法的返回结果和索引器
    result, indexer = idx.reindex(list(idx[:4]))
    assert isinstance(result, MultiIndex)  # 断言结果是 MultiIndex 类型
    assert result.names == ["first", "second"]  # 断言结果的命名为 ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]  # 断言结果的级别名称为 ["first", "second"]

    result, indexer = idx.reindex(list(idx))
    assert isinstance(result, MultiIndex)  # 再次断言结果是 MultiIndex 类型
    assert indexer is None  # 断言索引器为 None
    assert result.names == ["first", "second"]  # 断言结果的命名为 ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]  # 断言结果的级别名称为 ["first", "second"]

def test_reindex_level(idx):
    index = Index(["one"])  # 创建一个 Index 对象

    target, indexer = idx.reindex(index, level="second")  # 使用 reindex 方法重新索引，指定级别为 "second"
    target2, indexer2 = index.reindex(idx, level="second")  # 使用 reindex 方法重新索引，指定级别为 "second"

    exp_index = idx.join(index, level="second", how="right")  # 预期的索引结果，使用 join 方法
    exp_index2 = idx.join(index, level="second", how="left")  # 另一个预期的索引结果，使用 join 方法

    assert target.equals(exp_index)  # 断言 target 和预期的索引结果相等
    exp_indexer = np.array([0, 2, 4])  # 预期的索引器数组
    tm.assert_numpy_array_equal(indexer, exp_indexer, check_dtype=False)  # 使用测试模块断言索引器数组相等

    assert target2.equals(exp_index2)  # 断言 target2 和预期的第二个索引结果相等
    exp_indexer2 = np.array([0, -1, 0, -1, 0, -1])  # 另一个预期的索引器数组
    tm.assert_numpy_array_equal(indexer2, exp_indexer2, check_dtype=False)  # 使用测试模块断言索引器数组相等

    with pytest.raises(TypeError, match="Fill method not supported"):
        idx.reindex(idx, method="pad", level="second")  # 使用 reindex 方法，期望引发 TypeError 异常，匹配特定错误信息

def test_reindex_preserves_names_when_target_is_list_or_ndarray(idx):
    # 测试函数，验证 reindex 方法在目标为列表或数组时是否保留名称
    idx = idx.copy()
    target = idx.copy()
    idx.names = target.names = [None, None]  # 将索引对象的名称设置为 None

    other_dtype = MultiIndex.from_product([[1, 2], [3, 4]])  # 创建另一个 MultiIndex 对象

    # 不同情况下的断言
    assert idx.reindex([])[0].names == [None, None]
    assert idx.reindex(np.array([]))[0].names == [None, None]
    assert idx.reindex(target.tolist())[0].names == [None, None]
    assert idx.reindex(target.values)[0].names == [None, None]
    assert idx.reindex(other_dtype.tolist())[0].names == [None, None]
    assert idx.reindex(other_dtype.values)[0].names == [None, None]

    idx.names = ["foo", "bar"]  # 修改索引对象的名称
    assert idx.reindex([])[0].names == ["foo", "bar"]
    assert idx.reindex(np.array([]))[0].names == ["foo", "bar"]
    assert idx.reindex(target.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(target.values)[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.values)[0].names == ["foo", "bar"]

def test_reindex_lvl_preserves_names_when_target_is_list_or_array():
    # 测试函数，验证 reindex 方法在级别重新索引时是否保留名称
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]], names=["foo", "bar"])
    assert idx.reindex([], level=0)[0].names == ["foo", "bar"]
    assert idx.reindex([], level=1)[0].names == ["foo", "bar"]

def test_reindex_lvl_preserves_type_if_target_is_empty_list_or_array(
    using_infer_string,
):
    # 测试函数，验证 reindex 方法在目标为空列表或数组时是否保留类型
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]])
    assert idx.reindex([], level=0)[0].levels[0].dtype.type == np.int64  # 断言第一个级别的类型为 np.int64
    exp = np.object_ if not using_infer_string else str  # 根据 using_infer_string 参数确定预期的类型为 np.object_ 或 str
    # 使用断言验证索引对象 idx 通过重新索引后的第一个级别的第一个水平的数据类型是否等于 exp
    assert idx.reindex([], level=1)[0].levels[1].dtype.type == exp

    # 创建包含字符串 "foo" 和 "bar" 的分类数据 cat
    cat = pd.Categorical(["foo", "bar"])
    # 创建从 "2016-01-01" 开始的两个时期，时区设定为 "US/Pacific" 的日期时间索引 dti
    dti = pd.date_range("2016-01-01", periods=2, tz="US/Pacific")
    # 使用产品方式从分类数据 cat 和日期时间索引 dti 创建多级索引 mi
    mi = MultiIndex.from_product([cat, dti])
    # 使用断言验证多级索引 mi 通过重新索引后的第一个级别的第一个水平的数据类型是否与分类数据 cat 的数据类型相等
    assert mi.reindex([], level=0)[0].levels[0].dtype == cat.dtype
    # 使用断言验证多级索引 mi 通过重新索引后的第二个级别的第一个水平的数据类型是否与日期时间索引 dti 的数据类型相等
    assert mi.reindex([], level=1)[0].levels[1].dtype == dti.dtype
def test_reindex_base(idx):
    # 创建一个预期的 numpy 数组，包含从 0 到 idx.size 的整数，数据类型为 np.intp
    expected = np.arange(idx.size, dtype=np.intp)

    # 调用 idx 对象的 get_indexer 方法，传入自身作为参数，返回一个实际的索引数组
    actual = idx.get_indexer(idx)
    # 使用测试工具函数，验证预期的 numpy 数组与实际数组是否相等
    tm.assert_numpy_array_equal(expected, actual)

    # 使用 pytest 的上下文管理器，期望抛出 ValueError 异常，并且异常信息匹配 "Invalid fill method"
    with pytest.raises(ValueError, match="Invalid fill method"):
        # 再次调用 idx 对象的 get_indexer 方法，传入自身和一个无效的方法名作为参数
        idx.get_indexer(idx, method="invalid")


def test_reindex_non_unique():
    # 创建一个包含非唯一值的 MultiIndex 对象 idx
    idx = MultiIndex.from_tuples([(0, 0), (1, 1), (1, 1), (2, 2)])
    # 创建一个 Series 对象 a，使用 idx 作为索引，数据为 0 到 3
    a = pd.Series(np.arange(4), index=idx)
    # 创建一个新的 MultiIndex 对象 new_idx，包含三个元组
    new_idx = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)])

    # 定义一个错误消息字符串
    msg = "cannot handle a non-unique multi-index!"
    # 使用 pytest 的上下文管理器，期望抛出 ValueError 异常，并且异常信息匹配预定义的消息
    with pytest.raises(ValueError, match=msg):
        # 调用 Series 对象 a 的 reindex 方法，传入 new_idx 作为参数
        a.reindex(new_idx)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # GH41170
    # 创建一个 MultiIndex 对象 idx，根据给定的 values 参数
    idx = MultiIndex.from_arrays(values)
    # 调用 idx 对象的 reindex 方法，传入一个包含单个元素 "b" 的 numpy 数组，并指定 level=0
    result, result_indexer = idx.reindex(np.array(["b"]), level=0)
    # 创建一个预期的 MultiIndex 对象 expected，包含一个 "b" 元素和 values[1] 元素
    expected = MultiIndex(levels=[["b"], values[1]], codes=[[], []])
    # 创建一个预期的索引器数组 expected_indexer，空数组，其数据类型与 result_indexer 相同
    expected_indexer = np.array([], dtype=result_indexer.dtype)
    # 使用测试工具函数，验证 reindex 方法返回的结果与预期结果是否相等
    tm.assert_index_equal(result, expected)
    # 使用测试工具函数，验证 result_indexer 数组与预期索引器数组是否相等
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_not_all_tuples():
    # 创建一个 keys 列表，包含多个元组和一个字符串
    keys = [("i", "i"), ("i", "j"), ("j", "i"), "j"]
    # 使用 MultiIndex 的 from_tuples 方法，创建一个 MultiIndex 对象 mi，从 keys 列表中选择前三个元组
    mi = MultiIndex.from_tuples(keys[:-1])
    # 创建一个 Index 对象 idx，使用 keys 作为索引
    idx = Index(keys)
    # 调用 mi 对象的 reindex 方法，传入 idx 作为参数
    res, indexer = mi.reindex(idx)

    # 使用测试工具函数，验证 reindex 方法返回的结果 res 与输入的 idx 索引对象是否相等
    tm.assert_index_equal(res, idx)
    # 创建一个预期的 numpy 数组 expected，包含 [0, 1, 2, -1] 四个整数，数据类型为 np.intp
    expected = np.array([0, 1, 2, -1], dtype=np.intp)
    # 使用测试工具函数，验证 indexer 数组与预期数组 expected 是否相等
    tm.assert_numpy_array_equal(indexer, expected)


def test_reindex_limit_arg_with_multiindex():
    # GH21247
    # 创建一个 MultiIndex 对象 idx，包含三个元组
    idx = MultiIndex.from_tuples([(3, "A"), (4, "A"), (4, "B")])

    # 创建一个 Series 对象 df，使用 idx 作为索引，数据为 [0.02, 0.01, 0.012]
    df = pd.Series([0.02, 0.01, 0.012], index=idx)

    # 创建一个新的 MultiIndex 对象 new_idx，包含多个元组
    new_idx = MultiIndex.from_tuples(
        [
            (3, "A"),
            (3, "B"),
            (4, "A"),
            (4, "B"),
            (4, "C"),
            (5, "B"),
            (5, "C"),
            (6, "B"),
            (6, "C"),
        ]
    )

    # 使用 pytest 的上下文管理器，期望抛出 ValueError 异常，异常信息匹配指定的消息
    with pytest.raises(
        ValueError,
        match="limit argument only valid if doing pad, backfill or nearest reindexing",
    ):
        # 调用 Series 对象 df 的 reindex 方法，传入 new_idx、fill_value=0 和 limit=1 作为参数
        df.reindex(new_idx, fill_value=0, limit=1)


def test_reindex_with_none_in_nested_multiindex():
    # GH42883
    # 创建一个嵌套的 MultiIndex 对象 index，包含两个元组
    index = MultiIndex.from_tuples([(("a", None), 1), (("b", None), 2)])
    # 创建另一个嵌套的 MultiIndex 对象 index2，包含相同的两个元组，但顺序相反
    index2 = MultiIndex.from_tuples([(("b", None), 2), (("a", None), 1)])
    # 创建一个 DataFrame 对象 df1_dtype，包含两行数据 [1, 2]，使用 index 作为索引
    df1_dtype = pd.DataFrame([1, 2], index=index)
    # 创建一个 DataFrame 对象 df2_dtype，包含两行数据 [2, 1]，使用 index2 作为索引
    df2_dtype = pd.DataFrame([2, 1], index=index2)

    # 调用 df1_dtype 对象的 reindex_like 方法，传入 df2_dtype 作为参数，返回一个结果 DataFrame 对象 result
    result = df1_dtype.reindex_like(df2_dtype)
    # 创建一个预期的 DataFrame 对象 expected，与 df2_dtype 相同
    expected = df2_dtype
    # 使用测试工具函数，验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```