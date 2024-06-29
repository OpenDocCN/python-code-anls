# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_compat.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库及其相关模块
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm


# 定义测试函数，用于测试索引对象的数值兼容性
def test_numeric_compat(idx):
    # 测试乘法操作是否会引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match="cannot perform __mul__"):
        idx * 1

    # 测试右乘法操作是否会引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match="cannot perform __rmul__"):
        1 * idx

    # 设置除法操作错误信息
    div_err = "cannot perform __truediv__"
    # 测试除法操作是否会引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match=div_err):
        idx / 1

    # 修改错误信息，以测试反向除法操作是否会引发 TypeError 异常，匹配指定的错误信息
    div_err = div_err.replace(" __", " __r")
    with pytest.raises(TypeError, match=div_err):
        1 / idx

    # 测试整除操作是否会引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match="cannot perform __floordiv__"):
        idx // 1

    # 测试反向整除操作是否会引发 TypeError 异常，匹配指定的错误信息
    with pytest.raises(TypeError, match="cannot perform __rfloordiv__"):
        1 // idx


# 使用参数化测试，测试逻辑方法的兼容性
@pytest.mark.parametrize("method", ["all", "any", "__invert__"])
def test_logical_compat(idx, method):
    # 构造错误信息，用于测试逻辑方法是否会引发 TypeError 异常，匹配指定的错误信息
    msg = f"cannot perform {method}"

    with pytest.raises(TypeError, match=msg):
        # 调用对象的指定逻辑方法，期望引发 TypeError 异常
        getattr(idx, method)()


# 测试索引对象的就地变异是否会重置值
def test_inplace_mutation_resets_values():
    # 定义多级索引对象及其设置
    levels = [["a", "b", "c"], [4]]
    levels2 = [[1, 2, 3], ["a"]]
    codes = [[0, 1, 0, 2, 2, 0], [0, 0, 0, 0, 0, 0]]

    mi1 = MultiIndex(levels=levels, codes=codes)
    mi2 = MultiIndex(levels=levels2, codes=codes)

    # 确保在实例化 MultiIndex 对象时不会访问或缓存 _values
    assert "_values" not in mi1._cache
    assert "_values" not in mi2._cache

    vals = mi1.values.copy()
    vals2 = mi2.values.copy()

    # 访问 .values 属性应该会缓存 ._values 属性
    assert mi1._values is mi1._cache["_values"]
    assert mi1.values is mi1._cache["_values"]
    assert isinstance(mi1._cache["_values"], np.ndarray)

    # 确保级别设置有效
    new_vals = mi1.set_levels(levels2).values
    tm.assert_almost_equal(vals2, new_vals)

    # 不应该从 _cache 中删除 _values [实现细节]
    tm.assert_almost_equal(mi1._cache["_values"], vals)

    # .values 属性仍然保持不变
    tm.assert_almost_equal(mi1.values, vals)

    # 确保标签设置有效
    codes2 = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    exp_values = np.empty((6,), dtype=object)
    exp_values[:] = [(1, "a")] * 6

    # 必须是元组的一维数组
    assert exp_values.shape == (6,)

    new_mi = mi2.set_codes(codes2)
    assert "_values" not in new_mi._cache
    new_values = new_mi.values
    assert "_values" in new_mi._cache

    # 不应该改变缓存
    tm.assert_almost_equal(mi2._cache["_values"], vals2)

    # 应该有正确的值
    tm.assert_almost_equal(exp_values, new_values)


# 测试箱式分类值
def test_boxable_categorical_values():
    # 创建一个 pandas 的分类对象
    cat = pd.Categorical(pd.date_range("2012-01-01", periods=3, freq="h"))
    # 使用 MultiIndex.from_product 创建多级索引对象，并获取其值
    result = MultiIndex.from_product([["a", "b", "c"], cat]).values
    # 创建一个预期结果，包含一个包含元组的 Pandas Series，每个元组包含字符串和时间戳
    expected = pd.Series(
        [
            ("a", pd.Timestamp("2012-01-01 00:00:00")),
            ("a", pd.Timestamp("2012-01-01 01:00:00")),
            ("a", pd.Timestamp("2012-01-01 02:00:00")),
            ("b", pd.Timestamp("2012-01-01 00:00:00")),
            ("b", pd.Timestamp("2012-01-01 01:00:00")),
            ("b", pd.Timestamp("2012-01-01 02:00:00")),
            ("c", pd.Timestamp("2012-01-01 00:00:00")),
            ("c", pd.Timestamp("2012-01-01 01:00:00")),
            ("c", pd.Timestamp("2012-01-01 02:00:00")),
        ]
    ).values
    # 使用 numpy.testing 模块的函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)
    
    # 创建一个 DataFrame，包含三列 'a', 'b', 'c'，其中 'a' 是字符串列，'b', 'c' 是 cat 和 np.array(cat) 的值
    result = pd.DataFrame({"a": ["a", "b", "c"], "b": cat, "c": np.array(cat)}).values
    # 创建预期的 DataFrame，与 result 结构相同，'a' 列是字符串，'b', 'c' 列是预定义的时间戳数组
    expected = pd.DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": [
                pd.Timestamp("2012-01-01 00:00:00"),
                pd.Timestamp("2012-01-01 01:00:00"),
                pd.Timestamp("2012-01-01 02:00:00"),
            ],
            "c": [
                pd.Timestamp("2012-01-01 00:00:00"),
                pd.Timestamp("2012-01-01 01:00:00"),
                pd.Timestamp("2012-01-01 02:00:00"),
            ],
        }
    ).values
    # 使用 numpy.testing 模块的函数比较 result 和 expected 数组是否相等
    tm.assert_numpy_array_equal(result, expected)
```