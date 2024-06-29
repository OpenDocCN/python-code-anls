# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_rank.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime

# 导入 numpy 库并使用 np 别名
import numpy as np

# 导入 pytest 库进行单元测试
import pytest

# 导入 pandas 库，并从中导入 DataFrame、NaT（Not a Time）、Series、concat 等类和函数
import pandas as pd
from pandas import (
    DataFrame,
    NaT,
    Series,
    concat,
)

# 导入 pandas 内部的测试模块
import pandas._testing as tm


# 定义测试函数 test_rank_unordered_categorical_typeerror
def test_rank_unordered_categorical_typeerror():
    # 创建一个无序的空分类变量 cat
    cat = pd.Categorical([], ordered=False)
    # 使用该分类变量创建一个 Series 对象 ser
    ser = Series(cat)
    # 将 Series 对象转换为 DataFrame 对象 df
    df = ser.to_frame()

    # 错误消息字符串
    msg = "Cannot perform rank with non-ordered Categorical"

    # 对 ser 进行分组，按照分类变量 cat，观察方式为非显式
    gb = ser.groupby(cat, observed=False)
    # 使用 pytest 检测是否抛出 TypeError 异常，且异常信息匹配指定的 msg 字符串
    with pytest.raises(TypeError, match=msg):
        gb.rank()

    # 对 df 进行分组，按照分类变量 cat，观察方式为非显式
    gb2 = df.groupby(cat, observed=False)
    # 使用 pytest 检测是否抛出 TypeError 异常，且异常信息匹配指定的 msg 字符串
    with pytest.raises(TypeError, match=msg):
        gb2.rank()


# 定义测试函数 test_rank_apply
def test_rank_apply():
    # 创建长度为 100 的对象数组 lev1，元素为字符串 "a" 重复 10 次
    lev1 = np.array(["a" * 10] * 100, dtype=object)
    # 创建长度为 130 的对象数组 lev2，元素为字符串 "b" 重复 10 次
    lev2 = np.array(["b" * 10] * 130, dtype=object)
    # 创建长度为 500 的整数数组 lab1，范围在 0 到 100 之间，随机整数
    lab1 = np.random.default_rng(2).integers(0, 100, size=500, dtype=int)
    # 创建长度为 500 的整数数组 lab2，范围在 0 到 130 之间，随机整数
    lab2 = np.random.default_rng(2).integers(0, 130, size=500, dtype=int)

    # 创建 DataFrame 对象 df，包含三列：'value' 列为标准正态分布的随机数列，'key1' 列为 lev1 对应 lab1 的取值，'key2' 列为 lev2 对应 lab2 的取值
    df = DataFrame(
        {
            "value": np.random.default_rng(2).standard_normal(500),
            "key1": lev1.take(lab1),
            "key2": lev2.take(lab2),
        }
    )

    # 对 df 按照 'key1' 和 'key2' 列进行分组，并对 'value' 列进行排名
    result = df.groupby(["key1", "key2"]).value.rank()

    # 期望的排名结果，为每个分组的子 DataFrame 对象 piece 的 'value' 列排名
    expected = [piece.value.rank() for key, piece in df.groupby(["key1", "key2"])]
    # 将所有排名结果连接成一个 Series 对象
    expected = concat(expected, axis=0)
    # 按照 result 的索引重新排序期望的结果
    expected = expected.reindex(result.index)
    # 使用 pandas 内部的测试模块 tm 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 对 df 按照 'key1' 和 'key2' 列进行分组，并对 'value' 列进行百分位排名
    result = df.groupby(["key1", "key2"]).value.rank(pct=True)

    # 期望的百分位排名结果，为每个分组的子 DataFrame 对象 piece 的 'value' 列百分位排名
    expected = [
        piece.value.rank(pct=True) for key, piece in df.groupby(["key1", "key2"])
    ]
    # 将所有百分位排名结果连接成一个 Series 对象
    expected = concat(expected, axis=0)
    # 按照 result 的索引重新排序期望的结果
    expected = expected.reindex(result.index)
    # 使用 pandas 内部的测试模块 tm 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，参数化 grps 列表为 ["qux"] 和 ["qux", "quux"]
@pytest.mark.parametrize("grps", [["qux"], ["qux", "quux"]])
# 使用 pytest 的 parametrize 装饰器，参数化 vals 列表为不同类型的整数和浮点数数组
@pytest.mark.parametrize(
    "vals",
    [
        np.array([2, 2, 8, 2, 6], dtype=dtype)
        for dtype in ["i8", "i4", "i2", "i1", "u8", "u4", "u2", "u1", "f8", "f4", "f2"]
    ]
    # 创建包含多个列表的列表，每个列表包含多个时间戳对象
    [
        # 第一个列表包含未指定时区的时间戳对象
        [
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-08"),
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-06"),
        ],
        # 第二个列表包含指定了时区为'US/Pacific'的时间戳对象
        [
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-08", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-06", tz="US/Pacific"),
        ],
        # 第三个列表包含通过减去零时间戳对象得到的时间差
        [
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-08") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-06") - pd.Timestamp(0),
        ],
        # 第四个列表包含转换为日周期的时间戳对象
        [
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-08").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-06").to_period("D"),
        ],
    ],
    # 使用 lambda 函数为每个子列表返回其第一个元素的类型作为 id
    ids=lambda x: type(x[0]),
# 导入pytest模块，用于编写和运行测试用例
import pytest

# 参数化测试用例，测试不同的排名参数组合
@pytest.mark.parametrize(
    "ties_method,ascending,pct,exp",
    [
        ("average", True, False, [2.0, 2.0, 5.0, 2.0, 4.0]),
        ("average", True, True, [0.4, 0.4, 1.0, 0.4, 0.8]),
        ("average", False, False, [4.0, 4.0, 1.0, 4.0, 2.0]),
        ("average", False, True, [0.8, 0.8, 0.2, 0.8, 0.4]),
        ("min", True, False, [1.0, 1.0, 5.0, 1.0, 4.0]),
        ("min", True, True, [0.2, 0.2, 1.0, 0.2, 0.8]),
        ("min", False, False, [3.0, 3.0, 1.0, 3.0, 2.0]),
        ("min", False, True, [0.6, 0.6, 0.2, 0.6, 0.4]),
        ("max", True, False, [3.0, 3.0, 5.0, 3.0, 4.0]),
        ("max", True, True, [0.6, 0.6, 1.0, 0.6, 0.8]),
        ("max", False, False, [5.0, 5.0, 1.0, 5.0, 2.0]),
        ("max", False, True, [1.0, 1.0, 0.2, 1.0, 0.4]),
        ("first", True, False, [1.0, 2.0, 5.0, 3.0, 4.0]),
        ("first", True, True, [0.2, 0.4, 1.0, 0.6, 0.8]),
        ("first", False, False, [3.0, 4.0, 1.0, 5.0, 2.0]),
        ("first", False, True, [0.6, 0.8, 0.2, 1.0, 0.4]),
        ("dense", True, False, [1.0, 1.0, 3.0, 1.0, 2.0]),
        ("dense", True, True, [1.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]),
        ("dense", False, False, [3.0, 3.0, 1.0, 3.0, 2.0]),
        ("dense", False, True, [3.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 2.0 / 3.0]),
    ],
)
# 定义测试函数，测试不同参数组合下的排名功能
def test_rank_args(grps, vals, ties_method, ascending, pct, exp):
    # 根据grps重复vals，以构建DataFrame的key列
    key = np.repeat(grps, len(vals))

    # 保存原始的vals值
    orig_vals = vals
    # 如果vals是numpy数组，则转换为列表，并重复以匹配key的长度
    vals = list(vals) * len(grps)
    if isinstance(orig_vals, np.ndarray):
        vals = np.array(vals, dtype=orig_vals.dtype)

    # 创建DataFrame对象，包含key和vals列
    df = DataFrame({"key": key, "val": vals})
    
    # 对DataFrame按key分组，并使用指定的排名方法、升序/降序和百分比参数进行排名
    result = df.groupby("key").rank(method=ties_method, ascending=ascending, pct=pct)

    # 创建期望的结果DataFrame，根据exp重复grps的长度，并命名列为"val"
    exp_df = DataFrame(exp * len(grps), columns=["val"])
    
    # 断言实际结果与期望结果是否相等
    tm.assert_frame_equal(result, exp_df)


# 参数化测试用例，测试不同的grps参数组合
@pytest.mark.parametrize("grps", [["qux"], ["qux", "quux"]])
# 参数化测试用例，测试不同的vals参数组合
@pytest.mark.parametrize(
    "vals", [[-np.inf, -np.inf, np.nan, 1.0, np.nan, np.inf, np.inf]]
)
# 参数化测试用例，测试不同的ties_method、ascending、na_option和exp参数组合
@pytest.mark.parametrize(
    "ties_method,ascending,na_option,exp",
    [
        # 示例数据集，每个元组包含以下信息：
        #   1. 聚合函数类型：average, min, max, first, dense
        #   2. 是否忽略 NaN 值：True 或 False
        #   3. 统计方式：keep（保留），top（取最大），bottom（取最小）
        #   4. 数据列表：包含数值和可能的 NaN 值
        ("average", True, "keep", [1.5, 1.5, np.nan, 3, np.nan, 4.5, 4.5]),
        ("average", True, "top", [3.5, 3.5, 1.5, 5.0, 1.5, 6.5, 6.5]),
        ("average", True, "bottom", [1.5, 1.5, 6.5, 3.0, 6.5, 4.5, 4.5]),
        ("average", False, "keep", [4.5, 4.5, np.nan, 3, np.nan, 1.5, 1.5]),
        ("average", False, "top", [6.5, 6.5, 1.5, 5.0, 1.5, 3.5, 3.5]),
        ("average", False, "bottom", [4.5, 4.5, 6.5, 3.0, 6.5, 1.5, 1.5]),
        ("min", True, "keep", [1.0, 1.0, np.nan, 3.0, np.nan, 4.0, 4.0]),
        ("min", True, "top", [3.0, 3.0, 1.0, 5.0, 1.0, 6.0, 6.0]),
        ("min", True, "bottom", [1.0, 1.0, 6.0, 3.0, 6.0, 4.0, 4.0]),
        ("min", False, "keep", [4.0, 4.0, np.nan, 3.0, np.nan, 1.0, 1.0]),
        ("min", False, "top", [6.0, 6.0, 1.0, 5.0, 1.0, 3.0, 3.0]),
        ("min", False, "bottom", [4.0, 4.0, 6.0, 3.0, 6.0, 1.0, 1.0]),
        ("max", True, "keep", [2.0, 2.0, np.nan, 3.0, np.nan, 5.0, 5.0]),
        ("max", True, "top", [4.0, 4.0, 2.0, 5.0, 2.0, 7.0, 7.0]),
        ("max", True, "bottom", [2.0, 2.0, 7.0, 3.0, 7.0, 5.0, 5.0]),
        ("max", False, "keep", [5.0, 5.0, np.nan, 3.0, np.nan, 2.0, 2.0]),
        ("max", False, "top", [7.0, 7.0, 2.0, 5.0, 2.0, 4.0, 4.0]),
        ("max", False, "bottom", [5.0, 5.0, 7.0, 3.0, 7.0, 2.0, 2.0]),
        ("first", True, "keep", [1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0]),
        ("first", True, "top", [3.0, 4.0, 1.0, 5.0, 2.0, 6.0, 7.0]),
        ("first", True, "bottom", [1.0, 2.0, 6.0, 3.0, 7.0, 4.0, 5.0]),
        ("first", False, "keep", [4.0, 5.0, np.nan, 3.0, np.nan, 1.0, 2.0]),
        ("first", False, "top", [6.0, 7.0, 1.0, 5.0, 2.0, 3.0, 4.0]),
        ("first", False, "bottom", [4.0, 5.0, 6.0, 3.0, 7.0, 1.0, 2.0]),
        ("dense", True, "keep", [1.0, 1.0, np.nan, 2.0, np.nan, 3.0, 3.0]),
        ("dense", True, "top", [2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 4.0]),
        ("dense", True, "bottom", [1.0, 1.0, 4.0, 2.0, 4.0, 3.0, 3.0]),
        ("dense", False, "keep", [3.0, 3.0, np.nan, 2.0, np.nan, 1.0, 1.0]),
        ("dense", False, "top", [4.0, 4.0, 1.0, 3.0, 1.0, 2.0, 2.0]),
        ("dense", False, "bottom", [3.0, 3.0, 4.0, 2.0, 4.0, 1.0, 1.0]),
    ],
# 定义一个测试函数，用于测试处理无穷大和 NaN 值的排名计算
def test_infs_n_nans(grps, vals, ties_method, ascending, na_option, exp):
    # 将组重复以匹配值的长度
    key = np.repeat(grps, len(vals))
    # 根据组长度调整值的长度
    vals = vals * len(grps)
    # 创建一个包含键值对的数据帧，键为 'key'，值为 'val'
    df = DataFrame({"key": key, "val": vals})
    # 使用分组后的 'key' 列对 'val' 列进行排名，指定排名方法、升序还是降序、处理 NaN 的选项
    result = df.groupby("key").rank(
        method=ties_method, ascending=ascending, na_option=na_option
    )
    # 创建一个预期结果的数据帧，用于与计算结果进行比较
    exp_df = DataFrame(exp * len(grps), columns=["val"])
    # 使用测试工具比较计算结果与预期结果的一致性
    tm.assert_frame_equal(result, exp_df)


# 使用参数化测试，对不同的输入组合进行排名参数的缺失测试
@pytest.mark.parametrize("grps", [["qux"], ["qux", "quux"]])
@pytest.mark.parametrize(
    "vals",
    [
        np.array([2, 2, np.nan, 8, 2, 6, np.nan, np.nan], dtype=dtype)
        for dtype in ["f8", "f4", "f2"]
    ]
    + [
        [
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-02"),
            np.nan,
            pd.Timestamp("2018-01-08"),
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-06"),
            np.nan,
            np.nan,
        ],
        [
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            np.nan,
            pd.Timestamp("2018-01-08", tz="US/Pacific"),
            pd.Timestamp("2018-01-02", tz="US/Pacific"),
            pd.Timestamp("2018-01-06", tz="US/Pacific"),
            np.nan,
            np.nan,
        ],
        [
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            np.nan,
            pd.Timestamp("2018-01-08") - pd.Timestamp(0),
            pd.Timestamp("2018-01-02") - pd.Timestamp(0),
            pd.Timestamp("2018-01-06") - pd.Timestamp(0),
            np.nan,
            np.nan,
        ],
        [
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            np.nan,
            pd.Timestamp("2018-01-08").to_period("D"),
            pd.Timestamp("2018-01-02").to_period("D"),
            pd.Timestamp("2018-01-06").to_period("D"),
            np.nan,
            np.nan,
        ],
    ],
    ids=lambda x: type(x[0]),
)
@pytest.mark.parametrize(
    "ties_method,ascending,na_option,pct,exp",
    ],
)
def test_rank_args_missing(grps, vals, ties_method, ascending, na_option, pct, exp):
    # 将组重复以匹配值的长度
    key = np.repeat(grps, len(vals))
    # 保留原始值以备份
    orig_vals = vals
    # 将值列表扩展以匹配组的长度
    vals = list(vals) * len(grps)
    # 如果原始值是 NumPy 数组，则将值列表转换为相同类型的 NumPy 数组
    if isinstance(orig_vals, np.ndarray):
        vals = np.array(vals, dtype=orig_vals.dtype)
    # 创建一个包含键值对的数据帧，键为 'key'，值为 'val'
    df = DataFrame({"key": key, "val": vals})
    # 使用分组后的 'key' 列对 'val' 列进行排名，指定排名方法、升序还是降序、处理 NaN 的选项以及百分比参数
    result = df.groupby("key").rank(
        method=ties_method, ascending=ascending, na_option=na_option, pct=pct
    )
    # 创建一个预期结果的数据帧，用于与计算结果进行比较
    exp_df = DataFrame(exp * len(grps), columns=["val"])
    # 使用测试工具比较计算结果与预期结果的一致性
    tm.assert_frame_equal(result, exp_df)


# 使用参数化测试，对百分比参数进行不同设置的排名重置测试
@pytest.mark.parametrize(
    "pct,exp", [(False, [3.0, 3.0, 3.0, 3.0, 3.0]), (True, [0.6, 0.6, 0.6, 0.6, 0.6])]
)
def test_rank_resets_each_group(pct, exp):
    # 创建一个包含 'key' 和 'val' 列的数据帧， 'key' 列包含两个不同的组， 'val' 列为每个组包含的值
    df = DataFrame(
        {"key": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"], "val": [1] * 10}
    )
    # 使用分组后的 'key' 列对 'val' 列进行排名，指定是否使用百分比参数
    result = df.groupby("key").rank(pct=pct)
    # 创建一个新的 DataFrame 对象 `exp_df`，其中包含原始 DataFrame `exp` 中所有数值乘以2后的结果，并且只有一列命名为 "val"。
    exp_df = DataFrame(exp * 2, columns=["val"])
    
    # 使用 `tm.assert_frame_equal()` 函数来比较两个 DataFrame 对象 `result` 和 `exp_df` 是否相等。
    tm.assert_frame_equal(result, exp_df)
@pytest.mark.parametrize(
    "dtype", ["int64", "int32", "uint64", "uint32", "float64", "float32"]
)
@pytest.mark.parametrize("upper", [True, False])
def test_rank_avg_even_vals(dtype, upper):
    if upper:
        # 如果 upper 为 True，则转换数据类型的首字母为大写
        dtype = dtype[0].upper() + dtype[1:]
        # 替换数据类型字符串中的 "Ui" 为 "UI"
        dtype = dtype.replace("Ui", "UI")
    # 创建包含两列数据的 DataFrame，其中一列为 "key"，内容为 4 个 "a"；另一列为 "val"，内容为 4 个 1
    df = DataFrame({"key": ["a"] * 4, "val": [1] * 4})
    # 将 "val" 列的数据类型转换为指定的 dtype
    df["val"] = df["val"].astype(dtype)
    # 断言 "val" 列的数据类型与指定的 dtype 相同
    assert df["val"].dtype == dtype

    # 对 DataFrame 按 "key" 列进行分组，并计算排名
    result = df.groupby("key").rank()
    # 创建期望的 DataFrame，包含一列 "val"，值为 2.5
    exp_df = DataFrame([2.5, 2.5, 2.5, 2.5], columns=["val"])
    if upper:
        # 如果 upper 为 True，则将期望的 DataFrame 的数据类型转换为 "Float64"
        exp_df = exp_df.astype("Float64")
    # 使用 pytest 的 assert 函数，比较 result 和 exp_df 是否相等
    tm.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize(
    "vals", [["bar", "bar", "foo", "bar", "baz"], ["bar", np.nan, "foo", np.nan, "baz"]]
)
def test_rank_object_dtype(rank_method, ascending, na_option, pct, vals):
    # 创建包含两列数据的 DataFrame，其中一列为 "key"，内容为 5 个 "foo"；另一列为 "val"，内容为 vals 参数指定的值
    df = DataFrame({"key": ["foo"] * 5, "val": vals})
    # 创建一个布尔掩码，用于标识 "val" 列中的缺失值
    mask = df["val"].isna()

    # 对 DataFrame 按 "key" 列进行分组
    gb = df.groupby("key")
    # 使用指定的参数对分组后的数据进行排名
    res = gb.rank(method=rank_method, ascending=ascending, na_option=na_option, pct=pct)

    # 根据 "mask" 的情况，构建预期的 DataFrame
    if mask.any():
        df2 = DataFrame({"key": ["foo"] * 5, "val": [0, np.nan, 2, np.nan, 1]})
    else:
        df2 = DataFrame({"key": ["foo"] * 5, "val": [0, 0, 2, 0, 1]})

    # 再次对预期的 DataFrame 按 "key" 列进行分组，并使用相同的参数对数据进行排名
    gb2 = df2.groupby("key")
    alt = gb2.rank(
        method=rank_method, ascending=ascending, na_option=na_option, pct=pct
    )

    # 使用 pytest 的 assert 函数，比较 res 和 alt 是否相等
    tm.assert_frame_equal(res, alt)


@pytest.mark.parametrize("na_option", [True, "bad", 1])
@pytest.mark.parametrize("pct", [True, False])
@pytest.mark.parametrize(
    "vals",
    [
        ["bar", "bar", "foo", "bar", "baz"],
        ["bar", np.nan, "foo", np.nan, "baz"],
        [1, np.nan, 2, np.nan, 3],
    ],
)
def test_rank_naoption_raises(rank_method, ascending, na_option, pct, vals):
    # 创建包含两列数据的 DataFrame，其中一列为 "key"，内容为 5 个 "foo"；另一列为 "val"，内容为 vals 参数指定的值
    df = DataFrame({"key": ["foo"] * 5, "val": vals})
    # 定义错误消息的内容
    msg = "na_option must be one of 'keep', 'top', or 'bottom'"

    # 使用 pytest 的 context manager 检查是否引发了 ValueError，并验证错误消息是否符合预期
    with pytest.raises(ValueError, match=msg):
        df.groupby("key").rank(
            method=rank_method, ascending=ascending, na_option=na_option, pct=pct
        )


def test_rank_empty_group():
    # see gh-22519
    column = "A"
    # 创建包含两列数据的 DataFrame，其中一列为 "A"，内容为 [0, 1, 0]；另一列为 "B"，内容为 [1.0, NaN, 2.0]
    df = DataFrame({"A": [0, 1, 0], "B": [1.0, np.nan, 2.0]})

    # 对 DataFrame 按 "A" 列进行分组，并对 "B" 列进行排名，返回排名结果的百分比
    result = df.groupby(column).B.rank(pct=True)
    # 创建期望的 Series，包含排名结果的百分比，期望的名称为 "B"
    expected = Series([0.5, np.nan, 1.0], name="B")
    # 使用 pytest 的 assert 函数，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 按 "A" 列进行分组，并对整个 DataFrame 进行排名，返回排名结果的百分比
    result = df.groupby(column).rank(pct=True)
    # 创建期望的 DataFrame，包含一列 "B"，值为 [0.5, NaN, 1.0]
    expected = DataFrame({"B": [0.5, np.nan, 1.0]})
    # 使用 pytest 的 assert 函数，比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input_key,input_value,output_value",
    [
        ([1, 2], [1, 1], [1.0, 1.0]),
        ([1, 1, 2, 2], [1, 2, 1, 2], [0.5, 1.0, 0.5, 1.0]),
        ([1, 1, 2, 2], [1, 2, 1, np.nan], [0.5, 1.0, 1.0, np.nan]),
        ([1, 1, 2], [1, 2, np.nan], [0.5, 1.0, np.nan]),
    ],
)
# 定义一个测试函数，用于测试按照组进行排名时的零除法情况
def test_rank_zero_div(input_key, input_value, output_value):
    # 创建一个DataFrame对象，包含"A"列和"B"列，其中"A"列由input_key提供，"B"列由input_value提供
    df = DataFrame({"A": input_key, "B": input_value})

    # 对DataFrame按照"A"列进行分组，然后计算每组中"B"列的排名，并使用dense方法，并计算百分位数
    result = df.groupby("A").rank(method="dense", pct=True)
    
    # 创建一个期望的DataFrame对象，包含"B"列和output_value提供的值
    expected = DataFrame({"B": output_value})
    
    # 使用测试工具函数assert_frame_equal比较计算结果和期望结果
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试最小整数的排名情况
def test_rank_min_int():
    # 创建一个DataFrame对象，包含"grp"列和"int_col"列，"grp"列包含1, 1, 2三个值，
    # "int_col"列包含np.int64类型的最小值、最大值以及最小值
    df = DataFrame(
        {
            "grp": [1, 1, 2],
            "int_col": [
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).max,
                np.iinfo(np.int64).min,
            ],
            "datetimelike": [NaT, datetime(2001, 1, 1), NaT],
        }
    )

    # 对DataFrame按照"grp"列进行分组，然后计算每组中每列的排名
    result = df.groupby("grp").rank()
    
    # 创建一个期望的DataFrame对象，包含"int_col"列和"datetimelike"列的预期排名结果
    expected = DataFrame(
        {"int_col": [1.0, 2.0, 1.0], "datetimelike": [np.nan, 1.0, np.nan]}
    )

    # 使用测试工具函数assert_frame_equal比较计算结果和期望结果
    tm.assert_frame_equal(result, expected)


# 定义一个参数化测试函数，测试在组之间转换时等值百分比排名的情况
@pytest.mark.parametrize("use_nan", [True, False])
def test_rank_pct_equal_values_on_group_transition(use_nan):
    # 根据use_nan参数选择填充值为NaN或者3
    fill_value = np.nan if use_nan else 3
    
    # 创建一个DataFrame对象，包含"group"列和"val"列，其中"val"列根据use_nan的值选择填充
    df = DataFrame(
        [
            [-1, 1],
            [-1, 2],
            [1, fill_value],
            [-1, fill_value],
        ],
        columns=["group", "val"],
    )
    
    # 对DataFrame按照"group"列进行分组，然后计算每组中"val"列的排名，使用dense方法，并计算百分位数
    result = df.groupby(["group"])["val"].rank(
        method="dense",
        pct=True,
    )
    
    # 根据use_nan参数选择期望的Series对象，其中包含val列的预期排名结果
    if use_nan:
        expected = Series([0.5, 1, np.nan, np.nan], name="val")
    else:
        expected = Series([1 / 3, 2 / 3, 1, 1], name="val")

    # 使用测试工具函数assert_series_equal比较计算结果和期望结果
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试非唯一索引情况下的排名
def test_non_unique_index():
    # 创建一个DataFrame对象，包含"A"列和"value"列，其中"A"列包含浮点数和NaN，
    # 使用pd.Timestamp("20170101", tz="US/Eastern")作为索引
    df = DataFrame(
        {"A": [1.0, 2.0, 3.0, np.nan], "value": 1.0},
        index=[pd.Timestamp("20170101", tz="US/Eastern")] * 4,
    )
    
    # 对DataFrame按照索引和"A"列进行分组，然后计算每组中"value"列的排名，升序，并计算百分位数
    result = df.groupby([df.index, "A"]).value.rank(ascending=True, pct=True)
    
    # 创建一个期望的Series对象，包含"value"列的预期排名结果，使用与df相同的索引
    expected = Series(
        [1.0, 1.0, 1.0, np.nan],
        index=[pd.Timestamp("20170101", tz="US/Eastern")] * 4,
        name="value",
    )
    
    # 使用测试工具函数assert_series_equal比较计算结果和期望结果
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试分类数据的排名情况
def test_rank_categorical():
    # 创建一个包含分类数据的DataFrame对象，包含"col1"列、"col2"列和"col3"列
    cat = pd.Categorical(["a", "a", "b", np.nan, "c", "b"], ordered=True)
    cat2 = pd.Categorical([1, 2, 3, np.nan, 4, 5], ordered=True)

    df = DataFrame({"col1": [0, 1, 0, 1, 0, 1], "col2": cat, "col3": cat2})

    # 对DataFrame按照"col1"列进行分组，然后计算每组中各列的排名
    gb = df.groupby("col1")

    res = gb.rank()

    # 创建一个期望的DataFrame对象，包含所有列的预期排名结果，将df转换为对象类型
    expected = df.astype(object).groupby("col1").rank()
    
    # 使用测试工具函数assert_frame_equal比较计算结果和期望结果
    tm.assert_frame_equal(res, expected)


# 定义一个参数化测试函数，测试在可空数据情况下的分组操作和排名计算
@pytest.mark.parametrize("na_option", ["top", "bottom"])
def test_groupby_op_with_nullables(na_option):
    # 创建一个包含Float64类型的DataFrame对象，包含"x"列，列值为None
    df = DataFrame({"x": [None]}, dtype="Float64")
    
    # 对DataFrame按照"x"列进行分组，不删除NaN值，然后计算每组中"x"列的排名，使用最小值方法，根据na_option选择处理NaN的方式
    result = df.groupby("x", dropna=False)["x"].rank(method="min", na_option=na_option)
    
    # 创建一个期望的Series对象，包含"x"列的预期排名结果，结果值为1.0
    expected = Series([1.0], dtype="Float64", name=result.name)
    
    # 使用测试工具函数assert_series_equal比较计算结果和期望结果
    tm.assert_series_equal(result, expected)
```