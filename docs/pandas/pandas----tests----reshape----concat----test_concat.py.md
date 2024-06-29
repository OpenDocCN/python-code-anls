# `D:\src\scipysrc\pandas\pandas\tests\reshape\concat\test_concat.py`

```
from collections import (
    abc,  # 导入collections模块中的abc子模块，用于抽象基类
    deque,  # 导入collections模块中的deque，用于双向队列的实现
)
from collections.abc import Iterator  # 导入collections.abc模块中的Iterator抽象基类
from datetime import datetime  # 导入datetime模块中的datetime类
from decimal import Decimal  # 导入decimal模块中的Decimal类
import itertools  # 导入itertools模块，用于高效的迭代工具函数

import numpy as np  # 导入NumPy库，并使用np作为别名
import pytest  # 导入pytest测试框架

from pandas.errors import InvalidIndexError  # 从pandas.errors模块中导入InvalidIndexError异常类

import pandas as pd  # 导入Pandas库，并使用pd作为别名
from pandas import (  # 从Pandas库中导入多个类和函数
    DataFrame,
    Index,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm  # 导入Pandas测试相关的模块，并使用tm作为别名
from pandas.core.arrays import SparseArray  # 从Pandas核心数组模块中导入SparseArray类
from pandas.tests.extension.decimal import to_decimal  # 从Pandas测试扩展模块中导入to_decimal函数


class TestConcatenate:
    def test_append_concat(self):
        # GH#1815
        # 创建两个日期范围对象，分别覆盖1990年至1999年和2000年至2009年，频率为每年年末
        d1 = date_range("12/31/1990", "12/31/1999", freq="YE-DEC")
        d2 = date_range("12/31/2000", "12/31/2009", freq="YE-DEC")

        # 使用随机生成的标准正态分布数据创建两个Series对象，分别使用d1和d2作为索引
        s1 = Series(np.random.default_rng(2).standard_normal(10), d1)
        s2 = Series(np.random.default_rng(2).standard_normal(10), d2)

        # 将两个Series对象转换为Period类型的Series
        s1 = s1.to_period()
        s2 = s2.to_period()

        # 执行concatenation操作，将s1和s2连接起来，返回连接后的结果
        result = concat([s1, s2])
        assert isinstance(result.index, PeriodIndex)  # 断言结果的索引类型为PeriodIndex
        assert result.index[0] == s1.index[0]  # 断言结果的第一个索引与s1的第一个索引相同

    def test_concat_copy(self):
        # 创建一个随机生成的4行3列标准正态分布数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
        # 创建一个随机生成的4行1列整数数据的DataFrame对象
        df2 = DataFrame(np.random.default_rng(2).integers(0, 10, size=4).reshape(4, 1))
        # 创建一个包含单列字典数据的DataFrame对象，索引为0到3
        df3 = DataFrame({5: "foo"}, index=range(4))

        # 执行concatenation操作，沿着列轴将df、df2和df3连接起来，返回连接后的结果
        result = concat([df, df2, df3], axis=1)
        # 遍历结果中的每一个数据块，断言其值的基础存储不为None，即确保它们是实际的副本
        for block in result._mgr.blocks:
            assert block.values.base is not None

        # 再次执行concatenation操作，沿着列轴将df、df2和df3连接起来，返回连接后的结果
        result = concat([df, df2, df3], axis=1)

        # 遍历结果中的每一个数据块
        for block in result._mgr.blocks:
            arr = block.values
            if arr.dtype.kind == "f":
                # 断言浮点数块的基础存储与df中的第一个数据块的值的基础存储相同
                assert arr.base is df._mgr.blocks[0].values.base
            elif arr.dtype.kind in ["i", "u"]:
                # 断言整数或无符号整数块的基础存储与df2中的第一个数据块的值的基础存储相同
                assert arr.base is df2._mgr.blocks[0].values.base
            elif arr.dtype == object:
                # 断言对象块的基础存储不为None
                assert arr.base is not None

        # 创建一个随机生成的4行1列标准正态分布数据的DataFrame对象
        df4 = DataFrame(np.random.default_rng(2).standard_normal((4, 1)))
        # 执行concatenation操作，沿着列轴将df、df2、df3和df4连接起来，返回连接后的结果
        result = concat([df, df2, df3, df4], axis=1)
        # 遍历结果中的每一个数据块
        for blocks in result._mgr.blocks:
            arr = blocks.values
            if arr.dtype.kind == "f":
                # 断言浮点数块是df或df4中某个数组的视图
                assert any(
                    np.shares_memory(arr, block.values)
                    for block in itertools.chain(df._mgr.blocks, df4._mgr.blocks)
                )
            elif arr.dtype.kind in ["i", "u"]:
                # 断言整数或无符号整数块的基础存储与df2中的第一个数据块的值的基础存储相同
                assert arr.base is df2._mgr.blocks[0].values.base
            elif arr.dtype == object:
                # 断言对象块是df3中某个数组的视图
                assert any(
                    np.shares_memory(arr, block.values) for block in df3._mgr.blocks
                )
    def test_concat_with_group_keys(self):
        # 创建一个 3x4 的随机数数据框 df
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)))
        # 创建一个 4x4 的随机数数据框 df2，与 df 共享相同的随机种子
        df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))

        # 使用 concat 函数按照指定键 [0, 1] 沿 axis=0 进行连接
        result = concat([df, df2], keys=[0, 1])
        # 创建预期的 MultiIndex 对象 exp_index，用于预期的 DataFrame 结果
        exp_index = MultiIndex.from_arrays(
            [[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2, 3]]
        )
        # 创建预期的 DataFrame 结果 expected，合并 df 和 df2 的值
        expected = DataFrame(np.r_[df.values, df2.values], index=exp_index)
        # 使用 assert_frame_equal 检查 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 concat 函数按照相同的键 [0, 1] 沿 axis=0 进行连接
        result = concat([df, df], keys=[0, 1])
        # 创建另一个预期的 MultiIndex 对象 exp_index2
        exp_index2 = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
        # 创建另一个预期的 DataFrame 结果 expected，合并 df 和 df 的值
        expected = DataFrame(np.r_[df.values, df.values], index=exp_index2)
        # 使用 assert_frame_equal 检查 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 使用 concat 函数按照指定键 [0, 1] 沿 axis=1 进行连接
        result = concat([df, df2], keys=[0, 1], axis=1)
        # 创建预期的 DataFrame 结果 expected，沿 axis=1 合并 df 和 df2 的值
        expected = DataFrame(np.c_[df.values, df2.values], columns=exp_index)
        # 使用 assert_frame_equal 检查 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 concat 函数按照相同的键 [0, 1] 沿 axis=1 进行连接
        result = concat([df, df], keys=[0, 1], axis=1)
        # 创建另一个预期的 DataFrame 结果 expected，沿 axis=1 合并 df 和 df 的值
        expected = DataFrame(np.c_[df.values, df.values], columns=exp_index2)
        # 使用 assert_frame_equal 检查 result 是否与 expected 相等
        tm.assert_frame_equal(result, expected)

    def test_concat_keys_specific_levels(self):
        # 创建一个 10x4 的随机数数据框 df
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 将 df 按照列索引切片，生成三个部分数据框，存储在列表 pieces 中
        pieces = [df.iloc[:, [0, 1]], df.iloc[:, [2]], df.iloc[:, [3]]]
        # 创建一个字符串列表 level 用于指定合并结果的层级标签
        level = ["three", "two", "one", "zero"]
        # 使用 concat 函数按照指定的 pieces 列表、axis=1、keys 和 levels 进行连接
        result = concat(
            pieces,
            axis=1,
            keys=["one", "two", "three"],
            levels=[level],
            names=["group_key"],
        )

        # 使用 tm.assert_index_equal 检查 result 的列级别是否与预期的 level 列表相等
        tm.assert_index_equal(result.columns.levels[0], Index(level, name="group_key"))
        # 使用 tm.assert_index_equal 检查 result 的列索引是否为预期的 [0, 1, 2, 3]
        tm.assert_index_equal(result.columns.levels[1], Index([0, 1, 2, 3]))

        # 使用 assert 检查 result 的列名是否符合预期的 ["group_key", None]
        assert result.columns.names == ["group_key", None]

    @pytest.mark.parametrize("mapping", ["mapping", "dict"])
    # 测试函数，用于测试 concat 函数在不同参数下的行为
    def test_concat_mapping(self, mapping, non_dict_mapping_subclass):
        # 根据 mapping 类型选择构造函数，如果是 "dict" 则选择 dict，否则选择 non_dict_mapping_subclass
        constructor = dict if mapping == "dict" else non_dict_mapping_subclass
        # 使用选择的构造函数构造 frames 字典，包含不同键对应的 DataFrame 对象
        frames = constructor(
            {
                "foo": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "bar": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "baz": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
                "qux": DataFrame(np.random.default_rng(2).standard_normal((4, 3))),
            }
        )

        # 对 frames 字典的键进行排序
        sorted_keys = list(frames.keys())

        # 测试 concat 函数，合并 frames 中所有 DataFrame，结果存入 result
        result = concat(frames)
        # 使用 concat 函数手动合并 frames 中的 DataFrame，结果存入 expected
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys)
        # 使用 assert_frame_equal 断言 result 与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 测试 concat 函数在指定 axis=1 下的行为
        result = concat(frames, axis=1)
        # 使用 concat 函数手动在 axis=1 方向合并 frames 中的 DataFrame，结果存入 expected
        expected = concat([frames[k] for k in sorted_keys], keys=sorted_keys, axis=1)
        # 使用 assert_frame_equal 断言 result 与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 测试 concat 函数在指定 keys 下的行为
        keys = ["baz", "foo", "bar"]
        result = concat(frames, keys=keys)
        # 使用 concat 函数手动按照指定 keys 合并 frames 中的 DataFrame，结果存入 expected
        expected = concat([frames[k] for k in keys], keys=keys)
        # 使用 assert_frame_equal 断言 result 与 expected 相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，用于测试 concat 函数在指定 keys 和 levels 下的行为
    def test_concat_keys_and_levels(self):
        # 创建两个随机数据框 df 和 df2
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)))

        # 定义 levels 和 names 用于 MultiIndex 的创建
        levels = [["foo", "baz"], ["one", "two"]]
        names = ["first", "second"]

        # 测试 concat 函数，传入多个数据框及其 keys、levels、names 参数
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
            names=names,
        )
        # 创建期望的合并结果 expected
        expected = concat([df, df2, df, df2])
        # 创建期望的索引对象 exp_index
        exp_index = MultiIndex(
            levels=levels + [[0]],
            codes=[[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]],
            names=names + [None],
        )
        # 将期望的索引对象赋给 expected 的索引
        expected.index = exp_index

        # 使用 assert_frame_equal 断言 result 与 expected 相等
        tm.assert_frame_equal(result, expected)

        # 测试不带 names 参数时的行为
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            levels=levels,
        )
        # 断言 result 的索引名称为空
        assert result.index.names == (None,) * 3

        # 测试不带 levels 参数时的行为
        result = concat(
            [df, df2, df, df2],
            keys=[("foo", "one"), ("foo", "two"), ("baz", "one"), ("baz", "two")],
            names=["first", "second"],
        )
        # 断言 result 的索引名称为 ("first", "second", None)
        assert result.index.names == ("first", "second", None)
        # 使用 assert_index_equal 断言 result 的第一个索引级别与指定的 Index 对象相等
        tm.assert_index_equal(
            result.index.levels[0], Index(["baz", "foo"], name="first")
        )
    def test_concat_keys_levels_no_overlap(self):
        # 测试用例：检查在不重叠情况下使用多级键和级别的连接
        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=["a"])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=["b"])

        # 设置错误消息
        msg = "Values not found in passed level"
        # 断言在连接时引发 ValueError，并检查错误消息
        with pytest.raises(ValueError, match=msg):
            concat([df, df], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

        # 设置错误消息
        msg = "Key one not in level"
        # 断言在连接时引发 ValueError，并检查错误消息
        with pytest.raises(ValueError, match=msg):
            concat([df, df2], keys=["one", "two"], levels=[["foo", "bar", "baz"]])

    def test_crossed_dtypes_weird_corner(self):
        # 测试用例：处理交叉数据类型的特殊情况
        columns = ["A", "B", "C", "D"]
        df1 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="f8"),
                "B": np.array([1, 2, 3, 4], dtype="i8"),
                "C": np.array([1, 2, 3, 4], dtype="f8"),
                "D": np.array([1, 2, 3, 4], dtype="i8"),
            },
            columns=columns,
        )

        df2 = DataFrame(
            {
                "A": np.array([1, 2, 3, 4], dtype="i8"),
                "B": np.array([1, 2, 3, 4], dtype="f8"),
                "C": np.array([1, 2, 3, 4], dtype="i8"),
                "D": np.array([1, 2, 3, 4], dtype="f8"),
            },
            columns=columns,
        )

        # 连接两个 DataFrame，忽略索引
        appended = concat([df1, df2], ignore_index=True)
        # 创建预期的 DataFrame
        expected = DataFrame(
            np.concatenate([df1.values, df2.values], axis=0), columns=columns
        )
        # 断言连接结果与预期结果相等
        tm.assert_frame_equal(appended, expected)

        df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=["a"])
        df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=["b"])
        # 连接两个 DataFrame，并指定键和名称
        result = concat([df, df2], keys=["one", "two"], names=["first", "second"])
        # 断言结果的索引名称正确
        assert result.index.names == ("first", "second")

    def test_with_mixed_tuples(self, sort):
        # 测试用例：处理包含混合元组的列
        # 创建第一个 DataFrame
        df1 = DataFrame({"A": "foo", ("B", 1): "bar"}, index=range(2))
        # 创建第二个 DataFrame
        df2 = DataFrame({"B": "foo", ("B", 1): "bar"}, index=range(2))

        # 进行连接操作，根据参数 sort 进行排序
        concat([df1, df2], sort=sort)
    def test_concat_mixed_objs_columns(self):
        # 测试混合类型序列/数据框的列连接操作（axis=1）
        # G2385

        # 创建一个日期范围作为索引，频率为每小时
        index = date_range("01-Jan-2013", periods=10, freq="h")
        # 创建一个包含10个元素的整数数组，数据类型为int64
        arr = np.arange(10, dtype="int64")
        # 使用arr数组和index索引创建一个Series对象s1
        s1 = Series(arr, index=index)
        # 再次使用相同的arr数组和index索引创建另一个Series对象s2
        s2 = Series(arr, index=index)
        # 使用arr数组创建一个数据框df，形状为(-1, 1)，索引为index
        df = DataFrame(arr.reshape(-1, 1), index=index)

        # 期望的结果数据框，使用arr数组重复两次形成(-1, 2)形状，索引为index，列名为[0, 0]
        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 0]
        )
        # 对df和df进行列方向上的连接，结果保存在result中
        result = concat([df, df], axis=1)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 期望的结果数据框，使用arr数组重复两次形成(-1, 2)形状，索引为index，列名为[0, 1]
        expected = DataFrame(
            np.repeat(arr, 2).reshape(-1, 2), index=index, columns=[0, 1]
        )
        # 对s1和s2进行列方向上的连接，结果保存在result中
        result = concat([s1, s2], axis=1)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 期望的结果数据框，使用arr数组重复三次形成(-1, 3)形状，索引为index，列名为[0, 1, 2]
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        # 对s1、s2和再次使用s1进行列方向上的连接，结果保存在result中
        result = concat([s1, s2, s1], axis=1)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 期望的结果数据框，使用arr数组重复五次形成(-1, 5)形状，索引为index，列名为[0, 0, 1, 2, 3]
        expected = DataFrame(
            np.repeat(arr, 5).reshape(-1, 5), index=index, columns=[0, 0, 1, 2, 3]
        )
        # 对s1、df、s2、s2、s1进行列方向上的连接，结果保存在result中
        result = concat([s1, df, s2, s2, s1], axis=1)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 设置s1的名称为"foo"
        s1.name = "foo"
        # 期望的结果数据框，使用arr数组重复三次形成(-1, 3)形状，索引为index，列名为["foo", 0, 0]
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, 0]
        )
        # 对s1、df、s2进行列方向上的连接，结果保存在result中
        result = concat([s1, df, s2], axis=1)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 设置s2的名称为"bar"
        s2.name = "bar"
        # 期望的结果数据框，使用arr数组重复三次形成(-1, 3)形状，索引为index，列名为["foo", 0, "bar"]
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=["foo", 0, "bar"]
        )
        # 对s1、df、s2进行列方向上的连接，结果保存在result中
        result = concat([s1, df, s2], axis=1)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

        # 期望的结果数据框，使用arr数组重复三次形成(-1, 3)形状，索引为index，列名为[0, 1, 2]
        expected = DataFrame(
            np.repeat(arr, 3).reshape(-1, 3), index=index, columns=[0, 1, 2]
        )
        # 对s1、df、s2进行列方向上的连接，并忽略索引，结果保存在result中
        result = concat([s1, df, s2], axis=1, ignore_index=True)
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)

    def test_concat_mixed_objs_index(self):
        # 测试混合类型序列/数据框的行连接操作，它们有一个共同的索引名称
        # GH2385, GH15047

        # 创建一个日期范围作为索引，频率为每小时
        index = date_range("01-Jan-2013", periods=10, freq="h")
        # 创建一个包含10个元素的整数数组，数据类型为int64
        arr = np.arange(10, dtype="int64")
        # 使用arr数组和index索引创建一个Series对象s1
        s1 = Series(arr, index=index)
        # 再次使用相同的arr数组和index索引创建另一个Series对象s2
        s2 = Series(arr, index=index)
        # 使用arr数组创建一个数据框df，形状为(-1, 1)，索引为index
        df = DataFrame(arr.reshape(-1, 1), index=index)

        # 期望的结果数据框，使用arr数组重复三次形成(-1, 1)形状，索引为index的三倍，列名为[0]
        expected = DataFrame(
            np.tile(arr, 3).reshape(-1, 1), index=index.tolist() * 3, columns=[0]
        )
        # 对s1、df和s2进行行方向上的连接，结果保存在result中
        result = concat([s1, df, s2])
        # 断言result与expected相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，测试混合序列/数据框按行连接，并确保列名唯一
    def test_concat_mixed_objs_index_names(self):
        # 使用日期范围生成时间索引，从 "01-Jan-2013" 开始，每小时频率，共 10 个时间点
        index = date_range("01-Jan-2013", periods=10, freq="h")
        # 创建一个包含 0 到 9 的整数数组，数据类型为 int64
        arr = np.arange(10, dtype="int64")
        # 使用 arr 数组和时间索引创建 Series 对象 s1，列名为 "foo"
        s1 = Series(arr, index=index, name="foo")
        # 使用 arr 数组和时间索引创建另一个 Series 对象 s2，列名为 "bar"
        s2 = Series(arr, index=index, name="bar")
        # 使用 arr 数组和时间索引创建 DataFrame 对象 df，包含一列数据
        df = DataFrame(arr.reshape(-1, 1), index=index)

        # 生成预期的 DataFrame 对象 expected，包含根据条件生成的数据
        expected = DataFrame(
            np.kron(np.where(np.identity(3) == 1, 1, np.nan), arr).T,
            index=index.tolist() * 3,
            columns=["foo", 0, "bar"],
        )
        # 将 s1、df 和 s2 按行连接，得到连接后的结果 result
        result = concat([s1, df, s2])
        # 断言 result 和 expected 的数据框内容一致
        tm.assert_frame_equal(result, expected)

        # 当 ignore_index=True 时，将所有 Series 对象重命名为 0
        expected = DataFrame(np.tile(arr, 3).reshape(-1, 1), columns=[0])
        # 将 s1、df 和 s2 按行连接，并忽略索引，得到连接后的结果 result
        result = concat([s1, df, s2], ignore_index=True)
        # 断言 result 和 expected 的数据框内容一致
        tm.assert_frame_equal(result, expected)

    # 测试数据类型强制转换的函数
    def test_dtype_coercion(self):
        # 测试 12411 情况
        df = DataFrame({"date": [pd.Timestamp("20130101").tz_localize("UTC"), pd.NaT]})
        # 选择 DataFrame 中的第一行，并将其与自身连接，得到连接后的结果 result
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        # 断言 result 的数据类型与 df 的数据类型相等
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 测试 12045 情况
        df = DataFrame({"date": [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        # 选择 DataFrame 中的第一行，并将其与自身连接，得到连接后的结果 result
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        # 断言 result 的数据类型与 df 的数据类型相等
        tm.assert_series_equal(result.dtypes, df.dtypes)

        # 测试 11594 情况
        df = DataFrame({"text": ["some words"] + [None] * 9})
        # 选择 DataFrame 中的第一行，并将其与自身连接，得到连接后的结果 result
        result = concat([df.iloc[[0]], df.iloc[[1]]])
        # 断言 result 的数据类型与 df 的数据类型相等
        tm.assert_series_equal(result.dtypes, df.dtypes)

    # 测试带有键值的单一数据框连接函数
    def test_concat_single_with_key(self):
        # 创建一个大小为 (10, 4) 的随机标准正态分布的 DataFrame 对象 df
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        # 将 df 和键值 "foo" 连接，得到连接后的结果 result
        result = concat([df], keys=["foo"])
        # 生成预期的 DataFrame 对象 expected，包含两次 df 的连接结果，分别以 "foo" 和 "bar" 命名
        expected = concat([df, df], keys=["foo", "bar"])
        # 断言 result 和 expected 的前 10 行数据框内容一致
        tm.assert_frame_equal(result, expected[:10])

    # 测试当没有要连接的对象时引发异常的函数
    def test_concat_no_items_raises(self):
        # 使用 pytest 引发 ValueError 异常，匹配错误信息 "No objects to concatenate"
        with pytest.raises(ValueError, match="No objects to concatenate"):
            concat([])

    # 测试排除所有 None 对象的连接函数
    def test_concat_exclude_none(self):
        # 创建一个大小为 (10, 4) 的随机标准正态分布的 DataFrame 对象 df
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))

        # 创建一个包含 None 和部分 df 片段的列表 pieces，并将它们连接，得到连接后的结果 result
        pieces = [df[:5], None, None, df[5:]]
        result = concat(pieces)
        # 断言 result 和 df 的数据框内容一致
        tm.assert_frame_equal(result, df)
        
        # 使用 pytest 引发 ValueError 异常，匹配错误信息 "All objects passed were None"
        with pytest.raises(ValueError, match="All objects passed were None"):
            concat([None, None])

    # 测试带有 None 对象键值的连接函数
    def test_concat_keys_with_none(self):
        # #1649 情况
        df0 = DataFrame([[10, 20, 30], [10, 20, 30], [10, 20, 30]])

        # 使用键值连接包含 None 对象的字典，得到连接后的结果 result
        result = concat({"a": None, "b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        # 生成预期的 DataFrame 对象 expected，去除键值为 None 的部分
        expected = concat({"b": df0, "c": df0[:2], "d": df0[:1], "e": df0})
        # 断言 result 和 expected 的数据框内容一致
        tm.assert_frame_equal(result, expected)

        # 使用带有 None 对象的列表连接函数，同时指定键值，得到连接后的结果 result
        result = concat(
            [None, df0, df0[:2], df0[:1], df0], keys=["a", "b", "c", "d", "e"]
        )
        # 生成预期的 DataFrame 对象 expected，去除键值为 "a" 的部分
        expected = concat([df0, df0[:2], df0[:1], df0], keys=["b", "c", "d", "e"])
        # 断言 result 和 expected 的数据框内容一致
        tm.assert_frame_equal(result, expected)
    @pytest.mark.parametrize("include_none", [True, False])
    # 使用 pytest 的参数化功能，测试函数 test_concat_preserves_rangeindex 将会分别以 include_none=True 和 include_none=False 的参数运行两次
    
    def test_concat_preserves_rangeindex(self, klass, include_none):
        # 创建包含两个元素的 DataFrame 对象 df 和 df2，分别包含数值 1, 2 和 3, 4
        df = DataFrame([1, 2])
        df2 = DataFrame([3, 4])
        
        # 根据 include_none 参数决定 data 列表的内容，可能包含 None
        data = [df, None, df2, None] if include_none else [df, df2]
        
        # 根据 include_none 参数设定 keys_length 的值，可能为 4 或 2
        keys_length = 4 if include_none else 2
        
        # 使用 concat 函数将 data 列表中的 DataFrame 对象连接起来，指定 keys 参数为 klass(keys_length)
        result = concat(data, keys=klass(keys_length))
        
        # 创建期望的 DataFrame 对象 expected，包含数值 [1, 2, 3, 4]，并设定其索引为 MultiIndex 对象
        expected = DataFrame(
            [1, 2, 3, 4],
            index=MultiIndex(
                levels=(
                    # 设定第一层级索引为 RangeIndex，从 0 到 keys_length 步长为 keys_length / 2
                    RangeIndex(start=0, stop=keys_length, step=keys_length / 2),
                    # 设定第二层级索引为 RangeIndex，从 0 到 2 步长为 1
                    RangeIndex(start=0, stop=2, step=1),
                ),
                codes=(
                    np.array([0, 0, 1, 1], dtype=np.int8),  # 第一层级的编码
                    np.array([0, 1, 0, 1], dtype=np.int8),  # 第二层级的编码
                ),
            ),
        )
        
        # 使用 pandas 的 tm.assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    
    
    def test_concat_bug_1719(self):
        # 创建一个包含 10 个浮点数的 Series 对象 ts1，并设定其索引为从 "2020-01-01" 开始的 10 个日期
        ts1 = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        
        # 从 ts1 中取出每隔一个元素的子序列，创建 Series 对象 ts2
        ts2 = ts1[::2]
        
        # 使用 concat 函数将 ts1 和 ts2 进行外连接（join="outer"），沿着 axis=1 轴连接起来
        left = concat([ts1, ts2], join="outer", axis=1)
        right = concat([ts2, ts1], join="outer", axis=1)
        
        # 断言 left 和 right 的长度是否相等
        assert len(left) == len(right)
    
    
    def test_concat_bug_2972(self):
        # 创建两个 Series 对象 ts0 和 ts1，分别包含 5 个零和 5 个一，并设定它们的 name 属性为 "same name"
        ts0 = Series(np.zeros(5))
        ts1 = Series(np.ones(5))
        ts0.name = ts1.name = "same name"
        
        # 使用 concat 函数将 ts0 和 ts1 沿着 axis=1 轴连接起来，结果为 DataFrame 对象 result
        result = concat([ts0, ts1], axis=1)
        
        # 创建期望的 DataFrame 对象 expected，包含 ts0 和 ts1 作为列，列名为 "same name"
        expected = DataFrame({0: ts0, 1: ts1})
        expected.columns = ["same name", "same name"]
        
        # 使用 pandas 的 tm.assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    
    
    def test_concat_bug_3602(self):
        # 创建 DataFrame 对象 df1 和 df2，分别包含列 "firmNo", "prc", "stringvar" 和 "C", "misc", "prc" 的数据
        df1 = DataFrame(
            {
                "firmNo": [0, 0, 0, 0],
                "prc": [6, 6, 6, 6],
                "stringvar": ["rrr", "rrr", "rrr", "rrr"],
            }
        )
        df2 = DataFrame(
            {"C": [9, 10, 11, 12], "misc": [1, 2, 3, 4], "prc": [6, 6, 6, 6]}
        )
        
        # 创建期望的 DataFrame 对象 expected，合并 df1 和 df2，列名可能会有重复，需要处理
        expected = DataFrame(
            [
                [0, 6, "rrr", 9, 1, 6],
                [0, 6, "rrr", 10, 2, 6],
                [0, 6, "rrr", 11, 3, 6],
                [0, 6, "rrr", 12, 4, 6],
            ]
        )
        expected.columns = ["firmNo", "prc", "stringvar", "C", "misc", "prc"]
        
        # 使用 concat 函数将 df1 和 df2 沿着 axis=1 轴连接起来，结果为 DataFrame 对象 result
        result = concat([df1, df2], axis=1)
        
        # 使用 pandas 的 tm.assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    def test_concat_iterables(self):
        # GH8645 检查 concat 能够处理元组、列表、生成器以及特殊的数据结构如deque和自定义可迭代对象
        df1 = DataFrame([1, 2, 3])  # 创建一个包含整数的DataFrame df1
        df2 = DataFrame([4, 5, 6])  # 创建一个包含整数的DataFrame df2
        expected = DataFrame([1, 2, 3, 4, 5, 6])  # 创建预期的DataFrame，合并df1和df2
        tm.assert_frame_equal(concat((df1, df2), ignore_index=True), expected)  # 使用concat合并df1和df2，忽略索引，检查结果是否与预期相同
        tm.assert_frame_equal(concat([df1, df2], ignore_index=True), expected)  # 使用concat合并df1和df2（作为列表提供），忽略索引，检查结果是否与预期相同
        tm.assert_frame_equal(
            concat((df for df in (df1, df2)), ignore_index=True), expected
        )  # 使用concat合并生成器中的df1和df2，忽略索引，检查结果是否与预期相同
        tm.assert_frame_equal(concat(deque((df1, df2)), ignore_index=True), expected)  # 使用concat合并deque中的df1和df2，忽略索引，检查结果是否与预期相同

        class CustomIterator1:
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index):
                try:
                    return {0: df1, 1: df2}[index]  # 返回自定义索引对应的df1或df2
                except KeyError as err:
                    raise IndexError from err

        tm.assert_frame_equal(concat(CustomIterator1(), ignore_index=True), expected)  # 使用concat合并自定义迭代器CustomIterator1，忽略索引，检查结果是否与预期相同

        class CustomIterator2(abc.Iterable):
            def __iter__(self) -> Iterator:
                yield df1
                yield df2

        tm.assert_frame_equal(concat(CustomIterator2(), ignore_index=True), expected)  # 使用concat合并自定义迭代器CustomIterator2，忽略索引，检查结果是否与预期相同

    def test_concat_order(self):
        # GH 17344, GH#47331
        dfs = [DataFrame(index=range(3), columns=["a", 1, None])]  # 创建一个指定索引和列名的DataFrame列表
        dfs += [DataFrame(index=range(3), columns=[None, 1, "a"]) for _ in range(100)]  # 将相同结构的DataFrame重复100次加入列表中

        result = concat(dfs, sort=True).columns  # 使用concat合并dfs列表中的DataFrame，按列名排序，获取合并后的列索引
        expected = Index([1, "a", None])  # 创建预期的列索引
        tm.assert_index_equal(result, expected)  # 检查实际结果的列索引是否与预期相同

    def test_concat_different_extension_dtypes_upcasts(self):
        a = Series(pd.array([1, 2], dtype="Int64"))  # 创建整数Series a
        b = Series(to_decimal([1, 2]))  # 创建Decimal类型Series b

        result = concat([a, b], ignore_index=True)  # 使用concat合并Series a 和 b，忽略索引
        expected = Series([1, 2, Decimal(1), Decimal(2)], dtype=object)  # 创建预期的Series对象，包含合并后的数据
        tm.assert_series_equal(result, expected)  # 检查实际结果的Series对象是否与预期相同

    def test_concat_ordered_dict(self):
        # GH 21510
        expected = concat(
            [Series(range(3)), Series(range(4))], keys=["First", "Another"]
        )  # 使用concat合并具有不同键的Series列表，创建预期的结果Series
        result = concat({"First": Series(range(3)), "Another": Series(range(4))})  # 使用concat合并具有不同键的Series字典，创建实际结果Series
        tm.assert_series_equal(result, expected)  # 检查实际结果的Series对象是否与预期相同

    def test_concat_duplicate_indices_raise(self):
        # GH 45888: 测试合并具有重复索引的DataFrame时是否引发异常
        df1 = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=[0, 1, 2, 3, 3],  # 创建具有重复索引的DataFrame df1
            columns=["a"],
        )
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal(5),
            index=[0, 1, 2, 2, 4],  # 创建具有重复索引的DataFrame df2
            columns=["b"],
        )
        msg = "Reindexing only valid with uniquely valued Index objects"  # 预期的异常消息
        with pytest.raises(InvalidIndexError, match=msg):  # 断言是否引发指定异常并匹配消息
            concat([df1, df2], axis=1)  # 使用concat合并具有重复索引的DataFrame列表，按列合并
def test_concat_no_unnecessary_upcast(float_numpy_dtype, frame_or_series):
    # GH 13247
    # 使用对象构造函数调用传入的frame_or_series函数，返回其维度
    dims = frame_or_series(dtype=object).ndim
    # 将float_numpy_dtype赋值给dt变量
    dt = float_numpy_dtype

    # 创建包含3个元素的数据帧或系列列表
    dfs = [
        # 使用np.array创建一个具有1的数组，并设置其数据类型为dt，ndmin为dims的数组
        frame_or_series(np.array([1], dtype=dt, ndmin=dims)),
        # 使用np.array创建一个包含NaN的数组，并设置其数据类型为dt，ndmin为dims的数组
        frame_or_series(np.array([np.nan], dtype=dt, ndmin=dims)),
        # 使用np.array创建一个具有5的数组，并设置其数据类型为dt，ndmin为dims的数组
        frame_or_series(np.array([5], dtype=dt, ndmin=dims)),
    ]
    # 使用concat函数将dfs列表中的数据帧或系列连接起来，赋值给x
    x = concat(dfs)
    # 使用断言确保x的值的数据类型等于dt
    assert x.values.dtype == dt


def test_concat_will_upcast(frame_or_series, any_signed_int_numpy_dtype):
    # 获取任意有符号整数数据类型，并将其赋值给dt变量
    dt = any_signed_int_numpy_dtype
    # 获取frame_or_series函数的维度
    dims = frame_or_series().ndim
    # 创建包含3个元素的数据帧或系列列表
    dfs = [
        # 使用np.array创建一个具有1的数组，并设置其数据类型为dt，ndmin为dims的数组
        frame_or_series(np.array([1], dtype=dt, ndmin=dims)),
        # 使用np.array创建一个包含NaN的数组，并设置其维度
        frame_or_series(np.array([np.nan], ndmin=dims)),
        # 使用np.array创建一个具有5的数组，并设置其数据类型为dt，ndmin为dims的数组
        frame_or_series(np.array([5], dtype=dt, ndmin=dims)),
    ]
    # 使用concat函数将dfs列表中的数据帧或系列连接起来，赋值给x
    x = concat(dfs)
    # 使用断言确保x的值的数据类型为"float64"
    assert x.values.dtype == "float64"


def test_concat_empty_and_non_empty_frame_regression():
    # GH 18178 regression test
    # 创建一个包含{"foo": [1]}的数据帧，并将其赋值给df1
    df1 = DataFrame({"foo": [1]})
    # 创建一个空的数据帧，并将其赋值给df2
    df2 = DataFrame({"foo": []})
    # 创建一个期望的数据帧，包含{"foo": [1.0]}，并将其赋值给expected
    expected = DataFrame({"foo": [1.0]})
    # 使用concat函数将df1和df2连接起来，赋值给result
    result = concat([df1, df2])
    # 使用tm.assert_frame_equal函数确保result和expected的内容相等
    tm.assert_frame_equal(result, expected)


def test_concat_sparse():
    # GH 23557
    # 创建一个稀疏系列，其数据为SparseArray([0, 1, 2])，并将其赋值给a
    a = Series(SparseArray([0, 1, 2]))
    # 创建一个期望的数据帧，其数据为[[0, 0], [1, 1], [2, 2]]，并将其赋值给expected
    expected = DataFrame(data=[[0, 0], [1, 1], [2, 2]]).astype(
        pd.SparseDtype(np.int64, 0)
    )
    # 使用concat函数将a连接起来，赋值给result
    result = concat([a, a], axis=1)
    # 使用tm.assert_frame_equal函数确保result和expected的内容相等
    tm.assert_frame_equal(result, expected)


def test_concat_dense_sparse():
    # GH 30668
    # 创建一个稀疏数据类型为pd.SparseDtype(np.float64, None)的系列，其数据为pd.arrays.SparseArray([1, None])，并将其赋值给a
    a = Series(pd.arrays.SparseArray([1, None]), dtype=pd.SparseDtype(np.float64, None))
    # 创建一个浮点数数据类型的系列，其数据为[1]，并将其赋值给b
    b = Series([1], dtype=float)
    # 创建一个期望的系列，其数据为[1, None, 1]，索引为[0, 1, 0]，并将其赋值给expected
    expected = Series(data=[1, None, 1], index=[0, 1, 0]).astype(pd.SparseDtype(np.float64, None))
    # 使用concat函数将a和b连接起来，赋值给result
    result = concat([a, b], axis=0)
    # 使用tm.assert_series_equal函数确保result和expected的内容相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", [["e", "f", "f"], ["f", "e", "f"]])
def test_duplicate_keys(keys):
    # GH 33654
    # 创建一个数据帧，包含列"a": [1, 2, 3]和"b": [4, 5, 6]，并将其赋值给df
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 创建一个包含Series([7, 8, 9], name="c")的系列，并将其赋值给s1
    s1 = Series([7, 8, 9], name="c")
    # 创建一个包含Series([10, 11, 12], name="d")的系列，并将其赋值给s2
    s2 = Series([10, 11, 12], name="d")
    # 使用concat函数将df、s1和s2连接起来，设置keys参数为传入的keys列表，赋值给result
    result = concat([df, s1, s2], axis=1, keys=keys)
    # 创建一个期望的数据帧，其值为[[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]，列为MultiIndex.from_tuples([(keys[0], "a"), (keys[0], "b"), (keys[1], "c"), (keys[2], "d")])，并将其赋值给expected
    expected_values = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1], "c"), (keys[2], "d")]
    )
    expected = DataFrame(expected_values, columns=expected_columns)
    # 使用tm.assert_frame_equal函数确保result和expected的内容相等
    tm.assert_frame_equal(result, expected)


def test_duplicate_keys_same_frame():
    # GH 43595
    # 创建一个包含列"a": [1, 2, 3]和"b": [4, 5, 6]的数据帧，并将其赋值给df
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 使用concat函数将df和df连接起来，设置keys参数为传入的keys列表，赋值给result
    keys = ["e", "e"]
    result = concat([df, df], axis=1, keys=keys)
    # 创建一个期望的数据帧，其值为[[1, 4, 1, 4], [2, 5, 2, 5], [3, 6, 3, 6]]，列为MultiIndex.from_tuples([(keys[0], "a"), (keys[0], "b"), (keys[1], "a"), (keys[1], "b")])，并将其赋值给expected
    expected_values = [[1, 4, 1, 4], [2, 5, 2, 5], [3, 6, 3, 6]]
    expected_columns = MultiIndex.from_tuples(
        [(keys[0], "a"), (keys[0], "b"), (keys[1
    [
        # 创建一个 SubclassedDataFrame 对象，其中包含一列从 0 到 9 的整数，列名为 "A"
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        # 创建一个 SubclassedSeries 对象，包含从 0 到 9 的整数，且命名为 "A"
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
def test_concat_preserves_subclass(obj):
    # GH28330 -- preserve subclass
    # 对象拼接函数 concat 应保持对象的子类特性

    result = concat([obj, obj])
    # 对象拼接操作

    assert isinstance(result, type(obj))
    # 断言结果应为原对象的同一类型


def test_concat_frame_axis0_extension_dtypes():
    # preserve extension dtype (through common_dtype mechanism)
    # 保留扩展数据类型（通过 common_dtype 机制）

    df1 = DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})
    # 创建 DataFrame df1，使用扩展类型 Int64

    df2 = DataFrame({"a": np.array([4, 5, 6])})
    # 创建 DataFrame df2，使用普通 NumPy 数组

    result = concat([df1, df2], ignore_index=True)
    # 拼接 df1 和 df2，忽略索引，使用公共数据类型机制

    expected = DataFrame({"a": [1, 2, 3, 4, 5, 6]}, dtype="Int64")
    # 期望的拼接结果 DataFrame，保持扩展类型 Int64

    tm.assert_frame_equal(result, expected)
    # 使用测试工具断言结果与期望相等

    result = concat([df2, df1], ignore_index=True)
    # 拼接 df2 和 df1，忽略索引，使用公共数据类型机制

    expected = DataFrame({"a": [4, 5, 6, 1, 2, 3]}, dtype="Int64")
    # 期望的拼接结果 DataFrame，保持扩展类型 Int64

    tm.assert_frame_equal(result, expected)
    # 使用测试工具断言结果与期望相等


def test_concat_preserves_extension_int64_dtype():
    # GH 24768
    # 保持扩展类型 Int64 的数据类型

    df_a = DataFrame({"a": [-1]}, dtype="Int64")
    # 创建 DataFrame df_a，使用扩展类型 Int64

    df_b = DataFrame({"b": [1]}, dtype="Int64")
    # 创建 DataFrame df_b，使用扩展类型 Int64

    result = concat([df_a, df_b], ignore_index=True)
    # 拼接 df_a 和 df_b，忽略索引

    expected = DataFrame({"a": [-1, None], "b": [None, 1]}, dtype="Int64")
    # 期望的拼接结果 DataFrame，保持扩展类型 Int64

    tm.assert_frame_equal(result, expected)
    # 使用测试工具断言结果与期望相等


@pytest.mark.parametrize(
    "dtype1,dtype2,expected_dtype",
    [
        ("bool", "bool", "bool"),
        ("boolean", "bool", "boolean"),
        ("bool", "boolean", "boolean"),
        ("boolean", "boolean", "boolean"),
    ],
)
def test_concat_bool_types(dtype1, dtype2, expected_dtype):
    # GH 42800
    # 布尔类型拼接测试

    ser1 = Series([True, False], dtype=dtype1)
    # 创建 Series ser1，指定数据类型 dtype1

    ser2 = Series([False, True], dtype=dtype2)
    # 创建 Series ser2，指定数据类型 dtype2

    result = concat([ser1, ser2], ignore_index=True)
    # 拼接 ser1 和 ser2，忽略索引

    expected = Series([True, False, False, True], dtype=expected_dtype)
    # 期望的拼接结果 Series，指定数据类型 expected_dtype

    tm.assert_series_equal(result, expected)
    # 使用测试工具断言结果与期望相等


@pytest.mark.parametrize(
    ("keys", "integrity"),
    [
        (["red"] * 3, True),
        (["red"] * 3, False),
        (["red", "blue", "red"], False),
        (["red", "blue", "red"], True),
    ],
)
def test_concat_repeated_keys(keys, integrity):
    # GH: 20816
    # 处理重复键的拼接测试

    series_list = [Series({"a": 1}), Series({"b": 2}), Series({"c": 3})]
    # 创建一组 Series 列表

    result = concat(series_list, keys=keys, verify_integrity=integrity)
    # 拼接 series_list，使用给定的键和完整性验证参数

    tuples = list(zip(keys, ["a", "b", "c"]))
    # 创建键和列名的元组列表

    expected = Series([1, 2, 3], index=MultiIndex.from_tuples(tuples))
    # 期望的拼接结果 Series，使用多重索引

    tm.assert_series_equal(result, expected)
    # 使用测试工具断言结果与期望相等


def test_concat_null_object_with_dti():
    # GH#40841
    # 处理包含 DatetimeIndex 的空对象拼接测试

    dti = pd.DatetimeIndex(
        ["2021-04-08 21:21:14+00:00"], dtype="datetime64[ns, UTC]", name="Time (UTC)"
    )
    # 创建 DatetimeIndex dti，指定数据类型和名称

    right = DataFrame(data={"C": [0.5274]}, index=dti)
    # 创建 DataFrame right，使用 dti 作为索引

    idx = Index([None], dtype="object", name="Maybe Time (UTC)")
    # 创建对象索引 idx，使用对象类型和名称

    left = DataFrame(data={"A": [None], "B": [np.nan]}, index=idx)
    # 创建 DataFrame left，使用 idx 作为索引

    result = concat([left, right], axis="columns")
    # 按列拼接 left 和 right

    exp_index = Index([None, dti[0]], dtype=object)
    # 创建期望的索引对象，包括 None 和 dti 的第一个元素

    expected = DataFrame(
        {
            "A": np.array([None, np.nan], dtype=object),
            "B": [np.nan, np.nan],
            "C": [np.nan, 0.5274],
        },
        index=exp_index,
    )
    # 期望的拼接结果 DataFrame，包括指定的索引对象

    tm.assert_frame_equal(result, expected)
    # 使用测试工具断言结果与期望相等


def test_concat_multiindex_with_empty_rangeindex():
    # GH#41234
    # 处理包含空 RangeIndex 的多重索引拼接测试
    # 使用 MultiIndex.from_tuples 创建一个包含多级索引的元组列表 mi
    mi = MultiIndex.from_tuples([("B", 1), ("C", 1)])
    # 使用 DataFrame 创建一个包含数据 [[1, 2]] 的 DataFrame，列索引使用 mi
    df1 = DataFrame([[1, 2]], columns=mi)
    # 使用 DataFrame 创建一个指定索引和列的空 DataFrame
    df2 = DataFrame(index=[1], columns=RangeIndex(0))
    
    # 使用 concat 函数将 df1 和 df2 沿行方向合并，生成一个新的 DataFrame result
    result = concat([df1, df2])
    # 创建一个期望的 DataFrame，包含数据 [[1, 2], [np.nan, np.nan]] 和与 df1 相同的列索引 mi
    expected = DataFrame([[1, 2], [np.nan, np.nan]], columns=mi)
    
    # 使用 tm.assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "data",
    [
        Series(data=[1, 2]),  # 创建一个包含整数数据的 Pandas Series 对象
        DataFrame(
            data={
                "col1": [1, 2],  # 创建一个包含整数数据的 Pandas DataFrame 对象，具有一个列名为 'col1'
            }
        ),
        DataFrame(dtype=float),  # 创建一个空的 Pandas DataFrame 对象，列的数据类型为 float
        Series(dtype=float),  # 创建一个空的 Pandas Series 对象，数据类型为 float
    ],
)
def test_concat_drop_attrs(data):
    # GH#41828
    df1 = data.copy()  # 复制传入的数据对象
    df1.attrs = {1: 1}  # 给 df1 对象添加属性字典 {1: 1}
    df2 = data.copy()  # 复制传入的数据对象
    df2.attrs = {1: 2}  # 给 df2 对象添加属性字典 {1: 2}
    df = concat([df1, df2])  # 将 df1 和 df2 进行连接操作，生成一个新的 DataFrame 对象 df
    assert len(df.attrs) == 0  # 断言 df 对象的属性字典长度为 0


@pytest.mark.parametrize(
    "data",
    [
        Series(data=[1, 2]),  # 创建一个包含整数数据的 Pandas Series 对象
        DataFrame(
            data={
                "col1": [1, 2],  # 创建一个包含整数数据的 Pandas DataFrame 对象，具有一个列名为 'col1'
            }
        ),
        DataFrame(dtype=float),  # 创建一个空的 Pandas DataFrame 对象，列的数据类型为 float
        Series(dtype=float),  # 创建一个空的 Pandas Series 对象，数据类型为 float
    ],
)
def test_concat_retain_attrs(data):
    # GH#41828
    df1 = data.copy()  # 复制传入的数据对象
    df1.attrs = {1: 1}  # 给 df1 对象添加属性字典 {1: 1}
    df2 = data.copy()  # 复制传入的数据对象
    df2.attrs = {1: 1}  # 给 df2 对象添加属性字典 {1: 1}
    df = concat([df1, df2])  # 将 df1 和 df2 进行连接操作，生成一个新的 DataFrame 对象 df
    assert df.attrs[1] == 1  # 断言 df 对象的属性字典中键为 1 的值为 1


@pytest.mark.parametrize("df_dtype", ["float64", "int64", "datetime64[ns]"])
@pytest.mark.parametrize("empty_dtype", [None, "float64", "object"])
def test_concat_ignore_empty_object_float(empty_dtype, df_dtype):
    # https://github.com/pandas-dev/pandas/issues/45637
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]}, dtype=df_dtype)  # 创建一个指定数据类型的 Pandas DataFrame 对象 df
    empty = DataFrame(columns=["foo", "bar"], dtype=empty_dtype)  # 创建一个空的 Pandas DataFrame 对象 empty

    needs_update = False
    if df_dtype == "datetime64[ns]" or (
        df_dtype == "float64" and empty_dtype != "float64"
    ):
        needs_update = True

    result = concat([empty, df])  # 将 empty 和 df 进行连接操作，生成一个新的 DataFrame 对象 result
    expected = df  # 将 df 赋值给 expected

    if df_dtype == "int64":
        # TODO what exact behaviour do we want for integer eventually?
        if empty_dtype == "float64":
            expected = df.astype("float64")  # 将 df 的数据类型转换为 float64，并赋值给 expected
        else:
            expected = df.astype("object")  # 将 df 的数据类型转换为 object，并赋值给 expected

    if needs_update:
        # GH#40893 changed the expected here to retain dependence on empty
        expected = expected.astype(object)  # 将 expected 的数据类型转换为 object

    tm.assert_frame_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 的内容是否相等


@pytest.mark.parametrize("df_dtype", ["float64", "int64", "datetime64[ns]"])
@pytest.mark.parametrize("empty_dtype", [None, "float64", "object"])
def test_concat_ignore_all_na_object_float(empty_dtype, df_dtype):
    df = DataFrame({"foo": [1, 2], "bar": [1, 2]}, dtype=df_dtype)  # 创建一个指定数据类型的 Pandas DataFrame 对象 df
    empty = DataFrame({"foo": [np.nan], "bar": [np.nan]}, dtype=empty_dtype)  # 创建一个指定数据类型的 Pandas DataFrame 对象 empty

    if df_dtype == "int64":
        # TODO what exact behaviour do we want for integer eventually?
        if empty_dtype == "object":
            df_dtype = "object"
        else:
            df_dtype = "float64"

    needs_update = False
    if empty_dtype != df_dtype and empty_dtype is not None:
        needs_update = True
    elif df_dtype == "datetime64[ns]":
        needs_update = True

    result = concat([empty, df], ignore_index=True)  # 将 empty 和 df 进行连接操作，并忽略索引，生成一个新的 DataFrame 对象 result

    expected = DataFrame({"foo": [np.nan, 1, 2], "bar": [np.nan, 1, 2]}, dtype=df_dtype)  # 创建一个指定数据类型的 Pandas DataFrame 对象 expected
    if needs_update:
        # GH#40893 changed the expected here to retain dependence on empty
        expected = expected.astype(object)  # 将 expected 的数据类型转换为 object
        expected.iloc[0] = np.nan  # 设置 expected 的第一行数据为 NaN
    # 使用测试框架中的函数来比较两个数据帧（DataFrame）是否相等
    tm.assert_frame_equal(result, expected)
# 测试拼接函数对忽略空值重新索引的处理
def test_concat_ignore_empty_from_reindex():
    # 创建第一个数据帧 df1，包含整数列 'a' 和日期列 'b'
    df1 = DataFrame({"a": [1], "b": [pd.Timestamp("2012-01-01")]})
    # 创建第二个数据帧 df2，包含整数列 'a'
    df2 = DataFrame({"a": [2]})

    # 使用 df1 的列重新索引 df2，使其与 df1 的列对齐
    aligned = df2.reindex(columns=df1.columns)

    # 将 df1 和重新索引后的 df2 拼接在一起，忽略索引，生成结果数据帧 result
    result = concat([df1, aligned], ignore_index=True)

    # 创建预期的结果数据帧 expected，包含整数列 'a' 和日期列 'b'，'b' 列中的第二行为 NaN
    expected = DataFrame(
        {
            "a": [1, 2],
            "b": pd.array([pd.Timestamp("2012-01-01"), np.nan], dtype=object),
        },
        dtype=object,
    )
    # 将 'a' 列的数据类型转换为 int64
    expected["a"] = expected["a"].astype("int64")
    # 使用测试工具函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试拼接函数对键长度不匹配的处理
def test_concat_mismatched_keys_length():
    # 创建 Series，包含范围为 [0, 4] 的整数
    ser = Series(range(5))
    # 创建包含四个元素的 Series 列表 sers
    sers = [ser + n for n in range(4)]
    # 创建键列表 keys
    keys = ["A", "B", "C"]

    # 设置预期的异常消息
    msg = r"The length of the keys"

    # 使用 pytest 断言，检查在不同轴上拼接 sers 时是否会抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        concat(sers, keys=keys, axis=1)
    with pytest.raises(ValueError, match=msg):
        concat(sers, keys=keys, axis=0)
    with pytest.raises(ValueError, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=1)
    with pytest.raises(ValueError, match=msg):
        concat((x for x in sers), keys=(y for y in keys), axis=0)


# 测试拼接函数在多级索引和分类数据处理时的行为
def test_concat_multiindex_with_category():
    # 创建第一个数据帧 df1，包含两个分类列 'c1' 和 'c2'，以及整数列 'i2'
    df1 = DataFrame(
        {
            "c1": Series(list("abc"), dtype="category"),
            "c2": Series(list("eee"), dtype="category"),
            "i2": Series([1, 2, 3]),
        }
    )
    # 将 'c1' 和 'c2' 列设置为索引
    df1 = df1.set_index(["c1", "c2"])

    # 创建第二个数据帧 df2，与 df1 结构相同
    df2 = DataFrame(
        {
            "c1": Series(list("abc"), dtype="category"),
            "c2": Series(list("eee"), dtype="category"),
            "i2": Series([4, 5, 6]),
        }
    )
    # 将 'c1' 和 'c2' 列设置为索引
    df2 = df2.set_index(["c1", "c2"])

    # 使用拼接函数将 df1 和 df2 拼接在一起，生成结果数据帧 result
    result = concat([df1, df2])

    # 创建预期的结果数据帧 expected，包含两个分类列 'c1' 和 'c2'，以及整数列 'i2'
    expected = DataFrame(
        {
            "c1": Series(list("abcabc"), dtype="category"),
            "c2": Series(list("eeeeee"), dtype="category"),
            "i2": Series([1, 2, 3, 4, 5, 6]),
        }
    )
    # 将 'c1' 和 'c2' 列设置为索引
    expected = expected.set_index(["c1", "c2"])

    # 使用测试工具函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试拼接函数对类型转换的处理（字符串和整数）
def test_concat_ea_upcast():
    # 创建包含一个字符串的数据帧 df1
    df1 = DataFrame(["a"], dtype="string")
    # 创建包含一个整数的数据帧 df2
    df2 = DataFrame([1], dtype="Int64")

    # 使用拼接函数将 df1 和 df2 拼接在一起，生成结果数据帧 result
    result = concat([df1, df2])

    # 创建预期的结果数据帧 expected，包含一个字符串和一个整数
    expected = DataFrame(["a", 1], index=[0, 0])

    # 使用测试工具函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试拼接函数对时区时间戳和空值处理的行为
def test_concat_none_with_timezone_timestamp():
    # 创建包含一个空值的数据帧 df1
    df1 = DataFrame([{"A": None}])
    # 创建包含一个带时区的时间戳的数据帧 df2
    df2 = DataFrame([{"A": pd.Timestamp("1990-12-20 00:00:00+00:00")}])

    # 使用拼接函数将 df1 和 df2 拼接在一起，忽略索引，生成结果数据帧 result
    result = concat([df1, df2], ignore_index=True)

    # 创建预期的结果数据帧 expected，包含一列 'A'，其中一个值为 None，另一个为带时区的时间戳
    expected = DataFrame(
        {"A": [None, pd.Timestamp("1990-12-20 00:00:00+00:00")]}, dtype=object
    )

    # 使用测试工具函数检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 测试拼接函数在处理 Series 和 DataFrame 混合时返回列为 RangeIndex 的情况
def test_concat_with_series_and_frame_returns_rangeindex_columns():
    # 创建包含一个元素的 Series ser
    ser = Series([0])
    # 创建包含两个元素的数据帧 df
    df = DataFrame([1, 2])

    # 使用拼接函数将 ser 和 df 拼接在一起，生成结果数据帧 result
    result = concat([ser, df])

    # 创建预期的结果数据帧 expected，包含一个列为 RangeIndex 的列
    expected = DataFrame([0, 1, 2], index=[0, 0, 1])

    # 使用测试工具函数检查 result 和 expected 是否相等，并且检查列类型是否匹配
    tm.assert_frame_equal(result, expected, check_column_type=True)
```