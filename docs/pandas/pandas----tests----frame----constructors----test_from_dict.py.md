# `D:\src\scipysrc\pandas\pandas\tests\frame\constructors\test_from_dict.py`

```
from collections import OrderedDict  # 导入有序字典模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架

from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 内部配置模块中的字符串类型判断函数

from pandas import (  # 导入 pandas 库中的多个类和函数
    DataFrame,       # 数据框类
    Index,           # 索引类
    MultiIndex,      # 多重索引类
    RangeIndex,      # 范围索引类
    Series,          # 系列类
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestFromDict:
    # 注意：这些测试方法特定用于 from_dict 方法，而不是传递字典给 DataFrame.__init__

    def test_constructor_list_of_odicts(self):
        data = [  # 定义测试数据，包含多个有序字典
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4], ["d", 6]]),
            OrderedDict([["a", 1.5], ["b", 3], ["d", 6]]),
            OrderedDict([["a", 1.5], ["d", 6]]),
            OrderedDict(),  # 空有序字典
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4]]),
            OrderedDict([["b", 3], ["c", 4], ["d", 6]]),
        ]

        result = DataFrame(data)  # 使用 data 创建数据框
        expected = DataFrame.from_dict(  # 从字典生成数据框，指定字典的键作为索引
            dict(zip(range(len(data)), data)), orient="index"
        )
        tm.assert_frame_equal(result, expected.reindex(result.index))  # 断言结果与预期相等，重建索引后比较

    def test_constructor_single_row(self):
        data = [OrderedDict([["a", 1.5], ["b", 3], ["c", 4], ["d", 6]])]  # 单行数据

        result = DataFrame(data)  # 使用 data 创建数据框
        expected = DataFrame.from_dict(dict(zip([0], data)), orient="index").reindex(
            result.index
        )  # 从字典生成数据框，指定字典的键作为索引，重建索引后与结果比较
        tm.assert_frame_equal(result, expected)

    @pytest.mark.skipif(
        using_pyarrow_string_dtype(), reason="columns inferring logic broken"
    )
    # 定义测试方法，测试DataFrame的构造函数以及不同参数的使用情况
    def test_constructor_list_of_series(self):
        # 定义测试数据，包含两个有序字典的列表
        data = [
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 4.0]]),
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 6.0]]),
        ]
        # 将有序字典列表转换为有序字典，键为["x", "y"]
        sdict = OrderedDict(zip(["x", "y"], data))
        # 创建索引对象，包含标签["a", "b", "c"]
        idx = Index(["a", "b", "c"])

        # 第一种情况：所有序列都有名称
        data2 = [
            Series([1.5, 3, 4], idx, dtype="O", name="x"),
            Series([1.5, 3, 6], idx, name="y"),
        ]
        # 使用Series对象列表创建DataFrame对象
        result = DataFrame(data2)
        # 从有序字典sdict中创建预期的DataFrame对象
        expected = DataFrame.from_dict(sdict, orient="index")
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(result, expected)

        # 第二种情况：部分序列没有名称
        data2 = [
            Series([1.5, 3, 4], idx, dtype="O", name="x"),
            Series([1.5, 3, 6], idx),
        ]
        # 使用Series对象列表创建DataFrame对象
        result = DataFrame(data2)
        # 更新有序字典sdict中的键，其中一个序列未命名
        sdict = OrderedDict(zip(["x", "Unnamed 0"], data))
        # 从有序字典sdict中创建预期的DataFrame对象
        expected = DataFrame.from_dict(sdict, orient="index")
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(result, expected)

        # 第三种情况：所有序列都没有名称
        # 更新测试数据，包含多个无序字典
        data = [
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4], ["d", 6]]),
            OrderedDict([["a", 1.5], ["b", 3], ["d", 6]]),
            OrderedDict([["a", 1.5], ["d", 6]]),
            OrderedDict(),
            OrderedDict([["a", 1.5], ["b", 3], ["c", 4]]),
            OrderedDict([["b", 3], ["c", 4], ["d", 6]]),
        ]
        # 使用无序字典列表创建Series对象列表
        data = [Series(d) for d in data]
        # 使用Series对象列表创建DataFrame对象
        result = DataFrame(data)
        # 更新有序字典sdict中的键，与数据长度相对应
        sdict = OrderedDict(zip(range(len(data)), data))
        # 从有序字典sdict中创建预期的DataFrame对象，并重新索引
        expected = DataFrame.from_dict(sdict, orient="index")
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(result, expected.reindex(result.index))

        # 使用不同的索引创建DataFrame对象
        result2 = DataFrame(data, index=np.arange(6, dtype=np.int64))
        # 使用测试工具函数验证结果和result2对象相等
        tm.assert_frame_equal(result, result2)

        # 使用单个空Series对象创建DataFrame对象
        result = DataFrame([Series(dtype=object)])
        # 创建一个只有索引的预期DataFrame对象
        expected = DataFrame(index=[0])
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(result, expected)

        # 更新测试数据，包含两个有序字典的列表
        data = [
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 4.0]]),
            OrderedDict([["a", 1.5], ["b", 3.0], ["c", 6.0]]),
        ]
        # 将有序字典列表转换为有序字典，键为数据的长度范围
        sdict = OrderedDict(zip(range(len(data)), data))

        # 创建索引对象，包含标签["a", "b", "c"]
        idx = Index(["a", "b", "c"])
        # 使用Series对象列表创建DataFrame对象
        data2 = [Series([1.5, 3, 4], idx, dtype="O"), Series([1.5, 3, 6], idx)]
        result = DataFrame(data2)
        # 从有序字典sdict中创建预期的DataFrame对象
        expected = DataFrame.from_dict(sdict, orient="index")
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，测试DataFrame的构造函数中orient参数的使用情况
    def test_constructor_orient(self, float_string_frame):
        # 获取转置后的数据字典，并从中创建DataFrame对象
        data_dict = float_string_frame.T._series
        recons = DataFrame.from_dict(data_dict, orient="index")
        # 重新索引预期的DataFrame对象
        expected = float_string_frame.reindex(index=recons.index)
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(recons, expected)

        # 定义字典对象a，包含不同标签的序列
        a = {"hi": [32, 3, 3], "there": [3, 5, 3]}
        # 从字典a创建DataFrame对象，并转置后重新索引
        rs = DataFrame.from_dict(a, orient="index")
        xp = DataFrame.from_dict(a).T.reindex(list(a.keys()))
        # 使用测试工具函数验证结果和预期DataFrame对象相等
        tm.assert_frame_equal(rs, xp)
    def test_constructor_from_ordered_dict(self):
        # GH#8425
        # 创建一个有序字典 a，包含三个键值对，每个值是一个有序字典，用于构建测试数据
        a = OrderedDict(
            [
                ("one", OrderedDict([("col_a", "foo1"), ("col_b", "bar1")])),
                ("two", OrderedDict([("col_a", "foo2"), ("col_b", "bar2")])),
                ("three", OrderedDict([("col_a", "foo3"), ("col_b", "bar3")])),
            ]
        )
        # 期望的结果是根据 a 构建的 DataFrame，转置后的形式
        expected = DataFrame.from_dict(a, orient="columns").T
        # 使用 orient="index" 从 a 构建 DataFrame
        result = DataFrame.from_dict(a, orient="index")
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_from_dict_columns_parameter(self):
        # GH#18529
        # 测试新的 columns 参数，用于简化 orient='index' 情况下的 from_dict(...) 的复制
        # 从有序字典构建 DataFrame，指定 orient="index" 和 columns=["one", "two"]
        result = DataFrame.from_dict(
            OrderedDict([("A", [1, 2]), ("B", [4, 5])]),
            orient="index",
            columns=["one", "two"],
        )
        # 期望的结果 DataFrame，包含指定的索引和列名
        expected = DataFrame([[1, 2], [4, 5]], index=["A", "B"], columns=["one", "two"])
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 准备错误信息
        msg = "cannot use columns parameter with orient='columns'"
        # 预期抛出 ValueError 异常，且异常消息与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_dict(
                {"A": [1, 2], "B": [4, 5]},
                orient="columns",
                columns=["one", "two"],
            )
        # 预期抛出 ValueError 异常，且异常消息与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            DataFrame.from_dict({"A": [1, 2], "B": [4, 5]}, columns=["one", "two"])

    @pytest.mark.parametrize(
        "data_dict, orient, expected",
        [
            # 测试空字典的情况，orient="index"，期望结果是 RangeIndex(0)
            ({}, "index", RangeIndex(0)),
            (
                # 测试包含 tuple 键的字典列表，orient="columns"，期望结果是特定的 Index 对象
                [{("a",): 1}, {("a",): 2}],
                "columns",
                Index([("a",)], tupleize_cols=False),
            ),
            (
                # 测试包含有序字典的列表，orient="columns"，期望结果是特定的 Index 对象
                [OrderedDict([(("a",), 1), (("b",), 2)])],
                "columns",
                Index([("a",), ("b",)], tupleize_cols=False),
            ),
            (
                # 测试包含多元组键的字典，orient="columns"，期望结果是特定的 Index 对象
                [{("a", "b"): 1}],
                "columns",
                Index([("a", "b")], tupleize_cols=False),
            ),
        ],
    )
    def test_constructor_from_dict_tuples(self, data_dict, orient, expected):
        # GH#16769
        # 从字典 data_dict 使用 orient 构建 DataFrame
        df = DataFrame.from_dict(data_dict, orient)
        # 获取 DataFrame 的列索引，与期望的 Index 对象进行比较
        result = df.columns
        # 断言两个 Index 对象是否相等
        tm.assert_index_equal(result, expected)

    def test_frame_dict_constructor_empty_series(self):
        # 创建一个 Series 对象 s1
        s1 = Series(
            [1, 2, 3, 4], index=MultiIndex.from_tuples([(1, 2), (1, 3), (2, 2), (2, 4)])
        )
        # 创建一个 Series 对象 s2
        s2 = Series(
            [1, 2, 3, 4], index=MultiIndex.from_tuples([(1, 2), (1, 3), (3, 2), (3, 4)])
        )
        # 创建一个空对象 Series 对象 s3
        s3 = Series(dtype=object)

        # 创建一个 DataFrame 对象，包含三列，每列对应一个 Series 对象
        DataFrame({"foo": s1, "bar": s2, "baz": s3})
        # 从字典构建 DataFrame 对象，包含三列，每列对应一个 Series 对象
        DataFrame.from_dict({"foo": s1, "baz": s3, "bar": s2})

    def test_from_dict_scalars_requires_index(self):
        # 准备错误信息
        msg = "If using all scalar values, you must pass an index"
        # 预期抛出 ValueError 异常，且异常消息与 msg 匹配
        with pytest.raises(ValueError, match=msg):
            # 从有序字典构建 DataFrame，其中包含重复键
            DataFrame.from_dict(OrderedDict([("b", 8), ("a", 5), ("a", 6)]))
    # 定义一个测试方法，用于测试从字典创建 DataFrame 时 orient 参数为非法值的情况
    def test_from_dict_orient_invalid(self):
        # 设置错误消息，用于匹配 pytest 抛出的 ValueError 异常
        msg = (
            "Expected 'index', 'columns' or 'tight' for orient parameter. "
            "Got 'abc' instead"
        )
        # 使用 pytest 的 raises 方法验证是否会抛出预期的 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 DataFrame.from_dict 方法，传入一个字典和非法的 orient 参数值 "abc"
            DataFrame.from_dict({"foo": 1, "baz": 3, "bar": 2}, orient="abc")

    # 定义一个测试方法，用于测试从字典创建 DataFrame 时 orient 参数为 "columns" 的情况
    def test_from_dict_order_with_single_column(self):
        # 定义一个包含嵌套字典的数据结构
        data = {
            "alpha": {
                "value2": 123,
                "value1": 532,
                "animal": 222,
                "plant": False,
                "name": "test",
            }
        }
        # 调用 DataFrame.from_dict 方法，传入数据字典和 orient 参数值 "columns"
        result = DataFrame.from_dict(
            data,
            orient="columns",
        )
        # 定义预期的 DataFrame 对象，包含特定的数据和索引
        expected = DataFrame(
            [[123], [532], [222], [False], ["test"]],
            index=["value2", "value1", "animal", "plant", "name"],
            columns=["alpha"],
        )
        # 使用 tm.assert_frame_equal 方法验证 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```