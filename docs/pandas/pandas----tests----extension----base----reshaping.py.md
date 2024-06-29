# `D:\src\scipysrc\pandas\pandas\tests\extension\base\reshaping.py`

```
# 导入所需的库
import itertools  # 导入 itertools 库，用于生成迭代器的函数

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.api.extensions import ExtensionArray  # 导入 Pandas 扩展数组
from pandas.core.internals.blocks import EABackedBlock  # 导入 Pandas 内部块相关模块


class BaseReshapingTests:
    """Tests for reshaping and concatenation."""

    @pytest.mark.parametrize("in_frame", [True, False])
    def test_concat(self, data, in_frame):
        # 将数据包装为 Pandas Series
        wrapped = pd.Series(data)
        if in_frame:
            wrapped = pd.DataFrame(wrapped)  # 如果 in_frame 为 True，则转换为 DataFrame
        # 进行 concat 操作，将 wrapped 与自身连接，忽略索引
        result = pd.concat([wrapped, wrapped], ignore_index=True)

        assert len(result) == len(data) * 2  # 断言结果的长度为原数据长度的两倍

        if in_frame:
            dtype = result.dtypes[0]  # 如果 in_frame 为 True，则获取第一个列的数据类型
        else:
            dtype = result.dtype  # 否则获取结果的数据类型

        assert dtype == data.dtype  # 断言结果的数据类型与原数据相同
        if hasattr(result._mgr, "blocks"):
            assert isinstance(result._mgr.blocks[0], EABackedBlock)  # 断言结果的第一个块是 EABackedBlock 类型
        assert isinstance(result._mgr.blocks[0].values, ExtensionArray)  # 断言结果的第一个块的值是 ExtensionArray 类型

    @pytest.mark.parametrize("in_frame", [True, False])
    def test_concat_all_na_block(self, data_missing, in_frame):
        # 创建一个有效块和一个包含 NA 的块
        valid_block = pd.Series(data_missing.take([1, 1]), index=[0, 1])
        na_block = pd.Series(data_missing.take([0, 0]), index=[2, 3])
        if in_frame:
            valid_block = pd.DataFrame({"a": valid_block})  # 如果 in_frame 为 True，则将有效块转换为 DataFrame
            na_block = pd.DataFrame({"a": na_block})  # 如果 in_frame 为 True，则将 NA 块转换为 DataFrame
        # 对有效块和 NA 块进行 concat 操作
        result = pd.concat([valid_block, na_block])
        if in_frame:
            expected = pd.DataFrame({"a": data_missing.take([1, 1, 0, 0])})  # 创建预期的 DataFrame 结果
            tm.assert_frame_equal(result, expected)  # 断言结果与预期的 DataFrame 相等
        else:
            expected = pd.Series(data_missing.take([1, 1, 0, 0]))  # 创建预期的 Series 结果
            tm.assert_series_equal(result, expected)  # 断言结果与预期的 Series 相等

    def test_concat_mixed_dtypes(self, data):
        # 创建三个不同数据类型的 DataFrame
        df1 = pd.DataFrame({"A": data[:3]})
        df2 = pd.DataFrame({"A": [1, 2, 3]})
        df3 = pd.DataFrame({"A": ["a", "b", "c"]}).astype("category")
        dfs = [df1, df2, df3]

        # 对 DataFrame 进行 concat 操作
        result = pd.concat(dfs)
        expected = pd.concat([x.astype(object) for x in dfs])  # 预期的结果是将每个 DataFrame 转换为 object 类型后进行 concat
        tm.assert_frame_equal(result, expected)  # 断言 DataFrame 的结果与预期的 DataFrame 相等

        # 对 Series 进行 concat 操作
        result = pd.concat([x["A"] for x in dfs])
        expected = pd.concat([x["A"].astype(object) for x in dfs])  # 预期的结果是将每个 Series 转换为 object 类型后进行 concat
        tm.assert_series_equal(result, expected)  # 断言 Series 的结果与预期的 Series 相等

        # 对混合类型的 DataFrame 进行 concat 操作
        result = pd.concat([df1, df2.astype(object)])
        expected = pd.concat([df1.astype("object"), df2.astype("object")])  # 预期的结果是将每个 DataFrame 转换为 object 类型后进行 concat
        tm.assert_frame_equal(result, expected)  # 断言 DataFrame 的结果与预期的 DataFrame 相等

        result = pd.concat([df1["A"], df2["A"].astype(object)])
        expected = pd.concat([df1["A"].astype("object"), df2["A"].astype("object")])  # 预期的结果是将每个 Series 转换为 object 类型后进行 concat
        tm.assert_series_equal(result, expected)  # 断言 Series 的结果与预期的 Series 相等
    def test_concat_columns(self, data, na_value):
        # 创建第一个DataFrame，列名为"A"，数据从参数data中取前三个元素
        df1 = pd.DataFrame({"A": data[:3]})
        # 创建第二个DataFrame，列名为"B"，数据为[1, 2, 3]
        df2 = pd.DataFrame({"B": [1, 2, 3]})

        # 创建期望的DataFrame，合并df1和df2，期望结果包含列"A"和列"B"
        expected = pd.DataFrame({"A": data[:3], "B": [1, 2, 3]})
        # 使用pd.concat将df1和df2按列(axis=1)合并，保存在result中
        result = pd.concat([df1, df2], axis=1)
        # 使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)
        
        # 再次使用pd.concat，这次合并df1["A"]和df2["B"]，按列(axis=1)合并
        result = pd.concat([df1["A"], df2["B"]], axis=1)
        # 再次使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)

        # 非对齐情况下的测试
        # 重新定义df2，此次指定了索引为[1, 2, 3]
        df2 = pd.DataFrame({"B": [1, 2, 3]}, index=[1, 2, 3])
        # 创建期望的DataFrame，合并df1和df2，期望结果包含列"A"和列"B"，对应索引位置填充NaN
        expected = pd.DataFrame(
            {
                "A": data._from_sequence(list(data[:3]) + [na_value], dtype=data.dtype),
                "B": [np.nan, 1, 2, 3],
            }
        )

        # 使用pd.concat将df1和df2按列(axis=1)合并，保存在result中
        result = pd.concat([df1, df2], axis=1)
        # 使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)
        
        # 再次使用pd.concat，这次合并df1["A"]和df2["B"]，按列(axis=1)合并
        result = pd.concat([df1["A"], df2["B"]], axis=1)
        # 再次使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)

    def test_concat_extension_arrays_copy_false(self, data, na_value):
        # GH 20756
        # 创建第一个DataFrame，列名为"A"，数据为参数data中的前三个元素
        df1 = pd.DataFrame({"A": data[:3]})
        # 创建第二个DataFrame，列名为"B"，数据为参数data中第3到第6个元素
        df2 = pd.DataFrame({"B": data[3:7]})
        # 创建期望的DataFrame，合并df1和df2，期望结果包含列"A"和列"B"
        expected = pd.DataFrame(
            {
                "A": data._from_sequence(list(data[:3]) + [na_value], dtype=data.dtype),
                "B": data[3:7],
            }
        )
        # 使用pd.concat将df1和df2按列(axis=1)合并，保存在result中
        result = pd.concat([df1, df2], axis=1)
        # 使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)

    def test_concat_with_reindex(self, data):
        # GH-33027
        # 创建DataFrame a，列名为"a"，数据为参数data的前五个元素
        a = pd.DataFrame({"a": data[:5]})
        # 创建DataFrame b，列名为"b"，数据为参数data的前五个元素
        b = pd.DataFrame({"b": data[:5]})
        # 使用pd.concat将a和b合并，并重新索引(ignore_index=True)
        result = pd.concat([a, b], ignore_index=True)
        # 创建期望的DataFrame，包含列"a"和"b"，重新索引后的数据从参数data中获取
        expected = pd.DataFrame(
            {
                "a": data.take(list(range(5)) + ([-1] * 5), allow_fill=True),
                "b": data.take(([-1] * 5) + list(range(5)), allow_fill=True),
            }
        )
        # 使用tm.assert_frame_equal比较result和expected，确认它们相等
        tm.assert_frame_equal(result, expected)

    def test_align(self, data, na_value):
        # 从参数data中取前三个元素，存入变量a
        a = data[:3]
        # 从参数data中取第2到第5个元素，存入变量b
        b = data[2:5]
        # 使用pd.Series.align方法对两个Series进行对齐
        r1, r2 = pd.Series(a).align(pd.Series(b, index=[1, 2, 3]))

        # 假设构造函数可以接受一个标量值列表作为参数类型
        # 创建期望的Series，数据从参数data中获取，最后一个位置填充na_value
        e1 = pd.Series(data._from_sequence(list(a) + [na_value], dtype=data.dtype))
        e2 = pd.Series(data._from_sequence([na_value] + list(b), dtype=data.dtype))
        # 使用tm.assert_series_equal比较r1和e1，确认它们相等
        tm.assert_series_equal(r1, e1)
        # 使用tm.assert_series_equal比较r2和e2，确认它们相等
        tm.assert_series_equal(r2, e2)

    def test_align_frame(self, data, na_value):
        # 从参数data中取前三个元素，存入变量a
        a = data[:3]
        # 从参数data中取第2到第5个元素，存入变量b
        b = data[2:5]
        # 使用pd.DataFrame.align方法对两个DataFrame进行对齐
        r1, r2 = pd.DataFrame({"A": a}).align(pd.DataFrame({"A": b}, index=[1, 2, 3]))

        # 假设构造函数可以接受一个标量值列表作为参数类型
        # 创建期望的DataFrame，数据从参数data中获取，最后一个位置填充na_value
        e1 = pd.DataFrame(
            {"A": data._from_sequence(list(a) + [na_value], dtype=data.dtype)}
        )
        e2 = pd.DataFrame(
            {"A": data._from_sequence([na_value] + list(b), dtype=data.dtype)}
        )
        # 使用tm.assert_frame_equal比较r1和e1，确认它们相等
        tm.assert_frame_equal(r1, e1)
        # 使用tm.assert_frame_equal比较r2和e2，确认它们相等
        tm.assert_frame_equal(r2, e2)
    # 测试函数：将数据系列与数据框进行对齐
    def test_align_series_frame(self, data, na_value):
        # GitHub issue链接，详细描述问题及讨论
        # https://github.com/pandas-dev/pandas/issues/20576
        # 创建数据系列，命名为"a"
        ser = pd.Series(data, name="a")
        # 创建数据框，包含一个列"col"，列长度为数据系列长度加一
        df = pd.DataFrame({"col": np.arange(len(ser) + 1)})
        # 将数据系列与数据框进行对齐操作
        r1, r2 = ser.align(df)

        # 创建预期的数据系列e1，使用扩展数据序列来填充，最后一个元素使用na_value填充
        e1 = pd.Series(
            data._from_sequence(list(data) + [na_value], dtype=data.dtype),
            name=ser.name,
        )

        # 断言数据系列r1与预期e1相等
        tm.assert_series_equal(r1, e1)
        # 断言数据框r2与预期df相等
        tm.assert_frame_equal(r2, df)

    # 测试函数：在数据框中扩展常规数据列与扩展数据列
    def test_set_frame_expand_regular_with_extension(self, data):
        # 创建数据框，列"A"的值为常数1，列"B"的值为数据data
        df = pd.DataFrame({"A": [1] * len(data)})
        df["B"] = data
        # 创建预期的数据框，列"A"的值为常数1，列"B"的值为数据data
        expected = pd.DataFrame({"A": [1] * len(data), "B": data})
        # 断言数据框df与预期expected相等
        tm.assert_frame_equal(df, expected)

    # 测试函数：在数据框中扩展扩展数据列与常规数据列
    def test_set_frame_expand_extension_with_regular(self, data):
        # 创建数据框，列"A"的值为数据data，列"B"的值为常数1
        df = pd.DataFrame({"A": data})
        df["B"] = [1] * len(data)
        # 创建预期的数据框，列"A"的值为数据data，列"B"的值为常数1
        expected = pd.DataFrame({"A": data, "B": [1] * len(data)})
        # 断言数据框df与预期expected相等
        tm.assert_frame_equal(df, expected)

    # 测试函数：在数据框中覆盖对象类型的列
    def test_set_frame_overwrite_object(self, data):
        # GitHub issue链接，详细描述问题及讨论
        # https://github.com/pandas-dev/pandas/issues/20555
        # 创建数据框，列"A"的值为常数1，数据类型为对象类型
        df = pd.DataFrame({"A": [1] * len(data)}, dtype=object)
        # 将数据列"A"覆盖为data
        df["A"] = data
        # 断言数据框列"A"的数据类型与data相同
        assert df.dtypes["A"] == data.dtype

    # 测试函数：合并数据框
    def test_merge(self, data, na_value):
        # GitHub issue链接，详细描述问题及讨论
        # GH-20743
        # 创建第一个数据框df1，包含列"ext"、"int1"和"key"
        df1 = pd.DataFrame({"ext": data[:3], "int1": [1, 2, 3], "key": [0, 1, 2]})
        # 创建第二个数据框df2，包含列"int2"和"key"
        df2 = pd.DataFrame({"int2": [1, 2, 3, 4], "key": [0, 0, 1, 3]})

        # 执行内连接合并操作
        res = pd.merge(df1, df2)
        # 创建预期的数据框exp，包含"ext"、"int1"、"key"和"int2"列，"ext"使用data的前三个元素填充
        exp = pd.DataFrame(
            {
                "int1": [1, 1, 2],
                "int2": [1, 2, 3],
                "key": [0, 0, 1],
                "ext": data._from_sequence(
                    [data[0], data[0], data[1]], dtype=data.dtype
                ),
            }
        )
        # 断言数据框res与预期exp的子集["ext", "int1", "key", "int2"]相等
        tm.assert_frame_equal(res, exp[["ext", "int1", "key", "int2"]])

        # 执行外连接合并操作
        res = pd.merge(df1, df2, how="outer")
        # 创建预期的数据框exp，包含"ext"、"int1"、"key"和"int2"列，"ext"使用data的前三个元素填充，最后一个元素使用na_value填充
        exp = pd.DataFrame(
            {
                "int1": [1, 1, 2, 3, np.nan],
                "int2": [1, 2, 3, np.nan, 4],
                "key": [0, 0, 1, 2, 3],
                "ext": data._from_sequence(
                    [data[0], data[0], data[1], data[2], na_value], dtype=data.dtype
                ),
            }
        )
        # 断言数据框res与预期exp的子集["ext", "int1", "key", "int2"]相等
        tm.assert_frame_equal(res, exp[["ext", "int1", "key", "int2"]])

    # 测试函数：在扩展数组上执行合并操作
    def test_merge_on_extension_array(self, data):
        # GH 23020
        # 获取数据序列data的前两个元素a和b
        a, b = data[:2]
        # 使用数据序列的类型创建关键字key
        key = type(data)._from_sequence([a, b], dtype=data.dtype)

        # 创建数据框df，包含列"key"和"val"
        df = pd.DataFrame({"key": key, "val": [1, 2]})
        # 执行基于"key"列的合并操作
        result = pd.merge(df, df, on="key")
        # 创建预期的数据框expected，包含"key"、"val_x"和"val_y"列
        expected = pd.DataFrame({"key": key, "val_x": [1, 2], "val_y": [1, 2]})
        # 断言数据框result与预期expected相等
        tm.assert_frame_equal(result, expected)

        # 按照特定顺序进行合并操作
        result = pd.merge(df.iloc[[1, 0]], df, on="key")
        # 重设索引并重新排序预期的数据框expected
        expected = expected.iloc[[1, 0]].reset_index(drop=True)
        # 断言数据框result与预期expected相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试处理扩展数组中的重复项合并
    def test_merge_on_extension_array_duplicates(self, data):
        # GH 23020
        # 从数据中取出前两个元素
        a, b = data[:2]
        # 从数据类型中创建一个序列，包含三个元素，其中有重复
        key = type(data)._from_sequence([a, b, a], dtype=data.dtype)
        # 创建第一个数据帧，包含列 "key" 和 "val"，值为 [1, 2, 3]
        df1 = pd.DataFrame({"key": key, "val": [1, 2, 3]})
        # 创建第二个数据帧，包含列 "key" 和 "val"，值为 [1, 2, 3]
        df2 = pd.DataFrame({"key": key, "val": [1, 2, 3]})

        # 使用 "key" 列进行数据帧合并
        result = pd.merge(df1, df2, on="key")
        # 创建预期结果的数据帧，包含 "key" 列，以及 "val_x" 和 "val_y" 列
        expected = pd.DataFrame(
            {
                "key": key.take([0, 0, 1, 2, 2]),
                "val_x": [1, 1, 2, 3, 3],
                "val_y": [1, 3, 2, 1, 3],
            }
        )
        # 断言结果数据帧与预期数据帧相等
        tm.assert_frame_equal(result, expected)

    # 标记测试，忽略警告信息："The previous implementation of stack is deprecated"
    @pytest.mark.filterwarnings(
        "ignore:The previous implementation of stack is deprecated"
    )
    # 参数化测试，columns 参数为 ["A", "B"] 和多级索引
    @pytest.mark.parametrize(
        "columns",
        [
            ["A", "B"],
            pd.MultiIndex.from_tuples(
                [("A", "a"), ("A", "b")], names=["outer", "inner"]
            ),
        ],
    )
    # 参数化测试，future_stack 参数为 True 和 False
    @pytest.mark.parametrize("future_stack", [True, False])
    # 定义一个测试方法，用于测试数据帧的堆叠操作
    def test_stack(self, data, columns, future_stack):
        # 创建一个数据帧，包含列 "A" 和 "B"，每列取前五个数据
        df = pd.DataFrame({"A": data[:5], "B": data[:5]})
        # 重置数据帧的列名为参数化传入的列名
        df.columns = columns
        # 对数据帧执行堆叠操作，根据 future_stack 参数选择不同行为
        result = df.stack(future_stack=future_stack)
        # 创建预期结果的堆叠数据帧，保留对象类型
        expected = df.astype(object).stack(future_stack=future_stack)
        # 对预期结果再次执行类型转换，确保类型匹配
        expected = expected.astype(object)

        # 如果预期结果是一个 Series 对象，则断言结果的数据类型与第一列的数据类型相等
        if isinstance(expected, pd.Series):
            assert result.dtype == df.iloc[:, 0].dtype
        else:
            # 否则，断言所有结果列的数据类型与第一列的数据类型相等
            assert all(result.dtypes == df.iloc[:, 0].dtype)

        # 将结果数据帧转换为对象类型，再次断言结果与预期结果相等
        result = result.astype(object)
        tm.assert_equal(result, expected)

    # 参数化测试，index 参数为不同的多级索引类型
    @pytest.mark.parametrize(
        "index",
        [
            # 两级均匀多级索引
            pd.MultiIndex.from_product(([["A", "B"], ["a", "b"]]), names=["a", "b"]),
            # 非均匀多级索引
            pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "b")]),
            # 三级非均匀多级索引
            pd.MultiIndex.from_product([("A", "B"), ("a", "b", "c"), (0, 1, 2)]),
            pd.MultiIndex.from_tuples(
                [
                    ("A", "a", 1),
                    ("A", "b", 0),
                    ("A", "a", 0),
                    ("B", "a", 0),
                    ("B", "c", 1),
                ]
            ),
        ],
    )
    # 参数化测试，obj 参数为 "series" 和 "frame"
    @pytest.mark.parametrize("obj", ["series", "frame"])
    # 定义测试函数 test_unstack，用于测试数据的展开操作
    def test_unstack(self, data, index, obj):
        # 截取数据，保证长度与索引相同
        data = data[: len(index)]
        # 根据对象类型创建 Series 或 DataFrame
        if obj == "series":
            ser = pd.Series(data, index=index)
        else:
            ser = pd.DataFrame({"A": data, "B": data}, index=index)

        # 获取索引的层级数
        n = index.nlevels
        # 生成层级的列表 [0, 1, 2]
        levels = list(range(n))
        
        # 生成所有可能的层级组合
        # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        combinations = itertools.chain.from_iterable(
            itertools.permutations(levels, i) for i in range(1, n)
        )

        # 遍历所有组合
        for level in combinations:
            # 对数据进行展开操作，使用给定的层级
            result = ser.unstack(level=level)
            # 检查结果中的每列是否都是给定数据类型的数组
            assert all(
                isinstance(result[col].array, type(data)) for col in result.columns
            )

            if obj == "series":
                # 使用 to_frame+unstack+droplevel 应该得到相同的结果
                df = ser.to_frame()
                alt = df.unstack(level=level).droplevel(0, axis=1)
                tm.assert_frame_equal(result, alt)

            # 将对象转换为对象类型
            obj_ser = ser.astype(object)

            # 使用对象类型进行展开，并填充缺失值
            expected = obj_ser.unstack(level=level, fill_value=data.dtype.na_value)
            if obj == "series":
                # 对于 Series，检查所有列的数据类型是否为对象
                assert (expected.dtypes == object).all()

            # 将结果转换为对象类型
            result = result.astype(object)
            # 比较结果与期望值是否相等
            tm.assert_frame_equal(result, expected)

    # 定义测试函数 test_ravel，用于测试数据的展平操作
    def test_ravel(self, data):
        # 对数据进行展平操作
        result = data.ravel()
        # 检查结果的类型与原数据类型是否相同
        assert type(result) == type(data)

        if data.dtype._is_immutable:
            # 如果数据类型是不可变的，跳过测试
            pytest.skip(f"test_ravel assumes mutability and {data.dtype} is immutable")

        # 检查结果是否为视图而非副本
        result[0] = result[1]
        assert data[0] == data[1]

    # 定义测试函数 test_transpose，用于测试数据的转置操作
    def test_transpose(self, data):
        # 对数据进行转置操作
        result = data.transpose()
        # 检查结果的类型与原数据类型是否相同
        assert type(result) == type(data)

        # 检查结果是否为新对象
        assert result is not data

        # 如果数据类型是不可变的，跳过测试
        if data.dtype._is_immutable:
            pytest.skip(
                f"test_transpose assumes mutability and {data.dtype} is immutable"
            )

        # 检查结果是否为视图而非副本
        result[0] = result[1]
        assert data[0] == data[1]
    # 定义一个测试方法，用于验证 DataFrame 的转置操作
    def test_transpose_frame(self, data):
        # 创建一个包含两列和指定索引的 DataFrame
        df = pd.DataFrame({"A": data[:4], "B": data[:4]}, index=["a", "b", "c", "d"])
        
        # 对 DataFrame 进行转置操作
        result = df.T
        
        # 创建一个期望的转置后的 DataFrame，其中每列的数据根据给定数据和类型生成
        expected = pd.DataFrame(
            {
                "a": type(data)._from_sequence([data[0]] * 2, dtype=data.dtype),
                "b": type(data)._from_sequence([data[1]] * 2, dtype=data.dtype),
                "c": type(data)._from_sequence([data[2]] * 2, dtype=data.dtype),
                "d": type(data)._from_sequence([data[3]] * 2, dtype=data.dtype),
            },
            index=["A", "B"],
        )
        
        # 使用测试框架的方法验证结果 DataFrame 和期望 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
        
        # 使用测试框架的方法验证连续两次转置是否会还原原始 DataFrame
        tm.assert_frame_equal(np.transpose(np.transpose(df)), df)
        
        # 使用测试框架的方法验证选择单列后连续两次转置是否还原原始单列 DataFrame
        tm.assert_frame_equal(np.transpose(np.transpose(df[["A"]])), df[["A"]])
```