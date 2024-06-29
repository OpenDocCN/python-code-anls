# `D:\src\scipysrc\pandas\pandas\tests\generic\test_frame.py`

```
    # 导入深拷贝函数 deepcopy 和方法调用器 methodcaller
    from copy import deepcopy
    from operator import methodcaller

    # 导入 numpy 库，并将其别名为 np
    import numpy as np

    # 导入 pytest 库，用于单元测试
    import pytest

    # 导入 pandas 库，并从中导入 DataFrame、MultiIndex、Series、date_range 等
    import pandas as pd
    from pandas import (
        DataFrame,
        MultiIndex,
        Series,
        date_range,
    )

    # 导入 pandas 内部测试工具模块
    import pandas._testing as tm

    # 定义 TestDataFrame 类，用于测试 DataFrame 的功能
    class TestDataFrame:
        
        # 使用 pytest 的参数化装饰器，测试 _set_axis_name 和 rename_axis 方法
        @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
        def test_set_axis_name(self, func):
            # 创建一个包含两行两列的 DataFrame 对象
            df = DataFrame([[1, 2], [3, 4]])

            # 使用 methodcaller 调用指定的 func 方法，并传入参数 "foo"
            result = methodcaller(func, "foo")(df)
            # 断言原 DataFrame 的行索引名为 None
            assert df.index.name is None
            # 断言结果 DataFrame 的行索引名为 "foo"
            assert result.index.name == "foo"

            # 再次使用 methodcaller 调用 func 方法，传入参数 "cols" 和 axis=1
            result = methodcaller(func, "cols", axis=1)(df)
            # 断言原 DataFrame 的列索引名为 None
            assert df.columns.name is None
            # 断言结果 DataFrame 的列索引名为 "cols"
            assert result.columns.name == "cols"

        # 使用 pytest 的参数化装饰器，测试 _set_axis_name 和 rename_axis 方法（针对多级索引）
        @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
        def test_set_axis_name_mi(self, func):
            # 创建一个包含空值的 3x3 DataFrame，同时指定多级行和列索引
            df = DataFrame(
                np.empty((3, 3)),
                index=MultiIndex.from_tuples([("A", x) for x in list("aBc")]),
                columns=MultiIndex.from_tuples([("C", x) for x in list("xyz")]),
            )

            # 定义多级索引的名称列表
            level_names = ["L1", "L2"]

            # 使用 methodcaller 调用 func 方法，并传入 level_names 参数
            result = methodcaller(func, level_names)(df)
            # 断言结果 DataFrame 的行索引名称为 level_names
            assert result.index.names == level_names
            # 断言结果 DataFrame 的列索引名称为 [None, None]
            assert result.columns.names == [None, None]

            # 再次使用 methodcaller 调用 func 方法，传入 level_names 和 axis=1 参数
            result = methodcaller(func, level_names, axis=1)(df)
            # 断言结果 DataFrame 的列索引名称为 ["L1", "L2"]
            assert result.columns.names == ["L1", "L2"]
            # 断言结果 DataFrame 的行索引名称为 [None, None]
            assert result.index.names == [None, None]

        # 测试当 DataFrame 包含单个元素时的布尔运算
        def test_nonzero_single_element(self):
            # 创建一个包含单个元素为 False 的 DataFrame
            df = DataFrame([[False, False]])
            # 定义错误消息
            msg_err = "The truth value of a DataFrame is ambiguous"
            # 使用 pytest 的上下文管理器断言抛出 ValueError 异常，并匹配错误消息
            with pytest.raises(ValueError, match=msg_err):
                bool(df)

        # 测试元数据在个别 groupby 操作后的传播
        def test_metadata_propagation_indiv_groupby(self):
            # 创建一个包含多列数据的 DataFrame，包括 "A", "B", "C", "D" 列
            df = DataFrame(
                {
                    "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                    "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                    "C": np.random.default_rng(2).standard_normal(8),
                    "D": np.random.default_rng(2).standard_normal(8),
                }
            )
            # 对 DataFrame 进行 groupby 操作，并求和
            result = df.groupby("A").sum()
            # 使用测试工具模块的方法断言 df 和 result 的元数据是等价的
            tm.assert_metadata_equivalent(df, result)

        # 测试元数据在个别 resample 操作后的传播
        def test_metadata_propagation_indiv_resample(self):
            # 创建一个包含随机数据的 DataFrame，包含 1000 行 2 列
            df = DataFrame(
                np.random.default_rng(2).standard_normal((1000, 2)),
                index=date_range("20130101", periods=1000, freq="s"),
            )
            # 对 DataFrame 进行 resample 操作，按 1 分钟进行重采样
            result = df.resample("1min")
            # 使用测试工具模块的方法断言 df 和 result 的元数据是等价的
            tm.assert_metadata_equivalent(df, result)
    def test_metadata_propagation_indiv(self, monkeypatch):
        # 定义一个测试方法，用于测试元数据的传播，使用 monkeypatch 来模拟环境

        # 定义一个 finalize 方法，用于处理元数据的合并，根据不同的方法进行不同的处理
        def finalize(self, other, method=None, **kwargs):
            # 遍历当前对象的元数据
            for name in self._metadata:
                if method == "merge":
                    # 如果方法是 merge，则合并 left 和 right 对象中的元数据值
                    left, right = other.left, other.right
                    value = getattr(left, name, "") + "|" + getattr(right, name, "")
                    object.__setattr__(self, name, value)
                elif method == "concat":
                    # 如果方法是 concat，则连接所有对象中的同名元数据值
                    value = "+".join(
                        [getattr(o, name) for o in other.objs if getattr(o, name, None)]
                    )
                    object.__setattr__(self, name, value)
                else:
                    # 否则直接将 other 对象的同名元数据值赋给当前对象
                    object.__setattr__(self, name, getattr(other, name, ""))

            return self

        # 使用 monkeypatch 来修改 DataFrame 类的 _metadata 和 __finalize__ 属性
        with monkeypatch.context() as m:
            m.setattr(DataFrame, "_metadata", ["filename"])
            m.setattr(DataFrame, "__finalize__", finalize)

            # 创建两个 DataFrame 对象 df1 和 df2，并设置它们各自的 filename 属性
            df1 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=["a", "b"]
            )
            df2 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=["c", "d"]
            )
            DataFrame._metadata = ["filename"]
            df1.filename = "fname1.csv"
            df2.filename = "fname2.csv"

            # 使用 merge 方法将 df1 和 df2 合并，设置合并后的 filename 属性，并断言结果
            result = df1.merge(df2, left_on=["a"], right_on=["c"], how="inner")
            assert result.filename == "fname1.csv|fname2.csv"

            # 测试 concat 方法，创建一个 DataFrame df1，设置其 filename 属性
            df1 = DataFrame(
                np.random.default_rng(2).integers(0, 4, (3, 2)), columns=list("ab")
            )
            df1.filename = "foo"

            # 使用 concat 方法将 df1 和自身连接，设置连接后的 filename 属性，并断言结果
            result = pd.concat([df1, df1])
            assert result.filename == "foo+foo"

    def test_set_attribute(self):
        # 测试在属性名与列名相同时，setattr 行为的一致性

        # 创建一个包含列 'x' 的 DataFrame 对象 df
        df = DataFrame({"x": [1, 2, 3]})

        # 设置 df 的 y 属性为 2
        df.y = 2
        # 设置 df 的 'y' 列为 [2, 4, 6]
        df["y"] = [2, 4, 6]
        # 再次设置 df 的 y 属性为 5
        df.y = 5

        # 断言 df 的 y 属性为 5
        assert df.y == 5
        # 使用 assert_series_equal 检查 'y' 列的值是否正确
        tm.assert_series_equal(df["y"], Series([2, 4, 6], name="y"))

    def test_deepcopy_empty(self):
        # 测试空 DataFrame 的深拷贝操作

        # 创建一个空的 DataFrame 对象 empty_frame，只包含列 'A'
        empty_frame = DataFrame(data=[], index=[], columns=["A"])
        # 对 empty_frame 进行深拷贝操作
        empty_frame_copy = deepcopy(empty_frame)

        # 使用 tm.assert_frame_equal 检查拷贝后的结果是否与原始对象相等
        tm.assert_frame_equal(empty_frame_copy, empty_frame)
# 以前在 Generic 类中，但只测试 DataFrame
class TestDataFrame2:
    # 使用 pytest 的参数化装饰器，对不同的 value 参数进行测试
    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        # 创建一个 DataFrame 对象 df，包含两列 "a" 和 "b"
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # 准备错误信息片段
        msg = 'For argument "inplace" expected type bool, received type'

        # 用 pytest 的 raises 方法检测是否抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 rename_axis 方法，设置 inplace 参数为 value
            df.copy().rename_axis(mapper={"a": "x", "b": "y"}, axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 drop 方法，设置 inplace 参数为 value
            df.copy().drop("a", axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 fillna 方法，设置 inplace 参数为 value
            df.copy().fillna(value=0, inplace=value)

        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 replace 方法，设置 inplace 参数为 value
            df.copy().replace(to_replace=1, value=7, inplace=value)

        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 interpolate 方法，设置 inplace 参数为 value
            df.copy().interpolate(inplace=value)

        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 _where 方法，设置 inplace 参数为 value
            df.copy()._where(cond=df.a > 2, inplace=value)

        with pytest.raises(ValueError, match=msg):
            # 在复制的 df 上调用 mask 方法，设置 inplace 参数为 value
            df.copy().mask(cond=df.a > 2, inplace=value)

    # 测试不期望的关键字错误
    def test_unexpected_keyword(self):
        # GH8597
        # 创建一个 DataFrame 对象 df，包含 5 行 2 列的随机标准正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["jim", "joe"]
        )
        # 创建一个分类数据对象 ca
        ca = pd.Categorical([0, 0, 2, 2, 3, np.nan])
        # 从 df 的 "joe" 列复制出一个时间序列对象 ts
        ts = df["joe"].copy()
        # 将 ts 中的第 2 个元素设为 NaN
        ts[2] = np.nan

        # 准备错误信息片段
        msg = "unexpected keyword"

        # 用 pytest 的 raises 方法检测是否抛出 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 在 df 上调用 drop 方法，设置 in_place 参数为 True（错误拼写）
            df.drop("joe", axis=1, in_place=True)

        with pytest.raises(TypeError, match=msg):
            # 在 df 上调用 reindex 方法，设置 inplace 参数为 True
            df.reindex([1, 0], inplace=True)

        with pytest.raises(TypeError, match=msg):
            # 在 ca 上调用 fillna 方法，设置 inplace 参数为 True
            ca.fillna(0, inplace=True)

        with pytest.raises(TypeError, match=msg):
            # 在 ts 上调用 fillna 方法，设置 in_place 参数为 True（错误拼写）
            ts.fillna(0, in_place=True)
```