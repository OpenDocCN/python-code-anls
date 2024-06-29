# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_rename.py`

```
# 导入必要的库和模块
from collections import ChainMap  # 导入 ChainMap 类，用于合并字典
import inspect  # 导入 inspect 模块，用于获取对象的签名信息

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试

from pandas import (  # 从 Pandas 库中导入多个对象
    DataFrame,  # Pandas 的 DataFrame 类，用于处理表格数据
    Index,  # Pandas 的 Index 类，用于处理索引
    MultiIndex,  # Pandas 的 MultiIndex 类，用于处理多层索引
    merge,  # Pandas 的 merge 函数，用于合并数据
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于执行测试

# 定义一个测试类 TestRename，用于测试 DataFrame 的重命名功能
class TestRename:

    # 测试 DataFrame.rename 方法的签名是否符合预期
    def test_rename_signature(self):
        sig = inspect.signature(DataFrame.rename)
        parameters = set(sig.parameters)
        assert parameters == {
            "self",
            "mapper",
            "index",
            "columns",
            "axis",
            "inplace",
            "copy",
            "level",
            "errors",
        }

    # 测试在多层索引对象上使用 DataFrame.rename 方法
    def test_rename_mi(self, frame_or_series):
        obj = frame_or_series(
            [11, 21, 31],
            index=MultiIndex.from_tuples([("A", x) for x in ["a", "B", "c"]]),
        )
        obj.rename(str.lower)

    # 测试在普通 DataFrame 上使用 DataFrame.rename 方法
    def test_rename(self, float_frame):
        mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}

        # 根据指定的列名映射进行重命名
        renamed = float_frame.rename(columns=mapping)
        # 根据函数 str.lower 对列名进行重命名
        renamed2 = float_frame.rename(columns=str.lower)

        # 断言两个重命名后的 DataFrame 是否相等
        tm.assert_frame_equal(renamed, renamed2)
        tm.assert_frame_equal(
            renamed2.rename(columns=str.upper), float_frame, check_names=False
        )

        # 测试对索引进行重命名操作
        data = {"A": {"foo": 0, "bar": 1}}
        df = DataFrame(data)

        # 根据指定的索引映射进行重命名
        renamed = df.rename(index={"foo": "bar", "bar": "foo"})
        tm.assert_index_equal(renamed.index, Index(["bar", "foo"]))

        # 根据函数 str.upper 对索引进行重命名
        renamed = df.rename(index=str.upper)
        tm.assert_index_equal(renamed.index, Index(["FOO", "BAR"]))

        # 测试在没有传递索引参数时是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match="must pass an index to rename"):
            float_frame.rename()

        # 测试部分列的重命名操作
        renamed = float_frame.rename(columns={"C": "foo", "D": "bar"})
        tm.assert_index_equal(renamed.columns, Index(["A", "B", "foo", "bar"]))

        # 测试在另一个轴上进行重命名操作
        renamed = float_frame.T.rename(index={"C": "foo", "D": "bar"})
        tm.assert_index_equal(renamed.index, Index(["A", "B", "foo", "bar"]))

        # 测试带有名称的索引的重命名操作
        index = Index(["foo", "bar"], name="name")
        renamer = DataFrame(data, index=index)
        renamed = renamer.rename(index={"foo": "bar", "bar": "foo"})
        tm.assert_index_equal(renamed.index, Index(["bar", "foo"], name="name"))
        assert renamed.index.name == renamer.index.name

    # 使用参数化测试来测试 ChainMap 在 DataFrame.rename 中的使用情况
    @pytest.mark.parametrize(
        "args,kwargs",
        [
            ((ChainMap({"A": "a"}, {"B": "b"}),), {"axis": "columns"}),
            ((), {"columns": ChainMap({"A": "a"}, {"B": "b"})}),
        ],
    )
    def test_rename_chainmap(self, args, kwargs):
        # 见 GitHub 问题编号 gh-23859
        colAData = range(1, 11)
        colBdata = np.random.default_rng(2).standard_normal(10)

        # 创建 DataFrame 对象
        df = DataFrame({"A": colAData, "B": colBdata})
        # 对 DataFrame 对象进行重命名操作
        result = df.rename(*args, **kwargs)

        expected = DataFrame({"a": colAData, "b": colBdata})
        # 断言重命名后的结果是否与预期结果相等
        tm.assert_frame_equal(result, expected)
    # 测试在不复制的情况下重命名列
    def test_rename_nocopy(self, float_frame):
        # 使用新列名字典重命名 DataFrame 的列
        renamed = float_frame.rename(columns={"C": "foo"})

        # 断言新列与原始列共享内存
        assert np.shares_memory(renamed["foo"]._values, float_frame["C"]._values)

        # 修改重命名后的列的所有元素为 1.0，断言原始列没有被修改
        renamed.loc[:, "foo"] = 1.0
        assert not (float_frame["C"] == 1.0).all()

    # 测试原地修改列名
    def test_rename_inplace(self, float_frame):
        # 在原地修改列名，但不返回新的 DataFrame
        float_frame.rename(columns={"C": "foo"})
        assert "C" in float_frame  # 断言原始列名仍在 DataFrame 中
        assert "foo" not in float_frame  # 断言新列名不在 DataFrame 中

        # 获取原始列 "C" 的值
        c_values = float_frame["C"]

        # 复制 DataFrame，并在复制上进行重命名
        float_frame = float_frame.copy()
        return_value = float_frame.rename(columns={"C": "foo"}, inplace=True)

        # 断言原地修改返回 None
        assert return_value is None

        # 断言 "C" 列不在修改后的 DataFrame 中
        assert "C" not in float_frame
        # 断言 "foo" 列在修改后的 DataFrame 中
        assert "foo" in float_frame

        # 断言修改后的 "foo" 列不再指向原始列 "C" 的值
        assert float_frame["foo"] is not c_values

    # 测试重命名时的 bug
    def test_rename_bug(self):
        # GH 5344
        # 在设置索引后，重命名设置了 ref_locs，但 set_index 没有重置
        df = DataFrame({0: ["foo", "bar"], 1: ["bah", "bas"], 2: [1, 2]})
        df = df.rename(columns={0: "a"})
        df = df.rename(columns={1: "b"})
        df = df.set_index(["a", "b"])
        df.columns = ["2001-01-01"]
        expected = DataFrame(
            [[1], [2]],
            index=MultiIndex.from_tuples(
                [("foo", "bah"), ("bar", "bas")], names=["a", "b"]
            ),
            columns=["2001-01-01"],
        )
        # 断言修改后的 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    # 测试重命名时的另一个 bug
    def test_rename_bug2(self):
        # GH 19497
        # 如果 Index 包含元组，则重命名会将 Index 更改为 MultiIndex
        df = DataFrame(data=np.arange(3), index=[(0, 0), (1, 1), (2, 2)], columns=["a"])
        df = df.rename({(1, 1): (5, 4)}, axis="index")
        expected = DataFrame(
            data=np.arange(3), index=[(0, 0), (5, 4), (2, 2)], columns=["a"]
        )
        # 断言修改后的 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(df, expected)

    # 测试重命名时的错误处理
    def test_rename_errors_raises(self):
        df = DataFrame(columns=["A", "B", "C", "D"])
        # 使用 pytest 断言重命名时引发 KeyError 异常，且异常信息匹配
        with pytest.raises(KeyError, match="'E'] not found in axis"):
            df.rename(columns={"A": "a", "E": "e"}, errors="raise")

    # 参数化测试重命名时的错误处理
    @pytest.mark.parametrize(
        "mapper, errors, expected_columns",
        [
            ({"A": "a", "E": "e"}, "ignore", ["a", "B", "C", "D"]),
            ({"A": "a"}, "raise", ["a", "B", "C", "D"]),
            (str.lower, "raise", ["a", "b", "c", "d"]),
        ],
    )
    def test_rename_errors(self, mapper, errors, expected_columns):
        # GH 13473
        # 现在可以使用 errors 参数进行重命名
        df = DataFrame(columns=["A", "B", "C", "D"])
        result = df.rename(columns=mapper, errors=errors)
        expected = DataFrame(columns=expected_columns)
        # 断言重命名后的 DataFrame 与预期 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 测试重命名对象类型的列名
    def test_rename_objects(self, float_string_frame):
        # 使用函数将列名转为大写重命名 DataFrame 的列
        renamed = float_string_frame.rename(columns=str.upper)

        # 断言新列名存在于重命名后的 DataFrame 中
        assert "FOO" in renamed
        # 断言旧列名不再存在于重命名后的 DataFrame 中
        assert "foo" not in renamed
    def test_rename_axis_style(self):
        # 创建一个测试数据框，包括两列"A"和"B"，行索引为["X", "Y"]
        df = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["X", "Y"])
        # 创建预期结果的数据框，列名转为小写
        expected = DataFrame({"a": [1, 2], "b": [1, 2]}, index=["X", "Y"])

        # 使用函数str.lower对列名进行重命名，axis=1表示列方向
        result = df.rename(str.lower, axis=1)
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 再次使用函数str.lower对列名进行重命名，这次使用axis="columns"
        result = df.rename(str.lower, axis="columns")
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 使用字典形式对列名进行重命名，axis=1表示列方向
        result = df.rename({"A": "a", "B": "b"}, axis=1)
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 再次使用字典形式对列名进行重命名，这次使用axis="columns"
        result = df.rename({"A": "a", "B": "b"}, axis="columns")
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # Index方向的重命名
        # 创建预期结果的数据框，行索引转为小写
        expected = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["x", "y"])
        # 使用函数str.lower对行索引进行重命名，axis=0表示行方向
        result = df.rename(str.lower, axis=0)
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 再次使用函数str.lower对行索引进行重命名，这次使用axis="index"
        result = df.rename(str.lower, axis="index")
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 使用字典形式对行索引进行重命名，axis=0表示行方向
        result = df.rename({"X": "x", "Y": "y"}, axis=0)
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 再次使用字典形式对行索引进行重命名，这次使用axis="index"
        result = df.rename({"X": "x", "Y": "y"}, axis="index")
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

        # 使用函数mapper=str.lower对行索引进行重命名，这里指定axis="index"
        result = df.rename(mapper=str.lower, axis="index")
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

    def test_rename_mapper_multi(self):
        # 创建一个测试数据框，包括三列"A", "B", "C"，多重索引为["A", "B"]
        df = DataFrame({"A": ["a", "b"], "B": ["c", "d"], "C": [1, 2]}).set_index(["A", "B"])
        # 使用函数str.upper对索引标签进行重命名
        result = df.rename(str.upper)
        # 创建预期结果的数据框，只对索引的第一层进行重命名
        expected = df.rename(index=str.upper)
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)

    def test_rename_positional_named(self):
        # 创建一个测试数据框，包括两列"a"和"b"，行索引为["X", "Y"]
        df = DataFrame({"a": [1, 2], "b": [1, 2]}, index=["X", "Y"])
        # 使用函数str.lower对行索引和列名进行重命名
        result = df.rename(index=str.lower, columns=str.upper)
        # 创建预期结果的数据框，行索引转为小写，列名转为大写
        expected = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["x", "y"])
        # 断言结果与预期数据框相等
        tm.assert_frame_equal(result, expected)
    # 测试函数：test_rename_axis_style_raises，用于测试重命名操作是否能正确抛出异常
    def test_rename_axis_style_raises(self):
        # 创建一个包含两列的 DataFrame，带有字符串索引
        df = DataFrame({"A": [1, 2], "B": [1, 2]}, index=["0", "1"])

        # 测试指定了轴和索引或列时是否抛出 TypeError 异常
        over_spec_msg = "Cannot specify both 'axis' and any of 'index' or 'columns'"
        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(index=str.lower, axis=1)

        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(index=str.lower, axis="columns")

        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(columns=str.lower, axis="columns")

        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(index=str.lower, axis=0)

        # 测试同时指定多个目标和轴时是否抛出 TypeError 异常
        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(str.lower, index=str.lower, axis="columns")

        # 测试同时指定了映射器和索引或列时是否抛出 TypeError 异常
        over_spec_msg = "Cannot specify both 'mapper' and any of 'index' or 'columns'"
        with pytest.raises(TypeError, match=over_spec_msg):
            df.rename(str.lower, index=str.lower, columns=str.lower)

        # 测试当映射器和位置参数同时使用时是否抛出 TypeError 异常
        # 此处是测试用例 GH 29136
        df = DataFrame(columns=["A", "B"])
        msg = r"rename\(\) takes from 1 to 2 positional arguments"
        with pytest.raises(TypeError, match=msg):
            df.rename(None, str.lower)

    # 测试函数：test_rename_no_mappings_raises，用于测试在没有传递映射参数时是否能正确抛出异常
    def test_rename_no_mappings_raises(self):
        # 创建一个包含单元素的 DataFrame
        df = DataFrame([[1]])
        
        # 测试在不传递任何参数的情况下是否抛出 TypeError 异常
        msg = "must pass an index to rename"
        with pytest.raises(TypeError, match=msg):
            df.rename()

        # 测试在只传递 None 作为索引或列时是否抛出 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            df.rename(None, index=None)

        with pytest.raises(TypeError, match=msg):
            df.rename(None, columns=None)

        with pytest.raises(TypeError, match=msg):
            df.rename(None, columns=None, index=None)

    # 测试函数：test_rename_mapper_and_positional_arguments_raises，用于测试同时指定映射器和位置参数时是否能正确抛出异常
    def test_rename_mapper_and_positional_arguments_raises(self):
        # 创建一个包含单元素的 DataFrame
        df = DataFrame([[1]])

        # 测试在同时指定映射器和索引或列时是否抛出 TypeError 异常
        # 此处是测试用例 GH 29136
        msg = "Cannot specify both 'mapper' and any of 'index' or 'columns'"
        with pytest.raises(TypeError, match=msg):
            df.rename({}, index={})

        with pytest.raises(TypeError, match=msg):
            df.rename({}, columns={})

        with pytest.raises(TypeError, match=msg):
            df.rename({}, columns={}, index={})
    def test_rename_with_duplicate_columns(self):
        # GH#4403
        # 创建一个包含重复列名的DataFrame对象df4
        df4 = DataFrame(
            {"RT": [0.0454], "TClose": [22.02], "TExg": [0.0422]},
            index=MultiIndex.from_tuples(
                [(600809, 20130331)], names=["STK_ID", "RPT_Date"]
            ),
        )

        # 创建另一个DataFrame对象df5，包含重复的STK_ID和RPT_Date，以及TClose列
        df5 = DataFrame(
            {
                "RPT_Date": [20120930, 20121231, 20130331],
                "STK_ID": [600809] * 3,
                "STK_Name": ["饡驦", "饡驦", "饡驦"],
                "TClose": [38.05, 41.66, 30.01],
            },
            index=MultiIndex.from_tuples(
                [(600809, 20120930), (600809, 20121231), (600809, 20130331)],
                names=["STK_ID", "RPT_Date"],
            ),
        )

        # 执行DataFrame对象df4和df5的内连接，基于索引进行合并
        k = merge(df4, df5, how="inner", left_index=True, right_index=True)

        # 对合并后的DataFrame对象k进行列重命名，将"TClose_x"重命名为"TClose"，"TClose_y"重命名为"QT_Close"
        result = k.rename(columns={"TClose_x": "TClose", "TClose_y": "QT_Close"})

        # 创建期望的DataFrame对象expected，设置了特定的列和索引
        expected = DataFrame(
            [[0.0454, 22.02, 0.0422, 20130331, 600809, "饡驦", 30.01]],
            columns=[
                "RT",
                "TClose",
                "TExg",
                "RPT_Date",
                "STK_ID",
                "STK_Name",
                "QT_Close",
            ],
        ).set_index(["STK_ID", "RPT_Date"], drop=False)

        # 使用测试工具函数tm.assert_frame_equal比较result和expected是否相等
        tm.assert_frame_equal(result, expected)

    def test_rename_boolean_index(self):
        # 创建一个DataFrame对象df，使用布尔值作为列名
        df = DataFrame(np.arange(15).reshape(3, 5), columns=[False, True, 2, 3, 4])

        # 创建一个字典mapper，用于将索引重命名为字符串
        mapper = {0: "foo", 1: "bar", 2: "bah"}

        # 使用字典mapper对DataFrame对象df进行索引重命名
        res = df.rename(index=mapper)

        # 创建期望的DataFrame对象exp，设置了特定的列和索引
        exp = DataFrame(
            np.arange(15).reshape(3, 5),
            columns=[False, True, 2, 3, 4],
            index=["foo", "bar", "bah"],
        )

        # 使用测试工具函数tm.assert_frame_equal比较res和exp是否相等
        tm.assert_frame_equal(res, exp)
```