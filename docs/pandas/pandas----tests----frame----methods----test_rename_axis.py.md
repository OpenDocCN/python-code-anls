# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_rename_axis.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入 DataFrame、Index 和 MultiIndex 类
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestDataFrameRenameAxis:
    def test_rename_axis_inplace(self, float_frame):
        # 测试用例：重命名轴 inplace 操作
        # GH#15704
        # 预期结果：使用 "foo" 作为新的轴名称
        expected = float_frame.rename_axis("foo")
        result = float_frame.copy()
        # 调用 rename_axis 方法，通过 inplace=True 实现原地修改，返回值为 None
        return_value = no_return = result.rename_axis("foo", inplace=True)
        assert return_value is None  # 断言返回值为 None

        assert no_return is None  # 断言 no_return 变量为 None
        # 使用测试模块中的 assert_frame_equal 方法比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 预期结果：使用 "bar" 作为新的列轴名称
        expected = float_frame.rename_axis("bar", axis=1)
        result = float_frame.copy()
        # 调用 rename_axis 方法，通过 inplace=True 实现原地修改，返回值为 None
        return_value = no_return = result.rename_axis("bar", axis=1, inplace=True)
        assert return_value is None  # 断言返回值为 None

        assert no_return is None  # 断言 no_return 变量为 None
        # 使用测试模块中的 assert_frame_equal 方法比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_rename_axis_raises(self):
        # 测试用例：测试 rename_axis 方法抛出异常
        # GH#17833
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        # 使用 pytest 的 raises 方法检查 ValueError 异常是否被正确抛出，并匹配指定的错误信息
        with pytest.raises(ValueError, match="Use `.rename`"):
            df.rename_axis(id, axis=0)

        with pytest.raises(ValueError, match="Use `.rename`"):
            df.rename_axis({0: 10, 1: 20}, axis=0)

        with pytest.raises(ValueError, match="Use `.rename`"):
            df.rename_axis(id, axis=1)

        with pytest.raises(ValueError, match="Use `.rename`"):
            df["A"].rename_axis(id)
    def test_rename_axis_mapper(self):
        # GH#19978
        # 创建一个多级索引对象
        mi = MultiIndex.from_product([["a", "b", "c"], [1, 2]], names=["ll", "nn"])
        # 创建一个数据框架，使用上述多级索引作为行索引，包含两列 'x' 和 'y'
        df = DataFrame(
            {"x": list(range(len(mi))), "y": [i * 10 for i in range(len(mi))]}, index=mi
        )

        # 测试重命名列的 Index 对象
        result = df.rename_axis("cols", axis=1)
        # 断言重命名后的列 Index 名称为 'cols'
        tm.assert_index_equal(result.columns, Index(["x", "y"], name="cols"))

        # 使用字典测试重命名列的 Index 对象
        result = result.rename_axis(columns={"cols": "new"}, axis=1)
        # 断言重命名后的列 Index 名称为 'new'
        tm.assert_index_equal(result.columns, Index(["x", "y"], name="new"))

        # 使用字典测试重命名行索引
        result = df.rename_axis(index={"ll": "foo"})
        # 断言重命名后的行索引名称为 ['foo', 'nn']
        assert result.index.names == ["foo", "nn"]

        # 使用函数测试重命名行索引
        result = df.rename_axis(index=str.upper, axis=0)
        # 断言重命名后的行索引名称为 ['LL', 'NN']
        assert result.index.names == ["LL", "NN"]

        # 提供完整列表测试重命名行索引
        result = df.rename_axis(index=["foo", "goo"])
        # 断言重命名后的行索引名称为 ['foo', 'goo']
        assert result.index.names == ["foo", "goo"]

        # 同时测试修改行索引和列的名称
        sdf = df.reset_index().set_index("nn").drop(columns=["ll", "y"])
        result = sdf.rename_axis(index="foo", columns="meh")
        # 断言修改后的行索引名称为 'foo'，列索引名称为 'meh'
        assert result.index.name == "foo"
        assert result.columns.name == "meh"

        # 测试不同的错误情况
        with pytest.raises(TypeError, match="Must pass"):
            df.rename_axis(index="wrong")

        with pytest.raises(ValueError, match="Length of names"):
            df.rename_axis(index=["wrong"])

        with pytest.raises(TypeError, match="bogus"):
            df.rename_axis(bogus=None)

    @pytest.mark.parametrize(
        "kwargs, rename_index, rename_columns",
        [
            ({"mapper": None, "axis": 0}, True, False),
            ({"mapper": None, "axis": 1}, False, True),
            ({"index": None}, True, False),
            ({"columns": None}, False, True),
            ({"index": None, "columns": None}, True, True),
            ({}, False, False),
        ],
    )
    def test_rename_axis_none(self, kwargs, rename_index, rename_columns):
        # GH 25034
        # 创建行索引和列索引对象
        index = Index(list("abc"), name="foo")
        columns = Index(["col1", "col2"], name="bar")
        data = np.arange(6).reshape(3, 2)
        # 创建数据框架
        df = DataFrame(data, index, columns)

        # 测试使用不同参数重命名行索引和列索引
        result = df.rename_axis(**kwargs)
        expected_index = index.rename(None) if rename_index else index
        expected_columns = columns.rename(None) if rename_columns else columns
        expected = DataFrame(data, expected_index, expected_columns)
        # 断言重命名后的数据框架与期望的数据框架相等
        tm.assert_frame_equal(result, expected)
```