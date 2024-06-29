# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_groupby.py`

```
"""Test cases for GroupBy.plot"""

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于测试框架

from pandas import (  # 从 pandas 库中导入 DataFrame、Index、Series 类
    DataFrame,
    Index,
    Series,
)
from pandas.tests.plotting.common import (  # 从 pandas 测试模块中导入通用的绘图函数
    _check_axes_shape,
    _check_legend_labels,
)

pytest.importorskip("matplotlib")  # 检查并导入 matplotlib 库，如果不存在则跳过

class TestDataFrameGroupByPlots:
    def test_series_groupby_plotting_nominally_works(self):
        n = 10
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))  # 创建一个正态分布的随机数 Series
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)  # 从两个性别中随机选择

        weight.groupby(gender).plot()  # 对体重数据按性别分组并绘制图表

    def test_series_groupby_plotting_nominally_works_hist(self):
        n = 10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))  # 创建一个正态分布的随机数 Series
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)  # 从两个性别中随机选择
        height.groupby(gender).hist()  # 对身高数据按性别分组并绘制直方图

    def test_series_groupby_plotting_nominally_works_alpha(self):
        n = 10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))  # 创建一个正态分布的随机数 Series
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)  # 从两个性别中随机选择
        # 对身高数据按性别分组并绘制散点图，alpha=0.5 是 GH8733 的回归测试
        height.groupby(gender).plot(alpha=0.5)

    def test_plotting_with_float_index_works(self):
        # GH 7025 的测试用例
        df = DataFrame(
            {
                "def": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "val": np.random.default_rng(2).standard_normal(9),  # 创建一个标准正态分布的随机数列
            },
            index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],  # 指定 DataFrame 的浮点索引
        )

        df.groupby("def")["val"].plot()  # 根据 "def" 列的分组对 "val" 列数据绘制图表

    def test_plotting_with_float_index_works_apply(self):
        # GH 7025 的测试用例
        df = DataFrame(
            {
                "def": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "val": np.random.default_rng(2).standard_normal(9),  # 创建一个标准正态分布的随机数列
            },
            index=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],  # 指定 DataFrame 的浮点索引
        )
        # 对 "val" 列数据按 "def" 列分组，并应用 lambda 函数绘制图表
        df.groupby("def")["val"].apply(lambda x: x.plot())

    def test_hist_single_row(self):
        # GH10214 的测试用例
        bins = np.arange(80, 100 + 2, 1)  # 创建一个区间为 [80, 100] 的数组
        df = DataFrame({"Name": ["AAA", "BBB"], "ByCol": [1, 2], "Mark": [85, 89]})  # 创建包含姓名、列、标记的 DataFrame
        df["Mark"].hist(by=df["ByCol"], bins=bins)  # 对 "Mark" 列按 "ByCol" 列分组并绘制直方图

    def test_hist_single_row_single_bycol(self):
        # GH10214 的测试用例
        bins = np.arange(80, 100 + 2, 1)  # 创建一个区间为 [80, 100] 的数组
        df = DataFrame({"Name": ["AAA"], "ByCol": [1], "Mark": [85]})  # 创建包含姓名、列、标记的 DataFrame
        df["Mark"].hist(by=df["ByCol"], bins=bins)  # 对 "Mark" 列按 "ByCol" 列分组并绘制直方图

    def test_plot_submethod_works(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})  # 创建包含 x、y、z 列的 DataFrame
        df.groupby("z").plot.scatter("x", "y")  # 对 "z" 列分组并绘制散点图，指定 x、y 列作为坐标轴

    def test_plot_submethod_works_line(self):
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})  # 创建包含 x、y、z 列的 DataFrame
        df.groupby("z")["x"].plot.line()  # 对 "z" 列分组并绘制线图，指定 "x" 列为绘图数据
    def test_plot_kwargs(self):
        # 创建一个测试用的 DataFrame 包含列 'x', 'y', 'z'，分别为 [1, 2, 3, 4, 5], [1, 2, 3, 2, 1], 和 ['a', 'b', 'a', 'b', 'a']
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})

        # 对 DataFrame 按照 'z' 列进行分组，并绘制散点图
        res = df.groupby("z").plot(kind="scatter", x="x", y="y")

        # 断言：检查是否成功绘制了散点图，即判定每个分组中是否只有一个散点集合（PathCollection）
        assert len(res["a"].collections) == 1

    def test_plot_kwargs_scatter(self):
        # 创建一个测试用的 DataFrame 包含列 'x', 'y', 'z'，分别为 [1, 2, 3, 4, 5], [1, 2, 3, 2, 1], 和 ['a', 'b', 'a', 'b', 'a']
        df = DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 2, 1], "z": list("ababa")})

        # 对 DataFrame 按照 'z' 列进行分组，并绘制散点图
        res = df.groupby("z").plot.scatter(x="x", y="y")

        # 断言：检查是否成功绘制了散点图，即判定每个分组中是否只有一个散点集合（PathCollection）
        assert len(res["a"].collections) == 1

    @pytest.mark.parametrize("column, expected_axes_num", [(None, 2), ("b", 1)])
    def test_groupby_hist_frame_with_legend(self, column, expected_axes_num):
        # GH 6279 - DataFrameGroupBy 可以带有图例的直方图
        expected_layout = (1, expected_axes_num)
        expected_labels = column or [["a"], ["b"]]

        # 创建一个带有索引 'c' 的 DataFrame，包含两列随机正态分布数据 'a' 和 'b'
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        # 按照 'c' 列分组
        g = df.groupby("c")

        # 遍历分组后的直方图并进行断言
        for axes in g.hist(legend=True, column=column):
            _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
            for ax, expected_label in zip(axes[0], expected_labels):
                _check_legend_labels(ax, expected_label)

    @pytest.mark.parametrize("column", [None, "b"])
    def test_groupby_hist_frame_with_legend_raises(self, column):
        # GH 6279 - 带有图例和标签的 DataFrameGroupBy 直方图会抛出异常
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        # 使用断言检查是否抛出 ValueError 异常，提示不能同时使用图例和标签
        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            g.hist(legend=True, column=column, label="d")

    def test_groupby_hist_series_with_legend(self):
        # GH 6279 - SeriesGroupBy 可以带有图例的直方图
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        g = df.groupby("c")

        # 遍历分组后的 'a' 列直方图并进行断言
        for ax in g["a"].hist(legend=True):
            _check_axes_shape(ax, axes_num=1, layout=(1, 1))
            _check_legend_labels(ax, ["1", "2"])
    def test_groupby_hist_series_with_legend_raises(self):
        # 定义一个测试函数，用于验证 SeriesGroupBy 对象在使用 legend 和 label 时会引发异常
        # GH 6279 - SeriesGroupBy histogram with legend and label raises
        
        # 创建一个具有重复索引的 Index 对象，索引名称为 "c"
        index = Index(15 * ["1"] + 15 * ["2"], name="c")
        
        # 创建一个 DataFrame 对象，包含随机生成的标准正态分布数据，索引为上面创建的 index 对象，列名为 "a" 和 "b"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((30, 2)),
            index=index,
            columns=["a", "b"],
        )
        
        # 对 DataFrame 进行按列 "c" 分组，返回一个 GroupBy 对象
        g = df.groupby("c")

        # 使用 pytest 来验证 g.hist() 方法在 legend=True 且 label="d" 时会抛出 ValueError 异常
        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            g.hist(legend=True, label="d")
```