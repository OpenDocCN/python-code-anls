# `D:\src\scipysrc\seaborn\tests\test_categorical.py`

```
import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数
from functools import partial  # 导入 functools 模块中的 partial 函数，用于部分函数应用
import warnings  # 导入 warnings 模块，用于处理警告信息

import numpy as np  # 导入 NumPy 库，并用 np 别名表示
import pandas as pd  # 导入 Pandas 库，并用 pd 别名表示
import matplotlib as mpl  # 导入 Matplotlib 库，并用 mpl 别名表示
import matplotlib.pyplot as plt  # 导入 Matplotlib 库中的 pyplot 模块，并用 plt 别名表示
from matplotlib.colors import same_color, to_rgb, to_rgba  # 导入 Matplotlib 中的颜色相关函数

import pytest  # 导入 Pytest 测试框架
from pytest import approx  # 从 Pytest 中导入 approx，用于近似值比较
from numpy.testing import (  # 从 NumPy 测试工具中导入以下函数，用于数组比较
    assert_array_equal,
    assert_array_less,
    assert_array_almost_equal,
)

from seaborn import categorical as cat  # 从 Seaborn 库中导入 categorical 函数，并用 cat 别名表示

from seaborn._base import categorical_order  # 从 Seaborn 库中导入 categorical_order 函数
from seaborn._compat import get_colormap, get_legend_handles  # 从 Seaborn 库中导入兼容性相关函数
from seaborn._testing import assert_plots_equal  # 从 Seaborn 测试模块中导入 assert_plots_equal 函数
from seaborn.categorical import (  # 从 Seaborn 的 categorical 模块中导入以下绘图函数
    _CategoricalPlotter,
    Beeswarm,
    BoxPlotContainer,
    catplot,
    barplot,
    boxplot,
    boxenplot,
    countplot,
    pointplot,
    stripplot,
    swarmplot,
    violinplot,
)
from seaborn.palettes import color_palette  # 从 Seaborn 的调色板模块中导入 color_palette 函数
from seaborn.utils import _draw_figure, _version_predates, desaturate  # 从 Seaborn 的工具模块中导入相关函数


PLOT_FUNCS = [  # 定义 PLOT_FUNCS 列表，包含了 Seaborn 中各种绘图函数
    catplot,
    barplot,
    boxplot,
    boxenplot,
    pointplot,
    stripplot,
    swarmplot,
    violinplot,
]


class TestCategoricalPlotterNew:

    @pytest.mark.parametrize(  # 使用 Pytest 的 parametrize 装饰器进行参数化测试
        "func,kwargs",
        itertools.product(  # 使用 itertools.product 生成 PLOT_FUNCS 和参数组合的迭代器
            PLOT_FUNCS,
            [
                {"x": "x", "y": "a"},  # 参数组合1
                {"x": "a", "y": "y"},  # 参数组合2
                {"x": "y"},            # 参数组合3
                {"y": "x"},            # 参数组合4
            ],
        ),
    )
    def test_axis_labels(self, long_df, func, kwargs):
        """
        测试函数：test_axis_labels
        参数：
        - long_df: 长格式数据框
        - func: 要测试的绘图函数
        - kwargs: 绘图函数的关键字参数
        """
        func(data=long_df, **kwargs)  # 调用绘图函数，传入长格式数据和关键字参数

        ax = plt.gca()  # 获取当前的坐标轴对象
        for axis in "xy":  # 遍历坐标轴 "x" 和 "y"
            val = kwargs.get(axis, "")  # 获取关键字参数中的坐标轴标签
            label_func = getattr(ax, f"get_{axis}label")  # 获取坐标轴标签的获取函数
            assert label_func() == val  # 断言获取的坐标轴标签与预期值相等

    @pytest.mark.parametrize("func", PLOT_FUNCS)
    def test_empty(self, func):
        """
        测试函数：test_empty
        参数：
        - func: 要测试的绘图函数
        """
        func()  # 调用绘图函数，不传入数据，测试空数据情况

        ax = plt.gca()  # 获取当前的坐标轴对象
        assert not ax.collections  # 断言坐标轴中没有集合对象
        assert not ax.patches      # 断言坐标轴中没有补丁对象
        assert not ax.lines        # 断言坐标轴中没有线条对象

        func(x=[], y=[])  # 调用绘图函数，传入空的 x 和 y 数据，再次测试空数据情况
        ax = plt.gca()  # 获取当前的坐标轴对象
        assert not ax.collections  # 断言坐标轴中没有集合对象
        assert not ax.patches      # 断言坐标轴中没有补丁对象
        assert not ax.lines        # 断言坐标轴中没有线条对象

    def test_redundant_hue_backcompat(self, long_df):
        """
        测试函数：test_redundant_hue_backcompat
        参数：
        - long_df: 长格式数据框
        """
        p = _CategoricalPlotter(  # 创建 _CategoricalPlotter 对象，传入长格式数据框和变量映射
            data=long_df,
            variables={"x": "s", "y": "y"},
        )

        color = None  # 设置颜色为 None
        palette = dict(zip(long_df["s"].unique(), color_palette()))  # 创建调色板字典
        hue_order = None  # 设置 hue_order 为 None

        palette, _ = p._hue_backcompat(color, palette, hue_order, force_hue=True)  # 调用 _hue_backcompat 方法

        assert p.variables["hue"] == "s"  # 断言变量映射中的 hue 变量为 "s"
        assert_array_equal(p.plot_data["hue"], p.plot_data["x"])  # 断言绘图数据中的 hue 列与 x 列相等
        assert all(isinstance(k, str) for k in palette)  # 断言调色板中的所有键都是字符串类型


class SharedAxesLevelTests:

    def orient_indices(self, orient):
        """
        方法：orient_indices
        参数：
        - orient: 方向参数，可以是 "x" 或 "y"
        返回：
        - pos_idx: 位置索引，"x" 返回 0，"y" 返回 1
        - val_idx: 值索引，"y" 返回 0，"x" 返回 1
        """
        pos_idx = ["x", "y"].index(orient)  # 获取方向参数在列表中的位置索引
        val_idx = ["y", "x"].index(orient)  # 获取方向参数在列表中的值索引
        return pos_idx, val_idx  # 返回位置索引和值索引

    @pytest.fixture
    def common_kws(self):
        """
        夹具：common_kws
        返回：
        - 空字典，用于共享参数
        """
        return {}

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_orient_indices(self, orient):
        """
        测试函数：test_orient_indices
        参数：
        - orient: 方向参数，可以是 "x" 或 "y"
        """
        pos_idx, val_idx = self.orient_indices(orient)  # 调用 orient_indices 方法获取索引

        if orient == "x":
            assert pos_idx == 0  # 断言位置索引为 0
            assert val_idx == 1  # 断言值索引为 1
        elif orient == "y":
            assert pos_idx == 1  # 断言位置索引为 1
            assert val_idx == 0  # 断言值索引为 0
    def test_labels_long(self, long_df, orient):
        # 根据指定的 orient 参数获取其依赖的另一个方向的名称
        depend = {"x": "y", "y": "x"}[orient]
        # 构建关键字参数字典 kws，包括 orient 对应的值、depend 对应的值、以及固定的 hue 对应的值
        kws = {orient: "a", depend: "y", "hue": "b"}

        # 调用被测试的函数 func，传入 long_df 数据和关键字参数 kws，返回一个 AxesSubplot 对象 ax
        ax = self.func(long_df, **kws)

        # 在旧版本的 matplotlib 上可能需要调用 _draw_figure 来填充图形中的文本信息
        _draw_figure(ax.figure)

        # 断言获取的 ax 对象的 orient 标签的文本内容等于 kws 中 orient 对应的值
        assert getattr(ax, f"get_{orient}label")() == kws[orient]
        # 断言获取的 ax 对象的 depend 标签的文本内容等于 kws 中 depend 对应的值
        assert getattr(ax, f"get_{depend}label")() == kws[depend]

        # 获取 ax 对象的 orient 方向的刻度标签文本内容列表
        get_ori_labels = getattr(ax, f"get_{orient}ticklabels")
        ori_labels = [t.get_text() for t in get_ori_labels()]
        # 获取 long_df 数据中 kws[orient] 对应列的分类顺序
        ori_levels = categorical_order(long_df[kws[orient]])
        # 断言 ax 对象的 orient 方向的刻度标签文本内容与预期的分类顺序一致
        assert ori_labels == ori_levels

        # 获取 ax 对象的图例
        legend = ax.get_legend()
        # 断言图例的标题文本内容等于 kws 中 hue 对应的值
        assert legend.get_title().get_text() == kws["hue"]

        # 获取图例中所有文本标签的文本内容列表
        hue_labels = [t.get_text() for t in legend.texts]
        # 获取 long_df 数据中 kws["hue"] 对应列的分类顺序
        hue_levels = categorical_order(long_df[kws["hue"]])
        # 断言图例中所有文本标签的文本内容与预期的分类顺序一致
        assert hue_labels == hue_levels

    def test_labels_wide(self, wide_df):
        # 将 wide_df 数据框的列名设置为 "cols"
        wide_df = wide_df.rename_axis("cols", axis=1)
        # 调用被测试的函数 func，传入 wide_df 数据，返回一个 AxesSubplot 对象 ax
        ax = self.func(wide_df)

        # 在旧版本的 matplotlib 上可能需要调用 _draw_figure 来填充图形中的文本信息
        _draw_figure(ax.figure)

        # 断言获取的 ax 对象的 x 轴标签文本内容等于 wide_df 数据框的列名
        assert ax.get_xlabel() == wide_df.columns.name
        # 获取 ax 对象的 x 轴刻度标签文本内容列表
        labels = [t.get_text() for t in ax.get_xticklabels()]
        # 逐一断言获取的刻度标签文本内容与 wide_df 数据框的列名一致
        for label, level in zip(labels, wide_df.columns):
            assert label == level

    def test_labels_hue_order(self, long_df):
        # 指定 hue 变量名为 "b"
        hue_var = "b"
        # 获取 long_df 数据中 hue_var 列的分类顺序，并反转顺序
        hue_order = categorical_order(long_df[hue_var])[::-1]
        # 调用被测试的函数 func，传入 long_df 数据和指定的参数，返回一个 AxesSubplot 对象 ax
        ax = self.func(long_df, x="a", y="y", hue=hue_var, hue_order=hue_order)
        # 获取 ax 对象的图例
        legend = ax.get_legend()
        # 获取图例中所有文本标签的文本内容列表
        hue_labels = [t.get_text() for t in legend.texts]
        # 断言图例中所有文本标签的文本内容与预期的 hue_order 顺序一致
        assert hue_labels == hue_order

    def test_color(self, long_df, common_kws):
        # 更新 common_kws 字典，添加 data、x 和 y 键，对应值为 long_df 数据和字符串 "a"、"y"
        common_kws.update(data=long_df, x="a", y="y")

        # 创建一个新的 Figure 对象，并获取其子图对象 ax
        ax = plt.figure().subplots()
        # 调用被测试的函数 func，传入 ax 对象和更新后的 common_kws 字典
        self.func(ax=ax, **common_kws)
        # 断言获取的 ax 对象中最后一个使用的颜色与字符串 "C0" 对应的 RGBA 颜色值一致
        assert self.get_last_color(ax) == to_rgba("C0")

        # 重复上述过程，调用两次 self.func，并断言最后一个使用的颜色与字符串 "C1" 对应的 RGBA 颜色值一致
        ax = plt.figure().subplots()
        self.func(ax=ax, **common_kws)
        self.func(ax=ax, **common_kws)
        assert self.get_last_color(ax) == to_rgba("C1")

        # 创建一个新的 Figure 对象，并获取其子图对象 ax
        ax = plt.figure().subplots()
        # 调用被测试的函数 func，传入指定的颜色 "C2"、ax 对象和更新后的 common_kws 字典
        self.func(color="C2", ax=ax, **common_kws)
        # 断言获取的 ax 对象中最后一个使用的颜色与字符串 "C2" 对应的 RGBA 颜色值一致
        assert self.get_last_color(ax) == to_rgba("C2")

        # 创建一个新的 Figure 对象，并获取其子图对象 ax
        ax = plt.figure().subplots()
        # 调用被测试的函数 func，传入指定的颜色 "C3"、ax 对象和更新后的 common_kws 字典
        self.func(color="C3", ax=ax, **common_kws)
        # 断言获取的 ax 对象中最后一个使用的颜色与字符串 "C3" 对应的 RGBA 颜色值一致
        assert self.get_last_color(ax) == to_rgba("C3")

    def test_two_calls(self):
        # 创建一个新的 Figure 对象，并获取其子图对象 ax
        ax = plt.figure().subplots()
        # 调用两次被测试的函数 func，分别传入不同的 x 和 y 值列表，以及相同的 ax 对象
        self.func(x=["a", "b", "c"], y=[1, 2, 3], ax=ax)
        self.func(x=["e", "f"], y=[4, 5], ax=ax)
        # 断言获取的 ax 对象的 x 轴限制范围为 (-0.5, 4.5)
        assert ax.get_xlim() == (-.5, 4.5)

    def test_redundant_hue_legend(self, long_df):
        # 调用被测试的函数 func，传入 long_df 数据、x 和 y 参数，以及 hue 参数，不显示图例
        ax = self.func(long_df, x="a", y="y", hue="a")
        # 断言获取的 ax 对象的图例为空
        assert ax.get_legend() is None
        # 清除 ax 对象的内容
        ax.clear()

        # 再次调用被测试的函数 func，传入 long_df 数据、x 和 y 参数，以及 hue 参数，并显示图例
        self.func(long_df, x="a", y="y", hue="a", legend=True)
        # 断言获取的 ax 对象的图例不为空
        assert ax.get_legend() is not None

    @pytest.mark.parametrize("orient", ["x", "y"])
    # 定义一个测试方法，用于测试对数比例尺的功能
    def test_log_scale(self, long_df, orient):
        # 根据给定的方向确定因变量和自变量
        depvar = {"x": "y", "y": "x"}[orient]
        # 构建变量字典，根据方向设置自变量和因变量
        variables = {orient: "a", depvar: "z"}
        # 调用被测试对象的方法，并传入长格式数据及变量字典，启用对数比例尺
        ax = self.func(long_df, **variables, log_scale=True)
        # 断言获取的自变量比例尺为线性比例尺
        assert getattr(ax, f"get_{orient}scale")() == "linear"
        # 断言获取的因变量比例尺为对数比例尺
        assert getattr(ax, f"get_{depvar}scale")() == "log"
# 创建一个测试类 SharedScatterTests，继承自 SharedAxesLevelTests，用于测试 stripplot 和 swarmplot 的共同功能。
class SharedScatterTests(SharedAxesLevelTests):
    """Tests functionality common to stripplot and swarmplot."""

    # 获取最后一个图表的颜色
    def get_last_color(self, ax):
        # 获取最后一个集合（collection），并获取其面部颜色
        colors = ax.collections[-1].get_facecolors()
        # 找出唯一的颜色值
        unique_colors = np.unique(colors, axis=0)
        # 断言只有一个唯一的颜色
        assert len(unique_colors) == 1
        # 将唯一的颜色值转换成 RGBA 格式并返回
        return to_rgba(unique_colors.squeeze())

    # ------------------------------------------------------------------------------

    # 测试颜色相关功能
    def test_color(self, long_df, common_kws):
        # 调用父类方法进行颜色测试
        super().test_color(long_df, common_kws)

        # 创建一个新的子图并调用被测试的函数，设置面部颜色为 "C4"
        ax = plt.figure().subplots()
        self.func(data=long_df, x="a", y="y", facecolor="C4", ax=ax)
        # 断言最后一个图表的颜色与 "C4" 转换成 RGBA 后的颜色相同
        assert self.get_last_color(ax) == to_rgba("C4")

        # 创建另一个新的子图并调用被测试的函数，设置面部颜色为 "C5"
        ax = plt.figure().subplots()
        self.func(data=long_df, x="a", y="y", fc="C5", ax=ax)
        # 断言最后一个图表的颜色与 "C5" 转换成 RGBA 后的颜色相同
        assert self.get_last_color(ax) == to_rgba("C5")

    # 测试提供的颜色数组
    def test_supplied_color_array(self, long_df):
        # 获取 "Blues" 颜色映射
        cmap = get_colormap("Blues")
        # 创建归一化对象
        norm = mpl.colors.Normalize()
        # 根据数据中的 "y" 列生成颜色数组
        colors = cmap(norm(long_df["y"].to_numpy()))

        # 需要测试的键列表
        keys = ["c", "fc", "facecolor", "facecolors"]

        # 遍历每个键进行测试
        for key in keys:
            # 创建一个新的子图并调用被测试的函数，设置颜色为 colors 数组
            ax = plt.figure().subplots()
            self.func(x=long_df["y"], **{key: colors})
            _draw_figure(ax.figure)
            # 断言第一个集合的面部颜色与 colors 数组相同
            assert_array_equal(ax.collections[0].get_facecolors(), colors)

        # 创建一个新的子图并调用被测试的函数，设置颜色为 long_df["y"] 列的颜色映射后的值
        ax = plt.figure().subplots()
        self.func(x=long_df["y"], c=long_df["y"], cmap=cmap)
        _draw_figure(ax.figure)
        # 断言第一个集合的面部颜色与 colors 数组相同
        assert_array_equal(ax.collections[0].get_facecolors(), colors)

    # 测试未填充的标记
    def test_unfilled_marker(self, long_df):
        # 捕获 UserWarning 警告并抛出错误
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # 调用被测试的函数，设置标记为 "x"，颜色为 "r"
            ax = self.func(long_df, x="y", y="a", marker="x", color="r")
            # 遍历每个集合，并断言其面部颜色与 "r" 相同
            for points in ax.collections:
                assert same_color(points.get_facecolors().squeeze(), "r")
                assert same_color(points.get_edgecolors().squeeze(), "r")

    # 使用 pytest 的参数化功能进行测试
    @pytest.mark.parametrize(
        "orient,data_type", [
            ("h", "dataframe"), ("h", "dict"),
            ("v", "dataframe"), ("v", "dict"),
            ("y", "dataframe"), ("y", "dict"),
            ("x", "dataframe"), ("x", "dict"),
        ]
    )
    def test_wide(self, wide_df, orient, data_type):
        # 检查数据类型是否为字典，若是，则将字典的值转换为 NumPy 数组
        if data_type == "dict":
            wide_df = {k: v.to_numpy() for k, v in wide_df.items()}

        # 调用指定的绘图函数，传入数据、方向和颜色参数，并获取绘图对象 ax
        ax = self.func(data=wide_df, orient=orient, color="C0")
        # 调用内部函数 _draw_figure，传入 ax 对象的图形属性
        _draw_figure(ax.figure)

        # 确定分类轴和值轴的索引位置
        cat_idx = 0 if orient in "vx" else 1
        val_idx = int(not cat_idx)

        # 获取图形对象的 x 轴和 y 轴对象
        axis_objs = ax.xaxis, ax.yaxis
        cat_axis = axis_objs[cat_idx]

        # 遍历分类轴上的主要刻度标签
        for i, label in enumerate(cat_axis.get_majorticklabels()):
            # 获取当前刻度标签的文本内容作为键
            key = label.get_text()
            # 获取与当前刻度相关联的点集合对象
            points = ax.collections[i]
            # 获取点集合对象中点的偏移位置
            point_pos = points.get_offsets().T
            # 获取值轴和分类轴的位置
            val_pos = point_pos[val_idx]
            cat_pos = point_pos[cat_idx]

            # 断言分类轴位置四舍五入后等于当前索引 i
            assert_array_equal(cat_pos.round(), i)
            # 断言值轴位置等于 wide_df 中对应键的值
            assert_array_equal(val_pos, wide_df[key])

            # 遍历点集合对象的面颜色
            for point_color in points.get_facecolors():
                # 断言点的面颜色转换为 RGBA 后与指定颜色 "C0" 相等
                assert tuple(point_color) == to_rgba("C0")

    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_flat(self, flat_series, orient):
        # 调用指定的绘图函数，传入数据和方向参数，并获取绘图对象 ax
        ax = self.func(data=flat_series, orient=orient)
        # 调用内部函数 _draw_figure，传入 ax 对象的图形属性
        _draw_figure(ax.figure)

        # 确定分类轴和值轴的索引位置
        cat_idx = ["v", "h"].index(orient)
        val_idx = int(not cat_idx)

        # 获取第一个集合对象中的点集合对象
        points = ax.collections[0]
        # 获取点集合对象的偏移位置
        pos = points.get_offsets().T

        # 断言分类轴位置四舍五入后等于零向量
        assert_array_equal(pos[cat_idx].round(), np.zeros(len(flat_series)))
        # 断言值轴位置等于 flat_series 的值
        assert_array_equal(pos[val_idx], flat_series)

    @pytest.mark.parametrize(
        "variables,orient",
        [
            # 不同的变量组合和方向参数，用于绘图函数的参数化测试
            ({"cat": "a", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "a", "hue": None}, None),
            ({"cat": "a", "val": "y", "hue": "a"}, None),
            ({"val": "y", "cat": "a", "hue": "a"}, None),
            ({"cat": "a", "val": "y", "hue": "b"}, None),
            ({"val": "y", "cat": "a", "hue": "x"}, None),
            ({"cat": "s", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "s", "hue": None}, "h"),
            ({"cat": "a", "val": "b", "hue": None}, None),
            ({"val": "a", "cat": "b", "hue": None}, "h"),
            ({"cat": "a", "val": "t", "hue": None}, None),
            ({"val": "t", "cat": "a", "hue": None}, None),
            ({"cat": "d", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "d", "hue": None}, None),
            ({"cat": "a_cat", "val": "y", "hue": None}, None),
            ({"val": "y", "cat": "s_cat", "hue": None}, None),
        ],
    )
    # 测试函数，用于验证长格式数据框中不同变量的位置关系
    def test_positions(self, long_df, variables, orient):
        
        # 从变量字典中提取分类变量、数值变量和色调变量
        cat_var = variables["cat"]
        val_var = variables["val"]
        hue_var = variables["hue"]
        
        # 将变量名列表化
        var_names = list(variables.values())
        
        # 分别提取 x 轴和 y 轴的变量名
        x_var, y_var, *_ = var_names
        
        # 调用给定的绘图函数，传入数据、x 轴、y 轴、色调和方向参数，并返回绘图对象 ax
        ax = self.func(
            data=long_df, x=x_var, y=y_var, hue=hue_var, orient=orient,
        )
        
        # 绘制图形
        _draw_figure(ax.figure)
        
        # 获取分类变量和数值变量在变量名列表中的索引位置
        cat_idx = var_names.index(cat_var)
        val_idx = var_names.index(val_var)
        
        # 获取 x 轴和 y 轴的对象
        axis_objs = ax.xaxis, ax.yaxis
        cat_axis = axis_objs[cat_idx]
        val_axis = axis_objs[val_idx]
        
        # 获取长格式数据框中的分类数据和其顺序
        cat_data = long_df[cat_var]
        cat_levels = categorical_order(cat_data)
        
        # 遍历分类水平和索引
        for i, label in enumerate(cat_levels):
            
            # 获取特定分类标签下的数值数据
            vals = long_df.loc[cat_data == label, val_var]
            
            # 获取绘图对象中第 i 组数据点的偏移量
            points = ax.collections[i].get_offsets().T
            
            # 获取分类变量和数值变量在变量名列表中的位置
            cat_pos = points[var_names.index(cat_var)]
            val_pos = points[var_names.index(val_var)]
            
            # 检查数值位置是否与数值轴的单位转换后相等
            assert_array_equal(val_pos, val_axis.convert_units(vals))
            
            # 检查分类位置是否与当前索引 i 的四舍五入值相等
            assert_array_equal(cat_pos.round(), i)
            
            # 检查分类位置的范围是否在 [0, 0.8] 之间
            assert 0 <= np.ptp(cat_pos) <= .8
            
            # 确保分类轴的主刻度标签文本与当前标签值相匹配
            label = pd.Index([label]).astype(str)[0]
            assert cat_axis.get_majorticklabels()[i].get_text() == label
    
    # 使用参数化装饰器，对测试函数 test_positions_dodged 进行多次参数化测试
    @pytest.mark.parametrize(
        "variables",
        [
            # 按顺序分配给 x/y 的变量
            {"cat": "a", "val": "y", "hue": "b"},
            {"val": "y", "cat": "a", "hue": "c"},
            {"cat": "a", "val": "y", "hue": "f"},
        ],
    )
    # 测试函数，用于验证长格式数据框中不同变量的位置关系（dodge 模式）
    def test_positions_dodged(self, long_df, variables):
        
        # 从变量字典中提取分类变量、数值变量和色调变量
        cat_var = variables["cat"]
        val_var = variables["val"]
        hue_var = variables["hue"]
        
        # 将变量名列表化
        var_names = list(variables.values())
        
        # 分别提取 x 轴和 y 轴的变量名
        x_var, y_var, *_ = var_names
        
        # 调用给定的绘图函数，传入数据、x 轴、y 轴、色调和 dodge=True 参数，并返回绘图对象 ax
        ax = self.func(
            data=long_df, x=x_var, y=y_var, hue=hue_var, dodge=True,
        )
        
        # 获取分类变量和色调变量的顺序
        cat_vals = categorical_order(long_df[cat_var])
        hue_vals = categorical_order(long_df[hue_var])
        
        # 计算色调数量
        n_hue = len(hue_vals)
        
        # 计算每个色调的偏移量，并调整至均值为零
        offsets = np.linspace(0, .8, n_hue + 1)[:-1]
        offsets -= offsets.mean()
        
        # 计算每组嵌套宽度
        nest_width = .8 / n_hue
        
        # 遍历分类变量和色调变量的值
        for i, cat_val in enumerate(cat_vals):
            for j, hue_val in enumerate(hue_vals):
                
                # 选取特定分类变量和色调变量值对应的行
                rows = (long_df[cat_var] == cat_val) & (long_df[hue_var] == hue_val)
                
                # 获取特定行数值变量的数据
                vals = long_df.loc[rows, val_var]
                
                # 获取绘图对象中第 n_hue*i+j 组数据点的偏移量
                points = ax.collections[n_hue * i + j].get_offsets().T
                
                # 获取分类变量和数值变量在变量名列表中的位置
                cat_pos = points[var_names.index(cat_var)]
                val_pos = points[var_names.index(val_var)]
                
                # 如果数值数据是日期时间类型，则转换为 matplotlib 的日期序号
                if pd.api.types.is_datetime64_any_dtype(vals):
                    vals = mpl.dates.date2num(vals)
                
                # 检查数值位置是否与 vals 相等
                assert_array_equal(val_pos, vals)
                
                # 检查分类位置是否与当前索引 i 的四舍五入值相等
                assert_array_equal(cat_pos.round(), i)
                
                # 检查分类位置是否在 (i+offsets[j])/nest_width 的范围内
                assert_array_equal((cat_pos - (i + offsets[j])).round() / nest_width, 0)
                
                # 检查分类位置的范围是否在 [0, nest_width] 之间
                assert 0 <= np.ptp(cat_pos) <= nest_width
    
    # 使用参数化装饰器，对测试函数 test_positions_dodged 进行参数化测试，测试分类变量 cat_var 的不同值
    @pytest.mark.parametrize("cat_var", ["a", "s", "d"])
    def test_positions_unfixed(self, long_df, cat_var):
        # 对长格式数据按照指定的分类变量进行排序
        long_df = long_df.sort_values(cat_var)

        # 设置绘图参数，初始化为空字典
        kws = dict(size=.001)
        
        # 如果函数名包含 "stripplot"，则不能使用 __name__ 获取部分函数名，禁用抖动效果
        if "stripplot" in str(self.func):
            kws["jitter"] = False

        # 使用给定的数据绘制图形，并返回轴对象
        ax = self.func(data=long_df, x=cat_var, y="y", native_scale=True, **kws)

        # 遍历按分类变量分组的数据
        for i, (cat_level, cat_data) in enumerate(long_df.groupby(cat_var)):
            # 获取绘图对象中第 i 个集合的偏移点集合，并转置
            points = ax.collections[i].get_offsets().T
            cat_pos = points[0]  # 提取分类位置信息
            val_pos = points[1]  # 提取数值位置信息

            # 断言数值位置与数据中的 y 值数组相等
            assert_array_equal(val_pos, cat_data["y"])

            # 获取分类级别在 x 轴上的位置，并断言分类位置与计算的位置相近
            comp_level = np.squeeze(ax.xaxis.convert_units(cat_level)).item()
            assert_array_equal(cat_pos.round(), comp_level)

    @pytest.mark.parametrize(
        "x_type,order",
        [
            (str, None),
            (str, ["a", "b", "c"]),
            (str, ["c", "a"]),
            (str, ["a", "b", "c", "d"]),
            (int, None),
            (int, [3, 1, 2]),
            (int, [3, 1]),
            (int, [1, 2, 3, 4]),
            (int, ["3", "1", "2"]),
        ]
    )
    def test_order(self, x_type, order):
        # 根据 x_type 类型设置 x 和 y 的值
        if x_type is str:
            x = ["b", "a", "c"]
        else:
            x = [2, 1, 3]
        y = [1, 2, 3]

        # 使用给定的数据绘制图形，并返回轴对象
        ax = self.func(x=x, y=y, order=order)
        _draw_figure(ax.figure)  # 绘制图形

        # 如果 order 为 None，则使用默认顺序 x，并在 x_type 为 int 时进行排序
        if order is None:
            order = x
            if x_type is int:
                order = np.sort(order)

        # 断言集合数量与 order 的长度相等
        assert len(ax.collections) == len(order)
        # 获取 x 轴的主要刻度标签
        tick_labels = ax.xaxis.get_majorticklabels()

        # 断言 x 轴的上限值与 (order 长度 - 0.5) 相等
        assert ax.get_xlim()[1] == (len(order) - .5)

        # 遍历每个集合，并断言刻度标签与预期的分类值相等
        for i, points in enumerate(ax.collections):
            cat = order[i]
            assert tick_labels[i].get_text() == str(cat)

            # 获取集合的偏移点集合，并根据分类值检查数值是否正确
            positions = points.get_offsets()
            if x_type(cat) in x:
                val = y[x.index(x_type(cat))]
                assert positions[0, 1] == val
            else:
                assert not positions.size  # 集合为空时断言

    @pytest.mark.parametrize("hue_var", ["a", "b"])
    def test_hue_categorical(self, long_df, hue_var):
        # 设置分类变量
        cat_var = "b"

        # 获取长格式数据中 hue_var 列的分类顺序
        hue_levels = categorical_order(long_df[hue_var])
        # 获取长格式数据中 cat_var 列的分类顺序
        cat_levels = categorical_order(long_df[cat_var])

        # 设置调色板名称和调色板字典
        pal_name = "muted"
        palette = dict(zip(hue_levels, color_palette(pal_name)))
        # 使用给定的数据绘制图形，并返回轴对象
        ax = self.func(data=long_df, x=cat_var, y="y", hue=hue_var, palette=pal_name)

        # 遍历每个 cat_levels 中的分类级别
        for i, level in enumerate(cat_levels):
            # 从长格式数据中提取特定分类级别的子数据集
            sub_df = long_df[long_df[cat_var] == level]
            # 获取子数据集中的 hue_var 值
            point_hues = sub_df[hue_var]

            # 获取绘图对象中第 i 个集合对象
            points = ax.collections[i]
            # 获取集合对象的面颜色
            point_colors = points.get_facecolors()

            # 断言 hue 值的长度与集合对象面颜色的长度相等
            assert len(point_hues) == len(point_colors)

            # 遍历每个 hue 值及其对应的颜色，断言颜色与调色板中的颜色一致
            for hue, color in zip(point_hues, point_colors):
                assert tuple(color) == to_rgba(palette[hue])

    @pytest.mark.parametrize("hue_var", ["a", "b"])
    @pytest.mark.parametrize(
        "val_var,val_col,hue_col",
        list(itertools.product(["x", "y"], ["b", "y", "t"], [None, "a"])),
    )


# 使用 pytest 的参数化标记，定义了多组参数化测试参数：
# - val_var: 可以是"x"或"y"
# - val_col: 可以是"b"、"y"或"t"
# - hue_col: 可以是None或"a"
def test_single(self, long_df, val_var, val_col, hue_col):

    # 根据参数设置变量字典
    var_kws = {val_var: val_col, "hue": hue_col}
    # 调用被测试的函数，并传入数据 long_df 和上述变量字典
    ax = self.func(data=long_df, **var_kws)
    # 调用辅助函数 _draw_figure 处理绘图对象
    _draw_figure(ax.figure)

    # 定义轴变量顺序
    axis_vars = ["x", "y"]
    # 获取当前 val_var 在 axis_vars 中的索引
    val_idx = axis_vars.index(val_var)
    # 计算其余轴的索引
    cat_idx = int(not val_idx)
    # 获取相应的轴变量名
    cat_var = axis_vars[cat_idx]

    # 获取绘图对象 ax 中的第一个集合对象（通常是点集合）
    points = ax.collections[0]
    # 获取点集合的位置信息并转置
    point_pos = points.get_offsets().T
    # 获取类别轴的位置信息
    cat_pos = point_pos[cat_idx]
    # 获取数值轴的位置信息
    val_pos = point_pos[val_idx]

    # 断言类别轴位置四舍五入后为零
    assert_array_equal(cat_pos.round(), 0)
    # 断言类别轴位置的最大值小于等于0.4
    assert cat_pos.max() <= 0.4
    # 断言类别轴位置的最小值大于等于-0.4
    assert cat_pos.min() >= -0.4

    # 将 long_df[val_col] 转换为数值轴上的数值
    num_vals = val_axis.convert_units(long_df[val_col])
    # 断言数值轴上的位置与转换后的数值一致
    assert_array_equal(val_pos, num_vals)

    # 如果 hue_col 不为 None，则进行以下操作
    if hue_col is not None:
        # 创建调色板，将长数据框 long_df 中 hue_col 列的值作为键，颜色调色板中的颜色作为值
        palette = dict(zip(
            categorical_order(long_df[hue_col]), color_palette()
        ))

    # 获取点集合的面颜色
    facecolors = points.get_facecolors()
    # 遍历面颜色
    for i, color in enumerate(facecolors):
        # 如果 hue_col 为 None，则断言颜色为 "C0"
        if hue_col is None:
            assert tuple(color) == to_rgba("C0")
        else:
            # 否则，获取当前行的 hue_col 值，并断言颜色与调色板中对应颜色一致
            hue_level = long_df.loc[i, hue_col]
            expected_color = palette[hue_level]
            assert tuple(color) == to_rgba(expected_color)

    # 获取类别轴的主刻度标签
    ticklabels = cat_axis.get_majorticklabels()
    # 断言主刻度标签的数量为1
    assert len(ticklabels) == 1
    # 断言主刻度标签的文本为空字符串
    assert not ticklabels[0].get_text()


    def test_attributes(self, long_df):
        # 定义参数字典 kwargs
        kwargs = dict(
            size=2,
            linewidth=1,
            edgecolor="C2",
        )

        # 调用被测试函数，并传入数据 long_df 和参数字典 kwargs
        ax = self.func(x=long_df["y"], **kwargs)
        # 获取绘图对象中的点集合
        points, = ax.collections

        # 断言点集合的大小为 size^2
        assert points.get_sizes().item() == kwargs["size"] ** 2
        # 断言点集合的线宽为 linewidth
        assert points.get_linewidths().item() == kwargs["linewidth"]
        # 断言点集合的边缘颜色为 kwargs["edgecolor"] 对应的 RGBA 颜色
        assert tuple(points.get_edgecolors().squeeze()) == to_rgba(kwargs["edgecolor"])
    # 定义一个测试方法，用于验证绘图函数在三个点上的表现
    def test_three_points(self):
        # 创建一个包含三个元素的数组
        x = np.arange(3)
        # 调用被测试的绘图函数，并传入数组 x 作为参数，获取返回的绘图对象
        ax = self.func(x=x)
        # 遍历绘图对象的第一个集合的填充颜色，验证其是否与颜色 "C0" 相同
        for point_color in ax.collections[0].get_facecolor():
            assert tuple(point_color) == to_rgba("C0")

    # 定义一个测试方法，用于验证绘图函数在具有分类变量的情况下的图例表现
    def test_legend_categorical(self, long_df):
        # 调用绘图函数，并传入长格式数据框、x 列为 "y"、y 列为 "a"、hue 列为 "b"，获取返回的绘图对象
        ax = self.func(data=long_df, x="y", y="a", hue="b")
        # 获取图例对象中文本的列表
        legend_texts = [t.get_text() for t in ax.legend_.texts]
        # 获取预期的分类顺序
        expected = categorical_order(long_df["b"])
        # 断言获取的图例文本列表与预期的分类顺序列表相同
        assert legend_texts == expected

    # 定义一个测试方法，用于验证绘图函数在具有数值变量的情况下的图例表现
    def test_legend_numeric(self, long_df):
        # 调用绘图函数，并传入长格式数据框、x 列为 "y"、y 列为 "a"、hue 列为 "z"，获取返回的绘图对象
        ax = self.func(data=long_df, x="y", y="a", hue="z")
        # 获取图例对象中文本的数值列表
        vals = [float(t.get_text()) for t in ax.legend_.texts]
        # 断言数值列表中相邻数值之间的差值近似相等
        assert (vals[1] - vals[0]) == approx(vals[2] - vals[1])

    # 定义一个测试方法，用于验证绘图函数在具有属性参数的图例表现
    def test_legend_attributes(self, long_df):
        # 定义属性参数字典
        kws = {"edgecolor": "r", "linewidth": 1}
        # 调用绘图函数，并传入长格式数据框、x 列为 "x"、y 列为 "y"、hue 列为 "a"，以及额外的属性参数，获取返回的绘图对象
        ax = self.func(data=long_df, x="x", y="y", hue="a", **kws)
        # 获取图例句柄对象，并遍历检查其边缘颜色是否与预期一致，以及边缘宽度是否与预期一致
        for pt in get_legend_handles(ax.get_legend()):
            assert same_color(pt.get_markeredgecolor(), kws["edgecolor"])
            assert pt.get_markeredgewidth() == kws["linewidth"]

    # 定义一个测试方法，用于验证绘图函数在禁用图例时的表现
    def test_legend_disabled(self, long_df):
        # 调用绘图函数，并传入长格式数据框、x 列为 "y"、y 列为 "a"、hue 列为 "b"，并禁用图例，获取返回的绘图对象
        ax = self.func(data=long_df, x="y", y="a", hue="b", legend=False)
        # 断言图例对象为空
        assert ax.legend_ is None

    # 定义一个测试方法，用于验证绘图函数在使用颜色创建调色板时的表现
    def test_palette_from_color_deprecation(self, long_df):
        # 定义颜色元组和其对应的十六进制颜色代码
        color = (.9, .4, .5)
        hex_color = mpl.colors.to_hex(color)
        # 定义 hue 变量并计算其唯一值的数量
        hue_var = "a"
        n_hue = long_df[hue_var].nunique()
        # 使用颜色创建渐变调色板
        palette = color_palette(f"dark:{hex_color}", n_hue)
        # 在测试中使用未来警告，检查绘图函数在给定数据和参数的情况下的表现
        with pytest.warns(FutureWarning, match="Setting a gradient palette"):
            ax = self.func(data=long_df, x="z", hue=hue_var, color=color)
        # 获取绘图对象中第一个集合的所有点，并检查它们的颜色是否在预期的调色板中
        points = ax.collections[0]
        for point_color in points.get_facecolors():
            assert to_rgb(point_color) in palette

    # 定义一个测试方法，用于验证绘图函数在使用 hue 参数创建调色板时的表现
    def test_palette_with_hue_deprecation(self, long_df):
        # 定义调色板名称
        palette = "Blues"
        # 在测试中使用未来警告，检查绘图函数在给定数据和参数的情况下的表现
        with pytest.warns(FutureWarning, match="Passing `palette` without"):
            ax = self.func(data=long_df, x="a", y=long_df["y"], palette=palette)
        # 获取绘图对象中的所有条带，并获取它们的填充颜色
        strips = ax.collections
        # 使用指定的调色板创建颜色列表
        colors = color_palette(palette, len(strips))
        # 遍历每个条带和对应的颜色，检查它们的颜色是否匹配
        for strip, color in zip(strips, colors):
            assert same_color(strip.get_facecolor()[0], color)

    # 定义一个测试方法，用于验证绘图函数在对数尺度下的表现
    def test_log_scale(self):
        # 定义 x 值列表
        x = [1, 10, 100, 1000]
        # 创建一个新的图形对象并获取其子图对象
        ax = plt.figure().subplots()
        # 设置子图对象的 x 轴为对数尺度
        ax.set_xscale("log")
        # 调用绘图函数，并传入 x 值列表作为参数
        self.func(x=x)
        # 获取绘图对象中第一个集合的偏移值，并提取其 x 轴坐标值
        vals = ax.collections[0].get_offsets()[:, 0]
        # 断言获取的 x 轴坐标值与输入的 x 值列表相等
        assert_array_equal(x, vals)

        # 定义新的 y 值列表
        y = [1, 2, 3, 4]
        # 创建一个新的图形对象并获取其子图对象
        ax = plt.figure().subplots()
        # 设置子图对象的 x 轴为对数尺度
        ax.set_xscale("log")
        # 调用绘图函数，并传入 x 和 y 值列表作为参数，以及原生尺度标志
        self.func(x=x, y=y, native_scale=True)
        # 遍历绘图对象的每个集合，并检查其偏移值中第一个点的 x 轴坐标值
        for i, point in enumerate(ax.collections):
            val = point.get_offsets()[0, 0]
            # 断言获取的 x 轴坐标值近似等于输入的 x 值列表中对应位置的值
            assert val == approx(x[i])

        # 将 x 和 y 值列表均设为包含一百个 1 的数组
        x = y = np.ones(100)
        # 创建一个新的图形对象并获取其子图对象
        ax = plt.figure().subplots()
        # 设置子图对象的 y 轴为对数尺度
        ax.set_yscale("log")
        # 调用绘图函数，并传入 x 和 y 值列表作为参数，方向设为水平，以及
    @pytest.mark.parametrize(
        "kwargs",
        [  # 参数化测试：定义不同的kwargs字典作为测试输入
            dict(data="wide"),  # 参数字典1：指定"data"为"wide"
            dict(data="wide", orient="h"),  # 参数字典2：指定"data"为"wide"，并设置"orient"为"h"
            dict(data="long", x="x", color="C3"),  # 参数字典3：指定"data"为"long"，并设置"x"为"x"，"color"为"C3"
            dict(data="long", y="y", hue="a", jitter=False),  # 参数字典4：指定"data"为"long"，并设置"y"为"y"，"hue"为"a"，"jitter"为False
            dict(data="long", x="a", y="y", hue="z", edgecolor="w", linewidth=.5),  # 参数字典5：指定"data"为"long"，并设置"x"为"a"，"y"为"y"，"hue"为"z"，"edgecolor"为"w"，"linewidth"为0.5
            dict(data="long", x="a", y="y", hue="z", edgecolor="auto", linewidth=.5),  # 参数字典6：指定"data"为"long"，并设置"x"为"a"，"y"为"y"，"hue"为"z"，"edgecolor"为"auto"，"linewidth"为0.5
            dict(data="long", x="a_cat", y="y", hue="z"),  # 参数字典7：指定"data"为"long"，并设置"x"为"a_cat"，"y"为"y"，"hue"为"z"
            dict(data="long", x="y", y="s", hue="c", orient="h", dodge=True),  # 参数字典8：指定"data"为"long"，并设置"x"为"y"，"y"为"s"，"hue"为"c"，"orient"为"h"，"dodge"为True
            dict(data="long", x="s", y="y", hue="c", native_scale=True),  # 参数字典9：指定"data"为"long"，并设置"x"为"s"，"y"为"y"，"hue"为"c"，"native_scale"为True
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, kwargs):
        kwargs = kwargs.copy()  # 复制参数字典以避免修改原始参数

        if kwargs["data"] == "long":
            kwargs["data"] = long_df  # 如果"data"为"long"，则将"data"替换为long_df数据框
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df  # 如果"data"为"wide"，则将"data"替换为wide_df数据框

        try:
            name = self.func.__name__[:-4]  # 尝试获取函数名，去掉末尾的"_test"
        except AttributeError:
            name = self.func.func.__name__[:-4]  # 如果出现AttributeError，则尝试获取函数的函数名，去掉末尾的"_test"

        if name == "swarm":
            kwargs.pop("jitter", None)  # 如果函数名为"swarm"，则从kwargs中移除"jitter"参数，如果不存在则不做任何操作

        np.random.seed(0)  # 设置随机种子为0，用于控制随机化行为

        ax = self.func(**kwargs)  # 调用被测试函数，传入kwargs参数，返回绘图的Axes对象

        np.random.seed(0)  # 再次设置随机种子为0，确保测试条件一致

        g = catplot(**kwargs, kind=name)  # 调用catplot函数，传入kwargs和kind参数为name，返回绘图对象g

        assert_plots_equal(ax, g.ax)  # 断言：验证两个绘图的Axes对象是否相等

    def test_empty_palette(self):
        self.func(x=[], y=[], hue=[], palette=[])  # 调用被测试函数，传入空列表作为参数
class SharedAggTests(SharedAxesLevelTests):

    def test_labels_flat(self):

        ind = pd.Index(["a", "b", "c"], name="x")  # 创建一个带有名称的索引对象
        ser = pd.Series([1, 2, 3], ind, name="y")  # 创建一个带有名称的序列对象

        ax = self.func(ser)  # 调用被测试函数，返回一个Axes对象

        # 在较旧版本的matplotlib中，用于填充文本；在新版本中不需要
        _draw_figure(ax.figure)

        assert ax.get_xlabel() == ind.name  # 断言X轴标签与索引名称相同
        assert ax.get_ylabel() == ser.name  # 断言Y轴标签与序列名称相同
        labels = [t.get_text() for t in ax.get_xticklabels()]  # 获取X轴刻度标签的文本内容
        for label, level in zip(labels, ind):
            assert label == level  # 断言每个标签与对应的索引级别相同


class SharedPatchArtistTests:

    @pytest.mark.parametrize("fill", [True, False])
    def test_legend_fill(self, long_df, fill):

        palette = color_palette()  # 获取调色板
        ax = self.func(
            long_df, x="x", y="y", hue="a",
            saturation=1, linecolor="k", fill=fill,
        )  # 调用被测试函数，返回一个Axes对象
        for i, patch in enumerate(get_legend_handles(ax.get_legend())):
            fc = patch.get_facecolor()  # 获取图例条目的填充颜色
            ec = patch.get_edgecolor()  # 获取图例条目的边界颜色
            if fill:
                assert same_color(fc, palette[i])  # 断言填充颜色与调色板中的颜色相同
                assert same_color(ec, "k")  # 断言边界颜色为黑色
            else:
                assert fc == (0, 0, 0, 0)  # 断言无填充颜色
                assert same_color(ec, palette[i])  # 断言边界颜色与调色板中的颜色相同

    def test_legend_attributes(self, long_df):

        ax = self.func(long_df, x="x", y="y", hue="a", linewidth=3)  # 调用被测试函数，返回一个Axes对象
        for patch in get_legend_handles(ax.get_legend()):
            assert patch.get_linewidth() == 3  # 断言图例条目的线宽为3


class TestStripPlot(SharedScatterTests):

    func = staticmethod(stripplot)  # 设置被测试函数为静态方法

    def test_jitter_unfixed(self, long_df):

        ax1, ax2 = plt.figure().subplots(2)  # 创建包含两个子图的Figure对象
        kws = dict(data=long_df, x="y", orient="h", native_scale=True)  # 创建关键字参数字典

        np.random.seed(0)
        stripplot(**kws, y="s", ax=ax1)  # 在第一个子图上绘制带有jitter的条带图

        np.random.seed(0)
        stripplot(**kws, y=long_df["s"] * 2, ax=ax2)  # 在第二个子图上绘制带有jitter的条带图

        p1 = ax1.collections[0].get_offsets()[1]  # 获取第一个子图中第一个集合的偏移量
        p2 = ax2.collections[0].get_offsets()[1]  # 获取第二个子图中第一个集合的偏移量

        assert p2.std() > p1.std()  # 断言第二个子图中数据的标准差大于第一个子图中数据的标准差

    @pytest.mark.parametrize(
        "orient,jitter",
        itertools.product(["v", "h"], [True, .1]),
    )
    # 定义一个测试方法，用于测试 jitter 功能在长格式数据中的效果
    def test_jitter(self, long_df, orient, jitter):

        # 设置类别变量和数值变量的默认值
        cat_var, val_var = "a", "y"

        # 根据 orient 参数确定 x 和 y 变量的设置
        if orient == "x":
            x_var, y_var = cat_var, val_var
            cat_idx, val_idx = 0, 1
        else:
            x_var, y_var = val_var, cat_var
            cat_idx, val_idx = 1, 0

        # 获取类别变量的分类顺序
        cat_vals = categorical_order(long_df[cat_var])

        # 创建一个 stripplot 对象，用于绘制数据分布图
        ax = stripplot(
            data=long_df, x=x_var, y=y_var, jitter=jitter,
        )

        # 根据 jitter 参数设置 jitter 的范围
        if jitter is True:
            jitter_range = .4
        else:
            jitter_range = 2 * jitter

        # 对每一个类别进行遍历
        for i, level in enumerate(cat_vals):

            # 获取特定类别下的数值变量的值
            vals = long_df.loc[long_df[cat_var] == level, val_var]

            # 获取当前类别的数据点的坐标
            points = ax.collections[i].get_offsets().T
            cat_points = points[cat_idx]
            val_points = points[val_idx]

            # 断言数值变量的值和绘图中的对应数据点值相等
            assert_array_equal(val_points, vals)

            # 断言类别变量的数据点的标准差大于零
            assert np.std(cat_points) > 0

            # 断言类别变量的数据点的极差小于等于 jitter 范围
            assert np.ptp(cat_points) <= jitter_range
class TestSwarmPlot(SharedScatterTests):

    # 使用 swarmplot 函数的静态方法，并设置 warn_thresh 参数为 1
    func = staticmethod(partial(swarmplot, warn_thresh=1))


class TestBoxPlot(SharedAxesLevelTests, SharedPatchArtistTests):

    # 使用 boxplot 函数作为测试函数
    func = staticmethod(boxplot)

    @pytest.fixture
    def common_kws(self):
        # 返回一个字典，包含 saturation 键值对，saturation 值为 1
        return {"saturation": 1}

    def get_last_color(self, ax):
        # 获取最后一个容器的所有箱子的面颜色，并返回
        colors = [b.get_facecolor() for b in ax.containers[-1].boxes]
        unique_colors = np.unique(colors, axis=0)
        assert len(unique_colors) == 1
        return to_rgba(unique_colors.squeeze())

    def get_box_verts(self, box):
        # 获取箱子对象的路径，筛选出 MOVETO 和 LINETO 类型的路径点，并返回转置后的顶点坐标
        path = box.get_path()
        visible_codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO]
        visible = np.isin(path.codes, visible_codes)
        return path.vertices[visible].T

    def check_box(self, bxp, data, orient, pos, width=0.8):
        # 获取指定方向上的索引，计算数据的 25th、50th、75th 百分位数，并验证箱子的边界和中位数
        pos_idx, val_idx = self.orient_indices(orient)

        p25, p50, p75 = np.percentile(data, [25, 50, 75])

        box = self.get_box_verts(bxp.box)
        assert box[val_idx].min() == approx(p25, 1e-3)
        assert box[val_idx].max() == approx(p75, 1e-3)
        assert box[pos_idx].min() == approx(pos - width / 2)
        assert box[pos_idx].max() == approx(pos + width / 2)

        med = bxp.median.get_xydata().T
        assert np.allclose(med[val_idx], (p50, p50), rtol=1e-3)
        assert np.allclose(med[pos_idx], (pos - width / 2, pos + width / 2))

    def check_whiskers(self, bxp, data, orient, pos, capsize=0.4, whis=1.5):
        # 获取指定方向上的索引，验证箱须图的边界和异常值
        pos_idx, val_idx = self.orient_indices(orient)

        whis_lo = bxp.whiskers[0].get_xydata().T
        whis_hi = bxp.whiskers[1].get_xydata().T
        caps_lo = bxp.caps[0].get_xydata().T
        caps_hi = bxp.caps[1].get_xydata().T
        fliers = bxp.fliers.get_xydata().T

        p25, p75 = np.percentile(data, [25, 75])
        iqr = p75 - p25

        adj_lo = data[data >= (p25 - iqr * whis)].min()
        adj_hi = data[data <= (p75 + iqr * whis)].max()

        assert whis_lo[val_idx].max() == approx(p25, 1e-3)
        assert whis_lo[val_idx].min() == approx(adj_lo)
        assert np.allclose(whis_lo[pos_idx], (pos, pos))
        assert np.allclose(caps_lo[val_idx], (adj_lo, adj_lo))
        assert np.allclose(caps_lo[pos_idx], (pos - capsize / 2, pos + capsize / 2))

        assert whis_hi[val_idx].min() == approx(p75, 1e-3)
        assert whis_hi[val_idx].max() == approx(adj_hi)
        assert np.allclose(whis_hi[pos_idx], (pos, pos))
        assert np.allclose(caps_hi[val_idx], (adj_hi, adj_hi))
        assert np.allclose(caps_hi[pos_idx], (pos - capsize / 2, pos + capsize / 2))

        flier_data = data[(data < adj_lo) | (data > adj_hi)]
        assert sorted(fliers[val_idx]) == sorted(flier_data)
        assert np.allclose(fliers[pos_idx], pos)

    @pytest.mark.parametrize("orient,col", [("x", "y"), ("y", "z")])
    # 使用参数化装饰器，定义测试函数的参数化输入为 orient 和 col，分别为 "x", "y" 和 "y", "z"
    # 使用给定的方向和列进行单变量测试
    def test_single_var(self, long_df, orient, col):

        # 根据方向选择正确的变量，用于绘制图表
        var = {"x": "y", "y": "x"}[orient]
        # 调用箱线图函数绘制图表，并传入相应的参数
        ax = boxplot(long_df, **{var: col})
        # 获取第一个箱线图容器
        bxp = ax.containers[0][0]
        # 检查箱体的数据是否符合预期
        self.check_box(bxp, long_df[col], orient, 0)
        # 检查图表的触须数据是否符合预期
        self.check_whiskers(bxp, long_df[col], orient, 0)

    # 使用参数化测试向量数据
    @pytest.mark.parametrize("orient,col", [(None, "x"), ("x", "y"), ("y", "z")])
    def test_vector_data(self, long_df, orient, col):

        # 根据方向绘制箱线图，并选择正确的变量
        ax = boxplot(long_df[col], orient=orient)
        # 如果方向为None，则设定为"x"
        orient = "x" if orient is None else orient
        # 获取第一个箱线图容器
        bxp = ax.containers[0][0]
        # 检查箱体的数据是否符合预期
        self.check_box(bxp, long_df[col], orient, 0)
        # 检查图表的触须数据是否符合预期
        self.check_whiskers(bxp, long_df[col], orient, 0)

    # 使用参数化测试宽数据
    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_wide_data(self, wide_df, orient):

        # 根据方向选择正确的变量
        orient = {"h": "y", "v": "x"}[orient]
        # 调用箱线图函数绘制图表，并传入相应的参数
        ax = boxplot(wide_df, orient=orient, color="C0")
        # 遍历每个箱线图容器
        for i, bxp in enumerate(ax.containers):
            # 获取当前列的名称
            col = wide_df.columns[i]
            # 检查箱体的数据是否符合预期
            self.check_box(bxp[i], wide_df[col], orient, i)
            # 检查图表的触须数据是否符合预期
            self.check_whiskers(bxp[i], wide_df[col], orient, i)

    # 使用参数化测试分组数据
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_grouped(self, long_df, orient):

        # 根据方向选择正确的变量和值
        value = {"x": "y", "y": "x"}[orient]
        # 调用箱线图函数绘制图表，并传入相应的参数
        ax = boxplot(long_df, **{orient: "a", value: "z"})
        # 获取第一个箱线图容器
        bxp, = ax.containers
        # 获取长数据中"a"列的分类顺序
        levels = categorical_order(long_df["a"])
        # 遍历分类级别
        for i, level in enumerate(levels):
            # 根据条件选择数据
            data = long_df.loc[long_df["a"] == level, "z"]
            # 检查箱体的数据是否符合预期
            self.check_box(bxp[i], data, orient, i)
            # 检查图表的触须数据是否符合预期
            self.check_whiskers(bxp[i], data, orient, i)

    # 使用参数化测试分组数据和色调
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_hue_grouped(self, long_df, orient):

        # 根据方向选择正确的变量和值
        value = {"x": "y", "y": "x"}[orient]
        # 调用箱线图函数绘制图表，并传入相应的参数
        ax = boxplot(long_df, hue="c", **{orient: "a", value: "z"})
        # 遍历每个色调水平
        for i, hue_level in enumerate(categorical_order(long_df["c"])):
            # 获取当前箱线图容器
            bxp = ax.containers[i]
            # 遍历每个分组水平
            for j, level in enumerate(categorical_order(long_df["a"])):
                # 根据条件选择数据
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                # 设置箱体的位置偏移量、宽度和误差线的大小
                pos = j + [-.2, +.2][i]
                width, capsize = 0.4, 0.2
                # 检查箱体的数据是否符合预期
                self.check_box(bxp[j], data, orient, pos, width)
                # 检查图表的触须数据是否符合预期
                self.check_whiskers(bxp[j], data, orient, pos, capsize)

    # 使用参数化测试色调未分组
    def test_hue_not_dodged(self, long_df):

        # 获取长数据中"b"列的分类顺序
        levels = categorical_order(long_df["b"])
        # 根据条件选择数据
        hue = long_df["b"].isin(levels[:2])
        # 调用箱线图函数绘制图表，并传入相应的参数
        ax = boxplot(long_df, x="b", y="z", hue=hue)
        # 获取所有箱线图容器
        bxps = ax.containers
        # 遍历每个级别
        for i, level in enumerate(levels):
            # 根据条件选择数据
            idx = int(i < 2)
            data = long_df.loc[long_df["b"] == level, "z"]
            # 检查箱体的数据是否符合预期
            self.check_box(bxps[idx][i % 2], data, "x", i)
            # 检查图表的触须数据是否符合预期
            self.check_whiskers(bxps[idx][i % 2], data, "x", i)
    # 测试用例：使用原生缩放设置绘制箱线图，传入长格式数据框
    def test_dodge_native_scale(self, long_df):
        # 获取分类变量"s"的顺序
        centers = categorical_order(long_df["s"])
        # 获取色调变量"c"的顺序
        hue_levels = categorical_order(long_df["c"])
        # 计算中心点之间的最小间距
        spacing = min(np.diff(centers))
        # 计算每个箱线图的宽度
        width = 0.8 * spacing / len(hue_levels)
        # 计算偏移量
        offset = width / len(hue_levels)
        # 绘制箱线图，并返回绘图对象ax
        ax = boxplot(long_df, x="s", y="z", hue="c", native_scale=True)
        # 遍历每个色调水平
        for i, hue_level in enumerate(hue_levels):
            # 获取当前色调水平对应的容器
            bxp = ax.containers[i]
            # 遍历每个中心点
            for j, center in enumerate(centers):
                # 选择符合当前中心点和色调水平的数据行
                rows = (long_df["s"] == center) & (long_df["c"] == hue_level)
                # 获取数据
                data = long_df.loc[rows, "z"]
                # 计算箱线图的位置
                pos = center + [-offset, +offset][i]
                # 检查箱体部分
                self.check_box(bxp[j], data, "x", pos, width)
                # 检查箱须部分
                self.check_whiskers(bxp[j], data, "x", pos, width / 2)

    # 测试用例：使用对数缩放设置绘制箱线图，传入长格式数据框
    def test_dodge_native_scale_log(self, long_df):
        # 计算对数缩放后的"s"列
        pos = 10 ** long_df["s"]
        # 创建一个Figure对象的子图ax
        ax = mpl.figure.Figure().subplots()
        # 设置x轴为对数缩放
        ax.set_xscale("log")
        # 绘制箱线图，并传入对数缩放后的"s"列数据
        boxplot(long_df, x=pos, y="z", hue="c", native_scale=True, ax=ax)
        widths = []
        # 遍历每个容器中的箱线图
        for bxp in ax.containers:
            # 遍历每个箱线图中的箱体
            for box in bxp.boxes:
                # 获取箱体顶点的对数坐标
                coords = np.log10(box.get_path().vertices.T[0])
                # 计算箱体顶点坐标的极差并加入列表
                widths.append(np.ptp(coords))
        # 断言箱体宽度的标准差近似为0
        assert np.std(widths) == approx(0)

    # 测试用例：绘制没有色调的箱线图，传入长格式数据框
    def test_dodge_without_hue(self, long_df):
        # 绘制箱线图，并传入"x"列为"a"，"y"列为"y"，开启分组
        ax = boxplot(long_df, x="a", y="y", dodge=True)
        # 获取第一个容器，它包含所有的箱线图
        bxp, = ax.containers
        # 获取"a"列的分类顺序
        levels = categorical_order(long_df["a"])
        # 遍历每个分类水平
        for i, level in enumerate(levels):
            # 选择当前分类水平对应的数据
            data = long_df.loc[long_df["a"] == level, "y"]
            # 检查当前箱线图的箱体部分
            self.check_box(bxp[i], data, "x", i)
            # 检查当前箱线图的箱须部分
            self.check_whiskers(bxp[i], data, "x", i)

    # 测试用例：绘制数据对数缩放的箱线图，传入长格式数据框和方向orient
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_log_data_scale(self, long_df, orient):
        # 获取变量"y"或"x"的名称
        var = {"x": "y", "y": "x"}[orient]
        # 获取长格式数据框中"z"列的数据
        s = long_df["z"]
        # 创建一个Figure对象的子图ax
        ax = mpl.figure.Figure().subplots()
        # 根据orient参数设置对应轴的对数缩放
        getattr(ax, f"set_{var}scale")("log")
        # 绘制箱线图，并传入对应轴的数据和无限长的whiskers
        boxplot(**{var: s}, whis=np.inf, ax=ax)
        # 获取第一个容器中的第一个箱线图
        bxp = ax.containers[0][0]
        # 检查箱体部分
        self.check_box(bxp, s, orient, 0)
        # 检查箱须部分
        self.check_whiskers(bxp, s, orient, 0, whis=np.inf)

    # 测试用例：绘制指定颜色的箱线图，传入长格式数据框
    def test_color(self, long_df):
        # 指定颜色值
        color = "#123456"
        # 绘制箱线图，并传入"x"列为"a"，"y"列为"y"，颜色为color，饱和度为1
        ax = boxplot(long_df, x="a", y="y", color=color, saturation=1)
        # 遍历第一个容器中的所有箱线图
        for box in ax.containers[0].boxes:
            # 断言箱体颜色与指定颜色相同
            assert same_color(box.get_facecolor(), color)

    # 测试用例：绘制多色数据的宽格式箱线图，传入宽格式数据框
    def test_wide_data_multicolored(self, wide_df):
        # 绘制宽格式数据的箱线图，并获取所有容器的数量
        ax = boxplot(wide_df)
        # 断言容器数量等于宽格式数据框的列数
        assert len(ax.containers) == wide_df.shape[1]

    # 测试用例：绘制单色数据的宽格式箱线图，传入宽格式数据框
    def test_wide_data_single_color(self, wide_df):
        # 绘制宽格式数据的箱线图，箱体颜色设置为"C1"，饱和度为1
        ax = boxplot(wide_df, color="C1", saturation=1)
        # 断言只有一个容器
        assert len(ax.containers) == 1
        # 遍历第一个容器中的所有箱线图
        for box in ax.containers[0].boxes:
            # 断言箱体颜色为"C1"
            assert same_color(box.get_facecolor(), "C1")

    # 测试用例：绘制带色调颜色的箱线图，传入长格式数据框
    def test_hue_colors(self, long_df):
        # 绘制箱线图，并传入"x"列为"a"，"y"列为"y"，"hue"列为"b"，饱和度为1
        ax = boxplot(long_df, x="a", y="y", hue="b", saturation=1)
        # 遍历所有容器中的每个箱线图
        for i, bxp in enumerate(ax.containers):
            # 遍历每个箱线图中的箱体
            for box in bxp.boxes:
                # 断言箱体颜色与"C{i}"对应的颜色相同
                assert same_color(box.get_facecolor(), f"C{i}")
    # 测试箱线图的线条颜色设置
    def test_linecolor(self, long_df):
        # 设置颜色值
        color = "#778815"
        # 绘制箱线图，并指定线条颜色
        ax = boxplot(long_df, x="a", y="y", linecolor=color)
        # 获取第一个箱线图容器
        bxp = ax.containers[0]
        # 验证每条线的颜色是否与指定的颜色相同
        for line in [*bxp.medians, *bxp.whiskers, *bxp.caps]:
            assert same_color(line.get_color(), color)
        # 验证每个箱体的边缘颜色是否与指定的颜色相同
        for box in bxp.boxes:
            assert same_color(box.get_edgecolor(), color)
        # 验证每个异常值标记的边缘颜色是否与指定的颜色相同
        for flier in bxp.fliers:
            assert same_color(flier.get_markeredgecolor(), color)

    # 测试使用灰色线条时的警告信息
    def test_linecolor_gray_warning(self, long_df):
        # 在使用灰色线条时，测试是否触发未来警告
        with pytest.warns(FutureWarning, match="Use \"auto\" to set automatic"):
            boxplot(long_df, x="y", linecolor="gray")

    # 测试饱和度设置
    def test_saturation(self, long_df):
        # 设置颜色值和饱和度
        color = "#8912b0"
        ax = boxplot(long_df["x"], color=color, saturation=.5)
        # 获取第一个箱线图容器的第一个箱体
        box = ax.containers[0].boxes[0]
        # 验证箱体的颜色是否与指定的颜色和饱和度相匹配
        assert np.allclose(box.get_facecolor()[:3], desaturate(color, 0.5))

    # 测试线宽设置
    def test_linewidth(self, long_df):
        # 设置线宽值
        width = 5
        # 绘制箱线图，并指定线宽
        ax = boxplot(long_df, x="a", y="y", linewidth=width)
        # 获取第一个箱线图容器
        bxp = ax.containers[0]
        # 验证每条线的线宽是否与指定的线宽相同
        for line in [*bxp.boxes, *bxp.medians, *bxp.whiskers, *bxp.caps]:
            assert line.get_linewidth() == width

    # 测试填充设置
    def test_fill(self, long_df):
        # 设置颜色值
        color = "#459900"
        # 绘制箱线图，并指定不填充箱体和颜色
        ax = boxplot(x=long_df["z"], fill=False, color=color)
        # 获取第一个箱线图容器
        bxp = ax.containers[0]
        # 验证每条线的颜色是否与指定的颜色相同
        for line in [*bxp.boxes, *bxp.medians, *bxp.whiskers, *bxp.caps]:
            assert same_color(line.get_color(), color)

    # 测试凹口设置
    @pytest.mark.parametrize("notch_param", ["notch", "shownotches"])
    def test_notch(self, long_df, notch_param):
        # 绘制箱线图，并指定凹口参数为True
        ax = boxplot(x=long_df["z"], **{notch_param: True})
        # 获取第一个箱线图容器的第一个箱体，检查其路径顶点数
        verts = ax.containers[0].boxes[0].get_path().vertices
        assert len(verts) == 12

    # 测试whis参数设置
    def test_whis(self, long_df):
        # 获取数据
        data = long_df["z"]
        # 绘制箱线图，并指定whis参数为2
        ax = boxplot(x=data, whis=2)
        # 获取第一个箱线图容器的第一个子容器，检查其与数据的对比
        bxp = ax.containers[0][0]
        self.check_whiskers(bxp, data, "y", 0, whis=2)

    # 测试间隔设置
    def test_gap(self, long_df):
        # 绘制箱线图，并指定不同的组和类别的间隔
        ax = boxplot(long_df, x="a", y="z", hue="c", gap=.1)
        # 遍历每个hue级别
        for i, hue_level in enumerate(categorical_order(long_df["c"])):
            # 获取每个箱线图容器
            bxp = ax.containers[i]
            # 遍历每个level
            for j, level in enumerate(categorical_order(long_df["a"])):
                # 筛选出当前level和hue_level对应的数据
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                # 计算位置和宽度
                pos = j + [-.2, +.2][i]
                width = 0.9 * 0.4
                # 检查每个箱体
                self.check_box(bxp[j], data, "x", pos, width)
    # 对给定的长格式数据进行箱线图分析，测试属性字典的设置
    def test_prop_dicts(self, long_df):
        
        # 设置属性字典，包含箱线图的各种视觉属性
        prop_dicts = dict(
            boxprops=dict(linewidth=3),        # 箱体属性设置为线宽为3
            medianprops=dict(color=".1"),       # 中位数线属性设置为颜色为0.1
            whiskerprops=dict(linestyle="--"), # 须线属性设置为虚线
            capprops=dict(solid_capstyle="butt"),  # 箱线顶端属性设置为平头
            flierprops=dict(marker="s"),       # 异常值标记属性设置为正方形
        )
        
        # 属性映射字典，用于将元素名称映射到容器对象属性
        attr_map = dict(box="boxes", flier="fliers")
        
        # 创建箱线图并返回 Axes 对象
        ax = boxplot(long_df, x="a", y="z", hue="c", **prop_dicts)
        
        # 遍历箱线图容器对象
        for bxp in ax.containers:
            # 遍历每种元素（箱体、中位数线、须线、顶端、异常值）
            for element in ["box", "median", "whisker", "cap", "flier"]:
                # 获取对应的属性名称，如果没有匹配则默认为元素名称的复数形式
                attr = attr_map.get(element, f"{element}s")
                # 遍历当前元素的所有艺术家对象
                for artist in getattr(bxp, attr):
                    # 遍历当前元素的属性字典，验证每个属性的设置与预期相符
                    for k, v in prop_dicts[f"{element}props"].items():
                        assert plt.getp(artist, k) == v

    # 测试在不显示异常值的情况下创建箱线图
    def test_showfliers(self, long_df):
        
        # 创建不显示异常值的箱线图
        ax = boxplot(long_df["x"], showfliers=False)
        
        # 断言第一个容器中不存在异常值标记
        assert not ax.containers[0].fliers

    # 使用参数化测试框架，测试 boxplot 和 catplot 的视觉效果是否相同
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),      # 宽格式数据
            dict(data="wide", orient="h"),  # 宽格式数据，横向显示
            dict(data="flat"),      # 平坦序列数据
            dict(data="long", x="a", y="y"),  # 长格式数据，指定 x 和 y
            dict(data=None, x="a", y="y"),    # 没有数据，指定 x 和 y
            dict(data="long", x="a", y="y", hue="a"),  # 长格式数据，指定 x、y 和 hue
            dict(data=None, x="a", y="y", hue="a"),    # 没有数据，指定 x、y 和 hue
            dict(data="long", x="a", y="y", hue="b"),  # 长格式数据，指定 x、y 和不同的 hue
            dict(data=None, x="s", y="y", hue="a"),    # 没有数据，指定 x、y 和 hue
            dict(data="long", x="a", y="y", hue="s"),  # 长格式数据，指定 x、y 和不同的 hue
            dict(data="null", x="a", y="y", hue="a"),  # 无效数据，指定 x、y 和 hue
            dict(data="long", x="s", y="y", hue="a", native_scale=True),  # 长格式数据，指定 x、y 和 hue，并使用本地比例
            dict(data="long", x="d", y="y", hue="a", native_scale=True),  # 长格式数据，指定 x、y 和 hue，并使用本地比例
            dict(data="null", x="a", y="y", hue="b", fill=False, gap=.2),  # 无效数据，指定 x、y、hue，取消填充，设置间距
            dict(data="null", x="a", y="y", whis=1, showfliers=False),    # 无效数据，指定 x、y，设置须的长度和不显示异常值
            dict(data="null", x="a", y="y", linecolor="r", linewidth=5),  # 无效数据，指定 x、y，设置线的颜色和宽度
            dict(data="null", x="a", y="y", shownotches=True, showcaps=False),  # 无效数据，指定 x、y，显示切口，不显示顶端
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):
        
        # 根据参数化的 kwargs 设置不同的数据输入
        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            # 如果数据为 None，则根据参数中的 x、y、hue 在长格式数据中获取相应列
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        # 创建箱线图并获取 Axes 对象
        ax = boxplot(**kwargs)
        
        # 创建 catplot 并获取 AxesSubplot 对象
        g = catplot(**kwargs, kind="box")
        
        # 断言两个图的视觉效果是否相同
        assert_plots_equal(ax, g.ax)
class TestBoxenPlot(SharedAxesLevelTests, SharedPatchArtistTests):
    # 定义一个测试类 TestBoxenPlot，继承自 SharedAxesLevelTests 和 SharedPatchArtistTests

    func = staticmethod(boxenplot)
    # 设置 func 属性为 boxenplot 的静态方法

    @pytest.fixture
    def common_kws(self):
        # 定义一个 pytest 的 fixture，返回一个包含 {"saturation": 1} 的字典
        return {"saturation": 1}

    def get_last_color(self, ax):
        # 定义一个方法 get_last_color，接受一个 ax 参数
        # 获取 ax 中倒数第二个集合的面颜色
        fcs = ax.collections[-2].get_facecolors()
        return to_rgba(fcs[len(fcs) // 2])
        # 返回转换后的 rgba 颜色值

    def get_box_width(self, path, orient="x"):
        # 定义一个方法 get_box_width，接受 path 和 orient 参数，默认为 "x"
        # 获取路径对象 path 的顶点并转置
        verts = path.vertices.T
        # 确定 orient 参数对应的索引
        idx = ["y", "x"].index(orient)
        # 返回 verts 中指定索引轴上的极差（即最大值减最小值）
        return np.ptp(verts[idx])

    def check_boxen(self, patches, data, orient, pos, width=0.8):
        # 定义一个方法 check_boxen，接受 patches, data, orient, pos 和 width 参数（width 默认为 0.8）
        # 根据 orient 参数获取位置和值的索引
        pos_idx, val_idx = self.orient_indices(orient)
        # 提取 patches 集合中所有路径的顶点，并进行堆叠和转置操作
        verts = np.stack([v.vertices for v in patches.get_paths()], 1).T

        # 断言位置索引的最小值大于等于 pos - width / 2
        assert verts[pos_idx].min().round(4) >= np.round(pos - width / 2, 4)
        # 断言位置索引的最大值小于等于 pos + width / 2
        assert verts[pos_idx].max().round(4) <= np.round(pos + width / 2, 4)
        # 断言数据的第 25% 和第 75% 分位数在值索引中存在
        assert np.in1d(
            np.percentile(data, [25, 75]).round(4), verts[val_idx].round(4).flat
        ).all()
        # 断言值索引的后一部分的第一个元素等于前一部分的最后一个元素
        assert_array_equal(verts[val_idx, 1:, 0], verts[val_idx, :-1, 2])

    @pytest.mark.parametrize("orient,col", [("x", "y"), ("y", "z")])
    def test_single_var(self, long_df, orient, col):
        # 定义一个参数化测试方法 test_single_var，接受 long_df, orient 和 col 参数
        # 根据 orient 参数选择 var 字典中的键
        var = {"x": "y", "y": "x"}[orient]
        # 调用 boxenplot 方法，并返回 ax 对象
        ax = boxenplot(long_df, **{var: col})
        # 获取 ax 集合中的第一个 patches
        patches = ax.collections[0]
        # 调用 self.check_boxen 方法检查 patches 和 long_df[col] 的箱线图
        self.check_boxen(patches, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient,col", [(None, "x"), ("x", "y"), ("y", "z")])
    def test_vector_data(self, long_df, orient, col):
        # 定义一个参数化测试方法 test_vector_data，接受 long_df, orient 和 col 参数
        # 如果 orient 为 None，则 orient 设为 "x"
        orient = "x" if orient is None else orient
        # 调用 boxenplot 方法，并返回 ax 对象
        ax = boxenplot(long_df[col], orient=orient)
        # 获取 ax 集合中的第一个 patches
        patches = ax.collections[0]
        # 调用 self.check_boxen 方法检查 patches 和 long_df[col] 的箱线图
        self.check_boxen(patches, long_df[col], orient, 0)

    @pytest.mark.parametrize("orient", ["h", "v"])
    def test_wide_data(self, wide_df, orient):
        # 定义一个参数化测试方法 test_wide_data，接受 wide_df 和 orient 参数
        # 根据 orient 参数选择对应的轴向
        orient = {"h": "y", "v": "x"}[orient]
        # 调用 boxenplot 方法，并返回 ax 对象
        ax = boxenplot(wide_df, orient=orient)
        # 获取所有的集合，并赋值给 collections
        collections = ax.findobj(mpl.collections.PatchCollection)
        # 遍历 collections 中的索引和 patches
        for i, patches in enumerate(collections):
            # 获取 wide_df 中的列名作为 col
            col = wide_df.columns[i]
            # 调用 self.check_boxen 方法检查 patches 和 wide_df[col] 的箱线图
            self.check_boxen(patches, wide_df[col], orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_grouped(self, long_df, orient):
        # 定义一个参数化测试方法 test_grouped，接受 long_df 和 orient 参数
        # 根据 orient 参数选择 value 字典中的键
        value = {"x": "y", "y": "x"}[orient]
        # 调用 boxenplot 方法，并返回 ax 对象
        ax = boxenplot(long_df, **{orient: "a", value: "z"})
        # 获取 long_df["a"] 中分类的顺序，赋值给 levels
        levels = categorical_order(long_df["a"])
        # 获取所有的集合，并赋值给 collections
        collections = ax.findobj(mpl.collections.PatchCollection)
        # 遍历 levels 中的索引和 level
        for i, level in enumerate(levels):
            # 根据 long_df["a"] == level 筛选数据并赋值给 data
            data = long_df.loc[long_df["a"] == level, "z"]
            # 调用 self.check_boxen 方法检查 collections[i] 和 data 的箱线图
            self.check_boxen(collections[i], data, orient, i)

    @pytest.mark.parametrize("orient", ["x", "y"])
    # 测试带有分组的盒形图绘制，根据给定的方向（orient）确定横轴或纵轴，绘制相应的盒形图
    def test_hue_grouped(self, long_df, orient):
        # 根据方向（orient）选择对应的值，用于确定盒形图的横轴或纵轴
        value = {"x": "y", "y": "x"}[orient]
        # 调用boxenplot函数绘制盒形图，其中hue参数指定分组依据，orient和value确定数据的展示方向
        ax = boxenplot(long_df, hue="c", **{orient: "a", value: "z"})
        # 获取图中的PatchCollection集合对象
        collections = iter(ax.findobj(mpl.collections.PatchCollection))
        # 遍历长数据框long_df中'a'列的分类顺序
        for i, level in enumerate(categorical_order(long_df["a"])):
            # 遍历长数据框long_df中'c'列的分类顺序
            for j, hue_level in enumerate(categorical_order(long_df["c"])):
                # 筛选出符合当前分类水平的数据行
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                # 获取符合条件的数据列'z'
                data = long_df.loc[rows, "z"]
                # 根据索引和分类水平调整盒形图的位置
                pos = i + [-.2, +.2][j]
                # 检查并绘制盒形图
                self.check_boxen(next(collections), data, orient, pos, width=0.4)

    # 测试在本地缩放模式下绘制盒形图，根据长数据框中's'列的分类顺序确定盒形图的中心位置
    def test_dodge_native_scale(self, long_df):
        # 获取长数据框long_df中's'列的分类顺序
        centers = categorical_order(long_df["s"])
        # 获取长数据框long_df中'c'列的分类顺序
        hue_levels = categorical_order(long_df["c"])
        # 计算相邻分类中心点之间的最小间距
        spacing = min(np.diff(centers))
        # 计算盒形图的宽度
        width = 0.8 * spacing / len(hue_levels)
        # 计算盒形图位置的偏移量
        offset = width / len(hue_levels)
        # 调用boxenplot函数绘制盒形图，设置x轴为's'，y轴为'z'，hue参数为'c'，启用本地缩放模式
        ax = boxenplot(long_df, x="s", y="z", hue="c", native_scale=True)
        # 获取图中的PatchCollection集合对象
        collections = iter(ax.findobj(mpl.collections.PatchCollection))
        # 遍历分类中心点
        for center in centers:
            # 遍历'hue_levels'中的分类水平
            for i, hue_level in enumerate(hue_levels):
                # 筛选出符合当前分类中心点和分类水平的数据行
                rows = (long_df["s"] == center) & (long_df["c"] == hue_level)
                # 获取符合条件的数据列'z'
                data = long_df.loc[rows, "z"]
                # 根据中心点和偏移量调整盒形图的位置
                pos = center + [-offset, +offset][i]
                # 检查并绘制盒形图
                self.check_boxen(next(collections), data, "x", pos, width)

    # 测试设置盒形图的颜色，绘制x轴为'a'，y轴为'y'的盒形图，颜色设置为指定的色彩
    def test_color(self, long_df):
        # 设置颜色值
        color = "#123456"
        # 调用boxenplot函数绘制盒形图，设置x轴为'a'，y轴为'y'，颜色为指定的颜色，饱和度为1
        ax = boxenplot(long_df, x="a", y="y", color=color, saturation=1)
        # 获取图中的PatchCollection集合对象
        collections = ax.findobj(mpl.collections.PatchCollection)
        # 遍历PatchCollection集合中的patches对象
        for patches in collections:
            # 获取patches对象的填充颜色
            fcs = patches.get_facecolors()
            # 断言中心位置的颜色与设置的颜色相同
            assert same_color(fcs[len(fcs) // 2], color)

    # 测试设置不同色调的盒形图，根据长数据框中'a'列的分类顺序绘制，颜色由'b'列的不同水平决定
    def test_hue_colors(self, long_df):
        # 调用boxenplot函数绘制盒形图，设置x轴为'a'，y轴为'y'，hue参数为'b'，饱和度为1
        ax = boxenplot(long_df, x="a", y="y", hue="b", saturation=1)
        # 获取长数据框中'b'列的唯一值数量
        n_levels = long_df["b"].nunique()
        # 获取图中的PatchCollection集合对象
        collections = ax.findobj(mpl.collections.PatchCollection)
        # 遍历PatchCollection集合中的patches对象
        for i, patches in enumerate(collections):
            # 获取patches对象的填充颜色
            fcs = patches.get_facecolors()
            # 断言中心位置的颜色与按序循环的色调相同
            assert same_color(fcs[len(fcs) // 2], f"C{i % n_levels}")

    # 测试设置盒形图边线的颜色，绘制x轴为'a'，y轴为'y'的盒形图，边线颜色设置为指定的色彩
    def test_linecolor(self, long_df):
        # 设置边线颜色值
        color = "#669913"
        # 调用boxenplot函数绘制盒形图，设置x轴为'a'，y轴为'y'，边线颜色为指定的颜色
        ax = boxenplot(long_df, x="a", y="y", linecolor=color)
        # 遍历PatchCollection集合中的patches对象
        for patches in ax.findobj(mpl.collections.PatchCollection):
            # 断言边线颜色与设置的颜色相同
            assert same_color(patches.get_edgecolor(), color)

    # 测试设置盒形图边线的宽度，绘制x轴为'a'，y轴为'y'的盒形图，边线宽度设置为指定的宽度
    def test_linewidth(self, long_df):
        # 设置边线宽度值
        width = 5
        # 调用boxenplot函数绘制盒形图，设置x轴为'a'，y轴为'y'，边线宽度为指定的宽度
        ax = boxenplot(long_df, x="a", y="y", linewidth=width)
        # 遍历PatchCollection集合中的patches对象
        for patches in ax.findobj(mpl.collections.PatchCollection):
            # 断言边线宽度与设置的宽度相同
            assert patches.get_linewidth() == width

    # 测试设置盒形图的饱和度，绘制x轴为'x'的盒形图，颜色设置为指定的色彩，并指定饱和度
    def test_saturation(self, long_df):
        # 设置颜色值和饱和度
        color = "#8912b0"
        saturation = .5
        # 调用boxenplot函数绘制盒形图，设置x轴为'x'，颜色为指定的颜色，饱和度为指定的饱和度
        ax = boxenplot(long_df["x"], color=color, saturation=saturation)
        # 获取图中的PatchCollection集合对象
        fcs = ax
    # 测试箱线图在不同情况下的间隙效果
    def test_gap(self, long_df):
        # 创建包含两个子图的 Figure 对象，并获取这两个子图的引用
        ax1, ax2 = mpl.figure.Figure().subplots(2)
        # 在第一个子图上绘制箱线图，根据指定的列和参数进行绘制
        boxenplot(long_df, x="a", y="y", hue="s", ax=ax1)
        # 在第二个子图上绘制箱线图，设置间隙参数为0.2
        boxenplot(long_df, x="a", y="y", hue="s", gap=.2, ax=ax2)
        # 在第一个子图中查找所有的 PatchCollection 对象
        c1 = ax1.findobj(mpl.collections.PatchCollection)
        # 在第二个子图中查找所有的 PatchCollection 对象
        c2 = ax2.findobj(mpl.collections.PatchCollection)
        # 对两个子图中的 PatchCollection 进行逐一比较
        for p1, p2 in zip(c1, c2):
            # 计算第一个图中第一个路径的宽度
            w1 = np.ptp(p1.get_paths()[0].vertices[:, 0])
            # 计算第二个图中第一个路径的宽度
            w2 = np.ptp(p2.get_paths()[0].vertices[:, 0])
            # 断言第二个图的宽度是第一个图宽度的80%
            assert (w2 / w1) == pytest.approx(0.8)

    # 测试不填充箱线图的效果
    def test_fill(self, long_df):
        # 在指定的 DataFrame 上绘制不填充的箱线图，并获取 Axes 对象的引用
        ax = boxenplot(long_df, x="a", y="y", hue="s", fill=False)
        # 在绘制的图中查找所有的 PatchCollection 对象
        for c in ax.findobj(mpl.collections.PatchCollection):
            # 断言所有路径都没有填充颜色
            assert not c.get_facecolors().size

    # 测试箱线图中 k_depth 参数为整数的情况
    def test_k_depth_int(self, rng):
        # 从正态分布中生成随机样本
        x = rng.normal(0, 1, 10_000)
        # 绘制箱线图，并设置 k_depth 参数
        ax = boxenplot(x, k_depth=(k := 8))
        # 断言第一个 PatchCollection 包含的路径数量符合预期
        assert len(ax.collections[0].get_paths()) == (k * 2 - 1)

    # 测试箱线图中 k_depth 参数为"full"的情况
    def test_k_depth_full(self, rng):
        # 从正态分布中生成随机样本
        x = rng.normal(0, 1, 10_000)
        # 绘制箱线图，并设置 k_depth 参数为"full"
        ax = boxenplot(x=x, k_depth="full")
        # 获取绘制的图中第一个 PatchCollection 包含的路径
        paths = ax.collections[0].get_paths()
        # 断言路径数量符合预期
        assert len(paths) == 2 * int(np.log2(x.size)) + 1
        # 获取所有路径的顶点坐标，并转置以便按行访问
        verts = np.concatenate([p.vertices for p in paths]).T
        # 断言顶点坐标中第一行的最小值与样本的最小值相等
        assert verts[0].min() == x.min()
        # 断言顶点坐标中第一行的最大值与样本的最大值相等
        assert verts[0].max() == x.max()
        # 断言第二个 PatchCollection 中没有偏移量
        assert not ax.collections[1].get_offsets().size

    # 测试箱线图中 k_depth 参数为"trustworthy"的情况
    def test_trust_alpha(self, rng):
        # 从正态分布中生成随机样本
        x = rng.normal(0, 1, 10_000)
        # 绘制箱线图，并设置 k_depth 和 trust_alpha 参数
        ax = boxenplot(x, k_depth="trustworthy", trust_alpha=.1)
        # 在已有的图上再次绘制箱线图，并设置不同的 trust_alpha 参数
        boxenplot(x, k_depth="trustworthy", trust_alpha=.001, ax=ax)
        # 在图中查找所有的 PatchCollection 对象
        cs = ax.findobj(mpl.collections.PatchCollection)
        # 断言第一个 PatchCollection 中路径数量大于第二个
        assert len(cs[0].get_paths()) > len(cs[1].get_paths())

    # 测试箱线图中 k_depth 参数为"proportion"的情况
    def test_outlier_prop(self, rng):
        # 从正态分布中生成随机样本
        x = rng.normal(0, 1, 10_000)
        # 绘制箱线图，并设置 k_depth 和 outlier_prop 参数
        ax = boxenplot(x, k_depth="proportion", outlier_prop=.001)
        # 在已有的图上再次绘制箱线图，并设置不同的 outlier_prop 参数
        boxenplot(x, k_depth="proportion", outlier_prop=.1, ax=ax)
        # 在图中查找所有的 PatchCollection 对象
        cs = ax.findobj(mpl.collections.PatchCollection)
        # 断言第一个 PatchCollection 中路径数量大于第二个
        assert len(cs[0].get_paths()) > len(cs[1].get_paths())

    # 测试箱线图中 width_method 参数为"exponential"的情况
    def test_exponential_width_method(self, rng):
        # 从正态分布中生成随机样本
        x = rng.normal(0, 1, 10_000)
        # 绘制箱线图，并设置 width_method 参数为"exponential"
        ax = boxenplot(x=x, width_method="exponential")
        # 获取绘制的图中的第一个 PatchCollection
        c = ax.findobj(mpl.collections.PatchCollection)[0]
        # 获取各路径的箱子宽度
        ws = [self.get_box_width(p) for p in c.get_paths()]
        # 断言路径宽度的比例关系符合预期
        assert (ws[1] / ws[0]) == pytest.approx(ws[2] / ws[1])

    # 测试箱线图中 width_method 参数为"linear"的情况
    def test_linear_width_method(self, rng):
        # 从正态分布中生成随机样本
        x = rng.normal(0, 1, 10_000)
        # 绘制箱线图，并设置 width_method 参数为"linear"
        ax = boxenplot(x=x, width_method="linear")
        # 获取绘制的图中的第一个 PatchCollection
        c = ax.findobj(mpl.collections.PatchCollection)[0]
        # 获取各路径的箱子宽度
        ws = [self.get_box_width(p) for p in c.get_paths()]
        # 断言路径宽度的差异符合预期
        assert (ws[1] - ws[0]) == pytest.approx(ws[2] - ws[1])

    # 测试箱线图中 width_method 参数为"area"的情况
    def test_area_width_method(self, rng):
        # 从均匀分布中生成随机样本
        x = rng.uniform(0, 1, 10_000)
        # 绘制箱线图，并设置 width_method 参数为"area" 和 k_depth 参数为2
        ax = boxenplot(x=x, width_method="area", k_depth=2)
        # 获取绘制的图中的第一个 PatchCollection，并获取其中的路径
        ps = ax.findobj(mpl.collections.PatchCollection)[0].get_paths()
        # 获取各路径的箱子宽度
        ws = [self.get_box_width(p) for p in ps]
        # 断言所有箱子的宽度大于0.7
        assert np.greater(ws, 0.7).all()
    def test_box_kws(self, long_df):
        # 使用 `boxenplot` 函数绘制长格式数据 `long_df` 的箱线图，设置箱子属性参数 `linewidth` 为 7.1
        ax = boxenplot(long_df, x="a", y="y", box_kws={"linewidth": (lw := 7.1)})
        # 遍历图形对象集合，验证所有路径集合的线宽是否为预期的 `lw`
        for c in ax.findobj(mpl.collections.PatchCollection):
            assert c.get_linewidths() == lw

    def test_line_kws(self, long_df):
        # 使用 `boxenplot` 函数绘制长格式数据 `long_df` 的箱线图，设置线条属性参数 `linewidth` 为 6.2
        ax = boxenplot(long_df, x="a", y="y", line_kws={"linewidth": (lw := 6.2)})
        # 遍历图形对象集合，验证所有线条的线宽是否为预期的 `lw`
        for line in ax.lines:
            assert line.get_linewidth() == lw

    def test_flier_kws(self, long_df):
        # 使用 `boxenplot` 函数绘制长格式数据 `long_df` 的箱线图，设置异常值属性参数 `marker` 为 "X"
        ax = boxenplot(long_df, x="a", y="y", flier_kws={"marker": (marker := "X")})
        # 获取预期的标记路径顶点，验证所有路径集合的顶点是否与预期相同
        expected = mpl.markers.MarkerStyle(marker).get_path().vertices
        for c in ax.findobj(mpl.collections.PathCollection):
            assert_array_equal(c.get_paths()[0].vertices, expected)

    def test_k_depth_checks(self, long_df):
        # 使用 `pytest.raises` 确保 `boxenplot` 函数在 `k_depth` 参数为 "auto" 时引发 `ValueError`
        with pytest.raises(ValueError, match="The value for `k_depth`"):
            boxenplot(x=long_df["y"], k_depth="auto")
        # 使用 `pytest.raises` 确保 `boxenplot` 函数在 `k_depth` 参数为元组时引发 `TypeError`
        with pytest.raises(TypeError, match="The `k_depth` parameter"):
            boxenplot(x=long_df["y"], k_depth=(1, 2))

    def test_width_method_check(self, long_df):
        # 使用 `pytest.raises` 确保 `boxenplot` 函数在 `width_method` 参数为 "uniform" 时引发 `ValueError`
        with pytest.raises(ValueError, match="The value for `width_method`"):
            boxenplot(x=long_df["y"], width_method="uniform")

    def test_scale_deprecation(self, long_df):
        # 使用 `pytest.warns` 确保 `boxenplot` 函数在 `scale` 参数为 "linear" 时引发 `FutureWarning`
        with pytest.warns(FutureWarning, match="The `scale` parameter has been"):
            boxenplot(x=long_df["y"], scale="linear")
        # 使用 `pytest.warns` 确保 `boxenplot` 函数在 `scale` 参数为 "area" 时引发 `FutureWarning`
        with pytest.warns(FutureWarning, match=".+result for 'area' will appear"):
            boxenplot(x=long_df["y"], scale="area")

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y"),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s", showfliers=False),
            dict(data="null", x="a", y="y", hue="a", saturation=.5),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="null", x="a", y="y", hue="b", fill=False, gap=.2),
            dict(data="null", x="a", y="y", linecolor="r", linewidth=5),
            dict(data="long", x="a", y="y", k_depth="trustworthy", trust_alpha=.1),
            dict(data="long", x="a", y="y", k_depth="proportion", outlier_prop=.1),
            dict(data="long", x="a", y="z", width_method="area"),
            dict(data="long", x="a", y="z", box_kws={"alpha": .2}, alpha=.4)
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):
        # 比较测试函数，用于比较不同数据结构的可视化效果

        # 根据传入的参数确定使用哪种数据结构进行测试
        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            # 如果数据为 None，则根据 x、y、hue 变量从 long_df 中选择相应列作为数据
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        # 绘制 boxenplot 并获取其轴对象
        ax = boxenplot(**kwargs)
        # 绘制 catplot 并获取其图形对象
        g = catplot(**kwargs, kind="boxen")

        # 断言两个绘图对象的轴对象相等
        assert_plots_equal(ax, g.ax)
# 定义一个名为 TestViolinPlot 的测试类，继承自 SharedAxesLevelTests 和 SharedPatchArtistTests
class TestViolinPlot(SharedAxesLevelTests, SharedPatchArtistTests):

    # 设定类方法 func 为 violinplot 的静态方法
    func = staticmethod(violinplot)

    # 定义一个 pytest 的测试 fixture，返回一个包含 {"saturation": 1} 的字典
    @pytest.fixture
    def common_kws(self):
        return {"saturation": 1}

    # 定义一个实例方法 get_last_color，接收参数 ax
    def get_last_color(self, ax):
        # 获取 ax.collections 中最后一个元素的颜色并转换成 RGBA 格式后返回
        color = ax.collections[-1].get_facecolor()
        return to_rgba(color)

    # 定义一个实例方法 violin_width，接收参数 poly 和 orient，默认为 "x"
    def violin_width(self, poly, orient="x"):
        # 根据 orient 获取索引 idx
        idx, _ = self.orient_indices(orient)
        # 获取 poly 的第一个路径的顶点并计算在 idx 方向上的极差
        return np.ptp(poly.get_paths()[0].vertices[:, idx])

    # 定义一个实例方法 check_violin，接收参数 poly, data, orient, pos, width 默认为 0.8
    def check_violin(self, poly, data, orient, pos, width=0.8):
        # 根据 orient 获取位置索引 pos_idx 和值索引 val_idx
        pos_idx, val_idx = self.orient_indices(orient)
        # 获取 poly 的第一个路径的顶点的转置
        verts = poly.get_paths()[0].vertices.T

        # 断言顶点的位置在预期范围内
        assert verts[pos_idx].min() >= (pos - width / 2)
        assert verts[pos_idx].max() <= (pos + width / 2)
        # 假定 violin 是在 cut=0 的情况下计算的
        assert verts[val_idx].min() == approx(data.min())
        assert verts[val_idx].max() == approx(data.max())

    # 使用 pytest.mark.parametrize 标记的参数化测试，参数为 orient 和 col
    @pytest.mark.parametrize("orient,col", [("x", "y"), ("y", "z")])
    # 定义一个测试方法 test_single_var，接收参数 long_df, orient, col
    def test_single_var(self, long_df, orient, col):
        # 根据 orient 确定 var
        var = {"x": "y", "y": "x"}[orient]
        # 调用 violinplot 绘制图形，获取 ax
        ax = violinplot(long_df, **{var: col}, cut=0)
        # 获取 ax.collections 中的第一个 poly
        poly = ax.collections[0]
        # 调用 self.check_violin 方法检查 poly 的数据
        self.check_violin(poly, long_df[col], orient, 0)

    # 使用 pytest.mark.parametrize 标记的参数化测试，参数为 orient 和 col
    @pytest.mark.parametrize("orient,col", [(None, "x"), ("x", "y"), ("y", "z")])
    # 定义一个测试方法 test_vector_data，接收参数 long_df, orient, col
    def test_vector_data(self, long_df, orient, col):
        # 如果 orient 是 None，则设置 orient 为 "x"
        orient = "x" if orient is None else orient
        # 调用 violinplot 绘制图形，获取 ax
        ax = violinplot(long_df[col], cut=0, orient=orient)
        # 获取 ax.collections 中的第一个 poly
        poly = ax.collections[0]
        # 调用 self.check_violin 方法检查 poly 的数据
        self.check_violin(poly, long_df[col], orient, 0)

    # 使用 pytest.mark.parametrize 标记的参数化测试，参数为 orient
    @pytest.mark.parametrize("orient", ["h", "v"])
    # 定义一个测试方法 test_wide_data，接收参数 wide_df, orient
    def test_wide_data(self, wide_df, orient):
        # 根据 orient 确定 orient 的值
        orient = {"h": "y", "v": "x"}[orient]
        # 调用 violinplot 绘制图形，获取 ax
        ax = violinplot(wide_df, cut=0, orient=orient)
        # 遍历 ax.collections 中的 poly
        for i, poly in enumerate(ax.collections):
            # 获取 wide_df 的列名作为 col
            col = wide_df.columns[i]
            # 调用 self.check_violin 方法检查 poly 的数据
            self.check_violin(poly, wide_df[col], orient, i)

    # 使用 pytest.mark.parametrize 标记的参数化测试，参数为 orient
    @pytest.mark.parametrize("orient", ["x", "y"])
    # 定义一个测试方法 test_grouped，接收参数 long_df, orient
    def test_grouped(self, long_df, orient):
        # 根据 orient 确定 value
        value = {"x": "y", "y": "x"}[orient]
        # 调用 violinplot 绘制图形，获取 ax
        ax = violinplot(long_df, **{orient: "a", value: "z"}, cut=0)
        # 获取 long_df["a"] 的分类顺序 levels
        levels = categorical_order(long_df["a"])
        # 遍历 levels 中的每一个 level 和对应的索引 i
        for i, level in enumerate(levels):
            # 从 long_df 中选择 long_df["a"] == level 的数据作为 data
            data = long_df.loc[long_df["a"] == level, "z"]
            # 调用 self.check_violin 方法检查 ax.collections[i] 的数据
            self.check_violin(ax.collections[i], data, orient, i)

    # 使用 pytest.mark.parametrize 标记的参数化测试，参数为 orient
    @pytest.mark.parametrize("orient", ["x", "y"])
    # 测试函数，用于绘制分组的小提琴图
    def test_hue_grouped(self, long_df, orient):
        # 根据传入的 orient 参数确定 x 和 y 的值
        value = {"x": "y", "y": "x"}[orient]
        # 调用 violinplot 函数生成小提琴图，并将返回的对象赋给 ax
        ax = violinplot(long_df, hue="c", **{orient: "a", value: "z"}, cut=0)
        # 获取 ax 中的多边形集合的迭代器
        polys = iter(ax.collections)
        # 遍历长数据框中变量 a 的分类顺序
        for i, level in enumerate(categorical_order(long_df["a"])):
            # 遍历长数据框中变量 c 的分类顺序
            for j, hue_level in enumerate(categorical_order(long_df["c"])):
                # 选择行，其中 a 等于 level 且 c 等于 hue_level
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                # 从长数据框中提取符合条件的数据列 z
                data = long_df.loc[rows, "z"]
                # 计算位置偏移量，根据 j 的值选择 -.2 或 +.2
                pos = i + [-.2, +.2][j]
                # 设置小提琴的宽度为 0.4
                width = 0.4
                # 调用 self.check_violin 方法检查小提琴
                self.check_violin(next(polys), data, orient, pos, width)

    # 测试函数，用于绘制未躲避的小提琴图
    def test_hue_not_dodged(self, long_df):
        # 获取长数据框变量 b 的分类顺序
        levels = categorical_order(long_df["b"])
        # 根据 b 的前两个水平创建 hue 序列
        hue = long_df["b"].isin(levels[:2])
        # 调用 violinplot 函数生成小提琴图，并将返回的对象赋给 ax
        ax = violinplot(long_df, x="b", y="z", hue=hue, cut=0)
        # 遍历分类顺序中的每个水平
        for i, level in enumerate(levels):
            # 获取第 i 个多边形对象
            poly = ax.collections[i]
            # 选择长数据框中变量 b 等于 level 的行，并提取 z 列数据
            data = long_df.loc[long_df["b"] == level, "z"]
            # 调用 self.check_violin 方法检查小提琴
            self.check_violin(poly, data, "x", i)

    # 测试函数，用于绘制躲避的小提琴图（本地比例）
    def test_dodge_native_scale(self, long_df):
        # 获取长数据框变量 s 的分类顺序
        centers = categorical_order(long_df["s"])
        # 获取长数据框变量 c 的分类顺序
        hue_levels = categorical_order(long_df["c"])
        # 计算中心之间的最小间距
        spacing = min(np.diff(centers))
        # 计算小提琴的宽度
        width = 0.8 * spacing / len(hue_levels)
        # 计算偏移量
        offset = width / len(hue_levels)
        # 调用 violinplot 函数生成小提琴图，并将返回的对象赋给 ax
        ax = violinplot(long_df, x="s", y="z", hue="c", native_scale=True, cut=0)
        # 获取 ax 中的小提琴对象集合的迭代器
        violins = iter(ax.collections)
        # 遍历中心点序列
        for center in centers:
            # 遍历 hue 水平序列
            for i, hue_level in enumerate(hue_levels):
                # 选择 s 等于 center 且 c 等于 hue_level 的行
                rows = (long_df["s"] == center) & (long_df["c"] == hue_level)
                # 从长数据框中提取符合条件的数据列 z
                data = long_df.loc[rows, "z"]
                # 计算位置偏移量，根据 i 的值选择 -offset 或 +offset
                pos = center + [-offset, +offset][i]
                # 获取下一个小提琴对象
                poly = next(violins)
                # 调用 self.check_violin 方法检查小提琴
                self.check_violin(poly, data, "x", pos, width)

    # 测试函数，用于绘制躲避的小提琴图（对数尺度）
    def test_dodge_native_scale_log(self, long_df):
        # 计算 s 列的对数值
        pos = 10 ** long_df["s"]
        # 创建一个 Figure 对象并生成子图 ax
        ax = mpl.figure.Figure().subplots()
        # 设置 x 轴为对数尺度
        ax.set_xscale("log")
        # 构建包含 x, y, hue 的变量字典
        variables = dict(x=pos, y="z", hue="c")
        # 调用 violinplot 函数生成小提琴图，并将返回的对象赋给 ax
        violinplot(long_df, **variables, native_scale=True, density_norm="width", ax=ax)
        # 初始化一个空列表 widths
        widths = []
        # 计算小提琴数量
        n_violins = long_df["s"].nunique() * long_df["c"].nunique()
        # 遍历前 n_violins 个多边形对象
        for poly in ax.collections[:n_violins]:
            # 获取多边形路径的顶点坐标 x 值
            verts = poly.get_paths()[0].vertices[:, 0]
            # 将顶点坐标的对数值计算出来
            coords = np.log10(verts)
            # 计算坐标范围的极差并添加到 widths 列表中
            widths.append(np.ptp(coords))
        # 断言 widths 列表的标准差近似为 0
        assert np.std(widths) == approx(0)

    # 测试函数，用于测试指定颜色的小提琴图
    def test_color(self, long_df):
        # 指定颜色值
        color = "#123456"
        # 调用 violinplot 函数生成指定颜色的小提琴图，并将返回的对象赋给 ax
        ax = violinplot(long_df, x="a", y="y", color=color, saturation=1)
        # 遍历 ax 中的多边形对象集合
        for poly in ax.collections:
            # 断言多边形的填充颜色与指定颜色相同
            assert same_color(poly.get_facecolor(), color)

    # 测试函数，用于测试根据 hue 变量生成颜色的小提琴图
    def test_hue_colors(self, long_df):
        # 调用 violinplot 函数生成根据 hue 变量生成颜色的小提琴图，并将返回的对象赋给 ax
        ax = violinplot(long_df, x="a", y="y", hue="b", saturation=1)
        # 获取长数据框中变量 b 的唯一值数量
        n_levels = long_df["b"].nunique()
        # 遍历 ax 中的多边形对象集合
        for i, poly in enumerate(ax.collections):
            # 断言多边形的填充颜色与自动生成的颜色相同
            assert same_color(poly.get_facecolor(), f"C{i % n_levels}")
    # 测试绘制小提琴图时的线条颜色设置
    def test_linecolor(self, long_df, inner):
        # 设置线条颜色为 "#669913"
        color = "#669913"
        # 调用 violinplot 函数绘制小提琴图，并指定线条颜色和内部显示方式
        ax = violinplot(long_df, x="a", y="y", linecolor=color, inner=inner)
        # 验证所有多边形的边框颜色是否与指定的颜色相同
        for poly in ax.findobj(mpl.collections.PolyCollection):
            assert same_color(poly.get_edgecolor(), color)
        # 验证所有线条集合的颜色是否与指定的颜色相同
        for lines in ax.findobj(mpl.collections.LineCollection):
            assert same_color(lines.get_color(), color)
        # 验证所有线条的颜色是否与指定的颜色相同
        for line in ax.lines:
            assert same_color(line.get_color(), color)

    # 测试绘制小提琴图时的线条宽度设置
    def test_linewidth(self, long_df):
        # 设置线条宽度为 5
        width = 5
        # 调用 violinplot 函数绘制小提琴图，并指定线条宽度
        ax = violinplot(long_df, x="a", y="y", linewidth=width)
        # 验证第一个多边形集合的线条宽度是否等于指定的宽度
        poly = ax.collections[0]
        assert poly.get_linewidth() == width

    # 测试绘制小提琴图时的颜色饱和度设置
    def test_saturation(self, long_df):
        # 设置颜色为 "#8912b0"，饱和度为 0.5
        color = "#8912b0"
        # 调用 violinplot 函数绘制小提琴图，并指定颜色和饱和度
        ax = violinplot(long_df["x"], color=color, saturation=.5)
        # 验证第一个多边形集合的面颜色饱和度是否与指定的颜色和饱和度相匹配
        poly = ax.collections[0]
        assert np.allclose(poly.get_facecolors()[0, :3], desaturate(color, 0.5))

    # 使用不同的内部显示方式参数化测试绘制小提琴图时的填充设置
    @pytest.mark.parametrize("inner", ["box", "quart", "stick", "point"])
    def test_fill(self, long_df, inner):
        # 设置颜色为 "#459900"
        color = "#459900"
        # 调用 violinplot 函数绘制小提琴图，并指定颜色、内部显示方式和填充为 False
        ax = violinplot(x=long_df["z"], fill=False, color=color, inner=inner)
        # 验证所有多边形的面颜色是否为空（因为 fill=False）
        for poly in ax.findobj(mpl.collections.PolyCollection):
            assert poly.get_facecolor().size == 0
            # 验证所有多边形的边框颜色是否与指定的颜色相同
            assert same_color(poly.get_edgecolor(), color)
        # 验证所有线条集合的颜色是否与指定的颜色相同
        for lines in ax.findobj(mpl.collections.LineCollection):
            assert same_color(lines.get_color(), color)
        # 验证所有线条的颜色是否与指定的颜色相同
        for line in ax.lines:
            assert same_color(line.get_color(), color)

    # 使用不同的方向参数化测试绘制小提琴图时的内部显示方式设置为 "box"
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_box(self, long_df, orient):
        # 获取指定方向的位置索引和值索引
        pos_idx, val_idx = self.orient_indices(orient)
        # 调用 violinplot 函数绘制小提琴图，并指定数据和方向
        ax = violinplot(long_df["y"], orient=orient)
        # 计算数据的箱线图统计信息
        stats = mpl.cbook.boxplot_stats(long_df["y"])[0]

        # 验证箱线图的须线位置是否与统计信息中的最小和最大值相匹配
        whiskers = ax.lines[0].get_xydata()
        assert whiskers[0, val_idx] == stats["whislo"]
        assert whiskers[1, val_idx] == stats["whishi"]
        # 验证箱线图的须线在指定方向上的位置是否为零
        assert whiskers[:, pos_idx].tolist() == [0, 0]

        # 验证箱线图的箱体位置是否与统计信息中的第一四分位数和第三四分位数相匹配
        box = ax.lines[1].get_xydata()
        assert box[0, val_idx] == stats["q1"]
        assert box[1, val_idx] == stats["q3"]
        # 验证箱线图的箱体在指定方向上的位置是否为零
        assert box[:, pos_idx].tolist() == [0, 0]

        # 验证箱线图的中位数位置是否与统计信息中的中位数相匹配
        median = ax.lines[2].get_xydata()
        assert median[0, val_idx] == stats["med"]
        # 验证箱线图的中位数在指定方向上的位置是否为零
        assert median[0, pos_idx] == 0

    # 使用不同的方向参数化测试绘制小提琴图时的内部显示方式设置为 "quart"
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_quartiles(self, long_df, orient):
        # 获取指定方向的位置索引和值索引
        pos_idx, val_idx = self.orient_indices(orient)
        # 调用 violinplot 函数绘制小提琴图，并指定数据、方向和内部显示方式
        ax = violinplot(long_df["y"], orient=orient, inner="quart")
        # 计算数据的四分位数
        quartiles = np.percentile(long_df["y"], [25, 50, 75])

        # 验证每个四分位数对应的线条位置是否与计算出的四分位数相匹配
        for q, line in zip(quartiles, ax.lines):
            pts = line.get_xydata()
            for pt in pts:
                assert pt[val_idx] == q
            # 验证线条在指定方向上的位置是否对称
            assert pts[0, pos_idx] == -pts[1, pos_idx]

    # 使用不同的方向参数化测试绘制小提琴图时的内部显示方式设置为 "stick" 或 "point"
    # 测试用例：测试在内部使用"stick"选项时的小提琴图表现
    def test_inner_stick(self, long_df, orient):
        # 根据方向获取位置索引和数值索引
        pos_idx, val_idx = self.orient_indices(orient)
        # 创建小提琴图，使用"stick"内部样式
        ax = violinplot(long_df["y"], orient=orient, inner="stick")
        # 遍历第二个集合中的线段
        for i, pts in enumerate(ax.collections[1].get_segments()):
            # 检查每个点的值索引是否与数据框中对应位置的值相等
            for pt in pts:
                assert pt[val_idx] == long_df["y"].iloc[i]
            # 检查线段的两个端点位置索引是否相反
            assert pts[0, pos_idx] == -pts[1, pos_idx]

    # 测试用例：测试在内部使用"points"选项时的小提琴图表现
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_inner_points(self, long_df, orient):
        # 根据方向获取位置索引和数值索引
        pos_idx, val_idx = self.orient_indices(orient)
        # 创建小提琴图，使用"points"内部样式
        ax = violinplot(long_df["y"], orient=orient, inner="points")
        # 获取第二个集合中的点
        points = ax.collections[1]
        # 遍历点的偏移量
        for i, pt in enumerate(points.get_offsets()):
            # 检查每个点的值索引是否与数据框中对应位置的值相等
            assert pt[val_idx] == long_df["y"].iloc[i]
            # 检查点的位置索引是否为0
            assert pt[pos_idx] == 0

    # 测试用例：测试单一分组小提琴图的表现
    def test_split_single(self, long_df):
        # 创建单一分组的小提琴图
        ax = violinplot(long_df, x="a", y="z", split=True, cut=0)
        # 获取分类顺序的水平
        levels = categorical_order(long_df["a"])
        # 遍历水平和其对应的数据
        for i, level in enumerate(levels):
            # 选择该水平对应的数据
            data = long_df.loc[long_df["a"] == level, "z"]
            # 检查每个小提琴的表现
            self.check_violin(ax.collections[i], data, "x", i)
            # 获取小提琴路径的顶点
            verts = ax.collections[i].get_paths()[0].vertices
            # 断言顶点的X坐标是否接近于i + .4的数量
            assert np.isclose(verts[:, 0], i + .4).sum() >= 100

    # 测试用例：测试多重分组小提琴图的表现
    def test_split_multi(self, long_df):
        # 创建多重分组的小提琴图
        ax = violinplot(long_df, x="a", y="z", hue="c", split=True, cut=0)
        # 迭代小提琴图的集合
        polys = iter(ax.collections)
        # 遍历水平和其对应的数据
        for i, level in enumerate(categorical_order(long_df["a"])):
            for j, hue_level in enumerate(categorical_order(long_df["c"])):
                # 选择符合条件的行数据
                rows = (long_df["a"] == level) & (long_df["c"] == hue_level)
                data = long_df.loc[rows, "z"]
                # 计算位置
                pos = i + [-.2, +.2][j]
                # 获取下一个小提琴图形对象
                poly = next(polys)
                # 检查每个小提琴的表现
                self.check_violin(poly, data, "x", pos, width=0.4)
                # 获取小提琴路径的顶点
                verts = poly.get_paths()[0].vertices
                # 断言顶点的X坐标是否接近于i的数量
                assert np.isclose(verts[:, 0], i).sum() >= 100

    # 测试用例：测试密度归一化为面积的小提琴图表现
    def test_density_norm_area(self, long_df):
        # 将数据转换为数组
        y = long_df["y"].to_numpy()
        # 创建密度归一化为面积的小提琴图
        ax = violinplot([y, y * 5], color="C0")
        # 存储每个小提琴的宽度
        widths = []
        # 遍历每个小提琴对象
        for poly in ax.collections:
            # 计算小提琴的宽度
            widths.append(self.violin_width(poly))
        # 断言第一个小提琴的宽度与第二个小提琴的宽度之比是否接近于5
        assert widths[0] / widths[1] == approx(5)

    # 测试用例：测试密度归一化为计数的小提琴图表现
    def test_density_norm_count(self, long_df):
        # 将数据转换为数组
        y = long_df["y"].to_numpy()
        # 创建密度归一化为计数的小提琴图
        ax = violinplot([np.repeat(y, 3), y], density_norm="count", color="C0")
        # 存储每个小提琴的宽度
        widths = []
        # 遍历每个小提琴对象
        for poly in ax.collections:
            # 计算小提琴的宽度
            widths.append(self.violin_width(poly))
        # 断言第一个小提琴的宽度与第二个小提琴的宽度之比是否接近于3
        assert widths[0] / widths[1] == approx(3)

    # 测试用例：测试密度归一化为宽度的小提琴图表现
    def test_density_norm_width(self, long_df):
        # 创建密度归一化为宽度的小提琴图
        ax = violinplot(long_df, x="a", y="y", density_norm="width")
        # 遍历每个小提琴对象
        for poly in ax.collections:
            # 检查每个小提琴的宽度是否接近于0.8
            assert self.violin_width(poly) == approx(0.8)
    # 测试用例：验证在常规化的小提琴图中，violinplot 函数使用 common_norm=True 参数
    def test_common_norm(self, long_df):

        # 调用 violinplot 函数生成小提琴图，并设置 common_norm=True
        ax = violinplot(long_df, x="a", y="y", hue="c", common_norm=True)

        # 初始化一个空列表，用于存储所有小提琴的宽度
        widths = []

        # 遍历 ax.collections 中的每一个图形对象
        for poly in ax.collections:
            # 调用 self.violin_width 方法获取当前小提琴的宽度，并添加到 widths 列表中
            widths.append(self.violin_width(poly))

        # 断言：确保宽度大于0.3999的小提琴数量为1
        assert sum(w > 0.3999 for w in widths) == 1

    # 测试用例：验证在使用 scale="count" 参数时，violinplot 函数会发出 FutureWarning 警告
    def test_scale_deprecation(self, long_df):

        # 使用 pytest.warns 检查是否有 FutureWarning 警告，并匹配特定的消息格式
        with pytest.warns(FutureWarning, match=r".+Pass `density_norm='count'`"):
            violinplot(long_df, x="a", y="y", hue="b", scale="count")

    # 测试用例：验证在使用 scale_hue=False 参数时，violinplot 函数会发出 FutureWarning 警告
    def test_scale_hue_deprecation(self, long_df):

        # 使用 pytest.warns 检查是否有 FutureWarning 警告，并匹配特定的消息格式
        with pytest.warns(FutureWarning, match=r".+Pass `common_norm=True`"):
            violinplot(long_df, x="a", y="y", hue="b", scale_hue=False)

    # 测试用例：验证在调整带宽（bw_adjust）时，violinplot 函数生成的核密度估计结果的标准差的差异
    def test_bw_adjust(self, long_df):

        # 调用 violinplot 函数，调整 bw_adjust 参数为 0.2
        ax = violinplot(long_df["y"], bw_adjust=.2)
        
        # 再次调用 violinplot 函数，调整 bw_adjust 参数为 2，不过结果未使用
        violinplot(long_df["y"], bw_adjust=2)

        # 从 ax.collections 中获取第一个集合的第一个路径，并限制顶点为前100个，以获取 kde1
        kde1 = ax.collections[0].get_paths()[0].vertices[:100, 0]

        # 从 ax.collections 中获取第二个集合的第一个路径，并限制顶点为前100个，以获取 kde2
        kde2 = ax.collections[1].get_paths()[0].vertices[:100, 0]

        # 断言：确保 kde1 的标准差大于 kde2 的标准差
        assert np.std(np.diff(kde1)) > np.std(np.diff(kde2))

    # 测试用例：验证在使用 bw="silverman" 参数时，violinplot 函数会发出 FutureWarning 警告
    def test_bw_deprecation(self, long_df):

        # 使用 pytest.warns 检查是否有 FutureWarning 警告，并匹配特定的消息格式
        with pytest.warns(FutureWarning, match=r".*Setting `bw_method='silverman'`"):
            violinplot(long_df["y"], bw="silverman")

    # 测试用例：验证在设置 gap 参数时，violinplot 函数生成的两个小提琴图的间隔值是否正确
    def test_gap(self, long_df):

        # 调用 violinplot 函数，设置 gap 参数为 0.2
        ax = violinplot(long_df, y="y", hue="c", gap=.2)

        # 获取第一个集合的第一个路径的最大 x 值
        a = ax.collections[0].get_paths()[0].vertices[:, 0].max()

        # 获取第二个集合的第一个路径的最小 x 值
        b = ax.collections[1].get_paths()[0].vertices[:, 0].min()

        # 断言：确保两个小提琴图之间的距离等于预期的计算结果
        assert (b - a) == approx(0.2 * 0.8 / 2)

    # 测试用例：验证在使用 inner="stick" 参数时，violinplot 函数生成的内部线条的线宽度是否正确
    def test_inner_kws(self, long_df):

        # 定义一个包含 "linewidth": 3 的字典 kws
        kws = {"linewidth": 3}

        # 调用 violinplot 函数，设置 inner="stick" 和 inner_kws 参数为 kws
        ax = violinplot(long_df, x="a", y="y", inner="stick", inner_kws=kws)

        # 遍历 ax.lines 中的每一条线条
        for line in ax.lines:
            # 断言：确保每条线的线宽度与 kws 中定义的值相等
            assert line.get_linewidth() == kws["linewidth"]

    # 测试用例：验证在设置 inner_kws 参数时，violinplot 函数生成的不同组件的线宽度和标记是否正确
    def test_box_inner_kws(self, long_df):

        # 定义一个包含多个键值对的字典 kws
        kws = {"box_width": 10, "whis_width": 2, "marker": "x"}

        # 调用 violinplot 函数，设置 inner_kws 参数为 kws
        ax = violinplot(long_df, x="a", y="y", inner_kws=kws)

        # 遍历 ax.lines 中每隔三个索引位置的线条，用于箱线图的线宽度检查
        for line in ax.lines[::3]:
            # 断言：确保每条线的线宽度与 kws 中定义的 whis_width 值相等
            assert line.get_linewidth() == kws["whis_width"]

        # 遍历 ax.lines 中每隔三个索引位置偏移1的线条，用于箱线图的盒子线宽度检查
        for line in ax.lines[1::3]:
            # 断言：确保每条线的线宽度与 kws 中定义的 box_width 值相等
            assert line.get_linewidth() == kws["box_width"]

        # 遍历 ax.lines 中每隔三个索引位置偏移2的线条，用于盒子图的标记检查
        for line in ax.lines[2::3]:
            # 断言：确保每条线的标记类型与 kws 中定义的 marker 值相等
            assert line.get_marker() == kws["marker"]
    @pytest.mark.parametrize(
        "kwargs",  # 定义参数化测试，参数为 kwargs
        [  # 参数列表开始
            dict(data="wide"),  # 使用字典定义参数 data="wide"
            dict(data="wide", orient="h"),  # 使用字典定义参数 data="wide", orient="h"
            dict(data="flat"),  # 使用字典定义参数 data="flat"
            dict(data="long", x="a", y="y"),  # 使用字典定义参数 data="long", x="a", y="y"
            dict(data=None, x="a", y="y", split=True),  # 使用字典定义参数 data=None, x="a", y="y", split=True
            dict(data="long", x="a", y="y", hue="a"),  # 使用字典定义参数 data="long", x="a", y="y", hue="a"
            dict(data=None, x="a", y="y", hue="a"),  # 使用字典定义参数 data=None, x="a", y="y", hue="a"
            dict(data="long", x="a", y="y", hue="b"),  # 使用字典定义参数 data="long", x="a", y="y", hue="b"
            dict(data=None, x="s", y="y", hue="a"),  # 使用字典定义参数 data=None, x="s", y="y", hue="a"
            dict(data="long", x="a", y="y", hue="s", split=True),  # 使用字典定义参数 data="long", x="a", y="y", hue="s", split=True
            dict(data="null", x="a", y="y", hue="a"),  # 使用字典定义参数 data="null", x="a", y="y", hue="a"
            dict(data="long", x="s", y="y", hue="a", native_scale=True),  # 使用字典定义参数 data="long", x="s", y="y", hue="a", native_scale=True
            dict(data="long", x="d", y="y", hue="a", native_scale=True),  # 使用字典定义参数 data="long", x="d", y="y", hue="a", native_scale=True
            dict(data="null", x="a", y="y", hue="b", fill=False, gap=.2),  # 使用字典定义参数 data="null", x="a", y="y", hue="b", fill=False, gap=.2
            dict(data="null", x="a", y="y", linecolor="r", linewidth=5),  # 使用字典定义参数 data="null", x="a", y="y", linecolor="r", linewidth=5
            dict(data="long", x="a", y="y", inner="stick"),  # 使用字典定义参数 data="long", x="a", y="y", inner="stick"
            dict(data="long", x="a", y="y", inner="points"),  # 使用字典定义参数 data="long", x="a", y="y", inner="points"
            dict(data="long", x="a", y="y", hue="b", inner="quartiles", split=True),  # 使用字典定义参数 data="long", x="a", y="y", hue="b", inner="quartiles", split=True
            dict(data="long", x="a", y="y", density_norm="count", common_norm=True),  # 使用字典定义参数 data="long", x="a", y="y", density_norm="count", common_norm=True
            dict(data="long", x="a", y="y", bw_adjust=2),  # 使用字典定义参数 data="long", x="a", y="y", bw_adjust=2
        ]  # 参数列表结束
    )  # 参数化测试装饰器结束
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):
        # 定义参数化测试函数，接受 long_df, wide_df, null_df, flat_series 和 kwargs 参数

        if kwargs["data"] == "long":  # 如果 kwargs 中的 data 等于 "long"
            kwargs["data"] = long_df  # 将 kwargs 中的 data 替换为 long_df
        elif kwargs["data"] == "wide":  # 如果 kwargs 中的 data 等于 "wide"
            kwargs["data"] = wide_df  # 将 kwargs 中的 data 替换为 wide_df
        elif kwargs["data"] == "flat":  # 如果 kwargs 中的 data 等于 "flat"
            kwargs["data"] = flat_series  # 将 kwargs 中的 data 替换为 flat_series
        elif kwargs["data"] == "null":  # 如果 kwargs 中的 data 等于 "null"
            kwargs["data"] = null_df  # 将 kwargs 中的 data 替换为 null_df
        elif kwargs["data"] is None:  # 如果 kwargs 中的 data 为 None
            for var in ["x", "y", "hue"]:  # 遍历 ["x", "y", "hue"]
                if var in kwargs:  # 如果 kwargs 中包含 var
                    kwargs[var] = long_df[kwargs[var]]  # 将 kwargs[var] 替换为 long_df[kwargs[var]]

        ax = violinplot(**kwargs)  # 调用 violinplot 函数，传入 kwargs 的参数，并将返回值赋给 ax
        g = catplot(**kwargs, kind="violin")  # 调用 catplot 函数，传入 kwargs 的参数和额外的 kind="violin" 参数，并将返回值赋给 g

        assert_plots_equal(ax, g.ax)  # 断言 ax 和 g.ax 相等
# 创建一个名为 TestBarPlot 的测试类，继承自 SharedAggTests
class TestBarPlot(SharedAggTests):

    # 将 barplot 方法设为静态方法 func
    func = staticmethod(barplot)

    # 为测试方法提供公共参数的 fixture
    @pytest.fixture
    def common_kws(self):
        return {"saturation": 1}

    # 获取最后一个容器中的颜色，返回其 RGBA 表示
    def get_last_color(self, ax):
        # 获取最后一个容器中所有条形图的颜色
        colors = [p.get_facecolor() for p in ax.containers[-1]]
        # 找出唯一的颜色
        unique_colors = np.unique(colors, axis=0)
        assert len(unique_colors) == 1  # 断言颜色列表中只有一个唯一颜色
        return to_rgba(unique_colors.squeeze())  # 返回唯一颜色的 RGBA 表示

    # 使用参数化测试 "orient"，测试单变量情况
    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_single_var(self, orient):
        # 创建包含数值的 Series
        vals = pd.Series([1, 3, 10])
        # 调用 barplot 函数生成图表，并根据 orient 参数绘制条形图
        ax = barplot(**{orient: vals})
        # 获取第一个条形图对象
        bar, = ax.patches
        # 根据 orient 参数获取属性名称
        prop = {"x": "width", "y": "height"}[orient]
        # 断言条形图的宽度或高度等于数值的均值
        assert getattr(bar, f"get_{prop}")() == approx(vals.mean())

    # 使用参数化测试 "orient"，测试宽格式数据框情况
    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_wide_df(self, wide_df, orient):
        # 调用 barplot 函数生成图表，并根据 orient 参数绘制条形图
        ax = barplot(wide_df, orient=orient)
        # 根据 orient 参数确定属性名称
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        prop = {"x": "height", "y": "width"}[orient]
        # 遍历所有条形图，断言其高度或宽度等于数据框中对应列的均值
        for i, bar in enumerate(ax.patches):
            assert getattr(bar, f"get_{prop}")() == approx(wide_df.iloc[:, i].mean())

    # 使用参数化测试 "orient"，测试向量导向的情况
    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_vector_orient(self, orient):
        # 定义键和值的列表
        keys, vals = ["a", "b", "c"], [1, 2, 3]
        # 创建包含键值对的字典
        data = dict(zip(keys, vals))
        # 根据 orient 参数确定属性名称
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        prop = {"x": "height", "y": "width"}[orient]
        # 调用 barplot 函数生成图表，并根据 orient 参数绘制条形图
        ax = barplot(data, orient=orient)
        # 遍历所有条形图，断言其 x 或 y 值等于索引减去 0.4 的近似值
        for i, bar in enumerate(ax.patches):
            assert getattr(bar, f"get_{orient}")() == approx(i - 0.4)
            # 断言条形图的高度或宽度等于对应值的近似值
            assert getattr(bar, f"get_{prop}")() == approx(vals[i])

    # 测试 xy 垂直方向的条形图绘制
    def test_xy_vertical(self):
        # 定义 x 和 y 的列表
        x, y = ["a", "b", "c"], [1, 3, 2.5]
        # 调用 barplot 函数生成图表，并根据 x 和 y 参数绘制垂直条形图
        ax = barplot(x=x, y=y)
        # 遍历所有条形图，断言其 x 值加上宽度一半等于索引的近似值
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == approx(0)  # 断言条形图的 y 值近似为 0
            assert bar.get_height() == approx(y[i])  # 断言条形图的高度近似为 y 列表中的值
            assert bar.get_width() == approx(0.8)  # 断言条形图的宽度近似为 0.8

    # 测试 xy 水平方向的条形图绘制
    def test_xy_horizontal(self):
        # 定义 x 和 y 的列表
        x, y = [1, 3, 2.5], ["a", "b", "c"]
        # 调用 barplot 函数生成图表，并根据 x 和 y 参数绘制水平条形图
        ax = barplot(x=x, y=y)
        # 遍历所有条形图，断言其 y 值加上高度一半等于索引的近似值
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == approx(0)  # 断言条形图的 x 值近似为 0
            assert bar.get_y() + bar.get_height() / 2 == approx(i)
            assert bar.get_height() == approx(0.8)  # 断言条形图的高度近似为 0.8
            assert bar.get_width() == approx(x[i])  # 断言条形图的宽度近似为 x 列表中的值

    # 测试 xy 数据中包含 NA 分组器的情况
    def test_xy_with_na_grouper(self):
        # 定义 x 和 y 的列表，其中包含 None
        x, y = ["a", None, "b"], [1, 2, 3]
        # 调用 barplot 函数生成图表，并根据 x 和 y 参数绘制条形图
        ax = barplot(x=x, y=y)
        _draw_figure(ax.figure)  # 对于 matplotlib<3.5，绘制图表
        assert ax.get_xticks() == [0, 1]  # 断言 x 轴刻度值为 [0, 1]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]  # 断言 x 轴刻度标签为 ["a", "b"]
        assert ax.patches[0].get_height() == 1  # 断言第一个条形图的高度为 1
        assert ax.patches[1].get_height() == 3  # 断言第二个条形图的高度为 3
    # 定义一个测试函数，测试在存在缺失值的情况下绘制柱状图
    def test_xy_with_na_value(self):
        # 设定测试数据，x为类别列表，y为数值列表，其中包含一个None值
        x, y = ["a", "b", "c"], [1, None, 3]
        # 调用barplot函数绘制柱状图，并返回绘图的Axes对象
        ax = barplot(x=x, y=y)
        # 绘制图形，针对matplotlib版本低于3.5的兼容处理
        _draw_figure(ax.figure)  # For matplotlib<3.5
        # 断言检查柱状图的x轴刻度值是否符合预期
        assert ax.get_xticks() == [0, 1, 2]
        # 断言检查柱状图的x轴刻度标签文本是否符合预期
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
        # 断言检查第一个柱子的高度是否符合预期
        assert ax.patches[0].get_height() == 1
        # 断言检查第二个柱子的高度是否符合预期（包含了None值的处理）
        assert ax.patches[1].get_height() == 3

    # 定义一个测试函数，测试在指定色调下绘制柱状图
    def test_hue_redundant(self):
        # 设定测试数据，x为类别列表，y为数值列表
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 调用barplot函数绘制柱状图，同时指定色调和饱和度
        ax = barplot(x=x, y=y, hue=x, saturation=1)
        # 遍历每个柱子，进行多个断言检查柱状图的各项属性是否符合预期
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8)
            # 断言检查柱子颜色是否符合预期，使用C{i}来匹配颜色
            assert same_color(bar.get_facecolor(), f"C{i}")

    # 定义一个测试函数，测试在匹配的色调下绘制柱状图
    def test_hue_matched(self):
        # 设定测试数据，x为类别列表，y为数值列表，hue为色调列表
        x, y = ["a", "b", "c"], [1, 2, 3]
        hue = ["x", "x", "y"]
        # 调用barplot函数绘制柱状图，同时指定色调、饱和度和是否显示图例
        ax = barplot(x=x, y=y, hue=hue, saturation=1, legend=False)
        # 遍历每个柱子，进行多个断言检查柱状图的各项属性是否符合预期
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8)
            # 断言检查柱子颜色是否符合预期，使用C{i // 2}来匹配颜色
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    # 定义一个测试函数，测试在匹配的色调名称下绘制柱状图
    def test_hue_matched_by_name(self):
        # 设定测试数据，数据为字典形式，包含类别x和对应数值y的列表
        data = {"x": ["a", "b", "c"], "y": [1, 2, 3]}
        # 调用barplot函数绘制柱状图，指定x和y的数据列，hue为x列，饱和度为1
        ax = barplot(data, x="x", y="y", hue="x", saturation=1)
        # 遍历每个柱子，进行多个断言检查柱状图的各项属性是否符合预期
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == data["y"][i]
            assert bar.get_width() == approx(0.8)
            # 断言检查柱子颜色是否符合预期，使用C{i}来匹配颜色
            assert same_color(bar.get_facecolor(), f"C{i}")

    # 定义一个测试函数，测试在分组的色调下绘制柱状图
    def test_hue_dodged(self):
        # 设定测试数据，x为类别列表，y为数值列表，hue为分组依据
        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        # 调用barplot函数绘制柱状图，指定x和y的数据列，hue为分组依据，饱和度为1，不显示图例
        ax = barplot(x=x, y=y, hue=hue, saturation=1, legend=False)
        # 遍历每个柱子，进行多个断言检查柱状图的各项属性是否符合预期
        for i, bar in enumerate(ax.patches):
            sign = 1 if i // 2 else -1
            assert bar.get_x() + bar.get_width() / 2 == approx(i % 2 + sign * 0.8 / 4)
            assert bar.get_y() == 0
            assert bar.get_height() == y[i]
            assert bar.get_width() == approx(0.8 / 2)
            # 断言检查柱子颜色是否符合预期，使用C{i // 2}来匹配颜色
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    # 定义一个测试函数，测试在指定间隙下绘制柱状图
    def test_gap(self):
        # 设定测试数据，x为类别列表，y为数值列表，hue为分组依据
        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        # 调用barplot函数绘制柱状图，指定x和y的数据列，hue为分组依据，间隙为0.25，不显示图例
        ax = barplot(x=x, y=y, hue=hue, gap=.25, legend=False)
        # 遍历每个柱子，进行断言检查柱状图的宽度是否符合预期
        for i, bar in enumerate(ax.patches):
            assert bar.get_width() == approx(0.8 / 2 * .75)
    # 测试未调节的色调功能
    def test_hue_undodged(self):
        # 创建示例数据
        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        # 调用 barplot 函数生成柱状图，禁用调节和图例
        ax = barplot(x=x, y=y, hue=hue, saturation=1, dodge=False, legend=False)
        
        # 遍历每一个柱形图条目
        for i, bar in enumerate(ax.patches):
            # 断言柱形图条目的中心 x 坐标是否符合预期
            assert bar.get_x() + bar.get_width() / 2 == approx(i % 2)
            # 断言柱形图条目的起始 y 坐标是否为 0
            assert bar.get_y() == 0
            # 断言柱形图条目的高度是否符合预期
            assert bar.get_height() == y[i]
            # 断言柱形图条目的宽度是否符合预期
            assert bar.get_width() == approx(0.8)
            # 断言柱形图条目的填充颜色是否与预期一致
            assert same_color(bar.get_facecolor(), f"C{i // 2}")

    # 测试指定色调顺序的功能
    def test_hue_order(self):
        # 创建示例数据
        x, y = ["a", "b", "c"], [1, 2, 3]
        hue_order = ["c", "b", "a"]

        # 调用 barplot 函数生成柱状图，使用指定的色调顺序和默认饱和度
        ax = barplot(x=x, y=y, hue=x, hue_order=hue_order, saturation=1)
        
        # 遍历每一个柱形图条目
        for i, bar in enumerate(ax.patches):
            # 断言柱形图条目的填充颜色是否与预期一致
            assert same_color(bar.get_facecolor(), f"C{i}")
            # 断言柱形图条目的中心 x 坐标是否符合预期
            assert bar.get_x() + bar.get_width() / 2 == approx(2 - i)

    # 测试自定义色调规范化的功能
    def test_hue_norm(self):
        # 创建示例数据
        x, y = [1, 2, 3, 4], [1, 2, 3, 4]

        # 调用 barplot 函数生成柱状图，使用自定义的色调规范化
        ax = barplot(x=x, y=y, hue=x, hue_norm=(2, 3))
        
        # 获取所有柱形图条目的填充颜色
        colors = [bar.get_facecolor() for bar in ax.patches]
        
        # 断言柱形图条目的填充颜色符合预期
        assert colors[0] == colors[1]
        assert colors[1] != colors[2]
        assert colors[2] == colors[3]

    # 测试未填充柱形图的功能
    def test_fill(self):
        # 创建示例数据
        x = ["a", "b", "a", "b"]
        y = [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        # 调用 barplot 函数生成柱状图，禁用填充和图例
        ax = barplot(x=x, y=y, hue=hue, fill=False, legend=False)
        
        # 遍历每一个柱形图条目
        for i, bar in enumerate(ax.patches):
            # 断言柱形图条目的边缘颜色是否与预期一致
            assert same_color(bar.get_edgecolor(), f"C{i // 2}")
            # 断言柱形图条目的填充颜色是否为透明
            assert same_color(bar.get_facecolor(), (0, 0, 0, 0))

    # 测试使用原生比例的 x 和 y 轴的功能
    def test_xy_native_scale(self):
        # 创建示例数据
        x, y = [2, 4, 8], [1, 2, 3]

        # 调用 barplot 函数生成柱状图，启用原生比例的 x 轴
        ax = barplot(x=x, y=y, native_scale=True)
        
        # 遍历每一个柱形图条目
        for i, bar in enumerate(ax.patches):
            # 断言柱形图条目的中心 x 坐标是否符合预期
            assert bar.get_x() + bar.get_width() / 2 == approx(x[i])
            # 断言柱形图条目的起始 y 坐标是否为 0
            assert bar.get_y() == 0
            # 断言柱形图条目的高度是否符合预期
            assert bar.get_height() == y[i]
            # 断言柱形图条目的宽度是否符合预期
            assert bar.get_width() == approx(0.8 * 2)

    # 测试对数变换的原生比例 x 轴的功能
    def test_xy_native_scale_log_transform(self):
        # 创建示例数据
        x, y = [1, 10, 100], [1, 2, 3]

        # 创建 Matplotlib 的 Figure 对象，并在其上创建子图
        ax = mpl.figure.Figure().subplots()
        # 设置 x 轴为对数刻度
        ax.set_xscale("log")
        
        # 调用 barplot 函数生成柱状图，启用原生比例 x 轴和指定的坐标轴
        barplot(x=x, y=y, native_scale=True, ax=ax)
        
        # 遍历每一个柱形图条目
        for i, bar in enumerate(ax.patches):
            # 计算柱形图条目的中心 x 坐标对应的原始数值
            x0, x1 = np.log10([bar.get_x(), bar.get_x() + bar.get_width()])
            center = 10 ** (x0 + (x1 - x0) / 2)
            # 断言柱形图条目的中心 x 坐标是否符合预期
            assert center == approx(x[i])
            # 断言柱形图条目的起始 y 坐标是否为 0
            assert bar.get_y() == 0
            # 断言柱形图条目的高度是否符合预期
            assert bar.get_height() == y[i]
        # 断言后一个柱形图条目的宽度大于前一个柱形图条目的宽度
        assert ax.patches[1].get_width() > ax.patches[0].get_width()

    # 测试日期时间原生比例 x 轴的功能
    def test_datetime_native_scale_axis(self):
        # 创建示例数据
        x = pd.date_range("2010-01-01", periods=20, freq="MS")
        y = np.arange(20)

        # 调用 barplot 函数生成柱状图，启用原生比例 x 轴
        ax = barplot(x=x, y=y, native_scale=True)
        
        # 断言 x 轴的主要刻度定位器类的类名中包含 "Date"
        assert "Date" in ax.xaxis.get_major_locator().__class__.__name__
        # 检查一个特定日期是否能正确地转换为 Matplotlib 的日期格式
        day = "2003-02-28"
        assert_array_equal(ax.xaxis.convert_units([day]), mpl.dates.date2num([day]))
    def test_native_scale_dodged(self):
        # 定义测试数据 x, y 和 hue
        x, y = [2, 4, 2, 4], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        # 调用 barplot 函数，设置 native_scale=True，并获取返回的图形对象 ax
        ax = barplot(x=x, y=y, hue=hue, native_scale=True)

        # 验证前两个条形图的位置和宽度
        for x_i, bar in zip(x[:2], ax.patches[:2]):
            assert bar.get_x() + bar.get_width() == approx(x_i)
        
        # 验证后两个条形图的位置
        for x_i, bar in zip(x[2:], ax.patches[2:]):
            assert bar.get_x() == approx(x_i)

    def test_native_scale_log_transform_dodged(self):
        # 定义测试数据 x, y 和 hue
        x, y = [1, 100, 1, 100], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]

        # 创建一个带有对数坐标轴的图形对象 ax
        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")

        # 调用 barplot 函数，设置 native_scale=True，并传入预设的图形对象 ax
        barplot(x=x, y=y, hue=hue, native_scale=True, ax=ax)

        # 验证前两个条形图的位置和宽度
        for x_i, bar in zip(x[:2], ax.patches[:2]):
            assert bar.get_x() + bar.get_width() == approx(x_i)
        
        # 验证后两个条形图的位置
        for x_i, bar in zip(x[2:], ax.patches[2:]):
            assert bar.get_x() == approx(x_i)

    def test_estimate_default(self, long_df):
        # 设置聚合变量和值变量
        agg_var, val_var = "a", "y"

        # 对长格式数据 long_df 按 agg_var 分组，计算 val_var 的均值
        agg_df = long_df.groupby(agg_var)[val_var].mean()

        # 调用 barplot 函数，显示 x=agg_var, y=val_var 的条形图，无误差条
        ax = barplot(long_df, x=agg_var, y=val_var, errorbar=None)

        # 获取分类顺序
        order = categorical_order(long_df[agg_var])

        # 验证每个条形图的高度是否等于对应分组的均值
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(agg_df[order[i]])

    def test_estimate_string(self, long_df):
        # 设置聚合变量和值变量
        agg_var, val_var = "a", "y"

        # 对长格式数据 long_df 按 agg_var 分组，计算 val_var 的中位数
        agg_df = long_df.groupby(agg_var)[val_var].median()

        # 调用 barplot 函数，显示 x=agg_var, y=val_var 的条形图，估算值为中位数，无误差条
        ax = barplot(long_df, x=agg_var, y=val_var, estimator="median", errorbar=None)

        # 获取分类顺序
        order = categorical_order(long_df[agg_var])

        # 验证每个条形图的高度是否等于对应分组的中位数
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(agg_df[order[i]])

    def test_estimate_func(self, long_df):
        # 设置聚合变量和值变量
        agg_var, val_var = "a", "y"

        # 对长格式数据 long_df 按 agg_var 分组，计算 val_var 的中位数
        agg_df = long_df.groupby(agg_var)[val_var].median()

        # 调用 barplot 函数，显示 x=agg_var, y=val_var 的条形图，估算值为 np.median 函数计算结果，无误差条
        ax = barplot(long_df, x=agg_var, y=val_var, estimator=np.median, errorbar=None)

        # 获取分类顺序
        order = categorical_order(long_df[agg_var])

        # 验证每个条形图的高度是否等于对应分组的中位数
        for i, bar in enumerate(ax.patches):
            assert bar.get_height() == approx(agg_df[order[i]])

    def test_weighted_estimate(self, long_df):
        # 调用 barplot 函数，显示 y="y"，权重为 x 的条形图
        ax = barplot(long_df, y="y", weights="x")

        # 获取第一个条形图的高度
        height = ax.patches[0].get_height()

        # 计算长格式数据 long_df 的 "y" 列加权平均值
        expected = np.average(long_df["y"], weights=long_df["x"])

        # 验证第一个条形图的高度是否等于加权平均值
        assert height == expected

    def test_estimate_log_transform(self, long_df):
        # 创建一个带有对数坐标轴的图形对象 ax
        ax = mpl.figure.Figure().subplots()
        ax.set_xscale("log")

        # 调用 barplot 函数，显示 x=long_df["z"] 的条形图，预设图形对象 ax
        barplot(x=long_df["z"], ax=ax)

        # 获取第一个条形图对象 bar
        bar, = ax.patches

        # 验证第一个条形图的宽度是否等于 long_df["z"] 取对数后的平均值的 10 的对数
        assert bar.get_width() == 10 ** np.log10(long_df["z"]).mean()
    # 定义一个测试函数，用于测试带有误差线的条形图的生成
    def test_errorbars(self, long_df):
        # 设定聚合变量和数值变量
        agg_var, val_var = "a", "y"
        # 对长格式数据进行按聚合变量分组，并计算数值变量的均值和标准差
        agg_df = long_df.groupby(agg_var)[val_var].agg(["mean", "std"])

        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax
        ax = barplot(long_df, x=agg_var, y=val_var, errorbar="sd")
        # 根据长格式数据中聚合变量的顺序，确定条形图中条的顺序
        order = categorical_order(long_df[agg_var])
        # 遍历条形图中的每一条线
        for i, line in enumerate(ax.lines):
            # 获取当前聚合变量对应的均值和标准差
            row = agg_df.loc[order[i]]
            # 获取误差线的上下界数据
            lo, hi = line.get_ydata()
            # 断言：下界应接近于均值减去标准差
            assert lo == approx(row["mean"] - row["std"])
            # 断言：上界应接近于均值加上标准差
            assert hi == approx(row["mean"] + row["std"])

    # 定义一个测试函数，测试条形图的宽度设置
    def test_width(self):
        # 设定条形图的宽度
        width = .5
        # 设定 x 和 y 轴数据
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax
        ax = barplot(x=x, y=y, width=width)
        # 遍历条形图中的每一个条形
        for i, bar in enumerate(ax.patches):
            # 断言：每个条形的中心横坐标应接近于索引 i
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            # 断言：每个条形的宽度应等于设定的宽度
            assert bar.get_width() == width

    # 定义一个测试函数，测试带有原生比例尺的条形图的宽度设置
    def test_width_native_scale(self):
        # 设定条形图的宽度
        width = .5
        # 设定 x 和 y 轴数据
        x, y = [4, 6, 10], [1, 2, 3]
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax，使用原生比例尺
        ax = barplot(x=x, y=y, width=width, native_scale=True)
        # 遍历条形图中的每一个条形
        for bar in ax.patches:
            # 断言：每个条形的宽度应为设定宽度的两倍
            assert bar.get_width() == (width * 2)

    # 定义一个测试函数，测试带有间隔类别的条形图的宽度设置
    def test_width_spaced_categories(self):
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax
        ax = barplot(x=["a", "b", "c"], y=[4, 5, 6])
        # 再次调用 barplot 函数生成另一个条形图，并添加到同一轴对象 ax 上
        barplot(x=["a", "c"], y=[1, 3], ax=ax)
        # 遍历条形图中的每一个条形
        for bar in ax.patches:
            # 断言：每个条形的宽度应接近于 0.8
            assert bar.get_width() == pytest.approx(0.8)

    # 定义一个测试函数，测试条形图的饱和度颜色设置
    def test_saturation_color(self):
        # 设定颜色
        color = (.1, .9, .2)
        # 设定 x 和 y 轴数据
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax
        ax = barplot(x=x, y=y)
        # 遍历条形图中的每一个条形
        for bar in ax.patches:
            # 断言：每个条形的颜色的 RGB 方差应小于设定颜色的 RGB 方差
            assert np.var(bar.get_facecolor()[:3]) < np.var(color)

    # 定义一个测试函数，测试条形图的饱和度调色板设置
    def test_saturation_palette(self):
        # 设定调色板
        palette = color_palette("viridis", 3)
        # 设定 x 和 y 轴数据
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax，使用指定调色板
        ax = barplot(x=x, y=y, hue=x, palette=palette)
        # 遍历条形图中的每一个条形
        for i, bar in enumerate(ax.patches):
            # 断言：每个条形的颜色的 RGB 方差应小于对应调色板中颜色的 RGB 方差
            assert np.var(bar.get_facecolor()[:3]) < np.var(palette[i])

    # 定义一个测试函数，测试自动生成数值型图例
    def test_legend_numeric_auto(self, long_df):
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax
        ax = barplot(long_df, x="x", y="y", hue="x")
        # 断言：图例中的文本数量不超过 6
        assert len(ax.get_legend().texts) <= 6

    # 定义一个测试函数，测试完整显示数值型图例
    def test_legend_numeric_full(self, long_df):
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax，指定完整图例显示
        ax = barplot(long_df, x="x", y="y", hue="x", legend="full")
        # 获取图例中显示的文本列表
        labels = [t.get_text() for t in ax.get_legend().texts]
        # 获取长格式数据中 x 列的唯一值，并排序后转换为字符串列表
        levels = [str(x) for x in sorted(long_df["x"].unique())]
        # 断言：图例中的文本列表应与长格式数据中 x 列的唯一值（排序后）一致
        assert labels == levels

    # 定义一个测试函数，测试禁用图例显示
    def test_legend_disabled(self, long_df):
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax，禁用图例显示
        ax = barplot(long_df, x="x", y="y", hue="b", legend=False)
        # 断言：条形图对象中不应包含图例
        assert ax.get_legend() is None

    # 定义一个测试函数，测试误差线末端标记
    def test_error_caps(self):
        # 设定 x 和 y 轴数据
        x, y = ["a", "b", "c"] * 2, [1, 2, 3, 4, 5, 6]
        # 调用 barplot 函数生成条形图，并将返回的轴对象赋给 ax，设定误差线标记为 pi
        ax = barplot(x=x, y=y, capsize=.8, errorbar="pi")

        # 断言：条形图中的条形数量应与误差线的数量相同
        assert len(ax.patches) == len(ax.lines)
        # 遍历条形图中的每一个条形和对应的误差线
        for bar, error in zip(ax.patches, ax.lines):
            # 获取误差线的
    `
    # 定义一个测试函数，测试在原生比例下发生错误时的情况
    def test_error_caps_native_scale(self):
        # 定义两个列表，x 包含多个重复的数值，y 包含一组数值
        x, y = [2, 4, 20] * 2, [1, 2, 3, 4, 5, 6]
        # 调用 barplot 函数绘制条形图，设置误差线帽大小为 0.8，使用原生比例，误差条形类型为 "pi"
        ax = barplot(x=x, y=y, capsize=.8, native_scale=True, errorbar="pi")
    
        # 断言条形图的数量等于误差线的数量
        assert len(ax.patches) == len(ax.lines)
        # 遍历每个条形和对应的误差线
        for bar, error in zip(ax.patches, ax.lines):
            # 获取误差线的位置数据
            pos = error.get_xdata()
            # 断言位置数据的长度为 8
            assert len(pos) == 8
            # 断言误差线的最小值近似等于条形的 x 坐标
            assert np.nanmin(pos) == approx(bar.get_x())
            # 断言误差线的最大值近似等于条形的 x 坐标加上条形的宽度
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())
    
    # 定义一个测试函数，测试对数变换下发生错误时的情况
    def test_error_caps_native_scale_log_transform(self):
        # 定义两个列表，x 包含多个数值，y 包含一组数值
        x, y = [1, 10, 1000] * 2, [1, 2, 3, 4, 5, 6]
        # 创建一个 Figure 对象并获取子图
        ax = mpl.figure.Figure().subplots()
        # 设置子图 x 轴为对数尺度
        ax.set_xscale("log")
        # 调用 barplot 函数绘制条形图，设置误差线帽大小为 0.8，使用原生比例，误差条形类型为 "pi"，并指定子图 ax
        barplot(x=x, y=y, capsize=.8, native_scale=True, errorbar="pi", ax=ax)
    
        # 断言条形图的数量等于误差线的数量
        assert len(ax.patches) == len(ax.lines)
        # 遍历每个条形和对应的误差线
        for bar, error in zip(ax.patches, ax.lines):
            # 获取误差线的位置数据
            pos = error.get_xdata()
            # 断言位置数据的长度为 8
            assert len(pos) == 8
            # 断言误差线的最小值近似等于条形的 x 坐标
            assert np.nanmin(pos) == approx(bar.get_x())
            # 断言误差线的最大值近似等于条形的 x 坐标加上条形的宽度
            assert np.nanmax(pos) == approx(bar.get_x() + bar.get_width())
    
    # 定义一个测试函数，测试条形图的参数设置
    def test_bar_kwargs(self):
        # 定义两个列表，x 包含多个字符串，y 包含一组数值
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 定义一个包含条形图参数的字典
        kwargs = dict(linewidth=3, facecolor=(.5, .4, .3, .2), rasterized=True)
        # 调用 barplot 函数绘制条形图，传入参数 kwargs
        ax = barplot(x=x, y=y, **kwargs)
        # 遍历每个条形
        for bar in ax.patches:
            # 断言条形的线宽等于参数字典中的 linewidth
            assert bar.get_linewidth() == kwargs["linewidth"]
            # 断言条形的填充颜色等于参数字典中的 facecolor
            assert bar.get_facecolor() == kwargs["facecolor"]
            # 断言条形是否光栅化等于参数字典中的 rasterized
            assert bar.get_rasterized() == kwargs["rasterized"]
    
    # 定义一个测试函数，测试图例属性设置
    def test_legend_attributes(self, long_df):
        # 获取调色板
        palette = color_palette()
        # 调用 barplot 函数绘制长格式数据的条形图，指定 x、y 和 hue 等参数，并设置边缘颜色和线宽度
        ax = barplot(
            long_df, x="a", y="y", hue="c", saturation=1, edgecolor="k", linewidth=3
        )
        # 遍历图例中的每个图例句柄
        for i, patch in enumerate(get_legend_handles(ax.get_legend())):
            # 断言图例句柄的填充颜色与调色板中对应位置的颜色相同
            assert same_color(patch.get_facecolor(), palette[i])
            # 断言图例句柄的边缘颜色为黑色
            assert same_color(patch.get_edgecolor(), "k")
            # 断言图例句柄的线宽度为3
            assert patch.get_linewidth() == 3
    
    # 定义一个测试函数，测试无填充的图例属性设置
    def test_legend_unfilled(self, long_df):
        # 获取调色板
        palette = color_palette()
        # 调用 barplot 函数绘制长格式数据的条形图，指定 x、y 和 hue 等参数，并设置不填充和线宽度
        ax = barplot(long_df, x="a", y="y", hue="c", fill=False, linewidth=3)
        # 遍历图例中的每个图例句柄
        for i, patch in enumerate(get_legend_handles(ax.get_legend())):
            # 断言图例句柄的填充颜色为透明色
            assert patch.get_facecolor() == (0, 0, 0, 0)
            # 断言图例句柄的边缘颜色与调色板中对应位置的颜色相同
            assert same_color(patch.get_edgecolor(), palette[i])
            # 断言图例句柄的线宽度为3
            assert patch.get_linewidth() == 3
    
    # 定义一个测试函数，测试误差线的参数设置
    @pytest.mark.parametrize("fill", [True, False])
    def test_err_kws(self, fill):
        # 定义两个列表，x 包含多个字符串，y 包含一组数值
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 定义一个包含误差线参数的字典
        err_kws = dict(color=(1, 1, .5, .5), linewidth=5)
        # 调用 barplot 函数绘制条形图，指定是否填充和误差线参数
        ax = barplot(x=x, y=y, fill=fill, err_kws=err_kws)
        # 遍历每条误差线
        for line in ax.lines:
            # 断言误差线的颜色等于误差线参数中的 color
            assert line.get_color() == err_kws["color"]
            # 断言误差线的线宽度等于误差线参数中的 linewidth
            assert line.get_linewidth() == err_kws["linewidth"]
    @pytest.mark.parametrize(
        "kwargs",
        [  # 参数化测试用例，kwargs 是一个字典，包含不同的参数组合
            dict(data="wide"),  # 使用 "wide" 数据的参数字典
            dict(data="wide", orient="h"),  # 使用 "wide" 数据和横向方向参数的字典
            dict(data="flat"),  # 使用 "flat" 数据的参数字典
            dict(data="long", x="a", y="y"),  # 使用 "long" 数据，并指定 x 和 y 的参数字典
            dict(data=None, x="a", y="y"),  # 使用 None 数据，但指定了 x 和 y 的参数字典
            dict(data="long", x="a", y="y", hue="a"),  # 使用 "long" 数据，并指定 x、y 和 hue 的参数字典
            dict(data=None, x="a", y="y", hue="a"),  # 使用 None 数据，但指定了 x、y 和 hue 的参数字典
            dict(data="long", x="a", y="y", hue="b"),  # 使用 "long" 数据，并指定 x、y 和 hue 的参数字典
            dict(data=None, x="s", y="y", hue="a"),  # 使用 None 数据，但指定了 x、y 和 hue 的参数字典
            dict(data="long", x="a", y="y", hue="s"),  # 使用 "long" 数据，并指定 x、y 和 hue 的参数字典
            dict(data="long", x="a", y="y", units="c"),  # 使用 "long" 数据，并指定 x、y 和 units 的参数字典
            dict(data="null", x="a", y="y", hue="a", gap=.1, fill=False),  # 使用 "null" 数据，并指定其他参数的字典
            dict(data="long", x="s", y="y", hue="a", native_scale=True),  # 使用 "long" 数据，并指定其他参数的字典
            dict(data="long", x="d", y="y", hue="a", native_scale=True),  # 使用 "long" 数据，并指定其他参数的字典
            dict(data="long", x="a", y="y", errorbar=("pi", 50)),  # 使用 "long" 数据，并指定 errorbar 参数的字典
            dict(data="long", x="a", y="y", errorbar=None),  # 使用 "long" 数据，但不指定 errorbar 参数的字典
            dict(data="long", x="a", y="y", capsize=.3, err_kws=dict(c="k")),  # 使用 "long" 数据，并指定 capsize 和 err_kws 参数的字典
            dict(data="long", x="a", y="y", color="blue", edgecolor="green", alpha=.5),  # 使用 "long" 数据，并指定颜色相关参数的字典
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):
        kwargs = kwargs.copy()  # 复制参数字典，以防修改原始数据
        kwargs["seed"] = 0  # 添加 seed 参数
        kwargs["n_boot"] = 10  # 添加 n_boot 参数

        if kwargs["data"] == "long":
            kwargs["data"] = long_df  # 如果参数中的 data 是 "long"，则使用 long_df 数据
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df  # 如果参数中的 data 是 "wide"，则使用 wide_df 数据
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series  # 如果参数中的 data 是 "flat"，则使用 flat_series 数据
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df  # 如果参数中的 data 是 "null"，则使用 null_df 数据
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]  # 如果 data 参数为 None，则使用 long_df 中对应的列作为数据

        ax = barplot(**kwargs)  # 调用 barplot 函数，传入参数 kwargs，返回 axes 对象 ax
        g = catplot(**kwargs, kind="bar")  # 调用 catplot 函数，传入参数 kwargs 和 kind="bar"，返回图形对象 g

        assert_plots_equal(ax, g.ax)  # 断言 ax 和 g.ax 是相等的

    def test_errwidth_deprecation(self):
        x, y = ["a", "b", "c"], [1, 2, 3]  # 定义示例数据 x 和 y
        val = 5  # 错误宽度参数的值
        with pytest.warns(FutureWarning, match="\n\nThe `errwidth` parameter"):  # 使用 pytest 的 warns 方法，捕获 FutureWarning 类型的警告信息
            ax = barplot(x=x, y=y, errwidth=val)  # 调用 barplot 函数，传入参数 x, y 和 errwidth=val，返回 axes 对象 ax
        for line in ax.lines:
            assert line.get_linewidth() == val  # 断言绘图对象的线宽度等于 val

    def test_errcolor_deprecation(self):
        x, y = ["a", "b", "c"], [1, 2, 3]  # 定义示例数据 x 和 y
        val = (1, .7, .4, .8)  # 错误颜色参数的值
        with pytest.warns(FutureWarning, match="\n\nThe `errcolor` parameter"):  # 使用 pytest 的 warns 方法，捕获 FutureWarning 类型的警告信息
            ax = barplot(x=x, y=y, errcolor=val)  # 调用 barplot 函数，传入参数 x, y 和 errcolor=val，返回 axes 对象 ax
        for line in ax.lines:
            assert line.get_color() == val  # 断言绘图对象的颜色等于 val

    def test_capsize_as_none_deprecation(self):
        x, y = ["a", "b", "c"], [1, 2, 3]  # 定义示例数据 x 和 y
        with pytest.warns(FutureWarning, match="\n\nPassing `capsize=None`"):  # 使用 pytest 的 warns 方法，捕获 FutureWarning 类型的警告信息
            ax = barplot(x=x, y=y, capsize=None)  # 调用 barplot 函数，传入参数 x, y 和 capsize=None，返回 axes 对象 ax
        for line in ax.lines:
            assert len(line.get_xdata()) == 2  # 断言绘图对象的 x 数据长度为 2
    # 定义一个测试方法，测试调色板暗示的色调参数弃用情况
    def test_hue_implied_by_palette_deprecation(self):
        # 创建一个包含字符串的列表
        x = ["a", "b", "c"]
        # 创建一个包含整数的列表
        y = [1, 2, 3]
        # 指定调色板名称
        palette = "Set1"
        # 使用指定调色板生成与 x 长度相同的颜色列表
        colors = color_palette(palette, len(x))
        # 设置警告信息内容
        msg = "Passing `palette` without assigning `hue` is deprecated."
        # 检查是否产生未来警告，并匹配特定的警告信息内容
        with pytest.warns(FutureWarning, match=msg):
            # 创建条形图，并传入 x, y 数据，设定饱和度和调色板
            ax = barplot(x=x, y=y, saturation=1, palette=palette)
        # 遍历条形图中的每个条形，确保每个条形的颜色与预期的颜色相同
        for i, bar in enumerate(ax.patches):
            assert same_color(bar.get_facecolor(), colors[i])
class TestPointPlot(SharedAggTests):
    # TestPointPlot 类，继承自 SharedAggTests 类，用于测试 pointplot 函数的功能

    func = staticmethod(pointplot)
    # 使用静态方法 func 来引用 pointplot 函数

    def get_last_color(self, ax):
        # 获取图形对象 ax 中最后一条线的颜色
        color = ax.lines[-1].get_color()
        return to_rgba(color)
        # 将颜色转换为 RGBA 格式并返回

    @pytest.mark.parametrize("orient", ["x", "y"])
    def test_single_var(self, orient):
        # 测试单个变量的情况，参数 orient 可以是 "x" 或 "y"
        
        # 创建一个包含数值 [1, 3, 10] 的 Pandas Series
        vals = pd.Series([1, 3, 10])
        # 调用 pointplot 函数生成绘图对象 ax
        ax = pointplot(**{orient: vals})
        # 获取第一条线的引用
        line = ax.lines[0]
        # 断言第一条线的 get_{orient}data() 方法的返回值约等于 vals 的均值
        assert getattr(line, f"get_{orient}data")() == approx(vals.mean())

    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_wide_df(self, wide_df, orient):
        # 测试宽格式 DataFrame 的情况，参数 orient 可以是 "x", "y", "h", "v"
        
        # 调用 pointplot 函数生成绘图对象 ax，传入 wide_df 和 orient 参数
        ax = pointplot(wide_df, orient=orient)
        # 根据 orient 更新 orient 变量
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        # 根据 depend 字典更新 depend 变量
        depend = {"x": "y", "y": "x"}[orient]
        # 获取第一条线的引用
        line = ax.lines[0]
        # 断言第一条线的 get_{orient}data() 方法的返回值等于 wide_df 列数的范围数组
        assert_array_equal(
            getattr(line, f"get_{orient}data")(),
            np.arange(len(wide_df.columns)),
        )
        # 断言第一条线的 get_{depend}data() 方法的返回值约等于 wide_df 每列的均值
        assert_array_almost_equal(
            getattr(line, f"get_{depend}data")(),
            wide_df.mean(axis=0),
        )

    @pytest.mark.parametrize("orient", ["x", "y", "h", "v"])
    def test_vector_orient(self, orient):
        # 测试向量导向的情况，参数 orient 可以是 "x", "y", "h", "v"
        
        # 创建一个包含键为 ["a", "b", "c"]，值为 [1, 2, 3] 的字典 data
        keys, vals = ["a", "b", "c"], [1, 2, 3]
        data = dict(zip(keys, vals))
        # 根据 orient 更新 orient 变量
        orient = {"h": "y", "v": "x"}.get(orient, orient)
        # 根据 depend 字典更新 depend 变量
        depend = {"x": "y", "y": "x"}[orient]
        # 调用 pointplot 函数生成绘图对象 ax，传入 data 和 orient 参数
        ax = pointplot(data, orient=orient)
        # 获取第一条线的引用
        line = ax.lines[0]
        # 断言第一条线的 get_{orient}data() 方法的返回值等于 keys 列表的范围数组
        assert_array_equal(
            getattr(line, f"get_{orient}data")(),
            np.arange(len(keys)),
        )
        # 断言第一条线的 get_{depend}data() 方法的返回值等于 vals 列表
        assert_array_equal(getattr(line, f"get_{depend}data")(), vals)

    def test_xy_vertical(self):
        # 测试纵向的 x, y 值的情况
        
        # 定义 x 和 y 列表
        x, y = ["a", "b", "c"], [1, 3, 2.5]
        # 调用 pointplot 函数生成绘图对象 ax，传入 x 和 y 参数
        ax = pointplot(x=x, y=y)
        # 遍历第一条线的所有坐标数据，断言每个坐标数据与预期的值相等
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (i, y[i])

    def test_xy_horizontal(self):
        # 测试横向的 x, y 值的情况
        
        # 定义 x 和 y 列表
        x, y = [1, 3, 2.5], ["a", "b", "c"]
        # 调用 pointplot 函数生成绘图对象 ax，传入 x 和 y 参数
        ax = pointplot(x=x, y=y)
        # 遍历第一条线的所有坐标数据，断言每个坐标数据与预期的值相等
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (x[i], i)

    def test_xy_with_na_grouper(self):
        # 测试带有 NA 分组器的 x, y 值的情况
        
        # 定义 x 和 y 列表
        x, y = ["a", None, "b"], [1, 2, 3]
        # 调用 pointplot 函数生成绘图对象 ax，传入 x 和 y 参数
        ax = pointplot(x=x, y=y)
        # 绘制 ax.figure 对象（适用于 matplotlib < 3.5）
        _draw_figure(ax.figure)
        # 断言 x 轴的刻度值为 [0, 1]
        assert ax.get_xticks() == [0, 1]
        # 断言 x 轴的刻度标签文本为 ["a", "b"]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
        # 断言第一条线的 x 数据为 [0, 1]
        assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
        # 断言第一条线的 y 数据为 [1, 3]
        assert_array_equal(ax.lines[0].get_ydata(), [1, 3])

    def test_xy_with_na_value(self):
        # 测试带有 NA 值的 x, y 值的情况
        
        # 定义 x 和 y 列表
        x, y = ["a", "b", "c"], [1, np.nan, 3]
        # 调用 pointplot 函数生成绘图对象 ax，传入 x 和 y 参数
        ax = pointplot(x=x, y=y)
        # 绘制 ax.figure 对象（适用于 matplotlib < 3.5）
        _draw_figure(ax.figure)
        # 断言 x 轴的刻度值为 [0, 1, 2]
        assert ax.get_xticks() == [0, 1, 2]
        # 断言 x 轴的刻度标签文本为 x 列表的值
        assert [t.get_text() for t in ax.get_xticklabels()] == x
        # 断言第一条线的 x 数据为 [0, 1, 2]
        assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
        # 断言第一条线的 y 数据为 y 列表的值
        assert_array_equal(ax.lines[0].get_ydata(), y)
    # 定义测试函数 test_hue，用于测试点图（pointplot）功能
    def test_hue(self):

        # 设置测试数据
        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        # 设置颜色分类变量
        hue = ["x", "y", "x", "y"]
        # 调用 pointplot 函数生成图形并返回轴对象 ax
        ax = pointplot(x=x, y=y, hue=hue, errorbar=None)
        # 遍历前两条线
        for i, line in enumerate(ax.lines[:2]):
            # 断言线的 Y 数据与预期相等
            assert_array_equal(line.get_ydata(), y[i::2])
            # 断言线的颜色与预期相同
            assert same_color(line.get_color(), f"C{i}")

    # 定义测试函数 test_wide_data_is_joined，用于测试宽格式数据联接功能
    def test_wide_data_is_joined(self, wide_df):

        # 调用 pointplot 函数生成图形并返回轴对象 ax
        ax = pointplot(wide_df, errorbar=None)
        # 断言轴对象的线条数量为1
        assert len(ax.lines) == 1

    # 定义测试函数 test_xy_native_scale，测试原生比例的 XY 图
    def test_xy_native_scale(self):

        # 设置测试数据
        x, y = [2, 4, 8], [1, 2, 3]
        # 调用 pointplot 函数生成图形并返回轴对象 ax
        ax = pointplot(x=x, y=y, native_scale=True)
        # 获取第一条线
        line = ax.lines[0]
        # 断言线的 X 数据与预期相等
        assert_array_equal(line.get_xdata(), x)
        # 断言线的 Y 数据与预期相等
        assert_array_equal(line.get_ydata(), y)

    # 使用 lambda 包装 np.mean 以避免不必要的 pandas 弃用警告
    @pytest.mark.parametrize("estimator", ["mean", lambda x: np.mean(x)])
    # 定义测试函数 test_estimate，用于测试估计功能
    def test_estimate(self, long_df, estimator):

        # 设置聚合变量和值变量
        agg_var, val_var = "a", "y"
        # 对长格式数据按聚合变量分组并对值变量进行聚合
        agg_df = long_df.groupby(agg_var)[val_var].agg(estimator)

        # 调用 pointplot 函数生成图形并返回轴对象 ax
        ax = pointplot(long_df, x=agg_var, y=val_var, errorbar=None)
        # 获取分类顺序
        order = categorical_order(long_df[agg_var])
        # 遍历第一条线的 XY 数据
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            # 断言 XY 数据元组与预期接近
            assert tuple(xy) == approx((i, agg_df[order[i]]))

    # 定义测试函数 test_weighted_estimate，用于测试加权估计功能
    def test_weighted_estimate(self, long_df):

        # 调用 pointplot 函数生成图形并返回轴对象 ax，使用权重列 "x"
        ax = pointplot(long_df, y="y", weights="x")
        # 获取第一条线的 Y 数据并转为标量
        val = ax.lines[0].get_ydata().item()
        # 计算长格式数据 "y" 列的加权平均值
        expected = np.average(long_df["y"], weights=long_df["x"])
        # 断言获取的值与预期值相等
        assert val == expected

    # 定义测试函数 test_estimate_log_transform，测试估计对数变换功能
    def test_estimate_log_transform(self, long_df):

        # 创建 Matplotlib 图形对象的子图 ax
        ax = mpl.figure.Figure().subplots()
        # 设置 X 轴比例为对数尺度
        ax.set_xscale("log")
        # 调用 pointplot 函数生成图形并绘制在 ax 上，使用长格式数据 "z" 列作为 X 数据
        pointplot(x=long_df["z"], ax=ax)
        # 获取第一条线的 X 数据
        val, = ax.lines[0].get_xdata()
        # 断言获取的值与预期值相等
        assert val == 10 ** np.log10(long_df["z"]).mean()

    # 定义测试函数 test_errorbars，测试误差条功能
    def test_errorbars(self, long_df):

        # 设置聚合变量和值变量
        agg_var, val_var = "a", "y"
        # 对长格式数据按聚合变量分组并对值变量进行均值和标准差聚合
        agg_df = long_df.groupby(agg_var)[val_var].agg(["mean", "std"])

        # 调用 pointplot 函数生成图形并返回轴对象 ax，同时显示标准差误差条
        ax = pointplot(long_df, x=agg_var, y=val_var, errorbar="sd")
        # 获取分类顺序
        order = categorical_order(long_df[agg_var])
        # 遍历从第二条线开始的所有线条
        for i, line in enumerate(ax.lines[1:]):
            # 获取对应行的均值和标准差数据
            row = agg_df.loc[order[i]]
            # 获取线条的 Y 数据下限和上限
            lo, hi = line.get_ydata()
            # 断言下限值接近于均值减去标准差
            assert lo == approx(row["mean"] - row["std"])
            # 断言上限值接近于均值加上标准差
            assert hi == approx(row["mean"] + row["std"])

    # 定义测试函数 test_marker_linestyle，测试标记和线型功能
    def test_marker_linestyle(self):

        # 设置测试数据
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 调用 pointplot 函数生成图形并返回轴对象 ax，设置标记为 "s"，线型为 "--"
        ax = pointplot(x=x, y=y, marker="s", linestyle="--")
        # 获取第一条线
        line = ax.lines[0]
        # 断言线的标记类型为 "s"
        assert line.get_marker() == "s"
        # 断言线的线型为 "--"
        assert line.get_linestyle() == "--"

    # 定义测试函数 test_markers_linestyles_single，测试单一标记和线型功能
    def test_markers_linestyles_single(self):

        # 设置测试数据
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 调用 pointplot 函数生成图形并返回轴对象 ax，设置标记为 "s"，线型为 "--"
        ax = pointplot(x=x, y=y, markers="s", linestyles="--")
        # 获取第一条线
        line = ax.lines[0]
        # 断言线的标记类型为 "s"
        assert line.get_marker() == "s"
        # 断言线的线型为 "--"
        assert line.get_linestyle() == "--"
    # 测试函数：test_markers_linestyles_mapped
    def test_markers_linestyles_mapped(self):
        # 定义测试数据
        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        hue = ["x", "y", "x", "y"]
        markers = ["d", "s"]
        linestyles = ["--", ":"]
        # 调用 pointplot 函数绘制图形，并返回 AxesSubplot 对象
        ax = pointplot(
            x=x, y=y, hue=hue,
            markers=markers, linestyles=linestyles,
            errorbar=None,
        )
        # 验证前两条线的标记和线型是否符合预期
        for i, line in enumerate(ax.lines[:2]):
            assert line.get_marker() == markers[i]
            assert line.get_linestyle() == linestyles[i]

    # 测试函数：test_dodge_boolean
    def test_dodge_boolean(self):
        # 定义测试数据
        x, y = ["a", "b", "a", "b"], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        # 调用 pointplot 函数绘制图形，并返回 AxesSubplot 对象
        ax = pointplot(x=x, y=y, hue=hue, dodge=True, errorbar=None)
        # 验证第一条线的位置是否符合 dodge=True 的预期位置
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (i - .025, y[i])
        # 验证第二条线的位置是否符合 dodge=True 的预期位置
        for i, xy in enumerate(ax.lines[1].get_xydata()):
            assert tuple(xy) == (i + .025, y[2 + i])

    # 测试函数：test_dodge_float
    def test_dodge_float(self):
        # 定义测试数据
        x, y = ["a", "b", "a", "b"], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        # 调用 pointplot 函数绘制图形，并返回 AxesSubplot 对象
        ax = pointplot(x=x, y=y, hue=hue, dodge=.2, errorbar=None)
        # 验证第一条线的位置是否符合 dodge=0.2 的预期位置
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == (i - .1, y[i])
        # 验证第二条线的位置是否符合 dodge=0.2 的预期位置
        for i, xy in enumerate(ax.lines[1].get_xydata()):
            assert tuple(xy) == (i + .1, y[2 + i])

    # 测试函数：test_dodge_log_scale
    def test_dodge_log_scale(self):
        # 定义测试数据
        x, y = [10, 1000, 10, 1000], [1, 2, 3, 4]
        hue = ["x", "x", "y", "y"]
        # 创建一个带对数坐标轴的 Figure 对象，并获取其 AxesSubplot 对象
        ax = mpl.figure.Figure().subplots()
        # 设置 x 轴为对数坐标轴
        ax.set_xscale("log")
        # 调用 pointplot 函数绘制图形，并返回 AxesSubplot 对象
        pointplot(x=x, y=y, hue=hue, dodge=.2, native_scale=True, errorbar=None, ax=ax)
        # 验证第一条线的位置是否符合 dodge=0.2 在对数尺度下的预期位置
        for i, xy in enumerate(ax.lines[0].get_xydata()):
            assert tuple(xy) == approx((10 ** (np.log10(x[i]) - .2), y[i]))
        # 验证第二条线的位置是否符合 dodge=0.2 在对数尺度下的预期位置
        for i, xy in enumerate(ax.lines[1].get_xydata()):
            assert tuple(xy) == approx((10 ** (np.log10(x[2 + i]) + .2), y[2 + i]))

    # 测试函数：test_err_kws
    def test_err_kws(self):
        # 定义测试数据
        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        err_kws = dict(color=(.2, .5, .3), linewidth=10)
        # 调用 pointplot 函数绘制图形，并返回 AxesSubplot 对象
        ax = pointplot(x=x, y=y, errorbar=("pi", 100), err_kws=err_kws)
        # 验证除第一条线外的所有线的颜色和线宽是否符合预期
        for line in ax.lines[1:]:
            assert same_color(line.get_color(), err_kws["color"])
            assert line.get_linewidth() == err_kws["linewidth"]

    # 测试函数：test_err_kws_inherited
    def test_err_kws_inherited(self):
        # 定义测试数据
        x, y = ["a", "a", "b", "b"], [1, 2, 3, 4]
        kws = dict(color=(.2, .5, .3), linewidth=10)
        # 调用 pointplot 函数绘制图形，并返回 AxesSubplot 对象
        ax = pointplot(x=x, y=y, errorbar=("pi", 100), **kws)
        # 验证除第一条线外的所有线的颜色和线宽是否符合预期
        for line in ax.lines[1:]:
            assert same_color(line.get_color(), kws["color"])
            assert line.get_linewidth() == kws["linewidth"]

    # 标记为跳过的测试函数，条件是 matplotlib 版本小于 3.6
    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="Legend handle missing marker property"
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a", y="y"),
            dict(data=None, x="a", y="y"),
            dict(data="long", x="a", y="y", hue="a"),
            dict(data=None, x="a", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="b"),
            dict(data=None, x="s", y="y", hue="a"),
            dict(data="long", x="a", y="y", hue="s"),
            dict(data="long", x="a", y="y", units="c"),
            dict(data="null", x="a", y="y", hue="a"),
            dict(data="long", x="s", y="y", hue="a", native_scale=True),
            dict(data="long", x="d", y="y", hue="a", native_scale=True),
            dict(data="long", x="a", y="y", errorbar=("pi", 50)),
            dict(data="long", x="a", y="y", errorbar=None),
            dict(data="null", x="a", y="y", hue="a", dodge=True),
            dict(data="null", x="a", y="y", hue="a", dodge=.2),
            dict(data="long", x="a", y="y", capsize=.3, err_kws=dict(c="k")),
            dict(data="long", x="a", y="y", color="blue", marker="s"),
            dict(data="long", x="a", y="y", hue="a", markers=["s", "d", "p"]),
        ]
    )



        @pytest.mark.parametrize(
            "kwargs",
            [  # 参数化测试的参数字典列表
                dict(data="wide"),  # 测试数据为“wide”格式
                dict(data="wide", orient="h"),  # 测试数据为“wide”格式，且方向为水平
                dict(data="flat"),  # 测试数据为“flat”格式
                dict(data="long", x="a", y="y"),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"
                dict(data=None, x="a", y="y"),  # 测试数据为None，横轴为"a"，纵轴为"y"
                dict(data="long", x="a", y="y", hue="a"),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，色调为"a"
                dict(data=None, x="a", y="y", hue="a"),  # 测试数据为None，横轴为"a"，纵轴为"y"，色调为"a"
                dict(data="long", x="a", y="y", hue="b"),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，色调为"b"
                dict(data=None, x="s", y="y", hue="a"),  # 测试数据为None，横轴为"s"，纵轴为"y"，色调为"a"
                dict(data="long", x="a", y="y", hue="s"),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，色调为"s"
                dict(data="long", x="a", y="y", units="c"),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，单位为"c"
                dict(data="null", x="a", y="y", hue="a"),  # 测试数据为“null”格式，横轴为"a"，纵轴为"y"，色调为"a"
                dict(data="long", x="s", y="y", hue="a", native_scale=True),  # 测试数据为“long”格式，横轴为"s"，纵轴为"y"，色调为"a"，使用原生比例
                dict(data="long", x="d", y="y", hue="a", native_scale=True),  # 测试数据为“long”格式，横轴为"d"，纵轴为"y"，色调为"a"，使用原生比例
                dict(data="long", x="a", y="y", errorbar=("pi", 50)),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，误差条为("pi", 50)
                dict(data="long", x="a", y="y", errorbar=None),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，无误差条
                dict(data="null", x="a", y="y", hue="a", dodge=True),  # 测试数据为“null”格式，横轴为"a"，纵轴为"y"，色调为"a"，使用躲避
                dict(data="null", x="a", y="y", hue="a", dodge=.2),  # 测试数据为“null”格式，横轴为"a"，纵轴为"y"，色调为"a"，躲避距离为0.2
                dict(data="long", x="a", y="y", capsize=.3, err_kws=dict(c="k")),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，误差帽大小为0.3，误差参数为{"c": "k"}
                dict(data="long", x="a", y="y", color="blue", marker="s"),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，颜色为"blue"，标记为"s"
                dict(data="long", x="a", y="y", hue="a", markers=["s", "d", "p"]),  # 测试数据为“long”格式，横轴为"a"，纵轴为"y"，色调为"a"，标记列表为["s", "d", "p"]
            ]
        )
    # 测试函数：验证 pointplot 和 catplot 的输出是否相等
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):
        # 复制参数字典，设置种子和 bootstrap 迭代次数
        kwargs = kwargs.copy()
        kwargs["seed"] = 0
        kwargs["n_boot"] = 10

        # 根据数据类型设置 kwargs["data"] 的值
        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        elif kwargs["data"] is None:
            # 如果 data 为 None，则根据 "x", "y", "hue" 参数从 long_df 中选择数据
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        # 使用给定参数绘制 pointplot 并获取轴对象 ax
        ax = pointplot(**kwargs)
        # 使用给定参数绘制 kind="point" 的 catplot 并获取轴对象 g
        g = catplot(**kwargs, kind="point")

        # 断言两个绘图的轴对象 ax 和 g.ax 是否相等
        assert_plots_equal(ax, g.ax)

    # 测试函数：验证禁用图例功能
    def test_legend_disabled(self, long_df):
        # 绘制 pointplot，禁用图例
        ax = pointplot(long_df, x="x", y="y", hue="b", legend=False)
        # 断言轴对象 ax 中是否不存在图例
        assert ax.get_legend() is None

    # 测试函数：验证警告消息关于 `join` 参数的提示
    def test_join_deprecation(self):
        # 绘制 pointplot，设置 join=False，并期望触发 UserWarning
        with pytest.warns(UserWarning, match="The `join` parameter"):
            ax = pointplot(x=["a", "b", "c"], y=[1, 2, 3], join=False)
        # 断言第一条线的线型为 "none"
        assert ax.lines[0].get_linestyle().lower() == "none"

    # 测试函数：验证警告消息关于 `scale` 参数的提示，并比较线条宽度和标记大小
    def test_scale_deprecation(self):
        x, y = ["a", "b", "c"], [1, 2, 3]
        # 绘制 pointplot，禁用 errorbar
        ax = pointplot(x=x, y=y, errorbar=None)
        # 使用 scale=2 绘制 pointplot，并期望触发 UserWarning
        with pytest.warns(UserWarning, match="The `scale` parameter"):
            pointplot(x=x, y=y, errorbar=None, scale=2)
        # 获取前两条线的对象，并比较线条宽度和标记大小
        l1, l2 = ax.lines
        assert l2.get_linewidth() == 2 * l1.get_linewidth()
        assert l2.get_markersize() > l1.get_markersize()

    # 测试函数：验证层叠绘图的剪裁效果
    def test_layered_plot_clipping(self):
        # 绘制具有单个数据点 'a' 的 pointplot
        x, y = ['a'], [4]
        pointplot(x=x, y=y)
        # 绘制具有单个数据点 'b' 的 pointplot，并获取轴对象 ax
        x, y = ['b'], [5]
        ax = pointplot(x=x, y=y)
        # 获取 y 轴的视图范围，断言第一个点的位置低于 4，第二个点的位置高于 5
        y_range = ax.viewLim.intervaly
        assert y_range[0] < 4 and y_range[1] > 5
class TestCountPlot:

    # 测试空数据集的情况
    def test_empty(self):

        # 调用 countplot() 创建一个图形对象 ax
        ax = countplot()
        # 断言 ax.patches 应该为空
        assert not ax.patches

        # 再次调用 countplot()，这次传入空列表作为 x 参数
        ax = countplot(x=[])
        # 断言 ax.patches 应该为空
        assert not ax.patches

    # 测试长格式数据集的情况
    def test_labels_long(self, long_df):

        # 创建一个包含两个子图的 Figure 对象 fig
        fig = mpl.figure.Figure()
        axs = fig.subplots(2)
        # 在第一个子图 axs[0] 上绘制 long_df 数据集中 x="a" 的 countplot
        countplot(long_df, x="a", ax=axs[0])
        # 在第二个子图 axs[1] 上绘制 long_df 数据集中 x="b" 的 stat="percent" 的 countplot
        countplot(long_df, x="b", stat="percent", ax=axs[1])

        # 在老版本的 matplotlib 上，需要调用 _draw_figure(fig) 来填充文本信息
        _draw_figure(fig)

        # 断言两个子图的 x 标签和 y 标签符合预期
        assert axs[0].get_xlabel() == "a"
        assert axs[1].get_xlabel() == "b"
        assert axs[0].get_ylabel() == "count"
        assert axs[1].get_ylabel() == "percent"

    # 测试宽格式数据集的情况
    def test_wide_data(self, wide_df):

        # 创建 wide_df 数据集的 countplot，并将结果保存在 ax 变量中
        ax = countplot(wide_df)
        # 断言 ax.patches 的数量应该等于 wide_df 数据集的列数
        assert len(ax.patches) == len(wide_df.columns)
        # 遍历每个条形图（bar）对象，进行一系列断言来验证它们的位置和大小等属性
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == len(wide_df)
            assert bar.get_width() == approx(0.8)

    # 测试扁平化序列数据的情况
    def test_flat_series(self):

        # 创建一个包含值 "a", "b", "c" 和对应计数 [2, 1, 4] 的序列 vals
        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])
        # 创建 vals 的 countplot，并将结果保存在 ax 变量中
        ax = countplot(vals)
        # 遍历每个条形图对象，进行断言以验证它们的位置和大小等属性
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == 0
            assert bar.get_y() + bar.get_height() / 2 == approx(i)
            assert bar.get_height() == approx(0.8)
            assert bar.get_width() == counts[i]

    # 测试 x 序列数据的情况
    def test_x_series(self):

        # 创建一个包含值 "a", "b", "c" 和对应计数 [2, 1, 4] 的序列 vals
        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])
        # 创建 x 序列数据的 countplot，并将结果保存在 ax 变量中
        ax = countplot(x=vals)
        # 遍历每个条形图对象，进行断言以验证它们的位置和大小等属性
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == counts[i]
            assert bar.get_width() == approx(0.8)

    # 测试 y 序列数据的情况
    def test_y_series(self):

        # 创建一个包含值 "a", "b", "c" 和对应计数 [2, 1, 4] 的序列 vals
        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])
        # 创建 y 序列数据的 countplot，并将结果保存在 ax 变量中
        ax = countplot(y=vals)
        # 遍历每个条形图对象，进行断言以验证它们的位置和大小等属性
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() == 0
            assert bar.get_y() + bar.get_height() / 2 == approx(i)
            assert bar.get_height() == approx(0.8)
            assert bar.get_width() == counts[i]

    # 测试使用冗余的 hue 参数的情况
    def test_hue_redundant(self):

        # 创建一个包含值 "a", "b", "c" 和对应计数 [2, 1, 4] 的序列 vals
        vals = ["a", "b", "c"]
        counts = [2, 1, 4]
        vals = pd.Series([x for x, n in zip(vals, counts) for _ in range(n)])

        # 创建 x 序列数据的 countplot，并指定 hue=vals 和 saturation=1
        ax = countplot(x=vals, hue=vals, saturation=1)
        # 遍历每个条形图对象，进行断言以验证它们的位置、大小和颜色等属性
        for i, bar in enumerate(ax.patches):
            assert bar.get_x() + bar.get_width() / 2 == approx(i)
            assert bar.get_y() == 0
            assert bar.get_height() == counts[i]
            assert bar.get_width() == approx(0.8)
            assert same_color(bar.get_facecolor(), f"C{i}")
    def test_hue_dodged(self):
        # 定义测试函数，测试带有分组的计数图的正确性

        vals = ["a", "a", "a", "b", "b", "b"]
        # 定义数据值列表

        hue = ["x", "y", "y", "x", "x", "x"]
        # 定义分组标签列表

        counts = [1, 3, 2, 0]
        # 定义预期的计数值列表

        ax = countplot(x=vals, hue=hue, saturation=1, legend=False)
        # 创建计数图对象，根据值和分组标签，饱和度为1，不显示图例

        for i, bar in enumerate(ax.patches):
            # 遍历计数图中的每一个条形块

            sign = 1 if i // 2 else -1
            # 根据索引值确定符号

            assert (
                bar.get_x() + bar.get_width() / 2
                == approx(i % 2 + sign * 0.8 / 4)
            )
            # 断言条形块的中心 x 坐标与预期相符

            assert bar.get_y() == 0
            # 断言条形块的 y 坐标为0

            assert bar.get_height() == counts[i]
            # 断言条形块的高度与预期的计数值相符

            assert bar.get_width() == approx(0.8 / 2)
            # 断言条形块的宽度与预期相符

            assert same_color(bar.get_facecolor(), f"C{i // 2}")
            # 断言条形块的颜色与预期相符

    @pytest.mark.parametrize("stat", ["percent", "probability", "proportion"])
    def test_stat(self, long_df, stat):
        # 参数化测试函数，测试不同统计方法的正确性

        col = "a"
        # 设置列名

        order = categorical_order(long_df[col])
        # 获取分类顺序

        expected = long_df[col].value_counts(normalize=True)
        # 获取预期值，按照比例归一化计数

        if stat == "percent":
            expected *= 100
        # 如果统计方法为百分比，将预期值乘以100

        ax = countplot(long_df, x=col, stat=stat)
        # 创建计数图对象，根据数据框和列名，指定统计方法

        for i, bar in enumerate(ax.patches):
            # 遍历计数图中的每一个条形块

            assert bar.get_height() == approx(expected[order[i]])
            # 断言条形块的高度与预期值相符

    def test_xy_error(self, long_df):
        # 测试函数，测试传递 x 和 y 值时的错误处理

        with pytest.raises(TypeError, match="Cannot pass values for both"):
            countplot(long_df, x="a", y="b")
        # 断言调用计数图函数同时传递 x 和 y 值会引发 TypeError 异常，并且异常信息包含特定字符串

    def test_legend_numeric_auto(self, long_df):
        # 测试函数，测试自动显示数字型图例的功能

        ax = countplot(long_df, x="x", hue="x")
        # 创建计数图对象，根据数据框和列名，指定 x 和 hue 参数

        assert len(ax.get_legend().texts) <= 6
        # 断言图例文本数量不超过6个

    def test_legend_disabled(self, long_df):
        # 测试函数，测试禁用图例的功能

        ax = countplot(long_df, x="x", hue="b", legend=False)
        # 创建计数图对象，根据数据框和列名，指定 x 和 hue 参数，并禁用图例显示

        assert ax.get_legend() is None
        # 断言计数图对象中的图例为空

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(data="wide"),
            dict(data="wide", orient="h"),
            dict(data="flat"),
            dict(data="long", x="a"),
            dict(data=None, x="a"),
            dict(data="long", y="b"),
            dict(data="long", x="a", hue="a"),
            dict(data=None, x="a", hue="a"),
            dict(data="long", x="a", hue="b"),
            dict(data=None, x="s", hue="a"),
            dict(data="long", x="a", hue="s"),
            dict(data="null", x="a", hue="a"),
            dict(data="long", x="s", hue="a", native_scale=True),
            dict(data="long", x="d", hue="a", native_scale=True),
            dict(data="long", x="a", stat="percent"),
            dict(data="long", x="a", hue="b", stat="proportion"),
            dict(data="long", x="a", color="blue", ec="green", alpha=.5),
        ]
    )
    def test_vs_catplot(self, long_df, wide_df, null_df, flat_series, kwargs):
        # 复制 kwargs，以防修改原始参数
        kwargs = kwargs.copy()

        # 如果 kwargs 中的 data 参数为 "long"，则使用 long_df 数据帧
        if kwargs["data"] == "long":
            kwargs["data"] = long_df
        # 如果 kwargs 中的 data 参数为 "wide"，则使用 wide_df 数据帧
        elif kwargs["data"] == "wide":
            kwargs["data"] = wide_df
        # 如果 kwargs 中的 data 参数为 "flat"，则使用 flat_series 序列
        elif kwargs["data"] == "flat":
            kwargs["data"] = flat_series
        # 如果 kwargs 中的 data 参数为 "null"，则使用 null_df 数据帧
        elif kwargs["data"] == "null":
            kwargs["data"] = null_df
        # 如果 kwargs 中的 data 参数为 None，则根据 x、y、hue 参数从 long_df 中选择数据
        elif kwargs["data"] is None:
            for var in ["x", "y", "hue"]:
                if var in kwargs:
                    kwargs[var] = long_df[kwargs[var]]

        # 通过 countplot 函数绘制计数图，使用参数 kwargs
        ax = countplot(**kwargs)
        
        # 使用 catplot 函数绘制分类图，类型为计数图，使用参数 kwargs
        g = catplot(**kwargs, kind="count")

        # 断言两个绘图对象的图形是否相等
        assert_plots_equal(ax, g.ax)
class CategoricalFixture:
    """Test boxplot (also base class for things like violinplots)."""
    # 创建一个固定的随机状态对象，种子为30
    rs = np.random.RandomState(30)
    # 总共的样本数
    n_total = 60
    # 生成一个形状为 (20, 3) 的随机正态分布数组
    x = rs.randn(int(n_total / 3), 3)
    # 将随机数组转换为 DataFrame，列名为 ['X', 'Y', 'Z']
    x_df = pd.DataFrame(x, columns=pd.Series(list("XYZ"), name="big"))
    # 生成一个随机的 Series，列名为 'y_data'
    y = pd.Series(rs.randn(n_total), name="y_data")
    # 对 y 进行随机置换
    y_perm = y.reindex(rs.choice(y.index, y.size, replace=False))
    # 生成一个固定的 Series，内容为 ['a', 'b', 'c'] 重复的随机序列
    g = pd.Series(np.repeat(list("abc"), int(n_total / 3)), name="small")
    # 生成一个固定的 Series，内容为 ['m', 'n'] 循环重复的随机序列
    h = pd.Series(np.tile(list("mn"), int(n_total / 2)), name="medium")
    # 生成一个固定的 Series，内容为 ['j', 'k', 'h'] 循环重复的随机序列
    u = pd.Series(np.tile(list("jkh"), int(n_total / 3)))
    # 生成一个包含以上 Series 的 DataFrame，列名分别为 'y', 'g', 'h', 'u'
    df = pd.DataFrame(dict(y=y, g=g, h=h, u=u))
    # 在 x_df DataFrame 中新增一列 'W'，内容为 g 列的数据

    def get_box_artists(self, ax):
        # 检查 Matplotlib 版本是否早于 3.5.0b0，如果是，则返回 ax.artists
        if _version_predates(mpl, "3.5.0b0"):
            return ax.artists
        else:
            # 否则，返回不带标签的 ax.patches（用于图例的排除）
            return [p for p in ax.patches if not p.get_label()]


class TestCatPlot(CategoricalFixture):

    def test_facet_organization(self):
        # 创建一个 catplot 对象 g，按列 'g' 绘制 'y' 的数据分布
        g = cat.catplot(x="g", y="y", data=self.df)
        # 断言 g 的轴形状为 (1, 1)
        assert g.axes.shape == (1, 1)

        # 创建一个 catplot 对象 g，按列 'g' 绘制 'y' 的数据分布，按列 'h' 分面绘制
        g = cat.catplot(x="g", y="y", col="h", data=self.df)
        # 断言 g 的轴形状为 (1, 2)
        assert g.axes.shape == (1, 2)

        # 创建一个 catplot 对象 g，按列 'g' 绘制 'y' 的数据分布，按行 'h' 分面绘制
        g = cat.catplot(x="g", y="y", row="h", data=self.df)
        # 断言 g 的轴形状为 (2, 1)
        assert g.axes.shape == (2, 1)

        # 创建一个 catplot 对象 g，按列 'g' 绘制 'y' 的数据分布，按行 'h'，按列 'u' 分面绘制
        g = cat.catplot(x="g", y="y", col="u", row="h", data=self.df)
        # 断言 g 的轴形状为 (2, 3)
        assert g.axes.shape == (2, 3)
    # 定义一个测试方法，用于测试绘图元素的相关功能
    def test_plot_elements(self):

        # 生成一个分类图，以点的形式展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", data=self.df, kind="point")
        # 期望的线条数量为 g 列的唯一值数加一
        want_lines = 1 + self.g.unique().size
        # 断言生成的图形对象中线条的数量与期望值相等
        assert len(g.ax.lines) == want_lines

        # 生成一个带有色调分组的分类图，以点的形式展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="point")
        # 计算期望的线条数量
        want_lines = (
            len(self.g.unique()) * len(self.h.unique()) + 2 * len(self.h.unique())
        )
        # 断言生成的图形对象中线条的数量与期望值相等
        assert len(g.ax.lines) == want_lines

        # 生成一个简单柱状图，展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", data=self.df, kind="bar")
        # 计算期望的元素（柱子和线条）数量
        want_elements = self.g.unique().size
        # 断言生成的图形对象中柱子的数量与期望值相等
        assert len(g.ax.patches) == want_elements
        # 断言生成的图形对象中线条的数量与期望值相等
        assert len(g.ax.lines) == want_elements

        # 生成一个带有色调分组的柱状图，展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="bar")
        # 计算期望的元素（柱子和线条）数量
        want_elements = self.g.nunique() * self.h.nunique()
        # 断言生成的图形对象中柱子的数量与期望值相等，加上线条的数量
        assert len(g.ax.patches) == (want_elements + self.h.nunique())
        # 断言生成的图形对象中线条的数量与期望值相等
        assert len(g.ax.lines) == want_elements

        # 生成一个计数图，仅计数 x 列的数据，并获取生成的图形对象
        g = cat.catplot(x="g", data=self.df, kind="count")
        # 计算期望的元素（柱子）数量
        want_elements = self.g.unique().size
        # 断言生成的图形对象中柱子的数量与期望值相等
        assert len(g.ax.patches) == want_elements
        # 断言生成的图形对象中线条的数量为零
        assert len(g.ax.lines) == 0

        # 生成一个带有色调分组的计数图，仅计数 x 和 y 列的数据，并获取生成的图形对象
        g = cat.catplot(x="g", hue="h", data=self.df, kind="count")
        # 计算期望的元素（柱子）数量
        want_elements = self.g.nunique() * self.h.nunique() + self.h.nunique()
        # 断言生成的图形对象中柱子的数量与期望值相等
        assert len(g.ax.patches) == want_elements
        # 断言生成的图形对象中线条的数量为零
        assert len(g.ax.lines) == 0

        # 生成一个箱线图，仅展示 y 列的数据，并获取生成的图形对象
        g = cat.catplot(y="y", data=self.df, kind="box")
        # 期望的艺术品（箱线图对象）数量为一
        want_artists = 1
        # 断言生成的图形对象中箱线图对象的数量与期望值相等
        assert len(self.get_box_artists(g.ax)) == want_artists

        # 生成一个箱线图，以 g 列作为 x 轴，y 列作为 y 轴展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", data=self.df, kind="box")
        # 计算期望的艺术品（箱线图对象）数量
        want_artists = self.g.unique().size
        # 断言生成的图形对象中箱线图对象的数量与期望值相等
        assert len(self.get_box_artists(g.ax)) == want_artists

        # 生成一个带有色调分组的箱线图，以 g 列作为 x 轴，y 列作为 y 轴展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="box")
        # 计算期望的艺术品（箱线图对象）数量
        want_artists = self.g.nunique() * self.h.nunique()
        # 断言生成的图形对象中箱线图对象的数量与期望值相等
        assert len(self.get_box_artists(g.ax)) == want_artists

        # 生成一个小提琴图，以 g 列作为 x 轴，y 列作为 y 轴展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", data=self.df, kind="violin", inner=None)
        # 计算期望的元素（小提琴图对象）数量
        want_elements = self.g.unique().size
        # 断言生成的图形对象中小提琴图对象的数量与期望值相等
        assert len(g.ax.collections) == want_elements

        # 生成一个带有色调分组的小提琴图，以 g 列作为 x 轴，y 列作为 y 轴展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="violin", inner=None)
        # 计算期望的元素（小提琴图对象）数量
        want_elements = self.g.nunique() * self.h.nunique()
        # 断言生成的图形对象中小提琴图对象的数量与期望值相等
        assert len(g.ax.collections) == want_elements

        # 生成一个散点图，以 g 列作为 x 轴，y 列作为 y 轴展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", data=self.df, kind="strip")
        # 计算期望的元素（散点对象）数量
        want_elements = self.g.unique().size
        # 断言生成的图形对象中散点对象的数量与期望值相等
        assert len(g.ax.collections) == want_elements
        # 断言生成的图形对象中每个散点对象的颜色均为 "C0"
        for strip in g.ax.collections:
            assert same_color(strip.get_facecolors(), "C0")

        # 生成一个带有色调分组的散点图，以 g 列作为 x 轴，y 列作为 y 轴展示数据，并获取生成的图形对象
        g = cat.catplot(x="g", y="y", hue="h", data=self.df, kind="strip")
        # 计算期望的元素（散点对象）数量
        want_elements = self.g.nunique()
        # 断言生成的图形对象中散点对象的数量与期望值相等
        assert len(g.ax.collections) == want_elements

    # 定义一个测试方法，测试在传入不支持的图形类型时是否能捕获到 ValueError 异常
    def test_bad_plot_kind_error(self):

        # 使用 pytest 的断言来确保调用 catplot 时传入不支持的图形类型会触发 ValueError 异常
        with pytest.raises(ValueError):
            cat.catplot(x="g", y="y", data=self.df, kind="not_a_kind")

    # 定义一个测试方法，测试在计数图中同时指定 x 和 y 列时是否能捕获
    def test_plot_colors(self):
        # 使用 seaborn 的 barplot 方法创建柱状图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df
        ax = cat.barplot(x="g", y="y", data=self.df)
        # 使用 seaborn 的 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，图类型为柱状图，数据来源为 self.df
        g = cat.catplot(x="g", y="y", data=self.df, kind="bar")
        # 对比两个图中的每一个矩形区域的颜色是否相同
        for p1, p2 in zip(ax.patches, g.ax.patches):
            assert p1.get_facecolor() == p2.get_facecolor()
        # 关闭所有 matplotlib 图形
        plt.close("all")

        # 使用 seaborn 的 barplot 方法创建柱状图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df，指定颜色为紫色
        ax = cat.barplot(x="g", y="y", data=self.df, color="purple")
        # 使用 seaborn 的 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，图类型为柱状图，数据来源为 self.df，指定颜色为紫色
        g = cat.catplot(x="g", y="y", data=self.df, kind="bar", color="purple")
        # 对比两个图中的每一个矩形区域的颜色是否相同
        for p1, p2 in zip(ax.patches, g.ax.patches):
            assert p1.get_facecolor() == p2.get_facecolor()
        # 关闭所有 matplotlib 图形
        plt.close("all")

        # 使用 seaborn 的 barplot 方法创建柱状图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df，使用 Set2 调色板，按 'h' 列进行分组
        ax = cat.barplot(x="g", y="y", data=self.df, palette="Set2", hue="h")
        # 使用 seaborn 的 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，图类型为柱状图，数据来源为 self.df，使用 Set2 调色板，按 'h' 列进行分组
        g = cat.catplot(x="g", y="y", data=self.df, kind="bar", palette="Set2", hue="h")
        # 对比两个图中的每一个矩形区域的颜色是否相同
        for p1, p2 in zip(ax.patches, g.ax.patches):
            assert p1.get_facecolor() == p2.get_facecolor()
        # 关闭所有 matplotlib 图形
        plt.close("all")

        # 使用 seaborn 的 pointplot 方法创建折线图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df
        ax = cat.pointplot(x="g", y="y", data=self.df)
        # 使用 seaborn 的 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df
        g = cat.catplot(x="g", y="y", data=self.df)
        # 对比两个图中的每一条折线的颜色是否相同
        for l1, l2 in zip(ax.lines, g.ax.lines):
            assert l1.get_color() == l2.get_color()
        # 关闭所有 matplotlib 图形
        plt.close("all")

        # 使用 seaborn 的 pointplot 方法创建折线图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df，指定颜色为紫色
        ax = cat.pointplot(x="g", y="y", data=self.df, color="purple")
        # 使用 seaborn 的 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，图类型为折线图，数据来源为 self.df，指定颜色为紫色
        g = cat.catplot(x="g", y="y", data=self.df, color="purple", kind="point")
        # 对比两个图中的每一条折线的颜色是否相同
        for l1, l2 in zip(ax.lines, g.ax.lines):
            assert l1.get_color() == l2.get_color()
        # 关闭所有 matplotlib 图形
        plt.close("all")

        # 使用 seaborn 的 pointplot 方法创建折线图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df，使用 Set2 调色板，按 'h' 列进行分组
        ax = cat.pointplot(x="g", y="y", data=self.df, palette="Set2", hue="h")
        # 使用 seaborn 的 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，图类型为折线图，数据来源为 self.df，使用 Set2 调色板，按 'h' 列进行分组
        g = cat.catplot(
            x="g", y="y", data=self.df, palette="Set2", hue="h", kind="point"
        )
        # 对比两个图中的每一条折线的颜色是否相同
        for l1, l2 in zip(ax.lines, g.ax.lines):
            assert l1.get_color() == l2.get_color()
        # 关闭所有 matplotlib 图形
        plt.close("all")

    def test_ax_kwarg_removal(self):
        # 创建一个新的图形和坐标轴对象，返回 f 和 ax
        f, ax = plt.subplots()
        # 使用 catplot 方法创建分类图，以 'g' 列为 x 轴，'y' 列为 y 轴，数据来源为 self.df，将 ax 参数设置为指定的 ax 对象
        with pytest.warns(UserWarning, match="catplot is a figure-level"):
            g = cat.catplot(x="g", y="y", data=self.df, ax=ax)
        # 断言 ax 上没有任何图形集合
        assert len(ax.collections) == 0
        # 断言 g.ax 上有图形集合
        assert len(g.ax.collections) > 0
    def test_share_xy(self):
        # 测试默认行为是否正常工作
        g = cat.catplot(x="g", y="y", col="g", data=self.df, sharex=True)
        # 遍历所有子图的坐标轴，断言集合的数量等于数据框中唯一的 g 值数量
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

        g = cat.catplot(x="y", y="g", col="g", data=self.df, sharey=True)
        # 遍历所有子图的坐标轴，断言集合的数量等于数据框中唯一的 g 值数量
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

        # 测试取消共享功能是否正常工作
        g = cat.catplot(
            x="g", y="y", col="g", data=self.df, sharex=False, kind="bar",
        )
        # 遍历所有子图的坐标轴，断言条形图的数量为1
        for ax in g.axes.flat:
            assert len(ax.patches) == 1

        g = cat.catplot(
            x="y", y="g", col="g", data=self.df, sharey=False, kind="bar",
        )
        # 遍历所有子图的坐标轴，断言条形图的数量为1
        for ax in g.axes.flat:
            assert len(ax.patches) == 1

        g = cat.catplot(
            x="g", y="y", col="g", data=self.df, sharex=False, color="b"
        )
        # 遍历所有子图的坐标轴，断言 x 轴的限制范围为 (-0.5, 0.5)
        for ax in g.axes.flat:
            assert ax.get_xlim() == (-0.5, 0.5)

        g = cat.catplot(
            x="y", y="g", col="g", data=self.df, sharey=False, color="r"
        )
        # 遍历所有子图的坐标轴，断言 y 轴的限制范围为 (0.5, -0.5)
        for ax in g.axes.flat:
            assert ax.get_ylim() == (0.5, -0.5)

        # 确保如果给定了顺序参数，无论 sharex 的值如何都会使用它
        order = self.df.g.unique()
        g = cat.catplot(x="g", y="y", col="g", data=self.df, sharex=False, order=order)
        # 遍历所有子图的坐标轴，断言集合的数量等于数据框中唯一的 g 值数量
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

        g = cat.catplot(x="y", y="g", col="g", data=self.df, sharey=False, order=order)
        # 遍历所有子图的坐标轴，断言集合的数量等于数据框中唯一的 g 值数量
        for ax in g.axes.flat:
            assert len(ax.collections) == len(self.df.g.unique())

    def test_facetgrid_data(self, long_df):
        g1 = catplot(data=long_df, x="a", y="y", col="c")
        # 断言 g1 的数据源是 long_df
        assert g1.data is long_df

        g2 = catplot(x=long_df["a"], y=long_df["y"], col=long_df["c"])
        # 断言 g2 的数据源与 long_df 的子集（["a", "y", "c"]）相等
        assert g2.data.equals(long_df[["a", "y", "c"]])

    @pytest.mark.parametrize("var", ["col", "row"])
    def test_array_faceter(self, long_df, var):
        g1 = catplot(data=long_df, x="y", **{var: "a"})
        g2 = catplot(data=long_df, x="y", **{var: long_df["a"].to_numpy()})
        # 逐一比较 g1 和 g2 的每个子图，确保它们相等
        for ax1, ax2 in zip(g1.axes.flat, g2.axes.flat):
            assert_plots_equal(ax1, ax2)

    def test_invalid_kind(self, long_df):
        # 使用 catplot 尝试使用无效的 kind 参数，断言是否引发 ValueError 异常
        with pytest.raises(ValueError, match="Invalid `kind`: 'wrong'"):
            catplot(long_df, kind="wrong")

    def test_legend_with_auto(self):
        g1 = catplot(self.df, x="g", y="y", hue="g", legend='auto')
        # 断言 g1 的图例对象为 None
        assert g1._legend is None

        g2 = catplot(self.df, x="g", y="y", hue="g", legend=True)
        # 断言 g2 的图例对象不为 None
        assert g2._legend is not None

    def test_weights_warning(self, long_df):
        # 使用 long_df 数据，尝试使用 weights 参数，断言是否发出 UserWarning 警告
        with pytest.warns(UserWarning, match="The `weights` parameter"):
            g = catplot(long_df, x="a", y="y", weights="z")
        # 断言 g 的坐标轴对象不为 None
        assert g.ax is not None
class TestBeeswarm:

    def test_could_overlap(self):

        # 创建 Beeswarm 对象实例
        p = Beeswarm()
        # 调用 could_overlap 方法，计算与给定点可能重叠的邻居点
        neighbors = p.could_overlap(
            (1, 1, .5),  # 给定的中心点坐标和半径
            [(0, 0, .5),  # 可能的邻居点列表，每个点包括坐标和半径
             (1, .1, .2),
             (.5, .5, .5)]
        )
        # 断言结果与预期邻居点列表相等
        assert_array_equal(neighbors, [(.5, .5, .5)])

    def test_position_candidates(self):

        # 创建 Beeswarm 对象实例
        p = Beeswarm()
        # 定义中心点的坐标和半径
        xy_i = (0, 1, .5)
        # 已知的邻居点列表
        neighbors = [(0, 1, .5), (0, 1.5, .5)]
        # 计算可能的位置候选点
        candidates = p.position_candidates(xy_i, neighbors)
        # 计算预期的偏移距离
        dx1 = 1.05
        dx2 = np.sqrt(1 - .5 ** 2) * 1.05
        # 断言计算结果与预期候选点列表相等
        assert_array_equal(
            candidates,
            [(0, 1, .5), (-dx1, 1, .5), (dx1, 1, .5), (dx2, 1, .5), (-dx2, 1, .5)]
        )

    def test_find_first_non_overlapping_candidate(self):

        # 创建 Beeswarm 对象实例
        p = Beeswarm()
        # 已知的候选点列表
        candidates = [(.5, 1, .5), (1, 1, .5), (1.5, 1, .5)]
        # 已知的邻居点数组
        neighbors = np.array([(0, 1, .5)])

        # 查找第一个不重叠的候选点
        first = p.first_non_overlapping_candidate(candidates, neighbors)
        # 断言结果与预期的第一个不重叠的候选点相等
        assert_array_equal(first, (1, 1, .5))

    def test_beeswarm(self, long_df):

        # 创建 Beeswarm 对象实例
        p = Beeswarm()
        # 获取长数据框中的 y 列数据
        data = long_df["y"]
        # 计算数据的差分均值乘以1.5
        d = data.diff().mean() * 1.5
        # 初始化 x 数组为零
        x = np.zeros(data.size)
        # 对数据进行排序并赋值给 y 数组
        y = np.sort(data)
        # 初始化 r 数组，其长度与 y 数组相同，每个元素为 d
        r = np.full_like(y, d)
        # 将 x, y, r 数组合并成 orig_xyr 数组
        orig_xyr = np.c_[x, y, r]
        # 使用 Beeswarm 对象的 beeswarm 方法生成 swarm 数组，并只保留前两列
        swarm = p.beeswarm(orig_xyr)[:, :2]
        # 计算 swarm 数组中点与点之间的欧氏距离
        dmat = np.sqrt(np.sum(np.square(swarm[:, np.newaxis] - swarm), axis=-1))
        # 提取距离矩阵的上三角部分元素
        triu = dmat[np.triu_indices_from(dmat, 1)]
        # 断言所有上三角部分的距离都大于 d
        assert_array_less(d, triu)
        # 断言 y 数组与 swarm 数组中的第二列元素完全相等
        assert_array_equal(y, swarm[:, 1])

    def test_add_gutters(self):

        # 创建带有宽度参数的 Beeswarm 对象实例
        p = Beeswarm(width=1)

        # 初始化一个包含 10 个零元素的 points 数组
        points = np.zeros(10)
        # 定义两个 lambda 函数 t_fwd 和 t_inv，它们将直接返回输入值
        t_fwd = t_inv = lambda x: x
        # 断言 points 数组与添加 gutters 后的 points 数组相等
        assert_array_equal(points, p.add_gutters(points, 0, t_fwd, t_inv))

        # 初始化包含四个浮点数的 points 数组
        points = np.array([0, -1, .4, .8])
        # 定义警告信息的正则表达式字符串
        msg = r"50.0% of the points cannot be placed.+$"
        # 使用 pytest 的 warns 方法捕获 UserWarning，验证警告信息
        with pytest.warns(UserWarning, match=msg):
            # 添加 gutters 到 points 数组，得到 new_points 数组
            new_points = p.add_gutters(points, 0, t_fwd, t_inv)
        # 断言 new_points 数组与预期的数组相等
        assert_array_equal(new_points, np.array([0, -.5, .4, .5]))


class TestBoxPlotContainer:

    @pytest.fixture
    def container(self, wide_array):

        # 创建包含宽数据数组的图形对象
        ax = mpl.figure.Figure().subplots()
        # 使用 wide_array 创建箱线图，并返回艺术家对象字典
        artist_dict = ax.boxplot(wide_array)
        # 返回 BoxPlotContainer 对象，包装艺术家对象字典
        return BoxPlotContainer(artist_dict)

    def test_repr(self, container, wide_array):

        # 获取 wide_array 的列数
        n = wide_array.shape[1]
        # 断言容器对象的字符串表示是否符合预期格式
        assert str(container) == f"<BoxPlotContainer object with {n} boxes>"

    def test_iteration(self, container):
        # 遍历容器对象中的每个艺术家元组
        for artist_tuple in container:
            # 对每个元组中的属性进行断言，确保存在
            for attr in ["box", "median", "whiskers", "caps", "fliers", "mean"]:
                assert hasattr(artist_tuple, attr)

    def test_label(self, container):

        # 设置容器对象的标签
        label = "a box plot"
        container.set_label(label)
        # 断言容器对象的标签是否与设置的标签相等
        assert container.get_label() == label

    def test_children(self, container):

        # 获取容器对象的所有子艺术家对象
        children = container.get_children()
        # 断言每个子对象是否都是 mpl.artist.Artist 的实例
        for child in children:
            assert isinstance(child, mpl.artist.Artist)
```