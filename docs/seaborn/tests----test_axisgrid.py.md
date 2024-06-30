# `D:\src\scipysrc\seaborn\tests\test_axisgrid.py`

```
# 导入 NumPy 库，并使用 np 别名
import numpy as np
# 导入 Pandas 库，并使用 pd 别名
import pandas as pd
# 导入 Matplotlib 库，并使用 mpl 别名
import matplotlib as mpl
# 导入 Matplotlib.pyplot 模块，并使用 plt 别名
import matplotlib.pyplot as plt

# 导入 pytest 库
import pytest
# 导入 NumPy 测试模块，并使用 npt 别名
import numpy.testing as npt
# 从 NumPy 测试模块中导入数组相等断言函数
from numpy.testing import assert_array_equal, assert_array_almost_equal
# 导入 Pandas 测试模块，并使用 tm 别名
import pandas.testing as tm

# 从 seaborn._base 模块中导入 categorical_order 函数
from seaborn._base import categorical_order
# 从 seaborn 模块中导入 rcmod 模块
from seaborn import rcmod
# 从 seaborn.palettes 模块中导入 color_palette 函数
from seaborn.palettes import color_palette
# 从 seaborn.relational 模块中导入 scatterplot 函数
from seaborn.relational import scatterplot
# 从 seaborn.distributions 模块中导入 histplot, kdeplot, distplot 函数
from seaborn.distributions import histplot, kdeplot, distplot
# 从 seaborn.categorical 模块中导入 pointplot 函数
from seaborn.categorical import pointplot
# 从 seaborn.utils 模块中导入 _version_predates 函数
from seaborn.utils import _version_predates
# 从 seaborn 模块中导入 axisgrid，并使用 ag 别名
from seaborn import axisgrid as ag
# 从 seaborn._testing 模块中导入 assert_plots_equal, assert_colors_equal 函数
from seaborn._testing import (
    assert_plots_equal,
    assert_colors_equal,
)
# 从 seaborn._compat 模块中导入 get_legend_handles 函数

from seaborn._compat import get_legend_handles

# 创建一个随机数生成器对象 rs，使用 NumPy 中的 RandomState 对象
rs = np.random.RandomState(0)


class TestFacetGrid:

    # 创建一个测试数据集 DataFrame df，包含多列数据
    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           y=rs.gamma(4, size=60),
                           a=np.repeat(list("abc"), 20),
                           b=np.tile(list("mn"), 30),
                           c=np.tile(list("tuv"), 20),
                           d=np.tile(list("abcdefghijkl"), 5)))

    # 测试 FacetGrid 对象的 data 属性是否等于测试数据集 df
    def test_self_data(self):
        g = ag.FacetGrid(self.df)
        assert g.data is self.df

    # 测试 FacetGrid 对象的 figure 属性是否为 matplotlib 的 Figure 对象
    def test_self_figure(self):
        g = ag.FacetGrid(self.df)
        assert isinstance(g.figure, plt.Figure)
        assert g.figure is g._figure

    # 测试 FacetGrid 对象的 axes 属性是否为包含 plt.Axes 对象的数组
    def test_self_axes(self):
        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        for ax in g.axes.flat:
            assert isinstance(ax, plt.Axes)

    # 测试不同参数下 FacetGrid 对象的 axes 属性数组形状是否正确
    def test_axes_array_size(self):
        g = ag.FacetGrid(self.df)
        assert g.axes.shape == (1, 1)

        g = ag.FacetGrid(self.df, row="a")
        assert g.axes.shape == (3, 1)

        g = ag.FacetGrid(self.df, col="b")
        assert g.axes.shape == (1, 2)

        g = ag.FacetGrid(self.df, hue="c")
        assert g.axes.shape == (1, 1)

        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        assert g.axes.shape == (3, 2)
        for ax in g.axes.flat:
            assert isinstance(ax, plt.Axes)

    # 测试 FacetGrid 对象的 ax 属性是否为单个 plt.Axes 对象
    def test_single_axes(self):
        g = ag.FacetGrid(self.df)
        assert isinstance(g.ax, plt.Axes)

        g = ag.FacetGrid(self.df, row="a")
        with pytest.raises(AttributeError):
            g.ax

        g = ag.FacetGrid(self.df, col="a")
        with pytest.raises(AttributeError):
            g.ax

        g = ag.FacetGrid(self.df, col="a", row="b")
        with pytest.raises(AttributeError):
            g.ax
    def test_col_wrap(self):
        # 获取 self.df.d 列的唯一值数量
        n = len(self.df.d.unique())

        # 创建一个 FacetGrid 对象 g，按照列 "d" 进行分面显示
        g = ag.FacetGrid(self.df, col="d")
        # 断言 g 的 axes 形状为 (1, n)
        assert g.axes.shape == (1, n)
        # 断言 facet_axis(0, 8) 返回的对象与 g.axes[0, 8] 相同
        assert g.facet_axis(0, 8) is g.axes[0, 8]

        # 创建一个 FacetGrid 对象 g_wrap，按照列 "d" 进行分面显示，每行最多 4 列
        g_wrap = ag.FacetGrid(self.df, col="d", col_wrap=4)
        # 断言 g_wrap 的 axes 形状为 (n,)
        assert g_wrap.axes.shape == (n,)
        # 断言 facet_axis(0, 8) 返回的对象与 g_wrap.axes[8] 相同
        assert g_wrap.facet_axis(0, 8) is g_wrap.axes[8]
        # 断言 g_wrap 的列数为 4
        assert g_wrap._ncol == 4
        # 断言 g_wrap 的行数为 n / 4
        assert g_wrap._nrow == (n / 4)

        # 使用 pytest 检查以下代码块会引发 ValueError 异常
        with pytest.raises(ValueError):
            g = ag.FacetGrid(self.df, row="b", col="d", col_wrap=4)

        # 复制 self.df 数据框为 df
        df = self.df.copy()
        # 将 df 中 d 列为 "j" 的行设置为 NaN
        df.loc[df.d == "j"] = np.nan
        # 创建一个 FacetGrid 对象 g_missing，按照列 "d" 进行分面显示
        g_missing = ag.FacetGrid(df, col="d")
        # 断言 g_missing 的 axes 形状为 (1, n-1)
        assert g_missing.axes.shape == (1, n - 1)

        # 创建一个 FacetGrid 对象 g_missing_wrap，按照列 "d" 进行分面显示，每行最多 4 列
        g_missing_wrap = ag.FacetGrid(df, col="d", col_wrap=4)
        # 断言 g_missing_wrap 的 axes 形状为 (n-1,)
        assert g_missing_wrap.axes.shape == (n - 1,)

        # 创建一个 FacetGrid 对象 g，按照列 "d" 进行分面显示，每行只有 1 列
        g = ag.FacetGrid(self.df, col="d", col_wrap=1)
        # 断言 facet_data() 返回的迭代器长度为 n
        assert len(list(g.facet_data())) == n

    def test_normal_axes(self):
        # 创建一个空的 numpy 数组 null
        null = np.empty(0, object).flat

        # 创建一个 FacetGrid 对象 g
        g = ag.FacetGrid(self.df)
        # 断言 g 的 _bottom_axes 与 g 的 axes 相同
        npt.assert_array_equal(g._bottom_axes, g.axes.flat)
        # 断言 g 的 _not_bottom_axes 为 null
        npt.assert_array_equal(g._not_bottom_axes, null)
        # 断言 g 的 _left_axes 与 g 的 axes 相同
        npt.assert_array_equal(g._left_axes, g.axes.flat)
        # 断言 g 的 _not_left_axes 为 null
        npt.assert_array_equal(g._not_left_axes, null)
        # 断言 g 的 _inner_axes 为 null
        npt.assert_array_equal(g._inner_axes, null)

        # 创建一个 FacetGrid 对象 g，按照列 "c" 进行分面显示
        g = ag.FacetGrid(self.df, col="c")
        # 断言 g 的 _bottom_axes 与 g 的 axes 相同
        npt.assert_array_equal(g._bottom_axes, g.axes.flat)
        # 断言 g 的 _not_bottom_axes 为 null
        npt.assert_array_equal(g._not_bottom_axes, null)
        # 断言 g 的 _left_axes 为 g 的第一列 axes
        npt.assert_array_equal(g._left_axes, g.axes[:, 0].flat)
        # 断言 g 的 _not_left_axes 为 g 的除了第一列以外的所有 axes
        npt.assert_array_equal(g._not_left_axes, g.axes[:, 1:].flat)
        # 断言 g 的 _inner_axes 为 null
        npt.assert_array_equal(g._inner_axes, null)

        # 创建一个 FacetGrid 对象 g，按照行 "c" 进行分面显示
        g = ag.FacetGrid(self.df, row="c")
        # 断言 g 的 _bottom_axes 为 g 的最后一行 axes
        npt.assert_array_equal(g._bottom_axes, g.axes[-1, :].flat)
        # 断言 g 的 _not_bottom_axes 为 g 的除了最后一行以外的所有 axes
        npt.assert_array_equal(g._not_bottom_axes, g.axes[:-1, :].flat)
        # 断言 g 的 _left_axes 与 g 的 axes 相同
        npt.assert_array_equal(g._left_axes, g.axes.flat)
        # 断言 g 的 _not_left_axes 为 null
        npt.assert_array_equal(g._not_left_axes, null)
        # 断言 g 的 _inner_axes 为 null
        npt.assert_array_equal(g._inner_axes, null)

        # 创建一个 FacetGrid 对象 g，按照列 "a" 和行 "c" 进行分面显示
        g = ag.FacetGrid(self.df, col="a", row="c")
        # 断言 g 的 _bottom_axes 为 g 的最后一行 axes
        npt.assert_array_equal(g._bottom_axes, g.axes[-1, :].flat)
        # 断言 g 的 _not_bottom_axes 为 g 的除了最后一行以外的所有 axes
        npt.assert_array_equal(g._not_bottom_axes, g.axes[:-1, :].flat)
        # 断言 g 的 _left_axes 为 g 的第一列 axes
        npt.assert_array_equal(g._left_axes, g.axes[:, 0].flat)
        # 断言 g 的 _not_left_axes 为 g 的除了第一列和最后一行以外的所有 axes
        npt.assert_array_equal(g._not_left_axes, g.axes[:, 1:].flat)
        # 断言 g 的 _inner_axes 为 g 的除了第一列和最后一行以外的所有 axes
        npt.assert_array_equal(g._inner_axes, g.axes[:-1, 1:].flat)

    def test_wrapped_axes(self):
        # 创建一个空的 numpy 数组 null
        null = np.empty(0, object).flat

        # 创建一个 FacetGrid 对象 g，按照列 "a" 进行分面显示，每行最多 2 列
        g = ag.FacetGrid(self.df, col="a", col_wrap=2)
        # 断言 g 的 _bottom_axes 为 g 的第 2 和第 3 列 axes
        npt.assert_array_equal(g._bottom_axes,
                               g.axes[np.array([1, 2])].flat)
        # 断言 g 的 _not_bottom_axes 为 g 的第 1 列 axes
        npt.assert_array_equal(g._not_bottom_axes, g.axes[:1].flat)
        # 断言 g 的 _left_axes 为 g 的第 1 和第 3 列 axes
        npt.assert_array_equal(g._left_axes, g.axes[np.array([0, 2])].flat)
        # 断言 g 的 _not_left_axes 为 g 的第 2 列 axes
        npt.assert_array_equal(g._not_left_axes, g.axes[np.array([1])].flat)
        # 断言 g 的 _inner_axes 为 null
        npt.assert_array_equal(g._inner_axes, null)
    # 测试 FacetGrid 类的 axes_dict 属性

    # 创建 FacetGrid 对象 g，使用默认参数（无行或列）
    g = ag.FacetGrid(self.df)
    # 断言 g.axes_dict 是一个字典类型
    assert isinstance(g.axes_dict, dict)
    # 断言 g.axes_dict 是空的
    assert not g.axes_dict

    # 创建带有行参数的 FacetGrid 对象 g
    g = ag.FacetGrid(self.df, row="c")
    # 断言 g.axes_dict 的键列表应该与 g.row_names 相同
    assert list(g.axes_dict.keys()) == g.row_names
    # 遍历 g.row_names 和 g.axes.flat，断言 g.axes_dict 中的每个键名对应的值是 g.axes 中的对应轴对象
    for (name, ax) in zip(g.row_names, g.axes.flat):
        assert g.axes_dict[name] is ax

    # 创建带有列参数的 FacetGrid 对象 g
    g = ag.FacetGrid(self.df, col="c")
    # 断言 g.axes_dict 的键列表应该与 g.col_names 相同
    assert list(g.axes_dict.keys()) == g.col_names
    # 遍历 g.col_names 和 g.axes.flat，断言 g.axes_dict 中的每个键名对应的值是 g.axes 中的对应轴对象
    for (name, ax) in zip(g.col_names, g.axes.flat):
        assert g.axes_dict[name] is ax

    # 创建带有列参数和 col_wrap 参数的 FacetGrid 对象 g
    g = ag.FacetGrid(self.df, col="a", col_wrap=2)
    # 断言 g.axes_dict 的键列表应该与 g.col_names 相同
    assert list(g.axes_dict.keys()) == g.col_names
    # 遍历 g.col_names 和 g.axes.flat，断言 g.axes_dict 中的每个键名对应的值是 g.axes 中的对应轴对象
    for (name, ax) in zip(g.col_names, g.axes.flat):
        assert g.axes_dict[name] is ax

    # 创建同时带有行和列参数的 FacetGrid 对象 g
    g = ag.FacetGrid(self.df, row="a", col="c")
    # 遍历 g.axes_dict，断言 g.axes_dict 中每个项的键值对应 g.axes 中的对应轴对象
    for (row_var, col_var), ax in g.axes_dict.items():
        # 获取行变量 row_var 在 g.row_names 中的索引 i
        i = g.row_names.index(row_var)
        # 获取列变量 col_var 在 g.col_names 中的索引 j
        j = g.col_names.index(col_var)
        # 断言 g.axes[i, j] 是 ax
        assert g.axes[i, j] is ax
    def test_legend_data_missing_level(self):
        # 创建一个 FacetGrid 对象 g，根据 self.df 数据框的 'a' 列进行着色，顺序为 "azbc"
        g = ag.FacetGrid(self.df, hue="a", hue_order=list("azbc"))
        # 将 g 映射为 plt.plot 函数，绘制 "x" 列对应的数据与 "y" 列对应的数据
        g.map(plt.plot, "x", "y")
        # 添加图例到 g 上
        g.add_legend()

        # 生成长度为 4 的颜色列表 palette
        c1, c2, c3, c4 = color_palette(n_colors=4)
        palette = [c1, c3, c4]

        # 断言图例标题的文本为 "a"
        assert g._legend.get_title().get_text() == "a"

        # 获取图例中的线条对象
        lines = g._legend.get_lines()
        # 断言线条对象的数量与 self.df.a 唯一值的数量相等
        assert len(lines) == len(a_levels)

        # 对图例中的每条线条和颜色进行断言
        for line, hue in zip(lines, palette):
            assert_colors_equal(line.get_color(), hue)

        # 获取图例中的文本标签对象
        labels = g._legend.get_texts()
        # 断言文本标签对象的数量为 4
        assert len(labels) == 4

        # 对图例中的每个文本标签和预期级别进行断言
        for label, level in zip(labels, list("azbc")):
            assert label.get_text() == level

    def test_get_boolean_legend_data(self):
        # 在 self.df 上创建一个新列 "b_bool"，标记为 self.df.b == "m" 的布尔值
        self.df["b_bool"] = self.df.b == "m"
        # 创建一个 FacetGrid 对象 g，根据 "b_bool" 列进行着色
        g = ag.FacetGrid(self.df, hue="b_bool")
        # 将 g 映射为 plt.plot 函数，绘制 "x" 列对应的数据与 "y" 列对应的数据
        g.map(plt.plot, "x", "y")
        # 添加图例到 g 上
        g.add_legend()
        # 生成长度为 2 的颜色列表 palette
        palette = color_palette(n_colors=2)

        # 断言图例标题的文本为 "b_bool"
        assert g._legend.get_title().get_text() == "b_bool"

        # 获取图例中的线条对象
        lines = g._legend.get_lines()
        # 断言线条对象的数量与 b_levels 列表的长度相等
        assert len(lines) == len(b_levels)

        # 对图例中的每条线条和颜色进行断言
        for line, hue in zip(lines, palette):
            assert_colors_equal(line.get_color(), hue)

        # 获取图例中的文本标签对象
        labels = g._legend.get_texts()
        # 断言文本标签对象的数量与 b_levels 列表的长度相等
        assert len(labels) == len(b_levels)

        # 对图例中的每个文本标签和预期级别进行断言
        for label, level in zip(labels, b_levels):
            assert label.get_text() == level

    def test_legend_tuples(self):
        # 创建一个 FacetGrid 对象 g，根据 self.df 数据框的 'a' 列进行着色
        g = ag.FacetGrid(self.df, hue="a")
        # 将 g 映射为 plt.plot 函数，绘制 "x" 列对应的数据与 "y" 列对应的数据
        g.map(plt.plot, "x", "y")

        # 获取图例中的句柄和标签
        handles, labels = g.ax.get_legend_handles_labels()
        # 生成空的标签元组列表 label_tuples
        label_tuples = [("", l) for l in labels]
        # 创建字典 legend_data，将标签元组与句柄一一对应
        legend_data = dict(zip(label_tuples, handles))
        # 将自定义图例数据 legend_data 添加到图例中
        g.add_legend(legend_data, label_tuples)
        
        # 对图例中的每个条目和标签进行断言
        for entry, label in zip(g._legend.get_texts(), labels):
            assert entry.get_text() == label

    def test_legend_options(self):
        # 创建一个 FacetGrid 对象 g，根据 self.df 数据框的 'b' 列进行着色
        g = ag.FacetGrid(self.df, hue="b")
        # 将 g 映射为 plt.plot 函数，绘制 "x" 列对应的数据与 "y" 列对应的数据
        g.map(plt.plot, "x", "y")
        # 添加图例到 g 上
        g.add_legend()

        # 创建一个不显示图例的 FacetGrid 对象 g1
        g1 = ag.FacetGrid(self.df, hue="b", legend_out=False)
        # 向 g1 添加图例，并调整子标题
        g1.add_legend(adjust_subtitles=True)

        # 创建一个不显示图例的 FacetGrid 对象 g1
        g1 = ag.FacetGrid(self.df, hue="b", legend_out=False)
        # 向 g1 添加图例，不调整子标题
        g1.add_legend(adjust_subtitles=False)

    def test_legendout_with_colwrap(self):
        # 创建一个 FacetGrid 对象 g，根据 self.df 数据框的 'd' 列进行分列，根据 'b' 列进行着色
        g = ag.FacetGrid(self.df, col="d", hue='b',
                         col_wrap=4, legend_out=False)
        # 将 g 映射为 plt.plot 函数，绘制 "x" 列对应的数据与 "y" 列对应的数据，线宽为 3
        g.map(plt.plot, "x", "y", linewidth=3)
        # 添加图例到 g 上
        g.add_legend()

    def test_legend_tight_layout(self):
        # 创建一个 FacetGrid 对象 g，根据 self.df 数据框的 'b' 列进行着色
        g = ag.FacetGrid(self.df, hue='b')
        # 将 g 映射为 plt.plot 函数，绘制 "x" 列对应的数据与 "y" 列对应的数据，线宽为 3
        g.map(plt.plot, "x", "y", linewidth=3)
        # 添加图例到 g 上
        g.add_legend()
        # 调整图的紧凑布局
        g.tight_layout()

        # 获取坐标轴的右边缘位置
        axes_right_edge = g.ax.get_window_extent().xmax
        # 获取图例的左边缘位置
        legend_left_edge = g._legend.get_window_extent().xmin

        # 断言坐标轴的右边缘位置小于图例的左边缘位置
        assert axes_right_edge < legend_left_edge
    # 定义测试方法，用于验证 FacetGrid 对象的 subplot_kws 参数设置
    def test_subplot_kws(self):
        # 创建 FacetGrid 对象 g，使用 self.df 数据框作为数据源，关闭去除轴线选项，设置极坐标投影
        g = ag.FacetGrid(self.df, despine=False,
                         subplot_kws=dict(projection="polar"))
        # 遍历 g 对象中的所有子图 ax
        for ax in g.axes.flat:
            # 断言子图 ax 是极坐标类型
            assert "PolarAxes" in ax.__class__.__name__

    # 定义测试方法，验证 FacetGrid 对象的 gridspec_kws 参数设置
    def test_gridspec_kws(self):
        # 定义子图宽度比例
        ratios = [3, 1, 2]

        # 设置 gridspec_kws 参数，指定子图宽度比例
        gskws = dict(width_ratios=ratios)
        # 创建 FacetGrid 对象 g，使用 self.df 数据框作为数据源，指定列 'c' 和行 'a'，应用 gridspec_kws 参数
        g = ag.FacetGrid(self.df, col='c', row='a', gridspec_kws=gskws)

        # 遍历 g 对象中的所有子图 ax
        for ax in g.axes.flat:
            # 设置子图 ax 的 x 轴和 y 轴刻度为空
            ax.set_xticks([])
            ax.set_yticks([])

        # 调整图形布局
        g.figure.tight_layout()

        # 遍历 g 对象中的每一列 (l, m, r)
        for (l, m, r) in g.axes:
            # 断言左侧列 l 的宽度大于中间列 m 的宽度
            assert l.get_position().width > m.get_position().width
            # 断言右侧列 r 的宽度大于中间列 m 的宽度
            assert r.get_position().width > m.get_position().width

    # 定义测试方法，验证 FacetGrid 对象的 gridspec_kws 参数在列换行模式下的设置
    def test_gridspec_kws_col_wrap(self):
        # 定义子图宽度比例
        ratios = [3, 1, 2, 1, 1]

        # 设置 gridspec_kws 参数，指定子图宽度比例
        gskws = dict(width_ratios=ratios)
        # 使用 pytest 的警告捕捉，创建 FacetGrid 对象，使用 self.df 数据框作为数据源，指定列 'd'，列换行数 5，应用 gridspec_kws 参数
        with pytest.warns(UserWarning):
            ag.FacetGrid(self.df, col='d', col_wrap=5, gridspec_kws=gskws)

    # 定义测试方法，验证 FacetGrid 对象的 facet_data 方法的行为
    def test_data_generator(self):
        # 创建 FacetGrid 对象 g，指定行 'a' 作为分面变量
        g = ag.FacetGrid(self.df, row="a")
        # 调用 facet_data 方法生成数据列表 d
        d = list(g.facet_data())
        # 断言数据列表 d 的长度为 3
        assert len(d) == 3

        # 检查第一个元组 tup 和其数据 data
        tup, data = d[0]
        # 断言元组 tup 的值
        assert tup == (0, 0, 0)
        # 断言数据 data 的 'a' 列所有值为 "a"
        assert (data["a"] == "a").all()

        # 检查第二个元组 tup 和其数据 data
        tup, data = d[1]
        # 断言元组 tup 的值
        assert tup == (1, 0, 0)
        # 断言数据 data 的 'a' 列所有值为 "b"
        assert (data["a"] == "b").all()

        # 创建 FacetGrid 对象 g，同时指定行 'a' 和列 'b' 作为分面变量
        g = ag.FacetGrid(self.df, row="a", col="b")
        # 调用 facet_data 方法生成数据列表 d
        d = list(g.facet_data())
        # 断言数据列表 d 的长度为 6
        assert len(d) == 6

        # 检查第一个元组 tup 和其数据 data
        tup, data = d[0]
        # 断言元组 tup 的值
        assert tup == (0, 0, 0)
        # 断言数据 data 的 'a' 列所有值为 "a"，'b' 列所有值为 "m"
        assert (data["a"] == "a").all()
        assert (data["b"] == "m").all()

        # 检查第二个元组 tup 和其数据 data
        tup, data = d[1]
        # 断言元组 tup 的值
        assert tup == (0, 1, 0)
        # 断言数据 data 的 'a' 列所有值为 "a"，'b' 列所有值为 "n"
        assert (data["a"] == "a").all()
        assert (data["b"] == "n").all()

        # 检查第三个元组 tup 和其数据 data
        tup, data = d[2]
        # 断言元组 tup 的值
        assert tup == (1, 0, 0)
        # 断言数据 data 的 'a' 列所有值为 "b"，'b' 列所有值为 "m"
        assert (data["a"] == "b").all()
        assert (data["b"] == "m").all()

        # 创建 FacetGrid 对象 g，指定色彩变量 'c' 作为分面变量
        g = ag.FacetGrid(self.df, hue="c")
        # 调用 facet_data 方法生成数据列表 d
        d = list(g.facet_data())
        # 断言数据列表 d 的长度为 3
        assert len(d) == 3
        # 检查第二个元组 tup 和其数据 data
        tup, data = d[1]
        # 断言元组 tup 的值
        assert tup == (0, 0, 1)
        # 断言数据 data 的 'c' 列所有值为 "u"
        assert (data["c"] == "u").all()

    # 定义测试方法，验证 FacetGrid 对象的 map 方法的行为
    def test_map(self):
        # 创建 FacetGrid 对象 g，同时指定行 'a'、列 'b' 和色彩变量 'c' 作为分面变量
        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        # 使用 plot 函数绘制数据列 "x" 和 "y" 的折线图，设置线宽为 3
        g.map(plt.plot, "x", "y", linewidth=3)

        # 获取 g 对象中第一个子图的所有线条
        lines = g.axes[0, 0].lines
        # 断言线条的数量为 3
        assert len(lines) == 3

        # 获取第一个线条 line1 及其属性
        line1, _, _ = lines
        # 断言线条 line1 的线宽为 3
        assert line1.get_linewidth() == 3
        # 获取线条 line1 的数据 x 和 y
        x, y = line1.get_data()
        # 根据条件筛选 self.df 数据框的数据，并使用 numpy 的数组断言函数比较 x 和 y 的值
        mask = (self.df.a == "a") & (self.df.b == "m") & (self.df.c == "t")
        npt.assert_array_equal(x, self.df.x[mask])
        npt.assert_array_equal(y, self.df.y[mask])
    def test_map_dataframe(self):
        # 创建一个 FacetGrid 对象，基于 self.df 数据框，按照 "a" 列分行，"b" 列分列，"c" 列作为颜色变量
        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")

        # 定义一个自定义的绘图函数 plot，用于在 FacetGrid 中绘制 x 和 y 列的数据
        def plot(x, y, data=None, **kws):
            plt.plot(data[x], data[y], **kws)
        
        # 修改 plot 函数的 __module__ 属性，使其不再看起来像 seaborn 的函数
        plot.__module__ = "test"

        # 在 FacetGrid 中应用 plot 函数，绘制 "x" 列和 "y" 列的数据，使用虚线样式 "--"
        g.map_dataframe(plot, "x", "y", linestyle="--")

        # 获取 FacetGrid 中第一个子图的所有线条对象
        lines = g.axes[0, 0].lines
        # 断言第一个子图中存在三条线条
        assert len(g.axes[0, 0].lines) == 3

        # 分别获取第一条线条的对象
        line1, _, _ = lines
        # 断言第一条线条的线型为虚线 "--"
        assert line1.get_linestyle() == "--"
        # 获取第一条线条的 x 和 y 数据
        x, y = line1.get_data()
        # 创建一个布尔掩码，用于选择 self.df 数据框中满足条件的行
        mask = (self.df.a == "a") & (self.df.b == "m") & (self.df.c == "t")
        # 断言第一条线条的 x 数据与满足掩码条件的 self.df.x 列数据相等
        npt.assert_array_equal(x, self.df.x[mask])
        # 断言第一条线条的 y 数据与满足掩码条件的 self.df.y 列数据相等
        npt.assert_array_equal(y, self.df.y[mask])

    def test_set(self):
        # 创建一个 FacetGrid 对象，基于 self.df 数据框，按照 "a" 列分行，"b" 列分列
        g = ag.FacetGrid(self.df, row="a", col="b")
        # 设置所有子图的 x 和 y 轴限制、刻度值
        xlim = (-2, 5)
        ylim = (3, 6)
        xticks = [-2, 0, 3, 5]
        yticks = [3, 4.5, 6]
        g.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
        # 对每个子图进行断言，检查其 x 和 y 轴的限制、刻度值是否与设置一致
        for ax in g.axes.flat:
            npt.assert_array_equal(ax.get_xlim(), xlim)
            npt.assert_array_equal(ax.get_ylim(), ylim)
            npt.assert_array_equal(ax.get_xticks(), xticks)
            npt.assert_array_equal(ax.get_yticks(), yticks)

    def test_set_titles(self):
        # 创建一个 FacetGrid 对象，基于 self.df 数据框，按照 "a" 列分行，"b" 列分列
        g = ag.FacetGrid(self.df, row="a", col="b")
        # 在 FacetGrid 中绘制 "x" 列和 "y" 列的数据
        g.map(plt.plot, "x", "y")

        # 测试默认的子图标题
        assert g.axes[0, 0].get_title() == "a = a | b = m"
        assert g.axes[0, 1].get_title() == "a = a | b = n"
        assert g.axes[1, 0].get_title() == "a = b | b = m"

        # 设置自定义的子图标题格式
        g.set_titles("{row_var} == {row_name} \\/ {col_var} == {col_name}")
        assert g.axes[0, 0].get_title() == "a == a \\/ b == m"
        assert g.axes[0, 1].get_title() == "a == a \\/ b == n"
        assert g.axes[1, 0].get_title() == "a == b \\/ b == m"

        # 测试单行情况
        g = ag.FacetGrid(self.df, col="b")
        g.map(plt.plot, "x", "y")

        # 测试默认的子图标题
        assert g.axes[0, 0].get_title() == "b = m"
        assert g.axes[0, 1].get_title() == "b = n"

        # 测试 dropna=False 的情况
        g = ag.FacetGrid(self.df, col="b", hue="b", dropna=False)
        g.map(plt.plot, 'x', 'y')
    # 定义一个测试方法，用于测试设置边距标题的功能
    def test_set_titles_margin_titles(self):
        
        # 创建一个 FacetGrid 对象 g，使用 self.df 中的数据，按行 "a" 和列 "b" 排列，启用边距标题
        g = ag.FacetGrid(self.df, row="a", col="b", margin_titles=True)
        
        # 在 g 上映射 plt.plot 函数，绘制 "x" 对 "y" 的图形
        g.map(plt.plot, "x", "y")

        # 测试默认的标题
        assert g.axes[0, 0].get_title() == "b = m"
        assert g.axes[0, 1].get_title() == "b = n"
        assert g.axes[1, 0].get_title() == ""

        # 测试行标题
        assert g.axes[0, 1].texts[0].get_text() == "a = a"
        assert g.axes[1, 1].texts[0].get_text() == "a = b"
        assert g.axes[0, 1].texts[0] is g._margin_titles_texts[0]

        # 测试提供的标题模板
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        assert g.axes[0, 0].get_title() == "m"
        assert g.axes[0, 1].get_title() == "n"
        assert g.axes[1, 0].get_title() == ""

        assert len(g.axes[1, 1].texts) == 1
        assert g.axes[1, 1].texts[0].get_text() == "b"

    # 定义一个测试方法，用于测试设置刻度标签的功能
    def test_set_ticklabels(self):
        
        # 创建一个 FacetGrid 对象 g，使用 self.df 中的数据，按行 "a" 和列 "b" 排列
        g = ag.FacetGrid(self.df, row="a", col="b")
        
        # 在 g 上映射 plt.plot 函数，绘制 "x" 对 "y" 的图形
        g.map(plt.plot, "x", "y")

        # 获取最后一行第一列的轴对象 ax
        ax = g.axes[-1, 0]
        
        # 获取 x 轴刻度标签，并添加 "h" 后缀
        xlab = [l.get_text() + "h" for l in ax.get_xticklabels()]
        
        # 获取 y 轴刻度标签，并添加 "i" 后缀
        ylab = [l.get_text() + "i" for l in ax.get_yticklabels()]

        # 设置新的 x 轴刻度标签
        g.set_xticklabels(xlab)
        
        # 设置新的 y 轴刻度标签
        g.set_yticklabels(ylab)
        
        # 获取最后一行第二列的轴对象的 x 轴刻度标签
        got_x = [l.get_text() for l in g.axes[-1, 1].get_xticklabels()]
        
        # 获取第一行第一列的轴对象的 y 轴刻度标签
        got_y = [l.get_text() for l in g.axes[0, 0].get_yticklabels()]
        
        # 断言新的 x 轴刻度标签与预期一致
        npt.assert_array_equal(got_x, xlab)
        
        # 断言新的 y 轴刻度标签与预期一致
        npt.assert_array_equal(got_y, ylab)

        # 创建数据框 df，包含 x 和 y 列
        x, y = np.arange(10), np.arange(10)
        df = pd.DataFrame(np.c_[x, y], columns=["x", "y"])
        
        # 创建 FacetGrid 对象 g，使用 df，绘制 x 对 y 的点图，按 x 的顺序
        g = ag.FacetGrid(df).map_dataframe(pointplot, x="x", y="y", order=x)
        
        # 设置 x 轴刻度标签的步长为 2
        g.set_xticklabels(step=2)
        
        # 获取第一行第一列的轴对象的 x 轴刻度标签，转换为整数类型
        got_x = [int(l.get_text()) for l in g.axes[0, 0].get_xticklabels()]
        
        # 断言新的 x 轴刻度标签与预期一致
        npt.assert_array_equal(x[::2], got_x)

        # 创建 FacetGrid 对象 g，使用 self.df，按列 "d" 排列，每行最多显示 5 个图
        g = ag.FacetGrid(self.df, col="d", col_wrap=5)
        
        # 在 g 上映射 plt.plot 函数，绘制 "x" 对 "y" 的图形
        g.map(plt.plot, "x", "y")
        
        # 设置 x 轴刻度标签的旋转角度为 45 度
        g.set_xticklabels(rotation=45)
        
        # 设置 y 轴刻度标签的旋转角度为 75 度
        g.set_yticklabels(rotation=75)
        
        # 遍历 g 的底部轴对象，断言其 x 轴刻度标签的旋转角度为 45 度
        for ax in g._bottom_axes:
            for l in ax.get_xticklabels():
                assert l.get_rotation() == 45
        
        # 遍历 g 的左侧轴对象，断言其 y 轴刻度标签的旋转角度为 75 度
        for ax in g._left_axes:
            for l in ax.get_yticklabels():
                assert l.get_rotation() == 75

    # 定义一个测试方法，用于测试设置轴标签的功能
    def test_set_axis_labels(self):
        
        # 创建一个 FacetGrid 对象 g，使用 self.df 中的数据，按行 "a" 和列 "b" 排列
        g = ag.FacetGrid(self.df, row="a", col="b")
        
        # 在 g 上映射 plt.plot 函数，绘制 "x" 对 "y" 的图形
        g.map(plt.plot, "x", "y")
        
        # 设置 x 轴标签和 y 轴标签
        xlab = 'xx'
        ylab = 'yy'
        g.set_axis_labels(xlab, ylab)
        
        # 获取最后一行的所有列的 x 轴标签
        got_x = [ax.get_xlabel() for ax in g.axes[-1, :]]
        
        # 获取第一列的所有行的 y 轴标签
        got_y = [ax.get_ylabel() for ax in g.axes[:, 0]]
        
        # 断言新设置的 x 轴标签与预期一致
        npt.assert_array_equal(got_x, xlab)
        
        # 断言新设置的 y 轴标签与预期一致
        npt.assert_array_equal(got_y, ylab)

        # 遍历所有轴对象，设置它们的 x 轴标签为 "x"，y 轴标签为 "y"
        for ax in g.axes.flat:
            ax.set(xlabel="x", ylabel="y")

        # 再次设置 x 轴标签和 y 轴标签
        g.set_axis_labels(xlab, ylab)
        
        # 遍历除了底部轴对象之外的所有轴对象，断言它们的 x 轴标签为空
        for ax in g._not_bottom_axes:
            assert not ax.get_xlabel()
        
        # 遍历除了左侧轴对象之外的所有轴对象，断言它们的 y 轴标签为空
        for ax in g._not_left_axes:
            assert not ax.get_ylabel()
    def test_axis_lims(self):
        # 创建一个 FacetGrid 对象，指定行和列，并设置 x 和 y 轴的限制
        g = ag.FacetGrid(self.df, row="a", col="b", xlim=(0, 4), ylim=(-2, 3))
        # 断言获取到的 x 轴限制是否符合预期
        assert g.axes[0, 0].get_xlim() == (0, 4)
        # 断言获取到的 y 轴限制是否符合预期
        assert g.axes[0, 0].get_ylim() == (-2, 3)

    def test_data_orders(self):
        # 创建一个 FacetGrid 对象，指定行、列和色调
        g = ag.FacetGrid(self.df, row="a", col="b", hue="c")
        # 断言行名称是否与预期列表相同
        assert g.row_names == list("abc")
        # 断言列名称是否与预期列表相同
        assert g.col_names == list("mn")
        # 断言色调名称是否与预期列表相同
        assert g.hue_names == list("tuv")
        # 断言 FacetGrid 的 axes 属性形状是否为 (3, 2)
        assert g.axes.shape == (3, 2)

        # 创建一个 FacetGrid 对象，同时指定行、列、色调的顺序
        g = ag.FacetGrid(self.df, row="a", col="b", hue="c",
                         row_order=list("bca"),
                         col_order=list("nm"),
                         hue_order=list("vtu"))
        # 断言行名称是否与指定的顺序列表相同
        assert g.row_names == list("bca")
        # 断言列名称是否与指定的顺序列表相同
        assert g.col_names == list("nm")
        # 断言色调名称是否与指定的顺序列表相同
        assert g.hue_names == list("vtu")
        # 断言 FacetGrid 的 axes 属性形状是否为 (3, 2)
        assert g.axes.shape == (3, 2)

        # 创建一个 FacetGrid 对象，同时指定行、列、色调的顺序，这次行数不同
        g = ag.FacetGrid(self.df, row="a", col="b", hue="c",
                         row_order=list("bcda"),
                         col_order=list("nom"),
                         hue_order=list("qvtu"))
        # 断言行名称是否与指定的顺序列表相同
        assert g.row_names == list("bcda")
        # 断言列名称是否与指定的顺序列表相同
        assert g.col_names == list("nom")
        # 断言色调名称是否与指定的顺序列表相同
        assert g.hue_names == list("qvtu")
        # 断言 FacetGrid 的 axes 属性形状是否为 (4, 3)
        assert g.axes.shape == (4, 3)

    def test_palette(self):
        # 设置默认的绘图参数
        rcmod.set()

        # 创建一个 FacetGrid 对象，指定色调为 "c"
        g = ag.FacetGrid(self.df, hue="c")
        # 断言颜色列表是否与数据集中唯一值的数量相匹配
        assert g._colors == color_palette(n_colors=len(self.df.c.unique()))

        # 创建一个 FacetGrid 对象，指定色调为 "d"
        g = ag.FacetGrid(self.df, hue="d")
        # 断言颜色列表是否与指定色调方法 "husl" 下数据集唯一值的数量相匹配
        assert g._colors == color_palette("husl", len(self.df.d.unique()))

        # 创建一个 FacetGrid 对象，指定色调为 "c"，并使用 "Set2" 调色板
        g = ag.FacetGrid(self.df, hue="c", palette="Set2")
        # 断言颜色列表是否与指定的 "Set2" 调色板下数据集唯一值的数量相匹配
        assert g._colors == color_palette("Set2", len(self.df.c.unique()))

        # 创建一个颜色字典和对应的调色板列表
        dict_pal = dict(t="red", u="green", v="blue")
        list_pal = color_palette(["red", "green", "blue"], 3)
        # 创建一个 FacetGrid 对象，指定色调为 "c"，使用自定义的调色板
        g = ag.FacetGrid(self.df, hue="c", palette=dict_pal)
        # 断言颜色列表是否与指定的调色板列表相匹配
        assert g._colors == list_pal

        # 创建一个 FacetGrid 对象，指定色调为 "c"，行、列和色调的顺序，使用自定义的调色板
        list_pal = color_palette(["green", "blue", "red"], 3)
        g = ag.FacetGrid(self.df, hue="c", hue_order=list("uvt"),
                         palette=dict_pal)
        # 断言颜色列表是否与指定的调色板列表相匹配
        assert g._colors == list_pal

    def test_hue_kws(self):
        # 设置绘图参数字典，指定不同的标记样式
        kws = dict(marker=["o", "s", "D"])
        # 创建一个 FacetGrid 对象，指定色调为 "c"，并传入标记样式参数
        g = ag.FacetGrid(self.df, hue="c", hue_kws=kws)
        # 在 FacetGrid 中绘制 x 到 y 的散点图
        g.map(plt.plot, "x", "y")

        # 循环遍历第一个子图中的所有线条，并与预期的标记样式进行比较
        for line, marker in zip(g.axes[0, 0].lines, kws["marker"]):
            # 断言每条线的标记样式是否与预期相符
            assert line.get_marker() == marker

    def test_dropna(self):
        # 复制数据集
        df = self.df.copy()
        # 创建一个包含缺失值的 Series，并添加到数据集中
        hasna = pd.Series(np.tile(np.arange(6), 10), dtype=float)
        hasna[hasna == 5] = np.nan
        df["hasna"] = hasna
        # 创建一个 FacetGrid 对象，指定行为 "hasna"，同时保留缺失值
        g = ag.FacetGrid(df, dropna=False, row="hasna")
        # 断言非缺失值数量是否为预期值
        assert g._not_na.sum() == 60

        # 创建一个 FacetGrid 对象，指定行为 "hasna"，同时删除缺失值
        g = ag.FacetGrid(df, dropna=True, row="hasna")
        # 断言非缺失值数量是否为预期值
        assert g._not_na.sum() == 50

    def test_categorical_column_missing_categories(self):
        # 复制数据集，并将列 'a' 转换为分类数据类型
        df = self.df.copy()
        df['a'] = df['a'].astype('category')

        # 创建一个 FacetGrid 对象，按列 "a" 分组，每行包含一个分类
        g = ag.FacetGrid(df[df['a'] == 'a'], col="a", col_wrap=1)

        # 断言 FacetGrid 的 axes 属性形状是否为指定分类的数量
        assert g.axes.shape == (len(df['a'].cat.categories),)
    # 测试用例：检查在使用点图映射时是否会引发用户警告
    def test_categorical_warning(self):
        # 创建一个带有列参数的 FacetGrid 对象 g
        g = ag.FacetGrid(self.df, col="b")
        # 使用 pytest 的 warns 方法检查是否会引发 UserWarning
        with pytest.warns(UserWarning):
            # 对 g 对象应用点图映射
            g.map(pointplot, "b", "x")

    # 测试用例：检查在添加参考线后的行为
    def test_refline(self):
        # 创建一个带有行和列参数的 FacetGrid 对象 g
        g = ag.FacetGrid(self.df, row="a", col="b")
        # 在 FacetGrid 上添加参考线
        g.refline()
        # 遍历所有的子图对象，确保没有绘制的线条
        for ax in g.axes.flat:
            assert not ax.lines

        # 定义参考线的 x 和 y 值
        refx = refy = 0.5
        # 定义水平和垂直参考线的坐标数组
        hline = np.array([[0, refy], [1, refy]])
        vline = np.array([[refx, 0], [refx, 1]])
        # 在 FacetGrid 上添加带有指定 x 和 y 值的参考线
        g.refline(x=refx, y=refy)
        # 再次遍历所有的子图对象，检查绘制的线条特性
        for ax in g.axes.flat:
            assert ax.lines[0].get_color() == '.5'
            assert ax.lines[0].get_linestyle() == '--'
            assert len(ax.lines) == 2
            npt.assert_array_equal(ax.lines[0].get_xydata(), vline)
            npt.assert_array_equal(ax.lines[1].get_xydata(), hline)

        # 定义颜色和线型
        color, linestyle = 'red', '-'
        # 在 FacetGrid 上添加带有指定颜色和线型的参考线
        g.refline(x=refx, color=color, linestyle=linestyle)
        # 断言特定子图对象的最后一条线条的坐标与颜色和线型相符
        npt.assert_array_equal(g.axes[0, 0].lines[-1].get_xydata(), vline)
        assert g.axes[0, 0].lines[-1].get_color() == color
        assert g.axes[0, 0].lines[-1].get_linestyle() == linestyle

    # 测试用例：检查在应用函数后 FacetGrid 对象的行为
    def test_apply(self, long_df):
        # 定义一个函数 f，用于修改图形对象的背景颜色
        def f(grid, color):
            grid.figure.set_facecolor(color)

        # 定义颜色
        color = (.1, .6, .3, .9)
        # 创建一个 FacetGrid 对象 g
        g = ag.FacetGrid(long_df)
        # 应用函数 f 到 FacetGrid 对象 g，传入颜色参数
        res = g.apply(f, color)
        # 断言应用函数后返回的对象是原对象 g
        assert res is g
        # 断言图形对象的背景颜色与预期颜色相符
        assert g.figure.get_facecolor() == color

    # 测试用例：检查在管道操作后 FacetGrid 对象的行为
    def test_pipe(self, long_df):
        # 定义一个函数 f，用于修改图形对象的背景颜色，并返回颜色
        def f(grid, color):
            grid.figure.set_facecolor(color)
            return color

        # 定义颜色
        color = (.1, .6, .3, .9)
        # 创建一个 FacetGrid 对象 g
        g = ag.FacetGrid(long_df)
        # 使用管道方式应用函数 f 到 FacetGrid 对象 g，传入颜色参数
        res = g.pipe(f, color)
        # 断言管道操作返回的结果与预期颜色相符
        assert res == color
        # 断言图形对象的背景颜色与预期颜色相符
        assert g.figure.get_facecolor() == color

    # 测试用例：检查在设置刻度参数后 FacetGrid 对象的行为
    def test_tick_params(self):
        # 创建一个带有行和列参数的 FacetGrid 对象 g
        g = ag.FacetGrid(self.df, row="a", col="b")
        # 定义颜色和间距
        color = "blue"
        pad = 3
        # 设置刻度参数，包括颜色和间距
        g.tick_params(pad=pad, color=color)
        # 遍历所有的子图对象
        for ax in g.axes.flat:
            # 遍历 x 轴和 y 轴的主刻度
            for axis in ["xaxis", "yaxis"]:
                for tick in getattr(ax, axis).get_major_ticks():
                    # 断言刻度线的颜色与设置的颜色相符
                    assert mpl.colors.same_color(tick.tick1line.get_color(), color)
                    assert mpl.colors.same_color(tick.tick2line.get_color(), color)
                    # 断言刻度线的间距与设置的间距相符
                    assert tick.get_pad() == pad

    # 根据条件标记跳过的测试用例：测试数据交换行为，前提是支持数据框交换
    @pytest.mark.skipif(
        condition=not hasattr(pd.api, "interchange"),
        reason="Tests behavior assuming support for dataframe interchange"
    )
    def test_data_interchange(self, mock_long_df, long_df):
        # 创建一个带有列和行参数的 FacetGrid 对象 g
        g = ag.FacetGrid(mock_long_df, col="a", row="b")
        # 对 FacetGrid 对象 g 应用散点图映射
        g.map(scatterplot, "x", "y")

        # 断言生成的子图数组的形状与预期的数据框唯一值数量相符
        assert g.axes.shape == (long_df["b"].nunique(), long_df["a"].nunique())
        # 遍历所有的子图对象
        for ax in g.axes.flat:
            # 断言每个子图对象只包含一个数据集合（散点图）
            assert len(ax.collections) == 1
# 定义一个测试类 TestPairGrid，用于测试 PairGrid 类的功能
class TestPairGrid:

    # 使用随机数生成器创建一个随机状态对象 rs
    rs = np.random.RandomState(sum(map(ord, "PairGrid")))
    
    # 创建一个 DataFrame 对象 df，包含以下列：x（正态分布随机数）、y（0到3之间的随机整数）、z（Gamma分布随机数）、a（字符重复组成的列表）、b（字符重复组成的列表）
    df = pd.DataFrame(dict(x=rs.normal(size=60),
                           y=rs.randint(0, 4, size=(60)),
                           z=rs.gamma(3, size=60),
                           a=np.repeat(list("abc"), 20),
                           b=np.repeat(list("abcdefghijkl"), 5)))

    # 测试方法：验证 PairGrid 对象的数据属性与预期是否一致
    def test_self_data(self):
        g = ag.PairGrid(self.df)
        assert g.data is self.df

    # 测试方法：验证在添加日期列后，PairGrid 对象的数据属性是否正确移除了该列
    def test_ignore_datelike_data(self):
        df = self.df.copy()
        df['date'] = pd.date_range('2010-01-01', periods=len(df), freq='d')
        result = ag.PairGrid(self.df).data
        expected = df.drop('date', axis=1)
        tm.assert_frame_equal(result, expected)

    # 测试方法：验证 PairGrid 对象的 figure 属性是 matplotlib 的 Figure 对象，并且 figure 和 _figure 是同一个对象
    def test_self_figure(self):
        g = ag.PairGrid(self.df)
        assert isinstance(g.figure, plt.Figure)
        assert g.figure is g._figure

    # 测试方法：验证 PairGrid 对象中每个子图对象都是 matplotlib 的 Axes 对象
    def test_self_axes(self):
        g = ag.PairGrid(self.df)
        for ax in g.axes.flat:
            assert isinstance(ax, plt.Axes)

    # 测试方法：验证默认情况下 PairGrid 对象的属性设置是否正确
    def test_default_axes(self):
        g = ag.PairGrid(self.df)
        assert g.axes.shape == (3, 3)
        assert g.x_vars == ["x", "y", "z"]
        assert g.y_vars == ["x", "y", "z"]
        assert g.square_grid

    # 测试方法：通过参数化测试，验证指定特定变量后 PairGrid 对象的属性设置是否正确
    @pytest.mark.parametrize("vars", [["z", "x"], np.array(["z", "x"])])
    def test_specific_square_axes(self, vars):
        g = ag.PairGrid(self.df, vars=vars)
        assert g.axes.shape == (len(vars), len(vars))
        assert g.x_vars == list(vars)
        assert g.y_vars == list(vars)
        assert g.square_grid

    # 测试方法：验证在指定 hue 参数后，PairGrid 对象的 x_vars 和 y_vars 是否正确移除了该参数
    def test_remove_hue_from_default(self):
        hue = "z"
        g = ag.PairGrid(self.df, hue=hue)
        assert hue not in g.x_vars
        assert hue not in g.y_vars

        vars = ["x", "y", "z"]
        g = ag.PairGrid(self.df, hue=hue, vars=vars)
        assert hue in g.x_vars
        assert hue in g.y_vars

    # 测试方法：通过参数化测试，验证在指定非方形的 x_vars 和 y_vars 后，PairGrid 对象的属性设置是否正确
    @pytest.mark.parametrize(
        "x_vars, y_vars",
        [
            (["x", "y"], ["z", "y", "x"]),
            (["x", "y"], "z"),
            (np.array(["x", "y"]), np.array(["z", "y", "x"])),
        ],
    )
    def test_specific_nonsquare_axes(self, x_vars, y_vars):
        g = ag.PairGrid(self.df, x_vars=x_vars, y_vars=y_vars)
        assert g.axes.shape == (len(y_vars), len(x_vars))
        assert g.x_vars == list(x_vars)
        assert g.y_vars == list(y_vars)
        assert not g.square_grid
    def test_corner(self):
        # 定义要绘制的变量列表
        plot_vars = ["x", "y", "z"]
        # 创建 PairGrid 对象，设置 corner 参数为 True
        g = ag.PairGrid(self.df, vars=plot_vars, corner=True)
        # 计算角落图的总数
        corner_size = sum(i + 1 for i in range(len(plot_vars)))
        # 断言图形中的轴数量等于角落图的总数
        assert len(g.figure.axes) == corner_size

        # 对角线上的图使用 plt.hist 绘制
        g.map_diag(plt.hist)
        # 断言图形中的轴数量增加到角落图总数加上变量数量
        assert len(g.figure.axes) == (corner_size + len(plot_vars))

        # 对角线上的图的 y 轴不可见
        for ax in np.diag(g.axes):
            assert not ax.yaxis.get_visible()

        # 重新定义要绘制的变量列表
        plot_vars = ["x", "y", "z"]
        # 再次创建 PairGrid 对象，设置 corner 参数为 True
        g = ag.PairGrid(self.df, vars=plot_vars, corner=True)
        # 使用 scatterplot 函数绘制图形
        g.map(scatterplot)
        # 断言图形中的轴数量等于角落图的总数
        assert len(g.figure.axes) == corner_size
        # 断言第一个轴的 y 轴标签为 "x"
        assert g.axes[0, 0].get_ylabel() == "x"

    def test_size(self):
        # 创建高度为 3 的 PairGrid 对象
        g1 = ag.PairGrid(self.df, height=3)
        # 断言图形的尺寸为 (9, 9) 英寸
        npt.assert_array_equal(g1.fig.get_size_inches(), (9, 9))

        # 创建高度为 4，宽高比为 0.5 的 PairGrid 对象
        g2 = ag.PairGrid(self.df, height=4, aspect=.5)
        # 断言图形的尺寸为 (6, 12) 英寸
        npt.assert_array_equal(g2.fig.get_size_inches(), (6, 12))

        # 创建只包含特定变量的 PairGrid 对象，设置高度为 2，宽高比为 2
        g3 = ag.PairGrid(self.df, y_vars=["z"], x_vars=["x", "y"],
                         height=2, aspect=2)
        # 断言图形的尺寸为 (8, 2) 英寸
        npt.assert_array_equal(g3.fig.get_size_inches(), (8, 2))

    def test_empty_grid(self):
        # 断言创建仅包含变量 "a" 和 "b" 的 PairGrid 对象会引发 ValueError，错误信息包含 "No variables found"
        with pytest.raises(ValueError, match="No variables found"):
            ag.PairGrid(self.df[["a", "b"]])

    def test_map(self):
        # 定义要绘制的变量列表
        vars = ["x", "y", "z"]
        # 创建空的 PairGrid 对象
        g1 = ag.PairGrid(self.df)
        # 使用 plt.scatter 函数映射到所有轴
        g1.map(plt.scatter)

        # 遍历每个轴及其索引
        for i, axes_i in enumerate(g1.axes):
            for j, ax in enumerate(axes_i):
                # 获取对应的输入数据和输出数据，并进行断言比较
                x_in = self.df[vars[j]]
                y_in = self.df[vars[i]]
                x_out, y_out = ax.collections[0].get_offsets().T
                npt.assert_array_equal(x_in, x_out)
                npt.assert_array_equal(y_in, y_out)

        # 创建带有色调变量 "a" 的 PairGrid 对象
        g2 = ag.PairGrid(self.df, hue="a")
        # 使用 plt.scatter 函数映射到所有轴
        g2.map(plt.scatter)

        # 遍历每个轴及其索引
        for i, axes_i in enumerate(g2.axes):
            for j, ax in enumerate(axes_i):
                # 获取对应的输入数据和输出数据，并进行断言比较
                x_in = self.df[vars[j]]
                y_in = self.df[vars[i]]
                for k, k_level in enumerate(self.df.a.unique()):
                    x_in_k = x_in[self.df.a == k_level]
                    y_in_k = y_in[self.df.a == k_level]
                    x_out, y_out = ax.collections[k].get_offsets().T
                npt.assert_array_equal(x_in_k, x_out)
                npt.assert_array_equal(y_in_k, y_out)

    def test_map_nonsquare(self):
        # 定义 x 轴和 y 轴变量列表
        x_vars = ["x"]
        y_vars = ["y", "z"]
        # 创建 PairGrid 对象，指定 x 轴和 y 轴的变量
        g = ag.PairGrid(self.df, x_vars=x_vars, y_vars=y_vars)
        # 使用 plt.scatter 函数映射到所有轴
        g.map(plt.scatter)

        # 获取 x 轴和 y 轴的输入数据
        x_in = self.df.x
        for i, i_var in enumerate(y_vars):
            # 获取当前轴对象
            ax = g.axes[i, 0]
            # 获取 y 轴的输入数据
            y_in = self.df[i_var]
            # 获取散点图的偏移数据并进行断言比较
            x_out, y_out = ax.collections[0].get_offsets().T
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)
    # 定义一个测试方法，用于测试 PairGrid 对象的 map_lower 方法
    def test_map_lower(self):
        # 定义变量列表
        vars = ["x", "y", "z"]
        # 创建一个 PairGrid 对象 g，传入数据框 self.df
        g = ag.PairGrid(self.df)
        # 对 PairGrid 对象应用 map_lower 方法，使用 plt.scatter 函数
        g.map_lower(plt.scatter)

        # 遍历下三角矩阵的索引，返回对角线以下的索引
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            # 获取 g.axes 中指定位置的子图对象
            ax = g.axes[i, j]
            # 获取数据框 self.df 中指定变量的数据，并赋值给 x_in 和 y_in
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            # 获取子图中第一个集合对象的偏移坐标，并赋值给 x_out 和 y_out
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言 x_in 和 x_out 的数组内容相等
            npt.assert_array_equal(x_in, x_out)
            # 断言 y_in 和 y_out 的数组内容相等
            npt.assert_array_equal(y_in, y_out)

        # 遍历上三角矩阵的索引，返回对角线以上的索引
        for i, j in zip(*np.triu_indices_from(g.axes)):
            # 获取 g.axes 中指定位置的子图对象
            ax = g.axes[i, j]
            # 断言子图对象的集合长度为 0
            assert len(ax.collections) == 0

    # 定义一个测试方法，用于测试 PairGrid 对象的 map_upper 方法
    def test_map_upper(self):
        # 定义变量列表
        vars = ["x", "y", "z"]
        # 创建一个 PairGrid 对象 g，传入数据框 self.df
        g = ag.PairGrid(self.df)
        # 对 PairGrid 对象应用 map_upper 方法，使用 plt.scatter 函数
        g.map_upper(plt.scatter)

        # 遍历上三角矩阵的索引，返回对角线以上的索引
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            # 获取 g.axes 中指定位置的子图对象
            ax = g.axes[i, j]
            # 获取数据框 self.df 中指定变量的数据，并赋值给 x_in 和 y_in
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            # 获取子图中第一个集合对象的偏移坐标，并赋值给 x_out 和 y_out
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言 x_in 和 x_out 的数组内容相等
            npt.assert_array_equal(x_in, x_out)
            # 断言 y_in 和 y_out 的数组内容相等
            npt.assert_array_equal(y_in, y_out)

        # 遍历下三角矩阵的索引，返回对角线以下的索引
        for i, j in zip(*np.tril_indices_from(g.axes)):
            # 获取 g.axes 中指定位置的子图对象
            ax = g.axes[i, j]
            # 断言子图对象的集合长度为 0
            assert len(ax.collections) == 0

    # 定义一个测试方法，用于测试 PairGrid 对象的 map_lower 和 map_upper 方法结合使用的情况
    def test_map_mixed_funcsig(self):
        # 定义变量列表
        vars = ["x", "y", "z"]
        # 创建一个 PairGrid 对象 g，传入数据框 self.df 和变量列表 vars
        g = ag.PairGrid(self.df, vars=vars)
        # 对 PairGrid 对象应用 map_lower 方法，使用 scatterplot 函数
        g.map_lower(scatterplot)
        # 对 PairGrid 对象应用 map_upper 方法，使用 plt.scatter 函数
        g.map_upper(plt.scatter)

        # 遍历上三角矩阵的索引，返回对角线以上的索引
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            # 获取 g.axes 中指定位置的子图对象
            ax = g.axes[i, j]
            # 获取数据框 self.df 中指定变量的数据，并赋值给 x_in 和 y_in
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            # 获取子图中第一个集合对象的偏移坐标，并赋值给 x_out 和 y_out
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言 x_in 和 x_out 的数组内容相等
            npt.assert_array_equal(x_in, x_out)
            # 断言 y_in 和 y_out 的数组内容相等
            npt.assert_array_equal(y_in, y_out)

    # 定义一个测试方法，用于测试 PairGrid 对象的 map_diag 方法
    def test_map_diag(self):
        # 创建一个 PairGrid 对象 g，传入数据框 self.df
        g = ag.PairGrid(self.df)
        # 对 PairGrid 对象应用 map_diag 方法，使用 plt.hist 函数
        g.map_diag(plt.hist)

        # 遍历对角线变量和对应的轴对象
        for var, ax in zip(g.diag_vars, g.diag_axes):
            # 断言轴对象中补丁的数量为 10
            assert len(ax.patches) == 10
            # 断言补丁中第一个补丁的 x 坐标近似于数据框 self.df 中变量 var 的最小值
            assert pytest.approx(ax.patches[0].get_x()) == self.df[var].min()

        # 创建一个 PairGrid 对象 g，传入数据框 self.df 和色调变量 "a"
        g = ag.PairGrid(self.df, hue="a")
        # 对 PairGrid 对象应用 map_diag 方法，使用 plt.hist 函数
        g.map_diag(plt.hist)

        # 遍历对角线轴对象
        for ax in g.diag_axes:
            # 断言轴对象中补丁的数量为 30
            assert len(ax.patches) == 30

        # 创建一个 PairGrid 对象 g，传入数据框 self.df 和色调变量 "a"
        g = ag.PairGrid(self.df, hue="a")
        # 对 PairGrid 对象应用 map_diag 方法，使用 plt.hist 函数，并设置 histtype='step'
        g.map_diag(plt.hist, histtype='step')

        # 遍历对角线轴对象
        for ax in g.diag_axes:
            # 遍历轴对象中的每个补丁
            for ptch in ax.patches:
                # 断言补丁不填充
                assert not ptch.fill
    def test_map_diag_rectangular(self):
        # 定义 x 和 y 轴变量列表
        x_vars = ["x", "y"]
        y_vars = ["x", "z", "y"]
        
        # 创建 PairGrid 对象 g1，设置 x 和 y 轴变量
        g1 = ag.PairGrid(self.df, x_vars=x_vars, y_vars=y_vars)
        
        # 在对角线上的格子上绘制直方图
        g1.map_diag(plt.hist)
        
        # 在非对角线格子上绘制散点图
        g1.map_offdiag(plt.scatter)
        
        # 断言对角线变量集合应该是 x_vars 和 y_vars 的交集
        assert set(g1.diag_vars) == (set(x_vars) & set(y_vars))
        
        # 对每个对角线变量和其对应的轴进行断言
        for var, ax in zip(g1.diag_vars, g1.diag_axes):
            assert len(ax.patches) == 10
            assert pytest.approx(ax.patches[0].get_x()) == self.df[var].min()
        
        # 遍历 x_vars 和 y_vars 进行断言
        for j, x_var in enumerate(x_vars):
            for i, y_var in enumerate(y_vars):
                ax = g1.axes[i, j]
                if x_var == y_var:
                    diag_ax = g1.diag_axes[j]  # 因为 x_vars 比 y_vars 少
                    assert ax.bbox.bounds == diag_ax.bbox.bounds
                else:
                    x, y = ax.collections[0].get_offsets().T
                    assert_array_equal(x, self.df[x_var])
                    assert_array_equal(y, self.df[y_var])
        
        # 创建 PairGrid 对象 g2，设置 x 和 y 轴变量，同时设置色调为 "a"
        g2 = ag.PairGrid(self.df, x_vars=x_vars, y_vars=y_vars, hue="a")
        g2.map_diag(plt.hist)
        g2.map_offdiag(plt.scatter)
        
        # 断言对角线变量集合应该是 x_vars 和 y_vars 的交集
        assert set(g2.diag_vars) == (set(x_vars) & set(y_vars))
        
        # 对每个对角线轴进行断言
        for ax in g2.diag_axes:
            assert len(ax.patches) == 30
        
        # 更新 x_vars 和 y_vars 变量列表
        x_vars = ["x", "y", "z"]
        y_vars = ["x", "z"]
        
        # 创建 PairGrid 对象 g3，设置 x 和 y 轴变量
        g3 = ag.PairGrid(self.df, x_vars=x_vars, y_vars=y_vars)
        g3.map_diag(plt.hist)
        g3.map_offdiag(plt.scatter)
        
        # 断言对角线变量集合应该是 x_vars 和 y_vars 的交集
        assert set(g3.diag_vars) == (set(x_vars) & set(y_vars))
        
        # 对每个对角线变量和其对应的轴进行断言
        for var, ax in zip(g3.diag_vars, g3.diag_axes):
            assert len(ax.patches) == 10
            assert pytest.approx(ax.patches[0].get_x()) == self.df[var].min()
        
        # 遍历 x_vars 和 y_vars 进行断言
        for j, x_var in enumerate(x_vars):
            for i, y_var in enumerate(y_vars):
                ax = g3.axes[i, j]
                if x_var == y_var:
                    diag_ax = g3.diag_axes[i]  # 因为 y_vars 比 x_vars 少
                    assert ax.bbox.bounds == diag_ax.bbox.bounds
                else:
                    x, y = ax.collections[0].get_offsets().T
                    assert_array_equal(x, self.df[x_var])
                    assert_array_equal(y, self.df[y_var])

    def test_map_diag_color(self):
        # 设置颜色为红色
        color = "red"
        
        # 创建 PairGrid 对象 g1，不设置 x 和 y 轴变量
        g1 = ag.PairGrid(self.df)
        
        # 在对角线格子上绘制直方图，并使用指定的颜色
        g1.map_diag(plt.hist, color=color)
        
        # 对每个对角线轴上的直方图图形进行颜色断言
        for ax in g1.diag_axes:
            for patch in ax.patches:
                assert_colors_equal(patch.get_facecolor(), color)
        
        # 创建 PairGrid 对象 g2，不设置 x 和 y 轴变量
        g2 = ag.PairGrid(self.df)
        
        # 在对角线格子上绘制核密度估计图，并使用指定的颜色
        g2.map_diag(kdeplot, color='red')
        
        # 对每个对角线轴上的核密度估计图形进行颜色断言
        for ax in g2.diag_axes:
            for line in ax.lines:
                assert_colors_equal(line.get_color(), color)
    # 定义测试方法，用于测试 PairGrid 对象的 map_diag 方法
    def test_map_diag_palette(self):

        # 设置调色板名称
        palette = "muted"
        # 使用指定调色板生成颜色列表，颜色数量等于数据框 self.df.a 的唯一值数量
        pal = color_palette(palette, n_colors=len(self.df.a.unique()))
        # 创建 PairGrid 对象 g，指定颜色映射的列为 "a"，使用给定调色板 palette
        g = ag.PairGrid(self.df, hue="a", palette=palette)
        # 对 PairGrid 对象 g 的对角线上的每个图形应用 kdeplot 函数
        g.map_diag(kdeplot)

        # 遍历每个对角线轴上的对象 ax
        for ax in g.diag_axes:
            # 反转顺序遍历 ax.lines 中的线条对象 line，并与颜色列表 pal 中的颜色进行断言比较
            for line, color in zip(ax.lines[::-1], pal):
                # 断言 line 的颜色与 color 相等
                assert_colors_equal(line.get_color(), color)

    # 定义测试方法，测试 PairGrid 对象的 map_diag 和 map_offdiag 方法
    def test_map_diag_and_offdiag(self):

        # 定义变量列表
        vars = ["x", "y", "z"]
        # 创建 PairGrid 对象 g，不指定 hue 列
        g = ag.PairGrid(self.df)
        # 对 PairGrid 对象 g 的非对角线图形应用 plt.scatter 函数
        g.map_offdiag(plt.scatter)
        # 对 PairGrid 对象 g 的对角线图形应用 plt.hist 函数
        g.map_diag(plt.hist)

        # 遍历每个对角线轴上的对象 ax
        for ax in g.diag_axes:
            # 断言 ax.patches 的数量为 10
            assert len(ax.patches) == 10

        # 遍历 g.axes 的上三角部分的索引对 (i, j)
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            # 获取 g.axes[i, j] 对应的轴对象 ax
            ax = g.axes[i, j]
            # 从数据框 self.df 中获取 vars[j] 和 vars[i] 列的数据
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            # 获取 ax.collections[0] 中的偏移坐标并分别赋给 x_out 和 y_out
            x_out, y_out = ax.collections[0].get_offsets().T
            # 使用 numpy.testing 模块断言 x_in 与 x_out 的数组相等
            npt.assert_array_equal(x_in, x_out)
            # 使用 numpy.testing 模块断言 y_in 与 y_out 的数组相等
            npt.assert_array_equal(y_in, y_out)

        # 遍历 g.axes 的下三角部分的索引对 (i, j)
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            # 获取 g.axes[i, j] 对应的轴对象 ax
            ax = g.axes[i, j]
            # 从数据框 self.df 中获取 vars[j] 和 vars[i] 列的数据
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            # 获取 ax.collections[0] 中的偏移坐标并分别赋给 x_out 和 y_out
            x_out, y_out = ax.collections[0].get_offsets().T
            # 使用 numpy.testing 模块断言 x_in 与 x_out 的数组相等
            npt.assert_array_equal(x_in, x_out)
            # 使用 numpy.testing 模块断言 y_in 与 y_out 的数组相等
            npt.assert_array_equal(y_in, y_out)

        # 遍历 g.axes 的对角线部分的索引对 (i, j)
        for i, j in zip(*np.diag_indices_from(g.axes)):
            # 获取 g.axes[i, j] 对应的轴对象 ax
            ax = g.axes[i, j]
            # 断言 ax.collections 的长度为 0

    # 定义测试方法，测试 PairGrid 对象的 diag_sharey 参数
    def test_diag_sharey(self):

        # 创建 PairGrid 对象 g，设置 diag_sharey 参数为 True
        g = ag.PairGrid(self.df, diag_sharey=True)
        # 对 PairGrid 对象 g 的对角线图形应用 kdeplot 函数
        g.map_diag(kdeplot)
        # 遍历 g.diag_axes 的除第一个以外的每个轴对象 ax
        for ax in g.diag_axes[1:]:
            # 断言 ax 的纵坐标限制与 g.diag_axes[0] 的纵坐标限制相等

    # 定义测试方法，测试 PairGrid 对象的 map_diag 方法，使用 matplotlib 的 hist 函数
    def test_map_diag_matplotlib(self):

        # 定义 bins 参数
        bins = 10
        # 创建 PairGrid 对象 g，不指定 hue 列
        g = ag.PairGrid(self.df)
        # 对 PairGrid 对象 g 的对角线图形应用 plt.hist 函数，设置 bins 参数
        g.map_diag(plt.hist, bins=bins)
        # 遍历每个对角线轴上的对象 ax
        for ax in g.diag_axes:
            # 断言 ax.patches 的数量为 bins

        # 获取数据框 self.df 中 "a" 列唯一值的数量
        levels = len(self.df["a"].unique())
        # 创建 PairGrid 对象 g，设置 hue 列为 "a"
        g = ag.PairGrid(self.df, hue="a")
        # 对 PairGrid 对象 g 的对角线图形应用 plt.hist 函数，设置 bins 参数
        g.map_diag(plt.hist, bins=bins)
        # 遍历每个对角线轴上的对象 ax
        for ax in g.diag_axes:
            # 断言 ax.patches 的数量为 bins * levels

    # 定义测试方法，测试 PairGrid 对象的 palette 参数
    def test_palette(self):

        # 调用 rcmod 模块的 set 函数

        # 创建 PairGrid 对象 g，设置 hue 列为 "a"
        g = ag.PairGrid(self.df, hue="a")
        # 断言 g.palette 与使用 color_palette 函数生成的颜色列表相等，颜色数量等于数据框 self.df.a 的唯一值数量

        # 创建 PairGrid 对象 g，设置 hue 列为 "b"
        g = ag.PairGrid(self.df, hue="b")
        # 断言 g.palette 与使用 color_palette 函数生成的 "husl" 调色板颜色列表相等，颜色数量等于数据框 self.df.b 的唯一值数量

        # 创建 PairGrid 对象 g，设置 hue 列为 "a"，palette 参数为 "Set2"
        g = ag.PairGrid(self.df, hue="a", palette="Set2")
        # 断言 g.palette 与使用 color_palette 函数生成的 "Set2" 调色板颜色列表相等，颜色数量等于数据框 self.df.a 的唯一值数量

        # 创建字典调色板 dict_pal 和列表调色板 list_pal
        dict_pal = dict(a="red", b="green", c="blue")
        list_pal = color_palette(["red", "green", "blue"])
        # 创建 PairGrid 对象 g，设置 hue 列为 "a"，palette 参数为 dict_pal
        g = ag.PairGrid(self.df, hue="a", palette=dict_pal)
        # 断言 g.palette 与列表调色板 list_pal 相等

        # 更新列表调色板 list_pal
        list_pal = color_palette(["blue", "red", "green"])
        # 创建 PairGrid 对象 g，设置 hue 列为 "a"，hue_order 参数为 ['c', 'a', 'b']，palette 参数为 dict_pal
        g = ag.PairGrid(self.df, hue="a", hue_order=list("cab"),
                        palette=dict_pal)
        # 断言 g.palette 与更新后的列表调色板 list_pal 相等
    # 定义测试方法，测试带有色调关键字参数的绘图功能
    def test_hue_kws(self):
        # 设置标记参数字典，指定不同类别的标记类型
        kws = dict(marker=["o", "s", "d", "+"])
        # 创建一个 PairGrid 对象 g，使用数据框 self.df，根据列 'a' 进行分组，应用 kws 中的标记类型
        g = ag.PairGrid(self.df, hue="a", hue_kws=kws)
        # 在 g 上应用 plot 函数
        g.map(plt.plot)

        # 验证每条线的标记类型与 kws 中指定的一致
        for line, marker in zip(g.axes[0, 0].lines, kws["marker"]):
            assert line.get_marker() == marker

        # 创建另一个 PairGrid 对象 g，使用数据框 self.df，根据列 'a' 进行分组，指定顺序为 'dcab'
        g = ag.PairGrid(self.df, hue="a", hue_kws=kws,
                        hue_order=list("dcab"))
        # 在 g 上应用 plot 函数
        g.map(plt.plot)

        # 再次验证每条线的标记类型与 kws 中指定的一致
        for line, marker in zip(g.axes[0, 0].lines, kws["marker"]):
            assert line.get_marker() == marker

    # 定义测试方法，测试色调顺序功能
    def test_hue_order(self):
        # 指定顺序列表为 'dcab'
        order = list("dcab")
        # 创建一个 PairGrid 对象 g，使用数据框 self.df，根据列 'a' 进行分组，指定顺序为 order
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        # 在 g 上应用 plot 函数
        g.map(plt.plot)

        # 遍历 g 中第二行第一列的每条线和顺序列表中的每个级别
        for line, level in zip(g.axes[1, 0].lines, order):
            # 获取每条线的数据点坐标
            x, y = line.get_xydata().T
            # 验证 x 数据点与 self.df 中符合条件的 'x' 列数据一致
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "x"])
            # 验证 y 数据点与 self.df 中符合条件的 'y' 列数据一致
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "y"])

        # 关闭所有图形窗口
        plt.close("all")

        # 创建另一个 PairGrid 对象 g，使用数据框 self.df，根据列 'a' 进行分组，指定顺序为 order
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        # 在 g 上应用 map_diag 函数
        g.map_diag(plt.plot)

        # 遍历 g 中第一行第一列的每条线和顺序列表中的每个级别
        for line, level in zip(g.axes[0, 0].lines, order):
            # 获取每条线的数据点坐标
            x, y = line.get_xydata().T
            # 验证 x 数据点与 self.df 中符合条件的 'x' 列数据一致
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "x"])
            # 验证 y 数据点与 self.df 中符合条件的 'x' 列数据一致（应为 'y' 列，原注释存在错误）
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "x"])

        # 关闭所有图形窗口
        plt.close("all")

        # 创建另一个 PairGrid 对象 g，使用数据框 self.df，根据列 'a' 进行分组，指定顺序为 order
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        # 在 g 上应用 map_lower 函数
        g.map_lower(plt.plot)

        # 遍历 g 中第二行第一列的每条线和顺序列表中的每个级别
        for line, level in zip(g.axes[1, 0].lines, order):
            # 获取每条线的数据点坐标
            x, y = line.get_xydata().T
            # 验证 x 数据点与 self.df 中符合条件的 'x' 列数据一致
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "x"])
            # 验证 y 数据点与 self.df 中符合条件的 'y' 列数据一致
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "y"])

        # 关闭所有图形窗口
        plt.close("all")

        # 创建另一个 PairGrid 对象 g，使用数据框 self.df，根据列 'a' 进行分组，指定顺序为 order
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        # 在 g 上应用 map_upper 函数
        g.map_upper(plt.plot)

        # 遍历 g 中第一行第二列的每条线和顺序列表中的每个级别
        for line, level in zip(g.axes[0, 1].lines, order):
            # 获取每条线的数据点坐标
            x, y = line.get_xydata().T
            # 验证 x 数据点与 self.df 中符合条件的 'y' 列数据一致
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "y"])
            # 验证 y 数据点与 self.df 中符合条件的 'x' 列数据一致
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "x"])

        # 关闭所有图形窗口
        plt.close("all")
    # 测试函数：测试在缺少级别的情况下的色调顺序
    def test_hue_order_missing_level(self):
        
        # 定义色调顺序
        order = list("dcaeb")
        # 创建一个 PairGrid 对象，使用数据框 self.df，并指定色调列为 "a"，并按照指定的顺序 order 进行排列
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        # 在 PairGrid 上映射 plt.plot 函数
        g.map(plt.plot)
        
        # 遍历第二行第一列的子图中的线条和色调顺序中的级别
        for line, level in zip(g.axes[1, 0].lines, order):
            # 获取线条的数据并转置
            x, y = line.get_xydata().T
            # 使用断言检查 x 数据与 self.df 中 a 列等于 level 的行的 "x" 列是否相等
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "x"])
            # 使用断言检查 y 数据与 self.df 中 a 列等于 level 的行的 "y" 列是否相等
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "y"])
        
        # 关闭所有图形窗口
        plt.close("all")
        
        # 重新创建 PairGrid 对象，使用相同的数据框和设置，但是这次映射是对角线上的 plt.plot 函数
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        g.map_diag(plt.plot)
        
        # 遍历第一行第一列的子图中的线条和色调顺序中的级别
        for line, level in zip(g.axes[0, 0].lines, order):
            # 获取线条的数据并转置
            x, y = line.get_xydata().T
            # 使用断言检查 x 数据与 self.df 中 a 列等于 level 的行的 "x" 列是否相等
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "x"])
            # 使用断言检查 y 数据与 self.df 中 a 列等于 level 的行的 "x" 列是否相等（应为 "y" 列）
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "x"])
        
        # 关闭所有图形窗口
        plt.close("all")
        
        # 重新创建 PairGrid 对象，使用相同的数据框和设置，但这次映射是下三角区域的 plt.plot 函数
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        g.map_lower(plt.plot)
        
        # 遍历第二行第一列的子图中的线条和色调顺序中的级别
        for line, level in zip(g.axes[1, 0].lines, order):
            # 获取线条的数据并转置
            x, y = line.get_xydata().T
            # 使用断言检查 x 数据与 self.df 中 a 列等于 level 的行的 "x" 列是否相等
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "x"])
            # 使用断言检查 y 数据与 self.df 中 a 列等于 level 的行的 "y" 列是否相等
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "y"])
        
        # 关闭所有图形窗口
        plt.close("all")
        
        # 重新创建 PairGrid 对象，使用相同的数据框和设置，但这次映射是上三角区域的 plt.plot 函数
        g = ag.PairGrid(self.df, hue="a", hue_order=order)
        g.map_upper(plt.plot)
        
        # 遍历第一行第二列的子图中的线条和色调顺序中的级别
        for line, level in zip(g.axes[0, 1].lines, order):
            # 获取线条的数据并转置
            x, y = line.get_xydata().T
            # 使用断言检查 x 数据与 self.df 中 a 列等于 level 的行的 "y" 列是否相等
            npt.assert_array_equal(x, self.df.loc[self.df.a == level, "y"])
            # 使用断言检查 y 数据与 self.df 中 a 列等于 level 的行的 "x" 列是否相等（应为 "y" 列）
            npt.assert_array_equal(y, self.df.loc[self.df.a == level, "x"])
        
        # 关闭所有图形窗口
        plt.close("all")
    
    # 测试函数：测试非默认索引
    def test_nondefault_index(self):
        
        # 复制数据框 self.df，并将其索引设置为 "b" 列
        df = self.df.copy().set_index("b")
        
        # 定义要绘制的变量列表
        plot_vars = ["x", "y", "z"]
        
        # 创建一个 PairGrid 对象，使用新的索引数据框 df，并映射 plt.scatter 函数
        g1 = ag.PairGrid(df)
        g1.map(plt.scatter)
        
        # 遍历 PairGrid 对象的子图及其索引
        for i, axes_i in enumerate(g1.axes):
            for j, ax in enumerate(axes_i):
                # 获取 self.df 中相应列的数据作为输入数据
                x_in = self.df[plot_vars[j]]
                y_in = self.df[plot_vars[i]]
                # 获取子图中散点图的偏移数据，并分别检查是否与输入数据相等
                x_out, y_out = ax.collections[0].get_offsets().T
                npt.assert_array_equal(x_in, x_out)
                npt.assert_array_equal(y_in, y_out)
        
        # 创建一个带有色调的 PairGrid 对象，使用新的索引数据框 df，并映射 plt.scatter 函数
        g2 = ag.PairGrid(df, hue="a")
        g2.map(plt.scatter)
        
        # 遍历 PairGrid 对象的子图及其索引
        for i, axes_i in enumerate(g2.axes):
            for j, ax in enumerate(axes_i):
                # 获取 self.df 中相应列的数据作为输入数据
                x_in = self.df[plot_vars[j]]
                y_in = self.df[plot_vars[i]]
                # 遍历数据框中唯一的色调水平，并检查散点图的偏移数据是否与输入数据相等
                for k, k_level in enumerate(self.df.a.unique()):
                    x_in_k = x_in[self.df.a == k_level]
                    y_in_k = y_in[self.df.a == k_level]
                    x_out, y_out = ax.collections[k].get_offsets().T
                    npt.assert_array_equal(x_in_k, x_out)
                    npt.assert_array_equal(y_in_k, y_out)
    @pytest.mark.parametrize("func", [scatterplot, plt.scatter])
    # 使用 pytest 的参数化功能，对 func 进行测试，分别使用 scatterplot 和 plt.scatter 函数
    def test_dropna(self, func):
        # 复制 self.df 数据框，以便在测试中进行修改而不影响原始数据
        df = self.df.copy()
        # 将前 n_null 行的 "x" 列设置为 NaN
        n_null = 20
        df.loc[np.arange(n_null), "x"] = np.nan

        plot_vars = ["x", "y", "z"]

        # 创建一个 PairGrid 对象 g1，用于绘制多变量图，dropna=True 表示忽略 NaN 值
        g1 = ag.PairGrid(df, vars=plot_vars, dropna=True)
        # 对 g1 中的每个子图应用 func 函数（scatterplot 或 plt.scatter）
        g1.map(func)

        # 遍历 g1 中的每个子图
        for i, axes_i in enumerate(g1.axes):
            for j, ax in enumerate(axes_i):
                # 获取原始数据中的 x_in 和 y_in
                x_in = df[plot_vars[j]]
                y_in = df[plot_vars[i]]
                # 获取绘图中散点的 x_out 和 y_out 坐标
                x_out, y_out = ax.collections[0].get_offsets().T

                # 计算有效数据点的数量
                n_valid = (x_in * y_in).notnull().sum()

                # 断言绘图中的数据点数量与有效数据点数量相等
                assert n_valid == len(x_out)
                assert n_valid == len(y_out)

        # 对 g1 中的对角线位置应用 histplot 函数
        g1.map_diag(histplot)
        # 遍历 g1 的对角线子图
        for i, ax in enumerate(g1.diag_axes):
            var = plot_vars[i]
            # 计算每个子图中的直方图柱数目
            count = sum(p.get_height() for p in ax.patches)
            # 断言直方图柱的数量与 var 列中非 NaN 值的数量相等
            assert count == df[var].notna().sum()

    def test_histplot_legend(self):

        # Tests _extract_legend_handles
        # 创建一个 PairGrid 对象 g，用于绘制 x 和 y 变量的图，并根据 "a" 列进行着色
        g = ag.PairGrid(self.df, vars=["x", "y"], hue="a")
        # 对 g 中的非对角线位置应用 histplot 函数
        g.map_offdiag(histplot)
        # 添加图例到 g 中
        g.add_legend()

        # 断言图例中的句柄数量与 self.df["a"] 中唯一值的数量相等
        assert len(get_legend_handles(g._legend)) == len(self.df["a"].unique())

    def test_pairplot(self):

        vars = ["x", "y", "z"]
        # 创建一个 pairplot 对象 g，用于绘制 self.df 的两两变量关系图
        g = ag.pairplot(self.df)

        # 遍历 g 的对角线子图
        for ax in g.diag_axes:
            # 断言每个对角线子图中的直方图柱数大于 1
            assert len(ax.patches) > 1

        # 遍历 g 中上三角位置的子图
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            ax = g.axes[i, j]
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言原始数据 x_in 和 y_in 与绘图中的 x_out 和 y_out 相等
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)

        # 遍历 g 中下三角位置的子图
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            ax = g.axes[i, j]
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言原始数据 x_in 和 y_in 与绘图中的 x_out 和 y_out 相等
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)

        # 遍历 g 的对角线子图
        for i, j in zip(*np.diag_indices_from(g.axes)):
            ax = g.axes[i, j]
            # 断言每个对角线子图中没有集合（collections）
            assert len(ax.collections) == 0

        # 创建一个带有 hue="a" 的 pairplot 对象 g
        g = ag.pairplot(self.df, hue="a")
        n = len(self.df.a.unique())

        # 遍历 g 的对角线子图
        for ax in g.diag_axes:
            # 断言每个对角线子图中集合（collections）的数量等于唯一值数量 n
            assert len(ax.collections) == n
    # 定义一个测试方法，用于测试带有回归线的对角线直方图的成对绘图
    def test_pairplot_reg(self):

        vars = ["x", "y", "z"]
        # 使用带有对角线直方图和回归线的成对绘图函数来创建图形对象 g
        g = ag.pairplot(self.df, diag_kind="hist", kind="reg")

        # 验证对角线上的每个子图中都有直方图柱状块
        for ax in g.diag_axes:
            assert len(ax.patches)

        # 遍历上三角矩阵中的每对坐标 i, j
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            ax = g.axes[i, j]
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言输入数据与绘图对象中的数据点一致
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)
            # 断言每个轴上有且仅有一条回归线和两个集合
            assert len(ax.lines) == 1
            assert len(ax.collections) == 2

        # 遍历下三角矩阵中的每对坐标 i, j
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            ax = g.axes[i, j]
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言输入数据与绘图对象中的数据点一致
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)
            # 断言每个轴上有且仅有一条回归线和两个集合
            assert len(ax.lines) == 1
            assert len(ax.collections) == 2

        # 遍历对角线上的每对坐标 i, j
        for i, j in zip(*np.diag_indices_from(g.axes)):
            ax = g.axes[i, j]
            # 断言每个轴上没有集合
            assert len(ax.collections) == 0

    # 定义一个测试方法，用于测试带有色调的回归线的成对绘图
    def test_pairplot_reg_hue(self):

        markers = ["o", "s", "d"]
        # 使用带有色调和自定义标记的成对绘图函数来创建图形对象 g
        g = ag.pairplot(self.df, kind="reg", hue="a", markers=markers)

        # 获取最后一个行中第一个列的轴对象 ax
        ax = g.axes[-1, 0]
        # 获取第一个集合 c1 和第三个集合 c2
        c1 = ax.collections[0]
        c2 = ax.collections[2]

        # 断言两个集合的面部颜色不相等
        assert not np.array_equal(c1.get_facecolor(), c2.get_facecolor())
        # 断言两个集合的顶点坐标不相等
        assert not np.array_equal(
            c1.get_paths()[0].vertices, c2.get_paths()[0].vertices,
        )

    # 定义一个测试方法，用于测试带有核密度估计的成对绘图
    def test_pairplot_diag_kde(self):

        vars = ["x", "y", "z"]
        # 使用带有核密度估计的成对绘图函数来创建图形对象 g
        g = ag.pairplot(self.df, diag_kind="kde")

        # 验证对角线上的每个子图中都只有一个集合
        for ax in g.diag_axes:
            assert len(ax.collections) == 1

        # 遍历上三角矩阵中的每对坐标 i, j
        for i, j in zip(*np.triu_indices_from(g.axes, 1)):
            ax = g.axes[i, j]
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言输入数据与绘图对象中的数据点一致
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)

        # 遍历下三角矩阵中的每对坐标 i, j
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            ax = g.axes[i, j]
            x_in = self.df[vars[j]]
            y_in = self.df[vars[i]]
            x_out, y_out = ax.collections[0].get_offsets().T
            # 断言输入数据与绘图对象中的数据点一致
            npt.assert_array_equal(x_in, x_out)
            npt.assert_array_equal(y_in, y_out)

        # 遍历对角线上的每对坐标 i, j
        for i, j in zip(*np.diag_indices_from(g.axes)):
            ax = g.axes[i, j]
            # 断言每个轴上没有集合
            assert len(ax.collections) == 0

    # 定义一个测试方法，用于测试带有核密度估计的成对绘图
    def test_pairplot_kde(self):

        f, ax1 = plt.subplots()
        # 使用单独的核密度估计绘图函数来创建图形对象 ax1
        kdeplot(data=self.df, x="x", y="y", ax=ax1)

        # 使用带有核密度估计的成对绘图函数来创建图形对象 g
        g = ag.pairplot(self.df, kind="kde")
        # 获取第二行第一列的轴对象 ax2
        ax2 = g.axes[1, 0]

        # 断言图形对象 ax1 和 ax2 在视觉上相等，但不比较标签
        assert_plots_equal(ax1, ax2, labels=False)
    def test_pairplot_hist(self):
        # 创建一个新的图形窗口和一个轴对象
        f, ax1 = plt.subplots()
        # 绘制数据集 self.df 的直方图，x 轴为 "x"，y 轴为 "y"
        histplot(data=self.df, x="x", y="y", ax=ax1)

        # 使用 ag.pairplot 绘制 self.df 的变量关系图，类型为直方图
        g = ag.pairplot(self.df, kind="hist")
        # 获取 g 对象的第二行第一列的轴对象
        ax2 = g.axes[1, 0]

        # 断言 ax1 和 ax2 的绘图内容相等，不包括标签
        assert_plots_equal(ax1, ax2, labels=False)

    @pytest.mark.skipif(_version_predates(mpl, "3.7.0"), reason="Matplotlib bug")
    def test_pairplot_markers(self):
        # 定义变量列表和标记样式列表
        vars = ["x", "y", "z"]
        markers = ["o", "X", "s"]
        # 使用 ag.pairplot 绘制 self.df 的变量关系图，根据 "a" 列分组，并使用指定标记样式
        g = ag.pairplot(self.df, hue="a", vars=vars, markers=markers)
        # 获取图例的第一个和第二个标记样式
        m1 = get_legend_handles(g._legend)[0].get_marker()
        m2 = get_legend_handles(g._legend)[1].get_marker()
        # 断言两个标记样式不相同
        assert m1 != m2

        # 检查是否会发出 UserWarning 警告
        with pytest.warns(UserWarning):
            # 使用 ag.pairplot 绘制 self.df 的变量关系图，根据 "a" 列分组，但只使用前两种标记样式
            g = ag.pairplot(self.df, hue="a", vars=vars, markers=markers[:-2])

    def test_pairplot_column_multiindex(self):
        # 创建具有多级索引的列对象
        cols = pd.MultiIndex.from_arrays([["x", "y"], [1, 2]])
        # 从 self.df 中选择 "x" 和 "y" 列，并设置其轴标签为 cols
        df = self.df[["x", "y"]].set_axis(cols, axis=1)
        # 使用 ag.pairplot 绘制 DataFrame df 的变量关系图
        g = ag.pairplot(df)
        # 断言 g 对象的对角线变量等于 cols 的列表形式
        assert g.diag_vars == list(cols)

    def test_corner_despine(self):
        # 创建一个带有角落显示和不去掉边框的 PairGrid 对象
        g = ag.PairGrid(self.df, corner=True, despine=False)
        # 绘制对角线的直方图
        g.map_diag(histplot)
        # 断言 g 的第一个图的顶部边框是可见的
        assert g.axes[0, 0].spines["top"].get_visible()

    def test_corner_set(self):
        # 创建一个带有角落显示和不去掉边框的 PairGrid 对象
        g = ag.PairGrid(self.df, corner=True, despine=False)
        # 设置 g 对象的 x 轴限制为 (0, 10)
        g.set(xlim=(0, 10))
        # 断言 g 对象最后一行第一列的 x 轴限制为 (0, 10)
        assert g.axes[-1, 0].get_xlim() == (0, 10)

    def test_legend(self):
        # 使用 ag.pairplot 绘制 self.df 的变量关系图，根据 "a" 列分组
        g1 = ag.pairplot(self.df, hue="a")
        # 断言 g1 的图例对象是 Matplotlib 的 Legend 类的实例
        assert isinstance(g1.legend, mpl.legend.Legend)

        # 使用 ag.pairplot 绘制 self.df 的变量关系图，不根据任何列分组
        g2 = ag.pairplot(self.df)
        # 断言 g2 的图例对象为 None
        assert g2.legend is None

    def test_tick_params(self):
        # 创建一个 PairGrid 对象
        g = ag.PairGrid(self.df)
        # 设置颜色为 "red"，间距为 3
        color = "red"
        pad = 3
        # 设置 g 对象的刻度参数：间距为 pad，颜色为 color
        g.tick_params(pad=pad, color=color)
        # 遍历 g 对象所有的轴对象
        for ax in g.axes.flat:
            # 遍历每个轴对象的 x 轴和 y 轴主刻度
            for axis in ["xaxis", "yaxis"]:
                for tick in getattr(ax, axis).get_major_ticks():
                    # 断言每个刻度的第一条线和第二条线的颜色与 color 相同，间距为 pad
                    assert mpl.colors.same_color(tick.tick1line.get_color(), color)
                    assert mpl.colors.same_color(tick.tick2line.get_color(), color)
                    assert tick.get_pad() == pad

    @pytest.mark.skipif(
        condition=not hasattr(pd.api, "interchange"),
        reason="Tests behavior assuming support for dataframe interchange"
    )
    def test_data_interchange(self, mock_long_df, long_df):
        # 使用 ag.PairGrid 绘制 mock_long_df 的变量关系图，选定变量为 ["x", "y", "z"]，根据 "a" 列分组
        g = ag.PairGrid(mock_long_df, vars=["x", "y", "z"], hue="a")
        # 对 g 对象的每个子图应用 scatterplot 函数
        g.map(scatterplot)
        # 断言 g 对象的 axes 形状为 (3, 3)
        assert g.axes.shape == (3, 3)
        # 遍历 g 对象所有的子图
        for ax in g.axes.flat:
            # 获取每个子图的集合对象的偏移量
            pts = ax.collections[0].get_offsets()
            # 断言偏移量的长度等于 long_df 的长度
            assert len(pts) == len(long_df)
class TestJointGrid:

    # 创建一个随机数生成器，用于生成测试数据
    rs = np.random.RandomState(sum(map(ord, "JointGrid")))

    # 生成随机的 x 和 y 数据，各有 100 个样本
    x = rs.randn(100)
    y = rs.randn(100)

    # 创建一个包含 NaN 值的 x_na 数组
    x_na = x.copy()
    x_na[10] = np.nan
    x_na[20] = np.nan

    # 将生成的数据存储在 DataFrame 中
    data = pd.DataFrame(dict(x=x, y=y, x_na=x_na))

    # 测试从列表创建 JointGrid 对象
    def test_margin_grid_from_lists(self):
        g = ag.JointGrid(x=self.x.tolist(), y=self.y.tolist())
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)

    # 测试从数组创建 JointGrid 对象
    def test_margin_grid_from_arrays(self):
        g = ag.JointGrid(x=self.x, y=self.y)
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)

    # 测试从 Series 创建 JointGrid 对象
    def test_margin_grid_from_series(self):
        g = ag.JointGrid(x=self.data.x, y=self.data.y)
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)

    # 测试从 DataFrame 创建 JointGrid 对象
    def test_margin_grid_from_dataframe(self):
        g = ag.JointGrid(x="x", y="y", data=self.data)
        npt.assert_array_equal(g.x, self.x)
        npt.assert_array_equal(g.y, self.y)

    # 测试从 DataFrame 创建 JointGrid 对象，并使用不存在的列名
    def test_margin_grid_from_dataframe_bad_variable(self):
        with pytest.raises(ValueError):
            ag.JointGrid(x="x", y="bad_column", data=self.data)

    # 测试设置轴标签
    def test_margin_grid_axis_labels(self):
        g = ag.JointGrid(x="x", y="y", data=self.data)

        # 获取并断言轴标签的初始值
        xlabel, ylabel = g.ax_joint.get_xlabel(), g.ax_joint.get_ylabel()
        assert xlabel == "x"
        assert ylabel == "y"

        # 设置新的轴标签，并再次断言
        g.set_axis_labels("x variable", "y variable")
        xlabel, ylabel = g.ax_joint.get_xlabel(), g.ax_joint.get_ylabel()
        assert xlabel == "x variable"
        assert ylabel == "y variable"

    # 测试 dropna 参数的影响
    def test_dropna(self):
        # 测试 dropna=False 的情况
        g = ag.JointGrid(x="x_na", y="y", data=self.data, dropna=False)
        assert len(g.x) == len(self.x_na)

        # 测试 dropna=True 的情况
        g = ag.JointGrid(x="x_na", y="y", data=self.data, dropna=True)
        assert len(g.x) == pd.notnull(self.x_na).sum()

    # 测试设置坐标轴限制
    def test_axlims(self):
        lim = (-3, 3)
        g = ag.JointGrid(x="x", y="y", data=self.data, xlim=lim, ylim=lim)

        # 断言主图的坐标轴限制
        assert g.ax_joint.get_xlim() == lim
        assert g.ax_joint.get_ylim() == lim

        # 断言边缘图的坐标轴限制
        assert g.ax_marg_x.get_xlim() == lim
        assert g.ax_marg_y.get_ylim() == lim

    # 测试边缘图的刻度显示设置
    def test_marginal_ticks(self):
        # 测试 marginal_ticks=False 的情况
        g = ag.JointGrid(marginal_ticks=False)
        assert not sum(t.get_visible() for t in g.ax_marg_x.get_yticklabels())
        assert not sum(t.get_visible() for t in g.ax_marg_y.get_xticklabels())

        # 测试 marginal_ticks=True 的情况
        g = ag.JointGrid(marginal_ticks=True)
        assert sum(t.get_visible() for t in g.ax_marg_x.get_yticklabels())
        assert sum(t.get_visible() for t in g.ax_marg_y.get_xticklabels())

    # 测试二元绘图函数
    def test_bivariate_plot(self):
        g = ag.JointGrid(x="x", y="y", data=self.data)

        # 绘制二元关系图并断言结果
        g.plot_joint(plt.plot)

        x, y = g.ax_joint.lines[0].get_xydata().T
        npt.assert_array_equal(x, self.x)
        npt.assert_array_equal(y, self.y)
    # 定义一个测试方法，用于测试单变量绘图的联合网格
    def test_univariate_plot(self):

        # 创建一个联合网格对象 g，设置 x 和 y 轴为相同数据列，数据来源为 self.data
        g = ag.JointGrid(x="x", y="x", data=self.data)
        # 使用 kdeplot 绘制边缘分布图
        g.plot_marginals(kdeplot)

        # 获取 g.ax_marg_x 中第一条线的 x 和 y 数据，并转置
        _, y1 = g.ax_marg_x.lines[0].get_xydata().T
        # 获取 g.ax_marg_y 中第一条线的 y 和 x 数据，并转置
        y2, _ = g.ax_marg_y.lines[0].get_xydata().T
        # 断言 y1 和 y2 数组相等
        npt.assert_array_equal(y1, y2)

    # 定义一个测试方法，用于测试使用 distplot 绘制单变量绘图的联合网格
    def test_univariate_plot_distplot(self):

        # 设置直方图的 bin 数量为 10
        bins = 10
        # 创建一个联合网格对象 g，设置 x 和 y 轴为相同数据列，数据来源为 self.data
        g = ag.JointGrid(x="x", y="x", data=self.data)
        # 使用 distplot 绘制边缘分布图，预期会产生 UserWarning 警告
        with pytest.warns(UserWarning):
            g.plot_marginals(distplot, bins=bins)
        # 断言 g.ax_marg_x 中的 patches 数量等于 bins
        assert len(g.ax_marg_x.patches) == bins
        # 断言 g.ax_marg_y 中的 patches 数量等于 bins
        assert len(g.ax_marg_y.patches) == bins
        # 遍历 g.ax_marg_x 和 g.ax_marg_y 的 patches，断言它们的高度和宽度相等
        for x, y in zip(g.ax_marg_x.patches, g.ax_marg_y.patches):
            assert x.get_height() == y.get_width()

    # 定义一个测试方法，用于测试使用 matplotlib 的 hist 绘制单变量绘图的联合网格
    def test_univariate_plot_matplotlib(self):

        # 设置直方图的 bin 数量为 10
        bins = 10
        # 创建一个联合网格对象 g，设置 x 和 y 轴为相同数据列，数据来源为 self.data
        g = ag.JointGrid(x="x", y="x", data=self.data)
        # 使用 plt.hist 绘制边缘分布图
        g.plot_marginals(plt.hist, bins=bins)
        # 断言 g.ax_marg_x 中的 patches 数量等于 bins
        assert len(g.ax_marg_x.patches) == bins
        # 断言 g.ax_marg_y 中的 patches 数量等于 bins
        assert len(g.ax_marg_y.patches) == bins

    # 定义一个测试方法，用于测试绘制整体联合网格图
    def test_plot(self):

        # 创建一个联合网格对象 g，设置 x 和 y 轴为相同数据列，数据来源为 self.data
        g = ag.JointGrid(x="x", y="x", data=self.data)
        # 使用 plt.plot 和 kdeplot 绘制整体联合网格图
        g.plot(plt.plot, kdeplot)

        # 获取 g.ax_joint 中第一条线的 x 和 y 数据，并转置
        x, y = g.ax_joint.lines[0].get_xydata().T
        # 断言 x 数据等于 self.x
        npt.assert_array_equal(x, self.x)
        # 断言 y 数据等于 self.x
        npt.assert_array_equal(y, self.x)

        # 获取 g.ax_marg_x 中第一条线的 x 和 y 数据，并转置
        _, y1 = g.ax_marg_x.lines[0].get_xydata().T
        # 获取 g.ax_marg_y 中第一条线的 y 和 x 数据，并转置
        y2, _ = g.ax_marg_y.lines[0].get_xydata().T
        # 断言 y1 和 y2 数组相等
        npt.assert_array_equal(y1, y2)

    # 定义一个测试方法，用于测试设置空间间隔为 0 的联合网格图
    def test_space(self):

        # 创建一个联合网格对象 g，设置 x 和 y 轴为不同数据列，数据来源为 self.data，空间间隔为 0
        g = ag.JointGrid(x="x", y="y", data=self.data, space=0)

        # 获取 g.ax_joint 的边界框尺寸信息
        joint_bounds = g.ax_joint.bbox.bounds
        # 获取 g.ax_marg_x 的边界框尺寸信息
        marg_x_bounds = g.ax_marg_x.bbox.bounds
        # 获取 g.ax_marg_y 的边界框尺寸信息
        marg_y_bounds = g.ax_marg_y.bbox.bounds

        # 断言 g.ax_joint 的宽度等于 g.ax_marg_x 的宽度
        assert joint_bounds[2] == marg_x_bounds[2]
        # 断言 g.ax_joint 的高度等于 g.ax_marg_y 的高度
        assert joint_bounds[3] == marg_y_bounds[3]

    # 使用参数化测试，测试带有色调参数的联合网格绘图
    @pytest.mark.parametrize(
        "as_vector", [True, False],
    )
    def test_hue(self, long_df, as_vector):

        # 如果 as_vector 为 True，则设置 data 为 None，否则使用 long_df 的数据列作为 x、y 和 hue
        if as_vector:
            data = None
            x, y, hue = long_df["x"], long_df["y"], long_df["a"]
        else:
            data = long_df
            x, y, hue = "x", "y", "a"

        # 创建一个带有色调参数的联合网格对象 g
        g = ag.JointGrid(data=data, x=x, y=y, hue=hue)
        # 使用 scatterplot 绘制联合分布图
        g.plot_joint(scatterplot)
        # 使用 histplot 绘制边缘分布图
        g.plot_marginals(histplot)

        # 创建一个新的联合网格对象 g2
        g2 = ag.JointGrid()
        # 使用 scatterplot 在 g2.ax_joint 上绘制联合分布图
        scatterplot(data=long_df, x=x, y=y, hue=hue, ax=g2.ax_joint)
        # 使用 histplot 在 g2.ax_marg_x 上绘制 x 轴的边缘分布图
        histplot(data=long_df, x=x, hue=hue, ax=g2.ax_marg_x)
        # 使用 histplot 在 g2.ax_marg_y 上绘制 y 轴的边缘分布图
        histplot(data=long_df, y=y, hue=hue, ax=g2.ax_marg_y)

        # 断言 g.ax_joint 和 g2.ax_joint 的绘图内容相等
        assert_plots_equal(g.ax_joint, g2.ax_joint)
        # 断言 g.ax_marg_x 和 g2.ax_marg_x 的绘图内容相等，忽略标签
        assert_plots_equal(g.ax_marg_x, g2.ax_marg_x, labels=False)
        # 断言 g.ax_marg_y 和 g2.ax_marg_y 的绘图内容相等，忽略标签
        assert_plots_equal(g.ax_marg_y, g2.ax_marg_y, labels=False)
    # 定义一个测试方法，用于测试 JointGrid 类的 refline 方法
    def test_refline(self):
        # 创建一个 JointGrid 对象 g，以数据 self.data 的 x 和 y 列作为坐标轴
        g = ag.JointGrid(x="x", y="y", data=self.data)
        # 绘制 JointGrid 的散点图和直方图
        g.plot(scatterplot, histplot)
        # 添加参考线到图形中
        g.refline()
        # 断言主轴和边缘轴上没有添加的线条
        assert not g.ax_joint.lines and not g.ax_marg_x.lines and not g.ax_marg_y.lines

        # 设置参考线的 x 和 y 值为 0.5
        refx = refy = 0.5
        # 创建水平参考线和垂直参考线的数据数组
        hline = np.array([[0, refy], [1, refy]])
        vline = np.array([[refx, 0], [refx, 1]])
        # 添加不显示联合轴和边缘轴上的参考线
        g.refline(x=refx, y=refy, joint=False, marginal=False)
        # 再次断言主轴和边缘轴上没有添加的线条
        assert not g.ax_joint.lines and not g.ax_marg_x.lines and not g.ax_marg_y.lines

        # 添加默认样式的参考线到图形中
        g.refline(x=refx, y=refy)
        # 断言第一个主轴线条的颜色为灰色
        assert g.ax_joint.lines[0].get_color() == '.5'
        # 断言第一个主轴线条的线型为虚线
        assert g.ax_joint.lines[0].get_linestyle() == '--'
        # 断言主轴上共有两条线条
        assert len(g.ax_joint.lines) == 2
        # 断言边缘轴上各有一条线条
        assert len(g.ax_marg_x.lines) == 1
        assert len(g.ax_marg_y.lines) == 1
        # 断言主轴上的第一条线条数据与预期的垂直线条数据相等
        npt.assert_array_equal(g.ax_joint.lines[0].get_xydata(), vline)
        # 断言主轴上的第二条线条数据与预期的水平线条数据相等
        npt.assert_array_equal(g.ax_joint.lines[1].get_xydata(), hline)
        # 断言边缘轴上的垂直线条数据与预期的垂直线条数据相等
        npt.assert_array_equal(g.ax_marg_x.lines[0].get_xydata(), vline)
        # 断言边缘轴上的水平线条数据与预期的水平线条数据相等
        npt.assert_array_equal(g.ax_marg_y.lines[0].get_xydata(), hline)

        # 设置参考线的颜色为红色，线型为实线
        color, linestyle = 'red', '-'
        # 添加边缘轴上不显示的红色实线参考线
        g.refline(x=refx, marginal=False, color=color, linestyle=linestyle)
        # 断言最后一条主轴线条的数据与预期的垂直线条数据相等
        npt.assert_array_equal(g.ax_joint.lines[-1].get_xydata(), vline)
        # 断言最后一条主轴线条的颜色为红色
        assert g.ax_joint.lines[-1].get_color() == color
        # 断言最后一条主轴线条的线型为实线
        assert g.ax_joint.lines[-1].get_linestyle() == linestyle
        # 断言边缘轴上的线条数量与边缘轴上的线条数量相等
        assert len(g.ax_marg_x.lines) == len(g.ax_marg_y.lines)

        # 添加联合轴上不显示的垂直参考线
        g.refline(x=refx, joint=False)
        # 断言边缘轴上的最后一条垂直线条的数据与预期的垂直线条数据相等
        npt.assert_array_equal(g.ax_marg_x.lines[-1].get_xydata(), vline)
        # 断言边缘轴上的线条数量比边缘轴上的线条数量多一条
        assert len(g.ax_marg_x.lines) == len(g.ax_marg_y.lines) + 1

        # 添加边缘轴上不显示的水平参考线
        g.refline(y=refy, joint=False)
        # 断言边缘轴上的最后一条水平线条的数据与预期的水平线条数据相等
        npt.assert_array_equal(g.ax_marg_y.lines[-1].get_xydata(), hline)
        # 断言边缘轴上的线条数量与边缘轴上的线条数量相等
        assert len(g.ax_marg_x.lines) == len(g.ax_marg_y.lines)

        # 添加主轴上不显示的水平参考线
        g.refline(y=refy, marginal=False)
        # 断言最后一条主轴线条的数据与预期的水平线条数据相等
        npt.assert_array_equal(g.ax_joint.lines[-1].get_xydata(), hline)
        # 断言边缘轴上的线条数量与边缘轴上的线条数量相等
        assert len(g.ax_marg_x.lines) == len(g.ax_marg_y.lines)
class TestJointPlot:

    # 使用随机种子创建随机数生成器
    rs = np.random.RandomState(sum(map(ord, "jointplot")))
    # 生成100个符合标准正态分布的随机数作为 x
    x = rs.randn(100)
    # 生成100个符合标准正态分布的随机数作为 y
    y = rs.randn(100)
    # 将 x 和 y 构建成一个 Pandas DataFrame 对象
    data = pd.DataFrame(dict(x=x, y=y))

    # 测试散点图功能
    def test_scatter(self):
        # 调用 seaborn 的 jointplot 函数绘制关联图 g
        g = ag.jointplot(x="x", y="y", data=self.data)
        # 断言主轴上的散点集合数量为1
        assert len(g.ax_joint.collections) == 1

        # 获取散点集合的偏移量并与 x, y 数组进行比较
        x, y = g.ax_joint.collections[0].get_offsets().T
        assert_array_equal(self.x, x)
        assert_array_equal(self.y, y)

        # 检查 x 轴边缘的直方图与 x 数组的直方图边缘数据几乎相等
        assert_array_almost_equal(
            [b.get_x() for b in g.ax_marg_x.patches],
            np.histogram_bin_edges(self.x, "auto")[:-1],
        )

        # 检查 y 轴边缘的直方图与 y 数组的直方图边缘数据几乎相等
        assert_array_almost_equal(
            [b.get_y() for b in g.ax_marg_y.patches],
            np.histogram_bin_edges(self.y, "auto")[:-1],
        )

    # 测试带有色调的散点图功能
    def test_scatter_hue(self, long_df):
        # 绘制带有色调的关联图 g1
        g1 = ag.jointplot(data=long_df, x="x", y="y", hue="a")

        # 创建一个新的 JointGrid 对象 g2
        g2 = ag.JointGrid()
        # 在 g2 上绘制散点图和 x 轴边缘的核密度估计图
        scatterplot(data=long_df, x="x", y="y", hue="a", ax=g2.ax_joint)
        kdeplot(data=long_df, x="x", hue="a", ax=g2.ax_marg_x, fill=True)
        # 在 g2 上绘制 y 轴边缘的核密度估计图
        kdeplot(data=long_df, y="y", hue="a", ax=g2.ax_marg_y, fill=True)

        # 断言 g1 和 g2 的主轴和边缘轴几乎相等
        assert_plots_equal(g1.ax_joint, g2.ax_joint)
        assert_plots_equal(g1.ax_marg_x, g2.ax_marg_x, labels=False)
        assert_plots_equal(g1.ax_marg_y, g2.ax_marg_y, labels=False)

    # 测试回归线功能
    def test_reg(self):
        # 绘制包含回归线的关联图 g
        g = ag.jointplot(x="x", y="y", data=self.data, kind="reg")
        # 断言主轴上的集合数量为2（包含散点集合和回归线集合）
        assert len(g.ax_joint.collections) == 2

        # 获取散点集合的偏移量并与 x, y 数组进行比较
        x, y = g.ax_joint.collections[0].get_offsets().T
        assert_array_equal(self.x, x)
        assert_array_equal(self.y, y)

        # 检查 x 轴边缘的直方图和 y 轴边缘的直方图是否存在
        assert g.ax_marg_x.patches
        assert g.ax_marg_y.patches

        # 检查 x 轴和 y 轴边缘的线是否存在
        assert g.ax_marg_x.lines
        assert g.ax_marg_y.lines

    # 测试残差图功能
    def test_resid(self):
        # 绘制包含残差图的关联图 g
        g = ag.jointplot(x="x", y="y", data=self.data, kind="resid")
        # 断言主轴上的集合是否存在
        assert g.ax_joint.collections
        # 断言 x 轴和 y 轴边缘上的线不存在
        assert g.ax_joint.lines
        assert not g.ax_marg_x.lines
        assert not g.ax_marg_y.lines

    # 测试直方图功能
    def test_hist(self, long_df):
        # 定义直方图的 bin 数组
        bins = 3, 6
        # 绘制包含直方图的关联图 g1
        g1 = ag.jointplot(data=long_df, x="x", y="y", kind="hist", bins=bins)

        # 创建一个新的 JointGrid 对象 g2
        g2 = ag.JointGrid()
        # 在 g2 上绘制散点图和 x 轴、y 轴边缘的直方图
        histplot(data=long_df, x="x", y="y", ax=g2.ax_joint, bins=bins)
        histplot(data=long_df, x="x", ax=g2.ax_marg_x, bins=bins[0])
        histplot(data=long_df, y="y", ax=g2.ax_marg_y, bins=bins[1])

        # 断言 g1 和 g2 的主轴和边缘轴几乎相等
        assert_plots_equal(g1.ax_joint, g2.ax_joint)
        assert_plots_equal(g1.ax_marg_x, g2.ax_marg_x, labels=False)
        assert_plots_equal(g1.ax_marg_y, g2.ax_marg_y, labels=False)

    # 测试六边形图功能
    def test_hex(self):
        # 绘制包含六边形图的关联图 g
        g = ag.jointplot(x="x", y="y", data=self.data, kind="hex")
        # 断言主轴和边缘轴上的集合和补丁是否存在
        assert g.ax_joint.collections
        assert g.ax_marg_x.patches
        assert g.ax_marg_y.patches
    # 测试核密度估计（KDE）在长格式数据框上的绘制

    # 使用 seaborn 的 jointplot 函数创建第一个图 g1，显示 x 和 y 的联合分布，采用 KDE 形式
    g1 = ag.jointplot(data=long_df, x="x", y="y", kind="kde")

    # 使用 seaborn 的 JointGrid 类创建第二个图 g2
    g2 = ag.JointGrid()

    # 在 g2 的联合图(ax_joint)上绘制 x 和 y 的 KDE
    kdeplot(data=long_df, x="x", y="y", ax=g2.ax_joint)

    # 在 g2 的边缘横轴(ax_marg_x)上绘制 x 的 KDE
    kdeplot(data=long_df, x="x", ax=g2.ax_marg_x)

    # 在 g2 的边缘纵轴(ax_marg_y)上绘制 y 的 KDE
    kdeplot(data=long_df, y="y", ax=g2.ax_marg_y)

    # 断言两个图 g1 和 g2 的联合图(ax_joint)相等
    assert_plots_equal(g1.ax_joint, g2.ax_joint)

    # 断言 g1 和 g2 的边缘横轴(ax_marg_x)相等，不包含标签
    assert_plots_equal(g1.ax_marg_x, g2.ax_marg_x, labels=False)

    # 断言 g1 和 g2 的边缘纵轴(ax_marg_y)相等，不包含标签
    assert_plots_equal(g1.ax_marg_y, g2.ax_marg_y, labels=False)

    # 测试带有色调变量的核密度估计（KDE）

    # 使用 seaborn 的 jointplot 函数创建第一个图 g1，显示 x 和 y 的联合分布，并带有色调变量 a，采用 KDE 形式
    g1 = ag.jointplot(data=long_df, x="x", y="y", hue="a", kind="kde")

    # 使用 seaborn 的 JointGrid 类创建第二个图 g2
    g2 = ag.JointGrid()

    # 在 g2 的联合图(ax_joint)上绘制 x 和 y 的 KDE，并带有色调变量 a
    kdeplot(data=long_df, x="x", y="y", hue="a", ax=g2.ax_joint)

    # 在 g2 的边缘横轴(ax_marg_x)上绘制 x 的 KDE，并带有色调变量 a
    kdeplot(data=long_df, x="x", hue="a", ax=g2.ax_marg_x)

    # 在 g2 的边缘纵轴(ax_marg_y)上绘制 y 的 KDE，并带有色调变量 a
    kdeplot(data=long_df, y="y", hue="a", ax=g2.ax_marg_y)

    # 断言两个图 g1 和 g2 的联合图(ax_joint)相等
    assert_plots_equal(g1.ax_joint, g2.ax_joint)

    # 断言 g1 和 g2 的边缘横轴(ax_marg_x)相等，不包含标签
    assert_plots_equal(g1.ax_marg_x, g2.ax_marg_x, labels=False)

    # 断言 g1 和 g2 的边缘纵轴(ax_marg_y)相等，不包含标签
    assert_plots_equal(g1.ax_marg_y, g2.ax_marg_y, labels=False)

    # 测试设置颜色的功能

    # 使用 seaborn 的 jointplot 函数创建图 g，显示 x 和 y 的联合分布，主要颜色为紫色
    g = ag.jointplot(x="x", y="y", data=self.data, color="purple")

    # 获取散点图的颜色，应为紫色
    scatter_color = g.ax_joint.collections[0].get_facecolor()
    assert_colors_equal(scatter_color, "purple")

    # 获取 x 边缘直方图的颜色，应为紫色
    hist_color = g.ax_marg_x.patches[0].get_facecolor()[:3]
    assert_colors_equal(hist_color, "purple")

    # 测试设置调色板的功能

    # 准备关键字参数字典 kws，包含数据为 long_df，色调变量为 a，调色板为 "Set2"
    kws = dict(data=long_df, hue="a", palette="Set2")

    # 使用 seaborn 的 jointplot 函数创建第一个图 g1，显示 x 和 y 的联合分布，并带有色调变量 a，使用调色板 "Set2"
    g1 = ag.jointplot(x="x", y="y", **kws)

    # 使用 seaborn 的 JointGrid 类创建第二个图 g2
    g2 = ag.JointGrid()

    # 在 g2 的联合图(ax_joint)上绘制 x 和 y 的散点图，并带有色调变量 a，使用调色板 "Set2"
    scatterplot(x="x", y="y", ax=g2.ax_joint, **kws)

    # 在 g2 的边缘横轴(ax_marg_x)上绘制 x 的 KDE，并填充，使用调色板 "Set2"
    kdeplot(x="x", ax=g2.ax_marg_x, fill=True, **kws)

    # 在 g2 的边缘纵轴(ax_marg_y)上绘制 y 的 KDE，并填充，使用调色板 "Set2"
    kdeplot(y="y", ax=g2.ax_marg_y, fill=True, **kws)

    # 断言两个图 g1 和 g2 的联合图(ax_joint)相等
    assert_plots_equal(g1.ax_joint, g2.ax_joint)

    # 断言 g1 和 g2 的边缘横轴(ax_marg_x)相等，不包含标签
    assert_plots_equal(g1.ax_marg_x, g2.ax_marg_x, labels=False)

    # 断言 g1 和 g2 的边缘纵轴(ax_marg_y)相等，不包含标签
    assert_plots_equal(g1.ax_marg_y, g2.ax_marg_y, labels=False)

    # 测试自定义六边形图的功能

    # 使用 seaborn 的 jointplot 函数创建图 g，显示 x 和 y 的联合分布，采用六边形图形式，设置网格大小为 5
    g = ag.jointplot(x="x", y="y", data=self.data, kind="hex",
                     joint_kws=dict(gridsize=5))

    # 断言 g 的联合图(ax_joint)中六边形的数量为 1
    assert len(g.ax_joint.collections) == 1

    # 获取 g 的联合图(ax_joint)中六边形数组的形状，预期应为 28，对应网格大小 5
    a = g.ax_joint.collections[0].get_array()
    assert a.shape[0] == 28  # 28 hexagons expected for gridsize 5

    # 测试不支持的图类型异常情况

    # 使用 pytest 的 raises 函数检测 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 seaborn 的 jointplot 函数，使用未支持的图类型 "not_a_kind"
        ag.jointplot(x="x", y="y", data=self.data, kind="not_a_kind")

    # 测试不支持的色调变量异常情况

    # 遍历图类型列表，检测对应的 ValueError 异常
    for kind in ["reg", "resid", "hex"]:
        with pytest.raises(ValueError):
            # 调用 seaborn 的 jointplot 函数，使用色调变量 "a" 和未支持的图类型
            ag.jointplot(x="x", y="y", hue="a", data=self.data, kind=kind)
    # 定义测试方法，用于验证 jointplot 绘图函数对输入的字典参数没有修改
    def test_leaky_dict(self):
        # 验证输入的字典参数在 jointplot 绘图函数中不被改变

        # 遍历两个关键字参数："joint_kws" 和 "marginal_kws"
        for kwarg in ("joint_kws", "marginal_kws"):
            # 遍历不同的绘图类型："hex", "kde", "resid", "reg", "scatter"
            for kind in ("hex", "kde", "resid", "reg", "scatter"):
                # 创建一个空字典
                empty_dict = {}
                # 使用 jointplot 绘制图形，并传入空字典作为关键字参数之一
                ag.jointplot(x="x", y="y", data=self.data, kind=kind,
                             **{kwarg: empty_dict})
                # 断言空字典未被修改
                assert empty_dict == {}

    # 定义测试方法，用于验证 distplot 的关键字参数导致警告
    def test_distplot_kwarg_warning(self, long_df):

        # 在捕获 UserWarning 警告时执行以下代码块
        with pytest.warns(UserWarning):
            # 使用 jointplot 绘制图形，并传入带有 rug=True 的 marginal_kws 字典参数
            g = ag.jointplot(data=long_df, x="x", y="y", marginal_kws=dict(rug=True))
        
        # 断言边缘 X 轴上存在补丁
        assert g.ax_marg_x.patches

    # 定义测试方法，用于验证传入自定义 axes 对象时的警告
    def test_ax_warning(self, long_df):

        # 获取当前的 Axes 对象
        ax = plt.gca()
        
        # 在捕获 UserWarning 警告时执行以下代码块
        with pytest.warns(UserWarning):
            # 使用 jointplot 绘制图形，并传入自定义的 Axes 对象
            g = ag.jointplot(data=long_df, x="x", y="y", ax=ax)
        
        # 断言联合图上存在集合对象
        assert g.ax_joint.collections
```