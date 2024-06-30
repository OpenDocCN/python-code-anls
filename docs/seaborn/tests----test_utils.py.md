# `D:\src\scipysrc\seaborn\tests\test_utils.py`

```
"""Tests for seaborn utility functions."""
# 导入必要的库和模块
import re
import tempfile
from types import ModuleType
from urllib.request import urlopen
from http.client import HTTPException

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# 导入 pytest 断言和测试工具
import pytest
from numpy.testing import (
    assert_array_equal,
)
from pandas.testing import (
    assert_series_equal,
    assert_frame_equal,
)

# 导入 seaborn 的相关模块和函数
from seaborn import utils, rcmod, scatterplot
from seaborn.utils import (
    get_dataset_names,
    get_color_cycle,
    remove_na,
    load_dataset,
    _assign_default_kwargs,
    _check_argument,
    _draw_figure,
    _deprecate_ci,
    _version_predates, DATASET_NAMES_URL,
)
# 导入兼容性函数
from seaborn._compat import get_legend_handles


# 生成一个服从标准正态分布的数组
a_norm = np.random.randn(100)


# 网络连接装饰器函数，用于测试跳过不可达 URL 的情况
def _network(t=None, url="https://github.com"):
    """
    Decorator that will skip a test if `url` is unreachable.

    Parameters
    ----------
    t : function, optional
    url : str, optional

    """
    if t is None:
        return lambda x: _network(x, url=url)

    def wrapper(*args, **kwargs):
        # 尝试连接 URL
        try:
            f = urlopen(url)
        except (OSError, HTTPException):
            # 如果连接失败，则跳过测试
            pytest.skip("No internet connection")
        else:
            f.close()
            return t(*args, **kwargs)
    return wrapper


# 测试 ci_to_errsize 函数的行为
def test_ci_to_errsize():
    """Test behavior of ci_to_errsize."""
    cis = [[.5, .5],
           [1.25, 1.5]]

    heights = [1, 1.5]

    actual_errsize = np.array([[.5, 1],
                               [.25, 0]])

    # 调用 ci_to_errsize 函数，检查输出是否符合预期
    test_errsize = utils.ci_to_errsize(cis, heights)
    assert_array_equal(actual_errsize, test_errsize)


# 测试颜色去饱和度函数 desaturate
def test_desaturate():
    """Test color desaturation."""
    out1 = utils.desaturate("red", .5)
    assert out1 == (.75, .25, .25)

    out2 = utils.desaturate("#00FF00", .5)
    assert out2 == (.25, .75, .25)

    out3 = utils.desaturate((0, 0, 1), .5)
    assert out3 == (.25, .25, .75)

    # 再次测试红色的去饱和度，确保结果一致
    out4 = utils.desaturate("red", .5)
    assert out4 == (.75, .25, .25)

    # 测试对 lightblue 去饱和度为 1，期望结果为原色
    out5 = utils.desaturate("lightblue", 1)
    assert out5 == mpl.colors.to_rgb("lightblue")


# 测试颜色饱和度函数 saturate
def test_saturate():
    """Test performance of saturation function."""
    out = utils.saturate((.75, .25, .25))
    assert out == (1, 0, 0)


# 使用 pytest 参数化装饰器测试 to_utf8 函数的多种输入
@pytest.mark.parametrize(
    "s,exp",
    [
        ("a", "a"),
        ("abc", "abc"),
        (b"a", "a"),
        (b"abc", "abc"),
        (bytearray("abc", "utf-8"), "abc"),
        (bytearray(), ""),
        (1, "1"),
        (0, "0"),
        ([], str([])),
    ],
)
def test_to_utf8(s, exp):
    """Test the to_utf8 function: object to string"""
    # 调用 to_utf8 函数，检查输出类型和内容是否符合预期
    u = utils.to_utf8(s)
    assert isinstance(u, str)
    assert u == exp


# 测试 SpineUtils 类的功能
class TestSpineUtils:

    # 定义测试用例中用到的测试边缘名称列表
    sides = ["left", "right", "bottom", "top"]
    # 定义外侧边和内侧边的列表
    outer_sides = ["top", "right"]
    inner_sides = ["left", "bottom"]

    # 设置偏移量
    offset = 10
    # 原始位置和偏移后的位置元组
    original_position = ("outward", 0)
    offset_position = ("outward", offset)

    # 测试函数：验证 despine 函数的功能
    def test_despine(self):
        # 创建一个图和轴对象
        f, ax = plt.subplots()
        # 验证默认所有边框均可见
        for side in self.sides:
            assert ax.spines[side].get_visible()

        # 调用 utils.despine 函数，隐藏外侧边框
        utils.despine()
        # 验证外侧边框是否不可见
        for side in self.outer_sides:
            assert not ax.spines[side].get_visible()
        # 验证内侧边框是否可见
        for side in self.inner_sides:
            assert ax.spines[side].get_visible()

        # 再次调用 utils.despine 函数，恢复所有边框可见状态
        utils.despine(**dict(zip(self.sides, [True] * 4)))
        # 验证所有边框是否均不可见
        for side in self.sides:
            assert not ax.spines[side].get_visible()

    # 测试函数：验证带指定轴的 despine 函数功能
    def test_despine_specific_axes(self):
        # 创建两个子图的图和轴对象
        f, (ax1, ax2) = plt.subplots(2, 1)

        # 调用 utils.despine 函数，隐藏指定轴的边框
        utils.despine(ax=ax2)

        # 验证 ax1 轴的边框是否可见
        for side in self.sides:
            assert ax1.spines[side].get_visible()

        # 验证 ax2 轴的外侧边框是否不可见，内侧边框是否可见
        for side in self.outer_sides:
            assert not ax2.spines[side].get_visible()
        for side in self.inner_sides:
            assert ax2.spines[side].get_visible()

    # 测试函数：验证带偏移量的 despine 函数功能
    def test_despine_with_offset(self):
        # 创建图和轴对象
        f, ax = plt.subplots()

        # 验证每个边框的位置是否为原始位置
        for side in self.sides:
            pos = ax.spines[side].get_position()
            assert pos == self.original_position

        # 调用 utils.despine 函数，带有指定偏移量
        utils.despine(ax=ax, offset=self.offset)

        # 验证每个边框的位置是否根据可见状态调整为新位置或保持原始位置
        for side in self.sides:
            is_visible = ax.spines[side].get_visible()
            new_position = ax.spines[side].get_position()
            if is_visible:
                assert new_position == self.offset_position
            else:
                assert new_position == self.original_position

    # 测试函数：验证带特定边框偏移量的 despine 函数功能
    def test_despine_side_specific_offset(self):
        # 创建图和轴对象
        f, ax = plt.subplots()
        # 调用 utils.despine 函数，带有特定的边框偏移量
        utils.despine(ax=ax, offset=dict(left=self.offset))

        # 验证每个边框的位置是否根据可见状态调整为新位置或保持原始位置
        for side in self.sides:
            is_visible = ax.spines[side].get_visible()
            new_position = ax.spines[side].get_position()
            if is_visible and side == "left":
                assert new_position == self.offset_position
            else:
                assert new_position == self.original_position

    # 测试函数：验证带偏移量和特定轴的 despine 函数功能
    def test_despine_with_offset_specific_axes(self):
        # 创建两个子图的图和轴对象
        f, (ax1, ax2) = plt.subplots(2, 1)

        # 调用 utils.despine 函数，带有指定偏移量和特定轴
        utils.despine(offset=self.offset, ax=ax2)

        # 验证每个边框的位置是否根据可见状态调整为新位置或保持原始位置
        for side in self.sides:
            pos1 = ax1.spines[side].get_position()
            pos2 = ax2.spines[side].get_position()
            assert pos1 == self.original_position
            if ax2.spines[side].get_visible():
                assert pos2 == self.offset_position
            else:
                assert pos2 == self.original_position

    # 测试函数：验证带修剪边框的 despine 函数功能
    def test_despine_trim_spines(self):
        # 创建图和轴对象，并绘制曲线
        f, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_xlim(.75, 3.25)

        # 调用 utils.despine 函数，修剪边框
        utils.despine(trim=True)
        # 验证内侧边框的边界是否设置为指定值
        for side in self.inner_sides:
            bounds = ax.spines[side].get_bounds()
            assert bounds == (1, 3)
    # 定义测试函数，用于测试 despine 函数对图表的影响
    def test_despine_trim_inverted(self):

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 在坐标轴上绘制一条直线
        ax.plot([1, 2, 3], [1, 2, 3])

        # 设置 y 轴的显示范围为 (0.85, 3.15)
        ax.set_ylim(.85, 3.15)

        # 反转 y 轴方向
        ax.invert_yaxis()

        # 调用 utils 模块的 despine 函数，去除图表周围的边框并进行修剪
        utils.despine(trim=True)

        # 遍历内侧边框列表，检查每个边框的边界值是否为 (1, 3)
        for side in self.inner_sides:
            bounds = ax.spines[side].get_bounds()
            assert bounds == (1, 3)

    # 定义测试函数，测试在不显示 y 轴刻度的情况下 despine 函数的效果
    def test_despine_trim_noticks(self):

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 在坐标轴上绘制一条直线
        ax.plot([1, 2, 3], [1, 2, 3])

        # 设置不显示 y 轴刻度
        ax.set_yticks([])

        # 调用 utils 模块的 despine 函数，去除图表周围的边框并进行修剪
        utils.despine(trim=True)

        # 断言 y 轴的刻度数目是否为 0
        assert ax.get_yticks().size == 0

    # 定义测试函数，测试在类别数据绘图时 despine 函数的效果
    def test_despine_trim_categorical(self):

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 在坐标轴上以类别数据绘制一条直线
        ax.plot(["a", "b", "c"], [1, 2, 3])

        # 调用 utils 模块的 despine 函数，去除图表周围的边框并进行修剪
        utils.despine(trim=True)

        # 检查左边框的边界值是否为 (1, 3)
        bounds = ax.spines["left"].get_bounds()
        assert bounds == (1, 3)

        # 检查底部边框的边界值是否为 (0, 2)
        bounds = ax.spines["bottom"].get_bounds()
        assert bounds == (0, 2)

    # 定义测试函数，测试移动的刻度线在使用 despine 函数后的效果
    def test_despine_moved_ticks(self):

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 对于 y 轴上的每一个主刻度，设置第一条刻度线可见
        for t in ax.yaxis.majorTicks:
            t.tick1line.set_visible(True)

        # 调用 utils 模块的 despine 函数，去除图表周围的边框，仅保留左侧边框，并进行修剪
        utils.despine(ax=ax, left=True, right=False)

        # 对于 y 轴上的每一个主刻度，断言第二条刻度线是否可见
        for t in ax.yaxis.majorTicks:
            assert t.tick2line.get_visible()

        # 关闭当前图形
        plt.close(f)

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 对于 y 轴上的每一个主刻度，设置第一条刻度线不可见
        for t in ax.yaxis.majorTicks:
            t.tick1line.set_visible(False)

        # 调用 utils 模块的 despine 函数，去除图表周围的边框，仅保留左侧边框，并进行修剪
        utils.despine(ax=ax, left=True, right=False)

        # 对于 y 轴上的每一个主刻度，断言第二条刻度线是否不可见
        for t in ax.yaxis.majorTicks:
            assert not t.tick2line.get_visible()

        # 关闭当前图形
        plt.close(f)

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 对于 x 轴上的每一个主刻度，设置第一条刻度线可见
        for t in ax.xaxis.majorTicks:
            t.tick1line.set_visible(True)

        # 调用 utils 模块的 despine 函数，去除图表周围的边框，仅保留底部边框，并进行修剪
        utils.despine(ax=ax, bottom=True, top=False)

        # 对于 x 轴上的每一个主刻度，断言第二条刻度线是否可见
        for t in ax.xaxis.majorTicks:
            assert t.tick2line.get_visible()

        # 关闭当前图形
        plt.close(f)

        # 创建一个图形和坐标轴对象
        f, ax = plt.subplots()

        # 对于 x 轴上的每一个主刻度，设置第一条刻度线不可见
        for t in ax.xaxis.majorTicks:
            t.tick1line.set_visible(False)

        # 调用 utils 模块的 despine 函数，去除图表周围的边框，仅保留底部边框，并进行修剪
        utils.despine(ax=ax, bottom=True, top=False)

        # 对于 x 轴上的每一个主刻度，断言第二条刻度线是否不可见
        for t in ax.xaxis.majorTicks:
            assert not t.tick2line.get_visible()

        # 关闭当前图形
        plt.close(f)
# 定义一个测试函数，用于检查图表中的刻度标签是否重叠
def test_ticklabels_overlap():

    # 调用 rcmod.set()，可能是用于设置 matplotlib 的默认参数
    rcmod.set()

    # 创建一个大小为 (2, 2) 的图表 f，并返回图表对象 f 和轴对象 ax
    f, ax = plt.subplots(figsize=(2, 2))

    # 调用 f.tight_layout()，确保 Agg 渲染器能够正常工作
    f.tight_layout()  # This gets the Agg renderer working

    # 断言检查轴对象 ax 的 x 轴刻度标签是否重叠，返回布尔值
    assert not utils.axis_ticklabels_overlap(ax.get_xticklabels())

    # 定义两个大字符串作为 x 轴刻度标签
    big_strings = "abcdefgh", "ijklmnop"
    
    # 设置 x 轴的数据范围为 -0.5 到 1.5
    ax.set_xlim(-.5, 1.5)
    
    # 设置 x 轴的刻度位置为 [0, 1]，刻度标签为 big_strings
    ax.set_xticks([0, 1])
    ax.set_xticklabels(big_strings)

    # 断言检查轴对象 ax 的 x 轴刻度标签是否重叠，返回布尔值
    assert utils.axis_ticklabels_overlap(ax.get_xticklabels())

    # 调用 utils.axes_ticklabels_overlap(ax)，返回 x、y 轴刻度标签是否重叠的布尔值
    x, y = utils.axes_ticklabels_overlap(ax)
    assert x  # 断言 x 为 True
    assert not y  # 断言 y 为 False


# 定义测试函数，测试将 Locator 转换为图例条目的功能
def test_locator_to_legend_entries():

    # 创建 MaxNLocator 对象 locator，指定 nbins=3
    locator = mpl.ticker.MaxNLocator(nbins=3)

    # 定义限制 limits 为 (0.09, 0.4)
    limits = (0.09, 0.4)
    
    # 调用 utils.locator_to_legend_entries，将 locator 转换为图例条目 levels 和字符串列表 str_levels
    levels, str_levels = utils.locator_to_legend_entries(
        locator, limits, float
    )
    
    # 断言检查 str_levels 是否等于预期值 ["0.15", "0.30"]
    assert str_levels == ["0.15", "0.30"]

    # 更改 limits 为 (0.8, 0.9)
    limits = (0.8, 0.9)
    
    # 再次调用 utils.locator_to_legend_entries，检查 str_levels 是否等于预期值 ["0.80", "0.84", "0.88"]
    levels, str_levels = utils.locator_to_legend_entries(
        locator, limits, float
    )
    assert str_levels == ["0.80", "0.84", "0.88"]

    # 更改 limits 为 (1, 6)
    limits = (1, 6)
    
    # 再次调用 utils.locator_to_legend_entries，检查 str_levels 是否等于预期值 ["2", "4", "6"]
    levels, str_levels = utils.locator_to_legend_entries(locator, limits, int)
    assert str_levels == ["2", "4", "6"]

    # 创建 LogLocator 对象 locator，指定 numticks=5
    locator = mpl.ticker.LogLocator(numticks=5)
    
    # 定义限制 limits 为 (5, 1425)
    limits = (5, 1425)
    
    # 再次调用 utils.locator_to_legend_entries，检查 str_levels 是否等于预期值 ['10', '100', '1000']
    levels, str_levels = utils.locator_to_legend_entries(locator, limits, int)
    assert str_levels == ['10', '100', '1000']

    # 更改 limits 为 (0.00003, 0.02)
    limits = (0.00003, 0.02)
    
    # 再次调用 utils.locator_to_legend_entries，使用正则表达式匹配检查 str_levels 是否符合预期
    _, str_levels = utils.locator_to_legend_entries(locator, limits, float)
    for i, exp in enumerate([4, 3, 2]):
        assert re.match(f"1e.0{exp}", str_levels[i])


# 定义测试函数，测试移动 Matplotlib 对象中图例的功能
def test_move_legend_matplotlib_objects():

    # 创建一个新的图形 fig 和轴对象 ax
    fig, ax = plt.subplots()

    # 定义颜色和标签列表
    colors = "C2", "C5"
    labels = "first label", "second label"
    title = "the legend"

    # 在轴对象 ax 上绘制两条曲线，每条曲线对应一个颜色和标签
    for color, label in zip(colors, labels):
        ax.plot([0, 1], color=color, label=label)
    
    # 在 ax 上创建图例，位置为 "upper right"，标题为 title
    ax.legend(loc="upper right", title=title)
    
    # 调用 utils._draw_figure(fig)，可能用于绘制图形
    utils._draw_figure(fig)
    
    # 获取坐标转换器 xfm，用于获取图例的位置
    xfm = ax.transAxes.inverted().transform

    # --- 测试轴上的图例

    # 获取旧的图例位置
    old_pos = xfm(ax.legend_.legendPatch.get_extents())

    # 设置新的字体大小为 14，调用 utils.move_legend 移动图例到 "lower left"
    utils.move_legend(ax, "lower left", title_fontsize=14)
    
    # 再次绘制图形，更新图例位置
    utils._draw_figure(fig)
    
    # 获取新的图例位置
    new_pos = xfm(ax.legend_.legendPatch.get_extents())

    # 断言新的图例位置是否在旧的图例位置之下
    assert (new_pos < old_pos).all()
    
    # 断言图例的标题是否与预期的 title 相同
    assert ax.legend_.get_title().get_text() == title
    
    # 断言图例标题的字体大小是否为设置的新字体大小
    assert ax.legend_.get_title().get_size() == 14

    # --- 测试替换标题

    # 设置新的标题为 "new title"，调用 utils.move_legend 更新图例标题
    new_title = "new title"
    utils.move_legend(ax, "lower left", title=new_title)
    utils._draw_figure(fig)
    
    # 断言图例的标题是否已更新为新标题
    assert ax.legend_.get_title().get_text() == new_title

    # --- 测试图形上的图例

    # 在图形上创建图例，位置为 "upper right"，标题为 title
    fig.legend(loc="upper right", title=title)
    _draw_figure(fig)
    
    # 获取坐标转换器 xfm，用于获取图例的位置
    xfm = fig.transFigure.inverted().transform
    
    # 获取图形上旧的图例位置
    old_pos = xfm(fig.legends[0].legendPatch.get_extents())

    # 调用 utils.move_legend 移动图例到 "lower left"，并更新标题为 new_title
    utils.move_legend(fig, "lower left", title=new_title)
    _draw_figure(fig)
    
    # 获取图形上新的图例位置
    new_pos = xfm(fig.legends[0].legendPatch.get_extents())
    
    # 断言新的图例位置是否在旧的图例位置之下
    assert (new_pos < old_pos).all()
    
    # 断言图例的标题是否已更新为 new_title
    assert fig.legends[0].get_title().get_text() == new_title
# 定义一个测试函数，用于测试移动图例和网格对象
def test_move_legend_grid_object(long_df):
    # 从 seaborn 库导入 FacetGrid 类
    from seaborn.axisgrid import FacetGrid
    
    # 指定色调变量名为 "a"，创建 FacetGrid 对象 g
    hue_var = "a"
    g = FacetGrid(long_df, hue=hue_var)
    
    # 对 g 应用 plt.plot 函数，绘制 "x" 到 "y" 的图形
    g.map(plt.plot, "x", "y")
    
    # 添加图例到 g
    g.add_legend()
    
    # 将 g.figure 绘制出来
    _draw_figure(g.figure)
    
    # 获取 g.legend.legendPatch 的范围，转换成新的位置信息
    xfm = g.figure.transFigure.inverted().transform
    old_pos = xfm(g.legend.legendPatch.get_extents())
    
    # 设置图例标题的字体大小为 20，移动图例到 "lower left" 位置
    fontsize = 20
    utils.move_legend(g, "lower left", title_fontsize=fontsize)
    
    # 再次绘制 g.figure
    _draw_figure(g.figure)
    
    # 获取移动后的图例的范围信息，转换成新的位置信息
    new_pos = xfm(g.legend.legendPatch.get_extents())
    
    # 断言新位置的范围小于旧位置的范围
    assert (new_pos < old_pos).all()
    
    # 断言图例的标题文本与 hue_var 相同
    assert g.legend.get_title().get_text() == hue_var
    
    # 断言图例的标题字体大小为 fontsize
    assert g.legend.get_title().get_size() == fontsize
    
    # 断言获取图例句柄，确保颜色与预期相符
    assert get_legend_handles(g.legend)
    for i, h in enumerate(get_legend_handles(g.legend)):
        assert mpl.colors.to_rgb(h.get_color()) == mpl.colors.to_rgb(f"C{i}")


# 定义一个测试函数，用于检查移动图例函数的输入验证
def test_move_legend_input_checks():
    # 创建一个 subplot 对象 ax
    ax = plt.figure().subplots()
    
    # 使用 pytest 检查调用 move_legend 函数时的 TypeError 异常
    with pytest.raises(TypeError):
        utils.move_legend(ax.xaxis, "best")
    
    # 使用 pytest 检查调用 move_legend 函数时的 ValueError 异常
    with pytest.raises(ValueError):
        utils.move_legend(ax, "best")
    
    # 使用 pytest 检查调用 move_legend 函数时的 ValueError 异常
    with pytest.raises(ValueError):
        utils.move_legend(ax.figure, "best")


# 定义一个测试函数，用于验证带标签的移动图例函数
def test_move_legend_with_labels(long_df):
    # 获取长数据集 "long_df" 中唯一的 "a" 列值的顺序
    order = long_df["a"].unique()
    
    # 根据 "x" 和 "y" 列绘制散点图，并添加颜色标签
    ax = scatterplot(long_df, x="x", y="y", hue="a", hue_order=order)
    
    # 获取图例句柄及其颜色信息
    handles_before = get_legend_handles(ax.get_legend())
    colors_before = [h.get_markerfacecolor() for h in handles_before]
    
    # 使用 utils.move_legend 函数移动图例到 "best" 位置，并指定新标签
    utils.move_legend(ax, "best", labels=[s.capitalize() for s in order])
    
    # 再次绘制 ax.figure
    _draw_figure(ax.figure)
    
    # 获取移动后的图例文本
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    
    # 断言图例文本与预期标签相符
    assert texts == labels
    
    # 获取移动后的图例句柄及其颜色信息
    handles_after = get_legend_handles(ax.get_legend())
    colors_after = [h.get_markerfacecolor() for h in handles_after]
    
    # 断言移动前后图例的颜色信息相同
    assert colors_before == colors_after
    
    # 使用 pytest 检查调用 move_legend 函数时的 ValueError 异常
    with pytest.raises(ValueError, match="Length of new labels"):
        utils.move_legend(ax, "best", labels=labels[:-1])


# 定义一个函数，用于检查加载数据集函数的正确性
def check_load_dataset(name):
    # 加载指定名称的数据集，确保返回类型为 pd.DataFrame
    ds = load_dataset(name, cache=False)
    assert isinstance(ds, pd.DataFrame)


# 定义一个函数，用于检查加载缓存数据集函数的正确性
def check_load_cached_dataset(name):
    # 使用临时目录测试缓存数据集加载
    with tempfile.TemporaryDirectory() as tmpdir:
        # 下载并缓存数据集
        ds = load_dataset(name, cache=True, data_home=tmpdir)
        
        # 使用缓存版本的数据集
        ds2 = load_dataset(name, cache=True, data_home=tmpdir)
        
        # 断言两个数据集对象相等
        assert_frame_equal(ds, ds2)


# 使用装饰器 @_network(url=DATASET_NAMES_URL) 定义一个测试函数，用于获取数据集名称
def test_get_dataset_names():
    # 获取所有可用数据集的名称列表
    names = get_dataset_names()
    
    # 断言数据集名称列表非空
    assert names
    
    # 断言 "tips" 数据集在名称列表中
    assert "tips" in names


# 使用装饰器 @_network(url=DATASET_NAMES_URL) 定义一个测试函数，用于加载数据集
def test_load_datasets():
    # 遍历所有可用数据集名称，依次调用 check_load_dataset 函数进行测试
    for name in get_dataset_names():
        check_load_dataset(name)


# 使用装饰器 @_network(url=DATASET_NAMES_URL) 定义一个测试函数，用于检查加载错误的数据集名称
def test_load_dataset_string_error():
    # 定义一个不存在的数据集名称
    name = "bad_name"
    # 生成错误消息，指示指定的数据集名称不是示例数据集之一
    err = f"'{name}' is not one of the example datasets."
    
    # 使用 pytest 的 raises() 上下文管理器来捕获 ValueError 异常，并检查其错误消息与预期的错误消息是否匹配
    with pytest.raises(ValueError, match=err):
        # 调用 load_dataset 函数，预期其抛出包含特定错误消息的 ValueError 异常
        load_dataset(name)
def test_load_dataset_passed_data_error():

    # 创建一个空的 DataFrame 对象
    df = pd.DataFrame()
    # 准备错误消息字符串
    err = "This function accepts only strings"
    # 使用 pytest 来检查 load_dataset 函数在传入 DataFrame 时是否会抛出 TypeError，并匹配特定错误消息
    with pytest.raises(TypeError, match=err):
        load_dataset(df)


@_network(url="https://github.com/mwaskom/seaborn-data")
def test_load_cached_datasets():

    # 这是一个重要的测试，验证我们能否加载所有可用的数据集
    for name in get_dataset_names():
        # 由于 @network 装饰器某种方式上阻碍了此生成器的作用，所以需要显式调用
        check_load_cached_dataset(name)


def test_relative_luminance():
    """Test relative luminance."""
    # 测试相对亮度函数，以下是不同输入情况的预期输出
    out1 = utils.relative_luminance("white")
    assert out1 == 1

    out2 = utils.relative_luminance("#000000")
    assert out2 == 0

    out3 = utils.relative_luminance((.25, .5, .75))
    assert out3 == pytest.approx(0.201624536)

    # 生成一组颜色，计算它们的相对亮度
    rgbs = mpl.cm.RdBu(np.linspace(0, 1, 10))
    lums1 = [utils.relative_luminance(rgb) for rgb in rgbs]
    lums2 = utils.relative_luminance(rgbs)

    # 检查每对 lums1 和 lums2 中的相对亮度是否近似相等
    for lum1, lum2 in zip(lums1, lums2):
        assert lum1 == pytest.approx(lum2)


@pytest.mark.parametrize(
    "cycler,result",
    [
        (cycler(color=["y"]), ["y"]),
        (cycler(color=["k"]), ["k"]),
        (cycler(color=["k", "y"]), ["k", "y"]),
        (cycler(color=["y", "k"]), ["y", "k"]),
        (cycler(color=["b", "r"]), ["b", "r"]),
        (cycler(color=["r", "b"]), ["r", "b"]),
        (cycler(lw=[1, 2]), [".15"]),  # 在循环中没有颜色
    ],
)
def test_get_color_cycle(cycler, result):
    # 使用指定的轴属性循环测试获取颜色循环
    with mpl.rc_context(rc={"axes.prop_cycle": cycler}):
        assert get_color_cycle() == result


def test_remove_na():

    # 测试从数组中删除 NaN 值的函数
    a_array = np.array([1, 2, np.nan, 3])
    a_array_rm = remove_na(a_array)
    assert_array_equal(a_array_rm, np.array([1, 2, 3]))

    # 测试从 Series 中删除 NaN 值的函数
    a_series = pd.Series([1, 2, np.nan, 3])
    a_series_rm = remove_na(a_series)
    assert_series_equal(a_series_rm, pd.Series([1., 2, 3], [0, 1, 3]))


def test_assign_default_kwargs():

    def f(a, b, c, d):
        pass

    def g(c=1, d=2):
        pass

    kws = {"c": 3}

    # 测试分配默认关键字参数的函数
    kws = _assign_default_kwargs(kws, f, g)
    assert kws == {"c": 3, "d": 2}


def test_check_argument():

    opts = ["a", "b", None]
    # 测试检查参数是否在选项列表中的函数
    assert _check_argument("arg", opts, "a") == "a"
    assert _check_argument("arg", opts, None) is None
    assert _check_argument("arg", opts, "aa", prefix=True) == "aa"
    assert _check_argument("arg", opts, None, prefix=True) is None
    with pytest.raises(ValueError, match="The value for `arg`"):
        _check_argument("arg", opts, "c")
    with pytest.raises(ValueError, match="The value for `arg`"):
        _check_argument("arg", opts, "c", prefix=True)
    with pytest.raises(ValueError, match="The value for `arg`"):
        _check_argument("arg", opts[:-1], None)
    with pytest.raises(ValueError, match="The value for `arg`"):
        _check_argument("arg", opts[:-1], None, prefix=True)


def test_draw_figure():

    # 这是一个待实现的测试函数，测试绘制图形功能的各个方面
    # 创建一个新的图形（figure）和一个子图（axes）
    f, ax = plt.subplots()
    # 在子图上绘制一条折线，横坐标为 ["a", "b", "c"]，纵坐标为 [1, 2, 3]
    ax.plot(["a", "b", "c"], [1, 2, 3])
    # 调用函数 _draw_figure 来处理图形 f
    _draw_figure(f)
    # 断言：确认图形 f 没有过时（stale）
    assert not f.stale
    # 断言：确认子图 ax 的第一个 x 轴刻度标签的文本内容是 "a"
    # 注：在绘制之前，刻度标签可能未被填充，但这可能会改变
    assert ax.get_xticklabels()[0].get_text() == "a"
# 定义一个测试函数，用于测试 _deprecate_ci 函数的行为
def test_deprecate_ci():

    # 警告消息，指出 'ci' 参数已被弃用，建议使用 errorbar=None 替代
    msg = "\n\nThe `ci` parameter is deprecated. Use `errorbar="

    # 在使用 _deprecate_ci 函数时，期望会触发 FutureWarning 警告，匹配特定的警告消息
    with pytest.warns(FutureWarning, match=msg + "None"):
        out = _deprecate_ci(None, None)
    # 确保函数返回 None
    assert out is None

    # 测试当传入 'sd' 参数时，是否触发相应的警告
    with pytest.warns(FutureWarning, match=msg + "'sd'"):
        out = _deprecate_ci(None, "sd")
    # 确保函数返回 'sd'
    assert out == "sd"

    # 测试当传入 ('ci', 68) 参数时，是否触发相应的警告
    with pytest.warns(FutureWarning, match=msg + r"\('ci', 68\)"):
        out = _deprecate_ci(None, 68)
    # 确保函数返回元组 ('ci', 68)
    assert out == ("ci", 68)


# 定义一个测试函数，用于测试 _version_predates 函数的行为
def test_version_predates():

    # 创建一个名为 'mock' 的模块对象，并设置其版本号为 '1.2.3'
    mock = ModuleType("mock")
    mock.__version__ = "1.2.3"

    # 测试 _version_predates 函数，检查 'mock' 模块版本是否早于 '1.2.4'
    assert _version_predates(mock, "1.2.4")
    # 测试 _version_predates 函数，检查 'mock' 模块版本是否早于 '1.3'
    assert _version_predates(mock, "1.3")

    # 测试 _version_predates 函数，检查 'mock' 模块版本是否不早于 '1.2.3'
    assert not _version_predates(mock, "1.2.3")
    # 测试 _version_predates 函数，检查 'mock' 模块版本是否不早于 '0.8'
    assert not _version_predates(mock, "0.8")
    # 测试 _version_predates 函数，检查 'mock' 模块版本是否不早于 '1'
    assert not _version_predates(mock, "1")
```