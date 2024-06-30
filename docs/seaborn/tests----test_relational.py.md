# `D:\src\scipysrc\seaborn\tests\test_relational.py`

```
# 导入必要的模块和函数
from itertools import product  # 导入 itertools 模块中的 product 函数
import warnings  # 导入 warnings 模块

import numpy as np  # 导入 NumPy 库并使用 np 别名
import matplotlib as mpl  # 导入 Matplotlib 库并使用 mpl 别名
import matplotlib.pyplot as plt  # 导入 Matplotlib 的 pyplot 模块并使用 plt 别名
from matplotlib.colors import same_color, to_rgba  # 导入 Matplotlib 的颜色相关函数

import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 导入 NumPy 测试相关函数

from seaborn.palettes import color_palette  # 从 seaborn 库中导入颜色调色板函数
from seaborn._base import categorical_order, unique_markers  # 从 seaborn 库中导入基础函数

from seaborn.relational import (  # 从 seaborn 中导入关系绘图相关模块和函数
    _RelationalPlotter,
    _LinePlotter,
    _ScatterPlotter,
    relplot,
    lineplot,
    scatterplot
)

from seaborn.utils import _draw_figure, _version_predates  # 从 seaborn 工具模块导入相关函数
from seaborn._compat import get_colormap, get_legend_handles  # 从 seaborn 兼容性模块导入相关函数
from seaborn._testing import assert_plots_equal  # 从 seaborn 测试模块导入测试函数

# 定义一个 pytest fixture，参数化测试用例
@pytest.fixture(params=[
    dict(x="x", y="y"),
    dict(x="t", y="y"),
    dict(x="a", y="y"),
    dict(x="x", y="y", hue="y"),
    dict(x="x", y="y", hue="a"),
    dict(x="x", y="y", size="a"),
    dict(x="x", y="y", style="a"),
    dict(x="x", y="y", hue="s"),
    dict(x="x", y="y", size="s"),
    dict(x="x", y="y", style="s"),
    dict(x="x", y="y", hue="a", style="a"),
    dict(x="x", y="y", hue="a", size="b", style="b"),
])
def long_semantics(request):
    return request.param  # 返回参数化的测试用例字典

# 定义一个 Helpers 类，包含一些辅助方法和 pytest fixtures
class Helpers:

    @pytest.fixture
    def levels(self, long_df):
        return {var: categorical_order(long_df[var]) for var in ["a", "b"]}  # 返回数据框中变量 'a' 和 'b' 的分类顺序字典

    # 获取散点图中集合的颜色值并返回为 RGB 元组列表
    def scatter_rgbs(self, collections):
        rgbs = []
        for col in collections:
            rgb = tuple(col.get_facecolor().squeeze()[:3])
            rgbs.append(rgb)
        return rgbs

    # 比较多个路径对象是否相等
    def paths_equal(self, *args):
        equal = all([len(a) == len(args[0]) for a in args])

        for p1, p2 in zip(*args):
            equal &= np.array_equal(p1.vertices, p2.vertices)
            equal &= np.array_equal(p1.codes, p2.codes)
        return equal

# 定义一个 SharedAxesLevelTests 类，包含测试共享轴级别的方法
class SharedAxesLevelTests:

    # 测试不同颜色配置的绘图功能
    def test_color(self, long_df):
        # 创建一个图形并获取其子图对象
        ax = plt.figure().subplots()
        # 调用被测试函数并验证最后绘制的颜色是否为预期的 "C0"
        self.func(data=long_df, x="x", y="y", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C0")

        ax = plt.figure().subplots()
        # 连续调用被测试函数并验证最后绘制的颜色是否为预期的 "C1"
        self.func(data=long_df, x="x", y="y", ax=ax)
        self.func(data=long_df, x="x", y="y", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C1")

        ax = plt.figure().subplots()
        # 调用被测试函数并验证指定颜色参数后最后绘制的颜色是否为预期的 "C2"
        self.func(data=long_df, x="x", y="y", color="C2", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C2")

        ax = plt.figure().subplots()
        # 调用被测试函数并验证指定颜色参数后最后绘制的颜色是否为预期的 "C2"
        self.func(data=long_df, x="x", y="y", c="C2", ax=ax)
        assert self.get_last_color(ax) == to_rgba("C2")

# 定义一个 TestRelationalPlotter 类，继承 Helpers 类，用于测试关系绘图功能
class TestRelationalPlotter(Helpers):

    # 这里应该包含关系绘图相关的测试方法，但由于截断，未提供完整代码
    # 测试宽格式数据框的变量分配功能
    def test_wide_df_variables(self, wide_df):
        # 创建 _RelationalPlotter 实例
        p = _RelationalPlotter()
        # 将数据分配给 plotter 实例
        p.assign_variables(data=wide_df)
        # 断言输入数据格式为 "wide"
        assert p.input_format == "wide"
        # 断言变量列表包含 ["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]
        # 断言 plot_data 的长度等于数据框中元素的乘积
        assert len(p.plot_data) == np.prod(wide_df.shape)

        # 检查 x 轴数据
        x = p.plot_data["x"]
        expected_x = np.tile(wide_df.index, wide_df.shape[1])
        assert_array_equal(x, expected_x)

        # 检查 y 轴数据
        y = p.plot_data["y"]
        expected_y = wide_df.to_numpy().ravel(order="f")
        assert_array_equal(y, expected_y)

        # 检查 hue 数据
        hue = p.plot_data["hue"]
        expected_hue = np.repeat(wide_df.columns.to_numpy(), wide_df.shape[0])
        assert_array_equal(hue, expected_hue)

        # 检查 style 数据
        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量字典的一致性
        assert p.variables["x"] == wide_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == wide_df.columns.name
        assert p.variables["style"] == wide_df.columns.name

    # 测试包含非数值变量的宽格式数据框
    def test_wide_df_with_nonnumeric_variables(self, long_df):
        # 创建 _RelationalPlotter 实例
        p = _RelationalPlotter()
        # 将数据分配给 plotter 实例
        p.assign_variables(data=long_df)
        # 断言输入数据格式为 "wide"
        assert p.input_format == "wide"
        # 断言变量列表包含 ["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 选择数据框中的数值列
        numeric_df = long_df.select_dtypes("number")

        # 断言 plot_data 的长度等于数值数据框中元素的乘积
        assert len(p.plot_data) == np.prod(numeric_df.shape)

        # 检查 x 轴数据
        x = p.plot_data["x"]
        expected_x = np.tile(numeric_df.index, numeric_df.shape[1])
        assert_array_equal(x, expected_x)

        # 检查 y 轴数据
        y = p.plot_data["y"]
        expected_y = numeric_df.to_numpy().ravel(order="f")
        assert_array_equal(y, expected_y)

        # 检查 hue 数据
        hue = p.plot_data["hue"]
        expected_hue = np.repeat(
            numeric_df.columns.to_numpy(), numeric_df.shape[0]
        )
        assert_array_equal(hue, expected_hue)

        # 检查 style 数据
        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量字典的一致性
        assert p.variables["x"] == numeric_df.index.name
        assert p.variables["y"] is None
        assert p.variables["hue"] == numeric_df.columns.name
        assert p.variables["style"] == numeric_df.columns.name
    # 定义测试方法，用于测试宽格式数组变量的处理
    def test_wide_array_variables(self, wide_array):
        # 创建关系绘图器对象
        p = _RelationalPlotter()
        # 将数据分配给关系绘图器对象
        p.assign_variables(data=wide_array)
        # 断言输入格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表包含 ["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]
        # 断言绘图数据的长度等于宽格式数组的元素个数
        assert len(p.plot_data) == np.prod(wide_array.shape)

        # 获取宽格式数组的行数和列数
        nrow, ncol = wide_array.shape

        # 检查绘图数据中的"x"变量
        x = p.plot_data["x"]
        # 生成预期的"x"数据，使用 np.tile 将行索引复制 ncol 次
        expected_x = np.tile(np.arange(nrow), ncol)
        assert_array_equal(x, expected_x)

        # 检查绘图数据中的"y"变量
        y = p.plot_data["y"]
        # 生成预期的"y"数据，使用 ravel(order="f") 将宽格式数组展平
        expected_y = wide_array.ravel(order="f")
        assert_array_equal(y, expected_y)

        # 检查绘图数据中的"hue"变量
        hue = p.plot_data["hue"]
        # 生成预期的"hue"数据，使用 np.repeat 将列索引重复 nrow 次
        expected_hue = np.repeat(np.arange(ncol), nrow)
        assert_array_equal(hue, expected_hue)

        # 检查绘图数据中的"style"变量
        style = p.plot_data["style"]
        # "style"数据应与"hue"相同
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量字典中的"x"、"y"、"hue"、"style"都为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    # 定义测试方法，用于测试平坦格式数组变量的处理
    def test_flat_array_variables(self, flat_array):
        # 创建关系绘图器对象
        p = _RelationalPlotter()
        # 将数据分配给关系绘图器对象
        p.assign_variables(data=flat_array)
        # 断言输入格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表包含 ["x", "y"]
        assert list(p.variables) == ["x", "y"]
        # 断言绘图数据的长度等于平坦格式数组的元素个数
        assert len(p.plot_data) == np.prod(flat_array.shape)

        # 检查绘图数据中的"x"变量
        x = p.plot_data["x"]
        # 生成预期的"x"数据，使用 np.arange 生成等差数组
        expected_x = np.arange(flat_array.shape[0])
        assert_array_equal(x, expected_x)

        # 检查绘图数据中的"y"变量
        y = p.plot_data["y"]
        # "y"数据应与输入的平坦数组相同
        expected_y = flat_array
        assert_array_equal(y, expected_y)

        # 断言变量字典中的"x"、"y"都为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None

    # 定义测试方法，用于测试平坦格式列表变量的处理
    def test_flat_list_variables(self, flat_list):
        # 创建关系绘图器对象
        p = _RelationalPlotter()
        # 将数据分配给关系绘图器对象
        p.assign_variables(data=flat_list)
        # 断言输入格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表包含 ["x", "y"]
        assert list(p.variables) == ["x", "y"]
        # 断言绘图数据的长度等于平坦列表的长度
        assert len(p.plot_data) == len(flat_list)

        # 检查绘图数据中的"x"变量
        x = p.plot_data["x"]
        # 生成预期的"x"数据，使用 np.arange 生成等差数组
        expected_x = np.arange(len(flat_list))
        assert_array_equal(x, expected_x)

        # 检查绘图数据中的"y"变量
        y = p.plot_data["y"]
        # "y"数据应与输入的平坦列表相同
        expected_y = flat_list
        assert_array_equal(y, expected_y)

        # 断言变量字典中的"x"、"y"都为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None

    # 定义测试方法，用于测试平坦格式系列变量的处理
    def test_flat_series_variables(self, flat_series):
        # 创建关系绘图器对象
        p = _RelationalPlotter()
        # 将数据分配给关系绘图器对象
        p.assign_variables(data=flat_series)
        # 断言输入格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表包含 ["x", "y"]
        assert list(p.variables) == ["x", "y"]
        # 断言绘图数据的长度等于平坦系列的长度
        assert len(p.plot_data) == len(flat_series)

        # 检查绘图数据中的"x"变量
        x = p.plot_data["x"]
        # 生成预期的"x"数据，使用 flat_series 的索引
        expected_x = flat_series.index
        assert_array_equal(x, expected_x)

        # 检查绘图数据中的"y"变量
        y = p.plot_data["y"]
        # "y"数据应与输入的平坦系列相同
        expected_y = flat_series
        assert_array_equal(y, expected_y)

        # 断言变量字典中的"x"为 flat_series 的索引名，"y"为 flat_series 的名称
        assert p.variables["x"] is flat_series.index.name
        assert p.variables["y"] is flat_series.name
    # 测试函数：对宽格式数据列表变量进行测试
    def test_wide_list_of_series_variables(self, wide_list_of_series):

        # 创建_RelationalPlotter对象实例
        p = _RelationalPlotter()
        # 将数据分配给_RelationalPlotter对象
        p.assign_variables(data=wide_list_of_series)
        # 断言输入格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表为["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 计算数据块数
        chunks = len(wide_list_of_series)
        # 计算每个数据块的最大长度
        chunk_size = max(len(l) for l in wide_list_of_series)

        # 断言plot_data的长度符合预期
        assert len(p.plot_data) == chunks * chunk_size

        # 获取索引的并集
        index_union = np.unique(
            np.concatenate([s.index for s in wide_list_of_series])
        )

        # 获取x轴数据
        x = p.plot_data["x"]
        # 生成期望的x轴数据
        expected_x = np.tile(index_union, chunks)
        assert_array_equal(x, expected_x)

        # 获取y轴数据
        y = p.plot_data["y"]
        # 生成期望的y轴数据
        expected_y = np.concatenate([
            s.reindex(index_union) for s in wide_list_of_series
        ])
        assert_array_equal(y, expected_y)

        # 获取hue数据
        hue = p.plot_data["hue"]
        # 获取各序列的名称列表
        series_names = [s.name for s in wide_list_of_series]
        # 生成期望的hue数据
        expected_hue = np.repeat(series_names, chunk_size)
        assert_array_equal(hue, expected_hue)

        # 获取style数据
        style = p.plot_data["style"]
        # 生成期望的style数据
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量的值为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    # 测试函数：对宽格式数据数组列表变量进行测试
    def test_wide_list_of_arrays_variables(self, wide_list_of_arrays):

        # 创建_RelationalPlotter对象实例
        p = _RelationalPlotter()
        # 将数据分配给_RelationalPlotter对象
        p.assign_variables(data=wide_list_of_arrays)
        # 断言输入格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表为["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 计算数据块数
        chunks = len(wide_list_of_arrays)
        # 计算每个数据块的最大长度
        chunk_size = max(len(l) for l in wide_list_of_arrays)

        # 断言plot_data的长度符合预期
        assert len(p.plot_data) == chunks * chunk_size

        # 获取x轴数据
        x = p.plot_data["x"]
        # 生成期望的x轴数据
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        # 获取y轴数据并去除NaN值
        y = p.plot_data["y"].dropna()
        # 生成期望的y轴数据
        expected_y = np.concatenate(wide_list_of_arrays)
        assert_array_equal(y, expected_y)

        # 获取hue数据
        hue = p.plot_data["hue"]
        # 生成期望的hue数据
        expected_hue = np.repeat(np.arange(chunks), chunk_size)
        assert_array_equal(hue, expected_hue)

        # 获取style数据
        style = p.plot_data["style"]
        # 生成期望的style数据
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量的值为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None
    # 测试广泛列表变量的情况
    def test_wide_list_of_list_variables(self, wide_list_of_lists):
        # 创建 _RelationalPlotter 的实例
        p = _RelationalPlotter()
        # 分配变量到 plotter 实例中，使用 wide_list_of_lists 作为数据
        p.assign_variables(data=wide_list_of_lists)
        # 断言输入格式为 "wide"
        assert p.input_format == "wide"
        # 断言变量列表为 ["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 计算 wide_list_of_lists 的长度作为 chunks
        chunks = len(wide_list_of_lists)
        # 计算 wide_list_of_lists 中最大子列表的长度作为 chunk_size
        chunk_size = max(len(l) for l in wide_list_of_lists)

        # 断言 plot_data 的长度等于 chunks * chunk_size
        assert len(p.plot_data) == chunks * chunk_size

        # 获取 plot_data 中的 "x" 列
        x = p.plot_data["x"]
        # 生成预期的 x 值，使用 np.tile 来扩展 np.arange(chunk_size) 的内容
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        # 获取 plot_data 中的 "y" 列，并删除其中的 NaN 值
        y = p.plot_data["y"].dropna()
        # 生成预期的 y 值，使用 np.concatenate 将 wide_list_of_lists 中的所有子列表连接起来
        expected_y = np.concatenate(wide_list_of_lists)
        assert_array_equal(y, expected_y)

        # 获取 plot_data 中的 "hue" 列
        hue = p.plot_data["hue"]
        # 生成预期的 hue 值，使用 np.repeat 将列表的元素重复 chunk_size 次
        expected_hue = np.repeat(np.arange(chunks), chunk_size)
        assert_array_equal(hue, expected_hue)

        # 获取 plot_data 中的 "style" 列
        style = p.plot_data["style"]
        # style 应当与 expected_hue 相等
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言 p.variables 中的 "x", "y", "hue", "style" 均为 None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    # 测试广泛字典变量的情况
    def test_wide_dict_of_series_variables(self, wide_dict_of_series):
        # 创建 _RelationalPlotter 的实例
        p = _RelationalPlotter()
        # 分配变量到 plotter 实例中，使用 wide_dict_of_series 作为数据
        p.assign_variables(data=wide_dict_of_series)
        # 断言输入格式为 "wide"
        assert p.input_format == "wide"
        # 断言变量列表为 ["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 计算 wide_dict_of_series 的长度作为 chunks
        chunks = len(wide_dict_of_series)
        # 计算 wide_dict_of_series 中最大 Series 的长度作为 chunk_size
        chunk_size = max(len(l) for l in wide_dict_of_series.values())

        # 断言 plot_data 的长度等于 chunks * chunk_size
        assert len(p.plot_data) == chunks * chunk_size

        # 获取 plot_data 中的 "x" 列
        x = p.plot_data["x"]
        # 生成预期的 x 值，使用 np.tile 来扩展 np.arange(chunk_size) 的内容
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        # 获取 plot_data 中的 "y" 列，并删除其中的 NaN 值
        y = p.plot_data["y"].dropna()
        # 生成预期的 y 值，使用 np.concatenate 将 wide_dict_of_series 中所有 Series 的值连接起来
        expected_y = np.concatenate(list(wide_dict_of_series.values()))
        assert_array_equal(y, expected_y)

        # 获取 plot_data 中的 "hue" 列
        hue = p.plot_data["hue"]
        # 生成预期的 hue 值，使用 np.repeat 将 wide_dict_of_series 的键重复 chunk_size 次
        expected_hue = np.repeat(list(wide_dict_of_series), chunk_size)
        assert_array_equal(hue, expected_hue)

        # 获取 plot_data 中的 "style" 列
        style = p.plot_data["style"]
        # style 应当与 expected_hue 相等
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言 p.variables 中的 "x", "y", "hue", "style" 均为 None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None
    # 测试宽格式数据的字典数组变量情况
    def test_wide_dict_of_arrays_variables(self, wide_dict_of_arrays):

        # 创建一个_RelationalPlotter实例
        p = _RelationalPlotter()
        # 将数据分配给_RelationalPlotter实例的变量
        p.assign_variables(data=wide_dict_of_arrays)
        # 断言输入数据格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表为["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 计算宽格式数据中的块数
        chunks = len(wide_dict_of_arrays)
        # 计算每个块的大小，即最大列表长度
        chunk_size = max(len(l) for l in wide_dict_of_arrays.values())

        # 断言绘图数据的长度是否正确
        assert len(p.plot_data) == chunks * chunk_size

        # 提取并验证绘图数据中的"x"列
        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        # 提取并验证绘图数据中的"y"列（去除NaN值）
        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(list(wide_dict_of_arrays.values()))
        assert_array_equal(y, expected_y)

        # 提取并验证绘图数据中的"hue"列
        hue = p.plot_data["hue"]
        expected_hue = np.repeat(list(wide_dict_of_arrays), chunk_size)
        assert_array_equal(hue, expected_hue)

        # 提取并验证绘图数据中的"style"列
        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量字典中的特定变量为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    # 测试宽格式数据的字典列表变量情况
    def test_wide_dict_of_lists_variables(self, wide_dict_of_lists):

        # 创建一个_RelationalPlotter实例
        p = _RelationalPlotter()
        # 将数据分配给_RelationalPlotter实例的变量
        p.assign_variables(data=wide_dict_of_lists)
        # 断言输入数据格式为"wide"
        assert p.input_format == "wide"
        # 断言变量列表为["x", "y", "hue", "style"]
        assert list(p.variables) == ["x", "y", "hue", "style"]

        # 计算宽格式数据中的块数
        chunks = len(wide_dict_of_lists)
        # 计算每个块的大小，即最大列表长度
        chunk_size = max(len(l) for l in wide_dict_of_lists.values())

        # 断言绘图数据的长度是否正确
        assert len(p.plot_data) == chunks * chunk_size

        # 提取并验证绘图数据中的"x"列
        x = p.plot_data["x"]
        expected_x = np.tile(np.arange(chunk_size), chunks)
        assert_array_equal(x, expected_x)

        # 提取并验证绘图数据中的"y"列（去除NaN值）
        y = p.plot_data["y"].dropna()
        expected_y = np.concatenate(list(wide_dict_of_lists.values()))
        assert_array_equal(y, expected_y)

        # 提取并验证绘图数据中的"hue"列
        hue = p.plot_data["hue"]
        expected_hue = np.repeat(list(wide_dict_of_lists), chunk_size)
        assert_array_equal(hue, expected_hue)

        # 提取并验证绘图数据中的"style"列
        style = p.plot_data["style"]
        expected_style = expected_hue
        assert_array_equal(style, expected_style)

        # 断言变量字典中的特定变量为None
        assert p.variables["x"] is None
        assert p.variables["y"] is None
        assert p.variables["hue"] is None
        assert p.variables["style"] is None

    # 测试简单的关系绘图情况
    def test_relplot_simple(self, long_df):

        # 生成一个关系图对象g，使用长格式数据，x轴为"x"列，y轴为"y"列，绘制散点图
        g = relplot(data=long_df, x="x", y="y", kind="scatter")
        # 从散点图对象中获取x坐标和y坐标的偏移量
        x, y = g.ax.collections[0].get_offsets().T
        # 断言x坐标数组与长格式数据中的"x"列相等
        assert_array_equal(x, long_df["x"])
        # 断言y坐标数组与长格式数据中的"y"列相等
        assert_array_equal(y, long_df["y"])

        # 生成一个关系图对象g，使用长格式数据，x轴为"x"列，y轴为"y"列，绘制线图
        g = relplot(data=long_df, x="x", y="y", kind="line")
        # 从线图对象中获取x坐标和y坐标数据
        x, y = g.ax.lines[0].get_xydata().T
        # 通过对"x"列分组并计算平均值，生成期望的x坐标数组
        expected = long_df.groupby("x").y.mean()
        assert_array_equal(x, expected.index)
        # 断言y坐标数组与期望的y坐标值数组接近（使用pytest.approx检查）
        assert y == pytest.approx(expected.values)

        # 使用pytest.raises检查是否引发值错误异常
        with pytest.raises(ValueError):
            g = relplot(data=long_df, x="x", y="y", kind="not_a_kind")
    # 测试复杂情况下的 relplot 函数
    def test_relplot_complex(self, long_df):

        # 对于每种语义 ["hue", "size", "style"] 分别进行测试
        for sem in ["hue", "size", "style"]:
            # 创建一个 relplot 图形对象 g，使用 long_df 数据，设置 x 和 y 轴，以及当前语义 sem
            g = relplot(data=long_df, x="x", y="y", **{sem: "a"})
            # 从图形对象 g 中获取第一个集合的偏移量，并分别提取 x 和 y 坐标
            x, y = g.ax.collections[0].get_offsets().T
            # 断言 x 数组与 long_df 中的 "x" 列相等
            assert_array_equal(x, long_df["x"])
            # 断言 y 数组与 long_df 中的 "y" 列相等
            assert_array_equal(y, long_df["y"])

        # 对于每种语义 ["hue", "size", "style"] 再次进行测试
        for sem in ["hue", "size", "style"]:
            # 创建一个 relplot 图形对象 g，使用 long_df 数据，设置 x 和 y 轴，以及列 "c" 和当前语义 sem
            g = relplot(
                data=long_df, x="x", y="y", col="c", **{sem: "a"}
            )
            # 根据列 "c" 对 long_df 进行分组
            grouped = long_df.groupby("c")
            # 遍历每个分组 grp_df 和 g 的平坦化轴 ax
            for (_, grp_df), ax in zip(grouped, g.axes.flat):
                # 从每个轴对象 ax 中获取第一个集合的偏移量，并分别提取 x 和 y 坐标
                x, y = ax.collections[0].get_offsets().T
                # 断言 x 数组与 grp_df 中的 "x" 列相等
                assert_array_equal(x, grp_df["x"])
                # 断言 y 数组与 grp_df 中的 "y" 列相等
                assert_array_equal(y, grp_df["y"])

        # 对于每种语义 ["size", "style"] 再次进行测试
        for sem in ["size", "style"]:
            # 创建一个 relplot 图形对象 g，使用 long_df 数据，设置 x 和 y 轴，颜色 "b"、列 "c"，以及当前语义 sem
            g = relplot(
                data=long_df, x="x", y="y", hue="b", col="c", **{sem: "a"}
            )
            # 根据列 "c" 对 long_df 进行分组
            grouped = long_df.groupby("c")
            # 遍历每个分组 grp_df 和 g 的平坦化轴 ax
            for (_, grp_df), ax in zip(grouped, g.axes.flat):
                # 从每个轴对象 ax 中获取第一个集合的偏移量，并分别提取 x 和 y 坐标
                x, y = ax.collections[0].get_offsets().T
                # 断言 x 数组与 grp_df 中的 "x" 列相等
                assert_array_equal(x, grp_df["x"])
                # 断言 y 数组与 grp_df 中的 "y" 列相等
                assert_array_equal(y, grp_df["y"])

        # 对于每种语义 ["hue", "size", "style"] 再次进行测试
        for sem in ["hue", "size", "style"]:
            # 创建一个 relplot 图形对象 g，使用 long_df 数据并根据列 "c" 和 "b" 排序，设置 x 和 y 轴，行 "c" 和列 "b"，以及当前语义 sem
            g = relplot(
                data=long_df.sort_values(["c", "b"]),
                x="x", y="y", col="b", row="c", **{sem: "a"}
            )
            # 根据列 ["c", "b"] 对 long_df 进行分组
            grouped = long_df.groupby(["c", "b"])
            # 遍历每个分组 grp_df 和 g 的平坦化轴 ax
            for (_, grp_df), ax in zip(grouped, g.axes.flat):
                # 从每个轴对象 ax 中获取第一个集合的偏移量，并分别提取 x 和 y 坐标
                x, y = ax.collections[0].get_offsets().T
                # 断言 x 数组与 grp_df 中的 "x" 列相等
                assert_array_equal(x, grp_df["x"])
                # 断言 y 数组与 grp_df 中的 "y" 列相等
                assert_array_equal(y, grp_df["y"])

    # 使用参数化测试，针对不同的 vector_type 进行 relplot 函数的测试
    @pytest.mark.parametrize("vector_type", ["series", "numpy", "list"])
    def test_relplot_vectors(self, long_df, vector_type):

        # 定义语义的映射关系
        semantics = dict(x="x", y="y", hue="f", col="c")
        # 根据 semantics 中的映射关系创建关键字参数 kws
        kws = {key: long_df[val] for key, val in semantics.items()}
        # 根据不同的 vector_type 转换 kws 中的值为 numpy 数组或列表
        if vector_type == "numpy":
            kws = {k: v.to_numpy() for k, v in kws.items()}
        elif vector_type == "list":
            kws = {k: v.to_list() for k, v in kws.items()}
        # 创建一个 relplot 图形对象 g，使用 long_df 数据和转换后的 kws
        g = relplot(data=long_df, **kws)
        # 根据列 "c" 对 long_df 进行分组
        grouped = long_df.groupby("c")
        # 断言 g 的轴字典长度与分组数目相等
        assert len(g.axes_dict) == len(grouped)
        # 遍历每个分组 grp_df 和 g 的平坦化轴 ax
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            # 从每个轴对象 ax 中获取第一个集合的偏移量，并分别提取 x 和 y 坐标
            x, y = ax.collections[0].get_offsets().T
            # 断言 x 数组与 grp_df 中的 "x" 列相等
            assert_array_equal(x, grp_df["x"])
            # 断言 y 数组与 grp_df 中的 "y" 列相等
            assert_array_equal(y, grp_df["y"])

    # 测试宽格式数据的 relplot 函数
    def test_relplot_wide(self, wide_df):

        # 创建一个 relplot 图形对象 g，使用 wide_df 数据
        g = relplot(data=wide_df)
        # 从图形对象 g 的主轴中获取第一个集合的偏移量，并分别提取 x 和 y 坐标
        x, y = g.ax.collections[0].get_offsets().T
        # 断言 y 数组与 wide_df 转置后的 numpy 数组扁平化后的结果相等
        assert_array_equal(y, wide_df.to_numpy().T.ravel())
        # 断言 g 的主轴的 y 轴标签为空
        assert not g.ax.get_ylabel()
    def test_relplot_hues(self, long_df):
        # 设定颜色调色板，用于绘制图形中不同分类的颜色
        palette = ["r", "b", "g"]
        # 创建关系图，设置 x 轴为 "x"，y 轴为 "y"，根据 "a" 列进行颜色编码，"b" 列进行样式编码，"c" 列分组
        g = relplot(
            x="x", y="y", hue="a", style="b", col="c",
            palette=palette, data=long_df
        )

        # 创建一个字典，将数据集中唯一的 "a" 列值与颜色调色板中的颜色对应起来
        palette = dict(zip(long_df["a"].unique(), palette))
        # 按照 "c" 列分组数据集
        grouped = long_df.groupby("c")
        # 遍历分组数据和图形对象中的轴
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            # 获取图形对象中的点集合
            points = ax.collections[0]
            # 根据组内 "a" 列的值获取预期的颜色编码列表
            expected_hues = [palette[val] for val in grp_df["a"]]
            # 断言图形对象中的点的颜色与预期颜色编码列表相符
            assert same_color(points.get_facecolors(), expected_hues)

    def test_relplot_sizes(self, long_df):
        # 设定大小列表，用于不同分类数据点的大小
        sizes = [5, 12, 7]
        # 创建关系图，设置 x 轴为 "x"，y 轴为 "y"，根据 "a" 列进行大小编码，"b" 列进行颜色编码，"c" 列分组
        g = relplot(
            data=long_df,
            x="x", y="y", size="a", hue="b", col="c",
            sizes=sizes,
        )

        # 创建一个字典，将数据集中唯一的 "a" 列值与大小列表中的大小对应起来
        sizes = dict(zip(long_df["a"].unique(), sizes))
        # 按照 "c" 列分组数据集
        grouped = long_df.groupby("c")
        # 遍历分组数据和图形对象中的轴
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            # 获取图形对象中的点集合
            points = ax.collections[0]
            # 根据组内 "a" 列的值获取预期的大小列表
            expected_sizes = [sizes[val] for val in grp_df["a"]]
            # 断言图形对象中的点的大小与预期大小列表相符
            assert_array_equal(points.get_sizes(), expected_sizes)

    def test_relplot_styles(self, long_df):
        # 设定标记样式列表，用于不同分类数据点的标记样式
        markers = ["o", "d", "s"]
        # 创建关系图，设置 x 轴为 "x"，y 轴为 "y"，根据 "a" 列进行样式编码，"b" 列进行颜色编码，"c" 列分组
        g = relplot(
            data=long_df,
            x="x", y="y", style="a", hue="b", col="c",
            markers=markers,
        )

        # 创建一个字典，将数据集中唯一的 "a" 列值与对应标记样式的路径对象对应起来
        paths = []
        for m in markers:
            m = mpl.markers.MarkerStyle(m)
            paths.append(m.get_path().transformed(m.get_transform()))
        paths = dict(zip(long_df["a"].unique(), paths))

        # 按照 "c" 列分组数据集
        grouped = long_df.groupby("c")
        # 遍历分组数据和图形对象中的轴
        for (_, grp_df), ax in zip(grouped, g.axes.flat):
            # 获取图形对象中的点集合
            points = ax.collections[0]
            # 根据组内 "a" 列的值获取预期的路径对象列表
            expected_paths = [paths[val] for val in grp_df["a"]]
            # 断言图形对象中的点的路径对象与预期路径对象列表相符
            assert self.paths_equal(points.get_paths(), expected_paths)

    def test_relplot_weighted_estimator(self, long_df):
        # 创建关系图，设置 x 轴为 "a"，y 轴为 "y"，根据 "x" 列进行权重计算，类型为线性图
        g = relplot(data=long_df, x="a", y="y", weights="x", kind="line")
        # 获取图形对象中线条的纵坐标数据
        ydata = g.ax.lines[0].get_ydata()
        # 遍历数据集中 "a" 列的分类顺序
        for i, level in enumerate(categorical_order(long_df["a"])):
            # 提取数据集中 "a" 列值为当前分类的子数据集
            pos_df = long_df[long_df["a"] == level]
            # 计算子数据集中 "y" 列的加权平均值，权重为 "x" 列的值
            expected = np.average(pos_df["y"], weights=pos_df["x"])
            # 断言图形对象中的线条纵坐标值与预期加权平均值接近
            assert ydata[i] == pytest.approx(expected)

    def test_relplot_stringy_numerics(self, long_df):
        # 将数据集中 "x" 列转换为字符串类型
        long_df["x_str"] = long_df["x"].astype(str)

        # 创建关系图，设置 x 轴为 "x"，y 轴为 "y"，根据 "x_str" 列进行颜色编码
        g = relplot(data=long_df, x="x", y="y", hue="x_str")
        # 获取图形对象中的点集合
        points = g.ax.collections[0]
        # 获取点集合的偏移量
        xys = points.get_offsets()
        # 获取偏移量的掩码
        mask = np.ma.getmask(xys)
        # 断言掩码中没有任何 True 值
        assert not mask.any()
        # 断言偏移量与数据集中的 "x" 和 "y" 列相等
        assert_array_equal(xys, long_df[["x", "y"]])

        # 创建关系图，设置 x 轴为 "x"，y 轴为 "y"，根据 "x_str" 列进行大小编码
        g = relplot(data=long_df, x="x", y="y", size="x_str")
        # 获取图形对象中的点集合
        points = g.ax.collections[0]
        # 获取点集合的偏移量
        xys = points.get_offsets()
        # 获取偏移量的掩码
        mask = np.ma.getmask(xys)
        # 断言掩码中没有任何 True 值
        assert not mask.any()
        # 断言偏移量与数据集中的 "x" 和 "y" 列相等
        assert_array_equal(xys, long_df[["x", "y"]])
    # 测试相对绘图（relplot）的图例是否正确显示，当不使用颜色编码时，期望图例为 None
    def test_relplot_legend(self, long_df):

        # 绘制不带颜色编码的相对绘图
        g = relplot(data=long_df, x="x", y="y")
        # 断言图例对象为 None
        assert g._legend is None

        # 绘制带有颜色编码的相对绘图
        g = relplot(data=long_df, x="x", y="y", hue="a")
        # 获取图例文本列表
        texts = [t.get_text() for t in g._legend.texts]
        # 期望的图例文本为长数据中变量 "a" 的唯一值
        expected_texts = long_df["a"].unique()
        assert_array_equal(texts, expected_texts)

        # 绘制同时使用颜色和大小编码的相对绘图
        g = relplot(data=long_df, x="x", y="y", hue="s", size="s")
        # 获取图例文本列表并断言其按字典序排序
        texts = [t.get_text() for t in g._legend.texts]
        assert_array_equal(texts, np.sort(texts))

        # 绘制带有颜色编码但不显示图例的相对绘图
        g = relplot(data=long_df, x="x", y="y", hue="a", legend=False)
        # 断言图例对象为 None
        assert g._legend is None

        # 根据数据 "b" 的唯一值数量创建调色板
        palette = color_palette("deep", len(long_df["b"].unique()))
        # 创建一个将数据 "a" 的值映射到数据 "b" 唯一值的字典
        a_like_b = dict(zip(long_df["a"].unique(), long_df["b"].unique()))
        # 将映射结果作为新列添加到长数据中
        long_df["a_like_b"] = long_df["a"].map(a_like_b)
        # 绘制使用线型图形式展示数据的相对绘图
        g = relplot(
            data=long_df,
            x="x", y="y", hue="b", style="a_like_b",
            palette=palette, kind="line", estimator=None,
        )
        # 获取图例对象的线条列表，并断言其颜色与调色板匹配
        lines = g._legend.get_lines()[1:]  # 去掉标题占位符
        for line, color in zip(lines, palette):
            assert line.get_color() == color

    # 测试相对绘图（relplot）是否正确显示未共享的坐标轴标签
    def test_relplot_unshared_axis_labels(self, long_df):

        col, row = "a", "b"
        # 绘制具有不共享坐标轴的相对绘图
        g = relplot(
            data=long_df, x="x", y="y", col=col, row=row,
            facet_kws=dict(sharex=False, sharey=False),
        )

        # 断言底部行的所有坐标轴标签为 "x"
        for ax in g.axes[-1, :].flat:
            assert ax.get_xlabel() == "x"
        # 断言非底部行的所有坐标轴标签为空字符串
        for ax in g.axes[:-1, :].flat:
            assert ax.get_xlabel() == ""
        # 断言左侧列的所有坐标轴标签为 "y"
        for ax in g.axes[:, 0].flat:
            assert ax.get_ylabel() == "y"
        # 断言非左侧列的所有坐标轴标签为空字符串
        for ax in g.axes[:, 1:].flat:
            assert ax.get_ylabel() == ""

    # 测试相对绘图（relplot）是否正确处理输入数据
    def test_relplot_data(self, long_df):

        # 绘制相对绘图，并将长数据转换为字典列表形式作为输入
        g = relplot(
            data=long_df.to_dict(orient="list"),
            x="x",
            y=long_df["y"].rename("y_var"),
            hue=long_df["a"].to_numpy(),
            col="c",
        )
        # 期望的数据列包括长数据中的所有列及额外的 "_hue_" 和 "y_var"
        expected_cols = set(long_df.columns.to_list() + ["_hue_", "y_var"])
        assert set(g.data.columns) == expected_cols
        # 断言 "y_var" 列的值与长数据中的 "y" 列相等
        assert_array_equal(g.data["y_var"], long_df["y"])
        # 断言 "_hue_" 列的值与长数据中的 "a" 列相等
        assert_array_equal(g.data["_hue_"], long_df["a"])

    # 测试相对绘图（relplot）是否正确处理分面变量冲突
    def test_facet_variable_collision(self, long_df):

        # 获取长数据中的 "c" 列数据
        col_data = long_df["c"]
        # 将 "size" 列添加到长数据中，与 "c" 列相同
        long_df = long_df.assign(size=col_data)

        # 绘制带有分面变量 "size" 的相对绘图
        g = relplot(
            data=long_df,
            x="x", y="y", col="size",
        )
        # 断言生成的子图形状为 (1, 数据 "c" 列的唯一值数量)
        assert g.axes.shape == (1, len(col_data.unique()))

    # 测试相对绘图（relplot）是否正确处理未使用的散点图变量
    def test_relplot_scatter_unused_variables(self, long_df):

        # 捕获使用未使用的参数 "units" 时的用户警告
        with pytest.warns(UserWarning, match="The `units` parameter"):
            g = relplot(long_df, x="x", y="y", units="a")
        # 断言生成的图形对象不为空
        assert g.ax is not None

        # 捕获使用未使用的参数 "weights" 时的用户警告
        with pytest.warns(UserWarning, match="The `weights` parameter"):
            g = relplot(long_df, x="x", y="y", weights="x")
        # 断言生成的图形对象不为空
        assert g.ax is not None
    # 测试函数：测试相对图的 x 和 y 轴的关系图中的关键字参数移除情况
    def test_ax_kwarg_removal(self, long_df):
        # 创建一个新的图形和坐标轴对象
        f, ax = plt.subplots()
        # 检查是否会发出 UserWarning 警告
        with pytest.warns(UserWarning):
            # 绘制相对图并返回图形对象 g
            g = relplot(data=long_df, x="x", y="y", ax=ax)
        # 断言：坐标轴对象中不包含任何集合对象
        assert len(ax.collections) == 0
        # 断言：图形对象 g 的坐标轴对象中包含大于 0 个集合对象
        assert len(g.ax.collections) > 0
    
    # 测试函数：测试图例是否没有偏移
    def test_legend_has_no_offset(self, long_df):
        # 绘制相对图 g，并指定色调为 long_df["z"] + 1e8
        g = relplot(data=long_df, x="x", y="y", hue=long_df["z"] + 1e8)
        # 遍历图例中的文本对象
        for text in g.legend.texts:
            # 断言：图例文本转换为浮点数后大于 1e7
            assert float(text.get_text()) > 1e7
    
    # 测试函数：测试线图中虚线的使用
    def test_lineplot_2d_dashes(self, long_df):
        # 绘制线图 ax，并指定虚线样式 [(5, 5), (10, 10)]
        ax = lineplot(data=long_df[["x", "y"]], dashes=[(5, 5), (10, 10)])
        # 遍历图中的每一条线
        for line in ax.get_lines():
            # 断言：判断线条是否为虚线
            assert line.is_dashed()
    
    # 测试函数：测试图例中色调属性的属性
    def test_legend_attributes_hue(self, long_df):
        # 定义图形关键字参数 kws
        kws = {"s": 50, "linewidth": 1, "marker": "X"}
        # 绘制相对图 g，指定 x 和 y 轴，以及色调为 "a"，并传入 kws 中的参数
        g = relplot(long_df, x="x", y="y", hue="a", **kws)
        # 获取默认调色板
        palette = color_palette()
        # 遍历图例中的每一个标记对象
        for i, pt in enumerate(get_legend_handles(g.legend)):
            # 断言：判断标记的颜色是否与调色板中对应的颜色相同
            assert same_color(pt.get_color(), palette[i])
            # 断言：判断标记的大小是否为 s 的平方根
            assert pt.get_markersize() == np.sqrt(kws["s"])
            # 断言：判断标记的边缘宽度是否为 linewidth
            assert pt.get_markeredgewidth() == kws["linewidth"]
            # 如果 matplotlib 版本不早于 "3.7.0"
            if not _version_predates(mpl, "3.7.0"):
                # 断言：判断标记的类型是否为 marker 中指定的类型
                assert pt.get_marker() == kws["marker"]
    
    # 测试函数：测试图例中样式属性的属性
    def test_legend_attributes_style(self, long_df):
        # 定义图形关键字参数 kws
        kws = {"s": 50, "linewidth": 1, "color": "r"}
        # 绘制相对图 g，指定 x 和 y 轴，以及样式为 "a"，并传入 kws 中的参数
        g = relplot(long_df, x="x", y="y", style="a", **kws)
        # 遍历图例中的每一个标记对象
        for pt in get_legend_handles(g.legend):
            # 断言：判断标记的大小是否为 s 的平方根
            assert pt.get_markersize() == np.sqrt(kws["s"])
            # 断言：判断标记的边缘宽度是否为 linewidth
            assert pt.get_markeredgewidth() == kws["linewidth"]
            # 断言：判断标记的颜色是否与 "r" 相同
            assert same_color(pt.get_color(), "r")
    
    # 测试函数：测试图例中色调和样式属性的属性
    def test_legend_attributes_hue_and_style(self, long_df):
        # 定义图形关键字参数 kws
        kws = {"s": 50, "linewidth": 1}
        # 绘制相对图 g，指定 x 和 y 轴，色调为 "a"，样式为 "b"，并传入 kws 中的参数
        g = relplot(long_df, x="x", y="y", hue="a", style="b", **kws)
        # 遍历图例中的每一个标记对象
        for pt in get_legend_handles(g.legend):
            # 如果标记的标签不是 "a" 或 "b"
            if pt.get_label() not in ["a", "b"]:
                # 断言：判断标记的大小是否为 s 的平方根
                assert pt.get_markersize() == np.sqrt(kws["s"])
                # 断言：判断标记的边缘宽度是否为 linewidth
                assert pt.get_markeredgewidth() == kws["linewidth"]
# 创建一个测试类 `TestLinePlotter`，继承了 `SharedAxesLevelTests` 和 `Helpers` 两个类
class TestLinePlotter(SharedAxesLevelTests, Helpers):

    # 定义一个静态方法 `func`，指向 `lineplot` 函数
    func = staticmethod(lineplot)

    # 定义一个实例方法 `get_last_color`，接收参数 `ax`，返回最后一条线的颜色
    def get_last_color(self, ax):
        return to_rgba(ax.lines[-1].get_color())

    # 定义一个测试方法 `test_legend_no_semantics`，接收参数 `long_df`
    def test_legend_no_semantics(self, long_df):
        # 调用 `lineplot` 函数绘制图表，指定 x 轴为 "x"，y 轴为 "y"
        ax = lineplot(long_df, x="x", y="y")
        # 获取图例的句柄和标签
        handles, _ = ax.get_legend_handles_labels()
        # 断言图例句柄为空列表
        assert handles == []

    # 定义一个测试方法 `test_legend_hue_categorical`，接收参数 `long_df` 和 `levels`
    def test_legend_hue_categorical(self, long_df, levels):
        # 调用 `lineplot` 函数绘制图表，指定 x 轴为 "x"，y 轴为 "y"，颜色编码为 "a"
        ax = lineplot(long_df, x="x", y="y", hue="a")
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取图例句柄的颜色列表
        colors = [h.get_color() for h in handles]
        # 断言图例标签与参数 `levels` 中 "a" 键对应的值相等
        assert labels == levels["a"]
        # 断言图例颜色列表与使用 `color_palette` 函数生成的颜色列表相等
        assert colors == color_palette(n_colors=len(labels))

    # 定义一个测试方法 `test_legend_hue_and_style_same`，接收参数 `long_df` 和 `levels`
    def test_legend_hue_and_style_same(self, long_df, levels):
        # 调用 `lineplot` 函数绘制图表，指定 x 轴为 "x"，y 轴为 "y"，颜色编码为 "a"，线条样式为 "a"，显示标记
        ax = lineplot(long_df, x="x", y="y", hue="a", style="a", markers=True)
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取图例句柄的颜色列表
        colors = [h.get_color() for h in handles]
        # 获取图例句柄的标记列表
        markers = [h.get_marker() for h in handles]
        # 断言图例标签与参数 `levels` 中 "a" 键对应的值相等
        assert labels == levels["a"]
        # 断言图例颜色列表与使用 `color_palette` 函数生成的颜色列表相等
        assert colors == color_palette(n_colors=len(labels))
        # 断言图例标记列表与使用 `unique_markers` 函数生成的唯一标记列表相等
        assert markers == unique_markers(len(labels))

    # 定义一个测试方法 `test_legend_hue_and_style_diff`，接收参数 `long_df` 和 `levels`
    def test_legend_hue_and_style_diff(self, long_df, levels):
        # 调用 `lineplot` 函数绘制图表，指定 x 轴为 "x"，y 轴为 "y"，颜色编码为 "a"，线条样式为 "b"，显示标记
        ax = lineplot(long_df, x="x", y="y", hue="a", style="b", markers=True)
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取图例句柄的颜色列表
        colors = [h.get_color() for h in handles]
        # 获取图例句柄的标记列表
        markers = [h.get_marker() for h in handles]
        # 预期的图例标签列表，包含 "a"，参数 `levels` 中 "a" 和 "b" 键对应的值
        expected_labels = ["a", *levels["a"], "b", *levels["b"]]
        # 预期的图例颜色列表，包含白色、使用 `color_palette` 函数生成的颜色列表，以及 ".2" 的列表
        expected_colors = [
            "w", *color_palette(n_colors=len(levels["a"])),
            "w", *[".2" for _ in levels["b"]],
        ]
        # 预期的图例标记列表，包含空字符串、使用 `unique_markers` 函数生成的标记列表
        expected_markers = [
            "", *["None" for _ in levels["a"]]
            + [""] + unique_markers(len(levels["b"]))
        ]
        # 断言图例标签与预期的图例标签列表相等
        assert labels == expected_labels
        # 断言图例颜色列表与预期的图例颜色列表相等
        assert colors == expected_colors
        # 断言图例标记列表与预期的图例标记列表相等
        assert markers == expected_markers

    # 定义一个测试方法 `test_legend_hue_and_size_same`，接收参数 `long_df` 和 `levels`
    def test_legend_hue_and_size_same(self, long_df, levels):
        # 调用 `lineplot` 函数绘制图表，指定 x 轴为 "x"，y 轴为 "y"，颜色编码为 "a"，线条宽度为 "a"
        ax = lineplot(long_df, x="x", y="y", hue="a", size="a")
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取图例句柄的颜色列表
        colors = [h.get_color() for h in handles]
        # 获取图例句柄的线条宽度列表
        widths = [h.get_linewidth() for h in handles]
        # 断言图例标签与参数 `levels` 中 "a" 键对应的值相等
        assert labels == levels["a"]
        # 断言图例颜色列表与使用 `color_palette` 函数生成的颜色列表相等
        assert colors == color_palette(n_colors=len(levels["a"]))
        # 生成预期的线条宽度列表，使用 `mpl.rcParams["lines.linewidth"]` 乘以线条宽度调整
        expected_widths = [
            w * mpl.rcParams["lines.linewidth"]
            for w in np.linspace(2, 0.5, len(levels["a"]))
        ]
        # 断言图例线条宽度列表与预期的线条宽度列表相等
        assert widths == expected_widths

    # 使用 `pytest.mark.parametrize` 装饰器标记参数化测试
    @pytest.mark.parametrize("var", ["hue", "size", "style"])
    # 定义一个测试方法 `test_legend_numerical_full`，接收参数 `long_df` 和 `var`
    def test_legend_numerical_full(self, long_df, var):
        # 生成随机数据 `x` 和 `y`
        x, y = np.random.randn(2, 40)
        # 生成重复的数列 `z`，范围是 0 到 19，重复两次
        z = np.tile(np.arange(20), 2)
        # 调用 `lineplot` 函数绘制图表，指定 x 轴为 `x`，y 轴为 `y`，根据参数 `var` 指定图例的编码
        ax = lineplot(x=x, y=y, **{var: z}, legend="full")
        # 获取图例的句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 断言图例标签与重复数列 `z` 的字符串表示列表相等
        assert labels == [str(z_i) for z_i in sorted(set(z))]
    # 测试函数：检查 legend 参数为 "brief" 时的行为
    def test_legend_numerical_brief(self, var):
        # 生成两组随机数据
        x, y = np.random.randn(2, 40)
        # 创建重复的数组 [0, 1, ..., 19, 0, 1, ..., 19]
        z = np.tile(np.arange(20), 2)

        # 调用 lineplot 函数绘制线图，使用 var 参数作为关键字参数传递
        ax = lineplot(x=x, y=y, **{var: z}, legend="brief")
        # 获取图例句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 根据 var 的值检查标签的内容
        if var == "style":
            assert labels == [str(z_i) for z_i in sorted(set(z))]
        else:
            assert labels == ["0", "4", "8", "12", "16"]

    # 测试函数：检查 legend 参数为非法值时是否抛出 ValueError 异常
    def test_legend_value_error(self, long_df):
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=r"`legend` must be"):
            lineplot(long_df, x="x", y="y", hue="a", legend="bad_value")

    # 参数化测试函数：检查对数标准化时 legend 标签的比例关系
    @pytest.mark.parametrize("var", ["hue", "size"])
    def test_legend_log_norm(self, var):
        # 生成两组随机数据
        x, y = np.random.randn(2, 40)
        # 创建重复的数组 [0, 1, ..., 19, 0, 1, ..., 19]
        z = np.tile(np.arange(20), 2)

        # 创建对数标准化对象
        norm = mpl.colors.LogNorm()
        # 调用 lineplot 函数绘制线图，传入 var 和对应的标准化参数
        ax = lineplot(x=x, y=y, **{var: z + 1, f"{var}_norm": norm})
        # 获取图例句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 检查第二个标签与第一个标签的比例关系是否为 10
        assert float(labels[1]) / float(labels[0]) == 10

    # 参数化测试函数：检查二元变量的 legend 标签
    @pytest.mark.parametrize("var", ["hue", "size"])
    def test_legend_binary_var(self, var):
        # 生成两组随机数据
        x, y = np.random.randn(2, 40)
        # 创建重复的数组 [0, 1, ..., 19, 0, 1, ..., 19]
        z = np.tile(np.arange(20), 2)

        # 调用 lineplot 函数绘制线图，传入 var 参数作为关键字参数
        ax = lineplot(x=x, y=y, hue=z % 2)
        # 获取图例句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 检查标签是否为 ["0", "1"]
        assert labels == ["0", "1"]

    # 参数化测试函数：检查二元数值变量的 brief 类型 legend 标签
    @pytest.mark.parametrize("var", ["hue", "size"])
    def test_legend_binary_numberic_brief(self, long_df, var):
        # 调用 lineplot 函数绘制线图，传入 var 和 legend 参数作为关键字参数
        ax = lineplot(long_df, x="x", y="y", **{var: "f"}, legend="brief")
        # 获取图例句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 期望的标签列表
        expected_labels = ['0.20', '0.22', '0.24', '0.26', '0.28']
        # 检查标签是否符合预期
        assert labels == expected_labels

    # 测试函数：检查权重参数在非聚合数据上的正确性
    def test_weights(self, long_df):
        # 调用 lineplot 函数绘制线图，使用权重参数
        ax = lineplot(long_df, x="a", y="y", weights="x")
        # 获取第一条线的 y 数据
        vals = ax.lines[0].get_ydata()
        # 遍历分类顺序，计算加权平均值，并与预期值进行比较
        for i, level in enumerate(categorical_order(long_df["a"])):
            pos_df = long_df[long_df["a"] == level]
            expected = np.average(pos_df["y"], weights=pos_df["x"])
            assert vals[i] == pytest.approx(expected)

    # 测试函数：检查非聚合数据的正确绘制
    def test_non_aggregated_data(self):
        # 创建简单的 x 和 y 数据
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        # 调用 lineplot 函数绘制线图
        ax = lineplot(x=x, y=y)
        # 获取第一条线的 x 和 y 数据，并与原始数据进行比较
        line, = ax.lines
        assert_array_equal(line.get_xdata(), x)
        assert_array_equal(line.get_ydata(), y)
    # 测试长格式数据的绘图方向调整
    def test_orient(self, long_df):
        
        # 从长格式数据中删除"x"列，并重命名"s"列为"y"，"y"列为"x"
        long_df = long_df.drop("x", axis=1).rename(columns={"s": "y", "y": "x"})

        # 创建一个包含单个子图的图形对象
        ax1 = plt.figure().subplots()
        
        # 绘制线图，以"x"为横轴，"y"为纵轴，指定绘图方向为"y"，误差条样式为"sd"
        lineplot(data=long_df, x="x", y="y", orient="y", errorbar="sd")
        
        # 断言线条数量与集合数量相等
        assert len(ax1.lines) == len(ax1.collections)
        
        # 获取第一条线条对象
        line, = ax1.lines
        
        # 通过分组"y"列计算"x"列的均值，并重置索引
        expected = long_df.groupby("y").agg({"x": "mean"}).reset_index()
        
        # 断言线条的x坐标数据与期望的"x"列数据近似相等
        assert_array_almost_equal(line.get_xdata(), expected["x"])
        
        # 断言线条的y坐标数据与期望的"y"列数据近似相等
        assert_array_almost_equal(line.get_ydata(), expected["y"])
        
        # 获取第一个集合（误差条）的y坐标
        ribbon_y = ax1.collections[0].get_paths()[0].vertices[:, 1]
        
        # 断言唯一化后的误差条y坐标与排序后的"y"列唯一值相等
        assert_array_equal(np.unique(ribbon_y), long_df["y"].sort_values().unique())

        # 创建一个包含单个子图的图形对象
        ax2 = plt.figure().subplots()
        
        # 绘制线图，以"x"为横轴，"y"为纵轴，指定绘图方向为"y"，误差条样式为"sd"，错误样式为"bars"
        lineplot(
            data=long_df, x="x", y="y", orient="y", errorbar="sd", err_style="bars"
        )
        
        # 获取第一个集合（误差条）的线段
        segments = ax2.collections[0].get_segments()
        
        # 遍历并断言排序后的"y"列的唯一值与每个线段的y坐标相等
        for i, val in enumerate(sorted(long_df["y"].unique())):
            assert (segments[i][:, 1] == val).all()

        # 使用pytest断言引发的异常是否包含特定消息
        with pytest.raises(ValueError, match="`orient` must be either 'x' or 'y'"):
            lineplot(long_df, x="y", y="x", orient="bad")

    # 测试对数刻度的绘图
    def test_log_scale(self):

        # 创建一个包含单个子图的图形对象，获取图形和轴对象
        f, ax = plt.subplots()
        
        # 设置x轴为对数刻度
        
        ax.set_xscale("log")

        # 设置数据点
        x = [1, 10, 100]
        y = [1, 2, 3]

        # 绘制线图，传入x和y数据
        lineplot(x=x, y=y)
        
        # 获取第一条线条对象
        line = ax.lines[0]
        
        # 断言线条的x坐标数据与x数据相等
        assert_array_equal(line.get_xdata(), x)
        
        # 断言线条的y坐标数据与y数据相等
        assert_array_equal(line.get_ydata(), y)

        # 创建一个包含单个子图的图形对象，获取图形和轴对象
        f, ax = plt.subplots()
        
        # 同时设置x轴和y轴为对数刻度
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 设置数据点
        x = [1, 1, 2, 2]
        y = [1, 10, 1, 100]

        # 绘制线图，传入x和y数据，错误样式为"bars"，误差条为("pi", 100)
        lineplot(x=x, y=y, err_style="bars", errorbar=("pi", 100))
        
        # 获取第一条线条对象
        line = ax.lines[0]
        
        # 断言线条的y坐标数据的第二个值为10
        assert line.get_ydata()[1] == 10

        # 获取误差条的线段
        ebars = ax.collections[0].get_segments()
        
        # 断言第一个误差条线段的y坐标与y的前两个值相等
        assert_array_equal(ebars[0][:, 1], y[:2])
        
        # 断言第二个误差条线段的y坐标与y的后两个值相等
        assert_array_equal(ebars[1][:, 1], y[2:])

    # 测试轴标签的设置
    def test_axis_labels(self, long_df):

        # 创建一个包含两个子图的图形对象，并获取轴对象
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        # 创建_LinePlotter对象，传入数据和变量字典
        p = _LinePlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )

        # 在第一个子图上绘制数据
        p.plot(ax1, {})
        
        # 断言第一个子图的x轴标签为"x"
        assert ax1.get_xlabel() == "x"
        
        # 断言第一个子图的y轴标签为"y"
        assert ax1.get_ylabel() == "y"

        # 在第二个子图上绘制数据
        p.plot(ax2, {})
        
        # 断言第二个子图的x轴标签为"x"
        assert ax2.get_xlabel() == "x"
        
        # 断言第二个子图的y轴标签为"y"
        assert ax2.get_ylabel() == "y"
        
        # 断言第二个子图的y轴标签不可见
        assert not ax2.yaxis.label.get_visible()

    # 测试Matplotlib关键字参数的设置
    def test_matplotlib_kwargs(self, long_df):

        # 定义Matplotlib关键字参数的字典
        kws = {
            "linestyle": "--",
            "linewidth": 3,
            "color": (1, .5, .2),
            "markeredgecolor": (.2, .5, .2),
            "markeredgewidth": 1,
        }
        
        # 绘制线图，传入数据和Matplotlib关键字参数，获取轴对象
        ax = lineplot(data=long_df, x="x", y="y", **kws)

        # 获取第一条线条对象
        line, *_ = ax.lines
        
        # 遍历Matplotlib关键字参数字典，断言每个属性的值与线条对象的对应属性值相等
        for key, val in kws.items():
            plot_val = getattr(line, f"get_{key}")()
            assert plot_val == val
    # 定义测试方法，验证在没有映射到特定风格的情况下绘制的线条样式
    def test_nonmapped_dashes(self):
        # 调用lineplot函数生成图表，并指定虚线样式为(2, 1)
        ax = lineplot(x=[1, 2], y=[1, 2], dashes=(2, 1))
        # 获取第一个线条对象
        line = ax.lines[0]
        # 断言线条的线型是否为短横线"--"
        # 注意：这里的断言测试并不完善，因为线条的虚线样式不会公开显示
        assert line.get_linestyle() == "--"

    # 定义测试方法，验证lineplot函数在指定不同图表对象(ax)时的行为
    def test_lineplot_axes(self, wide_df):
        # 创建两个子图对象
        f1, ax1 = plt.subplots()
        f2, ax2 = plt.subplots()

        # 调用lineplot函数，并将结果ax与预期的ax2进行比较
        ax = lineplot(data=wide_df)
        assert ax is ax2

        # 再次调用lineplot函数，这次指定ax=ax1，与预期的ax1进行比较
        ax = lineplot(data=wide_df, ax=ax1)
        assert ax is ax1

    # 定义测试方法，验证带有hue参数的lineplot函数对图例属性的影响
    def test_legend_attributes_with_hue(self, long_df):
        # 定义线条的样式参数字典
        kws = {"marker": "o", "linewidth": 3}
        # 调用lineplot函数绘制图表，并指定hue="a"，获取生成的图表对象ax
        ax = lineplot(long_df, x="x", y="y", hue="a", **kws)
        # 获取默认的调色板
        palette = color_palette()
        # 遍历图例中的每个线条对象，并进行属性断言
        for i, line in enumerate(get_legend_handles(ax.get_legend())):
            # 断言线条的颜色与调色板中对应颜色一致
            assert same_color(line.get_color(), palette[i])
            # 断言线条的线宽与预期一致
            assert line.get_linewidth() == kws["linewidth"]
            # 如果Matplotlib版本不早于3.7.0，断言线条的标记(marker)与预期一致
            if not _version_predates(mpl, "3.7.0"):
                assert line.get_marker() == kws["marker"]

    # 定义测试方法，验证带有style参数的lineplot函数对图例属性的影响
    def test_legend_attributes_with_style(self, long_df):
        # 定义线条的样式参数字典
        kws = {"color": "r", "marker": "o", "linewidth": 3}
        # 调用lineplot函数绘制图表，并指定style="a"，获取生成的图表对象ax
        ax = lineplot(long_df, x="x", y="y", style="a", **kws)
        # 遍历图例中的每个线条对象，并进行属性断言
        for line in get_legend_handles(ax.get_legend()):
            # 断言线条的颜色与预期颜色一致
            assert same_color(line.get_color(), kws["color"])
            # 如果Matplotlib版本不早于3.7.0，断言线条的标记(marker)与预期一致
            if not _version_predates(mpl, "3.7.0"):
                assert line.get_marker() == kws["marker"]
            # 断言线条的线宽与预期一致
            assert line.get_linewidth() == kws["linewidth"]

    # 定义测试方法，验证同时带有hue和style参数的lineplot函数对图例属性的影响
    def test_legend_attributes_with_hue_and_style(self, long_df):
        # 定义线条的样式参数字典
        kws = {"marker": "o", "linewidth": 3}
        # 调用lineplot函数绘制图表，并指定hue="a"，style="b"，获取生成的图表对象ax
        ax = lineplot(long_df, x="x", y="y", hue="a", style="b", **kws)
        # 遍历图例中的每个线条对象，并进行属性断言
        for line in get_legend_handles(ax.get_legend()):
            # 如果线条的标签不是"a"或"b"，则进行属性断言
            if line.get_label() not in ["a", "b"]:
                # 如果Matplotlib版本不早于3.7.0，断言线条的标记(marker)与预期一致
                if not _version_predates(mpl, "3.7.0"):
                    assert line.get_marker() == kws["marker"]
                # 断言线条的线宽与预期一致
                assert line.get_linewidth() == kws["linewidth"]

    # 定义测试方法，验证lineplot函数与relplot(kind="line")函数生成的线条属性比较
    def test_lineplot_vs_relplot(self, long_df, long_semantics):
        # 调用lineplot函数绘制线图，并关闭图例显示
        ax = lineplot(data=long_df, legend=False, **long_semantics)
        # 调用relplot(kind="line")函数绘制线图，并关闭图例显示
        g = relplot(data=long_df, kind="line", legend=False, **long_semantics)

        # 获取lineplot生成的线条对象列表
        lin_lines = ax.lines
        # 获取relplot生成的线条对象列表
        rel_lines = g.ax.lines

        # 遍历对应位置上的lineplot和relplot生成的每对线条对象，进行属性断言
        for l1, l2 in zip(lin_lines, rel_lines):
            # 断言两个线条对象的数据点(xydata)完全一致
            assert_array_equal(l1.get_xydata(), l2.get_xydata())
            # 断言两个线条对象的颜色完全一致
            assert same_color(l1.get_color(), l2.get_color())
            # 断言两个线条对象的线宽完全一致
            assert l1.get_linewidth() == l2.get_linewidth()
            # 断言两个线条对象的线型完全一致
            assert l1.get_linestyle() == l2.get_linestyle()

    # 定义简单的smoke测试方法，验证lineplot函数在各种数据类型下的运行情况
    def test_lineplot_smoke(
        self,
        wide_df, wide_array,
        wide_list_of_series, wide_list_of_arrays, wide_list_of_lists,
        flat_array, flat_series, flat_list,
        long_df, null_df, object_df
    ):
    ):
        # 创建一个新的图形和子图
        f, ax = plt.subplots()

        # 调用自定义函数lineplot绘制空的线图，并清除轴
        lineplot(x=[], y=[])
        ax.clear()

        # 清除轴并绘制宽格式数据的线图
        lineplot(data=wide_df)
        ax.clear()

        # 清除轴并绘制宽格式数组的线图
        lineplot(data=wide_array)
        ax.clear()

        # 清除轴并绘制宽格式系列列表的线图
        lineplot(data=wide_list_of_series)
        ax.clear()

        # 清除轴并绘制宽格式数组列表的线图
        lineplot(data=wide_list_of_arrays)
        ax.clear()

        # 清除轴并绘制宽格式列表列表的线图
        lineplot(data=wide_list_of_lists)
        ax.clear()

        # 清除轴并绘制扁平格式系列的线图
        lineplot(data=flat_series)
        ax.clear()

        # 清除轴并绘制扁平格式数组的线图
        lineplot(data=flat_array)
        ax.clear()

        # 清除轴并绘制扁平格式列表的线图
        lineplot(data=flat_list)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图
        lineplot(x="x", y="y", data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame列的线图
        lineplot(x=long_df.x, y=long_df.y)
        ax.clear()

        # 清除轴并绘制长格式DataFrame列和数据的线图
        lineplot(x=long_df.x, y="y", data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame列和转换为NumPy数组的数据的线图
        lineplot(x="x", y=long_df.y.to_numpy(), data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图，但y轴指定错误的列名
        lineplot(x="x", y="t", data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图，带有hue参数
        lineplot(x="x", y="y", hue="a", data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图，带有hue和style参数
        lineplot(x="x", y="y", hue="a", style="a", data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图，带有hue和style参数，style参数指定错误的列名
        lineplot(x="x", y="y", hue="a", style="b", data=long_df)
        ax.clear()

        # 清除轴并绘制空DataFrame的线图，带有hue和style参数，data为null_df
        lineplot(x="x", y="y", hue="a", style="a", data=null_df)
        ax.clear()

        # 清除轴并绘制空DataFrame的线图，带有hue和style参数，style参数指定错误的列名，data为null_df
        lineplot(x="x", y="y", hue="a", style="b", data=null_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图，带有hue和size参数
        lineplot(x="x", y="y", hue="a", size="a", data=long_df)
        ax.clear()

        # 清除轴并绘制长格式DataFrame的线图，带有hue和size参数，size参数指定错误的列名
        lineplot(x="x", y="y", hue="a", size="s", data=long_df)
        ax.clear()

        # 清除轴并绘制空DataFrame的线图，带有hue和size参数，data为null_df
        lineplot(x="x", y="y", hue="a", size="a", data=null_df)
        ax.clear()

        # 清除轴并绘制空DataFrame的线图，带有hue和size参数，size参数指定错误的列名，data为null_df
        lineplot(x="x", y="y", hue="a", size="s", data=null_df)
        ax.clear()

        # 清除轴并绘制对象类型DataFrame的线图，带有hue参数
        lineplot(x="x", y="y", hue="f", data=object_df)
        ax.clear()

        # 清除轴并绘制对象类型DataFrame的线图，带有hue和size参数
        lineplot(x="x", y="y", hue="c", size="f", data=object_df)
        ax.clear()

        # 清除轴并绘制对象类型DataFrame的线图，带有hue和size参数，size参数指定错误的列名
        lineplot(x="x", y="y", hue="f", size="s", data=object_df)
        ax.clear()

        # 清除轴并绘制空的长格式DataFrame的线图，不包含任何数据点
        lineplot(x="x", y="y", hue="a", data=long_df.iloc[:0])
        ax.clear()

    def test_ci_deprecation(self, long_df):
        # 创建一个包含两个子图的图形对象
        axs = plt.figure().subplots(2)

        # 在第一个子图上绘制长格式DataFrame的线图，带有误差条和seed参数，清除轴
        lineplot(data=long_df, x="x", y="y", errorbar=("ci", 95), seed=0, ax=axs[0])

        # 使用pytest断言检查是否有FutureWarning并匹配特定消息
        with pytest.warns(FutureWarning, match="\n\nThe `ci` parameter is deprecated"):
            # 在第二个子图上绘制长格式DataFrame的线图，带有ci参数和seed参数，清除轴
            lineplot(data=long_df, x="x", y="y", ci=95, seed=0, ax=axs[1])

        # 使用assert_plots_equal比较两个子图的绘图结果
        assert_plots_equal(*axs)

        # 创建一个包含两个子图的图形对象
        axs = plt.figure().subplots(2)

        # 在第一个子图上绘制长格式DataFrame的线图，带有误差条和sd参数，清除轴
        lineplot(data=long_df, x="x", y="y", errorbar="sd", ax=axs[0])

        # 使用pytest断言检查是否有FutureWarning并匹配特定消息
        with pytest.warns(FutureWarning, match="\n\nThe `ci` parameter is deprecated"):
            # 在第二个子图上绘制长格式DataFrame的线图，带有ci参数和sd参数，清除轴
            lineplot(data=long_df, x="x", y="y", ci="sd", ax=axs[1])

        # 使用assert_plots_equal比较两个子图的绘图结果
        assert_plots_equal(*axs)
class TestScatterPlotter(SharedAxesLevelTests, Helpers):
    # 定义测试类 TestScatterPlotter，继承自 SharedAxesLevelTests 和 Helpers

    func = staticmethod(scatterplot)
    # 设置 func 为 scatterplot 的静态方法

    def get_last_color(self, ax):
        # 定义获取最后一个图表对象的颜色的方法，参数为 ax（坐标轴对象）

        colors = ax.collections[-1].get_facecolors()
        # 获取最后一个集合对象的面部颜色

        unique_colors = np.unique(colors, axis=0)
        # 沿着第 0 轴（颜色的维度）获取唯一的颜色

        assert len(unique_colors) == 1
        # 断言唯一颜色的数量为 1

        return to_rgba(unique_colors.squeeze())
        # 返回转换为 RGBA 格式的唯一颜色值

    def test_color(self, long_df):
        # 定义颜色测试方法，参数为 long_df（长格式数据框）

        super().test_color(long_df)
        # 调用父类的 test_color 方法

        ax = plt.figure().subplots()
        # 创建一个图形并获取其子图对象

        self.func(data=long_df, x="x", y="y", facecolor="C5", ax=ax)
        # 调用 scatterplot 方法绘制散点图，指定 x 和 y 轴，面部颜色为 "C5"，使用给定的坐标轴 ax

        assert self.get_last_color(ax) == to_rgba("C5")
        # 断言最后一个图表对象的颜色是否与 "C5" 的 RGBA 值相同

        ax = plt.figure().subplots()
        # 创建一个新图形并获取其子图对象

        self.func(data=long_df, x="x", y="y", facecolors="C6", ax=ax)
        # 调用 scatterplot 方法绘制散点图，指定 x 和 y 轴，面部颜色为 "C6"，使用给定的坐标轴 ax

        assert self.get_last_color(ax) == to_rgba("C6")
        # 断言最后一个图表对象的颜色是否与 "C6" 的 RGBA 值相同

        ax = plt.figure().subplots()
        # 创建一个新图形并获取其子图对象

        self.func(data=long_df, x="x", y="y", fc="C4", ax=ax)
        # 调用 scatterplot 方法绘制散点图，指定 x 和 y 轴，面部颜色为 "C4"，使用给定的坐标轴 ax

        assert self.get_last_color(ax) == to_rgba("C4")
        # 断言最后一个图表对象的颜色是否与 "C4" 的 RGBA 值相同

    def test_legend_no_semantics(self, long_df):
        # 定义测试无语义图例的方法，参数为 long_df（长格式数据框）

        ax = scatterplot(long_df, x="x", y="y")
        # 绘制散点图并获取坐标轴对象

        handles, _ = ax.get_legend_handles_labels()
        # 获取图例的句柄和标签

        assert not handles
        # 断言图例句柄不存在

    def test_legend_hue(self, long_df):
        # 定义测试基于 hue 参数的图例方法，参数为 long_df（长格式数据框）

        ax = scatterplot(long_df, x="x", y="y", hue="a")
        # 绘制带有 hue 参数的散点图并获取坐标轴对象

        handles, labels = ax.get_legend_handles_labels()
        # 获取图例的句柄和标签

        colors = [h.get_color() for h in handles]
        # 获取句柄对象的颜色属性

        expected_colors = color_palette(n_colors=len(handles))
        # 根据句柄数量生成预期颜色列表

        assert same_color(colors, expected_colors)
        # 断言实际颜色与预期颜色相同

        assert labels == categorical_order(long_df["a"])
        # 断言图例标签顺序与数据中 "a" 列的类别顺序相同

    def test_legend_hue_style_same(self, long_df):
        # 定义测试同时使用 hue 和 style 参数的图例方法，参数为 long_df（长格式数据框）

        ax = scatterplot(long_df, x="x", y="y", hue="a", style="a")
        # 绘制带有 hue 和 style 参数的散点图并获取坐标轴对象

        handles, labels = ax.get_legend_handles_labels()
        # 获取图例的句柄和标签

        colors = [h.get_color() for h in handles]
        # 获取句柄对象的颜色属性

        expected_colors = color_palette(n_colors=len(labels))
        # 根据标签数量生成预期颜色列表

        markers = [h.get_marker() for h in handles]
        # 获取句柄对象的标记属性

        expected_markers = unique_markers(len(handles))
        # 根据句柄数量生成预期标记列表

        assert same_color(colors, expected_colors)
        # 断言实际颜色与预期颜色相同

        assert markers == expected_markers
        # 断言实际标记与预期标记相同

        assert labels == categorical_order(long_df["a"])
        # 断言图例标签顺序与数据中 "a" 列的类别顺序相同

    def test_legend_hue_style_different(self, long_df):
        # 定义测试同时使用不同 hue 和 style 参数的图例方法，参数为 long_df（长格式数据框）

        ax = scatterplot(long_df, x="x", y="y", hue="a", style="b")
        # 绘制带有不同 hue 和 style 参数的散点图并获取坐标轴对象

        handles, labels = ax.get_legend_handles_labels()
        # 获取图例的句柄和标签

        colors = [h.get_color() for h in handles]
        # 获取句柄对象的颜色属性

        expected_colors = [
            "w", *color_palette(n_colors=long_df["a"].nunique()),
            "w", *[".2" for _ in long_df["b"].unique()],
        ]
        # 生成预期的颜色列表，包括空白、唯一的 "a" 列颜色和 ".2" 颜色

        markers = [h.get_marker() for h in handles]
        # 获取句柄对象的标记属性

        expected_markers = [
            "", *["o" for _ in long_df["a"].unique()],
            "", *unique_markers(long_df["b"].nunique()),
        ]
        # 生成预期的标记列表，包括空白、唯一的 "a" 列标记和长格式数据框 "b" 列唯一标记

        assert same_color(colors, expected_colors)
        # 断言实际颜色与预期颜色相同

        assert markers == expected_markers
        # 断言实际标记与预期标记相同

        assert labels == [
            "a", *categorical_order(long_df["a"]),
            "b", *categorical_order(long_df["b"]),
        ]
        # 断言图例标签顺序与数据中 "a" 和 "b" 列的类别顺序相同
    # 测试函数：测试绘制散点图时，hue 和 size 相同时的图例显示
    def test_legend_data_hue_size_same(self, long_df):
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 hue 和 size 的列名
        ax = scatterplot(long_df, x="x", y="y", hue="a", size="a")
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取所有句柄的颜色
        colors = [h.get_color() for h in handles]
        # 期望的颜色列表，根据标签数量生成颜色调色板
        expected_colors = color_palette(n_colors=len(labels))
        # 获取所有句柄的大小
        sizes = [h.get_markersize() for h in handles]
        # 计算期望的大小，基于全局配置的标记大小
        ms = mpl.rcParams["lines.markersize"] ** 2
        expected_sizes = np.sqrt(
            [ms * scl for scl in np.linspace(2, 0.5, len(handles))]
        ).tolist()
        # 断言实际颜色与期望颜色一致
        assert same_color(colors, expected_colors)
        # 断言实际大小与期望大小一致
        assert sizes == expected_sizes
        # 断言标签顺序与长数据框中列 'a' 的分类顺序一致
        assert labels == categorical_order(long_df["a"])
        # 断言图例标题为 'a'
        assert ax.get_legend().get_title().get_text() == "a"

    # 测试函数：测试绘制散点图时，size 为数值列表时的图例显示
    def test_legend_size_numeric_list(self, long_df):
        # 指定 size 的列表
        size_list = [10, 100, 200]
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 size 列，并设置大小列表
        ax = scatterplot(long_df, x="x", y="y", size="s", sizes=size_list)
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取所有句柄的大小
        sizes = [h.get_markersize() for h in handles]
        # 期望的大小列表，根据输入的大小列表进行平方根处理
        expected_sizes = list(np.sqrt(size_list))
        # 断言实际大小与期望大小一致
        assert sizes == expected_sizes
        # 断言标签顺序与长数据框中列 's' 的分类顺序一致，并转换为字符串列表
        assert labels == list(map(str, categorical_order(long_df["s"])))
        # 断言图例标题为 's'
        assert ax.get_legend().get_title().get_text() == "s"

    # 测试函数：测试绘制散点图时，size 为数值字典时的图例显示
    def test_legend_size_numeric_dict(self, long_df):
        # 指定 size 的字典
        size_dict = {2: 10, 4: 100, 8: 200}
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 size 列，并设置大小字典
        ax = scatterplot(long_df, x="x", y="y", size="s", sizes=size_dict)
        # 获取图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 获取所有句柄的大小
        sizes = [h.get_markersize() for h in handles]
        # 根据长数据框中列 's' 的分类顺序，计算期望的大小列表
        order = categorical_order(long_df["s"])
        expected_sizes = [np.sqrt(size_dict[k]) for k in order]
        # 断言实际大小与期望大小一致
        assert sizes == expected_sizes
        # 断言标签顺序与长数据框中列 's' 的分类顺序一致，并转换为字符串列表
        assert labels == list(map(str, order))
        # 断言图例标题为 's'
        assert ax.get_legend().get_title().get_text() == "s"

    # 测试函数：测试绘制散点图时，hue 为数值且 legend 为 'full' 时的图例显示
    def test_legend_numeric_hue_full(self):
        # 生成随机数据
        x, y = np.random.randn(2, 40)
        z = np.tile(np.arange(20), 2)
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 hue 列，并设置 legend 为 'full'
        ax = scatterplot(x=x, y=y, hue=z, legend="full")
        # 获取图例的句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 断言标签内容为排序后的 z 列的字符串表示
        assert labels == [str(z_i) for z_i in sorted(set(z))]
        # 断言图例标题为空字符串
        assert ax.get_legend().get_title().get_text() == ""

    # 测试函数：测试绘制散点图时，hue 为数值且 legend 为 'brief' 时的图例显示
    def test_legend_numeric_hue_brief(self):
        # 生成随机数据
        x, y = np.random.randn(2, 40)
        z = np.tile(np.arange(20), 2)
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 hue 列，并设置 legend 为 'brief'
        ax = scatterplot(x=x, y=y, hue=z, legend="brief")
        # 获取图例的句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 断言标签数量少于 z 列中唯一值的数量
        assert len(labels) < len(set(z))

    # 测试函数：测试绘制散点图时，size 为数值且 legend 为 'full' 时的图例显示
    def test_legend_numeric_size_full(self):
        # 生成随机数据
        x, y = np.random.randn(2, 40)
        z = np.tile(np.arange(20), 2)
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 size 列，并设置 legend 为 'full'
        ax = scatterplot(x=x, y=y, size=z, legend="full")
        # 获取图例的句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 断言标签内容为排序后的 z 列的字符串表示
        assert labels == [str(z_i) for z_i in sorted(set(z))]

    # 测试函数：测试绘制散点图时，size 为数值且 legend 为 'brief' 时的图例显示
    def test_legend_numeric_size_brief(self):
        # 生成随机数据
        x, y = np.random.randn(2, 40)
        z = np.tile(np.arange(20), 2)
        # 调用 scatterplot 函数生成散点图，并指定 x, y 轴以及 size 列，并设置 legend 为 'brief'
        ax = scatterplot(x=x, y=y, size=z, legend="brief")
        # 获取图例的句柄和标签
        _, labels = ax.get_legend_handles_labels()
        # 断言标签数量少于 z 列中唯一值的数量
        assert len(labels) < len(set(z))
    # 测试函数，验证散点图中图例的属性（根据色调）
    def test_legend_attributes_hue(self, long_df):
        # 定义散点图的绘制参数
        kws = {"s": 50, "linewidth": 1, "marker": "X"}
        # 调用散点图函数绘制图像，返回绘图的坐标轴对象
        ax = scatterplot(long_df, x="x", y="y", hue="a", **kws)
        # 获取默认的调色板
        palette = color_palette()
        # 遍历图例中的每个句柄，并验证其属性
        for i, pt in enumerate(get_legend_handles(ax.get_legend())):
            # 断言句柄的颜色与调色板中对应位置的颜色相同
            assert same_color(pt.get_color(), palette[i])
            # 断言句柄的标记大小为散点图参数定义的大小的平方根
            assert pt.get_markersize() == np.sqrt(kws["s"])
            # 断言句柄的标记边缘宽度与散点图参数定义的宽度相同
            assert pt.get_markeredgewidth() == kws["linewidth"]
            # 如果当前 matplotlib 版本不早于 3.7.0
            if not _version_predates(mpl, "3.7.0"):
                # 断言句柄的标记类型与散点图参数定义的标记类型相同
                assert pt.get_marker() == kws["marker"]
    
    # 测试函数，验证散点图中图例的属性（根据样式）
    def test_legend_attributes_style(self, long_df):
        # 定义散点图的绘制参数
        kws = {"s": 50, "linewidth": 1, "color": "r"}
        # 调用散点图函数绘制图像，返回绘图的坐标轴对象
        ax = scatterplot(long_df, x="x", y="y", style="a", **kws)
        # 遍历图例中的每个句柄，并验证其属性
        for pt in get_legend_handles(ax.get_legend()):
            # 断言句柄的标记大小为散点图参数定义的大小的平方根
            assert pt.get_markersize() == np.sqrt(kws["s"])
            # 断言句柄的标记边缘宽度与散点图参数定义的宽度相同
            assert pt.get_markeredgewidth() == kws["linewidth"]
            # 断言句柄的颜色与预定义的红色相同
            assert same_color(pt.get_color(), "r")
    
    # 测试函数，验证散点图中图例的属性（同时根据色调和样式）
    def test_legend_attributes_hue_and_style(self, long_df):
        # 定义散点图的绘制参数
        kws = {"s": 50, "linewidth": 1}
        # 调用散点图函数绘制图像，返回绘图的坐标轴对象
        ax = scatterplot(long_df, x="x", y="y", hue="a", style="b", **kws)
        # 遍历图例中的每个句柄，并验证其属性
        for pt in get_legend_handles(ax.get_legend()):
            # 如果句柄的标签不是 "a" 或 "b"
            if pt.get_label() not in ["a", "b"]:
                # 断言句柄的标记大小为散点图参数定义的大小的平方根
                assert pt.get_markersize() == np.sqrt(kws["s"])
                # 断言句柄的标记边缘宽度与散点图参数定义的宽度相同
                assert pt.get_markeredgewidth() == kws["linewidth"]
    
    # 测试函数，验证散点图在指定错误的图例参数时抛出 ValueError 异常
    def test_legend_value_error(self, long_df):
        # 使用 pytest 断言期望抛出 ValueError 异常，并匹配给定的错误信息
        with pytest.raises(ValueError, match=r"`legend` must be"):
            scatterplot(long_df, x="x", y="y", hue="a", legend="bad_value")
    # 定义一个测试方法，用于测试 ScatterPlotter 类的绘图功能
    def test_plot(self, long_df, repeated_df):

        # 创建一个新的图形和坐标轴对象
        f, ax = plt.subplots()

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"
        p = _ScatterPlotter(data=long_df, variables=dict(x="x", y="y"))

        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 断言点的偏移量与 long_df 中 "x" 和 "y" 列的值相等
        assert_array_equal(points.get_offsets(), long_df[["x", "y"]].to_numpy())

        # 清空坐标轴
        ax.clear()
        # 重新绘制图像，指定颜色为黑色，标签为 "test"
        p.plot(ax, {"color": "k", "label": "test"})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 断言点的颜色与指定的黑色相同
        assert same_color(points.get_facecolor(), "k")
        # 断言点的标签为 "test"
        assert points.get_label() == "test"

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，颜色变量为 "a"
        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", hue="a")
        )

        # 清空坐标轴
        ax.clear()
        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 计算预期的颜色值列表
        expected_colors = p._hue_map(p.plot_data["hue"])
        # 断言点的颜色与预期的颜色列表相同
        assert same_color(points.get_facecolors(), expected_colors)

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，样式变量为 "c"
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="c"),
        )
        # 映射样式，指定标记为 "+" 和 "x"
        p.map_style(markers=["+", "x"])

        # 清空坐标轴
        ax.clear()
        # 指定颜色为 (1, .3, .8)，在指定的坐标轴上绘制图像
        color = (1, .3, .8)
        p.plot(ax, {"color": color})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 断言点的边缘颜色与指定的颜色列表相同
        assert same_color(points.get_edgecolors(), [color])

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，大小变量为 "a"
        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", size="a"),
        )

        # 清空坐标轴
        ax.clear()
        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 计算预期的大小值列表
        expected_sizes = p._size_map(p.plot_data["size"])
        # 断言点的大小与预期的大小值列表相同
        assert_array_equal(points.get_sizes(), expected_sizes)

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，颜色变量为 "a"，样式变量为 "a"
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="a"),
        )
        # 映射样式，标记为默认
        p.map_style(markers=True)

        # 清空坐标轴
        ax.clear()
        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 计算预期的颜色值列表和路径列表
        expected_colors = p._hue_map(p.plot_data["hue"])
        expected_paths = p._style_map(p.plot_data["style"], "path")
        # 断言点的面部颜色与预期的颜色值列表相同，路径是否相等
        assert same_color(points.get_facecolors(), expected_colors)
        assert self.paths_equal(points.get_paths(), expected_paths)

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，颜色变量为 "a"，样式变量为 "b"
        p = _ScatterPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", style="b"),
        )
        # 映射样式，标记为默认
        p.map_style(markers=True)

        # 清空坐标轴
        ax.clear()
        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})
        # 获取绘制的点集合
        points = ax.collections[0]
        # 计算预期的颜色值列表和路径列表
        expected_colors = p._hue_map(p.plot_data["hue"])
        expected_paths = p._style_map(p.plot_data["style"], "path")
        # 断言点的面部颜色与预期的颜色值列表相同，路径是否相等
        assert same_color(points.get_facecolors(), expected_colors)
        assert self.paths_equal(points.get_paths(), expected_paths)

        # 将 long_df 中的 "x" 列转换为字符串类型
        x_str = long_df["x"].astype(str)
        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，颜色变量为 x_str
        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", hue=x_str),
        )
        # 清空坐标轴
        ax.clear()
        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})

        # 创建 ScatterPlotter 对象 p，指定数据为 long_df，x 轴和 y 轴变量为 "x" 和 "y"，大小变量为 x_str
        p = _ScatterPlotter(
            data=long_df, variables=dict(x="x", y="y", size=x_str),
        )
        # 清空坐标轴
        ax.clear()
        # 在指定的坐标轴上绘制图像
        p.plot(ax, {})
    # 测试散点图的坐标轴标签设置是否正确
    def test_axis_labels(self, long_df):

        # 创建包含两个子图的图形对象
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        # 创建散点图绘制器对象，指定数据和变量映射
        p = _ScatterPlotter(data=long_df, variables=dict(x="x", y="y"))

        # 在第一个子图上绘制散点图，并检查 x 轴和 y 轴的标签是否正确
        p.plot(ax1, {})
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "y"

        # 在第二个子图上绘制散点图，并检查 x 轴和 y 轴的标签是否正确，以及 y 轴标签是否可见
        p.plot(ax2, {})
        assert ax2.get_xlabel() == "x"
        assert ax2.get_ylabel() == "y"
        assert not ax2.yaxis.label.get_visible()

    # 测试散点图绘制在不同的轴上
    def test_scatterplot_axes(self, wide_df):

        # 创建两个单独的图形对象和相应的坐标轴对象
        f1, ax1 = plt.subplots()
        f2, ax2 = plt.subplots()

        # 在 wide_df 数据上绘制散点图，并检查是否正确设置了坐标轴
        ax = scatterplot(data=wide_df)
        assert ax is ax2

        # 在 wide_df 数据上绘制散点图，指定了特定的坐标轴，并检查是否正确设置了坐标轴
        ax = scatterplot(data=wide_df, ax=ax1)
        assert ax is ax1

    # 测试直接使用字面量定义的属性向量绘制散点图
    def test_literal_attribute_vectors(self):

        # 创建图形对象和坐标轴对象
        f, ax = plt.subplots()

        # 定义 x, y, s, c 向量
        x = y = [1, 2, 3]
        s = [5, 10, 15]
        c = [(1, 1, 0, 1), (1, 0, 1, .5), (.5, 1, 0, 1)]

        # 绘制散点图，使用指定的向量作为参数
        scatterplot(x=x, y=y, c=c, s=s, ax=ax)

        # 获取绘制的散点图集合对象
        points, = ax.collections

        # 断言散点图的大小属性与给定的 s 向量一致
        assert_array_equal(points.get_sizes().squeeze(), s)
        # 断言散点图的颜色属性与给定的 c 向量一致
        assert_array_equal(points.get_facecolors(), c)

    # 测试使用提供的颜色数组绘制散点图
    def test_supplied_color_array(self, long_df):

        # 获取指定色图和归一化对象
        cmap = get_colormap("Blues")
        norm = mpl.colors.Normalize()
        # 根据 long_df 数据生成颜色数组
        colors = cmap(norm(long_df["y"].to_numpy()))

        # 需要检查的属性键列表
        keys = ["c", "fc", "facecolor", "facecolors"]

        # 遍历每个属性键，在单独的图形对象上绘制散点图，并断言颜色属性是否正确
        for key in keys:
            ax = plt.figure().subplots()
            scatterplot(data=long_df, x="x", y="y", **{key: colors})
            _draw_figure(ax.figure)
            assert_array_equal(ax.collections[0].get_facecolors(), colors)

        # 在单独的图形对象上绘制散点图，使用 long_df 数据的 y 列作为颜色参数，并断言颜色属性是否正确
        ax = plt.figure().subplots()
        scatterplot(data=long_df, x="x", y="y", c=long_df["y"], cmap=cmap)
        _draw_figure(ax.figure)
        assert_array_equal(ax.collections[0].get_facecolors(), colors)

    # 测试指定 hue_order 参数绘制散点图
    def test_hue_order(self, long_df):

        # 获取长格式数据 long_df 中列 "a" 的分类顺序
        order = categorical_order(long_df["a"])
        unused = order.pop()  # 弹出最后一个元素，用于后续断言

        # 绘制散点图，指定 x, y, hue 和 hue_order 参数
        ax = scatterplot(data=long_df, x="x", y="y", hue="a", hue_order=order)
        points = ax.collections[0]

        # 断言散点图中使用了 hue_order 中未使用的颜色
        assert (points.get_facecolors()[long_df["a"] == unused] == 0).all()
        # 断言图例中的文本与 hue_order 中的顺序一致
        assert [t.get_text() for t in ax.legend_.texts] == order

    # 测试散点图的线宽设置
    def test_linewidths(self, long_df):

        # 创建图形对象和坐标轴对象
        f, ax = plt.subplots()

        # 使用指定大小 s 绘制散点图，并检查散点图的线宽属性是否正确设置
        scatterplot(data=long_df, x="x", y="y", s=10)
        scatterplot(data=long_df, x="x", y="y", s=20)
        points1, points2 = ax.collections
        assert (
            points1.get_linewidths().item() < points2.get_linewidths().item()
        )

        # 使用 long_df 数据的 x 列大小作为 s 绘制散点图，并检查散点图的线宽属性是否正确设置
        ax.clear()
        scatterplot(data=long_df, x="x", y="y", s=long_df["x"])
        scatterplot(data=long_df, x="x", y="y", s=long_df["x"] * 2)
        points1, points2 = ax.collections
        assert (
            points1.get_linewidths().item() < points2.get_linewidths().item()
        )

        # 使用指定的线宽绘制散点图，并检查散点图的线宽属性是否正确设置
        ax.clear()
        lw = 2
        scatterplot(data=long_df, x="x", y="y", linewidth=lw)
        assert ax.collections[0].get_linewidths().item() == lw
    def test_size_norm_extrapolation(self):
        # 测试大小规范的外推功能

        # 创建一个包含0到18的数组，步长为2
        x = np.arange(0, 20, 2)
        # 创建包含两个子图的图形对象，共享x轴和y轴
        f, axs = plt.subplots(1, 2, sharex=True, sharey=True)

        # 设置要测试的数据切片大小
        slc = 5
        # 定义关键字参数字典
        kws = dict(sizes=(50, 200), size_norm=(0, x.max()), legend="brief")

        # 在第一个子图上绘制散点图，使用指定的参数
        scatterplot(x=x, y=x, size=x, ax=axs[0], **kws)
        # 在第二个子图上绘制散点图，使用指定的参数和数据切片
        scatterplot(x=x[:slc], y=x[:slc], size=x[:slc], ax=axs[1], **kws)

        # 断言两个子图的第一个集合对象的大小数据是否近似相等
        assert np.allclose(
            axs[0].collections[0].get_sizes()[:slc],
            axs[1].collections[0].get_sizes()
        )

        # 获取每个子图的图例对象
        legends = [ax.legend_ for ax in axs]
        # 提取每个图例对象的文本和标记大小，组成列表
        legend_data = [
            {
                label.get_text(): handle.get_markersize()
                for label, handle in zip(legend.get_texts(), get_legend_handles(legend))
            } for legend in legends
        ]

        # 检查两个图例数据字典的交集，确保它们匹配
        for key in set(legend_data[0]) & set(legend_data[1]):
            if key == "y":
                # 在某个版本（大约3.0）中，matplotlib自动将pandas系列添加到图例中，
                # 这会干扰本测试。在这里预期并忽略这种情况。
                continue
            assert legend_data[0][key] == legend_data[1][key]

    def test_datetime_scale(self, long_df):
        # 测试日期时间刻度的处理

        # 绘制数据框中"t"和"y"列的散点图，并返回轴对象
        ax = scatterplot(data=long_df, x="t", y="y")
        # 检查避免奇怪的matplotlib默认自动缩放问题
        assert ax.get_xlim()[0] > ax.xaxis.convert_units(np.datetime64("2002-01-01"))

    def test_unfilled_marker_edgecolor_warning(self, long_df):  # GH2636
        # 测试未填充标记的边缘颜色警告处理

        # 捕获警告并将其设置为错误级别
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 绘制数据框中"x"和"y"列的散点图，标记类型为"+"
            scatterplot(data=long_df, x="x", y="y", marker="+")

    def test_short_form_kwargs(self, long_df):
        # 测试使用简短形式关键字参数的处理

        # 绘制数据框中"x"和"y"列的散点图，边缘颜色设置为绿色，并返回轴对象
        ax = scatterplot(data=long_df, x="x", y="y", ec="g")
        # 获取轴对象的第一个集合对象
        pts = ax.collections[0]
        # 断言集合对象的边缘颜色是否与"g"相同
        assert same_color(pts.get_edgecolors().squeeze(), "g")

    def test_scatterplot_vs_relplot(self, long_df, long_semantics):
        # 测试散点图与relplot函数的一致性

        # 绘制长格式数据框的散点图，使用长语义参数
        ax = scatterplot(data=long_df, **long_semantics)
        # 使用relplot函数绘制相同数据的散点图，类型为scatter，使用长语义参数
        g = relplot(data=long_df, kind="scatter", **long_semantics)

        # 检查两个绘图对象的集合对象是否相等
        for s_pts, r_pts in zip(ax.collections, g.ax.collections):
            assert_array_equal(s_pts.get_offsets(), r_pts.get_offsets())
            assert_array_equal(s_pts.get_sizes(), r_pts.get_sizes())
            assert_array_equal(s_pts.get_facecolors(), r_pts.get_facecolors())
            # 检查两个路径对象是否相等，使用self.paths_equal方法
            assert self.paths_equal(s_pts.get_paths(), r_pts.get_paths())

    def test_scatterplot_smoke(
        self,
        wide_df, wide_array,
        flat_series, flat_array, flat_list,
        wide_list_of_series, wide_list_of_arrays, wide_list_of_lists,
        long_df, null_df, object_df
    ):
        # 烟雾测试散点图函数

        # 此方法主要测试散点图函数在多种输入数据格式下的基本功能。
        # 没有特定的断言，仅用于检查是否会引发异常。
    # 创建一个新的图形和轴对象
    f, ax = plt.subplots()
    
    # 调用 scatterplot 函数，绘制空的散点图，不传入数据
    scatterplot(x=[], y=[])
    # 清空当前轴的内容，准备绘制下一个图
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 wide_df 数据的散点图
    scatterplot(data=wide_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 wide_array 数据的散点图
    scatterplot(data=wide_array)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 wide_list_of_series 数据的散点图
    scatterplot(data=wide_list_of_series)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 wide_list_of_arrays 数据的散点图
    scatterplot(data=wide_list_of_arrays)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 wide_list_of_lists 数据的散点图
    scatterplot(data=wide_list_of_lists)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 flat_series 数据的散点图
    scatterplot(data=flat_series)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 flat_array 数据的散点图
    scatterplot(data=flat_array)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 flat_list 数据的散点图
    scatterplot(data=flat_list)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x 和 y 列
    scatterplot(x="x", y="y", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，使用 long_df 的 x 和 y 列
    scatterplot(x=long_df.x, y=long_df.y)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x 列和 y 列
    scatterplot(x=long_df.x, y="y", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x 列和 long_df.y 转换为 NumPy 数组的结果作为 y 值
    scatterplot(x="x", y=long_df.y.to_numpy(), data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x、y 列，并根据 a 列的不同取值进行着色
    scatterplot(x="x", y="y", hue="a", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x、y 列，同时根据 a 列和 b 列的不同取值进行着色和样式区分
    scatterplot(x="x", y="y", hue="a", style="a", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x、y 列，根据 a 列和 b 列的不同取值进行着色和样式区分
    scatterplot(x="x", y="y", hue="a", style="b", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 null_df 数据的散点图，指定 x、y 列，并根据 a 列的不同取值进行着色
    scatterplot(x="x", y="y", hue="a", style="a", data=null_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 null_df 数据的散点图，指定 x、y 列，根据 a 列和 b 列的不同取值进行着色和样式区分
    scatterplot(x="x", y="y", hue="a", style="b", data=null_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x、y 列，并根据 a 列的不同取值调整点的大小
    scatterplot(x="x", y="y", hue="a", size="a", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 long_df 数据的散点图，指定 x、y 列，并根据 s 列的不同取值调整点的大小
    scatterplot(x="x", y="y", hue="a", size="s", data=long_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 null_df 数据的散点图，指定 x、y 列，并根据 a 列的不同取值调整点的大小
    scatterplot(x="x", y="y", hue="a", size="a", data=null_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 null_df 数据的散点图，指定 x、y 列，并根据 s 列的不同取值调整点的大小
    scatterplot(x="x", y="y", hue="a", size="s", data=null_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 object_df 数据的散点图，指定 x、y 列，并根据 f 列的不同取值进行着色
    scatterplot(x="x", y="y", hue="f", data=object_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 object_df 数据的散点图，指定 x、y 列，并根据 c 列和 f 列的不同取值调整点的大小和颜色
    scatterplot(x="x", y="y", hue="f", size="f", data=object_df)
    ax.clear()
    
    # 调用 scatterplot 函数，绘制 object_df 数据的散点图，指定 x、y 列，并根据 f 列和 s 列的不同取值调整点的大小和颜色
    scatterplot(x="x", y="y", hue="f", size="s", data=object_df)
    ax.clear()
```