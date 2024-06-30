# `D:\src\scipysrc\seaborn\tests\test_base.py`

```
# 导入必要的库和模块：itertools、numpy、pandas、matplotlib
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 导入 pytest 库及其子模块
import pytest
# 从 numpy.testing 中导入数组比较的断言方法 assert_array_equal 和 assert_array_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
# 从 pandas.testing 中导入数据框比较的断言方法 assert_frame_equal
from pandas.testing import assert_frame_equal

# 从 seaborn 库中导入 FacetGrid 类和 get_colormap 函数
from seaborn.axisgrid import FacetGrid
from seaborn._compat import get_colormap
# 从 seaborn._base 中导入多个类和函数
from seaborn._base import (
    SemanticMapping,
    HueMapping,
    SizeMapping,
    StyleMapping,
    VectorPlotter,
    variable_type,
    infer_orient,
    unique_dashes,
    unique_markers,
    categorical_order,
)
# 从 seaborn.utils 中导入 desaturate 函数
from seaborn.utils import desaturate
# 从 seaborn.palettes 中导入 color_palette 函数
from seaborn.palettes import color_palette

# 定义 pytest 的 fixture，返回一个字典参数列表
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
# 返回参数化的 fixture，用于测试函数的参数化测试
def long_variables(request):
    return request.param


# 定义测试类 TestSemanticMapping
class TestSemanticMapping:

    # 定义测试方法 test_call_lookup
    def test_call_lookup(self):

        # 创建 SemanticMapping 对象，传入 VectorPlotter 实例
        m = SemanticMapping(VectorPlotter())
        # 创建查找表 lookup_table，将字符串 "abc" 映射为 (1, 2, 3)
        lookup_table = dict(zip("abc", (1, 2, 3)))
        # 将查找表赋值给 SemanticMapping 对象的 lookup_table 属性
        m.lookup_table = lookup_table
        # 遍历查找表的键值对，断言 SemanticMapping 对象根据键查找值是否正确
        for key, val in lookup_table.items():
            assert m(key) == val


# 定义测试类 TestHueMapping
class TestHueMapping:

    # 定义测试方法 test_plotter_default_init，测试默认初始化情况
    def test_plotter_default_init(self, long_df):

        # 创建 VectorPlotter 对象，传入长格式数据框 long_df 和变量字典
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        # 断言对象 p 没有 _hue_map 属性
        assert not hasattr(p, "_hue_map")

        # 创建 VectorPlotter 对象，传入长格式数据框 long_df 和带 hue 变量的变量字典
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
        )
        # 断言对象 p 的 _hue_map 属性是 HueMapping 的实例
        assert isinstance(p._hue_map, HueMapping)
        # 断言 _hue_map 对象的 map_type 属性等于 p.var_types["hue"]
        assert p._hue_map.map_type == p.var_types["hue"]

    # 定义测试方法 test_plotter_customization，测试自定义设置
    def test_plotter_customization(self, long_df):

        # 创建 VectorPlotter 对象，传入长格式数据框 long_df 和带 hue 变量的变量字典
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
        )
        # 设置调色板 palette 和 hue 的顺序 hue_order
        palette = "muted"
        hue_order = ["b", "a", "c"]
        # 调用 map_hue 方法进行 hue 映射的设置
        p.map_hue(palette=palette, order=hue_order)
        # 断言 _hue_map 对象的 palette 属性等于指定的调色板 palette
        assert p._hue_map.palette == palette
        # 断言 _hue_map 对象的 levels 属性等于指定的 hue_order
        assert p._hue_map.levels == hue_order

    # 定义测试方法 test_hue_map_null，测试空值情况下的 HueMapping
    def test_hue_map_null(self, flat_series, null_series):

        # 创建 VectorPlotter 对象，传入扁平化序列和空值序列作为变量
        p = VectorPlotter(variables=dict(x=flat_series, hue=null_series))
        # 创建 HueMapping 对象 m，传入 VectorPlotter 对象 p
        m = HueMapping(p)
        # 断言 HueMapping 对象 m 的 levels 属性为 None
        assert m.levels is None
        # 断言 HueMapping 对象 m 的 map_type 属性为 None
        assert m.map_type is None
        # 断言 HueMapping 对象 m 的 palette 属性为 None
        assert m.palette is None
        # 断言 HueMapping 对象 m 的 cmap 属性为 None
        assert m.cmap is None
        # 断言 HueMapping 对象 m 的 norm 属性为 None
        assert m.norm is None
        # 断言 HueMapping 对象 m 的 lookup_table 属性为 None
        assert m.lookup_table is None
    # 定义一个测试方法，用于测试 HueMapping 类的不同配置
    def test_hue_map_numeric(self, long_df):

        # 创建一个包含指定值的 NumPy 数组，用于测试色调映射
        vals = np.concatenate([np.linspace(0, 1, 256), [-.1, 1.1, np.nan]])

        # 测试默认的色调映射
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="s")
        )
        # 获取长数据框中唯一的色调值，并排序
        hue_levels = list(np.sort(long_df["s"].unique()))
        # 创建 HueMapping 对象
        m = HueMapping(p)
        # 断言 HueMapping 对象的属性值符合预期
        assert m.levels == hue_levels
        assert m.map_type == "numeric"
        assert m.cmap.name == "seaborn_cubehelix"

        # 测试命名的色彩板
        palette = "Purples"
        m = HueMapping(p, palette=palette)
        # 断言使用指定的色彩板生成的颜色映射与预期一致
        assert_array_equal(m.cmap(vals), get_colormap(palette)(vals))

        # 测试色彩板对象
        palette = get_colormap("Greens")
        m = HueMapping(p, palette=palette)
        # 断言使用指定的色彩板对象生成的颜色映射与预期一致
        assert_array_equal(m.cmap(vals), palette(vals))

        # 测试 cubehelix 简写
        palette = "ch:2,0,light=.2"
        m = HueMapping(p, palette=palette)
        # 断言生成的色彩映射是 ListedColormap 对象
        assert isinstance(m.cmap, mpl.colors.ListedColormap)

        # 测试指定的色调限制
        hue_norm = 1, 4
        m = HueMapping(p, norm=hue_norm)
        # 断言生成的 norm 是 Normalize 对象，并且边界值符合预期
        assert isinstance(m.norm, mpl.colors.Normalize)
        assert m.norm.vmin == hue_norm[0]
        assert m.norm.vmax == hue_norm[1]

        # 测试 Normalize 对象
        hue_norm = mpl.colors.PowerNorm(2, vmin=1, vmax=10)
        m = HueMapping(p, norm=hue_norm)
        # 断言生成的 norm 对象与指定的 hue_norm 对象一致
        assert m.norm is hue_norm

        # 测试默认色彩映射值
        hmin, hmax = p.plot_data["hue"].min(), p.plot_data["hue"].max()
        m = HueMapping(p)
        # 断言生成的 lookup_table 中的值与使用默认色彩映射的预期一致
        assert m.lookup_table[hmin] == pytest.approx(m.cmap(0.0))
        assert m.lookup_table[hmax] == pytest.approx(m.cmap(1.0))

        # 测试指定色彩映射值
        hue_norm = hmin - 1, hmax - 1
        m = HueMapping(p, norm=hue_norm)
        norm_min = (hmin - hue_norm[0]) / (hue_norm[1] - hue_norm[0])
        # 断言生成的 lookup_table 中的值与指定色彩映射的预期一致
        assert m.lookup_table[hmin] == pytest.approx(m.cmap(norm_min))
        assert m.lookup_table[hmax] == pytest.approx(m.cmap(1.0))

        # 测试颜色列表
        hue_levels = list(np.sort(long_df["s"].unique()))
        palette = color_palette("Blues", len(hue_levels))
        m = HueMapping(p, palette=palette)
        # 断言生成的 lookup_table 与指定的颜色列表一致
        assert m.lookup_table == dict(zip(hue_levels, palette))

        # 测试颜色列表长度不匹配的情况
        palette = color_palette("Blues", len(hue_levels) + 1)
        with pytest.warns(UserWarning):
            HueMapping(p, palette=palette)

        # 测试颜色字典
        palette = dict(zip(hue_levels, color_palette("Reds")))
        m = HueMapping(p, palette=palette)
        # 断言生成的 lookup_table 与指定的颜色字典一致
        assert m.lookup_table == palette

        # 测试颜色字典中删除键的情况
        palette.pop(hue_levels[0])
        with pytest.raises(ValueError):
            HueMapping(p, palette=palette)

        # 测试无效的色彩板名称
        with pytest.raises(ValueError):
            HueMapping(p, palette="not a valid palette")

        # 测试无效的 norm 参数
        with pytest.raises(ValueError):
            HueMapping(p, norm="not a norm")
    # 测试在没有色调数据的情况下的色调映射，使用长格式数据集 `long_df`
    def test_hue_map_without_hue_dataa(self, long_df):
        # 创建一个 VectorPlotter 对象 `p`，使用数据集 `long_df`，指定 x 和 y 变量
        p = VectorPlotter(data=long_df, variables=dict(x="x", y="y"))
        # 使用 pytest 检查是否发出 UserWarning，并匹配特定的警告消息 "Ignoring `palette`"
        with pytest.warns(UserWarning, match="Ignoring `palette`"):
            # 创建一个 HueMapping 对象 `HueMapping(p, palette="viridis")`，此处期望触发警告
            HueMapping(p, palette="viridis")

    # 测试饱和度设置
    def test_saturation(self, long_df):
        # 创建一个 VectorPlotter 对象 `p`，使用数据集 `long_df`，指定 x、y 和 hue="a" 变量
        p = VectorPlotter(data=long_df, variables=dict(x="x", y="y", hue="a"))
        # 获取 `long_df["a"]` 列的分类顺序
        levels = categorical_order(long_df["a"])
        # 使用 "viridis" 调色板生成与 levels 数量相等的调色板
        palette = color_palette("viridis", len(levels))
        # 设置饱和度为 0.8
        saturation = 0.8

        # 创建一个 HueMapping 对象 `m`，使用指定的调色板 `palette` 和饱和度 `saturation`
        m = HueMapping(p, palette=palette, saturation=saturation)
        # 对于每个索引 `i` 和颜色 `color` 在 `m(levels)` 中
        for i, color in enumerate(m(levels)):
            # 使用 assert 语句来确保 `color` 与 `desaturate(palette[i], saturation)` 是相同的颜色
            assert mpl.colors.same_color(color, desaturate(palette[i], saturation))
    # 定义测试类 TestSizeMapping，用于测试 SizeMapping 相关功能
class TestSizeMapping:

    # 测试默认初始化的情况下的 VectorPlotter 对象
    def test_plotter_default_init(self, long_df):
        # 创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 和 y 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        # 断言 p 对象没有 _size_map 属性
        assert not hasattr(p, "_size_map")

        # 再次创建 VectorPlotter 对象，指定 x、y 和 size 变量为 "a"
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="a"),
        )
        # 断言 p 对象的 _size_map 属性是 SizeMapping 类型的实例
        assert isinstance(p._size_map, SizeMapping)
        # 断言 p 对象的 _size_map 的 map_type 属性等于 p 对象中 var_types 字典中 "size" 键对应的值
        assert p._size_map.map_type == p.var_types["size"]

    # 测试自定义设置的情况下的 VectorPlotter 对象
    def test_plotter_customization(self, long_df):
        # 创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x、y 和 size 变量为 "a"
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="a"),
        )
        # 设置 size 的映射表，sizes 指定大小列表，order 指定大小的顺序
        sizes = [1, 4, 2]
        size_order = ["b", "a", "c"]
        p.map_size(sizes=sizes, order=size_order)
        # 断言 p 对象的 _size_map 的 lookup_table 属性与给定的字典一致
        assert p._size_map.lookup_table == dict(zip(size_order, sizes))
        # 断言 p 对象的 _size_map 的 levels 属性与给定的顺序列表 size_order 一致
        assert p._size_map.levels == size_order

    # 测试 size 变量为空值的情况
    def test_size_map_null(self, flat_series, null_series):
        # 创建 VectorPlotter 对象，指定 x 变量为 flat_series，size 变量为 null_series
        p = VectorPlotter(variables=dict(x=flat_series, size=null_series))
        # 创建 HueMapping 对象，传入 p 对象
        m = HueMapping(p)
        # 断言 HueMapping 对象的 levels 属性为 None
        assert m.levels is None
        # 断言 HueMapping 对象的 map_type 属性为 None
        assert m.map_type is None
        # 断言 HueMapping 对象的 norm 属性为 None
        assert m.norm is None
        # 断言 HueMapping 对象的 lookup_table 属性为 None
        assert m.lookup_table is None

    # 测试 SizeMapping 对象对数值类型 size 变量的映射
    def test_map_size_numeric(self, long_df):
        # 创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x、y 和 size 变量为 "s"
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="s"),
        )

        # 测试默认的大小范围
        m = SizeMapping(p)
        size_values = m.lookup_table.values()
        value_range = min(size_values), max(size_values)
        # 断言默认的大小范围与 p 对象的 _default_size_range 属性一致
        assert value_range == p._default_size_range

        # 测试指定的大小值范围
        sizes = 1, 5
        m = SizeMapping(p, sizes=sizes)
        size_values = m.lookup_table.values()
        # 断言指定的大小值范围与实际映射表中的值范围一致
        assert min(size_values), max(size_values) == sizes

        # 测试带有归一化范围的大小值
        norm = 1, 10
        m = SizeMapping(p, sizes=sizes, norm=norm)
        normalize = mpl.colors.Normalize(*norm, clip=True)
        # 断言使用指定的归一化范围后，映射表中的值符合归一化计算的结果
        for key, val in m.lookup_table.items():
            assert val == sizes[0] + (sizes[1] - sizes[0]) * normalize(key)

        # 测试带有归一化对象的大小值
        norm = mpl.colors.LogNorm(1, 10, clip=False)
        m = SizeMapping(p, sizes=sizes, norm=norm)
        # 断言 SizeMapping 对象的 norm 属性的 clip 属性为 True
        assert m.norm.clip
        # 断言使用指定的归一化对象后，映射表中的值符合归一化计算的结果
        for key, val in m.lookup_table.items():
            assert val == sizes[0] + (sizes[1] - sizes[0]) * norm(key)

        # 测试错误的 sizes 参数
        with pytest.raises(ValueError):
            SizeMapping(p, sizes="bad_sizes")

        # 测试错误的 sizes 参数
        with pytest.raises(ValueError):
            SizeMapping(p, sizes=(1, 2, 3))

        # 测试错误的 norm 参数
        with pytest.raises(ValueError):
            SizeMapping(p, norm="bad_norm")
    # 测试映射的大小分类功能
    def test_map_size_categorical(self, long_df):
        # 创建矢量绘图器对象，使用长格式数据，并指定变量的映射关系
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size="a"),
        )

        # 测试指定的大小顺序
        levels = p.plot_data["size"].unique()  # 获取绘图数据中大小变量的唯一值
        sizes = [1, 4, 6]  # 定义测试用的大小列表
        order = [levels[1], levels[2], levels[0]]  # 指定顺序，与绘图数据中大小变量的顺序相对应
        m = SizeMapping(p, sizes=sizes, order=order)  # 创建大小映射对象
        assert m.lookup_table == dict(zip(order, sizes))  # 断言大小映射的结果是否符合预期

        # 测试大小列表
        order = categorical_order(p.plot_data["size"])  # 获取绘图数据中大小变量的排序
        sizes = list(np.random.rand(len(levels)))  # 生成与大小变量数目相同长度的随机大小列表
        m = SizeMapping(p, sizes=sizes)  # 创建大小映射对象
        assert m.lookup_table == dict(zip(order, sizes))  # 断言大小映射的结果是否符合预期

        # 测试大小字典
        sizes = dict(zip(levels, np.random.rand(len(levels))))  # 生成与大小变量数目相同的随机大小字典
        m = SizeMapping(p, sizes=sizes)  # 创建大小映射对象
        assert m.lookup_table == sizes  # 断言大小映射的结果是否符合预期

        # 测试指定的大小范围
        sizes = (2, 5)  # 指定大小范围
        m = SizeMapping(p, sizes=sizes)  # 创建大小映射对象
        values = np.linspace(*sizes, len(m.levels))[::-1]  # 生成指定范围内的等分数值，以及反向排序
        assert m.lookup_table == dict(zip(m.levels, values))  # 断言大小映射的结果是否符合预期

        # 测试显式类别
        p = VectorPlotter(data=long_df, variables=dict(x="x", size="a_cat"))  # 使用分类数据列重新创建矢量绘图器对象
        m = SizeMapping(p)  # 创建大小映射对象
        assert m.levels == long_df["a_cat"].cat.categories.to_list()  # 断言映射对象的类别列表是否符合预期
        assert m.map_type == "categorical"  # 断言映射类型是否为分类类型

        # 测试大小列表长度错误的情况
        sizes = list(np.random.rand(len(levels) + 1))  # 生成一个比大小变量数目多一的随机大小列表
        with pytest.warns(UserWarning):  # 捕获并断言是否会产生用户警告
            SizeMapping(p, sizes=sizes)  # 创建大小映射对象

        # 测试大小字典缺少级别的情况
        sizes = dict(zip(levels, np.random.rand(len(levels) - 1)))  # 生成一个比大小变量数目少一的随机大小字典
        with pytest.raises(ValueError):  # 捕获并断言是否会产生值错误异常
            SizeMapping(p, sizes=sizes)  # 创建大小映射对象

        # 测试错误的大小参数
        with pytest.raises(ValueError):  # 捕获并断言是否会产生值错误异常
            SizeMapping(p, sizes="bad_size")  # 创建大小映射对象

    # 测试数组调色板不推荐使用
    def test_array_palette_deprecation(self, long_df):
        # 创建矢量绘图器对象，使用长格式数据，并指定变量的映射关系
        p = VectorPlotter(long_df, {"y": "y", "hue": "s"})
        pal = mpl.cm.Blues([.3, .8])[:, :3]  # 生成调色板
        with pytest.warns(UserWarning, match="Numpy array is not a supported type"):  # 捕获并断言是否会产生用户警告
            m = HueMapping(p, pal)  # 创建色调映射对象
        assert m.palette == pal.tolist()  # 断言色调映射的调色板是否符合预期
class TestStyleMapping:

    # 测试默认初始化的情况
    def test_plotter_default_init(self, long_df):
        # 创建一个 VectorPlotter 实例，使用长格式数据 long_df，并指定 x 和 y 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        # 断言 p 实例中没有 _map_style 属性
        assert not hasattr(p, "_map_style")

        # 再次创建 VectorPlotter 实例，指定 x、y 和 style 变量为 "a"
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="a"),
        )
        # 断言 p 实例中 _style_map 属性是 StyleMapping 类的实例
        assert isinstance(p._style_map, StyleMapping)

    # 测试样式定制的情况
    def test_plotter_customization(self, long_df):
        # 创建一个 VectorPlotter 实例，指定 x、y 和 style 变量为 "a"
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="a"),
        )
        # 定义标记和样式顺序列表
        markers = ["s", "p", "h"]
        style_order = ["b", "a", "c"]
        # 调用 map_style 方法进行样式映射
        p.map_style(markers=markers, order=style_order)
        # 断言 _style_map 的 levels 属性与 style_order 相同
        assert p._style_map.levels == style_order
        # 断言 _style_map 方法可以正确返回 markers 列表
        assert p._style_map(style_order, "marker") == markers

    # 测试空样式映射的情况
    def test_style_map_null(self, flat_series, null_series):
        # 创建 VectorPlotter 实例，指定 x 变量为 flat_series，style 变量为 null_series
        p = VectorPlotter(variables=dict(x=flat_series, style=null_series))
        # 创建 HueMapping 实例 m
        m = HueMapping(p)
        # 断言 m 的 levels 属性为 None
        assert m.levels is None
        # 断言 m 的 map_type 属性为 None
        assert m.map_type is None
        # 断言 m 的 lookup_table 属性为 None
        assert m.lookup_table is None
    # 定义测试函数，测试样式映射的不同情况
    def test_map_style(self, long_df):

        # 创建一个矢量绘图器对象，传入长格式数据和变量映射字典
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", style="a"),
        )

        # 测试默认设置
        # 创建样式映射对象，启用标记和虚线
        m = StyleMapping(p, markers=True, dashes=True)

        # 获取样式层级的数量
        n = len(m.levels)
        # 遍历样式层级，验证每个层级对应的虚线设置是否正确
        for key, dashes in zip(m.levels, unique_dashes(n)):
            assert m(key, "dashes") == dashes

        # 创建实际标记路径字典，每个样式层级对应一个标记路径对象
        actual_marker_paths = {
            k: mpl.markers.MarkerStyle(m(k, "marker")).get_path()
            for k in m.levels
        }
        # 创建预期标记路径字典，每个样式层级对应一个标记路径对象
        expected_marker_paths = {
            k: mpl.markers.MarkerStyle(m).get_path()
            for k, m in zip(m.levels, unique_markers(n))
        }
        # 验证实际标记路径与预期标记路径是否一致
        assert actual_marker_paths == expected_marker_paths

        # 测试列表设置
        markers, dashes = ["o", "s", "d"], [(1, 0), (1, 1), (2, 1, 3, 1)]
        # 创建样式映射对象，指定标记和虚线列表
        m = StyleMapping(p, markers=markers, dashes=dashes)
        # 遍历样式层级，验证每个层级对应的标记和虚线设置是否正确
        for key, mark, dash in zip(m.levels, markers, dashes):
            assert m(key, "marker") == mark
            assert m(key, "dashes") == dash

        # 测试字典设置
        # 根据数据中的样式唯一值创建标记和虚线字典
        markers = dict(zip(p.plot_data["style"].unique(), markers))
        dashes = dict(zip(p.plot_data["style"].unique(), dashes))
        # 创建样式映射对象，指定标记和虚线字典
        m = StyleMapping(p, markers=markers, dashes=dashes)
        # 遍历样式层级，验证每个层级对应的标记和虚线设置是否正确
        for key in m.levels:
            assert m(key, "marker") == markers[key]
            assert m(key, "dashes") == dashes[key]

        # 测试显式类别设置
        # 使用样式分类列重新创建矢量绘图器对象
        p = VectorPlotter(data=long_df, variables=dict(x="x", style="a_cat"))
        # 创建样式映射对象，自动识别样式分类的层级
        m = StyleMapping(p)
        # 验证样式层级是否与数据中的样式分类类别列表一致
        assert m.levels == long_df["a_cat"].cat.categories.to_list()

        # 测试使用默认顺序设置样式
        # 根据指定顺序创建样式映射对象，启用标记和虚线
        order = p.plot_data["style"].unique()[[1, 2, 0]]
        m = StyleMapping(p, markers=True, dashes=True, order=order)
        # 获取顺序中的样式数量
        n = len(order)
        # 遍历顺序，验证每个样式对应的标记、虚线和路径设置是否正确
        for key, mark, dash in zip(order, unique_markers(n), unique_dashes(n)):
            assert m(key, "dashes") == dash
            assert m(key, "marker") == mark
            obj = mpl.markers.MarkerStyle(mark)
            path = obj.get_path().transformed(obj.get_transform())
            assert_array_equal(m(key, "path").vertices, path.vertices)

        # 测试样式列表中层级过多的情况
        # 测试标记列表过多时发出警告
        with pytest.warns(UserWarning):
            StyleMapping(p, markers=["o", "s"], dashes=False)

        # 测试虚线列表过多时发出警告
        with pytest.warns(UserWarning):
            StyleMapping(p, markers=False, dashes=[(2, 1)])

        # 测试样式字典中缺少键时引发异常
        markers, dashes = {"a": "o", "b": "s"}, False
        with pytest.raises(ValueError):
            StyleMapping(p, markers=markers, dashes=dashes)

        markers, dashes = False, {"a": (1, 0), "b": (2, 1)}
        with pytest.raises(ValueError):
            StyleMapping(p, markers=markers, dashes=dashes)

        # 测试填充和未填充标记混合的情况
        markers, dashes = ["o", "x", "s"], None
        with pytest.raises(ValueError):
            StyleMapping(p, markers=markers, dashes=dashes)
# 定义一个名为 TestVectorPlotter 的测试类
class TestVectorPlotter:

    # 测试方法，测试处理扁平化数据的情况
    def test_flat_variables(self, flat_data):

        # 创建一个 VectorPlotter 实例
        p = VectorPlotter()
        # 调用 assign_variables 方法，将 flat_data 赋给 VectorPlotter 实例 p
        p.assign_variables(data=flat_data)
        # 断言输入格式为 "wide"
        assert p.input_format == "wide"
        # 断言变量列表为 ["x", "y"]
        assert list(p.variables) == ["x", "y"]
        # 断言 plot_data 的长度与 flat_data 的长度相等
        assert len(p.plot_data) == len(flat_data)

        # 尝试获取 flat_data 的索引及其名称
        try:
            expected_x = flat_data.index
            expected_x_name = flat_data.index.name
        except AttributeError:
            # 如果 flat_data 没有索引，则使用长度为 len(flat_data) 的 numpy 数组作为预期的 x 值
            expected_x = np.arange(len(flat_data))
            expected_x_name = None

        # 获取 p 中 plot_data 字典中的 "x" 键对应的值，并断言其与预期的 x 值数组相等
        x = p.plot_data["x"]
        assert_array_equal(x, expected_x)

        # 设置预期的 y 值为 flat_data，获取 p 中 plot_data 字典中的 "y" 键对应的值，并断言其与预期的 y 值相等
        expected_y = flat_data
        expected_y_name = getattr(flat_data, "name", None)
        y = p.plot_data["y"]
        assert_array_equal(y, expected_y)

        # 断言 p 中 variables 字典中的 "x" 键对应的值与预期的 x 名称相等
        assert p.variables["x"] == expected_x_name
        # 断言 p 中 variables 字典中的 "y" 键对应的值与预期的 y 名称相等
        assert p.variables["y"] == expected_y_name

    # 测试方法，测试处理长格式数据框的情况
    def test_long_df(self, long_df, long_variables):

        p = VectorPlotter()
        # 调用 assign_variables 方法，将 long_df 和 long_variables 赋给 VectorPlotter 实例 p
        p.assign_variables(data=long_df, variables=long_variables)
        # 断言输入格式为 "long"
        assert p.input_format == "long"
        # 断言变量字典与 long_variables 相等
        assert p.variables == long_variables

        # 遍历 long_variables 字典，断言 p 中 plot_data 中对应键的值与 long_df 中对应列的值相等
        for key, val in long_variables.items():
            assert_array_equal(p.plot_data[key], long_df[val])

    # 测试方法，测试处理带有索引的长格式数据框的情况
    def test_long_df_with_index(self, long_df, long_variables):

        p = VectorPlotter()
        # 调用 assign_variables 方法，将 long_df 设置为索引为 "a" 的数据框和 long_variables 赋给 VectorPlotter 实例 p
        p.assign_variables(
            data=long_df.set_index("a"),
            variables=long_variables,
        )
        # 断言输入格式为 "long"
        assert p.input_format == "long"
        # 断言变量字典与 long_variables 相等
        assert p.variables == long_variables

        # 遍历 long_variables 字典，断言 p 中 plot_data 中对应键的值与 long_df 中对应列的值相等
        for key, val in long_variables.items():
            assert_array_equal(p.plot_data[key], long_df[val])

    # 测试方法，测试处理带有多重索引的长格式数据框的情况
    def test_long_df_with_multiindex(self, long_df, long_variables):

        p = VectorPlotter()
        # 调用 assign_variables 方法，将 long_df 设置为索引为 ["a", "x"] 的数据框和 long_variables 赋给 VectorPlotter 实例 p
        p.assign_variables(
            data=long_df.set_index(["a", "x"]),
            variables=long_variables,
        )
        # 断言输入格式为 "long"
        assert p.input_format == "long"
        # 断言变量字典与 long_variables 相等
        assert p.variables == long_variables

        # 遍历 long_variables 字典，断言 p 中 plot_data 中对应键的值与 long_df 中对应列的值相等
        for key, val in long_variables.items():
            assert_array_equal(p.plot_data[key], long_df[val])

    # 测试方法，测试处理长格式字典的情况
    def test_long_dict(self, long_dict, long_variables):

        p = VectorPlotter()
        # 调用 assign_variables 方法，将 long_dict 和 long_variables 赋给 VectorPlotter 实例 p
        p.assign_variables(
            data=long_dict,
            variables=long_variables,
        )
        # 断言输入格式为 "long"
        assert p.input_format == "long"
        # 断言变量字典与 long_variables 相等
        assert p.variables == long_variables

        # 遍历 long_variables 字典，断言 p 中 plot_data 中对应键的值与 long_dict 中对应序列的值相等
        for key, val in long_variables.items():
            assert_array_equal(p.plot_data[key], pd.Series(long_dict[val]))

    # 使用 pytest 的参数化标记，测试不同类型的向量处理情况
    @pytest.mark.parametrize(
        "vector_type",
        ["series", "numpy", "list"],
    )
    # 测试处理长格式数据的向量绘图器的函数
    def test_long_vectors(self, long_df, long_variables, vector_type):
        
        # 从长格式数据中选择特定变量组成字典
        variables = {key: long_df[val] for key, val in long_variables.items()}
        
        # 根据指定的向量类型转换变量数据格式
        if vector_type == "numpy":
            variables = {key: val.to_numpy() for key, val in variables.items()}
        elif vector_type == "list":
            variables = {key: val.to_list() for key, val in variables.items()}
        
        # 创建一个向量绘图器对象
        p = VectorPlotter()
        # 将变量分配给绘图器
        p.assign_variables(variables=variables)
        
        # 断言输入数据格式为长格式
        assert p.input_format == "long"
        
        # 断言变量列表与长格式数据变量列表相同
        assert list(p.variables) == list(long_variables)
        
        # 如果向量类型为系列（Series），则断言绘图器的变量与长格式数据变量完全相同
        if vector_type == "series":
            assert p.variables == long_variables
        
        # 遍历长格式数据变量字典，断言绘图数据与长格式数据一致
        for key, val in long_variables.items():
            assert_array_equal(p.plot_data[key], long_df[val])

    # 测试处理长格式数据的向量绘图器在变量未定义时的行为
    def test_long_undefined_variables(self, long_df):
        
        # 创建一个向量绘图器对象
        p = VectorPlotter()

        # 使用 pytest 断言变量未定义时会引发 ValueError 异常
        with pytest.raises(ValueError):
            p.assign_variables(
                data=long_df, variables=dict(x="not_in_df"),
            )

        with pytest.raises(ValueError):
            p.assign_variables(
                data=long_df, variables=dict(x="x", y="not_in_df"),
            )

        with pytest.raises(ValueError):
            p.assign_variables(
                data=long_df, variables=dict(x="x", y="y", hue="not_in_df"),
            )

    # 使用 pytest 参数化装饰器来测试空数据输入时的向量绘图器行为
    @pytest.mark.parametrize(
        "arg", [[], np.array([]), pd.DataFrame()],
    )
    def test_empty_data_input(self, arg):
        
        # 创建一个向量绘图器对象
        p = VectorPlotter()
        
        # 将空数据赋予绘图器
        p.assign_variables(data=arg)
        
        # 断言变量列表为空
        assert not p.variables
        
        # 如果数据不是 DataFrame 类型，则再次创建绘图器并断言变量列表为空
        if not isinstance(arg, pd.DataFrame):
            p = VectorPlotter()
            p.assign_variables(variables=dict(x=arg, y=arg))
            assert not p.variables

    # 测试向量绘图器在使用单位变量时的行为
    def test_units(self, repeated_df):
        
        # 创建一个向量绘图器对象
        p = VectorPlotter()
        
        # 将重复数据赋予绘图器，同时指定 x、y 和单位（units）变量
        p.assign_variables(
            data=repeated_df,
            variables=dict(x="x", y="y", units="u"),
        )
        
        # 断言单位变量的绘图数据与重复数据框中的单位列数据相同
        assert_array_equal(p.plot_data["units"], repeated_df["u"])

    # 使用 pytest 参数化装饰器测试具有数值名称的长格式数据绘图器行为
    @pytest.mark.parametrize("name", [3, 4.5])
    def test_long_numeric_name(self, long_df, name):
        
        # 将指定数值名称的列添加到长格式数据框中作为新的列名
        long_df[name] = long_df["x"]
        
        # 创建一个向量绘图器对象
        p = VectorPlotter()
        
        # 将长格式数据赋予绘图器，并指定变量 x 的数值名称
        p.assign_variables(data=long_df, variables={"x": name})
        
        # 断言绘图数据中的 x 列与长格式数据中的指定数值名称列数据相同
        assert_array_equal(p.plot_data["x"], long_df[name])
        
        # 断言绘图器的 x 变量等于数值名称的字符串表示
        assert p.variables["x"] == str(name)

    # 测试具有层次索引的长格式数据绘图器行为
    def test_long_hierarchical_index(self, rng):
        
        # 创建具有多级索引的数据框
        cols = pd.MultiIndex.from_product([["a"], ["x", "y"]])
        data = rng.uniform(size=(50, 2))
        df = pd.DataFrame(data, columns=cols)
        
        # 指定层次索引中的名称和变量
        name = ("a", "y")
        var = "y"
        
        # 创建一个向量绘图器对象
        p = VectorPlotter()
        
        # 将数据框及其变量赋予绘图器
        p.assign_variables(data=df, variables={var: name})
        
        # 断言绘图数据中的指定变量与数据框中层次索引的列数据相同
        assert_array_equal(p.plot_data[var], df[name])
        
        # 断言绘图器的变量等于层次索引名称的字符串表示
        assert p.variables[var] == str(name)

    # 测试具有标量和数据的长格式数据绘图器行为
    def test_long_scalar_and_data(self, long_df):
        
        # 指定标量值
        val = 22
        
        # 创建一个向量绘图器对象，并将长格式数据及其变量赋予绘图器
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": val})
        
        # 断言绘图数据中的 y 列全等于指定的标量值
        assert (p.plot_data["y"] == val).all()
        
        # 断言绘图器的 y 变量为 None
        assert p.variables["y"] is None
    # 测试宽格式数据语义错误的情况
    def test_wide_semantic_error(self, wide_df):
        # 定义错误消息
        err = "The following variable cannot be assigned with wide-form data: `hue`"
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=err):
            # 尝试创建 VectorPlotter 对象时，传入 wide_df 和错误的变量配置
            VectorPlotter(data=wide_df, variables={"hue": "a"})

    # 测试长格式数据中未知值的错误
    def test_long_unknown_error(self, long_df):
        # 定义错误消息
        err = "Could not interpret value `what` for `hue`"
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=err):
            # 尝试创建 VectorPlotter 对象时，传入 long_df 和错误的变量配置
            VectorPlotter(data=long_df, variables={"x": "x", "hue": "what"})

    # 测试长格式数据中向量长度不匹配的错误
    def test_long_unmatched_size_error(self, long_df, flat_array):
        # 定义错误消息
        err = "Length of ndarray vectors must match length of `data`"
        # 使用 pytest 检查是否抛出 ValueError 异常，并匹配特定错误消息
        with pytest.raises(ValueError, match=err):
            # 尝试创建 VectorPlotter 对象时，传入 long_df 和不匹配长度的变量配置
            VectorPlotter(data=long_df, variables={"x": "x", "hue": flat_array})

    # 测试宽格式数据中分类列的处理
    def test_wide_categorical_columns(self, wide_df):
        # 将 wide_df 的列转换为分类索引
        wide_df.columns = pd.CategoricalIndex(wide_df.columns)
        # 创建 VectorPlotter 对象，传入分类列的 wide_df
        p = VectorPlotter(data=wide_df)
        # 断言 hue 列的唯一值为 ["a", "b", "c"]
        assert_array_equal(p.plot_data["hue"].unique(), ["a", "b", "c"])
    # 定义一个测试方法，用于测试 VectorPlotter 类的数据迭代功能，传入长格式数据框 long_df 作为参数
    def test_iter_data_quantitites(self, long_df):

        # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        # 调用 iter_data 方法，传入 "hue" 作为参数，返回迭代器 out
        out = p.iter_data("hue")
        # 断言迭代器 out 的长度为 1
        assert len(list(out)) == 1

        # 设置变量 var 为 "a"，计算 long_df[var] 的唯一值数量赋值给 n_subsets
        var = "a"
        n_subsets = len(long_df[var].unique())

        # 遍历语义列表 ["hue", "size", "style"]
        for semantic in ["hue", "size", "style"]:
            # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"，语义变量为 semantic: var
            p = VectorPlotter(
                data=long_df,
                variables={"x": "x", "y": "y", semantic: var},
            )
            # 调用 VectorPlotter 对象 p 的 map_{semantic} 方法（例如 map_hue()），操作数据
            getattr(p, f"map_{semantic}")()
            # 调用 iter_data 方法，传入 semantic 列表作为参数，返回迭代器 out
            out = p.iter_data([semantic])
            # 断言迭代器 out 的长度为 n_subsets
            assert len(list(out)) == n_subsets

        # 设置变量 var 为 "a"，计算 long_df[var] 的唯一值数量赋值给 n_subsets
        var = "a"
        n_subsets = len(long_df[var].unique())

        # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"，语义变量为 hue 和 style 均为 var
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var, style=var),
        )
        # 调用 p 的 map_hue() 和 map_style() 方法，操作数据
        p.map_hue()
        p.map_style()
        # 调用 iter_data 方法，传入 semantics 列表作为参数，返回迭代器 out
        out = p.iter_data(semantics)
        # 断言迭代器 out 的长度为 n_subsets
        assert len(list(out)) == n_subsets

        # --

        # 调用 p 的 iter_data 方法，传入 semantics 列表和 reverse=True 参数，返回迭代器 out
        out = p.iter_data(semantics, reverse=True)
        # 断言迭代器 out 的长度为 n_subsets
        assert len(list(out)) == n_subsets

        # --

        # 设置变量 var1 和 var2 为 "a" 和 "s"
        var1, var2 = "a", "s"
        # 计算 long_df[var1] 和 long_df[var2] 列值对的集合长度赋值给 n_subsets
        n_subsets = len(set(list(map(tuple, long_df[[var1, var2]].values))))

        # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"，语义变量为 hue 和 style 分别为 var1 和 var2
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, style=var2),
        )
        # 调用 p 的 map_hue() 和 map_style() 方法，操作数据
        p.map_hue()
        p.map_style()
        # 调用 iter_data 方法，传入 ["hue"] 作为参数，返回迭代器 out
        out = p.iter_data(["hue"])
        # 断言迭代器 out 的长度为 n_subsets
        assert len(list(out)) == n_subsets

        # 计算 long_df[[var1, var2]] 所有行的元组集合长度赋值给 n_subsets
        n_subsets = len(set(list(map(tuple, long_df[[var1, var2]].values))))

        # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"，语义变量为 semantics 列表
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, style=var2),
        )
        # 调用 p 的 map_hue() 和 map_style() 方法，操作数据
        p.map_hue()
        p.map_style()
        # 调用 iter_data 方法，传入 semantics 列表作为参数，返回迭代器 out
        out = p.iter_data(semantics)
        # 断言迭代器 out 的长度为 n_subsets
        assert len(list(out)) == n_subsets

        # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"，语义变量为 hue、size 和 style 分别为 var1、var2 和 var1
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, size=var2, style=var1),
        )
        # 调用 p 的 map_hue()、map_size() 和 map_style() 方法，操作数据
        p.map_hue()
        p.map_size()
        p.map_style()
        # 调用 iter_data 方法，传入 semantics 列表作为参数，返回迭代器 out
        out = p.iter_data(semantics)
        # 断言迭代器 out 的长度为 n_subsets
        assert len(list(out)) == n_subsets

        # --

        # 设置变量 var1、var2 和 var3 为 "a"、"s" 和 "b"，创建包含这些变量的列表 cols
        var1, var2, var3 = "a", "s", "b"
        cols = [var1, var2, var3]
        # 计算 long_df[cols] 所有行的元组集合长度赋值给 n_subsets
        n_subsets = len(set(list(map(tuple, long_df[cols].values))))

        # 创建 VectorPlotter 实例 p，设置数据为 long_df，x 和 y 轴变量为 "x" 和 "y"，语义变量为 hue、size 和 style 分别为 var1、var2 和 var3
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, size=var2, style=var3),
        )
        # 调用 p 的 map_hue()、map_size() 和 map_style() 方法，操作数据
        p.map_hue()
        p.map_size()
        p.map_style()
        # 调用 iter_data 方法，传入 semantics 列表作为参数，返回迭代器 out
        out = p.iter_data(semantics)
        # 断言迭代器 out 的长度为 n_subsets
        assert len(list(out)) == n_subsets
    # 定义测试函数，测试 VectorPlotter 类的 iter_data 方法，用于处理长格式数据
    def test_iter_data_keys(self, long_df):

        # 定义语义列表，指定向量图的语义
        semantics = ["hue", "size", "style"]

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x 和 y 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, _ in p.iter_data("hue"):
            assert sub_vars == {}  # 断言子变量为空字典

        # --

        var = "a"  # 定义变量 var 为 "a"

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y 和 hue 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, _ in p.iter_data("hue"):
            assert list(sub_vars) == ["hue"]  # 断言子变量列表包含 "hue"
            assert sub_vars["hue"] in long_df[var].values  # 断言 hue 值在 long_df[var] 的值中

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y 和 size 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", size=var),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, _ in p.iter_data("size"):
            assert list(sub_vars) == ["size"]  # 断言子变量列表包含 "size"
            assert sub_vars["size"] in long_df[var].values  # 断言 size 值在 long_df[var] 的值中

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y、hue 和 style 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var, style=var),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, _ in p.iter_data(semantics):
            assert list(sub_vars) == ["hue", "style"]  # 断言子变量列表包含 "hue" 和 "style"
            assert sub_vars["hue"] in long_df[var].values  # 断言 hue 值在 long_df[var] 的值中
            assert sub_vars["style"] in long_df[var].values  # 断言 style 值在 long_df[var] 的值中
            assert sub_vars["hue"] == sub_vars["style"]  # 断言 hue 等于 style

        var1, var2 = "a", "s"  # 定义变量 var1 和 var2 分别为 "a" 和 "s"

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y、hue 和 size 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, size=var2),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, _ in p.iter_data(semantics):
            assert list(sub_vars) == ["hue", "size"]  # 断言子变量列表包含 "hue" 和 "size"
            assert sub_vars["hue"] in long_df[var1].values  # 断言 hue 值在 long_df[var1] 的值中
            assert sub_vars["size"] in long_df[var2].values  # 断言 size 值在 long_df[var2] 的值中

        # 重新定义语义列表为 ["hue", "col", "row"]
        semantics = ["hue", "col", "row"]
        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y、hue 和 col 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue=var1, col=var2),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, _ in p.iter_data("hue"):
            assert list(sub_vars) == ["hue", "col"]  # 断言子变量列表包含 "hue" 和 "col"
            assert sub_vars["hue"] in long_df[var1].values  # 断言 hue 值在 long_df[var1] 的值中
            assert sub_vars["col"] in long_df[var2].values  # 断言 col 值在 long_df[var2] 的值中

    # 定义测试函数，测试 VectorPlotter 类的 iter_data 方法，处理长格式数据并验证其值
    def test_iter_data_values(self, long_df):

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x 和 y 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y"),
        )

        # 设置排序为真
        p.sort = True
        # 获取下一个迭代的子变量和子数据
        _, sub_data = next(p.iter_data("hue"))
        # 断言子数据与绘图数据相等
        assert_frame_equal(sub_data, p.plot_data)

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y 和 hue 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a"),
        )

        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, sub_data in p.iter_data("hue"):
            # 根据 hue 值筛选绘图数据的行
            rows = p.plot_data["hue"] == sub_vars["hue"]
            # 断言子数据与筛选后的绘图数据相等
            assert_frame_equal(sub_data, p.plot_data[rows])

        # 创建 VectorPlotter 实例，使用长格式数据 long_df，指定 x、y、hue 和 size 变量
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a", size="s"),
        )
        # 迭代处理数据，对每个子变量和其数据执行断言
        for sub_vars, sub_data in p.iter_data(["hue", "size"]):
            # 根据 hue 和 size 值筛选绘图数据的行
            rows = p.plot_data["hue"] == sub_vars["hue"]
            rows &= p.plot_data["size"] == sub_vars["size"]
            # 断言子数据与筛选后的绘图数据相等
            assert_frame_equal(sub_data, p.plot_data[rows])
    # 测试方法：测试 VectorPlotter 类的 iter_data 方法的反向迭代功能
    def test_iter_data_reverse(self, long_df):

        # 获取长数据框 long_df 中列 'a' 的分类逆序排序
        reversed_order = categorical_order(long_df["a"])[::-1]

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 x、y 和 hue（根据 'a' 列）
        p = VectorPlotter(
            data=long_df,
            variables=dict(x="x", y="y", hue="a")
        )

        # 获取 hue 变量的反向数据迭代器
        iterator = p.iter_data("hue", reverse=True)

        # 遍历迭代器，验证每个子变量集的 hue 值是否与逆序列表中对应的值相匹配
        for i, (sub_vars, _) in enumerate(iterator):
            assert sub_vars["hue"] == reversed_order[i]

    # 测试方法：测试 VectorPlotter 类的 iter_data 方法的丢弃缺失值功能
    def test_iter_data_dropna(self, null_df):

        # 创建 VectorPlotter 实例 p，指定数据为 null_df，变量为 x、y 和 hue（根据 'a' 列）
        p = VectorPlotter(
            data=null_df,
            variables=dict(x="x", y="y", hue="a")
        )

        # 执行 map_hue() 方法，处理 hue 变量
        p.map_hue()

        # 使用 iter_data 方法迭代 hue 变量，确保子数据框不含有 NaN 值
        for _, sub_df in p.iter_data("hue"):
            assert not sub_df.isna().any().any()

        # 再次使用 iter_data 方法迭代 hue 变量，不丢弃 NaN 值，验证是否有缺失值存在
        some_missing = False
        for _, sub_df in p.iter_data("hue", dropna=False):
            some_missing |= sub_df.isna().any().any()
        assert some_missing

    # 测试方法：测试 VectorPlotter 类的 _add_axis_labels 方法
    def test_axis_labels(self, long_df):

        # 创建一个图形窗口 f 和坐标轴 ax
        f, ax = plt.subplots()

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 x（唯一变量）
        p = VectorPlotter(data=long_df, variables=dict(x="a"))

        # 添加坐标轴标签
        p._add_axis_labels(ax)

        # 验证 x 轴和 y 轴标签的正确性
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == ""

        # 清除当前图形窗口的内容
        ax.clear()

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 y（唯一变量）
        p = VectorPlotter(data=long_df, variables=dict(y="a"))

        # 添加坐标轴标签
        p._add_axis_labels(ax)

        # 验证 x 轴和 y 轴标签的正确性
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == "a"

        # 清除当前图形窗口的内容
        ax.clear()

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 x（唯一变量）
        p = VectorPlotter(data=long_df, variables=dict(x="a"))

        # 添加坐标轴标签，指定默认的 y 轴标签为 "default"
        p._add_axis_labels(ax, default_y="default")

        # 验证 x 轴和 y 轴标签的正确性
        assert ax.get_xlabel() == "a"
        assert ax.get_ylabel() == "default"

        # 清除当前图形窗口的内容
        ax.clear()

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 y（唯一变量）
        p = VectorPlotter(data=long_df, variables=dict(y="a"))

        # 添加坐标轴标签，指定默认的 x 轴和 y 轴标签都为 "default"
        p._add_axis_labels(ax, default_x="default", default_y="default")

        # 验证 x 轴和 y 轴标签的正确性
        assert ax.get_xlabel() == "default"
        assert ax.get_ylabel() == "a"

        # 清除当前图形窗口的内容
        ax.clear()

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 x 和 y
        p = VectorPlotter(data=long_df, variables=dict(x="x", y="a"))

        # 在 ax 上手动设置 x 轴和 y 轴标签
        ax.set(xlabel="existing", ylabel="also existing")

        # 添加坐标轴标签，此时不应改变手动设置的标签
        p._add_axis_labels(ax)

        # 验证 x 轴和 y 轴标签的正确性
        assert ax.get_xlabel() == "existing"
        assert ax.get_ylabel() == "also existing"

        # 创建一个包含两个子图的图形窗口 f，获取两个子图的坐标轴 ax1 和 ax2（共享 y 轴）
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量为 x 和 y
        p = VectorPlotter(data=long_df, variables=dict(x="x", y="y"))

        # 分别在 ax1 和 ax2 上添加坐标轴标签
        p._add_axis_labels(ax1)
        p._add_axis_labels(ax2)

        # 验证 ax1 和 ax2 的 x 轴和 y 轴标签正确性，并检查 y 轴标签是否可见
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "y"
        assert ax1.yaxis.label.get_visible()

        assert ax2.get_xlabel() == "x"
        assert ax2.get_ylabel() == "y"
        assert not ax2.yaxis.label.get_visible()

    # 使用 pytest 的参数化功能，测试 VectorPlotter 类的 _attach 方法
    @pytest.mark.parametrize(
        "variables",
        [
            dict(x="x", y="y"),
            dict(x="x"),
            dict(y="y"),
            dict(x="t", y="y"),
            dict(x="x", y="a"),
        ]
    )
    def test_attach_basics(self, long_df, variables):

        # 创建一个图形窗口 _ 和坐标轴 ax
        _, ax = plt.subplots()

        # 创建 VectorPlotter 实例 p，指定数据为 long_df，变量由参数 variables 决定
        p = VectorPlotter(data=long_df, variables=variables)

        # 将 p 实例与 ax 坐标轴关联
        p._attach(ax)

        # 验证 p 的 ax 属性是否与传入的 ax 相同
        assert p.ax is ax
    # 测试函数：测试不允许的附加操作，使用长数据框作为输入
    def test_attach_disallowed(self, long_df):

        # 创建一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"a"
        p = VectorPlotter(data=long_df, variables={"x": "a"})

        # 使用 pytest 来检测是否会抛出 TypeError 异常，因为类型不符合预期
        with pytest.raises(TypeError):
            # 尝试将 VectorPlotter 对象 p 附加到坐标轴 ax 上，但只允许 "numeric" 类型
            p._attach(ax, allowed_types="numeric")

        with pytest.raises(TypeError):
            # 再次尝试附加，但只允许 "datetime" 和 "numeric" 类型
            p._attach(ax, allowed_types=["datetime", "numeric"])

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"
        p = VectorPlotter(data=long_df, variables={"x": "x"})

        with pytest.raises(TypeError):
            # 尝试将 VectorPlotter 对象 p 附加到坐标轴 ax 上，但只允许 "categorical" 类型
            p._attach(ax, allowed_types="categorical")

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"，"y"为"t"
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "t"})

        with pytest.raises(TypeError):
            # 尝试将 VectorPlotter 对象 p 附加到坐标轴 ax 上，但只允许 "numeric" 和 "categorical" 类型
            p._attach(ax, allowed_types=["numeric", "categorical"])

    # 测试函数：测试对数刻度附加功能，使用长数据框作为输入
    def test_attach_log_scale(self, long_df):

        # 创建一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"
        p = VectorPlotter(data=long_df, variables={"x": "x"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，启用对数刻度
        p._attach(ax, log_scale=True)
        # 断言 X 轴的刻度类型为对数刻度，Y 轴的刻度类型为线性刻度
        assert ax.xaxis.get_scale() == "log"
        assert ax.yaxis.get_scale() == "linear"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"
        p = VectorPlotter(data=long_df, variables={"x": "x"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，设置 X 轴为对数刻度，但未设置 Y 轴
        p._attach(ax, log_scale=2)
        # 断言 X 轴的刻度类型为对数刻度，Y 轴的刻度类型为线性刻度
        assert ax.xaxis.get_scale() == "log"
        assert ax.yaxis.get_scale() == "linear"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"y"为"y"
        p = VectorPlotter(data=long_df, variables={"y": "y"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，启用对数刻度
        p._attach(ax, log_scale=True)
        # 断言 X 轴的刻度类型为线性刻度，Y 轴的刻度类型为对数刻度
        assert ax.xaxis.get_scale() == "linear"
        assert ax.yaxis.get_scale() == "log"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"，"y"为"y"
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，启用对数刻度
        p._attach(ax, log_scale=True)
        # 断言 X 轴的刻度类型为对数刻度，Y 轴的刻度类型为对数刻度
        assert ax.xaxis.get_scale() == "log"
        assert ax.yaxis.get_scale() == "log"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"，"y"为"y"
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，设置 X 轴为对数刻度，Y 轴为线性刻度
        p._attach(ax, log_scale=(True, False))
        # 断言 X 轴的刻度类型为对数刻度，Y 轴的刻度类型为线性刻度
        assert ax.xaxis.get_scale() == "log"
        assert ax.yaxis.get_scale() == "linear"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"，"y"为"y"
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，设置 X 轴为线性刻度，Y 轴为对数刻度
        p._attach(ax, log_scale=(False, 2))
        # 断言 X 轴的刻度类型为线性刻度，Y 轴的刻度类型为对数刻度
        assert ax.xaxis.get_scale() == "linear"
        assert ax.yaxis.get_scale() == "log"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"a"，"y"为"y"
        p = VectorPlotter(data=long_df, variables={"x": "a", "y": "y"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，启用对数刻度
        p._attach(ax, log_scale=True)
        # 断言 X 轴的刻度类型为线性刻度，Y 轴的刻度类型为对数刻度
        assert ax.xaxis.get_scale() == "linear"
        assert ax.yaxis.get_scale() == "log"

        # 创建另一个新的图表，并获取坐标轴对象
        _, ax = plt.subplots()
        # 创建一个 VectorPlotter 实例，使用长数据框作为数据，设置变量"x"为"x"，"y"为"t"
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "t"})
        # 将 VectorPlotter 对象 p 附加到坐标轴 ax 上，启用对数刻度
        p._attach(ax, log_scale=True)
        # 断言 X 轴的刻度类型为对数刻度，Y 轴的刻度类型为线性刻度
        assert ax.xaxis.get_scale() == "log"
        assert ax.yaxis.get_scale() ==
    # 测试向量绘图器的_attach方法，将图表附加到给定的轴上
    def test_attach_converters(self, long_df):
        
        # 创建一个新的子图，并返回该子图和轴对象
        _, ax = plt.subplots()
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "t"})
        
        # 将VectorPlotter对象的绘图附加到指定的轴上
        p._attach(ax)
        
        # 断言x轴的转换器为None
        assert ax.xaxis.converter is None
        
        # 断言y轴的转换器类名中包含"Date"
        assert "Date" in ax.yaxis.converter.__class__.__name__

        # 创建另一个新的子图，并返回该子图和轴对象
        _, ax = plt.subplots()
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量
        p = VectorPlotter(data=long_df, variables={"x": "a", "y": "y"})
        
        # 将VectorPlotter对象的绘图附加到指定的轴上
        p._attach(ax)
        
        # 断言x轴的转换器类名中包含"CategoryConverter"
        assert "CategoryConverter" in ax.xaxis.converter.__class__.__name__
        
        # 断言y轴的转换器为None
        assert ax.yaxis.converter is None

    # 测试向量绘图器的_attach方法，将图表附加到给定的FacetGrid对象上
    def test_attach_facets(self, long_df):

        # 创建一个FacetGrid对象，按列"a"分面化长格式数据
        g = FacetGrid(long_df, col="a")
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量，同时将绘图附加到FacetGrid对象上
        p = VectorPlotter(data=long_df, variables={"x": "x", "col": "a"})
        p._attach(g)
        
        # 断言VectorPlotter对象的ax属性为None
        assert p.ax is None
        
        # 断言VectorPlotter对象的facets属性等于之前创建的FacetGrid对象
        assert p.facets == g

    # 测试向量绘图器的_get_scale_transforms方法，对于标识（identity）缩放变换
    def test_scale_transform_identity(self, long_df):

        # 创建一个新的子图，并返回该子图和轴对象
        _, ax = plt.subplots()
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量
        p = VectorPlotter(data=long_df, variables={"x": "x"})
        
        # 将VectorPlotter对象的绘图附加到指定的轴上
        p._attach(ax)
        
        # 获取"x"变量的前向（fwd）和反向（inv）缩放变换函数
        fwd, inv = p._get_scale_transforms("x")

        # 创建一个从1到9的数组
        x = np.arange(1, 10)
        
        # 断言前向变换应保持不变
        assert_array_equal(fwd(x), x)
        
        # 断言反向变换应保持不变
        assert_array_equal(inv(x), x)

    # 测试向量绘图器的_get_scale_transforms方法，对于标识（identity）缩放变换与FacetGrid的结合使用
    def test_scale_transform_identity_facets(self, long_df):

        # 创建一个FacetGrid对象，按列"a"分面化长格式数据
        g = FacetGrid(long_df, col="a")
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量，同时将绘图附加到FacetGrid对象上
        p = VectorPlotter(data=long_df, variables={"x": "x", "col": "a"})
        p._attach(g)

        # 获取"x"变量的前向（fwd）和反向（inv）缩放变换函数
        fwd, inv = p._get_scale_transforms("x")
        
        # 创建一个从1到9的数组
        x = np.arange(1, 10)
        
        # 断言前向变换应保持不变
        assert_array_equal(fwd(x), x)
        
        # 断言反向变换应保持不变
        assert_array_equal(inv(x), x)

    # 测试向量绘图器的_get_scale_transforms方法，对于对数（log）缩放变换
    def test_scale_transform_log(self, long_df):

        # 创建一个新的子图，并返回该子图和轴对象
        _, ax = plt.subplots()
        
        # 设置轴的x轴为对数尺度
        ax.set_xscale("log")
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量
        p = VectorPlotter(data=long_df, variables={"x": "x"})
        
        # 将VectorPlotter对象的绘图附加到指定的轴上
        p._attach(ax)

        # 获取"x"变量的前向（fwd）和反向（inv）缩放变换函数
        fwd, inv = p._get_scale_transforms("x")
        
        # 创建一个从1到3的数组
        x = np.arange(1, 4)
        
        # 断言前向变换应接近于输入数组的以10为底的对数
        assert_array_almost_equal(fwd(x), np.log10(x))
        
        # 断言反向变换应接近于以10为底的输入数组的指数
        assert_array_almost_equal(inv(x), 10 ** x)

    # 测试向量绘图器的_get_scale_transforms方法，对于FacetGrid对象的缩放变换
    def test_scale_transform_facets(self, long_df):

        # 创建一个FacetGrid对象，按列"a"分面化长格式数据
        g = FacetGrid(long_df, col="a")
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量，同时将绘图附加到FacetGrid对象上
        p = VectorPlotter(data=long_df, variables={"x": "x", "col": "a"})
        p._attach(g)

        # 获取"x"变量的前向（fwd）和反向（inv）缩放变换函数
        fwd, inv = p._get_scale_transforms("x")
        
        # 创建一个长度为4的数组
        x = np.arange(4)
        
        # 断言反向变换应保持不变
        assert_array_equal(inv(fwd(x)), x)

    # 测试向量绘图器的_get_scale_transforms方法，对于在FacetGrid对象的混合缩放上的混合使用
    def test_scale_transform_mixed_facets(self, long_df):

        # 创建一个FacetGrid对象，按列"a"分面化长格式数据，并且不共享x轴
        g = FacetGrid(long_df, col="a", sharex=False)
        
        # 设置第一个子图的x轴为对数尺度
        g.axes.flat[0].set_xscale("log")
        
        # 创建VectorPlotter对象，使用长格式数据和指定的变量，同时将绘图附加到FacetGrid对象上
        p = VectorPlotter(data=long_df, variables={"x": "x", "col": "a"})
        p._attach(g)

        # 准备捕获的异常消息
        err = "Cannot determine transform with mixed scales on faceted axes"
        
        # 断言在获取"x"变量的缩放变换时，抛出RuntimeError异常，并且异常消息匹配预期的错误消息
        with pytest.raises(RuntimeError, match=err):
            p._get_scale_transforms("x")
    # 测试方法：测试在不同条件下，向 FacetGrid 对象附加 VectorPlotter 的行为和断言结果
    def test_attach_shared_axes(self, long_df):

        # 创建一个 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df)
        # 创建 VectorPlotter 对象 p，指定数据和变量
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y"})
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量为 1
        assert p.converters["x"].nunique() == 1

        # 创建一个具有列变量 "a" 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a")
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量）
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y", "col": "a"})
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量为 1
        assert p.converters["x"].nunique() == 1
        # 断言 y 变量的唯一值数量为 1
        assert p.converters["y"].nunique() == 1

        # 创建一个具有列变量 "a" 且 sharex=False 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a", sharex=False)
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量）
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y", "col": "a"})
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量等于 plot_data 的列变量 "col" 的唯一值数量
        assert p.converters["x"].nunique() == p.plot_data["col"].nunique()
        # 断言 x 变量按照列变量 "col" 分组后的唯一值数量最大值为 1
        assert p.converters["x"].groupby(p.plot_data["col"]).nunique().max() == 1
        # 断言 y 变量的唯一值数量为 1
        assert p.converters["y"].nunique() == 1

        # 创建一个具有列变量 "a" 且 sharex=False、col_wrap=2 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a", sharex=False, col_wrap=2)
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量）
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "y", "col": "a"})
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量等于 plot_data 的列变量 "col" 的唯一值数量
        assert p.converters["x"].nunique() == p.plot_data["col"].nunique()
        # 断言 x 变量按照列变量 "col" 分组后的唯一值数量最大值为 1
        assert p.converters["x"].groupby(p.plot_data["col"]).nunique().max() == 1
        # 断言 y 变量的唯一值数量为 1
        assert p.converters["y"].nunique() == 1

        # 创建一个具有列变量 "a" 和行变量 "b" 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a", row="b")
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量和行变量）
        p = VectorPlotter(
            data=long_df, variables={"x": "x", "y": "y", "col": "a", "row": "b"},
        )
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量为 1
        assert p.converters["x"].nunique() == 1
        # 断言 y 变量的唯一值数量为 1
        assert p.converters["y"].nunique() == 1

        # 创建一个具有列变量 "a"、行变量 "b" 和 sharex=False 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a", row="b", sharex=False)
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量和行变量）
        p = VectorPlotter(
            data=long_df, variables={"x": "x", "y": "y", "col": "a", "row": "b"},
        )
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量等于 g.axes.flat 的长度（子图数量）
        assert p.converters["x"].nunique() == len(g.axes.flat)
        # 断言 y 变量的唯一值数量为 1
        assert p.converters["y"].nunique() == 1

        # 创建一个具有列变量 "a"、行变量 "b" 和 sharex="col" 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a", row="b", sharex="col")
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量和行变量）
        p = VectorPlotter(
            data=long_df, variables={"x": "x", "y": "y", "col": "a", "row": "b"},
        )
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量等于 plot_data 的列变量 "col" 的唯一值数量
        assert p.converters["x"].nunique() == p.plot_data["col"].nunique()
        # 断言 x 变量按照列变量 "col" 分组后的唯一值数量最大值为 1
        assert p.converters["x"].groupby(p.plot_data["col"]).nunique().max() == 1
        # 断言 y 变量的唯一值数量为 1
        assert p.converters["y"].nunique() == 1

        # 创建一个具有列变量 "a"、行变量 "b" 和 sharey="row" 的 FacetGrid 对象 g，使用长格式数据 long_df
        g = FacetGrid(long_df, col="a", row="b", sharey="row")
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括列变量和行变量）
        p = VectorPlotter(
            data=long_df, variables={"x": "x", "y": "y", "col": "a", "row": "b"},
        )
        # 将 VectorPlotter p 附加到 FacetGrid g
        p._attach(g)
        # 断言 x 变量的唯一值数量为 1
        assert p.converters["x"].nunique() == 1
        # 断言 y 变量的唯一值数量等于 plot_data 的行变量 "row" 的唯一值数量
        assert p.converters["y"].nunique() == p.plot_data["row"].nunique()
        # 断言 y 变量按照行变量 "row" 分组后的唯一值数量最大值为 1
        assert p.converters["y"].groupby(p.plot_data["row"]).nunique().max() == 1

    # 测试方法：测试获取单个轴对象的行为和断言结果
    def test_get_axes_single(self, long_df):

        # 创建一个新的图形，并获取其子图 axes
        ax = plt.figure().subplots()
        # 创建 VectorPlotter 对象 p，指定数据和变量（包括色调变量 "a"）
        p =
    def test_get_axes_facets(self, long_df):
        # 创建 FacetGrid 对象，根据 "a" 列进行分面，长格式数据 long_df
        g = FacetGrid(long_df, col="a")
        # 创建 VectorPlotter 对象，指定数据为 long_df，变量为 {"x": "x", "col": "a"}
        p = VectorPlotter(data=long_df, variables={"x": "x", "col": "a"})
        # 将 VectorPlotter 对象 p 附加到 FacetGrid 对象 g 上
        p._attach(g)
        # 断言获取到的轴对象与 FacetGrid 对象 g 的 axes_dict 中的 "b" 键对应的值相同
        assert p._get_axes({"col": "b"}) is g.axes_dict["b"]

        # 创建 FacetGrid 对象，根据 "a" 和 "c" 列进行分面，长格式数据 long_df
        g = FacetGrid(long_df, col="a", row="c")
        # 创建 VectorPlotter 对象，指定数据为 long_df，变量为 {"x": "x", "col": "a", "row": "c"}
        p = VectorPlotter(
            data=long_df, variables={"x": "x", "col": "a", "row": "c"}
        )
        # 将 VectorPlotter 对象 p 附加到 FacetGrid 对象 g 上
        p._attach(g)
        # 断言获取到的轴对象与 FacetGrid 对象 g 的 axes_dict 中 (1, "b") 键对应的值相同
        assert p._get_axes({"row": 1, "col": "b"}) is g.axes_dict[(1, "b")]

    def test_comp_data(self, long_df):
        # 创建 VectorPlotter 对象，指定数据为 long_df，变量为 {"x": "x", "y": "t"}
        p = VectorPlotter(data=long_df, variables={"x": "x", "y": "t"})

        # 创建子图，返回 Figure 和 Axes 对象
        _, ax = plt.subplots()
        # 将 VectorPlotter 对象 p 附加到 Axes 对象 ax 上
        p._attach(ax)

        # 断言比较 comp_data 中的 "x" 数据与 plot_data 中的 "x" 数据是否相等
        assert_array_equal(p.comp_data["x"], p.plot_data["x"])
        # 断言比较 comp_data 中的 "y" 数据与 plot_data 中的 "y" 数据经过 ax.yaxis 单位转换后是否相等
        assert_array_equal(
            p.comp_data["y"], ax.yaxis.convert_units(p.plot_data["y"])
        )

        # 创建 VectorPlotter 对象，指定数据为 long_df，变量为 {"x": "a"}
        p = VectorPlotter(data=long_df, variables={"x": "a"})

        # 创建子图，返回 Figure 和 Axes 对象
        _, ax = plt.subplots()
        # 将 VectorPlotter 对象 p 附加到 Axes 对象 ax 上
        p._attach(ax)

        # 断言比较 comp_data 中的 "x" 数据与 plot_data 中的 "x" 数据经过 ax.xaxis 单位转换后是否相等
        assert_array_equal(
            p.comp_data["x"], ax.xaxis.convert_units(p.plot_data["x"])
        )

    def test_comp_data_log(self, long_df):
        # 创建 VectorPlotter 对象，指定数据为 long_df，变量为 {"x": "z", "y": "y"}
        p = VectorPlotter(data=long_df, variables={"x": "z", "y": "y"})
        # 创建子图，返回 Figure 和 Axes 对象
        _, ax = plt.subplots()
        # 将 VectorPlotter 对象 p 附加到 Axes 对象 ax 上，并指定对 x 轴使用对数尺度
        p._attach(ax, log_scale=(True, False))

        # 断言比较 comp_data 中的 "x" 数据与 plot_data 中的 "x" 数据经过 np.log10 函数转换后是否相等
        assert_array_equal(
            p.comp_data["x"], np.log10(p.plot_data["x"])
        )
        # 断言比较 comp_data 中的 "y" 数据与 plot_data 中的 "y" 数据是否相等
        assert_array_equal(p.comp_data["y"], p.plot_data["y"])

    def test_comp_data_category_order(self):
        # 创建分类数据 Series s，类别为 ["b", "c", "a"]，有序
        s = (pd.Series(["a", "b", "c", "a"], dtype="category")
             .cat.set_categories(["b", "c", "a"], ordered=True))

        # 创建 VectorPlotter 对象，变量为 {"x": s}
        p = VectorPlotter(variables={"x": s})
        # 创建子图，返回 Figure 和 Axes 对象
        _, ax = plt.subplots()
        # 将 VectorPlotter 对象 p 附加到 Axes 对象 ax 上
        p._attach(ax)
        # 断言比较 comp_data 中的 "x" 数据与预期的顺序是否相等
        assert_array_equal(
            p.comp_data["x"],
            [2, 0, 1, 2],
        )
    # 定义一个名为 comp_data_missing_fixture 的测试夹具函数，接收 request 参数
    def comp_data_missing_fixture(self, request):
        # 从 request.param 中解包出 NA 和 var_type
        NA, var_type = request.param

        # 定义一个包含缺失值的比较数据 comp_data
        comp_data = [0, 1, np.nan, 2, np.nan, 1]

        # 根据 var_type 的不同，设定原始数据 orig_data
        if var_type == "numeric":
            orig_data = [0, 1, NA, 2, np.inf, 1]
        elif var_type == "category":
            orig_data = ["a", "b", NA, "c", pd.NA, "b"]
        elif var_type == "datetime":
            # 使用基于数字的日期处理，避免在 matplotlib<3.2 版本上的问题
            # 等到版本升级后，可以简化测试
            comp_data = [1, 2, np.nan, 3, np.nan, 2]
            numbers = [1, 2, 3, 2]
            
            # 转换数字日期为日期对象
            orig_data = mpl.dates.num2date(numbers)
            # 在指定位置插入 NA 和 np.inf
            orig_data.insert(2, NA)
            orig_data.insert(4, np.inf)

        # 返回原始数据 orig_data 和比较数据 comp_data
        return orig_data, comp_data

    # 定义一个测试函数 test_comp_data_missing，接收 comp_data_missing_fixture 的返回值作为参数
    def test_comp_data_missing(self, comp_data_missing_fixture):
        # 解包 comp_data_missing_fixture 的返回值
        orig_data, comp_data = comp_data_missing_fixture
        
        # 创建一个 VectorPlotter 对象，使用原始数据 orig_data
        p = VectorPlotter(variables={"x": orig_data})
        # 创建一个子图对象
        ax = plt.figure().subplots()
        # 将 VectorPlotter 对象 p 附加到子图 ax 上
        p._attach(ax)
        
        # 断言比较数据 comp_data 与 VectorPlotter 对象 p 中的 "x" 数据相等
        assert_array_equal(p.comp_data["x"], comp_data)
        # 断言 VectorPlotter 对象 p 中的 "x" 数据类型为 float
        assert p.comp_data["x"].dtype == "float"

    # 定义一个测试函数 test_comp_data_duplicate_index
    def test_comp_data_duplicate_index(self):
        # 创建一个带有重复索引的 Series 对象 x
        x = pd.Series([1, 2, 3, 4, 5], [1, 1, 1, 2, 2])
        # 创建一个 VectorPlotter 对象，使用变量 "x"
        p = VectorPlotter(variables={"x": x})
        # 创建一个子图对象
        ax = plt.figure().subplots()
        # 将 VectorPlotter 对象 p 附加到子图 ax 上
        p._attach(ax)
        
        # 断言 VectorPlotter 对象 p 中的 "x" 数据与原始 Series 对象 x 相等
        assert_array_equal(p.comp_data["x"], x)

    # 定义一个测试函数 test_comp_data_nullable_dtype
    def test_comp_data_nullable_dtype(self):
        # 创建一个具有可空数据类型的 Series 对象 x
        x = pd.Series([1, 2, 3, 4], dtype="Int64")
        # 创建一个 VectorPlotter 对象，使用变量 "x"
        p = VectorPlotter(variables={"x": x})
        # 创建一个子图对象
        ax = plt.figure().subplots()
        # 将 VectorPlotter 对象 p 附加到子图 ax 上
        p._attach(ax)
        
        # 断言 VectorPlotter 对象 p 中的 "x" 数据与原始 Series 对象 x 相等
        assert_array_equal(p.comp_data["x"], x)
        # 断言 VectorPlotter 对象 p 中的 "x" 数据类型为 float
        assert p.comp_data["x"].dtype == "float"

    # 定义一个测试函数 test_var_order，接收 long_df 夹具
    def test_var_order(self, long_df):
        # 设定变量的顺序 order
        order = ["c", "b", "a"]
        # 遍历变量列表 ["hue", "size", "style"]
        for var in ["hue", "size", "style"]:
            # 创建一个 VectorPlotter 对象，使用 long_df 数据和变量映射
            p = VectorPlotter(data=long_df, variables={"x": "x", var: "a"})

            # 获取对应 var 的映射函数，并设定顺序 order
            mapper = getattr(p, f"map_{var}")
            mapper(order=order)

            # 断言 VectorPlotter 对象 p 中的 var_levels[var] 等于设定的顺序 order
            assert p.var_levels[var] == order

    # 定义一个测试函数 test_scale_native，接收 long_df 夹具
    def test_scale_native(self, long_df):
        # 创建一个 VectorPlotter 对象，使用 long_df 数据和变量映射
        p = VectorPlotter(data=long_df, variables={"x": "x"})
        
        # 断言调用 scale_native 方法会抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            p.scale_native("x")

    # 定义一个测试函数 test_scale_numeric，接收 long_df 夹具
    def test_scale_numeric(self, long_df):
        # 创建一个 VectorPlotter 对象，使用 long_df 数据和变量映射
        p = VectorPlotter(data=long_df, variables={"y": "y"})
        
        # 断言调用 scale_numeric 方法会抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            p.scale_numeric("y")

    # 定义一个测试函数 test_scale_datetime，接收 long_df 夹具
    def test_scale_datetime(self, long_df):
        # 创建一个 VectorPlotter 对象，使用 long_df 数据和变量映射
        p = VectorPlotter(data=long_df, variables={"x": "t"})
        
        # 断言调用 scale_datetime 方法会抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            p.scale_datetime("x")
    # 定义一个测试方法，用于测试 VectorPlotter 类中关于类别变量缩放的功能
    def test_scale_categorical(self, long_df):
        
        # 创建一个 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 变量为 "x"
        p = VectorPlotter(data=long_df, variables={"x": "x"})
        
        # 对变量 "y" 进行类别变量缩放
        p.scale_categorical("y")
        
        # 断言变量 "y" 在 VectorPlotter 对象中为 None
        assert p.variables["y"] is None
        
        # 断言变量 "y" 在 VectorPlotter 对象的变量类型字典中为 "categorical"
        assert p.var_types["y"] == "categorical"
        
        # 断言变量 "y" 在 VectorPlotter 对象的绘图数据中所有元素为空字符串
        assert (p.plot_data["y"] == "").all()
        
        # 重新创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 变量为 "s"
        p = VectorPlotter(data=long_df, variables={"x": "s"})
        
        # 对变量 "x" 进行类别变量缩放
        p.scale_categorical("x")
        
        # 断言变量 "x" 在 VectorPlotter 对象的变量类型字典中为 "categorical"
        assert p.var_types["x"] == "categorical"
        
        # 断言变量 "x" 在 VectorPlotter 对象的绘图数据中具有 "str" 属性
        assert hasattr(p.plot_data["x"], "str")
        
        # 断言变量 "x" 不是有序变量
        assert not p._var_ordered["x"]
        
        # 断言变量 "x" 在 VectorPlotter 对象的绘图数据中是单调递增的
        assert p.plot_data["x"].is_monotonic_increasing
        
        # 断言变量 "x" 在 VectorPlotter 对象的变量水平中与绘图数据中的唯一值相等
        assert_array_equal(p.var_levels["x"], p.plot_data["x"].unique())
        
        # 重新创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 变量为 "a"
        p = VectorPlotter(data=long_df, variables={"x": "a"})
        
        # 对变量 "x" 进行类别变量缩放
        p.scale_categorical("x")
        
        # 断言变量 "x" 不是有序变量
        assert not p._var_ordered["x"]
        
        # 断言变量 "x" 在 VectorPlotter 对象的变量水平中与长格式数据 long_df 中 "a" 列的分类顺序相等
        assert_array_equal(p.var_levels["x"], categorical_order(long_df["a"]))
        
        # 重新创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 变量为 "a_cat"
        p = VectorPlotter(data=long_df, variables={"x": "a_cat"})
        
        # 对变量 "x" 进行类别变量缩放
        p.scale_categorical("x")
        
        # 断言变量 "x" 是有序变量
        assert p._var_ordered["x"]
        
        # 断言变量 "x" 在 VectorPlotter 对象的变量水平中与长格式数据 long_df 中 "a_cat" 列的分类顺序相等
        assert_array_equal(p.var_levels["x"], categorical_order(long_df["a_cat"]))
        
        # 重新创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 变量为 "a"
        p = VectorPlotter(data=long_df, variables={"x": "a"})
        
        # 生成一个新的顺序数组，将 long_df 中 "a" 列的唯一值向前移动一位
        order = np.roll(long_df["a"].unique(), 1)
        
        # 对变量 "x" 进行类别变量缩放，并指定顺序
        p.scale_categorical("x", order=order)
        
        # 断言变量 "x" 是有序变量
        assert p._var_ordered["x"]
        
        # 断言变量 "x" 在 VectorPlotter 对象的变量水平中与指定顺序数组 order 相等
        assert_array_equal(p.var_levels["x"], order)
        
        # 重新创建 VectorPlotter 对象，使用长格式数据 long_df，并指定 x 变量为 "s"
        p = VectorPlotter(data=long_df, variables={"x": "s"})
        
        # 对变量 "x" 进行类别变量缩放，并指定格式化函数为 lambda x: f"{x:%}"
        p.scale_categorical("x", formatter=lambda x: f"{x:%}")
        
        # 断言变量 "x" 在 VectorPlotter 对象的绘图数据中所有字符串元素以 "%" 结尾
        assert p.plot_data["x"].str.endswith("%").all()
        
        # 断言变量 "x" 在 VectorPlotter 对象的变量水平中所有字符串元素以 "%" 结尾
        assert all(s.endswith("%") for s in p.var_levels["x"])
class TestCoreFunc:
    
    def test_unique_dashes(self):
        
        # 设置测试用例数量
        n = 24
        # 调用 unique_dashes 函数，获取结果
        dashes = unique_dashes(n)
        
        # 断言：返回的列表长度应为 n
        assert len(dashes) == n
        # 断言：列表中的元素应该唯一
        assert len(set(dashes)) == n
        # 断言：第一个元素应为空字符串
        assert dashes[0] == ""
        # 遍历除第一个元素外的所有元素
        for spec in dashes[1:]:
            # 断言：每个元素应为元组
            assert isinstance(spec, tuple)
            # 断言：每个元组的长度应为偶数
            assert not len(spec) % 2
    
    def test_unique_markers(self):
        
        # 设置测试用例数量
        n = 24
        # 调用 unique_markers 函数，获取结果
        markers = unique_markers(n)
        
        # 断言：返回的列表长度应为 n
        assert len(markers) == n
        # 断言：列表中的元素应该唯一
        assert len(set(markers)) == n
        # 遍历所有标记
        for m in markers:
            # 断言：使用 Matplotlib 的 MarkerStyle 判断标记是否是填充的
            assert mpl.markers.MarkerStyle(m).is_filled()

    def test_variable_type(self):
        
        # 创建一个包含浮点数的 Pandas Series
        s = pd.Series([1., 2., 3.])
        # 断言：检查 Series 的变量类型是否为 "numeric"
        assert variable_type(s) == "numeric"
        # 断言：将 Series 转换为整数类型后，变量类型仍然是 "numeric"
        assert variable_type(s.astype(int)) == "numeric"
        # 断言：将 Series 转换为对象类型后，变量类型仍然是 "numeric"
        assert variable_type(s.astype(object)) == "numeric"
        # 断言：将 Series 转换为 NumPy 数组后，变量类型仍然是 "numeric"
        assert variable_type(s.to_numpy()) == "numeric"
        # 断言：将 Series 转换为列表后，变量类型仍然是 "numeric"
        assert variable_type(s.to_list()) == "numeric"
        
        # 创建一个包含对象类型的 Pandas Series
        s = pd.Series([1, 2, 3, np.nan], dtype=object)
        # 断言：检查 Series 的变量类型是否为 "numeric"
        assert variable_type(s) == "numeric"
        
        # 创建一个包含 NaN 值的 Pandas Series
        s = pd.Series([np.nan, np.nan])
        # 断言：检查 Series 的变量类型是否为 "numeric"
        assert variable_type(s) == "numeric"
        
        # 创建一个包含 Pandas NA 值的 Pandas Series
        s = pd.Series([pd.NA, pd.NA])
        # 断言：检查 Series 的变量类型是否为 "numeric"
        assert variable_type(s) == "numeric"
        
        # 创建一个包含 Int64 类型和 Pandas NA 值的 Pandas Series
        s = pd.Series([1, 2, pd.NA], dtype="Int64")
        # 断言：检查 Series 的变量类型是否为 "numeric"
        assert variable_type(s) == "numeric"
        
        # 创建一个包含字符串的 Pandas Series
        s = pd.Series(["1", "2", "3"])
        # 断言：检查 Series 的变量类型是否为 "categorical"
        assert variable_type(s) == "categorical"
        # 断言：将 Series 转换为 NumPy 数组后，变量类型仍然是 "categorical"
        assert variable_type(s.to_numpy()) == "categorical"
        # 断言：将 Series 转换为列表后，变量类型仍然是 "categorical"
        assert variable_type(s.to_list()) == "categorical"
        
        # 创建一个包含 timedelta 类型的 Pandas Series
        s = pd.timedelta_range(1, periods=3, freq="D").to_series()
        # 断言：检查 Series 的变量类型是否为 "categorical"
        assert variable_type(s) == "categorical"
        
        # 创建一个包含布尔值的 Pandas Series
        s = pd.Series([True, False, False])
        # 断言：检查 Series 的变量类型是否为 "numeric"
        assert variable_type(s) == "numeric"
        # 断言：将 boolean_type 参数设置为 "categorical" 后，变量类型变为 "categorical"
        assert variable_type(s, boolean_type="categorical") == "categorical"
        # 将 Series 转换为分类类型后，变量类型仍然是 "categorical"
        s_cat = s.astype("category")
        assert variable_type(s_cat, boolean_type="categorical") == "categorical"
        # 将 Series 转换为分类类型后，将 boolean_type 参数设置为 "numeric"，变量类型变为 "categorical"
        assert variable_type(s_cat, boolean_type="numeric") == "categorical"
        
        # 创建一个包含日期时间类型的 Pandas Series
        s = pd.Series([pd.Timestamp(1), pd.Timestamp(2)])
        # 断言：检查 Series 的变量类型是否为 "datetime"
        assert variable_type(s) == "datetime"
        # 断言：将 Series 转换为对象类型后，变量类型仍然是 "datetime"
        assert variable_type(s.astype(object)) == "datetime"
        # 断言：将 Series 转换为 NumPy 数组后，变量类型仍然是 "datetime"
        assert variable_type(s.to_numpy()) == "datetime"
        # 断言：将 Series 转换为列表后，变量类型仍然是 "datetime"
        assert variable_type(s.to_list()) == "datetime"
    # 定义单元测试函数 test_infer_orient(self)
    def test_infer_orient(self):
        
        # 创建包含数字的 pandas Series
        nums = pd.Series(np.arange(6))
        # 创建包含字符串的 pandas Series
        cats = pd.Series(["a", "b"] * 3)
        # 创建日期范围的 pandas Series
        dates = pd.date_range("1999-09-22", "2006-05-14", 6)

        # 断言根据不同数据类型推断方向，预期结果为 "x"
        assert infer_orient(cats, nums) == "x"
        # 断言根据不同数据类型推断方向，预期结果为 "y"
        assert infer_orient(nums, cats) == "y"

        # 断言根据不同数据类型推断方向，设置 require_numeric=False，预期结果为 "x"
        assert infer_orient(cats, dates, require_numeric=False) == "x"
        # 断言根据不同数据类型推断方向，设置 require_numeric=False，预期结果为 "y"
        assert infer_orient(dates, cats, require_numeric=False) == "y"

        # 断言根据数据类型推断方向，nums 为非 None，预期结果为 "y"
        assert infer_orient(nums, None) == "y"
        # 使用 pytest 监控 UserWarning 异常，断言根据数据类型推断方向，nums 为 None 且 orient="v"，预期结果为 "y"
        with pytest.warns(UserWarning, match="Vertical .+ `x`"):
            assert infer_orient(nums, None, "v") == "y"

        # 断言根据数据类型推断方向，nums 为 None，预期结果为 "x"
        assert infer_orient(None, nums) == "x"
        # 使用 pytest 监控 UserWarning 异常，断言根据数据类型推断方向，nums 为 None 且 orient="h"，预期结果为 "x"
        with pytest.warns(UserWarning, match="Horizontal .+ `y`"):
            assert infer_orient(None, nums, "h") == "x"

        # 调用 infer_orient，cats 为非 None，设置 require_numeric=False，预期结果为 "y"（未使用 assert）
        infer_orient(cats, None, require_numeric=False) == "y"
        # 使用 pytest 触发 TypeError 异常，断言 cats 为 None 时调用 infer_orient，预期结果为异常匹配 "Horizontal .+ `x`"
        with pytest.raises(TypeError, match="Horizontal .+ `x`"):
            infer_orient(cats, None)

        # 调用 infer_orient，cats 为非 None，设置 require_numeric=False，预期结果为 "x"（未使用 assert）
        infer_orient(cats, None, require_numeric=False) == "x"
        # 使用 pytest 触发 TypeError 异常，断言 cats 为 None 时调用 infer_orient，预期结果为异常匹配 "Vertical .+ `y`"
        with pytest.raises(TypeError, match="Vertical .+ `y`"):
            infer_orient(None, cats)

        # 断言根据数据类型推断方向，nums 与 nums 相同，orient="vert"，预期结果为 "x"
        assert infer_orient(nums, nums, "vert") == "x"
        # 断言根据数据类型推断方向，nums 与 nums 相同，orient="hori"，预期结果为 "y"
        assert infer_orient(nums, nums, "hori") == "y"

        # 断言根据数据类型推断方向，cats 与 cats 相同，orient="h"，设置 require_numeric=False，预期结果为 "y"
        assert infer_orient(cats, cats, "h", require_numeric=False) == "y"
        # 断言根据数据类型推断方向，cats 与 cats 相同，orient="v"，设置 require_numeric=False，预期结果为 "x"
        assert infer_orient(cats, cats, "v", require_numeric=False) == "x"
        # 断言根据数据类型推断方向，cats 与 cats 相同，设置 require_numeric=False，预期结果为 "x"
        assert infer_orient(cats, cats, require_numeric=False) == "x"

        # 使用 pytest 触发 TypeError 异常，断言 cats 与 cats 相同，orient="x" 时调用 infer_orient，预期结果为异常匹配 "Vertical .+ `y`"
        with pytest.raises(TypeError, match="Vertical .+ `y`"):
            infer_orient(cats, cats, "x")
        # 使用 pytest 触发 TypeError 异常，断言 cats 与 cats 相同，orient="y" 时调用 infer_orient，预期结果为异常匹配 "Horizontal .+ `x`"
        with pytest.raises(TypeError, match="Horizontal .+ `x`"):
            infer_orient(cats, cats, "y")
        # 使用 pytest 触发 ValueError 异常，断言调用 infer_orient 时未提供 orient 参数，预期结果为异常匹配 "`orient` must start with"
        with pytest.raises(ValueError, match="`orient` must start with"):
            infer_orient(cats, nums, orient="bad value")
    # 定义一个测试函数，测试 categorical_order 函数的不同用法和返回结果
    def test_categorical_order(self):
    
        # 准备测试数据
        x = ["a", "c", "c", "b", "a", "d"]  # 字符串列表
        y = [3, 2, 5, 1, 4]  # 数字列表
        order = ["a", "b", "c", "d"]  # 预定义的顺序列表
    
        # 调用 categorical_order 函数，不指定 order 参数，期望返回按照默认顺序排序的结果
        out = categorical_order(x)
        assert out == ["a", "c", "b", "d"]
    
        # 调用 categorical_order 函数，指定 order 参数为预定义的顺序列表，期望返回顺序列表
        out = categorical_order(x, order)
        assert out == order
    
        # 调用 categorical_order 函数，指定 order 参数为 ["b", "a"]，期望返回 ["b", "a"] 的顺序
        out = categorical_order(x, ["b", "a"])
        assert out == ["b", "a"]
    
        # 调用 categorical_order 函数，传入 numpy 数组，期望返回按默认顺序排序的结果
        out = categorical_order(np.array(x))
        assert out == ["a", "c", "b", "d"]
    
        # 调用 categorical_order 函数，传入 pandas Series 对象，期望返回按默认顺序排序的结果
        out = categorical_order(pd.Series(x))
        assert out == ["a", "c", "b", "d"]
    
        # 调用 categorical_order 函数，传入数字列表 y，期望返回按默认顺序排序的结果
        out = categorical_order(y)
        assert out == [1, 2, 3, 4, 5]
    
        # 调用 categorical_order 函数，传入 numpy 数组，期望返回按默认顺序排序的结果
        out = categorical_order(np.array(y))
        assert out == [1, 2, 3, 4, 5]
    
        # 调用 categorical_order 函数，传入 pandas Series 对象，期望返回按默认顺序排序的结果
        out = categorical_order(pd.Series(y))
        assert out == [1, 2, 3, 4, 5]
    
        # 将 x 转换为 pandas Categorical 对象，指定顺序为预定义的顺序列表 order
        x = pd.Categorical(x, order)
        out = categorical_order(x)
        # 期望返回的结果为 x 的分类的有序列表
        assert out == list(x.categories)
    
        # 将 x 转换为 pandas Series 对象，并再次调用 categorical_order 函数，期望返回按默认顺序排序的结果
        x = pd.Series(x)
        out = categorical_order(x)
        # 期望返回的结果为 x 的分类的有序列表
        assert out == list(x.cat.categories)
    
        # 调用 categorical_order 函数，传入参数 x 和 ["b", "a"]，期望返回 ["b", "a"] 的顺序
        out = categorical_order(x, ["b", "a"])
        assert out == ["b", "a"]
    
        # 准备新的测试数据 x，包含一个 NaN 值
        x = ["a", np.nan, "c", "c", "b", "a", "d"]
        # 调用 categorical_order 函数，期望返回按默认顺序排序的结果
        out = categorical_order(x)
        assert out == ["a", "c", "b", "d"]
```