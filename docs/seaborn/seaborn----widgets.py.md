# `D:\src\scipysrc\seaborn\seaborn\widgets.py`

```
# 导入numpy库，用于数值计算
import numpy as np
# 导入matplotlib.pyplot库，并用plt作为别名，用于绘图
import matplotlib.pyplot as plt
# 从matplotlib.colors中导入LinearSegmentedColormap类
from matplotlib.colors import LinearSegmentedColormap

# 尝试导入ipywidgets库中的interact、FloatSlider、IntSlider模块
try:
    from ipywidgets import interact, FloatSlider, IntSlider
# 处理ImportError异常，定义一个空函数interact以便在缺少ipywidgets时提供友好错误信息
except ImportError:
    def interact(f):
        msg = "Interactive palettes require `ipywidgets`, which is not installed."
        raise ImportError(msg)

# 导入当前包中的miscplot模块中的palplot函数
from .miscplot import palplot
# 导入当前包中的palettes模块中的color_palette、dark_palette、light_palette、diverging_palette、cubehelix_palette函数
from .palettes import (color_palette, dark_palette, light_palette,
                       diverging_palette, cubehelix_palette)

# 指定模块中公开的接口列表
__all__ = ["choose_colorbrewer_palette", "choose_cubehelix_palette",
           "choose_dark_palette", "choose_light_palette",
           "choose_diverging_palette"]

# 创建一个可变的matplotlib colormap，将会被小部件更新
def _init_mutable_colormap():
    """Create a matplotlib colormap that will be updated by the widgets."""
    # 使用"Greys"调色板创建灰度颜色列表
    greys = color_palette("Greys", 256)
    # 创建一个名为"interactive"的线性分段色彩映射对象
    cmap = LinearSegmentedColormap.from_list("interactive", greys)
    # 初始化色彩映射对象的内部状态
    cmap._init()
    # 设置色彩映射对象的极端值
    cmap._set_extremes()
    return cmap

# 在一个matplotlib colormap中原地修改LUT值（Look-Up Table）
def _update_lut(cmap, colors):
    """Change the LUT values in a matplotlib colormap in-place."""
    # 将输入的颜色值替换到色彩映射对象的前256个索引位置
    cmap._lut[:256] = colors
    # 更新色彩映射对象的极端值
    cmap._set_extremes()

# 显示一个连续的matplotlib colormap
def _show_cmap(cmap):
    """Show a continuous matplotlib colormap."""
    # 导入rcmod模块中的axes_style函数，避免循环导入
    from .rcmod import axes_style  # Avoid circular import
    # 使用"white"风格设置图形的背景
    with axes_style("white"):
        # 创建一个大小为(8.25, .75)的子图
        f, ax = plt.subplots(figsize=(8.25, .75))
    # 设置图形的x轴和y轴刻度为空
    ax.set(xticks=[], yticks=[])
    # 在子图中使用色彩映射对象绘制一个彩色网格
    x = np.linspace(0, 1, 256)[np.newaxis, :]
    ax.pcolormesh(x, cmap=cmap)

# 从ColorBrewer调色板中选择一个调色板
def choose_colorbrewer_palette(data_type, as_cmap=False):
    """Select a palette from the ColorBrewer set.

    These palettes are built into matplotlib and can be used by name in
    many seaborn functions, or by passing the object returned by this function.

    Parameters
    ----------
    data_type : {'sequential', 'diverging', 'qualitative'}
        This describes the kind of data you want to visualize. See the seaborn
        color palette docs for more information about how to choose this value.
        Note that you can pass substrings (e.g. 'q' for 'qualitative.

    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.
    diverging_palette : Create a diverging palette from selected colors.
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.
    """
    # 如果数据类型以"q"开头并且要求返回的是色彩映射对象，则抛出异常
    if data_type.startswith("q") and as_cmap:
        raise ValueError("Qualitative palettes cannot be colormaps.")
    
    # 初始化一个空列表用于存储颜色值
    pal = []
    # 如果要求返回的是色彩映射对象，则初始化一个可变的色彩映射对象
    if as_cmap:
        cmap = _init_mutable_colormap()
    # 如果数据类型以 "s" 开头，选择顺序调色板
    if data_type.startswith("s"):
        # 可选的调色板名称列表
        opts = ["Greys", "Reds", "Greens", "Blues", "Oranges", "Purples",
                "BuGn", "BuPu", "GnBu", "OrRd", "PuBu", "PuRd", "RdPu", "YlGn",
                "PuBuGn", "YlGnBu", "YlOrBr", "YlOrRd"]
        # 可选的变体类型
        variants = ["regular", "reverse", "dark"]

        # 定义一个交互式函数，用于选择顺序调色板及其参数
        @interact
        def choose_sequential(name=opts, n=(2, 18),
                              desat=FloatSlider(min=0, max=1, value=1),
                              variant=variants):
            # 根据变体类型调整调色板名称
            if variant == "reverse":
                name += "_r"
            elif variant == "dark":
                name += "_d"

            # 如果需要作为颜色映射使用
            if as_cmap:
                # 获取指定名称和参数的颜色列表
                colors = color_palette(name, 256, desat)
                # 更新颜色映射的颜色表
                _update_lut(cmap, np.c_[colors, np.ones(256)])
                # 显示颜色映射
                _show_cmap(cmap)
            else:
                # 更新调色板中的颜色列表
                pal[:] = color_palette(name, n, desat)
                # 显示调色板
                palplot(pal)

    # 如果数据类型以 "d" 开头，选择离散调色板
    elif data_type.startswith("d"):
        # 可选的调色板名称列表
        opts = ["RdBu", "RdGy", "PRGn", "PiYG", "BrBG",
                "RdYlBu", "RdYlGn", "Spectral"]
        # 可选的变体类型
        variants = ["regular", "reverse"]

        # 定义一个交互式函数，用于选择离散调色板及其参数
        @interact
        def choose_diverging(name=opts, n=(2, 16),
                             desat=FloatSlider(min=0, max=1, value=1),
                             variant=variants):
            # 根据变体类型调整调色板名称
            if variant == "reverse":
                name += "_r"
            # 如果需要作为颜色映射使用
            if as_cmap:
                # 获取指定名称和参数的颜色列表
                colors = color_palette(name, 256, desat)
                # 更新颜色映射的颜色表
                _update_lut(cmap, np.c_[colors, np.ones(256)])
                # 显示颜色映射
                _show_cmap(cmap)
            else:
                # 更新调色板中的颜色列表
                pal[:] = color_palette(name, n, desat)
                # 显示调色板
                palplot(pal)

    # 如果数据类型以 "q" 开头，选择定性调色板
    elif data_type.startswith("q"):
        # 可选的调色板名称列表
        opts = ["Set1", "Set2", "Set3", "Paired", "Accent",
                "Pastel1", "Pastel2", "Dark2"]

        # 定义一个交互式函数，用于选择定性调色板及其参数
        @interact
        def choose_qualitative(name=opts, n=(2, 16),
                               desat=FloatSlider(min=0, max=1, value=1)):
            # 更新调色板中的颜色列表
            pal[:] = color_palette(name, n, desat)
            # 显示调色板
            palplot(pal)

    # 如果需要作为颜色映射使用，则返回颜色映射对象
    if as_cmap:
        return cmap
    # 否则返回调色板对象
    return pal
# 启动交互式小部件以创建暗色顺序调色板
# 与 dark_palette 函数对应。这种调色板适用于数据，其范围从相对无趣的低值到有趣的高值。

# 需要 IPython 2+ 并且必须在笔记本中使用。

# 定义函数 choose_dark_palette，接受两个参数：input 和 as_cmap
def choose_dark_palette(input="husl", as_cmap=False):

    # 创建一个空列表 pal 用于存储调色板的颜色
    pal = []

    # 如果参数 as_cmap 为 True，则创建一个可变的 colormap 对象 cmap
    if as_cmap:
        cmap = _init_mutable_colormap()

    # 如果 input 参数为 "rgb"
    if input == "rgb":

        # 定义一个交互函数 choose_dark_palette_rgb，使用 IPython 的 @interact 装饰器
        @interact
        def choose_dark_palette_rgb(r=(0., 1.),
                                    g=(0., 1.),
                                    b=(0., 1.),
                                    n=(3, 17)):
            # 将 r, g, b 参数组合成一个颜色元组 color
            color = r, g, b

            # 如果 as_cmap 为 True，则生成一个包含 256 个颜色的 dark_palette
            # 并更新 colormap 对象 cmap
            if as_cmap:
                colors = dark_palette(color, 256, input="rgb")
                _update_lut(cmap, colors)
                _show_cmap(cmap)
            else:
                # 否则，生成一个包含 n 个颜色的 dark_palette，并将结果赋给 pal 列表
                pal[:] = dark_palette(color, n, input="rgb")
                # 绘制调色板
                palplot(pal)

    # 如果 input 参数为 "hls"
    elif input == "hls":

        # 定义一个交互函数 choose_dark_palette_hls，使用 IPython 的 @interact 装饰器
        @interact
        def choose_dark_palette_hls(h=(0., 1.),
                                    l=(0., 1.),  # noqa: E741
                                    s=(0., 1.),
                                    n=(3, 17)):
            # 将 h, l, s 参数组合成一个颜色元组 color
            color = h, l, s

            # 如果 as_cmap 为 True，则生成一个包含 256 个颜色的 dark_palette
            # 并更新 colormap 对象 cmap
            if as_cmap:
                colors = dark_palette(color, 256, input="hls")
                _update_lut(cmap, colors)
                _show_cmap(cmap)
            else:
                # 否则，生成一个包含 n 个颜色的 dark_palette，并将结果赋给 pal 列表
                pal[:] = dark_palette(color, n, input="hls")
                # 绘制调色板
                palplot(pal)

    # 如果 input 参数为 "husl"
    elif input == "husl":

        # 定义一个交互函数 choose_dark_palette_husl，使用 IPython 的 @interact 装饰器
        @interact
        def choose_dark_palette_husl(h=(0, 359),
                                     s=(0, 99),
                                     l=(0, 99),  # noqa: E741
                                     n=(3, 17)):
            # 将 h, s, l 参数组合成一个颜色元组 color
            color = h, s, l

            # 如果 as_cmap 为 True，则生成一个包含 256 个颜色的 dark_palette
            # 并更新 colormap 对象 cmap
            if as_cmap:
                colors = dark_palette(color, 256, input="husl")
                _update_lut(cmap, colors)
                _show_cmap(cmap)
            else:
                # 否则，生成一个包含 n 个颜色的 dark_palette，并将结果赋给 pal 列表
                pal[:] = dark_palette(color, n, input="husl")
                # 绘制调色板
                palplot(pal)

    # 如果 as_cmap 参数为 True，则返回 colormap 对象 cmap
    if as_cmap:
        return cmap
    # 否则，返回调色板列表 pal
    return pal
    """
    Launch an interactive widget to create a light sequential palette.
    
    This corresponds with the :func:`light_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values.
    
    Requires IPython 2+ and must be used in the notebook.
    
    Parameters
    ----------
    input : {'husl', 'hls', 'rgb'}
        Color space for defining the seed value. Note that the default is
        different than the default input for :func:`light_palette`.
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.
    
    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.
    
    See Also
    --------
    light_palette : Create a sequential palette with bright low values.
    dark_palette : Create a sequential palette with dark low values.
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.
    """
    pal = []  # 初始化空的调色板列表
    
    if as_cmap:
        cmap = _init_mutable_colormap()  # 如果需要生成 colormap，则初始化一个可变的 colormap
    
    if input == "rgb":
        @interact
        def choose_light_palette_rgb(r=(0., 1.),
                                     g=(0., 1.),
                                     b=(0., 1.),
                                     n=(3, 17)):
            color = r, g, b  # 根据用户输入的 RGB 值创建颜色元组
            if as_cmap:
                colors = light_palette(color, 256, input="rgb")  # 生成基于 RGB 的亮度调色板颜色
                _update_lut(cmap, colors)  # 更新 colormap 的颜色映射表
                _show_cmap(cmap)  # 在交互式界面中展示 colormap
            else:
                pal[:] = light_palette(color, n, input="rgb")  # 生成基于 RGB 的亮度调色板颜色列表
                palplot(pal)  # 展示调色板颜色
    
    elif input == "hls":
        @interact
        def choose_light_palette_hls(h=(0., 1.),
                                     l=(0., 1.),  # noqa: E741
                                     s=(0., 1.),
                                     n=(3, 17)):
            color = h, l, s  # 根据用户输入的 HLS 值创建颜色元组
            if as_cmap:
                colors = light_palette(color, 256, input="hls")  # 生成基于 HLS 的亮度调色板颜色
                _update_lut(cmap, colors)  # 更新 colormap 的颜色映射表
                _show_cmap(cmap)  # 在交互式界面中展示 colormap
            else:
                pal[:] = light_palette(color, n, input="hls")  # 生成基于 HLS 的亮度调色板颜色列表
                palplot(pal)  # 展示调色板颜色
    
    elif input == "husl":
        @interact
        def choose_light_palette_husl(h=(0, 359),
                                      s=(0, 99),
                                      l=(0, 99),  # noqa: E741
                                      n=(3, 17)):
            color = h, s, l  # 根据用户输入的 HUSL 值创建颜色元组
            if as_cmap:
                colors = light_palette(color, 256, input="husl")  # 生成基于 HUSL 的亮度调色板颜色
                _update_lut(cmap, colors)  # 更新 colormap 的颜色映射表
                _show_cmap(cmap)  # 在交互式界面中展示 colormap
            else:
                pal[:] = light_palette(color, n, input="husl")  # 生成基于 HUSL 的亮度调色板颜色列表
                palplot(pal)  # 展示调色板颜色
    
    if as_cmap:
        return cmap  # 如果需要生成 colormap，则返回 colormap
    return pal  # 否则返回调色板颜色列表
def choose_diverging_palette(as_cmap=False):
    """Launch an interactive widget to choose a diverging color palette.

    This corresponds with the :func:`diverging_palette` function. This kind
    of palette is good for data that range between interesting low values
    and interesting high values with a meaningful midpoint. (For example,
    change scores relative to some baseline value).

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    diverging_palette : Create a diverging color palette or colormap.
    choose_colorbrewer_palette : Interactively choose palettes from the
                                 colorbrewer set, including diverging palettes.

    """
    pal = []
    # 如果需要返回一个 colormap，则初始化一个可变的 colormap 对象
    if as_cmap:
        cmap = _init_mutable_colormap()

    @interact
    def choose_diverging_palette(
        h_neg=IntSlider(min=0,
                        max=359,
                        value=220),
        h_pos=IntSlider(min=0,
                        max=359,
                        value=10),
        s=IntSlider(min=0, max=99, value=74),
        l=IntSlider(min=0, max=99, value=50),  # noqa: E741
        sep=IntSlider(min=1, max=50, value=10),
        n=(2, 16),
        center=["light", "dark"]
    ):
        if as_cmap:
            # 根据用户交互选择的参数生成一个 diverging colormap，并更新 colormap 对象
            colors = diverging_palette(h_neg, h_pos, s, l, sep, 256, center)
            _update_lut(cmap, colors)
            _show_cmap(cmap)
        else:
            # 根据用户交互选择的参数生成一个 diverging palette，并存入 pal 列表
            pal[:] = diverging_palette(h_neg, h_pos, s, l, sep, n, center)
            # 显示生成的颜色 palette
            palplot(pal)

    if as_cmap:
        return cmap
    return pal


def choose_cubehelix_palette(as_cmap=False):
    """Launch an interactive widget to create a sequential cubehelix palette.

    This corresponds with the :func:`cubehelix_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values. The cubehelix system allows the
    palette to have more hue variance across the range, which can be helpful
    for distinguishing a wider range of values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    pal = []
    # 如果需要返回一个 colormap，则初始化一个可变的 colormap 对象
    if as_cmap:
        cmap = _init_mutable_colormap()

    @interact
    # 交互式函数，用于选择 cubehelix palette 的参数
    # 定义一个函数，用于生成 Cubehelix 调色板或调色映射
    def choose_cubehelix(n_colors=IntSlider(min=2, max=16, value=9),
                         start=FloatSlider(min=0, max=3, value=0),
                         rot=FloatSlider(min=-1, max=1, value=.4),
                         gamma=FloatSlider(min=0, max=5, value=1),
                         hue=FloatSlider(min=0, max=1, value=.8),
                         light=FloatSlider(min=0, max=1, value=.85),
                         dark=FloatSlider(min=0, max=1, value=.15),
                         reverse=False):

        # 如果要生成调色映射（as_cmap 为真），则执行以下操作
        if as_cmap:
            # 使用 cubehelix_palette 函数生成一个包含 256 种颜色的调色板
            colors = cubehelix_palette(256, start, rot, gamma,
                                       hue, light, dark, reverse)
            # 更新颜色映射的查找表，使其包含颜色和不透明度
            _update_lut(cmap, np.c_[colors, np.ones(256)])
            # 显示颜色映射的效果
            _show_cmap(cmap)
        else:
            # 否则，生成包含 n_colors 种颜色的 Cubehelix 调色板，并将其存储在 pal 中
            pal[:] = cubehelix_palette(n_colors, start, rot, gamma,
                                       hue, light, dark, reverse)
            # 使用 palplot 函数展示调色板的颜色
            palplot(pal)

    # 如果要生成调色映射，则返回 cmap
    if as_cmap:
        return cmap
    # 否则返回调色板 pal
    return pal
```