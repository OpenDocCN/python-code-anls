# `D:\src\scipysrc\seaborn\seaborn\palettes.py`

```
import colorsys  # 导入colorsys模块，用于颜色系统的转换
from itertools import cycle  # 导入cycle函数，用于创建循环迭代器

import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib as mpl  # 导入Matplotlib库，用于绘图

from .external import husl  # 导入外部husl模块，用于HUSL颜色空间转换

from .utils import desaturate, get_color_cycle  # 导入自定义工具函数desaturate和get_color_cycle
from .colors import xkcd_rgb, crayons  # 导入自定义颜色映射xkcd_rgb和crayons
from ._compat import get_colormap  # 导入兼容性模块中的get_colormap函数

__all__ = ["color_palette", "hls_palette", "husl_palette", "mpl_palette",
           "dark_palette", "light_palette", "diverging_palette",
           "blend_palette", "xkcd_palette", "crayon_palette",
           "cubehelix_palette", "set_color_codes"]

# 定义Seaborn调色板字典，包含不同主题的颜色组合
SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    deep6=["#4C72B0", "#55A868", "#C44E52",
           "#8172B3", "#CCB974", "#64B5CD"],
    muted=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
           "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    muted6=["#4878D0", "#6ACC64", "#D65F5F",
            "#956CB4", "#D5BB67", "#82C6E2"],
    pastel=["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
            "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"],
    pastel6=["#A1C9F4", "#8DE5A1", "#FF9F9B",
             "#D0BBFF", "#FFFEA3", "#B9F2F0"],
    bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"],
    bright6=["#023EFF", "#1AC938", "#E8000B",
             "#8B2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
          "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"],
    dark6=["#001C7F", "#12711C", "#8C0800",
           "#591E71", "#B8850A", "#006374"],
    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
    colorblind6=["#0173B2", "#029E73", "#D55E00",
                 "#CC78BC", "#ECE133", "#56B4E9"]
)

# 定义Matplotlib质量调色板大小的字典
MPL_QUAL_PALS = {
    "tab10": 10, "tab20": 20, "tab20b": 20, "tab20c": 20,
    "Set1": 9, "Set2": 8, "Set3": 12,
    "Accent": 8, "Paired": 12,
    "Pastel1": 9, "Pastel2": 8, "Dark2": 8,
}

# 从Seaborn调色板字典中获取质量调色板大小信息，更新到MPL_QUAL_PALS中
QUAL_PALETTE_SIZES = MPL_QUAL_PALS.copy()
QUAL_PALETTE_SIZES.update({k: len(v) for k, v in SEABORN_PALETTES.items()})
# 创建包含所有调色板名称的列表QUAL_PALETTES
QUAL_PALETTES = list(QUAL_PALETTE_SIZES.keys())


class _ColorPalette(list):
    """Set the color palette in a with statement, otherwise be a list."""
    def __enter__(self):
        """Open the context."""
        # 导入set_palette函数，并保存当前颜色配置到_orig_palette中
        from .rcmod import set_palette
        self._orig_palette = color_palette()
        # 设置当前调色板为self，并返回self对象
        set_palette(self)
        return self

    def __exit__(self, *args):
        """Close the context."""
        # 导入set_palette函数，并恢复原始的颜色配置
        from .rcmod import set_palette
        set_palette(self._orig_palette)

    def as_hex(self):
        """Return a color palette with hex codes instead of RGB values."""
        # 将调色板中的RGB值转换为十六进制颜色码，并返回新的_ColorPalette对象
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return _ColorPalette(hex)
    def _repr_html_(self):
        """Rich display of the color palette in an HTML frontend."""
        # 设置单个方块的大小
        s = 55
        # 获取颜色列表的长度
        n = len(self)
        # 构建 SVG 标签的起始部分，设置宽度和高度
        html = f'<svg  width="{n * s}" height="{s}">'
        # 遍历颜色列表，生成每个颜色方块的 SVG 矩形
        for i, c in enumerate(self.as_hex()):
            # 拼接每个颜色方块的 SVG 矩形代码
            html += (
                f'<rect x="{i * s}" y="0" width="{s}" height="{s}" style="fill:{c};'
                'stroke-width:2;stroke:rgb(255,255,255)"/>'
            )
        # 结束 SVG 标签
        html += '</svg>'
        # 返回生成的 HTML 字符串，用于富文本显示
        return html
# 定义一个函数，用于简化在笔记本中显示 matplotlib 颜色映射的富显示。
def _patch_colormap_display():
    # 定义一个内部函数，用于生成颜色映射的 PNG 表示。
    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        import io
        from PIL import Image
        import numpy as np
        IMAGE_SIZE = (400, 50)
        # 生成一个二维数组 X，用于表示颜色映射的色彩变化。
        X = np.tile(np.linspace(0, 1, IMAGE_SIZE[0]), (IMAGE_SIZE[1], 1))
        # 调用颜色映射对象的方法，将 X 转换为像素数据，返回字节流表示的 PNG 数据。
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        # 将像素数据转换为 PNG 格式，并保存到字节流中。
        Image.fromarray(pixels).save(png_bytes, format='png')
        return png_bytes.getvalue()

    # 定义一个内部函数，用于生成颜色映射的 HTML 表示。
    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        import base64
        # 调用生成 PNG 的函数获取 PNG 字节数据。
        png_bytes = self._repr_png_()
        # 将 PNG 字节数据转换为 Base64 编码的字符串。
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        # 返回 HTML 格式的表示，包含颜色映射的名称、标题和 Base64 编码的 PNG 数据。
        return ('<img '
                + 'alt="' + self.name + ' color map" '
                + 'title="' + self.name + '"'
                + 'src="data:image/png;base64,' + png_base64 + '">')

    # 将定义的两个内部函数分别绑定到 matplotlib.colors.Colormap 类的 _repr_png_ 和 _repr_html_ 方法上。
    mpl.colors.Colormap._repr_png_ = _repr_png_
    mpl.colors.Colormap._repr_html_ = _repr_html_


# 定义一个函数，用于返回颜色板的列表或连续的颜色映射，定义调色板的方式多种多样。
def color_palette(palette=None, n_colors=None, desat=None, as_cmap=False):
    """Return a list of colors or continuous colormap defining a palette.

    Possible ``palette`` values include:
        - Name of a seaborn palette (deep, muted, bright, pastel, dark, colorblind)
        - Name of matplotlib colormap
        - 'husl' or 'hls'
        - 'ch:<cubehelix arguments>'
        - 'light:<color>', 'dark:<color>', 'blend:<color>,<color>',
        - A sequence of colors in any format matplotlib accepts

    Calling this function with ``palette=None`` will return the current
    matplotlib color cycle.

    This function can also be used in a ``with`` statement to temporarily
    set the color cycle for a plot or set of plots.

    See the :ref:`tutorial <palette_tutorial>` for more information.

    Parameters
    ----------
    palette : None, string, or sequence, optional
        Name of palette or None to return current palette. If a sequence, input
        colors are used but possibly cycled and desaturated.
    n_colors : int, optional
        Number of colors in the palette. If ``None``, the default will depend
        on how ``palette`` is specified. Named palettes default to 6 colors,
        but grabbing the current palette or passing in a list of colors will
        not change the number of colors unless this is specified. Asking for
        more colors than exist in the palette will cause it to cycle. Ignored
        when ``as_cmap`` is True.
    desat : float, optional
        Proportion to desaturate each color by.
    as_cmap : bool
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    set_palette : Set the default color cycle for all plots.
    """
    # 如果未指定调色板，则使用默认的色彩循环
    if palette is None:
        palette = get_color_cycle()
        # 如果未指定颜色数量，则使用调色板中的颜色数量
        if n_colors is None:
            n_colors = len(palette)

    # 如果指定的调色板不是字符串类型，则直接使用该调色板
    elif not isinstance(palette, str):
        palette = palette
        # 如果未指定颜色数量，则使用调色板中的颜色数量
        if n_colors is None:
            n_colors = len(palette)
    
    # 否则，根据不同的字符串类型调色板进行处理
    else:
        # 如果未指定颜色数量，则根据预定义的调色板大小选择数量
        if n_colors is None:
            n_colors = QUAL_PALETTE_SIZES.get(palette, 6)

        # 如果调色板属于 seaborn 预定义调色板，则转换为对应的调色板
        if palette in SEABORN_PALETTES:
            palette = SEABORN_PALETTES[palette]

        # 如果调色板为 "hls"，则使用 hls 颜色空间创建调色板
        elif palette == "hls":
            palette = hls_palette(n_colors, as_cmap=as_cmap)

        # 如果调色板为 "husl"，则使用 husl 颜色空间创建调色板
        elif palette == "husl":
            palette = husl_palette(n_colors, as_cmap=as_cmap)

        # 如果调色板为 "jet"，则引发错误，不支持使用 jet 调色板
        elif palette.lower() == "jet":
            raise ValueError("No.")

        # 如果调色板以 "ch:" 开头，则解析参数创建 cubehelix 调色板
        elif palette.startswith("ch:"):
            args, kwargs = _parse_cubehelix_args(palette)
            palette = cubehelix_palette(n_colors, *args, **kwargs, as_cmap=as_cmap)

        # 如果调色板以 "light:" 开头，则创建由指定颜色生成的浅色调色板
        elif palette.startswith("light:"):
            _, color = palette.split(":")
            reverse = color.endswith("_r")
            if reverse:
                color = color[:-2]
            palette = light_palette(color, n_colors, reverse=reverse, as_cmap=as_cmap)

        # 如果调色板以 "dark:" 开头，则创建由指定颜色生成的深色调色板
        elif palette.startswith("dark:"):
            _, color = palette.split(":")
            reverse = color.endswith("_r")
            if reverse:
                color = color[:-2]
            palette = dark_palette(color, n_colors, reverse=reverse, as_cmap=as_cmap)

        # 如果调色板以 "blend:" 开头，则创建两种颜色之间的混合调色板
        elif palette.startswith("blend:"):
            _, colors = palette.split(":")
            colors = colors.split(",")
            palette = blend_palette(colors, n_colors, as_cmap=as_cmap)

        else:
            try:
                # 尝试使用 matplotlib 中的命名 colormap
                palette = mpl_palette(palette, n_colors, as_cmap=as_cmap)
            except (ValueError, KeyError):
                # 捕获异常，指出调色板名称无效
                raise ValueError(f"{palette!r} is not a valid palette name")

    # 如果指定了 desat 参数，则对调色板中的颜色进行降饱和处理
    if desat is not None:
        palette = [desaturate(c, desat) for c in palette]
    # 如果不作为颜色映射返回

        # 始终返回与请求的颜色数量相同的颜色
        pal_cycle = cycle(palette)  # 创建一个循环迭代器，用于循环使用调色板颜色
        palette = [next(pal_cycle) for _ in range(n_colors)]  # 从调色板中取出指定数量的颜色列表

        # 始终以 r, g, b 元组格式返回
        try:
            palette = map(mpl.colors.colorConverter.to_rgb, palette)  # 将调色板中的颜色转换为 RGB 格式
            palette = _ColorPalette(palette)  # 创建一个颜色调色板对象
        except ValueError:
            raise ValueError(f"Could not generate a palette for {palette}")  # 如果转换出错，则抛出异常信息

    return palette  # 返回最终的颜色调色板
def hls_palette(n_colors=6, h=.01, l=.6, s=.65, as_cmap=False):  # noqa
    """
    Return hues with constant lightness and saturation in the HLS system.

    The hues are evenly sampled along a circular path. The resulting palette will be
    appropriate for categorical or cyclical data.

    The `h`, `l`, and `s` values should be between 0 and 1.

    .. note::
        While the separation of the resulting colors will be mathematically
        constant, the HLS system does not construct a perceptually-uniform space,
        so their apparent intensity will vary.

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    h : float
        The value of the first hue.
    l : float
        The lightness value.
    s : float
        The saturation intensity.
    as_cmap : bool
        If True, return a matplotlib colormap object.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    husl_palette : Make a palette using evenly spaced hues in the HUSL system.

    Examples
    --------
    .. include:: ../docstrings/hls_palette.rst

    """
    if as_cmap:
        # 如果要求返回 matplotlib colormap 对象，则将颜色数量设为256
        n_colors = 256
    # 在0到1之间均匀分布色调值，然后移除最后一个值
    hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
    # 添加初始色调值h到每个色调
    hues += h
    # 将色调值限制在0到1之间
    hues %= 1
    # 将色调值转换为整数并减去整数部分，确保在0到1之间
    hues -= hues.astype(int)
    # 使用HLS颜色空间转换为RGB颜色，生成调色板列表
    palette = [colorsys.hls_to_rgb(h_i, l, s) for h_i in hues]
    if as_cmap:
        # 如果要求返回matplotlib colormap对象，则以"hls"命名创建ListedColormap对象
        return mpl.colors.ListedColormap(palette, "hls")
    else:
        # 否则返回自定义的_ColorPalette对象
        return _ColorPalette(palette)


def husl_palette(n_colors=6, h=.01, s=.9, l=.65, as_cmap=False):  # noqa
    """
    Return hues with constant lightness and saturation in the HUSL system.

    The hues are evenly sampled along a circular path. The resulting palette will be
    appropriate for categorical or cyclical data.

    The `h`, `l`, and `s` values should be between 0 and 1.

    This function is similar to :func:`hls_palette`, but it uses a nonlinear color
    space that is more perceptually uniform.

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    h : float
        The value of the first hue.
    l : float
        The lightness value.
    s : float
        The saturation intensity.
    as_cmap : bool
        If True, return a matplotlib colormap object.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    hls_palette : Make a palette using evenly spaced hues in the HSL system.

    Examples
    --------
    .. include:: ../docstrings/husl_palette.rst

    """
    if as_cmap:
        # 如果要求返回 matplotlib colormap 对象，则将颜色数量设为256
        n_colors = 256
    # 在0到1之间均匀分布色调值，然后移除最后一个值
    hues = np.linspace(0, 1, int(n_colors) + 1)[:-1]
    # 添加初始色调值h到每个色调
    hues += h
    # 将色调值限制在0到1之间
    hues %= 1
    # 将色调值转换为360度空间
    hues *= 359
    # 调整饱和度和亮度值的范围为0到99
    s *= 99
    l *= 99  # noqa
    # 使用HUSL颜色空间转换为RGB颜色，生成调色板列表
    palette = [_color_to_rgb((h_i, s, l), input="husl") for h_i in hues]
    if as_cmap:
        # 如果要求返回matplotlib colormap对象，则以"hsl"命名创建ListedColormap对象
        return mpl.colors.ListedColormap(palette, "hsl")
    else:
        # 否则返回自定义的_ColorPalette对象
        return _ColorPalette(palette)
# 返回一个matplotlib注册表中的调色板或颜色映射

# 对于连续的调色板，选择均匀间隔的离散样本，但不包括颜色映射的最小值和最大值，以提供更好的极端对比度
# 对于质性调色板（例如来自colorbrewer的调色板），返回确切的索引值（而不是插值），但如果调色板未定义那么多颜色，则返回少于n_colors个数的颜色

def mpl_palette(name, n_colors=6, as_cmap=False):
    """
    Return a palette or colormap from the matplotlib registry.

    For continuous palettes, evenly-spaced discrete samples are chosen while
    excluding the minimum and maximum value in the colormap to provide better
    contrast at the extremes.

    For qualitative palettes (e.g. those from colorbrewer), exact values are
    indexed (rather than interpolated), but fewer than `n_colors` can be returned
    if the palette does not define that many.

    Parameters
    ----------
    name : string
        Name of the palette. This should be a named matplotlib colormap.
    n_colors : int
        Number of discrete colors in the palette.

    Returns
    -------
    list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    Examples
    --------
    .. include:: ../docstrings/mpl_palette.rst

    """
    if name.endswith("_d"):  # 如果调色板名称以 "_d" 结尾
        sub_name = name[:-2]  # 截取除了最后两个字符以外的部分作为子名称
        if sub_name.endswith("_r"):  # 如果子名称以 "_r" 结尾
            reverse = True  # 设置反转标志为True
            sub_name = sub_name[:-2]  # 再次截取除了最后两个字符以外的部分作为子名称
        else:
            reverse = False  # 反转标志设置为False
        pal = color_palette(sub_name, 2) + ["#333333"]  # 获取子名称对应的颜色调色板，并添加一个灰色到调色板中
        if reverse:  # 如果需要反转调色板
            pal = pal[::-1]  # 反转调色板顺序
        cmap = blend_palette(pal, n_colors, as_cmap=True)  # 使用混合调色板函数创建颜色映射
    else:
        cmap = get_colormap(name)  # 否则，直接从名称获取颜色映射

    if name in MPL_QUAL_PALS:  # 如果名称存在于质性调色板字典中
        bins = np.linspace(0, 1, MPL_QUAL_PALS[name])[:n_colors]  # 生成颜色索引的数组，长度为质性调色板定义的长度和n_colors的最小值
    else:
        bins = np.linspace(0, 1, int(n_colors) + 2)[1:-1]  # 生成等间距的n_colors + 2个值的数组，并取第二到倒数第二个值作为bins
    palette = list(map(tuple, cmap(bins)[:, :3]))  # 调用颜色映射函数生成颜色列表，并将RGB值转换为元组形式

    if as_cmap:  # 如果需要返回颜色映射
        return cmap  # 返回颜色映射
    else:
        return _ColorPalette(palette)  # 否则返回颜色调色板对象


# 为颜色选择增加灵活性的私有函数
def _color_to_rgb(color, input):
    """Add some more flexibility to color choices."""
    if input == "hls":  # 如果输入是"hls"
        color = colorsys.hls_to_rgb(*color)  # 将HLS颜色空间转换为RGB
    elif input == "husl":  # 如果输入是"husl"
        color = husl.husl_to_rgb(*color)  # 将HUSL颜色空间转换为RGB
        color = tuple(np.clip(color, 0, 1))  # 将颜色值限制在0到1之间
    elif input == "xkcd":  # 如果输入是"xkcd"
        color = xkcd_rgb[color]  # 使用XKCD颜色名获取RGB值

    return mpl.colors.to_rgb(color)  # 返回颜色的RGB表示


# 创建从深色向基础颜色混合的连续调色板
def dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    """Make a sequential palette that blends from dark to ``color``.

    This kind of palette is good for data that range between relatively
    uninteresting low values and interesting high values.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_dark_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex, rgb-tuple, or html color name
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        if True, return a matplotlib colormap object
    input : {'rgb', 'hls', 'husl', 'xkcd'}, optional
        color space for the input color

    """
    # 如果 as_cmap 为 True，则返回一个 matplotlib.colors.ListedColormap 对象
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.
    
    # 指定输入颜色的颜色空间，可以是 'rgb', 'hls', 'husl', 'xkcd' 中的一种
    # 前三种选项适用于元组形式的输入颜色，最后一种选项适用于字符串形式的输入颜色
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    # 返回一个调色板，其中包含 RGB 元组或者 matplotlib.colors.ListedColormap 对象
    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    # 查看相关函数，创建一个具有明亮低值的连续调色板
    See Also
    --------
    light_palette : Create a sequential palette with bright low values.
    
    # 查看相关函数，创建一个包含两种颜色的发散调色板
    diverging_palette : Create a diverging palette with two colors.

    # 示例部分，引用外部文档中的 dark_palette.rst 文件
    Examples
    --------
    .. include:: ../docstrings/dark_palette.rst

    """
    # 将输入的颜色转换为 RGB 格式
    rgb = _color_to_rgb(color, input)
    # 将 RGB 转换为 HUSL 颜色空间中的色调（hue）、饱和度（sat）以及亮度（_）
    hue, sat, _ = husl.rgb_to_husl(*rgb)
    # 根据 HUSL 颜色空间中的色调和饱和度，设置灰度颜色的饱和度和亮度
    gray_s, gray_l = .15 * sat, 15
    # 根据色调、灰度的饱和度和亮度，将其转换为 RGB 格式的灰度颜色
    gray = _color_to_rgb((hue, gray_s, gray_l), input="husl")
    # 如果 reverse 为 True，则颜色列表为 [灰度颜色, RGB 颜色]，否则为 [RGB 颜色, 灰度颜色]
    colors = [rgb, gray] if reverse else [gray, rgb]
    # 调用 blend_palette 函数，混合颜色列表中的颜色，生成 n_colors 个颜色，并返回结果
    return blend_palette(colors, n_colors, as_cmap)
# 创建一个顺序调色板，从浅色渐变到指定的颜色。

def light_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    """Make a sequential palette that blends from light to ``color``.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using a Jupyter notebook, you can also choose this palette
    interactively with the :func:`choose_light_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex code, html color name, or tuple in `input` space.
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    diverging_palette : Create a diverging palette with two colors.

    Examples
    --------
    .. include:: ../docstrings/light_palette.rst

    """
    
    # 将输入的颜色转换为 RGB 格式
    rgb = _color_to_rgb(color, input)
    
    # 将 RGB 转换为 HUSL 色彩空间中的色调、饱和度和亮度
    hue, sat, _ = husl.rgb_to_husl(*rgb)
    
    # 计算灰色调的饱和度和亮度
    gray_s, gray_l = .15 * sat, 95
    
    # 将灰色调转换为 RGB 格式
    gray = _color_to_rgb((hue, gray_s, gray_l), input="husl")
    
    # 根据 reverse 参数确定颜色的顺序
    colors = [rgb, gray] if reverse else [gray, rgb]
    
    # 调用 blend_palette 函数，创建混合调色板
    return blend_palette(colors, n_colors, as_cmap)


def diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6,  # noqa
                      center="light", as_cmap=False):
    """Make a diverging palette between two HUSL colors.

    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_diverging_palette` function.

    Parameters
    ----------
    h_neg, h_pos : float in [0, 359]
        Anchor hues for negative and positive extents of the map.
    s : float in [0, 100], optional
        Anchor saturation for both extents of the map.
    l : float in [0, 100], optional
        Anchor lightness for both extents of the map.
    sep : int, optional
        Size of the intermediate region.
    n : int, optional
        Number of colors in the palette (if not returning a cmap)
    center : {"light", "dark"}, optional
        Whether the center of the palette is light or dark
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    dark_palette : Create a sequential palette with dark values.

    """
    # 创建一个带有浅色值的顺序调色板函数。
    
    palfunc = dict(dark=dark_palette, light=light_palette)[center]
    # 从给定的调色板字典中选择适当的调色板函数（dark_palette 或 light_palette），并赋给 palfunc。
    
    n_half = int(128 - (sep // 2))
    # 计算调色板中负半部分的颜色数量，128 是颜色空间的一半，sep 是两个颜色之间的间隔。

    neg = palfunc((h_neg, s, l), n_half, reverse=True, input="husl")
    # 使用选定的调色板函数 palfunc，生成负半部分的颜色序列，采用逆序，使用 husl 颜色空间。

    pos = palfunc((h_pos, s, l), n_half, input="husl")
    # 使用选定的调色板函数 palfunc，生成正半部分的颜色序列，使用 husl 颜色空间。

    midpoint = dict(light=[(.95, .95, .95)], dark=[(.133, .133, .133)])[center]
    # 根据给定的 center 值选择中点颜色，light 对应浅色中点，dark 对应深色中点。

    mid = midpoint * sep
    # 将中点颜色扩展到与负半部分和正半部分相同的长度。

    pal = blend_palette(np.concatenate([neg, mid, pos]), n, as_cmap=as_cmap)
    # 使用负半部分、中点和正半部分的颜色序列创建一个混合调色板，n 是最终调色板中的颜色数量，as_cmap 表示是否作为 colormap 使用。

    return pal
    # 返回生成的调色板 pal。
def cubehelix_palette(n_colors=6, start=0, rot=.4, gamma=1.0, hue=0.8,
                      light=.85, dark=.15, reverse=False, as_cmap=False):
    """Make a sequential palette from the cubehelix system.

    This produces a colormap with linearly-decreasing (or increasing)
    brightness. That means that information will be preserved if printed to
    black and white or viewed by someone who is colorblind.  "cubehelix" is
    also available as a matplotlib-based palette, but this function gives the
    user more control over the look of the palette and has a different set of
    defaults.

    In addition to using this function, it is also possible to generate a
    """
    # 调用 cubehelix_palette 函数时，根据参数生成一个 cubehelix 调色板
    pal = mpl.colors.LinearSegmentedColormap(
        # 生成调色板名为 "cubehelix"，基于指定的参数生成颜色序列
        name="cubehelix", 
        # 使用 cubehelix 系统生成颜色映射
        segmentdata=cubehelix_segment(
            start, rot, gamma, hue, light, dark, reverse)
    )
    if as_cmap:
        # 如果指定返回为 colormap 对象，则直接返回 pal
        return pal
    else:
        # 否则，生成 n_colors 数量的颜色序列
        rgb_array = pal(np.linspace(0, 1, n_colors))[:, :3]  # no alpha
        # 转换为 _ColorPalette 对象，再返回
        return _ColorPalette(map(tuple, rgb_array))


def cubehelix_segment(start=0, rot=.4, gamma=1.0, hue=0.8,
                      light=.85, dark=.15, reverse=False):
    """Generate a segment of the cubehelix color map.

    Parameters
    ----------
    start : float, optional
        The hue at the start of the helix.
    rot : float, optional
        The number of rotations through the rainbow.
    gamma : float, optional
        Gamma factor to emphasize darker (gamma < 1) or lighter (gamma > 1) colors.
    hue : float, optional
        Color hue, approximately in the range [0, 3].
    light : float, optional
        Intensity of the brightest part of the palette.
    dark : float, optional
        Intensity of the darkest part of the palette.
    reverse : bool, optional
        If True, reverse the direction of the light-to-dark transition.

    Returns
    -------
    segmentdata
        Dictionary with keys 'red', 'green', 'blue' each mapping to a list
        of (position, intensity) pairs.

    Notes
    -----
    This function returns the data in a format suitable for LinearSegmentedColormap.

    """
    # 根据参数生成 cubehelix 系统的颜色映射
    return {
        'red': _cubehelix_func(start, rot, gamma, hue, light, dark, reverse, 0),
        'green': _cubehelix_func(start, rot, gamma, hue, light, dark, reverse, 1),
        'blue': _cubehelix_func(start, rot, gamma, hue, light, dark, reverse, 2)
    }


def _cubehelix_func(start, rot, gamma, hue, light, dark, reverse, channel):
    """Internal function for generating cubehelix color maps."""
    # 计算 cubehelix 系统中的颜色映射
    # 具体数学公式来自 cubehelix 系统的定义
    a = np.cos(hue * (2.0 + rot - 7.0 * channel))
    b = np.sin(hue * (1.0 + rot - 7.0 * channel))
    return [
        (0.5 + 0.5 * (gamma * (light - dark) * (a - b) + dark)) ** gamma,
        (0.5 + 0.5 * (gamma * (light - dark) * (a + b) + dark)) ** gamma,
        0.5 ** gamma
    ]
    cubehelix palette generally in seaborn using a string starting with
    `ch:` and containing other parameters (e.g. `"ch:s=.25,r=-.5"`).

    Parameters
    ----------
    n_colors : int
        Number of colors in the palette.
    start : float, 0 <= start <= 3
        The hue value at the start of the helix.
    rot : float
        Rotations around the hue wheel over the range of the palette.
    gamma : float 0 <= gamma
        Nonlinearity to emphasize dark (gamma < 1) or light (gamma > 1) colors.
    hue : float, 0 <= hue <= 1
        Saturation of the colors.
    dark : float 0 <= dark <= 1
        Intensity of the darkest color in the palette.
    light : float 0 <= light <= 1
        Intensity of the lightest color in the palette.
    reverse : bool
        If True, the palette will go from dark to light.
    as_cmap : bool
        If True, return a :class:`matplotlib.colors.ListedColormap`.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    choose_cubehelix_palette : Launch an interactive widget to select cubehelix
                               palette parameters.
    dark_palette : Create a sequential palette with dark low values.
    light_palette : Create a sequential palette with bright low values.

    References
    ----------
    Green, D. A. (2011). "A colour scheme for the display of astronomical
    intensity images". Bulletin of the Astromical Society of India, Vol. 39,
    p. 289-295.

    Examples
    --------
    .. include:: ../docstrings/cubehelix_palette.rst

    """
    def get_color_function(p0, p1):
        # Copied from matplotlib because it lives in private module
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            xg = x ** gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = hue * xg * (1 - xg) / 2

            phi = 2 * np.pi * (start / 3 + rot * x)

            # Calculate and return the color value using cubic helix interpolation
            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    cdict = {
        "red": get_color_function(-0.14861, 1.78277),
        "green": get_color_function(-0.29227, -0.90649),
        "blue": get_color_function(1.97294, 0.0),
    }

    # Create a LinearSegmentedColormap using the defined color dictionary
    cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)

    # Generate an array of color values using the colormap and specified parameters
    x = np.linspace(light, dark, int(n_colors))
    pal = cmap(x)[:, :3].tolist()  # Convert to RGB tuples and convert to list

    # Reverse the palette if specified
    if reverse:
        pal = pal[::-1]

    # Return a ListedColormap if `as_cmap` is True, otherwise return a ColorPalette
    if as_cmap:
        x_256 = np.linspace(light, dark, 256)
        if reverse:
            x_256 = x_256[::-1]
        pal_256 = cmap(x_256)
        cmap = mpl.colors.ListedColormap(pal_256, "seaborn_cubehelix")
        return cmap
    else:
        return _ColorPalette(pal)
# 将字符串化的 cubehelix 参数转换为 args 和 kwargs
def _parse_cubehelix_args(argstr):
    # 如果参数以 "ch:" 开头，则去掉前三个字符
    if argstr.startswith("ch:"):
        argstr = argstr[3:]

    # 如果参数以 "_r" 结尾，则设置 reverse 为 True，并去掉末尾两个字符
    if argstr.endswith("_r"):
        reverse = True
        argstr = argstr[:-2]
    else:
        reverse = False

    # 如果参数字符串为空，则返回空列表和包含 reverse 的 kwargs 字典
    if not argstr:
        return [], {"reverse": reverse}

    # 将所有参数按逗号分割成列表 all_args
    all_args = argstr.split(",")

    # 从 all_args 中提取没有 "=" 的参数作为 args 列表
    args = [float(a.strip(" ")) for a in all_args if "=" not in a]

    # 从 all_args 中提取含有 "=" 的参数，分割成键值对列表 kwargs
    kwargs = [a.split("=") for a in all_args if "=" in a]
    # 将 kwargs 列表转换为字典，去除键和值两边的空格，并转换值为 float 类型
    kwargs = {k.strip(" "): float(v.strip(" ")) for k, v in kwargs}

    # 定义一个映射关系，将缩写参数映射为全名
    kwarg_map = dict(
        s="start", r="rot", g="gamma",
        h="hue", l="light", d="dark",  # noqa: E741
    )

    # 将 kwargs 中的缩写参数映射为全名参数
    kwargs = {kwarg_map.get(k, k): v for k, v in kwargs.items()}

    # 如果 reverse 为 True，则在 kwargs 中设置 reverse 为 True
    if reverse:
        kwargs["reverse"] = True

    # 返回 args 和 kwargs
    return args, kwargs


# 更改 matplotlib 颜色简称的解释方式
def set_color_codes(palette="deep"):
    """Change how matplotlib color shorthands are interpreted.

    Calling this will change how shorthand codes like "b" or "g"
    are interpreted by matplotlib in subsequent plots.

    Parameters
    ----------
    palette : {deep, muted, pastel, dark, bright, colorblind}
        Named seaborn palette to use as the source of colors.

    See Also
    --------
    set : Color codes can be set through the high-level seaborn style
          manager.
    set_palette : Color codes can also be set through the function that
                  sets the matplotlib color cycle.

    """
    # 如果 palette 参数为 "reset"，则设置默认颜色列表
    if palette == "reset":
        colors = [
            (0., 0., 1.),
            (0., .5, 0.),
            (1., 0., 0.),
            (.75, 0., .75),
            (.75, .75, 0.),
            (0., .75, .75),
            (0., 0., 0.)
        ]
    # 如果 palette 不是字符串类型或不在 SEABORN_PALETTES 中，则抛出错误
    elif not isinstance(palette, str):
        err = "set_color_codes requires a named seaborn palette"
        raise TypeError(err)
    elif palette in SEABORN_PALETTES:
        # 如果 palette 在 SEABORN_PALETTES 中，且不以 "6" 结尾，则添加 "6"
        if not palette.endswith("6"):
            palette = palette + "6"
        # 设置颜色列表为 SEABORN_PALETTES[palette] 加上一个默认颜色 (.1, .1, .1)
        colors = SEABORN_PALETTES[palette] + [(.1, .1, .1)]
    else:
        # 如果无法设置指定的 palette，则抛出 ValueError 错误
        err = f"Cannot set colors with palette '{palette}'"
        raise ValueError(err)

    # 将 colors 列表中的颜色与字符代码 "bgrmyck" 对应，并将其 RGB 值更新到 matplotlib 的颜色映射中
    for code, color in zip("bgrmyck", colors):
        rgb = mpl.colors.colorConverter.to_rgb(color)
        mpl.colors.colorConverter.colors[code] = rgb
```