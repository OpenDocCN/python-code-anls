# `D:\src\scipysrc\seaborn\seaborn\rcmod.py`

```
"""Control plot style and scaling using the matplotlib rcParams interface."""
# 导入 functools 模块，用于高阶函数的操作
import functools
# 导入 matplotlib 库，并指定别名 mpl
import matplotlib as mpl
# 从 cycler 模块中导入 cycler 对象，用于定义颜色循环
from cycler import cycler
# 从当前包中导入 palettes 模块
from . import palettes

# 定义模块中公开的函数和变量列表
__all__ = ["set_theme", "set", "reset_defaults", "reset_orig",
           "axes_style", "set_style", "plotting_context", "set_context",
           "set_palette"]

# 定义样式相关的参数键列表
_style_keys = [

    "axes.facecolor",      # 坐标轴背景颜色
    "axes.edgecolor",      # 坐标轴边缘颜色
    "axes.grid",           # 坐标轴是否显示网格线
    "axes.axisbelow",      # 网格线是否在图像元素之后绘制
    "axes.labelcolor",     # 坐标轴标签颜色

    "figure.facecolor",    # 图形的背景颜色

    "grid.color",          # 网格线颜色
    "grid.linestyle",      # 网格线样式

    "text.color",          # 文本颜色

    "xtick.color",         # x 轴刻度颜色
    "ytick.color",         # y 轴刻度颜色
    "xtick.direction",     # x 轴刻度的方向
    "ytick.direction",     # y 轴刻度的方向
    "lines.solid_capstyle",# 线条末端风格

    "patch.edgecolor",     # 补丁边缘颜色
    "patch.force_edgecolor",# 是否强制显示补丁的边缘颜色

    "image.cmap",          # 图像的颜色映射
    "font.family",         # 字体家族
    "font.sans-serif",     # 无衬线字体

    "xtick.bottom",        # x 轴刻度是否显示在底部
    "xtick.top",           # x 轴刻度是否显示在顶部
    "ytick.left",          # y 轴刻度是否显示在左侧
    "ytick.right",         # y 轴刻度是否显示在右侧

    "axes.spines.left",    # 左边框线
    "axes.spines.bottom",  # 底边框线
    "axes.spines.right",   # 右边框线
    "axes.spines.top",     # 顶边框线

]

# 定义上下文相关的参数键列表
_context_keys = [

    "font.size",           # 字体大小
    "axes.labelsize",      # 坐标轴标签大小
    "axes.titlesize",      # 坐标轴标题大小
    "xtick.labelsize",     # x 轴刻度标签大小
    "ytick.labelsize",     # y 轴刻度标签大小
    "legend.fontsize",     # 图例字体大小
    "legend.title_fontsize", # 图例标题字体大小

    "axes.linewidth",      # 坐标轴线宽
    "grid.linewidth",      # 网格线宽度
    "lines.linewidth",     # 线条宽度
    "lines.markersize",    # 线条标记大小
    "patch.linewidth",     # 补丁边框宽度

    "xtick.major.width",   # x 轴主刻度宽度
    "ytick.major.width",   # y 轴主刻度宽度
    "xtick.minor.width",   # x 轴次刻度宽度
    "ytick.minor.width",   # y 轴次刻度宽度

    "xtick.major.size",    # x 轴主刻度长度
    "ytick.major.size",    # y 轴主刻度长度
    "xtick.minor.size",    # x 轴次刻度长度
    "ytick.minor.size",    # y 轴次刻度长度

]

# 设置主题风格的函数，控制 matplotlib 和 seaborn 绘图的默认风格
def set_theme(context="notebook", style="darkgrid", palette="deep",
              font="sans-serif", font_scale=1, color_codes=True, rc=None):
    """
    Set aspects of the visual theme for all matplotlib and seaborn plots.

    This function changes the global defaults for all plots using the
    matplotlib rcParams system. The themeing is decomposed into several distinct
    sets of parameter values.

    The options are illustrated in the :doc:`aesthetics <../tutorial/aesthetics>`
    and :doc:`color palette <../tutorial/color_palettes>` tutorials.

    Parameters
    ----------
    context : string or dict
        Scaling parameters, see :func:`plotting_context`.
    style : string or dict
        Axes style parameters, see :func:`axes_style`.
    palette : string or sequence
        Color palette, see :func:`color_palette`.
    font : string
        Font family, see matplotlib font manager.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    color_codes : bool
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.
    rc : dict or None
        Dictionary of rc parameter mappings to override the above.

    Examples
    --------

    .. include:: ../docstrings/set_theme.rst

    """
    # 设置上下文参数
    set_context(context, font_scale)
    # 设置风格参数，其中包括设置字体家族
    set_style(style, rc={"font.family": font})
    # 设置调色板参数
    set_palette(palette, color_codes=color_codes)
    # 如果 rc 不是 None，则更新 matplotlib 的全局配置参数
    if rc is not None:
        mpl.rcParams.update(rc)
def set(*args, **kwargs):
    """
    别名，调用 :func:`set_theme`，推荐使用此接口。

    此函数可能在将来被移除。
    """
    set_theme(*args, **kwargs)


def reset_defaults():
    """
    将所有的 RC 参数重置为默认设置。
    """
    mpl.rcParams.update(mpl.rcParamsDefault)


def reset_orig():
    """
    将所有的 RC 参数重置为原始设置（保留自定义 RC）。
    """
    from . import _orig_rc_params
    mpl.rcParams.update(_orig_rc_params)


def axes_style(style=None, rc=None):
    """
    获取控制绘图通用风格的参数。

    风格参数控制诸如背景颜色和是否默认启用网格等属性。这是通过
    matplotlib 的 rcParams 系统实现的。

    选项在 :doc:`美学教程 <../tutorial/aesthetics>` 中有详细说明。

    此函数还可以作为上下文管理器使用，临时改变全局默认设置。参见
    :func:`set_theme` 或 :func:`set_style` 修改所有绘图的全局默认设置。

    Parameters
    ----------
    style : None、dict 或 {darkgrid, whitegrid, dark, white, ticks} 中的一个
        参数字典或预配置风格的名称。
    rc : dict, optional
        映射参数，用于覆盖预设的 seaborn 风格字典中的值。仅更新被认为是
        风格定义的参数。

    Examples
    --------

    .. include:: ../docstrings/axes_style.rst

    """
    if style is None:
        style_dict = {k: mpl.rcParams[k] for k in _style_keys}

    elif isinstance(style, dict):
        style_dict = style
    else:
        styles = ["white", "dark", "whitegrid", "darkgrid", "ticks"]
        if style not in styles:
            raise ValueError(f"style must be one of {', '.join(styles)}")

        # Define colors here
        dark_gray = ".15"  # Dark gray color value
        light_gray = ".8"   # Light gray color value

        # Common parameters
        style_dict = {

            "figure.facecolor": "white",       # Background color of the figure
            "axes.labelcolor": dark_gray,      # Color of axis labels

            "xtick.direction": "out",          # Direction of x-axis ticks
            "ytick.direction": "out",          # Direction of y-axis ticks
            "xtick.color": dark_gray,          # Color of x-axis ticks
            "ytick.color": dark_gray,          # Color of y-axis ticks

            "axes.axisbelow": True,            # Whether to place axis below other elements
            "grid.linestyle": "-",             # Style of grid lines

            "text.color": dark_gray,           # Color of text
            "font.family": ["sans-serif"],     # Font family for text
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
                                "Bitstream Vera Sans", "sans-serif"],  # Sans-serif font options

            "lines.solid_capstyle": "round",   # Style of line caps
            "patch.edgecolor": "w",            # Edge color of patches
            "patch.force_edgecolor": True,     # Whether to force patch edge color

            "image.cmap": "rocket",            # Colormap for images

            "xtick.top": False,                # Whether to show ticks on top of the plot
            "ytick.right": False,              # Whether to show ticks on the right side of the plot

        }

        # Set grid on or off
        if "grid" in style:
            style_dict.update({
                "axes.grid": True,              # Whether to show grid on the plot
            })
        else:
            style_dict.update({
                "axes.grid": False,             # No grid on the plot
            })

        # Set the color of the background, spines, and grids based on style
        if style.startswith("dark"):
            style_dict.update({
                "axes.facecolor": "#EAEAF2",    # Background color for dark styles
                "axes.edgecolor": "white",      # Color of axes edges
                "grid.color": "white",          # Color of grid lines

                "axes.spines.left": True,       # Show left spine
                "axes.spines.bottom": True,     # Show bottom spine
                "axes.spines.right": True,      # Show right spine
                "axes.spines.top": True,        # Show top spine

            })

        elif style == "whitegrid":
            style_dict.update({
                "axes.facecolor": "white",      # Background color for whitegrid style
                "axes.edgecolor": light_gray,   # Color of axes edges
                "grid.color": light_gray,       # Color of grid lines

                "axes.spines.left": True,       # Show left spine
                "axes.spines.bottom": True,     # Show bottom spine
                "axes.spines.right": True,      # Show right spine
                "axes.spines.top": True,        # Show top spine

            })

        elif style in ["white", "ticks"]:
            style_dict.update({
                "axes.facecolor": "white",      # Background color for white and ticks styles
                "axes.edgecolor": dark_gray,    # Color of axes edges
                "grid.color": light_gray,       # Color of grid lines

                "axes.spines.left": True,       # Show left spine
                "axes.spines.bottom": True,     # Show bottom spine
                "axes.spines.right": True,      # Show right spine
                "axes.spines.top": True,        # Show top spine

            })

        # Show or hide the axes ticks based on style
        if style == "ticks":
            style_dict.update({
                "xtick.bottom": True,           # Show x-axis ticks at the bottom
                "ytick.left": True,             # Show y-axis ticks on the left
            })
        else:
            style_dict.update({
                "xtick.bottom": False,          # Hide x-axis ticks at the bottom
                "ytick.left": False,            # Hide y-axis ticks on the left
            })

    # Remove entries that are not defined in the base list of valid keys
    # 用于处理 matplotlib 版本 <=/> 2.0 的样式字典，只保留在 _style_keys 中存在的键值对
    style_dict = {k: v for k, v in style_dict.items() if k in _style_keys}

    # 如果提供了 rc 字典，则用其覆盖 style_dict 中对应的设置
    if rc is not None:
        # 从 rc 字典中筛选出与 _style_keys 相关的键值对，更新到 style_dict 中
        rc = {k: v for k, v in rc.items() if k in _style_keys}
        style_dict.update(rc)

    # 创建一个 _AxesStyle 对象，使其可以在 with 语句中使用
    style_object = _AxesStyle(style_dict)

    # 返回创建的 _AxesStyle 对象作为结果
    return style_object
# 设置绘图的整体风格参数，影响背景颜色、是否启用网格等，默认使用 matplotlib rcParams 系统
def set_style(style=None, rc=None):
    # 调用 axes_style 函数获取风格参数对象
    style_object = axes_style(style, rc)
    # 更新 matplotlib 的全局参数
    mpl.rcParams.update(style_object)


# 获取控制绘图元素缩放的参数
# 这些参数包括标签大小、线条粗细等，更多信息见 aesthetics tutorial
def plotting_context(context=None, font_scale=1, rc=None):
    # 如果 context 为 None，则返回当前默认的参数字典
    if context is None:
        context_dict = {k: mpl.rcParams[k] for k in _context_keys}

    # 如果 context 是字典类型，则直接使用该字典作为参数
    elif isinstance(context, dict):
        context_dict = context
    else:
        # 可选的上下文列表
        contexts = ["paper", "notebook", "talk", "poster"]
        # 如果给定的上下文不在可选列表中，抛出数值错误异常
        if context not in contexts:
            raise ValueError(f"context must be in {', '.join(contexts)}")

        # 设置默认参数字典
        texts_base_context = {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
        }

        base_context = {
            "axes.linewidth": 1.25,
            "grid.linewidth": 1,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "patch.linewidth": 1,

            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,

            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
        }

        # 将文本相关的默认参数更新到基础参数字典中
        base_context.update(texts_base_context)

        # 根据上下文缩放因子调整所有参数
        scaling = dict(paper=.8, notebook=1, talk=1.5, poster=2)[context]
        context_dict = {k: v * scaling for k, v in base_context.items()}

        # 独立地缩放字体大小
        font_keys = texts_base_context.keys()
        font_dict = {k: context_dict[k] * font_scale for k in font_keys}
        context_dict.update(font_dict)

    # 如果提供了自定义的rc参数，则用其覆盖设置
    if rc is not None:
        # 只保留有效的rc参数键
        rc = {k: v for k, v in rc.items() if k in _context_keys}
        context_dict.update(rc)

    # 封装成_PlottingContext对象以便在with语句中使用
    context_object = _PlottingContext(context_dict)

    return context_object
def set_context(context=None, font_scale=1, rc=None):
    """
    Set the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc.
    Calling this function modifies the global matplotlib `rcParams`. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are versions of the notebook parameters scaled by different
    values. Font elements can also be scaled independently of (but relative to)
    the other values.

    See :func:`plotting_context` to get the parameter values.

    Parameters
    ----------
    context : dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.

    Examples
    --------

    .. include:: ../docstrings/set_context.rst

    """
    # 使用给定的参数设置绘图元素的缩放参数
    context_object = plotting_context(context, font_scale, rc)
    # 更新全局的 matplotlib `rcParams` 参数
    mpl.rcParams.update(context_object)


class _RCAesthetics(dict):
    def __enter__(self):
        # 获取当前的 matplotlib `rcParams` 参数
        rc = mpl.rcParams
        # 备份原始的参数字典
        self._orig = {k: rc[k] for k in self._keys}
        # 使用新的参数字典来设置绘图样式
        self._set(self)

    def __exit__(self, exc_type, exc_value, exc_tb):
        # 恢复原始的 matplotlib `rcParams` 参数
        self._set(self._orig)

    def __call__(self, func):
        # 定义一个装饰器，用于在函数调用时临时应用新的绘图样式
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class _AxesStyle(_RCAesthetics):
    """Light wrapper on a dict to set style temporarily."""
    # 继承自 `_RCAesthetics` 的子类，用于临时设置绘图样式
    _keys = _style_keys
    _set = staticmethod(set_style)


class _PlottingContext(_RCAesthetics):
    """Light wrapper on a dict to set context temporarily."""
    # 继承自 `_RCAesthetics` 的子类，用于临时设置绘图上下文
    _keys = _context_keys
    _set = staticmethod(set_context)


def set_palette(palette, n_colors=None, desat=None, color_codes=False):
    """Set the matplotlib color cycle using a seaborn palette.

    Parameters
    ----------
    palette : seaborn color palette | matplotlib colormap | hls | husl
        Palette definition. Should be something :func:`color_palette` can process.
    n_colors : int
        Number of colors in the cycle. The default number of colors will depend
        on the format of ``palette``, see the :func:`color_palette`
        documentation for more information.
    desat : float
        Proportion to desaturate each color by.
    color_codes : bool
        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand
        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.

    See Also
    --------
    color_palette : Generate a color palette using seaborn's color utilities.

    """
    # 使用 seaborn 的调色板设置 matplotlib 的颜色循环
    pass  # This function doesn't perform any additional actions beyond its docstring
    # 使用指定的调色板创建颜色板，并设置颜色循环
    colors = palettes.color_palette(palette, n_colors, desat)
    
    # 使用颜色循环创建一个循环迭代器
    cyl = cycler('color', colors)
    
    # 将轴的属性循环设置为新创建的循环迭代器
    mpl.rcParams['axes.prop_cycle'] = cyl
    
    # 如果需要，尝试根据调色板设置颜色编码
    if color_codes:
        try:
            # 调用函数尝试设置颜色编码
            palettes.set_color_codes(palette)
        except (ValueError, TypeError):
            # 如果遇到异常则不做任何操作
            pass
```