# `D:\src\scipysrc\matplotlib\lib\matplotlib\_enums.py`

```
# 引入枚举类 Enum 和 auto 方法
from enum import Enum, auto
# 从 matplotlib 模块中引入 _docstring

from matplotlib import _docstring

# 定义一个名为 _AutoStringNameEnum 的类，继承自 Enum 类
class _AutoStringNameEnum(Enum):
    """Automate the ``name = 'name'`` part of making a (str, Enum)."""

    # 定义一个方法，用于生成枚举值的下一个值
    def _generate_next_value_(name, start, count, last_values):
        return name

    # 重写 __hash__ 方法，返回当前对象的哈希值
    def __hash__(self):
        return str(self).__hash__()

# 定义一个名为 JoinStyle 的类，继承自 str 类和 _AutoStringNameEnum 类
class JoinStyle(str, _AutoStringNameEnum):
    """
    Define how the connection between two line segments is drawn.

    For a visual impression of each *JoinStyle*, `view these docs online
    <JoinStyle>`, or run `JoinStyle.demo`.

    Lines in Matplotlib are typically defined by a 1D `~.path.Path` and a
    finite ``linewidth``, where the underlying 1D `~.path.Path` represents the
    center of the stroked line.

    By default, `~.backend_bases.GraphicsContextBase` defines the boundaries of
    a stroked line to simply be every point within some radius,
    ``linewidth/2``, away from any point of the center line. However, this
    results in corners appearing "rounded", which may not be the desired
    behavior if you are drawing, for example, a polygon or pointed star.

    **Supported values:**

    .. rst-class:: value-list

        'miter'
            the "arrow-tip" style. Each boundary of the filled-in area will
            extend in a straight line parallel to the tangent vector of the
            centerline at the point it meets the corner, until they meet in a
            sharp point.
        'round'
            stokes every point within a radius of ``linewidth/2`` of the center
            lines.
        'bevel'
            the "squared-off" style. It can be thought of as a rounded corner
            where the "circular" part of the corner has been cut off.

    .. note::

        Very long miter tips are cut off (to form a *bevel*) after a
        backend-dependent limit called the "miter limit", which specifies the
        maximum allowed ratio of miter length to line width. For example, the
        PDF backend uses the default value of 10 specified by the PDF standard,
        while the SVG backend does not even specify the miter limit, resulting
        in a default value of 4 per the SVG specification. Matplotlib does not
        currently allow the user to adjust this parameter.

        A more detailed description of the effect of a miter limit can be found
        in the `Mozilla Developer Docs
        <https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/stroke-miterlimit>`_
    """
    pass
    """
    # 示例演示不同连接风格的效果

    # 导入 matplotlib 的 JoinStyle 枚举类型
    from matplotlib._enums import JoinStyle

    # 自动分配 JoinStyle 的枚举值
    miter = auto()
    round = auto()
    bevel = auto()

    @staticmethod
    # 静态方法，演示每种 JoinStyle 在不同连接角度下的效果
    def demo():
        """Demonstrate how each JoinStyle looks for various join angles."""
        import numpy as np
        import matplotlib.pyplot as plt

        # 绘制连接角度的函数
        def plot_angle(ax, x, y, angle, style):
            phi = np.radians(angle)
            xx = [x + .5, x, x + .5*np.cos(phi)]
            yy = [y, y, y + .5*np.sin(phi)]
            # 绘制连接线，使用指定的 JoinStyle
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_joinstyle=style)
            # 绘制连接线的辅助线
            ax.plot(xx, yy, lw=1, color='black')
            # 标记连接点
            ax.plot(xx[1], yy[1], 'o', color='tab:red', markersize=3)

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.set_title('Join style')

        # 遍历不同的 JoinStyle 类型
        for x, style in enumerate(['miter', 'round', 'bevel']):
            ax.text(x, 5, style)  # 在图中标记 JoinStyle 类型
            # 遍历不同的连接角度
            for y, angle in enumerate([20, 45, 60, 90, 120]):
                # 绘制每个连接角度下的效果
                plot_angle(ax, x, y, angle, style)
                if x == 0:
                    ax.text(-1.3, y, f'{angle} degrees')  # 标记连接角度

        # 设置坐标轴范围和关闭轴显示
        ax.set_xlim(-1.5, 2.75)
        ax.set_ylim(-.5, 5.5)
        ax.set_axis_off()

        # 显示图形
        fig.show()
    ```
# 设置 JoinStyle 类的 input_description 属性，描述可接受的输入格式
JoinStyle.input_description = "{" \
        + ", ".join([f"'{js.name}'" for js in JoinStyle]) \
        + "}"

# 定义 CapStyle 类，继承自 str 和 _AutoStringNameEnum
class CapStyle(str, _AutoStringNameEnum):
    r"""
    Define how the two endpoints (caps) of an unclosed line are drawn.

    How to draw the start and end points of lines that represent a closed curve
    (i.e. that end in a `~.path.Path.CLOSEPOLY`) is controlled by the line's
    `JoinStyle`. For all other lines, how the start and end points are drawn is
    controlled by the *CapStyle*.

    For a visual impression of each *CapStyle*, `view these docs online
    <CapStyle>` or run `CapStyle.demo`.

    By default, `~.backend_bases.GraphicsContextBase` draws a stroked line as
    squared off at its endpoints.

    **Supported values:**

    .. rst-class:: value-list

        'butt'
            the line is squared off at its endpoint.
        'projecting'
            the line is squared off as in *butt*, but the filled in area
            extends beyond the endpoint a distance of ``linewidth/2``.
        'round'
            like *butt*, but a semicircular cap is added to the end of the
            line, of radius ``linewidth/2``.

    .. plot::
        :alt: Demo of possible CapStyle's

        from matplotlib._enums import CapStyle
        CapStyle.demo()

    """
    # 自动定义枚举值 'butt', 'projecting', 'round'
    butt = auto()
    projecting = auto()
    round = auto()

    @staticmethod
    def demo():
        """Demonstrate how each CapStyle looks for a thick line segment."""
        import matplotlib.pyplot as plt

        # 创建一个绘图窗口
        fig = plt.figure(figsize=(4, 1.2))
        ax = fig.add_axes([0, 0, 1, 0.8])
        ax.set_title('Cap style')

        # 遍历每种 CapStyle，展示其效果
        for x, style in enumerate(['butt', 'round', 'projecting']):
            ax.text(x+0.25, 0.85, style, ha='center')  # 在图上显示 CapStyle 名称
            xx = [x, x+0.5]
            yy = [0, 0]
            # 绘制线段，并设置线段末端风格为当前 CapStyle
            ax.plot(xx, yy, lw=12, color='tab:blue', solid_capstyle=style)
            ax.plot(xx, yy, lw=1, color='black')  # 绘制黑色边框
            ax.plot(xx, yy, 'o', color='tab:red', markersize=3)  # 绘制红色圆点

        ax.set_ylim(-.5, 1.5)  # 设置 Y 轴范围
        ax.set_axis_off()  # 关闭坐标轴显示
        fig.show()  # 显示图形

# 设置 CapStyle 类的 input_description 属性，描述可接受的输入格式
CapStyle.input_description = "{" \
        + ", ".join([f"'{cs.name}'" for cs in CapStyle]) \
        + "}"

# 更新 _docstring 的 interpd 字典，使得文档中能够动态插入 JoinStyle 和 CapStyle 的输入描述
_docstring.interpd.update({'JoinStyle': JoinStyle.input_description,
                          'CapStyle': CapStyle.input_description})
```