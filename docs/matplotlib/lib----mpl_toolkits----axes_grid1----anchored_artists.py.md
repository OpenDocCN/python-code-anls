# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\anchored_artists.py`

```py
# 从 matplotlib 中导入 _api 和 transforms 模块
from matplotlib import _api, transforms
# 从 matplotlib.offsetbox 中导入 AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker 等类
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
                                  DrawingArea, TextArea, VPacker)
# 从 matplotlib.patches 中导入 Rectangle, Ellipse, ArrowStyle, FancyArrowPatch, PathPatch 等类
from matplotlib.patches import (Rectangle, Ellipse, ArrowStyle,
                                FancyArrowPatch, PathPatch)
# 从 matplotlib.text 中导入 TextPath 类
from matplotlib.text import TextPath

# 定义 __all__ 列表，指定模块中公开的类名
__all__ = ['AnchoredDrawingArea', 'AnchoredAuxTransformBox',
           'AnchoredEllipse', 'AnchoredSizeBar', 'AnchoredDirectionArrows']

# 定义 AnchoredDrawingArea 类，继承自 AnchoredOffsetbox 类
class AnchoredDrawingArea(AnchoredOffsetbox):
    def __init__(self, width, height, xdescent, ydescent,
                 loc, pad=0.4, borderpad=0.5, prop=None, frameon=True,
                 **kwargs):
        """
        An anchored container with a fixed size and fillable `.DrawingArea`.

        Artists added to the *drawing_area* will have their coordinates
        interpreted as pixels. Any transformations set on the artists will be
        overridden.

        Parameters
        ----------
        width, height : float
            Width and height of the container, in pixels.
        xdescent, ydescent : float
            Descent of the container in the x- and y- direction, in pixels.
        loc : str
            Location of this artist.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.4
            Padding around the child objects, in fraction of the font size.
        borderpad : float, default: 0.5
            Border padding, in fraction of the font size.
        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property used as a reference for paddings.
        frameon : bool, default: True
            If True, draw a box around this artist.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        drawing_area : `~matplotlib.offsetbox.DrawingArea`
            A container for artists to display.

        Examples
        --------
        To display blue and red circles of different sizes in the upper right
        of an Axes *ax*:

        >>> ada = AnchoredDrawingArea(20, 20, 0, 0,
        ...                           loc='upper right', frameon=False)
        >>> ada.drawing_area.add_artist(Circle((10, 10), 10, fc="b"))
        >>> ada.drawing_area.add_artist(Circle((30, 10), 5, fc="r"))
        >>> ax.add_artist(ada)
        """
        # 创建一个 DrawingArea 对象，用于容纳艺术家（artists）
        self.da = DrawingArea(width, height, xdescent, ydescent)
        self.drawing_area = self.da

        # 调用父类 AnchoredOffsetbox 的构造函数，初始化 AnchoredDrawingArea
        super().__init__(
            loc, pad=pad, borderpad=borderpad, child=self.da, prop=None,
            frameon=frameon, **kwargs
        )
    def __init__(self, transform, loc,
                 pad=0.4, borderpad=0.5, prop=None, frameon=True, **kwargs):
        """
        An anchored container with transformed coordinates.

        Artists added to the *drawing_area* are scaled according to the
        coordinates of the transformation used. The dimensions of this artist
        will scale to contain the artists added.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transData`.
            用于当前坐标系的转换对象，例如：`matplotlib.axes.Axes.transData`。
        loc : str
            Location of this artist.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
            该艺术家的位置。有效位置包括'upper left'、'upper center'、'upper right'等。
            也可以使用数值表示位置，详见`.Legend`的*loc*参数说明。
        pad : float, default: 0.4
            Padding around the child objects, in fraction of the font size.
            子对象周围的填充量，以字体大小的比例表示，默认为0.4。
        borderpad : float, default: 0.5
            Border padding, in fraction of the font size.
            边框的填充量，以字体大小的比例表示，默认为0.5。
        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property used as a reference for paddings.
            用作填充参考的字体属性。
        frameon : bool, default: True
            If True, draw a box around this artist.
            如果为True，则在该艺术家周围绘制一个框。
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.
            转发给`.AnchoredOffsetbox`的关键字参数。

        Attributes
        ----------
        drawing_area : `~matplotlib.offsetbox.AuxTransformBox`
            A container for artists to display.
            用于显示艺术家的容器。

        Examples
        --------
        To display an ellipse in the upper left, with a width of 0.1 and
        height of 0.4 in data coordinates:

        >>> box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
        >>> el = Ellipse((0, 0), width=0.1, height=0.4, angle=30)
        >>> box.drawing_area.add_artist(el)
        >>> ax.add_artist(box)
        """
        self.drawing_area = AuxTransformBox(transform)

        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self.drawing_area, prop=prop, frameon=frameon,
                         **kwargs)
# 使用装饰器标记类为已弃用，推荐在版本3.8中停止使用
@_api.deprecated("3.8")
class AnchoredEllipse(AnchoredOffsetbox):
    def __init__(self, transform, width, height, angle, loc,
                 pad=0.1, borderpad=0.1, prop=None, frameon=True, **kwargs):
        """
        Draw an anchored ellipse of a given size.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            使用的坐标系的变换对象，例如：`matplotlib.axes.Axes.transData`。
        width, height : float
            椭圆的宽度和高度，以 *transform* 坐标为单位。
        angle : float
            椭圆的旋转角度，单位为度，逆时针方向。
        loc : str
            椭圆的位置。有效的位置有：
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'。
            同样兼容数值类型的输入。详细信息参见 `.Legend` 的 *loc* 参数。
        pad : float, default: 0.1
            椭圆周围的填充空间，相对于字体大小的比例。
        borderpad : float, default: 0.1
            边框的填充空间，相对于字体大小的比例。
        frameon : bool, default: True
            如果为 True，在椭圆周围绘制一个框。
        prop : `~matplotlib.font_manager.FontProperties`, optional
            用作填充的字体属性。
        **kwargs
            传递给 `.AnchoredOffsetbox` 的关键字参数。

        Attributes
        ----------
        ellipse : `~matplotlib.patches.Ellipse`
            绘制的椭圆图形对象。
        """
        # 使用给定的变换对象创建辅助变换框对象
        self._box = AuxTransformBox(transform)
        # 创建指定大小和角度的椭圆对象
        self.ellipse = Ellipse((0, 0), width, height, angle=angle)
        # 将椭圆对象添加到辅助变换框中
        self._box.add_artist(self.ellipse)

        # 调用父类的构造函数，初始化偏移框的位置、填充、边框等属性
        super().__init__(loc, pad=pad, borderpad=borderpad, child=self._box,
                         prop=prop, frameon=frameon, **kwargs)


class AnchoredSizeBar(AnchoredOffsetbox):
class AnchoredDirectionArrows(AnchoredOffsetbox):
```