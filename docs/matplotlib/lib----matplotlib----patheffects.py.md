# `D:\src\scipysrc\matplotlib\lib\matplotlib\patheffects.py`

```py
"""
Defines classes for path effects. The path effects are supported in `.Text`,
`.Line2D` and `.Patch`.

.. seealso::
   :ref:`patheffects_guide`
"""

from matplotlib.backend_bases import RendererBase  # 导入RendererBase类
from matplotlib import colors as mcolors  # 导入颜色模块中的mcolors
from matplotlib import patches as mpatches  # 导入patches模块中的mpatches
from matplotlib import transforms as mtransforms  # 导入transforms模块中的mtransforms
from matplotlib.path import Path  # 导入Path类
import numpy as np  # 导入NumPy库


class AbstractPathEffect:
    """
    A base class for path effects.

    Subclasses should override the ``draw_path`` method to add effect
    functionality.
    """

    def __init__(self, offset=(0., 0.)):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, measured in points.
        """
        self._offset = offset  # 初始化偏移量属性_offset

    def _offset_transform(self, renderer):
        """Apply the offset to the given transform."""
        return mtransforms.Affine2D().translate(
            *map(renderer.points_to_pixels, self._offset))  # 根据renderer将偏移量转换成像素并应用到仿射变换上

    def _update_gc(self, gc, new_gc_dict):
        """
        Update the given GraphicsContext with the given dict of properties.

        The keys in the dictionary are used to identify the appropriate
        ``set_`` method on the *gc*.
        """
        new_gc_dict = new_gc_dict.copy()  # 复制输入的属性字典

        dashes = new_gc_dict.pop("dashes", None)  # 弹出并获取dashes属性值
        if dashes:
            gc.set_dashes(**dashes)  # 如果dashes存在，则设置GraphicsContext的虚线样式

        for k, v in new_gc_dict.items():
            set_method = getattr(gc, 'set_' + k, None)  # 获取对应的set方法
            if not callable(set_method):
                raise AttributeError(f'Unknown property {k}')  # 如果找不到对应的set方法，则抛出异常
            set_method(v)  # 调用对应的set方法设置GraphicsContext的属性值
        return gc  # 返回更新后的GraphicsContext对象

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        """
        Derived should override this method. The arguments are the same
        as :meth:`matplotlib.backend_bases.RendererBase.draw_path`
        except the first argument is a renderer.
        """
        # Get the real renderer, not a PathEffectRenderer.
        if isinstance(renderer, PathEffectRenderer):
            renderer = renderer._renderer  # 如果renderer是PathEffectRenderer的实例，则获取其内部的真实renderer
        return renderer.draw_path(gc, tpath, affine, rgbFace)


class PathEffectRenderer(RendererBase):
    """
    Implements a Renderer which contains another renderer.

    This proxy then intercepts draw calls, calling the appropriate
    :class:`AbstractPathEffect` draw method.

    .. note::
        Not all methods have been overridden on this RendererBase subclass.
        It may be necessary to add further methods to extend the PathEffects
        capabilities further.
    """

    def __init__(self, path_effects, renderer):
        """
        Parameters
        ----------
        path_effects : iterable of :class:`AbstractPathEffect`
            The path effects which this renderer represents.
        renderer : `~matplotlib.backend_bases.RendererBase` subclass

        """
        self._path_effects = path_effects  # 初始化path_effects属性，表示此renderer代表的路径效果
        self._renderer = renderer  # 初始化_renderer属性，表示此Renderer包含的另一个renderer
    # 创建一个带有指定路径效果的新对象并返回
    def copy_with_path_effect(self, path_effects):
        return self.__class__(path_effects, self._renderer)

    # 自定义属性访问方法，如果属性名在指定列表中，则从渲染器对象中获取对应属性值；否则调用父类的同名方法
    def __getattribute__(self, name):
        if name in ['flipy', 'get_canvas_width_height', 'new_gc',
                    'points_to_pixels', '_text2path', 'height', 'width']:
            return getattr(self._renderer, name)
        else:
            return object.__getattribute__(self, name)

    # 使用指定路径效果绘制路径
    def draw_path(self, gc, tpath, affine, rgbFace=None):
        for path_effect in self._path_effects:
            path_effect.draw_path(self._renderer, gc, tpath, affine,
                                  rgbFace)

    # 使用指定路径效果绘制标记点
    def draw_markers(
            self, gc, marker_path, marker_trans, path, *args, **kwargs):
        # 当路径效果列表长度为1时，调用父类的draw_markers方法，否则递归调用当前方法
        if len(self._path_effects) == 1:
            return super().draw_markers(gc, marker_path, marker_trans, path,
                                        *args, **kwargs)

        for path_effect in self._path_effects:
            # 创建一个带有当前路径效果的新渲染器对象，然后递归调用绘制标记点方法
            renderer = self.copy_with_path_effect([path_effect])
            renderer.draw_markers(gc, marker_path, marker_trans, path,
                                  *args, **kwargs)

    # 使用指定路径效果绘制路径集合
    def draw_path_collection(self, gc, master_transform, paths, *args,
                             **kwargs):
        # 当路径效果列表长度为1时，调用父类的draw_path_collection方法，否则递归调用当前方法
        if len(self._path_effects) == 1:
            return super().draw_path_collection(gc, master_transform, paths,
                                                *args, **kwargs)

        for path_effect in self._path_effects:
            # 创建一个带有当前路径效果的新渲染器对象，然后递归调用绘制路径集合方法
            renderer = self.copy_with_path_effect([path_effect])
            renderer.draw_path_collection(gc, master_transform, paths,
                                          *args, **kwargs)

    # 在渲染器对象上打开一个新的分组
    def open_group(self, s, gid=None):
        return self._renderer.open_group(s, gid)

    # 在渲染器对象上关闭指定分组
    def close_group(self, s):
        return self._renderer.close_group(s)
class Normal(AbstractPathEffect):
    """
    The "identity" PathEffect.

    The Normal PathEffect's sole purpose is to draw the original artist with
    no special path effect.
    """


def _subclass_with_normal(effect_class):
    """
    Create a PathEffect class combining *effect_class* and a normal draw.
    """

    # 定义一个新的类，继承自effect_class，并且覆盖draw_path方法
    class withEffect(effect_class):
        def draw_path(self, renderer, gc, tpath, affine, rgbFace):
            # 调用父类的draw_path方法，即effect_class的draw_path方法
            super().draw_path(renderer, gc, tpath, affine, rgbFace)
            # 再次调用renderer的draw_path方法，以实现正常的绘制效果

    # 设置新类的名称
    withEffect.__name__ = f"with{effect_class.__name__}"
    # 设置新类的限定名称
    withEffect.__qualname__ = f"with{effect_class.__name__}"
    # 设置新类的文档字符串，用于说明其作用
    withEffect.__doc__ = f"""
    A shortcut PathEffect for applying `.{effect_class.__name__}` and then
    drawing the original Artist.

    With this class you can use ::

        artist.set_path_effects([patheffects.with{effect_class.__name__}()])

    as a shortcut for ::

        artist.set_path_effects([patheffects.{effect_class.__name__}(),
                                 patheffects.Normal()])
    """
    # 由于局部定义的子类不继承文档字符串，因此需要手动设置draw_path方法的文档字符串
    withEffect.draw_path.__doc__ = effect_class.draw_path.__doc__
    return withEffect


class Stroke(AbstractPathEffect):
    """A line based PathEffect which re-draws a stroke."""

    def __init__(self, offset=(0, 0), **kwargs):
        """
        The path will be stroked with its gc updated with the given
        keyword arguments, i.e., the keyword arguments should be valid
        gc parameter values.
        """
        # 调用父类的初始化方法，并传入偏移量参数
        super().__init__(offset)
        # 保存gc参数到实例变量_gc中
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """Draw the path with updated gc."""
        # 创建一个新的gc对象，以避免修改原始gc对象
        gc0 = renderer.new_gc()
        # 复制原始gc对象的属性到新gc对象中
        gc0.copy_properties(gc)
        # 使用保存的gc参数更新新gc对象
        gc0 = self._update_gc(gc0, self._gc)
        # 调用renderer的draw_path方法来绘制路径，应用偏移转换和rgbFace参数
        renderer.draw_path(
            gc0, tpath, affine + self._offset_transform(renderer), rgbFace)
        # 恢复gc0的状态，以确保不影响后续的绘制


withStroke = _subclass_with_normal(effect_class=Stroke)


class SimplePatchShadow(AbstractPathEffect):
    """A simple shadow via a filled patch."""
    def __init__(self, offset=(2, -2),
                 shadow_rgbFace=None, alpha=None,
                 rho=0.3, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (2, -2)
            阴影在点数中的偏移量 (x, y)。
        shadow_rgbFace : :mpltype:`color`
            阴影颜色。
        alpha : float, default: 0.3
            创建阴影补丁的透明度。
        rho : float, default: 0.3
            如果未指定 *shadow_rgbFace*，则应用于 rgbFace 颜色的比例因子。
        **kwargs
            额外的关键字参数将存储并传递给 :meth:`AbstractPathEffect._update_gc`。

        """
        super().__init__(offset)

        if shadow_rgbFace is None:
            self._shadow_rgbFace = shadow_rgbFace
        else:
            self._shadow_rgbFace = mcolors.to_rgba(shadow_rgbFace)

        if alpha is None:
            alpha = 0.3

        self._alpha = alpha
        self._rho = rho

        #: 更新图形集合的关键字字典。
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        覆盖标准的 draw_path 方法，以添加阴影偏移和阴影所需的颜色更改。
        """
        gc0 = renderer.new_gc()  # 不修改 gc，而是创建一个副本！
        gc0.copy_properties(gc)

        if self._shadow_rgbFace is None:
            r, g, b = (rgbFace or (1., 1., 1.))[:3]
            # 通过因子缩放颜色以改善阴影效果。
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_rgbFace

        gc0.set_foreground("none")
        gc0.set_alpha(self._alpha)
        gc0.set_linewidth(0)

        gc0 = self._update_gc(gc0, self._gc)
        renderer.draw_path(
            gc0, tpath, affine + self._offset_transform(renderer),
            shadow_rgbFace)
        gc0.restore()
# 通过 _subclass_with_normal 方法创建带有 SimplePatchShadow 效果的类
withSimplePatchShadow = _subclass_with_normal(effect_class=SimplePatchShadow)

# 定义 SimpleLineShadow 类，继承自 AbstractPathEffect
class SimpleLineShadow(AbstractPathEffect):
    """A simple shadow via a line."""

    def __init__(self, offset=(2, -2),
                 shadow_color='k', alpha=0.3, rho=0.3, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (2, -2)
            The (x, y) offset to apply to the path, in points.
        shadow_color : :mpltype:`color`, default: 'black'
            The shadow color.
            A value of ``None`` takes the original artist's color
            with a scale factor of *rho*.
        alpha : float, default: 0.3
            The alpha transparency of the created shadow patch.
        rho : float, default: 0.3
            A scale factor to apply to the rgbFace color if *shadow_color*
            is ``None``.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.
        """
        # 调用父类的构造方法，初始化偏移量
        super().__init__(offset)
        # 如果 shadow_color 为 None，则直接使用 None；否则将 shadow_color 转换为 RGBA 格式
        if shadow_color is None:
            self._shadow_color = shadow_color
        else:
            self._shadow_color = mcolors.to_rgba(shadow_color)
        self._alpha = alpha  # 设置阴影的透明度
        self._rho = rho  # 设置阴影颜色的缩放因子
        #: 用于更新图形集合的关键字字典
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """
        Overrides the standard draw_path to add the shadow offset and
        necessary color changes for the shadow.
        """
        gc0 = renderer.new_gc()  # 创建渲染器的新图形上下文，不修改原有的 gc，而是创建一个副本

        # 复制原始 gc 的属性到新创建的 gc0 上
        gc0.copy_properties(gc)

        if self._shadow_color is None:
            r, g, b = (gc0.get_foreground() or (1., 1., 1.))[:3]
            # 通过缩放因子改善阴影效果，对颜色进行缩放处理
            shadow_rgbFace = (r * self._rho, g * self._rho, b * self._rho)
        else:
            shadow_rgbFace = self._shadow_color

        gc0.set_foreground(shadow_rgbFace)  # 设置 gc0 的前景色为阴影颜色
        gc0.set_alpha(self._alpha)  # 设置 gc0 的透明度为阴影透明度

        gc0 = self._update_gc(gc0, self._gc)  # 更新 gc0 的属性
        renderer.draw_path(
            gc0, tpath, affine + self._offset_transform(renderer))  # 在渲染器上绘制路径
        gc0.restore()


class PathPatchEffect(AbstractPathEffect):
    """
    Draws a `.PathPatch` instance whose Path comes from the original
    PathEffect artist.
    """

    def __init__(self, offset=(0, 0), **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        **kwargs
            All keyword arguments are passed through to the
            :class:`~matplotlib.patches.PathPatch` constructor. The
            properties which cannot be overridden are "path", "clip_box"
            "transform" and "clip_path".
        """
        # 调用父类的构造方法，初始化偏移量
        super().__init__(offset=offset)
        self.patch = mpatches.PathPatch([], **kwargs)  # 创建一个 PathPatch 实例，使用传递的关键字参数
    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        # 设置路径属性为传入的路径对象
        self.patch._path = tpath
        # 设置变换属性为传入的仿射变换加上偏移变换后的结果
        self.patch.set_transform(affine + self._offset_transform(renderer))
        # 设置裁剪框属性为图形上下文中的裁剪矩形
        self.patch.set_clip_box(gc.get_clip_rectangle())
        # 获取图形上下文中的裁剪路径
        clip_path = gc.get_clip_path()
        # 如果有裁剪路径且图形对象的裁剪路径为空，则设置图形对象的裁剪路径为获取到的裁剪路径
        if clip_path and self.patch.get_clip_path() is None:
            self.patch.set_clip_path(*clip_path)
        # 使用渲染器绘制图形对象
        self.patch.draw(renderer)
# 定义一个名为 TickedStroke 的类，继承自 AbstractPathEffect
class TickedStroke(AbstractPathEffect):
    """
    A line-based PathEffect which draws a path with a ticked style.

    This line style is frequently used to represent constraints in
    optimization.  The ticks may be used to indicate that one side
    of the line is invalid or to represent a closed boundary of a
    domain (i.e. a wall or the edge of a pipe).

    The spacing, length, and angle of ticks can be controlled.

    This line style is sometimes referred to as a hatched line.

    See also the :doc:`/gallery/misc/tickedstroke_demo` example.
    """

    # 定义初始化方法，接受一些参数来配置 ticked stroke 的特性
    def __init__(self, offset=(0, 0),
                 spacing=10.0, angle=45.0, length=np.sqrt(2),
                 **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        spacing : float, default: 10.0
            The spacing between ticks in points.
        angle : float, default: 45.0
            The angle between the path and the tick in degrees.  The angle
            is measured as if you were an ant walking along the curve, with
            zero degrees pointing directly ahead, 90 to your left, -90
            to your right, and 180 behind you. To change side of the ticks,
            change sign of the angle.
        length : float, default: 1.414
            The length of the tick relative to spacing.
            Recommended length = 1.414 (sqrt(2)) when angle=45, length=1.0
            when angle=90 and length=2.0 when angle=60.
        **kwargs
            Extra keywords are stored and passed through to
            :meth:`AbstractPathEffect._update_gc`.

        Examples
        --------
        See :doc:`/gallery/misc/tickedstroke_demo`.
        """
        # 调用父类的初始化方法
        super().__init__(offset)

        # 设置各种属性来控制 ticked stroke 的特性
        self._spacing = spacing
        self._angle = angle
        self._length = length
        self._gc = kwargs

# 创建一个子类实例，这里用到了一个名为 _subclass_with_normal 的函数
withTickedStroke = _subclass_with_normal(effect_class=TickedStroke)
```