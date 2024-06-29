# `D:\src\scipysrc\matplotlib\lib\matplotlib\image.py`

```
"""
The image module supports basic image loading, rescaling and display
operations.
"""

# 导入数学、操作系统、日志记录等相关模块
import math
import os
import logging
from pathlib import Path
import warnings

# 导入numpy和PIL库的相关模块
import numpy as np
import PIL.Image
import PIL.PngImagePlugin

# 导入matplotlib相关模块
import matplotlib as mpl
from matplotlib import _api, cbook, cm
# 显式导入_image模块的名称，以便在此模块中使用
from matplotlib import _image
# 将_image模块的名称也导入到image命名空间中，方便用户使用
from matplotlib._image import *  # noqa: F401, F403
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
    Affine2D, BboxBase, Bbox, BboxTransform, BboxTransformTo,
    IdentityTransform, TransformedBbox)

_log = logging.getLogger(__name__)

# 将插值字符串映射到_image模块中对应的常量
_interpd_ = {
    'antialiased': _image.NEAREST,  # 使用最近邻或Hanning插值...
    'none': _image.NEAREST,  # 不支持时回退到最近邻
    'nearest': _image.NEAREST,
    'bilinear': _image.BILINEAR,
    'bicubic': _image.BICUBIC,
    'spline16': _image.SPLINE16,
    'spline36': _image.SPLINE36,
    'hanning': _image.HANNING,
    'hamming': _image.HAMMING,
    'hermite': _image.HERMITE,
    'kaiser': _image.KAISER,
    'quadric': _image.QUADRIC,
    'catrom': _image.CATROM,
    'gaussian': _image.GAUSSIAN,
    'bessel': _image.BESSEL,
    'mitchell': _image.MITCHELL,
    'sinc': _image.SINC,
    'lanczos': _image.LANCZOS,
    'blackman': _image.BLACKMAN,
}

interpolations_names = set(_interpd_)

def composite_images(images, renderer, magnification=1.0):
    """
    Composite a number of RGBA images into one.  The images are
    composited in the order in which they appear in the *images* list.

    Parameters
    ----------
    images : list of Images
        Each must have a `make_image` method.  For each image,
        `can_composite` should return `True`, though this is not
        enforced by this function.  Each image must have a purely
        affine transformation with no shear.

    renderer : `.RendererBase`

    magnification : float, default: 1
        The additional magnification to apply for the renderer in use.

    Returns
    -------
    image : (M, N, 4) `numpy.uint8` array
        The composited RGBA image.
    offset_x, offset_y : float
        The (left, bottom) offset where the composited image should be placed
        in the output figure.
    """
    # 如果没有传入任何images，则返回一个空的RGBA图像和偏移量0, 0
    if len(images) == 0:
        return np.empty((0, 0, 4), dtype=np.uint8), 0, 0

    parts = []  # 用于存放各个部分图像的数据、位置和透明度
    bboxes = []  # 用于存放各个部分图像的包围框
    for image in images:
        # 调用图像的make_image方法获取图像数据、位置和变换信息
        data, x, y, trans = image.make_image(renderer, magnification)
        if data is not None:
            x *= magnification  # 调整x位置以考虑放大倍数
            y *= magnification  # 调整y位置以考虑放大倍数
            parts.append((data, x, y, image._get_scalar_alpha()))  # 将数据、位置和透明度存入parts列表
            bboxes.append(
                Bbox([[x, y], [x + data.shape[1], y + data.shape[0]]]))  # 计算并存入包围框
    # 如果 parts 列表为空，返回一个空的 3D 数组和零宽度与高度
    if len(parts) == 0:
        return np.empty((0, 0, 4), dtype=np.uint8), 0, 0
    
    # 根据所有边界框的联合创建一个总边界框
    bbox = Bbox.union(bboxes)
    
    # 创建一个与总边界框大小相匹配的全零数组，用于存储输出图像数据
    output = np.zeros(
        (int(bbox.height), int(bbox.width), 4), dtype=np.uint8)
    
    # 遍历 parts 列表中的每个元素，每个元素包含数据、x、y 和 alpha 值
    for data, x, y, alpha in parts:
        # 创建一个仿射变换，将当前数据的坐标系转换到总边界框的起始点
        trans = Affine2D().translate(x - bbox.x0, y - bbox.y0)
        # 使用最近邻插值将当前数据按照仿射变换放置到输出数组中
        _image.resample(data, output, trans, _image.NEAREST,
                        resample=False, alpha=alpha)
    
    # 返回输出数组以及总边界框的起始点相对于放大倍率的 x 和 y 坐标
    return output, bbox.x0 / magnification, bbox.y0 / magnification
def _draw_list_compositing_images(
        renderer, parent, artists, suppress_composite=None):
    """
    Draw a sorted list of artists, compositing images into a single
    image where possible.

    For internal Matplotlib use only: It is here to reduce duplication
    between `Figure.draw` and `Axes.draw`, but otherwise should not be
    generally useful.
    """
    # 检查列表中是否存在图像对象
    has_images = any(isinstance(x, _ImageBase) for x in artists)

    # 根据 suppress_composite 参数或者 renderer 的默认设置来决定是否关闭图像合成
    not_composite = (suppress_composite if suppress_composite is not None
                     else renderer.option_image_nocomposite())

    # 如果不需要图像合成或者列表中没有图像对象，则直接绘制每个艺术家对象
    if not_composite or not has_images:
        for a in artists:
            a.draw(renderer)
    else:
        # 合成相邻的图像对象
        image_group = []
        mag = renderer.get_image_magnification()

        def flush_images():
            if len(image_group) == 1:
                image_group[0].draw(renderer)
            elif len(image_group) > 1:
                # 调用 composite_images 函数合成图像，并在指定的区域绘制合成后的图像数据
                data, l, b = composite_images(image_group, renderer, mag)
                if data.size != 0:
                    # 创建新的图形上下文，设置裁剪矩形和裁剪路径，绘制合成后的图像数据
                    gc = renderer.new_gc()
                    gc.set_clip_rectangle(parent.bbox)
                    gc.set_clip_path(parent.get_clip_path())
                    renderer.draw_image(gc, round(l), round(b), data)
                    gc.restore()
            del image_group[:]

        for a in artists:
            if (isinstance(a, _ImageBase) and a.can_composite() and
                    a.get_clip_on() and not a.get_clip_path()):
                # 如果当前艺术家对象是图像对象，并且可以进行合成，并且未设置裁剪路径，则加入图像组合中
                image_group.append(a)
            else:
                # 否则刷新当前已积累的图像组合，并绘制当前艺术家对象
                flush_images()
                a.draw(renderer)
        # 清空剩余的图像组合
        flush_images()


def _resample(
        image_obj, data, out_shape, transform, *, resample=None, alpha=1):
    """
    Convenience wrapper around `._image.resample` to resample *data* to
    *out_shape* (with a third dimension if *data* is RGBA) that takes care of
    allocating the output array and fetching the relevant properties from the
    Image object *image_obj*.
    """
    # AGG 只能处理小于 24 位有符号整数的坐标，如果输入数据超过这个范围则引发错误
    msg = ('Data with more than {n} cannot be accurately displayed. '
           'Downsampling to less than {n} before displaying. '
           'To remove this warning, manually downsample your data.')
    if data.shape[1] > 2**23:
        # 如果数据列数超过 2**23，则发出警告，并进行降采样处理
        warnings.warn(msg.format(n='2**23 columns'))
        step = int(np.ceil(data.shape[1] / 2**23))
        data = data[:, ::step]
        transform = Affine2D().scale(step, 1) + transform
    if data.shape[0] > 2**24:
        # 如果数据行数超过 2**24，则发出警告，并进行降采样处理
        warnings.warn(msg.format(n='2**24 rows'))
        step = int(np.ceil(data.shape[0] / 2**24))
        data = data[::step, :]
        transform = Affine2D().scale(1, step) + transform
    # 决定是否需要在数据上采样时应用反锯齿处理
    # 获取图像对象的插值方法
    interpolation = image_obj.get_interpolation()
    # 如果插值方法是 'antialiased'，则进行以下判断
    if interpolation == 'antialiased':
        # 如果横向或纵向的变化大于数据的三倍，或者等于数据的宽度或高度，或者等于两倍的数据的宽度或高度，则设置插值方法为 'nearest'，否则设置为 'hanning'
        pos = np.array([[0, 0], [data.shape[1], data.shape[0]]])
        disp = transform.transform(pos)
        dispx = np.abs(np.diff(disp[:, 0]))
        dispy = np.abs(np.diff(disp[:, 1]))
        if ((dispx > 3 * data.shape[1] or
                dispx == data.shape[1] or
                dispx == 2 * data.shape[1]) and
            (dispy > 3 * data.shape[0] or
                dispy == data.shape[0] or
                dispy == 2 * data.shape[0])):
            interpolation = 'nearest'
        else:
            interpolation = 'hanning'
    # 创建一个与输出形状相同的全零数组，数据类型与输入数据相同
    out = np.zeros(out_shape + data.shape[2:], data.dtype)  # 2D->2D, 3D->3D.
    # 如果 resample 参数为 None，则获取图像对象的重采样方法
    if resample is None:
        resample = image_obj.get_resample()
    # 调用 _image 模块的 resample 函数进行图像重采样
    _image.resample(data, out, transform,
                    _interpd_[interpolation],
                    resample,
                    alpha,
                    image_obj.get_filternorm(),
                    image_obj.get_filterrad())
    # 返回处理后的输出数组
    return out
# 将 RGB 图像转换为 RGBA 格式，以便与图像重采样的 C++ 扩展兼容
def _rgb_to_rgba(A):
    # 创建一个与 A 相同大小的全零数组，数据类型为 A 的数据类型，但增加了一个 alpha 通道
    rgba = np.zeros((A.shape[0], A.shape[1], 4), dtype=A.dtype)
    # 将 A 的 RGB 数据复制到 rgba 的前三个通道
    rgba[:, :, :3] = A
    # 根据 rgba 的数据类型判断，如果是 np.uint8 类型，则将 alpha 通道设置为 255；否则设置为 1.0
    if rgba.dtype == np.uint8:
        rgba[:, :, 3] = 255
    else:
        rgba[:, :, 3] = 1.0
    # 返回转换后的 RGBA 图像数组
    return rgba


class _ImageBase(martist.Artist, cm.ScalarMappable):
    """
    图像的基类。

    interpolation 和 cmap 默认使用其 rc 设置。

    cmap 是一个 colors.Colormap 实例。
    norm 是一个 colors.Normalize 实例，用于将亮度映射到 0-1 之间。

    extent 是数据坐标轴（左，右，底，顶），用于生成注册到数据绘图的图像绘制。
    默认情况下，以零为基础的行和列索引标记像素中心。

    其他关键字参数是 matplotlib.artist 属性。
    """
    zorder = 0

    def __init__(self, ax,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 *,
                 interpolation_stage=None,
                 **kwargs
                 ):
        # 初始化 Artist 类
        martist.Artist.__init__(self)
        # 初始化 ScalarMappable 类，传入 norm 和 cmap 参数
        cm.ScalarMappable.__init__(self, norm, cmap)
        # 如果 origin 为 None，则使用 matplotlib.rcParams 中的设置
        if origin is None:
            origin = mpl.rcParams['image.origin']
        # 检查 origin 必须在 ["upper", "lower"] 中
        _api.check_in_list(["upper", "lower"], origin=origin)
        self.origin = origin
        # 设置过滤器的标准化参数
        self.set_filternorm(filternorm)
        # 设置过滤器的半径参数
        self.set_filterrad(filterrad)
        # 设置插值方法
        self.set_interpolation(interpolation)
        # 设置插值阶段
        self.set_interpolation_stage(interpolation_stage)
        # 设置重采样标志
        self.set_resample(resample)
        # 将 ax 参数赋给对象的 axes 属性
        self.axes = ax

        # 图像缓存，初始为 None
        self._imcache = None

        # 内部更新关键字参数
        self._internal_update(kwargs)

    def __str__(self):
        try:
            # 尝试获取图像的形状信息
            shape = self.get_shape()
            return f"{type(self).__name__}(shape={shape!r})"
        except RuntimeError:
            # 如果获取形状信息时出错，返回类名
            return type(self).__name__

    def __getstate__(self):
        # 在 pickle 中保存对象状态时，不保存图像缓存以节省空间
        return {**super().__getstate__(), "_imcache": None}

    def get_size(self):
        """返回图像的大小，以元组 (numrows, numcols) 的形式。"""
        return self.get_shape()[:2]

    def get_shape(self):
        """
        返回图像的形状，以元组 (numrows, numcols, channels) 的形式。

        如果图像数组 _A 为 None，则引发 RuntimeError。
        """
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape

    def set_alpha(self, alpha):
        """
        设置用于混合的 alpha 值 - 不是所有后端都支持。

        Parameters
        ----------
        alpha : float or 2D array-like or None
            alpha 值必须是一个浮点数、二维数组或 None。
        """
        # 调用 Artist 类的 _set_alpha_for_array 方法，设置 alpha 值
        martist.Artist._set_alpha_for_array(self, alpha)
        # 如果 alpha 的维度不是 0 或 2，则引发 TypeError 异常
        if np.ndim(alpha) not in (0, 2):
            raise TypeError('alpha must be a float, two-dimensional '
                            'array, or None')
        # 将图像缓存设置为 None
        self._imcache = None
    # 获取应用于整个图形对象的标量 alpha 值

    """
    如果 alpha 值是一个矩阵，则返回 1.0，因为像素具有单独的 alpha 值
    （参见 `~._ImageBase._make_image` 获取详细信息）。
    如果 alpha 值是标量，则返回该值，以便应用于整个图形对象，因为像素没有单独的 alpha 值。
    """
    return 1.0 if self._alpha is None or np.ndim(self._alpha) > 0 \
        else self._alpha

    # 当可映射对象被修改时调用此方法，以便观察者可以更新

    # 将图像缓存置为 None，表示需要重新生成图像
    self._imcache = None
    # 调用基类的 changed 方法，通知观察者进行更新
    cm.ScalarMappable.changed(self)

    # 为渲染器创建并返回归一化、重新缩放并上色的图像数据

    """
    如果 unsampled 参数为 True，则图像将不会被缩放，但会返回适当的仿射变换。
    返回值：
    image : (M, N, 4) `numpy.uint8` 数组，RGBA 图像数据，除非 unsampled 参数为 True，否则会重新采样。
    x, y : float，图像应绘制的左上角位置，以像素空间表示。
    trans : `~matplotlib.transforms.Affine2D`，从图像空间到像素空间的仿射变换。
    """
    raise NotImplementedError('The make_image method must be overridden')

    # 检查是否最好以未采样的方式绘制图像

    # 派生类需要覆盖此方法
    return False

    # 使用 allow_rasterization 装饰器允许栅格化
    def draw(self, renderer):
        # 如果不可见，则标记为不需要更新并返回
        if not self.get_visible():
            self.stale = False
            return
        # 对于空图像，没有东西可绘制！
        if self.get_array().size == 0:
            self.stale = False
            return
        # 实际渲染图像。
        gc = renderer.new_gc()  # 创建新的图形上下文
        self._set_gc_clip(gc)   # 设置图形上下文的剪切区域
        gc.set_alpha(self._get_scalar_alpha())  # 设置图形上下文的透明度
        gc.set_url(self.get_url())   # 设置图形上下文的 URL
        gc.set_gid(self.get_gid())   # 设置图形上下文的全局唯一标识符
        if (renderer.option_scale_image()  # 渲染器支持变换参数。
                and self._check_unsampled_image()
                and self.get_transform().is_affine):
            im, l, b, trans = self.make_image(renderer, unsampled=True)
            if im is not None:
                trans = Affine2D().scale(im.shape[1], im.shape[0]) + trans
                renderer.draw_image(gc, l, b, im, trans)  # 在渲染器上绘制图像
        else:
            im, l, b, trans = self.make_image(
                renderer, renderer.get_image_magnification())
            if im is not None:
                renderer.draw_image(gc, l, b, im)   # 在渲染器上绘制图像
        gc.restore()   # 恢复图形上下文状态
        self.stale = False   # 标记为不需要更新

    def contains(self, mouseevent):
        """检测鼠标事件是否发生在图像内部。"""
        if (self._different_canvas(mouseevent)
                # 这对 figimage 不起作用。
                or not self.axes.contains(mouseevent)[0]):
            return False, {}
        # TODO: 确保这与非线性转换坐标上的 patch 和 patch collection 一致。
        # TODO: 考虑返回图像坐标（因为图像是矩形，所以不应该太困难）。
        trans = self.get_transform().inverted()   # 获取反转的坐标变换
        x, y = trans.transform([mouseevent.x, mouseevent.y])   # 转换鼠标事件的坐标
        xmin, xmax, ymin, ymax = self.get_extent()   # 获取图像的范围
        # 检查 x 是否在 xmin 和 xmax 之间，y 是否在 ymin 和 ymax 之间。
        inside = (x is not None and (x - xmin) * (x - xmax) <= 0
                  and y is not None and (y - ymin) * (y - ymax) <= 0)
        return inside, {}   # 返回是否在图像内部的布尔值和空字典

    def write_png(self, fname):
        """将图像写入 PNG 文件 *fname*。"""
        im = self.to_rgba(self._A[::-1] if self.origin == 'lower' else self._A,
                          bytes=True, norm=True)   # 转换图像数据为 RGBA 格式
        PIL.Image.fromarray(im).save(fname, format="png")   # 将图像保存为 PNG 文件

    @staticmethod
    def _normalize_image_array(A):
        """
        检查图像类输入 *A* 的有效性并将其规范化为适合 Image 子类的格式。
        """
        A = cbook.safe_masked_invalid(A, copy=True)  # 使用 cbook 中的函数处理 A，确保其安全性
        if A.dtype != np.uint8 and not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            f"converted to float")  # 如果 A 的数据类型不是 uint8 并且不能转换为 float，则抛出类型错误
        if A.ndim == 3 and A.shape[-1] == 1:
            A = A.squeeze(-1)  # 如果是 (M, N, 1)，则假定为标量并应用色彩映射
        if not (A.ndim == 2 or A.ndim == 3 and A.shape[-1] in [3, 4]):
            raise TypeError(f"Invalid shape {A.shape} for image data")  # 如果 A 的维度不符合要求，则抛出形状类型错误
        if A.ndim == 3:
            # 如果输入数据在规范化后超出有效范围，发出警告并将 A 裁剪到边界
            # - 否则，强制转换可能会隐藏异常值，使解释变得不可靠。
            high = 255 if np.issubdtype(A.dtype, np.integer) else 1
            if A.min() < 0 or high < A.max():
                _log.warning(
                    'Clipping input data to the valid range for imshow with '
                    'RGB data ([0..1] for floats or [0..255] for integers). '
                    'Got range [%s..%s].',
                    A.min(), A.max()
                )
                A = np.clip(A, 0, high)  # 将 A 裁剪到 [0, high] 范围内
            # 将不支持的整数类型强制转换为 uint8
            if A.dtype != np.uint8 and np.issubdtype(A.dtype, np.integer):
                A = A.astype(np.uint8)  # 如果 A 的类型不是 uint8 且是整数类型，则转换为 uint8
        return A  # 返回规范化后的图像数组

    def set_data(self, A):
        """
        设置图像数组。

        注意，此函数不会更新使用的规范化。

        Parameters
        ----------
        A : array-like or `PIL.Image.Image`
            图像数组或 PIL 图像对象。
        """
        if isinstance(A, PIL.Image.Image):
            A = pil_to_array(A)  # 将 PIL 图像对象转换为数组，例如应用 PNG 调色板。
        self._A = self._normalize_image_array(A)  # 使用前面定义的方法规范化图像数组 A
        self._imcache = None  # 清空图像缓存
        self.stale = True  # 标记对象状态为过时

    def set_array(self, A):
        """
        保留供向后兼容性使用 - 使用 set_data 代替。

        Parameters
        ----------
        A : array-like
            数组类输入。
        """
        # 这里也需要出现，以覆盖继承的 cm.ScalarMappable.set_array 方法，避免误调用。
        self.set_data(A)  # 调用 set_data 方法来设置数组

    def get_interpolation(self):
        """
        返回图像调整大小时使用的插值方法。

        返回值为 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16',
        'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
        'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos' 或 'none'。
        """
        return self._interpolation  # 返回对象内部的插值方法
    # 定义一个方法，用于设置图片在调整大小时所使用的插值方法

    """
    设置图片调整大小时使用的插值方法。

    如果为 None，则使用 :rc:`image.interpolation` 的设置。如果设置为 'none'，
    图片将不进行插值显示。'none' 仅在 agg、ps 和 pdf 后端中支持，并且对于其他
    后端会回退到 'nearest' 模式。

    Parameters
    ----------
    s : {'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', \
    def set_interpolation(self, s):
        """
        Set the interpolation method for the image.

        Parameters
        ----------
        s : str or None
            The interpolation method to be used. Should be one of:
            {'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
             'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos',
             'none'} or None.
        """
        # Retrieve the interpolation method from rc settings or input parameter
        s = mpl._val_or_rc(s, 'image.interpolation').lower()
        # Check if the interpolation method is valid
        _api.check_in_list(interpolations_names, interpolation=s)
        # Set the interpolation method for the image
        self._interpolation = s
        # Mark the image as stale to trigger redraw
        self.stale = True

    def get_interpolation_stage(self):
        """
        Return when interpolation happens during the transform to RGBA.

        Returns
        -------
        str
            The stage at which interpolation happens: 'data' or 'rgba'.
        """
        return self._interpolation_stage

    def set_interpolation_stage(self, s):
        """
        Set when interpolation happens during the transform to RGBA.

        Parameters
        ----------
        s : str or None
            When to apply up/downsampling interpolation in data or RGBA
            space. If None, uses :rc:`image.interpolation_stage`.
            Should be one of {'data', 'rgba'} or None.
        """
        # Retrieve the interpolation stage from rc settings or input parameter
        s = mpl._val_or_rc(s, 'image.interpolation_stage')
        # Check if the interpolation stage is valid
        _api.check_in_list(['data', 'rgba'], s=s)
        # Set the interpolation stage
        self._interpolation_stage = s
        # Mark the image as stale to trigger redraw
        self.stale = True

    def can_composite(self):
        """Return whether the image can be composited with its neighbors."""
        # Check if the image can be composited based on its transformation properties
        trans = self.get_transform()
        return (
            self._interpolation != 'none' and
            trans.is_affine and
            trans.is_separable)

    def set_resample(self, v):
        """
        Set whether image resampling is used.

        Parameters
        ----------
        v : bool or None
            If None, uses :rc:`image.resample`.
        """
        # Retrieve the resample setting from rc settings or input parameter
        v = mpl._val_or_rc(v, 'image.resample')
        # Set the resample flag
        self._resample = v
        # Mark the image as stale to trigger redraw
        self.stale = True

    def get_resample(self):
        """Return whether image resampling is used."""
        return self._resample

    def set_filternorm(self, filternorm):
        """
        Set whether the resize filter normalizes the weights.

        Parameters
        ----------
        filternorm : bool
            True if the resize filter normalizes the weights, False otherwise.
        """
        # Set the filter normalization flag
        self._filternorm = bool(filternorm)
        # Mark the image as stale to trigger redraw
        self.stale = True

    def get_filternorm(self):
        """Return whether the resize filter normalizes the weights."""
        return self._filternorm

    def set_filterrad(self, filterrad):
        """
        Set the resize filter radius only applicable to some
        interpolation schemes.

        Parameters
        ----------
        filterrad : float
            The radius of the resize filter, must be positive.
        """
        # Convert filter radius to float
        r = float(filterrad)
        # Check if filter radius is positive
        if r <= 0:
            raise ValueError("The filter radius must be a positive number")
        # Set the filter radius
        self._filterrad = r
        # Mark the image as stale to trigger redraw
        self.stale = True

    def get_filterrad(self):
        """Return the current setting of the filter radius."""
        return self._filterrad


class AxesImage(_ImageBase):
    """
    An image attached to an Axes.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The Axes the image will belong to.
    """
    # cmap参数可以是字符串或者matplotlib.colors.Colormap实例，用于将标量数据映射到颜色上
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar
        data to colors.
    
    # norm参数可以是字符串或者matplotlib.colors.Normalize实例，用于将亮度映射到0-1范围内
    norm : str or `~matplotlib.colors.Normalize`
        Maps luminance to 0-1.
    
    # interpolation参数指定插值方法，支持的值有 'none', 'antialiased', 'nearest', 'bilinear',
    # 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    # 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
    interpolation : str, default: :rc:`image.interpolation`
        Supported values are 'none', 'antialiased', 'nearest', 'bilinear',
        'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
        'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',
        'sinc', 'lanczos', 'blackman'.
    
    # interpolation_stage参数指定插值阶段，可以是 'data' 或 'rgba'。如果是 'data'，则在用户提供的数据上进行插值。
    # 如果是 'rgba'，则在应用颜色映射后进行插值（视觉插值）。
    interpolation_stage : {'data', 'rgba'}, default: 'data'
        If 'data', interpolation
        is carried out on the data provided by the user.  If 'rgba', the
        interpolation is carried out after the colormapping has been
        applied (visual interpolation).
    
    # origin参数指定坐标轴的原点位置，可以是 'upper' 或 'lower'。'upper'通常用于矩阵和图像。
    origin : {'upper', 'lower'}, default: :rc:`image.origin`
        Place the [0, 0] index of the array in the upper left or lower left
        corner of the Axes. The convention 'upper' is typically used for
        matrices and images.
    
    # extent参数是一个可选的元组，指定图像绘制时数据轴的左、右、底、顶位置。默认情况下，使用像素中心以及从零开始的行和列索引进行标记。
    extent : tuple, optional
        The data axes (left, right, bottom, top) for making image plots
        registered with data plots.  Default is to label the pixel
        centers with the zero-based row and column indices.
    
    # filternorm参数是一个布尔值，默认为True。它是反锯齿图像缩放滤波器的一个参数（参见反锯齿文档）。
    # 如果filternorm设置为True，则滤波器会对整数值进行归一化并修正舍入误差。
    # 它不对源浮点值做任何处理，仅根据1.0的规则修正整数值，这意味着任何像素权重的总和必须等于1.0。
    filternorm : bool, default: True
        A parameter for the antigrain image resize filter
        (see the antigrain documentation).
        If filternorm is set, the filter normalizes integer values and corrects
        the rounding errors. It doesn't do anything with the source floating
        point values, it corrects only integers according to the rule of 1.0
        which means that any sum of pixel weights must be equal to 1.0. So,
        the filter function must produce a graph of the proper shape.
    
    # filterrad参数是一个大于0的浮点数，默认为4。它是具有半径参数的滤波器的滤波半径，即当插值为'sinc'、'lanczos'或'blackman'时使用。
    filterrad : float > 0, default: 4
        The filter radius for filters that have a radius parameter, i.e. when
        interpolation is one of: 'sinc', 'lanczos' or 'blackman'.
    
    # resample参数是一个布尔值，默认为False。当为True时，使用全面的重采样方法。当为False时，
    # 仅在输出图像大于输入图像时进行重采样。
    resample : bool, default: False
        When True, use a full resampling method. When False, only resample when
        the output image is larger than the input image.
    
    # **kwargs参数可以传递给`~matplotlib.artist.Artist`的属性
    **kwargs : `~matplotlib.artist.Artist` properties
    # 初始化函数，设置图像的各种属性
    def __init__(self, ax,
                 *,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 extent=None,
                 filternorm=True,
                 filterrad=4.0,
                 resample=False,
                 interpolation_stage=None,
                 **kwargs
                 ):
        # 设置图像的范围
        self._extent = extent

        # 调用父类的初始化函数，设置图像的其他属性
        super().__init__(
            ax,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
            origin=origin,
            filternorm=filternorm,
            filterrad=filterrad,
            resample=resample,
            interpolation_stage=interpolation_stage,
            **kwargs
        )

    # 获取图像的窗口范围
    def get_window_extent(self, renderer=None):
        # 获取图像的范围坐标
        x0, x1, y0, y1 = self._extent
        # 创建一个边界框对象
        bbox = Bbox.from_extents([x0, y0, x1, y1])
        # 返回经过变换后的边界框
        return bbox.transformed(self.get_transform())

    # 创建图像
    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # 获取图像的变换
        trans = self.get_transform()
        # 获取图像的范围坐标
        x1, x2, y1, y2 = self.get_extent()
        # 创建一个边界框对象
        bbox = Bbox(np.array([[x1, y1], [x2, y2]]))
        # 对边界框进行变换
        transformed_bbox = TransformedBbox(bbox, trans)
        # 设置裁剪范围
        clip = ((self.get_clip_box() or self.axes.bbox) if self.get_clip_on()
                else self.figure.bbox)
        # 返回创建的图像
        return self._make_image(self._A, bbox, transformed_bbox, clip,
                                magnification, unsampled=unsampled)

    # 检查是否应该以未采样的方式绘制图像
    def _check_unsampled_image(self):
        return self.get_interpolation() == "none"
    def set_extent(self, extent, **kwargs):
        """
        Set the image extent.

        Parameters
        ----------
        extent : 4-tuple of float
            The position and size of the image as tuple
            ``(left, right, bottom, top)`` in data coordinates.
        **kwargs
            Other parameters from which unit info (i.e., the *xunits*,
            *yunits*, *zunits* (for 3D Axes), *runits* and *thetaunits* (for
            polar Axes) entries are applied, if present.

        Notes
        -----
        This updates ``ax.dataLim``, and, if autoscaling, sets ``ax.viewLim``
        to tightly fit the image, regardless of ``dataLim``.  Autoscaling
        state is not changed, so following this with ``ax.autoscale_view()``
        will redo the autoscaling in accord with ``dataLim``.
        """
        # 解析 extent 参数，获取左、右、下、上四个坐标值
        (xmin, xmax), (ymin, ymax) = self.axes._process_unit_info(
            [("x", [extent[0], extent[1]]),
             ("y", [extent[2], extent[3]])],
            kwargs)
        # 检查是否还有未处理的关键字参数，如果有则引发异常
        if kwargs:
            raise _api.kwarg_error("set_extent", kwargs)
        # 根据单位信息验证并转换 x 和 y 轴的限制
        xmin = self.axes._validate_converted_limits(
            xmin, self.convert_xunits)
        xmax = self.axes._validate_converted_limits(
            xmax, self.convert_xunits)
        ymin = self.axes._validate_converted_limits(
            ymin, self.convert_yunits)
        ymax = self.axes._validate_converted_limits(
            ymax, self.convert_yunits)
        # 更新 extent，确保在有效范围内
        extent = [xmin, xmax, ymin, ymax]

        # 将更新后的 extent 存储到对象属性中
        self._extent = extent
        # 根据更新后的坐标极限更新数据限制
        corners = (xmin, ymin), (xmax, ymax)
        self.axes.update_datalim(corners)
        # 更新 x 和 y 轴的粘性边界
        self.sticky_edges.x[:] = [xmin, xmax]
        self.sticky_edges.y[:] = [ymin, ymax]
        # 如果 x 轴开启了自动缩放，设置 x 轴的限制
        if self.axes.get_autoscalex_on():
            self.axes.set_xlim((xmin, xmax), auto=None)
        # 如果 y 轴开启了自动缩放，设置 y 轴的限制
        if self.axes.get_autoscaley_on():
            self.axes.set_ylim((ymin, ymax), auto=None)
        # 将对象标记为过时的
        self.stale = True

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
        # 如果已经设置了 extent 属性，则直接返回
        if self._extent is not None:
            return self._extent
        else:
            # 否则根据图像大小和原点位置计算默认的 extent
            sz = self.get_size()
            numrows, numcols = sz
            if self.origin == 'upper':
                return (-0.5, numcols-0.5, numrows-0.5, -0.5)
            else:
                return (-0.5, numcols-0.5, -0.5, numrows-0.5)
    # 获取事件发生位置处的图像值，如果事件发生在图像外部则返回 *None*
    def get_cursor_data(self, event):
        """
        Return the image value at the event position or *None* if the event is
        outside the image.

        See Also
        --------
        matplotlib.artist.Artist.get_cursor_data
        """
        # 获取图像在 x 和 y 方向的范围
        xmin, xmax, ymin, ymax = self.get_extent()
        # 如果坐标原点在上方，则调换 y 方向的最小值和最大值
        if self.origin == 'upper':
            ymin, ymax = ymax, ymin
        # 获取图像的数组数据
        arr = self.get_array()
        # 定义数据范围和数组范围的边界框
        data_extent = Bbox([[xmin, ymin], [xmax, ymax]])
        array_extent = Bbox([[0, 0], [arr.shape[1], arr.shape[0]]])
        # 获取坐标转换对象并进行反转
        trans = self.get_transform().inverted()
        # 将数据范围映射到数组范围
        trans += BboxTransform(boxin=data_extent, boxout=array_extent)
        # 将事件位置转换为数组索引
        point = trans.transform([event.x, event.y])
        # 如果转换后的点包含 NaN，则返回 None
        if any(np.isnan(point)):
            return None
        # 将点坐标转换为整数索引
        j, i = point.astype(int)
        # 如果索引超出数组边界则返回 None，否则返回数组中对应位置的值
        if not (0 <= i < arr.shape[0]) or not (0 <= j < arr.shape[1]):
            return None
        else:
            return arr[i, j]
class NonUniformImage(AxesImage):
    """
    Custom subclass of AxesImage for non-uniformly spaced data.

    This class inherits from AxesImage and provides additional methods and
    overrides for handling non-uniformly spaced image data.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The Axes the image will belong to.
    interpolation : {'nearest', 'bilinear'}, default: 'nearest'
        The interpolation scheme used in the resampling.
    **kwargs
        All other keyword arguments are identical to those of `.AxesImage`.
    """

    def __init__(self, ax, *, interpolation='nearest', **kwargs):
        super().__init__(ax, **kwargs)
        # 设置插值方法
        self.set_interpolation(interpolation)

    def _check_unsampled_image(self):
        """Return False. Do not use unsampled image."""
        return False

    def set_data(self, x, y, A):
        """
        Set the grid for the pixel centers, and the pixel values.

        Parameters
        ----------
        x, y : 1D array-like
            Monotonic arrays of shapes (N,) and (M,), respectively, specifying
            pixel centers.
        A : array-like
            (M, N) `~numpy.ndarray` or masked array of values to be
            colormapped, or (M, N, 3) RGB array, or (M, N, 4) RGBA array.
        """
        A = self._normalize_image_array(A)
        x = np.array(x, np.float32)
        y = np.array(y, np.float32)
        if not (x.ndim == y.ndim == 1 and A.shape[:2] == y.shape + x.shape):
            raise TypeError("Axes don't match array shape")
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None
        self.stale = True

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def set_interpolation(self, s):
        """
        Set the interpolation method for the image.

        Parameters
        ----------
        s : {'nearest', 'bilinear'} or None
            If None, use :rc:`image.interpolation`.
        """
        if s is not None and s not in ('nearest', 'bilinear'):
            raise NotImplementedError('Only nearest neighbor and '
                                      'bilinear interpolations are supported')
        super().set_interpolation(s)

    def get_extent(self):
        """
        Get the extent of the image.

        Returns
        -------
        tuple
            Extent of the image (xmin, xmax, ymin, ymax).

        Raises
        ------
        RuntimeError
            If data has not been set.
        """
        if self._A is None:
            raise RuntimeError('Must set data first')
        return self._Ax[0], self._Ax[-1], self._Ay[0], self._Ay[-1]

    @_api.rename_parameter("3.8", "s", "filternorm")
    def set_filternorm(self, filternorm):
        """
        Set filter normalization.

        Parameters
        ----------
        filternorm : bool
            Whether to normalize filter kernel by its area.
        """
        pass

    @_api.rename_parameter("3.8", "s", "filterrad")
    def set_filterrad(self, filterrad):
        """
        Set filter radius.

        Parameters
        ----------
        filterrad : float
            Radius of the filter kernel.
        """
        pass

    def set_norm(self, norm):
        """
        Set the normalization for the image.

        Parameters
        ----------
        norm : `~matplotlib.colors.Normalize`
            Normalization object.
        
        Raises
        ------
        RuntimeError
            If data has already been set.
        """
        if self._A is not None:
            raise RuntimeError('Cannot change colors after loading data')
        super().set_norm(norm)

    def set_cmap(self, cmap):
        """
        Set the colormap for the image.

        Parameters
        ----------
        cmap : `~matplotlib.colors.Colormap`
            Colormap object.
        
        Raises
        ------
        RuntimeError
            If data has already been set.
        """
        if self._A is not None:
            raise RuntimeError('Cannot change colors after loading data')
        super().set_cmap(cmap)
    # 从事件对象中获取鼠标位置的数据，返回对应位置的数据点
    def get_cursor_data(self, event):
        # 获取鼠标的 x 和 y 坐标数据
        x, y = event.xdata, event.ydata
        # 检查鼠标位置是否在数据范围内，如果不在则返回 None
        if (x < self._Ax[0] or x > self._Ax[-1] or
                y < self._Ay[0] or y > self._Ay[-1]):
            return None
        # 在 x 轴数组中查找小于 x 的最大值的索引 j
        j = np.searchsorted(self._Ax, x) - 1
        # 在 y 轴数组中查找小于 y 的最大值的索引 i
        i = np.searchsorted(self._Ay, y) - 1
        # 返回坐标 (i, j) 处的数据点
        return self._A[i, j]
class PcolorImage(AxesImage):
    """
    Make a pcolor-style plot with an irregular rectangular grid.

    This uses a variation of the original irregular image code,
    and it is used by pcolorfast for the corresponding grid type.
    """

    def __init__(self, ax,
                 x=None,
                 y=None,
                 A=None,
                 *,
                 cmap=None,
                 norm=None,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            图像所属的 Axes 对象。
        x, y : 1D array-like, optional
            长度为 N+1 和 M+1 的单调数组，分别指定矩形边界。如果未提供，则默认为
            ``range(N + 1)`` 和 ``range(M + 1)``。
        A : array-like
            待着色的数据。其解释取决于其形状：

            - (M, N) `~numpy.ndarray` 或者掩码数组：要进行颜色映射的值
            - (M, N, 3)：RGB 数组
            - (M, N, 4)：RGBA 数组

        cmap : str or `~matplotlib.colors.Colormap`, 默认为 :rc:`image.cmap`
            用于将标量数据映射到颜色的 Colormap 实例或注册的颜色映射名称。
        norm : str or `~matplotlib.colors.Normalize`
            将亮度映射到 0-1 范围内。
        **kwargs : `~matplotlib.artist.Artist` 属性
        """
        # 调用父类的构造方法初始化基类
        super().__init__(ax, norm=norm, cmap=cmap)
        # 内部更新方法，应用传入的关键字参数
        self._internal_update(kwargs)
        # 如果提供了数据 A，则设置数据
        if A is not None:
            self.set_data(x, y, A)
    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        # 如果图像数组为空，则抛出运行时错误
        if self._A is None:
            raise RuntimeError('You must first set the image array')
        # PColorImage 不支持 unsampled 参数，抛出值错误
        if unsampled:
            raise ValueError('unsampled not supported on PColorImage')

        # 如果缓存为空，将图像数组转换为 RGBA 格式并进行填充
        if self._imcache is None:
            A = self.to_rgba(self._A, bytes=True)
            self._imcache = np.pad(A, [(1, 1), (1, 1), (0, 0)], "constant")
        padded_A = self._imcache
        # 获取背景颜色并转换为 RGBA 数组
        bg = mcolors.to_rgba(self.axes.patch.get_facecolor(), 0)
        bg = (np.array(bg) * 255).astype(np.uint8)
        # 如果填充后的图像的左上角像素不等于背景色，则将四周边界像素设为背景色
        if (padded_A[0, 0] != bg).all():
            padded_A[[0, -1], :] = padded_A[:, [0, -1]] = bg

        # 获取坐标轴的边界框坐标，并计算宽度和高度
        l, b, r, t = self.axes.bbox.extents
        width = (round(r) + 0.5) - (round(l) - 0.5)
        height = (round(t) + 0.5) - (round(b) - 0.5)
        width = round(width * magnification)
        height = round(height * magnification)
        vl = self.axes.viewLim

        # 根据视图限制生成 x 和 y 像素数组
        x_pix = np.linspace(vl.x0, vl.x1, width)
        y_pix = np.linspace(vl.y0, vl.y1, height)
        # 在填充后的图像中根据 x_pix 和 y_pix 找到对应的整数索引
        x_int = self._Ax.searchsorted(x_pix)
        y_int = self._Ay.searchsorted(y_pix)
        # 根据索引从填充后的图像中提取图像数据，形成 RGBA 图像数组
        im = (
            padded_A.view(np.uint32).ravel()[
                np.add.outer(y_int * padded_A.shape[1], x_int)]
            .view(np.uint8).reshape((height, width, 4)))
        # 返回生成的图像、左边界、底边界和恒等变换对象
        return im, l, b, IdentityTransform()

    def _check_unsampled_image(self):
        # 始终返回 False，表示不支持未采样图像
        return False

    def set_data(self, x, y, A):
        """
        设置矩形边界的网格和数据值。

        Parameters
        ----------
        x, y : 1D array-like, optional
            长度为 N+1 和 M+1 的单调数组，分别指定矩形边界。如果未给出，则默认为 ``range(N + 1)`` 和 ``range(M + 1)``
        A : array-like
            要着色的数据。其解释取决于形状：

            - (M, N) `~numpy.ndarray` 或掩码数组：要进行颜色映射的值
            - (M, N, 3)：RGB 数组
            - (M, N, 4)：RGBA 数组
        """
        # 规范化图像数组 A
        A = self._normalize_image_array(A)
        # 如果未提供 x 和 y，则创建默认的范围数组
        x = np.arange(0., A.shape[1] + 1) if x is None else np.array(x, float).ravel()
        y = np.arange(0., A.shape[0] + 1) if y is None else np.array(y, float).ravel()
        # 如果 A 的形状与 y 和 x 的边界不匹配，则引发值错误
        if A.shape[:2] != (y.size - 1, x.size - 1):
            raise ValueError(
                "Axes don't match array shape. Got %s, expected %s." %
                (A.shape[:2], (y.size - 1, x.size - 1)))
        # 为了有效的光标读取，确保 x 和 y 是递增的
        if x[-1] < x[0]:
            x = x[::-1]
            A = A[:, ::-1]
        if y[-1] < y[0]:
            y = y[::-1]
            A = A[::-1]
        # 设置内部变量以存储数据
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None
        self.stale = True
    # 定义一个方法，用于设置数组，但是具体的实现未完成，所以抛出未实现错误
    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    # 定义一个方法，用于获取鼠标事件对应的数据
    # 从事件中获取鼠标的 x 和 y 坐标
    x, y = event.xdata, event.ydata
    # 检查鼠标位置是否在预设的数据范围内，如果不在则返回 None
    if (x < self._Ax[0] or x > self._Ax[-1] or
            y < self._Ay[0] or y > self._Ay[-1]):
        return None
    # 在数组 _Ax 中进行二分查找，找到 x 坐标所在位置的索引 j
    j = np.searchsorted(self._Ax, x) - 1
    # 在数组 _Ay 中进行二分查找，找到 y 坐标所在位置的索引 i
    i = np.searchsorted(self._Ay, y) - 1
    # 返回数组 _A 中对应索引 (i, j) 处的数据
    return self._A[i, j]
class FigureImage(_ImageBase):
    """An image attached to a figure."""

    zorder = 0  # 设置图像的层次顺序，默认为0

    _interpolation = 'nearest'  # 图像插值方法，默认为'nearest'

    def __init__(self, fig,
                 *,
                 cmap=None,  # 色彩映射对象，如果为None则不使用
                 norm=None,  # 归一化对象，用于将亮度映射到0-1之间
                 offsetx=0,  # 图像在x轴上的偏移量
                 offsety=0,  # 图像在y轴上的偏移量
                 origin=None,  # 原点位置设置
                 **kwargs
                 ):
        """
        cmap 是一个 colors.Colormap 的实例
        norm 是一个 colors.Normalize 的实例，用于将亮度映射到0-1之间

        kwargs 是一个可选的 Artist 关键字参数列表
        """
        super().__init__(
            None,
            norm=norm,
            cmap=cmap,
            origin=origin
        )
        self.figure = fig  # 关联的图形对象
        self.ox = offsetx  # 设置x轴偏移量
        self.oy = offsety  # 设置y轴偏移量
        self._internal_update(kwargs)  # 更新内部关键字参数
        self.magnification = 1.0  # 放大倍数，默认为1.0

    def get_extent(self):
        """Return the image extent as tuple (left, right, bottom, top)."""
        numrows, numcols = self.get_size()  # 获取图像的行数和列数
        return (-0.5 + self.ox, numcols-0.5 + self.ox,
                -0.5 + self.oy, numrows-0.5 + self.oy)  # 返回图像的范围

    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        fac = renderer.dpi/self.figure.dpi  # 计算缩放因子，考虑到不同分辨率的后端渲染
        # fac 用于 pdf、eps、svg 等后端，其中 figure.dpi 设置为 72。
        # 这意味着我们需要按比例缩放图像（使用放大倍数），并适当地偏移它。
        bbox = Bbox([[self.ox/fac, self.oy/fac],
                     [(self.ox/fac + self._A.shape[1]),
                     (self.oy/fac + self._A.shape[0])]])  # 计算图像的边界框
        width, height = self.figure.get_size_inches()  # 获取图形的尺寸（英寸）
        width *= renderer.dpi  # 转换为像素单位
        height *= renderer.dpi  # 转换为像素单位
        clip = Bbox([[0, 0], [width, height]])  # 创建剪辑框
        return self._make_image(
            self._A, bbox, bbox, clip, magnification=magnification / fac,
            unsampled=unsampled, round_to_pixel_border=False)  # 制作图像并返回

    def set_data(self, A):
        """Set the image array."""
        cm.ScalarMappable.set_array(self, A)  # 设置图像的数据数组
        self.stale = True  # 标记图像为过时（需要更新）


class BboxImage(_ImageBase):
    """The Image class whose size is determined by the given bbox."""

    def __init__(self, bbox,
                 *,
                 cmap=None,  # 色彩映射对象，如果为None则不使用
                 norm=None,  # 归一化对象，用于将亮度映射到0-1之间
                 interpolation=None,  # 图像插值方法
                 origin=None,  # 原点位置设置
                 filternorm=True,  # 控制图像过滤器的规范化
                 filterrad=4.0,  # 控制图像过滤器的半径
                 resample=False,  # 控制是否重采样图像
                 **kwargs
                 ):
        """
        cmap 是一个 colors.Colormap 的实例
        norm 是一个 colors.Normalize 的实例，用于将亮度映射到0-1之间

        kwargs 是一个可选的 Artist 关键字参数列表
        """
        super().__init__(
            None,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
            origin=origin,
            filternorm=filternorm,
            filterrad=filterrad,
            resample=resample,
            **kwargs
        )
        self.bbox = bbox  # 设置图像的边界框
    # 返回当前对象的窗口范围
    def get_window_extent(self, renderer=None):
        # 如果没有指定渲染器，则获取当前图形的渲染器
        if renderer is None:
            renderer = self.get_figure()._get_renderer()

        # 如果 bbox 是 BboxBase 类型，则直接返回
        if isinstance(self.bbox, BboxBase):
            return self.bbox
        # 如果 bbox 是可调用对象，则使用渲染器计算并返回 bbox
        elif callable(self.bbox):
            return self.bbox(renderer)
        else:
            # 如果无法识别 bbox 的类型，则引发 ValueError 异常
            raise ValueError("Unknown type of bbox")

    # 检查鼠标事件是否发生在图像内部
    def contains(self, mouseevent):
        """Test whether the mouse event occurred within the image."""
        # 如果鼠标事件发生在不同的画布上或者对象不可见，则返回 False
        if self._different_canvas(mouseevent) or not self.get_visible():
            return False, {}
        x, y = mouseevent.x, mouseevent.y
        # 检查鼠标事件的坐标是否在窗口范围内
        inside = self.get_window_extent().contains(x, y)
        return inside, {}

    # 使用指定的渲染器创建图像
    def make_image(self, renderer, magnification=1.0, unsampled=False):
        # docstring inherited
        # 获取画布的宽度和高度
        width, height = renderer.get_canvas_width_height()
        # 获取对象在窗口中的范围，并冻结其坐标
        bbox_in = self.get_window_extent(renderer).frozen()
        bbox_in._points /= [width, height]  # 将坐标标准化为比例
        # 获取对象在窗口中的范围
        bbox_out = self.get_window_extent(renderer)
        # 创建裁剪框
        clip = Bbox([[0, 0], [width, height]])
        # 将对象的变换设置为相对于裁剪框的变换
        self._transform = BboxTransformTo(clip)
        # 返回根据参数创建的图像
        return self._make_image(
            self._A,
            bbox_in, bbox_out, clip, magnification, unsampled=unsampled)
# 从文件中读取图像数据并转换为数组形式

def imread(fname, format=None):
    """
    Read an image from a file into an array.

    .. note::

        This function exists for historical reasons.  It is recommended to
        use `PIL.Image.open` instead for loading images.

    Parameters
    ----------
    fname : str or file-like
        The image file to read: a filename, a URL or a file-like object opened
        in read-binary mode.

        Passing a URL is deprecated.  Please open the URL
        for reading and pass the result to Pillow, e.g. with
        ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.
    format : str, optional
        The image file format assumed for reading the data.  The image is
        loaded as a PNG file if *format* is set to "png", if *fname* is a path
        or opened file with a ".png" extension, or if it is a URL.  In all
        other cases, *format* is ignored and the format is auto-detected by
        `PIL.Image.open`.

    Returns
    -------
    `numpy.array`
        The image data. The returned array has shape

        - (M, N) for grayscale images.
        - (M, N, 3) for RGB images.
        - (M, N, 4) for RGBA images.

        PNG images are returned as float arrays (0-1).  All other formats are
        returned as int arrays, with a bit depth determined by the file's
        contents.
    """

    # 隐藏导入以加快在链接速度较慢的系统上的初始导入
    from urllib import parse

    # 如果未指定格式，则根据文件名或文件对象推断格式
    if format is None:
        if isinstance(fname, str):
            # 解析 URL 或获取文件扩展名推断格式
            parsed = parse.urlparse(fname)
            if len(parsed.scheme) > 1:  # 如果字符串是 URL
                ext = 'png'  # 假定为 PNG 格式
            else:
                ext = Path(fname).suffix.lower()[1:]  # 获取文件扩展名
        elif hasattr(fname, 'geturl'):  # 如果是 urlopen() 返回的对象
            ext = 'png'  # 假定为 PNG 格式
        elif hasattr(fname, 'name'):  # 如果有 name 属性
            ext = Path(fname.name).suffix.lower()[1:]  # 获取文件扩展名
        else:
            ext = 'png'  # 默认假定为 PNG 格式
    else:
        ext = format  # 使用指定的格式

    # 根据文件格式选择 PIL 库中的对应打开函数
    img_open = (
        PIL.PngImagePlugin.PngImageFile if ext == 'png' else PIL.Image.open)

    # 如果 fname 是 URL，则抛出错误，建议使用正确的方法打开
    if isinstance(fname, str) and len(parse.urlparse(fname).scheme) > 1:
        raise ValueError(
            "Please open the URL for reading and pass the "
            "result to Pillow, e.g. with "
            "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
            )
    # 使用 with 语句打开图像文件 fname，并将其赋值给变量 image
    with img_open(fname) as image:
        # 如果 image 是 PIL 库中的 PNG 图像对象 (PIL.PngImagePlugin.PngImageFile 类型)
        return (_pil_png_to_float_array(image)
                if isinstance(image, PIL.PngImagePlugin.PngImageFile) else
                # 否则调用 pil_to_array 函数将 image 转换为数组
                pil_to_array(image))
# 导入必要的模块和函数
from matplotlib.figure import Figure

# 检查 fname 是否为路径类对象，如果是则转换为字符串路径
if isinstance(fname, os.PathLike):
    fname = os.fspath(fname)

# 如果未指定输出格式（format），则根据 fname 的后缀名推断格式；如果 fname 是字符串，则使用其后缀名推断；否则使用 matplotlib 的默认保存格式
if format is None:
    format = (Path(fname).suffix[1:] if isinstance(fname, str)
              else mpl.rcParams["savefig.format"]).lower()
    # 检查输出格式是否为不适用于 PIL 处理的向量格式
    if format in ["pdf", "ps", "eps", "svg"]:
        # 如果 pil_kwargs 不为空，则抛出数值错误异常
        if pil_kwargs is not None:
            raise ValueError(
                f"Cannot use 'pil_kwargs' when saving to {format}")
        
        # 创建一个 Figure 对象，设定分辨率 dpi，不显示边框
        fig = Figure(dpi=dpi, frameon=False)
        
        # 将数组 arr 渲染到 Figure 对象中，使用指定的 colormap (cmap) 和其他参数
        fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin,
                     resize=True)
        
        # 将 Figure 对象保存为文件 fname，设定分辨率 dpi、输出格式 format，背景透明，并附加元数据
        fig.savefig(fname, dpi=dpi, format=format, transparent=True,
                    metadata=metadata)
    else:
        # 不创建图像；这避免了在 dpi 分割和乘以时出现的舍入误差。
        if origin is None:
            # 如果未指定 origin，则使用默认值从 mpl.rcParams 中获取
            origin = mpl.rcParams["image.origin"]
        else:
            # 检查 origin 是否在指定的列表中
            _api.check_in_list(('upper', 'lower'), origin=origin)
        if origin == "lower":
            # 如果 origin 是 "lower"，则翻转数组 arr
            arr = arr[::-1]
        if (isinstance(arr, memoryview) and arr.format == "B"
                and arr.ndim == 3 and arr.shape[-1] == 4):
            # 如果 arr 是特定格式的 memoryview，直接使用它作为 rgba 数据
            # 这样处理有助于优化，因为 backend_agg 可能会直接传递这种格式
            rgba = arr
        else:
            # 创建一个 ScalarMappable 对象，用于转换 arr 到 RGBA 格式
            sm = cm.ScalarMappable(cmap=cmap)
            sm.set_clim(vmin, vmax)
            rgba = sm.to_rgba(arr, bytes=True)
        if pil_kwargs is None:
            # 如果 pil_kwargs 未定义，则初始化为空字典
            pil_kwargs = {}
        else:
            # 如果 pil_kwargs 已定义，则复制一份，避免修改调用者传入的字典
            pil_kwargs = pil_kwargs.copy()
        # 创建 PIL 图像的大小参数
        pil_shape = (rgba.shape[1], rgba.shape[0])
        # 确保 rgba 数组是连续存储的，以满足 PIL.Image.frombuffer 的要求
        rgba = np.require(rgba, requirements='C')
        # 使用 PIL.Image.frombuffer 创建图像对象 image
        image = PIL.Image.frombuffer(
            "RGBA", pil_shape, rgba, "raw", "RGBA", 0, 1)
        if format == "png":
            # 如果输出格式是 PNG，处理 metadata
            if "pnginfo" in pil_kwargs:
                if metadata:
                    # 如果 pil_kwargs 中包含 'pnginfo'，警告 metadata 会被覆盖
                    _api.warn_external("'metadata' is overridden by the "
                                       "'pnginfo' entry in 'pil_kwargs'.")
            else:
                # 否则，创建 metadata，并添加到 pil_kwargs 中
                metadata = {
                    "Software": (f"Matplotlib version{mpl.__version__}, "
                                 f"https://matplotlib.org/"),
                    **(metadata if metadata is not None else {}),
                }
                pil_kwargs["pnginfo"] = pnginfo = PIL.PngImagePlugin.PngInfo()
                for k, v in metadata.items():
                    if v is not None:
                        pnginfo.add_text(k, v)
        elif metadata is not None:
            # 如果输出格式不支持 metadata，则引发异常
            raise ValueError(f"metadata not supported for format {format!r}")
        if format in ["jpg", "jpeg"]:
            # 如果输出格式是 JPG 或 JPEG，使用 JPEG 格式保存图像
            format = "jpeg"  # Pillow 不识别 "jpg"
            # 获取背景色，默认使用 figure 的 facecolor
            facecolor = mpl.rcParams["savefig.facecolor"]
            if cbook._str_equal(facecolor, "auto"):
                facecolor = mpl.rcParams["figure.facecolor"]
            # 将 facecolor 转换为 RGB 值，并创建背景图像
            color = tuple(int(x * 255) for x in mcolors.to_rgb(facecolor))
            background = PIL.Image.new("RGB", pil_shape, color)
            background.paste(image, image)
            # 将合成的图像设置为最终的 image 对象
            image = background
        # 设置默认的输出格式和 dpi，并保存图像到指定文件 fname
        pil_kwargs.setdefault("format", format)
        pil_kwargs.setdefault("dpi", (dpi, dpi))
        image.save(fname, **pil_kwargs)
def thumbnail(infile, thumbfile, scale=0.1, interpolation='bilinear',
              preview=False):
    """
    Make a thumbnail of image in *infile* with output filename *thumbfile*.

    See :doc:`/gallery/misc/image_thumbnail_sgskip`.

    Parameters
    ----------
    infile : str
        Path to the input image file.
    thumbfile : str
        Path where the thumbnail will be saved.
    scale : float, optional
        Scaling factor for the thumbnail size relative to the original image.
        Default is 0.1 (i.e., 10% of the original size).
    interpolation : str, optional
        Interpolation method used for resizing the image. Possible values are
        'nearest', 'bilinear', 'bicubic', and 'lanczos'. Default is 'bilinear'.
    preview : bool, optional
        If True, display a preview of the thumbnail after saving. Default is False.

    Returns
    -------
    None
    """
    # infile 参数可以是文件路径或类似文件的对象，用于指定输入的图像文件，支持多种格式如 PNG、JPG、TIFF 等
    # Matplotlib 使用 Pillow 库进行图像读取，详细支持的格式参见 Pillow 官网：https://python-pillow.org/
    
    # thumbfile 参数用于指定缩略图的文件名或文件对象
    
    # scale 参数指定缩略图的缩放因子，默认为 0.1
    
    # interpolation 参数指定缩略图生成时的插值方案，影响图像重采样的质量，可参考 `~.Axes.imshow` 的 *interpolation* 参数
    
    # preview 参数指定是否使用默认的用户界面后端，若为 True，则会调用默认后端显示图像，通常与 `~matplotlib.pyplot.show` 一起使用
    
    # 返回值是一个 `.Figure` 对象，包含生成的缩略图
    
    im = imread(infile)
    rows, cols, depth = im.shape
    
    # 这个 dpi 实际上在最后没有实际意义（会被取消），但 API 需要它存在
    
    dpi = 100
    
    height = rows / dpi * scale
    width = cols / dpi * scale
    
    if preview:
        # 让用户界面后端处理所有事务
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(width, height), dpi=dpi)
    else:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasBase
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvasBase(fig)  # 创建 FigureCanvasBase 实例，以确保正确的绘图后端被选择
    
    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])
    ax.imshow(im, aspect='auto', resample=True, interpolation=interpolation)
    fig.savefig(thumbfile, dpi=dpi)
    return fig
```