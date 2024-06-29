# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\art3d.py`

```py
# art3d.py, original mplot3d version by John Porter
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>
# Minor additions by Ben Axelrod <baxelrod@coroware.com>
"""
Module containing 3D artist code and functions to convert 2D
artists into 3D versions which can be added to an Axes3D.
"""

import math  # 导入数学库

import numpy as np  # 导入NumPy库

from contextlib import contextmanager  # 导入上下文管理器

from matplotlib import (
    _api, artist, cbook, colors as mcolors, lines, text as mtext,
    path as mpath)  # 导入Matplotlib的各种模块

from matplotlib.collections import (
    Collection, LineCollection, PolyCollection, PatchCollection, PathCollection)  # 导入Matplotlib的集合类型

from matplotlib.colors import Normalize  # 导入Normalize类

from matplotlib.patches import Patch  # 导入Patch类

from . import proj3d  # 导入当前目录下的proj3d模块


def _norm_angle(a):
    """Return the given angle normalized to -180 < *a* <= 180 degrees."""
    a = (a + 360) % 360  # 将角度a调整为0到360度之间
    if a > 180:
        a = a - 360  # 将角度a调整为-180到180度之间
    return a


def _norm_text_angle(a):
    """Return the given angle normalized to -90 < *a* <= 90 degrees."""
    a = (a + 180) % 180  # 将角度a调整为0到180度之间
    if a > 90:
        a = a - 180  # 将角度a调整为-90到90度之间
    return a


def get_dir_vector(zdir):
    """
    Return a direction vector.

    Parameters
    ----------
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction. Possible values are:

        - 'x': equivalent to (1, 0, 0)
        - 'y': equivalent to (0, 1, 0)
        - 'z': equivalent to (0, 0, 1)
        - *None*: equivalent to (0, 0, 0)
        - an iterable (x, y, z) is converted to an array

    Returns
    -------
    x, y, z : array
        The direction vector.
    """
    if zdir == 'x':
        return np.array((1, 0, 0))  # 返回x方向的单位向量
    elif zdir == 'y':
        return np.array((0, 1, 0))  # 返回y方向的单位向量
    elif zdir == 'z':
        return np.array((0, 0, 1))  # 返回z方向的单位向量
    elif zdir is None:
        return np.array((0, 0, 0))  # 返回零向量
    elif np.iterable(zdir) and len(zdir) == 3:
        return np.array(zdir)  # 将可迭代对象zdir转换为NumPy数组
    else:
        raise ValueError("'x', 'y', 'z', None or vector of length 3 expected")


class Text3D(mtext.Text):
    """
    Text object with 3D position and direction.

    Parameters
    ----------
    x, y, z : float
        The position of the text.
    text : str
        The text string to display.
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction of the text. See `.get_dir_vector` for a description of
        the values.

    Other Parameters
    ----------------
    **kwargs
         All other parameters are passed on to `~matplotlib.text.Text`.
    """

    def __init__(self, x=0, y=0, z=0, text='', zdir='z', **kwargs):
        mtext.Text.__init__(self, x, y, text, **kwargs)  # 调用父类构造函数初始化文本属性
        self.set_3d_properties(z, zdir)  # 设置文本的3D属性

    def get_position_3d(self):
        """Return the (x, y, z) position of the text."""
        return self._x, self._y, self._z  # 返回文本的三维位置信息
    # 设置文本在三维空间中的位置 (*x*, *y*, *z*)。
    def set_position_3d(self, xyz, zdir=None):
        """
        Set the (*x*, *y*, *z*) position of the text.

        Parameters
        ----------
        xyz : (float, float, float)
            The position in 3D space.
        zdir : {'x', 'y', 'z', None, 3-tuple}
            The direction of the text. If unspecified, the *zdir* will not be
            changed. See `.get_dir_vector` for a description of the values.
        """
        # 调用父类方法设置文本在二维平面上的位置，忽略 z 轴坐标
        super().set_position(xyz[:2])
        # 设置文本在 z 轴上的位置
        self.set_z(xyz[2])
        # 如果指定了 zdir 参数，则根据其值设置文本的方向向量
        if zdir is not None:
            self._dir_vec = get_dir_vector(zdir)

    # 设置文本在 z 轴上的位置。
    def set_z(self, z):
        """
        Set the *z* position of the text.

        Parameters
        ----------
        z : float
        """
        # 设置文本在 z 轴上的具体位置
        self._z = z
        # 设定文本属性为需要更新
        self.stale = True

    # 设置文本在三维空间中的位置和方向。
    def set_3d_properties(self, z=0, zdir='z'):
        """
        Set the *z* position and direction of the text.

        Parameters
        ----------
        z : float
            The z-position in 3D space.
        zdir : {'x', 'y', 'z', 3-tuple}
            The direction of the text. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        # 设置文本在 z 轴上的位置
        self._z = z
        # 根据 zdir 参数获取文本的方向向量，并设置
        self._dir_vec = get_dir_vector(zdir)
        # 设定文本属性为需要更新
        self.stale = True

    # 使用允许栅格化的艺术家方法绘制文本。
    @artist.allow_rasterization
    def draw(self, renderer):
        # 计算文本在三维空间中的位置
        position3d = np.array((self._x, self._y, self._z))
        # 投影转换三维位置点到二维画布上
        proj = proj3d._proj_trans_points(
            [position3d, position3d + self._dir_vec], self.axes.M)
        # 计算文本在二维平面上的角度
        dx = proj[0][1] - proj[0][0]
        dy = proj[1][1] - proj[1][0]
        angle = math.degrees(math.atan2(dy, dx))
        # 在上下文中设置文本的位置、旋转角度
        with cbook._setattr_cm(self, _x=proj[0][0], _y=proj[1][0],
                               _rotation=_norm_text_angle(angle)):
            # 调用父类的绘制方法来绘制文本
            mtext.Text.draw(self, renderer)
        # 更新文本的绘制状态为最新
        self.stale = False

    # 获取紧凑的包围框，这里覆盖了二维文本行为，不适用于三维，返回 None 来排除其布局计算。
    def get_tightbbox(self, renderer=None):
        # 覆盖了二维文本的行为，对于三维文本返回 None，不进行布局计算。
        return None
def text_2d_to_3d(obj, z=0, zdir='z'):
    """
    Convert a `.Text` to a `.Text3D` object.

    Parameters
    ----------
    z : float
        The z-position in 3D space.
    zdir : {'x', 'y', 'z', 3-tuple}
        The direction of the text. Default: 'z'.
        See `.get_dir_vector` for a description of the values.
    """
    # 将输入对象转换为3D文本对象
    obj.__class__ = Text3D
    # 设置3D属性，包括z轴位置和方向
    obj.set_3d_properties(z, zdir)


class Line3D(lines.Line2D):
    """
    3D line object.

    .. note:: Use `get_data_3d` to obtain the data associated with the line.
            `~.Line2D.get_data`, `~.Line2D.get_xdata`, and `~.Line2D.get_ydata` return
            the x- and y-coordinates of the projected 2D-line, not the x- and y-data of
            the 3D-line. Similarly, use `set_data_3d` to set the data, not
            `~.Line2D.set_data`, `~.Line2D.set_xdata`, and `~.Line2D.set_ydata`.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """

        Parameters
        ----------
        xs : array-like
            The x-data to be plotted.
        ys : array-like
            The y-data to be plotted.
        zs : array-like
            The z-data to be plotted.
        *args, **kwargs
            Additional arguments are passed to `~matplotlib.lines.Line2D`.
        """
        # 调用父类构造函数初始化2D线对象，但传入空列表作为数据
        super().__init__([], [], *args, **kwargs)
        # 使用给定的数据设置3D数据属性
        self.set_data_3d(xs, ys, zs)

    def set_3d_properties(self, zs=0, zdir='z'):
        """
        Set the *z* position and direction of the line.

        Parameters
        ----------
        zs : float or array of floats
            The location along the *zdir* axis in 3D space to position the
            line.
        zdir : {'x', 'y', 'z'}
            Plane to plot line orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        # 获取当前线的x、y数据
        xs = self.get_xdata()
        ys = self.get_ydata()
        # 将zs转换为浮点数数组，并展平处理
        zs = cbook._to_unmasked_float_array(zs).ravel()
        # 将zs广播到与xs相同的长度
        zs = np.broadcast_to(zs, len(xs))
        # 使用给定的x、y、z数据和方向zdir，重新排列坐标轴
        self._verts3d = juggle_axes(xs, ys, zs, zdir)
        # 设置对象已经过时，需要更新
        self.stale = True

    def set_data_3d(self, *args):
        """
        Set the x, y and z data

        Parameters
        ----------
        x : array-like
            The x-data to be plotted.
        y : array-like
            The y-data to be plotted.
        z : array-like
            The z-data to be plotted.

        Notes
        -----
        Accepts x, y, z arguments or a single array-like (x, y, z)
        """
        # 如果参数长度为1，则将其解包
        if len(args) == 1:
            args = args[0]
        # 验证xyz数据是否可迭代，否则引发运行时错误
        for name, xyz in zip('xyz', args):
            if not np.iterable(xyz):
                raise RuntimeError(f'{name} must be a sequence')
        # 将x、y、z数据设置为对象的3D数据
        self._verts3d = args
        # 设置对象已经过时，需要更新
        self.stale = True

    def get_data_3d(self):
        """
        Get the current data

        Returns
        -------
        verts3d : length-3 tuple or array-like
            The current data as a tuple or array-like.
        """
        # 返回当前保存的3D数据
        return self._verts3d

    @artist.allow_rasterization
    # 定义一个绘制方法，接受一个渲染器作为参数
    def draw(self, renderer):
        # 从 self._verts3d 中解包出三个数组 xs3d, ys3d, zs3d，分别代表三维坐标系中的 x, y, z 值
        xs3d, ys3d, zs3d = self._verts3d
        # 使用 proj3d.proj_transform 方法将三维坐标投影到二维平面上，得到投影后的 xs, ys, zs 值
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        # 调用父类的 set_data 方法设置数据，这里设置了二维平面上的 xs 和 ys 值
        self.set_data(xs, ys)
        # 调用父类的 draw 方法进行绘制，使用传入的渲染器进行渲染操作
        super().draw(renderer)
        # 将对象的 stale 属性设置为 False，表示对象不需要更新
        self.stale = False
def line_2d_to_3d(line, zs=0, zdir='z'):
    """
    Convert a `.Line2D` to a `.Line3D` object.

    Parameters
    ----------
    zs : float
        The location along the *zdir* axis in 3D space to position the line.
    zdir : {'x', 'y', 'z'}
        Plane to plot line orthogonal to. Default: 'z'.
        See `.get_dir_vector` for a description of the values.
    """

    # 将传入的二维线对象转换为三维线对象
    line.__class__ = Line3D
    # 设置三维线对象的三维属性
    line.set_3d_properties(zs, zdir)


def _path_to_3d_segment(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment."""

    # 将路径对象转换为一维的三维段
    zs = np.broadcast_to(zs, len(path))
    # 使用简单模式和不包含曲线的段来迭代路径的段
    pathsegs = path.iter_segments(simplify=False, curves=False)
    # 根据路径段和 z 轴方向生成三维段
    seg = [(x, y, z) for (((x, y), code), z) in zip(pathsegs, zs)]
    # 根据 z 轴方向对坐标进行重新排列，生成最终的三维段
    seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    return seg3d


def _paths_to_3d_segments(paths, zs=0, zdir='z'):
    """Convert paths from a collection object to 3D segments."""

    # 如果 zs 不可迭代，扩展为与路径数量相同长度的数组
    if not np.iterable(zs):
        zs = np.broadcast_to(zs, len(paths))
    else:
        # 如果 zs 的长度与路径数量不匹配，抛出数值错误
        if len(zs) != len(paths):
            raise ValueError('Number of z-coordinates does not match paths.')

    # 对每个路径调用 _path_to_3d_segment 函数，生成路径对应的三维段列表
    segs = [_path_to_3d_segment(path, pathz, zdir)
            for path, pathz in zip(paths, zs)]
    return segs


def _path_to_3d_segment_with_codes(path, zs=0, zdir='z'):
    """Convert a path to a 3D segment with path codes."""

    # 将 zs 扩展为与路径长度相同的数组
    zs = np.broadcast_to(zs, len(path))
    # 使用简单模式和不包含曲线的段来迭代路径的段
    pathsegs = path.iter_segments(simplify=False, curves=False)
    # 构建包含路径代码的段列表
    seg_codes = [((x, y, z), code) for ((x, y), code), z in zip(pathsegs, zs)]
    if seg_codes:
        seg, codes = zip(*seg_codes)
        # 根据 z 轴方向对坐标进行重新排列，生成最终的三维段及其代码列表
        seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    else:
        seg3d = []
        codes = []
    return seg3d, list(codes)


def _paths_to_3d_segments_with_codes(paths, zs=0, zdir='z'):
    """
    Convert paths from a collection object to 3D segments with path codes.
    """

    # 将 zs 扩展为与路径数量相同长度的数组
    zs = np.broadcast_to(zs, len(paths))
    # 对每个路径调用 _path_to_3d_segment_with_codes 函数，生成包含代码的三维段列表
    segments_codes = [_path_to_3d_segment_with_codes(path, pathz, zdir)
                      for path, pathz in zip(paths, zs)]
    if segments_codes:
        segments, codes = zip(*segments_codes)
    else:
        segments, codes = [], []
    return list(segments), list(codes)


class Collection3D(Collection):
    """A collection of 3D paths."""

    def do_3d_projection(self):
        """Project the points according to renderer matrix."""
        # 对每个顶点进行三维投影转换
        xyzs_list = [proj3d.proj_transform(*vs.T, self.axes.M)
                     for vs, _ in self._3dverts_codes]
        # 根据投影后的结果重新构建路径对象列表
        self._paths = [mpath.Path(np.column_stack([xs, ys]), cs)
                       for (xs, ys, _), (_, cs) in zip(xyzs_list, self._3dverts_codes)]
        # 获取所有 z 坐标的最小值，若为空则返回一个很大的数
        zs = np.concatenate([zs for _, _, zs in xyzs_list])
        return zs.min() if len(zs) else 1e9


def collection_2d_to_3d(col, zs=0, zdir='z'):
    """Convert a `.Collection` to a `.Collection3D` object."""
    # 将 zs 扩展为与路径数量相同长度的数组
    zs = np.broadcast_to(zs, len(col.get_paths()))
    # 将 _3dverts_codes 属性设置为一个列表推导式的结果，该列表推导式通过对每个路径和相应的 z 值进行操作得到的结果组成
    col._3dverts_codes = [
        # 对于每个路径 p 和对应的 z 值，执行以下操作：
        (
            # 利用 juggle_axes 函数对顶点和 z 值进行处理，然后通过 column_stack 将结果按列堆叠成二维数组
            np.column_stack(juggle_axes(
                *np.column_stack([p.vertices, np.broadcast_to(z, len(p.vertices))]).T,
                zdir)),
            # 保留路径 p 的 codes 属性
            p.codes
        )
        # 遍历 Collection 对象 col 的所有路径及相应的 zs 值
        for p, z in zip(col.get_paths(), zs)
    ]
    # 修改 col 对象的类为 Collection3D 类型，通过 cbook._make_class_factory 动态创建新类并应用
    col.__class__ = cbook._make_class_factory(Collection3D, "{}3D")(type(col))
class Line3DCollection(LineCollection):
    """
    A collection of 3D lines.
    """

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        # 设置用于 z 排序的位置值
        self._sort_zpos = val
        self.stale = True

    def set_segments(self, segments):
        """
        Set 3D segments.
        """
        # 设置 3D 线段
        self._segments3d = segments
        super().set_segments([])

    def do_3d_projection(self):
        """
        Project the points according to renderer matrix.
        """
        # 将点根据渲染器矩阵投影
        xyslist = [proj3d._proj_trans_points(points, self.axes.M)
                   for points in self._segments3d]
        segments_2d = [np.column_stack([xs, ys]) for xs, ys, zs in xyslist]
        LineCollection.set_segments(self, segments_2d)

        # FIXME
        # 计算投影后点集中最小的 z 值
        minz = 1e9
        for xs, ys, zs in xyslist:
            minz = min(minz, min(zs))
        return minz


def line_collection_2d_to_3d(col, zs=0, zdir='z'):
    """Convert a `.LineCollection` to a `.Line3DCollection` object."""
    # 将 `.LineCollection` 转换为 `.Line3DCollection` 对象
    segments3d = _paths_to_3d_segments(col.get_paths(), zs, zdir)
    col.__class__ = Line3DCollection
    col.set_segments(segments3d)


class Patch3D(Patch):
    """
    3D patch object.
    """

    def __init__(self, *args, zs=(), zdir='z', **kwargs):
        """
        Parameters
        ----------
        verts :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            patch.
        zdir : {'x', 'y', 'z'}
            Plane to plot patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def set_3d_properties(self, verts, zs=0, zdir='z'):
        """
        Set the *z* position and direction of the patch.

        Parameters
        ----------
        verts :
        zs : float
            The location along the *zdir* axis in 3D space to position the
            patch.
        zdir : {'x', 'y', 'z'}
            Plane to plot patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for a description of the values.
        """
        zs = np.broadcast_to(zs, len(verts))
        # 设置 patch 的 3D 属性
        self._segment3d = [juggle_axes(x, y, z, zdir)
                           for ((x, y), z) in zip(verts, zs)]

    def get_path(self):
        # docstring inherited
        # self._path2d is not initialized until do_3d_projection
        if not hasattr(self, '_path2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return self._path2d

    def do_3d_projection(self):
        s = self._segment3d
        xs, ys, zs = zip(*s)
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs,
                                                        self.axes.M)
        # 将 3D patch 投影到 2D 平面
        self._path2d = mpath.Path(np.column_stack([vxs, vys]))
        return min(vzs)


class PathPatch3D(Patch3D):
    """
    3D PathPatch object.
    """
    def __init__(self, path, *, zs=(), zdir='z', **kwargs):
        """
        Parameters
        ----------
        path : Path
            The path object representing the path patch.
        zs : float
            The position along the *zdir* axis in 3D space to place the path patch.
        zdir : {'x', 'y', 'z', 3-tuple}
            The plane to plot the path patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for details on accepted values.
        kwargs : dict
            Additional keyword arguments passed to the Patch initializer.
        """
        # 初始化 Patch 类的实例，使用传入的关键字参数
        Patch.__init__(self, **kwargs)
        # 调用 set_3d_properties 方法设置 3D 属性
        self.set_3d_properties(path, zs, zdir)

    def set_3d_properties(self, path, zs=0, zdir='z'):
        """
        Set the *z* position and direction of the path patch.

        Parameters
        ----------
        path : Path
            The path object representing the path patch.
        zs : float
            The location along the *zdir* axis in 3D space to position the path patch.
        zdir : {'x', 'y', 'z', 3-tuple}
            The plane to plot the path patch orthogonal to. Default: 'z'.
            See `.get_dir_vector` for details on accepted values.
        """
        # 调用 Patch3D 类的 set_3d_properties 方法设置 3D 属性
        Patch3D.set_3d_properties(self, path.vertices, zs=zs, zdir=zdir)
        # 将 path 的代码存储在 _code3d 属性中
        self._code3d = path.codes

    def do_3d_projection(self):
        # 获取 _segment3d 属性的值
        s = self._segment3d
        # 将 _segment3d 属性解压缩为 xs, ys, zs 三个列表
        xs, ys, zs = zip(*s)
        # 调用 proj3d.proj_transform_clip 方法进行投影转换，获取投影后的坐标和可见性
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs,
                                                        self.axes.M)
        # 使用投影后的坐标创建二维路径对象 _path2d
        self._path2d = mpath.Path(np.column_stack([vxs, vys]), self._code3d)
        # 返回投影后所有点的最小 z 值
        return min(vzs)
def _get_patch_verts(patch):
    """
    Return a list of vertices for the path of a patch.

    Parameters
    ----------
    patch : matplotlib.patch.Patch
        The patch object from which vertices are extracted.

    Returns
    -------
    numpy.ndarray
        Array of vertices representing the path of the patch.
    """
    trans = patch.get_patch_transform()  # 获取 patch 的变换对象
    path = patch.get_path()  # 获取 patch 的路径对象
    polygons = path.to_polygons(trans)  # 将路径转换为多边形顶点数组
    return polygons[0] if len(polygons) else np.array([])  # 返回第一个多边形的顶点数组


def patch_2d_to_3d(patch, z=0, zdir='z'):
    """
    Convert a `.Patch` to a `.Patch3D` object.

    Parameters
    ----------
    patch : matplotlib.patch.Patch
        The 2D patch object to convert.
    z : float, optional
        The z-coordinate of the patch in 3D space (default is 0).
    zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, optional
        The direction along which to apply the z-coordinate (default is 'z').

    Notes
    -----
    Modifies the input `patch` object to become a 3D patch.
    """
    verts = _get_patch_verts(patch)  # 获取 patch 的路径顶点数组
    patch.__class__ = Patch3D  # 将 patch 的类修改为 Patch3D
    patch.set_3d_properties(verts, z, zdir)  # 设置 patch 的 3D 属性


def pathpatch_2d_to_3d(pathpatch, z=0, zdir='z'):
    """
    Convert a `.PathPatch` to a `.PathPatch3D` object.

    Parameters
    ----------
    pathpatch : matplotlib.patches.PathPatch
        The 2D PathPatch object to convert.
    z : float, optional
        The z-coordinate of the patch in 3D space (default is 0).
    zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, optional
        The direction along which to apply the z-coordinate (default is 'z').

    Notes
    -----
    Modifies the input `pathpatch` object to become a 3D PathPatch.
    """
    path = pathpatch.get_path()  # 获取 PathPatch 的路径对象
    trans = pathpatch.get_patch_transform()  # 获取 PathPatch 的变换对象

    mpath = trans.transform_path(path)  # 对路径进行变换得到新的路径对象
    pathpatch.__class__ = PathPatch3D  # 将 pathpatch 的类修改为 PathPatch3D
    pathpatch.set_3d_properties(mpath, z, zdir)  # 设置 pathpatch 的 3D 属性


class Patch3DCollection(PatchCollection):
    """
    A collection of 3D patches.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        """
        Create a collection of flat 3D patches with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of patches in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PatchCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument *depthshade* is available to indicate
        whether to shade the patches in order to give the appearance of depth
        (default is *True*). This is typically desired in scatter plots.
        """
        self._depthshade = depthshade  # 设置是否进行深度阴影
        super().__init__(*args, **kwargs)  # 调用父类构造函数初始化

        # 设置集合的 3D 属性
        self.set_3d_properties(zs, zdir)

    def get_depthshade(self):
        """
        Get the current depth shading status.

        Returns
        -------
        bool
            Current depth shading status.
        """
        return self._depthshade  # 返回深度阴影状态

    def set_depthshade(self, depthshade):
        """
        Set whether depth shading is performed on collection members.

        Parameters
        ----------
        depthshade : bool
            Whether to shade the patches in order to give the appearance of
            depth.
        """
        self._depthshade = depthshade  # 设置深度阴影状态
        self.stale = True  # 标记对象为过时

    def set_sort_zpos(self, val):
        """
        Set the position to use for z-sorting.

        Parameters
        ----------
        val : float
            The z-position value to use for sorting.
        """
        self._sort_zpos = val  # 设置用于 z 排序的位置值
        self.stale = True  # 标记对象为过时
    def set_3d_properties(self, zs, zdir):
        """
        设置图形元素的三维属性。

        Parameters
        ----------
        zs : float or array of floats
            放置图形元素的位置或位置数组，沿着指定的 *zdir* 轴。
        zdir : {'x', 'y', 'z'}
            绘制图形元素的平面。
            所有的图形元素必须具有相同的方向。
            参见 `.get_dir_vector` 获取值的描述。
        """
        # 强制集合初始化面和边缘颜色，
        # 以防它是带有颜色映射的标量映射。
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            xs, ys = offsets.T
        else:
            xs = []
            ys = []
        # 根据给定的 zdir 轴重新排列偏移坐标和 zs，以得到三维偏移坐标。
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        # 设置索引，用于标记 z 方向的坐标点
        self._z_markers_idx = slice(-1)
        # 重置 _vzs 属性为 None，标记数据已过期
        self._vzs = None
        # 设置 stale 属性为 True，标记数据已过期
        self.stale = True

    def do_3d_projection(self):
        xs, ys, zs = self._offsets3d
        # 对三维坐标进行投影变换，并进行剪裁
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs,
                                                        self.axes.M)
        # 更新 _vzs 属性为投影后的 z 坐标
        self._vzs = vzs
        # 调用父类方法设置偏移坐标为投影后的二维坐标
        super().set_offsets(np.column_stack([vxs, vys]))

        # 如果投影后的 z 坐标数组大小大于 0，返回最小的 z 坐标
        if vzs.size > 0:
            return min(vzs)
        else:
            return np.nan

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        # 如果存在 _vzs 属性且启用了深度阴影，对颜色数组进行深度阴影处理
        color_array = (
            _zalpha(color_array, self._vzs)
            if self._vzs is not None and self._depthshade
            else color_array
        )
        # 如果颜色数组长度大于 1，只保留与 _z_markers_idx 对应的颜色
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        # 将颜色数组转换为 RGBA 格式并应用 alpha 值
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        # 返回处理后的面颜色数组
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        # 在边缘颜色为 "face" 时，确保不重复应用基于深度的 alpha 阴影
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        # 返回处理后的边缘颜色数组
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())
class Path3DCollection(PathCollection):
    """
    A collection of 3D paths.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        """
        Create a collection of flat 3D paths with its normal vector
        pointed in *zdir* direction, and located at *zs* on the *zdir*
        axis. 'zs' can be a scalar or an array-like of the same length as
        the number of paths in the collection.

        Constructor arguments are the same as for
        :class:`~matplotlib.collections.PathCollection`. In addition,
        keywords *zs=0* and *zdir='z'* are available.

        Also, the keyword argument *depthshade* is available to indicate
        whether to shade the patches in order to give the appearance of depth
        (default is *True*). This is typically desired in scatter plots.
        """
        # 设置深度阴影标志
        self._depthshade = depthshade
        # 初始化绘制状态为非绘制中
        self._in_draw = False
        # 调用父类的构造方法，传递所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
        # 设置3D属性，包括zs和zdir
        self.set_3d_properties(zs, zdir)
        # 初始化偏移z排序为None
        self._offset_zordered = None

    def draw(self, renderer):
        # 使用偏移进行z排序
        with self._use_zordered_offset():
            # 设置绘制状态为绘制中
            with cbook._setattr_cm(self, _in_draw=True):
                # 调用父类的绘制方法
                super().draw(renderer)

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        # 设置用于z排序的位置值
        self._sort_zpos = val
        # 标记为需要重新绘制
        self.stale = True
    def set_3d_properties(self, zs, zdir):
        """
        Set the *z* positions and direction of the paths.

        Parameters
        ----------
        zs : float or array of floats
            The location or locations to place the paths in the collection
            along the *zdir* axis.
        zdir : {'x', 'y', 'z'}
            Plane to plot paths orthogonal to.
            All paths must have the same direction.
            See `.get_dir_vector` for a description of the values.
        """
        # 强制更新集合以初始化面和边缘颜色，
        # 以防它是一个带有颜色映射的标量映射对象。
        self.update_scalarmappable()
        
        # 获取当前偏移量
        offsets = self.get_offsets()
        
        if len(offsets) > 0:
            # 如果存在偏移量，提取出xs和ys
            xs, ys = offsets.T
        else:
            # 如果没有偏移量，则置为空列表
            xs = []
            ys = []
        
        # 根据给定的z方向，重新排列偏移量数组
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        
        # 在基本绘图方法中，我们直接访问属性，这意味着我们不能像边缘和面颜色的获取方法那样解析洗牌过程。
        #
        # 这意味着我们需要携带一个未排序大小和宽度的缓存（后缀为3d），并且在`do_3d_projection`方法中，
        # 将排序后的数据设置到基本集合类的绘图方法中使用的私有状态中。
        #
        # 获取当前的大小和线宽以保留它们。
        self._sizes3d = self._sizes
        self._linewidths3d = np.array(self._linewidths)
        
        # 提取重新排列后的xs、ys和zs
        xs, ys, zs = self._offsets3d
        
        # 根据z坐标对点进行排序
        # 性能优化：创建一个排序后的索引数组，并根据索引数组重新排序点和点属性
        self._z_markers_idx = slice(-1)
        self._vzs = None
        self.stale = True
    ```
    def do_3d_projection(self):
        xs, ys, zs = self._offsets3d
        vxs, vys, vzs, vis = proj3d.proj_transform_clip(xs, ys, zs,
                                                        self.axes.M)
        # 根据 z 坐标对点进行排序
        # 性能优化：创建排序后的索引数组，并根据该索引数组重新排序点和点属性
        z_markers_idx = self._z_markers_idx = np.argsort(vzs)[::-1]
        self._vzs = vzs

        # 由于 collections.py 中的代码，我们必须特殊处理大小
        #      self.set_sizes(self._sizes, self.figure.dpi)
        # 因此我们不能依赖 get_* 方法中的排序方式

        if len(self._sizes3d) > 1:
            self._sizes = self._sizes3d[z_markers_idx]

        if len(self._linewidths3d) > 1:
            self._linewidths = self._linewidths3d[z_markers_idx]

        # 设置偏移量
        PathCollection.set_offsets(self, np.column_stack((vxs, vys)))

        # 重新排序项目
        vzs = vzs[z_markers_idx]
        vxs = vxs[z_markers_idx]
        vys = vys[z_markers_idx]

        # 存储有序的偏移量以供绘制使用
        self._offset_zordered = np.column_stack((vxs, vys))

        return np.min(vzs) if vzs.size else np.nan

    @contextmanager
    def _use_zordered_offset(self):
        if self._offset_zordered is None:
            # 如果没有有序偏移量，不做任何操作
            yield
        else:
            # 使用有序偏移量替换当前偏移量
            old_offset = self._offsets
            super().set_offsets(self._offset_zordered)
            try:
                yield
            finally:
                self._offsets = old_offset

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        # 根据深度进行颜色排序和深度阴影处理
        color_array = (
            _zalpha(color_array, self._vzs)
            if self._vzs is not None and self._depthshade
            else color_array
        )
        if len(color_array) > 1:
            # 根据 z 索引数组排序颜色数组
            color_array = color_array[self._z_markers_idx]
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        # 获取面颜色，可能会根据深度进行阴影处理和排序颜色
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        # 我们需要在此处进行检查，以确保当边缘颜色为 "face" 时不重复应用基于深度的 alpha 阴影
        # 这意味着边缘颜色应与面颜色相同
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        # 否则可能根据深度进行阴影处理和排序边缘颜色
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())
def patch_collection_2d_to_3d(col, zs=0, zdir='z', depthshade=True):
    """
    Convert a `.PatchCollection` into a `.Patch3DCollection` object
    (or a `.PathCollection` into a `.Path3DCollection` object).

    Parameters
    ----------
    col : `~matplotlib.collections.PatchCollection` or \
`~matplotlib.collections.PathCollection`
        The collection to convert.
    zs : float or array of floats
        The location or locations to place the patches in the collection along
        the *zdir* axis. Default: 0.
    zdir : {'x', 'y', 'z'}
        The axis in which to place the patches. Default: "z".
        See `.get_dir_vector` for a description of the values.
    depthshade : bool, default: True
        Whether to shade the patches to give a sense of depth.

    """
    # 如果传入的集合是 PathCollection 类型
    if isinstance(col, PathCollection):
        # 将其类别转换为 Path3DCollection
        col.__class__ = Path3DCollection
        # 清除 z 轴排序偏移量
        col._offset_zordered = None
    # 如果传入的集合是 PatchCollection 类型
    elif isinstance(col, PatchCollection):
        # 将其类别转换为 Patch3DCollection
        col.__class__ = Patch3DCollection
    # 设置集合的深度阴影属性
    col._depthshade = depthshade
    # 设置集合不在绘制过程中
    col._in_draw = False
    # 设置集合的 3D 属性，即在指定的 zdir 轴上放置 patch 的位置
    col.set_3d_properties(zs, zdir)


class Poly3DCollection(PolyCollection):
    """
    A collection of 3D polygons.

    .. note::
        **Filling of 3D polygons**

        There is no simple definition of the enclosed surface of a 3D polygon
        unless the polygon is planar.

        In practice, Matplotlib fills the 2D projection of the polygon. This
        gives a correct filling appearance only for planar polygons. For all
        other polygons, you'll find orientations in which the edges of the
        polygon intersect in the projection. This will lead to an incorrect
        visualization of the 3D area.

        If you need filled areas, it is recommended to create them via
        `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`, which creates a
        triangulation and thus generates consistent surfaces.
    """
    def __init__(self, verts, *args, zsort='average', shade=False,
                 lightsource=None, **kwargs):
        """
        Parameters
        ----------
        verts : list of (N, 3) array-like
            定义多边形的顶点序列 [*verts0*, *verts1*, ...]，其中每个
            元素 *verts_i* 定义多边形 *i* 的顶点，形状为 (N, 3) 的类数组对象。
        zsort : {'average', 'min', 'max'}, default: 'average'
            用于计算 z-order 的方法。
            详见 `~.Poly3DCollection.set_zsort` 获取详细信息。
        shade : bool, default: False
            是否对 *facecolors* 和 *edgecolors* 进行阴影处理。启用
            *shade* 时，必须提供 *facecolors* 和/或 *edgecolors*。

            .. versionadded:: 3.7

        lightsource : `~matplotlib.colors.LightSource`, optional
            当 *shade* 为 True 时使用的光源。

            .. versionadded:: 3.7

        *args, **kwargs
            所有其他参数都会传递给 `.PolyCollection`。

        Notes
        -----
        请注意，此类通过 _facecolors 和 _edgecolors 属性进行一些特殊处理。
        """
        if shade:
            # 如果 shade 参数为 True，则生成法线向量
            normals = _generate_normals(verts)
            # 获取 facecolors 参数，并进行阴影处理
            facecolors = kwargs.get('facecolors', None)
            if facecolors is not None:
                kwargs['facecolors'] = _shade_colors(
                    facecolors, normals, lightsource
                )

            # 获取 edgecolors 参数，并进行阴影处理
            edgecolors = kwargs.get('edgecolors', None)
            if edgecolors is not None:
                kwargs['edgecolors'] = _shade_colors(
                    edgecolors, normals, lightsource
                )
            # 如果未提供 facecolors 和 edgecolors，则抛出 ValueError
            if facecolors is None and edgecolors is None:
                raise ValueError(
                    "You must provide facecolors, edgecolors, or both for "
                    "shade to work.")
        
        # 调用父类构造函数初始化 Poly3DCollection 对象
        super().__init__(verts, *args, **kwargs)
        
        # 检查 verts 是否为 ndarray 类型，并验证其维度是否为 3
        if isinstance(verts, np.ndarray):
            if verts.ndim != 3:
                raise ValueError('verts must be a list of (N, 3) array-like')
        else:
            # 检查 verts 中的每个元素是否为 (N, 3) 形状的二维数组
            if any(len(np.shape(vert)) != 2 for vert in verts):
                raise ValueError('verts must be a list of (N, 3) array-like')
        
        # 设置 z-sort 方法
        self.set_zsort(zsort)
        
        # 初始化 _codes3d 属性为 None
        self._codes3d = None

    _zsort_functions = {
        'average': np.average,
        'min': np.min,
        'max': np.max,
    }

    def set_zsort(self, zsort):
        """
        设置 z-order 的计算方法。

        Parameters
        ----------
        zsort : {'average', 'min', 'max'}
            应用于查看器坐标系中顶点 z 坐标的函数，用于确定 z-order。
        """
        # 根据 zsort 参数设置 z-sort 函数
        self._zsortfunc = self._zsort_functions[zsort]
        # 将 _sort_zpos 属性设置为 None
        self._sort_zpos = None
        # 将 stale 属性设置为 True，表示需要更新
        self.stale = True

    @_api.deprecated("3.10")
    def get_vector(self, segments3d):
        return self._get_vector(segments3d)
    def _get_vector(self, segments3d):
        """Optimize points for projection."""
        # 检查传入的3D段落数组是否非空
        if len(segments3d):
            # 如果非空，将各个段落的点分别按列（x、y、z）堆叠成数组
            xs, ys, zs = np.vstack(segments3d).T
        else:  # vstack can't stack zero arrays.
            # 如果为空，则分别创建空数组
            xs, ys, zs = [], [], []
        # 创建一个全为1的数组，用于点的齐次坐标表示
        ones = np.ones(len(xs))
        # 将所有点的坐标及齐次坐标组成一个4行N列的数组
        self._vec = np.array([xs, ys, zs, ones])

        # 计算各个段落在_vec数组中的切片位置
        indices = [0, *np.cumsum([len(segment) for segment in segments3d])]
        self._segslices = [*map(slice, indices[:-1], indices[1:])]

    def set_verts(self, verts, closed=True):
        """
        Set 3D vertices.

        Parameters
        ----------
        verts : list of (N, 3) array-like
            The sequence of polygons [*verts0*, *verts1*, ...] where each
            element *verts_i* defines the vertices of polygon *i* as a 2D
            array-like of shape (N, 3).
        closed : bool, default: True
            Whether the polygon should be closed by adding a CLOSEPOLY
            connection at the end.
        """
        # 调用_get_vector方法处理传入的顶点数组
        self._get_vector(verts)
        # 在绘制时更新2D顶点数组
        super().set_verts([], False)
        # 设置多边形是否闭合的标志
        self._closed = closed

    def set_verts_and_codes(self, verts, codes):
        """Set 3D vertices with path codes."""
        # 调用set_verts方法，将闭合标志设为False，以防止PolyCollection设置路径代码
        self.set_verts(verts, closed=False)
        # 设置自定义的路径代码
        self._codes3d = codes

    def set_3d_properties(self):
        # 强制集合初始化面颜色和边缘颜色，以防它是一个使用颜色映射的标量映射对象
        self.update_scalarmappable()
        # 清空_zpos排序位置
        self._sort_zpos = None
        # 设置Z轴排序方式为平均值
        self.set_zsort('average')
        # 获取PolyCollection的面颜色属性
        self._facecolor3d = PolyCollection.get_facecolor(self)
        # 获取PolyCollection的边缘颜色属性
        self._edgecolor3d = PolyCollection.get_edgecolor(self)
        # 获取PolyCollection的透明度属性
        self._alpha3d = PolyCollection.get_alpha(self)
        # 设置为失效状态，需要更新
        self.stale = True

    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting."""
        # 设置用于Z轴排序的位置值
        self._sort_zpos = val
        # 设置为失效状态，需要更新
        self.stale = True
    def do_3d_projection(self):
        """
        Perform the 3D projection for this object.
        """
        # 如果存在顶点数据（self._A不为None），则进行以下操作：
        if self._A is not None:
            # 强制更新颜色映射，因为我们重新排序了顶点数据。
            # 如果不在这里进行更新，2D绘制将调用此方法，但我们将无法将颜色映射的值传回3D版本。
            #
            # 我们保持顶点数据在固定顺序中（用户传入的顺序），并按视图深度对2D版本进行排序。
            self.update_scalarmappable()
            # 如果面的颜色已映射，则将其设置为_facecolors
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            # 如果边的颜色已映射，则将其设置为_edgecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors

        # 对顶点进行投影变换，得到投影后的坐标
        txs, tys, tzs = proj3d._proj_transform_vec(self._vec, self.axes.M)
        # 根据切片将顶点坐标组成列表
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]

        # 重新排序面和边的颜色，确保它们与顶点一一对应
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)
        if len(cedge) != len(xyzlist):
            if len(cedge) == 0:
                cedge = cface
            else:
                cedge = cedge.repeat(len(xyzlist), axis=0)

        # 如果存在顶点列表，则按照深度排序（最远的先绘制）
        if xyzlist:
            z_segments_2d = sorted(
                ((self._zsortfunc(zs), np.column_stack([xs, ys]), fc, ec, idx)
                 for idx, ((xs, ys, zs), fc, ec)
                 in enumerate(zip(xyzlist, cface, cedge))),
                key=lambda x: x[0], reverse=True)

            _, segments_2d, self._facecolors2d, self._edgecolors2d, idxs = \
                zip(*z_segments_2d)
        else:
            segments_2d = []
            self._facecolors2d = np.empty((0, 4))
            self._edgecolors2d = np.empty((0, 4))
            idxs = []

        # 如果存在self._codes3d，则设置顶点和代码
        if self._codes3d is not None:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)
        else:
            PolyCollection.set_verts(self, segments_2d, self._closed)

        # 如果边的颜色长度与面的颜色长度不相等，则将边的颜色设置为_facecolor3d
        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d

        # 返回 zorder 值
        if self._sort_zpos is not None:
            # 计算特定深度的投影坐标
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            # 如果存在投影后的 z 值，则返回最小的值
            # FIXME: 有些结果看起来仍然不太对。
            #        特别是，查看带有 az = -54 和 elev = -45 的 contourf3d_demo2.py。
            return np.min(tzs)
        else:
            # 如果没有有效的投影后 z 值，则返回 NaN
            return np.nan

    def set_facecolor(self, colors):
        # 继承的文档字符串
        super().set_facecolor(colors)
        # 将设置的面颜色赋值给 _facecolor3d
        self._facecolor3d = PolyCollection.get_facecolor(self)
    # 继承的文档字符串说明，设置3D图形对象的边缘颜色
    def set_edgecolor(self, colors):
        # 调用父类方法设置边缘颜色
        super().set_edgecolor(colors)
        # 获取3D图形对象的边缘颜色集合
        self._edgecolor3d = PolyCollection.get_edgecolor(self)

    # 继承的文档字符串说明，设置3D图形对象的透明度
    def set_alpha(self, alpha):
        # 调用父类方法设置透明度
        artist.Artist.set_alpha(self, alpha)
        # 尝试将3D图形对象的面颜色转换为RGBA数组
        try:
            self._facecolor3d = mcolors.to_rgba_array(
                self._facecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        # 尝试将3D图形对象的边缘颜色转换为RGBA数组
        try:
            self._edgecolors = mcolors.to_rgba_array(
                    self._edgecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        # 将对象标记为过时状态
        self.stale = True

    # 继承的文档字符串说明，获取3D图形对象的面颜色
    def get_facecolor(self):
        # 如果没有初始化self._facecolors2d，执行三维投影处理
        if not hasattr(self, '_facecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        # 返回面颜色的NumPy数组表示
        return np.asarray(self._facecolors2d)

    # 继承的文档字符串说明，获取3D图形对象的边缘颜色
    def get_edgecolor(self):
        # 如果没有初始化self._edgecolors2d，执行三维投影处理
        if not hasattr(self, '_edgecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        # 返回边缘颜色的NumPy数组表示
        return np.asarray(self._edgecolors2d)
# 将二维的 PolyCollection 对象转换为三维的 Poly3DCollection 对象
def poly_collection_2d_to_3d(col, zs=0, zdir='z'):
    """
    Convert a `.PolyCollection` into a `.Poly3DCollection` object.

    Parameters
    ----------
    col : `~matplotlib.collections.PolyCollection`
        The collection to convert.
    zs : float or array of floats
        The location or locations to place the polygons in the collection along
        the *zdir* axis. Default: 0.
    zdir : {'x', 'y', 'z'}
        The axis in which to place the patches. Default: 'z'.
        See `.get_dir_vector` for a description of the values.
    """
    # 调用内部函数将二维路径转换为三维路径段和代码
    segments_3d, codes = _paths_to_3d_segments_with_codes(
            col.get_paths(), zs, zdir)
    # 将对象的类更改为 Poly3DCollection
    col.__class__ = Poly3DCollection
    # 设置对象的顶点和代码
    col.set_verts_and_codes(segments_3d, codes)
    # 设置对象的三维属性
    col.set_3d_properties()


# 根据 zdir 参数重新排序坐标，使得 2D 的 xs 和 ys 可以在垂直于 zdir 的平面上绘制
def juggle_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that 2D *xs*, *ys* can be plotted in the plane
    orthogonal to *zdir*. *zdir* is normally 'x', 'y' or 'z'. However, if
    *zdir* starts with a '-' it is interpreted as a compensation for
    `rotate_axes`.
    """
    if zdir == 'x':
        return zs, xs, ys
    elif zdir == 'y':
        return xs, zs, ys
    elif zdir[0] == '-':
        return rotate_axes(xs, ys, zs, zdir)
    else:
        return xs, ys, zs


# 根据 zdir 参数旋转坐标轴
def rotate_axes(xs, ys, zs, zdir):
    """
    Reorder coordinates so that the axes are rotated with *zdir* along
    the original z axis. Prepending the axis with a '-' does the
    inverse transform, so *zdir* can be 'x', '-x', 'y', '-y', 'z' or '-z'.
    """
    if zdir in ('x', '-y'):
        return ys, zs, xs
    elif zdir in ('-x', 'y'):
        return zs, xs, ys
    else:
        return xs, ys, zs


# 根据深度调整颜色列表中的透明度
def _zalpha(colors, zs):
    """Modify the alphas of the color list according to depth."""
    # FIXME: This only works well if the points for *zs* are well-spaced
    #        in all three dimensions. Otherwise, at certain orientations,
    #        the min and max zs are very close together.
    #        Should really normalize against the viewing depth.
    if len(colors) == 0 or len(zs) == 0:
        return np.zeros((0, 4))
    norm = Normalize(min(zs), max(zs))
    sats = 1 - norm(zs) * 0.7
    rgba = np.broadcast_to(mcolors.to_rgba_array(colors), (len(zs), 4))
    return np.column_stack([rgba[:, :3], rgba[:, 3] * sats])


# 检查所有点是否在同一平面上
def _all_points_on_plane(xs, ys, zs, atol=1e-8):
    """
    Check if all points are on the same plane. Note that NaN values are
    ignored.

    Parameters
    ----------
    xs, ys, zs : array-like
        The x, y, and z coordinates of the points.
    atol : float, default: 1e-8
        The tolerance for the equality check.
    """
    xs, ys, zs = np.asarray(xs), np.asarray(ys), np.asarray(zs)
    points = np.column_stack([xs, ys, zs])
    points = points[~np.isnan(points).any(axis=1)]
    # 检查是否有少于 3 个唯一点的情况
    points = np.unique(points, axis=0)
    if len(points) <= 3:
        return True
    # 计算从第一个点到所有其他点的向量
    vs = (points - points[0])[1:]
    # 将所有向量标准化为单位向量
    vs = vs / np.linalg.norm(vs, axis=1)[:, np.newaxis]
    # 过滤掉平行的向量，保留唯一的向量
    vs = np.unique(vs, axis=0)
    # 如果向量数量不超过2个，则返回 True，表示所有点共线或者重合
    if len(vs) <= 2:
        return True
    # 计算第一个向量与其他向量的叉乘的模长
    cross_norms = np.linalg.norm(np.cross(vs[0], vs[1:]), axis=1)
    # 找到模长接近零的索引，它们表示与第一个向量平行或反平行的向量
    zero_cross_norms = np.where(np.isclose(cross_norms, 0, atol=atol))[0] + 1
    # 删除与第一个向量平行或反平行的向量
    vs = np.delete(vs, zero_cross_norms, axis=0)
    # 如果向量数量不超过2个，则返回 True，表示所有点共线或者重合
    if len(vs) <= 2:
        return True
    # 计算通过前三个点计算的法向量
    n = np.cross(vs[0], vs[1])
    # 将法向量标准化为单位向量
    n = n / np.linalg.norm(n)
    # 计算法向量与所有其他向量的点积，如果全部为零，则表示所有点在同一个平面上
    dots = np.dot(n, vs.transpose())
    # 检查点积是否在给定的误差范围内全部为零
    return np.allclose(dots, 0, atol=atol)
def _generate_normals(polygons):
    """
    计算多边形列表的法线向量，每个多边形一个法线向量。

    法线向量指向观察者，对于顶点按逆时针顺序排列的面，遵循右手规则。

    使用多边形周围等间距的三个点来计算法线向量。该方法假设点在一个平面上。否则，需要多个阴影，但此处不支持。

    Parameters
    ----------
    polygons : list of (M_i, 3) array-like, or (..., M, 3) array-like
        需要计算法线向量的多边形序列，可以具有不同数量的顶点。如果多边形具有相同数量的顶点且为数组，则可以进行向量化计算。

    Returns
    -------
    normals : (..., 3) array
        估计的多边形法线向量。
    """
    if isinstance(polygons, np.ndarray):
        # 优化：所有多边形具有相同的顶点数，因此可以进行向量化计算
        n = polygons.shape[-2]
        i1, i2, i3 = 0, n//3, 2*n//3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        # 减法操作无法向量化，因为polygons是不规则的
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = 0, n//3, 2*n//3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]
    return np.cross(v1, v2)


def _shade_colors(color, normals, lightsource=None):
    """
    使用给定的法线向量 *normals* 和光源 *lightsource*（如果未给出则使用默认位置）对 *color* 进行阴影处理。
    *color* 也可以是与 *normals* 长度相同的数组。
    """
    if lightsource is None:
        # 为了向后兼容而选择的默认光源位置
        lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)

    with np.errstate(invalid="ignore"):
        shade = ((normals / np.linalg.norm(normals, axis=1, keepdims=True))
                 @ lightsource.direction)
    mask = ~np.isnan(shade)

    if mask.any():
        # 将点积转换为允许的阴影分数
        in_norm = mcolors.Normalize(-1, 1)
        out_norm = mcolors.Normalize(0.3, 1).inverse

        def norm(x):
            return out_norm(in_norm(x))

        shade[~mask] = 0

        color = mcolors.to_rgba_array(color)
        # color的形状应为 (M, 4)（其中M是面的数量）
        # shade的形状应为 (M,)
        # colors的最终形状应为 (M, 4)
        alpha = color[:, 3]
        colors = norm(shade)[:, np.newaxis] * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors
```