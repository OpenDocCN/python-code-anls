# `D:\src\scipysrc\matplotlib\lib\matplotlib\transforms.py`

```py
"""
Matplotlib includes a framework for arbitrary geometric
transformations that is used determine the final position of all
elements drawn on the canvas.

Transforms are composed into trees of `TransformNode` objects
whose actual value depends on their children.  When the contents of
children change, their parents are automatically invalidated.  The
next time an invalidated transform is accessed, it is recomputed to
reflect those changes.  This invalidation/caching approach prevents
unnecessary recomputations of transforms, and contributes to better
interactive performance.

For example, here is a graph of the transform tree used to plot data
to the graph:

.. image:: ../_static/transforms.png

The framework can be used for both affine and non-affine
transformations.  However, for speed, we want to use the backend
renderers to perform affine transformations whenever possible.
Therefore, it is possible to perform just the affine or non-affine
part of a transformation on a set of data.  The affine is always
assumed to occur after the non-affine.  For any transform::

  full transform == non-affine part + affine part

The backends are not expected to handle non-affine transformations
themselves.

See the tutorial :ref:`transforms_tutorial` for examples
of how to use transforms.
"""

# Note: There are a number of places in the code where we use `np.min` or
# `np.minimum` instead of the builtin `min`, and likewise for `max`.  This is
# done so that `nan`s are propagated, instead of being silently dropped.

import copy  # 导入用于深复制对象的模块
import functools  # 导入用于创建高阶函数的模块
import textwrap  # 导入用于格式化文本的模块
import weakref  # 导入用于创建弱引用对象的模块
import math  # 导入数学函数模块

import numpy as np  # 导入 NumPy 库，并使用 np 作为别名
from numpy.linalg import inv  # 从 NumPy 的线性代数模块中导入 inv 函数

from matplotlib import _api  # 导入 Matplotlib 的私有 API
from matplotlib._path import (  # 从 Matplotlib 的 _path 模块中导入以下函数
    affine_transform, count_bboxes_overlapping_bbox, update_path_extents)
from .path import Path  # 从当前包中导入 Path 类

DEBUG = False  # 设置调试标志为 False


def _make_str_method(*args, **kwargs):
    """
    Generate a ``__str__`` method for a `.Transform` subclass.

    After ::

        class T:
            __str__ = _make_str_method("attr", key="other")

    ``str(T(...))`` will be

    .. code-block:: text

        {type(T).__name__}(
            {self.attr},
            key={self.other})
    """
    indent = functools.partial(textwrap.indent, prefix=" " * 4)  # 定义缩进函数
    def strrepr(x): return repr(x) if isinstance(x, str) else str(x)  # 定义转换为字符串函数
    return lambda self: (
        type(self).__name__ + "("
        + ",".join([*(indent("\n" + strrepr(getattr(self, arg)))
                      for arg in args),
                    *(indent("\n" + k + "=" + strrepr(getattr(self, arg)))
                      for k, arg in kwargs.items())])
        + ")")


class TransformNode:
    """
    The base class for anything that participates in the transform tree
    and needs to invalidate its parents or be invalidated.  This includes
    classes that are not really transforms, such as bounding boxes, since some
    transforms depend on bounding boxes to compute their values.
    """
    # INVALID_NON_AFFINE, INVALID_AFFINE, and INVALID are deprecated class properties,
    # each returning a unique integer value, signaling different types of invalidation.
    INVALID_NON_AFFINE = _api.deprecated("3.8")(_api.classproperty(lambda cls: 1))
    INVALID_AFFINE = _api.deprecated("3.8")(_api.classproperty(lambda cls: 2))
    INVALID = _api.deprecated("3.8")(_api.classproperty(lambda cls: 3))

    # Possible states for the _invalid attribute, representing validity and types of invalidation.
    _VALID, _INVALID_AFFINE_ONLY, _INVALID_FULL = range(3)

    # Metadata flags about the nature of the transform.
    is_affine = False
    is_bbox = _api.deprecated("3.9")(_api.classproperty(lambda cls: False))

    pass_through = False
    """
    If pass_through is True, all ancestors will always be
    invalidated, even if 'self' is already invalid.
    """

    def __init__(self, shorthand_name=None):
        """
        Constructor for TransformNode.

        Parameters
        ----------
        shorthand_name : str
            A string representing the "name" of the transform.
            Used for better readability in debug mode.
        """
        self._parents = {}  # Dictionary to hold weak references to parent nodes.
        self._invalid = self._INVALID_FULL  # Initial invalidation state is set to fully invalid.
        self._shorthand_name = shorthand_name or ''  # Initialize shorthand name.

    if DEBUG:
        def __str__(self):
            """
            Debug string representation of the TransformNode.

            Returns
            -------
            str
                Returns the shorthand name if available, otherwise the object's standard representation.
            """
            return self._shorthand_name or repr(self)

    def __getstate__(self):
        """
        Serialize the object's state for pickling.

        Returns
        -------
        dict
            Returns a dictionary containing the object's state,
            with weak references in '_parents' converted to normal references.
        """
        return {**self.__dict__,
                '_parents': {k: v() for k, v in self._parents.items()}}

    def __setstate__(self, data_dict):
        """
        Deserialize the object's state from a dictionary.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the object's state to restore.

        Notes
        -----
        Converts normal dictionary '_parents' entries back into weak references.
        """
        self.__dict__ = data_dict
        self._parents = {
            k: weakref.ref(v, lambda _, pop=self._parents.pop, k=k: pop(k))
            for k, v in self._parents.items() if v is not None}

    def __copy__(self):
        """
        Create a shallow copy of the TransformNode instance.

        Returns
        -------
        TransformNode
            Returns a new instance that is a shallow copy of the current instance.
        """
        other = copy.copy(super())
        other._parents = {}  # Clear parent references for the new copy.
        for key, val in vars(self).items():
            if isinstance(val, TransformNode) and id(self) in val._parents:
                other.set_children(val)  # Establish parent-child relationship for copied nodes.
        return other
    def invalidate(self):
        """
        Invalidate this `TransformNode` and triggers an invalidation of its
        ancestors.  Should be called any time the transform changes.
        """
        # 调用此方法将使当前的 `TransformNode` 失效，并且会使其祖先节点也失效。
        # 在每次变换发生时应该调用此方法。
        return self._invalidate_internal(
            level=self._INVALID_AFFINE_ONLY if self.is_affine else self._INVALID_FULL,
            invalidating_node=self)

    def _invalidate_internal(self, level, invalidating_node):
        """
        Called by :meth:`invalidate` and subsequently ascends the transform
        stack calling each TransformNode's _invalidate_internal method.
        """
        # 如果当前节点的失效级别已经超过传播的失效级别，则无需执行任何操作。
        if level <= self._invalid and not self.pass_through:
            return
        # 更新当前节点的失效级别
        self._invalid = level
        # 遍历当前节点的所有父节点
        for parent in list(self._parents.values()):
            parent = parent()  # 解除弱引用
            if parent is not None:
                # 递归调用父节点的 _invalidate_internal 方法
                parent._invalidate_internal(level=level, invalidating_node=self)

    def set_children(self, *children):
        """
        Set the children of the transform, to let the invalidation
        system know which transforms can invalidate this transform.
        Should be called from the constructor of any transforms that
        depend on other transforms.
        """
        # 将当前节点设为指定子节点的父节点，并使用弱引用来避免过时节点保持存活状态。
        id_self = id(self)
        for child in children:
            # 使用弱引用来确保字典不会使过时的节点保持存活状态；
            # 回调函数删除字典条目，这比使用 WeakValueDictionary 性能更好。
            ref = weakref.ref(
                self, lambda _, pop=child._parents.pop, k=id_self: pop(k))
            child._parents[id_self] = ref

    def frozen(self):
        """
        Return a frozen copy of this transform node.  The frozen copy will not
        be updated when its children change.  Useful for storing a previously
        known state of a transform where ``copy.deepcopy()`` might normally be
        used.
        """
        # 返回当前变换节点的冻结副本。
        # 冻结副本在其子节点更改时不会更新。
        # 在需要存储已知变换状态时很有用，通常可以使用 `copy.deepcopy()`。
        return self
    def xmax(self):
        """The right edge of the bounding box."""
        return np.max(self.get_points()[:, 0])

    @property
    def ymax(self):
        """The top edge of the bounding box."""
        return np.max(self.get_points()[:, 1])

    @property
    def width(self):
        """The width of the bounding box."""
        return self.x1 - self.x0

    @property
    def height(self):
        """The height of the bounding box."""
        return self.y1 - self.y0

    def get_points(self):
        """Return the two points defining the bounding box."""
        raise NotImplementedError("Derived must override")

    @staticmethod
    def from_extents(*args):
        """Create a BboxBase from *args* (left, bottom, right, top)."""
        if len(args) == 1:
            points, = args
            if isinstance(points, BboxBase):
                return points
            if not np.iterable(points):
                raise ValueError("BboxBase.from_extents() argument must be iterable")
            points = np.asarray(points)
            if points.shape != (4,):
                raise ValueError("BboxBase.from_extents() requires a 4 element array-like")
        else:
            points = np.array(args, dtype=float)
            if points.shape != (4,):
                raise ValueError("BboxBase.from_extents() requires a 4 element array-like")
        return BboxBase(points)

    def __init__(self, points):
        """Initialize the BboxBase instance with the given points."""
        if points.shape != (2, 2):
            raise ValueError("BboxBase points must be of shape (2, 2)")
        self._points = points

    def __repr__(self):
        """Return a string representation of the BboxBase object."""
        return f'{self.__class__.__name__}({self._points})'

    def __eq__(self, other):
        """Check equality with another BboxBase object."""
        if isinstance(other, self.__class__):
            return np.allclose(self._points, other._points)
        return NotImplemented

    def __ne__(self, other):
        """Check inequality with another BboxBase object."""
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return not equal

    def __hash__(self):
        """Return the hash value of the BboxBase object."""
        return hash(tuple(self._points.ravel()))

    def __bool__(self):
        """Return True if the bounding box is valid (width > 0 and height > 0)."""
        return self.width > 0 and self.height > 0
    # 返回边界框的右边缘 x 坐标的最大值
    def xmax(self):
        return np.max(self.get_points()[:, 0])

    @property
    # 返回边界框的顶边缘 y 坐标的最大值
    def ymax(self):
        return np.max(self.get_points()[:, 1])

    @property
    # 返回边界框的左下角坐标
    def min(self):
        return np.min(self.get_points(), axis=0)

    @property
    # 返回边界框的右上角坐标
    def max(self):
        return np.max(self.get_points(), axis=0)

    @property
    # 返回定义边界框的一对 x 坐标，不保证从左到右排序
    def intervalx(self):
        return self.get_points()[:, 0]

    @property
    # 返回定义边界框的一对 y 坐标，不保证从下到上排序
    def intervaly(self):
        return self.get_points()[:, 1]

    @property
    # 返回边界框的有符号宽度
    def width(self):
        points = self.get_points()
        return points[1, 0] - points[0, 0]

    @property
    # 返回边界框的有符号高度
    def height(self):
        points = self.get_points()
        return points[1, 1] - points[0, 1]

    @property
    # 返回边界框的有符号宽度和高度
    def size(self):
        points = self.get_points()
        return points[1] - points[0]

    @property
    # 返回边界框的边界参数 (x0, y0, width, height)
    def bounds(self):
        (x0, y0), (x1, y1) = self.get_points()
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    # 返回边界框的范围参数 (x0, y0, x1, y1)
    def extents(self):
        # flatten 返回一个复制的数组副本
        return self.get_points().flatten()

    # 抽象方法，应当在子类中实现
    def get_points(self):
        raise NotImplementedError

    # 检查 x 是否在闭区间 (x0, x1) 内
    def containsx(self, x):
        x0, x1 = self.intervalx
        return x0 <= x <= x1 or x0 >= x >= x1

    # 检查 y 是否在闭区间 (y0, y1) 内
    def containsy(self, y):
        y0, y1 = self.intervaly
        return y0 <= y <= y1 or y0 >= y >= y1

    # 检查点 (x, y) 是否在边界框内或者在其边缘上
    def contains(self, x, y):
        return self.containsx(x) and self.containsy(y)
    def overlaps(self, other):
        """
        Return whether this bounding box overlaps with the other bounding box.

        Parameters
        ----------
        other : `.BboxBase`
            Another bounding box object to compare against.
        """
        # Extract extents (coordinates) of both bounding boxes
        ax1, ay1, ax2, ay2 = self.extents
        bx1, by1, bx2, by2 = other.extents
        
        # Ensure ax1 < ax2 and ay1 < ay2 by swapping if necessary
        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2
        
        # Check for overlap between the bounding boxes
        return ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2

    def fully_containsx(self, x):
        """
        Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.

        Parameters
        ----------
        x : float
            The x-coordinate to check against the bounding box interval.
        """
        # Extract the x-coordinate interval of the bounding box
        x0, x1 = self.intervalx
        
        # Check if x lies strictly inside the interval (x0, x1)
        return x0 < x < x1 or x0 > x > x1

    def fully_containsy(self, y):
        """
        Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.

        Parameters
        ----------
        y : float
            The y-coordinate to check against the bounding box interval.
        """
        # Extract the y-coordinate interval of the bounding box
        y0, y1 = self.intervaly
        
        # Check if y lies strictly inside the interval (y0, y1)
        return y0 < y < y1 or y0 > y > y1

    def fully_contains(self, x, y):
        """
        Return whether ``x, y`` is in the bounding box, but not on its edge.

        Parameters
        ----------
        x : float
            The x-coordinate to check against the bounding box.
        y : float
            The y-coordinate to check against the bounding box.
        """
        # Check if both x and y are fully contained within the bounding box
        return self.fully_containsx(x) and self.fully_containsy(y)

    def fully_overlaps(self, other):
        """
        Return whether this bounding box overlaps with the other bounding box,
        not including the edges.

        Parameters
        ----------
        other : `.BboxBase`
            Another bounding box object to compare against.
        """
        # Extract extents (coordinates) of both bounding boxes
        ax1, ay1, ax2, ay2 = self.extents
        bx1, by1, bx2, by2 = other.extents
        
        # Ensure ax1 < ax2 and ay1 < ay2 by swapping if necessary
        if ax2 < ax1:
            ax2, ax1 = ax1, ax2
        if ay2 < ay1:
            ay2, ay1 = ay1, ay2
        if bx2 < bx1:
            bx2, bx1 = bx1, bx2
        if by2 < by1:
            by2, by1 = by1, by2
        
        # Check for overlap between the bounding boxes without including edges
        return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2

    def transformed(self, transform):
        """
        Construct a `Bbox` by statically transforming this one by *transform*.

        Parameters
        ----------
        transform : callable
            A transformation function to apply to the bounding box.
        """
        # Get the points defining the current bounding box
        pts = self.get_points()
        
        # Apply transformation to three key points and construct a new bounding box
        ll, ul, lr = transform.transform(np.array(
            [pts[0], [pts[0, 0], pts[1, 1]], [pts[1, 0], pts[0, 1]]]))
        
        # Return a new bounding box based on the transformed points
        return Bbox([ll, [lr[0], ul[1]]])

    coefs = {'C':  (0.5, 0.5),
             'SW': (0, 0),
             'S':  (0.5, 0),
             'SE': (1.0, 0),
             'E':  (1.0, 0.5),
             'NE': (1.0, 1.0),
             'N':  (0.5, 1.0),
             'NW': (0, 1.0),
             'W':  (0, 0.5)}
    def anchored(self, c, container=None):
        """
        Return a copy of the `Bbox` anchored to *c* within *container*.

        Parameters
        ----------
        c : (float, float) or {'C', 'SW', 'S', 'SE', 'E', 'NE', ...}
            Either an (*x*, *y*) pair of relative coordinates (0 is left or
            bottom, 1 is right or top), 'C' (center), or a cardinal direction
            ('SW', southwest, is bottom left, etc.).
        container : `Bbox`, optional
            The box within which the `Bbox` is positioned.

        See Also
        --------
        .Axes.set_anchor
        """
        if container is None:
            # 如果未提供容器参数，发出弃用警告并使用自身作为容器
            _api.warn_deprecated(
                "3.8", message="Calling anchored() with no container bbox "
                "returns a frozen copy of the original bbox and is deprecated "
                "since %(since)s.")
            container = self
        # 获取容器的左下角坐标 (l, b) 以及宽度 w 和高度 h
        l, b, w, h = container.bounds
        # 获取当前 `Bbox` 的左下角坐标 (L, B) 以及宽度 W 和高度 H
        L, B, W, H = self.bounds
        # 根据锚点 c 确定偏移量 cx, cy
        cx, cy = self.coefs[c] if isinstance(c, str) else c
        # 返回一个新的 `Bbox` 对象，根据计算得到的偏移量调整位置
        return Bbox(self._points +
                    [(l + cx * (w - W)) - L,
                     (b + cy * (h - H)) - B])

    def shrunk(self, mx, my):
        """
        Return a copy of the `Bbox`, shrunk by the factor *mx*
        in the *x* direction and the factor *my* in the *y* direction.
        The lower left corner of the box remains unchanged.  Normally
        *mx* and *my* will be less than 1, but this is not enforced.
        """
        # 获取当前 `Bbox` 的宽度 w 和高度 h
        w, h = self.size
        # 返回一个新的 `Bbox` 对象，根据缩放因子 mx 和 my 调整大小
        return Bbox([self._points[0],
                     self._points[0] + [mx * w, my * h]])

    def shrunk_to_aspect(self, box_aspect, container=None, fig_aspect=1.0):
        """
        Return a copy of the `Bbox`, shrunk so that it is as
        large as it can be while having the desired aspect ratio,
        *box_aspect*.  If the box coordinates are relative (i.e.
        fractions of a larger box such as a figure) then the
        physical aspect ratio of that figure is specified with
        *fig_aspect*, so that *box_aspect* can also be given as a
        ratio of the absolute dimensions, not the relative dimensions.
        """
        # 检查 box_aspect 和 fig_aspect 是否为正数，否则抛出 ValueError
        if box_aspect <= 0 or fig_aspect <= 0:
            raise ValueError("'box_aspect' and 'fig_aspect' must be positive")
        # 如果未提供容器参数，使用当前对象自身作为容器
        if container is None:
            container = self
        # 获取容器的宽度 w 和高度 h
        w, h = container.size
        # 根据给定的 aspect ratio 计算出新的高度 H
        H = w * box_aspect / fig_aspect
        # 根据新计算的高度 H 决定新的宽度 W
        if H <= h:
            W = w
        else:
            W = h * fig_aspect / box_aspect
            H = h
        # 返回一个新的 `Bbox` 对象，调整大小以满足指定的 aspect ratio
        return Bbox([self._points[0],
                     self._points[0] + (W, H)])
    def splitx(self, *args):
        """
        Return a list of new `Bbox` objects formed by splitting the original
        one with vertical lines at fractional positions given by *args*.
        """
        # Create fractions for splitting along the x-axis
        xf = [0, *args, 1]
        # Extract coordinates of the bounding box
        x0, y0, x1, y1 = self.extents
        # Calculate the width of the bounding box
        w = x1 - x0
        # Generate a list of new Bbox objects by splitting at specified fractions
        return [Bbox([[x0 + xf0 * w, y0], [x0 + xf1 * w, y1]])
                for xf0, xf1 in zip(xf[:-1], xf[1:])]

    def splity(self, *args):
        """
        Return a list of new `Bbox` objects formed by splitting the original
        one with horizontal lines at fractional positions given by *args*.
        """
        # Create fractions for splitting along the y-axis
        yf = [0, *args, 1]
        # Extract coordinates of the bounding box
        x0, y0, x1, y1 = self.extents
        # Calculate the height of the bounding box
        h = y1 - y0
        # Generate a list of new Bbox objects by splitting at specified fractions
        return [Bbox([[x0, y0 + yf0 * h], [x1, y0 + yf1 * h]])
                for yf0, yf1 in zip(yf[:-1], yf[1:])]

    def count_contains(self, vertices):
        """
        Count the number of vertices contained in the `Bbox`.
        Any vertices with a non-finite x or y value are ignored.

        Parameters
        ----------
        vertices : (N, 2) array
        """
        # Check if the vertices array is empty
        if len(vertices) == 0:
            return 0
        # Convert vertices to a numpy array
        vertices = np.asarray(vertices)
        # Ignore invalid operations (like comparison with NaN or Inf)
        with np.errstate(invalid='ignore'):
            # Count vertices that fall within the bounding box
            return (((self.min < vertices) &
                     (vertices < self.max)).all(axis=1).sum())

    def count_overlaps(self, bboxes):
        """
        Count the number of bounding boxes that overlap this one.

        Parameters
        ----------
        bboxes : sequence of `.BboxBase`
        """
        # Count overlaps with other bounding boxes using a helper function
        return count_bboxes_overlapping_bbox(
            self, np.atleast_3d([np.array(x) for x in bboxes]))

    def expanded(self, sw, sh):
        """
        Construct a `Bbox` by expanding this one around its center by the
        factors *sw* and *sh*.
        """
        # Calculate width and height of the current bounding box
        width = self.width
        height = self.height
        # Calculate deltas for expansion around the center
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        # Define a transformation matrix for expansion
        a = np.array([[-deltaw, -deltah], [deltaw, deltah]])
        # Return a new Bbox expanded around its center
        return Bbox(self._points + a)

    @_api.rename_parameter("3.8", "p", "w_pad")
    def padded(self, w_pad, h_pad=None):
        """
        Construct a `Bbox` by padding this one on all four sides.

        Parameters
        ----------
        w_pad : float
            Width pad
        h_pad : float, optional
            Height pad.  Defaults to *w_pad*.

        """
        # Retrieve the current points defining the bounding box
        points = self.get_points()
        # If height padding is not specified, use width padding for both dimensions
        if h_pad is None:
            h_pad = w_pad
        # Return a new Bbox object padded on all sides
        return Bbox(points + [[-w_pad, -h_pad], [w_pad, h_pad]])

    def translated(self, tx, ty):
        """Construct a `Bbox` by translating this one by *tx* and *ty*."""
        # Translate the current bounding box by the given amounts
        return Bbox(self._points + (tx, ty))
    def corners(self):
        """
        Return the corners of this rectangle as an array of points.

        Specifically, this returns the array
        ``[[x0, y0], [x0, y1], [x1, y0], [x1, y1]]``.
        """
        # 获取当前矩形的两个角点坐标 (x0, y0) 和 (x1, y1)
        (x0, y0), (x1, y1) = self.get_points()
        # 将角点坐标组成一个 NumPy 数组并返回
        return np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])

    def rotated(self, radians):
        """
        Return the axes-aligned bounding box that bounds the result of rotating
        this `Bbox` by an angle of *radians*.
        """
        # 获取当前矩形的角点坐标
        corners = self.corners()
        # 创建一个 Affine2D 变换对象，对角点进行旋转变换
        corners_rotated = Affine2D().rotate(radians).transform(corners)
        # 创建一个单位矩形 Bbox 对象
        bbox = Bbox.unit()
        # 根据旋转后的角点更新 Bbox 对象
        bbox.update_from_data_xy(corners_rotated, ignore=True)
        # 返回更新后的 Bbox 对象
        return bbox

    @staticmethod
    def union(bboxes):
        """Return a `Bbox` that contains all of the given *bboxes*."""
        # 如果输入列表为空，则抛出 ValueError 异常
        if not len(bboxes):
            raise ValueError("'bboxes' cannot be empty")
        # 计算所有输入 Bbox 的最小 x 值和最大 x 值，以及最小 y 值和最大 y 值
        x0 = np.min([bbox.xmin for bbox in bboxes])
        x1 = np.max([bbox.xmax for bbox in bboxes])
        y0 = np.min([bbox.ymin for bbox in bboxes])
        y1 = np.max([bbox.ymax for bbox in bboxes])
        # 创建一个包含所有输入 Bbox 区域的新 Bbox 对象并返回
        return Bbox([[x0, y0], [x1, y1]])

    @staticmethod
    def intersection(bbox1, bbox2):
        """
        Return the intersection of *bbox1* and *bbox2* if they intersect, or
        None if they don't.
        """
        # 计算两个 Bbox 的交集的边界
        x0 = np.maximum(bbox1.xmin, bbox2.xmin)
        x1 = np.minimum(bbox1.xmax, bbox2.xmax)
        y0 = np.maximum(bbox1.ymin, bbox2.ymin)
        y1 = np.minimum(bbox1.ymax, bbox2.ymax)
        # 如果交集存在（即边界合理），则返回一个包含交集区域的新 Bbox 对象，否则返回 None
        return Bbox([[x0, y0], [x1, y1]]) if x0 <= x1 and y0 <= y1 else None
# 默认的最小位置向量，初始化为无穷大的numpy数组
_default_minpos = np.array([np.inf, np.inf])

# 定义一个Bbox类，继承自BboxBase类
class Bbox(BboxBase):
    """
    可变的边界框类。

    示例
    --------
    **从已知边界创建**

    默认构造函数接受边界点“points” ``[[xmin, ymin], [xmax, ymax]]``。

        >>> Bbox([[1, 1], [3, 7]])
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    或者，可以从扁平化的点数组，即所谓的“extents” ``(xmin, ymin, xmax, ymax)`` 创建Bbox。

        >>> Bbox.from_extents(1, 1, 3, 7)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    或者从“bounds” ``(xmin, ymin, width, height)`` 创建Bbox。

        >>> Bbox.from_bounds(1, 1, 2, 6)
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    **从点集合创建**

    用于累积Bbox的“空”对象是空bbox，它是空集合的替代。

        >>> Bbox.null()
        Bbox([[inf, inf], [-inf, -inf]])

    将点添加到空bbox将给出这些点的bbox。

        >>> box = Bbox.null()
        >>> box.update_from_data_xy([[1, 1]])
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])
        >>> box.update_from_data_xy([[2, 3], [3, 2]], ignore=False)
        >>> box
        Bbox([[1.0, 1.0], [3.0, 3.0]])

    设置 ``ignore=True`` 等效于从空bbox重新开始。

        >>> box.update_from_data_xy([[1, 1]], ignore=True)
        >>> box
        Bbox([[1.0, 1.0], [1.0, 1.0]])

    .. warning::

        建议始终明确指定 ``ignore``。如果不指定，可以随时由访问您的Bbox的代码更改 ``ignore`` 的默认值，
        例如使用方法 `~.Bbox.ignore`。

    **空bbox的属性**

    .. note::

        `Bbox.null()` 的当前行为可能会让人感到意外，因为它不具备“空集合”的所有属性，因此在数学意义上不像“零”对象。
        在未来我们可能会改变这一点（带有废弃期）。

    空bbox是交集的身份元素

        >>> Bbox.intersection(Bbox([[1, 1], [3, 7]]), Bbox.null())
        Bbox([[1.0, 1.0], [3.0, 7.0]])

    但是与自身的交集将返回整个空间。

        >>> Bbox.intersection(Bbox.null(), Bbox.null())
        Bbox([[-inf, -inf], [inf, inf]])

    包含空bbox的并集将始终返回整个空间（而不是其他集合！）

        >>> Bbox.union([Bbox([[0, 0], [0, 0]]), Bbox.null()])
        Bbox([[-inf, -inf], [inf, inf]])
    """
    def __init__(self, points, **kwargs):
        """
        Parameters
        ----------
        points : `~numpy.ndarray`
            A (2, 2) array of the form ``[[x0, y0], [x1, y1]]``.
        """
        # 调用父类的初始化方法，传入额外的关键字参数
        super().__init__(**kwargs)
        # 将传入的points参数转换为浮点型的NumPy数组
        points = np.asarray(points, float)
        # 检查points数组的形状是否为(2, 2)，如果不是则抛出值错误异常
        if points.shape != (2, 2):
            raise ValueError('Bbox points must be of the form '
                             '"[[x0, y0], [x1, y1]]".')
        # 将points数组赋值给对象的_points属性
        self._points = points
        # 将默认的最小位置值复制给对象的_minpos属性
        self._minpos = _default_minpos.copy()
        # 将_ignore属性设置为True
        self._ignore = True
        # 在某些上下文中，了解bbox是否是默认值或已经被修改对帮助很有用；
        # 存储原始的points以支持修改后的方法
        self._points_orig = self._points.copy()

    if DEBUG:
        ___init__ = __init__

        def __init__(self, points, **kwargs):
            # 调用_check方法检查points参数
            self._check(points)
            # 调用___init__方法初始化对象
            self.___init__(points, **kwargs)

        def invalidate(self):
            # 再次检查_points属性
            self._check(self._points)
            # 调用父类的invalidate方法
            super().invalidate()

    def frozen(self):
        # 继承自父类的文档字符串
        # 调用父类的frozen方法并赋值给frozen_bbox变量
        frozen_bbox = super().frozen()
        # 将当前对象的_minpos属性的副本赋值给frozen_bbox的_minpos属性
        frozen_bbox._minpos = self.minpos.copy()
        # 返回修改后的frozen_bbox对象
        return frozen_bbox

    @staticmethod
    def unit():
        """Create a new unit `Bbox` from (0, 0) to (1, 1)."""
        # 返回一个从(0, 0)到(1, 1)的新单位Bbox对象
        return Bbox([[0, 0], [1, 1]])

    @staticmethod
    def null():
        """Create a new null `Bbox` from (inf, inf) to (-inf, -inf)."""
        # 返回一个从(inf, inf)到(-inf, -inf)的新空Bbox对象
        return Bbox([[np.inf, np.inf], [-np.inf, -np.inf]])

    @staticmethod
    def from_bounds(x0, y0, width, height):
        """
        Create a new `Bbox` from *x0*, *y0*, *width* and *height*.

        *width* and *height* may be negative.
        """
        # 根据左下角点(x0, y0)和宽度width、高度height创建一个新的Bbox对象
        return Bbox.from_extents(x0, y0, x0 + width, y0 + height)

    @staticmethod
    def from_extents(*args, minpos=None):
        """
        Create a new Bbox from *left*, *bottom*, *right* and *top*.

        The *y*-axis increases upwards.

        Parameters
        ----------
        left, bottom, right, top : float
            The four extents of the bounding box.
        minpos : float or None
            If this is supplied, the Bbox will have a minimum positive value
            set. This is useful when dealing with logarithmic scales and other
            scales where negative bounds result in floating point errors.
        """
        # 将参数args重新整形为(2, 2)的NumPy数组，创建一个新的Bbox对象
        bbox = Bbox(np.reshape(args, (2, 2)))
        # 如果minpos不为None，则将其赋值给bbox对象的_minpos属性
        if minpos is not None:
            bbox._minpos[:] = minpos
        # 返回创建的bbox对象
        return bbox

    def __format__(self, fmt):
        # 返回格式化后的Bbox对象的字符串表示，使用给定的fmt格式
        return (
            'Bbox(x0={0.x0:{1}}, y0={0.y0:{1}}, x1={0.x1:{1}}, y1={0.y1:{1}})'.
            format(self, fmt))

    def __str__(self):
        # 返回Bbox对象的字符串表示，空字符串表示没有格式
        return format(self, '')

    def __repr__(self):
        # 返回Bbox对象的官方字符串表示，包含四个角点的坐标信息
        return 'Bbox([[{0.x0}, {0.y0}], [{0.x1}, {0.y1}]])'.format(self)
    def ignore(self, value):
        """
        设置是否忽略盒子的现有边界，以便在后续调用 :meth:`update_from_data_xy` 时使用。

        value : bool
            - 当为 ``True`` 时，后续调用 `update_from_data_xy` 将忽略 `Bbox` 的现有边界。
            - 当为 ``False`` 时，后续调用 `update_from_data_xy` 将包括 `Bbox` 的现有边界。
        """
        self._ignore = value

    def update_from_path(self, path, ignore=None, updatex=True, updatey=True):
        """
        根据提供的路径更新 `Bbox` 的边界框，以包含路径的顶点。更新后，边界框将具有正的 *width*
        和 *height*；*x0* 和 *y0* 将是最小值。

        Parameters
        ----------
        path : `~matplotlib.path.Path`
            路径对象，描述了一条路径。
        ignore : bool, optional
           - 当为 ``True`` 时，忽略 `Bbox` 的现有边界。
           - 当为 ``False`` 时，包括 `Bbox` 的现有边界。
           - 当为 ``None`` 时，使用上次传递给 :meth:`ignore` 的值。
        updatex, updatey : bool, 默认为 True
            当为 ``True`` 时，更新 x/y 值。
        """
        if ignore is None:
            ignore = self._ignore

        if path.vertices.size == 0:
            return

        points, minpos, changed = update_path_extents(
            path, None, self._points, self._minpos, ignore)

        if changed:
            self.invalidate()
            if updatex:
                self._points[:, 0] = points[:, 0]
                self._minpos[0] = minpos[0]
            if updatey:
                self._points[:, 1] = points[:, 1]
                self._minpos[1] = minpos[1]

    def update_from_data_x(self, x, ignore=None):
        """
        根据传入的数据更新 `Bbox` 的 x 边界。更新后，边界框将具有正的 *width*，
        *x0* 将是最小值。

        Parameters
        ----------
        x : `~numpy.ndarray`
            包含 x 值的数组。
        ignore : bool, optional
           - 当为 ``True`` 时，忽略 `Bbox` 的现有边界。
           - 当为 ``False`` 时，包括 `Bbox` 的现有边界。
           - 当为 ``None`` 时，使用上次传递给 :meth:`ignore` 的值。
        """
        x = np.ravel(x)
        self.update_from_data_xy(np.column_stack([x, np.ones(x.size)]),
                                 ignore=ignore, updatey=False)
    @BboxBase.x0.setter
    def x0(self, val):
        # 设置 bbox 左下角 x 值
        self._points[0, 0] = val
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.y0.setter
    def y0(self, val):
        # 设置 bbox 左下角 y 值
        self._points[0, 1] = val
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.x1.setter
    def x1(self, val):
        # 设置 bbox 右上角 x 值
        self._points[1, 0] = val
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.y1.setter
    def y1(self, val):
        # 设置 bbox 右上角 y 值
        self._points[1, 1] = val
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.p0.setter
    def p0(self, val):
        # 设置 bbox 左下角点坐标
        self._points[0] = val
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.p1.setter
    def p1(self, val):
        # 设置 bbox 右上角点坐标
        self._points[1] = val
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.intervalx.setter
    def intervalx(self, interval):
        # 设置 bbox 水平间隔
        self._points[:, 0] = interval
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.intervaly.setter
    def intervaly(self, interval):
        # 设置 bbox 垂直间隔
        self._points[:, 1] = interval
        # 使 bbox 失效，需要重新计算
        self.invalidate()

    @BboxBase.bounds.setter
    def bounds(self, bounds):
        # 设置 bbox 边界
        l, b, w, h = bounds
        points = np.array([[l, b], [l + w, b + h]], float)
        # 如果新边界和当前点不一致，则更新点，并使 bbox 失效
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    @property
    def width(self):
        # 计算 bbox 宽度
        return abs(self._points[1, 0] - self._points[0, 0])

    @property
    def height(self):
        # 计算 bbox 高度
        return abs(self._points[1, 1] - self._points[0, 1])
    def minpos(self):
        """
        返回 Bbox 中最小的正值。

        在处理对数刻度和其他可能导致浮点错误的刻度时非常有用，将用作最小范围的一部分，而不是 *p0*。
        """
        return self._minpos

    @minpos.setter
    def minpos(self, val):
        """
        设置 Bbox 中的最小正值。

        参数：
        val -- 要设置的值
        """
        self._minpos[:] = val

    @property
    def minposx(self):
        """
        返回 Bbox 中 *x* 方向上的最小正值。

        在处理对数刻度和其他可能导致浮点错误的刻度时非常有用，将用作最小 *x* 范围的一部分，而不是 *x0*。
        """
        return self._minpos[0]

    @minposx.setter
    def minposx(self, val):
        """
        设置 Bbox 中 *x* 方向上的最小正值。

        参数：
        val -- 要设置的值
        """
        self._minpos[0] = val

    @property
    def minposy(self):
        """
        返回 Bbox 中 *y* 方向上的最小正值。

        在处理对数刻度和其他可能导致浮点错误的刻度时非常有用，将用作最小 *y* 范围的一部分，而不是 *y0*。
        """
        return self._minpos[1]

    @minposy.setter
    def minposy(self, val):
        """
        设置 Bbox 中 *y* 方向上的最小正值。

        参数：
        val -- 要设置的值
        """
        self._minpos[1] = val

    def get_points(self):
        """
        返回 Bbox 的边界框点数组，格式为 ``[[x0, y0], [x1, y1]]``。
        """
        self._invalid = 0
        return self._points

    def set_points(self, points):
        """
        直接从形如 ``[[x0, y0], [x1, y1]]`` 的数组设置 Bbox 的边界框点。

        参数：
        points -- 要设置的点数组，没有进行错误检查，主要用于内部使用。
        """
        if np.any(self._points != points):
            self._points = points
            self.invalidate()

    def set(self, other):
        """
        从另一个 `Bbox` 的“冻结”边界设置此边界框。
        
        参数：
        other -- 另一个 `Bbox` 对象
        """
        if np.any(self._points != other.get_points()):
            self._points = other.get_points()
            self.invalidate()

    def mutated(self):
        """返回自初始化以来边界框是否发生变化。"""
        return self.mutatedx() or self.mutatedy()

    def mutatedx(self):
        """返回自初始化以来 *x* 范围是否发生变化。"""
        return (self._points[0, 0] != self._points_orig[0, 0] or
                self._points[1, 0] != self._points_orig[1, 0])

    def mutatedy(self):
        """返回自初始化以来 *y* 范围是否发生变化。"""
        return (self._points[0, 1] != self._points_orig[0, 1] or
                self._points[1, 1] != self._points_orig[1, 1])
class TransformedBbox(BboxBase):
    """
    A `Bbox` that is automatically transformed by a given
    transform.  When either the child bounding box or transform
    changes, the bounds of this bbox will update accordingly.
    """

    def __init__(self, bbox, transform, **kwargs):
        """
        Parameters
        ----------
        bbox : `Bbox`
            The original bounding box to be transformed.
        transform : `Transform`
            The transformation to be applied to the bounding box.
        """
        # Ensure bbox is an instance of BboxBase
        _api.check_isinstance(BboxBase, bbox=bbox)
        # Ensure transform is an instance of Transform
        _api.check_isinstance(Transform, transform=transform)
        # Check that transform operates in 2 dimensions
        if transform.input_dims != 2 or transform.output_dims != 2:
            raise ValueError(
                "The input and output dimensions of 'transform' must be 2")

        super().__init__(**kwargs)
        self._bbox = bbox  # Store the original bounding box
        self._transform = transform  # Store the transformation
        self.set_children(bbox, transform)  # Set bbox and transform as children
        self._points = None  # Initialize _points as None

    # Define __str__ method using _make_str_method for string representation
    __str__ = _make_str_method("_bbox", "_transform")

    def get_points(self):
        # docstring inherited
        if self._invalid:
            p = self._bbox.get_points()  # Get points of the original bbox
            # Transform all four points using self._transform
            points = self._transform.transform(
                [[p[0, 0], p[0, 1]],
                 [p[1, 0], p[0, 1]],
                 [p[0, 0], p[1, 1]],
                 [p[1, 0], p[1, 1]]])
            points = np.ma.filled(points, 0.0)  # Fill masked points with 0.0

            # Determine the minimum and maximum x, y coordinates
            xs = min(points[:, 0]), max(points[:, 0])
            if p[0, 0] > p[1, 0]:
                xs = xs[::-1]  # Reverse xs if necessary

            ys = min(points[:, 1]), max(points[:, 1])
            if p[0, 1] > p[1, 1]:
                ys = ys[::-1]  # Reverse ys if necessary

            # Create a new array of transformed points and store it in self._points
            self._points = np.array([
                [xs[0], ys[0]],
                [xs[1], ys[1]]
            ])

            self._invalid = 0  # Reset the invalid flag

        return self._points  # Return the transformed points

    if DEBUG:
        _get_points = get_points

        def get_points(self):
            # Execute the original get_points method
            points = self._get_points()
            # Check the validity of the points and update if necessary
            self._check(points)
            return points

    def contains(self, x, y):
        # Docstring inherited.
        # Invert the transformation, then check if (x, y) is within the original bbox
        return self._bbox.contains(*self._transform.inverted().transform((x, y)))

    def fully_contains(self, x, y):
        # Docstring inherited.
        # Invert the transformation, then check if (x, y) is fully within the original bbox
        return self._bbox.fully_contains(*self._transform.inverted().transform((x, y)))


class LockableBbox(BboxBase):
    """
    A `Bbox` where some elements may be locked at certain values.

    When the child bounding box changes, the bounds of this bbox will update
    accordingly with the exception of the locked elements.
    """
    def __init__(self, bbox, x0=None, y0=None, x1=None, y1=None, **kwargs):
        """
        Parameters
        ----------
        bbox : `Bbox`
            要包装的子边界框。

        x0 : float or None
            x0 的固定值，如果为 None 则表示未固定。

        y0 : float or None
            y0 的固定值，如果为 None 则表示未固定。

        x1 : float or None
            x1 的固定值，如果为 None 则表示未固定。

        y1 : float or None
            y1 的固定值，如果为 None 则表示未固定。
        """
        # 检查 bbox 是否为 BboxBase 实例
        _api.check_isinstance(BboxBase, bbox=bbox)
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 设置 _bbox 属性为传入的 bbox 对象
        self._bbox = bbox
        # 设置子元素为 bbox
        self.set_children(bbox)
        # 初始化 _points 属性为 None
        self._points = None
        # 创建长度为 4 的 fp 列表，包含 x0, y0, x1, y1
        fp = [x0, y0, x1, y1]
        # 创建长度为 4 的 mask 列表，标记哪些值为 None
        mask = [val is None for val in fp]
        # 使用 np.ma.array 创建具有浮点类型和 mask 的 _locked_points 数组，形状为 (2, 2)
        self._locked_points = np.ma.array(fp, float, mask=mask).reshape((2, 2))

    __str__ = _make_str_method("_bbox", "_locked_points")

    def get_points(self):
        """
        获取锚定点数组。

        Returns
        -------
        ndarray
            锚定点的 numpy 数组。
        """
        # 如果标记为无效，则重新计算点数组
        if self._invalid:
            # 获取 bbox 的顶点
            points = self._bbox.get_points()
            # 根据锁定的点和 bbox 的点生成最终的点数组
            self._points = np.where(self._locked_points.mask,
                                    points,
                                    self._locked_points)
            # 标记为有效
            self._invalid = 0
        # 返回计算得到的点数组
        return self._points

    if DEBUG:
        _get_points = get_points

        def get_points(self):
            """
            调试模式下，获取锚定点数组并检查其有效性。

            Returns
            -------
            ndarray
                锚定点的 numpy 数组。
            """
            # 调用 _get_points 方法获取点数组
            points = self._get_points()
            # 检查点数组的有效性
            self._check(points)
            # 返回检查后的点数组
            return points

    @property
    def locked_x0(self):
        """
        float or None: 用于锁定 x0 的值。
        """
        # 如果 _locked_points 中 x0 位置的 mask 为 True，则返回 None，否则返回对应的值
        if self._locked_points.mask[0, 0]:
            return None
        else:
            return self._locked_points[0, 0]

    @locked_x0.setter
    def locked_x0(self, x0):
        # 根据 x0 是否为 None，设置 _locked_points 中 x0 位置的 mask 和 data
        self._locked_points.mask[0, 0] = x0 is None
        self._locked_points.data[0, 0] = x0
        # 使对象无效，需要重新计算点数组
        self.invalidate()

    @property
    def locked_y0(self):
        """
        float or None: 用于锁定 y0 的值。
        """
        # 如果 _locked_points 中 y0 位置的 mask 为 True，则返回 None，否则返回对应的值
        if self._locked_points.mask[0, 1]:
            return None
        else:
            return self._locked_points[0, 1]

    @locked_y0.setter
    def locked_y0(self, y0):
        # 根据 y0 是否为 None，设置 _locked_points 中 y0 位置的 mask 和 data
        self._locked_points.mask[0, 1] = y0 is None
        self._locked_points.data[0, 1] = y0
        # 使对象无效，需要重新计算点数组
        self.invalidate()

    @property
    def locked_x1(self):
        """
        float or None: 用于锁定 x1 的值。
        """
        # 如果 _locked_points 中 x1 位置的 mask 为 True，则返回 None，否则返回对应的值
        if self._locked_points.mask[1, 0]:
            return None
        else:
            return self._locked_points[1, 0]

    @locked_x1.setter
    def locked_x1(self, x1):
        # 根据 x1 是否为 None，设置 _locked_points 中 x1 位置的 mask 和 data
        self._locked_points.mask[1, 0] = x1 is None
        self._locked_points.data[1, 0] = x1
        # 使对象无效，需要重新计算点数组
        self.invalidate()

    @property
    def locked_y1(self):
        """
        float or None: 用于锁定 y1 的值。
        """
        # 如果 _locked_points 中 y1 位置的 mask 为 True，则返回 None，否则返回对应的值
        if self._locked_points.mask[1, 1]:
            return None
        else:
            return self._locked_points[1, 1]
    # 定义属性的 setter 方法，用于设置 locked_y1 属性
    @locked_y1.setter
    def locked_y1(self, y1):
        # 更新 _locked_points 对象中索引为 (1, 1) 的元素，根据 y1 是否为 None 来设置其值
        self._locked_points.mask[1, 1] = y1 is None
        # 设置 _locked_points 对象中索引为 (1, 1) 的元素的值为 y1
        self._locked_points.data[1, 1] = y1
        # 调用 invalidate() 方法，使得对象状态无效化，可能需要重新计算或重新绘制
        self.invalidate()
class Transform(TransformNode):
    """
    The base class of all `TransformNode` instances that
    actually perform a transformation.

    All non-affine transformations should be subclasses of this class.
    New affine transformations should be subclasses of `Affine2D`.

    Subclasses of this class should override the following members (at
    minimum):

    - :attr:`input_dims`
    - :attr:`output_dims`
    - :meth:`transform`
    - :meth:`inverted` (if an inverse exists)

    The following attributes may be overridden if the default is unsuitable:

    - :attr:`is_separable` (defaults to True for 1D -> 1D transforms, False
      otherwise)
    - :attr:`has_inverse` (defaults to True if :meth:`inverted` is overridden,
      False otherwise)

    If the transform needs to do something non-standard with
    `matplotlib.path.Path` objects, such as adding curves
    where there were once line segments, it should override:

    - :meth:`transform_path`
    """

    input_dims = None
    """
    The number of input dimensions of this transform.
    Must be overridden (with integers) in the subclass.
    """

    output_dims = None
    """
    The number of output dimensions of this transform.
    Must be overridden (with integers) in the subclass.
    """

    is_separable = False
    """True if this transform is separable in the x- and y- dimensions."""

    has_inverse = False
    """True if this transform has a corresponding inverse transform."""

    def __init_subclass__(cls):
        """
        Hook called whenever a subclass of `Transform` is created.
        
        Checks and sets the `is_separable` and `has_inverse` attributes
        based on the subclass's properties and methods.
        """
        # 1d transforms are always separable; we assume higher-dimensional ones
        # are not but subclasses can also directly set is_separable -- this is
        # verified by checking whether "is_separable" appears more than once in
        # the class's MRO (it appears once in Transform).
        if (sum("is_separable" in vars(parent) for parent in cls.__mro__) == 1
                and cls.input_dims == cls.output_dims == 1):
            cls.is_separable = True
        # Transform.inverted raises NotImplementedError; we assume that if this
        # is overridden then the transform is invertible but subclass can also
        # directly set has_inverse.
        if (sum("has_inverse" in vars(parent) for parent in cls.__mro__) == 1
                and hasattr(cls, "inverted")
                and cls.inverted is not Transform.inverted):
            cls.has_inverse = True

    def __add__(self, other):
        """
        Compose two transforms together so that *self* is followed by *other*.

        ``A + B`` returns a transform ``C`` so that
        ``C.transform(x) == B.transform(A.transform(x))``.
        """
        return (composite_transform_factory(self, other)
                if isinstance(other, Transform) else
                NotImplemented)

    # Equality is based on object identity for `Transform`s (so we don't
    # override `__eq__`), but some subclasses, such as TransformWrapper &
    # AffineBase, override this behavior.
    def _iter_break_from_left_to_right(self):
        """
        Return an iterator breaking down this transform stack from left to
        right recursively. If self == ((A, N), A), then the result will be an
        iterator which yields IdentityTransform() : ((A, N), A), followed by
        self : A, followed by (A, N) : A, but not ((A, N), A) : IdentityTransform().

        This is equivalent to flattening the stack then yielding
        ``flat_stack[:i], flat_stack[i:]`` where i=0..(n-1).
        """
        # Yield the initial tuple with IdentityTransform and self
        yield IdentityTransform(), self

    @property
    def depth(self):
        """
        Return the number of transforms which have been chained
        together to form this Transform instance.

        .. note::

            For the special case of a Composite transform, the maximum depth
            of the two is returned.

        """
        # Always returns 1 for this simple case
        return 1

    def contains_branch(self, other):
        """
        Return whether the given transform is a sub-tree of this transform.

        This routine uses transform equality to identify sub-trees, therefore
        in many situations it is object id which will be used.

        For the case where the given transform represents the whole
        of this transform, returns True.
        """
        # Check if other is a subtree of self by iterating through the transform stack
        if self.depth < other.depth:
            return False

        # Iterate through the transform stack and check for equality with other
        for _, sub_tree in self._iter_break_from_left_to_right():
            if sub_tree == other:
                return True
        return False

    def contains_branch_seperately(self, other_transform):
        """
        Return whether the given branch is a sub-tree of this transform on
        each separate dimension.

        A common use for this method is to identify if a transform is a blended
        transform containing an Axes' data transform. e.g.::

            x_isdata, y_isdata = trans.contains_branch_seperately(ax.transData)

        """
        # Ensure the output dimension is 2, otherwise raise an error
        if self.output_dims != 2:
            raise ValueError('contains_branch_seperately only supports '
                             'transforms with 2 output dimensions')
        # For non-blended transforms, each dimension separately checks if other_transform is a branch
        return [self.contains_branch(other_transform)] * 2
    def __sub__(self, other):
        """
        Compose *self* with the inverse of *other*, cancelling identical terms
        if any::

            # In general:
            A - B == A + B.inverted()
            # (but see note regarding frozen transforms below).

            # If A "ends with" B (i.e. A == A' + B for some A') we can cancel
            # out B:
            (A' + B) - B == A'

            # Likewise, if B "starts with" A (B = A + B'), we can cancel out A:
            A - (A + B') == B'.inverted() == B'^-1

        Cancellation (rather than naively returning ``A + B.inverted()``) is
        important for multiple reasons:

        - It avoids floating-point inaccuracies when computing the inverse of
          B: ``B - B`` is guaranteed to cancel out exactly (resulting in the
          identity transform), whereas ``B + B.inverted()`` may differ by a
          small epsilon.
        - ``B.inverted()`` always returns a frozen transform: if one computes
          ``A + B + B.inverted()`` and later mutates ``B``, then
          ``B.inverted()`` won't be updated and the last two terms won't cancel
          out anymore; on the other hand, ``A + B - B`` will always be equal to
          ``A`` even if ``B`` is mutated.
        """
        # 检查是否可以执行减法操作，要求other必须是Transform的实例
        if not isinstance(other, Transform):
            return NotImplemented
        
        # 从左到右遍历self的子树，尝试找到与other相等的子树并取消之
        for remainder, sub_tree in self._iter_break_from_left_to_right():
            if sub_tree == other:
                return remainder
        
        # 如果self中没有找到与other相等的子树，则从左到右遍历other的子树
        for remainder, sub_tree in other._iter_break_from_left_to_right():
            if sub_tree == self:
                # 如果找到self与other相等的子树，则返回remainder的逆变换
                if not remainder.has_inverse:
                    raise ValueError(
                        "The shortcut cannot be computed since 'other' "
                        "includes a non-invertible component")
                return remainder.inverted()
        
        # 如果无法通过简化找到相等的子树，则根据other是否可逆返回结果
        if other.has_inverse:
            return self + other.inverted()
        else:
            raise ValueError('It is not possible to compute transA - transB '
                             'since transB cannot be inverted and there is no '
                             'shortcut possible.')

    def __array__(self, *args, **kwargs):
        """Array interface to get at this Transform's affine matrix."""
        # 返回当前Transform对象的仿射矩阵作为数组接口的表示
        return self.get_affine().get_matrix()
    def transform(self, values):
        """
        Apply this transformation on the given array of *values*.

        Parameters
        ----------
        values : array-like
            The input values as an array of length :attr:`input_dims` or
            shape (N, :attr:`input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`output_dims` or
            shape (N, :attr:`output_dims`), depending on the input.
        """
        # 将输入的数据转换为 NumPy 数组，并记录其原始维度
        values = np.asanyarray(values)
        ndim = values.ndim
        # 将输入数据重塑为二维数组，行数为自动推断的 N，列数为 input_dims
        values = values.reshape((-1, self.input_dims))

        # 对输入数据进行仿射变换和非仿射变换
        res = self.transform_affine(self.transform_non_affine(values))

        # 将结果重塑回原始输入数据的形状
        if ndim == 0:
            assert not np.ma.is_masked(res)  # 为了安全起见
            return res[0, 0]
        if ndim == 1:
            return res.reshape(-1)
        elif ndim == 2:
            return res
        raise ValueError(
            "Input values must have shape (N, {dims}) or ({dims},)"
            .format(dims=self.input_dims))

    def transform_affine(self, values):
        """
        Apply only the affine part of this transformation on the
        given array of values.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally a no-op.  In
        affine transformations, this is equivalent to
        ``transform(values)``.

        Parameters
        ----------
        values : array
            The input values as an array of length :attr:`input_dims` or
            shape (N, :attr:`input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`output_dims` or
            shape (N, :attr:`output_dims`), depending on the input.
        """
        # 调用 get_affine 方法获取仿射变换对象，并对输入数据进行仿射变换
        return self.get_affine().transform(values)

    def transform_non_affine(self, values):
        """
        Apply only the non-affine part of this transformation.

        ``transform(values)`` is always equivalent to
        ``transform_affine(transform_non_affine(values))``.

        In non-affine transformations, this is generally equivalent to
        ``transform(values)``.  In affine transformations, this is
        always a no-op.

        Parameters
        ----------
        values : array
            The input values as an array of length :attr:`input_dims` or
            shape (N, :attr:`input_dims`).

        Returns
        -------
        array
            The output values as an array of length :attr:`output_dims` or
            shape (N, :attr:`output_dims`), depending on the input.
        """
        # 直接返回输入数据，表示非仿射变换没有实际操作
        return values
    # 将给定的边界框进行变换处理，并返回一个新的边界框对象
    def transform_bbox(self, bbox):
        """
        Transform the given bounding box.

        For smarter transforms including caching (a common requirement in
        Matplotlib), see `TransformedBbox`.
        """
        # 调用自身的 transform 方法对边界框的顶点进行变换，并用变换后的顶点创建新的 Bbox 对象
        return Bbox(self.transform(bbox.get_points()))

    # 获取此变换的仿射部分
    def get_affine(self):
        """Get the affine part of this transform."""
        # 返回一个标识仿射变换的 IdentityTransform 对象
        return IdentityTransform()

    # 获取此变换的仿射部分的矩阵
    def get_matrix(self):
        """Get the matrix for the affine part of this transform."""
        # 获取仿射部分的变换矩阵
        return self.get_affine().get_matrix()

    # 对给定的点进行变换，并返回变换后的点
    def transform_point(self, point):
        """
        Return a transformed point.

        This function is only kept for backcompatibility; the more general
        `.transform` method is capable of transforming both a list of points
        and a single point.

        The point is given as a sequence of length :attr:`input_dims`.
        The transformed point is returned as a sequence of length
        :attr:`output_dims`.
        """
        # 检查点的长度是否与输入维度相同
        if len(point) != self.input_dims:
            raise ValueError("The length of 'point' must be 'self.input_dims'")
        # 调用 transform 方法对点进行变换
        return self.transform(point)

    # 对给定的路径对象应用变换，并返回一个新的路径对象
    def transform_path(self, path):
        """
        Apply the transform to `.Path` *path*, returning a new `.Path`.

        In some cases, this transform may insert curves into the path
        that began as line segments.
        """
        # 首先对路径对象进行非仿射部分的变换，然后再对仿射部分进行变换，并返回新的路径对象
        return self.transform_path_affine(self.transform_path_non_affine(path))

    # 对给定的路径对象应用仿射变换，并返回一个新的路径对象
    def transform_path_affine(self, path):
        """
        Apply the affine part of this transform to `.Path` *path*, returning a
        new `.Path`.

        ``transform_path(path)`` is equivalent to
        ``transform_path_affine(transform_path_non_affine(values))``.
        """
        # 调用仿射部分的 transform_path_affine 方法对路径对象进行变换，并返回新的路径对象
        return self.get_affine().transform_path_affine(path)

    # 对给定的路径对象应用非仿射变换，并返回一个新的路径对象
    def transform_path_non_affine(self, path):
        """
        Apply the non-affine part of this transform to `.Path` *path*,
        returning a new `.Path`.

        ``transform_path(path)`` is equivalent to
        ``transform_path_affine(transform_path_non_affine(values))``.
        """
        # 调用自身的 transform_non_affine 方法对路径对象的顶点进行非仿射变换，并用结果创建新的路径对象
        x = self.transform_non_affine(path.vertices)
        return Path._fast_from_codes_and_verts(x, path.codes, path)
    def transform_angles(self, angles, pts, radians=False, pushoff=1e-5):
        """
        Transform a set of angles anchored at specific locations.

        Parameters
        ----------
        angles : (N,) array-like
            The angles to transform.
        pts : (N, 2) array-like
            The points where the angles are anchored.
        radians : bool, default: False
            Whether *angles* are radians or degrees.
        pushoff : float
            For each point in *pts* and angle in *angles*, the transformed
            angle is computed by transforming a segment of length *pushoff*
            starting at that point and making that angle relative to the
            horizontal axis, and measuring the angle between the horizontal
            axis and the transformed segment.

        Returns
        -------
        (N,) array
            Array of transformed angles.

        """
        # Must be 2D
        # 检查输入和输出维度是否为2，如果不是，则抛出异常
        if self.input_dims != 2 or self.output_dims != 2:
            raise NotImplementedError('Only defined in 2D')
        
        # 将输入的角度转换为 NumPy 数组
        angles = np.asarray(angles)
        # 将输入的点坐标转换为 NumPy 数组
        pts = np.asarray(pts)
        
        # 检查点坐标的形状是否为 (None, 2)
        _api.check_shape((None, 2), pts=pts)
        # 检查角度数组的形状是否为 (None,)
        _api.check_shape((None,), angles=angles)
        
        # 如果角度数组的长度不等于点坐标数组的长度，则抛出数值错误
        if len(angles) != len(pts):
            raise ValueError("There must be as many 'angles' as 'pts'")
        
        # 如果 radians 参数为 False，则将角度转换为弧度制
        if not radians:
            angles = np.deg2rad(angles)
        
        # 在每个点的基础上，沿着给定角度方向移动一个小距离 pushoff
        pts2 = pts + pushoff * np.column_stack([np.cos(angles),
                                                np.sin(angles)])
        
        # 对原始点集和新点集分别进行变换
        tpts = self.transform(pts)
        tpts2 = self.transform(pts2)
        
        # 计算变换后的角度
        d = tpts2 - tpts
        a = np.arctan2(d[:, 1], d[:, 0])
        
        # 如果 radians 参数为 False，则将角度转换回度数
        if not radians:
            a = np.rad2deg(a)
        
        # 返回计算得到的角度数组
        return a

    def inverted(self):
        """
        Return the corresponding inverse transformation.

        It holds ``x == self.inverted().transform(self.transform(x))``.

        The return value of this method should be treated as
        temporary.  An update to *self* does not cause a corresponding
        update to its inverted copy.
        """
        # 抛出未实现错误，表明该方法未被实现
        raise NotImplementedError()
# 定义一个继承自 Transform 的辅助类，用于包装单个子 Transform 并表现为其等效的类。
# 在运行时，如果必须用不同类型的 Transform 替换转换树中的节点，则此类非常有用。
# 这个类允许正确触发失效，以便正确替换节点。

# `TransformWrapper` 实例在其整个生命周期内必须具有相同的输入和输出维度，
# 因此子 Transform 只能被具有相同维度的另一个子 Transform 替换。

class TransformWrapper(Transform):
    pass_through = True  # 设置类属性 pass_through 为 True

    def __init__(self, child):
        """
        初始化方法。

        Args:
            child: `Transform` 实例，这个子 Transform 可以稍后被替换为其他 Transform。
        """
        _api.check_isinstance(Transform, child=child)  # 检查 child 是否为 Transform 的实例
        super().__init__()
        self.set(child)  # 调用 set 方法设置子 Transform

    def __eq__(self, other):
        """
        比较方法，检查两个 TransformWrapper 实例是否相等。

        Args:
            other: 另一个对象

        Returns:
            bool: 如果两个对象相等则返回 True，否则返回 False。
        """
        return self._child.__eq__(other)

    __str__ = _make_str_method("_child")  # 定义 __str__ 方法，用于返回对象描述信息

    def frozen(self):
        """
        返回子 Transform 的冻结状态。

        Returns:
            bool: 子 Transform 的冻结状态。
        """
        return self._child.frozen()

    def set(self, child):
        """
        替换此 Transform 的当前子 Transform。

        新子 Transform 必须具有与当前子 Transform 相同的输入和输出维度。

        Args:
            child: 新的子 Transform
        """
        if hasattr(self, "_child"):  # 在初始化期间不执行此操作
            self.invalidate()  # 失效当前 TransformWrapper 实例
            new_dims = (child.input_dims, child.output_dims)
            old_dims = (self._child.input_dims, self._child.output_dims)
            if new_dims != old_dims:
                raise ValueError(
                    f"The input and output dims of the new child {new_dims} "
                    f"do not match those of current child {old_dims}")
            self._child._parents.pop(id(self), None)

        self._child = child
        self.set_children(child)

        # 以下属性均从子 Transform 中继承
        self.transform = child.transform
        self.transform_affine = child.transform_affine
        self.transform_non_affine = child.transform_non_affine
        self.transform_path = child.transform_path
        self.transform_path_affine = child.transform_path_affine
        self.transform_path_non_affine = child.transform_path_non_affine
        self.get_affine = child.get_affine
        self.inverted = child.inverted
        self.get_matrix = child.get_matrix

        # 注意，这里没有包装其他属性，因为 Transform 的子类可以使用 WrappedTransform.set 更改，
        # 因此检查 is_affine 等其他属性可能是危险的。

        self._invalid = 0  # 重置失效标志
        self.invalidate()  # 失效当前 TransformWrapper 实例
        self._invalid = 0  # 重置失效标志

    # 以下是通过 property 定义的一些属性，它们从子 Transform 中继承
    input_dims = property(lambda self: self._child.input_dims)
    output_dims = property(lambda self: self._child.output_dims)
    is_affine = property(lambda self: self._child.is_affine)
    is_separable = property(lambda self: self._child.is_separable)
    has_inverse = property(lambda self: self._child.has_inverse)
    # 所有维度仿射变换的基类。
    """
    is_affine = True

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，并初始化 _inverted 属性为 None
        super().__init__(*args, **kwargs)
        self._inverted = None

    def __array__(self, *args, **kwargs):
        # 优化对变换矩阵的访问，相对于调用超类的方法
        return self.get_matrix()

    def __eq__(self, other):
        # 比较当前对象与另一个对象是否相等
        if getattr(other, "is_affine", False) and hasattr(other, "get_matrix"):
            return (self.get_matrix() == other.get_matrix()).all()
        return NotImplemented

    def transform(self, values):
        # 继承文档字符串
        return self.transform_affine(values)

    def transform_affine(self, values):
        # 继承文档字符串
        raise NotImplementedError('Affine subclasses should override this '
                                  'method.')

    @_api.rename_parameter("3.8", "points", "values")
    def transform_non_affine(self, values):
        # 继承文档字符串
        return values

    def transform_path(self, path):
        # 继承文档字符串
        return self.transform_path_affine(path)

    def transform_path_affine(self, path):
        # 继承文档字符串，并返回仿射变换后的路径对象
        return Path(self.transform_affine(path.vertices),
                    path.codes, path._interpolation_steps)

    def transform_path_non_affine(self, path):
        # 继承文档字符串，并返回非仿射变换后的路径对象
        return path

    def get_affine(self):
        # 继承文档字符串
        return self
class Affine2DBase(AffineBase):
    """
    The base class of all 2D affine transformations.

    2D affine transformations are performed using a 3x3 numpy array::

        a c e
        b d f
        0 0 1

    This class provides the read-only interface.  For a mutable 2D
    affine transformation, use `Affine2D`.

    Subclasses of this class will generally only need to override a
    constructor and `~.Transform.get_matrix` that generates a custom 3x3 matrix.
    """

    input_dims = 2
    output_dims = 2

    def frozen(self):
        # docstring inherited
        # Creates a new Affine2D object with a copy of the current transformation matrix.
        return Affine2D(self.get_matrix().copy())

    @property
    def is_separable(self):
        # Determines if the affine transformation matrix is separable.
        mtx = self.get_matrix()
        return mtx[0, 1] == mtx[1, 0] == 0.0

    def to_values(self):
        """
        Return the values of the matrix as an ``(a, b, c, d, e, f)`` tuple.
        """
        mtx = self.get_matrix()
        return tuple(mtx[:2].swapaxes(0, 1).flat)

    @_api.rename_parameter("3.8", "points", "values")
    def transform_affine(self, values):
        # Transforms the input values using the affine transformation matrix.
        mtx = self.get_matrix()
        if isinstance(values, np.ma.MaskedArray):
            # Apply transformation to the unmasked data, maintaining masked status.
            tpoints = affine_transform(values.data, mtx)
            return np.ma.MaskedArray(tpoints, mask=np.ma.getmask(values))
        return affine_transform(values, mtx)

    if DEBUG:
        _transform_affine = transform_affine

        @_api.rename_parameter("3.8", "points", "values")
        def transform_affine(self, values):
            # docstring inherited
            # Warns if input values are not a numpy array, affecting performance.
            if not isinstance(values, np.ndarray):
                _api.warn_external(
                    f'A non-numpy array of type {type(values)} was passed in '
                    f'for transformation, which results in poor performance.')
            return self._transform_affine(values)

    def inverted(self):
        # docstring inherited
        # Computes and caches the inverse of the current affine transformation matrix.
        if self._inverted is None or self._invalid:
            mtx = self.get_matrix()
            shorthand_name = None
            if self._shorthand_name:
                shorthand_name = '(%s)-1' % self._shorthand_name
            self._inverted = Affine2D(inv(mtx), shorthand_name=shorthand_name)
            self._invalid = 0
        return self._inverted


class Affine2D(Affine2DBase):
    """
    A mutable 2D affine transformation.
    """

    def __init__(self, matrix=None, **kwargs):
        """
        Initialize an Affine transform from a 3x3 numpy float array::

          a c e
          b d f
          0 0 1

        If *matrix* is None, initialize with the identity transform.
        """
        super().__init__(**kwargs)
        if matrix is None:
            # Initialize with the identity matrix if no matrix is provided.
            matrix = IdentityTransform._mtx
        self._mtx = matrix.copy()
        self._invalid = 0
    # 创建一个私有变量 _base_str，并调用 _make_str_method 方法来生成一个方法名
    _base_str = _make_str_method("_mtx")

    # 定义 __str__ 方法，返回对象的字符串表示
    def __str__(self):
        # 如果矩阵 _mtx 中存在非对角线上的非零元素，则返回 _base_str 方法的结果
        return (self._base_str()
                # 否则，如果 _mtx 是对角矩阵但非全为对角元素相同，则返回缩放操作的字符串表示
                if (self._mtx != np.diag(np.diag(self._mtx))).any()
                # 否则，返回单位矩阵
                else f"Affine2D().scale({self._mtx[0, 0]}, {self._mtx[1, 1]})"
                if self._mtx[0, 0] != self._mtx[1, 1]
                # 否则，返回仅包含一个对角元素的缩放操作字符串表示
                else f"Affine2D().scale({self._mtx[0, 0]})")

    # 静态方法：从给定的值创建一个 Affine2D 实例
    @staticmethod
    def from_values(a, b, c, d, e, f):
        """
        Create a new Affine2D instance from the given values::

          a c e
          b d f
          0 0 1

        .
        """
        return Affine2D(
            np.array([a, c, e, b, d, f, 0.0, 0.0, 1.0], float).reshape((3, 3)))

    # 获取当前对象的变换矩阵
    def get_matrix(self):
        """
        Get the underlying transformation matrix as a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        # 如果矩阵无效，则重置缓存的逆矩阵和无效标志
        if self._invalid:
            self._inverted = None
            self._invalid = 0
        return self._mtx

    # 设置当前对象的变换矩阵
    def set_matrix(self, mtx):
        """
        Set the underlying transformation matrix from a 3x3 array::

          a c e
          b d f
          0 0 1

        .
        """
        self._mtx = mtx
        # 标记矩阵为无效，需要重新计算逆矩阵
        self.invalidate()

    # 从另一个 Affine2DBase 对象的冻结副本设置当前对象的状态
    def set(self, other):
        """
        Set this transformation from the frozen copy of another
        `Affine2DBase` object.
        """
        # 检查 other 是否为 Affine2DBase 类型
        _api.check_isinstance(Affine2DBase, other=other)
        # 获取 other 对象的矩阵并设置为当前对象的矩阵
        self._mtx = other.get_matrix()
        # 标记当前对象的矩阵为无效
        self.invalidate()

    # 将当前对象的矩阵重置为单位矩阵
    def clear(self):
        """
        Reset the underlying matrix to the identity transform.
        """
        # 比使用 np.identity(3) 更快的方法
        self._mtx = IdentityTransform._mtx.copy()
        # 标记当前对象的矩阵为无效并返回自身
        self.invalidate()
        return self

    # 在原地添加一个以弧度表示的旋转到当前变换
    def rotate(self, theta):
        """
        Add a rotation (in radians) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 计算 cos(theta) 和 sin(theta)
        a = math.cos(theta)
        b = math.sin(theta)
        # 获取当前对象的矩阵
        mtx = self._mtx
        # 从矩阵中提取元素
        (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
        # 更新矩阵中的值以添加旋转变换
        mtx[0, 0] = a * xx - b * yx
        mtx[0, 1] = a * xy - b * yy
        mtx[0, 2] = a * x0 - b * y0
        mtx[1, 0] = b * xx + a * yx
        mtx[1, 1] = b * xy + a * yy
        mtx[1, 2] = b * x0 + a * y0
        # 标记当前对象的矩阵为无效
        self.invalidate()
        return self

    # 在原地添加一个以度数表示的旋转到当前变换
    def rotate_deg(self, degrees):
        """
        Add a rotation (in degrees) to this transform in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 将角度转换为弧度并调用 rotate 方法
        return self.rotate(math.radians(degrees))
    def rotate_around(self, x, y, theta):
        """
        Add a rotation (in radians) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 在给定点 (x, y) 处按照弧度 theta 进行旋转
        # 返回 *self*，以便可以轻松地与更多的 :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate` 和 :meth:`scale` 方法链接
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        """
        Add a rotation (in degrees) around the point (x, y) in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 在给定点 (x, y) 处按照角度 degrees 进行旋转
        # 将 x 和 y 转换为浮点数，以避免 uint8 类型的环绕问题
        x, y = float(x), float(y)
        # 返回 *self*，以便可以轻松地与更多的 :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate` 和 :meth:`scale` 方法链接
        return self.translate(-x, -y).rotate_deg(degrees).translate(x, y)

    def translate(self, tx, ty):
        """
        Add a translation in place.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 在原地进行平移操作
        self._mtx[0, 2] += tx
        self._mtx[1, 2] += ty
        # 使变换矩阵无效
        self.invalidate()
        return self

    def scale(self, sx, sy=None):
        """
        Add a scale in place.

        If *sy* is None, the same scale is applied in both the *x*- and
        *y*-directions.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 在原地进行缩放操作
        # 如果 sy 为 None，则在 *x* 和 *y* 方向上应用相同的缩放
        if sy is None:
            sy = sx
        # 显式逐元素缩放是最快的
        self._mtx[0, 0] *= sx
        self._mtx[0, 1] *= sx
        self._mtx[0, 2] *= sx
        self._mtx[1, 0] *= sy
        self._mtx[1, 1] *= sy
        self._mtx[1, 2] *= sy
        # 使变换矩阵无效
        self.invalidate()
        return self

    def skew(self, xShear, yShear):
        """
        Add a skew in place.

        *xShear* and *yShear* are the shear angles along the *x*- and
        *y*-axes, respectively, in radians.

        Returns *self*, so this method can easily be chained with more
        calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
        and :meth:`scale`.
        """
        # 在原地进行倾斜操作
        # *xShear* 和 *yShear* 分别是 *x* 和 *y* 轴上的倾斜角度，单位为弧度
        rx = math.tan(xShear)
        ry = math.tan(yShear)
        mtx = self._mtx
        # 逐个标量操作和赋值速度更快
        (xx, xy, x0), (yx, yy, y0), _ = mtx.tolist()
        # mtx = [[1 rx 0], [ry 1 0], [0 0 1]] * mtx
        mtx[0, 0] += rx * yx
        mtx[0, 1] += rx * yy
        mtx[0, 2] += rx * y0
        mtx[1, 0] += ry * xx
        mtx[1, 1] += ry * xy
        mtx[1, 2] += ry * x0
        # 使变换矩阵无效
        self.invalidate()
        return self
    # 定义一个方法 skew_deg，用于在原地进行斜切变换。

    """
    Add a skew in place.

    *xShear* and *yShear* are the shear angles along the *x*- and
    *y*-axes, respectively, in degrees.

    Returns *self*, so this method can easily be chained with more
    calls to :meth:`rotate`, :meth:`rotate_deg`, :meth:`translate`
    and :meth:`scale`.
    """
    
    # 调用 self 对象的 skew 方法，将角度转换为弧度后施加 xShear 和 yShear 的斜切变换
    return self.skew(math.radians(xShear), math.radians(yShear))
class IdentityTransform(Affine2DBase):
    """
    A special class that does one thing, the identity transform, in a
    fast way.
    """
    _mtx = np.identity(3)  # 定义一个3x3的单位矩阵作为类变量 _mtx

    def frozen(self):
        # 返回当前对象自身，用于实现“冻结”效果
        return self

    __str__ = _make_str_method()  # 将 __str__ 方法设置为 _make_str_method() 的返回值

    def get_matrix(self):
        # 返回当前变换的矩阵 _mtx
        return self._mtx

    @_api.rename_parameter("3.8", "points", "values")
    def transform(self, values):
        # 对输入的 values 进行类型转换，返回一个 NumPy 数组
        return np.asanyarray(values)

    @_api.rename_parameter("3.8", "points", "values")
    def transform_affine(self, values):
        # 对输入的 values 进行类型转换，返回一个 NumPy 数组
        return np.asanyarray(values)

    @_api.rename_parameter("3.8", "points", "values")
    def transform_non_affine(self, values):
        # 对输入的 values 进行类型转换，返回一个 NumPy 数组
        return np.asanyarray(values)

    def transform_path(self, path):
        # 直接返回输入的 path，不做任何变换
        return path

    def transform_path_affine(self, path):
        # 直接返回输入的 path，不做任何变换
        return path

    def transform_path_non_affine(self, path):
        # 直接返回输入的 path，不做任何变换
        return path

    def get_affine(self):
        # 返回当前对象自身，用于获取当前的仿射变换
        return self

    def inverted(self):
        # 返回当前对象自身，用于获取当前的反转变换
        return self


class _BlendedMixin:
    """Common methods for `BlendedGenericTransform` and `BlendedAffine2D`."""

    def __eq__(self, other):
        # 检查两个对象是否相等，如果类型为 BlendedAffine2D 或 BlendedGenericTransform，则比较内部变量 _x 和 _y 是否相等
        if isinstance(other, (BlendedAffine2D, BlendedGenericTransform)):
            return (self._x == other._x) and (self._y == other._y)
        # 否则，比较当前对象的 _x 和 _y 是否都等于 other
        elif self._x == self._y:
            return self._x == other
        else:
            return NotImplemented

    def contains_branch_seperately(self, transform):
        # 返回一个元组，包含 self._x 和 self._y 对 transform 的 contains_branch 操作的结果
        return (self._x.contains_branch(transform),
                self._y.contains_branch(transform))

    __str__ = _make_str_method("_x", "_y")


class BlendedGenericTransform(_BlendedMixin, Transform):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This "generic" version can handle any given child transform in the
    *x*- and *y*-directions.
    """
    input_dims = 2  # 输入维度为2
    output_dims = 2  # 输出维度为2
    is_separable = True  # 可分离变换
    pass_through = True  # 通过变换

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to transform the
        *x*-axis and *y_transform* to transform the *y*-axis.

        You will generally not call this constructor directly but use the
        `blended_transform_factory` function instead, which can determine
        automatically which kind of blended transform to create.
        """
        Transform.__init__(self, **kwargs)  # 调用父类 Transform 的构造函数
        self._x = x_transform  # 将 x_transform 设置为类成员 _x
        self._y = y_transform  # 将 y_transform 设置为类成员 _y
        self.set_children(x_transform, y_transform)  # 调用 set_children 方法设置子变换
        self._affine = None  # 初始化 _affine 为 None

    @property
    def depth(self):
        # 返回 _x 和 _y 中深度较大的那个
        return max(self._x.depth, self._y.depth)
    # 判断当前混合变换是否包含来自不同变换的分支；由于混合变换不可能同时包含来自两个不同变换的分支，因此直接返回 False
    def contains_branch(self, other):
        return False

    # 检查混合变换是否是仿射的属性，返回结果是 self._x 和 self._y 的仿射属性的逻辑与
    is_affine = property(lambda self: self._x.is_affine and self._y.is_affine)

    # 检查混合变换是否具有逆变换的属性，返回结果是 self._x 和 self._y 的逆变换属性的逻辑与
    has_inverse = property(
        lambda self: self._x.has_inverse and self._y.has_inverse)

    # 返回冻结后的混合变换对象，调用 blended_transform_factory 方法，使用 self._x 和 self._y 的冻结版本作为参数
    def frozen(self):
        return blended_transform_factory(self._x.frozen(), self._y.frozen())

    # 针对非仿射变换的转换方法，根据 self._x 和 self._y 的仿射属性进行条件判断和数据转换
    @_api.rename_parameter("3.8", "points", "values")
    def transform_non_affine(self, values):
        # 如果 self._x 和 self._y 都是仿射的，则直接返回输入的 values
        if self._x.is_affine and self._y.is_affine:
            return values
        x = self._x
        y = self._y

        # 如果 self._x 和 self._y 相等且输入维度是 2，则只对 x 进行非仿射变换
        if x == y and x.input_dims == 2:
            return x.transform_non_affine(values)

        # 根据 self._x 的输入维度不同进行 x_points 的转换处理
        if x.input_dims == 2:
            x_points = x.transform_non_affine(values)[:, 0:1]
        else:
            x_points = x.transform_non_affine(values[:, 0])
            x_points = x_points.reshape((len(x_points), 1))

        # 根据 self._y 的输入维度不同进行 y_points 的转换处理
        if y.input_dims == 2:
            y_points = y.transform_non_affine(values)[:, 1:]
        else:
            y_points = y.transform_non_affine(values[:, 1])
            y_points = y_points.reshape((len(y_points), 1))

        # 如果 x_points 或 y_points 是 MaskedArray，则返回合并后的 MaskedArray
        if (isinstance(x_points, np.ma.MaskedArray) or
                isinstance(y_points, np.ma.MaskedArray)):
            return np.ma.concatenate((x_points, y_points), 1)
        else:
            return np.concatenate((x_points, y_points), 1)

    # 返回混合变换的逆变换对象，调用 BlendedGenericTransform 构造器，使用 self._x 和 self._y 的逆变换版本作为参数
    def inverted(self):
        return BlendedGenericTransform(self._x.inverted(), self._y.inverted())

    # 获取混合变换的仿射变换对象，如果当前对象无效或者仿射变换为空，则重新计算仿射变换矩阵并返回 Affine2D 对象
    def get_affine(self):
        if self._invalid or self._affine is None:
            if self._x == self._y:
                self._affine = self._x.get_affine()
            else:
                x_mtx = self._x.get_affine().get_matrix()
                y_mtx = self._y.get_affine().get_matrix()
                # 由于已知变换是可分离的，因此可以跳过将 b 和 c 设置为零的步骤
                mtx = np.array([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])
                self._affine = Affine2D(mtx)
            self._invalid = 0
        return self._affine
class BlendedAffine2D(_BlendedMixin, Affine2DBase):
    """
    A "blended" transform uses one transform for the *x*-direction, and
    another transform for the *y*-direction.

    This version is an optimization for the case where both child
    transforms are of type `Affine2DBase`.
    """

    is_separable = True  # 设定属性，表明该类的变换是可分离的

    def __init__(self, x_transform, y_transform, **kwargs):
        """
        Create a new "blended" transform using *x_transform* to transform the
        *x*-axis and *y_transform* to transform the *y*-axis.

        Both *x_transform* and *y_transform* must be 2D affine transforms.

        You will generally not call this constructor directly but use the
        `blended_transform_factory` function instead, which can determine
        automatically which kind of blended transform to create.
        """
        is_affine = x_transform.is_affine and y_transform.is_affine  # 检查是否是仿射变换
        is_separable = x_transform.is_separable and y_transform.is_separable  # 检查是否是可分离变换
        is_correct = is_affine and is_separable  # 检查是否同时满足仿射和可分离性
        if not is_correct:
            raise ValueError("Both *x_transform* and *y_transform* must be 2D "
                             "affine transforms")  # 如果不满足条件，抛出值错误异常

        Transform.__init__(self, **kwargs)  # 调用父类 Transform 的构造函数
        self._x = x_transform  # 存储 x 轴方向的变换
        self._y = y_transform  # 存储 y 轴方向的变换
        self.set_children(x_transform, y_transform)  # 设置子变换对象

        Affine2DBase.__init__(self)  # 调用 Affine2DBase 的构造函数
        self._mtx = None  # 初始化变换矩阵为 None

    def get_matrix(self):
        # docstring inherited
        if self._invalid:
            if self._x == self._y:
                self._mtx = self._x.get_matrix()  # 如果 x 和 y 变换相同，直接获取其变换矩阵
            else:
                x_mtx = self._x.get_matrix()  # 获取 x 变换的变换矩阵
                y_mtx = self._y.get_matrix()  # 获取 y 变换的变换矩阵
                # We already know the transforms are separable, so we can skip
                # setting b and c to zero.
                self._mtx = np.array([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])  # 组装新的变换矩阵
            self._inverted = None
            self._invalid = 0
        return self._mtx  # 返回计算得到的变换矩阵


def blended_transform_factory(x_transform, y_transform):
    """
    Create a new "blended" transform using *x_transform* to transform
    the *x*-axis and *y_transform* to transform the *y*-axis.

    A faster version of the blended transform is returned for the case
    where both child transforms are affine.
    """
    if (isinstance(x_transform, Affine2DBase) and
            isinstance(y_transform, Affine2DBase)):
        return BlendedAffine2D(x_transform, y_transform)  # 如果 x_transform 和 y_transform 都是 Affine2DBase 类型，返回 BlendedAffine2D 实例
    return BlendedGenericTransform(x_transform, y_transform)  # 否则返回通用的 BlendedGenericTransform 实例


class CompositeGenericTransform(Transform):
    """
    A composite transform formed by applying transform *a* then
    transform *b*.

    This "generic" version can handle any two arbitrary
    transformations.
    """
    pass_through = True  # 设定属性，表明此类可以直接通过
    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying transform *a* then transform *b*.

        You will generally not call this constructor directly but write ``a +
        b`` instead, which will automatically choose the best kind of composite
        transform instance to create.
        """
        # 检查输入维度是否匹配，如果不匹配则抛出数值错误异常
        if a.output_dims != b.input_dims:
            raise ValueError("The output dimension of 'a' must be equal to "
                             "the input dimensions of 'b'")
        # 设置输入维度为 a 的输入维度，输出维度为 b 的输出维度
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims

        # 调用父类的构造函数，传递任意关键字参数
        super().__init__(**kwargs)
        # 设置内部属性 _a 和 _b 分别为传入的 a 和 b
        self._a = a
        self._b = b
        # 设置此组合变换的子变换为 a 和 b
        self.set_children(a, b)

    def frozen(self):
        # docstring inherited
        # 将内部属性 _invalid 设置为 0，表示不再无效
        self._invalid = 0
        # 使用 composite_transform_factory 函数冻结 _a 和 _b 得到新的组合变换
        frozen = composite_transform_factory(
            self._a.frozen(), self._b.frozen())
        # 如果冻结后的结果不是 CompositeGenericTransform 类型，则继续冻结直到得到有效的冻结结果
        if not isinstance(frozen, CompositeGenericTransform):
            return frozen.frozen()
        return frozen

    def _invalidate_internal(self, level, invalidating_node):
        # When the left child is invalidated at AFFINE_ONLY level and the right child is
        # non-affine, the composite transform is FULLY invalidated.
        # 当左侧子节点在 AFFINE_ONLY 级别上无效，并且右侧子节点不是仿射变换时，整个组合变换被完全无效化
        if invalidating_node is self._a and not self._b.is_affine:
            level = Transform._INVALID_FULL
        # 调用父类的 _invalidate_internal 方法进行无效化处理
        super()._invalidate_internal(level, invalidating_node)

    def __eq__(self, other):
        # 判断两个组合变换是否相等
        if isinstance(other, (CompositeGenericTransform, CompositeAffine2D)):
            return self is other or (self._a == other._a
                                     and self._b == other._b)
        else:
            return False

    def _iter_break_from_left_to_right(self):
        # 从左到右迭代断开变换
        for left, right in self._a._iter_break_from_left_to_right():
            yield left, right + self._b
        for left, right in self._b._iter_break_from_left_to_right():
            yield self._a + left, right

    # 获取组合变换的深度属性
    depth = property(lambda self: self._a.depth + self._b.depth)
    # 判断组合变换是否是仿射的
    is_affine = property(lambda self: self._a.is_affine and self._b.is_affine)
    # 判断组合变换是否是可分离的
    is_separable = property(
        lambda self: self._a.is_separable and self._b.is_separable)
    # 判断组合变换是否有逆变换
    has_inverse = property(
        lambda self: self._a.has_inverse and self._b.has_inverse)

    # 将 __str__ 方法设置为生成字符串表示的方法，内部调用 _make_str_method
    __str__ = _make_str_method("_a", "_b")

    @_api.rename_parameter("3.8", "points", "values")
    def transform_affine(self, values):
        # docstring inherited
        # 调用 get_affine() 方法进行仿射变换
        return self.get_affine().transform(values)

    @_api.rename_parameter("3.8", "points", "values")
    def transform_non_affine(self, values):
        # docstring inherited
        # 如果两个子变换都是仿射的，则返回原始值
        if self._a.is_affine and self._b.is_affine:
            return values
        # 如果第一个子变换不是仿射的而第二个是仿射的，则对第一个进行非仿射变换
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_non_affine(values)
        # 否则对第二个子变换进行非仿射变换
        else:
            return self._b.transform_non_affine(self._a.transform(values))
    def transform_path_non_affine(self, path):
        # docstring inherited
        # 如果self._a和self._b都是仿射变换，则直接返回路径path
        if self._a.is_affine and self._b.is_affine:
            return path
        # 如果self._a不是仿射变换而self._b是仿射变换，则对路径path应用self._a的非仿射变换
        elif not self._a.is_affine and self._b.is_affine:
            return self._a.transform_path_non_affine(path)
        # 否则，self._a和self._b都不是仿射变换，先对路径path应用self._a的变换，然后再对结果应用self._b的非仿射变换
        else:
            return self._b.transform_path_non_affine(
                                    self._a.transform_path(path))

    def get_affine(self):
        # docstring inherited
        # 如果self._b不是仿射变换，则递归获取self._b的仿射变换
        if not self._b.is_affine:
            return self._b.get_affine()
        # 否则，计算self._a和self._b的仿射变换的复合结果
        else:
            return Affine2D(np.dot(self._b.get_affine().get_matrix(),
                                   self._a.get_affine().get_matrix()))

    def inverted(self):
        # docstring inherited
        # 返回一个新的CompositeGenericTransform对象，其中self._a和self._b的逆变换互换
        return CompositeGenericTransform(
            self._b.inverted(), self._a.inverted())
class CompositeAffine2D(Affine2DBase):
    """
    A composite transform formed by applying transform *a* then transform *b*.

    This version is an optimization that handles the case where both *a*
    and *b* are 2D affines.
    """
    def __init__(self, a, b, **kwargs):
        """
        Create a new composite transform that is the result of
        applying `Affine2DBase` *a* then `Affine2DBase` *b*.

        You will generally not call this constructor directly but write ``a +
        b`` instead, which will automatically choose the best kind of composite
        transform instance to create.
        """
        # 检查参数a和b是否都是仿射变换
        if not a.is_affine or not b.is_affine:
            raise ValueError("'a' and 'b' must be affine transforms")
        # 检查a的输出维度是否等于b的输入维度
        if a.output_dims != b.input_dims:
            raise ValueError("The output dimension of 'a' must be equal to "
                             "the input dimensions of 'b'")
        # 设置输入维度和输出维度
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims

        super().__init__(**kwargs)
        # 将a和b设为此对象的子对象
        self._a = a
        self._b = b
        self.set_children(a, b)
        self._mtx = None

    @property
    def depth(self):
        # 返回此复合变换的深度，即a和b深度之和
        return self._a.depth + self._b.depth

    def _iter_break_from_left_to_right(self):
        # 从左到右依次迭代a和b的分解
        for left, right in self._a._iter_break_from_left_to_right():
            yield left, right + self._b
        for left, right in self._b._iter_break_from_left_to_right():
            yield self._a + left, right

    __str__ = _make_str_method("_a", "_b")

    def get_matrix(self):
        # docstring inherited
        # 如果当前变换无效，则重新计算变换矩阵_mtx
        if self._invalid:
            self._mtx = np.dot(
                self._b.get_matrix(),
                self._a.get_matrix())
            self._inverted = None
            self._invalid = 0
        return self._mtx


def composite_transform_factory(a, b):
    """
    Create a new composite transform that is the result of applying
    transform a then transform b.

    Shortcut versions of the blended transform are provided for the
    case where both child transforms are affine, or one or the other
    is the identity transform.

    Composite transforms may also be created using the '+' operator,
    e.g.::

      c = a + b
    """
    # 检查a或b是否为IdentityTransform，使用isinstance确保始终为IdentityTransform
    if isinstance(a, IdentityTransform):
        return b
    elif isinstance(b, IdentityTransform):
        return a
    elif isinstance(a, Affine2D) and isinstance(b, Affine2D):
        # 如果a和b均为Affine2D，则返回CompositeAffine2D对象
        return CompositeAffine2D(a, b)
    # 否则返回通用的CompositeGenericTransform对象
    return CompositeGenericTransform(a, b)


class BboxTransform(Affine2DBase):
    """
    `BboxTransform` linearly transforms points from one `Bbox` to another.
    """

    is_separable = True
    def __init__(self, boxin, boxout, **kwargs):
        """
        Create a new `BboxTransform` that linearly transforms
        points from *boxin* to *boxout*.
        """
        # 检查 boxin 和 boxout 是否是 BboxBase 的实例
        _api.check_isinstance(BboxBase, boxin=boxin, boxout=boxout)

        # 调用父类的初始化方法，并传递额外的关键字参数
        super().__init__(**kwargs)
        # 将 boxin 和 boxout 分别保存到实例变量 _boxin 和 _boxout 中
        self._boxin = boxin
        self._boxout = boxout
        # 调用 set_children 方法，设置 boxin 和 boxout 为其子元素
        self.set_children(boxin, boxout)
        # 初始化矩阵变量和反转标记变量为 None
        self._mtx = None
        self._inverted = None

    # 将 __str__ 方法重载为根据 _boxin 和 _boxout 生成字符串表示的方法
    __str__ = _make_str_method("_boxin", "_boxout")

    def get_matrix(self):
        # 继承的文档字符串：获取变换矩阵
        if self._invalid:
            # 如果标记为无效，则重新计算矩阵
            inl, inb, inw, inh = self._boxin.bounds
            outl, outb, outw, outh = self._boxout.bounds
            # 计算 x 和 y 方向的缩放比例
            x_scale = outw / inw
            y_scale = outh / inh
            # 如果处于调试模式并且任一方向的缩放比例为零，则引发 ValueError
            if DEBUG and (x_scale == 0 or y_scale == 0):
                raise ValueError(
                    "Transforming from or to a singular bounding box")
            # 构造变换矩阵并存储到实例变量 _mtx 中
            self._mtx = np.array([[x_scale, 0.0    , (-inl*x_scale+outl)],
                                  [0.0    , y_scale, (-inb*y_scale+outb)],
                                  [0.0    , 0.0    , 1.0        ]],
                                 float)
            # 将反转标记和无效标记重置为 None 和 False
            self._inverted = None
            self._invalid = 0
        # 返回存储的变换矩阵
        return self._mtx
class BboxTransformTo(Affine2DBase):
    """
    `BboxTransformTo` is a transformation that linearly transforms points from
    the unit bounding box to a given `Bbox`.
    """

    is_separable = True  # 设置一个类属性 `is_separable`，表示此类的实例可以分离变换

    def __init__(self, boxout, **kwargs):
        """
        Create a new `BboxTransformTo` that linearly transforms
        points from the unit bounding box to *boxout*.
        """
        _api.check_isinstance(BboxBase, boxout=boxout)  # 使用 _api 模块检查 boxout 是否为 BboxBase 的实例

        super().__init__(**kwargs)  # 调用父类的构造方法
        self._boxout = boxout  # 设置实例变量 `_boxout`，表示要转换到的目标 Bbox
        self.set_children(boxout)  # 设置此变换的子对象为 boxout
        self._mtx = None  # 初始化变换矩阵 `_mtx` 为 None
        self._inverted = None  # 初始化逆变换矩阵 `_inverted` 为 None

    __str__ = _make_str_method("_boxout")  # 创建一个 `__str__` 方法，返回 `_boxout` 的字符串表示形式

    def get_matrix(self):
        # docstring inherited
        if self._invalid:  # 如果变换无效
            outl, outb, outw, outh = self._boxout.bounds  # 获取目标 Bbox 的左、下、宽、高
            if DEBUG and (outw == 0 or outh == 0):  # 如果调试模式开启且宽或高为零
                raise ValueError("Transforming to a singular bounding box.")  # 抛出值错误异常
            # 构造变换矩阵 `_mtx`，将单位边界框线性变换到目标 Bbox
            self._mtx = np.array([[outw,  0.0, outl],
                                  [ 0.0, outh, outb],
                                  [ 0.0,  0.0,  1.0]],
                                 float)
            self._inverted = None  # 重置逆变换矩阵 `_inverted` 为 None
            self._invalid = 0  # 标记变换为有效
        return self._mtx  # 返回变换矩阵 `_mtx`


@_api.deprecated("3.9")  # 使用 _api 模块标记此类为已弃用，弃用信息为 "3.9"
class BboxTransformToMaxOnly(BboxTransformTo):
    """
    `BboxTransformToMaxOnly` is a transformation that linearly transforms points from
    the unit bounding box to a given `Bbox` with a fixed upper left of (0, 0).
    """
    def get_matrix(self):
        # docstring inherited
        if self._invalid:  # 如果变换无效
            xmax, ymax = self._boxout.max  # 获取目标 Bbox 的右上角坐标
            if DEBUG and (xmax == 0 or ymax == 0):  # 如果调试模式开启且 xmax 或 ymax 为零
                raise ValueError("Transforming to a singular bounding box.")  # 抛出值错误异常
            # 构造变换矩阵 `_mtx`，将单位边界框线性变换到目标 Bbox，固定左上角在 (0, 0)
            self._mtx = np.array([[xmax,  0.0, 0.0],
                                  [ 0.0, ymax, 0.0],
                                  [ 0.0,  0.0, 1.0]],
                                 float)
            self._inverted = None  # 重置逆变换矩阵 `_inverted` 为 None
            self._invalid = 0  # 标记变换为有效
        return self._mtx  # 返回变换矩阵 `_mtx`


class BboxTransformFrom(Affine2DBase):
    """
    `BboxTransformFrom` linearly transforms points from a given `Bbox` to the
    unit bounding box.
    """
    is_separable = True  # 设置一个类属性 `is_separable`，表示此类的实例可以分离变换

    def __init__(self, boxin, **kwargs):
        _api.check_isinstance(BboxBase, boxin=boxin)  # 使用 _api 模块检查 boxin 是否为 BboxBase 的实例

        super().__init__(**kwargs)  # 调用父类的构造方法
        self._boxin = boxin  # 设置实例变量 `_boxin`，表示要从中转换的源 Bbox
        self.set_children(boxin)  # 设置此变换的子对象为 boxin
        self._mtx = None  # 初始化变换矩阵 `_mtx` 为 None
        self._inverted = None  # 初始化逆变换矩阵 `_inverted` 为 None

    __str__ = _make_str_method("_boxin")  # 创建一个 `__str__` 方法，返回 `_boxin` 的字符串表示形式
    def get_matrix(self):
        # 继承的文档字符串，说明这个方法继承自父类，并且没有额外的注释说明
        if self._invalid:
            # 如果标记为无效
            inl, inb, inw, inh = self._boxin.bounds
            # 解构赋值获取边界框的左、底、宽度和高度
            if DEBUG and (inw == 0 or inh == 0):
                # 如果调试模式开启且宽度或高度为零
                raise ValueError("Transforming from a singular bounding box.")
                # 抛出值错误异常，表示从一个奇异的边界框进行变换
            x_scale = 1.0 / inw
            # 计算 x 方向的缩放比例
            y_scale = 1.0 / inh
            # 计算 y 方向的缩放比例
            self._mtx = np.array([[x_scale, 0.0    , (-inl*x_scale)],
                                  [0.0    , y_scale, (-inb*y_scale)],
                                  [0.0    , 0.0    , 1.0        ]],
                                 float)
            # 创建一个 3x3 的浮点数类型的变换矩阵，应用缩放和平移
            self._inverted = None
            # 清空反转标记
            self._invalid = 0
            # 重置无效标记为 0，表示计算完成且有效
        return self._mtx
        # 返回变换矩阵
class ScaledTranslation(Affine2DBase):
    """
    A transformation that translates by *xt* and *yt*, after *xt* and *yt*
    have been transformed by *scale_trans*.
    """

    def __init__(self, xt, yt, scale_trans, **kwargs):
        # 调用父类的构造方法，初始化基类 Affine2DBase
        super().__init__(**kwargs)
        # 设置平移量 _t，通过元组 (xt, yt) 表示
        self._t = (xt, yt)
        # 设置缩放变换 _scale_trans
        self._scale_trans = scale_trans
        # 将 scale_trans 设置为当前对象的子节点
        self.set_children(scale_trans)
        # 初始化变换矩阵为 None
        self._mtx = None
        # 初始化反转后的矩阵为 None
        self._inverted = None

    __str__ = _make_str_method("_t")

    def get_matrix(self):
        # docstring inherited
        # 如果标记为无效，则重新计算变换矩阵
        if self._invalid:
            # 使用 IdentityTransform 的复制来加速，而不是 np.identity(3)
            self._mtx = IdentityTransform._mtx.copy()
            # 设置变换矩阵的平移部分，通过 scale_trans.transform(self._t) 转换后的结果
            self._mtx[:2, 2] = self._scale_trans.transform(self._t)
            # 标记为有效
            self._invalid = 0
            # 清空反转后的矩阵缓存
            self._inverted = None
        # 返回计算后的变换矩阵
        return self._mtx


class AffineDeltaTransform(Affine2DBase):
    r"""
    A transform wrapper for transforming displacements between pairs of points.

    This class is intended to be used to transform displacements ("position
    deltas") between pairs of points (e.g., as the ``offset_transform``
    of `.Collection`\s): given a transform ``t`` such that ``t =
    AffineDeltaTransform(t) + offset``, ``AffineDeltaTransform``
    satisfies ``AffineDeltaTransform(a - b) == AffineDeltaTransform(a) -
    AffineDeltaTransform(b)``.

    This is implemented by forcing the offset components of the transform
    matrix to zero.

    This class is experimental as of 3.3, and the API may change.
    """

    def __init__(self, transform, **kwargs):
        # 调用父类的构造方法，初始化基类 Affine2DBase
        super().__init__(**kwargs)
        # 设置基础变换 _base_transform
        self._base_transform = transform

    __str__ = _make_str_method("_base_transform")

    def get_matrix(self):
        # 如果标记为无效，则重新计算变换矩阵
        if self._invalid:
            # 复制基础变换的变换矩阵
            self._mtx = self._base_transform.get_matrix().copy()
            # 强制偏移部分置零，即变换矩阵的最后一列的前两行置零
            self._mtx[:2, -1] = 0
        # 返回计算后的变换矩阵
        return self._mtx


class TransformedPath(TransformNode):
    """
    A `TransformedPath` caches a non-affine transformed copy of the
    `~.path.Path`.  This cached copy is automatically updated when the
    non-affine part of the transform changes.

    .. note::

        Paths are considered immutable by this class. Any update to the
        path's vertices/codes will not trigger a transform recomputation.

    """

    def __init__(self, path, transform):
        """
        Parameters
        ----------
        path : `~.path.Path`
            要缓存的 `~.path.Path` 的非仿射变换副本。
        transform : `Transform`
            变换对象 `Transform`。
        """
        # 检查 transform 是否为 Transform 类的实例
        _api.check_isinstance(Transform, transform=transform)
        # 调用父类构造方法，初始化基类 TransformNode
        super().__init__()
        # 设置原始路径 _path
        self._path = path
        # 设置变换对象 _transform
        self._transform = transform
        # 将 transform 设置为当前对象的子节点
        self.set_children(transform)
        # 初始化缓存的转换路径为 None
        self._transformed_path = None
        # 初始化缓存的转换后的点集为 None
        self._transformed_points = None
    # 重新验证路径的有效性，只有当无效标记为INVALID_FULL或者_transformed_path为空时才重新计算
    def _revalidate(self):
        if (self._invalid == self._INVALID_FULL
                or self._transformed_path is None):
            # 对路径进行非仿射部分的变换，并将结果赋给_transformed_path
            self._transformed_path = \
                self._transform.transform_path_non_affine(self._path)
            # 对路径顶点进行非仿射变换，通过codes和vertices快速创建路径对象，并将结果赋给_transformed_points
            self._transformed_points = \
                Path._fast_from_codes_and_verts(
                    self._transform.transform_non_affine(self._path.vertices),
                    None, self._path)
        # 将_invalid标记重置为0，表示路径已经重新验证
        self._invalid = 0

    # 返回已经应用非仿射变换的子路径的副本，以及完成变换所需的仿射部分
    def get_transformed_points_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.  Unlike
        :meth:`get_transformed_path_and_affine`, no interpolation will
        be performed.
        """
        # 确保路径已经重新验证
        self._revalidate()
        # 返回已经应用非仿射变换的路径点集_transformed_points，以及仿射变换部分
        return self._transformed_points, self.get_affine()

    # 返回已经应用非仿射变换的子路径的副本，以及完成变换所需的仿射部分
    def get_transformed_path_and_affine(self):
        """
        Return a copy of the child path, with the non-affine part of
        the transform already applied, along with the affine part of
        the path necessary to complete the transformation.
        """
        # 确保路径已经重新验证
        self._revalidate()
        # 返回已经应用非仿射变换的路径_transformed_path，以及仿射变换部分
        return self._transformed_path, self.get_affine()

    # 返回完全变换后的子路径的副本
    def get_fully_transformed_path(self):
        """
        Return a fully-transformed copy of the child path.
        """
        # 确保路径已经重新验证
        self._revalidate()
        # 返回完全变换后的路径_transformed_path
        return self._transform.transform_path_affine(self._transformed_path)

    # 返回当前变换对象的仿射变换
    def get_affine(self):
        return self._transform.get_affine()
class TransformedPatchPath(TransformedPath):
    """
    A `TransformedPatchPath` caches a non-affine transformed copy of the
    `~.patches.Patch`. This cached copy is automatically updated when the
    non-affine part of the transform or the patch changes.
    """

    def __init__(self, patch):
        """
        Parameters
        ----------
        patch : `~.patches.Patch`
            The patch object to be encapsulated.
        """
        # Defer to TransformedPath.__init__.
        super().__init__(patch.get_path(), patch.get_transform())
        self._patch = patch

    def _revalidate(self):
        patch_path = self._patch.get_path()
        # Force invalidation if the patch path changed; otherwise, let base
        # class check invalidation.
        if patch_path != self._path:
            self._path = patch_path
            self._transformed_path = None
        super()._revalidate()


def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    """
    Modify the endpoints of a range as needed to avoid singularities.

    Parameters
    ----------
    vmin, vmax : float
        The initial endpoints of the range.
    expander : float, default: 0.001
        Fractional amount by which *vmin* and *vmax* are expanded if
        the original interval is too small, based on *tiny*.
    tiny : float, default: 1e-15
        Threshold for the ratio of the interval to the maximum absolute
        value of its endpoints. If the interval is smaller than
        this, it will be expanded. This value should be around
        1e-15 or larger; otherwise, the interval will be approaching
        the double precision resolution limit.
    increasing : bool, default: True
        If True, swap *vmin* and *vmax* if *vmin* > *vmax*.

    Returns
    -------
    vmin, vmax : float
        Modified endpoints, expanded and/or swapped if necessary.
        Returns -*expander*, *expander* if either input is inf or NaN,
        or if both inputs are 0 or very close to zero.
    """

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        return -expander, expander

    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True

    # Convert vmin, vmax to float to handle potential integer type wrapping
    vmin, vmax = map(float, [vmin, vmax])

    maxabsvalue = max(abs(vmin), abs(vmax))
    if maxabsvalue < (1e6 / tiny) * np.finfo(float).tiny:
        vmin = -expander
        vmax = expander

    elif vmax - vmin <= maxabsvalue * tiny:
        if vmax == 0 and vmin == 0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander * abs(vmin)
            vmax += expander * abs(vmax)

    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def interval_contains(interval, val):
    """
    Check, inclusively, whether an interval includes a given value.

    Parameters
    ----------
    interval : tuple
        A tuple representing the interval (start, end).
    val : float
        The value to check for inclusion in the interval.
    # 解构赋值，将 interval 元组的两个值分别赋给 a 和 b
    a, b = interval
    # 如果区间端点 a 大于 b，则交换它们，确保 a <= b
    if a > b:
        a, b = b, a
    # 返回 val 是否在闭区间 [a, b] 内的布尔值
    return a <= val <= b
def _interval_contains_close(interval, val, rtol=1e-10):
    """
    检查闭区间是否包含给定值，考虑浮点误差，允许一定的相对容差。

    Parameters
    ----------
    interval : (float, float)
        区间的端点。
    val : float
        要检查的值是否在区间内。
    rtol : float, 默认值: 1e-10
        允许区间端点外的相对容差。
        对于区间 ``[a, b]``，满足条件的值为
        ``a - rtol * (b - a) <= val <= b + rtol * (b - a)``。

    Returns
    -------
    bool
        *val* 是否在 *interval* 内（带有容差）。
    """
    a, b = interval
    if a > b:
        a, b = b, a
    rtol = (b - a) * rtol  # 计算允许的容差范围
    return a - rtol <= val <= b + rtol


def interval_contains_open(interval, val):
    """
    检查开区间是否包含给定值。

    Parameters
    ----------
    interval : (float, float)
        区间的端点。
    val : float
        要检查的值是否在区间内。

    Returns
    -------
    bool
        *val* 是否在 *interval* 内（不包含端点）。
    """
    a, b = interval
    return a < val < b or a > val > b


def offset_copy(trans, fig=None, x=0.0, y=0.0, units='inches'):
    """
    返回一个带有偏移量的新变换对象。

    Parameters
    ----------
    trans : `Transform` 的子类
        任何需要应用偏移的变换对象。
    fig : `~matplotlib.figure.Figure`, 默认值: None
        当前的图形对象。如果 *units* 是 'dots'，可以为 None。
    x, y : float, 默认值: 0.0
        要应用的偏移量。
    units : {'inches', 'points', 'dots'}, 默认值: 'inches'
        偏移量的单位。

    Returns
    -------
    `Transform` 的子类
        应用偏移量后的变换对象。
    """
    _api.check_in_list(['dots', 'points', 'inches'], units=units)  # 检查单位是否有效
    if units == 'dots':
        return trans + Affine2D().translate(x, y)  # 使用点作为单位时直接平移
    if fig is None:
        raise ValueError('For units of inches or points a fig kwarg is needed')  # 如果单位是英寸或点，需要提供图形对象
    if units == 'points':
        x /= 72.0
        y /= 72.0
    # 默认单位为 'inches'
    return trans + ScaledTranslation(x, y, fig.dpi_scale_trans)  # 使用英寸作为单位时，使用图形对象的 DPI 缩放转换进行平移
```