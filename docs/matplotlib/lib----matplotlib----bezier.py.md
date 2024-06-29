# `D:\src\scipysrc\matplotlib\lib\matplotlib\bezier.py`

```py
# 导入必要的模块和库
"""
A module providing some utility functions regarding Bézier path manipulation.
"""
from functools import lru_cache  # 导入 lru_cache 装饰器，用于函数结果缓存
import math  # 导入数学函数库
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 数学库

from matplotlib import _api  # 导入 Matplotlib 的 _api 模块


# 定义组合数计算函数，使用 NumPy 的向量化实现
@np.vectorize
@lru_cache(maxsize=128)
def _comb(n, k):
    # 如果 k 大于 n，则返回 0
    if k > n:
        return 0
    # 确保 k 不大于 n-k，以优化组合数计算
    k = min(k, n - k)
    # 使用 NumPy 计算组合数并返回整数结果
    i = np.arange(1, k + 1)
    return np.prod((n + 1 - i)/i).astype(int)


# 定义异常类，用于非交叉路径情况
class NonIntersectingPathException(ValueError):
    pass


# 定义获取两条直线交点的函数
def get_intersection(cx1, cy1, cos_t1, sin_t1,
                     cx2, cy2, cos_t2, sin_t2):
    """
    Return the intersection between the line through (*cx1*, *cy1*) at angle
    *t1* and the line through (*cx2*, *cy2*) at angle *t2*.
    """

    # 计算第一条直线的右手边
    line1_rhs = sin_t1 * cx1 - cos_t1 * cy1
    # 计算第二条直线的右手边
    line2_rhs = sin_t2 * cx2 - cos_t2 * cy2

    # 计算行列式的值
    a, b = sin_t1, -cos_t1
    c, d = sin_t2, -cos_t2
    ad_bc = a * d - b * c

    # 如果行列式的值接近于零，则抛出异常，表示直线不相交
    if abs(ad_bc) < 1e-12:
        raise ValueError("Given lines do not intersect. Please verify that "
                         "the angles are not equal or differ by 180 degrees.")

    # 计算右手边向量的逆矩阵，并计算交点的坐标
    a_, b_ = d, -b
    c_, d_ = -c, a
    a_, b_, c_, d_ = [k / ad_bc for k in [a_, b_, c_, d_]]

    x = a_ * line1_rhs + b_ * line2_rhs
    y = c_ * line1_rhs + d_ * line2_rhs

    return x, y


# 定义获取直线上法线点的函数
def get_normal_points(cx, cy, cos_t, sin_t, length):
    """
    For a line passing through (*cx*, *cy*) and having an angle *t*, return
    locations of the two points located along its perpendicular line at the
    distance of *length*.
    """

    # 如果长度为 0，则返回原始点坐标
    if length == 0.:
        return cx, cy, cx, cy

    # 计算垂直线的两个点的坐标
    cos_t1, sin_t1 = sin_t, -cos_t
    cos_t2, sin_t2 = -sin_t, cos_t

    x1, y1 = length * cos_t1 + cx, length * sin_t1 + cy
    x2, y2 = length * cos_t2 + cx, length * sin_t2 + cy

    return x1, y1, x2, y2


# Bézier 曲线相关函数

# 使用 de Casteljau 算法分割 Bézier 曲线
# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html

# 定义 de Casteljau 算法的一步计算函数
def _de_casteljau1(beta, t):
    next_beta = beta[:-1] * (1 - t) + beta[1:] * t
    return next_beta


# 分割 Bézier 曲线函数，返回分割后的两段曲线的控制点
def split_de_casteljau(beta, t):
    """
    Split a Bézier segment defined by its control points *beta* into two
    separate segments divided at *t* and return their control points.
    """
    beta = np.asarray(beta)
    beta_list = [beta]
    while True:
        beta = _de_casteljau1(beta, t)
        beta_list.append(beta)
        if len(beta) == 1:
            break
    left_beta = [beta[0] for beta in beta_list]
    right_beta = [beta[-1] for beta in reversed(beta_list)]

    return left_beta, right_beta


# 寻找 Bézier 曲线与闭合路径的交点函数
def find_bezier_t_intersecting_with_closedpath(
        bezier_point_at_t, inside_closedpath, t0=0., t1=1., tolerance=0.01):
    """
    Find the intersection of the Bézier curve with a closed path.

    The intersection point *t* is approximated by two parameters *t0*, *t1*
    """
    such that *t0* <= *t* <= *t1*.

    Search starts from *t0* and *t1* and uses a simple bisecting algorithm
    therefore one of the end points must be inside the path while the other
    doesn't. The search stops when the distance of the points parametrized by
    *t0* and *t1* gets smaller than the given *tolerance*.

    Parameters
    ----------
    bezier_point_at_t : callable
        A function returning x, y coordinates of the Bézier at parameter *t*.
        It must have the signature::

            bezier_point_at_t(t: float) -> tuple[float, float]

    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. It must have the signature::

            inside_closedpath(point: tuple[float, float]) -> bool

    t0, t1 : float
        Start parameters for the search.

    tolerance : float
        Maximal allowed distance between the final points.

    Returns
    -------
    t0, t1 : float
        The Bézier path parameters.
    """
    # 获取起始点和结束点的坐标
    start = bezier_point_at_t(t0)
    end = bezier_point_at_t(t1)

    # 判断起始点和结束点是否在封闭路径内
    start_inside = inside_closedpath(start)
    end_inside = inside_closedpath(end)

    # 如果起始点和结束点在同一侧且不重合，则抛出异常
    if start_inside == end_inside and start != end:
        raise NonIntersectingPathException(
            "Both points are on the same side of the closed path")

    while True:

        # 如果两点之间的距离小于给定的容差值，则返回结果
        if np.hypot(start[0] - end[0], start[1] - end[1]) < tolerance:
            return t0, t1

        # 计算中间点的参数值
        middle_t = 0.5 * (t0 + t1)
        middle = bezier_point_at_t(middle_t)
        middle_inside = inside_closedpath(middle)

        # 根据中间点的位置更新搜索范围
        if start_inside ^ middle_inside:
            t1 = middle_t
            if end == middle:
                # 可能出现无限循环的边缘情况
                # 由于相对于容差值的大数值引起
                return t0, t1
            end = middle
        else:
            t0 = middle_t
            if start == middle:
                # 可能出现无限循环的边缘情况
                # 由于相对于容差值的大数值引起
                return t0, t1
            start = middle
            start_inside = middle_inside
class BezierSegment:
    """
    A d-dimensional Bézier segment.

    Parameters
    ----------
    control_points : (N, d) array
        Location of the *N* control points.
    """

    def __init__(self, control_points):
        # 将控制点转换为 NumPy 数组
        self._cpoints = np.asarray(control_points)
        # 控制点的数量和维度
        self._N, self._d = self._cpoints.shape
        # 阶乘计算，用于计算贝塞尔曲线系数
        self._orders = np.arange(self._N)
        coeff = [math.factorial(self._N - 1)
                 // (math.factorial(i) * math.factorial(self._N - 1 - i))
                 for i in range(self._N)]
        # 计算贝塞尔曲线的系数矩阵
        self._px = (self._cpoints.T * coeff).T

    def __call__(self, t):
        """
        Evaluate the Bézier curve at point(s) *t* in [0, 1].

        Parameters
        ----------
        t : (k,) array-like
            Points at which to evaluate the curve.

        Returns
        -------
        (k, d) array
            Value of the curve for each point in *t*.
        """
        t = np.asarray(t)
        # 计算贝塞尔曲线在给定参数 t 处的值
        return (np.power.outer(1 - t, self._orders[::-1])
                * np.power.outer(t, self._orders)) @ self._px

    def point_at_t(self, t):
        """
        Evaluate the curve at a single point, returning a tuple of *d* floats.
        """
        # 在单一参数 t 处评估曲线，并返回结果元组
        return tuple(self(t))

    @property
    def control_points(self):
        """The control points of the curve."""
        # 返回曲线的控制点
        return self._cpoints

    @property
    def dimension(self):
        """The dimension of the curve."""
        # 返回曲线的维度
        return self._d

    @property
    def degree(self):
        """Degree of the polynomial. One less the number of control points."""
        # 返回曲线的次数，即控制点数减一
        return self._N - 1

    @property
    def polynomial_coefficients(self):
        r"""
        The polynomial coefficients of the Bézier curve.

        .. warning:: Follows opposite convention from `numpy.polyval`.

        Returns
        -------
        (n+1, d) array
            Coefficients after expanding in polynomial basis, where :math:`n`
            is the degree of the Bézier curve and :math:`d` its dimension.
            These are the numbers (:math:`C_j`) such that the curve can be
            written :math:`\sum_{j=0}^n C_j t^j`.

        Notes
        -----
        The coefficients are calculated as

        .. math::

            {n \choose j} \sum_{i=0}^j (-1)^{i+j} {j \choose i} P_i

        where :math:`P_i` are the control points of the curve.
        """
        n = self.degree
        # 对于高阶贝塞尔曲线，警告用户可能的数值不稳定性
        if n > 10:
            warnings.warn("Polynomial coefficients formula unstable for high "
                          "order Bezier curves!", RuntimeWarning)
        P = self.control_points
        j = np.arange(n+1)[:, None]
        i = np.arange(n+1)[None, :]  # _comb is non-zero for i <= j
        prefactor = (-1)**(i + j) * _comb(j, i)  # j on axis 0, i on axis 1
        # 计算多项式展开后的系数矩阵
        return _comb(n, j) * prefactor @ P  # j on axis 0, self.dimension on 1
    def axis_aligned_extrema(self):
        """
        Return the dimension and location of the curve's interior extrema.

        The extrema are the points along the curve where one of its partial
        derivatives is zero.

        Returns
        -------
        dims : array of int
            Index :math:`i` of the partial derivative which is zero at each
            interior extrema.
        dzeros : array of float
            Of same size as dims. The :math:`t` such that :math:`d/dx_i B(t) =
            0`
        """
        # 获取曲线的阶数
        n = self.degree
        # 如果阶数小于等于1，直接返回空数组
        if n <= 1:
            return np.array([]), np.array([])
        
        # 获取多项式系数
        Cj = self.polynomial_coefficients
        # 计算导数的系数
        dCj = np.arange(1, n+1)[:, None] * Cj[1:]
        
        # 初始化存储导数为零的维度和根的列表
        dims = []
        roots = []
        
        # 对每一个导数系数进行处理
        for i, pi in enumerate(dCj.T):
            # 求解多项式的根，注意要反转系数顺序
            r = np.roots(pi[::-1])
            roots.append(r)
            # 将对应的维度信息存入dims列表
            dims.append(np.full_like(r, i))
        
        # 将所有根和对应维度信息连接成数组
        roots = np.concatenate(roots)
        dims = np.concatenate(dims)
        
        # 筛选出实数根且处于 [0, 1] 范围内的根
        in_range = np.isreal(roots) & (roots >= 0) & (roots <= 1)
        
        # 返回符合条件的维度和实根部分
        return dims[in_range], np.real(roots)[in_range]
def split_bezier_intersecting_with_closedpath(
        bezier, inside_closedpath, tolerance=0.01):
    """
    将贝塞尔曲线在与闭合路径相交处分割成两段。

    Parameters
    ----------
    bezier : (N, 2) array-like
        贝塞尔曲线的控制点数组。参见 `.BezierSegment`。
    inside_closedpath : callable
        一个函数，如果给定点 (x, y) 在闭合路径内，则返回 True。
        参见 `.find_bezier_t_intersecting_with_closedpath`。
    tolerance : float
        相交的容差。参见 `.find_bezier_t_intersecting_with_closedpath`。

    Returns
    -------
    left, right
        两段贝塞尔曲线的控制点列表。
    """

    bz = BezierSegment(bezier)
    bezier_point_at_t = bz.point_at_t

    # 找到贝塞尔曲线与闭合路径相交处的参数 t0 和 t1
    t0, t1 = find_bezier_t_intersecting_with_closedpath(
        bezier_point_at_t, inside_closedpath, tolerance=tolerance)

    # 使用 de Casteljau 算法在 t0 和 t1 处分割贝塞尔曲线
    _left, _right = split_de_casteljau(bezier, (t0 + t1) / 2.)
    return _left, _right


# matplotlib specific


def split_path_inout(path, inside, tolerance=0.01, reorder_inout=False):
    """
    将路径在内外点变为 False 处分成两段。

    Parameters
    ----------
    path : matplotlib.path.Path
        要分割的路径对象。
    inside : callable
        一个函数，返回 True 如果点 (x, y) 在内部。
    tolerance : float
        相交的容差。
    reorder_inout : bool
        是否重新排序内外点。

    Raises
    ------
    ValueError
        如果路径与补丁不相交。

    """

    from .path import Path
    path_iter = path.iter_segments()

    ctl_points, command = next(path_iter)
    begin_inside = inside(ctl_points[-2:])  # 如果起始点在内部，则为 True

    ctl_points_old = ctl_points

    iold = 0
    i = 1

    for ctl_points, command in path_iter:
        iold = i
        i += len(ctl_points) // 2
        if inside(ctl_points[-2:]) != begin_inside:
            bezier_path = np.concatenate([ctl_points_old[-2:], ctl_points])
            break
        ctl_points_old = ctl_points
    else:
        raise ValueError("路径与补丁不相交")

    bp = bezier_path.reshape((-1, 2))
    left, right = split_bezier_intersecting_with_closedpath(
        bp, inside, tolerance)

    # 根据左右部分的控制点数确定路径代码
    if len(left) == 2:
        codes_left = [Path.LINETO]
        codes_right = [Path.MOVETO, Path.LINETO]
    elif len(left) == 3:
        codes_left = [Path.CURVE3, Path.CURVE3]
        codes_right = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    elif len(left) == 4:
        codes_left = [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        codes_right = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        raise AssertionError("永远不应该到达此处")

    verts_left = left[1:]
    verts_right = right[:]

    if path.codes is None:
        path_in = Path(np.concatenate([path.vertices[:i], verts_left]))
        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]))

    else:
        path_in = Path(np.concatenate([path.vertices[:iold], verts_left]),
                       np.concatenate([path.codes[:iold], codes_left]))

        path_out = Path(np.concatenate([verts_right, path.vertices[i:]]),
                        np.concatenate([codes_right, path.codes[i:]]))
    # 如果 reorder_inout 为真并且 begin_inside 为假，则交换 path_in 和 path_out 的值
    if reorder_inout and not begin_inside:
        path_in, path_out = path_out, path_in
    
    # 返回经过可能的交换后的 path_in 和 path_out 的值
    return path_in, path_out
def inside_circle(cx, cy, r):
    """
    Return a function that checks whether a point is in a circle with center
    (*cx*, *cy*) and radius *r*.

    The returned function has the signature::

        f(xy: tuple[float, float]) -> bool
    """
    # Calculate the square of the radius for faster computation
    r2 = r ** 2

    def _f(xy):
        # Unpack the tuple into x and y coordinates
        x, y = xy
        # Check if the squared distance from (x, y) to the center (cx, cy)
        # is less than the squared radius
        return (x - cx) ** 2 + (y - cy) ** 2 < r2
    return _f


# quadratic Bezier lines

def get_cos_sin(x0, y0, x1, y1):
    """
    Calculate the cosine and sine of the angle between the vector (x0, y0)->(x1, y1)
    and the x-axis.

    Parameters
    ----------
    x0, y0 : float
        Coordinates of the starting point of the vector.
    x1, y1 : float
        Coordinates of the ending point of the vector.

    Returns
    -------
    cos_t, sin_t : float
        Cosine and sine of the angle of the vector with respect to the x-axis.
    """
    dx, dy = x1 - x0, y1 - y0
    d = (dx * dx + dy * dy) ** .5
    # Avoid division by zero; return (0.0, 0.0) when d is zero
    if d == 0:
        return 0.0, 0.0
    return dx / d, dy / d


def check_if_parallel(dx1, dy1, dx2, dy2, tolerance=1.e-5):
    """
    Check if two lines defined by their gradients are parallel within a given tolerance.

    Parameters
    ----------
    dx1, dy1, dx2, dy2 : float
        The gradients *dy*/*dx* of the two lines.
    tolerance : float
        The angular tolerance in radians up to which the lines are considered
        parallel.

    Returns
    -------
    is_parallel
        - 1 if two lines are parallel in the same direction.
        - -1 if two lines are parallel in opposite directions.
        - False otherwise.
    """
    theta1 = np.arctan2(dx1, dy1)
    theta2 = np.arctan2(dx2, dy2)
    dtheta = abs(theta1 - theta2)
    if dtheta < tolerance:
        return 1
    elif abs(dtheta - np.pi) < tolerance:
        return -1
    else:
        return False


def get_parallels(bezier2, width):
    """
    Given the quadratic Bézier control points *bezier2*, returns
    control points of quadratic Bézier lines roughly parallel to the given one separated by *width*.
    """

    # Extract the control points from the quadratic Bezier curve
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c2x, c2y = bezier2[2]

    # Check if the start and end segments of the Bezier curve are parallel
    parallel_test = check_if_parallel(c1x - cmx, c1y - cmy,
                                      cmx - c2x, cmy - c2y)

    if parallel_test == -1:
        # Warn if the lines are parallel but do not intersect exactly
        _api.warn_external(
            "Lines do not intersect. A straight line is used instead.")
        # Calculate the cosine and sine of the angles of the tangential lines at c1 and c2
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, c2x, c2y)
        cos_t2, sin_t2 = cos_t1, sin_t1
    else:
        # Calculate the cosine and sine of the angles between c1 and cm, and cm and c2
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
        cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c2x, c2y)

    # Find the control points c1_left, c1_right that lie along lines
    # perpendicular to the tangents of the Bezier path at a distance of width.
    c1x_left, c1y_left, c1x_right, c1y_right = (
        get_normal_points(c1x, c1y, cos_t1, sin_t1, width)
    )
    c2x_left, c2y_left, c2x_right, c2y_right = (
        get_normal_points(c2x, c2y, cos_t2, sin_t2, width)
    )

    # Find the intersecting point of a line through cm perpendicular to the tangents
    #
    # 尝试计算左侧曲线的控制点 cmx_left, cmy_left 和右侧曲线的控制点 cmx_right, cmy_right，
    # 通过调用 get_intersection 函数，传入各自的起始点、角度（cos_t1, sin_t1）、结束点、角度（cos_t2, sin_t2）
    try:
        cmx_left, cmy_left = get_intersection(c1x_left, c1y_left, cos_t1,
                                              sin_t1, c2x_left, c2y_left,
                                              cos_t2, sin_t2)
        cmx_right, cmy_right = get_intersection(c1x_right, c1y_right, cos_t1,
                                                sin_t1, c2x_right, c2y_right,
                                                cos_t2, sin_t2)
    except ValueError:
        # 处理特殊情况，即两条直线的角度小于 get_intersection 函数使用的阈值
        # 这里不使用 check_if_parallel 函数，因为阈值不同
        cmx_left, cmy_left = (
            0.5 * (c1x_left + c2x_left), 0.5 * (c1y_left + c2y_left)
        )
        cmx_right, cmy_right = (
            0.5 * (c1x_right + c2x_right), 0.5 * (c1y_right + c2y_right)
        )

    # 创建平行贝塞尔曲线的控制点路径
    # 左侧路径的控制点为 [c1_left, cm_left, c2_left]
    path_left = [(c1x_left, c1y_left),
                 (cmx_left, cmy_left),
                 (c2x_left, c2y_left)]
    # 右侧路径的控制点为 [c1_right, cm_right, c2_right]
    path_right = [(c1x_right, c1y_right),
                  (cmx_right, cmy_right),
                  (c2x_right, c2y_right)]

    # 返回左侧路径和右侧路径的控制点列表
    return path_left, path_right
# 定义函数，计算 Bézier 曲线的控制点，使曲线经过 (c1x, c1y)，(mmx, mmy)，(c2x, c2y) 这三个点，参数值为 0, 0.5, 1
def find_control_points(c1x, c1y, mmx, mmy, c2x, c2y):
    # 计算控制点 cmx 和 cmy，使得曲线在参数值为 0, 0.5, 1 时经过给定点
    cmx = .5 * (4 * mmx - (c1x + c2x))
    cmy = .5 * (4 * mmy - (c1y + c2y))
    # 返回控制点的列表 [(c1x, c1y), (cmx, cmy), (c2x, c2y)]
    return [(c1x, c1y), (cmx, cmy), (c2x, c2y)]


# 定义函数，生成两条二次 Bézier 曲线的控制点，使其宽度大致与给定宽度平行，并相距 *width*
def make_wedged_bezier2(bezier2, width, w1=1., wm=0.5, w2=0.):
    """
    与 `get_parallels` 类似，返回两条二次 Bézier 曲线的控制点，其宽度大致平行于给定宽度，并相距 *width*。
    """

    # 从 bezier2 中提取控制点 c1, cm, c2
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c3x, c3y = bezier2[2]

    # 计算 c1 到 cm 和 cm 到 c3 的切线角度 t1 和 t2
    cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
    cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c3x, c3y)

    # 计算 c1_left, c1_right，它们位于通过 c1 并且垂直于 Bézier 曲线路径的切线线上，距离为 width * w1
    c1x_left, c1y_left, c1x_right, c1y_right = (
        get_normal_points(c1x, c1y, cos_t1, sin_t1, width * w1)
    )

    # 计算 c3_left, c3_right，它们与 c3 相关，同样位于垂直于其路径的切线上，距离为 width * w2
    c3x_left, c3y_left, c3x_right, c3y_right = (
        get_normal_points(c3x, c3y, cos_t2, sin_t2, width * w2)
    )

    # 计算 c12, c23 和 c123，它们分别是 c1-cm, cm-c3 和 c12-c23 的中点
    c12x, c12y = (c1x + cmx) * .5, (c1y + cmy) * .5
    c23x, c23y = (cmx + c3x) * .5, (cmy + c3y) * .5
    c123x, c123y = (c12x + c23x) * .5, (c12y + c23y) * .5

    # 计算 c123 的切线角度 t123（即 c12 和 c23 之间的角度）
    cos_t123, sin_t123 = get_cos_sin(c12x, c12y, c23x, c23y)

    # 计算 c123_left, c123_right，它们位于通过 c123 并且垂直于其路径的切线线上，距离为 width * wm
    c123x_left, c123y_left, c123x_right, c123y_right = (
        get_normal_points(c123x, c123y, cos_t123, sin_t123, width * wm)
    )

    # 使用 find_control_points 函数计算左右路径的控制点
    path_left = find_control_points(c1x_left, c1y_left,
                                    c123x_left, c123y_left,
                                    c3x_left, c3y_left)
    path_right = find_control_points(c1x_right, c1y_right,
                                     c123x_right, c123y_right,
                                     c3x_right, c3y_right)

    # 返回左右路径的控制点
    return path_left, path_right
```