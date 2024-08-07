# `D:\src\scipysrc\matplotlib\lib\matplotlib\streamplot.py`

```py
"""
Streamline plotting for 2D vector fields.

"""

import numpy as np  # 导入NumPy库，用于处理数组和数值计算

import matplotlib as mpl  # 导入Matplotlib库的主模块
from matplotlib import _api, cm, patches  # 导入Matplotlib中的私有API、颜色映射和图形绘制相关模块
import matplotlib.colors as mcolors  # 导入Matplotlib中的颜色模块
import matplotlib.collections as mcollections  # 导入Matplotlib中的集合模块
import matplotlib.lines as mlines  # 导入Matplotlib中的线条模块


__all__ = ['streamplot']  # 定义本模块对外暴露的函数名


def streamplot(axes, x, y, u, v, density=1, linewidth=None, color=None,
               cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
               minlength=0.1, transform=None, zorder=None, start_points=None,
               maxlength=4.0, integration_direction='both',
               broken_streamlines=True):
    """
    Draw streamlines of a vector flow.

    Parameters
    ----------
    x, y : 1D/2D arrays
        Evenly spaced strictly increasing arrays to make a grid.  If 2D, all
        rows of *x* must be equal and all columns of *y* must be equal; i.e.,
        they must be as if generated by ``np.meshgrid(x_1d, y_1d)``.
    u, v : 2D arrays
        *x* and *y*-velocities. The number of rows and columns must match
        the length of *y* and *x*, respectively.
    density : float or (float, float)
        Controls the closeness of streamlines. When ``density = 1``, the domain
        is divided into a 30x30 grid. *density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use a tuple
        (density_x, density_y).
    linewidth : float or 2D array
        The width of the streamlines. With a 2D array the line width can be
        varied across the grid. The array must have the same shape as *u*
        and *v*.
    color : :mpltype:`color` or 2D array
        The streamline color. If given an array, its values are converted to
        colors using *cmap* and *norm*.  The array must have the same shape
        as *u* and *v*.
    cmap, norm
        Data normalization and colormapping parameters for *color*; only used
        if *color* is an array of floats. See `~.Axes.imshow` for a detailed
        description.
    arrowsize : float
        Scaling factor for the arrow size.
    arrowstyle : str
        Arrow style specification.
        See `~matplotlib.patches.FancyArrowPatch`.
    minlength : float
        Minimum length of streamline in axes coordinates.
    start_points : (N, 2) array
        Coordinates of starting points for the streamlines in data coordinates
        (the same coordinates as the *x* and *y* arrays).
    zorder : float
        The zorder of the streamlines and arrows.
        Artists with lower zorder values are drawn first.
    maxlength : float
        Maximum length of streamline in axes coordinates.
    integration_direction : {'forward', 'backward', 'both'}, default: 'both'
        Integrate the streamline in forward, backward or both directions.
    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER
    """
    # 实现矢量流场的流线绘制

    # 下面的代码实现具体的绘图功能，参数详细说明见上方的函数说明文档
    pass
    broken_streamlines : boolean, default: True
        如果为 False，则强制流线延续直到离开绘图域。如果为 True，则可能会在流线接近时被终止。

    Returns
    -------
    StreamplotSet
        包含以下属性的容器对象：

        - ``lines``: 包含流线的 `.LineCollection`

        - ``arrows``: 包含半路箭头的 `.PatchCollection`，其中包含 `.FancyArrowPatch` 对象，表示流线的箭头。

        该容器对象未来可能会更改，以允许对线条和箭头的色图、透明度等进行修改，但这些变化应该是向后兼容的。
    """
    grid = Grid(x, y)  # 创建网格对象，使用给定的 x 和 y 坐标
    mask = StreamMask(density)  # 创建流线掩码对象，使用给定的密度
    dmap = DomainMap(grid, mask)  # 创建域映射对象，使用给定的网格和掩码信息

    if zorder is None:
        zorder = mlines.Line2D.zorder  # 如果未指定 zorder，则使用默认的线条 zorder

    # 默认使用数据坐标系
    if transform is None:
        transform = axes.transData  # 如果未指定 transform，则使用 axes 的数据坐标变换

    if color is None:
        color = axes._get_lines.get_next_color()  # 如果未指定 color，则获取下一个线条颜色

    if linewidth is None:
        linewidth = mpl.rcParams['lines.linewidth']  # 如果未指定 linewidth，则使用默认的线条宽度设置

    line_kw = {}  # 初始化线条关键字参数字典
    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)  # 初始化箭头关键字参数字典

    _api.check_in_list(['both', 'forward', 'backward'],
                       integration_direction=integration_direction)  # 检查 integration_direction 是否有效

    if integration_direction == 'both':
        maxlength /= 2.  # 如果 integration_direction 是 'both'，则将最大长度 maxlength 减半

    use_multicolor_lines = isinstance(color, np.ndarray)  # 检查 color 是否为 ndarray 类型
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError("If 'color' is given, it must match the shape of "
                             "the (x, y) grid")  # 如果 color 不匹配网格的形状，则引发 ValueError
        line_colors = [[]]  # 空条目允许连接零数组。
        color = np.ma.masked_invalid(color)  # 屏蔽无效的 color 值
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError("If 'linewidth' is given, it must match the "
                             "shape of the (x, y) grid")  # 如果 linewidth 不匹配网格的形状，则引发 ValueError
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder  # 设置线条 zorder
    arrow_kw['zorder'] = zorder  # 设置箭头 zorder

    # 检查合法性。
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' and 'v' must match the shape of the (x, y) grid")  # 如果 u 和 v 不匹配网格的形状，则引发 ValueError

    u = np.ma.masked_invalid(u)  # 屏蔽无效的 u 值
    v = np.ma.masked_invalid(v)  # 屏蔽无效的 v 值

    integrate = _get_integrator(u, v, dmap, minlength, maxlength,
                                integration_direction)  # 获取积分器对象

    trajectories = []  # 初始化轨迹列表
    if start_points is None:
        for xm, ym in _gen_starting_points(mask.shape):  # 遍历生成的起始点
            if mask[ym, xm] == 0:  # 如果起始点未被掩码覆盖
                xg, yg = dmap.mask2grid(xm, ym)  # 将掩码坐标映射到网格坐标
                t = integrate(xg, yg, broken_streamlines)  # 对流线进行积分
                if t is not None:
                    trajectories.append(t)  # 将有效的轨迹添加到轨迹列表中
    else:
        # 将 start_points 转换为浮点数类型的 NumPy 数组，并进行复制
        sp2 = np.asanyarray(start_points, dtype=float).copy()

        # 检查 start_points 是否超出数据边界
        for xs, ys in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width and
                    grid.y_origin <= ys <= grid.y_origin + grid.height):
                raise ValueError(f"Starting point ({xs}, {ys}) outside of "
                                 "data boundaries")

        # 将 start_points 从数据坐标系转换为数组坐标系
        # 通过将种子点从数据的左下角偏移，使得 data2grid 函数能够正常工作
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin

        for xs, ys in sp2:
            # 将数据坐标转换为网格坐标
            xg, yg = dmap.data2grid(xs, ys)

            # 浮点数精度问题可能导致 xg, yg 略微超出 xs, ys 的上界
            # 由于已经检查了起始点是否在原始网格内部，因此将 xg, yg 修剪到网格范围内以解决此问题
            xg = np.clip(xg, 0, grid.nx - 1)
            yg = np.clip(yg, 0, grid.ny - 1)

            # 对路径进行积分计算
            t = integrate(xg, yg, broken_streamlines)
            if t is not None:
                trajectories.append(t)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        cmap = cm._ensure_cmap(cmap)

    # 初始化流线和箭头列表
    streamlines = []
    arrows = []

    # 遍历每条轨迹
    for t in trajectories:
        tgx, tgy = t.T
        # 将网格坐标重新缩放为数据坐标
        tx, ty = dmap.grid2data(tgx, tgy)
        tx += grid.x_origin
        ty += grid.y_origin

        # 如果需要变宽或多彩线条，创建多个小线段
        if isinstance(linewidth, np.ndarray) or use_multicolor_lines:
            points = np.transpose([tx, ty]).reshape(-1, 1, 2)
            streamlines.extend(np.hstack([points[:-1], points[1:]]))
        else:
            points = np.transpose([tx, ty])
            streamlines.append(points)

        # 在每条轨迹的中间添加箭头
        s = np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))
        n = np.searchsorted(s, s[-1] / 2.)
        arrow_tail = (tx[n], ty[n])
        arrow_head = (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2]))

        # 如果线宽是数组，使用插值网格进行线宽计算
        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        # 如果使用多彩线条，使用插值网格进行颜色计算
        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))

        # 创建箭头对象并添加到箭头列表
        p = patches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, **arrow_kw)
        arrows.append(p)

    # 创建线条集合对象
    lc = mcollections.LineCollection(
        streamlines, transform=transform, **line_kw)
    # 设置水平粘滞边界，限制在网格水平方向的起点和终点
    lc.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
    # 设置垂直粘滞边界，限制在网格垂直方向的起点和终点
    lc.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
    
    # 如果使用多彩线条，则设置线条的颜色数组、颜色映射和规范化方式
    if use_multicolor_lines:
        lc.set_array(np.ma.hstack(line_colors))  # 设置线条颜色数组
        lc.set_cmap(cmap)  # 设置颜色映射
        lc.set_norm(norm)  # 设置数据规范化方式
    
    # 将 LineCollection 对象 lc 添加到图形 axes 中
    axes.add_collection(lc)

    # 创建一个 PatchCollection 对象 ac，包含箭头列表 arrows 中的所有箭头
    ac = mcollections.PatchCollection(arrows)
    
    # 由于存在已知问题，暂时无法直接添加 PatchCollection 对象；参见 issue #2341。
    # 所以需要逐个添加 arrows 列表中的每个箭头到 axes 中
    for p in arrows:
        axes.add_patch(p)
    
    # 自动调整 axes 视图范围
    axes.autoscale_view()
    
    # 创建一个 StreamplotSet 对象 stream_container，包含 LineCollection 和 PatchCollection
    stream_container = StreamplotSet(lc, ac)
    
    # 返回 stream_container 对象作为函数结果
    return stream_container
class StreamplotSet:
    # 定义包含流线和箭头的集合的类

    def __init__(self, lines, arrows):
        # 初始化方法，接受流线和箭头作为参数，并将它们存储在实例变量中
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================

class DomainMap:
    """
    Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid, mask):
        # 初始化方法，接受网格和掩模对象作为参数

        self.grid = grid
        self.mask = mask

        # Constants for conversion between grid- and mask-coordinates
        # 用于网格坐标与掩模坐标之间转换的常量
        self.x_grid2mask = (mask.nx - 1) / (grid.nx - 1)
        self.y_grid2mask = (mask.ny - 1) / (grid.ny - 1)

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def grid2mask(self, xi, yi):
        """Return nearest space in mask-coords from given grid-coords."""
        # 将给定的网格坐标转换为最接近的掩模坐标空间
        return round(xi * self.x_grid2mask), round(yi * self.y_grid2mask)

    def mask2grid(self, xm, ym):
        # 将给定的掩模坐标转换为网格坐标
        return xm * self.x_mask2grid, ym * self.y_mask2grid

    def data2grid(self, xd, yd):
        # 将给定的数据坐标转换为网格坐标
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        # 将给定的网格坐标转换为数据坐标
        return xg / self.x_data2grid, yg / self.y_data2grid

    def start_trajectory(self, xg, yg, broken_streamlines=True):
        # 开始轨迹方法，将网格坐标转换为掩模坐标，并在 StreamMask 中跟踪轨迹
        xm, ym = self.grid2mask(xg, yg)
        self.mask._start_trajectory(xm, ym, broken_streamlines)

    def reset_start_point(self, xg, yg):
        # 重置起始点方法，将网格坐标转换为掩模坐标
        xm, ym = self.grid2mask(xg, yg)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, xg, yg, broken_streamlines=True):
        # 更新轨迹方法，检查是否在网格内，将网格坐标转换为掩模坐标，并更新轨迹
        if not self.grid.within_grid(xg, yg):
            raise InvalidIndexError
        xm, ym = self.grid2mask(xg, yg)
        self.mask._update_trajectory(xm, ym, broken_streamlines)

    def undo_trajectory(self):
        # 撤销轨迹方法，调用 StreamMask 中的撤销轨迹方法
        self.mask._undo_trajectory()


class Grid:
    """Grid of data."""
    def __init__(self, x, y):
        # 检查输入的 x 是否为一维数组
        if np.ndim(x) == 1:
            pass
        # 检查输入的 x 是否为二维数组，并且所有行是否相等
        elif np.ndim(x) == 2:
            x_row = x[0]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        # 检查输入的 y 是否为一维数组
        if np.ndim(y) == 1:
            pass
        # 检查输入的 y 是否为二维数组，并且所有列是否相等
        elif np.ndim(y) == 2:
            yt = np.transpose(y)  # 支持嵌套列表。
            y_col = yt[0]
            if not np.allclose(y_col, yt):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        # 检查 x 是否严格递增
        if not (np.diff(x) > 0).all():
            raise ValueError("'x' must be strictly increasing")
        # 检查 y 是否严格递增
        if not (np.diff(y) > 0).all():
            raise ValueError("'y' must be strictly increasing")

        # 设置网格的尺寸属性
        self.nx = len(x)
        self.ny = len(y)

        # 计算 x 和 y 的步长
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        # 设置网格的原点位置
        self.x_origin = x[0]
        self.y_origin = y[0]

        # 计算网格的宽度和高度
        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]

        # 检查 x 和 y 的值是否等间隔
        if not np.allclose(np.diff(x), self.width / (self.nx - 1)):
            raise ValueError("'x' values must be equally spaced")
        if not np.allclose(np.diff(y), self.height / (self.ny - 1)):
            raise ValueError("'y' values must be equally spaced")

    @property
    def shape(self):
        # 返回网格的形状，即 (ny, nx)
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return whether (*xi*, *yi*) is a valid index of the grid."""
        # 检查给定的索引 (xi, yi) 是否在网格内
        # 注意，xi 和 yi 可能是浮点数；因此不能简单地检查 `xi < self.nx`，因为可能有 `self.nx - 1 < xi < self.nx`
        return 0 <= xi <= self.nx - 1 and 0 <= yi <= self.ny - 1
class StreamMask:
    """
    Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        try:
            # Calculate grid dimensions based on density
            self.nx, self.ny = (30 * np.broadcast_to(density, 2)).astype(int)
        except ValueError as err:
            raise ValueError("'density' must be a scalar or be of length "
                             "2") from err
        if self.nx < 0 or self.ny < 0:
            raise ValueError("'density' must be positive")
        # Initialize a grid mask with zeros
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, args):
        # Retrieve item from the mask grid
        return self._mask[args]

    def _start_trajectory(self, xm, ym, broken_streamlines=True):
        """Start recording streamline trajectory"""
        # Initialize an empty list to store trajectory points
        self._traj = []
        # Update the trajectory with initial coordinates
        self._update_trajectory(xm, ym, broken_streamlines)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        # Reset mask values for all points in the current trajectory
        for t in self._traj:
            self._mask[t] = 0

    def _update_trajectory(self, xm, ym, broken_streamlines=True):
        """
        Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xy != (xm, ym):
            # Check if the new position is available (zeroed)
            if self[ym, xm] == 0:
                # Add the new position to the trajectory list
                self._traj.append((ym, xm))
                # Mark the position in the mask as filled (1)
                self._mask[ym, xm] = 1
                self._current_xy = (xm, ym)
            else:
                # If the position is already filled
                if broken_streamlines:
                    raise InvalidIndexError
                else:
                    pass


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
# =======================

def _get_integrator(u, v, dmap, minlength, maxlength, integration_direction):

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)

    # speed (path length) will be in axes-coordinates
    u_ax = u / (dmap.grid.nx - 1)
    v_ax = v / (dmap.grid.ny - 1)
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2)

    def forward_time(xi, yi):
        if not dmap.grid.within_grid(xi, yi):
            raise OutOfBounds
        # Interpolate speed from grid
        ds_dt = interpgrid(speed, xi, yi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        # Interpolate velocities from grid
        ui = interpgrid(u, xi, yi)
        vi = interpgrid(v, xi, yi)
        return ui * dt_ds, vi * dt_ds

    def backward_time(xi, yi):
        # Calculate backward time step
        dxi, dyi = forward_time(xi, yi)
        return -dxi, -dyi
    def integrate(x0, y0, broken_streamlines=True):
        """
        Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        # 初始化轨迹长度和坐标点列表
        stotal, xy_traj = 0., []

        # 尝试开始轨迹
        try:
            dmap.start_trajectory(x0, y0, broken_streamlines)
        except InvalidIndexError:
            # 如果起始点无效则返回空
            return None
        
        # 如果集成方向是 'both' 或 'backward'
        if integration_direction in ['both', 'backward']:
            # 向后积分，使用 RK12 方法
            s, xyt = _integrate_rk12(x0, y0, dmap, backward_time, maxlength,
                                     broken_streamlines)
            stotal += s  # 累加积分长度
            xy_traj += xyt[::-1]  # 反向添加轨迹点

        # 如果集成方向是 'both' 或 'forward'
        if integration_direction in ['both', 'forward']:
            # 重置起始点
            dmap.reset_start_point(x0, y0)
            # 向前积分，使用 RK12 方法
            s, xyt = _integrate_rk12(x0, y0, dmap, forward_time, maxlength,
                                     broken_streamlines)
            stotal += s  # 累加积分长度
            xy_traj += xyt[1:]  # 正向添加轨迹点

        # 如果轨迹长度大于最小长度 minlength
        if stotal > minlength:
            # 返回广播后的轨迹数组
            return np.broadcast_arrays(xy_traj, np.empty((1, 2)))[0]
        else:  # 拒绝过短的轨迹
            dmap.undo_trajectory()  # 撤销轨迹
            return None

    # 返回积分函数 integrate
    return integrate
# 定义一个自定义异常类，用于表示索引超出边界的错误
class OutOfBounds(IndexError):
    pass

# 定义一个函数，实现二阶龙格-库塔法（Runge-Kutta方法）进行数值积分，同时具备自适应步长功能
def _integrate_rk12(x0, y0, dmap, f, maxlength, broken_streamlines=True):
    """
    2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as RK4 and RK45 solvers (using
    similar Python implementations) in most setups.
    """

    # 这个错误限制设置得比RK4积分器所需的更低。这是为了视觉上的原因 -- 如果设置得太低，角落会显得不自然和锯齿状。可以进行调整。
    maxerror = 0.003

    # 这个限制对于所有的积分器都很重要，以避免轨迹跳过某些掩膜单元。如果使用下面注释掉的代码逐步增加位置，可以放宽此条件。
    # 然而，由于插值的高效性质，这并不能显著提高速度，但增加了复杂性。
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.1)

    # 初始步长设为最大步长
    ds = maxds
    # 积分总长度初始化为0
    stotal = 0
    # 初始位置设为起始点
    xi = x0
    yi = y0
    # 存储轨迹点的列表
    xyf_traj = []
    while True:
        try:
            # 如果(xi, yi)在网格内部，则将(xi, yi)添加到轨迹列表中
            if dmap.grid.within_grid(xi, yi):
                xyf_traj.append((xi, yi))
            else:
                # 如果(xi, yi)不在网格内部，则抛出OutOfBounds异常
                raise OutOfBounds

            # 计算两个中间梯度值。
            # 如果给定的位置超出网格范围，函数f应该抛出OutOfBounds异常。
            k1x, k1y = f(xi, yi)
            k2x, k2y = f(xi + ds * k1x, yi + ds * k1y)

        except OutOfBounds:
            # 在这一步时超出了定义域。
            # 如果轨迹当前不为空，则采用欧拉步进到边界以提高整洁度。
            if xyf_traj:
                ds, xyf_traj = _euler_step(xyf_traj, dmap, f)
                stotal += ds
            break
        except TerminateTrajectory:
            break

        # 计算四个方向的位移量
        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)

        ny, nx = dmap.grid.shape
        # 计算误差，标准化为轴坐标
        error = np.hypot((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1))

        # 只有当误差小于最大误差容许值时保存这一步
        if error < maxerror:
            xi += dx2
            yi += dy2
            try:
                # 更新轨迹，处理断开的流线
                dmap.update_trajectory(xi, yi, broken_streamlines)
            except InvalidIndexError:
                break
            if stotal + ds > maxlength:
                break
            stotal += ds

        # 根据步长误差重新计算步长大小
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xyf_traj
def _euler_step(xyf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    # 获取网格的形状，ny为行数，nx为列数
    ny, nx = dmap.grid.shape
    # 获取当前轨迹的最后一个点的坐标
    xi, yi = xyf_traj[-1]
    # 计算在当前点上的速度向量
    cx, cy = f(xi, yi)
    # 根据速度向量计算横向和纵向可以行进的最大距离
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    # 取横向和纵向可行进距离的最小值作为步长
    ds = min(dsx, dsy)
    # 根据步长更新轨迹点并返回步长和更新后的轨迹
    xyf_traj.append((xi + cx * ds, yi + cy * ds))
    return ds, xyf_traj


# Utility functions
# ========================

def interpgrid(a, xi, yi):
    """Fast 2D, linear interpolation on an integer grid"""
    # 获取数组a的形状
    Ny, Nx = np.shape(a)
    if isinstance(xi, np.ndarray):
        # 如果xi是数组，将其转换为整数类型
        x = xi.astype(int)
        y = yi.astype(int)
        # 检查xn, yn不超过最大索引值
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
    else:
        # 如果xi是单个值，将其转换为整数类型
        x = int(xi)
        y = int(yi)
        # 对于整数，条件语句比剪裁更快
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1

    # 根据插值公式计算插值结果
    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]
    xt = xi - x
    yt = yi - y
    a0 = a00 * (1 - xt) + a01 * xt
    a1 = a10 * (1 - xt) + a11 * xt
    ai = a0 * (1 - yt) + a1 * yt

    # 如果xi不是数组并且ai是掩码，抛出终止轨迹异常
    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai):
            raise TerminateTrajectory

    return ai


def _gen_starting_points(shape):
    """
    Yield starting points for streamlines.

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    # 获取网格的形状
    ny, nx = shape
    # 定义起始点的初始位置和方向
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = 0, 0
    direction = 'right'
    # 生成起始点序列的迭代器
    for i in range(nx * ny):
        yield x, y

        # 根据当前方向更新下一个点的位置
        if direction == 'right':
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = 'right'
```