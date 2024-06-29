# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\axis3d.py`

```py
# axis3d.py, original mplot3d version by John Porter
# Created: 23 Sep 2005
# Parts rewritten by Reinier Heeres <reinier@heeres.eu>

import inspect  # 导入inspect模块，用于获取对象信息

import numpy as np  # 导入NumPy库，用于数值计算

import matplotlib as mpl  # 导入Matplotlib库的主模块
from matplotlib import (  # 从Matplotlib中导入以下子模块和类别
    _api, artist, lines as mlines, axis as maxis, patches as mpatches,
    transforms as mtransforms, colors as mcolors)
from . import art3d, proj3d  # 从当前包导入art3d和proj3d模块

def _move_from_center(coord, centers, deltas, axmask=(True, True, True)):
    """
    For each coordinate where *axmask* is True, move *coord* away from
    *centers* by *deltas*.
    """
    coord = np.asarray(coord)  # 将coord转换为NumPy数组
    return coord + axmask * np.copysign(1, coord - centers) * deltas  # 根据axmask条件调整coord的位置

def _tick_update_position(tick, tickxs, tickys, labelpos):
    """Update tick line and label position and style."""
    tick.label1.set_position(labelpos)  # 设置主刻度标签位置
    tick.label2.set_position(labelpos)  # 设置次刻度标签位置
    tick.tick1line.set_visible(True)  # 设置主刻度线可见
    tick.tick2line.set_visible(False)  # 设置次刻度线不可见
    tick.tick1line.set_linestyle('-')  # 设置主刻度线样式为实线
    tick.tick1line.set_marker('')  # 设置主刻度线无标记
    tick.tick1line.set_data(tickxs, tickys)  # 设置主刻度线的数据
    tick.gridline.set_data([0], [0])  # 设置网格线数据为原点位置

class Axis(maxis.XAxis):  # 定义Axis类，继承自maxis.XAxis类
    """An Axis class for the 3D plots."""
    # These points from the unit cube make up the x, y and z-planes
    _PLANES = (
        (0, 3, 7, 4), (1, 2, 6, 5),  # yz平面
        (0, 1, 5, 4), (3, 2, 6, 7),  # xz平面
        (0, 1, 2, 3), (4, 5, 6, 7),  # xy平面
    )

    # Some properties for the axes
    _AXINFO = {
        'x': {'i': 0, 'tickdir': 1, 'juggled': (1, 0, 2)},
        'y': {'i': 1, 'tickdir': 0, 'juggled': (0, 1, 2)},
        'z': {'i': 2, 'tickdir': 0, 'juggled': (0, 2, 1)},
    }

    def _old_init(self, adir, v_intervalx, d_intervalx, axes, *args,
                  rotate_label=None, **kwargs):
        return locals()  # 返回当前作用域的所有局部变量字典

    def _new_init(self, axes, *, rotate_label=None, **kwargs):
        return locals()  # 返回当前作用域的所有局部变量字典

    __init__.__signature__ = inspect.signature(_new_init)  # 设置构造函数的签名为_new_init的签名
    adir = _api.deprecated("3.6", pending=True)(  # 使用API装饰器标记adir属性为已弃用
        property(lambda self: self.axis_name))  # 返回axis_name属性的值

    def _init3d(self):
        self.line = mlines.Line2D(  # 创建2D线对象line
            xdata=(0, 0), ydata=(0, 0),  # 设置线的x和y数据
            linewidth=self._axinfo['axisline']['linewidth'],  # 设置线宽度
            color=self._axinfo['axisline']['color'],  # 设置线颜色
            antialiased=True)  # 开启抗锯齿

        # Store dummy data in Polygon object
        self.pane = mpatches.Polygon([[0, 0], [0, 1]], closed=False)  # 创建多边形对象pane
        self.set_pane_color(self._axinfo['color'])  # 设置pane的颜色

        self.axes._set_artist_props(self.line)  # 在axes上设置line的属性
        self.axes._set_artist_props(self.pane)  # 在axes上设置pane的属性
        self.gridlines = art3d.Line3DCollection([])  # 创建3D线集合对象gridlines
        self.axes._set_artist_props(self.gridlines)  # 在axes上设置gridlines的属性
        self.axes._set_artist_props(self.label)  # 在axes上设置label的属性
        self.axes._set_artist_props(self.offsetText)  # 在axes上设置offsetText的属性
        # Need to be able to place the label at the correct location
        self.label._transform = self.axes.transData  # 设置label的变换为数据坐标系
        self.offsetText._transform = self.axes.transData  # 设置offsetText的变换为数据坐标系

    @_api.deprecated("3.6", pending=True)
    # 在不推荐使用 _init3d 后，将其内联到 __init__ 方法中
    def init3d(self):  # After deprecation elapses, inline _init3d to __init__.
        self._init3d()

    # 获取主要刻度线
    def get_major_ticks(self, numticks=None):
        # 调用父类方法获取主要刻度线
        ticks = super().get_major_ticks(numticks)
        # 遍历每个刻度线对象
        for t in ticks:
            # 设置刻度线、网格线和标签对象的坐标变换为当前坐标系的数据坐标变换
            for obj in [
                    t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
                obj.set_transform(self.axes.transData)
        return ticks

    # 获取次要刻度线
    def get_minor_ticks(self, numticks=None):
        # 调用父类方法获取次要刻度线
        ticks = super().get_minor_ticks(numticks)
        # 遍历每个刻度线对象
        for t in ticks:
            # 设置刻度线、网格线和标签对象的坐标变换为当前坐标系的数据坐标变换
            for obj in [
                    t.tick1line, t.tick2line, t.gridline, t.label1, t.label2]:
                obj.set_transform(self.axes.transData)
        return ticks

    # 设置刻度线位置
    def set_ticks_position(self, position):
        """
        Set the ticks position.

        Parameters
        ----------
        position : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the bolded axis lines, ticks, and tick labels.
        """
        # 如果位置为 'top' 或 'bottom'，则发出警告并返回
        if position in ['top', 'bottom']:
            _api.warn_deprecated('3.8', name=f'{position=}',
                                 obj_type='argument value',
                                 alternative="'upper' or 'lower'")
            return
        # 检查位置参数是否在预定义列表中
        _api.check_in_list(['lower', 'upper', 'both', 'default', 'none'],
                           position=position)
        # 设置内部变量 _tick_position 为指定位置
        self._tick_position = position

    # 获取刻度线位置
    def get_ticks_position(self):
        """
        Get the ticks position.

        Returns
        -------
        str : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the bolded axis lines, ticks, and tick labels.
        """
        # 返回存储的刻度线位置信息
        return self._tick_position

    # 设置标签位置
    def set_label_position(self, position):
        """
        Set the label position.

        Parameters
        ----------
        position : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the axis label.
        """
        # 如果位置为 'top' 或 'bottom'，则发出警告并返回
        if position in ['top', 'bottom']:
            _api.warn_deprecated('3.8', name=f'{position=}',
                                 obj_type='argument value',
                                 alternative="'upper' or 'lower'")
            return
        # 检查位置参数是否在预定义列表中
        _api.check_in_list(['lower', 'upper', 'both', 'default', 'none'],
                           position=position)
        # 设置内部变量 _label_position 为指定位置
        self._label_position = position

    # 获取标签位置
    def get_label_position(self):
        """
        Get the label position.

        Returns
        -------
        str : {'lower', 'upper', 'both', 'default', 'none'}
            The position of the axis label.
        """
        # 返回存储的标签位置信息
        return self._label_position
    def set_pane_color(self, color, alpha=None):
        """
        设置面板颜色。

        Parameters
        ----------
        color : :mpltype:`color`
            坐标轴面板的颜色。
        alpha : float, optional
            坐标轴面板的透明度值。如果为 None，则基于 *color* 设置透明度。
        """
        # 将颜色转换为 RGBA 格式
        color = mcolors.to_rgba(color, alpha)
        # 设置面板的颜色
        self._axinfo['color'] = color
        self.pane.set_edgecolor(color)
        self.pane.set_facecolor(color)
        self.pane.set_alpha(color[-1])
        self.stale = True

    def set_rotate_label(self, val):
        """
        设置是否旋转坐标轴标签：True、False 或 None。
        如果设置为 None，则当标签长度超过 4 个字符时进行旋转。
        """
        self._rotate_label = val
        self.stale = True

    def get_rotate_label(self, text):
        """
        获取是否旋转坐标轴标签的设定。
        """
        if self._rotate_label is not None:
            return self._rotate_label
        else:
            return len(text) > 4

    def _get_coord_info(self):
        mins, maxs = np.array([
            self.axes.get_xbound(),
            self.axes.get_ybound(),
            self.axes.get_zbound(),
        ]).T

        # 投影当前立方体位置上的边界：
        bounds = mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]
        bounds_proj = self.axes._transformed_cube(bounds)

        # 确定哪一个平行面更高：
        means_z0 = np.zeros(3)
        means_z1 = np.zeros(3)
        for i in range(3):
            means_z0[i] = np.mean(bounds_proj[self._PLANES[2 * i], 2])
            means_z1[i] = np.mean(bounds_proj[self._PLANES[2 * i + 1], 2])
        highs = means_z0 < means_z1

        # 对边缘视图进行特殊处理
        equals = np.abs(means_z0 - means_z1) <= np.finfo(float).eps
        if np.sum(equals) == 2:
            vertical = np.where(~equals)[0][0]
            if vertical == 2:  # 查看 XY 平面
                highs = np.array([True, True, highs[2]])
            elif vertical == 1:  # 查看 XZ 平面
                highs = np.array([True, highs[1], False])
            elif vertical == 0:  # 查看 YZ 平面
                highs = np.array([highs[0], False, False])

        return mins, maxs, bounds_proj, highs

    def _calc_centers_deltas(self, maxs, mins):
        """
        计算中心点和变化量。

        mpl3.8 中，比例因子为 1/12。mpl3.9 为保持外观一致，将此调整为
        1/12 * 24/25 = 0.08。24/25 的因子来自于 mpl3.8 中每个轴添加的 1/48
        的填充。
        """
        centers = 0.5 * (maxs + mins)
        scale = 0.08  # 比例因子
        deltas = (maxs - mins) * scale
        return centers, deltas
    def _get_axis_line_edge_points(self, minmax, maxmin, position=None):
        """Get the edge points for the black bolded axis line."""
        # 当改变垂直轴时，某些轴必须移动到另一个平面，以便看起来与 z 轴为垂直轴时一样。
        mb = [minmax, maxmin]  # 从原点到最靠近摄像机的角的线段
        mb_rev = mb[::-1]
        mm = [[mb, mb_rev, mb_rev], [mb_rev, mb_rev, mb], [mb, mb, mb]]
        mm = mm[self.axes._vertical_axis][self._axinfo["i"]]

        juggled = self._axinfo["juggled"]
        edge_point_0 = mm[0].copy()  # 原点

        if ((position == 'lower' and mm[1][juggled[-1]] < mm[0][juggled[-1]]) or
                (position == 'upper' and mm[1][juggled[-1]] > mm[0][juggled[-1]])):
            edge_point_0[juggled[-1]] = mm[1][juggled[-1]]
        else:
            edge_point_0[juggled[0]] = mm[1][juggled[0]]

        edge_point_1 = edge_point_0.copy()
        edge_point_1[juggled[1]] = mm[1][juggled[1]]

        return edge_point_0, edge_point_1

    def _get_all_axis_line_edge_points(self, minmax, maxmin, axis_position=None):
        """Determine edge points for the axis lines across all positions."""
        edgep1s = []
        edgep2s = []
        position = []
        if axis_position in (None, 'default'):
            edgep1, edgep2 = self._get_axis_line_edge_points(minmax, maxmin)
            edgep1s = [edgep1]
            edgep2s = [edgep2]
            position = ['default']
        else:
            edgep1_l, edgep2_l = self._get_axis_line_edge_points(minmax, maxmin,
                                                                 position='lower')
            edgep1_u, edgep2_u = self._get_axis_line_edge_points(minmax, maxmin,
                                                                 position='upper')
            if axis_position in ('lower', 'both'):
                edgep1s.append(edgep1_l)
                edgep2s.append(edgep2_l)
                position.append('lower')
            if axis_position in ('upper', 'both'):
                edgep1s.append(edgep1_u)
                edgep2s.append(edgep2_u)
                position.append('upper')
        return edgep1s, edgep2s, position
    def _get_tickdir(self, position):
        """
        Get the direction of the tick.

        Parameters
        ----------
        position : str, optional : {'upper', 'lower', 'default'}
            The position of the axis.

        Returns
        -------
        tickdir : int
            Index which indicates which coordinate the tick line will
            align with.
        """
        # 检查 position 是否在指定的列表中
        _api.check_in_list(('upper', 'lower', 'default'), position=position)

        # 获取所有子图的 tickdir 默认值
        tickdirs_base = [v["tickdir"] for v in self._AXINFO.values()]  # default

        # 计算方位角和俯仰角的模
        elev_mod = np.mod(self.axes.elev + 180, 360) - 180
        azim_mod = np.mod(self.axes.azim, 360)

        # 根据 position 不同设置 tickdirs_base 的值
        if position == 'upper':
            if elev_mod >= 0:
                tickdirs_base = [2, 2, 0]
            else:
                tickdirs_base = [1, 0, 0]
            if 0 <= azim_mod < 180:
                tickdirs_base[2] = 1
        elif position == 'lower':
            if elev_mod >= 0:
                tickdirs_base = [1, 0, 1]
            else:
                tickdirs_base = [2, 2, 1]
            if 0 <= azim_mod < 180:
                tickdirs_base[2] = 0

        # 获取当前子图的信息列表
        info_i = [v["i"] for v in self._AXINFO.values()]

        # 计算 tickdir 的值
        i = self._axinfo["i"]
        vert_ax = self.axes._vertical_axis
        j = vert_ax - 2
        tickdir = np.roll(info_i, -j)[np.roll(tickdirs_base, j)][i]
        return tickdir

    def active_pane(self):
        """
        Determine the active pane's coordinates and location.

        Returns
        -------
        xys : numpy.ndarray
            Array of coordinates defining the active pane.
        loc : float
            Location of the active pane.
        """
        mins, maxs, tc, highs = self._get_coord_info()
        info = self._axinfo
        index = info['i']

        # 根据 highs[index] 的值确定 loc 和 plane 的值
        if not highs[index]:
            loc = mins[index]
            plane = self._PLANES[2 * index]
        else:
            loc = maxs[index]
            plane = self._PLANES[2 * index + 1]

        # 获取平面的坐标数组 xys
        xys = np.array([tc[p] for p in plane])
        return xys, loc

    def draw_pane(self, renderer):
        """
        Draw the 3D pane using the provided renderer.

        Parameters
        ----------
        renderer : `~matplotlib.backend_bases.RendererBase` subclass
            Renderer object used for drawing.
        """
        # 开启绘制组 'pane3d'，使用对象的 GID 标识
        renderer.open_group('pane3d', gid=self.get_gid())
        
        # 获取当前活动平面的坐标和位置
        xys, loc = self.active_pane()
        
        # 设置绘制对象的坐标
        self.pane.xy = xys[:, :2]
        
        # 使用渲染器绘制平面
        self.pane.draw(renderer)
        
        # 关闭绘制组 'pane3d'
        renderer.close_group('pane3d')

    def _axmask(self):
        """
        Generate a mask indicating the active axis.

        Returns
        -------
        axmask : list of bool
            Mask indicating which axes are active.
        """
        # 创建一个包含三个 True 元素的列表
        axmask = [True, True, True]
        
        # 将当前子图的索引对应位置设为 False
        axmask[self._axinfo["i"]] = False
        return axmask
    def _draw_ticks(self, renderer, edgep1, centers, deltas, highs,
                    deltas_per_point, pos):
        # 获取当前轴上的刻度对象
        ticks = self._update_ticks()
        # 获取当前轴的信息
        info = self._axinfo
        # 获取当前轴的索引和是否交换标志
        index = info["i"]
        juggled = info["juggled"]

        # 获取坐标轴的最小值、最大值、轴线颜色和高度
        mins, maxs, tc, highs = self._get_coord_info()
        # 计算刻度的中心位置和间距
        centers, deltas = self._calc_centers_deltas(maxs, mins)

        # 绘制刻度:
        # 获取刻度方向
        tickdir = self._get_tickdir(pos)
        # 根据高度确定刻度的增量
        tickdelta = deltas[tickdir] if highs[tickdir] else -deltas[tickdir]

        # 获取刻度信息
        tick_info = info['tick']
        tick_out = tick_info['outward_factor'] * tickdelta
        tick_in = tick_info['inward_factor'] * tickdelta
        tick_lw = tick_info['linewidth']
        edgep1_tickdir = edgep1[tickdir]
        out_tickdir = edgep1_tickdir + tick_out
        in_tickdir = edgep1_tickdir - tick_in

        default_label_offset = 8.  # 粗略估计的默认标签偏移量

        # 计算每个刻度点的位置
        points = deltas_per_point * deltas
        for tick in ticks:
            # 获取刻度线的位置
            pos = edgep1.copy()
            pos[index] = tick.get_loc()
            pos[tickdir] = out_tickdir
            x1, y1, z1 = proj3d.proj_transform(*pos, self.axes.M)
            pos[tickdir] = in_tickdir
            x2, y2, z2 = proj3d.proj_transform(*pos, self.axes.M)

            # 获取标签的位置
            labeldeltas = (tick.get_pad() + default_label_offset) * points

            pos[tickdir] = edgep1_tickdir
            pos = _move_from_center(pos, centers, labeldeltas, self._axmask())
            lx, ly, lz = proj3d.proj_transform(*pos, self.axes.M)

            # 更新刻度线和标签的位置
            _tick_update_position(tick, (x1, x2), (y1, y2), (lx, ly))
            tick.tick1line.set_linewidth(tick_lw[tick._major])
            tick.draw(renderer)

    def _draw_labels(self, renderer, edgep1, edgep2, labeldeltas, centers, dx, dy):
        # 获取轴标签信息
        label = self._axinfo["label"]

        # 绘制标签
        lxyz = 0.5 * (edgep1 + edgep2)
        lxyz = _move_from_center(lxyz, centers, labeldeltas, self._axmask())
        tlx, tly, tlz = proj3d.proj_transform(*lxyz, self.axes.M)
        self.label.set_position((tlx, tly))
        # 如果标签需要旋转，则计算角度并设置旋转
        if self.get_rotate_label(self.label.get_text()):
            angle = art3d._norm_text_angle(np.rad2deg(np.arctan2(dy, dx)))
            self.label.set_rotation(angle)
        self.label.set_va(label['va'])
        self.label.set_ha(label['ha'])
        self.label.set_rotation_mode(label['rotation_mode'])
        self.label.draw(renderer)

    @artist.allow_rasterization
    def draw(self, renderer):
        # 设置标签和偏移文本的坐标变换为数据坐标系
        self.label._transform = self.axes.transData
        self.offsetText._transform = self.axes.transData
        # 在渲染器中打开一个名为"axis3d"的组，并分配一个唯一的gid
        renderer.open_group("axis3d", gid=self.get_gid())

        # 获取坐标信息：最小值、最大值、是否翻转、是否高处
        mins, maxs, tc, highs = self._get_coord_info()
        # 计算中心点和增量
        centers, deltas = self._calc_centers_deltas(maxs, mins)

        # 计算偏移距离
        # 一个粗略的估计；由于3D图形会旋转，点的位置不确定
        reltoinches = self.figure.dpi_scale_trans.inverted()
        ax_inches = reltoinches.transform(self.axes.bbox.size)
        ax_points_estimate = sum(72. * ax_inches)
        deltas_per_point = 48 / ax_points_estimate
        default_offset = 21.
        labeldeltas = (self.labelpad + default_offset) * deltas_per_point * deltas

        # 确定坐标轴线的边缘点
        minmax = np.where(highs, maxs, mins)  # "origin" point
        maxmin = np.where(~highs, maxs, mins)  # "opposite" corner near camera

        # 对于每一条边缘点，根据当前位置投影边缘点
        for edgep1, edgep2, pos in zip(*self._get_all_axis_line_edge_points(
                                           minmax, maxmin, self._tick_position)):
            pep = proj3d._proj_trans_points([edgep1, edgep2], self.axes.M)
            pep = np.asarray(pep)

            # 使用transAxes变换，因为Text对象相对于显示坐标系旋转文本。
            # 因此，如果希望标签保持与轴平行，不受纵横比影响，需要将平面的边缘点转换为显示坐标，并计算角度。
            # TODO: 或许Text对象应该自行处理这个问题？
            dx, dy = (self.axes.transAxes.transform([pep[0:2, 1]]) -
                      self.axes.transAxes.transform([pep[0:2, 0]]))[0]

            # 绘制线条
            self.line.set_data(pep[0], pep[1])
            self.line.draw(renderer)

            # 绘制刻度
            self._draw_ticks(renderer, edgep1, centers, deltas, highs,
                             deltas_per_point, pos)

            # 绘制偏移文本
            self._draw_offset_text(renderer, edgep1, edgep2, labeldeltas,
                                   centers, highs, pep, dx, dy)

        # 对于每一条边缘点，根据当前位置投影边缘点
        for edgep1, edgep2, pos in zip(*self._get_all_axis_line_edge_points(
                                           minmax, maxmin, self._label_position)):
            pep = proj3d._proj_trans_points([edgep1, edgep2], self.axes.M)
            pep = np.asarray(pep)
            dx, dy = (self.axes.transAxes.transform([pep[0:2, 1]]) -
                      self.axes.transAxes.transform([pep[0:2, 0]]))[0]

            # 绘制标签
            self._draw_labels(renderer, edgep1, edgep2, labeldeltas, centers, dx, dy)

        # 在渲染器中关闭名为"axis3d"的组
        renderer.close_group('axis3d')
        # 更新绘图对象的状态
        self.stale = False
    # 使用装饰器允许对艺术家进行光栅化操作
    @artist.allow_rasterization
    # 定义绘制网格的方法，接受渲染器作为参数
    def draw_grid(self, renderer):
        # 如果图表不需要绘制网格，则直接返回
        if not self.axes._draw_grid:
            return

        # 在渲染器中打开一个名为 "grid3d" 的分组，并指定当前对象的 gid
        renderer.open_group("grid3d", gid=self.get_gid())

        # 更新坐标轴刻度
        ticks = self._update_ticks()
        if len(ticks):
            # 获取坐标轴的一般信息
            info = self._axinfo
            # 获取当前坐标轴的索引
            index = info["i"]

            # 获取坐标信息：最小值、最大值、坐标数量、高度标志
            mins, maxs, tc, highs = self._get_coord_info()

            # 根据高度标志选择最大或最小的坐标值
            minmax = np.where(highs, maxs, mins)
            maxmin = np.where(~highs, maxs, mins)

            # 定义平面相交的网格点
            xyz0 = np.tile(minmax, (len(ticks), 1))
            xyz0[:, index] = [tick.get_loc() for tick in ticks]

            # 定义网格线，从一个平面的末端经过相交点（在 xyz0 处）到另一个平面的末端。
            # 第一个点（0）在维度索引-2上有所不同，最后一个点（2）在维度索引-1上有所不同。
            lines = np.stack([xyz0, xyz0, xyz0], axis=1)
            lines[:, 0, index - 2] = maxmin[index - 2]
            lines[:, 2, index - 1] = maxmin[index - 1]

            # 设置网格线的线段
            self.gridlines.set_segments(lines)
            # 获取网格信息
            gridinfo = info['grid']
            # 设置网格线的颜色、线宽、线型
            self.gridlines.set_color(gridinfo['color'])
            self.gridlines.set_linewidth(gridinfo['linewidth'])
            self.gridlines.set_linestyle(gridinfo['linestyle'])
            # 进行三维投影
            self.gridlines.do_3d_projection()
            # 绘制网格线
            self.gridlines.draw(renderer)

        # 在渲染器中关闭名为 "grid3d" 的分组
        renderer.close_group('grid3d')

    # TODO: 让这部分在 mplot3d 支持变换框架时更加正常工作。
    def get_tightbbox(self, renderer=None, *, for_layout_only=False):
        # docstring inherited
        # 检查当前对象是否可见，如果不可见则直接返回
        if not self.get_visible():
            return
        
        # 获取主刻度的位置信息
        major_locs = self.get_majorticklocs()
        # 获取次刻度的位置信息
        minor_locs = self.get_minorticklocs()

        # 获取所有刻度对象，包括主刻度和次刻度
        ticks = [*self.get_minor_ticks(len(minor_locs)),
                 *self.get_major_ticks(len(major_locs))]
        
        # 获取当前视图的数据间隔
        view_low, view_high = self.get_view_interval()
        if view_low > view_high:
            view_low, view_high = view_high, view_low
        
        # 将视图间隔转换为画布坐标系下的间隔
        interval_t = self.get_transform().transform([view_low, view_high])

        # 筛选出需要绘制的刻度对象
        ticks_to_draw = []
        for tick in ticks:
            try:
                # 尝试将刻度位置转换为画布坐标系下的位置
                loc_t = self.get_transform().transform(tick.get_loc())
            except AssertionError:
                # Transform.transform不允许掩码值，但某些刻度可能会生成掩码值，因此需要此try/except块
                pass
            else:
                # 检查刻度的位置是否在视图间隔内
                if mtransforms._interval_contains_close(interval_t, loc_t):
                    ticks_to_draw.append(tick)

        # 更新ticks为需要绘制的刻度对象列表
        ticks = ticks_to_draw

        # 获取刻度标签边界框
        bb_1, bb_2 = self._get_ticklabel_bboxes(ticks, renderer)
        other = []

        # 如果刻度线可见，则添加其窗口边界到other列表中
        if self.line.get_visible():
            other.append(self.line.get_window_extent(renderer))
        
        # 如果标签可见且不仅用于布局且有文本内容，则添加其窗口边界到other列表中
        if (self.label.get_visible() and not for_layout_only and
                self.label.get_text()):
            other.append(self.label.get_window_extent(renderer))

        # 返回所有边界框的并集作为结果
        return mtransforms.Bbox.union([*bb_1, *bb_2, *other])

    d_interval = _api.deprecated(
        "3.6", alternative="get_data_interval", pending=True)(
            property(lambda self: self.get_data_interval(),
                     lambda self, minmax: self.set_data_interval(*minmax)))
    v_interval = _api.deprecated(
        "3.6", alternative="get_view_interval", pending=True)(
            property(lambda self: self.get_view_interval(),
                     lambda self, minmax: self.set_view_interval(*minmax)))
# 定义 X 轴类，继承自 Axis 类
class XAxis(Axis):
    # 设置轴名称为 "x"
    axis_name = "x"
    # 使用 maxis._make_getset_interval 函数生成获取和设置视图间隔的方法
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "xy_viewLim", "intervalx")
    # 使用 maxis._make_getset_interval 函数生成获取和设置数据间隔的方法
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "xy_dataLim", "intervalx")


# 定义 Y 轴类，继承自 Axis 类
class YAxis(Axis):
    # 设置轴名称为 "y"
    axis_name = "y"
    # 使用 maxis._make_getset_interval 函数生成获取和设置视图间隔的方法
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "xy_viewLim", "intervaly")
    # 使用 maxis._make_getset_interval 函数生成获取和设置数据间隔的方法
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "xy_dataLim", "intervaly")


# 定义 Z 轴类，继承自 Axis 类
class ZAxis(Axis):
    # 设置轴名称为 "z"
    axis_name = "z"
    # 使用 maxis._make_getset_interval 函数生成获取和设置视图间隔的方法
    get_view_interval, set_view_interval = maxis._make_getset_interval(
        "view", "zz_viewLim", "intervalx")
    # 使用 maxis._make_getset_interval 函数生成获取和设置数据间隔的方法
    get_data_interval, set_data_interval = maxis._make_getset_interval(
        "data", "zz_dataLim", "intervalx")
```