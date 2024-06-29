# `D:\src\scipysrc\matplotlib\lib\matplotlib\sankey.py`

```py
"""
Module for creating Sankey diagrams using Matplotlib.
"""

# 导入 logging 模块，用于记录日志信息
import logging
# 导入 SimpleNamespace 类型，用于创建简单的命名空间对象
from types import SimpleNamespace

# 导入 NumPy 库，并重命名为 np
import numpy as np

# 导入 Matplotlib 库，并重命名为 mpl
import matplotlib as mpl
# 从 Matplotlib 中导入 Path 类
from matplotlib.path import Path
# 从 Matplotlib 中导入 PathPatch 类
from matplotlib.patches import PathPatch
# 从 Matplotlib 中导入 Affine2D 类
from matplotlib.transforms import Affine2D
# 导入 Matplotlib 中的 _docstring
from matplotlib import _docstring

# 创建日志记录器对象，并命名为 _log
_log = logging.getLogger(__name__)

# 定义几个角度常量，用于表示箭头的方向
# Angles [deg/90]
RIGHT = 0
UP = 1
# LEFT = 2
DOWN = 3

# 定义 Sankey 类，用于创建桑基图
class Sankey:
    """
    Sankey diagram.

      Sankey diagrams are a specific type of flow diagram, in which
      the width of the arrows is shown proportionally to the flow
      quantity.  They are typically used to visualize energy or
      material or cost transfers between processes.
      `Wikipedia (6/1/2011) <https://en.wikipedia.org/wiki/Sankey_diagram>`_

    """
    def _arc(self, quadrant=0, cw=True, radius=1, center=(0, 0)):
        """
        Return the codes and vertices for a rotated, scaled, and translated
        90 degree arc.

        Other Parameters
        ----------------
        quadrant : {0, 1, 2, 3}, default: 0
            Uses 0-based indexing (0, 1, 2, or 3).
        cw : bool, default: True
            If True, the arc vertices are produced clockwise; counter-clockwise
            otherwise.
        radius : float, default: 1
            The radius of the arc.
        center : (float, float), default: (0, 0)
            (x, y) tuple of the arc's center.
        """
        # Note:  It would be possible to use matplotlib's transforms to rotate,
        # scale, and translate the arc, but since the angles are discrete,
        # it's just as easy and maybe more efficient to do it here.
        # 定义代表路径代码的常量，对应于路径的直线和曲线段
        ARC_CODES = [Path.LINETO,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4,
                     Path.CURVE4]
        # 用于近似90度圆弧的立方贝塞尔曲线的顶点坐标
        # 这些坐标可以通过 Path.arc(0, 90) 得到
        ARC_VERTICES = np.array([[1.00000000e+00, 0.00000000e+00],
                                 [1.00000000e+00, 2.65114773e-01],
                                 [8.94571235e-01, 5.19642327e-01],
                                 [7.07106781e-01, 7.07106781e-01],
                                 [5.19642327e-01, 8.94571235e-01],
                                 [2.65114773e-01, 1.00000000e+00],
                                 [0.00000000e+00, 1.00000000e+00]])
        # 如果 quadrant 是 0 或 2
        if quadrant in (0, 2):
            # 如果 cw 是 True，顶点保持不变
            if cw:
                vertices = ARC_VERTICES
            else:
                # 如果 cw 是 False，交换 x 和 y 坐标
                vertices = ARC_VERTICES[:, ::-1]  # Swap x and y.
        else:  # 如果 quadrant 是 1 或 3
            # 如果 cw 是 True
            if cw:
                # 交换 x 和 y 坐标，并取负值
                vertices = np.column_stack((-ARC_VERTICES[:, 1],
                                             ARC_VERTICES[:, 0]))
            else:
                # 取负值并交换 x 和 y 坐标
                vertices = np.column_stack((-ARC_VERTICES[:, 0],
                                             ARC_VERTICES[:, 1]))
        # 如果 quadrant 大于 1，将半径取负值，相当于旋转180度
        if quadrant > 1:
            radius = -radius  # Rotate 180 deg.
        # 返回由顶点组成的列表，每个顶点进行了旋转、缩放和平移
        return list(zip(ARC_CODES, radius * vertices +
                        np.tile(center, (ARC_VERTICES.shape[0], 1))))
    def _add_input(self, path, angle, flow, length):
        """
        Add an input to a path and return its tip and label locations.
        """
        # 如果角度为 None，则返回 [0, 0] 和 [0, 0]
        if angle is None:
            return [0, 0], [0, 0]
        else:
            # 使用路径中的最后一个点作为参考点
            x, y = path[-1][1]

            # 计算 dipdepth，即流动深度的一半乘以间距
            dipdepth = (flow / 2) * self.pitch

            # 如果角度为 RIGHT
            if angle == RIGHT:
                x -= length  # 向左移动长度
                dip = [x + dipdepth, y + flow / 2.0]  # 计算 dip 的位置
                # 添加直线路径段和 dip 路径段到路径中
                path.extend([(Path.LINETO, [x, y]),
                             (Path.LINETO, dip),
                             (Path.LINETO, [x, y + flow]),
                             (Path.LINETO, [x + self.gap, y + flow])])
                # 计算标签位置
                label_location = [dip[0] - self.offset, dip[1]]
            
            # 如果角度为 Vertical
            else:  
                x -= self.gap  # 向左移动间距
                # 根据角度确定符号
                if angle == UP:
                    sign = 1
                else:
                    sign = -1
                
                # 计算 dip 的位置
                dip = [x - flow / 2, y - sign * (length - dipdepth)]
                
                # 根据角度确定象限
                if angle == DOWN:
                    quadrant = 2
                else:
                    quadrant = 1
                
                # 如果内半径不为零，则添加内弧段到路径中
                if self.radius:
                    path.extend(self._arc(quadrant=quadrant,
                                          cw=angle == UP,
                                          radius=self.radius,
                                          center=(x + self.radius,
                                                  y - sign * self.radius)))
                else:
                    path.append((Path.LINETO, [x, y]))
                
                # 添加直线路径段和 dip 路径段到路径中
                path.extend([(Path.LINETO, [x, y - sign * length]),
                             (Path.LINETO, dip),
                             (Path.LINETO, [x - flow, y - sign * length])])
                
                # 添加外弧段到路径中
                path.extend(self._arc(quadrant=quadrant,
                                      cw=angle == DOWN,
                                      radius=flow + self.radius,
                                      center=(x + self.radius,
                                              y - sign * self.radius)))
                
                # 添加最后一个直线路径段到路径中
                path.append((Path.LINETO, [x - flow, y + sign * flow]))
                
                # 计算标签位置
                label_location = [dip[0], dip[1] - sign * self.offset]

            # 返回 dip 和 label 位置
            return dip, label_location
    def _add_output(self, path, angle, flow, length):
        """
        Append an output to a path and return its tip and label locations.

        .. note:: *flow* is negative for an output.
        """
        # 如果角度为 None，则返回初始位置 [0, 0] 和 [0, 0]
        if angle is None:
            return [0, 0], [0, 0]
        else:
            x, y = path[-1][1]  # 使用路径的最后一个点作为参考点
            # 计算尖端高度，根据流量的负数计算
            tipheight = (self.shoulder - flow / 2) * self.pitch
            if angle == RIGHT:
                # 如果角度为 RIGHT，则沿着 x 轴正方向移动长度
                x += length
                # 计算尖端位置
                tip = [x + tipheight, y + flow / 2.0]
                # 扩展路径，绘制输出箭头
                path.extend([(Path.LINETO, [x, y]),
                             (Path.LINETO, [x, y + self.shoulder]),
                             (Path.LINETO, tip),
                             (Path.LINETO, [x, y - self.shoulder + flow]),
                             (Path.LINETO, [x, y + flow]),
                             (Path.LINETO, [x - self.gap, y + flow])])
                # 标签位置
                label_location = [tip[0] + self.offset, tip[1]]
            else:  # 如果角度为 Vertical
                x += self.gap
                if angle == UP:
                    sign, quadrant = 1, 3
                else:
                    sign, quadrant = -1, 0

                tip = [x - flow / 2.0, y + sign * (length + tipheight)]
                # 如果内半径为零，则不需要内弧线
                if self.radius:
                    path.extend(self._arc(quadrant=quadrant,
                                          cw=angle == UP,
                                          radius=self.radius,
                                          center=(x - self.radius,
                                                  y + sign * self.radius)))
                else:
                    path.append((Path.LINETO, [x, y]))
                path.extend([(Path.LINETO, [x, y + sign * length]),
                             (Path.LINETO, [x - self.shoulder,
                                            y + sign * length]),
                             (Path.LINETO, tip),
                             (Path.LINETO, [x + self.shoulder - flow,
                                            y + sign * length]),
                             (Path.LINETO, [x - flow, y + sign * length])])
                path.extend(self._arc(quadrant=quadrant,
                                      cw=angle == DOWN,
                                      radius=self.radius - flow,
                                      center=(x - self.radius,
                                              y + sign * self.radius)))
                path.append((Path.LINETO, [x - flow, y + sign * flow]))
                # 标签位置
                label_location = [tip[0], tip[1] + sign * self.offset]
            # 返回尖端位置和标签位置
            return tip, label_location
    # 定义一个方法 `_revert`，用于反转路径 `path`，根据参数 `first_action` 指定的第一个操作
    def _revert(self, path, first_action=Path.LINETO):
        """
        A path is not simply reversible by path[::-1] since the code
        specifies an action to take from the **previous** point.
        """
        # 初始化一个空列表 `reverse_path`，用于存储反转后的路径
        reverse_path = []
        # 初始设定下一个动作码为 `first_action`
        next_code = first_action
        # 遍历反向的路径 `path[::-1]`
        for code, position in path[::-1]:
            # 将当前动作码 `next_code` 和位置 `position` 添加到 `reverse_path` 中
            reverse_path.append((next_code, position))
            # 更新下一个动作码为当前循环迭代的动作码 `code`
            next_code = code
        # 返回反转后的路径 `reverse_path`
        return reverse_path
        # 下面的代码尝试以更高效的方式反转路径，但由于元组对象不支持项目赋值而失败：
        # path[1] = path[1][-1:0:-1]
        # path[1][0] = first_action
        # path[2] = path[2][::-1]
        # return path

    @_docstring.dedent_interpd
    # 定义方法 `finish`，用于完成 Sankey 图的调整，并返回关于子图的信息列表
    def finish(self):
        """
        Adjust the Axes and return a list of information about the Sankey
        subdiagram(s).

        Returns a list of subdiagrams with the following fields:

        ========  =============================================================
        Field     Description
        ========  =============================================================
        *patch*   Sankey outline (a `~matplotlib.patches.PathPatch`).
        *flows*   Flow values (positive for input, negative for output).
        *angles*  List of angles of the arrows [deg/90].
                  For example, if the diagram has not been rotated,
                  an input to the top side has an angle of 3 (DOWN),
                  and an output from the top side has an angle of 1 (UP).
                  If a flow has been skipped (because its magnitude is less
                  than *tolerance*), then its angle will be *None*.
        *tips*    (N, 2)-array of the (x, y) positions of the tips (or "dips")
                  of the flow paths.
                  If the magnitude of a flow is less the *tolerance* of this
                  `Sankey` instance, the flow is skipped and its tip will be at
                  the center of the diagram.
        *text*    `.Text` instance for the diagram label.
        *texts*   List of `.Text` instances for the flow labels.
        ========  =============================================================

        See Also
        --------
        Sankey.add
        """
        # 设置图表的坐标轴范围，以 `self.extent` 和 `self.margin` 为基础
        self.ax.axis([self.extent[0] - self.margin,
                      self.extent[1] + self.margin,
                      self.extent[2] - self.margin,
                      self.extent[3] + self.margin])
        # 设置图表的纵横比为等比例，可调整为数据限制
        self.ax.set_aspect('equal', adjustable='datalim')
        # 返回 Sankey 图的子图信息列表 `self.diagrams`
        return self.diagrams
```