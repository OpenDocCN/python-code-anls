# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_surface.py`

```
import pyglet.gl as pgl  # 导入 Pyglet 的 OpenGL 接口

from sympy.core import S  # 导入 SymPy 的 S 对象
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase  # 导入绘图模式基类


class PlotSurface(PlotModeBase):  # 定义 PlotSurface 类，继承自 PlotModeBase 类

    default_rot_preset = 'perspective'  # 默认的旋转预设为透视

    def _on_calculate_verts(self):
        self.u_interval = self.intervals[0]  # 设置 u_interval 为 intervals 的第一个元素
        self.u_set = list(self.u_interval.frange())  # 根据 u_interval 的范围生成 u_set 列表
        self.v_interval = self.intervals[1]  # 设置 v_interval 为 intervals 的第二个元素
        self.v_set = list(self.v_interval.frange())  # 根据 v_interval 的范围生成 v_set 列表
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],  # 初始化 bounds 为一个包含三个列表的列表
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        evaluate = self._get_evaluator()  # 获取评估器函数

        self._calculating_verts_pos = 0.0  # 初始化计算顶点位置
        self._calculating_verts_len = float(  # 设置计算顶点长度
            self.u_interval.v_len*self.v_interval.v_len)

        verts = []  # 初始化顶点列表
        b = self.bounds  # 别名 b 用于 bounds
        for u in self.u_set:  # 遍历 u_set
            column = []  # 初始化列
            for v in self.v_set:  # 遍历 v_set
                try:
                    _e = evaluate(u, v)  # 计算顶点
                except ZeroDivisionError:
                    _e = None
                if _e is not None:  # 更新边界框
                    for axis in range(3):
                        b[axis][0] = min([b[axis][0], _e[axis]])
                        b[axis][1] = max([b[axis][1], _e[axis]])
                column.append(_e)  # 将顶点添加到列中
                self._calculating_verts_pos += 1.0  # 更新计算顶点位置

            verts.append(column)  # 将列添加到顶点列表中
        for axis in range(3):  # 对于每个坐标轴
            b[axis][2] = b[axis][1] - b[axis][0]  # 计算边界框的范围
            if b[axis][2] == 0.0:
                b[axis][2] = 1.0

        self.verts = verts  # 将顶点列表赋值给 self.verts
        self.push_wireframe(self.draw_verts(False, False))  # 推送线框绘制
        self.push_solid(self.draw_verts(False, True))  # 推送实体绘制

    def _on_calculate_cverts(self):
        if not self.verts or not self.color:  # 如果顶点列表或颜色不存在则返回
            return

        def set_work_len(n):  # 定义设置工作长度的函数
            self._calculating_cverts_len = float(n)

        def inc_work_pos():  # 定义增加工作位置的函数
            self._calculating_cverts_pos += 1.0

        set_work_len(1)  # 设置工作长度为 1
        self._calculating_cverts_pos = 0  # 初始化计算颜色顶点位置
        self.cverts = self.color.apply_to_surface(self.verts,  # 计算颜色顶点
                                                  self.u_set,
                                                  self.v_set,
                                                  set_len=set_work_len,
                                                  inc_pos=inc_work_pos)
        self.push_solid(self.draw_verts(True, True))  # 推送实体绘制

    def calculate_one_cvert(self, u, v):
        vert = self.verts[u][v]  # 获取指定位置的顶点
        return self.color(vert[0], vert[1], vert[2],  # 计算单个颜色顶点
                          self.u_set[u], self.v_set[v])  # 使用 u 和 v 的值作为参数
    # 定义一个内部函数f，用于绘制顶点
    def draw_verts(self, use_cverts, use_solid_color):
        def f():
            # 遍历self.u_set中的索引，从1到倒数第二个元素
            for u in range(1, len(self.u_set)):
                # 使用OpenGL开始绘制四边形条带
                pgl.glBegin(pgl.GL_QUAD_STRIP)
                # 遍历self.v_set中的索引
                for v in range(len(self.v_set)):
                    # 获取相邻顶点pa和pb
                    pa = self.verts[u - 1][v]
                    pb = self.verts[u][v]
                    # 如果任一顶点为None，则结束当前四边形条带，开始新的条带
                    if pa is None or pb is None:
                        pgl.glEnd()
                        pgl.glBegin(pgl.GL_QUAD_STRIP)
                        continue
                    # 如果使用自定义顶点颜色
                    if use_cverts:
                        # 获取顶点颜色ca和cb，若为None则设为(0, 0, 0)
                        ca = self.cverts[u - 1][v] if self.cverts[u - 1][v] is not None else (0, 0, 0)
                        cb = self.cverts[u][v] if self.cverts[u][v] is not None else (0, 0, 0)
                    else:
                        # 如果不使用自定义顶点颜色
                        if use_solid_color:
                            # 如果使用单一填充颜色，则设为默认填充颜色
                            ca = cb = self.default_solid_color
                        else:
                            # 否则使用默认线框颜色
                            ca = cb = self.default_wireframe_color
                    # 设置顶点颜色并绘制顶点pa
                    pgl.glColor3f(*ca)
                    pgl.glVertex3f(*pa)
                    # 设置顶点颜色并绘制顶点pb
                    pgl.glColor3f(*cb)
                    pgl.glVertex3f(*pb)
                # 结束当前四边形条带的绘制
                pgl.glEnd()
        # 返回内部函数f作为结果
        return f
```