# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_curve.py`

```
# 导入pyglet.gl模块，别名为pgl，用于OpenGL相关操作
import pyglet.gl as pgl
# 从sympy.core模块导入S对象
from sympy.core import S
# 从sympy.plotting.pygletplot.plot_mode_base模块导入PlotModeBase类
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase

# 定义PlotCurve类，继承自PlotModeBase类
class PlotCurve(PlotModeBase):

    # 类变量style_override，设定为'wireframe'，表示样式覆盖为线框模式
    style_override = 'wireframe'

    # 私有方法，计算顶点位置
    def _on_calculate_verts(self):
        # 设定时间间隔为self.intervals的第一个元素
        self.t_interval = self.intervals[0]
        # 将时间间隔转换为列表，并存储在self.t_set中
        self.t_set = list(self.t_interval.frange())
        # 初始化边界框，分别为x、y、z轴的最大值、最小值和间隔
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        # 获取计算函数评估器
        evaluate = self._get_evaluator()

        # 初始化计算顶点位置和长度
        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = float(self.t_interval.v_len)

        # 初始化顶点列表
        self.verts = []
        b = self.bounds
        # 遍历时间集合self.t_set中的每一个时间点t
        for t in self.t_set:
            try:
                _e = evaluate(t)    # 计算顶点位置
            except (NameError, ZeroDivisionError):
                _e = None
            if _e is not None:      # 更新边界框
                for axis in range(3):
                    b[axis][0] = min([b[axis][0], _e[axis]])
                    b[axis][1] = max([b[axis][1], _e[axis]])
            self.verts.append(_e)   # 将顶点位置_e加入顶点列表self.verts
            self._calculating_verts_pos += 1.0

        # 计算每个轴的边界框间隔
        for axis in range(3):
            b[axis][2] = b[axis][1] - b[axis][0]
            if b[axis][2] == 0.0:
                b[axis][2] = 1.0

        # 将当前线框绘制函数推入绘制栈
        self.push_wireframe(self.draw_verts(False))

    # 私有方法，计算着色顶点
    def _on_calculate_cverts(self):
        # 若顶点列表self.verts为空或者颜色为空，则返回
        if not self.verts or not self.color:
            return

        # 定义内部函数，设置工作长度为n
        def set_work_len(n):
            self._calculating_cverts_len = float(n)

        # 定义内部函数，增加工作位置
        def inc_work_pos():
            self._calculating_cverts_pos += 1.0

        # 设置计算着色顶点的工作长度为1
        set_work_len(1)
        # 初始化计算着色顶点的位置为0
        self._calculating_cverts_pos = 0
        # 使用颜色对象self.color对顶点列表self.verts进行着色，存入self.cverts中
        self.cverts = self.color.apply_to_curve(self.verts,
                                                self.t_set,
                                                set_len=set_work_len,
                                                inc_pos=inc_work_pos)
        # 将当前线框绘制函数推入绘制栈
        self.push_wireframe(self.draw_verts(True))

    # 公有方法，计算单个着色顶点
    def calculate_one_cvert(self, t):
        # 获取第t个顶点
        vert = self.verts[t]
        # 根据顶点的坐标和时间t，使用颜色对象self.color计算着色顶点并返回
        return self.color(vert[0], vert[1], vert[2],
                          self.t_set[t], None)

    # 公有方法，绘制顶点
    def draw_verts(self, use_cverts):
        # 定义内部函数f，用于绘制顶点
        def f():
            pgl.glBegin(pgl.GL_LINE_STRIP)  # 开始绘制线条
            # 遍历时间集合self.t_set中的每一个时间点t
            for t in range(len(self.t_set)):
                p = self.verts[t]   # 获取第t个顶点位置
                if p is None:       # 若顶点位置为None，结束当前线条绘制并开始新的线条
                    pgl.glEnd()
                    pgl.glBegin(pgl.GL_LINE_STRIP)
                    continue
                if use_cverts:      # 若使用着色顶点
                    c = self.cverts[t]  # 获取第t个着色顶点颜色
                    if c is None:
                        c = (0, 0, 0)   # 若着色顶点颜色为None，设定为黑色
                    pgl.glColor3f(*c)   # 设置OpenGL颜色为c
                else:
                    pgl.glColor3f(*self.default_wireframe_color)   # 使用默认线框颜色
                pgl.glVertex3f(*p)  # 绘制顶点p
            pgl.glEnd()   # 结束绘制
        return f   # 返回内部函数f作为绘制顶点函数
```