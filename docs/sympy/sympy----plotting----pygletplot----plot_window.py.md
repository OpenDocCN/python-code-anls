# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_window.py`

```
# 导入性能计数器模块
from time import perf_counter

# 导入 Pyglet 的 OpenGL 接口
import pyglet.gl as pgl

# 导入绘图窗口管理类、绘图相机、绘图控制器
from sympy.plotting.pygletplot.managed_window import ManagedWindow
from sympy.plotting.pygletplot.plot_camera import PlotCamera
from sympy.plotting.pygletplot.plot_controller import PlotController

# 继承自 ManagedWindow 类，用于绘制 SymPy 图形的窗口类
class PlotWindow(ManagedWindow):

    def __init__(self, plot, antialiasing=True, ortho=False,
                 invert_mouse_zoom=False, linewidth=1.5, caption="SymPy Plot",
                 **kwargs):
        """
        Named Arguments
        ===============

        antialiasing = True
            True OR False
        ortho = False
            True OR False
        invert_mouse_zoom = False
            True OR False
        """
        # 调用父类的构造函数
        super().__init__(**kwargs)
        
        # 存储传入的绘图对象
        self.plot = plot

        # 初始化相机为 None，并设置计算状态为 False
        self.camera = None
        self._calculating = False

        # 设置各种参数，如抗锯齿、正交投影、鼠标缩放反转等
        self.antialiasing = antialiasing
        self.ortho = ortho
        self.invert_mouse_zoom = invert_mouse_zoom
        self.linewidth = linewidth
        self.title = caption
        self.last_caption_update = 0
        self.caption_update_interval = 0.2
        self.drawing_first_object = True

    # 设置窗口的 OpenGL 环境
    def setup(self):
        # 创建绘图相机对象，并传入当前窗口对象和正交投影参数
        self.camera = PlotCamera(self, ortho=self.ortho)
        
        # 创建绘图控制器对象，并传入当前窗口对象和鼠标缩放反转参数
        self.controller = PlotController(self, invert_mouse_zoom=self.invert_mouse_zoom)
        self.push_handlers(self.controller)

        # 设置 OpenGL 的清屏颜色为白色
        pgl.glClearColor(1.0, 1.0, 1.0, 0.0)
        pgl.glClearDepth(1.0)

        # 设置深度测试函数
        pgl.glDepthFunc(pgl.GL_LESS)
        pgl.glEnable(pgl.GL_DEPTH_TEST)

        # 启用线条平滑处理
        pgl.glEnable(pgl.GL_LINE_SMOOTH)
        pgl.glShadeModel(pgl.GL_SMOOTH)

        # 设置线条宽度
        pgl.glLineWidth(self.linewidth)

        # 启用混合功能，设置混合因子
        pgl.glEnable(pgl.GL_BLEND)
        pgl.glBlendFunc(pgl.GL_SRC_ALPHA, pgl.GL_ONE_MINUS_SRC_ALPHA)

        # 如果启用了抗锯齿，设置相应的渲染提示
        if self.antialiasing:
            pgl.glHint(pgl.GL_LINE_SMOOTH_HINT, pgl.GL_NICEST)
            pgl.glHint(pgl.GL_POLYGON_SMOOTH_HINT, pgl.GL_NICEST)

        # 设置相机的投影
        self.camera.setup_projection()

    # 当窗口大小改变时调用的方法，重新设置投影
    def on_resize(self, w, h):
        super().on_resize(w, h)
        if self.camera is not None:
            self.camera.setup_projection()

    # 更新方法，由控制器调用以更新状态
    def update(self, dt):
        self.controller.update(dt)
    # 获取绘图对象的渲染锁，并进行加锁操作
    self.plot._render_lock.acquire()

    # 应用相机的变换操作
    self.camera.apply_transformation()

    # 初始化计算顶点位置和长度的变量
    calc_verts_pos, calc_verts_len = 0, 0
    calc_cverts_pos, calc_cverts_len = 0, 0

    # 检查是否应该更新标题栏内容，根据时间间隔决定
    should_update_caption = (perf_counter() - self.last_caption_update >
                             self.caption_update_interval)

    # 如果绘图对象中没有任何函数，则标记为正在绘制第一个对象
    if len(self.plot._functions.values()) == 0:
        self.drawing_first_object = True

    # 获取绘图函数的迭代器
    iterfunctions = iter(self.plot._functions.values())

    # 遍历绘图函数
    for r in iterfunctions:
        # 如果是第一次绘制对象，则设置相机的默认旋转预设
        if self.drawing_first_object:
            self.camera.set_rot_preset(r.default_rot_preset)
            self.drawing_first_object = False

        # 压栈当前矩阵
        pgl.glPushMatrix()
        # 调用绘图函数的绘制方法
        r._draw()
        # 出栈当前矩阵
        pgl.glPopMatrix()

        # 在迭代和加锁的同时，更新计算顶点和颜色顶点的位置和长度
        if should_update_caption:
            try:
                if r.calculating_verts:
                    calc_verts_pos += r.calculating_verts_pos
                    calc_verts_len += r.calculating_verts_len
                if r.calculating_cverts:
                    calc_cverts_pos += r.calculating_cverts_pos
                    calc_cverts_len += r.calculating_cverts_len
            except ValueError:
                pass

    # 遍历绘图对象中的物体
    for r in self.plot._pobjects:
        # 压栈当前矩阵
        pgl.glPushMatrix()
        # 调用物体的绘制方法
        r._draw()
        # 出栈当前矩阵
        pgl.glPopMatrix()

    # 如果应该更新标题栏内容，则调用更新标题栏的方法
    if should_update_caption:
        self.update_caption(calc_verts_pos, calc_verts_len,
                            calc_cverts_pos, calc_cverts_len)
        # 更新最后一次更新标题时间
        self.last_caption_update = perf_counter()

    # 如果存在截图对象，则执行保存操作
    if self.plot._screenshot:
        self.plot._screenshot._execute_saving()

    # 释放绘图对象的渲染锁
    self.plot._render_lock.release()
```