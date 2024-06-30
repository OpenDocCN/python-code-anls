# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot_mode_base.py`

```
import pyglet.gl as pgl  # 导入 pyglet.gl 模块并将其重命名为 pgl
from sympy.core import S  # 导入 SymPy 核心模块中的 S 对象
from sympy.plotting.pygletplot.color_scheme import ColorScheme  # 导入颜色方案类 ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode  # 导入绘图模式类 PlotMode
from sympy.utilities.iterables import is_sequence  # 导入 SymPy 工具模块中的 is_sequence 函数
from time import sleep  # 导入 sleep 函数
from threading import Thread, Event, RLock  # 导入线程相关的类 Thread, Event, RLock
import warnings  # 导入警告模块


class PlotModeBase(PlotMode):
    """
    Intended parent class for plotting
    modes. Provides base functionality
    in conjunction with its parent,
    PlotMode.
    """

    ##
    ## Class-Level Attributes
    ##

    """
    The following attributes are meant
    to be set at the class level, and serve
    as parameters to the plot mode registry
    (in PlotMode). See plot_modes.py for
    concrete examples.
    """

    """
    i_vars
        'x' for Cartesian2D
        'xy' for Cartesian3D
        etc.

    d_vars
        'y' for Cartesian2D
        'r' for Polar
        etc.
    """
    i_vars, d_vars = '', ''

    """
    intervals
        Default intervals for each i_var, and in the
        same order. Specified [min, max, steps].
        No variable can be given (it is bound later).
    """
    intervals = []

    """
    aliases
        A list of strings which can be used to
        access this mode.
        'cartesian' for Cartesian2D and Cartesian3D
        'polar' for Polar
        'cylindrical', 'polar' for Cylindrical

        Note that _init_mode chooses the first alias
        in the list as the mode's primary_alias, which
        will be displayed to the end user in certain
        contexts.
    """
    aliases = []

    """
    is_default
        Whether to set this mode as the default
        for arguments passed to PlotMode() containing
        the same number of d_vars as this mode and
        at most the same number of i_vars.
    """
    is_default = False

    """
    All of the above attributes are defined in PlotMode.
    The following ones are specific to PlotModeBase.
    """

    """
    A list of the render styles. Do not modify.
    """
    styles = {'wireframe': 1, 'solid': 2, 'both': 3}

    """
    style_override
        Always use this style if not blank.
    """
    style_override = ''

    """
    default_wireframe_color
    default_solid_color
        Can be used when color is None or being calculated.
        Used by PlotCurve and PlotSurface, but not anywhere
        in PlotModeBase.
    """

    default_wireframe_color = (0.85, 0.85, 0.85)
    default_solid_color = (0.6, 0.6, 0.9)
    default_rot_preset = 'xy'

    ##
    ## Instance-Level Attributes
    ##

    ## 'Abstract' member functions
    def _get_evaluator(self):
        if self.use_lambda_eval:
            try:
                e = self._get_lambda_evaluator()  # 如果使用 lambda_eval，则尝试获取 lambda 评估器
                return e
            except Exception:
                warnings.warn("\nWarning: creating lambda evaluator failed. "
                       "Falling back on SymPy subs evaluator.")  # 如果创建 lambda 评估器失败，则发出警告并返回 SymPy subs 评估器
        return self._get_sympy_evaluator()  # 返回 SymPy 符号计算评估器
    # 抽象基类方法，子类需要实现具体功能
    def _get_sympy_evaluator(self):
        raise NotImplementedError()

    # 抽象基类方法，子类需要实现具体功能
    def _get_lambda_evaluator(self):
        raise NotImplementedError()

    # 抽象基类方法，子类需要实现具体功能
    def _on_calculate_verts(self):
        raise NotImplementedError()

    # 抽象基类方法，子类需要实现具体功能
    def _on_calculate_cverts(self):
        raise NotImplementedError()

    ## Base member functions

    # 初始化函数，设置初始值和事件处理对象
    def __init__(self, *args, bounds_callback=None, **kwargs):
        self.verts = []
        self.cverts = []
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        self.cbounds = [[S.Infinity, S.NegativeInfinity, 0],
                        [S.Infinity, S.NegativeInfinity, 0],
                        [S.Infinity, S.NegativeInfinity, 0]]

        self._draw_lock = RLock()  # 创建线程锁对象

        # 创建事件对象，用于通知计算顶点和控制顶点的状态
        self._calculating_verts = Event()
        self._calculating_cverts = Event()

        # 初始化计算顶点和控制顶点的进度状态
        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = 0.0
        self._calculating_cverts_pos = 0.0
        self._calculating_cverts_len = 0.0

        # 设置渲染栈的最大大小
        self._max_render_stack_size = 3

        # 初始化线框和实体的绘制列表
        self._draw_wireframe = [-1]
        self._draw_solid = [-1]

        self._style = None  # 初始化样式和颜色
        self._color = None

        self.predraw = []  # 预绘制函数列表
        self.postdraw = []  # 后绘制函数列表

        # 根据选项设置是否使用 lambda 表达式进行评估
        self.use_lambda_eval = self.options.pop('use_sympy_eval', None) is None
        self.style = self.options.pop('style', '')  # 设置样式
        self.color = self.options.pop('color', 'rainbow')  # 设置颜色
        self.bounds_callback = bounds_callback  # 设置边界回调函数

        self._on_calculate()  # 执行初始化计算

    # 同步装饰器，确保线程安全性，加锁调用指定函数
    def synchronized(f):
        def w(self, *args, **kwargs):
            self._draw_lock.acquire()
            try:
                r = f(self, *args, **kwargs)
                return r
            finally:
                self._draw_lock.release()
        return w

    # 使用同步装饰器包装的方法，将函数添加到线框绘制列表中
    @synchronized
    def push_wireframe(self, function):
        """
        将一个函数推入线框绘制列表，用于构建显示列表（列表在函数外部构建）
        """
        assert callable(function)  # 断言函数可调用
        self._draw_wireframe.append(function)
        if len(self._draw_wireframe) > self._max_render_stack_size:
            del self._draw_wireframe[1]  # 删除标记元素，保持列表大小

    # 使用同步装饰器包装的方法，将函数添加到实体绘制列表中
    @synchronized
    def push_solid(self, function):
        """
        将一个函数推入实体绘制列表，用于构建显示列表（列表在函数外部构建）
        """
        assert callable(function)  # 断言函数可调用
        self._draw_solid.append(function)
        if len(self._draw_solid) > self._max_render_stack_size:
            del self._draw_solid[1]  # 删除标记元素，保持列表大小

    # 创建显示列表的内部方法，用指定函数构建并返回显示列表
    def _create_display_list(self, function):
        dl = pgl.glGenLists(1)  # 创建一个新的显示列表
        pgl.glNewList(dl, pgl.GL_COMPILE)  # 开始编译显示列表
        function()  # 执行传入的函数
        pgl.glEndList()  # 结束显示列表的编译
        return dl  # 返回创建的显示列表
    # 从渲染堆栈中获取顶部元素
    top = render_stack[-1]
    # 如果顶部元素为-1，表示没有要显示的内容，直接返回-1
    if top == -1:
        return -1  # nothing to display
    # 如果顶部元素为可调用对象（函数），则创建显示列表并更新堆栈顶部元素为显示列表和原函数
    elif callable(top):
        dl = self._create_display_list(top)
        render_stack[-1] = (dl, top)
        return dl  # display newly added list
    # 如果顶部元素是包含两个元素的元组
    elif len(top) == 2:
        # 如果第一个元素是已存储的显示列表 ID，则直接返回该 ID
        if pgl.GL_TRUE == pgl.glIsList(top[0]):
            return top[0]  # display stored list
        # 否则，重新创建显示列表，并更新堆栈顶部元素为新的显示列表和原函数
        dl = self._create_display_list(top[1])
        render_stack[-1] = (dl, top[1])
        return dl  # display regenerated list

def _draw_solid_display_list(self, dl):
    # 保存当前 OpenGL 状态
    pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
    # 设置多边形模式为填充模式
    pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_FILL)
    # 调用指定的显示列表来绘制实心图形
    pgl.glCallList(dl)
    # 恢复之前保存的 OpenGL 状态
    pgl.glPopAttrib()

def _draw_wireframe_display_list(self, dl):
    # 保存当前 OpenGL 状态
    pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
    # 设置多边形模式为线框模式
    pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_LINE)
    # 启用多边形偏移以防止 Z-fighting
    pgl.glEnable(pgl.GL_POLYGON_OFFSET_LINE)
    pgl.glPolygonOffset(-0.005, -50.0)
    # 调用指定的显示列表来绘制线框图形
    pgl.glCallList(dl)
    # 恢复之前保存的 OpenGL 状态
    pgl.glPopAttrib()

@synchronized
def draw(self):
    # 在绘制前执行预绘制操作列表中的每个函数
    for f in self.predraw:
        if callable(f):
            f()
    # 根据当前样式选择渲染风格
    if self.style_override:
        style = self.styles[self.style_override]
    else:
        style = self.styles[self._style]
    # 如果样式包括实心图形（bit 2为1）
    if style & 2:
        # 获取顶部实心图形的显示列表并绘制
        dl = self._render_stack_top(self._draw_solid)
        if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
            self._draw_solid_display_list(dl)
    # 如果样式包括线框图形（bit 1为1）
    if style & 1:
        # 获取顶部线框图形的显示列表并绘制
        dl = self._render_stack_top(self._draw_wireframe)
        if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
            self._draw_wireframe_display_list(dl)
    # 在绘制后执行后绘制操作列表中的每个函数
    for f in self.postdraw:
        if callable(f):
            f()

def _on_change_color(self, color):
    # 启动线程以计算所有顶点
    Thread(target=self._calculate_cverts).start()

def _on_calculate(self):
    # 启动线程以计算所有顶点
    Thread(target=self._calculate_all).start()

def _calculate_all(self):
    # 计算所有顶点和颜色顶点
    self._calculate_verts()
    self._calculate_cverts()

def _calculate_verts(self):
    # 如果正在计算顶点，则直接返回
    if self._calculating_verts.is_set():
        return
    # 设置正在计算顶点的标志
    self._calculating_verts.set()
    try:
        # 执行计算顶点的具体方法
        self._on_calculate_verts()
    finally:
        # 清除正在计算顶点的标志
        self._calculating_verts.clear()
    # 如果定义了边界回调函数，则调用它
    if callable(self.bounds_callback):
        self.bounds_callback()

def _calculate_cverts(self):
    # 如果正在计算顶点，则直接返回
    if self._calculating_verts.is_set():
        return
    # 等待之前的计算顶点完成
    while self._calculating_cverts.is_set():
        sleep(0)  # wait for previous calculation
    # 设置正在计算颜色顶点的标志
    self._calculating_cverts.set()
    try:
        # 执行计算颜色顶点的具体方法
        self._on_calculate_cverts()
    finally:
        # 清除正在计算颜色顶点的标志
        self._calculating_cverts.clear()
    # 返回计算顶点是否设置的状态
    def _get_calculating_verts(self):
        return self._calculating_verts.is_set()

    # 返回计算顶点位置的属性值
    def _get_calculating_verts_pos(self):
        return self._calculating_verts_pos

    # 返回计算顶点长度的属性值
    def _get_calculating_verts_len(self):
        return self._calculating_verts_len

    # 返回计算顶点颜色的状态
    def _get_calculating_cverts(self):
        return self._calculating_cverts.is_set()

    # 返回计算顶点颜色位置的属性值
    def _get_calculating_cverts_pos(self):
        return self._calculating_cverts_pos

    # 返回计算顶点颜色长度的属性值
    def _get_calculating_cverts_len(self):
        return self._calculating_cverts_len

    ## Property handlers

    # 返回样式属性值
    def _get_style(self):
        return self._style

    # 设置样式属性值，如果传入值为 None，则不进行操作；如果为 ''，则根据逻辑设置默认值
    @synchronized
    def _set_style(self, v):
        if v is None:
            return
        if v == '':
            step_max = 0
            for i in self.intervals:
                if i.v_steps is None:
                    continue
                step_max = max([step_max, int(i.v_steps)])
            v = ['both', 'solid'][step_max > 40]
        if v not in self.styles:
            raise ValueError("v should be there in self.styles")
        if v == self._style:
            return
        self._style = v

    # 返回颜色属性值
    def _get_color(self):
        return self._color

    # 设置颜色属性值，根据传入值的类型进行适当处理，同时触发颜色变更事件
    @synchronized
    def _set_color(self, v):
        try:
            if v is not None:
                if is_sequence(v):
                    v = ColorScheme(*v)
                else:
                    v = ColorScheme(v)
            if repr(v) == repr(self._color):
                return
            self._on_change_color(v)
            self._color = v
        except Exception as e:
            raise RuntimeError("Color change failed. "
                               "Reason: %s" % (str(e)))

    # 定义样式属性的属性(property)
    style = property(_get_style, _set_style)
    
    # 定义颜色属性的属性(property)
    color = property(_get_color, _set_color)

    # 定义计算顶点是否设置的属性(property)
    calculating_verts = property(_get_calculating_verts)
    
    # 定义计算顶点位置的属性(property)
    calculating_verts_pos = property(_get_calculating_verts_pos)
    
    # 定义计算顶点长度的属性(property)
    calculating_verts_len = property(_get_calculating_verts_len)

    # 定义计算顶点颜色是否设置的属性(property)
    calculating_cverts = property(_get_calculating_cverts)
    
    # 定义计算顶点颜色位置的属性(property)
    calculating_cverts_pos = property(_get_calculating_cverts_pos)
    
    # 定义计算顶点颜色长度的属性(property)
    calculating_cverts_len = property(_get_calculating_cverts_len)

    ## String representations

    # 返回对象的字符串表示，包括数据变量和主要别名
    def __str__(self):
        f = ", ".join(str(d) for d in self.d_vars)
        o = "'mode=%s'" % (self.primary_alias)
        return ", ".join([f, o])

    # 返回对象的详细字符串表示，包括数据变量、区间和主要属性
    def __repr__(self):
        f = ", ".join(str(d) for d in self.d_vars)
        i = ", ".join(str(i) for i in self.intervals)
        d = [('mode', self.primary_alias),
             ('color', str(self.color)),
             ('style', str(self.style))]

        o = "'%s'" % ("; ".join("%s=%s" % (k, v)
                                for k, v in d if v != 'None'))
        return ", ".join([f, i, o])
```