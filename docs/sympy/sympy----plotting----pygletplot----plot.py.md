# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\plot.py`

```
# 从 threading 模块中导入 RLock 类，用于实现可重入锁机制
from threading import RLock

# 尝试导入 pyglet 的 OpenGL 接口模块，并将其重命名为 pgl
try:
    import pyglet.gl as pgl
# 如果导入失败，抛出 ImportError 异常并显示错误消息
except ImportError:
    raise ImportError("pyglet is required for plotting.\n "
                      "visit https://pyglet.org/")

# 从 sympy.core.numbers 模块中导入 Integer 类
from sympy.core.numbers import Integer

# 从 sympy.external.gmpy 模块中导入 SYMPY_INTS 对象
from sympy.external.gmpy import SYMPY_INTS

# 从 sympy.geometry.entity 模块中导入 GeometryEntity 类
from sympy.geometry.entity import GeometryEntity

# 从 sympy.plotting.pygletplot.plot_axes 模块中导入 PlotAxes 类
from sympy.plotting.pygletplot.plot_axes import PlotAxes

# 从 sympy.plotting.pygletplot.plot_mode 模块中导入 PlotMode 类
from sympy.plotting.pygletplot.plot_mode import PlotMode

# 从 sympy.plotting.pygletplot.plot_object 模块中导入 PlotObject 类
from sympy.plotting.pygletplot.plot_object import PlotObject

# 从 sympy.plotting.pygletplot.plot_window 模块中导入 PlotWindow 类
from sympy.plotting.pygletplot.plot_window import PlotWindow

# 从 sympy.plotting.pygletplot.util 模块中导入 parse_option_string 函数
from sympy.plotting.pygletplot.util import parse_option_string

# 从 sympy.utilities.decorator 模块中导入 doctest_depends_on 装饰器
from sympy.utilities.decorator import doctest_depends_on

# 从 sympy.utilities.iterables 模块中导入 is_sequence 函数
from sympy.utilities.iterables import is_sequence

# 从 time 模块中导入 sleep 函数
from time import sleep

# 从 os 模块中导入 getcwd 函数和 listdir 函数
from os import getcwd, listdir

# 导入 ctypes 模块，用于处理 C 数据类型
import ctypes

# 使用 doctest_depends_on 装饰器，依赖于 'pyglet' 模块的存在
@doctest_depends_on(modules=('pyglet',))
class PygletPlot:
    """
    Plot Examples
    =============

    See examples/advanced/pyglet_plotting.py for many more examples.

    >>> from sympy.plotting.pygletplot import PygletPlot as Plot
    >>> from sympy.abc import x, y, z

    >>> Plot(x*y**3-y*x**3)
    [0]: -x**3*y + x*y**3, 'mode=cartesian'

    >>> p = Plot()
    >>> p[1] = x*y
    >>> p[1].color = z, (0.4,0.4,0.9), (0.9,0.4,0.4)

    >>> p = Plot()
    >>> p[1] =  x**2+y**2
    >>> p[2] = -x**2-y**2


    Variable Intervals
    ==================

    The basic format is [var, min, max, steps], but the
    syntax is flexible and arguments left out are taken
    from the defaults for the current coordinate mode:

    >>> Plot(x**2) # implies [x,-5,5,100]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [], []) # [x,-1,1,40], [y,-1,1,40]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2-y**2, [100], [100]) # [x,-1,1,100], [y,-1,1,100]
    [0]: x**2 - y**2, 'mode=cartesian'
    >>> Plot(x**2, [x,-13,13,100])
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [-13,13]) # [x,-13,13,100]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(x**2, [x,-13,13]) # [x,-13,13,10]
    [0]: x**2, 'mode=cartesian'
    >>> Plot(1*x, [], [x], mode='cylindrical')
    ... # [unbound_theta,0,2*Pi,40], [x,-1,1,20]
    [0]: x, 'mode=cartesian'


    Coordinate Modes
    ================

    Plot supports several curvilinear coordinate modes, and
    they independent for each plotted function. You can specify
    a coordinate mode explicitly with the 'mode' named argument,
    but it can be automatically determined for Cartesian or
    parametric plots, and therefore must only be specified for
    polar, cylindrical, and spherical modes.

    Specifically, Plot(function arguments) and Plot[n] =
    (function arguments) will interpret your arguments as a
    Cartesian plot if you provide one function and a parametric
    plot if you provide two or three functions. Similarly, the
    arguments will be interpreted as a curve if one variable is
    used, and a surface if two are used.

    Supported mode names by number of variables:
"""
    # 定义三个元组变量 parametric, cartesian, polar
    parametric, cartesian, polar

    # 将元组 cartesian, cylindrical 赋值为元组 polar, spherical 的值
    parametric, cartesian, cylindrical = polar, spherical

    # 创建一个 Plot 对象，设置模式为 'spherical'
    >>> Plot(1, mode='spherical')

    # Calculator-like Interface
    # =========================

    # 创建一个 Plot 对象，并设置可见性为 False
    >>> p = Plot(visible=False)

    # 定义一个函数 f = x**2
    >>> f = x**2

    # 将函数 f 添加到 Plot 对象的索引 1 处
    >>> p[1] = f

    # 将函数 f 的一阶导数添加到 Plot 对象的索引 2 处
    >>> p[2] = f.diff(x)

    # 将函数 f 的二阶导数添加到 Plot 对象的索引 3 处
    >>> p[3] = f.diff(x).diff(x)

    # 打印 Plot 对象的内容
    >>> p
    [1]: x**2, 'mode=cartesian'
    [2]: 2*x, 'mode=cartesian'
    [3]: 2, 'mode=cartesian'

    # 显示 Plot 对象的图形
    >>> p.show()

    # 清空 Plot 对象的内容
    >>> p.clear()

    # 打印空白的 Plot 对象
    >>> p
    <blank plot>

    # 将表达式 x**2 + y**2 添加到 Plot 对象的索引 1 处
    >>> p[1] =  x**2+y**2

    # 设置 Plot 对象索引 1 处的样式为 'solid'
    >>> p[1].style = 'solid'

    # 将表达式 -x**2 - y**2 添加到 Plot 对象的索引 2 处
    >>> p[2] = -x**2-y**2

    # 设置 Plot 对象索引 2 处的样式为 'wireframe'
    >>> p[2].style = 'wireframe'

    # 设置 Plot 对象索引 1 处的颜色为 z, (0.4,0.4,0.9), (0.9,0.4,0.4)
    >>> p[1].color = z, (0.4,0.4,0.9), (0.9,0.4,0.4)

    # 设置 Plot 对象索引 1 和索引 2 处的样式为 'both'
    >>> p[1].style = 'both'
    >>> p[2].style = 'both'

    # 关闭 Plot 对象
    >>> p.close()

    # Plot 窗口的键盘控制
    # =============================

    # 屏幕旋转：
    # X,Y 轴使用方向键，A,S,D,W，小键盘 4,6,8,2
    # Z 轴使用 Q,E，小键盘 7,9

    # 模型旋转：
    # Z 轴使用 Z,C，小键盘 1,3

    # 缩放：R,F，页面向上/向下，小键盘 +,-

    # 重置摄像头：X，小键盘 5

    # 摄像头预设：
    # XY 按键 F1
    # XZ 按键 F2
    # YZ 按键 F3
    # 透视模式按键 F4

    # 灵敏度调节器：SHIFT

    # 轴开关：
    # 可见性按键 F5
    # 颜色按键 F6

    # 关闭窗口：ESCAPE

    # =============================

    """

    # 根据需要依赖 pyglet 模块执行 doctest
    @doctest_depends_on(modules=('pyglet',))
    def __init__(self, *fargs, **win_args):
        """
        Positional Arguments
        ====================

        Any given positional arguments are used to
        initialize a plot function at index 1. In
        other words...

        >>> from sympy.plotting.pygletplot import PygletPlot as Plot
        >>> from sympy.abc import x
        >>> p = Plot(x**2, visible=False)

        ...is equivalent to...

        >>> p = Plot(visible=False)
        >>> p[1] = x**2

        Note that in earlier versions of the plotting
        module, you were able to specify multiple
        functions in the initializer. This functionality
        has been dropped in favor of better automatic
        plot plot_mode detection.


        Named Arguments
        ===============

        axes
            An option string of the form
            "key1=value1; key2 = value2" which
            can use the following options:

            style = ordinate
                none OR frame OR box OR ordinate

            stride = 0.25
                val OR (val_x, val_y, val_z)

            overlay = True (draw on top of plot)
                True OR False

            colored = False (False uses Black,
                             True uses colors
                             R,G,B = X,Y,Z)
                True OR False

            label_axes = False (display axis names
                                at endpoints)
                True OR False

        visible = True (show immediately
            True OR False


        The following named arguments are passed as
        arguments to window initialization:

        antialiasing = True
            True OR False

        ortho = False
            True OR False

        invert_mouse_zoom = False
            True OR False

        """
        # Register the plot modes
        from . import plot_modes # noqa

        # Store the window arguments internally
        self._win_args = win_args
        # Initialize window as None
        self._window = None

        # Initialize a reentrant lock for rendering
        self._render_lock = RLock()

        # Initialize dictionaries and lists for functions, plot objects, and screenshots
        self._functions = {}
        self._pobjects = []
        self._screenshot = ScreenShot(self)

        # Parse axes options from the provided arguments
        axe_options = parse_option_string(win_args.pop('axes', ''))
        # Create PlotAxes object based on parsed options and add to plot objects
        self.axes = PlotAxes(**axe_options)
        self._pobjects.append(self.axes)

        # Initialize the plot functions with positional arguments
        self[0] = fargs
        # Show the plot if 'visible' is True in window arguments
        if win_args.get('visible', True):
            self.show()

    ## Window Interfaces

    def show(self):
        """
        Creates and displays a plot window, or activates it
        (gives it focus) if it has already been created.
        """
        # Check if the window exists and has not been closed
        if self._window and not self._window.has_exit:
            # Activate the existing window
            self._window.activate()
        else:
            # Set 'visible' to True in window arguments
            self._win_args['visible'] = True
            # Reset resources associated with axes
            self.axes.reset_resources()

            # Uncomment the following block if a specific condition is met
            #if hasattr(self, '_doctest_depends_on'):
            #    self._win_args['runfromdoctester'] = True

            # Create a new PlotWindow object with current plot and window arguments
            self._window = PlotWindow(self, **self._win_args)
    def close(self):
        """
        Closes the plot window.
        """
        # 检查窗口是否存在，如果存在则关闭窗口
        if self._window:
            self._window.close()

    def saveimage(self, outfile=None, format='', size=(600, 500)):
        """
        Saves a screen capture of the plot window to an
        image file.

        If outfile is given, it can either be a path
        or a file object. Otherwise a png image will
        be saved to the current working directory.
        If the format is omitted, it is determined from
        the filename extension.
        """
        # 将绘图窗口的屏幕截图保存为图像文件
        self._screenshot.save(outfile, format, size)

    ## Function List Interfaces

    def clear(self):
        """
        Clears the function list of this plot.
        """
        # 获取渲染锁，清空函数列表，并调整所有边界
        self._render_lock.acquire()
        self._functions = {}
        self.adjust_all_bounds()
        self._render_lock.release()

    def __getitem__(self, i):
        """
        Returns the function at position i in the
        function list.
        """
        # 返回函数列表中位置为 i 的函数
        return self._functions[i]

    def __setitem__(self, i, args):
        """
        Parses and adds a PlotMode to the function
        list.
        """
        # 检查索引 i 是否为非负整数
        if not (isinstance(i, (SYMPY_INTS, Integer)) and i >= 0):
            raise ValueError("Function index must "
                             "be an integer >= 0.")

        # 如果 args 是 PlotObject 类型，则直接使用；否则根据参数创建 PlotMode 对象
        if isinstance(args, PlotObject):
            f = args
        else:
            if (not is_sequence(args)) or isinstance(args, GeometryEntity):
                args = [args]
            if len(args) == 0:
                return  # 没有给定参数

            kwargs = {"bounds_callback": self.adjust_all_bounds}
            f = PlotMode(*args, **kwargs)

        # 如果成功创建了函数对象，则加入到函数列表中
        if f:
            self._render_lock.acquire()
            self._functions[i] = f
            self._render_lock.release()
        else:
            raise ValueError("Failed to parse '%s'."
                    % ', '.join(str(a) for a in args))

    def __delitem__(self, i):
        """
        Removes the function in the function list at
        position i.
        """
        # 获取渲染锁，删除函数列表中位置为 i 的函数，并调整所有边界
        self._render_lock.acquire()
        del self._functions[i]
        self.adjust_all_bounds()
        self._render_lock.release()

    def firstavailableindex(self):
        """
        Returns the first unused index in the function list.
        """
        # 获取渲染锁，查找第一个未使用的索引，并返回
        i = 0
        self._render_lock.acquire()
        while i in self._functions:
            i += 1
        self._render_lock.release()
        return i

    def append(self, *args):
        """
        Parses and adds a PlotMode to the function
        list at the first available index.
        """
        # 在第一个可用索引处将 PlotMode 对象添加到函数列表中
        self.__setitem__(self.firstavailableindex(), args)

    def __len__(self):
        """
        Returns the number of functions in the function list.
        """
        # 返回函数列表中的函数数量
        return len(self._functions)

    def __iter__(self):
        """
        Allows iteration of the function list.
        """
        # 返回函数列表的迭代器
        return self._functions.itervalues()
    def __repr__(self):
        return str(self)



    def __str__(self):
        """
        Returns a string containing a new-line separated
        list of the functions in the function list.
        """
        # Initialize an empty string to build the result
        s = ""
        
        # Check if the function list is empty
        if len(self._functions) == 0:
            # If empty, indicate it's a blank plot
            s += "<blank plot>"
        else:
            # Acquire a lock to prevent rendering interference
            self._render_lock.acquire()
            
            # Construct a string representation for each function in _functions
            s += "\n".join(["%s[%i]: %s" % ("", i, str(self._functions[i]))
                            for i in self._functions])
            
            # Release the rendering lock after constructing the string
            self._render_lock.release()
        
        # Return the constructed string representation of the object
        return s



    def adjust_all_bounds(self):
        # Acquire a lock to prevent simultaneous rendering
        self._render_lock.acquire()
        
        # Reset bounding box of axes
        self.axes.reset_bounding_box()
        
        # Adjust bounds for each function in _functions
        for f in self._functions:
            self.axes.adjust_bounds(self._functions[f].bounds)
        
        # Release the rendering lock after adjustments
        self._render_lock.release()



    def wait_for_calculations(self):
        # Sleep for a minimal duration (0 seconds)
        sleep(0)
        
        # Acquire a lock to prevent concurrent rendering
        self._render_lock.acquire()
        
        # Iterate through each function in _functions
        for f in self._functions:
            # Define functions to check calculating vertices and cubic vertices
            a = self._functions[f]._get_calculating_verts
            b = self._functions[f]._get_calculating_cverts
            
            # Wait while either calculating vertices or cubic vertices are being computed
            while a() or b():
                sleep(0)  # Allow other processes to execute while waiting
        
        # Release the rendering lock after calculations are complete
        self._render_lock.release()
# 定义一个名为 ScreenShot 的类，用于截图操作
class ScreenShot:
    # 初始化方法，接受一个 plot 参数
    def __init__(self, plot):
        # 将 plot 参数保存到实例的 _plot 属性中
        self._plot = plot
        # 初始化截图请求状态为 False
        self.screenshot_requested = False
        # 初始化输出文件名为 None
        self.outfile = None
        # 初始化截图格式为空字符串
        self.format = ''
        # 初始化隐藏模式状态为 False
        self.invisibleMode = False
        # 初始化标志位为 0
        self.flag = 0

    # 定义 bool 方法，返回截图请求状态
    def __bool__(self):
        return self.screenshot_requested

    # 私有方法，执行保存操作
    def _execute_saving(self):
        # 如果标志位小于 3，则增加标志位并返回
        if self.flag < 3:
            self.flag += 1
            return

        # 获取窗口大小
        size_x, size_y = self._plot._window.get_size()
        # 计算图像数据大小
        size = size_x * size_y * 4 * ctypes.sizeof(ctypes.c_ubyte)
        # 创建一个字符串缓冲区来保存图像数据
        image = ctypes.create_string_buffer(size)
        # 使用 OpenGL 读取像素数据
        pgl.glReadPixels(0, 0, size_x, size_y, pgl.GL_RGBA, pgl.GL_UNSIGNED_BYTE, image)
        # 导入 PIL 库
        from PIL import Image
        # 从缓冲区创建图像
        im = Image.frombuffer('RGBA', (size_x, size_y),
                              image.raw, 'raw', 'RGBA', 0, 1)
        # 翻转图像并保存到指定文件中
        im.transpose(Image.FLIP_TOP_BOTTOM).save(self.outfile, self.format)

        # 重置标志位和截图请求状态
        self.flag = 0
        self.screenshot_requested = False
        # 如果处于隐藏模式，则关闭窗口
        if self.invisibleMode:
            self._plot._window.close()

    # 公有方法，用于设置保存截图的参数
    def save(self, outfile=None, format='', size=(600, 500)):
        # 设置输出文件名、格式和大小，并标记截图请求为 True
        self.outfile = outfile
        self.format = format
        self.size = size
        self.screenshot_requested = True

        # 如果窗口不存在或已经关闭，则设置窗口参数并重新创建窗口
        if not self._plot._window or self._plot._window.has_exit:
            self._plot._win_args['visible'] = False
            self._plot._win_args['width'] = size[0]
            self._plot._win_args['height'] = size[1]
            self._plot.axes.reset_resources()
            self._plot._window = PlotWindow(self._plot, **self._plot._win_args)
            self.invisibleMode = True

        # 如果未指定输出文件名，则创建一个唯一路径并打印
        if self.outfile is None:
            self.outfile = self._create_unique_path()
            print(self.outfile)

    # 私有方法，创建一个唯一的文件路径
    def _create_unique_path(self):
        # 获取当前工作目录和文件列表
        cwd = getcwd()
        l = listdir(cwd)
        path = ''
        i = 0
        # 循环直到找到一个不存在的文件名
        while True:
            if not 'plot_%s.png' % i in l:
                path = cwd + '/plot_%s.png' % i
                break
            i += 1
        return path
```