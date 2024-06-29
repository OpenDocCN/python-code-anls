# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_wx.py`

```py
"""
A wxPython backend for matplotlib.

Originally contributed by Jeremy O'Donoghue (jeremy@o-donoghue.com) and John
Hunter (jdhunter@ace.bsd.uchicago.edu).

Copyright (C) Jeremy O'Donoghue & John Hunter, 2003-4.
"""

# 导入必要的库和模块
import functools  # 导入 functools 模块，用于装饰器相关操作
import logging    # 导入 logging 模块，用于日志记录
import math       # 导入 math 模块，用于数学运算
import pathlib    # 导入 pathlib 模块，用于处理文件路径
import sys        # 导入 sys 模块，用于系统相关操作
import weakref    # 导入 weakref 模块，用于创建弱引用

# 导入第三方库
import numpy as np        # 导入 numpy 库，用于数值计算
import PIL.Image          # 导入 PIL.Image 模块，用于图像处理

# 导入 matplotlib 相关模块和类
import matplotlib as mpl
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase,
    GraphicsContextBase, MouseButton, NavigationToolbar2, RendererBase,
    TimerBase, ToolContainerBase, cursors,
    CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent)
from matplotlib import _api, cbook, backend_tools, _c_internal_utils
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

# 导入 wxPython 库及其相关模块
import wx
import wx.svg

# 设置日志记录器
_log = logging.getLogger(__name__)

# 屏幕上的实际每英寸像素数；应该与显示器相关；参考：
# http://groups.google.com/d/msg/comp.lang.postscript/-/omHAc9FEuAsJ?hl=en
PIXELS_PER_INCH = 75

# 使用 functools.lru_cache 装饰器创建 wxApp 实例，并保持引用，避免被垃圾回收
@functools.lru_cache(1)
def _create_wxapp():
    wxapp = wx.App(False)  # 创建一个不显示图形界面的 wx.App 实例
    wxapp.SetExitOnFrameDelete(True)  # 设置在主窗口关闭时退出应用程序
    cbook._setup_new_guiapp()  # 设置新的 GUI 应用程序
    # 设置进程级别的 DPI 意识。在 MSW（Windows） 下才生效
    _c_internal_utils.Win32_SetProcessDpiAwareness_max()
    return wxapp


class TimerWx(TimerBase):
    """Subclass of `.TimerBase` using wx.Timer events."""

    def __init__(self, *args, **kwargs):
        self._timer = wx.Timer()  # 创建 wx.Timer 实例
        self._timer.Notify = self._on_timer  # 设置定时器通知方法
        super().__init__(*args, **kwargs)

    def _timer_start(self):
        self._timer.Start(self._interval, self._single)  # 启动定时器

    def _timer_stop(self):
        self._timer.Stop()  # 停止定时器

    def _timer_set_interval(self):
        if self._timer.IsRunning():
            self._timer_start()  # 如果定时器正在运行，则使用新的间隔重新启动。


@_api.deprecated(
    "2.0", name="wx", obj_type="backend", removal="the future",
    alternative="wxagg",
    addendum="See the Matplotlib usage FAQ for more info on backends.")
class RendererWx(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles. It acts as the
    'renderer' instance used by many classes in the hierarchy.
    """
    # In wxPython, drawing is performed on a wxDC instance, which will
    # generally be mapped to the client area of the window displaying
    # the plot. Under wxPython, the wxDC instance has a wx.Pen which
    # describes the colour and weight of any lines drawn, and a wxBrush
    # which describes the fill colour of any closed polygon.

    # Font styles, families and weight.
    fontweights = {
        100: wx.FONTWEIGHT_LIGHT,  # 定义字体权重值为100的映射到 wxPython 轻字重
        200: wx.FONTWEIGHT_LIGHT,  # 定义字体权重值为200的映射到 wxPython 轻字重
        300: wx.FONTWEIGHT_LIGHT,  # 定义字体权重值为300的映射到 wxPython 轻字重
        400: wx.FONTWEIGHT_NORMAL,  # 定义字体权重值为400的映射到 wxPython 普通字重
        500: wx.FONTWEIGHT_NORMAL,  # 定义字体权重值为500的映射到 wxPython 普通字重
        600: wx.FONTWEIGHT_NORMAL,  # 定义字体权重值为600的映射到 wxPython 普通字重
        700: wx.FONTWEIGHT_BOLD,    # 定义字体权重值为700的映射到 wxPython 粗体
        800: wx.FONTWEIGHT_BOLD,    # 定义字体权重值为800的映射到 wxPython 粗体
        900: wx.FONTWEIGHT_BOLD,    # 定义字体权重值为900的映射到 wxPython 粗体
        'ultralight': wx.FONTWEIGHT_LIGHT,  # 定义 "ultralight" 字体映射到 wxPython 轻字重
        'light': wx.FONTWEIGHT_LIGHT,        # 定义 "light" 字体映射到 wxPython 轻字重
        'normal': wx.FONTWEIGHT_NORMAL,      # 定义 "normal" 字体映射到 wxPython 普通字重
        'medium': wx.FONTWEIGHT_NORMAL,      # 定义 "medium" 字体映射到 wxPython 普通字重
        'semibold': wx.FONTWEIGHT_NORMAL,    # 定义 "semibold" 字体映射到 wxPython 普通字重
        'bold': wx.FONTWEIGHT_BOLD,          # 定义 "bold" 字体映射到 wxPython 粗体
        'heavy': wx.FONTWEIGHT_BOLD,         # 定义 "heavy" 字体映射到 wxPython 粗体
        'ultrabold': wx.FONTWEIGHT_BOLD,     # 定义 "ultrabold" 字体映射到 wxPython 粗体
        'black': wx.FONTWEIGHT_BOLD,         # 定义 "black" 字体映射到 wxPython 粗体
    }

    fontangles = {
        'italic': wx.FONTSTYLE_ITALIC,    # 定义 "italic" 字体映射到 wxPython 斜体
        'normal': wx.FONTSTYLE_NORMAL,    # 定义 "normal" 字体映射到 wxPython 正常样式
        'oblique': wx.FONTSTYLE_SLANT,    # 定义 "oblique" 字体映射到 wxPython 斜体
    }

    # wxPython 允许跨平台使用字体风格，根据目标平台选择合适的字体风格。
    # 将一些标准字体名称映射到可移植的风格。
    fontnames = {
        'Sans': wx.FONTFAMILY_SWISS,         # 将 "Sans" 映射到 wxPython 瑞士风格字体
        'Roman': wx.FONTFAMILY_ROMAN,        # 将 "Roman" 映射到 wxPython 罗马风格字体
        'Script': wx.FONTFAMILY_SCRIPT,      # 将 "Script" 映射到 wxPython 脚本风格字体
        'Decorative': wx.FONTFAMILY_DECORATIVE,  # 将 "Decorative" 映射到 wxPython 装饰风格字体
        'Modern': wx.FONTFAMILY_MODERN,      # 将 "Modern" 映射到 wxPython 现代风格字体
        'Courier': wx.FONTFAMILY_MODERN,     # 将 "Courier" 映射到 wxPython 现代风格字体
        'courier': wx.FONTFAMILY_MODERN,     # 将 "courier" 映射到 wxPython 现代风格字体
    }

    def __init__(self, bitmap, dpi):
        """初始化一个 wxWindows 渲染器实例。"""
        super().__init__()
        _log.debug("%s - __init__()", type(self))
        self.width = bitmap.GetWidth()      # 获取位图的宽度
        self.height = bitmap.GetHeight()    # 获取位图的高度
        self.bitmap = bitmap                # 存储位图对象的引用
        self.fontd = {}                     # 初始化字体字典
        self.dpi = dpi                      # 存储 DPI（每英寸点数）信息
        self.gc = None                      # 初始化图形上下文为 None

    def flipy(self):
        # 继承的文档字符串
        return True

    def get_text_width_height_descent(self, s, prop, ismath):
        # 继承的文档字符串

        if ismath:
            s = cbook.strip_math(s)

        if self.gc is None:
            gc = self.new_gc()
        else:
            gc = self.gc
        gfx_ctx = gc.gfx_ctx
        font = self.get_wx_font(s, prop)        # 获取适合的 wxPython 字体对象
        gfx_ctx.SetFont(font, wx.BLACK)         # 设置字体和颜色（黑色）
        w, h, descent, leading = gfx_ctx.GetFullTextExtent(s)  # 获取文本的宽度、高度、下降、领导（行间距）

        return w, h, descent

    def get_canvas_width_height(self):
        # 继承的文档字符串
        return self.width, self.height        # 返回画布的宽度和高度

    def handle_clip_rectangle(self, gc):
        new_bounds = gc.get_clip_rectangle()          # 获取剪裁矩形区域
        if new_bounds is not None:
            new_bounds = new_bounds.bounds
        gfx_ctx = gc.gfx_ctx
        if gfx_ctx._lastcliprect != new_bounds:
            gfx_ctx._lastcliprect = new_bounds
            if new_bounds is None:
                gfx_ctx.ResetClip()                 # 如果新的剪裁区域为空，重置剪裁
            else:
                gfx_ctx.Clip(new_bounds[0],
                             self.height - new_bounds[1] - new_bounds[3],
                             new_bounds[2], new_bounds[3])   # 设置新的剪裁区域的坐标和尺寸
    # 创建一个用于转换路径的方法，将路径对象转换为 wxWidgets 的路径对象
    def convert_path(gfx_ctx, path, transform):
        # 创建一个空的 wxWidgets 路径对象
        wxpath = gfx_ctx.CreatePath()
        # 遍历路径对象中的各个段落和对应的代码
        for points, code in path.iter_segments(transform):
            # 如果是移动到指令，将点添加到路径中
            if code == Path.MOVETO:
                wxpath.MoveToPoint(*points)
            # 如果是直线到指令，添加直线到路径中
            elif code == Path.LINETO:
                wxpath.AddLineToPoint(*points)
            # 如果是三次贝塞尔曲线指令，添加二次贝塞尔曲线到路径中
            elif code == Path.CURVE3:
                wxpath.AddQuadCurveToPoint(*points)
            # 如果是四次贝塞尔曲线指令，添加三次贝塞尔曲线到路径中
            elif code == Path.CURVE4:
                wxpath.AddCurveToPoint(*points)
            # 如果是闭合多边形指令，闭合当前路径子路径
            elif code == Path.CLOSEPOLY:
                wxpath.CloseSubpath()
        # 返回转换后的 wxWidgets 路径对象
        return wxpath

    # 绘制路径的方法，将路径对象绘制到图形上下文中
    def draw_path(self, gc, path, transform, rgbFace=None):
        # docstring 继承
        # 选择当前图形上下文
        gc.select()
        # 处理剪切矩形区域
        self.handle_clip_rectangle(gc)
        # 获取图形上下文对象
        gfx_ctx = gc.gfx_ctx
        # 对变换进行缩放和平移操作
        transform = transform + \
            Affine2D().scale(1.0, -1.0).translate(0.0, self.height)
        # 转换路径对象为 wxWidgets 的路径对象
        wxpath = self.convert_path(gfx_ctx, path, transform)
        # 如果提供了填充颜色，设置填充刷子并绘制路径
        if rgbFace is not None:
            gfx_ctx.SetBrush(wx.Brush(gc.get_wxcolour(rgbFace)))
            gfx_ctx.DrawPath(wxpath)
        else:
            # 否则只描边路径
            gfx_ctx.StrokePath(wxpath)
        # 取消选择当前图形上下文
        gc.unselect()

    # 绘制图像的方法，将图像绘制到指定位置
    def draw_image(self, gc, x, y, im):
        # 获取剪切矩形区域的边界框
        bbox = gc.get_clip_rectangle()
        # 如果存在剪切矩形区域，则获取其边界
        if bbox is not None:
            l, b, w, h = bbox.bounds
        else:
            # 否则设置默认的左、底、宽和高
            l = 0
            b = 0
            w = self.width
            h = self.height
        # 获取图像的行数和列数
        rows, cols = im.shape[:2]
        # 从 RGBA 缓冲区创建 wxWidgets 的位图对象
        bitmap = wx.Bitmap.FromBufferRGBA(cols, rows, im.tobytes())
        # 选择当前图形上下文
        gc.select()
        # 将位图绘制到图形上下文中的指定位置
        gc.gfx_ctx.DrawBitmap(bitmap, int(l), int(self.height - b),
                              int(w), int(-h))
        # 取消选择当前图形上下文
        gc.unselect()

    # 绘制文本的方法，将文本绘制到指定位置
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring 继承

        # 如果需要绘制数学公式，则去除数学标记
        if ismath:
            s = cbook.strip_math(s)
        # 调试日志记录当前操作类型和方法
        _log.debug("%s - draw_text()", type(self))
        # 选择当前图形上下文
        gc.select()
        # 处理剪切矩形区域
        self.handle_clip_rectangle(gc)
        # 获取图形上下文对象
        gfx_ctx = gc.gfx_ctx

        # 获取 wxWidgets 字体对象和颜色
        font = self.get_wx_font(s, prop)
        color = gc.get_wxcolour(gc.get_rgb())
        gfx_ctx.SetFont(font, color)

        # 获取文本的宽度、高度和下降值
        w, h, d = self.get_text_width_height_descent(s, prop, ismath)
        x = int(x)
        y = int(y - h)

        # 如果角度为 0.0，则直接绘制文本
        if angle == 0.0:
            gfx_ctx.DrawText(s, x, y)
        else:
            # 否则根据角度绘制旋转文本
            rads = math.radians(angle)
            xo = h * math.sin(rads)
            yo = h * math.cos(rads)
            gfx_ctx.DrawRotatedText(s, x - xo, y - yo, rads)

        # 取消选择当前图形上下文
        gc.unselect()

    # 创建新的图形上下文对象的方法
    def new_gc(self):
        # docstring 继承
        # 调试日志记录当前操作类型和方法
        _log.debug("%s - new_gc()", type(self))
        # 创建基于 wxWidgets 的图形上下文对象
        self.gc = GraphicsContextWx(self.bitmap, self)
        # 选择当前图形上下文
        self.gc.select()
        # 取消选择当前图形上下文
        self.gc.unselect()
        # 返回创建的图形上下文对象
        return self.gc
    def get_wx_font(self, s, prop):
        """Return a wx font.  Cache font instances for efficiency."""
        # 记录调试信息，显示函数调用的类型
        _log.debug("%s - get_wx_font()", type(self))
        # 使用属性prop的哈希值作为字典的键，尝试从字典中获取已缓存的字体实例
        key = hash(prop)
        font = self.fontd.get(key)
        if font is not None:
            # 如果已缓存的字体实例存在，则直接返回
            return font
        # 将点数大小转换为像素大小，考虑屏幕分辨率以及点数与像素的换算关系
        size = self.points_to_pixels(prop.get_size_in_points())
        # 字体颜色由活动的 wx.Pen 决定
        # TODO: 可能需要缓存字体信息，提高效率
        # 创建新的 wx.Font 实例并缓存
        self.fontd[key] = font = wx.Font(
            pointSize=round(size),  # 设定字体的点大小，四舍五入取整
            family=self.fontnames.get(prop.get_name(), wx.ROMAN),  # 设定字体族名
            style=self.fontangles[prop.get_style()],  # 设定字体风格
            weight=self.fontweights[prop.get_weight()]  # 设定字体粗细
        )
        return font

    def points_to_pixels(self, points):
        # 继承文档字符串说明的功能，将点数转换为像素数
        return points * (PIXELS_PER_INCH / 72.0 * self.dpi / 72.0)
# GraphicsContextWx 类，继承自 GraphicsContextBase 类，提供绘图上下文的颜色、线条样式等设置
class GraphicsContextWx(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc.

    This class stores a reference to a wxMemoryDC, and a
    wxGraphicsContext that draws to it.  Creating a wxGraphicsContext
    seems to be fairly heavy, so these objects are cached based on the
    bitmap object that is passed in.

    The base GraphicsContext stores colors as an RGB tuple on the unit
    interval, e.g., (0.5, 0.0, 1.0).  wxPython uses an int interval, but
    since wxPython colour management is rather simple, I have not chosen
    to implement a separate colour manager class.
    """

    # _capd 和 _joind 是字典，用于将字符串形式的线段端点风格映射到 wxPython 的常量
    _capd = {'butt': wx.CAP_BUTT,
             'projecting': wx.CAP_PROJECTING,
             'round': wx.CAP_ROUND}

    # _joind 是字典，用于将字符串形式的连接风格映射到 wxPython 的常量
    _joind = {'bevel': wx.JOIN_BEVEL,
              'miter': wx.JOIN_MITER,
              'round': wx.JOIN_ROUND}

    # _cache 是弱引用字典，用于缓存 wxMemoryDC 和 wxGraphicsContext 对象的键值对
    _cache = weakref.WeakKeyDictionary()

    # 构造函数，初始化 GraphicsContextWx 对象
    def __init__(self, bitmap, renderer):
        super().__init__()  # 调用父类的构造函数
        # _log.debug 是日志记录器的调试信息输出
        _log.debug("%s - __init__(): %s", type(self), bitmap)

        # 从缓存中获取与 bitmap 对应的 wxMemoryDC 和 wxGraphicsContext 对象
        dc, gfx_ctx = self._cache.get(bitmap, (None, None))
        if dc is None:
            # 如果缓存中不存在，则创建 wxMemoryDC 和 wxGraphicsContext 对象，并加入缓存
            dc = wx.MemoryDC(bitmap)
            gfx_ctx = wx.GraphicsContext.Create(dc)
            gfx_ctx._lastcliprect = None
            self._cache[bitmap] = dc, gfx_ctx

        # 初始化对象的属性
        self.bitmap = bitmap
        self.dc = dc
        self.gfx_ctx = gfx_ctx
        self._pen = wx.Pen('BLACK', 1, wx.SOLID)  # 创建黑色实线画笔
        gfx_ctx.SetPen(self._pen)  # 设置 wxGraphicsContext 使用 _pen 画笔
        self.renderer = renderer  # 设置渲染器属性

    # 选择当前位图到 wxDC 实例中
    def select(self):
        """Select the current bitmap into this wxDC instance."""
        if sys.platform == 'win32':
            self.dc.SelectObject(self.bitmap)
            self.IsSelected = True  # 标记位图已选中状态

    # 选择 Null 位图到 wxDC 实例中
    def unselect(self):
        """Select a Null bitmap into this wxDC instance."""
        if sys.platform == 'win32':
            self.dc.SelectObject(wx.NullBitmap)
            self.IsSelected = False  # 标记位图未选中状态

    # 设置前景色，isRGBA 参数暂未使用
    def set_foreground(self, fg, isRGBA=None):
        # docstring inherited 继承文档字符串
        # Implementation note: wxPython has a separate concept of pen and
        # brush - the brush fills any outline trace left by the pen.
        # Here we set both to the same colour - if a figure is not to be
        # filled, the renderer will set the brush to be transparent
        # Same goes for text foreground...

        _log.debug("%s - set_foreground()", type(self))  # 记录调试信息

        self.select()  # 选择当前位图

        super().set_foreground(fg, isRGBA)  # 调用父类方法设置前景色

        # 设置 _pen 画笔的颜色为当前 RGB 值对应的 wxPython 颜色值
        self._pen.SetColour(self.get_wxcolour(self.get_rgb()))
        self.gfx_ctx.SetPen(self._pen)  # 在 wxGraphicsContext 中设置画笔
        self.unselect()  # 取消位图选择
    def set_linewidth(self, w):
        # 将传入的线宽参数转换为浮点数
        w = float(w)
        # 记录调试信息，显示设置线宽操作的对象类型
        _log.debug("%s - set_linewidth()", type(self))
        # 选择当前对象
        self.select()
        # 如果线宽在0到1之间，则设置为1
        if 0 < w < 1:
            w = 1
        # 调用父类方法设置线宽
        super().set_linewidth(w)
        # 将渲染器中的点转换为像素，确保线宽不为0
        lw = int(self.renderer.points_to_pixels(self._linewidth))
        if lw == 0:
            lw = 1
        # 设置绘制上下文的笔的宽度
        self._pen.SetWidth(lw)
        self.gfx_ctx.SetPen(self._pen)
        # 取消选择当前对象
        self.unselect()

    def set_capstyle(self, cs):
        # 记录调试信息，显示设置端点样式操作的对象类型
        _log.debug("%s - set_capstyle()", type(self))
        # 选择当前对象
        self.select()
        # 调用父类方法设置端点样式
        super().set_capstyle(cs)
        # 根据设置的端点样式设置绘制上下文的笔的端点样式
        self._pen.SetCap(GraphicsContextWx._capd[self._capstyle])
        self.gfx_ctx.SetPen(self._pen)
        # 取消选择当前对象
        self.unselect()

    def set_joinstyle(self, js):
        # 记录调试信息，显示设置连接点样式操作的对象类型
        _log.debug("%s - set_joinstyle()", type(self))
        # 选择当前对象
        self.select()
        # 调用父类方法设置连接点样式
        super().set_joinstyle(js)
        # 根据设置的连接点样式设置绘制上下文的笔的连接点样式
        self._pen.SetJoin(GraphicsContextWx._joind[self._joinstyle])
        self.gfx_ctx.SetPen(self._pen)
        # 取消选择当前对象
        self.unselect()

    def get_wxcolour(self, color):
        """Convert an RGB(A) color to a wx.Colour."""
        # 记录调试信息，显示获取 wx.Colour 对象的颜色转换操作的对象类型
        _log.debug("%s - get_wx_color()", type(self))
        # 将 RGB(A) 颜色转换为 wx.Colour 对象
        return wx.Colour(*[int(255 * x) for x in color])
class _FigureCanvasWxBase(FigureCanvasBase, wx.Panel):
    """
    The FigureCanvas contains the figure and does event handling.

    In the wxPython backend, it is derived from wxPanel, and (usually) lives
    inside a frame instantiated by a FigureManagerWx. The parent window
    probably implements a wx.Sizer to control the displayed control size - but
    we give a hint as to our preferred minimum size.
    """

    # 指定需要的交互式框架为 wx
    required_interactive_framework = "wx"
    
    # 定义计时器类为 TimerWx
    _timer_cls = TimerWx
    
    # 指定管理器类为 FigureManagerWx
    manager_class = _api.classproperty(lambda cls: FigureManagerWx)
    
    # 键值映射表，将 wxPython 中的键映射到对应的名称
    keyvald = {
        wx.WXK_CONTROL: 'control',
        wx.WXK_SHIFT: 'shift',
        wx.WXK_ALT: 'alt',
        wx.WXK_CAPITAL: 'caps_lock',
        wx.WXK_LEFT: 'left',
        wx.WXK_UP: 'up',
        wx.WXK_RIGHT: 'right',
        wx.WXK_DOWN: 'down',
        wx.WXK_ESCAPE: 'escape',
        wx.WXK_F1: 'f1',
        wx.WXK_F2: 'f2',
        wx.WXK_F3: 'f3',
        wx.WXK_F4: 'f4',
        wx.WXK_F5: 'f5',
        wx.WXK_F6: 'f6',
        wx.WXK_F7: 'f7',
        wx.WXK_F8: 'f8',
        wx.WXK_F9: 'f9',
        wx.WXK_F10: 'f10',
        wx.WXK_F11: 'f11',
        wx.WXK_F12: 'f12',
        wx.WXK_SCROLL: 'scroll_lock',
        wx.WXK_PAUSE: 'break',
        wx.WXK_BACK: 'backspace',
        wx.WXK_RETURN: 'enter',
        wx.WXK_INSERT: 'insert',
        wx.WXK_DELETE: 'delete',
        wx.WXK_HOME: 'home',
        wx.WXK_END: 'end',
        wx.WXK_PAGEUP: 'pageup',
        wx.WXK_PAGEDOWN: 'pagedown',
        wx.WXK_NUMPAD0: '0',
        wx.WXK_NUMPAD1: '1',
        wx.WXK_NUMPAD2: '2',
        wx.WXK_NUMPAD3: '3',
        wx.WXK_NUMPAD4: '4',
        wx.WXK_NUMPAD5: '5',
        wx.WXK_NUMPAD6: '6',
        wx.WXK_NUMPAD7: '7',
        wx.WXK_NUMPAD8: '8',
        wx.WXK_NUMPAD9: '9',
        wx.WXK_NUMPAD_ADD: '+',
        wx.WXK_NUMPAD_SUBTRACT: '-',
        wx.WXK_NUMPAD_MULTIPLY: '*',
        wx.WXK_NUMPAD_DIVIDE: '/',
        wx.WXK_NUMPAD_DECIMAL: 'dec',
        wx.WXK_NUMPAD_ENTER: 'enter',
        wx.WXK_NUMPAD_UP: 'up',
        wx.WXK_NUMPAD_RIGHT: 'right',
        wx.WXK_NUMPAD_DOWN: 'down',
        wx.WXK_NUMPAD_LEFT: 'left',
        wx.WXK_NUMPAD_PAGEUP: 'pageup',
        wx.WXK_NUMPAD_PAGEDOWN: 'pagedown',
        wx.WXK_NUMPAD_HOME: 'home',
        wx.WXK_NUMPAD_END: 'end',
        wx.WXK_NUMPAD_INSERT: 'insert',
        wx.WXK_NUMPAD_DELETE: 'delete',
    }
    def __init__(self, parent, id, figure=None):
        """
        Initialize a FigureWx instance.

        - Initialize the FigureCanvasBase and wxPanel parents.
        - Set event handlers for resize, paint, and keyboard and mouse
          interaction.
        """

        # 调用 FigureCanvasBase 的初始化方法，将图形对象传入
        FigureCanvasBase.__init__(self, figure)

        # 计算并设置画布的大小为图形边界框大小的上取整值
        size = wx.Size(*map(math.ceil, self.figure.bbox.size))

        # 如果不是在 Windows 平台，将大小转换为设备独立像素 (DIP)
        if wx.Platform != '__WXMSW__':
            size = parent.FromDIP(size)

        # 创建 wx.Panel 实例，设置父窗口、ID 和大小
        wx.Panel.__init__(self, parent, id, size=size)

        # 初始化一些属性
        self.bitmap = None  # 位图对象
        self._isDrawn = False  # 标记是否已绘制
        self._rubberband_rect = None  # 橡皮筋矩形框
        self._rubberband_pen_black = wx.Pen('BLACK', 1, wx.PENSTYLE_SHORT_DASH)  # 黑色破折号笔
        self._rubberband_pen_white = wx.Pen('WHITE', 1, wx.PENSTYLE_SOLID)  # 白色实线笔

        # 绑定各种事件处理函数
        self.Bind(wx.EVT_SIZE, self._on_size)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_key_down)
        self.Bind(wx.EVT_KEY_UP, self._on_key_up)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_LEFT_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_LEFT_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MIDDLE_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_MIDDLE_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_MIDDLE_UP, self._on_mouse_button)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_RIGHT_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_RIGHT_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX1_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX1_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX2_DOWN, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX2_UP, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX1_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSE_AUX2_DCLICK, self._on_mouse_button)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_ENTER_WINDOW, self._on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)

        # 绑定鼠标捕获相关事件处理函数
        self.Bind(wx.EVT_MOUSE_CAPTURE_CHANGED, self._on_capture_lost)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_capture_lost)

        # 设置背景样式为绘制，以减少闪烁
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(wx.WHITE)

        # 如果是 macOS 平台，进行初始缩放设置
        if wx.Platform == '__WXMAC__':
            dpiScale = self.GetDPIScaleFactor()  # 获取 DPI 缩放因子
            self.SetInitialSize(self.GetSize() * (1 / dpiScale))  # 设置初始大小
            self._set_device_pixel_ratio(dpiScale)  # 设置设备像素比例
    def Copy_to_Clipboard(self, event=None):
        """Copy bitmap of canvas to system clipboard."""
        # 创建一个位图数据对象
        bmp_obj = wx.BitmapDataObject()
        # 将位图对象设置为当前画布的位图
        bmp_obj.SetBitmap(self.bitmap)

        # 如果剪贴板没有打开，则尝试打开剪贴板
        if not wx.TheClipboard.IsOpened():
            open_success = wx.TheClipboard.Open()
            # 如果成功打开剪贴板，则设置位图数据并刷新剪贴板，最后关闭剪贴板
            if open_success:
                wx.TheClipboard.SetData(bmp_obj)
                wx.TheClipboard.Flush()
                wx.TheClipboard.Close()

    def _update_device_pixel_ratio(self, *args, **kwargs):
        # 在设备像素比变化的情况下，需要注意混合分辨率显示的问题
        if self._set_device_pixel_ratio(self.GetDPIScaleFactor()):
            self.draw()

    def draw_idle(self):
        # docstring inherited
        # 记录调试信息，表示正在执行 draw_idle() 方法
        _log.debug("%s - draw_idle()", type(self))
        self._isDrawn = False  # 强制重新绘制
        # 刷新画布，触发绘图事件，擦除背景设置为 False
        self.Refresh(eraseBackground=False)

    def flush_events(self):
        # docstring inherited
        # 使用 wx.Yield() 处理挂起的事件
        wx.Yield()

    def start_event_loop(self, timeout=0):
        # docstring inherited
        # 如果事件循环已经在运行，则抛出运行时错误
        if hasattr(self, '_event_loop'):
            raise RuntimeError("Event loop already running")
        # 创建一个定时器
        timer = wx.Timer(self, id=wx.ID_ANY)
        if timeout > 0:
            # 如果设置了超时时间，则启动定时器，并绑定停止事件循环的处理方法
            timer.Start(int(timeout * 1000), oneShot=True)
            self.Bind(wx.EVT_TIMER, self.stop_event_loop, id=timer.GetId())
        # 创建 GUI 事件循环对象并运行
        self._event_loop = wx.GUIEventLoop()
        self._event_loop.Run()
        timer.Stop()

    def stop_event_loop(self, event=None):
        # docstring inherited
        # 如果事件循环存在且正在运行，则退出事件循环
        if hasattr(self, '_event_loop'):
            if self._event_loop.IsRunning():
                self._event_loop.Exit()
            del self._event_loop

    def _get_imagesave_wildcards(self):
        """Return the wildcard string for the filesave dialog."""
        # 获取默认文件类型和支持的文件类型列表
        default_filetype = self.get_default_filetype()
        filetypes = self.get_supported_filetypes_grouped()
        # 对文件类型按名称排序
        sorted_filetypes = sorted(filetypes.items())
        wildcards = []
        extensions = []
        filter_index = 0
        # 遍历排序后的文件类型列表
        for i, (name, exts) in enumerate(sorted_filetypes):
            # 生成文件扩展名列表的通配符字符串
            ext_list = ';'.join(['*.%s' % ext for ext in exts])
            extensions.append(exts[0])
            wildcard = f'{name} ({ext_list})|{ext_list}'
            # 如果默认文件类型在当前文件类型组中，则设置过滤器索引
            if default_filetype in exts:
                filter_index = i
            wildcards.append(wildcard)
        wildcards = '|'.join(wildcards)
        return wildcards, extensions, filter_index
    def gui_repaint(self, drawDC=None):
        """
        Update the displayed image on the GUI canvas, using the supplied
        wx.PaintDC device context.
        """
        _log.debug("%s - gui_repaint()", type(self))
        # 检查self是否存在且在屏幕上显示，避免在窗口关闭后出现"wrapped C/C++ object has been deleted"的运行时错误。
        if not (self and self.IsShownOnScreen()):
            return
        if not drawDC:  # 如果未从OnPaint方法调用，则使用ClientDC
            drawDC = wx.ClientDC(self)
        # 对于'WX'后端在Windows上，如果位图被其他DC使用（参见GraphicsContextWx._cache），需要进行转换。
        bmp = (self.bitmap.ConvertToImage().ConvertToBitmap()
               if wx.Platform == '__WXMSW__'
                  and isinstance(self.figure.canvas.get_renderer(), RendererWx)
               else self.bitmap)
        drawDC.DrawBitmap(bmp, 0, 0)
        if self._rubberband_rect is not None:
            # 一些版本的wx+python不支持在此处使用numpy.float64。
            x0, y0, x1, y1 = map(round, self._rubberband_rect)
            rect = [(x0, y0, x1, y0), (x1, y0, x1, y1),
                    (x0, y0, x0, y1), (x0, y1, x1, y1)]
            # 绘制橡皮筋效果的矩形边框
            drawDC.DrawLineList(rect, self._rubberband_pen_white)
            drawDC.DrawLineList(rect, self._rubberband_pen_black)

    filetypes = {
        **FigureCanvasBase.filetypes,
        'bmp': 'Windows bitmap',
        'jpeg': 'JPEG',
        'jpg': 'JPEG',
        'pcx': 'PCX',
        'png': 'Portable Network Graphics',
        'tif': 'Tagged Image Format File',
        'tiff': 'Tagged Image Format File',
        'xpm': 'X pixmap',
    }

    def _on_paint(self, event):
        """Called when wxPaintEvt is generated."""
        _log.debug("%s - _on_paint()", type(self))
        # 创建绘图上下文对象drawDC，用于绘制窗口内容
        drawDC = wx.PaintDC(self)
        if not self._isDrawn:
            # 如果还未绘制过，则调用draw方法进行绘制
            self.draw(drawDC=drawDC)
        else:
            # 否则调用gui_repaint方法更新显示图像
            self.gui_repaint(drawDC=drawDC)
        # 销毁绘图上下文对象，释放资源
        drawDC.Destroy()
    def _on_size(self, event):
        """
        Called when wxEventSize is generated.

        In this application we attempt to resize to fit the window, so it
        is better to take the performance hit and redraw the whole window.
        """
        # 调用方法用于更新设备像素比例
        self._update_device_pixel_ratio()
        # 记录调试日志，表示响应大小变化事件
        _log.debug("%s - _on_size()", type(self))
        # 获取父级窗口的 Sizer
        sz = self.GetParent().GetSizer()
        if sz:
            # 获取当前窗口在 Sizer 中的信息
            si = sz.GetItem(self)
        if sz and si and not si.Proportion and not si.Flag & wx.EXPAND:
            # 由 Sizer 管理，但是大小是固定的
            size = self.GetMinSize()
        else:
            # 大小是可变的
            size = self.GetClientSize()
            # 不允许大小小于最小尺寸
            size.IncTo(self.GetMinSize())
        if getattr(self, "_width", None):
            if size == (self._width, self._height):
                # 大小没有变化
                return
        # 记录当前的宽度和高度
        self._width, self._height = size
        # 标记未绘制
        self._isDrawn = False

        if self._width <= 1 or self._height <= 1:
            return  # 空图形

        # 根据当前窗口大小创建一个新的位图
        dpival = self.figure.dpi
        if not wx.Platform == '__WXMSW__':
            # 获取 DPI 缩放因子
            scale = self.GetDPIScaleFactor()
            dpival /= scale
        winch = self._width / dpival
        hinch = self._height / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)

        # 渲染将在关联的绘制事件中进行
        # 这里只需确保整个背景被重新绘制
        self.Refresh(eraseBackground=False)
        # 处理调整大小事件
        ResizeEvent("resize_event", self)._process()
        # 绘制空闲图形
        self.draw_idle()

    @staticmethod
    def _mpl_modifiers(event=None, *, exclude=None):
        """
        Determine the modifiers related to a wx event or current key state.

        Returns a list of modifier names based on the event or current key state.
        """
        # 定义修改键和对应的 wxPython 常量
        mod_table = [
            ("ctrl", wx.MOD_CONTROL, wx.WXK_CONTROL),
            ("alt", wx.MOD_ALT, wx.WXK_ALT),
            ("shift", wx.MOD_SHIFT, wx.WXK_SHIFT),
        ]
        if event is not None:
            # 如果有事件，获取事件的修改键状态
            modifiers = event.GetModifiers()
            # 返回符合条件的修改键名称列表
            return [name for name, mod, key in mod_table
                    if modifiers & mod and exclude != key]
        else:
            # 没有事件时，获取当前键盘状态的修改键名称列表
            return [name for name, mod, key in mod_table
                    if wx.GetKeyState(key)]

    def _get_key(self, event):
        """
        Determine and return the key associated with a wx key event.

        Returns the key combination string based on the event.
        """
        # 获取事件的键码
        keyval = event.KeyCode
        # 如果键码存在于键值字典中，则返回对应的键名
        if keyval in self.keyvald:
            key = self.keyvald[keyval]
        elif keyval < 256:
            # 如果键码小于 256，将其转换为字符
            key = chr(keyval)
            # wxPython 总是返回大写字符，如果没有按下 Shift 键，则转换为小写
            if not event.ShiftDown():
                key = key.lower()
        else:
            return None
        # 获取事件中的修改键列表
        mods = self._mpl_modifiers(event, exclude=keyval)
        # 如果 Shift 在修改键列表中并且键是大写，则移除 Shift
        if "shift" in mods and key.isupper():
            mods.remove("shift")
        # 返回修饰键和键名的组合字符串
        return "+".join([*mods, key])
    def _mpl_coords(self, pos=None):
        """
        Convert a wx position, defaulting to the current cursor position, to
        Matplotlib coordinates.
        """
        if pos is None:
            # 如果未提供位置参数，则获取当前鼠标状态
            pos = wx.GetMouseState()
            # 将鼠标位置转换为窗口客户区坐标系下的坐标
            x, y = self.ScreenToClient(pos.X, pos.Y)
        else:
            # 使用提供的位置参数作为坐标
            x, y = pos.X, pos.Y
        
        # 翻转 y 坐标，以便使 y=0 对应于画布底部
        if not wx.Platform == '__WXMSW__':
            # 获取 DPI 缩放因子
            scale = self.GetDPIScaleFactor()
            # 返回经过缩放后的 x 坐标和翻转后的 y 坐标
            return x * scale, self.figure.bbox.height - y * scale
        else:
            # 对于 Windows 平台，仅返回翻转后的 y 坐标，不进行 DPI 缩放
            return x, self.figure.bbox.height - y

    def _on_key_down(self, event):
        """Capture key press."""
        # 创建 KeyEvent 对象以处理按键按下事件，并传递相关参数
        KeyEvent("key_press_event", self,
                 self._get_key(event), *self._mpl_coords(),
                 guiEvent=event)._process()
        # 调用事件的默认处理方法
        if self:
            event.Skip()

    def _on_key_up(self, event):
        """Release key."""
        # 创建 KeyEvent 对象以处理按键释放事件，并传递相关参数
        KeyEvent("key_release_event", self,
                 self._get_key(event), *self._mpl_coords(),
                 guiEvent=event)._process()
        # 调用事件的默认处理方法
        if self:
            event.Skip()

    def set_cursor(self, cursor):
        # docstring inherited
        # 根据给定的 cursor 参数设置对应的光标类型
        cursor = wx.Cursor(_api.check_getitem({
            cursors.MOVE: wx.CURSOR_HAND,
            cursors.HAND: wx.CURSOR_HAND,
            cursors.POINTER: wx.CURSOR_ARROW,
            cursors.SELECT_REGION: wx.CURSOR_CROSS,
            cursors.WAIT: wx.CURSOR_WAIT,
            cursors.RESIZE_HORIZONTAL: wx.CURSOR_SIZEWE,
            cursors.RESIZE_VERTICAL: wx.CURSOR_SIZENS,
        }, cursor=cursor))
        # 设置窗口的光标
        self.SetCursor(cursor)
        # 刷新窗口以更新光标显示
        self.Refresh()

    def _set_capture(self, capture=True):
        """Control wx mouse capture."""
        # 如果窗口当前已经捕获了鼠标事件，则释放鼠标捕获
        if self.HasCapture():
            self.ReleaseMouse()
        # 根据 capture 参数决定是否重新捕获鼠标事件
        if capture:
            self.CaptureMouse()

    def _on_capture_lost(self, event):
        """Capture changed or lost"""
        # 当鼠标捕获状态改变或丢失时，调用 _set_capture 方法释放鼠标捕获
        self._set_capture(False)
    def _on_mouse_button(self, event):
        """处理鼠标按钮事件的回调函数。"""
        event.Skip()  # 跳过事件处理，继续传递事件
        self._set_capture(event.ButtonDown() or event.ButtonDClick())  # 设置捕获状态，检查是否按下或双击按钮
        x, y = self._mpl_coords(event)  # 获取鼠标事件的Matplotlib坐标
        button_map = {  # 映射wxPython鼠标按钮到Matplotlib鼠标按钮
            wx.MOUSE_BTN_LEFT: MouseButton.LEFT,
            wx.MOUSE_BTN_MIDDLE: MouseButton.MIDDLE,
            wx.MOUSE_BTN_RIGHT: MouseButton.RIGHT,
            wx.MOUSE_BTN_AUX1: MouseButton.BACK,
            wx.MOUSE_BTN_AUX2: MouseButton.FORWARD,
        }
        button = event.GetButton()  # 获取触发事件的鼠标按钮
        button = button_map.get(button, button)  # 映射或保持原始按钮
        modifiers = self._mpl_modifiers(event)  # 获取事件的修饰键状态
        if event.ButtonDown():  # 如果是按钮按下事件
            MouseEvent("button_press_event", self, x, y, button,
                       modifiers=modifiers, guiEvent=event)._process()  # 创建并处理按钮按下事件
        elif event.ButtonDClick():  # 如果是按钮双击事件
            MouseEvent("button_press_event", self, x, y, button,
                       dblclick=True, modifiers=modifiers,
                       guiEvent=event)._process()  # 创建并处理按钮双击事件
        elif event.ButtonUp():  # 如果是按钮释放事件
            MouseEvent("button_release_event", self, x, y, button,
                       modifiers=modifiers, guiEvent=event)._process()  # 创建并处理按钮释放事件

    def _on_mouse_wheel(self, event):
        """处理鼠标滚轮事件的回调函数，转换为Matplotlib事件。"""
        x, y = self._mpl_coords(event)  # 获取鼠标事件的Matplotlib坐标
        # 将delta/rotation/rate转换为浮点步进大小
        step = event.LinesPerAction * event.WheelRotation / event.WheelDelta
        event.Skip()  # 跳过事件处理，继续传递事件
        # 对于Mac系统，每次滚轮事件会产生两个事件；跳过第二个事件。
        if wx.Platform == '__WXMAC__':
            if not hasattr(self, '_skipwheelevent'):
                self._skipwheelevent = True
            elif self._skipwheelevent:
                self._skipwheelevent = False
                return  # 不处理事件，直接返回
            else:
                self._skipwheelevent = True
        MouseEvent("scroll_event", self, x, y, step=step,
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()  # 创建并处理滚轮事件

    def _on_motion(self, event):
        """处理鼠标移动事件的回调函数。"""
        event.Skip()  # 跳过事件处理，继续传递事件
        MouseEvent("motion_notify_event", self,
                   *self._mpl_coords(event),
                   modifiers=self._mpl_modifiers(event),
                   guiEvent=event)._process()  # 创建并处理鼠标移动事件

    def _on_enter(self, event):
        """处理鼠标进入窗口事件的回调函数。"""
        event.Skip()  # 跳过事件处理，继续传递事件
        LocationEvent("figure_enter_event", self,
                      *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(),
                      guiEvent=event)._process()  # 创建并处理鼠标进入窗口事件

    def _on_leave(self, event):
        """处理鼠标离开窗口事件的回调函数。"""
        event.Skip()  # 跳过事件处理，继续传递事件
        LocationEvent("figure_leave_event", self,
                      *self._mpl_coords(event),
                      modifiers=self._mpl_modifiers(),
                      guiEvent=event)._process()  # 创建并处理鼠标离开窗口事件
# 定义一个继承自 _FigureCanvasWxBase 的 FigureCanvasWx 类，用于在 Wx 画布上渲染图形，使用了已废弃的 Wx 渲染器。

    # 渲染图形到画布上，使用指定的绘图设备对象 drawDC，如果未指定则使用先前定义的渲染器。
    def draw(self, drawDC=None):
        """
        使用 RendererWx 实例 renderer 渲染图形，如果未指定渲染器则使用先前定义的。
        """
        _log.debug("%s - draw()", type(self))  # 记录调试信息到日志中
        self.renderer = RendererWx(self.bitmap, self.figure.dpi)  # 使用给定的位图和 DPI 创建 RendererWx 实例
        self.figure.draw(self.renderer)  # 使用创建的渲染器绘制图形
        self._isDrawn = True  # 标记图形已经被绘制
        self.gui_repaint(drawDC=drawDC)  # 更新 GUI 显示

    # 将图形保存为指定类型的图像文件，使用给定的文件名
    def _print_image(self, filetype, filename):
        bitmap = wx.Bitmap(math.ceil(self.figure.bbox.width),  # 创建指定宽度和高度的位图
                           math.ceil(self.figure.bbox.height))
        self.figure.draw(RendererWx(bitmap, self.figure.dpi))  # 使用新创建的位图和 DPI 绘制图形
        saved_obj = (bitmap.ConvertToImage()  # 将位图转换为图像对象
                     if cbook.is_writable_file_like(filename)  # 如果文件名指定的是可写文件
                     else bitmap)  # 否则直接使用位图对象
        if not saved_obj.SaveFile(filename, filetype):  # 将图像对象保存为指定类型的文件
            raise RuntimeError(f'Could not save figure to {filename}')  # 如果保存失败则抛出异常
        # 在此处调用 draw() 是必要的，因为关于最后渲染器的某些状态会分散在艺术家的绘制方法中。
        # 不要在确认这些状态已经清理之前删除 draw() 调用。否则艺术家的 contains() 方法将失败。
        if self._isDrawn:  # 如果图形已经绘制过
            self.draw()  # 重新绘制以确保状态正确
        # 如果 self 存在，避免在窗口关闭后出现 "wrapped C/C++ object has been deleted" 的 RuntimeError。
        if self:  # 如果对象仍然存在
            self.Refresh()  # 刷新对象的显示状态

    # 定义了几个部分特化方法，用于打印不同类型的图像文件
    print_bmp = functools.partialmethod(  # 打印 BMP 格式图像的特化方法
        _print_image, wx.BITMAP_TYPE_BMP)
    print_jpeg = print_jpg = functools.partialmethod(  # 打印 JPEG 格式图像的特化方法
        _print_image, wx.BITMAP_TYPE_JPEG)
    print_pcx = functools.partialmethod(  # 打印 PCX 格式图像的特化方法
        _print_image, wx.BITMAP_TYPE_PCX)
    print_png = functools.partialmethod(  # 打印 PNG 格式图像的特化方法
        _print_image, wx.BITMAP_TYPE_PNG)
    print_tiff = print_tif = functools.partialmethod(  # 打印 TIFF 格式图像的特化方法
        _print_image, wx.BITMAP_TYPE_TIF)
    print_xpm = functools.partialmethod(  # 打印 XPM 格式图像的特化方法
        _print_image, wx.BITMAP_TYPE_XPM)
    # 初始化方法，接受一个数字 num 和一个图形 fig，以及一个必须使用的画布类 canvas_class
    def __init__(self, num, fig, *, canvas_class):
        # 在非 Windows 平台上，显式设置位置以修复某些 Linux 平台上的定位 bug
        if wx.Platform == '__WXMSW__':
            pos = wx.DefaultPosition  # 如果是 Windows 平台，则使用默认位置
        else:
            pos = wx.Point(20, 20)  # 如果不是 Windows 平台，则设置位置为 (20, 20)

        # 调用父类的初始化方法，设置父窗口为 None，ID 为 -1，位置为 pos
        super().__init__(parent=None, id=-1, pos=pos)

        # 记录调试信息，标记初始化函数的调用
        _log.debug("%s - __init__()", type(self))

        # 设置窗口图标
        _set_frame_icon(self)

        # 创建一个 canvas 对象，使用传入的 canvas_class 类创建，并将图形 fig 绑定到 canvas 上
        self.canvas = canvas_class(self, -1, fig)

        # 创建 FigureManagerWx 对象，管理 self.canvas 的图形，传入 num 和 self 作为参数
        manager = FigureManagerWx(self.canvas, num, self)

        # 获取 canvas 的工具栏对象
        toolbar = self.canvas.manager.toolbar

        # 如果工具栏不为空，则将工具栏设置为窗口的工具栏
        if toolbar is not None:
            self.SetToolBar(toolbar)

        # 在 Windows 平台上，必须在添加工具栏之后设置 canvas 的大小；
        # 否则工具栏会进一步调整 canvas 的大小
        w, h = map(math.ceil, fig.bbox.size)  # 计算图形的大小并向上取整
        self.canvas.SetInitialSize(self.FromDIP(wx.Size(w, h)))  # 设置 canvas 的初始大小
        self.canvas.SetMinSize(self.FromDIP(wx.Size(2, 2)))  # 设置 canvas 的最小大小
        self.canvas.SetFocus()  # 设置 canvas 获得焦点

        self.Fit()  # 调整窗口大小以适应其内容

        # 绑定窗口关闭事件处理方法 _on_close
        self.Bind(wx.EVT_CLOSE, self._on_close)

    # 窗口关闭事件处理方法
    def _on_close(self, event):
        # 记录调试信息，标记窗口关闭事件的处理
        _log.debug("%s - on_close()", type(self))

        # 创建 CloseEvent 对象，处理关闭事件，将 self.canvas 作为参数传入
        CloseEvent("close_event", self.canvas)._process()

        # 停止 canvas 的事件循环
        self.canvas.stop_event_loop()

        # 将 FigureManagerWx 的 frame 属性设置为 None，防止从 FigureManagerWx.destroy() 中重复关闭该窗口
        self.canvas.manager.frame = None

        # 从 Gcf.figs 中销毁 self.canvas.manager 对象
        Gcf.destroy(self.canvas.manager)

        try:
            # 尝试断开与 self.canvas.toolbar._id_drag 的连接，处理拖动事件
            self.canvas.mpl_disconnect(self.canvas.toolbar._id_drag)
        except AttributeError:
            # 如果没有工具栏，则捕获 AttributeError 异常
            pass

        # 允许事件继续传播，销毁窗口及其子控件
        event.Skip()
class FigureManagerWx(FigureManagerBase):
    """
    Container/controller for the FigureCanvas and GUI frame.

    It is instantiated by Gcf whenever a new figure is created.  Gcf is
    responsible for managing multiple instances of FigureManagerWx.

    Attributes
    ----------
    canvas : `FigureCanvas`
        a FigureCanvasWx(wx.Panel) instance
    window : wxFrame
        a wxFrame instance - wxpython.org/Phoenix/docs/html/Frame.html
    """

    def __init__(self, canvas, num, frame):
        # 记录调试信息到日志，显示初始化函数的类名
        _log.debug("%s - __init__()", type(self))
        self.frame = self.window = frame  # 将传入的 frame 参数同时赋值给 self.frame 和 self.window
        super().__init__(canvas, num)  # 调用父类的初始化函数

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        """
        Create a FigureManagerWx instance with a given canvas class, figure, and number.

        Parameters
        ----------
        canvas_class : class
            Class of the canvas to use.
        figure : `matplotlib.figure.Figure`
            The figure to manage.
        num : int
            The figure number.

        Returns
        -------
        manager : FigureManagerWx
            The created FigureManagerWx instance.
        """
        # docstring inherited
        wxapp = wx.GetApp() or _create_wxapp()  # 获取当前 wx 应用实例，如果不存在则创建一个
        frame = FigureFrameWx(num, figure, canvas_class=canvas_class)  # 创建 FigureFrameWx 实例
        manager = figure.canvas.manager  # 获取 figure 对应的 canvas 管理器
        if mpl.is_interactive():  # 检查是否处于交互模式
            manager.frame.Show()  # 显示管理器的 frame
            figure.canvas.draw_idle()  # 绘制画布内容
        return manager

    @classmethod
    def start_main_loop(cls):
        """
        Start the main event loop of the wx application if it's not already running.
        """
        if not wx.App.IsMainLoopRunning():  # 检查主事件循环是否在运行
            wxapp = wx.GetApp()  # 获取当前 wx 应用实例
            if wxapp is not None:
                wxapp.MainLoop()  # 运行主事件循环

    def show(self):
        """
        Show the GUI frame associated with this manager, draw the canvas, and optionally raise the window.
        """
        self.frame.Show()  # 显示关联的 GUI 窗口
        self.canvas.draw()  # 绘制画布
        if mpl.rcParams['figure.raise_window']:  # 如果设置要求窗口提升到前台
            self.frame.Raise()  # 提升窗口到前台显示

    def destroy(self, *args):
        """
        Destroy the GUI frame associated with this manager.

        Parameters
        ----------
        *args : tuple
            Additional arguments (ignored in this method).
        """
        _log.debug("%s - destroy()", type(self))  # 记录调试信息到日志，显示销毁函数的类名
        frame = self.frame  # 获取关联的 GUI 窗口
        if frame:  # 如果窗口存在
            # 由于可能从非 GUI 线程调用，例如 plt.close 使用 wx.CallAfter 来确保线程安全
            wx.CallAfter(frame.Close)

    def full_screen_toggle(self):
        """
        Toggle full screen mode of the GUI frame associated with this manager.
        """
        self.frame.ShowFullScreen(not self.frame.IsFullScreen())  # 切换全屏模式状态

    def get_window_title(self):
        """
        Get the title of the GUI frame associated with this manager.

        Returns
        -------
        str
            The window title.
        """
        return self.window.GetTitle()  # 获取窗口标题

    def set_window_title(self, title):
        """
        Set the title of the GUI frame associated with this manager.

        Parameters
        ----------
        title : str
            The new title for the window.
        """
        self.window.SetTitle(title)  # 设置窗口标题

    def resize(self, width, height):
        """
        Resize the GUI frame associated with this manager to the specified width and height.

        Parameters
        ----------
        width : int
            New width of the frame.
        height : int
            New height of the frame.
        """
        # 直接使用 SetClientSize 在 Windows 上无法正确处理工具栏，使用 ClientToWindowSize 来确保正确调整大小
        self.window.SetSize(self.window.ClientToWindowSize(wx.Size(
            math.ceil(width), math.ceil(height))))  # 设置窗口大小
    # 初始化方法，创建一个工具栏对象，并设置其样式为底部布局
    def __init__(self, canvas, coordinates=True, *, style=wx.TB_BOTTOM):
        # 调用父类 wx.ToolBar 的初始化方法，将工具栏添加到 canvas 的父窗口中
        wx.ToolBar.__init__(self, canvas.GetParent(), -1, style=style)
        
        # 如果运行平台是 macOS，并且开启了高分辨率显示，根据 DPI 缩放因子设置工具栏图标大小
        if wx.Platform == '__WXMAC__':
            self.SetToolBitmapSize(self.GetToolBitmapSize()*self.GetDPIScaleFactor())

        # 存储工具栏项的 wx ID 的字典
        self.wx_ids = {}
        
        # 遍历工具栏项列表，每个项包含文本、工具提示、图像文件和回调函数
        for text, tooltip_text, image_file, callback in self.toolitems:
            # 如果文本为 None，添加一个分隔符到工具栏并继续下一次循环
            if text is None:
                self.AddSeparator()
                continue
            
            # 构建工具栏项并获取其 wx ID，根据文本、图像文件和是否为 "Pan" 或 "Zoom" 设置不同的类型
            self.wx_ids[text] = (
                self.AddTool(
                    -1,
                    bitmap=self._icon(f"{image_file}.svg"),
                    bmpDisabled=wx.NullBitmap,
                    label=text, shortHelp=tooltip_text,
                    kind=(wx.ITEM_CHECK if text in ["Pan", "Zoom"]
                          else wx.ITEM_NORMAL))
                .Id)
            
            # 将工具栏项的事件绑定到相应的回调函数上
            self.Bind(wx.EVT_TOOL, getattr(self, callback),
                      id=self.wx_ids[text])

        # 根据 coordinates 参数决定是否显示坐标信息控件
        self._coordinates = coordinates
        if self._coordinates:
            # 添加可伸缩空间到工具栏
            self.AddStretchableSpace()
            # 创建一个静态文本控件，用于显示坐标信息，并将其添加到工具栏
            self._label_text = wx.StaticText(self, style=wx.ALIGN_RIGHT)
            self.AddControl(self._label_text)

        # 实现工具栏的布局
        self.Realize()

        # 调用 NavigationToolbar2 的初始化方法，传入 canvas 对象
        NavigationToolbar2.__init__(self, canvas)

    # 静态方法，根据图像文件名构建适用于 wx 的位图对象
    @staticmethod
    def _icon(name):
        """
        Construct a `wx.Bitmap` suitable for use as icon from an image file
        *name*, including the extension and relative to Matplotlib's "images"
        data directory.
        """
        # 尝试获取系统外观是否为暗色模式（仅适用于 wxPython >= 4.1）
        try:
            dark = wx.SystemSettings.GetAppearance().IsDark()
        except AttributeError:  # wxpython < 4.1
            # 处理旧版本 wxPython 的兼容性问题，模拟 IsDark() 方法的行为
            bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
            bg_lum = (.299 * bg.red + .587 * bg.green + .114 * bg.blue) / 255
            fg_lum = (.299 * fg.red + .587 * fg.green + .114 * fg.blue) / 255
            dark = fg_lum - bg_lum > .2
        
        # 获取图像文件的路径，假设在 Matplotlib 的 "images" 数据目录中
        path = cbook._get_data_path('images', name)
        
        # 如果图像文件是 SVG 格式，根据外观模式调整颜色并生成 wx.Bitmap 对象
        if path.suffix == '.svg':
            svg = path.read_bytes()
            if dark:
                svg = svg.replace(b'fill:black;', b'fill:white;')
            toolbarIconSize = wx.ArtProvider().GetDIPSizeHint(wx.ART_TOOLBAR)
            return wx.BitmapBundle.FromSVG(svg, toolbarIconSize)
        else:
            # 如果是其他格式（如 PNG），使用 PIL 库打开图像，并确保转换为 RGBA 格式
            pilimg = PIL.Image.open(path)
            image = np.array(pilimg.convert("RGBA"))
            # 如果是暗色模式，将图像中的黑色部分替换为与系统文本颜色相匹配的颜色
            if dark:
                fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
                black_mask = (image[..., :3] == 0).all(axis=-1)
                image[black_mask, :3] = (fg.Red(), fg.Green(), fg.Blue())
            # 将 RGBA 图像数据转换为 wx.Bitmap 对象并返回
            return wx.Bitmap.FromBufferRGBA(
                image.shape[1], image.shape[0], image.tobytes())
    # 更新工具栏上的按钮状态，根据当前模式选择是否激活“Pan”和“Zoom”按钮
    def _update_buttons_checked(self):
        if "Pan" in self.wx_ids:
            self.ToggleTool(self.wx_ids["Pan"], self.mode.name == "PAN")
        if "Zoom" in self.wx_ids:
            self.ToggleTool(self.wx_ids["Zoom"], self.mode.name == "ZOOM")

    # 调用父类方法进行缩放操作，然后更新按钮状态
    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    # 调用父类方法进行平移操作，然后更新按钮状态
    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    # 弹出文件保存对话框，并保存图形到指定路径
    def save_figure(self, *args):
        # 获取文件保存类型和扩展名
        filetypes, exts, filter_index = self.canvas._get_imagesave_wildcards()
        default_file = self.canvas.get_default_filename()
        # 创建文件保存对话框
        dialog = wx.FileDialog(
            self.canvas.GetParent(), "Save to file",
            mpl.rcParams["savefig.directory"], default_file, filetypes,
            wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        dialog.SetFilterIndex(filter_index)
        # 如果用户点击了保存
        if dialog.ShowModal() == wx.ID_OK:
            # 获取保存路径
            path = pathlib.Path(dialog.GetPath())
            # 记录保存路径到日志中
            _log.debug('%s - Save file path: %s', type(self), path)
            fmt = exts[dialog.GetFilterIndex()]
            ext = path.suffix[1:]
            # 检查文件扩展名和格式是否匹配
            if ext in self.canvas.get_supported_filetypes() and fmt != ext:
                # 如果扩展名和格式不匹配，则警告
                _log.warning('extension %s did not match the selected '
                             'image type %s; going with %s',
                             ext, fmt, ext)
                fmt = ext
            # 保存上次保存的目录设置，除非为空字符串（表示使用当前工作目录）
            if mpl.rcParams["savefig.directory"]:
                mpl.rcParams["savefig.directory"] = str(path.parent)
            try:
                # 尝试保存图形到指定路径
                self.canvas.figure.savefig(path, format=fmt)
            except Exception as e:
                # 如果保存失败，显示错误对话框
                dialog = wx.MessageDialog(
                    parent=self.canvas.GetParent(), message=str(e),
                    caption='Matplotlib error')
                dialog.ShowModal()
                dialog.Destroy()

    # 在画布上绘制橡皮筋矩形框
    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        sf = 1 if wx.Platform == '__WXMSW__' else self.canvas.GetDPIScaleFactor()
        # 计算橡皮筋矩形框的坐标，并刷新画布
        self.canvas._rubberband_rect = (x0/sf, (height - y0)/sf,
                                        x1/sf, (height - y1)/sf)
        self.canvas.Refresh()

    # 移除画布上的橡皮筋矩形框
    def remove_rubberband(self):
        self.canvas._rubberband_rect = None
        self.canvas.Refresh()

    # 设置消息文本到标签控件上显示
    def set_message(self, s):
        if self._coordinates:
            self._label_text.SetLabel(s)
    # 定义设置历史按钮的方法，用于根据导航堆栈的位置设置按钮的可用性
    def set_history_buttons(self):
        # 检查是否可以后退，即导航堆栈的当前位置大于0
        can_backward = self._nav_stack._pos > 0
        # 检查是否可以前进，即导航堆栈的当前位置小于导航堆栈长度减1
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        # 如果存在名为 'Back' 的工具按钮 ID，则根据 can_backward 设置其可用性
        if 'Back' in self.wx_ids:
            self.EnableTool(self.wx_ids['Back'], can_backward)
        # 如果存在名为 'Forward' 的工具按钮 ID，则根据 can_forward 设置其可用性
        if 'Forward' in self.wx_ids:
            self.EnableTool(self.wx_ids['Forward'], can_forward)
# tools for matplotlib.backend_managers.ToolManager:

class ToolbarWx(ToolContainerBase, wx.ToolBar):
    _icon_extension = '.svg'

    def __init__(self, toolmanager, parent=None, style=wx.TB_BOTTOM):
        # 如果没有指定父窗口，则使用工具管理器的画布的父窗口作为父窗口
        if parent is None:
            parent = toolmanager.canvas.GetParent()
        # 调用 ToolContainerBase 的初始化方法
        ToolContainerBase.__init__(self, toolmanager)
        # 调用 wx.ToolBar 的初始化方法
        wx.ToolBar.__init__(self, parent, -1, style=style)
        # 添加一个可伸缩空间到工具栏
        self._space = self.AddStretchableSpace()
        # 添加一个静态文本控件到工具栏，用于显示标签文本
        self._label_text = wx.StaticText(self, style=wx.ALIGN_RIGHT)
        self.AddControl(self._label_text)
        # 初始化工具项字典和分组字典
        self._toolitems = {}
        self._groups = {}  # Mapping of groups to the separator after them.

    def _get_tool_pos(self, tool):
        """
        Find the position (index) of a wx.ToolBarToolBase in a ToolBar.

        ``ToolBar.GetToolPos`` is not useful because wx assigns the same Id to
        all Separators and StretchableSpaces.
        """
        # 返回指定工具在工具栏中的位置索引
        pos, = [pos for pos in range(self.ToolsCount)
                if self.GetToolByPos(pos) == tool]
        return pos

    def add_toolitem(self, name, group, position, image_file, description,
                     toggle):
        # 查找或创建跟在该分组后面的分隔符
        if group not in self._groups:
            self._groups[group] = self.InsertSeparator(
                self._get_tool_pos(self._space))
        sep = self._groups[group]
        # 列出所有分隔符
        seps = [t for t in map(self.GetToolByPos, range(self.ToolsCount))
                if t.IsSeparator() and not t.IsStretchableSpace()]
        # 确定工具插入的位置
        if position >= 0:
            # 通过查找当前分隔符前面的分隔符确定分组的起始位置，然后从起始位置向前移动
            start = (0 if sep == seps[0]
                     else self._get_tool_pos(seps[seps.index(sep) - 1]) + 1)
        else:
            # 从当前分隔符向后移动
            start = self._get_tool_pos(sep) + 1
        idx = start + position
        if image_file:
            # 如果有图像文件，则创建带图标的工具按钮
            bmp = NavigationToolbar2Wx._icon(image_file)
            kind = wx.ITEM_NORMAL if not toggle else wx.ITEM_CHECK
            tool = self.InsertTool(idx, -1, name, bmp, wx.NullBitmap, kind,
                                   description or "")
        else:
            # 如果没有图像文件，则创建带文本标签的控件
            size = (self.GetTextExtent(name)[0] + 10, -1)
            if toggle:
                control = wx.ToggleButton(self, -1, name, size=size)
            else:
                control = wx.Button(self, -1, name, size=size)
            tool = self.InsertControl(idx, control, label=name)
        self.Realize()

        def handler(event):
            self.trigger_tool(name)

        if image_file:
            # 绑定工具按钮的事件处理程序
            self.Bind(wx.EVT_TOOL, handler, tool)
        else:
            # 绑定控件的事件处理程序
            control.Bind(wx.EVT_LEFT_DOWN, handler)

        # 将工具项添加到工具项字典中
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((tool, handler))
    # 根据给定名称和状态，切换工具条项目的可见性或状态
    def toggle_toolitem(self, name, toggled):
        # 如果指定名称不在工具条项目字典中，则直接返回
        if name not in self._toolitems:
            return
        # 遍历指定名称下的每个工具条项目和对应的处理函数
        for tool, handler in self._toolitems[name]:
            # 如果工具条项目不是控件，调用 ToggleTool 方法切换状态
            if not tool.IsControl():
                self.ToggleTool(tool.Id, toggled)
            # 如果是控件，设置其值为指定的 toggled 状态
            else:
                tool.GetControl().SetValue(toggled)
        # 刷新界面以更新显示
        self.Refresh()
    
    # 根据给定名称，从工具条项目字典中移除对应的工具条项目
    def remove_toolitem(self, name):
        # 使用 pop 方法移除指定名称的工具条项目列表，并迭代每个项目
        for tool, handler in self._toolitems.pop(name, []):
            # 根据工具条项目的 Id 删除对应的工具条
            self.DeleteTool(tool.Id)
    
    # 设置界面上显示的消息文本
    def set_message(self, s):
        # 使用 _label_text 对象的 SetLabel 方法设置界面消息文本
        self._label_text.SetLabel(s)
@backend_tools._register_tool_class(_FigureCanvasWxBase)
class ConfigureSubplotsWx(backend_tools.ConfigureSubplotsBase):
    # 继承自 ConfigureSubplotsBase 的工具类，注册到 _FigureCanvasWxBase 上
    def trigger(self, *args):
        # 当触发时调用，使用 NavigationToolbar2Wx 配置子图
        NavigationToolbar2Wx.configure_subplots(self)


@backend_tools._register_tool_class(_FigureCanvasWxBase)
class SaveFigureWx(backend_tools.SaveFigureBase):
    # 继承自 SaveFigureBase 的工具类，注册到 _FigureCanvasWxBase 上
    def trigger(self, *args):
        # 当触发时调用，使用 _make_classic_style_pseudo_toolbar 创建经典样式的伪工具栏，并保存图像
        NavigationToolbar2Wx.save_figure(
            self._make_classic_style_pseudo_toolbar())


@backend_tools._register_tool_class(_FigureCanvasWxBase)
class RubberbandWx(backend_tools.RubberbandBase):
    # 继承自 RubberbandBase 的工具类，注册到 _FigureCanvasWxBase 上
    def draw_rubberband(self, x0, y0, x1, y1):
        # 绘制橡皮筋，使用 _make_classic_style_pseudo_toolbar 创建经典样式的伪工具栏
        NavigationToolbar2Wx.draw_rubberband(
            self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    def remove_rubberband(self):
        # 移除橡皮筋，使用 _make_classic_style_pseudo_toolbar 创建经典样式的伪工具栏
        NavigationToolbar2Wx.remove_rubberband(
            self._make_classic_style_pseudo_toolbar())


class _HelpDialog(wx.Dialog):
    _instance = None  # 一个打开的对话框单例的引用
    headers = [("Action", "Shortcuts", "Description")]
    widths = [100, 140, 300]

    def __init__(self, parent, help_entries):
        super().__init__(parent, title="Help",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        sizer = wx.BoxSizer(wx.VERTICAL)
        grid_sizer = wx.FlexGridSizer(0, 3, 8, 6)
        # 创建并添加条目
        bold = self.GetFont().MakeBold()
        for r, row in enumerate(self.headers + help_entries):
            for (col, width) in zip(row, self.widths):
                label = wx.StaticText(self, label=col)
                if r == 0:
                    label.SetFont(bold)
                label.Wrap(width)
                grid_sizer.Add(label, 0, 0, 0)
        # 完成布局，创建按钮
        sizer.Add(grid_sizer, 0, wx.ALL, 6)
        ok = wx.Button(self, wx.ID_OK)
        sizer.Add(ok, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 8)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()
        self.Bind(wx.EVT_CLOSE, self._on_close)
        ok.Bind(wx.EVT_BUTTON, self._on_close)

    def _on_close(self, event):
        _HelpDialog._instance = None  # 移除全局引用
        self.DestroyLater()
        event.Skip()

    @classmethod
    def show(cls, parent, help_entries):
        # 如果没有显示对话框，则创建一个；否则重新抛出它
        if cls._instance:
            cls._instance.Raise()
            return
        cls._instance = cls(parent, help_entries)
        cls._instance.Show()


@backend_tools._register_tool_class(_FigureCanvasWxBase)
class HelpWx(backend_tools.ToolHelpBase):
    # 继承自 ToolHelpBase 的工具类，注册到 _FigureCanvasWxBase 上
    def trigger(self, *args):
        # 当触发时调用，使用 _get_help_entries 获取帮助条目，并显示帮助对话框
        _HelpDialog.show(self.figure.canvas.GetTopLevelParent(),
                         self._get_help_entries())


@backend_tools._register_tool_class(_FigureCanvasWxBase)
class ToolCopyToClipboardWx(backend_tools.ToolCopyToClipboardBase):
    # 继承自 ToolCopyToClipboardBase 的工具类，注册到 _FigureCanvasWxBase 上
    pass  # 无具体的触发动作，继承基类功能
    # 定义触发器方法，接受任意位置参数和关键字参数
    def trigger(self, *args, **kwargs):
        # 如果画布尚未绘制，则进行绘制操作
        if not self.canvas._isDrawn:
            self.canvas.draw()
        
        # 检查画布的位图是否有效（IsOk），或者剪贴板未打开，则返回
        if not self.canvas.bitmap.IsOk() or not wx.TheClipboard.Open():
            return
        
        try:
            # 将画布的位图数据设置到剪贴板中，使用wx.BitmapDataObject进行包装
            wx.TheClipboard.SetData(wx.BitmapDataObject(self.canvas.bitmap))
        finally:
            # 无论是否成功设置数据，都关闭剪贴板
            wx.TheClipboard.Close()
FigureManagerWx._toolbar2_class = NavigationToolbar2Wx
FigureManagerWx._toolmanager_toolbar_class = ToolbarWx


# 设置 FigureManagerWx 类的 _toolbar2_class 属性为 NavigationToolbar2Wx 类
# 设置 FigureManagerWx 类的 _toolmanager_toolbar_class 属性为 ToolbarWx 类
FigureManagerWx._toolbar2_class = NavigationToolbar2Wx
FigureManagerWx._toolmanager_toolbar_class = ToolbarWx



@_Backend.export
class _BackendWx(_Backend):
    FigureCanvas = FigureCanvasWx
    FigureManager = FigureManagerWx
    mainloop = FigureManagerWx.start_main_loop


# 将 _BackendWx 类导出为 _Backend 的子类，并将其注册为 Backend 中的一部分
@_Backend.export
class _BackendWx(_Backend):
    # 设置 _BackendWx 类的 FigureCanvas 属性为 FigureCanvasWx 类
    FigureCanvas = FigureCanvasWx
    # 设置 _BackendWx 类的 FigureManager 属性为 FigureManagerWx 类
    FigureManager = FigureManagerWx
    # 设置 _BackendWx 类的 mainloop 属性为 FigureManagerWx 类的 start_main_loop 方法
    mainloop = FigureManagerWx.start_main_loop
```