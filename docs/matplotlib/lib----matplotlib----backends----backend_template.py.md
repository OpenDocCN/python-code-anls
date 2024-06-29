# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_template.py`

```
"""
A fully functional, do-nothing backend intended as a template for backend
writers.  It is fully functional in that you can select it as a backend e.g.
with ::

    import matplotlib
    matplotlib.use("template")

and your program will (should!) run without error, though no output is
produced.  This provides a starting point for backend writers; you can
selectively implement drawing methods (`~.RendererTemplate.draw_path`,
`~.RendererTemplate.draw_image`, etc.) and slowly see your figure come to life
instead having to have a full-blown implementation before getting any results.

Copy this file to a directory outside the Matplotlib source tree, somewhere
where Python can import it (by adding the directory to your ``sys.path`` or by
packaging it as a normal Python package); if the backend is importable as
``import my.backend`` you can then select it using ::

    import matplotlib
    matplotlib.use("module://my.backend")

If your backend implements support for saving figures (i.e. has a `print_xyz`
method), you can register it as the default handler for a given file type::

    from matplotlib.backend_bases import register_backend
    register_backend('xyz', 'my_backend', 'XYZ File Format')
    ...
    plt.savefig("figure.xyz")
"""

from matplotlib import _api                    # 导入 matplotlib 的 _api 模块
from matplotlib._pylab_helpers import Gcf       # 导入 matplotlib 的 Gcf 类
from matplotlib.backend_bases import (          # 导入 matplotlib 的以下基础类
     FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase)
from matplotlib.figure import Figure            # 导入 matplotlib 的 Figure 类


class RendererTemplate(RendererBase):
    """
    The renderer handles drawing/rendering operations.

    This is a minimal do-nothing class that can be used to get started when
    writing a new backend.  Refer to `.backend_bases.RendererBase` for
    documentation of the methods.
    """

    def __init__(self, dpi):
        super().__init__()                      # 调用父类 RendererBase 的初始化方法
        self.dpi = dpi                          # 设置 dpi 属性

    def draw_path(self, gc, path, transform, rgbFace=None):
        pass                                    # draw_path 方法暂时不执行任何操作，保留空白以供实现

    # draw_markers is optional, and we get more correct relative
    # timings by leaving it out.  backend implementers concerned with
    # performance will probably want to implement it
#     def draw_markers(self, gc, marker_path, marker_trans, path, trans,
#                      rgbFace=None):
#         pass

    # draw_path_collection is optional, and we get more correct
    # relative timings by leaving it out. backend implementers concerned with
    # performance will probably want to implement it
#     def draw_path_collection(self, gc, master_transform, paths,
#                              all_transforms, offsets, offset_trans,
#                              facecolors, edgecolors, linewidths, linestyles,
#                              antialiaseds):
#         pass

    # draw_quad_mesh is optional, and we get more correct
    # relative timings by leaving it out.  backend implementers concerned with
    # performance will probably want to implement it
#     def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
    def draw_image(self, gc, x, y, im):
        # 绘制图像，但在此模板中并未实现具体功能
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # 绘制文本，但在此模板中并未实现具体功能
        pass

    def flipy(self):
        # 返回 True，翻转 y 轴，继承自父类
        return True

    def get_canvas_width_height(self):
        # 返回画布的宽度和高度，继承自父类
        return 100, 100

    def get_text_width_height_descent(self, s, prop, ismath):
        # 返回文本的宽度、高度和下降距离，但在此模板中均返回 1
        return 1, 1, 1

    def new_gc(self):
        # 创建并返回一个新的 GraphicsContextTemplate 实例，继承自父类
        return GraphicsContextTemplate()

    def points_to_pixels(self, points):
        # 将点数转换为像素数，但在此模板中直接返回点数
        return points


class GraphicsContextTemplate(GraphicsContextBase):
    """
    图形上下文模板，提供颜色、线条样式等信息。参见 cairo 和 postscript 后端的例子，
    映射图形上下文属性（线条样式、颜色等）到特定后端。
    如果在渲染器级别执行映射更合适（如 postscript 后端），则无需覆盖任何 GC 方法。
    如果在这里执行映射更合适（如 cairo 后端），则需要覆盖几个 setter 方法。

    基础的 GraphicsContext 将颜色存储为单位间隔上的 RGB 元组，例如 (0.5, 0.0, 1.0)。
    您可能需要将其映射到适合您后端的颜色空间。
    """


########################################################################
#
# 以下函数和类用于 pyplot 并实现窗口/图形管理等。
#
########################################################################


class FigureManagerTemplate(FigureManagerBase):
    """
    pyplot 模式的辅助类，将所有内容封装成一个整洁的包。

    对于非交互式后端，基类已经足够。对于交互式后端，请参阅 `.FigureManagerBase` 类的文档，
    查看可以/应该覆盖的方法列表。
    """


class FigureCanvasTemplate(FigureCanvasBase):
    """
    图形渲染的画布。调用绘制和打印图形的方法，创建渲染器等。

    注意：GUI 模板可能需要将按钮按下、鼠标移动和按键事件连接到调用基类方法
    button_press_event、button_release_event 等的函数。
    """
    # 继承自 FigureCanvasBase，表示一个可交互的 Figure 管理器模板
    motion_notify_event, key_press_event, and key_release_event.  See the
    implementations of the interactive backends for examples.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level Figure instance
    """

    # 实例化的管理器类。为了进一步定制，也可以重写 ``FigureManager.create_with_canvas`` 方法；参考基于 wx 的后端实现作为示例。
    manager_class = FigureManagerTemplate

    def draw(self):
        """
        Draw the figure using the renderer.

        It is important that this method actually walk the artist tree
        even if not output is produced because this will trigger
        deferred work (like computing limits auto-limits and tick
        values) that users may want access to before saving to disk.
        """
        # 使用指定 DPI 创建渲染器对象
        renderer = RendererTemplate(self.figure.dpi)
        # 调用 Figure 对象的 draw 方法进行绘制
        self.figure.draw(renderer)

    # You should provide a print_xxx function for every file format
    # you can write.

    # If the file type is not in the base set of filetypes,
    # you should add it to the class-scope filetypes dictionary as follows:
    # 在类作用域的 filetypes 字典中添加新的文件类型，扩展基本的文件类型集合
    filetypes = {**FigureCanvasBase.filetypes, 'foo': 'My magic Foo format'}

    def print_foo(self, filename, **kwargs):
        """
        Write out format foo.

        This method is normally called via `.Figure.savefig` and
        `.FigureCanvasBase.print_figure`, which take care of setting the figure
        facecolor, edgecolor, and dpi to the desired output values, and will
        restore them to the original values.  Therefore, `print_foo` does not
        need to handle these settings.
        """
        # 绘制图形
        self.draw()

    def get_default_filetype(self):
        # 返回默认的文件类型为 'foo'
        return 'foo'
########################################################################
#
# 现在提供 backend.__init__ 预期的标准名称
#
########################################################################

# 将 FigureCanvas 设置为 FigureCanvasTemplate 的别名
FigureCanvas = FigureCanvasTemplate

# 将 FigureManager 设置为 FigureManagerTemplate 的别名
FigureManager = FigureManagerTemplate
```