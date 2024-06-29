# `D:\src\scipysrc\matplotlib\lib\matplotlib\backend_tools.py`

```
"""
Abstract base classes define the primitives for Tools.
These tools are used by `matplotlib.backend_managers.ToolManager`

:class:`ToolBase`
    Simple stateless tool

:class:`ToolToggleBase`
    Tool that has two states, only one Toggle tool can be
    active at any given time for the same
    `matplotlib.backend_managers.ToolManager`
"""

import enum  # 导入枚举类型模块
import functools  # 导入函数装饰器模块
import re  # 导入正则表达式模块
import time  # 导入时间模块
from types import SimpleNamespace  # 从 types 模块导入 SimpleNamespace 类
import uuid  # 导入 UUID 模块
from weakref import WeakKeyDictionary  # 从 weakref 模块导入 WeakKeyDictionary 类

import numpy as np  # 导入 NumPy 库，并用 np 别名引用

import matplotlib as mpl  # 导入 Matplotlib 库，并用 mpl 别名引用
from matplotlib._pylab_helpers import Gcf  # 从 _pylab_helpers 模块导入 Gcf 类
from matplotlib import _api, cbook  # 从 Matplotlib 模块导入 _api 和 cbook 模块


class Cursors(enum.IntEnum):  # 定义 Cursors 枚举类，必须是 int 的子类，用于 macOS 后端
    """Backend-independent cursor types."""
    POINTER = enum.auto()  # 光标类型：指针
    HAND = enum.auto()  # 光标类型：手
    SELECT_REGION = enum.auto()  # 光标类型：选择区域
    MOVE = enum.auto()  # 光标类型：移动
    WAIT = enum.auto()  # 光标类型：等待
    RESIZE_HORIZONTAL = enum.auto()  # 光标类型：水平调整大小
    RESIZE_VERTICAL = enum.auto()  # 光标类型：垂直调整大小
cursors = Cursors  # Backcompat.


# _tool_registry, _register_tool_class, and _find_tool_class implement a
# mechanism through which ToolManager.add_tool can determine whether a subclass
# of the requested tool class has been registered (either for the current
# canvas class or for a parent class), in which case that tool subclass will be
# instantiated instead.  This is the mechanism used e.g. to allow different
# GUI backends to implement different specializations for ConfigureSubplots.


_tool_registry = set()  # 初始化一个空的集合，用于存储注册的工具类


def _register_tool_class(canvas_cls, tool_cls=None):
    """Decorator registering *tool_cls* as a tool class for *canvas_cls*."""
    if tool_cls is None:
        return functools.partial(_register_tool_class, canvas_cls)
    _tool_registry.add((canvas_cls, tool_cls))  # 将 (canvas_cls, tool_cls) 元组添加到 _tool_registry 集合中
    return tool_cls  # 返回注册的工具类


def _find_tool_class(canvas_cls, tool_cls):
    """Find a subclass of *tool_cls* registered for *canvas_cls*."""
    for canvas_parent in canvas_cls.__mro__:  # 遍历 canvas_cls 的方法解析顺序（Method Resolution Order, MRO）
        for tool_child in _api.recursive_subclasses(tool_cls):  # 递归查找 tool_cls 的子类
            if (canvas_parent, tool_child) in _tool_registry:  # 如果找到 (canvas_parent, tool_child) 在 _tool_registry 中
                return tool_child  # 返回找到的工具子类
    return tool_cls  # 如果没有找到注册的子类，则返回原始的工具类


# Views positions tool
_views_positions = 'viewpos'  # 设置视图位置工具的标识为 'viewpos'


class ToolBase:
    """
    Base tool class.

    A base tool, only implements `trigger` method or no method at all.
    The tool is instantiated by `matplotlib.backend_managers.ToolManager`.
    """

    default_keymap = None
    """
    Keymap to associate with this tool.

    ``list[str]``: List of keys that will trigger this tool when a keypress
    event is emitted on ``self.figure.canvas``.  Note that this attribute is
    looked up on the instance, and can therefore be a property (this is used
    e.g. by the built-in tools to load the rcParams at instantiation time).
    """

    description = None
    """
    Description of the Tool.

    `str`: Tooltip used if the Tool is included in a Toolbar.
    """

    image = None
    """
    Icon filename.

    ``str | None``: Filename of the Toolbar icon; either absolute, or relative to the
    directory containing the Python source file where the ``Tool.image`` class attribute
    is defined (in the latter case, this cannot be defined as an instance attribute).
    In either case, the extension is optional; leaving it off lets individual backends
    select the icon format they prefer.  If None, the *name* is used as a label in the
    toolbar button.
    """

    # Tool 类的构造函数，初始化工具名称和工具管理器
    def __init__(self, toolmanager, name):
        self._name = name  # 设置工具名称
        self._toolmanager = toolmanager  # 设置工具管理器
        self._figure = None  # 初始化图形对象为 None

    # name 属性，返回工具的唯一标识符 (str，在工具管理器中必须唯一)
    name = property(
        lambda self: self._name,
        doc="The tool id (str, must be unique among tools of a tool manager).")
    
    # toolmanager 属性，返回控制该工具的 ToolManager 对象
    toolmanager = property(
        lambda self: self._toolmanager,
        doc="The `.ToolManager` that controls this tool.")
    
    # canvas 属性，返回受此工具影响的图形对象的画布，如果未设置图形对象则返回 None
    canvas = property(
        lambda self: self._figure.canvas if self._figure is not None else None,
        doc="The canvas of the figure affected by this tool, or None.")
    
    # 设置图形对象的方法
    def set_figure(self, figure):
        self._figure = figure
    
    # figure 属性，返回受此工具影响的图形对象，如果未设置图形对象则返回 None
    figure = property(
        lambda self: self._figure,
        # setter 方法必须显式调用 self.set_figure，以便子类可以有意义地重写它
        lambda self, figure: self.set_figure(figure),
        doc="The Figure affected by this tool, or None.")

    # 创建经典风格伪工具栏的方法
    def _make_classic_style_pseudo_toolbar(self):
        """
        Return a placeholder object with a single `canvas` attribute.

        This is useful to reuse the implementations of tools already provided
        by the classic Toolbars.
        """
        return SimpleNamespace(canvas=self.canvas)

    # 触发工具使用时调用的方法，由 ToolManager.trigger_tool 调用
    def trigger(self, sender, event, data=None):
        """
        Called when this tool gets used.

        This method is called by `.ToolManager.trigger_tool`.

        Parameters
        ----------
        event : `.Event`
            The canvas event that caused this tool to be called.
        sender : object
            Object that requested the tool to be triggered.
        data : object
            Extra data.
        """
        pass
class ToolToggleBase(ToolBase):
    """
    Toggleable tool.

    Every time it is triggered, it switches between enable and disable.

    Parameters
    ----------
    ``*args``
        Variable length argument to be used by the Tool.
    ``**kwargs``
        `toggled` if present and True, sets the initial state of the Tool
        Arbitrary keyword arguments to be consumed by the Tool
    """

    radio_group = None
    """
    Attribute to group 'radio' like tools (mutually exclusive).

    `str` that identifies the group or **None** if not belonging to a group.
    """

    cursor = None
    """Cursor to use when the tool is active."""

    default_toggled = False
    """Default of toggled state."""

    def __init__(self, *args, **kwargs):
        # Initialize the tool, setting its initial toggled state
        self._toggled = kwargs.pop('toggled', self.default_toggled)
        super().__init__(*args, **kwargs)

    def trigger(self, sender, event, data=None):
        """Calls `enable` or `disable` based on `toggled` value."""
        if self._toggled:
            # If currently toggled on, disable the tool
            self.disable(event)
        else:
            # If currently toggled off, enable the tool
            self.enable(event)
        # Toggle the state for the next trigger
        self._toggled = not self._toggled

    def enable(self, event=None):
        """
        Enable the toggle tool.

        `trigger` calls this method when `toggled` is False.
        """
        pass

    def disable(self, event=None):
        """
        Disable the toggle tool.

        `trigger` calls this method when `toggled` is True.

        This can happen in different circumstances.

        * Click on the toolbar tool button.
        * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`.
        * Another `ToolToggleBase` derived tool is triggered
          (from the same `.ToolManager`).
        """
        pass

    @property
    def toggled(self):
        """State of the toggled tool."""
        return self._toggled

    def set_figure(self, figure):
        toggled = self.toggled
        if toggled:
            if self.figure:
                # If already toggled on and there is a current figure, toggle off
                self.trigger(self, None)
            else:
                # If no current figure, maintain the toggle state until next trigger
                self._toggled = False
        super().set_figure(figure)
        if toggled:
            if figure:
                # If a figure is set after being toggled on, toggle off
                self.trigger(self, None)
            else:
                # If no figure is set after being toggled on, maintain toggle state
                self._toggled = True


class ToolSetCursor(ToolBase):
    """
    Change to the current cursor while inaxes.

    This tool, keeps track of all `ToolToggleBase` derived tools, and updates
    the cursor when a tool gets triggered.
    """
    # 初始化方法，调用父类的初始化方法，接收任意位置和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，传入相同的位置和关键字参数
        super().__init__(*args, **kwargs)
        # 初始化拖动标识为 None
        self._id_drag = None
        # 当前工具初始化为 None
        self._current_tool = None
        # 默认光标设为指针形状
        self._default_cursor = cursors.POINTER
        # 上一个光标状态设为默认光标
        self._last_cursor = self._default_cursor
        # 将处理工具添加事件连接到工具管理器的工具添加事件
        self.toolmanager.toolmanager_connect('tool_added_event',
                                             self._add_tool_cbk)
        # 处理当前已存在的工具
        for tool in self.toolmanager.tools.values():
            # 添加每个工具到界面
            self._add_tool(tool)

    # 设置图形的方法，接收一个图形参数
    def set_figure(self, figure):
        # 如果存在正在拖动的标识
        if self._id_drag:
            # 断开当前的拖动事件连接
            self.canvas.mpl_disconnect(self._id_drag)
        # 调用父类的设置图形方法，传入图形参数
        super().set_figure(figure)
        # 如果图形存在
        if figure:
            # 连接图形的鼠标移动事件到设定光标的回调方法
            self._id_drag = self.canvas.mpl_connect(
                'motion_notify_event', self._set_cursor_cbk)

    # 工具触发回调方法，处理工具触发事件对象
    def _tool_trigger_cbk(self, event):
        # 如果工具被触发
        if event.tool.toggled:
            # 设置当前工具为触发事件的工具
            self._current_tool = event.tool
        else:
            # 否则将当前工具设为 None
            self._current_tool = None
        # 调用设置光标的回调方法，传入事件对象
        self._set_cursor_cbk(event.canvasevent)

    # 添加工具的方法，当工具对象传入时设置工具的光标
    def _add_tool(self, tool):
        """Set the cursor when the tool is triggered."""
        # 如果工具对象具有光标属性
        if getattr(tool, 'cursor', None) is not None:
            # 连接工具触发事件到工具触发回调方法
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name,
                                                 self._tool_trigger_cbk)

    # 工具添加回调方法，处理每个新添加的工具事件
    def _add_tool_cbk(self, event):
        """Process every newly added tool."""
        # 如果事件中的工具是自身，则返回
        if event.tool is self:
            return
        # 否则将事件中的工具添加到界面
        self._add_tool(event.tool)

    # 设置光标的回调方法，处理鼠标事件
    def _set_cursor_cbk(self, event):
        # 如果事件不存在或者画布不存在，则返回
        if not event or not self.canvas:
            return
        # 如果当前工具存在，并且事件在坐标轴内且可以导航
        if (self._current_tool and getattr(event, "inaxes", None)
                and event.inaxes.get_navigate()):
            # 如果上一个光标状态不等于当前工具的光标状态
            if self._last_cursor != self._current_tool.cursor:
                # 设置画布的光标为当前工具的光标
                self.canvas.set_cursor(self._current_tool.cursor)
                # 更新上一个光标状态为当前工具的光标状态
                self._last_cursor = self._current_tool.cursor
        # 否则如果上一个光标状态不等于默认光标状态
        elif self._last_cursor != self._default_cursor:
            # 设置画布的光标为默认光标
            self.canvas.set_cursor(self._default_cursor)
            # 更新上一个光标状态为默认光标状态
            self._last_cursor = self._default_cursor
class ToolCursorPosition(ToolBase):
    """
    Send message with the current pointer position.

    This tool runs in the background reporting the position of the cursor.
    """
    def __init__(self, *args, **kwargs):
        # 初始化，设置拖动标识符为 None
        self._id_drag = None
        super().__init__(*args, **kwargs)

    def set_figure(self, figure):
        # 如果存在拖动标识符，断开与 canvas 的连接
        if self._id_drag:
            self.canvas.mpl_disconnect(self._id_drag)
        super().set_figure(figure)
        # 如果有新的 figure，连接 'motion_notify_event' 事件到 send_message 方法
        if figure:
            self._id_drag = self.canvas.mpl_connect(
                'motion_notify_event', self.send_message)

    def send_message(self, event):
        """
        Call `matplotlib.backend_managers.ToolManager.message_event`.
        """
        # 如果消息锁定被锁住，直接返回
        if self.toolmanager.messagelock.locked():
            return

        from matplotlib.backend_bases import NavigationToolbar2
        # 转换鼠标事件为消息对象
        message = NavigationToolbar2._mouse_event_to_message(event)
        # 发送消息事件到 ToolManager
        self.toolmanager.message_event(message, self)


class RubberbandBase(ToolBase):
    """
    Draw and remove a rubberband.
    """
    def trigger(self, sender, event, data=None):
        """
        Call `draw_rubberband` or `remove_rubberband` based on data.
        """
        # 如果 sender 不可用，返回
        if not self.figure.canvas.widgetlock.available(sender):
            return
        # 根据传入的 data 调用 draw_rubberband 或 remove_rubberband
        if data is not None:
            self.draw_rubberband(*data)
        else:
            self.remove_rubberband()

    def draw_rubberband(self, *data):
        """
        Draw rubberband.

        This method must get implemented per backend.
        """
        # 抛出未实现错误，具体实现由后端决定
        raise NotImplementedError

    def remove_rubberband(self):
        """
        Remove rubberband.

        This method should get implemented per backend.
        """
        # 空方法，具体实现由后端决定
        pass


class ToolQuit(ToolBase):
    """
    Tool to call the figure manager destroy method.
    """

    description = 'Quit the figure'
    default_keymap = property(lambda self: mpl.rcParams['keymap.quit'])

    def trigger(self, sender, event, data=None):
        # 销毁当前 figure
        Gcf.destroy_fig(self.figure)


class ToolQuitAll(ToolBase):
    """
    Tool to call the figure manager destroy method.
    """

    description = 'Quit all figures'
    default_keymap = property(lambda self: mpl.rcParams['keymap.quit_all'])

    def trigger(self, sender, event, data=None):
        # 销毁所有 figures
        Gcf.destroy_all()


class ToolGrid(ToolBase):
    """
    Tool to toggle the major grids of the figure.
    """

    description = 'Toggle major grids'
    default_keymap = property(lambda self: mpl.rcParams['keymap.grid'])

    def trigger(self, sender, event, data=None):
        # 创建一个唯一的 sentinel 标识符
        sentinel = str(uuid.uuid4())
        # 通过临时设置 'keymap.grid' 到唯一的 sentinel 键，触发网格切换事件
        with cbook._setattr_cm(event, key=sentinel), \
             mpl.rc_context({'keymap.grid': sentinel}):
            mpl.backend_bases.key_press_handler(event, self.figure)


class ToolMinorGrid(ToolBase):
    """
    Tool to toggle the major and minor grids of the figure.
    """

    description = 'Toggle major and minor grids'
    # 定义一个属性 default_keymap，使用 lambda 表达式返回 mpl.rcParams 中 'keymap.grid_minor' 的值
    default_keymap = property(lambda self: mpl.rcParams['keymap.grid_minor'])

    # 定义触发器方法 trigger，接受发送者、事件和可选数据作为参数
    def trigger(self, sender, event, data=None):
        # 生成一个唯一的标识符作为 sentinel
        sentinel = str(uuid.uuid4())
        
        # 通过临时设置 :rc:`keymap.grid_minor` 为唯一的键，并发送适当的事件来触发网格切换
        with cbook._setattr_cm(event, key=sentinel), \
             mpl.rc_context({'keymap.grid_minor': sentinel}):
            # 调用 key_press_handler 方法处理事件，传入事件和关联的画布
            mpl.backend_bases.key_press_handler(event, self.figure.canvas)
class ToolFullScreen(ToolBase):
    """Tool to toggle full screen."""

    # 描述工具的作用：切换全屏模式
    description = 'Toggle fullscreen mode'

    # 获取默认的快捷键映射
    default_keymap = property(lambda self: mpl.rcParams['keymap.fullscreen'])

    def trigger(self, sender, event, data=None):
        # 触发器方法：切换图形的全屏显示
        self.figure.canvas.manager.full_screen_toggle()


class AxisScaleBase(ToolToggleBase):
    """Base Tool to toggle between linear and logarithmic."""

    def trigger(self, sender, event, data=None):
        # 如果事件不在坐标轴上，则直接返回
        if event.inaxes is None:
            return
        super().trigger(sender, event, data)

    def enable(self, event=None):
        # 启用方法：设置坐标轴的比例为对数
        self.set_scale(event.inaxes, 'log')
        self.figure.canvas.draw_idle()

    def disable(self, event=None):
        # 禁用方法：设置坐标轴的比例为线性
        self.set_scale(event.inaxes, 'linear')
        self.figure.canvas.draw_idle()


class ToolYScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the Y axis."""

    # 描述工具的作用：在Y轴上切换线性和对数比例尺
    description = 'Toggle scale Y axis'

    # 获取默认的快捷键映射
    default_keymap = property(lambda self: mpl.rcParams['keymap.yscale'])

    def set_scale(self, ax, scale):
        # 设置比例尺方法：设置Y轴的比例尺
        ax.set_yscale(scale)


class ToolXScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the X axis."""

    # 描述工具的作用：在X轴上切换线性和对数比例尺
    description = 'Toggle scale X axis'

    # 获取默认的快捷键映射
    default_keymap = property(lambda self: mpl.rcParams['keymap.xscale'])

    def set_scale(self, ax, scale):
        # 设置比例尺方法：设置X轴的比例尺
        ax.set_xscale(scale)


class ToolViewsPositions(ToolBase):
    """
    Auxiliary Tool to handle changes in views and positions.

    Runs in the background and should get used by all the tools that
    need to access the figure's history of views and positions, e.g.

    * `ToolZoom`
    * `ToolPan`
    * `ToolHome`
    * `ToolBack`
    * `ToolForward`
    """

    def __init__(self, *args, **kwargs):
        # 初始化方法：设置视图和位置的弱引用字典
        self.views = WeakKeyDictionary()
        self.positions = WeakKeyDictionary()
        self.home_views = WeakKeyDictionary()
        super().__init__(*args, **kwargs)

    def add_figure(self, figure):
        """Add the current figure to the stack of views and positions."""

        # 添加图形方法：将当前图形添加到视图和位置的堆栈中
        if figure not in self.views:
            self.views[figure] = cbook._Stack()
            self.positions[figure] = cbook._Stack()
            self.home_views[figure] = WeakKeyDictionary()
            # 定义主视图
            self.push_current(figure)
            # 确保在添加新坐标轴时为其添加主视图
            figure.add_axobserver(lambda fig: self.update_home_views(fig))

    def clear(self, figure):
        """Reset the Axes stack."""

        # 清除方法：重置坐标轴堆栈
        if figure in self.views:
            self.views[figure].clear()
            self.positions[figure].clear()
            self.home_views[figure].clear()
            self.update_home_views()
    def update_view(self):
        """
        Update the view limits and position for each Axes from the current
        stack position. If any Axes are present in the figure that aren't in
        the current stack position, use the home view limits for those Axes and
        don't update *any* positions.
        """

        # 获取当前视图和位置的弱引用对象
        views = self.views[self.figure]()
        if views is None:
            return
        pos = self.positions[self.figure]()
        if pos is None:
            return
        # 获取默认的起始视图限制
        home_views = self.home_views[self.figure]
        # 获取图形中的所有 Axes 对象
        all_axes = self.figure.get_axes()
        for a in all_axes:
            if a in views:
                # 如果当前 Axes 存在于视图中，则使用当前视图
                cur_view = views[a]
            else:
                # 如果当前 Axes 不存在于视图中，则使用默认的起始视图
                cur_view = home_views[a]
            # 更新 Axes 对象的视图
            a._set_view(cur_view)

        # 检查是否所有的 Axes 都在位置数据中
        if set(all_axes).issubset(pos):
            for a in all_axes:
                # 恢复原始和活动位置
                a._set_position(pos[a][0], 'original')
                a._set_position(pos[a][1], 'active')

        # 绘制更新后的图形
        self.figure.canvas.draw_idle()

    def push_current(self, figure=None):
        """
        Push the current view limits and position onto their respective stacks.
        """
        if not figure:
            figure = self.figure
        # 创建弱引用字典来保存当前视图和位置
        views = WeakKeyDictionary()
        pos = WeakKeyDictionary()
        for a in figure.get_axes():
            views[a] = a._get_view()
            pos[a] = self._axes_pos(a)
        # 将当前视图和位置推入堆栈
        self.views[figure].push(views)
        self.positions[figure].push(pos)

    def _axes_pos(self, ax):
        """
        Return the original and modified positions for the specified Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The `.Axes` to get the positions for.

        Returns
        -------
        original_position, modified_position
            A tuple of the original and modified positions.
        """

        # 获取指定 Axes 的原始和修改后的位置
        return (ax.get_position(True).frozen(),
                ax.get_position().frozen())

    def update_home_views(self, figure=None):
        """
        Make sure that ``self.home_views`` has an entry for all Axes present
        in the figure.
        """

        if not figure:
            figure = self.figure
        # 确保 self.home_views 中包含图形中所有 Axes 的条目
        for a in figure.get_axes():
            if a not in self.home_views[figure]:
                self.home_views[figure][a] = a._get_view()

    def home(self):
        """Recall the first view and position from the stack."""
        # 从堆栈中召回第一个视图和位置
        self.views[self.figure].home()
        self.positions[self.figure].home()

    def back(self):
        """Back one step in the stack of views and positions."""
        # 在视图和位置堆栈中后退一步
        self.views[self.figure].back()
        self.positions[self.figure].back()

    def forward(self):
        """Forward one step in the stack of views and positions."""
        # 在视图和位置堆栈中前进一步
        self.views[self.figure].forward()
        self.positions[self.figure].forward()
class ViewsPositionsBase(ToolBase):
    """Base class for `ToolHome`, `ToolBack` and `ToolForward`."""

    _on_trigger = None  # 触发器名称，默认为 None

    def trigger(self, sender, event, data=None):
        # 获取工具管理器中的 _views_positions 工具，并向其中添加图形
        self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
        # 根据当前对象的 _on_trigger 属性调用 _views_positions 工具的对应方法
        getattr(self.toolmanager.get_tool(_views_positions),
                self._on_trigger)()
        # 更新 _views_positions 工具中的视图
        self.toolmanager.get_tool(_views_positions).update_view()


class ToolHome(ViewsPositionsBase):
    """Restore the original view limits."""

    description = 'Reset original view'  # 描述信息：恢复原始视图
    image = 'mpl-data/images/home'  # 图像路径：home 图标
    default_keymap = property(lambda self: mpl.rcParams['keymap.home'])  # 默认键盘映射：home 操作


class ToolBack(ViewsPositionsBase):
    """Move back up the view limits stack."""

    description = 'Back to previous view'  # 描述信息：返回上一个视图
    image = 'mpl-data/images/back'  # 图像路径：back 图标
    default_keymap = property(lambda self: mpl.rcParams['keymap.back'])  # 默认键盘映射：back 操作
    _on_trigger = 'back'  # 触发器名称为 'back'


class ToolForward(ViewsPositionsBase):
    """Move forward in the view lim stack."""

    description = 'Forward to next view'  # 描述信息：前进到下一个视图
    image = 'mpl-data/images/forward'  # 图像路径：forward 图标
    default_keymap = property(lambda self: mpl.rcParams['keymap.forward'])  # 默认键盘映射：forward 操作
    _on_trigger = 'forward'  # 触发器名称为 'forward'


class ConfigureSubplotsBase(ToolBase):
    """Base tool for the configuration of subplots."""

    description = 'Configure subplots'  # 描述信息：配置子图
    image = 'mpl-data/images/subplots'  # 图像路径：subplots 图标


class SaveFigureBase(ToolBase):
    """Base tool for figure saving."""

    description = 'Save the figure'  # 描述信息：保存图形
    image = 'mpl-data/images/filesave'  # 图像路径：filesave 图标
    default_keymap = property(lambda self: mpl.rcParams['keymap.save'])  # 默认键盘映射：save 操作


class ZoomPanBase(ToolToggleBase):
    """Base class for `ToolZoom` and `ToolPan`."""

    def __init__(self, *args):
        super().__init__(*args)
        self._button_pressed = None  # 按钮状态：未按下
        self._xypress = None  # 保存鼠标按下时的位置信息
        self._idPress = None  # 按下事件的连接 ID
        self._idRelease = None  # 释放事件的连接 ID
        self._idScroll = None  # 滚动事件的连接 ID
        self.base_scale = 2.  # 基础缩放比例为 2
        self.scrollthresh = .5  # 滚动阈值为 0.5 秒
        self.lastscroll = time.time()-self.scrollthresh  # 上次滚动时间为当前时间减去阈值的时间

    def enable(self, event=None):
        """Connect press/release events and lock the canvas."""
        self.figure.canvas.widgetlock(self)  # 锁定画布以防止其他操作
        self._idPress = self.figure.canvas.mpl_connect(
            'button_press_event', self._press)  # 连接鼠标按下事件并调用 self._press 方法
        self._idRelease = self.figure.canvas.mpl_connect(
            'button_release_event', self._release)  # 连接鼠标释放事件并调用 self._release 方法
        self._idScroll = self.figure.canvas.mpl_connect(
            'scroll_event', self.scroll_zoom)  # 连接滚动事件并调用 self.scroll_zoom 方法

    def disable(self, event=None):
        """Release the canvas and disconnect press/release events."""
        self._cancel_action()  # 取消当前操作
        self.figure.canvas.widgetlock.release(self)  # 释放画布锁定
        self.figure.canvas.mpl_disconnect(self._idPress)  # 断开鼠标按下事件的连接
        self.figure.canvas.mpl_disconnect(self._idRelease)  # 断开鼠标释放事件的连接
        self.figure.canvas.mpl_disconnect(self._idScroll)  # 断开滚动事件的连接
    # 触发方法，处理特定事件的响应
    def trigger(self, sender, event, data=None):
        # 使用工具管理器获取视图位置工具，并添加图形到当前图形
        self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
        # 调用父类的触发方法，继续处理事件
        super().trigger(sender, event, data)
        # 根据当前按钮状态设置新的导航模式（大写），或者设为 None
        new_navigate_mode = self.name.upper() if self.toggled else None
        # 遍历图形对象的所有坐标轴，设置新的导航模式
        for ax in self.figure.axes:
            ax.set_navigate_mode(new_navigate_mode)

    # 滚动缩放方法，处理鼠标滚轮事件
    def scroll_zoom(self, event):
        # 如果事件不在坐标轴内，则返回
        if event.inaxes is None:
            return

        # 根据鼠标滚轮方向设置缩放因子
        if event.button == 'up':
            # 处理放大
            scl = self.base_scale
        elif event.button == 'down':
            # 处理缩小
            scl = 1/self.base_scale
        else:
            # 处理不应该发生的情况
            scl = 1

        # 获取事件所在的坐标轴对象
        ax = event.inaxes
        # 使用新的边界框坐标设置视图
        ax._set_view_from_bbox([event.x, event.y, scl])

        # 如果上次滚动发生在时间阈值内，则删除之前的视图
        if (time.time()-self.lastscroll) < self.scrollthresh:
            self.toolmanager.get_tool(_views_positions).back()

        # 强制重新绘制图形画布
        self.figure.canvas.draw_idle()

        # 更新上次滚动的时间戳
        self.lastscroll = time.time()
        # 推送当前视图状态到视图位置工具
        self.toolmanager.get_tool(_views_positions).push_current()
class ToolZoom(ZoomPanBase):
    """A Tool for zooming using a rectangle selector."""

    description = 'Zoom to rectangle'  # 工具描述，用于在界面上显示
    image = 'mpl-data/images/zoom_to_rect'  # 图标路径，用于界面展示
    default_keymap = property(lambda self: mpl.rcParams['keymap.zoom'])  # 默认按键映射，使用matplotlib默认设置
    cursor = cursors.SELECT_REGION  # 光标样式设定为选择区域
    radio_group = 'default'  # 单选按钮组，默认为'default'

    def __init__(self, *args):
        super().__init__(*args)
        self._ids_zoom = []  # 存储连接 ID 的列表，用于后续取消动作

    def _cancel_action(self):
        """取消动作的方法"""
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)  # 断开与canvas的连接
        self.toolmanager.trigger_tool('rubberband', self)  # 触发橡皮筋工具，重置状态
        self.figure.canvas.draw_idle()  # 绘制空闲画布，更新界面
        self._xypress = None  # 重置存储的坐标信息
        self._button_pressed = None  # 重置按下的按钮状态
        self._ids_zoom = []  # 清空连接 ID 列表
        return

    def _press(self, event):
        """鼠标按下的回调函数，用于响应在缩放到矩形选择模式下的按键操作"""

        # 如果已经处于缩放状态，按下其他按钮可取消当前动作
        if self._ids_zoom:
            self._cancel_action()

        # 根据按下的按钮设置按钮状态
        if event.button == 1:
            self._button_pressed = 1  # 左键按下
        elif event.button == 3:
            self._button_pressed = 3  # 右键按下
        else:
            self._cancel_action()  # 其他按键取消操作
            return

        x, y = event.x, event.y

        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event) and
                    a.get_navigate() and a.can_zoom()):
                self._xypress.append((x, y, a, i, a._get_view()))

        # 连接鼠标移动、按键按下和按键释放事件的监听器
        id1 = self.figure.canvas.mpl_connect(
            'motion_notify_event', self._mouse_move)
        id2 = self.figure.canvas.mpl_connect(
            'key_press_event', self._switch_on_zoom_mode)
        id3 = self.figure.canvas.mpl_connect(
            'key_release_event', self._switch_off_zoom_mode)

        self._ids_zoom = id1, id2, id3  # 存储连接 ID，以便后续取消操作
        self._zoom_mode = event.key  # 设置缩放模式为按下的按键

    def _switch_on_zoom_mode(self, event):
        """开启缩放模式的方法"""
        self._zoom_mode = event.key  # 设置缩放模式为按下的按键
        self._mouse_move(event)  # 执行鼠标移动操作

    def _switch_off_zoom_mode(self, event):
        """关闭缩放模式的方法"""
        self._zoom_mode = None  # 清空缩放模式
        self._mouse_move(event)  # 执行鼠标移动操作

    def _mouse_move(self, event):
        """鼠标移动的回调函数，用于响应缩放到矩形选择模式下的鼠标移动操作"""

        if self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, ind, view = self._xypress[0]
            (x1, y1), (x2, y2) = np.clip(
                [[lastx, lasty], [x, y]], a.bbox.min, a.bbox.max)
            if self._zoom_mode == "x":
                y1, y2 = a.bbox.intervaly  # 在 x 方向上进行缩放
            elif self._zoom_mode == "y":
                x1, x2 = a.bbox.intervalx  # 在 y 方向上进行缩放
            self.toolmanager.trigger_tool(
                'rubberband', self, data=(x1, y1, x2, y2))  # 触发橡皮筋工具，传递数据
    def _release(self, event):
        """Callback for mouse button releases in zoom-to-rectangle mode."""

        # 断开所有与缩放操作相关的事件监听器
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []  # 清空已存储的事件监听器标识列表

        # 如果没有记录下按下的点坐标（即没有有效的操作），则取消操作并返回
        if not self._xypress:
            self._cancel_action()
            return

        done_ax = []  # 存储已经处理过的 Axes 对象，避免重复操作

        # 遍历所有记录的按下鼠标的坐标点信息
        for cur_xypress in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, _ind, view = cur_xypress
            # 忽略单点点击，认为小于5个像素为点击
            if abs(x - lastx) < 5 or abs(y - lasty) < 5:
                self._cancel_action()
                return

            # 检测当前 Axes 是否与之前处理的 Axes 共享 x 轴或 y 轴，避免重复缩放
            twinx = any(a.get_shared_x_axes().joined(a, a1) for a1 in done_ax)
            twiny = any(a.get_shared_y_axes().joined(a, a1) for a1 in done_ax)
            done_ax.append(a)

            # 根据按下的鼠标按钮确定缩放方向
            if self._button_pressed == 1:
                direction = 'in'
            elif self._button_pressed == 3:
                direction = 'out'
            else:
                continue

            # 根据选定的区域设置视图的变化
            a._set_view_from_bbox((lastx, lasty, x, y), direction,
                                  self._zoom_mode, twinx, twiny)

        self._zoom_mode = None  # 清除缩放模式
        self.toolmanager.get_tool(_views_positions).push_current()  # 将当前操作状态推入历史记录
        self._cancel_action()  # 取消当前操作
class ToolPan(ZoomPanBase):
    """Pan Axes with left mouse, zoom with right."""

    # 定义一个属性，返回用于平移的默认按键映射
    default_keymap = property(lambda self: mpl.rcParams['keymap.pan'])
    
    # 工具描述信息
    description = 'Pan axes with left mouse, zoom with right'
    
    # 工具图标路径
    image = 'mpl-data/images/move'
    
    # 鼠标光标类型为移动光标
    cursor = cursors.MOVE
    
    # 单选按钮组名为默认
    radio_group = 'default'

    def __init__(self, *args):
        super().__init__(*args)
        
        # 初始化时设定用于拖动操作的标识为 None
        self._id_drag = None

    def _cancel_action(self):
        # 取消当前操作，清空按下的按钮标识和坐标点信息
        self._button_pressed = None
        self._xypress = []
        
        # 断开与画布的拖动事件连接
        self.figure.canvas.mpl_disconnect(self._id_drag)
        
        # 释放消息锁
        self.toolmanager.messagelock.release(self)
        
        # 重新绘制画布
        self.figure.canvas.draw_idle()

    def _press(self, event):
        # 根据按下的鼠标按钮执行不同操作
        if event.button == 1:
            self._button_pressed = 1  # 左键按下
        elif event.button == 3:
            self._button_pressed = 3  # 右键按下
        else:
            self._cancel_action()
            return
        
        x, y = event.x, event.y
        
        # 清空坐标点列表
        self._xypress = []
        
        # 遍历图形对象中的每个坐标系
        for i, a in enumerate(self.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event) and
                    a.get_navigate() and a.can_pan()):
                # 在符合条件的坐标系中开始平移操作
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
                
                # 锁定消息传递
                self.toolmanager.messagelock(self)
                
                # 连接鼠标移动事件处理函数
                self._id_drag = self.figure.canvas.mpl_connect(
                    'motion_notify_event', self._mouse_move)

    def _release(self, event):
        # 如果没有按下按钮，则取消操作
        if self._button_pressed is None:
            self._cancel_action()
            return
        
        # 断开与画布的鼠标移动事件连接
        self.figure.canvas.mpl_disconnect(self._id_drag)
        
        # 释放消息锁
        self.toolmanager.messagelock.release(self)

        # 结束所有平移操作
        for a, _ind in self._xypress:
            a.end_pan()
        
        # 如果没有符合条件的坐标系，则取消操作
        if not self._xypress:
            self._cancel_action()
            return
        
        # 获取视图位置工具，并推送当前状态
        self.toolmanager.get_tool(_views_positions).push_current()
        
        # 取消当前操作
        self._cancel_action()

    def _mouse_move(self, event):
        # 遍历每个坐标系，执行拖动操作
        for a, _ind in self._xypress:
            # 使用按下的按钮执行拖动操作，而不是当前按钮，因为在移动过程中可能按下多个按钮
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
        
        # 画布重新绘制
        self.toolmanager.canvas.draw_idle()


class ToolHelpBase(ToolBase):
    # 工具描述信息
    description = 'Print tool list, shortcuts and description'
    
    # 默认按键映射为帮助菜单的配置
    default_keymap = property(lambda self: mpl.rcParams['keymap.help'])
    
    # 工具图标路径
    image = 'mpl-data/images/help'

    @staticmethod
    def format_shortcut(key_sequence):
        """
        Convert a shortcut string from the notation used in rc config to the
        standard notation for displaying shortcuts, e.g. 'ctrl+a' -> 'Ctrl+A'.
        """
        # 将配置中的快捷键格式转换为标准的显示格式，如 'ctrl+a' -> 'Ctrl+A'
        return (key_sequence if len(key_sequence) == 1 else
                re.sub(r"\+[A-Z]", r"+Shift\g<0>", key_sequence).title())

    def _format_tool_keymap(self, name):
        # 获取指定工具的快捷键列表，并格式化输出
        keymaps = self.toolmanager.get_tool_keymap(name)
        return ", ".join(self.format_shortcut(keymap) for keymap in keymaps)
    # 获取帮助条目的列表，每个条目包括工具名、格式化后的键映射、工具描述
    def _get_help_entries(self):
        return [(name, self._format_tool_keymap(name), tool.description)
                for name, tool in sorted(self.toolmanager.tools.items())
                if tool.description]

    # 获取帮助文本，格式为每行包括工具名、键映射、工具描述
    def _get_help_text(self):
        # 获取所有帮助条目
        entries = self._get_help_entries()
        # 格式化每个条目，生成字符串列表
        entries = ["{}: {}\n\t{}".format(*entry) for entry in entries]
        # 将所有条目连接成一个字符串，用换行符分隔
        return "\n".join(entries)

    # 获取帮助内容的 HTML 格式
    def _get_help_html(self):
        # HTML 表格每行的格式字符串
        fmt = "<tr><td>{}</td><td>{}</td><td>{}</td></tr>"
        # 表头行，列分别为Action、Shortcuts、Description
        rows = [fmt.format(
            "<b>Action</b>", "<b>Shortcuts</b>", "<b>Description</b>")]
        # 添加每个工具的帮助条目到表格的行中
        rows += [fmt.format(*row) for row in self._get_help_entries()]
        # 构建包含样式和表格内容的完整 HTML 结果
        return ("<style>td {padding: 0px 4px}</style>"
                "<table><thead>" + rows[0] + "</thead>"
                "<tbody>".join(rows[1:]) + "</tbody></table>")
class ToolCopyToClipboardBase(ToolBase):
    """Tool to copy the figure to the clipboard."""

    # 描述工具的功能
    description = 'Copy the canvas figure to clipboard'

    # 默认的快捷键映射，使用了属性(lambda self: mpl.rcParams['keymap.copy'])来动态获取
    default_keymap = property(lambda self: mpl.rcParams['keymap.copy'])

    # 触发函数，当调用时显示“Copy tool is not available”的消息
    def trigger(self, *args, **kwargs):
        message = "Copy tool is not available"
        self.toolmanager.message_event(message, self)


# 默认工具字典，将工具名映射到对应的工具类
default_tools = {'home': ToolHome, 'back': ToolBack, 'forward': ToolForward,
                 'zoom': ToolZoom, 'pan': ToolPan,
                 'subplots': ConfigureSubplotsBase,
                 'save': SaveFigureBase,
                 'grid': ToolGrid,
                 'grid_minor': ToolMinorGrid,
                 'fullscreen': ToolFullScreen,
                 'quit': ToolQuit,
                 'quit_all': ToolQuitAll,
                 'xscale': ToolXScale,
                 'yscale': ToolYScale,
                 'position': ToolCursorPosition,
                 _views_positions: ToolViewsPositions,
                 'cursor': ToolSetCursor,
                 'rubberband': RubberbandBase,
                 'help': ToolHelpBase,
                 'copy': ToolCopyToClipboardBase,  # 添加了“copy”工具，使用ToolCopyToClipboardBase类
                 }

# 默认工具栏工具列表，分组展示不同的工具
default_toolbar_tools = [['navigation', ['home', 'back', 'forward']],
                         ['zoompan', ['pan', 'zoom', 'subplots']],
                         ['io', ['save', 'help']]]


def add_tools_to_manager(toolmanager, tools=default_tools):
    """
    Add multiple tools to a `.ToolManager`.

    Parameters
    ----------
    toolmanager : `.backend_managers.ToolManager`
        Manager to which the tools are added.
    tools : {str: class_like}, optional
        The tools to add in a {name: tool} dict, see
        `.backend_managers.ToolManager.add_tool` for more info.
    """

    # 遍历工具字典，将每个工具添加到ToolManager中
    for name, tool in tools.items():
        toolmanager.add_tool(name, tool)


def add_tools_to_container(container, tools=default_toolbar_tools):
    """
    Add multiple tools to the container.

    Parameters
    ----------
    container : Container
        `.backend_bases.ToolContainerBase` object that will get the tools
        added.
    tools : list, optional
        List in the form ``[[group1, [tool1, tool2 ...]], [group2, [...]]]``
        where the tools ``[tool1, tool2, ...]`` will display in group1.
        See `.backend_bases.ToolContainerBase.add_tool` for details.
    """

    # 遍历工具列表，将每个工具添加到指定的容器中
    for group, grouptools in tools:
        for position, tool in enumerate(grouptools):
            container.add_tool(tool, group, position)
```