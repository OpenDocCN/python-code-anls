# `D:\src\scipysrc\matplotlib\galleries\examples\user_interfaces\mplcvd.py`

```py
"""
mplcvd -- an example of figure hook
===================================

To use this hook, ensure that this module is in your ``PYTHONPATH``, and set
``rcParams["figure.hooks"] = ["mplcvd:setup"]``.  This hook depends on
the ``colorspacious`` third-party module.
"""

# 导入必要的库和模块
import functools  # 用于高阶函数（Higher-order functions）
from pathlib import Path  # 处理路径的模块

import colorspacious  # 使用颜色空间转换的第三方模块

import numpy as np  # 处理数组的数值运算

# 按钮名称和帮助文本
_BUTTON_NAME = "Filter"
_BUTTON_HELP = "Simulate color vision deficiencies"

# 菜单条目字典，定义不同的视觉缺陷模拟选项
_MENU_ENTRIES = {
    "None": None,
    "Greyscale": "greyscale",
    "Deuteranopia": "deuteranomaly",
    "Protanopia": "protanomaly",
    "Tritanopia": "tritanomaly",
}

def _get_color_filter(name):
    """
    根据给定的颜色过滤器名称，创建对应的颜色过滤器函数。

    Parameters
    ----------
    name : str
        颜色过滤器名称，可以是以下之一：

        - ``"none"``: ...
        - ``"greyscale"``: 将输入转换为亮度。
        - ``"deuteranopia"``: 模拟最常见的红绿色盲。
        - ``"protanopia"``: 模拟较少见的红绿色盲。
        - ``"tritanopia"``: 模拟罕见的蓝黄色盲。

        使用 `colorspacious`_ 进行颜色转换。

    Returns
    -------
    callable
        形式为：

        def filter(input: np.ndarray[M, N, D])-> np.ndarray[M, N, D]

        其中 (M, N) 是图像尺寸，D 是颜色深度（RGB 为 3，RGBA 为 4）。Alpha 通道保持不变或忽略。
    """
    # 如果名称不在支持的菜单条目中，则抛出值错误异常
    if name not in _MENU_ENTRIES:
        raise ValueError(f"Unsupported filter name: {name!r}")
    name = _MENU_ENTRIES[name]

    if name is None:
        return None

    elif name == "greyscale":
        # 定义将图像转换为灰度的函数
        rgb_to_jch = colorspacious.cspace_converter("sRGB1", "JCh")
        jch_to_rgb = colorspacious.cspace_converter("JCh", "sRGB1")

        def convert(im):
            greyscale_JCh = rgb_to_jch(im)
            greyscale_JCh[..., 1] = 0  # 将色度通道设为0，保留亮度信息
            im = jch_to_rgb(greyscale_JCh)
            return im

    else:
        # 定义根据给定视觉缺陷模拟名称转换图像颜色的函数
        cvd_space = {"name": "sRGB1+CVD", "cvd_type": name, "severity": 100}
        convert = colorspacious.cspace_converter(cvd_space, "sRGB1")

    # 定义应用颜色过滤器的函数
    def filter_func(im, dpi):
        alpha = None
        if im.shape[-1] == 4:
            im, alpha = im[..., :3], im[..., 3]
        im = convert(im)
        if alpha is not None:
            im = np.dstack((im, alpha))
        return np.clip(im, 0, 1), 0, 0

    return filter_func

def _set_menu_entry(tb, name):
    """
    设置菜单条目的处理函数，根据名称设置颜色过滤器。

    Parameters
    ----------
    tb : toolbar object
        工具栏对象，用于设置颜色过滤器。
    name : str
        要设置的颜色过滤器名称。
    """
    tb.canvas.figure.set_agg_filter(_get_color_filter(name))
    tb.canvas.draw_idle()

def setup(figure):
    """
    设置图形对象的回调函数，根据工具栏类型设置不同的界面处理。

    Parameters
    ----------
    figure : Figure
        要设置的图形对象。
    """
    tb = figure.canvas.toolbar
    if tb is None:
        return
    for cls in type(tb).__mro__:
        pkg = cls.__module__.split(".")[0]
        if pkg != "matplotlib":
            break
    # 根据工具栏类型的包名设置不同的界面处理
    if pkg == "gi":
        _setup_gtk(tb)
    elif pkg in ("PyQt5", "PySide2", "PyQt6", "PySide6"):
        _setup_qt(tb)
    # 如果 pkg 变量的取值为 "tkinter"，则调用 _setup_tk(tb) 函数
    elif pkg == "tkinter":
        _setup_tk(tb)
    # 如果 pkg 变量的取值为 "wx"，则调用 _setup_wx(tb) 函数
    elif pkg == "wx":
        _setup_wx(tb)
    # 如果 pkg 变量的取值既不是 "tkinter" 也不是 "wx"，则抛出未实现错误
    else:
        raise NotImplementedError("The current backend is not supported")
def _setup_gtk(tb):
    # 导入 GTK 所需的模块
    from gi.repository import Gio, GLib, Gtk

    # 遍历工具栏中的项目，找到第一个包含子元素且第一个子元素是 Gtk.Label 的项目索引
    for idx in range(tb.get_n_items()):
        children = tb.get_nth_item(idx).get_children()
        if children and isinstance(children[0], Gtk.Label):
            break

    # 在找到的项目索引处插入一个分隔符工具项
    toolitem = Gtk.SeparatorToolItem()
    tb.insert(toolitem, idx)

    # 从指定路径加载 SVG 图像作为 Gtk.Image 对象
    image = Gtk.Image.new_from_gicon(
        Gio.Icon.new_for_string(
            str(Path(__file__).parent / "images/eye-symbolic.svg")),
        Gtk.IconSize.LARGE_TOOLBAR)

    # 根据 GTK 版本选择不同的菜单类型
    if Gtk.check_version(3, 6, 0) is None:
        # 创建简单动作组并添加状态动作
        group = Gio.SimpleActionGroup.new()
        action = Gio.SimpleAction.new_stateful("cvdsim",
                                               GLib.VariantType("s"),
                                               GLib.Variant("s", "none"))
        group.add_action(action)

        # 连接动作的激活信号以设置菜单条目
        @functools.partial(action.connect, "activate")
        def set_filter(action, parameter):
            _set_menu_entry(tb, parameter.get_string())
            action.set_state(parameter)

        # 创建 Gio.Menu 并添加条目
        menu = Gio.Menu()
        for name in _MENU_ENTRIES:
            menu.append(name, f"local.cvdsim::{name}")

        # 创建带图像的菜单按钮并设置相关属性
        button = Gtk.MenuButton.new()
        button.remove(button.get_children()[0])
        button.add(image)
        button.insert_action_group("local", group)
        button.set_menu_model(menu)
        button.get_style_context().add_class("flat")

        # 创建工具项并插入到工具栏中
        item = Gtk.ToolItem()
        item.add(button)
        tb.insert(item, idx + 1)

    else:
        # 创建带有单选菜单项的 Gtk.Menu
        menu = Gtk.Menu()
        group = []
        for name in _MENU_ENTRIES:
            item = Gtk.RadioMenuItem.new_with_label(group, name)
            item.set_active(name == "None")
            item.connect(
                "activate", lambda item: _set_menu_entry(tb, item.get_label()))
            group.append(item)
            menu.append(item)
        menu.show_all()

        # 创建带菜单的工具按钮并插入到工具栏中
        tbutton = Gtk.MenuToolButton.new(image, _BUTTON_NAME)
        tbutton.set_menu(menu)
        tb.insert(tbutton, idx + 1)

    # 显示工具栏中的所有项目
    tb.show_all()
    # 设置按钮的显示文本为 _BUTTON_NAME 变量所指定的内容
    button.setText(_BUTTON_NAME)
    # 设置按钮的工具提示文本为 _BUTTON_HELP 变量所指定的内容
    button.setToolTip(_BUTTON_HELP)
    # 设置按钮的弹出模式为即时弹出模式，即鼠标移动到按钮上时立即弹出菜单
    button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    # 将按钮与预先创建好的菜单 menu 绑定，并插入到工具栏 tb 中指定位置 before 的前面
    tb.insertWidget(before, button)
# 导入 tkinter 库作为 tk 别名
def _setup_tk(tb):
    import tkinter as tk

    # 创建一个空间对象 _Spacer()，这里使用 FIXME 注释指出 _Spacer 需要公共 API
    tb._Spacer()  # FIXME: _Spacer needs public API.

    # 创建一个 Menubutton 对象 button，并设置它的外观为 "raised"
    button = tk.Menubutton(master=tb, relief="raised")
    # 将图片文件路径赋值给 button 的 _image_file 属性
    button._image_file = str(Path(__file__).parent / "images/eye.png")
    # FIXME: _set_image_for_button 需要公共 API（可能类似于 _icon）
    # 为 button 设置图像
    tb._set_image_for_button(button)
    # 将 button 放置在左侧
    button.pack(side=tk.LEFT)

    # 创建一个 Menu 对象 menu，其父对象是 button，且不允许 tearoff
    menu = tk.Menu(master=button, tearoff=False)
    # 遍历 _MENU_ENTRIES 中的每个条目名，为 menu 添加单选按钮
    for name in _MENU_ENTRIES:
        menu.add("radiobutton", label=name,
                 command=lambda _name=name: _set_menu_entry(tb, _name))
    # 执行 menu 的第一个条目对应的命令
    menu.invoke(0)
    # 将 menu 配置为 button 的菜单
    button.config(menu=menu)


# 导入 wx 库
def _setup_wx(tb):
    import wx

    # 找到 tb 中第一个 IsStretchableSpace() 的工具索引 idx，并在此处插入分隔符
    idx = next(idx for idx in range(tb.ToolsCount)
               if tb.GetToolByPos(idx).IsStretchableSpace())
    tb.InsertSeparator(idx)
    # 在 idx + 1 处插入一个工具 tool，设置工具名称为 _BUTTON_NAME
    # FIXME: _icon 需要公共 API
    # 使用图像文件创建工具，并将 kind 设置为 wx.ITEM_DROPDOWN，shortHelp 设置为 _BUTTON_HELP
    tool = tb.InsertTool(
        idx + 1, -1, _BUTTON_NAME,
        tb._icon(str(Path(__file__).parent / "images/eye.png")),
        # FIXME: ITEM_DROPDOWN 在 macOS 上不支持
        kind=wx.ITEM_DROPDOWN, shortHelp=_BUTTON_HELP)

    # 创建一个 Menu 对象 menu
    menu = wx.Menu()
    # 遍历 _MENU_ENTRIES 中的每个条目名，为 menu 添加单选项
    for name in _MENU_ENTRIES:
        # 创建一个单选菜单项 item，标签为 name
        item = menu.AppendRadioItem(-1, name)
        # 将菜单项的 Id 与 lambda 函数绑定，当选中时调用 _set_menu_entry(tb, _name)
        menu.Bind(
            wx.EVT_MENU,
            lambda event, _name=name: _set_menu_entry(tb, _name),
            id=item.Id,
        )
    # 设置工具的下拉菜单为 menu
    tb.SetDropdownMenu(tool.Id, menu)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from matplotlib import cbook

    # 将 'mplcvd:setup' 添加到 plt.rcParams['figure.hooks'] 列表中
    plt.rcParams['figure.hooks'].append('mplcvd:setup')

    # 创建一个图形 fig 和一个子图布局 axd
    fig, axd = plt.subplot_mosaic(
        [
            ['viridis', 'turbo'],
            ['photo', 'lines']
        ]
    )

    # 设置 delta 为 0.025，创建一组坐标点 x 和 y
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    # 计算 Z1 和 Z2
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    # 在 axd['viridis'] 中显示 Z 的图像，设置插值方法为 'bilinear'，原点为 'lower'，范围为 [-3, 3, -3, 3]
    # 设置颜色映射范围 vmax 和 vmin
    imv = axd['viridis'].imshow(
        Z, interpolation='bilinear',
        origin='lower', extent=[-3, 3, -3, 3],
        vmax=abs(Z).max(), vmin=-abs(Z).max()
    )
    # 为 fig 添加颜色条
    fig.colorbar(imv)
    # 在 axd['turbo'] 中显示 Z 的图像，设置颜色映射为 'turbo'
    imt = axd['turbo'].imshow(
        Z, interpolation='bilinear', cmap='turbo',
        origin='lower', extent=[-3, 3, -3, 3],
        vmax=abs(Z).max(), vmin=-abs(Z).max()
    )
    # 为 fig 添加颜色条
    fig.colorbar(imt)

    # 使用 cbook.get_sample_data 获取 'grace_hopper.jpg' 图像文件，并作为 photo 加载
    with cbook.get_sample_data('grace_hopper.jpg') as image_file:
        photo = plt.imread(image_file)
    # 在 axd['photo'] 中显示 photo 图像
    axd['photo'].imshow(photo)

    # 创建一个 theta 数组 th，以及多个正弦函数图像，并在 axd['lines'] 中显示，每个图像使用不同的标签
    th = np.linspace(0, 2*np.pi, 1024)
    for j in [1, 2, 4, 6]:
        axd['lines'].plot(th, np.sin(th * j), label=f'$\\omega={j}$')
    # 在 axd['lines'] 中显示图例，设置列数为 2，位置为 'upper right'
    axd['lines'].legend(ncols=2, loc='upper right')
    # 显示图形
    plt.show()
```