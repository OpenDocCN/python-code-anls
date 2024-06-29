# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_widgets.py`

```py
# 导入必要的模块和函数
import functools
import io
from unittest import mock

import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
                                        mock_event, noop)

import numpy as np
from numpy.testing import assert_allclose

import pytest

# 定义一个 pytest fixture，返回一个图轴对象
@pytest.fixture
def ax():
    return get_ax()

# 测试函数：测试将绘制的小部件保存为 PDF 文件
def test_save_blitted_widget_as_pdf():
    from matplotlib.widgets import CheckButtons, RadioButtons
    from matplotlib.cbook import _get_running_interactive_framework
    # 如果不是在无头模式或者未指定交互框架下运行，标记测试为预期失败
    if _get_running_interactive_framework() not in ['headless', None]:
        pytest.xfail("Callback exceptions are not raised otherwise.")

    # 创建一个包含 2x2 子图的图形对象
    fig, ax = plt.subplots(
        nrows=2, ncols=2, figsize=(5, 2), width_ratios=[1, 2]
    )
    # 在第一个子图中创建默认样式的单选按钮
    default_rb = RadioButtons(ax[0, 0], ['Apples', 'Oranges'])
    # 在第二个子图中创建自定义样式的单选按钮
    styled_rb = RadioButtons(
        ax[0, 1], ['Apples', 'Oranges'],
        label_props={'color': ['red', 'orange'],
                     'fontsize': [16, 20]},
        radio_props={'edgecolor': ['red', 'orange'],
                     'facecolor': ['mistyrose', 'peachpuff']}
    )

    # 在第三个子图中创建默认样式的复选框
    default_cb = CheckButtons(ax[1, 0], ['Apples', 'Oranges'],
                              actives=[True, True])
    # 在第四个子图中创建自定义样式的复选框
    styled_cb = CheckButtons(
        ax[1, 1], ['Apples', 'Oranges'],
        actives=[True, True],
        label_props={'color': ['red', 'orange'],
                     'fontsize': [16, 20]},
        frame_props={'edgecolor': ['red', 'orange'],
                     'facecolor': ['mistyrose', 'peachpuff']},
        check_props={'color': ['darkred', 'darkorange']}
    )

    # 设置第一个子图的标题
    ax[0, 0].set_title('Default')
    # 设置第二个子图的标题
    ax[0, 1].set_title('Stylized')
    # 强制进行 Agg 渲染
    fig.canvas.draw()
    # 强制将图形保存为 PDF 格式
    with io.BytesIO() as result_after:
        fig.savefig(result_after, format='pdf')

# 使用 pytest 的参数化装饰器来测试矩形选择器的不同参数组合
@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(useblit=True, button=1),
    dict(minspanx=10, minspany=10, spancoords='pixels'),
    dict(props=dict(fill=True)),
])
def test_rectangle_selector(ax, kwargs):
    # 创建一个模拟对象来替代 noop 函数
    onselect = mock.Mock(spec=noop, return_value=None)

    # 创建矩形选择器工具对象
    tool = widgets.RectangleSelector(ax, onselect, **kwargs)
    # 模拟按下事件，并指定坐标位置和按下的按钮
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    # 模拟移动事件，并指定移动后的坐标位置
    do_event(tool, 'onmove', xdata=199, ydata=199, button=1)

    # 故意在轴外拖动以释放
    do_event(tool, 'release', xdata=250, ydata=250, button=1)

    # 如果 drawtype 参数不是 'line' 或 'none'，则验证几何形状是否符合预期
    if kwargs.get('drawtype', None) not in ['line', 'none']:
        assert_allclose(tool.geometry,
                        [[100., 100, 199, 199, 100],
                         [100, 199, 199, 100, 100]],
                        err_msg=tool.geometry)

    # 验证回调函数被调用一次
    onselect.assert_called_once()
    # 检查回调函数被调用时传递的参数
    (epress, erelease), kwargs = onselect.call_args
    assert epress.xdata == 100
    # 断言 epress.ydata 等于 100
    assert epress.ydata == 100
    # 断言 erelease.xdata 等于 199
    assert erelease.xdata == 199
    # 断言 erelease.ydata 等于 199
    assert erelease.ydata == 199
    # 断言 kwargs 是一个空字典
    assert kwargs == {}
@pytest.mark.parametrize('spancoords', ['data', 'pixels'])
@pytest.mark.parametrize('minspanx, x1', [[0, 10], [1, 10.5], [1, 11]])
@pytest.mark.parametrize('minspany, y1', [[0, 10], [1, 10.5], [1, 11]])
def test_rectangle_minspan(ax, spancoords, minspanx, x1, minspany, y1):
    # 创建一个名为onselect的模拟对象，该对象基于noop函数，返回值为None
    onselect = mock.Mock(spec=noop, return_value=None)

    # 初始化起始点的坐标
    x0, y0 = (10, 10)
    # 如果spancoords为'pixels'，则重新计算minspanx和minspany的值
    if spancoords == 'pixels':
        minspanx, minspany = (ax.transData.transform((x1, y1)) -
                              ax.transData.transform((x0, y0)))

    # 创建一个RectangleSelector对象，并传入参数
    tool = widgets.RectangleSelector(ax, onselect, interactive=True,
                                     spancoords=spancoords,
                                     minspanx=minspanx, minspany=minspany)

    # 模拟点击并拖动操作，从(x0, y0)到(x1, y1)，断言选择未完成
    click_and_drag(tool, start=(x0, x1), end=(y0, y1))
    assert not tool._selection_completed
    onselect.assert_not_called()

    # 再次模拟点击并拖动操作，从(20, 20)到(30, 30)，断言选择已完成
    click_and_drag(tool, start=(20, 20), end=(30, 30))
    assert tool._selection_completed
    onselect.assert_called_once()

    # 再次模拟点击并拖动操作，从(x0, y0)到(x1, y1)，断言选择未完成，并触发onselect
    onselect.reset_mock()
    click_and_drag(tool, start=(x0, y0), end=(x1, y1))
    assert not tool._selection_completed
    onselect.assert_called_once()
    (epress, erelease), kwargs = onselect.call_args

    # 断言调用onselect时的参数
    assert epress.xdata == x0
    assert epress.ydata == y0
    assert erelease.xdata == x1
    assert erelease.ydata == y1
    assert kwargs == {}


def test_deprecation_selector_visible_attribute(ax):
    # 创建一个RectangleSelector对象，并断言其可见性为True
    tool = widgets.RectangleSelector(ax, lambda *args: None)
    assert tool.get_visible()

    # 使用pytest的warns函数断言访问tool.visible属性时会出现MatplotlibDeprecationWarning
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match="was deprecated in Matplotlib 3.8"):
        tool.visible


@pytest.mark.parametrize('drag_from_anywhere, new_center',
                         [[True, (60, 75)],
                          [False, (30, 20)]])
def test_rectangle_drag(ax, drag_from_anywhere, new_center):
    # 创建一个RectangleSelector对象，并传入参数
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True,
                                     drag_from_anywhere=drag_from_anywhere)

    # 创建矩形，从(0, 10)到(100, 120)，断言矩形的中心为(50, 65)
    click_and_drag(tool, start=(0, 10), end=(100, 120))
    assert tool.center == (50, 65)

    # 在矩形内部拖动，但离开中心手柄
    #
    # 若drag_from_anywhere为True，将移动矩形(10, 10)，新中心为(60, 75)
    #
    # 若drag_from_anywhere为False，将创建一个新矩形，中心为(30, 20)
    click_and_drag(tool, start=(25, 15), end=(35, 25))
    assert tool.center == new_center

    # 检查两种情况下，拖动矩形外部会绘制一个新矩形
    click_and_drag(tool, start=(175, 185), end=(185, 195))
    assert tool.center == (180, 190)


def test_rectangle_selector_set_props_handle_props(ax):
    # 待添加...
    # 创建一个矩形选择工具，并绑定到指定的轴对象 `ax` 上，设置交互模式为 True，当选择完成时调用 `noop` 函数。
    # 使用蓝色半透明的样式属性创建矩形选择框。
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True,
                                     props=dict(facecolor='b', alpha=0.2),
                                     handle_props=dict(alpha=0.5))
    
    # 在创建的矩形选择工具上进行点击并拖拽操作，定义起始点为 (0, 10) 和结束点为 (100, 120)。
    click_and_drag(tool, start=(0, 10), end=(100, 120))
    
    # 获取矩形选择工具的选择区域艺术家对象。
    artist = tool._selection_artist
    
    # 断言矩形选择区域的填充颜色是否为蓝色，并且透明度为 0.2。
    assert artist.get_facecolor() == mcolors.to_rgba('b', alpha=0.2)
    
    # 设置矩形选择工具的样式属性为红色填充，并且透明度为 0.3。
    tool.set_props(facecolor='r', alpha=0.3)
    
    # 再次断言矩形选择区域的填充颜色是否已更改为红色，并且透明度为 0.3。
    assert artist.get_facecolor() == mcolors.to_rgba('r', alpha=0.3)
    
    # 遍历矩形选择工具的所有手柄艺术家对象。
    for artist in tool._handles_artists:
        # 断言每个手柄的标记边缘颜色是否为黑色。
        assert artist.get_markeredgecolor() == 'black'
        # 断言每个手柄的透明度是否为 0.5。
        assert artist.get_alpha() == 0.5
    
    # 设置矩形选择工具的手柄属性，将标记边缘颜色设置为红色，并且透明度为 0.3。
    tool.set_handle_props(markeredgecolor='r', alpha=0.3)
    
    # 再次遍历矩形选择工具的所有手柄艺术家对象。
    for artist in tool._handles_artists:
        # 断言每个手柄的标记边缘颜色是否已更改为红色。
        assert artist.get_markeredgecolor() == 'r'
        # 断言每个手柄的透明度是否已更改为 0.3。
        assert artist.get_alpha() == 0.3
# 定义一个测试函数，用于测试矩形选择工具的调整大小功能
def test_rectangle_resize(ax):
    # 创建一个矩形选择工具，并指定回调函数为 `noop`，启用交互模式
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    
    # 创建一个矩形，起始点为 (0, 10)，结束点为 (100, 120)
    click_and_drag(tool, start=(0, 10), end=(100, 120))
    # 断言矩形工具的范围是否正确
    assert tool.extents == (0.0, 100.0, 10.0, 120.0)

    # 调整 NE 拖拽手柄
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdata_new, ydata_new = xdata + 10, ydata + 5
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    # 断言调整后的矩形工具范围是否正确
    assert tool.extents == (extents[0], xdata_new, extents[2], ydata_new)

    # 调整 E 拖拽手柄
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdata_new, ydata_new = xdata + 10, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    # 断言调整后的矩形工具范围是否正确
    assert tool.extents == (extents[0], xdata_new, extents[2], extents[3])

    # 调整 W 拖拽手柄
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdata_new, ydata_new = xdata + 15, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    # 断言调整后的矩形工具范围是否正确
    assert tool.extents == (xdata_new, extents[1], extents[2], extents[3])

    # 调整 SW 拖拽手柄
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdata_new, ydata_new = xdata + 20, ydata + 25
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    # 断言调整后的矩形工具范围是否正确
    assert tool.extents == (xdata_new, extents[1], ydata_new, extents[3])


# 定义一个测试函数，用于测试矩形选择工具添加状态的功能
def test_rectangle_add_state(ax):
    # 创建一个矩形选择工具，并指定回调函数为 `noop`，启用交互模式
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    
    # 创建一个矩形，起始点为 (70, 65)，结束点为 (125, 130)
    click_and_drag(tool, start=(70, 65), end=(125, 130))

    # 测试添加不支持的状态时是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        tool.add_state('unsupported_state')

    # 测试添加 'clear' 状态时是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        tool.add_state('clear')

    # 添加 'move'、'square'、'center' 状态
    tool.add_state('move')
    tool.add_state('square')
    tool.add_state('center')


# 使用参数化测试函数，用于测试矩形选择工具在添加状态时的调整大小功能
@pytest.mark.parametrize('add_state', [True, False])
def test_rectangle_resize_center(ax, add_state):
    # 创建一个矩形选择工具，并指定回调函数为 `noop`，启用交互模式
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    
    # 创建一个矩形，起始点为 (70, 65)，结束点为 (125, 130)
    click_and_drag(tool, start=(70, 65), end=(125, 130))
    # 断言矩形工具的范围是否正确
    assert tool.extents == (70.0, 125.0, 65.0, 130.0)

    if add_state:
        # 如果 `add_state` 为 True，则添加 'center' 状态，不使用任何键
        tool.add_state('center')
        use_key = None
    else:
        # 如果 `add_state` 为 False，则不添加状态，并使用 'control' 键
        use_key = 'control'

    # 调整 NE 拖拽手柄
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdiff, ydiff = 10, 5
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 断言调整后的矩形工具范围是否正确
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2] - ydiff, ydata_new)

    # 调整 E 拖拽手柄
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 10
    xdata_new, ydata_new = xdata + xdiff, ydata
    # 调用函数 click_and_drag()，使用指定工具、起始点和终点坐标进行点击拖拽操作，指定使用的键
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 断言工具对象的范围是否符合预期，左、右侧范围发生改变
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2], extents[3])

    # 调整东侧手柄，确保负偏移
    extents = tool.extents
    # 计算当前范围的中心点坐标
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -20
    # 计算新的 X 坐标，Y 坐标保持不变
    xdata_new, ydata_new = xdata + xdiff, ydata
    # 再次调用 click_and_drag() 进行拖拽操作
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 断言工具对象的范围是否符合预期，左侧范围发生改变
    assert tool.extents == (extents[0] - xdiff, xdata_new,
                            extents[2], extents[3])

    # 调整西侧手柄
    extents = tool.extents
    # 获取当前范围的左侧坐标和中心 Y 坐标
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 15
    # 计算新的 X 坐标，Y 坐标保持不变
    xdata_new, ydata_new = xdata + xdiff, ydata
    # 再次调用 click_and_drag() 进行拖拽操作
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 断言工具对象的范围是否符合预期，右侧范围发生改变
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            extents[2], extents[3])

    # 调整西侧手柄，确保负偏移
    extents = tool.extents
    # 获取当前范围的左侧坐标和中心 Y 坐标
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -25
    # 计算新的 X 坐标，Y 坐标保持不变
    xdata_new, ydata_new = xdata + xdiff, ydata
    # 再次调用 click_and_drag() 进行拖拽操作
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 断言工具对象的范围是否符合预期，右侧范围发生改变
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            extents[2], extents[3])

    # 调整西南侧手柄
    extents = tool.extents
    # 获取当前范围的左侧和下侧坐标
    xdata, ydata = extents[0], extents[2]
    xdiff, ydiff = 20, 25
    # 计算新的 X 和 Y 坐标
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    # 再次调用 click_and_drag() 进行拖拽操作
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 断言工具对象的范围是否符合预期，右侧和下侧范围发生改变
    assert tool.extents == (xdata_new, extents[1] - xdiff,
                            ydata_new, extents[3] - ydiff)
@pytest.mark.parametrize('add_state', [True, False])
def test_rectangle_resize_square(ax, add_state):
    # 使用 pytest 的 parametrize 功能，测试函数 test_rectangle_resize_square 会被分别传入 add_state 为 True 和 False 的参数

    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    # 创建一个矩形选择工具，绑定到给定的坐标轴 ax 上，当选择完成时调用 noop 函数，开启交互模式

    # 创建矩形
    click_and_drag(tool, start=(70, 65), end=(120, 115))
    # 模拟点击并拖动操作，从坐标 (70, 65) 到 (120, 115) 创建矩形
    assert tool.extents == (70.0, 120.0, 65.0, 115.0)
    # 断言当前工具的范围是否符合预期

    if add_state:
        tool.add_state('square')
        use_key = None
    else:
        use_key = 'shift'
    # 根据 add_state 决定是否将工具状态设为 'square'，并设置 use_key 作为拖动时的修饰键（如果 add_state 为 False，则为 'shift'）

    # 调整 NE 句柄
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdiff, ydiff = 10, 5
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 模拟拖动 NE 句柄，从 (xdata, ydata) 到 (xdata_new, ydata_new)，使用 use_key 作为修饰键
    assert tool.extents == (extents[0], xdata_new,
                            extents[2], extents[3] + xdiff)
    # 断言调整后工具的范围是否符合预期

    # 调整 E 句柄
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 10
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 模拟拖动 E 句柄，从 (xdata, ydata) 到 (xdata_new, ydata_new)，使用 use_key 作为修饰键
    assert tool.extents == (extents[0], xdata_new,
                            extents[2], extents[3] + xdiff)
    # 断言调整后工具的范围是否符合预期

    # 调整 E 句柄，负向调整
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -20
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 模拟负向拖动 E 句柄，从 (xdata, ydata) 到 (xdata_new, ydata_new)，使用 use_key 作为修饰键
    assert tool.extents == (extents[0], xdata_new,
                            extents[2], extents[3] + xdiff)
    # 断言调整后工具的范围是否符合预期

    # 调整 W 句柄
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 15
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 模拟拖动 W 句柄，从 (xdata, ydata) 到 (xdata_new, ydata_new)，使用 use_key 作为修饰键
    assert tool.extents == (xdata_new, extents[1],
                            extents[2], extents[3] - xdiff)
    # 断言调整后工具的范围是否符合预期

    # 调整 W 句柄，负向调整
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -25
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 模拟负向拖动 W 句柄，从 (xdata, ydata) 到 (xdata_new, ydata_new)，使用 use_key 作为修饰键
    assert tool.extents == (xdata_new, extents[1],
                            extents[2], extents[3] - xdiff)
    # 断言调整后工具的范围是否符合预期

    # 调整 SW 句柄
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdiff, ydiff = 20, 25
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new),
                   key=use_key)
    # 模拟拖动 SW 句柄，从 (xdata, ydata) 到 (xdata_new, ydata_new)，使用 use_key 作为修饰键
    assert tool.extents == (extents[0] + ydiff, extents[1],
                            ydata_new, extents[3])
    # 断言调整后工具的范围是否符合预期


def test_rectangle_resize_square_center(ax):
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    # 创建一个矩形选择工具，绑定到给定的坐标轴 ax 上，当选择完成时调用 noop 函数，开启交互模式
    # 点击并拖动工具，从起始点 (70, 65) 到终点 (120, 115)
    click_and_drag(tool, start=(70, 65), end=(120, 115))
    
    # 向工具添加状态 'square'
    tool.add_state('square')
    
    # 向工具添加状态 'center'
    tool.add_state('center')
    
    # 断言工具的范围是否与期望值 (70.0, 120.0, 65.0, 115.0) 几乎相等
    assert_allclose(tool.extents, (70.0, 120.0, 65.0, 115.0))

    # 调整东北角手柄大小
    extents = tool.extents
    xdata, ydata = extents[1], extents[3]
    xdiff, ydiff = 10, 5
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert_allclose(tool.extents, (extents[0] - xdiff, xdata_new,
                                   extents[2] - xdiff, extents[3] + ydiff))

    # 调整东方手柄大小
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 10
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert_allclose(tool.extents, (extents[0] - xdiff, xdata_new,
                                   extents[2] - xdiff, extents[3] + xdiff))

    # 调整东方手柄大小，负差异
    extents = tool.extents
    xdata, ydata = extents[1], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -20
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert_allclose(tool.extents, (extents[0] - xdiff, xdata_new,
                                   extents[2] - xdiff, extents[3] + xdiff))

    # 调整西方手柄大小
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = 5
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert_allclose(tool.extents, (xdata_new, extents[1] - xdiff,
                                   extents[2] + xdiff, extents[3] - xdiff))

    # 调整西方手柄大小，负差异
    extents = tool.extents
    xdata, ydata = extents[0], extents[2] + (extents[3] - extents[2]) / 2
    xdiff = -25
    xdata_new, ydata_new = xdata + xdiff, ydata
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert_allclose(tool.extents, (xdata_new, extents[1] - xdiff,
                                   extents[2] + xdiff, extents[3] - xdiff))

    # 调整西南角手柄大小
    extents = tool.extents
    xdata, ydata = extents[0], extents[2]
    xdiff, ydiff = 20, 25
    xdata_new, ydata_new = xdata + xdiff, ydata + ydiff
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert_allclose(tool.extents, (extents[0] + ydiff, extents[1] - ydiff,
                                   ydata_new, extents[3] - ydiff))
@pytest.mark.parametrize('selector_class',
                         [widgets.RectangleSelector, widgets.EllipseSelector])
# 定义测试函数 test_rectangle_rotate，使用参数化装饰器，测试矩形选择器和椭圆选择器两种选择器类
def test_rectangle_rotate(ax, selector_class):
    # 使用给定的选择器类在给定的轴上创建选择器工具，设定选择时的回调函数为 noop，启用交互模式
    tool = selector_class(ax, onselect=noop, interactive=True)
    # 绘制矩形
    click_and_drag(tool, start=(100, 100), end=(130, 140))
    # 断言选择工具的范围是否正确
    assert tool.extents == (100, 130, 100, 140)
    # 断言工具状态列表长度是否为 0
    assert len(tool._state) == 0

    # 使用键盘事件模拟逆时针旋转，以右上角为中心
    do_event(tool, 'on_key_press', key='r')
    # 断言工具状态包含 'rotate'
    assert tool._state == {'rotate'}
    # 断言工具状态列表长度为 1
    assert len(tool._state) == 1
    # 拖动以模拟旋转操作
    click_and_drag(tool, start=(130, 140), end=(120, 145))
    # 再次使用键盘事件模拟旋转
    do_event(tool, 'on_key_press', key='r')
    # 断言工具状态列表长度为 0
    assert len(tool._state) == 0
    # 确保矩形形状未改变，因此范围不应改变
    assert tool.extents == (100, 130, 100, 140)
    # 断言角度近似于 25.56 度，容差为 0.01
    assert_allclose(tool.rotation, 25.56, atol=0.01)
    # 设置旋转角度为 45 度，并断言设置成功
    tool.rotation = 45
    assert tool.rotation == 45
    # 断言矩形角落的位置近似于给定的数组，容差为 0.01
    assert_allclose(tool.corners,
                    np.array([[118.53, 139.75, 111.46, 90.25],
                              [95.25, 116.46, 144.75, 123.54]]), atol=0.01)

    # 使用右上角作为中心进行缩放
    click_and_drag(tool, start=(110, 145), end=(110, 160))
    # 断言范围是否近似于给定的值，容差为 0.01
    assert_allclose(tool.extents, (100, 139.75, 100, 151.82), atol=0.01)

    # 如果选择器类是矩形选择器，应当引发 ValueError 异常
    if selector_class == widgets.RectangleSelector:
        with pytest.raises(ValueError):
            tool._selection_artist.rotation_point = 'unvalid_value'


# 定义测试函数 test_rectangle_add_remove_set，测试矩形选择器的添加、移除和设置状态功能
def test_rectangle_add_remove_set(ax):
    # 在给定的轴上创建矩形选择器，设定选择时的回调函数为 noop，启用交互模式
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    # 绘制矩形
    click_and_drag(tool, start=(100, 100), end=(130, 140))
    # 断言选择工具的范围是否正确
    assert tool.extents == (100, 130, 100, 140)
    # 断言工具状态列表长度是否为 0
    assert len(tool._state) == 0
    # 遍历多种状态，测试添加和移除状态后状态列表长度的变化
    for state in ['rotate', 'square', 'center']:
        tool.add_state(state)
        assert len(tool._state) == 1
        tool.remove_state(state)
        assert len(tool._state) == 0


@pytest.mark.parametrize('use_data_coordinates', [False, True])
# 定义测试函数 test_rectangle_resize_square_center_aspect，测试矩形选择器的调整、正方形和中心状态的应用
def test_rectangle_resize_square_center_aspect(ax, use_data_coordinates):
    # 设置轴的纵横比为 0.8
    ax.set_aspect(0.8)

    # 在给定的轴上创建矩形选择器，设定选择时的回调函数为 noop，启用交互模式，根据 use_data_coordinates 参数选择是否使用数据坐标
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True,
                                     use_data_coordinates=use_data_coordinates)
    # 创建矩形
    click_and_drag(tool, start=(70, 65), end=(120, 115))
    # 断言选择工具的范围是否正确
    assert tool.extents == (70.0, 120.0, 65.0, 115.0)
    # 添加正方形和中心状态
    tool.add_state('square')
    tool.add_state('center')

    if use_data_coordinates:
        # 调整 E 操作柄
        extents = tool.extents
        xdata, ydata, width = extents[1], extents[3], extents[1] - extents[0]
        xdiff, ycenter = 10,  extents[2] + (extents[3] - extents[2]) / 2
        xdata_new, ydata_new = xdata + xdiff, ydata
        ychange = width / 2 + xdiff
        click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
        # 断言调整后的范围近似于给定的值，容差为 0.01
        assert_allclose(tool.extents, [extents[0] - xdiff, xdata_new,
                                       ycenter - ychange, ycenter + ychange])
    else:
        # 如果不是特定条件下的情况，执行以下操作
        # 获取工具对象的边界信息
        extents = tool.extents
        # 从边界信息中获取 x 和 y 的数据
        xdata, ydata = extents[1], extents[3]
        # 设定 x 轴的偏移量为 10
        xdiff = 10
        # 计算新的 x 数据和不变的 y 数据
        xdata_new, ydata_new = xdata + xdiff, ydata
        # 根据比例修正 y 轴的变化量
        ychange = xdiff * 1 / tool._aspect_ratio_correction
        # 调用函数进行点击并拖动操作，起始点为原始 x 和 y 数据，终点为新的 x 和不变的 y 数据
        click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
        # 断言检查工具对象的边界信息是否与预期值接近
        assert_allclose(tool.extents, [extents[0] - xdiff, xdata_new,
                                       46.25, 133.75])
# 定义测试椭圆工具的函数，接受一个绘图轴对象 ax 作为参数
def test_ellipse(ax):
    """For ellipse, test out the key modifiers"""

    # 创建一个椭圆选择工具，绑定到绘图轴 ax 上，当选择结束时调用 noop 函数
    tool = widgets.EllipseSelector(ax, onselect=noop,
                                   grab_range=10, interactive=True)
    # 设置工具的范围为 (100, 150, 100, 150)
    tool.extents = (100, 150, 100, 150)

    # 拖动矩形，起点 (125, 125)，终点 (145, 145)
    click_and_drag(tool, start=(125, 125), end=(145, 145))
    # 断言工具的范围更新为 (120, 170, 120, 170)
    assert tool.extents == (120, 170, 120, 170)

    # 从中心创建椭圆，起点 (100, 100)，终点 (125, 125)，按下 control 键
    click_and_drag(tool, start=(100, 100), end=(125, 125), key='control')
    # 断言工具的范围更新为 (75, 125, 75, 125)
    assert tool.extents == (75, 125, 75, 125)

    # 创建一个正方形，起点 (10, 10)，终点 (35, 30)，按下 shift 键
    click_and_drag(tool, start=(10, 10), end=(35, 30), key='shift')
    # 将工具范围的四个值转换为整数并进行断言
    extents = [int(e) for e in tool.extents]
    assert extents == [10, 35, 10, 35]

    # 从中心创建一个正方形，起点 (100, 100)，终点 (125, 130)，按下 ctrl+shift 键
    click_and_drag(tool, start=(100, 100), end=(125, 130), key='ctrl+shift')
    # 将工具范围的四个值转换为整数并进行断言
    extents = [int(e) for e in tool.extents]
    assert extents == [70, 130, 70, 130]

    # 断言工具的几何形状为 (2, 73)
    assert tool.geometry.shape == (2, 73)
    # 断言工具的几何形状的第一列元素近似于 [70., 100]
    assert_allclose(tool.geometry[:, 0], [70., 100])


# 定义测试矩形工具手柄的函数，接受一个绘图轴对象 ax 作为参数
def test_rectangle_handles(ax):
    tool = widgets.RectangleSelector(ax, onselect=noop,
                                     grab_range=10,
                                     interactive=True,
                                     handle_props={'markerfacecolor': 'r',
                                                   'markeredgecolor': 'b'})
    # 设置工具的范围为 (100, 150, 100, 150)
    tool.extents = (100, 150, 100, 150)

    # 断言所有角落的坐标近似为 ((100, 150, 150, 100), (100, 100, 150, 150))
    assert_allclose(tool.corners, ((100, 150, 150, 100), (100, 100, 150, 150)))
    # 再次断言工具的范围为 (100, 150, 100, 150)
    assert tool.extents == (100, 150, 100, 150)
    # 断言所有边缘中心点的坐标近似为 ((100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150))
    assert_allclose(tool.edge_centers,
                    ((100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150)))
    # 再次断言工具的范围为 (100, 150, 100, 150)

    # 抓取一个角落并移动它，起点 (100, 100)，终点 (120, 120)
    click_and_drag(tool, start=(100, 100), end=(120, 120))
    # 断言工具的范围更新为 (120, 150, 120, 150)
    assert tool.extents == (120, 150, 120, 150)

    # 抓取中心并移动它，起点 (132, 132)，终点 (120, 120)
    click_and_drag(tool, start=(132, 132), end=(120, 120))
    # 断言工具的范围更新为 (108, 138, 108, 138)
    assert tool.extents == (108, 138, 108, 138)

    # 创建一个新的矩形，起点 (10, 10)，终点 (100, 100)
    click_and_drag(tool, start=(10, 10), end=(100, 100))
    # 断言工具的范围更新为 (10, 100, 10, 100)

    # 检查 marker_props 是否生效
    assert mcolors.same_color(
        tool._corner_handles.artists[0].get_markerfacecolor(), 'r')
    assert mcolors.same_color(
        tool._corner_handles.artists[0].get_markeredgecolor(), 'b')


@pytest.mark.parametrize('interactive', [True, False])
def test_rectangle_selector_onselect(ax, interactive):
    """Test RectangleSelector with onselect callback"""
    # 创建一个矩形选择工具，绑定到绘图轴 ax 上，当选择结束时调用 onselect 函数
    onselect = mock.Mock(spec=noop, return_value=None)

    tool = widgets.RectangleSelector(ax, onselect, interactive=interactive)

    # 模拟移动到轴外的事件，起点 (100, 110)，终点 (150, 120)
    click_and_drag(tool, start=(100, 110), end=(150, 120))

    # 断言 onselect 函数被调用一次
    onselect.assert_called_once()
    # 断言工具的范围更新为 (100.0, 150.0, 110.0, 120.0)

    onselect.reset_mock()
    # 模拟在同一位置按下和释放事件，起点 (10, 100)，终点 (10, 100)
    click_and_drag(tool, start=(10, 100), end=(10, 100))
    # 断言 onselect 函数被调用一次
    onselect.assert_called_once()
# 测试矩形选择器是否忽略外部事件设置
def test_rectangle_selector_ignore_outside(ax, ignore_event_outside):
    # 创建一个模拟对象 onselect，其规范是 noop，返回值为 None
    onselect = mock.Mock(spec=noop, return_value=None)

    # 使用给定的轴对象 ax 和 onselect 函数创建一个矩形选择器工具
    tool = widgets.RectangleSelector(ax, onselect,
                                     ignore_event_outside=ignore_event_outside)
    # 模拟鼠标点击并拖动操作，起点 (100, 110)，终点 (150, 120)
    click_and_drag(tool, start=(100, 110), end=(150, 120))
    # 确保 onselect 函数被调用了一次
    onselect.assert_called_once()
    # 确保工具的范围属性正确设置为 (100.0, 150.0, 110.0, 120.0)
    assert tool.extents == (100.0, 150.0, 110.0, 120.0)

    # 重置 onselect 模拟对象的状态
    onselect.reset_mock()
    # 触发一个超出范围的事件
    click_and_drag(tool, start=(150, 150), end=(160, 160))
    if ignore_event_outside:
        # 如果设置忽略外部事件，确保 onselect 函数没有被调用
        onselect.assert_not_called()
        # 确保工具的范围属性保持为 (100.0, 150.0, 110.0, 120.0)
        assert tool.extents == (100.0, 150.0, 110.0, 120.0)
    else:
        # 如果不忽略外部事件，确保 onselect 函数被调用了一次
        onselect.assert_called_once()
        # 确保工具的范围属性更新为新选择的范围 (150.0, 160.0, 150.0, 160.0)
        assert tool.extents == (150.0, 160.0, 150.0, 160.0)


# 使用参数化测试来测试不同配置下的 span selector 行为
@pytest.mark.parametrize('orientation, onmove_callback, kwargs', [
    ('horizontal', False, dict(minspan=10, useblit=True)),
    ('vertical', True, dict(button=1)),
    ('horizontal', False, dict(props=dict(fill=True))),
    ('horizontal', False, dict(interactive=True)),
])
def test_span_selector(ax, orientation, onmove_callback, kwargs):
    # 创建一个模拟对象 onselect，其规范是 noop，返回值为 None
    onselect = mock.Mock(spec=noop, return_value=None)
    onmove = mock.Mock(spec=noop, return_value=None)
    if onmove_callback:
        kwargs['onmove_callback'] = onmove

    # 在这个测试中，同时测试 span selector 在存在双轴的情况下的工作。
    # 需要注意，这里需要取消轴的纵横比限制，否则双轴会强制原始轴的限制
    ax.set_aspect("auto")
    tax = ax.twinx()

    # 使用给定的轴对象 ax 和 onselect 函数创建一个 span selector 工具
    tool = widgets.SpanSelector(ax, onselect, orientation, **kwargs)
    # 模拟事件：按下 (press) 事件，坐标 (100, 100)，按钮为 1
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    # 模拟事件：移动 (onmove) 事件，坐标 (199, 199)，按钮为 1
    do_event(tool, 'onmove', xdata=199, ydata=199, button=1)
    # 模拟事件：释放 (release) 事件，坐标 (250, 250)，按钮为 1
    do_event(tool, 'release', xdata=250, ydata=250, button=1)

    # 确保 onselect 函数被正确调用一次，传入的参数为 (100, 199)
    onselect.assert_called_once_with(100, 199)
    if onmove_callback:
        # 如果有 onmove 回调函数，确保其被正确调用一次，传入的参数为 (100, 199)
        onmove.assert_called_once_with(100, 199)


# 使用参数化测试来测试 span selector 在不同交互模式下的行为
@pytest.mark.parametrize('interactive', [True, False])
def test_span_selector_onselect(ax, interactive):
    # 创建一个模拟对象 onselect，其规范是 noop，返回值为 None
    onselect = mock.Mock(spec=noop, return_value=None)

    # 使用给定的轴对象 ax 和 onselect 函数创建一个 span selector 工具
    tool = widgets.SpanSelector(ax, onselect, 'horizontal',
                                interactive=interactive)
    # 模拟鼠标点击并拖动操作，起点 (100, 100)，终点 (150, 100)
    click_and_drag(tool, start=(100, 100), end=(150, 100))
    # 确保 onselect 函数被调用了一次
    onselect.assert_called_once()
    # 确保工具的范围属性正确设置为 (100, 150)
    assert tool.extents == (100, 150)

    # 重置 onselect 模拟对象的状态
    onselect.reset_mock()
    # 模拟鼠标点击并拖动操作，起点 (10, 100)，终点 (10, 100)
    click_and_drag(tool, start=(10, 100), end=(10, 100))
    # 确保 onselect 函数被调用了一次
    onselect.assert_called_once()


# 使用参数化测试来测试 span selector 是否忽略外部事件设置
@pytest.mark.parametrize('ignore_event_outside', [True, False])
def test_span_selector_ignore_outside(ax, ignore_event_outside):
    # 创建一个模拟对象 onselect，其规范是 noop，返回值为 None
    onselect = mock.Mock(spec=noop, return_value=None)
    onmove = mock.Mock(spec=noop, return_value=None)
    # 创建一个 SpanSelector 对象，绑定到图形 ax 上，水平方向选择，使用指定的回调函数和事件忽略标志
    tool = widgets.SpanSelector(ax, onselect, 'horizontal',
                                onmove_callback=onmove,
                                ignore_event_outside=ignore_event_outside)
    # 模拟点击并拖动操作，从起点 (100, 100) 到终点 (125, 125)
    click_and_drag(tool, start=(100, 100), end=(125, 125))
    # 确保 onselect 回调函数被调用一次
    onselect.assert_called_once()
    # 确保 onmove 回调函数被调用一次
    onmove.assert_called_once()
    # 确保 SpanSelector 对象的范围被设置为 (100, 125)
    assert tool.extents == (100, 125)

    # 重置 mock 对象，为下一次调用做准备
    onselect.reset_mock()
    onmove.reset_mock()
    # 触发一个超出选择范围的事件
    click_and_drag(tool, start=(150, 150), end=(160, 160))
    if ignore_event_outside:
        # 如果忽略超出范围的事件，则不应该调用 onselect 和 onmove 函数
        onselect.assert_not_called()
        onmove.assert_not_called()
        # 确保 SpanSelector 对象的范围仍然是 (100, 125)
        assert tool.extents == (100, 125)
    else:
        # 如果不忽略超出范围的事件，则应该调用一次 onselect 和 onmove 函数
        onselect.assert_called_once()
        onmove.assert_called_once()
        # 确保 SpanSelector 对象的范围已更新为 (150, 160)
        assert tool.extents == (150, 160)
@pytest.mark.parametrize('drag_from_anywhere', [True, False])
def test_span_selector_drag(ax, drag_from_anywhere):
    # 使用pytest标记parametrize装饰器，为单元测试函数提供多组参数化输入

    # 创建一个SpanSelector工具对象，绑定到给定的Axes对象ax上，当选择完成时调用noop函数
    # 设置工具的方向为水平，启用交互模式，并根据drag_from_anywhere参数决定是否可以从任意位置拖动
    tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal',
                                interactive=True,
                                drag_from_anywhere=drag_from_anywhere)
    
    # 模拟鼠标点击和拖动操作，从(10, 10)到(100, 120)
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    
    # 断言SpanSelector对象的选区范围是否为(10, 100)
    assert tool.extents == (10, 100)
    
    # 在选区内部进行拖动操作

    # 如果drag_from_anywhere为True，则将选区向右移动10个单位，期望结果范围为(20, 110)
    #
    # 如果drag_from_anywhere为False，则创建一个新的选区，期望结果范围为(25, 35)
    click_and_drag(tool, start=(25, 15), end=(35, 25))
    if drag_from_anywhere:
        assert tool.extents == (20, 110)
    else:
        assert tool.extents == (25, 35)

    # 检查在两种情况下，拖动选区外部会创建一个新的选区
    click_and_drag(tool, start=(175, 185), end=(185, 195))
    assert tool.extents == (175, 185)


def test_span_selector_direction(ax):
    # 创建一个水平方向的SpanSelector工具对象，并绑定到给定的Axes对象ax上，当选择完成时调用noop函数
    tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal',
                                interactive=True)
    
    # 断言工具对象的方向属性为水平
    assert tool.direction == 'horizontal'
    
    # 断言工具内部边缘处理对象的方向属性为水平
    assert tool._edge_handles.direction == 'horizontal'

    # 使用pytest断言预期引发值错误异常
    with pytest.raises(ValueError):
        # 创建一个SpanSelector工具对象，方向属性为无效字符串'invalid_direction'
        tool = widgets.SpanSelector(ax, onselect=noop,
                                    direction='invalid_direction')

    # 设置工具对象的方向属性为垂直
    tool.direction = 'vertical'
    # 断言工具对象的方向属性为垂直
    assert tool.direction == 'vertical'
    # 断言工具内部边缘处理对象的方向属性为垂直
    assert tool._edge_handles.direction == 'vertical'

    # 使用pytest断言预期引发值错误异常
    with pytest.raises(ValueError):
        # 设置工具对象的方向属性为无效字符串'invalid_string'
        tool.direction = 'invalid_string'


def test_span_selector_set_props_handle_props(ax):
    # 创建一个水平方向的SpanSelector工具对象，并绑定到给定的Axes对象ax上，当选择完成时调用noop函数
    # 设置工具对象的面属性为蓝色，透明度为0.2，手柄属性的透明度为0.5
    tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal',
                                interactive=True,
                                props=dict(facecolor='b', alpha=0.2),
                                handle_props=dict(alpha=0.5))
    
    # 创建一个矩形选择区域，从(0, 10)到(100, 120)
    click_and_drag(tool, start=(0, 10), end=(100, 120))

    # 获取选区的图形对象
    artist = tool._selection_artist
    # 断言选区图形对象的面颜色和透明度是否与预期一致
    assert artist.get_facecolor() == mcolors.to_rgba('b', alpha=0.2)
    
    # 修改工具对象的面属性为红色，透明度为0.3
    tool.set_props(facecolor='r', alpha=0.3)
    # 断言选区图形对象的面颜色和透明度是否与预期一致
    assert artist.get_facecolor() == mcolors.to_rgba('r', alpha=0.3)

    # 遍历并断言所有手柄图形对象的颜色为蓝色，透明度为0.5
    for artist in tool._handles_artists:
        assert artist.get_color() == 'b'
        assert artist.get_alpha() == 0.5
    
    # 修改工具对象的手柄属性的颜色为红色，透明度为0.3
    tool.set_handle_props(color='r', alpha=0.3)
    # 遍历并断言所有手柄图形对象的颜色为红色，透明度为0.3
    for artist in tool._handles_artists:
        assert artist.get_color() == 'r'
        assert artist.get_alpha() == 0.3


@pytest.mark.parametrize('selector', ['span', 'rectangle'])
def test_selector_clear(ax, selector):
    kwargs = dict(ax=ax, onselect=noop, interactive=True)
    if selector == 'span':
        # 如果选择器为'span'，则使用SpanSelector工具类
        Selector = widgets.SpanSelector
        kwargs['direction'] = 'horizontal'
    else:
        # 否则，使用RectangleSelector工具类
        Selector = widgets.RectangleSelector

    # 根据选择器类型创建相应的选择器工具对象，并绑定到给定的Axes对象ax上，当选择完成时调用noop函数
    tool = Selector(**kwargs)
    
    # 创建一个选择区域，从(10, 10)到(100, 120)
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    # 在选择器外部进行点击和拖动，以清除选择器状态
    click_and_drag(tool, start=(130, 130), end=(130, 130))
    # 断言选择器的选择完成状态为假
    assert not tool._selection_completed

    # 设置关键字参数中的 ignore_event_outside 为 True，并创建一个选择器对象
    kwargs['ignore_event_outside'] = True
    tool = Selector(**kwargs)
    # 断言选择器的 ignore_event_outside 属性为真
    assert tool.ignore_event_outside
    # 在选择器内部进行点击和拖动操作
    click_and_drag(tool, start=(10, 10), end=(100, 120))

    # 在选择器外部进行点击和拖动，预期选择器将忽略这个事件
    click_and_drag(tool, start=(130, 130), end=(130, 130))
    # 断言选择器的选择完成状态为真
    assert tool._selection_completed

    # 向选择器发送按键事件 'escape'
    do_event(tool, 'on_key_press', key='escape')
    # 断言选择器的选择完成状态为假
    assert not tool._selection_completed
@pytest.mark.parametrize('selector', ['span', 'rectangle'])
# 使用pytest的参数化装饰器，指定参数'selector'为'span'和'rectangle'，用于多次运行测试
def test_selector_clear_method(ax, selector):
    # 根据不同的'selector'值，创建不同的选择工具对象
    if selector == 'span':
        # 创建水平方向的SpanSelector对象
        tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal',
                                    interactive=True,
                                    ignore_event_outside=True)
    else:
        # 创建RectangleSelector对象
        tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    # 模拟鼠标点击和拖动操作
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    # 断言选择操作已完成
    assert tool._selection_completed
    # 断言选择工具可见
    assert tool.get_visible()
    # 如果'selector'为'span'，断言选择范围为(10, 100)
    if selector == 'span':
        assert tool.extents == (10, 100)

    # 清除选择工具的状态
    tool.clear()
    # 断言选择操作未完成
    assert not tool._selection_completed
    # 断言选择工具不可见
    assert not tool.get_visible()

    # 再次模拟鼠标点击和拖动操作，以确保操作正常
    click_and_drag(tool, start=(10, 10), end=(50, 120))
    # 断言选择操作已完成
    assert tool._selection_completed
    # 断言选择工具可见
    assert tool.get_visible()
    # 如果'selector'为'span'，断言选择范围为(10, 50)
    if selector == 'span':
        assert tool.extents == (10, 50)


def test_span_selector_add_state(ax):
    # 创建水平方向的SpanSelector对象
    tool = widgets.SpanSelector(ax, noop, 'horizontal',
                                interactive=True)

    # 使用pytest的断言检查，确保添加不支持的状态会引发 ValueError 异常
    with pytest.raises(ValueError):
        tool.add_state('unsupported_state')
    with pytest.raises(ValueError):
        tool.add_state('center')
    with pytest.raises(ValueError):
        tool.add_state('square')

    # 添加支持的状态'move'
    tool.add_state('move')


def test_tool_line_handle(ax):
    # 定义位置列表
    positions = [20, 30, 50]
    # 创建水平方向的ToolLineHandles对象
    tool_line_handle = widgets.ToolLineHandles(ax, positions, 'horizontal',
                                               useblit=False)

    # 遍历所有艺术家，断言动画和可见性状态为False
    for artist in tool_line_handle.artists:
        assert not artist.get_animated()
        assert not artist.get_visible()

    # 设置工具线条可见
    tool_line_handle.set_visible(True)
    # 设置工具线条为动画状态
    tool_line_handle.set_animated(True)

    # 再次遍历所有艺术家，断言动画和可见性状态为True
    for artist in tool_line_handle.artists:
        assert artist.get_animated()
        assert artist.get_visible()

    # 断言工具线条的位置等于定义的位置列表
    assert tool_line_handle.positions == positions


@pytest.mark.parametrize('direction', ("horizontal", "vertical"))
# 使用pytest的参数化装饰器，指定参数'direction'为'horizontal'和'vertical'，用于多次运行测试
def test_span_selector_bound(direction):
    # 创建单个子图的图像和坐标轴
    fig, ax = plt.subplots(1, 1)
    # 绘制直线
    ax.plot([10, 20], [10, 30])
    # 更新图像的绘制
    ax.figure.canvas.draw()
    # 获取当前坐标轴的X轴和Y轴范围
    x_bound = ax.get_xbound()
    y_bound = ax.get_ybound()

    # 创建与方向相关的SpanSelector对象
    tool = widgets.SpanSelector(ax, print, direction, interactive=True)
    # 断言坐标轴的X轴和Y轴范围未改变
    assert ax.get_xbound() == x_bound
    assert ax.get_ybound() == y_bound

    # 根据方向确定边界值
    bound = x_bound if direction == 'horizontal' else y_bound
    # 断言工具的边界处理器位置与边界值列表相匹配
    assert tool._edge_handles.positions == list(bound)

    # 定义鼠标按下、移动和释放时的数据
    press_data = (10.5, 11.5)
    move_data = (11, 13)  # 更新选择器在onmove中完成
    release_data = move_data
    # 模拟鼠标点击和拖动操作
    click_and_drag(tool, start=press_data, end=move_data)

    # 再次断言坐标轴的X轴和Y轴范围未改变
    assert ax.get_xbound() == x_bound
    assert ax.get_ybound() == y_bound

    # 根据方向确定位置索引
    index = 0 if direction == 'horizontal' else 1
    # 定义处理器位置列表
    handle_positions = [press_data[index], release_data[index]]
    # 断言工具的边界处理器位置与处理器位置列表相匹配
    assert tool._edge_handles.positions == handle_positions


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
# 使用pytest的backend装饰器，指定后端为'QtAgg'，并在导入错误时跳过测试
def test_span_selector_animated_artists_callback():
    """Check that the animated artists changed in callbacks are updated."""
    # 生成一个包含 0 到 2π 之间100个均匀间隔的数值序列
    x = np.linspace(0, 2 * np.pi, 100)
    # 计算 x 对应的正弦值
    values = np.sin(x)

    # 创建一个包含单个轴和图形对象的图表
    fig, ax = plt.subplots()
    # 在轴上绘制一条动画线，使用预设的数值 values
    ln, = ax.plot(x, values, animated=True)
    # 创建一个空的动画线对象
    ln2, = ax.plot([], animated=True)

    # 暂停0.1秒，让后端处理任何挂起的操作，然后绘制艺术家对象
    # 参见 blitting 教程
    plt.pause(0.1)
    # 绘制 ln 艺术家对象
    ax.draw_artist(ln)
    # 将整个图表的边界框更新到画布上
    fig.canvas.blit(fig.bbox)

    def mean(vmin, vmax):
        # 返回在 x 中 *vmin* 和 *vmax* 之间数值的平均值
        indmin, indmax = np.searchsorted(x, (vmin, vmax))
        v = values[indmin:indmax].mean()
        # 更新 ln2 的数据为 x 和填充值为 v 的数组
        ln2.set_data(x, np.full_like(x, v))

    # 创建一个水平方向的 SpanSelector 对象
    span = widgets.SpanSelector(ax, mean, direction='horizontal',
                                onmove_callback=mean,
                                interactive=True,
                                drag_from_anywhere=True,
                                useblit=True)

    # 添加 span 选择器，并检查回调函数更新后的线是否绘制
    press_data = [1, 2]
    move_data = [2, 2]
    # 模拟“按下”事件，并传入数据点 press_data
    do_event(span, 'press', xdata=press_data[0], ydata=press_data[1], button=1)
    # 模拟“移动”事件，并传入数据点 move_data
    do_event(span, 'onmove', xdata=move_data[0], ydata=move_data[1], button=1)
    # 断言 span 选择器的动画艺术家是否包括 ln 和 ln2
    assert span._get_animated_artists() == (ln, ln2)
    # 断言 ln 的状态不应为 stale
    assert ln.stale is False
    # 断言 ln2 应为 stale
    assert ln2.stale
    # 检查 ln2 的 y 数据是否接近给定的值
    assert_allclose(ln2.get_ydata(), 0.9547335049088455)
    # 更新 span 选择器
    span.update()
    # 断言 ln2 的状态不应为 stale
    assert ln2.stale is False

    # 改变 span 选择器，并检查其值更新后是否绘制/更新了线
    press_data = [4, 0]
    move_data = [5, 2]
    release_data = [5, 2]
    # 模拟“按下”事件，并传入数据点 press_data
    do_event(span, 'press', xdata=press_data[0], ydata=press_data[1], button=1)
    # 模拟“移动”事件，并传入数据点 move_data
    do_event(span, 'onmove', xdata=move_data[0], ydata=move_data[1], button=1)
    # 断言 ln 的状态不应为 stale
    assert ln.stale is False
    # 断言 ln2 应为 stale
    assert ln2.stale
    # 检查 ln2 的 y 数据是否接近给定的值
    assert_allclose(ln2.get_ydata(), -0.9424150707548072)
    # 模拟“释放”事件，并传入数据点 release_data
    do_event(span, 'release', xdata=release_data[0],
             ydata=release_data[1], button=1)
    # 断言 ln2 的状态不应为 stale
    assert ln2.stale is False
def test_snapping_values_span_selector(ax):
    # 定义一个空函数，用作SpanSelector的回调函数
    def onselect(*args):
        pass

    # 创建一个水平方向的SpanSelector工具，关联回调函数onselect
    tool = widgets.SpanSelector(ax, onselect, direction='horizontal',)
    # 获取SpanSelector对象的_snap方法
    snap_function = tool._snap

    # 定义一组等间隔的值作为“捕捉点”
    snap_values = np.linspace(0, 5, 11)
    # 定义一组待处理的数值
    values = np.array([-0.1, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 4.76, 5.0, 5.5])
    # 定义预期的结果值，这些值将被捕捉到最接近的“捕捉点”
    expect = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 5.00, 5.0, 5.0])
    # 使用_snap方法对values数组中的值进行捕捉
    values = snap_function(values, snap_values)
    # 断言捕捉后的值与预期结果一致
    assert_allclose(values, expect)


def test_span_selector_snap(ax):
    # 定义一个回调函数，用来标记SpanSelector是否触发了选择事件
    def onselect(vmin, vmax):
        ax._got_onselect = True

    # 定义一组“捕捉点”的数值范围
    snap_values = np.arange(50) * 4

    # 创建一个水平方向的SpanSelector工具，关联回调函数onselect，并指定“捕捉点”
    tool = widgets.SpanSelector(ax, onselect, direction='horizontal',
                                snap_values=snap_values)
    # 设置SpanSelector的范围
    tool.extents = (17, 35)
    # 断言SpanSelector的范围已被调整为(16, 36)
    assert tool.extents == (16, 36)

    # 将“捕捉点”设为None
    tool.snap_values = None
    # 断言“捕捉点”已经被清除
    assert tool.snap_values is None
    # 再次设置SpanSelector的范围
    tool.extents = (17, 35)
    # 断言SpanSelector的范围保持不变为(17, 35)
    assert tool.extents == (17, 35)


def test_span_selector_extents(ax):
    # 创建一个水平方向的SpanSelector工具，使用lambda函数作为回调函数
    tool = widgets.SpanSelector(
        ax, lambda a, b: None, "horizontal", ignore_event_outside=True
        )
    # 设置SpanSelector的范围
    tool.extents = (5, 10)

    # 断言SpanSelector的范围已经设置为(5, 10)
    assert tool.extents == (5, 10)
    # 断言选择操作已完成
    assert tool._selection_completed

    # 因为ignore_event_outside=True，此事件应该被忽略
    press_data = (12, 14)
    release_data = (20, 14)
    # 模拟点击并拖动的事件
    click_and_drag(tool, start=press_data, end=release_data)

    # 断言SpanSelector的范围仍然保持不变为(5, 10)
    assert tool.extents == (5, 10)


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(useblit=False, props=dict(color='red')),
    dict(useblit=True, button=1),
])
def test_lasso_selector(ax, kwargs):
    # 创建一个Mock对象作为LassoSelector的回调函数
    onselect = mock.Mock(spec=noop, return_value=None)

    # 创建一个LassoSelector工具，关联Mock对象作为回调函数，并传入参数kwargs
    tool = widgets.LassoSelector(ax, onselect, **kwargs)
    # 模拟按下鼠标、移动鼠标、释放鼠标的事件
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=125, ydata=125, button=1)
    do_event(tool, 'release', xdata=150, ydata=150, button=1)

    # 断言回调函数被正确调用，传入了模拟的坐标数据
    onselect.assert_called_once_with([(100, 100), (125, 125), (150, 150)])


def test_lasso_selector_set_props(ax):
    # 创建一个Mock对象作为LassoSelector的回调函数
    onselect = mock.Mock(spec=noop, return_value=None)

    # 创建一个LassoSelector工具，关联Mock对象作为回调函数，并设置属性为蓝色和透明度为0.2
    tool = widgets.LassoSelector(ax, onselect, props=dict(color='b', alpha=0.2))

    # 获取LassoSelector内部的绘图对象
    artist = tool._selection_artist
    # 断言绘图对象的颜色与设定的一致
    assert mcolors.same_color(artist.get_color(), 'b')
    # 断言绘图对象的透明度与设定的一致
    assert artist.get_alpha() == 0.2
    # 修改绘图对象的属性为红色和透明度为0.3
    tool.set_props(color='r', alpha=0.3)
    # 断言绘图对象的颜色已经修改为红色
    assert mcolors.same_color(artist.get_color(), 'r')
    # 断言绘图对象的透明度已经修改为0.3
    assert artist.get_alpha() == 0.3


def test_lasso_set_props(ax):
    # 创建一个Mock对象作为Lasso的回调函数
    onselect = mock.Mock(spec=noop, return_value=None)
    # 创建一个Lasso对象，关联Mock对象作为回调函数，并设置起始点为(100, 100)
    tool = widgets.Lasso(ax, (100, 100), onselect)
    # 获取Lasso对象内部的线条对象
    line = tool.line
    # 断言线条对象的颜色为黑色
    assert mcolors.same_color(line.get_color(), 'black')
    # 断言线条对象的线型为实线
    assert line.get_linestyle() == '-'
    # 断言线条对象的线宽为2
    assert line.get_lw() == 2
    # 创建一个Lasso对象，关联Mock对象作为回调函数，并设置属性为蓝色、透明度0.2、线宽1、线型为实线
    tool = widgets.Lasso(ax, (100, 100), onselect, props=dict(
        linestyle='-', color='darkblue', alpha=0.2, lw=1))

    # 获取Lasso对象内部的线条对象
    line = tool.line
    # 断言线条对象的颜色为深蓝色
    assert mcolors.same_color(line.get_color(), 'darkblue')
    # 断言线条对象的透明度为0.2
    assert line.get_alpha() == 0.2
    # 断言线条对象的线宽为1
    assert line.get_lw() == 1
    # 断言线条对象的线型为实线
    assert line.get_linestyle() == '-'
    # 将线条颜色修改为红色
    line.set_color('r')
    # 将线条对象的透明度设置为 0.3
    line.set_alpha(0.3)
    # 断言线条的颜色与字符串 'r' 表示的红色相同
    assert mcolors.same_color(line.get_color(), 'r')
    # 断言线条的当前透明度为 0.3
    assert line.get_alpha() == 0.3
def test_CheckButtons(ax):
    # 定义标签列表
    labels = ('a', 'b', 'c')
    # 创建复选框按钮组件，并设置初始状态为 (True, False, True)
    check = widgets.CheckButtons(ax, labels, (True, False, True))
    # 断言初始状态是否正确
    assert check.get_status() == [True, False, True]
    # 设置第一个复选框为非选中状态
    check.set_active(0)
    # 断言状态是否更新正确
    assert check.get_status() == [False, False, True]
    # 断言已选中的标签列表是否正确
    assert check.get_checked_labels() == ['c']
    # 清除所有选择
    check.clear()
    # 断言状态是否完全非选中
    assert check.get_status() == [False, False, False]
    # 断言已选中的标签列表是否为空
    assert check.get_checked_labels() == []

    # 对于无效的索引进行异常断言
    for invalid_index in [-1, len(labels), len(labels)+5]:
        with pytest.raises(ValueError):
            check.set_active(index=invalid_index)

    # 对于无效的状态值进行异常断言
    for invalid_value in ['invalid', -1]:
        with pytest.raises(TypeError):
            check.set_active(1, state=invalid_value)

    # 添加点击事件处理器，并断开连接
    cid = check.on_clicked(lambda: None)
    check.disconnect(cid)


@pytest.mark.parametrize("toolbar", ["none", "toolbar2", "toolmanager"])
def test_TextBox(ax, toolbar):
    # 避免 "toolmanager is provisional" 警告
    plt.rcParams._set("toolbar", toolbar)

    # 创建模拟的提交事件和文本改变事件
    submit_event = mock.Mock(spec=noop, return_value=None)
    text_change_event = mock.Mock(spec=noop, return_value=None)
    # 创建文本框组件，并绑定提交事件和文本改变事件
    tool = widgets.TextBox(ax, '')
    tool.on_submit(submit_event)
    tool.on_text_change(text_change_event)

    # 断言文本框初始文本是否为空
    assert tool.text == ''

    # 模拟点击事件，并设置文本为 'x**2'
    do_event(tool, '_click')

    tool.set_val('x**2')

    # 断言文本框当前文本是否为 'x**2'，并检查文本改变事件调用次数
    assert tool.text == 'x**2'
    assert text_change_event.call_count == 1

    # 模拟开始输入和停止输入事件，并检查提交事件调用次数
    tool.begin_typing()
    tool.stop_typing()

    assert submit_event.call_count == 2

    # 模拟点击事件，确保点击在坐标轴内部
    do_event(tool, '_click', xdata=.5, ydata=.5)
    # 模拟按键事件，输入 '+5'
    do_event(tool, '_keypress', key='+')
    do_event(tool, '_keypress', key='5')

    # 再次检查文本改变事件调用次数
    assert text_change_event.call_count == 3


def test_RadioButtons(ax):
    # 创建单选按钮组件，并设置第二个按钮为活动状态
    radio = widgets.RadioButtons(ax, ('Radio 1', 'Radio 2', 'Radio 3'))
    radio.set_active(1)
    # 断言当前选中的值和索引是否正确
    assert radio.value_selected == 'Radio 2'
    assert radio.index_selected == 1
    # 清除当前选择
    radio.clear()
    # 断言恢复到初始状态
    assert radio.value_selected == 'Radio 1'
    assert radio.index_selected == 0


@image_comparison(['check_radio_buttons.png'], style='mpl20', remove_text=True)
def test_check_radio_buttons_image():
    ax = get_ax()
    fig = ax.figure
    fig.subplots_adjust(left=0.3)

    # 在图形中添加单选框和复选框，并配置其属性
    rax1 = fig.add_axes((0.05, 0.7, 0.2, 0.15))
    rb1 = widgets.RadioButtons(rax1, ('Radio 1', 'Radio 2', 'Radio 3'))

    rax2 = fig.add_axes((0.05, 0.5, 0.2, 0.15))
    cb1 = widgets.CheckButtons(rax2, ('Check 1', 'Check 2', 'Check 3'),
                               (False, True, True))

    rax3 = fig.add_axes((0.05, 0.3, 0.2, 0.15))
    rb3 = widgets.RadioButtons(
        rax3, ('Radio 1', 'Radio 2', 'Radio 3'),
        label_props={'fontsize': [8, 12, 16],
                     'color': ['red', 'green', 'blue']},
        radio_props={'edgecolor': ['red', 'green', 'blue'],
                     'facecolor': ['mistyrose', 'palegreen', 'lightblue']})

    rax4 = fig.add_axes((0.05, 0.1, 0.2, 0.15))
    # 创建一个带有多个复选框的小部件 CheckButtons
    cb4 = widgets.CheckButtons(
        # 在指定的 Axes 对象 rax4 中创建 CheckButtons 小部件
        rax4,
        # 指定复选框的标签文本，分别为 'Check 1', 'Check 2', 'Check 3'
        ('Check 1', 'Check 2', 'Check 3'),
        # 指定每个复选框的初始选中状态，分别为 False, True, True
        (False, True, True),
        # 指定标签的属性，包括不同标签字体大小 [8, 12, 16] 和颜色 ['red', 'green', 'blue']
        label_props={'fontsize': [8, 12, 16],
                     'color': ['red', 'green', 'blue']},
        # 指定框架的属性，包括边框颜色 ['red', 'green', 'blue'] 和背景颜色 ['mistyrose', 'palegreen', 'lightblue']
        frame_props={'edgecolor': ['red', 'green', 'blue'],
                     'facecolor': ['mistyrose', 'palegreen', 'lightblue']},
        # 指定复选框的属性，包括选中时的颜色 ['red', 'green', 'blue']
        check_props={'color': ['red', 'green', 'blue']}
    )
# 用装饰器 @check_figures_equal 标记的测试函数，用于比较两个图形的相等性，生成 PNG 扩展名的图像
@check_figures_equal(extensions=["png"])
def test_radio_buttons(fig_test, fig_ref):
    # 在测试图中创建单选按钮，选项为 ["tea", "coffee"]
    widgets.RadioButtons(fig_test.subplots(), ["tea", "coffee"])
    # 在参考图中添加子图，并设置无刻度线
    ax = fig_ref.add_subplot(xticks=[], yticks=[])
    # 在参考图的子图上绘制散点图，位置为 (0.15, 2/3) 和 (0.15, 1/3)，颜色为 "C0" 和 "none"
    ax.scatter([.15, .15], [2/3, 1/3], transform=ax.transAxes,
               s=(plt.rcParams["font.size"] / 2) ** 2, c=["C0", "none"])
    # 在参考图的子图上添加文本 "tea" 和 "coffee"，位置为 (0.25, 2/3) 和 (0.25, 1/3)
    ax.text(.25, 2/3, "tea", transform=ax.transAxes, va="center")
    ax.text(.25, 1/3, "coffee", transform=ax.transAxes, va="center")


# 用装饰器 @check_figures_equal 标记的测试函数，用于比较两个图形的相等性，生成 PNG 扩展名的图像
@check_figures_equal(extensions=['png'])
def test_radio_buttons_props(fig_test, fig_ref):
    # 定义标签属性和单选按钮属性的字典
    label_props = {'color': ['red'], 'fontsize': [24]}
    radio_props = {'facecolor': 'green', 'edgecolor': 'blue', 'linewidth': 2}

    # 在参考图中创建带有自定义标签属性和单选按钮属性的单选按钮，选项为 ['tea', 'coffee']
    widgets.RadioButtons(fig_ref.subplots(), ['tea', 'coffee'],
                         label_props=label_props, radio_props=radio_props)

    # 在测试图中创建单选按钮，选项为 ['tea', 'coffee']，并设置标签属性
    cb = widgets.RadioButtons(fig_test.subplots(), ['tea', 'coffee'])
    cb.set_label_props(label_props)
    # 设置单选按钮属性，增加默认标记大小
    cb.set_radio_props({**radio_props, 's': (24 / 2)**2})


# 测试函数，用于验证在活动颜色（activecolor）冲突情况下是否会发出警告
def test_radio_button_active_conflict(ax):
    # 使用 pytest 的警告上下文，验证是否会警告关于活动颜色（activecolor）参数冲突的用户警告
    with pytest.warns(UserWarning,
                      match=r'Both the \*activecolor\* parameter'):
        # 在给定的子图上创建单选按钮，选项为 ['tea', 'coffee']，活动颜色为 'red'，单选按钮属性为 {'facecolor': 'green'}
        rb = widgets.RadioButtons(ax, ['tea', 'coffee'], activecolor='red',
                                  radio_props={'facecolor': 'green'})
    # 断言单选按钮的面颜色与预期的颜色列表相同
    assert mcolors.same_color(rb._buttons.get_facecolor(), ['green', 'none'])


# 用装饰器 @check_figures_equal 标记的测试函数，用于比较两个图形的相等性，生成 PNG 扩展名的图像
@check_figures_equal(extensions=['png'])
def test_radio_buttons_activecolor_change(fig_test, fig_ref):
    # 在参考图中创建单选按钮，选项为 ['tea', 'coffee']，活动颜色为 'green'
    widgets.RadioButtons(fig_ref.subplots(), ['tea', 'coffee'],
                         activecolor='green')

    # 在测试图中创建单选按钮，选项为 ['tea', 'coffee']，并设置活动颜色为 'red'
    cb = widgets.RadioButtons(fig_test.subplots(), ['tea', 'coffee'],
                              activecolor='red')
    # 修改单选按钮的活动颜色为 'green'
    cb.activecolor = 'green'


# 用装饰器 @check_figures_equal 标记的测试函数，用于比较两个图形的相等性，生成 PNG 扩展名的图像
@check_figures_equal(extensions=["png"])
def test_check_buttons(fig_test, fig_ref):
    # 在测试图中创建复选框按钮，选项为 ["tea", "coffee"]，初始状态为 [True, True]
    widgets.CheckButtons(fig_test.subplots(), ["tea", "coffee"], [True, True])
    # 在参考图中添加子图，并设置无刻度线
    ax = fig_ref.add_subplot(xticks=[], yticks=[])
    # 在参考图的子图上绘制散点图，使用不同标记符号，位置为 (0.15, 2/3) 和 (0.15, 1/3)，颜色为 "k" 和 "none"
    ax.scatter([.15, .15], [2/3, 1/3], marker='s', transform=ax.transAxes,
               s=(plt.rcParams["font.size"] / 2) ** 2, c=["none", "none"])
    ax.scatter([.15, .15], [2/3, 1/3], marker='x', transform=ax.transAxes,
               s=(plt.rcParams["font.size"] / 2) ** 2, c=["k", "k"])
    # 在参考图的子图上添加文本 "tea" 和 "coffee"，位置为 (0.25, 2/3) 和 (0.25, 1/3)
    ax.text(.25, 2/3, "tea", transform=ax.transAxes, va="center")
    ax.text(.25, 1/3, "coffee", transform=ax.transAxes, va="center")


# 用装饰器 @check_figures_equal 标记的测试函数，用于比较两个图形的相等性，生成 PNG 扩展名的图像
@check_figures_equal(extensions=['png'])
def test_check_button_props(fig_test, fig_ref):
    # 定义标签属性、框架属性和复选框按钮属性的字典
    label_props = {'color': ['red'], 'fontsize': [24]}
    frame_props = {'facecolor': 'green', 'edgecolor': 'blue', 'linewidth': 2}
    check_props = {'facecolor': 'red', 'linewidth': 2}
    # 在给定的图形对象中创建带有多个选择按钮的小部件，选项为 ['tea', 'coffee']，初始选中状态为 [True, True]
    # 使用指定的标签和框架属性，以及检查框属性来配置这些按钮
    widgets.CheckButtons(fig_ref.subplots(), ['tea', 'coffee'], [True, True],
                         label_props=label_props, frame_props=frame_props,
                         check_props=check_props)

    # 在另一个图形对象中创建带有多个选择按钮的小部件，选项为 ['tea', 'coffee']，初始选中状态为 [True, True]
    cb = widgets.CheckButtons(fig_test.subplots(), ['tea', 'coffee'],
                              [True, True])
    # 设置选择按钮的标签属性
    cb.set_label_props(label_props)
    
    # 由于设置标签大小会自动增加默认标记大小，因此我们需要在这里手动调整标记大小
    cb.set_frame_props({**frame_props, 's': (24 / 2)**2})
    
    # FIXME: Axes.scatter 在未填充的标记上将 facecolor 提升为 edgecolor，
    # 但 Collection.update 不会这样做（它已经忘记了标记）。这意味着我们不能直接将 facecolor 传递给两个设置器。
    # 将检查框属性字典中的 'facecolor' 键重命名为 'edgecolor' 键
    check_props['edgecolor'] = check_props.pop('facecolor')
    # 设置选择按钮的检查框属性，并手动调整标记大小
    cb.set_check_props({**check_props, 's': (24 / 2)**2})
def test_slider_slidermin_slidermax_invalid():
    fig, ax = plt.subplots()
    # 测试使用浮点数设置最小/最大值
    with pytest.raises(ValueError):
        widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                       slidermin=10.0)
    with pytest.raises(ValueError):
        widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                       slidermax=10.0)


def test_slider_slidermin_slidermax():
    fig, ax = plt.subplots()
    # 创建一个 Slider 对象，设置初始值为 5.0
    slider_ = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                             valinit=5.0)

    # 创建另一个 Slider 对象，使用前一个 Slider 作为 slidermin，验证其值是否一致
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=1.0, slidermin=slider_)
    assert slider.val == slider_.val

    # 创建另一个 Slider 对象，使用前一个 Slider 作为 slidermax，验证其值是否一致
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=10.0, slidermax=slider_)
    assert slider.val == slider_.val


def test_slider_valmin_valmax():
    fig, ax = plt.subplots()
    # 创建一个 Slider 对象，验证初始值为负数时，其值是否等于 valmin
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=-10.0)
    assert slider.val == slider.valmin

    # 创建一个 Slider 对象，验证初始值超过最大值时，其值是否等于 valmax
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=25.0)
    assert slider.val == slider.valmax


def test_slider_valstep_snapping():
    fig, ax = plt.subplots()
    # 创建一个 Slider 对象，验证设置 valstep 后，初始值会自动吸附到最接近的步长
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=11.4, valstep=1)
    assert slider.val == 11

    # 创建一个 Slider 对象，验证设置 valstep 列表后，初始值会吸附到最接近的步长值
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0,
                            valinit=11.4, valstep=[0, 1, 5.5, 19.7])
    assert slider.val == 5.5


def test_slider_horizontal_vertical():
    fig, ax = plt.subplots()
    # 创建一个水平方向的 Slider 对象，设置初始值为 12
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=24,
                            valinit=12, orientation='horizontal')
    slider.set_val(10)
    assert slider.val == 10
    # 检查 Slider 在坐标轴单位中的尺寸
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.bounds, [0, .25, 10/24, .5])

    fig, ax = plt.subplots()
    # 创建一个垂直方向的 Slider 对象，设置初始值为 12
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=24,
                            valinit=12, orientation='vertical')
    slider.set_val(10)
    assert slider.val == 10
    # 检查 Slider 在坐标轴单位中的尺寸
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.bounds, [.25, 0, .5, 10/24])


def test_slider_reset():
    fig, ax = plt.subplots()
    # 创建一个 Slider 对象，设置初始值为 0.5，然后重置为默认值 0.5
    slider = widgets.Slider(ax=ax, label='', valmin=0, valmax=1, valinit=.5)
    slider.set_val(0.75)
    slider.reset()
    assert slider.val == 0.5


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_range_slider(orientation):
    if orientation == "vertical":
        idx = [1, 0, 3, 2]
    else:
        idx = [0, 1, 2, 3]

    fig, ax = plt.subplots()
    # 创建一个范围滑块部件，用于在指定的坐标轴上显示，设置初始值范围为 [0.1, 0.34]
    slider = widgets.RangeSlider(
        ax=ax, label="", valmin=0.0, valmax=1.0, orientation=orientation,
        valinit=[0.1, 0.34]
    )
    # 获取滑块的边界框，并将其转换为相对于坐标轴的坐标系
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    # 断言滑块指定位置处的值是否等于预期值
    assert_allclose(box.get_points().flatten()[idx], [0.1, 0.25, 0.34, 0.75])

    # 检查初始值是否设置正确
    assert_allclose(slider.val, (0.1, 0.34))

    # 定义处理滑块位置变化的函数
    def handle_positions(slider):
        if orientation == "vertical":
            return [h.get_ydata()[0] for h in slider._handles]
        else:
            return [h.get_xdata()[0] for h in slider._handles]

    # 设置滑块的值范围为 (0.4, 0.6)，并断言设置后的值是否符合预期
    slider.set_val((0.4, 0.6))
    assert_allclose(slider.val, (0.4, 0.6))
    assert_allclose(handle_positions(slider), (0.4, 0.6))

    # 再次获取滑块的边界框，并将其转换为相对于坐标轴的坐标系
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    # 断言滑块指定位置处的值是否等于预期值
    assert_allclose(box.get_points().flatten()[idx], [0.4, .25, 0.6, .75])

    # 设置滑块的值范围为 (0.2, 0.1)，并断言设置后的值是否符合预期
    slider.set_val((0.2, 0.1))
    assert_allclose(slider.val, (0.1, 0.2))
    assert_allclose(handle_positions(slider), (0.1, 0.2))

    # 设置滑块的值范围为 (-1, 10)，滑块不接受无效值，自动设置为最小值和最大值
    slider.set_val((-1, 10))
    assert_allclose(slider.val, (0, 1))
    assert_allclose(handle_positions(slider), (0, 1))

    # 重置滑块的值为初始值 (0.1, 0.34)，并断言重置后的值是否符合预期
    slider.reset()
    assert_allclose(slider.val, (0.1, 0.34))
    assert_allclose(handle_positions(slider), (0.1, 0.34))
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_range_slider_same_init_values 提供多组参数
def test_range_slider_same_init_values(orientation):
    if orientation == "vertical":
        idx = [1, 0, 3, 2]
    else:
        idx = [0, 1, 2, 3]

    fig, ax = plt.subplots()
    # 创建一个新的图形和坐标轴

    slider = widgets.RangeSlider(
         ax=ax, label="", valmin=0.0, valmax=1.0, orientation=orientation,
         valinit=[0, 0]
     )
    # 创建一个范围滑块控件，并将其放置在指定的坐标轴上

    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    # 获取滑块的多边形边界框，并进行坐标变换

    assert_allclose(box.get_points().flatten()[idx], [0, 0.25, 0, 0.75])
    # 使用断言检查滑块的边界框的特定点的坐标是否与预期值接近


def check_polygon_selector(event_sequence, expected_result, selections_count,
                           **kwargs):
    """
    Helper function to test Polygon Selector.

    Parameters
    ----------
    event_sequence : list of tuples (etype, dict())
        A sequence of events to perform. The sequence is a list of tuples
        where the first element of the tuple is an etype (e.g., 'onmove',
        'press', etc.), and the second element of the tuple is a dictionary of
         the arguments for the event (e.g., xdata=5, key='shift', etc.).
    expected_result : list of vertices (xdata, ydata)
        The list of vertices that are expected to result from the event
        sequence.
    selections_count : int
        Wait for the tool to call its `onselect` function `selections_count`
        times, before comparing the result to the `expected_result`
    **kwargs
        Keyword arguments are passed to PolygonSelector.
    """
    ax = get_ax()
    # 获取一个坐标轴对象

    onselect = mock.Mock(spec=noop, return_value=None)
    # 创建一个模拟对象用于替代函数

    tool = widgets.PolygonSelector(ax, onselect, **kwargs)
    # 使用给定的参数创建一个多边形选择器工具对象

    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)
        # 执行事件序列中定义的事件类型和参数

    assert onselect.call_count == selections_count
    # 使用断言检查 onselect 方法被调用的次数是否符合预期
    assert onselect.call_args == ((expected_result, ), {})
    # 使用断言检查 onselect 方法被调用时的参数是否与预期相符


def polygon_place_vertex(xdata, ydata):
    return [('onmove', dict(xdata=xdata, ydata=ydata)),
            ('press', dict(xdata=xdata, ydata=ydata)),
            ('release', dict(xdata=xdata, ydata=ydata))]
    # 返回一个包含放置顶点所需事件序列的列表


def polygon_remove_vertex(xdata, ydata):
    return [('onmove', dict(xdata=xdata, ydata=ydata)),
            ('press', dict(xdata=xdata, ydata=ydata, button=3)),
            ('release', dict(xdata=xdata, ydata=ydata, button=3))]
    # 返回一个包含移除顶点所需事件序列的列表


@pytest.mark.parametrize('draw_bounding_box', [False, True])
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_polygon_selector 提供多组参数
def test_polygon_selector(draw_bounding_box):
    check_selector = functools.partial(
        check_polygon_selector, draw_bounding_box=draw_bounding_box)
    # 创建一个局部函数 check_selector 作为 check_polygon_selector 的偏函数，固定 draw_bounding_box 参数

    # Simple polygon
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [
        *polygon_place_vertex(50, 50),
        *polygon_place_vertex(150, 50),
        *polygon_place_vertex(50, 150),
        *polygon_place_vertex(50, 50),
    ]
    check_selector(event_sequence, expected_result, 1)
    # 使用创建的检查函数测试简单多边形的放置顶点事件序列和预期结果

    # Move first vertex before completing the polygon.
    expected_result = [(75, 50), (150, 50), (50, 150)]
    # 创建事件序列，用于模拟多个操作事件来测试多边形顶点的位置和状态
    event_sequence = [
        *polygon_place_vertex(50, 50),  # 在 (50, 50) 处放置多边形顶点
        *polygon_place_vertex(150, 50),  # 在 (150, 50) 处放置多边形顶点
        ('on_key_press', dict(key='control')),  # 按下控制键
        ('onmove', dict(xdata=50, ydata=50)),  # 移动鼠标到 (50, 50)
        ('press', dict(xdata=50, ydata=50)),  # 按下鼠标在 (50, 50)
        ('onmove', dict(xdata=75, ydata=50)),  # 移动鼠标到 (75, 50)
        ('release', dict(xdata=75, ydata=50)),  # 释放鼠标在 (75, 50)
        ('on_key_release', dict(key='control')),  # 松开控制键
        *polygon_place_vertex(50, 150),  # 在 (50, 150) 处放置多边形顶点
        *polygon_place_vertex(75, 50),  # 在 (75, 50) 处放置多边形顶点
    ]
    check_selector(event_sequence, expected_result, 1)  # 检查事件序列是否符合预期，期望结果是一个多边形

    # 同时移动前两个顶点以完成多边形
    expected_result = [(50, 75), (150, 75), (50, 150)]
    event_sequence = [
        *polygon_place_vertex(50, 50),  # 在 (50, 50) 处放置多边形顶点
        *polygon_place_vertex(150, 50),  # 在 (150, 50) 处放置多边形顶点
        ('on_key_press', dict(key='shift')),  # 按下 Shift 键
        ('onmove', dict(xdata=100, ydata=100)),  # 移动鼠标到 (100, 100)
        ('press', dict(xdata=100, ydata=100)),  # 按下鼠标在 (100, 100)
        ('onmove', dict(xdata=100, ydata=125)),  # 移动鼠标到 (100, 125)
        ('release', dict(xdata=100, ydata=125)),  # 释放鼠标在 (100, 125)
        ('on_key_release', dict(key='shift')),  # 松开 Shift 键
        *polygon_place_vertex(50, 150),  # 在 (50, 150) 处放置多边形顶点
        *polygon_place_vertex(50, 75),  # 在 (50, 75) 处放置多边形顶点
    ]
    check_selector(event_sequence, expected_result, 1)  # 检查事件序列是否符合预期，期望结果是一个多边形

    # 在完成多边形后移动第一个顶点
    expected_result = [(75, 50), (150, 50), (50, 150)]
    event_sequence = [
        *polygon_place_vertex(50, 50),  # 在 (50, 50) 处放置多边形顶点
        *polygon_place_vertex(150, 50),  # 在 (150, 50) 处放置多边形顶点
        *polygon_place_vertex(50, 150),  # 在 (50, 150) 处放置多边形顶点
        *polygon_place_vertex(50, 50),  # 再次在 (50, 50) 处放置多边形顶点，完成多边形
        ('onmove', dict(xdata=50, ydata=50)),  # 移动鼠标到 (50, 50)
        ('press', dict(xdata=50, ydata=50)),  # 按下鼠标在 (50, 50)
        ('onmove', dict(xdata=75, ydata=50)),  # 移动鼠标到 (75, 50)
        ('release', dict(xdata=75, ydata=50)),  # 释放鼠标在 (75, 50)
    ]
    check_selector(event_sequence, expected_result, 2)  # 检查事件序列是否符合预期，期望结果是两个多边形

    # 在完成多边形后移动所有顶点
    expected_result = [(75, 75), (175, 75), (75, 175)]
    event_sequence = [
        *polygon_place_vertex(50, 50),  # 在 (50, 50) 处放置多边形顶点
        *polygon_place_vertex(150, 50),  # 在 (150, 50) 处放置多边形顶点
        *polygon_place_vertex(50, 150),  # 在 (50, 150) 处放置多边形顶点
        *polygon_place_vertex(50, 50),  # 再次在 (50, 50) 处放置多边形顶点，完成多边形
        ('on_key_press', dict(key='shift')),  # 按下 Shift 键
        ('onmove', dict(xdata=100, ydata=100)),  # 移动鼠标到 (100, 100)
        ('press', dict(xdata=100, ydata=100)),  # 按下鼠标在 (100, 100)
        ('onmove', dict(xdata=125, ydata=125)),  # 移动鼠标到 (125, 125)
        ('release', dict(xdata=125, ydata=125)),  # 释放鼠标在 (125, 125)
        ('on_key_release', dict(key='shift')),  # 松开 Shift 键
    ]
    check_selector(event_sequence, expected_result, 2)  # 检查事件序列是否符合预期，期望结果是两个多边形

    # 尝试在放置任何顶点之前移动一个顶点并移动所有顶点
    expected_result = [(50, 50), (150, 50), (50, 150)]
    event_sequence = [
        ('on_key_press', dict(key='control')),  # 模拟按下控制键事件
        ('onmove', dict(xdata=100, ydata=100)),  # 模拟鼠标移动事件到坐标 (100, 100)
        ('press', dict(xdata=100, ydata=100)),  # 模拟鼠标按下事件到坐标 (100, 100)
        ('onmove', dict(xdata=125, ydata=125)),  # 模拟鼠标移动事件到坐标 (125, 125)
        ('release', dict(xdata=125, ydata=125)),  # 模拟鼠标释放事件到坐标 (125, 125)
        ('on_key_release', dict(key='control')),  # 模拟释放控制键事件
        ('on_key_press', dict(key='shift')),  # 模拟按下 Shift 键事件
        ('onmove', dict(xdata=100, ydata=100)),  # 模拟鼠标移动事件到坐标 (100, 100)
        ('press', dict(xdata=100, ydata=100)),  # 模拟鼠标按下事件到坐标 (100, 100)
        ('onmove', dict(xdata=125, ydata=125)),  # 模拟鼠标移动事件到坐标 (125, 125)
        ('release', dict(xdata=125, ydata=125)),  # 模拟鼠标释放事件到坐标 (125, 125)
        ('on_key_release', dict(key='shift')),  # 模拟释放 Shift 键事件
        *polygon_place_vertex(50, 50),  # 添加多边形顶点到 (50, 50)
        *polygon_place_vertex(150, 50),  # 添加多边形顶点到 (150, 50)
        *polygon_place_vertex(50, 150),  # 添加多边形顶点到 (50, 150)
        *polygon_place_vertex(50, 50),  # 添加多边形顶点到 (50, 50)（闭合多边形）
    ]
    check_selector(event_sequence, expected_result, 1)  # 检查事件序列是否符合预期结果，期望结果为 expected_result，参数为 1

    # 尝试将顶点放置在超出边界的位置，然后重置，并开始一个新的多边形。
    expected_result = [(50, 50), (150, 50), (50, 150)]  # 预期的多边形顶点坐标列表
    event_sequence = [
        *polygon_place_vertex(50, 50),  # 添加多边形顶点到 (50, 50)
        *polygon_place_vertex(250, 50),  # 尝试将顶点添加到超出边界的位置 (250, 50)
        ('on_key_press', dict(key='escape')),  # 模拟按下 Escape 键事件
        ('on_key_release', dict(key='escape')),  # 模拟释放 Escape 键事件
        *polygon_place_vertex(50, 50),  # 重置多边形顶点到 (50, 50)
        *polygon_place_vertex(150, 50),  # 添加多边形顶点到 (150, 50)
        *polygon_place_vertex(50, 150),  # 添加多边形顶点到 (50, 150)
        *polygon_place_vertex(50, 50),  # 添加多边形顶点到 (50, 50)（闭合多边形）
    ]
    check_selector(event_sequence, expected_result, 1)  # 检查事件序列是否符合预期结果，期望结果为 expected_result，参数为 1
# 使用 pytest 的 parametrize 装饰器，为 test_polygon_selector_set_props_handle_props 函数参数化两个测试用例
@pytest.mark.parametrize('draw_bounding_box', [False, True])
def test_polygon_selector_set_props_handle_props(ax, draw_bounding_box):
    # 创建 PolygonSelector 对象，绑定到指定的坐标轴 ax 上，设置一些属性和处理器属性，根据 draw_bounding_box 参数决定是否绘制边界框
    tool = widgets.PolygonSelector(ax, onselect=noop,
                                   props=dict(color='b', alpha=0.2),
                                   handle_props=dict(alpha=0.5),
                                   draw_bounding_box=draw_bounding_box)

    # 定义事件序列，模拟用户在坐标轴上绘制多边形的操作
    event_sequence = [
        *polygon_place_vertex(50, 50),   # 在 (50, 50) 处放置顶点
        *polygon_place_vertex(150, 50),  # 在 (150, 50) 处放置顶点
        *polygon_place_vertex(50, 150),  # 在 (50, 150) 处放置顶点
        *polygon_place_vertex(50, 50),   # 再次在 (50, 50) 处放置顶点，完成多边形闭合
    ]

    # 执行每一个事件，模拟用户操作
    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)

    # 获取选择区域的图形对象
    artist = tool._selection_artist
    # 断言选择区域的颜色和透明度
    assert artist.get_color() == 'b'
    assert artist.get_alpha() == 0.2

    # 修改 PolygonSelector 的属性并断言变化
    tool.set_props(color='r', alpha=0.3)
    assert artist.get_color() == 'r'
    assert artist.get_alpha() == 0.3

    # 对每一个处理器对象断言其颜色和透明度
    for artist in tool._handles_artists:
        assert artist.get_color() == 'b'
        assert artist.get_alpha() == 0.5

    # 修改处理器属性并断言变化
    tool.set_handle_props(color='r', alpha=0.3)
    for artist in tool._handles_artists:
        assert artist.get_color() == 'r'
        assert artist.get_alpha() == 0.3


# 使用 check_figures_equal 装饰器，定义 test_rect_visibility 测试函数，验证矩形选择器的可见性设置
@check_figures_equal()
def test_rect_visibility(fig_test, fig_ref):
    # 创建测试和参考图形的坐标轴对象
    ax_test = fig_test.subplots()
    _ = fig_ref.subplots()

    # 创建 RectangleSelector 对象并设置其属性为不可见
    tool = widgets.RectangleSelector(ax_test, onselect=noop,
                                     props={'visible': False})
    tool.extents = (0.2, 0.8, 0.3, 0.7)


# 使用两个 parametrize 装饰器为 test_polygon_selector_remove 函数参数化测试用例
@pytest.mark.parametrize('idx', [1, 2, 3])
@pytest.mark.parametrize('draw_bounding_box', [False, True])
def test_polygon_selector_remove(idx, draw_bounding_box):
    # 定义多边形的顶点列表
    verts = [(50, 50), (150, 50), (50, 150)]
    # 定义事件序列，模拟用户在坐标轴上绘制多边形的操作，并根据 idx 插入和移除额外的顶点
    event_sequence = [polygon_place_vertex(*verts[0]),
                      polygon_place_vertex(*verts[1]),
                      polygon_place_vertex(*verts[2]),
                      # 完成多边形
                      polygon_place_vertex(*verts[0])]
    # 在指定位置插入额外的顶点
    event_sequence.insert(idx, polygon_place_vertex(200, 200))
    # 移除额外的顶点
    event_sequence.append(polygon_remove_vertex(200, 200))
    # 展开事件序列的列表
    event_sequence = sum(event_sequence, [])
    # 调用 check_polygon_selector 函数验证多边形选择器的行为
    check_polygon_selector(event_sequence, verts, 2,
                           draw_bounding_box=draw_bounding_box)


# 使用 parametrize 装饰器为 test_polygon_selector_remove_first_point 函数参数化测试用例
@pytest.mark.parametrize('draw_bounding_box', [False, True])
def test_polygon_selector_remove_first_point(draw_bounding_box):
    # 定义多边形的顶点列表
    verts = [(50, 50), (150, 50), (50, 150)]
    # 定义事件序列，模拟用户在坐标轴上绘制多边形的操作，并移除第一个顶点
    event_sequence = [
        *polygon_place_vertex(*verts[0]),
        *polygon_place_vertex(*verts[1]),
        *polygon_place_vertex(*verts[2]),
        *polygon_place_vertex(*verts[0]),
        *polygon_remove_vertex(*verts[0]),
    ]
    # 调用 check_polygon_selector 函数验证多边形选择器的行为
    check_polygon_selector(event_sequence, verts[1:], 2,
                           draw_bounding_box=draw_bounding_box)
@pytest.mark.parametrize('draw_bounding_box', [False, True])
# 使用 pytest 的参数化功能，分别测试是否绘制边界框的情况
def test_polygon_selector_redraw(ax, draw_bounding_box):
    # 定义多边形顶点的坐标
    verts = [(50, 50), (150, 50), (50, 150)]
    # 定义事件序列，模拟在顶点处放置顶点和移除顶点的操作
    event_sequence = [
        *polygon_place_vertex(*verts[0]),  # 放置第一个顶点
        *polygon_place_vertex(*verts[1]),  # 放置第二个顶点
        *polygon_place_vertex(*verts[2]),  # 放置第三个顶点
        *polygon_place_vertex(*verts[0]),  # 回到第一个顶点，完成多边形闭合
        # 多边形完成，移除前两个顶点
        *polygon_remove_vertex(*verts[1]),
        *polygon_remove_vertex(*verts[2]),
        # 此时工具应重置，可以再次添加顶点
        *polygon_place_vertex(*verts[1]),
    ]

    # 创建多边形选择器工具
    tool = widgets.PolygonSelector(ax, onselect=noop,
                                   draw_bounding_box=draw_bounding_box)
    # 模拟事件序列
    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)
    
    # 断言验证：移除两个顶点后，应该只剩一个顶点，选择器应自动重置
    assert tool.verts == verts[0:2]


@pytest.mark.parametrize('draw_bounding_box', [False, True])
@check_figures_equal(extensions=['png'])
# 使用自定义的装饰器 check_figures_equal，比较两个图形对象是否相等（包括图像文件扩展名为 png）
def test_polygon_selector_verts_setter(fig_test, fig_ref, draw_bounding_box):
    # 定义多边形顶点的坐标
    verts = [(0.1, 0.4), (0.5, 0.9), (0.3, 0.2)]
    # 在测试图上添加子图
    ax_test = fig_test.add_subplot()

    # 创建多边形选择器工具并设置顶点
    tool_test = widgets.PolygonSelector(
        ax_test, onselect=noop, draw_bounding_box=draw_bounding_box)
    tool_test.verts = verts
    # 断言验证：设置后的顶点应与定义的顶点相等
    assert tool_test.verts == verts

    # 在参考图上添加子图
    ax_ref = fig_ref.add_subplot()
    # 创建多边形选择器工具
    tool_ref = widgets.PolygonSelector(
        ax_ref, onselect=noop, draw_bounding_box=draw_bounding_box)
    # 模拟事件序列，放置多边形顶点
    event_sequence = [
        *polygon_place_vertex(*verts[0]),
        *polygon_place_vertex(*verts[1]),
        *polygon_place_vertex(*verts[2]),
        *polygon_place_vertex(*verts[0]),
    ]
    for (etype, event_args) in event_sequence:
        do_event(tool_ref, etype, **event_args)


def test_polygon_selector_box(ax):
    # 设置坐标轴限制，创建一个菱形，使其位于轴限制内
    ax.set(xlim=(-10, 50), ylim=(-10, 50))
    # 定义菱形顶点的坐标
    verts = [(20, 0), (0, 20), (20, 40), (40, 20)]
    # 定义事件序列，模拟放置菱形顶点的操作
    event_sequence = [
        *polygon_place_vertex(*verts[0]),
        *polygon_place_vertex(*verts[1]),
        *polygon_place_vertex(*verts[2]),
        *polygon_place_vertex(*verts[3]),
        *polygon_place_vertex(*verts[0]),
    ]

    # 创建多边形选择器工具，绘制边界框
    tool = widgets.PolygonSelector(ax, onselect=noop, draw_bounding_box=True)
    # 模拟事件序列
    for (etype, event_args) in event_sequence:
        do_event(tool, etype, **event_args)

    # 为了触发正确的回调，通过触发画布上的事件来触发回调，而不是直接操作工具
    t = ax.transData
    canvas = ax.figure.canvas

    # 缩小到一半尺寸，使用边界框的右上角
    MouseEvent(
        "button_press_event", canvas, *t.transform((40, 40)), 1)._process()
    MouseEvent(
        "motion_notify_event", canvas, *t.transform((20, 20)))._process()
    # 触发一个鼠标事件，模拟按钮释放在画布上的坐标 (20, 20)，使用转换后的坐标，按钮索引为 1
    MouseEvent(
        "button_release_event", canvas, *t.transform((20, 20)), 1)._process()

    # 使用 NumPy 测试工具断言，验证工具的顶点坐标是否接近于 [(10, 0), (0, 10), (10, 20), (20, 10)]
    np.testing.assert_allclose(
        tool.verts, [(10, 0), (0, 10), (10, 20), (20, 10)])

    # 使用鼠标事件模拟按钮按下在画布上的坐标 (10, 10)，使用转换后的坐标，按钮索引为 1
    MouseEvent(
        "button_press_event", canvas, *t.transform((10, 10)), 1)._process()

    # 使用鼠标事件模拟鼠标移动事件在画布上的坐标 (30, 30)，使用转换后的坐标
    MouseEvent(
        "motion_notify_event", canvas, *t.transform((30, 30)))._process()

    # 使用鼠标事件模拟按钮释放在画布上的坐标 (30, 30)，使用转换后的坐标，按钮索引为 1
    MouseEvent(
        "button_release_event", canvas, *t.transform((30, 30)), 1)._process()

    # 使用 NumPy 测试工具断言，验证工具的顶点坐标是否接近于 [(30, 20), (20, 30), (30, 40), (40, 30)]
    np.testing.assert_allclose(
        tool.verts, [(30, 20), (20, 30), (30, 40), (40, 30)])

    # 移除多边形中的一个点，并验证包围框的范围是否更新为 (20.0, 40.0, 20.0, 40.0)
    np.testing.assert_allclose(
        tool._box.extents, (20.0, 40.0, 20.0, 40.0))

    # 使用鼠标事件模拟按钮按下在画布上的坐标 (30, 20)，使用转换后的坐标，按钮索引为 3
    MouseEvent(
        "button_press_event", canvas, *t.transform((30, 20)), 3)._process()

    # 使用鼠标事件模拟按钮释放在画布上的坐标 (30, 20)，使用转换后的坐标，按钮索引为 3
    MouseEvent(
        "button_release_event", canvas, *t.transform((30, 20)), 3)._process()

    # 使用 NumPy 测试工具断言，验证工具的顶点坐标是否接近于 [(20, 30), (30, 40), (40, 30)]
    np.testing.assert_allclose(
        tool.verts, [(20, 30), (30, 40), (40, 30)])

    # 使用 NumPy 测试工具断言，验证包围框的范围是否更新为 (20.0, 40.0, 30.0, 40.0)
    np.testing.assert_allclose(
        tool._box.extents, (20.0, 40.0, 30.0, 40.0))
# 定义一个测试函数，用于测试多边形选择器的清除方法
def test_polygon_selector_clear_method(ax):
    # 创建一个模拟对象，代表一个空操作函数，并设定其返回值为 None
    onselect = mock.Mock(spec=noop, return_value=None)
    # 使用给定的轴对象和模拟的空操作函数创建一个多边形选择器工具
    tool = widgets.PolygonSelector(ax, onselect)

    # 针对每一个多边形顶点坐标组合进行测试
    for result in ([(50, 50), (150, 50), (50, 150), (50, 50)],
                   [(50, 50), (100, 50), (50, 150), (50, 50)]):
        # 遍历每个顶点坐标并调用 polygon_place_vertex 函数生成事件类型和参数
        for x, y in result:
            for etype, event_args in polygon_place_vertex(x, y):
                # 使用 do_event 函数触发工具的事件处理
                do_event(tool, etype, **event_args)

        # 获取选择器工具的选择艺术家对象
        artist = tool._selection_artist

        # 断言选择已完成、工具可见并且艺术家可见
        assert tool._selection_completed
        assert tool.get_visible()
        assert artist.get_visible()
        # 使用 np.testing.assert_equal 断言艺术家的顶点数据与预期结果相等
        np.testing.assert_equal(artist.get_xydata(), result)
        # 断言 onselect 函数被调用，并且传入正确的参数
        assert onselect.call_args == ((result[:-1],), {})

        # 清除选择器工具的状态
        tool.clear()
        # 断言选择未完成，并且艺术家的顶点数据被重置为 [(0, 0)]
        assert not tool._selection_completed
        np.testing.assert_equal(artist.get_xydata(), [(0, 0)])


# 使用 pytest 的参数化功能，定义一个测试函数，测试 MultiCursor 类
@pytest.mark.parametrize("horizOn", [False, True])
@pytest.mark.parametrize("vertOn", [False, True])
def test_MultiCursor(horizOn, vertOn):
    # 创建包含两个子图的图形对象，并分别获取每个子图的轴对象
    (ax1, ax3) = plt.figure().subplots(2, sharex=True)
    ax2 = plt.figure().subplots()

    # 创建 MultiCursor 对象，将其应用于指定的轴列表，并设定 useblit=False
    multi = widgets.MultiCursor(
        None, (ax1, ax2), useblit=False, horizOn=horizOn, vertOn=vertOn
    )

    # 断言只有两个轴对象上有绘制的垂直和水平线
    assert len(multi.vlines) == 2
    assert len(multi.hlines) == 2

    # 模拟一个 motion_notify_event 事件
    event = mock_event(ax1, xdata=.5, ydata=.25)
    # 调用 MultiCursor 对象的 onmove 方法处理事件
    multi.onmove(event)
    # 手动触发画布的绘制事件，以测试清除功能
    ax1.figure.canvas.draw()

    # 断言前两个轴上的垂直线都移动到了新的位置
    for l in multi.vlines:
        assert l.get_xdata() == (.5, .5)
    # 断言前两个轴上的水平线都移动到了新的位置
    for l in multi.hlines:
        assert l.get_ydata() == (.25, .25)
    # 断言根据 horizOn 和 vertOn 的设置，垂直和水平线应该相应地可见
    assert len([line for line in multi.vlines if line.get_visible()]) == (
        2 if vertOn else 0)
    assert len([line for line in multi.hlines if line.get_visible()]) == (
        2 if horizOn else 0)

    # 切换设置后，再次测试，应该看到相反设置下的垂直和水平线可见
    multi.horizOn = not multi.horizOn
    multi.vertOn = not multi.vertOn
    event = mock_event(ax1, xdata=.5, ydata=.25)
    multi.onmove(event)
    assert len([line for line in multi.vlines if line.get_visible()]) == (
        0 if vertOn else 2)
    assert len([line for line in multi.hlines if line.get_visible()]) == (
        0 if horizOn else 2)

    # 测试在不属于 MultiCursor 的轴上触发移动事件，应该没有任何线条移动
    event = mock_event(ax3, xdata=.75, ydata=.75)
    multi.onmove(event)
    for l in multi.vlines:
        assert l.get_xdata() == (.5, .5)
    for l in multi.hlines:
        assert l.get_ydata() == (.25, .25)
```