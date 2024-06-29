# `D:\src\scipysrc\matplotlib\galleries\examples\widgets\rectangle_selector.py`

```
"""
===============================
Rectangle and ellipse selectors
===============================

Click somewhere, move the mouse, and release the mouse button.
`.RectangleSelector` and `.EllipseSelector` draw a rectangle or an ellipse
from the initial click position to the current mouse position (within the same
axes) until the button is released.  A connected callback receives the click-
and release-events.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy数值计算库
from matplotlib.widgets import EllipseSelector, RectangleSelector  # 导入椭圆和矩形选择器

# 定义回调函数，用于处理选择事件
def select_callback(eclick, erelease):
    """
    Callback for line selection.

    *eclick* and *erelease* are the press and release events.
    """
    x1, y1 = eclick.xdata, eclick.ydata  # 获取点击事件的坐标
    x2, y2 = erelease.xdata, erelease.ydata  # 获取释放事件的坐标
    print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")  # 打印选择的起始和结束坐标
    print(f"The buttons you used were: {eclick.button} {erelease.button}")  # 打印所使用的鼠标按钮信息


# 定义按键切换选择器状态的函数
def toggle_selector(event):
    print('Key pressed.')
    if event.key == 't':
        for selector in selectors:
            name = type(selector).__name__  # 获取选择器的类型名称
            if selector.active:
                print(f'{name} deactivated.')  # 如果选择器处于激活状态，则停用它
                selector.set_active(False)
            else:
                print(f'{name} activated.')  # 如果选择器处于未激活状态，则激活它
                selector.set_active(True)


# 创建绘图窗口和子图
fig = plt.figure(layout='constrained')
axs = fig.subplots(2)

N = 100000  # 如果N很大，使用blitting可以看到性能提升
x = np.linspace(0, 10, N)

selectors = []
# 在每个子图上绘制图形并创建选择器
for ax, selector_class in zip(axs, [RectangleSelector, EllipseSelector]):
    ax.plot(x, np.sin(2*np.pi*x))  # 绘制一些图形
    ax.set_title(f"Click and drag to draw a {selector_class.__name__}.")  # 设置子图标题，指示如何使用选择器
    selectors.append(selector_class(
        ax, select_callback,
        useblit=True,  # 使用blitting以提高性能
        button=[1, 3],  # 禁用中间鼠标按钮
        minspanx=5, minspany=5,  # 设置最小尺寸限制
        spancoords='pixels',  # 尺寸限制以像素为单位
        interactive=True))  # 启用交互模式
    fig.canvas.mpl_connect('key_press_event', toggle_selector)  # 连接按键事件到切换选择器状态的函数
axs[0].set_title("Press 't' to toggle the selectors on and off.\n"
                 + axs[0].get_title())  # 设置主标题，说明按键切换选择器的功能
plt.show()  # 显示绘图窗口

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.RectangleSelector`
#    - `matplotlib.widgets.EllipseSelector`
```