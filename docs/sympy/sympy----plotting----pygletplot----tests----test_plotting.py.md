# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\tests\test_plotting.py`

```
# 导入模块 import_module 用于动态导入指定名称的模块，如果导入失败则捕获 OSError 异常
from sympy.external.importtools import import_module

# 初始化一个变量 disabled，用于标记测试是否被禁用，默认为 False
disabled = False

# 尝试导入 pyglet.gl 模块，如果失败（如缺少 opengl），则禁用测试
pyglet_gl = import_module("pyglet.gl", catch=(OSError,))
# 尝试导入 pyglet.window 模块，如果失败（如缺少 opengl），则禁用测试
pyglet_window = import_module("pyglet.window", catch=(OSError,))
# 如果 pyglet_gl 或者 pyglet_window 任何一个导入失败，则将 disabled 设置为 True
if not pyglet_gl or not pyglet_window:
    disabled = True

# 导入符号类 symbols 从 sympy.core.symbol 模块
# 导入对数函数 log 从 sympy.functions.elementary.exponential 模块
# 导入三角函数 cos 和 sin 从 sympy.functions.elementary.trigonometric 模块
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)

# 创建符号变量 x, y, z
x, y, z = symbols('x, y, z')

# 定义测试函数 test_plot_2d，用于绘制二维图形
def test_plot_2d():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 x 在区间 [-5, 5]，步长为 4 的二维图形，但不可见
    p = PygletPlot(x, [x, -5, 5, 4], visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_2d_discontinuous，用于绘制具有间断点的二维图形
def test_plot_2d_discontinuous():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 1/x 在区间 [-1, 1]，步长为 2 的二维图形，但不可见
    p = PygletPlot(1/x, [x, -1, 1, 2], visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_3d，用于绘制三维图形
def test_plot_3d():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 x*y 在区间 [-5, 5]，步长为 5 的三维图形，但不可见
    p = PygletPlot(x*y, [x, -5, 5, 5], [y, -5, 5, 5], visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_3d_discontinuous，用于绘制具有间断点的三维图形
def test_plot_3d_discontinuous():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 1/x 在区间 [-3, 3]，步长为 6 的三维图形，但不可见
    p = PygletPlot(1/x, [x, -3, 3, 6], [y, -1, 1, 1], visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_2d_polar，用于绘制极坐标系下的二维图形
def test_plot_2d_polar():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 1/x 在极坐标模式下，区间 [-1, 1]，步长为 4 的二维图形，但不可见
    p = PygletPlot(1/x, [x, -1, 1, 4], 'mode=polar', visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_3d_cylinder，用于绘制圆柱坐标系下的三维图形
def test_plot_3d_cylinder():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 1/y 在圆柱坐标模式下，区间 [0, 6.282]，步长为 4 的三维图形，但不可见
    p = PygletPlot(
        1/y, [x, 0, 6.282, 4], [y, -1, 1, 4], 'mode=polar;style=solid',
        visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_3d_spherical，用于绘制球坐标系下的三维图形
def test_plot_3d_spherical():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制函数 1 在球坐标模式下，区间 [0, 6.282]，[0, 3.141]，步长为 4 的三维图形，但不可见
    p = PygletPlot(
        1, [x, 0, 6.282, 4], [y, 0, 3.141,
            4], 'mode=spherical;style=wireframe',
        visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_2d_parametric，用于绘制参数方程下的二维图形
def test_plot_2d_parametric():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制参数方程 sin(x), cos(x) 在区间 [0, 6.282]，步长为 4 的二维图形，但不可见
    p = PygletPlot(sin(x), cos(x), [x, 0, 6.282, 4], visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 test_plot_3d_parametric，用于绘制参数方程下的三维图形
def test_plot_3d_parametric():
    # 导入 PygletPlot 类从 sympy.plotting.pygletplot 模块
    from sympy.plotting.pygletplot import PygletPlot
    # 创建 PygletPlot 对象 p，绘制参数方程 sin(x), cos(x), x/5.0 在区间 [0, 6.282]，步长为 4 的三维图形，但不可见
    p = PygletPlot(sin(x), cos(x), x/5.0, [x, 0, 6.282, 4], visible=False)
    # 等待图形计算完成
    p.wait_for_calculations()

# 定义测试函数 _test_plot_log，用于绘制对数函数的图形
def _test_plot_log
```