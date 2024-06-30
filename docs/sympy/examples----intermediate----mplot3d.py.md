# `D:\src\scipysrc\sympy\examples\intermediate\mplot3d.py`

```
#!/usr/bin/env python

"""Matplotlib 3D plotting example

Demonstrates plotting with matplotlib.
"""

import sys  # 导入 sys 模块，用于处理系统相关的功能

from sample import sample  # 从 sample 模块中导入 sample 函数

from sympy import Symbol  # 从 sympy 模块中导入 Symbol 符号类
from sympy.external import import_module  # 从 sympy.external 模块导入 import_module 函数


def mplot3d(f, var1, var2, *, show=True):
    """
    Plot a 3d function using matplotlib/Tk.
    """

    import warnings  # 导入 warnings 模块，用于控制警告信息的显示
    warnings.filterwarnings("ignore", r"Could not match \S")  # 忽略特定类型的警告信息

    p = import_module('pylab')  # 尝试导入 pylab 模块
    # Try newer version first
    p3 = import_module('mpl_toolkits.mplot3d',
        import_kwargs={'fromlist': ['something']}) or import_module('matplotlib.axes3d')  # 尝试导入新版的 mpl_toolkits.mplot3d 或者 matplotlib.axes3d
    if not p or not p3:  # 如果导入失败
        sys.exit("Matplotlib is required to use mplot3d.")  # 输出错误信息并退出程序

    x, y, z = sample(f, var1, var2)  # 调用 sample 函数，获取函数 f 在 (var1, var2) 区间上的采样点 x, y, z

    fig = p.figure()  # 创建一个新的图形窗口
    ax = p3.Axes3D(fig)  # 在图形窗口上创建一个 3D 坐标轴对象

    # ax.plot_surface(x, y, z, rstride=2, cstride=2)
    ax.plot_wireframe(x, y, z)  # 在 3D 坐标轴上绘制函数曲线的网格线

    ax.set_xlabel('X')  # 设置 X 轴标签
    ax.set_ylabel('Y')  # 设置 Y 轴标签
    ax.set_zlabel('Z')  # 设置 Z 轴标签

    if show:  # 如果 show 参数为 True
        p.show()  # 显示绘制的图形


def main():
    x = Symbol('x')  # 创建符号 x
    y = Symbol('y')  # 创建符号 y

    mplot3d(x**2 - y**2, (x, -10.0, 10.0, 20), (y, -10.0, 10.0, 20))
    # mplot3d(x**2+y**2, (x, -10.0, 10.0, 20), (y, -10.0, 10.0, 20))
    # mplot3d(sin(x)+sin(y), (x, -3.14, 3.14, 10), (y, -3.14, 3.14, 10))

if __name__ == "__main__":
    main()
```