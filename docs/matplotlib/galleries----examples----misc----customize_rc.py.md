# `D:\src\scipysrc\matplotlib\galleries\examples\misc\customize_rc.py`

```py
"""
============
Customize Rc
============

I'm not trying to make a good-looking figure here, but just to show
some examples of customizing `.rcParams` on the fly.

If you like to work interactively, and need to create different sets
of defaults for figures (e.g., one set of defaults for publication, one
set for interactive exploration), you may want to define some
functions in a custom module that set the defaults, e.g.,::

    def set_pub():
        rcParams.update({
            "font.weight": "bold",  # 加粗字体
            "tick.labelsize": 15,   # 较大的刻度标签
            "lines.linewidth": 1,   # 较粗的线条
            "lines.color": "k",     # 黑色线条
            "grid.color": "0.5",    # 灰色网格线
            "grid.linestyle": "-",  # 实线网格线
            "grid.linewidth": 0.5,  # 细网格线
            "savefig.dpi": 300,     # 更高分辨率的输出
        })

Then as you are working interactively, you just need to do::

    >>> set_pub()
    >>> plot([1, 2, 3])
    >>> savefig('myfig')
    >>> rcdefaults()  # 恢复默认设置
"""

import matplotlib.pyplot as plt

# 在调用 subplot 之前需要设置轴的属性
plt.subplot(311)
plt.plot([1, 2, 3])

# 更新全局的 rcParams 来自定义绘图参数
plt.rcParams.update({
    "font.weight": "bold",         # 设置字体加粗
    "xtick.major.size": 5,         # X轴主刻度大小
    "xtick.major.pad": 7,          # X轴主刻度标签与轴线的间距
    "xtick.labelsize": 15,         # X轴刻度标签大小
    "grid.color": "0.5",           # 灰色网格线
    "grid.linestyle": "-",         # 实线网格线
    "grid.linewidth": 5,           # 网格线宽度
    "lines.linewidth": 2,          # 线条宽度
    "lines.color": "g",            # 绿色线条
})

plt.subplot(312)
plt.plot([1, 2, 3])
plt.grid(True)  # 打开网格显示

plt.rcdefaults()  # 恢复默认的 rcParams 设置
plt.subplot(313)
plt.plot([1, 2, 3])
plt.grid(True)  # 打开网格显示
plt.show()  # 显示绘图结果
```