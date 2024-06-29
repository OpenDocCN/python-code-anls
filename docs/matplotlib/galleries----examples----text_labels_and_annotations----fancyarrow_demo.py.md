# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\fancyarrow_demo.py`

```
"""
================================
Annotation arrow style reference
================================

Overview of the arrow styles available in `~.Axes.annotate`.
"""

# 导入需要的模块
import inspect  # 用于检查对象的属性和方法
import itertools  # 用于创建迭代器的函数
import re  # 用于处理正则表达式的模块

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

import matplotlib.patches as mpatches  # 导入 matplotlib 的 patches 模块，并用 mpatches 别名

# 获取所有箭头样式
styles = mpatches.ArrowStyle.get_styles()

# 设置网格的列数和行数
ncol = 2
nrow = (len(styles) + 1) // ncol

# 创建一个 Figure 对象，并设置其布局
axs = (plt.figure(figsize=(4 * ncol, 1 + nrow))
       .add_gridspec(1 + nrow, ncol,
                     wspace=.7, left=.1, right=.9, bottom=0, top=1).subplots())

# 遍历所有的子图，并设置它们的属性
for ax in axs.flat:
    ax.set_axis_off()

# 设置第一行子图的文本内容
for ax in axs[0, :]:
    ax.text(0, .5, "arrowstyle",
            transform=ax.transAxes, size="large", color="tab:blue",
            horizontalalignment="center", verticalalignment="center")
    ax.text(.35, .5, "default parameters",
            transform=ax.transAxes,
            horizontalalignment="left", verticalalignment="center")

# 遍历箭头样式，并在子图中进行标注
for ax, (stylename, stylecls) in zip(axs[1:, :].T.flat, styles.items()):
    # 在子图中绘制一个点，并注释箭头样式名称
    l, = ax.plot(.25, .5, "ok", transform=ax.transAxes)
    ax.annotate(stylename, (.25, .5), (-0.1, .5),
                xycoords="axes fraction", textcoords="axes fraction",
                size="large", color="tab:blue",
                horizontalalignment="center", verticalalignment="center",
                arrowprops=dict(
                    arrowstyle=stylename, connectionstyle="arc3,rad=-0.05",
                    color="k", shrinkA=5, shrinkB=5, patchB=l,
                ),
                bbox=dict(boxstyle="square", fc="w"))
    
    # 格式化箭头样式类的签名信息，并在子图中显示
    s = str(inspect.signature(stylecls))[1:-1]
    n = 2 if s.count(',') > 3 else 1
    ax.text(.35, .5,
            re.sub(', ', lambda m, c=itertools.count(1): m.group()
                   if next(c) % n else '\n', s),
            transform=ax.transAxes,
            horizontalalignment="left", verticalalignment="center")

plt.show()
```