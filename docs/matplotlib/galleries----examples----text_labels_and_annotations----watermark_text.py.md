# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\watermark_text.py`

```
"""
==============
Text watermark
==============

A watermark effect can be achieved by drawing a semi-transparent text.
"""
# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库

# 设置随机种子以便结果可重现
np.random.seed(19680801)

# 创建一个图像和一个坐标轴
fig, ax = plt.subplots()

# 绘制随机数据的折线图，并设定样式
ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')

# 添加网格线到图中
ax.grid()

# 在图的指定位置添加水印文本
ax.text(0.5, 0.5, 'created with matplotlib', transform=ax.transAxes,
        fontsize=40, color='gray', alpha=0.5,
        ha='center', va='center', rotation=30)

# 显示绘制的图像
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.text`
```