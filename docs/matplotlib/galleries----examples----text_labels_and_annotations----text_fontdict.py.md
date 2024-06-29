# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\text_fontdict.py`

```py
"""
=======================================================
Controlling style of text and labels using a dictionary
=======================================================

This example shows how to share parameters across many text objects and labels
by creating a dictionary of options passed across several functions.
"""

# 导入 matplotlib 的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并简写为 np
import numpy as np

# 定义字体样式的字典
font = {'family': 'serif',     # 字体系列为 serif
        'color':  'darkred',   # 字体颜色为深红色
        'weight': 'normal',    # 字体粗细为正常
        'size': 16,            # 字体大小为 16
        }

# 生成一组 x 值，范围从 0 到 5，总共 100 个点
x = np.linspace(0.0, 5.0, 100)
# 根据 x 计算对应的 y 值，表示为 cos(2πx) * e^(-x)
y = np.cos(2*np.pi*x) * np.exp(-x)

# 绘制 x 和 y 的图形，线条颜色为黑色
plt.plot(x, y, 'k')

# 设置图形的标题，并应用上面定义的字体样式
plt.title('Damped exponential decay', fontdict=font)

# 在图中指定位置添加文本，使用 LaTeX 渲染公式，应用定义的字体样式
plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)

# 设置 x 轴的标签，并应用定义的字体样式
plt.xlabel('time (s)', fontdict=font)

# 设置 y 轴的标签，并应用定义的字体样式
plt.ylabel('voltage (mV)', fontdict=font)

# 调整子图的布局，防止 y 轴标签被裁剪
plt.subplots_adjust(left=0.15)

# 显示绘制的图形
plt.show()
```