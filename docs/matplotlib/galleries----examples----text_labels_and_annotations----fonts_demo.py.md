# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\fonts_demo.py`

```py
`
"""
==================================
Fonts demo (object-oriented style)
==================================

Set font properties using setters.

See :doc:`fonts_demo_kw` to achieve the same effect using keyword arguments.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

# 导入字体管理器中的字体属性类
from matplotlib.font_manager import FontProperties

# 创建一个新的图形对象
fig = plt.figure()

# 水平和垂直对齐方式的字典
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}

# 列表，表示各个位置的垂直坐标
yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# 定义标题字体属性
heading_font = FontProperties(size='large')

# 显示字体家族选项
fig.text(0.1, 0.9, 'family', fontproperties=heading_font, **alignment)
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
for k, family in enumerate(families):
    font = FontProperties()  # 创建字体属性对象
    font.set_family(family)  # 设置字体家族
    fig.text(0.1, yp[k], family, fontproperties=font, **alignment)

# 显示字体样式选项
styles = ['normal', 'italic', 'oblique']
fig.text(0.3, 0.9, 'style', fontproperties=heading_font, **alignment)
for k, style in enumerate(styles):
    font = FontProperties()
    font.set_family('sans-serif')
    font.set_style(style)  # 设置字体样式
    fig.text(0.3, yp[k], style, fontproperties=font, **alignment)

# 显示字体变体选项
variants = ['normal', 'small-caps']
fig.text(0.5, 0.9, 'variant', fontproperties=heading_font, **alignment)
for k, variant in enumerate(variants):
    font = FontProperties()
    font.set_family('serif')
    font.set_variant(variant)  # 设置字体变体
    fig.text(0.5, yp[k], variant, fontproperties=font, **alignment)

# 显示字体粗细选项
weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
fig.text(0.7, 0.9, 'weight', fontproperties=heading_font, **alignment)
for k, weight in enumerate(weights):
    font = FontProperties()
    font.set_weight(weight)  # 设置字体粗细
    fig.text(0.7, yp[k], weight, fontproperties=font, **alignment)

# 显示字体大小选项
sizes = [
    'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
fig.text(0.9, 0.9, 'size', fontproperties=heading_font, **alignment)
for k, size in enumerate(sizes):
    font = FontProperties()
    font.set_size(size)  # 设置字体大小
    fig.text(0.9, yp[k], size, fontproperties=font, **alignment)

# 显示粗斜体选项
font = FontProperties(style='italic', weight='bold', size='x-small')
fig.text(0.3, 0.1, 'bold italic', fontproperties=font, **alignment)
font = FontProperties(style='italic', weight='bold', size='medium')
fig.text(0.3, 0.2, 'bold italic', fontproperties=font, **alignment)
font = FontProperties(style='italic', weight='bold', size='x-large')
fig.text(0.3, 0.3, 'bold italic', fontproperties=font, **alignment)

# 显示图形
plt.show()
```