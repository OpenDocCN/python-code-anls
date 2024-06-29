# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\fonts_demo_kw.py`

```py
"""
==============================
Fonts demo (keyword arguments)
==============================

Set font properties using keyword arguments.

See :doc:`fonts_demo` to achieve the same effect using setters.
"""

# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 创建一个新的图形对象
fig = plt.figure()

# 定义文本对齐方式的字典
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}

# 在图形中显示字体族选项
fig.text(0.1, 0.9, 'family', size='large', **alignment)
# 不同字体族的选项列表
families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
# 遍历每个字体族，显示其名称
for k, family in enumerate(families):
    fig.text(0.1, yp[k], family, family=family, **alignment)

# 在图形中显示字体样式选项
fig.text(0.3, 0.9, 'style', **alignment)
# 不同样式的选项列表
styles = ['normal', 'italic', 'oblique']
# 遍历每个样式，显示其名称
for k, style in enumerate(styles):
    fig.text(0.3, yp[k], style, family='sans-serif', style=style, **alignment)

# 在图形中显示字体变体选项
fig.text(0.5, 0.9, 'variant', **alignment)
# 不同变体的选项列表
variants = ['normal', 'small-caps']
# 遍历每个变体，显示其名称
for k, variant in enumerate(variants):
    fig.text(0.5, yp[k], variant, family='serif', variant=variant, **alignment)

# 在图形中显示字体粗细选项
fig.text(0.7, 0.9, 'weight', **alignment)
# 不同粗细的选项列表
weights = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']
# 遍历每种粗细，显示其名称
for k, weight in enumerate(weights):
    fig.text(0.7, yp[k], weight, weight=weight, **alignment)

# 在图形中显示字体大小选项
fig.text(0.9, 0.9, 'size', **alignment)
# 不同大小的选项列表
sizes = [
    'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
# 遍历每种大小，显示其名称
for k, size in enumerate(sizes):
    fig.text(0.9, yp[k], size, size=size, **alignment)

# 在图形中显示粗体斜体文本选项
fig.text(0.3, 0.1, 'bold italic',
         style='italic', weight='bold', size='x-small', **alignment)
fig.text(0.3, 0.2, 'bold italic',
         style='italic', weight='bold', size='medium', **alignment)
fig.text(0.3, 0.3, 'bold italic',
         style='italic', weight='bold', size='x-large', **alignment)

# 显示绘图结果
plt.show()
```