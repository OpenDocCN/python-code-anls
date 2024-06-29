# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\usetex_fonteffects.py`

```py
"""
==================
Usetex Fonteffects
==================

This script demonstrates that font effects specified in your pdftex.map
are now supported in usetex mode.
"""

# 导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 定义一个函数，返回一个用于 LaTeX 渲染的字体字符串
def setfont(font):
    return rf'\font\a {font} at 14pt\a '

# 创建一个新的图形对象
fig = plt.figure()

# 使用循环在图形上绘制文本，展示不同的字体效果
for y, font, text in zip(
    range(5),
    ['ptmr8r', 'ptmri8r', 'ptmro8r', 'ptmr8rn', 'ptmrr8re'],
    [f'Nimbus Roman No9 L {x}'
     for x in ['', 'Italics (real italics for comparison)',
               '(slanted)', '(condensed)', '(extended)']],
):
    # 在图形上添加文本，使用 LaTeX 渲染
    fig.text(.1, 1 - (y + 1) / 6, setfont(font) + text, usetex=True)

# 添加图形的总标题
fig.suptitle('Usetex font effects')

# 显示图形
plt.show()
```