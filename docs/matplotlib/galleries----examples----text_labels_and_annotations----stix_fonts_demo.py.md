# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\stix_fonts_demo.py`

```
"""
==========
STIX Fonts
==========

Demonstration of `STIX Fonts <https://www.stixfonts.org/>`_ used in LaTeX
rendering.
"""

# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 定义一个包含圆圈数字字符的字符串
circle123 = "\N{CIRCLED DIGIT ONE}\N{CIRCLED DIGIT TWO}\N{CIRCLED DIGIT THREE}"

# 定义一组测试字符串，包含不同的数学字体和样式
tests = [
    r'$%s\;\mathrm{%s}\;\mathbf{%s}$' % ((circle123,) * 3),
    r'$\mathsf{Sans \Omega}\;\mathrm{\mathsf{Sans \Omega}}\;'
    r'\mathbf{\mathsf{Sans \Omega}}$',
    r'$\mathtt{Monospace}$',
    r'$\mathcal{CALLIGRAPHIC}$',
    r'$\mathbb{Blackboard\;\pi}$',
    r'$\mathrm{\mathbb{Blackboard\;\pi}}$',
    r'$\mathbf{\mathbb{Blackboard\;\pi}}$',
    r'$\mathfrak{Fraktur}\;\mathbf{\mathfrak{Fraktur}}$',
    r'$\mathscr{Script}$',
]

# 创建一个新的图形对象，并设置其尺寸
fig = plt.figure(figsize=(8, len(tests) + 2))

# 遍历测试字符串列表，并将每个字符串添加到图形中
for i, s in enumerate(tests[::-1]):
    fig.text(0, (i + .5) / len(tests), s, fontsize=32)

# 显示图形
plt.show()
```