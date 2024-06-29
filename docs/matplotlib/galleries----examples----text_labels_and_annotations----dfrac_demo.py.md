# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\dfrac_demo.py`

```py
"""
=========================================
The difference between \\dfrac and \\frac
=========================================

In this example, the differences between the \\dfrac and \\frac TeX macros are
illustrated; in particular, the difference between display style and text style
fractions when using Mathtex.

.. versionadded:: 2.1

.. note::
    To use \\dfrac with the LaTeX engine (text.usetex : True), you need to
    import the amsmath package with the text.latex.preamble rc, which is
    an unsupported feature; therefore, it is probably a better idea to just
    use the \\displaystyle option before the \\frac macro to get this behavior
    with the LaTeX engine.

"""

# 导入 matplotlib 的 pyplot 模块，命名为 plt
import matplotlib.pyplot as plt

# 创建一个大小为 5.25x0.75 英寸的新图像
fig = plt.figure(figsize=(5.25, 0.75))

# 在图像中添加文本，展示 \\dfrac 宏的使用及其效果
fig.text(0.5, 0.3, r'\dfrac: $\dfrac{a}{b}$',
         horizontalalignment='center', verticalalignment='center')

# 在图像中添加文本，展示 \\frac 宏的使用及其效果
fig.text(0.5, 0.7, r'\frac: $\frac{a}{b}$',
         horizontalalignment='center', verticalalignment='center')

# 显示绘制的图像
plt.show()
```