# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\mathtext_fontfamily_example.py`

```
"""
===============
Math fontfamily
===============

A simple example showcasing the new *math_fontfamily* parameter that can
be used to change the family of fonts for each individual text
element in a plot.

If no parameter is set, the global value
:rc:`mathtext.fontset` will be used.
"""

# 导入matplotlib.pyplot库，用于绘图操作
import matplotlib.pyplot as plt

# 创建一个6x5大小的图形对象和相应的轴对象
fig, ax = plt.subplots(figsize=(6, 5))

# 在图中绘制一个背景简单的曲线
ax.plot(range(11), color="0.9")

# 创建一个包含普通文本和数学文本的字符串
msg = (r"Normal Text. $Text\ in\ math\ mode:\ "
       r"\int_{0}^{\infty } x^2 dx$")

# 在图中设置文本，其中数学文本使用了指定的字体家族'cm'
ax.text(1, 7, msg, size=12, math_fontfamily='cm')

# 在图中设置另一个文本，其中数学文本使用了指定的字体家族'dejavuserif'
ax.text(1, 3, msg, size=12, math_fontfamily='dejavuserif')

# *math_fontfamily* 参数可以用在大部分需要文本的地方，比如标题中
ax.set_title(r"$Title\ in\ math\ mode:\ \int_{0}^{\infty } x^2 dx$",
             math_fontfamily='stixsans', size=14)

# 注意，普通文本不受*math_fontfamily*参数的影响
# 显示绘制的图形
plt.show()
```