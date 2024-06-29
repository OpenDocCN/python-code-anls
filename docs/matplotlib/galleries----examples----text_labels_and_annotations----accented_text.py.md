# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\accented_text.py`

```
r"""
=============
Accented text
=============

Matplotlib supports accented characters via TeX mathtext or Unicode.

Using mathtext, the following accents are provided: \\hat, \\breve, \\grave,
\\bar, \\acute, \\tilde, \\vec, \\dot, \\ddot.  All of them have the same
syntax, e.g. \\bar{o} yields "o overbar", \\ddot{o} yields "o umlaut".
Shortcuts such as \\"o \\'e \\`e \\~n \\.x \\^y are also supported.
"""

# 导入 matplotlib.pyplot 模块
import matplotlib.pyplot as plt

# Mathtext demo
# 创建一个包含一条简单折线图的图形对象和坐标轴对象
fig, ax = plt.subplots()
# 在图形上绘制一个简单的折线
ax.plot(range(10))
# 设置标题，使用 mathtext 显示带有重音的字符
ax.set_title(r'$\ddot{o}\acute{e}\grave{e}\hat{O}'
             r'\breve{i}\bar{A}\tilde{n}\vec{q}$', fontsize=20)

# 使用简写形式也是支持的，大括号是可选的
# 设置 x 轴标签，使用 mathtext 显示带有简写形式的重音字符
ax.set_xlabel(r"""$\"o\ddot o \'e\`e\~n\.x\^y$""", fontsize=20)
# 在图上添加文本，显示一个物理公式
ax.text(4, 0.5, r"$F=m\ddot{x}$")
# 调整图形布局
fig.tight_layout()

# %%
# You can also use Unicode characters directly in strings.
# 创建一个新的图形对象和坐标轴对象
fig, ax = plt.subplots()
# 设置标题，使用 Unicode 字符直接显示带有重音的文本
ax.set_title("GISCARD CHAHUTÉ À L'ASSEMBLÉE")
# 设置 x 轴标签，显示带有特定重音的文本
ax.set_xlabel("LE COUP DE DÉ DE DE GAULLE")
# 设置 y 轴标签，显示普通文本
ax.set_ylabel('André was here!')
# 在图上添加文本，显示德文带有特定重音的文本，并旋转 45 度
ax.text(0.2, 0.8, 'Institut für Festkörperphysik', rotation=45)
# 在图上添加文本，显示一个英文缩写并检查字符之间的间距
ax.text(0.4, 0.2, 'AVA (check kerning)')

# 显示图形
plt.show()
```