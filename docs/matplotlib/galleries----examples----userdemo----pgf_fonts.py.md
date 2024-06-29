# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\pgf_fonts.py`

```
"""
=========
PGF fonts
=========
"""

# 导入 matplotlib.pyplot 库，用于绘图和设置字体
import matplotlib.pyplot as plt

# 更新全局绘图参数，设置字体系列为 serif
plt.rcParams.update({
    "font.family": "serif",
    # 使用 LaTeX 默认的 serif 字体。
    "font.serif": [],
    # 使用特定的草书字体。
    "font.cursive": ["Comic Neue", "Comic Sans MS"],
})

# 创建一个新的图形和子图对象，设置图形大小为 4.5x2.5 英寸
fig, ax = plt.subplots(figsize=(4.5, 2.5))

# 在子图上绘制一条简单的线
ax.plot(range(5))

# 在指定位置添加文本标签，使用默认的 serif 字体
ax.text(0.5, 3., "serif")

# 在指定位置添加文本标签，指定使用 monospace 字体
ax.text(0.5, 2., "monospace", family="monospace")

# 在指定位置添加文本标签，指定使用 sans-serif 字体（DejaVu Sans）
ax.text(2.5, 2., "sans-serif", family="DejaVu Sans")

# 在指定位置添加文本标签，指定使用 cursive 字体（Comic Neue 或 Comic Sans MS）
ax.text(2.5, 1., "comic", family="cursive")

# 设置 X 轴标签文本，包含特殊符号 $\\mu$
ax.set_xlabel("µ is not $\\mu$")

# 调整图形布局，设置填充为 0.5
fig.tight_layout(pad=.5)

# 将图形保存为 PDF 格式文件
fig.savefig("pgf_fonts.pdf")

# 将图形保存为 PNG 格式文件
fig.savefig("pgf_fonts.png")
```