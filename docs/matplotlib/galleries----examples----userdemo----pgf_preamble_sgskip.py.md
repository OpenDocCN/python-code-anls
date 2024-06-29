# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\pgf_preamble_sgskip.py`

```
"""
============
PGF preamble
============
"""

# 导入matplotlib库，别名为mpl
import matplotlib as mpl

# 使用PGF后端进行绘图
mpl.use("pgf")

# 导入matplotlib.pyplot模块，别名为plt
import matplotlib.pyplot as plt

# 更新全局参数设置
plt.rcParams.update({
    "font.family": "serif",  # 使用衬线字体作为文本元素的主字体
    "text.usetex": True,     # 使用内联数学符号作为刻度
    "pgf.rcfonts": False,    # 不从rc参数设置字体
    "pgf.preamble": "\n".join([
         r"\usepackage{url}",            # 加载额外的包
         r"\usepackage{unicode-math}",   # 设置Unicode数学模式
         r"\setmainfont{DejaVu Serif}",  # 使用DejaVu Serif作为主要字体
    ])
})

# 创建一个图形对象和一个轴对象
fig, ax = plt.subplots(figsize=(4.5, 2.5))

# 在轴对象上绘制一条线
ax.plot(range(5))

# 设置X轴标签，包含Unicode文本
ax.set_xlabel("unicode text: я, ψ, €, ü")

# 设置Y轴标签，包含LaTeX格式的URL
ax.set_ylabel(r"\url{https://matplotlib.org}")

# 添加图例，包含Unicode数学符号
ax.legend(["unicode math: $λ=∑_i^∞ μ_i^2$"])

# 调整图形布局，设置填充
fig.tight_layout(pad=.5)

# 将图形保存为PDF格式
fig.savefig("pgf_preamble.pdf")

# 将图形保存为PNG格式
fig.savefig("pgf_preamble.png")
```