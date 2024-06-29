# `D:\src\scipysrc\matplotlib\galleries\examples\userdemo\pgf_texsystem.py`

```
"""
=============
PGF texsystem
=============
"""

# 导入 matplotlib.pyplot 库，用于绘图操作
import matplotlib.pyplot as plt

# 更新 matplotlib 的配置参数，设置 PGF 绘图系统相关选项
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",  # 设置 PGF 使用的 TeX 系统为 pdflatex
    "pgf.preamble": "\n".join([    # 添加 LaTeX 的 preamble 部分，包括以下几行
         r"\usepackage[utf8x]{inputenc}",  # 使用 utf8x 编码处理输入
         r"\usepackage[T1]{fontenc}",      # 使用 T1 字体编码
         r"\usepackage{cmbright}",         # 使用 cmbright 字体
    ]),
})

# 创建一个新的图形对象和坐标轴对象
fig, ax = plt.subplots(figsize=(4.5, 2.5))

# 在坐标轴上绘制简单的线图，显示从 0 到 4 的整数
ax.plot(range(5))

# 在指定位置添加文本，指定字体族为 serif
ax.text(0.5, 3., "serif", family="serif")
# 在指定位置添加文本，指定字体族为 monospace
ax.text(0.5, 2., "monospace", family="monospace")
# 在指定位置添加文本，指定字体族为 sans-serif
ax.text(2.5, 2., "sans-serif", family="sans-serif")

# 设置 x 轴标签，使用 LaTeX 语法显示特殊字符 µ
ax.set_xlabel(r"µ is not $\mu$")

# 调整图形布局，设置填充值为 0.5
fig.tight_layout(pad=.5)

# 将图形保存为 PDF 格式文件
fig.savefig("pgf_texsystem.pdf")
# 将图形保存为 PNG 格式文件
fig.savefig("pgf_texsystem.png")
```