# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\tex_demo.py`

```
"""
==================================
Rendering math equations using TeX
==================================

You can use TeX to render all of your Matplotlib text by setting
:rc:`text.usetex` to True.  This requires that you have TeX and the other
dependencies described in the :ref:`usetex` tutorial properly
installed on your system.  Matplotlib caches processed TeX expressions, so that
only the first occurrence of an expression triggers a TeX compilation. Later
occurrences reuse the rendered image from the cache and are thus faster.

Unicode input is supported, e.g. for the y-axis label in this example.
"""

# 导入Matplotlib库，用于绘图
import matplotlib.pyplot as plt
# 导入NumPy库，用于数学运算和数组操作
import numpy as np

# 设置Matplotlib参数，使其支持TeX渲染
plt.rcParams['text.usetex'] = True

# 生成时间轴数据
t = np.linspace(0.0, 1.0, 100)
# 生成信号数据
s = np.cos(4 * np.pi * t) + 2

# 创建一个绘图窗口和子图
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
# 绘制信号图
ax.plot(t, s)

# 设置X轴标签，使用粗体文本和TeX格式
ax.set_xlabel(r'\textbf{time (s)}')
# 设置Y轴标签，使用斜体文本和特殊Unicode符号（角度符号）
ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
# 设置图标题，包含TeX公式
ax.set_title(r'\TeX\ is Number $\displaystyle\sum_{n=1}^\infty'
             r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')

# %%
# 更复杂的例子

# 创建另一个绘图窗口和子图
fig, ax = plt.subplots()
# 绘制三条曲线，并指定不同的颜色和线型
N = 500
delta = 0.6
X = np.linspace(-1, 1, N)
ax.plot(X, (1 - np.tanh(4 * X / delta)) / 2,    # 相场tanh曲线
        X, (1.4 + np.tanh(4 * X / delta)) / 4, "C2",  # 成分曲线
        X, X < 0, "k--")                        # 锐利界面曲线

# 添加图例
ax.legend(("phase field", "level set", "sharp interface"),
          shadow=True, loc=(0.01, 0.48), handlelength=1.5, fontsize=16)

# 添加箭头和标签
ax.annotate("", xy=(-delta / 2., 0.1), xytext=(delta / 2., 0.1),
            arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
ax.text(0, 0.1, r"$\delta$",
        color="black", fontsize=24,
        horizontalalignment="center", verticalalignment="center",
        bbox=dict(boxstyle="round", fc="white", ec="black", pad=0.2))

# 使用TeX渲染标签
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(["$-1$", r"$\pm 0$", "$+1$"], color="k", size=20)

# 设置左侧Y轴标签，结合数学模式和文本模式
ax.set_ylabel(r"\bf{phase field} $\phi$", color="C0", fontsize=20)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([r"\bf{0}", r"\bf{.5}", r"\bf{1}"], color="k", size=20)

# 设置右侧Y轴标签
ax.text(1.02, 0.5, r"\bf{level set} $\phi$",
        color="C2", fontsize=20, rotation=90,
        horizontalalignment="left", verticalalignment="center",
        clip_on=False, transform=ax.transAxes)

# 在文本中使用多行环境
# 设置相场方程
eq1 = (r"\begin{eqnarray*}"
       r"|\nabla\phi| &=& 1,\\"
       r"\frac{\partial \phi}{\partial t} + U|\nabla \phi| &=& 0 "
       r"\end{eqnarray*}")
ax.text(1, 0.9, eq1, color="C2", fontsize=18,
        horizontalalignment="right", verticalalignment="top")

# 设置相场方程
# 创建多行数学公式的 LaTeX 表达式，包含多个方程式
eq2 = (r"\begin{eqnarray*}"
       r"\mathcal{F} &=& \int f\left( \phi, c \right) dV, \\ "
       r"\frac{ \partial \phi } { \partial t } &=& -M_{ \phi } "
       r"\frac{ \delta \mathcal{F} } { \delta \phi }"
       r"\end{eqnarray*}")

# 在图形对象 ax 上添加文本，显示数学公式 eq2，位置 (0.18, 0.18)，颜色为蓝色 ("C0")，字体大小为 16
ax.text(0.18, 0.18, eq2, color="C0", fontsize=16)

# 在图形对象 ax 上添加文本，显示文本 "gamma: $\gamma$"，位置 (-1, 0.30)，颜色为红色 ("r")，字体大小为 20
ax.text(-1, .30, r"gamma: $\gamma$", color="r", fontsize=20)

# 在图形对象 ax 上添加文本，显示文本 "Omega: $\Omega$"，位置 (-1, 0.18)，颜色为蓝色 ("b")，字体大小为 20
ax.text(-1, .18, r"Omega: $\Omega$", color="b", fontsize=20)

# 显示图形对象 ax 中的所有内容
plt.show()
```