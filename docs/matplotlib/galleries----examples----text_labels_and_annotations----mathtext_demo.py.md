# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\mathtext_demo.py`

```py
"""
========
Mathtext
========

Use Matplotlib's internal LaTeX parser and layout engine.  For true LaTeX
rendering, see the text.usetex option.
"""

# 导入 matplotlib.pyplot 模块，通常用 plt 作为别名
import matplotlib.pyplot as plt

# 创建一个包含单个子图的 Figure 对象和一个 Axes 对象
fig, ax = plt.subplots()

# 在 Axes 对象上绘制曲线
ax.plot([1, 2, 3], label=r'$\sqrt{x^2}$')

# 在图例中添加标签
ax.legend()

# 设置 X 轴标签，并使用 LaTeX 标记显示
ax.set_xlabel(r'$\Delta_i^j$', fontsize=20)

# 设置 Y 轴标签，并使用 LaTeX 标记显示
ax.set_ylabel(r'$\Delta_{i+1}^j$', fontsize=20)

# 设置图表标题，并使用 LaTeX 标记显示
ax.set_title(r'$\Delta_i^j \hspace{0.4} \mathrm{versus} \hspace{0.4} '
             r'\Delta_{i+1}^j$', fontsize=20)

# 定义一个包含 LaTeX 公式的文本
tex = r'$\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\sin(2 \pi f x_i)$'

# 在指定位置添加文本标注
ax.text(1, 1.6, tex, fontsize=20, va='bottom')

# 调整图表布局，确保所有元素适当显示
fig.tight_layout()

# 显示图表
plt.show()
```