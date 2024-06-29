# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\usetex_baseline_test.py`

```py
"""
====================
Usetex Baseline Test
====================

Comparison of text baselines computed for mathtext and usetex.
"""

# 导入 matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt

# 更新全局配置，设置 mathtext 使用的字体集为 'cm'，数学文本的默认字体为 'serif'
plt.rcParams.update({"mathtext.fontset": "cm", "mathtext.rm": "serif"})

# 创建一个 1x2 的子图布局，返回子图对象的数组 axs
axs = plt.figure(figsize=(2 * 3, 6.5)).subplots(1, 2)

# 遍历 axs 数组和 [False, True] 列表，分别给每个子图对象添加内容
for ax, usetex in zip(axs, [False, True]):
    # 在当前子图 ax 上绘制红色竖线
    ax.axvline(0, color="r")

    # 定义测试文本字符串列表，包含普通文本和 LaTeX 数学表达式
    test_strings = ["lg", r"$\frac{1}{2}\pi$", r"$p^{3^A}$", r"$p_{3_2}$"]
    
    # 遍历测试文本列表，每个文本字符串在子图 ax 上绘制相应的文本
    for i, s in enumerate(test_strings):
        # 在子图 ax 上绘制红色水平线
        ax.axhline(i, color="r")
        
        # 在坐标 (0, 3-i) 处绘制文本 s，根据 usetex 参数选择是否启用 LaTeX 渲染
        ax.text(0., 3 - i, s,
                usetex=usetex,
                verticalalignment="baseline",  # 垂直对齐方式为基线
                size=50,  # 文本大小为 50
                bbox=dict(pad=0, ec="k", fc="none"))  # 文本周围边框设置

    # 设置子图 ax 的 x 轴和 y 轴范围、刻度和标题，标题中显示当前 usetex 的取值
    ax.set(xlim=(-0.1, 1.1), ylim=(-.8, 3.9), xticks=[], yticks=[],
           title=f"usetex={usetex}\n")

# 显示绘制的图形
plt.show()
```