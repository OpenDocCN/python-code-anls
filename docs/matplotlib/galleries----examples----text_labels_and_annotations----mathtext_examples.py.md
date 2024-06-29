# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\mathtext_examples.py`

```py
"""
=================
Mathtext Examples
=================

Selected features of Matplotlib's math rendering engine.
"""

# 导入必要的模块
import re  # 导入正则表达式模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

# 选取特定的数学表达式特征，参考 "Writing mathematical expressions" 教程，选择了随机的示例。
mathtext_demos = {
    "Header demo":
        r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = "
        r"U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} "
        r"\int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ "
        r"U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_"
        r"{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$",

    "Subscripts and superscripts":
        r"$\alpha_i > \beta_i,\ "
        r"\alpha_{i+1}^j = {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau},\ "
        r"\ldots$",

    "Fractions, binomials and stacked numbers":
        r"$\frac{3}{4},\ \binom{3}{4},\ \genfrac{}{}{0}{}{3}{4},\ "
        r"\left(\frac{5 - \frac{1}{x}}{4}\right),\ \ldots$",

    "Radicals":
        r"$\sqrt{2},\ \sqrt[3]{x},\ \ldots$",

    "Fonts":
        r"$\mathrm{Roman}\ , \ \mathit{Italic}\ , \ \mathtt{Typewriter} \ "
        r"\mathrm{or}\ \mathcal{CALLIGRAPHY}$",

    "Accents":
        r"$\acute a,\ \bar a,\ \breve a,\ \dot a,\ \ddot a, \ \grave a, \ "
        r"\hat a,\ \tilde a,\ \vec a,\ \widehat{xyz},\ \widetilde{xyz},\ "
        r"\ldots$",

    "Greek, Hebrew":
        r"$\alpha,\ \beta,\ \chi,\ \delta,\ \lambda,\ \mu,\ "
        r"\Delta,\ \Gamma,\ \Omega,\ \Phi,\ \Pi,\ \Upsilon,\ \nabla,\ "
        r"\aleph,\ \beth,\ \daleth,\ \gimel,\ \ldots$",

    "Delimiters, functions and Symbols":
        r"$\coprod,\ \int,\ \oint,\ \prod,\ \sum,\ "
        r"\log,\ \sin,\ \approx,\ \oplus,\ \star,\ \varpropto,\ "
        r"\infty,\ \partial,\ \Re,\ \leftrightsquigarrow, \ \ldots$",
}

n_lines = len(mathtext_demos)  # 统计数学表达式示例的行数

def doall():
    # Matplotlib 在线文档中使用的灰色 RGB 值
    mpl_grey_rgb = (51 / 255, 51 / 255, 51 / 255)

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.90],
                      facecolor="white", frameon=True)
    ax.set_xlim(0, 1)  # 设置 x 轴范围
    ax.set_ylim(0, 1)  # 设置 y 轴范围
    ax.set_title("Matplotlib's math rendering engine",
                 color=mpl_grey_rgb, fontsize=14, weight='bold')  # 设置标题
    ax.set_xticks([])  # 隐藏 x 轴刻度
    ax.set_yticks([])  # 隐藏 y 轴刻度

    line_axesfrac = 1 / n_lines  # 计算每行之间的间隔，以坐标轴单位表示

    # 绘制头部示例公式
    full_demo = mathtext_demos['Header demo']
    ax.annotate(full_demo,
                xy=(0.5, 1. - 0.59 * line_axesfrac),  # 注释文本的位置
                color='tab:orange', ha='center', fontsize=20)  # 设置注释文本的颜色、水平对齐方式和字体大小

    # 绘制特征示例公式
    # 对 mathtext_demos 字典中的每对键值对进行遍历，同时获取索引和内容
    for i_line, (title, demo) in enumerate(mathtext_demos.items()):
        # 打印当前索引和 demo 内容
        print(i_line, demo)
        
        # 如果当前索引为 0，则跳过本次循环
        if i_line == 0:
            continue

        # 计算当前行的基线和下一行的基线位置
        baseline = 1 - i_line * line_axesfrac
        baseline_next = baseline - line_axesfrac
        
        # 根据行号确定填充色，奇数行为 'tab:blue'，偶数行为 'white'
        fill_color = ['white', 'tab:blue'][i_line % 2]
        
        # 在图形上创建水平跨度，用指定颜色进行填充
        ax.axhspan(baseline, baseline_next, color=fill_color, alpha=0.2)
        
        # 在图形上添加注释，显示标题，位于固定位置
        ax.annotate(f'{title}:',
                    xy=(0.06, baseline - 0.3 * line_axesfrac),
                    color=mpl_grey_rgb, weight='bold')
        
        # 在图形上添加注释，显示 demo 内容，位于固定位置
        ax.annotate(demo,
                    xy=(0.04, baseline - 0.75 * line_axesfrac),
                    color=mpl_grey_rgb, fontsize=16)

    # 显示整个图形
    plt.show()
if '--latex' in sys.argv:
    # 检查命令行参数中是否包含 '--latex'，如果是则执行以下操作
    # 运行命令示例：python mathtext_examples.py --latex
    # 需要使用 amsmath 和 amssymb 宏包

    with open("mathtext_examples.ltx", "w") as fd:
        # 打开文件 "mathtext_examples.ltx" 以写入模式
        fd.write("\\documentclass{article}\n")
        fd.write("\\usepackage{amsmath, amssymb}\n")
        fd.write("\\begin{document}\n")
        fd.write("\\begin{enumerate}\n")

        for s in mathtext_demos.values():
            # 遍历 mathtext_demos 字典中的值，并对每个字符串 s 进行处理
            s = re.sub(r"(?<!\\)\$", "$$", s)
            # 将 s 中非转义的 $ 替换为 $$
            fd.write("\\item %s\n" % s)
            # 将处理后的 s 写入 LaTeX 文件中的一个 enumerate 项

        fd.write("\\end{enumerate}\n")
        fd.write("\\end{document}\n")
        # 写入 LaTeX 文档的结束部分

    subprocess.call(["pdflatex", "mathtext_examples.ltx"])
    # 调用系统命令行执行 pdflatex 命令，生成 PDF 文件
else:
    doall()
    # 如果命令行参数中没有 '--latex'，则调用函数 doall() 执行其他操作
```