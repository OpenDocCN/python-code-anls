# `D:\src\scipysrc\seaborn\doc\sphinxext\tutorial_builder.py`

```
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path
# 导入 warnings 模块，用于管理警告信息
import warnings

# 从 jinja2 模块中导入 Environment 类，用于处理模板渲染
from jinja2 import Environment
# 导入 yaml 模块，用于读取和解析 YAML 格式的文件
import yaml

# 导入 numpy 库并使用 np 别名
import numpy as np
# 导入 matplotlib 库并使用 mpl 别名
import matplotlib as mpl
# 导入 seaborn 库并使用 sns 别名
import seaborn as sns
# 从 seaborn 库中导入对象模块，并使用 so 别名
import seaborn.objects as so

# 定义一个多行字符串模板 TEMPLATE，用于生成用户指南和教程的 RST 文件
TEMPLATE = """
:notoc:

.. _tutorial:

User guide and tutorial
=======================
{% for section in sections %}
{{ section.header }}
{% for page in section.pages %}
.. grid:: 1
  :gutter: 2

  .. grid-item-card::

    .. grid:: 2

      .. grid-item::
        :columns: 3

        .. image:: ./tutorial/{{ page }}.svg
          :target: ./tutorial/{{ page }}.html

      .. grid-item::
        :columns: 9
        :margin: auto

        .. toctree::
          :maxdepth: 2

          tutorial/{{ page }}
{% endfor %}
{% endfor %}
"""


# 主函数 main(app)，接收一个应用对象 app
def main(app):
    # 构建内容的 YAML 文件路径
    content_yaml = Path(app.builder.srcdir) / "tutorial.yaml"
    # 输出的教程 RST 文件路径
    tutorial_rst = Path(app.builder.srcdir) / "tutorial.rst"

    # 教程目录路径
    tutorial_dir = Path(app.builder.srcdir) / "tutorial"
    # 如果教程目录不存在，则创建它
    tutorial_dir.mkdir(exist_ok=True)

    # 打开内容 YAML 文件
    with open(content_yaml) as fid:
        # 使用 YAML 加载器读取文件内容并解析为 Python 对象
        sections = yaml.load(fid, yaml.BaseLoader)

    # 对每个章节进行处理，为每个章节添加标题
    for section in sections:
        title = section["title"]
        # 如果章节有标题，则生成标题行
        section["header"] = "\n".join([title, "-" * len(title)]) if title else ""

    # 使用 TEMPLATE 字符串创建一个 Jinja2 环境对象
    env = Environment().from_string(TEMPLATE)
    # 渲染模板，传入章节信息
    content = env.render(sections=sections)

    # 将渲染后的内容写入教程 RST 文件
    with open(tutorial_rst, "w") as fid:
        fid.write(content)

    # 对每个章节中的页面生成缩略图
    for section in sections:
        for page in section["pages"]:
            # 构建 SVG 文件路径
            svg_path = tutorial_dir / f"{page}.svg"
            # 如果 SVG 文件不存在或者其修改时间早于当前文件的修改时间，则生成缩略图
            if (
                not svg_path.exists()
                or svg_path.stat().st_mtime < Path(__file__).stat().st_mtime
            ):
                write_thumbnail(svg_path, page)


# 写缩略图函数 write_thumbnail，接收 SVG 文件路径和页面名称
def write_thumbnail(svg_path, page):
    # 设置 seaborn 的样式、绘图上下文和颜色板
    with (
        sns.axes_style("dark"),
        sns.plotting_context("notebook"),
        sns.color_palette("deep")
    ):
        # 根据页面名称调用全局函数，生成图形对象 fig
        fig = globals()[page]()
        # 遍历图形对象的每个轴，设置坐标轴标签和标题为空，仅保留网格线
        for ax in fig.axes:
            ax.set(xticklabels=[], yticklabels=[], xlabel="", ylabel="", title="")
        # 忽略警告，调整图形布局并保存为 SVG 格式的缩略图
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        fig.savefig(svg_path, format="svg")


# 简介函数 introduction，加载示例数据集并绘制图形
def introduction():
    # 加载示例数据集 tips、fmri 和 penguins
    tips = sns.load_dataset("tips")
    fmri = sns.load_dataset("fmri").query("region == 'parietal'")
    penguins = sns.load_dataset("penguins")

    # 创建一个大小为 5x5 的图形对象 f
    f = mpl.figure.Figure(figsize=(5, 5))
    # 使用 seaborn 的 whitegrid 样式创建带有 2x2 子图的图形对象
    with sns.axes_style("whitegrid"):
        f.subplots(2, 2)

    # 在第一个子图上绘制散点图，设置 x 轴为 total_bill，y 轴为 tip，按性别分组和大小标记
    sns.scatterplot(
        tips, x="total_bill", y="tip", hue="sex", size="size",
        alpha=.75, palette=["C0", ".5"], legend=False, ax=f.axes[0],
    )
    # 在第二个子图上绘制 KDE 图，设置 x 轴为 total_bill，按大小分组，使用混合调色板
    sns.kdeplot(
        tips.query("size != 5"), x="total_bill", hue="size",
        palette="blend:C0,.5", fill=True, linewidth=.5,
        legend=False, common_norm=False, ax=f.axes[1],
    )
    # 在第三个子图上绘制折线图，设置 x 轴为 timepoint，y 轴为 signal，按事件类型分组
    sns.lineplot(
        fmri, x="timepoint", y="signal", hue="event",
        errorbar=("se", 2), legend=False, palette=["C0", ".5"], ax=f.axes[2],
    )
    # 绘制箱线图，显示企鹅数据中嘴喙深度与物种及性别的关系
    sns.boxplot(
        penguins, x="bill_depth_mm", y="species", hue="sex",
        whiskerprops=dict(linewidth=1.5), medianprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.5), capprops=dict(linewidth=0),
        width=.5, palette=["C0", ".8"], whis=5, ax=f.axes[3],
    )
    # 移除第四个子图的图例
    f.axes[3].legend_ = None
    # 针对所有子图，设置不显示刻度线
    for ax in f.axes:
        ax.set(xticks=[], yticks=[])
    # 返回包含所有子图的图形对象
    return f
# 定义一个函数，生成一个包含示意图的 matplotlib Figure 对象
def function_overview():
    # 导入 FancyBboxPatch 类，用于创建带边框的矩形图形
    from matplotlib.patches import FancyBboxPatch

    # 创建一个 Figure 对象，设置大小为 (7, 5) inches
    f = mpl.figure.Figure(figsize=(7, 5))
    
    # 使用 seaborn 提供的白色风格创建子图
    with sns.axes_style("white"):
        ax = f.subplots()
    
    # 调整子图边界使其填满整个 Figure
    f.subplots_adjust(0, 0, 1, 1)
    
    # 关闭子图的坐标轴
    ax.set_axis_off()
    
    # 设置子图的 x 和 y 范围为 (0, 1)
    ax.set(xlim=(0, 1), ylim=(0, 1))

    # 创建颜色字典，指定不同功能类别的颜色
    deep = sns.color_palette("deep")
    colors = dict(relational=deep[0], distributions=deep[1], categorical=deep[2])
    
    # 创建文本颜色字典，与功能类别对应
    dark = sns.color_palette("dark")
    text_colors = dict(relational=dark[0], distributions=dark[1], categorical=dark[2])

    # 定义不同功能类别及其包含的函数列表
    functions = dict(
        relational=["scatterplot", "lineplot"],
        distributions=["histplot", "kdeplot", "ecdfplot", "rugplot"],
        categorical=[
            "stripplot", "swarmplot", "boxplot", "violinplot", "pointplot", "barplot"
        ],
    )
    
    # 定义矩形框的填充间距（pad）、宽度（w）、高度（h）
    pad, w, h = .06, .2, .15
    
    # 在 x 轴上创建函数类别的位置
    xs, y = np.arange(0, 1, 1 / 3) + pad * 1.05, .7
    
    # 遍历每个函数类别及其对应的位置
    for x, mod in zip(xs, functions):
        # 设置当前类别的颜色，稍微透明一些
        color = colors[mod] + (.2,)
        
        # 获取当前类别的文本颜色
        text_color = text_colors[mod]
        
        # 添加一个白色背景的 FancyBboxPatch 到子图 ax 上
        ax.add_artist(FancyBboxPatch((x, y), w, h, f"round,pad={pad}", color="white"))
        
        # 添加一个带有边框和填充颜色的 FancyBboxPatch 到子图 ax 上
        ax.add_artist(FancyBboxPatch(
            (x, y), w, h, f"round,pad={pad}",
            linewidth=1, edgecolor=text_color, facecolor=color,
        ))
        
        # 在矩形框中心添加类别名称的文本
        ax.text(
            x + w / 2, y + h / 2, f"{mod[:3]}plot\n({mod})",
            ha="center", va="center", size=20, color=text_color
        )
        
        # 遍历当前类别的函数列表
        for i, func in enumerate(functions[mod]):
            # 计算当前函数名称的位置坐标
            x_i, y_i = x + w / 2, y - i * .1 - h / 2 - pad
            xy = x_i - w / 2, y_i - pad / 3
            
            # 添加一个白色背景的 FancyBboxPatch 到子图 ax 上
            ax.add_artist(
                FancyBboxPatch(xy, w, h / 4, f"round,pad={pad / 3}", color="white")
            )
            
            # 添加一个带有边框和填充颜色的 FancyBboxPatch 到子图 ax 上
            ax.add_artist(FancyBboxPatch(
                xy, w, h / 4, f"round,pad={pad / 3}",
                linewidth=1, edgecolor=text_color, facecolor=color
            ))
            
            # 在函数名称的中心添加文本
            ax.text(x_i, y_i, func, ha="center", va="center", size=16, color=text_color)
        
        # 绘制从类别名称到第一个函数名称之间的连接线
        ax.plot([x_i, x_i], [y, y_i], zorder=-100, color=text_color, lw=1)
    
    # 返回生成的 Figure 对象
    return f


# 定义一个函数，生成一个包含数据结构示意图的 matplotlib Figure 对象
def data_structure():
    # 创建一个 Figure 对象，设置大小为 (7, 5) inches
    f = mpl.figure.Figure(figsize=(7, 5))
    
    # 使用 gridspec 定义 Figure 的网格布局，设置不同子图的位置、颜色
    gs = mpl.gridspec.GridSpec(
        figure=f, ncols=6, nrows=2, height_ratios=(1, 20),
        left=0, right=.35, bottom=0, top=.9, wspace=.1, hspace=.01
    )
    
    # 创建深色调色板，为每个子图添加不同的颜色背景
    colors = [c + (.5,) for c in sns.color_palette("deep")]
    
    # 在第一个子图位置添加一个灰色背景的块
    f.add_subplot(gs[0, :], facecolor=".8")
    
    # 遍历网格布局的列数，在每列的子图位置添加不同颜色背景的块
    for i in range(gs.ncols):
        f.add_subplot(gs[1:, i], facecolor=colors[i])

    # 使用 gridspec 定义 Figure 的另一个网格布局，设置不同子图的位置、颜色
    gs = mpl.gridspec.GridSpec(
        figure=f, ncols=2, nrows=2, height_ratios=(1, 8), width_ratios=(1, 11),
        left=.4, right=1, bottom=.2, top=.8, wspace=.015, hspace=.02
    )
    
    # 在第一个子图位置添加第三个颜色的背景块
    f.add_subplot(gs[0, 1:], facecolor=colors[2])
    
    # 在第二个子图位置添加第二个颜色的背景块
    f.add_subplot(gs[1:, 0], facecolor=colors[1])
    
    # 在第三个子图位置添加第一个颜色的背景块
    f.add_subplot(gs[1, 1], facecolor=colors[0])
    
    # 返回生成的 Figure 对象
    return f


# 定义一个函数，加载名为 "diamonds" 的数据集并返回
def error_bars():
    diamonds = sns.load_dataset("diamonds")
    # 使用 seaborn 库设置图表样式为白色网格风格
    with sns.axes_style("whitegrid"):
        # 使用 catplot 函数创建分类图表 g，显示钻石数据
        g = sns.catplot(
            diamonds,             # 使用 diamonds 数据集
            x="carat",            # x 轴为钻石的克拉数
            y="clarity",          # y 轴为钻石的清晰度
            hue="clarity",        # 根据清晰度分类并进行颜色区分
            kind="point",         # 绘制点图
            errorbar=("sd", .5),  # 错误条形图的设置
            join=False,           # 不连接点
            legend=False,         # 不显示图例
            facet_kws={"despine": False},  # 子图设置，不去除轴线
            palette="ch:s=-.2,r=-.2,d=.4,l=.6_r",  # 使用自定义调色板
            scale=.75,            # 图形缩放比例
            capsize=.3,           # 误差线末端帽子的大小
        )
    
    # 将图表 g 中的 y 轴反向显示（正常方向）
    g.ax.yaxis.set_inverted(False)
    
    # 返回图表对象 g 的图形（figure）
    return g.figure
# 创建一个新的 Figure 对象，指定尺寸为 5x5 英寸
def properties():
    f = mpl.figure.Figure(figsize=(5, 5))
    
    # 创建一个包含整数序列的 numpy 数组，从 1 到 10
    x = np.arange(1, 11)
    # 创建与 x 大小相同的零数组
    y = np.zeros_like(x)
    
    # 使用自定义的 Plot 类创建一个 Plot 对象
    p = so.Plot(x, y)
    # 设置点的大小为 14
    ps = 14
    
    # 创建包含多个图表对象的列表 plots
    plots = [
        # 向 p 对象添加点图，设置点的大小和颜色
        p.add(so.Dot(pointsize=ps), color=map(str, x)),
        # 向 p 对象添加点图，设置点的颜色和透明度
        p.add(so.Dot(color=".3", pointsize=ps), alpha=x),
        # 向 p 对象添加点图，设置点的颜色、大小和边缘宽度
        p.add(so.Dot(color=".9", pointsize=ps, edgewidth=2), edgecolor=x),
        # 向 p 对象添加点图，设置点的颜色和大小，然后调整点的大小范围
        p.add(so.Dot(color=".3"), pointsize=x).scale(pointsize=(4, 18)),
        # 向 p 对象添加点图，设置点的大小、颜色和边缘颜色
        p.add(so.Dot(pointsize=ps, color=".9", edgecolor=".2"), edgewidth=x),
        # 向 p 对象添加点图，设置点的大小、颜色和标记
        p.add(so.Dot(pointsize=ps, color=".3"), marker=map(str, x)),
        # 向 p 对象添加点图，设置点的大小、颜色和标记为 'x'
        p.add(so.Dot(pointsize=ps, color=".3", marker="x"), stroke=x),
    ]
    
    # 使用 seaborn 的 axes_style 创建带有 "ticks" 风格的子图集合
    with sns.axes_style("ticks"):
        axs = f.subplots(len(plots))
    
    # 将每个 plot 对象分别绘制到对应的 axs 子图上
    for p, ax in zip(plots, axs):
        p.on(ax).plot()
        # 设置 x 轴标签和范围，隐藏 y 轴和其标签，设置 y 轴范围
        ax.set(xticks=x, yticks=[], xticklabels=[], ylim=(-.2, .3))
        # 去除子图 ax 的左边框
        sns.despine(ax=ax, left=True)
    
    # 清空 Figure 对象的图例列表
    f.legends = []
    
    # 返回绘制好的 Figure 对象
    return f


# 创建一个新的 Figure 对象，指定尺寸为 5x4 英寸
def objects_interface():
    f = mpl.figure.Figure(figsize=(5, 4))
    # 使用 seaborn 中的 color_palette 函数创建调色板 C
    C = sns.color_palette("deep")
    # 在 Figure 对象上创建子图 ax
    ax = f.subplots()
    # 设置字体大小为 22
    fontsize = 22
    # 创建包含多个矩形块信息的列表 rects
    rects = [((.135, .50), .69), ((.275, .38), .26), ((.59, .38), .40)]
    
    # 遍历 rects 列表，为每个矩形块添加到子图 ax 上
    for i, (xy, w) in enumerate(rects):
        ax.add_artist(mpl.patches.Rectangle(xy, w, .09, color=C[i], alpha=.2, lw=0))
    
    # 在子图 ax 上添加文本说明
    ax.text(0, .52, "Plot(data, 'x', 'y', color='var1')", size=fontsize, color=".2")
    ax.text(0, .40, ".add(Dot(alpha=.5), marker='var2')", size=fontsize, color=".2")
    
    # 创建包含多个注释文本信息的列表 annots
    annots = [
        ("Mapped\nin all layers", (.48, .62), (0, 55)),
        ("Set directly", (.41, .35), (0, -55)),
        ("Mapped\nin this layer", (.80, .35), (0, -55)),
    ]
    
    # 遍历 annots 列表，为每个注释信息添加到子图 ax 上
    for i, (text, xy, xytext) in enumerate(annots):
        ax.annotate(
            text, xy, xytext,
            textcoords="offset points", fontsize=18, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color=C[i]), color=C[i],
        )
    
    # 设置子图 ax 的坐标轴关闭
    ax.set_axis_off()
    # 调整 Figure 对象的子图布局
    f.subplots_adjust(0, 0, 1, 1)
    
    # 返回绘制好的 Figure 对象
    return f


# 创建一个新的 Figure 对象，用于绘制关系图
def relational():
    # 使用 seaborn 加载 mpg 数据集
    mpg = sns.load_dataset("mpg")
    # 使用 seaborn 的 axes_style 创建带有 "ticks" 风格的关系图 g
    with sns.axes_style("ticks"):
        g = sns.relplot(
            data=mpg, x="horsepower", y="mpg", size="displacement", hue="weight",
            sizes=(50, 500), hue_norm=(2000, 4500), alpha=.75, legend=False,
            palette="ch:start=-.5,rot=.7,dark=.3,light=.7_r",
        )
    
    # 设置关系图 g 的尺寸为 5x5 英寸
    g.figure.set_size_inches(5, 5)
    
    # 返回绘制好的 Figure 对象
    return g.figure


# 创建一个新的 Figure 对象，用于绘制分布图
def distributions():
    # 使用 seaborn 加载 penguins 数据集，且删除包含缺失值的行
    penguins = sns.load_dataset("penguins").dropna()
    # 使用 seaborn 的 axes_style 创建带有 "white" 风格的分布图 g
    with sns.axes_style("white"):
        g = sns.displot(
            penguins, x="flipper_length_mm", row="island",
            binwidth=4, kde=True, line_kws=dict(linewidth=2), legend=False,
        )
    
    # 去除分布图 g 的左边框
    sns.despine(left=True)
    # 设置分布图 g 的尺寸为 5x5 英寸
    g.figure.set_size_inches(5, 5)
    
    # 返回绘制好的 Figure 对象
    return g.figure


# 创建一个新的 Figure 对象，用于绘制分类图
def categorical():
    # 使用 seaborn 加载 penguins 数据集，且删除包含缺失值的行
    penguins = sns.load_dataset("penguins").dropna()
    
    # 使用 seaborn 库设置图形的风格为白色网格风格
    with sns.axes_style("whitegrid"):
        # 使用 seaborn 库创建一个分类图形
        g = sns.catplot(
            penguins,                # 数据集，假设为 penguins
            x="sex",                 # x 轴变量为性别
            y="body_mass_g",         # y 轴变量为体重
            hue="island",            # 根据岛屿区分不同颜色
            col="sex",               # 根据性别分列显示
            kind="box",              # 使用箱线图展示数据分布
            whis=np.inf,             # 箱线图中离群点的范围设为无限
            legend=False,            # 不显示图例
            sharex=False,            # 不共享 x 轴
        )
    # 去除图形的左边框线
    sns.despine(left=True)
    # 设置图形的尺寸为 5x5 英寸
    g.figure.set_size_inches(5, 5)
    # 返回图形对象的 Figure 对象
    return g.figure
# 定义一个函数用于展示 Anscombe 数据集的线性回归关系图
def regression():

    # 加载 Anscombe 数据集
    anscombe = sns.load_dataset("anscombe")
    
    # 使用白色风格设置绘图参数
    with sns.axes_style("white"):
        
        # 创建一个网格图，展示每个数据集的线性回归关系
        g = sns.lmplot(
            anscombe,  # 数据集
            x="x",     # x 轴数据
            y="y",     # y 轴数据
            hue="dataset",  # 根据数据集不同进行颜色分组
            col="dataset",  # 根据数据集不同创建子图列
            col_wrap=2,     # 每行最多展示 2 列子图
            scatter_kws=dict(edgecolor=".2", facecolor=".7", s=80),  # 散点图参数
            line_kws=dict(lw=4),  # 回归线参数
            ci=None,    # 不显示置信区间
        )
    
    # 设置 x 轴和 y 轴的显示范围
    g.set(xlim=(2, None), ylim=(2, None))
    
    # 设置整体图像大小
    g.figure.set_size_inches(5, 5)
    
    # 返回图像对象
    return g.figure


# 定义一个函数用于展示 penguins 数据集的成对关系图
def axis_grids():

    # 加载 penguins 数据集，并随机抽样 200 条数据
    penguins = sns.load_dataset("penguins").sample(200, random_state=0)
    
    # 使用 ticks 风格设置绘图参数
    with sns.axes_style("ticks"):
        
        # 创建一个成对关系图
        g = sns.pairplot(
            penguins.drop("flipper_length_mm", axis=1),  # 去除 flipper_length_mm 列后的数据集
            diag_kind="kde",    # 对角线上使用核密度估计
            diag_kws=dict(fill=False),  # 不填充对角线的图形
            plot_kws=dict(s=40, fc="none", ec="C0", alpha=.75, linewidth=.75),  # 散点图参数
        )
    
    # 设置整体图像大小
    g.figure.set_size_inches(5, 5)
    
    # 返回图像对象
    return g.figure


# 定义一个函数用于展示不同美学风格下的子图
def aesthetics():

    # 创建一个大小为 5x5 的空白画布对象
    f = mpl.figure.Figure(figsize=(5, 5))
    
    # 遍历不同的风格名和索引
    for i, style in enumerate(["darkgrid", "white", "ticks", "whitegrid"], 1):
        
        # 使用当前风格设置绘图参数
        with sns.axes_style(style):
            
            # 在 2x2 的子图中添加第 i 个子图
            ax = f.add_subplot(2, 2, i)
        
        # 设置 x 轴和 y 轴的刻度
        ax.set(xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
    
    # 移除部分子图的边框
    sns.despine(ax=f.axes[1])
    sns.despine(ax=f.axes[2])
    
    # 返回画布对象
    return f


# 定义一个函数用于展示不同颜色调色板下的子图
def color_palettes():

    # 创建一个大小为 5x5 的空白画布对象
    f = mpl.figure.Figure(figsize=(5, 5))
    
    # 定义一组调色板名称
    palettes = ["deep", "husl", "gray", "ch:", "mako", "vlag", "icefire"]
    
    # 在画布上创建与调色板数量相同的子图
    axs = f.subplots(len(palettes))
    
    # 定义一个 x 轴数据
    x = np.arange(10)
    
    # 遍历子图和调色板名称
    for ax, name in zip(axs, palettes):
        
        # 使用给定名称创建颜色映射
        cmap = mpl.colors.ListedColormap(sns.color_palette(name, x.size))
        
        # 绘制彩色网格
        ax.pcolormesh(x[None, :], linewidth=.5, edgecolor="w", alpha=.8, cmap=cmap)
        
        # 关闭子图的坐标轴
        ax.set_axis_off()
    
    # 返回画布对象
    return f


# 定义一个函数设置 Sphinx 应用程序并连接到主函数
def setup(app):
    app.connect("builder-inited", main)
```