# `D:\src\scipysrc\seaborn\doc\tools\generate_logos.py`

```
    # 导入所需的库
    import numpy as np  # 导入NumPy库，用于数值计算
    import seaborn as sns  # 导入Seaborn库，用于统计数据可视化
    from matplotlib import patches  # 导入Matplotlib库中的patches模块，用于图形绘制
    import matplotlib.pyplot as plt  # 导入Matplotlib库中的pyplot模块，用于绘制图表
    from scipy.signal import gaussian  # 从SciPy库中导入gaussian函数，用于生成高斯信号
    from scipy.spatial import distance  # 导入SciPy库中的distance模块，用于计算空间距离

    XY_CACHE = {}  # 定义空字典作为全局变量，用于缓存计算结果

    STATIC_DIR = "_static"  # 设置静态目录路径

    plt.rcParams["savefig.dpi"] = 300  # 设置Matplotlib的全局参数，保存图像的分辨率为300dpi

    def poisson_disc_sample(array_radius, pad_radius, candidates=100, d=2, seed=None):
        """使用泊松盘采样查找位置。"""
        # 参考链接：http://bost.ocks.org/mike/algorithms/
        rng = np.random.default_rng(seed)  # 使用给定的种子创建随机数生成器
        uniform = rng.uniform  # 创建均匀分布随机数生成函数的引用
        randint = rng.integers  # 创建整数随机数生成函数的引用

        # 缓存计算结果
        key = array_radius, pad_radius, seed
        if key in XY_CACHE:
            return XY_CACHE[key]

        start = np.zeros(d)  # 初始化起始点为零向量
        samples = [start]  # 将起始点添加到样本列表中
        queue = [start]  # 将起始点添加到队列中，用于扩展

        while queue:
            s_idx = randint(len(queue))  # 从队列中随机选择一个样本索引
            s = queue[s_idx]  # 获取选中的样本

            for i in range(candidates):
                # 从当前样本生成候选点坐标
                coords = uniform(s - 2 * pad_radius, s + 2 * pad_radius, d)

                # 检查接受候选点的三个条件
                in_array = np.sqrt(np.sum(coords ** 2)) < array_radius  # 候选点在半径内
                in_ring = np.all(distance.cdist(samples, [coords]) > pad_radius)  # 候选点在环形区域内

                if in_array and in_ring:
                    samples.append(coords)  # 接受候选点
                    queue.append(coords)  # 将候选点添加到扩展队列中
                    break

            if (i + 1) == candidates:
                queue.pop(s_idx)  # 如果所有候选点都不符合条件，则从队列中移除当前样本

        samples = np.array(samples)  # 将样本列表转换为NumPy数组
        XY_CACHE[key] = samples  # 将计算结果缓存起来
        return samples  # 返回计算得到的样本点集合

    def logo(
        ax,
        color_kws, ring, ring_idx, edge,
        pdf_means, pdf_sigma, dy, y0, w, h,
        hist_mean, hist_sigma, hist_y0, lw, skip,
        scatter, pad, scale,
    ):
        """绘制Logo图。"""

        ax.set(xlim=(35 + w, 95 - w), ylim=(-3, 53))  # 设置图表的X和Y轴范围
        ax.set_axis_off()  # 关闭坐标轴
        ax.set_aspect('equal')  # 设置坐标轴纵横比

        radius = 27  # 定义Logo圆的半径
        center = 65, 25  # 定义Logo圆的中心点坐标

        x = np.arange(101)  # 创建从0到100的数组，用于生成高斯曲线的X坐标
        y = gaussian(x.size, pdf_sigma)  # 生成标准高斯曲线

        x0 = 30  # 魔术数，指定X坐标的起始位置
        xx = x[x0:]  # 截取X坐标数组的一部分

        n = len(pdf_means)  # PDF均值的数量
        dys = np.linspace(0, (n - 1) * dy, n) - (n * dy / 2)  # 计算PDF曲线的垂直偏移量
        dys -= dys.mean()  # 减去平均偏移量

        pdfs = [h * (y[x0 - m:-m] + y0 + dy) for m, dy in zip(pdf_means, dys)]
        # 生成具有垂直偏移的PDF曲线，使用给定的高度因子和Y坐标偏移量

        pdfs.insert(0, np.full(xx.shape, -h))  # 在底部插入常数，用于填充底部空白区域
        pdfs.append(np.full(xx.shape, 50 + h))  # 在顶部插入常数，用于填充顶部空白区域

        colors = sns.cubehelix_palette(n + 1 + bool(hist_mean), **color_kws)
        # 使用Seaborn生成指定数量颜色的调色板

        bg = patches.Circle(
            center, radius=radius - 1 + ring, color="white",
            transform=ax.transData, zorder=0,
        )
        ax.add_artist(bg)
        # 在图表上添加圆形背景，用于显示Logo的边界和填充

        # 不显示的剪切元素，用于内部元素
    # 创建一个圆形补丁对象，用于表示一个圆
    fg = patches.Circle(center, radius=radius - edge, transform=ax.transData)

    # 如果需要显示环形，则创建一个楔形补丁对象，围绕圆周围绘制环形
    if ring:
        wedge = patches.Wedge(
            center, r=radius + edge / 2, theta1=0, theta2=360, width=edge / 2,
            transform=ax.transData, color=colors[ring_idx], alpha=1
        )
        ax.add_artist(wedge)

    # 添加直方图条形
    if hist_mean:
        # 选择直方图的颜色
        hist_color = colors.pop(0)
        # 使用高斯函数创建直方图的 Y 值
        hist_y = gaussian(x.size, hist_sigma)
        # 计算直方图的高度
        hist = 1.1 * h * (hist_y[x0 - hist_mean:-hist_mean] + hist_y0)
        # 计算直方图的宽度
        dx = x[skip] - x[0]
        # 计算直方图的 X 值
        hist_x = xx[::skip]
        # 计算直方图的高度
        hist_h = h + hist[::skip]
        # 设置使用的直方图条的条件，避免在边缘出现细小的条
        use = hist_x < center[0] + radius * .5
        # 在坐标轴上绘制直方图条
        bars = ax.bar(
            hist_x[use], hist_h[use], bottom=-h, width=dx,
            align="edge", color=hist_color, ec="w", lw=lw,
            zorder=3,
        )
        # 设置每个直方图条的剪辑路径
        for bar in bars:
            bar.set_clip_path(fg)

    # 添加每个平滑 PDF 波浪线
    for i, pdf in enumerate(pdfs[1:], 1):
        # 在两个 PDF 之间填充颜色，表示波浪线
        u = ax.fill_between(xx, pdfs[i - 1] + w, pdf, color=colors[i - 1], lw=0)
        # 设置填充区域的剪辑路径
        u.set_clip_path(fg)

    # 在顶部波浪区域添加散点图
    if scatter:
        # 设定随机种子值
        seed = sum(map(ord, "seaborn logo"))
        # 使用泊松分布对散点进行采样
        xy = poisson_disc_sample(radius - edge - ring, pad, seed=seed)
        # 计算散点与顶部波浪线的最小间隔
        clearance = distance.cdist(xy + center, np.c_[xx, pdfs[-2]])
        # 选择符合条件的散点
        use = clearance.min(axis=1) > pad / 1.8
        x, y = xy[use].T
        # 计算每个散点的大小
        sizes = (x - y) % 9

        # 在坐标轴上绘制散点图
        points = ax.scatter(
            x + center[0], y + center[1], s=scale * (10 + sizes * 5),
            zorder=5, color=colors[-1], ec="w", lw=scale / 2,
        )
        # 获取填充区域的路径
        path = u.get_paths()[0]
        # 设置散点图的剪辑路径和变换
        points.set_clip_path(path, transform=u.get_transform())
        # 隐藏填充区域的显示
        u.set_visible(False)
    # 调整图形边距使其填充整个画布
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    # 根据背景颜色设置画布背景色
    facecolor = (1, 1, 1, 1) if bg == "white" else (1, 1, 1, 0)

    # 针对 "png" 和 "svg" 两种格式保存图形文件到指定路径
    for ext in ["png", "svg"]:
        fig.savefig(f"{STATIC_DIR}/logo-{shape}-{variant}bg.{ext}", facecolor=facecolor)


if __name__ == "__main__":

    # 遍历不同背景颜色选项
    for bg in ["white", "light", "dark"]:

        # 根据背景颜色选择颜色索引
        color_idx = -1 if bg == "dark" else 0

        # 定义参数字典
        kwargs = dict(
            color_kws=dict(start=.3, rot=-.4, light=.8, dark=.3, reverse=True),
            ring=True, ring_idx=color_idx, edge=1,
            pdf_means=[8, 24], pdf_sigma=16,
            dy=1, y0=1.8, w=.5, h=12,
            hist_mean=2, hist_sigma=10, hist_y0=.6, lw=1, skip=6,
            scatter=True, pad=1.8, scale=.5,
        )

        # 根据颜色参数获取对应的颜色
        color = sns.cubehelix_palette(**kwargs["color_kws"])[color_idx]

        # 创建新图形和轴对象
        fig, ax = plt.subplots(figsize=(2, 2), facecolor="w", dpi=100)
        
        # 在轴上绘制 logo 图案
        logo(ax, **kwargs)
        
        # 保存图形到文件
        savefig(fig, "mark", bg)

        # ------------------------------------------------------------------------ #

        # 创建包含两个子图的图形对象
        fig, axs = plt.subplots(1, 2, figsize=(8, 2), dpi=100,
                                gridspec_kw=dict(width_ratios=[1, 3]))

        # 在第一个子图上绘制 logo 图案
        logo(axs[0], **kwargs)

        # 设置第二个子图的文本属性
        font = {
            "family": "avenir",
            "color": color,
            "weight": "regular",
            "size": 120,
        }
        axs[1].text(.01, .35, "seaborn", ha="left", va="center",
                    fontdict=font, transform=axs[1].transAxes)
        axs[1].set_axis_off()
        
        # 保存图形到文件
        savefig(fig, "wide", bg)

        # ------------------------------------------------------------------------ #

        # 创建包含两个子图的图形对象
        fig, axs = plt.subplots(2, 1, figsize=(2, 2.5), dpi=100,
                                gridspec_kw=dict(height_ratios=[4, 1]))

        # 在第一个子图上绘制 logo 图案
        logo(axs[0], **kwargs)

        # 设置第二个子图的文本属性
        font = {
            "family": "avenir",
            "color": color,
            "weight": "regular",
            "size": 34,
        }
        axs[1].text(.5, 1, "seaborn", ha="center", va="top",
                    fontdict=font, transform=axs[1].transAxes)
        axs[1].set_axis_off()
        
        # 保存图形到文件
        savefig(fig, "tall", bg)
```