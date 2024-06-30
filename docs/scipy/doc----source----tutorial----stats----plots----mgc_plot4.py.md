# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\mgc_plot4.py`

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multiscale_graphcorr


def mgc_plot(x, y, mgc_dict):
    """Plot sim and MGC-plot"""
    # 创建一个 8x8 英寸大小的图像
    plt.figure(figsize=(8, 8))
    # 获取当前的图形轴对象
    ax = plt.gca()

    # 获取 MGC 字典中的本地相关性映射
    mgc_map = mgc_dict["mgc_map"]

    # 绘制热图
    ax.set_title("Local Correlation Map", fontsize=20)
    im = ax.imshow(mgc_map, cmap='YlGnBu')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    ax.invert_yaxis()

    # 关闭图的边框线，并创建白色网格
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    # 获取 MGC 字典中的最优尺度
    opt_scale = mgc_dict["opt_scale"]
    # 在图上绘制最优尺度点，使用红色的 X 标记，大小为 200
    ax.scatter(opt_scale[0], opt_scale[1],
               marker='X', s=200, color='red')

    # 其他格式设置
    ax.tick_params(bottom="off", left="off")
    ax.set_xlabel('#Neighbors for X', fontsize=15)
    ax.set_ylabel('#Neighbors for Y', fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)


# 创建一个随机数生成器对象
rng = np.random.default_rng()
# 生成均匀分布的随机数数组
unif = np.array(rng.uniform(0, 5, size=100))
# 根据公式生成 x 坐标
x = unif * np.cos(np.pi * unif)
# 根据公式生成 y 坐标，并添加一些随机扰动
y = unif * np.sin(np.pi * unif) + 0.4 * rng.random(x.size)

# 调用多尺度图形相关性分析函数，获取结果
_, _, mgc_dict = multiscale_graphcorr(x, y, random_state=rng)
# 调用绘图函数，绘制 MGC 相关的图像
mgc_plot(x, y, mgc_dict)
```