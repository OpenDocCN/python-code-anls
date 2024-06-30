# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\mgc_plot2.py`

```
# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multiscale_graphcorr

# 定义函数 mgc_plot，用于绘制相关图和 MGC 图
def mgc_plot(x, y, mgc_dict):
    """Plot sim and MGC-plot"""
    # 创建一个 8x8 英寸大小的图像
    plt.figure(figsize=(8, 8))
    # 获取当前的坐标轴
    ax = plt.gca()

    # 从 MGC 字典中获取局部相关性地图
    mgc_map = mgc_dict["mgc_map"]

    # 绘制热力图
    ax.set_title("Local Correlation Map", fontsize=20)
    im = ax.imshow(mgc_map, cmap='YlGnBu')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    ax.invert_yaxis()

    # 关闭坐标轴的边框线，并创建白色网格
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    # 获取最优尺度
    opt_scale = mgc_dict["opt_scale"]
    # 在图上绘制红色的 X 标记，表示最优尺度点
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

# 使用 NumPy 生成随机数生成器
rng = np.random.default_rng()
# 生成一组在 [-1, 1] 区间内等间距的 100 个数作为 x 轴数据
x = np.linspace(-1, 1, num=100)
# 根据 x 轴数据生成 y 轴数据，加入随机噪声
y = x + 0.3 * rng.random(x.size)

# 调用 multiscale_graphcorr 函数计算相关性分析，并获取返回的 MGC 字典
_, _, mgc_dict = multiscale_graphcorr(x, y, random_state=rng)
# 调用 mgc_plot 函数绘制 MGC 图
mgc_plot(x, y, mgc_dict)
```