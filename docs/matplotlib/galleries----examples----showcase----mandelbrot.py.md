# `D:\src\scipysrc\matplotlib\galleries\examples\showcase\mandelbrot.py`

```py
"""
===================================
Shaded & power normalized rendering
===================================

The Mandelbrot set rendering can be improved by using a normalized recount
associated with a power normalized colormap (gamma=0.3). Rendering can be
further enhanced thanks to shading.

The ``maxiter`` gives the precision of the computation. ``maxiter=200`` should
take a few seconds on most modern laptops.
"""

import numpy as np  # 导入 NumPy 库

def mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    X = np.linspace(xmin, xmax, xn).astype(np.float32)  # 生成 X 坐标轴的线性空间
    Y = np.linspace(ymin, ymax, yn).astype(np.float32)  # 生成 Y 坐标轴的线性空间
    C = X + Y[:, None] * 1j  # 构建复数 C，表示 Mandelbrot 集合中的各个点
    N = np.zeros_like(C, dtype=int)  # 创建与 C 相同形状的零数组，用于存储迭代次数
    Z = np.zeros_like(C)  # 创建与 C 相同形状的零数组，用于存储迭代过程中的 Z 值
    for n in range(maxiter):
        I = abs(Z) < horizon  # 找到 Z 值的模小于预定的 horizon 的位置索引
        N[I] = n  # 将迭代次数 n 赋给满足条件的位置
        Z[I] = Z[I]**2 + C[I]  # Mandelbrot 迭代公式
    N[N == maxiter-1] = 0  # 将迭代次数达到 maxiter 的点置为零，以避免过早收敛
    return Z, N  # 返回最终的 Z 值和迭代次数 N

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import colors

    xmin, xmax, xn = -2.25, +0.75, 3000 // 2  # X 轴范围和点数
    ymin, ymax, yn = -1.25, +1.25, 2500 // 2  # Y 轴范围和点数
    maxiter = 200  # 最大迭代次数
    horizon = 2.0 ** 40  # 逃逸判据
    log_horizon = np.log2(np.log(horizon))  # 计算逃逸判据的对数

    Z, N = mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon)

    # Normalized recount as explained in:
    # https://linas.org/art-gallery/escape/smooth.html
    # https://web.archive.org/web/20160331171238/https://www.ibm.com/developerworks/community/blogs/jfp/entry/My_Christmas_Gift?lang=en

    # This line will generate warnings for null values, but it is faster to
    # process them afterwards using the nan_to_num
    with np.errstate(invalid='ignore'):
        M = np.nan_to_num(N + 1 - np.log2(np.log(abs(Z))) + log_horizon)

    dpi = 72  # 设置图像 DPI
    width = 10  # 图像宽度
    height = 10 * yn / xn  # 图像高度，根据 X 和 Y 的点数比例确定
    fig = plt.figure(figsize=(width, height), dpi=dpi)  # 创建画布对象

    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)  # 添加子图，全屏显示

    # Shaded rendering
    light = colors.LightSource(azdeg=315, altdeg=10)  # 光照方向设置
    M = light.shade(M, cmap=plt.cm.hot, vert_exag=1.5,
                    norm=colors.PowerNorm(0.3), blend_mode='hsv')  # 对 M 进行阴影处理

    ax.imshow(M, extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")  # 在子图上显示处理后的 M 矩阵

    ax.set_xticks([])  # 隐藏 X 轴刻度
    ax.set_yticks([])  # 隐藏 Y 轴刻度

    # Some advertisement for matplotlib
    year = time.strftime("%Y")  # 获取当前年份
    text = ("The Mandelbrot fractal set\n"
            "Rendered with matplotlib %s, %s - https://matplotlib.org"
            % (matplotlib.__version__, year))  # 图像底部注释文本
    ax.text(xmin + .025, ymin + .025, text, color="white", fontsize=12, alpha=0.5)  # 在子图上添加文本信息

    plt.show()  # 显示图像
```