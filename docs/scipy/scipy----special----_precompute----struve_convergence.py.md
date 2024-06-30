# `D:\src\scipysrc\scipy\scipy\special\_precompute\struve_convergence.py`

```
"""
Convergence regions of the expansions used in ``struve.c``

Note that for v >> z both functions tend rapidly to 0,
and for v << -z, they tend to infinity.

The floating-point functions over/underflow in the lower left and right
corners of the figure.


Figure legend
=============

Red region
    Power series is close (1e-12) to the mpmath result

Blue region
    Asymptotic series is close to the mpmath result

Green region
    Bessel series is close to the mpmath result

Dotted colored lines
    Boundaries of the regions

Solid colored lines
    Boundaries estimated by the routine itself. These will be used
    for determining which of the results to use.

Black dashed line
    The line z = 0.7*|v| + 12

"""
# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt

# 导入 mpmath 库用于精确计算
import mpmath

# 定义计算误差的函数
def err_metric(a, b, atol=1e-290):
    m = abs(a - b) / (atol + abs(b))
    m[np.isinf(b) & (a == b)] = 0
    return m

# 定义绘图函数，参数 is_h 表示是否为 Struve 函数的第一类或第二类
def do_plot(is_h=True):
    # 导入 scipy.special._ufuncs 中的函数
    from scipy.special._ufuncs import (_struve_power_series,
                                       _struve_asymp_large_z,
                                       _struve_bessel_series)

    # 生成待计算的 v 和 z 的数组
    vs = np.linspace(-1000, 1000, 91)
    zs = np.sort(np.r_[1e-5, 1.0, np.linspace(0, 700, 91)[1:]])

    # 计算三种不同方法的 Struve 函数值
    rp = _struve_power_series(vs[:,None], zs[None,:], is_h)
    ra = _struve_asymp_large_z(vs[:,None], zs[None,:], is_h)
    rb = _struve_bessel_series(vs[:,None], zs[None,:], is_h)

    # 设置 mpmath 的精度
    mpmath.mp.dps = 50
    # 定义用 mpmath 计算 Struve 函数的方法
    if is_h:
        def sh(v, z):
            return float(mpmath.struveh(mpmath.mpf(v), mpmath.mpf(z)))
    else:
        def sh(v, z):
            return float(mpmath.struvel(mpmath.mpf(v), mpmath.mpf(z)))
    # 使用 numpy 的 vectorize 函数将 mpmath 函数向量化，加快计算速度
    ex = np.vectorize(sh, otypes='d')(vs[:,None], zs[None,:])

    # 计算三种方法的误差
    err_a = err_metric(ra[0], ex) + 1e-300
    err_p = err_metric(rp[0], ex) + 1e-300
    err_b = err_metric(rb[0], ex) + 1e-300

    # 估计误差的比率
    err_est_a = abs(ra[1]/ra[0])
    err_est_p = abs(rp[1]/rp[0])
    err_est_b = abs(rb[1]/rb[0])

    # 设置 z 的截断值
    z_cutoff = 0.7*abs(vs) + 12

    # 设置绘图的水平线等级
    levels = [-1000, -12]

    # 清空当前图形
    plt.cla()

    # 保持绘图
    plt.hold(1)

    # 绘制填充的等高线图，表示不同误差范围
    plt.contourf(vs, zs, np.log10(err_p).T,
                 levels=levels, colors=['r', 'r'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_a).T,
                 levels=levels, colors=['b', 'b'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_b).T,
                 levels=levels, colors=['g', 'g'], alpha=0.1)

    # 绘制虚线的等高线，表示误差范围的边界
    plt.contour(vs, zs, np.log10(err_p).T,
                levels=levels, colors=['r', 'r'], linestyles=[':', ':'])
    plt.contour(vs, zs, np.log10(err_a).T,
                levels=levels, colors=['b', 'b'], linestyles=[':', ':'])
    plt.contour(vs, zs, np.log10(err_b).T,
                levels=levels, colors=['g', 'g'], linestyles=[':', ':'])

    # 绘制实线的等高线，表示误差估计的边界
    lp = plt.contour(vs, zs, np.log10(err_est_p).T,
                     levels=levels, colors=['r', 'r'], linestyles=['-', '-'])
    la = plt.contour(vs, zs, np.log10(err_est_a).T,
                     levels=levels, colors=['b', 'b'], linestyles=['-', '-'])
    # 创建等高线图并将其赋值给变量lb，绘制对数化的误差估计值
    lb = plt.contour(vs, zs, np.log10(err_est_b).T,
                     levels=levels, colors=['g', 'g'], linestyles=['-', '-'])

    # 添加等高线标签lp，格式化标签为{'-1000': 'P', '-12': 'P'}
    plt.clabel(lp, fmt={-1000: 'P', -12: 'P'})
    # 添加等高线标签la，格式化标签为{'-1000': 'A', '-12': 'A'}
    plt.clabel(la, fmt={-1000: 'A', -12: 'A'})
    # 添加等高线标签lb，格式化标签为{'-1000': 'B', '-12': 'B'}
    plt.clabel(lb, fmt={-1000: 'B', -12: 'B'})

    # 绘制黑色虚线，表示截止线
    plt.plot(vs, z_cutoff, 'k--')

    # 设置x轴范围，最小值为vs的最小值，最大值为vs的最大值
    plt.xlim(vs.min(), vs.max())
    # 设置y轴范围，最小值为zs的最小值，最大值为zs的最大值
    plt.ylim(zs.min(), zs.max())

    # 设置x轴标签为'v'
    plt.xlabel('v')
    # 设置y轴标签为'z'
    plt.ylabel('z')
# 主程序入口函数
def main():
    # 清除当前所有的图形
    plt.clf()
    
    # 在当前图形中创建一个子图，位置为1行2列中的第1个
    plt.subplot(121)
    
    # 调用函数 do_plot，并传入参数 True，进行绘图操作
    do_plot(True)
    
    # 设置当前子图的标题为 'Struve H'
    plt.title('Struve H')

    # 在当前图形中创建一个子图，位置为1行2列中的第2个
    plt.subplot(122)
    
    # 调用函数 do_plot，并传入参数 False，进行绘图操作
    do_plot(False)
    
    # 设置当前子图的标题为 'Struve L'
    plt.title('Struve L')

    # 将当前图形保存为文件 'struve_convergence.png'
    plt.savefig('struve_convergence.png')
    
    # 显示当前图形
    plt.show()


# 如果该脚本作为主程序运行，则执行 main() 函数
if __name__ == "__main__":
    main()
```