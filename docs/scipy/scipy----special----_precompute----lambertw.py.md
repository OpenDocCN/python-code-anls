# `D:\src\scipysrc\scipy\scipy\special\_precompute\lambertw.py`

```
"""Compute a Pade approximation for the principal branch of the
Lambert W function around 0 and compare it to various other
approximations.

"""
# 导入必要的库
import numpy as np

try:
    import mpmath  # 导入mpmath库
    import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库
except ImportError:
    pass  # 如果导入失败，继续执行


def lambertw_pade():
    # 计算 Lambert W 函数在0点的前6阶导数
    derivs = [mpmath.diff(mpmath.lambertw, 0, n=n) for n in range(6)]
    # 对前6阶导数进行 Pade逼近
    p, q = mpmath.pade(derivs, 3, 2)
    return p, q  # 返回逼近得到的分子p和分母q


def main():
    print(__doc__)  # 打印脚本开头的文档字符串
    with mpmath.workdps(50):  # 设置mpmath的工作精度为50
        p, q = lambertw_pade()  # 计算 Pade逼近的分子p和分母q
        p, q = p[::-1], q[::-1]  # 将分子p和分母q反转（为了与polyval函数兼容）
        print(f"p = {p}")  # 打印分子p
        print(f"q = {q}")  # 打印分母q

    x, y = np.linspace(-1.5, 1.5, 75), np.linspace(-1.5, 1.5, 75)
    x, y = np.meshgrid(x, y)
    z = x + 1j*y  # 创建复数网格z

    lambertw_std = []
    # 计算 Lambert W 函数在复数网格z上的标准值
    for z0 in z.flatten():
        lambertw_std.append(complex(mpmath.lambertw(z0)))
    lambertw_std = np.array(lambertw_std).reshape(x.shape)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    
    # 比较 Pade逼近 与真实结果的误差
    p = np.array([float(p0) for p0 in p])
    q = np.array([float(q0) for q0 in q])
    pade_approx = np.polyval(p, z)/np.polyval(q, z)
    pade_err = abs(pade_approx - lambertw_std)
    axes[0].pcolormesh(x, y, pade_err)
    
    # 比较渐近级数的前两项与真实结果的误差
    asy_approx = np.log(z) - np.log(np.log(z))
    asy_err = abs(asy_approx - lambertw_std)
    axes[1].pcolormesh(x, y, asy_err)
    
    # 比较围绕分支点的级数的前两项与真实结果的误差
    p = np.sqrt(2*(np.exp(1)*z + 1))
    series_approx = -1 + p - p**2/3
    series_err = abs(series_approx - lambertw_std)
    im = axes[2].pcolormesh(x, y, series_err)

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    # 显示 Pade逼近 是否优于渐近级数的比较
    pade_better = pade_err < asy_err
    im = ax.pcolormesh(x, y, pade_better)
    t = np.linspace(-0.3, 0.3)
    ax.plot(-2.5*abs(t) - 0.2, t, 'r')
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
```