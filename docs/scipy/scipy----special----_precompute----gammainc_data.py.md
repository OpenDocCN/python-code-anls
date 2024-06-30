# `D:\src\scipysrc\scipy\scipy\special\_precompute\gammainc_data.py`

```
"""
Compute gammainc and gammaincc for large arguments and parameters
and save the values to data files for use in tests. We can't just
compare to mpmath's gammainc in test_mpmath.TestSystematic because it
would take too long.

Note that mpmath's gammainc is computed using hypercomb, but since it
doesn't allow the user to increase the maximum number of terms used in
the series it doesn't converge for many arguments. To get around this
we copy the mpmath implementation but use more terms.

This takes about 17 minutes to run on a 2.3 GHz Macbook Pro with 4GB
ram.

Sources:
[1] Fredrik Johansson and others. mpmath: a Python library for
    arbitrary-precision floating-point arithmetic (version 0.19),
    December 2013. http://mpmath.org/.
"""

import os                             # 导入操作系统功能模块
from time import time                 # 从时间模块导入时间函数
import numpy as np                    # 导入 NumPy 库，并用 np 别名表示
from numpy import pi                  # 导入 pi 常数

from scipy.special._mptestutils import mpf2float  # 导入特殊科学函数模块中的 mpf2float 函数

try:
    import mpmath as mp               # 尝试导入 mpmath 库，并使用 mp 别名表示
except ImportError:
    pass                              # 如果导入失败，则忽略该错误


def gammainc(a, x, dps=50, maxterms=10**8):
    """Compute gammainc exactly like mpmath does but allow for more
    summands in hypercomb. See

    mpmath/functions/expintegrals.py#L134

    in the mpmath github repository.

    """
    with mp.workdps(dps):             # 设置工作精度为 dps
        z, a, b = mp.mpf(a), mp.mpf(x), mp.mpf(x)  # 将 a, x 转换为 mpmath 的 mpf 类型
        G = [z]                        # 初始化 G 为包含 z 的列表
        negb = mp.fneg(b, exact=True)  # 计算 b 的负值

        def h(z):
            T1 = [mp.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
            return (T1,)               # 返回超函数组合 h(z)

        res = mp.hypercomb(h, [z], maxterms=maxterms)  # 调用 mpmath 的超函数组合计算
        return mpf2float(res)          # 将 mpmath 的结果转换为浮点数


def gammaincc(a, x, dps=50, maxterms=10**8):
    """Compute gammaincc exactly like mpmath does but allow for more
    terms in hypercomb. See

    mpmath/functions/expintegrals.py#L187

    in the mpmath github repository.

    """
    with mp.workdps(dps):             # 设置工作精度为 dps
        z, a = a, x                    # 直接赋值 z 和 a
        if mp.isint(z):                # 判断 z 是否为整数
            try:
                # mpmath has a fast integer path
                return mpf2float(mp.gammainc(z, a=a, regularized=True))  # 调用 mpmath 的 gammainc 函数
            except mp.libmp.NoConvergence:
                pass                    # 如果计算失败，则继续执行

        nega = mp.fneg(a, exact=True)  # 计算 a 的负值
        G = [z]                        # 初始化 G 为包含 z 的列表

        try:
            def h(z):
                r = z-1
                return [([mp.exp(nega), a], [1, r], [], G, [1, -r], [], 1/nega)]  # 返回超函数组合 h(z)
            return mpf2float(mp.hypercomb(h, [z], force_series=True))  # 调用 mpmath 的超函数组合计算
        except mp.libmp.NoConvergence:
            def h(z):
                T1 = [], [1, z-1], [z], G, [], [], 0
                T2 = [-mp.exp(nega), a, z], [1, z, -1], [], G, [1], [1+z], a
                return T1, T2        # 返回超函数组合 h(z)

            return mpf2float(mp.hypercomb(h, [z], maxterms=maxterms))  # 调用 mpmath 的超函数组合计算


def main():
    t0 = time()                       # 记录当前时间
    # It would be nice to have data for larger values, but either this
    # requires prohibitively large precision (dps > 800) or mpmath has
    # a bug. For example, gammainc(1e20, 1e20, dps=800) returns a
    # value around 0.03, while the true value should be close to 0.5
    # (DLMF 8.12.15).
    print(__doc__)                    # 打印当前文件的文档字符串
    # 获取当前文件的目录路径
    pwd = os.path.dirname(__file__)
    # 在对数空间内生成30个均匀间隔的数值，从10^4到10^14
    r = np.logspace(4, 14, 30)
    # 在对数空间内生成30个均匀间隔的数值，从pi/4到arctan(0.6)
    ltheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(0.6)), 30)
    # 在对数空间内生成30个均匀间隔的数值，从pi/4到arctan(1.4)
    utheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(1.4)), 30)
    
    # 定义两个函数和它们对应的角度范围
    regimes = [(gammainc, ltheta), (gammaincc, utheta)]
    for func, theta in regimes:
        # 创建一个网格，其中rg为r和theta的网格，thetag为theta的网格
        rg, thetag = np.meshgrid(r, theta)
        # 根据极坐标转换为直角坐标系的坐标a和x
        a, x = rg*np.cos(thetag), rg*np.sin(thetag)
        # 将a和x展平为一维数组
        a, x = a.flatten(), x.flatten()
        dataset = []
        # 遍历每个坐标对(a0, x0)，计算对应的函数值
        for i, (a0, x0) in enumerate(zip(a, x)):
            if func == gammaincc:
                # 如果函数是gammaincc，则利用其快速整数路径，以避免计算时间过长
                a0, x0 = np.floor(a0), np.floor(x0)
            # 将计算结果(a0, x0, func(a0, x0))添加到数据集中
            dataset.append((a0, x0, func(a0, x0)))
        # 将数据集转换为NumPy数组
        dataset = np.array(dataset)
        # 构建保存文件的完整路径和文件名
        filename = os.path.join(pwd, '..', 'tests', 'data', 'local',
                                f'{func.__name__}.txt')
        # 将数据集保存到文本文件中
        np.savetxt(filename, dataset)
    
    # 计算程序运行时间并打印出经过的分钟数
    print(f"{(time() - t0)/60} minutes elapsed")
# 如果当前脚本作为主程序运行（而不是被导入到其他模块），则执行下面的代码
if __name__ == "__main__":
    # 调用主程序的入口函数 main()
    main()
```