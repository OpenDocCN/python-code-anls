# `D:\src\scipysrc\scipy\scipy\special\_precompute\wright_bessel_data.py`

```
"""Compute a grid of values for Wright's generalized Bessel function
and save the values to data files for use in tests. Using mpmath directly in
tests would take too long.

This takes about 10 minutes to run on a 2.7 GHz i7 Macbook Pro.
"""
# 导入 functools 模块中的 lru_cache 装饰器，用于缓存函数调用结果
from functools import lru_cache
# 导入 os 模块，用于操作文件路径
import os
# 导入 time 模块中的 time 函数，用于计时
from time import time

# 导入 numpy 库，并将其中的 float 类型的 eps 常量扩展 100 倍，存入 eps 变量
import numpy as np
# 导入 scipy.special._mptestutils 模块中的 mpf2float 函数
from scipy.special._mptestutils import mpf2float

try:
    # 尝试导入 mpmath 库，如果失败则不处理
    import mpmath as mp
except ImportError:
    pass

# exp_inf: smallest value x for which exp(x) == inf
# 定义常量 exp_inf，表示 exp(x) == inf 的最小 x 值
exp_inf = 709.78271289338403


# 64 Byte per value
# 使用 lru_cache 装饰器，缓存 rgamma_cached 函数的调用结果，最多缓存 100,000 个调用
@lru_cache(maxsize=100_000)
def rgamma_cached(x, dps):
    # 使用 mpmath 库设置精度为 dps，计算 gamma 函数的值并返回
    with mp.workdps(dps):
        return mp.rgamma(x)


def mp_wright_bessel(a, b, x, dps=50, maxterms=2000):
    """Compute Wright's generalized Bessel function as Series with mpmath.
    """
    # 使用 mpmath 库设置精度为 dps
    with mp.workdps(dps):
        # 将输入的 a, b, x 转换为 mpmath 中的 mpf 类型
        a, b, x = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        # 计算 Wright's generalized Bessel 函数的级数表示，并返回其浮点数值
        res = mp.nsum(lambda k: x**k / mp.fac(k)
                      * rgamma_cached(a * k + b, dps=dps),
                      [0, mp.inf],
                      tol=dps, method='s', steps=[maxterms]
                      )
        return mpf2float(res)


def main():
    t0 = time()  # 记录程序开始运行的时间戳
    print(__doc__)  # 打印当前脚本的文档字符串
    pwd = os.path.dirname(__file__)  # 获取当前脚本文件所在的目录路径
    eps = np.finfo(float).eps * 100  # 计算 float 类型的机器精度，并扩展 100 倍

    # 定义不同参数范围的数组 a_range, b_range, x_range
    a_range = np.array([eps,
                        1e-4 * (1 - eps), 1e-4, 1e-4 * (1 + eps),
                        1e-3 * (1 - eps), 1e-3, 1e-3 * (1 + eps),
                        0.1, 0.5,
                        1 * (1 - eps), 1, 1 * (1 + eps),
                        1.5, 2, 4.999, 5, 10])
    b_range = np.array([0, eps, 1e-10, 1e-5, 0.1, 1, 2, 10, 20, 100])
    x_range = np.array([0, eps, 1 - eps, 1, 1 + eps,
                        1.5,
                        2 - eps, 2, 2 + eps,
                        9 - eps, 9, 9 + eps,
                        10 * (1 - eps), 10, 10 * (1 + eps),
                        100 * (1 - eps), 100, 100 * (1 + eps),
                        500, exp_inf, 1e3, 1e5, 1e10, 1e20])

    # 创建参数范围的网格
    a_range, b_range, x_range = np.meshgrid(a_range, b_range, x_range,
                                            indexing='ij')
    a_range = a_range.flatten()  # 将 a_range 扁平化为一维数组
    b_range = b_range.flatten()  # 将 b_range 扁平化为一维数组
    x_range = x_range.flatten()  # 将 x_range 扁平化为一维数组

    # 过滤掉部分数值，特别是过大的 x 值
    bool_filter = ~((a_range < 5e-3) & (x_range >= exp_inf))
    bool_filter = bool_filter & ~((a_range < 0.2) & (x_range > exp_inf))
    bool_filter = bool_filter & ~((a_range < 0.5) & (x_range > 1e3))
    bool_filter = bool_filter & ~((a_range < 0.56) & (x_range > 5e3))
    bool_filter = bool_filter & ~((a_range < 1) & (x_range > 1e4))
    bool_filter = bool_filter & ~((a_range < 1.4) & (x_range > 1e5))
    bool_filter = bool_filter & ~((a_range < 1.8) & (x_range > 1e6))
    bool_filter = bool_filter & ~((a_range < 2.2) & (x_range > 1e7))
    bool_filter = bool_filter & ~((a_range < 2.5) & (x_range > 1e8))
    bool_filter = bool_filter & ~((a_range < 2.9) & (x_range > 1e9))
    # 根据条件过滤布尔数组，排除不符合要求的数据点
    bool_filter = bool_filter & ~((a_range < 3.3) & (x_range > 1e10))
    bool_filter = bool_filter & ~((a_range < 3.7) & (x_range > 1e11))
    bool_filter = bool_filter & ~((a_range < 4) & (x_range > 1e12))
    bool_filter = bool_filter & ~((a_range < 4.4) & (x_range > 1e13))
    bool_filter = bool_filter & ~((a_range < 4.7) & (x_range > 1e14))
    bool_filter = bool_filter & ~((a_range < 5.1) & (x_range > 1e15))
    bool_filter = bool_filter & ~((a_range < 5.4) & (x_range > 1e16))
    bool_filter = bool_filter & ~((a_range < 5.8) & (x_range > 1e17))
    bool_filter = bool_filter & ~((a_range < 6.2) & (x_range > 1e18))
    bool_filter = bool_filter & ~((a_range < 6.2) & (x_range > 1e18))
    bool_filter = bool_filter & ~((a_range < 6.5) & (x_range > 1e19))
    bool_filter = bool_filter & ~((a_range < 6.9) & (x_range > 1e20))

    # 过滤出已知数值，这些数值不符合所需的数值精度要求
    # 参见测试 test_wright_data_grid_failures
    failing = np.array([
        [0.1, 100, 709.7827128933841],
        [0.5, 10, 709.7827128933841],
        [0.5, 10, 1000],
        [0.5, 100, 1000],
        [1, 20, 100000],
        [1, 100, 100000],
        [1.0000000000000222, 20, 100000],
        [1.0000000000000222, 100, 100000],
        [1.5, 0, 500],
        [1.5, 2.220446049250313e-14, 500],
        [1.5, 1.e-10, 500],
        [1.5, 1.e-05, 500],
        [1.5, 0.1, 500],
        [1.5, 20, 100000],
        [1.5, 100, 100000],
        ]).tolist()

    # 创建一个布尔数组，标记是否存在测试失败的数据点
    does_fail = np.full_like(a_range, False, dtype=bool)
    for i in range(x_range.size):
        if [a_range[i], b_range[i], x_range[i]] in failing:
            does_fail[i] = True

    # 进行过滤和压平操作
    a_range = a_range[bool_filter]
    b_range = b_range[bool_filter]
    x_range = x_range[bool_filter]
    does_fail = does_fail[bool_filter]

    dataset = []
    # 输出计算单点的数量
    print(f"Computing {x_range.size} single points.")
    print("Tests will fail for the following data points:")
    for i in range(x_range.size):
        a = a_range[i]
        b = b_range[i]
        x = x_range[i]
        # 处理一些特殊的边界情况
        maxterms = 1000
        if a < 1e-6 and x >= exp_inf/10:
            maxterms = 2000
        # 调用 mp_wright_bessel 函数计算值 f
        f = mp_wright_bessel(a, b, x, maxterms=maxterms)
        if does_fail[i]:
            # 输出测试失败的数据点及其数值
            print("failing data point a, b, x, value = "
                  f"[{a}, {b}, {x}, {f}]")
        else:
            dataset.append((a, b, x, f))
    dataset = np.array(dataset)

    # 定义文件名并保存数据集到文本文件
    filename = os.path.join(pwd, '..', 'tests', 'data', 'local',
                            'wright_bessel.txt')
    np.savetxt(filename, dataset)

    # 输出运行时间
    print(f"{(time() - t0)/60:.1f} minutes elapsed")
if __name__ == "__main__":
    # 如果当前模块被直接执行（而不是被导入到其他模块），则执行以下代码
    main()
```