# `D:\src\scipysrc\scipy\scipy\special\_precompute\loggamma.py`

```
"""Precompute series coefficients for log-Gamma."""

# 尝试导入 mpmath 库，如果导入失败则忽略
try:
    import mpmath
except ImportError:
    pass


def stirling_series(N):
    # 设置 mpmath 的工作精度为100位小数
    with mpmath.workdps(100):
        # 计算斯特林级数的系数，存储在列表中
        coeffs = [mpmath.bernoulli(2*n)/(2*n*(2*n - 1)) for n in range(1, N + 1)]
    return coeffs


def taylor_series_at_1(N):
    coeffs = []
    # 设置 mpmath 的工作精度为100位小数
    with mpmath.workdps(100):
        # 添加泰勒级数在 x=1 处的首个系数 (-euler)
        coeffs.append(-mpmath.euler)
        # 计算余下的泰勒级数系数，并存储在列表中
        for n in range(2, N + 1):
            coeffs.append((-1)**n*mpmath.zeta(n)/n)
    return coeffs


def main():
    # 打印本脚本的文档字符串
    print(__doc__)
    print()
    # 计算斯特林级数的系数，每个系数转换为字符串并倒序输出
    stirling_coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0) for x in stirling_series(8)[::-1]]
    # 计算泰勒级数在 x=1 处的系数，每个系数转换为字符串并倒序输出
    taylor_coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0) for x in taylor_series_at_1(23)[::-1]]
    # 打印斯特林级数系数的标题
    print("Stirling series coefficients")
    print("----------------------------")
    # 打印斯特林级数系数，每个系数占一行
    print("\n".join(stirling_coeffs))
    print()
    # 打印泰勒级数系数的标题
    print("Taylor series coefficients")
    print("--------------------------")
    # 打印泰勒级数系数，每个系数占一行
    print("\n".join(taylor_coeffs))
    print()


if __name__ == '__main__':
    main()
```