# `D:\src\scipysrc\scipy\scipy\special\_precompute\zetac.py`

```
"""Compute the Taylor series for zeta(x) - 1 around x = 0."""
# 导入mpmath库，用于高精度数学计算
try:
    import mpmath
except ImportError:
    pass

# 计算 ζ(x) - 1 的泰勒级数展开，N 为展开的阶数
def zetac_series(N):
    # 系数列表
    coeffs = []
    
    # 设置工作精度为100位小数
    with mpmath.workdps(100):
        # 初始系数
        coeffs.append(-1.5)
        
        # 计算泰勒系数
        for n in range(1, N):
            # 计算 zeta(x) 在 x=0 处的 n 阶导数除以 n 阶阶乘
            coeff = mpmath.diff(mpmath.zeta, 0, n) / mpmath.factorial(n)
            coeffs.append(coeff)
    
    return coeffs

# 主函数
def main():
    # 打印本脚本的文档字符串
    print(__doc__)
    
    # 计算泰勒级数的系数，取 N=10
    coeffs = zetac_series(10)
    
    # 将系数列表中的每个系数转换为字符串，精度为20位小数
    coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0) for x in coeffs]
    
    # 逆序输出每个系数的字符串表示
    print("\n".join(coeffs[::-1]))

# 如果本脚本作为主程序运行，则调用主函数
if __name__ == '__main__':
    main()
```