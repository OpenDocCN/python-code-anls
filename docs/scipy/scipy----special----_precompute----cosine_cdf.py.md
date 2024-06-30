# `D:\src\scipysrc\scipy\scipy\special\_precompute\cosine_cdf.py`

```
import mpmath  # 导入 mpmath 库，用于高精度数学运算


def f(x):
    return (mpmath.pi + x + mpmath.sin(x)) / (2*mpmath.pi)
    # 定义函数 f(x)，计算表达式 (π + x + sin(x)) / (2π)


# Note: 40 digits might be overkill; a few more digits than the default
# might be sufficient.
mpmath.mp.dps = 40  # 设置 mpmath 的精度为 40 位小数
ts = mpmath.taylor(f, -mpmath.pi, 20)  # 计算函数 f 在 -π 处展开的泰勒级数，展开到 20 阶
p, q = mpmath.pade(ts, 9, 10)  # 计算泰勒级数的 (9, 10) 佩德逊近似

p = [float(c) for c in p]  # 将 p 中的每个元素转换为浮点数
q = [float(c) for c in q]  # 将 q 中的每个元素转换为浮点数
print('p =', p)  # 输出 p 的值
print('q =', q)  # 输出 q 的值
```