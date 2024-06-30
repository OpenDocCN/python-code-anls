# `D:\src\scipysrc\sympy\sympy\benchmarks\bench_symbench.py`

```
#!/usr/bin/env python
from sympy.core.random import random                     # 导入 random 函数
from sympy.core.numbers import (I, Integer, pi)          # 导入复数 I，整数 Integer，圆周率 pi
from sympy.core.symbol import Symbol                     # 导入符号变量 Symbol
from sympy.core.sympify import sympify                   # 导入 sympify 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数 sqrt
from sympy.functions.elementary.trigonometric import sin  # 导入正弦函数 sin
from sympy.polys.polytools import factor                # 导入多项式因式分解函数 factor
from sympy.simplify.simplify import simplify            # 导入简化函数 simplify
from sympy.abc import x, y, z                           # 导入符号变量 x, y, z
from timeit import default_timer as clock                # 导入计时器函数 default_timer as clock

def bench_R1():
    "real(f(f(f(f(f(f(f(f(f(f(i/2)))))))))))"
    def f(z):
        return sqrt(Integer(1)/3)*z**2 + I/3               # 定义函数 f(z) 返回表达式的实部
    f(f(f(f(f(f(f(f(f(f(I/2))))))))).as_real_imag()[0]    # 调用函数 f 十次并取其实部

def bench_R2():
    "Hermite polynomial hermite(15, y)"
    def hermite(n, y):
        if n == 1:
            return 2*y                                      # 返回一阶 Hermite 多项式
        if n == 0:
            return 1                                        # 返回零阶 Hermite 多项式
        return (2*y*hermite(n - 1, y) - 2*(n - 1)*hermite(n - 2, y)).expand()  # 返回 Hermite 多项式

    hermite(15, y)                                          # 计算 Hermite 多项式的 15 阶

def bench_R3():
    "a = [bool(f==f) for _ in range(10)]"
    f = x + y + z                                           # 定义符号表达式 f
    [bool(f == f) for _ in range(10)]                        # 创建包含 10 个布尔值的列表，每个元素值为 True

def bench_R4():
    # we don't have Tuples                                   # 不执行任何操作的占位函数

def bench_R5():
    "blowup(L, 8); L=uniq(L)"
    def blowup(L, n):
        for i in range(n):
            L.append( (L[i] + L[i + 1]) * L[i + 2] )          # 执行列表 L 的扩展操作

    def uniq(x):
        v = set(x)                                           # 创建集合 v 包含唯一元素
        return v

    L = [x, y, z]                                            # 创建列表 L 包含符号变量 x, y, z
    blowup(L, 8)                                              # 对列表 L 执行 blowup 操作
    L = uniq(L)                                               # 对列表 L 执行 uniq 操作

def bench_R6():
    "sum(simplify((x+sin(i))/x+(x-sin(i))/x) for i in range(100))"
    sum(simplify((x + sin(i))/x + (x - sin(i))/x) for i in range(100))  # 对简化后的表达式求和

def bench_R7():
    "[f.subs(x, random()) for _ in range(10**4)]"
    f = x**24 + 34*x**12 + 45*x**3 + 9*x**18 + 34*x**10 + 32*x**21  # 定义多项式 f
    [f.subs(x, random()) for _ in range(10**4)]                    # 生成包含 10000 个随机数替换 x 后的列表

def bench_R8():
    "right(x^2,0,5,10^4)"
    def right(f, a, b, n):
        a = sympify(a)                                             # 将 a 转换为 SymPy 对象
        b = sympify(b)                                             # 将 b 转换为 SymPy 对象
        n = sympify(n)                                             # 将 n 转换为 SymPy 对象
        x = f.atoms(Symbol).pop()                                  # 获取函数 f 的符号变量
        Deltax = (b - a)/n                                          # 计算区间步长
        c = a                                                       # 初始化 c 为 a
        est = 0                                                     # 初始化估计值 est 为 0
        for i in range(n):
            c += Deltax                                              # 更新 c
            est += f.subs(x, c)                                       # 更新估计值 est
        return est*Deltax                                           # 返回估计积分值

    right(x**2, 0, 5, 10**4)                                        # 计算函数 x^2 在 [0, 5] 区间上的积分估计值

def _bench_R9():
    "factor(x^20 - pi^5*y^20)"
    factor(x**20 - pi**5*y**20)                                     # 对表达式进行因式分解

def bench_R10():
    "v = [-pi,-pi+1/10..,pi]"
    def srange(min, max, step):
        v = [min]                                                   # 初始化列表 v 包含 min
        while (max - v[-1]).evalf() > 0:                             # 循环直到 max - v[-1] 大于 0
            v.append(v[-1] + step)                                   # 将 step 加入 v 的末尾
        return v[:-1]                                                # 返回列表 v，去掉最后一个元素

    srange(-pi, pi, sympify(1)/10)                                   # 生成包含 -pi 到 pi 的数列，步长为 1/10

def bench_R11():
    "a = [random() + random()*I for w in [0..1000]]"
    [random() + random()*I for w in range(1000)]                     # 生成包含 1000 个随机复数的列表

def bench_S1():
    "e=(x+y+z+1)**7;f=e*(e+1);f.expand()"
    e = (x + y + z + 1)**7                                          # 计算表达式 e
    f = e*(e + 1)                                                    # 计算表达式 f
    f.expand()                                                       # 展开表达式 f

if __name__ == '__main__':
    benchmarks = [
        bench_R1,
        bench_R2,
        bench_R3,
        bench_R5,
        bench_R6,
        bench_R7,
        bench_R8,
        #_bench_R9,
        bench_R10,
        bench_R11,
        #bench_S1,
    ]

    report = []
    for b in benchmarks:
        t = clock()                                                 # 记录当前时间
        b()                                                         # 执行基准测试函数 b
        t = clock() - t                                             # 计算执行时间
        print("%s%65s: %f" % (b.__name__, b.__doc__, t))             # 输出基准测试函数名、文档字符串和执行时间
```