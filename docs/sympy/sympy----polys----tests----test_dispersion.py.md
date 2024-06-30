# `D:\src\scipysrc\sympy\sympy\polys\tests\test_dispersion.py`

```
from sympy.core import Symbol, S, oo  # 导入符号、常量和无穷大
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.polys import poly  # 导入多项式对象
from sympy.polys.dispersion import dispersion, dispersionset  # 导入离散集合和离散度函数


def test_dispersion():
    x = Symbol("x")  # 创建符号变量 x
    a = Symbol("a")  # 创建符号变量 a

    fp = poly(S.Zero, x)  # 创建多项式对象 fp = 0
    assert sorted(dispersionset(fp)) == [0]  # 断言离散集合为 [0]

    fp = poly(S(2), x)  # 创建多项式对象 fp = 2
    assert sorted(dispersionset(fp)) == [0]  # 断言离散集合为 [0]

    fp = poly(x + 1, x)  # 创建多项式对象 fp = x + 1
    assert sorted(dispersionset(fp)) == [0]  # 断言离散集合为 [0]
    assert dispersion(fp) == 0  # 断言离散度为 0

    fp = poly((x + 1)*(x + 2), x)  # 创建多项式对象 fp = (x + 1)*(x + 2)
    assert sorted(dispersionset(fp)) == [0, 1]  # 断言离散集合为 [0, 1]
    assert dispersion(fp) == 1  # 断言离散度为 1

    fp = poly(x*(x + 3), x)  # 创建多项式对象 fp = x*(x + 3)
    assert sorted(dispersionset(fp)) == [0, 3]  # 断言离散集合为 [0, 3]
    assert dispersion(fp) == 3  # 断言离散度为 3

    fp = poly((x - 3)*(x + 3), x)  # 创建多项式对象 fp = (x - 3)*(x + 3)
    assert sorted(dispersionset(fp)) == [0, 6]  # 断言离散集合为 [0, 6]
    assert dispersion(fp) == 6  # 断言离散度为 6

    fp = poly(x**4 - 3*x**2 + 1, x)  # 创建多项式对象 fp = x**4 - 3*x**2 + 1
    gp = fp.shift(-3)  # 平移 fp 多项式 -3
    assert sorted(dispersionset(fp, gp)) == [2, 3, 4]  # 断言离散集合为 [2, 3, 4]
    assert dispersion(fp, gp) == 4  # 断言离散度为 4
    assert sorted(dispersionset(gp, fp)) == []  # 断言离散集合为空集
    assert dispersion(gp, fp) is -oo  # 断言离散度为负无穷大

    fp = poly(x*(3*x**2+a)*(x-2536)*(x**3+a), x)  # 创建多项式对象 fp
    gp = fp.as_expr().subs(x, x-345).as_poly(x)  # 将 fp 替换后的表达式转换为多项式 gp
    assert sorted(dispersionset(fp, gp)) == [345, 2881]  # 断言离散集合为 [345, 2881]
    assert sorted(dispersionset(gp, fp)) == [2191]  # 断言离散集合为 [2191]

    gp = poly((x-2)**2*(x-3)**3*(x-5)**3, x)  # 创建多项式对象 gp
    assert sorted(dispersionset(gp)) == [0, 1, 2, 3]  # 断言离散集合为 [0, 1, 2, 3]
    assert sorted(dispersionset(gp, (gp+4)**2)) == [1, 2]  # 断言离散集合为 [1, 2]

    fp = poly(x*(x+2)*(x-1), x)  # 创建多项式对象 fp = x*(x+2)*(x-1)
    assert sorted(dispersionset(fp)) == [0, 1, 2, 3]  # 断言离散集合为 [0, 1, 2, 3]

    fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')  # 创建多项式对象 fp
    gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')  # 创建多项式对象 gp
    assert sorted(dispersionset(fp, gp)) == [2]  # 断言离散集合为 [2]
    assert sorted(dispersionset(gp, fp)) == [1, 4]  # 断言离散集合为 [1, 4]

    # 计算在 Z[a] 而不是简单的 Z 中 alpha 是否为整数，存在困难
    fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)  # 创建多项式对象 fp
    assert sorted(dispersionset(fp)) == [0, 1]  # 断言离散集合为 [0, 1]

    # 对于特定的 a 值，离散度为 3*a，但算法通常无法一般性地找到这个结果
    # 这是结果为基础的方法比当前方法更优越的地方
    fp = poly(a**2*x**3 + (a**3 + a**2 + a + 1)*x, x)  # 创建多项式对象 fp
    gp = fp.as_expr().subs(x, x - 3*a).as_poly(x)  # 将 fp 替换后的表达式转换为多项式 gp
    assert sorted(dispersionset(fp, gp)) == []  # 断言离散集合为空集

    fpa = fp.as_expr().subs(a, 2).as_poly(x)  # 将 fp 替换 a 后的表达式转换为多项式 fpa
    gpa = gp.as_expr().subs(a, 2).as_poly(x)  # 将 gp 替换 a 后的表达式转换为多项式 gpa
    assert sorted(dispersionset(fpa, gpa)) == [6]  # 断言离散集合为 [6]

    # 使用表达式而不是多项式进行计算
    f = (x + 1)*(x + 2)  # 创建表达式 f = (x + 1)*(x + 2)
    assert sorted(dispersionset(f)) == [0, 1]  # 断言离散集合为 [0, 1]
    assert dispersion(f) == 1  # 断言离散度为 1

    f = x**4 - 3*x**2 + 1  # 创建表达式 f = x**4 - 3*x**2 + 1
    g = x**4 - 12*x**3 + 51*x**2 - 90*x + 55  # 创建表达式 g
    assert sorted(dispersionset(f, g)) == [2, 3, 4]  # 断言离散集合为 [2, 3, 4]
    assert dispersion(f, g) == 4  # 断言离散度为 4

    # 使用表达式并指定一个生成器
    f = (x + 1)*(x + 2)  # 创建表达式 f = (x + 1)*(x + 2)
    assert sorted(dispersionset(f, None, x)) == [0, 1]  # 断言离散集合为 [0, 1]
    # 断言调用 dispersion 函数，检查返回值是否为 1
    assert dispersion(f, None, x) == 1

    # 定义函数 f，代表 x 的四次方减去 3 倍 x 的平方加 1
    f = x**4 - 3*x**2 + 1
    # 定义函数 g，代表 x 的四次方减去 12 倍 x 的立方加 51 倍 x 的平方减去 90 倍 x 加 55
    g = x**4 - 12*x**3 + 51*x**2 - 90*x + 55
    # 断言排序后的 dispersionset 函数返回结果是否与预期列表 [2, 3, 4] 相等
    assert sorted(dispersionset(f, g, x)) == [2, 3, 4]
    # 断言 dispersion 函数返回值是否为 4
    assert dispersion(f, g, x) == 4
```