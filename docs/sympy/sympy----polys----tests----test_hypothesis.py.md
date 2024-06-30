# `D:\src\scipysrc\sympy\sympy\polys\tests\test_hypothesis.py`

```
# 导入所需的模块和函数
from hypothesis import given  # 导入给定条件的测试框架
from hypothesis import strategies as st  # 导入策略模块，并简称为st
from sympy.abc import x  # 导入符号x，用于多项式的创建
from sympy.polys.polytools import Poly  # 导入多项式工具中的Poly类


def polys(*, nonzero=False, domain="ZZ"):
    # 定义生成多项式的策略函数，可以指定是否非零以及定义域
    elems = {"ZZ": st.integers(), "QQ": st.fractions()}  # 定义元素类型的字典
    coeff_st = st.lists(elems[domain])  # 根据指定域生成系数的策略
    if nonzero:
        coeff_st = coeff_st.filter(any)  # 如果要求非零多项式，则筛选非零系数
    return st.builds(Poly, coeff_st, st.just(x), domain=st.just(domain))  # 构建多项式对象的策略


@given(f=polys(), g=polys(), r=polys())
def test_gcd_hypothesis(f, g, r):
    # 测试多项式的最大公因子性质
    gcd_1 = f.gcd(g)  # 计算f和g的最大公因子
    gcd_2 = g.gcd(f)  # 计算g和f的最大公因子
    assert gcd_1 == gcd_2  # 断言两种计算方法得到的最大公因子应相等

    # 通过乘以r进行测试
    gcd_3 = g.gcd(f + r * g)  # 计算g和f + r*g的最大公因子
    assert gcd_1 == gcd_3  # 断言与之前计算得到的最大公因子相等


@given(f_z=polys(), g_z=polys(nonzero=True))
def test_poly_hypothesis_integers(f_z, g_z):
    # 测试整数域下的多项式性质
    remainder_z = f_z.rem(g_z)  # 计算f_z除以g_z的余数
    assert g_z.degree() >= remainder_z.degree() or remainder_z.degree() == 0  # 断言g_z的次数大于等于余数的次数或者余数的次数为0


@given(f_q=polys(domain="QQ"), g_q=polys(nonzero=True, domain="QQ"))
def test_poly_hypothesis_rationals(f_q, g_q):
    # 测试有理数域下的多项式性质
    remainder_q = f_q.rem(g_q)  # 计算f_q除以g_q的余数
    assert g_q.degree() >= remainder_q.degree() or remainder_q.degree() == 0  # 断言g_q的次数大于等于余数的次数或者余数的次数为0
```