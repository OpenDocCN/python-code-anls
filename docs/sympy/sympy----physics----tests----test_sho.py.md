# `D:\src\scipysrc\sympy\sympy\physics\tests\test_sho.py`

```
# 从 sympy.core 模块中导入 symbols（符号变量）、Rational（有理数表示）、Function（函数）、diff（求导函数）
# 从 sympy.physics.sho 模块中导入 R_nl（径向波函数）、E_nl（能级能量）
# 从 sympy.simplify.simplify 模块中导入 simplify（简化表达式函数）
from sympy.core import symbols, Rational, Function, diff
from sympy.physics.sho import R_nl, E_nl
from sympy.simplify.simplify import simplify

# 定义测试函数 test_sho_R_nl，用于验证径向波函数 R_nl 是否符合薛定谔方程
def test_sho_R_nl():
    # 定义符号变量 omega 和 r
    omega, r = symbols('omega r')
    # 定义符号变量 l，限定为整数
    l = symbols('l', integer=True)
    # 定义函数 u
    u = Function('u')

    # 遍历前 5 个量子数 n，验证径向波函数 R_nl 是否满足薛定谔方程
    for n in range(5):
        # 构造薛定谔方程
        schreq = ( -diff(u(r), r, 2)/2 + ((l*(l + 1))/(2*r**2)
                    + omega**2*r**2/2 - E_nl(n, l, omega))*u(r) )
        # 将波函数 u(r) 替换为径向波函数 R_nl，并计算薛定谔方程的结果
        result = schreq.subs(u(r), r*R_nl(n, l, omega/2, r))
        # 断言薛定谔方程的结果经过化简后应为 0
        assert simplify(result.doit()) == 0

# 定义测试函数 test_energy，用于验证能级能量 E_nl 的计算是否正确
def test_energy():
    # 定义符号变量 n、l 和 hw
    n, l, hw = symbols('n l hw')
    # 断言能级能量 E_nl 的计算结果与表达式 (2*n + l + 3/2)*hw 化简后应为 0
    assert simplify(E_nl(n, l, hw) - (2*n + l + Rational(3, 2))*hw) == 0
```