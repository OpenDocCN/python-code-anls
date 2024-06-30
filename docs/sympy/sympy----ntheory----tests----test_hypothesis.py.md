# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_hypothesis.py`

```
# 导入假设测试需要的模块和函数
from hypothesis import given
from hypothesis import strategies as st
# 导入 sympy 库中的函数和类
from sympy import divisors
from sympy.functions.combinatorial.numbers import divisor_sigma, totient
from sympy.ntheory.primetest import is_square

# 定义一个假设测试函数，测试 tau 函数的假设
@given(n=st.integers(1, 10**10))
def test_tau_hypothesis(n):
    # 获取 n 的所有正因子
    div = divisors(n)
    # 计算 n 的正因子个数 tau(n)
    tau_n = len(div)
    # 断言：n 是否为完全平方数的判断
    assert is_square(n) == (tau_n % 2 == 1)
    # 计算每个正因子的除数和 sigma 函数值
    sigmas = [divisor_sigma(i) for i in div]
    # 计算每个正因子的欧拉函数值
    totients = [totient(n // i) for i in div]
    # 计算 sigma 函数值和欧拉函数值的乘积列表
    mul = [a * b for a, b in zip(sigmas, totients)]
    # 断言：n * tau(n) 是否等于乘积列表的总和
    assert n * tau_n == sum(mul)

# 定义一个假设测试函数，测试欧拉函数假设
@given(n=st.integers(1, 10**10))
def test_totient_hypothesis(n):
    # 断言：欧拉函数值不大于 n
    assert totient(n) <= n
    # 获取 n 的所有正因子
    div = divisors(n)
    # 计算每个正因子的欧拉函数值列表
    totients = [totient(i) for i in div]
    # 断言：n 是否等于欧拉函数值列表的总和
    assert n == sum(totients)
```