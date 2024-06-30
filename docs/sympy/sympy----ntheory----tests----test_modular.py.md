# `D:\src\scipysrc\sympy\sympy\ntheory\tests\test_modular.py`

```
# 导入所需的模块和函数
from sympy.ntheory.modular import crt, crt1, crt2, solve_congruence
from sympy.testing.pytest import raises


# 定义测试函数 test_crt，用于测试 Chinese Remainder Theorem 相关函数
def test_crt():
    # 定义内部函数 mcrt，用于测试 crt、crt1 和 crt2 函数的组合情况
    def mcrt(m, v, r, symmetric=False):
        # 断言使用 crt 函数得到的结果的第一个元素与预期结果 r 相等
        assert crt(m, v, symmetric)[0] == r
        # 使用 crt1 函数获得 mm、e、s 三个返回值
        mm, e, s = crt1(m)
        # 断言使用 crt2 函数得到的结果与预期结果 (r, mm) 相等
        assert crt2(m, v, mm, e, s, symmetric) == (r, mm)

    # 测试不同输入情况下的 mcrt 函数
    mcrt([2, 3, 5], [0, 0, 0], 0)
    mcrt([2, 3, 5], [1, 1, 1], 1)

    mcrt([2, 3, 5], [-1, -1, -1], -1, True)
    mcrt([2, 3, 5], [-1, -1, -1], 2*3*5 - 1, False)

    # 断言使用 crt 函数得到的结果与预期结果 (-56917, 114800) 相等
    assert crt([656, 350], [811, 133], symmetric=True) == (-56917, 114800)


# 定义测试函数 test_modular，用于测试 solve_congruence 函数
def test_modular():
    # 断言 solve_congruence 函数处理多个模同余方程的结果与预期结果 (1719, 7140) 相等
    assert solve_congruence(*list(zip([3, 4, 2], [12, 35, 17]))) == (1719, 7140)
    # 断言 solve_congruence 函数对无解情况的返回值为 None
    assert solve_congruence(*list(zip([3, 4, 2], [12, 6, 17]))) is None
    # 断言 solve_congruence 函数处理多个模同余方程的结果与预期结果 (172, 1547) 相等
    assert solve_congruence(*list(zip([3, 4, 2], [13, 7, 17]))) == (172, 1547)
    # 断言 solve_congruence 函数处理负数模同余方程的结果与预期结果 (172, 1547) 相等
    assert solve_congruence(*list(zip([-10, -3, -15], [13, 7, 17]))) == (172, 1547)
    # 断言 solve_congruence 函数对无解情况的返回值为 None
    assert solve_congruence(*list(zip([-10, -3, 1, -15], [13, 7, 7, 17]))) is None
    # 断言 solve_congruence 函数处理多个模同余方程的结果与预期结果 (835, 1547) 相等
    assert solve_congruence(*list(zip([-10, -5, 2, -15], [13, 7, 7, 17]))) == (835, 1547)
    # 断言 solve_congruence 函数处理多个模同余方程的结果与预期结果 (2382, 3094) 相等
    assert solve_congruence(*list(zip([-10, -5, 2, -15], [13, 7, 14, 17]))) == (2382, 3094)
    # 断言 solve_congruence 函数处理多个模同余方程的结果与预期结果 (2382, 3094) 相等
    assert solve_congruence(*list(zip([-10, 2, 2, -15], [13, 7, 14, 17]))) == (2382, 3094)
    # 断言 solve_congruence 函数对无解情况的返回值为 None
    assert solve_congruence(*list(zip((1, 1, 2), (3, 2, 4)))) is None
    # 使用 raises 函数断言 solve_congruence 函数对错误输入的处理会抛出 ValueError 异常
    raises(
        ValueError, lambda: solve_congruence(*list(zip([3, 4, 2], [12.1, 35, 17])))
    )
```