# `D:\src\scipysrc\sympy\sympy\polys\tests\test_distributedmodules.py`

```
# 导入所需模块和函数
from sympy.polys.distributedmodules import (
    sdm_monomial_mul, sdm_monomial_deg, sdm_monomial_divides,
    sdm_add, sdm_LM, sdm_LT, sdm_mul_term, sdm_zero, sdm_deg,
    sdm_LC, sdm_from_dict,
    sdm_spoly, sdm_ecart, sdm_nf_mora, sdm_groebner,
    sdm_from_vector, sdm_to_vector, sdm_monomial_lcm
)

from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ

from sympy.abc import x, y, z

# 测试单个函数 sdm_monomial_mul
def test_sdm_monomial_mul():
    assert sdm_monomial_mul((1, 1, 0), (1, 3)) == (1, 2, 3)

# 测试单个函数 sdm_monomial_deg
def test_sdm_monomial_deg():
    assert sdm_monomial_deg((5, 2, 1)) == 3

# 测试单个函数 sdm_monomial_lcm
def test_sdm_monomial_lcm():
    assert sdm_monomial_lcm((1, 2, 3), (1, 5, 0)) == (1, 5, 3)

# 测试多参数的函数 sdm_monomial_divides
def test_sdm_monomial_divides():
    assert sdm_monomial_divides((1, 0, 0), (1, 0, 0)) is True
    assert sdm_monomial_divides((1, 0, 0), (1, 2, 1)) is True
    assert sdm_monomial_divides((5, 1, 1), (5, 2, 1)) is True

    assert sdm_monomial_divides((1, 0, 0), (2, 0, 0)) is False
    assert sdm_monomial_divides((1, 1, 0), (1, 0, 0)) is False
    assert sdm_monomial_divides((5, 1, 2), (5, 0, 1)) is False

# 测试函数 sdm_LC
def test_sdm_LC():
    assert sdm_LC([((1, 2, 3), QQ(5))], QQ) == QQ(5)

# 测试函数 sdm_from_dict
def test_sdm_from_dict():
    dic = {(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1), (1, 0, 2, 1): QQ(1),
           (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)}
    assert sdm_from_dict(dic, grlex) == \
        [((1, 2, 1, 1), QQ(1)), ((1, 1, 2, 1), QQ(1)),
         ((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))]

# 测试函数 sdm_add
def test_sdm_add():
    assert sdm_add([((1, 1, 1), QQ(1))], [((2, 0, 0), QQ(1))], lex, QQ) == \
        [((2, 0, 0), QQ(1)), ((1, 1, 1), QQ(1))]
    assert sdm_add([((1, 1, 1), QQ(1))], [((1, 1, 1), QQ(-1))], lex, QQ) == []
    assert sdm_add([((1, 0, 0), QQ(1))], [((1, 0, 0), QQ(2))], lex, QQ) == \
        [((1, 0, 0), QQ(3))]
    assert sdm_add([((1, 0, 1), QQ(1))], [((1, 1, 0), QQ(1))], lex, QQ) == \
        [((1, 1, 0), QQ(1)), ((1, 0, 1), QQ(1))]

# 测试函数 sdm_LM
def test_sdm_LM():
    dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(1), (4, 0, 1): QQ(1)}
    assert sdm_LM(sdm_from_dict(dic, lex)) == (4, 0, 1)

# 测试函数 sdm_LT
def test_sdm_LT():
    dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(2), (4, 0, 1): QQ(3)}
    assert sdm_LT(sdm_from_dict(dic, lex)) == ((4, 0, 1), QQ(3))

# 测试函数 sdm_mul_term
def test_sdm_mul_term():
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ) == []
    assert sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ) == []
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ) == \
        [((1, 1, 0), QQ(1))]
    f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    assert sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ) == \
        [((2, 1, 2), QQ(8)), ((1, 2, 1), QQ(6))]

# 测试函数 sdm_zero
def test_sdm_zero():
    assert sdm_zero() == []

# 测试函数 sdm_deg
def test_sdm_deg():
    assert sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)]) == 7

# 测试函数 sdm_spoly
def test_sdm_spoly():
    f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    # 定义一个包含单个元组的列表 g，元组中包含三个整数和一个有理数 QQ(1)
    g = [((2, 3, 0), QQ(1))]
    # 定义一个包含单个元组的列表 h，元组中包含三个整数和一个有理数 QQ(1)
    h = [((1, 2, 3), QQ(1))]
    # 断言调用 sdm_spoly 函数，传入 f, h, lex, QQ 参数后返回空列表
    assert sdm_spoly(f, h, lex, QQ) == []
    # 断言调用 sdm_spoly 函数，传入 f, g, lex, QQ 参数后返回包含一个元组的列表
    assert sdm_spoly(f, g, lex, QQ) == [((1, 2, 1), QQ(1))]
# 定义一个测试函数 test_sdm_ecart，用于测试 sdm_ecart 函数的正确性
def test_sdm_ecart():
    # 断言调用 sdm_ecart 函数并检查其返回值是否为 0
    assert sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)]) == 0
    # 断言调用 sdm_ecart 函数并检查其返回值是否为 3
    assert sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)]) == 3


# 定义一个测试函数 test_sdm_nf_mora，用于测试 sdm_nf_mora 函数的正确性
def test_sdm_nf_mora():
    # 调用 sdm_from_dict 函数创建多项式 f 和 f1, f2，使用 grlex 排序
    f = sdm_from_dict({(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1),
                       (1, 0, 2, 1): QQ(1), (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)},
                      grlex)
    f1 = sdm_from_dict({(1, 1, 1, 0): QQ(1), (1, 0, 2, 0): QQ(1),
                        (1, 0, 0, 0): QQ(-1)}, grlex)
    f2 = sdm_from_dict({(1, 1, 1, 0): QQ(1)}, grlex)
    # 创建 id0, id1, id2 三个多项式
    (id0, id1, id2) = [sdm_from_dict({(i, 0, 0, 0): QQ(1)}, grlex)
                       for i in range(3)]

    # 断言调用 sdm_nf_mora 函数并检查其返回值是否符合预期
    assert sdm_nf_mora(f, [f1, f2], grlex, QQ, phantom=(id0, [id1, id2])) == \
        ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1)),
          ((1, 1, 0, 1), QQ(1))],
         [((1, 1, 0, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])
    assert sdm_nf_mora(f, [f2, f1], grlex, QQ, phantom=(id0, [id2, id1])) == \
        ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))],
         [((2, 1, 0, 1), QQ(-1)), ((2, 0, 1, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])

    # 使用 sdm_from_vector 函数创建多项式 f, f1, f2，使用 lex 排序
    f = sdm_from_vector([x*z, y**2 + y*z - z, y], lex, QQ, gens=[x, y, z])
    f1 = sdm_from_vector([x, y, 1], lex, QQ, gens=[x, y, z])
    f2 = sdm_from_vector([x*y, z, z**2], lex, QQ, gens=[x, y, z])
    # 断言调用 sdm_nf_mora 函数并检查其返回值是否符合预期
    assert sdm_nf_mora(f, [f1, f2], lex, QQ) == \
        sdm_nf_mora(f, [f2, f1], lex, QQ) == \
        [((1, 0, 1, 1), QQ(1)), ((1, 0, 0, 1), QQ(-1)), ((0, 1, 1, 0), QQ(-1)),
         ((0, 1, 0, 1), QQ(1))]


# 定义一个测试函数 test_conversion，用于测试 sdm_to_vector 和 sdm_from_vector 函数的正确性
def test_conversion():
    f = [x**2 + y**2, 2*z]
    g = [((1, 0, 0, 1), QQ(2)), ((0, 2, 0, 0), QQ(1)), ((0, 0, 2, 0), QQ(1))]
    # 断言调用 sdm_to_vector 函数并检查其返回值是否符合预期
    assert sdm_to_vector(g, [x, y, z], QQ) == f
    # 断言调用 sdm_from_vector 函数并检查其返回值是否符合预期
    assert sdm_from_vector(f, lex, QQ) == g
    # 断言调用 sdm_from_vector 函数并检查其返回值是否符合预期
    assert sdm_from_vector(
        [x, 1], lex, QQ) == [((1, 0), QQ(1)), ((0, 1), QQ(1))]
    # 断言调用 sdm_to_vector 函数并检查其返回值是否符合预期
    assert sdm_to_vector([((1, 1, 0, 0), 1)], [x, y, z], QQ, n=3) == [0, x, 0]
    # 断言调用 sdm_from_vector 函数并检查其返回值是否符合预期
    assert sdm_from_vector([0, 0], lex, QQ, gens=[x, y]) == sdm_zero()


# 定义一个测试函数 test_nontrivial，用于测试 contains 函数的正确性
def test_nontrivial():
    gens = [x, y, z]

    def contains(I, f):
        S = [sdm_from_vector([g], lex, QQ, gens=gens) for g in I]
        G = sdm_groebner(S, sdm_nf_mora, lex, QQ)
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens),
                           G, lex, QQ) == sdm_zero()

    # 断言调用 contains 函数并检查其返回值是否为 True 或 False，用于测试不同的情况
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    # 断言，检查列表中是否包含指定的表达式 x**3
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    
    # 断言，检查列表中是否不包含指定的表达式 x**2 + y**2
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)
    
    # 断言，检查列表中是否不包含指定的表达式 x*(1 + x + y)
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    
    # 断言，检查列表中是否不包含指定的表达式 x + y
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)
# 定义一个本地测试函数，用于测试特定算法的功能
def test_local():
    # 创建一个反序的字典序对象
    igrlex = InverseOrder(grlex)
    # 定义生成器列表
    gens = [x, y, z]

    # 定义一个内部函数用于检查多项式是否包含在理想中
    def contains(I, f):
        # 根据给定生成器生成单项理想集合
        S = [sdm_from_vector([g], igrlex, QQ, gens=gens) for g in I]
        # 使用单项理想集合计算格罗布纳基基础G
        G = sdm_groebner(S, sdm_nf_mora, igrlex, QQ)
        # 判断给定的多项式f是否在理想I中
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens),
                           G, lex, QQ) == sdm_zero()

    # 断言各种情况下函数contains的正确性
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


# 定义一个测试未覆盖的行函数
def test_uncovered_line():
    # 定义生成器列表
    gens = [x, y]
    # 定义零多项式
    f1 = sdm_zero()
    # 根据向量创建多项式f2
    f2 = sdm_from_vector([x, 0], lex, QQ, gens=gens)
    # 根据向量创建多项式f3
    f3 = sdm_from_vector([0, y], lex, QQ, gens=gens)

    # 断言特定多项式的斯波利多项式为零
    assert sdm_spoly(f1, f2, lex, QQ) == sdm_zero()
    assert sdm_spoly(f3, f2, lex, QQ) == sdm_zero()


# 定义一个测试链条件的函数
def test_chain_criterion():
    # 定义生成器列表
    gens = [x]
    # 根据向量创建多项式f1
    f1 = sdm_from_vector([1, x], grlex, QQ, gens=gens)
    # 根据向量创建多项式f2
    f2 = sdm_from_vector([0, x - 2], grlex, QQ, gens=gens)
    
    # 断言格罗布纳基基础的长度是否符合预期
    assert len(sdm_groebner([f1, f2], sdm_nf_mora, grlex, QQ)) == 2
```