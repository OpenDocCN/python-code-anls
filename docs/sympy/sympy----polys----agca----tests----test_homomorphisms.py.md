# `D:\src\scipysrc\sympy\sympy\polys\agca\tests\test_homomorphisms.py`

```
# 导入所需模块和类
from sympy.core.singleton import S
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
from sympy.polys.agca import homomorphism
from sympy.testing.pytest import raises

# 定义测试函数，用于测试打印功能
def test_printing():
    # 创建有理函数域 QQ 上关于变量 x 的旧多项式环 R
    R = QQ.old_poly_ring(x)

    # 断言：测试打印单模的同态映射结果
    assert str(homomorphism(R.free_module(1), R.free_module(1), [0])) == \
        'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1'
    
    # 断言：测试打印双模的同态映射结果
    assert str(homomorphism(R.free_module(2), R.free_module(2), [0, 0])) == \
        'Matrix([                       \n[0, 0], : QQ[x]**2 -> QQ[x]**2\n[0, 0]])                       '
    
    # 断言：测试打印单模除法的同态映射结果
    assert str(homomorphism(R.free_module(1), R.free_module(1) / [[x]], [0])) == \
        'Matrix([[0]]) : QQ[x]**1 -> QQ[x]**1/<[x]>'
    
    # 断言：测试打印空模的单位同态映射结果
    assert str(R.free_module(0).identity_hom()) == 'Matrix(0, 0, []) : QQ[x]**0 -> QQ[x]**0'

# 定义测试函数，用于测试各种操作
def test_operations():
    # 创建关于变量 x 的旧多项式环 QQ.old_poly_ring(x) 的二维自由模 F 和三维自由模 G
    F = QQ.old_poly_ring(x).free_module(2)
    G = QQ.old_poly_ring(x).free_module(3)
    
    # 获取 F 的单位同态映射
    f = F.identity_hom()
    
    # 创建具有给定映射矩阵的同态映射 g, h, i
    g = homomorphism(F, F, [0, [1, x]])
    h = homomorphism(F, F, [[1, 0], 0])
    i = homomorphism(F, G, [[1, 0, 0], [0, 1, 0]])
    
    # 各种操作的断言测试
    assert f == f
    assert f != g
    assert f != i
    assert (f != F.identity_hom()) is False
    assert 2*f == f*2 == homomorphism(F, F, [[2, 0], [0, 2]])
    assert f/2 == homomorphism(F, F, [[S.Half, 0], [0, S.Half]])
    assert f + g == homomorphism(F, F, [[1, 0], [1, x + 1]])
    assert f - g == homomorphism(F, F, [[1, 0], [-1, 1 - x]])
    assert f*g == g == g*f
    assert h*g == homomorphism(F, F, [0, [1, 0]])
    assert g*h == homomorphism(F, F, [0, 0])
    assert i*f == i
    assert f([1, 2]) == [1, 2]
    assert g([1, 2]) == [2, 2*x]

    # 断言：映射到子模的限制结果与映射本身结果相同
    assert i.restrict_domain(F.submodule([x, x]))([x, x]) == i([x, x])
    
    # 创建 h 关于 F 子模 [0, 1] 的商域同态映射 h1
    h1 = h.quotient_domain(F.submodule([0, 1]))
    assert h1([1, 0]) == h([1, 0])
    
    # 断言：限制到 h1 的定义域子模 [x, 0] 的结果与 h 相同
    assert h1.restrict_domain(h1.domain.submodule([x, 0]))([x, 0]) == h([x, 0])
    
    # 检查各种异常情况的 raises
    raises(TypeError, lambda: f/g)
    raises(TypeError, lambda: f + 1)
    raises(TypeError, lambda: f + i)
    raises(TypeError, lambda: f - 1)
    raises(TypeError, lambda: f*i)

# 定义测试函数，用于测试创建过程
def test_creation():
    # 创建关于变量 x 的旧多项式环 QQ.old_poly_ring(x) 的三维和二维自由模 F, G
    F = QQ.old_poly_ring(x).free_module(3)
    G = QQ.old_poly_ring(x).free_module(2)
    
    # 创建 F 的子模 [1, 1, 1] 和相应的商模 Q, 及其子模 SQ
    SM = F.submodule([1, 1, 1])
    Q = F / SM
    SQ = Q.submodule([1, 0, 0])
    
    # 创建具有给定映射矩阵的同态映射 h 和 h2
    matrix = [[1, 0], [0, 1], [-1, -1]]
    h = homomorphism(F, G, matrix)
    h2 = homomorphism(Q, G, matrix)
    
    # 断言：h 关于 SM 的商域同态映射等于 h2
    assert h.quotient_domain(SM) == h2
    
    # 断言：限制到 SQ 的结果等于映射到 SQ 的同态映射
    assert h2.restrict_domain(SQ) == homomorphism(SQ, G, matrix)
    
    # 检查各种异常情况的 raises
    raises(ValueError, lambda: h.quotient_domain(F.submodule([1, 0, 0])))
    raises(ValueError, lambda: h.restrict_domain(G))
    raises(ValueError, lambda: h.restrict_codomain(G.submodule([1, 0])))
    raises(ValueError, lambda: h.quotient_codomain(F))
    
    # 断言：各模的单位同态映射的创建与预期相符
    im = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for M in [F, SM, Q, SQ]:
        assert M.identity_hom() == homomorphism(M, M, im)
    assert SM.inclusion_hom() == homomorphism(SM, F, im)
    assert SQ.inclusion_hom() == homomorphism(SQ, Q, im)
    # 断言：验证 Q 的商同态等于使用 F、Q 和 im 创建的同态
    assert Q.quotient_hom() == homomorphism(F, Q, im)

    # 断言：验证 SQ 的商同态等于使用 SQ 的基础、SQ 和 im 创建的同态
    assert SQ.quotient_hom() == homomorphism(SQ.base, SQ, im)

    # 定义一个名为 conv 的类
    class conv:
        # convert 方法，返回其第一个参数 x
        def convert(x, y=None):
            return x

    # 定义一个名为 dummy 的类
    class dummy:
        # container 属性，指向 conv 类的实例
        container = conv()

        # submodule 方法，始终返回 None
        def submodule(*args):
            return None

    # 断言：验证尝试创建 homomorphism 对象时，使用了不合法的参数组合，应引发 TypeError
    raises(TypeError, lambda: homomorphism(dummy(), G, matrix))
    raises(TypeError, lambda: homomorphism(F, dummy(), matrix))

    # 断言：验证尝试创建 homomorphism 对象时，使用了不合法的参数组合，应引发 ValueError
    raises(ValueError, lambda: homomorphism(QQ.old_poly_ring(x, y).free_module(3), G, matrix))
    raises(ValueError, lambda: homomorphism(F, G, [0, 0]))
# 定义测试函数 test_properties
def test_properties():
    # 在有理数域 QQ 上创建旧的多项式环 R，包含变量 x 和 y
    R = QQ.old_poly_ring(x, y)
    # 在 R 上创建自由模块 F，维度为 2
    F = R.free_module(2)
    # 创建模块同态映射 h，从 F 到 F，映射矩阵为 [[x, 0], [y, 0]]
    h = homomorphism(F, F, [[x, 0], [y, 0]])
    # 断言 h 的核与 F 的子模块 [-y, x] 相等
    assert h.kernel() == F.submodule([-y, x])
    # 断言 h 的像与 F 的子模块 [x, 0], [y, 0] 相等
    assert h.image() == F.submodule([x, 0], [y, 0])
    # 断言 h 不是单射映射
    assert not h.is_injective()
    # 断言 h 不是满射映射
    assert not h.is_surjective()
    # 断言限制 h 的定义域为 F 的子模块 [x, 0] 的映射是满射映射
    assert h.restrict_codomain(h.image()).is_surjective()
    # 断言限制 h 的值域为 F 的子模块 [1, 0] 的映射是单射映射
    assert h.restrict_domain(F.submodule([1, 0])).is_injective()
    # 断言 h 在商域 h.kernel() 上限制 h.image() 的映射是同构映射
    assert h.quotient_domain(h.kernel()).restrict_codomain(h.image()).is_isomorphism()

    # 在有理数域 QQ 上，按照 lexicographic (lex, x) 和 inverse lexicographic (ilex, y) 的顺序创建多项式环 R2，除去多项式 x**2 + 1
    R2 = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y))) / [x**2 + 1]
    # 在 R2 上创建自由模块 F，维度为 2
    F = R2.free_module(2)
    # 创建模块同态映射 h，从 F 到 F，映射矩阵为 [[x, 0], [y, y + 1]]
    h = homomorphism(F, F, [[x, 0], [y, y + 1]])
    # 断言 h 是同构映射
    assert h.is_isomorphism()
```