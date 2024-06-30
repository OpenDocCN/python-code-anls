# `D:\src\scipysrc\sympy\sympy\polys\agca\tests\test_modules.py`

```
"""Test modules.py code."""

# 导入所需模块和函数
from sympy.polys.agca.modules import FreeModule, ModuleOrder, FreeModulePolyRing
from sympy.polys import CoercionFailed, QQ, lex, grlex, ilex, ZZ
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.core.numbers import Rational

# 定义测试函数test_FreeModuleElement
def test_FreeModuleElement():
    # 创建一个多项式环 QQ.old_poly_ring(x)，生成自由模 M，维度为 3
    M = QQ.old_poly_ring(x).free_module(3)
    # 创建元素 e，将列表 [1, x, x**2] 转换为 M 中的元素
    e = M.convert([1, x, x**2])
    # 创建列表 f，包含了将 [1, x, x**2] 在 QQ.old_poly_ring(x) 下转换的结果
    f = [QQ.old_poly_ring(x).convert(1), QQ.old_poly_ring(x).convert(x), QQ.old_poly_ring(x).convert(x**2)]
    # 断言 e 转换为列表与 f 相等
    assert list(e) == f
    # 断言 f 的第一个元素与 e 的第一个元素相等
    assert f[0] == e[0]
    # 断言 f 的第二个元素与 e 的第二个元素相等
    assert f[1] == e[1]
    # 断言 f 的第三个元素与 e 的第三个元素相等
    assert f[2] == e[2]
    # 断言使用 lambda 表达式执行 e[3] 会引发 IndexError 异常
    raises(IndexError, lambda: e[3])

    # 创建元素 g，将列表 [x, 0, 0] 转换为 M 中的元素
    g = M.convert([x, 0, 0])
    # 断言 e 加 g 等于 M 中的元素 [x + 1, x, x**2]
    assert e + g == M.convert([x + 1, x, x**2])
    # 断言 f 加 g 等于 M 中的元素 [x + 1, x, x**2]
    assert f + g == M.convert([x + 1, x, x**2])
    # 断言 -e 等于 M 中的元素 [-1, -x, -x**2]
    assert -e == M.convert([-1, -x, -x**2])
    # 断言 e 减 g 等于 M 中的元素 [1 - x, x, x**2]
    assert e - g == M.convert([1 - x, x, x**2])
    # 断言 e 不等于 g
    assert e != g

    # 断言将 M 中的元素 [x, x, x] 除以 QQ.old_poly_ring(x) 中的元素 x 得到 [1, 1, 1]
    assert M.convert([x, x, x]) / QQ.old_poly_ring(x).convert(x) == [1, 1, 1]
    # 创建多项式环 R，使用 ilex 排序，自由模维度为 1
    R = QQ.old_poly_ring(x, order="ilex")
    # 断言 R 中的自由模维度为 1 转换的 [x] 除以 R 中的 x 等于 [1]
    assert R.free_module(1).convert([x]) / R.convert(x) == [1]

# 定义测试函数 test_FreeModule
def test_FreeModule():
    # 创建自由模 M1，使用 QQ.old_poly_ring(x)，维度为 2
    M1 = FreeModule(QQ.old_poly_ring(x), 2)
    # 断言 M1 与另一个使用 QQ.old_poly_ring(x)，维度为 2 的 FreeModule 对象相等
    assert M1 == FreeModule(QQ.old_poly_ring(x), 2)
    # 断言 M1 与使用 QQ.old_poly_ring(y)，维度为 2 的 FreeModule 对象不相等
    assert M1 != FreeModule(QQ.old_poly_ring(y), 2)
    # 断言 M1 与使用 QQ.old_poly_ring(x)，维度为 3 的 FreeModule 对象不相等
    assert M1 != FreeModule(QQ.old_poly_ring(x), 3)
    # 创建自由模 M2，使用 QQ.old_poly_ring(x, order="ilex")，维度为 2
    M2 = FreeModule(QQ.old_poly_ring(x, order="ilex"), 2)

    # 断言 [x, 1] 在 M1 中
    assert [x, 1] in M1
    # 断言 [x] 不在 M1 中
    assert [x] not in M1
    # 断言 [2, y] 不在 M1 中
    assert [2, y] not in M1
    # 断言 [1/(x + 1), 2] 不在 M1 中
    assert [1/(x + 1), 2] not in M1

    # 创建元素 e，将列表 [x, x**2 + 1] 转换为 M1 中的元素
    e = M1.convert([x, x**2 + 1])
    # X 表示 QQ.old_poly_ring(x) 中的 x
    X = QQ.old_poly_ring(x).convert(x)
    # 断言 e 等于 [X, X**2 + 1]
    assert e == [X, X**2 + 1]
    # 断言 e 等于 [x, x**2 + 1]
    assert e == [x, x**2 + 1]
    # 断言 2*e 等于 [2*x, 2*x**2 + 2]
    assert 2*e == [2*x, 2*x**2 + 2]
    # 断言 e*2 等于 [2*x, 2*x**2 + 2]
    assert e*2 == [2*x, 2*x**2 + 2]
    # 断言 e/2 等于 [x/2, (x**2 + 1)/2]
    assert e/2 == [x/2, (x**2 + 1)/2]
    # 断言 x*e 等于 [x**2, x**3 + x]
    assert x*e == [x**2, x**3 + x]
    # 断言 e*x 等于 [x**2, x**3 + x]
    assert e*x == [x**2, x**3 + x]
    # 断言 X*e 等于 [x**2, x**3 + x]
    assert X*e == [x**2, x**3 + x]
    # 断言 e*X 等于 [x**2, x**3 + x]
    assert e*X == [x**2, x**3 + x]

    # 断言 [x, 1] 在 M2 中
    assert [x, 1] in M2
    # 断言 [x] 不在 M2 中
    assert [x] not in M2
    # 断言 [2, y] 不在 M2 中
    assert [2, y] not in M2
    # 断言 [1/(x + 1), 2] 在 M2 中
    assert [1/(x + 1), 2] in M2

    # 创建元素 e，将列表 [x, x**2 + 1] 转换为 M2 中的元素
    e = M2.convert([x, x**2 + 1])
    # X 表示 QQ.old_poly_ring(x, order="ilex") 中的 x
    X = QQ.old_poly_ring(x, order="ilex").convert(x)
    # 断言 e 等于 [X, X**2 + 1]
    assert e == [X, X**2 + 1]
    # 断言 e 等于 [x, x**2 + 1]
    assert e == [x, x**2 + 1]
    # 断言 2*e 等于 [2*x, 2*x**2 + 2]
    assert 2*e == [2*x, 2*x**2 + 2]
    # 断言 e*2 等于 [2*x, 2*x**2 + 2]
    assert e*2 == [2*x, 2*x**2 + 2]
    # 断言 e/2 等于 [x/2, (x**2 + 1)/2]
    assert e/2 == [x/2,
    # 断言，验证 o1 是否等于 ModuleOrder(lex, grlex, False)
    assert o1 == ModuleOrder(lex, grlex, False)
    # 断言，验证 o1 是否不等于 ModuleOrder(lex, grlex, False) 的结果是否为 False
    assert (o1 != ModuleOrder(lex, grlex, False)) is False
    # 断言，验证 o1 是否不等于 o2
    assert o1 != o2

    # 断言，验证 o1 应用于元组 (1, 2, 3) 的结果是否等于 (1, (5, (2, 3)))
    assert o1((1, 2, 3)) == (1, (5, (2, 3)))
    # 断言，验证 o2 应用于元组 (1, 2, 3) 的结果是否等于 (-1, (2, 3))
    assert o2((1, 2, 3)) == (-1, (2, 3))
def test_SubModulePolyRing_global():
    R = QQ.old_poly_ring(x, y)  # 创建一个多项式环 R，包含变量 x 和 y
    F = R.free_module(3)  # 创建一个自由模 F，维度为 3
    Fd = F.submodule([1, 0, 0], [1, 2, 0], [1, 2, 3])  # 创建 F 的子模块 Fd，由三个向量生成
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])  # 创建 F 的子模块 M，由两个向量生成

    assert F == Fd  # 断言 F 等于 Fd
    assert Fd == F  # 断言 Fd 等于 F
    assert F != M  # 断言 F 不等于 M
    assert M != F  # 断言 M 不等于 F
    assert Fd != M  # 断言 Fd 不等于 M
    assert M != Fd  # 断言 M 不等于 Fd
    assert Fd == F.submodule(*F.basis())  # 断言 Fd 等于使用 F 的基底生成的子模块

    assert Fd.is_full_module()  # 断言 Fd 是全模块
    assert not M.is_full_module()  # 断言 M 不是全模块
    assert not Fd.is_zero()  # 断言 Fd 不是零模块
    assert not M.is_zero()  # 断言 M 不是零模块
    assert Fd.submodule().is_zero()  # 断言 Fd 的子模块是零模块

    assert M.contains([x**2 + y**2 + x, 1 + y, 1])  # 断言 M 包含给定向量
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])  # 断言 M 不包含给定向量
    assert M.contains([y**2, 1 - x*y, -x])  # 断言 M 包含给定向量

    assert not F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])  # 断言两个子模块不相等
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F  # 断言两个子模块的并集等于 F
    assert not M.is_submodule(0)  # 断言 M 不是 F 的零子模块

    m = F.convert([x**2 + y**2, 1, 0])  # 将向量转换为 F 中的元素 m
    n = M.convert(m)  # 将 m 转换为 M 中的元素 n
    assert m.module is F  # 断言 m 属于模块 F
    assert n.module is M  # 断言 n 属于模块 M

    raises(ValueError, lambda: M.submodule([1, 0, 0]))  # 断言调用 M.submodule([1, 0, 0]) 抛出 ValueError 异常
    raises(TypeError, lambda: M.union(1))  # 断言调用 M.union(1) 抛出 TypeError 异常
    raises(ValueError, lambda: M.union(R.free_module(1).submodule([x])))  # 断言调用 M.union(...) 抛出 ValueError 异常

    assert F.submodule([x, x, x]) != F.submodule([x, x, x], order="ilex")  # 断言两个子模块不相等


def test_SubModulePolyRing_local():
    R = QQ.old_poly_ring(x, y, order=ilex)  # 创建一个多项式环 R，包含变量 x, y，并指定排序方式为 ilex
    F = R.free_module(3)  # 创建一个自由模 F，维度为 3
    Fd = F.submodule([1 + x, 0, 0], [1 + y, 2 + 2*y, 0], [1, 2, 3])  # 创建 F 的子模块 Fd，由三个向量生成
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])  # 创建 F 的子模块 M，由两个向量生成

    assert F == Fd  # 断言 F 等于 Fd
    assert Fd == F  # 断言 Fd 等于 F
    assert F != M  # 断言 F 不等于 M
    assert M != F  # 断言 M 不等于 F
    assert Fd != M  # 断言 Fd 不等于 M
    assert M != Fd  # 断言 M 不等于 Fd
    assert Fd == F.submodule(*F.basis())  # 断言 Fd 等于使用 F 的基底生成的子模块

    assert Fd.is_full_module()  # 断言 Fd 是全模块
    assert not M.is_full_module()  # 断言 M 不是全模块
    assert not Fd.is_zero()  # 断言 Fd 不是零模块
    assert not M.is_zero()  # 断言 M 不是零模块
    assert Fd.submodule().is_zero()  # 断言 Fd 的子模块是零模块

    assert M.contains([x**2 + y**2 + x, 1 + y, 1])  # 断言 M 包含给定向量
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])  # 断言 M 不包含给定向量
    assert M.contains([y**2, 1 - x*y, -x])  # 断言 M 包含给定向量

    assert F.submodule([1 + x, 0, 0]) == F.submodule([1, 0, 0])  # 断言两个子模块相等
    assert F.submodule(
        [1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1 + x*y])) == F  # 断言两个子模块的并集等于 F

    raises(ValueError, lambda: M.submodule([1, 0, 0]))  # 断言调用 M.submodule([1, 0, 0]) 抛出 ValueError 异常


def test_SubModulePolyRing_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)  # 创建一个多项式环 R，包含变量 x, y, z
    F = R.free_module(1)  # 创建一个自由模 F，维度为 1

    def contains(I, f):
        return F.submodule(*[[g] for g in I]).contains([f])  # 定义一个函数 contains，检查子模块是否包含给定向量

    assert contains([x, y], x)  # 断言子模块包含向量 [x, y] 的 x 分量
    assert contains([x, y], x + y)  # 断言子模块包含向量 [x, y] 的 x + y
    assert not contains([x, y], 1)  # 断言子模块不包含向量 [x, y] 的 1
    assert not contains([x, y], z)  # 断言子模块不包含向量 [x, y] 的 z
    assert contains([x**2 + y, x**2 + x], x - y)  # 断言子模块包含向量 [x**2 + y, x**2 + x] 的 x - y
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)  # 断言子模块不包含向量 [x + y + z, x*y + x*z + y*z, x*y*z] 的 x**2
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)  # 断言子模块包含向量 [x + y + z, x*y + x*z + y*z, x*y*z] 的 x**3
    # 断言：检查列表中是否包含 x**3，如果包含则断言通过，否则断言失败
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    
    # 断言：检查列表中是否不包含 x**2 + y**2，如果不包含则断言通过，否则断言失败
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)
    
    # 断言：检查列表中是否不包含 x*(1 + x + y)，如果不包含则断言通过，否则断言失败
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    
    # 断言：检查列表中是否不包含 x*(1 + x + y) + y*(1 + z)，如果不包含则断言通过，否则断言失败
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)
def test_SubModulePolyRing_nontriv_local():
    # 使用 QQ.old_poly_ring() 创建多项式环 R，指定变量 x, y, z，使用 ilex 排序
    R = QQ.old_poly_ring(x, y, z, order=ilex)
    # 在 R 上创建自由模块 F，维度为 1
    F = R.free_module(1)

    # 定义内部函数 contains，用于检查子模块是否包含给定元素
    def contains(I, f):
        # 将列表 I 转换为 F 中的子模块，然后检查是否包含向量 [f]
        return F.submodule(*[[g] for g in I]).contains([f])

    # 以下为一系列断言，验证 contains 函数的行为
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_syzygy():
    # 使用 QQ.old_poly_ring() 创建多项式环 R，指定变量 x, y, z
    R = QQ.old_poly_ring(x, y, z)
    # 在 R 上创建自由模块 M，并定义其子模块，使用给定的生成元
    M = R.free_module(1).submodule([x*y], [y*z], [x*z])
    # 定义预期的同调模块 S
    S = R.free_module(3).submodule([0, x, -y], [z, -x, 0])
    # 验证 M 的同调模块与预期的 S 相等
    assert M.syzygy_module() == S

    # 将 M 模块除以理想 [x*y*z]
    M2 = M / ([x*y*z],)
    # 定义预期的同调模块 S2
    S2 = R.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    # 验证 M2 的同调模块与预期的 S2 相等
    assert M2.syzygy_module() == S2

    # 在 R 上创建自由模块 F
    F = R.free_module(3)
    # 验证 F 模块生成元的同调模块等于空子模块
    assert F.submodule(*F.basis()).syzygy_module() == F.submodule()

    # 在 R / [x*y*z] 上创建多项式环 R2
    R2 = QQ.old_poly_ring(x, y, z) / [x*y*z]
    # 在 R2 上创建自由模块 M3，并定义其子模块，使用给定的生成元
    M3 = R2.free_module(1).submodule([x*y], [y*z], [x*z])
    # 定义预期的同调模块 S3
    S3 = R2.free_module(3).submodule([z, 0, 0], [0, x, 0], [0, 0, y])
    # 验证 M3 的同调模块与预期的 S3 相等
    assert M3.syzygy_module() == S3


def test_in_terms_of_generators():
    # 使用 QQ.old_poly_ring() 创建多项式环 R，指定变量 x, y，使用 ilex 排序
    R = QQ.old_poly_ring(x, y, order="ilex")
    # 在 R 上创建自由模块 M，并定义其子模块，使用给定的生成元
    M = R.free_module(2).submodule([2*x, 0], [1, 2])
    # 验证 M 模块在给定生成元下的表达式
    assert M.in_terms_of_generators([x, x]) == [R.convert(Rational(1, 4)), R.convert(x/2)]
    # 验证当提供无效生成元时，引发 ValueError 异常
    raises(ValueError, lambda: M.in_terms_of_generators([1, 0]))

    # 将 M 模块除以理想 [x, 0], [1, 1]
    M = R.free_module(2) / ([x, 0], [1, 1])
    # 在 M 上创建子模块 SM，使用给定的生成元
    SM = M.submodule([1, x])
    # 验证 SM 模块在给定生成元下的表达式
    assert SM.in_terms_of_generators([2, 0]) == [R.convert(-2/(x - 1))]

    # 在 R / [x**2 - y**2] 上创建多项式环 R
    R = QQ.old_poly_ring(x, y) / [x**2 - y**2]
    # 在 R 上创建自由模块 M
    M = R.free_module(2)
    # 在 M 上创建子模块 SM，使用给定的生成元
    SM = M.submodule([x, 0], [0, y])
    # 验证 SM 模块在给定生成元下的表达式
    assert SM.in_terms_of_generators([x**2, x**2]) == [R.convert(x), R.convert(y)]


def test_QuotientModuleElement():
    # 使用 QQ.old_poly_ring() 创建多项式环 R，指定变量 x
    R = QQ.old_poly_ring(x)
    # 在 R 上创建自由模块 F，维度为 3
    F = R.free_module(3)
    # 定义子模块 N，使用给定的生成元
    N = F.submodule([1, x, x**2])
    # 将 F 模块除以子模块 N，得到商模块 M
    M = F/N
    # 创建元素 e，属于商模块 M，使用给定的坐标表示
    e = M.convert([x**2, 2, 0])

    # 一系列断言，验证元素在商模块 M 下的表达式与相等性
    assert M.convert([x + 1, x**2 + x, x**3 + x**2]) == 0
    assert e == [x**2, 2, 0] + N == F.convert([x**2, 2, 0]) + N == \
        M.convert(F.convert([x**2, 2, 0]))

    assert M.convert([x**2 + 1, 2*x + 2, x**2]) == e + [0, x, 0] == \
        e + M.convert([0, x, 0]) == e + F.convert([0, x, 0])
    assert M.convert([x**2 + 1, 2, x**2]) == e - [0, x, 0] == \
        e - M.convert([0, x, 0]) == e - F.convert([0, x, 0])
    assert M.convert([0, 2, 0]) == M.convert([x**2, 4, 0]) - e == \
        [x**2, 4, 0] - e == F.convert([x**2, 4, 0]) - e
    assert M.convert([x**3 + x**2, 2*x + 2, 0]) == (1 + x)*e == \
        R.convert(1 + x)*e == e*(1 + x) == e*R.convert(1 + x)
    assert -e == [-x**2, -2, 0]

    # 创建元素 f，属于商模块 M，使用给定的坐标表示
    f = [x, x, 0] + N
    # 验证元素在商模块 M 下的表达式
    assert M.convert([1, 1, 0]) == f / x == f / R.convert(x)

    # 将 F 模块除以多个理想 [(2, 2*x, 2*x**2), (0, 0, 1)]
    M2 = F/[(2, 2*x, 2*x**2), (0, 0, 1)]
    # 在 R 上创建自由模块 G，维度为 2
    G = R
    # 断言引发 CoercionFailed 异常，使用 lambda 表达式调用 M.convert(G.convert([1, x]))
    raises(CoercionFailed, lambda: M.convert(G.convert([1, x])))
    # 断言引发 CoercionFailed 异常，使用 lambda 表达式调用 M.convert(M3.convert([1, x]))
    raises(CoercionFailed, lambda: M.convert(M3.convert([1, x])))
    # 断言引发 CoercionFailed 异常，使用 lambda 表达式调用 M.convert(M2.convert([1, x, x]))
    raises(CoercionFailed, lambda: M.convert(M2.convert([1, x, x])))
    # 断言检查 M2.convert(M.convert([2, x, x**2])) 是否等于 [2, x, 0]
    assert M2.convert(M.convert([2, x, x**2])) == [2, x, 0]
    # 断言检查 M.convert(M4.convert([2, 0, 0])) 是否等于 [2, 0, 0]
    assert M.convert(M4.convert([2, 0, 0])) == [2, 0, 0]
def test_QuotientModule():
    # 创建有理数域上关于变量 x 的多项式环
    R = QQ.old_poly_ring(x)
    # 创建自由模块 F，维度为 3
    F = R.free_module(3)
    # 创建子模块 N，包含基向量 [1, x, x**2]
    N = F.submodule([1, x, x**2])
    # 构造模 M 作为 F 对 N 的商模

    # 断言：M 不等于 F
    assert M != F
    # 断言：M 不等于 N
    assert M != N
    # 断言：M 等于 F 对于 [(1, x, x**2)] 的商模
    assert M == F / [(1, x, x**2)]
    # 断言：M 不是零模
    assert not M.is_zero()
    # 断言：F 对于其基向量的商模是零模
    assert (F / F.basis()).is_zero()

    # 创建子模 SQ，作为 F 对 N 的商模，由基向量 [1, x, x**2] 和 [2, 0, 0] 生成
    SQ = F.submodule([1, x, x**2], [2, 0, 0]) / N
    # 断言：SQ 等于 M 的子模，由基向量 [2, x, x**2] 生成
    assert SQ == M.submodule([2, x, x**2])
    # 断言：SQ 不等于 M 的子模，由基向量 [2, 1, 0] 生成
    assert SQ != M.submodule([2, 1, 0])
    # 断言：SQ 不等于 M
    assert SQ != M
    # 断言：M 是 SQ 的子模
    assert M.is_submodule(SQ)
    # 断言：SQ 不是全模
    assert not SQ.is_full_module()

    # 断言：使用 lambda 函数测试 ValueError 是否会被引发，因为无法计算 N/F
    raises(ValueError, lambda: N/F)
    # 断言：使用 lambda 函数测试 ValueError 是否会被引发，因为无法计算 F.submodule([2, 0, 0]) / N
    raises(ValueError, lambda: F.submodule([2, 0, 0]) / N)
    # 断言：使用 lambda 函数测试 CoercionFailed 是否会被引发，因为无法将 [1, x, x**2] 转换为 M
    raises(CoercionFailed, lambda: F.convert(M.convert([1, x, x**2])))

    # 创建 M1 作为 F 对于 [[1, 1, 1]] 的商模
    M1 = F / [[1, 1, 1]]
    # 创建 M2 作为 M1 的子模，由基向量 [1, 0, 0] 和 [0, 1, 0] 生成
    M2 = M1.submodule([1, 0, 0], [0, 1, 0])
    # 断言：M1 等于 M2
    assert M1 == M2



def test_ModulesQuotientRing():
    # 创建有理数域上关于变量 x, y 的多项式环，使用 lexicographic 和 inverse lexicographic 排序
    R = QQ.old_poly_ring(x, y, order=(("lex", x), ("ilex", y))) / [x**2 + 1]
    # 创建自由模块 M1，维度为 2
    M1 = R.free_module(2)
    # 断言：M1 等于 R 的自由模块，维度为 2
    assert M1 == R.free_module(2)
    # 断言：M1 不等于 QQ.old_poly_ring(x) 的自由模块，维度为 2
    assert M1 != QQ.old_poly_ring(x).free_module(2)
    # 断言：M1 不等于 R 的自由模块，维度为 3
    assert M1 != R.free_module(3)

    # 断言：向量 [x, 1] 在 M1 中
    assert [x, 1] in M1
    # 断言：向量 [x] 不在 M1 中
    assert [x] not in M1
    # 断言：向量 [1/(R.convert(x) + 1), 2] 在 M1 中
    assert [1/(R.convert(x) + 1), 2] in M1
    # 断言：向量 [1, 2/(1 + y)] 在 M1 中
    assert [1, 2/(1 + y)] in M1
    # 断言：向量 [1, 2/y] 不在 M1 中
    assert [1, 2/y] not in M1

    # 断言：M1.convert([x**2, y]) 等于 [-1, y]
    assert M1.convert([x**2, y]) == [-1, y]

    # 创建自由模块 F，维度为 3
    F = R.free_module(3)
    # 创建 F 的子模块 Fd，由基向量 [x**2, 0, 0], [1, 2, 0], [1, 2, 3] 生成
    Fd = F.submodule([x**2, 0, 0], [1, 2, 0], [1, 2, 3])
    # 创建 M，由基向量 [x**2 + y**2, 1, 0], [x, y, 1] 生成
    M = F.submodule([x**2 + y**2, 1, 0], [x, y, 1])

    # 断言：F 等于 Fd
    assert F == Fd
    # 断言：Fd 等于 F
    assert Fd == F
    # 断言：F 不等于 M
    assert F != M
    # 断言：M 不等于 F
    assert M != F
    # 断言：Fd 不等于 M
    assert Fd != M
    # 断言：M 不等于 Fd
    assert M != Fd
    # 断言：Fd 等于 F 的基向量的子模
    assert Fd == F.submodule(*F.basis())

    # 断言：Fd 是全模
    assert Fd.is_full_module()
    # 断言：M 不是全模
    assert not M.is_full_module()
    # 断言：Fd 不是零模
    assert not Fd.is_zero()
    # 断言：M 不是零模
    assert not M.is_zero()
    # 断言：Fd 的子模是零模
    assert Fd.submodule().is_zero()

    # 断言：向量 [x**2 + y**2 + x, -x**2 + y, 1] 在 M 中
    assert M.contains([x**2 + y**2 + x, -x**2 + y, 1])
    # 断言：向量 [x**2 + y**2 + x, 1 + y, 2] 不在 M 中
    assert not M.contains([x**2 + y**2 + x, 1 + y, 2])
    # 断言：向量 [y**2, 1 - x*y, -x] 在 M 中
    assert M.contains([y**2, 1 - x*y, -x])

    # 断言：F 的子模 [x, 0, 0] 等于 F 的子模 [1, 0, 0]
    assert F.submodule([x, 0, 0]) == F.submodule([1, 0, 0])
    # 断言：F 的子模 [y, 0, 0] 不等于 F 的子模 [1, 0, 0]
    assert not F.submodule([y, 0, 0]) == F.submodule([1, 0, 0])
    # 断言：F 的子模 [1, 0, 0], [0, 1, 0] 的并集等于 F
    assert F.submodule([1, 0, 0], [0, 1, 0]).union(F.submodule([0, 0, 1])) == F
    # 断言：M 不是子模 0 的子模
    assert not M.is_submodule(0)



def test_module_mul():
    # 在环 R 上创建自由模 F，这里是一个二维自由模
    F = R.free_module(2)
    
    # 断言：构建一个子模，然后对其进行模商操作，并检查结果是否等于给定的理想
    assert F.submodule([x*y, x*z], [y*z, x*y]).module_quotient(
        F.submodule([y, z], [z, y])) == QQ.old_poly_ring(x, y, z).ideal(x**2*y**2 - x*y*z**2)
    
    # 断言：检查子模 F.submodule([x, y]) 对于整个环是否构成全环
    assert F.submodule([x, y]).module_quotient(F.submodule()).is_whole_ring()

    # 创建两个子模 M 和 N
    M = F.submodule([x**2, x**2], [y**2, y**2])
    N = F.submodule([x + y, x + y])
    
    # 对子模 M 和 N 进行模商操作，返回商模和关系（如果指定）
    q, rel = M.module_quotient(N, relations=True)
    
    # 断言：检查商模 q 是否等于环 R 中的理想 (y**2, x - y)
    assert q == R.ideal(y**2, x - y)
    
    # 遍历商模的生成元，对每个生成元执行关系的验证
    for i, g in enumerate(q.gens):
        # 断言：检查每个生成元 g 是否满足关系式 g*N.gens[0] == sum(c*x for c, x in zip(rel[i], M.gens))
        assert g*N.gens[0] == sum(c*x for c, x in zip(rel[i], M.gens))
# 定义名为 test_groebner_extendend 的测试函数
def test_groebner_extendend():
    # 在有理数域 QQ 上创建多项式环，使用变量 x, y, z
    # 创建自由模块，维度为 3，生成子模块由生成元 [x + 1, y, 1] 和关系 [x*y, z, z**2] 定义
    M = QQ.old_poly_ring(x, y, z).free_module(3).submodule([x + 1, y, 1], [x*y, z, z**2])
    
    # 调用模块 M 的 _groebner_vec 方法，传入 extended=True 参数
    G, R = M._groebner_vec(extended=True)
    
    # 遍历 G 中的元素，同时获取索引 i 和元素 g
    for i, g in enumerate(G):
        # 断言当前 g 等于 M.gens 的线性组合，系数由 R[i] 给出
        assert g == sum(c*gen for c, gen in zip(R[i], M.gens))
```