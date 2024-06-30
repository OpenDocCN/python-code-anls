# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_modules.py`

```
from sympy.abc import x, zeta  # 导入符号变量 x 和 zeta
from sympy.polys import Poly, cyclotomic_poly  # 导入多项式类和周期多项式函数
from sympy.polys.domains import FF, QQ, ZZ  # 导入有限域、有理数域和整数域
from sympy.polys.matrices import DomainMatrix, DM  # 导入域矩阵和 DM 类
from sympy.polys.numberfields.exceptions import (  # 导入数域异常类
    ClosureFailure, MissingUnityError, StructureError
)
from sympy.polys.numberfields.modules import (  # 导入数域模块相关类和函数
    Module, ModuleElement, ModuleEndomorphism, PowerBasis, PowerBasisElement,
    find_min_poly, is_sq_maxrank_HNF, make_mod_elt, to_col,
)
from sympy.polys.numberfields.utilities import is_int  # 导入判断是否整数的实用函数
from sympy.polys.polyerrors import UnificationFailed  # 导入多项式错误类 UnificationFailed
from sympy.testing.pytest import raises  # 导入 pytest 中的 raises 函数


def test_to_col():
    c = [1, 2, 3, 4]
    m = to_col(c)  # 调用 to_col 函数
    assert m.domain.is_ZZ  # 断言域为整数环
    assert m.shape == (4, 1)  # 断言形状为 (4, 1)
    assert m.flat() == c  # 断言扁平化结果与原始列表相同


def test_Module_NotImplemented():
    M = Module()  # 创建 Module 实例
    raises(NotImplementedError, lambda: M.n)  # 断言调用 M.n 时引发 NotImplementedError
    raises(NotImplementedError, lambda: M.mult_tab())  # 断言调用 M.mult_tab() 时引发 NotImplementedError
    raises(NotImplementedError, lambda: M.represent(None))  # 断言调用 M.represent(None) 时引发 NotImplementedError
    raises(NotImplementedError, lambda: M.starts_with_unity())  # 断言调用 M.starts_with_unity() 时引发 NotImplementedError
    raises(NotImplementedError, lambda: M.element_from_rational(QQ(2, 3)))  # 断言调用 M.element_from_rational(QQ(2, 3)) 时引发 NotImplementedError


def test_Module_ancestors():
    T = Poly(cyclotomic_poly(5, x))  # 创建周期多项式 T
    A = PowerBasis(T)  # 创建以 T 为基的 PowerBasis 实例 A
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))  # 使用 2 倍单位矩阵创建子模块 B
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))  # 使用 3 倍单位矩阵创建子模块 C
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))  # 使用 5 倍单位矩阵创建子模块 D
    assert C.ancestors(include_self=True) == [A, B, C]  # 断言 C 的祖先包括 A、B 和 C 自身
    assert D.ancestors(include_self=True) == [A, B, D]  # 断言 D 的祖先包括 A、B 和 D 自身
    assert C.power_basis_ancestor() == A  # 断言 C 的幂基祖先是 A
    assert C.nearest_common_ancestor(D) == B  # 断言 C 和 D 的最近共同祖先是 B
    M = Module()
    assert M.power_basis_ancestor() is None  # 断言空模块 M 的幂基祖先为 None


def test_Module_compat_col():
    T = Poly(cyclotomic_poly(5, x))  # 创建周期多项式 T
    A = PowerBasis(T)  # 创建以 T 为基的 PowerBasis 实例 A
    col = to_col([1, 2, 3, 4])  # 创建域列向量 col
    row = col.transpose()  # 转置 col 得到行向量 row
    assert A.is_compat_col(col) is True  # 断言 col 与 A 兼容
    assert A.is_compat_col(row) is False  # 断言 row 与 A 不兼容
    assert A.is_compat_col(1) is False  # 断言数字 1 与 A 不兼容
    assert A.is_compat_col(DomainMatrix.eye(3, ZZ)[:, 0]) is False  # 断言不同维度的单位矩阵列与 A 不兼容
    assert A.is_compat_col(DomainMatrix.eye(4, QQ)[:, 0]) is False  # 断言有理数域的单位矩阵列与 A 不兼容
    assert A.is_compat_col(DomainMatrix.eye(4, ZZ)[:, 0]) is True  # 断言整数域的单位矩阵列与 A 兼容


def test_Module_call():
    T = Poly(cyclotomic_poly(5, x))  # 创建周期多项式 T
    B = PowerBasis(T)  # 创建以 T 为基的 PowerBasis 实例 B
    assert B(0).col.flat() == [1, 0, 0, 0]  # 断言 B(0) 的列向量为 [1, 0, 0, 0]
    assert B(1).col.flat() == [0, 1, 0, 0]  # 断言 B(1) 的列向量为 [0, 1, 0, 0]
    col = DomainMatrix.eye(4, ZZ)[:, 2]  # 创建整数域的单位矩阵第三列向量 col
    assert B(col).col == col  # 断言 B(col) 的列向量与 col 相同
    raises(ValueError, lambda: B(-1))  # 断言 B(-1) 会引发 ValueError


def test_Module_starts_with_unity():
    T = Poly(cyclotomic_poly(5, x))  # 创建周期多项式 T
    A = PowerBasis(T)  # 创建以 T 为基的 PowerBasis 实例 A
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))  # 使用 2 倍单位矩阵创建子模块 B
    assert A.starts_with_unity() is True  # 断言 A 以单位元素开头
    assert B.starts_with_unity() is False  # 断言 B 不以单位元素开头


def test_Module_basis_elements():
    T = Poly(cyclotomic_poly(5, x))  # 创建周期多项式 T
    A = PowerBasis(T)  # 创建以 T 为基的 PowerBasis 实例 A
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))  # 使用 2 倍单位矩阵创建子模块 B
    basis = B.basis_elements()  # 获取 B 的基元素
    bp = B.basis_element_pullbacks()  # 获取 B 的基元素回推
    # 使用 enumerate 遍历 basis 和 bp 列表，i 是索引，(e, p) 是元组，分别来自 basis 和 bp
    for i, (e, p) in enumerate(zip(basis, bp)):
        # 初始化一个长度为 4 的全零列表 c
        c = [0] * 4
        # 断言 e 的 module 属性等于 B
        assert e.module == B
        # 断言 p 的 module 属性等于 A
        assert p.module == A
        # 设置 c[i] 为 1
        c[i] = 1
        # 断言 e 等于 B(to_col(c))，to_col 函数可能将 c 转换为特定列的向量
        assert e == B(to_col(c))
        # 将 c[i] 设置为 2
        c[i] = 2
        # 断言 p 等于 A(to_col(c))，再次使用 to_col 函数对 c 进行处理
        assert p == A(to_col(c))
def test_Module_zero():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 使用单位矩阵生成一个 DomainMatrix 对象，再乘以 2，作为参数构建子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 断言 A 的零向量的列向量平铺后等于 [0, 0, 0, 0]
    assert A.zero().col.flat() == [0, 0, 0, 0]
    # 断言 A 的零向量所属的模块是 A 本身
    assert A.zero().module == A
    # 断言 B 的零向量的列向量平铺后等于 [0, 0, 0, 0]
    assert B.zero().col.flat() == [0, 0, 0, 0]
    # 断言 B 的零向量所属的模块是 B 自身
    assert B.zero().module == B


def test_Module_one():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 使用单位矩阵生成一个 DomainMatrix 对象，再乘以 2，作为参数构建子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 断言 A 的单位向量的列向量平铺后等于 [1, 0, 0, 0]
    assert A.one().col.flat() == [1, 0, 0, 0]
    # 断言 A 的单位向量所属的模块是 A 本身
    assert A.one().module == A
    # 断言 B 的单位向量的列向量平铺后等于 [1, 0, 0, 0]
    assert B.one().col.flat() == [1, 0, 0, 0]
    # 断言 B 的单位向量所属的模块是 A
    assert B.one().module == A


def test_Module_element_from_rational():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 使用单位矩阵生成一个 DomainMatrix 对象，再乘以 2，作为参数构建子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 使用有理数 QQ(22, 7) 生成 A 的一个元素 rA
    rA = A.element_from_rational(QQ(22, 7))
    # 使用有理数 QQ(22, 7) 生成 B 的一个元素 rB
    rB = B.element_from_rational(QQ(22, 7))
    # 断言 rA 的系数列表等于 [22, 0, 0, 0]
    assert rA.coeffs == [22, 0, 0, 0]
    # 断言 rA 的分母等于 7
    assert rA.denom == 7
    # 断言 rA 的模块是 A
    assert rA.module == A
    # 断言 rB 的系数列表等于 [22, 0, 0, 0]
    assert rB.coeffs == [22, 0, 0, 0]
    # 断言 rB 的分母等于 7
    assert rB.denom == 7
    # 断言 rB 的模块是 A
    assert rB.module == A


def test_Module_submodule_from_gens():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 生成一组基元素 gens
    gens = [2*A(0), 2*A(1), 6*A(0), 6*A(1)]
    # 使用 gens 构建子模块 B
    B = A.submodule_from_gens(gens)
    # 断言 B 的矩阵等于由 gens 的前两个元素的列水平拼接而成的矩阵 M
    M = gens[0].column().hstack(gens[1].column())
    assert B.matrix == M
    # 断言至少提供一个生成元
    raises(ValueError, lambda: A.submodule_from_gens([]))
    # 断言所有生成元必须属于 A
    raises(ValueError, lambda: A.submodule_from_gens([3*A(0), B(0)]))


def test_Module_submodule_from_matrix():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 使用单位矩阵生成一个 DomainMatrix 对象，再乘以 2，作为参数构建子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 使用向量 [1, 2, 3, 4] 作为参数调用 B，得到元素 e
    e = B(to_col([1, 2, 3, 4]))
    # 将 e 转换回其父对象，得到元素 f
    f = e.to_parent()
    # 断言 f 的列向量平铺后等于 [2, 4, 6, 8]
    assert f.col.flat() == [2, 4, 6, 8]
    # 矩阵必须是整数矩阵 ZZ
    raises(ValueError, lambda: A.submodule_from_matrix(DomainMatrix.eye(4, QQ)))
    # 矩阵的行数必须等于模块 A 的生成元数量
    raises(ValueError, lambda: A.submodule_from_matrix(2 * DomainMatrix.eye(5, ZZ)))


def test_Module_whole_submodule():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 构建 A 的整个子模块 B
    B = A.whole_submodule()
    # 使用向量 [1, 2, 3, 4] 作为参数调用 B，得到元素 e
    e = B(to_col([1, 2, 3, 4]))
    # 将 e 转换回其父对象，得到元素 f
    f = e.to_parent()
    # 断言 f 的列向量平铺后等于 [1, 2, 3, 4]
    assert f.col.flat() == [1, 2, 3, 4]
    # 获取 B 的前四个生成元 e0, e1, e2, e3
    e0, e1, e2, e3 = B(0), B(1), B(2), B(3)
    # 断言 e2 * e3 == e0
    assert e2 * e3 == e0
    # 断言 e3 ** 2 == e1
    assert e3 ** 2 == e1


def test_PowerBasis_repr():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成一个次数为 5 的多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 T 构建一个 PowerBasis 对象 A
    A = PowerBasis(T)
    # 断言 A 的字符串表示等于 'PowerBasis(x**4 + x**3 + x**2 + x + 1)'
    assert repr(A) == 'PowerBasis(x**4 + x**3 + x**2 + x + 1)'


def test_PowerBasis_eq():
    # 创建
    # 定义一个预期的字典，表示一个四阶矩阵M，每个键值对对应一个矩阵元素的位置和数值
    exp = {
        0: {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]},
        1: {1: [0, 0, 1, 0], 2: [0, 0, 0, 1], 3: [-1, -1, -1, -1]},
        2: {2: [-1, -1, -1, -1], 3: [1, 0, 0, 0]},
        3: {3: [0, 1, 0, 0]}
    }
    # 进行断言，确保变量M与预期的exp相等
    assert M == exp
    # 再次进行断言，确保矩阵M中所有元素都是整数类型
    assert all(is_int(c) for u in M for v in M[u] for c in M[u][v])
def test_PowerBasis_represent():
    # 创建一个 Poly 对象，表示一个五次周期多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类将 Poly 对象 T 转换为功率基表示
    A = PowerBasis(T)
    # 创建一个列向量，将其转换为 PowerBasisElement 类型
    col = to_col([1, 2, 3, 4])
    # 将列向量 col 转换为 A 所在的基底下的元素 a
    a = A(col)
    # 断言 A 的 represent 方法将 a 转换回原始的列向量 col
    assert A.represent(a) == col
    # 创建一个带有分母为 2 的列向量 b，并断言 represent 方法引发 ClosureFailure 异常
    b = A(col, denom=2)
    raises(ClosureFailure, lambda: A.represent(b))


def test_PowerBasis_element_from_poly():
    # 创建一个 Poly 对象，表示一个五次周期多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类将 Poly 对象 T 转换为功率基表示
    A = PowerBasis(T)
    # 创建多项式 f = 1 + 2*x，并断言 element_from_poly 方法将其转换为列向量，检查系数
    f = Poly(1 + 2*x)
    assert A.element_from_poly(f).coeffs == [1, 2, 0, 0]
    # 创建多项式 g = x**4，并断言 element_from_poly 方法将其转换为列向量，检查系数
    g = Poly(x**4)
    assert A.element_from_poly(g).coeffs == [-1, -1, -1, -1]
    # 创建零多项式 h = 0，并断言 element_from_poly 方法将其转换为列向量，检查系数
    h = Poly(0, x)
    assert A.element_from_poly(h).coeffs == [0, 0, 0, 0]


def test_PowerBasis_element__conversions():
    # 创建一个 QQ 上的五次周期域 k 和七次周期域 L
    k = QQ.cyclotomic_field(5)
    L = QQ.cyclotomic_field(7)
    # 使用 PowerBasis 类将域 k 转换为 PowerBasis 类型 B
    B = PowerBasis(k)

    # ANP --> PowerBasisElement
    # 创建一个 ANP 元素 a，并使用 element_from_ANP 方法将其转换为 PowerBasisElement 类型 e
    a = k([QQ(1, 2), QQ(1, 3), 5, 7])
    e = B.element_from_ANP(a)
    # 断言 e 的系数和分母符合预期
    assert e.coeffs == [42, 30, 2, 3]
    assert e.denom == 6

    # PowerBasisElement --> ANP
    # 断言 e 的 to_ANP 方法将其转换回原始的 ANP 元素 a
    assert e.to_ANP() == a

    # 不能从不同域的 ANP 转换
    # 创建一个来自域 L 的 ANP 元素 d，并断言 element_from_ANP 方法引发 UnificationFailed 异常
    d = L([QQ(1, 2), QQ(1, 3), 5, 7])
    raises(UnificationFailed, lambda: B.element_from_ANP(d))

    # AlgebraicNumber --> PowerBasisElement
    # 将 ANP 元素 a 转换为代数数 alpha
    alpha = k.to_alg_num(a)
    # 使用 element_from_alg_num 方法将 alpha 转换为 PowerBasisElement 类型 eps
    eps = B.element_from_alg_num(alpha)
    # 断言 eps 的系数和分母符合预期
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6

    # PowerBasisElement --> AlgebraicNumber
    # 断言 eps 的 to_alg_num 方法将其转换回原始的代数数 alpha
    assert eps.to_alg_num() == alpha

    # 不能从不同域的代数数转换
    # 将 ANP 元素 d 转换为代数数 delta，并断言 element_from_alg_num 方法引发 UnificationFailed 异常
    delta = L.to_alg_num(d)
    raises(UnificationFailed, lambda: B.element_from_alg_num(delta))

    # 当我们不知道域时：
    # 创建一个 PowerBasis 类型 C，其基底是 k 的最小多项式
    C = PowerBasis(k.ext.minpoly)
    # 可以从代数数 alpha 转换为 PowerBasisElement 类型 eps
    eps = C.element_from_alg_num(alpha)
    # 断言 eps 的系数和分母符合预期
    assert eps.coeffs == [42, 30, 2, 3]
    assert eps.denom == 6
    # 但无法从 PowerBasisElement 类型 eps 转换回代数数
    raises(StructureError, lambda: eps.to_alg_num())


def test_Submodule_repr():
    # 创建一个 Poly 对象，表示一个五次周期多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类将 Poly 对象 T 转换为功率基表示
    A = PowerBasis(T)
    # 使用 A 的 submodule_from_matrix 方法创建一个子模块 B，并断言其字符串表示与预期相符
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ), denom=3)
    assert repr(B) == 'Submodule[[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]/3'


def test_Submodule_reduced():
    # 创建一个 Poly 对象，表示一个五次周期多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类将 Poly 对象 T 转换为功率基表示
    A = PowerBasis(T)
    # 使用 A 的 submodule_from_matrix 方法创建一个子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 使用 A 的 submodule_from_matrix 方法创建一个子模块 C，带有分母为 3
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    # 使用 C 的 reduced 方法创建子模块 D，并断言其分母为 1，且 D、C、B 相等
    D = C.reduced()
    assert D.denom == 1 and D == C == B


def test_Submodule_discard_before():
    # 创建一个 Poly 对象，表示一个五次周期多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类将 Poly 对象 T 转换为功率基表示
    A = PowerBasis(T)
    # 使用 A 的 submodule_from_matrix 方法创建一个子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 计算子模块 B 的乘法表
    B.compute_mult_tab()
    # 使用 B 的 discard_before 方法创建子模块 C，并断言 C 的父对象与 B 相同
    C = B.discard_before(2)
    assert C.parent == B.parent
    # 断言 B 是方形的、最大秩的 Hermite 标准形式，而 C 不是
    assert B.is_sq_maxrank_HNF() and not C.is_sq_maxrank_HNF()
    # 断言 C 的矩阵是 B 的矩阵的列切片，且 C 的乘法表与预期相符
    assert C.matrix == B.matrix[:, 2:]
    assert C.mult_tab() == {0: {0: [-2, -2], 1: [0, 0]}, 1: {1: [0, 0]}}


def test_Submodule_QQ_matrix():
    # 创建一个 Poly 对象，表示一个五次周期多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类将 Poly 对象 T 转换为功率基表示
    A = PowerBasis(T)
    # 使用 A 的 submodule_from_matrix 方法创建一个子模块 B
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4
    # 断言语句，用于检查 C.QQ_matrix 是否等于 B.QQ_matrix
    assert C.QQ_matrix == B.QQ_matrix
def test_Submodule_represent():
    # 创建多项式 T，使用 cyclotomic_poly 函数生成 5 次多项式
    T = Poly(cyclotomic_poly(5, x))
    # 创建 T 的幂基
    A = PowerBasis(T)
    # 创建 A 的子模块 B，由单位矩阵乘以 2 构成
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 创建 B 的子模块 C，由单位矩阵乘以 3 构成
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 创建向量并映射到 A 的列向量
    a0 = A(to_col([6, 12, 18, 24]))
    a1 = A(to_col([2, 4, 6, 8]))
    a2 = A(to_col([1, 3, 5, 7]))

    # 用 B 表示 a1，并断言结果平坦化后为 [1, 2, 3, 4]
    b1 = B.represent(a1)
    assert b1.flat() == [1, 2, 3, 4]

    # 用 C 表示 a0，并断言结果平坦化后为 [1, 2, 3, 4]
    c0 = C.represent(a0)
    assert c0.flat() == [1, 2, 3, 4]

    # 创建 Y，作为 A 的子模块，由给定矩阵转置生成
    Y = A.submodule_from_matrix(DomainMatrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], (3, 4), ZZ).transpose())

    # 创建多项式 U，使用 cyclotomic_poly 函数生成 7 次多项式
    U = Poly(cyclotomic_poly(7, x))
    # 创建 U 的幂基
    Z = PowerBasis(U)
    # 创建向量并映射到 Z 的列向量
    z0 = Z(to_col([1, 2, 3, 4, 5, 6]))

    # 断言 Y 无法表示 A(3)，引发 ClosureFailure 异常
    raises(ClosureFailure, lambda: Y.represent(A(3)))
    # 断言 B 无法表示 a2，引发 ClosureFailure 异常
    raises(ClosureFailure, lambda: B.represent(a2))
    # 断言 B 无法表示 z0，引发 ClosureFailure 异常
    raises(ClosureFailure, lambda: B.represent(z0))


def test_Submodule_is_compat_submodule():
    # 创建多项式 T，使用 cyclotomic_poly 函数生成 5 次多项式
    T = Poly(cyclotomic_poly(5, x))
    # 创建 T 的幂基
    A = PowerBasis(T)
    # 创建 A 的子模块 B，由单位矩阵乘以 2 构成
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 创建 A 的子模块 C，由单位矩阵乘以 3 构成
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 创建 C 的子模块 D，由单位矩阵乘以 5 构成
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    
    # 断言 B 是 C 的兼容子模块，结果为 True
    assert B.is_compat_submodule(C) is True
    # 断言 B 不是 A 的兼容子模块，结果为 False
    assert B.is_compat_submodule(A) is False
    # 断言 B 不是 D 的兼容子模块，结果为 False
    assert B.is_compat_submodule(D) is False


def test_Submodule_eq():
    # 创建多项式 T，使用 cyclotomic_poly 函数生成 5 次多项式
    T = Poly(cyclotomic_poly(5, x))
    # 创建 T 的幂基
    A = PowerBasis(T)
    # 创建 A 的子模块 B，由单位矩阵乘以 2 构成
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 创建 A 的子模块 C，由单位矩阵乘以 6 构成，denom=3
    C = A.submodule_from_matrix(6 * DomainMatrix.eye(4, ZZ), denom=3)
    
    # 断言 C 等于 B
    assert C == B


def test_Submodule_add():
    # 创建多项式 T，使用 cyclotomic_poly 函数生成 5 次多项式
    T = Poly(cyclotomic_poly(5, x))
    # 创建 T 的幂基
    A = PowerBasis(T)
    # 创建 A 的子模块 B，由给定矩阵转置生成，denom=6
    B = A.submodule_from_matrix(DomainMatrix([
        [4, 0, 0, 0],
        [0, 4, 0, 0],
    ], (2, 4), ZZ).transpose(), denom=6)
    # 创建 A 的子模块 C，由给定矩阵转置生成，denom=15
    C = A.submodule_from_matrix(DomainMatrix([
        [0, 10, 0, 0],
        [0,  0, 7, 0],
    ], (2, 4), ZZ).transpose(), denom=15)
    # 创建 A 的子模块 D，由给定矩阵转置生成，denom=30
    D = A.submodule_from_matrix(DomainMatrix([
        [20,  0,  0, 0],
        [ 0, 20,  0, 0],
        [ 0,  0, 14, 0],
    ], (3, 4), ZZ).transpose(), denom=30)
    
    # 断言 B + C 等于 D
    assert B + C == D

    # 创建多项式 U，使用 cyclotomic_poly 函数生成 7 次多项式
    U = Poly(cyclotomic_poly(7, x))
    # 创建 U 的幂基
    Z = PowerBasis(U)
    # 创建 Z 的子模块 Y，由 Z(0) 和 Z(1) 构成
    Y = Z.submodule_from_gens([Z(0), Z(1)])
    
    # 断言 B + Y 引发 TypeError 异常
    raises(TypeError, lambda: B + Y)


def test_Submodule_mul():
    # 创建多项式 T，使用 cyclotomic_poly 函数生成 5 次多项式
    T = Poly(cyclotomic_poly(5, x))
    # 创建 T 的幂基
    A = PowerBasis(T)
    # 创建 A 的子模块 C，由给定矩阵转置生成，denom=15
    C = A.submodule_from_matrix(DomainMatrix([
        [0, 10, 0, 0],
        [0, 0, 7, 0],
    ], (2, 4), ZZ).transpose(), denom=15)
    # 创建 A 的子模块 C1，由给定矩阵转置生成，denom=3
    C1 = A.submodule_from_matrix(DomainMatrix([
        [0, 20, 0, 0],
        [0, 0, 14, 0],
    ], (2, 4), ZZ).transpose(), denom=3)
    # 创建 A 的子模块 C2，由给定矩阵转置生成，denom=15
    C2 = A.submodule_from_matrix(DomainMatrix([
        [0, 0, 10, 0],
        [0, 0,  0, 7],
    ], (2, 4), ZZ).transpose(), denom=15)
    # 创建 A 的子模块 C3_unred，由给定矩阵转置生成，denom=225
    C3_unred = A.submodule_from_matrix(DomainMatrix([
        [0, 0, 100, 0],
        [0, 0, 0, 70],
        [0, 0, 0, 70],
        [-49, -49, -49, -49]
    ], (4, 4), ZZ).transpose(), denom=225)
    # 创建 A 的子模
    ], (3, 4), ZZ).transpose(), denom=225)
    # 对一个矩阵进行转置操作，返回转置后的结果
    assert C * 1 == C
    # 断言：C 乘以标量 1 等于 C 自身
    assert C ** 1 == C
    # 断言：C 的 1 次幂等于 C 自身
    assert C * 10 == C1
    # 断言：C 乘以标量 10 等于 C1
    assert C * A(1) == C2
    # 断言：C 乘以 A 类型的对象 1 等于 C2
    assert C.mul(C, hnf=False) == C3_unred
    # 断言：使用 C 的乘法方法（hnf=False 参数）与 C 相乘的结果等于 C3_unred
    assert C * C == C3
    # 断言：C 乘以 C 等于 C3
    assert C ** 2 == C3
    # 断言：C 的 2 次幂等于 C3
def test_Submodule_reduce_element():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 生成整体子模 B，并计算其作用于给定列向量 [90, 84, 80, 75] 的结果 b
    B = A.whole_submodule()
    b = B(to_col([90, 84, 80, 75]), denom=120)

    # 使用单位矩阵创建子模 C，将 b 降到以 DomainMatrix.eye(4, ZZ) 为基的子模上，denom=2
    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=2)
    # 预期的 b_bar_expected 是 B 作用于 [30, 24, 20, 15] 后的结果
    b_bar_expected = B(to_col([30, 24, 20, 15]), denom=120)
    # 通过 C.reduce_element 方法将 b 降到 C 上得到 b_bar
    b_bar = C.reduce_element(b)
    # 断言 b_bar 与预期值 b_bar_expected 相等
    assert b_bar == b_bar_expected

    # 使用单位矩阵创建子模 C，将 b 降到以 DomainMatrix.eye(4, ZZ) 为基的子模上，denom=4
    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=4)
    # 预期的 b_bar_expected 是 B 作用于 [0, 24, 20, 15] 后的结果
    b_bar_expected = B(to_col([0, 24, 20, 15]), denom=120)
    # 通过 C.reduce_element 方法将 b 降到 C 上得到 b_bar
    b_bar = C.reduce_element(b)
    # 断言 b_bar 与预期值 b_bar_expected 相等
    assert b_bar == b_bar_expected

    # 使用单位矩阵创建子模 C，将 b 降到以 DomainMatrix.eye(4, ZZ) 为基的子模上，denom=8
    C = B.submodule_from_matrix(DomainMatrix.eye(4, ZZ), denom=8)
    # 预期的 b_bar_expected 是 B 作用于 [0, 9, 5, 0] 后的结果
    b_bar_expected = B(to_col([0, 9, 5, 0]), denom=120)
    # 通过 C.reduce_element 方法将 b 降到 C 上得到 b_bar
    b_bar = C.reduce_element(b)
    # 断言 b_bar 与预期值 b_bar_expected 相等
    assert b_bar == b_bar_expected

    # 对于 A 生成的整体子模，试图将其作用于 [1, 2, 3, 4]，断言会引发 NotImplementedError
    a = A(to_col([1, 2, 3, 4]))
    raises(NotImplementedError, lambda: C.reduce_element(a))

    # 使用给定的系数矩阵创建子模 C，试图将其作用于 b，断言会引发 StructureError
    C = B.submodule_from_matrix(DomainMatrix([
        [5, 4, 3, 2],
        [0, 8, 7, 6],
        [0, 0, 11, 12],
        [0, 0, 0, 1]
    ], (4, 4), ZZ).transpose())
    raises(StructureError, lambda: C.reduce_element(b))


def test_is_HNF():
    # 定义不同的矩阵 M, M1, M2
    M = DM([
        [3, 2, 1],
        [0, 2, 1],
        [0, 0, 1]
    ], ZZ)
    M1 = DM([
        [3, 2, 1],
        [0, -2, 1],
        [0, 0, 1]
    ], ZZ)
    M2 = DM([
        [3, 2, 3],
        [0, 2, 1],
        [0, 0, 1]
    ], ZZ)
    # 断言 M 是方阵最大秩 HNF
    assert is_sq_maxrank_HNF(M) is True
    # 断言 M1 不是方阵最大秩 HNF
    assert is_sq_maxrank_HNF(M1) is False
    # 断言 M2 不是方阵最大秩 HNF
    assert is_sq_maxrank_HNF(M2) is False


def test_make_mod_elt():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 创建 B 作为 A 的子模，并将列向量 [1, 2, 3, 4] 转换为模元素 eA
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    col = to_col([1, 2, 3, 4])
    eA = make_mod_elt(A, col)
    eB = make_mod_elt(B, col)
    # 断言 eA 是 PowerBasisElement 类型，eB 不是
    assert isinstance(eA, PowerBasisElement)
    assert not isinstance(eB, PowerBasisElement)


def test_ModuleElement_repr():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 创建模元素 e，使用 to_col([1, 2, 3, 4]) 和 denom=2
    e = A(to_col([1, 2, 3, 4]), denom=2)
    # 断言 e 的字符串表示与预期相等
    assert repr(e) == '[1, 2, 3, 4]/2'


def test_ModuleElement_reduced():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 创建模元素 e，使用 to_col([2, 4, 6, 8]) 和 denom=2
    e = A(to_col([2, 4, 6, 8]), denom=2)
    # 对 e 进行约化操作，得到 f
    f = e.reduced()
    # 断言 f 的分母为 1，且 f 等于 e
    assert f.denom == 1 and f == e


def test_ModuleElement_reduced_mod_p():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 创建模元素 e，使用 to_col([20, 40, 60, 80])
    e = A(to_col([20, 40, 60, 80]))
    # 对 e 进行模 p=7 的约化操作，得到 f
    f = e.reduced_mod_p(7)
    # 断言 f 的系数为 [-1, -2, -3, 3]
    assert f.coeffs == [-1, -2, -3, 3]


def test_ModuleElement_from_int_list():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    c = [1, 2, 3, 4]
    # 使用整数列表 c 创建模元素，并断言其系数与 c 相等
    assert ModuleElement.from_int_list(A, c).coeffs == c


def test_ModuleElement_len():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 创建模元素 e，使用 0 作为输入
    e = A(0)
    # 断言 e 的长度为 4
    assert len(e) == 4


def test_ModuleElement_column():
    # 创建多项式 T，使用其生成整体子模
    T = Poly(cyclotomic_poly(5, x))
    A = PowerBasis(T)
    # 创建模元素 e，使用 0 作为输入
    e = A(0)
    # 获取 e 的列向量表示 col1
    col1 = e.column()
    # 断言 col1 等于 e.col 且不是同一个对象
    assert col1 == e.col and col1 is not e.col
    # 获取在有限域 FF(5) 上的列向量表示 col2
    col
    # 创建对象 e，使用列表 [1, 2, 3, 4] 转换为列，并指定分母为 1
    e = A(to_col([1, 2, 3, 4]), denom=1)
    # 创建对象 f，使用列表 [3, 6, 9, 12] 转换为列，并指定分母为 3
    f = A(to_col([3, 6, 9, 12]), denom=3)
    # 断言对象 e 的 QQ_col 属性等于对象 f 的 QQ_col 属性
    assert e.QQ_col == f.QQ_col
def test_ModuleElement_to_ancestors():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类构造 A，表示 T 的幂基
    A = PowerBasis(T)
    # 使用 A 的子模块方法，生成 B，它是由单位矩阵乘以 2 构成的子模块
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 使用 B 的子模块方法，生成 C，它是由单位矩阵乘以 3 构成的子模块
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 使用 C 的子模块方法，生成 D，它是由单位矩阵乘以 5 构成的子模块
    D = C.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    # 创建 eD，作为 D 中的一个元素，取值为 0
    eD = D(0)
    # 调用 eD 的 to_parent 方法，返回它的父模块元素 eC
    eC = eD.to_parent()
    # 调用 eD 的 to_ancestor 方法，返回它在 B 中的祖先 eB
    eB = eD.to_ancestor(B)
    # 调用 eD 的 over_power_basis 方法，返回它相对于 A 的幂基的元素 eA
    eA = eD.over_power_basis()
    # 断言语句，验证 eC 的模块是 C，且系数为 [5, 0, 0, 0]
    assert eC.module is C and eC.coeffs == [5, 0, 0, 0]
    # 断言语句，验证 eB 的模块是 B，且系数为 [15, 0, 0, 0]
    assert eB.module is B and eB.coeffs == [15, 0, 0, 0]
    # 断言语句，验证 eA 的模块是 A，且系数为 [30, 0, 0, 0]
    assert eA.module is A and eA.coeffs == [30, 0, 0, 0]

    # 创建 A 的一个元素 a，取值为 0
    a = A(0)
    # 断言语句，使用 lambda 函数验证当 a 调用 to_parent 方法时会引发 ValueError 异常
    raises(ValueError, lambda: a.to_parent())


def test_ModuleElement_compatibility():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类构造 A，表示 T 的幂基
    A = PowerBasis(T)
    # 使用 A 的子模块方法，生成 B，它是由单位矩阵乘以 2 构成的子模块
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 使用 B 的子模块方法，生成 C，它是由单位矩阵乘以 3 构成的子模块
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 使用 B 的子模块方法，生成 D，它是由单位矩阵乘以 5 构成的子模块
    D = B.submodule_from_matrix(5 * DomainMatrix.eye(4, ZZ))
    # 断言语句，验证 C 中的元素 C(0) 与 C 中的元素 C(1) 兼容
    assert C(0).is_compat(C(1)) is True
    # 断言语句，验证 C 中的元素 C(0) 与 D 中的元素 D(0) 不兼容
    assert C(0).is_compat(D(0)) is False
    # 调用 C(0) 的 unify 方法，传入 D(0) 作为参数，返回 u 和 v
    u, v = C(0).unify(D(0))
    # 断言语句，验证 u 的模块是 B，v 的模块也是 B
    assert u.module is B and v.module is B
    # 断言语句，验证 C 中的元素 C.represent(u) 等于 C(0)，D 中的元素 D.represent(v) 等于 D(0)
    assert C(C.represent(u)) == C(0) and D(D.represent(v)) == D(0)

    # 再次调用 C(0) 的 unify 方法，传入 C(1) 作为参数，返回 u 和 v
    u, v = C(0).unify(C(1))
    # 断言语句，验证 u 等于 C(0)，v 等于 C(1)
    assert u == C(0) and v == C(1)

    # 创建一个多项式 U，使用 cyclotomic_poly 函数生成 x 的第七个分圆多项式
    U = Poly(cyclotomic_poly(7, x))
    # 使用 PowerBasis 类构造 Z，表示 U 的幂基
    Z = PowerBasis(U)
    # 使用 lambda 函数验证当 C(0) 的 unify 方法传入 Z(1) 时会引发 UnificationFailed 异常
    raises(UnificationFailed, lambda: C(0).unify(Z(1)))


def test_ModuleElement_eq():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类构造 A，表示 T 的幂基
    A = PowerBasis(T)
    # 创建 e，作为 A 的一个元素，值为 [1, 2, 3, 4] 的列向量，分母为 1
    e = A(to_col([1, 2, 3, 4]), denom=1)
    # 创建 f，作为 A 的一个元素，值为 [3, 6, 9, 12] 的列向量，分母为 3
    f = A(to_col([3, 6, 9, 12]), denom=3)
    # 断言语句，验证 e 等于 f
    assert e == f

    # 创建一个多项式 U，使用 cyclotomic_poly 函数生成 x 的第七个分圆多项式
    U = Poly(cyclotomic_poly(7, x))
    # 使用 PowerBasis 类构造 Z，表示 U 的幂基
    Z = PowerBasis(U)
    # 断言语句，验证 e 不等于 Z(0)
    assert e != Z(0)
    # 断言语句，验证 e 不等于 3.14
    assert e != 3.14


def test_ModuleElement_equiv():
    # 创建一个多项式 T，使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类构造 A，表示 T 的幂基
    A = PowerBasis(T)
    # 创建 e，作为 A 的一个元素，值为 [1, 2, 3, 4] 的列向量，分母为 1
    e = A(to_col([1, 2, 3, 4]), denom=1)
    # 创建 f，作为 A 的一个元素，值为 [3, 6, 9, 12] 的列向量，分母为 3
    f = A(to_col([3, 6, 9, 12]), denom=3)
    # 断言语句，验证 e 等价于 f
    assert e.equiv(f)

    # 使用 A 的子模块方法，生成 C，它是由单位矩阵乘以 3 构成的子模块
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 创建 g，作为 C 的一个元素，值为 [1, 2, 3, 4] 的列向量，分母为 1
    g = C(to_col([1, 2, 3, 4]), denom=1)
    # 创建 h，作为
    # 使用 A 对象的方法 submodule_from_matrix 创建一个新的子模块 C，其内容为 3 倍的单位矩阵的域矩阵
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 使用 A 对象创建一个新的向量 e，元素为 [0, 2, 0, 0]，分母为 3
    e = A(to_col([0, 2, 0, 0]), denom=3)
    # 使用 A 对象创建一个新的向量 f，元素为 [0, 0, 0, 7]，分母为 5
    f = A(to_col([0, 0, 0, 7]), denom=5)
    # 使用 C 对象创建一个新的向量 g，元素为 [0, 0, 0, 1]，分母为 2
    g = C(to_col([0, 0, 0, 1]), denom=2)
    # 使用 A 对象创建一个新的向量 h，元素为 [0, 0, 3, 1]，分母为 7
    h = A(to_col([0, 0, 3, 1]), denom=7)
    # 断言：验证 e 与 f 的乘积等于 A 对象创建的向量，元素为 [-14, -14, -14, -14]，分母为 15
    assert e * f == A(to_col([-14, -14, -14, -14]), denom=15)
    # 断言：验证 e 与 g 的乘积等于 A 对象创建的向量，元素为 [-1, -1, -1, -1]
    assert e * g == A(to_col([-1, -1, -1, -1]))
    # 断言：验证 e 与 h 的乘积等于 A 对象创建的向量，元素为 [-2, -2, -2, 4]，分母为 21
    assert e * h == A(to_col([-2, -2, -2, 4]), denom=21)
    # 断言：验证 e 与 QQ(6, 5) 的乘积等于 A 对象创建的向量，元素为 [0, 4, 0, 0]，分母为 5
    assert e * QQ(6, 5) == A(to_col([0, 4, 0, 0]), denom=5)
    # 断言：验证 g 与 QQ(10, 21) 的乘积等价于 A 对象创建的向量，元素为 [0, 0, 0, 5]，分母为 7
    assert (g * QQ(10, 21)).equiv(A(to_col([0, 0, 0, 5]), denom=7))
    # 断言：验证 e 与 QQ(6, 5) 的整数除法结果等于 A 对象创建的向量，元素为 [0, 5, 0, 0]，分母为 9
    assert e // QQ(6, 5) == A(to_col([0, 5, 0, 0]), denom=9)

    # 使用 Poly 对象创建一个新的多项式 U，内容为 7 次旋转多项式
    U = Poly(cyclotomic_poly(7, x))
    # 使用 PowerBasis 对象创建一个基于多项式 U 的幂基
    Z = PowerBasis(U)
    # 断言：验证 e 与 Z(0) 的乘积会引发 TypeError 异常
    raises(TypeError, lambda: e * Z(0))
    # 断言：验证 e 与浮点数 3.14 的乘积会引发 TypeError 异常
    raises(TypeError, lambda: e * 3.14)
    # 断言：验证 e 与浮点数 3.14 的整数除法会引发 TypeError 异常
    raises(TypeError, lambda: e // 3.14)
    # 断言：验证 e 与整数 0 的整数除法会引发 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: e // 0)
# 定义测试函数 test_ModuleElement_div，用于测试模块元素的除法操作
def test_ModuleElement_div():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 从单位矩阵生成 A 的子模块 C
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 创建 A 中的模块元素 e 和 f，分别作为分子和分母的部分元素
    e = A(to_col([0, 2, 0, 0]), denom=3)
    f = A(to_col([0, 0, 0, 7]), denom=5)
    # 创建 C 中的模块元素 g
    g = C(to_col([1, 1, 1, 1]))
    # 断言 e 除以 f 的结果与给定的多项式对象相等
    assert e // f == 10*A(3)//21
    # 断言 e 除以 g 的结果与给定的多项式对象相等
    assert e // g == -2*A(2)//9
    # 断言 3 除以 g 的结果与给定的多项式对象相等
    assert 3 // g == -A(1)


# 定义测试函数 test_ModuleElement_pow，用于测试模块元素的幂运算
def test_ModuleElement_pow():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 从单位矩阵生成 A 的子模块 C
    C = A.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 创建 A 中的模块元素 e
    e = A(to_col([0, 2, 0, 0]), denom=3)
    # 创建 C 中的模块元素 g
    g = C(to_col([0, 0, 0, 1]), denom=2)
    # 断言 e 的三次幂与给定的多项式对象相等
    assert e ** 3 == A(to_col([0, 0, 0, 8]), denom=27)
    # 断言 g 的平方与给定的多项式对象相等
    assert g ** 2 == C(to_col([0, 3, 0, 0]), denom=4)
    # 断言 e 的零次幂与给定的多项式对象相等
    assert e ** 0 == A(to_col([1, 0, 0, 0]))
    # 断言 g 的零次幂与给定的多项式对象相等
    assert g ** 0 == A(to_col([1, 0, 0, 0]))
    # 断言 e 的一次幂与自身相等
    assert e ** 1 == e
    # 断言 g 的一次幂与自身相等
    assert g ** 1 == g


# 定义测试函数 test_ModuleElement_mod，用于测试模块元素的取模运算
def test_ModuleElement_mod():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 创建 A 中的模块元素 e
    e = A(to_col([1, 15, 8, 0]), denom=2)
    # 断言 e 对 7 取模的结果与给定的多项式对象相等
    assert e % 7 == A(to_col([1, 1, 8, 0]), denom=2)
    # 断言 e 对 QQ(1, 2) 取模的结果为零
    assert e % QQ(1, 2) == A.zero()
    # 断言 e 对 QQ(1, 3) 取模的结果与给定的多项式对象相等
    assert e % QQ(1, 3) == A(to_col([1, 1, 0, 0]), denom=6)

    # 从给定的生成元创建 A 的子模块 B
    B = A.submodule_from_gens([A(0), 5*A(1), 3*A(2), A(3)])
    # 断言 e 对 B 取模的结果与给定的多项式对象相等
    assert e % B == A(to_col([1, 5, 2, 0]), denom=2)

    # 获取 B 的整体子模块 C
    C = B.whole_submodule()
    # 断言尝试对 e 取模时引发 TypeError 异常
    raises(TypeError, lambda: e % C)


# 定义测试函数 test_PowerBasisElement_polys，用于测试幂基元素的多项式方法
def test_PowerBasisElement_polys():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 创建 A 中的模块元素 e
    e = A(to_col([1, 15, 8, 0]), denom=2)
    # 断言 e 在 zeta 处的分子多项式与给定的多项式对象相等
    assert e.numerator(x=zeta) == Poly(8 * zeta ** 2 + 15 * zeta + 1, domain=ZZ)
    # 断言 e 在 zeta 处的多项式与给定的多项式对象相等
    assert e.poly(x=zeta) == Poly(4 * zeta ** 2 + QQ(15, 2) * zeta + QQ(1, 2), domain=QQ)


# 定义测试函数 test_PowerBasisElement_norm，用于测试幂基元素的范数方法
def test_PowerBasisElement_norm():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 创建 A 中的模块元素 lam
    lam = A(to_col([1, -1, 0, 0]))
    # 断言 lam 的范数与给定的值相等
    assert lam.norm() == 5


# 定义测试函数 test_PowerBasisElement_inverse，用于测试幂基元素的逆运算
def test_PowerBasisElement_inverse():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 创建 A 中的模块元素 e
    e = A(to_col([1, 1, 1, 1]))
    # 断言 e 的两倍除以 e 的结果与给定的多项式对象相等
    assert 2 // e == -2*A(1)
    # 断言 e 的负三次幂与给定的多项式对象相等
    assert e ** -3 == -A(3)


# 定义测试函数 test_ModuleHomomorphism_matrix，用于测试模块同态的矩阵表示方法
def test_ModuleHomomorphism_matrix():
    # 使用 cyclotomic_poly 函数生成 x 的第五个分圆多项式，并转换为多项式对象 T
    T = Poly(cyclotomic_poly(5, x))
    # 构建 T 的幂基 A
    A = PowerBasis(T)
    # 创建以 a ↦ a^2 为映射的 A 的模块同态 phi
    phi = ModuleEndomorphism(A, lambda a: a ** 2)
    # 获取 phi 的矩阵表示 M
    M = phi.matrix()
    # 断言 M 与给定的整数矩阵对象相等
    assert M
    # 调用 raises 函数，验证 lambda 表达式 R.represent(3.14) 是否会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: R.represent(3.14))
# 定义一个测试函数，用于测试找到最小多项式的相关功能
def test_find_min_poly():
    # 创建一个多项式对象 T，表示 5 次分圆多项式
    T = Poly(cyclotomic_poly(5, x))
    # 使用 PowerBasis 类创建多项式 T 的基
    A = PowerBasis(T)
    # 初始化一个空列表 powers 用于存储幂次信息
    powers = []
    # 调用 find_min_poly 函数，找到 A(1) 的最小多项式 m
    m = find_min_poly(A(1), QQ, x=x, powers=powers)
    # 断言 m 应该等于 Poly(T, domain=QQ)，即 T 的多项式表示在有理数域 QQ 上
    assert m == Poly(T, domain=QQ)
    # 断言 powers 列表应该有 5 个元素，对应于 5 次分圆多项式的幂次
    assert len(powers) == 5

    # 第二个测试情况：不需要传递 powers 列表
    m = find_min_poly(A(1), QQ, x=x)
    # 再次断言 m 应该等于 Poly(T, domain=QQ)
    assert m == Poly(T, domain=QQ)

    # 创建 A 的一个子模块 B，使用 2 倍的单位矩阵创建 DomainMatrix 对象
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 断言调用 find_min_poly 函数时会抛出 MissingUnityError 异常
    raises(MissingUnityError, lambda: find_min_poly(B(1), QQ))
```