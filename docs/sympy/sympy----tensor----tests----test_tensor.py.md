# `D:\src\scipysrc\sympy\sympy\tensor\tests\test_tensor.py`

```
# 导入符号计算库中的特定模块和函数
from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add, Wild
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
    get_symmetric_group_sgs, TensorIndex, tensor_mul, TensAdd, \
    riemann_cyclic_replace, riemann_cyclic, TensMul, tensor_heads, \
    TensorManager, TensExpr, TensorHead, canon_bp, \
    tensorhead, tensorsymmetry, TensorType, substitute_indices, \
    WildTensorIndex, WildTensorHead, _WildTensExpr
# 导入测试相关的函数和装饰器
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
# 导入矩阵对角元素的函数
from sympy.matrices import diag

def _is_equal(arg1, arg2):
    # 判断两个对象是否相等，如果其中一个是张量表达式，则使用其equals方法比较
    if isinstance(arg1, TensExpr):
        return arg1.equals(arg2)
    elif isinstance(arg2, TensExpr):
        return arg2.equals(arg1)
    # 否则直接比较它们的值
    return arg1 == arg2


#################### Tests from tensor_can.py #######################

def test_canonicalize_no_slot_sym():
    # 创建洛伦兹指标类型
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义洛伦兹指标对象
    a, b, d0, d1 = tensor_indices('a,b,d0,d1', Lorentz)
    # 定义张量头
    A, B = tensor_heads('A,B', [Lorentz], TensorSymmetry.no_symmetry(1))
    # 创建张量表达式 A(-d0)*B(d0)
    t = A(-d0)*B(d0)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(L_0)*B(-L_0)'

    # 创建另一个张量表达式 A(a)*B(b)
    t = A(a)*B(b)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后结果与原始结果相同
    assert tc == t

    # 创建另一个张量表达式 B(b)*A(a)
    t1 = B(b)*A(a)
    # 使用Bianchi恒等式进行规范化
    tc = t1.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(a)*B(b)'

    # 创建对称的张量头A
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建张量表达式 A(b, -d0)*A(d0, a)
    t = A(b, -d0)*A(d0, a)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(a, L_0)*A(b, -L_0)'

    # 创建张量头B和C，它们都没有对称性
    B, C = tensor_heads('B,C', [Lorentz], TensorSymmetry.no_symmetry(1))
    # 创建张量表达式 A(d1, -d0)*B(d0)*C(-d1)
    t = A(d1, -d0)*B(d0)*C(-d1)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(L_0, L_1)*B(-L_0)*C(-L_1)'

    # 创建没有对称性的张量头A
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    # 创建张量表达式 A(d1, -d0)*B(d0)*C(-d1)
    t = A(d1, -d0)*B(d0)*C(-d1)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(L_0, L_1)*B(-L_1)*C(-L_0)'

    # 创建两个没有对称性的张量头A和B
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    # 创建张量表达式 A(d1, -d0)*B(-d1, d0)
    t = A(d1, -d0)*B(-d1, d0)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(L_0, L_1)*B(-L_0, -L_1)'
    
    # 创建张量表达式 A(-d0, d1)*B(-d1, d0)
    t = A(-d0, d1)*B(-d1, d0)
    # 使用Bianchi恒等式进行规范化
    tc = t.canon_bp()
    # 断言规范化后的字符串表示
    assert str(tc) == 'A(L_0, L_1)*B(-L_1, -L_0)'
    # A, B, C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b}
    # 定义一个张量 C，它有两个 Lorentz 指标，没有对称性
    C = TensorHead('C', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    # 创建一个张量 t，按照给定的指标顺序计算
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    # 对张量 t 进行正则化（规范化），使其按照一定规则重新排列指标
    tc = t.canon_bp()
    # 断言正则化后的字符串表示与预期字符串相等
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_1)*C(-L_0, -b)'
    
    # A symmetric, B and C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b}
    # 定义一个对称的张量 A，它有两个 Lorentz 指标
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建一个张量 t，按照给定的指标顺序计算
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    # 对张量 t 进行正则化（规范化），使其按照一定规则重新排列指标
    tc = t.canon_bp()
    # 断言正则化后的字符串表示与预期字符串相等
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-L_1, -b)'
    
    # A and C symmetric, B without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # 定义一个对称的张量 C，它有两个 Lorentz 指标
    C = TensorHead('C', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建一个张量 t，按照给定的指标顺序计算
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    # 对张量 t 进行正则化（规范化），使其按照一定规则重新排列指标
    tc = t.canon_bp()
    # 断言正则化后的字符串表示与预期字符串相等
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-b, -L_1)'
def test_canonicalize_no_dummies():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a, b, c, d', Lorentz)

    # A commuting
    # A^c A^b A^a
    # 构造张量头部 A，并且它是交换的
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    # 构造张量表达式 t = A^c * A^b * A^a
    t = A(c)*A(b)*A(a)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(a)*A(b)*A(c)'
    assert str(tc) == 'A(a)*A(b)*A(c)'

    # A anticommuting
    # A^c A^b A^a
    # 构造张量头部 A，并且它是反交换的
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    # 构造张量表达式 t = A^c * A^b * A^a
    t = A(c)*A(b)*A(a)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 '-A(a)*A(b)*A(c)'
    assert str(tc) == '-A(a)*A(b)*A(c)'

    # A commuting and symmetric
    # A^{b,d}*A^{c,a}
    # 构造张量头部 A，并且它是对称的
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 构造张量表达式 t = A^{b,d} * A^{c,a}
    t = A(b, d)*A(c, a)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(a, c)*A(b, d)'
    assert str(tc) == 'A(a, c)*A(b, d)'

    # A anticommuting and symmetric
    # A^{b,d}*A^{c,a}
    # 构造张量头部 A，并且它是反交换且对称的
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2), 1)
    # 构造张量表达式 t = A^{b,d} * A^{c,a}
    t = A(b, d)*A(c, a)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 '-A(a, c)*A(b, d)'
    assert str(tc) == '-A(a, c)*A(b, d)'

    # A^{c,a}*A^{b,d}
    # 构造张量表达式 t = A^{c,a} * A^{b,d}
    t = A(c, a)*A(b, d)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(a, c)*A(b, d)'
    assert str(tc) == 'A(a, c)*A(b, d)'

def test_tensorhead_construction_without_symmetry():
    L = TensorIndexType('Lorentz')
    A1 = TensorHead('A', [L, L])
    A2 = TensorHead('A', [L, L], TensorSymmetry.no_symmetry(2))
    assert A1 == A2
    A3 = TensorHead('A', [L, L], TensorSymmetry.fully_symmetric(2))  # Symmetric
    assert A1 != A3

def test_no_metric_symmetry():
    # no metric symmetry; A no symmetry
    # A^d1_d0 * A^d0_d1
    # 构造张量头部 A，没有度量对称性
    Lorentz = TensorIndexType('Lorentz', dummy_name='L', metric_symmetry=0)
    d0, d1, d2, d3 = tensor_indices('d:4', Lorentz)
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    # 构造张量表达式 t = A^d1_d0 * A^d0_d1
    t = A(d1, -d0)*A(d0, -d1)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(L_0, -L_1)*A(L_1, -L_0)'
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)'

    # A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    # 构造张量表达式 t = A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    t = A(d1, -d2)*A(d0, -d3)*A(d2, -d1)*A(d3, -d0)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(L_0, -L_1)*A(L_1, -L_0)*A(L_2, -L_3)*A(L_3, -L_2)'
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)*A(L_2, -L_3)*A(L_3, -L_2)'

    # A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    # 构造张量表达式 t = A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    t = A(d0, -d1)*A(d1, -d2)*A(d2, -d3)*A(d3, -d0)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(L_0, -L_1)*A(L_1, -L_2)*A(L_2, -L_3)*A(L_3, -L_0)'

def test_canonicalize1():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Lorentz)

    # A_d0*A^d0; ord = [d0,-d0]
    # 构造张量头部 A，没有对称性
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    # 构造张量表达式 t = A_d0 * A^d0
    t = A(-d0)*A(d0)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言结果字符串化为 'A(L_0)*A(-L_0)'

    # A commuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0
    # 构造张量表达式 t = A_d0 * A_d1 * A_d2 * A^d2 * A^d1 * A^d0
    t = A(-d0)*A(-d1)*A(-d2)*A(d2)*A(d1)*A(d0)
    # 使用 BP 算法进行正则排序
    tc = t.canon_bp()
    # 断言，验证字符串表示的张量等式是否成立
    assert str(tc) == 'A(L_0)*A(-L_0)*A(L_1)*A(-L_1)*A(L_2)*A(-L_2)'

    # 定义 A 张量头，具有 Lorentz 指标，无对称性，秩为 1
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    # 构建张量表达式 t = A_d0*A_d1*A_d2*A^d2*A^d1*A^d0
    t = A(-d0)*A(-d1)*A(-d2)*A(d2)*A(d1)*A(d0)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的字符串表示是否为零
    assert tc == 0

    # 定义 A 张量头，具有 Lorentz 指标，全对称，秩为 2
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 构建张量表达式 t = A^{d0 b}*A^a_d1*A^d1_d0
    t = A(d0, b)*A(a, -d1)*A(d1, -d0)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的字符串表示是否符合预期
    assert str(tc) == 'A(a, L_0)*A(b, L_1)*A(-L_0, -L_1)'

    # 定义 A 张量头和 B 张量头，都具有 Lorentz 指标，全对称，秩为 2
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 构建张量表达式 t = A^{d0 b}*A^d1_d0*B^a_d1
    t = A(d0, b)*A(d1, -d0)*B(a, -d1)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的字符串表示是否符合预期
    assert str(tc) == 'A(b, L_0)*A(-L_0, L_1)*B(a, -L_1)'

    # 定义 A 张量头，具有 Lorentz 指标，全对称，秩为 3
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3))
    # 构建张量表达式 t = A^{d1 d0 b}*A^{a}_{d1 d0}
    t = A(d1, d0, b)*A(a, -d1, -d0)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的字符串表示是否符合预期
    assert str(tc) == 'A(a, L_0, L_1)*A(b, -L_0, -L_1)'

    # 构建张量表达式 t = A^{d3 d0 d2}*A^a0_{d1 d2}*A^d1_d3^a1*A^{a2 a3}_d0
    t = A(d3, d0, d2)*A(a0, -d1, -d2)*A(d1, -d3, a1)*A(a2, a3, -d0)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的字符串表示是否符合预期
    assert str(tc) == 'A(a0, L_0, L_1)*A(a1, -L_0, L_2)*A(a2, a3, L_3)*A(-L_1, -L_2, -L_3)'

    # 定义 A 张量头和 B 张量头，A 全对称，秩为 3，B 全反对称，秩为 -2
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3))
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
    # 构建张量表达式 t = A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的结果是否为零
    assert tc == 0

    # 定义 A 张量头和 B 张量头，A 全对称反交换，秩为 3，B 全反对称，秩为 -2
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
    # 构建张量表达式 t = A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言，验证正则排序后的字符串表示是否符合预期
    assert str(tc) == 'A(L_0, L_1, L_2)*A(-L_0, -L_1, L_3)*B(-L_2, -L_3)'

    # 定义 A 张量头和 B 张量头，A 全对称反交换，秩为 3，B 全反对称，秩为 -2
    Spinor = TensorIndexType('Spinor', dummy_name='S', metric_symmetry=-1)
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Spinor)
    A = TensorHead('A', [Spinor]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Spinor]*2, TensorSymmetry.fully_symmetric(-2))
    # 构建张量表达式 t = A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    # 使用 canon_bp 方法进行正则排序
    tc = t.canon_bp()
    # 断言：验证表达式是否相等
    assert str(tc) == '-A(S_0, S_1, S_2)*A(-S_0, -S_1, S_3)*B(-S_2, -S_3)'

    # 创建一个名称为 Mat 的张量索引类型，指定无度量对称性和虚拟名称 'M'
    Mat = TensorIndexType('Mat', metric_symmetry=0, dummy_name='M')

    # 分配张量指标给变量
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Mat)

    # 创建一个张量头部 A，具有三个 Mat 类型的指标，完全对称，等级为 1
    A = TensorHead('A', [Mat]*3, TensorSymmetry.fully_symmetric(3), 1)

    # 创建一个张量头部 B，具有两个 Mat 类型的指标，完全反对称
    B = TensorHead('B', [Mat]*2, TensorSymmetry.fully_symmetric(-2))

    # 创建张量 t，由 A 和 B 组成的表达式
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)

    # 使用 canon_bp() 方法对张量 t 进行某种规范化处理，赋值给 tc
    tc = t.canon_bp()

    # 再次断言：验证规范化后的表达式是否符合预期
    assert str(tc) == 'A(M_0, M_1, M_2)*A(-M_0, -M_1, -M_3)*B(-M_2, M_3)'

    # 分配 Lorentz 索引类型的张量指标给变量
    alpha, beta, gamma, mu, nu, rho = \
        tensor_indices('alpha,beta,gamma,mu,nu,rho', Lorentz)

    # 创建一个 Lorentz 类型的张量头部 Gamma，具有一个完全对称的指标，等级为 2
    Gamma = TensorHead('Gamma', [Lorentz],
                       TensorSymmetry.fully_symmetric(1), 2)

    # 创建一个 Lorentz 类型的张量头部 Gamma，具有两个完全对称的指标，等级为 2
    Gamma2 = TensorHead('Gamma', [Lorentz]*2,
                        TensorSymmetry.fully_symmetric(-2), 2)

    # 创建一个 Lorentz 类型的张量头部 Gamma，具有三个完全对称的指标，等级为 2
    Gamma3 = TensorHead('Gamma', [Lorentz]*3,
                        TensorSymmetry.fully_symmetric(-3), 2)

    # 创建张量 t，由 Gamma2, Gamma 和 Gamma3 组成的表达式
    t = Gamma2(-mu, -nu)*Gamma(rho)*Gamma3(nu, mu, alpha)

    # 使用 canon_bp() 方法对张量 t 进行某种规范化处理，赋值给 tc
    tc = t.canon_bp()

    # 再次断言：验证规范化后的表达式是否符合预期
    assert str(tc) == '-Gamma(L_0, L_1)*Gamma(rho)*Gamma(alpha, -L_0, -L_1)'

    # 创建张量 t，由 Gamma2, Gamma2, Gamma 和 Gamma3 组成的表达式
    t = Gamma2(mu, nu)*Gamma2(beta, gamma)*Gamma(-rho)*Gamma3(alpha, -mu, -nu)

    # 使用 canon_bp() 方法对张量 t 进行某种规范化处理，赋值给 tc
    tc = t.canon_bp()

    # 再次断言：验证规范化后的表达式是否符合预期
    assert str(tc) == 'Gamma(L_0, L_1)*Gamma(beta, gamma)*Gamma(-rho)*Gamma(alpha, -L_0, -L_1)'

    # 创建一个名称为 Flavor 的张量索引类型，指定虚拟名称 'F'
    Flavor = TensorIndexType('Flavor', dummy_name='F')

    # 分配张量指标给变量
    a, b, c, d, e, ff = tensor_indices('a,b,c,d,e,f', Flavor)

    # 分配 Lorentz 索引类型的张量指标给变量
    mu, nu = tensor_indices('mu,nu', Lorentz)

    # 创建一个张量头部 f，具有三个 Flavor 类型的指标，直接积，等级为 -2
    f = TensorHead('f', [Flavor]*3, TensorSymmetry.direct_product(1, -2))

    # 创建一个 Lorentz 和 Flavor 类型的张量头部 A，无对称性，等级为 2
    A = TensorHead('A', [Lorentz, Flavor], TensorSymmetry.no_symmetry(2))

    # 创建张量 t，由 f 和 A 组成的表达式
    t = f(c, -d, -a)*f(-c, -e, -b)*A(-mu, d)*A(-nu, a)*A(nu, e)*A(mu, b)

    # 使用 canon_bp() 方法对张量 t 进行某种规范化处理，赋值给 tc
    tc = t.canon_bp()

    # 再次断言：验证规范化后的表达式是否符合预期
    assert str(tc) == '-f(F_0, F_1, F_2)*f(-F_0, F_3, F_4)*A(L_0, -F_1)*A(-L_0, -F_3)*A(L_1, -F_2)*A(-L_1, -F_4)'
def test_bug_correction_tensor_indices():
    # 确保当创建一个索引时，tensor_indices 不返回列表或元组
    A = TensorIndexType("A")
    i = tensor_indices('i', A)
    assert not isinstance(i, (tuple, list))
    assert isinstance(i, TensorIndex)


def test_riemann_invariants():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 = \
        tensor_indices('d0:12', Lorentz)
    # R^{d0 d1}_{d1 d0}; ord = [d0,-d0,d1,-d1]
    # 定义张量 R
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    # 创建张量 t，并进行基本排序
    t = R(d0, d1, -d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == '-R(L_0, L_1, -L_0, -L_1)'

    # R_d11^d1_d0^d5 * R^{d6 d4 d0}_d5 * R_{d7 d2 d8 d9} *
    # R_{d10 d3 d6 d4} * R^{d2 d7 d11}_d1 * R^{d8 d9 d3 d10}
    # can = [0,2,4,6, 1,3,8,10, 5,7,12,14, 9,11,16,18, 13,15,20,22,
    #        17,19,21<F10,23, 24,25]
    # 创建张量 t 并进行基本排序
    t = R(-d11,d1,-d0,d5)*R(d6,d4,d0,-d5)*R(-d7,-d2,-d8,-d9)* \
        R(-d10,-d3,-d6,-d4)*R(d2,d7,d11,-d1)*R(d8,d9,d3,d10)
    tc = t.canon_bp()
    assert str(tc) == 'R(L_0, L_1, L_2, L_3)*R(-L_0, -L_1, L_4, L_5)*R(-L_2, -L_3, L_6, L_7)*R(-L_4, -L_5, L_8, L_9)*R(-L_6, -L_7, L_10, L_11)*R(-L_8, -L_9, -L_10, -L_11)'

def test_riemann_products():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    d0, d1, d2, d3, d4, d5, d6 = tensor_indices('d0:7', Lorentz)
    a0, a1, a2, a3, a4, a5 = tensor_indices('a0:6', Lorentz)
    a, b = tensor_indices('a,b', Lorentz)
    # 定义张量 R
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())

    # R^{a b d0}_d0 = 0
    # 创建张量 t 并进行基本排序
    t = R(a, b, d0, -d0)
    tc = t.canon_bp()
    assert tc == 0

    # R^{d0 b a}_d0
    # 创建张量 t 并进行基本排序
    t = R(d0, b, a, -d0)
    tc = t.canon_bp()
    assert str(tc) == '-R(a, L_0, b, -L_0)'

    # R^d1_d2^b_d0 * R^{d0 a}_d1^d2; ord=[a,b,d0,-d0,d1,-d1,d2,-d2]
    # 创建张量 t 并进行基本排序
    t = R(d1, -d2, b, -d0)*R(d0, a, -d1, d2)
    tc = t.canon_bp()
    assert str(tc) == '-R(a, L_0, L_1, L_2)*R(b, -L_0, -L_1, -L_2)'

    # A symmetric commuting
    # 创建张量 V
    V = TensorHead('V', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建张量 t 并进行基本排序
    t = R(d6, d5, -d2, d1)*R(d4, d0, d2, d3)*V(-d6, -d0)*V(-d3, -d1)*V(-d4, -d5)
    tc = t.canon_bp()
    assert str(tc) == '-R(L_0, L_1, L_2, L_3)*R(-L_0, L_4, L_5, L_6)*V(-L_1, -L_4)*V(-L_2, -L_5)*V(-L_3, -L_6)'

    # R^{d2 a0 a2 d0} * R^d1_d2^{a1 a3} * R^{a4 a5}_{d0 d1}
    # 创建张量 t 并进行基本排序
    t = R(d2, a0, a2, d0)*R(d1, -d2, a1, a3)*R(a4, a5, -d0, -d1)
    tc = t.canon_bp()
    # 断言语句，用于确保表达式 str(tc) 的结果与字符串 'R(a0, L_0, a2, L_1)*R(a1, a3, -L_0, L_2)*R(a4, a5, -L_1, -L_2)' 相等。
    assert str(tc) == 'R(a0, L_0, a2, L_1)*R(a1, a3, -L_0, L_2)*R(a4, a5, -L_1, -L_2)'
# 定义测试函数 `test_canonicalize2`
def test_canonicalize2():
    # 定义符号 `D`
    D = Symbol('D')
    # 定义张量索引类型 `Eucl`，具有对称度 1、维度 `D`，虚拟名称为 'E'
    Eucl = TensorIndexType('Eucl', metric_symmetry=1, dim=D, dummy_name='E')
    # 定义多个张量索引 `i0` 到 `i14`，属于类型 `Eucl`
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14 = \
        tensor_indices('i0:15', Eucl)
    # 定义张量头 `A`，包含三个 `Eucl` 类型的张量索引，符号对称度为 -3
    A = TensorHead('A', [Eucl]*3, TensorSymmetry.fully_symmetric(-3))

    # 创建张量 `t`，按照 Kuratowski 图的方式进行张量缩并
    t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i3,i7,i5)*A(-i2,-i5,i6)*A(-i4,-i6,i8)
    # 对张量 `t` 进行正则化处理
    t1 = t.canon_bp()
    # 断言正则化后的结果为零
    assert t1 == 0

    # 创建张量 `t`，按照另一种方式进行缩并，参考文献中的公式
    t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i2,i5,i6)*A(-i3,i7,i8)*A(-i6,-i7,i9)*\
        A(-i8,i10,i13)*A(-i5,-i10,i11)*A(-i4,-i11,i12)*A(-i9,-i12,i14)
    # 对张量 `t` 进行正则化处理
    t1 = t.canon_bp()
    # 断言正则化后的结果为零
    assert t1 == 0


# 定义测试函数 `test_canonicalize3`
def test_canonicalize3():
    # 定义符号 `D`
    D = Symbol('D')
    # 定义张量索引类型 `Spinor`，维度 `D`，度规对称性为 -1，虚拟名称为 'S'
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    # 定义多个张量索引 `a0` 到 `a4`，属于类型 `Spinor`
    a0,a1,a2,a3,a4 = tensor_indices('a0:5', Spinor)
    # 定义张量头 `chi` 和 `psi`，分别包含 `Spinor` 类型的张量索引，没有对称性，阶数为 1
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)

    # 创建张量 `t`，包含 `chi(a1)` 和 `psi(a0)` 的乘积
    t = chi(a1)*psi(a0)
    # 对张量 `t` 进行正则化处理
    t1 = t.canon_bp()
    # 断言正则化后的结果与原张量相等
    assert t1 == t

    # 创建张量 `t`，包含 `psi(a1)` 和 `chi(a0)` 的乘积
    t = psi(a1)*chi(a0)
    # 对张量 `t` 进行正则化处理
    t1 = t.canon_bp()
    # 断言正则化后的结果为 `-chi(a0)*psi(a1)`
    assert t1 == -chi(a0)*psi(a1)


# 定义测试函数 `test_TensorIndexType`
def test_TensorIndexType():
    # 定义符号 `D`
    D = Symbol('D')
    # 定义张量索引类型 `Lorentz`，度规名称为 'g'，度规对称性为 1，维度 `D`，虚拟名称为 'L'
    Lorentz = TensorIndexType('Lorentz', metric_name='g', metric_symmetry=1,
                              dim=D, dummy_name='L')
    # 定义多个张量索引 `m0` 到 `m4`，属于类型 `Lorentz`
    m0, m1, m2, m3, m4 = tensor_indices('m0:5', Lorentz)
    # 定义对称性为 2 的张量对称性对象 `sym2` 和从对称群生成的对称性对象 `sym2n`
    sym2 = TensorSymmetry.fully_symmetric(2)
    sym2n = TensorSymmetry(*get_symmetric_group_sgs(2))
    # 断言 `sym2` 和 `sym2n` 相等
    assert sym2 == sym2n
    # 获取 `Lorentz` 类型的度规 `g`
    g = Lorentz.metric
    # 断言度规 `g` 的字符串表示为 'g(Lorentz,Lorentz)'
    assert str(g) == 'g(Lorentz,Lorentz)'
    # 断言 `Lorentz` 类型的 ε 维度与维度 `D` 相等
    assert Lorentz.eps_dim == Lorentz.dim

    # 定义张量索引类型 `TSpace`，虚拟名称为 'TSpace'
    TSpace = TensorIndexType('TSpace', dummy_name='TSpace')
    # 定义两个张量索引 `i0` 和 `i1`，属于类型 `TSpace`
    i0, i1 = tensor_indices('i0 i1', TSpace)
    # 获取 `TSpace` 类型的度规 `g`
    g = TSpace.metric
    # 定义张量头 `A`，包含两个 `TSpace` 类型的张量索引，对称性为 `sym2`
    A = TensorHead('A', [TSpace]*2, sym2)
    # 断言 `A(i0,-i0).canon_bp()` 的字符串表示为 'A(TSpace_0, -TSpace_0)'
    assert str(A(i0,-i0).canon_bp()) == 'A(TSpace_0, -TSpace_0)'


# 定义测试函数 `test_indices`
def test_indices():
    # 定义张量索引类型 `Lorentz`，虚拟名称为 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义多个张量索引 `a, b, c, d`，属于类型 `Lorentz`
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    # 断言张量索引 `a` 的张量索引类型为 `Lorentz`
    assert a.tensor_index_type == Lorentz
    # 断言张量索引 `a` 不等于 `-a`
    assert a != -a
    # 定义张量头 `A` 和 `B`，分别包含两个 `Lorentz` 类型的张量索引，对称性为 2
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建张量 `t`，包含 `A(a,b)` 和 `B(-b,c)` 的乘积
    t = A(a,b)*B(-b,c)
    # 获取张量 `t` 的所有索引
    indices = t.get_indices()
    # 定义 `L_0` 作为 `Lorentz` 类型的张量索引
    L_0 = TensorIndex('L_0', Lorentz)
    # 断言 `indices` 列表与 `[a, L_0, -L_0, c]` 相等
    assert indices == [a, L_0, -L_0, c]
    # 断言调用 `tensor_indices(3, Lorentz)` 时会引发 `ValueError` 异常
    raises(ValueError, lambda: tensor_indices(3, Lorentz))
    # 断言调用 `A(a,b,c)` 时会引发 `ValueError` 异常
    raises(ValueError, lambda: A(a,b,c))

    # 定义张量头 `A`，包含两个 `Lorentz` 类
    # 断言：验证表达式 A('a', TensorIndex('b', Lorentz)) == A(TensorIndex('a', Lorentz), TensorIndex('b', Lorentz))
    assert A('a', TensorIndex('b', Lorentz)) == A(TensorIndex('a', Lorentz),
                                                  TensorIndex('b', Lorentz))
# 定义测试函数 test_TensorSymmetry，用于测试 TensorSymmetry 类的方法和属性
def test_TensorSymmetry():
    # 断言检查 TensorSymmetry.fully_symmetric(2) 返回的结果与预期相符
    assert TensorSymmetry.fully_symmetric(2) == \
        TensorSymmetry(get_symmetric_group_sgs(2))
    # 断言检查 TensorSymmetry.fully_symmetric(-3) 返回的结果与预期相符
    assert TensorSymmetry.fully_symmetric(-3) == \
        TensorSymmetry(get_symmetric_group_sgs(3, True))
    # 断言检查 TensorSymmetry.direct_product(-4) 返回的结果与 TensorSymmetry.fully_symmetric(-4) 相同
    assert TensorSymmetry.direct_product(-4) == \
        TensorSymmetry.fully_symmetric(-4)
    # 断言检查 TensorSymmetry.fully_symmetric(-1) 返回的结果与 TensorSymmetry.fully_symmetric(1) 相同
    assert TensorSymmetry.fully_symmetric(-1) == \
        TensorSymmetry.fully_symmetric(1)
    # 断言检查 TensorSymmetry.direct_product(1, -1, 1) 返回的结果与 TensorSymmetry.no_symmetry(3) 相同
    assert TensorSymmetry.direct_product(1, -1, 1) == \
        TensorSymmetry.no_symmetry(3)
    # 断言检查 TensorSymmetry(get_symmetric_group_sgs(2)) 返回的结果与 TensorSymmetry(*get_symmetric_group_sgs(2)) 相同
    assert TensorSymmetry(get_symmetric_group_sgs(2)) == \
        TensorSymmetry(*get_symmetric_group_sgs(2))
    # TODO: add check for *get_symmetric_group_sgs(0)
    # 创建 TensorSymmetry 对象 sym，用于后续断言检查
    sym = TensorSymmetry.fully_symmetric(-3)
    # 断言检查 sym.rank 的值是否为 3
    assert sym.rank == 3
    # 断言检查 sym.base 的值是否为 Tuple(0, 1)
    assert sym.base == Tuple(0, 1)
    # 断言检查 sym.generators 的值是否为 Tuple(Permutation(0, 1)(3, 4), Permutation(1, 2)(3, 4))
    assert sym.generators == Tuple(Permutation(0, 1)(3, 4), Permutation(1, 2)(3, 4))

# 定义测试函数 test_TensExpr，用于测试 TensExpr 类的方法和属性
def test_TensExpr():
    # 创建 Lorentz 张量指标类型
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 创建张量指标 a, b, c, d
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    # 获取 Lorentz 度量张量 g
    g = Lorentz.metric
    # 创建张量头 A, B，并指定其对称性为 TensorSymmetry.fully_symmetric(2)
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 断言检查 g(c, d)/g(a, b) 抛出 ValueError 异常
    raises(ValueError, lambda: g(c, d)/g(a, b))
    # 断言检查 S.One/g(a, b) 抛出 ValueError 异常
    raises(ValueError, lambda: S.One/g(a, b))
    # 断言检查 (A(c, d) + g(c, d))/g(a, b) 抛出 ValueError 异常
    raises(ValueError, lambda: (A(c, d) + g(c, d))/g(a, b))
    # 断言检查 S.One/(A(c, d) + g(c, d)) 抛出 ValueError 异常
    raises(ValueError, lambda: S.One/(A(c, d) + g(c, d)))
    # 断言检查 A(a, b) + A(a, c) 抛出 ValueError 异常
    raises(ValueError, lambda: A(a, b) + A(a, c))

    # 创建 A(b, -d0)*B(d0, a) 的张量表达式 t1，并进行断言检查
    t1 = A(b, -d0)*B(d0, a)
    assert TensAdd(t1).equals(t1)
    # 创建 t2a = B(d0, a) + A(d0, a)，并用其构造张量表达式 t2
    t2a = B(d0, a) + A(d0, a)
    t2 = A(b, -d0)*t2a

# 定义测试函数 test_TensorHead，用于测试 TensorHead 类的方法和属性
def test_TensorHead():
    # 创建 Lorentz 张量指标类型
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 创建张量头 A，指定其张量指标类型为 Lorentz*2，即 [Lorentz, Lorentz]
    A = TensorHead('A', [Lorentz]*2)
    # 断言检查 A.name 的值是否为 'A'
    assert A.name == 'A'
    # 断言检查 A.index_types 的值是否为 [Lorentz, Lorentz]
    assert A.index_types == [Lorentz, Lorentz]
    # 断言检查 A.rank 的值是否为 2
    assert A.rank == 2
    # 断言检查 A.symmetry 的值是否为 TensorSymmetry.no_symmetry(2)
    assert A.symmetry == TensorSymmetry.no_symmetry(2)
    # 断言检查 A.comm 的值是否为 0
    assert A.comm == 0

# 定义测试函数 test_add1，用于测试 TensAdd 类的方法和属性
def test_add1():
    # 断言检查 TensAdd().args 的值是否为空元组 ()
    assert TensAdd().args == ()
    # 断言检查 TensAdd().doit() 的值是否为 0
    assert TensAdd().doit() == 0
    # 创建 Lorentz 张量指标类型
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 创建张量指标 a, b, d0, d1, i, j, k
    a, b, d0, d1, i, j, k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    # 创建 A, B 并指定其对称性为 TensorSymmetry.fully_symmetric(2)
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建张量表达式 t1 = A(b, -d0)*B(d0, a)，并进行断言检查
    t1 = A(b, -d0)*B(d0, a)
    assert TensAdd(t1).equals(t1)
    # 创建 t2a = B(d0, a) + A(d0, a)，并用其构造张量表达式 t2
    t2a = B(d0, a) + A(d0, a)
    t2 = A(b, -d0)*t2a
    # 断言确保字符串表示形式与预期相符
    assert str(t2) == 'A(b, -L_0)*(A(L_0, a) + B(L_0, a))'
    # 展开表达式
    t2 = t2.expand()
    # 断言展开后的字符串表示形式与预期相符
    assert str(t2) == 'A(b, -L_0)*A(L_0, a) + A(b, -L_0)*B(L_0, a)'
    # 对表达式进行 BP 规范化处理
    t2 = t2.canon_bp()
    # 断言规范化后的字符串表示形式与预期相符
    assert str(t2) == 'A(a, L_0)*A(b, -L_0) + A(b, L_0)*B(a, -L_0)'
    # 将 t2 和 t1 相加
    t2b = t2 + t1
    # 断言相加后的字符串表示形式与预期相符
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + A(b, -L_0)*B(L_0, a) + A(b, L_0)*B(a, -L_0)'
    # 对相加后的表达式进行 BP 规范化处理
    t2b = t2b.canon_bp()
    # 断言规范化后的字符串表示形式与预期相符
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + 2*A(b, L_0)*B(a, -L_0)'
    
    # 定义张量头部 p, q, r，并指定其类型为 Lorentz
    p, q, r = tensor_heads('p,q,r', [Lorentz])
    # 创建一个张量 t，其中 q(d0) 乘以 2
    t = q(d0)*2
    # 断言张量 t 的字符串表示形式与预期相符
    assert str(t) == '2*q(d0)'
    
    # 将 2 与 q(d0) 相乘，创建张量 t
    t = 2*q(d0)
    # 断言张量 t 的字符串表示形式与预期相符
    assert str(t) == '2*q(d0)'
    
    # 创建张量 t1，其中包含 p(d0) 和 2*q(d0)
    t1 = p(d0) + 2*q(d0)
    # 断言张量 t1 的字符串表示形式与预期相符
    assert str(t1) == '2*q(d0) + p(d0)'
    
    # 创建张量 t2，其中包含 p(-d0) 和 2*q(-d0)
    t2 = p(-d0) + 2*q(-d0)
    # 断言张量 t2 的字符串表示形式与预期相符
    assert str(t2) == '2*q(-d0) + p(-d0)'
    
    # 创建张量 t1，其中包含 p(d0)
    t1 = p(d0)
    # 创建张量 t3，为 t1 与 t2 的乘积
    t3 = t1*t2
    # 断言张量 t3 的字符串表示形式与预期相符
    assert str(t3) == 'p(L_0)*(2*q(-L_0) + p(-L_0))'
    
    # 展开张量 t3 的表达式
    t3 = t3.expand()
    # 断言展开后的张量 t3 的字符串表示形式与预期相符
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    
    # 将 t2 与 t1 相乘，然后展开表达式
    t3 = t2*t1
    t3 = t3.expand()
    # 断言展开后的张量 t3 的字符串表示形式与预期相符
    assert str(t3) == 'p(-L_0)*p(L_0) + 2*q(-L_0)*p(L_0)'
    
    # 对展开后的张量 t3 进行 BP 规范化处理
    t3 = t3.canon_bp()
    # 断言规范化后的张量 t3 的字符串表示形式与预期相符
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    
    # 重新定义张量 t1，并将其与 t2 相乘，然后对结果进行 BP 规范化处理
    t1 = p(d0) + 2*q(d0)
    t3 = t1*t2
    t3 = t3.canon_bp()
    # 断言规范化后的张量 t3 的字符串表示形式与预期相符
    assert str(t3) == 'p(L_0)*p(-L_0) + 4*p(L_0)*q(-L_0) + 4*q(L_0)*q(-L_0)'
    
    # 重新定义张量 t1，并将其定义为 p(d0) 减去 2*q(d0)
    t1 = p(d0) - 2*q(d0)
    # 断言张量 t1 的字符串表示形式与预期相符
    assert str(t1) == '-2*q(d0) + p(d0)'
    
    # 重新定义张量 t2，并将其定义为 p(-d0) 加上 2*q(-d0)
    t2 = p(-d0) + 2*q(-d0)
    # 创建张量 t3，为 t1 与 t2 的乘积，然后对结果进行 BP 规范化处理
    t3 = t1*t2
    t3 = t3.canon_bp()
    # 断言规范化后的张量 t3 与预期相符
    assert t3 == p(d0)*p(-d0) - 4*q(d0)*q(-d0)
    
    # 创建张量 t，包含多个乘积和加法项
    t = p(i)*p(j)*(p(k) + q(k)) + p(i)*(p(j) + q(j))*(p(k) - 3*q(k))
    # 对张量 t 进行 BP 规范化处理
    t = t.canon_bp()
    # 断言规范化后的张量 t 与预期相符
    assert t == 2*p(i)*p(j)*p(k) - 2*p(i)*p(j)*q(k) + p(i)*p(k)*q(j) - 3*p(i)*q(j)*q(k)
    
    # 创建张量 t1 和 t2
    t1 = (p(i) + q(i) + 2*r(i))*(p(j) - q(j))
    t2 = (p(j) + q(j) + 2*r(j))*(p(i) - q(i))
    # 将张量 t1 和 t2 相加
    t = t1 + t2
    # 对结果进行 BP 规范化处理
    t = t.canon_bp()
    # 断言规范化后的张量 t 与预期相符
    assert t == 2*p(i)*p(j) + 2*p(i)*r(j) + 2*p(j)*r(i) - 2*q(i)*q(j) - 2*q(i)*r(j) - 2*q(j)*r(i)
    
    # 创建张量 t，为 p(i)*q(j) 的一半
    t = p(i)*q(j)/2
    # 断言 2*t 与 p(i)*q(j) 相等
    assert 2*t == p(i)*q(j)
    
    # 创建张量 t，为 (p(i) + q(i)) 的一半
    t = (p(i) + q(i))/2
    # 断言 2*t 与 p(i) + q(i) 相等
    assert 2*t == p(i) + q(i)
    
    # 创建张量 t，为 S.One 减去 p(i)*p(-i)
    t = S.One - p(i)*p(-i)
    # 对 t 进行 BP 规范化处理
    t = t.canon_bp()
    # 创建张量 tz1，为 t 加上 p(-j)*p(j)
    tz1 = t + p(-j)*p(j)
    # 断言 tz1 不等于 1
    assert tz1 !=
def test_special_eq_ne():
    # 定义张量索引类型 'Lorentz'，并指定虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义张量索引变量 a, b, d0, d1, i, j, k，使用 Lorentz 类型
    a, b, d0, d1, i, j, k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    # 定义张量头 A, B，对称性为完全对称的二阶张量
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 定义张量头 p, q, r，使用 Lorentz 类型
    p, q, r = tensor_heads('p,q,r', [Lorentz])

    # 创建 t = 0*A(a, b)，并断言 t 等于数值 0
    t = 0*A(a, b)
    assert _is_equal(t, 0)
    # 断言 t 等于符号 S.Zero
    assert _is_equal(t, S.Zero)

    # 断言 p(i) 不等于 A(a, b)
    assert p(i) != A(a, b)
    # 断言 A(a, -a) 不等于 A(a, b)
    assert A(a, -a) != A(a, b)
    # 断言 0*(A(a, b) + B(a, b)) 等于 0
    assert 0*(A(a, b) + B(a, b)) == 0
    # 断言 0*(A(a, b) + B(a, b)) 是符号 S.Zero
    assert 0*(A(a, b) + B(a, b)) is S.Zero

    # 断言 3*(A(a, b) - A(a, b)) 是符号 S.Zero
    assert 3*(A(a, b) - A(a, b)) is S.Zero

    # 断言 p(i) + q(i) 不等于 A(a, b)
    assert p(i) + q(i) != A(a, b)
    # 断言 p(i) + q(i) 不等于 A(a, b) + B(a, b)
    assert p(i) + q(i) != A(a, b) + B(a, b)

    # 断言 p(i) - p(i) 等于 0
    assert p(i) - p(i) == 0
    # 断言 p(i) - p(i) 是符号 S.Zero
    assert p(i) - p(i) is S.Zero

    # 断言 A(a, b) 等于 A(b, a)
    assert _is_equal(A(a, b), A(b, a))

def test_add2():
    # 定义张量索引类型 'Lorentz'，并指定虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义张量索引变量 m, n, p, q，使用 Lorentz 类型
    m, n, p, q = tensor_indices('m,n,p,q', Lorentz)
    # 定义张量头 R，具有 Riemann 对称性
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    # 定义张量头 A，对称性为完全对称的三阶张量
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(-3))
    # 创建 t1 = 2*R(m, n, p, q) - R(m, q, n, p) + R(m, p, n, q)
    t1 = 2*R(m, n, p, q) - R(m, q, n, p) + R(m, p, n, q)
    # 创建 t2 = t1*A(-n, -p, -q)，并进行基础模式规范化
    t2 = t1*A(-n, -p, -q)
    t2 = t2.canon_bp()
    # 断言 t2 等于 0
    assert t2 == 0
    # 创建有理数张量 t1，并对其进行基础模式规范化
    t1 = Rational(2, 3)*R(m,n,p,q) - Rational(1, 3)*R(m,q,n,p) + Rational(1, 3)*R(m,p,n,q)
    t2 = t1*A(-n, -p, -q)
    t2 = t2.canon_bp()
    # 断言 t2 等于 0
    assert t2 == 0
    # 创建张量 t = A(m, -m, n) + A(n, p, -p)，并进行基础模式规范化
    t = A(m, -m, n) + A(n, p, -p)
    t = t.canon_bp()
    # 断言 t 等于 0
    assert t == 0

def test_add3():
    # 定义张量索引类型 'Lorentz'，并指定虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义张量索引变量 i0, i1，使用 Lorentz 类型
    i0, i1 = tensor_indices('i0:2', Lorentz)
    # 定义符号变量 E, px, py, pz
    E, px, py, pz = symbols('E px py pz')
    # 定义张量头 A, B，使用 Lorentz 类型
    A = TensorHead('A', [Lorentz])
    B = TensorHead('B', [Lorentz])

    # 创建表达式 expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)，并断言其参数列表
    expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)
    assert expr1.args == (-E**2, px**2, py**2, pz**2, A(i0)*A(-i0))

    # 创建表达式 expr2 = E**2 - px**2 - py**2 - pz**2 - A(i0)*A(-i0)，并断言其参数列表
    assert expr2.args == (E**2, -px**2, -py**2, -pz**2, -A(i0)*A(-i0))

    # 创建表达式 expr3 = A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2，并断言其参数列表
    expr3 = A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2
    assert expr3.args == (-E**2, px**2, py**2, pz**2, A(i0)*A(-i0))

    # 创建表达式 expr4 = B(i1)*B(-i1) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)，并断言其参数列表
    expr4 = B(i1)*B(-i1) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)
    assert expr4.args == (2*E**2, -2*px**2, -2*py**2, -2*pz**2, B(i1)*B(-i1), -A(i0)*A(-i0))

def test_mul():
    # 导入符号 x
    from sympy.abc import x
    # 定义张量索引类型 'Lorentz'，并指定虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义张量索引变量 a, b, c, d，使用 Lorentz 类型
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    # 创建单位张量 t，并断言其字符串表示为 '1'
    t = TensMul.from_data(S.One, [], [], [])
    assert str(t) == '1'
    # 定义张量头 A, B，对称性为完全对称的二阶张量
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 创建张量表达式 t = (1 + x
    # 使用类 A 和 B 的实例创建张量 t，通过 split 方法将其拆分后进行张量乘积
    t = A(-b, a)*B(-a, c)*A(-c, d)
    # 对拆分后的张量 t 进行张量乘积，得到结果 t1
    t1 = tensor_mul(*t.split())
    # 断言拆分前后的张量应该相等
    assert t == t1
    # 断言空列表的张量乘积应当返回一个标量张量
    assert tensor_mul(*[]) == TensMul.from_data(S.One, [], [], [])

    # 创建一个空的 TensMul 对象
    t = TensMul.from_data(1, [], [], [])
    # 创建一个名称为 'C' 的张量头
    C = TensorHead('C', [])
    # 断言张量头 C 的字符串表示为 'C'
    assert str(C()) == 'C'
    # 断言空的 TensMul 对象 t 的字符串表示为 '1'
    assert str(t) == '1'
    # 断言空的 TensMul 对象 t 等于标量 1
    assert t == 1
    # 断言创建具有重复指标的张量 A(a, b)*A(a, c) 会引发 ValueError 异常
    raises(ValueError, lambda: A(a, b)*A(a, c))
# 定义一个测试函数，用于测试张量代数中的指标替换功能
def test_substitute_indices():
    # 定义一个 Lorentz 张量指标类型，使用默认的虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义多个 Lorentz 类型的张量指标
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    # 定义两个 Lorentz 类型的张量头
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))

    # 定义一个 Lorentz 类型的张量头 p
    p = TensorHead('p', [Lorentz])
    # 创建一个张量 t，表示 p(i)
    t = p(i)
    # 对 t 进行指标替换 (j, k)，预期结果应为原始张量 t 自身
    t1 = t.substitute_indices((j, k))
    assert t1 == t
    # 对 t 进行指标替换 (i, j)，预期结果应为 p(j)
    t1 = t.substitute_indices((i, j))
    assert t1 == p(j)
    # 对 t 进行指标替换 (i, -j)，预期结果应为 p(-j)
    t1 = t.substitute_indices((i, -j))
    assert t1 == p(-j)
    # 对 t 进行指标替换 (-i, j)，预期结果应为 p(-j)
    t1 = t.substitute_indices((-i, j))
    assert t1 == p(-j)
    # 对 t 进行指标替换 (-i, -j)，预期结果应为 p(j)
    t1 = t.substitute_indices((-i, -j))
    assert t1 == p(j)

    # 创建一个张量 t，表示 A(m, n)*B(-k, -j)
    t = A(m, n)*B(-k, -j)
    # 对 t 进行指标替换 (i, j)，(j, k)，预期结果应为 A(j, l)*B(-l, -k)
    t1 = t.substitute_indices((i, j), (j, k))
    t1a = A(j, l)*B(-l, -k)
    assert t1 == t1a
    # 使用全局函数 substitute_indices 进行相同的指标替换操作
    t1 = substitute_indices(t, (i, j), (j, k))
    assert t1 == t1a

    # 创建一个张量 t，表示 A(i, j) + B(i, j)
    t = A(i, j) + B(i, j)
    # 对 t 进行指标替换 (j, -i)，预期结果应为 A(i, -i) + B(i, -i)
    t1 = t.substitute_indices((j, -i))
    t1a = A(i, -i) + B(i, -i)
    assert t1 == t1a
    # 使用全局函数 substitute_indices 进行相同的指标替换操作
    t1 = substitute_indices(t, (j, -i))
    assert t1 == t1a

# 定义一个测试函数，用于测试 Riemann 张量的循环置换替换功能
def test_riemann_cyclic_replace():
    # 定义一个 Lorentz 张量指标类型，使用默认的虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义多个 Lorentz 类型的张量指标
    m0, m1, m2, m3 = tensor_indices('m:4', Lorentz)
    # 定义一个四阶 Riemann 张量头 R
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    # 创建一个张量 t，表示 R(m0, m2, m1, m3)
    t = R(m0, m2, m1, m3)
    # 对 t 进行 Riemann 循环置换替换，预期结果应为给定的线性组合表达式
    t1 = riemann_cyclic_replace(t)
    t1a = Rational(-1, 3)*R(m0, m3, m2, m1) + Rational(1, 3)*R(m0, m1, m2, m3) + Rational(2, 3)*R(m0, m2, m1, m3)
    assert t1 == t1a

# 定义一个测试函数，用于测试 Riemann 张量的循环性质
def test_riemann_cyclic():
    # 定义一个 Lorentz 张量指标类型，使用默认的虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 定义多个 Lorentz 类型的张量指标
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    # 定义一个四阶 Riemann 张量头 R
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    # 创建一个张量 t，表示 R(i,j,k,l) + R(i,l,j,k) + R(i,k,l,j) - R(i,j,l,k) - R(i,l,k,j) - R(i,k,j,l)
    t = R(i,j,k,l) + R(i,l,j,k) + R(i,k,l,j) - \
        R(i,j,l,k) - R(i,l,k,j) - R(i,k,j,l)
    # 创建一个张量 t2，表示 t*R(-i,-j,-k,-l)
    t2 = t*R(-i,-j,-k,-l)
    # 对 t2 进行 Riemann 循环操作，预期结果应为零张量
    t3 = riemann_cyclic(t2)
    assert t3 == 0
    # 创建一个张量 t，表示 R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    t = R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    # 对 t 进行 Riemann 循环操作，预期结果应为零张量
    t1 = riemann_cyclic(t)
    assert t1 == 0
    # 创建一个张量 t，表示 R(i,j,k,l)
    t = R(i,j,k,l)
    # 对 t 进行 Riemann 循环操作，预期结果应为给定的线性组合表达式
    t1 = riemann_cyclic(t)
    assert t1 == Rational(-1, 3)*R(i, l, j, k) + Rational(1, 3)*R(i, k, j, l) + Rational(2, 3)*R(i, j, k, l)

    # 创建一个张量 t，表示 R(i,j,k,l)*R(-k,-l,m,n)*(R(-m,-n,-i,-j) + 2*R(-m,-j,-n,-i))
    t = R(i,j,k,l)*R(-k,-l,m,n)*(R(-m,-n,-i,-j) + 2*R(-m,-j,-n,-i))
    # 对 t 进行 Riemann 循环操作，预期结果应为零张量
    t1 = riemann_cyclic(t)
    assert t1 == 0

# 定义一个测试函数，用于测试度规收缩功能
def test_contract_metric1():
    # 定义一个符号 D
    D = Symbol('D')
    # 定义一个 Lorentz 张量指标类型，使用维度 D 和默认的虚拟名称 'L'
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    # 定义多个 Lorentz 类型的张量指标
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    # 获取 Lorentz 类型的度规 g
    g = Lorentz.metric
    # 定义一个 Lorentz 类型的张量头 p
    p = TensorHead('p', [Lorentz])
    # 创建一个张量 t，表示 g(a, b)*p(-b)
    t = g(a, b)*p(-b)
    # 对 t 进行度规收缩操作
    # 断言t1等于p(a)，确保计算正确性
    assert t1 == p(a)
    
    # 使用函数tensor_heads创建张量头部A和B，分别是Lorentz张量类型的完全对称张量
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    
    # case with g with all free indices
    # 创建张量t1，包含A(a,b)*B(-b,c)*g(d, e)，使用g对象对其进行度规收缩
    t1 = A(a,b)*B(-b,c)*g(d, e)
    t2 = t1.contract_metric(g)
    # 断言度规收缩后t1等于t2，确保计算正确性
    assert t1 == t2
    
    # case of g(d, -d)
    # 创建张量t1，包含A(a,b)*B(-b,c)*g(-d, d)，使用g对象对其进行度规收缩
    t1 = A(a,b)*B(-b,c)*g(-d, d)
    t2 = t1.contract_metric(g)
    # 断言度规收缩后t2等于D*A(a, d)*B(-d, c)，确保计算正确性
    assert t2 == D*A(a, d)*B(-d, c)
    
    # g with one free index
    # 创建张量t1，包含A(a,b)*B(-b,-c)*g(c, d)，使用g对象对其进行度规收缩
    t1 = A(a,b)*B(-b,-c)*g(c, d)
    t2 = t1.contract_metric(g)
    # 断言度规收缩后t2等于A(a, c)*B(-c, d)，确保计算正确性
    assert t2 == A(a, c)*B(-c, d)
    
    # g with both indices contracted with another tensor
    # 创建张量t1，包含A(a,b)*B(-b,-c)*g(c, -a)，使用g对象对其进行度规收缩
    t1 = A(a,b)*B(-b,-c)*g(c, -a)
    t2 = t1.contract_metric(g)
    # 断言度规收缩后_t2等于A(a, b)*B(-b, -a)，确保计算正确性
    assert _is_equal(t2, A(a, b)*B(-b, -a))
    
    # 创建张量t1，包含A(a,b)*B(-b,-c)*g(c, d)*g(-a, -d)，使用g对象对其进行度规收缩
    t1 = A(a,b)*B(-b,-c)*g(c, d)*g(-a, -d)
    t2 = t1.contract_metric(g)
    # 断言度规收缩后_t2等于A(a,b)*B(-b,-a)，确保计算正确性
    assert _is_equal(t2, A(a,b)*B(-b,-a))
    
    # 创建张量t1，包含A(a,b)*g(-a,-b)，使用g对象对其进行度规收缩
    t1 = A(a,b)*g(-a,-b)
    t2 = t1.contract_metric(g)
    # 断言度规收缩后_t2等于A(a, -a)，确保计算正确性
    assert _is_equal(t2, A(a, -a))
    # 断言t2没有自由指标，确保度规收缩后所有指标均被约化
    assert not t2.free
    
    # 定义Lorentz张量类型，使用dummy_name='L'作为虚拟名称
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    # 创建张量指标a和b，属于Lorentz张量类型
    a, b = tensor_indices('a,b', Lorentz)
    # 定义g为Lorentz度规张量
    g = Lorentz.metric
    
    # 断言度规收缩g(a, -a).contract_metric(g)的结果等于Lorentz的维度，表示没有指标
    assert _is_equal(g(a, -a).contract_metric(g), Lorentz.dim) # no dim
def test_contract_metric2():
    # 定义符号 D
    D = Symbol('D')
    # 定义 Lorentz 张量指标类型，维度为 D，虚拟名称为 'L'
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    # 定义张量指标 a, b, c, d, e, L_0 属于 Lorentz 张量指标类型
    a, b, c, d, e, L_0 = tensor_indices('a,b,c,d,e,L_0', Lorentz)
    # 获取 Lorentz 度规张量 g
    g = Lorentz.metric
    # 定义张量头 p, q 属于 Lorentz 张量类型
    p, q = tensor_heads('p,q', [Lorentz])

    # 计算第一个张量 t1
    t1 = g(a,b)*p(c)*p(-c)
    # 计算第二个张量 t2
    t2 = 3*g(-a,-b)*q(c)*q(-c)
    # 计算张量积 t
    t = t1*t2
    # 应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)
    # 断言结果与预期相等
    assert t == 3*D*p(a)*p(-a)*q(b)*q(-b)

    # 重新计算 t1 和 t2
    t1 = g(a,b)*p(c)*p(-c)
    t2 = 3*q(-a)*q(-b)
    # 计算张量积 t
    t = t1*t2
    # 应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)
    # 对结果进行规范化
    t = t.canon_bp()
    # 断言结果与预期相等
    assert t == 3*p(a)*p(-a)*q(b)*q(-b)

    # 计算 t1 和 t2
    t1 = 2*g(a,b)*p(c)*p(-c)
    t2 = - 3*g(-a,-b)*q(c)*q(-c)
    # 计算张量积 t
    t = t1*t2
    # 应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)
    # 计算新的张量表达式
    t = 6*g(a,b)*g(-a,-b)*p(c)*p(-c)*q(d)*q(-d)
    # 再次应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)

    # 计算 t1 和 t2
    t1 = 2*g(a,b)*p(c)*p(-c)
    t2 = q(-a)*q(-b) + 3*g(-a,-b)*q(c)*q(-c)
    # 计算张量积 t
    t = t1*t2
    # 应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)
    # 断言结果与预期相等
    assert t == (2 + 6*D)*p(a)*p(-a)*q(b)*q(-b)

    # 计算 t1 和 t2
    t1 = p(a)*p(b) + p(a)*q(b) + 2*g(a,b)*p(c)*p(-c)
    t2 = q(-a)*q(-b) - g(-a,-b)*q(c)*q(-c)
    # 计算张量积 t
    t = t1*t2
    # 应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)
    # 计算新的张量表达式 t1
    t1 = (1 - 2*D)*p(a)*p(-a)*q(b)*q(-b) + p(a)*q(-a)*p(b)*q(-b)
    # 断言 t - t1 为规范变换后的零张量
    assert canon_bp(t - t1) == 0

    # 计算 t
    t = g(a,b)*g(c,d)*g(-b,-c)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(g)
    # 断言结果与预期相等
    assert t1 == g(a, d)

    # 计算 t1 和 t2
    t1 = g(a,b)*g(c,d) + g(a,c)*g(b,d) + g(a,d)*g(b,c)
    # 替换指标进行变换 t2
    t2 = t1.substitute_indices((a,-a),(b,-b),(c,-c),(d,-d))
    # 计算张量积 t
    t = t1*t2
    # 应用度规收缩操作到张量 t 上
    t = t.contract_metric(g)
    # 断言结果与预期相等
    assert t.equals(3*D**2 + 6*D)

    # 计算 t
    t = 2*p(a)*g(b,-b)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(g)
    # 断言结果与预期相等
    assert t1.equals(2*D*p(a))

    # 计算 t
    t = 2*p(a)*g(b,-a)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(g)
    # 断言结果与预期相等
    assert t1 == 2*p(b)

    # 定义符号 M
    M = Symbol('M')
    # 计算 t
    t = (p(a)*p(b) + g(a, b)*M**2)*g(-a, -b) - D*M**2
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(g)
    # 断言结果与预期相等
    assert t1 == p(a)*p(-a)

    # 定义张量头 A
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    # 计算 t
    t = A(a, b)*p(L_0)*g(-a, -b)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(g)
    # 断言结果为两种可能的字符串表达
    assert str(t1) == 'A(L_1, -L_1)*p(L_0)' or str(t1) == 'A(-L_1, L_1)*p(L_0)'

def test_metric_contract3():
    # 定义符号 D
    D = Symbol('D')
    # 定义 Spinor 张量指标类型，维度为 D，度规对称性为 -1，虚拟名称为 'S'
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    # 定义张量指标 a0, a1, a2, a3, a4 属于 Spinor 张量指标类型
    a0, a1, a2, a3, a4 = tensor_indices('a0:5', Spinor)
    # 获取 Spinor 度规张量 C
    C = Spinor.metric
    # 定义张量头 chi, psi 属于 Spinor 张量类型
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)
    # 定义张量头 B
    B = TensorHead('B', [Spinor]*2, TensorSymmetry.no_symmetry(2))

    # 计算 t
    t = C(a0,-a0)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(C)
    # 断言结果与预期相等
    assert t1.equals(-D)

    # 计算 t
    t = C(-a0,a0)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(C)
    # 断言结果与预期相等
    assert t1.equals(D)

    # 计算 t
    t = C(a0,a1)*C(-a0,-a1)
    # 应用度规收缩操作到张量 t 上
    t1 = t.contract_metric(C)
    # 断言结果与预期相等
    assert t1.equals(D)

    # 计算 t
    t = C(a1,a0)*C(-a
    # 计算张量乘积 C(a0, -a1) * B(a1, -a0)，并用度规 C 对其进行度规收缩
    t = C(a0,-a1)*B(a1,-a0)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -B(a0, -a0)
    assert _is_equal(t1, -B(a0,-a0))

    # 计算张量乘积 C(-a0, a1) * B(-a1, a0)，并用度规 C 对其进行度规收缩
    t = C(-a0,a1)*B(-a1,a0)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -B(a0, -a0)
    assert _is_equal(t1, -B(a0,-a0))

    # 计算张量乘积 C(-a0, -a1) * B(a1, a0)，并用度规 C 对其进行度规收缩
    t = C(-a0,-a1)*B(a1,a0)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 B(a0, -a0)
    assert _is_equal(t1, B(a0,-a0))

    # 计算张量乘积 C(-a1, a0) * B(a1, -a0)，并用度规 C 对其进行度规收缩
    t = C(-a1, a0)*B(a1,-a0)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 B(a0, -a0)
    assert _is_equal(t1, B(a0,-a0))

    # 计算张量乘积 C(a0, a1) * psi(-a1)，并用度规 C 对其进行度规收缩
    t = C(a0,a1)*psi(-a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 psi(a0)
    assert _is_equal(t1, psi(a0))

    # 计算张量乘积 C(a1, a0) * psi(-a1)，并用度规 C 对其进行度规收缩
    t = C(a1,a0)*psi(-a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -psi(a0)
    assert _is_equal(t1, -psi(a0))

    # 计算张量乘积 C(a0, a1) * chi(-a0) * psi(-a1)，并用度规 C 对其进行度规收缩
    t = C(a0,a1)*chi(-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -chi(a1) * psi(-a1)
    assert _is_equal(t1, -chi(a1)*psi(-a1))

    # 计算张量乘积 C(a1, a0) * chi(-a0) * psi(-a1)，并用度规 C 对其进行度规收缩
    t = C(a1,a0)*chi(-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 chi(a1) * psi(-a1)
    assert _is_equal(t1, chi(a1)*psi(-a1))

    # 计算张量乘积 C(-a1, a0) * chi(-a0) * psi(a1)，并用度规 C 对其进行度规收缩
    t = C(-a1,a0)*chi(-a0)*psi(a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 chi(-a1) * psi(a1)
    assert _is_equal(t1, chi(-a1)*psi(a1))

    # 计算张量乘积 C(a0, -a1) * chi(-a0) * psi(a1)，并用度规 C 对其进行度规收缩
    t = C(a0,-a1)*chi(-a0)*psi(a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -chi(-a1) * psi(a1)
    assert _is_equal(t1, -chi(-a1)*psi(a1))

    # 计算张量乘积 C(-a0, -a1) * chi(a0) * psi(a1)，并用度规 C 对其进行度规收缩
    t = C(-a0,-a1)*chi(a0)*psi(a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 chi(-a1) * psi(a1)
    assert _is_equal(t1, chi(-a1)*psi(a1))

    # 计算张量乘积 C(-a1, -a0) * chi(a0) * psi(a1)，并用度规 C 对其进行度规收缩
    t = C(-a1,-a0)*chi(a0)*psi(a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -chi(-a1) * psi(a1)
    assert _is_equal(t1, -chi(-a1)*psi(a1))

    # 计算张量乘积 C(-a1, -a0) * B(a0, a2) * psi(a1)，并用度规 C 对其进行度规收缩
    t = C(-a1,-a0)*B(a0,a2)*psi(a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 -B(-a1, a2) * psi(a1)
    assert _is_equal(t1, -B(-a1,a2)*psi(a1))

    # 计算张量乘积 C(a1, a0) * B(-a2, -a0) * psi(-a1)，并用度规 C 对其进行度规收缩
    t = C(a1,a0)*B(-a2,-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    # 断言结果是否等于 B(-a2, a1) * psi(-a1)
    assert _is_equal(t1, B(-a2,a1)*psi(-a1))
def test_contract_metric4():
    # 定义一个三维张量指标类型 R3
    R3 = TensorIndexType('R3', dim=3)
    # 创建三个张量指标 p, q, r，并分配给变量 p, q, r
    p, q, r = tensor_indices("p q r", R3)
    # 取出 R3 中的度量 delta，并赋值给变量 delta
    delta = R3.delta
    # 取出 R3 中的 Levi-Civita 符号 epsilon，并赋值给变量 eps
    eps = R3.epsilon
    # 创建一个名为 K 的张量头，具有 R3 作为其指标类型
    K = TensorHead("K", [R3])

    # 检查是否 contract_metric 在一个可展开的表达式上处理规范化后为零的情况（问题编号 #24354）
    # 创建一个表达式，包含 Levi-Civita 符号和两个张量 K 的乘积及 delta 函数
    expr = eps(p,q,r)*( K(-p)*K(-q) + delta(-p,-q) )
    # 断言表达式经过 contract_metric 方法处理后结果为零
    assert expr.contract_metric(delta) == 0


def test_epsilon():
    # 定义一个四维洛伦兹张量指标类型 Lorentz
    Lorentz = TensorIndexType('Lorentz', dim=4, dummy_name='L')
    # 创建五个张量指标 a, b, c, d, e，并分配给变量 a, b, c, d, e
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    # 取出 Lorentz 中的 Levi-Civita 符号 epsilon，并赋值给变量 epsilon
    epsilon = Lorentz.epsilon
    # 创建四个张量头 p, q, r, s，具有 Lorentz 作为其指标类型
    p, q, r, s = tensor_heads('p,q,r,s', [Lorentz])

    # 创建一个 epsilon 函数的应用，对张量指标进行排列，并赋值给变量 t
    t = epsilon(b,a,c,d)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 -epsilon(a,b,c,d)
    assert t1 == -epsilon(a,b,c,d)

    # 创建另一个 epsilon 函数的应用，对张量指标进行排列，并赋值给变量 t
    t = epsilon(c,b,d,a)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 epsilon(a,b,c,d)
    assert t1 == epsilon(a,b,c,d)

    # 创建另一个 epsilon 函数的应用，对张量指标进行排列，并赋值给变量 t
    t = epsilon(c,a,d,b)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 -epsilon(a,b,c,d)
    assert t1 == -epsilon(a,b,c,d)

    # 创建一个 epsilon 函数的应用，对张量指标和两个张量 p, q 进行排列，并赋值给变量 t
    t = epsilon(a,b,c,d)*p(-a)*q(-b)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 epsilon(c,d,a,b)*p(-a)*q(-b)
    assert t1 == epsilon(c,d,a,b)*p(-a)*q(-b)

    # 创建一个 epsilon 函数的应用，对张量指标和两个张量 p, q 进行排列，并赋值给变量 t
    t = epsilon(c,b,d,a)*p(-a)*q(-b)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 epsilon(c,d,a,b)*p(-a)*q(-b)
    assert t1 == epsilon(c,d,a,b)*p(-a)*q(-b)

    # 创建一个 epsilon 函数的应用，对张量指标和两个张量 p, q 进行排列，并赋值给变量 t
    t = epsilon(c,a,d,b)*p(-a)*q(-b)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 -epsilon(c,d,a,b)*p(-a)*q(-b)
    assert t1 == -epsilon(c,d,a,b)*p(-a)*q(-b)

    # 创建一个 epsilon 函数的应用，对张量指标和两个张量 p, q 进行排列，并赋值给变量 t
    t = epsilon(c,a,d,b)*p(-a)*p(-b)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 0
    assert t1 == 0

    # 创建一个 epsilon 函数的应用，对张量指标和两个张量 p, q 进行排列，并赋值给变量 t
    t = epsilon(c,a,d,b)*p(-a)*q(-b) + epsilon(a,b,c,d)*p(-b)*q(-a)
    # 对 t 进行规范化，赋值给变量 t1
    t1 = t.canon_bp()
    # 断言 t1 等于 -2*epsilon(c,d,a,b)*p(-a)*q(-b)
    assert t1 == -2*epsilon(c,d,a,b)*p(-a)*q(-b)

    # 测试 epsilon 能否与 SymPy 整数一起创建：
    # 定义一个四维洛伦兹张量指标类型 Lorentz，维度为整数 4
    Lorentz = TensorIndexType('Lorentz', dim=Integer(4), dummy_name='L')
    # 取出 Lorentz 中的 Levi-Civita 符号 epsilon，并赋值给变量 epsilon
    epsilon = Lorentz.epsilon
    # 断言 epsilon 是 TensorHead 类型的实例
    assert isinstance(epsilon, TensorHead)


def test_contract_delta1():
    # 见 Cvitanovic 的《群论》，第 9 页
    # 定义一个符号 n
    n = Symbol('n')
    # 定义一个颜色张量指标类型 Color，维度为符号 n
    Color = TensorIndexType('Color', dim=n, dummy_name='C')
    # 创建六个张量指标 a, b, c, d, e, f，并分配给变量 a, b, c, d, e, f
    a, b, c, d, e, f = tensor_indices('a,b,c,d,e,f', Color)
    # 取出 Color 中的 delta 函数，并赋值给变量 delta
    delta = Color.delta

    # 定义一个函数 idn，接受四个张量指标作为参数，并返回一个表达式
    def idn(a, b, d, c):
        # 断言 a 和 d 是上指标，b 和 c 不是上指标
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a,c)*delta(d,b)

    # 定义一个函数 T，接受四个张量指标作为参数，并返回一个表达式
    def T(a, b, d, c):
        # 断言 a 和 d 是上指标，b 和 c 不是上指标
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a,b)*delta(d,c)

    # 定义一个函数 P1，接受四个张量指标作为参数，并返回一个表达式
    def P1(a, b, c, d):
        return idn(a,b,c,d) - 1/n*T(a,b,c,d)

    # 定义一个函数 P2，接受四个张量指标作为参数，并返回一个表达式
    def P2(a, b, c, d):
        return 1/n*T(a,b,c,d)
    # 使用 `warns_deprecated_sympy()` 上下文管理器，用于处理 SymPy 废弃警告
    with warns_deprecated_sympy():
        # 定义符号 D 作为一个符号对象
        D = Symbol('D')
        # 创建一个 Lorentz 张量指标类型，维度为 D，指标名为 L
        Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
        # 定义 a, b, c, d, e 作为 Lorentz 类型的张量指标
        a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
        # 获取 Lorentz 度规张量
        g = Lorentz.metric

        # 定义 p, q 作为 Lorentz 类型的张量头
        p, q = tensor_heads('p q', [Lorentz])
        # 定义张量 t，表示 q(c)*p(a)*q(b) + g(a,b)*g(c,d)*q(-d)
        t = q(c)*p(a)*q(b) + g(a,b)*g(c,d)*q(-d)
        # 断言 t(a,b,c) 与 t 相等
        assert t(a,b,c) == t
        # 断言 canon_bp(t - t(b,a,c) - q(c)*p(a)*q(b) + q(c)*p(b)*q(a)) 等于 0
        assert canon_bp(t - t(b,a,c) - q(c)*p(a)*q(b) + q(c)*p(b)*q(a)) == 0
        # 断言 t(b,c,d) 等于 q(d)*p(b)*q(c) + g(b,c)*g(d,e)*q(-e)
        assert t(b,c,d) == q(d)*p(b)*q(c) + g(b,c)*g(d,e)*q(-e)
        
        # 定义 t1 为 t 替换指标 (a,b) -> (b,a) 后的结果
        t1 = t.substitute_indices((a,b),(b,a))
        # 断言 canon_bp(t1 - q(c)*p(b)*q(a) - g(a,b)*g(c,d)*q(-d)) 等于 0
        assert canon_bp(t1 - q(c)*p(b)*q(a) - g(a,b)*g(c,d)*q(-d)) == 0

        # 检查 g_{a b; c} = 0
        # 示例来自 L. Brewin 的文献 "A brief introduction to Cadabra" arxiv:0903.2085
        # dg_{a b c} = \partial_{a} g_{b c} 在 b, c 中是对称的
        dg = TensorHead('dg', [Lorentz]*3, TensorSymmetry.direct_product(1, 2))
        # gamma^a_{b c} 是 Christoffel 符号
        gamma = S.Half*g(a,d)*(dg(-b,-d,-c) + dg(-c,-b,-d) - dg(-d,-b,-c))
        # 定义 t = g_{a b; c}
        t = dg(-c,-a,-b) - g(-a,-d)*gamma(d,-b,-c) - g(-b,-d)*gamma(d,-a,-c)
        # 使用度规 g 来收缩 t
        t = t.contract_metric(g)
        # 断言 t 等于 0
        assert t == 0

        # 定义 t 为 q(c)*p(a)*q(b)
        t = q(c)*p(a)*q(b)
        # 断言 t(b,c,d) 等于 q(d)*p(b)*q(c)
        assert t(b,c,d) == q(d)*p(b)*q(c)
# 定义测试函数 test_TensorManager
def test_TensorManager():
    # 定义张量指标类型 Lorentz 和 LorentzH
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    LorentzH = TensorIndexType('LorentzH', dummy_name='LH')
    # 创建张量指标 i, j 和 ih, jh，分别属于 Lorentz 和 LorentzH 类型
    i, j = tensor_indices('i,j', Lorentz)
    ih, jh = tensor_indices('ih,jh', LorentzH)
    # 创建张量头 p 和 q，属于 Lorentz 类型
    p, q = tensor_heads('p q', [Lorentz])
    # 创建张量头 ph 和 qh，属于 LorentzH 类型
    ph, qh = tensor_heads('ph qh', [LorentzH])

    # 创建符号 Gsymbol 和 GHsymbol
    Gsymbol = Symbol('Gsymbol')
    GHsymbol = Symbol('GHsymbol')
    # 设置 Gsymbol 和 GHsymbol 之间的交换关系为 0
    TensorManager.set_comm(Gsymbol, GHsymbol, 0)
    # 创建张量头 G，属于 Lorentz 类型，使用 Gsymbol 作为符号
    G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), Gsymbol)
    # 断言：TensorManager 中 _comm_i2symbol 字典中 G.comm 对应的符号为 Gsymbol
    assert TensorManager._comm_i2symbol[G.comm] == Gsymbol
    # 创建张量头 GH，属于 LorentzH 类型，使用 GHsymbol 作为符号
    GH = TensorHead('GH', [LorentzH], TensorSymmetry.no_symmetry(1), GHsymbol)
    # 创建张量 ps 和 psh，分别表示 G(i)*p(-i) 和 GH(ih)*ph(-ih)
    ps = G(i)*p(-i)
    psh = GH(ih)*ph(-ih)
    # 创建张量 t，表示 ps + psh
    t = ps + psh
    # 创建张量 t1，表示 t*t
    t1 = t*t
    # 断言：canonical basis expansion 后 t1 - ps*ps - 2*ps*psh - psh*psh 等于 0
    assert canon_bp(t1 - ps*ps - 2*ps*psh - psh*psh) == 0
    # 创建张量 qs 和 qsh，分别表示 G(i)*q(-i) 和 GH(ih)*qh(-ih)
    qs = G(i)*q(-i)
    qsh = GH(ih)*qh(-ih)
    # 断言：ps*qsh 等于 qsh*ps
    assert _is_equal(ps*qsh, qsh*ps)
    # 断言：ps*qs 不等于 qs*ps
    assert not _is_equal(ps*qs, qs*ps)
    # 获取符号 Gsymbol 对应的索引 n
    n = TensorManager.comm_symbols2i(Gsymbol)
    # 断言：通过索引 n 获取的符号应该是 Gsymbol
    assert TensorManager.comm_i2symbol(n) == Gsymbol

    # 断言：GHsymbol 在 TensorManager 的 _comm_symbols2i 中
    assert GHsymbol in TensorManager._comm_symbols2i
    # 断言：调用 set_comm 方法设置 GHsymbol 的交换关系会抛出 ValueError 异常
    raises(ValueError, lambda: TensorManager.set_comm(GHsymbol, 1, 2))
    # 设置多个符号对之间的交换关系
    TensorManager.set_comms((Gsymbol,GHsymbol,0),(Gsymbol,1,1))
    # 断言：获取符号 n 和 1 之间的交换关系应该等于 1
    assert TensorManager.get_comm(n, 1) == TensorManager.get_comm(1, n) == 1
    # 清空 TensorManager 中的所有设置
    TensorManager.clear()
    # 断言：TensorManager 中的交换关系应该为 [{0:0, 1:0, 2:0}, {0:0, 1:1, 2:None}, {0:0, 1:None}]
    assert TensorManager.comm == [{0:0, 1:0, 2:0}, {0:0, 1:1, 2:None}, {0:0, 1:None}]
    # 断言：GHsymbol 不在 TensorManager 的 _comm_symbols2i 中
    assert GHsymbol not in TensorManager._comm_symbols2i
    # 获取 GHsymbol 对应的索引 nh
    nh = TensorManager.comm_symbols2i(GHsymbol)
    # 断言：通过索引 nh 获取的符号应该是 GHsymbol
    assert TensorManager.comm_i2symbol(nh) == GHsymbol
    # 断言：GHsymbol 应该在 TensorManager 的 _comm_symbols2i 中
    assert GHsymbol in TensorManager._comm_symbols2i


# 定义测试函数 test_hash
def test_hash():
    # 创建符号 D
    D = Symbol('D')
    # 创建张量指标类型 Lorentz，其维度为 D，使用 dummy_name 'L'
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    # 创建张量指标 a, b, c, d, e 属于 Lorentz 类型
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    # 获取 Lorentz 的度规 g
    g = Lorentz.metric

    # 创建张量头 p 和 q，属于 Lorentz 类型
    p, q = tensor_heads('p q', [Lorentz])
    # 获取 p 的类型
    p_type = p.args[1]
    # 创建张量 t1，表示 p(a)*q(b)
    t1 = p(a)*q(b)
    # 创建张量 t2，表示 p(a)*p(b)
    t2 = p(a)*p(b)
    # 断言：t1 和 t2 的哈希值不相等
    assert hash(t1) != hash(t2)
    # 创建张量 t3，表示 p(a)*p(b) + g(a,b)
    t3 = p(a)*p(b) + g(a,b)
    # 创建张量 t4，表示 p(a)*p(b) - g(a,b)
    t4 = p(a)*p(b) - g(a,b)
    # 断言：t3 和 t4 的哈希值不相等
    assert hash(t3) != hash(t4)

    # 断言：a 的构造函数返回 a 本身
    assert a.func(*a.args) == a
    # 断言：Lorentz 的构造函数返回 Lorentz 本身
    assert Lorentz.func(*Lorentz.args) == Lorentz
    # 断言：g 的构造函数返回 g 本身
    assert g.func(*g.args) == g
    # 断言：p 的构造函数返回 p 本身
    assert p.func(*p.args) == p
    # 断言：p_type 的构造函数返回 p_type 本身
    assert p_type.func(*p_type.args) == p_type
    # 断言：p(a) 的构造函数返回 p(a) 本身
    assert p(a).func(*(p(a)).args) == p(a)
    # 断言：t1 的构造函数返回 t1 本身
    assert t1.func(*t1.args) == t1
    # 断言：t2 的构造函数返回 t2 本身
    assert t2.func(*t2.args) == t2
    # 断言：t3 的构造函数返回 t3 本身
    assert t3.func(*t3.args) == t3
    # 断言：t4 的构造函数返回 t4 本身
    assert t4.func(*t4.args) == t4

    # 断言：a 的哈希值构造函数返回的结果应该等于 a 的哈希值
    assert hash(a.func(*a.args)) == hash(a)
    # 断言：Lorentz 的哈希值构造函数返回的结果应该等于 Lorentz 的哈希值
    assert hash(Lorentz.func(*Lorentz.args)) == hash(Lorentz)
    # 断言：g 的哈希值构造函数返回的结果应该等于 g 的哈希值
    assert hash(g.func(*g.args)) == hash(g)
    # 断言：p 的哈希值构造函数返回的结果应该等于 p 的哈希值
    assert hash(p.func(*p
    # 断言检查 Lorentz 对象的所有属性
    assert check_all(Lorentz)
    # 断言检查 g 对象的所有属性
    assert check_all(g)
    # 断言检查 p 对象的所有属性
    assert check_all(p)
    # 断言检查 p_type 对象的所有属性
    assert check_all(p_type)
    # 断言检查 p(a) 对象的所有属性
    assert check_all(p(a))
    # 断言检查 t1 对象的所有属性
    assert check_all(t1)
    # 断言检查 t2 对象的所有属性
    assert check_all(t2)
    # 断言检查 t3 对象的所有属性
    assert check_all(t3)
    # 断言检查 t4 对象的所有属性
    assert check_all(t4)

    # 创建一个特定的张量对称性对象，其参数为 (-2, 1, 3)
    tsymmetry = TensorSymmetry.direct_product(-2, 1, 3)

    # 断言验证该张量对称性对象的函数应用后等于对象本身
    assert tsymmetry.func(*tsymmetry.args) == tsymmetry
    # 断言验证该张量对称性对象的哈希值等于对象本身的哈希值
    assert hash(tsymmetry.func(*tsymmetry.args)) == hash(tsymmetry)
    # 断言检查该张量对称性对象的所有属性
    assert check_all(tsymmetry)
### TEST VALUED TENSORS ###

# 定义函数，返回一组已赋值的测试变量
def _get_valued_base_test_variables():
    # 创建一个4x4的Minkowski度规矩阵
    minkowski = Matrix((
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
    ))
    # 创建一个名为Lorentz的张量索引类型，维度为4
    Lorentz = TensorIndexType('Lorentz', dim=4)
    Lorentz.data = minkowski

    # 创建5个张量索引对象
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', Lorentz)

    # 定义符号变量E, px, py, pz
    E, px, py, pz = symbols('E px py pz')

    # 创建一个张量头A，接受Lorentz类型的索引
    A = TensorHead('A', [Lorentz])
    A.data = [E, px, py, pz]

    # 创建一个张量头B，接受Lorentz类型的索引，对称性为Gcomm
    B = TensorHead('B', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
    B.data = range(4)

    # 创建一个张量头AB，接受两个Lorentz类型的索引
    AB = TensorHead("AB", [Lorentz]*2)
    AB.data = minkowski

    # 创建一个4x4的矩阵ba_matrix
    ba_matrix = Matrix((
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 0, -1, -2),
        (-3, -4, -5, -6),
    ))

    # 创建一个张量头BA，接受两个Lorentz类型的索引
    BA = TensorHead("BA", [Lorentz]*2)
    BA.data = ba_matrix

    # 测试对角度规，使用反转的Minkowski度规：
    # 创建一个名为LorentzD的张量索引类型
    LorentzD = TensorIndexType('LorentzD')
    LorentzD.data = [-1, 1, 1, 1]
    # 创建三个张量索引对象mu0, mu1, mu2
    mu0, mu1, mu2 = tensor_indices('mu0:3', LorentzD)

    # 创建一个张量头C，接受LorentzD类型的索引
    C = TensorHead('C', [LorentzD])
    C.data = [E, px, py, pz]

    ### non-diagonal metric ###
    # 创建一个非对角度规矩阵ndm_matrix
    ndm_matrix = (
        (1, 1, 0,),
        (1, 0, 1),
        (0, 1, 0,),
    )
    # 创建一个名为ndm的张量索引类型
    ndm = TensorIndexType("ndm")
    ndm.data = ndm_matrix
    # 创建三个张量索引对象n0, n1, n2
    n0, n1, n2 = tensor_indices('n0:3', ndm)

    # 创建一个张量头NA，接受ndm类型的索引
    NA = TensorHead('NA', [ndm])
    NA.data = range(10, 13)

    # 创建一个张量头NB，接受两个ndm类型的索引
    NB = TensorHead('NB', [ndm]*2)
    NB.data = [[i+j for j in range(10, 13)] for i in range(10, 13)]

    # 创建一个张量头NC，接受三个ndm类型的索引
    NC = TensorHead('NC', [ndm]*3)
    NC.data = [[[i+j+k for k in range(4, 7)] for j in range(1, 4)] for i in range(2, 5)]

    # 返回所有已定义的变量和对象
    return (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
            n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4)


# 测试值张量的迭代功能
def test_valued_tensor_iter():
    # 在给定警告时执行测试
    with warns_deprecated_sympy():
        # 从_get_valued_base_test_variables函数获取所有已定义的变量和对象
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 定义列表list_BA
        list_BA = [Array([1, 2, 3, 4]), Array([5, 6, 7, 8]), Array([9, 0, -1, -2]), Array([-3, -4, -5, -6])]
        
        # 对VTensorHead A 进行迭代
        assert list(A) == [E, px, py, pz]
        # 对矩阵ba_matrix进行迭代
        assert list(ba_matrix) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6]
        # 对VTensorHead BA 进行迭代
        assert list(BA) == list_BA

        # 对VTensMul A(i1) 进行迭代
        assert list(A(i1)) == [E, px, py, pz]
        # 对VTensMul BA(i1, i2) 进行迭代
        assert list(BA(i1, i2)) == list_BA
        # 对VTensMul 3 * BA(i1, i2) 进行迭代
        assert list(3 * BA(i1, i2)) == [3 * i for i in list_BA]
        # 对VTensMul -5 * BA(i1, i2) 进行迭代
        assert list(-5 * BA(i1, i2)) == [-5 * i for i in list_BA]

        # 对VTensAdd A(i1) + A(i1) 进行迭代
        assert list(A(i1) + A(i1)) == [2*E, 2*px, 2*py, 2*pz]
        # 对VTensAdd BA(i1, i2) - BA(i1, i2) 进行迭代
        assert BA(i1, i2) - BA(i1, i2) == 0
        # 对VTensAdd BA(i1, i2) - 2 * BA(i1, i2) 进行迭代
        assert list(BA(i1, i2) - 2 * BA(i1, i2)) == [-i for i in list_BA]


# 测试值张量的协变和逆变元素
def test_valued_tensor_covariant_contravariant_elements():
    # 使用 warns_deprecated_sympy() 上下文管理器捕获 Sympy 废弃警告
    with warns_deprecated_sympy():
        # 调用 _get_valued_base_test_variables() 函数获取多个变量的值
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 断言 A(-i0)[0] 与 A(i0)[0] 相等
        assert A(-i0)[0] == A(i0)[0]
        # 断言 A(-i0)[1] 与 -A(i0)[1] 相等
        assert A(-i0)[1] == -A(i0)[1]

        # 断言 AB(i0, i1) 矩阵中的 [1, 1] 元素等于 -1
        assert AB(i0, i1)[1, 1] == -1
        # 断言 AB(i0, -i1) 矩阵中的 [1, 1] 元素等于 1
        assert AB(i0, -i1)[1, 1] == 1
        # 断言 AB(-i0, -i1) 矩阵中的 [1, 1] 元素等于 -1
        assert AB(-i0, -i1)[1, 1] == -1
        # 断言 AB(-i0, i1) 矩阵中的 [1, 1] 元素等于 1
        assert AB(-i0, i1)[1, 1] == 1
def test_valued_tensor_get_matrix():
    # 使用 warns_deprecated_sympy 上下文管理器，捕获关于 SymPy 的过时警告
    with warns_deprecated_sympy():
        # 调用 _get_valued_base_test_variables() 函数，获取测试变量元组
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 调用 AB(i0, i1) 对象的 get_matrix() 方法，返回矩阵 matab
        matab = AB(i0, i1).get_matrix()
        # 断言 matab 等于给定的 Matrix 对象
        assert matab == Matrix([
                                [1,  0,  0,  0],
                                [0, -1,  0,  0],
                                [0,  0, -1,  0],
                                [0,  0,  0, -1],
                                ])

        # 断言当使用 [1, -1, -1, -1] 指标度规时，AB(i0, -i1) 的矩阵等于单位矩阵
        assert AB(i0, -i1).get_matrix() == eye(4)

        # 断言协变和逆变形式的矩阵
        assert A(i0).get_matrix() == Matrix([E, px, py, pz])
        assert A(-i0).get_matrix() == Matrix([E, -px, -py, -pz])

def test_valued_tensor_contraction():
    # 使用 warns_deprecated_sympy 上下文管理器，捕获关于 SymPy 的过时警告
    with warns_deprecated_sympy():
        # 调用 _get_valued_base_test_variables() 函数，获取测试变量元组
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 断言 A(i0) * A(-i0) 的数据等于 E ** 2 - px ** 2 - py ** 2 - pz ** 2
        assert (A(i0) * A(-i0)).data == E ** 2 - px ** 2 - py ** 2 - pz ** 2
        # 断言 A(i0) * A(-i0) 的数据等于 A ** 2
        assert (A(i0) * A(-i0)).data == A ** 2
        # 断言 A(i0) * A(-i0) 的数据等于 A(i0) ** 2
        assert (A(i0) * A(-i0)).data == A(i0) ** 2
        # 断言 A(i0) * B(-i0) 的数据等于 -px - 2 * py - 3 * pz
        assert (A(i0) * B(-i0)).data == -px - 2 * py - 3 * pz

        # 嵌套循环遍历，断言 A(i0) * B(-i1) 的每个元素等于给定的乘积
        for i in range(4):
            for j in range(4):
                assert (A(i0) * B(-i1))[i, j] == [E, px, py, pz][i] * [0, -1, -2, -3][j]

        # 使用备选的 Minkowski 度规 [-1, 1, 1, 1] 测试收缩
        assert (C(mu0) * C(-mu0)).data == -E ** 2 + px ** 2 + py ** 2 + pz ** 2

        # 对于 A(i0) * AB(i1, -i0)，进行收缩并断言其秩为 1
        contrexp = A(i0) * AB(i1, -i0)
        assert A(i0).rank == 1
        assert AB(i1, -i0).rank == 2
        assert contrexp.rank == 1
        # 嵌套循环遍历，断言 contrexp 的每个元素等于给定的值
        for i in range(4):
            assert contrexp[i] == [E, px, py, pz][i]

def test_valued_tensor_self_contraction():
    # 使用 warns_deprecated_sympy 上下文管理器，捕获关于 SymPy 的过时警告
    with warns_deprecated_sympy():
        # 调用 _get_valued_base_test_variables() 函数，获取测试变量元组
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 断言 AB(i0, -i0) 的数据等于 4
        assert AB(i0, -i0).data == 4
        # 断言 BA(i0, -i0) 的数据等于 2
        assert BA(i0, -i0).data == 2

def test_valued_tensor_pow():
    # 使用 warns_deprecated_sympy 上下文管理器，捕获关于 SymPy 的过时警告
    with warns_deprecated_sympy():
        # 调用 _get_valued_base_test_variables() 函数，获取测试变量元组
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 断言 C**2 的值等于 -E**2 + px**2 + py**2 + pz**2
        assert C**2 == -E**2 + px**2 + py**2 + pz**2
        # 断言 C**1 的值等于 sqrt(-E**2 + px**2 + py**2 + pz**2)
        assert C**1 == sqrt(-E**2 + px**2 + py**2 + pz**2)
        # 断言 C(mu0)**2 的值等于 C**2
        assert C(mu0)**2 == C**2
        # 断言 C(mu0)**1 的值等于 C**1
        assert C(mu0)**1 == C**1
    # 使用 `warns_deprecated_sympy()` 上下文管理器来捕获 SymPy 废弃警告
    with warns_deprecated_sympy():
        # 从测试变量中获取一系列符号并赋值给对应的变量
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 创建符号变量 x1, x2, x3
        x1, x2, x3 = symbols('x1:4')

        # 测试收缩中的系数：
        # 创建二阶张量的系数 rank2coeff
        rank2coeff = x1 * A(i3) * B(i2)
        # 断言检查张量的特定元素值
        assert rank2coeff[1, 1] == x1 * px
        assert rank2coeff[3, 3] == 3 * pz * x1

        # 计算系数表达式 coeff_expr
        coeff_expr = ((x1 * A(i4)) * (B(-i4) / x2)).data
        # 断言检查展开后的表达式值
        assert coeff_expr.expand() == -px*x1/x2 - 2*py*x1/x2 - 3*pz*x1/x2

        # 创建加法表达式 add_expr
        add_expr = A(i0) + B(i0)
        # 断言检查加法表达式的各个元素值
        assert add_expr[0] == E
        assert add_expr[1] == px + 1
        assert add_expr[2] == py + 2
        assert add_expr[3] == pz + 3

        # 创建减法表达式 sub_expr
        sub_expr = A(i0) - B(i0)
        # 断言检查减法表达式的各个元素值
        assert sub_expr[0] == E
        assert sub_expr[1] == px - 1
        assert sub_expr[2] == py - 2
        assert sub_expr[3] == pz - 3

        # 检查乘积表达式的数据
        assert (add_expr * B(-i0)).data == -px - 2*py - 3*pz - 14

        # 创建复合表达式 expr1 到 expr5
        expr1 = x1 * A(i0) + x2 * B(i0)
        expr2 = expr1 * B(i1) * (-4)
        expr3 = expr2 + 3 * x3 * AB(i0, i1)
        expr4 = expr3 / 2
        # 断言检查乘法和除法的正确性
        assert expr4 * 2 == expr3
        expr5 = (expr4 * BA(-i1, -i0))

        # 断言检查展开后的表达式值
        assert expr5.data.expand() == 28 * E * x1 + 12 * px * x1 + 20 * py * x1 + 28 * pz * x1 + 136 * x2 + 3 * x3
def test_valued_tensor_add_scalar():
    with warns_deprecated_sympy():
        # 从辅助函数获取测试变量
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 在缩并张量后添加一个标量加数
        expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)
        assert expr1.data == 0

        # 在缩并张量前面添加多个标量加数
        expr2 = E**2 - px**2 - py**2 - pz**2 - A(i0)*A(-i0)
        assert expr2.data == 0

        # 在缩并张量后面添加多个标量加数
        expr3 =  A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2
        assert expr3.data == 0

        # 添加多个标量加数和多个张量
        expr4 = C(mu0)*C(-mu0) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)
        assert expr4.data == 0


def test_noncommuting_components():
    with warns_deprecated_sympy():
        # 从辅助函数获取测试变量
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 创建欧几里德张量类型
        euclid = TensorIndexType('Euclidean')
        euclid.data = [1, 1]
        i1, i2, i3 = tensor_indices('i1:4', euclid)

        # 创建非交换符号的符号
        a, b, c, d = symbols('a b c d', commutative=False)
        V1 = TensorHead('V1', [euclid]*2)
        V1.data = [[a, b], (c, d)]
        V2 = TensorHead('V2', [euclid]*2)
        V2.data = [[a, c], [b, d]]

        # 计算张量积
        vtp = V1(i1, i2) * V2(-i2, -i1)

        assert vtp.data == a**2 + b**2 + c**2 + d**2
        assert vtp.data != a**2 + 2*b*c + d**2

        vtp2 = V1(i1, i2)*V1(-i2, -i1)

        assert vtp2.data == a**2 + b*c + c*b + d**2
        assert vtp2.data != a**2 + 2*b*c + d**2

        # 计算张量与标量的乘积
        Vc = (b * V1(i1, -i1)).data
        assert Vc.expand() == b * a + b * d


def test_valued_non_diagonal_metric():
    with warns_deprecated_sympy():
        # 从辅助函数获取测试变量
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 将数值赋给矩阵
        mmatrix = Matrix(ndm_matrix)
        assert (NA(n0)*NA(-n0)).data == (NA(n0).get_matrix().T * mmatrix * NA(n0).get_matrix())[0, 0]


def test_valued_assign_numpy_ndarray():
    with warns_deprecated_sympy():
        # 用于在使用过程中警告关于 SymPy 库的过时功能
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 将列表 arr 赋值给 A.data，以确保 numpy.ndarray 可以赋值给 tensor
        arr = [E+1, px-1, py, pz]
        A.data = Array(arr)
        # 验证 A(i0).data[i] 是否等于 arr[i]
        for i in range(4):
                assert A(i0).data[i] == arr[i]

        # 使用 symbols 函数创建 qx, qy, qz 符号对象
        qx, qy, qz = symbols('qx qy qz')
        # 将数组赋值给 A(-i0).data
        A(-i0).data = Array([E, qx, qy, qz])
        # 验证 A(i0).data[i] 是否等于 [E, -qx, -qy, -qz][i]
        for i in range(4):
            assert A(i0).data[i] == [E, -qx, -qy, -qz][i]
            # 验证 A.data[i] 是否等于 [E, -qx, -qy, -qz][i]
            assert A.data[i] == [E, -qx, -qy, -qz][i]

        # 对多重索引张量进行测试
        random_4x4_data = [[(i**3-3*i**2)%(j+7) for i in range(4)] for j in range(4)]
        # 将 random_4x4_data 赋值给 AB(-i0, -i1).data
        AB(-i0, -i1).data = random_4x4_data

        # 验证多重索引条件下的数值
        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)*(-1 if j else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if j else 1)
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]

        # 将 random_4x4_data 赋值给 AB(-i0, i1).data
        AB(-i0, i1).data = random_4x4_data
        # 验证多重索引条件下的数值
        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)*(-1 if j else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if j else 1)
def test_valued_metric_inverse():
    with warns_deprecated_sympy():
        # 获取一组基础测试变量
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 创建一个对称矩阵用于测试，没有物理意义，仅用于测试 sympy；
        md = [[2, 2, 2, 1], [2, 3, 1, 0], [2, 1, 2, 3], [1, 0, 3, 2]]
        # 将矩阵赋值给 Lorentz 对象的 data 属性
        Lorentz.data = md
        m = Matrix(md)
        # 获取 Lorentz 对象的度量张量
        metric = Lorentz.metric
        # 计算矩阵 m 的逆矩阵
        minv = m.inv()

        # 创建单位矩阵
        meye = eye(4)

        # 获取 Lorentz 对象的 Kronecker Delta
        KD = Lorentz.get_kronecker_delta()

        # 进行多重断言，验证度量张量的性质
        for i in range(4):
            for j in range(4):
                assert metric(i0, i1).data[i, j] == m[i, j]
                assert metric(-i0, -i1).data[i, j] == minv[i, j]
                assert metric(i0, -i1).data[i, j] == meye[i, j]
                assert metric(-i0, i1).data[i, j] == meye[i, j]
                assert metric(i0, i1)[i, j] == m[i, j]
                assert metric(-i0, -i1)[i, j] == minv[i, j]
                assert metric(i0, -i1)[i, j] == meye[i, j]
                assert metric(-i0, i1)[i, j] == meye[i, j]

                # 验证 Kronecker Delta 的性质
                assert KD(i0, -i1)[i, j] == meye[i, j]


def test_valued_canon_bp_swapaxes():
    with warns_deprecated_sympy():
        # 获取一组基础测试变量
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # 创建张量 e1 和其它对象进行对称性检验
        e1 = A(i1)*A(i0)
        # 对 e1 进行 BP 规范化处理
        e2 = e1.canon_bp()
        # 断言经过规范化处理的 e2 应与原始 e1 等价
        assert e2 == A(i0)*A(i1)
        # 验证交换轴后 e1 和 e2 的元素相等
        for i in range(4):
            for j in range(4):
                assert e1[i, j] == e2[j, i]

        # 创建张量 o1 和其它对象进行对称性检验
        o1 = B(i2)*A(i1)*B(i0)
        # 对 o1 进行 BP 规范化处理
        o2 = o1.canon_bp()
        # 验证 o1 和 o2 的元素在交换轴后相等
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    assert o1[i, j, k] == o2[j, i, k]


def test_valued_components_with_wrong_symmetry():
    # 待实现，未提供具体代码
    pass
    # 在引发过时警告的上下文中执行以下操作
    with warns_deprecated_sympy():
        # 创建一个张量索引类型 'IT'，维度为 3
        IT = TensorIndexType('IT', dim=3)
        # 创建四个张量索引 i0, i1, i2, i3，它们都属于 IT 类型
        i0, i1, i2, i3 = tensor_indices('i0:4', IT)
        # 设置 IT 对象的数据属性为 [1, 1, 1]
        IT.data = [1, 1, 1]
        # 创建一个非对称张量头 'A'，包含两个 IT 类型的索引
        A_nosym = TensorHead('A', [IT]*2)
        # 创建一个完全对称张量头 'A'，包含两个 IT 类型的索引，并设定为完全对称的
        A_sym = TensorHead('A', [IT]*2, TensorSymmetry.fully_symmetric(2))
        # 创建一个完全反对称张量头 'A'，包含两个 IT 类型的索引，并设定为完全反对称的
        A_antisym = TensorHead('A', [IT]*2, TensorSymmetry.fully_symmetric(-2))

        # 创建一个非对称的 3x3 矩阵
        mat_nosym = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        # 创建一个对称的 3x3 矩阵，通过将矩阵与其转置相加得到
        mat_sym = mat_nosym + mat_nosym.T
        # 创建一个反对称的 3x3 矩阵，通过将矩阵与其转置相减得到
        mat_antisym = mat_nosym - mat_nosym.T

        # 将非对称张量头 A_nosym 的数据属性设置为 mat_nosym
        A_nosym.data = mat_nosym
        # 将非对称张量头 A_nosym 的数据属性设置为 mat_sym
        A_nosym.data = mat_sym
        # 将非对称张量头 A_nosym 的数据属性设置为 mat_antisym
        A_nosym.data = mat_antisym

        # 定义一个函数 assign，用于设置张量头 A 的数据属性为 dat
        def assign(A, dat):
            A.data = dat

        # 将对称张量头 A_sym 的数据属性设置为 mat_sym
        A_sym.data = mat_sym
        # 引发 ValueError 异常，lambda 函数调用 assign(A_sym, mat_nosym)
        raises(ValueError, lambda: assign(A_sym, mat_nosym))
        # 引发 ValueError 异常，lambda 函数调用 assign(A_sym, mat_antisym)
        raises(ValueError, lambda: assign(A_sym, mat_antisym))

        # 将反对称张量头 A_antisym 的数据属性设置为 mat_antisym
        A_antisym.data = mat_antisym
        # 引发 ValueError 异常，lambda 函数调用 assign(A_antisym, mat_sym)
        raises(ValueError, lambda: assign(A_antisym, mat_sym))
        # 引发 ValueError 异常，lambda 函数调用 assign(A_antisym, mat_nosym)
        raises(ValueError, lambda: assign(A_antisym, mat_nosym))

        # 将对称张量头 A_sym 的数据属性设置为全零的 3x3 列表
        A_sym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # 将反对称张量头 A_antisym 的数据属性设置为全零的 3x3 列表
        A_antisym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
# 定义一个测试函数，用于测试问题10972中的张量乘积
def test_issue_10972_TensMul_data():
    # 在使用即将弃用的Sympy警告环境中执行以下代码
    with warns_deprecated_sympy():
        # 定义洛伦兹张量类型，指定度规对称性为1，虚拟名称为'i'，维度为2
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        # 指定洛伦兹张量类型的数据为[-1, 1]
        Lorentz.data = [-1, 1]

        # 定义四个张量指标mu, nu, alpha, beta，并与洛伦兹张量类型相关联
        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta', Lorentz)

        # 定义名为'u'的张量头，参数为洛伦兹张量类型列表
        u = TensorHead('u', [Lorentz])
        # 指定'u'张量头的数据为[1, 0]
        u.data = [1, 0]

        # 定义名为'F'的张量头，参数为两个洛伦兹张量类型的列表，并指定其完全对称性为-2
        F = TensorHead('F', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        # 指定'F'张量的数据为二维数组，内容为[[0, 1], [-1, 0]]
        F.data = [[0, 1],
                  [-1, 0]]

        # 计算mul_1，等于F(mu, alpha) * u(-alpha) * F(nu, beta) * u(-beta)
        mul_1 = F(mu, alpha) * u(-alpha) * F(nu, beta) * u(-beta)
        # 断言mul_1的数据等于[[0, 0], [0, 1]]
        assert (mul_1.data == Array([[0, 0], [0, 1]]))

        # 计算mul_2，等于F(mu, alpha) * F(nu, beta) * u(-alpha) * u(-beta)
        mul_2 = F(mu, alpha) * F(nu, beta) * u(-alpha) * u(-beta)
        # 断言mul_2的数据等于mul_1的数据
        assert (mul_2.data == mul_1.data)

        # 断言(mul_1 + mul_1)的数据等于2乘以mul_1的数据
        assert ((mul_1 + mul_1).data == 2 * mul_1.data)


# 定义另一个测试函数，用于测试张量乘积数据的操作
def test_TensMul_data():
    # 在使用即将弃用的Sympy警告环境中执行以下代码
    with warns_deprecated_sympy():
        # 定义洛伦兹张量类型，指定度规对称性为1，虚拟名称为'L'，维度为4
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='L', dim=4)
        # 指定洛伦兹张量类型的数据为[-1, 1, 1, 1]
        Lorentz.data = [-1, 1, 1, 1]

        # 定义四个张量指标mu, nu, alpha, beta，并与洛伦兹张量类型相关联
        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta', Lorentz)

        # 定义名为'u'的张量头，参数为洛伦兹张量类型列表
        u = TensorHead('u', [Lorentz])
        # 指定'u'张量头的数据为[1, 0, 0, 0]
        u.data = [1, 0, 0, 0]

        # 定义名为'F'的张量头，参数为两个洛伦兹张量类型的列表，并指定其完全对称性为-2
        F = TensorHead('F', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        # 定义符号变量Ex, Ey, Ez, Bx, By, Bz
        Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
        # 指定'F'张量的数据为一个二维数组
        F.data = [
            [0, Ex, Ey, Ez],
            [-Ex, 0, Bz, -By],
            [-Ey, -Bz, 0, Bx],
            [-Ez, By, -Bx, 0]]

        # 计算E，等于F(mu, nu) * u(-nu)
        E = F(mu, nu) * u(-nu)

        # 断言(E(mu) * E(nu))的数据等于指定的数组
        assert ((E(mu) * E(nu)).data ==
                Array([[0, 0, 0, 0],
                      [0, Ex ** 2, Ex * Ey, Ex * Ez],
                      [0, Ex * Ey, Ey ** 2, Ey * Ez],
                      [0, Ex * Ez, Ey * Ez, Ez ** 2]])
                )

        # 断言(E(mu) * E(nu))的规范化基础表达式数据等于其原始数据
        assert ((E(mu) * E(nu)).canon_bp().data == (E(mu) * E(nu)).data)

        # 断言(F(mu, alpha) * F(beta, nu) * u(-alpha) * u(-beta))的数据等于-(E(mu) * E(nu))的数据
        assert ((F(mu, alpha) * F(beta, nu) * u(-alpha) * u(-beta)).data ==
                - (E(mu) * E(nu)).data
                )
        
        # 断言(F(alpha, mu) * F(beta, nu) * u(-alpha) * u(-beta))的数据等于(E(mu) * E(nu))的数据
        assert ((F(alpha, mu) * F(beta, nu) * u(-alpha) * u(-beta)).data ==
                (E(mu) * E(nu)).data
                )

        # 定义名为'g'的张量头，参数为两个洛伦兹张量类型的列表，并指定其完全对称性为2
        g = TensorHead('g', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        # 指定'g'张量的数据为洛伦兹张量类型的数据
        g.data = Lorentz.data

        # 定义张量'perp'，表示与向量'u'正交的张量
        perp = u(mu) * u(nu) + g(mu, nu)

        # 计算mul_1，等于u(-mu) * perp(mu, nu)
        mul_1 = u(-mu) * perp(mu, nu)
        # 断言mul_1的数据为全零数组
        assert (mul_1.data == Array([0, 0, 0, 0]))

        # 计算mul_2，等于u(-mu) * perp(mu, alpha) * perp(nu, beta)
        mul_2 = u(-mu) * perp(mu, alpha) * perp(nu, beta)
        # 断言mul_2的数据为全零4x4x4数组
        assert (mul_2.data == Array.zeros(4, 4, 4))

        # 计算Fperp，等于perp(mu, alpha) * perp(nu, beta) * F(-alpha, -beta)
        Fperp = perp(mu, alpha) * perp(nu, beta) * F(-alpha, -beta)
        # 断言Fperp的第一行数据为全零数组
        assert (Fperp.data[0, :] == Array([0, 0, 0, 0]))
        # 断言Fperp的第一列数据为全零数组
        assert (Fperp.data[:, 0] == Array([0, 0, 0, 0]))

        # 计算mul_3，等于u(-mu) * Fperp(mu, nu)
        mul_3 = u(-mu) * Fperp(mu, nu)
        # 断
    # 使用 warns_deprecated_sympy 上下文管理器，捕获 sympy 中的弃用警告
    with warns_deprecated_sympy():
        # 创建一个 Lorentz 张量指标类型，指定度规对称性为 1，虚拟名称为 'i'，维度为 2
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        # 设置 Lorentz 张量指标类型的数据为 [-1, 1]
        Lorentz.data = [-1, 1]

        # 创建张量指标 a, b, c, d，它们属于 Lorentz 张量指标类型
        a, b, c, d = tensor_indices('a, b, c, d', Lorentz)
        # 创建张量指标 i0, i1，它们属于 Lorentz 张量指标类型，并命名为 i_0, i_1
        i0, i1 = tensor_indices('i_0:2', Lorentz)

        # 创建度规张量 g，它有两个 Lorentz 张量指标，且完全对称
        g = TensorHead('g', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        # 设置度规张量 g 的数据为 Lorentz 张量类型的数据
        g.data = Lorentz.data

        # 创建张量头 u，它有一个 Lorentz 张量指标
        u = TensorHead('u', [Lorentz])
        # 设置张量头 u 的数据为 [1, 0]
        u.data = [1, 0]

        # 定义 add_1 张量表达式
        add_1 = g(b, c) * g(d, i0) * u(-i0) - g(b, c) * u(d)
        # 断言 add_1 的数据应为一个形状为 (2, 2, 2) 的全零数组
        assert (add_1.data == Array.zeros(2, 2, 2))
        
        # 替换索引 `d` 为 `a` 后的张量表达式 add_2
        add_2 = g(b, c) * g(a, i0) * u(-i0) - g(b, c) * u(a)
        # 断言 add_2 的数据应为一个形状为 (2, 2, 2) 的全零数组
        assert (add_2.data == Array.zeros(2, 2, 2))

        # 定义 perp 张量表达式，它是与 u^mu 正交的张量
        perp = u(a) * u(b) + g(a, b)
        # 定义 mul_1 张量表达式
        mul_1 = u(-a) * perp(a, b)
        # 断言 mul_1 的数据应为一个形状为 [0, 0] 的数组
        assert (mul_1.data == Array([0, 0]))

        # 定义 mul_2 张量表达式
        mul_2 = u(-c) * perp(c, a) * perp(d, b)
        # 断言 mul_2 的数据应为一个形状为 (2, 2, 2) 的全零数组
        assert (mul_2.data == Array.zeros(2, 2, 2))
def test_index_iteration():
    # 创建一个张量索引类型对象，名称为"Lorentz"，使用默认的虚拟名称"L"
    L = TensorIndexType("Lorentz", dummy_name="L")
    # 创建五个张量索引对象，命名为i0到i4，使用上面定义的张量索引类型L
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', L)
    # 创建一个张量索引对象，命名为L_0，使用上面定义的张量索引类型L
    L0 = tensor_indices('L_0', L)
    # 创建一个张量索引对象，命名为L_1，使用上面定义的张量索引类型L
    L1 = tensor_indices('L_1', L)
    # 创建一个张量头对象A，指定索引为[L, L]
    A = TensorHead("A", [L, L])
    # 创建一个张量头对象B，指定索引为[L, L]，使用完全对称的张量对称性（fully_symmetric）
    B = TensorHead("B", [L, L], TensorSymmetry.fully_symmetric(2))

    # 创建张量表达式e1，表示A(i0, i2)
    e1 = A(i0, i2)
    # 创建张量表达式e2，表示A(i0, -i0)
    e2 = A(i0, -i0)
    # 创建张量表达式e3，表示A(i0, i1)*B(i2, i3)
    e3 = A(i0, i1) * B(i2, i3)
    # 创建张量表达式e4，表示A(i0, i1)*B(i2, -i1)
    e4 = A(i0, i1) * B(i2, -i1)
    # 创建张量表达式e5，表示A(i0, i1)*B(-i0, -i1)
    e5 = A(i0, i1) * B(-i0, -i1)
    # 创建张量表达式e6，表示e1 + e4
    e6 = e1 + e4

    # 断言e1中的自由索引的迭代结果
    assert list(e1._iterate_free_indices) == [(i0, (1, 0)), (i2, (1, 1))]
    # 断言e1中的虚拟索引的迭代结果
    assert list(e1._iterate_dummy_indices) == []
    # 断言e1中所有索引的迭代结果
    assert list(e1._iterate_indices) == [(i0, (1, 0)), (i2, (1, 1))]

    # 断言e2中的自由索引的迭代结果
    assert list(e2._iterate_free_indices) == []
    # 断言e2中的虚拟索引的迭代结果
    assert list(e2._iterate_dummy_indices) == [(L0, (1, 0)), (-L0, (1, 1))]
    # 断言e2中所有索引的迭代结果
    assert list(e2._iterate_indices) == [(L0, (1, 0)), (-L0, (1, 1))]

    # 断言e3中的自由索引的迭代结果
    assert list(e3._iterate_free_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]
    # 断言e3中的虚拟索引的迭代结果
    assert list(e3._iterate_dummy_indices) == []
    # 断言e3中所有索引的迭代结果
    assert list(e3._iterate_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]

    # 断言e4中的自由索引的迭代结果
    assert list(e4._iterate_free_indices) == [(i0, (0, 1, 0)), (i2, (1, 1, 0))]
    # 断言e4中的虚拟索引的迭代结果
    assert list(e4._iterate_dummy_indices) == [(L0, (0, 1, 1)), (-L0, (1, 1, 1))]
    # 断言e4中所有索引的迭代结果
    assert list(e4._iterate_indices) == [(i0, (0, 1, 0)), (L0, (0, 1, 1)), (i2, (1, 1, 0)), (-L0, (1, 1, 1))]

    # 断言e5中的自由索引的迭代结果
    assert list(e5._iterate_free_indices) == []
    # 断言e5中的虚拟索引的迭代结果
    assert list(e5._iterate_dummy_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]
    # 断言e5中所有索引的迭代结果
    assert list(e5._iterate_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]

    # 断言e6中的自由索引的迭代结果
    assert list(e6._iterate_free_indices) == [(i0, (0, 0, 1, 0)), (i2, (0, 1, 1, 0)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]
    # 断言e6中的虚拟索引的迭代结果
    assert list(e6._iterate_dummy_indices) == [(L0, (0, 0, 1, 1)), (-L0, (0, 1, 1, 1))]
    # 断言e6中所有索引的迭代结果
    assert list(e6._iterate_indices) == [(i0, (0, 0, 1, 0)), (L0, (0, 0, 1, 1)), (i2, (0, 1, 1, 0)), (-L0, (0, 1, 1, 1)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]

    # 断言e1的索引获取方法的结果
    assert e1.get_indices() == [i0, i2]
    # 断言e1的自由索引获取方法的结果
    assert e1.get_free_indices() == [i0, i2]
    # 断言e2的索引获取方法的结果
    assert e2.get_indices() == [L0, -L0]
    # 断言e2的自由索引获取方法的结果
    assert e2.get_free_indices() == []
    # 断言e3的索引获取方法的结果
    assert e3.get_indices() == [i0, i1, i2, i3]
    # 断言e3的自由索引获取方法的结果
    assert e3.get_free_indices() == [i0, i1, i2, i3]
    # 断言e4的索引获取方法的结果
    assert e4.get_indices() == [i0, L0, i2, -L0]
    # 断言e4的自由索引获取方法的结果
    assert e4.get_free_indices() == [i0, i2]
    # 断言e5的索引获取方法的结果
    assert e5.get_indices() == [L0, L1, -L0, -L1]
    # 断言e5的自由索引获取方法的结果
    assert e5.get_free_indices() == []


def test_tensor_expand():
    # 创建一个张量索引类型对象，名称为"L"
    L = TensorIndexType("L")

    # 创建三个张量索引对象，命名为i, j, k，使用上面定义的张量索引类型L
    i, j, k = tensor_indices("i j k", L)
    # 创建一个张量索
    # 创建表达式 expr，表示为 A(i)*A(j) + A(i)*B(j)
    expr = A(i)*A(j) + A(i)*B(j)
    # 断言表达式的字符串表示是否为指定的值
    assert str(expr) == "A(i)*A(j) + A(i)*B(j)"

    # 创建包含更复杂结构的表达式 expr
    expr = A(-i)*(A(i)*A(j) + A(i)*B(j)*C(k)*C(-k))
    # 断言表达式不等于展开后的部分表达式
    assert expr != A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)*C(k)*C(-k)
    # 断言展开后的表达式与预期相等
    assert expr.expand() == A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)*C(k)*C(-k)
    # 断言表达式的字符串表示是否为指定的值
    assert str(expr) == "A(-L_0)*(A(L_0)*A(j) + A(L_0)*B(j)*C(L_1)*C(-L_1))"
    # 断言表达式的最简形式的字符串表示是否为指定的值
    assert str(expr.canon_bp()) == 'A(j)*A(L_0)*A(-L_0) + A(L_0)*A(-L_0)*B(j)*C(L_1)*C(-L_1)'

    # 创建包含系数的表达式 expr
    expr = A(-i)*(2*A(i)*A(j) + A(i)*B(j))
    # 断言展开后的表达式是否与预期相等
    assert expr.expand() == 2*A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)

    # 创建包含常数系数的表达式 expr
    expr = 2*A(i)*A(-i)
    # 断言表达式的系数是否为指定的常数
    assert expr.coeff == 2

    # 创建包含多个项的表达式 expr
    expr = A(i)*(B(j)*C(k) + C(j)*(A(k) + D(k)))
    # 断言表达式的字符串表示是否为指定的值
    assert str(expr) == "A(i)*(B(j)*C(k) + C(j)*(A(k) + D(k)))"
    # 断言展开后的表达式的字符串表示是否为指定的值
    assert str(expr.expand()) == "A(i)*B(j)*C(k) + A(i)*C(j)*A(k) + A(i)*C(j)*D(k)"

    # 断言 TensMul(3) 的类型是否为 TensMul
    assert isinstance(TensMul(3), TensMul)
    # 对 TensMul(3) 执行 doit() 方法
    tm = TensMul(3).doit()
    # 断言结果是否等于预期的值
    assert tm == 3
    # 断言结果的类型是否为 Integer
    assert isinstance(tm, Integer)

    # 创建多个中间变量和表达式，并对它们进行展开和断言
    p1 = B(j)*B(-j) + B(j)*C(-j)
    p2 = C(-i)*p1
    p3 = A(i)*p2
    assert p3.expand() == A(i)*C(-i)*B(j)*B(-j) + A(i)*C(-i)*B(j)*C(-j)

    # 创建复杂的表达式 expr，并对其进行展开和断言
    expr = A(i)*(B(-i) + C(-i)*(B(j)*B(-j) + B(j)*C(-j)))
    assert expr.expand() == A(i)*B(-i) + A(i)*C(-i)*B(j)*B(-j) + A(i)*C(-i)*B(j)*C(-j)

    # 创建表达式 expr，并对其进行展开和断言
    expr = C(-i)*(B(j)*B(-j) + B(j)*C(-j))
    assert expr.expand() == C(-i)*B(j)*B(-j) + C(-i)*B(j)*C(-j)
# 定义一个函数用于测试张量的备选构造方式
def test_tensor_alternative_construction():
    # 创建一个张量索引类型对象"L"
    L = TensorIndexType("L")
    # 创建四个张量索引对象"i0, i1, i2, i3"，它们都属于类型"L"
    i0, i1, i2, i3 = tensor_indices('i0:4', L)
    # 创建一个张量头对象"A"，它的索引类型是[L]
    A = TensorHead("A", [L])
    # 创建符号"x, y"
    x, y = symbols("x y")

    # 断言：A(i0)应该等于A(Symbol("i0"))
    assert A(i0) == A(Symbol("i0"))
    # 断言：A(-i0)应该等于A(-Symbol("i0"))
    assert A(-i0) == A(-Symbol("i0"))
    # 断言：lambda表达式应该引发TypeError异常，因为x+y不是有效的张量索引
    raises(TypeError, lambda: A(x+y))
    # 断言：lambda表达式应该引发ValueError异常，因为2*x不是有效的张量索引
    raises(ValueError, lambda: A(2*x))


# 定义一个函数用于测试张量的替换功能
def test_tensor_replacement():
    # 创建一个张量索引类型对象"L"
    L = TensorIndexType("L")
    # 创建一个维度为2的张量索引类型对象"L2"
    L2 = TensorIndexType("L2", dim=2)
    # 创建四个张量索引对象"i, j, k, l"，它们都属于类型"L"
    i, j, k, l = tensor_indices("i j k l", L)
    # 创建四个张量头对象"A, B, C, D"，它们的索引类型都是[L]
    A, B, C, D = tensor_heads("A B C D", [L])
    # 创建一个张量头对象"H"，它的索引类型是[L, L]
    H = TensorHead("H", [L, L])
    # 创建一个张量头对象"K"，它的索引类型是[L, L, L, L]
    K = TensorHead("K", [L]*4)

    # 创建一个表示H(i, j)的表达式
    expr = H(i, j)
    # 创建一个替换字典repl，用于替换H(i, -j)的值
    repl = {H(i,-j): [[1,2],[3,4]], L: diag(1, -1)}
    # 断言：_extract_data方法应该返回正确的索引列表[i, j]和对应的数组Array([[1, -2], [3, -4]])
    assert expr._extract_data(repl) == ([i, j], Array([[1, -2], [3, -4]]))

    # 断言：replace_with_arrays方法应该返回Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])
    # 断言：replace_with_arrays方法指定了索引[i, j]后，应该返回Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, -2], [3, -4]])
    # 断言：replace_with_arrays方法指定了索引[i, -j]后，应该返回Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, 2], [3, 4]])
    # 断言：replace_with_arrays方法指定了索引[Symbol("i"), -Symbol("j")]后，应该返回Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [Symbol("i"), -Symbol("j")]) == Array([[1, 2], [3, 4]])
    # 断言：replace_with_arrays方法指定了索引[-i, j]后，应该返回Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, -2], [-3, 4]])
    # 断言：replace_with_arrays方法指定了索引[-i, -j]后，应该返回Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, 2], [-3, -4]])
    # 断言：replace_with_arrays方法指定了索引[j, i]后，应该返回Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [-2, -4]])
    # 断言：replace_with_arrays方法指定了索引[j, -i]后，应该返回Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [2, -4]])
    # 断言：replace_with_arrays方法指定了索引[-j, i]后，应该返回Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [2, 4]])
    # 断言：replace_with_arrays方法指定了索引[-j, -i]后，应该返回Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [2, -4]])
    # 断言：replace_with_arrays方法不指定indices参数时，应该返回Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])

    # 重新定义表达式expr为H(i, j)
    expr = H(i,j)
    # 重新定义替换字典repl，用于替换H(i, j)的值
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    # 断言：_extract_data方法应该返回正确的索引列表[i, j]和对应的数组Array([[1, 2], [3, 4]])
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))

    # 断言：replace_with_arrays方法应该返回Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl) == Array([[1, 2], [3, 4]])
    # 断言：replace_with_arrays方法指定了索引[i, j]后，应该返回Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, 2], [3, 4]])
    # 断言：replace_with_arrays方法指定了索引[i, -j]后，应该返回Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, -2], [3, -4]])
    # 断言：replace_with_arrays方法指定了索引[-i, j]后，应该返回Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, 2], [-3, -4]])
    # 断言：replace_with_arrays方法指定了索引[-i, -j]后，应该返回Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, -2], [-3, 4]])
    # 断言：replace_with_arrays方法指定了索引[j, i]后，应该返回Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [2, 4]])
    # 断言：replace_with_arrays方法指定了索引[j, -i]后，应该返回Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [2, -4]])
    # 断言：replace_with_arrays方法指定了索引[-j, i]后，应该返回Array([[1, 3], [-2, -
    # 断言表达式的提取数据操作成功
    assert expr._extract_data(repl)

    # 初始化表达式和替换字典，期望引发 ValueError 异常
    expr = H(j, k)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    raises(ValueError, lambda: expr._extract_data(repl))

    # 初始化表达式和替换字典，期望引发 ValueError 异常
    expr = A(i)
    repl = {B(i): [1, 2]}
    raises(ValueError, lambda: expr._extract_data(repl))

    # 初始化表达式和替换字典，期望引发 ValueError 异常
    expr = A(i)
    repl = {A(i): [[1, 2], [3, 4]]}
    raises(ValueError, lambda: expr._extract_data(repl))

    # TensAdd:
    # 初始化表达式和替换字典，验证提取数据后的结果是否正确
    expr = A(k)*H(i, j) + B(k)*H(i, j)
    repl = {A(k): [1], B(k): [1], H(i, j): [[1, 2],[3,4]], L:diag(1,1)}
    assert expr._extract_data(repl) == ([k, i, j], Array([[[2, 4], [6, 8]]]))
    # 验证替换数组后的结果是否正确
    assert expr.replace_with_arrays(repl, [k, i, j]) == Array([[[2, 4], [6, 8]]])
    assert expr.replace_with_arrays(repl, [k, j, i]) == Array([[[2, 6], [4, 8]]])

    # 初始化表达式和替换字典，验证替换数组后的结果是否正确
    expr = A(k)*A(-k) + 100
    repl = {A(k): [2, 3], L: diag(1, 1)}
    assert expr.replace_with_arrays(repl, []) == 113

    ## Symmetrization:
    # 初始化表达式和替换字典，验证提取数据后的结果是否正确
    expr = H(i, j) + H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[2, 5], [5, 8]]))
    # 验证替换数组后的结果是否正确
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[2, 5], [5, 8]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[2, 5], [5, 8]])

    ## Anti-symmetrization:
    # 初始化表达式和替换字典，验证替换数组后的结果是否正确
    expr = H(i, j) - H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[0, -1], [1, 0]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[0, 1], [-1, 0]])

    # Tensors with contractions in replacements:
    # 初始化表达式和替换字典，验证提取数据后的结果是否正确
    expr = K(i, j, k, -k)
    repl = {K(i, j, k, -k): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))

    # 初始化表达式和替换字典，验证提取数据后的结果是否正确
    expr = H(i, -i)
    repl = {H(i, -i): 42}
    assert expr._extract_data(repl) == ([], 42)

    # 初始化表达式和替换字典，验证提取数据后的结果是否正确
    expr = H(i, -i)
    repl = {
        H(-i, -j): Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        L: Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
    }
    assert expr._extract_data(repl) == ([], 4)

    # Replace with array, raise exception if indices are not compatible:
    # 初始化表达式和替换字典，期望引发 ValueError 异常，因为索引不兼容
    expr = A(i)*A(j)
    repl = {A(i): [1, 2]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [j]))

    # Raise exception if array dimension is not compatible:
    # 初始化表达式和替换字典，期望引发 ValueError 异常，因为数组维度不兼容
    expr = A(i)
    repl = {A(i): [[1, 2]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [i]))

    # TensorIndexType with dimension, wrong dimension in replacement array:
    # 初始化表达式和替换字典，期望引发 ValueError 异常，因为替换数组中的维度不正确
    u1, u2, u3 = tensor_indices("u1:4", L2)
    U = TensorHead("U", [L2])
    expr = U(u1)*U(-u2)
    repl = {U(u1): [[1]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [u1, -u2]))
def test_rewrite_tensor_to_Indexed():
    # 定义张量索引类型L，维度为4
    L = TensorIndexType("L", dim=4)
    # 定义张量头A，包含四个L类型的指标
    A = TensorHead("A", [L]*4)
    # 定义张量头B，包含一个L类型的指标
    B = TensorHead("B", [L])

    # 定义符号变量i0, i1, i2, i3
    i0, i1, i2, i3 = symbols("i0:4")
    # 定义符号变量L_0, L_1
    L_0, L_1 = symbols("L_0:2")

    # 创建张量A的实例a1，包含指标i0, i1, i2, i3
    a1 = A(i0, i1, i2, i3)
    # 断言a1重写为Indexed类，表示为Indexed(Symbol("A"), i0, i1, i2, i3)
    assert a1.rewrite(Indexed) == Indexed(Symbol("A"), i0, i1, i2, i3)

    # 创建张量A的实例a2，包含指标i0, -i0, i2, i3
    a2 = A(i0, -i0, i2, i3)
    # 断言a2重写为Sum类，表示为Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3))
    assert a2.rewrite(Indexed) == Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3))

    # 创建a2和A(i2, i3, i0, -i0)的和a3
    a3 = a2 + A(i2, i3, i0, -i0)
    # 断言a3重写为Sum类，表示为Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3)) +
    # Sum(Indexed(Symbol("A"), i2, i3, L_0, L_0), (L_0, 0, 3))
    assert a3.rewrite(Indexed) == \
        Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3)) +\
        Sum(Indexed(Symbol("A"), i2, i3, L_0, L_0), (L_0, 0, 3))

    # 创建张量B的实例b1，包含指标-i0，乘以张量A的实例a1
    b1 = B(-i0)*a1
    # 断言b1重写为Sum类，表示为Sum(Indexed(Symbol("B"), L_0)*Indexed(Symbol("A"), L_0, i1, i2, i3), (L_0, 0, 3))
    assert b1.rewrite(Indexed) == Sum(Indexed(Symbol("B"), L_0)*Indexed(Symbol("A"), L_0, i1, i2, i3), (L_0, 0, 3))

    # 创建张量B的实例b2，包含指标-i3，乘以张量A的实例a2
    b2 = B(-i3)*a2
    # 断言b2重写为Sum类，表示为Sum(Indexed(Symbol("B"), L_1)*Indexed(Symbol("A"), L_0, L_0, i2, L_1), (L_0, 0, 3), (L_1, 0, 3))
    assert b2.rewrite(Indexed) == Sum(Indexed(Symbol("B"), L_1)*Indexed(Symbol("A"), L_0, L_0, i2, L_1), (L_0, 0, 3), (L_1, 0, 3))

def test_tensor_matching():
    """
    Test match and replace with the pattern being a WildTensor or a WildTensorIndex
    """
    # 定义三维张量索引类型R3
    R3 = TensorIndexType('R3', dim=3)
    # 定义张量索引p, q, r
    p, q, r = tensor_indices("p q r", R3)
    # 定义符号变量a, b, c，它们是WildTensorIndex类，张量索引类型为R3，忽略升降符号
    a, b, c = symbols("a b c", cls=WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
    # 定义WildTensorIndex实例g，张量索引类型为R3
    g = WildTensorIndex("g", R3)
    # 定义R3类型的度量delta和Levi-Civita符号eps
    delta = R3.delta
    eps = R3.epsilon
    # 定义张量头K, V, A，分别包含R3类型的指标
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    A = TensorHead("A", [R3, R3])
    # 定义WildTensorHead实例W，未排序的张量索引
    W = WildTensorHead('W', unordered_indices=True)
    # 定义WildTensorHead实例U，通用的WildTensorHead

    # 断言a匹配q，返回匹配结果字典{a: q}
    assert a.matches(q) == {a: q}
    # 断言a匹配-q，返回匹配结果字典{a: -q}
    assert a.matches(-q) == {a: -q}
    # 断言g不匹配-q，返回None
    assert g.matches(-q) == None
    # 断言g匹配q，返回匹配结果字典{g: q}
    assert g.matches(q) == {g: q}
    # 断言eps(p, -a, a)不匹配eps(p, q, r)，返回None
    assert eps(p, -a, a).matches(eps(p, q, r)) == None
    # 断言eps(p, -b, a)匹配eps(p, q, r)，返回匹配结果字典{a: r, -b: q}
    assert eps(p, -b, a).matches(eps(p, q, r)) == {a: r, -b: q}
    # 断言eps(p, -q, r)替换eps(a, b, c)为1，返回1
    assert eps(p, -q, r).replace(eps(a, b, c), 1) == 1
    # 断言W()匹配K(p)*V(q)，返回匹配结果字典{W(): K(p)*V(q)}
    assert W().matches(K(p)*V(q)) == {W(): K(p)*V(q)}
    # 断言W(a)匹配K(p)，返回匹配结果字典{a: p, W(a).head: _WildTensExpr(K(p))}
    assert W(a).matches(K(p)) == {a: p, W(a).head: _WildTensExpr(K(p))}
    # 断言W(a, p)匹配K(p)*V(q)，返回匹配结果字典{a: q, W(a, p).head: _WildTensExpr(K(p)*V(q))}
    assert W(a, p).matches(K(p)*V(q)) == {a: q, W(a, p).head: _WildTensExpr(K(p)*V(q))}
    # 断言W(p, q)匹配K(p)*V(q)，返回匹配结果字典{W(p, q).head: _WildTensExpr(K(p)*V(q))}
    assert W(p, q).matches(K(p)*V(q)) == {W(p, q).head: _WildTensExpr(K(p)*V(q))}
    # 断言W(p, q)匹配A(q, p)，返回匹配结果字典{W(p, q).head: _WildTensExpr(A(q, p))}
    assert W(p, q).matches(A(q, p)) == {W(p, q).head: _WildTensExpr(A(q, p))}
    # 断言U(p, q)不匹配A(q, p)，返回None
    assert U(p, q).matches(A(q, p)) == None
    # 断言(K(q)*K(p)).replace(W(q, p), 1)为1
    assert (K(q)*K(p)).replace(W(q
    # 断言语句，验证一个复杂的数学表达式是否成立
    assert (
        # 计算表达式中的第一项
        K(p)*K(q) + V(p)*V(q) + K(p)*V(q) + K(q)*V(p)
    ).replace(
        # 在表达式中替换第一个参数为第二个参数
        W(p,q) + K(p)*K(q) + V(p)*V(q),
        # 替换后的结果
        W(p,q) + 3*K(p)*V(q)
    ).doit() == K(q)*V(p) + 4*K(p)*V(q)
def test_TensMul_matching():
    """
    Test match and replace with the pattern being a TensMul
    """
    # 定义一个维数为3的张量指标类型'R3'
    R3 = TensorIndexType('R3', dim=3)
    # 定义五个张量指标 p, q, r, s, t，并分配给指标类型 R3
    p, q, r, s, t = tensor_indices("p q r s t", R3)
    # 定义一个通配符 'wi'
    wi = Wild("wi")
    # 定义六个通配符 a, b, c, d, e, f，其类型为 WildTensorIndex，指标类型为 R3，忽略上下标的区别
    a, b, c, d, e, f = symbols("a b c d e f", cls=WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
    # 定义 R3 类型的 delta 符号
    delta = R3.delta
    # 定义 R3 类型的 epsilon 符号
    eps = R3.epsilon
    # 定义一个张量头 'K'，参数为 R3 类型的列表
    K = TensorHead("K", [R3])
    # 定义一个张量头 'V'，参数为 R3 类型的列表
    V = TensorHead("V", [R3])
    # 定义一个具有无序指标的通配符张量头 'W'
    W = WildTensorHead('W', unordered_indices=True)
    # 定义一个通配符张量头 'U'
    U = WildTensorHead('U')
    # 定义一个普通符号 'k'
    k = Symbol("K")

    # 断言语句：验证 wi*K(p) 与 K(p) 的匹配，返回 {wi: 1}
    assert (wi * K(p)).matches(K(p)) == {wi: 1}
    # 断言语句：验证 wi * eps(p,q,r) 与 eps(p,r,q) 的匹配，返回 {wi: -1}
    assert (wi * eps(p, q, r)).matches(eps(p, r, q)) == {wi: -1}
    # 断言语句：验证 K(p)*V(-p) 替换为 W(a)*V(-a)，结果为 1
    assert (K(p) * V(-p)).replace(W(a) * V(-a), 1) == 1
    # 断言语句：验证 K(q)*K(p)*V(-p) 替换为 W(q,a)*V(-a)，结果为 1
    assert (K(q) * K(p) * V(-p)).replace(W(q, a) * V(-a), 1) == 1
    # 断言语句：验证 K(p)*V(-p) 替换为 K(-a)*V(a)，结果为 1
    assert (K(p) * V(-p)).replace(K(-a) * V(a), 1) == 1
    # 断言语句：验证 K(q)*K(p)*V(-p) 替换为 W(q)*U(p)*V(-p)，结果为 1
    assert (K(q) * K(p) * V(-p)).replace(W(q) * U(p) * V(-p), 1) == 1
    # 断言语句：验证 (K(p)*V(q)).replace(W()*K(p)*V(q), W()*V(p)*V(q)).doit() 结果为 V(p)*V(q)
    assert (
        (K(p) * V(q)).replace(W() * K(p) * V(q), W() * V(p) * V(q)).doit()
        == V(p) * V(q)
        )
    # 断言语句：验证 ( eps(r,p,q)*eps(-r,-s,-t) ).replace(eps(e,a,b)*eps(-e,c,d), delta(a,c)*delta(b,d) - delta(a,d)*delta(b,c)).doit().canon_bp() 结果为 delta(p,-s)*delta(q,-t) - delta(p,-t)*delta(q,-s)
    assert (
        (eps(r, p, q) * eps(-r, -s, -t)).replace(
            eps(e, a, b) * eps(-e, c, d),
            delta(a, c) * delta(b, d) - delta(a, d) * delta(b, c),
            ).doit().canon_bp()
        == delta(p, -s) * delta(q, -t) - delta(p, -t) * delta(q, -s)
        )
    # 断言语句：验证 ( eps(r,p,q)*eps(-r,-p,-q) ).replace(eps(c,a,b)*eps(-c,d,f), delta(a,d)*delta(b,f) - delta(a,f)*delta(b,d)).contract_delta(delta).doit() 结果为 6
    assert (
        (eps(r, p, q) * eps(-r, -p, -q)).replace(
            eps(c, a, b) * eps(-c, d, f),
            delta(a, d) * delta(b, f) - delta(a, f) * delta(b, d),
            ).contract_delta(delta).doit()
        == 6
        )
    # 断言语句：验证 V(-p)*V(q)*V(-q) 替换为 wi*W()*V(a)*V(-a)，结果为 wi*W()
    assert (V(-p) * V(q) * V(-q)).replace(wi * W() * V(a) * V(-a), wi * W()).doit() == V(-p)
    # 断言语句：验证 k**4*K(r)*K(-r) 替换为 wi*W()*K(a)*K(-a)，结果为 wi*W()*k**2
    assert (k**4 * K(r) * K(-r)).replace(wi * W() * K(a) * K(-a), wi * W() * k**2).doit() == k**6

    # 多次出现的 WildTensorHead 在值中
    # 断言语句：验证 K(p)*V(q) 替换为 W(q)*K(p)，结果为 V(p)*V(q)
    assert (
        (K(p) * V(q)).replace(W(q) * K(p), W(p) * W(q))
        == V(p) * V(q)
        )
    # 断言语句：验证 K(p)*V(q)*V(r) 替换为 W(q,r)*K(p)，结果为 V(p)*V(r)*V(q)*V(s)*V(-s)
    assert (
        (K(p) * V(q) * V(r)).replace(W(q, r) * K(p), W(p, r) * W(q, s) * V(-s))
        == V(p) * V(r) * V(q) * V(s) * V(-s)
        )

    # 边界情况，涉及自动索引重新标记
    # 创建四个张量指标 'R_0 R_1 R_2 R_3'，分配给指标类型 R3
    D0, D1, D2, D3 = tensor_indices("R_0 R_1 R_2 R_3", R3)
    # 创建表达式 delta(-D0, -D1)*K(D2)*K(D3)*K(-D3)
    expr = delta(-D0, -D1) * K(D2) * K(D3) * K(-D3)
    # 使用 ( W()*K(a)*K(-a) ).matches(expr) 匹配表达式
    m = (W() * K(a) * K(-a)).matches(expr)
    # 断言语句：验证 D2 不在匹配结果的值中
    assert D2 not in m.values()

def test_TensMul_subs():
    """
    Test subs and xreplace in TensMul. See bug #24337
    """
    # 定义一个维数为3的张量指标类型'R3'
    R3 = TensorIndexType('R3', dim=3)
    # 定义三个张量指标 p, q, r，并分配给指标类型 R3
    p, q, r = tensor_indices("p q r", R3)
    # 定义一个张量头 'K'，参数为 R3 类型的列表
    K = TensorHead("K", [R3])
    # 定义一个张量头 'V'，参数为 R3 类型的列表
    V = TensorHead("V", [R3])
    # 定义一个张量头 'A'，参数为 R3 类型的列表
    A = TensorHead("A", [R3, R3])
    # 创建一个张量指标 C0，与 R3.dummy_name +
    # 使用 `warns_deprecated_sympy()` 上下文管理器捕获关于 SymPy 废弃警告
    with warns_deprecated_sympy():
        # 创建一个完全对称的张量符号 `sym2`，阶数为 2
        sym2 = TensorSymmetry.fully_symmetric(2)
        # 创建一个张量索引类型 'Lorentz'
        Lorentz = TensorIndexType('Lorentz')
        # 创建一个张量类型 `S2`，包含两个 'Lorentz' 类型的索引，使用 `sym2` 的对称性
        S2 = TensorType([Lorentz]*2, sym2)
        # 断言 `S2` 是 `TensorType` 的实例，用于验证类型
        assert isinstance(S2, TensorType)
# 定义一个名为 test_dummy_fmt 的测试函数
def test_dummy_fmt():
    # 使用 warns_deprecated_sympy 上下文管理器来测试
    with warns_deprecated_sympy():
        # 创建一个名为 'Lorentz' 的 TensorIndexType 对象，设置 dummy_fmt='L'
        TensorIndexType('Lorentz', dummy_fmt='L')

# 定义一个名为 test_postprocessor 的测试函数
def test_postprocessor():
    """
    测试当将 Tensor 插入 Mul 或 Add 时，是否自动转换为 TensMul 或 TensAdd，参见 GitHub 问题 #25051
    """
    # 创建一个维数为 3 的 TensorIndexType 对象，名称为 'R3'
    R3 = TensorIndexType('R3', dim=3)
    # 创建一个 R3 类型的张量索引对象 'i'
    i = tensor_indices("i", R3)
    # 创建一个头部名称为 'K' 的张量，参数是一个包含 R3 的列表
    K = TensorHead("K", [R3])
    # 创建符号变量 x, y, z
    x, y, z = symbols("x y z")

    # 断言：当将 x*2 替换为 {x: K(i)} 时，结果是否为 TensMul 类型
    assert isinstance((x*2).xreplace({x: K(i)}), TensMul)
    # 断言：当将 x+2 替换为 {x: K(i)*K(-i)} 时，结果是否为 TensAdd 类型
    assert isinstance((x+2).xreplace({x: K(i)*K(-i)}), TensAdd)

    # 断言：当将 x*2 替换为 {x: K(i)} 时，结果是否为 TensMul 类型
    assert isinstance((x*2).subs({x: K(i)}), TensMul)
    # 断言：当将 x+2 替换为 {x: K(i)*K(-i)} 时，结果是否为 TensAdd 类型
    assert isinstance((x+2).subs({x: K(i)*K(-i)}), TensAdd)

    # 断言：当将 x*2 替换为 K(i) 时，结果是否为 TensMul 类型
    assert isinstance((x*2).replace(x, K(i)), TensMul)
    # 断言：当将 x+2 替换为 K(i)*K(-i) 时，结果是否为 TensAdd 类型
    assert isinstance((x+2).replace(x, K(i)*K(-i)), TensAdd)


这段代码是关于张量代数操作的测试代码，测试了在不同情况下将数学表达式中的变量替换为张量对象，并验证替换后的类型是否正确。
```