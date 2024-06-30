# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_cse.py`

```
# 从 functools 模块导入 reduce 函数
from functools import reduce
# 导入 itertools 模块，用于高效的迭代工具
import itertools
# 从 operator 模块导入 add 函数，用于加法操作
from operator import add

# 从 sympy.codegen.matrix_nodes 模块导入 MatrixSolve 类
from sympy.codegen.matrix_nodes import MatrixSolve
# 从 sympy.core.add 模块导入 Add 类
from sympy.core.add import Add
# 从 sympy.core.containers 模块导入 Tuple 类
from sympy.core.containers import Tuple
# 从 sympy.core.expr 模块导入 UnevaluatedExpr 类
from sympy.core.expr import UnevaluatedExpr
# 从 sympy.core.function 模块导入 Function 类
from sympy.core.function import Function
# 从 sympy.core.mul 模块导入 Mul 类
from sympy.core.mul import Mul
# 从 sympy.core.power 模块导入 Pow 类
from sympy.core.power import Pow
# 从 sympy.core.relational 模块导入 Eq 类
from sympy.core.relational import Eq
# 从 sympy.core.singleton 模块导入 S 对象
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol 和 symbols 函数
from sympy.core.symbol import Symbol, symbols
# 从 sympy.core.sympify 模块导入 sympify 函数，用于将输入转换为 SymPy 对象
from sympy.core.sympify import sympify
# 从 sympy.functions.elementary.exponential 模块导入 exp 函数
from sympy.functions.elementary.exponential import exp
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入 Piecewise 类
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.elementary.trigonometric 模块导入 cos 和 sin 函数
from sympy.functions.elementary.trigonometric import (cos, sin)
# 从 sympy.matrices.dense 模块导入 Matrix 类
from sympy.matrices.dense import Matrix
# 从 sympy.matrices.expressions 模块导入 Inverse, MatAdd, MatMul, Transpose 类
from sympy.matrices.expressions import Inverse, MatAdd, MatMul, Transpose
# 从 sympy.polys.rootoftools 模块导入 CRootOf 类
from sympy.polys.rootoftools import CRootOf
# 从 sympy.series.order 模块导入 O 类
from sympy.series.order import O
# 从 sympy.simplify.cse_main 模块导入 cse 函数
from sympy.simplify.cse_main import cse
# 从 sympy.simplify.simplify 模块导入 signsimp 函数
from sympy.simplify.simplify import signsimp
# 从 sympy.tensor.indexed 模块导入 Idx, IndexedBase 类
from sympy.tensor.indexed import (Idx, IndexedBase)

# 从 sympy.core.function 模块导入 count_ops 函数
from sympy.core.function import count_ops
# 从 sympy.simplify.cse_opts 模块导入 sub_pre, sub_post 函数
from sympy.simplify.cse_opts import sub_pre, sub_post
# 从 sympy.functions.special.hyper 模块导入 meijerg 函数
from sympy.functions.special.hyper import meijerg
# 从 sympy.simplify 模块导入 cse_main, cse_opts 模块
from sympy.simplify import cse_main, cse_opts
# 从 sympy.utilities.iterables 模块导入 subsets 函数
from sympy.utilities.iterables import subsets
# 从 sympy.testing.pytest 模块导入 XFAIL, raises 函数
from sympy.testing.pytest import XFAIL, raises
# 从 sympy.matrices 模块导入 MutableDenseMatrix, MutableSparseMatrix,
# ImmutableDenseMatrix, ImmutableSparseMatrix 类
from sympy.matrices import (MutableDenseMatrix, MutableSparseMatrix,
        ImmutableDenseMatrix, ImmutableSparseMatrix)
# 从 sympy.matrices.expressions 模块导入 MatrixSymbol 类
from sympy.matrices.expressions import MatrixSymbol

# 创建符号变量 w, x, y, z
w, x, y, z = symbols('w,x,y,z')
# 创建符号变量 x0 到 x12
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = symbols('x:13')


# 定义测试函数 test_numbered_symbols
def test_numbered_symbols():
    # 使用 cse_main.numbered_symbols 函数生成以 'y' 为前缀的编号符号生成器 ns
    ns = cse_main.numbered_symbols(prefix='y')
    # 断言从 ns 中取前 10 个符号与列表 [Symbol('y%s' % i) for i in range(0, 10)] 相等
    assert list(itertools.islice(
        ns, 0, 10)) == [Symbol('y%s' % i) for i in range(0, 10)]
    # 重新生成编号符号生成器 ns，仍使用 'y' 为前缀
    ns = cse_main.numbered_symbols(prefix='y')
    # 断言从 ns 中取 10 到 20 之间的符号与列表 [Symbol('y%s' % i) for i in range(10, 20)] 相等
    assert list(itertools.islice(
        ns, 10, 20)) == [Symbol('y%s' % i) for i in range(10, 20)]
    # 重新生成编号符号生成器 ns，使用默认前缀 'x'
    ns = cse_main.numbered_symbols()
    # 断言从 ns 中取前 10 个符号与列表 [Symbol('x%s' % i) for i in range(0, 10)] 相等
    assert list(itertools.islice(
        ns, 0, 10)) == [Symbol('x%s' % i) for i in range(0, 10)]

# Dummy "optimization" functions for testing.


# 定义优化函数 opt1，返回表达式 expr + y
def opt1(expr):
    return expr + y

# 定义优化函数 opt2，返回表达式 expr * z
def opt2(expr):
    return expr*z

# 定义测试函数 test_preprocess_for_cse
def test_preprocess_for_cse():
    # 断言 cse_main.preprocess_for_cse(x, [(opt1, None)]) 的结果为 x + y
    assert cse_main.preprocess_for_cse(x, [(opt1, None)]) == x + y
    # 断言 cse_main.preprocess_for_cse(x, [(None, opt1)]) 的结果为 x
    assert cse_main.preprocess_for_cse(x, [(None, opt1)]) == x
    # 断言 cse_main.preprocess_for_cse(x, [(None, None)]) 的结果为 x
    assert cse_main.preprocess_for_cse(x, [(None, None)]) == x
    # 断言 cse_main.preprocess_for_cse(x, [(opt1, opt2)]) 的结果为 x + y
    assert cse_main.preprocess_for_cse(x, [(opt1, opt2)]) == x + y
    # 断言 cse_main.preprocess_for_cse(x, [(opt1, None), (opt2, None)]) 的结果为 (x + y) * z
    assert cse_main.preprocess_for_cse(
        x, [(opt1, None), (opt2, None)]) == (x + y)*z

# 定义测试函数 test_postprocess_for_cse
def test_postprocess_for_cse():
    # 断言 cse_main.postprocess_for_cse(x, [(opt1, None)]) 的结果为 x
    assert cse_main.postprocess_for_cse(x, [(opt1, None)]) == x
    # 断言 cse_main.postprocess_for_cse(x, [(None, opt1)]) 的结果为 x + y
    assert cse_main.postprocess_for_cse(x, [(None, opt1)]) == x + y
    # 断言 cse_main.postprocess_for_cse(x, [(None, None)]) 的结果为 x
    assert cse_main.postprocess_for_cse(x, [(None, None)]) == x
    # 断言 cse_main.postprocess_for_cse(x, [(opt1, opt2)]) 的结果为 x * z
    assert cse_main.postprocess_for_cse(x, [(opt1, opt
    # 使用断言验证 cse_main.postprocess_for_cse 函数的返回值是否等于 x*z + y
    assert cse_main.postprocess_for_cse(
        x, [(None, opt1), (None, opt2)]) == x*z + y
def test_cse_single():
    # 简单的替换操作。
    e = Add(Pow(x + y, 2), sqrt(x + y))
    # 进行公共子表达式消除（CSE），返回替换列表和简化后的表达式列表
    substs, reduced = cse([e])
    # 检查替换列表和简化后的表达式列表是否符合预期
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]

    # 对常数进行CSE，测试issue_15082
    subst42, (red42,) = cse([42])
    assert len(subst42) == 0 and red42 == 42
    subst_half, (red_half,) = cse([0.5])
    assert len(subst_half) == 0 and red_half == 0.5


def test_cse_single2():
    # 简单的替换操作，测试直接传递表达式的情况
    e = Add(Pow(x + y, 2), sqrt(x + y))
    substs, reduced = cse(e)
    assert substs == [(x0, x + y)]
    assert reduced == [sqrt(x0) + x0**2]
    # 测试传递矩阵表达式的情况
    substs, reduced = cse(Matrix([[1]]))
    assert isinstance(reduced[0], Matrix)

    # 对常数进行CSE，测试issue 15082
    subst42, (red42,) = cse(42)
    assert len(subst42) == 0 and red42 == 42
    subst_half, (red_half,) = cse(0.5)
    assert len(subst_half) == 0 and red_half == 0.5


def test_cse_not_possible():
    # 无法进行替换操作的情况
    e = Add(x, y)
    substs, reduced = cse([e])
    assert substs == []
    assert reduced == [x + y]
    # issue 6329
    eq = (meijerg((1, 2), (y, 4), (5,), [], x) +
          meijerg((1, 3), (y, 4), (5,), [], x))
    # 检查issue 6329的情况
    assert cse(eq) == ([], [eq])


def test_nested_substitution():
    # 嵌套替换操作
    e = Add(Pow(w*x + y, 2), sqrt(w*x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, w*x + y)]
    assert reduced == [sqrt(x0) + x0**2]


def test_subtraction_opt():
    # 确保减法被优化
    e = (x - y)*(z - y) + exp((x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [-x0 + exp(-x0)]
    e = -(x - y)*(z - y) + exp(-(x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [x0 + exp(x0)]
    # issue 4077
    n = -1 + 1/x
    e = n/x/(-n)**2 - 1/n/x
    assert cse(e, optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)]) == \
        ([], [0])
    assert cse(((w + x + y + z)*(w - y - z))/(w + x)**3) == \
        ([(x0, w + x), (x1, y + z)], [(w - x1)*(x0 + x1)/x0**3])


def test_multiple_expressions():
    e1 = (x + y)*z
    e2 = (x + y)*w
    substs, reduced = cse([e1, e2])
    assert substs == [(x0, x + y)]
    assert reduced == [x0*z, x0*w]
    l = [w*x*y + z, w*y]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == rsubsts
    assert reduced == [z + x*x0, x0]
    l = [w*x*y, w*x*y + z, w*y]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == rsubsts
    assert reduced == [x1, x1 + z, x0]
    l = [(x - z)*(y - z), x - z, y - z]
    substs, reduced = cse(l)
    rsubsts, _ = cse(reversed(l))
    assert substs == [(x0, -z), (x1, x + x0), (x2, x0 + y)]
    assert rsubsts == [(x0, -z), (x1, x0 + y), (x2, x + x0)]
    # 断言检查变量 reduced 是否等于列表 [x1*x2, x1, x2]
    assert reduced == [x1*x2, x1, x2]
    # 创建列表 l，其中包含表达式 w*y + w + x + y + z 和 w*x*y
    l = [w*y + w + x + y + z, w*x*y]
    # 断言应用 cse 函数对列表 l 进行计算的结果是否等于元组形式的表达式和结果
    assert cse(l) == ([(x0, w*y)], [w + x + x0 + y + z, x*x0])
    # 断言应用 cse 函数对列表 [x + y, x + y + z] 进行计算的结果是否等于元组形式的表达式和结果
    assert cse([x + y, x + y + z]) == ([(x0, x + y)], [x0, z + x0])
    # 断言应用 cse 函数对列表 [x + y, x + z] 进行计算的结果是否等于空列表和结果列表 [x + y, x + z]
    assert cse([x + y, x + z]) == ([], [x + y, x + z])
    # 断言应用 cse 函数对列表 [x*y, z + x*y, x*y*z + 3] 进行计算的结果是否等于元组形式的表达式和结果
    assert cse([x*y, z + x*y, x*y*z + 3]) == ([(x0, x*y)], [x0, z + x0, 3 + x0*z])
# 使用 @XFAIL 装饰器标记的测试函数，表示预期这些测试会失败，主要是非交换乘法项的公共子表达式消除被禁用。
@XFAIL # CSE of non-commutative Mul terms is disabled
def test_non_commutative_cse():
    # 定义三个符号变量 A, B, C，并指定它们为非交换的
    A, B, C = symbols('A B C', commutative=False)
    # 创建一个列表 l，包含两个非交换乘法项 A*B*C 和 A*C
    l = [A*B*C, A*C]
    # 断言公共子表达式消除函数 cse 处理 l 后不会发现任何公共子表达式，返回空的替换列表和原始列表 l
    assert cse(l) == ([], l)
    # 修改 l 包含 A*B*C 和 A*B 两个非交换乘法项
    l = [A*B*C, A*B]
    # 断言公共子表达式消除函数 cse 处理 l 后发现 A*B 是一个公共子表达式，返回替换列表 [(x0, A*B)] 和更新后的列表 [x0*C, x0]
    assert cse(l) == ([(x0, A*B)], [x0*C, x0])


# 测试非交换乘法项的公共子表达式消除是否禁用
def test_bypass_non_commutatives():
    # 定义三个符号变量 A, B, C，并指定它们为非交换的
    A, B, C = symbols('A B C', commutative=False)
    # 创建一个列表 l，包含两个非交换乘法项 A*B*C 和 A*C
    l = [A*B*C, A*C]
    # 断言公共子表达式消除函数 cse 处理 l 后不会发现任何公共子表达式，返回空的替换列表和原始列表 l
    assert cse(l) == ([], l)
    # 修改 l 包含 A*B*C 和 A*B 两个非交换乘法项
    l = [A*B*C, A*B]
    # 断言公共子表达式消除函数 cse 处理 l 后不会发现任何公共子表达式，返回空的替换列表和原始列表 l
    assert cse(l) == ([], l)
    # 修改 l 包含 B*C 和 A*B*C 两个非交换乘法项
    l = [B*C, A*B*C]
    # 断言公共子表达式消除函数 cse 处理 l 后不会发现任何公共子表达式，返回空的替换列表和原始列表 l
    assert cse(l) == ([], l)


@XFAIL # CSE fails when replacing non-commutative sub-expressions
def test_non_commutative_order():
    # 定义三个符号变量 A, B, C，并指定它们为非交换的
    A, B, C = symbols('A B C', commutative=False)
    # 定义一个新的符号变量 x0，并指定它为非交换的
    x0 = symbols('x0', commutative=False)
    # 创建一个列表 l，包含两个项 B+C 和 A*(B+C)
    l = [B+C, A*(B+C)]
    # 断言公共子表达式消除函数 cse 处理 l 后发现 B+C 是一个公共子表达式，返回替换列表 [(x0, B+C)] 和更新后的列表 [x0, A*x0]
    assert cse(l) == ([(x0, B+C)], [x0, A*x0])


@XFAIL # Worked in gh-11232, but was reverted due to performance considerations
def test_issue_10228():
    # 断言公共子表达式消除函数 cse 处理 x*y**2 + x*y 后发现 x*y 是一个公共子表达式，返回替换列表 [(x0, x*y)] 和更新后的列表 [x0*y + x0]
    assert cse([x*y**2 + x*y]) == ([(x0, x*y)], [x0*y + x0])
    # 断言公共子表达式消除函数 cse 处理 x + y, 2*x + y 后发现 x + y 是一个公共子表达式，返回替换列表 [(x0, x + y)] 和更新后的列表 [x0, x + x0]
    assert cse([x + y, 2*x + y]) == ([(x0, x + y)], [x0, x + x0])
    # 断言公共子表达式消除函数 cse 处理 (w + 2*x + y + z, w + x + 1) 后发现 w + x 是一个公共子表达式，返回替换列表 [(x0, w + x)] 和更新后的列表 [x0 + x + y + z, x0 + 1]
    assert cse((w + 2*x + y + z, w + x + 1)) == ([(x0, w + x)], [x0 + x + y + z, x0 + 1])
    # 断言公共子表达式消除函数 cse 处理 ((w + x + y + z)*(w - x))/(w + x) 后发现 w + x 是一个公共子表达式，返回替换列表 [(x0, w + x)] 和更新后的列表 [(x0 + y + z)*(w - x)/x0]
    assert cse(((w + x + y + z)*(w - x))/(w + x)) == ([(x0, w + x)], [(x0 + y + z)*(w - x)/x0])
    # 定义多个符号变量，并将它们用于表达式 exprs
    a, b, c, d, f, g, j, m = symbols('a, b, c, d, f, g, j, m')
    exprs = (d*g**2*j*m, 4*a*f*g*m, a*b*c*f**2)
    # 断言公共子表达式消除函数 cse 处理 exprs 后发现 g*m 和 a*f 是公共子表达式，返回替换列表 [(x0, g*m), (x1, a*f)] 和更新后的列表 [d*g*j*x0, 4*x0*x1, b*c*f*x1]
    assert cse(exprs) == ([(x0, g*m), (x1, a*f)], [d*g*j*x0, 4*x0*x1, b*c*f*x1])


@XFAIL
def test_powers():
    # 断言公共子表达式消除函数 cse 处理 x*y**2 + x*y 后发现 x*y 是一个公共子表达式，返回替换列表 [(x0, x*y)] 和更新后的列表 [x0*y + x0]


def test_issue_4498():
    # 断言公共子表达式消除函数 cse 处理 w/(x - y) + z/(y - x) 后发现没有公共子表达式可消除，返回空的替换列表和更新后的列表 [(w - z)/(x - y)]


def test_issue_4020():
    # 断言公共子表达式消除函数 cse 处理 x**5 + x**4 + x**3 + x**2 后发现 x**2 是一个公共子表达式，返回替换列表 [(x0, x**2)] 和更新后的列表 [x0*(x**3 + x + x0 + 1)]


def test_issue_4203():
    # 断言公共子表达式消除函数 cse 处理 sin(x**x)/x**x 后发现 x**x 是一个公共子表达式，返回替换列表 [(x0, x**x)] 和更新后的列表 [sin(x0)/x0]


def test_issue_6263():
    # 创建一个方程表达式 e
    e = Eq(x*(-x + 1) + x*(x - 1), 0)
    # 断言公共子表达式消除函数 cse 处理 e 后发现没有公共子表达式可消除，返回空的替换列表和更新后的列表 [True]


def test_issue_25043():
    #
    #`
# 断言：简化表达式 x**2 + (1 + 1/x**2)/x**2 的结果
assert cse(x**2 + (1 + 1/x**2)/x**2) == \
    ([(x0, x**2), (x1, 1/x0)], [x0 + x1*(x1 + 1)])

# 断言：简化表达式 1/x**2 + (1 + 1/x**2)*x**2 的结果
assert cse(1/x**2 + (1 + 1/x**2)*x**2) == \
    ([(x0, x**2), (x1, 1/x0)], [x0*(x1 + 1) + x1])

# 断言：简化表达式 cos(1/x**2) + sin(1/x**2) 的结果
assert cse(cos(1/x**2) + sin(1/x**2)) == \
    ([(x0, x**(-2))], [sin(x0) + cos(x0)])

# 断言：简化表达式 cos(x**2) + sin(x**2) 的结果
assert cse(cos(x**2) + sin(x**2)) == \
    ([(x0, x**2)], [sin(x0) + cos(x0)])

# 断言：简化表达式 y/(2 + x**2) + z/x**2/y 的结果
assert cse(y/(2 + x**2) + z/x**2/y) == \
    ([(x0, x**2)], [y/(x0 + 2) + z/(x0*y)])

# 断言：简化表达式 exp(x**2) + x**2*cos(1/x**2) 的结果
assert cse(exp(x**2) + x**2*cos(1/x**2)) == \
    ([(x0, x**2)], [x0*cos(1/x0) + exp(x0)])

# 断言：简化表达式 (1 + 1/x**2)/x**2 的结果
assert cse((1 + 1/x**2)/x**2) == \
    ([(x0, x**(-2))], [x0*(x0 + 1)])

# 断言：简化表达式 x**(2*y) + x**(-2*y) 的结果
assert cse(x**(2*y) + x**(-2*y)) == \
    ([(x0, x**(2*y))], [x0 + 1/x0])
def test_postprocess():
    # 定义等式表达式
    eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))
    # 断言常量表达式的结果
    assert cse([eq, Eq(x, z + 1), z - 2, (z + 1)*(x + 1)],
        postprocess=cse_main.cse_separate) == \
        [[(x0, y + 1), (x2, z + 1), (x, x2), (x1, x + 1)],
        [x1 + exp(x1/x0) + cos(x0), z - 2, x1*x2]]

def test_issue_4499():
    # 之前，这会生成16个常量
    from sympy.abc import a, b
    # 定义元组表达式
    B = Function('B')
    G = Function('G')
    t = Tuple(*
        (a, a + S.Half, 2*a, b, 2*a - b + 1, (sqrt(z)/2)**(-2*a + 1)*B(2*a -
        b, sqrt(z))*B(b - 1, sqrt(z))*G(b)*G(2*a - b + 1),
        sqrt(z)*(sqrt(z)/2)**(-2*a + 1)*B(b, sqrt(z))*B(2*a - b,
        sqrt(z))*G(b)*G(2*a - b + 1), sqrt(z)*(sqrt(z)/2)**(-2*a + 1)*B(b - 1,
        sqrt(z))*B(2*a - b + 1, sqrt(z))*G(b)*G(2*a - b + 1),
        (sqrt(z)/2)**(-2*a + 1)*B(b, sqrt(z))*B(2*a - b + 1,
        sqrt(z))*G(b)*G(2*a - b + 1), 1, 0, S.Half, z/2, -b + 1, -2*a + b,
        -2*a))
    # 进行常量表达式的公共子表达式消除
    c = cse(t)
    # 定义预期结果
    ans = (
        [(x0, 2*a), (x1, -b + x0), (x2, x1 + 1), (x3, b - 1), (x4, sqrt(z)),
         (x5, B(x3, x4)), (x6, (x4/2)**(1 - x0)*G(b)*G(x2)), (x7, x6*B(x1, x4)),
         (x8, B(b, x4)), (x9, x6*B(x2, x4))],
        [(a, a + S.Half, x0, b, x2, x5*x7, x4*x7*x8, x4*x5*x9, x8*x9,
          1, 0, S.Half, z/2, -x3, -x1, -x0)])
    # 断言结果等于预期
    assert ans == c

def test_issue_6169():
    # 定义根的表达式
    r = CRootOf(x**6 - 4*x**5 - 2, 1)
    # 断言常量表达式的公共子表达式消除结果
    assert cse(r) == ([], [r])
    # 并检查使用新机制的正确性
    assert sub_post(sub_pre((-x - y)*z - x - y)) == -z*(x + y) - x - y

def test_cse_Indexed():
    # 定义索引基础
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    i = Idx('i', len_y-1)

    # 定义表达式1和表达式2
    expr1 = (y[i+1]-y[i])/(x[i+1]-x[i])
    expr2 = 1/(x[i+1]-x[i])
    # 进行常量表达式的公共子表达式消除
    replacements, reduced_exprs = cse([expr1, expr2])
    # 断言替换结果的长度大于0
    assert len(replacements) > 0

def test_cse_MatrixSymbol():
    # MatrixSymbol具有非Basic参数，确保能正常工作
    A = MatrixSymbol("A", 3, 3)
    assert cse(A) == ([], [A])

    n = symbols('n', integer=True)
    B = MatrixSymbol("B", n, n)
    assert cse(B) == ([], [B])

    assert cse(A[0] * A[0]) == ([], [A[0]*A[0]])

    assert cse(A[0,0]*A[0,1] + A[0,0]*A[0,1]*A[0,2]) == ([(x0, A[0, 0]*A[0, 1])], [x0*A[0, 2] + x0])

def test_cse_MatrixExpr():
    # 定义MatrixSymbol和表达式
    A = MatrixSymbol('A', 3, 3)
    y = MatrixSymbol('y', 3, 1)

    expr1 = (A.T*A).I * A * y
    expr2 = (A.T*A) * A * y
    # 进行常量表达式的公共子表达式消除
    replacements, reduced_exprs = cse([expr1, expr2])
    # 断言替换结果的长度大于0
    assert len(replacements) > 0

    replacements, reduced_exprs = cse([expr1 + expr2, expr1])
    # 断言替换结果的长度大于0
    assert replacements

    replacements, reduced_exprs = cse([A**2, A + A**2])
    # 断言替换结果的长度大于0
    assert replacements

def test_Piecewise():
    # 定义Piecewise函数
    f = Piecewise((-z + x*y, Eq(y, 0)), (-z - x*y, True))
    # 进行常量表达式的公共子表达式消除
    ans = cse(f)
    # 定义预期结果
    actual_ans = ([(x0, x*y)],
        [Piecewise((x0 - z, Eq(y, 0)), (-z - x0, True))])
    # 断言结果等于预期
    assert ans == actual_ans

def test_ignore_order_terms():
    # 定义表达式
    eq = exp(x).series(x,0,3) + sin(y+x**3) - 1
    # 使用断言来验证 cse(eq) 的返回结果是否符合预期
    assert cse(eq) == ([], [sin(x**3 + y) + x + x**2/2 + O(x**3)])
# 定义一个测试函数，用于测试变量命名冲突的情况
def test_name_conflict():
    # 计算 z1，它是 x0 和 y 的和
    z1 = x0 + y
    # 计算 z2，它是 x2 和 x3 的和
    z2 = x2 + x3
    # 创建列表 l，包含 cos(z1) + z1，cos(z2) + z2，以及 x0 + x2
    l = [cos(z1) + z1, cos(z2) + z2, x0 + x2]
    # 进行常数部分提取 (Common Subexpression Elimination, CSE)，得到替换和简化后的表达式
    substs, reduced = cse(l)
    # 断言简化后的表达式与原始表达式 l 相等
    assert [e.subs(reversed(substs)) for e in reduced] == l


# 定义一个测试函数，测试在自定义符号集合下的变量命名冲突情况
def test_name_conflict_cust_symbols():
    # 计算 z1，它是 x0 和 y 的和
    z1 = x0 + y
    # 计算 z2，它是 x2 和 x3 的和
    z2 = x2 + x3
    # 创建列表 l，包含 cos(z1) + z1，cos(z2) + z2，以及 x0 + x2
    l = [cos(z1) + z1, cos(z2) + z2, x0 + x2]
    # 进行常数部分提取 (Common Subexpression Elimination, CSE)，使用自定义的符号集合进行替换
    substs, reduced = cse(l, symbols("x:10"))
    # 断言简化后的表达式与原始表达式 l 相等
    assert [e.subs(reversed(substs)) for e in reduced] == l


# 定义一个测试函数，测试在符号用尽时引发错误的情况
def test_symbols_exhausted_error():
    # 创建表达式 l，包含 cos(x+y)+x+y+cos(w+y)+sin(w+y)
    l = cos(x+y)+x+y+cos(w+y)+sin(w+y)
    # 定义符号列表 sym，包含 x, y, z
    sym = [x, y, z]
    # 使用 pytest 的 raises 函数检查调用 cse 函数时是否会引发 ValueError 异常
    with raises(ValueError):
        cse(l, symbols=sym)


# 定义一个测试函数，测试特定问题的情况
def test_issue_7840():
    # daveknippers 提供的示例
    # 创建符号 C393 和 C391 的表达式
    C393 = sympify( \
        'Piecewise((C391 - 1.65, C390 < 0.5), (Piecewise((C391 - 1.65, \
        C391 > 2.35), (C392, True)), True))'
    )
    C391 = sympify( \
        'Piecewise((2.05*C390**(-1.03), C390 < 0.5), (2.5*C390**(-0.625), True))'
    )
    # 使用 C391 替换 C393 中的 'C391'
    C393 = C393.subs('C391',C391)
    # 简单的符号替换
    sub = {}
    sub['C390'] = 0.703451854
    sub['C392'] = 1.01417794
    ss_answer = C393.subs(sub)
    # 进行常数部分提取 (CSE)
    substitutions,new_eqn = cse(C393)
    # 对每对替换进行符号替换
    for pair in substitutions:
        sub[pair[0].name] = pair[1].subs(sub)
    # 对新方程进行符号替换
    cse_answer = new_eqn[0].subs(sub)
    # 断言简化后的两种方法得到的结果相同
    assert ss_answer == cse_answer

    # GitRay 提供的示例
    # 创建符号表达式 expr
    expr = sympify(
        "Piecewise((Symbol('ON'), Equality(Symbol('mode'), Symbol('ON'))), \
        (Piecewise((Piecewise((Symbol('OFF'), StrictLessThan(Symbol('x'), \
        Symbol('threshold'))), (Symbol('ON'), true)), Equality(Symbol('mode'), \
        Symbol('AUTO'))), (Symbol('OFF'), true)), true))"
    )
    # 进行常数部分提取 (CSE)
    substitutions, new_eqn = cse(expr)
    # 断言新方程与原始表达式相等
    assert new_eqn[0] == expr
    # 断言没有任何替换
    assert len(substitutions) < 1


# 定义一个测试函数，测试特定问题的情况
def test_issue_8891():
    # 对于四种矩阵类型，进行测试
    for cls in (MutableDenseMatrix, MutableSparseMatrix,
            ImmutableDenseMatrix, ImmutableSparseMatrix):
        # 创建一个特定的矩阵 m
        m = cls(2, 2, [x + y, 0, 0, 0])
        # 进行常数部分提取 (CSE)
        res = cse([x + y, m])
        # 定义预期的答案 ans
        ans = ([(x0, x + y)], [x0, cls([[x0, 0], [0, 0]])])
        # 断言 cse 函数返回的结果与预期的答案相等
        assert res == ans
        # 断言返回的第二个元素是预期的矩阵类型
        assert isinstance(res[1][-1], cls)


# 定义一个测试函数，测试特定问题的情况
def test_issue_11230():
    # 定义一组符号 a, b, f, k, l, i
    a, b, f, k, l, i = symbols('a b f k l i')
    # 创建一个列表 p，包含几个乘法表达式
    p = [a*b*f*k*l, a*i*k**2*l, f*i*k**2*l]
    # 进行常数部分提取 (CSE)
    R, C = cse(p)
    # 断言 C 中没有包含乘法的情况
    assert not any(i.is_Mul for a in C for i in a.args)

    # 对问题进行随机测试
    from sympy.core.random import choice
    from sympy.core.function import expand_mul
    # 创建符号 s
    s = symbols('a:m')
    # 创建 35 个乘法测试用例
    ex = [Mul(*[choice(s) for i in range(5)]) for i in range(7)]
    # 对每一个子集进行测试
    for p in subsets(ex, 3):
        p = list(p)
        # 进行常数部分提取 (CSE)
        R, C = cse(p)
        # 断言 C 中没有包含乘法的情况
        assert not any(i.is_Mul for a in C for i in a.args)
        # 对每一对替换进行符号替换
        for ri in reversed(R):
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        # 断言 p 和 C 相等
        assert p == C
    # 生成一个包含7个随机生成的表达式的列表ex，每个表达式是5个随机选取的s中的元素的和
    ex = [Add(*[choice(s[:7]) for i in range(5)]) for i in range(7)]
    # 遍历ex的所有三元子集p
    for p in subsets(ex, 3):
        # 将p转换为列表形式
        p = list(p)
        # 将p拆解为R和C两部分，其中R是p中每个表达式中的符号集合，C是p中每个表达式的系数集合
        R, C = cse(p)
        # 断言：C中任何一个元素的args属性中不包含Add对象，即C中的每个元素不是加法表达式
        assert not any(i.is_Add for a in C for i in a.args)
        # 逆序遍历R中的元素
        for ri in reversed(R):
            # 遍历C中的每个元素，将ri中的第一个符号用第二个符号进行替换
            for i in range(len(C)):
                C[i] = C[i].subs(*ri)
        # 断言：p经过expand_mul处理后应该等于C中每个元素的展开乘积形式
        # 用于处理如下情况的展开乘积：p = [a + 2*b + 2*e, 2*b + c + 2*e, b + 2*c + 2*g]
        # x0 = 2*(b + e) 被识别并重建p为 `[a + 2*(b + e), c + 2*(b + e), b + 2*c + 2*g]`
        assert p == [expand_mul(i) for i in C]
# 定义一个测试函数，用于测试 Issue 11577
@XFAIL
def test_issue_11577():
    # 定义一个嵌套函数 check，用于验证符号表达式的公共子表达式消除结果
    def check(eq):
        # 对表达式进行公共子表达式消除
        r, c = cse(eq)
        # 断言表达式的操作数数量大于等于以下值的总和：
        # 符号表达式列表 r 的长度，以及 r 中每个元素第二项的操作数数量的总和，以及 count_ops(c)
        assert eq.count_ops() >= \
            len(r) + sum(i[1].count_ops() for i in r) + \
            count_ops(c)

    # 定义一个符号表达式 eq
    eq = x**5*y**2 + x**5*y + x**5
    # 断言公共子表达式消除结果等于以下元组：
    # 第一个元素是包含 (x0, x**4) 和 (x1, x*y) 的列表 r，第二个元素是 [x**5 + x0*x1*y + x0*x1]
    assert cse(eq) == (
        [(x0, x**4), (x1, x*y)], [x**5 + x0*x1*y + x0*x1])
        # 或者可能的结果是：([(x0, x**5*y)], [x0*y + x0 + x**5]) 或 [(x0, x**5)], [x0*y**2 + x0*y + x0]
    # 调用 check 函数验证结果
    check(eq)

    # 定义另一个符号表达式 eq
    eq = x**2/(y + 1)**2 + x/(y + 1)
    # 断言公共子表达式消除结果等于以下元组：
    # 第一个元素是包含 (x0, y + 1) 的列表 r，第二个元素是 [x**2/x0**2 + x/x0]
    assert cse(eq) == (
        [(x0, y + 1)], [x**2/x0**2 + x/x0])
        # 或者可能的结果是：([(x0, x/(y + 1))], [x0**2 + x0])
    # 调用 check 函数验证结果
    check(eq)
    python
    # 定义一个包含数学表达式的列表
    eqs = [(x + y - 1)**2, x,
        x + y, (x + y)/(2*x + 1) + (x + y - 1)**2,
        (2*x + 1)**(x + y)]
    # 对表达式进行符号计算并返回结果及中间步骤
    r, e = cse(eqs, postprocess=cse_release_variables)
    # 断言确保返回的结果符合预期
    # 此处的断言验证了返回的元组结构和具体值的正确性
    assert r, e == ([
    (x0, x + y), (x1, (x0 - 1)**2), (x2, 2*x + 1),
    (_3, x0/x2 + x1), (_4, x2**x0), (x2, None), (_0, x1),
    (x1, None), (_2, x0), (x0, None), (_1, x)], (_0, _1, _2, _3, _4))
    # 将符号计算的结果逆序排列
    r.reverse()
    # 筛选出值不为None的项，重新赋值给r
    r = [(s, v) for s, v in r if v is not None]
    # 再次断言验证原始表达式是否与符号计算后的结果匹配
    assert eqs == [i.subs(r) for i in e]
def test_cse_list():
    # 定义一个 lambda 函数 _cse，用于调用 cse 函数，设置 list 参数为 False
    _cse = lambda x: cse(x, list=False)
    # 断言对单个变量 x 调用 _cse 返回空列表和 x 本身
    assert _cse(x) == ([], x)
    # 断言对字符串 'x' 调用 _cse 返回空列表和字符串 'x'
    assert _cse('x') == ([], 'x')
    # 创建包含 x 的列表 it
    it = [x]
    # 遍历包含 list、tuple、set 的元组
    for c in (list, tuple, set):
        # 断言对 c(it) 调用 _cse 返回空列表和应用 c 后的列表
        assert _cse(c(it)) == ([], c(it))
    # 断言对 Tuple(*it) 调用 _cse 返回空列表和应用 Tuple 后的结果
    # Tuple 与 tuple 不同
    assert _cse(Tuple(*it)) == ([], Tuple(*it))
    # 创建包含一个键值对的字典 d
    d = {x: 1}
    # 断言对字典 d 调用 _cse 返回空列表和字典 d 本身
    assert _cse(d) == ([], d)


def test_issue_18991():
    # 创建一个 2x2 的矩阵符号 A
    A = MatrixSymbol('A', 2, 2)
    # 断言对 signsimp(-A * A - A) 的结果为 -A * A - A
    assert signsimp(-A * A - A) == -A * A - A


def test_unevaluated_Mul():
    # 创建一个 Mul 对象列表 m，其中一个元素不进行评估
    m = [Mul(1, 2, evaluate=False)]
    # 断言对 m 调用 cse 返回空列表和 m 本身
    assert cse(m) == ([], m)


def test_cse_matrix_expression_inverse():
    # 创建一个 2x2 的不可变稠密矩阵 A
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    # 创建 Inverse(A) 表达式并调用 cse，保存结果
    cse_expr = cse(Inverse(A))
    # 断言 cse_expr 返回空列表和 [Inverse(A)]
    assert cse_expr == ([], [Inverse(A)])


def test_cse_matrix_expression_matmul_inverse():
    # 创建一个 2x2 的不可变稠密矩阵 A 和一个向量 b
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    b = ImmutableDenseMatrix(symbols('b:2'))
    # 创建 MatMul(Inverse(A), b) 表达式并调用 cse，保存结果
    cse_expr = cse(MatMul(Inverse(A), b))
    # 断言 cse_expr 返回空列表和 [MatMul(Inverse(A), b)]
    assert cse_expr == ([], [MatMul(Inverse(A), b)])


def test_cse_matrix_negate_matrix():
    # 创建一个 2x2 的不可变稠密矩阵 A，以及 MatMul(-1, A) 表达式
    x = MatMul(S.NegativeOne, A)
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [x]，这里不进行常数抽取
    assert cse_expr == ([], [x])


def test_cse_matrix_negate_matmul_not_extracted():
    # 创建两个 2x2 的不可变稠密矩阵 A 和 B，以及 MatMul(-1, A, B) 表达式
    x = MatMul(S.NegativeOne, A, B)
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [x]，这里不进行常数抽取
    assert cse_expr == ([], [x])


@XFAIL  # 嵌套关联操作没有简化规则
def test_cse_matrix_nested_matmul_collapsed():
    # 创建两个 2x2 的不可变稠密矩阵 A 和 B，以及 MatMul(-1, MatMul(A, B)) 表达式
    x = MatMul(S.NegativeOne, MatMul(A, B))
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [MatMul(S.NegativeOne, A, B)]
    assert cse_expr == ([], [MatMul(S.NegativeOne, A, B)])


def test_cse_matrix_optimize_out_single_argument_mul():
    # 创建一个 2x2 的不可变稠密矩阵 A，以及 MatMul(MatMul(MatMul(A))) 表达式
    x = MatMul(MatMul(MatMul(A)))
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [A]
    assert cse_expr == ([], [A])


@XFAIL  # CSE 不支持多个简化操作合并
def test_cse_matrix_optimize_out_single_argument_mul_combined():
    # 创建一个 2x2 的不可变稠密矩阵 A，以及 MatMul(MatAdd(MatMul(MatMul(A))), MatMul(MatMul(A)), MatMul(A), A) 表达式
    x = MatAdd(MatMul(MatMul(MatMul(A))), MatMul(MatMul(A)), MatMul(A), A)
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [MatMul(4, A)]
    assert cse_expr == ([], [MatMul(4, A)])


def test_cse_matrix_optimize_out_single_argument_add():
    # 创建一个 2x2 的不可变稠密矩阵 A，以及 MatAdd(MatAdd(MatAdd(MatAdd(A)))) 表达式
    x = MatAdd(MatAdd(MatAdd(MatAdd(A))))
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [A]
    assert cse_expr == ([], [A])


@XFAIL  # CSE 不支持多个简化操作合并
def test_cse_matrix_optimize_out_single_argument_add_combined():
    # 创建一个 2x2 的不可变稠密矩阵 A，以及 MatMul(MatAdd(MatAdd(MatAdd(A))), MatAdd(MatAdd(A)), MatAdd(A), A) 表达式
    x = MatMul(MatAdd(MatAdd(MatAdd(A))), MatAdd(MatAdd(A)), MatAdd(A), A)
    # 调用 cse 对 x 进行常数抽取，保存结果
    cse_expr = cse(x)
    # 断言 cse_expr 返回空列表和 [MatMul(4, A)]
    assert cse_expr == ([], [MatMul(4, A)])


def test_cse_matrix_expression_matrix_solve():
    # 创建一个 2x2 的不可变稠密矩阵 A
    A = ImmutableDenseMatrix(symbols('A:4')).reshape(2, 2)
    # 创建一个包含符号变量 'b0', 'b1' 的不可变密集矩阵
    b = ImmutableDenseMatrix(symbols('b:2'))
    
    # 使用矩阵 A 对 b 进行求解，得到解 x
    x = MatrixSolve(A, b)
    
    # 对 x 进行公共子表达式消除（Common Subexpression Elimination, CSE）
    cse_expr = cse(x)
    
    # 断言，确保经过 CSE 处理后的表达式 cse_expr 等于空的列表和列表 [x]，表示没有找到重复的子表达式
    assert cse_expr == ([], [x])
def test_cse_matrix_matrix_expression():
    # 创建一个 2x2 的不可变稠密矩阵 X，元素为符号 'X:4' 的符号对象
    X = ImmutableDenseMatrix(symbols('X:4')).reshape(2, 2)
    # 创建一个不可变稠密矩阵 y，元素为符号 'y:2' 的符号对象
    y = ImmutableDenseMatrix(symbols('y:2'))
    # 计算线性代数表达式 b，包括矩阵乘法和逆运算
    b = MatMul(Inverse(MatMul(Transpose(X), X)), Transpose(X), y)
    # 进行公共子表达式消除（CSE）
    cse_expr = cse(b)
    # 创建一个 2x2 的符号矩阵 x0
    x0 = MatrixSymbol('x0', 2, 2)
    # 预期的简化后表达式 reduced_expr_expected
    reduced_expr_expected = MatMul(Inverse(MatMul(x0, X)), x0, y)
    # 断言公共子表达式消除后的结果
    assert cse_expr == ([(x0, Transpose(X))], [reduced_expr_expected])


def test_cse_matrix_kalman_filter():
    """Matthew Rocklin 的 SciPy 2013 演讲中的卡尔曼滤波器示例。

    演讲标题： "Matrix Expressions and BLAS/LAPACK; SciPy 2013 Presentation"

    视频链接：https://pyvideo.org/scipy-2013/matrix-expressions-and-blaslapack-scipy-2013-pr.html

    Notes
    =====

    方程如下：

    new_mu = mu + Sigma*H.T * (R + H*Sigma*H.T).I * (H*mu - data)
           = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    new_Sigma = Sigma - Sigma*H.T * (R + H*Sigma*H.T).I * H * Sigma
              = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H)), Inverse(MatAdd(R, MatMul(H*Sigma*Transpose(H)))), H, Sigma))

    """
    N = 2
    # 创建一个不可变稠密矩阵 mu，元素为符号 'mu:{N}' 的符号对象
    mu = ImmutableDenseMatrix(symbols(f'mu:{N}'))
    # 创建一个不可变稠密矩阵 Sigma，元素为符号 'Sigma:{N * N}' 的符号对象，并重塑为 NxN 矩阵
    Sigma = ImmutableDenseMatrix(symbols(f'Sigma:{N * N}')).reshape(N, N)
    # 创建一个不可变稠密矩阵 H，元素为符号 'H:{N * N}' 的符号对象，并重塑为 NxN 矩阵
    H = ImmutableDenseMatrix(symbols(f'H:{N * N}')).reshape(N, N)
    # 创建一个不可变稠密矩阵 R，元素为符号 'R:{N * N}' 的符号对象，并重塑为 NxN 矩阵
    R = ImmutableDenseMatrix(symbols(f'R:{N * N}')).reshape(N, N)
    # 创建一个不可变稠密矩阵 data，元素为符号 'data:{N}' 的符号对象
    data = ImmutableDenseMatrix(symbols(f'data:{N}'))
    # 计算新的 mu 表达式
    new_mu = MatAdd(mu, MatMul(Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data))))
    # 计算新的 Sigma 表达式
    new_Sigma = MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, Transpose(H), Inverse(MatAdd(R, MatMul(H, Sigma, Transpose(H)))), H, Sigma))
    # 进行公共子表达式消除（CSE）
    cse_expr = cse([new_mu, new_Sigma])
    # 创建两个符号矩阵 x0 和 x1，形状均为 NxN
    x0 = MatrixSymbol('x0', N, N)
    x1 = MatrixSymbol('x1', N, N)
    # 预期的替换结果 replacements_expected
    replacements_expected = [
        (x0, Transpose(H)),
        (x1, Inverse(MatAdd(R, MatMul(H, Sigma, x0)))),
    ]
    # 预期的简化后表达式 reduced_exprs_expected
    reduced_exprs_expected = [
        MatAdd(mu, MatMul(Sigma, x0, x1, MatAdd(MatMul(H, mu), MatMul(S.NegativeOne, data)))),
        MatAdd(Sigma, MatMul(S.NegativeOne, Sigma, x0, x1, H, Sigma)),
    ]
    # 断言公共子表达式消除后的结果
    assert cse_expr == (replacements_expected, reduced_exprs_expected)
```