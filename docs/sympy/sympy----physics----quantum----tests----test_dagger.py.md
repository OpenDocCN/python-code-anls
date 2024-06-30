# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_dagger.py`

```
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices.dense import Matrix

from sympy.physics.quantum.dagger import adjoint, Dagger
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.physics.quantum.operator import Operator, IdentityOperator


def test_scalars():
    # 定义一个复数符号 x
    x = symbols('x', complex=True)
    # 断言 Dagger(x) 等于 x 的共轭
    assert Dagger(x) == conjugate(x)
    # 断言 Dagger(I*x) 等于 -I*x 的共轭
    assert Dagger(I*x) == -I*conjugate(x)

    # 定义一个实数符号 i
    i = symbols('i', real=True)
    # 断言 Dagger(i) 等于 i 本身
    assert Dagger(i) == i

    # 定义一个符号 p
    p = symbols('p')
    # 断言 Dagger(p) 是 adjoint 类的实例
    assert isinstance(Dagger(p), adjoint)

    # 定义一个整数符号 i
    i = Integer(3)
    # 断言 Dagger(i) 等于 i 本身
    assert Dagger(i) == i

    # 定义一个非交换符号 A
    A = symbols('A', commutative=False)
    # 断言 Dagger(A) 的交换性为 False
    assert Dagger(A).is_commutative is False


def test_matrix():
    # 定义一个符号 x
    x = symbols('x')
    # 创建一个复数矩阵 m
    m = Matrix([[I, x*I], [2, 4]])
    # 断言 Dagger(m) 等于 m 的共轭转置
    assert Dagger(m) == m.H


def test_dagger_mul():
    # 创建一个算符 O 和单位算符 I
    O = Operator('O')
    I = IdentityOperator()
    # 断言 Dagger(O)*O 等于 Dagger(O)*O
    assert Dagger(O)*O == Dagger(O)*O
    # 断言 Dagger(O)*O*I 等于 Dagger(O)*O*I
    assert Dagger(O)*O*I == Mul(Dagger(O), O)*I
    # 断言 Dagger(O)*Dagger(O) 等于 Dagger(O) 的平方
    assert Dagger(O)*Dagger(O) == Dagger(O)**2
    # 断言 Dagger(O)*Dagger(I) 等于 Dagger(O)
    assert Dagger(O)*Dagger(I) == Dagger(O)


class Foo(Expr):

    def _eval_adjoint(self):
        # 返回复数 I，用于测试 adjoint 方法
        return I


def test_eval_adjoint():
    # 创建一个 Foo 类实例 f
    f = Foo()
    # 计算 f 的共轭转置
    d = Dagger(f)
    # 断言 d 等于复数 I
    assert d == I

np = import_module('numpy')


def test_numpy_dagger():
    if not np:
        skip("numpy not installed.")

    # 创建一个 NumPy 数组 a
    a = np.array([[1.0, 2.0j], [-1.0j, 2.0]])
    # 计算 a 的共轭转置
    adag = a.copy().transpose().conjugate()
    # 断言 Dagger(a) 等于 adag 的所有元素
    assert (Dagger(a) == adag).all()


scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def test_scipy_sparse_dagger():
    if not np:
        skip("numpy not installed.")
    if not scipy:
        skip("scipy not installed.")
    else:
        sparse = scipy.sparse

    # 创建一个 SciPy 稀疏矩阵 a
    a = sparse.csr_matrix([[1.0 + 0.0j, 2.0j], [-1.0j, 2.0 + 0.0j]])
    # 计算 a 的共轭转置
    adag = a.copy().transpose().conjugate()
    # 断言 Dagger(a) 和 adag 的矩阵范数的差异为 0
    assert np.linalg.norm((Dagger(a) - adag).todense()) == 0.0


def test_unknown():
    """Check treatment of unknown objects.
    Objects without adjoint or conjugate/transpose methods
    are sympified and wrapped in dagger.
    """
    # 创建一个符号 x
    x = symbols("x")
    # 对象 x 没有 adjoint 或 conjugate/transpose 方法，将被 sympify 和包装成 dagger
    result = Dagger(x)
    # 断言 result 的参数为 (x,)，且 result 是 adjoint 类的实例
    assert result.args == (x,) and isinstance(result, adjoint)


def test_unevaluated():
    """Check that evaluate=False returns unevaluated Dagger.
    """
    # 创建一个实数符号 x
    x = symbols("x", real=True)
    # 断言 Dagger(x) 等于 x
    assert Dagger(x) == x
    # 创建一个 evaluate=False 的 Dagger 对象
    result = Dagger(x, evaluate=False)
    # 断言 result 的参数为 (x,)，且 result 是 adjoint 类的实例
    assert result.args == (x,) and isinstance(result, adjoint)
```