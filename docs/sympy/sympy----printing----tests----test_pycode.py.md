# `D:\src\scipysrc\sympy\sympy\printing\tests\test_pycode.py`

```
# 导入所需的符号、表达式和函数定义
from sympy.codegen import Assignment
from sympy.codegen.ast import none
from sympy.codegen.cfunctions import expm1, log1p
from sympy.codegen.scipy_nodes import cosm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core import Expr, Mod, symbols, Eq, Le, Gt, zoo, oo, Rational, Pow
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions import acos, KroneckerDelta, Piecewise, sign, sqrt, Min, Max, cot, acsch, asec, coth, sec
from sympy.logic import And, Or
from sympy.matrices import SparseMatrix, MatrixSymbol, Identity
from sympy.printing.pycode import (
    MpmathPrinter, PythonCodePrinter, pycode, SymPyPrinter
)
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
from sympy.testing.pytest import raises, skip
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayDiagonal, ArrayContraction, ZeroArray, OneArray
from sympy.external import import_module
from sympy.functions.special.gamma_functions import loggamma

# 定义符号变量 x, y, z 和 IndexedBase 对象 p
x, y, z = symbols('x y z')
p = IndexedBase("p")

# 定义测试函数 test_PythonCodePrinter
def test_PythonCodePrinter():
    # 创建 PythonCodePrinter 对象
    prntr = PythonCodePrinter()

    # 断言模块导入为空
    assert not prntr.module_imports

    # 测试不同表达式的输出
    assert prntr.doprint(x**y) == 'x**y'
    assert prntr.doprint(Mod(x, 2)) == 'x % 2'
    assert prntr.doprint(-Mod(x, y)) == '-(x % y)'
    assert prntr.doprint(Mod(-x, y)) == '(-x) % y'
    assert prntr.doprint(And(x, y)) == 'x and y'
    assert prntr.doprint(Or(x, y)) == 'x or y'
    assert prntr.doprint(1/(x+y)) == '1/(x + y)'
    assert not prntr.module_imports

    # 断言模块导入包含 math.pi
    assert prntr.doprint(pi) == 'math.pi'
    assert prntr.module_imports == {'math': {'pi'}}

    # 测试特定数学函数的输出
    assert prntr.doprint(x**Rational(1, 2)) == 'math.sqrt(x)'
    assert prntr.doprint(sqrt(x)) == 'math.sqrt(x)'
    assert prntr.module_imports == {'math': {'pi', 'sqrt'}}

    assert prntr.doprint(acos(x)) == 'math.acos(x)'
    assert prntr.doprint(cot(x)) == '(1/math.tan(x))'
    assert prntr.doprint(coth(x)) == '((math.exp(x) + math.exp(-x))/(math.exp(x) - math.exp(-x)))'
    assert prntr.doprint(asec(x)) == '(math.acos(1/x))'
    assert prntr.doprint(acsch(x)) == '(math.log(math.sqrt(1 + x**(-2)) + 1/x))'

    # 测试赋值表达式的输出
    assert prntr.doprint(Assignment(x, 2)) == 'x = 2'
    
    # 测试分段函数的输出
    assert prntr.doprint(Piecewise((1, Eq(x, 0)),
                        (2, x>6))) == '((1) if (x == 0) else (2) if (x > 6) else None)'
    assert prntr.doprint(Piecewise((2, Le(x, 0)),
                        (3, Gt(x, 0)), evaluate=False)) == '((2) if (x <= 0) else'\
                                                        ' (3) if (x > 0) else None)'
    
    # 测试符号函数的输出
    assert prntr.doprint(sign(x)) == '(0.0 if x == 0 else math.copysign(1, x))'
    
    # 测试 IndexedBase 对象的输出
    assert prntr.doprint(p[0, 1]) == 'p[0, 1]'
    
    # 测试 KroneckerDelta 函数的输出
    assert prntr.doprint(KroneckerDelta(x,y)) == '(1 if x == y else 0)'
    
    # 测试元组和列表的输出
    assert prntr.doprint((2,3)) == "(2, 3)"
    assert prntr.doprint([2,3]) == "[2, 3]"
    
    # 测试 Min 函数的输出
    assert prntr.doprint(Min(x, y)) == "min(x, y)"
    # 使用 assert 语句验证 prntr.doprint(Max(x, y)) 的输出是否等于字符串 "max(x, y)"
    assert prntr.doprint(Max(x, y)) == "max(x, y)"
def test_PythonCodePrinter_standard():
    # 创建 PythonCodePrinter 实例
    prntr = PythonCodePrinter()

    # 断言标准属性为 'python3'
    assert prntr.standard == 'python3'

    # 使用 lambda 表达式断言传入无效参数 'python4' 时会引发 ValueError 异常
    raises(ValueError, lambda: PythonCodePrinter({'standard':'python4'}))


def test_MpmathPrinter():
    # 创建 MpmathPrinter 实例
    p = MpmathPrinter()

    # 测试对 sympy.sign(x) 函数的打印
    assert p.doprint(sign(x)) == 'mpmath.sign(x)'
    
    # 测试对 sympy.Rational(1, 2) 的打印
    assert p.doprint(Rational(1, 2)) == 'mpmath.mpf(1)/mpmath.mpf(2)'

    # 测试常量的打印
    assert p.doprint(S.Exp1) == 'mpmath.e'
    assert p.doprint(S.Pi) == 'mpmath.pi'
    assert p.doprint(S.GoldenRatio) == 'mpmath.phi'
    assert p.doprint(S.EulerGamma) == 'mpmath.euler'
    assert p.doprint(S.NaN) == 'mpmath.nan'
    assert p.doprint(S.Infinity) == 'mpmath.inf'
    assert p.doprint(S.NegativeInfinity) == 'mpmath.ninf'

    # 测试对 mpmath.loggamma(x) 函数的打印
    assert p.doprint(loggamma(x)) == 'mpmath.loggamma(x)'


def test_NumPyPrinter():
    # 导入所需模块和类
    from sympy.core.function import Lambda
    from sympy.matrices.expressions.adjoint import Adjoint
    from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix, DiagonalOf)
    from sympy.matrices.expressions.funcmatrix import FunctionMatrix
    from sympy.matrices.expressions.hadamard import HadamardProduct
    from sympy.matrices.expressions.kronecker import KroneckerProduct
    from sympy.matrices.expressions.special import (OneMatrix, ZeroMatrix)
    from sympy.abc import a, b

    # 创建 NumPyPrinter 实例
    p = NumPyPrinter()

    # 测试对 sympy.sign(x) 函数的打印
    assert p.doprint(sign(x)) == 'numpy.sign(x)'
    
    # 创建符号矩阵实例 A, B, C, D
    A = MatrixSymbol("A", 2, 2)
    B = MatrixSymbol("B", 2, 2)
    C = MatrixSymbol("C", 1, 5)
    D = MatrixSymbol("D", 3, 4)

    # 测试矩阵求逆和矩阵幂的打印
    assert p.doprint(A**(-1)) == "numpy.linalg.inv(A)"
    assert p.doprint(A**5) == "numpy.linalg.matrix_power(A, 5)"
    
    # 测试单位矩阵的打印
    assert p.doprint(Identity(3)) == "numpy.eye(3)"

    # 创建符号向量实例 u, v
    u = MatrixSymbol('x', 2, 1)
    v = MatrixSymbol('y', 2, 1)

    # 测试矩阵求解和向量加法的打印
    assert p.doprint(MatrixSolve(A, u)) == 'numpy.linalg.solve(A, x)'
    assert p.doprint(MatrixSolve(A, u) + v) == 'numpy.linalg.solve(A, x) + y'

    # 测试零矩阵和全一矩阵的打印
    assert p.doprint(ZeroMatrix(2, 3)) == "numpy.zeros((2, 3))"
    assert p.doprint(OneMatrix(2, 3)) == "numpy.ones((2, 3))"

    # 测试函数矩阵的打印
    assert p.doprint(FunctionMatrix(4, 5, Lambda((a, b), a + b))) == \
        "numpy.fromfunction(lambda a, b: a + b, (4, 5))"

    # 测试哈达玛积和克罗内克积的打印
    assert p.doprint(HadamardProduct(A, B)) == "numpy.multiply(A, B)"
    assert p.doprint(KroneckerProduct(A, B)) == "numpy.kron(A, B)"

    # 测试伴随矩阵、矩阵对角线、对角矩阵的打印
    assert p.doprint(Adjoint(A)) == "numpy.conjugate(numpy.transpose(A))"
    assert p.doprint(DiagonalOf(A)) == "numpy.reshape(numpy.diag(A), (-1, 1))"
    assert p.doprint(DiagMatrix(C)) == "numpy.diagflat(C)"
    assert p.doprint(DiagonalMatrix(D)) == "numpy.multiply(D, numpy.eye(3, 4))"

    # 处理负整数幂的打印错误
    assert p.doprint(x**-1) == 'x**(-1.0)'
    assert p.doprint(x**-2) == 'x**(-2.0)'

    # 测试对常量的打印
    expr = Pow(2, -1, evaluate=False)
    assert p.doprint(expr) == "2**(-1.0)"

    # 测试常量的打印
    assert p.doprint(S.Exp1) == 'numpy.e'
    assert p.doprint(S.Pi) == 'numpy.pi'
    assert p.doprint(S.EulerGamma) == 'numpy.euler_gamma'
    assert p.doprint(S.NaN) == 'numpy.nan'
    # 断言：确保对 SymPy 中正负无穷的打印输出正确
    assert p.doprint(S.Infinity) == 'numpy.inf'
    assert p.doprint(S.NegativeInfinity) == '-numpy.inf'

    # 断言：验证函数重写操作符优先级修复
    assert p.doprint(sec(x)**2) == '(numpy.cos(x)**(-1.0))**2'
def test_issue_18770():
    # 导入 numpy 模块，并检查是否成功导入
    numpy = import_module('numpy')
    if not numpy:
        # 如果 numpy 模块未安装，则跳过测试并给出相应的提示信息
        skip("numpy not installed.")

    # 导入 sympy 库中的特定函数和类
    from sympy.functions.elementary.miscellaneous import (Max, Min)
    from sympy.utilities.lambdify import lambdify

    # 创建一个表达式 expr1，表示最小值函数的组合
    expr1 = Min(0.1*x + 3, x + 1, 0.5*x + 1)
    # 使用 lambdify 将表达式转换为一个可以使用 numpy 数组作为输入的函数
    func = lambdify(x, expr1, "numpy")
    # 断言函数在给定输入下的输出与预期值相匹配
    assert (func(numpy.linspace(0, 3, 3)) == [1.0, 1.75, 2.5 ]).all()
    assert func(4) == 3

    # 更新 expr1 为另一个表达式，表示最大值函数的组合
    expr1 = Max(x**2, x**3)
    # 使用 lambdify 创建新的函数
    func = lambdify(x, expr1, "numpy")
    # 断言函数在给定输入下的输出与预期值相匹配
    assert (func(numpy.linspace(-1, 2, 4)) == [1, 0, 1, 8] ).all()
    assert func(4) == 64


def test_SciPyPrinter():
    # 创建 SciPyPrinter 的实例对象 p
    p = SciPyPrinter()
    # 创建一个表达式 expr，表示 acos(x)
    expr = acos(x)
    # 断言 module_imports 中不包含 'numpy' 的导入
    assert 'numpy' not in p.module_imports
    # 断言 doprint 方法输出的结果与预期相符
    assert p.doprint(expr) == 'numpy.arccos(x)'
    # 断言 module_imports 中包含 'numpy' 的导入
    assert 'numpy' in p.module_imports
    # 断言 module_imports 中不包含以 'scipy' 开头的导入
    assert not any(m.startswith('scipy') for m in p.module_imports)
    
    # 创建一个稀疏矩阵 smat
    smat = SparseMatrix(2, 5, {(0, 1): 3})
    # 断言 doprint 方法输出的结果与预期相符
    assert p.doprint(smat) == \
        'scipy.sparse.coo_matrix(([3], ([0], [1])), shape=(2, 5))'
    # 断言 module_imports 中包含 'scipy.sparse' 的导入
    assert 'scipy.sparse' in p.module_imports

    # 断言 doprint 方法输出 S.GoldenRatio 的结果与预期相符
    assert p.doprint(S.GoldenRatio) == 'scipy.constants.golden_ratio'
    # 断言 doprint 方法输出 S.Pi 的结果与预期相符
    assert p.doprint(S.Pi) == 'scipy.constants.pi'
    # 断言 doprint 方法输出 S.Exp1 的结果与预期相符
    assert p.doprint(S.Exp1) == 'numpy.e'


def test_pycode_reserved_words():
    # 创建两个符号 s1 和 s2
    s1, s2 = symbols('if else')
    # 使用 lambda 函数和 raises 函数检查保留字错误
    raises(ValueError, lambda: pycode(s1 + s2, error_on_reserved=True))
    # 对 s1 + s2 进行转换为 Python 代码的断言
    py_str = pycode(s1 + s2)
    assert py_str in ('else_ + if_', 'if_ + else_')


def test_issue_20762():
    # 确保 pycode 能够去除带有花括号的下标变量
    a_b, b, a_11 = symbols('a_{b} b a_{11}')
    # 创建一个表达式 expr，表示 a_b * b
    expr = a_b * b
    # 断言 pycode 方法输出的结果与预期相符
    assert pycode(expr) == 'a_b*b'
    # 更新 expr，表示 a_11 * b
    expr = a_11 * b
    # 断言 pycode 方法输出的结果与预期相符
    assert pycode(expr) == 'a_11*b'


def test_sqrt():
    # 创建 PythonCodePrinter 的实例对象 prntr
    prntr = PythonCodePrinter()
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=False) == 'math.sqrt(x)'
    # 断言 _print_Pow 方法输出 1/sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(1/sqrt(x), rational=False) == '1/math.sqrt(x)'

    # 创建 PythonCodePrinter 的实例对象 prntr，并传入特定参数
    prntr = PythonCodePrinter({'standard': 'python3'})
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'
    # 断言 _print_Pow 方法输出 1/sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(1/sqrt(x), rational=True) == 'x**(-1/2)'

    # 创建 MpmathPrinter 的实例对象 prntr
    prntr = MpmathPrinter()
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=False) == 'mpmath.sqrt(x)'
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=True) == "x**(mpmath.mpf(1)/mpmath.mpf(2))"

    # 创建 NumPyPrinter 的实例对象 prntr
    prntr = NumPyPrinter()
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=False) == 'numpy.sqrt(x)'
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

    # 创建 SciPyPrinter 的实例对象 prntr
    prntr = SciPyPrinter()
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=False) == 'numpy.sqrt(x)'
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

    # 创建 SymPyPrinter 的实例对象 prntr
    prntr = SymPyPrinter()
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=False) == 'sympy.sqrt(x)'
    # 断言 _print_Pow 方法输出 sqrt(x) 的结果与预期相符
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'
    # 断言：验证表达式在使用 prntr.doprint(expr) 函数处理后的输出是否等于 'numpy.mod(x, 1)'
    assert prntr.doprint(expr) == 'numpy.mod(x, 1)'
    
    # 创建 PythonCodePrinter 的实例 prntr
    prntr = PythonCodePrinter()
    # 断言：验证表达式在使用 prntr.doprint(expr) 函数处理后的输出是否等于 'x % 1'
    assert prntr.doprint(expr) == 'x % 1'
    
    # 创建 MpmathPrinter 的实例 prntr
    prntr = MpmathPrinter()
    # 断言：验证表达式在使用 prntr.doprint(expr) 函数处理后的输出是否等于 'mpmath.frac(x)'
    assert prntr.doprint(expr) == 'mpmath.frac(x)'
    
    # 创建 SymPyPrinter 的实例 prntr
    prntr = SymPyPrinter()
    # 断言：验证表达式在使用 prntr.doprint(expr) 函数处理后的输出是否等于 'sympy.functions.elementary.integers.frac(x)'
    assert prntr.doprint(expr) == 'sympy.functions.elementary.integers.frac(x)'
class CustomPrintedObject(Expr):
    # 定义一个自定义的打印对象，继承自Expr类

    def _numpycode(self, printer):
        # 定义一个返回字符串'numpy'的方法，用于NumPy打印器
        return 'numpy'

    def _mpmathcode(self, printer):
        # 定义一个返回字符串'mpmath'的方法，用于Mpmath打印器
        return 'mpmath'


def test_printmethod():
    # 测试打印方法的功能

    obj = CustomPrintedObject()
    # 创建一个CustomPrintedObject对象

    assert NumPyPrinter().doprint(obj) == 'numpy'
    # 使用NumPy打印器打印obj对象，预期结果为'numpy'

    assert MpmathPrinter().doprint(obj) == 'mpmath'
    # 使用Mpmath打印器打印obj对象，预期结果为'mpmath'


def test_codegen_ast_nodes():
    # 测试代码生成AST节点的功能

    assert pycode(none) == 'None'
    # 使用pycode函数处理none对象，预期结果为字符串'None'


def test_issue_14283():
    # 测试14283号问题

    prntr = PythonCodePrinter()
    # 创建一个PythonCodePrinter对象prntr

    assert prntr.doprint(zoo) == "math.nan"
    # 使用prntr打印zoo对象，预期结果为字符串"math.nan"

    assert prntr.doprint(-oo) == "float('-inf')"
    # 使用prntr打印-oo对象，预期结果为字符串"float('-inf')"


def test_NumPyPrinter_print_seq():
    # 测试NumPy打印器的序列打印功能

    n = NumPyPrinter()
    # 创建一个NumPyPrinter对象n

    assert n._print_seq(range(2)) == '(0, 1,)'
    # 使用n对象的_print_seq方法打印range(2)，预期结果为字符串'(0, 1,)'


def test_issue_16535_16536():
    # 测试16535和16536号问题

    from sympy.functions.special.gamma_functions import (lowergamma, uppergamma)
    # 导入lowergamma和uppergamma函数

    a = symbols('a')
    # 创建符号'a'

    expr1 = lowergamma(a, x)
    # 创建lowergamma(a, x)表达式对象expr1

    expr2 = uppergamma(a, x)
    # 创建uppergamma(a, x)表达式对象expr2

    prntr = SciPyPrinter()
    # 创建一个SciPyPrinter对象prntr

    assert prntr.doprint(expr1) == 'scipy.special.gamma(a)*scipy.special.gammainc(a, x)'
    # 使用prntr打印expr1对象，预期结果为字符串'scipy.special.gamma(a)*scipy.special.gammainc(a, x)'

    assert prntr.doprint(expr2) == 'scipy.special.gamma(a)*scipy.special.gammaincc(a, x)'
    # 使用prntr打印expr2对象，预期结果为字符串'scipy.special.gamma(a)*scipy.special.gammaincc(a, x)'

    p_numpy = NumPyPrinter()
    # 创建一个NumPyPrinter对象p_numpy

    p_pycode = PythonCodePrinter({'strict': False})
    # 创建一个PythonCodePrinter对象p_pycode，设置严格模式为False

    for expr in [expr1, expr2]:
        with raises(NotImplementedError):
            p_numpy.doprint(expr1)
        # 对于expr1，预期抛出NotImplementedError异常

        assert "Not supported" in p_pycode.doprint(expr)
        # 对于expr，使用p_pycode打印，预期结果包含字符串"Not supported"


def test_Integral():
    # 测试积分函数的功能

    from sympy.functions.elementary.exponential import exp
    # 导入指数函数exp

    from sympy.integrals.integrals import Integral
    # 导入积分函数Integral

    single = Integral(exp(-x), (x, 0, oo))
    # 创建指数函数exp(-x)的单重积分对象single

    double = Integral(x**2*exp(x*y), (x, -z, z), (y, 0, z))
    # 创建包含两个变量的双重积分对象double

    indefinite = Integral(x**2, x)
    # 创建不定积分对象indefinite

    evaluateat = Integral(x**2, (x, 1))
    # 创建在特定点评估的积分对象evaluateat

    prntr = SciPyPrinter()
    # 创建一个SciPyPrinter对象prntr

    assert prntr.doprint(single) == 'scipy.integrate.quad(lambda x: numpy.exp(-x), 0, numpy.inf)[0]'
    # 使用prntr打印single对象，预期结果为字符串'scipy.integrate.quad(lambda x: numpy.exp(-x), 0, numpy.inf)[0]'

    assert prntr.doprint(double) == 'scipy.integrate.nquad(lambda x, y: x**2*numpy.exp(x*y), ((-z, z), (0, z)))[0]'
    # 使用prntr打印double对象，预期结果为字符串'scipy.integrate.nquad(lambda x, y: x**2*numpy.exp(x*y), ((-z, z), (0, z)))[0]'

    raises(NotImplementedError, lambda: prntr.doprint(indefinite))
    # 使用prntr打印indefinite对象，预期抛出NotImplementedError异常

    raises(NotImplementedError, lambda: prntr.doprint(evaluateat))
    # 使用prntr打印evaluateat对象，预期抛出NotImplementedError异常

    prntr = MpmathPrinter()
    # 创建一个MpmathPrinter对象prntr

    assert prntr.doprint(single) == 'mpmath.quad(lambda x: mpmath.exp(-x), (0, mpmath.inf))'
    # 使用prntr打印single对象，预期结果为字符串'mpmath.quad(lambda x: mpmath.exp(-x), (0, mpmath.inf))'

    assert prntr.doprint(double) == 'mpmath.quad(lambda x, y: x**2*mpmath.exp(x*y), (-z, z), (0, z))'
    # 使用prntr打印double对象，预期结果为字符串'mpmath.quad(lambda x, y: x**2*mpmath.exp(x*y), (-z, z), (0, z))'

    raises(NotImplementedError, lambda: prntr.doprint(indefinite))
    # 使用prntr打印indefinite对象，预期抛出NotImplementedError异常

    raises(NotImplementedError, lambda: prntr.doprint(evaluateat))
    # 使用prntr打印evaluateat对象，预期抛出NotImplementedError异常


def test_fresnel_integrals():
    # 测试Fresnel积分函数的功能

    from sympy.functions.special.error_functions import (fresnelc, fresnels)
    # 导入Fresnel余弦积分和Fresnel正弦积分函数

    expr1 = fresnelc(x)
    # 创建Fresnel余弦积分对象expr1

    expr2 = fresnels(x)
    # 创建Fresnel正弦积分对象expr2

    prntr = SciPyPrinter()
    # 创建一个SciPyPrinter对象prntr

    assert prntr.doprint(expr1) == 'scipy.special.fresnel(x)[1]'
    # 使用prntr打印expr1对象，预期结果为字符串'scipy.special.fresnel(x)[1]'

    assert prntr.doprint(expr2) == 'scipy.special.fresnel(x)[0]'
    # 使用prntr打印expr2对象，预期结果为字符串'scipy.special.fresnel(x)[0]'

    p_numpy = NumPyPrinter()
    # 创建一个NumPyPrinter对象p_numpy

    p_pycode = PythonCodePrinter()
    # 创建一个PythonCodePrinter对象p_pycode

    p_mpmath = MpmathPrinter()
    # 创建一个MpmathPrinter对象p_mpmath

    for expr in [expr1, expr2]:
        with raises(NotImplementedError):
            p_numpy.doprint(expr)
        # 对于expr，使用p_numpy打印，预期抛出NotImplementedError异常

        with raises(NotImplementedError):
            p_pycode.doprint(expr)
        # 对于expr，使用p_pycode打印，预期抛出NotImplementedError异常

    assert p_mpmath.doprint(expr1) == '
    # 断言检查表达式 `p_mpmath.doprint(expr2)` 的输出是否等于字符串 `'mpmath.fresnels(x)'`
    assert p_mpmath.doprint(expr2) == 'mpmath.fresnels(x)'
def test_beta():
    # 导入 sympy 库中的 beta 函数
    from sympy.functions.special.beta_functions import beta

    # 计算 beta 函数表达式
    expr = beta(x, y)

    # 创建 SciPyPrinter 实例
    prntr = SciPyPrinter()
    # 断言 SciPyPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr) == 'scipy.special.beta(x, y)'

    # 创建 NumPyPrinter 实例
    prntr = NumPyPrinter()
    # 断言 NumPyPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    # 创建 PythonCodePrinter 实例
    prntr = PythonCodePrinter()
    # 断言 PythonCodePrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    # 创建允许未知函数的 PythonCodePrinter 实例
    prntr = PythonCodePrinter({'allow_unknown_functions': True})
    # 断言 PythonCodePrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    # 创建 MpmathPrinter 实例
    prntr = MpmathPrinter()
    # 断言 MpmathPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr) == 'mpmath.beta(x, y)'

def test_airy():
    # 导入 sympy 库中的 airyai 和 airybi 函数
    from sympy.functions.special.bessel import (airyai, airybi)

    # 计算 airyai 和 airybi 函数的表达式
    expr1 = airyai(x)
    expr2 = airybi(x)

    # 创建 SciPyPrinter 实例
    prntr = SciPyPrinter()
    # 断言 SciPyPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[0]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[2]'

    # 创建 NumPyPrinter 实例，允许宽松模式（strict=False）
    prntr = NumPyPrinter({'strict': False})
    # 断言输出包含 "Not supported" 字符串，因为 NumPyPrinter 不支持 airyai 和 airybi 函数
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

    # 创建 PythonCodePrinter 实例，允许宽松模式（strict=False）
    prntr = PythonCodePrinter({'strict': False})
    # 断言输出包含 "Not supported" 字符串，因为 PythonCodePrinter 不支持 airyai 和 airybi 函数
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

def test_airy_prime():
    # 导入 sympy 库中的 airyaiprime 和 airybiprime 函数
    from sympy.functions.special.bessel import (airyaiprime, airybiprime)

    # 计算 airyaiprime 和 airybiprime 函数的表达式
    expr1 = airyaiprime(x)
    expr2 = airybiprime(x)

    # 创建 SciPyPrinter 实例
    prntr = SciPyPrinter()
    # 断言 SciPyPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[1]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[3]'

    # 创建 NumPyPrinter 实例，允许宽松模式（strict=False）
    prntr = NumPyPrinter({'strict': False})
    # 断言输出包含 "Not supported" 字符串，因为 NumPyPrinter 不支持 airyaiprime 和 airybiprime 函数
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

    # 创建 PythonCodePrinter 实例，允许宽松模式（strict=False）
    prntr = PythonCodePrinter({'strict': False})
    # 断言输出包含 "Not supported" 字符串，因为 PythonCodePrinter 不支持 airyaiprime 和 airybiprime 函数
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

def test_numerical_accuracy_functions():
    # 创建 SciPyPrinter 实例
    prntr = SciPyPrinter()
    # 断言 SciPyPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(expm1(x)) == 'numpy.expm1(x)'
    assert prntr.doprint(log1p(x)) == 'numpy.log1p(x)'
    assert prntr.doprint(cosm1(x)) == 'scipy.special.cosm1(x)'

def test_array_printer():
    # 定义 ArraySymbol 和 IndexedBase 对象
    A = ArraySymbol('A', (4,4,6,6,6))
    I = IndexedBase('I')
    # 定义索引对象 i, j, k
    i,j,k = Idx('i', (0,1)), Idx('j', (2,3)), Idx('k', (4,5))

    # 创建 NumPyPrinter 实例
    prntr = NumPyPrinter()
    # 断言 NumPyPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(ZeroArray(5)) == 'numpy.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'numpy.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2,3])) == 'numpy.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2,3,4])) == 'numpy.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) == 'numpy.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'numpy.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) == 'I = I'

    # 创建 TensorflowPrinter 实例
    prntr = TensorflowPrinter()
    # 断言 TensorflowPrinter 输出的表达式字符串是否正确
    assert prntr.doprint(ZeroArray(5)) == 'tensorflow.zeros((5,))'
    # 断言语句，验证 prntr.doprint(OneArray(5)) 的输出是否等于 'tensorflow.ones((5,))'
    assert prntr.doprint(OneArray(5)) == 'tensorflow.ones((5,))'
    
    # 断言语句，验证 prntr.doprint(ArrayContraction(A, [2,3])) 的输出是否等于 'tensorflow.linalg.einsum("abccd->abd", A)'
    assert prntr.doprint(ArrayContraction(A, [2,3])) == 'tensorflow.linalg.einsum("abccd->abd", A)'
    
    # 断言语句，验证 prntr.doprint(I) 的输出是否等于 'I'
    assert prntr.doprint(I) == 'I'
    
    # 断言语句，验证 prntr.doprint(ArrayDiagonal(A, [2,3,4])) 的输出是否等于 'tensorflow.linalg.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [2,3,4])) == 'tensorflow.linalg.einsum("abccc->abc", A)'
    
    # 断言语句，验证 prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) 的输出是否等于 'tensorflow.linalg.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) == 'tensorflow.linalg.einsum("aabbc->cab", A)'
    
    # 断言语句，验证 prntr.doprint(ArrayContraction(A, [2], [3])) 的输出是否等于 'tensorflow.linalg.einsum("abcde->abe", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'tensorflow.linalg.einsum("abcde->abe", A)'
    
    # 断言语句，验证 prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) 的输出是否等于 'I = I'
    assert prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) == 'I = I'
```