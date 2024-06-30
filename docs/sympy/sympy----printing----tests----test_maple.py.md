# `D:\src\scipysrc\sympy\sympy\printing\tests\test_maple.py`

```
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
                        Tuple, Symbol, Eq, Ne, Le, Lt, Gt, Ge)
# 导入 sympy.core 中的多个符号和函数类

from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
# 导入 sympy.core 中的特殊常数和 Lambda、Mul、Pow 函数

from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
# 导入 sympy.functions 中的特殊函数

from sympy.testing.pytest import raises
# 导入 sympy.testing.pytest 中的 raises 函数

from sympy.utilities.lambdify import implemented_function
# 导入 sympy.utilities.lambdify 中的 implemented_function 函数

from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
                            HadamardProduct, SparseMatrix)
# 导入 sympy.matrices 中的矩阵相关类

from sympy.functions.special.bessel import besseli
# 导入 sympy.functions.special.bessel 中的 besseli 函数

from sympy.printing.maple import maple_code
# 导入 sympy.printing.maple 中的 maple_code 函数

x, y, z = symbols('x,y,z')
# 创建符号 x, y, z

def test_Integer():
    assert maple_code(Integer(67)) == "67"
    assert maple_code(Integer(-1)) == "-1"
    # 测试整数的 maple_code 输出

def test_Rational():
    assert maple_code(Rational(3, 7)) == "3/7"
    assert maple_code(Rational(18, 9)) == "2"
    assert maple_code(Rational(3, -7)) == "-3/7"
    assert maple_code(Rational(-3, -7)) == "3/7"
    assert maple_code(x + Rational(3, 7)) == "x + 3/7"
    assert maple_code(Rational(3, 7) * x) == '(3/7)*x'
    # 测试有理数的 maple_code 输出

def test_Relational():
    assert maple_code(Eq(x, y)) == "x = y"
    assert maple_code(Ne(x, y)) == "x <> y"
    assert maple_code(Le(x, y)) == "x <= y"
    assert maple_code(Lt(x, y)) == "x < y"
    assert maple_code(Gt(x, y)) == "x > y"
    assert maple_code(Ge(x, y)) == "x >= y"
    # 测试关系运算符的 maple_code 输出

def test_Function():
    assert maple_code(sin(x) ** cos(x)) == "sin(x)^cos(x)"
    assert maple_code(abs(x)) == "abs(x)"
    assert maple_code(ceiling(x)) == "ceil(x)"
    # 测试函数的 maple_code 输出

def test_Pow():
    assert maple_code(x ** 3) == "x^3"
    assert maple_code(x ** (y ** 3)) == "x^(y^3)"

    assert maple_code((x ** 3) ** y) == "(x^3)^y"
    assert maple_code(x ** Rational(2, 3)) == 'x^(2/3)'

    g = implemented_function('g', Lambda(x, 2 * x))
    assert maple_code(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == \
           "(3.5*2*x)^(-x + y^x)/(x^2 + y)"
    # 测试幂函数的 maple_code 输出，包括 Lambda 函数的处理

    # For issue 14160
    assert maple_code(Mul(-2, x, Pow(Mul(y, y, evaluate=False), -1, evaluate=False),
                          evaluate=False)) == '-2*x/(y*y)'
    # 针对特定问题测试的例子

def test_basic_ops():
    assert maple_code(x * y) == "x*y"
    assert maple_code(x + y) == "x + y"
    assert maple_code(x - y) == "x - y"
    assert maple_code(-x) == "-x"
    # 测试基本运算的 maple_code 输出

def test_1_over_x_and_sqrt():
    assert maple_code(1 / x) == '1/x'
    assert maple_code(x ** -1) == maple_code(x ** -1.0) == '1/x'
    assert maple_code(1 / sqrt(x)) == '1/sqrt(x)'
    assert maple_code(x ** -S.Half) == maple_code(x ** -0.5) == '1/sqrt(x)'
    assert maple_code(sqrt(x)) == 'sqrt(x)'
    assert maple_code(x ** S.Half) == maple_code(x ** 0.5) == 'sqrt(x)'
    assert maple_code(1 / pi) == '1/Pi'
    assert maple_code(pi ** -1) == maple_code(pi ** -1.0) == '1/Pi'
    assert maple_code(pi ** -0.5) == '1/sqrt(Pi)'
    # 测试倒数和平方根的 maple_code 输出
    assert maple_code(3 * x) == "3*x"
    # 断言：调用 maple_code 函数，验证表达式 3 * x 返回字符串 "3*x"

    assert maple_code(pi * x) == "Pi*x"
    # 断言：调用 maple_code 函数，验证表达式 π * x 返回字符串 "Pi*x"

    assert maple_code(3 / x) == "3/x"
    # 断言：调用 maple_code 函数，验证表达式 3 / x 返回字符串 "3/x"

    assert maple_code(pi / x) == "Pi/x"
    # 断言：调用 maple_code 函数，验证表达式 π / x 返回字符串 "Pi/x"

    assert maple_code(x / 3) == '(1/3)*x'
    # 断言：调用 maple_code 函数，验证表达式 x / 3 返回字符串 '(1/3)*x'

    assert maple_code(x / pi) == "x/Pi"
    # 断言：调用 maple_code 函数，验证表达式 x / π 返回字符串 "x/Pi"

    assert maple_code(x * y) == "x*y"
    # 断言：调用 maple_code 函数，验证表达式 x * y 返回字符串 "x*y"

    assert maple_code(3 * x * y) == "3*x*y"
    # 断言：调用 maple_code 函数，验证表达式 3 * x * y 返回字符串 "3*x*y"

    assert maple_code(3 * pi * x * y) == "3*Pi*x*y"
    # 断言：调用 maple_code 函数，验证表达式 3 * π * x * y 返回字符串 "3*Pi*x*y"

    assert maple_code(x / y) == "x/y"
    # 断言：调用 maple_code 函数，验证表达式 x / y 返回字符串 "x/y"

    assert maple_code(3 * x / y) == "3*x/y"
    # 断言：调用 maple_code 函数，验证表达式 3 * x / y 返回字符串 "3*x/y"

    assert maple_code(x * y / z) == "x*y/z"
    # 断言：调用 maple_code 函数，验证表达式 x * y / z 返回字符串 "x*y/z"

    assert maple_code(x / y * z) == "x*z/y"
    # 断言：调用 maple_code 函数，验证表达式 x / y * z 返回字符串 "x*z/y"

    assert maple_code(1 / x / y) == "1/(x*y)"
    # 断言：调用 maple_code 函数，验证表达式 1 / x / y 返回字符串 "1/(x*y)"

    assert maple_code(2 * pi * x / y / z) == "2*Pi*x/(y*z)"
    # 断言：调用 maple_code 函数，验证表达式 2 * π * x / y / z 返回字符串 "2*Pi*x/(y*z)"

    assert maple_code(3 * pi / x) == "3*Pi/x"
    # 断言：调用 maple_code 函数，验证表达式 3 * π / x 返回字符串 "3*Pi/x"

    assert maple_code(S(3) / 5) == "3/5"
    # 断言：调用 maple_code 函数，验证有理数 3/5 返回字符串 "3/5"

    assert maple_code(S(3) / 5 * x) == '(3/5)*x'
    # 断言：调用 maple_code 函数，验证表达式 3/5 * x 返回字符串 '(3/5)*x'

    assert maple_code(x / y / z) == "x/(y*z)"
    # 断言：调用 maple_code 函数，验证表达式 x / y / z 返回字符串 "x/(y*z)"

    assert maple_code((x + y) / z) == "(x + y)/z"
    # 断言：调用 maple_code 函数，验证表达式 (x + y) / z 返回字符串 "(x + y)/z"

    assert maple_code((x + y) / (z + x)) == "(x + y)/(x + z)"
    # 断言：调用 maple_code 函数，验证表达式 (x + y) / (z + x) 返回字符串 "(x + y)/(x + z)"

    assert maple_code((x + y) / EulerGamma) == '(x + y)/gamma'
    # 断言：调用 maple_code 函数，验证表达式 (x + y) / EulerGamma 返回字符串 '(x + y)/gamma'

    assert maple_code(x / 3 / pi) == '(1/3)*x/Pi'
    # 断言：调用 maple_code 函数，验证表达式 x / 3 / pi 返回字符串 '(1/3)*x/Pi'

    assert maple_code(S(3) / 5 * x * y / pi) == '(3/5)*x*y/Pi'
    # 断言：调用 maple_code 函数，验证表达式 3/5 * x * y / pi 返回字符串 '(3/5)*x*y/Pi'
# 测试混合数字和符号幂运算的函数
def test_mix_number_pow_symbols():
    # 断言：对 pi 的立方进行 maple_code 处理，预期结果是 'Pi^3'
    assert maple_code(pi ** 3) == 'Pi^3'
    # 断言：对 x 的平方进行 maple_code 处理，预期结果是 'x^2'
    assert maple_code(x ** 2) == 'x^2'

    # 断言：对 x 的 pi 的立方次幂进行 maple_code 处理，预期结果是 'x^(Pi^3)'
    assert maple_code(x ** (pi ** 3)) == 'x^(Pi^3)'
    # 断言：对 x 和 y 的幂运算进行 maple_code 处理，预期结果是 'x^y'
    assert maple_code(x ** y) == 'x^y'

    # 断言：对 x 和 y 的 z 次幂进行 maple_code 处理，预期结果是 'x^(y^z)'
    assert maple_code(x ** (y ** z)) == 'x^(y^z)'
    # 断言：对 x 的 y 次幂再 z 次幂进行 maple_code 处理，预期结果是 '(x^y)^z'
    assert maple_code((x ** y) ** z) == '(x^y)^z'


# 测试复数
def test_imag():
    # 定义复数 I
    I = S('I')
    # 断言：对复数 I 进行 maple_code 处理，预期结果是 "I"
    assert maple_code(I) == "I"
    # 断言：对 5 乘以复数 I 进行 maple_code 处理，预期结果是 "5*I"
    assert maple_code(5 * I) == "5*I"

    # 断言：对 (3/2) 乘以复数 I 进行 maple_code 处理，预期结果是 "(3/2)*I"
    assert maple_code((S(3) / 2) * I) == "(3/2)*I"
    # 断言：对 3 加上 4 乘以复数 I 进行 maple_code 处理，预期结果是 "3 + 4*I"
    assert maple_code(3 + 4 * I) == "3 + 4*I"


# 测试常数
def test_constants():
    # 断言：对 pi 进行 maple_code 处理，预期结果是 "Pi"
    assert maple_code(pi) == "Pi"
    # 断言：对 oo（正无穷大）进行 maple_code 处理，预期结果是 "infinity"
    assert maple_code(oo) == "infinity"
    # 断言：对 -oo（负无穷大）进行 maple_code 处理，预期结果是 "-infinity"
    assert maple_code(-oo) == "-infinity"
    # 断言：对 S.NegativeInfinity 进行 maple_code 处理，预期结果是 "-infinity"
    assert maple_code(S.NegativeInfinity) == "-infinity"
    # 断言：对 S.NaN 进行 maple_code 处理，预期结果是 "undefined"
    assert maple_code(S.NaN) == "undefined"
    # 断言：对 S.Exp1（自然对数的底 e）进行 maple_code 处理，预期结果是 "exp(1)"
    assert maple_code(S.Exp1) == "exp(1)"
    # 断言：对 exp(1) 进行 maple_code 处理，预期结果是 "exp(1)"
    assert maple_code(exp(1)) == "exp(1)"


# 测试其他常数
def test_constants_other():
    # 断言：对 2 乘以 GoldenRatio 进行 maple_code 处理，预期结果是 '2*(1/2 + (1/2)*sqrt(5))'
    assert maple_code(2 * GoldenRatio) == '2*(1/2 + (1/2)*sqrt(5))'
    # 断言：对 2 乘以 Catalan 进行 maple_code 处理，预期结果是 '2*Catalan'
    assert maple_code(2 * Catalan) == '2*Catalan'
    # 断言：对 2 乘以 EulerGamma 进行 maple_code 处理，预期结果是 "2*gamma"
    assert maple_code(2 * EulerGamma) == "2*gamma"


# 测试布尔运算
def test_boolean():
    # 断言：对 x 和 y 的与运算进行 maple_code 处理，预期结果是 "x and y"
    assert maple_code(x & y) == "x and y"
    # 断言：对 x 和 y 的或运算进行 maple_code 处理，预期结果是 "x or y"
    assert maple_code(x | y) == "x or y"
    # 断言：对 x 的非运算进行 maple_code 处理，预期结果是 "not x"
    assert maple_code(~x) == "not x"
    # 断言：对 x、y 和 z 的多重与运算进行 maple_code 处理，预期结果是 "x and y and z"
    assert maple_code(x & y & z) == "x and y and z"
    # 断言：对 x、y 和 z 的多重或运算进行 maple_code 处理，预期结果是 "x or y or z"
    assert maple_code(x | y | z) == "x or y or z"
    # 断言：对 (x 和 y 的与运算) 或 z 的组合进行 maple_code 处理，预期结果是 "z or x and y"
    assert maple_code((x & y) | z) == "z or x and y"
    # 断言：对 (x 和 y 的或运算) 与 z 的组合进行 maple_code 处理，预期结果是 "z and (x or y)"
    assert maple_code((x | y) & z) == "z and (x or y)"


# 测试矩阵
def test_Matrices():
    # 断言：对包含单个元素 10 的 Matrix 进行 maple_code 处理，预期结果是 'Matrix([[10]], storage = rectangular)'
    assert maple_code(Matrix(1, 1, [10])) == \
           'Matrix([[10]], storage = rectangular)'

    # 创建一个复杂的 Matrix A
    A = Matrix([[1, sin(x / 2), abs(x)],
                [0, 1, pi],
                [0, exp(1), ceiling(x)]])
    # 预期的结果字符串
    expected = \
        'Matrix(' \
        '[[1, sin((1/2)*x), abs(x)],' \
        ' [0, 1, Pi],' \
        ' [0, exp(1), ceil(x)]], ' \
        'storage = rectangular)'
    # 断言：对 Matrix A 进行 maple_code 处理，预期结果是 expected 字符串
    assert maple_code(A) == expected

    # 断言：对 A 的第一列进行 maple_code 处理，预期结果是 'Matrix([[1], [0], [0]], storage = rectangular)'
    assert maple_code(A[:, 0]) == \
           'Matrix([[1], [0], [0]], storage = rectangular)'
    # 断言：对 A 的第一行进行 maple_code 处理，预期结果是 'Matrix([[1, sin((1/2)*x), abs(x)]], storage = rectangular)'
    assert maple_code(A[0, :]) == \
           'Matrix([[1, sin((1/2)*x), abs(x)]], storage = rectangular)'
    # 断言：对包含向量 [[x, x - y, -y]] 的 Matrix 进行 maple_code 处理，预期结果是 'Matrix([[x, x - y, -y]], storage = rectangular)'
    assert maple_code(Matrix([[x, x - y, -y]])) == \
           'Matrix([[x, x - y, -y]], storage = rectangular)'

    # 断言：对空的 0x0 Matrix 进行 maple_code 处理，预期结果是 'Matrix([], storage = rectangular)'
    assert maple_code(Matrix(0, 0, [])) == \
           'Matrix([], storage = rectangular)'
    # 断言：对空的 0x3 Matrix 进行 maple_code 处理，预期结果是 'Matrix([], storage = rectangular)'
    assert maple_code(Matrix(0, 3, [])) == \
           'Matrix([], storage = rectangular)'


# 测试稀疏矩阵
def test_SparseMatrices():
    # 断言：对 Identity(2) 的稀疏矩阵进行 maple_code 处理，预期结果是 'Matrix([[1, 0], [0
    # 定义预期的字符串，包含一个表示矩阵和其它选项的Maple代码的示例
    expected = \
        'Matrix([[1, sin(2/x), (3/5)*Pi/x], [1, 2, x*y]], ' \
        'storage = rectangular)'
    # 使用断言检查调用 maple_code 函数对矩阵 A 的输出是否与预期的字符串相匹配
    assert maple_code(A) == expected
# 定义一个函数用于测试 MatrixSymbol 的功能
def test_MatrixSymbol():
    # 创建一个整数符号 n
    n = Symbol('n', integer=True)
    # 创建一个 n x n 的矩阵符号 A
    A = MatrixSymbol('A', n, n)
    # 创建另一个 n x n 的矩阵符号 B
    B = MatrixSymbol('B', n, n)
    # 断言 A * B 的 Maple 代码等于 "A.B"
    assert maple_code(A * B) == "A.B"
    # 断言 B * A 的 Maple 代码等于 "B.A"
    assert maple_code(B * A) == "B.A"
    # 断言 2 * A * B 的 Maple 代码等于 "2*A.B"
    assert maple_code(2 * A * B) == "2*A.B"
    # 断言 B * 2 * A 的 Maple 代码等于 "2*B.A"
    assert maple_code(B * 2 * A) == "2*B.A"

    # 断言 A * (B + 3 * Identity(n)) 的 Maple 代码等于 "A.(3*Matrix(n, shape = identity) + B)"
    assert maple_code(
        A * (B + 3 * Identity(n))) == "A.(3*Matrix(n, shape = identity) + B)"

    # 断言 A ** (x ** 2) 的 Maple 代码等于 "MatrixPower(A, x^2)"
    assert maple_code(A ** (x ** 2)) == "MatrixPower(A, x^2)"
    # 断言 A ** 3 的 Maple 代码等于 "MatrixPower(A, 3)"
    assert maple_code(A ** 3) == "MatrixPower(A, 3)"
    # 断言 A ** (S.Half) 的 Maple 代码等于 "MatrixPower(A, 1/2)"
    assert maple_code(A ** (S.Half)) == "MatrixPower(A, 1/2)"


# 定义一个函数用于测试特殊矩阵的功能
def test_special_matrices():
    # 断言 6 * Identity(3) 的 Maple 代码等于 "6*Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = sparse)"
    assert maple_code(6 * Identity(3)) == "6*Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = sparse)"
    # 断言 Identity(x) 的 Maple 代码等于 'Matrix(x, shape = identity)'
    assert maple_code(Identity(x)) == 'Matrix(x, shape = identity)'


# 定义一个函数用于测试容器类型的功能
def test_containers():
    # 断言列表的 Maple 代码与其自身的字符串表示相等
    assert maple_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
           "[1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]"

    # 断言元组的 Maple 代码与其自身的字符串表示相等
    assert maple_code((1, 2, (3, 4))) == "[1, 2, [3, 4]]"
    # 断言单元素列表的 Maple 代码与其自身的字符串表示相等
    assert maple_code([1]) == "[1]"
    # 断言单元素元组的 Maple 代码与其自身的字符串表示相等
    assert maple_code((1,)) == "[1]"
    # 断言使用 Tuple 函数生成的 Maple 代码与其自身的字符串表示相等
    assert maple_code(Tuple(*[1, 2, 3])) == "[1, 2, 3]"
    # 断言包含表达式的元组的 Maple 代码与其自身的字符串表示相等
    assert maple_code((1, x * y, (3, x ** 2))) == "[1, x*y, [3, x^2]]"
    # 断言包含不同类型元素的元组的 Maple 代码与其自身的字符串表示相等
    assert maple_code((1, eye(3), Matrix(0, 0, []), [])) == \
           "[1, Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = rectangular), Matrix([], storage = rectangular), []]"


# 定义一个函数用于测试非内联 Maple 代码生成的功能
def test_maple_noninline():
    # 生成非内联的 Maple 代码，并断言其与预期结果相等
    source = maple_code((x + y)/Catalan, assign_to='me', inline=False)
    expected = "me := (x + y)/Catalan"
    assert source == expected


# 定义一个函数用于测试将矩阵赋值给 Maple 变量的功能
def test_maple_matrix_assign_to():
    # 创建一个矩阵 A，并生成其对应的 Maple 代码，并断言与预期结果相等
    A = Matrix([[1, 2, 3]])
    assert maple_code(A, assign_to='a') == "a := Matrix([[1, 2, 3]], storage = rectangular)"
    # 更新矩阵 A，并生成其对应的 Maple 代码，并断言与预期结果相等
    A = Matrix([[1, 2], [3, 4]])
    assert maple_code(A, assign_to='A') == "A := Matrix([[1, 2], [3, 4]], storage = rectangular)"


# 定义一个函数用于测试更多矩阵赋值给 Maple 变量的功能
def test_maple_matrix_assign_to_more():
    # 创建一个矩阵 A 和两个 MatrixSymbol B 和 C
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    # 断言将矩阵 A 赋给 MatrixSymbol B 生成的 Maple 代码与预期结果相等
    assert maple_code(A, assign_to=B) == "B := Matrix([[1, 2, 3]], storage = rectangular)"
    # 使用 Lambda 函数测试将矩阵 A 赋给非矩阵符号 x 时的异常情况
    raises(ValueError, lambda: maple_code(A, assign_to=x))
    # 使用 Lambda 函数测试将矩阵 A 赋给与其维度不匹配的 MatrixSymbol C 时的异常情况
    raises(ValueError, lambda: maple_code(A, assign_to=C))


# 定义一个函数用于测试 1x1 矩阵的 Maple 代码生成功能
def test_maple_matrix_1x1():
    # 创建一个 1x1 的矩阵 A，并生成其对应的 Maple 代码，并断言与预期结果相等
    A = Matrix([[3]])
    assert maple_code(A, assign_to='B') == "B := Matrix([[3]], storage = rectangular)"


# 定义一个函数用于测试矩阵元素的 Maple 代码生成功能
def test_maple_matrix_elements():
    # 创建一个矩阵 A 包含符号 x、数值 2 和符号表达式 x * y
    A = Matrix([[x, 2, x * y]])

    # 断言生成的 Maple 代码与预期的表达式相等
    assert maple_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == "x^2 + x*y + 2"
    
    # 创建一个 MatrixSymbol AA，它是一个 1x3 的矩阵符号
    AA = MatrixSymbol('AA', 1, 3)
    # 断言生成的 Maple 代码与预期的表达式相等
    assert maple_code(AA) == "AA"

    # 断言生成的 Maple 代码与预期的表达式相等
    assert maple_code(AA[0, 0] ** 2 + sin(AA[0, 1]) + AA[0, 2]) == \
           "sin(AA[1, 2]) + AA[1, 1]^2 + AA[1, 3]"
    
    # 断言生成的 Maple 代码与预期的表达式相等
    assert maple_code(sum(AA)) == "AA[1, 1] + AA[1, 2] + AA[1, 3]"


# 定义一个函数用于
    # 断言：确保调用 maple_code 函数并传入 S.false 参数时返回字符串 "false"
    assert maple_code(S.false) == "false"
# 定义测试函数 test_sparse
def test_sparse():
    # 创建一个 5x6 的稀疏矩阵 SparseMatrix，初始为空
    M = SparseMatrix(5, 6, {})
    # 在位置 (2, 2) 设置值为 10
    M[2, 2] = 10
    # 在位置 (1, 2) 设置值为 20
    M[1, 2] = 20
    # 在位置 (1, 3) 设置值为 22
    M[1, 3] = 22
    # 在位置 (0, 3) 设置值为 30
    M[0, 3] = 30
    # 在位置 (3, 0) 设置值为 x * y，这里 x 和 y 应该是之前定义过的变量
    M[3, 0] = x * y
    # 断言：调用 maple_code 函数对 M 进行处理后得到的字符串应符合指定格式
    assert maple_code(M) == \
           'Matrix([[0, 0, 0, 30, 0, 0],' \
           ' [0, 0, 20, 22, 0, 0],' \
           ' [0, 0, 10, 0, 0, 0],' \
           ' [x*y, 0, 0, 0, 0, 0],' \
           ' [0, 0, 0, 0, 0, 0]], ' \
           'storage = sparse)'

# 定义测试函数 test_maple_not_supported，测试处理 S.ComplexInfinity 时是否会抛出 NotImplementedError
def test_maple_not_supported():
    # 使用 raises 上下文管理器检查是否抛出 NotImplementedError 异常
    with raises(NotImplementedError):
        maple_code(S.ComplexInfinity)

# 定义测试函数 test_MatrixElement_printing，测试矩阵元素的打印
def test_MatrixElement_printing():
    # 创建符号矩阵 A 和 B
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)

    # 断言：调用 maple_code 函数对 A[0, 0] 进行处理后得到的字符串应符合指定格式
    assert maple_code(A[0, 0]) == "A[1, 1]"
    # 断言：调用 maple_code 函数对 3 * A[0, 0] 进行处理后得到的字符串应符合指定格式
    assert maple_code(3 * A[0, 0]) == "3*A[1, 1]"

    # 创建矩阵 F 为 A - B
    F = A - B

    # 断言：调用 maple_code 函数对 F[0, 0] 进行处理后得到的字符串应符合指定格式
    assert maple_code(F[0,0]) == "A[1, 1] - B[1, 1]"

# 定义测试函数 test_hadamard，测试哈达玛积运算
def test_hadamard():
    # 创建符号矩阵 A、B、v、h
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)

    # 计算 A 和 B 的哈达玛积 C
    C = HadamardProduct(A, B)

    # 断言：调用 maple_code 函数对 C 进行处理后得到的字符串应符合指定格式
    assert maple_code(C) == "A*B"

    # 断言：调用 maple_code 函数对 C * v 进行处理后得到的字符串应符合指定格式
    assert maple_code(C * v) == "(A*B).v"

    # 断言：调用 maple_code 函数对 h * C * v 进行处理后得到的字符串应符合指定格式
    assert maple_code(h * C * v) == "h.(A*B).v"

    # 断言：调用 maple_code 函数对 C * A 进行处理后得到的字符串应符合指定格式
    assert maple_code(C * A) == "(A*B).A"

    # 断言：调用 maple_code 函数对 C * x * y 进行处理后得到的字符串应符合指定格式
    assert maple_code(C * x * y) == "x*y*(A*B)"

# 定义测试函数 test_maple_piecewise，测试 Piecewise 函数
def test_maple_piecewise():
    # 创建 Piecewise 表达式 expr
    expr = Piecewise((x, x < 1), (x ** 2, True))

    # 断言：调用 maple_code 函数对 expr 进行处理后得到的字符串应符合指定格式
    assert maple_code(expr) == "piecewise(x < 1, x, x^2)"

    # 断言：调用 maple_code 函数对 expr，并指定 assign_to 参数为 "r"，得到的字符串应符合指定格式
    assert maple_code(expr, assign_to="r") == (
        "r := piecewise(x < 1, x, x^2)")

    # 创建复杂的 Piecewise 表达式 expr
    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = "piecewise(x < 1, x^2, x < 2, x^3, x < 3, x^4, x^5)"

    # 断言：调用 maple_code 函数对 expr 进行处理后得到的字符串应符合指定格式
    assert maple_code(expr) == expected

    # 断言：调用 maple_code 函数对 expr，并指定 assign_to 参数为 "r"，得到的字符串应符合指定格式
    assert maple_code(expr, assign_to="r") == "r := " + expected

    # 检查不包含 True (默认条件) 的 Piecewise 表达式是否会抛出 ValueError 异常
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: maple_code(expr))

# 定义测试函数 test_maple_piecewise_times_const，测试 Piecewise 乘以常数
def test_maple_piecewise_times_const():
    # 创建 Piecewise 表达式 pw
    pw = Piecewise((x, x < 1), (x ** 2, True))

    # 断言：调用 maple_code 函数对 2 * pw 进行处理后得到的字符串应符合指定格式
    assert maple_code(2 * pw) == "2*piecewise(x < 1, x, x^2)"

    # 断言：调用 maple_code 函数对 pw / x 进行处理后得到的字符串应符合指定格式
    assert maple_code(pw / x) == "piecewise(x < 1, x, x^2)/x"

    # 断言：调用 maple_code 函数对 pw / (x * y) 进行处理后得到的字符串应符合指定格式
    assert maple_code(pw / (x * y)) == "piecewise(x < 1, x, x^2)/(x*y)"

    # 断言：调用 maple_code 函数对 pw / 3 进行处理后得到的字符串应符合指定格式
    assert maple_code(pw / 3) == "(1/3)*piecewise(x < 1, x, x^2)"

# 定义测试函数 test_maple_derivatives，测试导数
def test_maple_derivatives():
    # 创建函数 f
    f = Function('f')

    # 断言：调用 maple_code 函数对 f(x).diff(x) 进行处理后得到的字符串应符合指定格式
    assert maple_code(f(x).diff(x)) == 'diff(f(x), x)'
    
    # 断言：调用 maple_code 函数对 f(x).diff(x, 2) 进行处理后得到的字符串应符合指定格式
    assert maple_code(f(x).diff(x, 2)) == 'diff(f(x), x$2)'

# 定义测试函数 test_automatic_rewrites，测试自动重写
def test_automatic_rewrites():
    # 断言：调用 maple_code 函数对 lucas(x) 进行处理后得到的字符串应符合指定格式
    assert maple
```