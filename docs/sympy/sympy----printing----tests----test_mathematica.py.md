# `D:\src\scipysrc\sympy\sympy\printing\tests\test_mathematica.py`

```
# 从 sympy.core 模块中导入多个符号和函数，用于符号计算和数学操作
# 导入的内容包括 S, pi, oo, symbols, Function, Rational, Integer, Tuple,
# Derivative, Eq, Ne, Le, Lt, Gt, Ge 等
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
                        Derivative, Eq, Ne, Le, Lt, Gt, Ge)

# 从 sympy.integrals 模块中导入 Integral，用于符号积分操作
from sympy.integrals import Integral

# 从 sympy.concrete 模块中导入 Sum，用于符号求和操作
from sympy.concrete import Sum

# 从 sympy.functions 模块中导入多个数学函数，包括指数、三角函数、特殊函数等
# 导入的内容包括 exp, sin, cos, fresnelc, fresnels, conjugate, Max,
# Min, gamma, polygamma, loggamma, erf, erfi, erfc, erf2, expint, erfinv,
# erfcinv, Ei, Si, Ci, li, Shi, Chi, uppergamma, beta 等
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
                             Min, gamma, polygamma, loggamma, erf, erfi, erfc,
                             erf2, expint, erfinv, erfcinv, Ei, Si, Ci, li,
                             Shi, Chi, uppergamma, beta, subfactorial, erf2inv,
                             factorial, factorial2, catalan, RisingFactorial,
                             FallingFactorial, harmonic, atan2, sec, acsc,
                             hermite, laguerre, assoc_laguerre, jacobi,
                             gegenbauer, chebyshevt, chebyshevu, legendre,
                             assoc_legendre, Li, LambertW)

# 从 sympy.printing.mathematica 模块中导入 mathematica_code 函数，用于生成数学家代码
from sympy.printing.mathematica import mathematica_code as mcode

# 定义符号变量 x, y, z, w
x, y, z, w = symbols('x,y,z,w')

# 定义一个未知函数 f
f = Function('f')

# 定义一个测试函数 test_Integer，用于测试 Integer 类型在 Mathematica 代码中的输出
def test_Integer():
    assert mcode(Integer(67)) == "67"
    assert mcode(Integer(-1)) == "-1"

# 定义一个测试函数 test_Rational，用于测试 Rational 类型在 Mathematica 代码中的输出
def test_Rational():
    assert mcode(Rational(3, 7)) == "3/7"
    assert mcode(Rational(18, 9)) == "2"
    assert mcode(Rational(3, -7)) == "-3/7"
    assert mcode(Rational(-3, -7)) == "3/7"
    assert mcode(x + Rational(3, 7)) == "x + 3/7"
    assert mcode(Rational(3, 7)*x) == "(3/7)*x"

# 定义一个测试函数 test_Relational，用于测试关系运算符在 Mathematica 代码中的输出
def test_Relational():
    assert mcode(Eq(x, y)) == "x == y"
    assert mcode(Ne(x, y)) == "x != y"
    assert mcode(Le(x, y)) == "x <= y"
    assert mcode(Lt(x, y)) == "x < y"
    assert mcode(Gt(x, y)) == "x > y"
    assert mcode(Ge(x, y)) == "x >= y"

# 定义一个测试函数 test_Function，用于测试函数调用和数学函数在 Mathematica 代码中的输出
def test_Function():
    assert mcode(f(x, y, z)) == "f[x, y, z]"
    assert mcode(sin(x) ** cos(x)) == "Sin[x]^Cos[x]"
    assert mcode(sec(x) * acsc(x)) == "ArcCsc[x]*Sec[x]"
    assert mcode(atan2(x, y)) == "ArcTan[x, y]"
    assert mcode(conjugate(x)) == "Conjugate[x]"
    assert mcode(Max(x, y, z)*Min(y, z)) == "Max[x, y, z]*Min[y, z]"
    assert mcode(fresnelc(x)) == "FresnelC[x]"
    assert mcode(fresnels(x)) == "FresnelS[x]"
    assert mcode(gamma(x)) == "Gamma[x]"
    assert mcode(uppergamma(x, y)) == "Gamma[x, y]"
    assert mcode(polygamma(x, y)) == "PolyGamma[x, y]"
    assert mcode(loggamma(x)) == "LogGamma[x]"
    assert mcode(erf(x)) == "Erf[x]"
    assert mcode(erfc(x)) == "Erfc[x]"
    assert mcode(erfi(x)) == "Erfi[x]"
    assert mcode(erf2(x, y)) == "Erf[x, y]"
    assert mcode(expint(x, y)) == "ExpIntegralE[x, y]"
    assert mcode(erfcinv(x)) == "InverseErfc[x]"
    assert mcode(erfinv(x)) == "InverseErf[x]"
    assert mcode(erf2inv(x, y)) == "InverseErf[x, y]"
    assert mcode(Ei(x)) == "ExpIntegralEi[x]"
    assert mcode(Ci(x)) == "CosIntegral[x]"
    assert mcode(li(x)) == "LogIntegral[x]"
    assert mcode(Si(x)) == "SinIntegral[x]"
    assert mcode(Shi(x)) == "SinhIntegral[x]"
    assert mcode(Chi(x)) == "CoshIntegral[x]"
    assert mcode(beta(x, y)) == "Beta[x, y]"
    # 断言：检查函数 mcode(factorial(x)) 的输出是否等于 "Factorial[x]"
    assert mcode(factorial(x)) == "Factorial[x]"
    
    # 断言：检查函数 mcode(factorial2(x)) 的输出是否等于 "Factorial2[x]"
    assert mcode(factorial2(x)) == "Factorial2[x]"
    
    # 断言：检查函数 mcode(subfactorial(x)) 的输出是否等于 "Subfactorial[x]"
    assert mcode(subfactorial(x)) == "Subfactorial[x]"
    
    # 断言：检查函数 mcode(FallingFactorial(x, y)) 的输出是否等于 "FactorialPower[x, y]"
    assert mcode(FallingFactorial(x, y)) == "FactorialPower[x, y]"
    
    # 断言：检查函数 mcode(RisingFactorial(x, y)) 的输出是否等于 "Pochhammer[x, y]"
    assert mcode(RisingFactorial(x, y)) == "Pochhammer[x, y]"
    
    # 断言：检查函数 mcode(catalan(x)) 的输出是否等于 "CatalanNumber[x]"
    assert mcode(catalan(x)) == "CatalanNumber[x]"
    
    # 断言：检查函数 mcode(harmonic(x)) 的输出是否等于 "HarmonicNumber[x]"
    assert mcode(harmonic(x)) == "HarmonicNumber[x]"
    
    # 断言：检查函数 mcode(harmonic(x, y)) 的输出是否等于 "HarmonicNumber[x, y]"
    assert mcode(harmonic(x, y)) == "HarmonicNumber[x, y]"
    
    # 断言：检查函数 mcode(Li(x)) 的输出是否等于 "LogIntegral[x] - LogIntegral[2]"
    assert mcode(Li(x)) == "LogIntegral[x] - LogIntegral[2]"
    
    # 断言：检查函数 mcode(LambertW(x)) 的输出是否等于 "ProductLog[x]"
    assert mcode(LambertW(x)) == "ProductLog[x]"
    
    # 断言：检查函数 mcode(LambertW(x, -1)) 的输出是否等于 "ProductLog[-1, x]"
    assert mcode(LambertW(x, -1)) == "ProductLog[-1, x]"
    
    # 断言：检查函数 mcode(LambertW(x, y)) 的输出是否等于 "ProductLog[y, x]"
    assert mcode(LambertW(x, y)) == "ProductLog[y, x]"
# 测试特殊多项式函数的代码块
def test_special_polynomials():
    # 断言 Hermite 多项式的 Mathematica 代码表示
    assert mcode(hermite(x, y)) == "HermiteH[x, y]"
    # 断言 Laguerre 多项式的 Mathematica 代码表示
    assert mcode(laguerre(x, y)) == "LaguerreL[x, y]"
    # 断言关联 Laguerre 多项式的 Mathematica 代码表示
    assert mcode(assoc_laguerre(x, y, z)) == "LaguerreL[x, y, z]"
    # 断言 Jacobi 多项式的 Mathematica 代码表示
    assert mcode(jacobi(x, y, z, w)) == "JacobiP[x, y, z, w]"
    # 断言 Gegenbauer 多项式的 Mathematica 代码表示
    assert mcode(gegenbauer(x, y, z)) == "GegenbauerC[x, y, z]"
    # 断言 Chebyshev T 多项式的 Mathematica 代码表示
    assert mcode(chebyshevt(x, y)) == "ChebyshevT[x, y]"
    # 断言 Chebyshev U 多项式的 Mathematica 代码表示
    assert mcode(chebyshevu(x, y)) == "ChebyshevU[x, y]"
    # 断言 Legendre 多项式的 Mathematica 代码表示
    assert mcode(legendre(x, y)) == "LegendreP[x, y]"
    # 断言关联 Legendre 多项式的 Mathematica 代码表示
    assert mcode(assoc_legendre(x, y, z)) == "LegendreP[x, y, z]"


# 测试幂运算的代码块
def test_Pow():
    # 断言 x 的三次幂的 Mathematica 代码表示
    assert mcode(x**3) == "x^3"
    # 断言 x 的 y 的三次幂的 Mathematica 代码表示
    assert mcode(x**(y**3)) == "x^(y^3)"
    # 断言复杂表达式的 Mathematica 代码表示
    assert mcode(1/(f(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*f[x])^(-x + y^x)/(x^2 + y)"
    # 断言 x 的 -1.0 次幂的 Mathematica 代码表示
    assert mcode(x**-1.0) == 'x^(-1.0)'
    # 断言 x 的有理数次幂的 Mathematica 代码表示
    assert mcode(x**Rational(2, 3)) == 'x^(2/3)'


# 测试乘法运算的代码块
def test_Mul():
    # 定义非交换符号变量
    A, B, C, D = symbols('A B C D', commutative=False)
    # 断言乘法表达式的 Mathematica 代码表示
    assert mcode(x*y*z) == "x*y*z"
    assert mcode(x*y*A) == "x*y*A"
    assert mcode(x*y*A*B) == "x*y*A**B"
    assert mcode(x*y*A*B*C) == "x*y*A**B**C"
    assert mcode(x*A*B*(C + D)*A*y) == "x*y*A**B**(C + D)**A"


# 测试常数的代码块
def test_constants():
    # 断言常数的 Mathematica 代码表示
    assert mcode(S.Zero) == "0"
    assert mcode(S.One) == "1"
    assert mcode(S.NegativeOne) == "-1"
    assert mcode(S.Half) == "1/2"
    assert mcode(S.ImaginaryUnit) == "I"
    assert mcode(oo) == "Infinity"
    assert mcode(S.NegativeInfinity) == "-Infinity"
    assert mcode(S.ComplexInfinity) == "ComplexInfinity"
    assert mcode(S.NaN) == "Indeterminate"
    assert mcode(S.Exp1) == "E"
    assert mcode(pi) == "Pi"
    assert mcode(S.GoldenRatio) == "GoldenRatio"
    assert mcode(S.TribonacciConstant) == \
        "(1/3 + (1/3)*(19 - 3*33^(1/2))^(1/3) + " \
        "(1/3)*(3*33^(1/2) + 19)^(1/3))"
    assert mcode(2*S.TribonacciConstant) == \
        "2*(1/3 + (1/3)*(19 - 3*33^(1/2))^(1/3) + " \
        "(1/3)*(3*33^(1/2) + 19)^(1/3))"
    assert mcode(S.EulerGamma) == "EulerGamma"
    assert mcode(S.Catalan) == "Catalan"


# 测试容器类型的代码块
def test_containers():
    # 断言列表的 Mathematica 代码表示
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
        "{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}"
    # 断言元组的 Mathematica 代码表示
    assert mcode((1, 2, (3, 4))) == "{1, 2, {3, 4}}"
    assert mcode([1]) == "{1}"
    assert mcode((1,)) == "{1}"
    # 断言符号表达式的 Mathematica 代码表示
    assert mcode(Tuple(*[1, 2, 3])) == "{1, 2, 3}"


# 测试矩阵的代码块
def test_matrices():
    # 导入矩阵类
    from sympy.matrices import MutableDenseMatrix, MutableSparseMatrix, \
        ImmutableDenseMatrix, ImmutableSparseMatrix
    # 创建可变密集矩阵
    A = MutableDenseMatrix(
        [[1, -1, 0, 0],
         [0, 1, -1, 0],
         [0, 0, 1, -1],
         [0, 0, 0, 1]]
    )
    # 根据 A 创建稀疏矩阵 B
    B = MutableSparseMatrix(A)
    # 根据 A 创建不可变密集矩阵 C
    C = ImmutableDenseMatrix(A)
    # 根据 A 创建不可变稀疏矩阵 D
    D = ImmutableSparseMatrix(A)
    
    # 断言矩阵的 Mathematica 代码表示
    assert mcode(C) == mcode(A) == \
        "{{1, -1, 0, 0}, " \
        "{0, 1, -1, 0}, " \
        "{0, 0, 1, -1}, " \
        "{0, 0, 0, 1}}"
    # 断言语句，验证函数 mcode 的返回值是否符合预期
    assert mcode(D) == mcode(B) == \
        "SparseArray[{" \
        "{1, 1} -> 1, {1, 2} -> -1, {2, 2} -> 1, {2, 3} -> -1, " \
        "{3, 3} -> 1, {3, 4} -> -1, {4, 4} -> 1" \
        "}, {4, 4}]"

    # 对于空矩阵的边界情况进行验证
    assert mcode(MutableDenseMatrix(0, 0, [])) == '{}'
    assert mcode(MutableSparseMatrix(0, 0, [])) == 'SparseArray[{}, {0, 0}]'
    assert mcode(MutableDenseMatrix(0, 3, [])) == '{}'
    assert mcode(MutableSparseMatrix(0, 3, [])) == 'SparseArray[{}, {0, 3}]'
    assert mcode(MutableDenseMatrix(3, 0, [])) == '{{}, {}, {}}'
    assert mcode(MutableSparseMatrix(3, 0, [])) == 'SparseArray[{}, {3, 0}]'
# 导入相关的数组类
def test_NDArray():
    from sympy.tensor.array import (
        MutableDenseNDimArray, ImmutableDenseNDimArray,
        MutableSparseNDimArray, ImmutableSparseNDimArray)

    # 创建一个可变密集多维数组示例
    example = MutableDenseNDimArray(
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12]],
         [[13, 14, 15, 16],
          [17, 18, 19, 20],
          [21, 22, 23, 24]]]
    )

    # 断言语句，验证 mcode 函数处理后的输出是否符合预期
    assert mcode(example) == \
    "{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, " \
    "{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}"

    # 将示例数组转换为不可变密集多维数组
    example = ImmutableDenseNDimArray(example)

    # 再次验证 mcode 函数处理后的输出是否符合预期
    assert mcode(example) == \
    "{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, " \
    "{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}"

    # 将示例数组转换为可变稀疏多维数组
    example = MutableSparseNDimArray(example)

    # 再次验证 mcode 函数处理后的输出是否符合预期
    assert mcode(example) == \
    "SparseArray[{" \
        "{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, " \
        "{1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, " \
        "{1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, " \
        "{1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, " \
        "{2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, " \
        "{2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, " \
        "{2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, " \
        "{2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24" \
        "}, {2, 3, 4}]"

    # 将示例数组转换为不可变稀疏多维数组
    example = ImmutableSparseNDimArray(example)

    # 最后一次验证 mcode 函数处理后的输出是否符合预期
    assert mcode(example) == \
    "SparseArray[{" \
        "{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, " \
        "{1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, " \
        "{1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, " \
        "{1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, " \
        "{2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, " \
        "{2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, " \
        "{2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, " \
        "{2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24" \
        "}, {2, 3, 4}]"


# 对积分表达式进行测试
def test_Integral():
    # 断言语句，验证 mcode 函数处理后的输出是否符合预期
    assert mcode(Integral(sin(sin(x)), x)) == "Hold[Integrate[Sin[Sin[x]], x]]"
    assert mcode(Integral(exp(-x**2 - y**2),
                          (x, -oo, oo),
                          (y, -oo, oo))) == \
        "Hold[Integrate[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, " \
        "{y, -Infinity, Infinity}]]"


# 对导数表达式进行测试
def test_Derivative():
    # 断言语句，验证 mcode 函数处理后的输出是否符合预期
    assert mcode(Derivative(sin(x), x)) == "Hold[D[Sin[x], x]]"
    assert mcode(Derivative(x, x)) == "Hold[D[x, x]]"
    assert mcode(Derivative(sin(x)*y**4, x, 2)) == "Hold[D[y^4*Sin[x], {x, 2}]]"
    assert mcode(Derivative(sin(x)*y**4, x, y, x)) == "Hold[D[y^4*Sin[x], x, y, x]]"
    assert mcode(Derivative(sin(x)*y**4, x, y, 3, x)) == "Hold[D[y^4*Sin[x], x, {y, 3}, x]]"


# 对求和表达式进行测试
def test_Sum():
    # 断言语句，验证 mcode 函数处理后的输出是否符合预期
    assert mcode(Sum(sin(x), (x, 0, 10))) == "Hold[Sum[Sin[x], {x, 0, 10}]]"
    # 断言语句，用于验证 mcode 函数对给定表达式的处理结果是否符合预期
    assert mcode(
        Sum(exp(-x**2 - y**2),        # 对 exp(-x**2 - y**2) 进行求和
            (x, -oo, oo),             # x 从负无穷到正无穷
            (y, -oo, oo))             # y 从负无穷到正无穷
    ) == \
        "Hold[Sum[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, " \
        "{y, -Infinity, Infinity}]]"
    # 验证 mcode 函数是否能正确将数学表达式转换为指定的字符串表示形式
def test_comment():
    # 从 sympy.printing.mathematica 模块导入 MCodePrinter 类
    from sympy.printing.mathematica import MCodePrinter
    # 断言调用 MCodePrinter 实例的 _get_comment 方法，传入参数 "Hello World"，返回预期的注释格式
    assert MCodePrinter()._get_comment("Hello World") == \
        "(* Hello World *)"


def test_userfuncs():
    # Dictionary mutation test
    # 创建一个 SymPy 函数符号对象，命名为 some_function
    some_function = symbols("some_function", cls=Function)
    # 创建包含一个键值对的字典 my_user_functions，用于用户自定义函数的映射
    my_user_functions = {"some_function": "SomeFunction"}
    # 断言调用 mcode 函数，传入 some_function(z) 和 my_user_functions 参数，返回预期的数学代码字符串
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeFunction[z]'
    # 再次断言相同的调用，验证输出结果是否一致
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeFunction[z]'

    # List argument test
    # 重新赋值 my_user_functions，这次使用包含 lambda 函数的列表形式的值
    my_user_functions = \
        {"some_function": [(lambda x: True, "SomeOtherFunction")]}
    # 断言调用 mcode 函数，传入 some_function(z) 和 my_user_functions 参数，返回预期的数学代码字符串
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeOtherFunction[z]'
```