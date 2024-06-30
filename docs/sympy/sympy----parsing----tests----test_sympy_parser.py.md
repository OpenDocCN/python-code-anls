# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_sympy_parser.py`

```
# -*- coding: utf-8 -*-


import sys
import builtins
import types

from sympy.assumptions import Q  # 导入 Q 对象，用于符号推理
from sympy.core import Symbol, Function, Float, Rational, Integer, I, Mul, Pow, Eq, Lt, Le, Gt, Ge, Ne  # 导入符号代数运算相关的类和函数
from sympy.functions import exp, factorial, factorial2, sin, Min, Max  # 导入数学函数，如指数函数、阶乘等
from sympy.logic import And  # 导入逻辑运算相关的类
from sympy.series import Limit  # 导入极限相关的类
from sympy.testing.pytest import raises, skip  # 导入测试相关的函数和装饰器

from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, rationalize, TokenError,
    split_symbols, implicit_multiplication, convert_equals_signs,
    convert_xor, function_exponentiation, lambda_notation, auto_symbol,
    repeated_decimals, implicit_multiplication_application,
    auto_number, factorial_notation, implicit_application,
    _transformation, T
    )  # 导入用于解析和转换表达式的函数和类


def test_sympy_parser():
    x = Symbol('x')  # 创建符号变量 x
    inputs = {
        '2*x': 2 * x,  # 测试表达式 2*x
        '3.00': Float(3),  # 测试浮点数转换
        '22/7': Rational(22, 7),  # 测试有理数转换
        '2+3j': 2 + 3*I,  # 测试复数转换
        'exp(x)': exp(x),  # 测试指数函数
        'x!': factorial(x),  # 测试阶乘函数
        'x!!': factorial2(x),  # 测试双阶乘函数
        '(x + 1)! - 1': factorial(x + 1) - 1,  # 测试复合表达式
        '3.[3]': Rational(10, 3),  # 测试无限循环小数转换
        '.0[3]': Rational(1, 30),  # 测试无限循环小数转换
        '3.2[3]': Rational(97, 30),  # 测试无限循环小数转换
        '1.3[12]': Rational(433, 330),  # 测试无限循环小数转换
        '1 + 3.[3]': Rational(13, 3),  # 测试复合表达式
        '1 + .0[3]': Rational(31, 30),  # 测试复合表达式
        '1 + 3.2[3]': Rational(127, 30),  # 测试复合表达式
        '.[0011]': Rational(1, 909),  # 测试无限循环小数转换
        '0.1[00102] + 1': Rational(366697, 333330),  # 测试复合表达式
        '1.[0191]': Rational(10190, 9999),  # 测试无限循环小数转换
        '10!': 3628800,  # 测试阶乘函数
        '-(2)': -Integer(2),  # 测试整数转换
        '[-1, -2, 3]': [Integer(-1), Integer(-2), Integer(3)],  # 测试列表中整数转换
        'Symbol("x").free_symbols': x.free_symbols,  # 测试符号的属性访问
        "S('S(3).n(n=3)')": Float(3, 3),  # 测试浮点数精度
        'factorint(12, visual=True)': Mul(
            Pow(2, 2, evaluate=False),
            Pow(3, 1, evaluate=False),
            evaluate=False),  # 测试因式分解函数
        'Limit(sin(x), x, 0, dir="-")': Limit(sin(x), x, 0, dir='-'),  # 测试极限函数
        'Q.even(x)': Q.even(x),  # 测试符号推理函数

    }
    for text, result in inputs.items():
        assert parse_expr(text) == result  # 断言解析结果与预期结果一致

    raises(TypeError, lambda:
        parse_expr('x', standard_transformations))  # 测试处理类型错误
    raises(TypeError, lambda:
        parse_expr('x', transformations=lambda x,y: 1))  # 测试处理类型错误
    raises(TypeError, lambda:
        parse_expr('x', transformations=(lambda x,y: 1,)))  # 测试处理类型错误
    raises(TypeError, lambda: parse_expr('x', transformations=((),)))  # 测试处理类型错误
    raises(TypeError, lambda: parse_expr('x', {}, [], []))  # 测试处理类型错误
    raises(TypeError, lambda: parse_expr('x', [], [], {}))  # 测试处理类型错误
    raises(TypeError, lambda: parse_expr('x', [], [], {}))  # 测试处理类型错误


def test_rationalize():
    inputs = {
        '0.123': Rational(123, 1000)  # 测试有理数近似转换
    }
    transformations = standard_transformations + (rationalize,)
    for text, result in inputs.items():
        assert parse_expr(text, transformations=transformations) == result  # 断言有理数近似转换结果正确


def test_factorial_fail():
    inputs = ['x!!!', 'x!!!!', '(!)']

    for text in inputs:
        try:
            parse_expr(text)  # 尝试解析表达式，期望抛出 TokenError 异常
            assert False
        except TokenError:
            assert True  # 捕获到 TokenError 异常，测试通过
def test_repeated_fail():
    # 定义输入的列表，包含了一些不符合语法的字符串表达式
    inputs = ['1[1]', '.1e1[1]', '0x1[1]', '1.1j[1]', '1.1[1 + 1]',
              '0.1[[1]]', '0x1.1[1]']

    # 对每个字符串表达式进行测试，期望抛出 TypeError 异常
    for text in inputs:
        raises(TypeError, lambda: parse_expr(text))

    # 定义另一个输入的列表，包含了一些语法上不完整的字符串表达式
    inputs = ['0.1[', '0.1[1', '0.1[]']
    # 对每个字符串表达式进行测试，期望抛出 TokenError 或 SyntaxError 异常
    for text in inputs:
        raises((TokenError, SyntaxError), lambda: parse_expr(text))


def test_repeated_dot_only():
    # 断言特定的解析表达式结果
    assert parse_expr('.[1]') == Rational(1, 9)
    assert parse_expr('1 + .[1]') == Rational(10, 9)


def test_local_dict():
    # 定义一个本地变量字典，包含一个 lambda 函数作为值
    local_dict = {
        'my_function': lambda x: x + 2
    }
    # 定义一个输入字典，包含了需要测试的字符串表达式及其预期结果
    inputs = {
        'my_function(2)': Integer(4)
    }
    # 对每个输入进行测试，断言解析表达式的结果与预期结果相等
    for text, result in inputs.items():
        assert parse_expr(text, local_dict=local_dict) == result


def test_local_dict_split_implmult():
    # 定义转换和符号处理的元组
    t = standard_transformations + (split_symbols, implicit_multiplication,)
    w = Symbol('w', real=True)
    y = Symbol('y')
    # 断言解析特定表达式的结果与预期相等，使用本地字典和指定的转换
    assert parse_expr('yx', local_dict={'x':w}, transformations=t) == y*w


def test_local_dict_symbol_to_fcn():
    x = Symbol('x')
    d = {'foo': Function('bar')}
    # 断言解析特定表达式的结果与预期相等，使用本地字典进行函数调用
    assert parse_expr('foo(x)', local_dict=d) == d['foo'](x)
    # 期望解析包含错误类型的表达式时抛出 TypeError 异常
    d = {'foo': Symbol('baz')}
    raises(TypeError, lambda: parse_expr('foo(x)', local_dict=d))


def test_global_dict():
    # 定义一个全局字典，包含了一个 Symbol 类型的对象
    global_dict = {
        'Symbol': Symbol
    }
    # 定义一个输入字典，包含了需要测试的字符串表达式及其预期结果
    inputs = {
        'Q & S': And(Symbol('Q'), Symbol('S'))
    }
    # 对每个输入进行测试，断言解析表达式的结果与预期结果相等
    for text, result in inputs.items():
        assert parse_expr(text, global_dict=global_dict) == result


def test_no_globals():
    # 复制创建默认全局字典的过程
    default_globals = {}
    exec('from sympy import *', default_globals)
    builtins_dict = vars(builtins)
    for name, obj in builtins_dict.items():
        if isinstance(obj, types.BuiltinFunctionType):
            default_globals[name] = obj
    default_globals['max'] = Max
    default_globals['min'] = Min

    # 移除 Symbol 来确保 parse_expr 能够工作
    default_globals.pop('Symbol')
    global_dict = {'Symbol':Symbol}

    # 对于默认全局字典中的每个名称，断言解析表达式的结果与预期的符号相等
    for name in default_globals:
        obj = parse_expr(name, global_dict=global_dict)
        assert obj == Symbol(name)


def test_issue_2515():
    # 期望解析特定的表达式时抛出 TokenError 异常
    raises(TokenError, lambda: parse_expr('(()'))
    raises(TokenError, lambda: parse_expr('"""'))


def test_issue_7663():
    x = Symbol('x')
    e = '2*(x+1)'
    # 断言解析表达式并禁用求值时的结果与手动求值结果相等
    assert parse_expr(e, evaluate=0) == parse_expr(e, evaluate=False)
    # 断言解析表达式并禁用求值时的结果与标准乘法运算结果相等
    assert parse_expr(e, evaluate=0).equals(2*(x+1))


def test_recursive_evaluate_false_10560():
    inputs = {
        '4*-3' : '4*-3',
        '-4*3' : '(-4)*3',
        "-2*x*y": '(-2)*x*y',
        "x*-4*x": "x*(-4)*x"
    }
    # 对于每个输入表达式及其预期结果，断言解析表达式的结果与预期结果相等，禁用求值
    for text, result in inputs.items():
        assert parse_expr(text, evaluate=False) == parse_expr(result, evaluate=False)


def test_function_evaluate_false():
    # 这个测试函数未完整给出，没有需要添加注释的内容，暂不提供注释
    # 定义输入列表，包含各种数学函数的字符串表示
    inputs = [
        'Abs(0)', 'im(0)', 're(0)', 'sign(0)', 'arg(0)', 'conjugate(0)',
        'acos(0)', 'acot(0)', 'acsc(0)', 'asec(0)', 'asin(0)', 'atan(0)',
        'acosh(0)', 'acoth(0)', 'acsch(0)', 'asech(0)', 'asinh(0)', 'atanh(0)',
        'cos(0)', 'cot(0)', 'csc(0)', 'sec(0)', 'sin(0)', 'tan(0)',
        'cosh(0)', 'coth(0)', 'csch(0)', 'sech(0)', 'sinh(0)', 'tanh(0)',
        'exp(0)', 'log(0)', 'sqrt(0)',
    ]
    
    # 遍历输入列表中的每个数学表达式字符串
    for case in inputs:
        # 解析数学表达式字符串为符号表达式对象，但不进行求值
        expr = parse_expr(case, evaluate=False)
        # 断言：字符串表达式与其字符串表示相同，并且不等于表达式进行求值后的字符串表示
        assert case == str(expr) != str(expr.doit())
    
    # 断言：解析'ln(0)'的结果字符串为'log(0)'
    assert str(parse_expr('ln(0)', evaluate=False)) == 'log(0)'
    # 断言：解析'cbrt(0)'的结果字符串为'0**(1/3)'
    assert str(parse_expr('cbrt(0)', evaluate=False)) == '0**(1/3)'
# 测试函数，用于验证 Issue 10773
def test_issue_10773():
    # 输入字典，包含测试用例和预期结果
    inputs = {
        '-10/5': '(-10)/5',
        '-10/-5' : '(-10)/(-5)',
    }
    # 遍历输入字典的每一项
    for text, result in inputs.items():
        # 断言解析表达式后的结果与预期结果相等（不执行求值）
        assert parse_expr(text, evaluate=False) == parse_expr(result, evaluate=False)


# 测试函数，验证符号分割的转换是否有效
def test_split_symbols():
    # 定义变换列表，包括标准变换、符号分割和隐式乘法
    transformations = standard_transformations + \
                      (split_symbols, implicit_multiplication,)
    # 创建符号变量 x, y, xy
    x = Symbol('x')
    y = Symbol('y')
    xy = Symbol('xy')

    # 断言解析表达式后与预期的符号变量相等
    assert parse_expr("xy") == xy
    assert parse_expr("xy", transformations=transformations) == x*y


# 测试函数，验证带函数的符号分割转换是否有效
def test_split_symbols_function():
    # 定义变换列表，包括标准变换、符号分割和隐式乘法
    transformations = standard_transformations + \
                      (split_symbols, implicit_multiplication,)
    # 创建符号变量 x, y, a 和函数 f
    x = Symbol('x')
    y = Symbol('y')
    a = Symbol('a')
    f = Function('f')

    # 断言解析表达式后与预期的函数表达式相等
    assert parse_expr("ay(x+1)", transformations=transformations) == a*y*(x+1)
    assert parse_expr("af(x+1)", transformations=transformations,
                      local_dict={'f':f}) == a*f(x+1)


# 测试函数，验证函数指数化转换是否有效
def test_functional_exponent():
    # 定义变换列表，包括标准变换、异或运算转换和函数指数化
    t = standard_transformations + (convert_xor, function_exponentiation)
    # 创建符号变量 x, y, a 和函数 yfcn
    x = Symbol('x')
    y = Symbol('y')
    a = Symbol('a')
    yfcn = Function('y')
    
    # 断言解析表达式后与预期的函数指数化表达式相等
    assert parse_expr("sin^2(x)", transformations=t) == (sin(x))**2
    assert parse_expr("sin^y(x)", transformations=t) == (sin(x))**y
    assert parse_expr("exp^y(x)", transformations=t) == (exp(x))**y
    assert parse_expr("E^y(x)", transformations=t) == exp(yfcn(x))
    assert parse_expr("a^y(x)", transformations=t) == a**(yfcn(x))


# 测试函数，验证匹配括号和隐式乘法的转换是否有效
def test_match_parentheses_implicit_multiplication():
    # 定义变换列表，包括标准变换和隐式乘法应用
    transformations = standard_transformations + \
                      (implicit_multiplication,)
    # 使用 lambda 函数断言解析表达式会抛出 TokenError 异常
    raises(TokenError, lambda: parse_expr('(1,2),(3,4]', transformations=transformations))


# 测试函数，验证等号符号转换是否有效
def test_convert_equals_signs():
    # 定义变换列表，包括标准变换和等号符号转换
    transformations = standard_transformations + \
                      (convert_equals_signs, )
    # 创建符号变量 x 和 y
    x = Symbol('x')
    y = Symbol('y')
    
    # 断言解析表达式后与预期的等式表达式相等
    assert parse_expr("1*2=x", transformations=transformations) == Eq(2, x)
    assert parse_expr("y = x", transformations=transformations) == Eq(y, x)
    assert parse_expr("(2*y = x) = False",
                      transformations=transformations) == Eq(Eq(2*y, x), False)


# 测试函数，验证解析函数表达式的功能
def test_parse_function_issue_3539():
    # 创建符号变量 x 和函数 f
    x = Symbol('x')
    f = Function('f')
    
    # 断言解析表达式后与预期的函数表达式相等
    assert parse_expr('f(x)') == f(x)


# 测试函数，验证 Issue 24288 中的比较运算符转换是否有效
def test_issue_24288():
    # 定义输入字典，包含比较表达式和预期的比较对象
    inputs = {
        "1 < 2": Lt(1, 2, evaluate=False),
        "1 <= 2": Le(1, 2, evaluate=False),
        "1 > 2": Gt(1, 2, evaluate=False),
        "1 >= 2": Ge(1, 2, evaluate=False),
        "1 != 2": Ne(1, 2, evaluate=False),
        "1 == 2": Eq(1, 2, evaluate=False)
    }
    # 遍历输入字典的每一项
    for text, result in inputs.items():
        # 断言解析表达式后与预期的比较对象相等（不执行求值）
        assert parse_expr(text, evaluate=False) == result


# 测试函数，验证数字与符号的分割转换是否有效
def test_split_symbols_numeric():
    # 定义变换列表，包括标准变换和隐式乘法应用
    transformations = (
        standard_transformations +
        (implicit_multiplication_application,))
    # 创建符号变量 n
    n = Symbol('n')
    
    # 解析表达式并断言与预期的结果相等
    expr1 = parse_expr('2**n * 3**n')
    expr2 = parse_expr('2**n3**n', transformations=transformations)
    # 断言，确保 expr1 等于 expr2，且它们都等于 2 的 n 次方乘以 3 的 n 次方
    assert expr1 == expr2 == 2**n*3**n
    
    # 使用指定的转换规则 transformations 解析字符串 'n12n34' 得到数学表达式 expr1
    expr1 = parse_expr('n12n34', transformations=transformations)
    # 断言，确保 expr1 等于 n * 12 * n * 34
    assert expr1 == n*12*n*34
# 测试函数，验证解析表达式处理 Unicode 名称时的行为是否正确
def test_unicode_names():
    # 断言解析表达式 'α' 是否等于符号对象 Symbol('α')
    assert parse_expr('α') == Symbol('α')


# 测试函数，验证解析器是否能够处理 Python 3 特有的语法特性
def test_python3_features():
    # 如果 Python 版本低于 3.8，则跳过该测试
    if sys.version_info < (3, 8):
        skip("test_python3_features requires Python 3.8 or newer")

    # 断言解析表达式 "123_456" 是否等于整数 123456
    assert parse_expr("123_456") == 123456
    # 断言解析表达式 "1.2[3_4]" 和 "1.2[34]" 是否等于有理数 Rational(611, 495)
    assert parse_expr("1.2[3_4]") == parse_expr("1.2[34]") == Rational(611, 495)
    # 断言解析表达式 "1.2[012_012]" 和 "1.2[012012]" 是否等于有理数 Rational(400, 333)
    assert parse_expr("1.2[012_012]") == parse_expr("1.2[012012]") == Rational(400, 333)
    # 断言解析表达式 '.[3_4]' 和 '.[34]' 是否等于有理数 Rational(34, 99)
    assert parse_expr('.[3_4]') == parse_expr('.[34]') == Rational(34, 99)
    # 断言解析表达式 '.1[3_4]' 和 '.1[34]' 是否等于有理数 Rational(133, 990)
    assert parse_expr('.1[3_4]') == parse_expr('.1[34]') == Rational(133, 990)
    # 断言解析表达式 '123_123.123_123[3_4]' 和 '123123.123123[34]' 是否等于有理数 Rational(12189189189211, 99000000)
    assert parse_expr('123_123.123_123[3_4]') == parse_expr('123123.123123[34]') == Rational(12189189189211, 99000000)


# 测试函数，验证解析器在处理 issue 19501 时的行为是否正确
def test_issue_19501():
    # 创建符号对象 x
    x = Symbol('x')
    # 使用指定的本地变量字典和转换来解析表达式 'E**x(1+x)'
    eq = parse_expr('E**x(1+x)', local_dict={'x': x}, transformations=(
        standard_transformations +
        (implicit_multiplication_application,)))
    # 断言解析后的表达式中自由符号集合是否等于 {x}
    assert eq.free_symbols == {x}


# 测试函数，验证解析器在解析定义时的行为是否正确
def test_parsing_definitions():
    # 从 sympy.abc 导入符号 x
    from sympy.abc import x
    # 断言 _transformation 的长度是否为 12
    assert len(_transformation) == 12  # if this changes, extend below
    # 逐个断言 _transformation 中的元素是否与预期的转换函数相等
    assert _transformation[0] == lambda_notation
    assert _transformation[1] == auto_symbol
    assert _transformation[2] == repeated_decimals
    assert _transformation[3] == auto_number
    assert _transformation[4] == factorial_notation
    assert _transformation[5] == implicit_multiplication_application
    assert _transformation[6] == convert_xor
    assert _transformation[7] == implicit_application
    assert _transformation[8] == implicit_multiplication
    assert _transformation[9] == convert_equals_signs
    assert _transformation[10] == function_exponentiation
    assert _transformation[11] == rationalize
    # 断言 T 的前五个元素是否等于 standard_transformations
    assert T[:5] == T[0,1,2,3,4] == standard_transformations
    # 设置 t 等于 _transformation
    t = _transformation
    # 断言 T 的最后一个元素和第一个元素是否等于 (t[len(t) - 1], t[0])
    assert T[-1, 0] == (t[len(t) - 1], t[0])
    # 断言 T 的前五个元素加上第 8 个元素是否等于 standard_transformations + (t[8],)
    assert T[:5, 8] == standard_transformations + (t[8],)
    # 断言使用 'all' 转换时解析表达式 '0.3x^2' 是否等于 3*x**2/10
    assert parse_expr('0.3x^2', transformations='all') == 3*x**2/10
    # 断言使用 'implicit' 转换时解析表达式 'sin 3x' 是否等于 sin(3*x)
    assert parse_expr('sin 3x', transformations='implicit') == sin(3*x)


# 测试函数，验证解析器在处理内建函数时的行为是否正确
def test_builtins():
    # 定义测试用例列表
    cases = [
        ('abs(x)', 'Abs(x)'),
        ('max(x, y)', 'Max(x, y)'),
        ('min(x, y)', 'Min(x, y)'),
        ('pow(x, y)', 'Pow(x, y)'),
    ]
    # 对于每个内建函数调用和相应的 SymPy 函数调用，断言解析结果是否相等
    for built_in_func_call, sympy_func_call in cases:
        assert parse_expr(built_in_func_call) == parse_expr(sympy_func_call)
    # 断言解析表达式 'pow(38, -1, 97)' 的字符串表示是否等于 '23'
    assert str(parse_expr('pow(38, -1, 97)')) == '23'


# 测试函数，验证解析器在处理 issue 22822 时的异常处理行为是否正确
def test_issue_22822():
    # 使用 lambda 表达式断言解析表达式 'x' 时是否引发 ValueError
    raises(ValueError, lambda: parse_expr('x', {'': 1}))
    # 创建数据字典 data
    data = {'some_parameter': None}
    # 断言解析表达式 'some_parameter is None' 是否为 True
    assert parse_expr('some_parameter is None', data) is True
```