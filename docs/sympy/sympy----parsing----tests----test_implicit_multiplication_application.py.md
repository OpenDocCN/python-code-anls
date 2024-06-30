# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_implicit_multiplication_application.py`

```
# 导入sympy库，用于符号计算
import sympy
# 从sympy.parsing.sympy_parser模块中导入相关函数和转换器
from sympy.parsing.sympy_parser import (
    parse_expr,  # 解析表达式字符串为sympy表达式对象
    standard_transformations,  # 标准转换列表
    convert_xor,  # 转换异或运算符（^）
    implicit_multiplication_application,  # 隐式乘法应用
    implicit_multiplication,  # 隐式乘法
    implicit_application,  # 隐式应用
    function_exponentiation,  # 函数指数
    split_symbols,  # 分割符号
    split_symbols_custom,  # 自定义分割符号
    _token_splittable  # 可分割的标记
)
# 从sympy.testing.pytest导入raises，用于测试异常情况
from sympy.testing.pytest import raises


# 定义测试函数test_implicit_multiplication，测试隐式乘法转换器
def test_implicit_multiplication():
    # 测试用例字典，键为输入，值为预期输出
    cases = {
        '5x': '5*x',
        'abc': 'a*b*c',
        '3sin(x)': '3*sin(x)',
        '(x+1)(x+2)': '(x+1)*(x+2)',
        '(5 x**2)sin(x)': '(5*x**2)*sin(x)',
        '2 sin(x) cos(x)': '2*sin(x)*cos(x)',
        'pi x': 'pi*x',
        'x pi': 'x*pi',
        'E x': 'E*x',
        'EulerGamma y': 'EulerGamma*y',
        'E pi': 'E*pi',
        'pi (x + 2)': 'pi*(x+2)',
        '(x + 2) pi': '(x+2)*pi',
        'pi sin(x)': 'pi*sin(x)',
    }
    # 转换器列表，包括标准转换和convert_xor转换
    transformations = standard_transformations + (convert_xor,)
    # 第二个转换器列表，包括前者和split_symbols、implicit_multiplication转换
    transformations2 = transformations + (split_symbols, implicit_multiplication)
    # 遍历测试用例字典
    for case in cases:
        # 使用隐式乘法应用转换解析输入的表达式
        implicit = parse_expr(case, transformations=transformations2)
        # 使用标准转换解析预期输出的表达式
        normal = parse_expr(cases[case], transformations=transformations)
        # 断言隐式转换后的表达式与预期输出的表达式相等
        assert(implicit == normal)

    # 以下是一些应该引发异常的隐式乘法应用测试案例
    application = ['sin x', 'cos 2*x', 'sin cos x']
    for case in application:
        # 使用raises检查是否抛出SyntaxError异常
        raises(SyntaxError, lambda: parse_expr(case, transformations=transformations2))
    # 检查是否抛出TypeError异常
    raises(TypeError, lambda: parse_expr('sin**2(x)', transformations=transformations2))


# 定义测试函数test_implicit_application，测试隐式应用转换器
def test_implicit_application():
    # 测试用例字典，键为输入，值为预期输出
    cases = {
        'factorial': 'factorial',
        'sin x': 'sin(x)',
        'tan y**3': 'tan(y**3)',
        'cos 2*x': 'cos(2*x)',
        '(cot)': 'cot',
        'sin cos tan x': 'sin(cos(tan(x)))'
    }
    # 转换器列表，包括标准转换和convert_xor转换
    transformations = standard_transformations + (convert_xor,)
    # 第二个转换器列表，包括前者和implicit_application转换
    transformations2 = transformations + (implicit_application,)
    # 遍历测试用例字典
    for case in cases:
        # 使用隐式应用转换解析输入的表达式
        implicit = parse_expr(case, transformations=transformations2)
        # 使用标准转换解析预期输出的表达式
        normal = parse_expr(cases[case], transformations=transformations)
        # 断言隐式转换后的表达式与预期输出的表达式相等
        assert(implicit == normal), (implicit, normal)

    # 以下是一些应该引发异常的隐式应用测试案例
    multiplication = ['x y', 'x sin x', '2x']
    for case in multiplication:
        # 使用raises检查是否抛出SyntaxError异常
        raises(SyntaxError, lambda: parse_expr(case, transformations=transformations2))
    # 检查是否抛出TypeError异常
    raises(TypeError, lambda: parse_expr('sin**2(x)', transformations=transformations2))


# 定义测试函数test_function_exponentiation，测试函数指数转换器
def test_function_exponentiation():
    # 测试用例字典，键为输入，值为预期输出
    cases = {
        'sin**2(x)': 'sin(x)**2',
        'exp^y(z)': 'exp(z)^y',
        'sin**2(E^(x))': 'sin(E^(x))**2'
    }
    # 转换器列表，包括标准转换和convert_xor转换
    transformations = standard_transformations + (convert_xor,)
    # 第二个转换器列表，包括前者和function_exponentiation转换
    transformations2 = transformations + (function_exponentiation,)
    # 遍历测试用例字典
    for case in cases:
        # 使用函数指数转换解析输入的表达式
        implicit = parse_expr(case, transformations=transformations2)
        # 使用标准转换解析预期输出的表达式
        normal = parse_expr(cases[case], transformations=transformations)
        # 断言转换后的表达式与预期输出的表达式相等
        assert(implicit == normal)
    # 定义一个包含隐式乘法表达式的列表
    other_implicit = ['x y', 'x sin x', '2x', 'sin x',
                      'cos 2*x', 'sin cos x']
    
    # 对于列表中的每个表达式，使用 parse_expr 函数解析它，并期望抛出 SyntaxError 异常
    for case in other_implicit:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))
    
    # 使用 parse_expr 函数分别解析 'x**2' 表达式，并比较结果是否相等
    assert parse_expr('x**2', local_dict={ 'x': sympy.Symbol('x') },
                      transformations=transformations2) == parse_expr('x**2')
def test_symbol_splitting():
    # 默认情况下，希腊字母名称不应拆分（lambda 是关键字，因此跳过它）
    transformations = standard_transformations + (split_symbols,)

    # 定义希腊字母列表
    greek_letters = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
                     'eta', 'theta', 'iota', 'kappa', 'mu', 'nu', 'xi',
                     'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
                     'phi', 'chi', 'psi', 'omega')

    # 对每个希腊字母进行测试，确保解析结果不变
    for letter in greek_letters:
        assert(parse_expr(letter, transformations=transformations) ==
               parse_expr(letter))

    # 确保符号拆分可以解析变量名
    transformations += (implicit_multiplication,)
    local_dict = {'e': sympy.E}

    # 定义测试案例及其预期结果
    cases = {
        'xe': 'E*x',
        'Iy': 'I*y',
        'ee': 'E*E',
    }

    # 对每个测试案例进行断言，验证解析结果是否符合预期
    for case, expected in cases.items():
        assert(parse_expr(case, local_dict=local_dict,
                          transformations=transformations) ==
               parse_expr(expected))

    # 确保自定义拆分函数能正常工作
    def can_split(symbol):
        if symbol not in ('unsplittable', 'names'):
            return _token_splittable(symbol)
        return False

    transformations = standard_transformations
    transformations += (split_symbols_custom(can_split),
                        implicit_multiplication)

    # 对特定的符号进行断言，确保其解析结果不变
    assert(parse_expr('unsplittable', transformations=transformations) ==
           parse_expr('unsplittable'))
    assert(parse_expr('names', transformations=transformations) ==
           parse_expr('names'))
    assert(parse_expr('xy', transformations=transformations) ==
           parse_expr('x*y'))

    # 再次对希腊字母进行测试，确保自定义拆分函数未对其产生影响
    for letter in greek_letters:
        assert(parse_expr(letter, transformations=transformations) ==
               parse_expr(letter))


def test_all_implicit_steps():
    # 这个函数定义在另外的地方，这里未提供其完整定义，因此不需要添加注释
    pass
    cases = {
        '2x': '2*x',  # 隐式乘法
        'x y': 'x*y',  # 显式乘法
        'xy': 'x*y',  # 显式乘法
        'sin x': 'sin(x)',  # 添加括号以明确函数调用
        '2sin x': '2*sin(x)',  # 添加乘号以明确乘法操作
        'x y z': 'x*y*z',  # 多个变量乘法
        'sin(2 * 3x)': 'sin(2 * 3 * x)',  # 显示乘法
        'sin(x) (1 + cos(x))': 'sin(x) * (1 + cos(x))',  # 显示乘法和括号以明确运算顺序
        '(x + 2) sin(x)': '(x + 2) * sin(x)',  # 显示乘法
        '(x + 2) sin x': '(x + 2) * sin(x)',  # 显示乘法
        'sin(sin x)': 'sin(sin(x))',  # 多重函数嵌套
        'sin x!': 'sin(factorial(x))',  # 阶乘函数应用
        'sin x!!': 'sin(factorial2(x))',  # 双阶乘函数应用
        'factorial': 'factorial',  # 不应用函数
        'x sin x': 'x * sin(x)',  # 函数应用和乘法
        'xy sin x': 'x * y * sin(x)',  # 多个变量乘法和函数应用
        '(x+2)(x+3)': '(x + 2) * (x+3)',  # 显示乘法
        'x**2 + 2xy + y**2': 'x**2 + 2 * x * y + y**2',  # 显示乘法以分隔变量乘积
        'pi': 'pi',  # 不更改常数
        'None': 'None',  # 不更改None
        'ln sin x': 'ln(sin(x))',  # 多重隐式函数应用
        'factorial': 'factorial',  # 不应用括号
        'sin x**2': 'sin(x**2)',  # 对指数的隐式函数应用
        'alpha': 'Symbol("alpha")',  # 不分割希腊字母/下标
        'x_2': 'Symbol("x_2")',  # 不分割带下标的变量
        'sin^2 x**2': 'sin(x**2)**2',  # 函数的幂运算
        'sin**3(x)': 'sin(x)**3',  # 函数的幂运算
        '(factorial)': 'factorial',  # 不应用括号
        'tan 3x': 'tan(3*x)',  # 显示乘法
        'sin^2(3*E^(x))': 'sin(3*E**(x))**2',  # 函数的幂运算
        'sin**2(E^(3x))': 'sin(E**(3*x))**2',  # 函数的幂运算
        'sin^2 (3x*E^(x))': 'sin(3*x*E^x)**2',  # 函数的幂运算
        'pi sin x': 'pi*sin(x)',  # 显示乘法
    }
    transformations = standard_transformations + (convert_xor,)  # 转换操作列表
    transformations2 = transformations + (implicit_multiplication_application,)  # 扩展转换操作列表
    for case in cases:  # 遍历所有测试案例
        implicit = parse_expr(case, transformations=transformations2)  # 使用扩展转换进行隐式表达式解析
        normal = parse_expr(cases[case], transformations=transformations)  # 使用标准转换进行表达式解析
        assert(implicit == normal)  # 断言隐式解析和正常解析结果相等
# 定义一个测试函数，用于验证隐式乘法在 sympy 中的应用
def test_no_methods_implicit_multiplication():
    # 创建一个符号变量 u
    u = sympy.Symbol('u')
    # 将隐式乘法转换器添加到标准转换列表中，形成新的转换器列表
    transformations = standard_transformations + \
                      (implicit_multiplication,)
    # 使用指定的转换器列表解析表达式 'x.is_polynomial(x)'
    expr = parse_expr('x.is_polynomial(x)', transformations=transformations)
    # 断言解析后的表达式应该等于 True
    assert expr == True
    # 解析表达式 '(exp(x) / (1 + exp(2x))).subs(exp(x), u)'，并应用指定的转换器列表
    expr = parse_expr('(exp(x) / (1 + exp(2x))).subs(exp(x), u)',
                      transformations=transformations)
    # 断言解析后的表达式应该等于 u / (u**2 + 1)
    assert expr == u/(u**2 + 1)
```