# `D:\src\scipysrc\sympy\sympy\parsing\tests\test_mathematica.py`

```
# 导入必要的符号运算模块和函数
from sympy import sin, Function, symbols, Dummy, Lambda, cos
# 导入 Mathematica 解析器相关的函数和类
from sympy.parsing.mathematica import parse_mathematica, MathematicaParser
# 导入 sympify 函数，用于将字符串转换为 SymPy 表达式
from sympy.core.sympify import sympify
# 导入 SymPy 内置的符号变量
from sympy.abc import n, w, x, y, z
# 导入用于测试的异常处理函数
from sympy.testing.pytest import raises

# 定义一个测试函数，用于验证 Mathematica 格式表达式的解析
def test_mathematica():
    # 定义测试用例字典，键为 Mathematica 格式表达式，值为其 SymPy 等价形式字符串
    d = {
        '- 6x': '-6*x',
        'Sin[x]^2': 'sin(x)**2',
        '2(x-1)': '2*(x-1)',
        '3y+8': '3*y+8',
        'ArcSin[2x+9(4-x)^2]/x': 'asin(2*x+9*(4-x)**2)/x',
        'x+y': 'x+y',
        '355/113': '355/113',
        '2.718281828': '2.718281828',
        'Cos(1/2 * π)': 'Cos(π/2)',
        'Sin[12]': 'sin(12)',
        'Exp[Log[4]]': 'exp(log(4))',
        '(x+1)(x+3)': '(x+1)*(x+3)',
        'Cos[ArcCos[3.6]]': 'cos(acos(3.6))',
        'Cos[x]==Sin[y]': 'Eq(cos(x), sin(y))',
        '2*Sin[x+y]': '2*sin(x+y)',
        'Sin[x]+Cos[y]': 'sin(x)+cos(y)',
        'Sin[Cos[x]]': 'sin(cos(x))',
        '2*Sqrt[x+y]': '2*sqrt(x+y)',   # 用例来自问题编号 4259
        '+Sqrt[2]': 'sqrt(2)',
        '-Sqrt[2]': '-sqrt(2)',
        '-1/Sqrt[2]': '-1/sqrt(2)',
        '-(1/Sqrt[3])': '-(1/sqrt(3))',
        '1/(2*Sqrt[5])': '1/(2*sqrt(5))',
        'Mod[5,3]': 'Mod(5,3)',
        '-Mod[5,3]': '-Mod(5,3)',
        '(x+1)y': '(x+1)*y',
        'x(y+1)': 'x*(y+1)',
        'Sin[x]Cos[y]': 'sin(x)*cos(y)',
        'Sin[x]^2Cos[y]^2': 'sin(x)**2*cos(y)**2',
        'Cos[x]^2(1 - Cos[y]^2)': 'cos(x)**2*(1-cos(y)**2)',
        'x y': 'x*y',
        'x  y': 'x*y',
        '2 x': '2*x',
        'x 8': 'x*8',
        '2 8': '2*8',
        '4.x': '4.*x',
        '4. 3': '4.*3',
        '4. 3.': '4.*3.',
        '1 2 3': '1*2*3',
        ' -  2 *  Sqrt[  2 3 *   ( 1   +  5 ) ]  ': '-2*sqrt(2*3*(1+5))',
        'Log[2,4]': 'log(4,2)',
        'Log[Log[2,4],4]': 'log(4,log(4,2))',
        'Exp[Sqrt[2]^2Log[2, 8]]': 'exp(sqrt(2)**2*log(8,2))',
        'ArcSin[Cos[0]]': 'asin(cos(0))',
        'Log2[16]': 'log(16,2)',
        'Max[1,-2,3,-4]': 'Max(1,-2,3,-4)',
        'Min[1,-2,3]': 'Min(1,-2,3)',
        'Exp[I Pi/2]': 'exp(I*pi/2)',
        'ArcTan[x,y]': 'atan2(y,x)',
        'Pochhammer[x,y]': 'rf(x,y)',
        'ExpIntegralEi[x]': 'Ei(x)',
        'SinIntegral[x]': 'Si(x)',
        'CosIntegral[x]': 'Ci(x)',
        'AiryAi[x]': 'airyai(x)',
        'AiryAiPrime[5]': 'airyaiprime(5)',
        'AiryBi[x]': 'airybi(x)',
        'AiryBiPrime[7]': 'airybiprime(7)',
        'LogIntegral[4]': ' li(4)',
        'PrimePi[7]': 'primepi(7)',
        'Prime[5]': 'prime(5)',
        'PrimeQ[5]': 'isprime(5)',
        'Rational[2,19]': 'Rational(2,19)',    # 测试问题编号 25716 的案例
        }

    # 对每个测试用例进行验证
    for e in d:
        assert parse_mathematica(e) == sympify(d[e])

    # 测试 Lambda 对象不应评估解析后的表达式
    assert parse_mathematica("Sin[#]^2 + Cos[#]^2 &[x]") == sin(x)**2 + cos(x)**2

    # 定义三个虚拟符号变量，用于测试 Lambda 表达式是否正确解析
    d1, d2, d3 = symbols("d1:4", cls=Dummy)
    # 验证 Lambda 表达式解析是否与预期的 Lambda 函数匹配
    assert parse_mathematica("Sin[#] + Cos[#3] &").dummy_eq(Lambda((d1, d2, d3), sin(d1) + cos(d3)))
    # 调用 parse_mathematica 函数，检查是否返回预期的 Lambda 表达式
    assert parse_mathematica("Sin[#^2] &").dummy_eq(Lambda(d1, sin(d1**2)))
    
    # 调用 parse_mathematica 函数，检查是否返回预期的 Lambda 表达式
    assert parse_mathematica("Function[x, x^3]") == Lambda(x, x**3)
    
    # 调用 parse_mathematica 函数，检查是否返回预期的 Lambda 表达式
    assert parse_mathematica("Function[{x, y}, x^2 + y^2]") == Lambda((x, y), x**2 + y**2)
def test_parser_mathematica_tokenizer():
    # 创建 MathematicaParser 的实例
    parser = MathematicaParser()

    # 定义一个 lambda 函数 chain，用于将表达式转换为完整形式列表
    chain = lambda expr: parser._from_tokens_to_fullformlist(parser._from_mathematica_to_tokens(expr))

    # 基本模式
    assert chain("x") == "x"  # 单个变量 x
    assert chain("42") == "42"  # 单个数字 42
    assert chain(".2") == ".2"  # 单个小数点后跟数字的浮点数
    assert chain("+x") == "x"  # 正号前缀的变量 x
    assert chain("-1") == "-1"  # 单个负数 -1
    assert chain("- 3") == "-3"  # 负号后跟空格的数字 -3
    assert chain("α") == "α"  # 希腊字母 α
    assert chain("+Sin[x]") == ["Sin", "x"]  # 正号前缀的 Sin 函数
    assert chain("-Sin[x]") == ["Times", "-1", ["Sin", "x"]]  # 负号前缀的 Sin 函数
    assert chain("x(a+1)") == ["Times", "x", ["Plus", "a", "1"]]  # 包含加法的乘法表达式
    assert chain("(x)") == "x"  # 单个括号内的变量 x
    assert chain("(+x)") == "x"  # 单个带正号的括号内的变量 x
    assert chain("-a") == ["Times", "-1", "a"]  # 单个带负号的变量 a
    assert chain("(-x)") == ["Times", "-1", "x"]  # 单个带负号的括号内的变量 x
    assert chain("(x + y)") == ["Plus", "x", "y"]  # 加法表达式
    assert chain("3 + 4") == ["Plus", "3", "4"]  # 加法操作
    assert chain("a - 3") == ["Plus", "a", "-3"]  # 减法操作
    assert chain("a - b") == ["Plus", "a", ["Times", "-1", "b"]]  # 减法操作转化为加法
    assert chain("7 * 8") == ["Times", "7", "8"]  # 乘法操作
    assert chain("a + b*c") == ["Plus", "a", ["Times", "b", "c"]]  # 加法和乘法的组合
    assert chain("a + b* c* d + 2 * e") == ["Plus", "a", ["Times", "b", "c", "d"], ["Times", "2", "e"]]  # 复合表达式
    assert chain("a / b") == ["Times", "a", ["Power", "b", "-1"]]  # 除法操作

    # 缺失乘号(*)模式
    assert chain("x y") == ["Times", "x", "y"]  # 没有乘号的乘法操作
    assert chain("3 4") == ["Times", "3", "4"]  # 没有乘号的乘法操作
    assert chain("a[b] c") == ["Times", ["a", "b"], "c"]  # 中括号和变量的乘法操作
    assert chain("(x) (y)") == ["Times", "x", "y"]  # 括号内的乘法操作
    assert chain("3 (a)") == ["Times", "3", "a"]  # 数字和括号内的乘法操作
    assert chain("(a) b") == ["Times", "a", "b"]  # 括号内和变量的乘法操作
    assert chain("4.2") == "4.2"  # 浮点数
    assert chain("4 2") == ["Times", "4", "2"]  # 没有乘号的乘法操作
    assert chain("4  2") == ["Times", "4", "2"]  # 多个空格的乘法操作
    assert chain("3 . 4") == ["Dot", "3", "4"]  # 点号操作
    assert chain("4. 2") == ["Times", "4.", "2"]  # 浮点数和数字的乘法操作
    assert chain("x.y") == ["Dot", "x", "y"]  # 点号操作
    assert chain("4.y") == ["Times", "4.", "y"]  # 浮点数和变量的乘法操作
    assert chain("4 .y") == ["Dot", "4", "y"]  # 点号操作
    assert chain("x.4") == ["Times", "x", ".4"]  # 点号操作
    assert chain("x0.3") == ["Times", "x0", ".3"]  # 点号操作
    assert chain("x. 4") == ["Dot", "x", "4"]  # 点号操作

    # 注释
    assert chain("a (* +b *) + c") == ["Plus", "a", "c"]  # 包含注释的加法操作
    assert chain("a (* + b *) + (**)c (* +d *) + e") == ["Plus", "a", "c", "e"]  # 多个注释的表达式
    assert chain("""a + (*
    + b
    *) c + (* d
    *) e
    """) == ["Plus", "a", "c", "e"]  # 跨多行的表达式含有注释

    # 运算符对 + 和 -，* 和 / 是相互关联的：
    # (即混合这些运算符时表达式会被展开)
    assert chain("a*b/c") == ["Times", "a", "b", ["Power", "c", "-1"]]  # 混合运算符的表达式
    assert chain("a/b*c") == ["Times", "a", ["Power", "b", "-1"], "c"]  # 混合运算符的表达式
    assert chain("a+b-c") == ["Plus", "a", "b", ["Times", "-1", "c"]]  # 混合运算符的表达式
    assert chain("a-b+c") == ["Plus", "a", ["Times", "-1", "b"], "c"]  # 混合运算符的表达式
    assert chain("-a + b -c ") == ["Plus", ["Times", "-1", "a"], "b", ["Times", "-1", "c"]]  # 混合运算符的表达式
    assert chain("a/b/c*d") == ["Times", "a", ["Power", "b", "-1"], ["Power", "c", "-1"], "d"]  # 混合运算符的表达式
    assert chain("a/b/c") == ["Times", "a", ["Power", "b", "-1"], ["Power", "c", "-1"]]
    # 测试"/"分割的表达式，返回正确的数据结构
    
    assert chain("a-b-c") == ["Plus", "a", ["Times", "-1", "b"], ["Times", "-1", "c"]]
    # 测试"-"分割的表达式，返回正确的数据结构
    
    assert chain("1/a") == ["Times", "1", ["Power", "a", "-1"]]
    # 测试带数字和"/"的表达式，返回正确的数据结构
    
    assert chain("1/a/b") == ["Times", "1", ["Power", "a", "-1"], ["Power", "b", "-1"]]
    # 测试多个"/"的表达式，返回正确的数据结构
    
    assert chain("-1/a*b") == ["Times", "-1", ["Power", "a", "-1"], "b"]
    # 测试带负号和"*"的表达式，返回正确的数据结构
    
    # Enclosures of various kinds, i.e. ( )  [ ]  [[ ]]  { }
    assert chain("(a + b) + c") == ["Plus", ["Plus", "a", "b"], "c"]
    # 测试带括号的表达式，返回正确的数据结构
    
    assert chain(" a + (b + c) + d ") == ["Plus", "a", ["Plus", "b", "c"], "d"]
    # 测试带括号和空格的表达式，返回正确的数据结构
    
    assert chain("a * (b + c)") == ["Times", "a", ["Plus", "b", "c"]]
    # 测试带乘号和括号的表达式，返回正确的数据结构
    
    assert chain("a b (c d)") == ["Times", "a", "b", ["Times", "c", "d"]]
    # 测试带空格和括号的表达式，返回正确的数据结构
    
    assert chain("{a, b, 2, c}") == ["List", "a", "b", "2", "c"]
    # 测试带花括号的表达式，返回正确的数据结构
    
    assert chain("{a, {b, c}}") == ["List", "a", ["List", "b", "c"]]
    # 测试嵌套花括号的表达式，返回正确的数据结构
    
    assert chain("{{a}}") == ["List", ["List", "a"]]
    # 测试多重嵌套花括号的表达式，返回正确的数据结构
    
    assert chain("a[b, c]") == ["a", "b", "c"]
    # 测试带方括号的表达式，返回正确的数据结构
    
    assert chain("a[[b, c]]") == ["Part", "a", "b", "c"]
    # 测试带双方括号的表达式，返回正确的数据结构
    
    assert chain("a[b[c]]") == ["a", ["b", "c"]]
    # 测试带嵌套方括号的表达式，返回正确的数据结构
    
    assert chain("a[[b, c[[d, {e,f}]]]]") == ["Part", "a", "b", ["Part", "c", "d", ["List", "e", "f"]]]
    # 测试多层嵌套带方括号和花括号的表达式，返回正确的数据结构
    
    assert chain("a[b[[c,d]]]") == ["a", ["Part", "b", "c", "d"]]
    # 测试带嵌套方括号和双方括号的表达式，返回正确的数据结构
    
    assert chain("a[[b[c]]]") == ["Part", "a", ["b", "c"]]
    # 测试带嵌套双方括号和方括号的表达式，返回正确的数据结构
    
    assert chain("a[[b[[c]]]]") == ["Part", "a", ["Part", "b", "c"]]
    # 测试带多层嵌套双方括号和方括号的表达式，返回正确的数据结构
    
    assert chain("a[[b[c[[d]]]]]") == ["Part", "a", ["b", ["Part", "c", "d"]]]
    # 测试带多层嵌套方括号和双方括号的表达式，返回正确的数据结构
    
    assert chain("a[b[[c[d]]]]") == ["a", ["Part", "b", ["c", "d"]]]
    # 测试带嵌套方括号和方括号内再嵌套的表达式，返回正确的数据结构
    
    assert chain("x[[a+1, b+2, c+3]]") == ["Part", "x", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    # 测试带加号的表达式，返回正确的数据结构
    
    assert chain("x[a+1, b+2, c+3]") == ["x", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    # 测试带加号和逗号的表达式，返回正确的数据结构
    
    assert chain("{a+1, b+2, c+3}") == ["List", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    # 测试带加号和花括号的表达式，返回正确的数据结构
    
    # Flat operator:
    assert chain("a*b*c*d*e") == ["Times", "a", "b", "c", "d", "e"]
    # 测试带乘号的表达式，返回正确的数据结构
    
    assert chain("a +b + c+ d+e") == ["Plus", "a", "b", "c", "d", "e"]
    # 测试带加号和空格的表达式，返回正确的数据结构
    
    # Right priority operator:
    assert chain("a^b") == ["Power", "a", "b"]
    # 测试带指数运算符的表达式，返回正确的数据结构
    
    assert chain("a^b^c") == ["Power", "a", ["Power", "b", "c"]]
    # 测试带连续指数运算符的表达式，返回正确的数据结构
    
    assert chain("a^b^c^d") == ["Power", "a", ["Power", "b", ["Power", "c", "d"]]]
    # 测试带多重连续指数运算符的表达式，返回正确的数据结构
    
    # Left priority operator:
    assert chain("a/.b") == ["ReplaceAll", "a", "b"]
    # 测试带替换运算符的表达式，返回正确的数据结构
    
    assert chain("a/.b/.c/.d") == ["ReplaceAll", ["ReplaceAll", ["ReplaceAll", "a", "b"], "c"], "d"]
    # 测试带多重替换运算符的表达式，返回正确的数据结构
    
    assert chain("a//b") == ["a", "b"]
    # 测试带递归替换运算符的表达式，返回正确的数据结构
    
    assert chain("a//b//c") == [["a", "b"], "c"]
    # 测试带多重递归替换运算符的表达式，返回正确的数据结构
    
    assert chain("a//b//c//d") == [[["a", "b"], "c"], "d"]
    # 测试带多重递归替换运算符的表达式，返回正确的数据结构
    
    # Compound expressions
    assert chain("a;b") == ["CompoundExpression", "a", "b"]
    # 测试带分号的表达式，返回正确的数据结构
    
    assert chain("a;") == ["CompoundExpression", "a", "Null"]
    # 测试带分号结尾的表达式，返回正确的数据结构
    
    assert chain("a;b;") == ["CompoundExpression", "a", "b", "Null"]
    # 测试带多个分号的表达式，返回正确的数据结构
    
    assert chain("a[b;c]") == ["a", ["CompoundExpression", "b", "c"]]
    # 测试带分号和方括号的表达式，返回正确的数据结构
    
    assert chain("a[b,c;d,e]") == ["a", "b", ["Compound
    # 断言：对函数 chain 的输入进行测试，并验证返回结果是否符合预期
    assert chain("a[b,c;,d]") == ["a", "b", ["CompoundExpression", "c", "Null"], "d"]

    # New lines
    # 断言：测试包含换行符的表达式，验证函数 chain 的解析结果
    assert chain("a\nb\n") == ["CompoundExpression", "a", "b"]
    assert chain("a\n\nb\n (c \nd)  \n") == ["CompoundExpression", "a", "b", ["Times", "c", "d"]]
    assert chain("\na; b\nc") == ["CompoundExpression", "a", "b", "c"]
    assert chain("a + \nb\n") == ["Plus", "a", "b"]
    assert chain("a\nb; c; d\n e; (f \n g); h + \n i") == ["CompoundExpression", "a", "b", "c", "d", "e", ["Times", "f", "g"], ["Plus", "h", "i"]]
    assert chain("\n{\na\nb; c; d\n e (f \n g); h + \n i\n\n}\n") == ["List", ["CompoundExpression", ["Times", "a", "b"], "c", ["Times", "d", "e", ["Times", "f", "g"]], ["Plus", "h", "i"]]]

    # Patterns
    # 断言：测试模式匹配的表达式，验证函数 chain 的解析结果
    assert chain("y_") == ["Pattern", "y", ["Blank"]]
    assert chain("y_.") == ["Optional", ["Pattern", "y", ["Blank"]]]
    assert chain("y__") == ["Pattern", "y", ["BlankSequence"]]
    assert chain("y___") == ["Pattern", "y", ["BlankNullSequence"]]
    assert chain("a[b_.,c_]") == ["a", ["Optional", ["Pattern", "b", ["Blank"]]], ["Pattern", "c", ["Blank"]]]
    assert chain("b_. c") == ["Times", ["Optional", ["Pattern", "b", ["Blank"]]], "c"]

    # Slots for lambda functions
    # 断言：测试 lambda 函数中的占位符表达式，验证函数 chain 的解析结果
    assert chain("#") == ["Slot", "1"]
    assert chain("#3") == ["Slot", "3"]
    assert chain("#n") == ["Slot", "n"]
    assert chain("##") == ["SlotSequence", "1"]
    assert chain("##a") == ["SlotSequence", "a"]

    # Lambda functions
    # 断言：测试 lambda 函数的表达式，验证函数 chain 的解析结果
    assert chain("x&") == ["Function", "x"]
    assert chain("#&") == ["Function", ["Slot", "1"]]
    assert chain("#+3&") == ["Function", ["Plus", ["Slot", "1"], "3"]]
    assert chain("#1 + #2&") == ["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]]
    assert chain("# + #&") == ["Function", ["Plus", ["Slot", "1"], ["Slot", "1"]]]
    assert chain("#&[x]") == [["Function", ["Slot", "1"]], "x"]
    assert chain("#1 + #2 & [x, y]") == [["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]], "x", "y"]
    assert chain("#1^2#2^3&") == ["Function", ["Times", ["Power", ["Slot", "1"], "2"], ["Power", ["Slot", "2"], "3"]]]

    # Strings inside Mathematica expressions:
    # 断言：测试 Mathematica 表达式中的字符串，验证函数 chain 的解析结果
    assert chain('"abc"') == ["_Str", "abc"]
    assert chain('"a\\"b"') == ["_Str", 'a"b']
    # This expression does not make sense mathematically, it's just testing the parser:
    assert chain('x + "abc" ^ 3') == ["Plus", "x", ["Power", ["_Str", "abc"], "3"]]
    assert chain('"a (* b *) c"') == ["_Str", "a (* b *) c"]
    assert chain('"a" (* b *) ') == ["_Str", "a"]
    assert chain('"a [ b] "') == ["_Str", "a [ b] "]
    raises(SyntaxError, lambda: chain('"'))
    raises(SyntaxError, lambda: chain('"\\"'))
    raises(SyntaxError, lambda: chain('"abc'))
    raises(SyntaxError, lambda: chain('"abc\\"def'))

    # Invalid expressions:
    # 断言：测试无效的表达式，验证函数 chain 是否正确地抛出 SyntaxError
    raises(SyntaxError, lambda: chain("(,"))
    raises(SyntaxError, lambda: chain("()"))
    raises(SyntaxError, lambda: chain("a (* b"))
# 定义一个测试函数，用于测试 MathematicaParser 类的功能
def test_parser_mathematica_exp_alt():
    # 创建 MathematicaParser 的实例对象
    parser = MathematicaParser()

    # 定义一个 Lambda 函数，用于将表达式转换为 FullFormSympy 格式
    convert_chain2 = lambda expr: parser._from_fullformlist_to_fullformsympy(parser._from_fullform_to_fullformlist(expr))
    # 定义另一个 Lambda 函数，用于将 FullFormSympy 格式转换为 SymPy 表达式
    convert_chain3 = lambda expr: parser._from_fullformsympy_to_sympy(convert_chain2(expr))

    # 使用 symbols 函数创建 SymPy 符号 Sin, Times, Plus, Power
    Sin, Times, Plus, Power = symbols("Sin Times Plus Power", cls=Function)

    # 定义几个 Mathematica 表达式的字符串表示形式
    full_form1 = "Sin[Times[x, y]]"
    full_form2 = "Plus[Times[x, y], z]"
    full_form3 = "Sin[Times[x, Plus[y, z], Power[w, n]]]]"
    full_form4 = "Rational[Rational[x, y], z]"

    # 断言：将字符串表达式转换为 FullFormList 格式的列表
    assert parser._from_fullform_to_fullformlist(full_form1) == ["Sin", ["Times", "x", "y"]]
    assert parser._from_fullform_to_fullformlist(full_form2) == ["Plus", ["Times", "x", "y"], "z"]
    assert parser._from_fullform_to_fullformlist(full_form3) == ["Sin", ["Times", "x", ["Plus", "y", "z"], ["Power", "w", "n"]]]
    assert parser._from_fullform_to_fullformlist(full_form4) == ["Rational", ["Rational", "x", "y"], "z"]

    # 断言：将字符串表达式转换为 SymPy 表达式
    assert convert_chain2(full_form1) == Sin(Times(x, y))
    assert convert_chain2(full_form2) == Plus(Times(x, y), z)
    assert convert_chain2(full_form3) == Sin(Times(x, Plus(y, z), Power(w, n)))

    # 断言：将字符串表达式转换为最终的 SymPy 表达式（包括数学函数的转换）
    assert convert_chain3(full_form1) == sin(x*y)
    assert convert_chain3(full_form2) == x*y + z
    assert convert_chain3(full_form3) == sin(x*(y + z)*w**n)
```