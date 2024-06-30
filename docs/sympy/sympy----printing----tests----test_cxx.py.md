# `D:\src\scipysrc\sympy\sympy\printing\tests\test_cxx.py`

```
from sympy.core.numbers import Float, Integer, Rational
from sympy.core.symbol import symbols
from sympy.functions import beta, Ei, zeta, Max, Min, sqrt, riemann_xi, frac
from sympy.printing.cxx import CXX98CodePrinter, CXX11CodePrinter, CXX17CodePrinter, cxxcode
from sympy.codegen.cfunctions import log1p

# 定义符号变量
x, y, u, v = symbols('x y u v')

# 定义测试 CXX98CodePrinter 类的函数
def test_CXX98CodePrinter():
    # 断言语句，验证 Max 函数的 C++98 代码输出
    assert CXX98CodePrinter().doprint(Max(x, 3)) in ('std::max(x, 3)', 'std::max(3, x)')
    # 断言语句，验证 Min 函数的 C++98 代码输出
    assert CXX98CodePrinter().doprint(Min(x, 3, sqrt(x))) == 'std::min(3, std::min(x, std::sqrt(x)))'
    # 创建 CXX98CodePrinter 对象
    cxx98printer = CXX98CodePrinter()
    # 断言语句，验证 CXX98CodePrinter 对象的属性和方法
    assert cxx98printer.language == 'C++'
    assert cxx98printer.standard == 'C++98'
    assert 'template' in cxx98printer.reserved_words
    assert 'alignas' not in cxx98printer.reserved_words

# 定义测试 CXX11CodePrinter 类的函数
def test_CXX11CodePrinter():
    # 断言语句，验证 log1p 函数的 C++11 代码输出
    assert CXX11CodePrinter().doprint(log1p(x)) == 'std::log1p(x)'
    # 创建 CXX11CodePrinter 对象
    cxx11printer = CXX11CodePrinter()
    # 断言语句，验证 CXX11CodePrinter 对象的属性和方法
    assert cxx11printer.language == 'C++'
    assert cxx11printer.standard == 'C++11'
    assert 'operator' in cxx11printer.reserved_words
    assert 'noexcept' in cxx11printer.reserved_words
    assert 'concept' not in cxx11printer.reserved_words

# 定义测试子类化 CXX11CodePrinter 类的函数
def test_subclass_print_method():
    # 定义 MyPrinter 类，继承自 CXX11CodePrinter
    class MyPrinter(CXX11CodePrinter):
        # 自定义打印 log1p 函数的方法
        def _print_log1p(self, expr):
            return 'my_library::log1p(%s)' % ', '.join(map(self._print, expr.args))
    # 断言语句，验证 MyPrinter 对象输出的 log1p 函数的代码
    assert MyPrinter().doprint(log1p(x)) == 'my_library::log1p(x)'

# 定义测试带命名空间子类化 CXX11CodePrinter 类的函数
def test_subclass_print_method__ns():
    # 定义 MyPrinter 类，继承自 CXX11CodePrinter，并设定命名空间为 'my_library::'
    class MyPrinter(CXX11CodePrinter):
        _ns = 'my_library::'
    # 创建 CXX11CodePrinter 和 MyPrinter 对象
    p = CXX11CodePrinter()
    myp = MyPrinter()
    # 断言语句，验证不同对象输出 log1p 函数的代码
    assert p.doprint(log1p(x)) == 'std::log1p(x)'
    assert myp.doprint(log1p(x)) == 'my_library::log1p(x)'

# 定义测试 CXX17CodePrinter 类的函数
def test_CXX17CodePrinter():
    # 断言语句，验证 beta 函数的 C++17 代码输出
    assert CXX17CodePrinter().doprint(beta(x, y)) == 'std::beta(x, y)'
    # 断言语句，验证 Ei 函数的 C++17 代码输出
    assert CXX17CodePrinter().doprint(Ei(x)) == 'std::expint(x)'
    # 断言语句，验证 zeta 函数的 C++17 代码输出
    assert CXX17CodePrinter().doprint(zeta(x)) == 'std::riemann_zeta(x)'
    # 断言语句，验证 frac 函数的 C++17 代码输出
    assert CXX17CodePrinter().doprint(frac(x)) == '(x - std::floor(x))'
    # 断言语句，验证 riemann_xi 函数的 C++17 代码输出
    assert CXX17CodePrinter().doprint(riemann_xi(x)) == '((1.0/2.0)*std::pow(M_PI, -1.0/2.0*x)*x*(x - 1)*std::tgamma((1.0/2.0)*x)*std::riemann_zeta(x))'

# 定义测试 cxxcode 函数的函数
def test_cxxcode():
    # 断言语句，验证 cxxcode 函数的输出
    assert sorted(cxxcode(sqrt(x)*.5).split('*')) == sorted(['0.5', 'std::sqrt(x)'])

# 定义测试嵌套 Min 和 Max 函数的函数
def test_cxxcode_nested_minmax():
    # 断言语句，验证 cxxcode 函数嵌套 Min 和 Max 函数的输出
    assert cxxcode(Max(Min(x, y), Min(u, v))) == 'std::max(std::min(u, v), std::min(x, y))'
    assert cxxcode(Min(Max(x, y), Max(u, v))) == 'std::min(std::max(u, v), std::max(x, y))'

# 定义测试子类化 Integer 和 Float 的函数
def test_subclass_Integer_Float():
    # 定义 MyPrinter 类，继承自 CXX17CodePrinter
    class MyPrinter(CXX17CodePrinter):
        # 自定义打印 Integer 函数的方法
        def _print_Integer(self, arg):
            return 'bigInt("%s")' % super()._print_Integer(arg)
        # 自定义打印 Float 函数的方法
        def _print_Float(self, arg):
            rat = Rational(arg)
            return 'bigFloat(%s, %s)' % (
                self._print(Integer(rat.p)),
                self._print(Integer(rat.q))
            )
    # 创建 MyPrinter 对象
    p = MyPrinter()
    # 循环遍历范围为 0 到 12 的整数（共13个数）
    for i in range(13):
        # 使用 p 对象的 doprint 方法打印整数 i 的字符串表示，并使用格式化输出
        assert p.doprint(i) == 'bigInt("%d")' % i
    
    # 断言：使用 p 对象的 doprint 方法打印浮点数 0.5 的字符串表示
    assert p.doprint(Float(0.5)) == 'bigFloat(bigInt("1"), bigInt("2"))'
    
    # 断言：使用 p 对象的 doprint 方法打印 x 的 -1.0 次幂的字符串表示
    assert p.doprint(x**-1.0) == 'bigFloat(bigInt("1"), bigInt("1"))/x'
```