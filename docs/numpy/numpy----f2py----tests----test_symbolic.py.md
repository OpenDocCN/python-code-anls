# `.\numpy\numpy\f2py\tests\test_symbolic.py`

```
# 导入 pytest 模块，用于单元测试
import pytest

# 从 numpy.f2py.symbolic 模块中导入多个符号化相关的类和函数
from numpy.f2py.symbolic import (
    Expr,           # 符号表达式类
    Op,             # 操作符类
    ArithOp,        # 算术操作符类
    Language,       # 语言类
    as_symbol,      # 转换为符号对象函数
    as_number,      # 转换为数字函数
    as_string,      # 转换为字符串函数
    as_array,       # 转换为数组函数
    as_complex,     # 转换为复数函数
    as_terms,       # 转换为项函数
    as_factors,     # 转换为因子函数
    eliminate_quotes,   # 消除引号函数
    insert_quotes,      # 插入引号函数
    fromstring,         # 从字符串创建对象函数
    as_expr,            # 转换为表达式函数
    as_apply,           # 转换为应用函数
    as_numer_denom,     # 转换为分子分母函数
    as_ternary,         # 转换为三元操作符函数
    as_ref,             # 转换为引用函数
    as_deref,           # 转换为解引用函数
    normalize,          # 标准化函数
    as_eq,              # 转换为等于操作符函数
    as_ne,              # 转换为不等于操作符函数
    as_lt,              # 转换为小于操作符函数
    as_gt,              # 转换为大于操作符函数
    as_le,              # 转换为小于等于操作符函数
    as_ge,              # 转换为大于等于操作符函数
)

# 从当前目录下的 util 模块中导入 F2PyTest 类
from . import util


# 定义 TestSymbolic 类，继承自 util.F2PyTest 类
class TestSymbolic(util.F2PyTest):
    
    # 定义单元测试方法 test_eliminate_quotes
    def test_eliminate_quotes(self):
        
        # 定义内部函数 worker，接收字符串 s 作为参数
        def worker(s):
            # 调用 eliminate_quotes 函数处理字符串 s，返回结果保存在 r 和 d 中
            r, d = eliminate_quotes(s)
            # 调用 insert_quotes 函数，使用 r 和 d，得到重新插入引号后的字符串 s1
            s1 = insert_quotes(r, d)
            # 断言重新插入引号后的字符串 s1 应该等于原始字符串 s
            assert s1 == s
        
        # 遍历测试用例中的两种情况：空字符串和 'mykind_' 字符串前缀
        for kind in ["", "mykind_"]:
            # 调用 worker 函数，传入拼接好的测试字符串
            worker(kind + '"1234" // "ABCD"')
            worker(kind + '"1234" // ' + kind + '"ABCD"')
            worker(kind + "\"1234\" // 'ABCD'")
            worker(kind + '"1234" // ' + kind + "'ABCD'")
            worker(kind + '"1\\"2\'AB\'34"')
            worker("a = " + kind + "'1\\'2\"AB\"34'")
    # 定义一个单元测试方法，用于测试符号、数字、复数、字符串、数组、项、因子、三元组和关系表达式的符号化函数
    def test_sanity(self):
        # 创建符号表达式 x, y, z
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        # 断言符号表达式 x 的操作类型为 Op.SYMBOL
        assert x.op == Op.SYMBOL
        # 断言符号表达式 x 的字符串表示为 "Expr(Op.SYMBOL, 'x')"
        assert repr(x) == "Expr(Op.SYMBOL, 'x')"
        # 断言 x 等于自身
        assert x == x
        # 断言 x 不等于 y
        assert x != y
        # 断言 x 的哈希值不为 None
        assert hash(x) is not None

        # 创建整数表达式 n, m
        n = as_number(123)
        m = as_number(456)
        # 断言整数表达式 n 的操作类型为 Op.INTEGER
        assert n.op == Op.INTEGER
        # 断言整数表达式 n 的字符串表示为 "Expr(Op.INTEGER, (123, 4))"
        assert repr(n) == "Expr(Op.INTEGER, (123, 4))"
        # 断言 n 等于自身
        assert n == n
        # 断言 n 不等于 m
        assert n != m
        # 断言 n 的哈希值不为 None
        assert hash(n) is not None

        # 创建实数表达式 fn, fm
        fn = as_number(12.3)
        fm = as_number(45.6)
        # 断言实数表达式 fn 的操作类型为 Op.REAL
        assert fn.op == Op.REAL
        # 断言实数表达式 fn 的字符串表示为 "Expr(Op.REAL, (12.3, 4))"
        assert repr(fn) == "Expr(Op.REAL, (12.3, 4))"
        # 断言 fn 等于自身
        assert fn == fn
        # 断言 fn 不等于 fm
        assert fn != fm
        # 断言 fn 的哈希值不为 None
        assert hash(fn) is not None

        # 创建复数表达式 c, c2
        c = as_complex(1, 2)
        c2 = as_complex(3, 4)
        # 断言复数表达式 c 的操作类型为 Op.COMPLEX
        assert c.op == Op.COMPLEX
        # 断言复数表达式 c 的字符串表示为复杂的表达式
        assert repr(c) == ("Expr(Op.COMPLEX, (Expr(Op.INTEGER, (1, 4)), "
                           "Expr(Op.INTEGER, (2, 4))))")
        # 断言 c 等于自身
        assert c == c
        # 断言 c 不等于 c2
        assert c != c2
        # 断言 c 的哈希值不为 None
        assert hash(c) is not None

        # 创建字符串表达式 s, s2
        s = as_string("'123'")
        s2 = as_string('"ABC"')
        # 断言字符串表达式 s 的操作类型为 Op.STRING
        assert s.op == Op.STRING
        # 断言字符串表达式 s 的字符串表示为 "Expr(Op.STRING, (\"'123'\", 1))"
        assert repr(s) == "Expr(Op.STRING, (\"'123'\", 1))"
        # 断言 s 等于自身
        assert s == s
        # 断言 s 不等于 s2
        assert s != s2

        # 创建数组表达式 a, b
        a = as_array((n, m))
        b = as_array((n, ))
        # 断言数组表达式 a 的操作类型为 Op.ARRAY
        assert a.op == Op.ARRAY
        # 断言数组表达式 a 的字符串表示为复杂的数组表达式
        assert repr(a) == ("Expr(Op.ARRAY, (Expr(Op.INTEGER, (123, 4)), "
                           "Expr(Op.INTEGER, (456, 4))))")
        # 断言 a 等于自身
        assert a == a
        # 断言 a 不等于 b
        assert a != b

        # 创建项表达式 t, u
        t = as_terms(x)
        u = as_terms(y)
        # 断言项表达式 t 的操作类型为 Op.TERMS
        assert t.op == Op.TERMS
        # 断言项表达式 t 的字符串表示为 "Expr(Op.TERMS, {Expr(Op.SYMBOL, 'x'): 1})"
        assert repr(t) == "Expr(Op.TERMS, {Expr(Op.SYMBOL, 'x'): 1})"
        # 断言 t 等于自身
        assert t == t
        # 断言 t 不等于 u
        assert t != u
        # 断言 t 的哈希值不为 None
        assert hash(t) is not None

        # 创建因子表达式 v, w
        v = as_factors(x)
        w = as_factors(y)
        # 断言因子表达式 v 的操作类型为 Op.FACTORS
        assert v.op == Op.FACTORS
        # 断言因子表达式 v 的字符串表示为 "Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1})"
        assert repr(v) == "Expr(Op.FACTORS, {Expr(Op.SYMBOL, 'x'): 1})"
        # 断言 v 等于自身
        assert v == v
        # 断言 w 不等于 v
        assert w != v
        # 断言 v 的哈希值不为 None
        assert hash(v) is not None

        # 创建三元表达式 t, u
        t = as_ternary(x, y, z)
        u = as_ternary(x, z, y)
        # 断言三元表达式 t 的操作类型为 Op.TERNARY
        assert t.op == Op.TERNARY
        # 断言 t 等于自身
        assert t == t
        # 断言 t 不等于 u
        assert t != u
        # 断言 t 的哈希值不为 None
        assert hash(t) is not None

        # 创建等式表达式 e, f
        e = as_eq(x, y)
        f = as_lt(x, y)
        # 断言等式表达式 e 的操作类型为 Op.RELATIONAL
        assert e.op == Op.RELATIONAL
        # 断言 e 等于自身
        assert e == e
        # 断言 e 不等于 f
        assert e != f
        # 断言 e 的哈希值不为 None
        assert hash(e) is not None
    # 定义一个测试函数，用于测试生成字符串的函数
    def test_tostring_fortran(self):
        # 创建符号 'x', 'y', 'z'，以及数字 123 和 456 的表示对象
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        n = as_number(123)
        m = as_number(456)
        # 创建包含数字 n 和 m 的数组对象 a
        a = as_array((n, m))
        # 创建复数对象 c，以数字 n 和 m 作为实部和虚部
        c = as_complex(n, m)

        # 断言表达式对象的字符串表示与预期相符
        assert str(x) == "x"
        assert str(n) == "123"
        assert str(a) == "[123, 456]"
        assert str(c) == "(123, 456)"

        # 测试表达式对象中操作为 TERMS 的情况
        assert str(Expr(Op.TERMS, {x: 1})) == "x"
        assert str(Expr(Op.TERMS, {x: 2})) == "2 * x"
        assert str(Expr(Op.TERMS, {x: -1})) == "-x"
        assert str(Expr(Op.TERMS, {x: -2})) == "-2 * x"
        assert str(Expr(Op.TERMS, {x: 1, y: 1})) == "x + y"
        assert str(Expr(Op.TERMS, {x: -1, y: -1})) == "-x - y"
        assert str(Expr(Op.TERMS, {x: 2, y: 3})) == "2 * x + 3 * y"
        assert str(Expr(Op.TERMS, {x: -2, y: 3})) == "-2 * x + 3 * y"
        assert str(Expr(Op.TERMS, {x: 2, y: -3})) == "2 * x - 3 * y"

        # 测试表达式对象中操作为 FACTORS 的情况
        assert str(Expr(Op.FACTORS, {x: 1})) == "x"
        assert str(Expr(Op.FACTORS, {x: 2})) == "x ** 2"
        assert str(Expr(Op.FACTORS, {x: -1})) == "x ** -1"
        assert str(Expr(Op.FACTORS, {x: -2})) == "x ** -2"
        assert str(Expr(Op.FACTORS, {x: 1, y: 1})) == "x * y"
        assert str(Expr(Op.FACTORS, {x: 2, y: 3})) == "x ** 2 * y ** 3"

        # 测试混合操作：FACTORS 中包含 TERMS 的情况
        v = Expr(Op.FACTORS, {x: 2, Expr(Op.TERMS, {x: 1, y: 1}): 3})
        assert str(v) == "x ** 2 * (x + y) ** 3", str(v)
        v = Expr(Op.FACTORS, {x: 2, Expr(Op.FACTORS, {x: 1, y: 1}): 3})
        assert str(v) == "x ** 2 * (x * y) ** 3", str(v)

        # 测试 APPLY 操作
        assert str(Expr(Op.APPLY, ("f", (), {}))) == "f()"
        assert str(Expr(Op.APPLY, ("f", (x, ), {}))) == "f(x)"
        assert str(Expr(Op.APPLY, ("f", (x, y), {}))) == "f(x, y)"

        # 测试 INDEXING 操作
        assert str(Expr(Op.INDEXING, ("f", x))) == "f[x]"

        # 测试特定的字符串生成函数
        assert str(as_ternary(x, y, z)) == "merge(y, z, x)"
        assert str(as_eq(x, y)) == "x .eq. y"
        assert str(as_ne(x, y)) == "x .ne. y"
        assert str(as_lt(x, y)) == "x .lt. y"
        assert str(as_le(x, y)) == "x .le. y"
        assert str(as_gt(x, y)) == "x .gt. y"
        assert str(as_ge(x, y)) == "x .ge. y"
    # 定义一个测试函数，用于测试将表达式转换为字符串的功能
    def test_tostring_c(self):
        # 设置编程语言为C语言
        language = Language.C
        # 创建符号对象x, y, z，并将字符串与符号关联
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        # 创建数字对象n，并将整数123与之关联
        n = as_number(123)

        # 断言：将Op.FACTORS操作应用于{x: 2}表达式，并将结果转换为字符串后与 "x * x" 比较
        assert Expr(Op.FACTORS, {x: 2}).tostring(language=language) == "x * x"
        # 断言：将Op.FACTORS操作应用于{x + y: 2}表达式，并将结果转换为字符串后与 "(x + y) * (x + y)" 比较
        assert (Expr(Op.FACTORS, {
            x + y: 2
        }).tostring(language=language) == "(x + y) * (x + y)")
        # 断言：将Op.FACTORS操作应用于{x: 12}表达式，并将结果转换为字符串后与 "pow(x, 12)" 比较
        assert Expr(Op.FACTORS, {
            x: 12
        }).tostring(language=language) == "pow(x, 12)"

        # 断言：将ArithOp.DIV操作应用于x和y，并将结果转换为字符串后与 "x / y" 比较
        assert as_apply(ArithOp.DIV, x,
                        y).tostring(language=language) == "x / y"
        # 断言：将ArithOp.DIV操作应用于x和(x + y)，并将结果转换为字符串后与 "x / (x + y)" 比较
        assert (as_apply(ArithOp.DIV, x,
                         x + y).tostring(language=language) == "x / (x + y)")
        # 断言：将ArithOp.DIV操作应用于(x - y)和(x + y)，并将结果转换为字符串后与 "(x - y) / (x + y)" 比较
        assert (as_apply(ArithOp.DIV, x - y, x +
                         y).tostring(language=language) == "(x - y) / (x + y)")
        # 断言：将x + (x - y) / (x + y) + n表达式转换为字符串后与 "123 + x + (x - y) / (x + y)" 比较
        assert (x + (x - y) / (x + y) +
                n).tostring(language=language) == "123 + x + (x - y) / (x + y)"

        # 断言：将x, y, z作为条件表达式的条件部分，转换为字符串后与 "(x?y:z)" 比较
        assert as_ternary(x, y, z).tostring(language=language) == "(x?y:z)"
        # 断言：将x == y关系运算转换为字符串后与 "x == y" 比较
        assert as_eq(x, y).tostring(language=language) == "x == y"
        # 断言：将x != y关系运算转换为字符串后与 "x != y" 比较
        assert as_ne(x, y).tostring(language=language) == "x != y"
        # 断言：将x < y关系运算转换为字符串后与 "x < y" 比较
        assert as_lt(x, y).tostring(language=language) == "x < y"
        # 断言：将x <= y关系运算转换为字符串后与 "x <= y" 比较
        assert as_le(x, y).tostring(language=language) == "x <= y"
        # 断言：将x > y关系运算转换为字符串后与 "x > y" 比较
        assert as_gt(x, y).tostring(language=language) == "x > y"
        # 断言：将x >= y关系运算转换为字符串后与 "x >= y" 比较
        assert as_ge(x, y).tostring(language=language) == "x >= y"
    # 定义一个测试函数，用于测试表达式操作
    def test_operations(self):
        # 创建符号变量 x, y, z
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        # 断言：x + x 应该等于一个包含两个 x 的表达式
        assert x + x == Expr(Op.TERMS, {x: 2})
        # 断言：x - x 应该等于一个整数表达式，表示值为 0
        assert x - x == Expr(Op.INTEGER, (0, 4))
        # 断言：x + y 应该等于一个包含 x 和 y 的表达式
        assert x + y == Expr(Op.TERMS, {x: 1, y: 1})
        # 断言：x - y 应该等于一个包含 x 和 -y 的表达式
        assert x - y == Expr(Op.TERMS, {x: 1, y: -1})
        # 断言：x * x 应该等于一个包含两个 x 的因子表达式
        assert x * x == Expr(Op.FACTORS, {x: 2})
        # 断言：x * y 应该等于一个包含 x 和 y 的因子表达式
        assert x * y == Expr(Op.FACTORS, {x: 1, y: 1})

        # 断言：+x 应该等于 x
        assert +x == x
        # 断言：-x 应该等于一个包含 -1 倍 x 的表达式
        assert -x == Expr(Op.TERMS, {x: -1}), repr(-x)
        # 断言：2 * x 应该等于一个包含 2 倍 x 的表达式
        assert 2 * x == Expr(Op.TERMS, {x: 2})
        # 断言：2 + x 应该等于一个包含 1 个 x 和 2 的表达式
        assert 2 + x == Expr(Op.TERMS, {x: 1, as_number(1): 2})
        # 断言：2 * x + 3 * y 应该等于一个包含 2 倍 x 和 3 倍 y 的表达式
        assert 2 * x + 3 * y == Expr(Op.TERMS, {x: 2, y: 3})
        # 断言：(x + y) * 2 应该等于一个包含 2 倍 (x + y) 的表达式
        assert (x + y) * 2 == Expr(Op.TERMS, {x: 2, y: 2})

        # 断言：x 的平方应该等于一个包含两个 x 的因子表达式
        assert x**2 == Expr(Op.FACTORS, {x: 2})
        # 断言：(x + y) 的平方应该等于一个表达式，包含 x^2, y^2 和 2xy 的系数
        assert (x + y)**2 == Expr(
            Op.TERMS,
            {
                Expr(Op.FACTORS, {x: 2}): 1,
                Expr(Op.FACTORS, {y: 2}): 1,
                Expr(Op.FACTORS, {
                    x: 1,
                    y: 1
                }): 2,
            },
        )
        # 断言：(x + y) * x 应该等于 x^2 + xy 的表达式
        assert (x + y) * x == x**2 + x * y
        # 断言：(x + y) 的平方应该等于 x^2 + 2xy + y^2 的表达式
        assert (x + y)**2 == x**2 + 2 * x * y + y**2
        # 断言：(x + y) 的平方加上 (x - y) 的平方应该等于 2x^2 + 2y^2 的表达式
        assert (x + y)**2 + (x - y)**2 == 2 * x**2 + 2 * y**2
        # 断言：(x + y) * z 应该等于 xz + yz 的表达式
        assert (x + y) * z == x * z + y * z
        # 断言：z * (x + y) 应该等于 zx + zy 的表达式
        assert z * (x + y) == x * z + y * z

        # 断言：(x / 2) 应该等于一个包含 x 和 2 的除法表达式
        assert (x / 2) == as_apply(ArithOp.DIV, x, as_number(2))
        # 断言：(2 * x / 2) 应该等于 x
        assert (2 * x / 2) == x
        # 断言：(3 * x / 2) 应该等于一个包含 3x 和 2 的除法表达式
        assert (3 * x / 2) == as_apply(ArithOp.DIV, 3 * x, as_number(2))
        # 断言：(4 * x / 2) 应该等于 2x 的表达式
        assert (4 * x / 2) == 2 * x
        # 断言：(5 * x / 2) 应该等于一个包含 5x 和 2 的除法表达式
        assert (5 * x / 2) == as_apply(ArithOp.DIV, 5 * x, as_number(2))
        # 断言：(6 * x / 2) 应该等于 3x 的表达式
        assert (6 * x / 2) == 3 * x
        # 断言：((3 * 5) * x / 6) 应该等于一个包含 5x 和 2 的除法表达式
        assert ((3 * 5) * x / 6) == as_apply(ArithOp.DIV, 5 * x, as_number(2))
        # 断言：(30 * x^2 * y^4 / (24 * x^3 * y^3)) 应该等于一个包含 5y 和 4x 的除法表达式
        assert (30 * x**2 * y**4 / (24 * x**3 * y**3)) == as_apply(
            ArithOp.DIV, 5 * y, 4 * x)
        # 断言：((15 * x / 6) / 5) 应该等于一个包含 x 和 2 的除法表达式
        assert ((15 * x / 6) / 5) == as_apply(ArithOp.DIV, x,
                                              as_number(2)), (15 * x / 6) / 5
        # 断言：(x / (5 / x)) 应该等于一个包含 x^2 和 5 的除法表达式
        assert (x / (5 / x)) == as_apply(ArithOp.DIV, x**2, as_number(5))

        # 断言：(x / 2.0) 应该等于一个包含 x 和 0.5 的表达式
        assert (x / 2.0) == Expr(Op.TERMS, {x: 0.5})

        # 创建字符串 s 和 t
        s = as_string('"ABC"')
        t = as_string('"123"')

        # 断言：s // t 应该等于一个包含 "ABC123" 的字符串连接表达式
        assert s // t == Expr(Op.STRING, ('"ABC123"', 1))
        # 断言：s // x 应该等于一个包含 s 和 x 的字符串连接表达式
        assert s // x == Expr(Op.CONCAT, (s, x))
        # 断言：x // s 应该等于一个包含 x 和 s 的字符串连接表达式
        assert x // s == Expr(Op.CONCAT, (x, s))

        # 创建复数 c
        c = as_complex(1.0, 2.0)
        # 断言：-c 应该等于一个包含 -1-2j 的复数表达式
        assert -c == as_complex(-1.0, -2.0)
        # 断言：c + c 应该等于一个包含 (1+2j)*2 的表达式
        assert c + c == as_expr((1 + 2j) * 2)
        # 断言：c * c 应该等于一个
    # 定义测试函数 test_substitute，用于测试符号替换的功能
    def test_substitute(self):
        # 创建符号 x, y, z 并分别初始化为相应的符号对象
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        # 创建数组 a，并将 x, y 作为元素
        a = as_array((x, y))

        # 断言语句：测试替换 x 为 y 的情况
        assert x.substitute({x: y}) == y
        # 断言语句：测试替换 x 为 z 后加法表达式的情况
        assert (x + y).substitute({x: z}) == y + z
        # 断言语句：测试替换 x 为 z 后乘法表达式的情况
        assert (x * y).substitute({x: z}) == y * z
        # 断言语句：测试替换 x 为 z 后幂运算的情况
        assert (x**4).substitute({x: z}) == z**4
        # 断言语句：测试替换 x 为 z 后除法表达式的情况
        assert (x / y).substitute({x: z}) == z / y
        # 断言语句：测试替换 x 为 y + z 后的情况
        assert x.substitute({x: y + z}) == y + z
        # 断言语句：测试数组 a 替换 x 为 y + z 后的情况
        assert a.substitute({x: y + z}) == as_array((y + z, y))

        # 断言语句：测试替换 x 为 y + z 后三元运算符的情况
        assert as_ternary(x, y, z).substitute({x: y + z}) == as_ternary(y + z, y, z)
        # 断言语句：测试替换 x 为 y + z 后等式表达式的情况
        assert as_eq(x, y).substitute({x: y + z}) == as_eq(y + z, y)

    # 定义测试函数 test_traverse，用于测试符号遍历的功能
    def test_traverse(self):
        # 创建符号 x, y, z, f 并分别初始化为相应的符号对象
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")
        f = as_symbol("f")

        # 定义替换函数 replace_visit，用于符号遍历中的替换操作，默认替换为 z
        def replace_visit(s, r=z):
            if s == x:
                return r

        # 断言语句：测试使用替换函数 replace_visit 替换 x 的情况
        assert x.traverse(replace_visit) == z
        # 断言语句：测试替换其他符号时不变的情况
        assert y.traverse(replace_visit) == y
        assert z.traverse(replace_visit) == z
        assert (f(y)).traverse(replace_visit) == f(y)
        assert (f(x)).traverse(replace_visit) == f(z)
        assert (f[y]).traverse(replace_visit) == f[y]
        assert (f[z]).traverse(replace_visit) == f[z]
        # 断言语句：测试复杂表达式替换 x 后的情况
        assert (x + y + z).traverse(replace_visit) == (2 * z + y)
        assert (x + f(y, x - z)).traverse(replace_visit) == (z + f(y, as_number(0)))

        # 定义符号收集函数 collect_symbols，用于遍历中收集符号和函数
        function_symbols = set()
        symbols = set()

        def collect_symbols(s):
            if s.op is Op.APPLY:
                oper = s.data[0]
                function_symbols.add(oper)
                if oper in symbols:
                    symbols.remove(oper)
            elif s.op is Op.SYMBOL and s not in function_symbols:
                symbols.add(s)

        # 断言语句：测试符号收集函数 collect_symbols 的功能
        (x + f(y, x - z)).traverse(collect_symbols)
        assert function_symbols == {f}
        assert symbols == {x, y, z}

        # 定义第二种符号收集函数 collect_symbols2，用于遍历中收集符号
        def collect_symbols2(expr, symbols):
            if expr.op is Op.SYMBOL:
                symbols.add(expr)

        symbols = set()
        # 断言语句：测试第二种符号收集函数 collect_symbols2 的功能
        (x + f(y, x - z)).traverse(collect_symbols2, symbols)
        assert symbols == {x, y, z, f}

        # 定义第三种部分符号收集函数 collect_symbols3，用于部分符号的遍历收集
        def collect_symbols3(expr, symbols):
            if expr.op is Op.APPLY:
                # 跳过对函数调用的遍历
                return expr
            if expr.op is Op.SYMBOL:
                symbols.add(expr)

        symbols = set()
        # 断言语句：测试第三种部分符号收集函数 collect_symbols3 的功能
        (x + f(y, x - z)).traverse(collect_symbols3, symbols)
        assert symbols == {x}
    # 定义测试线性求解的函数
    def test_linear_solve(self):
        # 创建符号变量 x, y, z
        x = as_symbol("x")
        y = as_symbol("y")
        z = as_symbol("z")

        # 测试 x.linear_solve(x)，期望返回 (1, 0)
        assert x.linear_solve(x) == (as_number(1), as_number(0))
        # 测试 (x + 1).linear_solve(x)，期望返回 (1, 1)
        assert (x + 1).linear_solve(x) == (as_number(1), as_number(1))
        # 测试 (2 * x).linear_solve(x)，期望返回 (2, 0)
        assert (2 * x).linear_solve(x) == (as_number(2), as_number(0))
        # 测试 (2 * x + 3).linear_solve(x)，期望返回 (2, 3)
        assert (2 * x + 3).linear_solve(x) == (as_number(2), as_number(3))
        # 测试 as_number(3).linear_solve(x)，期望返回 (0, 3)
        assert as_number(3).linear_solve(x) == (as_number(0), as_number(3))
        # 测试 y.linear_solve(x)，期望返回 (0, y)
        assert y.linear_solve(x) == (as_number(0), y)
        # 测试 (y * z).linear_solve(x)，期望返回 (0, y * z)
        assert (y * z).linear_solve(x) == (as_number(0), y * z)

        # 测试 (x + y).linear_solve(x)，期望返回 (1, y)
        assert (x + y).linear_solve(x) == (as_number(1), y)
        # 测试 (z * x + y).linear_solve(x)，期望返回 (z, y)
        assert (z * x + y).linear_solve(x) == (z, y)
        # 测试 ((z + y) * x + y).linear_solve(x)，期望返回 (z + y, y)
        assert ((z + y) * x + y).linear_solve(x) == (z + y, y)
        # 测试 (z * y * x + y).linear_solve(x)，期望返回 (z * y, y)
        assert (z * y * x + y).linear_solve(x) == (z * y, y)

        # 测试抛出 RuntimeError 的情况：(x * x).linear_solve(x)
        pytest.raises(RuntimeError, lambda: (x * x).linear_solve(x))

    # 定义测试将表达式转换为分子分母形式的函数
    def test_as_numer_denom(self):
        # 创建符号变量 x, y，并设定常数 n = 123
        x = as_symbol("x")
        y = as_symbol("y")
        n = as_number(123)

        # 测试 as_numer_denom(x)，期望返回 (x, 1)
        assert as_numer_denom(x) == (x, as_number(1))
        # 测试 as_numer_denom(x / n)，期望返回 (x, n)
        assert as_numer_denom(x / n) == (x, n)
        # 测试 as_numer_denom(n / x)，期望返回 (n, x)
        assert as_numer_denom(n / x) == (n, x)
        # 测试 as_numer_denom(x / y)，期望返回 (x, y)
        assert as_numer_denom(x / y) == (x, y)
        # 测试 as_numer_denom(x * y)，期望返回 (x * y, 1)
        assert as_numer_denom(x * y) == (x * y, as_number(1))
        # 测试 as_numer_denom(n + x / y)，期望返回 (x + n * y, y)
        assert as_numer_denom(n + x / y) == (x + n * y, y)
        # 测试 as_numer_denom(n + x / (y - x / n))，期望返回 (y * n**2, y * n - x)
        assert as_numer_denom(n + x / (y - x / n)) == (y * n**2, y * n - x)

    # 定义测试多项式的原子项的函数
    def test_polynomial_atoms(self):
        # 创建符号变量 x, y，并设定常数 n = 123
        x = as_symbol("x")
        y = as_symbol("y")
        n = as_number(123)

        # 测试 x.polynomial_atoms()，期望返回 {x}
        assert x.polynomial_atoms() == {x}
        # 测试 n.polynomial_atoms()，期望返回空集合
        assert n.polynomial_atoms() == set()
        # 测试 (y[x]).polynomial_atoms()，期望返回 {y[x]}
        assert (y[x]).polynomial_atoms() == {y[x]}
        # 测试 (y(x)).polynomial_atoms()，期望返回 {y(x)}
        assert (y(x)).polynomial_atoms() == {y(x)}
        # 测试 (y(x) + x).polynomial_atoms()，期望返回 {y(x), x}
        assert (y(x) + x).polynomial_atoms() == {y(x), x}
        # 测试 (y(x) * x[y]).polynomial_atoms()，期望返回 {y(x), x[y]}
        assert (y(x) * x[y]).polynomial_atoms() == {y(x), x[y]}
        # 测试 (y(x)**x).polynomial_atoms()，期望返回 {y(x)}
        assert (y(x)**x).polynomial_atoms() == {y(x)}
```