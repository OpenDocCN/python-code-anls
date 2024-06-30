# `D:\src\scipysrc\sympy\sympy\codegen\tests\test_ast.py`

```
import math  # 导入数学函数库
from sympy.core.containers import Tuple  # 导入元组类
from sympy.core.numbers import nan, oo, Float, Integer  # 导入特定的数值类
from sympy.core.relational import Lt  # 导入小于符号类
from sympy.core.symbol import symbols, Symbol  # 导入符号和符号类
from sympy.functions.elementary.trigonometric import sin  # 导入正弦函数
from sympy.matrices.dense import Matrix  # 导入密集矩阵类
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号类
from sympy.sets.fancysets import Range  # 导入范围类
from sympy.tensor.indexed import Idx, IndexedBase  # 导入索引和索引基类
from sympy.testing.pytest import raises  # 导入测试框架的异常抛出函数

# 导入代码生成抽象语法树（AST）所需的类和函数
from sympy.codegen.ast import (
    Assignment, Attribute, aug_assign, CodeBlock, For, Type, Variable, Pointer, Declaration,
    AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
    DivAugmentedAssignment, ModAugmentedAssignment, value_const, pointer_const,
    integer, real, complex_, int8, uint8, float16 as f16, float32 as f32,
    float64 as f64, float80 as f80, float128 as f128, complex64 as c64, complex128 as c128,
    While, Scope, String, Print, QuotedString, FunctionPrototype, FunctionDefinition, Return,
    FunctionCall, untyped, IntBaseType, intc, Node, none, NoneToken, Token, Comment
)

x, y, z, t, x0, x1, x2, a, b = symbols("x, y, z, t, x0, x1, x2, a, b")  # 定义符号变量
n = symbols("n", integer=True)  # 定义整数符号变量
A = MatrixSymbol('A', 3, 1)  # 定义3行1列的矩阵符号
mat = Matrix([1, 2, 3])  # 定义具体的矩阵
B = IndexedBase('B')  # 定义索引基类
i = Idx("i", n)  # 定义整数索引
A22 = MatrixSymbol('A22', 2, 2)  # 定义2行2列的矩阵符号
B22 = MatrixSymbol('B22', 2, 2)  # 定义2行2列的矩阵符号


def test_Assignment():
    # 这里我们执行一些操作以确保它们不会引发错误
    Assignment(x, y)  # 赋值表达式，将y赋给x
    Assignment(x, 0)  # 赋值表达式，将0赋给x
    Assignment(A, mat)  # 赋值表达式，将mat赋给A
    Assignment(A[1, 0], 0)  # 赋值表达式，将0赋给A的第(1, 0)位置
    Assignment(A[1, 0], x)  # 赋值表达式，将x赋给A的第(1, 0)位置
    Assignment(B[i], x)  # 赋值表达式，将x赋给B的第i个位置
    Assignment(B[i], 0)  # 赋值表达式，将0赋给B的第i个位置
    a = Assignment(x, y)  # 赋值表达式，将y赋给x，并将结果赋给变量a
    assert a.func(*a.args) == a  # 断言表达式，确保函数调用的结果与a相等
    assert a.op == ':='  # 断言表达式，确保操作符为赋值符号':='

    # 这里我们测试一些错误情况
    # 矩阵赋给标量
    raises(ValueError, lambda: Assignment(B[i], A))
    raises(ValueError, lambda: Assignment(B[i], mat))
    raises(ValueError, lambda: Assignment(x, mat))
    raises(ValueError, lambda: Assignment(x, A))
    raises(ValueError, lambda: Assignment(A[1, 0], mat))
    # 标量赋给矩阵
    raises(ValueError, lambda: Assignment(A, x))
    raises(ValueError, lambda: Assignment(A, 0))
    # 非原子左侧
    raises(TypeError, lambda: Assignment(mat, A))
    raises(TypeError, lambda: Assignment(0, x))
    raises(TypeError, lambda: Assignment(x*x, 1))
    raises(TypeError, lambda: Assignment(A + A, mat))
    raises(TypeError, lambda: Assignment(B, 0))


def test_AugAssign():
    # 这里我们执行一些操作以确保它们不会引发错误
    aug_assign(x, '+', y)  # 增强赋值表达式，x = x + y
    aug_assign(x, '+', 0)  # 增强赋值表达式，x = x + 0
    aug_assign(A, '+', mat)  # 增强赋值表达式，A = A + mat
    aug_assign(A[1, 0], '+', 0)  # 增强赋值表达式，A[1, 0] = A[1, 0] + 0
    aug_assign(A[1, 0], '+', x)  # 增强赋值表达式，A[1, 0] = A[1, 0] + x
    aug_assign(B[i], '+', x)  # 增强赋值表达式，B[i] = B[i] + x
    aug_assign(B[i], '+', 0)  # 增强赋值表达式，B[i] = B[i] + 0

    # 通过增强赋值表达式和构造函数进行创建的检查
    # 遍历一个包含二元操作符和对应类的列表
    for binop, cls in [
            ('+', AddAugmentedAssignment),
            ('-', SubAugmentedAssignment),
            ('*', MulAugmentedAssignment),
            ('/', DivAugmentedAssignment),
            ('%', ModAugmentedAssignment),
        ]:
        # 使用 aug_assign 函数执行增强赋值操作，并将结果存储在变量 a 中
        a = aug_assign(x, binop, y)
        # 使用给定的类创建对象，表示相同的增强赋值操作，并将结果存储在变量 b 中
        b = cls(x, y)
        # 断言对象 a 调用其 func 方法返回的结果等于 a 自身，并且等于对象 b
        assert a.func(*a.args) == a == b
        # 断言对象 a 的 binop 属性等于当前的二元操作符
        assert a.binop == binop
        # 断言对象 a 的 op 属性等于当前的二元操作符加上 '='
        assert a.op == binop + '='

    # 在这里测试以显示错误情况
    # 矩阵到标量的赋值操作，预期会引发 ValueError
    raises(ValueError, lambda: aug_assign(B[i], '+', A))
    raises(ValueError, lambda: aug_assign(B[i], '+', mat))
    raises(ValueError, lambda: aug_assign(x, '+', mat))
    raises(ValueError, lambda: aug_assign(x, '+', A))
    raises(ValueError, lambda: aug_assign(A[1, 0], '+', mat))
    # 标量到矩阵的赋值操作，预期会引发 ValueError
    raises(ValueError, lambda: aug_assign(A, '+', x))
    raises(ValueError, lambda: aug_assign(A, '+', 0))
    # 左操作数不是原子类型的赋值操作，预期会引发 TypeError
    raises(TypeError, lambda: aug_assign(mat, '+', A))
    raises(TypeError, lambda: aug_assign(0, '+', x))
    raises(TypeError, lambda: aug_assign(x * x, '+', 1))
    raises(TypeError, lambda: aug_assign(A + A, '+', mat))
    raises(TypeError, lambda: aug_assign(B, '+', 0))
# 定义一个测试函数，用于测试赋值类的打印功能
def test_Assignment_printing():
    # 定义赋值类的列表，包括基本赋值和增强赋值运算符的类
    assignment_classes = [
        Assignment,
        AddAugmentedAssignment,
        SubAugmentedAssignment,
        MulAugmentedAssignment,
        DivAugmentedAssignment,
        ModAugmentedAssignment,
    ]
    # 定义一些赋值对，每个对由左操作数和右操作数组成
    pairs = [
        (x, 2 * y + 2),
        (B[i], x),
        (A22, B22),
        (A[0, 0], x),
    ]

    # 对于每个赋值类和每个赋值对，创建赋值对象，然后断言其打印形式与预期相符
    for cls in assignment_classes:
        for lhs, rhs in pairs:
            a = cls(lhs, rhs)
            assert repr(a) == '%s(%s, %s)' % (cls.__name__, repr(lhs), repr(rhs))


# 定义测试函数，用于测试代码块类的功能
def test_CodeBlock():
    # 创建一个代码块对象c，包含两个赋值操作
    c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    # 断言执行c的函数功能与c本身相同
    assert c.func(*c.args) == c

    # 断言c的左操作数集合等于(x, y)
    assert c.left_hand_sides == Tuple(x, y)
    # 断言c的右操作数集合等于(1, x + 1)
    assert c.right_hand_sides == Tuple(1, x + 1)


# 定义测试函数，测试代码块类的拓扑排序功能
def test_CodeBlock_topological_sort():
    # 定义赋值操作列表
    assignments = [
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(t, x),
        Assignment(y, 2),
    ]

    # 定义按照拓扑排序后的赋值操作列表
    ordered_assignments = [
        # 注意，不相关的z=1和y=2按照原顺序保留
        Assignment(z, 1),
        Assignment(y, 2),
        Assignment(x, y + z),
        Assignment(t, x),
    ]
    # 对赋值操作列表进行拓扑排序，预期结果应与ordered_assignments相同
    c1 = CodeBlock.topological_sort(assignments)
    assert c1 == CodeBlock(*ordered_assignments)

    # 测试循环依赖的情况
    invalid_assignments = [
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(y, x),
        Assignment(y, 2),
    ]
    # 应当抛出值错误，因为存在循环依赖
    raises(ValueError, lambda: CodeBlock.topological_sort(invalid_assignments))

    # 测试自由符号的情况
    free_assignments = [
        Assignment(x, y + z),
        Assignment(z, a * b),
        Assignment(t, x),
        Assignment(y, b + 3),
    ]
    # 定义按照拓扑排序后的自由符号赋值操作列表
    free_assignments_ordered = [
        Assignment(z, a * b),
        Assignment(y, b + 3),
        Assignment(x, y + z),
        Assignment(t, x),
    ]
    # 对自由符号赋值操作列表进行拓扑排序，预期结果应与free_assignments_ordered相同
    c2 = CodeBlock.topological_sort(free_assignments)
    assert c2 == CodeBlock(*free_assignments_ordered)


# 定义测试函数，测试代码块类的自由符号集合功能
def test_CodeBlock_free_symbols():
    # 创建代码块c1，包含四个赋值操作
    c1 = CodeBlock(
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(t, x),
        Assignment(y, 2),
    )
    # 断言c1的自由符号集合为空集
    assert c1.free_symbols == set()

    # 创建代码块c2，包含四个带有自由符号a和b的赋值操作
    c2 = CodeBlock(
        Assignment(x, y + z),
        Assignment(z, a * b),
        Assignment(t, x),
        Assignment(y, b + 3),
    )
    # 断言c2的自由符号集合为{a, b}
    assert c2.free_symbols == {a, b}


# 定义测试函数，测试代码块类的公共子表达式消除功能
def test_CodeBlock_cse():
    # 创建代码块c1，包含四个赋值操作，其中有两个赋值操作引用相同的子表达式sin(y)
    c1 = CodeBlock(
        Assignment(y, 1),
        Assignment(x, sin(y)),
        Assignment(z, sin(y)),
        Assignment(t, x*z),
    )
    # 断言c1进行公共子表达式消除后的结果
    assert c1.cse() == CodeBlock(
        Assignment(y, 1),
        Assignment(x0, sin(y)),
        Assignment(x, x0),
        Assignment(z, x0),
        Assignment(t, x*z),
    )

    # 测试多次对同一符号进行赋值的情况，应当抛出未实现错误
    raises(NotImplementedError, lambda: CodeBlock(
        Assignment(x, 1),
        Assignment(y, 1), Assignment(y, 2)
    ).cse())

    # 检查自动生成的符号不与现有符号冲突
    # 创建一个代码块对象 `c2`，其中包含多个赋值语句
    c2 = CodeBlock(
        # 将 `sin(y) + 1` 的结果赋给变量 `x0`
        Assignment(x0, sin(y) + 1),
        # 将 `2 * sin(y)` 的结果赋给变量 `x1`
        Assignment(x1, 2 * sin(y)),
        # 将 `x * y` 的结果赋给变量 `z`
        Assignment(z, x * y),
    )
    
    # 断言检查 `c2` 对象调用共享子表达式消除 (Common Subexpression Elimination, CSE) 方法后的结果
    assert c2.cse() == CodeBlock(
        # 将 `sin(y)` 的结果赋给新变量 `x2`
        Assignment(x2, sin(y)),
        # 使用 `x2 + 1` 的结果更新 `x0` 的赋值语句
        Assignment(x0, x2 + 1),
        # 使用 `2 * x2` 的结果更新 `x1` 的赋值语句
        Assignment(x1, 2 * x2),
        # 保持不变的赋值语句，将 `x * y` 的结果赋给 `z`
        Assignment(z, x * y),
    )
# 定义一个名为 test_CodeBlock_cse__issue_14118 的测试函数，用于测试特定问题的代码块
def test_CodeBlock_cse__issue_14118():
    # 添加一个注释，指向 GitHub 上的特定问题链接，以便查看更多信息
    # see https://github.com/sympy/sympy/issues/14118
    # 创建一个 CodeBlock 对象 c，其中包含两个 Assignment 对象
    c = CodeBlock(
        Assignment(A22, Matrix([[x, sin(y)],[3, 4]])),
        Assignment(B22, Matrix([[sin(y), 2*sin(y)], [sin(y)**2, 7]]))
    )
    # 使用 cse 方法对 CodeBlock 进行公共子表达式消除，预期返回另一个 CodeBlock 对象
    assert c.cse() == CodeBlock(
        Assignment(x0, sin(y)),
        Assignment(A22, Matrix([[x, x0],[3, 4]])),
        Assignment(B22, Matrix([[x0, 2*x0], [x0**2, 7]]))
    )

# 定义一个名为 test_For 的测试函数，用于测试 For 类的功能
def test_For():
    # 创建一个 For 对象 f，其迭代范围是 Range(0, 3)，包含两个操作
    f = For(n, Range(0, 3), (Assignment(A[n, 0], x + n), aug_assign(x, '+', y)))
    # 更新 For 对象 f，将迭代范围改为 (1, 2, 3, 4, 5)，仅包含一个操作
    f = For(n, (1, 2, 3, 4, 5), (Assignment(A[n, 0], x + n),))
    # 断言更新后的 f 对象与其调用 func 方法后返回的对象相等
    assert f.func(*f.args) == f
    # 断言创建 For 对象时，传入非法迭代对象 x 时会抛出 TypeError 异常
    raises(TypeError, lambda: For(n, x, (x + y,)))


# 定义一个名为 test_none 的测试函数，用于测试 none 对象的行为
def test_none():
    # 断言 none 对象具有 is_Atom 属性
    assert none.is_Atom
    # 断言 none 对象与自身相等
    assert none == none
    # 定义一个类 Foo，继承自 Token
    class Foo(Token):
        pass
    # 创建 Foo 类的实例 foo
    foo = Foo()
    # 断言 foo 对象与 none 不相等
    assert foo != none
    # 断言 none 对象与 Python 内置的 None 相等
    assert none == None
    # 断言 none 对象与 NoneToken() 相等
    assert none == NoneToken()
    # 断言调用 none 对象的 func 方法后返回的结果仍为 none
    assert none.func(*none.args) == none


# 定义一个名为 test_String 的测试函数，用于测试 String 对象的行为
def test_String():
    # 创建一个 String 对象 st，其内容为 'foobar'
    st = String('foobar')
    # 断言 st 对象具有 is_Atom 属性
    assert st.is_Atom
    # 断言 st 对象与另一个包含相同内容的 String 对象相等
    assert st == String('foobar')
    # 断言 st 对象的 text 属性为 'foobar'
    assert st.text == 'foobar'
    # 断言调用 st 对象的 func 方法并传入其 kwargs() 后返回的结果与 st 相等
    assert st.func(**st.kwargs()) == st
    # 断言调用 st 对象的 func 方法并传入其 args 后返回的结果与 st 相等

    # 定义一个名为 Signifier 的子类，继承自 String
    class Signifier(String):
        pass
    # 创建一个 Signifier 对象 si，其内容为 'foobar'
    si = Signifier('foobar')
    # 断言 si 对象与 st 不相等
    assert si != st
    # 断言 si 对象的 text 属性与 st 对象的 text 属性相等
    assert si.text == st.text
    # 创建一个 String 对象 s，其内容为 'foo'
    s = String('foo')
    # 断言将 s 对象转换为字符串后结果为 'foo'
    assert str(s) == 'foo'
    # 断言将 s 对象转换为其表达式形式后结果为 "String('foo')"
    assert repr(s) == "String('foo')"


# 定义一个名为 test_Comment 的测试函数，用于测试 Comment 对象的行为
def test_Comment():
    # 创建一个 Comment 对象 c，其内容为 'foobar'
    c = Comment('foobar')
    # 断言 c 对象的 text 属性为 'foobar'
    assert c.text == 'foobar'
    # 断言将 c 对象转换为字符串后结果为 'foobar'


# 定义一个名为 test_Node 的测试函数，用于测试 Node 对象的行为
def test_Node():
    # 创建一个 Node 对象 n
    n = Node()
    # 断言两个 Node 对象 n 相等
    assert n == Node()
    # 断言调用 n 对象的 func 方法后返回的结果仍为 n


# 定义一个名为 test_Type 的测试函数，用于测试 Type 对象的行为
def test_Type():
    # 创建一个 Type 对象 t，其名称为 'MyType'
    t = Type('MyType')
    # 断言 t 对象的 args 属性长度为 1
    assert len(t.args) == 1
    # 断言 t 对象的 name 属性为 String('MyType')
    assert t.name == String('MyType')
    # 断言将 t 对象转换为字符串后结果为 'MyType'
    assert str(t) == 'MyType'
    # 断言将 t 对象转换为其表达式形式后结果为 "Type(String('MyType'))"
    assert repr(t) == "Type(String('MyType'))"
    # 断言 Type(t) 与 t 对象相等
    assert Type(t) == t
    # 断言调用 t 对象的 func 方法并传入其 args 后返回的结果与 t 相等
    assert t.func(*t.args) == t
    # 创建两个 Type 对象 t1 和 t2，名称分别为 't1' 和 't2'
    t1 = Type('t1')
    t2 = Type('t2')
    # 断言 t1 对象与 t2 对象不相等
    assert t1 != t2
    # 断言 t1 对象与自身相等，t2 对象与自身相等
    assert t1 == t1 and t2 == t2
    # 创建另一个名称为 't1' 的 Type 对象 t1b
    t1b = Type('t1')
    # 断言 t1 对象与 t1b 对象相等
    assert t1 == t1b
    # 断言 t2 对象与 t1b 对象不相等


# 定义一个名为 test_Type__from_expr 的测试函数，用于测试 Type 类的 from_expr 方法
def test_Type__from_expr():
    # 断言 Type.from_expr(i) 返回 integer
    assert Type.from_expr(i) == integer
    # 创建一个实数符号 u
    u = symbols('u', real=True)
    # 断言 Type.from_expr(u) 返回 real
    assert Type.from_expr(u) == real
    # 断言 Type.from_expr(n) 返回 integer
    assert Type.from_expr(n) == integer
    # 断言 Type.from_expr(3) 返回 integer
    assert Type.from_expr(3) == integer
    # 断言 Type.from_expr(3.0) 返回 real
    assert Type.from_expr(3.0) == real
    # 断言 Type.from_expr(3+1j) 返回 complex_
    assert Type.from_expr(3+1j) == complex_
    # 断言调用 Type.from_expr(sum) 时会抛出 ValueError 异常
    raises(ValueError, lambda: Type.from_expr(sum))


# 定义一个名为 test_Type__cast_check__integers 的测试函数，用于测试整数类型的 cast_check 方法
def test_Type__cast_check__integers():
    # 断言调用 integer.cast_check(3.5) 时会抛出 ValueError 异常
    raises(ValueError, lambda: integer.cast_check(3.5))
    # 断言调用 integer.cast_check('3')
    # 创建名为 `noexcept` 的属性对象，表示具有 `noexcept` 属性
    noexcept = Attribute('noexcept')
    # 断言 `noexcept` 对象与另一个具有相同属性的 `Attribute` 对象相等
    assert noexcept == Attribute('noexcept')
    # 创建名为 `alignas16` 的属性对象，指定 `alignas` 属性为 16
    alignas16 = Attribute('alignas', [16])
    # 创建名为 `alignas32` 的属性对象，指定 `alignas` 属性为 32
    alignas32 = Attribute('alignas', [32])
    # 断言 `alignas16` 对象与 `alignas32` 对象不相等
    assert alignas16 != alignas32
    # 断言调用 `alignas16` 对象的方法并传入其参数与 `alignas16` 对象本身相等
    assert alignas16.func(*alignas16.args) == alignas16
# 定义一个名为 test_Variable 的测试函数
def test_Variable():
    # 创建一个变量 v，其符号为 x，类型为 real
    v = Variable(x, type=real)
    # 断言 v 等于 Variable(v)
    assert v == Variable(v)
    # 断言 v 等于 Variable('x', type=real)
    assert v == Variable('x', type=real)
    # 断言 v 的符号为 x
    assert v.symbol == x
    # 断言 v 的类型为 real
    assert v.type == real
    # 断言 value_const 不在 v 的属性中
    assert value_const not in v.attrs
    # 断言 v 的 func 方法返回与 v 相同的对象
    assert v.func(*v.args) == v
    # 断言 str(v) 的输出为 'Variable(x, type=real)'
    assert str(v) == 'Variable(x, type=real)'

    # 创建一个变量 w，其符号为 y，类型为 f32，属性包含 value_const
    w = Variable(y, f32, attrs={value_const})
    # 断言 w 的符号为 y
    assert w.symbol == y
    # 断言 w 的类型为 f32
    assert w.type == f32
    # 断言 value_const 在 w 的属性中
    assert value_const in w.attrs
    # 断言 w 的 func 方法返回与 w 相同的对象
    assert w.func(*w.args) == w

    # 创建一个变量 v_n，其符号为 n，类型为 Type.from_expr(n) 的结果
    v_n = Variable(n, type=Type.from_expr(n))
    # 断言 v_n 的类型为 integer
    assert v_n.type == integer
    # 断言 v_n 的 func 方法返回与 v_n 相同的对象
    assert v_n.func(*v_n.args) == v_n

    # 创建一个变量 v_i，其符号为 i，类型为 Type.from_expr(n) 的结果
    v_i = Variable(i, type=Type.from_expr(n))
    # 断言 v_i 的类型为 integer
    assert v_i.type == integer
    # 断言 v_i 不等于 v_n
    assert v_i != v_n

    # 创建一个变量 a_i，通过 Variable.deduced(i) 方法推导得到其类型为 integer
    a_i = Variable.deduced(i)
    # 断言 a_i 的类型为 integer
    assert a_i.type == integer
    # 通过 Variable.deduced 方法推导符号为 'x' 和 real=True 的变量类型为 real
    assert Variable.deduced(Symbol('x', real=True)).type == real
    # 断言 a_i 的 func 方法返回与 a_i 相同的对象
    assert a_i.func(*a_i.args) == a_i

    # 创建一个变量 v_n2，通过 Variable.deduced(n, value=3.5, cast_check=False) 方法推导得到
    v_n2 = Variable.deduced(n, value=3.5, cast_check=False)
    # 断言 v_n2 的 func 方法返回与 v_n2 相同的对象
    assert v_n2.func(*v_n2.args) == v_n2
    # 断言 v_n2 的值与 3.5 的差的绝对值小于 1e-15
    assert abs(v_n2.value - 3.5) < 1e-15
    # 断言当 value=3.5 时，调用 Variable.deduced(n, value=3.5, cast_check=True) 会抛出 ValueError
    raises(ValueError, lambda: Variable.deduced(n, value=3.5, cast_check=True))

    # 创建一个变量 v_n3，通过 Variable.deduced(n) 方法推导得到其类型为 integer
    v_n3 = Variable.deduced(n)
    # 断言 v_n3 的类型为 integer
    assert v_n3.type == integer
    # 断言 str(v_n3) 的输出为 'Variable(n, type=integer)'
    assert str(v_n3) == 'Variable(n, type=integer)'
    # 断言当 value=3 时，调用 Variable.deduced(z, value=3).type 得到 integer 类型
    assert Variable.deduced(z, value=3).type == integer
    # 断言当 value=3.0 时，调用 Variable.deduced(z, value=3.0).type 得到 real 类型
    assert Variable.deduced(z, value=3.0).type == real
    # 断言当 value=3.0+1j 时，调用 Variable.deduced(z, value=3.0+1j).type 得到 complex_ 类型
    assert Variable.deduced(z, value=3.0+1j).type == complex_


# 定义一个名为 test_Pointer 的测试函数
def test_Pointer():
    # 创建一个指针 p，其符号为 x，类型为 untyped
    p = Pointer(x)
    # 断言 p 的符号为 x
    assert p.symbol == x
    # 断言 p 的类型为 untyped
    assert p.type == untyped
    # 断言 value_const 不在 p 的属性中
    assert value_const not in p.attrs
    # 断言 pointer_const 不在 p 的属性中
    assert pointer_const not in p.attrs
    # 断言 p 的 func 方法返回与 p 相同的对象
    assert p.func(*p.args) == p

    # 创建一个符号为 u，real=True 的符号
    u = symbols('u', real=True)
    # 创建一个指针 pu，其符号为 u，类型为 Type.from_expr(u) 的结果，属性包含 value_const 和 pointer_const
    pu = Pointer(u, type=Type.from_expr(u), attrs={value_const, pointer_const})
    # 断言 pu 的符号为 u
    assert pu.symbol is u
    # 断言 pu 的类型为 real
    assert pu.type == real
    # 断言 value_const 在 pu 的属性中
    assert value_const in pu.attrs
    # 断言 pointer_const 在 pu 的属性中
    assert pointer_const in pu.attrs
    # 断言 pu 的 func 方法返回与 pu 相同的对象
    assert pu.func(*pu.args) == pu

    # 创建一个整数符号 i
    i = symbols('i', integer=True)
    # 创建一个 deref 变量，通过 pu[i] 获取指针 pu 的索引为 i 的结果
    deref = pu[i]
    # 断言 deref 的索引为 (i,)
    assert deref.indices == (i,)


# 定义一个名为 test_Declaration 的测试函数
def test_Declaration():
    # 创建一个符号为 u，real=True 的变量 vu
    u = symbols('u', real=True)
    vu = Variable(u, type=Type.from_expr(u))
    # 断言 Declaration(vu).variable 的类型为 real
    assert Declaration(vu).variable.type == real

    # 创建一个符号为 n 的变量 vn
    vn = Variable(n, type=Type.from_expr(n))
    # 断言 Declaration(vn).variable 的类型为 integer
    assert Declaration(vn).variable.type == integer

    # PR 19107，不允许表达式和 Basic 类型之间的比较
    # lt = StrictLessThan(vu, vn)
    # assert isinstance(lt, StrictLessThan)

    # 创建一个符号为 u，real=True，value=3.0，属性包含 value_const 的变量 vuc
    vuc = Variable(u, Type.from_expr(u), value=3.0, attrs={value_const})
    # 断言 value_const 在 vuc 的属性中
    assert value_const in vuc.attrs
    # 断言 pointer_const 不在 vuc 的属性中
    assert pointer_const not in vuc.attrs
    # 创建一个 Declaration 对象 decl，其变量为 vuc
    decl = Declaration(vuc)
    # 断言 decl 的变量为 vuc
    assert decl.variable == vuc
    # 断言 decl 的变量的值为 Float(3.0) 类型
    assert isinstance(decl.variable.value, Float)
    # 断言 decl 的变量的值为 3.0
    assert decl.variable.value == 3.0
    # 断言 decl 的 func 方法返回与 decl 相同的对象
    assert decl.func(*decl.args) == decl
    # 断言 vuc 的 as_Declaration 方法返回 decl
    assert vuc.as_Declaration() == decl
    # 断言 vuc 的 as_Declaration 方法使用 value=None, attrs=None 参数时返回 Declaration(vu)
    assert vuc.as_Declaration(value=None, attrs=None) == Declaration(vu)

    # 创建一个符号为 y，类型为
    # 断言声明3的变量类型为整数（integer）
    assert decl3.variable.type == integer
    # 断言声明3的变量值为3.0
    assert decl3.variable.value == 3.0
    
    # 使用 lambda 表达式调用 Declaration(vi, 42)，预期会引发 ValueError 异常
    raises(ValueError, lambda: Declaration(vi, 42))
# 定义测试函数 test_IntBaseType，用于测试整数基本类型的属性和行为
def test_IntBaseType():
    # 断言 intc 的名称为 'intc'
    assert intc.name == String('intc')
    # 断言 intc 的参数为 (intc.name,)
    assert intc.args == (intc.name,)
    # 断言创建 IntBaseType 实例 'a' 的名称转换为字符串后为 'a'
    assert str(IntBaseType('a').name) == 'a'


# 定义测试函数 test_FloatType，用于测试浮点数类型的属性和行为
def test_FloatType():
    # 测试不同浮点数类型的精度
    assert f16.dig == 3
    assert f32.dig == 6
    assert f64.dig == 15
    assert f80.dig == 18
    assert f128.dig == 33

    # 测试不同浮点数类型的十进制精度
    assert f16.decimal_dig == 5
    assert f32.decimal_dig == 9
    assert f64.decimal_dig == 17
    assert f80.decimal_dig == 21
    assert f128.decimal_dig == 36

    # 测试不同浮点数类型的最大指数
    assert f16.max_exponent == 16
    assert f32.max_exponent == 128
    assert f64.max_exponent == 1024
    assert f80.max_exponent == 16384
    assert f128.max_exponent == 16384

    # 测试不同浮点数类型的最小指数
    assert f16.min_exponent == -13
    assert f32.min_exponent == -125
    assert f64.min_exponent == -1021
    assert f80.min_exponent == -16381
    assert f128.min_exponent == -16381

    # 测试不同浮点数类型的机器精度（epsilon）
    assert abs(f16.eps / Float('0.00097656', precision=16) - 1) < 0.1*10**-f16.dig
    assert abs(f32.eps / Float('1.1920929e-07', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.eps / Float('2.2204460492503131e-16', precision=64) - 1) < 0.1*10**-f64.dig
    assert abs(f80.eps / Float('1.08420217248550443401e-19', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.eps / Float(' 1.92592994438723585305597794258492732e-34', precision=128) - 1) < 0.1*10**-f128.dig

    # 测试不同浮点数类型的最大值
    assert abs(f16.max / Float('65504', precision=16) - 1) < .1*10**-f16.dig
    assert abs(f32.max / Float('3.40282347e+38', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.max / Float('1.79769313486231571e+308', precision=64) - 1) < 0.1*10**-f64.dig  # cf. np.finfo(np.float64).max
    assert abs(f80.max / Float('1.18973149535723176502e+4932', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.max / Float('1.18973149535723176508575932662800702e+4932', precision=128) - 1) < 0.1*10**-f128.dig

    # 测试不同浮点数类型的最小正子规范数（tiny）
    # cf. np.finfo(np.float32).tiny
    assert abs(f16.tiny / Float('6.1035e-05', precision=16) - 1) < 0.1*10**-f16.dig
    assert abs(f32.tiny / Float('1.17549435e-38', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.tiny / Float('2.22507385850720138e-308', precision=64) - 1) < 0.1*10**-f64.dig
    assert abs(f80.tiny / Float('3.36210314311209350626e-4932', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.tiny / Float('3.3621031431120935062626778173217526e-4932', precision=128) - 1) < 0.1*10**-f128.dig

    # 测试浮点数类型的 cast_check 方法
    assert f64.cast_check(0.5) == Float(0.5, 17)
    assert abs(f64.cast_check(3.7) - 3.7) < 3e-17
    assert isinstance(f64.cast_check(3), (Float, float))

    # 测试浮点数类型的 cast_nocheck 方法
    assert f64.cast_nocheck(oo) == float('inf')
    assert f64.cast_nocheck(-oo) == float('-inf')
    assert f64.cast_nocheck(float(oo)) == float('inf')
    assert f64.cast_nocheck(float(-oo)) == float('-inf')
    assert math.isnan(f64.cast_nocheck(nan))

    # 测试浮点数类型的相等性比较
    assert f32 != f64
    assert f64 == f64.func(*f64.args)


# 定义测试函数 test_Type__cast_check__floating_point，用于测试浮点数类型转换的异常情况
def test_Type__cast_check__floating_point():
    # 断言尝试将非法浮点数值 123.45678949 转换为 f32 时引发 ValueError
    raises(ValueError, lambda: f32.cast_check(123.45678949))
    # 断言尝试将非法浮点数值 12.345678949 转换为 f32 时引发 ValueError
    raises(ValueError, lambda: f32.cast_check(12.345678949))
    # 调用 f32.cast_check 函数，验证是否会引发 ValueError 异常，检查小数位数是否超出范围
    raises(ValueError, lambda: f32.cast_check(1.2345678949))
    # 同上，验证是否会引发 ValueError 异常，检查小数位数是否超出范围
    raises(ValueError, lambda: f32.cast_check(.12345678949))
    # 断言验证 f32.cast_check 函数返回值的精度，与预期值的绝对误差小于给定阈值
    assert abs(123.456789049 - f32.cast_check(123.456789049) - 4.9e-8) < 1e-8
    # 断言验证 f32.cast_check 函数返回值的精度，与预期值的绝对误差小于给定阈值
    assert abs(0.12345678904 - f32.cast_check(0.12345678904) - 4e-11) < 1e-11

    # 创建一个 Float 对象 dcm21，含有 21 位小数
    dcm21 = Float('0.123456789012345670499')  # 21 decimals
    # 断言验证 f64.cast_check 函数返回值的精度，与预期值的绝对误差小于给定阈值
    assert abs(dcm21 - f64.cast_check(dcm21) - 4.99e-19) < 1e-19

    # 调用 f80.cast_check 函数，验证是否会引发 ValueError 异常，检查小数位数是否超出范围
    f80.cast_check(Float('0.12345678901234567890103', precision=88))
    # 同上，验证是否会引发 ValueError 异常，检查小数位数是否超出范围
    raises(ValueError, lambda: f80.cast_check(Float('0.12345678901234567890149', precision=88)))

    # 创建变量 v10，赋值为 12345.67894
    v10 = 12345.67894
    # 调用 f32.cast_check 函数，验证是否会引发 ValueError 异常，检查小数位数是否超出范围
    raises(ValueError, lambda: f32.cast_check(v10))
    # 断言验证 Float 对象的精度，与 f64.cast_check 函数返回值的绝对误差小于给定阈值
    assert abs(Float(str(v10), precision=64+8) - f64.cast_check(v10)) < v10*1e-16

    # 断言验证 f32.cast_check 函数返回值的精度，与预期值的绝对误差小于给定阈值
    assert abs(f32.cast_check(2147483647) - 2147483650) < 1
# 定义测试函数，用于检查复杂浮点数类型转换的功能
def test_Type__cast_check__complex_floating_point():
    # 创建一个复数值 val9_11
    val9_11 = 123.456789049 + 0.123456789049j
    # 断言对 .12345678949 + .12345678949j 进行类型转换会抛出 ValueError 异常
    raises(ValueError, lambda: c64.cast_check(.12345678949 + .12345678949j))
    # 断言 val9_11 经过类型转换后与期望值的差的绝对值小于 1e-8
    assert abs(val9_11 - c64.cast_check(val9_11) - 4.9e-8) < 1e-8

    # 创建一个复数值 dcm21，21 位小数
    dcm21 = Float('0.123456789012345670499') + 1e-20j
    # 断言 dcm21 经过类型转换后与期望值的差的绝对值小于 1e-19
    assert abs(dcm21 - c128.cast_check(dcm21) - 4.99e-19) < 1e-19
    # 创建一个复数值 v19
    v19 = Float('0.1234567890123456749') + 1j * Float('0.1234567890123456749')
    # 断言对 v19 进行类型转换会抛出 ValueError 异常
    raises(ValueError, lambda: c128.cast_check(v19))


# 定义测试函数，用于测试 While 循环对象
def test_While():
    # 创建增加赋值操作对象 xpp
    xpp = AddAugmentedAssignment(x, 1)
    # 创建 While 循环对象 whl1
    whl1 = While(x < 2, [xpp])
    # 断言 whl1 的条件参数第一个元素是 x
    assert whl1.condition.args[0] == x
    # 断言 whl1 的条件参数第二个元素是 2
    assert whl1.condition.args[1] == 2
    # 断言 whl1 的条件是 Lt(x, 2, evaluate=False)
    assert whl1.condition == Lt(x, 2, evaluate=False)
    # 断言 whl1 的主体参数是包含 xpp 的元组
    assert whl1.body.args == (xpp,)
    # 断言 whl1 的调用结果等于 whl1 自身
    assert whl1.func(*whl1.args) == whl1

    # 创建代码块对象 cblk，包含增加赋值操作对象 xpp
    cblk = CodeBlock(AddAugmentedAssignment(x, 1))
    # 创建 While 循环对象 whl2
    whl2 = While(x < 2, cblk)
    # 断言 whl1 等于 whl2
    assert whl1 == whl2
    # 断言 whl1 不等于条件为 x < 3 的 While 循环对象
    assert whl1 != While(x < 3, [xpp])


# 定义测试函数，用于测试 Scope 作用域对象
def test_Scope():
    # 创建赋值操作对象 assign
    assign = Assignment(x, y)
    # 创建增加赋值操作对象 incr
    incr = AddAugmentedAssignment(x, 1)
    # 创建 Scope 作用域对象 scp，包含 assign 和 incr
    scp = Scope([assign, incr])
    # 创建代码块对象 cblk，包含 assign 和 incr
    cblk = CodeBlock(assign, incr)
    # 断言 scp 的主体部分等于 cblk
    assert scp.body == cblk
    # 断言 scp 等于具有相同代码块 cblk 的 Scope 对象
    assert scp == Scope(cblk)
    # 断言 scp 不等于包含 incr 和 assign 的 Scope 对象
    assert scp != Scope([incr, assign])
    # 断言 scp 的调用结果等于 scp 自身
    assert scp.func(*scp.args) == scp


# 定义测试函数，用于测试 Print 打印对象
def test_Print():
    # 创建格式化字符串 fmt
    fmt = "%d %.3f"
    # 创建 Print 打印对象 ps，打印 n 和 x，使用格式化字符串 fmt
    ps = Print([n, x], fmt)
    # 断言 ps 的格式化字符串与 fmt 相等
    assert str(ps.format_string) == fmt
    # 断言 ps 的打印参数是元组 (n, x)
    assert ps.print_args == Tuple(n, x)
    # 断言 ps 的参数是 (Tuple(n, x), QuotedString(fmt), none)
    assert ps.args == (Tuple(n, x), QuotedString(fmt), none)
    # 断言 ps 等于具有相同参数的 Print 对象
    assert ps == Print((n, x), fmt)
    # 断言 ps 不等于交换顺序后的 Print 对象
    assert ps != Print([x, n], fmt)
    # 断言 ps 的调用结果等于 ps 自身
    assert ps.func(*ps.args) == ps

    # 创建 Print 打印对象 ps2，打印 n 和 x，不使用格式化字符串
    ps2 = Print([n, x])
    # 断言 ps2 等于具有相同参数的 Print 对象
    assert ps2 == Print([n, x])
    # 断言 ps2 不等于 ps
    assert ps2 != ps
    # 断言 ps2 的格式化字符串是 None
    assert ps2.format_string == None


# 定义测试函数，用于测试 FunctionPrototype 和 FunctionDefinition 函数原型与定义对象
def test_FunctionPrototype_and_FunctionDefinition():
    # 创建实数类型的变量 vx
    vx = Variable(x, type=real)
    # 创建整数类型的变量 vn
    vn = Variable(n, type=integer)
    # 创建函数原型对象 fp1，返回类型为实数，函数名为 'power'，参数为 vx 和 vn
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    # 断言 fp1 的返回类型为实数
    assert fp1.return_type == real
    # 断言 fp1 的函数名为 'power'
    assert fp1.name == String('power')
    # 断言 fp1 的参数为元组 (vx, vn)
    assert fp1.parameters == Tuple(vx, vn)
    # 断言 fp1 等于具有相同返回类型、函数名和参数的 FunctionPrototype 对象
    assert fp1 == FunctionPrototype(real, 'power', [vx, vn])
    # 断言 fp1 不等于参数顺序颠倒后的 FunctionPrototype 对象
    assert fp1 != FunctionPrototype(real, 'power', [vn, vx])
    # 断言 fp1 的调用结果等于 fp1 自身
    assert fp1.func(*fp1.args) == fp1

    # 创建函数体对象 body，包含赋值操作和返回操作
    body = [Assignment(x, x ** n), Return(x)]
    # 创建函数定义对象 fd1，返回类型为实数，函数名为 'power'，参数为 vx 和 vn，函数体为 body
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    # 断言 fd1 的返回类型为实数
    assert fd1.return_type == real
    # 断言 fd1 的函数名为 'power'
    assert str(fd1.name) == 'power'
    # 断言 fd1 的参数为元组 (vx, vn)
    assert fd1.parameters == Tuple(vx, vn)
    # 断言 fd1 的函数体为包含 body 所有元素的代码块对象
    assert fd1.body == CodeBlock(*body)
    # 断言 fd1 等于具有相同返回类型、函数名、参数和函数体的 FunctionDefinition 对象
    assert fd1 == FunctionDefinition(real, 'power', [vx, vn], body)
    # 断言 fd1 不等于函数体顺序颠倒后的 FunctionDefinition 对象
    assert fd1 != FunctionDefinition(real, 'power', [vx, vn], body[::-1])
    # 断言 fd1 的调用结果等于 fd1 自身
    assert fd1.func(*fd1.args) == fd1

    # 从函数定义对象 fd1 创建函数原型对象 fp2
    fp2 = FunctionPrototype.from_FunctionDefinition(fd1)
    # 断言 fp2 等于 fp1
    assert fp2 == fp
    # 断言：验证函数调用对象的第二个参数是否为整数3
    assert fc.function_args[1] == 3
    # 断言：验证函数调用对象的参数数量是否为2
    assert len(fc.function_args) == 2
    # 断言：验证函数调用对象的第二个参数是否为整数类型
    assert isinstance(fc.function_args[1], Integer)
    # 断言：验证函数调用对象是否等于指定的函数调用对象
    assert fc == FunctionCall('power', (x, 3))
    # 断言：验证函数调用对象是否不等于指定的函数调用对象
    assert fc != FunctionCall('power', (3, x))
    # 断言：验证函数调用对象是否不等于指定的函数调用对象（大小写敏感）
    assert fc != FunctionCall('Power', (x, 3))
    # 断言：验证函数调用对象调用其自身函数是否返回自身
    assert fc.func(*fc.args) == fc

    # 创建新的函数调用对象fc2，调用函数'fma'，参数为[2, 3, 4]
    fc2 = FunctionCall('fma', [2, 3, 4])
    # 断言：验证函数调用对象fc2的参数数量是否为3
    assert len(fc2.function_args) == 3
    # 断言：验证函数调用对象fc2的第一个参数是否为2
    assert fc2.function_args[0] == 2
    # 断言：验证函数调用对象fc2的第二个参数是否为3
    assert fc2.function_args[1] == 3
    # 断言：验证函数调用对象fc2的第三个参数是否为4
    assert fc2.function_args[2] == 4
    # 断言：验证函数调用对象fc2的字符串表示是否在指定的字符串列表中
    assert str(fc2) in (
        'FunctionCall(fma, function_args=(2, 3, 4))',
        'FunctionCall("fma", function_args=(2, 3, 4))',
    )
# 定义一个测试函数，用于测试 AST（抽象语法树）中的替换操作
def test_ast_replace():
    # 创建三个变量对象，分别表示实数型变量 x、y 和整数型变量 n
    x = Variable('x', real)
    y = Variable('y', real)
    n = Variable('n', integer)

    # 创建一个实数型函数定义对象 'pwer'，接受 x 和 n 两个参数，执行 pow(x, n) 操作
    pwer = FunctionDefinition(real, 'pwer', [x, n], [pow(x.symbol, n.symbol)])
    # 将函数名 'pwer' 存储在 pname 变量中
    pname = pwer.name
    # 创建一个函数调用对象 'pwer'，传入参数 y 和常数 3
    pcall = FunctionCall('pwer', [y, 3])

    # 创建一个代码块对象 tree1，包含 pwer 和 pcall 两个操作
    tree1 = CodeBlock(pwer, pcall)
    # 断言 tree1 的第一个参数的名称为 'pwer'
    assert str(tree1.args[0].name) == 'pwer'
    # 断言 tree1 的第二个参数的名称为 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    # 使用 zip 函数遍历 tree1 和 [pwer, pcall]，并逐一断言它们相等
    for a, b in zip(tree1, [pwer, pcall]):
        assert a == b

    # 在 tree1 中将函数名为 pname 的部分替换为字符串对象 'power'，生成 tree2
    tree2 = tree1.replace(pname, String('power'))
    # 断言 tree1 的第一个参数的名称仍为 'pwer'
    assert str(tree1.args[0].name) == 'pwer'
    # 断言 tree1 的第二个参数的名称仍为 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    # 断言 tree2 的第一个参数的名称为 'power'
    assert str(tree2.args[0].name) == 'power'
    # 断言 tree2 的第二个参数的名称为 'power'
    assert str(tree2.args[1].name) == 'power'
```