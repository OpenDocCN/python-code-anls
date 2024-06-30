# `D:\src\scipysrc\sympy\sympy\physics\tests\test_secondquant.py`

```
# 导入从 sympy.physics.secondquant 模块中所需的所有符号和函数
from sympy.physics.secondquant import (
    Dagger, Bd, VarBosonicBasis, BBra, B, BKet, FixedBosonicBasis,
    matrix_rep, apply_operators, InnerProduct, Commutator, KroneckerDelta,
    AnnihilateBoson, CreateBoson, BosonicOperator,
    F, Fd, FKet, BosonState, CreateFermion, AnnihilateFermion,
    evaluate_deltas, AntiSymmetricTensor, contraction, NO, wicks,
    PermutationOperator, simplify_index_permutations,
    _sort_anticommuting_fermions, _get_ordered_dummies,
    substitute_dummies, FockStateBosonKet,
    ContractionAppliesOnlyToFermions
)

# 导入从 sympy.concrete.summations 模块中的 Sum 符号
from sympy.concrete.summations import Sum
# 导入从 sympy.core.function 模块中的 Function 和 expand 函数
from sympy.core.function import (Function, expand)
# 导入从 sympy.core.numbers 模块中的 I 和 Rational 符号
from sympy.core.numbers import (I, Rational)
# 导入从 sympy.core.singleton 模块中的 S 符号
from sympy.core.singleton import S
# 导入从 sympy.core.symbol 模块中的 Dummy, Symbol 和 symbols 函数
from sympy.core.symbol import (Dummy, Symbol, symbols)
# 导入从 sympy.functions.elementary.miscellaneous 模块中的 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 导入从 sympy.printing.repr 模块中的 srepr 函数
from sympy.printing.repr import srepr
# 导入从 sympy.simplify.simplify 模块中的 simplify 函数
from sympy.simplify.simplify import simplify

# 导入从 sympy.testing.pytest 模块中的 slow 和 raises 装饰器
from sympy.testing.pytest import slow, raises
# 导入从 sympy.printing.latex 模块中的 latex 函数


# 定义测试函数 test_PermutationOperator
def test_PermutationOperator():
    # 定义符号变量 p, q, r, s
    p, q, r, s = symbols('p,q,r,s')
    # 定义函数符号变量 f, g, h, i
    f, g, h, i = map(Function, 'fghi')
    # 定义 PermutationOperator 为 P
    P = PermutationOperator
    # 验证置换算符作用于 f(p)*g(q) 的结果是否正确
    assert P(p, q).get_permuted(f(p)*g(q)) == -f(q)*g(p)
    # 验证置换算符作用于 f(p, q) 的结果是否正确
    assert P(p, q).get_permuted(f(p, q)) == -f(q, p)
    # 验证置换算符作用于 f(p) 的结果是否正确
    assert P(p, q).get_permuted(f(p)) == f(p)
    # 定义表达式 expr
    expr = (f(p)*g(q)*h(r)*i(s)
        - f(q)*g(p)*h(r)*i(s)
        - f(p)*g(q)*h(s)*i(r)
        + f(q)*g(p)*h(s)*i(r))
    # 定义置换算符列表 perms
    perms = [P(p, q), P(r, s)]
    # 验证 simplify_index_permutations 函数的结果是否正确
    assert (simplify_index_permutations(expr, perms) ==
        P(p, q)*P(r, s)*f(p)*g(q)*h(r)*i(s))
    # 验证 latex 函数是否正确生成 P(p, q) 的 LaTeX 表示
    assert latex(P(p, q)) == 'P(pq)'


# 定义测试函数 test_index_permutations_with_dummies
def test_index_permutations_with_dummies():
    # 定义符号变量 a, b, c, d
    a, b, c, d = symbols('a b c d')
    # 定义虚拟符号变量 p, q, r, s
    p, q, r, s = symbols('p q r s', cls=Dummy)
    # 定义函数符号变量 f, g
    f, g = map(Function, 'fg')
    # 定义 PermutationOperator 为 P
    P = PermutationOperator

    # Case 1: 不需要虚拟符号替换的情况
    # 定义表达式 expr
    expr = f(a, b, p, q) - f(b, a, p, q)
    # 验证 simplify_index_permutations 函数的结果是否正确
    assert simplify_index_permutations(
        expr, [P(a, b)]) == P(a, b)*f(a, b, p, q)

    # Case 2: 需要虚拟符号替换的情况
    # 定义预期结果 expected
    expected = P(a, b)*substitute_dummies(f(a, b, p, q))

    # 验证 simplify_index_permutations 函数的结果是否正确
    expr = f(a, b, p, q) - f(b, a, q, p)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)

    # 验证 simplify_index_permutations 函数的结果是否正确
    expr = f(a, b, q, p) - f(b, a, p, q)
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expected == substitute_dummies(result)

    # Case 3: 无法进行替换的情况
    # 定义表达式 expr
    expr = f(a, b, q, p) - g(b, a, p, q)
    # 验证 simplify_index_permutations 函数的结果是否正确
    result = simplify_index_permutations(expr, [P(a, b)])
    assert expr == result


# 定义测试函数 test_dagger
def test_dagger():
    # 定义符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 验证 Dagger 函数的基本性质
    assert Dagger(1) == 1
    assert Dagger(1.0) == 1.0
    assert Dagger(2*I) == -2*I
    assert Dagger(S.Half*I/3.0) == I*Rational(-1, 2)/3.0
    # 验证 Dagger 函数作用于不同类型对象的结果
    assert Dagger(BKet([n])) == BBra([n])
    assert Dagger(B(0)) == Bd(0)
    assert Dagger(Bd(0)) == B(0)
    assert Dagger(B(n)) == Bd(n)
    assert Dagger(Bd(n)) == B(n)
    assert Dagger(B(0) + B(1)) == Bd(0) + Bd(1)


这些注释将每个函数和语句的作用清晰地解释出来，保持了原始代码的结构和缩进。
    # 断言：对于给定的 n 和 m，验证 Dagger(n*m) 等于 Dagger(n)*Dagger(m)，即 n 和 m 是可交换的
    assert Dagger(n*m) == Dagger(n)*Dagger(m)  # n, m commute
    
    # 断言：验证 Dagger(B(n)*B(m)) 等于 Bd(m)*Bd(n)，即对 B(n)*B(m) 进行转置操作应等效于分别对 B(m) 和 B(n) 进行转置后的乘积
    assert Dagger(B(n)*B(m)) == Bd(m)*Bd(n)
    
    # 断言：验证 Dagger(B(n)**10) 等于 Dagger(B(n))**10，即对 B(n) 的 10 次幂进行转置操作应等效于对 B(n) 进行转置后的 10 次幂
    assert Dagger(B(n)**10) == Dagger(B(n))**10
    
    # 断言：验证 Dagger('a') 等于 Dagger(Symbol('a'))，即对字符串 'a' 进行转置操作应等效于对其符号表示进行转置
    assert Dagger('a') == Dagger(Symbol('a'))
    
    # 断言：验证 Dagger(Dagger('a')) 等于 Symbol('a')，即两次对 'a' 进行转置操作应恢复原始符号 'a'
    assert Dagger(Dagger('a')) == Symbol('a')
# 定义测试函数 test_operator，用于测试 BosonicOperator 类的基本操作
def test_operator():
    # 创建符号变量 i, j
    i, j = symbols('i,j')
    # 创建一个 BosonicOperator 对象 o，其状态为 i
    o = BosonicOperator(i)
    # 断言 o 的状态应为 i
    assert o.state == i
    # 断言 o 是一个符号变量状态
    assert o.is_symbolic
    # 创建一个 BosonicOperator 对象 o，其状态为 1
    o = BosonicOperator(1)
    # 断言 o 的状态应为 1
    assert o.state == 1
    # 断言 o 不是符号变量状态
    assert not o.is_symbolic


# 定义测试函数 test_create，用于测试 Bd 类的创建操作
def test_create():
    # 创建符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 创建一个 Bd 对象 o，其状态为 i
    o = Bd(i)
    # 断言 o 的 LaTeX 表示应为 "{b^\\dagger_{i}}"
    assert latex(o) == "{b^\\dagger_{i}}"
    # 断言 o 是 CreateBoson 的实例
    assert isinstance(o, CreateBoson)
    # 将 o 中的符号 i 替换为 j
    o = o.subs(i, j)
    # 断言 o 中只包含符号 j
    assert o.atoms(Symbol) == {j}
    # 创建一个 Bd 对象 o，其状态为 0
    o = Bd(0)
    # 断言对 o 应用算符 B(0) 在 BKet([n]) 上的作用结果应为 sqrt(n + 1)*BKet([n + 1])
    assert o.apply_operator(BKet([n])) == sqrt(n + 1)*BKet([n + 1])
    # 创建一个 Bd 对象 o，其状态为 n
    o = Bd(n)
    # 断言对 o 应用算符 B(n) 在 BKet([n]) 上的作用结果应为 o*BKet([n])
    assert o.apply_operator(BKet([n])) == o*BKet([n])


# 定义测试函数 test_annihilate，用于测试 B 类的湮灭操作
def test_annihilate():
    # 创建符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 创建一个 B 对象 o，其状态为 i
    o = B(i)
    # 断言 o 的 LaTeX 表示应为 "b_{i}"
    assert latex(o) == "b_{i}"
    # 断言 o 是 AnnihilateBoson 的实例
    assert isinstance(o, AnnihilateBoson)
    # 将 o 中的符号 i 替换为 j
    o = o.subs(i, j)
    # 断言 o 中只包含符号 j
    assert o.atoms(Symbol) == {j}
    # 创建一个 B 对象 o，其状态为 0
    o = B(0)
    # 断言对 o 应用算符 B(0) 在 BKet([n]) 上的作用结果应为 sqrt(n)*BKet([n - 1])
    assert o.apply_operator(BKet([n])) == sqrt(n)*BKet([n - 1])
    # 创建一个 B 对象 o，其状态为 n
    o = B(n)
    # 断言对 o 应用算符 B(n) 在 BKet([n]) 上的作用结果应为 o*BKet([n])
    assert o.apply_operator(BKet([n])) == o*BKet([n])


# 定义测试函数 test_basic_state，用于测试 BosonState 类的基本状态操作
def test_basic_state():
    # 创建符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 创建一个 BosonState 对象 s，其状态为 [0, 1, 2, 3, 4]
    s = BosonState([0, 1, 2, 3, 4])
    # 断言 s 的长度应为 5
    assert len(s) == 5
    # 断言 s 的第一个参数应为元组 (0, 1, 2, 3, 4)
    assert s.args[0] == tuple(range(5))
    # 断言对 s 应用 up(0) 操作后的结果应为 BosonState([1, 1, 2, 3, 4])
    assert s.up(0) == BosonState([1, 1, 2, 3, 4])
    # 断言对 s 应用 down(4) 操作后的结果应为 BosonState([0, 1, 2, 3, 3])
    assert s.down(4) == BosonState([0, 1, 2, 3, 3])
    # 断言对 s 中的每个位置 i，应用 up(i) 和 down(i) 操作后结果应为 s 本身
    for i in range(5):
        assert s.up(i).down(i) == s
    # 断言对 s 应用 down(0) 操作后的结果应为 0
    assert s.down(0) == 0
    # 创建一个 BosonState 对象 s，其状态为 [n, m]
    s = BosonState([n, m])
    # 断言对 s 应用 down(0) 操作后的结果应为 BosonState([n - 1, m])
    assert s.down(0) == BosonState([n - 1, m])
    # 断言对 s 应用 up(0) 操作后的结果应为 BosonState([n + 1, m])
    assert s.up(0) == BosonState([n + 1, m])


# 定义测试函数 test_basic_apply，用于测试 BosonicOperator 类的基本应用操作
def test_basic_apply():
    # 创建符号变量 n
    n = symbols("n")
    # 创建一个表达式 e，表示为 B(0)*BKet([n])
    e = B(0)*BKet([n])
    # 断言对表达式 e 应用 apply_operators 后的结果应为 sqrt(n)*BKet([n - 1])
    assert apply_operators(e) == sqrt(n)*BKet([n - 1])
    # 创建一个表达式 e，表示为 Bd(0)*BKet([n])
    e = Bd(0)*BKet([n])
    # 断言对表达式 e 应用 apply_operators 后的结果应为 sqrt(n + 1)*BKet([n + 1])


# 定义测试函数 test_complex_apply，用于测试 BosonicOperator 类的复杂应用操作
def test_complex_apply():
    # 创建符号变量 n, m
    n, m = symbols("n,m")
    # 创建一个算符组合 o，表示为 Bd(0)*B(0)*Bd(1)*B(0)
    o = Bd(0)*B(0)*Bd(1)*B(0)
    # 对 o 作用于 BKet([n, m]) 后的结果 e
    e = apply_operators(o*BKet([n, m]))
    # 预期的结果 answer
    answer = sqrt(n)*sqrt(m + 1)*(-1 + n)*BKet([-1 + n, 1 + m])
    # 断言对 e 和 answer 扩展后应相等
    assert expand(e) == expand(answer)


# 定义测试函数 test_number_operator，用于测试 BosonicOperator 类的数值算符操作
def test_number_operator():
    # 创建符号变量 n
    n = symbols("n")
    # 创建一个算符组合 o，表示为 Bd(0)*B(0)
    o = Bd(0)*B(0)
    # 对 o 作用于 BKet([n]) 后的结果 e
    e = apply_operators(o*BKet([n]))
    # 断言 e 应为 n*BKet([n])


# 定义测试函数 test_inner_product，用于测试 BosonBra 和 BosonKet 的内积操作
def test_inner_product():
    # 创建符号变量 i, j, k, l
    i, j, k, l = symbols('i,j,k,l')
    # 创建一个 BosonBra 对象 s1，其状态为 [0]
    s1 = BBra([0])
    # 创建一个 BosonKet 对象 s2，其状态为 [1]
    s2 = BKet([1])
    # 断言：检查在字节对象 b 中状态 state 的索引是否为 1
    assert b.index(state) == 1
    
    # 断言：检查字节对象 b 中索引为 1 的状态是否等于 b[1]
    assert b.state(1) == b[1]
    
    # 断言：检查字节对象 b 的长度是否为 3
    assert len(b) == 3
    
    # 断言：检查字节对象 b 的字符串表示形式是否为指定的格式
    assert str(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
    
    # 断言：检查字节对象 b 的详细字符串表示形式是否为指定的格式
    assert repr(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
    
    # 断言：检查字节对象 b 的详细字符串表示形式是否为指定的格式（通常用于调试）
    assert srepr(b) == '[FockState((2, 0)), FockState((1, 1)), FockState((0, 2))]'
@slow
def test_sho():
    # 定义符号变量 n, m
    n, m = symbols('n,m')
    # 定义算符表达式 h_n
    h_n = Bd(n)*B(n)*(n + S.Half)
    # 对 h_n 进行求和，得到表达式 H
    H = Sum(h_n, (n, 0, 5))
    # 计算 H 的值，不进行深度计算
    o = H.doit(deep=False)
    # 创建一个固定基的玻色子基础
    b = FixedBosonicBasis(2, 6)
    # 将 o 用 b 的矩阵表示
    m = matrix_rep(o, b)
    # 我们需要双重检查这些能量值，确保它们正确并具有适当的简并度！
    # 检查对角线元素是否与 diag 列表中的值相等
    diag = [1, 2, 3, 3, 4, 5, 4, 5, 6, 7, 5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 11]
    for i in range(len(diag)):
        assert diag[i] == m[i, i]


def test_commutation():
    # 定义符号变量 n, m，并指定 n 为费米面以上的符号
    n, m = symbols("n,m", above_fermi=True)
    # 计算 B(0) 和 Bd(0) 的对易子
    c = Commutator(B(0), Bd(0))
    assert c == 1
    # 计算 Bd(0) 和 B(0) 的对易子
    c = Commutator(Bd(0), B(0))
    assert c == -1
    # 计算 B(n) 和 Bd(0) 的对易子
    c = Commutator(B(n), Bd(0))
    assert c == KroneckerDelta(n, 0)
    # 计算 B(0) 和 B(0) 的对易子
    c = Commutator(B(0), B(0))
    assert c == 0
    # 计算 B(0) 和 Bd(0) 的对易子，并简化其作用在 BKet([n]) 上的结果
    c = Commutator(B(0), Bd(0))
    e = simplify(apply_operators(c*BKet([n])))
    assert e == BKet([n])
    # 计算 B(0) 和 B(1) 的对易子，并简化其作用在 BKet([n, m]) 上的结果
    c = Commutator(B(0), B(1))
    e = simplify(apply_operators(c*BKet([n, m])))
    assert e == 0

    # 计算 F(m) 和 Fd(m) 的对易子
    c = Commutator(F(m), Fd(m))
    assert c == +1 - 2*NO(Fd(m)*F(m))
    # 计算 Fd(m) 和 F(m) 的对易子，并展开结果
    c = Commutator(Fd(m), F(m))
    assert c.expand() == -1 + 2*NO(Fd(m)*F(m))

    # 定义符号变量 X, Y, Z，并指定它们为非交换的符号
    C = Commutator
    X, Y, Z = symbols('X,Y,Z', commutative=False)
    # 检查嵌套的对易子 C(C(X, Y), Z) 是否不等于 0
    assert C(C(X, Y), Z) != 0
    # 检查嵌套的对易子 C(C(X, Z), Y) 是否不等于 0
    assert C(C(X, Z), Y) != 0
    # 检查对易子 C(Y, C(X, Z)) 是否不等于 0

    # 定义符号变量 i, j, k, l，且指定 i, j, k, l 均为费米面以下的符号
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    # 定义符号变量 a, b, c, d，且指定 a, b, c, d 均为费米面以上的符号
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    # 定义符号变量 p, q, r, s
    p, q, r, s = symbols('p,q,r,s')
    # 定义 KroneckerDelta 函数
    D = KroneckerDelta

    # 检查 Fd(a) 和 F(i) 的对易子
    assert C(Fd(a), F(i)) == -2*NO(F(i)*Fd(a))
    # 检查对 C(Fd(j), NO(Fd(a)*F(i))) 进行 Wick 缩并后的结果是否为 -D(j, i)*Fd(a)
    assert C(Fd(j), NO(Fd(a)*F(i))).doit(wicks=True) == -D(j, i)*Fd(a)
    # 检查对 C(Fd(a)*F(i), Fd(b)*F(j)) 进行 Wick 缩并后的结果是否为 0
    assert C(Fd(a)*F(i), Fd(b)*F(j)).doit(wicks=True) == 0

    # 计算 Fd(a) 和 F(a) 的对易子 c1，并检查它的值是否为 0
    c1 = Commutator(F(a), Fd(a))
    assert Commutator.eval(c1, c1) == 0
    # 检查 Fd(a)*F(i) 和 Fd(b)*F(j) 的对易子 c 的 LaTeX 表示是否正确
    c = Commutator(Fd(a)*F(i), Fd(b)*F(j))
    assert latex(c) == r'\left[{a^\dagger_{a}} a_{i},{a^\dagger_{b}} a_{j}\right]'
    # 检查 Fd(a)*F(i) 和 Fd(b)*F(j) 的对易子 c 的 Python 表示是否正确
    assert repr(c) == 'Commutator(CreateFermion(a)*AnnihilateFermion(i),CreateFermion(b)*AnnihilateFermion(j))'
    # 检查 Fd(a)*F(i) 和 Fd(b)*F(j) 的对易子 c 的字符串表示是否正确
    assert str(c) == '[CreateFermion(a)*AnnihilateFermion(i),CreateFermion(b)*AnnihilateFermion(j)]'


def test_create_f():
    # 定义符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 创建 Fd(i) 算符对象 o，并检查它的类型是否为 CreateFermion
    o = Fd(i)
    assert isinstance(o, CreateFermion)
    # 将 o 中的符号 i 替换为 j，并检查替换后的 o 中的符号集合是否只有 j
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    # 创建 Fd(1) 算符对象 o，并检查其应用在 FKet([n]) 上的结果是否为 FKet([1, n])
    o = Fd(1)
    assert o.apply_operator(FKet([n])) == FKet([1, n])
    # 再次检查 o 在 FKet([n]) 上的应用结果是否为 -FKet([n, 1])
    assert o.apply_operator(FKet([n])) == -FKet([n, 1])
    # 创建 Fd(n) 算符对象 o，并检查其应用在 FKet([]) 上的结果是否为 FKet([n])
    o = Fd(n)
    assert o.apply_operator(FKet([])) == FKet([n])

    # 创建具有费米面级别为 4 的真空态 FKet 对象
    vacuum = FKet([], fermi_level=4)
    assert vacuum == FKet([], fermi_level=4)

    # 定义符号变量 i, j, k, l，且指定 i, j, k, l 均为费米面以下的符号
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    # 定义符号变量 a, b, c, d，且指定 a, b, c, d 均为费
    # 定义符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 调用函数 F，返回 AnnihilateFermion 类的实例，并赋值给变量 o
    o = F(i)
    # 断言 o 是 AnnihilateFermion 的实例
    assert isinstance(o, AnnihilateFermion)
    # 使用 subs 方法将 o 中的 i 替换为 j
    o = o.subs(i, j)
    # 断言 o 中的符号元素只包含 j
    assert o.atoms(Symbol) == {j}
    # 调用函数 F，返回 AnnihilateFermion 类的实例，并赋值给变量 o
    o = F(1)
    # 断言对 o 应用 apply_operator 方法后的结果
    assert o.apply_operator(FKet([1, n])) == FKet([n])
    # 断言对 o 应用 apply_operator 方法后的结果
    assert o.apply_operator(FKet([n, 1])) == -FKet([n])
    # 调用函数 F，返回 AnnihilateFermion 类的实例，并赋值给变量 o
    o = F(n)
    # 断言对 o 应用 apply_operator 方法后的结果
    assert o.apply_operator(FKet([n])) == FKet([])
    
    # 定义符号变量 i, j, k, l，其中 l 在费米面以下
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    # 定义符号变量 a, b, c, d，其中 a 在费米面以上
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    # 定义符号变量 p, q, r, s
    p, q, r, s = symbols('p,q,r,s')
    # 断言对 F(i) 应用 apply_operator 方法后的结果为零
    assert F(i).apply_operator(FKet([i, j, k], 4)) == 0
    # 断言对 F(a) 应用 apply_operator 方法后的结果为零
    assert F(a).apply_operator(FKet([i, b, k], 4)) == 0
    # 断言对 F(l) 应用 apply_operator 方法后的结果为零
    assert F(l).apply_operator(FKet([i, j, k], 3)) == 0
    # 断言对 F(l) 应用 apply_operator 方法后的结果
    assert F(l).apply_operator(FKet([i, j, k], 4)) == FKet([l, i, j, k], 4)
    # 断言将 F(p) 转换为字符串的结果
    assert str(F(p)) == 'f(p)'
    # 断言将 F(p) 转换为 repr 字符串的结果
    assert repr(F(p)) == 'AnnihilateFermion(p)'
    # 断言将 F(p) 应用 srepr 函数后的结果
    assert srepr(F(p)) == "AnnihilateFermion(Symbol('p'))"
    # 断言将 F(p) 转换为 LaTeX 格式的字符串的结果
    assert latex(F(p)) == 'a_{p}'
def test_create_b():
    # 定义符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 创建 Bd(i) 对象，并验证其类型为 CreateBoson 的实例
    o = Bd(i)
    assert isinstance(o, CreateBoson)
    # 将 Bd(i) 中的 i 替换为 j
    o = o.subs(i, j)
    # 验证 o 中的符号元素为 {j}
    assert o.atoms(Symbol) == {j}
    # 创建 Bd(0) 对象，并验证其作用于 BKet([n]) 后的结果
    o = Bd(0)
    assert o.apply_operator(BKet([n])) == sqrt(n + 1)*BKet([n + 1])
    # 创建 Bd(n) 对象，并验证其作用于 BKet([n]) 后的结果
    o = Bd(n)
    assert o.apply_operator(BKet([n])) == o*BKet([n])


def test_annihilate_b():
    # 定义符号变量 i, j, n, m
    i, j, n, m = symbols('i,j,n,m')
    # 创建 B(i) 对象，并验证其类型为 AnnihilateBoson 的实例
    o = B(i)
    assert isinstance(o, AnnihilateBoson)
    # 将 B(i) 中的 i 替换为 j
    o = o.subs(i, j)
    # 不执行任何断言的后续测试


def test_wicks():
    # 定义符号变量 p, q, r, s，且 p, q, r, s 都定义为费米子以上的符号
    p, q, r, s = symbols('p,q,r,s', above_fermi=True)

    # Testing for particles only

    # 创建表达式 str = F(p)*Fd(q)，并验证 Wick 对称化结果
    str = F(p)*Fd(q)
    assert wicks(str) == NO(F(p)*Fd(q)) + KroneckerDelta(p, q)
    
    # 创建表达式 str = Fd(p)*F(q)，并验证 Wick 对称化结果
    str = Fd(p)*F(q)
    assert wicks(str) == NO(Fd(p)*F(q))
    
    # 创建表达式 str = F(p)*Fd(q)*F(r)*Fd(s)，并验证 Wick 对称化结果
    str = F(p)*Fd(q)*F(r)*Fd(s)
    nstr = wicks(str)
    fasit = NO(
        KroneckerDelta(p, q)*KroneckerDelta(r, s)
        + KroneckerDelta(p, q)*AnnihilateFermion(r)*CreateFermion(s)
        + KroneckerDelta(r, s)*AnnihilateFermion(p)*CreateFermion(q)
        - KroneckerDelta(p, s)*AnnihilateFermion(r)*CreateFermion(q)
        - AnnihilateFermion(p)*AnnihilateFermion(r)*CreateFermion(q)*CreateFermion(s))
    assert nstr == fasit
    
    # 验证表达式 (p*q*nstr).expand() 的展开结果等于 wicks(p*q*str)
    assert (p*q*nstr).expand() == wicks(p*q*str)
    # 验证表达式 (nstr*p*q*2).expand() 的展开结果等于 wicks(str*p*q*2)

    # Testing CC equations particles and holes
    # 定义符号变量 i, j, k, l，且 i, j, k, l 都定义为费米子以下的虚拟符号
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    # 定义符号变量 a, b, c, d，且 a, b, c, d 都定义为费米子以上的虚拟符号
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    # 定义符号变量 p, q, r, s
    p, q, r, s = symbols('p q r s', cls=Dummy)

    # 验证 Wick 对称化结果
    assert (wicks(F(a)*NO(F(i)*F(j))*Fd(b)) ==
            NO(F(a)*F(i)*F(j)*Fd(b)) +
            KroneckerDelta(a, b)*NO(F(i)*F(j)))
    
    # 验证 Wick 对称化结果
    assert (wicks(F(a)*NO(F(i)*F(j)*F(k))*Fd(b)) ==
            NO(F(a)*F(i)*F(j)*F(k)*Fd(b)) -
            KroneckerDelta(a, b)*NO(F(i)*F(j)*F(k)))
    
    # 验证 Wick 对称化结果
    expr = wicks(Fd(i)*NO(Fd(j)*F(k))*F(l))
    assert (expr ==
           -KroneckerDelta(i, k)*NO(Fd(j)*F(l)) -
            KroneckerDelta(j, l)*NO(Fd(i)*F(k)) -
            KroneckerDelta(i, k)*KroneckerDelta(j, l) +
            KroneckerDelta(i, l)*NO(Fd(j)*F(k)) +
            NO(Fd(i)*Fd(j)*F(k)*F(l)))
    
    # 验证 Wick 对称化结果
    expr = wicks(F(a)*NO(F(b)*Fd(c))*Fd(d))
    assert (expr ==
           -KroneckerDelta(a, c)*NO(F(b)*Fd(d)) -
            KroneckerDelta(b, d)*NO(F(a)*Fd(c)) -
            KroneckerDelta(a, c)*KroneckerDelta(b, d) +
            KroneckerDelta(a, d)*NO(F(b)*Fd(c)) +
            NO(F(a)*F(b)*Fd(c)*Fd(d)))


def test_NO():
    # 定义符号变量 i, j, k, l，且 i, j, k, l 都定义为费米子以下的符号
    i, j, k, l = symbols('i j k l', below_fermi=True)
    # 定义符号变量 a, b, c, d，且 a, b, c, d 都定义为费米子以上的符号
    a, b, c, d = symbols('a b c d', above_fermi=True)
    # 定义符号变量 p, q, r, s
    p, q, r, s = symbols('p q r s', cls=Dummy)

    # 验证 NO 运算结果
    assert (NO(Fd(p)*F(q) + Fd(a)*F(b)) ==
       NO(Fd(p)*F(q)) + NO(Fd(a)*F(b)))
    
    # 验证 NO 运算结果
    assert (NO(Fd(i)*NO(F(j)*Fd(a))) ==
       NO(Fd(i)*F(j)*Fd(a)))
    
    # 验证 NO 运算结果
    assert NO(1) == 1
    assert NO(i) == i
    
    # 验证 NO 运算结果
    assert (NO(Fd(a)*Fd(b)*(F(c) + F(d))) ==
            NO(Fd(a)*Fd(b)*F(c)) +
            NO(Fd(a)*Fd(b)*F(d)))
    
    # 验证 NO 运算结果
    assert NO(Fd(a)*F(b))._remove_brackets() == Fd(a)*F(b)
    # 断言确保 F(j)*Fd(i) 没有多余的括号，并且结果等于 F(j)*Fd(i)
    assert NO(F(j)*Fd(i))._remove_brackets() == F(j)*Fd(i)

    # 断言替换 Fd(p) 为 Fd(a) + Fd(i)，并计算结果与预期相符
    assert (NO(Fd(p)*F(q)).subs(Fd(p), Fd(a) + Fd(i)) ==
            NO(Fd(a)*F(q)) + NO(Fd(i)*F(q)))

    # 断言替换 F(q) 为 F(a) + F(i)，并计算结果与预期相符
    assert (NO(Fd(p)*F(q)).subs(F(q), F(a) + F(i)) ==
            NO(Fd(p)*F(a)) + NO(Fd(p)*F(i)))

    # 对表达式 Fd(p)*F(q) 去除多余的括号，并使用 Wick 定理计算结果
    expr = NO(Fd(p)*F(q))._remove_brackets()
    assert wicks(expr) == NO(expr)

    # 断言 Fd(a)*F(b) 的正规排序等于 - NO(F(b)*Fd(a))
    assert NO(Fd(a)*F(b)) == - NO(F(b)*Fd(a))

    # 创建一个 NO 对象，包含 Fd(a)*F(i)*F(b)*Fd(j)，并分别获取其产生算符和湮灭算符列表
    no = NO(Fd(a)*F(i)*F(b)*Fd(j))
    l1 = list(no.iter_q_creators())
    assert l1 == [0, 1]  # 产生算符的索引列表应为 [0, 1]
    l2 = list(no.iter_q_annihilators())
    assert l2 == [3, 2]  # 湮灭算符的索引列表应为 [3, 2]

    # 创建一个 NO 对象，包含 Fd(a)*Fd(i)，并验证其产生算符和湮灭算符数量
    no = NO(Fd(a)*Fd(i))
    assert no.has_q_creators == 1  # 产生算符数应为 1
    assert no.has_q_annihilators == -1  # 湮灭算符数应为 -1
    assert str(no) == ':CreateFermion(a)*CreateFermion(i):'  # 字符串表示应为指定格式
    assert repr(no) == 'NO(CreateFermion(a)*CreateFermion(i))'  # 表示形式应为指定格式
    assert latex(no) == r'\left\{{a^\dagger_{a}} {a^\dagger_{i}}\right\}'  # LaTeX 表示应为指定格式

    # 断言 NO(Bd(p)*F(q)) 会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda:  NO(Bd(p)*F(q)))
def test_sorting():
    # 定义一些符号，i 和 j 是 Fermi 表面以下的，a 和 b 是 Fermi 表面以上的，p 和 q 是一般符号
    i, j = symbols('i,j', below_fermi=True)
    a, b = symbols('a,b', above_fermi=True)
    p, q = symbols('p,q')

    # 测试反对易费米子排序函数 _sort_anticommuting_fermions
    assert _sort_anticommuting_fermions([Fd(p), F(q)]) == ([Fd(p), F(q)], 0)
    assert _sort_anticommuting_fermions([F(p), Fd(q)]) == ([Fd(q), F(p)], 1)

    # 测试不同组合下的排序效果
    assert _sort_anticommuting_fermions([F(p), Fd(i)]) == ([F(p), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), F(p)]) == ([F(p), Fd(i)], 1)
    assert _sort_anticommuting_fermions([Fd(p), Fd(i)]) == ([Fd(p), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), Fd(p)]) == ([Fd(p), Fd(i)], 1)
    assert _sort_anticommuting_fermions([F(p), F(i)]) == ([F(i), F(p)], 1)
    assert _sort_anticommuting_fermions([F(i), F(p)]) == ([F(i), F(p)], 0)
    assert _sort_anticommuting_fermions([Fd(p), F(i)]) == ([F(i), Fd(p)], 1)
    assert _sort_anticommuting_fermions([F(i), Fd(p)]) == ([F(i), Fd(p)], 0)

    # 继续测试其他符号组合
    assert _sort_anticommuting_fermions([F(p), Fd(a)]) == ([Fd(a), F(p)], 1)
    assert _sort_anticommuting_fermions([Fd(a), F(p)]) == ([Fd(a), F(p)], 0)
    assert _sort_anticommuting_fermions([Fd(p), Fd(a)]) == ([Fd(a), Fd(p)], 1)
    assert _sort_anticommuting_fermions([Fd(a), Fd(p)]) == ([Fd(a), Fd(p)], 0)
    assert _sort_anticommuting_fermions([F(p), F(a)]) == ([F(p), F(a)], 0)
    assert _sort_anticommuting_fermions([F(a), F(p)]) == ([F(p), F(a)], 1)
    assert _sort_anticommuting_fermions([Fd(p), F(a)]) == ([Fd(p), F(a)], 0)
    assert _sort_anticommuting_fermions([F(a), Fd(p)]) == ([Fd(p), F(a)], 1)

    # 最后的测试组合
    assert _sort_anticommuting_fermions([F(i), Fd(j)]) == ([F(i), Fd(j)], 0)
    assert _sort_anticommuting_fermions([Fd(j), F(i)]) == ([F(i), Fd(j)], 1)
    assert _sort_anticommuting_fermions([Fd(a), Fd(i)]) == ([Fd(a), Fd(i)], 0)
    assert _sort_anticommuting_fermions([Fd(i), Fd(a)]) == ([Fd(a), Fd(i)], 1)
    assert _sort_anticommuting_fermions([F(a), F(i)]) == ([F(i), F(a)], 1)
    assert _sort_anticommuting_fermions([F(i), F(a)]) == ([F(i), F(a)], 0)


def test_contraction():
    # 定义一些符号，i, j, k, l 是 Fermi 表面以下的，a, b, c, d 是 Fermi 表面以上的，p, q, r, s 是一般符号
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    p, q, r, s = symbols('p,q,r,s')

    # 测试缩并函数 contraction
    assert contraction(Fd(i), F(j)) == KroneckerDelta(i, j)
    assert contraction(F(a), Fd(b)) == KroneckerDelta(a, b)
    assert contraction(F(a), Fd(i)) == 0
    assert contraction(Fd(a), F(i)) == 0
    assert contraction(F(i), Fd(a)) == 0
    assert contraction(Fd(i), F(a)) == 0
    assert contraction(Fd(i), F(p)) == KroneckerDelta(i, p)

    # 测试 evaluate_deltas 函数的结果
    restr = evaluate_deltas(contraction(Fd(p), F(q)))
    assert restr.is_only_below_fermi
    restr = evaluate_deltas(contraction(F(p), Fd(q)))
    assert restr.is_only_above_fermi

    # 测试异常处理
    raises(ContractionAppliesOnlyToFermions, lambda: contraction(B(a), Fd(b)))


def test_evaluate_deltas():
    # 定义一些符号，i, j, k 是一般符号
    i, j, k = symbols('i,j,k')

    # 测试 evaluate_deltas 函数
    r = KroneckerDelta(i, j) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(i, k)
    # 创建 KroneckerDelta 对象 r，其值为 KroneckerDelta(i, 0) * KroneckerDelta(j, k)
    r = KroneckerDelta(i, 0) * KroneckerDelta(j, k)
    # 使用 evaluate_deltas 函数验证 r 的计算结果是否等于 KroneckerDelta(i, 0) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(i, 0) * KroneckerDelta(j, k)

    # 创建 KroneckerDelta 对象 r，其值为 KroneckerDelta(1, j) * KroneckerDelta(j, k)
    r = KroneckerDelta(1, j) * KroneckerDelta(j, k)
    # 使用 evaluate_deltas 函数验证 r 的计算结果是否等于 KroneckerDelta(1, k)
    assert evaluate_deltas(r) == KroneckerDelta(1, k)

    # 创建 KroneckerDelta 对象 r，其值为 KroneckerDelta(j, 2) * KroneckerDelta(k, j)
    r = KroneckerDelta(j, 2) * KroneckerDelta(k, j)
    # 使用 evaluate_deltas 函数验证 r 的计算结果是否等于 KroneckerDelta(2, k)
    assert evaluate_deltas(r) == KroneckerDelta(2, k)

    # 创建 KroneckerDelta 对象 r，其值为 KroneckerDelta(i, 0) * KroneckerDelta(i, j) * KroneckerDelta(j, 1)
    r = KroneckerDelta(i, 0) * KroneckerDelta(i, j) * KroneckerDelta(j, 1)
    # 使用 evaluate_deltas 函数验证 r 的计算结果是否等于 0
    assert evaluate_deltas(r) == 0

    # 创建 KroneckerDelta 对象 r，其值为 KroneckerDelta(0, i) * KroneckerDelta(0, j)
    #                             * KroneckerDelta(1, j) * KroneckerDelta(1, j)
    r = (KroneckerDelta(0, i) * KroneckerDelta(0, j)
         * KroneckerDelta(1, j) * KroneckerDelta(1, j))
    # 使用 evaluate_deltas 函数验证 r 的计算结果是否等于 0
    assert evaluate_deltas(r) == 0
# 定义一个测试函数，用于测试 AntiSymmetricTensor 类的功能
def test_Tensors():
    # 定义符号变量 i, j, k, l，并指定 below_fermi=True，表示在费米面以下
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    # 定义符号变量 a, b, c, d，并指定 above_fermi=True，表示在费米面以上
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    # 定义符号变量 p, q, r, s
    p, q, r, s = symbols('p q r s')

    # AT 表示 AntiSymmetricTensor 类的别名
    AT = AntiSymmetricTensor
    # 断言 AntiSymmetricTensor 的对称性质：t_{ab,ij} = -t_{ba,ij}
    assert AT('t', (a, b), (i, j)) == -AT('t', (b, a), (i, j))
    # 断言 AntiSymmetricTensor 的对称性质：t_{ab,ij} = t_{ab,ji}
    assert AT('t', (a, b), (i, j)) == AT('t', (b, a), (j, i))
    # 断言 AntiSymmetricTensor 的反对称性质：t_{ab,ij} = -t_{ab,ji}
    assert AT('t', (a, b), (i, j)) == -AT('t', (a, b), (j, i))
    # 断言对角元素为零：t_{aa,ij} = 0
    assert AT('t', (a, a), (i, j)) == 0
    # 断言对角元素为零：t_{ab,ii} = 0
    assert AT('t', (a, b), (i, i)) == 0
    # 断言 AntiSymmetricTensor 的对称性质：t_{abc,ij} = -t_{bac,ij}
    assert AT('t', (a, b, c), (i, j)) == -AT('t', (b, a, c), (i, j))
    # 断言 AntiSymmetricTensor 的对称性质：t_{abc,ijk} = t_{bac,kji}
    assert AT('t', (a, b, c), (i, j, k)) == AT('t', (b, a, c), (i, k, j))

    # 生成一个 AT 对象用于后续断言操作
    tabij = AT('t', (a, b), (i, j))
    # 断言 AT 对象中包含变量 a
    assert tabij.has(a)
    # 断言 AT 对象中包含变量 b
    assert tabij.has(b)
    # 断言 AT 对象中包含变量 i
    assert tabij.has(i)
    # 断言 AT 对象中包含变量 j
    assert tabij.has(j)
    # 断言替换变量 b 后结果与预期相符：t_{ac,ij} = t_{ac,ij}
    assert tabij.subs(b, c) == AT('t', (a, c), (i, j))
    # 断言替换变量 i 后结果与预期相符：2 * t_{ac,ci} = 2 * t_{ab,ci}
    assert (2*tabij).subs(i, c) == 2*AT('t', (a, b), (c, j))
    # 断言 AT 对象的 symbol 属性为符号 't'
    assert tabij.symbol == Symbol('t')
    # 断言 AT 对象的 LaTeX 表示形式为 '{t^{ab}_{ij}}'
    assert latex(tabij) == '{t^{ab}_{ij}}'
    # 断言 AT 对象的字符串表示形式为 't((_a, _b),(_i, _j))'
    assert str(tabij) == 't((_a, _b),(_i, _j))'

    # 断言替换变量 a 后结果与预期相符：t_{bb,ij} = t_{bb,ij}
    assert AT('t', (a, a), (i, j)).subs(a, b) == AT('t', (b, b), (i, j))
    # 断言替换变量 a 后结果与预期相符：t_{ai,ai} = t_{bi,bi}
    assert AT('t', (a, i), (a, j)).subs(a, b) == AT('t', (b, i), (b, j))


# 定义测试函数，测试 fully contracted 的符号代换
def test_fully_contracted():
    # 定义符号变量 i, j, k, l，并指定 below_fermi=True
    i, j, k, l = symbols('i j k l', below_fermi=True)
    # 定义符号变量 a, b, c, d，并指定 above_fermi=True
    a, b, c, d = symbols('a b c d', above_fermi=True)
    # 定义符号变量 p, q, r, s，并指定为虚拟变量
    p, q, r, s = symbols('p q r s', cls=Dummy)

    # 定义 Fock 和 V 的符号表达式
    Fock = (AntiSymmetricTensor('f', (p,), (q,)) *
            NO(Fd(p) * F(q)))
    V = (AntiSymmetricTensor('v', (p, q), (r, s)) *
         NO(Fd(p) * Fd(q) * F(s) * F(r))) / 4

    # 计算 fully contracted 的 Fock 表达式
    Fai = wicks(NO(Fd(i)*F(a))*Fock,
                keep_only_fully_contracted=True,
                simplify_kronecker_deltas=True)
    # 断言结果与预期相符：f_{a,i} = f_{a,i}
    assert Fai == AntiSymmetricTensor('f', (a,), (i,))

    # 计算 fully contracted 的 V 表达式
    Vabij = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*V,
                  keep_only_fully_contracted=True,
                  simplify_kronecker_deltas=True)
    # 断言结果与预期相符：v_{ab,ij} = v_{ab,ij}
    assert Vabij == AntiSymmetricTensor('v', (a, b), (i, j))


# 定义测试函数，测试 substitute_dummies 函数在没有虚拟变量的情况下的表现
def test_substitute_dummies_without_dummies():
    # 定义符号变量 i, j
    i, j = symbols('i,j')
    # 断言 substitute_dummies 函数应不改变表达式：att(i, j) + 2 = att(i, j) + 2
    assert substitute_dummies(att(i, j) + 2) == att(i, j) + 2
    # 断言 substitute_dummies 函数应不改变表达式：att(i, j) + 1 = att(i, j) + 1
    assert substitute_dummies(att(i, j) + 1) == att(i, j) + 1


# 定义测试函数，测试 substitute_dummies 函数在包含 NO 操作符的表达式中的表现
def test_substitute_dummies_NO_operator():
    # 定义虚拟变量 i, j
    i, j = symbols('i j', cls=Dummy)
    # 断言 substitute_dummies 函数能正确化简表达式：att(i, j) * NO(Fd(i)*F(j)) - att(j, i) * NO(Fd(j)*F(i)) = 0
    assert substitute_dummies(att(i, j)*NO(Fd(i)*F(j))
                - att(j, i)*NO(Fd(j)*F(i))) == 0


# 定义测试函数，测试 substitute_dummies 函数在不含 NO 操作符的表达式中的表现
def test_substitute_dummies_SQ_operator():
    # 定义虚拟变量 i, j
    i, j = symbols('i j', cls=Dummy)
    # 断言 substitute_dummies 函数能正确化简表达式：att(i, j)*Fd(i)*F(j) - att(j, i)*Fd(j)*F(i) = 0
    assert substitute_dummies(att(i, j)*Fd(i)*F(j)
                - att(j, i)*Fd(j)*F(i)) == 0


# 定
    # 对于给定的列表 [i, j, k, l]，生成其所有的排列组合，并逐个赋值给变量 permut
    for permut in variations([i, j, k, l], 4):
        # 使用函数 f 对当前排列 permut 进行调用，并与 f(i, j, k, l) 的结果相减，
        # 然后调用 substitute_dummies 函数处理结果，确保最终结果为 0
        assert substitute_dummies(f(*permut) - f(i, j, k, l)) == 0
def test_dummy_order_inner_outer_lines_VT1T1T1():
    # 定义下 Fermi 表示符号 i
    ii = symbols('i', below_fermi=True)
    # 定义上 Fermi 表示符号 a
    aa = symbols('a', above_fermi=True)
    # 定义下 Fermi 虚拟符号 k 和 l
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    # 定义上 Fermi 虚拟符号 c 和 d
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # 定义函数符号 v 和 t
    v = Function('v')
    t = Function('t')
    # 获取有序虚拟符号列表的函数
    dums = _get_ordered_dummies

    # Coupled-Cluster T1 terms with V*T1*T1*T1
    # t^{a}_{k} t^{c}_{i} t^{d}_{l} v^{lk}_{dc}
    # 根据 V*T1*T1*T1 的顺序列出 Coupled-Cluster T1 项表达式
    exprs = [
        # 对 v 和 t 进行置换 <=> 交换内部线，不考虑 v 中的对称性
        v(k, l, c, d)*t(c, ii)*t(d, l)*t(aa, k),
        v(l, k, c, d)*t(c, ii)*t(d, k)*t(aa, l),
        v(k, l, d, c)*t(d, ii)*t(c, l)*t(aa, k),
        v(l, k, d, c)*t(d, ii)*t(c, k)*t(aa, l),
    ]
    # 断言表达式的第一个排列与后续排列的虚拟符号顺序不同
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        # 断言进行虚拟符号的替换后，第一个排列与后续排列不相等
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_dummy_order_inner_outer_lines_VT1T1T1T1():
    # 定义下 Fermi 表示符号 i 和 j
    ii, jj = symbols('i j', below_fermi=True)
    # 定义上 Fermi 表示符号 a 和 b
    aa, bb = symbols('a b', above_fermi=True)
    # 定义下 Fermi 虚拟符号 k 和 l
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    # 定义上 Fermi 虚拟符号 c 和 d
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # 定义函数符号 v 和 t
    v = Function('v')
    t = Function('t')
    # 获取有序虚拟符号列表的函数
    dums = _get_ordered_dummies

    # Coupled-Cluster T2 terms with V*T1*T1*T1*T1
    # 根据 V*T1*T1*T1*T1 的顺序列出 Coupled-Cluster T2 项表达式
    exprs = [
        # 对 t 进行置换 <=> 交换外部线，除非 v 具有特定的对称性，否则不等价
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(k, l, c, d)*t(c, jj)*t(d, ii)*t(aa, k)*t(bb, l),
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(bb, k)*t(aa, l),
        v(k, l, c, d)*t(c, jj)*t(d, ii)*t(bb, k)*t(aa, l),
    ]
    # 断言表达式的第一个排列与后续排列的虚拟符号顺序不同
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        # 断言进行虚拟符号的替换后，第一个排列与后续排列不相等
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # 重新定义表达式以测试更多置换情况
    exprs = [
        # 对 v 进行置换 <=> 交换外部线，除非 v 具有特定的对称性，否则不等价
        #
        # 注意，与上面不同的是，这些排列具有相同的虚拟符号顺序。这是因为接近外部指标
        # 对规范虚拟符号顺序的影响比因子上的虚拟符号位置更大。事实上，这些项在结构上
        # 与上面进行的虚拟符号替换的结果相似。
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(l, k, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(k, l, d, c)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
        v(l, k, d, c)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),
    ]
    # 断言表达式的第一个排列与后续排列的虚拟符号顺序相同
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        # 断言进行虚拟符号的替换后，第一个排列与后续排列不相等
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [
        # 定义一个包含四个表达式的列表，这些表达式描述了一些特定的数学操作
        # permut t and v <=> swapping internal lines, equivalent.
        # Canonical dummy order is different, and a consistent
        # substitution reveals the equivalence.
        # 这些表达式表示了在交换内部线路时，两种操作是等效的。标准的虚数顺序不同，
        # 一致的替换揭示了它们的等价性。
        v(k, l, c, d)*t(c, ii)*t(d, jj)*t(aa, k)*t(bb, l),  # 第一个表达式
        v(k, l, d, c)*t(c, jj)*t(d, ii)*t(aa, k)*t(bb, l),  # 第二个表达式
        v(l, k, c, d)*t(c, ii)*t(d, jj)*t(bb, k)*t(aa, l),  # 第三个表达式
        v(l, k, d, c)*t(c, jj)*t(d, ii)*t(bb, k)*t(aa, l),  # 第四个表达式
    ]
    # 对于 exprs 中的每个排列（除了第一个表达式），进行以下断言：
    for permut in exprs[1:]:
        # 确保第一个表达式与当前排列的虚数顺序不同
        assert dums(exprs[0]) != dums(permut)
        # 确保对第一个表达式和当前排列进行虚数替换后结果相同
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
def test_get_subNO():
    # 定义符号变量 p, q, r
    p, q, r = symbols('p,q,r')
    # 断言表达式，验证 NO(F(p)*F(q)*F(r)) 的子表达式获取结果
    assert NO(F(p)*F(q)*F(r)).get_subNO(1) == NO(F(p)*F(r))
    assert NO(F(p)*F(q)*F(r)).get_subNO(0) == NO(F(q)*F(r))
    assert NO(F(p)*F(q)*F(r)).get_subNO(2) == NO(F(p)*F(q))


def test_equivalent_internal_lines_VT1T1():
    # 定义符号变量 i, j, k, l，其中 i, j 为费米子以下的虚拟自由度
    # a, b, c, d 为费米子以上的虚拟自由度
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    # 定义函数符号 v 和 t
    v = Function('v')
    t = Function('t')
    # 获取有序虚拟自由度的函数
    dums = _get_ordered_dummies

    # 表达式列表，各种 v 排列，不同虚拟自由度顺序，不等效
    exprs = [
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(j, i, a, b)*t(a, i)*t(b, j),
        v(i, j, b, a)*t(a, i)*t(b, j),
    ]
    for permut in exprs[1:]:
        # 断言有序虚拟自由度不相等
        assert dums(exprs[0]) != dums(permut)
        # 断言替换虚拟自由度后不等效
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # 表达式列表，各种 v 排列，同一虚拟自由度顺序，等效
    exprs = [
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(j, i, b, a)*t(a, i)*t(b, j),
    ]
    for permut in exprs[1:]:
        # 断言有序虚拟自由度不相等
        assert dums(exprs[0]) != dums(permut)
        # 断言替换虚拟自由度后等效
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    # 表达式列表，各种 t 排列，相同虚拟自由度顺序，不等效
    exprs = [
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(i, j, a, b)*t(b, i)*t(a, j),
    ]
    for permut in exprs[1:]:
        # 断言有序虚拟自由度相等
        assert dums(exprs[0]) == dums(permut)
        # 断言替换虚拟自由度后不等效
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # 表达式列表，各种 v 和 t 排列，不同虚拟自由度顺序，等效
    exprs = [
        v(i, j, a, b)*t(a, i)*t(b, j),
        v(j, i, a, b)*t(a, j)*t(b, i),
        v(i, j, b, a)*t(b, i)*t(a, j),
        v(j, i, b, a)*t(b, j)*t(a, i),
    ]
    for permut in exprs[1:]:
        # 断言有序虚拟自由度不相等
        assert dums(exprs[0]) != dums(permut)
        # 断言替换虚拟自由度后等效
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_equivalent_internal_lines_VT2conjT2():
    # this diagram requires special handling in TCE
    # 定义符号变量 i, j, k, l, m, n 为费米子以下的虚拟自由度
    # a, b, c, d, e, f 为费米子以上的虚拟自由度
    # p1, p2, p3, p4 为费米子以上的虚拟自由度
    # h1, h2, h3, h4 为费米子以下的虚拟自由度
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    # 导入 permutations 函数
    from sympy.utilities.iterables import variations

    # 定义函数符号 v 和 t
    v = Function('v')
    t = Function('t')
    # 获取有序虚拟自由度的函数
    dums = _get_ordered_dummies

    # v(abcd)t(abij)t(ijcd) 模板
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(i, j, p3, p4)
    # 生成所有 a, b, c, d 的排列组合
    permutator = variations([a, b, c, d], 4)
    # 设定基础表达式为第一个排列
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 对于每个排列
    for permut in permutator:
        # 生成替换列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 生成表达式
        expr = template.subs(subslist)
        # 断言有序虚拟自由度不相等
        assert dums(base) != dums(expr)
        # 断言替换虚拟自由度后等效
        assert substitute_dummies(expr) == substitute_dummies(base)

    # v(abcd)t(abji)t(jiac) 模板
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(j, i, p3, p4)
    # 生成所有 a, b, c, d 的排列组合
    permutator = variations([a, b, c, d], 4)
    # 设定基础表达式为第一个排列
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))


这段代码中，每个函数和循环的作用都有详细的注释，以帮助理解代码的功能和逻辑。
    # 对于每个排列，生成替换列表，应用到模板表达式中
    for permut in permutator:
        # 使用每个排列生成一个包含替换项的列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 根据替换列表替换模板表达式中的符号
        expr = template.subs(subslist)
        # 断言：基础表达式和替换后的表达式的虚拟指标不相等
        assert dums(base) != dums(expr)
        # 断言：替换虚拟指标后，基础表达式和替换后的表达式相等
        assert substitute_dummies(expr) == substitute_dummies(base)
    
    # 定义模板表达式：v(abcd)t(abij)t(jicd)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(j, i, p3, p4)
    # 生成四个元素的排列
    permutator = variations([a, b, c, d], 4)
    # 使用下一个排列生成基础表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    
    # 对于每个排列，生成替换列表，应用到模板表达式中
    for permut in permutator:
        # 使用每个排列生成一个包含替换项的列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 根据替换列表替换模板表达式中的符号
        expr = template.subs(subslist)
        # 断言：基础表达式和替换后的表达式的虚拟指标不相等
        assert dums(base) != dums(expr)
        # 断言：替换虚拟指标后，基础表达式和替换后的表达式相等
        assert substitute_dummies(expr) == substitute_dummies(base)
    
    # 重新定义模板表达式：v(abcd)t(abij)t(jicd) 中的 t 参数交换
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(i, j, p3, p4)
    # 生成四个元素的排列
    permutator = variations([a, b, c, d], 4)
    # 使用下一个排列生成基础表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    
    # 对于每个排列，生成替换列表，应用到模板表达式中
    for permut in permutator:
        # 使用每个排列生成一个包含替换项的列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 根据替换列表替换模板表达式中的符号
        expr = template.subs(subslist)
        # 断言：基础表达式和替换后的表达式的虚拟指标不相等
        assert dums(base) != dums(expr)
        # 断言：替换虚拟指标后，基础表达式和替换后的表达式相等
        assert substitute_dummies(expr) == substitute_dummies(base)
def test_equivalent_internal_lines_VT2conjT2_ambiguous_order():
    # 定义一个测试函数，用于检查在特定情况下表达式的等价性

    # 定义一些符号变量，分别表示费米面以下和以上的虚拟粒子
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    # 导入变化函数
    from sympy.utilities.iterables import variations

    # 定义函数符号
    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies

    # 第一个模板表达式 v(abcd)t(abij)t(cdij)
    template = v(p1, p2, p3, p4)*t(p1, p2, i, j)*t(p3, p4, i, j)
    
    # 生成变量排列的迭代器
    permutator = variations([a, b, c, d], 4)
    
    # 应用第一个排列生成基础表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    
    # 对每个排列进行循环检查
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        
        # 断言基础表达式和当前表达式在虚拟粒子的顺序上不等价
        assert dums(base) != dums(expr)
        
        # 断言经过虚拟粒子替换后，基础表达式和当前表达式等价
        assert substitute_dummies(expr) == substitute_dummies(base)
    
    # 第二个模板表达式 v(abcd)t(abji)t(cdij)
    template = v(p1, p2, p3, p4)*t(p1, p2, j, i)*t(p3, p4, i, j)
    permutator = variations([a, b, c, d], 4)
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    
    # 对每个排列进行循环检查
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        
        # 断言基础表达式和当前表达式在虚拟粒子的顺序上不等价
        assert dums(base) != dums(expr)
        
        # 断言经过虚拟粒子替换后，基础表达式和当前表达式等价
        assert substitute_dummies(expr) == substitute_dummies(base)


def test_equivalent_internal_lines_VT2():
    # 定义一个测试函数，用于检查在特定情况下表达式的等价性

    # 定义一些符号变量，分别表示费米面以下和以上的虚拟粒子
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    # 定义函数符号
    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies
    
    # 第一组表达式集合，测试虚拟粒子顺序不同的情况
    exprs = [
        v(i, j, a, b)*t(a, b, i, j),
        v(j, i, a, b)*t(a, b, i, j),
        v(i, j, b, a)*t(a, b, i, j),
        v(j, i, b, a)*t(a, b, i, j),
    ]
    
    # 对每个表达式进行循环检查
    for permut in exprs[1:]:
        # 断言第一个表达式和当前表达式在虚拟粒子的顺序上等价
        assert dums(exprs[0]) == dums(permut)
        
        # 断言经过虚拟粒子替换后，第一个表达式和当前表达式不等价
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    
    # 第二组表达式集合，测试 t 的顺序不同的情况
    exprs = [
        v(i, j, a, b)*t(a, b, i, j),
        v(i, j, a, b)*t(b, a, i, j),
        v(i, j, a, b)*t(a, b, j, i),
        v(i, j, a, b)*t(b, a, j, i),
    ]
    
    # 对每个表达式进行循环检查
    for permut in exprs[1:]:
        # 断言第一个表达式和当前表达式在虚拟粒子的顺序上不等价
        assert dums(exprs[0]) != dums(permut)
        
        # 断言经过虚拟粒子替换后，第一个表达式和当前表达式不等价
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [  # 生成包含四个表达式的列表，表达式进行了v和t的置换。虚拟变量的重标记应该是等效的。
        v(i, j, a, b)*t(a, b, i, j),   # 表达式1: v函数和t函数按照指定顺序的乘积
        v(j, i, a, b)*t(a, b, j, i),   # 表达式2: v函数和t函数的参数交换顺序后的乘积
        v(i, j, b, a)*t(b, a, i, j),   # 表达式3: v函数和t函数的参数顺序再次交换的乘积
        v(j, i, b, a)*t(b, a, j, i),   # 表达式4: v函数和t函数的参数交换顺序后再次交换的乘积
    ]
    for permut in exprs[1:]:
        # 断言：检查第一个表达式和当前遍历的表达式的虚拟变量不同
        assert dums(exprs[0]) != dums(permut)
        # 断言：检查用统一虚拟变量替换后，第一个表达式与当前遍历的表达式是否相等
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
# 定义一个测试函数，用于验证在不同上下标记号和虚拟符号约束下的表达式排列顺序
def test_internal_external_VT2T2():
    # 定义符号 ii 和 jj，这些符号位于费米面以下
    ii, jj = symbols('i j', below_fermi=True)
    # 定义符号 aa 和 bb，这些符号位于费米面以上
    aa, bb = symbols('a b', above_fermi=True)
    # 定义符号 k 和 l，这些符号位于费米面以下，同时定义为虚拟符号
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    # 定义符号 c 和 d，这些符号位于费米面以上，同时定义为虚拟符号
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # 定义函数 v 和 t
    v = Function('v')
    t = Function('t')
    # 获取有序虚拟符号的函数引用
    dums = _get_ordered_dummies

    # 定义表达式列表，包含四个表达式
    exprs = [
        v(k, l, c, d)*t(aa, c, ii, k)*t(bb, d, jj, l),
        v(l, k, c, d)*t(aa, c, ii, l)*t(bb, d, jj, k),
        v(k, l, d, c)*t(aa, d, ii, k)*t(bb, c, jj, l),
        v(l, k, d, c)*t(aa, d, ii, l)*t(bb, c, jj, k),
    ]

    # 对每个表达式进行循环
    for permut in exprs[1:]:
        # 断言：每个表达式的有序虚拟符号列表不同
        assert dums(exprs[0]) != dums(permut)
        # 断言：替换虚拟符号后每个表达式相等
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


# 定义一个测试函数，验证在不同符号约束下的表达式排列顺序
def test_internal_external_pqrs():
    # 定义符号 ii 和 jj，这些符号没有特定的上下标记号约束
    ii, jj = symbols('i j')
    # 定义符号 aa 和 bb，这些符号没有特定的上下标记号约束
    aa, bb = symbols('a b')
    # 定义符号 k 和 l，这些符号没有特定的上下标记号约束，同时定义为虚拟符号
    k, l = symbols('k l', cls=Dummy)
    # 定义符号 c 和 d，这些符号没有特定的上下标记号约束，同时定义为虚拟符号
    c, d = symbols('c d', cls=Dummy)

    # 定义函数 v 和 t
    v = Function('v')
    t = Function('t')
    # 获取有序虚拟符号的函数引用
    dums = _get_ordered_dummies

    # 定义表达式列表，包含四个表达式
    exprs = [
        v(k, l, c, d)*t(aa, c, ii, k)*t(bb, d, jj, l),
        v(l, k, c, d)*t(aa, c, ii, l)*t(bb, d, jj, k),
        v(k, l, d, c)*t(aa, d, ii, k)*t(bb, c, jj, l),
        v(l, k, d, c)*t(aa, d, ii, l)*t(bb, c, jj, k),
    ]

    # 对每个表达式进行循环
    for permut in exprs[1:]:
        # 断言：每个表达式的有序虚拟符号列表不同
        assert dums(exprs[0]) != dums(permut)
        # 断言：替换虚拟符号后每个表达式相等
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


# 定义一个测试函数，验证在指定符号约束下的表达式排列顺序
def test_dummy_order_well_defined():
    # 定义符号 aa 和 bb，这些符号位于费米面以上
    aa, bb = symbols('a b', above_fermi=True)
    # 定义符号 k, l, m，这些符号位于费米面以下，同时定义为虚拟符号
    k, l, m = symbols('k l m', below_fermi=True, cls=Dummy)
    # 定义符号 c 和 d，这些符号位于费米面以上，同时定义为虚拟符号
    c, d = symbols('c d', above_fermi=True, cls=Dummy)
    # 定义符号 p 和 q，这些符号没有特定的上下标记号约束，同时定义为虚拟符号
    p, q = symbols('p q', cls=Dummy)

    # 定义函数 A, B, C
    A = Function('A')
    B = Function('B')
    C = Function('C')
    # 获取有序虚拟符号的函数引用
    dums = _get_ordered_dummies

    # 断言：在特定排序中，每个表达式的有序虚拟符号列表
    # pos in first factor determines sort order
    assert dums(A(k, l)*B(l, k)) == [k, l]
    assert dums(A(l, k)*B(l, k)) == [l, k]
    assert dums(A(k, l)*B(k, l)) == [k, l]
    assert dums(A(l, k)*B(k, l)) == [l, k]

    # factors involving the index
    assert dums(A(k, l)*B(l, m)*C(k, m)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(k, l)*B(l, m)*C(m, k) 返回正确的结果 [l, k, m]
    assert dums(A(k, l)*B(l, m)*C(m, k)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(l, k)*B(l, m)*C(k, m) 返回正确的结果 [l, k, m]
    assert dums(A(l, k)*B(l, m)*C(k, m)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(l, k)*B(l, m)*C(m, k) 返回正确的结果 [l, k, m]
    assert dums(A(l, k)*B(l, m)*C(m, k)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(k, l)*B(m, l)*C(k, m) 返回正确的结果 [l, k, m]
    assert dums(A(k, l)*B(m, l)*C(k, m)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(k, l)*B(m, l)*C(m, k) 返回正确的结果 [l, k, m]
    assert dums(A(k, l)*B(m, l)*C(m, k)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(l, k)*B(m, l)*C(k, m) 返回正确的结果 [l, k, m]
    assert dums(A(l, k)*B(m, l)*C(k, m)) == [l, k, m]
    # 断言：验证 dums 函数对给定的 A(l, k)*B(m, l)*C(m, k) 返回正确的结果 [l, k, m]
    assert dums(A(l, k)*B(m, l)*C(m, k)) == [l, k, m]

    # 使用非虚数确定的因子顺序进行验证，断言验证 dums 函数对给定的 A(k, aa, l)*A(l, bb, m)*A(bb, k, m) 返回正确的结果 [l, k, m]
    assert dums(A(k, aa, l)*A(l, bb, m)*A(bb, k, m)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(k, aa, l)*A(l, bb, m)*A(bb, m, k) 返回正确的结果 [l, k, m]
    assert dums(A(k, aa, l)*A(l, bb, m)*A(bb, m, k)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(k, aa, l)*A(m, bb, l)*A(bb, k, m) 返回正确的结果 [l, k, m]
    assert dums(A(k, aa, l)*A(m, bb, l)*A(bb, k, m)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(k, aa, l)*A(m, bb, l)*A(bb, m, k) 返回正确的结果 [l, k, m]
    assert dums(A(k, aa, l)*A(m, bb, l)*A(bb, m, k)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(l, aa, k)*A(l, bb, m)*A(bb, k, m) 返回正确的结果 [l, k, m]
    assert dums(A(l, aa, k)*A(l, bb, m)*A(bb, k, m)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(l, aa, k)*A(l, bb, m)*A(bb, m, k) 返回正确的结果 [l, k, m]
    assert dums(A(l, aa, k)*A(l, bb, m)*A(bb, m, k)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(l, aa, k)*A(m, bb, l)*A(bb, k, m) 返回正确的结果 [l, k, m]
    assert dums(A(l, aa, k)*A(m, bb, l)*A(bb, k, m)) == [l, k, m]
    # 断言验证 dums 函数对给定的 A(l, aa, k)*A(m, bb, l)*A(bb, m, k) 返回正确的结果 [l, k, m]
    assert dums(A(l, aa, k)*A(m, bb, l)*A(bb, m, k)) == [l, k, m]

    # 索引范围的断言验证，断言验证 dums 函数对给定的 A(p, c, k)*B(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(A(p, c, k)*B(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 A(p, k, c)*B(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(A(p, k, c)*B(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 A(c, k, p)*B(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(A(c, k, p)*B(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 A(c, p, k)*B(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(A(c, p, k)*B(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 A(k, c, p)*B(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(A(k, c, p)*B(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 A(k, p, c)*B(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(A(k, p, c)*B(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 B(p, c, k)*A(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(B(p, c, k)*A(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 B(p, k, c)*A(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(B(p, k, c)*A(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 B(c, k, p)*A(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(B(c, k, p)*A(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 B(c, p, k)*A(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(B(c, p, k)*A(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 B(k, c, p)*A(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(B(k, c, p)*A(p, c, k)) == [k, c, p]
    # 断言验证 dums 函数对给定的 B(k, p, c)*A(p, c, k) 返回正确的结果 [k, c, p]
    assert dums(B(k, p, c)*A(p, c, k)) == [k, c, p]
# 定义一个测试函数，用于测试含有 Dummy 符号的排列组合
def test_dummy_order_ambiguous():
    # 定义符号 aa 和 bb，位于费米面以上
    aa, bb = symbols('a b', above_fermi=True)
    # 定义符号 i, j, k, l, m，位于费米面以下，使用 Dummy 类型
    i, j, k, l, m = symbols('i j k l m', below_fermi=True, cls=Dummy)
    # 定义符号 a, b, c, d, e，位于费米面以上，使用 Dummy 类型
    a, b, c, d, e = symbols('a b c d e', above_fermi=True, cls=Dummy)
    # 定义符号 p, q，使用 Dummy 类型
    p, q = symbols('p q', cls=Dummy)
    # 定义符号 p1, p2, p3, p4，位于费米面以上，使用 Dummy 类型
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    # 定义符号 p5, p6, p7, p8，位于费米面以上，使用 Dummy 类型
    p5, p6, p7, p8 = symbols('p5 p6 p7 p8', above_fermi=True, cls=Dummy)
    # 定义符号 h1, h2, h3, h4，位于费米面以下，使用 Dummy 类型
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)
    # 定义符号 h5, h6, h7, h8，位于费米面以下，使用 Dummy 类型
    h5, h6, h7, h8 = symbols('h5 h6 h7 h8', below_fermi=True, cls=Dummy)

    # 定义函数符号 A 和 B
    A = Function('A')
    B = Function('B')

    # 导入 variations 函数，用于生成给定元素的排列
    from sympy.utilities.iterables import variations

    # 第一个模板表达式：A*A*A*A*B，使用 p5 和 p4 的顺序确定其余参数的排列
    template = A(p1, p2)*A(p4, p1)*A(p2, p3)*A(p3, p5)*B(p5, p4)
    # 生成给定元素的所有排列组合
    permutator = variations([a, b, c, d, e], 5)
    # 基准表达式为模板表达式使用第一个排列的结果
    base = template.subs(zip([p1, p2, p3, p4, p5], next(permutator)))
    # 遍历所有排列组合
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        # 断言：使用 substitute_dummies 函数替换 Dummy 符号后的表达式应与基准表达式相等
        assert substitute_dummies(expr) == substitute_dummies(base)

    # 第二个模板表达式：A*A*A*A*A，随机分配一个索引并确定其余参数的排列
    template = A(p1, p2)*A(p4, p1)*A(p2, p3)*A(p3, p5)*A(p5, p4)
    permutator = variations([a, b, c, d, e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)

    # 第三个模板表达式：A*A*A，使用 p5 和 p4 的顺序确定其余参数的排列
    template = A(p1, p2, p4, p1)*A(p2, p3, p3, p5)*A(p5, p4)
    permutator = variations([a, b, c, d, e], 5)
    base = template.subs(zip([p1, p2, p3, p4, p5], next(permutator)))
    for permut in permutator:
        subslist = zip([p1, p2, p3, p4, p5], permut)
        expr = template.subs(subslist)
        assert substitute_dummies(expr) == substitute_dummies(base)


# 定义一个返回 AntiSymmetricTensor 对象的函数 atv
def atv(*args):
    return AntiSymmetricTensor('v', args[:2], args[2:])


# 定义一个根据参数个数返回 AntiSymmetricTensor 对象的函数 att
def att(*args):
    if len(args) == 4:
        return AntiSymmetricTensor('t', args[:2], args[2:])
    elif len(args) == 2:
        return AntiSymmetricTensor('t', (args[0],), (args[1],))


# 定义一个测试函数，用于测试内部和外部线性组合的排列
def test_dummy_order_inner_outer_lines_VT1T1T1_AT():
    # 定义符号 ii，位于费米面以下
    ii = symbols('i', below_fermi=True)
    # 定义符号 aa，位于费米面以上
    aa = symbols('a', above_fermi=True)
    # 定义符号 k, l，位于费米面以下，使用 Dummy 类型
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    # 定义符号 c, d，位于费米面以上，使用 Dummy 类型
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # 定义表达式列表，用于存储不同的排列组合
    exprs = [
        # v^{lk}_{dc} t^{c}_{i} t^{d}_{l} t^{a}_{k}
        atv(k, l, c, d)*att(c, ii)*att(d, l)*att(aa, k),
        # v^{lk}_{dc} t^{c}_{i} t^{d}_{k} t^{a}_{l}
        atv(l, k, c, d)*att(c, ii)*att(d, k)*att(aa, l),
        # v^{lk}_{dc} t^{d}_{i} t^{c}_{l} t^{a}_{k}
        atv(k, l, d, c)*att(d, ii)*att(c, l)*att(aa, k),
        # v^{lk}_{dc} t^{d}_{i} t^{c}_{k} t^{a}_{l}
        atv(l, k, d, c)*att(d, ii)*att(c, k)*att(aa, l),
    ]
    # 对于列表 exprs 中从第二个元素开始的每个排列 permut 进行循环
    for permut in exprs[1:]:
        # 断言：验证表达式列表的第一个元素经过虚拟变量替换后与当前排列 permut 经过相同替换后是否相等
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
def test_dummy_order_inner_outer_lines_VT1T1T1T1_AT():
    # 定义符号变量 ii 和 jj，表示在费米面以下的自由指标
    ii, jj = symbols('i j', below_fermi=True)
    # 定义符号变量 aa 和 bb，表示在费米面以上的自由指标
    aa, bb = symbols('a b', above_fermi=True)
    # 定义符号变量 k 和 l，表示在费米面以下的虚指标，使用 Dummy 类
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    # 定义符号变量 c 和 d，表示在费米面以上的虚指标，使用 Dummy 类
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # Coupled-Cluster T2 terms with V*T1*T1*T1*T1
    # non-equivalent substitutions (change of sign)
    # 定义表达式列表，表示不等价的替换情况（符号变量顺序变化导致符号改变）
    exprs = [
        # 根据外线交换的不同排列，对应不同的符号变化
        atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(aa, k)*att(bb, l),
        atv(k, l, c, d)*att(c, jj)*att(d, ii)*att(aa, k)*att(bb, l),
        atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(bb, k)*att(aa, l),
    ]
    # 对于表达式列表中的每个排列，验证其与第一个表达式的相反性
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == -substitute_dummies(permut)

    # equivalent substitutions
    # 等价的替换情况
    exprs = [
        atv(k, l, c, d)*att(c, ii)*att(d, jj)*att(aa, k)*att(bb, l),
        # 根据外线交换的不同排列，对应不同的符号变化
        atv(k, l, c, d)*att(c, jj)*att(d, ii)*att(bb, k)*att(aa, l),
    ]
    # 对于表达式列表中的每个排列，验证其与第一个表达式的相等性
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_equivalent_internal_lines_VT1T1_AT():
    # 定义符号变量 i, j, k, l，表示在费米面以下的自由指标，使用 Dummy 类
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    # 定义符号变量 a, b, c, d，表示在费米面以上的自由指标，使用 Dummy 类
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    # permute v.  Different dummy order. Not equivalent.
    # 对 v 进行排列，符号变量顺序不同，不等价
    exprs = [
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(j, i, a, b)*att(a, i)*att(b, j),
        atv(i, j, b, a)*att(a, i)*att(b, j),
    ]
    # 对于表达式列表中的每个排列，验证其不等价
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # permute v.  Different dummy order. Equivalent
    # 对 v 进行排列，符号变量顺序不同，但等价
    exprs = [
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(j, i, b, a)*att(a, i)*att(b, j),
    ]
    # 对于表达式列表中的每个排列，验证其等价
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    # permute t.  Same dummy order, not equivalent.
    # 对 t 进行排列，符号变量顺序相同，但不等价
    exprs = [
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(i, j, a, b)*att(b, i)*att(a, j),
    ]
    # 对于表达式列表中的每个排列，验证其不等价
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # permute v and t.  Different dummy order, equivalent
    # 对 v 和 t 同时进行排列，符号变量顺序不同，但等价
    exprs = [
        atv(i, j, a, b)*att(a, i)*att(b, j),
        atv(j, i, a, b)*att(a, j)*att(b, i),
        atv(i, j, b, a)*att(b, i)*att(a, j),
        atv(j, i, b, a)*att(b, j)*att(a, i),
    ]
    # 对于表达式列表中的每个排列，验证其等价
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_equivalent_internal_lines_VT2conjT2_AT():
    # this diagram requires special handling in TCE
    # 定义符号变量 i, j, k, l, m, n，表示在费米面以下的自由指标，使用 Dummy 类
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    # 定义符号变量 a, b, c, d, e, f，表示在费米面以上的自由指标，使用 Dummy 类
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    # 定义符号变量 p1, p2, p3, p4，表示在费米面以上的自由指标，使用 Dummy 类
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    # 定义符号变量 h1, h2, h3, h4，表示在费米面以下的自由指标，使用 Dummy 类
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    # atv(abcd)att(abij)att(ijcd)
    # 创建模板表达式，这里使用了 atv, att 函数，组合成一个复杂的表达式 template
    template = atv(p1, p2, p3, p4)*att(p1, p2, p3, p4)*att(p3, p4, p1, p2)
    # 生成所有四个元素的排列组合
    permutator = variations([a, b, c, d], 4)
    # 用排列的第一个组合替换模板中的变量，生成基准表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 对所有排列进行迭代
    for permut in permutator:
        # 生成变量替换列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 用当前排列替换模板中的变量，生成当前表达式
        expr = template.subs(subslist)
        # 断言当前表达式与基准表达式相等，去除假名后
        assert substitute_dummies(expr) == substitute_dummies(base)

    # 创建另一个模板表达式，类似前一个模板，但 att 的参数顺序略有不同
    template = atv(p1, p2, p3, p4)*att(p1, p2, p4, p3)*att(p4, p3, p1, p2)
    # 生成所有四个元素的排列组合
    permutator = variations([a, b, c, d], 4)
    # 用排列的第一个组合替换模板中的变量，生成基准表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 对所有排列进行迭代
    for permut in permutator:
        # 生成变量替换列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 用当前排列替换模板中的变量，生成当前表达式
        expr = template.subs(subslist)
        # 断言当前表达式与基准表达式相等，去除假名后
        assert substitute_dummies(expr) == substitute_dummies(base)

    # 创建第三个模板表达式，这里使用了 atv, att 函数，组合成一个复杂的表达式 template
    template = atv(p1, p2, p3, p4)*att(p1, p2, p3, p4)*att(p3, p4, p1, p2)
    # 生成所有四个元素的排列组合
    permutator = variations([a, b, c, d], 4)
    # 用排列的第一个组合替换模板中的变量，生成基准表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 对所有排列进行迭代
    for permut in permutator:
        # 生成变量替换列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 用当前排列替换模板中的变量，生成当前表达式
        expr = template.subs(subslist)
        # 断言当前表达式与基准表达式相等，去除假名后
        assert substitute_dummies(expr) == substitute_dummies(base)

    # 创建第四个模板表达式，类似前一个模板，但 att 的参数顺序略有不同
    template = atv(p1, p2, p3, p4)*att(p1, p2, p4, p3)*att(p4, p3, p1, p2)
    # 生成所有四个元素的排列组合
    permutator = variations([a, b, c, d], 4)
    # 用排列的第一个组合替换模板中的变量，生成基准表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 对所有排列进行迭代
    for permut in permutator:
        # 生成变量替换列表
        subslist = zip([p1, p2, p3, p4], permut)
        # 用当前排列替换模板中的变量，生成当前表达式
        expr = template.subs(subslist)
        # 断言当前表达式与基准表达式相等，去除假名后
        assert substitute_dummies(expr) == substitute_dummies(base)
def test_equivalent_internal_lines_VT2conjT2_ambiguous_order_AT():
    # 这些表达式调用 _determine_ambiguous()，因为仅凭键无法明确有序排列虚拟的变量
    # 定义一系列虚拟变量，分为上费米面和下费米面的符号
    i, j, k, l, m, n = symbols('i j k l m n', below_fermi=True, cls=Dummy)
    a, b, c, d, e, f = symbols('a b c d e f', above_fermi=True, cls=Dummy)
    p1, p2, p3, p4 = symbols('p1 p2 p3 p4', above_fermi=True, cls=Dummy)
    h1, h2, h3, h4 = symbols('h1 h2 h3 h4', below_fermi=True, cls=Dummy)

    from sympy.utilities.iterables import variations

    # 创建模板表达式 atv(abcd)att(abij)att(cdij)
    template = atv(p1, p2, p3, p4)*att(p1, p2, i, j)*att(p3, p4, i, j)
    # 生成虚拟变量的所有排列组合
    permutator = variations([a, b, c, d], 4)
    # 使用第一个排列替换模板中的变量，形成基准表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 遍历剩余的排列组合
    for permut in permutator:
        # 用当前排列替换模板中的变量
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        # 断言当前表达式与基准表达式在虚拟变量重标记后的等效性
        assert substitute_dummies(expr) == substitute_dummies(base)

    # 创建第二个模板表达式 atv(abcd)att(abji)att(cdij)
    template = atv(p1, p2, p3, p4)*att(p1, p2, j, i)*att(p3, p4, i, j)
    # 重新生成虚拟变量的所有排列组合
    permutator = variations([a, b, c, d], 4)
    # 使用第一个排列替换模板中的变量，形成基准表达式
    base = template.subs(zip([p1, p2, p3, p4], next(permutator)))
    # 遍历剩余的排列组合
    for permut in permutator:
        # 用当前排列替换模板中的变量
        subslist = zip([p1, p2, p3, p4], permut)
        expr = template.subs(subslist)
        # 断言当前表达式与基准表达式在虚拟变量重标记后的等效性
        assert substitute_dummies(expr) == substitute_dummies(base)


def test_equivalent_internal_lines_VT2_AT():
    # 定义一系列虚拟变量，分为上费米面和下费米面的符号
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)

    # 比较不同排列的表达式
    exprs = [
        # v 的排列不同，表达式不等价
        atv(i, j, a, b)*att(a, b, i, j),
        atv(j, i, a, b)*att(a, b, i, j),
        atv(i, j, b, a)*att(a, b, i, j),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # 比较 t 的排列
    exprs = [
        atv(i, j, a, b)*att(a, b, i, j),
        atv(i, j, a, b)*att(b, a, i, j),
        atv(i, j, a, b)*att(a, b, j, i),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)

    # 比较 v 和 t 的排列，虚拟变量重标记后等价
    exprs = [
        atv(i, j, a, b)*att(a, b, i, j),
        atv(j, i, a, b)*att(a, b, j, i),
        atv(i, j, b, a)*att(b, a, i, j),
        atv(j, i, b, a)*att(b, a, j, i),
    ]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


def test_internal_external_VT2T2_AT():
    # 定义虚拟变量，分别为上费米面和下费米面的符号
    ii, jj = symbols('i j', below_fermi=True)
    aa, bb = symbols('a b', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)

    # 创建一系列表达式，比较不同的排列
    exprs = [
        atv(k, l, c, d)*att(aa, c, ii, k)*att(bb, d, jj, l),
        atv(l, k, c, d)*att(aa, c, ii, l)*att(bb, d, jj, k),
        atv(k, l, d, c)*att(aa, d, ii, k)*att(bb, c, jj, l),
        atv(l, k, d, c)*att(aa,
    # 对于列表 `exprs` 中除第一个元素外的每个排列 `permut`，
    # 确保第一个元素和当前排列 `permut` 的变量替换后相等
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    # 将 `exprs` 重新赋值为一个包含四个数学表达式的列表
    exprs = [
        atv(k, l, c, d)*att(aa, c, ii, k)*att(d, bb, jj, l),
        atv(l, k, c, d)*att(aa, c, ii, l)*att(d, bb, jj, k),
        atv(k, l, d, c)*att(aa, d, ii, k)*att(c, bb, jj, l),
        atv(l, k, d, c)*att(aa, d, ii, l)*att(c, bb, jj, k),
    ]

    # 对于列表 `exprs` 中除第一个元素外的每个排列 `permut`，
    # 确保第一个元素和当前排列 `permut` 的变量替换后相等
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)

    # 将 `exprs` 重新赋值为一个包含四个数学表达式的列表
    exprs = [
        atv(k, l, c, d)*att(c, aa, ii, k)*att(bb, d, jj, l),
        atv(l, k, c, d)*att(c, aa, ii, l)*att(bb, d, jj, k),
        atv(k, l, d, c)*att(d, aa, ii, k)*att(bb, c, jj, l),
        atv(l, k, d, c)*att(d, aa, ii, l)*att(bb, c, jj, k),
    ]

    # 对于列表 `exprs` 中除第一个元素外的每个排列 `permut`，
    # 确保第一个元素和当前排列 `permut` 的变量替换后相等
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
# 定义测试函数，用于测试特定功能或问题
def test_internal_external_pqrs_AT():
    # 创建符号变量 i, j
    ii, jj = symbols('i j')
    # 创建符号变量 a, b
    aa, bb = symbols('a b')
    # 创建符号变量 k, l，使用 Dummy 类表示虚拟符号
    k, l = symbols('k l', cls=Dummy)
    # 创建符号变量 c, d，使用 Dummy 类表示虚拟符号
    c, d = symbols('c d', cls=Dummy)

    # 定义表达式列表，每个表达式都是一组乘积
    exprs = [
        atv(k, l, c, d)*att(aa, c, ii, k)*att(bb, d, jj, l),
        atv(l, k, c, d)*att(aa, c, ii, l)*att(bb, d, jj, k),
        atv(k, l, d, c)*att(aa, d, ii, k)*att(bb, c, jj, l),
        atv(l, k, d, c)*att(aa, d, ii, l)*att(bb, c, jj, k),
    ]

    # 对于表达式列表中的每个排列，验证其与第一个表达式的替换后结果是否相同
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)


# 定义测试函数，用于验证特定问题（issue）19661
def test_issue_19661():
    # 创建符号变量 a，其名为 '0'
    a = Symbol('0')
    # 断言：计算给定 Commutator 的 LaTeX 表示是否与预期值匹配
    assert latex(Commutator(Bd(a)**2, B(a))) == '- \\left[b_{0},{b^\\dagger_{0}}^{2}\\right]'


# 定义测试函数，用于验证特定问题的 Canonical Ordering 问题
def test_canonical_ordering_AntiSymmetricTensor():
    # 创建符号变量 v
    v = symbols("v")

    # 创建符号变量 c, d，用于表示费米面上方的虚拟符号
    c, d = symbols(('c','d'), above_fermi=True, cls=Dummy)
    # 创建符号变量 k, l，用于表示费米面下方的虚拟符号
    k, l = symbols(('k','l'), below_fermi=True, cls=Dummy)

    # 断言：验证给定 AntiSymmetricTensor 对象是否满足反对称性质
    assert AntiSymmetricTensor(v, (k, l), (d, c)) == -AntiSymmetricTensor(v, (l, k), (d, c))
```