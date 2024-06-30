# `D:\src\scipysrc\sympy\sympy\logic\tests\test_lra_theory.py`

```
# 导入 Rational、I、oo 符号和数学操作
from sympy.core.numbers import Rational, I, oo
# 导入关系运算 Eq
from sympy.core.relational import Eq
# 导入符号变量 symbols
from sympy.core.symbol import symbols
# 导入单例 S
from sympy.core.singleton import S
# 导入矩阵 Matrix 和随机矩阵 randMatrix
from sympy.matrices.dense import Matrix, randMatrix
# 导入问题假设 Q
from sympy.assumptions.ask import Q
# 导入布尔代数操作 And
from sympy.logic.boolalg import And
# 导入预定义符号 x, y, z
from sympy.abc import x, y, z
# 导入 CNF 和 EncodedCNF
from sympy.assumptions.cnf import CNF, EncodedCNF
# 导入三角函数 cos
from sympy.functions.elementary.trigonometric import cos
# 导入外部模块导入函数
from sympy.external import import_module

# 导入线性实数算法 LRA 求解器和相关类和常量
from sympy.logic.algorithms.lra_theory import LRASolver, UnhandledInput, LRARational, HANDLE_NEGATION
# 导入随机数生成函数和工具
from sympy.core.random import random, choice, randint
# 导入表达式转换函数 sympify
from sympy.core.sympify import sympify
# 导入数论生成函数 randprime
from sympy.ntheory.generate import randprime
# 导入比较运算符 StrictLessThan 和 StrictGreaterThan
from sympy.core.relational import StrictLessThan, StrictGreaterThan

# 导入 itertools 用于迭代工具函数
import itertools

# 导入测试框架函数 raises, XFAIL, skip
from sympy.testing.pytest import raises, XFAIL, skip

# 定义随机生成约束条件的函数
def make_random_problem(num_variables=2, num_constraints=2, sparsity=.1, rational=True,
                        disable_strict = False, disable_nonstrict=False, disable_equality=False):
    # 定义随机数生成函数
    def rand(sparsity=sparsity):
        # 根据稀疏度生成随机数
        if random() < sparsity:
            return sympify(0)
        # 如果是有理数，生成随机有理数
        if rational:
            int1, int2 = [randprime(0, 50) for _ in range(2)]
            return Rational(int1, int2) * choice([-1, 1])
        else:
            # 否则生成随机整数
            return randint(1, 10) * choice([-1, 1])

    # 创建符号变量列表
    variables = symbols('x1:%s' % (num_variables + 1))
    constraints = []
    # 循环生成约束条件
    for _ in range(num_constraints):
        # 左侧表达式由变量加权和与随机数构成
        lhs, rhs = sum(rand() * x for x in variables), rand(sparsity=0) # sparsity=0  bc of bug with smtlib_code
        options = []
        # 如果未禁用等式约束，添加等式约束
        if not disable_equality:
            options += [Eq(lhs, rhs)]
        # 如果未禁用非严格不等式约束，添加非严格不等式约束
        if not disable_nonstrict:
            options += [lhs <= rhs, lhs >= rhs]
        # 如果未禁用严格不等式约束，添加严格不等式约束
        if not disable_strict:
            options += [lhs < rhs, lhs > rhs]

        # 随机选择一个约束条件加入约束列表
        constraints.append(choice(options))

    return constraints

# 使用 z3 检查布尔公式是否可满足
def check_if_satisfiable_with_z3(constraints):
    # 导入 z3 模块
    from sympy.external.importtools import import_module
    # 导入 SMT-LIB 格式代码生成函数
    from sympy.printing.smtlib import smtlib_code
    # 构建布尔公式
    boolean_formula = And(*constraints)
    # 尝试导入 z3 求解器
    z3 = import_module("z3")
    if z3:
        # 生成 SMT-LIB 格式字符串
        smtlib_string = smtlib_code(boolean_formula)
        # 创建 z3 Solver 对象
        s = z3.Solver()
        # 加载 SMT-LIB 格式字符串
        s.from_string(smtlib_string)
        # 检查公式可满足性
        res = str(s.check())
        # 如果可满足返回 True，不可满足返回 False
        if res == 'sat':
            return True
        elif res == 'unsat':
            return False
        else:
            # 如果 z3 无法处理公式，抛出错误
            raise ValueError(f"z3 was not able to check the satisfiability of {boolean_formula}")

# 寻找满足有理数解的赋值
def find_rational_assignment(constr, assignment, iter=20):
    # 设置步长 ε 初始值为 1
    eps = sympify(1)

    # 迭代寻找满足条件的赋值
    for _ in range(iter):
        # 生成当前迭代的赋值副本
        assign = {key: val[0] + val[1]*eps for key, val in assignment.items()}
        try:
            # 检查是否所有约束条件都满足
            for cons in constr:
                assert cons.subs(assign) == True
            # 如果满足，返回当前赋值
            return assign
        except AssertionError:
            # 如果不满足，减小步长 ε
            eps = eps/2

    # 如果未找到满足条件的赋值，返回 None
    return None

# 将布尔公式转换为编码 CNF 格式
def boolean_formula_to_encoded_cnf(bf):
    # 使用 CNF 类的方法将布尔公式转换为 CNF 格式
    cnf = CNF.from_prop(bf)
    # 创建 EncodedCNF 的实例对象
    enc = EncodedCNF()
    # 调用 EncodedCNF 对象的 from_cnf 方法，将 cnf 参数作为输入
    enc.from_cnf(cnf)
    # 返回调用 from_cnf 方法后的 enc 对象
    return enc
def test_random_problems():
    # 导入 z3 模块
    z3 = import_module("z3")
    # 如果 z3 未安装，跳过测试
    if z3 is None:
        skip("z3 is not installed")

    # 定义特殊情况列表和符号变量
    special_cases = []; x1, x2, x3 = symbols("x1 x2 x3")
    
    # 向特殊情况列表添加约束条件列表
    special_cases.append([x1 - 3 * x2 <= -5, 6 * x1 + 4 * x2 <= 0, -7 * x1 + 3 * x2 <= 3])
    special_cases.append([-3 * x1 >= 3, Eq(4 * x1, -1)])
    special_cases.append([-4 * x1 < 4, 6 * x1 <= -6])
    special_cases.append([-3 * x2 >= 7, 6 * x1 <= -5, -3 * x2 <= -4])
    special_cases.append([x + y >= 2, x + y <= 1])
    special_cases.append([x >= 0, x + y <= 2, x + 2 * y - z >= 6])  # 来自论文例子
    special_cases.append([-2 * x1 - 2 * x2 >= 7, -9 * x1 >= 7, -6 * x1 >= 5])
    special_cases.append([2 * x1 > -3, -9 * x1 < -6, 9 * x1 <= 6])
    special_cases.append([-2*x1 < -4, 9*x1 > -9])
    special_cases.append([-6*x1 >= -1, -8*x1 + x2 >= 5, -8*x1 + 7*x2 < 4, x1 > 7])
    special_cases.append([Eq(x1, 2), Eq(5*x1, -2), Eq(-7*x2, -6), Eq(9*x1 + 10*x2, 9)])
    special_cases.append([Eq(3*x1, 6), Eq(x1 - 8*x2, -9), Eq(-7*x1 + 5*x2, 3), Eq(3*x2, 7)])
    special_cases.append([-4*x1 < 4, 6*x1 <= -6])
    special_cases.append([-3*x1 + 8*x2 >= -8, -10*x2 > 9, 8*x1 - 4*x2 < 8, 10*x1 - 9*x2 >= -9])
    special_cases.append([x1 + 5*x2 >= -6, 9*x1 - 3*x2 >= -9, 6*x1 + 6*x2 < -10, -3*x1 + 3*x2 < -7])
    special_cases.append([-9*x1 < 7, -5*x1 - 7*x2 < -1, 3*x1 + 7*x2 > 1, -6*x1 - 6*x2 > 9])
    special_cases.append([9*x1 - 6*x2 >= -7, 9*x1 + 4*x2 < -8, -7*x2 <= 1, 10*x2 <= -7])

    # 可行解计数初始化
    feasible_count = 0
    # 对于布尔公式中的每个文字，逐个检查其断言是否为真
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    # 断言：编码到边界的映射应该有三个条目
    assert len(lra.enc_to_boundary) == 3
    # 检查逻辑和算法是否不通过
    assert lra.check()[0] == False

    # 创建一个正的、小于-1的布尔公式
    bf = Q.positive(x) & Q.lt(x, -1)
    # 将布尔公式转换成编码的 CNF 形式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用编码 CNF 创建一个线性实数算法对象，启用测试模式
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 对于布尔公式中的每个文字，逐个检查其断言是否为真
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    # 断言：编码到边界的映射应该有两个条目
    assert len(lra.enc_to_boundary) == 2
    # 检查逻辑和算法是否不通过
    assert lra.check()[0] == False

    # 创建一个正的、零的布尔公式
    bf = Q.positive(x) & Q.zero(x)
    # 将布尔公式转换成编码的 CNF 形式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用编码 CNF 创建一个线性实数算法对象，启用测试模式
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 对于布尔公式中的每个文字，逐个检查其断言是否为真
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    # 断言：编码到边界的映射应该有两个条目
    assert len(lra.enc_to_boundary) == 2
    # 检查逻辑和算法是否不通过
    assert lra.check()[0] == False

    # 创建一个正的、零的布尔公式，但是应用到变量 y 上
    bf = Q.positive(x) & Q.zero(y)
    # 将布尔公式转换成编码的 CNF 形式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用编码 CNF 创建一个线性实数算法对象，启用测试模式
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 对于布尔公式中的每个文字，逐个检查其断言是否为真
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    # 断言：编码到边界的映射应该有两个条目
    assert len(lra.enc_to_boundary) == 2
    # 检查逻辑和算法是否通过
    assert lra.check()[0] == True
@XFAIL
# 定义一个测试函数，用于测试某些布尔公式的行为
def test_pos_neg_infinite():
    # 创建一个布尔表达式，包含 x 是正无穷小于 10000000，y 是正无穷
    bf = Q.positive_infinite(x) & Q.lt(x, 10000000) & Q.positive_infinite(y)
    # 将布尔表达式编码成 CNF，并获得编码结果
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用 LRASolver 从编码后的 CNF 创建一个线性关系算术求解器（LRA Solver），测试模式为真
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 对编码结果的每个文字进行断言，如果断言成功，则退出循环
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    # 断言编码后的边界数量为 3
    assert len(lra.enc_to_boundary) == 3
    # 断言 LRA 求解器的检查结果第一个元素为假
    assert lra.check()[0] == False

    # 创建一个布尔表达式，包含 x 是正无穷大于 10000000，y 是正无穷
    bf = Q.positive_infinite(x) & Q.gt(x, 10000000) & Q.positive_infinite(y)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    assert len(lra.enc_to_boundary) == 3
    assert lra.check()[0] == True

    # 创建一个布尔表达式，包含 x 是正无穷且 x 是负无穷
    bf = Q.positive_infinite(x) & Q.negative_infinite(x)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    for lit in enc.encoding.values():
        if lra.assert_lit(lit) is not None:
            break
    assert len(lra.enc_to_boundary) == 2
    assert lra.check()[0] == False


# 定义一个测试函数，用于测试二元关系的评估
def test_binrel_evaluation():
    # 创建一个布尔表达式，判断 3 是否大于 2
    bf = Q.gt(3, 2)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, conflicts = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 断言编码后的边界数量为 0
    assert len(lra.enc_to_boundary) == 0
    # 断言冲突集为 [[1]]
    assert conflicts == [[1]]

    # 创建一个布尔表达式，判断 3 是否小于 2
    bf = Q.lt(3, 2)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, conflicts = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    assert len(lra.enc_to_boundary) == 0
    # 断言冲突集为 [[-1]]
    assert conflicts == [[-1]]


# 定义一个测试函数，用于测试否定操作
def test_negation():
    # 断言 HANDLE_NEGATION 为真
    assert HANDLE_NEGATION is True
    # 创建一个布尔表达式，判断 x 是否大于 1 且 x 是否不大于 0
    bf = Q.gt(x, 1) & ~Q.gt(x, 0)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    for clause in enc.data:
        for lit in clause:
            lra.assert_lit(lit)
    assert len(lra.enc_to_boundary) == 2
    assert lra.check()[0] == False
    # 断言排序后的检查结果为 [[-1, 2]] 或者 [[-2, 1]]
    assert sorted(lra.check()[1]) in [[-1, 2], [-2, 1]]

    # 创建一个布尔表达式，判断 x 是否不大于 1 且 x 是否不小于 0
    bf = ~Q.gt(x, 1) & ~Q.lt(x, 0)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    for clause in enc.data:
        for lit in clause:
            lra.assert_lit(lit)
    assert len(lra.enc_to_boundary) == 2
    assert lra.check()[0] == True

    # 创建一个布尔表达式，判断 x 是否不大于 0 且 x 是否不小于 1
    bf = ~Q.gt(x, 0) & ~Q.lt(x, 1)
    enc = boolean_formula_to_encoded_cnf(bf)
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    for clause in enc.data:
        for lit in clause:
            lra.assert_lit(lit)
    assert len(lra.enc_to_boundary) == 2
    assert lra.check()[0] == False

    # 创建一个布尔表达式，判断 x+y 是否不小于 2 且 x-y 是否不小于 2 且 y 是否不小于 0
    bf = ~Q.le(x+y, 2) & ~Q.ge(x-y, 2) & ~Q.ge(y, 0)
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用 LRASolver 类的静态方法从编码后的 CNF 数据中创建对象 lra，并开启测试模式
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    
    # 遍历编码后的 CNF 数据中的每个子句
    for clause in enc.data:
        # 遍历每个子句中的文字（literal）
        for lit in clause:
            # 向 lra 对象中添加文字（literal）
            lra.assert_lit(lit)
    
    # 断言 lra 对象中 enc_to_boundary 字典的长度为 3
    assert len(lra.enc_to_boundary) == 3
    
    # 断言调用 lra 对象的 check 方法返回的第一个元素为 False
    assert lra.check()[0] == False
    
    # 断言调用 lra 对象的 check 方法返回的第二个元素（解的列表）的长度为 3
    assert len(lra.check()[1]) == 3
    
    # 断言 lra 对象的解中的所有元素都大于 0
    assert all(i > 0 for i in lra.check()[1])
def test_unhandled_input():
    # 将 S 模块中的 NaN 赋值给 nan 变量
    nan = S.NaN
    # 创建一个布尔表达式 bf，其中包含 Q.gt(3, nan) 和 Q.gt(x, nan)
    bf = Q.gt(3, nan) & Q.gt(x, nan)
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 检查 LRASolver 是否会引发 ValueError 异常
    raises(ValueError, lambda: LRASolver.from_encoded_cnf(enc, testing_mode=True))

    # 创建一个新的布尔表达式 bf，包含 Q.gt(3, I) 和 Q.gt(x, I)
    bf = Q.gt(3, I) & Q.gt(x, I)
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 检查 LRASolver 是否会引发 UnhandledInput 异常
    raises(UnhandledInput, lambda: LRASolver.from_encoded_cnf(enc, testing_mode=True))

    # 创建一个新的布尔表达式 bf，包含 Q.gt(3, float("inf")) 和 Q.gt(x, float("inf"))
    bf = Q.gt(3, float("inf")) & Q.gt(x, float("inf"))
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 检查 LRASolver 是否会引发 UnhandledInput 异常
    raises(UnhandledInput, lambda: LRASolver.from_encoded_cnf(enc, testing_mode=True))

    # 创建一个新的布尔表达式 bf，包含 Q.gt(3, oo) 和 Q.gt(x, oo)
    bf = Q.gt(3, oo) & Q.gt(x, oo)
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 检查 LRASolver 是否会引发 UnhandledInput 异常
    raises(UnhandledInput, lambda: LRASolver.from_encoded_cnf(enc, testing_mode=True))

    # 创建一个新的布尔表达式 bf，包含 Q.gt(x**2 + x, 2)
    bf = Q.gt(x**2 + x, 2)
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 检查 LRASolver 是否会引发 UnhandledInput 异常
    raises(UnhandledInput, lambda: LRASolver.from_encoded_cnf(enc, testing_mode=True))

    # 创建一个新的布尔表达式 bf，包含 Q.gt(cos(x) + x, 2)
    bf = Q.gt(cos(x) + x, 2)
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 检查 LRASolver 是否会引发 UnhandledInput 异常
    raises(UnhandledInput, lambda: LRASolver.from_encoded_cnf(enc, testing_mode=True))


@XFAIL
def test_infinite_strict_inequalities():
    # 对严格不等式与包含无穷大的约束进行详尽测试
    # 需要检查严格不等式与无穷大交互时的行为，
    # 因为使用论文中的规则处理严格不等式在允许无穷大时可能无效。
    # 使用论文中的规则可能会导致类似 oo + delta > oo 被视为 True，
    # 而实际上 oo + delta 应该等于 oo。
    # 参见 https://math.stackexchange.com/questions/4757069/can-this-method-of-converting-strict-inequalities-to-equisatisfiable-nonstrict-i
    bf = (-x - y >= -float("inf")) & (x > 0) & (y >= float("inf"))
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用 LRASolver 解析 CNF 格式，并在测试模式下进行检查
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 遍历编码中的所有文字，并断言解析的文字是否有效
    for lit in sorted(enc.encoding.values()):
        if lra.assert_lit(lit) is not None:
            break
    # 断言 lra 的 enc_to_boundary 字典长度为 3
    assert len(lra.enc_to_boundary) == 3
    # 断言调用 lra 的 check 方法返回 True
    assert lra.check()[0] == True


def test_pivot():
    # 循环进行 10 次测试
    for _ in range(10):
        # 生成一个 5x5 的随机矩阵 m
        m = randMatrix(5)
        # 对矩阵 m 进行行简化
        rref = m.rref()
        # 再次循环 5 次
        for _ in range(5):
            # 随机选择两个索引 i 和 j
            i, j = randint(0, 4), randint(0, 4)
            # 如果 m[i, j] 不为 0，则断言调用 LRASolver 的 _pivot 方法后的行简化结果与 rref 相等
            if m[i, j] != 0:
                assert LRASolver._pivot(m, i, j).rref() == rref


def test_reset_bounds():
    # 创建一个布尔表达式 bf，包含 Q.ge(x, 1) 和 Q.lt(x, 1)
    bf = Q.ge(x, 1) & Q.lt(x, 1)
    # 将布尔表达式编码成 CNF 格式
    enc = boolean_formula_to_encoded_cnf(bf)
    # 使用 LRASolver 解析 CNF 格式，并在测试模式下进行检查
    lra, _ = LRASolver.from_encoded_cnf(enc, testing_mode=True)
    # 遍历编码中的所有子句和文字，并断言所有文字都被 lra 接受
    for clause in enc.data:
        for lit in clause:
            lra.assert_lit(lit)
    # 断言 lra 的 enc_to_boundary 字典长度为 2
    assert len(lra.enc_to_boundary) == 2
    # 断言调用 lra 的 check 方法返回 False
    assert lra.check()[0] == False

    # 重置 lra 的边界
    lra.reset_bounds()
    # 断言调用 lra 的 check 方法返回 True
    assert lra.check()[0] == True
    # 遍历所有变量对象 `lra` 中的变量
    for var in lra.all_var:
        # 断言变量的上界为正无穷大，下界为负无穷大，初始赋值为零
        assert var.upper == LRARational(float("inf"), 0)
        assert var.upper_from_eq == False  # 断言变量的上界不是从等式推导得到的
        assert var.upper_from_neg == False  # 断言变量的上界不是从不等式取反推导得到的
        assert var.lower == LRARational(-float("inf"), 0)  # 断言变量的下界为负无穷大，初始赋值为零
        assert var.lower_from_eq == False  # 断言变量的下界不是从等式推导得到的
        assert var.lower_from_neg == False  # 断言变量的下界不是从不等式取反推导得到的
        assert var.assign == LRARational(0, 0)  # 断言变量的初始赋值为零
        assert var.var is not None  # 断言变量对象存在
        assert var.col_idx is not None  # 断言变量的列索引存在
# 定义一个测试函数，用于测试空 CNF（合取范式）情况
def test_empty_cnf():
    # 创建一个空的 CNF 对象
    cnf = CNF()
    # 创建一个空的 EncodedCNF 对象
    enc = EncodedCNF()
    # 将空的 CNF 转换为 EncodedCNF 对象
    enc.from_cnf(cnf)
    # 使用 EncodedCNF 对象创建 LRASolver 实例，并返回 LRA 算法的结果和冲突信息
    lra, conflict = LRASolver.from_encoded_cnf(enc)
    # 断言冲突信息长度为 0，即没有冲突
    assert len(conflict) == 0
    # 断言使用 LRA 算法检查结果为 True，并且没有任何冲突
    assert lra.check() == (True, {})
```