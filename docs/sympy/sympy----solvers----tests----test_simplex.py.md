# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_simplex.py`

```
from sympy.core.numbers import Rational
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from sympy.core.random import random, choice
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import randprime
from sympy.matrices.dense import Matrix
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.solvers.simplex import (_lp as lp, _primal_dual,
    UnboundedLPError, InfeasibleLPError, lpmin, lpmax,
    _m, _abcd, _simplex, linprog)

from sympy.external.importtools import import_module

from sympy.testing.pytest import raises

from sympy.abc import x, y, z


np = import_module("numpy")  # 导入并重命名 numpy 模块
scipy = import_module("scipy")  # 导入并重命名 scipy 模块


def test_lp():
    # 第一组线性规划问题
    r1 = y + 2*z <= 3  # 定义约束条件 r1
    r2 = -x - 3*z <= -2  # 定义约束条件 r2
    r3 = 2*x + y + 7*z <= 5  # 定义约束条件 r3
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]  # 将所有约束条件存入列表
    objective = -x - y - 5 * z  # 设置优化目标函数
    ans = optimum, argmax = lp(max, objective, constraints)  # 求解线性规划问题
    assert ans == lpmax(objective, constraints)  # 断言最优解与 lpmax 函数计算的结果相同
    assert objective.subs(argmax) == optimum  # 断言目标函数在最优解处的取值等于最优解值
    for constr in constraints:
        assert constr.subs(argmax) == True  # 断言所有约束条件在最优解处成立

    # 第二组线性规划问题
    r1 = x - y + 2*z <= 3  # 定义约束条件 r1
    r2 = -x + 2*y - 3*z <= -2  # 定义约束条件 r2
    r3 = 2*x + y - 7*z <= -5  # 定义约束条件 r3
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]  # 将所有约束条件存入列表
    objective = -x - y - 5*z  # 设置优化目标函数
    ans = optimum, argmax = lp(max, objective, constraints)  # 求解线性规划问题
    assert ans == lpmax(objective, constraints)  # 断言最优解与 lpmax 函数计算的结果相同
    assert objective.subs(argmax) == optimum  # 断言目标函数在最优解处的取值等于最优解值
    for constr in constraints:
        assert constr.subs(argmax) == True  # 断言所有约束条件在最优解处成立

    # 第三组线性规划问题
    r1 = x - y + 2*z <= -4  # 定义约束条件 r1
    r2 = -x + 2*y - 3*z <= 8  # 定义约束条件 r2
    r3 = 2*x + y - 7*z <= 10  # 定义约束条件 r3
    constraints = [r1, r2, r3, x >= 0, y >= 0, z >= 0]  # 将所有约束条件存入列表
    const = 2  # 设置常数项
    objective = -x-y-5*z+const  # 设置优化目标函数（含常数项）
    ans = optimum, argmax = lp(max, objective, constraints)  # 求解线性规划问题
    assert ans == lpmax(objective, constraints)  # 断言最优解与 lpmax 函数计算的结果相同
    assert objective.subs(argmax) == optimum  # 断言目标函数在最优解处的取值等于最优解值
    for constr in constraints:
        assert constr.subs(argmax) == True  # 断言所有约束条件在最优解处成立

    # Section 4 Problem 1 from
    # http://web.tecnico.ulisboa.pt/mcasquilho/acad/or/ftp/FergusonUCLA_LP.pdf
    # answer on page 55
    v = x1, x2, x3, x4 = symbols('x1 x2 x3 x4')  # 定义变量符号列表
    r1 = x1 - x2 - 2*x3 - x4 <= 4  # 定义约束条件 r1
    r2 = 2*x1 + x3 -4*x4 <= 2  # 定义约束条件 r2
    r3 = -2*x1 + x2 + x4 <= 1  # 定义约束条件 r3
    objective, constraints = x1 - 2*x2 - 3*x3 - x4, [r1, r2, r3] + [
        i >= 0 for i in v]  # 设置优化目标函数和所有约束条件
    ans = optimum, argmax = lp(max, objective, constraints)  # 求解线性规划问题
    assert ans == lpmax(objective, constraints)  # 断言最优解与 lpmax 函数计算的结果相同
    assert ans == (4, {x1: 7, x2: 0, x3: 0, x4: 3})  # 断言最优解符合预期结果

    # input contains Floats
    r1 = x - y + 2.0*z <= -4  # 定义约束条件 r1
    r2 = -x + 2*y - 3.0*z <= 8  # 定义约束条件 r2
    r3 = 2*x + y - 7*z <= 10  # 定义约束条件 r3
    constraints = [r1, r2, r3] + [i >= 0 for i in (x, y, z)]  # 将所有约束条件存入列表
    objective = -x-y-5*z  # 设置优化目标函数
    optimum, argmax = lp(max, objective, constraints)  # 求解线性规划问题
    assert objective.subs(argmax) == optimum  # 断言目标函数在最优解处的取值等于最优解值
    for constr in constraints:
        assert constr.subs(argmax) == True  # 断言所有约束条件在最优解处成立

    # input contains non-float or non-Rational
    r1 = x - y + sqrt(2) * z <= -4  # 定义约束条件 r1
    # 定义线性规划问题的约束条件
    r2 = -x + 2*y - 3*z <= 8
    r3 = 2*x + y - 7*z <= 10
    # 使用线性规划函数进行求解，预期会引发 TypeError 异常
    raises(TypeError, lambda: lp(max, -x-y-5*z, [r1, r2, r3]))

    # 定义线性规划问题的约束条件
    r1 = x >= 0
    # 使用线性规划函数进行求解，预期会引发 UnboundedLPError 异常
    raises(UnboundedLPError, lambda: lp(max, x, [r1]))
    r2 = x <= -1
    # 使用线性规划函数进行求解，预期会引发 InfeasibleLPError 异常
    raises(InfeasibleLPError, lambda: lp(max, x, [r1, r2]))

    # 严格不等式不被允许
    r1 = x > 0
    # 使用线性规划函数进行求解，预期会引发 TypeError 异常
    raises(TypeError, lambda: lp(max, x, [r1]))

    # 不等号约束不被允许
    r1 = Ne(x, 0)
    # 使用线性规划函数进行求解，预期会引发 TypeError 异常
    raises(TypeError, lambda: lp(max, x, [r1]))

    # 定义一个随机生成的线性规划问题
    def make_random_problem(nvar=2, num_constraints=2, sparsity=.1):
        def rand():
            if random() < sparsity:
                return sympify(0)
            int1, int2 = [randprime(0, 200) for _ in range(2)]
            return Rational(int1, int2)*choice([-1, 1])
        variables = symbols('x1:%s' % (nvar + 1))
        constraints = [(sum(rand()*x for x in variables) <= rand())
                       for _ in range(num_constraints)]
        objective = sum(rand() * x for x in variables)
        return objective, constraints, variables

    # 定义等式约束条件
    r1 = Eq(x, y)
    r2 = Eq(y, z)
    r3 = z <= 3
    constraints = [r1, r2, r3]
    objective = x
    # 解决线性规划问题，返回最优值和最优解
    ans = optimum, argmax = lp(max, objective, constraints)
    # 使用 assert 断言验证结果
    assert ans == lpmax(objective, constraints)
    assert objective.subs(argmax) == optimum
    for constr in constraints:
        assert constr.subs(argmax) == True
def test_simplex():
    # 定义线性规划问题的系数矩阵 L
    L = [
        [[1, 1], [-1, 1], [0, 1], [-1, 0]],  # 系数矩阵 A
        [5, 1, 2, -1],  # 等式右侧的常数向量 B
        [[1, 1]],  # 系数矩阵 C
        [-1]  # 等式右侧的常数 D
    ]
    # 调用函数 _m 将 L 解析为需要的格式，并计算得到 A, B, C, D
    A, B, C, D = _abcd(_m(*L), list=False)
    # 断言使用 _simplex 函数求解线性规划问题的最小值和解向量
    assert _simplex(A, B, -C, -D) == (-6, [3, 2], [1, 0, 0, 0])
    # 断言使用 _simplex 函数求解对偶问题的最小值和解向量
    assert _simplex(A, B, -C, -D, dual=True) == (-6,
        [1, 0, 0, 0], [5, 0])

    # 断言特殊情况下的线性规划问题的最小值和解向量
    assert _simplex([[]],[],[[1]],[0]) == (0, [0], [])

    # 处理等式和不等式约束条件的线性规划问题，并断言其最小值和解向量
    assert lpmax(x - y, [x <= y + 2, x >= y + 2, x >= 0, y >= 0]
        ) == (2, {x: 2, y: 0})
    assert lpmax(x - y, [x <= y + 2, Eq(x, y + 2), x >= 0, y >= 0]
        ) == (2, {x: 2, y: 0})
    assert lpmax(x - y, [x <= y + 2, Eq(x, 2)]) == (2, {x: 2, y: 0})
    assert lpmax(y, [Eq(y, 2)]) == (2, {y: 2})

    # 断言处理等式条件等效于 Eq(x, y + 2) 的线性规划问题的最小值和解向量
    assert lpmin(y, [x <= y + 2, x >= y + 2, y >= 0]
        ) == (0, {x: 2, y: 0})
    # 断言处理等式条件等效于 Eq(y, -2) 的线性规划问题的最大值和解向量
    assert lpmax(y, [0 <= y + 2, 0 >= y + 2]) == (-2, {y: -2})
    assert lpmax(y, [0 <= y + 2, 0 >= y + 2, y <= 0]
        ) == (-2, {y: -2})

    # 处理额外的符号变量，并断言相应线性规划问题的最小值和解向量
    assert lpmin(x, [y >= 1, x >= y]) == (1, {x: 1, y: 1})
    assert lpmin(x, [y >= 1, x >= y + z, x >= 0, z >= 0]
        ) == (1, {x: 1, y: 1, z: 0})

    # 检测振荡情况，使用 lpmin 函数断言引发 InfeasibleLPError 异常
    # o1
    v = x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
    raises(InfeasibleLPError, lambda: lpmin(
        9*x2 - 8*x3 + 3*x4 + 6,
        [5*x2 - 2*x3 <= 0,
        -x1 - 8*x2 + 9*x3 <= -3,
        10*x1 - x2+ 9*x4 <= -4] + [i >= 0 for i in v]))
    # o2 - 将等式输入 lpmin 函数，并转换为矩阵系统，以避免振荡并得到相同的解
    M = linear_eq_to_matrix
    f = 5*x2 + x3 + 4*x4 - x1
    L = 5*x2 + 2*x3 + 5*x4 - (x1 + 5)
    cond = [L <= 0] + [Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)]
    c, d = M(f, v)
    a, b = M(L, v)
    aeq, beq = M(cond[1:], v)
    ans = (S(9)/2, [0, S(1)/2, 0, S(1)/2])
    # 断言使用 linprog 函数求解线性规划问题的最小值和解向量
    assert linprog(c, a, b, aeq, beq, bounds=(0, 1)) == ans
    lpans = lpmin(f, cond + [x1 >= 0, x1 <= 1,
        x2 >= 0, x2 <= 1, x3 >= 0, x3 <= 1, x4 >= 0, x4 <= 1])
    # 断言处理线性规划问题的最小值和解向量
    assert (lpans[0], list(lpans[1].values())) == ans


def test_lpmin_lpmax():
    v = x1, x2, y1, y2 = symbols('x1 x2 y1 y2')
    # 定义线性规划问题的系数矩阵 L
    L = [[1, -1]], [1], [[1, 1]], [2]
    a, b, c, d = [Matrix(i) for i in L]
    m = Matrix([[a, b], [c, d]])
    # 使用函数 _primal_dual 计算得到线性规划问题的最小值和解向量
    f, constr = _primal_dual(m)[0]
    ans = lpmin(f, constr + [i >= 0 for i in v[:2]])
    # 断言计算得到的最小值和解向量
    assert ans == (-1, {x1: 1, x2: 0}),ans

    # 定义线性规划问题的系数矩阵 L
    L = [[1, -1], [1, 1]], [1, 1], [[1, 1]], [2]
    a, b, c, d = [Matrix(i) for i in L]
    m = Matrix([[a, b], [c, d]])
    # 使用函数 _primal_dual 计算得到线性规划问题的最大值和解向量
    f, constr = _primal_dual(m)[1]
    ans = lpmax(f, constr + [i >= 0 for i in v[-2:]])
    # 断言计算得到的最大值和解向量
    assert ans == (-1, {y1: 1, y2: 0})
    # 对于变量 do 在范围为 0 到 1 的循环中执行以下操作
    for do in range(2):
        # 如果 do 为假（即 do == 0），则使用 lambda 函数 M 计算线性方程组的矩阵表示
        if not do:
            M = lambda a, b: linear_eq_to_matrix(a, b)
        else:
            # 否则（即 do 不为 0），将 M 定义为 lambda 函数，将线性方程组转换为矩阵，并将结果转换为列表形式
            # 检查矩阵作为列表
            M = lambda a, b: tuple([
                i.tolist() for i in linear_eq_to_matrix(a, b)])

        # 符号变量声明：x, y, z 分别代表 x1, x2, x3
        v = x, y, z = symbols('x1:4')
        # 定义线性函数 f = x + y - 2*z
        f = x + y - 2*z
        # 计算线性方程组的系数矩阵，并取其第一个元素
        c = M(f, v)[0]
        # 定义不等式条件列表
        ineq = [7*x + 4*y - 7*z <= 3,
                3*x - y + 10*z <= 6,
                x >= 0, y >= 0, z >= 0]
        # 计算线性不等式组的矩阵表示
        ab = M([i.lts - i.gts for i in ineq], v)
        # 断言线性规划问题的最小值与预期答案相等
        ans = (-S(6)/5, [0, 0, S(3)/5])
        assert lpmin(f, ineq) == (ans[0], dict(zip(v, ans[1])))
        # 断言使用线性规划解决后的结果与预期结果相等
        assert linprog(c, *ab) == ans

        # 更新线性函数 f 为 f + 1
        f += 1
        # 计算更新后的线性方程组的系数矩阵，并取其第一个元素
        c = M(f, v)[0]
        # 定义等式条件列表
        eq = [Eq(y - 9*x, 1)]
        # 计算等式组的矩阵表示
        abeq = M([i.lhs - i.rhs for i in eq], v)
        # 更新预期答案
        ans = (1 - S(2)/5, [0, 1, S(7)/10])
        # 断言线性规划问题的最小值与预期答案相等
        assert lpmin(f, ineq + eq) == (ans[0], dict(zip(v, ans[1])))
        # 断言使用线性规划解决后的结果与预期结果相等
        assert linprog(c, *ab, *abeq) == (ans[0] - 1, ans[1])

        # 更新等式条件列表
        eq = [z - y <= S.Half]
        # 计算等式组的矩阵表示
        abeq = M([i.lhs - i.rhs for i in eq], v)
        # 更新预期答案
        ans = (1 - S(10)/9, [0, S(1)/9, S(11)/18])
        # 断言线性规划问题的最小值与预期答案相等
        assert lpmin(f, ineq + eq) == (ans[0], dict(zip(v, ans[1])))
        # 断言使用线性规划解决后的结果与预期结果相等
        assert linprog(c, *ab, *abeq) == (ans[0] - 1, ans[1])

        # 定义变量的上下界条件列表
        bounds = [(0, None), (0, None), (None, S.Half)]
        # 更新预期答案
        ans = (0, [0, 0, S.Half])
        # 断言线性规划问题的最小值与预期答案相等
        assert lpmin(f, ineq + [z <= S.Half]) == (
            ans[0], dict(zip(v, ans[1])))
        # 断言使用线性规划解决后的结果与预期结果相等
        assert linprog(c, *ab, bounds=bounds) == (ans[0] - 1, ans[1])
        # 断言使用线性规划解决后的结果与预期结果相等，采用单独指定 z 变量上下界的方式
        assert linprog(c, *ab, bounds={v.index(z): bounds[-1]}
            ) == (ans[0] - 1, ans[1])
        # 更新等式条件列表
        eq = [z - y <= S.Half]

    # 断言线性规划问题的结果与预期答案相等，特定情况下使用 linprog 的调用
    assert linprog([[1]], [], [], bounds=(2, 3)) == (2, [2])
    assert linprog([1], [], [], bounds=(2, 3)) == (2, [2])
    assert linprog([1], bounds=(2, 3)) == (2, [2])
    assert linprog([1, -1], [[1, 1]], [2], bounds={1:(None, None)}
        ) == (-2, [0, 2])
    assert linprog([1, -1], [[1, 1]], [5], bounds={1:(3, None)}
        ) == (-5, [0, 5])
```