# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_pde.py`

```
from sympy.core.function import (Derivative as D, Function)  # 导入 Derivative 类并重命名为 D，导入 Function 类
from sympy.core.relational import Eq  # 导入 Eq 类
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol 类和 symbols 函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入 exp 和 log 函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入 cos 和 sin 函数
from sympy.core import S  # 导入 S 对象
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
    pdsolve, classify_pde, checkpdesol)  # 导入偏微分方程求解相关的函数
from sympy.testing.pytest import raises  # 导入 raises 函数用于测试

a, b, c, x, y = symbols('a b c x y')  # 定义符号变量 a, b, c, x, y

def test_pde_separate_add():
    x, y, z, t = symbols("x,y,z,t")  # 定义符号变量 x, y, z, t
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')  # 映射为函数对象

    eq = Eq(D(u(x, t), x), D(u(x, t), t)*exp(u(x, t)))  # 定义偏微分方程
    res = pde_separate_add(eq, u(x, t), [X(x), T(t)])  # 对偏微分方程进行分离操作
    assert res == [D(X(x), x)*exp(-X(x)), D(T(t), t)*exp(T(t))]  # 断言分离结果正确


def test_pde_separate():
    x, y, z, t = symbols("x,y,z,t")  # 定义符号变量 x, y, z, t
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')  # 映射为函数对象

    eq = Eq(D(u(x, t), x), D(u(x, t), t)*exp(u(x, t)))  # 定义偏微分方程
    raises(ValueError, lambda: pde_separate(eq, u(x, t), [X(x), T(t)], 'div'))  # 引发异常检查


def test_pde_separate_mul():
    x, y, z, t = symbols("x,y,z,t")  # 定义符号变量 x, y, z, t
    c = Symbol("C", real=True)  # 定义实数符号变量 C
    Phi = Function('Phi')  # 定义函数对象 Phi
    F, R, T, X, Y, Z, u = map(Function, 'FRTXYZu')  # 映射为函数对象

    # Something simple :)
    eq = Eq(D(F(x, y, z), x) + D(F(x, y, z), y) + D(F(x, y, z), z), 0)  # 定义偏微分方程

    # Duplicate arguments in functions
    raises(
        ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), u(z, z)]))  # 引发异常检查：函数参数重复
    # Wrong number of arguments
    raises(ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), Y(y)]))  # 引发异常检查：参数数量错误
    # Wrong variables: [x, y] -> [x, z]
    raises(
        ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(t), Y(x, y)]))  # 引发异常检查：变量错误

    assert pde_separate_mul(eq, F(x, y, z), [Y(y), u(x, z)]) == \
        [D(Y(y), y)/Y(y), -D(u(x, z), x)/u(x, z) - D(u(x, z), z)/u(x, z)]  # 断言分离结果正确
    assert pde_separate_mul(eq, F(x, y, z), [X(x), Y(y), Z(z)]) == \
        [D(X(x), x)/X(x), -D(Z(z), z)/Z(z) - D(Y(y), y)/Y(y)]  # 断言分离结果正确

    # wave equation
    wave = Eq(D(u(x, t), t, t), c**2*D(u(x, t), x, x))  # 定义波动方程
    res = pde_separate_mul(wave, u(x, t), [X(x), T(t)])  # 对波动方程进行分离操作
    assert res == [D(X(x), x, x)/X(x), D(T(t), t, t)/(c**2*T(t))]  # 断言分离结果正确

    # Laplace equation in cylindrical coords
    eq = Eq(1/r * D(Phi(r, theta, z), r) + D(Phi(r, theta, z), r, 2) +
            1/r**2 * D(Phi(r, theta, z), theta, 2) + D(Phi(r, theta, z), z, 2), 0)  # 定义在柱坐标中的拉普拉斯方程
    # Separate z
    res = pde_separate_mul(eq, Phi(r, theta, z), [Z(z), u(theta, r)])  # 对方程进行分离操作
    assert res == [D(Z(z), z, z)/Z(z),
            -D(u(theta, r), r, r)/u(theta, r) -
        D(u(theta, r), r)/(r*u(theta, r)) -
        D(u(theta, r), theta, theta)/(r**2*u(theta, r))]  # 断言分离结果正确
    # Lets use the result to create a new equation...
    eq = Eq(res[1], c)
    # ...and separate theta...
    res = pde_separate_mul(eq, u(theta, r), [T(theta), R(r)])  # 对方程进行分离操作
    assert res == [D(T(theta), theta, theta)/T(theta),
            -r*D(R(r), r)/R(r) - r**2*D(R(r), r, r)/R(r) - c*r**2]  # 断言分离结果正确
    # ...or r...
    # 调用函数 pde_separate_mul 处理方程 eq，并传入函数 u(theta, r) 和参数列表 [R(r), T(theta)]
    res = pde_separate_mul(eq, u(theta, r), [R(r), T(theta)])
    
    # 使用断言检查变量 res 是否等于指定的列表值
    assert res == [
        # 第一个元素：r*D(R(r), r)/R(r) + r**2*D(R(r), r, r)/R(r) + c*r**2
        r * D(R(r), r) / R(r) + r**2 * D(R(r), r, r) / R(r) + c * r**2,
        # 第二个元素：-D(T(theta), theta, theta)/T(theta)
        -D(T(theta), theta, theta) / T(theta)
    ]
# 定义测试函数 test_issue_11726，用于测试某个问题编号的相关功能
def test_issue_11726():
    # 创建符号变量 x 和 t
    x, t = symbols("x t")
    # 创建一个函数符号 f
    f = symbols("f", cls=Function)
    # 创建函数符号 X 和 T
    X, T = symbols("X T", cls=Function)

    # 定义 u 为 f(x, t)
    u = f(x, t)
    # 构造偏微分方程 u_xx - u_tt
    eq = u.diff(x, 2) - u.diff(t, 2)
    # 调用 pde_separate 函数，将 eq 用 [T(x), X(t)] 分离
    res = pde_separate(eq, u, [T(x), X(t)])
    # 断言 res 应该等于 [D(T(x), x, x)/T(x), D(X(t), t, t)/X(t)]
    assert res == [D(T(x), x, x)/T(x), D(X(t), t, t)/X(t)]


# 定义测试函数 test_pde_classify，用于测试偏微分方程分类功能
def test_pde_classify():
    # 创建函数符号 f
    f = Function('f')
    # 定义多个偏微分方程 eq1 到 eq6
    eq1 = a*f(x,y) + b*f(x,y).diff(x) + c*f(x,y).diff(y)
    eq2 = 3*f(x,y) + 2*f(x,y).diff(x) + f(x,y).diff(y)
    eq3 = a*f(x,y) + b*f(x,y).diff(x) + 2*f(x,y).diff(y)
    eq4 = x*f(x,y) + f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq5 = x**2*f(x,y) + x*f(x,y).diff(x) + x*y*f(x,y).diff(y)
    eq6 = y*x**2*f(x,y) + y*f(x,y).diff(x) + f(x,y).diff(y)
    
    # 对于 eq1, eq2, eq3，断言其分类结果为 ('1st_linear_constant_coeff_homogeneous',)
    for eq in [eq1, eq2, eq3]:
        assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    
    # 对于 eq4, eq5, eq6，断言其分类结果为 ('1st_linear_variable_coeff',)
    for eq in [eq4, eq5, eq6]:
        assert classify_pde(eq) == ('1st_linear_variable_coeff',)


# 定义测试函数 test_checkpdesol，用于测试偏微分方程解的检验功能
def test_checkpdesol():
    # 创建函数符号 f 和 F
    f, F = map(Function, ['f', 'F'])
    # 定义多个偏微分方程 eq1 到 eq6
    eq1 = a*f(x,y) + b*f(x,y).diff(x) + c*f(x,y).diff(y)
    eq2 = 3*f(x,y) + 2*f(x,y).diff(x) + f(x,y).diff(y)
    eq3 = a*f(x,y) + b*f(x,y).diff(x) + 2*f(x,y).diff(y)
    
    # 对于 eq1, eq2, eq3，断言其通过 pdsolve 求解后的结果可以通过 checkpdesol 验证
    for eq in [eq1, eq2, eq3]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    
    # 定义更多的偏微分方程 eq4 到 eq6
    eq4 = x*f(x,y) + f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq5 = 2*f(x,y) + 1*f(x,y).diff(x) + 3*f(x,y).diff(y)
    eq6 = f(x,y) + 1*f(x,y).diff(x) + 3*f(x,y).diff(y)
    
    # 对于 eq4, eq5, eq6，断言其通过 pdsolve 求解后的结果可以通过 checkpdesol 验证
    assert checkpdesol(eq4, [pdsolve(eq5), pdsolve(eq6)]) == [
        (False, (x - 2)*F(3*x - y)*exp(-x/S(5) - 3*y/S(5))),
        (False, (x - 1)*F(3*x - y)*exp(-x/S(10) - 3*y/S(10)))]
    for eq in [eq4, eq5, eq6]:
        assert checkpdesol(eq, pdsolve(eq))[0]
    
    # 对于 eq4，使用 sol = pdsolve(eq4) 求解结果，然后断言当传入的 sol4 不支持函数解析时会引发 NotImplementedError
    sol = pdsolve(eq4)
    sol4 = Eq(sol.lhs - sol.rhs, 0)
    raises(NotImplementedError, lambda:
        checkpdesol(eq4, sol4, solve_for_func=False))


# 定义测试函数 test_solvefun，用于测试偏微分方程求解函数的指定功能
def test_solvefun():
    # 创建函数符号 f, F, G, H
    f, F, G, H = map(Function, ['f', 'F', 'G', 'H'])
    # 定义偏微分方程 eq1
    eq1 = f(x,y) + f(x,y).diff(x) + f(x,y).diff(y)
    
    # 断言使用默认求解函数 pdsolve(eq1) 的结果
    assert pdsolve(eq1) == Eq(f(x, y), F(x - y)*exp(-x/2 - y/2))
    
    # 断言使用 solvefun=G 指定求解函数的结果
    assert pdsolve(eq1, solvefun=G) == Eq(f(x, y), G(x - y)*exp(-x/2 - y/2))
    
    # 断言使用 solvefun=H 指定求解函数的结果
    assert pdsolve(eq1, solvefun=H) == Eq(f(x, y), H(x - y)*exp(-x/2 - y/2))


# 定义测试函数 test_pde_1st_linear_constant_coeff_homogeneous，用于测试一阶线性恒定系数齐次偏微分方程
def test_pde_1st_linear_constant_coeff_homogeneous():
    # 创建函数符号 f 和 F
    f, F = map(Function, ['f', 'F'])
    # 定义函数 u = f(x, y)
    u = f(x, y)
    
    # 定义偏微分方程 eq1
    eq1 = 2*u + u.diff(x) + u.diff(y)
    # 断言对 eq1 的分类结果为 ('1st_linear_constant_coeff_homogeneous',)
    assert classify_pde(eq1) == ('1st_linear_constant_coeff_homogeneous',)
    # 使用 pdsolve 求解 eq1，断言解为指定形式
    sol = pdsolve(eq1)
    assert sol == Eq(u, F(x - y)*exp(-x - y))
    # 断言解通过 checkpdesol 函数验证
    assert checkpdesol(eq1, sol)[0]

    # 定义偏微分方程 eq2
    eq2 = 4 + (3*u.diff(x)/u) + (2*u.diff(y)/u)
    # 断言对 eq2 的分类结果为 ('1st_linear_constant_coeff_homogeneous',)
    assert classify_pde(eq2) == ('1st_linear_constant_coeff_homogeneous',)
    # 使用 pdsolve 求解 eq2，断言解为指定形式
    sol = pdsolve(eq2)
    assert sol == Eq(u, F(2*x - 3*y)*exp(-S(12)*x/13 - S(8)*y/13))
    # 断言解通过 checkpdesol 函数验证
    assert checkpdesol(eq2, sol)[0]

    # 定义偏微分方程 eq3
    eq3 = u + (6*u.diff(x)) + (7*u.diff(y))
    # 断言对 eq3 的分类结果为 ('1st_linear_constant_coeff_homogeneous',)
    assert classify_pde(eq3) == ('1st_linear_constant_coeff_homogeneous',)
    # 使用 pdsolve 求解 eq3，断言解为指定形式
    sol = pdsolve(eq3)
    assert sol == Eq(u, F(7*x - 6*y)*exp(-6*x/S(85) - 7*y/S(85)))
    # 断言检查偏微分方程是否满足条件，返回一个布尔值
    assert checkpdesol(eq, sol)[0]
    
    # 定义偏微分方程，包含未知函数 u 在空间中的一阶偏导数
    eq = a*u + b*u.diff(x) + c*u.diff(y)
    
    # 求解偏微分方程 eq，得到解 sol
    sol = pdsolve(eq)
    
    # 断言检查偏微分方程的解是否满足条件，返回一个布尔值
    assert checkpdesol(eq, sol)[0]
# 定义用于测试一阶线性常系数偏微分方程的函数
def test_pde_1st_linear_constant_coeff():
    # 创建函数对象 f(x, y) 和 F(x, y)
    f, F = map(Function, ['f', 'F'])
    # 定义 u = f(x, y)
    u = f(x, y)
    # 第一个偏微分方程
    eq = -2*u.diff(x) + 4*u.diff(y) + 5*u - exp(x + 3*y)
    # 求解偏微分方程
    sol = pdsolve(eq)
    # 断言解 sol 符合预期的表达式
    assert sol == Eq(f(x,y), (F(4*x + 2*y)*exp(x/2) + exp(x + 4*y)/15)*exp(-y))
    # 断言分类函数 classify_pde 返回正确的结果
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    # 断言解 sol 符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第二个偏微分方程
    eq = (u.diff(x)/u) + (u.diff(y)/u) + 1 - (exp(x + y)/u)
    # 求解偏微分方程
    sol = pdsolve(eq)
    # 断言解 sol 符合预期的表达式
    assert sol == Eq(f(x, y), F(x - y)*exp(-x/2 - y/2) + exp(x + y)/3)
    # 断言分类函数 classify_pde 返回正确的结果
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    # 断言解 sol 符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第三个偏微分方程
    eq = 2*u + -u.diff(x) + 3*u.diff(y) + sin(x)
    # 求解偏微分方程
    sol = pdsolve(eq)
    # 断言解 sol 符合预期的表达式
    assert sol == Eq(f(x, y), F(3*x + y)*exp(x/5 - 3*y/5) - 2*sin(x)/5 - cos(x)/5)
    # 断言分类函数 classify_pde 返回正确的结果
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    # 断言解 sol 符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第四个偏微分方程
    eq = u + u.diff(x) + u.diff(y) + x*y
    # 求解偏微分方程
    sol = pdsolve(eq)
    # 断言解 sol 扩展后符合预期的表达式
    assert sol.expand() == Eq(f(x, y), x + y + (x - y)**2/4 - (x + y)**2/4 + F(x - y)*exp(-x/2 - y/2) - 2).expand()
    # 断言分类函数 classify_pde 返回正确的结果
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    # 断言解 sol 符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第五个偏微分方程
    eq = u + u.diff(x) + u.diff(y) + log(x)
    # 断言分类函数 classify_pde 返回正确的结果
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')


# 定义测试所有偏微分方程求解器函数
def test_pdsolve_all():
    # 创建函数对象 f(x, y) 和 F(x, y)
    f, F = map(Function, ['f', 'F'])
    # 定义 u = f(x, y)
    u = f(x,y)
    # 定义一个新的偏微分方程
    eq = u + u.diff(x) + u.diff(y) + x**2*y
    # 求解偏微分方程，使用 'all' 提示
    sol = pdsolve(eq, hint='all')
    # 断言解包含特定的键
    keys = ['1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral', 'default', 'order']
    assert sorted(sol.keys()) == keys
    # 断言 order 键的值为 1
    assert sol['order'] == 1
    # 断言 default 键的值为 '1st_linear_constant_coeff'
    assert sol['default'] == '1st_linear_constant_coeff'
    # 断言 '1st_linear_constant_coeff' 键的解符合预期的表达式
    assert sol['1st_linear_constant_coeff'].expand() == Eq(f(x, y), -x**2*y + x**2 + 2*x*y - 4*x - 2*y + F(x - y)*exp(-x/2 - y/2) + 6).expand()


# 定义测试一阶线性可变系数偏微分方程的函数
def test_pdsolve_variable_coeff():
    # 创建函数对象 f(x, y) 和 F(x, y)
    f, F = map(Function, ['f', 'F'])
    # 定义 u = f(x, y)
    u = f(x, y)
    # 第一个变系数偏微分方程
    eq = x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    # 求解偏微分方程，使用 '1st_linear_variable_coeff' 提示
    sol = pdsolve(eq, hint="1st_linear_variable_coeff")
    # 断言解符合预期的表达式
    assert sol == Eq(u, F(x*y)*exp(y**2/2) + 1)
    # 断言解符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第二个变系数偏微分方程
    eq = x**2*u + x*u.diff(x) + x*y*u.diff(y)
    # 求解偏微分方程，使用 '1st_linear_variable_coeff' 提示
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    # 断言解符合预期的表达式
    assert sol == Eq(u, F(y*exp(-x))*exp(-x**2/2))
    # 断言解符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第三个变系数偏微分方程
    eq = y*x**2*u + y*u.diff(x) + u.diff(y)
    # 求解偏微分方程，使用 '1st_linear_variable_coeff' 提示
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    # 断言解符合预期的表达式
    assert sol == Eq(u, F(-2*x + y**2)*exp(-x**3/3))
    # 断言解符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第四个变系数偏微分方程
    eq = exp(x)**2*(u.diff(x)) + y
    # 求解偏微分方程，使用 '1st_linear_variable_coeff' 提示
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    # 断言解符合预期的表达式
    assert sol == Eq(u, y*exp(-2*x)/2 + F(y))
    # 断言解符合偏微分方程的检验条件
    assert checkpdesol(eq, sol)[0]

    # 第五个变系数偏微分方程
    eq = exp(2*x)*(u.diff(y)) + y*u - u
    # 求解偏微分方程，使用 '1st_linear_variable_coeff' 提示
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    # 断言解符合预期的表达式
    assert sol == Eq(u, F(x)*exp(-y*(y - 2)*exp(-2*x)/2))
```