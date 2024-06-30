# `D:\src\scipysrc\sympy\sympy\stats\tests\test_symbolic_probability.py`

```
from sympy.concrete.summations import Sum  # 导入 Sum 类，用于表示和式
from sympy.core.mul import Mul  # 导入 Mul 类，用于表示乘法
from sympy.core.numbers import (oo, pi)  # 导入 oo（无穷大）和 pi（圆周率）
from sympy.core.relational import Eq  # 导入 Eq 类，用于表示相等关系
from sympy.core.symbol import (Dummy, symbols)  # 导入 Dummy 和 symbols，用于创建符号变量
from sympy.functions.elementary.exponential import exp  # 导入 exp 函数，指数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数，平方根函数
from sympy.functions.elementary.trigonometric import sin  # 导入 sin 函数，正弦函数
from sympy.integrals.integrals import Integral  # 导入 Integral 类，用于表示积分
from sympy.core.expr import unchanged  # 导入 unchanged 类，表示不变量
from sympy.stats import (Normal, Poisson, variance, Covariance, Variance,  # 导入统计相关的函数和分布
                         Probability, Expectation, Moment, CentralMoment)
from sympy.stats.rv import probability, expectation  # 导入概率和期望相关的函数

def test_literal_probability():
    X = Normal('X', 2, 3)  # 定义正态分布随机变量 X，均值为 2，标准差为 3
    Y = Normal('Y', 3, 4)  # 定义正态分布随机变量 Y，均值为 3，标准差为 4
    Z = Poisson('Z', 4)    # 定义泊松分布随机变量 Z，参数为 4
    W = Poisson('W', 3)    # 定义泊松分布随机变量 W，参数为 3
    x = symbols('x', real=True)  # 创建实数符号变量 x
    y, w, z = symbols('y, w, z')  # 创建符号变量 y, w, z

    assert Probability(X > 0).evaluate_integral() == probability(X > 0)  # 断言：计算 X > 0 的概率的积分值等于通过 probability 函数计算的概率
    assert Probability(X > x).evaluate_integral() == probability(X > x)  # 断言：计算 X > x 的概率的积分值等于通过 probability 函数计算的概率
    assert Probability(X > 0).rewrite(Integral).doit() == probability(X > 0)  # 断言：使用积分形式重写 X > 0 的概率，并计算其值，结果应等于通过 probability 函数计算的概率
    assert Probability(X > x).rewrite(Integral).doit() == probability(X > x)  # 断言：使用积分形式重写 X > x 的概率，并计算其值，结果应等于通过 probability 函数计算的概率

    assert Expectation(X).evaluate_integral() == expectation(X)  # 断言：计算 X 的期望值的积分结果等于通过 expectation 函数计算的期望值
    assert Expectation(X).rewrite(Integral).doit() == expectation(X)  # 断言：使用积分形式重写 X 的期望值，并计算其值，结果应等于通过 expectation 函数计算的期望值
    assert Expectation(X**2).evaluate_integral() == expectation(X**2)  # 断言：计算 X^2 的期望值的积分结果等于通过 expectation 函数计算的期望值
    assert Expectation(x*X).args == (x*X,)  # 断言：验证 x*X 的期望值的参数为 (x*X,)
    assert Expectation(x*X).expand() == x*Expectation(X)  # 断言：展开 x*X 的期望值，结果应等于 x 乘以 X 的期望值
    assert Expectation(2*X + 3*Y + z*X*Y).expand() == 2*Expectation(X) + 3*Expectation(Y) + z*Expectation(X*Y)  # 断言：展开复杂表达式的期望值，结果应等于各部分的期望值之和
    assert Expectation(2*X + 3*Y + z*X*Y).args == (2*X + 3*Y + z*X*Y,)  # 断言：验证复杂表达式的期望值的参数为 (2*X + 3*Y + z*X*Y,)
    assert Expectation(sin(X)) == Expectation(sin(X)).expand()  # 断言：验证 sin(X) 的期望值等于展开后的 sin(X) 的期望值
    assert Expectation(2*x*sin(X)*Y + y*X**2 + z*X*Y).expand() == 2*x*Expectation(sin(X)*Y) + y*Expectation(X**2) + z*Expectation(X*Y)  # 断言：展开复杂表达式的期望值，结果应等于各部分的期望值之和
    assert Expectation(X + Y).expand() ==  Expectation(X) + Expectation(Y)  # 断言：展开 X + Y 的期望值，结果应等于 X 的期望值加上 Y 的期望值
    assert Expectation((X + Y)*(X - Y)).expand() == Expectation(X**2) - Expectation(Y**2)  # 断言：展开 (X + Y)*(X - Y) 的期望值，结果应等于 X^2 的期望值减去 Y^2 的期望值
    assert Expectation((X + Y)*(X - Y)).expand().doit() == -12  # 断言：计算展开后表达式的期望值，结果应为 -12
    assert Expectation(X + Y, evaluate=True).doit() == 5  # 断言：计算 X + Y 的期望值，结果应为 5
    assert Expectation(X + Expectation(Y)).doit() == 5  # 断言：计算 X + E[Y] 的期望值，结果应为 5
    assert Expectation(X + Expectation(Y)).doit(deep=False) == 2 + Expectation(Expectation(Y))  # 断言：计算 X + E[Y] 的期望值，不展开内部期望值，结果应为 2 + E[E[Y]]
    assert Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False) == 2 + Expectation(Expectation(Y + Expectation(2*X)))  # 断言：计算复杂表达式的期望值，不展开内部期望值，结果应为 2 + E[E[Y + E[2*X]]]
    assert Expectation(X + Expectation(Y + Expectation(2*X))).doit() == 9  # 断言：计算复杂表达式的期望值，结果应为 9
    assert Expectation(Expectation(2*X)).doit() == 4  # 断言：计算 E[E[2*X]] 的期望值，结果应为 4
    assert Expectation(Expectation(2*X)).doit(deep=False) == Expectation(2*X)  # 断言：计算 E[E[2*X]] 的期望值，不展开内部期望值，结果应为 E[2*X]
    assert Expectation(4*Expectation(2*X)).doit(deep=False) == 4*Expectation(2*X)  # 断言：计算 4*E[2*X] 的期望值，不展开内部期望值，结果应为 4*E[2*X]
    assert Expectation((X + Y)**3).expand() == 3*Expectation(X*Y**2) + 3*Expectation(X**2*Y) + Expectation(X**3) + Expectation(Y**3)  # 断言：展开 (X + Y)^3 的期望值，结果应等于各部分的期望值之和
    # 断言：方差期望方程的立方展开
    assert Expectation((X - Y)**3).expand() == 3*Expectation(X*Y**2) -\
                3*Expectation(X**2*Y) + Expectation(X**3) - Expectation(Y**3)
    
    # 断言：方差期望方程的平方展开
    assert Expectation((X - Y)**2).expand() == -2*Expectation(X*Y) +\
                Expectation(X**2) + Expectation(Y**2)
    
    # 断言：方差函数的参数检查
    assert Variance(w).args == (w,)
    
    # 断言：方差函数的展开结果为0
    assert Variance(w).expand() == 0
    
    # 断言：方差函数积分求值等价性
    assert Variance(X).evaluate_integral() == Variance(X).rewrite(Integral).doit() == variance(X)
    
    # 断言：方差函数对(X + z)的参数检查
    assert Variance(X + z).args == (X + z,)
    
    # 断言：方差函数对(X + z)的展开结果等于方差函数对X的展开结果
    assert Variance(X + z).expand() == Variance(X)
    
    # 断言：方差函数对(X*Y)的参数检查
    assert Variance(X*Y).args == (Mul(X, Y),)
    
    # 断言：方差函数对(X*Y)的类型为Variance
    assert type(Variance(X*Y)) == Variance
    
    # 断言：方差函数对(z*X)的展开结果
    assert Variance(z*X).expand() == z**2*Variance(X)
    
    # 断言：方差函数对(X + Y)的展开结果
    assert Variance(X + Y).expand() == Variance(X) + Variance(Y) + 2*Covariance(X, Y)
    
    # 断言：方差函数对(X + Y + Z + W)的展开结果
    assert Variance(X + Y + Z + W).expand() == (Variance(X) + Variance(Y) + Variance(Z) + Variance(W) +
                                       2 * Covariance(X, Y) + 2 * Covariance(X, Z) + 2 * Covariance(X, W) +
                                       2 * Covariance(Y, Z) + 2 * Covariance(Y, W) + 2 * Covariance(W, Z))
    
    # 断言：方差函数对(X**2)的积分求值等价性
    assert Variance(X**2).evaluate_integral() == variance(X**2)
    
    # 断言：函数unchanged的应用结果
    assert unchanged(Variance, X**2)
    
    # 断言：方差函数对(x*X**2)的展开结果
    assert Variance(x*X**2).expand() == x**2*Variance(X**2)
    
    # 断言：方差函数对(sin(X))的参数检查
    assert Variance(sin(X)).args == (sin(X),)
    
    # 断言：方差函数对(sin(X))的展开结果
    assert Variance(sin(X)).expand() == Variance(sin(X))
    
    # 断言：方差函数对(x*sin(X))的展开结果
    assert Variance(x*sin(X)).expand() == x**2*Variance(sin(X))
    
    # 断言：协方差函数对(w, z)的参数检查
    assert Covariance(w, z).args == (w, z)
    
    # 断言：协方差函数对(w, z)的展开结果
    assert Covariance(w, z).expand() == 0
    
    # 断言：协方差函数对(X, w)的展开结果
    assert Covariance(X, w).expand() == 0
    
    # 断言：协方差函数对(w, X)的展开结果
    assert Covariance(w, X).expand() == 0
    
    # 断言：协方差函数对(X, Y)的参数检查
    assert Covariance(X, Y).args == (X, Y)
    
    # 断言：协方差函数对(X, Y)的类型为Covariance
    assert type(Covariance(X, Y)) == Covariance
    
    # 断言：协方差函数对(z*X + 3, Y)的展开结果
    assert Covariance(z*X + 3, Y).expand() == z*Covariance(X, Y)
    
    # 断言：协方差函数对(X, X)的参数检查
    assert Covariance(X, X).args == (X, X)
    
    # 断言：协方差函数对(X, X)的展开结果等于方差函数对X的展开结果
    assert Covariance(X, X).expand() == Variance(X)
    
    # 断言：协方差函数对(z*X + 3, w*Y + 4)的展开结果
    assert Covariance(z*X + 3, w*Y + 4).expand() == w*z*Covariance(X,Y)
    
    # 断言：协方差函数对(X, Y)等于协方差函数对(Y, X)
    assert Covariance(X, Y) == Covariance(Y, X)
    
    # 断言：协方差函数对(X + Y, Z + W)的展开结果
    assert Covariance(X + Y, Z + W).expand() == Covariance(W, X) + Covariance(W, Y) + Covariance(X, Z) + Covariance(Y, Z)
    
    # 断言：协方差函数对(x*X + y*Y, z*Z + w*W)的展开结果
    assert Covariance(x*X + y*Y, z*Z + w*W).expand() == (x*w*Covariance(W, X) + w*y*Covariance(W, Y) +
                                                x*z*Covariance(X, Z) + y*z*Covariance(Y, Z))
    
    # 断言：协方差函数对(x*X**2 + y*sin(Y), z*Y*Z**2 + w*W)的展开结果
    assert Covariance(x*X**2 + y*sin(Y), z*Y*Z**2 + w*W).expand() == (w*x*Covariance(W, X**2) + w*y*Covariance(sin(Y), W) +
                                                        x*z*Covariance(Y*Z**2, X**2) + y*z*Covariance(Y*Z**2, sin(Y)))
    
    # 断言：协方差函数对(X, X**2)的展开结果
    assert Covariance(X, X**2).expand() == Covariance(X, X**2)
    
    # 断言：协方差函数对(X, sin(X))的展开结果
    assert Covariance(X, sin(X)).expand() == Covariance(sin(X), X)
    
    # 断言：协方差函数对(X**2, sin(X)*Y)的展开结果
    assert Covariance(X**2, sin(X)*Y).expand() == Covariance(sin(X)*Y, X**2)
    
    # 断言：协方差函数对(w, X)的积分求值结果
    assert Covariance(w, X).evaluate_integral() == 0
# 定义测试函数 `test_probability_rewrite`
def test_probability_rewrite():
    # 定义正态分布随机变量 X，均值为 2，标准差为 3
    X = Normal('X', 2, 3)
    # 定义正态分布随机变量 Y，均值为 3，标准差为 4
    Y = Normal('Y', 3, 4)
    # 定义泊松分布随机变量 Z，参数为 4
    Z = Poisson('Z', 4)
    # 定义泊松分布随机变量 W，参数为 3
    W = Poisson('W', 3)
    # 定义符号 x, y, w, z
    x, y, w, z = symbols('x, y, w, z')

    # 断言方差 Variance(w) 重写为期望 Expectation 的结果为 0
    assert Variance(w).rewrite(Expectation) == 0
    # 断言方差 Variance(X) 重写为期望 Expectation 的结果为 E[X^2] - (E[X])^2
    assert Variance(X).rewrite(Expectation) == Expectation(X ** 2) - Expectation(X) ** 2
    # 断言在给定条件 Y 下，方差 Variance(X) 重写为期望 Expectation 的结果为 E[X^2, Y] - (E[X, Y])^2
    assert Variance(X, condition=Y).rewrite(Expectation) == Expectation(X ** 2, Y) - Expectation(X, Y) ** 2
    # 断言方差 Variance(X, Y) 不等于 E[X^2] - (E[X])^2
    assert Variance(X, Y) != Expectation(X**2) - Expectation(X)**2
    # 断言方差 Variance(X + z) 重写为期望 Expectation 的结果为 E[(X + z)^2] - (E[X + z])^2
    assert Variance(X + z).rewrite(Expectation) == Expectation((X + z) ** 2) - Expectation(X + z) ** 2
    # 断言方差 Variance(X * Y) 重写为期望 Expectation 的结果为 E[X^2 * Y^2] - (E[X * Y])^2
    assert Variance(X * Y).rewrite(Expectation) == Expectation(X ** 2 * Y ** 2) - Expectation(X * Y) ** 2

    # 断言协方差 Covariance(w, X) 重写为期望 Expectation 的结果为 -w*E[X] + E[w*X]
    assert Covariance(w, X).rewrite(Expectation) == -w*Expectation(X) + Expectation(w*X)
    # 断言协方差 Covariance(X, Y) 重写为期望 Expectation 的结果为 E[X*Y] - E[X]*E[Y]
    assert Covariance(X, Y).rewrite(Expectation) == Expectation(X*Y) - Expectation(X)*Expectation(Y)
    # 断言在给定条件 W 下，协方差 Covariance(X, Y) 重写为期望 Expectation 的结果为 E[X * Y, W] - E[X, W] * E[Y, W]
    assert Covariance(X, Y, condition=W).rewrite(Expectation) == Expectation(X * Y, W) - Expectation(X, W) * Expectation(Y, W)

    # 重新定义符号 w, x, z
    w, x, z = symbols("W, x, z")
    # 定义随机变量 X = x 的概率 px
    px = Probability(Eq(X, x))
    # 定义随机变量 Z = z 的概率 pz
    pz = Probability(Eq(Z, z))

    # 断言期望 Expectation(X) 重写为概率 Probability 的结果为 积分(-oo到oo) x * px 的积分
    assert Expectation(X).rewrite(Probability) == Integral(x*px, (x, -oo, oo))
    # 断言期望 Expectation(Z) 重写为概率 Probability 的结果为 和(0到oo) z * pz 的和
    assert Expectation(Z).rewrite(Probability) == Sum(z*pz, (z, 0, oo))
    # 断言方差 Variance(X) 重写为概率 Probability 的结果为 积分(-oo到oo) x^2 * px 的积分 - (积分(-oo到oo) x * px 的积分)^2
    assert Variance(X).rewrite(Probability) == Integral(x**2*px, (x, -oo, oo)) - Integral(x*px, (x, -oo, oo))**2
    # 断言方差 Variance(Z) 重写为概率 Probability 的结果为 和(0到oo) z^2 * pz 的和 - (和(0到oo) z * pz 的和)^2
    assert Variance(Z).rewrite(Probability) == Sum(z**2*pz, (z, 0, oo)) - Sum(z*pz, (z, 0, oo))**2
    # 断言协方差 Covariance(w, X) 重写为概率 Probability 的结果为 -w*Integral(x*Probability(Eq(X, x)), (x, -oo, oo)) + Integral(w*x*Probability(Eq(X, x)), (x, -oo, oo))
    assert Covariance(w, X).rewrite(Probability) == \
           -w*Integral(x*Probability(Eq(X, x)), (x, -oo, oo)) + Integral(w*x*Probability(Eq(X, x)), (x, -oo, oo))

    # 测试将重写为和函数 Sum 的方差 Variance(X) 与重写为积分函数 Integral 的方差 Variance(X) 是否相等
    assert Variance(X).rewrite(Sum) == Variance(X).rewrite(Integral)
    # 测试将重写为和函数 Sum 的期望 Expectation(X) 与重写为积分函数 Integral 的期望 Expectation(X) 是否相等
    assert Expectation(X).rewrite(Sum) == Expectation(X).rewrite(Integral)

    # 断言协方差 Covariance(w, X) 重写为和函数 Sum 的结果为 0
    assert Covariance(w, X).rewrite(Sum) == 0

    # 断言协方差 Covariance(w, X) 重写为积分函数 Integral 的结果为 0
    assert Covariance(w, X).rewrite(Integral) == 0

    # 断言在给定条件 Y 下，方差 Variance(X) 重写为概率 Probability 的结果为 积分(-oo到oo) x^2 * Probability(Eq(X, x), Y) 的积分 - (积分(-oo到oo) x * Probability(Eq(X, x), Y) 的积分)^2
    assert Variance(X, condition=Y).rewrite(Probability) == Integral(x**2*Probability(Eq(X, x), Y), (x, -oo, oo)) - \
                                                            Integral(x*Probability(Eq(X, x), Y), (x, -oo, oo))**2


# 定义测试函数 `test_symbolic_Moment`
def test_symbolic_Moment():
    # 定义符号 mu 和 sigma
    mu = symbols('mu', real=True)
    sigma = symbols('sigma', positive=True)
    # 定义符号 x
    x = symbols('x')
    # 定义正态分布随机变量 X，均值为 mu，标准差为 sigma
    X = Normal('X', mu, sigma)
    # 定义 X 的第四个中心矩 M = E[(X - 2)^4]
    M = Moment(X, 4, 2)
    # 断言 M 重写为期望 Expectation 的结果为 Expectation((X - 2)^4)
    assert M.rewrite(Expectation) == Expectation((X - 2)**4)
    # 断言 M 重写为概率 Probability 的结果为 积分((x - 2)^4 * Probability(Eq(X, x)), (x, -oo, oo))
    assert M.rewrite(Probability) == Integral((x - 2)**4*Probability(Eq(X, x)),
                                    (x, -oo, oo))
    # 定义虚拟变量 k
    k = Dummy('k')
    # 定义积分表达式 expri
    expri = Integral(sqrt(2)*(k -
    # 定义正态分布的标准差符号 sigma，要求其为正值
    sigma = symbols('sigma', positive=True)
    # 定义符号变量 x
    x = symbols('x')
    # 创建正态分布随机变量 X，其均值为 mu，标准差为 sigma
    X = Normal('X', mu, sigma)
    # 计算 X 的中心矩，此处计算第 6 阶中心矩
    CM = CentralMoment(X, 6)
    # 使用期望值重写中心矩，期望值计算为 (X - E[X])**6
    assert CM.rewrite(Expectation) == Expectation((X - Expectation(X))**6)
    # 使用概率密度函数重写中心矩，计算积分来表示 (X - E[X])**6 * P(X=x)
    assert CM.rewrite(Probability) == Integral((x - Integral(x*Probability(True),
                        (x, -oo, oo)))**6*Probability(Eq(X, x)), (x, -oo, oo))
    # 定义一个虚拟变量 k，用于积分过程中的临时变量
    k = Dummy('k')
    # 计算期望值的积分表达式，包括正态分布的密度函数
    expri = Integral(sqrt(2)*(k - Integral(sqrt(2)*k*exp(-(k - \
            mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo)))**6*exp(-(k - \
            mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (k, -oo, oo))
    # 验证中心矩的积分重写结果
    assert CM.rewrite(Integral).dummy_eq(expri)
    # 计算并简化中心矩的值，得到结果为 15*sigma**6
    assert CM.doit().simplify() == 15*sigma**6
    # 创建矩对象，计算其值
    CM = Moment(5, 5)
    # 验证矩的计算结果为 5 的 5 次方
    assert CM.doit() == 5**5
```