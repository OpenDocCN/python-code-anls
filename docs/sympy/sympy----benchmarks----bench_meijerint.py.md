# `D:\src\scipysrc\sympy\sympy\benchmarks\bench_meijerint.py`

```
# 导入 sympy 库中的特定模块和函数
from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.bessel import besseli
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.integrals.transforms import (mellin_transform,
    inverse_fourier_transform, inverse_mellin_transform,
    laplace_transform, inverse_laplace_transform, fourier_transform)

# 定义一些缩写以简化代码中的函数调用
LT = laplace_transform
FT = fourier_transform
MT = mellin_transform
IFT = inverse_fourier_transform
ILT = inverse_laplace_transform
IMT = inverse_mellin_transform

# 导入 sympy.abc 中的符号 x, y
from sympy.abc import x, y

# 定义符号 nu, beta, rho
nu, beta, rho = symbols('nu beta rho')

# 定义正值的符号 apos, bpos, cpos, dpos, posk, p 和一个实数符号 k
apos, bpos, cpos, dpos, posk, p = symbols('a b c d k p', positive=True)
k = Symbol('k', real=True)
negk = Symbol('k', negative=True)

# 定义实数且非零的 mu1, mu2
mu1, mu2 = symbols('mu1 mu2', real=True, nonzero=True, finite=True)
# 定义正值且非零的 sigma1, sigma2
sigma1, sigma2 = symbols('sigma1 sigma2', real=True, nonzero=True,
                         finite=True, positive=True)
# 定义正值的 lambda 符号
rate = Symbol('lambda', positive=True)

# 定义正值的 alpha, beta 符号
alpha, beta = symbols('alpha beta', positive=True)

# 定义符合贝塔分布的概率密度函数
betadist = x**(alpha - 1)*(1 + x)**(-alpha - beta)*gamma(alpha + beta) \
    /gamma(alpha)/gamma(beta)

# 定义正整数的 k 符号
kint = Symbol('k', integer=True, positive=True)

# 定义符合卡方分布的概率密度函数
chi = 2**(1 - kint/2)*x**(kint - 1)*exp(-x**2/2)/gamma(kint/2)

# 定义符合卡方分布的概率密度函数
chisquared = 2**(-k/2)/gamma(k/2)*x**(k/2 - 1)*exp(-x/2)

# 定义符合达古分布的概率密度函数
dagum = apos*p/x*(x/bpos)**(apos*p)/(1 + x**apos/bpos**apos)**(p + 1)

# 定义正值的 d1, d2 符号
d1, d2 = symbols('d1 d2', positive=True)

# 定义 f 函数
f = sqrt(((d1*x)**d1 * d2**d2)/(d1*x + d2)**(d1 + d2))/x \
    /gamma(d1/2)/gamma(d2/2)*gamma((d1 + d2)/2)

# 定义正值的 nu, sigma 符号
nupos, sigmapos = symbols('nu sigma', positive=True)

# 定义符合 Rice 分布的概率密度函数
rice = x/sigmapos**2*exp(-(x**2 + nupos**2)/2/sigmapos**2)*besseli(0, x*
                         nupos/sigmapos**2)

# 定义实数 mu 符号
mu = Symbol('mu', real=True)

# 定义 Laplace 分布的概率密度函数
laplace = exp(-abs(x - mu)/bpos)/2/bpos

# 定义极坐标中的正值符号 u
u = Symbol('u', polar=True)

# 定义正值符号 tpos
tpos = Symbol('t', positive=True)

# 定义函数 E，计算指定表达式的期望值
def E(expr):
    # 对表达式进行指数分布、正态分布的积分计算
    integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                     (x, 0, oo), (y, -oo, oo), meijerg=True)
    integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                     (y, -oo, oo), (x, 0, oo), meijerg=True)

# 定义一个用于性能基准测试的列表 bench
bench = [
    'MT(x**nu*Heaviside(x - 1), x, s)',
    'MT(x**nu*Heaviside(1 - x), x, s)',
    'MT((1-x)**(beta - 1)*Heaviside(1-x), x, s)',
    'MT((x-1)**(beta - 1)*Heaviside(x-1), x, s)',
    'MT((1+x)**(-rho), x, s)',
    'MT(abs(1-x)**(-rho), x, s)',
    'MT((1-x)**(beta-1)*Heaviside(1-x) + a*(x-1)**(beta-1)*Heaviside(x-1), x, s)',
    'MT((x**a-b**a)/(x-b), x, s)',
    'MT((x**a-bpos**a)/(x-bpos), x, s)',
    'MT(exp(-x), x, s)',
    'MT(exp(-1/x), x, s)',
    'MT(log(x)**4*Heaviside(1-x), x, s)',
    'MT(log(x)**3*Heaviside(x-1), x, s)',
]
    'MT(log(x + 1), x, s)',
    'MT(log(1/x + 1), x, s)',
    'MT(log(abs(1 - x)), x, s)',
    'MT(log(abs(1 - 1/x)), x, s)',
    'MT(log(x)/(x+1), x, s)',
    'MT(log(x)**2/(x+1), x, s)',
    'MT(log(x)/(x+1)**2, x, s)',
    'MT(erf(sqrt(x)), x, s)',

    'MT(besselj(a, 2*sqrt(x)), x, s)',
    'MT(sin(sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(cos(sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))**2, x, s)',
    'MT(besselj(a, sqrt(x))*besselj(-a, sqrt(x)), x, s)',
    'MT(besselj(a - 1, sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*besselj(b, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))**2 + besselj(-a, sqrt(x))**2, x, s)',
    'MT(bessely(a, 2*sqrt(x)), x, s)',
    'MT(sin(sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(cos(sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*bessely(b, sqrt(x)), x, s)',
    'MT(bessely(a, sqrt(x))**2, x, s)',

    'MT(besselk(a, 2*sqrt(x)), x, s)',
    'MT(besselj(a, 2*sqrt(2*sqrt(x)))*besselk(a, 2*sqrt(2*sqrt(x))), x, s)',
    'MT(besseli(a, sqrt(x))*besselk(a, sqrt(x)), x, s)',
    'MT(besseli(b, sqrt(x))*besselk(a, sqrt(x)), x, s)',
    'MT(exp(-x/2)*besselk(a, x/2), x, s)',
    # 上面是一系列的积分变换（Mellin Transform），分别对应不同的函数形式

    # 后续：逆拉普拉斯变换、逆Mellin变换

    'LT((t-apos)**bpos*exp(-cpos*(t-apos))*Heaviside(t-apos), t, s)',
    'LT(t**apos, t, s)',
    'LT(Heaviside(t), t, s)',
    'LT(Heaviside(t - apos), t, s)',
    'LT(1 - exp(-apos*t), t, s)',
    'LT((exp(2*t)-1)*exp(-bpos - t)*Heaviside(t)/2, t, s, noconds=True)',
    'LT(exp(t), t, s)',
    'LT(exp(2*t), t, s)',
    'LT(exp(apos*t), t, s)',
    'LT(log(t/apos), t, s)',
    'LT(erf(t), t, s)',
    'LT(sin(apos*t), t, s)',
    'LT(cos(apos*t), t, s)',
    'LT(exp(-apos*t)*sin(bpos*t), t, s)',
    'LT(exp(-apos*t)*cos(bpos*t), t, s)',
    'LT(besselj(0, t), t, s, noconds=True)',
    'LT(besselj(1, t), t, s, noconds=True)',
    # 上面是一系列的拉普拉斯变换（Laplace Transform）

    'FT(Heaviside(1 - abs(2*apos*x)), x, k)',
    'FT(Heaviside(1-abs(apos*x))*(1-abs(apos*x)), x, k)',
    'FT(exp(-apos*x)*Heaviside(x), x, k)',
    'IFT(1/(apos + 2*pi*I*x), x, posk, noconds=False)',
    'IFT(1/(apos + 2*pi*I*x), x, -posk, noconds=False)',
    'IFT(1/(apos + 2*pi*I*x), x, negk)',
    'FT(x*exp(-apos*x)*Heaviside(x), x, k)',
    'FT(exp(-apos*x)*sin(bpos*x)*Heaviside(x), x, k)',
    'FT(exp(-apos*x**2), x, k)',
    'IFT(sqrt(pi/apos)*exp(-(pi*k)**2/apos), k, x)',
    'FT(exp(-apos*abs(x)), x, k)',
    # 上面是一系列的傅里叶变换（Fourier Transform）

    'integrate(normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x**2*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x**3*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 上面是一系列的积分计算，采用正则化梅耶尔函数（Meijer G-function）
    'integrate(y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 对于给定的表达式，计算 y 乘以两个正态分布的密度函数的积分，其中 x 和 y 的范围是负无穷到正无穷，使用 Meijer G 函数进行积分。
    
    'integrate(x*y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 对于给定的表达式，计算 x 乘以 y 乘以两个正态分布的密度函数的积分，其中 x 和 y 的范围是负无穷到正无穷，使用 Meijer G 函数进行积分。
    
    'integrate((x+y+1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 对于给定的表达式，计算 (x+y+1) 乘以两个正态分布的密度函数的积分，其中 x 和 y 的范围是负无穷到正无穷，使用 Meijer G 函数进行积分。
    
    'integrate((x+y-1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '                   (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 对于给定的表达式，计算 (x+y-1) 乘以两个正态分布的密度函数的积分，其中 x 和 y 的范围是负无穷到正无穷，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '                (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 对于给定的表达式，计算 x 的平方乘以两个正态分布的密度函数的积分，其中 x 和 y 的范围是负无穷到正无穷，使用 Meijer G 函数进行积分。
    
    'integrate(y**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    # 对于给定的表达式，计算 y 的平方乘以两个正态分布的密度函数的积分，其中 x 和 y 的范围是负无穷到正无穷，使用 Meijer G 函数进行积分。
    
    'integrate(exponential(x, rate), (x, 0, oo), meijerg=True)',
    # 计算指数分布（exponential distribution）在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x*exponential(x, rate), (x, 0, oo), meijerg=True)',
    # 计算 x 乘以指数分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*exponential(x, rate), (x, 0, oo), meijerg=True)',
    # 计算 x 的平方乘以指数分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'E(1)',
    # 计算常数 1 的期望值。
    
    'E(x*y)',
    # 计算变量 x 和 y 的乘积的期望值。
    
    'E(x*y**2)',
    # 计算变量 x 和 y 的平方乘积的期望值。
    
    'E((x+y+1)**2)',
    # 计算 (x+y+1) 的平方的期望值。
    
    'E(x+y+1)',
    # 计算 x+y+1 的期望值。
    
    'E((x+y-1)**2)',
    # 计算 (x+y-1) 的平方的期望值。
    
    'integrate(betadist, (x, 0, oo), meijerg=True)',
    # 计算 Beta 分布（betadist）在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x*betadist, (x, 0, oo), meijerg=True)',
    # 计算 x 乘以 Beta 分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*betadist, (x, 0, oo), meijerg=True)',
    # 计算 x 的平方乘以 Beta 分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(chi, (x, 0, oo), meijerg=True)',
    # 计算 Chi 分布（chi distribution）在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x*chi, (x, 0, oo), meijerg=True)',
    # 计算 x 乘以 Chi 分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*chi, (x, 0, oo), meijerg=True)',
    # 计算 x 的平方乘以 Chi 分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(chisquared, (x, 0, oo), meijerg=True)',
    # 计算 Chi 平方分布（chisquared distribution）在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x*chisquared, (x, 0, oo), meijerg=True)',
    # 计算 x 乘以 Chi 平方分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*chisquared, (x, 0, oo), meijerg=True)',
    # 计算 x 的平方乘以 Chi 平方分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(((x-k)/sqrt(2*k))**3*chisquared, (x, 0, oo), meijerg=True)',
    # 计算表达式 (((x-k)/sqrt(2*k))**3)*Chi 平方分布 在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(dagum, (x, 0, oo), meijerg=True)',
    # 计算 Dagum 分布（dagum distribution）在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x*dagum, (x, 0, oo), meijerg=True)',
    # 计算 x 乘以 Dagum 分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*dagum, (x, 0, oo), meijerg=True)',
    # 计算 x 的平方乘以 Dagum 分布在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(f, (x, 0, oo), meijerg=True)',
    # 计算给定函数 f 在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x*f, (x, 0, oo), meijerg=True)',
    # 计算 x 乘以给定函数 f 在 x 从 0 到正无穷的积分，使用 Meijer G 函数进行积分。
    
    'integrate(x**2*f, (x, 0, oo), meijerg=True)',
    # 计算 x 的平方乘以给定函数 f 在 x 从 0 到正无穷
    # 这段代码包含多个字符串，每个字符串代表一个数学表达式或积分操作，可能是用于数学计算的输入。
    
    "gammasimp(S('2**(2*s)*(-pi*gamma(-a + 1)*gamma(a + 1)*gamma(-a - s + 1)*gamma(-a + s - 1/2)*gamma(a - s + 3/2)*gamma(a + s + 1)/(a*(a + s)) - gamma(-a - 1/2)*gamma(-a + 1)*gamma(a + 1)*gamma(a + 3/2)*gamma(-s + 3/2)*gamma(s - 1/2)*gamma(-a + s + 1)*gamma(a - s + 1)/(a*(-a + s)))*gamma(-2*s + 1)*gamma(s + 1)/(pi*s*gamma(-a - 1/2)*gamma(a + 3/2)*gamma(-s + 1)*gamma(-s + 3/2)*gamma(s - 1/2)*gamma(-a - s + 1)*gamma(-a + s - 1/2)*gamma(a - s + 1)*gamma(a - s + 3/2))'))",
    
    # 计算函数 E1(x) 的 Mellin 变换关于变量 x 和 s
    'mellin_transform(E1(x), x, s)',
    
    # 计算 gamma(s)/s 的逆 Mellin 变换关于变量 s 和 x，区间为 (0, oo)
    'inverse_mellin_transform(gamma(s)/s, s, x, (0, oo))',
    
    # 计算 expint(a, x) 的 Mellin 变换关于变量 x 和 s
    'mellin_transform(expint(a, x), x, s)',
    
    # 计算 Si(x) 的 Mellin 变换关于变量 x 和 s
    'mellin_transform(Si(x), x, s)',
    
    # 计算 -2**s*sqrt(pi)*gamma((s + 1)/2)/(2*s*gamma(-s/2 + 1)) 的逆 Mellin 变换关于变量 s 和 x，区间为 (-1, 0)
    'inverse_mellin_transform(-2**s*sqrt(pi)*gamma((s + 1)/2)/(2*s*gamma(-s/2 + 1)), s, x, (-1, 0))',
    
    # 计算 Ci(sqrt(x)) 的 Mellin 变换关于变量 x 和 s
    'mellin_transform(Ci(sqrt(x)), x, s)',
    
    # 计算 -4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S(1)/2)) 的逆 Mellin 变换关于变量 s 和 u，区间为 (0, 1)
    'inverse_mellin_transform(-4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S(1)/2)), s, u, (0, 1))',
    
    # 计算 Ci(x) 的 Laplace 变换关于变量 x 和 s
    'laplace_transform(Ci(x), x, s)',
    
    # 计算 expint(a, x) 的 Laplace 变换关于变量 x 和 s
    'laplace_transform(expint(a, x), x, s)',
    
    # 计算 expint(1, x) 的 Laplace 变换关于变量 x 和 s
    'laplace_transform(expint(1, x), x, s)',
    
    # 计算 expint(2, x) 的 Laplace 变换关于变量 x 和 s
    'laplace_transform(expint(2, x), x, s)',
    
    # 计算 -log(1 + s**2)/2/s 的逆 Laplace 变换关于变量 s 和 u
    'inverse_laplace_transform(-log(1 + s**2)/2/s, s, u)',
    
    # 计算 log(s + 1)/s 的逆 Laplace 变换关于变量 s 和 x
    'inverse_laplace_transform(log(s + 1)/s, s, x)',
    
    # 计算 (s - log(s + 1))/s**2 的逆 Laplace 变换关于变量 s 和 x
    'inverse_laplace_transform((s - log(s + 1))/s**2, s, x)',
    
    # 计算 Chi(x) 的 Laplace 变换关于变量 x 和 s
    'laplace_transform(Chi(x), x, s)',
    
    # 计算 Shi(x) 的 Laplace 变换关于变量 x 和 s
    'laplace_transform(Shi(x), x, s)',
    
    # 计算 integrate(exp(-z*x)/x, (x, 1, oo), meijerg=True, conds="none")，使用 meijerg=True 和 conds="none" 参数
    'integrate(exp(-z*x)/x, (x, 1, oo), meijerg=True, conds="none")',
    
    # 计算 integrate(exp(-z*x)/x**2, (x, 1, oo), meijerg=True, conds="none")，使用 meijerg=True 和 conds="none" 参数
    'integrate(exp(-z*x)/x**2, (x, 1, oo), meijerg=True, conds="none")',
    
    # 计算 integrate(exp(-z*x)/x**3, (x, 1, oo), meijerg=True, conds="none")，使用 meijerg=True 和 conds="none" 参数
    'integrate(exp(-z*x)/x**3, (x, 1, oo), meijerg=True, conds="none")',
    
    # 计算 integrate(-cos(x)/x, (x, tpos, oo), meijerg=True)，使用 meijerg=True 参数
    'integrate(-cos(x)/x, (x, tpos, oo), meijerg=True)',
    
    # 计算 integrate(-sin(x)/x, (x, tpos, oo), meijerg=True)，使用 meijerg=True 参数
    'integrate(-sin(x)/x, (x, tpos, oo), meijerg=True)',
    
    # 计算 integrate(sin(x)/x, (x, 0, z), meijerg=True)，使用 meijerg=True 参数
    'integrate(sin(x)/x, (x, 0, z), meijerg=True)',
    
    # 计算 integrate(sinh(x)/x, (x, 0, z), meijerg=True)，使用 meijerg=True 参数
    'integrate(sinh(x)/x, (x, 0, z), meijerg=True)',
    
    # 计算 integrate(exp(-x)/x, x, meijerg=True)，使用 meijerg=True 参数
    'integrate(exp(-x)/x, x, meijerg=True)',
    
    # 计算 integrate(exp(-x)/x**2, x, meijerg=True)，使用 meijerg=True 参数
    'integrate(exp(-x)/x**2, x, meijerg=True)',
    
    # 计算 integrate(cos(u)/u, u, meijerg=True)，使用 meijerg=True 参数
    'integrate(cos(u)/u, u, meijerg=True)',
    
    # 计算 integrate(cosh(u)/u, u, meijerg=True)，使用 meijerg=True 参数
    'integrate(cosh(u)/u, u, meijerg=True)',
    
    # 计算 integrate(expint(1, x), x, meijerg=True)，使用 meijerg=True 参数
    'integrate(expint(1, x), x, meijerg=True)',
    
    # 计算 integrate(expint(2, x), x, meijerg=True)，使用 meijerg=True 参数
    'integrate(expint(2, x), x, meijerg=True)',
    
    # 计算 integrate(Si(x), x, meijerg=True)，使用 meijerg=True 参数
    'integrate(Si(x), x, meijerg=True)',
    
    # 计算 integrate(Ci(u), u, meijerg=True)，使用 meijerg=True 参数
    'integrate(Ci(u), u, meijerg=True)',
    
    # 计算 integrate(Shi(x), x, meijerg=True)，使用 meijerg=True 参数
    'integrate(Shi(x), x, meijerg=True)',
    
    # 计算 integrate(Chi(u), u, meijerg=True)，使用 meijerg=True 参数
    'integrate(Chi(u), u, meijerg=True)',
    
    # 计算 integrate(Si(x)*exp(-x), (x, 0, oo), meijerg=True)，使用 meijerg=True 参数
    'integrate(Si(x)*exp(-x), (x, 0, oo), meijerg=True)',
    
    # 计算 integrate(expint(1, x)*sin(x), (x, 0, oo), meijerg=True)，使用 meijerg=True 参数
    'integrate(expint(1, x)*sin(x), (x, 0, oo), meijerg=True)'
# 导入所需的模块和函数
from time import time
from sympy.core.cache import clear_cache
import sys

# 存储各个代码片段的执行时间和代码字符串的列表
timings = []

# 如果这是主程序的入口
if __name__ == '__main__':
    # 遍历 bench 列表中的每个代码片段及其索引
    for n, string in enumerate(bench):
        # 清除 sympy 的缓存
        clear_cache()
        # 记录当前时间
        _t = time()
        # 执行当前代码片段
        exec(string)
        # 计算执行所花费的时间
        _t = time() - _t
        # 将执行时间和代码字符串添加到 timings 列表中
        timings += [(_t, string)]
        # 在控制台输出一个点，表示进度
        sys.stdout.write('.')
        sys.stdout.flush()
        # 每处理 10% 的代码片段，输出当前完成的进度百分比
        if n % (len(bench) // 10) == 0:
            sys.stdout.write('%s' % (10*n // len(bench)))
    print()

    # 根据执行时间排序 timings 列表，按降序排列
    timings.sort(key=lambda x: -x[0])

    # 输出按执行时间降序排列的代码片段和其执行时间
    for ti, string in timings:
        print('%.2fs %s' % (ti, string))
```