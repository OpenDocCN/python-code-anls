# `D:\src\scipysrc\sympy\sympy\printing\tests\test_latex.py`

```
# 从 sympy 库导入各种符号和函数

from sympy import MatAdd, MatMul, Array  # 导入矩阵加法、矩阵乘法、数组
from sympy.algebras.quaternion import Quaternion  # 导入四元数
from sympy.calculus.accumulationbounds import AccumBounds  # 导入积累边界
from sympy.combinatorics.permutations import Cycle, Permutation, AppliedPermutation  # 导入循环、置换和应用置换
from sympy.concrete.products import Product  # 导入乘积
from sympy.concrete.summations import Sum  # 导入求和
from sympy.core.containers import Tuple, Dict  # 导入元组和字典
from sympy.core.expr import UnevaluatedExpr  # 导入未评估表达式
from sympy.core.function import Derivative, Function, Lambda, Subs, diff  # 导入导数、函数、Lambda函数、Subs、diff
from sympy.core.mod import Mod  # 导入模运算
from sympy.core.mul import Mul  # 导入乘法
from sympy.core.numbers import (AlgebraicNumber, Float, I, Integer, Rational, oo, pi)  # 导入各种数值类型
from sympy.core.parameters import evaluate  # 导入参数评估
from sympy.core.power import Pow  # 导入幂运算
from sympy.core.relational import Eq, Ne  # 导入相等和不相等关系
from sympy.core.singleton import S  # 导入单例对象
from sympy.core.symbol import Symbol, Wild, symbols  # 导入符号、通配符和符号集合
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial,  # 导入下降阶乘、上升阶乘
                                                      binomial, factorial, factorial2, subfactorial)  # 导入二项式系数、阶乘、双阶乘、子阶乘
from sympy.functions.combinatorial.numbers import (bernoulli, bell, catalan, euler, genocchi,  # 导入伯努利数、贝尔数、卡特兰数、欧拉数、
                                                   lucas, fibonacci, tribonacci, divisor_sigma, udivisor_sigma,  # 阿几里德除数函数、udivisor_sigma函数、
                                                   mobius, primenu, primeomega,  # 莫比乌斯函数、primenu函数、primeomega函数、
                                                   totient, reduced_totient)  # 导入欧拉函数、约化欧拉函数
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, polar_lift, re)  # 导入复数函数
from sympy.functions.elementary.exponential import (LambertW, exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import (asinh, coth)  # 导入反双曲正弦和反双曲余切函数
from sympy.functions.elementary.integers import (ceiling, floor, frac)  # 导入向上取整、向下取整、取小数部分函数
from sympy.functions.elementary.miscellaneous import (Max, Min, root, sqrt)  # 导入最大值、最小值、根号、平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import (acsc, asin, cos, cot, sin, tan)  # 导入三角函数
from sympy.functions.special.beta_functions import beta  # 导入贝塔函数
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)  # 导入德尔塔函数和海维赛德函数
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f, elliptic_k, elliptic_pi)  # 导入椭圆积分函数
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)  # 导入特殊误差函数
from sympy.functions.special.gamma_functions import (gamma, uppergamma)  # 导入伽玛函数和上伽玛函数
from sympy.functions.special.hyper import (hyper, meijerg)  # 导入超几何函数和Meijer G 函数
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)  # 导入马修函数
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu,  # 导入相关拉盖尔函数、相关勒让德函数、
                                                 gegenbauer, hermite, jacobi, laguerre, legendre)  # 切比雪夫函数、切比雪夫多项式、
from sympy.functions.special.singularity_functions import SingularityFunction  # 导入奇异性函数
from sympy.functions.special.spherical_harmonics import (Ynm, Znm)  # 导入球谐函数
from sympy.functions.special.tensor_functions import (KroneckerDelta, LeviCivita)  # 导入克罗内克三角函数和莱维-奇维塔符号
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, stieltjes, zeta)  # 导入Zeta函数系列
from sympy.integrals.integrals import Integral  # 导入积分函数
# 导入 SymPy 库中的不同变换和逻辑运算模块
from sympy.integrals.transforms import (CosineTransform, FourierTransform, InverseCosineTransform, InverseFourierTransform, InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform, LaplaceTransform, MellinTransform, SineTransform)
from sympy.logic import Implies  # 导入逻辑模块中的 Implies 函数
from sympy.logic.boolalg import (And, Or, Xor, Equivalent, false, Not, true)  # 导入逻辑布尔代数模块的函数和常量
from sympy.matrices.dense import Matrix  # 导入密集矩阵模块的 Matrix 类
from sympy.matrices.expressions.kronecker import KroneckerProduct  # 导入克罗内克积表达式模块的 KroneckerProduct 类
from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵表达式模块的 MatrixSymbol 类
from sympy.matrices.expressions.permutation import PermutationMatrix  # 导入排列矩阵表达式模块的 PermutationMatrix 类
from sympy.matrices.expressions.slice import MatrixSlice  # 导入切片矩阵表达式模块的 MatrixSlice 类
from sympy.physics.control.lti import (TransferFunction, Series, Parallel, Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback)  # 导入控制系统理论模块中的类
from sympy.physics.quantum import Commutator, Operator  # 导入量子物理模块中的 Commutator 和 Operator 类
from sympy.physics.quantum.trace import Tr  # 导入量子物理模块中的 Tr 函数
from sympy.physics.units import (meter, gibibyte, gram, microgram, second, milli, micro)  # 导入物理单位模块中的单位对象
from sympy.polys.domains.integerring import ZZ  # 导入多项式整数环模块中的 ZZ 对象
from sympy.polys.fields import field  # 导入多项式域模块的 field 函数
from sympy.polys.polytools import Poly  # 导入多项式工具模块的 Poly 类
from sympy.polys.rings import ring  # 导入多项式环模块的 ring 函数
from sympy.polys.rootoftools import (RootSum, rootof)  # 导入多项式根工具模块的 RootSum 和 rootof 函数
from sympy.series.formal import fps  # 导入级数形式模块的 fps 函数
from sympy.series.fourier import fourier_series  # 导入傅里叶级数模块的 fourier_series 函数
from sympy.series.limits import Limit  # 导入极限级数模块的 Limit 类
from sympy.series.order import Order  # 导入级数阶数模块的 Order 类
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)  # 导入序列模块中的类和函数
from sympy.sets.conditionset import ConditionSet  # 导入条件集模块中的 ConditionSet 类
from sympy.sets.contains import Contains  # 导入包含关系模块中的 Contains 类
from sympy.sets.fancysets import (ComplexRegion, ImageSet, Range)  # 导入特殊集合模块中的类
from sympy.sets.ordinals import (Ordinal, OrdinalOmega, OmegaPower)  # 导入序数集合模块中的类
from sympy.sets.powerset import PowerSet  # 导入幂集模块中的 PowerSet 类
from sympy.sets.sets import (FiniteSet, Interval, Union, Intersection, Complement, SymmetricDifference, ProductSet)  # 导入集合运算模块中的类
from sympy.sets.setexpr import SetExpr  # 导入集合表达式模块的 SetExpr 类
from sympy.stats.crv_types import Normal  # 导入统计分布模块中的 Normal 类
from sympy.stats.symbolic_probability import (Covariance, Expectation, Probability, Variance)  # 导入符号概率模块中的类
from sympy.tensor.array import (ImmutableDenseNDimArray, ImmutableSparseNDimArray, MutableSparseNDimArray, MutableDenseNDimArray, tensorproduct)  # 导入张量数组模块中的类和函数
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement  # 导入数组表达式模块中的类
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)  # 导入索引张量模块中的类
from sympy.tensor.toperators import PartialDerivative  # 导入张量运算符模块中的 PartialDerivative 类
from sympy.vector import (CoordSys3D, Cross, Curl, Dot, Divergence, Gradient, Laplacian)  # 导入向量分析模块中的类和函数
from sympy.testing.pytest import (XFAIL, raises, _both_exp_pow, warns_deprecated_sympy)  # 导入 SymPy 测试框架模块中的函数和常量
from sympy.printing.latex import (latex, translate, greek_letters_set, tex_greek_dictionary, multiline_latex, latex_escape, LatexPrinter)  # 导入 LaTeX 打印模块中的函数和常量
import sympy as sym  # 导入 SymPy 库并命名为 sym

from sympy.abc import mu, tau  # 从 sympy.abc 模块导入 mu 和 tau 符号

# 定义一个自定义的 lowergamma 类，继承自 sym.lowergamma
class lowergamma(sym.lowergamma):
    pass
    pass   # 用于占位，表示当前位置暂无实际代码逻辑，保留关键字 pass 的作用是为了在语法上保持完整性
# 定义符号变量 x, y, z, t, w, a, b, c, s, p
x, y, z, t, w, a, b, c, s, p = symbols('x y z t w a b c s p')

# 定义整数符号变量 k, m, n
k, m, n = symbols('k m n', integer=True)

# 定义测试函数 test_printmethod
def test_printmethod():
    # 定义类 R，继承自 Abs
    class R(Abs):
        # 定义 _latex 方法，用于打印 Latex 格式
        def _latex(self, printer):
            # 返回格式化的 Latex 字符串，包含参数 self.args[0] 的打印结果
            return "foo(%s)" % printer._print(self.args[0])
    
    # 断言：调用 latex 函数，将 R(x) 转换为 Latex 格式，应为 "foo(x)"
    assert latex(R(x)) == r"foo(x)"
    
    # 重新定义类 R，继承自 Abs
    class R(Abs):
        # 定义 _latex 方法，用于打印 Latex 格式
        def _latex(self, printer):
            # 返回固定的 Latex 字符串 "foo"
            return "foo"
    
    # 断言：调用 latex 函数，将 R(x) 转换为 Latex 格式，应为 "foo"
    assert latex(R(x)) == r"foo"

# 定义测试函数 test_latex_basic
def test_latex_basic():
    # 断言：将表达式 1 + x 转换为 Latex 格式，应为 "x + 1"
    assert latex(1 + x) == r"x + 1"
    
    # 断言：将表达式 x**2 转换为 Latex 格式，应为 "x^{2}"
    assert latex(x**2) == r"x^{2}"
    
    # 断言：将表达式 x**(1 + x) 转换为 Latex 格式，应为 "x^{x + 1}"
    assert latex(x**(1 + x)) == r"x^{x + 1}"
    
    # 断言：将表达式 x**3 + x + 1 + x**2 转换为 Latex 格式，应为 "x^{3} + x^{2} + x + 1"
    assert latex(x**3 + x + 1 + x**2) == r"x^{3} + x^{2} + x + 1"

    # 断言：将表达式 2*x*y 转换为 Latex 格式，应为 "2 x y"
    assert latex(2*x*y) == r"2 x y"
    
    # 断言：将表达式 2*x*y 转换为 Latex 格式，使用乘号符号 'dot'，应为 "2 \cdot x \cdot y"
    assert latex(2*x*y, mul_symbol='dot') == r"2 \cdot x \cdot y"
    
    # 断言：将表达式 3*x**2*y 转换为 Latex 格式，使用乘号符号 '\,'，应为 "3\,x^{2}\,y"
    assert latex(3*x**2*y, mul_symbol='\\,') == r"3\,x^{2}\,y"
    
    # 断言：将表达式 1.5*3**x 转换为 Latex 格式，使用乘号符号 '\,'，应为 "1.5 \cdot 3^{x}"
    assert latex(1.5*3**x, mul_symbol='\\,') == r"1.5 \cdot 3^{x}"

    # 断言：将表达式 x**S.Half**5 转换为 Latex 格式，应为 "\sqrt[32]{x}"
    assert latex(x**S.Half**5) == r"\sqrt[32]{x}"
    
    # 断言：将 Mul 对象转换为 Latex 格式，表达式为 \frac{1}{2} x^{2} \left(-5\right)
    assert latex(Mul(S.Half, x**2, -5, evaluate=False)) == r"\frac{1}{2} x^{2} \left(-5\right)"
    
    # 断言：将 Mul 对象转换为 Latex 格式，表达式为 \frac{1}{2} x^{2} \cdot 5
    assert latex(Mul(S.Half, x**2, 5, evaluate=False)) == r"\frac{1}{2} x^{2} \cdot 5"
    
    # 省略多个 Mul 对象的注释，按照同样的方式添加 Latex 格式的断言

    # 断言：将 Rational(2, 3) * Rational(5, 7) 转换为 Latex 格式，应为 "\frac{2}{3} \cdot \frac{5}{7}"
    assert latex(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == r"\frac{2}{3} \cdot \frac{5}{7}"

    # 断言：将表达式 1/x 转换为 Latex 格式，应为 "\frac{1}{x}"
    assert latex(1/x) == r"\frac{1}{x}"
    
    # 断言：将表达式 1/x 转换为 Latex 格式，启用 fold_short_frac 参数，应为 "1 / x"
    assert latex(1/x, fold_short_frac=True) == r"1 / x"
    
    # 省略其余 Mul 对象的注释，按照同样的方式添加 Latex 格式的断言
    assert latex(1/x**2) == r"\frac{1}{x^{2}}"
    # 测试 LaTeX 表示：分数 1/x^2 应转换为字符串 "\frac{1}{x^{2}}"

    assert latex(1/(x + y)/2) == r"\frac{1}{2 \left(x + y\right)}"
    # 测试 LaTeX 表示：分数 \frac{1}{(x + y)}/2 应转换为字符串 "\frac{1}{2 \left(x + y\right)}"

    assert latex(x/2) == r"\frac{x}{2}"
    # 测试 LaTeX 表示：分数 x/2 应转换为字符串 "\frac{x}{2}"

    assert latex(x/2, fold_short_frac=True) == r"x / 2"
    # 测试 LaTeX 表示：分数 x/2，带有 fold_short_frac 参数应转换为字符串 "x / 2"

    assert latex((x + y)/(2*x)) == r"\frac{x + y}{2 x}"
    # 测试 LaTeX 表示：分数 (x + y)/(2*x) 应转换为字符串 "\frac{x + y}{2 x}"

    assert latex((x + y)/(2*x), fold_short_frac=True) == \
        r"\left(x + y\right) / 2 x"
    # 测试 LaTeX 表示：分数 (x + y)/(2*x)，带有 fold_short_frac 参数应转换为字符串 "\left(x + y\right) / 2 x"

    assert latex((x + y)/(2*x), long_frac_ratio=0) == \
        r"\frac{1}{2 x} \left(x + y\right)"
    # 测试 LaTeX 表示：分数 (x + y)/(2*x)，带有 long_frac_ratio 参数应转换为字符串 "\frac{1}{2 x} \left(x + y\right)"

    assert latex((x + y)/x) == r"\frac{x + y}{x}"
    # 测试 LaTeX 表示：分数 (x + y)/x 应转换为字符串 "\frac{x + y}{x}"

    assert latex((x + y)/x, long_frac_ratio=3) == r"\frac{x + y}{x}"
    # 测试 LaTeX 表示：分数 (x + y)/x，带有 long_frac_ratio 参数应转换为字符串 "\frac{x + y}{x}"

    assert latex((2*sqrt(2)*x)/3) == r"\frac{2 \sqrt{2} x}{3}"
    # 测试 LaTeX 表示：分数 \frac{2 \sqrt{2} x}{3} 应转换为字符串 "\frac{2 \sqrt{2} x}{3}"

    assert latex((2*sqrt(2)*x)/3, long_frac_ratio=2) == \
        r"\frac{2 x}{3} \sqrt{2}"
    # 测试 LaTeX 表示：分数 \frac{2 \sqrt{2} x}{3}，带有 long_frac_ratio 参数应转换为字符串 "\frac{2 x}{3} \sqrt{2}"

    assert latex(binomial(x, y)) == r"{\binom{x}{y}}"
    # 测试 LaTeX 表示：二项式系数 \binom{x}{y} 应转换为字符串 "{\binom{x}{y}}"

    x_star = Symbol('x^*')
    f = Function('f')
    assert latex(x_star**2) == r"\left(x^{*}\right)^{2}"
    # 测试 LaTeX 表示：幂运算 (x^*)^2 应转换为字符串 "\left(x^{*}\right)^{2}"

    assert latex(x_star**2, parenthesize_super=False) == r"{x^{*}}^{2}"
    # 测试 LaTeX 表示：幂运算 (x^*)^2，不带括号应转换为字符串 "{x^{*}}^{2}"

    assert latex(Derivative(f(x_star), x_star,2)) == \
        r"\frac{d^{2}}{d \left(x^{*}\right)^{2}} f{\left(x^{*} \right)}"
    # 测试 LaTeX 表示：二阶导数 \frac{d^{2}}{d (x^*)^{2}} f(x^*) 应转换为字符串 "\frac{d^{2}}{d \left(x^{*}\right)^{2}} f{\left(x^{*} \right)}"

    assert latex(Derivative(f(x_star), x_star,2), parenthesize_super=False) == \
        r"\frac{d^{2}}{d {x^{*}}^{2}} f{\left(x^{*} \right)}"
    # 测试 LaTeX 表示：二阶导数 \frac{d^{2}}{d (x^*)^{2}} f(x^*)，不带括号应转换为字符串 "\frac{d^{2}}{d {x^{*}}^{2}} f{\left(x^{*} \right)}"

    assert latex(2*Integral(x, x)/3) == r"\frac{2 \int x\, dx}{3}"
    # 测试 LaTeX 表示：分数 \frac{2 \int x\, dx}{3} 应转换为字符串 "\frac{2 \int x\, dx}{3}"

    assert latex(2*Integral(x, x)/3, fold_short_frac=True) == \
        r"\left(2 \int x\, dx\right) / 3"
    # 测试 LaTeX 表示：分数 \frac{2 \int x\, dx}{3}，带有 fold_short_frac 参数应转换为字符串 "\left(2 \int x\, dx\right) / 3"

    assert latex(sqrt(x)) == r"\sqrt{x}"
    # 测试 LaTeX 表示：平方根 \sqrt{x} 应转换为字符串 "\sqrt{x}"

    assert latex(x**Rational(1, 3)) == r"\sqrt[3]{x}"
    # 测试 LaTeX 表示：立方根 x^{1/3} 应转换为字符串 "\sqrt[3]{x}"

    assert latex(x**Rational(1, 3), root_notation=False) == r"x^{\frac{1}{3}}"
    # 测试 LaTeX 表示：立方根 x^{1/3}，关闭根号符号应转换为字符串 "x^{\frac{1}{3}}"

    assert latex(sqrt(x)**3) == r"x^{\frac{3}{2}}"
    # 测试 LaTeX 表示：平方根的立方 \sqrt{x}^3 应转换为字符串 "x^{\frac{3}{2}}"

    assert latex(sqrt(x), itex=True) == r"\sqrt{x}"
    # 测试 LaTeX 表示：使用 itex 格式的平方根 \sqrt{x} 应转换为字符串 "\sqrt{x}"

    assert latex(x**Rational(1, 3), itex=True) == r"\root{3}{x}"
    # 测试 LaTeX 表示：使用 itex 格式的立方根 x^{1/3} 应转换为字符串 "\root{3}{x}"

    assert latex(sqrt(x)**3, itex=True) == r"x^{\frac{3}{2}}"
    # 测试 LaTeX 表示：使用 itex 格式的平方根的立方 \sqrt{x}^3 应转换为字符串 "x^{\frac{3}{2}}"

    assert latex(x**Rational(3, 4)) == r"x^{\frac{3}{4}}"
    # 测试 LaTeX 表示：x^{3/4} 应转换为字符串 "x^{\frac{3}{4}}"

    assert latex(x**Rational(3, 4), fold_frac_powers=True) == r"x^{3/4}"
    # 测试 LaTeX 表示：x^{3/4}，带有 fold_frac_powers 参数应转换为字符串 "x^{3/4}"

    assert latex((x + 1)**Rational(3, 4)) == \
        r"\left(x + 1\right)^{\frac{3}{4}}"
    # 测试 LaTeX 表示：(x + 1)^{3/4} 应转换为字符串 "\left(x + 1\right)^{\frac{3}{4}}"

    assert latex((x + 1)**Rational(3, 4), fold_frac_powers=True) == \
        r"\left(x + 1\right)^{3/4}"
    # 测试 LaTeX 表示：(x + 1)^{3/4}，带有 fold_frac_powers 参数应转换为字符串 "\left(x + 1\right)^{3/4}"

    assert latex(AlgebraicNumber(sqrt(2))) == r"\sqrt{2}"
    # 测试 LaTeX 表示：代数数 \sqrt{2} 应转换为字符串 "\
    # 断言：检查使用旧的顺序生成的大于19的素数的LaTeX表达式是否正确
    assert latex(k.primes_above(19)[0], order='old') == \
           r"\left(19, 1 + 5 \zeta + \zeta^{2}\right)"

    # 断言：检查生成大于7的素数的LaTeX表达式是否正确
    assert latex(k.primes_above(7)[0]) == r"\left(7\right)"

    # 断言：检查对数的乘法表示是否正确
    assert latex(1.5e20*x) == r"1.5 \cdot 10^{20} x"
    assert latex(1.5e20*x, mul_symbol='dot') == r"1.5 \cdot 10^{20} \cdot x"
    assert latex(1.5e20*x, mul_symbol='times') == \
        r"1.5 \times 10^{20} \times x"

    # 断言：检查正弦函数的倒数的LaTeX表达式是否正确
    assert latex(1/sin(x)) == r"\frac{1}{\sin{\left(x \right)}}"
    assert latex(sin(x)**-1) == r"\frac{1}{\sin{\left(x \right)}}"
    assert latex(sin(x)**Rational(3, 2)) == \
        r"\sin^{\frac{3}{2}}{\left(x \right)}"
    assert latex(sin(x)**Rational(3, 2), fold_frac_powers=True) == \
        r"\sin^{3/2}{\left(x \right)}"

    # 断言：检查逻辑非运算的LaTeX表达式是否正确
    assert latex(~x) == r"\neg x"

    # 断言：检查逻辑与运算的LaTeX表达式是否正确
    assert latex(x & y) == r"x \wedge y"
    assert latex(x & y & z) == r"x \wedge y \wedge z"

    # 断言：检查逻辑或运算的LaTeX表达式是否正确
    assert latex(x | y) == r"x \vee y"
    assert latex(x | y | z) == r"x \vee y \vee z"
    assert latex((x & y) | z) == r"z \vee \left(x \wedge y\right)"

    # 断言：检查蕴含（条件）运算的LaTeX表达式是否正确
    assert latex(Implies(x, y)) == r"x \Rightarrow y"
    assert latex(~(x >> ~y)) == r"x \not\Rightarrow \neg y"
    assert latex(Implies(Or(x,y), z)) == r"\left(x \vee y\right) \Rightarrow z"
    assert latex(Implies(z, Or(x,y))) == r"z \Rightarrow \left(x \vee y\right)"
    assert latex(~(x & y)) == r"\neg \left(x \wedge y\right)"

    # 断言：使用自定义符号名称检查逻辑表达式的LaTeX表达式是否正确
    assert latex(~x, symbol_names={x: "x_i"}) == r"\neg x_i"
    assert latex(x & y, symbol_names={x: "x_i", y: "y_i"}) == \
        r"x_i \wedge y_i"
    assert latex(x & y & z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"x_i \wedge y_i \wedge z_i"
    assert latex(x | y, symbol_names={x: "x_i", y: "y_i"}) == r"x_i \vee y_i"
    assert latex(x | y | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"x_i \vee y_i \vee z_i"
    assert latex((x & y) | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"z_i \vee \left(x_i \wedge y_i\right)"
    assert latex(Implies(x, y), symbol_names={x: "x_i", y: "y_i"}) == \
        r"x_i \Rightarrow y_i"

    # 断言：检查有理数指数的LaTeX表达式是否正确
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == r"\frac{1}{\frac{1}{3}}"
    assert latex(Pow(Rational(1, 3), -2, evaluate=False)) == r"\frac{1}{(\frac{1}{3})^{2}}"
    assert latex(Pow(Integer(1)/100, -1, evaluate=False)) == r"\frac{1}{\frac{1}{100}}"

    # 断言：检查指数函数和对数函数的LaTeX表达式是否正确
    p = Symbol('p', positive=True)
    assert latex(exp(-p)*log(p)) == r"e^{- p} \log{\left(p \right)}"
# 测试 LaTeX 输出函数对布尔值 True 的处理是否正确
assert latex(True) == r"\text{True}"

# 测试 LaTeX 输出函数对布尔值 False 的处理是否正确
assert latex(False) == r"\text{False}"

# 测试 LaTeX 输出函数对 None 的处理是否正确
assert latex(None) == r"\text{None}"

# 测试 LaTeX 输出函数对小写 true 的处理是否正确（应当会导致错误，因为 true 应为 True）
assert latex(true) == r"\text{True}"

# 测试 LaTeX 输出函数对小写 false 的处理是否正确（应当会导致错误，因为 false 应为 False）
assert latex(false) == r'\text{False}'


# 测试 LaTeX 输出函数对 SingularityFunction 类的输出是否正确
assert latex(SingularityFunction(x, 4, 5)) == \
    r"{\left\langle x - 4 \right\rangle}^{5}"
assert latex(SingularityFunction(x, -3, 4)) == \
    r"{\left\langle x + 3 \right\rangle}^{4}"
assert latex(SingularityFunction(x, 0, 4)) == \
    r"{\left\langle x \right\rangle}^{4}"
assert latex(SingularityFunction(x, a, n)) == \
    r"{\left\langle - a + x \right\rangle}^{n}"
assert latex(SingularityFunction(x, 4, -2)) == \
    r"{\left\langle x - 4 \right\rangle}^{-2}"
assert latex(SingularityFunction(x, 4, -1)) == \
    r"{\left\langle x - 4 \right\rangle}^{-1}"

# 测试 LaTeX 输出函数对 SingularityFunction 类的幂次处理是否正确
assert latex(SingularityFunction(x, 4, 5)**3) == \
    r"{\left({\langle x - 4 \rangle}^{5}\right)}^{3}"
assert latex(SingularityFunction(x, -3, 4)**3) == \
    r"{\left({\langle x + 3 \rangle}^{4}\right)}^{3}"
assert latex(SingularityFunction(x, 0, 4)**3) == \
    r"{\left({\langle x \rangle}^{4}\right)}^{3}"
assert latex(SingularityFunction(x, a, n)**3) == \
    r"{\left({\langle - a + x \rangle}^{n}\right)}^{3}"
assert latex(SingularityFunction(x, 4, -2)**3) == \
    r"{\left({\langle x - 4 \rangle}^{-2}\right)}^{3}"
assert latex((SingularityFunction(x, 4, -1)**3)**3) == \
    r"{\left({\langle x - 4 \rangle}^{-1}\right)}^{9}"


# 测试 LaTeX 输出函数对 Cycle 类的输出是否正确
assert latex(Cycle(1, 2, 4)) == r"\left( 1\; 2\; 4\right)"
assert latex(Cycle(1, 2)(4, 5, 6)) == \
    r"\left( 1\; 2\right)\left( 4\; 5\; 6\right)"
assert latex(Cycle()) == r"\left( \right)"


# 测试 LaTeX 输出函数对 Permutation 类的输出是否正确
assert latex(Permutation(1, 2, 4)) == r"\left( 1\; 2\; 4\right)"
assert latex(Permutation(1, 2)(4, 5, 6)) == \
    r"\left( 1\; 2\right)\left( 4\; 5\; 6\right)"
assert latex(Permutation()) == r"\left( \right)"
assert latex(Permutation(2, 4)*Permutation(5)) == \
    r"\left( 2\; 4\right)\left( 5\right)"
assert latex(Permutation(5)) == r"\left( 5\right)"

# 测试 LaTeX 输出函数对 Permutation 类的非循环表达式输出是否正确
assert latex(Permutation(0, 1), perm_cyclic=False) == \
    r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
assert latex(Permutation(0, 1)(2, 3), perm_cyclic=False) == \
    r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
assert latex(Permutation(), perm_cyclic=False) == \
    r"\left( \right)"

# 测试 LaTeX 输出函数对 Permutation 类的已弃用警告输出是否正确
with warns_deprecated_sympy():
    old_print_cyclic = Permutation.print_cyclic
    Permutation.print_cyclic = False
    assert latex(Permutation(0, 1)(2, 3)) == \
        r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
    Permutation.print_cyclic = old_print_cyclic


# 测试 LaTeX 输出函数对 Float 类的输出是否正确
assert latex(Float(1.0e100)) == r"1.0 \cdot 10^{100}"
    # 使用 SymPy 的 latex 函数将浮点数转换为 LaTeX 格式的字符串表示，并进行断言测试
    assert latex(Float(1.0e-100)) == r"1.0 \cdot 10^{-100}"
    
    # 使用 SymPy 的 latex 函数将浮点数转换为 LaTeX 格式的字符串表示，并指定乘号符号为 "times"，进行断言测试
    assert latex(Float(1.0e-100), mul_symbol="times") == \
        r"1.0 \times 10^{-100}"
    
    # 使用 SymPy 的 latex 函数将浮点数转换为 LaTeX 格式的字符串表示，并限制指数在 -2 到 2 之间，不显示完整精度，进行断言测试
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=2) == \
        r"1.0 \cdot 10^{4}"
    
    # 使用 SymPy 的 latex 函数将浮点数转换为 LaTeX 格式的字符串表示，并限制指数在 -2 到 4 之间，不显示完整精度，进行断言测试
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=4) == \
        r"1.0 \cdot 10^{4}"
    
    # 使用 SymPy 的 latex 函数将浮点数转换为 LaTeX 格式的字符串表示，并限制指数在 -2 到 5 之间，不显示完整精度，进行断言测试
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=5) == \
        r"10000.0"
    
    # 使用 SymPy 的 latex 函数将浮点数转换为 LaTeX 格式的字符串表示，并限制指数在 -2 到 5 之间，显示完整精度，进行断言测试
    assert latex(Float('0.099999'), full_prec=True, min=-2, max=5) == \
        r"9.99990000000000 \cdot 10^{-2}"
# 定义一个测试函数，用于测试 LaTeX 中矢量表达式的转换
def test_latex_vector_expressions():
    # 创建一个三维坐标系对象 A
    A = CoordSys3D('A')

    # 断言：计算两个向量的叉乘后转换为 LaTeX 格式的表达式
    assert latex(Cross(A.i, A.j*A.x*3+A.k)) == \
        r"\mathbf{\hat{i}_{A}} \times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}} + \mathbf{\hat{k}_{A}}\right)"
    
    # 断言：计算两个向量的叉乘后转换为 LaTeX 格式的表达式
    assert latex(Cross(A.i, A.j)) == \
        r"\mathbf{\hat{i}_{A}} \times \mathbf{\hat{j}_{A}}"
    
    # 断言：计算一个标量与向量叉乘后转换为 LaTeX 格式的表达式
    assert latex(x*Cross(A.i, A.j)) == \
        r"x \left(\mathbf{\hat{i}_{A}} \times \mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个向量与另一个向量的叉乘后转换为 LaTeX 格式的表达式
    assert latex(Cross(x*A.i, A.j)) == \
        r'- \mathbf{\hat{j}_{A}} \times \left(\left(x\right)\mathbf{\hat{i}_{A}}\right)'
    
    # 断言：计算一个向量的旋度后转换为 LaTeX 格式的表达式
    assert latex(Curl(3*A.x*A.j)) == \
        r"\nabla\times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个向量表达式的旋度后转换为 LaTeX 格式的表达式
    assert latex(Curl(3*A.x*A.j+A.i)) == \
        r"\nabla\times \left(\mathbf{\hat{i}_{A}} + \left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个标量与向量表达式的旋度后转换为 LaTeX 格式的表达式
    assert latex(Curl(3*x*A.x*A.j)) == \
        r"\nabla\times \left(\left(3 \mathbf{{x}_{A}} x\right)\mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个标量与向量旋度的乘积后转换为 LaTeX 格式的表达式
    assert latex(x*Curl(3*A.x*A.j)) == \
        r"x \left(\nabla\times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)\right)"
    
    # 断言：计算一个向量的散度后转换为 LaTeX 格式的表达式
    assert latex(Divergence(3*A.x*A.j+A.i)) == \
        r"\nabla\cdot \left(\mathbf{\hat{i}_{A}} + \left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个向量表达式的散度后转换为 LaTeX 格式的表达式
    assert latex(Divergence(3*A.x*A.j)) == \
        r"\nabla\cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个标量与向量表达式的散度后转换为 LaTeX 格式的表达式
    assert latex(x*Divergence(3*A.x*A.j)) == \
        r"x \left(\nabla\cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)\right)"
    
    # 断言：计算两个向量的点积后转换为 LaTeX 格式的表达式
    assert latex(Dot(A.i, A.j*A.x*3+A.k)) == \
        r"\mathbf{\hat{i}_{A}} \cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}} + \mathbf{\hat{k}_{A}}\right)"
    
    # 断言：计算两个向量的点积后转换为 LaTeX 格式的表达式
    assert latex(Dot(A.i, A.j)) == \
        r"\mathbf{\hat{i}_{A}} \cdot \mathbf{\hat{j}_{A}}"
    
    # 断言：计算一个标量与向量的点积后转换为 LaTeX 格式的表达式
    assert latex(Dot(x*A.i, A.j)) == \
        r"\mathbf{\hat{j}_{A}} \cdot \left(\left(x\right)\mathbf{\hat{i}_{A}}\right)"
    
    # 断言：计算一个标量与两个向量的点积后转换为 LaTeX 格式的表达式
    assert latex(x*Dot(A.i, A.j)) == \
        r"x \left(\mathbf{\hat{i}_{A}} \cdot \mathbf{\hat{j}_{A}}\right)"
    
    # 断言：计算一个标量与梯度后转换为 LaTeX 格式的表达式
    assert latex(Gradient(A.x)) == r"\nabla \mathbf{{x}_{A}}"
    
    # 断言：计算一个向量表达式的梯度后转换为 LaTeX 格式的表达式
    assert latex(Gradient(A.x + 3*A.y)) == \
        r"\nabla \left(\mathbf{{x}_{A}} + 3 \mathbf{{y}_{A}}\right)"
    
    # 断言：计算一个标量与向量的乘积的梯度后转换为 LaTeX 格式的表达式
    assert latex(x*Gradient(A.x)) == r"x \left(\nabla \mathbf{{x}_{A}}\right)"
    
    # 断言：计算一个标量与向量表达式的梯度后转换为 LaTeX 格式的表达式
    assert latex(Gradient(x*A.x)) == r"\nabla \left(\mathbf{{x}_{A}} x\right)"
    
    # 断言：计算一个标量与拉普拉斯算子后转换为 LaTeX 格式的表达式
    assert latex(Laplacian(A.x)) == r"\Delta \mathbf{{x}_{A}}"
    
    # 断言：计算一个向量表达式的拉普拉斯算子后转换为 LaTeX 格式的表达式
    assert latex(Laplacian(A.x + 3*A.y)) == \
        r"\Delta \left(\mathbf{{x}_{A}} + 3 \mathbf{{y}_{A}}\right)"
    
    # 断言：计算一个标量与向量的乘积的拉普拉斯算子后转换为 LaTeX 格式的表达式
    assert latex(x*Laplacian(A.x)) == r"x \left(\Delta \mathbf{{x}_{A}}\right)"
    
    # 断言：计算一个标量与向量表达式的拉普拉斯算子后转换为 LaTeX 格式的表达式
    assert latex(Laplacian(x*A.x)) == r"\Delta \left(\mathbf{{x}_{A}} x\right)"

# 定义一个测试函数，用于测试 LaTeX 中的符号转换
def test_latex_symbols():
    # 定义几个符号变量
    Gamma, lmbda, rho = symbols('Gamma, lambda, rho')
    tau, Tau, TAU, taU = symbols('tau, Tau, TAU, taU')
    
    # 断言：将 tau 符号转换为 LaTeX 格式的表达式
    assert latex(tau) == r"\tau"
    
    # 断言：将 Tau 符号转换为 LaTeX 格式的表达式
    assert latex(Tau) == r"\mathrm{T}"
    # 断言，确保 TAU 的 LaTeX 表示为 r"\tau"
    assert latex(TAU) == r"\tau"
    
    # 断言，确保 taU 的 LaTeX 表示也为 r"\tau"
    assert latex(taU) == r"\tau"
    
    # 检查所有大写希腊字母是否都已显式处理
    capitalized_letters = {l.capitalize() for l in greek_letters_set}
    
    # 断言，确保所有大写希腊字母在 tex_greek_dictionary 中都有对应的 LaTeX 表示
    assert len(capitalized_letters - set(tex_greek_dictionary.keys())) == 0
    
    # 断言，确保 Gamma + lmbda 的 LaTeX 表示为 r"\Gamma + \lambda"
    assert latex(Gamma + lmbda) == r"\Gamma + \lambda"
    
    # 断言，确保 Gamma * lmbda 的 LaTeX 表示为 r"\Gamma \lambda"
    assert latex(Gamma * lmbda) == r"\Gamma \lambda"
    
    # 断言，确保 Symbol('q1') 的 LaTeX 表示为 r"q_{1}"
    assert latex(Symbol('q1')) == r"q_{1}"
    
    # 断言，确保 Symbol('q21') 的 LaTeX 表示为 r"q_{21}"
    assert latex(Symbol('q21')) == r"q_{21}"
    
    # 断言，确保 Symbol('epsilon0') 的 LaTeX 表示为 r"\epsilon_{0}"
    assert latex(Symbol('epsilon0')) == r"\epsilon_{0}"
    
    # 断言，确保 Symbol('omega1') 的 LaTeX 表示为 r"\omega_{1}"
    assert latex(Symbol('omega1')) == r"\omega_{1}"
    
    # 断言，确保 Symbol('91') 的 LaTeX 表示为 r"91"
    assert latex(Symbol('91')) == r"91"
    
    # 断言，确保 Symbol('alpha_new') 的 LaTeX 表示为 r"\alpha_{new}"
    assert latex(Symbol('alpha_new')) == r"\alpha_{new}"
    
    # 断言，确保 Symbol('C^orig') 的 LaTeX 表示为 r"C^{orig}"
    assert latex(Symbol('C^orig')) == r"C^{orig}"
    
    # 断言，确保 Symbol('x^alpha') 的 LaTeX 表示为 r"x^{\alpha}"
    assert latex(Symbol('x^alpha')) == r"x^{\alpha}"
    
    # 断言，确保 Symbol('beta^alpha') 的 LaTeX 表示为 r"\beta^{\alpha}"
    assert latex(Symbol('beta^alpha')) == r"\beta^{\alpha}"
    
    # 断言，确保 Symbol('e^Alpha') 的 LaTeX 表示为 r"e^{\mathrm{A}}"
    assert latex(Symbol('e^Alpha')) == r"e^{\mathrm{A}}"
    
    # 断言，确保 Symbol('omega_alpha^beta') 的 LaTeX 表示为 r"\omega^{\beta}_{\alpha}"
    assert latex(Symbol('omega_alpha^beta')) == r"\omega^{\beta}_{\alpha}"
    
    # 断言，确保 Symbol('omega') ** Symbol('beta') 的 LaTeX 表示为 r"\omega^{\beta}"
    assert latex(Symbol('omega') ** Symbol('beta')) == r"\omega^{\beta}"
# 标记该测试函数为预期失败，不希望通过测试
@XFAIL
def test_latex_symbols_failing():
    # 定义符号 rho, mass, volume
    rho, mass, volume = symbols('rho, mass, volume')
    # 验证 LaTeX 表达式转换：volume * rho == mass
    assert latex(volume * rho == mass) == r"\rho \mathrm{volume} = \mathrm{mass}"
    # 验证 LaTeX 表达式转换：volume / mass * rho == 1
    assert latex(volume / mass * rho == 1) == \
        r"\rho \mathrm{volume} {\mathrm{mass}}^{(-1)} = 1"
    # 验证 LaTeX 表达式转换：mass**3 * volume**3
    assert latex(mass**3 * volume**3) == \
        r"{\mathrm{mass}}^{3} \cdot {\mathrm{volume}}^{3}"


# 使用装饰器 @_both_exp_pow 标记的测试函数
@_both_exp_pow
def test_latex_functions():
    # 验证 LaTeX 表达式转换：exp(x)
    assert latex(exp(x)) == r"e^{x}"
    # 验证 LaTeX 表达式转换：exp(1) + exp(2)
    assert latex(exp(1) + exp(2)) == r"e + e^{2}"

    # 定义函数 f
    f = Function('f')
    # 验证 LaTeX 表达式转换：f(x)
    assert latex(f(x)) == r'f{\left(x \right)}'
    # 验证 LaTeX 表达式转换：f
    assert latex(f) == r'f'

    # 定义函数 g
    g = Function('g')
    # 验证 LaTeX 表达式转换：g(x, y)
    assert latex(g(x, y)) == r'g{\left(x,y \right)}'
    # 验证 LaTeX 表达式转换：g
    assert latex(g) == r'g'

    # 定义函数 h
    h = Function('h')
    # 验证 LaTeX 表达式转换：h(x, y, z)
    assert latex(h(x, y, z)) == r'h{\left(x,y,z \right)}'
    # 验证 LaTeX 表达式转换：h
    assert latex(h) == r'h'

    # 定义函数 Li
    Li = Function('Li')
    # 验证 LaTeX 表达式转换：Li
    assert latex(Li) == r'\operatorname{Li}'
    # 验证 LaTeX 表达式转换：Li(x)
    assert latex(Li(x)) == r'\operatorname{Li}{\left(x \right)}'

    # 定义函数 mybeta
    mybeta = Function('beta')
    # 验证 LaTeX 表达式转换：mybeta(x, y, z)
    assert latex(mybeta(x, y, z)) == r"\beta{\left(x,y,z \right)}"
    # 验证 LaTeX 表达式转换：beta(x, y)
    assert latex(beta(x, y)) == r'\operatorname{B}\left(x, y\right)'
    # 验证 LaTeX 表达式转换：beta(x, evaluate=False)
    assert latex(beta(x, evaluate=False)) == r'\operatorname{B}\left(x, x\right)'
    # 验证 LaTeX 表达式转换：beta(x, y)**2
    assert latex(beta(x, y)**2) == r'\operatorname{B}^{2}\left(x, y\right)'
    # 验证 LaTeX 表达式转换：mybeta(x)
    assert latex(mybeta(x)) == r"\beta{\left(x \right)}"
    # 验证 LaTeX 表达式转换：mybeta
    assert latex(mybeta) == r"\beta"

    # 定义函数 g
    g = Function('gamma')
    # 验证 LaTeX 表达式转换：g(x, y, z)
    assert latex(g(x, y, z)) == r"\gamma{\left(x,y,z \right)}"
    # 验证 LaTeX 表达式转换：g(x)
    assert latex(g(x)) == r"\gamma{\left(x \right)}"
    # 验证 LaTeX 表达式转换：g
    assert latex(g) == r"\gamma"

    # 定义函数 a_1
    a_1 = Function('a_1')
    # 验证 LaTeX 表达式转换：a_1
    assert latex(a_1) == r"a_{1}"
    # 验证 LaTeX 表达式转换：a_1(x)
    assert latex(a_1(x)) == r"a_{1}{\left(x \right)}"
    # 验证 LaTeX 表达式转换：Function('a_1')
    assert latex(Function('a_1')) == r"a_{1}"

    # 验证 Issue #16925 中多字母函数名的 LaTeX 表达式转换
    # > 简单函数名
    assert latex(Function('ab')) == r"\operatorname{ab}"
    assert latex(Function('ab1')) == r"\operatorname{ab}_{1}"
    assert latex(Function('ab12')) == r"\operatorname{ab}_{12}"
    assert latex(Function('ab_1')) == r"\operatorname{ab}_{1}"
    assert latex(Function('ab_12')) == r"\operatorname{ab}_{12}"
    assert latex(Function('ab_c')) == r"\operatorname{ab}_{c}"
    assert latex(Function('ab_cd')) == r"\operatorname{ab}_{cd}"
    # > 带参数的函数名
    assert latex(Function('ab')(Symbol('x'))) == r"\operatorname{ab}{\left(x \right)}"
    assert latex(Function('ab1')(Symbol('x'))) == r"\operatorname{ab}_{1}{\left(x \right)}"
    assert latex(Function('ab12')(Symbol('x'))) == r"\operatorname{ab}_{12}{\left(x \right)}"
    assert latex(Function('ab_1')(Symbol('x'))) == r"\operatorname{ab}_{1}{\left(x \right)}"
    assert latex(Function('ab_c')(Symbol('x'))) == r"\operatorname{ab}_{c}{\left(x \right)}"
    assert latex(Function('ab_cd')(Symbol('x'))) == r"\operatorname{ab}_{cd}{\left(x \right)}"

    # > 带指数的函数名
    #   函数没有括号时不起作用
    # > with argument and power combined
    # 检查函数名称为 'ab' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab')()**2) == r"\operatorname{ab}^{2}{\left( \right)}"
    # 检查函数名称为 'ab1' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab1')()**2) == r"\operatorname{ab}_{1}^{2}{\left( \right)}"
    # 检查函数名称为 'ab12' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab12')()**2) == r"\operatorname{ab}_{12}^{2}{\left( \right)}"
    # 检查函数名称为 'ab_1' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab_1')()**2) == r"\operatorname{ab}_{1}^{2}{\left( \right)}"
    # 检查函数名称为 'ab_12' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab_12')()**2) == r"\operatorname{ab}_{12}^{2}{\left( \right)}"
    # 检查函数名称为 'ab' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab')(Symbol('x'))**2) == r"\operatorname{ab}^{2}{\left(x \right)}"
    # 检查函数名称为 'ab1' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab1')(Symbol('x'))**2) == r"\operatorname{ab}_{1}^{2}{\left(x \right)}"
    # 检查函数名称为 'ab12' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab12')(Symbol('x'))**2) == r"\operatorname{ab}_{12}^{2}{\left(x \right)}"
    # 检查函数名称为 'ab_1' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab_1')(Symbol('x'))**2) == r"\operatorname{ab}_{1}^{2}{\left(x \right)}"
    # 检查函数名称为 'ab_12' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('ab_12')(Symbol('x'))**2) == \
        r"\operatorname{ab}_{12}^{2}{\left(x \right)}"

    # single letter function names
    # > simple
    # 检查函数名称为 'a' 的函数，返回其 LaTeX 表示
    assert latex(Function('a')) == r"a"
    # 检查函数名称为 'a1' 的函数，返回其 LaTeX 表示
    assert latex(Function('a1')) == r"a_{1}"
    # 检查函数名称为 'a12' 的函数，返回其 LaTeX 表示
    assert latex(Function('a12')) == r"a_{12}"
    # 检查函数名称为 'a_1' 的函数，返回其 LaTeX 表示
    assert latex(Function('a_1')) == r"a_{1}"
    # 检查函数名称为 'a_12' 的函数，返回其 LaTeX 表示
    assert latex(Function('a_12')) == r"a_{12}"

    # > with argument
    # 检查函数名称为 'a' 的函数，带有一个空参数，返回其 LaTeX 表示
    assert latex(Function('a')()) == r"a{\left( \right)}"
    # 检查函数名称为 'a1' 的函数，带有一个空参数，返回其 LaTeX 表示
    assert latex(Function('a1')()) == r"a_{1}{\left( \right)}"
    # 检查函数名称为 'a12' 的函数，带有一个空参数，返回其 LaTeX 表示
    assert latex(Function('a12')()) == r"a_{12}{\left( \right)}"
    # 检查函数名称为 'a_1' 的函数，带有一个空参数，返回其 LaTeX 表示
    assert latex(Function('a_1')()) == r"a_{1}{\left( \right)}"
    # 检查函数名称为 'a_12' 的函数，带有一个空参数，返回其 LaTeX 表示
    assert latex(Function('a_12')()) == r"a_{12}{\left( \right)}"

    # > with power
    # 不适用于没有括号的函数

    # > with argument and power combined
    # 检查函数名称为 'a' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a')()**2) == r"a^{2}{\left( \right)}"
    # 检查函数名称为 'a1' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a1')()**2) == r"a_{1}^{2}{\left( \right)}"
    # 检查函数名称为 'a12' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a12')()**2) == r"a_{12}^{2}{\left( \right)}"
    # 检查函数名称为 'a_1' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a_1')()**2) == r"a_{1}^{2}{\left( \right)}"
    # 检查函数名称为 'a_12' 的函数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a_12')()**2) == r"a_{12}^{2}{\left( \right)}"
    # 检查函数名称为 'a' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a')(Symbol('x'))**2) == r"a^{2}{\left(x \right)}"
    # 检查函数名称为 'a1' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a1')(Symbol('x'))**2) == r"a_{1}^{2}{\left(x \right)}"
    # 检查函数名称为 'a12' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a12')(Symbol('x'))**2) == r"a_{12}^{2}{\left(x \right)}"
    # 检查函数名称为 'a_1' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a_1')(Symbol('x'))**2) == r"a_{1}^{2}{\left(x \right)}"
    # 检查函数名称为 'a_12' 的函数，带有一个符号 'x' 作为参数，返回其调用结果的平方的 LaTeX 表示
    assert latex(Function('a_12')(Symbol('x'))**2) == r"a_{12}^{2}{\left(x \right)}"

    # 检查函数名称为 'a' 的函数，返回其调用结果的32次方的 LaTeX 表示
    assert latex(Function('a')()**32) == r"a^{32}{\left( \right)}"
    # 检查函数名称为 'a1' 的函数，返回其调用结果的32次方的 LaTeX 表示
    assert latex(Function('a1')()**32) == r"a_{1}^{32}{\left( \right)}"
    # 检查函数名称为 'a12' 的函数，返回其调用结果的32次方的 LaTeX 表示
    assert latex(Function('a12')()**32) == r"a_{12}^{32}{\left( \right)}"
    # 检查函数名称为 'a_1' 的函数，返回其调用结果的32次方的
    # 第一组断言，测试 Function 类生成的表达式转换成 LaTeX 的输出是否符合预期
    assert latex(Function('a12')(Symbol('x'))**32) == r"a_{12}^{32}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**32) == r"a_{1}^{32}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**32) == r"a_{12}^{32}{\left(x \right)}"
    
    # 第二组断言，测试带参数的 Function 类生成的表达式转换成 LaTeX 的输出是否符合预期
    assert latex(Function('a')()**a) == r"a^{a}{\left( \right)}"
    assert latex(Function('a1')()**a) == r"a_{1}^{a}{\left( \right)}"
    assert latex(Function('a12')()**a) == r"a_{12}^{a}{\left( \right)}"
    assert latex(Function('a_1')()**a) == r"a_{1}^{a}{\left( \right)}"
    assert latex(Function('a_12')()**a) == r"a_{12}^{a}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**a) == r"a^{a}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**a) == r"a_{1}^{a}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**a) == r"a_{12}^{a}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**a) == r"a_{1}^{a}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**a) == r"a_{12}^{a}{\left(x \right)}"
    
    # 第三组断言，测试带 Symbol 参数的 Function 类生成的表达式转换成 LaTeX 的输出是否符合预期
    ab = Symbol('ab')
    assert latex(Function('a')()**ab) == r"a^{ab}{\left( \right)}"
    assert latex(Function('a1')()**ab) == r"a_{1}^{ab}{\left( \right)}"
    assert latex(Function('a12')()**ab) == r"a_{12}^{ab}{\left( \right)}"
    assert latex(Function('a_1')()**ab) == r"a_{1}^{ab}{\left( \right)}"
    assert latex(Function('a_12')()**ab) == r"a_{12}^{ab}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**ab) == r"a^{ab}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**ab) == r"a_{1}^{ab}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**ab) == r"a_{12}^{ab}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**ab) == r"a_{1}^{ab}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**ab) == r"a_{12}^{ab}{\left(x \right)}"
    
    # 第四组断言，测试带幂操作的 Function 类生成的表达式转换成 LaTeX 的输出是否符合预期
    assert latex(Function('a^12')(x)) == r"a^{12}{\left(x \right)}"
    assert latex(Function('a^12')(x) ** ab) == r"\left(a^{12}\right)^{ab}{\left(x \right)}"
    assert latex(Function('a__12')(x)) == r"a^{12}{\left(x \right)}"
    assert latex(Function('a__12')(x) ** ab) == r"\left(a^{12}\right)^{ab}{\left(x \right)}"
    assert latex(Function('a_1__1_2')(x)) == r"a^{1}_{1 2}{\left(x \right)}"
    
    # issue 5868
    # 测试特定问题的修复，确保 Function('omega1') 生成的表达式转换成 LaTeX 的输出是否符合预期
    omega1 = Function('omega1')
    assert latex(omega1) == r"\omega_{1}"
    assert latex(omega1(x)) == r"\omega_{1}{\left(x \right)}"
    
    # 测试标准三角函数 sin(x) 的表达式转换成 LaTeX 的输出是否符合预期
    assert latex(sin(x)) == r"\sin{\left(x \right)}"
    assert latex(sin(x), fold_func_brackets=True) == r"\sin {x}"
    assert latex(sin(2*x**2), fold_func_brackets=True) == r"\sin {2 x^{2}}"
    assert latex(sin(x**2), fold_func_brackets=True) == r"\sin {x^{2}}"
    
    # 测试反三角函数 asin(x) 的表达式转换成 LaTeX 的输出是否符合预期
    assert latex(asin(x)**2) == r"\operatorname{asin}^{2}{\left(x \right)}"
    assert latex(asin(x)**2, inv_trig_style="full") == r"\arcsin^{2}{\left(x \right)}"
    assert latex(asin(x)**2, inv_trig_style="power") == r"\sin^{-1}{\left(x \right)}^{2}"
    # 检查 asin(x**2) 的 LaTeX 表示是否与预期的反三角函数幂次样式匹配
    assert latex(asin(x**2), inv_trig_style="power",
                 fold_func_brackets=True) == \
        r"\sin^{-1} {x^{2}}"

    # 检查 acsc(x) 的 LaTeX 表示是否与预期的完整反三角函数样式匹配
    assert latex(acsc(x), inv_trig_style="full") == \
        r"\operatorname{arccsc}{\left(x \right)}"

    # 检查 asinh(x) 的 LaTeX 表示是否与预期的完整反双曲正弦函数样式匹配
    assert latex(asinh(x), inv_trig_style="full") == \
        r"\operatorname{arsinh}{\left(x \right)}"

    # 检查阶乘函数 factorial(k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(factorial(k)) == r"k!"

    # 检查阶乘函数 factorial(-k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(factorial(-k)) == r"\left(- k\right)!"

    # 检查阶乘函数 factorial(k)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(factorial(k)**2) == r"k!^{2}"

    # 检查下列函数 subfactorial(k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(subfactorial(k)) == r"!k"

    # 检查下列函数 subfactorial(-k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(subfactorial(-k)) == r"!\left(- k\right)"

    # 检查下列函数 subfactorial(k)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(subfactorial(k)**2) == r"\left(!k\right)^{2}"

    # 检查下列函数 factorial2(k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(factorial2(k)) == r"k!!"

    # 检查下列函数 factorial2(-k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(factorial2(-k)) == r"\left(- k\right)!!"

    # 检查下列函数 factorial2(k)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(factorial2(k)**2) == r"k!!^{2}"

    # 检查二项式系数函数 binomial(2, k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(binomial(2, k)) == r"{\binom{2}{k}}"

    # 检查二项式系数函数 binomial(2, k)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(binomial(2, k)**2) == r"{\binom{2}{k}}^{2}"

    # 检查 FallingFactorial(3, k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(FallingFactorial(3, k)) == r"{\left(3\right)}_{k}"

    # 检查 RisingFactorial(3, k) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(RisingFactorial(3, k)) == r"{3}^{\left(k\right)}"

    # 检查 floor(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(floor(x)) == r"\left\lfloor{x}\right\rfloor"

    # 检查 ceiling(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(ceiling(x)) == r"\left\lceil{x}\right\rceil"

    # 检查 frac(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(frac(x)) == r"\operatorname{frac}{\left(x\right)}"

    # 检查 floor(x)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(floor(x)**2) == r"\left\lfloor{x}\right\rfloor^{2}"

    # 检查 ceiling(x)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(ceiling(x)**2) == r"\left\lceil{x}\right\rceil^{2}"

    # 检查 frac(x)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(frac(x)**2) == r"\operatorname{frac}{\left(x\right)}^{2}"

    # 检查 Min(x, 2, x**3) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Min(x, 2, x**3)) == r"\min\left(2, x, x^{3}\right)"

    # 检查 Min(x, y)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Min(x, y)**2) == r"\min\left(x, y\right)^{2}"

    # 检查 Max(x, 2, x**3) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Max(x, 2, x**3)) == r"\max\left(2, x, x^{3}\right)"

    # 检查 Max(x, y)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Max(x, y)**2) == r"\max\left(x, y\right)^{2}"

    # 检查 Abs(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Abs(x)) == r"\left|{x}\right|"

    # 检查 Abs(x)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Abs(x)**2) == r"\left|{x}\right|^{2}"

    # 检查 re(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(re(x)) == r"\operatorname{re}{\left(x\right)}"

    # 检查 re(x + y) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(re(x + y)) == \
        r"\operatorname{re}{\left(x\right)} + \operatorname{re}{\left(y\right)}"

    # 检查 im(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(im(x)) == r"\operatorname{im}{\left(x\right)}"

    # 检查 conjugate(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(conjugate(x)) == r"\overline{x}"

    # 检查 conjugate(x)**2 的 LaTeX 表示是否与预期的格式匹配
    assert latex(conjugate(x)**2) == r"\overline{x}^{2}"

    # 检查 conjugate(x**2) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(conjugate(x**2)) == r"\overline{x}^{2}"

    # 检查 gamma(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(gamma(x)) == r"\Gamma\left(x\right)"

    # 检查 gamma(w) 的 LaTeX 表示是否与预期的格式匹配
    w = Wild('w')
    assert latex(gamma(w)) == r"\Gamma\left(w\right)"

    # 检查 Order(x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Order(x)) == r"O\left(x\right)"

    # 检查 Order(x, x) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Order(x, x)) == r"O\left(x\right)"

    # 检查 Order(x, (x, 0)) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Order(x, (x, 0))) == r"O\left(x\right)"

    # 检查 Order(x, (x, oo)) 的 LaTeX 表示是否与预期的格式匹配
    assert latex(Order(x, (x, oo))) == r"O\left(x; x\rightarrow \infty\right)"

    # 检查 Order(x - y, (x, y)) 的 LaTeX 表示是否与预期的格式
    # 断言latex(Order(x, (x, oo), (y, oo)))的结果与指定的 LaTeX 表达式相等
    assert latex(Order(x, (x, oo), (y, oo))) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( \infty, \  \infty\right)\right)"
    
    # 断言latex(lowergamma(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(lowergamma(x, y)) == r'\gamma\left(x, y\right)'
    
    # 断言latex(lowergamma(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(lowergamma(x, y)**2) == r'\gamma^{2}\left(x, y\right)'
    
    # 断言latex(uppergamma(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(uppergamma(x, y)) == r'\Gamma\left(x, y\right)'
    
    # 断言latex(uppergamma(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(uppergamma(x, y)**2) == r'\Gamma^{2}\left(x, y\right)'
    
    # 断言latex(cot(x))的结果与指定的 LaTeX 表达式相等
    assert latex(cot(x)) == r'\cot{\left(x \right)}'
    
    # 断言latex(coth(x))的结果与指定的 LaTeX 表达式相等
    assert latex(coth(x)) == r'\coth{\left(x \right)}'
    
    # 断言latex(re(x))的结果与指定的 LaTeX 表达式相等
    assert latex(re(x)) == r'\operatorname{re}{\left(x\right)}'
    
    # 断言latex(im(x))的结果与指定的 LaTeX 表达式相等
    assert latex(im(x)) == r'\operatorname{im}{\left(x\right)}'
    
    # 断言latex(root(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(root(x, y)) == r'x^{\frac{1}{y}}'
    
    # 断言latex(arg(x))的结果与指定的 LaTeX 表达式相等
    assert latex(arg(x)) == r'\arg{\left(x \right)}'
    
    # 断言latex(zeta(x))的结果与指定的 LaTeX 表达式相等
    assert latex(zeta(x)) == r"\zeta\left(x\right)"
    
    # 断言latex(zeta(x)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(zeta(x)**2) == r"\zeta^{2}\left(x\right)"
    
    # 断言latex(zeta(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(zeta(x, y)) == r"\zeta\left(x, y\right)"
    
    # 断言latex(zeta(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(zeta(x, y)**2) == r"\zeta^{2}\left(x, y\right)"
    
    # 断言latex(dirichlet_eta(x))的结果与指定的 LaTeX 表达式相等
    assert latex(dirichlet_eta(x)) == r"\eta\left(x\right)"
    
    # 断言latex(dirichlet_eta(x)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(dirichlet_eta(x)**2) == r"\eta^{2}\left(x\right)"
    
    # 断言latex(polylog(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(polylog(x, y)) == r"\operatorname{Li}_{x}\left(y\right)"
    
    # 断言latex(polylog(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(polylog(x, y)**2) == r"\operatorname{Li}_{x}^{2}\left(y\right)"
    
    # 断言latex(lerchphi(x, y, n))的结果与指定的 LaTeX 表达式相等
    assert latex(lerchphi(x, y, n)) == r"\Phi\left(x, y, n\right)"
    
    # 断言latex(lerchphi(x, y, n)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(lerchphi(x, y, n)**2) == r"\Phi^{2}\left(x, y, n\right)"
    
    # 断言latex(stieltjes(x))的结果与指定的 LaTeX 表达式相等
    assert latex(stieltjes(x)) == r"\gamma_{x}"
    
    # 断言latex(stieltjes(x)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(stieltjes(x)**2) == r"\gamma_{x}^{2}"
    
    # 断言latex(stieltjes(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(stieltjes(x, y)) == r"\gamma_{x}\left(y\right)"
    
    # 断言latex(stieltjes(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(stieltjes(x, y)**2) == r"\gamma_{x}\left(y\right)^{2}"
    
    # 断言latex(elliptic_k(z))的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_k(z)) == r"K\left(z\right)"
    
    # 断言latex(elliptic_k(z)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_k(z)**2) == r"K^{2}\left(z\right)"
    
    # 断言latex(elliptic_f(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_f(x, y)) == r"F\left(x\middle| y\right)"
    
    # 断言latex(elliptic_f(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_f(x, y)**2) == r"F^{2}\left(x\middle| y\right)"
    
    # 断言latex(elliptic_e(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_e(x, y)) == r"E\left(x\middle| y\right)"
    
    # 断言latex(elliptic_e(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_e(x, y)**2) == r"E^{2}\left(x\middle| y\right)"
    
    # 断言latex(elliptic_e(z))的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_e(z)) == r"E\left(z\right)"
    
    # 断言latex(elliptic_e(z)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_e(z)**2) == r"E^{2}\left(z\right)"
    
    # 断言latex(elliptic_pi(x, y, z))的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_pi(x, y, z)) == r"\Pi\left(x; y\middle| z\right)"
    
    # 断言latex(elliptic_pi(x, y, z)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_pi(x, y, z)**2) == r"\Pi^{2}\left(x; y\middle| z\right)"
    
    # 断言latex(elliptic_pi(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_pi(x, y)) == r"\Pi\left(x\middle| y\right)"
    
    # 断言latex(elliptic_pi(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(elliptic_pi(x, y)**2) == r"\Pi^{2}\left(x\middle| y\right)"
    
    # 断言latex(Ei(x))的结果与指定的 LaTeX 表达式相等
    assert latex(Ei(x)) == r'\operatorname{Ei}{\left(x \right)}'
    
    # 断言latex(Ei(x)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(Ei(x)**2) == r'\operatorname{Ei}^{2}{\left(x \right)}'
    
    # 断言latex(expint(x, y))的结果与指定的 LaTeX 表达式相等
    assert latex(expint(x, y)) == r'\operatorname{E}_{x}\left(y\right)'
    
    # 断言latex(expint(x, y)**2)的结果与指定的 LaTeX 表达式相等
    assert latex(expint(x
    # 断言，验证 Ci(x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Ci(x)**2) == r'\operatorname{Ci}^{2}{\left(x \right)}'

    # 断言，验证 Chi(x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Chi(x)**2) == r'\operatorname{Chi}^{2}\left(x\right)'

    # 断言，验证 Chi(x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Chi(x)) == r'\operatorname{Chi}\left(x\right)'

    # 断言，验证 jacobi(n, a, b, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(jacobi(n, a, b, x)) == \
        r'P_{n}^{\left(a,b\right)}\left(x\right)'

    # 断言，验证 jacobi(n, a, b, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(jacobi(n, a, b, x)**2) == \
        r'\left(P_{n}^{\left(a,b\right)}\left(x\right)\right)^{2}'

    # 断言，验证 gegenbauer(n, a, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(gegenbauer(n, a, x)) == \
        r'C_{n}^{\left(a\right)}\left(x\right)'

    # 断言，验证 gegenbauer(n, a, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(gegenbauer(n, a, x)**2) == \
        r'\left(C_{n}^{\left(a\right)}\left(x\right)\right)^{2}'

    # 断言，验证 chebyshevt(n, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(chebyshevt(n, x)) == r'T_{n}\left(x\right)'

    # 断言，验证 chebyshevt(n, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(chebyshevt(n, x)**2) == \
        r'\left(T_{n}\left(x\right)\right)^{2}'

    # 断言，验证 chebyshevu(n, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(chebyshevu(n, x)) == r'U_{n}\left(x\right)'

    # 断言，验证 chebyshevu(n, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(chebyshevu(n, x)**2) == \
        r'\left(U_{n}\left(x\right)\right)^{2}'

    # 断言，验证 legendre(n, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(legendre(n, x)) == r'P_{n}\left(x\right)'

    # 断言，验证 legendre(n, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(legendre(n, x)**2) == r'\left(P_{n}\left(x\right)\right)^{2}'

    # 断言，验证 assoc_legendre(n, a, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(assoc_legendre(n, a, x)) == \
        r'P_{n}^{\left(a\right)}\left(x\right)'

    # 断言，验证 assoc_legendre(n, a, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(assoc_legendre(n, a, x)**2) == \
        r'\left(P_{n}^{\left(a\right)}\left(x\right)\right)^{2}'

    # 断言，验证 laguerre(n, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(laguerre(n, x)) == r'L_{n}\left(x\right)'

    # 断言，验证 laguerre(n, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(laguerre(n, x)**2) == r'\left(L_{n}\left(x\right)\right)^{2}'

    # 断言，验证 assoc_laguerre(n, a, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(assoc_laguerre(n, a, x)) == \
        r'L_{n}^{\left(a\right)}\left(x\right)'

    # 断言，验证 assoc_laguerre(n, a, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(assoc_laguerre(n, a, x)**2) == \
        r'\left(L_{n}^{\left(a\right)}\left(x\right)\right)^{2}'

    # 断言，验证 hermite(n, x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(hermite(n, x)) == r'H_{n}\left(x\right)'

    # 断言，验证 hermite(n, x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(hermite(n, x)**2) == r'\left(H_{n}\left(x\right)\right)^{2}'

    # 定义符号变量 theta，并指定其为实数
    theta = Symbol("theta", real=True)

    # 定义符号变量 phi，并指定其为实数
    phi = Symbol("phi", real=True)

    # 断言，验证 Ynm(n, m, theta, phi) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Ynm(n, m, theta, phi)) == r'Y_{n}^{m}\left(\theta,\phi\right)'

    # 断言，验证 Ynm(n, m, theta, phi)**3 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Ynm(n, m, theta, phi)**3) == \
        r'\left(Y_{n}^{m}\left(\theta,\phi\right)\right)^{3}'

    # 断言，验证 Znm(n, m, theta, phi) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Znm(n, m, theta, phi)) == r'Z_{n}^{m}\left(\theta,\phi\right)'

    # 断言，验证 Znm(n, m, theta, phi)**3 的 LaTeX 表达式是否等于给定的字符串
    assert latex(Znm(n, m, theta, phi)**3) == \
        r'\left(Z_{n}^{m}\left(\theta,\phi\right)\right)^{3}'

    # 断言，验证 polar_lift(0) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(polar_lift(0)) == \
        r"\operatorname{polar\_lift}{\left(0 \right)}"

    # 断言，验证 polar_lift(0)**3 的 LaTeX 表达式是否等于给定的字符串
    assert latex(polar_lift(0)**3) == \
        r"\operatorname{polar\_lift}^{3}{\left(0 \right)}"

    # 断言，验证 totient(n) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(totient(n)) == r'\phi\left(n\right)'

    # 断言，验证 totient(n) ** 2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(totient(n) ** 2) == r'\left(\phi\left(n\right)\right)^{2}'

    # 断言，验证 reduced_totient(n) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(reduced_totient(n)) == r'\lambda\left(n\right)'

    # 断言，验证 reduced_totient(n) ** 2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(reduced_totient(n) ** 2) == \
        r'\left(\lambda\left(n\right)\right)^{2}'

    # 断言，验证 divisor_sigma(x) 的 LaTeX 表达式是否等于给定的字符串
    assert latex(divisor_sigma(x)) == r"\sigma\left(x\right)"

    # 断言，验证 divisor_sigma(x)**2 的 LaTeX 表达式是否等于给定的字符串
    assert latex(divisor_sigma(x)**2) == r"\sigma^{2}\left(x
    # 断言：验证 divisor_sigma 函数返回的 LaTeX 表达式是否等于给定的字符串
    assert latex(divisor_sigma(x, y)**2) == r"\sigma^{2}_y\left(x\right)"

    # 断言：验证 udivisor_sigma 函数返回的 LaTeX 表达式是否等于给定的字符串
    assert latex(udivisor_sigma(x)) == r"\sigma^*\left(x\right)"
    assert latex(udivisor_sigma(x)**2) == r"\sigma^*^{2}\left(x\right)"
    assert latex(udivisor_sigma(x, y)) == r"\sigma^*_y\left(x\right)"
    assert latex(udivisor_sigma(x, y)**2) == r"\sigma^*^{2}_y\left(x\right)"

    # 断言：验证 primenu 函数返回的 LaTeX 表达式是否等于给定的字符串
    assert latex(primenu(n)) == r'\nu\left(n\right)'
    assert latex(primenu(n) ** 2) == r'\left(\nu\left(n\right)\right)^{2}'

    # 断言：验证 primeomega 函数返回的 LaTeX 表达式是否等于给定的字符串
    assert latex(primeomega(n)) == r'\Omega\left(n\right)'
    assert latex(primeomega(n) ** 2) == \
        r'\left(\Omega\left(n\right)\right)^{2}'

    # 断言：验证 LambertW 函数返回的 LaTeX 表达式是否等于给定的字符串
    assert latex(LambertW(n)) == r'W\left(n\right)'
    assert latex(LambertW(n, -1)) == r'W_{-1}\left(n\right)'
    assert latex(LambertW(n, k)) == r'W_{k}\left(n\right)'
    assert latex(LambertW(n) * LambertW(n)) == r"W^{2}\left(n\right)"
    assert latex(Pow(LambertW(n), 2)) == r"W^{2}\left(n\right)"
    assert latex(LambertW(n)**k) == r"W^{k}\left(n\right)"
    assert latex(LambertW(n, k)**p) == r"W^{p}_{k}\left(n\right)"

    # 断言：验证 Mod 函数返回的 LaTeX 表达式是否等于给定的字符串
    assert latex(Mod(x, 7)) == r'x \bmod 7'
    assert latex(Mod(x + 1, 7)) == r'\left(x + 1\right) \bmod 7'
    assert latex(Mod(7, x + 1)) == r'7 \bmod \left(x + 1\right)'
    assert latex(Mod(2 * x, 7)) == r'2 x \bmod 7'
    assert latex(Mod(7, 2 * x)) == r'7 \bmod 2 x'
    assert latex(Mod(x, 7) + 1) == r'\left(x \bmod 7\right) + 1'
    assert latex(2 * Mod(x, 7)) == r'2 \left(x \bmod 7\right)'
    assert latex(Mod(7, 2 * x)**n) == r'\left(7 \bmod 2 x\right)^{n}'

    # 未知函数名 fjlkd 应该用 \operatorname 渲染
    fjlkd = Function('fjlkd')
    assert latex(fjlkd(x)) == r'\operatorname{fjlkd}{\left(x \right)}'
    # 即使在没有参数的情况下引用时也应该使用 \operatorname
    assert latex(fjlkd) == r'\operatorname{fjlkd}'
# 测试确保符号名称只传递给同名的子类
def test_function_subclass_different_name():
    # 定义名为mygamma的gamma子类
    class mygamma(gamma):
        pass
    # 断言latex函数对mygamma的输出为"\operatorname{mygamma}"
    assert latex(mygamma) == r"\operatorname{mygamma}"
    # 断言latex函数对mygamma(x)的输出为"\operatorname{mygamma}{\left(x \right)}"
    assert latex(mygamma(x)) == r"\operatorname{mygamma}{\left(x \right)}"


def test_hyper_printing():
    from sympy.abc import x, z

    # 断言latex函数对meijerg函数的输出
    assert latex(meijerg(Tuple(pi, pi, x), Tuple(1),
                         (0, 1), Tuple(1, 2, 3/pi), z)) == \
        r'{G_{4, 5}^{2, 3}\left(\begin{matrix} \pi, \pi, x & 1 \\0, 1 & 1, 2, '\
        r'\frac{3}{\pi} \end{matrix} \middle| {z} \right)}'
    # 断言latex函数对meijerg函数的输出
    assert latex(meijerg(Tuple(), Tuple(1), (0,), Tuple(), z)) == \
        r'{G_{1, 1}^{1, 0}\left(\begin{matrix}  & 1 \\0 &  \end{matrix} \middle| {z} \right)}'
    # 断言latex函数对hyper函数的输出
    assert latex(hyper((x, 2), (3,), z)) == \
        r'{{}_{2}F_{1}\left(\begin{matrix} 2, x ' \
        r'\\ 3 \end{matrix}\middle| {z} \right)}'
    # 断言latex函数对hyper函数的输出
    assert latex(hyper(Tuple(), Tuple(1), z)) == \
        r'{{}_{0}F_{1}\left(\begin{matrix}  ' \
        r'\\ 1 \end{matrix}\middle| {z} \right)}'


def test_latex_bessel():
    from sympy.functions.special.bessel import (besselj, bessely, besseli,
                                                besselk, hankel1, hankel2,
                                                jn, yn, hn1, hn2)
    from sympy.abc import z
    # 断言latex函数对besselj函数的输出
    assert latex(besselj(n, z**2)**k) == r'J^{k}_{n}\left(z^{2}\right)'
    # 断言latex函数对bessely函数的输出
    assert latex(bessely(n, z)) == r'Y_{n}\left(z\right)'
    # 断言latex函数对besseli函数的输出
    assert latex(besseli(n, z)) == r'I_{n}\left(z\right)'
    # 断言latex函数对besselk函数的输出
    assert latex(besselk(n, z)) == r'K_{n}\left(z\right)'
    # 断言latex函数对hankel1函数的输出
    assert latex(hankel1(n, z**2)**2) == \
        r'\left(H^{(1)}_{n}\left(z^{2}\right)\right)^{2}'
    # 断言latex函数对hankel2函数的输出
    assert latex(hankel2(n, z)) == r'H^{(2)}_{n}\left(z\right)'
    # 断言latex函数对jn函数的输出
    assert latex(jn(n, z)) == r'j_{n}\left(z\right)'
    # 断言latex函数对yn函数的输出
    assert latex(yn(n, z)) == r'y_{n}\left(z\right)'
    # 断言latex函数对hn1函数的输出
    assert latex(hn1(n, z)) == r'h^{(1)}_{n}\left(z\right)'
    # 断言latex函数对hn2函数的输出
    assert latex(hn2(n, z)) == r'h^{(2)}_{n}\left(z\right)'


def test_latex_fresnel():
    from sympy.functions.special.error_functions import (fresnels, fresnelc)
    from sympy.abc import z
    # 断言latex函数对fresnels函数的输出
    assert latex(fresnels(z)) == r'S\left(z\right)'
    # 断言latex函数对fresnelc函数的输出
    assert latex(fresnelc(z)) == r'C\left(z\right)'
    # 断言latex函数对fresnels函数的输出
    assert latex(fresnels(z)**2) == r'S^{2}\left(z\right)'
    # 断言latex函数对fresnelc函数的输出
    assert latex(fresnelc(z)**2) == r'C^{2}\left(z\right)'


def test_latex_brackets():
    # 断言latex函数对(-1)**x的输出
    assert latex((-1)**x) == r"\left(-1\right)^{x}"


def test_latex_indexed():
    # 创建复数符号Psi_symbol和索引基础Psi_indexed
    Psi_symbol = Symbol('Psi_0', complex=True, real=False)
    Psi_indexed = IndexedBase(Symbol('Psi', complex=True, real=False))
    # 测试latex函数对Psi_symbol * conjugate(Psi_symbol)的输出
    symbol_latex = latex(Psi_symbol * conjugate(Psi_symbol))
    # 测试latex函数对Psi_indexed[0] * conjugate(Psi_indexed[0])的输出
    indexed_latex = latex(Psi_indexed[0] * conjugate(Psi_indexed[0]))
    # 断言symbol_latex为'\Psi_{0} \overline{\Psi_{0}}'
    assert symbol_latex == r'\Psi_{0} \overline{\Psi_{0}}'
    # 断言indexed_latex为'\overline{{\Psi}_{0}} {\Psi}_{0}'
    assert indexed_latex == r'\overline{{\Psi}_{0}} {\Psi}_{0}'

    # Symbol('gamma')给出'\gamma'
    # 定义interval为'\\mathrel{..}\\nobreak '
    interval = '\\mathrel{..}\\nobreak '
    # 断言：将 Indexed('x1', Symbol('i')) 转换为 LaTeX 表示形式，并断言其结果是否等于 '{x_{1}}_{i}'
    assert latex(Indexed('x1', Symbol('i'))) == r'{x_{1}}_{i}'
    
    # 断言：将 Indexed('x2', Idx('i')) 转换为 LaTeX 表示形式，并断言其结果是否等于 '{x_{2}}_{i}'
    assert latex(Indexed('x2', Idx('i'))) == r'{x_{2}}_{i}'
    
    # 断言：将 Indexed('x3', Idx('i', Symbol('N'))) 转换为 LaTeX 表示形式，并断言其结果是否等于 '{x_{3}}_{{i}_{0'+interval+'N - 1}}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')))) == r'{x_{3}}_{{i}_{0'+interval+'N - 1}}'
    
    # 断言：将 Indexed('x3', Idx('i', Symbol('N')+1)) 转换为 LaTeX 表示形式，并断言其结果是否等于 '{x_{3}}_{{i}_{0'+interval+'N}}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')+1))) == r'{x_{3}}_{{i}_{0'+interval+'N}}'
    
    # 断言：将 Indexed('x4', Idx('i', (Symbol('a'),Symbol('b')))) 转换为 LaTeX 表示形式，并断言其结果是否等于 '{x_{4}}_{{i}_{a'+interval+'b}}'
    assert latex(Indexed('x4', Idx('i', (Symbol('a'),Symbol('b'))))) == r'{x_{4}}_{{i}_{a'+interval+'b}}'
    
    # 断言：将 IndexedBase('gamma') 转换为 LaTeX 表示形式，并断言其结果是否等于 '\gamma'
    assert latex(IndexedBase('gamma')) == r'\gamma'
    
    # 断言：将 IndexedBase('a b') 转换为 LaTeX 表示形式，并断言其结果是否等于 'a b'
    assert latex(IndexedBase('a b')) == r'a b'
    
    # 断言：将 IndexedBase('a_b') 转换为 LaTeX 表示形式，并断言其结果是否等于 'a_{b}'
    assert latex(IndexedBase('a_b')) == r'a_{b}'
def test_latex_derivatives():
    # 普通导数用 "d"
    assert latex(diff(x**3, x, evaluate=False)) == \
        r"\frac{d}{d x} x^{3}"
    assert latex(diff(sin(x) + x**2, x, evaluate=False)) == \
        r"\frac{d}{d x} \left(x^{2} + \sin{\left(x \right)}\right)"
    assert latex(diff(diff(sin(x) + x**2, x, evaluate=False), evaluate=False))\
        == \
        r"\frac{d^{2}}{d x^{2}} \left(x^{2} + \sin{\left(x \right)}\right)"
    assert latex(diff(diff(diff(sin(x) + x**2, x, evaluate=False), evaluate=False), evaluate=False)) == \
        r"\frac{d^{3}}{d x^{3}} \left(x^{2} + \sin{\left(x \right)}\right)"

    # 偏导数用 "\partial"
    assert latex(diff(sin(x * y), x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \sin{\left(x y \right)}"
    assert latex(diff(sin(x * y) + x**2, x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \left(x^{2} + \sin{\left(x y \right)}\right)"
    assert latex(diff(diff(sin(x*y) + x**2, x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial x^{2}} \left(x^{2} + \sin{\left(x y \right)}\right)"
    assert latex(diff(diff(diff(sin(x*y) + x**2, x, evaluate=False), x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial x^{3}} \left(x^{2} + \sin{\left(x y \right)}\right)"

    # 混合偏导数
    f = Function("f")
    assert latex(diff(diff(f(x, y), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial y\partial x} " + latex(f(x, y))

    assert latex(diff(diff(diff(f(x, y), x, evaluate=False), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial y\partial x^{2}} " + latex(f(x, y))

    # 负数嵌套导数
    assert latex(diff(-diff(y**2,x,evaluate=False),x,evaluate=False)) == r'\frac{d}{d x} \left(- \frac{d}{d x} y^{2}\right)'
    assert latex(diff(diff(-diff(diff(y,x,evaluate=False),x,evaluate=False),x,evaluate=False),x,evaluate=False)) == \
        r'\frac{d^{2}}{d x^{2}} \left(- \frac{d^{2}}{d x^{2}} y\right)'

    # 一个变量被积分后用普通 "d"
    assert latex(diff(Integral(exp(-x*y), (x, 0, oo)), y, evaluate=False)) == \
        r"\frac{d}{d y} \int\limits_{0}^{\infty} e^{- x y}\, dx"

    # 导数包裹在幂次中
    assert latex(diff(x, x, evaluate=False)**2) == \
        r"\left(\frac{d}{d x} x\right)^{2}"

    assert latex(diff(f(x), x)**2) == \
        r"\left(\frac{d}{d x} f{\left(x \right)}\right)^{2}"

    assert latex(diff(f(x), (x, n))) == \
        r"\frac{d^{n}}{d x^{n}} f{\left(x \right)}"

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    assert latex(diff(f(x1, x2), x1)) == r'\frac{\partial}{\partial x_{1}} f{\left(x_{1},x_{2} \right)}'

    n1 = Symbol('n1')
    assert latex(diff(f(x), (x, n1))) == r'\frac{d^{n_{1}}}{d x^{n_{1}}} f{\left(x \right)}'

    n2 = Symbol('n2')
    # 使用 SymPy 库中的 diff 函数计算 f(x) 对 x 的导数，然后生成 LaTeX 表达式
    assert latex(diff(f(x), (x, Max(n1, n2)))) == \
        r'\frac{d^{\max\left(n_{1}, n_{2}\right)}}{d x^{\max\left(n_{1}, n_{2}\right)}} f{\left(x \right)}'

    # 设置 diff 函数的 diff_operator 参数为 "rd"，生成不同形式的导数 LaTeX 表达式
    assert latex(diff(f(x), x), diff_operator="rd") == r'\frac{\mathrm{d}}{\mathrm{d} x} f{\left(x \right)}'
#`
# 定义测试函数，测试 SymPy 中 latex 函数对不同数学表达式的输出是否符合预期
def test_latex_subs():
    # 断言表达式 Subs(x*y, (x, y), (1, 2)) 转换为 LaTeX 格式是否正确
    assert latex(Subs(x*y, (x, y), (1, 2))) == r'\left. x y \right|_{\substack{ x=1\\ y=2 }}'


# 定义测试函数，测试 SymPy 中 latex 函数对积分表达式的输出是否符合预期
def test_latex_integrals():
    # 断言对单变量积分 Integral(log(x), x) 转换为 LaTeX 格式是否正确
    assert latex(Integral(log(x), x)) == r"\int \log{\left(x \right)}\, dx"
    # 断言对带上下限的积分 Integral(x**2, (x, 0, 1)) 转换为 LaTeX 格式是否正确
    assert latex(Integral(x**2, (x, 0, 1))) == \
        r"\int\limits_{0}^{1} x^{2}\, dx"
    # 断言对带变量和上下限的积分 Integral(x**2, (x, 10, 20)) 转换为 LaTeX 格式是否正确
    assert latex(Integral(x**2, (x, 10, 20))) == \
        r"\int\limits_{10}^{20} x^{2}\, dx"
    # 断言对双重积分 Integral(y*x**2, (x, 0, 1), y) 转换为 LaTeX 格式是否正确
    assert latex(Integral(y*x**2, (x, 0, 1), y)) == \
        r"\int\int\limits_{0}^{1} x^{2} y\, dx\, dy"
    # 断言对双重积分 Integral(y*x**2, (x, 0, 1), y) 以 equation* 模式输出是否正确
    assert latex(Integral(y*x**2, (x, 0, 1), y), mode='equation*') == \
        r"\begin{equation*}\int\int\limits_{0}^{1} x^{2} y\, dx\, dy\end{equation*}"
    # 断言对双重积分 Integral(y*x**2, (x, 0, 1), y) 以 equation* 模式并启用 itex 输出是否正确
    assert latex(Integral(y*x**2, (x, 0, 1), y), mode='equation*', itex=True) \
        == r"$$\int\int_{0}^{1} x^{2} y\, dx\, dy$$"
    # 断言对多重积分的 LaTeX 输出是否正确
    assert latex(Integral(x*y*z, x, y, z)) == r"\iiint x y z\, dx\, dy\, dz"
    # 断言对更高阶的多重积分的 LaTeX 输出是否正确
    assert latex(Integral(x*y*z*t, x, y, z, t)) == \
        r"\iiiint t x y z\, dx\, dy\, dz\, dt"
    # 断言对超过四次积分的 LaTeX 输出是否正确
    assert latex(Integral(x, x, x, x, x, x, x)) == \
        r"\int\int\int\int\int\int x\, dx\, dx\, dx\, dx\, dx\, dx"
    # 断言对混合积分和不定积分的 LaTeX 输出是否正确
    assert latex(Integral(x, x, y, (z, 0, 1))) == \
        r"\int\limits_{0}^{1}\int\int x\, dx\, dy\, dz"

    # 对负数嵌套积分的 LaTeX 输出进行断言
    assert latex(Integral(-Integral(y**2,x),x)) == \
        r'\int \left(- \int y^{2}\, dx\right)\, dx'
    assert latex(Integral(-Integral(-Integral(y,x),x),x)) == \
        r'\int \left(- \int \left(- \int y\, dx\right)\, dx\right)\, dx'

    # 修复问题 #10806 后的积分 LaTeX 输出是否正确
    assert latex(Integral(z, z)**2) == r"\left(\int z\, dz\right)^{2}"
    assert latex(Integral(x + z, z)) == r"\int \left(x + z\right)\, dz"
    assert latex(Integral(x+z/2, z)) == \
        r"\int \left(x + \frac{z}{2}\right)\, dz"
    assert latex(Integral(x**y, z)) == r"\int x^{y}\, dz"

    # 设置不同的微分操作符后的积分 LaTeX 输出是否正确
    assert latex(Integral(x, x), diff_operator="rd") == r'\int x\, \mathrm{d}x'
    assert latex(Integral(x, (x, 0, 1)), diff_operator="rd") == r'\int\limits_{0}^{1} x\, \mathrm{d}x'


# 定义测试函数，测试 SymPy 中 latex 函数对集合的输出是否符合预期
def test_latex_sets():
    # 对 frozenset 和 set 类型的集合转换为 LaTeX 输出进行断言
    for s in (frozenset, set):
        assert latex(s([x*y, x**2])) == r"\left\{x^{2}, x y\right\}"
        assert latex(s(range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
        assert latex(s(range(1, 13))) == \
            r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"

    # 对 FiniteSet 类型的集合转换为 LaTeX 输出进行断言
    s = FiniteSet
    assert latex(s(*[x*y, x**2])) == r"\left\{x^{2}, x y\right\}"
    assert latex(s(*range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
    assert latex(s(*range(1, 13))) == \
        r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"


# 定义测试函数，测试 SymPy 中 latex 函数对 SetExpr 的输出是否符合预期
def test_latex_SetExpr():
    # 创建 Interval(1, 3) 的 SetExpr 对象并进行 LaTeX 输出断言
    iv = Interval(1, 3)
    se = SetExpr(iv)
    assert latex(se) == r"SetExpr\left(\left[1, 3\right]\right)"


# 定义测试函数，测试 SymPy 中 latex 函数对 Range 类的输出是否符合预期
def test_latex_Range():
    # 断言对 Range(1, 51) 转换为 LaTeX 输出是否符合预期
    assert latex(Range(1, 51)) == r'\left\{1, 2, \ldots, 50\right\}'
    # 断言：验证 Range 函数生成的 LaTeX 输出是否与预期的数学表达式格式一致
    assert latex(Range(1, 4)) == r'\left\{1, 2, 3\right\}'
    assert latex(Range(0, 3, 1)) == r'\left\{0, 1, 2\right\}'
    assert latex(Range(0, 30, 1)) == r'\left\{0, 1, \ldots, 29\right\}'
    assert latex(Range(30, 1, -1)) == r'\left\{30, 29, \ldots, 2\right\}'
    assert latex(Range(0, oo, 2)) == r'\left\{0, 2, \ldots\right\}'
    assert latex(Range(oo, -2, -2)) == r'\left\{\ldots, 2, 0\right\}'
    assert latex(Range(-2, -oo, -1)) == r'\left\{-2, -3, \ldots\right\}'
    assert latex(Range(-oo, oo)) == r'\left\{\ldots, -1, 0, 1, \ldots\right\}'
    assert latex(Range(oo, -oo, -1)) == r'\left\{\ldots, 1, 0, -1, \ldots\right\}'

    # 创建符号变量 a, b, c
    a, b, c = symbols('a:c')
    assert latex(Range(a, b, c)) == r'\text{Range}\left(a, b, c\right)'
    assert latex(Range(a, 10, 1)) == r'\text{Range}\left(a, 10\right)'
    assert latex(Range(0, b, 1)) == r'\text{Range}\left(b\right)'
    assert latex(Range(0, 10, c)) == r'\text{Range}\left(0, 10, c\right)'

    # 创建整数符号变量 i, n, p，分别带有整数、负数、正数属性
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)

    # 验证 Range 函数生成的 LaTeX 输出是否与预期的数学表达式格式一致
    assert latex(Range(i, i + 3)) == r'\left\{i, i + 1, i + 2\right\}'
    assert latex(Range(-oo, n, 2)) == r'\left\{\ldots, n - 4, n - 2\right\}'
    assert latex(Range(p, oo)) == r'\left\{p, p + 1, \ldots\right\}'
    # 下面的断言需要 __iter__ 函数改进才能正常工作
    # assert latex(Range(-3, p + 7)) == r'\left\{-3, -2,  \ldots, p + 6\right\}'
    # 必须具有整数假设
    assert latex(Range(a, a + 3)) == r'\text{Range}\left(a, a + 3\right)'
# 定义用于测试 LaTeX 表示的函数
def test_latex_sequences():
    # 创建 SeqFormula 对象 s1，表示平方数序列 a**2 (0 到 oo)
    s1 = SeqFormula(a**2, (0, oo))
    # 创建 SeqPer 对象 s2，表示周期序列 (1, 2)
    s2 = SeqPer((1, 2))

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[0, 1, 4, 9, \ldots\right]'
    # 断言 s1 的 LaTeX 表示等于预期字符串
    assert latex(s1) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[1, 2, 1, 2, \ldots\right]'
    # 断言 s2 的 LaTeX 表示等于预期字符串
    assert latex(s2) == latex_str

    # 创建 SeqFormula 对象 s3，表示平方数序列 a**2 (0 到 2)
    s3 = SeqFormula(a**2, (0, 2))
    # 创建 SeqPer 对象 s4，表示周期序列 (1, 2) (0 到 2)
    s4 = SeqPer((1, 2), (0, 2))

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[0, 1, 4\right]'
    # 断言 s3 的 LaTeX 表示等于预期字符串
    assert latex(s3) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[1, 2, 1\right]'
    # 断言 s4 的 LaTeX 表示等于预期字符串
    assert latex(s4) == latex_str

    # 创建 SeqFormula 对象 s5，表示平方数序列 a**2 (-oo 到 0)
    s5 = SeqFormula(a**2, (-oo, 0))
    # 创建 SeqPer 对象 s6，表示周期序列 (1, 2) (-oo 到 0)
    s6 = SeqPer((1, 2), (-oo, 0))

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[\ldots, 9, 4, 1, 0\right]'
    # 断言 s5 的 LaTeX 表示等于预期字符串
    assert latex(s5) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[\ldots, 2, 1, 2, 1\right]'
    # 断言 s6 的 LaTeX 表示等于预期字符串
    assert latex(s6) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[1, 3, 5, 11, \ldots\right]'
    # 断言 SeqAdd(s1, s2) 的 LaTeX 表示等于预期字符串
    assert latex(SeqAdd(s1, s2)) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[1, 3, 5\right]'
    # 断言 SeqAdd(s3, s4) 的 LaTeX 表示等于预期字符串
    assert latex(SeqAdd(s3, s4)) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[\ldots, 11, 5, 3, 1\right]'
    # 断言 SeqAdd(s5, s6) 的 LaTeX 表示等于预期字符串
    assert latex(SeqAdd(s5, s6)) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[0, 2, 4, 18, \ldots\right]'
    # 断言 SeqMul(s1, s2) 的 LaTeX 表示等于预期字符串
    assert latex(SeqMul(s1, s2)) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[0, 2, 4\right]'
    # 断言 SeqMul(s3, s4) 的 LaTeX 表示等于预期字符串
    assert latex(SeqMul(s3, s4)) == latex_str

    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[\ldots, 18, 4, 2, 0\right]'
    # 断言 SeqMul(s5, s6) 的 LaTeX 表示等于预期字符串
    assert latex(SeqMul(s5, s6)) == latex_str

    # 创建 SeqFormula 对象 s7，表示平方数序列 a**2 (a 从 0 到 x)
    s7 = SeqFormula(a**2, (a, 0, x))
    # 预期的 LaTeX 字符串表示
    latex_str = r'\left\{a^{2}\right\}_{a=0}^{x}'
    # 断言 s7 的 LaTeX 表示等于预期字符串
    assert latex(s7) == latex_str

    # 创建符号 b
    b = Symbol('b')
    # 创建 SeqFormula 对象 s8，表示 b*a**2 (a 从 0 到 2)
    s8 = SeqFormula(b*a**2, (a, 0, 2))
    # 预期的 LaTeX 字符串表示
    latex_str = r'\left[0, b, 4 b\right]'
    # 断言 s8 的 LaTeX 表示等于预期字符串
    assert latex(s8) == latex_str


# 测试 FourierSeries 的 LaTeX 表示
def test_latex_FourierSeries():
    # 预期的 LaTeX 字符串表示
    latex_str = r'2 \sin{\left(x \right)} - \sin{\left(2 x \right)} + \frac{2 \sin{\left(3 x \right)}}{3} + \ldots'
    # 断言 FourierSeries 的 LaTeX 表示等于预期字符串
    assert latex(fourier_series(x, (x, -pi, pi))) == latex_str


# 测试 FormalPowerSeries 的 LaTeX 表示
def test_latex_FormalPowerSeries():
    # 预期的 LaTeX 字符串表示
    latex_str = r'\sum_{k=1}^{\infty} - \frac{\left(-1\right)^{- k} x^{k}}{k}'
    # 断言 FormalPowerSeries 的 LaTeX 表示等于预期字符串
    assert latex(fps(log(1 + x))) == latex_str


# 测试 Interval 对象的 LaTeX 表示
def test_latex_intervals():
    # 创建符号 a，限定为实数
    a = Symbol('a', real=True)
    # 断言空集的 LaTeX 表示
    assert latex(Interval(0, 0)) == r"\left\{0\right\}"
    # 断言闭区间 [0, a] 的 LaTeX 表示
    assert latex(Interval(0, a)) == r"\left[0, a\right]"
    # 断言半开区间 (0, a] 的 LaTeX 表示
    assert latex(Interval(0, a, True, False)) == r"\left(0, a\right]"
    # 断言左开右闭区间 [0, a) 的 LaTeX 表示
    assert latex(Interval(0, a, False, True)) == r"\left[0, a\right)"
    # 断言开区间 (0, a) 的 LaTeX 表示
    assert latex(Interval(0, a, True, True)) == r"\left(0, a\right)"


# 测试 AccumBounds 对象的 LaTeX 表示
def test_latex_AccumuBounds():
    # 创建符号 a，限定为实数
    a = Symbol('a', real=True)
    # 断言 AccumBounds 对象 (0, 1) 的 LaTeX 表示
    assert latex(AccumBounds(0, 1)) == r"\left\langle 0, 1\right\rangle"
    # 断言 AccumBounds 对象 (0, a) 的 LaTeX 表示
    assert latex(AccumBounds(0, a)) == r"\left\langle 0, a\right\rangle"
    # 断言 AccumBounds 对象 (a+1, a+2) 的 LaTeX 表示
    assert latex(AccumBounds(a + 1, a + 2)) == r"\left\langle a + 1, a + 2\right\rangle"

# 测试空集的 LaTeX 表示
def test_latex_emptyset():
    assert latex(S.EmptySet) == r"\emptyset"

# 测试全集
    # 使用 Commutator 类创建一个交换子对象，其中参数 B 和 A 表示交换子的运算对象顺序
    comm = Commutator(B, A)
    
    # 断言：确保对交换子对象调用 doit() 方法后得到的 LaTeX 表示字符串与指定的负交换子表达式匹配
    assert latex(comm.doit()) == r"- (A B - B A)"
def test_set_operators_parenthesis():
    a, b, c, d = symbols('a:d')
    A = FiniteSet(a)  # 创建集合 A，包含元素 a
    B = FiniteSet(b)  # 创建集合 B，包含元素 b
    C = FiniteSet(c)  # 创建集合 C，包含元素 c
    D = FiniteSet(d)  # 创建集合 D，包含元素 d

    U1 = Union(A, B, evaluate=False)  # 计算集合 A 和 B 的并集，不进行求值
    U2 = Union(C, D, evaluate=False)  # 计算集合 C 和 D 的并集，不进行求值
    I1 = Intersection(A, B, evaluate=False)  # 计算集合 A 和 B 的交集，不进行求值
    I2 = Intersection(C, D, evaluate=False)  # 计算集合 C 和 D 的交集，不进行求值
    C1 = Complement(A, B, evaluate=False)  # 计算集合 A 对 B 的补集，不进行求值
    C2 = Complement(C, D, evaluate=False)  # 计算集合 C 对 D 的补集，不进行求值
    D1 = SymmetricDifference(A, B, evaluate=False)  # 计算集合 A 和 B 的对称差，不进行求值
    D2 = SymmetricDifference(C, D, evaluate=False)  # 计算集合 C 和 D 的对称差，不进行求值
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(A, B)  # 创建集合 A 和 B 的笛卡尔积
    P2 = ProductSet(C, D)  # 创建集合 C 和 D 的笛卡尔积

    assert latex(Intersection(A, U2, evaluate=False)) == \
        r'\left\{a\right\} \cap ' \
        r'\left(\left\{c\right\} \cup \left\{d\right\}\right)'  # 计算集合 A 与集合 U2 的交集的 LaTeX 表示
    assert latex(Intersection(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\cap \left(\left\{c\right\} \cup \left\{d\right\}\right)'  # 计算集合 U1 和集合 U2 的交集的 LaTeX 表示
    assert latex(Intersection(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \cap \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'  # 计算集合 C1 和集合 C2 的交集的 LaTeX 表示
    assert latex(Intersection(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \cap \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'  # 计算集合 D1 和集合 D2 的交集的 LaTeX 表示
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Intersection(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\cap \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Union(A, I2, evaluate=False)) == \
        r'\left\{a\right\} \cup ' \
        r'\left(\left\{c\right\} \cap \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Union(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\cup \left(\left\{c\right\} \cap \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Union(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \cup \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Union(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \cup \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Union(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\cup \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Complement(A, C2, evaluate=False)) == \
        r'\left\{a\right\} \setminus \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Complement(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Complement(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Complement(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \setminus ' \
        r'\left(\left\{c\right\} \triangle \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(Complement(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(SymmetricDifference(A, D2, evaluate=False)) == \
        r'\left\{a\right\} \triangle \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(SymmetricDifference(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    
    # 断言：检查两个集合操作的 LaTeX 表示是否等于特定字符串
    assert latex(SymmetricDifference(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    # 断言：验证对称差集操作后的 LaTeX 表示是否与预期相符
    assert latex(SymmetricDifference(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \triangle ' \
        r'\left(\left\{c\right\} \setminus \left\{d\right\}\right)'
    
    # 断言：验证对称差集操作后的 LaTeX 表示是否与预期相符
    assert latex(SymmetricDifference(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'
    
    # XXX 这里可能不正确，因为笛卡尔积不满足结合律
    assert latex(ProductSet(A, P2).flatten()) == \
        r'\left\{a\right\} \times \left\{c\right\} \times ' \
        r'\left\{d\right\}'
    
    # 断言：验证并集操作后的 LaTeX 表示是否与预期相符
    assert latex(ProductSet(U1, U2)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\times \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    
    # 断言：验证交集操作后的 LaTeX 表示是否与预期相符
    assert latex(ProductSet(I1, I2)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\times \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    
    # 断言：验证差集操作后的 LaTeX 表示是否与预期相符
    assert latex(ProductSet(C1, C2)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \times \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    
    # 断言：验证对称差集操作后的 LaTeX 表示是否与预期相符
    assert latex(ProductSet(D1, D2)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \times \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
# 测试 latex 函数对 SymPy 符号 S.Complexes 的输出是否正确
def test_latex_Complexes():
    assert latex(S.Complexes) == r"\mathbb{C}"

# 测试 latex 函数对 SymPy 符号 S.Naturals 的输出是否正确
def test_latex_Naturals():
    assert latex(S.Naturals) == r"\mathbb{N}"

# 测试 latex 函数对 SymPy 符号 S.Naturals0 的输出是否正确
def test_latex_Naturals0():
    assert latex(S.Naturals0) == r"\mathbb{N}_0"

# 测试 latex 函数对 SymPy 符号 S.Integers 的输出是否正确
def test_latex_Integers():
    assert latex(S.Integers) == r"\mathbb{Z}"

# 测试 latex 函数对 ImageSet 类的输出是否正确
def test_latex_ImageSet():
    x = Symbol('x')
    # 检查 ImageSet 函数对 Lambda 表达式 x**2, S.Naturals 的输出是否正确
    assert latex(ImageSet(Lambda(x, x**2), S.Naturals)) == \
        r"\left\{x^{2}\; \middle|\; x \in \mathbb{N}\right\}"

    y = Symbol('y')
    # 创建一个新的 ImageSet 对象，并检查其 latex 输出是否正确
    imgset = ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {3, 4})
    assert latex(imgset) == \
        r"\left\{x + y\; \middle|\; x \in \left\{1, 2, 3\right\}, y \in \left\{3, 4\right\}\right\}"

    # 创建另一个 ImageSet 对象，并检查其 latex 输出是否正确
    imgset = ImageSet(Lambda(((x, y),), x + y), ProductSet({1, 2, 3}, {3, 4}))
    assert latex(imgset) == \
        r"\left\{x + y\; \middle|\; \left( x, \  y\right) \in \left\{1, 2, 3\right\} \times \left\{3, 4\right\}\right\}"

# 测试 latex 函数对 ConditionSet 类的输出是否正确
def test_latex_ConditionSet():
    x = Symbol('x')
    # 检查 ConditionSet 函数对 x**2 = 1, S.Reals 的输出是否正确
    assert latex(ConditionSet(x, Eq(x**2, 1), S.Reals)) == \
        r"\left\{x\; \middle|\; x \in \mathbb{R} \wedge x^{2} = 1 \right\}"
    # 检查 ConditionSet 函数对 x**2 = 1, S.UniversalSet 的输出是否正确
    assert latex(ConditionSet(x, Eq(x**2, 1), S.UniversalSet)) == \
        r"\left\{x\; \middle|\; x^{2} = 1 \right\}"

# 测试 latex 函数对 ComplexRegion 类的输出是否正确
def test_latex_ComplexRegion():
    # 检查 ComplexRegion 函数对 Interval(3, 5)*Interval(4, 6) 的输出是否正确
    assert latex(ComplexRegion(Interval(3, 5)*Interval(4, 6))) == \
        r"\left\{x + y i\; \middle|\; x, y \in \left[3, 5\right] \times \left[4, 6\right] \right\}"
    # 检查 ComplexRegion 函数对 Interval(0, 1)*Interval(0, 2*pi), polar=True 的输出是否正确
    assert latex(ComplexRegion(Interval(0, 1)*Interval(0, 2*pi), polar=True)) == \
        r"\left\{r \left(i \sin{\left(\theta \right)} + \cos{\left(\theta "\
        r"\right)}\right)\; \middle|\; r, \theta \in \left[0, 1\right] \times \left[0, 2 \pi\right) \right\}"

# 测试 latex 函数对 Contains 类的输出是否正确
def test_latex_Contains():
    x = Symbol('x')
    # 检查 Contains 函数对 x in S.Naturals 的输出是否正确
    assert latex(Contains(x, S.Naturals)) == r"x \in \mathbb{N}"

# 测试 latex 函数对 Sum 类的输出是否正确
def test_latex_sum():
    # 检查 Sum 函数对 x*y**2, (x, -2, 2), (y, -5, 5) 的输出是否正确
    assert latex(Sum(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        r"\sum_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    # 检查 Sum 函数对 x**2, (x, -2, 2) 的输出是否正确
    assert latex(Sum(x**2, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} x^{2}"
    # 检查 Sum 函数对 x**2 + y, (x, -2, 2) 的输出是否正确
    assert latex(Sum(x**2 + y, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} \left(x^{2} + y\right)"
    # 检查 Sum 函数对 (x**2 + y), (x, -2, 2))**2 的输出是否正确
    assert latex(Sum(x**2 + y, (x, -2, 2))**2) == \
        r"\left(\sum_{x=-2}^{2} \left(x^{2} + y\right)\right)^{2}"

# 测试 latex 函数对 Product 类的输出是否正确
def test_latex_product():
    # 检查 Product 函数对 x*y**2, (x, -2, 2), (y, -5, 5) 的输出是否正确
    assert latex(Product(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        r"\prod_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    # 检查 Product 函数对 x**2, (x, -2, 2) 的输出是否正确
    assert latex(Product(x**2, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} x^{2}"
    # 检查 Product 函数对 x**2 + y, (x, -2, 2) 的输出是否正确
    assert latex(Product(x**2 + y, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} \left(x^{2} + y\right)"
    # 检查 Product 函数对 (x), (x, -2, 2))**2 的输出是否正确
    assert latex(Product(x, (x, -2, 2))**2) == \
        r"\left(\prod_{x=-2}^{2} x\right)^{2}"

# 测试 latex 函数对 Limit 类的输出是否正确
def test_latex_limits():
    # 检查 Limit 函数对 x, x, oo 的输出是否正确
    assert latex(Limit(x, x, oo)) == r"\lim_{x \to \infty} x"

    # 检查 Limit 函数对 f(x), x, 0 的输出是否正确
    f = Function('f')
    assert latex(Limit(f(x), x, 0)) == r"\lim_{x \to 0^+} f{\left(x \right)}"
    # 断言：验证函数 latex 对 Limit 对象的输出是否符合预期的 LaTeX 表达式
    assert latex(Limit(f(x), x, 0, "-")) == \
        r"\lim_{x \to 0^-} f{\left(x \right)}"
    # 断言：验证函数 latex 对 Limit 对象的输出是否符合预期的 LaTeX 表达式，针对 issue #10806
    assert latex(Limit(f(x), x, 0)**2) == \
        r"\left(\lim_{x \to 0^+} f{\left(x \right)}\right)^{2}"
    # 断言：验证函数 latex 对 Limit 对象的输出是否符合预期的 LaTeX 表达式，包含双向极限
    assert latex(Limit(f(x), x, 0, dir='+-')) == \
        r"\lim_{x \to 0} f{\left(x \right)}"
# 定义测试函数，用于验证 latex 函数的输出是否符合预期
def test_latex_log():
    # 断言：log(x) 的 LaTeX 表示应为 "\log{\left(x \right)}"
    assert latex(log(x)) == r"\log{\left(x \right)}"
    # 断言：log(x) 的 LaTeX 表示（使用 ln_notation=True）应为 "\ln{\left(x \right)}"
    assert latex(log(x), ln_notation=True) == r"\ln{\left(x \right)}"
    # 断言：log(x) + log(y) 的 LaTeX 表示应为 "\log{\left(x \right)} + \log{\left(y \right)}"
    assert latex(log(x) + log(y)) == \
        r"\log{\left(x \right)} + \log{\left(y \right)}"
    # 断言：log(x) + log(y) 的 LaTeX 表示（使用 ln_notation=True）应为 "\ln{\left(x \right)} + \ln{\left(y \right)}"
    assert latex(log(x) + log(y), ln_notation=True) == \
        r"\ln{\left(x \right)} + \ln{\left(y \right)}"
    # 断言：pow(log(x), x) 的 LaTeX 表示应为 "\log{\left(x \right)}^{x}"
    assert latex(pow(log(x), x)) == r"\log{\left(x \right)}^{x}"
    # 断言：pow(log(x), x) 的 LaTeX 表示（使用 ln_notation=True）应为 "\ln{\left(x \right)}^{x}"
    assert latex(pow(log(x), x), ln_notation=True) == \
        r"\ln{\left(x \right)}^{x}"


# 定义测试函数，用于验证 latex 函数在不同 beta 符号表示时的输出是否符合预期
def test_issue_3568():
    # 创建符号 beta，使其表示为 LaTeX 字符 "\beta"
    beta = Symbol(r'\beta')
    # 创建表达式 y = beta + x
    y = beta + x
    # 断言：表达式 y 的 LaTeX 表示应为 "\beta + x" 或 "x + \beta"
    assert latex(y) in [r'\beta + x', r'x + \beta']

    # 创建符号 beta，使其表示为 LaTeX 字符 "beta"
    beta = Symbol(r'beta')
    # 创建表达式 y = beta + x
    y = beta + x
    # 断言：表达式 y 的 LaTeX 表示应为 "\beta + x" 或 "x + \beta"
    assert latex(y) in [r'\beta + x', r'x + \beta']


# 定义测试函数，用于验证 latex 函数在不同模式下的输出是否符合预期
def test_latex():
    # 断言：(2*tau)**Rational(7, 2) 的 LaTeX 表示应为 "8 \sqrt{2} \tau^{\frac{7}{2}}"
    assert latex((2*tau)**Rational(7, 2)) == r"8 \sqrt{2} \tau^{\frac{7}{2}}"
    # 断言：(2*mu)**Rational(7, 2) 的 LaTeX 表示（使用 mode='equation*'）应为 "\begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}"
    assert latex((2*mu)**Rational(7, 2), mode='equation*') == \
        r"\begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}"
    # 断言：(2*mu)**Rational(7, 2) 的 LaTeX 表示（使用 mode='equation', itex=True）应为 "$$8 \sqrt{2} \mu^{\frac{7}{2}}$$"
    assert latex((2*mu)**Rational(7, 2), mode='equation', itex=True) == \
        r"$$8 \sqrt{2} \mu^{\frac{7}{2}}$$"
    # 断言：[2/x, y] 的 LaTeX 表示应为 "\left[ \frac{2}{x}, \  y\right]"
    assert latex([2/x, y]) == r"\left[ \frac{2}{x}, \  y\right]"


# 定义测试函数，用于验证 latex 函数在处理带有符号的字典时的输出是否符合预期
def test_latex_dict():
    # 创建字典 d
    d = {Rational(1): 1, x**2: 2, x: 3, x**3: 4}
    # 断言：字典 d 的 LaTeX 表示应为 "\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}"
    assert latex(d) == \
        r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'
    # 创建 Dict 对象 D，其内容与字典 d 相同
    D = Dict(d)
    # 断言：Dict 对象 D 的 LaTeX 表示应为 "\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}"
    assert latex(D) == \
        r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'


# 定义测试函数，用于验证 latex 函数在处理符号列表时的输出是否符合预期
def test_latex_list():
    # 创建符号列表 ll
    ll = [Symbol('omega1'), Symbol('a'), Symbol('alpha')]
    # 断言：符号列表 ll 的 LaTeX 表示应为 "\left[ \omega_{1}, \  a, \  \alpha\right]"
    assert latex(ll) == r'\left[ \omega_{1}, \  a, \  \alpha\right]'


# 定义测试函数，用于验证 latex 函数在处理数学常数和符号时的输出是否符合预期
def test_latex_NumberSymbols():
    # 断言：S.Catalan 的 LaTeX 表示应为 "G"
    assert latex(S.Catalan) == "G"
    # 断言：S.EulerGamma 的 LaTeX 表示应为 "\gamma"
    assert latex(S.EulerGamma) == r"\gamma"
    # 断言：S.Exp1 的 LaTeX 表示应为 "e"
    assert latex(S.Exp1) == "e"
    # 断言：S.GoldenRatio 的 LaTeX 表示应为 "\phi"
    assert latex(S.GoldenRatio) == r"\phi"
    # 断言：S.Pi 的 LaTeX 表示应为 "\pi"
    assert latex(S.Pi) == r"\pi"
    # 断言：S.TribonacciConstant 的 LaTeX 表示应为 "\text{TribonacciConstant}"
    assert latex(S.TribonacciConstant) == r"\text{TribonacciConstant}"


# 定义测试函数，用于验证 latex 函数在处理有理数时的输出是否符合预期
def test_latex_rational():
    # 断言：-Rational(1, 2) 的 LaTeX 表示应为 "- \frac{1}{2}"
    assert latex(-Rational(1, 2)) == r"- \frac{1}{2}"
    # 断言：Rational(-1, 2) 的 LaTeX 表示应为 "- \frac{1}{2}"
    assert latex(Rational(-1, 2)) == r"- \frac{1}{2}"
    # 断言：Rational(1, -2) 的 LaTeX 表示应为 "- \frac{1}{2}"
    assert latex(Rational(1, -2)) == r"- \frac{1}{2}"
    # 断言：-Rational(-1, 2) 的 LaTeX 表示应为 "\frac{1}{2}"
    assert latex(-Rational(-1, 2)) == r"\frac{1}{2}"
    # 断言：-Rational(1, 2)*x 的 LaTeX 表示应为 "- \frac{x}{2}"
    assert latex(-Rational(1, 2)*x) == r"- \frac{x}{2}"
    # 断言：-Rational(1, 2)*x + Rational(-2, 3)*y 的 LaTeX 表示应为 "- \frac{x}{2} - \frac{2 y}{3}"
    assert latex(-Rational(1, 2)*x + Rational(-2, 3)*y
    # 使用 sympy 库中的 latex 函数，验证 Heaviside 函数的平方是否与给定的 LaTeX 字符串相匹配
    assert latex(Heaviside(x)**2) == r"\left(\theta\left(x\right)\right)^{2}"
# 定义一个测试函数，用于测试 KroneckerDelta 表达式的 LaTeX 输出
def test_latex_KroneckerDelta():
    # 断言：KroneckerDelta(x, y) 的 LaTeX 输出应该是 \delta_{x y}
    assert latex(KroneckerDelta(x, y)) == r"\delta_{x y}"
    # 断言：KroneckerDelta(x, y + 1) 的 LaTeX 输出应该是 \delta_{x, y + 1}
    assert latex(KroneckerDelta(x, y + 1)) == r"\delta_{x, y + 1}"
    # issue 6578
    # 断言：KroneckerDelta(x + 1, y) 的 LaTeX 输出应该是 \delta_{y, x + 1}
    assert latex(KroneckerDelta(x + 1, y)) == r"\delta_{y, x + 1}"
    # 断言：(KroneckerDelta(x, y))^2 的 LaTeX 输出应该是 \left(\delta_{x y}\right)^{2}
    assert latex(Pow(KroneckerDelta(x, y), 2, evaluate=False)) == \
        r"\left(\delta_{x y}\right)^{2}"


# 定义一个测试函数，用于测试 LeviCivita 符号的 LaTeX 输出
def test_latex_LeviCivita():
    # 断言：LeviCivita(x, y, z) 的 LaTeX 输出应该是 \varepsilon_{x y z}
    assert latex(LeviCivita(x, y, z)) == r"\varepsilon_{x y z}"
    # 断言：(LeviCivita(x, y, z))^2 的 LaTeX 输出应该是 \left(\varepsilon_{x y z}\right)^{2}
    assert latex(LeviCivita(x, y, z)**2) == \
        r"\left(\varepsilon_{x y z}\right)^{2}"
    # 断言：LeviCivita(x, y, z + 1) 的 LaTeX 输出应该是 \varepsilon_{x, y, z + 1}
    assert latex(LeviCivita(x, y, z + 1)) == r"\varepsilon_{x, y, z + 1}"
    # 断言：LeviCivita(x, y + 1, z) 的 LaTeX 输出应该是 \varepsilon_{x, y + 1, z}
    assert latex(LeviCivita(x, y + 1, z)) == r"\varepsilon_{x, y + 1, z}"
    # 断言：LeviCivita(x + 1, y, z) 的 LaTeX 输出应该是 \varepsilon_{x + 1, y, z}
    assert latex(LeviCivita(x + 1, y, z)) == r"\varepsilon_{x + 1, y, z}"


# 定义一个测试函数，用于测试 latex 函数在不同模式下输出的效果
def test_mode():
    # 定义一个表达式 expr = x + y
    expr = x + y
    # 断言：latex(expr) 的 LaTeX 输出应该是 'x + y'
    assert latex(expr) == r'x + y'
    # 断言：latex(expr, mode='plain') 的 LaTeX 输出应该是 'x + y'
    assert latex(expr, mode='plain') == r'x + y'
    # 断言：latex(expr, mode='inline') 的 LaTeX 输出应该是 '$x + y$'
    assert latex(expr, mode='inline') == r'$x + y$'
    # 断言：latex(expr, mode='equation*') 的 LaTeX 输出应该是 '\begin{equation*}x + y\end{equation*}'
    assert latex(expr, mode='equation*') == r'\begin{equation*}x + y\end{equation*}'
    # 断言：latex(expr, mode='equation') 的 LaTeX 输出应该是 '\begin{equation}x + y\end{equation}'
    assert latex(expr, mode='equation') == r'\begin{equation}x + y\end{equation}'
    # 断言：latex(expr, mode='foo') 应该引发 ValueError 异常
    raises(ValueError, lambda: latex(expr, mode='foo'))


# 定义一个测试函数，用于测试 Mathieu 函数的 LaTeX 输出
def test_latex_mathieu():
    # 断言：mathieuc(x, y, z) 的 LaTeX 输出应该是 "C\left(x, y, z\right)"
    assert latex(mathieuc(x, y, z)) == r"C\left(x, y, z\right)"
    # 断言：mathieus(x, y, z) 的 LaTeX 输出应该是 "S\left(x, y, z\right)"
    assert latex(mathieus(x, y, z)) == r"S\left(x, y, z\right)"
    # 断言：(mathieuc(x, y, z))^2 的 LaTeX 输出应该是 "C\left(x, y, z\right)^{2}"
    assert latex(mathieuc(x, y, z)**2) == r"C\left(x, y, z\right)^{2}"
    # 断言：(mathieus(x, y, z))^2 的 LaTeX 输出应该是 "S\left(x, y, z\right)^{2}"
    assert latex(mathieus(x, y, z)**2) == r"S\left(x, y, z\right)^{2}"
    # 断言：mathieucprime(x, y, z) 的 LaTeX 输出应该是 "C^{\prime}\left(x, y, z\right)"
    assert latex(mathieucprime(x, y, z)) == r"C^{\prime}\left(x, y, z\right)"
    # 断言：mathieusprime(x, y, z) 的 LaTeX 输出应该是 "S^{\prime}\left(x, y, z\right)"
    assert latex(mathieusprime(x, y, z)) == r"S^{\prime}\left(x, y, z\right)"
    # 断言：(mathieucprime(x, y, z))^2 的 LaTeX 输出应该是 "C^{\prime}\left(x, y, z\right)^{2}"
    assert latex(mathieucprime(x, y, z)**2) == r"C^{\prime}\left(x, y, z\right)^{2}"
    # 断言：(mathieusprime(x, y, z))^2 的 LaTeX 输出应该是 "S^{\prime}\left(x, y, z\right)^{2}"
    assert latex(mathieusprime(x, y, z)**2) == r"S^{\prime}\left(x, y, z\right)^{2}"


# 定义一个测试函数，用于测试 Piecewise 函数的 LaTeX 输出
def test_latex_Piecewise():
    # 创建一个 Piecewise 对象 p
    p = Piecewise((x, x < 1), (x**2, True))
    # 断言：latex(p) 的 LaTeX 输出应该是 '\begin{cases} x & \text{for}\: x < 1 \\x^{2} & \text{otherwise} \end{cases}'
    assert latex(p) == r"\begin{cases} x & \text{for}\: x < 1 \\x^{2} & \text{otherwise} \end{cases}"
    # 断言：latex(p, itex=True) 的 LaTeX 输出应该是 '\begin{cases} x & \text{for}\: x \lt 1 \\x^{2} & \text{otherwise} \end{cases}'
    assert latex(p, itex=True) == r"\begin{cases} x & \text{for}\: x \lt 1 \\x^{2} & \text{otherwise} \end{cases}"
    # 创建另一个 Piecewise 对象 p
    p = Piecewise((x, x < 0), (0, x >= 0))
    # 断言：latex(p) 的 LaTeX 输出应该是 '\begin{cases} x & \text{for}\: x < 0 \\0 & \text{otherwise} \end{cases}'
    assert latex(p) == r'\begin{cases} x & \text{for}\: x < 0 \\0 & \text{otherwise} \end{cases}'
    # 定义两个非可交换符号 A 和 B
    A, B = symbols("A B", commutative=False)
    # 创建一个 Piecewise 对象 p
    p = Piecewise((A**2, Eq(A, B)), (A*B, True))
    # 定义预期的 LaTeX 输出字符串 s
    s = r"\begin{cases} A^{2} & \text{for}\: A = B \\A B & \text{otherwise} \end{cases}"
    # 断言：latex(p) 的 LaTeX 输出应该
    # 断言：验证矩阵 M 的 LaTeX 表示是否与指定的内联模式字符串匹配
    assert latex(M, mode='inline') == \
        r'$\left[\begin{smallmatrix}x + 1 & y\\' \
        r'y & x - 1\end{smallmatrix}\right]$'
    
    # 断言：验证矩阵 M 的 LaTeX 表示是否与指定的 array 格式字符串匹配
    assert latex(M, mat_str='array') == \
        r'\left[\begin{array}{cc}x + 1 & y\\y & x - 1\end{array}\right]'
    
    # 断言：验证矩阵 M 的 LaTeX 表示是否与指定的 bmatrix 格式字符串匹配
    assert latex(M, mat_str='bmatrix') == \
        r'\left[\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}\right]'
    
    # 断言：验证矩阵 M 的 LaTeX 表示是否与指定的 bmatrix 格式字符串匹配，但不包含外部的定界符
    assert latex(M, mat_delim=None, mat_str='bmatrix') == \
        r'\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}'
    
    # 创建一个新的矩阵 M2，包含 1 行 11 列，从 0 到 10 的整数
    M2 = Matrix(1, 11, range(11))
    
    # 断言：验证矩阵 M2 的 LaTeX 表示是否与指定的格式字符串匹配
    assert latex(M2) == \
        r'\left[\begin{array}{ccccccccccc}' \
        r'0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'
# 定义测试函数，用于测试生成 LaTeX 代码的几种情况
def test_latex_matrix_with_functions():
    # 定义符号变量 t 和 theta1
    t = symbols('t')
    theta1 = symbols('theta1', cls=Function)

    # 创建一个 2x2 的矩阵 M，其中元素是关于 theta1(t) 和其导数的三角函数
    M = Matrix([[sin(theta1(t)), cos(theta1(t))],
                [cos(theta1(t).diff(t)), sin(theta1(t).diff(t))]])

    # 期望的 LaTeX 字符串，展示了矩阵 M 的数学表达式
    expected = (r'\left[\begin{matrix}\sin{\left('
                r'\theta_{1}{\left(t \right)} \right)} & '
                r'\cos{\left(\theta_{1}{\left(t \right)} \right)'
                r'}\\\cos{\left(\frac{d}{d t} \theta_{1}{\left(t '
                r'\right)} \right)} & \sin{\left(\frac{d}{d t} '
                r'\theta_{1}{\left(t \right)} \right'
                r')}\end{matrix}\right]')

    # 断言生成的 LaTeX 代码与期望的字符串相等
    assert latex(M) == expected


def test_latex_NDimArray():
    # 定义符号变量 x, y, z, w
    x, y, z, w = symbols("x y z w")

    # 遍历不同类型的 N 维数组
    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
                      MutableDenseNDimArray, MutableSparseNDimArray):
        # 创建一个标量数组 M，包含一个符号变量 x
        M = ArrayType(x)

        # 断言生成的 LaTeX 代码与期望的字符串相等
        assert latex(M) == r"x"

        # 创建一个 2x2 的数组 M，包含符号变量 x, y, z, w
        M = ArrayType([[1 / x, y], [z, w]])
        # 创建一个 1x3 的数组 M1，包含符号变量 x, y, z
        M1 = ArrayType([1 / x, y, z])

        # 创建 M1 和 M 的张量积 M2 和 M3
        M2 = tensorproduct(M1, M)
        M3 = tensorproduct(M, M)

        # 断言生成的 LaTeX 代码与期望的字符串相等
        assert latex(M) == \
            r'\left[\begin{matrix}\frac{1}{x} & y\\z & w\end{matrix}\right]'
        assert latex(M1) == \
            r"\left[\begin{matrix}\frac{1}{x} & y & z\end{matrix}\right]"
        assert latex(M2) == \
            r"\left[\begin{matrix}" \
            r"\left[\begin{matrix}\frac{1}{x^{2}} & \frac{y}{x}\\\frac{z}{x} & \frac{w}{x}\end{matrix}\right] & " \
            r"\left[\begin{matrix}\frac{y}{x} & y^{2}\\y z & w y\end{matrix}\right] & " \
            r"\left[\begin{matrix}\frac{z}{x} & y z\\z^{2} & w z\end{matrix}\right]" \
            r"\end{matrix}\right]"
        assert latex(M3) == \
            r"""\left[\begin{matrix}"""\
            r"""\left[\begin{matrix}\frac{1}{x^{2}} & \frac{y}{x}\\\frac{z}{x} & \frac{w}{x}\end{matrix}\right] & """\
            r"""\left[\begin{matrix}\frac{y}{x} & y^{2}\\y z & w y\end{matrix}\right]\\"""\
            r"""\left[\begin{matrix}\frac{z}{x} & y z\\z^{2} & w z\end{matrix}\right] & """\
            r"""\left[\begin{matrix}\frac{w}{x} & w y\\w z & w^{2}\end{matrix}\right]"""\
            r"""\end{matrix}\right]"""

        # 创建包含一行的数组 Mrow 和包含一列的数组 Mcolumn
        Mrow = ArrayType([[x, y, 1/z]])
        Mcolumn = ArrayType([[x], [y], [1/z]])
        # 创建一个列包含数组 Mcolumn 的数组 Mcol2
        Mcol2 = ArrayType([Mcolumn.tolist()])

        # 断言生成的 LaTeX 代码与期望的字符串相等
        assert latex(Mrow) == \
            r"\left[\left[\begin{matrix}x & y & \frac{1}{z}\end{matrix}\right]\right]"
        assert latex(Mcolumn) == \
            r"\left[\begin{matrix}x\\y\\\frac{1}{z}\end{matrix}\right]"
        assert latex(Mcol2) == \
            r'\left[\begin{matrix}\left[\begin{matrix}x\\y\\\frac{1}{z}\end{matrix}\right]\end{matrix}\right]'


def test_latex_mul_symbol():
    # 测试不同乘号符号生成的 LaTeX 代码
    assert latex(4*4**x, mul_symbol='times') == r"4 \times 4^{x}"
    assert latex(4*4**x, mul_symbol='dot') == r"4 \cdot 4^{x}"
    assert latex(4*4**x, mul_symbol='ldot') == r"4 \,.\, 4^{x}"
    # 断言：使用latex函数对表达式4*x进行转换，并指定乘号符号为'times'，期望输出为"4 \times x"
    assert latex(4*x, mul_symbol='times') == r"4 \times x"
    
    # 断言：使用latex函数对表达式4*x进行转换，并指定乘号符号为'dot'，期望输出为"4 \cdot x"
    assert latex(4*x, mul_symbol='dot') == r"4 \cdot x"
    
    # 断言：使用latex函数对表达式4*x进行转换，并指定乘号符号为'ldot'，期望输出为"4 \,.\, x"
    assert latex(4*x, mul_symbol='ldot') == r"4 \,.\, x"
def test_latex_issue_4381():
    # 计算表达式 y = 4 * 4 ** log(2)，其中 log 是对数函数
    y = 4*4**log(2)
    # 断言 LaTeX 格式化后的 y 是否等于预期值
    assert latex(y) == r'4 \cdot 4^{\log{\left(2 \right)}}'
    # 断言 LaTeX 格式化后的 1/y 是否等于预期值
    assert latex(1/y) == r'\frac{1}{4 \cdot 4^{\log{\left(2 \right)}}}'


def test_latex_issue_4576():
    # 断言符号 "beta_13_2" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("beta_13_2")) == r"\beta_{13 2}"
    # 断言符号 "beta_132_20" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("beta_132_20")) == r"\beta_{132 20}"
    # 断言符号 "beta_13" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("beta_13")) == r"\beta_{13}"
    # 断言符号 "x_a_b" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_a_b")) == r"x_{a b}"
    # 断言符号 "x_1_2_3" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_1_2_3")) == r"x_{1 2 3}"
    # 断言符号 "x_a_b1" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_a_b1")) == r"x_{a b1}"
    # 断言符号 "x_a_1" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_a_1")) == r"x_{a 1}"
    # 断言符号 "x_1_a" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_1_a")) == r"x_{1 a}"
    # 断言符号 "x_1^aa" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_1^aa")) == r"x^{aa}_{1}"
    # 断言符号 "x_1__aa" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_1__aa")) == r"x^{aa}_{1}"
    # 断言符号 "x_11^a" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_11^a")) == r"x^{a}_{11}"
    # 断言符号 "x_11__a" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_11__a")) == r"x^{a}_{11}"
    # 断言符号 "x_a_a_a_a" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_a_a_a_a")) == r"x_{a a a a}"
    # 断言符号 "x_a_a^a^a" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_a_a^a^a")) == r"x^{a a}_{a a}"
    # 断言符号 "x_a_a__a__a" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("x_a_a__a__a")) == r"x^{a a}_{a a}"
    # 断言符号 "alpha_11" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("alpha_11")) == r"\alpha_{11}"
    # 断言符号 "alpha_11_11" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("alpha_11_11")) == r"\alpha_{11 11}"
    # 断言符号 "alpha_alpha" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("alpha_alpha")) == r"\alpha_{\alpha}"
    # 断言符号 "alpha^aleph" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("alpha^aleph")) == r"\alpha^{\aleph}"
    # 断言符号 "alpha__aleph" 的 LaTeX 表示是否等于预期值
    assert latex(Symbol("alpha__aleph")) == r"\alpha^{\aleph}"


def test_latex_pow_fraction():
    x = Symbol('x')
    # 测试指数函数 exp
    assert r'e^{-x}' in latex(exp(-x)/2).replace(' ', '')  # 去除空格

    # 测试 e^{-x} 在未来更改时是否影响乘法或分数行为
    # 特别是当前输出为 \frac{1}{2}e^{- x}，但未来可能更改为 \frac{e^{-x}}{2}

    # 测试一般的非指数幂
    assert r'3^{-x}' in latex(3**-x/2).replace(' ', '')


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    # 断言非交换变量的 LaTeX 表示是否等于预期值
    assert latex(A*B*C**-1) == r"A B C^{-1}"
    assert latex(C**-1*A*B) == r"C^{-1} A B"
    assert latex(A*C**-1*B) == r"A C^{-1} B"


def test_latex_order():
    expr = x**3 + x**2*y + y**4 + 3*x*y**3

    # 断言按字典序 'lex' 排序的 LaTeX 表示是否等于预期值
    assert latex(expr, order='lex') == r"x^{3} + x^{2} y + 3 x y^{3} + y^{4}"
    # 断言按字典序 'rev-lex' 排序的 LaTeX 表示是否等于预期值
    assert latex(
        expr, order='rev-lex') == r"y^{4} + 3 x y^{3} + x^{2} y + x^{3}"
    # 断言不排序的 LaTeX 表示是否等于预期值
    assert latex(expr, order='none') == r"x^{3} + y^{4} + y x^{2} + 3 x y^{3}"


def test_latex_Lambda():
    # 断言 Lambda 函数的 LaTeX 表示是否等于预期值
    assert latex(Lambda(x, x + 1)) == r"\left( x \mapsto x + 1 \right)"
    assert latex(Lambda((x, y), x + 1)) == r"\left( \left( x, \  y\right) \mapsto x + 1 \right)"
    assert latex(Lambda(x, x)) == r"\left( x \mapsto x \right)"

def test_latex_PolyElement():
    Ruv, u, v = ring("u,v", ZZ)
    Rxyz, x, y, z = ring("x,y,z", Ruv)

    # 断言多项式元素的 LaTeX 表示是否等于预期值
    assert latex(x - x) == r"0"
    assert latex(x - 1) == r"x - 1"
    assert latex(x + 1) == r"x + 1"

    assert latex((u**2 + 3*u*v + 1)*x**2*y + u + 1) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + u + 1"
    # 断言：验证 LaTeX 表达式的转换是否正确
    assert latex((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x"
    
    # 断言：验证 LaTeX 表达式的转换是否正确，包括最后的常数项
    assert latex((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x + 1"
    
    # 断言：验证 LaTeX 表达式的转换是否正确，包括负号和最后的常数项
    assert latex((-u**2 + 3*u*v - 1)*x**2*y - (u + 1)*x - 1) == \
        r"-\left({u}^{2} - 3 u v + 1\right) {x}^{2} y - \left(u + 1\right) x - 1"
    
    # 断言：验证 LaTeX 表达式的转换是否正确，包括负号和常数项
    assert latex(-(v**2 + v + 1)*x + 3*u*v + 1) == \
        r"-\left({v}^{2} + v + 1\right) x + 3 u v + 1"
    
    # 断言：验证 LaTeX 表达式的转换是否正确，包括负号和常数项
    assert latex(-(v**2 + v + 1)*x - 3*u*v + 1) == \
        r"-\left({v}^{2} + v + 1\right) x - 3 u v + 1"
# 定义测试函数 test_latex_FracElement，用于测试 latex 函数对于有理数元素的输出
def test_latex_FracElement():
    # 定义有理数域 Fuv，包含变量 u 和 v
    Fuv, u, v = field("u,v", ZZ)
    # 在有理数域 Fuv 的基础上定义多项式环 Fxyzt，包含变量 x, y, z, t
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    # 断言以下 latex 函数的输出符合预期
    assert latex(x - x) == r"0"
    assert latex(x - 1) == r"x - 1"
    assert latex(x + 1) == r"x + 1"

    assert latex(x/3) == r"\frac{x}{3}"
    assert latex(x/z) == r"\frac{x}{z}"
    assert latex(x*y/z) == r"\frac{x y}{z}"
    assert latex(x/(z*t)) == r"\frac{x}{z t}"
    assert latex(x*y/(z*t)) == r"\frac{x y}{z t}"

    assert latex((x - 1)/y) == r"\frac{x - 1}{y}"
    assert latex((x + 1)/y) == r"\frac{x + 1}{y}"
    assert latex((-x - 1)/y) == r"\frac{-x - 1}{y}"
    assert latex((x + 1)/(y*z)) == r"\frac{x + 1}{y z}"
    assert latex(-y/(x + 1)) == r"\frac{-y}{x + 1}"
    assert latex(y*z/(x + 1)) == r"\frac{y z}{x + 1}"

    assert latex(((u + 1)*x*y + 1)/((v - 1)*z - 1)) == \
        r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - 1}"
    assert latex(((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)) == \
        r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - u v t - 1}"


# 定义测试函数 test_latex_Poly，用于测试 latex 函数对于多项式的输出
def test_latex_Poly():
    # 断言 latex 函数对于不同类型的多项式输出符合预期
    assert latex(Poly(x**2 + 2 * x, x)) == \
        r"\operatorname{Poly}{\left( x^{2} + 2 x, x, domain=\mathbb{Z} \right)}"
    assert latex(Poly(x/y, x)) == \
        r"\operatorname{Poly}{\left( \frac{1}{y} x, x, domain=\mathbb{Z}\left(y\right) \right)}"
    assert latex(Poly(2.0*x + y)) == \
        r"\operatorname{Poly}{\left( 2.0 x + 1.0 y, x, y, domain=\mathbb{R} \right)}"


# 定义测试函数 test_latex_Poly_order，用于测试 latex 函数对于高阶多项式的输出
def test_latex_Poly_order():
    # 断言 latex 函数对于不同高阶多项式输出符合预期
    assert latex(Poly([a, 1, b, 2, c, 3], x)) == \
        r'\operatorname{Poly}{\left( a x^{5} + x^{4} + b x^{3} + 2 x^{2} + c'\
        r' x + 3, x, domain=\mathbb{Z}\left[a, b, c\right] \right)}'
    assert latex(Poly([a, 1, b+c, 2, 3], x)) == \
        r'\operatorname{Poly}{\left( a x^{4} + x^{3} + \left(b + c\right) '\
        r'x^{2} + 2 x + 3, x, domain=\mathbb{Z}\left[a, b, c\right] \right)}'
    assert latex(Poly(a*x**3 + x**2*y - x*y - c*y**3 - b*x*y**2 + y - a*x + b,
                      (x, y))) == \
        r'\operatorname{Poly}{\left( a x^{3} + x^{2}y -  b xy^{2} - xy -  '\
        r'a x -  c y^{3} + y + b, x, y, domain=\mathbb{Z}\left[a, b, c\right] \right)}'


# 定义测试函数 test_latex_ComplexRootOf，用于测试 latex 函数对于复数根的输出
def test_latex_ComplexRootOf():
    # 断言 latex 函数对于复数根的输出符合预期
    assert latex(rootof(x**5 + x + 3, 0)) == \
        r"\operatorname{CRootOf} {\left(x^{5} + x + 3, 0\right)}"


# 定义测试函数 test_latex_RootSum，用于测试 latex 函数对于根之和的输出
def test_latex_RootSum():
    # 断言 latex 函数对于根之和的输出符合预期
    assert latex(RootSum(x**5 + x + 3, sin)) == \
        r"\operatorname{RootSum} {\left(x^{5} + x + 3, \left( x \mapsto \sin{\left(x \right)} \right)\right)}"


# 定义测试函数 test_settings，用于测试设置 latex 函数时的异常处理
def test_settings():
    # 断言设置 latex 函数时传入错误参数引发 TypeError 异常
    raises(TypeError, lambda: latex(x*y, method="garbage"))


# 定义测试函数 test_latex_numbers，用于测试 latex 函数对于数学函数的输出
def test_latex_numbers():
    # 断言 latex 函数对于不同数学函数输出符合预期
    assert latex(catalan(n)) == r"C_{n}"
    assert latex(catalan(n)**2) == r"C_{n}^{2}"
    assert latex(bernoulli(n)) == r"B_{n}"
    assert latex(bernoulli(n, x)) == r"B_{n}\left(x\right)"
    assert latex(bernoulli(n)**2) == r"B_{n}^{2}"
    assert latex(bernoulli(n, x)**2) == r"B_{n}^{2}\left(x\right)"
    assert latex(genocchi(n)) == r"G_{n}"
    # 断言：生成 n 阶 Genocchi 多项式关于 x 的 LaTeX 表示是否等于 G_{n}\left(x\right)
    assert latex(genocchi(n, x)) == r"G_{n}\left(x\right)"
    # 断言：生成 n 阶 Genocchi 多项式的平方的 LaTeX 表示是否等于 G_{n}^{2}
    assert latex(genocchi(n)**2) == r"G_{n}^{2}"
    # 断言：生成 n 阶 Genocchi 多项式关于 x 的平方的 LaTeX 表示是否等于 G_{n}^{2}\left(x\right)
    assert latex(genocchi(n, x)**2) == r"G_{n}^{2}\left(x\right)"
    # 断言：生成 n 阶 Bell 数的 LaTeX 表示是否等于 B_{n}
    assert latex(bell(n)) == r"B_{n}"
    # 断言：生成 n 阶 Bell 数关于 x 的 LaTeX 表示是否等于 B_{n}\left(x\right)
    assert latex(bell(n, x)) == r"B_{n}\left(x\right)"
    # 断言：生成 n, m 阶 Bell 数关于 (x, y) 的 LaTeX 表示是否等于 B_{n, m}\left(x, y\right)
    assert latex(bell(n, m, (x, y))) == r"B_{n, m}\left(x, y\right)"
    # 断言：生成 n 阶 Bell 数的平方的 LaTeX 表示是否等于 B_{n}^{2}
    assert latex(bell(n)**2) == r"B_{n}^{2}"
    # 断言：生成 n 阶 Bell 数关于 x 的平方的 LaTeX 表示是否等于 B_{n}^{2}\left(x\right)
    assert latex(bell(n, x)**2) == r"B_{n}^{2}\left(x\right)"
    # 断言：生成 n, m 阶 Bell 数关于 (x, y) 的平方的 LaTeX 表示是否等于 B_{n, m}^{2}\left(x, y\right)
    assert latex(bell(n, m, (x, y))**2) == r"B_{n, m}^{2}\left(x, y\right)"
    # 断言：生成 n 阶 Fibonacci 数的 LaTeX 表示是否等于 F_{n}
    assert latex(fibonacci(n)) == r"F_{n}"
    # 断言：生成 n 阶 Fibonacci 数关于 x 的 LaTeX 表示是否等于 F_{n}\left(x\right)
    assert latex(fibonacci(n, x)) == r"F_{n}\left(x\right)"
    # 断言：生成 n 阶 Fibonacci 数的平方的 LaTeX 表示是否等于 F_{n}^{2}
    assert latex(fibonacci(n)**2) == r"F_{n}^{2}"
    # 断言：生成 n 阶 Fibonacci 数关于 x 的平方的 LaTeX 表示是否等于 F_{n}^{2}\left(x\right)
    assert latex(fibonacci(n, x)**2) == r"F_{n}^{2}\left(x\right)"
    # 断言：生成 n 阶 Lucas 数的 LaTeX 表示是否等于 L_{n}
    assert latex(lucas(n)) == r"L_{n}"
    # 断言：生成 n 阶 Lucas 数的平方的 LaTeX 表示是否等于 L_{n}^{2}
    assert latex(lucas(n)**2) == r"L_{n}^{2}"
    # 断言：生成 n 阶 Tribonacci 数的 LaTeX 表示是否等于 T_{n}
    assert latex(tribonacci(n)) == r"T_{n}"
    # 断言：生成 n 阶 Tribonacci 数关于 x 的 LaTeX 表示是否等于 T_{n}\left(x\right)
    assert latex(tribonacci(n, x)) == r"T_{n}\left(x\right)"
    # 断言：生成 n 阶 Tribonacci 数的平方的 LaTeX 表示是否等于 T_{n}^{2}
    assert latex(tribonacci(n)**2) == r"T_{n}^{2}"
    # 断言：生成 n 阶 Tribonacci 数关于 x 的平方的 LaTeX 表示是否等于 T_{n}^{2}\left(x\right)
    assert latex(tribonacci(n, x)**2) == r"T_{n}^{2}\left(x\right)"
    # 断言：生成 n 阶 Mobius 函数的 LaTeX 表示是否等于 \mu\left(n\right)
    assert latex(mobius(n)) == r"\mu\left(n\right)"
    # 断言：生成 n 阶 Mobius 函数的平方的 LaTeX 表示是否等于 \mu^{2}\left(n\right)
    assert latex(mobius(n)**2) == r"\mu^{2}\left(n\right)"
def test_latex_euler():
    assert latex(euler(n)) == r"E_{n}"  # 测试 euler 函数的 LaTeX 渲染是否正确
    assert latex(euler(n, x)) == r"E_{n}\left(x\right)"  # 测试带参数的 euler 函数的 LaTeX 渲染是否正确
    assert latex(euler(n, x)**2) == r"E_{n}^{2}\left(x\right)"  # 测试 euler 函数平方的 LaTeX 渲染是否正确


def test_lamda():
    assert latex(Symbol('lamda')) == r"\lambda"  # 测试小写 lambda 符号的 LaTeX 渲染是否正确
    assert latex(Symbol('Lamda')) == r"\Lambda"  # 测试大写 Lambda 符号的 LaTeX 渲染是否正确


def test_custom_symbol_names():
    x = Symbol('x')
    y = Symbol('y')
    assert latex(x) == r"x"  # 测试普通符号 x 的 LaTeX 渲染是否正确
    assert latex(x, symbol_names={x: "x_i"}) == r"x_i"  # 测试自定义符号名称的 LaTeX 渲染是否正确
    assert latex(x + y, symbol_names={x: "x_i"}) == r"x_i + y"  # 测试符号名称在表达式中的应用
    assert latex(x**2, symbol_names={x: "x_i"}) == r"x_i^{2}"  # 测试带幂次的符号名称的 LaTeX 渲染是否正确
    assert latex(x + y, symbol_names={x: "x_i", y: "y_j"}) == r"x_i + y_j"  # 测试多个自定义符号名称的 LaTeX 渲染是否正确


def test_matAdd():
    C = MatrixSymbol('C', 5, 5)
    B = MatrixSymbol('B', 5, 5)

    n = symbols("n")
    h = MatrixSymbol("h", 1, 1)

    assert latex(C - 2*B) in [r'- 2 B + C', r'C -2 B']  # 测试矩阵相加和乘法的 LaTeX 渲染是否正确
    assert latex(C + 2*B) in [r'2 B + C', r'C + 2 B']  # 测试矩阵相加和乘法的 LaTeX 渲染是否正确
    assert latex(B - 2*C) in [r'B - 2 C', r'- 2 C + B']  # 测试矩阵相加和乘法的 LaTeX 渲染是否正确
    assert latex(B + 2*C) in [r'B + 2 C', r'2 C + B']  # 测试矩阵相加和乘法的 LaTeX 渲染是否正确

    assert latex(n * h - (-h + h.T) * (h + h.T)) == 'n h - \\left(- h + h^{T}\\right) \\left(h + h^{T}\\right)'  # 测试复杂表达式的 LaTeX 渲染是否正确
    assert latex(MatAdd(MatAdd(h, h), MatAdd(h, h))) == '\\left(h + h\\right) + \\left(h + h\\right)'  # 测试 MatAdd 函数的 LaTeX 渲染是否正确
    assert latex(MatMul(MatMul(h, h), MatMul(h, h))) == '\\left(h h\\right) \\left(h h\\right)'  # 测试 MatMul 函数的 LaTeX 渲染是否正确


def test_matMul():
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    x = Symbol('x')
    assert latex(2*A) == r'2 A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(2*x*A) == r'2 x A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(-2*A) == r'- 2 A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(1.5*A) == r'1.5 A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(sqrt(2)*A) == r'\sqrt{2} A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(-sqrt(2)*A) == r'- \sqrt{2} A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(2*sqrt(2)*x*A) == r'2 \sqrt{2} x A'  # 测试矩阵乘法的 LaTeX 渲染是否正确
    assert latex(-2*A*(A + 2*B)) in [r'- 2 A \left(A + 2 B\right)', r'- 2 A \left(2 B + A\right)']  # 测试复杂表达式的 LaTeX 渲染是否正确


def test_latex_MatrixSlice():
    n = Symbol('n', integer=True)
    x, y, z, w, t, = symbols('x y z w t')
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    assert latex(MatrixSlice(X, (None, None, None), (None, None, None))) == r'X\left[:, :\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[x:x + 1, y:y + 1]) == r'X\left[x:x + 1, y:y + 1\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[x:x + 1:2, y:y + 1:2]) == r'X\left[x:x + 1:2, y:y + 1:2\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[:x, y:]) == r'X\left[:x, y:\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[x:, :y]) == r'X\left[x:, :y\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[x:y, z:w]) == r'X\left[x:y, z:w\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[x:y:t, w:t:x]) == r'X\left[x:y:t, w:t:x\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[x::y, t::w]) == r'X\left[x::y, t::w\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[:x:y, :t:w]) == r'X\left[:x:y, :t:w\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(X[::x, ::y]) == r'X\left[::x, ::y\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(MatrixSlice(X, (0, None, None), (0, None, None))) == r'X\left[:, :\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    assert latex(MatrixSlice(X, (None, n, None), (None, n, None))) == r'X\left[:, :\right]'  # 测试矩阵切片的 LaTeX 渲染是否正确
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(MatrixSlice(X, (0, n, None), (0, n, None))) == r'X\left[:, :\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(MatrixSlice(X, (0, n, 2), (0, n, 2))) == r'X\left[::2, ::2\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(X[1:2:3, 4:5:6]) == r'X\left[1:2:3, 4:5:6\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(X[1:3:5, 4:6:8]) == r'X\left[1:3:5, 4:6:8\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(X[1:10:2]) == r'X\left[1:10:2, :\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(Y[:5, 1:9:2]) == r'Y\left[:5, 1:9:2\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(Y[:5, 1:10:2]) == r'Y\left[:5, 1::2\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(Y[5, :5:2]) == r'Y\left[5:6, :5:2\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(X[0:1, 0:1]) == r'X\left[:1, :1\right]'
    
    # 验证给定的矩阵切片操作是否产生正确的 LaTeX 表示
    assert latex(X[0:1:2, 0:1:2]) == r'X\left[:1:2, :1:2\right]'
    
    # 验证给定的矩阵加法与切片操作是否产生正确的 LaTeX 表示
    assert latex((Y + Z)[2:, 2:]) == r'\left(Y + Z\right)\left[2:, 2:\right]'
def test_latex_RandomDomain():
    # 导入需要的符号统计相关模块和函数
    from sympy.stats import Normal, Die, Exponential, pspace, where
    from sympy.stats.rv import RandomDomain

    # 创建正态分布随机变量 X，均值为 0，方差为 1
    X = Normal('x1', 0, 1)
    # 断言语句，验证 X > 0 的条件，生成对应的 LaTeX 表达式
    assert latex(where(X > 0)) == r"\text{Domain: }0 < x_{1} \wedge x_{1} < \infty"

    # 创建一个六面骰子的随机变量 D
    D = Die('d1', 6)
    # 断言语句，验证 D > 4 的条件，生成对应的 LaTeX 表达式
    assert latex(where(D > 4)) == r"\text{Domain: }d_{1} = 5 \vee d_{1} = 6"

    # 创建指数分布随机变量 A 和 B
    A = Exponential('a', 1)
    B = Exponential('b', 1)
    # 断言语句，验证随机向量 (A, B) 的定义域，生成对应的 LaTeX 表达式
    assert latex(pspace(Tuple(A, B)).domain) == \
        r"\text{Domain: }0 \leq a \wedge 0 \leq b \wedge a < \infty \wedge b < \infty"

    # 断言语句，验证 RandomDomain 对象的 LaTeX 表达式
    assert latex(RandomDomain(FiniteSet(x), FiniteSet(1, 2))) == \
        r'\text{Domain: }\left\{x\right\} \in \left\{1, 2\right\}'


def test_PrettyPoly():
    # 导入多项式环 QQ 和相关变量 x, y
    from sympy.polys.domains import QQ
    # 创建有理函数域 F 和多项式环 R
    F = QQ.frac_field(x, y)
    R = QQ[x, y]

    # 断言语句，验证 F.convert(x/(x + y)) 的 LaTeX 表达式
    assert latex(F.convert(x/(x + y))) == latex(x/(x + y))
    # 断言语句，验证 R.convert(x + y) 的 LaTeX 表达式
    assert latex(R.convert(x + y)) == latex(x + y)


def test_integral_transforms():
    # 导入符号相关模块和函数
    x = Symbol("x")
    k = Symbol("k")
    f = Function("f")
    a = Symbol("a")
    b = Symbol("b")

    # 断言语句，验证 MellinTransform 的 LaTeX 表达式
    assert latex(MellinTransform(f(x), x, k)) == \
        r"\mathcal{M}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    # 断言语句，验证 InverseMellinTransform 的 LaTeX 表达式
    assert latex(InverseMellinTransform(f(k), k, x, a, b)) == \
        r"\mathcal{M}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    # 断言语句，验证 LaplaceTransform 的 LaTeX 表达式
    assert latex(LaplaceTransform(f(x), x, k)) == \
        r"\mathcal{L}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    # 断言语句，验证 InverseLaplaceTransform 的 LaTeX 表达式
    assert latex(InverseLaplaceTransform(f(k), k, x, (a, b))) == \
        r"\mathcal{L}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    # 断言语句，验证 FourierTransform 的 LaTeX 表达式
    assert latex(FourierTransform(f(x), x, k)) == \
        r"\mathcal{F}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    # 断言语句，验证 InverseFourierTransform 的 LaTeX 表达式
    assert latex(InverseFourierTransform(f(k), k, x)) == \
        r"\mathcal{F}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    # 断言语句，验证 CosineTransform 的 LaTeX 表达式
    assert latex(CosineTransform(f(x), x, k)) == \
        r"\mathcal{COS}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    # 断言语句，验证 InverseCosineTransform 的 LaTeX 表达式
    assert latex(InverseCosineTransform(f(k), k, x)) == \
        r"\mathcal{COS}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    # 断言语句，验证 SineTransform 的 LaTeX 表达式
    assert latex(SineTransform(f(x), x, k)) == \
        r"\mathcal{SIN}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    # 断言语句，验证 InverseSineTransform 的 LaTeX 表达式
    assert latex(InverseSineTransform(f(k), k, x)) == \
        r"\mathcal{SIN}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"


def test_PolynomialRingBase():
    # 导入多项式环 QQ
    from sympy.polys.domains import QQ
    # 断言语句，验证 QQ.old_poly_ring(x, y) 的 LaTeX 表达式
    assert latex(QQ.old_poly_ring(x, y)) == r"\mathbb{Q}\left[x, y\right]"
    # 断言语句，验证 QQ.old_poly_ring(x, y, order="ilex") 的 LaTeX 表达式
    assert latex(QQ.old_poly_ring(x, y, order="ilex")) == \
        r"S_<^{-1}\mathbb{Q}\left[x, y\right]"


def test_categories():
    # 导入范畴相关模块和类
    from sympy.categories import (Object, IdentityMorphism,
                                  NamedMorphism, Category, Diagram,
                                  DiagramGrid)

    # 创建三个对象 A1, A2, A3
    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")

    # 创建从 A1 到 A2 的命名态射 f1
    f1 = NamedMorphism(A1, A2, "f1")
    # 创建从 A2 到 A3 的命名态射 f2
    f2 = NamedMorphism(A2, A3, "f2")
    # 创建 A1 的恒同态射
    id_A1 = IdentityMorphism(A1)
    # 创建一个名为 K1 的类别对象
    K1 = Category("K1")
    
    # 断言语句，验证 LaTeX 表达式是否符合预期
    assert latex(A1) == r"A_{1}"
    assert latex(f1) == r"f_{1}:A_{1}\rightarrow A_{2}"
    assert latex(id_A1) == r"id:A_{1}\rightarrow A_{1}"
    assert latex(f2*f1) == r"f_{2}\circ f_{1}:A_{1}\rightarrow A_{3}"
    
    # 断言语句，验证 LaTeX 表达式是否符合预期
    assert latex(K1) == r"\mathbf{K_{1}}"
    
    # 创建一个空的图表对象 d
    d = Diagram()
    # 断言语句，验证 LaTeX 表达式是否符合预期
    assert latex(d) == r"\emptyset"
    
    # 创建一个带有映射和标签的图表对象 d
    d = Diagram({f1: "unique", f2: S.EmptySet})
    # 断言语句，验证 LaTeX 表达式是否符合预期
    assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}, " \
        r"\  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}"
    
    # 创建一个带有映射、标签和关系的图表对象 d
    d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
    # 断言语句，验证 LaTeX 表达式是否符合预期
    assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}," \
        r" \  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}" \
        r"\Longrightarrow \left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \left\{unique\right\}\right\}"
    
    # 创建对象 A, B, C，并创建名为 f 和 g 的映射
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    # 创建一个包含 f 和 g 映射的图表对象 d
    d = Diagram([f, g])
    # 创建一个图表网格对象 grid
    grid = DiagramGrid(d)
    
    # 断言语句，验证 LaTeX 表达式是否符合预期
    assert latex(grid) == r"\begin{array}{cc}" + "\n" \
        r"A & B \\" + "\n" \
        r" & C " + "\n" \
        r"\end{array}" + "\n"
# 定义测试函数 test_Modules
def test_Modules():
    # 从 sympy.polys.domains 导入 QQ
    from sympy.polys.domains import QQ
    # 从 sympy.polys.agca 导入 homomorphism
    from sympy.polys.agca import homomorphism

    # 创建 QQ.old_poly_ring(x, y) 对象并赋值给 R
    R = QQ.old_poly_ring(x, y)
    # 使用 R 创建自由模 F，维度为 2
    F = R.free_module(2)
    # 使用 F 创建子模 M，基向量为 [x, y]，系数为 [1, x**2]
    M = F.submodule([x, y], [1, x**2])

    # 断言 F 的 LaTeX 表示
    assert latex(F) == r"{\mathbb{Q}\left[x, y\right]}^{2}"
    # 断言 M 的 LaTeX 表示
    assert latex(M) == \
        r"\left\langle {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle"

    # 使用 R 创建理想 I，包含元素 x**2 和 y
    I = R.ideal(x**2, y)
    # 断言 I 的 LaTeX 表示
    assert latex(I) == r"\left\langle {x^{2}},{y} \right\rangle"

    # 创建商模 Q，等于 F 模除以 M
    Q = F / M
    # 断言 Q 的 LaTeX 表示
    assert latex(Q) == \
        r"\frac{{\mathbb{Q}\left[x, y\right]}^{2}}{\left\langle {\left[ {x},"\
        r"{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle}"
    # 断言 Q 的子模表达式的 LaTeX 表示
    assert latex(Q.submodule([1, x**3/2], [2, y])) == \
        r"\left\langle {{\left[ {1},{\frac{x^{3}}{2}} \right]} + {\left"\
        r"\langle {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} "\
        r"\right\rangle}},{{\left[ {2},{y} \right]} + {\left\langle {\left[ "\
        r"{x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle}} \right\rangle"

    # 使用 homomorphism 函数创建映射 h
    h = homomorphism(QQ.old_poly_ring(x).free_module(2),
                     QQ.old_poly_ring(x).free_module(2), [0, 0])

    # 断言 h 的 LaTeX 表示
    assert latex(h) == \
        r"{\left[\begin{matrix}0 & 0\\0 & 0\end{matrix}\right]} : "\
        r"{{\mathbb{Q}\left[x\right]}^{2}} \to {{\mathbb{Q}\left[x\right]}^{2}}"


# 定义测试函数 test_QuotientRing
def test_QuotientRing():
    # 从 sympy.polys.domains 导入 QQ
    from sympy.polys.domains import QQ
    # 创建 R，等于 QQ.old_poly_ring(x) 模除以理想 [x**2 + 1]
    R = QQ.old_poly_ring(x)/[x**2 + 1]

    # 断言 R 的 LaTeX 表示
    assert latex(R) == \
        r"\frac{\mathbb{Q}\left[x\right]}{\left\langle {x^{2} + 1} \right\rangle}"
    # 断言 R 的单位元的 LaTeX 表示
    assert latex(R.one) == r"{1} + {\left\langle {x^{2} + 1} \right\rangle}"


# 定义测试函数 test_Tr
def test_Tr():
    # TODO: 处理指标
    # 创建符号 A 和 B，非交换
    A, B = symbols('A B', commutative=False)
    # 计算矩阵 A*B 的迹 t
    t = Tr(A*B)
    # 断言 t 的 LaTeX 表示
    assert latex(t) == r'\operatorname{tr}\left(A B\right)'


# 定义测试函数 test_Determinant
def test_Determinant():
    # 从 sympy.matrices 导入 Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    # 创建 2x2 矩阵 m
    m = Matrix(((1, 2), (3, 4)))
    # 断言 Determinant(m) 的 LaTeX 表示
    assert latex(Determinant(m)) == '\\left|{\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}}\\right|'
    # 断言 Determinant(Inverse(m)) 的 LaTeX 表示
    assert latex(Determinant(Inverse(m))) == \
        '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{-1}}\\right|'
    # 创建符号矩阵 X
    X = MatrixSymbol('X', 2, 2)
    # 断言 Determinant(X) 的 LaTeX 表示
    assert latex(Determinant(X)) == '\\left|{X}\\right|'
    # 断言 Determinant(X + m) 的 LaTeX 表示
    assert latex(Determinant(X + m)) == \
        '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X}\\right|'
    # 断言 Determinant(BlockMatrix(...)) 的 LaTeX 表示
    assert latex(Determinant(BlockMatrix(((OneMatrix(2, 2), X),
                                          (m, ZeroMatrix(2, 2)))))) == \
        '\\left|{\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}}\\right|'


# 定义测试函数 test_Adjoint
def test_Adjoint():
    # 从 sympy.matrices 导入 Adjoint, Inverse, Transpose
    from sympy.matrices import Adjoint, Inverse, Transpose
    # 创建符号矩阵 X 和 Y
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 断言 Adjoint(X) 的 LaTeX 表示
    assert latex(Adjoint(X)) == r'X^{\dagger}'
    # 断言 Adjoint(X + Y) 的 LaTeX 表示
    assert latex(Adjoint(X + Y)) == r'\left(X + Y\right)^{\dagger}'
    # 断言 Adjoint(X) + Adjoint(Y) 的 LaTeX 表示
    assert latex(Adjoint(X) + Adjoint(Y)) == r'X^{\dagger} + Y^{\dagger}'
    # 断言，验证 Adjoint(X*Y) 的 LaTeX 表示是否等于 r'\left(X Y\right)^{\dagger}'
    assert latex(Adjoint(X*Y)) == r'\left(X Y\right)^{\dagger}'
    
    # 断言，验证 Adjoint(Y)*Adjoint(X) 的 LaTeX 表示是否等于 r'Y^{\dagger} X^{\dagger}'
    assert latex(Adjoint(Y)*Adjoint(X)) == r'Y^{\dagger} X^{\dagger}'
    
    # 断言，验证 Adjoint(X**2) 的 LaTeX 表示是否等于 r'\left(X^{2}\right)^{\dagger}'
    assert latex(Adjoint(X**2)) == r'\left(X^{2}\right)^{\dagger}'
    
    # 断言，验证 Adjoint(X)**2 的 LaTeX 表示是否等于 r'\left(X^{\dagger}\right)^{2}'
    assert latex(Adjoint(X)**2) == r'\left(X^{\dagger}\right)^{2}'
    
    # 断言，验证 Adjoint(Inverse(X)) 的 LaTeX 表示是否等于 r'\left(X^{-1}\right)^{\dagger}'
    assert latex(Adjoint(Inverse(X))) == r'\left(X^{-1}\right)^{\dagger}'
    
    # 断言，验证 Inverse(Adjoint(X)) 的 LaTeX 表示是否等于 r'\left(X^{\dagger}\right)^{-1}'
    assert latex(Inverse(Adjoint(X))) == r'\left(X^{\dagger}\right)^{-1}'
    
    # 断言，验证 Adjoint(Transpose(X)) 的 LaTeX 表示是否等于 r'\left(X^{T}\right)^{\dagger}'
    assert latex(Adjoint(Transpose(X))) == r'\left(X^{T}\right)^{\dagger}'
    
    # 断言，验证 Transpose(Adjoint(X)) 的 LaTeX 表示是否等于 r'\left(X^{\dagger}\right)^{T}'
    assert latex(Transpose(Adjoint(X))) == r'\left(X^{\dagger}\right)^{T}'
    
    # 断言，验证 Transpose(Adjoint(X) + Y) 的 LaTeX 表示是否等于 r'\left(X^{\dagger} + Y\right)^{T}'
    assert latex(Transpose(Adjoint(X) + Y)) == r'\left(X^{\dagger} + Y\right)^{T}'
    
    # 创建矩阵 m，并验证 Adjoint(m) 的 LaTeX 表示是否等于 '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{\\dagger}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Adjoint(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{\\dagger}'
    
    # 断言，验证 Adjoint(m+X) 的 LaTeX 表示是否等于特定格式
    assert latex(Adjoint(m+X)) == \
        '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{\\dagger}'
    
    # 导入 BlockMatrix、OneMatrix 和 ZeroMatrix，创建复合矩阵并验证其 Adjoint 的 LaTeX 表示
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Adjoint(BlockMatrix(((OneMatrix(2, 2), X),
                                      (m, ZeroMatrix(2, 2)))))) == \
        '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{\\dagger}'
    
    # 断言，验证 MatrixSymbol Mx 的 Adjoint 的 LaTeX 表示是否等于 r'\left(M^{x}\right)^{\dagger}'
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Adjoint(Mx)) == r'\left(M^{x}\right)^{\dagger}'
    
    # 使用不同的 adjoint_style 参数，验证不同表示风格的 Adjoint 的 LaTeX 表示是否正确
    assert latex(Adjoint(X), adjoint_style="star") == r'X^{\ast}'
    assert latex(Adjoint(X + Y), adjoint_style="hermitian") == r'\left(X + Y\right)^{\mathsf{H}}'
    assert latex(Adjoint(X) + Adjoint(Y), adjoint_style="dagger") == r'X^{\dagger} + Y^{\dagger}'
    assert latex(Adjoint(Y)*Adjoint(X)) == r'Y^{\dagger} X^{\dagger}'
    assert latex(Adjoint(X**2), adjoint_style="star") == r'\left(X^{2}\right)^{\ast}'
    assert latex(Adjoint(X)**2, adjoint_style="hermitian") == r'\left(X^{\mathsf{H}}\right)^{2}'
def test_Transpose():
    # 导入需要的模块和类：Transpose, MatPow, HadamardPower
    from sympy.matrices import Transpose, MatPow, HadamardPower
    # 定义符号矩阵 X 和 Y，每个都是 2x2 的矩阵符号
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 断言转置后的 LaTeX 表示等于 'X^{T}'
    assert latex(Transpose(X)) == r'X^{T}'
    # 断言转置后的 LaTeX 表示等于 '\left(X + Y\right)^{T}'
    assert latex(Transpose(X + Y)) == r'\left(X + Y\right)^{T}'

    # 断言对 HadamardPower(X, 2) 的转置后的 LaTeX 表示等于 '\left(X^{\circ {2}}\right)^{T}'
    assert latex(Transpose(HadamardPower(X, 2))) == r'\left(X^{\circ {2}}\right)^{T}'
    # 断言对 Transpose(X) 的 HadamardPower(2) 的 LaTeX 表示等于 '\left(X^{T}\right)^{\circ {2}}'
    assert latex(HadamardPower(Transpose(X), 2)) == r'\left(X^{T}\right)^{\circ {2}}'
    # 断言对 MatPow(X, 2) 的转置后的 LaTeX 表示等于 '\left(X^{2}\right)^{T}'
    assert latex(Transpose(MatPow(X, 2))) == r'\left(X^{2}\right)^{T}'
    # 断言对 Transpose(X) 的 MatPow(2) 的 LaTeX 表示等于 '\left(X^{T}\right)^{2}'
    assert latex(MatPow(Transpose(X), 2)) == r'\left(X^{T}\right)^{2}'
    
    # 创建一个普通的矩阵 m
    m = Matrix(((1, 2), (3, 4)))
    # 断言矩阵 m 的转置后的 LaTeX 表示
    assert latex(Transpose(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{T}'
    # 断言矩阵 m 与 X 的和的转置后的 LaTeX 表示
    assert latex(Transpose(m+X)) == \
        '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{T}'
    
    # 导入 BlockMatrix, OneMatrix, ZeroMatrix 类
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    # 断言对 BlockMatrix 的转置后的 LaTeX 表示
    assert latex(Transpose(BlockMatrix(((OneMatrix(2, 2), X),
                                        (m, ZeroMatrix(2, 2)))))) == \
        '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{T}'
    # Issue 20959
    # 定义一个具有特殊符号的矩阵 Mx
    Mx = MatrixSymbol('M^x', 2, 2)
    # 断言对 Mx 的转置后的 LaTeX 表示
    assert latex(Transpose(Mx)) == r'\left(M^{x}\right)^{T}'


def test_Hadamard():
    # 导入需要的模块和类：HadamardProduct, HadamardPower
    from sympy.matrices import HadamardProduct, HadamardPower
    # 导入需要的表达式类：MatAdd, MatMul, MatPow
    from sympy.matrices.expressions import MatAdd, MatMul, MatPow
    # 定义符号矩阵 X 和 Y，每个都是 2x2 的矩阵符号
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 断言 HadamardProduct(X, Y*Y) 的 LaTeX 表示
    assert latex(HadamardProduct(X, Y*Y)) == r'X \circ Y^{2}'
    # 断言 HadamardProduct(X, Y)*Y 的 LaTeX 表示
    assert latex(HadamardProduct(X, Y)*Y) == r'\left(X \circ Y\right) Y'

    # 断言 HadamardPower(X, 2) 的 LaTeX 表示
    assert latex(HadamardPower(X, 2)) == r'X^{\circ {2}}'
    # 断言 HadamardPower(X, -1) 的 LaTeX 表示
    assert latex(HadamardPower(X, -1)) == r'X^{\circ \left({-1}\right)}'
    # 断言 HadamardPower(MatAdd(X, Y), 2) 的 LaTeX 表示
    assert latex(HadamardPower(MatAdd(X, Y), 2)) == \
        r'\left(X + Y\right)^{\circ {2}}'
    # 断言 HadamardPower(MatMul(X, Y), 2) 的 LaTeX 表示
    assert latex(HadamardPower(MatMul(X, Y), 2)) == \
        r'\left(X Y\right)^{\circ {2}}'

    # 断言 HadamardPower(MatPow(X, -1), -1) 的 LaTeX 表示
    assert latex(HadamardPower(MatPow(X, -1), -1)) == \
        r'\left(X^{-1}\right)^{\circ \left({-1}\right)}'
    # 断言 MatPow(HadamardPower(X, -1), -1) 的 LaTeX 表示
    assert latex(MatPow(HadamardPower(X, -1), -1)) == \
        r'\left(X^{\circ \left({-1}\right)}\right)^{-1}'

    # 断言 HadamardPower(X, n+1) 的 LaTeX 表示
    assert latex(HadamardPower(X, n+1)) == \
        r'X^{\circ \left({n + 1}\right)}'


def test_MatPow():
    # 导入需要的模块和类：MatPow
    from sympy.matrices.expressions import MatPow
    # 定义符号矩阵 X 和 Y，每个都是 2x2 的矩阵符号
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    # 断言 MatPow(X, 2) 的 LaTeX 表示
    assert latex(MatPow(X, 2)) == 'X^{2}'
    # 断言 MatPow(X*X, 2) 的 LaTeX 表示
    assert latex(MatPow(X*X, 2)) == '\\left(X^{2}\\right)^{2}'
    # 断言 MatPow(X*Y, 2) 的 LaTeX 表示
    assert latex(MatPow(X*Y, 2)) == '\\left(X Y\\right)^{2}'
    # 断言 MatPow(X + Y, 2) 的 LaTeX 表示
    assert latex(MatPow(X + Y, 2)) == '\\left(X + Y\\right)^{2}'
    # 断言 MatPow(X + X, 2) 的 LaTeX 表示
    assert latex(MatPow(X + X, 2)) == '\\left(2 X\\right)^{2}'
    # Issue 20959
    # 定义一个具有特殊符号的矩阵 Mx
    Mx = MatrixSymbol('M^x', 2, 2)
    # 断言对 Mx 的 MatPow(2) 的 LaTeX 表示
    assert latex(MatPow(Mx, 2)) == r'\left(M^{x}\right)^{2}'


def test_ElementwiseApplyFunction():
    # 定义符号矩阵 X，是一个 2x2 的矩阵符号
    X = MatrixSymbol('X', 2, 2)
    # 创建一个表达式，对矩阵 X 进行转置后与自身相乘，并应用 sin 函数
    expr = (X.T*X).applyfunc(sin)
    # 断言表达式，验证 latex(expr) 是否等于指定的 LaTeX 表示形式
    assert latex(expr) == r"{\left( d \mapsto \sin{\left(d \right)} \right)}_{\circ}\left({X^{T} X}\right)"
    
    # 使用 Lambda 函数将矩阵 X 中的每个元素 x 替换为 1/x，得到新的表达式 expr
    expr = X.applyfunc(Lambda(x, 1/x))
    
    # 再次断言表达式，验证 latex(expr) 是否等于新的指定 LaTeX 表示形式
    assert latex(expr) == r'{\left( x \mapsto \frac{1}{x} \right)}_{\circ}\left({X}\right)'
def test_ZeroMatrix():
    # 导入零矩阵类并验证生成的 LaTeX 表示是否符合预期
    from sympy.matrices.expressions.special import ZeroMatrix
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='plain') == r"0"
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='bold') == r"\mathbf{0}"


def test_OneMatrix():
    # 导入单位矩阵类并验证生成的 LaTeX 表示是否符合预期
    from sympy.matrices.expressions.special import OneMatrix
    assert latex(OneMatrix(3, 4), mat_symbol_style='plain') == r"1"
    assert latex(OneMatrix(3, 4), mat_symbol_style='bold') == r"\mathbf{1}"


def test_Identity():
    # 导入单位矩阵类并验证生成的 LaTeX 表示是否符合预期
    from sympy.matrices.expressions.special import Identity
    assert latex(Identity(1), mat_symbol_style='plain') == r"\mathbb{I}"
    assert latex(Identity(1), mat_symbol_style='bold') == r"\mathbf{I}"


def test_latex_DFT_IDFT():
    # 导入傅里叶变换相关类并验证生成的 LaTeX 表示是否符合预期
    from sympy.matrices.expressions.fourier import DFT, IDFT
    assert latex(DFT(13)) == r"\text{DFT}_{13}"
    assert latex(IDFT(x)) == r"\text{IDFT}_{x}"


def test_boolean_args_order():
    # 创建布尔逻辑表达式并验证生成的 LaTeX 表示是否符合预期
    syms = symbols('a:f')

    expr = And(*syms)
    assert latex(expr) == r'a \wedge b \wedge c \wedge d \wedge e \wedge f'

    expr = Or(*syms)
    assert latex(expr) == r'a \vee b \vee c \vee d \vee e \vee f'

    expr = Equivalent(*syms)
    assert latex(expr) == \
        r'a \Leftrightarrow b \Leftrightarrow c \Leftrightarrow d \Leftrightarrow e \Leftrightarrow f'

    expr = Xor(*syms)
    assert latex(expr) == \
        r'a \veebar b \veebar c \veebar d \veebar e \veebar f'


def test_imaginary():
    # 创建虚数并验证生成的 LaTeX 表示是否符合预期
    i = sqrt(-1)
    assert latex(i) == r'i'


def test_builtins_without_args():
    # 验证内置函数的 LaTeX 表示是否符合预期
    assert latex(sin) == r'\sin'
    assert latex(cos) == r'\cos'
    assert latex(tan) == r'\tan'
    assert latex(log) == r'\log'
    assert latex(Ei) == r'\operatorname{Ei}'
    assert latex(zeta) == r'\zeta'


def test_latex_greek_functions():
    # 验证希腊字母函数的 LaTeX 表示是否符合预期
    # 注：某些函数名在 LaTeX 中有特殊用途或者需要特殊处理
    s = Function('Alpha')
    assert latex(s) == r'\mathrm{A}'
    assert latex(s(x)) == r'\mathrm{A}{\left(x \right)}'
    s = Function('Beta')
    assert latex(s) == r'\mathrm{B}'
    s = Function('Eta')
    assert latex(s) == r'\mathrm{H}'
    assert latex(s(x)) == r'\mathrm{H}{\left(x \right)}'

    p = Function('Pi')
    assert latex(p) == r'\Pi'

    c = Function('chi')
    assert latex(c(x)) == r'\chi{\left(x \right)}'
    assert latex(c) == r'\chi'


def test_translate():
    # 验证翻译函数对特定字符串的 LaTeX 表示是否符合预期
    s = 'Alpha'
    assert translate(s) == r'\mathrm{A}'
    s = 'Beta'
    assert translate(s) == r'\mathrm{B}'
    s = 'Eta'
    assert translate(s) == r'\mathrm{H}'
    s = 'omicron'
    assert translate(s) == r'o'
    s = 'Pi'
    assert translate(s) == r'\Pi'
    s = 'pi'
    assert translate(s) == r'\pi'
    s = 'LamdaHatDOT'
    assert translate(s) == r'\dot{\hat{\Lambda}}'


def test_other_symbols():
    # 导入其他符号并检查是否正确导入
    from sympy.printing.latex import other_symbols
    for s in other_symbols:
        # 对于每个符号 s 在 other_symbols 中
        # 使用 symbols 函数获取其 LaTeX 表示，然后与预期的字符串进行断言比较
        assert latex(symbols(s)) == r"" "\\" + s
def test_modifiers():
    # Test each modifier individually in the simplest case
    # (with funny capitalizations)
    # 检查每个修饰符在最简单的情况下的转换效果（包括奇怪的大小写）
    assert latex(symbols("xMathring")) == r"\mathring{x}"
    # 断言：将字符串"xMathring"转换为LaTeX格式"\mathring{x}"
    assert latex(symbols("xCheck")) == r"\check{x}"
    # 断言：将字符串"xCheck"转换为LaTeX格式"\check{x}"
    assert latex(symbols("xBreve")) == r"\breve{x}"
    # 断言：将字符串"xBreve"转换为LaTeX格式"\breve{x}"
    assert latex(symbols("xAcute")) == r"\acute{x}"
    # 断言：将字符串"xAcute"转换为LaTeX格式"\acute{x}"
    assert latex(symbols("xGrave")) == r"\grave{x}"
    # 断言：将字符串"xGrave"转换为LaTeX格式"\grave{x}"
    assert latex(symbols("xTilde")) == r"\tilde{x}"
    # 断言：将字符串"xTilde"转换为LaTeX格式"\tilde{x}"
    assert latex(symbols("xPrime")) == r"{x}'"
    # 断言：将字符串"xPrime"转换为LaTeX格式"{x}'"
    assert latex(symbols("xddDDot")) == r"\ddddot{x}"
    # 断言：将字符串"xddDDot"转换为LaTeX格式"\ddddot{x}"
    assert latex(symbols("xDdDot")) == r"\dddot{x}"
    # 断言：将字符串"xDdDot"转换为LaTeX格式"\dddot{x}"
    assert latex(symbols("xDDot")) == r"\ddot{x}"
    # 断言：将字符串"xDDot"转换为LaTeX格式"\ddot{x}"
    assert latex(symbols("xBold")) == r"\boldsymbol{x}"
    # 断言：将字符串"xBold"转换为LaTeX格式"\boldsymbol{x}"
    assert latex(symbols("xnOrM")) == r"\left\|{x}\right\|"
    # 断言：将字符串"xnOrM"转换为LaTeX格式"\left\|{x}\right\|"
    assert latex(symbols("xAVG")) == r"\left\langle{x}\right\rangle"
    # 断言：将字符串"xAVG"转换为LaTeX格式"\left\langle{x}\right\rangle"
    assert latex(symbols("xHat")) == r"\hat{x}"
    # 断言：将字符串"xHat"转换为LaTeX格式"\hat{x}"
    assert latex(symbols("xDot")) == r"\dot{x}"
    # 断言：将字符串"xDot"转换为LaTeX格式"\dot{x}"
    assert latex(symbols("xBar")) == r"\bar{x}"
    # 断言：将字符串"xBar"转换为LaTeX格式"\bar{x}"
    assert latex(symbols("xVec")) == r"\vec{x}"
    # 断言：将字符串"xVec"转换为LaTeX格式"\vec{x}"
    assert latex(symbols("xAbs")) == r"\left|{x}\right|"
    # 断言：将字符串"xAbs"转换为LaTeX格式"\left|{x}\right|"
    assert latex(symbols("xMag")) == r"\left|{x}\right|"
    # 断言：将字符串"xMag"转换为LaTeX格式"\left|{x}\right|"
    assert latex(symbols("xPrM")) == r"{x}'"
    # 断言：将字符串"xPrM"转换为LaTeX格式"{x}'"
    assert latex(symbols("xBM")) == r"\boldsymbol{x}"
    # 断言：将字符串"xBM"转换为LaTeX格式"\boldsymbol{x}"
    # Test strings that are *only* the names of modifiers
    # 测试仅为修饰符名称的字符串
    assert latex(symbols("Mathring")) == r"Mathring"
    # 断言：将字符串"Mathring"转换为LaTeX格式"Mathring"
    assert latex(symbols("Check")) == r"Check"
    # 断言：将字符串"Check"转换为LaTeX格式"Check"
    assert latex(symbols("Breve")) == r"Breve"
    # 断言：将字符串"Breve"转换为LaTeX格式"Breve"
    assert latex(symbols("Acute")) == r"Acute"
    # 断言：将字符串"Acute"转换为LaTeX格式"Acute"
    assert latex(symbols("Grave")) == r"Grave"
    # 断言：将字符串"Grave"转换为LaTeX格式"Grave"
    assert latex(symbols("Tilde")) == r"Tilde"
    # 断言：将字符串"Tilde"转换为LaTeX格式"Tilde"
    assert latex(symbols("Prime")) == r"Prime"
    # 断言：将字符串"Prime"转换为LaTeX格式"Prime"
    assert latex(symbols("DDot")) == r"\dot{D}"
    # 断言：将字符串"DDot"转换为LaTeX格式"\dot{D}"
    assert latex(symbols("Bold")) == r"Bold"
    # 断言：将字符串"Bold"转换为LaTeX格式"Bold"
    assert latex(symbols("NORm")) == r"NORm"
    # 断言：将字符串"NORm"转换为LaTeX格式"NORm"
    assert latex(symbols("AVG")) == r"AVG"
    # 断言：将字符串"AVG"转换为LaTeX格式"AVG"
    assert latex(symbols("Hat")) == r"Hat"
    # 断言：将字符串"Hat"转换为LaTeX格式"Hat"
    assert latex(symbols("Dot")) == r"Dot"
    # 断言：将字符串"Dot"转换为LaTeX格式"Dot"
    assert latex(symbols("Bar")) == r"Bar"
    # 断言：将字符串"Bar"转换为LaTeX格式"Bar"
    assert latex(symbols("Vec")) == r"Vec"
    # 断言：将字符串"Vec"转换为LaTeX格式"Vec"
    assert latex(symbols("Abs")) == r"Abs"
    # 断言：将字符串"Abs"转换为LaTeX格式"Abs"
    assert latex(symbols("Mag")) == r"Mag"
    # 断言：将字符串"Mag"转换为LaTeX格式"Mag"
    assert latex(symbols("PrM")) == r"PrM"
    # 断言：将字符串"PrM"转换为LaTeX格式"PrM"
    assert latex(symbols("BM")) == r"BM"
    # 断言：将字符串"BM"转换为LaTeX格式"BM"
    assert latex(symbols("hbar")) == r"\hbar"
    # 断言：将字符串"hbar"转换为LaTeX格式"\hbar"
    # Check a few combinations
    # 检查一些组合情况
    assert latex(symbols("xvecdot")) == r"\dot{\vec{x}}"
    # 断言：将字符串"xvecdot"转换为LaTeX格式"\dot{\vec{x}}"
    assert latex(symbols("xDotVec")) == r"\vec{\dot{x}}"
    # 断言：将字符串"xDotVec"转换为LaTeX格式"\vec{\dot{x}}"
    assert latex(symbols("xHATNorm")) == r"\left\|{\hat{x}}\right\|"
    # 断言：将字符串"xHATNorm"转换为LaTeX格式"\left\|{\hat{x}}\right\|"
    # Check a couple big, ugly combinations
    # 检查几个复杂的组合情况
    assert latex(symbols('xMathringBm_yCheckPRM__zbreveAbs')) == \
        r"\boldsymbol{\mathring{x}}^{\left|{\breve{z}}\right|}_{{\check{y}}'}"
    # 断言：将字符串'xMathringBm_yCheckPRM__zbreveAbs'转换为对应的LaTeX格式
    assert latex(symbols('alphadothat_nVECDOT__tTildePrime')) == \
        r"\hat{\dot{\alpha}}^{{\tilde{t}}'}_{\dot{\vec{n}}}"
    # 断言：将字符串'alphadothat_nVECDOT__tTildePrime'转换为对应的LaTeX格式


def test_greek_symbols():
    assert latex(Symbol('alpha'))   == r'\alpha'
    # 断言：将符号'alpha'转换为LaTeX格式'\alpha'
    assert latex(Symbol('beta'))
    # 确认符号'zeta'的LaTeX表示正确
    assert latex(Symbol('zeta'))    == r'\zeta'
    # 确认符号'eta'的LaTeX表示正确
    assert latex(Symbol('eta'))     == r'\eta'
    # 确认符号'theta'的LaTeX表示正确
    assert latex(Symbol('theta'))   == r'\theta'
    # 确认符号'iota'的LaTeX表示正确
    assert latex(Symbol('iota'))    == r'\iota'
    # 确认符号'kappa'的LaTeX表示正确
    assert latex(Symbol('kappa'))   == r'\kappa'
    # 确认符号'lambda'的LaTeX表示正确
    assert latex(Symbol('lambda'))  == r'\lambda'
    # 确认符号'mu'的LaTeX表示正确
    assert latex(Symbol('mu'))      == r'\mu'
    # 确认符号'nu'的LaTeX表示正确
    assert latex(Symbol('nu'))      == r'\nu'
    # 确认符号'xi'的LaTeX表示正确
    assert latex(Symbol('xi'))      == r'\xi'
    # 确认符号'omicron'的LaTeX表示正确
    assert latex(Symbol('omicron')) == r'o'
    # 确认符号'pi'的LaTeX表示正确
    assert latex(Symbol('pi'))      == r'\pi'
    # 确认符号'rho'的LaTeX表示正确
    assert latex(Symbol('rho'))     == r'\rho'
    # 确认符号'sigma'的LaTeX表示正确
    assert latex(Symbol('sigma'))   == r'\sigma'
    # 确认符号'tau'的LaTeX表示正确
    assert latex(Symbol('tau'))     == r'\tau'
    # 确认符号'upsilon'的LaTeX表示正确
    assert latex(Symbol('upsilon')) == r'\upsilon'
    # 确认符号'phi'的LaTeX表示正确
    assert latex(Symbol('phi'))     == r'\phi'
    # 确认符号'chi'的LaTeX表示正确
    assert latex(Symbol('chi'))     == r'\chi'
    # 确认符号'psi'的LaTeX表示正确
    assert latex(Symbol('psi'))     == r'\psi'
    # 确认符号'omega'的LaTeX表示正确
    assert latex(Symbol('omega'))   == r'\omega'
    
    # 确认符号'Alpha'的LaTeX表示正确
    assert latex(Symbol('Alpha'))   == r'\mathrm{A}'
    # 确认符号'Beta'的LaTeX表示正确
    assert latex(Symbol('Beta'))    == r'\mathrm{B}'
    # 确认符号'Gamma'的LaTeX表示正确
    assert latex(Symbol('Gamma'))   == r'\Gamma'
    # 确认符号'Delta'的LaTeX表示正确
    assert latex(Symbol('Delta'))   == r'\Delta'
    # 确认符号'Epsilon'的LaTeX表示正确
    assert latex(Symbol('Epsilon')) == r'\mathrm{E}'
    # 确认符号'Zeta'的LaTeX表示正确
    assert latex(Symbol('Zeta'))    == r'\mathrm{Z}'
    # 确认符号'Eta'的LaTeX表示正确
    assert latex(Symbol('Eta'))     == r'\mathrm{H}'
    # 确认符号'Theta'的LaTeX表示正确
    assert latex(Symbol('Theta'))   == r'\Theta'
    # 确认符号'Iota'的LaTeX表示正确
    assert latex(Symbol('Iota'))    == r'\mathrm{I}'
    # 确认符号'Kappa'的LaTeX表示正确
    assert latex(Symbol('Kappa'))   == r'\mathrm{K}'
    # 确认符号'Lambda'的LaTeX表示正确
    assert latex(Symbol('Lambda'))  == r'\Lambda'
    # 确认符号'Mu'的LaTeX表示正确
    assert latex(Symbol('Mu'))      == r'\mathrm{M}'
    # 确认符号'Nu'的LaTeX表示正确
    assert latex(Symbol('Nu'))      == r'\mathrm{N}'
    # 确认符号'Xi'的LaTeX表示正确
    assert latex(Symbol('Xi'))      == r'\Xi'
    # 确认符号'Omicron'的LaTeX表示正确
    assert latex(Symbol('Omicron')) == r'\mathrm{O}'
    # 确认符号'Pi'的LaTeX表示正确
    assert latex(Symbol('Pi'))      == r'\Pi'
    # 确认符号'Rho'的LaTeX表示正确
    assert latex(Symbol('Rho'))     == r'\mathrm{P}'
    # 确认符号'Sigma'的LaTeX表示正确
    assert latex(Symbol('Sigma'))   == r'\Sigma'
    # 确认符号'Tau'的LaTeX表示正确
    assert latex(Symbol('Tau'))     == r'\mathrm{T}'
    # 确认符号'Upsilon'的LaTeX表示正确
    assert latex(Symbol('Upsilon')) == r'\Upsilon'
    # 确认符号'Phi'的LaTeX表示正确
    assert latex(Symbol('Phi'))     == r'\Phi'
    # 确认符号'Chi'的LaTeX表示正确
    assert latex(Symbol('Chi'))     == r'\mathrm{X}'
    # 确认符号'Psi'的LaTeX表示正确
    assert latex(Symbol('Psi'))     == r'\Psi'
    # 确认符号'Omega'的LaTeX表示正确
    assert latex(Symbol('Omega'))   == r'\Omega'
    
    # 确认符号'varepsilon'的LaTeX表示正确
    assert latex(Symbol('varepsilon')) == r'\varepsilon'
    # 确认符号'varkappa'的LaTeX表示正确
    assert latex(Symbol('varkappa')) == r'\varkappa'
    # 确认符号'varphi'的LaTeX表示正确
    assert latex(Symbol('varphi')) == r'\varphi'
    # 确认符号'varpi'的LaTeX表示正确
    assert latex(Symbol('varpi')) == r'\varpi'
    # 确认符号'varrho'的LaTeX表示正确
    assert latex(Symbol('varrho')) == r'\varrho'
    # 确认符号'varsigma'的LaTeX表示正确
    assert latex(Symbol('varsigma')) == r'\varsigma'
    # 确认符号'vartheta'的LaTeX表示正确
    assert latex(Symbol('vartheta')) == r'\vartheta'
# 测试函数，用于验证 sympy.latex 函数的输出是否符合预期
def test_fancyset_symbols():
    # 检查 S.Rationals 的 LaTeX 表示是否为 \mathbb{Q}
    assert latex(S.Rationals) == r'\mathbb{Q}'
    # 检查 S.Naturals 的 LaTeX 表示是否为 \mathbb{N}
    assert latex(S.Naturals) == r'\mathbb{N}'
    # 检查 S.Naturals0 的 LaTeX 表示是否为 \mathbb{N}_0
    assert latex(S.Naturals0) == r'\mathbb{N}_0'
    # 检查 S.Integers 的 LaTeX 表示是否为 \mathbb{Z}
    assert latex(S.Integers) == r'\mathbb{Z}'
    # 检查 S.Reals 的 LaTeX 表示是否为 \mathbb{R}
    assert latex(S.Reals) == r'\mathbb{R}'
    # 检查 S.Complexes 的 LaTeX 表示是否为 \mathbb{C}
    assert latex(S.Complexes) == r'\mathbb{C}'


@XFAIL
# 带有 XFAIL 装饰器的测试函数，预期这个测试会失败
def test_builtin_without_args_mismatched_names():
    # 检查 CosineTransform 的 LaTeX 表示是否为 \mathcal{COS}
    assert latex(CosineTransform) == r'\mathcal{COS}'


# 测试 sympy.latex 函数对不同对象的输出
def test_builtin_no_args():
    # 检查 Chi 函数的 LaTeX 表示是否为 \operatorname{Chi}
    assert latex(Chi) == r'\operatorname{Chi}'
    # 检查 beta 函数的 LaTeX 表示是否为 \operatorname{B}
    assert latex(beta) == r'\operatorname{B}'
    # 检查 gamma 函数的 LaTeX 表示是否为 \Gamma
    assert latex(gamma) == r'\Gamma'
    # 检查 KroneckerDelta 函数的 LaTeX 表示是否为 \delta
    assert latex(KroneckerDelta) == r'\delta'
    # 检查 DiracDelta 函数的 LaTeX 表示是否为 \delta
    assert latex(DiracDelta) == r'\delta'
    # 检查 lowergamma 函数的 LaTeX 表示是否为 \gamma
    assert latex(lowergamma) == r'\gamma'


# 测试 sympy.latex 函数在处理特定函数对象时的输出
def test_issue_6853():
    # 创建一个名称为 'Pi' 的函数对象 p，并检查其 LaTeX 表示是否为 \Pi(x)
    p = Function('Pi')
    assert latex(p(x)) == r"\Pi{\left(x \right)}"


# 测试 sympy.latex 函数在处理乘法表达式时的输出
def test_Mul():
    # 创建一个乘法表达式 e，检查其 LaTeX 表示是否为 - 2 \left(x + 1\right)
    e = Mul(-2, x + 1, evaluate=False)
    assert latex(e) == r'- 2 \left(x + 1\right)'
    # 创建其他乘法表达式并检查其 LaTeX 表示
    e = Mul(2, x + 1, evaluate=False)
    assert latex(e) == r'2 \left(x + 1\right)'
    e = Mul(S.Half, x + 1, evaluate=False)
    assert latex(e) == r'\frac{x + 1}{2}'
    e = Mul(y, x + 1, evaluate=False)
    assert latex(e) == r'y \left(x + 1\right)'
    e = Mul(-y, x + 1, evaluate=False)
    assert latex(e) == r'- y \left(x + 1\right)'
    e = Mul(-2, x + 1)
    assert latex(e) == r'- 2 x - 2'
    e = Mul(2, x + 1)
    assert latex(e) == r'2 x + 2'


# 测试 sympy.latex 函数在处理幂次表达式时的输出
def test_Pow():
    # 创建一个幂次表达式 e，检查其 LaTeX 表示是否为 2^{2}
    e = Pow(2, 2, evaluate=False)
    assert latex(e) == r'2^{2}'
    # 检查带有有理数幂次的输出是否正确
    assert latex(x**(Rational(-1, 3))) == r'\frac{1}{\sqrt[3]{x}}'
    x2 = Symbol(r'x^2')
    # 检查符号表达式的幂次输出是否正确
    assert latex(x2**2) == r'\left(x^{2}\right)^{2}'
    # 检查特定的数学表达式是否能正确地输出
    assert latex(S('1.453e4500')**x) == r'{1.453 \cdot 10^{4500}}^{x}'


# 测试 sympy.latex 函数在处理逻辑表达式时的输出
def test_issue_7180():
    # 检查等价关系表达式的 LaTeX 表示是否为 x \Leftrightarrow y
    assert latex(Equivalent(x, y)) == r"x \Leftrightarrow y"
    # 检查否定等价关系表达式的 LaTeX 表示是否为 x \not\Leftrightarrow y
    assert latex(Not(Equivalent(x, y))) == r"x \not\Leftrightarrow y"


# 测试 sympy.latex 函数在处理有理数幂次时的输出
def test_issue_8409():
    # 检查分数幂次的 LaTeX 表示是否为 \left(\frac{1}{2}\right)^{n}
    assert latex(S.Half**n) == r"\left(\frac{1}{2}\right)^{n}"


# 测试 sympy.latex 函数在处理复合表达式时的输出
def test_issue_8470():
    # 导入 sympy_parser 模块中的 parse_expr 函数
    from sympy.parsing.sympy_parser import parse_expr
    # 创建一个表达式 e，并检查其 LaTeX 表示是否为 A \left(- B\right)
    e = parse_expr("-B*A", evaluate=False)
    assert latex(e) == r"A \left(- B\right)"


# 测试 sympy.latex 函数在处理矩阵符号时的输出
def test_issue_15439():
    # 创建矩阵符号 x 和 y，并检查替换后的乘法表达式的 LaTeX 表示是否正确
    x = MatrixSymbol('x', 2, 2)
    y = MatrixSymbol('y', 2, 2)
    assert latex((x * y).subs(y, -y)) == r"x \left(- y\right)"
    assert latex((x * y).subs(y, -2*y)) == r"x \left(- 2 y\right)"
    assert latex((x * y).subs(x, -x)) == r"\left(- x\right) y"


# 测试 sympy.latex 函数在处理特定符号时的输出
def test_issue_2934():
    # 检查特定符号的 LaTeX 表示是否为 \frac{a_1}{b_1}
    assert latex(Symbol(r'\frac{a_1}{b_1}')) == r'\frac{a_1}{b_1}'


# 测试 sympy.latex 函数在处理特定符号和函数组合时的输出
def test_issue_10489():
    # 创建具有大括号的 LaTeX 符号并检查其输出是否为原始的 LaTeX 符号
    latexSymbolWithBrace = r'C_{x_{0}}'
    s = Symbol(latexSymbolWithBrace)
    assert latex(s) == latexSymbolWithBrace
    # 检查符号函数的组合表达式的 LaTeX 表示是否正确
    assert latex(cos(s)) == r'\cos{\left(C_{x_{0}} \right)}'


# 测试 sympy.latex 函数在处理指数符号时的输出
def test_issue_12886():
    # 创建指数符号并检查其 LaTeX 表示是否正确
    m__1, l__1 = symbols('m__1, l__1')
    assert latex(m__1**2 + l__1**2) == \
        r'\left(l^{1}\right)^{2} + \left(m^{1}\right)^{2}'


# 测试 sympy.latex 函数在处理分数表达式时的输出
def test_issue_13559():
    # 导入 sympy_parser 模块中的 parse_expr 函数
    # 使用 assert 语句来验证表达式 latex(expr) 是否等于字符串 r"\frac{5}{1}"
    assert latex(expr) == r"\frac{5}{1}"
def test_issue_13651():
    # 构造一个表达式，c 加上 (-1) 乘以 (a + b)，并设置 evaluate=False
    expr = c + Mul(-1, a + b, evaluate=False)
    # 断言 LaTeX 表示的结果与预期相符
    assert latex(expr) == r"c - \left(a + b\right)"


def test_latex_UnevaluatedExpr():
    # 定义符号变量 x
    x = symbols("x")
    # 创建一个未求值的表达式 he，表示为 1/x
    he = UnevaluatedExpr(1/x)
    # 断言 he 的 LaTeX 表示与 1/x 相同
    assert latex(he) == latex(1/x) == r"\frac{1}{x}"
    # 断言 he 的平方的 LaTeX 表示
    assert latex(he**2) == r"\left(\frac{1}{x}\right)^{2}"
    # 断言 he 加 1 的 LaTeX 表示
    assert latex(he + 1) == r"1 + \frac{1}{x}"
    # 断言 x 乘以 he 的 LaTeX 表示
    assert latex(x*he) == r"x \frac{1}{x}"


def test_MatrixElement_printing():
    # 测试问题 #11821 的情况
    # 定义 1x3 的矩阵符号 A, B, C
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言 A[0, 0] 的 LaTeX 表示
    assert latex(A[0, 0]) == r"{A}_{0,0}"
    # 断言 3 乘以 A[0, 0] 的 LaTeX 表示
    assert latex(3 * A[0, 0]) == r"3 {A}_{0,0}"

    # 定义 F 为 C[0, 0] 替换 C 为 A - B 后的结果
    F = C[0, 0].subs(C, A - B)
    # 断言 F 的 LaTeX 表示
    assert latex(F) == r"{\left(A - B\right)}_{0,0}"

    # 定义符号变量 i, j, k
    i, j, k = symbols("i j k")
    # 定义 kxk 的矩阵符号 M, N
    M = MatrixSymbol("M", k, k)
    N = MatrixSymbol("N", k, k)
    # 断言 (M*N)[i, j] 的 LaTeX 表示
    assert latex((M*N)[i, j]) == \
        r'\sum_{i_{1}=0}^{k - 1} {M}_{i,i_{1}} {N}_{i_{1},j}'

    # 定义 3x3 的矩阵符号 X_a
    X_a = MatrixSymbol('X_a', 3, 3)
    # 断言 X_a[0, 0] 的 LaTeX 表示
    assert latex(X_a[0, 0]) == r"{X_{a}}_{0,0}"


def test_MatrixSymbol_printing():
    # 测试问题 #14237 的情况
    # 定义 3x3 的矩阵符号 A, B, C
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    # 断言 -A 的 LaTeX 表示
    assert latex(-A) == r"- A"
    # 断言 A - A*B - B 的 LaTeX 表示
    assert latex(A - A*B - B) == r"A - A B - B"
    # 断言 -A*B - A*B*C - B 的 LaTeX 表示
    assert latex(-A*B - A*B*C - B) == r"- A B - A B C - B"


def test_KroneckerProduct_printing():
    # 定义 3x3 和 2x2 的矩阵符号 A, B
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 2, 2)
    # 断言 KroneckerProduct(A, B) 的 LaTeX 表示
    assert latex(KroneckerProduct(A, B)) == r'A \otimes B'


def test_Series_printing():
    # 定义传递函数 tf1, tf2, tf3
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    # 断言 Series(tf1, tf2) 的 LaTeX 表示
    assert latex(Series(tf1, tf2)) == \
        r'\left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right) \left(\frac{x - y}{x + y}\right)'
    # 断言 Series(tf1, tf2, tf3) 的 LaTeX 表示
    assert latex(Series(tf1, tf2, tf3)) == \
        r'\left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right) \left(\frac{x - y}{x + y}\right) \left(\frac{t x^{2} - t^{w} x + w}{t - y}\right)'
    # 断言 Series(-tf2, tf1) 的 LaTeX 表示
    assert latex(Series(-tf2, tf1)) == \
        r'\left(\frac{- x + y}{x + y}\right) \left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right)'

    # 定义 2x1 的矩阵 M_1, T_1, 2x2 的矩阵 M_2, T_2
    M_1 = Matrix([[5/s], [5/(2*s)]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5, 6*s**3]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    # 断言 T_1*(T_2 + T_2) 的 LaTeX 表示
    assert latex(T_1*(T_2 + T_2)) == \
        r'\left[\begin{matrix}\frac{5}{s}\\\frac{5}{2 s}\end{matrix}\right]_\tau\cdot\left(\left[\begin{matrix}\frac{5}{1} &' \
        r' \frac{6 s^{3}}{1}\end{matrix}\right]_\tau + \left[\begin{matrix}\frac{5}{1} & \frac{6 s^{3}}{1}\end{matrix}\right]_\tau\right)' \
            == latex(MIMOSeries(MIMOParallel(T_2, T_2), T_1))
    # 定义 2x2 的矩阵 M_3, T_3
    M_3 = Matrix([[5, 6], [6, 5/s]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    # 使用断言来验证一个表达式的 LaTeX 表示是否等于预期的值
    assert latex(T_1*T_2 + T_3) == r'\left[\begin{matrix}\frac{5}{s}\\\frac{5}{2 s}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}' \
        r'\frac{5}{1} & \frac{6 s^{3}}{1}\end{matrix}\right]_\tau + \left[\begin{matrix}\frac{5}{1} & \frac{6}{1}\\\frac{6}{1} & ' \
        r'\frac{5}{s}\end{matrix}\right]_\tau'
        # 断言：验证 MIMO 并联结构 (MIMOParallel(MIMOSeries(T_2, T_1), T_3)) 的 LaTeX 表达式是否与预期的字符串相等
        # latex 函数用于将表达式转换为 LaTeX 格式的字符串表示
# 定义用于测试传递函数打印功能的函数
def test_TransferFunction_printing():
    # 创建第一个传递函数对象 tf1
    tf1 = TransferFunction(x - 1, x + 1, x)
    # 断言第一个传递函数对象 tf1 的 LaTeX 表示是否为预期值
    assert latex(tf1) == r"\frac{x - 1}{x + 1}"
    
    # 创建第二个传递函数对象 tf2
    tf2 = TransferFunction(x + 1, 2 - y, x)
    # 断言第二个传递函数对象 tf2 的 LaTeX 表示是否为预期值
    assert latex(tf2) == r"\frac{x + 1}{2 - y}"
    
    # 创建第三个传递函数对象 tf3
    tf3 = TransferFunction(y, y**2 + 2*y + 3, y)
    # 断言第三个传递函数对象 tf3 的 LaTeX 表示是否为预期值
    assert latex(tf3) == r"\frac{y}{y^{2} + 2 y + 3}"


# 定义用于测试并联传递函数打印功能的函数
def test_Parallel_printing():
    # 创建并联传递函数对象 tf1 和 tf2
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    
    # 断言并联 tf1 和 tf2 的 LaTeX 表示是否为预期值
    assert latex(Parallel(tf1, tf2)) == \
        r'\frac{x y^{2} - z}{- t^{3} + y^{3}} + \frac{x - y}{x + y}'
    
    # 断言并联 -tf2 和 tf1 的 LaTeX 表示是否为预期值
    assert latex(Parallel(-tf2, tf1)) == \
        r'\frac{- x + y}{x + y} + \frac{x y^{2} - z}{- t^{3} + y^{3}}'

    # 创建矩阵 M_1、M_2、M_3，并转换为传递函数矩阵对象 T_1、T_2、T_3
    M_1 = Matrix([[5, 6], [6, 5/s]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5/s, 6], [6, 5/(s - 1)]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    M_3 = Matrix([[6, 5/(s*(s - 1))], [5, 6]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    
    # 断言并联 T_1、T_2、T_3 的 LaTeX 表示是否为预期值
    assert latex(T_1 + T_2 + T_3) == r'\left[\begin{matrix}\frac{5}{1} & \frac{6}{1}\\\frac{6}{1} & \frac{5}{s}\end{matrix}\right]' \
        r'_\tau + \left[\begin{matrix}\frac{5}{s} & \frac{6}{1}\\\frac{6}{1} & \frac{5}{s - 1}\end{matrix}\right]_\tau + \left[\begin{matrix}' \
            r'\frac{6}{1} & \frac{5}{s \left(s - 1\right)}\\\frac{5}{1} & \frac{6}{1}\end{matrix}\right]_\tau' \
                == latex(MIMOParallel(T_1, T_2, T_3)) == latex(MIMOParallel(T_1, MIMOParallel(T_2, T_3))) == latex(MIMOParallel(MIMOParallel(T_1, T_2), T_3))


# 定义用于测试传递函数矩阵打印功能的函数
def test_TransferFunctionMatrix_printing():
    # 创建多个传递函数对象 tf1、tf2、tf3
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    tf3 = TransferFunction(p, y**2 + 2*y + 3, p)
    
    # 断言传递函数矩阵 tf1 和 tf2 的 LaTeX 表示是否为预期值
    assert latex(TransferFunctionMatrix([[tf1], [tf2]])) == \
        r'\left[\begin{matrix}\frac{p}{p + x}\\\frac{p - s}{p + s}\end{matrix}\right]_\tau'
    
    # 断言传递函数矩阵包含 tf1、tf2、tf3 的 LaTeX 表示是否为预期值
    assert latex(TransferFunctionMatrix([[tf1, tf2], [tf3, -tf1]])) == \
        r'\left[\begin{matrix}\frac{p}{p + x} & \frac{p - s}{p + s}\\\frac{p}{y^{2} + 2 y + 3} & \frac{\left(-1\right) p}{p + x}\end{matrix}\right]_\tau'


# 定义用于测试反馈系统打印功能的函数
def test_Feedback_printing():
    # 创建多个传递函数对象 tf1、tf2
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    
    # 断言负反馈系统 tf1 和 tf2 的 LaTeX 表示是否为预期值
    assert latex(Feedback(tf1, tf2)) == \
        r'\frac{\frac{p}{p + x}}{\frac{1}{1} + \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    
    # 断言负反馈系统包含 tf1*tf2 和单位传递函数的 LaTeX 表示是否为预期值
    assert latex(Feedback(tf1*tf2, TransferFunction(1, 1, p))) == \
        r'\frac{\left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}{\frac{1}{1} + \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    
    # 断言正反馈系统 tf1 和 tf2 的 LaTeX 表示是否为预期值
    assert latex(Feedback(tf1, tf2, 1)) == \
        r'\frac{\frac{p}{p + x}}{\frac{1}{1} - \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    
    # 断言正反馈系统包含 tf1*tf2 和单位传递函数的 LaTeX 表示是否为预期值
    assert latex(Feedback(tf1*tf2, sign=1)) == \
        r'\frac{\left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}{\frac{1}{1} - \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
def test_MIMOFeedback_printing():
    # 创建四个传递函数对象，分别用不同的参数初始化
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**2 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s**2, s**2 - 1, s)

    # 创建传递函数矩阵对象 tfm_1 和 tfm_2，分别使用不同的传递函数组成
    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf4, tf3], [tf2, tf1]])

    # Negative Feedback (Default)
    # 断言：验证 MIMOFeedback 函数返回的 LaTeX 表达式是否符合预期
    assert latex(MIMOFeedback(tfm_1, tfm_2)) == \
        r'\left(I_{\tau} + \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left[' \
        r'\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1} & \frac{1}{s}\end{matrix}\right]_\tau\right)^{-1} \cdot \left[\begin{matrix}' \
        r'\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau'

    # Positive Feedback
    # 断言：验证 MIMOFeedback 函数返回的 LaTeX 表达式是否符合预期（使用不同的参数）
    assert latex(MIMOFeedback(tfm_1*tfm_2, tfm_1, 1)) == \
        r'\left(I_{\tau} - \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left' \
        r'[\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1} & \frac{1}{s}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}' \
        r'\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\right)^{-1} \cdot \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}' \
        r'\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1}' \
        r' & \frac{1}{s}\end{matrix}\right]_\tau'


def test_Quaternion_latex_printing():
    # 创建四元数对象，并验证其 LaTeX 表达式是否符合预期
    q = Quaternion(x, y, z, t)
    assert latex(q) == r"x + y i + z j + t k"
    q = Quaternion(x, y, z, x*t)
    assert latex(q) == r"x + y i + z j + t x k"
    q = Quaternion(x, y, z, x + t)
    assert latex(q) == r"x + y i + z j + \left(t + x\right) k"


def test_TensorProduct_printing():
    # 导入 TensorProduct 类并创建两个矩阵符号对象，验证其 LaTeX 表达式是否符合预期
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    assert latex(TensorProduct(A, B)) == r"A \otimes B"


def test_WedgeProduct_printing():
    # 导入 WedgeProduct 类并创建对象 wp，验证其 LaTeX 表达式是否符合预期
    from sympy.diffgeom.rn import R2
    from sympy.diffgeom import WedgeProduct
    wp = WedgeProduct(R2.dx, R2.dy)
    assert latex(wp) == r"\operatorname{d}x \wedge \operatorname{d}y"


def test_issue_9216():
    # 创建幂运算对象并验证其 LaTeX 表达式是否符合预期
    expr_1 = Pow(1, -1, evaluate=False)
    assert latex(expr_1) == r"1^{-1}"

    expr_2 = Pow(1, Pow(1, -1, evaluate=False), evaluate=False)
    assert latex(expr_2) == r"1^{1^{-1}}"

    expr_3 = Pow(3, -2, evaluate=False)
    assert latex(expr_3) == r"\frac{1}{9}"

    expr_4 = Pow(1, -2, evaluate=False)
    assert latex(expr_4) == r"1^{-2}"


def test_latex_printer_tensor():
    # 导入相关的 tensor 模块并创建 tensor 对象，此处未输出断言，仅验证 LaTeX 表达式是否符合预期
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensor_heads
    L = TensorIndexType("L")
    i, j, k, l = tensor_indices("i j k l", L)
    i0 = tensor_indices("i_0", L)
    # 使用 `tensor_heads` 函数从字符串和尺寸列表[L]创建张量头A, B, C, D
    A, B, C, D = tensor_heads("A B C D", [L])
    
    # 创建具有形状[L, L]的张量头H
    H = TensorHead("H", [L, L])
    
    # 创建具有形状[L, L, L, L]的张量头K
    K = TensorHead("K", [L, L, L, L])
    
    # 确认张量 `i` 的latex表达式为 `{}^{i}`
    assert latex(i) == r"{}^{i}"
    
    # 确认张量 `-i` 的latex表达式为 `{}_{i}`
    assert latex(-i) == r"{}_{i}"
    
    # 将张量A在索引i处的latex表达式赋给expr
    expr = A(i)
    assert latex(expr) == r"A{}^{i}"
    
    # 将张量A在索引i0处的latex表达式赋给expr
    expr = A(i0)
    assert latex(expr) == r"A{}^{i_{0}}"
    
    # 将张量A在负索引i处的latex表达式赋给expr
    expr = A(-i)
    assert latex(expr) == r"A{}_{i}"
    
    # 将表达式 `-3*A(i)` 的latex表达式赋给expr
    expr = -3*A(i)
    assert latex(expr) == r"-3A{}^{i}"
    
    # 将张量K在索引i, j, -k, -i0处的latex表达式赋给expr
    expr = K(i, j, -k, -i0)
    assert latex(expr) == r"K{}^{ij}{}_{ki_{0}}"
    
    # 将张量K在索引i, -j, -k, i0处的latex表达式赋给expr
    expr = K(i, -j, -k, i0)
    assert latex(expr) == r"K{}^{i}{}_{jk}{}^{i_{0}}"
    
    # 将张量K在索引i, -j, k, -i0处的latex表达式赋给expr
    expr = K(i, -j, k, -i0)
    assert latex(expr) == r"K{}^{i}{}_{j}{}^{k}{}_{i_{0}}"
    
    # 将张量H在索引i, -j处的latex表达式赋给expr
    expr = H(i, -j)
    assert latex(expr) == r"H{}^{i}{}_{j}"
    
    # 将张量H在索引i, j处的latex表达式赋给expr
    expr = H(i, j)
    assert latex(expr) == r"H{}^{ij}"
    
    # 将张量H在负索引i, j处的latex表达式赋给expr
    expr = H(-i, -j)
    assert latex(expr) == r"H{}_{ij}"
    
    # 将表达式 `(1+x)*A(i)` 的latex表达式赋给expr
    expr = (1+x)*A(i)
    assert latex(expr) == r"\left(x + 1\right)A{}^{i}"
    
    # 将张量H在索引i, -i处的latex表达式赋给expr
    expr = H(i, -i)
    assert latex(expr) == r"H{}^{L_{0}}{}_{L_{0}}"
    
    # 将张量H在索引i, -j处乘以张量A在索引j处乘以张量B在索引k处的latex表达式赋给expr
    expr = H(i, -j)*A(j)*B(k)
    assert latex(expr) == r"H{}^{i}{}_{L_{0}}A{}^{L_{0}}B{}^{k}"
    
    # 测试 `TensorElement`:
    from sympy.tensor.tensor import TensorElement
    
    # 将TensorElement(K(i, j, k, l), {i: 3, k: 2})的latex表达式赋给expr
    expr = TensorElement(K(i, j, k, l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3,j,k=2,l}'
    
    # 将TensorElement(K(i, j, k, l), {i: 3})的latex表达式赋给expr
    expr = TensorElement(K(i, j, k, l), {i: 3})
    assert latex(expr) == r'K{}^{i=3,jkl}'
    
    # 将TensorElement(K(i, -j, k, l), {i: 3, k: 2})的latex表达式赋给expr
    expr = TensorElement(K(i, -j, k, l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3}{}_{j}{}^{k=2,l}'
    
    # 将TensorElement(K(i, -j, k, -l), {i: 3, k: 2})的latex表达式赋给expr
    expr = TensorElement(K(i, -j, k, -l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3}{}_{j}{}^{k=2}{}_{l}'
    
    # 将TensorElement(K(i, j, -k, -l), {i: 3, -k: 2})的latex表达式赋给expr
    expr = TensorElement(K(i, j, -k, -l), {i: 3, -k: 2})
    assert latex(expr) == r'K{}^{i=3,j}{}_{k=2,l}'
    
    # 将TensorElement(K(i, j, -k, -l), {i: 3})的latex表达式赋给expr
    expr = TensorElement(K(i, j, -k, -l), {i: 3})
    assert latex(expr) == r'K{}^{i=3,j}{}_{kl}'
    
    # 将PartialDerivative(A(i), A(i))的latex表达式赋给expr
    expr = PartialDerivative(A(i), A(i))
    assert latex(expr) == r"\frac{\partial}{\partial {A{}^{L_{0}}}}{A{}^{L_{0}}}"
    
    # 将PartialDerivative(A(-i), A(-j))的latex表达式赋给expr
    expr = PartialDerivative(A(-i), A(-j))
    assert latex(expr) == r"\frac{\partial}{\partial {A{}_{j}}}{A{}_{i}}"
    
    # 将PartialDerivative(K(i, j, -k, -l), A(m), A(-n))的latex表达式赋给expr
    expr = PartialDerivative(K(i, j, -k, -l), A(m), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}^{m}} \partial {A{}_{n}}}{K{}^{ij}{}_{kl}}"
    
    # 将PartialDerivative(B(-i) + A(-i), A(-j), A(-n))的latex表达式赋给expr
    expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}_{j}} \partial {A{}_{n}}}{\left(A{}_{i} + B{}_{i}\right)}"
    
    # 将PartialDerivative(3*A(-i), A(-j), A(-n))的latex表达式赋给expr
    expr = PartialDerivative(3*A(-i), A(-j), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}_{j}} \partial {A{}_{n}}}{\left(3A{}_{i}\right)}"
def test_multiline_latex():
    # 定义符号变量 a, b, c, d, e, f
    a, b, c, d, e, f = symbols('a b c d e f')
    # 构建表达式 expr
    expr = -a + 2*b -3*c +4*d -5*e
    # 构建预期的 LaTeX 字符串 expected，表示为多行 LaTeX 环境 eqnarray
    expected = r"\begin{eqnarray}" + "\n"\
        r"f & = &- a \nonumber\\" + "\n"\
        r"& & + 2 b \nonumber\\" + "\n"\
        r"& & - 3 c \nonumber\\" + "\n"\
        r"& & + 4 d \nonumber\\" + "\n"\
        r"& & - 5 e " + "\n"\
        r"\end{eqnarray}"
    # 断言调用 multiline_latex 函数返回的结果与 expected 相等
    assert multiline_latex(f, expr, environment="eqnarray") == expected

    # 构建预期的 LaTeX 字符串 expected2，另一种表达形式
    expected2 = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b \nonumber\\' + '\n'\
        r'& & - 3 c + 4 d \nonumber\\' + '\n'\
        r'& & - 5 e ' + '\n'\
        r'\end{eqnarray}'
    # 断言调用 multiline_latex 函数返回的结果与 expected2 相等
    assert multiline_latex(f, expr, 2, environment="eqnarray") == expected2

    # 构建预期的 LaTeX 字符串 expected3，另一种表达形式
    expected3 = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b - 3 c \nonumber\\'+ '\n'\
        r'& & + 4 d - 5 e ' + '\n'\
        r'\end{eqnarray}'
    # 断言调用 multiline_latex 函数返回的结果与 expected3 相等
    assert multiline_latex(f, expr, 3, environment="eqnarray") == expected3

    # 构建预期的 LaTeX 字符串 expected3dots，使用省略号的表达形式
    expected3dots = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b - 3 c \dots\nonumber\\'+ '\n'\
        r'& & + 4 d - 5 e ' + '\n'\
        r'\end{eqnarray}'
    # 断言调用 multiline_latex 函数返回的结果与 expected3dots 相等
    assert multiline_latex(f, expr, 3, environment="eqnarray", use_dots=True) == expected3dots

    # 构建预期的 LaTeX 字符串 expected3align，使用 align* 环境的表达形式
    expected3align = r'\begin{align*}' + '\n'\
        r'f = &- a + 2 b - 3 c \\' + '\n'\
        r'& + 4 d - 5 e ' + '\n'\
        r'\end{align*}'
    # 断言调用 multiline_latex 函数返回的结果与 expected3align 相等
    assert multiline_latex(f, expr, 3) == expected3align
    assert multiline_latex(f, expr, 3, environment='align*') == expected3align

    # 构建预期的 LaTeX 字符串 expected2ieee，使用 IEEEeqnarray 环境的表达形式
    expected2ieee = r'\begin{IEEEeqnarray}{rCl}' + '\n'\
        r'f & = &- a + 2 b \nonumber\\' + '\n'\
        r'& & - 3 c + 4 d \nonumber\\' + '\n'\
        r'& & - 5 e ' + '\n'\
        r'\end{IEEEeqnarray}'
    # 断言调用 multiline_latex 函数返回的结果与 expected2ieee 相等
    assert multiline_latex(f, expr, 2, environment="IEEEeqnarray") == expected2ieee

    # 断言调用 multiline_latex 函数引发 ValueError 异常，用于环境参数为 "foo" 的情况
    raises(ValueError, lambda: multiline_latex(f, expr, environment="foo"))
    # 断言：验证 trace(A**2) 的 LaTeX 表示是否等于 "\operatorname{tr}\left(A^{2} \right)"
    assert latex(trace(A**2)) == r"\operatorname{tr}\left(A^{2} \right)"
def test_print_basic():
    # 导入需要的类和函数
    from sympy.core.basic import Basic
    from sympy.core.expr import Expr

    # 用于测试打印的虚拟类，其在latex.py中未实现
    class UnimplementedExpr(Expr):
        def __new__(cls, e):
            return Basic.__new__(cls, e)

    # 用于测试的虚拟函数
    def unimplemented_expr(expr):
        return UnimplementedExpr(expr).doit()

    # 修改类名以使用上标/下标
    def unimplemented_expr_sup_sub(expr):
        result = UnimplementedExpr(expr)
        result.__class__.__name__ = 'UnimplementedExpr_x^1'
        return result

    # 断言语句，验证输出的latex格式是否符合预期
    assert latex(unimplemented_expr(x)) == r'\operatorname{UnimplementedExpr}\left(x\right)'
    assert latex(unimplemented_expr(x**2)) == \
        r'\operatorname{UnimplementedExpr}\left(x^{2}\right)'
    assert latex(unimplemented_expr_sup_sub(x)) == \
        r'\operatorname{UnimplementedExpr^{1}_{x}}\left(x\right)'


def test_MatrixSymbol_bold():
    # 导入需要的类和函数
    from sympy.matrices.expressions.trace import trace
    from sympy.matrices.expressions import MatrixSymbol

    # 创建MatrixSymbol对象A
    A = MatrixSymbol("A", 2, 2)

    # 断言语句，验证输出的latex格式是否符合预期
    assert latex(trace(A), mat_symbol_style='bold') == \
        r"\operatorname{tr}\left(\mathbf{A} \right)"
    assert latex(trace(A), mat_symbol_style='plain') == \
        r"\operatorname{tr}\left(A \right)"

    # 创建多个MatrixSymbol对象并验证输出的latex格式是否符合预期
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    assert latex(-A, mat_symbol_style='bold') == r"- \mathbf{A}"
    assert latex(A - A*B - B, mat_symbol_style='bold') == \
        r"\mathbf{A} - \mathbf{A} \mathbf{B} - \mathbf{B}"
    assert latex(-A*B - A*B*C - B, mat_symbol_style='bold') == \
        r"- \mathbf{A} \mathbf{B} - \mathbf{A} \mathbf{B} \mathbf{C} - \mathbf{B}"

    # 创建具有下标的MatrixSymbol对象并验证输出的latex格式是否符合预期
    A_k = MatrixSymbol("A_k", 3, 3)
    assert latex(A_k, mat_symbol_style='bold') == r"\mathbf{A}_{k}"

    # 创建包含特殊符号的MatrixSymbol对象并验证输出的latex格式是否符合预期
    A = MatrixSymbol(r"\nabla_k", 3, 3)
    assert latex(A, mat_symbol_style='bold') == r"\mathbf{\nabla}_{k}"


def test_AppliedPermutation():
    # 导入需要的类和函数
    from sympy.combinatorics.permutations import Permutation
    from sympy.core.symbol import Symbol
    from sympy.core.function import AppliedPermutation

    # 创建Permutation对象p和Symbol对象x
    p = Permutation(0, 1, 2)
    x = Symbol('x')

    # 断言语句，验证输出的latex格式是否符合预期
    assert latex(AppliedPermutation(p, x)) == \
        r'\sigma_{\left( 0\; 1\; 2\right)}(x)'


def test_PermutationMatrix():
    # 导入需要的类和函数
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.permutation import PermutationMatrix

    # 创建Permutation对象p
    p = Permutation(0, 1, 2)

    # 断言语句，验证输出的latex格式是否符合预期
    assert latex(PermutationMatrix(p)) == r'P_{\left( 0\; 1\; 2\right)}'
    p = Permutation(0, 3)(1, 2)
    assert latex(PermutationMatrix(p)) == \
        r'P_{\left( 0\; 3\right)\left( 1\; 2\right)}'


def test_issue_21758():
    # 导入需要的类和函数
    from sympy.core.symbol import Symbol
    from sympy.sets.fancysets import SeqFormula
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.series.fourier import FourierSeries

    # 创建Symbol对象x, k, n和四阶傅立叶级数对象fo，并验证内容是否符合预期
    x = Symbol('x')
    k, n = symbols('k n')
    fo = FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(
        Piecewise((-2*pi*cos(n*pi)/n + 2*sin(n*pi)/n**2, (n > -oo) & (n < oo) & Ne(n, 0)),
                  (0, True))*sin(n*x)/pi, (n, 1, oo))))
    # 断言：验证 piecewise_fold(fo) 的 LaTeX 表示是否等于给定的数学表达式
    assert latex(piecewise_fold(fo)) == '\\begin{cases} 2 \\sin{\\left(x \\right)}' \
            ' - \\sin{\\left(2 x \\right)} + \\frac{2 \\sin{\\left(3 x \\right)}}{3} +' \
            ' \\ldots & \\text{for}\\: n > -\\infty \\wedge n < \\infty \\wedge ' \
                'n \\neq 0 \\\\0 & \\text{otherwise} \\end{cases}'
    
    # 断言：验证 FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)),
    # SeqFormula(0, (n, 1, oo)))) 的 LaTeX 表示是否等于给定的数学表达式 '0'
    assert latex(FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)),
                                                 SeqFormula(0, (n, 1, oo))))) == '0'
def test_imaginary_unit():
    # 检查默认虚数单位为 'i' 时的 LaTeX 表示是否正确
    assert latex(1 + I) == r'1 + i'
    # 检查指定虚数单位为 'i' 时的 LaTeX 表示是否正确（应该等同于默认情况）
    assert latex(1 + I, imaginary_unit='i') == r'1 + i'
    # 检查指定虚数单位为 'j' 时的 LaTeX 表示是否正确
    assert latex(1 + I, imaginary_unit='j') == r'1 + j'
    # 检查指定虚数单位为 'foo' 时的 LaTeX 表示是否正确
    assert latex(1 + I, imaginary_unit='foo') == r'1 + foo'
    # 检查单独虚数单位为 "ti" 时的 LaTeX 表示是否正确
    assert latex(I, imaginary_unit="ti") == r'\text{i}'
    # 检查单独虚数单位为 "tj" 时的 LaTeX 表示是否正确
    assert latex(I, imaginary_unit="tj") == r'\text{j}'


def test_text_re_im():
    # 检查使用哥特体符号表示虚部时的 LaTeX 表示是否正确
    assert latex(im(x), gothic_re_im=True) == r'\Im{\left(x\right)}'
    # 检查使用操作符名称表示虚部时的 LaTeX 表示是否正确
    assert latex(im(x), gothic_re_im=False) == r'\operatorname{im}{\left(x\right)}'
    # 检查使用哥特体符号表示实部时的 LaTeX 表示是否正确
    assert latex(re(x), gothic_re_im=True) == r'\Re{\left(x\right)}'
    # 检查使用操作符名称表示实部时的 LaTeX 表示是否正确
    assert latex(re(x), gothic_re_im=False) == r'\operatorname{re}{\left(x\right)}'


def test_latex_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential
    from sympy.diffgeom.rn import R2
    x,y = symbols('x y', real=True)
    # 创建一个二维流形对象
    m = Manifold('M', 2)
    assert latex(m) == r'\text{M}'
    # 在流形对象上创建一个 patch
    p = Patch('P', m)
    assert latex(p) == r'\text{P}_{\text{M}}'
    # 在 patch 上创建一个直角坐标系
    rect = CoordSystem('rect', p, [x, y])
    assert latex(rect) == r'\text{rect}^{\text{P}}_{\text{M}}'
    # 创建一个基础标量场对象
    b = BaseScalarField(rect, 0)
    assert latex(b) == r'\mathbf{x}'

    # 创建一个函数 g
    g = Function('g')
    # 定义一个标量场
    s_field = g(R2.x, R2.y)
    # 检查标量场的微分形式的 LaTeX 表示是否正确
    assert latex(Differential(s_field)) == \
        r'\operatorname{d}\left(g{\left(\mathbf{x},\mathbf{y} \right)}\right)'


def test_unit_printing():
    # 检查米单位的 LaTeX 表示是否正确
    assert latex(5*meter) == r'5 \text{m}'
    # 检查 gibibyte 单位的 LaTeX 表示是否正确
    assert latex(3*gibibyte) == r'3 \text{gibibyte}'
    # 检查每秒微克单位的 LaTeX 表示是否正确
    assert latex(4*microgram/second) == r'\frac{4 \mu\text{g}}{\text{s}}'
    # 检查每秒微克单位（带空格）的 LaTeX 表示是否正确
    assert latex(4*micro*gram/second) == r'\frac{4 \mu \text{g}}{\text{s}}'
    # 检查毫米单位的 LaTeX 表示是否正确（带多余空格）
    assert latex(5*milli*meter) == r'5 \text{m} \text{m}'
    # 检查 milli 单位的 LaTeX 表示是否正确
    assert latex(milli) == r'\text{m}'


def test_issue_17092():
    x_star = Symbol('x^*')
    # 检查二阶导数的 LaTeX 表示是否正确
    assert latex(Derivative(x_star, x_star,2)) == r'\frac{d^{2}}{d \left(x^{*}\right)^{2}} x^{*}'


def test_latex_decimal_separator():

    x, y, z, t = symbols('x y z t')
    k, m, n = symbols('k m n', integer=True)
    f, g, h = symbols('f g h', cls=Function)

    # 逗号作为小数分隔符时的 LaTeX 表示
    assert(latex([1, 2.3, 4.5], decimal_separator='comma') == r'\left[ 1; \  2{,}3; \  4{,}5\right]')
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='comma') == r'\left\{1; 2{,}3; 4{,}5\right\}')
    assert(latex((1, 2.3, 4.6), decimal_separator = 'comma') == r'\left( 1; \  2{,}3; \  4{,}6\right)')
    assert(latex((1,), decimal_separator='comma') == r'\left( 1;\right)')

    # 句点作为小数分隔符时的 LaTeX 表示
    assert(latex([1, 2.3, 4.5], decimal_separator='period') == r'\left[ 1, \  2.3, \  4.5\right]' )
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == r'\left\{1, 2.3, 4.5\right\}')
    assert(latex((1, 2.3, 4.6), decimal_separator = 'period') == r'\left( 1, \  2.3, \  4.6\right)')
    assert(latex((1,), decimal_separator='period') == r'\left( 1,\right)')

    # 默认小数分隔符时的 LaTeX 表示
    assert(latex([1, 2.3, 4.5]) == r'\left[ 1, \  2.3, \  4.5\right]')
    # 使用 SymPy 的 latex 函数将 FiniteSet(1, 2.3, 4.5) 转换为 LaTeX 表示形式
    assert(latex(FiniteSet(1, 2.3, 4.5)) == r'\left\{1, 2.3, 4.5\right\}')
    
    # 使用 SymPy 的 latex 函数将元组 (1, 2.3, 4.6) 转换为 LaTeX 表示形式
    assert(latex((1, 2.3, 4.6)) == r'\left( 1, \  2.3, \  4.6\right)')
    
    # 使用 SymPy 的 latex 函数将元组 (1,) 转换为 LaTeX 表示形式
    assert(latex((1,)) == r'\left( 1,\right)')
    
    # 使用 SymPy 的 latex 函数将 Mul(3.4,5.3) 的乘法运算转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(Mul(3.4,5.3), decimal_separator = 'comma') == r'18{,}02')
    
    # 使用 SymPy 的 latex 函数将 3.4*5.3 的乘法运算转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(3.4*5.3, decimal_separator = 'comma') == r'18{,}02')
    
    # 创建 SymPy 符号 x, y, z
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    
    # 使用 SymPy 的 latex 函数将数学表达式转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(x*5.3 + 2**y**3.4 + 4.5 + z, decimal_separator = 'comma') == r'2^{y^{3{,}4}} + 5{,}3 x + z + 4{,}5')
    
    # 使用 SymPy 的 latex 函数将小数 0.987 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(0.987, decimal_separator='comma') == r'0{,}987')
    
    # 使用 SymPy 的 latex 函数将小数 S(0.987) 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(S(0.987), decimal_separator='comma') == r'0{,}987')
    
    # 使用 SymPy 的 latex 函数将小数 .3 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(.3, decimal_separator='comma') == r'0{,}3')
    
    # 使用 SymPy 的 latex 函数将小数 S(.3) 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(S(.3), decimal_separator='comma') == r'0{,}3')
    
    # 使用 SymPy 的 latex 函数将科学计数法表示的数值转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(5.8*10**(-7), decimal_separator='comma') == r'5{,}8 \cdot 10^{-7}')
    
    # 使用 SymPy 的 latex 函数将科学计数法表示的数值 S(5.7)*10**(-7) 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(S(5.7)*10**(-7), decimal_separator='comma') == r'5{,}7 \cdot 10^{-7}')
    
    # 使用 SymPy 的 latex 函数将科学计数法表示的数值 S(5.7*10**(-7)) 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(S(5.7*10**(-7)), decimal_separator='comma') == r'5{,}7 \cdot 10^{-7}')
    
    # 使用 SymPy 的 latex 函数将表达式 1.2*x+3.4 转换为 LaTeX 表示形式，指定小数分隔符为逗号
    assert(latex(1.2*x+3.4, decimal_separator='comma') == r'1{,}2 x + 3{,}4')
    
    # 使用 SymPy 的 latex 函数将 FiniteSet(1, 2.3, 4.5) 转换为 LaTeX 表示形式，指定小数分隔符为句点
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == r'\left\{1, 2.3, 4.5\right\}')
    
    # 测试错误处理，确认 latex 函数能够捕获并抛出值错误，当使用不支持的小数分隔符时
    raises(ValueError, lambda: latex([1,2.3,4.5], decimal_separator='non_existing_decimal_separator_in_list'))
    raises(ValueError, lambda: latex(FiniteSet(1,2.3,4.5), decimal_separator='non_existing_decimal_separator_in_set'))
    raises(ValueError, lambda: latex((1,2.3,4.5), decimal_separator='non_existing_decimal_separator_in_tuple'))
# 定义一个测试函数 test_Str，用于测试 sympy.core.symbol 模块中的 Str 类
def test_Str():
    # 导入 Str 类
    from sympy.core.symbol import Str
    # 断言 Str('x') 转换为字符串后等于 'x'
    assert str(Str('x')) == r'x'

# 定义一个测试函数 test_latex_escape，测试 latex_escape 函数的转义功能
def test_latex_escape():
    # 断言对特殊字符 "~^&%$#_{}" 的 latex 转义结果与预期结果相同
    assert latex_escape(r"~^\&%$#_{}") == "".join([
        r'\textasciitilde',
        r'\textasciicircum',
        r'\textbackslash',
        r'\&',
        r'\%',
        r'\$',
        r'\#',
        r'\_',
        r'\{',
        r'\}',
    ])

# 定义一个测试函数 test_emptyPrinter，测试 latex 函数对未知对象的处理
def test_emptyPrinter():
    # 定义一个带有 __repr__ 方法的类 MyObject
    class MyObject:
        def __repr__(self):
            return "<MyObject with {...}>"

    # 断言 latex(MyObject()) 的输出为带有 monospaced 格式的字符串
    assert latex(MyObject()) == r"\mathtt{\text{<MyObject with \{...\}>}}"

    # 断言即使嵌套在其他对象中，latex 也能正确处理对象 MyObject
    assert latex((MyObject(),)) == r"\left( \mathtt{\text{<MyObject with \{...\}>}},\right)"

# 定义一个测试函数 test_global_settings，测试 latex 函数的全局设置功能
def test_global_settings():
    # 导入 inspect 模块
    import inspect

    # 断言 imaginary_unit 参数在 latex 函数签名中的默认值为 'i'
    assert inspect.signature(latex).parameters['imaginary_unit'].default == r'i'
    # 断言 latex(I) 输出为 'i'
    assert latex(I) == r'i'
    
    try:
        # 修改全局设置，将 imaginary_unit 设置为 'j'
        LatexPrinter.set_global_settings(imaginary_unit='j')
        # 断言修改后 imaginary_unit 参数在 latex 函数签名中的默认值为 'j'
        assert inspect.signature(latex).parameters['imaginary_unit'].default == r'j'
        # 断言 latex(I) 输出为 'j'
        assert latex(I) == r'j'
    finally:
        # 没有公共 API 可以撤销设置修改，但需要确保撤销以免影响其他测试
        del LatexPrinter._global_settings['imaginary_unit']

    # 再次断言 imaginary_unit 参数在 latex 函数签名中的默认值恢复为 'i'
    assert inspect.signature(latex).parameters['imaginary_unit'].default == r'i'
    # 断言 latex(I) 输出为 'i'
    assert latex(I) == r'i'

# 定义一个测试函数 test_pickleable，测试 _PrintFunction 实例的 pickle 可序列化性
def test_pickleable():
    # 导入 pickle 模块
    import pickle
    # 断言 pickle 序列化和反序列化后，latex 函数仍保持不变
    assert pickle.loads(pickle.dumps(latex)) is latex

# 定义一个测试函数 test_printing_latex_array_expressions，测试 latex 函数对数组表达式的处理
def test_printing_latex_array_expressions():
    # 断言对 ArraySymbol("A", (2, 3, 4)) 的 latex 输出为 "A"
    assert latex(ArraySymbol("A", (2, 3, 4))) == "A"
    # 断言对 ArrayElement("A", (2, 1/(1-x), 0)) 的 latex 输出为 "{{A}_{2, \\frac{1}{1 - x}, 0}}"
    assert latex(ArrayElement("A", (2, 1/(1-x), 0))) == "{{A}_{2, \\frac{1}{1 - x}, 0}}"
    # 定义矩阵符号 M 和 N
    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    # 断言对 ArrayElement(M*N, [x, 0]) 的 latex 输出为 "{{\\left(M N\\right)}_{x, 0}}"
    assert latex(ArrayElement(M*N, [x, 0])) == "{{\\left(M N\\right)}_{x, 0}}"

# 定义一个测试函数 test_Array，测试 latex 函数对数组的处理
def test_Array():
    # 创建一个包含 0 到 9 的数组 arr
    arr = Array(range(10))
    # 断言对数组 arr 的 latex 输出为 r'\left[\begin{matrix}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\end{matrix}\right]'
    assert latex(arr) == r'\left[\begin{matrix}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\end{matrix}\right]'

    # 创建一个包含 0 到 10 的数组 arr
    arr = Array(range(11))
    # 断言对数组 arr 的 latex 输出为 r'\left[\begin{array}{ccccccccccc}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'
    # 为了避免 latex 错误，填充空参数为一堆 'c'
    assert latex(arr) == r'\left[\begin{array}{ccccccccccc}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'

# 定义一个测试函数 test_latex_with_unevaluated，测试在 evaluate(False) 下 latex 函数的输出
def test_latex_with_unevaluated():
    # 使用 evaluate(False) 上下文
    with evaluate(False):
        # 断言在不求值的情况下，latex(a * a) 的输出为 r"a a"
        assert latex(a * a) == r"a a"
```