# `D:\src\scipysrc\sympy\sympy\integrals\laplace.py`

```
# 导入系统模块和Sympy库
import sys
import sympy
# 从Sympy核心模块导入特定类和函数
from sympy.core import S, pi, I
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (
    AppliedUndef, Derivative, expand, expand_complex, expand_mul, expand_trig,
    Lambda, WildFunction, diff, Subs)
from sympy.core.mul import Mul, prod
from sympy.core.relational import (
    _canonical, Ge, Gt, Lt, Unequality, Eq, Ne, Relational)
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.complexes import (
    re, im, arg, Abs, polar_lift, periodic_argument)
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, asinh
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import (
    Piecewise, piecewise_exclusive)
from sympy.functions.elementary.trigonometric import cos, sin, atan, sinc
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.special.gamma_functions import (
    digamma, gamma, lowergamma, uppergamma)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.integrals import integrate, Integral
from sympy.integrals.transforms import (
    _simplify, IntegralTransform, IntegralTransformError)
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.matrices.matrixbase import MatrixBase
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.utilities.exceptions import (
    sympy_deprecation_warning, SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.misc import debugf

# 设定调试级别初始值
_LT_level = 0


# 定义装饰器函数DEBUG_WRAP，用于包装其他函数以便调试
def DEBUG_WRAP(func):
    # 定义一个装饰器函数 `wrap`，用于包装一个函数，添加调试和追踪功能
    def wrap(*args, **kwargs):
        # 导入 `sympy` 库的 `SYMPY_DEBUG` 变量
        from sympy import SYMPY_DEBUG
        # 声明全局变量 `_LT_level`
        global _LT_level

        # 如果 `SYMPY_DEBUG` 未启用，则直接调用原始函数并返回结果
        if not SYMPY_DEBUG:
            return func(*args, **kwargs)

        # 如果 `_LT_level` 等于 0，打印分隔线
        if _LT_level == 0:
            print('\n' + '-'*78, file=sys.stderr)
        
        # 打印当前函数调用的信息，包括函数名和参数
        print('-LT- %s%s%s' % ('  '*_LT_level, func.__name__, args),
              file=sys.stderr)
        
        # 增加 `_LT_level` 的计数
        _LT_level += 1
        
        # 如果函数名是 '_laplace_transform_integration' 或 '_inverse_laplace_transform_integration'
        if (
                func.__name__ == '_laplace_transform_integration' or
                func.__name__ == '_inverse_laplace_transform_integration'):
            # 设置 `sympy` 库的 `SYMPY_DEBUG` 为 False，以禁用调试模式
            sympy.SYMPY_DEBUG = False
            # 打印集成操作的信息
            print('**** %sIntegrating ...' % ('  '*_LT_level), file=sys.stderr)
            # 调用原始函数并获取结果
            result = func(*args, **kwargs)
            # 恢复 `sympy` 库的 `SYMPY_DEBUG` 设置为 True，重新启用调试模式
            sympy.SYMPY_DEBUG = True
        else:
            # 对于其他函数名，直接调用原始函数并获取结果
            result = func(*args, **kwargs)
        
        # 减少 `_LT_level` 的计数
        _LT_level -= 1
        
        # 打印函数调用结束的信息，包括返回的结果
        print('-LT- %s---> %s' % ('  '*_LT_level, result), file=sys.stderr)
        
        # 如果 `_LT_level` 等于 0，打印结束分隔线
        if _LT_level == 0:
            print('-'*78 + '\n', file=sys.stderr)
        
        # 返回最终的函数调用结果
        return result
    # 返回 `wrap` 函数作为装饰器
    return wrap
# 定义一个用于调试输出的函数，打印文本到标准错误输出流
def _debug(text):
    # 从 sympy 库导入 SYMPY_DEBUG 变量
    from sympy import SYMPY_DEBUG
    # 导入全局变量 _LT_level
    global _LT_level

    # 如果 SYMPY_DEBUG 为真，则打印调试信息到标准错误输出流
    if SYMPY_DEBUG:
        print('-LT- %s%s' % ('  '*_LT_level, text), file=sys.stderr)


# 对表达式中的一些条件进行简化，条件为 `\operatorname{Re}(s) > a`
def _simplifyconds(expr, s, a):
    # 定义一个内部函数 power，用于提取指数
    def power(ex):
        if ex == s:
            return 1
        if ex.is_Pow and ex.base == s:
            return ex.exp
        return None

    # 定义一个内部函数 bigger，用于比较表达式大小关系
    def bigger(ex1, ex2):
        """ Return True only if |ex1| > |ex2|, False only if |ex1| < |ex2|.
            Else return None. """
        # 如果 ex1 和 ex2 都包含 s，则返回 None
        if ex1.has(s) and ex2.has(s):
            return None
        # 如果 ex1 是绝对值函数 Abs，则取其参数
        if isinstance(ex1, Abs):
            ex1 = ex1.args[0]
        # 如果 ex2 是绝对值函数 Abs，则取其参数
        if isinstance(ex2, Abs):
            ex2 = ex2.args[0]
        # 如果 ex1 包含 s，则调用 bigger(1/ex2, 1/ex1)
        if ex1.has(s):
            return bigger(1/ex2, 1/ex1)
        # 否则计算 ex2 的幂次数
        n = power(ex2)
        if n is None:
            return None
        try:
            # 根据幂次数比较 ex1 和 ex2 的大小关系
            if n > 0 and (Abs(ex1) <= Abs(a)**n) == S.true:
                return False
            if n < 0 and (Abs(ex1) >= Abs(a)**n) == S.true:
                return True
        except TypeError:
            return None

    # 定义一个内部函数 replie，用于简化 x < y 的表达式
    def replie(x, y):
        """ simplify x < y """
        # 如果 x 不是正数或不是绝对值，或者 y 不是正数或不是绝对值，则返回 x < y
        if (not (x.is_positive or isinstance(x, Abs))
                or not (y.is_positive or isinstance(y, Abs))):
            return (x < y)
        # 否则调用 bigger 函数比较 x 和 y 的大小关系
        r = bigger(x, y)
        if r is not None:
            return not r
        return (x < y)

    # 定义一个内部函数 replue，用于处理不等式表达式
    def replue(x, y):
        b = bigger(x, y)
        if b in (True, False):
            return True
        return Unequality(x, y)

    # 定义一个内部函数 repl，用于替换表达式中的特定符号
    def repl(ex, *args):
        if ex in (True, False):
            return bool(ex)
        return ex.replace(*args)

    # 从 sympy 库中导入函数 collect_abs，用于收集表达式中的绝对值
    from sympy.simplify.radsimp import collect_abs
    # 收集表达式中的绝对值
    expr = collect_abs(expr)
    # 使用 repl 函数替换表达式中的 Lt 符号
    expr = repl(expr, Lt, replie)
    # 使用 repl 函数替换表达式中的 Gt 符号
    expr = repl(expr, Gt, lambda x, y: replie(y, x))
    # 使用 repl 函数替换表达式中的 Unequality 符号
    expr = repl(expr, Unequality, replue)
    # 返回简化后的表达式
    return S(expr)


# 装饰器函数，用于将函数调用包装在调试输出中
@DEBUG_WRAP
# 对表达式中的 DiracDelta 进行展开，得到其作为 DiracDelta 函数的线性组合
def expand_dirac_delta(expr):
    """
    Expand an expression involving DiractDelta to get it as a linear
    combination of DiracDelta functions.
    """
    # 调用 _lin_eq2dict 函数，将表达式转换为包含 DiracDelta 的线性组合形式
    return _lin_eq2dict(expr, expr.atoms(DiracDelta))


# 积分求解 Laplace 变换的后端函数
def _laplace_transform_integration(f, t, s_, *, simplify):
    """ The backend function for doing Laplace transforms by integration.
    """
    This backend assumes that the frontend has already split sums
    such that `f` is not an addition anymore.
    """
    # 创建一个符号变量 s
    s = Dummy('s')

    # 如果 f 中包含 DiracDelta，则返回空
    if f.has(DiracDelta):
        return None

    # 计算 f * exp(-s*t) 在 (t, 0, 无穷大) 范围内的积分 F
    F = integrate(f * exp(-s*t), (t, S.Zero, S.Infinity))

    # 如果 F 中不含有积分符号，进行简化并返回结果
    if not F.has(Integral):
        return _simplify(F.subs(s, s_), simplify), S.NegativeInfinity, S.true

    # 如果 F 不是分段函数，则返回空
    if not F.is_Piecewise:
        return None

    # 获取分段函数的第一个分支 F 和其条件 cond
    F, cond = F.args[0]
    
    # 如果 F 中包含积分符号，则返回空
    if F.has(Integral):
        return None
    def process_conds(conds):
        """ Turn ``conds`` into a strip and auxiliary conditions. """
        # 导入求解不等式的函数
        from sympy.solvers.inequalities import _solve_inequality
        # 初始条件设置为负无穷和真
        a = S.NegativeInfinity
        aux = S.true
        # 将条件转换为合取范式
        conds = conjuncts(to_cnf(conds))
        # 定义符号变量
        p, q, w1, w2, w3, w4, w5 = symbols(
            'p q w1 w2 w3 w4 w5', cls=Wild, exclude=[s])
        # 设定模式
        patterns = (
            p*Abs(arg((s + w3)*q)) < w2,
            p*Abs(arg((s + w3)*q)) <= w2,
            Abs(periodic_argument((s + w3)**p*q, w1)) < w2,
            Abs(periodic_argument((s + w3)**p*q, w1)) <= w2,
            Abs(periodic_argument((polar_lift(s + w3))**p*q, w1)) < w2,
            Abs(periodic_argument((polar_lift(s + w3))**p*q, w1)) <= w2)
        # 遍历每个条件
        for c in conds:
            # 初始化无穷大和空辅助条件列表
            a_ = S.Infinity
            aux_ = []
            # 对于每个析取范式中的条件
            for d in disjuncts(c):
                # 如果条件中包含 s 并且 s 是右侧自由符号，进行反转
                if d.is_Relational and s in d.rhs.free_symbols:
                    d = d.reversed
                # 如果条件是关系型并且是大于等于或者大于，则进行符号反转
                if d.is_Relational and isinstance(d, (Ge, Gt)):
                    d = d.reversedsign
                # 遍历模式列表，尝试匹配条件
                for pat in patterns:
                    m = d.match(pat)
                    if m:
                        break
                # 如果匹配成功且 q 是正数且 w2/p 等于 pi/2，则进行特定条件的替换
                if m and m[q].is_positive and m[w2]/m[p] == pi/2:
                    d = -re(s + m[w3]) < 0
                # 如果没有匹配到，尝试其他模式
                m = d.match(p - cos(w1*Abs(arg(s*w5))*w2)*Abs(s**w3)**w4 < 0)
                if not m:
                    m = d.match(
                        cos(p - Abs(periodic_argument(s**w1*w5, q))*w2) *
                        Abs(s**w3)**w4 < 0)
                if not m:
                    m = d.match(
                        p - cos(
                            Abs(periodic_argument(polar_lift(s)**w1*w5, q))*w2
                            )*Abs(s**w3)**w4 < 0)
                # 如果匹配成功并且所有变量都是正数，则进行条件替换
                if m and all(m[wild].is_positive for wild in [
                        w1, w2, w3, w4, w5]):
                    d = re(s) > m[p]
                # 对条件中的 re(s) 进行展开和实部提取，替换为 t
                d_ = d.replace(
                    re, lambda x: x.expand().as_real_imag()[0]).subs(re(s), t)
                # 如果条件不是关系型或者是 '=='、'!='，或者包含 s 或不包含 t，则将其加入辅助条件列表
                if (
                        not d.is_Relational or d.rel_op in ('==', '!=')
                        or d_.has(s) or not d_.has(t)):
                    aux_ += [d]
                    continue
                # 解不等式，得到解决方案
                soln = _solve_inequality(d_, t)
                # 如果解不是关系型或者是 '=='、'!='，则将其加入辅助条件列表
                if not soln.is_Relational or soln.rel_op in ('==', '!='):
                    aux_ += [d]
                    continue
                # 如果解小于 t，则返回空
                if soln.lts == t:
                    return None
                else:
                    # 更新 a 的值为解的最大值
                    a_ = Min(soln.lts, a_)
            # 如果 a_ 不是负无穷，则更新 a 的值为 a_ 和当前 a 的最大值
            if a_ is not S.Infinity:
                a = Max(a_, a)
            else:
                # 否则更新辅助条件为当前辅助条件与新辅助条件的或
                aux = And(aux, Or(*aux_))
        # 返回结果，如果 aux 是关系型则返回其规范形式，否则返回 aux
        return a, aux.canonical if aux.is_Relational else aux

    # 对条件列表中的每个条件应用 process_conds 函数，返回结果列表
    conds = [process_conds(c) for c in disjuncts(cond)]
    # 从 conds 中筛选出辅助条件不是 S.false 且 a 不是负无穷的条件列表
    conds2 = [x for x in conds if x[1] != S.false and x[0] is not S.NegativeInfinity]
    # 如果 conds2 为空，则重新从 conds 中筛选出辅助条件不是 S.false 的条件列表
    if not conds2:
        conds2 = [x for x in conds if x[1] != S.false]
    # 对 conds2 列表进行排序，并转换为列表形式
    conds = list(ordered(conds2))
    # 定义一个函数 cnt，用于计算表达式中运算符的数量
    def cnt(expr):
        # 如果表达式是布尔值 True 或 False，返回 0
        if expr in (True, False):
            return 0
        # 否则调用表达式的 count_ops() 方法计算运算符数量并返回
        return expr.count_ops()

    # 对 conds 列表进行排序，排序依据是一个 lambda 函数：
    # 先按 x[0] 的负值降序排列，再按 cnt(x[1]) 的结果排序
    conds.sort(key=lambda x: (-x[0], cnt(x[1])))

    # 如果 conds 列表为空，则返回 None
    if not conds:
        return None
    
    # 取出 conds 列表中的第一个元素，并赋值给变量 a 和 aux
    a, aux = conds[0]  # XXX is [0] always the right one?

    # 定义一个函数 sbs，用于替换表达式中的符号 s 为 s_
    def sbs(expr):
        return expr.subs(s, s_)

    # 如果 simplify 为 True，则对 F 和 aux 进行简化操作
    if simplify:
        F = _simplifyconds(F, s, a)
        aux = _simplifyconds(aux, s, a)

    # 返回三个值的元组：
    # 1. 对 F 进行符号替换 s -> s_ 后再进行简化操作的结果
    # 2. 对 a 进行符号替换 s -> s_ 的结果
    # 3. 对 aux 进行符号替换 s -> s_ 后再进行规范化操作的结果
    return _simplify(F.subs(s, s_), simplify), sbs(a), _canonical(sbs(aux))
# 调试包装装饰器，用于标记调试相关的函数或方法
@DEBUG_WRAP
# 这是一个内部辅助函数，用于遍历 `f(t)` 表达式树并收集参数。
# 其目的在于将类似 `f(w*t-1*t-c)` 的表达式转换为 `f((w-1)*t-c)`，以便匹配 `f(a*t+b)`。
def _laplace_deep_collect(f, t):
    if not isinstance(f, Expr):  # 如果 `f` 不是 SymPy 的表达式对象，则直接返回 `f`
        return f
    if (p := f.as_poly(t)) is not None:  # 如果 `f` 可以表示为关于 `t` 的多项式，则返回其表达式形式
        return p.as_expr()
    func = f.func  # 获取函数的类型信息
    # 递归地对 `f` 的每个参数调用 `_laplace_deep_collect`，并用处理后的参数列表重新构造函数调用
    args = [_laplace_deep_collect(arg, t) for arg in f.args]
    return func(*args)


# 缓存装饰器，用于缓存函数的调用结果
@cacheit
# 这是一个内部辅助函数，返回 Laplace 变换规则表
# 规则以时间变量 `t` 和频率变量 `s` 表示，供 `_laplace_apply_rules` 使用
def _laplace_build_rules():
    t = Dummy('t')  # 创建虚拟变量 `t`
    s = Dummy('s')  # 创建虚拟变量 `s`
    a = Wild('a', exclude=[t])  # 创建通配符 `a`，排除与 `t` 相同的情况
    b = Wild('b', exclude=[t])  # 创建通配符 `b`，排除与 `t` 相同的情况
    n = Wild('n', exclude=[t])  # 创建通配符 `n`，排除与 `t` 相同的情况
    tau = Wild('tau', exclude=[t])  # 创建通配符 `tau`，排除与 `t` 相同的情况
    omega = Wild('omega', exclude=[t])  # 创建通配符 `omega`，排除与 `t` 相同的情况
    # 定义函数 `dco`，它接受一个参数 `f` 并调用 `_laplace_deep_collect` 处理它
    def dco(f): return _laplace_deep_collect(f, t)
    _debug('_laplace_build_rules is building rules')  # 调试信息：正在构建规则
    # 返回 Laplace 变换规则、时间变量 `t`、频率变量 `s`
    return laplace_transform_rules, t, s


# 调试包装装饰器，用于标记调试相关的函数或方法
@DEBUG_WRAP
# 这个函数应用 Laplace 变换的时间缩放规则
# 例如，如果输入 `(f(a*t), t, s)`，则会计算 `LaplaceTransform(f(t)/a, t, s/a)`，如果 `a > 0`
def _laplace_rule_timescale(f, t, s):
    a = Wild('a', exclude=[t])  # 创建通配符 `a`，排除与 `t` 相同的情况
    g = WildFunction('g', nargs=1)  # 创建具有一个参数的通配符函数 `g`
    ma1 = f.match(g)  # 尝试匹配 `f` 到 `g`
    if ma1:  # 如果匹配成功
        arg = ma1[g].args[0].collect(t)  # 收集参数 `g` 中的 `t`
        ma2 = arg.match(a*t)  # 尝试匹配参数 `arg` 到 `a*t`
        if ma2 and ma2[a].is_positive and ma2[a] != 1:  # 如果匹配成功且 `a` 是正数且不等于 1
            _debug('     rule: time scaling (4.1.4)')  # 调试信息：应用时间缩放规则
            # 调用 `_laplace_transform` 函数，应用 Laplace 变换规则
            r, pr, cr = _laplace_transform(1/ma2[a]*ma1[g].func(t), t, s/ma2[a], simplify=False)
            return (r, pr, cr)  # 返回计算结果
    return None  # 如果规则不适用，返回 None


# 调试包装装饰器，用于标记调试相关的函数或方法
@DEBUG_WRAP
# 这个函数处理时间偏移的 Heaviside 阶跃函数
# 如果时间偏移为正，则应用 Laplace 变换的时间偏移规则
# 例如，如果输入 `(Heaviside(t-a)*f(t), t, s)`，则会计算 `exp(-a*s)*LaplaceTransform(f(t+a), t, s)`
# 如果时间偏移为负，则简单地移除 Heaviside 函数，因为对 Laplace 变换无影响
# 函数不会移除因子 `Heaviside(t)`，这由简单规则处理
def _laplace_rule_heaviside(f, t, s):
    a = Wild('a', exclude=[t])  # 创建通配符 `a`，排除与 `t` 相同的情况
    y = Wild('y')  # 创建通配符 `y`
    g = Wild('g')  # 创建通配符 `g`
    # 这里将继续添加函数的内容，但已经结束了。
    # 如果匹配到 ma1 = f.match(Heaviside(y) * g)，进入条件判断
    if ma1 := f.match(Heaviside(y) * g):
        # 如果匹配到 ma2 = ma1[y].match(t - a)，进入条件判断
        if ma2 := ma1[y].match(t - a):
            # 如果 ma2[a] 是正数，执行时间移位操作
            if ma2[a].is_positive:
                # 调试信息：规则为时间移位 (4.1.4)
                _debug('     rule: time shift (4.1.4)')
                # 对 ma1[g] 在 t -> t + ma2[a] 的情况进行 Laplace 变换
                r, pr, cr = _laplace_transform(
                    ma1[g].subs(t, t + ma2[a]), t, s, simplify=False)
                # 返回结果，包括变换后的表达式及其相关信息
                return (exp(-ma2[a] * s) * r, pr, cr)
            # 如果 ma2[a] 是负数，执行负时间移位操作
            if ma2[a].is_negative:
                # 调试信息：规则为 Heaviside 因子；负时间移位 (4.1.4)
                _debug(
                    '     rule: Heaviside factor; negative time shift (4.1.4)')
                # 对 ma1[g] 在 t 上进行 Laplace 变换
                r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
                # 返回结果，包括变换后的表达式及其相关信息
                return (r, pr, cr)
        # 如果匹配到 ma2 = ma1[y].match(a - t)，进入条件判断
        if ma2 := ma1[y].match(a - t):
            # 如果 ma2[a] 是正数，执行 Heaviside 窗口开放操作
            if ma2[a].is_positive:
                # 调试信息：规则为 Heaviside 窗口开放
                _debug('     rule: Heaviside window open')
                # 对 (1 - Heaviside(t - ma2[a])) * ma1[g] 进行 Laplace 变换
                r, pr, cr = _laplace_transform(
                    (1 - Heaviside(t - ma2[a])) * ma1[g], t, s, simplify=False)
                # 返回结果，包括变换后的表达式及其相关信息
                return (r, pr, cr)
            # 如果 ma2[a] 是负数，执行 Heaviside 窗口关闭操作
            if ma2[a].is_negative:
                # 调试信息：规则为 Heaviside 窗口关闭
                _debug('     rule: Heaviside window closed')
                # 返回零值及相关信息，表示 Laplace 变换为零
                return (0, 0, S.true)
    # 如果没有匹配成功，返回空值
    return None
# 调试包装器，用于 Laplace 变换规则函数
@DEBUG_WRAP
def _laplace_rule_exp(f, t, s):
    """
    If this function finds a factor ``exp(a*t)``, it applies the
    frequency-shift rule of the Laplace transform and adjusts the convergence
    plane accordingly.  For example, if it gets ``(exp(-a*t)*f(t), t, s)``, it
    will compute ``LaplaceTransform(f(t), t, s+a)``.
    """

    # 定义通配符
    a = Wild('a', exclude=[t])
    y = Wild('y')
    z = Wild('z')
    
    # 匹配表达式是否包含形如 exp(y)*z 的因子
    ma1 = f.match(exp(y)*z)
    if ma1:
        # 如果匹配成功，尝试将 exp(y) 中的项收集为 a*t 的形式
        ma2 = ma1[y].collect(t).match(a*t)
        if ma2:
            # 输出调试信息
            _debug('     rule: multiply with exp (4.1.5)')
            # 执行 Laplace 变换
            r, pr, cr = _laplace_transform(ma1[z], t, s-ma2[a], simplify=False)
            return (r, pr+re(ma2[a]), cr)
    return None


@DEBUG_WRAP
def _laplace_rule_delta(f, t, s):
    """
    If this function finds a factor ``DiracDelta(b*t-a)``, it applies the
    masking property of the delta distribution. For example, if it gets
    ``(DiracDelta(t-a)*f(t), t, s)``, it will return
    ``(f(a)*exp(-a*s), -a, True)``.
    """
    # This rule is not in Bateman54

    # 定义通配符
    a = Wild('a', exclude=[t])
    b = Wild('b', exclude=[t])

    y = Wild('y')
    z = Wild('z')
    
    # 匹配表达式是否包含形如 DiracDelta(y)*z 的因子，并且 z 中不含有 DiracDelta
    ma1 = f.match(DiracDelta(y)*z)
    if ma1 and not ma1[z].has(DiracDelta):
        # 尝试将 DiracDelta(y) 匹配为 b*t-a 的形式
        ma2 = ma1[y].collect(t).match(b*t-a)
        if ma2:
            # 输出调试信息
            _debug('     rule: multiply with DiracDelta')
            # 计算位置参数
            loc = ma2[a]/ma2[b]
            if re(loc) >= 0 and im(loc) == 0:
                # 构造新的表达式 fn
                fn = exp(-ma2[a]/ma2[b]*s)*ma1[z]
                if fn.has(sin, cos):
                    # 如果 fn 中包含 sin 或 cos，尝试用 sinc() 重写
                    fn = fn.rewrite(sinc).ratsimp()
                # 将 fn 分子分母分别在 t=a/b 处求值
                n, d = [x.subs(t, ma2[a]/ma2[b]) for x in fn.as_numer_denom()]
                if d != 0:
                    return (n/d/ma2[b], S.NegativeInfinity, S.true)
                else:
                    return None
            else:
                return (0, S.NegativeInfinity, S.true)
        if ma1[y].is_polynomial(t):
            # 如果 y 是 t 的多项式，则尝试求其根
            ro = roots(ma1[y], t)
            if ro != {} and set(ro.values()) == {1}:
                # 计算斜率
                slope = diff(ma1[y], t)
                # 构造结果表达式 r
                r = Add(*[exp(-x*s)*ma1[z].subs(t, s)/slope.subs(t, x)
                          for x in list(ro.keys()) if im(x) == 0 and re(x) >= 0])
                return (r, S.NegativeInfinity, S.true)
    return None


@DEBUG_WRAP
def _laplace_trig_split(fn):
    """
    Helper function for `_laplace_rule_trig`.  This function returns two terms
    `f` and `g`.  `f` contains all product terms with sin, cos, sinh, cosh in
    them; `g` contains everything else.
    """
    # 初始化空列表
    trigs = [S.One]
    other = [S.One]
    # 遍历 fn 中的每一项
    for term in Mul.make_args(fn):
        # 如果包含 sin, cos, sinh, cosh 或 exp，则加入到 trigs 中，否则加入到 other 中
        if term.has(sin, cos, sinh, cosh, exp):
            trigs.append(term)
        else:
            other.append(term)
    # 构造结果 f 和 g
    f = Mul(*trigs)
    g = Mul(*other)
    return f, g


@DEBUG_WRAP
def _laplace_trig_expsum(f, t):
    """
    # 定义 `_laplace_rule_trig` 的辅助函数，用于处理 `f` 参数（来自 `_laplace_trig_split` 函数）。
    # 返回两个列表 `xm` 和 `xn`。`xm` 是一个包含字典的列表，字典有键 `k` 和 `a`，表示形如 `k*exp(a*t)` 的函数。
    # `xn` 是所有无法转换为这种形式的项的列表，例如当三角函数的参数中包含另一个函数时可能发生。
    """
    c1 = Wild('c1', exclude=[t])  # 定义一个通配符 `c1`，排除 `t` 以外的符号
    c0 = Wild('c0', exclude=[t])  # 定义一个通配符 `c0`，排除 `t` 以外的符号
    p = Wild('p', exclude=[t])    # 定义一个通配符 `p`，排除 `t` 以外的符号
    xm = []                       # 初始化空列表 `xm`，用于存储符合条件的项
    xn = []                       # 初始化空列表 `xn`，用于存储不符合条件的项
    
    x1 = f.rewrite(exp).expand()  # 对 `f` 应用指数重写并展开，存储结果到 `x1`
    
    for term in Add.make_args(x1):  # 遍历 `x1` 中的每一项
        if not term.has(t):         # 如果当前项中不包含 `t`
            xm.append({'k': term, 'a': 0, re: 0, im: 0})  # 将该项作为 `k*exp(0*t)` 形式加入 `xm` 列表
            continue
    
        term = _laplace_deep_collect(term.powsimp(combine='exp'), t)  # 对当前项进行深度收集和幂简化，用 `t` 作为参数
    
        if (r := term.match(p*exp(c1*t+c0))) is not None:  # 如果当前项可以匹配 `p*exp(c1*t+c0)` 的形式
            xm.append({
                'k': r[p]*exp(r[c0]),        # 将匹配结果中的 `p*exp(c0)` 作为 `k`
                'a': r[c1],                  # 将匹配结果中的 `c1` 作为 `a`
                re: re(r[c1]),               # 计算 `c1` 的实部
                im: im(r[c1])})              # 计算 `c1` 的虚部
        else:
            xn.append(term)  # 如果无法匹配到 `p*exp(c1*t+c0)` 的形式，则将当前项加入 `xn` 列表
    
    return xm, xn  # 返回处理结果 `xm` 和 `xn`
# 装饰器，用于调试包装函数，但在此处未提供实际装饰器的定义
@DEBUG_WRAP
# `_laplace_trig_ltex` 函数，用于处理 `_laplace_rule_trig` 的辅助功能，处理指数列表 `xm`，简化复共轭和实对称极点
def _laplace_trig_ltex(xm, t, s):
    """
    Helper function for `_laplace_rule_trig`.  This function takes the list of
    exponentials `xm` from `_laplace_trig_expsum` and simplifies complex
    conjugate and real symmetric poles.  It returns the result as a sum and
    the convergence plane.
    """
    # 结果列表
    results = []
    # 平面列表
    planes = []

    # 简化复极点和实对称极点的系数处理函数
    def _simpc(coeffs):
        nc = coeffs.copy()
        for k in range(len(nc)):
            ri = nc[k].as_real_imag()
            if ri[0].has(im):
                nc[k] = nc[k].rewrite(cos)
            else:
                nc[k] = (ri[0] + I*ri[1]).rewrite(cos)
        return nc

    # 处理四极点的函数
    def _quadpole(t1, k1, k2, k3, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [
            k0 + k1 + k2 + k3,
            a*(k0 + k1 - k2 - k3) - 2*I*a_i*k1 + 2*I*a_i*k2,
            (
                a**2*(-k0 - k1 - k2 - k3) +
                a*(4*I*a_i*k0 + 4*I*a_i*k3) +
                4*a_i**2*k0 + 4*a_i**2*k3),
            (
                a**3*(-k0 - k1 + k2 + k3) +
                a**2*(4*I*a_i*k0 + 2*I*a_i*k1 - 2*I*a_i*k2 - 4*I*a_i*k3) +
                a*(4*a_i**2*k0 - 4*a_i**2*k3))
        ]
        dc = [
            S.One, S.Zero, 2*a_i**2 - 2*a_r**2,
            S.Zero, a_i**4 + 2*a_i**2*a_r**2 + a_r**4]
        # 计算四极点的数值部分
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        # 计算四极点的分母部分
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    # 处理共轭极点的函数
    def _ccpole(t1, k1, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [k0 + k1, -a*k0 - a*k1 + 2*I*a_i*k0]
        dc = [S.One, -2*a_r, a_i**2 + a_r**2]
        # 计算共轭极点的数值部分
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        # 计算共轭极点的分母部分
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    # 处理实对称极点的函数
    def _rspole(t1, k2, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [k0 + k2, a*k0 - a*k2 - 2*I*a_i*k0]
        dc = [S.One, -2*I*a_i, -a_i**2 - a_r**2]
        # 计算实对称极点的数值部分
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        # 计算实对称极点的分母部分
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    # 处理简单极点的函数
    def _sypole(t1, k3, s):
        a, k0 = t1['a'], t1['k']
        nc = [k0 + k3, a*(k0 - k3)]
        dc = [S.One, S.Zero, -a**2]
        # 计算简单极点的数值部分
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        # 计算简单极点的分母部分
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    # 处理单一极点的函数
    def _simplepole(t1, s):
        a, k0 = t1['a'], t1['k']
        n = k0
        d = s - a
        return n/d
    # 当 xm 列表非空时执行循环
    while len(xm) > 0:
        # 弹出 xm 列表的最后一个元素，并赋值给 t1
        t1 = xm.pop()
        
        # 初始化索引变量
        i_imagsym = None
        i_realsym = None
        i_pointsym = None
        
        # 下面的代码检查所有剩余的极点。如果 t1 是形如 a+b*I 的极点，
        # 则检查是否存在 a-b*I, -a+b*I 和 -a-b*I 这几种极点，并分别
        # 将索引赋值给 i_imagsym, i_realsym, i_pointsym。
        # 当 a 和 b 都不为零时，才会将 -a-b*I 的索引赋给 i_pointsym。
        for i in range(len(xm)):
            real_eq = t1[re] == xm[i][re]
            realsym = t1[re] == -xm[i][re]
            imag_eq = t1[im] == xm[i][im]
            imagsym = t1[im] == -xm[i][im]
            if realsym and imagsym and t1[re] != 0 and t1[im] != 0:
                i_pointsym = i
            elif realsym and imag_eq and t1[re] != 0:
                i_realsym = i
            elif real_eq and imagsym and t1[im] != 0:
                i_imagsym = i
        
        # 接下来的部分检查四种可能的极点配置：
        # quad:   a+b*I, a-b*I, -a+b*I, -a-b*I
        # cc:     a+b*I, a-b*I (a 可能为零)
        # quad:   a+b*I, -a+b*I (b 可能为零)
        # point:  a+b*I, -a-b*I (需要 a!=0 和 b!=0，这在上面找到 i_pointsym 时已经断言过)
        # 如果以上情况都不符合，则 t1 是一个简单的极点。
        if (
                i_imagsym is not None and i_realsym is not None
                and i_pointsym is not None):
            # 将 t1 及其对应的三个额外极点的数据传递给 _quadpole 函数，并将结果添加到 results 列表中
            results.append(
                _quadpole(t1,
                          xm[i_imagsym]['k'], xm[i_realsym]['k'],
                          xm[i_pointsym]['k'], s))
            # 将 t1['a'] 的绝对值添加到 planes 列表中
            planes.append(Abs(re(t1['a'])))
            # 这三个额外的极点已经被使用，为了方便地弹出它们，需要从后向前排序
            indices_to_pop = [i_imagsym, i_realsym, i_pointsym]
            indices_to_pop.sort(reverse=True)
            for i in indices_to_pop:
                # 弹出 xm 列表中索引为 i 的元素
                xm.pop(i)
        elif i_imagsym is not None:
            # 将 t1 及其对应的 imag 类型极点数据传递给 _ccpole 函数，并将结果添加到 results 列表中
            results.append(_ccpole(t1, xm[i_imagsym]['k'], s))
            # 将 t1[re] 添加到 planes 列表中
            planes.append(t1[re])
            # 弹出 xm 列表中索引为 i_imagsym 的元素
            xm.pop(i_imagsym)
        elif i_realsym is not None:
            # 将 t1 及其对应的 real 类型极点数据传递给 _rspole 函数，并将结果添加到 results 列表中
            results.append(_rspole(t1, xm[i_realsym]['k'], s))
            # 将 t1[re] 的绝对值添加到 planes 列表中
            planes.append(Abs(t1[re]))
            # 弹出 xm 列表中索引为 i_realsym 的元素
            xm.pop(i_realsym)
        elif i_pointsym is not None:
            # 将 t1 及其对应的 point 类型极点数据传递给 _sypole 函数，并将结果添加到 results 列表中
            results.append(_sypole(t1, xm[i_pointsym]['k'], s))
            # 将 t1[re] 的绝对值添加到 planes 列表中
            planes.append(Abs(t1[re]))
            # 弹出 xm 列表中索引为 i_pointsym 的元素
            xm.pop(i_pointsym)
        else:
            # 如果以上所有情况都不适用，则将 t1 作为简单极点处理，并将结果添加到 results 列表中
            results.append(_simplepole(t1, s))
            # 将 t1[re] 添加到 planes 列表中
            planes.append(t1[re])

    # 返回 results 列表中所有元素的和作为第一个返回值，返回 planes 列表中最大值作为第二个返回值
    return Add(*results), Max(*planes)
# 对于给定函数 `fn`，处理包含三角函数的情况，将其拆分为指数函数的和，并收集复共轭极点和实对称极点
@DEBUG_WRAP
def _laplace_rule_trig(fn, t_, s):
    """
    This rule covers trigonometric factors by splitting everything into a
    sum of exponential functions and collecting complex conjugate poles and
    real symmetric poles.
    """
    # 定义一个实数虚拟变量 `t`
    t = Dummy('t', real=True)

    # 如果 `fn` 不包含 sin、cos、sinh、cosh 函数，则直接返回 None
    if not fn.has(sin, cos, sinh, cosh):
        return None

    # 将 `t_` 替换为 `t`，然后对 `fn` 进行三角函数拆分
    f, g = _laplace_trig_split(fn.subs(t_, t))
    # 计算 `f` 的指数函数和
    xm, xn = _laplace_trig_expsum(f, t)

    # 如果有未实现的情况（`xn` 非空），暂时返回 None
    if len(xn) > 0:
        # TODO not implemented yet, but also not important
        return None

    # 如果 `g` 不含有 `t`，则计算 `g` 的 Laplace 变换及其相关的频域条件
    if not g.has(t):
        r, p = _laplace_trig_ltex(xm, t, s)
        return g*r, p, S.true
    else:
        # 否则，对 `g` 进行变换并创建频率移位的副本
        planes = []
        results = []
        G, G_plane, G_cond = _laplace_transform(g, t, s, simplify=False)
        # 遍历 `xm` 中的每个元素，应用频率移位并添加到结果中
        for x1 in xm:
            results.append(x1['k']*G.subs(s, s-x1['a']))
            planes.append(G_plane+re(x1['a']))
        return Add(*results).subs(t, t_), Max(*planes), G_cond


# 对于给定函数 `f`，在时域寻找导数，并在频域进行替换为 `s` 的因子和初始条件
@DEBUG_WRAP
def _laplace_rule_diff(f, t, s):
    """
    This function looks for derivatives in the time domain and replaces it
    by factors of `s` and initial conditions in the frequency domain. For
    example, if it gets ``(diff(f(t), t), t, s)``, it will compute
    ``s*LaplaceTransform(f(t), t, s) - f(0)``.
    """

    # 定义排除 `t` 的通配符 `a` 和 `n`
    a = Wild('a', exclude=[t])
    n = Wild('n', exclude=[t])
    g = WildFunction('g')

    # 匹配 `f` 中的 `a*Derivative(g, (t, n))` 形式的表达式
    ma1 = f.match(a*Derivative(g, (t, n)))

    # 如果匹配成功且 `n` 是整数
    if ma1 and ma1[n].is_integer:
        # 检查 `ma1[g]` 的参数中是否含有 `t`
        m = [z.has(t) for z in ma1[g].args]
        if sum(m) == 1:
            _debug('     rule: time derivative (4.1.8)')
            d = []
            # 对于 `ma1[n]` 次导数的每一项，计算其在 `t=0` 处的值乘以相应的 `s` 次幂
            for k in range(ma1[n]):
                if k == 0:
                    y = ma1[g].subs(t, 0)
                else:
                    y = Derivative(ma1[g], (t, k)).subs(t, 0)
                d.append(s**(ma1[n]-k-1)*y)
            # 计算 `ma1[g]` 的 Laplace 变换，并替换为频域中的表达式
            r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
            return (ma1[a]*(s**ma1[n]*r - Add(*d)),  pr, cr)
    # 如果匹配失败，返回 None
    return None


# 对于给定函数 `f`，在频域寻找与 `t` 多项式乘积对应的导数
@DEBUG_WRAP
def _laplace_rule_sdiff(f, t, s):
    """
    This function looks for multiplications with polynomials in `t` as they
    correspond to differentiation in the frequency domain. For example, if it
    gets ``(t*f(t), t, s)``, it will compute
    ``-Derivative(LaplaceTransform(f(t), t, s), s)``.
    """

    # 匹配 `f` 中形如 `t*f(t)` 的表达式
    a = Wild('a', exclude=[t])
    g = WildFunction('g')
    ma1 = f.match(a*t*g)

    # 如果匹配成功，返回 `-Derivative(LaplaceTransform(g, t, s), s)`
    if ma1:
        return -Derivative(_laplace_transform(ma1[g], t, s, simplify=False)[0], s)

    # 如果匹配失败，返回 None
    return None
    # 检查表达式是否为乘法表达式
    if f.is_Mul:
        # 初始化多项式因子和其他因子的列表
        pfac = [1]  # 多项式因子列表，初始为1
        ofac = [1]  # 其他因子列表，初始为1
        
        # 遍历乘法表达式的每个因子
        for fac in Mul.make_args(f):
            # 如果因子是关于变量t的多项式
            if fac.is_polynomial(t):
                pfac.append(fac)  # 将多项式因子添加到多项式因子列表
            else:
                ofac.append(fac)  # 否则将因子添加到其他因子列表
        
        # 如果存在多个多项式因子
        if len(pfac) > 1:
            # 计算多项式因子的乘积
            pex = prod(pfac)
            # 将多项式转化为多项式对象，并获取其所有系数
            pc = Poly(pex, t).all_coeffs()
            N = len(pc)  # 获取系数的个数
            
            # 如果系数个数大于1
            if N > 1:
                # 计算其他因子的乘积
                oex = prod(ofac)
                # 进行 Laplace 变换，获取结果和参数
                r_, p_, c_ = _laplace_transform(oex, t, s, simplify=False)
                deri = [r_]  # 初始化导数列表，初始值为r_
                d1 = False
                
                # 尝试计算r_关于s的一阶导数
                try:
                    d1 = -diff(deri[-1], s)
                except ValueError:
                    d1 = False
                
                # 如果r_包含 LaplaceTransform
                if r_.has(LaplaceTransform):
                    # 对r_进行多阶导数计算
                    for k in range(N-1):
                        deri.append((-1)**(k+1)*Derivative(r_, s, k+1))
                elif d1:
                    # 如果能够计算得到一阶导数d1
                    deri.append(d1)
                    # 计算剩余的高阶导数
                    for k in range(N-2):
                        deri.append(-diff(deri[-1], s))
                
                # 如果存在一阶导数d1
                if d1:
                    # 计算最终的结果表达式r
                    r = Add(*[pc[N-n-1]*deri[n] for n in range(N)])
                    return (r, p_, c_)  # 返回结果及其它参数
    
    # 若以上条件均不满足，则继续检查是否有符号正整数幂的情况
    n = Wild('n', exclude=[t])  # 创建一个符号正整数幂的通配符
    g = Wild('g')  # 创建一个通配符g
    
    # 尝试匹配表达式是否符合形式 t**n * g
    if ma1 := f.match(t**n*g):
        # 如果匹配成功且n是整数且大于0
        if ma1[n].is_integer and ma1[n].is_positive:
            # 进行 Laplace 变换，获取结果和参数
            r_, p_, c_ = _laplace_transform(ma1[g], t, s, simplify=False)
            # 计算结果表达式的导数
            return (-1)**ma1[n]*diff(r_, (s, ma1[n])), p_, c_
    
    return None  # 如果没有匹配成功的情况，则返回None
# 定义一个函数，在尝试不同的扩展方法失败后，尝试计算参数的 Laplace 变换。
@DEBUG_WRAP
def _laplace_expand(f, t, s):
    """
    This function tries to expand its argument with successively stronger
    methods: first it will expand on the top level, then it will expand any
    multiplications in depth, then it will try all available expansion methods,
    and finally it will try to expand trigonometric functions.

    If it can expand, it will then compute the Laplace transform of the
    expanded term.
    """
    
    # 首先尝试在顶层展开表达式，不进行深度展开
    r = expand(f, deep=False)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    
    # 如果顶层展开失败，尝试深度展开表达式中的乘法
    r = expand_mul(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    
    # 如果乘法展开失败，尝试所有可用的展开方法
    r = expand(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    
    # 如果前面的展开方法都失败，并且展开后与原始表达式不同，则尝试计算其 Laplace 变换
    if r != f:
        return _laplace_transform(r, t, s, simplify=False)
    
    # 如果以上方法都失败，尝试对三角函数进行展开并计算其 Laplace 变换
    r = expand(expand_trig(f))
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    
    # 如果所有尝试都失败，返回 None
    return None


# 定义一个函数，应用所有程序规则并返回其中一个规则给出结果的情况。
@DEBUG_WRAP
def _laplace_apply_prog_rules(f, t, s):
    """
    This function applies all program rules and returns the result if one
    of them gives a result.
    """
    
    # 定义一组程序规则，按顺序尝试应用它们
    prog_rules = [_laplace_rule_heaviside, _laplace_rule_delta,
                  _laplace_rule_timescale, _laplace_rule_exp,
                  _laplace_rule_trig,
                  _laplace_rule_diff, _laplace_rule_sdiff]

    for p_rule in prog_rules:
        # 对每个规则依次调用，如果有结果则返回该结果
        if (L := p_rule(f, t, s)) is not None:
            return L
    # 如果所有规则都无结果，则返回 None
    return None


# 定义一个函数，应用所有简单规则并返回其中一个规则给出结果的情况。
@DEBUG_WRAP
def _laplace_apply_simple_rules(f, t, s):
    """
    This function applies all simple rules and returns the result if one
    of them gives a result.
    """
    
    # 调用函数构建简单规则集合
    simple_rules, t_, s_ = _laplace_build_rules()
    prep_old = ''
    prep_f = ''
    
    # 遍历简单规则集合，尝试应用每个规则
    for t_dom, s_dom, check, plane, prep in simple_rules:
        if prep_old != prep:
            prep_f = prep(f.subs({t: t_}))
            prep_old = prep
        ma = prep_f.match(t_dom)
        if ma:
            try:
                c = check.xreplace(ma)
            except TypeError:
                # 如果时间函数中存在虚数，则放弃当前规则
                continue
            # 如果检查通过，返回规则应用后的结果
            if c == S.true:
                return (s_dom.xreplace(ma).subs({s_: s}),
                        plane.xreplace(ma), S.true)
    
    # 如果所有规则都无结果，则返回 None
    return None


# 定义一个函数，将一个 Piecewise 表达式转换为使用 Heaviside 函数表示的表达式，
# 在 Laplace 变换的上下文中是有效的，但不是精确的。
@DEBUG_WRAP
def _piecewise_to_heaviside(f, t):
    """
    This function converts a Piecewise expression to an expression written
    with Heaviside. It is not exact, but valid in the context of the Laplace
    transform.
    """
    
    # 如果时间变量 t 不是实数，则定义一个实数虚拟变量 r 来进行转换
    if not t.is_real:
        r = Dummy('r', real=True)
        return _piecewise_to_heaviside(f.xreplace({t: r}), r).xreplace({r: t})
    
    # 对 Piecewise 表达式进行排他性处理，转换为 Heaviside 表示
    x = piecewise_exclusive(f)
    r = []
    # 遍历 x.args 中的每对 (fn, cond)
    for fn, cond in x.args:
        # 如果 cond 是 Relational 类型且 t 在其参数中
        if isinstance(cond, Relational) and t in cond.args:
            # 如果 cond 是 Eq 或者 Ne 类型
            if isinstance(cond, (Eq, Ne)):
                # 如果是 Eq 或 Ne 类型，返回 f，因为这些条件在 Laplace 变换中不起作用
                return f
            else:
                # 否则，根据 Heaviside 函数的结果计算 r.append(Heaviside(cond.gts - cond.lts)*fn)
                r.append(Heaviside(cond.gts - cond.lts) * fn)
        # 如果 cond 是 Or 类型且包含两个参数
        elif isinstance(cond, Or) and len(cond.args) == 2:
            # 遍历 Or 类型的两个参数 c2
            for c2 in cond.args:
                # 如果 c2 的左操作数是 t
                if c2.lhs == t:
                    # 根据 Heaviside 函数的结果计算 r.append(Heaviside(c2.gts - c2.lts)*fn)
                    r.append(Heaviside(c2.gts - c2.lts) * fn)
                else:
                    # 如果不是，则返回 f
                    return f
        # 如果 cond 是 And 类型且包含两个参数
        elif isinstance(cond, And) and len(cond.args) == 2:
            # 分别将 cond.args 的两个参数赋值给 c0 和 c1
            c0, c1 = cond.args
            # 如果 c0 和 c1 的左操作数都是 t
            if c0.lhs == t and c1.lhs == t:
                # 如果 c0 的关系运算符包含 '>'
                if '>' in c0.rel_op:
                    # 交换 c0 和 c1
                    c0, c1 = c1, c0
                # 根据 Heaviside 函数的结果计算 (Heaviside(c1.gts - c1.lts) - Heaviside(c0.lts - c0.gts))*fn
                r.append((Heaviside(c1.gts - c1.lts) - Heaviside(c0.lts - c0.gts)) * fn)
            else:
                # 如果不是，则返回 f
                return f
        else:
            # 如果不满足上述任何条件，则返回 f
            return f
    # 返回 Add(*r) 的结果，将 r 列表中的元素相加
    return Add(*r)
# 定义一个辅助函数 `laplace_correspondence`，用于处理包含 Laplace 变换或逆变换对象的表达式 `f`
def laplace_correspondence(f, fdict, /):
    # 导入通配符模块 Wild
    p = Wild('p')
    s = Wild('s')
    t = Wild('t')
    a = Wild('a')
    
    # 如果 f 不是表达式类型或者不包含 LaplaceTransform 或 InverseLaplaceTransform，直接返回 f
    if (
            not isinstance(f, Expr)
            or (not f.has(LaplaceTransform)
                and not f.has(InverseLaplaceTransform))):
        return f
    
    # 遍历 fdict 中的每一对 (y, Y)，其中 y 是原函数，Y 是其 Laplace 变换
    for y, Y in fdict.items():
        # 如果 f 匹配形如 LaplaceTransform(y(a), t, s) 的结构，并且 m[a] == m[t] 成立
        if (
                (m := f.match(LaplaceTransform(y(a), t, s))) is not None
                and m[a] == m[t]):
            # 返回 Y(m[s])，即使用对应的 Laplace 变换 Y(s)
            return Y(m[s])
        
        # 如果 f 匹配形如 InverseLaplaceTransform(Y(a), s, t, p) 的结构，并且 m[a] == m[s] 成立
        if (
                (m := f.match(InverseLaplaceTransform(Y(a), s, t, p)))
                is not None
                and m[a] == m[s]):
            # 返回 y(m[t])，即使用对应的逆 Laplace 变换 y(t)
            return y(m[t])
    
    # 获取 f 的函数和参数，递归调用 laplace_correspondence 处理参数中的每一个表达式，最后返回处理后的函数和参数
    func = f.func
    args = [laplace_correspondence(arg, fdict) for arg in f.args]
    return func(*args)
    # 导入 sympy 库中的函数和符号
    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, laplace_initial_conds
    >>> from sympy.abc import t, s
    # 定义两个函数 y 和 Y
    >>> y = Function("y")
    >>> Y = Function("Y")
    # 对 y(t) 进行三阶导数的拉普拉斯变换，结果赋给 f
    >>> f = laplace_transform(diff(y(t), t, 3), t, s, noconds=True)
    # 将 f 中的 y 替换为 Y，得到 g
    >>> g = laplace_correspondence(f, {y: Y})
    # 在 g 中应用初始条件 {y: [2, 4, 8, 16, 32]}，返回结果
    >>> laplace_initial_conds(g, t, {y: [2, 4, 8, 16, 32]})
    # 对于给定的字典 fdict 中的每个项进行处理
    """
    for y, ic in fdict.items():
        # 遍历初始条件列表 ic 的每个元素
        for k in range(len(ic)):
            # 如果 k 等于 0，用 ic[0] 替换 f 中的 y(0)
            if k == 0:
                f = f.replace(y(0), ic[0])
            # 如果 k 等于 1，用 ic[1] 替换 f 中的 Derivative(y(t), t).subs(t, 0)
            elif k == 1:
                f = f.replace(Subs(Derivative(y(t), t), t, 0), ic[1])
            # 否则，用 ic[k] 替换 f 中的 Derivative(y(t), (t, k)).subs(t, 0)
            else:
                f = f.replace(Subs(Derivative(y(t), (t, k)), t, 0), ic[k])
    # 返回处理后的结果 f
    return f
@DEBUG_WRAP
# 装饰器，用于调试包装函数，可能会添加额外的调试功能

def _laplace_transform(fn, t_, s_, *, simplify):
    """
    Front-end function of the Laplace transform. It tries to apply all known
    rules recursively, and if everything else fails, it tries to integrate.
    """
    # 将函数 fn 分解为加法操作的各项
    terms_t = Add.make_args(fn)
    terms_s = []  # 存储 Laplace 变换后的各项
    terms = []    # 存储处理后的各项
    planes = []   # 存储平面值
    conditions = []  # 存储条件

    # 遍历所有项进行处理
    for ff in terms_t:
        # 将 ff 表达式中的 t_ 独立出来
        k, ft = ff.as_independent(t_, as_Add=False)
        if ft.has(SingularityFunction):
            # 如果 ft 含有 SingularityFunction，重写为 Heaviside 函数
            _terms = Add.make_args(ft.rewrite(Heaviside))
            for _term in _terms:
                k1, f1 = _term.as_independent(t_, as_Add=False)
                terms.append((k*k1, f1))
        elif ft.func == Piecewise and not ft.has(DiracDelta(t_)):
            # 如果 ft 是 Piecewise 函数但不含 DiracDelta(t_)，转换为 Heaviside 函数
            _terms = Add.make_args(_piecewise_to_heaviside(ft, t_))
            for _term in _terms:
                k1, f1 = _term.as_independent(t_, as_Add=False)
                terms.append((k*k1, f1))
        else:
            terms.append((k, ft))

    # 对处理后的每一项进行 Laplace 变换
    for k, ft in terms:
        if ft.has(SingularityFunction):
            # 如果 ft 含有 SingularityFunction，计算其 Laplace 变换
            r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        else:
            if ft.has(Heaviside(t_)) and not ft.has(DiracDelta(t_)):
                # 对于 t>=0，Heaviside(t)=1，除非同时存在 DiracDelta(t)，这时保留 Heaviside(t)
                ft = ft.subs(Heaviside(t_), 1)
            # 尝试应用简单规则、递归规则、展开操作
            if (
                    (r := _laplace_apply_simple_rules(ft, t_, s_))
                    is not None or
                    (r := _laplace_apply_prog_rules(ft, t_, s_))
                    is not None or
                    (r := _laplace_expand(ft, t_, s_)) is not None):
                pass
            elif any(undef.has(t_) for undef in ft.atoms(AppliedUndef)):
                # 如果表达式中含有未定义函数 f(t)，则跳过积分，返回未计算的 Laplace 变换
                r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
            elif (r := _laplace_transform_integration(
                    ft, t_, s_, simplify=simplify)) is not None:
                pass
            else:
                r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        (ri_, pi_, ci_) = r
        terms_s.append(k*ri_)  # 将每项的 Laplace 变换结果加入 terms_s
        planes.append(pi_)     # 将平面值加入 planes
        conditions.append(ci_)  # 将条件值加入 conditions

    result = Add(*terms_s)  # 将所有 Laplace 变换后的项加起来得到最终结果
    if simplify:
        result = result.simplify(doit=False)  # 简化结果
    plane = Max(*planes)  # 计算平面值的最大值
    condition = And(*conditions)  # 计算条件的逻辑与

    return result, plane, condition


class LaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Laplace transforms, see the :func:`laplace_transform`
    docstring.
    """
    If this is called with ``.doit()``, it returns the Laplace transform as an
    expression. If it is called with ``.doit(noconds=False)``, it returns a
    tuple containing the same expression, a convergence plane, and conditions.
    """

    # LaplaceTransform 类的名称属性
    _name = 'Laplace'

    # 计算 Laplace 变换的内部方法，接受函数 f、变量 t、s，以及额外的 hints 参数
    def _compute_transform(self, f, t, s, **hints):
        _simplify = hints.get('simplify', False)
        # 调用 _laplace_transform_integration 函数进行 Laplace 变换的计算
        LT = _laplace_transform_integration(f, t, s, simplify=_simplify)
        return LT

    # 将 Laplace 变换表示为积分形式的方法，接受函数 f、变量 t、s
    def _as_integral(self, f, t, s):
        return Integral(f * exp(-s * t), (t, S.Zero, S.Infinity))

    # 执行 Laplace 变换的主要方法，根据 hints 参数进行不同模式的处理
    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """
        # 获取是否返回收敛条件的标志，默认为 True
        _noconds = hints.get('noconds', True)
        # 获取是否简化结果的标志，默认为 False
        _simplify = hints.get('simplify', False)

        # 调试输出信息，显示当前执行的 Laplace 变换的函数、函数变量、变换变量
        debugf('[LT doit] (%s, %s, %s)', (self.function,
                                          self.function_variable,
                                          self.transform_variable))

        # 获取函数变量和变换变量的引用
        t_ = self.function_variable
        s_ = self.transform_variable
        fn = self.function

        # 调用 _laplace_transform 函数计算 Laplace 变换结果
        r = _laplace_transform(fn, t_, s_, simplify=_simplify)

        # 根据 _noconds 参数决定返回的结果形式
        if _noconds:
            return r[0]  # 只返回变换后的表达式部分
        else:
            return r  # 返回包含表达式、收敛平面和条件的元组
# 定义计算拉普拉斯变换的函数 laplace_transform
def laplace_transform(f, t, s, legacy_matrix=True, **hints):
    r"""
    Compute the Laplace Transform `F(s)` of `f(t)`,

    .. math :: F(s) = \int_{0^{-}}^\infty e^{-st} f(t) \mathrm{d}t.

    Explanation
    ===========

    For all sensible functions, this converges absolutely in a
    half-plane

    .. math :: a < \operatorname{Re}(s)

    This function returns ``(F, a, cond)`` where ``F`` is the Laplace
    transform of ``f``, `a` is the half-plane of convergence, and `cond` are
    auxiliary convergence conditions.

    The implementation is rule-based, and if you are interested in which
    rules are applied, and whether integration is attempted, you can switch
    debug information on by setting ``sympy.SYMPY_DEBUG=True``. The numbers
    of the rules in the debug information (and the code) refer to Bateman's
    Tables of Integral Transforms [1].

    The lower bound is `0-`, meaning that this bound should be approached
    from the lower side. This is only necessary if distributions are involved.
    At present, it is only done if `f(t)` contains ``DiracDelta``, in which
    case the Laplace transform is computed implicitly as

    .. math ::
        F(s) = \lim_{\tau\to 0^{-}} \int_{\tau}^\infty e^{-st}
        f(t) \mathrm{d}t

    by applying rules.

    If the Laplace transform cannot be fully computed in closed form, this
    function returns expressions containing unevaluated
    :class:`LaplaceTransform` objects.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`. If
    ``noconds=True``, only `F` will be returned (i.e. not ``cond``, and also
    not the plane ``a``).

    .. deprecated:: 1.9
        Legacy behavior for matrices where ``laplace_transform`` with
        ``noconds=False`` (the default) returns a Matrix whose elements are
        tuples. The behavior of ``laplace_transform`` for matrices will change
        in a future release of SymPy to return a tuple of the transformed
        Matrix and the convergence conditions for the matrix as a whole. Use
        ``legacy_matrix=False`` to enable the new behavior.

    Examples
    ========

    >>> from sympy import DiracDelta, exp, laplace_transform
    >>> from sympy.abc import t, s, a
    >>> laplace_transform(t**4, t, s)
    (24/s**5, 0, True)
    >>> laplace_transform(t**a, t, s)
    (gamma(a + 1)/(s*s**a), 0, re(a) > -1)
    >>> laplace_transform(DiracDelta(t)-a*exp(-a*t), t, s, simplify=True)
    (s/(a + s), -re(a), True)

    There are also helper functions that make it easy to solve differential
    equations by Laplace transform. For example, to solve

    .. math :: m x''(t) + d x'(t) + k x(t) = 0

    with initial value `0` and initial derivative `v`:

    >>> from sympy import Function, laplace_correspondence, diff, solve
    >>> from sympy import laplace_initial_conds, inverse_laplace_transform
    >>> from sympy.abc import d, k, m, v
    >>> x = Function('x')
    # 定义一个函数 'X'，表示未知函数X(s)
    >>> X = Function('X')
    # 定义一个微分方程，表示弹簧-质量-阻尼系统的运动方程
    >>> f = m*diff(x(t), t, 2) + d*diff(x(t), t) + k*x(t)
    # 对该微分方程进行拉普拉斯变换，将其转换到复频域s
    >>> F = laplace_transform(f, t, s, noconds=True)
    # 根据变量映射关系，将变换后的表达式转换为X(s)的形式
    >>> F = laplace_correspondence(F, {x: X})
    # 应用初始条件，设置X(0) = 0和X'(0) = v
    >>> F = laplace_initial_conds(F, t, {x: [0, v]})
    # 输出处理后的X(s)表达式
    >>> F
    d*s*X(s) + k*X(s) + m*(s**2*X(s) - v)
    # 解出X(s)，表示系统的拉普拉斯域响应函数
    >>> Xs = solve(F, X(s))[0]
    # 输出解析后的X(s)表达式
    >>> Xs
    m*v/(d*s + k + m*s**2)
    # 对X(s)进行拉普拉斯逆变换，将响应函数转换回时间域
    >>> inverse_laplace_transform(Xs, s, t)
    2*v*exp(-d*t/(2*m))*sin(t*sqrt((-d**2 + 4*k*m)/m**2)/2)*Heaviside(t)/sqrt((-d**2 + 4*k*m)/m**2)

    References
    ==========

    .. [1] Erdelyi, A. (ed.), Tables of Integral Transforms, Volume 1,
           Bateman Manuscript Prooject, McGraw-Hill (1954), available:
           https://resolver.caltech.edu/CaltechAUTHORS:20140123-101456353

    See Also
    ========

    inverse_laplace_transform, mellin_transform, fourier_transform
    hankel_transform, inverse_hankel_transform

    """

    # 从提示中获取 'noconds' 和 'simplify' 的值，默认为 False
    _noconds = hints.get('noconds', False)
    _simplify = hints.get('simplify', False)

    # 如果 f 是矩阵类型并且具有 'applyfunc' 属性
    if isinstance(f, MatrixBase) and hasattr(f, 'applyfunc'):
        # 如果没有设置 'noconds'，则设置条件为真
        conds = not hints.get('noconds', False)
        # 如果满足条件且使用了传统矩阵，则给出警告信息
        if conds and legacy_matrix:
            adt = 'deprecated-laplace-transform-matrix'
            sympy_deprecation_warning(
# 定义 laplace_transform 函数，用于计算拉普拉斯变换
def laplace_transform(f, t, s, *, hints=None):
    # 检查是否存在不推荐使用的参数设置，若存在则发出警告
    if not _noconds:
        # 创建 LaplaceTransform 对象并执行变换
        LT, p, c = LaplaceTransform(f, t, s).doit(noconds=False,
                                                  simplify=_simplify)
        # 返回变换结果及相关参数
        return LT, p, c
    else:
        # 仅返回变换结果
        return LT


# 带调试包装的 _inverse_laplace_transform_integration 函数
@DEBUG_WRAP
def _inverse_laplace_transform_integration(F, s, t_, plane, *, simplify):
    """ The backend function for inverse Laplace transforms. """
    from sympy.integrals.meijerint import meijerint_inversion, _get_coeff_exp
    from sympy.integrals.transforms import inverse_mellin_transform

    # 定义虚拟变量 t，确保其为实数
    t = Dummy('t', real=True)

    # 定义函数 pw_simp，用于简化从 hyperexpand 得到的分段函数表达式
    def pw_simp(*args):
        """ Simplify a piecewise expression from hyperexpand. """
        if len(args) != 3:
            return Piecewise(*args)
        arg = args[2].args[0].argument
        coeff, exponent = _get_coeff_exp(arg, t)
        e1 = args[0].args[0]
        e2 = args[1].args[0]
        # 返回简化后的表达式
        return (
            Heaviside(1/Abs(coeff) - t**exponent)*e1 +
            Heaviside(t**exponent - 1/Abs(coeff))*e2)

    # 如果 F 是有理函数，则将其展开
    if F.is_rational_function(s):
        F = F.apart(s)

    # 如果 F 是加法表达式，则递归处理每个加法项
    if F.is_Add:
        f = Add(
            *[_inverse_laplace_transform_integration(X, s, t, plane, simplify)
              for X in F.args])
        # 返回简化后的结果及标志 True
        return _simplify(f.subs(t, t_), simplify), True

    try:
        # 尝试使用逆 Mellin 变换
        f, cond = inverse_mellin_transform(F, s, exp(-t), (None, S.Infinity),
                                           needeval=True, noconds=False)
    except IntegralTransformError:
        # 若出现积分变换错误，则置 f 为 None
        f = None

    # 如果 f 为 None，则尝试使用 meijerint_inversion 进行逆变换
    if f is None:
        f = meijerint_inversion(F, s, t)
        if f is None:
            return None
        # 如果 f 是 Piecewise 函数，则取第一个分支
        if f.is_Piecewise:
            f, cond = f.args[0]
            # 若 f 包含积分则返回 None
            if f.has(Integral):
                return None
        else:
            cond = S.true
        # 替换 f 中的 Piecewise 表达式为简化后的表达式
        f = f.replace(Piecewise, pw_simp)
    # 如果函数 f 是分段函数（Piecewise 类型）
    if f.is_Piecewise:
        # 如果函数是分段函数，下面的许多函数调用将无法处理，因为它包含一个布尔值作为参数
        # 因此直接返回 f.subs(t, t_) 和 cond
        return f.subs(t, t_), cond

    # 创建一个虚拟符号变量 u
    u = Dummy('u')

    # 定义简化 Heaviside 函数的内部函数 simp_heaviside
    def simp_heaviside(arg, H0=S.Half):
        # 将 arg 中的 exp(-t) 替换为 u
        a = arg.subs(exp(-t), u)
        # 如果 a 中仍然包含 t，则直接返回 Heaviside(arg, H0)
        if a.has(t):
            return Heaviside(arg, H0)
        # 导入不等式求解器模块
        from sympy.solvers.inequalities import _solve_inequality
        # 解决不等式 a > 0，并将结果赋给 rel
        rel = _solve_inequality(a > 0, u)
        # 如果 rel.lts == u，则执行以下操作
        if rel.lts == u:
            # 计算 k = log(rel.gts)，并返回 Heaviside(t + k, H0)
            k = log(rel.gts)
            return Heaviside(t + k, H0)
        else:
            # 计算 k = log(rel.lts)，并返回 Heaviside(-(t + k), H0)
            k = log(rel.lts)
            return Heaviside(-(t + k), H0)

    # 使用 simp_heaviside 函数替换 f 中的 Heaviside 函数
    f = f.replace(Heaviside, simp_heaviside)

    # 定义简化 exp 函数的内部函数 simp_exp
    def simp_exp(arg):
        # 对 exp(arg) 执行复杂数扩展
        return expand_complex(exp(arg))

    # 使用 simp_exp 函数替换 f 中的 exp 函数
    f = f.replace(exp, simp_exp)

    # 返回简化后的 f.subs(t, t_) 和 cond
    return _simplify(f.subs(t, t_), simplify), cond
@DEBUG_WRAP
# 装饰器，用于在调试模式下包装函数，添加调试信息
def _complete_the_square_in_denom(f, s):
    # 导入 sympy 库中的 fraction 函数，用于分解 f 为分子和分母
    from sympy.simplify.radsimp import fraction
    # 将 f 分解为分子 n 和分母 d
    [n, d] = fraction(f)
    # 检查 d 是否是关于 s 的多项式
    if d.is_polynomial(s):
        # 如果是，则将 d 转换为关于 s 的多项式的系数列表 cf
        cf = d.as_poly(s).all_coeffs()
        # 如果系数列表长度为 3，则执行完全平方化的操作
        if len(cf) == 3:
            a, b, c = cf
            # 完成分母的完全平方化操作，并返回结果
            d = a*((s+b/(2*a))**2+c/a-(b/(2*a))**2)
    return n/d


@cacheit
# 装饰器，用于缓存函数的结果，提高函数执行效率
def _inverse_laplace_build_rules():
    """
    This is an internal helper function that returns the table of inverse
    Laplace transform rules in terms of the time variable `t` and the
    frequency variable `s`.  It is used by `_inverse_laplace_apply_rules`.
    """
    # 创建符号变量 s 和 t
    s = Dummy('s')
    t = Dummy('t')
    # 创建 Wild 对象用于匹配模式
    a = Wild('a', exclude=[s])
    b = Wild('b', exclude=[s])
    c = Wild('c', exclude=[s])

    # 在调试模式下输出信息
    _debug('_inverse_laplace_build_rules is building rules')

    def _frac(f, s):
        try:
            return f.factor(s)
        except PolynomialError:
            return f

    def same(f): return f
    # 定义逆拉普拉斯变换的规则列表，按照需要的预处理函数进行排序
    _ILT_rules = [
        (a/s, a, S.true, same, 1),
        (
            b*(s+a)**(-c), t**(c-1)*exp(-a*t)/gamma(c),
            S.true, same, 1),
        (1/(s**2+a**2)**2, (sin(a*t) - a*t*cos(a*t))/(2*a**3),
         S.true, same, 1),
        # 下面两个规则必须按照指定的顺序出现。对于第二个规则，条件是 a != 0，
        # 或者在 a == 0 时，在变换后取极限。如果 a == 0，最好有其自己的规则。
        (1/(s**b), t**(b - 1)/gamma(b), S.true, same, 1),
        (1/(s*(s+a)**b), lowergamma(b, a*t)/(a**b*gamma(b)),
         S.true, same, 1)
    ]
    return _ILT_rules, s, t


@DEBUG_WRAP
# 装饰器，用于在调试模式下包装函数，添加调试信息
def _inverse_laplace_apply_simple_rules(f, s, t):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if f == 1:
        # 如果 f 为 1，则输出相应的规则信息并返回 DiracDelta(t)
        _debug('     rule: 1 o---o DiracDelta()')
        return DiracDelta(t), S.true

    # 调用 _inverse_laplace_build_rules 函数获取逆拉普拉斯变换规则列表
    _ILT_rules, s_, t_ = _inverse_laplace_build_rules()
    _prep = ''
    # 对 f 进行 s 的替换
    fsubs = f.subs({s: s_})

    for s_dom, t_dom, check, prep, fac in _ILT_rules:
        if _prep != (prep, fac):
            # 如果当前的预处理函数和因子与之前的不同，则对 fsubs 进行预处理
            _F = prep(fsubs*fac)
            _prep = (prep, fac)
        ma = _F.match(s_dom)
        if ma:
            c = check
            if c is not S.true:
                args = [x.xreplace(ma) for x in c[0]]
                c = c[1](*args)
            if c == S.true:
                # 如果条件满足，则返回逆拉普拉斯变换结果和 S.true
                return Heaviside(t)*t_dom.xreplace(ma).subs({t_: t}), S.true

    return None


@DEBUG_WRAP
# 装饰器，用于在调试模式下包装函数，添加调试信息
def _inverse_laplace_diff(f, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    # 创建 Wild 对象用于匹配模式
    a = Wild('a', exclude=[s])
    n = Wild('n', exclude=[s])
    g = Wild('g')
    # 尝试匹配 f 是否符合 a*Derivative(g, (s, n)) 的模式
    ma = f.match(a*Derivative(g, (s, n)))
    if ma and ma[n].is_integer:
        # 如果匹配成功且 n 是整数，则输出相应的规则信息并返回结果
        _debug('     rule: t**n*f(t) o---o (-1)**n*diff(F(s), s, n)')
        r, c = _inverse_laplace_transform(
            ma[g], s, t, plane, simplify=False, dorational=False)
        return (-t)**ma[n]*r, c
    return None
@DEBUG_WRAP
def _inverse_laplace_irrational(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    
    # 定义通配符，用于匹配表达式中的各种形式
    a = Wild('a', exclude=[s])
    b = Wild('b', exclude=[s])
    m = Wild('m', exclude=[s])
    n = Wild('n', exclude=[s])
    
    # 初始化结果和条件
    result = None
    condition = S.true
    
    # 将函数转换为有序因子
    fa = fn.as_ordered_factors()
    
    # 匹配每个因子中的(a*s**m + b)**n形式
    ma = [x.match((a*s**m + b)**n) for x in fa]
    
    # 如果有不匹配的项，则返回None
    if None in ma:
        return None
    
    # 初始化常数、零点、极点和其它项的列表
    constants = S.One
    zeros = []
    poles = []
    rest = []
    
    # 根据匹配的结果将项分类
    for term in ma:
        if term[a] == 0:
            constants = constants * term
        elif term[n].is_positive:
            zeros.append(term)
        elif term[n].is_negative:
            poles.append(term)
        else:
            rest.append(term)
    
    # 下面的代码假设极点按特定方式排序：
    # 按照指定的键排序poles列表，排序依据是元素的第n个和第b个字段的值以及第b个字段是否为0
    poles = sorted(poles, key=lambda x: (x[n], x[b] != 0, x[b]))
    # 按照指定的键排序zeros列表，排序依据同上
    zeros = sorted(zeros, key=lambda x: (x[n], x[b] != 0, x[b]))

    # 如果rest列表的长度不为0，则返回None
    if len(rest) != 0:
        return None

    # 如果poles列表长度为1且zeros列表长度为0，则执行以下条件语句
    if len(poles) == 1 and len(zeros) == 0:
        # 如果poles列表中第一个元素的第n个字段为-1且第m个字段为S.Half，则执行以下条件语句
        if poles[0][n] == -1 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0) == 1/a0 * 1/(sqrt(s)+b0/a0) 的特定情况处理
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]*constants
            # 如果a_为正数，则计算并存储结果
            if a_.is_positive:
                result = (
                    k_/sqrt(pi)/sqrt(t) -
                    k_*a_*exp(a_**2*t)*erfc(a_*sqrt(t)))
                # 调试信息输出
                _debug('     rule 5.3.4')
        # 如果poles列表中第一个元素的第n个字段为-2且第m个字段为S.Half，则执行以下条件语句
        elif poles[0][n] == -2 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**2 == 1/a0**2 * 1/(sqrt(s)+b0/a0)**2 的特定情况处理
            a_sq = poles[0][b]/poles[0][a]
            a_ = a_sq**2
            k_ = 1/poles[0][a]**2*constants
            # 如果a_sq为正数，则计算并存储结果
            if a_sq.is_positive:
                result = (
                    k_*(1 - 2/sqrt(pi)*sqrt(a_)*sqrt(t) +
                        (1-2*a_*t)*exp(a_*t)*(erf(sqrt(a_)*sqrt(t))-1)))
                # 调试信息输出
                _debug('     rule 5.3.10')
        # 如果poles列表中第一个元素的第n个字段为-3且第m个字段为S.Half，则执行以下条件语句
        elif poles[0][n] == -3 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**3 == 1/a0**3 * 1/(sqrt(s)+b0/a0)**3 的特定情况处理
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**3*constants
            # 如果a_为正数，则计算并存储结果
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*(a_**2*t+1)*sqrt(t) -
                        a_*t*exp(a_**2*t)*(2*a_**2*t+3)*erfc(a_*sqrt(t))))
                # 调试信息输出
                _debug('     rule 5.3.13')
        # 如果poles列表中第一个元素的第n个字段为-4且第m个字段为S.Half，则执行以下条件语句
        elif poles[0][n] == -4 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**4 == 1/a0**4 * 1/(sqrt(s)+b0/a0)**4 的特定情况处理
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**4*constants/3
            # 如果a_为正数，则计算并存储结果
            if a_.is_positive:
                result = (
                    k_*(t*(4*a_**4*t**2+12*a_**2*t+3)*exp(a_**2*t) *
                        erfc(a_*sqrt(t)) -
                        2/sqrt(pi)*a_**3*t**(S(5)/2)*(2*a_**2*t+5)))
                # 调试信息输出
                _debug('     rule 5.3.16')
        # 如果poles列表中第一个元素的第n个字段为-S.Half且第m个字段为2，则执行以下条件语句
        elif poles[0][n] == -S.Half and poles[0][m] == 2:
            # 1/sqrt(a0*s**2+b0) == 1/sqrt(a0) * 1/sqrt(s**2+b0/a0) 的特定情况处理
            a_ = sqrt(poles[0][b]/poles[0][a])
            k_ = 1/sqrt(poles[0][a])*constants
            # 计算并存储结果
            result = (k_*(besselj(0, a_*t)))
            # 调试信息输出
            _debug('     rule 5.3.35/44')
    elif len(poles) == 1 and len(zeros) == 1:
        # 检查极点和零点的个数是否分别为1
        if (
                poles[0][n] == -3 and poles[0][m] == S.Half and
                zeros[0][n] == S.Half and zeros[0][b] == 0):
            # 检查特定的极点和零点条件
            # sqrt(az*s)/(ap*sqrt(s+bp)**3)
            # == sqrt(az)/ap * sqrt(s)/(sqrt(s+bp)**3)
            a_ = poles[0][b]
            k_ = sqrt(zeros[0][a])/poles[0][a]*constants
            result = (
                k_*(2*a_**4*t**2+5*a_**2*t+1)*exp(a_**2*t) *
                erfc(a_*sqrt(t)) - 2/sqrt(pi)*a_*(a_**2*t+2)*sqrt(t))
            _debug('     rule 5.3.14')
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                zeros[0][n] == S.Half and zeros[0][m] == 1):
            # 检查特定的极点和零点条件
            # sqrt(az*s+bz)/(ap*s+bp)
            # == sqrt(az)/ap * (sqrt(s+bz/az)/(s+bp/ap))
            a_ = zeros[0][b]/zeros[0][a]
            b_ = poles[0][b]/poles[0][a]
            k_ = sqrt(zeros[0][a])/poles[0][a]*constants
            result = (
                k_*(exp(-a_*t)/sqrt(t)/sqrt(pi)+sqrt(a_-b_) *
                    exp(-b_*t)*erf(sqrt(a_-b_)*sqrt(t))))
            _debug('     rule 5.3.22')

    elif len(poles) == 3 and len(zeros) == 0:
        # 检查极点和零点的个数是否分别为3和0
        if (
                poles[0][n] == -1 and poles[0][b] == 0 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[2][n] == -S.Half and poles[2][m] == 1):
            # 检查特定的极点条件
            # 1/((a0*s)*(a1*s+b1)*sqrt(a2*s))
            # == 1/(a0*a1*sqrt(a2)) * 1/((s)*(s+b1/a1)*sqrt(s))
            a_ = -poles[1][b]/poles[1][a]
            k_ = 1/poles[0][a]/poles[1][a]/sqrt(poles[2][a])*constants
            if a_.is_positive:
                result = k_ * (
                    a_**(-S(3)/2) * exp(a_*t) * erf(sqrt(a_)*sqrt(t)) -
                    2/a_/sqrt(pi)*sqrt(t))
                _debug('     rule 5.3.2')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == S.Half and
                poles[2][n] == -S.Half and poles[2][m] == 1 and
                poles[2][b] == 0):
            # 检查特定的极点条件
            # 1/((a0*s+b0)*(a1*sqrt(s)+b1)*(sqrt(a2)*sqrt(s)))
            # == 1/(a0*a1*sqrt(a2)) * 1/((s+b0/a0)*(sqrt(s)+b1/a1)*sqrt(s))
            a_sq = poles[1][b]/poles[1][a]
            a_ = a_sq**2
            b_ = -poles[0][b]/poles[0][a]
            k_ = (
                1/poles[0][a]/poles[1][a]/sqrt(poles[2][a]) /
                (sqrt(b_)*(a_-b_)))
            if a_sq.is_positive and b_.is_positive:
                result = k_ * (
                    sqrt(b_)*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) +
                    sqrt(a_)*exp(b_*t)*erf(sqrt(b_)*sqrt(t)) -
                    sqrt(b_)*exp(b_*t))
                _debug('     rule 5.3.9')

    if result is None:
        return None
    else:
        # 若有结果则返回带有 Heaviside 函数的结果和条件
        return Heaviside(t)*result, condition
# 定义了一个装饰器函数，用于调试包装
@DEBUG_WRAP
# 对应于 InverseLaplaceTransform 类的辅助函数，用于提前应用不同的逆拉普拉斯变换规则
def _inverse_laplace_early_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    # 定义提前应用的逆拉普拉斯变换规则列表
    prog_rules = [_inverse_laplace_irrational]

    # 遍历规则列表，应用规则并返回结果（如果有结果）
    for p_rule in prog_rules:
        if (r := p_rule(F, s, t, plane)) is not None:
            return r
    # 如果没有匹配的规则应用，则返回 None
    return None


# 定义了一个装饰器函数，用于调试包装
@DEBUG_WRAP
# 对应于 InverseLaplaceTransform 类的辅助函数，用于应用逆拉普拉斯变换规则
def _inverse_laplace_apply_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    # 定义应用的逆拉普拉斯变换规则列表
    prog_rules = [_inverse_laplace_time_shift, _inverse_laplace_freq_shift,
                  _inverse_laplace_time_diff, _inverse_laplace_diff,
                  _inverse_laplace_irrational]

    # 遍历规则列表，应用规则并返回结果（如果有结果）
    for p_rule in prog_rules:
        if (r := p_rule(F, s, t, plane)) is not None:
            return r
    # 如果没有匹配的规则应用，则返回 None
    return None


# 定义了一个装饰器函数，用于调试包装
@DEBUG_WRAP
# 对应于 InverseLaplaceTransform 类的辅助函数，用于展开表达式并应用逆拉普拉斯变换
def _inverse_laplace_expand(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    # 如果输入的函数是加法表达式，则返回 None
    if fn.is_Add:
        return None
    
    # 对表达式进行展开，深度为 False
    r = expand(fn, deep=False)
    # 如果展开后仍然是加法表达式，则调用逆拉普拉斯变换函数处理
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    
    # 对乘法进行展开
    r = expand_mul(fn)
    # 如果展开后仍然是加法表达式，则调用逆拉普拉斯变换函数处理
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    
    # 再次尝试展开整个表达式
    r = expand(fn)
    # 如果展开后仍然是加法表达式，则调用逆拉普拉斯变换函数处理
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    
    # 如果是有理函数，则将其分解
    if fn.is_rational_function(s):
        r = fn.apart(s).doit()
    # 如果分解后是加法表达式，则调用逆拉普拉斯变换函数处理
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    
    # 如果以上条件都不满足，则返回 None
    return None


# 定义了一个装饰器函数，用于调试包装
@DEBUG_WRAP
# 对应于 InverseLaplaceTransform 类的辅助函数，用于处理有理函数的逆拉普拉斯变换
def _inverse_laplace_rational(fn, s, t, plane, *, simplify):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    # 定义符号 x_
    x_ = symbols('x_')
    # 将函数 fn 关于 s 进行部分分解
    f = fn.apart(s)
    # 提取部分分解后的项
    terms = Add.make_args(f)
    # 初始化一个空列表，用于存储条件
    terms_t = []
    # 初始化条件为真
    conditions = [S.true]
    # 对每个符号表达式中的项进行处理
    for term in terms:
        # 将符号表达式化为分子分母形式，分别获取分子和分母
        [n, d] = term.as_numer_denom()
        # 获取分母在变量s中的多项式系数
        dc = d.as_poly(s).all_coeffs()
        # 获取分母的主导系数
        dc_lead = dc[0]
        # 对分母的所有系数进行归一化处理
        dc = [x/dc_lead for x in dc]
        # 获取分子在变量s中的多项式系数
        nc = [x/dc_lead for x in n.as_poly(s).all_coeffs()]
        
        # 根据分母系数的长度进行不同情况的处理
        if len(dc) == 1:
            # 若分母只有一个系数，则生成对应的DiracDelta函数
            r = nc[0]*DiracDelta(t)
            terms_t.append(r)
        elif len(dc) == 2:
            # 若分母有两个系数，则生成指数衰减项
            r = nc[0]*exp(-dc[1]*t)
            terms_t.append(Heaviside(t)*r)
        elif len(dc) == 3:
            # 若分母有三个系数，则进行更复杂的处理
            a = dc[1]/2
            b = (dc[2]-a**2).factor()
            if len(nc) == 1:
                nc = [S.Zero] + nc
            l, m = tuple(nc)
            if b == 0:
                # 若参数b为零，则生成特定形式的指数衰减项
                r = (m*t+l*(1-a*t))*exp(-a*t)
            else:
                # 计算根据参数b生成的超越函数的组合
                hyp = False
                if b.is_negative:
                    b = -b
                    hyp = True
                b2 = list(roots(x_**2-b, x_).keys())[0]
                bs = sqrt(b).simplify()
                if hyp:
                    # 如果b为负，生成双曲余弦和正弦函数的组合
                    r = (
                        l*exp(-a*t)*cosh(b2*t) + (m-a*l) /
                        bs*exp(-a*t)*sinh(bs*t))
                else:
                    # 如果b为正，生成余弦和正弦函数的组合
                    r = l*exp(-a*t)*cos(b2*t) + (m-a*l)/bs*exp(-a*t)*sin(bs*t)
            terms_t.append(Heaviside(t)*r)
        else:
            # 若分母系数长度超过三个，调用逆拉普拉斯变换函数处理
            ft, cond = _inverse_laplace_transform(
                term, s, t, plane, simplify=simplify, dorational=False)
            terms_t.append(ft)
            conditions.append(cond)

    # 将所有处理后的项加和
    result = Add(*terms_t)
    # 若需要简化结果，则进行简化处理
    if simplify:
        result = result.simplify(doit=False)
    # 返回最终结果以及处理的条件
    return result, And(*conditions)
@DEBUG_WRAP
def _inverse_laplace_transform(fn, s_, t_, plane, *, simplify, dorational):
    """
    Front-end function of the inverse Laplace transform. It tries to apply all
    known rules recursively.  If everything else fails, it tries to integrate.
    """
    # 将函数 fn 转换为加法项列表
    terms = Add.make_args(fn)
    # 初始化空列表 terms_t 和 conditions
    terms_t = []
    conditions = []

    # 遍历每个加法项
    for term in terms:
        # 如果加法项包含指数函数 exp
        if term.has(exp):
            # 简化包含 exp() 的表达式，使得时间移位的表达式在分子中具有负指数，而不是分子和分母中具有正指数；这是一个必要的技巧
            # 例如，将 (s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1)) 转换为 (s**2 + 4*exp(-s) - 4*exp(-2*s))/(s*(s**2 + 1))
            term = term.subs(s_, -s_).together().subs(s_, -s_)
        
        # 将加法项 term 分解为系数 k 和函数 f
        k, f = term.as_independent(s_, as_Add=False)
        
        # 如果启用了有理化简选项并且 f 是关于 s_ 的有理函数
        if (
                dorational and term.is_rational_function(s_) and
                (r := _inverse_laplace_rational(
                    f, s_, t_, plane, simplify=simplify))
                is not None or
                (r := _inverse_laplace_apply_simple_rules(f, s_, t_))
                is not None or
                (r := _inverse_laplace_early_prog_rules(f, s_, t_, plane))
                is not None or
                (r := _inverse_laplace_expand(f, s_, t_, plane))
                is not None or
                (r := _inverse_laplace_apply_prog_rules(f, s_, t_, plane))
                is not None):
            pass
        # 如果 f 中包含未定义的函数 AppliedUndef
        elif any(undef.has(s_) for undef in f.atoms(AppliedUndef)):
            # 如果 f 中包含未定义函数 f(t)，则积分可能不会有用，因此跳过积分，返回未评估的 LaplaceTransform
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        # 否则尝试对 f 进行积分
        elif (
                r := _inverse_laplace_transform_integration(
                    f, s_, t_, plane, simplify=simplify)) is not None:
            pass
        else:
            # 若以上规则都不适用，则返回未评估的 LaplaceTransform
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        
        # 将结果 r 拆分为积分结果 ri_ 和条件 ci_
        (ri_, ci_) = r
        # 将 k*ri_ 加入到 terms_t 列表中
        terms_t.append(k*ri_)
        # 将条件 ci_ 加入到 conditions 列表中
        conditions.append(ci_)

    # 将 terms_t 中的所有项相加得到结果 result
    result = Add(*terms_t)
    # 如果启用了简化选项，则简化结果 result
    if simplify:
        result = result.simplify(doit=False)
    # 将 conditions 中的所有条件用 And 连接得到最终的条件 condition
    condition = And(*conditions)

    # 返回结果 result 和条件 condition
    return result, condition


class InverseLaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Laplace transforms, see the
    :func:`inverse_laplace_transform` docstring.
    """

    # 类属性 _name 设置为 'Inverse Laplace'
    _name = 'Inverse Laplace'
    # 类属性 _none_sentinel 为一个虚拟变量 'None'
    _none_sentinel = Dummy('None')
    # 类属性 _c 为一个虚拟变量 'c'
    _c = Dummy('c')

    def __new__(cls, F, s, x, plane, **opts):
        # 如果平面参数为 None，则设置为 _none_sentinel
        if plane is None:
            plane = InverseLaplaceTransform._none_sentinel
        # 调用父类 IntegralTransform 的构造方法
        return IntegralTransform.__new__(cls, F, s, x, plane, **opts)

    @property
    # 获取第四个参数作为平面变量，这里假设是一个特定对象的属性
    plane = self.args[3]
    # 如果平面变量是特定的标记值，则将其设为None，否则保持原值
    if plane is InverseLaplaceTransform._none_sentinel:
        plane = None
    # 返回确定的平面变量
    return plane

def _compute_transform(self, F, s, t, **hints):
    # 调用内部函数进行逆拉普拉斯变换的计算，传入函数 F、变量 s、t，以及额外的提示信息
    return _inverse_laplace_transform_integration(
        F, s, t, self.fundamental_plane, **hints)

def _as_integral(self, F, s, t):
    # 获取类属性 _c
    c = self.__class__._c
    # 返回积分表达式，对指数形式 exp(s*t)*F 在 s 上进行积分
    return (
        Integral(exp(s*t)*F, (s, c - S.ImaginaryUnit*S.Infinity,
                              c + S.ImaginaryUnit*S.Infinity)) /
        (2*S.Pi*S.ImaginaryUnit))

def doit(self, **hints):
    """
    尝试闭合形式评估变换结果。

    Explanation
    ===========

    标准提示如下：
    - ``noconds``: 如果为True，则不返回收敛条件。默认为 `True`。
    - ``simplify``: 如果为True，则简化最终结果。默认为 `False`。
    """
    # 获取提示中的 `noconds` 和 `simplify` 值，如果未指定，则使用默认值
    _noconds = hints.get('noconds', True)
    _simplify = hints.get('simplify', False)

    # 调试输出信息，记录相关函数、函数变量和变换变量
    debugf('[ILT doit] (%s, %s, %s)', (self.function,
                                       self.function_variable,
                                       self.transform_variable))

    # 获取函数变量和变换变量
    s_ = self.function_variable
    t_ = self.transform_variable
    fn = self.function
    # 获取基础平面
    plane = self.fundamental_plane

    # 调用内部函数执行逆拉普拉斯变换计算，返回结果 r
    r = _inverse_laplace_transform(
        fn, s_, t_, plane, simplify=_simplify, dorational=True)

    # 如果指定了 `noconds`，则返回计算结果的第一个元素；否则返回全部结果
    if _noconds:
        return r[0]
    else:
        return r
# 定义函数 inverse_laplace_transform，用于计算 F(s) 的逆 Laplace 变换 f(t)
def inverse_laplace_transform(F, s, t, plane=None, **hints):
    """
    Compute the inverse Laplace transform of `F(s)`, defined as

    .. math ::
        f(t) = \frac{1}{2\pi i} \int_{c-i\infty}^{c+i\infty} e^{st}
        F(s) \mathrm{d}s,

    for `c` so large that `F(s)` has no singularites in the
    half-plane `\operatorname{Re}(s) > c-\epsilon`.

    Explanation
    ===========

    The plane can be specified by
    argument ``plane``, but will be inferred if passed as None.

    Under certain regularity conditions, this recovers `f(t)` from its
    Laplace Transform `F(s)`, for non-negative `t`, and vice
    versa.

    If the integral cannot be computed in closed form, this function returns
    an unevaluated :class:`InverseLaplaceTransform` object.

    Note that this function will always assume `t` to be real,
    regardless of the SymPy assumption on `t`.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.

    Examples
    ========

    >>> from sympy import inverse_laplace_transform, exp, Symbol
    >>> from sympy.abc import s, t
    >>> a = Symbol('a', positive=True)
    >>> inverse_laplace_transform(exp(-a*s)/s, s, t)
    Heaviside(-a + t)

    See Also
    ========

    laplace_transform
    hankel_transform, inverse_hankel_transform
    """
    # 获取 hints 字典中的 noconds 键值，默认为 True
    _noconds = hints.get('noconds', True)
    # 获取 hints 字典中的 simplify 键值，默认为 False
    _simplify = hints.get('simplify', False)

    # 如果 F 是 MatrixBase 类型并且具有 applyfunc 方法，则对 F 中的每个元素应用逆 Laplace 变换
    if isinstance(F, MatrixBase) and hasattr(F, 'applyfunc'):
        return F.applyfunc(
            lambda Fij: inverse_laplace_transform(Fij, s, t, plane, **hints))

    # 调用 InverseLaplaceTransform 类的 doit 方法计算逆 Laplace 变换结果
    r, c = InverseLaplaceTransform(F, s, t, plane).doit(
        noconds=False, simplify=_simplify)

    # 根据 _noconds 决定返回结果
    if _noconds:
        return r
    else:
        return r, c


# 定义内部函数 _fast_inverse_laplace，用于快速计算包括 RootSum 的有理函数的逆 Laplace 变换
def _fast_inverse_laplace(e, s, t):
    """Fast inverse Laplace transform of rational function including RootSum"""
    # 定义通配符 a, b, n
    a, b, n = symbols('a, b, n', cls=Wild, exclude=[s])

    # 定义内部函数 _ilt，用于递归计算 e 的逆 Laplace 变换
    def _ilt(e):
        # 如果 e 中不包含 s，则直接返回 e
        if not e.has(s):
            return e
        # 如果 e 是加法表达式，则递归处理每个项
        elif e.is_Add:
            return _ilt_add(e)
        # 如果 e 是乘法表达式，则尝试进行处理
        elif e.is_Mul:
            return _ilt_mul(e)
        # 如果 e 是幂次表达式，则尝试匹配并处理
        elif e.is_Pow:
            return _ilt_pow(e)
        # 如果 e 是 RootSum 类型，则调用 _ilt_rootsum 处理
        elif isinstance(e, RootSum):
            return _ilt_rootsum(e)
        else:
            raise NotImplementedError

    # 定义内部函数 _ilt_add，用于处理加法表达式的逆 Laplace 变换
    def _ilt_add(e):
        return e.func(*map(_ilt, e.args))

    # 定义内部函数 _ilt_mul，用于处理乘法表达式的逆 Laplace 变换
    def _ilt_mul(e):
        # 将 e 分解为系数和表达式
        coeff, expr = e.as_independent(s)
        # 如果表达式是乘法，则暂不支持，抛出 NotImplementedError
        if expr.is_Mul:
            raise NotImplementedError
        # 否则，计算表达式的逆 Laplace 变换并乘以系数
        return coeff * _ilt(expr)

    # 定义内部函数 _ilt_pow，用于处理幂次表达式的逆 Laplace 变换
    def _ilt_pow(e):
        # 尝试匹配表达式是否符合 (a*s + b)**n 的形式
        match = e.match((a*s + b)**n)
        if match is not None:
            nm, am, bm = match[n], match[a], match[b]
            # 如果指数 nm 是负整数，则根据公式计算逆 Laplace 变换
            if nm.is_Integer and nm < 0:
                return t**(-nm-1)*exp(-(bm/am)*t)/(am**-nm*gamma(-nm))
            # 如果指数 nm 等于 1，则返回指数部分的逆 Laplace 变换结果
            if nm == 1:
                return exp(-(bm/am)*t) / am
        # 如果无法处理，则抛出 NotImplementedError
        raise NotImplementedError
    # 定义一个函数 _ilt_rootsum，接受一个参数 e
    def _ilt_rootsum(e):
        # 从参数 e 中获取表达式 e.fun.expr
        expr = e.fun.expr
        # 从参数 e 中获取变量列表 e.fun.variables，并将其解包赋值给变量 variable
        [variable] = e.fun.variables
        # 返回一个 RootSum 对象，使用 e.poly 作为多项式，Lambda(variable, together(_ilt(expr))) 作为函数
        return RootSum(e.poly, Lambda(variable, together(_ilt(expr))))

    # 返回调用 _ilt(e) 的结果
    return _ilt(e)
```