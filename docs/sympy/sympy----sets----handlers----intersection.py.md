# `D:\src\scipysrc\sympy\sympy\sets\handlers\intersection.py`

```
from sympy.core.basic import _aresame
from sympy.core.function import Lambda, expand_complex
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor, ceiling
from sympy.sets.fancysets import ComplexRegion
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.multipledispatch import Dispatcher
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import (Integers, Naturals, Reals, Range,
    ImageSet, Rationals)
from sympy.sets.sets import EmptySet, UniversalSet, imageset, ProductSet
from sympy.simplify.radsimp import numer

# 创建一个多分派调度器对象，用于处理集合的交集操作
intersection_sets = Dispatcher('intersection_sets')

# 注册函数，处理 ConditionSet 与 ConditionSet 的交集，总是返回 None
@intersection_sets.register(ConditionSet, ConditionSet)
def _(a, b):
    return None

# 注册函数，处理 ConditionSet 与任意 Set 类型的交集，返回一个新的 ConditionSet
@intersection_sets.register(ConditionSet, Set)
def _(a, b):
    return ConditionSet(a.sym, a.condition, Intersection(a.base_set, b))

# 注册函数，处理 Naturals 与 Integers 的交集，返回 Naturals
@intersection_sets.register(Naturals, Integers)
def _(a, b):
    return a

# 注册函数，处理两个 Naturals 集合的交集，如果其中一个是 S.Naturals，则返回该集合
@intersection_sets.register(Naturals, Naturals)
def _(a, b):
    return a if a is S.Naturals else b

# 注册函数，处理 Interval 与 Naturals 的交集，调用 intersection_sets 函数交换参数顺序后再次尝试
@intersection_sets.register(Interval, Naturals)
def _(a, b):
    return intersection_sets(b, a)

# 注册函数，处理 ComplexRegion 与任意 Set 类型的交集
@intersection_sets.register(ComplexRegion, Set)
def _(self, other):
    if other.is_ComplexRegion:
        # 如果两个集合均为直角坐标形式，则返回两者集合的交集形成的新 ComplexRegion
        if (not self.polar) and (not other.polar):
            return ComplexRegion(Intersection(self.sets, other.sets))

        # 如果两个集合均为极坐标形式，则计算极径和极角的交集，并处理特殊情况
        elif self.polar and other.polar:
            r1, theta1 = self.a_interval, self.b_interval
            r2, theta2 = other.a_interval, other.b_interval
            new_r_interval = Intersection(r1, r2)
            new_theta_interval = Intersection(theta1, theta2)

            # 处理特殊情况：0 和 2*Pi 视为相同，需加入额外的 FiniteSet(0)
            if ((2*S.Pi in theta1 and S.Zero in theta2) or
               (2*S.Pi in theta2 and S.Zero in theta1)):
                new_theta_interval = Union(new_theta_interval,
                                           FiniteSet(0))
            # 返回新的 ComplexRegion 对象，带有极坐标形式的标志
            return ComplexRegion(new_r_interval * new_theta_interval,
                                polar=True)
    # 检查 `other` 是否是实数集合的子集
    if other.is_subset(S.Reals):
        # 创建一个空列表来存储新的区间
        new_interval = []
        # 创建一个虚拟符号 `x`，被指定为实数
        x = symbols("x", cls=Dummy, real=True)

        # 如果自身是直角坐标形式
        if not self.polar:
            # 遍历自身的点集
            for element in self.psets:
                # 如果点集中包含零点
                if S.Zero in element.args[1]:
                    # 将对应的区间添加到 `new_interval` 中
                    new_interval.append(element.args[0])
            # 将所有区间并集起来
            new_interval = Union(*new_interval)
            # 返回新区间与 `other` 的交集
            return Intersection(new_interval, other)

        # 如果自身是极坐标形式
        elif self.polar:
            # 遍历自身的点集
            for element in self.psets:
                # 如果点集中包含零点
                if S.Zero in element.args[1]:
                    # 将对应的区间添加到 `new_interval` 中
                    new_interval.append(element.args[0])
                # 如果点集中包含 Pi
                if S.Pi in element.args[1]:
                    # 将对应的图像集添加到 `new_interval` 中
                    new_interval.append(ImageSet(Lambda(x, -x), element.args[0]))
                # 如果点集中包含零点
                if S.Zero in element.args[0]:
                    # 添加一个仅包含 0 的有限集合到 `new_interval` 中
                    new_interval.append(FiniteSet(0))
            # 将所有区间并集起来
            new_interval = Union(*new_interval)
            # 返回新区间与 `other` 的交集
            return Intersection(new_interval, other)
@intersection_sets.register(Integers, Reals)
def _(a, b):
    return a



# 注册一个函数，用于计算整数和实数的交集
@intersection_sets.register(Integers, Reals)
def _(a, b):
    # 对于整数和实数的交集，返回整数集合
    return a



@intersection_sets.register(Range, Interval)
def _(a, b):
    # 检查是否存在符号参数
    if not all(i.is_number for i in a.args + b.args[:2]):
        return
    
    # 如果范围为空，则返回一个空集合
    if a.size == 0:
        return S.EmptySet
    
    # 将范围调整为与自身大小相同，并用步长1表示
    start = ceiling(max(b.inf, a.inf))
    if start not in b:
        start += 1
    end = floor(min(b.sup, a.sup))
    if end not in b:
        end -= 1
    
    # 返回范围a与调整后的范围的交集
    return intersection_sets(a, Range(start, end + 1))



@intersection_sets.register(Range, Naturals)
def _(a, b):
    # 返回范围a与从b的最小值到无穷大的区间的交集
    return intersection_sets(a, Interval(b.inf, S.Infinity))



@intersection_sets.register(Range, Range)
def _(a, b):
    # 检查是否存在符号范围参数
    if not all(all(v.is_number for v in r.args) for r in [a, b]):
        return None
    
    # 快速退出，如果没有重叠部分，则返回空集合
    if not b:
        return S.EmptySet
    if not a:
        return S.EmptySet
    if b.sup < a.inf:
        return S.EmptySet
    if b.inf > a.sup:
        return S.EmptySet
    
    # 处理具有有限起始端点的情况
    r1 = a
    if r1.start.is_infinite:
        r1 = r1.reversed
    r2 = b
    if r2.start.is_infinite:
        r2 = r2.reversed
    
    # 如果两个端点都是无限的，那么其中一个范围就是所有整数的集合（步长必须为1）。
    if r1.start.is_infinite:
        return b
    if r2.start.is_infinite:
        return a
    
    from sympy.solvers.diophantine.diophantine import diop_linear
    
    # 这个方程表示范围的值；它是一个线性方程
    eq = lambda r, i: r.start + i*r.step
    
    # 我们想知道两个方程何时可能有整数解，因此使用整数线性方程求解器
    va, vb = diop_linear(eq(r1, Dummy('a')) - eq(r2, Dummy('b')))
    
    # 检查是否没有解
    no_solution = va is None and vb is None
    if no_solution:
        return S.EmptySet
    
    # 存在解
    # -------------------
    
    # 找到相交点c
    a0 = va.as_coeff_Add()[0]
    c = eq(r1, a0)
    
    # 如果可能，找到每个范围内的第一个点，因为c可能不是那个点
    # 定义函数 _first_finite_point，用于找到第一个有限点
    def _first_finite_point(r1, c):
        # 如果 c 等于 r1 的起始点，则直接返回 c
        if c == r1.start:
            return c
        # st 是从 c 到 r1.start 需要移动的有符号步长
        st = sign(r1.start - c)*step
        # 使用 Range 计算第一个点：
        # 我们希望尽可能接近 r1.start；Range 不会为空，因为至少包含 c
        s1 = Range(c, r1.start + st, st)[-1]
        # 如果 s1 等于 r1.start，则不执行操作
        if s1 == r1.start:
            pass
        else:
            # 如果没有命中 r1.start，且 st 的符号与 r1.step 的符号不匹配，则 s1 不在 r1 中
            if sign(r1.step) != sign(st):
                s1 -= st
        # 如果 s1 不在 r1 中，则返回空
        if s1 not in r1:
            return
        return s1

    # 计算新 Range 的步长大小
    step = abs(ilcm(r1.step, r2.step))
    # 获取 r1 中第一个有限点
    s1 = _first_finite_point(r1, c)
    # 如果 s1 为空，则返回空集合
    if s1 is None:
        return S.EmptySet
    # 获取 r2 中第一个有限点
    s2 = _first_finite_point(r2, c)
    # 如果 s2 为空，则返回空集合
    if s2 is None:
        return S.EmptySet

    # 更新原始 Ranges 中对应的起始或停止点；
    # 结果至少包含一个点，因为我们知道 s1 和 s2 在 Ranges 中
    def _updated_range(r, first):
        st = sign(r.step)*step
        # 如果 r.start 是有限的，则创建一个新的 Range
        if r.start.is_finite:
            rv = Range(first, r.stop, st)
        else:
            rv = Range(r.start, first + st, st)
        return rv
    
    # 更新 r1 和 r2
    r1 = _updated_range(a, s1)
    r2 = _updated_range(b, s2)

    # 在增加方向上处理它们
    # 如果 r1 的步长为负，则反转 r1
    if sign(r1.step) < 0:
        r1 = r1.reversed
    # 如果 r2 的步长为负，则反转 r2
    if sign(r2.step) < 0:
        r2 = r2.reversed

    # 返回具有正步长的截断 Range；此时不可能为空
    # 计算截断后的起始点和停止点
    start = max(r1.start, r2.start)
    stop = min(r1.stop, r2.stop)
    return Range(start, stop, step)
# 注册交集操作函数，当其中一个参数为 Range 类型，另一个参数为 Integers 类型时生效
@intersection_sets.register(Range, Integers)
def _(a, b):
    return a

# 注册交集操作函数，当其中一个参数为 Range 类型，另一个参数为 Rationals 类型时生效
@intersection_sets.register(Range, Rationals)
def _(a, b):
    return a

# 注册交集操作函数，当其中一个参数为 ImageSet 类型，另一个参数为 Set 类型时生效
def _(self, other):
    from sympy.solvers.diophantine import diophantine

    # 仅处理简单的单变量情况
    if (len(self.lamda.variables) > 1
            or self.lamda.signature != self.lamda.variables):
        return None
    base_set = self.base_sets[0]

    # 当基础集合是整数集合 Integers 时进行处理
    if base_set is S.Integers:
        gm = None
        # 如果另一个参数是 ImageSet，且基础集合也是整数集合 Integers
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            var = other.lamda.variables[0]
            # 第二个 ImageSet 的 lambda 符号必须与第一个不同
            m = Dummy('m')
            gm = gm.subs(var, m)
        # 如果另一个参数是整数集合 Integers
        elif other is S.Integers:
            m = gm = Dummy('m')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            try:
                # 解 Diophantine 方程 f(n) - g(m) = 0，寻找解集
                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
            except (TypeError, NotImplementedError):
                # 如果方程不是多项式或者没有解析求解器，则返回 None
                return
            # 可能有三种解的情况：
            # - 空集
            # - 一个或多个参数化（无限）解
            # - 有限数量的非参数化解
            # 其中，多个参数化解不适用于这里
            if len(solns) == 0:
                return S.EmptySet
            elif any(s.free_symbols for tupl in solns for s in tupl):
                if len(solns) == 1:
                    soln, solm = solns[0]
                    (t,) = soln.free_symbols
                    # 计算表达式 fn.subs(n, soln.subs(t, n))，并展开结果
                    expr = fn.subs(n, soln.subs(t, n)).expand()
                    return imageset(Lambda(n, expr), S.Integers)
                else:
                    return
            else:
                # 返回有限集合，包含所有解的映射值
                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))
    # 检查 `other` 是否等于 `S.Reals`
    if other == S.Reals:
        # 从 sympy.solvers.solvers 导入 denoms 和 solve_linear 函数
        from sympy.solvers.solvers import denoms, solve_linear

        # 定义一个函数 `_solution_union`，返回线性方程组解的并集
        # 如果无法求解，使用 ConditionSet 表示解集
        def _solution_union(exprs, sym):
            sols = []
            # 遍历输入的表达式列表
            for i in exprs:
                # 求解线性方程 i = 0 关于符号 sym 的解
                x, xis = solve_linear(i, 0, [sym])
                if x == sym:
                    # 如果解是符号本身，则将解集添加为有限集
                    sols.append(FiniteSet(xis))
                else:
                    # 否则添加为符合条件的解集
                    sols.append(ConditionSet(sym, Eq(i, 0)))
            # 返回所有解集的并集
            return Union(*sols)

        # 获取 self.lamda.expr 和 self.lamda.variables[0] 的引用
        f = self.lamda.expr
        n = self.lamda.variables[0]

        # 创建一个虚拟符号 n_，确保它是实数
        n_ = Dummy(n.name, real=True)
        # 将 f 中的 n 替换为 n_，得到 f_
        f_ = f.subs(n, n_)

        # 将 f_ 分解为实部 re 和虚部 im
        re, im = f_.as_real_imag()
        # 展开复数的虚部
        im = expand_complex(im)

        # 将 re 和 im 中的 n_ 替换回 n
        re = re.subs(n_, n)
        im = im.subs(n_, n)
        
        # 获取 im 中的自由符号集合
        ifree = im.free_symbols
        # 创建一个 Lambda 函数 lam，表示以 n 为变量的 re
        lam = Lambda(n, re)
        
        # 如果 im 是零，允许在这种情况下重新评估 self 来使结果规范化
        if im.is_zero:
            pass
        # 如果 im 不为零，返回空集 S.EmptySet
        elif im.is_zero is False:
            return S.EmptySet
        # 如果 im 的自由符号集合不是 {n}，返回 None
        elif ifree != {n}:
            return None
        else:
            # 对 base_set 应用 _solution_union 函数，排除解使得分母为零的值
            base_set &= _solution_union(
                Mul.make_args(numer(im)), n)
        
        # 排除使得 f 分母为零的值
        base_set -= _solution_union(denoms(f), n)
        
        # 返回以 lam 为映射的 base_set 的图像集
        return imageset(lam, base_set)
    # 如果 `other` 是 Interval 类型，则执行以下代码块
    elif isinstance(other, Interval):
        # 导入求解器相关模块
        from sympy.solvers.solveset import (invert_real, invert_complex,
                                            solveset)

        # 获取表达式和变量
        f = self.lamda.expr
        n = self.lamda.variables[0]
        new_inf, new_sup = None, None  # 初始化新的下界和上界为 None
        new_lopen, new_ropen = other.left_open, other.right_open  # 获取 `other` 的左右开闭性

        # 根据函数 `f` 是否为实数函数选择不同的反函数求解器
        if f.is_real:
            inverter = invert_real
        else:
            inverter = invert_complex

        # 使用反函数求解器求解 `f` 在 `other.inf` 和 `other.sup` 处的反函数值
        g1, h1 = inverter(f, other.inf, n)
        g2, h2 = inverter(f, other.sup, n)

        # 如果 `h1` 和 `h2` 均为有限集合，则处理单值反函数的情况
        if all(isinstance(i, FiniteSet) for i in (h1, h2)):
            # 如果 `g1` 等于变量 `n`，并且 `h1` 中只有一个元素，则更新 `new_inf`
            if g1 == n:
                if len(h1) == 1:
                    new_inf = h1.args[0]
            # 如果 `g2` 等于变量 `n`，并且 `h2` 中只有一个元素，则更新 `new_sup`
            if g2 == n:
                if len(h2) == 1:
                    new_sup = h2.args[0]

            # 如果 `new_sup` 或 `new_inf` 有任意一个未确定，则返回空值
            if any(i is None for i in (new_sup, new_inf)):
                return

            # 初始化空的区间集合
            range_set = S.EmptySet

            # 如果 `new_sup` 和 `new_inf` 均为实数，则创建新的 Interval 对象
            if all(i.is_real for i in (new_sup, new_inf)):
                # 考虑函数连续性，修正可能的逆序情况
                if new_inf > new_sup:
                    new_inf, new_sup = new_sup, new_inf
                new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                # 计算新区间与 `base_set` 的交集
                range_set = base_set.intersect(new_interval)
            else:
                # 如果 `other` 是实数集的子集，则求解 `f` 的解集与 `other` 的交集
                if other.is_subset(S.Reals):
                    solutions = solveset(f, n, S.Reals)
                    # 如果 `range_set` 不是 `ImageSet` 或 `ConditionSet` 类型，则取交集
                    if not isinstance(range_set, (ImageSet, ConditionSet)):
                        range_set = solutions.intersect(other)
                    else:
                        return

            # 如果 `range_set` 为空集，则返回空集
            if range_set is S.EmptySet:
                return S.EmptySet
            # 如果 `range_set` 是有限范围且大小不为无穷，则转换为有限集合
            elif isinstance(range_set, Range) and range_set.size is not S.Infinity:
                range_set = FiniteSet(*list(range_set))

            # 如果 `range_set` 不为 None，则返回 Lambda 函数映射后的结果集合
            if range_set is not None:
                return imageset(Lambda(n, f), range_set)
            return
        else:
            return
@intersection_sets.register(ProductSet, ProductSet)
def _(a, b):
    # 如果两个 ProductSet 的参数个数不同，返回空集
    if len(b.args) != len(a.args):
        return S.EmptySet
    # 对每对子集进行交集操作，返回新的 ProductSet
    return ProductSet(*(i.intersect(j) for i, j in zip(a.sets, b.sets)))


@intersection_sets.register(Interval, Interval)
def _(a, b):
    # 处理 (-oo, oo) 的情况
    infty = S.NegativeInfinity, S.Infinity
    if a == Interval(*infty):
        l, r = a.left, a.right
        # 如果左端点或右端点是实数或无穷大，则返回 b
        if l.is_real or l in infty or r.is_real or r in infty:
            return b

    # 如果 a 和 b 不可比较，则返回 None
    if not a._is_comparable(b):
        return None

    empty = False

    # 判断是否存在交集
    if a.start <= b.end and b.start <= a.end:
        # 确定交集的起点和开闭状态
        if a.start < b.start:
            start = b.start
            left_open = b.left_open
        elif a.start > b.start:
            start = a.start
            left_open = a.left_open
        else:
            start = a.start
            # 如果 a.start 和 b.start 不相同，选择具有 Float 类型的边界值
            if not _aresame(a.start, b.start):
                if b.start.has(Float) and not a.start.has(Float):
                    start = b.start
                elif a.start.has(Float) and not b.start.has(Float):
                    start = a.start
                else:
                    start = list(ordered([a,b]))[0].start
            left_open = a.left_open or b.left_open

        # 确定交集的终点和开闭状态
        if a.end < b.end:
            end = a.end
            right_open = a.right_open
        elif a.end > b.end:
            end = b.end
            right_open = b.right_open
        else:
            end = a.end
            # 如果 a.end 和 b.end 不相同，选择具有 Float 类型的边界值
            if not _aresame(a.end, b.end):
                if b.end.has(Float) and not a.end.has(Float):
                    end = b.end
                elif a.end.has(Float) and not b.end.has(Float):
                    end = a.end
                else:
                    end = list(ordered([a,b]))[0].end
            right_open = a.right_open or b.right_open

        # 如果交集为空，则返回空集
        if end - start == 0 and (left_open or right_open):
            empty = True
    else:
        empty = True

    if empty:
        return S.EmptySet

    # 返回计算得到的 Interval 对象作为交集的结果
    return Interval(start, end, left_open, right_open)


@intersection_sets.register(EmptySet, Set)
def _(a, b):
    # 如果其中一个是 EmptySet，则交集结果为 EmptySet
    return S.EmptySet


@intersection_sets.register(UniversalSet, Set)
def _(a, b):
    # 如果其中一个是 UniversalSet，则交集结果为另一个集合 b
    return b


@intersection_sets.register(FiniteSet, FiniteSet)
def _(a, b):
    # 对两个 FiniteSet 求交集，返回新的 FiniteSet
    return FiniteSet(*(a._elements & b._elements))


@intersection_sets.register(FiniteSet, Set)
def _(a, b):
    try:
        # 对 FiniteSet 和任何 Set 求交集，返回新的 FiniteSet，包含在 b 中的元素
        return FiniteSet(*[el for el in a if el in b])
    except TypeError:
        return None  # 如果出现 TypeError 异常，则返回 None
# 注册函数，用于计算两个集合的交集，针对 Set 和 Set 类型的参数
@intersection_sets.register(Set, Set)
def _(a, b):
    return None

# 注册函数，计算 Integers 和 Rationals 类型集合的交集，返回参数 a
@intersection_sets.register(Integers, Rationals)
def _(a, b):
    return a

# 注册函数，计算 Naturals 和 Rationals 类型集合的交集，返回参数 a
@intersection_sets.register(Naturals, Rationals)
def _(a, b):
    return a

# 注册函数，计算 Rationals 和 Reals 类型集合的交集，返回参数 a
@intersection_sets.register(Rationals, Reals)
def _(a, b):
    return a

# 定义一个函数 _intlike_interval，用于处理整数样式的区间计算
def _intlike_interval(a, b):
    try:
        # 如果区间 b 的下界是负无穷大，上界是正无穷大，则返回参数 a
        if b._inf is S.NegativeInfinity and b._sup is S.Infinity:
            return a
        # 计算一个 Range 对象 s，包含 a.inf 和 b.left 的最大值，以及 b.right + 1 和 floor(b.right) 的最小值
        s = Range(max(a.inf, ceiling(b.left)), floor(b.right) + 1)
        # 返回 s 和 b 的交集，如果是开区间则排除端点
        return intersection_sets(s, b)
    except ValueError:
        # 如果出现 ValueError 异常，返回 None
        return None

# 注册函数，计算 Integers 和 Interval 类型集合的交集，调用 _intlike_interval 函数处理
@intersection_sets.register(Integers, Interval)
def _(a, b):
    return _intlike_interval(a, b)

# 注册函数，计算 Naturals 和 Interval 类型集合的交集，调用 _intlike_interval 函数处理
@intersection_sets.register(Naturals, Interval)
def _(a, b):
    return _intlike_interval(a, b)
```