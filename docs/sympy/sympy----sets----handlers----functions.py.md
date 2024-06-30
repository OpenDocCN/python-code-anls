# `D:\src\scipysrc\sympy\sympy\sets\handlers\functions.py`

```
# 导入 SymPy 库中的特定模块和函数
from sympy.core.singleton import S
from sympy.sets.sets import Set
from sympy.calculus.singularities import singularities
from sympy.core import Expr, Add
from sympy.core.function import Lambda, FunctionClass, diff, expand_mul
from sympy.core.numbers import Float, oo
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import true
from sympy.multipledispatch import Dispatcher
from sympy.sets import (
    imageset, Interval, FiniteSet, Union, ImageSet,
    Intersection, Range, Complement
)
from sympy.sets.sets import EmptySet, is_function_invertible_in_set
from sympy.sets.fancysets import Integers, Naturals, Reals
from sympy.functions.elementary.exponential import match_real_imag

# 定义符号变量 x 和 y
_x, _y = symbols("x y")

# 定义 FunctionUnion 为包含 FunctionClass 和 Lambda 的元组
FunctionUnion = (FunctionClass, Lambda)

# 定义 _set_function 为分派器 Dispatcher 的实例，名称为 '_set_function'
_set_function = Dispatcher('_set_function')

# 使用分派器注册函数，针对 FunctionClass 和 Set 类型的参数
@_set_function.register(FunctionClass, Set)
def _(f, x):
    return None

# 使用分派器注册函数，针对 FunctionUnion 和 FiniteSet 类型的参数
@_set_function.register(FunctionUnion, FiniteSet)
def _(f, x):
    return FiniteSet(*map(f, x))

# 使用分派器注册函数，针对 Lambda 和 Interval 类型的参数
@_set_function.register(Lambda, Interval)
def _(f, x):
    from sympy.solvers.solveset import solveset
    from sympy.series import limit
    
    # TODO: 处理具有无限多解的函数（例如 sin、tan）
    # TODO: 处理多变量函数
    
    # 获取 Lambda 函数表达式
    expr = f.expr
    
    # 检查自由符号数目和函数变量数目
    if len(expr.free_symbols) > 1 or len(f.variables) != 1:
        return
    
    var = f.variables[0]
    
    # 如果变量不是实数，且替换为实数后仍不是实数，则返回空
    if not var.is_real:
        if expr.subs(var, Dummy(real=True)).is_real is False:
            return
    
    # 如果表达式是分段函数
    if expr.is_Piecewise:
        result = S.EmptySet
        domain_set = x
        
        # 遍历分段函数的每个条件和表达式
        for (p_expr, p_cond) in expr.args:
            if p_cond is true:
                intrvl = domain_set
            else:
                intrvl = p_cond.as_set()
                intrvl = Intersection(domain_set, intrvl)
            
            # 根据条件生成映像集合
            if p_expr.is_Number:
                image = FiniteSet(p_expr)
            else:
                image = imageset(Lambda(var, p_expr), intrvl)
            
            # 合并结果和映像集合
            result = Union(result, image)
            
            # 从定义域中移除已映射的部分
            domain_set = Complement(domain_set, intrvl)
            if domain_set is S.EmptySet:
                break
        
        return result
    
    # 如果区间端点不可比较，则返回空
    if not x.start.is_comparable or not x.end.is_comparable:
        return
    
    try:
        from sympy.polys.polyutils import _nsort
        
        # 计算表达式在变量和区间内的奇点
        sing = list(singularities(expr, var, x))
        
        # 如果存在多个奇点，则对其进行排序
        if len(sing) > 1:
            sing = _nsort(sing)
    
    except NotImplementedError:
        return
    
    # 根据区间的开闭性确定起始和结束点的值
    if x.left_open:
        _start = limit(expr, var, x.start, dir="+")
    elif x.start not in sing:
        _start = f(x.start)
    
    if x.right_open:
        _end = limit(expr, var, x.end, dir="-")
    elif x.end not in sing:
        _end = f(x.end)
    # 如果 sing 列表为空，则执行以下操作
    if len(sing) == 0:
        # 解出表达式关于变量 var 的导数的解集
        soln_expr = solveset(diff(expr, var), var)
        # 如果解集不是有限集或者是空集，则返回空值
        if not (isinstance(soln_expr, FiniteSet)
                or soln_expr is S.EmptySet):
            return
        
        # 将解集转换为列表
        solns = list(soln_expr)

        # 构建包含起始点、终止点及所有解点 f(i) 的列表
        extr = [_start, _end] + [f(i) for i in solns
                                 if i.is_real and i in x]
        # 计算列表中的最小值和最大值作为起始点和终止点
        start, end = Min(*extr), Max(*extr)

        # 初始化左右开区间标志
        left_open, right_open = False, False
        
        # 根据起始点和终止点的比较确定左右开区间的设置
        if _start <= _end:
            # 如果起始点和终止点与解集中的值相等且不在解集中，则设置左右开区间
            if start == _start and start not in solns:
                left_open = x.left_open
            if end == _end and end not in solns:
                right_open = x.right_open
        else:
            # 起始点和终止点方向相反的情况下设置左右开区间
            if start == _end and start not in solns:
                left_open = x.right_open
            if end == _start and end not in solns:
                right_open = x.left_open

        # 返回由起始点、终止点、左右开区间构成的区间对象
        return Interval(start, end, left_open, right_open)
    
    else:
        # 如果 sing 列表不为空，则执行以下操作
        
        # 构建由起始点和第一个 sing 元素作为终止点构成的区间
        interval1 = imageset(f, Interval(x.start, sing[0],
                                         x.left_open, True))
        
        # 构建所有相邻 sing 元素对构成的区间并使用 Union 合并
        interval2 = Union(*[imageset(f, Interval(sing[i], sing[i + 1], True, True))
                            for i in range(0, len(sing) - 1)])
        
        # 构建由最后一个 sing 元素和终止点构成的区间
        interval3 = imageset(f, Interval(sing[-1], x.end, True, x.right_open))
        
        # 返回三个区间的并集
        return interval1 + interval2 + interval3
# 注册一个特定类型的函数，用于将函数应用到区间上的操作
@_set_function.register(FunctionClass, Interval)
def _(f, x):
    # 如果函数是指数函数，对区间的起始和结束点应用指数函数，并保留区间开闭信息
    if f == exp:
        return Interval(exp(x.start), exp(x.end), x.left_open, x.right_open)
    # 如果函数是对数函数，对区间的起始和结束点应用对数函数，并保留区间开闭信息
    elif f == log:
        return Interval(log(x.start), log(x.end), x.left_open, x.right_open)
    # 对于其他函数，返回一个映射到区间上的图像集，使用 Lambda 表达式封装函数
    return ImageSet(Lambda(_x, f(_x)), x)

# 注册一个特定类型的函数，用于将函数应用到并集上的操作
@_set_function.register(FunctionUnion, Union)
def _(f, x):
    # 返回并集中每个参数应用函数 f 后的结果
    return Union(*(imageset(f, arg) for arg in x.args))

# 注册一个特定类型的函数，用于将函数应用到交集上的操作
@_set_function.register(FunctionUnion, Intersection)
def _(f, x):
    # 如果函数在集合 x 上是可逆的，则对集合的映射求交集
    if is_function_invertible_in_set(f, x):
        return Intersection(*(imageset(f, arg) for arg in x.args))
    else:
        # 否则，返回一个映射到集合 x 上的图像集，使用 Lambda 表达式封装函数
        return ImageSet(Lambda(_x, f(_x)), x)

# 注册一个特定类型的函数，用于将函数应用到空集上的操作
@_set_function.register(FunctionUnion, EmptySet)
def _(f, x):
    # 返回空集本身
    return x

# 注册一个特定类型的函数，用于将函数应用到一般集合上的操作
@_set_function.register(FunctionUnion, Set)
def _(f, x):
    # 返回一个映射到集合 x 上的图像集，使用 Lambda 表达式封装函数
    return ImageSet(Lambda(_x, f(_x)), x)

# 注册一个特定类型的函数，用于将函数应用到范围上的操作
@_set_function.register(FunctionUnion, Range)
def _(f, self):
    # 如果范围为空集，返回空集
    if not self:
        return S.EmptySet
    # 如果表达式不是 Expr 类型，则返回空值
    if not isinstance(f.expr, Expr):
        return
    # 如果范围大小为 1，返回包含唯一元素 f(self[0]) 的有限集
    if self.size == 1:
        return FiniteSet(f(self[0]))
    # 如果函数是恒等函数，返回范围本身
    if f is S.IdentityFunction:
        return self

    # 获取范围的起始点和步长
    x = f.variables[0]
    expr = f.expr

    # 处理线性关系的函数 f
    # 如果表达式中不存在自由符号或者其导数的自由符号，返回空值
    if x not in expr.free_symbols or x in expr.diff(x).free_symbols:
        return
    # 如果范围的起始点是有限的
    if self.start.is_finite:
        F = f(self.step*x + self.start)  # 对于范围中的每个元素 i
    else:
        F = f(-self.step*x + self[-1])
    F = expand_mul(F)
    # 如果 F 不等于原始表达式，则返回一个映射到范围上的图像集
    if F != expr:
        return imageset(x, F, Range(self.size))

# 注册一个特定类型的函数，用于将函数应用到整数集上的操作
@_set_function.register(FunctionUnion, Integers)
def _(f, self):
    expr = f.expr
    # 如果表达式不是 Expr 类型，则返回空值
    if not isinstance(expr, Expr):
        return

    # 获取变量 n
    n = f.variables[0]
    # 如果表达式是绝对值函数 abs(n)，返回非负整数集 Naturals0
    if expr == abs(n):
        return S.Naturals0

    # 选择形式少有负数的形式：f(x) + c 和 f(-x) + c 覆盖相同的整数
    c = f(0)
    fx = f(n) - c
    f_x = f(-n) - c
    neg_count = lambda e: sum(_.could_extract_minus_sign()
        for _ in Add.make_args(e))
    if neg_count(f_x) < neg_count(fx):
        expr = f_x + c

    # 定义通配符和模式匹配
    a = Wild('a', exclude=[n])
    b = Wild('b', exclude=[n])
    match = expr.match(a*n + b)
    # 检查 match 是否存在且 match[a] 和 match[b] 都存在，并且它们都不包含浮点数原子
    if match and match[a] and (
            not match[a].atoms(Float) and
            not match[b].atoms(Float)):
        # canonical shift 规范化移位操作
        a, b = match[a], match[b]  # 将 match[a] 和 match[b] 赋值给变量 a 和 b
        if a in [1, -1]:
            # 如果 a 是 1 或 -1，则在 b 中删除整数加数
            nonint = []
            for bi in Add.make_args(b):
                if not bi.is_integer:
                    nonint.append(bi)
            b = Add(*nonint)  # 将非整数加数重新组合成 b
        if b.is_number and a.is_real:
            # 如果 b 是数值且 a 是实数，避免对复数使用取模运算
            br, bi = match_real_imag(b)  # 提取 b 的实部和虚部
            if br and br.is_comparable and a.is_comparable:
                br %= a  # 对实部 br 取模 a
                b = br + S.ImaginaryUnit*bi  # 更新 b 为取模后的实部加上虚部乘以虚数单位
        elif b.is_number and a.is_imaginary:
            # 如果 b 是数值且 a 是虚数，避免对复数使用取模运算
            br, bi = match_real_imag(b)  # 提取 b 的实部和虚部
            ai = a / S.ImaginaryUnit  # 计算 a 的实部
            if bi and bi.is_comparable and ai.is_comparable:
                bi %= ai  # 对虚部 bi 取模 ai
                b = br + S.ImaginaryUnit*bi  # 更新 b 为取模后的实部加上虚部乘以虚数单位
        expr = a*n + b  # 计算表达式 a*n + b

    # 如果计算得到的表达式 expr 不等于原始表达式 f.expr，则返回一个新的 ImageSet 对象
    if expr != f.expr:
        return ImageSet(Lambda(n, expr), S.Integers)
# 注册一个特定函数模式的处理器，该函数处理两个参数：一个是 FunctionUnion 类型，一个是 Naturals 类型
@_set_function.register(FunctionUnion, Naturals)
def _(f, self):
    # 从函数对象 f 中获取表达式
    expr = f.expr
    # 如果表达式不是 Expr 类型，则返回空值
    if not isinstance(expr, Expr):
        return

    # 从函数对象 f 中获取变量 x，假设 f.variables 是一个变量列表
    x = f.variables[0]
    # 如果表达式中的自由符号除了 x 以外没有其他符号
    if not expr.free_symbols - {x}:
        # 如果表达式恰好是 abs(x)，则根据 self 的类型返回对应的自然数集合
        if expr == abs(x):
            if self is S.Naturals:
                return self
            return S.Naturals0
        # 获取表达式中 x 的系数
        step = expr.coeff(x)
        # 获取表达式中 x 的常数项
        c = expr.subs(x, 0)
        # 如果常数项和系数都是整数，并且表达式符合线性形式 step*x + c
        if c.is_Integer and step.is_Integer and expr == step*x + c:
            # 如果 self 是自然数集合，则将常数项增加 step
            if self is S.Naturals:
                c += step
            # 如果步长大于 0
            if step > 0:
                # 如果步长为 1
                if step == 1:
                    # 根据常数项 c 的值返回对应的自然数集合
                    if c == 0:
                        return S.Naturals0
                    elif c == 1:
                        return S.Naturals
                # 返回一个从常数项 c 开始，步长为 step 的无限自然数范围
                return Range(c, oo, step)
            # 返回一个从常数项 c 开始，步长为 step 的负无限自然数范围
            return Range(c, -oo, step)


# 注册一个特定函数模式的处理器，该函数处理两个参数：一个是 FunctionUnion 类型，一个是 Reals 类型
@_set_function.register(FunctionUnion, Reals)
def _(f, self):
    # 从函数对象 f 中获取表达式
    expr = f.expr
    # 如果表达式不是 Expr 类型，则返回空值
    if not isinstance(expr, Expr):
        return
    # 调用 _set_function 处理函数 f 和整个实数区间 (-oo, oo)
    return _set_function(f, Interval(-oo, oo))
```