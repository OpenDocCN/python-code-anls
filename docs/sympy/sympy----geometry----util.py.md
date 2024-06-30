# `D:\src\scipysrc\sympy\sympy\geometry\util.py`

```
# 导入必要的库和模块
from collections import deque
from math import sqrt as _sqrt  # 导入 sqrt 函数并重命名为 _sqrt

# 导入 SymPy 相关模块和类
from sympy import nsimplify
from .entity import GeometryEntity  # 导入几何实体类
from .exceptions import GeometryError  # 导入几何错误异常类
from .point import Point, Point2D, Point3D  # 导入点类、二维点类、三维点类
from sympy.core.containers import OrderedSet  # 导入有序集合类 OrderedSet
from sympy.core.exprtools import factor_terms  # 导入因式分解工具函数
from sympy.core.function import Function, expand_mul  # 导入函数类和展开乘法函数
from sympy.core.numbers import Float  # 导入浮点数类 Float
from sympy.core.sorting import ordered  # 导入排序函数 ordered
from sympy.core.symbol import Symbol  # 导入符号类 Symbol
from sympy.core.singleton import S  # 导入单例类 S
from sympy.polys.polytools import cancel  # 导入多项式工具中的取消函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.utilities.iterables import is_sequence  # 导入判断是否为序列的函数

# 导入 mpmath 库中的精度转换函数
from mpmath.libmp.libmpf import prec_to_dps


def find(x, equation):
    """
    Checks whether a Symbol matching ``x`` is present in ``equation``
    or not. If present, the matching symbol is returned, else a
    ValueError is raised. If ``x`` is a string the matching symbol
    will have the same name; if ``x`` is a Symbol then it will be
    returned if found.

    Examples
    ========

    >>> from sympy.geometry.util import find
    >>> from sympy import Dummy
    >>> from sympy.abc import x
    >>> find('x', x)
    x
    >>> find('x', Dummy('x'))
    _x

    The dummy symbol is returned since it has a matching name:

    >>> _.name == 'x'
    True
    >>> find(x, Dummy('x'))
    Traceback (most recent call last):
    ...
    ValueError: could not find x
    """
    # 获取等式中的自由符号集合
    free = equation.free_symbols
    # 筛选出与 x 匹配的符号集合
    xs = [i for i in free if (i.name if isinstance(x, str) else i) == x]
    # 如果没有找到匹配的符号，则抛出 ValueError 异常
    if not xs:
        raise ValueError('could not find %s' % x)
    # 如果找到多个匹配的符号，则抛出 ValueError 异常
    if len(xs) != 1:
        raise ValueError('ambiguous %s' % x)
    # 返回找到的符号
    return xs[0]


def _ordered_points(p):
    """Return the tuple of points sorted numerically according to args"""
    # 根据点的参数数值排序点的元组
    return tuple(sorted(p, key=lambda x: x.args))


def are_coplanar(*e):
    """ Returns True if the given entities are coplanar otherwise False

    Parameters
    ==========

    e: entities to be checked for being coplanar

    Returns
    =======

    Boolean

    Examples
    ========

    >>> from sympy import Point3D, Line3D
    >>> from sympy.geometry.util import are_coplanar
    >>> a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    >>> b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    >>> c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    >>> are_coplanar(a, b, c)
    False

    """
    from .line import LinearEntity3D  # 导入三维线类
    from .plane import Plane  # 导入平面类
    # XXX update tests for coverage

    e = set(e)  # 将输入的实体集合转换为集合类型
    # 如果集合中包含平面对象，则先处理平面对象
    for i in list(e):
        if isinstance(i, Plane):
            e.remove(i)
            # 检查剩余实体集合是否与该平面共面
            return all(p.is_coplanar(i) for p in e)
    # 检查列表 e 中的所有元素是否都是 Point3D 类型的实例
    if all(isinstance(i, Point3D) for i in e):
        # 如果 e 的长度小于 3，则返回 False
        if len(e) < 3:
            return False

        # 移除与前两个点共线的点
        a, b = e.pop(), e.pop()
        for i in list(e):
            # 如果点 i 与点 a, b 共线，则从列表 e 中移除点 i
            if Point3D.are_collinear(a, b, i):
                e.remove(i)

        # 如果移除共线点后列表 e 变为空，则返回 False
        if not e:
            return False
        else:
            # 定义一个平面 p，使用点 a, b 和列表 e 中的最后一个点
            p = Plane(a, b, e.pop())
            for i in e:
                # 如果点 i 不在平面 p 上，则返回 False
                if i not in p:
                    return False
            # 如果所有点都在平面 p 上，则返回 True
            return True
    else:
        # 如果列表 e 中有非 Point3D 类型的元素，则将所有 Point3D 类型的点添加到 pt3d 列表中
        pt3d = []
        for i in e:
            if isinstance(i, Point3D):
                pt3d.append(i)
            elif isinstance(i, LinearEntity3D):
                # 如果元素是 LinearEntity3D 类型，则将其所有参数添加到 pt3d 列表中
                pt3d.extend(i.args)
            elif isinstance(i, GeometryEntity):
                # 处理 GeometryEntity 类型的对象，将其中的 Point 对象转换为 Point3D 类型的点（z 坐标设为 0）
                for p in i.args:
                    if isinstance(p, Point):
                        pt3d.append(Point3D(*(p.args + (0,))))
        # 检查所有添加到 pt3d 列表中的点是否共面，并返回结果
        return are_coplanar(*pt3d)
# 定义函数 are_similar，用于判断两个几何实体是否相似
def are_similar(e1, e2):
    # 若两个实体相等，则它们是相似的
    if e1 == e2:
        return True
    # 获取 e1 的 is_similar 方法
    is_similar1 = getattr(e1, 'is_similar', None)
    # 若 e1 存在 is_similar 方法，则调用其对 e2 的比较
    if is_similar1:
        return is_similar1(e2)
    # 获取 e2 的 is_similar 方法
    is_similar2 = getattr(e2, 'is_similar', None)
    # 若 e2 存在 is_similar 方法，则调用其对 e1 的比较
    if is_similar2:
        return is_similar2(e1)
    # 获取 e1 和 e2 的类名
    n1 = e1.__class__.__name__
    n2 = e2.__class__.__name__
    # 抛出 GeometryError 异常，说明无法比较 e1 和 e2 的相似性
    raise GeometryError(
        "Cannot test similarity between %s and %s" % (n1, n2))


# 定义函数 centroid，用于计算由点、线段或多边形组成的集合的质心（重心）
def centroid(*args):
    # 导入所需的类：Segment（线段）和 Polygon（多边形）
    from .line import Segment
    from .polygon import Polygon
    # 如果传入了参数
    if args:
        # 如果所有参数都是 Point 类型的实例
        if all(isinstance(g, Point) for g in args):
            # 初始化一个 Point 对象 c，并将其设为 (0, 0)
            c = Point(0, 0)
            # 遍历所有的 Point 对象，累加它们的坐标值到 c
            for g in args:
                c += g
            # 分母设为参数的数量
            den = len(args)
        # 如果所有参数都是 Segment 类型的实例
        elif all(isinstance(g, Segment) for g in args):
            # 初始化一个 Point 对象 c，并将其设为 (0, 0)
            c = Point(0, 0)
            # 初始化长度总和 L 设为 0
            L = 0
            # 遍历所有的 Segment 对象
            for g in args:
                # 获取当前 Segment 对象的长度 l
                l = g.length
                # 将当前 Segment 的中点乘以其长度累加到 c
                c += g.midpoint * l
                # 累加长度到 L
                L += l
            # 分母设为所有 Segment 的总长度
            den = L
        # 如果所有参数都是 Polygon 类型的实例
        elif all(isinstance(g, Polygon) for g in args):
            # 初始化一个 Point 对象 c，并将其设为 (0, 0)
            c = Point(0, 0)
            # 初始化面积总和 A 设为 0
            A = 0
            # 遍历所有的 Polygon 对象
            for g in args:
                # 获取当前 Polygon 对象的面积 a
                a = g.area
                # 将当前 Polygon 的质心乘以其面积累加到 c
                c += g.centroid * a
                # 累加面积到 A
                A += a
            # 分母设为所有 Polygon 的总面积
            den = A
        # 计算重心 c 的坐标平均值
        c /= den
        # 调用重心对象 c 的 func 方法，对其参数中的每个对象进行简化处理并返回结果
        return c.func(*[i.simplify() for i in c.args])
def closest_points(*args):
    """Return the subset of points from a set of points that were
    the closest to each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy import closest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> closest_points(*_)
    {(Point2D(0, 0), Point2D(3, 0))}

    References
    ==========

    .. [1] https://www.cs.mcgill.ca/~cs251/ClosestPair/ClosestPairPS.html

    .. [2] Sweep line algorithm
        https://en.wikipedia.org/wiki/Sweep_line_algorithm

    """
    # 将输入的参数转换为点的列表
    p = [Point2D(i) for i in set(args)]
    # 如果点的数量少于2个，抛出异常
    if len(p) < 2:
        raise ValueError('At least 2 distinct points must be given.')

    # 尝试对点进行排序，如果无法排序则抛出异常
    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")

    # 如果所有坐标都是有理数，则使用 math 模块中的 hypot 函数计算距离
    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    # 初始化最近点对的列表和距离
    rv = [(0, 1)]
    best_dist = hypot(p[1].x - p[0].x, p[1].y - p[0].y)
    left = 0
    box = deque([0, 1])
    # 遍历剩余的点，找到最近的点对
    for i in range(2, len(p)):
        # 移动左边界直到满足条件
        while left < i and p[i][0] - p[left][0] > best_dist:
            box.popleft()
            left += 1

        # 在当前窗口中比较点的距离
        for j in box:
            d = hypot(p[i].x - p[j].x, p[i].y - p[j].y)
            if d < best_dist:
                rv = [(j, i)]
            elif d == best_dist:
                rv.append((j, i))
            else:
                continue
            best_dist = d
        box.append(i)

    # 返回最近点对的集合
    return {tuple([p[i] for i in pair]) for pair in rv}
    # 导入所需的模块和类
    from .line import Segment
    from .polygon import Polygon
    
    # 创建一个有序集合用于存放几何实体的点集合
    p = OrderedSet()
    
    # 遍历传入的参数列表
    for e in args:
        # 检查每个实体是否为几何实体，如果不是，则尝试转换为点对象
        if not isinstance(e, GeometryEntity):
            try:
                e = Point(e)
            except NotImplementedError:
                # 如果无法转换为点对象，则抛出异常
                raise ValueError('%s is not a GeometryEntity and cannot be made into Point' % str(e))
        
        # 根据实体类型将点对象、线段对象或多边形对象添加到集合中
        if isinstance(e, Point):
            p.add(e)
        elif isinstance(e, Segment):
            p.update(e.points)
        elif isinstance(e, Polygon):
            p.update(e.vertices)
        else:
            # 如果实体类型不支持凸包计算，则抛出异常
            raise NotImplementedError(
                'Convex hull for %s not implemented.' % type(e))

    # 确保所有点都是二维的
    if any(len(x) != 2 for x in p):
        raise ValueError('Can only compute the convex hull in two dimensions')

    # 将有序集合转换为列表
    p = list(p)
    
    # 如果集合中只有一个点，根据需要返回点或线段
    if len(p) == 1:
        return p[0] if polygon else (p[0], None)
    
    # 如果集合中只有两个点，根据需要返回线段或线段元组
    elif len(p) == 2:
        s = Segment(p[0], p[1])
        return s if polygon else (s, None)

    # 定义一个函数，用于计算三个点的方向（顺时针、逆时针或共线）
    def _orientation(p, q, r):
        '''Return positive if p-q-r are clockwise, neg if ccw, zero if
        collinear.'''
        return (q.y - p.y)*(r.x - p.x) - (q.x - p.x)*(r.y - p.y)

    # 初始化上下凸包的空列表
    U = []
    L = []
    
    # 尝试对点集合进行排序，按照参数值排序
    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        # 如果无法排序，则抛出异常
        raise ValueError("The points could not be sorted.")
    
    # 使用 Andrew's Monotone Chain 算法计算上下凸包
    for p_i in p:
        while len(U) > 1 and _orientation(U[-2], U[-1], p_i) <= 0:
            U.pop()
        while len(L) > 1 and _orientation(L[-2], L[-1], p_i) >= 0:
            L.pop()
        U.append(p_i)
        L.append(p_i)
    
    # 反转上凸包的顺序，并合并上下凸包得到整体凸包
    U.reverse()
    convexHull = tuple(L + U[1:-1])

    # 如果凸包只包含两个点，则根据需要返回线段或线段元组
    if len(convexHull) == 2:
        s = Segment(convexHull[0], convexHull[1])
        return s if polygon else (s, None)
    
    # 如果需要返回多边形，则创建多边形对象并返回
    if polygon:
        return Polygon(*convexHull)
    else:
        # 否则，返回上下凸包的元组
        U.reverse()
        return (U, L)
# 计算给定点集中距离最远的点对，返回作为集合的结果

def farthest_points(*args):
    """Return the subset of points from a set of points that were
    the furthest apart from each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy.geometry import farthest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> farthest_points(*_)
    {(Point2D(0, 0), Point2D(3, 4))}

    References
    ==========

    .. [1] https://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/

    .. [2] Rotating Callipers Technique
        https://en.wikipedia.org/wiki/Rotating_calipers

    """

    # 定义旋转卡尺算法，用于计算点集的最远点对
    def rotatingCalipers(Points):
        # 计算点集的凸包上界和下界
        U, L = convex_hull(*Points, **{"polygon": False})

        # 处理特殊情况：凸包不存在下界
        if L is None:
            if isinstance(U, Point):
                raise ValueError('At least two distinct points must be given.')
            yield U.args
        else:
            i = 0
            j = len(L) - 1
            while i < len(U) - 1 or j > 0:
                # 返回当前旋转卡尺的点对
                yield U[i], L[j]
                # 如果遍历完一个边界，则移动另一个边界
                if i == len(U) - 1:
                    j -= 1
                elif j == 0:
                    i += 1
                # 如果两个边界仍有未处理的点，比较下一个凸包边的斜率
                # 注意避免斜率计算中的除零错误
                elif (U[i+1].y - U[i].y) * (L[j].x - L[j-1].x) > \
                        (L[j].y - L[j-1].y) * (U[i+1].x - U[i].x):
                    i += 1
                else:
                    j -= 1

    # 将输入的参数转换为Point2D对象的列表
    p = [Point2D(i) for i in set(args)]

    # 如果所有点的坐标都是有理数，则使用标准库的hypot函数计算距离
    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    # 初始化结果集合和直径
    rv = []
    diam = 0
    # 遍历旋转卡尺算法返回的点对
    for pair in rotatingCalipers(args):
        h, q = _ordered_points(pair)
        # 计算两点间的距离
        d = hypot(h.x - q.x, h.y - q.y)
        # 根据距离更新结果集合
        if d > diam:
            rv = [(h, q)]
        elif d == diam:
            rv.append((h, q))
        else:
            continue
        diam = d

    # 返回距离最远的点对组成的集合
    return set(rv)


def idiff(eq, y, x, n=1):
    """Return ``dy/dx`` assuming that ``eq == 0``.

    Parameters
    ==========

    y : the dependent variable or a list of dependent variables (with y first)
    x : the variable that the derivative is being taken with respect to
    n : the order of the derivative (default is 1)

    Examples
    ========

    >>> from sympy.abc import x, y, a
    >>> from sympy.geometry.util import idiff

    >>> circ = x**2 + y**2 - 4
    >>> idiff(circ, y, x)
    -x/y

    """
    # 计算给定表达式关于变量 x 的 n 次导数，然后进行化简
    >>> idiff(circ, y, x, 2).simplify()
    (-x**2 - y**2)/y**3

    # 假设变量 a 不依赖于 x，计算表达式关于变量 y 的偏导数
    Here, ``a`` is assumed to be independent of ``x``:
    >>> idiff(x + a + y, y, x)
    -1

    # 现在明确列出了 a 在 y 之后依赖于 x 的情况，计算表达式的偏导数
    Now the x-dependence of ``a`` is made explicit by listing ``a`` after
    ``y`` in a list.
    >>> idiff(x + a + y, [y, a], x)
    -Derivative(a, x) - 1

    # 查看其它相关内容
    See Also
    ========
    sympy.core.function.Derivative: represents unevaluated derivatives
    sympy.core.function.diff: explicitly differentiates wrt symbols

    """

    # 如果 y 是序列，则将其作为依赖的变量集合；如果 y 是符号，则作为单个依赖的变量；如果 y 是函数，则忽略
    if is_sequence(y):
        dep = set(y)
        y = y[0]
    elif isinstance(y, Symbol):
        dep = {y}
    elif isinstance(y, Function):
        pass
    else:
        raise ValueError("expecting x-dependent symbol(s) or function(s) but got: %s" % y)

    # 根据表达式中的自由符号，构造一个字典 f，将非 x 的符号映射为它们在 x 下的函数形式
    f = {s: Function(s.name)(x) for s in eq.free_symbols
        if s != x and s in dep}

    # 如果 y 是符号，则计算它关于 x 的导数；否则直接使用 y 的导数
    if isinstance(y, Symbol):
        dydx = Function(y.name)(x).diff(x)
    else:
        dydx = y.diff(x)

    # 在表达式中替换 f 中定义的函数，并初始化 derivs 字典用于存储导数结果
    eq = eq.subs(f)
    derivs = {}

    # 迭代计算 n 次导数
    for i in range(n):
        # 计算表达式对 x 的导数，并根据等式的线性性质，解出 dydx
        deq = eq.diff(x)
        b = deq.xreplace({dydx: S.Zero})
        a = (deq - b).xreplace({dydx: S.One})
        yp = factor_terms(expand_mul(cancel((-b/a).subs(derivs)), deep=False))

        # 如果是最后一次迭代，则返回最终结果
        if i == n - 1:
            return yp.subs([(v, k) for k, v in f.items()])

        # 将当前导数结果添加到 derivs 字典中，并更新下一轮迭代的等式
        derivs[dydx] = yp
        eq = dydx - yp
        dydx = dydx.diff(x)
    if len(entities) <= 1:
        # 如果给定的实体数量少于等于1个，则返回空列表
        return []

    entities = list(entities)
    prec = None
    for i, e in enumerate(entities):
        if not isinstance(e, GeometryEntity):
            # 如果实体不是 GeometryEntity 类型，则转换为 Point 对象
            e = Point(e)
        # 将浮点数转换为精确的有理数
        d = {}
        for f in e.atoms(Float):
            prec = f._prec if prec is None else min(f._prec, prec)
            d.setdefault(f, nsimplify(f, rational=True))
        entities[i] = e.xreplace(d)

    if not pairwise:
        # 如果 pairwise 关键字参数为 False，则寻找所有实体的公共交点
        res = entities[0].intersection(entities[1])
        for entity in entities[2:]:
            newres = []
            for x in res:
                newres.extend(x.intersection(entity))
            res = newres
    else:
        # 如果 pairwise 关键字参数为 True，则寻找所有实体之间的两两交点
        ans = []
        for j in range(len(entities)):
            for k in range(j + 1, len(entities)):
                ans.extend(intersection(entities[j], entities[k]))
        res = list(ordered(set(ans)))

    # 将结果转换回浮点数
    # 如果给定精度 prec 不为 None，则转换精度为小数点数位数 p
    if prec is not None:
        p = prec_to_dps(prec)
        # 对 res 中的每个元素使用精度 p 进行格式化，并返回结果列表
        res = [i.n(p) for i in res]
    # 返回处理后的结果 res
    return res
```