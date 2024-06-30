# `D:\src\scipysrc\sympy\sympy\diffgeom\diffgeom.py`

```
# 导入必要的模块和函数
from __future__ import annotations
from typing import Any

from functools import reduce  # 导入reduce函数，用于对序列执行累积计算
from itertools import permutations  # 导入permutations函数，用于生成可迭代对象的所有排列方式

from sympy.combinatorics import Permutation  # 导入置换类Permutation
from sympy.core import (  # 导入sympy核心模块中的多个类和函数
    Basic, Expr, Function, diff,
    Pow, Mul, Add, Lambda, S, Tuple, Dict
)
from sympy.core.cache import cacheit  # 导入cacheit函数，用于缓存函数结果

from sympy.core.symbol import Symbol, Dummy  # 导入符号类Symbol和虚拟符号类Dummy
from sympy.core.symbol import Str  # 导入字符串符号类Str
from sympy.core.sympify import _sympify  # 导入_sympify函数，用于将对象转换为Sympy表达式
from sympy.functions import factorial  # 导入阶乘函数factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix  # 导入不可变稠密矩阵类Matrix
from sympy.solvers import solve  # 导入求解函数solve

from sympy.utilities.exceptions import (  # 导入异常类和函数
    sympy_deprecation_warning,
    SymPyDeprecationWarning,
    ignore_warnings
)
from sympy.tensor.array import ImmutableDenseNDimArray  # 导入不可变稠密n维数组类ImmutableDenseNDimArray


# TODO you are a bit excessive in the use of Dummies
# TODO dummy point, literal field
# TODO too often one needs to call doit or simplify on the output, check the
# tests and find out why
# 类Manifold继承自基类Basic，表示一个数学流形
class Manifold(Basic):
    """
    A mathematical manifold.

    Explanation
    ===========

    A manifold is a topological space that locally resembles
    Euclidean space near each point [1].
    This class does not provide any means to study the topological
    characteristics of the manifold that it represents, though.

    Parameters
    ==========

    name : str
        The name of the manifold.

    dim : int
        The dimension of the manifold.

    Examples
    ========

    >>> from sympy.diffgeom import Manifold
    >>> m = Manifold('M', 2)
    >>> m
    M
    >>> m.dim
    2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Manifold
    """

    # 创建新的Manifold对象
    def __new__(cls, name, dim, **kwargs):
        # 如果name不是Str类型，则转换为Str类型
        if not isinstance(name, Str):
            name = Str(name)
        # 将dim转换为Sympy表达式
        dim = _sympify(dim)
        # 调用基类的构造方法创建对象
        obj = super().__new__(cls, name, dim)

        # Manifold对象的patches属性已过时，此处创建一个空列表，并发出警告
        obj.patches = _deprecated_list(
            """
            Manifold.patches is deprecated. The Manifold object is now
            immutable. Instead use a separate list to keep track of the
            patches.
            """, [])
        return obj

    # 获取流形的名称属性
    @property
    def name(self):
        return self.args[0]

    # 获取流形的维度属性
    @property
    def dim(self):
        return self.args[1]


# 类Patch继承自基类Basic，表示一个数学流形上的一个坐标片段
class Patch(Basic):
    """
    A patch on a manifold.

    Explanation
    ===========

    Coordinate patch, or patch in short, is a simply-connected open set around
    a point in the manifold [1]. On a manifold one can have many patches that
    do not always include the whole manifold. On these patches coordinate
    charts can be defined that permit the parameterization of any point on the
    patch in terms of a tuple of real numbers (the coordinates).

    This class does not provide any means to study the topological
    characteristics of the patch that it represents.

    Parameters
    ==========

    name : str
        The name of the patch.

    manifold : Manifold
        The manifold on which the patch is defined.

    """
    # 定义一个新的类方法 __new__，用于创建 Patch 对象
    def __new__(cls, name, manifold, **kwargs):
        # 如果 name 不是 Str 类型，则转换为 Str 类型
        if not isinstance(name, Str):
            name = Str(name)
        # 调用父类的 __new__ 方法创建对象
        obj = super().__new__(cls, name, manifold)

        # 将当前 patch 对象添加到 manifold 的 patches 列表中（已废弃）
        obj.manifold.patches.append(obj) # deprecated
        
        # 设置 coord_systems 属性为一个空列表，并输出已废弃的警告信息
        obj.coord_systems = _deprecated_list(
            """
            Patch.coord_systms is deprecated. The Patch class is now
            immutable. Instead use a separate list to keep track of coordinate
            systems.
            """, [])
        
        # 返回创建的对象
        return obj

    # 定义一个属性方法 name，返回 Patch 对象的第一个参数，即名称
    @property
    def name(self):
        return self.args[0]

    # 定义一个属性方法 manifold，返回 Patch 对象的第二个参数，即流形
    @property
    def manifold(self):
        return self.args[1]

    # 定义一个属性方法 dim，返回 Patch 对象的流形的维度
    @property
    def dim(self):
        return self.manifold.dim


这些注释解释了每个方法和代码行的作用，确保代码的每一部分都得到了详细的说明。
class CoordSystem(Basic):
    """
    A coordinate system defined on the patch.

    Explanation
    ===========

    Coordinate system is a system that uses one or more coordinates to uniquely
    determine the position of the points or other geometric elements on a
    manifold [1].

    By passing ``Symbols`` to *symbols* parameter, user can define the name and
    assumptions of coordinate symbols of the coordinate system. If not passed,
    these symbols are generated automatically and are assumed to be real valued.

    By passing *relations* parameter, user can define the transform relations of
    coordinate systems. Inverse transformation and indirect transformation can
    be found automatically. If this parameter is not passed, coordinate
    transformation cannot be done.

    Parameters
    ==========

    name : str
        The name of the coordinate system.

    patch : Patch
        The patch where the coordinate system is defined.

    symbols : list of Symbols, optional
        Defines the names and assumptions of coordinate symbols.

    relations : dict, optional
        Key is a tuple of two strings, who are the names of the systems where
        the coordinates transform from and transform to.
        Value is a tuple of the symbols before transformation and a tuple of
        the expressions after transformation.

    Examples
    ========

    We define two-dimensional Cartesian coordinate system and polar coordinate
    system.

    >>> from sympy import symbols, pi, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    ... ('Pol', 'Car2D'): [(r, theta), (r*cos(theta), r*sin(theta))]
    ... }
    >>> Car2D = CoordSystem('Car2D', p, (x, y), relation_dict)
    >>> Pol = CoordSystem('Pol', p, (r, theta), relation_dict)

    ``symbols`` property returns ``CoordinateSymbol`` instances. These symbols
    are not same with the symbols used to construct the coordinate system.

    >>> Car2D
    Car2D
    >>> Car2D.dim
    2
    >>> Car2D.symbols
    (x, y)
    >>> _[0].func
    <class 'sympy.diffgeom.diffgeom.CoordinateSymbol'>

    ``transformation()`` method returns the transformation function from
    one coordinate system to another. ``transform()`` method returns the
    transformed coordinates.

    >>> Car2D.transformation(Pol)
    Lambda((x, y), Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]]))
    >>> Car2D.transform(Pol)
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> Car2D.transform(Pol, [1, 2])
    Matrix([
    [sqrt(5)],
    [atan(2)]])

    ``jacobian()`` method returns the Jacobian matrix of coordinate
    transformation between two systems. ``jacobian_determinant()`` method
    calculates the determinant of the Jacobian matrix.

    """
    pass
    """
    创建一个新的 CoordinateSystem 类的实例。

    Parameters
    ==========
    name : str or Symbol
        系统的名称，如果不是 Str 类型，则转换为 Str 类型。
    patch : Patch
        与此坐标系相关联的 Patch 对象。
    symbols : Tuple, optional
        用于表示坐标的符号元组，默认为 None。
    relations : dict, optional
        坐标符号之间的关系字典，默认为空字典。
    **kwargs : dict
        其他关键字参数，用于获取 names 参数。

    Returns
    =======
    CoordinateSystem
        返回新创建的 CoordinateSystem 类的实例。

    Notes
    =====
    此方法用于创建具有给定名称、关联 Patch 对象以及符号元组和关系字典的坐标系实例。

    Warnings
    ========
    如果 symbols 参数为 None，则会基于 patch.dim 创建符号元组。

    Examples
    ========
    >>> Pol = CoordinateSystem('Pol', Patch, symbols=(r, theta))
    >>> Car2D = CoordinateSystem('Car2D', Patch2D)
    """

    def __new__(cls, name, patch, symbols=None, relations={}, **kwargs):
        # 检查 name 是否为 Str 类型，如果不是则转换为 Str 类型
        if not isinstance(name, Str):
            name = Str(name)

        # 规范化 symbols 参数
        if symbols is None:
            # 如果 symbols 为 None，则尝试从 kwargs 中获取 names 参数
            names = kwargs.get('names', None)
            if names is None:
                # 如果 names 也为 None，则基于 patch.dim 创建符号元组
                symbols = Tuple(
                    *[Symbol('%s_%s' % (name.name, i), real=True)
                      for i in range(patch.dim)]
                )
            else:
                # 否则发出警告，使用 sympy_deprecation_warning 函数
                sympy_deprecation_warning(
                    f"""
                    Deprecated syntax: {names} should be passed as the symbols parameter instead of as a positional argument to CoordinateSystem.
                    """
                )

        # 创建 CoordinateSystem 类的实例并返回
        return super().__new__(cls, name, patch, symbols=symbols, relations=relations, **kwargs)
# 定义一个类CoordSystem，表示坐标系对象
class CoordSystem(Function):

    # 构造函数，初始化坐标系对象
    def __new__(cls, name, patch, symbols=None, relations=None):
        # 向用户发出警告，names参数已不推荐使用，应使用symbols参数
        sympy_deprecation_warning(
            """
            The 'names' argument to CoordSystem is deprecated. Use 'symbols' instead. That
            is, replace

            CoordSystem(..., names={names})

            with

            CoordSystem(..., symbols=[{', '.join(["Symbol(" + repr(n) + ", real=True)" for n in names])}])
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )
        
        # 如果symbols未提供，默认为空元组
        if symbols is None:
            symbols = ()
        
        # 如果relations未提供，默认为空字典
        if relations is None:
            relations = {}
        
        # 如果symbols是字符串列表，则生成Symbol对象列表
        if isinstance(symbols, list):
            symbols = Tuple(*[Symbol(n, real=True) for n in symbols])

        # 如果symbols是字符串，则发出警告，推荐使用Symbol对象
        else:
            syms = []
            for s in symbols:
                if isinstance(s, Symbol):
                    syms.append(Symbol(s.name, **s._assumptions.generator))
                elif isinstance(s, str):
                    sympy_deprecation_warning(
                        f"""
                        Passing a string as the coordinate symbol name to CoordSystem is deprecated.
                        Pass a Symbol with the appropriate name and assumptions instead.

                        That is, replace {s} with Symbol({s!r}, real=True).
                        """,
                        deprecated_since_version="1.7",
                        active_deprecations_target="deprecated-diffgeom-mutable",
                    )
                    syms.append(Symbol(s, real=True))
            symbols = Tuple(*syms)

        # 规范化relations字典中的键和值
        rel_temp = {}
        for k, v in relations.items():
            s1, s2 = k
            # 如果s1不是字符串，转换为字符串
            if not isinstance(s1, Str):
                s1 = Str(s1)
            # 如果s2不是字符串，转换为字符串
            if not isinstance(s2, Str):
                s2 = Str(s2)
            key = Tuple(s1, s2)

            # 如果v是Lambda函数对象，转换为元组表示
            if isinstance(v, Lambda):
                v = (tuple(v.signature), tuple(v.expr))
            else:
                v = (tuple(v[0]), tuple(v[1]))
            rel_temp[key] = v
        
        # 将转换后的关系字典赋值给relations
        relations = Dict(rel_temp)

        # 调用父类的构造方法创建对象
        obj = super().__new__(cls, name, patch, symbols, relations)

        # 添加被弃用的属性和方法
        obj.transforms = _deprecated_dict(
            """
            CoordSystem.transforms is deprecated. The CoordSystem class is now
            immutable. Use the 'relations' keyword argument to the
            CoordSystems() constructor to specify relations.
            """, {})
        obj._names = [str(n) for n in symbols]
        obj.patch.coord_systems.append(obj)  # 被弃用的操作，向patch的coord_systems列表添加当前对象
        obj._dummies = [Dummy(str(n)) for n in symbols]  # 被弃用的操作，创建Dummy对象列表
        obj._dummy = Dummy()

        return obj

    # 返回坐标系名称
    @property
    def name(self):
        return self.args[0]

    # 返回坐标系的patch对象
    @property
    def patch(self):
        return self.args[1]

    # 返回坐标系的manifold对象
    @property
    def manifold(self):
        return self.patch.manifold

    # 返回坐标系的symbols，以CoordinateSymbol对象的元组形式返回
    @property
    def symbols(self):
        return tuple(CoordinateSymbol(self, i, **s._assumptions.generator)
                     for i, s in enumerate(self.args[2]))

    # 返回坐标系的relations字典
    @property
    def relations(self):
        return self.args[3]

    # 属性结束
    # 返回当前对象中 patch 属性的 dim 属性值
    def dim(self):
        return self.patch.dim

    ##########################################################################
    # Finding transformation relation
    ##########################################################################

    # 返回从当前坐标系到指定坐标系 sys 的坐标变换函数
    def transformation(self, sys):
        """
        Return coordinate transformation function from *self* to *sys*.

        Parameters
        ==========

        sys : CoordSystem

        Returns
        =======

        sympy.Lambda

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transformation(R2_p)
        Lambda((x, y), Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]]))

        """
        # 获取当前对象的第三个参数，这是坐标系的标识符
        signature = self.args[2]

        # 构造关键字，标识当前坐标系与目标坐标系的对
        key = Tuple(self.name, sys.name)
        # 如果当前坐标系与目标坐标系相同，则直接返回当前坐标系的符号列表构成的矩阵
        if self == sys:
            expr = Matrix(self.symbols)
        # 如果在关系字典中找到了当前坐标系到目标坐标系的直接变换关系，则返回相应的表达式矩阵
        elif key in self.relations:
            expr = Matrix(self.relations[key][1])
        # 如果在关系字典中找到了目标坐标系到当前坐标系的变换关系，则返回其逆变换
        elif key[::-1] in self.relations:
            expr = Matrix(self._inverse_transformation(sys, self))
        # 否则，通过间接变换来计算当前坐标系到目标坐标系的变换关系
        else:
            expr = Matrix(self._indirect_transformation(self, sys))
        # 返回一个 Lambda 表达式，表示从当前坐标系到目标坐标系的坐标变换
        return Lambda(signature, expr)

    @staticmethod
    # 在静态方法中解决从 sym2 到 sym1 的逆变换关系
    def _solve_inverse(sym1, sym2, exprs, sys1_name, sys2_name):
        ret = solve(
            # 构造解决的方程列表，将 sym2 与对应的表达式进行比较
            [t[0] - t[1] for t in zip(sym2, exprs)],
            # 将 sym1 转换为列表形式，以便 solve 函数进行解析
            list(sym1), dict=True)

        # 如果解得结果为空，则抛出未实现错误，表示无法找到逆关系
        if len(ret) == 0:
            temp = "Cannot solve inverse relation from {} to {}."
            raise NotImplementedError(temp.format(sys1_name, sys2_name))
        # 如果解得结果大于1个，则抛出值错误，表示解得多个逆关系
        elif len(ret) > 1:
            temp = "Obtained multiple inverse relation from {} to {}."
            raise ValueError(temp.format(sys1_name, sys2_name))

        # 返回解得的第一个逆关系结果
        return ret[0]

    @classmethod
    # 返回从 sys2 到 sys1 的逆变换关系列表
    def _inverse_transformation(cls, sys1, sys2):
        # 执行从 sys1 到 sys2 的正向变换
        forward = sys1.transform(sys2)
        # 解决从 sys2 到 sys1 的逆变换关系
        inv_results = cls._solve_inverse(sys1.symbols, sys2.symbols, forward,
                                         sys1.name, sys2.name)
        # 获取当前坐标系的符号列表作为变换的标识符
        signature = tuple(sys1.symbols)
        # 返回逆变换关系的结果列表
        return [inv_results[s] for s in signature]

    @classmethod
    @cacheit
    # 返回两个间接连接坐标系之间的变换关系
    def _indirect_transformation(cls, sys1, sys2):
        # 获取当前坐标系的关系字典
        rel = sys1.relations
        # 使用 Dijkstra 算法找到 sys1 到 sys2 的路径
        path = cls._dijkstra(sys1, sys2)

        transforms = []
        # 遍历路径中的每一对坐标系
        for s1, s2 in zip(path, path[1:]):
            # 如果在关系字典中找到了 s1 到 s2 的直接变换关系，则添加到 transforms 列表中
            if (s1, s2) in rel:
                transforms.append(rel[(s1, s2)])
            # 否则，根据 s2 到 s1 的逆变换关系计算 s1 到 s2 的变换关系
            else:
                sym2, inv_exprs = rel[(s2, s1)]
                sym1 = tuple(Dummy() for i in sym2)
                ret = cls._solve_inverse(sym2, sym1, inv_exprs, s2, s1)
                ret = tuple(ret[s] for s in sym2)
                transforms.append((sym1, ret))
        # 获取当前坐标系的符号列表作为变换的标识符
        syms = sys1.args[2]
        exprs = syms
        # 根据 transforms 列表中的每一对新符号和表达式进行替换
        for newsyms, newexprs in transforms:
            exprs = tuple(e.subs(zip(newsyms, exprs)) for e in newexprs)
        # 返回最终计算得到的变换关系的表达式列表
        return exprs
    @staticmethod
    def _dijkstra(sys1, sys2):
        # 使用 Dijkstra 算法找到两个间接连接的坐标系之间的最短路径
        # 返回值是系统名称列表，表示最短路径上经过的系统名称

        # 获取系统1的关系映射
        relations = sys1.relations
        # 初始化图的字典
        graph = {}
        # 构建无向图
        for s1, s2 in relations.keys():
            if s1 not in graph:
                graph[s1] = {s2}
            else:
                graph[s1].add(s2)
            if s2 not in graph:
                graph[s2] = {s1}
            else:
                graph[s2].add(s1)

        # 初始化路径字典，记录最小距离、路径和访问次数
        path_dict = {sys:[0, [], 0] for sys in graph}

        # 定义访问函数，更新路径字典
        def visit(sys):
            path_dict[sys][2] = 1
            for newsys in graph[sys]:
                distance = path_dict[sys][0] + 1
                if path_dict[newsys][0] >= distance or not path_dict[newsys][1]:
                    path_dict[newsys][0] = distance
                    path_dict[newsys][1] = list(path_dict[sys][1])
                    path_dict[newsys][1].append(sys)

        # 从系统1开始访问
        visit(sys1.name)

        # 使用 Dijkstra 算法更新路径字典，直到所有系统都访问过
        while True:
            min_distance = max(path_dict.values(), key=lambda x:x[0])[0]
            newsys = None
            for sys, lst in path_dict.items():
                if 0 < lst[0] <= min_distance and not lst[2]:
                    min_distance = lst[0]
                    newsys = sys
            if newsys is None:
                break
            visit(newsys)

        # 获取最终路径
        result = path_dict[sys2.name][1]
        result.append(sys2.name)

        # 如果结果只包含系统2本身，抛出异常，表示两个坐标系不连接
        if result == [sys2.name]:
            raise KeyError("Two coordinate systems are not connected.")
        return result

    def connect_to(self, to_sys, from_coords, to_exprs, inverse=True, fill_in_gaps=False):
        sympy_deprecation_warning(
            """
            The CoordSystem.connect_to() method is deprecated. Instead,
            generate a new instance of CoordSystem with the 'relations'
            keyword argument (CoordSystem classes are now immutable).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )

        # 转换输入以适应连接方法的变化
        from_coords, to_exprs = dummyfy(from_coords, to_exprs)
        # 将目标系统和其对应的转换矩阵保存到当前系统的transforms属性中
        self.transforms[to_sys] = Matrix(from_coords), Matrix(to_exprs)

        # 如果需要反向转换，则在目标系统中保存反向转换矩阵
        if inverse:
            to_sys.transforms[self] = self._inv_transf(from_coords, to_exprs)

        # 如果需要填补转换中的空隙，则执行填补方法
        if fill_in_gaps:
            self._fill_gaps_in_transformations()

    @staticmethod
    def _inv_transf(from_coords, to_exprs):
        # connect_to 方法被移除后将不再需要这个方法
        # 使用代数方法计算反向转换矩阵
        inv_from = [i.as_dummy() for i in from_coords]
        inv_to = solve(
            [t[0] - t[1] for t in zip(inv_from, to_exprs)],
            list(from_coords), dict=True)[0]
        inv_to = [inv_to[fc] for fc in from_coords]
        return Matrix(inv_from), Matrix(inv_to)
    def _fill_gaps_in_transformations():
        # 将在删除 connect_to 方法时移除
        raise NotImplementedError



    ##########################################################################
    # Coordinate transformations
    ##########################################################################

    def transform(self, sys, coordinates=None):
        """
        Return the result of coordinate transformation from *self* to *sys*.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix containing CoordinateSymbol

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transform(R2_p)
        Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]])
        >>> R2_r.transform(R2_p, [0, 1])
        Matrix([
        [   1],
        [pi/2]])

        """
        if coordinates is None:
            coordinates = self.symbols
        if self != sys:
            # 获取从当前坐标系到目标坐标系的变换
            transf = self.transformation(sys)
            # 应用变换将坐标从当前坐标系转换到目标坐标系
            coordinates = transf(*coordinates)
        else:
            # 如果当前坐标系与目标坐标系相同，则直接将坐标转换为矩阵形式
            coordinates = Matrix(coordinates)
        return coordinates



    def coord_tuple_transform_to(self, to_sys, coords):
        """Transform ``coords`` to coord system ``to_sys``."""
        sympy_deprecation_warning(
            """
            The CoordSystem.coord_tuple_transform_to() method is deprecated.
            Use the CoordSystem.transform() method instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
        )

        # 将坐标向量转换为矩阵形式
        coords = Matrix(coords)
        if self != to_sys:
            # 如果当前坐标系与目标坐标系不同，则进行坐标变换
            with ignore_warnings(SymPyDeprecationWarning):
                transf = self.transforms[to_sys]
            # 替换变换中的符号，并将坐标转换为目标坐标系下的坐标
            coords = transf[1].subs(list(zip(transf[0], coords)))
        return coords



    def jacobian(self, sys, coordinates=None):
        """
        Return the jacobian matrix of a transformation on given coordinates.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_p.jacobian(R2_r)
        Matrix([
        [cos(theta), -rho*sin(theta)],
        [sin(theta),  rho*cos(theta)]])
        >>> R2_p.jacobian(R2_r, [1, 0])
        Matrix([
        [1, 0],
        [0, 1]])

        """
        # 获取从当前坐标系到目标坐标系的坐标变换，并计算其雅可比矩阵
        result = self.transform(sys).jacobian(self.symbols)
        if coordinates is not None:
            # 如果提供了目标坐标，则用目标坐标替换雅可比矩阵中的符号
            result = result.subs(list(zip(self.symbols, coordinates)))
        return result
    jacobian_matrix = jacobian


**注释：**
- `_fill_gaps_in_transformations()`：用于占位的函数，当删除 `connect_to` 方法时将移除。
- `transform(self, sys, coordinates=None)`：从当前坐标系转换到目标坐标系的方法。根据是否提供坐标，选择性地转换坐标。
- `coord_tuple_transform_to(self, to_sys, coords)`：将坐标从当前坐标系转换到目标坐标系的方法。支持向量坐标的转换。
- `jacobian(self, sys, coordinates=None)`：计算给定坐标变换的雅可比矩阵的方法。
    def jacobian_determinant(self, sys, coordinates=None):
        """
        Return the jacobian determinant of a transformation on given
        coordinates. If coordinates are not given, coordinate symbols of *self*
        are used.

        Parameters
        ==========

        sys : CoordSystem
            Another coordinate system object representing the target system.

        coordinates : Any iterable, optional.
            Optional iterable of symbols representing the coordinates.

        Returns
        =======

        sympy.Expr
            The resulting Jacobian determinant expression.

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.jacobian_determinant(R2_p)
        1/sqrt(x**2 + y**2)
        >>> R2_r.jacobian_determinant(R2_p, [1, 0])
        1

        """
        # Calculate the Jacobian determinant using the jacobian method and return it
        return self.jacobian(sys, coordinates).det()


    ##########################################################################
    # Points
    ##########################################################################

    def point(self, coords):
        """Create a ``Point`` with coordinates given in this coord system."""
        # Create a Point object using this coordinate system and given coordinates
        return Point(self, coords)

    def point_to_coords(self, point):
        """Calculate the coordinates of a point in this coord system."""
        # Calculate and return the coordinates of a given Point in this coordinate system
        return point.coords(self)

    ##########################################################################
    # Base fields.
    ##########################################################################

    def base_scalar(self, coord_index):
        """Return ``BaseScalarField`` that takes a point and returns one of the coordinates."""
        # Return a BaseScalarField object associated with this coordinate system and given index
        return BaseScalarField(self, coord_index)
    coord_function = base_scalar

    def base_scalars(self):
        """Returns a list of all coordinate functions.
        For more details see the ``base_scalar`` method of this class."""
        # Return a list of BaseScalarField objects for all coordinates in this coordinate system
        return [self.base_scalar(i) for i in range(self.dim)]
    coord_functions = base_scalars

    def base_vector(self, coord_index):
        """Return a basis vector field.
        The basis vector field for this coordinate system. It is also an
        operator on scalar fields."""
        # Return a BaseVectorField object associated with this coordinate system and given index
        return BaseVectorField(self, coord_index)

    def base_vectors(self):
        """Returns a list of all base vectors.
        For more details see the ``base_vector`` method of this class."""
        # Return a list of BaseVectorField objects for all basis vectors in this coordinate system
        return [self.base_vector(i) for i in range(self.dim)]

    def base_oneform(self, coord_index):
        """Return a basis 1-form field.
        The basis one-form field for this coordinate system. It is also an
        operator on vector fields."""
        # Return a Differential object for the basis one-form field at the given index
        return Differential(self.coord_function(coord_index))

    def base_oneforms(self):
        """Returns a list of all base oneforms.
        For more details see the ``base_oneform`` method of this class."""
        # Return a list of Differential objects for all basis one-forms in this coordinate system
        return [self.base_oneform(i) for i in range(self.dim)]
class CoordinateSymbol(Symbol):
    """A symbol which denotes an abstract value of i-th coordinate of
    the coordinate system with given context.

    Explanation
    ===========

    Each coordinates in coordinate system are represented by unique symbol,
    such as x, y, z in Cartesian coordinate system.

    You may not construct this class directly. Instead, use `symbols` method
    of CoordSystem.

    Parameters
    ==========

    coord_sys : CoordSystem
        The coordinate system to which this symbol belongs.

    index : integer
        The index indicating which coordinate this symbol represents.

    Examples
    ========

    >>> from sympy import symbols, Lambda, Matrix, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): Lambda((x, y), Matrix([sqrt(x**2 + y**2), atan2(y, x)])),
    ... ('Pol', 'Car2D'): Lambda((r, theta), Matrix([r*cos(theta), r*sin(theta)]))
    ... }
    >>> Car2D = CoordSystem('Car2D', p, [x, y], relation_dict)
    >>> Pol = CoordSystem('Pol', p, [r, theta], relation_dict)
    >>> x, y = Car2D.symbols

    ``CoordinateSymbol`` contains its coordinate symbol and index.

    >>> x.name
    'x'
    >>> x.coord_sys == Car2D
    True
    >>> x.index
    0
    >>> x.is_real
    True

    You can transform ``CoordinateSymbol`` into other coordinate system using
    ``rewrite()`` method.

    >>> x.rewrite(Pol)
    r*cos(theta)
    >>> sqrt(x**2 + y**2).rewrite(Pol).simplify()
    r

    """
    def __new__(cls, coord_sys, index, **assumptions):
        # 获取指定坐标系中特定索引的坐标符号名称
        name = coord_sys.args[2][index].name
        # 调用父类的构造方法创建新的符号对象
        obj = super().__new__(cls, name, **assumptions)
        # 设置符号对象的坐标系和索引属性
        obj.coord_sys = coord_sys
        obj.index = index
        return obj

    def __getnewargs__(self):
        # 返回符号对象的序列化参数，即坐标系和索引
        return (self.coord_sys, self.index)

    def _hashable_content(self):
        # 返回符号对象的可哈希内容，包括坐标系、索引和假设项
        return (
            self.coord_sys, self.index
        ) + tuple(sorted(self.assumptions0.items()))

    def _eval_rewrite(self, rule, args, **hints):
        # 如果转换规则是一个坐标系对象，则使用该规则转换当前坐标系中的坐标
        if isinstance(rule, CoordSystem):
            return rule.transform(self.coord_sys)[self.index]
        # 否则调用父类方法执行重写操作
        return super()._eval_rewrite(rule, args, **hints)


class Point(Basic):
    """Point defined in a coordinate system.

    Explanation
    ===========

    Mathematically, point is defined in the manifold and does not have any coordinates
    by itself. Coordinate system is what imbues the coordinates to the point by coordinate
    chart. However, due to the difficulty of realizing such logic, you must supply
    a coordinate system and coordinates to define a Point here.

    The usage of this object after its definition is independent of the
    coordinate system that was used in order to define it, however due to
    limitations in the simplification routines you can arrive at complicated
    expressions if you use inappropriate coordinate systems.

    Parameters
    ==========

    coord_sys : CoordSystem
        The coordinate system in which this point is defined.

    """
    coords : list
        The coordinates of the point.

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy.diffgeom import Point
    >>> from sympy.diffgeom.rn import R2, R2_r, R2_p
    >>> rho, theta = R2_p.symbols

    >>> p = Point(R2_p, [rho, 3*pi/4])

    >>> p.manifold == R2
    True

    >>> p.coords()
    Matrix([
    [   rho],
    [3*pi/4]])
    >>> p.coords(R2_r)
    Matrix([
    [-sqrt(2)*rho/2],
    [ sqrt(2)*rho/2]])

    """

    # __new__ method for initializing a Point object with specified coordinate system and coordinates
    def __new__(cls, coord_sys, coords, **kwargs):
        coords = Matrix(coords)  # Convert coordinates to a SymPy Matrix
        obj = super().__new__(cls, coord_sys, coords)  # Initialize superclass (Point)
        obj._coord_sys = coord_sys  # Store the coordinate system
        obj._coords = coords  # Store the coordinates
        return obj

    @property
    def patch(self):
        # Property to access the patch (domain) associated with the coordinate system
        return self._coord_sys.patch

    @property
    def manifold(self):
        # Property to access the manifold associated with the coordinate system
        return self._coord_sys.manifold

    @property
    def dim(self):
        # Property to retrieve the dimension of the manifold
        return self.manifold.dim

    def coords(self, sys=None):
        """
        Coordinates of the point in a given coordinate system. If no coordinate
        system is provided, returns coordinates in the system in which the point
        was originally defined.
        """
        if sys is None:
            return self._coords  # Return coordinates in the original coordinate system
        else:
            return self._coord_sys.transform(sys, self._coords)  # Transform coordinates to the specified system

    @property
    def free_symbols(self):
        # Property to retrieve the free symbols present in the coordinates
        return self._coords.free_symbols
# 表示一个基本的标量场，定义在给定坐标系上的一个坐标系统的一个坐标。

class BaseScalarField(Expr):
    """Base scalar field over a manifold for a given coordinate system.

    Explanation
    ===========

    A scalar field takes a point as an argument and returns a scalar.
    A base scalar field of a coordinate system takes a point and returns one of
    the coordinates of that point in the coordinate system in question.

    To define a scalar field you need to choose the coordinate system and the
    index of the coordinate.

    The use of the scalar field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in
    the simplification routines you may arrive at more complicated
    expression if you use unappropriate coordinate systems.
    You can build complicated scalar fields by just building up SymPy
    expressions containing ``BaseScalarField`` instances.

    Parameters
    ==========

    coord_sys : CoordSystem
        表示标量场所依赖的坐标系对象。

    index : integer
        表示标量场中坐标的索引。

    Examples
    ========

    >>> from sympy import Function, pi
    >>> from sympy.diffgeom import BaseScalarField
    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> rho, _ = R2_p.symbols
    >>> point = R2_p.point([rho, 0])
    >>> fx, fy = R2_r.base_scalars()
    >>> ftheta = BaseScalarField(R2_r, 1)

    >>> fx(point)
    rho
    >>> fy(point)
    0

    >>> (fx**2+fy**2).rcall(point)
    rho**2

    >>> g = Function('g')
    >>> fg = g(ftheta-pi)
    >>> fg.rcall(point)
    g(-pi)

    """

    is_commutative = True

    def __new__(cls, coord_sys, index, **kwargs):
        # 用于创建新的 BaseScalarField 实例，传入坐标系和坐标索引。
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        # 获取标量场所依赖的坐标系对象。
        return self.args[0]

    @property
    def index(self):
        # 获取标量场中坐标的索引。
        return self.args[1]

    @property
    def patch(self):
        # 获取标量场所在的坐标系的 patch。
        return self.coord_sys.patch

    @property
    def manifold(self):
        # 获取标量场所在的流形对象。
        return self.coord_sys.manifold

    @property
    def dim(self):
        # 获取标量场所在流形的维度。
        return self.manifold.dim

    def __call__(self, *args):
        """Evaluating the field at a point or doing nothing.
        If the argument is a ``Point`` instance, the field is evaluated at that
        point. The field is returned itself if the argument is any other
        object. It is so in order to have working recursive calling mechanics
        for all fields (check the ``__call__`` method of ``Expr``).
        """
        # 对标量场进行求值，如果参数是 Point 实例，则在该点求值。
        point = args[0]
        if len(args) != 1 or not isinstance(point, Point):
            return self
        coords = point.coords(self._coord_sys)
        # XXX Calling doit  is necessary with all the Subs expressions
        # XXX Calling simplify is necessary with all the trig expressions
        # 对坐标中的索引进行简化，并执行 doit() 操作。
        return simplify(coords[self._index]).doit()

    # XXX Workaround for limitations on the content of args
    free_symbols: set[Any] = set()
    # 定义一个基本向量场，用于给定坐标系的流形。

    Explanation
    ===========

    # 向量场是一个操作符，接受一个标量场并返回一个方向导数（也是标量场）。
    # 基本向量场是相同类型的操作符，但导数是特定选定坐标系的。

    To define a base vector field you need to choose the coordinate system and
    the index of the coordinate.

    # 要定义基本向量场，需要选择坐标系和坐标的索引。

    The use of the vector field after its definition is independent of the
    coordinate system in which it was defined, however due to limitations in the
    simplification routines you may arrive at more complicated expression if you
    use unappropriate coordinate systems.

    # 在定义向量场后，其使用与其定义的坐标系无关，但由于简化程序的限制，
    # 如果使用不合适的坐标系，则可能会得到更复杂的表达式。

    Parameters
    ==========

    coord_sys : CoordSystem
        # 坐标系对象

    index : integer
        # 坐标的索引

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import BaseVectorField
    >>> from sympy import pprint

    >>> x, y = R2_r.symbols
    >>> rho, theta = R2_p.symbols
    >>> fx, fy = R2_r.base_scalars()
    >>> point_p = R2_p.point([rho, theta])
    >>> point_r = R2_r.point([x, y])

    >>> g = Function('g')
    >>> s_field = g(fx, fy)

    >>> v = BaseVectorField(R2_r, 1)
    >>> pprint(v(s_field))
    / d           \|
    |---(g(x, xi))||
    \dxi          /|xi=y
    >>> pprint(v(s_field).rcall(point_r).doit())
    d
    --(g(x, y))
    dy
    >>> pprint(v(s_field).rcall(point_p))
    / d                        \|
    |---(g(rho*cos(theta), xi))||
    \dxi                       /|xi=rho*sin(theta)

    """

    # 指示这个类不是可交换的（在代数运算中的性质）

    is_commutative = False

    def __new__(cls, coord_sys, index, **kwargs):
        # 创建一个新的对象，确保索引是符号化的
        index = _sympify(index)
        obj = super().__new__(cls, coord_sys, index)
        obj._coord_sys = coord_sys
        obj._index = index
        return obj

    @property
    def coord_sys(self):
        # 返回这个向量场的坐标系
        return self.args[0]

    @property
    def index(self):
        # 返回这个向量场的索引
        return self.args[1]

    @property
    def patch(self):
        # 返回这个向量场的坐标系的修补程序
        return self.coord_sys.patch

    @property
    def manifold(self):
        # 返回这个向量场的流形
        return self.coord_sys.manifold

    @property
    def dim(self):
        # 返回这个向量场的流形的维度
        return self.manifold.dim
    def __call__(self, scalar_field):
        """Apply on a scalar field.
        The action of a vector field on a scalar field is a directional
        differentiation.
        If the argument is not a scalar field an error is raised.
        """
        # 检查输入的 scalar_field 是否具有协变或逆变阶数，如果有则抛出错误
        if covariant_order(scalar_field) or contravariant_order(scalar_field):
            raise ValueError('Only scalar fields can be supplied as arguments to vector fields.')

        # 如果 scalar_field 为 None，则返回当前对象自身
        if scalar_field is None:
            return self

        # 获取 scalar_field 中的基本标量字段列表
        base_scalars = list(scalar_field.atoms(BaseScalarField))

        # 第一步: 将 e_x(x+r**2) 转换为 e_x(x) + 2*r*e_x(r)
        d_var = self._coord_sys._dummy
        # TODO: 对于下一行，你需要一个真正的虚函数
        # 创建虚函数列表，以替换 scalar_field 中的基本标量字段
        d_funcs = [Function('_#_%s' % i)(d_var) for i, b in enumerate(base_scalars)]
        # 将 scalar_field 中的基本标量字段替换为虚函数，然后对结果求偏导数
        d_result = scalar_field.subs(list(zip(base_scalars, d_funcs)))
        d_result = d_result.diff(d_var)

        # 第二步: 将 e_x(x) 替换为 1，将 e_x(r) 替换为 cos(atan2(x, y))
        coords = self._coord_sys.symbols
        # 计算虚函数的导数
        d_funcs_deriv = [f.diff(d_var) for f in d_funcs]
        d_funcs_deriv_sub = []
        # 对每个基本标量字段执行以下操作
        for b in base_scalars:
            # 计算当前坐标系与基本标量字段的雅可比矩阵
            jac = self._coord_sys.jacobian(b._coord_sys, coords)
            # 提取所需的雅可比项
            d_funcs_deriv_sub.append(jac[b._index, self._index])
        # 将虚函数的导数替换为所需的雅可比项
        d_result = d_result.subs(list(zip(d_funcs_deriv, d_funcs_deriv_sub)))

        # 移除虚函数
        result = d_result.subs(list(zip(d_funcs, base_scalars)))
        # 将坐标替换为当前坐标系的坐标函数
        result = result.subs(list(zip(coords, self._coord_sys.coord_functions())))
        # 对最终结果应用 doit() 方法
        return result.doit()
def _find_coords(expr):
    # 查找表达式中存在的坐标系
    fields = expr.atoms(BaseScalarField, BaseVectorField)
    # 返回所有字段中使用的坐标系统的集合
    return {f._coord_sys for f in fields}


class Commutator(Expr):
    r"""Commutator of two vector fields.

    Explanation
    ===========

    The commutator of two vector fields `v_1` and `v_2` is defined as the
    vector field `[v_1, v_2]` that evaluated on each scalar field `f` is equal
    to `v_1(v_2(f)) - v_2(v_1(f))`.

    Examples
    ========


    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import Commutator
    >>> from sympy import simplify

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_r = R2_p.base_vector(0)

    >>> c_xy = Commutator(e_x, e_y)
    >>> c_xr = Commutator(e_x, e_r)
    >>> c_xy
    0

    Unfortunately, the current code is not able to compute everything:

    >>> c_xr
    Commutator(e_x, e_rho)
    >>> simplify(c_xr(fy**2))
    -2*cos(theta)*y**2/(x**2 + y**2)

    """
    def __new__(cls, v1, v2):
        # 检查输入的向量场是否是一阶的协变或逆变
        if (covariant_order(v1) or contravariant_order(v1) != 1
                or covariant_order(v2) or contravariant_order(v2) != 1):
            raise ValueError(
                'Only commutators of vector fields are supported.')
        # 如果两个向量场相同，返回零
        if v1 == v2:
            return S.Zero
        # 获取涉及向量场的所有坐标系统的集合
        coord_sys = set().union(*[_find_coords(v) for v in (v1, v2)])
        # 如果只使用了一个坐标系统，则可以直接计算交换子
        if len(coord_sys) == 1:
            if all(isinstance(v, BaseVectorField) for v in (v1, v2)):
                return S.Zero
            # 提取向量场中的基向量和系数
            bases_1, bases_2 = [list(v.atoms(BaseVectorField))
                                for v in (v1, v2)]
            coeffs_1 = [v1.expand().coeff(b) for b in bases_1]
            coeffs_2 = [v2.expand().coeff(b) for b in bases_2]
            res = 0
            # 计算交换子的表达式
            for c1, b1 in zip(coeffs_1, bases_1):
                for c2, b2 in zip(coeffs_2, bases_2):
                    res += c1 * b1(c2) * b2 - c2 * b2(c1) * b1
            return res
        else:
            # 如果涉及多个坐标系统，返回一个对象
            obj = super().__new__(cls, v1, v2)
            obj._v1 = v1 # 弃用的赋值
            obj._v2 = v2 # 弃用的赋值
            return obj

    @property
    def v1(self):
        return self.args[0]

    @property
    def v2(self):
        return self.args[1]

    def __call__(self, scalar_field):
        """Apply on a scalar field.
        If the argument is not a scalar field an error is raised.
        """
        # 在标量场上应用交换子操作
        return self.v1(self.v2(scalar_field)) - self.v2(self.v1(scalar_field))


class Differential(Expr):
    r"""Return the differential (exterior derivative) of a form field.

    Explanation
    ===========

    The differential of a form (i.e. the exterior derivative) has a complicated
    definition in the general case.
    The differential `df` of the 0-form `f` is defined for any vector field `v`
    as `df(v) = v(f)`.

    Examples
    ========

    ```
    is_commutative = False


    # 设置一个类属性，表示该类的实例不遵循交换律
    is_commutative = False



    def __new__(cls, form_field):
        # 检查输入的形式场是否为逆变序列
        if contravariant_order(form_field):
            raise ValueError(
                'A vector field was supplied as an argument to Differential.')
        # 如果输入的形式场已经是 Differential 类的实例，则返回零
        if isinstance(form_field, Differential):
            return S.Zero
        else:
            # 否则，调用父类的构造方法创建 Differential 实例
            obj = super().__new__(cls, form_field)
            # 将形式场赋值给实例的私有属性 _form_field（已过时的赋值方式）
            obj._form_field = form_field # deprecated assignment
            return obj


    # 返回 Differential 类的实例的形式场属性
    @property
    def form_field(self):
        return self.args[0]
    def __call__(self, *vector_fields):
        """
        在一组向量场上应用。

        Explanation
        ===========

        如果提供的向量场数量不等于 1 加上微分形式字段的阶数，结果是未定义的。

        对于 1-形式（即标量场的微分），评估为 `df(v)=v(f)`。然而，如果 `v` 是 ``None`` 而不是向量场，则返回不变的微分。这样做是为了允许高阶形式的部分缩并。

        在一般情况下，通过将每一对场替换为它们的交换子，将微分形式字段应用于比原始列表中元素少一个的列表中。

        如果参数不是向量或 ``None``，则会引发错误。
        """
        # 检查参数中是否有非向量场或 ``None``
        if any((contravariant_order(a) != 1 or covariant_order(a)) and a is not None
                for a in vector_fields):
            raise ValueError('The arguments supplied to Differential should be vector fields or Nones.')
        
        # 获取向量场数量
        k = len(vector_fields)
        
        if k == 1:
            # 对于单个向量场，如果不为 None，则调用其 rcall 方法，否则返回自身
            if vector_fields[0]:
                return vector_fields[0].rcall(self._form_field)
            return self
        else:
            # 对于高阶形式，应用更复杂：
            # 不变公式：
            # https://en.wikipedia.org/wiki/Exterior_derivative#Invariant_formula
            # df(v1, ... vn) = +/- vi(f(v1..no i..vn))
            #                  +/- f([vi,vj],v1..no i, no j..vn)
            f = self._form_field
            v = vector_fields
            ret = 0
            for i in range(k):
                # 计算第一项：vi(f(v1..no i..vn))
                t = v[i].rcall(f.rcall(*v[:i] + v[i + 1:]))
                ret += (-1)**i * t
                
                for j in range(i + 1, k):
                    # 计算第二项：f([vi,vj],v1..no i, no j..vn)
                    c = Commutator(v[i], v[j])
                    if c:  # 如果交换子不为零
                        t = f.rcall(*(c,) + v[:i] + v[i + 1:j] + v[j + 1:])
                        ret += (-1)**(i + j) * t
            
            return ret
# 定义一个名为 TensorProduct 的类，继承自 Expr 类
class TensorProduct(Expr):
    """Tensor product of forms.

    Explanation
    ===========

    The tensor product permits the creation of multilinear functionals (i.e.
    higher order tensors) out of lower order fields (e.g. 1-forms and vector
    fields). However, the higher tensors thus created lack the interesting
    features provided by the other type of product, the wedge product, namely
    they are not antisymmetric and hence are not form fields.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import TensorProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> TensorProduct(dx, dy)(e_x, e_y)
    1
    >>> TensorProduct(dx, dy)(e_y, e_x)
    0
    >>> TensorProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> TensorProduct(e_x, e_y)(fx**2, fy**2)
    4*x*y
    >>> TensorProduct(e_y, dx)(fy)
    dx

    You can nest tensor products.

    >>> tp1 = TensorProduct(dx, dy)
    >>> TensorProduct(tp1, dx)(e_x, e_y, e_x)
    1

    You can make partial contraction for instance when 'raising an index'.
    Putting ``None`` in the second argument of ``rcall`` means that the
    respective position in the tensor product is left as it is.

    >>> TP = TensorProduct
    >>> metric = TP(dx, dx) + 3*TP(dy, dy)
    >>> metric.rcall(e_y, None)
    3*dy

    Or automatically pad the args with ``None`` without specifying them.

    >>> metric.rcall(e_y)
    3*dy

    """
    
    # 定义 __new__ 方法，用于创建新的 TensorProduct 实例
    def __new__(cls, *args):
        # 计算输入参数中标量部分的乘积
        scalar = Mul(*[m for m in args if covariant_order(m) + contravariant_order(m) == 0])
        # 提取多重场部分，即具有协变和逆变次数的参数
        multifields = [m for m in args if covariant_order(m) + contravariant_order(m)]
        # 如果存在多重场部分
        if multifields:
            # 如果只有一个多重场参数，返回乘积后的结果
            if len(multifields) == 1:
                return scalar * multifields[0]
            # 否则，调用父类的 __new__ 方法，传递多重场参数，并返回乘积后的结果
            return scalar * super().__new__(cls, *multifields)
        else:
            # 如果没有多重场参数，直接返回标量乘积结果
            return scalar
    # 将该方法作为对象的调用接口，接受一组字段作为输入
    def __call__(self, *fields):
        """Apply on a list of fields.

        If the number of input fields supplied is not equal to the order of
        the tensor product field, the list of arguments is padded with ``None``'s.

        The list of arguments is divided in sublists depending on the order of
        the forms inside the tensor product. The sublists are provided as
        arguments to these forms and the resulting expressions are given to the
        constructor of ``TensorProduct``.

        """
        # 计算协变和逆变阶数之和
        tot_order = covariant_order(self) + contravariant_order(self)
        # 计算输入字段的总数
        tot_args = len(fields)
        # 如果输入字段总数不等于张量积场的总阶数，则用 None 填充输入列表
        if tot_args != tot_order:
            fields = list(fields) + [None]*(tot_order - tot_args)
        
        # 计算每个子表达式的阶数，根据表达式在张量积中的顺序
        orders = [covariant_order(f) + contravariant_order(f) for f in self._args]
        # 计算子表达式在输入字段中的索引位置
        indices = [sum(orders[:i + 1]) for i in range(len(orders) - 1)]
        # 划分输入字段为多个子列表，每个子列表作为一个表达式的输入
        fields = [fields[i:j] for i, j in zip([0] + indices, indices + [None])]
        # 对每个表达式和其对应的输入字段列表，调用其第一个元素的 rcall 方法
        multipliers = [t[0].rcall(*t[1]) for t in zip(self._args, fields)]
        # 返回张量积的结果，将每个表达式的结果作为参数传递给 TensorProduct 的构造函数
        return TensorProduct(*multipliers)
class WedgeProduct(TensorProduct):
    """Wedge product of forms.

    Explanation
    ===========

    In the context of integration only completely antisymmetric forms make
    sense. The wedge product permits the creation of such forms.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import WedgeProduct

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> WedgeProduct(dx, dy)(e_x, e_y)
    1
    >>> WedgeProduct(dx, dy)(e_y, e_x)
    -1
    >>> WedgeProduct(dx, fx*dy)(fx*e_x, e_y)
    x**2
    >>> WedgeProduct(e_x, e_y)(fy, None)
    -e_x

    You can nest wedge products.

    >>> wp1 = WedgeProduct(dx, dy)
    >>> WedgeProduct(wp1, dx)(e_x, e_y, e_x)
    0

    """
    # TODO the calculation of signatures is slow
    # TODO you do not need all these permutations (neither the prefactor)
    def __call__(self, *fields):
        """Apply on a list of vector_fields.
        The expression is rewritten internally in terms of tensor products and evaluated."""
        # 计算每个输入场的协变阶数和逆变阶数的总和
        orders = (covariant_order(e) + contravariant_order(e) for e in self.args)
        # 计算乘法因子，为每个排列的阶乘的倒数
        mul = 1/Mul(*(factorial(o) for o in orders))
        # 对输入场进行所有可能的排列
        perms = permutations(fields)
        # 计算每个排列的置换符号
        perms_par = (Permutation(
            p).signature() for p in permutations(range(len(fields))))
        # 创建一个包含所有参数的张量积对象
        tensor_prod = TensorProduct(*self.args)
        # 返回乘以乘法因子的结果，这是排列求和的总和
        return mul*Add(*[tensor_prod(*p[0])*p[1] for p in zip(perms, perms_par)])


class LieDerivative(Expr):
    """Lie derivative with respect to a vector field.

    Explanation
    ===========

    The transport operator that defines the Lie derivative is the pushforward of
    the field to be derived along the integral curve of the field with respect
    to which one derives.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r, R2_p
    >>> from sympy.diffgeom import (LieDerivative, TensorProduct)

    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> e_rho, e_theta = R2_p.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> LieDerivative(e_x, fy)
    0
    >>> LieDerivative(e_x, fx)
    1
    >>> LieDerivative(e_x, e_x)
    0

    The Lie derivative of a tensor field by another tensor field is equal to
    their commutator:

    >>> LieDerivative(e_x, e_rho)
    Commutator(e_x, e_rho)
    >>> LieDerivative(e_x + e_y, fx)
    1

    >>> tp = TensorProduct(dx, dy)
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))
    >>> LieDerivative(e_x, tp)
    LieDerivative(e_x, TensorProduct(dx, dy))

    """
    # 定义一个特殊方法 `__new__`，用于创建新的实例对象
    def __new__(cls, v_field, expr):
        # 计算表达式的协变阶数
        expr_form_ord = covariant_order(expr)
        
        # 检查逆变阶数是否为1且协变阶数是否非零，如果条件不满足则抛出错误
        if contravariant_order(v_field) != 1 or covariant_order(v_field):
            raise ValueError('Lie derivatives are defined only with respect to'
                             ' vector fields. The supplied argument was not a '
                             'vector field.')
        
        # 如果表达式的协变阶数大于0，则调用父类的 `__new__` 方法创建对象
        obj = super().__new__(cls, v_field, expr)
        
        # deprecated assignments
        # 设置对象的 `_v_field` 和 `_expr` 属性
        obj._v_field = v_field
        obj._expr = expr
        
        # 返回创建的对象
        return obj

    @property
    # 定义一个装饰器属性 `v_field`，返回对象的第一个参数，即向量场
    def v_field(self):
        return self.args[0]

    @property
    # 定义一个装饰器属性 `expr`，返回对象的第二个参数，即表达式
    def expr(self):
        return self.args[1]

    # 定义一个特殊方法 `__call__`，使对象可以被调用
    def __call__(self, *args):
        # 获取向量场和表达式
        v = self.v_field
        expr = self.expr
        
        # 计算主导项，即向量场作用于表达式再次参数化的结果
        lead_term = v(expr(*args))
        
        # 计算剩余项，使用交换子 `Commutator(v, args[i])` 替换参数的乘积，并求和
        rest = Add(*[Mul(*args[:i] + (Commutator(v, args[i]),) + args[i + 1:])
                     for i in range(len(args))])
        
        # 返回主导项减去剩余项的结果
        return lead_term - rest
# 定义一个基于表达式的类，表示相对于基向量的协变导数操作符
class BaseCovarDerivativeOp(Expr):
    """Covariant derivative operator with respect to a base vector.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2_r
    >>> from sympy.diffgeom import BaseCovarDerivativeOp
    >>> from sympy.diffgeom import metric_to_Christoffel_2nd, TensorProduct

    >>> TP = TensorProduct
    >>> fx, fy = R2_r.base_scalars()
    >>> e_x, e_y = R2_r.base_vectors()
    >>> dx, dy = R2_r.base_oneforms()

    >>> ch = metric_to_Christoffel_2nd(TP(dx, dx) + TP(dy, dy))
    >>> ch
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    >>> cvd(fx)
    1
    >>> cvd(fx*e_x)
    e_x
    """

    def __new__(cls, coord_sys, index, christoffel):
        # 将索引转换为符号表达式
        index = _sympify(index)
        # 将 Christoffel 符号转换为不可变的稠密多维数组
        christoffel = ImmutableDenseNDimArray(christoffel)
        # 调用父类的构造函数创建对象
        obj = super().__new__(cls, coord_sys, index, christoffel)
        # 废弃的属性赋值
        obj._coord_sys = coord_sys
        obj._index = index
        obj._christoffel = christoffel
        return obj

    @property
    def coord_sys(self):
        # 返回导数操作符的坐标系属性
        return self.args[0]

    @property
    def index(self):
        # 返回导数操作符的索引属性
        return self.args[1]

    @property
    def christoffel(self):
        # 返回导数操作符的 Christoffel 符号属性
        return self.args[2]
    def __call__(self, field):
        """
        在标量场上应用。

        矢量场对标量场的作用是方向导数。
        如果参数不是标量场，则行为是未定义的。
        """
        # 检查场的协变阶数是否为零，若不是则抛出未实现错误
        if covariant_order(field) != 0:
            raise NotImplementedError()

        # 将场按照给定的坐标系转换为基向量的形式
        field = vectors_in_basis(field, self._coord_sys)

        # 获取当前作用于的基向量和标量
        wrt_vector = self._coord_sys.base_vector(self._index)
        wrt_scalar = self._coord_sys.coord_function(self._index)

        # 提取场中的所有基向量
        vectors = list(field.atoms(BaseVectorField))

        # 第一步：用虚拟函数替换所有的基向量，并进行导数操作
        # TODO: 你需要一个真实的虚拟函数来替换下一行
        d_funcs = [Function('_#_%s' % i)(wrt_scalar) for i,
                   b in enumerate(vectors)]
        d_result = field.subs(list(zip(vectors, d_funcs)))
        d_result = wrt_vector(d_result)

        # 第二步：将基向量还原回去
        d_result = d_result.subs(list(zip(d_funcs, vectors)))

        # 第三步：计算基向量的导数
        derivs = []
        for v in vectors:
            d = Add(*[(self._christoffel[k, wrt_vector._index, v._index]
                       *v._coord_sys.base_vector(k))
                      for k in range(v._coord_sys.dim)])
            derivs.append(d)

        to_subs = [wrt_vector(d) for d in d_funcs]
        # XXX: 当存在虚拟符号并且缓存被禁用时，这个替换可能失败：
        # https://github.com/sympy/sympy/issues/17794
        result = d_result.subs(list(zip(to_subs, derivs)))

        # 移除虚拟函数
        result = result.subs(list(zip(d_funcs, vectors)))
        return result.doit()
class CovarDerivativeOp(Expr):
    """Covariant derivative operator."""

    def __new__(cls, wrt, christoffel):
        # 检查被求导变量的基向量字段的坐标系数量是否超过一个
        if len({v._coord_sys for v in wrt.atoms(BaseVectorField)}) > 1:
            raise NotImplementedError()
        # 检查被求导变量的逆变阶数是否为1，协变阶数是否为0
        if contravariant_order(wrt) != 1 or covariant_order(wrt):
            raise ValueError('Covariant derivatives are defined only with '
                             'respect to vector fields. The supplied argument '
                             'was not a vector field.')
        # 将 Christoffel 符号转为不可变的密集多维数组
        christoffel = ImmutableDenseNDimArray(christoffel)
        # 调用父类的构造方法创建对象
        obj = super().__new__(cls, wrt, christoffel)
        # deprecated assignments（不推荐使用的赋值）
        obj._wrt = wrt
        obj._christoffel = christoffel
        return obj

    @property
    def wrt(self):
        # 返回被求导变量
        return self.args[0]

    @property
    def christoffel(self):
        # 返回 Christoffel 符号
        return self.args[1]

    def __call__(self, field):
        # 获取被求导变量的所有基向量字段
        vectors = list(self._wrt.atoms(BaseVectorField))
        # 创建基于 Christoffel 符号的协变导数操作对象列表
        base_ops = [BaseCovarDerivativeOp(v._coord_sys, v._index, self._christoffel)
                    for v in vectors]
        # 替换被求导变量中的基向量字段，并调用 field 的右调函数
        return self._wrt.subs(list(zip(vectors, base_ops))).rcall(field)


###############################################################################
# Integral curves on vector fields
###############################################################################
def intcurve_series(vector_field, param, start_point, n=6, coord_sys=None, coeffs=False):
    r"""Return the series expansion for an integral curve of the field.

    Explanation
    ===========

    Integral curve is a function `\gamma` taking a parameter in `R` to a point
    in the manifold. It verifies the equation:

    `V(f)\big(\gamma(t)\big) = \frac{d}{dt}f\big(\gamma(t)\big)`

    where the given ``vector_field`` is denoted as `V`. This holds for any
    value `t` for the parameter and any scalar field `f`.

    This equation can also be decomposed of a basis of coordinate functions
    `V(f_i)\big(\gamma(t)\big) = \frac{d}{dt}f_i\big(\gamma(t)\big) \quad \forall i`

    This function returns a series expansion of `\gamma(t)` in terms of the
    coordinate system ``coord_sys``. The equations and expansions are necessarily
    done in coordinate-system-dependent way as there is no other way to
    represent movement between points on the manifold (i.e. there is no such
    thing as a difference of points for a general manifold).

    Parameters
    ==========
    vector_field
        给定的矢量场，将生成积分曲线

    param
        函数 `\gamma` 的参数，从实数到曲线

    start_point
        对应于 `\gamma(0)` 的点

    n
        要展开到的阶数

    coord_sys
        展开所用的坐标系
        coeffs (默认为 False) - 如果为 True，则返回展开项的列表

    Examples
    ========

    使用预定义的 R2 流形：

    >>> from sympy.abc import t, x, y
    >>> from sympy.diffgeom.rn import R2_p, R2_r
    >>> from sympy.diffgeom import intcurve_series

    指定起始点和矢量场：

    >>> start_point = R2_r.point([x, y])
    >>> vector_field = R2_r.e_x

    计算级数：

    >>> intcurve_series(vector_field, t, start_point, n=3)
    Matrix([
    [t + x],
    [    y]])

    或者获取展开项的列表：

    >>> series = intcurve_series(vector_field, t, start_point, n=3, coeffs=True)
    >>> series[0]
    Matrix([
    [x],
    [y]])
    >>> series[1]
    Matrix([
    [t],
    [0]])
    >>> series[2]
    Matrix([
    [0],
    [0]])

    极坐标系中的级数：

    >>> series = intcurve_series(vector_field, t, start_point,
    ...             n=3, coord_sys=R2_p, coeffs=True)
    >>> series[0]
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> series[1]
    Matrix([
    [t*x/sqrt(x**2 + y**2)],
    [   -t*y/(x**2 + y**2)]])
    >>> series[2]
    Matrix([
    [t**2*(-x**2/(x**2 + y**2)**(3/2) + 1/sqrt(x**2 + y**2))/2],
    [                                t**2*x*y/(x**2 + y**2)**2]])

    See Also
    ========

    intcurve_diffequ
    """
    # 检查矢量场的协变和逆变阶数是否满足要求
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')

    def iter_vfield(scalar_field, i):
        """在 ``scalar_field`` 上对 ``vector_field`` 迭代 `i` 次调用."""
        return reduce(lambda s, v: v.rcall(s), [vector_field, ]*i, scalar_field)

    def taylor_terms_per_coord(coord_function):
        """返回一个坐标函数的级数."""
        return [param**i*iter_vfield(coord_function, i).rcall(start_point)/factorial(i)
                for i in range(n)]
    
    # 如果未指定坐标系，则使用起始点的坐标系
    coord_sys = coord_sys if coord_sys else start_point._coord_sys
    coord_functions = coord_sys.coord_functions()
    taylor_terms = [taylor_terms_per_coord(f) for f in coord_functions]
    
    # 如果指定返回展开项的列表，则返回列表，否则返回合并后的矩阵
    if coeffs:
        return [Matrix(t) for t in zip(*taylor_terms)]
    else:
        return Matrix([sum(c) for c in taylor_terms])
def intcurve_diffequ(vector_field, param, start_point, coord_sys=None):
    r"""Return the differential equation for an integral curve of the field.

    Explanation
    ===========

    Integral curve is a function `\gamma` taking a parameter in `R` to a point
    in the manifold. It verifies the equation:

    `V(f)\big(\gamma(t)\big) = \frac{d}{dt}f\big(\gamma(t)\big)`

    where the given ``vector_field`` is denoted as `V`. This holds for any
    value `t` for the parameter and any scalar field `f`.

    This function returns the differential equation of `\gamma(t)` in terms of the
    coordinate system ``coord_sys``. The equations and expansions are necessarily
    done in a coordinate-system-dependent way as there is no other way to
    represent movement between points on the manifold (i.e. there is no such
    thing as a difference of points for a general manifold).

    Parameters
    ==========

    vector_field
        the vector field for which an integral curve will be given

    param
        the argument of the function `\gamma` from `R` to the curve

    start_point
        the point which corresponds to `\gamma(0)`

    coord_sys
        the coordinate system in which to give the equations

    Returns
    =======

    a tuple of (equations, initial conditions)

    Examples
    ========

    Use the predefined R2 manifold:

    >>> from sympy.abc import t
    >>> from sympy.diffgeom.rn import R2, R2_p, R2_r
    >>> from sympy.diffgeom import intcurve_diffequ

    Specify a starting point and a vector field:

    >>> start_point = R2_r.point([0, 1])
    >>> vector_field = -R2.y*R2.e_x + R2.x*R2.e_y

    Get the equation:

    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point)
    >>> equations
    [f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]
    >>> init_cond
    [f_0(0), f_1(0) - 1]

    The series in the polar coordinate system:

    >>> equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)
    >>> equations
    [Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]
    >>> init_cond
    [f_0(0) - 1, f_1(0) - pi/2]

    See Also
    ========

    intcurve_series

    """
    # Check if the vector field is a valid contravariant vector field
    if contravariant_order(vector_field) != 1 or covariant_order(vector_field):
        raise ValueError('The supplied field was not a vector field.')

    # Use the provided coordinate system or default to the start point's coordinate system
    coord_sys = coord_sys if coord_sys else start_point._coord_sys

    # Create symbols for the coordinates of the integral curve
    gammas = [Function('f_%d' % i)(param) for i in range(start_point._coord_sys.dim)]

    # Create a point with arbitrary coordinates based on the chosen coordinate system
    arbitrary_p = Point(coord_sys, gammas)

    # Obtain the coordinate functions of the coordinate system
    coord_functions = coord_sys.coord_functions()

    # Compute the differential equations for the integral curve
    equations = [simplify(diff(cf.rcall(arbitrary_p), param) - vector_field.rcall(cf).rcall(arbitrary_p))
                 for cf in coord_functions]

    # Compute the initial conditions for the integral curve
    init_cond = [simplify(cf.rcall(arbitrary_p).subs(param, 0) - cf.rcall(start_point))
                 for cf in coord_functions]

    # Return the computed equations and initial conditions as a tuple
    return equations, init_cond
###############################################################################
def dummyfy(args, exprs):
    # 创建一个矩阵，其中每个参数都转换为虚拟变量
    d_args = Matrix([s.as_dummy() for s in args])
    # 创建一个替换字典，将原始参数映射到虚拟变量
    reps = dict(zip(args, d_args))
    # 对表达式列表中的每个表达式进行符号计算，并使用替换字典替换参数
    d_exprs = Matrix([_sympify(expr).subs(reps) for expr in exprs])
    # 返回虚拟变量的列表和替换后的表达式列表
    return d_args, d_exprs

###############################################################################
# Helpers
###############################################################################
def contravariant_order(expr, _strict=False):
    """Return the contravariant order of an expression.

    Examples
    ========

    >>> from sympy.diffgeom import contravariant_order
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.abc import a

    >>> contravariant_order(a)
    0
    >>> contravariant_order(a*R2.x + 2)
    0
    >>> contravariant_order(a*R2.x*R2.e_y + R2.e_x)
    1

    """
    # 如果表达式是加法
    if isinstance(expr, Add):
        # 获取每个子表达式的逆变量阶数
        orders = [contravariant_order(e) for e in expr.args]
        # 检查是否所有子表达式的逆变量阶数相同
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing contravariant fields of varying order.')
        return orders[0]
    # 如果表达式是乘法
    elif isinstance(expr, Mul):
        # 获取每个子表达式的逆变量阶数
        orders = [contravariant_order(e) for e in expr.args]
        # 找出非零阶数的子表达式
        not_zero = [o for o in orders if o != 0]
        # 如果存在超过一个非零阶数的子表达式，则报错
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between vectors.')
        return 0 if not not_zero else not_zero[0]
    # 如果表达式是幂运算
    elif isinstance(expr, Pow):
        # 如果底数或指数具有协变量阶数，则报错
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError('Misformed expression containing a power of a vector.')
        return 0
    # 如果表达式是基本矢量场
    elif isinstance(expr, BaseVectorField):
        return 1
    # 如果表达式是张量积
    elif isinstance(expr, TensorProduct):
        # 计算表达式中所有部分的逆变量阶数之和
        return sum(contravariant_order(a) for a in expr.args)
    # 如果不需要严格检查或表达式中包含基本标量场
    elif not _strict or expr.atoms(BaseScalarField):
        return 0
    else:  # 如果表达式不包含与diffgeom模块相关的内容并且是严格模式
        return -1


def covariant_order(expr, _strict=False):
    """Return the covariant order of an expression.

    Examples
    ========

    >>> from sympy.diffgeom import covariant_order
    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.abc import a

    >>> covariant_order(a)
    0
    >>> covariant_order(a*R2.x + 2)
    0
    >>> covariant_order(a*R2.x*R2.dy + R2.dx)
    1

    """
    # 如果表达式是加法
    if isinstance(expr, Add):
        # 获取每个子表达式的协变量阶数
        orders = [covariant_order(e) for e in expr.args]
        # 检查是否所有子表达式的协变量阶数相同
        if len(set(orders)) != 1:
            raise ValueError('Misformed expression containing form fields of varying order.')
        return orders[0]
    # 如果表达式是乘法
    elif isinstance(expr, Mul):
        # 获取每个子表达式的协变量阶数
        orders = [covariant_order(e) for e in expr.args]
        # 找出非零阶数的子表达式
        not_zero = [o for o in orders if o != 0]
        # 如果存在超过一个非零阶数的子表达式，则报错
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between vectors.')
        return 0 if not not_zero else not_zero[0]
    else:
        # 如果不需要严格检查或表达式中包含基本标量场
        if not _strict or expr.atoms(BaseScalarField):
            return 0
        else:  # 如果表达式不包含与diffgeom模块相关的内容并且是严格模式
            return -1
    elif isinstance(expr, Mul):
        # 如果表达式是乘法表达式，计算每个因子的协变阶数
        orders = [covariant_order(e) for e in expr.args]
        # 找出所有非零阶数
        not_zero = [o for o in orders if o != 0]
        # 如果非零阶数超过一个，抛出数值错误异常
        if len(not_zero) > 1:
            raise ValueError('Misformed expression containing multiplication between forms.')
        # 如果没有非零阶数，返回0；否则返回第一个非零阶数
        return 0 if not not_zero else not_zero[0]
    elif isinstance(expr, Pow):
        # 如果表达式是幂次表达式，检查基数或指数是否包含协变阶数，若是则抛出数值错误异常
        if covariant_order(expr.base) or covariant_order(expr.exp):
            raise ValueError('Misformed expression containing a power of a form.')
        # 否则返回0
        return 0
    elif isinstance(expr, Differential):
        # 如果表达式是微分表达式，计算其参数的协变阶数并加1
        return covariant_order(*expr.args) + 1
    elif isinstance(expr, TensorProduct):
        # 如果表达式是张量积表达式，计算所有参数的协变阶数之和
        return sum(covariant_order(a) for a in expr.args)
    elif not _strict or expr.atoms(BaseScalarField):
        # 如果不是严格模式或表达式包含 BaseScalarField 类型的原子对象，返回0
        return 0
    else:  # 如果表达式不包含与 diffgeom 模块相关的任何内容，并且是 _strict
        # 返回-1
        return -1
###############################################################################
# Coordinate transformation functions
###############################################################################
def vectors_in_basis(expr, to_sys):
    """Transform all base vectors in base vectors of a specified coord basis.
    While the new base vectors are in the new coordinate system basis, any
    coefficients are kept in the old system.

    Examples
    ========

    >>> from sympy.diffgeom import vectors_in_basis
    >>> from sympy.diffgeom.rn import R2_r, R2_p

    >>> vectors_in_basis(R2_r.e_x, R2_p)
    -y*e_theta/(x**2 + y**2) + x*e_rho/sqrt(x**2 + y**2)
    >>> vectors_in_basis(R2_p.e_r, R2_r)
    sin(theta)*e_y + cos(theta)*e_x

    """
    # 获取所有在表达式中出现的基向量
    vectors = list(expr.atoms(BaseVectorField))
    new_vectors = []
    # 遍历每一个基向量
    for v in vectors:
        cs = v._coord_sys
        # 计算从当前坐标系到目标坐标系的雅可比矩阵
        jac = cs.jacobian(to_sys, cs.coord_functions())
        # 计算新的基向量
        new = (jac.T * Matrix(to_sys.base_vectors()))[v._index]
        new_vectors.append(new)
    # 将原始表达式中的基向量替换为新的基向量
    return expr.subs(list(zip(vectors, new_vectors)))


###############################################################################
# Coordinate-dependent functions
###############################################################################
def twoform_to_matrix(expr):
    """Return the matrix representing the twoform.

    For the twoform `w` return the matrix `M` such that `M[i,j]=w(e_i, e_j)`,
    where `e_i` is the i-th base vector field for the coordinate system in
    which the expression of `w` is given.

    Examples
    ========

    >>> from sympy.diffgeom.rn import R2
    >>> from sympy.diffgeom import twoform_to_matrix, TensorProduct
    >>> TP = TensorProduct

    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    Matrix([
    [1, 0],
    [0, 1]])
    >>> twoform_to_matrix(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    Matrix([
    [x, 0],
    [0, 1]])
    >>> twoform_to_matrix(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy) - TP(R2.dx, R2.dy)/2)
    Matrix([
    [   1, 0],
    [-1/2, 1]])

    """
    # 检查表达式是否是二形式
    if covariant_order(expr) != 2 or contravariant_order(expr):
        raise ValueError('The input expression is not a two-form.')
    # 确定表达式所在的坐标系
    coord_sys = _find_coords(expr)
    if len(coord_sys) != 1:
        raise ValueError('The input expression concerns more than one '
                         'coordinate systems, hence there is no unambiguous '
                         'way to choose a coordinate system for the matrix.')
    coord_sys = coord_sys.pop()
    vectors = coord_sys.base_vectors()
    # 展开表达式
    expr = expr.expand()
    # 构造表示矩阵的内容
    matrix_content = [[expr.rcall(v1, v2) for v1 in vectors]
                      for v2 in vectors]
    return Matrix(matrix_content)


def metric_to_Christoffel_1st(expr):
    """Return the nested list of Christoffel symbols for the given metric.
    This returns the Christoffel symbol of first kind that represents the
    Levi-Civita connection for the given metric.

    Examples
    ========

    # 略，需要继续注释的部分

    """
    # 这个函数将返回给定度量的Christoffel符号的嵌套列表。
    # 它代表给定度量的Levi-Civita连接的第一种Christoffel符号。
    # 示例将在函数之后提供。
    # 导入必要的库中的模块 R2
    >>> from sympy.diffgeom.rn import R2
    # 导入 sympy.diffgeom 库中的 metric_to_Christoffel_1st 和 TensorProduct 函数
    >>> from sympy.diffgeom import metric_to_Christoffel_1st, TensorProduct
    # 为 TensorProduct 函数创建别名 TP
    >>> TP = TensorProduct

    # 计算给定度量形式对应的 Christoffel 符号的第一类
    >>> metric_to_Christoffel_1st(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    # 返回的结果是一个三维数组，每个元素为二阶导数的组合值
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    # 计算带有乘法因子 R2.x 的度量形式对应的 Christoffel 符号的第一类
    >>> metric_to_Christoffel_1st(R2.x*TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    # 返回的结果是一个三维数组，其中包含了带有乘法因子的一阶导数组合值

    """
    # 将二次形式转换为对应的矩阵表示
    matrix = twoform_to_matrix(expr)
    # 检查矩阵是否对称，如果不对称则抛出 ValueError 异常
    if not matrix.is_symmetric():
        raise ValueError(
            'The two-form representing the metric is not symmetric.')
    # 确定坐标系
    coord_sys = _find_coords(expr).pop()
    # 计算每个坐标系基向量对应的导数矩阵
    deriv_matrices = [matrix.applyfunc(d) for d in coord_sys.base_vectors()]
    # 确定索引范围
    indices = list(range(coord_sys.dim))
    # 计算 Christoffel 符号
    christoffel = [[[(deriv_matrices[k][i, j] + deriv_matrices[j][i, k] - deriv_matrices[i][j, k])/2
                     for k in indices]
                    for j in indices]
                   for i in indices]
    # 返回不可变的稠密 N 维数组，其中包含 Christoffel 符号的计算结果
    return ImmutableDenseNDimArray(christoffel)
# 根据给定的度量表达式计算其对应的第二类 Christoffel 符号的嵌套列表
def metric_to_Christoffel_2nd(expr):
    # 调用函数 metric_to_Christoffel_1st 计算给定度量表达式的第一类 Christoffel 符号
    ch_1st = metric_to_Christoffel_1st(expr)
    # 查找度量表达式中的坐标系
    coord_sys = _find_coords(expr).pop()
    # 生成坐标系的索引列表
    indices = list(range(coord_sys.dim))
    # XXX 临时解决方案，如果矩阵包含非符号元素，则矩阵求逆操作将无法完成
    # 矩阵转换为二次形式后求逆
    #matrix = twoform_to_matrix(expr).inv()
    matrix = twoform_to_matrix(expr)
    # 将所有 BaseScalarField 原子元素添加到集合中
    s_fields = set()
    for e in matrix:
        s_fields.update(e.atoms(BaseScalarField))
    s_fields = list(s_fields)
    dums = coord_sys.symbols
    # 用坐标系的符号替换集合中的元素，然后求逆矩阵，再用坐标系的符号替换集合中的元素
    matrix = matrix.subs(list(zip(s_fields, dums))).inv().subs(list(zip(dums, s_fields)))
    # XXX 解决方案结束
    # 计算 Christoffel 符号的第二类嵌套列表
    christoffel = [[[Add(*[matrix[i, l]*ch_1st[l, j, k] for l in indices])
                     for k in indices]
                    for j in indices]
                   for i in indices]
    # 返回不可变的稠密多维数组 christoffel
    return ImmutableDenseNDimArray(christoffel)


# 根据给定的度量表达式计算其在给定基础上的 Riemann 张量分量
def metric_to_Riemann_components(expr):
    # 调用函数 metric_to_Christoffel_2nd 计算给定度量表达式的第二类 Christoffel 符号
    ch_2nd = metric_to_Christoffel_2nd(expr)
    # 查找度量表达式中的坐标系
    coord_sys = _find_coords(expr).pop()
    # 生成坐标系的索引列表
    indices = list(range(coord_sys.dim))
    # 计算对 Christoffel 符号的导数
    deriv_ch = [[[[d(ch_2nd[i, j, k])
                   for d in coord_sys.base_vectors()]
                  for k in indices]
                 for j in indices]
                for i in indices]
    # 计算黎曼张量的第一部分 a_{rho,sig,nu,mu} = deriv_ch[rho][sig][nu][mu] - deriv_ch[rho][sig][mu][nu]
    riemann_a = [[[[deriv_ch[rho][sig][nu][mu] - deriv_ch[rho][sig][mu][nu]
                    for nu in indices]
                   for mu in indices]
                  for sig in indices]
                 for rho in indices]

    # 计算黎曼张量的第二部分 b_{rho,sig,nu,mu} = Sum(ch_2nd[rho, l, mu] * ch_2nd[l, sig, nu] - ch_2nd[rho, l, nu] * ch_2nd[l, sig, mu] for l in indices)
    riemann_b = [[[[Add(*[ch_2nd[rho, l, mu]*ch_2nd[l, sig, nu] - ch_2nd[rho, l, nu]*ch_2nd[l, sig, mu] for l in indices])
                    for nu in indices]
                   for mu in indices]
                  for sig in indices]
                 for rho in indices]

    # 计算黎曼张量 Riemann_{rho,sig,nu,mu} = a_{rho,sig,nu,mu} + b_{rho,sig,nu,mu}
    riemann = [[[[riemann_a[rho][sig][mu][nu] + riemann_b[rho][sig][mu][nu]
                  for nu in indices]
                 for mu in indices]
                for sig in indices]
               for rho in indices]

    # 返回不可变的稠密 N 维数组，表示计算得到的黎曼张量
    return ImmutableDenseNDimArray(riemann)
# 定义函数，计算给定度规下的Ricci张量分量
def metric_to_Ricci_components(expr):
    # 调用metric_to_Riemann_components函数计算给定度规下的Riemann张量分量
    riemann = metric_to_Riemann_components(expr)
    # 获取度规表达式中的坐标系，并假设只有一个坐标系
    coord_sys = _find_coords(expr).pop()
    # 生成坐标系的索引列表
    indices = list(range(coord_sys.dim))
    # 计算Ricci张量的分量，使用给定的Riemann张量分量进行计算
    ricci = [[Add(*[riemann[k, i, k, j] for k in indices])  # 计算Ricci张量分量的表达式
              for j in indices]  # 对每个j索引遍历
             for i in indices]  # 对每个i索引遍历
    # 返回不可变的多维数组，表示计算得到的Ricci张量分量
    return ImmutableDenseNDimArray(ricci)

###############################################################################
# Classes for deprecation
###############################################################################

# 定义一个被弃用的容器类
class _deprecated_container:
    # 这个类用于发出弃用警告
    # 当完全删除弃用的特性时，应该同时移除这个类
    # 参见 https://github.com/sympy/sympy/pull/19368
    def __init__(self, message, data):
        super().__init__(data)
        self.message = message

    # 发出弃用警告
    def warn(self):
        sympy_deprecation_warning(
            self.message,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-diffgeom-mutable",
            stacklevel=4
        )

    # 重载迭代器方法，发出弃用警告后调用父类的方法
    def __iter__(self):
        self.warn()
        return super().__iter__()

    # 重载索引访问方法，发出弃用警告后调用父类的方法
    def __getitem__(self, key):
        self.warn()
        return super().__getitem__(key)

    # 重载包含性检查方法，发出弃用警告后调用父类的方法
    def __contains__(self, key):
        self.warn()
        return super().__contains__(key)


# _deprecated_container类的子类，继承list类，表示一个被弃用的列表
class _deprecated_list(_deprecated_container, list):
    pass


# _deprecated_container类的子类，继承dict类，表示一个被弃用的字典
class _deprecated_dict(_deprecated_container, dict):
    pass


# 在最后导入，以避免循环导入
# 导入simplify函数，用于简化操作
from sympy.simplify.simplify import simplify
```