# `D:\src\scipysrc\sympy\sympy\physics\units\dimensions.py`

```
"""
Definition of physical dimensions.

Unit systems will be constructed on top of these dimensions.

Most of the examples in the doc use MKS system and are presented from the
computer point of view: from a human point, adding length to time is not legal
in MKS but it is in natural system; for a computer in natural system there is
no time dimension (but a velocity dimension instead) - in the basis - so the
question of adding time to length has no meaning.
"""

from __future__ import annotations

import collections
from functools import reduce

from sympy.core.basic import Basic  # 导入基本数学对象
from sympy.core.containers import (Dict, Tuple)  # 导入容器类对象
from sympy.core.singleton import S  # 导入单例类对象
from sympy.core.sorting import default_sort_key  # 导入排序相关函数
from sympy.core.symbol import Symbol  # 导入符号类对象
from sympy.core.sympify import sympify  # 导入符号表达式化函数
from sympy.matrices.dense import Matrix  # 导入密集矩阵类对象
from sympy.functions.elementary.trigonometric import TrigonometricFunction  # 导入三角函数类对象
from sympy.core.expr import Expr  # 导入表达式类对象
from sympy.core.power import Pow  # 导入幂函数类对象


class _QuantityMapper:

    _quantity_scale_factors_global: dict[Expr, Expr] = {}  # 全局量的比例因子映射表
    _quantity_dimensional_equivalence_map_global: dict[Expr, Expr] = {}  # 全局量的维度等价映射表
    _quantity_dimension_global: dict[Expr, Expr] = {}  # 全局量的维度映射表

    def __init__(self, *args, **kwargs):
        self._quantity_dimension_map = {}  # 实例化时创建的量的维度映射表
        self._quantity_scale_factors = {}  # 实例化时创建的量的比例因子映射表

    def set_quantity_dimension(self, quantity, dimension):
        """
        Set the dimension for the quantity in a unit system.

        If this relation is valid in every unit system, use
        ``quantity.set_global_dimension(dimension)`` instead.
        """
        from sympy.physics.units import Quantity  # 导入单位量类对象
        dimension = sympify(dimension)  # 将维度参数转换为符号表达式
        if not isinstance(dimension, Dimension):  # 检查维度是否为维度对象的实例
            if dimension == 1:
                dimension = Dimension(1)  # 如果维度为1，创建一个维度为1的对象
            else:
                raise ValueError("expected dimension or 1")  # 抛出值错误异常
        elif isinstance(dimension, Quantity):  # 如果维度是单位量的实例
            dimension = self.get_quantity_dimension(dimension)  # 获取该单位量的维度
        self._quantity_dimension_map[quantity] = dimension  # 将量和其维度映射存入实例化的映射表中
    def set_quantity_scale_factor(self, quantity, scale_factor):
        """
        Set the scale factor of a quantity relative to another quantity.

        It should be used only once per quantity to just one other quantity,
        the algorithm will then be able to compute the scale factors to all
        other quantities.

        In case the scale factor is valid in every unit system, please use
        ``quantity.set_global_relative_scale_factor(scale_factor)`` instead.
        """
        from sympy.physics.units import Quantity
        from sympy.physics.units.prefixes import Prefix
        # 将scale_factor转换为Sympy符号对象
        scale_factor = sympify(scale_factor)
        # 将所有前缀替换为它们相对于标准单位的比例因子
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        # 将所有量替换为它们相对于标准单位的比例因子
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Quantity),
            lambda x: self.get_quantity_scale_factor(x)
        )
        # 将quantity与其对应的scale_factor关联存储在_quantity_scale_factors字典中
        self._quantity_scale_factors[quantity] = scale_factor

    def get_quantity_dimension(self, unit):
        from sympy.physics.units import Quantity
        # 首先在本地维度映射中查找，然后在全局维度映射中查找：
        if unit in self._quantity_dimension_map:
            return self._quantity_dimension_map[unit]
        if unit in self._quantity_dimension_global:
            return self._quantity_dimension_global[unit]
        if unit in self._quantity_dimensional_equivalence_map_global:
            dep_unit = self._quantity_dimensional_equivalence_map_global[unit]
            if isinstance(dep_unit, Quantity):
                return self.get_quantity_dimension(dep_unit)
            else:
                return Dimension(self.get_dimensional_expr(dep_unit))
        # 如果unit是Quantity对象，则返回其维度
        if isinstance(unit, Quantity):
            return Dimension(unit.name)
        else:
            return Dimension(1)

    def get_quantity_scale_factor(self, unit):
        # 如果unit存在于_quantity_scale_factors字典中，则返回对应的比例因子
        if unit in self._quantity_scale_factors:
            return self._quantity_scale_factors[unit]
        # 如果unit存在于_quantity_scale_factors_global字典中，则计算其全局比例因子
        if unit in self._quantity_scale_factors_global:
            mul_factor, other_unit = self._quantity_scale_factors_global[unit]
            return mul_factor*self.get_quantity_scale_factor(other_unit)
        # 默认返回1（即无缩放因子）
        return S.One
# 表示物理量的维度的类
class Dimension(Expr):
    """
    This class represent the dimension of a physical quantities.

    The ``Dimension`` constructor takes as parameters a name and an optional
    symbol.

    For example, in classical mechanics we know that time is different from
    temperature and dimensions make this difference (but they do not provide
    any measure of these quantites.

        >>> from sympy.physics.units import Dimension
        >>> length = Dimension('length')
        >>> length
        Dimension(length)
        >>> time = Dimension('time')
        >>> time
        Dimension(time)

    Dimensions can be composed using multiplication, division and
    exponentiation (by a number) to give new dimensions. Addition and
    subtraction is defined only when the two objects are the same dimension.

        >>> velocity = length / time
        >>> velocity
        Dimension(length/time)

    It is possible to use a dimension system object to get the dimensionsal
    dependencies of a dimension, for example the dimension system used by the
    SI units convention can be used:

        >>> from sympy.physics.units.systems.si import dimsys_SI
        >>> dimsys_SI.get_dimensional_dependencies(velocity)
        {Dimension(length, L): 1, Dimension(time, T): -1}
        >>> length + length
        Dimension(length)
        >>> l2 = length**2
        >>> l2
        Dimension(length**2)
        >>> dimsys_SI.get_dimensional_dependencies(l2)
        {Dimension(length, L): 2}

    """

    _op_priority = 13.0

    # XXX: This doesn't seem to be used anywhere...
    _dimensional_dependencies = {}  # type: ignore

    is_commutative = True
    is_number = False
    # make sqrt(M**2) --> M
    is_positive = True
    is_real = True

    def __new__(cls, name, symbol=None):
        # 将名称转换为符号对象，如果是字符串则转换为符号，否则用 sympify 处理
        if isinstance(name, str):
            name = Symbol(name)
        else:
            name = sympify(name)

        # 确保名称是有效的数学表达式
        if not isinstance(name, Expr):
            raise TypeError("Dimension name needs to be a valid math expression")

        # 如果符号是字符串，则转换为符号对象，否则确保是符号对象
        if isinstance(symbol, str):
            symbol = Symbol(symbol)
        elif symbol is not None:
            assert isinstance(symbol, Symbol)

        # 创建新的 Dimension 对象
        obj = Expr.__new__(cls, name)

        # 设置名称和符号属性
        obj._name = name
        obj._symbol = symbol
        return obj

    @property
    def name(self):
        # 返回维度的名称
        return self._name

    @property
    def symbol(self):
        # 返回维度的符号
        return self._symbol

    def __str__(self):
        """
        Display the string representation of the dimension.
        """
        if self.symbol is None:
            return "Dimension(%s)" % (self.name)
        else:
            return "Dimension(%s, %s)" % (self.name, self.symbol)

    def __repr__(self):
        # 返回维度对象的字符串表示形式
        return self.__str__()

    def __neg__(self):
        # 返回维度的负数
        return self
    def __add__(self, other):
        # 导入 Quantity 类
        from sympy.physics.units.quantities import Quantity
        # 将 other 转换为 SymPy 表达式
        other = sympify(other)
        # 如果 other 是 Basic 类型
        if isinstance(other, Basic):
            # 如果 other 包含 Quantity 类型，抛出类型错误
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            # 如果 other 是 Dimension 类型且与 self 相等，则返回 self
            if isinstance(other, Dimension) and self == other:
                return self
            # 否则调用父类的 __add__ 方法进行加法运算
            return super().__add__(other)
        # 如果 other 不是 Basic 类型，则返回 self
        return self

    def __radd__(self, other):
        # 调用 __add__ 方法
        return self.__add__(other)

    def __sub__(self, other):
        # 在维度中没有排序或大小的概念，减法等效于加法当操作合法时
        return self + other

    def __rsub__(self, other):
        # 在维度中没有排序或大小的概念，减法等效于加法当操作合法时
        return self + other

    def __pow__(self, other):
        # 调用 _eval_power 方法
        return self._eval_power(other)

    def _eval_power(self, other):
        # 将 other 转换为 SymPy 表达式
        other = sympify(other)
        # 返回新的 Dimension 对象，self 的 name 的 other 次方
        return Dimension(self.name**other)

    def __mul__(self, other):
        # 导入 Quantity 类
        from sympy.physics.units.quantities import Quantity
        # 如果 other 是 Basic 类型
        if isinstance(other, Basic):
            # 如果 other 包含 Quantity 类型，抛出类型错误
            if other.has(Quantity):
                raise TypeError("cannot sum dimension and quantity")
            # 如果 other 是 Dimension 类型，则返回新的 Dimension 对象，self.name 与 other.name 的乘积
            if isinstance(other, Dimension):
                return Dimension(self.name * other.name)
            # 如果 other 没有自由符号，返回 self
            if not other.free_symbols:  # other.is_number cannot be used
                return self
            # 否则调用父类的 __mul__ 方法进行乘法运算
            return super().__mul__(other)
        # 如果 other 不是 Basic 类型，则返回 self
        return self

    def __rmul__(self, other):
        # 调用 __mul__ 方法
        return self.__mul__(other)

    def __truediv__(self, other):
        # 返回 self 乘以 Pow(other, -1) 的结果
        return self * Pow(other, -1)

    def __rtruediv__(self, other):
        # 返回 other 乘以 self 的 -1 次方 的结果
        return other * pow(self, -1)

    @classmethod
    def _from_dimensional_dependencies(cls, dependencies):
        # 使用 reduce 函数，将 dimensions 和对应的 exponent 依次乘起来，初始值为 1
        return reduce(lambda x, y: x * y, (
            d**e for d, e in dependencies.items()
        ), 1)

    def has_integer_powers(self, dim_sys):
        """
        检查维度对象是否只有整数次幂。

        所有维度的次幂应该是整数，但是在中间步骤可能会出现有理次幂。该方法可用于检查最终结果是否定义良好。
        """
        # 检查所有维度的次幂是否为整数
        return all(dpow.is_Integer for dpow in dim_sys.get_dimensional_dependencies(self).values())
# 根据MKSA基本单位创建维度。
# 对于其他单位系统，可以通过转换基本维度依赖字典来派生它们。

class DimensionSystem(Basic, _QuantityMapper):
    """
    DimensionSystem表示一组连贯的维度。

    构造函数接受三个参数：

    - base dimensions（基本维度）；
    - derived dimensions（派生维度）：这些维度是基于基本维度定义的
      （例如速度是长度除以时间得到的）；
    - dependency of dimensions（维度依赖关系）：派生维度如何依赖于基本维度。

    可选地，派生维度或维度依赖关系可以省略。
    """
    def __new__(cls, base_dims, derived_dims=(), dimensional_dependencies={}):
        # 将传入的 dimensional_dependencies 转换为字典类型
        dimensional_dependencies = dict(dimensional_dependencies)

        def parse_dim(dim):
            # 如果 dim 是字符串，则将其转换为 Symbol 对象并封装成 Dimension 对象
            if isinstance(dim, str):
                dim = Dimension(Symbol(dim))
            # 如果 dim 已经是 Dimension 对象，则不进行任何操作
            elif isinstance(dim, Dimension):
                pass
            # 如果 dim 是 Symbol 对象，则封装成 Dimension 对象
            elif isinstance(dim, Symbol):
                dim = Dimension(dim)
            else:
                # 如果 dim 类型不符合预期，则抛出 TypeError 异常
                raise TypeError("%s wrong type" % dim)
            return dim

        # 对 base_dims 中的每个维度进行解析和处理，确保它们都是 Dimension 对象
        base_dims = [parse_dim(i) for i in base_dims]
        # 对 derived_dims 中的每个维度进行解析和处理，确保它们都是 Dimension 对象
        derived_dims = [parse_dim(i) for i in derived_dims]

        # 检查 base_dims 中是否有重复的维度，如果有则抛出 IndexError 异常
        for dim in base_dims:
            if (dim in dimensional_dependencies
                and (len(dimensional_dependencies[dim]) != 1 or
                     dimensional_dependencies[dim].get(dim, None) != 1)):
                raise IndexError("Repeated value in base dimensions")
            # 将每个 base_dims 的维度添加到 dimensional_dependencies 字典中
            dimensional_dependencies[dim] = Dict({dim: 1})

        def parse_dim_name(dim):
            # 如果 dim 是 Dimension 对象，则直接返回
            if isinstance(dim, Dimension):
                return dim
            # 如果 dim 是字符串，则转换为 Symbol 对象并封装成 Dimension 对象
            elif isinstance(dim, str):
                return Dimension(Symbol(dim))
            # 如果 dim 是 Symbol 对象，则封装成 Dimension 对象
            elif isinstance(dim, Symbol):
                return Dimension(dim)
            else:
                # 如果 dim 类型不符合预期，则抛出 TypeError 异常
                raise TypeError("unrecognized type %s for %s" % (type(dim), dim))

        # 对 dimensional_dependencies 中的每个键进行解析和处理，确保它们都是 Dimension 对象
        for dim in dimensional_dependencies.keys():
            dim = parse_dim(dim)
            # 如果 dim 不在 derived_dims 和 base_dims 中，则将其添加到 derived_dims 中
            if (dim not in derived_dims) and (dim not in base_dims):
                derived_dims.append(dim)

        def parse_dict(d):
            # 将字典 d 中的每个键和值解析为 Dimension 对象和 Dict 对象
            return Dict({parse_dim_name(i): j for i, j in d.items()})

        # 确保 dimensional_dependencies 中的每个键和值都是 SymPy 类型
        dimensional_dependencies = {parse_dim_name(i): parse_dict(j) for i, j in
                                    dimensional_dependencies.items()}

        # 检查 derived_dims 中是否有维度同时存在于 base_dims 中，如果有则抛出 ValueError 异常
        for dim in derived_dims:
            if dim in base_dims:
                raise ValueError("Dimension %s both in base and derived" % dim)
            # 如果 dimensional_dependencies 中没有 dim，则将其添加并设置为 {dim: 1}
            if dim not in dimensional_dependencies:
                dimensional_dependencies[dim] = Dict({dim: 1})

        # 对 base_dims、derived_dims、dimensional_dependencies 进行排序
        base_dims.sort(key=default_sort_key)
        derived_dims.sort(key=default_sort_key)

        # 将 base_dims、derived_dims、dimensional_dependencies 转换为 Tuple 和 Dict 类型，并创建对象
        base_dims = Tuple(*base_dims)
        derived_dims = Tuple(*derived_dims)
        dimensional_dependencies = Dict({i: Dict(j) for i, j in dimensional_dependencies.items()})
        # 使用 Basic 类的 __new__ 方法创建对象并返回
        obj = Basic.__new__(cls, base_dims, derived_dims, dimensional_dependencies)
        return obj

    @property
    def base_dims(self):
        # 返回对象的第一个参数，即 base_dims
        return self.args[0]

    @property
    def derived_dims(self):
        # 返回对象的第二个参数，即 derived_dims
        return self.args[1]

    @property
    def dimensional_dependencies(self):
        # 返回对象的第三个参数，即 dimensional_dependencies
        return self.args[2]
    # 根据给定的维度名称获取其依赖关系字典
    def _get_dimensional_dependencies_for_name(self, dimension):
        # 如果维度是字符串，则转换成符号维度对象
        if isinstance(dimension, str):
            dimension = Dimension(Symbol(dimension))
        # 如果维度不是维度对象，则尝试转换成维度对象
        elif not isinstance(dimension, Dimension):
            dimension = Dimension(dimension)

        # 如果维度的名称是符号对象，则返回其维度依赖字典的浅拷贝
        if dimension.name.is_Symbol:
            # 未包含在依赖关系中的维度视为基本维度
            return dict(self.dimensional_dependencies.get(dimension, {dimension: 1}))

        # 如果维度的名称是数值或数值符号，则返回空字典
        if dimension.name.is_number or dimension.name.is_NumberSymbol:
            return {}

        # 获取递归调用函数的引用
        get_for_name = self._get_dimensional_dependencies_for_name

        # 如果维度的名称是乘积类型，则计算其依赖关系
        if dimension.name.is_Mul:
            ret = collections.defaultdict(int)
            dicts = [get_for_name(i) for i in dimension.name.args]
            for d in dicts:
                for k, v in d.items():
                    ret[k] += v
            # 过滤掉值为零的项，返回非零项组成的字典
            return {k: v for (k, v) in ret.items() if v != 0}

        # 如果维度的名称是加法类型，则检查其依赖关系是否相等
        if dimension.name.is_Add:
            dicts = [get_for_name(i) for i in dimension.name.args]
            if all(d == dicts[0] for d in dicts[1:]):
                return dicts[0]
            else:
                # 抛出类型错误，只能对等价的维度进行加法或减法操作
                raise TypeError("Only equivalent dimensions can be added or subtracted.")

        # 如果维度的名称是幂次类型，则计算其依赖关系
        if dimension.name.is_Pow:
            dim_base = get_for_name(dimension.name.base)
            dim_exp = get_for_name(dimension.name.exp)
            # 如果指数为空字典或为符号，则按照指数调整依赖关系
            if dim_exp == {} or dimension.name.exp.is_Symbol:
                return {k: v * dimension.name.exp for (k, v) in dim_base.items()}
            else:
                # 抛出类型错误，幂运算的指数必须是符号或无量纲的
                raise TypeError("The exponent for the power operator must be a Symbol or dimensionless.")

        # 如果维度的名称是函数类型，则计算其依赖关系
        if dimension.name.is_Function:
            # 获取函数参数对应的维度依赖对象，并计算函数的结果
            args = (Dimension._from_dimensional_dependencies(
                get_for_name(arg)) for arg in dimension.name.args)
            result = dimension.name.func(*args)

            # 获取函数参数对应的维度依赖字典
            dicts = [get_for_name(i) for i in dimension.name.args]

            # 如果函数返回结果是维度对象，则返回其依赖关系字典
            if isinstance(result, Dimension):
                return self.get_dimensional_dependencies(result)
            # 如果函数返回结果与函数名相同，则根据特定条件处理
            elif result.func == dimension.name.func:
                if isinstance(dimension.name, TrigonometricFunction):
                    if dicts[0] in ({}, {Dimension('angle'): 1}):
                        return {}
                    else:
                        # 抛出类型错误，三角函数的输入参数必须是无量纲或角度维度
                        raise TypeError("The input argument for the function {} must be dimensionless or have dimensions of angle.".format(dimension.func))
                else:
                    if all(item == {} for item in dicts):
                        return {}
                    else:
                        # 抛出类型错误，函数的输入参数必须是无量纲
                        raise TypeError("The input arguments for the function {} must be dimensionless.".format(dimension.func))
            else:
                # 返回函数结果对应的维度依赖字典
                return get_for_name(result)

        # 抛出类型错误，未实现该类型的维度获取依赖关系操作
        raise TypeError("Type {} not implemented for get_dimensional_dependencies".format(type(dimension.name)))
    def get_dimensional_dependencies(self, name, mark_dimensionless=False):
        # 获取给定名称的维度依赖关系字典
        dimdep = self._get_dimensional_dependencies_for_name(name)
        # 如果标记为无量纲并且依赖关系为空字典，则返回一个包含单位维度的字典
        if mark_dimensionless and dimdep == {}:
            return {Dimension(1): 1}
        # 否则返回依赖关系字典的副本
        return dict(dimdep.items())

    def equivalent_dims(self, dim1, dim2):
        # 获取两个维度的维度依赖关系字典
        deps1 = self.get_dimensional_dependencies(dim1)
        deps2 = self.get_dimensional_dependencies(dim2)
        # 比较两个维度依赖关系字典是否相等
        return deps1 == deps2

    def extend(self, new_base_dims, new_derived_dims=(), new_dim_deps=None):
        # 复制当前维度系统的维度依赖关系字典
        deps = dict(self.dimensional_dependencies)
        # 如果提供了新的维度依赖关系字典，则更新到复制的字典中
        if new_dim_deps:
            deps.update(new_dim_deps)

        # 创建一个新的维度系统对象，包括新的基础维度和衍生维度，以及更新后的维度依赖关系字典
        new_dim_sys = DimensionSystem(
            tuple(self.base_dims) + tuple(new_base_dims),
            tuple(self.derived_dims) + tuple(new_derived_dims),
            deps
        )
        # 复制当前对象的量和维度映射到新创建的对象中
        new_dim_sys._quantity_dimension_map.update(self._quantity_dimension_map)
        new_dim_sys._quantity_scale_factors.update(self._quantity_scale_factors)
        # 返回新创建的维度系统对象
        return new_dim_sys

    def is_dimensionless(self, dimension):
        """
        检查维度对象是否真的是无量纲。

        一个维度对象至少应该有一个非零幂次分量。
        """
        # 如果维度名称为1，则认为是无量纲
        if dimension.name == 1:
            return True
        # 否则检查维度依赖关系字典是否为空
        return self.get_dimensional_dependencies(dimension) == {}

    @property
    def list_can_dims(self):
        """
        无用的方法，为了与旧版本兼容而保留。

        不要使用。

        列出所有规范维度的名称。
        """
        # 创建一个空集合用于存储规范维度名称
        dimset = set()
        # 遍历基础维度列表，更新规范维度名称集合
        for i in self.base_dims:
            dimset.update(set(self.get_dimensional_dependencies(i).keys()))
        # 返回按照字符串排序的规范维度名称元组
        return tuple(sorted(dimset, key=str))

    @property
    def inv_can_transf_matrix(self):
        """
        无用的方法，为了与旧版本兼容而保留。

        不要使用。

        计算从基础到规范维度基的逆转换矩阵。

        它对应于矩阵，其中列是规范基中的基础维度向量。

        几乎不会使用此矩阵，因为维度始终相对于规范基定义，因此不需要任何工作将它们转换为此基础。
        尽管如此，如果此矩阵不是方阵（或不可逆），则意味着我们选择了一个不好的基础。
        """
        # 使用 reduce 函数和维度对象向量的列表来计算逆转换矩阵
        matrix = reduce(lambda x, y: x.row_join(y),
                        [self.dim_can_vector(d) for d in self.base_dims])
        # 返回计算得到的矩阵
        return matrix
    def can_transf_matrix(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Return the canonical transformation matrix from the canonical to the
        base dimension basis.

        It is the inverse of the matrix computed with inv_can_transf_matrix().
        """

        # 矩阵求逆可能失败，例如系统不一致，例如矩阵不是方阵
        # 返回一个矩阵，该矩阵由按字符串排序的基本维度列表中的每个维度向量组成
        return reduce(lambda x, y: x.row_join(y),
                      [self.dim_can_vector(d) for d in sorted(self.base_dims, key=str)]
                      ).inv()

    def dim_can_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Dimensional representation in terms of the canonical base dimensions.
        """

        vec = []
        # 对于每个规范维度，获取维度依赖关系字典中的维度值，如果没有则默认为0
        for d in self.list_can_dims:
            vec.append(self.get_dimensional_dependencies(dim).get(d, 0))
        return Matrix(vec)

    def dim_vector(self, dim):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Vector representation in terms of the base dimensions.
        """
        # 返回基本维度的向量表示
        return self.can_transf_matrix * Matrix(self.dim_can_vector(dim))

    def print_dim_base(self, dim):
        """
        Give the string expression of a dimension in term of the basis symbols.
        """
        # 获取维度的向量表示
        dims = self.dim_vector(dim)
        # 如果符号存在，则使用符号；否则使用名称
        symbols = [i.symbol if i.symbol is not None else i.name for i in self.base_dims]
        res = S.One
        # 对于每个符号和维度值对，计算其幂次方乘积
        for (s, p) in zip(symbols, dims):
            res *= s**p
        return res

    @property
    def dim(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Give the dimension of the system.

        That is return the number of dimensions forming the basis.
        """
        # 返回基本维度列表的长度，即系统的维度
        return len(self.base_dims)

    @property
    def is_consistent(self):
        """
        Useless method, kept for compatibility with previous versions.

        DO NOT USE.

        Check if the system is well defined.
        """

        # 检查反变换矩阵是否是方阵
        # 如果反变换矩阵不是方阵，则系统不一致
        return self.inv_can_transf_matrix.is_square
```