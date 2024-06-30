# `D:\src\scipysrc\sympy\sympy\physics\units\util.py`

```
"""
Several methods to simplify expressions involving unit objects.
"""
# 导入必要的模块和函数
from functools import reduce
from collections.abc import Iterable
from typing import Optional

# 导入 SymPy 相关模块和类
from sympy import default_sort_key
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.core.function import Function
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.physics.units.dimensions import Dimension, DimensionSystem
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.utilities.iterables import sift


def _get_conversion_matrix_for_expr(expr, target_units, unit_system):
    # 导入 Matrix 类
    from sympy.matrices.dense import Matrix

    # 获取单位制度对象
    dimension_system = unit_system.get_dimension_system()

    # 获取表达式的维度对象
    expr_dim = Dimension(unit_system.get_dimensional_expr(expr))

    # 获取表达式的维度依赖关系
    dim_dependencies = dimension_system.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)

    # 获取目标单位的维度对象列表
    target_dims = [Dimension(unit_system.get_dimensional_expr(x)) for x in target_units]

    # 获取目标单位的规范维度依赖关系列表
    canon_dim_units = [i for x in target_dims for i in dimension_system.get_dimensional_dependencies(x, mark_dimensionless=True)]

    # 获取表达式的规范维度依赖关系集合
    canon_expr_units = set(dim_dependencies)

    # 如果表达式的规范维度依赖关系不是目标单位的规范维度依赖关系的子集，则返回 None
    if not canon_expr_units.issubset(set(canon_dim_units)):
        return None

    # 去重目标单位的规范维度依赖关系列表
    seen = set()
    canon_dim_units = [i for i in canon_dim_units if not (i in seen or seen.add(i))]

    # 创建转换矩阵 camat
    camat = Matrix([[dimension_system.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims] for j in canon_dim_units])

    # 创建表达式的矩阵 exprmat
    exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])

    # 尝试解方程 camat * res_exponents = exprmat
    try:
        res_exponents = camat.solve(exprmat)
    except NonInvertibleMatrixError:
        return None

    # 返回结果指数 res_exponents
    return res_exponents


def convert_to(expr, target_units, unit_system="SI"):
    """
    Convert ``expr`` to the same expression with all of its units and quantities
    represented as factors of ``target_units``, whenever the dimension is compatible.

    ``target_units`` may be a single unit/quantity, or a collection of
    units/quantities.

    Examples
    ========

    >>> from sympy.physics.units import speed_of_light, meter, gram, second, day
    >>> from sympy.physics.units import mile, newton, kilogram, atomic_mass_constant
    >>> from sympy.physics.units import kilometer, centimeter
    >>> from sympy.physics.units import gravitational_constant, hbar
    >>> from sympy.physics.units import convert_to
    >>> convert_to(mile, kilometer)
    25146*kilometer/15625
    >>> convert_to(mile, kilometer).n()
    1.609344*kilometer
    >>> convert_to(speed_of_light, meter/second)
    299792458*meter/second
    >>> convert_to(day, second)
    86400*second
    >>> 3*newton
    3*newton
    >>> convert_to(3*newton, kilogram*meter/second**2)
    """
    # 函数说明：将表达式 ``expr`` 转换为其单位和量的因子为 ``target_units`` 的表达式，如果维度兼容的话
    pass
    3*kilogram*meter/second**2
    >>> convert_to(atomic_mass_constant, gram)
    1.660539060e-24*gram
    
    Conversion to multiple units:
    
    >>> convert_to(speed_of_light, [meter, second])
    299792458*meter/second
    >>> convert_to(3*newton, [centimeter, gram, second])
    300000*centimeter*gram/second**2
    
    Conversion to Planck units:
    
    >>> convert_to(atomic_mass_constant, [gravitational_constant, speed_of_light, hbar]).n()
    7.62963087839509e-20*hbar**0.5*speed_of_light**0.5/gravitational_constant**0.5
    
    """
    # 导入单位系统相关的模块
    from sympy.physics.units import UnitSystem
    # 获取单位系统对象
    unit_system = UnitSystem.get_unit_system(unit_system)
    
    # 如果目标单位不是可迭代对象，则转换为列表
    if not isinstance(target_units, (Iterable, Tuple)):
        target_units = [target_units]
    
    # 处理加法表达式的函数
    def handle_Adds(expr):
        # 使用生成器表达式转换每个加法项到目标单位
        return Add.fromiter(convert_to(i, target_units, unit_system)
            for i in expr.args)
    
    # 如果表达式是加法类型，则调用处理加法函数
    if isinstance(expr, Add):
        return handle_Adds(expr)
    # 如果表达式是幂运算且基数是加法类型，则调用处理加法函数后进行幂运算
    elif isinstance(expr, Pow) and isinstance(expr.base, Add):
        return handle_Adds(expr.base) ** expr.exp
    
    # 将表达式和目标单位符号化（sympify）
    expr = sympify(expr)
    target_units = sympify(target_units)
    
    # 如果表达式是函数类型，则将其整理（together）
    if isinstance(expr, Function):
        expr = expr.together()
    
    # 如果表达式不是 Quantity 类型但包含 Quantity 类型，则转换其单位到目标单位
    if not isinstance(expr, Quantity) and expr.has(Quantity):
        expr = expr.replace(lambda x: isinstance(x, Quantity),
            lambda x: x.convert_to(target_units, unit_system))
    
    # 获取表达式的总比例因子
    def get_total_scale_factor(expr):
        if isinstance(expr, Mul):
            return reduce(lambda x, y: x * y,
                [get_total_scale_factor(i) for i in expr.args])
        elif isinstance(expr, Pow):
            return get_total_scale_factor(expr.base) ** expr.exp
        elif isinstance(expr, Quantity):
            return unit_system.get_quantity_scale_factor(expr)
        return expr
    
    # 获取表达式到目标单位的转换矩阵
    depmat = _get_conversion_matrix_for_expr(expr, target_units, unit_system)
    if depmat is None:
        return expr
    
    # 计算表达式的比例因子
    expr_scale_factor = get_total_scale_factor(expr)
    # 返回计算后的结果
    return expr_scale_factor * Mul.fromiter(
        (1/get_total_scale_factor(u)*u)**p for u, p in
        zip(target_units, depmat))
# 根据给定表达式简化量纲单位，替换带有前缀的数值，并默认以规范的方式统一给定维度的所有单位。
# `across_dimensions` 参数允许将不同维度的单位一起简化。
# 如果设置了 `across_dimensions` 参数为 True，则必须指定 `unit_system`。

def quantity_simplify(expr, across_dimensions: bool=False, unit_system=None):
    """Return an equivalent expression in which prefixes are replaced
    with numerical values and all units of a given dimension are the
    unified in a canonical manner by default. `across_dimensions` allows
    for units of different dimensions to be simplified together.

    `unit_system` must be specified if `across_dimensions` is True.

    Examples
    ========

    >>> from sympy.physics.units.util import quantity_simplify
    >>> from sympy.physics.units.prefixes import kilo
    >>> from sympy.physics.units import foot, inch, joule, coulomb
    >>> quantity_simplify(kilo*foot*inch)
    250*foot**2/3
    >>> quantity_simplify(foot - 6*inch)
    foot/2
    >>> quantity_simplify(5*joule/coulomb, across_dimensions=True, unit_system="SI")
    5*volt
    """

    if expr.is_Atom or not expr.has(Prefix, Quantity):
        return expr

    # 将表达式中的所有前缀替换为其数值
    p = expr.atoms(Prefix)
    expr = expr.xreplace({p: p.scale_factor for p in p})

    # 将给定维度的所有量纲替换为一个规范的量纲，从表达式中选择一个作为参考量纲
    d = sift(expr.atoms(Quantity), lambda i: i.dimension)
    for k in d:
        if len(d[k]) == 1:
            continue
        v = list(ordered(d[k]))
        ref = v[0]/v[0].scale_factor
        expr = expr.xreplace({vi: ref*vi.scale_factor for vi in v[1:]})

    if across_dimensions:
        # 将不同维度的量纲单位合并为一个等效于原始表达式的单一量纲单位

        if unit_system is None:
            raise ValueError("unit_system must be specified if across_dimensions is True")

        unit_system = UnitSystem.get_unit_system(unit_system)
        dimension_system: DimensionSystem = unit_system.get_dimension_system()
        dim_expr = unit_system.get_dimensional_expr(expr)
        dim_deps = dimension_system.get_dimensional_dependencies(dim_expr, mark_dimensionless=True)

        target_dimension: Optional[Dimension] = None
        for ds_dim, ds_dim_deps in dimension_system.dimensional_dependencies.items():
            if ds_dim_deps == dim_deps:
                target_dimension = ds_dim
                break

        if target_dimension is None:
            # 如果找不到目标量纲，无法进行处理，不确定如何处理此情况。
            return expr

        target_unit = unit_system.derived_units.get(target_dimension)
        if target_unit:
            expr = convert_to(expr, target_unit, unit_system)

    return expr


def check_dimensions(expr, unit_system="SI"):
    """Return expr if units in addends have the same
    base dimensions, else raise a ValueError."""
    # 对于 SymPy 核心程序来说，忽略将数字加到带有量纲的量中的情况。
    # 因此，如果发现这样的加数，该函数将引发错误。
    # 引入单位系统模块
    from sympy.physics.units import UnitSystem
    # 获取当前的单位系统
    unit_system = UnitSystem.get_unit_system(unit_system)

    # 定义一个函数，用于合并两个字典，将相同键的值相加，并移除值为0的键
    def addDict(dict1, dict2):
        """Merge dictionaries by adding values of common keys and
        removing keys with value of 0."""
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
            # 如果键同时存在于dict1和dict2中，则将对应的值相加
            if key in dict1 and key in dict2:
                   dict3[key] = value + dict1[key]
        # 返回合并后的字典，移除值为0的键
        return {key:val for key, val in dict3.items() if val != 0}

    # 获取表达式中所有的加法操作
    adds = expr.atoms(Add)
    # 获取维度系统的维度依赖关系
    DIM_OF = unit_system.get_dimension_system().get_dimensional_dependencies

    # 遍历每一个加法操作
    for a in adds:
        # 存放维度集合的集合
        deset = set()
        # 遍历加法操作的每一个参数
        for ai in a.args:
            # 如果参数是数值，则将空元组加入维度集合，然后继续下一个参数
            if ai.is_number:
                deset.add(())
                continue
            dims = []
            skip = False
            dimdict = {}
            # 将参数按乘法分解为因子，并处理每一个因子
            for i in Mul.make_args(ai):
                # 如果因子包含Quantity，则获取其维度表达式并转换为Dimension对象
                if i.has(Quantity):
                    i = Dimension(unit_system.get_dimensional_expr(i))
                # 如果因子包含Dimension，则将其维度依赖添加到dimdict中
                if i.has(Dimension):
                    dimdict = addDict(dimdict, DIM_OF(i))
                # 如果因子包含自由符号，则跳过当前参数的处理
                elif i.free_symbols:
                    skip = True
                    break
            # 将处理后的维度依赖添加到dims列表中
            dims.extend(dimdict.items())
            # 如果未跳过当前参数，则将排序后的维度依赖元组添加到维度集合中
            if not skip:
                deset.add(tuple(sorted(dims, key=default_sort_key)))
                # 如果维度集合的大小大于1，则抛出维度不兼容的错误
                if len(deset) > 1:
                    raise ValueError(
                        "addends have incompatible dimensions: {}".format(deset))

    # 清除表达式中留下的乘法常数，这些常数可能是替换后留下的
    reps = {}
    # 遍历表达式中所有的乘法操作
    for m in expr.atoms(Mul):
        # 如果乘法操作的参数中包含Dimension，则将其添加到替换字典中
        if any(isinstance(i, Dimension) for i in m.args):
            reps[m] = m.func(*[
                i for i in m.args if not i.is_number])

    # 使用替换字典替换表达式中的乘法常数，并返回结果
    return expr.xreplace(reps)
```