# `D:\src\scipysrc\sympy\sympy\physics\units\unitsystem.py`

```
"""
Unit system for physical quantities; include definition of constants.
"""

from typing import Dict as tDict, Set as tSet  # 导入类型提示

from sympy.core.add import Add  # 导入加法类
from sympy.core.function import (Derivative, Function)  # 导入导数和函数类
from sympy.core.mul import Mul  # 导入乘法类
from sympy.core.power import Pow  # 导入幂运算类
from sympy.core.singleton import S  # 导入单例类
from sympy.physics.units.dimensions import _QuantityMapper  # 导入物理量映射类
from sympy.physics.units.quantities import Quantity  # 导入物理量类

from .dimensions import Dimension  # 导入自定义的维度类


class UnitSystem(_QuantityMapper):
    """
    UnitSystem represents a coherent set of units.

    A unit system is basically a dimension system with notions of scales. Many
    of the methods are defined in the same way.

    It is much better if all base units have a symbol.
    """

    _unit_systems = {}  # 存储已创建的单位系统对象的字典，键为系统名称，值为对应的 UnitSystem 实例

    def __init__(self, base_units, units=(), name="", descr="", dimension_system=None, derived_units: tDict[Dimension, Quantity]={}):
        """
        Initialize a UnitSystem object.

        Parameters:
        - base_units: Tuple of base units for the unit system.
        - units: Additional units beyond base units.
        - name: Name of the unit system.
        - descr: Description of the unit system.
        - dimension_system: Associated dimension system.
        - derived_units: Dictionary mapping derived units to quantities.
        """

        UnitSystem._unit_systems[name] = self  # 将当前实例存储到类变量 _unit_systems 中

        self.name = name  # 设置单位系统的名称
        self.descr = descr  # 设置单位系统的描述

        self._base_units = base_units  # 设置基本单位
        self._dimension_system = dimension_system  # 设置关联的维度系统
        self._units = tuple(set(base_units) | set(units))  # 合并基本单位和附加单位，去重后转为元组
        self._base_units = tuple(base_units)  # 将基本单位转为元组
        self._derived_units = derived_units  # 设置派生单位的字典，映射维度到数量

        super().__init__()  # 调用父类 _QuantityMapper 的初始化方法

    def __str__(self):
        """
        Return the name of the system.

        If it does not exist, then it makes a list of symbols (or names) of
        the base dimensions.
        """

        if self.name != "":
            return self.name  # 返回单位系统的名称
        else:
            return "UnitSystem((%s))" % ", ".join(
                str(d) for d in self._base_units)  # 返回格式化的基本单位列表字符串

    def __repr__(self):
        return '<UnitSystem: %s>' % repr(self._base_units)  # 返回表示单位系统的字符串

    def extend(self, base, units=(), name="", description="", dimension_system=None, derived_units: tDict[Dimension, Quantity]={}):
        """Extend the current system into a new one.

        Take the base and normal units of the current system to merge
        them to the base and normal units given in argument.
        If not provided, name and description are overridden by empty strings.
        """

        base = self._base_units + tuple(base)  # 合并当前系统的基本单位和提供的新基本单位
        units = self._units + tuple(units)  # 合并当前系统的单位和提供的新单位

        return UnitSystem(base, units, name, description, dimension_system, {**self._derived_units, **derived_units})  # 返回新创建的扩展单位系统实例

    def get_dimension_system(self):
        return self._dimension_system  # 返回关联的维度系统

    def get_quantity_dimension(self, unit):
        qdm = self.get_dimension_system()._quantity_dimension_map  # 获取维度系统中的物理量维度映射
        if unit in qdm:
            return qdm[unit]  # 返回单位对应的维度
        return super().get_quantity_dimension(unit)  # 调用父类方法获取单位对应的维度

    def get_quantity_scale_factor(self, unit):
        qsfm = self.get_dimension_system()._quantity_scale_factors  # 获取维度系统中的物理量比例因子映射
        if unit in qsfm:
            return qsfm[unit]  # 返回单位对应的比例因子
        return super().get_quantity_scale_factor(unit)  # 调用父类方法获取单位对应的比例因子

    @staticmethod
    # 返回给定单位制系统的实例，如果已经是UnitSystem的实例，则直接返回
    def get_unit_system(unit_system):
        if isinstance(unit_system, UnitSystem):
            return unit_system

        # 如果unit_system不在已知的单位制系统列表中，则抛出数值错误
        if unit_system not in UnitSystem._unit_systems:
            raise ValueError(
                "Unit system is not supported. Currently"
                "supported unit systems are {}".format(
                    ", ".join(sorted(UnitSystem._unit_systems))
                )
            )

        # 返回对应名称的单位制系统
        return UnitSystem._unit_systems[unit_system]

    @staticmethod
    # 返回默认的单位制系统，这里默认返回国际单位制（SI）的单位制系统
    def get_default_unit_system():
        return UnitSystem._unit_systems["SI"]

    @property
    # 返回系统的维数，即基本单位的数量
    def dim(self):
        """
        Give the dimension of the system.

        That is return the number of units forming the basis.
        """
        return len(self._base_units)

    @property
    # 检查底层维度系统是否一致
    def is_consistent(self):
        """
        Check if the underlying dimension system is consistent.
        """
        # 测试由DimensionSystem执行
        return self.get_dimension_system().is_consistent

    @property
    # 返回派生单位的字典，键为维度，值为对应的Quantity对象
    def derived_units(self) -> tDict[Dimension, Quantity]:
        return self._derived_units

    # 返回表达式的维度
    def get_dimensional_expr(self, expr):
        from sympy.physics.units import Quantity
        if isinstance(expr, Mul):
            return Mul(*[self.get_dimensional_expr(i) for i in expr.args])
        elif isinstance(expr, Pow):
            return self.get_dimensional_expr(expr.base) ** expr.exp
        elif isinstance(expr, Add):
            return self.get_dimensional_expr(expr.args[0])
        elif isinstance(expr, Derivative):
            dim = self.get_dimensional_expr(expr.expr)
            for independent, count in expr.variable_count:
                dim /= self.get_dimensional_expr(independent)**count
            return dim
        elif isinstance(expr, Function):
            args = [self.get_dimensional_expr(arg) for arg in expr.args]
            if all(i == 1 for i in args):
                return S.One
            return expr.func(*args)
        elif isinstance(expr, Quantity):
            return self.get_quantity_dimension(expr).name
        return S.One
    # 定义一个方法，用于从表达式中提取比例因子和维度表达式的元组
    def _collect_factor_and_dimension(self, expr):
        """
        Return tuple with scale factor expression and dimension expression.
        """
        # 导入量纲单位 Quantity 类
        from sympy.physics.units import Quantity
        
        # 如果表达式是 Quantity 类型，则返回其比例因子和维度
        if isinstance(expr, Quantity):
            return expr.scale_factor, expr.dimension
        
        # 如果表达式是乘法 Mul 类型
        elif isinstance(expr, Mul):
            factor = 1  # 初始比例因子为 1
            dimension = Dimension(1)  # 初始维度为单位维度
            # 遍历乘法表达式中的每个参数
            for arg in expr.args:
                # 递归调用 _collect_factor_and_dimension 方法获取每个参数的比例因子和维度
                arg_factor, arg_dim = self._collect_factor_and_dimension(arg)
                factor *= arg_factor  # 计算总的比例因子
                dimension *= arg_dim  # 计算总的维度
            return factor, dimension
        
        # 如果表达式是指数 Pow 类型
        elif isinstance(expr, Pow):
            # 获取基数和指数的比例因子和维度
            base_factor, base_dim = self._collect_factor_and_dimension(expr.base)
            exp_factor, exp_dim = self._collect_factor_and_dimension(expr.exp)
            # 如果指数维度是无量纲的，则将其设置为 1
            if self.get_dimension_system().is_dimensionless(exp_dim):
                exp_dim = 1
            return base_factor ** exp_factor, base_dim ** (exp_factor * exp_dim)
        
        # 如果表达式是加法 Add 类型
        elif isinstance(expr, Add):
            # 获取第一个加数的比例因子和维度
            factor, dim = self._collect_factor_and_dimension(expr.args[0])
            # 遍历剩余的加数
            for addend in expr.args[1:]:
                # 获取每个加数的比例因子和维度
                addend_factor, addend_dim = self._collect_factor_and_dimension(addend)
                # 检查每个加数的维度是否与第一个加数的维度等效
                if not self.get_dimension_system().equivalent_dims(dim, addend_dim):
                    raise ValueError(
                        'Dimension of "{}" is {}, '
                        'but it should be {}'.format(
                            addend, addend_dim, dim))
                factor += addend_factor  # 累加每个加数的比例因子
            return factor, dim
        
        # 如果表达式是导数 Derivative 类型
        elif isinstance(expr, Derivative):
            # 获取导数表达式的比例因子和维度
            factor, dim = self._collect_factor_and_dimension(expr.args[0])
            # 遍历导数中的每个自变量和其数量
            for independent, count in expr.variable_count:
                # 获取每个自变量的比例因子和维度
                ifactor, idim = self._collect_factor_and_dimension(independent)
                factor /= ifactor ** count  # 计算每个自变量的总比例因子
                dim /= idim ** count  # 计算每个自变量的总维度
            return factor, dim
        
        # 如果表达式是函数 Function 类型
        elif isinstance(expr, Function):
            # 对函数中的每个参数递归调用 _collect_factor_and_dimension 方法
            fds = [self._collect_factor_and_dimension(arg) for arg in expr.args]
            # 如果参数的维度是无量纲的，则设置为单位维度
            dims = [Dimension(1) if self.get_dimension_system().is_dimensionless(d[1]) else d[1] for d in fds]
            # 返回函数结果及其参数的维度
            return (expr.func(*(f[0] for f in fds)), *dims)
        
        # 如果表达式是量纲 Dimension 类型
        elif isinstance(expr, Dimension):
            return S.One, expr  # 返回单位比例因子和表达式的维度
        
        else:
            return expr, Dimension(1)  # 其它情况，返回表达式及单位维度

    # 获取系统中没有前缀的单位集合的方法
    def get_units_non_prefixed(self) -> tSet[Quantity]:
        """
        Return the units of the system that do not have a prefix.
        """
        # 过滤出系统中没有前缀且不是物理常数的单位，返回集合
        return set(filter(lambda u: not u.is_prefixed and not u.is_physical_constant, self._units))
```