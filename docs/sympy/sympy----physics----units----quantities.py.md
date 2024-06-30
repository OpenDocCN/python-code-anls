# `D:\src\scipysrc\sympy\sympy\physics\units\quantities.py`

```
"""
Physical quantities.
"""

# 导入所需模块和类
from sympy.core.expr import AtomicExpr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.prefixes import Prefix

# 定义表示物理量的类
class Quantity(AtomicExpr):
    """
    Physical quantity: can be a unit of measure, a constant or a generic quantity.
    """

    is_commutative = True  # 表示这个物理量对象是可交换的
    is_real = True  # 表示这个物理量对象是实数
    is_number = False  # 表示这个物理量对象不是一个数字
    is_nonzero = True  # 表示这个物理量对象是非零的
    is_physical_constant = False  # 表示这个物理量对象不是一个物理常数
    _diff_wrt = True  # 表示这个物理量对象可以用于微分

    # 构造函数，用于初始化物理量对象
    def __new__(cls, name, abbrev=None,
                latex_repr=None, pretty_unicode_repr=None,
                pretty_ascii_repr=None, mathml_presentation_repr=None,
                is_prefixed=False,
                **assumptions):

        # 如果name不是Symbol类型，则转换为Symbol类型
        if not isinstance(name, Symbol):
            name = Symbol(name)

        # 如果abbrev为None，则设置为name；如果abbrev是字符串，则转换为Symbol类型
        if abbrev is None:
            abbrev = name
        elif isinstance(abbrev, str):
            abbrev = Symbol(abbrev)

        # 用于类型检查的Hack，实际赋值在下面进行
        cls._is_prefixed = is_prefixed

        # 创建物理量对象
        obj = AtomicExpr.__new__(cls, name, abbrev)
        obj._name = name
        obj._abbrev = abbrev
        obj._latex_repr = latex_repr
        obj._unicode_repr = pretty_unicode_repr
        obj._ascii_repr = pretty_ascii_repr
        obj._mathml_repr = mathml_presentation_repr
        obj._is_prefixed = is_prefixed
        return obj

    # 设置全局维度
    def set_global_dimension(self, dimension):
        _QuantityMapper._quantity_dimension_global[self] = dimension

    # 设置全局相对比例因子和参考物理量
    def set_global_relative_scale_factor(self, scale_factor, reference_quantity):
        """
        Setting a scale factor that is valid across all unit system.
        """
        from sympy.physics.units import UnitSystem
        scale_factor = sympify(scale_factor)  # 将比例因子转换为SymPy表达式
        if isinstance(scale_factor, Prefix):
            self._is_prefixed = True  # 如果比例因子是一个前缀，则标记为已前缀化
        # 将所有前缀替换为它们相对于标准单位的比率：
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        scale_factor = sympify(scale_factor)  # 再次确保比例因子是SymPy表达式
        # 设置全局比例因子和参考物理量的映射关系
        UnitSystem._quantity_scale_factors_global[self] = (scale_factor, reference_quantity)
        UnitSystem._quantity_dimensional_equivalence_map_global[self] = reference_quantity

    # 获取物理量的名称
    @property
    def name(self):
        return self._name

    # 获取物理量的维度
    @property
    def dimension(self):
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_dimension(self)

    # 获取物理量的缩写
    @property
    def abbrev(self):
        """
        Symbol representing the unit name.

        Prepend the abbreviation with the prefix symbol if it is defines.
        """
        return self._abbrev

    # 此处省略了部分代码
    # 计算量的比例因子，相对于标准单位的总体大小
    def scale_factor(self):
        # 导入单位系统模块
        from sympy.physics.units import UnitSystem
        # 获取默认的单位系统
        unit_system = UnitSystem.get_default_unit_system()
        # 返回该量在单位系统中的比例因子
        return unit_system.get_quantity_scale_factor(self)

    # 总是返回 True，表示该量永远是正数
    def _eval_is_positive(self):
        return True

    # 总是返回 True，表示该量永远是常数
    def _eval_is_constant(self):
        return True

    # 返回自身，表示绝对值运算下的结果是原始量
    def _eval_Abs(self):
        return self

    # 替换量中的旧值为新值，如果新值是 Quantity 类型并且不等于旧值，则返回自身
    def _eval_subs(self, old, new):
        if isinstance(new, Quantity) and self != old:
            return self

    # 将量转换为 LaTeX 表示形式，如果已有 LaTeX 表示，则返回该表示，否则生成默认的文本表示
    def _latex(self, printer):
        if self._latex_repr:
            return self._latex_repr
        else:
            return r'\text{{{}}}'.format(self.args[1] \
                          if len(self.args) >= 2 else self.args[0])

    # 将量转换为相同维度的另一个量
    def convert_to(self, other, unit_system="SI"):
        """
        Examples
        ========
        """
        # 导入单位转换工具函数
        from .util import convert_to
        # 调用转换函数，返回转换后的结果
        return convert_to(self, other, unit_system)

    # 返回量的自由符号集合为空集，表示该量没有自由符号
    @property
    def free_symbols(self):
        """Return free symbols from quantity."""
        return set()

    # 返回量是否带有单位前缀，例如千克是带有前缀的，而克则没有
    @property
    def is_prefixed(self):
        """Whether or not the quantity is prefixed. Eg. `kilogram` is prefixed, but `gram` is not."""
        return self._is_prefixed
# 定义一个物理常数类，继承自Quantity类
class PhysicalConstant(Quantity):
    """Represents a physical constant, eg. `speed_of_light` or `avogadro_constant`."""

    # 声明这是一个物理常数类的标志
    is_physical_constant = True
```