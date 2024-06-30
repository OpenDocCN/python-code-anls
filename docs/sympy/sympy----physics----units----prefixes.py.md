# `D:\src\scipysrc\sympy\sympy\physics\units\prefixes.py`

```
"""
Module defining unit prefixe class and some constants.

Constant dict for SI and binary prefixes are defined as PREFIXES and
BIN_PREFIXES.
"""
# 从 sympy 库中导入所需模块和类
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.singleton import S

# 定义 Prefix 类，表示单位前缀
class Prefix(Expr):
    """
    This class represent prefixes, with their name, symbol and factor.

    Prefixes are used to create derived units from a given unit. They should
    always be encapsulated into units.

    The factor is constructed from a base (default is 10) to some power, and
    it gives the total multiple or fraction. For example the kilometer km
    is constructed from the meter (factor 1) and the kilo (10 to the power 3,
    i.e. 1000). The base can be changed to allow e.g. binary prefixes.

    A prefix multiplied by something will always return the product of this
    other object times the factor, except if the other object:

    - is a prefix and they can be combined into a new prefix;
    - defines multiplication with prefixes (which is the case for the Unit
      class).
    """
    # 运算优先级设置为 13.0
    _op_priority = 13.0
    # 可交换性为 True
    is_commutative = True

    def __new__(cls, name, abbrev, exponent, base=sympify(10), latex_repr=None):
        # 将输入的参数转化为 sympy 对象
        name = sympify(name)
        abbrev = sympify(abbrev)
        exponent = sympify(exponent)
        base = sympify(base)

        # 创建 Prefix 对象实例
        obj = Expr.__new__(cls, name, abbrev, exponent, base)
        obj._name = name
        obj._abbrev = abbrev
        obj._scale_factor = base**exponent
        obj._exponent = exponent
        obj._base = base
        obj._latex_repr = latex_repr
        return obj

    @property
    def name(self):
        # 返回前缀的名称
        return self._name

    @property
    def abbrev(self):
        # 返回前缀的缩写符号
        return self._abbrev

    @property
    def scale_factor(self):
        # 返回前缀的比例因子
        return self._scale_factor

    def _latex(self, printer):
        # 返回前缀的 LaTeX 表示形式
        if self._latex_repr is None:
            return r'\text{%s}' % self._abbrev
        return self._latex_repr

    @property
    def base(self):
        # 返回前缀的基数
        return self._base

    def __str__(self):
        # 返回前缀的字符串表示形式
        return str(self._abbrev)

    def __repr__(self):
        # 返回前缀的详细表示形式
        if self.base == 10:
            return "Prefix(%r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent)
        else:
            return "Prefix(%r, %r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent, self.base)

    def __mul__(self, other):
        # 导入 Quantity 类，用于单位计算
        from sympy.physics.units import Quantity
        # 如果 other 不是 Quantity 或 Prefix 的实例，则调用父类的 __mul__ 方法
        if not isinstance(other, (Quantity, Prefix)):
            return super().__mul__(other)

        # 计算两者的乘积因子
        fact = self.scale_factor * other.scale_factor

        # 如果 other 是 Prefix 类的实例
        if isinstance(other, Prefix):
            # 如果乘积因子为 1，则返回 S.One
            if fact == 1:
                return S.One
            # 简化前缀
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        # 返回两者的乘积因子
        return self.scale_factor * other
    # 定义自定义类的真除运算符重载方法，用于处理除法操作
    def __truediv__(self, other):
        # 检查被除数是否具有 "scale_factor" 属性
        if not hasattr(other, "scale_factor"):
            # 如果没有 "scale_factor" 属性，调用父类的真除运算符方法
            return super().__truediv__(other)

        # 计算自身对象的缩放因子与被除数对象的缩放因子的比值
        fact = self.scale_factor / other.scale_factor

        # 如果比值等于1，返回符号“1”
        if fact == 1:
            return S.One
        # 如果被除数对象是 Prefix 类的实例
        elif isinstance(other, Prefix):
            # 遍历 PREFIXES 字典中的前缀
            for p in PREFIXES:
                # 如果某个前缀的缩放因子与比值相等，返回该前缀对象
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            # 如果没有找到匹配的前缀，返回比值本身
            return fact

        # 默认情况下，返回自身对象的缩放因子与被除数的比值
        return self.scale_factor / other

    # 定义自定义类的反向真除运算符重载方法
    def __rtruediv__(self, other):
        # 如果被除数是符号“1”
        if other == 1:
            # 遍历 PREFIXES 字典中的前缀
            for p in PREFIXES:
                # 如果某个前缀的缩放因子等于被除数对象的缩放因子的倒数，返回该前缀对象
                if PREFIXES[p].scale_factor == 1 / self.scale_factor:
                    return PREFIXES[p]
        # 返回被除数与自身对象的缩放因子的比值
        return other / self.scale_factor
# 定义一个函数，用于生成所有通过给定前缀组合形成的单位列表
def prefix_unit(unit, prefixes):
    """
    Return a list of all units formed by unit and the given prefixes.

    You can use the predefined PREFIXES or BIN_PREFIXES, but you can also
    pass as argument a subdict of them if you do not want all prefixed units.

        >>> from sympy.physics.units.prefixes import (PREFIXES,
        ...                                                 prefix_unit)
        >>> from sympy.physics.units import m
        >>> pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}
        >>> prefix_unit(m, pref)  # doctest: +SKIP
        [millimeter, centimeter, decimeter]
    """

    # 导入需要的类和模块
    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units import UnitSystem

    # 初始化一个空列表来存储带前缀的单位
    prefixed_units = []

    # 遍历传入的前缀字典中的每一个前缀
    for prefix in prefixes.values():
        # 创建一个 Quantity 对象，用于表示带有特定前缀的单位
        quantity = Quantity(
                "%s%s" % (prefix.name, unit.name),  # 使用前缀和单位名称生成全名
                abbrev=("%s%s" % (prefix.abbrev, unit.abbrev)),  # 使用前缀和单位缩写生成缩写
                is_prefixed=True,  # 标记此单位是带有前缀的
           )
        # 将新创建的 Quantity 对象与其对应的基本单位建立映射关系
        UnitSystem._quantity_dimensional_equivalence_map_global[quantity] = unit
        # 存储前缀和单位之间的比例因子
        UnitSystem._quantity_scale_factors_global[quantity] = (prefix.scale_factor, unit)
        # 将新创建的 Quantity 对象添加到列表中
        prefixed_units.append(quantity)

    # 返回包含所有带前缀单位的列表
    return prefixed_units


# 定义一系列 SI 单位的国际标准前缀，每个前缀都与一个 Prefix 对象相关联
yotta = Prefix('yotta', 'Y', 24)
zetta = Prefix('zetta', 'Z', 21)
exa = Prefix('exa', 'E', 18)
peta = Prefix('peta', 'P', 15)
tera = Prefix('tera', 'T', 12)
giga = Prefix('giga', 'G', 9)
mega = Prefix('mega', 'M', 6)
kilo = Prefix('kilo', 'k', 3)
hecto = Prefix('hecto', 'h', 2)
deca = Prefix('deca', 'da', 1)
deci = Prefix('deci', 'd', -1)
centi = Prefix('centi', 'c', -2)
milli = Prefix('milli', 'm', -3)
micro = Prefix('micro', 'mu', -6, latex_repr=r"\mu")
nano = Prefix('nano', 'n', -9)
pico = Prefix('pico', 'p', -12)
femto = Prefix('femto', 'f', -15)
atto = Prefix('atto', 'a', -18)
zepto = Prefix('zepto', 'z', -21)
yocto = Prefix('yocto', 'y', -24)

# 定义一个字典，将 SI 单位的符号映射到对应的 Prefix 对象
PREFIXES = {
    'Y': yotta,
    'Z': zetta,
    'E': exa,
    'P': peta,
    'T': tera,
    'G': giga,
    'M': mega,
    'k': kilo,
    'h': hecto,
    'da': deca,
    'd': deci,
    'c': centi,
    'm': milli,
    'mu': micro,
    'n': nano,
    'p': pico,
    'f': femto,
    'a': atto,
    'z': zepto,
    'y': yocto,
}

# 定义一系列二进制单位的前缀，每个前缀都与一个 Prefix 对象相关联
kibi = Prefix('kibi', 'Y', 10, 2)
mebi = Prefix('mebi', 'Y', 20, 2)
gibi = Prefix('gibi', 'Y', 30, 2)
tebi = Prefix('tebi', 'Y', 40, 2)
pebi = Prefix('pebi', 'Y', 50, 2)
exbi = Prefix('exbi', 'Y', 60, 2)

# 定义一个字典，将二进制单位的缩写映射到对应的 Prefix 对象
BIN_PREFIXES = {
    'Ki': kibi,
    'Mi': mebi,
    'Gi': gibi,
    'Ti': tebi,
    'Pi': pebi,
    'Ei': exbi,
}
```