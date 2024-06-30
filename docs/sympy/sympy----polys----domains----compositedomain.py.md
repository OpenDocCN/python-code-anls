# `D:\src\scipysrc\sympy\sympy\polys\domains\compositedomain.py`

```
"""
Implementation of :class:`CompositeDomain` class.
"""

# 导入所需模块和类
from sympy.polys.domains.domain import Domain
from sympy.polys.polyerrors import GeneratorsError
from sympy.utilities import public

# 声明一个公共类，继承自 Domain 类
@public
class CompositeDomain(Domain):
    """
    Base class for composite domains, e.g. ZZ[x], ZZ(X).
    """

    # 设置类属性
    is_Composite = True
    gens, ngens, symbols, domain = [None]*4

    def inject(self, *symbols):
        """
        Inject generators into this domain.
        """
        # 检查要注入的符号是否与已有符号有重复
        if not (set(self.symbols) & set(symbols)):
            # 如果没有重复，创建一个新的 CompositeDomain 实例并返回
            return self.__class__(self.domain, self.symbols + symbols, self.order)
        else:
            # 如果有重复，则抛出 GeneratorsError 异常
            raise GeneratorsError("common generators in %s and %s" % (self.symbols, symbols))

    def drop(self, *symbols):
        """
        Drop generators from this domain.
        """
        # 将要删除的符号转换为集合
        symset = set(symbols)
        # 过滤掉不在要删除符号集合中的符号，生成新的符号元组
        newsyms = tuple(s for s in self.symbols if s not in symset)
        # 调用 domain 对象的 drop 方法，返回新的 domain 对象
        domain = self.domain.drop(*symbols)
        # 如果新符号列表为空，则直接返回新的 domain 对象
        if not newsyms:
            return domain
        else:
            # 否则创建一个新的 CompositeDomain 实例并返回
            return self.__class__(domain, newsyms, self.order)

    def set_domain(self, domain):
        """
        Set the ground domain of this domain.
        """
        # 创建一个新的 CompositeDomain 实例，替换 domain 属性并返回
        return self.__class__(domain, self.symbols, self.order)

    @property
    def is_Exact(self):
        """
        Returns ``True`` if this domain is exact.
        """
        # 返回 domain 对象的 is_Exact 属性
        return self.domain.is_Exact

    def get_exact(self):
        """
        Returns an exact version of this domain.
        """
        # 调用 domain 对象的 get_exact 方法，返回一个精确版本的 domain 对象
        return self.set_domain(self.domain.get_exact())

    @property
    def has_CharacteristicZero(self):
        """
        Returns ``True`` if this domain has characteristic zero.
        """
        # 返回 domain 对象的 has_CharacteristicZero 属性
        return self.domain.has_CharacteristicZero

    def characteristic(self):
        """
        Returns the characteristic of this domain.
        """
        # 调用 domain 对象的 characteristic 方法，返回 domain 的特征值
        return self.domain.characteristic()
```