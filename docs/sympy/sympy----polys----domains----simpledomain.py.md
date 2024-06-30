# `D:\src\scipysrc\sympy\sympy\polys\domains\simpledomain.py`

```
"""
Implementation of :class:`SimpleDomain` class.
"""

# 导入 sympy 库中的相关模块和类
from sympy.polys.domains.domain import Domain
from sympy.utilities import public

# 使用 @public 装饰器标记这个类是公共可用的
@public
# 定义 SimpleDomain 类，继承自 Domain 类
class SimpleDomain(Domain):
    """Base class for simple domains, e.g. ZZ, QQ."""

    # 类属性，标识这个类是 SimpleDomain 类型的
    is_Simple = True

    # 定义 inject 方法，注入生成器到当前域中
    def inject(self, *gens):
        """Inject generators into this domain."""
        # 调用 poly_ring 方法，返回多项式环对象
        return self.poly_ring(*gens)
```