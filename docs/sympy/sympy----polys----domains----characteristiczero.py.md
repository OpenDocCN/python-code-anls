# `D:\src\scipysrc\sympy\sympy\polys\domains\characteristiczero.py`

```
"""Implementation of :class:`CharacteristicZero` class. """

# 导入 sympy.polys.domains.domain 中的 Domain 类
from sympy.polys.domains.domain import Domain
# 从 sympy.utilities 中导入 public 装饰器
from sympy.utilities import public

# 定义 CharacteristicZero 类，继承自 Domain 类
@public
class CharacteristicZero(Domain):
    """Domain that has infinite number of elements. """

    # 类属性，表示该域的特征为零
    has_CharacteristicZero = True

    # 方法：返回此域的特征
    def characteristic(self):
        """Return the characteristic of this domain. """
        return 0
```