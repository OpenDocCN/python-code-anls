# `.\numpy\numpy\polynomial\__init__.pyi`

```py
# 导入 numpy._pytesttester 模块中的 PytestTester 类
from numpy._pytesttester import PytestTester

# 从 numpy.polynomial 模块中导入以下多项式对象
# chebyshev 对应 numpy.polynomial.chebyshev 模块的别名 chebyshev
# hermite 对应 numpy.polynomial.hermite 模块的别名 hermite
# hermite_e 对应 numpy.polynomial.hermite_e 模块的别名 hermite_e
# laguerre 对应 numpy.polynomial.laguerre 模块的别名 laguerre
# legendre 对应 numpy.polynomial.legendre 模块的别名 legendre
# polynomial 对应 numpy.polynomial.polynomial 模块的别名 polynomial
from numpy.polynomial import (
    chebyshev as chebyshev,
    hermite as hermite,
    hermite_e as hermite_e,
    laguerre as laguerre,
    legendre as legendre,
    polynomial as polynomial,
)

# 从 numpy.polynomial.chebyshev 模块导入 Chebyshev 类，并将其命名为 Chebyshev
from numpy.polynomial.chebyshev import Chebyshev as Chebyshev
# 从 numpy.polynomial.hermite 模块导入 Hermite 类，并将其命名为 Hermite
from numpy.polynomial.hermite import Hermite as Hermite
# 从 numpy.polynomial.hermite_e 模块导入 HermiteE 类，并将其命名为 HermiteE
from numpy.polynomial.hermite_e import HermiteE as HermiteE
# 从 numpy.polynomial.laguerre 模块导入 Laguerre 类，并将其命名为 Laguerre
from numpy.polynomial.laguerre import Laguerre as Laguerre
# 从 numpy.polynomial.legendre 模块导入 Legendre 类，并将其命名为 Legendre
from numpy.polynomial.legendre import Legendre as Legendre
# 从 numpy.polynomial.polynomial 模块导入 Polynomial 类，并将其命名为 Polynomial
from numpy.polynomial.polynomial import Polynomial as Polynomial

# 定义 __all__ 列表，包含可以从当前模块导入的公共名称
__all__: list[str]
# 创建 PytestTester 类的实例 test
test: PytestTester

# 定义一个名为 set_default_printstyle 的函数，接受一个参数 style，并没有具体实现内容
def set_default_printstyle(style): ...
```