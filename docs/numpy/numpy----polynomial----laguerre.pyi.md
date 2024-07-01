# `.\numpy\numpy\polynomial\laguerre.pyi`

```py
# 引入类型提示模块中的Any类型
from typing import Any

# 引入numpy库中的int_类型和NDArray类型
from numpy import int_
from numpy.typing import NDArray

# 引入numpy.polynomial._polybase模块中的ABCPolyBase类
from numpy.polynomial._polybase import ABCPolyBase

# 引入numpy.polynomial.polyutils模块中的trimcoef函数，赋值给lagtrim变量
from numpy.polynomial.polyutils import trimcoef

# __all__列表，用于定义模块中公开的所有符号
__all__: list[str]

# 将trimcoef函数赋值给lagtrim变量
lagtrim = trimcoef

# 定义函数poly2lag，未完整定义，暂时省略

# 定义函数lag2poly，未完整定义，暂时省略

# lagdomain、lagzero、lagone、lagx，均为定义为NDArray[int_]类型的变量

# 定义函数lagline，未完整定义，暂时省略

# 定义函数lagfromroots，未完整定义，暂时省略

# 定义函数lagadd，未完整定义，暂时省略

# 定义函数lagsub，未完整定义，暂时省略

# 定义函数lagmulx，未完整定义，暂时省略

# 定义函数lagmul，未完整定义，暂时省略

# 定义函数lagdiv，未完整定义，暂时省略

# 定义函数lagpow，未完整定义，暂时省略

# 定义函数lagder，未完整定义，暂时省略

# 定义函数lagint，未完整定义，暂时省略

# 定义函数lagval，未完整定义，暂时省略

# 定义函数lagval2d，未完整定义，暂时省略

# 定义函数laggrid2d，未完整定义，暂时省略

# 定义函数lagval3d，未完整定义，暂时省略

# 定义函数laggrid3d，未完整定义，暂时省略

# 定义函数lagvander，未完整定义，暂时省略

# 定义函数lagvander2d，未完整定义，暂时省略

# 定义函数lagvander3d，未完整定义，暂时省略

# 定义函数lagfit，未完整定义，暂时省略

# 定义函数lagcompanion，未完整定义，暂时省略

# 定义函数lagroots，未完整定义，暂时省略

# 定义函数laggauss，未完整定义，暂时省略

# 定义函数lagweight，未完整定义，暂时省略

# Laguerre类，继承自ABCPolyBase类，定义了domain、window、basis_name属性
class Laguerre(ABCPolyBase):
    domain: Any
    window: Any
    basis_name: Any
```