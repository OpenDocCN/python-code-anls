# `D:\src\scipysrc\scipy\scipy\stats\_constants.py`

```
"""
Statistics-related constants.

"""
import numpy as np

# 最小可表示的正数，满足 1.0 + _EPS != 1.0
_EPS = np.finfo(float).eps

# 最大可用的浮点数值（按大小）
_XMAX = np.finfo(float).max

# 最大可用浮点数的自然对数；用于判断 exp(something) 是否会溢出
_LOGXMAX = np.log(_XMAX)

# 最小可用的浮点数值（按大小），非 subnormal 格式的双精度浮点数
_XMIN = np.finfo(float).tiny

# 最小可用浮点数的自然对数，非 subnormal 格式的双精度浮点数
_LOGXMIN = np.log(_XMIN)

# -special.psi(1)，欧拉常数
_EULER = 0.577215664901532860606512090082402431042

# special.zeta(3, 1)，阿泊利常数
_ZETA3 = 1.202056903159594285399738161511449990765

# sqrt(pi)
_SQRT_PI = 1.772453850905516027298167483341145182798

# sqrt(2/pi)
_SQRT_2_OVER_PI = 0.7978845608028654

# log(sqrt(2/pi))
_LOG_SQRT_2_OVER_PI = -0.22579135264472744
```