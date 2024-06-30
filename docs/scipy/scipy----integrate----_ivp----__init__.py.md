# `D:\src\scipysrc\scipy\scipy\integrate\_ivp\__init__.py`

```
"""Suite of ODE solvers implemented in Python."""
# 导入的模块提供了一套实现在 Python 中的常微分方程求解器

from .ivp import solve_ivp
# 从当前包中的 ivp 模块导入 solve_ivp 函数，用于求解初始值问题（IVP）

from .rk import RK23, RK45, DOP853
# 从当前包中的 rk 模块导入 RK23、RK45 和 DOP853 类，这些类实现了不同的 Runge-Kutta 方法

from .radau import Radau
# 从当前包中的 radau 模块导入 Radau 类，实现了 Radau 隐式 Runge-Kutta 方法

from .bdf import BDF
# 从当前包中的 bdf 模块导入 BDF 类，实现了基于后向差分公式的方法

from .lsoda import LSODA
# 从当前包中的 lsoda 模块导入 LSODA 类，实现了 Livermore Solver for Ordinary Differential Equations（LSODA）方法

from .common import OdeSolution
# 从当前包中的 common 模块导入 OdeSolution 类，提供了普遍的 ODE 解决方案表示

from .base import DenseOutput, OdeSolver
# 从当前包中的 base 模块导入 DenseOutput 和 OdeSolver 类，提供了稠密输出和常微分方程求解器基类
```