# `D:\src\scipysrc\scipy\scipy\special\_test_internal.pyi`

```
# 导入 NumPy 库，用于数值计算和数组操作
import numpy as np

# 检查是否有 fenv 模块，返回布尔值
def have_fenv() -> bool: ...

# 生成指定大小的随机双精度浮点数数组，返回 NumPy 的 float64 类型
def random_double(size: int) -> np.float64: ...

# 测试加法和舍入的函数，接受大小和模式参数
def test_add_round(size: int, mode: str): ...

# 计算双精度浮点数 xhi 和 xlo 的指数函数，返回一个元组 (高位结果, 低位结果)
def _dd_exp(xhi: float, xlo: float) -> tuple[float, float]: ...

# 计算双精度浮点数 xhi 和 xlo 的对数函数，返回一个元组 (高位结果, 低位结果)
def _dd_log(xhi: float, xlo: float) -> tuple[float, float]: ...

# 计算双精度浮点数 xhi 和 xlo 的 expm1 函数，返回一个元组 (高位结果, 低位结果)
def _dd_expm1(xhi: float, xlo: float) -> tuple[float, float]: ...
```